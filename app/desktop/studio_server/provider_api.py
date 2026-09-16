import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Literal

import httpx
import litellm
import openai
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from kiln_ai.adapters.docker_model_runner_tools import (
    DockerModelRunnerConnection,
    get_docker_model_runner_connection,
)
from kiln_ai.adapters.ml_embedding_model_list import (
    EmbeddingModelName,
    KilnEmbeddingModel,
    KilnEmbeddingModelProvider,
    built_in_embedding_models,
)
from kiln_ai.adapters.ml_model_list import (
    KilnModel,
    KilnModelProvider,
    ModelProviderName,
    StructuredOutputMode,
    built_in_models,
)
from kiln_ai.adapters.ollama_tools import (
    OllamaConnection,
    ollama_base_url,
    parse_ollama_tags,
)
from kiln_ai.adapters.provider_tools import (
    PLACEHOLDER_API_KEY,
    get_all_user_models,
    get_legacy_custom_models,
    provider_name_from_id,
    provider_warnings,
)
from kiln_ai.adapters.reranker_list import built_in_rerankers
from kiln_ai.adapters.user_model_entry import UserModelEntry
from kiln_ai.datamodel.finetune import Finetune
from kiln_ai.datamodel.registry import all_projects
from kiln_ai.utils.config import Config
from kiln_ai.utils.exhaustive_error import raise_exhaustive_enum_error
from kiln_ai.utils.wandb_utils import AuthenticationError, get_wandb_default_entity
from kiln_server.utils.agent_checks.policy import ALLOW_AGENT, DENY_AGENT
from pydantic import BaseModel, Field

from app.desktop.studio_server.api_client.kiln_ai_server_client.api.auth import (
    create_api_key_v1_create_api_key_post,
)
from app.desktop.studio_server.api_client.kiln_server_client import (
    get_oauth_authenticated_client,
)

logger = logging.getLogger(__name__)


async def connect_ollama(custom_ollama_url: str | None = None) -> OllamaConnection:
    # Tags is a list of Ollama models. Proves Ollama is running, and models are available.
    if (
        custom_ollama_url
        and not custom_ollama_url.startswith("http://")
        and not custom_ollama_url.startswith("https://")
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid Ollama URL. It must start with http:// or https://",
        )

    try:
        base_url = custom_ollama_url or ollama_base_url()
        tags = requests.get(base_url + "/api/tags", timeout=5).json()
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=417,
            detail="Failed to connect. Ensure Ollama app is running.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to Ollama: {e}",
        )

    ollama_connection = parse_ollama_tags(tags)
    if ollama_connection is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to parse Ollama data - unsure which models are installed.",
        )

    # attempt to get the Ollama version
    try:
        version_body = requests.get(base_url + "/api/version", timeout=5).json()
        ollama_connection.version = version_body.get("version", None)
    except Exception:
        pass

    # Save the custom Ollama URL if used to connect
    if custom_ollama_url and custom_ollama_url != Config.shared().ollama_base_url:
        Config.shared().save_setting("ollama_base_url", custom_ollama_url)

    return ollama_connection


async def connect_docker_model_runner(
    docker_model_runner_custom_url: str | None = None,
) -> DockerModelRunnerConnection:
    if (
        docker_model_runner_custom_url
        and not docker_model_runner_custom_url.startswith("http://")
        and not docker_model_runner_custom_url.startswith("https://")
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid Docker Model Runner URL. It must start with http:// or https://",
        )

    try:
        docker_connection = await get_docker_model_runner_connection(
            docker_model_runner_custom_url
        )

        if docker_connection is None:
            raise HTTPException(
                status_code=417,
                detail="Failed to connect. Ensure Docker Model Runner is running and you enabled TCP connections. See the Docker Model Runner docs for instructions.",
            )

    except HTTPException:
        # Preserve status/details from earlier raises (e.g., 400/417)
        raise
    except (openai.APIError, httpx.RequestError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to Docker Model Runner: {e}",
        )

    # Save the custom Docker URL if used to connect
    if (
        docker_model_runner_custom_url
        and docker_model_runner_custom_url
        != Config.shared().docker_model_runner_base_url
    ):
        Config.shared().save_setting(
            "docker_model_runner_base_url", docker_model_runner_custom_url
        )

    return docker_connection


class ModelDetails(BaseModel):
    id: str
    name: str
    supports_structured_output: bool
    supports_data_gen: bool
    suggested_for_data_gen: bool
    supports_logprobs: bool
    suggested_for_evals: bool
    suggested_for_synthetic_user: bool
    supports_function_calling: bool
    uncensored: bool
    suggested_for_uncensored_data_gen: bool
    supports_vision: bool
    supports_doc_extraction: bool
    suggested_for_doc_extraction: bool
    multimodal_capable: bool = Field(default=False)
    multimodal_mime_types: List[str] | None = Field(default=None)
    # the suggested structured output mode for this model.
    structured_output_mode: StructuredOutputMode
    available_thinking_levels: dict[str, str] | None = Field(default=None)
    default_thinking_level: str | None = Field(default=None)
    # True if this is a untested model (typically user added). We don't know if these support structured output, data gen, etc. They should appear in their own section in the UI.
    untested_model: bool = Field(default=False)
    task_filter: List[str] | None = Field(default=None)
    # if the model has a model-specific run config which should be used when running the model (like a fine-tune model's baked in run config)
    model_specific_run_config: str | None = Field(default=None)
    deprecated: bool = Field(default=False)


class AvailableModels(BaseModel):
    provider_name: str
    provider_id: str
    models: List[ModelDetails]


class EmbeddingModelDetails(BaseModel):
    id: str
    name: str
    n_dimensions: int
    max_input_tokens: int | None
    supports_custom_dimensions: bool
    suggested_for_chunk_embedding: bool
    deprecated: bool = Field(default=False)


class EmbeddingProvider(BaseModel):
    provider_name: str
    provider_id: str
    models: List[EmbeddingModelDetails]


class RerankerModelDetails(BaseModel):
    id: str
    name: str
    deprecated: bool = Field(default=False)


class RerankerProvider(BaseModel):
    provider_name: str
    provider_id: str
    models: List[RerankerModelDetails]


class ProviderModel(BaseModel):
    id: str
    name: str


class ProviderModels(BaseModel):
    models: Dict[str, ProviderModel]


class ProviderEmbeddingModels(BaseModel):
    models: Dict[EmbeddingModelName, ProviderModel]


class ProviderRerankerModels(BaseModel):
    models: Dict[str, ProviderModel]


class AvailableProviderInfo(BaseModel):
    """Information about an available AI provider."""

    id: str = Field(description="The unique provider identifier used in API calls.")
    name: str = Field(description="The human-readable display name of the provider.")
    provider_type: Literal["builtin", "custom"] = Field(
        description="Whether the provider is built-in or user-configured."
    )


class OpenAICompatibleProviderConfig(BaseModel):
    """Configuration for an OpenAI compatible provider."""

    name: str = Field(description="Name for the OpenAI compatible provider.")
    base_url: str = Field(description="Base URL for the OpenAI compatible API.")
    api_key: str = Field(description="API key for authentication.")


class CreateKilnCopilotApiKeyRequest(BaseModel):
    access_token: str = Field(description="Kinde OAuth access token.")


def connect_provider_api(app: FastAPI):
    @app.get(
        "/api/providers/models",
        summary="List Provider Models",
        tags=["Providers & Models"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_providers_models() -> ProviderModels:
        models = {}
        for model in built_in_models:
            models[model.name] = ProviderModel(id=model.name, name=model.friendly_name)
        return ProviderModels(models=models)

    # returns map, of provider name to list of model names
    @app.get(
        "/api/available_models",
        summary="List Available Models",
        tags=["Providers & Models"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_available_models() -> List[AvailableModels]:
        # Providers with just keys can return all their models if keys are set
        key_providers: List[str] = []

        for provider, provider_warning in provider_warnings.items():
            has_keys = True
            for required_key in provider_warning.required_config_keys:
                if Config.shared().get_value(required_key) is None:
                    has_keys = False
                    break
            if has_keys:
                key_providers.append(provider)
        models: List[AvailableModels] = [
            AvailableModels(
                provider_name=provider_name_from_id(provider),
                provider_id=provider,
                models=[],
            )
            for provider in key_providers
        ]

        for model in built_in_models:
            for provider in model.providers:
                if not provider.model_id:
                    # it's possible for models to not have an ID (fine-tune only model)
                    continue
                if provider.name in key_providers:
                    available_models = next(
                        (m for m in models if m.provider_id == provider.name), None
                    )
                    if available_models:
                        # no need to couple the frontend to our backend mimetypes enum
                        mime_types_as_str = (
                            [
                                str(mime_type)
                                for mime_type in provider.multimodal_mime_types
                            ]
                            if provider.multimodal_mime_types
                            else None
                        )
                        available_models.models.append(
                            ModelDetails(
                                id=model.name,
                                name=model.friendly_name,
                                supports_structured_output=provider.supports_structured_output,
                                supports_data_gen=provider.supports_data_gen,
                                suggested_for_data_gen=provider.suggested_for_data_gen,
                                supports_logprobs=provider.supports_logprobs,
                                supports_function_calling=provider.supports_function_calling,
                                suggested_for_evals=provider.suggested_for_evals,
                                suggested_for_synthetic_user=provider.suggested_for_synthetic_user,
                                uncensored=provider.uncensored,
                                suggested_for_uncensored_data_gen=provider.suggested_for_uncensored_data_gen,
                                structured_output_mode=provider.structured_output_mode,
                                supports_vision=provider.supports_vision,
                                supports_doc_extraction=provider.supports_doc_extraction,
                                suggested_for_doc_extraction=provider.suggested_for_doc_extraction,
                                multimodal_capable=provider.multimodal_capable,
                                multimodal_mime_types=mime_types_as_str,
                                available_thinking_levels=provider.available_thinking_levels,
                                default_thinking_level=provider.default_thinking_level,
                                deprecated=provider.deprecated,
                            )
                        )

        # Docker Model Runner is special: check which models are installed
        docker_models = await available_docker_model_runner_models()
        if docker_models:
            models.insert(0, docker_models)

        # Ollama is special: check which models are installed
        ollama_models = await available_ollama_models()
        if ollama_models:
            models.insert(0, ollama_models)

        # Add any fine tuned models
        fine_tuned_models = all_fine_tuned_models()
        if fine_tuned_models:
            models.append(fine_tuned_models)

        # Add any openai compatible providers
        openai_compatible = openai_compatible_providers()

        # Collect all custom models to merge into provider lists.
        # Legacy custom_models and user_model_registry are handled separately
        # but merged the same way (by provider key).
        models_to_merge: Dict[str, List[ModelDetails]] = {}

        # Legacy custom_models: keyed by kiln_custom_registry
        for key, model_list in legacy_custom_models_as_available().items():
            models_to_merge.setdefault(key, []).extend(model_list)

        # New user_model_registry: keyed by provider_id (builtin) or provider name (custom)
        for key, model_list in user_models_as_available().items():
            models_to_merge.setdefault(key, []).extend(model_list)

        # Merge custom models into their respective provider lists
        if models_to_merge:
            for provider in models:
                provider_id_str = str(provider.provider_id)
                if provider_id_str in models_to_merge:
                    provider.models.extend(models_to_merge.pop(provider_id_str))

            # Merge into openai compatible providers (by provider_name)
            openai_compatible = [
                AvailableModels(
                    provider_name=provider.provider_name,
                    provider_id=provider.provider_id,
                    models=provider.models
                    + models_to_merge.pop(provider.provider_name, []),
                )
                for provider in openai_compatible
            ]
            # Any remaining entries are custom providers with only user models
            for provider_name, model_list in models_to_merge.items():
                openai_compatible.append(
                    AvailableModels(
                        provider_name=provider_name,
                        provider_id=ModelProviderName.openai_compatible,
                        models=model_list,
                    )
                )

        models.extend(openai_compatible)

        return models

    @app.get(
        "/api/providers/embedding_models",
        summary="List Provider Embedding Models",
        tags=["Providers & Models"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_providers_embedding_models() -> ProviderEmbeddingModels:
        models = {}
        for model in built_in_embedding_models:
            models[model.name] = ProviderModel(id=model.name, name=model.friendly_name)
        return ProviderEmbeddingModels(models=models)

    # returns map, of provider name to list of model names
    @app.get(
        "/api/available_embedding_models",
        summary="List Available Embedding Models",
        tags=["Providers & Models"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_available_embedding_models() -> List[EmbeddingProvider]:
        # Providers with just keys can return all their models if keys are set
        key_providers: List[str] = []

        for provider, provider_warning in provider_warnings.items():
            has_keys = True
            for required_key in provider_warning.required_config_keys:
                if Config.shared().get_value(required_key) is None:
                    has_keys = False
                    break
            if has_keys:
                key_providers.append(provider)

        models: List[EmbeddingProvider] = [
            EmbeddingProvider(
                provider_name=provider_name_from_id(provider),
                provider_id=provider,
                models=[],
            )
            for provider in key_providers
        ]

        for model in built_in_embedding_models:
            for provider in model.providers:
                if provider.name in key_providers:
                    available_models = next(
                        (m for m in models if m.provider_id == provider.name), None
                    )
                    if available_models:
                        available_models.models.append(
                            EmbeddingModelDetails(
                                id=model.name,
                                name=model.friendly_name,
                                n_dimensions=provider.n_dimensions,
                                max_input_tokens=provider.max_input_tokens,
                                supports_custom_dimensions=provider.supports_custom_dimensions,
                                suggested_for_chunk_embedding=provider.suggested_for_chunk_embedding,
                                deprecated=provider.deprecated,
                            )
                        )

        # Ollama is special: check which models are installed
        ollama_models = await available_ollama_embedding_models()
        if ollama_models:
            models.insert(0, ollama_models)

        return models

    @app.get(
        "/api/providers/reranker_models",
        summary="List Provider Reranker Models",
        tags=["Providers & Models"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_providers_reranker_models() -> ProviderRerankerModels:
        models = {}
        for model in built_in_rerankers:
            models[model.name] = ProviderModel(id=model.name, name=model.friendly_name)
        return ProviderRerankerModels(models=models)

    # returns map, of provider name to list of model names
    @app.get(
        "/api/available_reranker_models",
        summary="List Available Reranker Models",
        tags=["Providers & Models"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_available_reranker_models() -> List[RerankerProvider]:
        # Providers with just keys can return all their models if keys are set
        key_providers: List[str] = []

        for provider, provider_warning in provider_warnings.items():
            has_keys = True
            for required_key in provider_warning.required_config_keys:
                if Config.shared().get_value(required_key) is None:
                    has_keys = False
                    break
            if has_keys:
                key_providers.append(provider)

        models: List[RerankerProvider] = [
            RerankerProvider(
                provider_name=provider_name_from_id(provider),
                provider_id=provider,
                models=[],
            )
            for provider in key_providers
        ]

        for model in built_in_rerankers:
            for provider in model.providers:
                if provider.name in key_providers:
                    available_models = next(
                        (m for m in models if m.provider_id == provider.name), None
                    )
                    if available_models:
                        available_models.models.append(
                            RerankerModelDetails(
                                id=model.name,
                                name=model.friendly_name,
                                deprecated=provider.deprecated,
                            )
                        )

        # ollama not supported yet - we can add it later

        return models

    @app.post(
        "/api/provider/ollama/connect",
        summary="Connect Ollama",
        tags=["Providers & Models"],
        openapi_extra=DENY_AGENT,
    )
    async def connect_ollama_api(
        custom_ollama_url: Annotated[
            str | None, Query(description="Custom URL for the Ollama server.")
        ] = None,
    ) -> OllamaConnection:
        return await connect_ollama(custom_ollama_url)

    @app.post(
        "/api/provider/docker_model_runner/connect",
        summary="Connect Docker Model Runner",
        tags=["Providers & Models"],
        openapi_extra=DENY_AGENT,
    )
    async def connect_docker_model_runner_api(
        docker_model_runner_custom_url: Annotated[
            str | None, Query(description="Custom URL for the Docker Model Runner.")
        ] = None,
    ) -> DockerModelRunnerConnection:
        chosen_url = docker_model_runner_custom_url
        return await connect_docker_model_runner(chosen_url)

    @app.post(
        "/api/provider/openai_compatible",
        summary="Add OpenAI Compatible Provider",
        tags=["Providers & Models"],
        openapi_extra=DENY_AGENT,
    )
    async def save_openai_compatible_providers(
        config: OpenAICompatibleProviderConfig,
    ):
        providers = Config.shared().openai_compatible_providers or []
        existing_provider = next(
            (p for p in providers if p["name"] == config.name), None
        )
        if existing_provider:
            raise HTTPException(
                status_code=400,
                detail="Provider with this name already exists",
            )
        providers.append(
            {
                "name": config.name,
                "base_url": config.base_url,
                "api_key": config.api_key,
            }
        )
        Config.shared().openai_compatible_providers = providers
        return JSONResponse(
            status_code=200,
            content={"message": "OpenAI compatible provider saved"},
        )

    @app.delete(
        "/api/provider/openai_compatible",
        summary="Delete OpenAI Compatible Provider",
        tags=["Providers & Models"],
        openapi_extra=DENY_AGENT,
    )
    async def delete_openai_compatible_providers(
        name: Annotated[
            str, Query(description="Name of the OpenAI compatible provider to delete.")
        ],
    ):
        if not name:
            return JSONResponse(
                status_code=400,
                content={"message": "Name is required"},
            )
        providers = Config.shared().openai_compatible_providers or []
        providers = [p for p in providers if p["name"] != name]
        Config.shared().openai_compatible_providers = providers
        return JSONResponse(
            status_code=200,
            content={"message": "OpenAI compatible provider deleted"},
        )

    @app.get(
        "/api/settings/available_providers",
        summary="List Available Providers",
        tags=["Providers & Models"],
        openapi_extra=DENY_AGENT,
    )
    async def get_available_providers() -> List[AvailableProviderInfo]:
        """Returns all providers that can have custom models added."""
        providers = []

        # Built-in providers with required API keys set
        for provider, warning in provider_warnings.items():
            has_keys = all(
                Config.shared().get_value(key) is not None
                for key in warning.required_config_keys
            )
            if has_keys:
                providers.append(
                    AvailableProviderInfo(
                        id=str(provider.value),
                        name=provider_name_from_id(provider),
                        provider_type="builtin",
                    )
                )

        # Custom OpenAI-compatible providers
        openai_compat_providers = Config.shared().openai_compatible_providers or []
        for provider in openai_compat_providers:
            if provider.get("name"):
                providers.append(
                    AvailableProviderInfo(
                        id=provider["name"],
                        name=provider["name"],
                        provider_type="custom",
                    )
                )

        return providers

    @app.get(
        "/api/settings/user_models",
        summary="List User Models",
        tags=["Providers & Models"],
        openapi_extra=DENY_AGENT,
    )
    async def get_user_models() -> List[UserModelEntry]:
        """Returns all user-defined models (new registry + legacy combined).

        Includes both user_model_registry entries and legacy custom_models
        (converted to UserModelEntry for display). Legacy models can be
        deleted via the tuple method (provider_type/provider_id/model_id).
        """
        result = list(get_all_user_models())

        # Include legacy custom_models for display on the management page
        for provider_id, model_id in get_legacy_custom_models():
            if not any(
                m.provider_type == "builtin"
                and m.provider_id == provider_id
                and m.model_id == model_id
                for m in result
            ):
                result.append(
                    UserModelEntry(
                        provider_type="builtin",
                        provider_id=provider_id,
                        model_id=model_id,
                    )
                )

        return result

    @app.post(
        "/api/settings/user_models",
        summary="Add User Model",
        tags=["Providers & Models"],
        openapi_extra=DENY_AGENT,
    )
    async def add_user_model(entry: UserModelEntry) -> JSONResponse:
        """Add a user-defined model to the registry."""

        # Validate provider exists
        if entry.provider_type == "builtin":
            if entry.provider_id not in ModelProviderName.__members__:
                raise HTTPException(
                    status_code=400, detail=f"Invalid provider: {entry.provider_id}"
                )
            # Check if provider is configured
            provider_warning = provider_warnings.get(
                ModelProviderName(entry.provider_id)
            )
            if provider_warning:
                for key in provider_warning.required_config_keys:
                    if Config.shared().get_value(key) is None:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Provider {entry.provider_id} is not configured. Please set up API keys first.",
                        )
        else:
            providers = Config.shared().openai_compatible_providers or []
            if not any(p.get("name") == entry.provider_id for p in providers):
                raise HTTPException(
                    status_code=400,
                    detail=f"Custom provider not found: {entry.provider_id}",
                )

        # Add to registry
        registry = Config.shared().user_model_registry or []

        # Check for exact duplicate (all identifying fields must match)
        if any(
            e.get("provider_id") == entry.provider_id
            and e.get("model_id") == entry.model_id
            and e.get("name") == entry.name
            and e.get("overrides") == (entry.overrides if entry.overrides else None)
            for e in registry
        ):
            raise HTTPException(status_code=400, detail="Model already exists")

        registry.append(entry.model_dump(exclude_none=True))
        Config.shared().user_model_registry = registry

        return JSONResponse(status_code=200, content={"message": "Model added"})

    @app.delete(
        "/api/settings/user_models",
        summary="Delete User Model",
        tags=["Providers & Models"],
        openapi_extra=DENY_AGENT,
    )
    async def delete_user_model(
        provider_type: Annotated[
            str | None,
            Query(
                description="Provider type: 'builtin' for built-in providers, 'custom' for OpenAI-compatible."
            ),
        ] = None,
        provider_id: Annotated[
            str | None,
            Query(
                description="The provider identifier (e.g., 'openai', 'anthropic', or custom provider name)."
            ),
        ] = None,
        model_id: Annotated[
            str | None, Query(description="The model identifier to delete.")
        ] = None,
        id: Annotated[
            str | None,
            Query(
                description="Unique ID of the model entry in user_model_registry (preferred deletion method)."
            ),
        ] = None,
    ) -> JSONResponse:
        """Delete a user-defined model from the registry.

        Supports two deletion methods:
        1. By ID (new): Pass `id` parameter to delete from user_model_registry
        2. By tuple (legacy): Pass provider_type, provider_id, model_id to delete from
           user_model_registry by matching those fields, or from legacy custom_models

        Legacy models in custom_models don't have IDs and must use the tuple method.
        """

        # Method 1: Delete by ID (new format)
        if id is not None:
            registry = Config.shared().user_model_registry or []
            original_len = len(registry)
            registry = [e for e in registry if e.get("id") != id]

            if len(registry) == original_len:
                raise HTTPException(status_code=404, detail="Model not found")

            Config.shared().user_model_registry = registry
            return JSONResponse(status_code=200, content={"message": "Model deleted"})

        # Method 2: Delete by tuple (legacy format, also supports user_model_registry)
        if provider_type and provider_id and model_id:
            registry = Config.shared().user_model_registry or []
            original_len = len(registry)

            # Try to delete from user_model_registry first
            registry = [
                e
                for e in registry
                if not (
                    e.get("provider_type") == provider_type
                    and e.get("provider_id") == provider_id
                    and e.get("model_id") == model_id
                )
            ]

            if len(registry) != original_len:
                Config.shared().user_model_registry = registry
                return JSONResponse(
                    status_code=200, content={"message": "Model deleted"}
                )

            # If not found in registry, try legacy custom_models
            legacy = Config.shared().custom_models or []
            legacy_id = f"{provider_id}::{model_id}"
            if legacy_id in legacy:
                legacy.remove(legacy_id)
                Config.shared().custom_models = legacy
                return JSONResponse(
                    status_code=200, content={"message": "Model deleted"}
                )

            raise HTTPException(status_code=404, detail="Model not found")

        raise HTTPException(
            status_code=400,
            detail="Must specify either 'id' or provider_type+provider_id+model_id",
        )

    def parse_api_key(key_data: dict) -> str:
        return parse_api_field(key_data, "API Key")

    def parse_api_field(key_data: dict, field_name: str) -> str:
        api_key = key_data.get(field_name)
        if not api_key or not isinstance(api_key, str):
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} not found",
            )
        return api_key

    @app.post(
        "/api/provider/connect_api_key",
        summary="Connect Provider API Key",
        tags=["Providers & Models"],
        openapi_extra=DENY_AGENT,
    )
    async def connect_api_key(payload: dict):
        provider = payload.get("provider")
        key_data = payload.get("key_data")
        if not isinstance(key_data, dict) or not isinstance(provider, str):
            return JSONResponse(
                status_code=400,
                content={"message": "Invalid key_data or provider"},
            )

        # Wandb is not a typical AI provider, but it's a provider you can connect through this UI/API
        if provider == "wandb":
            # Load optional base URL and entity
            base_url = None
            if "Base URL - Optional" in key_data:
                base_url = parse_url(key_data, "Base URL - Optional")
            custom_entity = None
            if "Custom Entity - Optional" in key_data:
                custom_entity = key_data["Custom Entity - Optional"].strip()
            return await connect_wandb(
                parse_api_key(key_data),
                custom_entity,
                base_url,
            )

        # Kiln Copilot is not a model provider but it's a provider you can connect through this UI/API
        if provider == "kiln_copilot":
            return await connect_kiln_copilot(parse_api_key(key_data))

        if provider not in ModelProviderName.__members__:
            return JSONResponse(
                status_code=400,
                content={"message": f"Provider {provider} not supported"},
            )

        typed_provider = ModelProviderName(provider)

        match typed_provider:
            case ModelProviderName.openai:
                return await connect_openai(parse_api_key(key_data))
            case ModelProviderName.groq:
                return await connect_groq(parse_api_key(key_data))
            case ModelProviderName.openrouter:
                return await connect_openrouter(parse_api_key(key_data))
            case ModelProviderName.fireworks_ai:
                return await connect_fireworks(key_data)
            case ModelProviderName.amazon_bedrock:
                return await connect_bedrock(key_data)
            case ModelProviderName.anthropic:
                return await connect_anthropic(parse_api_key(key_data))
            case ModelProviderName.gemini_api:
                return await connect_gemini(parse_api_key(key_data))
            case ModelProviderName.azure_openai:
                endpoint = parse_url(key_data, "Endpoint URL")
                return await connect_azure_openai(parse_api_key(key_data), endpoint)
            case ModelProviderName.huggingface:
                return await connect_huggingface(parse_api_key(key_data))
            case ModelProviderName.vertex:
                return await connect_vertex(
                    parse_api_field(key_data, "Project ID"),
                    parse_api_field(key_data, "Project Location"),
                )
            case ModelProviderName.together_ai:
                return await connect_together(parse_api_key(key_data))
            case ModelProviderName.siliconflow_cn:
                return await connect_siliconflow(parse_api_key(key_data))
            case ModelProviderName.cerebras:
                return await connect_cerebras(parse_api_key(key_data))
            case ModelProviderName.featherless_ai:
                return await connect_featherless(parse_api_key(key_data))
            case (
                ModelProviderName.kiln_custom_registry
                | ModelProviderName.kiln_fine_tune
                | ModelProviderName.openai_compatible
                | ModelProviderName.ollama
                | ModelProviderName.docker_model_runner
            ):
                return JSONResponse(
                    status_code=400,
                    content={"message": "Provider not supported for API keys"},
                )
            case _:
                raise_exhaustive_enum_error(typed_provider)

    @app.post(
        "/api/provider/disconnect_api_key",
        summary="Disconnect Provider API Key",
        tags=["Providers & Models"],
        openapi_extra=DENY_AGENT,
    )
    async def disconnect_api_key(
        provider_id: Annotated[
            str, Query(description="The provider identifier to disconnect.")
        ],
    ) -> JSONResponse:
        if provider_id == "wandb":
            # Wandb is not an AI provider, but it's a provider you can connect, supported by this UI/API
            Config.shared().wandb_api_key = None
            Config.shared().wandb_entity = None
            Config.shared().wandb_base_url = None
        elif provider_id == "kiln_copilot":
            # Kiln Copilot is not a model provider but it's a provider you can connect through this UI/API
            Config.shared().kiln_copilot_api_key = None
        else:
            if provider_id not in ModelProviderName.__members__:
                return JSONResponse(
                    status_code=400,
                    content={"message": f"Invalid provider: {provider_id}"},
                )

            typed_provider_id = ModelProviderName(provider_id)

            match typed_provider_id:
                case ModelProviderName.openai:
                    Config.shared().open_ai_api_key = None
                case ModelProviderName.groq:
                    Config.shared().groq_api_key = None
                case ModelProviderName.openrouter:
                    Config.shared().open_router_api_key = None
                case ModelProviderName.fireworks_ai:
                    Config.shared().fireworks_api_key = None
                    Config.shared().fireworks_account_id = None
                case ModelProviderName.amazon_bedrock:
                    Config.shared().bedrock_access_key = None
                    Config.shared().bedrock_secret_key = None
                case ModelProviderName.anthropic:
                    Config.shared().anthropic_api_key = None
                case ModelProviderName.gemini_api:
                    Config.shared().gemini_api_key = None
                case ModelProviderName.azure_openai:
                    Config.shared().azure_openai_api_key = None
                    Config.shared().azure_openai_endpoint = None
                case ModelProviderName.huggingface:
                    Config.shared().huggingface_api_key = None
                case ModelProviderName.vertex:
                    Config.shared().vertex_project_id = None
                    Config.shared().vertex_location = None
                case ModelProviderName.together_ai:
                    Config.shared().together_api_key = None
                case ModelProviderName.siliconflow_cn:
                    Config.shared().siliconflow_cn_api_key = None
                case ModelProviderName.cerebras:
                    Config.shared().cerebras_api_key = None
                case ModelProviderName.featherless_ai:
                    Config.shared().featherless_ai_api_key = None
                case (
                    ModelProviderName.kiln_custom_registry
                    | ModelProviderName.kiln_fine_tune
                    | ModelProviderName.openai_compatible
                    | ModelProviderName.ollama
                    | ModelProviderName.docker_model_runner
                ):
                    return JSONResponse(
                        status_code=400,
                        content={"message": "Provider not supported"},
                    )
                case _:
                    # Raises a pyright error if I miss a case
                    raise_exhaustive_enum_error(typed_provider_id)

        return JSONResponse(
            status_code=200,
            content={"message": "Provider disconnected"},
        )

    @app.post(
        "/api/provider/create_kiln_copilot_api_key",
        summary="Create Kiln Copilot API Key",
        tags=["Providers & Models"],
        openapi_extra=DENY_AGENT,
    )
    async def create_kiln_copilot_api_key(
        payload: CreateKilnCopilotApiKeyRequest,
    ) -> JSONResponse:
        return await _create_kiln_copilot_api_key(payload.access_token)

    @app.get(
        "/api/provider/verify_kiln_copilot_api_key",
        summary="Verify Kiln Copilot API Key",
        tags=["Providers & Models"],
        openapi_extra=ALLOW_AGENT,
    )
    async def verify_kiln_copilot_api_key() -> JSONResponse:
        """Verify the stored Kiln Copilot API key against the Kiln server.

        Returns `{is_valid: bool}`. A stale key the server rejects with
        401/403 is cleared from local config so subsequent flows fall back to
        the connect screen instead of silently using a dead key. Network
        failures leave the key in place and report `false` for this check
        only — they shouldn't punish the user for a transient blip.
        """
        key = Config.shared().kiln_copilot_api_key
        if not key:
            return JSONResponse(status_code=200, content={"is_valid": False})

        base_url = os.environ.get("KILN_SERVER_BASE_URL", "https://api.kiln.tech")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/v1/verify_api_key",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10,
                )
        except httpx.RequestError:
            return JSONResponse(status_code=200, content={"is_valid": False})

        if response.status_code == 200:
            return JSONResponse(status_code=200, content={"is_valid": True})

        if response.status_code in (401, 403):
            Config.shared().kiln_copilot_api_key = None

        return JSONResponse(status_code=200, content={"is_valid": False})


async def _create_kiln_copilot_api_key(access_token: str) -> JSONResponse:
    if not access_token:
        return JSONResponse(
            status_code=400,
            content={"message": "Access token is required"},
        )

    client = get_oauth_authenticated_client(access_token)
    try:
        response = await create_api_key_v1_create_api_key_post.asyncio_detailed(
            client=client,
        )
    except httpx.RequestError as e:
        return JSONResponse(
            status_code=502,
            content={"message": f"Failed to connect to Kiln server. Error: {e!s}"},
        )

    if response.parsed is not None:
        Config.shared().kiln_copilot_api_key = response.parsed.api_key
        return JSONResponse(
            status_code=200,
            content={"message": "Connected to Kiln Copilot"},
        )

    try:
        error_body = json.loads(response.content)
        error_message = error_body.get(
            "detail", f"Failed to create API key (HTTP {response.status_code.value})"
        )
    except json.JSONDecodeError:
        error_message = f"Failed to create API key (HTTP {response.status_code.value})"

    return JSONResponse(
        status_code=response.status_code.value,
        content={"message": error_message},
    )


async def connect_openrouter(key: str):
    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        # invalid body, but we just want to see if the key is valid
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={},
        )

        # 401 def means invalid API key
        if response.status_code == 401:
            return JSONResponse(
                status_code=401,
                content={
                    "message": "Failed to connect to OpenRouter. Invalid API key."
                },
            )
        else:
            # No 401 means key is valid (even it it's an error, which we expect with empty body)
            Config.shared().open_router_api_key = key

            return JSONResponse(
                status_code=200,
                content={"message": "Connected to OpenRouter"},
            )
            # Any non-200 status code is an error
    except Exception as e:
        # unexpected error
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to connect to OpenRouter. Error: {e!s}"},
        )


async def connect_siliconflow(key: str):
    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        response = requests.get(
            "https://api.siliconflow.cn/v1/models",
            headers=headers,
        )

        if response.status_code == 401:
            return JSONResponse(
                status_code=401,
                content={
                    "message": "Failed to connect to SiliconFlow. Invalid API key."
                },
            )
        elif response.status_code == 200:
            Config.shared().siliconflow_cn_api_key = key
            return JSONResponse(
                status_code=200,
                content={"message": "Connected to SiliconFlow"},
            )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Failed to connect to SiliconFlow. Error: [{response.status_code}] {response.text}"
                },
            )
    except Exception as e:
        # unexpected error
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to connect to SiliconFlow. Error: {e!s}"},
        )


async def connect_fireworks(key_data: dict):
    try:
        key = key_data.get("API Key")
        account_id = key_data.get("Account ID")
        if (
            not account_id
            or not isinstance(account_id, str)
            or not key
            or not isinstance(key, str)
        ):
            raise HTTPException(
                status_code=400,
                detail="Account ID or API Key not found",
            )

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        # list the shared models (fireworks account)
        response = requests.get(
            f"https://api.fireworks.ai/v1/accounts/{account_id}/models",
            headers=headers,
        )

        if response.status_code == 403:
            return JSONResponse(
                status_code=401,
                content={
                    "message": "Failed to connect to Fireworks. Invalid API key or Account ID."
                },
            )
        elif response.status_code == 200:
            Config.shared().fireworks_api_key = key
            Config.shared().fireworks_account_id = account_id

            return JSONResponse(
                status_code=200,
                content={"message": "Connected to Fireworks"},
            )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Failed to connect to Fireworks. Error: [{response.status_code}] {response.text}"
                },
            )
    except Exception as e:
        # unexpected error
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to connect to Fireworks. Error: {e!s}"},
        )


async def connect_openai(key: str):
    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        response = requests.get("https://api.openai.com/v1/models", headers=headers)

        # 401 def means invalid API key, so special case it
        if response.status_code == 401:
            return JSONResponse(
                status_code=401,
                content={"message": "Failed to connect to OpenAI. Invalid API key."},
            )

        # Any non-200 status code is an error
        response.raise_for_status()
        # If the request is successful, the function will continue
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to connect to OpenAI. Error: {e!s}"},
        )

    # It worked! Save the key and return success
    Config.shared().open_ai_api_key = key

    return JSONResponse(
        status_code=200,
        content={"message": "Connected to OpenAI"},
    )


async def connect_groq(key: str):
    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        response = requests.get(
            "https://api.groq.com/openai/v1/models", headers=headers
        )

        if "invalid_api_key" in response.text:
            return JSONResponse(
                status_code=401,
                content={"message": "Failed to connect to Groq. Invalid API key."},
            )

        # Any non-200 status code is an error
        response.raise_for_status()
        # If the request is successful, the function will continue
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to connect to Groq. Error: {e!s}"},
        )

    # It worked! Save the key and return success
    Config.shared().groq_api_key = key

    return JSONResponse(
        status_code=200,
        content={"message": "Connected to Groq"},
    )


async def connect_gemini(key: str):
    try:
        response = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
        )

        if "API_KEY_INVALID" in response.text:
            return JSONResponse(
                status_code=401,
                content={"message": "Failed to connect to Gemini. Invalid API key."},
            )
        elif response.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Failed to connect to Gemini. Error: [{response.status_code}]"
                },
            )
        else:
            Config.shared().gemini_api_key = key
            return JSONResponse(
                status_code=200,
                content={"message": "Connected to Gemini"},
            )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to connect to Gemini. Error: {e!s}"},
        )


async def connect_vertex(project_id: str, project_location: str):
    try:
        await litellm.acompletion(
            model="vertex_ai/gemini-3.5-flash",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            vertex_project=project_id,
            vertex_location=project_location,
        )

        Config.shared().vertex_project_id = project_id
        Config.shared().vertex_location = project_location

        return JSONResponse(
            status_code=200,
            content={"message": "Connected to Vertex"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to connect to Vertex. Error: {e!s}"},
        )


async def connect_together(key: str):
    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        response = requests.get(
            "https://api.together.xyz/v1/models",
            headers=headers,
        )

        if response.status_code == 401:
            return JSONResponse(
                status_code=401,
                content={
                    "message": "Failed to connect to Together.ai. Invalid API key."
                },
            )
        else:
            # Any non-401 status code is okay - auth passed. We expect other errors, but we don't care.
            Config.shared().together_api_key = key
            return JSONResponse(
                status_code=200,
                content={"message": "Connected to Together.ai"},
            )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to connect to Together.ai. Error: {e!s}"},
        )


async def connect_huggingface(key: str):
    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        response = requests.get(
            "https://huggingface.co/api/organizations/fake_org_for_auth_test/resource-groups",
            headers=headers,
        )

        if response.status_code == 401:
            return JSONResponse(
                status_code=401,
                content={
                    "message": "Failed to connect to Huggingface. Invalid API key."
                },
            )
        else:
            # Any non-401 status code is okay - auth passed. We expect other errors, but we don't care.
            Config.shared().huggingface_api_key = key
            return JSONResponse(
                status_code=200,
                content={"message": "Connected to Huggingface"},
            )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to connect to Huggingface. Error: {e!s}"},
        )


async def connect_anthropic(key: str):
    try:
        headers = {
            "x-api-key": key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        response = requests.get("https://api.anthropic.com/v1/models", headers=headers)

        if response.status_code == 401:
            return JSONResponse(
                status_code=401,
                content={"message": "Failed to connect to Anthropic. Invalid API key."},
            )
        elif response.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Failed to connect to Anthropic. Error: [{response.status_code}]"
                },
            )
        else:
            Config.shared().anthropic_api_key = key
            return JSONResponse(
                status_code=200,
                content={"message": "Connected to Anthropic"},
            )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to connect to Anthropic. Error: {e!s}"},
        )


async def connect_wandb(
    key: str, custom_entity: str | None, base_url: str | None
) -> JSONResponse:
    try:
        # This both checks the API key is valid, and gets the default entity for the user
        wandb_entity = await get_wandb_default_entity(key, base_url)
        if isinstance(wandb_entity, AuthenticationError):
            return JSONResponse(
                status_code=401,
                content={"message": "Failed to connect to W&B. Invalid API key."},
            )

        # Get their entity: custom if provided, default if not
        entity = custom_entity or wandb_entity
        if entity is None:
            return JSONResponse(
                status_code=400,
                content={
                    "message": "Failed to connect to W&B: No default entity found. You must either provide a custom entity name or set a default entity in your W&B account settings."
                },
            )

        # Save the credentials if valid
        Config.shared().wandb_api_key = key
        Config.shared().wandb_entity = entity
        Config.shared().wandb_base_url = base_url

        return JSONResponse(
            status_code=200,
            content={"message": "Connected to W&B"},
        )

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to connect to W&B. Error: {e!s}"},
        )


async def connect_azure_openai(key: str, endpoint: str):
    try:
        headers = {
            "api-key": key,
            "Content-Type": "application/json",
        }
        response = requests.get(
            f"{endpoint}/openai/files?api-version=2024-08-01-preview", headers=headers
        )

        if response.status_code == 401:
            return JSONResponse(
                status_code=401,
                content={
                    "message": "Failed to connect to Azure OpenAI. Invalid API key."
                },
            )
        elif response.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Failed to connect to Azure OpenAI. Error: [{response.status_code}]"
                },
            )
        else:
            Config.shared().azure_openai_api_key = key
            Config.shared().azure_openai_endpoint = endpoint
            return JSONResponse(
                status_code=200,
                content={"message": "Connected to Azure OpenAI"},
            )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to connect to Azure OpenAI. Error: {e!s}"},
        )


async def connect_cerebras(key: str):
    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        response = requests.get("https://api.cerebras.ai/v1/models", headers=headers)

        if response.status_code == 401:
            return JSONResponse(
                status_code=401,
                content={"message": "Failed to connect to Cerebras. Invalid API key."},
            )
        elif response.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Failed to connect to Cerebras. Error: [{response.status_code}]"
                },
            )
        else:
            Config.shared().cerebras_api_key = key
            return JSONResponse(
                status_code=200,
                content={"message": "Connected to Cerebras"},
            )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to connect to Cerebras. Error: {e!s}"},
        )


async def connect_featherless(key: str):
    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        # Featherless has no authenticated GET endpoint we can ping: /v1/models is
        # public (returns 200 without a key), so it can't validate anything.
        #
        # Instead we POST to chat/completions with a model slug that intentionally
        # doesn't exist. Featherless checks auth *before* resolving the model, so a
        # bad key returns 401 while a good key falls through to a model error. That
        # validates the key without spending tokens, and without depending on any
        # real model slug staying available.
        response = requests.post(
            "https://api.featherless.ai/v1/chat/completions",
            headers=headers,
            json={
                "model": "kiln-ai/__connection_test__",
                "messages": [{"role": "user", "content": "."}],
                "max_tokens": 1,
            },
        )

        if response.status_code == 401:
            return JSONResponse(
                status_code=401,
                content={
                    "message": "Failed to connect to Featherless AI. Invalid API key."
                },
            )
        elif response.status_code >= 500:
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Failed to connect to Featherless AI. Error: [{response.status_code}]"
                },
            )
        else:
            # Any non-auth response means the key was accepted.
            Config.shared().featherless_ai_api_key = key
            return JSONResponse(
                status_code=200,
                content={"message": "Connected to Featherless AI"},
            )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to connect to Featherless AI. Error: {e!s}"},
        )


async def connect_bedrock(key_data: dict):
    access_key = key_data.get("Access Key")
    secret_key = key_data.get("Secret Key")
    if (
        not access_key
        or not isinstance(access_key, str)
        or not secret_key
        or not isinstance(secret_key, str)
    ):
        raise HTTPException(
            status_code=400,
            detail="Access Key or Secret Key not found",
        )
    try:
        # Test credentials request, but invalid model so we don't use tokens
        await litellm.acompletion(
            model="bedrock/ai21.jamba-1-5-mini-v9999.8888",
            aws_region_name="us-west-2",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )
    except Exception as e:
        # Improve error message if it's a confirmed authentication error
        if isinstance(e, litellm.exceptions.AuthenticationError):
            return JSONResponse(
                status_code=401,
                content={
                    "message": "Failed to connect to Bedrock. Invalid credentials."
                },
            )
        # If it's a bad request, it's a valid key (but the model is fake)
        if isinstance(e, litellm.exceptions.BadRequestError):
            Config.shared().bedrock_access_key = access_key
            Config.shared().bedrock_secret_key = secret_key
            return JSONResponse(
                status_code=200,
                content={"message": "Connected to Bedrock"},
            )
        # Unknown error, raise it
        raise e

    return JSONResponse(
        status_code=400,
        content={"message": "Unknown Bedrock Error"},
    )


async def connect_kiln_copilot(key: str):
    base_url = os.environ.get("KILN_SERVER_BASE_URL", "https://api.kiln.tech")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/v1/verify_api_key",
                headers={"Authorization": f"Bearer {key}"},
                timeout=20,
            )
    except httpx.RequestError as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to connect to Kiln Copilot. Error: {e!s}"},
        )

    if response.status_code == 200:
        Config.shared().kiln_copilot_api_key = key
        return JSONResponse(
            status_code=200,
            content={"message": "Connected to Kiln Copilot"},
        )
    else:
        try:
            error_content = response.json()
        except Exception:
            error_content = {
                "message": f"Failed to verify API key (HTTP {response.status_code})"
            }

        return JSONResponse(
            status_code=response.status_code,
            content=error_content,
        )


async def available_ollama_models() -> AvailableModels | None:
    # Try to connect to Ollama, and get the list of installed models
    try:
        ollama_connection = await connect_ollama()
        ollama_models = AvailableModels(
            provider_name=provider_name_from_id(ModelProviderName.ollama),
            provider_id=ModelProviderName.ollama,
            models=[],
        )

        for ollama_model_tag in ollama_connection.supported_models:
            # Get all Kiln models that match the Ollama model tag
            # There may be multiple (Qwen 3 has thinking and non-thinking versions)
            models = models_from_ollama_tag(ollama_model_tag)
            for model, ollama_provider in models:
                if model and ollama_provider:
                    ollama_models.models.append(
                        ModelDetails(
                            id=model.name,
                            name=model.friendly_name,
                            supports_structured_output=ollama_provider.supports_structured_output,
                            supports_data_gen=ollama_provider.supports_data_gen,
                            supports_logprobs=False,  # Ollama doesn't support logprobs https://github.com/ollama/ollama/issues/2415
                            suggested_for_data_gen=ollama_provider.suggested_for_data_gen,
                            suggested_for_evals=ollama_provider.suggested_for_evals,
                            suggested_for_synthetic_user=ollama_provider.suggested_for_synthetic_user,
                            supports_function_calling=ollama_provider.supports_function_calling,
                            uncensored=False,
                            suggested_for_uncensored_data_gen=False,
                            # Ollama has constrained decode and all models support json_schema. Use it!
                            structured_output_mode=StructuredOutputMode.json_schema,
                            supports_vision=ollama_provider.supports_vision,
                            supports_doc_extraction=ollama_provider.supports_doc_extraction,
                            suggested_for_doc_extraction=ollama_provider.suggested_for_doc_extraction,
                            multimodal_capable=ollama_provider.multimodal_capable,
                            multimodal_mime_types=[
                                str(mime_type)
                                for mime_type in ollama_provider.multimodal_mime_types
                            ]
                            if ollama_provider.multimodal_mime_types
                            else None,
                            deprecated=ollama_provider.deprecated,
                        )
                    )
        for ollama_model in ollama_connection.untested_models:
            ollama_models.models.append(
                ModelDetails(
                    id=ollama_model,
                    name=ollama_model,
                    supports_structured_output=False,
                    supports_data_gen=False,
                    supports_logprobs=False,
                    supports_function_calling=False,
                    untested_model=True,
                    suggested_for_data_gen=False,
                    suggested_for_evals=False,
                    suggested_for_synthetic_user=False,
                    uncensored=False,
                    suggested_for_uncensored_data_gen=False,
                    # Ollama has constrained decode and all models support json_schema. Use it!
                    structured_output_mode=StructuredOutputMode.json_schema,
                    supports_vision=False,
                    supports_doc_extraction=False,
                    suggested_for_doc_extraction=False,
                    multimodal_capable=False,
                    multimodal_mime_types=None,
                    deprecated=False,
                )
            )

        if len(ollama_models.models) > 0:
            return ollama_models

        return None
    except HTTPException:
        # skip ollama if it's not available
        return None


async def available_ollama_embedding_models() -> EmbeddingProvider | None:
    # Try to connect to Ollama, and get the list of installed models
    try:
        ollama_connection = await connect_ollama()
        ollama_embedding_models = EmbeddingProvider(
            provider_name=provider_name_from_id(ModelProviderName.ollama),
            provider_id=ModelProviderName.ollama,
            models=[],
        )

        for ollama_model_tag in ollama_connection.supported_embedding_models:
            models = embedding_models_from_ollama_tag(ollama_model_tag)
            for model, ollama_provider in models:
                if model and ollama_provider:
                    ollama_embedding_models.models.append(
                        EmbeddingModelDetails(
                            id=model.name,
                            name=model.friendly_name,
                            n_dimensions=ollama_provider.n_dimensions,
                            max_input_tokens=ollama_provider.max_input_tokens,
                            supports_custom_dimensions=ollama_provider.supports_custom_dimensions,
                            suggested_for_chunk_embedding=ollama_provider.suggested_for_chunk_embedding,
                            deprecated=ollama_provider.deprecated,
                        )
                    )

        if len(ollama_embedding_models.models) > 0:
            return ollama_embedding_models

        return None
    except HTTPException:
        # skip ollama if it's not available
        return None


async def available_docker_model_runner_models() -> AvailableModels | None:
    # Try to connect to Docker Model Runner, and get the list of installed models
    try:
        docker_connection = await get_docker_model_runner_connection()
        if docker_connection is None:
            return None

        docker_models = AvailableModels(
            provider_name=provider_name_from_id(ModelProviderName.docker_model_runner),
            provider_id=ModelProviderName.docker_model_runner,
            models=[],
        )

        for docker_model_id in docker_connection.supported_models:
            # Get all Kiln models that match the Docker Model Runner model ID
            models = models_from_docker_model_runner_id(docker_model_id)
            for model, docker_provider in models:
                if model and docker_provider:
                    docker_models.models.append(
                        ModelDetails(
                            id=model.name,
                            name=model.friendly_name,
                            supports_structured_output=docker_provider.supports_structured_output,
                            supports_data_gen=docker_provider.supports_data_gen,
                            supports_logprobs=docker_provider.supports_logprobs,
                            suggested_for_data_gen=docker_provider.suggested_for_data_gen,
                            suggested_for_evals=docker_provider.suggested_for_evals,
                            suggested_for_synthetic_user=docker_provider.suggested_for_synthetic_user,
                            uncensored=docker_provider.uncensored,
                            suggested_for_uncensored_data_gen=docker_provider.suggested_for_uncensored_data_gen,
                            supports_vision=docker_provider.supports_vision,
                            supports_doc_extraction=docker_provider.supports_doc_extraction,
                            suggested_for_doc_extraction=docker_provider.suggested_for_doc_extraction,
                            # Docker Model Runner uses OpenAI-compatible API with JSON schema support
                            structured_output_mode=StructuredOutputMode.json_schema,
                            supports_function_calling=docker_provider.supports_function_calling,
                            deprecated=docker_provider.deprecated,
                        )
                    )
        for docker_model in docker_connection.untested_models:
            docker_models.models.append(
                ModelDetails(
                    id=docker_model,
                    name=docker_model,
                    supports_structured_output=True,
                    supports_data_gen=False,
                    supports_logprobs=False,
                    untested_model=True,
                    suggested_for_data_gen=False,
                    suggested_for_evals=False,
                    suggested_for_synthetic_user=False,
                    uncensored=False,
                    suggested_for_uncensored_data_gen=False,
                    supports_vision=False,
                    supports_doc_extraction=False,
                    suggested_for_doc_extraction=False,
                    # Docker Model Runner uses OpenAI-compatible API with JSON schema support
                    structured_output_mode=StructuredOutputMode.json_schema,
                    supports_function_calling=False,
                    deprecated=False,
                )
            )

        if len(docker_models.models) > 0:
            return docker_models

        return None
    except (HTTPException, openai.APIError, httpx.RequestError):
        # skip docker model runner if it's not available
        return None


def models_from_docker_model_runner_id(
    model_id: str,
) -> List[tuple[KilnModel | None, KilnModelProvider | None]]:
    models: list[tuple[KilnModel | None, KilnModelProvider | None]] = []
    for model in built_in_models:
        docker_provider = next(
            (
                p
                for p in model.providers
                if p.name == ModelProviderName.docker_model_runner
            ),
            None,
        )
        if not docker_provider:
            continue

        if model_id == docker_provider.model_id:
            models.append((model, docker_provider))

    return models


def models_from_ollama_tag(
    tag: str,
) -> List[tuple[KilnModel | None, KilnModelProvider | None]]:
    models: list[tuple[KilnModel | None, KilnModelProvider | None]] = []
    for model in built_in_models:
        ollama_provider = next(
            (p for p in model.providers if p.name == ModelProviderName.ollama), None
        )
        if not ollama_provider:
            continue

        model_name = ollama_provider.model_id
        if tag in [model_name, f"{model_name}:latest"]:
            models.append((model, ollama_provider))
        if ollama_provider.ollama_model_aliases is not None:
            # all aliases (and :latest)
            for alias in ollama_provider.ollama_model_aliases:
                if tag in [alias, f"{alias}:latest"]:
                    models.append((model, ollama_provider))

    return models


def embedding_models_from_ollama_tag(
    tag: str,
) -> List[tuple[KilnEmbeddingModel | None, KilnEmbeddingModelProvider | None]]:
    models: list[
        tuple[KilnEmbeddingModel | None, KilnEmbeddingModelProvider | None]
    ] = []
    for model in built_in_embedding_models:
        ollama_provider = next(
            (p for p in model.providers if p.name == ModelProviderName.ollama), None
        )
        if not ollama_provider:
            continue

        model_name = ollama_provider.model_id
        if tag in [model_name, f"{model_name}:latest"]:
            models.append((model, ollama_provider))
        if ollama_provider.ollama_model_aliases is not None:
            # all aliases (and :latest)
            for alias in ollama_provider.ollama_model_aliases:
                if tag in [alias, f"{alias}:latest"]:
                    models.append((model, ollama_provider))

    return models


def legacy_custom_models_as_available() -> Dict[str, List[ModelDetails]]:
    """
    Returns legacy custom_models keyed by "kiln_custom_registry" for merging
    into available_models.

    Legacy custom_models are stored as "provider::model_id" strings in config.
    They keep that full string as their model ID (for DB history compatibility).
    They appear under the "Custom Models" provider group.

    New custom models should use user_model_registry instead.
    """
    legacy_models = get_legacy_custom_models()
    if not legacy_models:
        return {}

    models: List[ModelDetails] = []
    for provider_id, model_name in legacy_models:
        full_model_id = f"{provider_id}::{model_name}"
        models.append(
            ModelDetails(
                id=full_model_id,
                name=f"{model_name} (Custom)",
                supports_structured_output=False,
                supports_data_gen=False,
                supports_logprobs=False,
                supports_function_calling=False,
                untested_model=True,
                suggested_for_data_gen=False,
                suggested_for_evals=False,
                suggested_for_synthetic_user=False,
                uncensored=False,
                suggested_for_uncensored_data_gen=False,
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_vision=False,
                supports_doc_extraction=False,
                suggested_for_doc_extraction=False,
                multimodal_capable=False,
                multimodal_mime_types=None,
                deprecated=False,
            )
        )
    return {"kiln_custom_registry": models}


def user_models_as_available() -> Dict[str, List[ModelDetails]]:
    """
    Returns user_model_registry entries keyed by provider_id for merging into
    available_models.

    These are the new-format custom models (not legacy custom_models).
    Legacy custom_models are handled separately by legacy_custom_models_as_available().

    Returns:
        - Dict keyed by provider_id (enum value for builtin, provider name for custom)
        - Each value is a list of ModelDetails with " (Custom)" suffix on names
    """
    user_models = get_all_user_models()
    if not user_models:
        return {}

    # Group by provider
    by_provider: Dict[str, List[ModelDetails]] = {}

    for entry in user_models:
        # Determine the key for this model
        if entry.provider_type == "builtin":
            key = entry.provider_id  # e.g., "openai"
        else:
            key = entry.provider_id  # e.g., "ABC"

        if key not in by_provider:
            by_provider[key] = []

        # Build model ID for selection using the entry's unique ID
        full_model_id = f"user_model::{entry.id}"

        # Get display name with " (Custom)" suffix
        display_name = (entry.name or entry.model_id) + " (Custom)"

        # Determine capabilities from overrides
        overrides = entry.overrides or {}

        # Convert string structured_output_mode to enum if present
        structured_output_mode_value = overrides.get(
            "structured_output_mode", StructuredOutputMode.json_instructions
        )
        if isinstance(structured_output_mode_value, str):
            try:
                structured_output_mode_value = StructuredOutputMode(
                    structured_output_mode_value
                )
            except ValueError:
                structured_output_mode_value = StructuredOutputMode.json_instructions

        by_provider[key].append(
            ModelDetails(
                id=full_model_id,
                name=display_name,
                supports_structured_output=overrides.get(
                    "supports_structured_output", False
                ),
                supports_data_gen=overrides.get("supports_data_gen", False),
                supports_logprobs=overrides.get("supports_logprobs", False),
                supports_function_calling=overrides.get(
                    "supports_function_calling", False
                ),
                untested_model=True,
                suggested_for_data_gen=False,
                suggested_for_evals=False,
                suggested_for_synthetic_user=False,
                uncensored=overrides.get("uncensored", False),
                suggested_for_uncensored_data_gen=False,
                structured_output_mode=structured_output_mode_value,
                supports_vision=overrides.get("supports_vision", False),
                supports_doc_extraction=overrides.get("supports_doc_extraction", False),
                suggested_for_doc_extraction=False,
                multimodal_capable=overrides.get("multimodal_capable", False),
                multimodal_mime_types=overrides.get("multimodal_mime_types"),
                deprecated=overrides.get("deprecated", False),
            )
        )

    return by_provider


def fine_tune_model_structured_output_mode(
    fine_tune: Finetune,
) -> StructuredOutputMode:
    # Current field
    if fine_tune.run_config and fine_tune.run_config.structured_output_mode is not None:
        return fine_tune.run_config.structured_output_mode
    # Legacy field
    legacy_structured_output_mode = fine_tune.structured_output_mode
    if legacy_structured_output_mode is not None and isinstance(
        legacy_structured_output_mode, StructuredOutputMode
    ):
        return legacy_structured_output_mode
    # Fallback
    return StructuredOutputMode.json_instructions


def all_fine_tuned_models() -> AvailableModels | None:
    # Add any fine tuned models
    models: List[ModelDetails] = []

    for project in all_projects():
        for task in project.tasks():
            for fine_tune in task.finetunes():
                # check if the fine tune is completed
                if fine_tune.fine_tune_model_id:
                    model_specific_run_config = (
                        f"finetune_run_config::{project.id}::{task.id}::{fine_tune.id}"
                        if fine_tune.run_config is not None
                        else None
                    )
                    models.append(
                        ModelDetails(
                            id=fine_tune.nested_id(),
                            name=fine_tune.name
                            + f" ({provider_name_from_id(fine_tune.provider)})",
                            # YMMV, but we'll assume all fine tuned models support structured output, data gen, and tools as they may have been trained with them
                            supports_structured_output=True,
                            supports_function_calling=True,
                            supports_data_gen=True,
                            supports_logprobs=False,
                            task_filter=[str(task.id)],
                            suggested_for_data_gen=False,
                            suggested_for_evals=False,
                            suggested_for_synthetic_user=False,
                            uncensored=False,
                            suggested_for_uncensored_data_gen=False,
                            structured_output_mode=fine_tune_model_structured_output_mode(
                                fine_tune
                            ),
                            model_specific_run_config=model_specific_run_config,
                            supports_vision=False,
                            supports_doc_extraction=False,
                            suggested_for_doc_extraction=False,
                            multimodal_capable=False,
                            multimodal_mime_types=None,
                            deprecated=False,
                        )
                    )

    if len(models) > 0:
        return AvailableModels(
            provider_name="Fine Tuned Models",
            provider_id=ModelProviderName.kiln_fine_tune,
            models=models,
        )
    return None


@dataclass
class OpenAICompatibleProviderCache:
    providers: List[AvailableModels]
    last_updated: datetime | None = None
    openai_compat_config_when_cached: Any | None = None
    had_error: bool = False

    # Cache for 60 minutes, or if the config changes
    def is_stale(self) -> bool:
        if self.last_updated is None:
            return True

        if self.had_error:
            return True

        if datetime.now().astimezone() - self.last_updated > timedelta(minutes=60):
            return True

        current_providers = Config.shared().openai_compatible_providers
        if current_providers != self.openai_compat_config_when_cached:
            return True

        return False


_openai_compatible_providers_cache: OpenAICompatibleProviderCache | None = None


def openai_compatible_providers() -> List[AvailableModels]:
    global _openai_compatible_providers_cache

    if (
        _openai_compatible_providers_cache is None
        or _openai_compatible_providers_cache.is_stale()
    ):
        # Load values and cache them
        cache = openai_compatible_providers_load_cache()
        _openai_compatible_providers_cache = cache

    if _openai_compatible_providers_cache is None:
        return []

    return _openai_compatible_providers_cache.providers


def openai_compatible_providers_load_cache() -> OpenAICompatibleProviderCache | None:
    provider_config = Config.shared().openai_compatible_providers
    if not provider_config or len(provider_config) == 0:
        return None

    # Errors that can be retried, like network issues, are tracked in cache.
    # We retry populating the cache on each call
    has_error = False

    openai_compatible_models: List[AvailableModels] = []
    for provider in provider_config:
        models: List[ModelDetails] = []
        base_url = provider.get("base_url")
        if not base_url or not base_url.startswith("http"):
            logger.warning(
                "No base URL for OpenAI compatible provider %s - %s", provider, base_url
            )
            continue
        name = provider.get("name")
        if not name:
            logger.warning("No name for OpenAI compatible provider %s", provider)
            continue

        # API key optional - some providers like Ollama don't use it, but the OpenAI client errors without one
        api_key = provider.get("api_key") or PLACEHOLDER_API_KEY

        try:
            openai_client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                # Important: max_retries must be 0 for performance.
                # It's common for these servers to be down sometimes (could be local app that isn't running)
                # OpenAI client will retry a few times, with a sleep in between! Big loading perf hit.
                max_retries=0,
            )
            provider_models = openai_client.models.list()
            for model in provider_models:
                models.append(
                    ModelDetails(
                        id=f"{name}::{model.id}",
                        name=model.id,
                        supports_structured_output=False,
                        supports_data_gen=False,
                        supports_logprobs=False,
                        supports_function_calling=False,
                        untested_model=True,
                        suggested_for_data_gen=False,
                        suggested_for_evals=False,
                        suggested_for_synthetic_user=False,
                        uncensored=False,
                        suggested_for_uncensored_data_gen=False,
                        # OpenAI compatible models could be anything. JSON instructions is the only safe bet that works everywhere.
                        structured_output_mode=StructuredOutputMode.json_instructions,
                        supports_vision=False,
                        supports_doc_extraction=False,
                        suggested_for_doc_extraction=False,
                        multimodal_capable=False,
                        multimodal_mime_types=None,
                        deprecated=False,
                    )
                )

            openai_compatible_models.append(
                AvailableModels(
                    provider_id=ModelProviderName.openai_compatible,
                    provider_name=name,
                    models=models,
                )
            )
        except Exception:
            logger.error(
                "Error loading models from OpenAI compatible provider %s",
                name,
                exc_info=True,
            )
            has_error = True
            continue

    cache = OpenAICompatibleProviderCache(
        providers=openai_compatible_models,
        last_updated=datetime.now().astimezone(),
        openai_compat_config_when_cached=provider_config,
        had_error=has_error,
    )

    return cache


def parse_url(key_data: dict, field_name: str) -> str:
    url = key_data.get(field_name)
    if not url or not isinstance(url, str):
        raise HTTPException(
            status_code=400,
            detail="Endpoint URL not found",
        )
    if url.endswith("/"):
        # remove last slash
        url = url[:-1]
    if not url.startswith("http"):
        raise HTTPException(
            status_code=400,
            detail="Endpoint URL must start with http or https",
        )
    return url
