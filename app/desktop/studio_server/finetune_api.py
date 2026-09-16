import logging
from enum import Enum
from typing import Annotated, Dict

import httpx
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import StreamingResponse
from kiln_ai.adapters.adapter_registry import (
    load_skills_for_task,
    load_skills_from_tool_ids,
)
from kiln_ai.adapters.fine_tune.base_finetune import FineTuneParameter, FineTuneStatus
from kiln_ai.adapters.fine_tune.dataset_formatter import DatasetFormat, DatasetFormatter
from kiln_ai.adapters.fine_tune.finetune_registry import finetune_registry
from kiln_ai.adapters.fine_tune.fireworks_finetune import (
    FIREWORKS_SUPPORTED_FINETUNE_MODELS,
)
from kiln_ai.adapters.ml_model_list import (
    KilnModel,
    KilnModelProvider,
    ModelParserID,
    ModelProviderName,
    built_in_models,
)
from kiln_ai.adapters.prompt_builders import (
    chain_of_thought_prompt,
    prompt_builder_from_id,
)
from kiln_ai.adapters.provider_tools import provider_enabled, provider_name_from_id
from kiln_ai.datamodel import DatasetSplit, Finetune, FineTuneStatusType, Task
from kiln_ai.datamodel.datamodel_enums import (
    THINKING_DATA_STRATEGIES,
    ChatStrategy,
    TurnMode,
)
from kiln_ai.datamodel.dataset_filters import (
    DatasetFilterId,
    HighRatingDatasetFilter,
    ThinkingModelDatasetFilter,
)
from kiln_ai.datamodel.dataset_split import (
    AllSplitDefinition,
    Train60Test20Val20SplitDefinition,
    Train80Test10Val10SplitDefinition,
    Train80Test20SplitDefinition,
    Train80Val20SplitDefinition,
)
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties
from kiln_ai.datamodel.skill import Skill
from kiln_ai.utils.config import Config
from kiln_ai.utils.name_generator import generate_memorable_name
from kiln_server.task_api import task_from_id
from kiln_server.utils.agent_checks.policy import (
    ALLOW_AGENT,
    DENY_AGENT,
    agent_policy_require_approval,
)
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class FinetuneProviderModel(BaseModel):
    """A model available for fine-tuning from a provider."""

    name: str = Field(description="The display name of the model.")
    id: str = Field(description="The provider's model identifier.")
    data_strategies_supported: list[ChatStrategy] = Field(
        default_factory=lambda: [
            ChatStrategy.single_turn,
            ChatStrategy.two_message_cot,
        ],
        description="The data strategies supported by this model.",
    )
    supports_function_calling: bool = Field(
        default=True, description="Whether the model supports function calling."
    )


class FinetuneProvider(BaseModel):
    """A provider that offers fine-tuning."""

    name: str = Field(description="The display name of the provider.")
    id: str = Field(description="The unique provider identifier used in API calls.")
    enabled: bool = Field(description="Whether this provider is enabled.")
    models: list[FinetuneProviderModel] = Field(
        description="The models available for fine-tuning."
    )


class FinetuneDatasetTagInfo(BaseModel):
    """Statistics about task runs with a specific tag."""

    tag: str = Field(description="The tag applied to matching runs.")
    count: int = Field(description="Total number of runs with this tag.")
    reasoning_count: int = Field(description="Number of runs with reasoning data.")
    high_quality_count: int = Field(description="Number of high-quality runs.")
    reasoning_and_high_quality_count: int = Field(
        description="Number of high-quality runs with reasoning data."
    )


class FinetuneDatasetInfo(BaseModel):
    """Information about datasets and finetunes for a task."""

    existing_datasets: list[DatasetSplit] = Field(
        description="Existing dataset splits."
    )
    existing_finetunes: list[Finetune] = Field(description="Existing finetunes.")
    finetune_tags: list[FinetuneDatasetTagInfo] = Field(
        description="Tag statistics for finetune data."
    )

    eligible_datasets: list[DatasetSplit] = Field(
        description="Eligible dataset splits based on tool selection."
    )
    eligible_finetune_tags: list[FinetuneDatasetTagInfo] = Field(
        description="Eligible tag statistics based on tool selection."
    )


class DatasetSplitType(Enum):
    """Dataset split types used in the API. Any split type can be created in code."""

    TRAIN_VAL = "train_val"
    TRAIN_TEST = "train_test"
    TRAIN_TEST_VAL = "train_test_val"
    TRAIN_TEST_VAL_80 = "train_test_val_80"
    ALL = "all"


api_split_types = {
    DatasetSplitType.TRAIN_TEST: Train80Test20SplitDefinition,
    DatasetSplitType.TRAIN_VAL: Train80Val20SplitDefinition,
    DatasetSplitType.TRAIN_TEST_VAL: Train60Test20Val20SplitDefinition,
    DatasetSplitType.TRAIN_TEST_VAL_80: Train80Test10Val10SplitDefinition,
    DatasetSplitType.ALL: AllSplitDefinition,
}


class CreateDatasetSplitRequest(BaseModel):
    """Request to create a dataset split."""

    dataset_split_type: DatasetSplitType = Field(
        description="The type of split to create."
    )
    filter_id: DatasetFilterId = Field(description="The dataset filter to use.")
    name: str | None = Field(default=None, description="The name of the dataset split.")
    description: str | None = Field(
        default=None, description="The description of the dataset split."
    )


class CreateFinetuneRequest(BaseModel):
    """Request to create a finetune"""

    name: str | None = None
    description: str | None = None
    dataset_id: str
    train_split_name: str
    validation_split_name: str | None = None
    parameters: dict[str, str | int | float | bool]
    provider: str
    base_model_id: str
    system_message_generator: str | None = None
    custom_system_message: str | None = None
    custom_thinking_instructions: str | None = None
    data_strategy: ChatStrategy
    run_config_properties: KilnAgentRunConfigProperties | None = None

    @model_validator(mode="after")
    def validate_data_strategy(self):
        if self.data_strategy not in infer_data_strategies_for_model(
            built_in_models, self.base_model_id, self.provider
        ):
            raise ValueError(
                f"The data strategy {self.data_strategy} is not supported for the provider model {self.base_model_id}"
            )
        return self


class FinetuneWithStatus(BaseModel):
    """A finetune with its current status."""

    finetune: Finetune = Field(description="The finetune configuration and metadata.")
    status: FineTuneStatus = Field(description="The current status of the finetune.")


class UpdateFinetuneRequest(BaseModel):
    """Request to update a finetune."""

    name: str = Field(description="The updated name.")
    description: str | None = Field(
        default=None, description="The updated description."
    )


def finetune_from_id(project_id: str, task_id: str, finetune_id: str) -> Finetune:
    task = task_from_id(project_id, task_id)
    finetune = Finetune.from_id_and_parent_path(finetune_id, task.path)
    if finetune is None:
        raise HTTPException(
            status_code=404,
            detail=f"Finetune with ID '{finetune_id}' not found",
        )
    return finetune


def compute_finetune_tag_info(
    task: Task, tool_filter: list[str] | None = None
) -> list[FinetuneDatasetTagInfo]:
    """
    Compute finetune tag counts with quality filters, optionally filtering by tools.

    Args:
        task: The task to analyze
        tool_filter: Optional list of tool IDs to filter by. Only runs with exactly these tools will be included.

    Returns:
        List of FinetuneDatasetTagInfo with counts for each fine_tune tag
    """
    finetune_tag_counts: Dict[str, int] = {}
    reasoning_count: Dict[str, int] = {}
    high_quality_count: Dict[str, int] = {}
    reasoning_and_high_quality_count: Dict[str, int] = {}

    # None means no filter; [] means explicitly match runs with no tools/skills.
    required_tools_set = None if tool_filter is None else set(tool_filter)

    for sample in task.runs(readonly=True):
        # filter by tools if provided
        if required_tools_set is not None:
            if sample.output is None:
                # A few places in code guards against this though the field isn't optional. Just be defensive.
                continue

            run_tools_set: set[str] = set()
            if (
                sample.output.source
                and sample.output.source.run_config
                and isinstance(
                    sample.output.source.run_config, KilnAgentRunConfigProperties
                )
                and sample.output.source.run_config.tools_config
            ):
                run_tools_set = set(sample.output.source.run_config.tools_config.tools)

            if run_tools_set != required_tools_set:
                continue

        for tag in sample.tags:
            if tag.startswith("fine_tune"):
                finetune_tag_counts[tag] = finetune_tag_counts.get(tag, 0) + 1
                is_reasoning = ThinkingModelDatasetFilter(sample)
                is_high_quality = HighRatingDatasetFilter(sample)
                if is_reasoning:
                    reasoning_count[tag] = reasoning_count.get(tag, 0) + 1
                if is_high_quality:
                    high_quality_count[tag] = high_quality_count.get(tag, 0) + 1
                if is_reasoning and is_high_quality:
                    reasoning_and_high_quality_count[tag] = (
                        reasoning_and_high_quality_count.get(tag, 0) + 1
                    )

    return [
        FinetuneDatasetTagInfo(
            tag=tag,
            count=count,
            reasoning_count=reasoning_count.get(tag, 0),
            high_quality_count=high_quality_count.get(tag, 0),
            reasoning_and_high_quality_count=reasoning_and_high_quality_count.get(
                tag, 0
            ),
        )
        for tag, count in finetune_tag_counts.items()
    ]


def connect_fine_tune_api(app: FastAPI):
    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/dataset_splits",
        summary="List Dataset Splits",
        tags=["Fine-tuning"],
        openapi_extra=ALLOW_AGENT,
    )
    async def dataset_splits(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
    ) -> list[DatasetSplit]:
        task = task_from_id(project_id, task_id)
        return task.dataset_splits()

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/finetunes",
        summary="List Finetunes",
        tags=["Fine-tuning"],
        openapi_extra=ALLOW_AGENT,
    )
    async def finetunes(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        update_status: Annotated[
            bool, Query(description="Whether to update the status of each finetune.")
        ] = False,
    ) -> list[Finetune]:
        task = task_from_id(project_id, task_id)
        finetunes = task.finetunes()

        # Update the status of each finetune
        if update_status:
            for finetune in finetunes:
                # Skip "final" status states, as they are not updated
                if finetune.latest_status not in [
                    FineTuneStatusType.completed,
                    FineTuneStatusType.failed,
                ]:
                    try:
                        provider_name = ModelProviderName[finetune.provider]
                    except (KeyError, ValueError):
                        continue
                    if provider_name not in finetune_registry:
                        continue
                    # fetching status updates the datamodel
                    ft_adapter = finetune_registry[provider_name](finetune)
                    await ft_adapter.status()

        return finetunes

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/finetunes/{finetune_id}",
        summary="Get Finetune",
        tags=["Fine-tuning"],
        openapi_extra=ALLOW_AGENT,
    )
    async def finetune(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        finetune_id: Annotated[
            str, Path(description="The unique identifier of the finetune.")
        ],
    ) -> FinetuneWithStatus:
        finetune = finetune_from_id(project_id, task_id, finetune_id)
        try:
            provider_name = ModelProviderName[finetune.provider]
        except (KeyError, ValueError):
            status = FineTuneStatus(
                status=FineTuneStatusType.unknown,
                message=f"Provider '{finetune.provider}' is not available for fine-tuning. Status cannot be refreshed.",
            )
            return FinetuneWithStatus(finetune=finetune, status=status)
        if provider_name not in finetune_registry:
            status = FineTuneStatus(
                status=FineTuneStatusType.unknown,
                message=f"Provider '{finetune.provider}' is not available for fine-tuning. Status cannot be refreshed.",
            )
            return FinetuneWithStatus(finetune=finetune, status=status)
        finetune_adapter = finetune_registry[provider_name]
        status = await finetune_adapter(finetune).status()
        return FinetuneWithStatus(finetune=finetune, status=status)

    @app.patch(
        "/api/projects/{project_id}/tasks/{task_id}/finetunes/{finetune_id}",
        summary="Update Finetune",
        tags=["Fine-tuning"],
        openapi_extra=agent_policy_require_approval(
            "Allow agent to edit fine-tune? Ensure you backup your project before allowing agentic edits."
        ),
    )
    async def update_finetune(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        finetune_id: Annotated[
            str, Path(description="The unique identifier of the finetune.")
        ],
        request: UpdateFinetuneRequest,
    ) -> Finetune:
        finetune = finetune_from_id(project_id, task_id, finetune_id)
        finetune.name = request.name
        finetune.description = request.description
        finetune.save_to_file()
        return finetune

    @app.get(
        "/api/finetune_providers",
        summary="List Finetune Providers",
        tags=["Fine-tuning"],
        openapi_extra=ALLOW_AGENT,
    )
    async def finetune_providers() -> list[FinetuneProvider]:
        provider_models: dict[ModelProviderName, list[FinetuneProviderModel]] = {}

        # Collect models by provider
        for model in built_in_models:
            for provider in model.providers:
                # Skip Fireworks models, as they are added separately
                if provider.name == ModelProviderName.fireworks_ai:
                    continue

                if provider.provider_finetune_id:
                    if provider.name not in provider_models:
                        provider_models[provider.name] = []

                    # special case, Together AI doesn't support fine-tuning with tools
                    supports_tools = (
                        provider.supports_function_calling
                        and provider.name != ModelProviderName.together_ai
                    )
                    provider_models[provider.name].append(
                        FinetuneProviderModel(
                            name=model.friendly_name,
                            id=provider.provider_finetune_id,
                            supports_function_calling=supports_tools,
                        )
                    )

        # Add models from Fireworks
        try:
            fireworks_models = await fetch_fireworks_finetune_models()
            provider_models[ModelProviderName.fireworks_ai] = fireworks_models
        except Exception as e:
            logger.error(f"Error fetching Fireworks models: {e}")

        # Create provider entries
        providers: list[FinetuneProvider] = []
        for provider_name, models in provider_models.items():
            # attach the compatible data strategies to each model
            for model in models:
                model.data_strategies_supported = infer_data_strategies_for_model(
                    built_in_models, model.id, provider_name
                )

            provider = FinetuneProvider(
                name=provider_name_from_id(provider_name),
                id=provider_name,
                enabled=await provider_enabled(provider_name),
                models=models,
            )
            providers.append(provider)

        return providers

    @app.get(
        "/api/finetune/hyperparameters/{provider_id}",
        summary="List Finetune Hyperparameters",
        tags=["Fine-tuning"],
        openapi_extra=ALLOW_AGENT,
    )
    async def finetune_hyperparameters(
        provider_id: Annotated[
            str, Path(description="The unique identifier of the finetune provider.")
        ],
    ) -> list[FineTuneParameter]:
        if provider_id not in finetune_registry:
            raise HTTPException(
                status_code=400, detail=f"Fine tune provider '{provider_id}' not found"
            )
        finetune_adapter_class = finetune_registry[provider_id]  # type: ignore[invalid-argument-type]
        return finetune_adapter_class.available_parameters()

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/finetune_dataset_info",
        summary="Get Finetune Dataset Info",
        tags=["Fine-tuning"],
        openapi_extra=ALLOW_AGENT,
    )
    async def finetune_dataset_info(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        tool_ids: Annotated[
            list[str] | None,
            Query(description="Optional list of tool/skill IDs to filter by."),
        ] = None,
        empty_tool_filter: Annotated[
            bool,
            Query(
                description="If true and tool_ids is None, treats as empty tool filter."
            ),
        ] = False,
    ) -> FinetuneDatasetInfo:
        # In the fine-tune UI, "no tools/skills selected" should mean `tool_ids=[]`,
        # but `openapi-fetch` omits empty arrays, so we recover that state from
        # `empty_tool_filter=true`.
        if empty_tool_filter and tool_ids is None:
            tool_ids = []

        task = task_from_id(project_id, task_id)
        # Only include datasets that is part of a finetune.
        # Orphan datasets are created when user creates a dataset but didn't create a finetune.
        existing_datasets = [
            dataset
            for dataset in task.dataset_splits()
            if any(
                finetune.dataset_split_id == dataset.id for finetune in task.finetunes()
            )
        ]

        existing_finetunes = task.finetunes()

        finetune_tags = compute_finetune_tag_info(task, tool_filter=None)
        eligible_finetune_tags = compute_finetune_tag_info(task, tool_filter=tool_ids)

        eligible_datasets = existing_datasets
        # Only filter datasets when the caller provided a tool/skill selection.
        # `tool_ids=[]` is a real filter meaning "match datasets with no tools/skills".
        if tool_ids is not None:
            required_tools_set = set(tool_ids)
            eligible_datasets = []
            for dataset in existing_datasets:
                tool_info = dataset.tool_info()
                # Reusable datasets must have a uniform tool/skill set.
                # `tool_info.tools=None` means the dataset mixes different tool/skill selections.
                if (
                    tool_info.tools is not None
                    and set(tool_info.tools) == required_tools_set
                ):
                    eligible_datasets.append(dataset)

        return FinetuneDatasetInfo(
            existing_datasets=existing_datasets,
            existing_finetunes=existing_finetunes,
            finetune_tags=finetune_tags,
            eligible_datasets=eligible_datasets,
            eligible_finetune_tags=eligible_finetune_tags,
        )

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/dataset_splits",
        summary="Create Dataset Split",
        tags=["Fine-tuning"],
        openapi_extra=ALLOW_AGENT,
    )
    async def create_dataset_split(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        request: CreateDatasetSplitRequest,
    ) -> DatasetSplit:
        task = task_from_id(project_id, task_id)
        split_definitions = api_split_types[request.dataset_split_type]

        name = request.name
        if not name:
            name = generate_memorable_name()

        dataset_split = DatasetSplit.from_task(
            name,
            task,
            split_definitions,
            filter_id=request.filter_id,
            description=request.description,
        )
        dataset_split.save_to_file()
        return dataset_split

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/finetunes",
        summary="Create Finetune",
        tags=["Fine-tuning"],
        openapi_extra=agent_policy_require_approval(
            "Creating a fine-tune incurs cost. Allow agent to proceed?"
        ),
    )
    async def create_finetune(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        request: CreateFinetuneRequest,
    ) -> Finetune:
        task = task_from_id(project_id, task_id)
        if task.turn_mode == TurnMode.multiturn:
            raise HTTPException(
                status_code=400,
                detail="Fine-tuning is not supported for multi-turn tasks.",
            )
        if request.provider not in finetune_registry:
            raise HTTPException(
                status_code=400,
                detail=f"Fine tune provider '{request.provider}' not found",
            )
        finetune_adapter_class = finetune_registry[request.provider]  # type: ignore[invalid-argument-type]

        dataset = DatasetSplit.from_id_and_parent_path(request.dataset_id, task.path)
        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset split with ID '{request.dataset_id}' not found",
            )

        if not request.system_message_generator and not request.custom_system_message:
            raise HTTPException(
                status_code=400,
                detail="System message generator or custom system message is required",
            )

        skills: list[Skill] = []
        if request.run_config_properties is not None:
            skills_dict = load_skills_for_task(task, request.run_config_properties)
            skills = list(skills_dict.values())

        system_message = system_message_from_request(
            task,
            request.custom_system_message,
            request.system_message_generator,
            skills=skills,
        )
        thinking_instructions = thinking_instructions_from_request(
            task, request.data_strategy, request.custom_thinking_instructions
        )

        _, finetune_model = await finetune_adapter_class.create_and_start(
            dataset=dataset,
            provider_id=request.provider,
            provider_base_model_id=request.base_model_id,
            train_split_name=request.train_split_name,
            system_message=system_message,
            thinking_instructions=thinking_instructions,
            parameters=request.parameters,
            name=request.name,
            description=request.description,
            validation_split_name=request.validation_split_name,
            data_strategy=request.data_strategy,
            run_config=request.run_config_properties,
        )

        return finetune_model

    @app.get(
        "/api/download_dataset_jsonl",
        summary="Download Dataset JSONL",
        tags=["Fine-tuning"],
        openapi_extra=DENY_AGENT,
    )
    async def download_dataset_jsonl(
        project_id: Annotated[
            str, Query(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Query(description="The unique identifier of the task within the project."),
        ],
        dataset_id: Annotated[
            str, Query(description="The unique identifier of the dataset split.")
        ],
        split_name: Annotated[
            str,
            Query(
                description="The name of the split to download (e.g., 'train', 'test', 'validation')."
            ),
        ],
        format_type: Annotated[
            str, Query(description="The format type for the JSONL output.")
        ],
        data_strategy: Annotated[
            str, Query(description="The data strategy for formatting the output.")
        ],
        system_message_generator: Annotated[
            str | None, Query(description="The system message generator to use.")
        ] = None,
        custom_system_message: Annotated[
            str | None,
            Query(description="Custom system message if not using a generator."),
        ] = None,
        custom_thinking_instructions: Annotated[
            str | None,
            Query(description="Custom thinking instructions for thinking models."),
        ] = None,
    ) -> StreamingResponse:
        if format_type not in [format.value for format in DatasetFormat]:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset format '{format_type}' not found",
            )
        format_type_typed = DatasetFormat(format_type)
        if data_strategy not in [strategy.value for strategy in ChatStrategy]:
            raise HTTPException(
                status_code=400,
                detail=f"Data strategy '{data_strategy}' not found",
            )

        data_strategy_typed = ChatStrategy(data_strategy)

        task = task_from_id(project_id, task_id)
        if task.turn_mode == TurnMode.multiturn:
            raise HTTPException(
                status_code=400,
                detail="Fine-tuning is not supported for multi-turn tasks.",
            )
        dataset = DatasetSplit.from_id_and_parent_path(dataset_id, task.path)
        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset split with ID '{dataset_id}' not found",
            )
        if split_name not in dataset.split_contents:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset split with name '{split_name}' not found",
            )

        tool_info = dataset.tool_info()
        if tool_info.tools is None:
            raise HTTPException(
                status_code=400,
                detail="Dataset contains mixed tool/skill selections and cannot be exported",
            )
        skills_dict = load_skills_from_tool_ids(task, tool_info.tools)
        skills = list(skills_dict.values())

        system_message = system_message_from_request(
            task, custom_system_message, system_message_generator, skills=skills
        )
        thinking_instructions = thinking_instructions_from_request(
            task, data_strategy_typed, custom_thinking_instructions
        )

        dataset_formatter = DatasetFormatter(
            dataset=dataset,
            system_message=system_message,
            thinking_instructions=thinking_instructions,
        )
        path = await dataset_formatter.dump_to_file(
            split_name,
            format_type_typed,
            data_strategy_typed,
        )

        # set headers to force download in a browser
        headers = {
            "Content-Disposition": f'attachment; filename="{path.name}"',
            "Content-Type": "application/jsonl",
        }

        return StreamingResponse(open(path, "rb"), headers=headers)


def system_message_from_request(
    task: Task,
    custom_system_message: str | None,
    system_message_generator: str | None,
    skills: list[Skill] | None = None,
) -> str:
    system_message = custom_system_message
    if not system_message or (
        len(system_message) == 0 and system_message_generator is not None
    ):
        if system_message_generator is None:
            raise HTTPException(
                status_code=400,
                detail="System message generator is required when custom system message is not provided",
            )
        try:
            prompt_builder = prompt_builder_from_id(system_message_generator, task)
            system_message = prompt_builder.build_prompt(
                include_json_instructions=False,
                skills=skills,
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error generating system message using generator: {system_message_generator}. Source error: {e!s}",
            )
    if system_message is None or len(system_message) == 0:
        raise HTTPException(
            status_code=400,
            detail="System message is required",
        )

    return system_message


def thinking_instructions_from_request(
    task: Task,
    data_strategy: ChatStrategy,
    custom_thinking_instructions: str | None,
) -> str | None:
    if data_strategy not in THINKING_DATA_STRATEGIES:
        # Not using COT/Thinking style
        return None

    if data_strategy == ChatStrategy.single_turn_r1_thinking:
        return None

    if custom_thinking_instructions:
        # prefer custom instructions
        return custom_thinking_instructions

    # default for this task
    return chain_of_thought_prompt(task)


async def fetch_fireworks_finetune_models() -> list[FinetuneProviderModel]:
    api_key = Config.shared().fireworks_api_key
    if not api_key:
        return []

    url = "https://api.fireworks.ai/v1/accounts/fireworks/models"

    params = {
        "pageSize": 200,  # Max allowed
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    models = []

    # Paginate through all models
    async with httpx.AsyncClient() as client:
        while True:
            response = await client.get(url, params=params, headers=headers)
            json = response.json()
            if "models" not in json or not isinstance(json["models"], list):
                raise ValueError(
                    f"Invalid response from Fireworks. Expected list of models in 'models' key: [{response.status_code}] {response.text}"
                )
            models.extend(json["models"])
            next_page_token = json.get("nextPageToken")
            if (
                next_page_token
                and isinstance(next_page_token, str)
                and len(next_page_token) > 0
            ):
                params = {
                    "pageSize": 200,
                    "pageToken": next_page_token,
                }
            else:
                break

    tuneable_models = []
    for model in models:
        is_supervised_tunable = model.get("supervisedLoraTunable", False) or model.get(
            "supervisedFullParameterTunable", False
        )
        if is_supervised_tunable and "displayName" in model and "name" in model:
            id = model["name"]
            id_tail = id.split("/")[-1]
            if id_tail not in FIREWORKS_SUPPORTED_FINETUNE_MODELS:
                continue
            # Display name is sometimes empty, so use the name from the API name if needed
            display_name = model["displayName"]
            if display_name.strip() == "":
                name = id_tail
            else:
                name = display_name + " (" + id_tail + ")"

            # Check if the model supports tools via 'supportTools'
            supports_tools = model.get("supportsTools", False)

            tuneable_models.append(
                FinetuneProviderModel(
                    name=name,
                    id=id,
                    supports_function_calling=supports_tools,
                )
            )

    return tuneable_models


# While technical COT legacy is supported, we don't want new tunes using it.
DEFAULT_DATA_STRATEGIES = [
    ChatStrategy.single_turn,
    ChatStrategy.two_message_cot,
]


def data_strategies_from_model_provider(
    provider: KilnModelProvider,
) -> list[ChatStrategy]:
    if provider.parser == ModelParserID.r1_thinking:
        return [
            ChatStrategy.single_turn_r1_thinking,
        ]
    return DEFAULT_DATA_STRATEGIES


def data_strategies_from_finetune_id(
    provider_finetune_id: str,
) -> list[ChatStrategy]:
    if "qwen3" in provider_finetune_id.lower():
        return [
            ChatStrategy.single_turn,
            ChatStrategy.single_turn_r1_thinking,
        ]

    r1_must_include = ["r1", "qwq"]
    if any(substring in provider_finetune_id.lower() for substring in r1_must_include):
        return [
            ChatStrategy.single_turn_r1_thinking,
        ]
    return DEFAULT_DATA_STRATEGIES


def infer_data_strategies_for_model(
    available_models: list[KilnModel],
    provider_finetune_id: str,
    provider_name: str,
) -> list[ChatStrategy]:
    # we don't have built-in models for fireworks models, so we infer the data strategy from the model name
    if provider_name == ModelProviderName.fireworks_ai:
        return data_strategies_from_finetune_id(provider_finetune_id)

    # where we have built-in models, we can infer the data strategy from the object itself
    for model in available_models:
        for provider in model.providers:
            if (
                provider.name == provider_name
                and provider.provider_finetune_id == provider_finetune_id
            ):
                return data_strategies_from_model_provider(provider)

    # for everything else, we don't know what the data strategy is, so we use the default
    return DEFAULT_DATA_STRATEGIES
