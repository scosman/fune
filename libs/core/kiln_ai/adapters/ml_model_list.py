from enum import Enum
from typing import List

from pydantic import BaseModel, model_validator

from kiln_ai.datamodel.datamodel_enums import (
    ChatStrategy,
    KilnMimeType,
    ModelProviderName,
    StructuredOutputMode,
)

"""
Provides model configuration and management for various LLM providers and models.
This module handles the integration with different AI model providers and their respective models,
including configuration, validation, and instantiation of language models.
"""


class ModelFamily(str, Enum):
    """
    Enumeration of supported model families/architectures.
    """

    gpt = "gpt"
    llama = "llama"
    muse = "muse"
    phi = "phi"
    mistral = "mistral"
    gemma = "gemma"
    gemini = "gemini"
    claude = "claude"
    mixtral = "mixtral"
    qwen = "qwen"
    deepseek = "deepseek"
    dolphin = "dolphin"
    grok = "grok"
    kimi = "kimi"
    hunyuan = "hunyuan"
    glm = "glm"
    ernie = "ernie"
    minimax = "minimax"
    pangu = "pangu"
    bytedance = "bytedance"
    stepfun = "stepfun"
    mimo = "mimo"
    nemotron = "nemotron"
    arcee = "arcee"
    sakana = "sakana"
    thinking_machines = "thinking_machines"


# Where models have instruct and raw versions, instruct is default and raw is specified
class ModelName(str, Enum):
    """
    Enumeration of specific model versions supported by the system.
    Where models have instruct and raw versions, instruct is default and raw is specified.
    """

    fugu_max = "fugu_max"
    fugu_ultra_v2 = "fugu_ultra_v2"
    fugu_ultra = "fugu_ultra"
    muse_spark_1_2 = "muse_spark_1_2"
    muse_spark_1_1 = "muse_spark_1_1"
    llama_3_1_8b = "llama_3_1_8b"
    llama_3_1_70b = "llama_3_1_70b"
    llama_3_1_405b = "llama_3_1_405b"
    llama_3_2_1b = "llama_3_2_1b"
    llama_3_2_3b = "llama_3_2_3b"
    llama_3_2_11b = "llama_3_2_11b"
    llama_3_2_90b = "llama_3_2_90b"
    llama_3_3_70b = "llama_3_3_70b"
    llama_4_maverick = "llama_4_maverick"
    llama_4_scout = "llama_4_scout"
    gpt_6_astra = "gpt_6_astra"
    gpt_5_6_sol = "gpt_5_6_sol"
    gpt_5_6_terra = "gpt_5_6_terra"
    gpt_5_6_luna = "gpt_5_6_luna"
    gpt_5_5 = "gpt_5_5"
    gpt_5_4 = "gpt_5_4"
    gpt_5_4_pro = "gpt_5_4_pro"
    gpt_5_4_mini = "gpt_5_4_mini"
    gpt_5_4_nano = "gpt_5_4_nano"
    gpt_5_2 = "gpt_5_2"
    gpt_5_2_pro = "gpt_5_2_pro"
    gpt_5_3_chat = "gpt_5_3_chat"
    gpt_5_2_chat = "gpt_5_2_chat"
    gpt_5 = "gpt_5"
    gpt_5_1 = "gpt_5_1"
    gpt_5_chat = "gpt_5_chat"
    gpt_5_mini = "gpt_5_mini"
    gpt_5_nano = "gpt_5_nano"
    gpt_4o_mini = "gpt_4o_mini"
    gpt_4o = "gpt_4o"
    gpt_4_1 = "gpt_4_1"
    gpt_4_1_mini = "gpt_4_1_mini"
    gpt_4_1_nano = "gpt_4_1_nano"
    gpt_o3_low = "gpt_o3_low"
    gpt_o3_medium = "gpt_o3_medium"
    gpt_o3_high = "gpt_o3_high"
    gpt_oss_20b = "gpt_oss_20b"
    gpt_oss_120b = "gpt_oss_120b"
    gpt_o1_low = "gpt_o1_low"
    gpt_o1_medium = "gpt_o1_medium"
    gpt_o1_high = "gpt_o1_high"
    gpt_o4_mini_low = "gpt_o4_mini_low"
    gpt_o4_mini_medium = "gpt_o4_mini_medium"
    gpt_o4_mini_high = "gpt_o4_mini_high"
    gpt_o3_mini_low = "gpt_o3_mini_low"
    gpt_o3_mini_medium = "gpt_o3_mini_medium"
    gpt_o3_mini_high = "gpt_o3_mini_high"
    phi_3_5 = "phi_3_5"
    phi_4 = "phi_4"
    phi_4_5p6b = "phi_4_5p6b"
    phi_4_mini = "phi_4_mini"
    mistral_large = "mistral_large"
    mistral_3_large_2512 = "mistral_3_large_2512"
    mistral_nemo = "mistral_nemo"
    mistral_small_4 = "mistral_small_4"
    mistral_small_3 = "mistral_small_3"
    mistral_medium_3_5 = "mistral_medium_3_5"
    mistral_medium_3_1 = "mistral_medium_3_1"
    mistral_small_creative = (
        "mistral_small_creative"  # mistralai/mistral-small-creative
    )
    ministral_3_14b_2512 = "ministral_3_14b_2512"
    ministral_3_8b_2512 = "ministral_3_8b_2512"
    ministral_3_3b_2512 = "ministral_3_3b_2512"
    magistral_medium = "magistral_medium"
    magistral_medium_thinking = "magistral_medium_thinking"
    gemma_4_31b = "gemma_4_31b"
    gemma_4_27b = "gemma_4_27b"
    gemma_4_e4b = "gemma_4_e4b"
    gemma_4_e2b = "gemma_4_e2b"
    gemma_2_2b = "gemma_2_2b"
    gemma_2_9b = "gemma_2_9b"
    gemma_2_27b = "gemma_2_27b"
    gemma_3_0p27b = "gemma_3_0p27b"
    gemma_3_1b = "gemma_3_1b"
    gemma_3_4b = "gemma_3_4b"
    gemma_3_12b = "gemma_3_12b"
    gemma_3_27b = "gemma_3_27b"
    gemma_3n_2b = "gemma_3n_2b"
    gemma_3n_4b = "gemma_3n_4b"
    claude_fable_5_1 = "claude_fable_5_1"
    claude_fable_5 = "claude_fable_5"
    claude_3_5_haiku = "claude_3_5_haiku"
    claude_4_5_haiku = "claude_4_5_haiku"
    claude_3_5_sonnet = "claude_3_5_sonnet"
    claude_3_7_sonnet = "claude_3_7_sonnet"
    claude_3_7_sonnet_thinking = "claude_3_7_sonnet_thinking"
    claude_sonnet_4 = "claude_sonnet_4"
    claude_sonnet_5 = "claude_sonnet_5"
    claude_sonnet_4_6 = "claude_sonnet_4_6"
    claude_sonnet_4_5 = "claude_sonnet_4_5"
    claude_opus_4 = "claude_opus_4"
    claude_opus_5 = "claude_opus_5"
    claude_opus_4_1 = "claude_opus_4_1"
    claude_opus_4_5 = "claude_opus_4_5"
    claude_opus_4_6 = "claude_opus_4_6"
    claude_opus_4_7 = "claude_opus_4_7"
    claude_opus_4_8 = "claude_opus_4_8"
    gemini_1_5_flash = "gemini_1_5_flash"
    gemini_1_5_flash_8b = "gemini_1_5_flash_8b"
    gemini_1_5_pro = "gemini_1_5_pro"
    gemini_2_0_flash = "gemini_2_0_flash"
    gemini_2_0_flash_lite = "gemini_2_0_flash_lite"
    gemini_2_5_pro = "gemini_2_5_pro"
    gemini_2_5_flash = "gemini_2_5_flash"
    gemini_2_5_flash_lite = "gemini_2_5_flash_lite"
    gemini_3_5_flash_lite = "gemini_3_5_flash_lite"
    gemini_3_1_flash_lite = "gemini_3_1_flash_lite"
    gemini_3_1_flash_lite_preview = "gemini_3_1_flash_lite_preview"
    gemini_3_1_pro_preview = "gemini_3_1_pro_preview"
    gemini_3_pro_preview = "gemini_3_pro_preview"
    gemini_3_8_flash = "gemini_3_8_flash"
    gemini_3_7_flash = "gemini_3_7_flash"
    gemini_3_6_flash = "gemini_3_6_flash"
    gemini_3_5_flash = "gemini_3_5_flash"
    gemini_3_flash = "gemini_3_flash"
    nemotron_70b = "nemotron_70b"
    nemotron_3_ultra = "nemotron_3_ultra"
    nemotron_3_super = "nemotron_3_super"
    nemotron_3_nano = "nemotron_3_nano"
    mixtral_8x7b = "mixtral_8x7b"
    qwen_2p5_7b = "qwen_2p5_7b"
    qwen_2p5_14b = "qwen_2p5_14b"
    qwen_2p5_72b = "qwen_2p5_72b"
    qwen_2p5_vl_3b = "qwen_2p5_vl_3b"
    qwen_2p5_vl_7b = "qwen_2p5_vl_7b"
    qwen_2p5_vl_32b = "qwen_2p5_vl_32b"
    qwen_2p5_vl_72b = "qwen_2p5_vl_72b"
    qwq_32b = "qwq_32b"
    deepseek_4_1_flash = "deepseek_4_1_flash"
    deepseek_4_pro = "deepseek_4_pro"
    deepseek_4_flash = "deepseek_4_flash"
    deepseek_3_2 = "deepseek_3_2"
    deepseek_3_1 = "deepseek_3_1"
    deepseek_3_1_terminus = "deepseek_3_1_terminus"
    deepseek_3 = "deepseek_3"
    deepseek_r1 = "deepseek_r1"
    deepseek_r1_0528 = "deepseek_r1_0528"
    deepseek_r1_0528_distill_qwen3_8b = "deepseek_r1_0528_distill_qwen3_8b"
    deepseek_r1_distill_qwen_32b = "deepseek_r1_distill_qwen_32b"
    deepseek_r1_distill_llama_70b = "deepseek_r1_distill_llama_70b"
    deepseek_r1_distill_qwen_14b = "deepseek_r1_distill_qwen_14b"
    deepseek_r1_distill_qwen_1p5b = "deepseek_r1_distill_qwen_1p5b"
    deepseek_r1_distill_qwen_7b = "deepseek_r1_distill_qwen_7b"
    deepseek_r1_distill_llama_8b = "deepseek_r1_distill_llama_8b"
    dolphin_2_9_8x22b = "dolphin_2_9_8x22b"
    grok_2 = "grok_2"
    grok_3 = "grok_3"
    grok_3_mini = "grok_3_mini"
    grok_4_5 = "grok_4_5"
    grok_4_3 = "grok_4_3"
    grok_4_20 = "grok_4_20"
    grok_4_1_fast = "grok_4_1_fast"
    grok_4 = "grok_4"
    qwen_3p5_flash = "qwen_3p5_flash"
    qwen_3p5_9b = "qwen_3p5_9b"
    qwen_3p5_122b_a10b = "qwen_3p5_122b_a10b"
    qwen_3p5_27b = "qwen_3p5_27b"
    qwen_3p5_35b_a3b = "qwen_3p5_35b_a3b"
    qwen_3p8_max = "qwen_3p8_max"
    qwen_3p8_2p4t_a95b = "qwen_3p8_2p4t_a95b"
    qwen_3p8_flash = "qwen_3p8_flash"
    qwen_3p8_27b = "qwen_3p8_27b"
    qwen_3p7_flash = "qwen_3p7_flash"
    qwen_3p7_plus = "qwen_3p7_plus"
    qwen_3p7_max = "qwen_3p7_max"
    qwen_3p6_flash = "qwen_3p6_flash"
    qwen_3p6_35b_a3b = "qwen_3p6_35b_a3b"
    qwen_3p6_27b = "qwen_3p6_27b"
    qwen_3p6_plus = "qwen_3p6_plus"
    qwen_3p5_plus = "qwen_3p5_plus"
    qwen_3p5_397b_a17b = "qwen_3p5_397b_a17b"
    qwen_3_next_80b_a3b = "qwen_3_next_80b_a3b"
    qwen_3_next_80b_a3b_thinking = "qwen_3_next_80b_a3b_thinking"
    qwen_3_max_thinking = "qwen_3_max_thinking"
    qwen_3_max = "qwen_3_max"
    qwen_3_0p6b = "qwen_3_0p6b"
    qwen_3_0p6b_no_thinking = "qwen_3_0p6b_no_thinking"
    qwen_3_1p7b = "qwen_3_1p7b"
    qwen_3_1p7b_no_thinking = "qwen_3_1p7b_no_thinking"
    qwen_3_4b = "qwen_3_4b"
    qwen_3_4b_no_thinking = "qwen_3_4b_no_thinking"
    qwen_3_8b = "qwen_3_8b"
    qwen_3_8b_no_thinking = "qwen_3_8b_no_thinking"
    qwen_3_14b = "qwen_3_14b"
    qwen_3_14b_no_thinking = "qwen_3_14b_no_thinking"
    qwen_3_30b_a3b_2507 = "qwen_3_30b_a3b_2507"
    qwen_3_30b_a3b = "qwen_3_30b_a3b"
    qwen_3_30b_a3b_2507_no_thinking = "qwen_3_30b_a3b_2507_no_thinking"
    qwen_3_30b_a3b_no_thinking = "qwen_3_30b_a3b_no_thinking"
    qwen_3_32b = "qwen_3_32b"
    qwen_3_32b_no_thinking = "qwen_3_32b_no_thinking"
    qwen_3_235b_a22b_2507 = "qwen_3_235b_a22b_2507"
    qwen_3_235b_a22b = "qwen_3_235b_a22b"
    qwen_3_235b_a22b_2507_no_thinking = "qwen_3_235b_a22b_2507_no_thinking"
    qwen_3_235b_a22b_no_thinking = "qwen_3_235b_a22b_no_thinking"
    qwen_3_vl_2b = "qwen_3_vl_2b"
    qwen_3_vl_4b = "qwen_3_vl_4b"
    qwen_3_vl_8b = "qwen_3_vl_8b"
    qwen_3_vl_30b = "qwen_3_vl_30b"
    qwen_3_vl_32b = "qwen_3_vl_32b"
    qwen_3_vl_235b_a22b = "qwen_3_vl_235b_a22b"
    qwen_3_vl_235b_a22b_no_thinking = "qwen_3_vl_235b_a22b_no_thinking"
    qwen_3_vl_32b_no_thinking = "qwen_3_vl_32b_no_thinking"
    qwen_3_vl_30b_a3b_no_thinking = "qwen_3_vl_30b_a3b_no_thinking"
    qwen_3_vl_8b_no_thinking = "qwen_3_vl_8b_no_thinking"
    qwen_long_l1_32b = "qwen_long_l1_32b"
    kimi_k3 = "kimi_k3"
    kimi_k2_6 = "kimi_k2_6"
    kimi_k2 = "kimi_k2"
    kimi_k2_0905 = "kimi_k2_0905"
    kimi_k2_thinking = "kimi_k2_thinking"
    kimi_k2_5 = "kimi_k2_5"
    kimi_dev_72b = "kimi_dev_72b"
    glm_5_3 = "glm_5_3"
    glm_5_3_flash = "glm_5_3_flash"
    glm_5_2 = "glm_5_2"
    glm_5_2_fast = "glm_5_2_fast"
    glm_5_1 = "glm_5_1"
    glm_5_turbo = "glm_5_turbo"
    glm_5v_turbo = "glm_5v_turbo"
    glm_5 = "glm_5"
    glm_4_7 = "glm_4_7"
    glm_4_7_flash = "glm_4_7_flash"
    glm_4_6 = "glm_4_6"
    glm_4_6v = "glm_4_6v"
    glm_4_5v = "glm_4_5v"
    glm_4_5 = "glm_4_5"
    glm_4_5_air = "glm_4_5_air"
    glm_4_1v_9b_thinking = "glm_4_1v_9b_thinking"
    glm_z1_32b_0414 = "glm_z1_32b_0414"
    glm_z1_9b_0414 = "glm_z1_9b_0414"
    ernie_4_5_300b_a47b = "ernie_4_5_300b_a47b"
    hunyuan_hy3 = "hunyuan_hy3"
    hunyuan_a13b = "hunyuan_a13b"
    hunyuan_a13b_no_thinking = "hunyuan_a13b_no_thinking"
    minimax_m3 = "minimax_m3"
    minimax_m2_7 = "minimax_m2_7"
    minimax_m2_5 = "minimax_m2_5"
    minimax_m2_1 = "minimax_m2_1"
    minimax_m2 = "minimax_m2"
    minimax_m1_80k = "minimax_m1_80k"
    pangu_pro_moe_72b_a16b = "pangu_pro_moe_72b_a16b"
    bytedance_seed_oss_36b = "bytedance_seed_oss_36b"
    bytedance_seed_1_6 = "bytedance_seed_1_6"
    bytedance_seed_1_6_flash = "bytedance_seed_1_6_flash"
    arcee_trinity_large_thinking = "arcee_trinity_large_thinking"
    stepfun_step3_7_flash = "stepfun_step3_7_flash"
    stepfun_step3 = "stepfun_step3"
    mimo_v2_pro = "mimo_v2_pro"
    mimo_v2_flash = "mimo_v2_flash"
    mimo_v2_omni = "mimo_v2_omni"
    mimo_v2_5 = "mimo_v2_5"
    mimo_v2_5_pro = "mimo_v2_5_pro"
    inkling = "inkling"


class ModelParserID(str, Enum):
    """
    Enumeration of supported model parsers.
    """

    r1_thinking = "r1_thinking"
    optional_r1_thinking = "optional_r1_thinking"


class ModelFormatterID(str, Enum):
    """
    Enumeration of supported model formatters.
    """

    qwen3_style_no_think = "qwen3_style_no_think"


class KilnModelProvider(BaseModel):
    """
    Configuration for a specific model provider.

    Attributes:
        name: The provider's identifier
        supports_structured_output: Whether the provider supports structured output formats
        supports_data_gen: Whether the provider supports data generation
        untested_model: Whether the model is untested (typically user added). The supports_ fields are not applicable.
        provider_finetune_id: The finetune ID for the provider, if applicable. Some providers like Fireworks load these from an API.
        structured_output_mode: The mode we should use to call the model for structured output, if it was trained with structured output.
        parser: A parser to use for the model, if applicable
        reasoning_capable: Whether the model is designed to output thinking in a structured format (eg <think></think>). If so we don't use COT across 2 calls, and ask for thinking and final response in the same call.
        tuned_chat_strategy: Used when a model is finetuned with a specific chat strategy, and it's best to use it at call time.
        supports_doc_extraction: Whether the provider is meant to support document extraction
        suggested_for_doc_extraction: Whether the model is suggested for document extraction
        suggested_for_synthetic_user: Whether the model is suggested to play the synthetic user in multi-turn evals: a fast, inexpensive chat model that reads a whole conversation each turn
        multimodal_capable: Whether the model supports multimodal inputs (e.g. images, audio, video, PDFs, etc.)
        multimodal_mime_types: The mime types that the model supports for multimodal inputs (e.g. image/jpeg, video/mp4, application/pdf, etc.)
        multimodal_requires_pdf_as_image: Whether the model requires PDFs to be processed as images
        supports_vision: Whether the model supports vision inputs (e.g. images, video)
    """

    name: ModelProviderName
    model_id: str | None = None
    supports_structured_output: bool = True
    supports_data_gen: bool = True
    suggested_for_data_gen: bool = False
    untested_model: bool = False
    provider_finetune_id: str | None = None
    structured_output_mode: StructuredOutputMode = StructuredOutputMode.default
    parser: ModelParserID | None = None
    formatter: ModelFormatterID | None = None
    reasoning_capable: bool = False
    supports_logprobs: bool = False
    suggested_for_evals: bool = False
    supports_function_calling: bool = True
    uncensored: bool = False
    suggested_for_uncensored_data_gen: bool = False
    tuned_chat_strategy: ChatStrategy | None = None
    supports_doc_extraction: bool = False
    suggested_for_doc_extraction: bool = False
    suggested_for_synthetic_user: bool = False
    multimodal_capable: bool = False
    multimodal_mime_types: List[str] | None = None
    multimodal_requires_pdf_as_image: bool = False
    supports_vision: bool = False

    # We need a more generalized way to handle custom provider parameters.
    # Making them quite declarative here for now, isolating provider specific logic
    # to this file. Later I should be able to override anything in this file via config.
    r1_openrouter_options: bool = False
    require_openrouter_reasoning: bool = False
    logprobs_openrouter_options: bool = False
    openrouter_skip_required_parameters: bool = False

    # OpenRouter-specific payload toggle for thinking effort.
    # When true, send `reasoning: {effort: <level>}` instead of `reasoning_effort`.
    # Use only for OpenRouter models that require the reasoning-object format.
    openrouter_reasoning_object: bool = False
    available_thinking_levels: dict[str, str] | None = None
    default_thinking_level: str | None = None
    ollama_model_aliases: List[str] | None = None
    anthropic_extended_thinking: bool = False
    # Opus 4.7/4.8 default thinking display to "omitted" (empty thinking text, signature
    # only). Set this for those models to request the readable reasoning summary.
    anthropic_summarized_thinking: bool = False
    gemini_reasoning_enabled: bool = False
    # Can only specify top_p or temp, not both. Opus 4.1 and Sonnet 4.5 for example.
    temp_top_p_exclusive: bool = False

    # some models on siliconflow allow dynamically disabling thinking
    # currently only supported by Qwen3 and tencent/Hunyuan-A13B-Instruct
    # ref: https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions
    siliconflow_enable_thinking: bool | None = None
    # enable this flag to make reasoning optional for structured output
    # some reasoning models on siliconflow do not return any reasoning for structured output
    # this is not uniform nor documented, so we need to test each model
    reasoning_optional_for_structured_output: bool | None = None

    # models have rate limits, which become very relevant when doing heavy processing like in RAG
    # this RPM gives a rough estimate of how many requests we should allow to run in parallel, it is
    # not exact and real rate limit rules are much more complex
    max_parallel_requests: int | None = None

    # For openai_compatible providers: the name of the custom provider (user specified)
    openai_compatible_provider_name: str | None = None

    # The model is deprecated and no longer supported on a specific provider
    deprecated: bool = False

    @model_validator(mode="after")
    def validate_openrouter_reasoning_object(self) -> "KilnModelProvider":
        if (
            self.openrouter_reasoning_object
            and self.name != ModelProviderName.openrouter
        ):
            raise ValueError(
                "openrouter_reasoning_object can only be true when provider is openrouter"
            )
        return self

    @model_validator(mode="after")
    def validate_default_thinking_level(self) -> "KilnModelProvider":
        if self.available_thinking_levels:
            if (
                self.default_thinking_level
                not in self.available_thinking_levels.values()
            ):
                raise ValueError(
                    "default_thinking_level must be one of the available_thinking_levels values"
                )
        return self


class KilnModel(BaseModel):
    """
    Configuration for a specific AI model.

    Attributes:
        family: The model's architecture family
        name: The model's identifier
        friendly_name: Human-readable name for the model
        providers: List of providers that offer this model
        supports_structured_output: Whether the model supports structured output formats
    """

    family: str
    name: str
    friendly_name: str
    providers: List[KilnModelProvider]

    # Editorial
    featured_rank: int | None = None
    editorial_notes: str | None = None


GPT_5_OPENROUTER_THINKING_LEVELS = {
    "Off/None": "none",
    "Minimal": "minimal",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
}

GPT_5_2_OPENROUTER_THINKING_LEVELS = {
    "Off/None": "none",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
}

GPT_5_4_OPENAI_THINKING_LEVELS = {
    "Off/None": "none",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
}

GPT_5_4_PRO_OPENAI_THINKING_LEVELS = {
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
}

# GPT-6 Astra supports reasoning effort levels low/medium/high/xhigh/max with a
# default of medium. Unlike the GPT-5.x models it does NOT support `none` or
# `minimal`, and it adds a new `max` level, so it needs its own constant.
GPT_6_ASTRA_OPENAI_THINKING_LEVELS = {
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
    "Max": "max",
}

GPT_5_1_OPENAI_THINKING_LEVELS = {
    "Off/None": "none",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
}

GPT_5_2_OPENAI_THINKING_LEVELS = {
    "Off/None": "none",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
}

GPT_5_2_PRO_OPENAI_THINKING_LEVELS = {
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
}

GEMINI_3_PRO_THINKING_LEVELS = {
    "Low": "low",
    "Medium": "medium",
    "High": "high",
}

GEMINI_3_FLASH_THINKING_LEVELS = {
    "Minimal": "minimal",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
}

GEMMA_4_THINKING_LEVELS = {
    "Off": "none",
    "On": "high",
}

GPT_5_OPENAI_THINKING_LEVELS = {
    "Minimal": "minimal",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
}


CLAUDE_OPENROUTER_THINKING_LEVELS = {
    "Off/None": "none",
    "Minimal": "minimal",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
}

CLAUDE_ANTHROPIC_EFFORT_THINKING_LEVELS = {
    "Off/None": "none",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
}

CLAUDE_SONNET_5_ANTHROPIC_THINKING_LEVELS = {
    "Off/None": "none",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
    "Max": "max",
}

CLAUDE_SONNET_5_OPENROUTER_THINKING_LEVELS = {
    "Off/None": "none",
    "Minimal": "minimal",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
    "Max": "max",
}

# Fable 5 requires reasoning and cannot disable it, so "none" is omitted (unlike
# the standard Claude OpenRouter levels which default to "none").
CLAUDE_FABLE_5_OPENROUTER_THINKING_LEVELS = {
    "Minimal": "minimal",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
}

CLAUDE_OPUS_4_7_ANTHROPIC_THINKING_LEVELS = {
    "Off/None": "none",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
    "Max": "max",
}

CLAUDE_OPUS_4_8_ANTHROPIC_THINKING_LEVELS = {
    "Off/None": "none",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
    "Max": "max",
}

CLAUDE_OPUS_5_ANTHROPIC_THINKING_LEVELS = {
    "Off/None": "none",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
    "Max": "max",
}

CLAUDE_FABLE_5_ANTHROPIC_THINKING_LEVELS = {
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
    "Max": "max",
}

DEEPSEEK_V4_OPENROUTER_THINKING_LEVELS = {
    "Off/None": "none",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Extra High": "xhigh",
}

GROK_4_3_OPENROUTER_THINKING_LEVELS = {
    "Off/None": "none",
    "Low": "low",
    "Medium": "medium",
    "High": "high",
}

# Fugu Ultra: reasoning is mandatory (no "none"); OpenRouter exposes only high/xhigh/max.
FUGU_ULTRA_OPENROUTER_THINKING_LEVELS = {
    "High": "high",
    "Extra High": "xhigh",
    "Max": "max",
}

KIMI_K3_OPENROUTER_THINKING_LEVELS = {
    "Low": "low",
    "High": "high",
    "Max": "max",
}

GROK_4_5_OPENROUTER_THINKING_LEVELS = {
    "Low": "low",
    "Medium": "medium",
    "High": "high",
}

# Groq restricts reasoning_effort on Qwen 3.6 to `none` or `default`; any other
# value is rejected with "`reasoning_effort` must be one of `none` or `default`".
QWEN_3P6_GROQ_THINKING_LEVELS = {
    "Off": "none",
    "On": "default",
}


built_in_models: List[KilnModel] = [
    # GPT 6 Astra
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_6_astra,
        friendly_name="GPT-6 Astra",
        featured_rank=2,
        editorial_notes="OpenAI's flagship GPT-6 model. Powerful reasoning and multimodal.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="gpt-6-astra",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_6_ASTRA_OPENAI_THINKING_LEVELS,
                default_thinking_level="medium",
                # OpenAI rejects reasoning_effort + tools on /v1/chat/completions
                # for gpt-5.4+. Disable function calling until Kiln routes these
                # models to /v1/responses.
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="openai/gpt-6-astra",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_6_ASTRA_OPENAI_THINKING_LEVELS,
                default_thinking_level="medium",
                # Use OpenRouter's reasoning object so reasoning is preserved
                # when tools are sent (the bare reasoning_effort param is
                # silently dropped on tool calls for these models).
                openrouter_reasoning_object=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5.6 Sol
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_6_sol,
        friendly_name="GPT-5.6 Sol",
        featured_rank=4,
        editorial_notes="OpenAI's most capable GPT model. Powerful reasoning and multimodal.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="gpt-5.6-sol",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_4_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                # OpenAI rejects reasoning_effort + tools on /v1/chat/completions
                # for gpt-5.4+. Disable function calling until Kiln routes these
                # models to /v1/responses.
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="openai/gpt-5.6-sol",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_4_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                # Use OpenRouter's reasoning object so reasoning is preserved
                # when tools are sent (the bare reasoning_effort param is
                # silently dropped on tool calls for these models).
                openrouter_reasoning_object=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5.6 Terra
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_6_terra,
        friendly_name="GPT-5.6 Terra",
        featured_rank=9,
        editorial_notes="OpenAI's balanced GPT-5.6 model. Strong reasoning and multimodal at a lower cost.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                suggested_for_synthetic_user=True,
                model_id="gpt-5.6-terra",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_4_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                # OpenAI rejects reasoning_effort + tools on /v1/chat/completions
                # for gpt-5.4+. Disable function calling until Kiln routes these
                # models to /v1/responses.
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                suggested_for_synthetic_user=True,
                model_id="openai/gpt-5.6-terra",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_4_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                # Use OpenRouter's reasoning object so reasoning is preserved
                # when tools are sent (the bare reasoning_effort param is
                # silently dropped on tool calls for these models).
                openrouter_reasoning_object=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5.6 Luna
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_6_luna,
        friendly_name="GPT-5.6 Luna",
        featured_rank=13,
        editorial_notes="OpenAI's fast, cost-efficient GPT-5.6 model. Optimized for speed and high-volume tasks.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                suggested_for_synthetic_user=True,
                model_id="gpt-5.6-luna",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_4_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                # OpenAI rejects reasoning_effort + tools on /v1/chat/completions
                # for gpt-5.4+. Disable function calling until Kiln routes these
                # models to /v1/responses.
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                suggested_for_synthetic_user=True,
                model_id="openai/gpt-5.6-luna",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_4_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                # Use OpenRouter's reasoning object so reasoning is preserved
                # when tools are sent (the bare reasoning_effort param is
                # silently dropped on tool calls for these models).
                openrouter_reasoning_object=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5.5
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_5,
        friendly_name="GPT-5.5",
        editorial_notes="OpenAI's most capable GPT model. Powerful reasoning and multimodal.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-5.5",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_4_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                # OpenAI rejects reasoning_effort + tools on /v1/chat/completions
                # for gpt-5.4+. Disable function calling until Kiln routes these
                # models to /v1/responses.
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-5.5",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_4_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                # Use OpenRouter's reasoning object so reasoning is preserved
                # when tools are sent (the bare reasoning_effort param is
                # silently dropped on tool calls for openai/gpt-5.5).
                openrouter_reasoning_object=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5.4
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_4,
        friendly_name="GPT-5.4",
        editorial_notes="OpenAI's most capable GPT model. Powerful reasoning and multimodal.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-5.4",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_4_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                # OpenAI rejects reasoning_effort + tools on /v1/chat/completions
                # for gpt-5.4 direct. Disable function calling until Kiln routes
                # these models to /v1/responses. The OpenRouter route is unaffected.
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-5.4",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_4_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5.4 Pro
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_4_pro,
        friendly_name="GPT-5.4 Pro",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-5.4-pro",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_4_PRO_OPENAI_THINKING_LEVELS,
                default_thinking_level="medium",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-5.4-pro",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_4_PRO_OPENAI_THINKING_LEVELS,
                default_thinking_level="medium",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5.4 Mini
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_4_mini,
        friendly_name="GPT-5.4 Mini",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-5.4-mini",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-5.4-mini",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_4_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5.4 Nano
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_4_nano,
        friendly_name="GPT-5.4 Nano",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-5.4-nano",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-5.4-nano",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_4_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5.3 Instant
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_3_chat,
        friendly_name="GPT-5.3 Instant",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-5.3-chat-latest",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-5.3-chat",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5.2
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_2,
        friendly_name="GPT-5.2",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-5.2",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_2_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-5.2",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_2_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5.2 Pro
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_2_pro,
        friendly_name="GPT-5.2 Pro",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-5.2-pro",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_2_PRO_OPENAI_THINKING_LEVELS,
                default_thinking_level="medium",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-5.2-pro",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_2_PRO_OPENAI_THINKING_LEVELS,
                default_thinking_level="medium",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5.2 Chat
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_2_chat,
        friendly_name="GPT-5.2 Chat",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-5.2-chat-latest",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-5.2-chat",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_2_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5.1
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_1,
        friendly_name="GPT-5.1",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-5.1",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_1_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-5.1",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_1_OPENAI_THINKING_LEVELS,
                default_thinking_level="none",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5,
        friendly_name="GPT-5",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-5",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_OPENAI_THINKING_LEVELS,
                default_thinking_level="medium",
                supports_doc_extraction=True,
                supports_vision=True,
                supports_logprobs=False,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-5",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_OPENAI_THINKING_LEVELS,
                default_thinking_level="medium",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5 Mini
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_mini,
        friendly_name="GPT-5 Mini",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-5-mini",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_OPENAI_THINKING_LEVELS,
                default_thinking_level="medium",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-5-mini",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_OPENAI_THINKING_LEVELS,
                default_thinking_level="medium",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5 Nano
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_nano,
        friendly_name="GPT-5 Nano",
        editorial_notes="OpenAI on a budget. 3% the cost of GPT 5. Great for easier tasks.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-5-nano",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_OPENAI_THINKING_LEVELS,
                default_thinking_level="medium",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-5-nano",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GPT_5_OPENAI_THINKING_LEVELS,
                default_thinking_level="medium",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 5 Chat
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_5_chat,
        friendly_name="GPT-5 Chat",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-5-chat-latest",
                # Oddly no json_schema support for this model.
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                available_thinking_levels=GPT_5_OPENAI_THINKING_LEVELS,
                default_thinking_level="medium",
                supports_function_calling=False,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GPT 4.1
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_4_1,
        friendly_name="GPT 4.1",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-4.1",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_logprobs=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-4.1",
                structured_output_mode=StructuredOutputMode.json_schema,
                # OpenRouter does not return the logprobs for this model
                supports_logprobs=False,
                # logprobs_openrouter_options does not help this particular
                # model at the moment
                # logprobs_openrouter_options=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="gpt-4.1",
            ),
        ],
    ),
    # GPT 4.1 Mini
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_4_1_mini,
        friendly_name="GPT 4.1 Mini",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-4.1-mini",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_logprobs=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-4.1-mini",
                structured_output_mode=StructuredOutputMode.json_schema,
                # OpenRouter does not return the logprobs for this model
                supports_logprobs=False,
                # logprobs_openrouter_options does not help this particular
                # model at the moment
                # logprobs_openrouter_options=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="gpt-4.1-mini",
            ),
        ],
    ),
    # GPT 4.1 Nano
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_4_1_nano,
        friendly_name="GPT 4.1 Nano",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-4.1-nano",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_logprobs=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-4.1-nano",
                structured_output_mode=StructuredOutputMode.json_schema,
                # OpenRouter does not return the logprobs for this model
                supports_logprobs=False,
                # logprobs_openrouter_options does not help this particular
                # model at the moment
                # logprobs_openrouter_options=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="gpt-4.1-nano",
            ),
        ],
    ),
    # GPT 4o
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_4o,
        friendly_name="GPT 4o",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-4o",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_logprobs=True,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-4o",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_logprobs=True,
                logprobs_openrouter_options=True,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="gpt-4o",
            ),
        ],
    ),
    # GPT 4o Mini
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_4o_mini,
        friendly_name="GPT 4o Mini",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="gpt-4o-mini",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_logprobs=True,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-4o-mini",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_logprobs=True,
                logprobs_openrouter_options=True,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="gpt-4o-mini",
            ),
        ],
    ),
    # GPT o4 Mini Low
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_o4_mini_low,
        friendly_name="GPT o4 Mini - Low",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="o4-mini",
                available_thinking_levels={"Low": "low"},
                default_thinking_level="low",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="o4-mini",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels={"Low": "low"},
                default_thinking_level="low",
            ),
        ],
    ),
    # GPT o4 Mini Medium
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_o4_mini_medium,
        friendly_name="GPT o4 Mini - Medium",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="o4-mini",
                available_thinking_levels={"Medium": "medium"},
                default_thinking_level="medium",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="o4-mini",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels={"Medium": "medium"},
                default_thinking_level="medium",
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/o4-mini",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
        ],
    ),
    # GPT o4 Mini High
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_o4_mini_high,
        friendly_name="GPT o4 Mini - High",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="o4-mini",
                available_thinking_levels={"High": "high"},
                default_thinking_level="high",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="o4-mini",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels={"High": "high"},
                default_thinking_level="high",
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/o4-mini-high",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
        ],
    ),
    # GPT o3 Low
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_o3_low,
        friendly_name="GPT o3 - Low",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="o3",
                available_thinking_levels={"Low": "low"},
                default_thinking_level="low",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="o3",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels={"Low": "low"},
                default_thinking_level="low",
            ),
        ],
    ),
    # GPT o3 Medium
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_o3_medium,
        friendly_name="GPT o3 - Medium",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="o3",
                available_thinking_levels={"Medium": "medium"},
                default_thinking_level="medium",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="o3",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels={"Medium": "medium"},
                default_thinking_level="medium",
            ),
        ],
    ),
    # GPT o3 High
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_o3_high,
        friendly_name="GPT o3 - High",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="o3",
                available_thinking_levels={"High": "high"},
                default_thinking_level="high",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="o3",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels={"High": "high"},
                default_thinking_level="high",
            ),
        ],
    ),
    # GPT o3 Mini Low
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_o3_mini_low,
        friendly_name="GPT o3 Mini - Low",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="o3-mini",
                available_thinking_levels={"Low": "low"},
                default_thinking_level="low",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="o3-mini",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels={"Low": "low"},
                default_thinking_level="low",
            ),
        ],
    ),
    # GPT o3 Mini Medium
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_o3_mini_medium,
        friendly_name="GPT o3 Mini - Medium",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="o3-mini",
                available_thinking_levels={"Medium": "medium"},
                default_thinking_level="medium",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="o3-mini",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels={"Medium": "medium"},
                default_thinking_level="medium",
            ),
        ],
    ),
    # GPT o3 Mini High
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_o3_mini_high,
        friendly_name="GPT o3 Mini - High",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="o3-mini",
                available_thinking_levels={"High": "high"},
                default_thinking_level="high",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="o3-mini",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels={"High": "high"},
                default_thinking_level="high",
            ),
        ],
    ),
    # GPT OSS 120B
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_oss_120b,
        friendly_name="GPT OSS 120B",
        editorial_notes="OpenAI's capable open-weight model. Speeds of >1,000 tokens/s on Cerebras and Groq.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-oss-120b:exacto",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                require_openrouter_reasoning=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.groq,
                model_id="openai/gpt-oss-120b",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
            KilnModelProvider(
                name=ModelProviderName.cerebras,
                model_id="gpt-oss-120b",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/gpt-oss-120b",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="gpt-oss:120b",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="openai/gpt-oss-120b",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                # Featherless doesn't advertise tool_use for this model
                supports_function_calling=False,
            ),
        ],
    ),
    # GPT OSS 20B
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_oss_20b,
        friendly_name="GPT OSS 20B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="openai/gpt-oss-20b",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                require_openrouter_reasoning=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.groq,
                model_id="openai/gpt-oss-20b",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/gpt-oss-20b",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="gpt-oss:20b",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
        ],
    ),
    # GPT o1 Low
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_o1_low,
        friendly_name="GPT o1 - Low",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="o1",
                available_thinking_levels={"Low": "low"},
                default_thinking_level="low",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="o1",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels={"Low": "low"},
                default_thinking_level="low",
            ),
        ],
    ),
    # GPT o1 Medium
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_o1_medium,
        friendly_name="GPT o1 - Medium",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="o1",
                available_thinking_levels={"Medium": "medium"},
                default_thinking_level="medium",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="o1",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels={"Medium": "medium"},
                default_thinking_level="medium",
            ),
        ],
    ),
    # GPT o1 High
    KilnModel(
        family=ModelFamily.gpt,
        name=ModelName.gpt_o1_high,
        friendly_name="GPT o1 - High",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openai,
                model_id="o1",
                available_thinking_levels={"High": "high"},
                default_thinking_level="high",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.azure_openai,
                model_id="o1",
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels={"High": "high"},
                default_thinking_level="high",
            ),
        ],
    ),
    # Claude Fable 5.1
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_fable_5_1,
        friendly_name="Claude Fable 5.1",
        featured_rank=1,
        editorial_notes="Anthropic's most powerful publicly-available model (Mythos-class). State-of-the-art across software engineering, knowledge work, and vision.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="anthropic/claude-fable-5.1",
                # OpenRouter routes Claude Fable to Bedrock/Azure backends that ignore response_format json_schema while reasoning is on (returns prose); use prompt-injected JSON instructions instead.
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                openrouter_reasoning_object=True,
                available_thinking_levels=CLAUDE_FABLE_5_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="high",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="claude-fable-5-1",
                structured_output_mode=StructuredOutputMode.json_schema,
                temp_top_p_exclusive=True,
                available_thinking_levels=CLAUDE_FABLE_5_ANTHROPIC_THINKING_LEVELS,
                default_thinking_level="high",
                anthropic_summarized_thinking=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Claude Fable 5
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_fable_5,
        friendly_name="Claude Fable 5",
        editorial_notes="Anthropic's previous-generation Mythos-class model. Strong across software engineering, knowledge work, and vision.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="anthropic/claude-fable-5",
                structured_output_mode=StructuredOutputMode.json_schema,
                openrouter_reasoning_object=True,
                available_thinking_levels=CLAUDE_FABLE_5_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="high",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                model_id="claude-fable-5",
                structured_output_mode=StructuredOutputMode.json_schema,
                temp_top_p_exclusive=True,
                available_thinking_levels=CLAUDE_FABLE_5_ANTHROPIC_THINKING_LEVELS,
                default_thinking_level="high",
                anthropic_summarized_thinking=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Claude Opus 5
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_opus_5,
        friendly_name="Claude Opus 5",
        featured_rank=3,
        editorial_notes="Anthropic's best Claude model. Expensive, but often the best.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="anthropic/claude-opus-5",
                structured_output_mode=StructuredOutputMode.json_schema,
                openrouter_reasoning_object=True,
                available_thinking_levels=CLAUDE_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="none",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="claude-opus-5",
                structured_output_mode=StructuredOutputMode.json_schema,
                temp_top_p_exclusive=True,
                available_thinking_levels=CLAUDE_OPUS_5_ANTHROPIC_THINKING_LEVELS,
                default_thinking_level="high",
                anthropic_summarized_thinking=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Claude Opus 4.8
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_opus_4_8,
        friendly_name="Claude Opus 4.8",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="anthropic/claude-opus-4.8",
                structured_output_mode=StructuredOutputMode.json_schema,
                openrouter_reasoning_object=True,
                available_thinking_levels=CLAUDE_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="none",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                model_id="claude-opus-4-8",
                structured_output_mode=StructuredOutputMode.json_schema,
                temp_top_p_exclusive=True,
                available_thinking_levels=CLAUDE_OPUS_4_8_ANTHROPIC_THINKING_LEVELS,
                default_thinking_level="high",
                anthropic_summarized_thinking=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Claude Opus 4.7
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_opus_4_7,
        friendly_name="Claude Opus 4.7",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="anthropic/claude-opus-4.7",
                structured_output_mode=StructuredOutputMode.json_schema,
                openrouter_reasoning_object=True,
                available_thinking_levels=CLAUDE_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="none",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                model_id="claude-opus-4-7",
                structured_output_mode=StructuredOutputMode.json_schema,
                temp_top_p_exclusive=True,
                available_thinking_levels=CLAUDE_OPUS_4_7_ANTHROPIC_THINKING_LEVELS,
                default_thinking_level="high",
                anthropic_summarized_thinking=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Claude Opus 4.6
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_opus_4_6,
        friendly_name="Claude Opus 4.6",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="anthropic/claude-opus-4.6",
                structured_output_mode=StructuredOutputMode.json_schema,
                openrouter_reasoning_object=True,
                available_thinking_levels=CLAUDE_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="none",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                model_id="claude-opus-4-6",
                structured_output_mode=StructuredOutputMode.json_schema,
                temp_top_p_exclusive=True,
                available_thinking_levels=CLAUDE_ANTHROPIC_EFFORT_THINKING_LEVELS,
                default_thinking_level="high",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Claude Opus 4.5
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_opus_4_5,
        friendly_name="Claude Opus 4.5",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="anthropic/claude-opus-4.5",
                structured_output_mode=StructuredOutputMode.json_schema,
                openrouter_reasoning_object=True,
                available_thinking_levels=CLAUDE_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="none",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                model_id="claude-opus-4-5-20251101",
                structured_output_mode=StructuredOutputMode.json_schema,
                temp_top_p_exclusive=True,
                available_thinking_levels=CLAUDE_ANTHROPIC_EFFORT_THINKING_LEVELS,
                default_thinking_level="high",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Claude Opus 4.1
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_opus_4_1,
        friendly_name="Claude Opus 4.1",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="anthropic/claude-opus-4.1",
                structured_output_mode=StructuredOutputMode.function_calling,
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                model_id="claude-opus-4-1-20250805",
                structured_output_mode=StructuredOutputMode.function_calling,
                temp_top_p_exclusive=True,
            ),
        ],
    ),
    # Claude Opus 4
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_opus_4,
        friendly_name="Claude Opus 4",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="anthropic/claude-opus-4",
                structured_output_mode=StructuredOutputMode.function_calling,
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                model_id="claude-opus-4-20250514",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.function_calling,
            ),
        ],
    ),
    # Claude Sonnet 5
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_sonnet_5,
        friendly_name="Claude 5 Sonnet",
        featured_rank=10,
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                suggested_for_synthetic_user=True,
                model_id="anthropic/claude-sonnet-5",
                structured_output_mode=StructuredOutputMode.json_schema,
                openrouter_reasoning_object=True,
                available_thinking_levels=CLAUDE_SONNET_5_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="none",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                suggested_for_synthetic_user=True,
                model_id="claude-sonnet-5",
                structured_output_mode=StructuredOutputMode.json_schema,
                temp_top_p_exclusive=True,
                available_thinking_levels=CLAUDE_SONNET_5_ANTHROPIC_THINKING_LEVELS,
                default_thinking_level="high",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Claude Sonnet 4.6
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_sonnet_4_6,
        friendly_name="Claude 4.6 Sonnet",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="anthropic/claude-sonnet-4.6",
                structured_output_mode=StructuredOutputMode.json_schema,
                openrouter_reasoning_object=True,
                available_thinking_levels=CLAUDE_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="none",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                model_id="claude-sonnet-4-6",
                structured_output_mode=StructuredOutputMode.json_schema,
                temp_top_p_exclusive=True,
                available_thinking_levels=CLAUDE_ANTHROPIC_EFFORT_THINKING_LEVELS,
                default_thinking_level="high",
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Claude Sonnet 4.5
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_sonnet_4_5,
        friendly_name="Claude 4.5 Sonnet",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="anthropic/claude-4.5-sonnet",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.function_calling,
                openrouter_reasoning_object=True,
                available_thinking_levels=CLAUDE_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="none",
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                model_id="claude-sonnet-4-5-20250929",
                structured_output_mode=StructuredOutputMode.json_schema,
                temp_top_p_exclusive=True,
                available_thinking_levels=CLAUDE_ANTHROPIC_EFFORT_THINKING_LEVELS,
                default_thinking_level="high",
            ),
        ],
    ),
    # Claude Sonnet 4
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_sonnet_4,
        friendly_name="Claude 4 Sonnet",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="anthropic/claude-sonnet-4",
                structured_output_mode=StructuredOutputMode.function_calling,
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                model_id="claude-sonnet-4-20250514",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.function_calling,
            ),
        ],
    ),
    # Claude 3.7 Sonnet
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_3_7_sonnet,
        friendly_name="Claude 3.7 Sonnet",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                structured_output_mode=StructuredOutputMode.function_calling,
                model_id="anthropic/claude-3.7-sonnet",
                deprecated=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                model_id="claude-3-7-sonnet-20250219",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.function_calling,
            ),
        ],
    ),
    # Claude 3.7 Sonnet Thinking
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_3_7_sonnet_thinking,
        friendly_name="Claude 3.7 Sonnet Thinking",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="anthropic/claude-3.7-sonnet:thinking",
                deprecated=True,
                reasoning_capable=True,
                # For reasoning models, we need to use json_instructions with OpenRouter
                structured_output_mode=StructuredOutputMode.json_instructions,
                require_openrouter_reasoning=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                reasoning_capable=True,
                model_id="claude-3-7-sonnet-20250219",
                deprecated=True,
                anthropic_extended_thinking=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
        ],
    ),
    # Claude 3.5 Sonnet
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_3_5_sonnet,
        friendly_name="Claude 3.5 Sonnet",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                structured_output_mode=StructuredOutputMode.function_calling,
                model_id="anthropic/claude-3.5-sonnet",
                deprecated=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                model_id="claude-3-5-sonnet-20241022",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.function_calling,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="claude-3-5-sonnet",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.function_calling_weak,
            ),
        ],
    ),
    # Claude 4.5 Haiku
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_4_5_haiku,
        friendly_name="Claude 4.5 Haiku",
        editorial_notes="Claude on a budget. 20% the cost of Claude Opus. Great for easier tasks.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                suggested_for_synthetic_user=True,
                model_id="anthropic/claude-haiku-4.5",
                structured_output_mode=StructuredOutputMode.function_calling,
                openrouter_reasoning_object=True,
                available_thinking_levels=CLAUDE_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="none",
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                suggested_for_synthetic_user=True,
                model_id="claude-haiku-4-5-20251001",
                structured_output_mode=StructuredOutputMode.json_schema,
                temp_top_p_exclusive=True,
                available_thinking_levels=CLAUDE_ANTHROPIC_EFFORT_THINKING_LEVELS,
                default_thinking_level="high",
            ),
        ],
    ),
    # Claude 3.5 Haiku
    KilnModel(
        family=ModelFamily.claude,
        name=ModelName.claude_3_5_haiku,
        friendly_name="Claude 3.5 Haiku",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                structured_output_mode=StructuredOutputMode.function_calling,
                model_id="anthropic/claude-3-5-haiku",
                deprecated=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.anthropic,
                model_id="claude-3-5-haiku-20241022",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.function_calling,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="claude-3-5-haiku",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.function_calling_weak,
            ),
        ],
    ),
    # Gemini 3.8 Flash
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_3_8_flash,
        friendly_name="Gemini 3.8 Flash",
        featured_rank=12,
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                suggested_for_synthetic_user=True,
                model_id="google/gemini-3.8-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                # 3.8 Flash exposes low/medium/high thinking levels (default medium)
                available_thinking_levels=GEMINI_3_PRO_THINKING_LEVELS,
                default_thinking_level="medium",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                suggested_for_synthetic_user=True,
                model_id="gemini-3.8-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_PRO_THINKING_LEVELS,
                default_thinking_level="medium",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                suggested_for_synthetic_user=True,
                model_id="gemini-3.8-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_PRO_THINKING_LEVELS,
                default_thinking_level="medium",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    # Unlike Gemini 3.7 Flash on Vertex, this model's compressed-audio
                    # transcriptions are not blocked by Vertex's content filter: MP3 and
                    # OGG extraction both passed 10/10 runs, so all audio types are enabled.
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                gemini_reasoning_enabled=True,
            ),
        ],
    ),
    # Gemini 3.7 Flash
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_3_7_flash,
        friendly_name="Gemini 3.7 Flash",
        editorial_notes="Google's latest Flash model. Stronger agentic and multimodal performance at a lower cost than Gemini 3.6 Flash.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-3.7-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                # 3.7 Flash exposes low/medium/high thinking levels (default medium)
                available_thinking_levels=GEMINI_3_PRO_THINKING_LEVELS,
                default_thinking_level="medium",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-3.7-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_PRO_THINKING_LEVELS,
                default_thinking_level="medium",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-3.7-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_PRO_THINKING_LEVELS,
                default_thinking_level="medium",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    # MP3 and OGG excluded: Vertex's content filter stochastically blocks
                    # this model's transcriptions of compressed audio (finish_reason
                    # content_filter; mp3 ~90%, ogg ~50% of requests). WAV is unaffected,
                    # as are all audio types on gemini_api.
                    KilnMimeType.WAV,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                gemini_reasoning_enabled=True,
            ),
        ],
    ),
    # Gemini 3.6 Flash
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_3_6_flash,
        friendly_name="Gemini 3.6 Flash",
        editorial_notes="Google's latest Flash model. Stronger agentic and multimodal performance at a lower cost than Gemini 3.5 Flash.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-3.6-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-3.6-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-3.6-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                gemini_reasoning_enabled=True,
            ),
        ],
    ),
    # Gemini 3.5 Flash
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_3_5_flash,
        friendly_name="Gemini 3.5 Flash",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-3.5-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-3.5-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-3.5-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                gemini_reasoning_enabled=True,
            ),
        ],
    ),
    # Gemini 3.5 Flash Lite
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_3_5_flash_lite,
        friendly_name="Gemini 3.5 Flash Lite",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-3.5-flash-lite",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                suggested_for_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-3.5-flash-lite",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                suggested_for_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-3.5-flash-lite",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                suggested_for_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                gemini_reasoning_enabled=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
            ),
        ],
    ),
    # Gemini 3.1 Pro Preview
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_3_1_pro_preview,
        friendly_name="Gemini 3.1 Pro Preview",
        editorial_notes="Google's state-of-the-art model. Great for tough problems.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-3.1-pro-preview",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                suggested_for_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_PRO_THINKING_LEVELS,
                default_thinking_level="high",
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-3.1-pro-preview",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                suggested_for_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_PRO_THINKING_LEVELS,
                default_thinking_level="high",
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                gemini_reasoning_enabled=True,
                max_parallel_requests=2,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-3.1-pro-preview",
                structured_output_mode=StructuredOutputMode.json_schema,
                gemini_reasoning_enabled=True,
                available_thinking_levels=GEMINI_3_PRO_THINKING_LEVELS,
                default_thinking_level="high",
            ),
        ],
    ),
    # Gemini 3.1 Flash Lite
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_3_1_flash_lite,
        friendly_name="Gemini 3.1 Flash Lite",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-3.1-flash-lite",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                gemini_reasoning_enabled=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-3.1-flash-lite",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                gemini_reasoning_enabled=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-3.1-flash-lite",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                gemini_reasoning_enabled=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
            ),
        ],
    ),
    # Gemini 3.1 Flash Lite Preview
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_3_1_flash_lite_preview,
        friendly_name="Gemini 3.1 Flash Lite Preview",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-3.1-flash-lite-preview",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                gemini_reasoning_enabled=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-3.1-flash-lite-preview",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                gemini_reasoning_enabled=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-3.1-flash-lite-preview",
                structured_output_mode=StructuredOutputMode.json_schema,
                gemini_reasoning_enabled=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
            ),
        ],
    ),
    # Gemini 3 Pro Preview
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_3_pro_preview,
        friendly_name="Gemini 3 Pro Preview",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-3-pro-preview",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_PRO_THINKING_LEVELS,
                default_thinking_level="high",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-3-pro-preview",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_PRO_THINKING_LEVELS,
                default_thinking_level="high",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                gemini_reasoning_enabled=True,
                max_parallel_requests=2,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-3-pro-preview",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                gemini_reasoning_enabled=True,
                available_thinking_levels=GEMINI_3_PRO_THINKING_LEVELS,
                default_thinking_level="high",
            ),
        ],
    ),
    # Gemini 3 Flash
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_3_flash,
        friendly_name="Gemini 3 Flash",
        editorial_notes="Google's faster and cheaper model. 25% the cost of Gemini 3 Pro.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-3-flash-preview",
                structured_output_mode=StructuredOutputMode.json_schema,
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-3-flash-preview",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-3-flash-preview",
                structured_output_mode=StructuredOutputMode.json_schema,
                # while the model is capable of reasoning, it doesn't always return it in the response
                # reasoning_capable=True,
                gemini_reasoning_enabled=True,
                available_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                default_thinking_level="high",
            ),
        ],
    ),
    # Gemini 2.5 Pro
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_2_5_pro,
        friendly_name="Gemini 2.5 Pro",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-2.5-pro",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-2.5-pro",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                reasoning_capable=True,
                gemini_reasoning_enabled=True,
                max_parallel_requests=2,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-2.5-pro",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                gemini_reasoning_enabled=True,
            ),
        ],
    ),
    # Gemini 2.5 Flash
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_2_5_flash,
        friendly_name="Gemini 2.5 Flash",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-2.5-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_doc_extraction=True,
                suggested_for_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-2.5-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_doc_extraction=True,
                suggested_for_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-2.5-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
            ),
        ],
    ),
    # Gemini 2.5 Flash Lite
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_2_5_flash_lite,
        friendly_name="Gemini 2.5 Flash Lite",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-2.5-flash-lite",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_doc_extraction=True,
                suggested_for_doc_extraction=False,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                gemini_reasoning_enabled=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-2.5-flash-lite",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_doc_extraction=True,
                suggested_for_doc_extraction=False,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-2.5-flash-lite",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
            ),
        ],
    ),
    # Gemini 2.0 Flash
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_2_0_flash,
        friendly_name="Gemini 2.0 Flash",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-2.0-flash-001",
                deprecated=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-2.0-flash",
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-2.0-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                provider_finetune_id="gemini-2.0-flash-001",
            ),
        ],
    ),
    # Gemini 2.0 Flash Lite
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_2_0_flash_lite,
        friendly_name="Gemini 2.0 Flash Lite",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-2.0-flash-lite-001",
                deprecated=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-2.0-flash-lite",
                supports_doc_extraction=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # audio
                    KilnMimeType.MP3,
                    KilnMimeType.WAV,
                    KilnMimeType.OGG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-2.0-flash-lite",
                structured_output_mode=StructuredOutputMode.json_schema,
                provider_finetune_id="gemini-2.0-flash-lite-001",
            ),
        ],
    ),
    # Gemini 1.5 Pro
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_1_5_pro,
        friendly_name="Gemini 1.5 Pro",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-pro-1.5",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-1.5-pro",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-1.5-pro",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
            ),
        ],
    ),
    # Gemini 1.5 Flash
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_1_5_flash,
        friendly_name="Gemini 1.5 Flash",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-flash-1.5",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-1.5-flash",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="gemini-1.5-flash",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
            ),
        ],
    ),
    # Gemini 1.5 Flash 8B
    KilnModel(
        family=ModelFamily.gemini,
        name=ModelName.gemini_1_5_flash_8b,
        friendly_name="Gemini 1.5 Flash 8B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemini-flash-1.5-8b",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemini-1.5-flash-8b",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
            ),
        ],
    ),
    # Nemotron 3 Ultra
    KilnModel(
        family=ModelFamily.nemotron,
        name=ModelName.nemotron_3_ultra,
        friendly_name="Nemotron 3 Ultra",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="nvidia/nemotron-3-ultra-550b-a55b",
                structured_output_mode=StructuredOutputMode.json_schema,
                require_openrouter_reasoning=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/nemotron-3-ultra-nvfp4",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="nvidia/nemotron-3-ultra-550b-a55b",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
            ),
        ],
    ),
    # Nemotron 3 Super
    KilnModel(
        family=ModelFamily.nemotron,
        name=ModelName.nemotron_3_super,
        friendly_name="Nemotron 3 Super",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="nvidia/nemotron-3-super-120b-a12b",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                require_openrouter_reasoning=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="nemotron-3-super",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
            ),
        ],
    ),
    # Nemotron 3 Nano
    KilnModel(
        family=ModelFamily.nemotron,
        name=ModelName.nemotron_3_nano,
        friendly_name="Nemotron 3 Nano",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="nvidia/nemotron-3-nano-30b-a3b:free",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="nemotron-3-nano",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                ollama_model_aliases=["nemotron-3-nano:30b"],
            ),
        ],
    ),
    # Nemotron 70B
    KilnModel(
        family=ModelFamily.llama,
        name=ModelName.nemotron_70b,
        friendly_name="Nemotron 70B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                supports_structured_output=False,
                supports_data_gen=False,
                model_id="nvidia/llama-3.1-nemotron-70b-instruct",
                deprecated=True,
                supports_function_calling=False,
            ),
        ],
    ),
    # Llama 4 Maverick Basic
    KilnModel(
        family=ModelFamily.llama,
        name=ModelName.llama_4_maverick,
        friendly_name="Llama 4 Maverick",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="meta-llama/llama-4-maverick",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/llama4-maverick-instruct-basic",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.cerebras,
                model_id="llama-4-maverick-17b-128e-instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="llama4:128x17b",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Llama 4 Scout Basic
    KilnModel(
        family=ModelFamily.llama,
        name=ModelName.llama_4_scout,
        friendly_name="Llama 4 Scout",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="meta-llama/llama-4-scout",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/llama4-scout-instruct-basic",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.cerebras,
                model_id="llama-4-scout-17b-16e-instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="llama4:16x17b",
                ollama_model_aliases=["llama4"],
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Llama 3.3 70B
    KilnModel(
        family=ModelFamily.llama,
        name=ModelName.llama_3_3_70b,
        friendly_name="Llama 3.3 70B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="meta-llama/llama-3.3-70b-instruct",
                structured_output_mode=StructuredOutputMode.json_schema,
                # Openrouter not working with json_schema or tools. JSON_schema sometimes works so force that, but not consistently so still not recommended.
                supports_structured_output=False,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.groq,
                supports_structured_output=True,
                supports_data_gen=True,
                model_id="llama-3.3-70b-versatile",
                deprecated=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                structured_output_mode=StructuredOutputMode.json_schema,
                model_id="llama3.3",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                # Tool calling forces schema -- fireworks doesn't support json_schema, just json_mode
                structured_output_mode=StructuredOutputMode.function_calling_weak,
                model_id="accounts/fireworks/models/llama-v3p3-70b-instruct",
                deprecated=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.vertex,
                model_id="meta/llama-3.3-70b-instruct-maas",
                # Doesn't work yet; needs debugging
                supports_structured_output=False,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                structured_output_mode=StructuredOutputMode.function_calling_weak,
                # Tools work. Probably constrained decode? Nice
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                structured_output_mode=StructuredOutputMode.json_schema,
                model_id="ai/llama3.3:70B-Q4_K_M",
                supports_function_calling=False,
            ),
            # Featherless provider omitted: meta-llama/Llama-3.3-70B-Instruct is
            # gated (HTTP 403 model_gated_needs_oauth), requiring each user to link
            # a HuggingFace org to their Featherless account. All 20 official
            # meta-llama repos on Featherless are gated, so no Llama variant works
            # here out of the box.
        ],
    ),
    # Llama 3.2 90B
    KilnModel(
        family=ModelFamily.llama,
        name=ModelName.llama_3_2_90b,
        friendly_name="Llama 3.2 90B (Vision)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                model_id="meta-llama/llama-3.2-90b-vision-instruct",
                deprecated=True,
                supports_function_calling=False,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                structured_output_mode=StructuredOutputMode.json_schema,
                model_id="llama3.2-vision:90b",
                ollama_model_aliases=["llama3.2-vision:90b"],
                supports_function_calling=False,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                # no longer available via serverless
                name=ModelProviderName.together_ai,
                model_id="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                deprecated=True,
                supports_structured_output=False,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # Llama 3.2 11B
    KilnModel(
        family=ModelFamily.llama,
        name=ModelName.llama_3_2_11b,
        friendly_name="Llama 3.2 11B (Vision)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                # Best mode, but fails to often to enable without warning
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_structured_output=False,
                supports_data_gen=False,
                model_id="meta-llama/llama-3.2-11b-vision-instruct",
                supports_function_calling=False,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                structured_output_mode=StructuredOutputMode.json_schema,
                model_id="llama3.2-vision",
                ollama_model_aliases=["llama3.2-vision:11b"],
                supports_function_calling=False,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                # no longer available via serverless
                name=ModelProviderName.together_ai,
                model_id="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                deprecated=True,
                supports_structured_output=False,
                supports_data_gen=False,
                supports_function_calling=False,  # weird 3b works and 11b doesn't but... vision?
            ),
        ],
    ),
    # Llama 3.2 3B
    KilnModel(
        family=ModelFamily.llama,
        name=ModelName.llama_3_2_3b,
        friendly_name="Llama 3.2 3B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                supports_structured_output=False,
                supports_data_gen=False,
                structured_output_mode=StructuredOutputMode.json_schema,
                model_id="meta-llama/llama-3.2-3b-instruct",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                supports_data_gen=False,
                model_id="llama3.2",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="meta-llama/Llama-3.2-3B-Instruct-Turbo",
                deprecated=True,
                supports_structured_output=False,
                supports_data_gen=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                model_id="ai/llama3.2:3B-Q4_K_M",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # Llama 3.2 1B
    KilnModel(
        family=ModelFamily.llama,
        name=ModelName.llama_3_2_1b,
        friendly_name="Llama 3.2 1B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                supports_structured_output=False,
                supports_data_gen=False,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                model_id="meta-llama/llama-3.2-1b-instruct",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                supports_structured_output=False,
                supports_data_gen=False,
                model_id="llama3.2:1b",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                supports_structured_output=False,
                supports_data_gen=False,
                model_id="ai/llama3.2:1B-F16",
                supports_function_calling=False,
            ),
        ],
    ),
    # Llama 3.1 405b
    KilnModel(
        family=ModelFamily.llama,
        name=ModelName.llama_3_1_405b,
        friendly_name="Llama 3.1 405B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.amazon_bedrock,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                model_id="meta.llama3-1-405b-instruct-v1:0",
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                structured_output_mode=StructuredOutputMode.json_schema,
                model_id="llama3.1:405b",
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                structured_output_mode=StructuredOutputMode.function_calling,
                model_id="meta-llama/llama-3.1-405b-instruct",
                deprecated=True,
                supports_function_calling=False,  # Not reliable
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                # No finetune support. https://docs.fireworks.ai/fine-tuning/fine-tuning-models
                structured_output_mode=StructuredOutputMode.function_calling_weak,
                model_id="accounts/fireworks/models/llama-v3p1-405b-instruct",
                deprecated=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                deprecated=True,
                supports_data_gen=False,
                structured_output_mode=StructuredOutputMode.function_calling_weak,
            ),
        ],
    ),
    # Llama 3.1 70b
    KilnModel(
        family=ModelFamily.llama,
        name=ModelName.llama_3_1_70b,
        friendly_name="Llama 3.1 70B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.amazon_bedrock,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                model_id="meta.llama3-1-70b-instruct-v1:0",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                supports_data_gen=False,
                structured_output_mode=StructuredOutputMode.json_schema,
                model_id="meta-llama/llama-3.1-70b-instruct",
                supports_logprobs=True,
                logprobs_openrouter_options=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                structured_output_mode=StructuredOutputMode.json_schema,
                model_id="llama3.1:70b",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                # Tool calling forces schema -- fireworks doesn't support json_schema, just json_mode
                structured_output_mode=StructuredOutputMode.function_calling_weak,
                model_id="accounts/fireworks/models/llama-v3p1-70b-instruct",
                deprecated=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                deprecated=True,
                supports_data_gen=False,
                structured_output_mode=StructuredOutputMode.function_calling_weak,
                provider_finetune_id="meta-llama/Meta-Llama-3.1-70B-Instruct-Reference",
            ),
        ],
    ),
    # Llama 3.1-8b
    KilnModel(
        family=ModelFamily.llama,
        name=ModelName.llama_3_1_8b,
        friendly_name="Llama 3.1 8B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.groq,
                model_id="llama-3.1-8b-instant",
                deprecated=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.amazon_bedrock,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_structured_output=False,
                model_id="meta.llama3-1-8b-instruct-v1:0",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                structured_output_mode=StructuredOutputMode.json_schema,
                model_id="llama3.1:8b",
                ollama_model_aliases=["llama3.1"],  # 8b is default
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                supports_data_gen=False,
                structured_output_mode=StructuredOutputMode.function_calling,
                model_id="meta-llama/llama-3.1-8b-instruct",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                # JSON mode not ideal (no schema), but tool calling doesn't work on 8b
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=False,
                model_id="accounts/fireworks/models/llama-v3p1-8b-instruct",
                deprecated=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                deprecated=True,
                supports_data_gen=False,
                structured_output_mode=StructuredOutputMode.function_calling_weak,
                provider_finetune_id="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
                # Constrained decode? They make function calling work when no one else does!
            ),
            KilnModelProvider(
                name=ModelProviderName.cerebras,
                model_id="llama3.1-8b",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.function_calling,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                structured_output_mode=StructuredOutputMode.json_schema,
                model_id="ai/llama3.1:8B-Q4_K_M",
                supports_function_calling=False,
            ),
        ],
    ),
    # Muse Spark 1.2
    KilnModel(
        family=ModelFamily.muse,
        name=ModelName.muse_spark_1_2,
        friendly_name="Muse Spark 1.2",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="meta/muse-spark-1.2",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents (Meta backend 400s on html/csv uploads, so excluded)
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Muse Spark 1.1
    KilnModel(
        family=ModelFamily.muse,
        name=ModelName.muse_spark_1_1,
        friendly_name="Muse Spark 1.1",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="meta/muse-spark-1.1",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.CSV,
                    KilnMimeType.TXT,
                    KilnMimeType.HTML,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Mistral Large 2512
    KilnModel(
        family=ModelFamily.mistral,
        name=ModelName.mistral_3_large_2512,
        friendly_name="Mistral Large 3 2512",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="mistralai/mistral-large-2512",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/mistral-large-3-fp8",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
        ],
    ),
    # Mistral Large
    KilnModel(
        family=ModelFamily.mistral,
        name=ModelName.mistral_large,
        friendly_name="Mistral Large",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.amazon_bedrock,
                structured_output_mode=StructuredOutputMode.json_instructions,
                model_id="mistral.mistral-large-2407-v1:0",
                uncensored=True,
                suggested_for_uncensored_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                structured_output_mode=StructuredOutputMode.json_schema,
                model_id="mistralai/mistral-large",
                uncensored=True,
                suggested_for_uncensored_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                structured_output_mode=StructuredOutputMode.json_schema,
                model_id="mistral-large",
                uncensored=True,
                suggested_for_uncensored_data_gen=True,
            ),
        ],
    ),
    # Mistral Medium 3.5
    KilnModel(
        family=ModelFamily.mistral,
        name=ModelName.mistral_medium_3_5,
        friendly_name="Mistral Medium 3.5",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="mistralai/mistral-medium-3-5",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
        ],
    ),
    # Mistral Medium 3.1
    KilnModel(
        family=ModelFamily.mistral,
        name=ModelName.mistral_medium_3_1,
        friendly_name="Mistral Medium 3.1",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="mistralai/mistral-medium-3.1",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
        ],
    ),
    # Magistral Medium (Thinking)
    KilnModel(
        family=ModelFamily.mistral,
        name=ModelName.magistral_medium_thinking,
        friendly_name="Magistral Medium (Thinking)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="mistralai/magistral-medium-2506:thinking",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                # Thinking tokens are hidden by Mistral so not "reasoning" from Kiln API POV
            ),
        ],
    ),
    # Magistral Medium (No Thinking)
    KilnModel(
        family=ModelFamily.mistral,
        name=ModelName.magistral_medium,
        friendly_name="Magistral Medium (No Thinking)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="mistralai/magistral-medium-2506",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
        ],
    ),
    # Mistral Small Creative
    KilnModel(
        family=ModelFamily.mistral,
        name=ModelName.mistral_small_creative,
        friendly_name="Mistral Small Creative",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="mistralai/mistral-small-creative",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
            ),
        ],
    ),
    # Mistral Small 4
    KilnModel(
        family=ModelFamily.mistral,
        name=ModelName.mistral_small_4,
        friendly_name="Mistral Small 4",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="mistralai/mistral-small-2603",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Mistral Small 3
    KilnModel(
        family=ModelFamily.mistral,
        name=ModelName.mistral_small_3,
        friendly_name="Mistral Small 3",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                model_id="mistralai/mistral-small-24b-instruct-2501",
                uncensored=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="mistral-small:24b",
                uncensored=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="mistralai/Mistral-Small-24B-Instruct-2501",
                structured_output_mode=StructuredOutputMode.json_instructions,
                uncensored=True,
                supports_function_calling=False,
            ),
        ],
    ),
    # Ministral 3 14B 2512
    KilnModel(
        family=ModelFamily.mistral,
        name=ModelName.ministral_3_14b_2512,
        friendly_name="Ministral 3 14B 2512",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="mistralai/ministral-14b-2512",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="mistralai/Ministral-3-14B-Instruct-2512",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_function_calling=False,
            ),
        ],
    ),
    # Ministral 3 8B 2512
    KilnModel(
        family=ModelFamily.mistral,
        name=ModelName.ministral_3_8b_2512,
        friendly_name="Ministral 3 8B 2512",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="mistralai/ministral-8b-2512",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
        ],
    ),
    # Ministral 3 3B 2512
    KilnModel(
        family=ModelFamily.mistral,
        name=ModelName.ministral_3_3b_2512,
        friendly_name="Ministral 3 3B 2512",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="mistralai/ministral-3b-2512",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
        ],
    ),
    # Mistral Nemo
    KilnModel(
        family=ModelFamily.mistral,
        name=ModelName.mistral_nemo,
        friendly_name="Mistral Nemo",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="mistralai/mistral-nemo",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_function_calling=False,  # Not reliable
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                model_id="ai/mistral-nemo:12B-Q4_K_M",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
        ],
    ),
    # Phi 4
    KilnModel(
        family=ModelFamily.phi,
        name=ModelName.phi_4,
        friendly_name="Phi 4 - 14B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                structured_output_mode=StructuredOutputMode.json_schema,
                model_id="phi4",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                # JSON mode not consistent enough to enable in UI
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=False,
                model_id="microsoft/phi-4",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                structured_output_mode=StructuredOutputMode.json_schema,
                model_id="ai/phi4:14B-Q4_K_M",
                supports_function_calling=False,
            ),
            # Featherless provider omitted: phi-4 wraps its JSON in prose and
            # markdown fences, and Featherless can't constrain it -- LiteLLM's
            # featherless_ai provider rejects response_format entirely, so the
            # json_instruction_and_object / json_schema modes aren't available.
            # llm_as_judge and structured-output COT fail persistently as a result.
        ],
    ),
    # Phi 4 5.6B
    KilnModel(
        family=ModelFamily.phi,
        name=ModelName.phi_4_5p6b,
        friendly_name="Phi 4 - 5.6B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="microsoft/phi-4-multimodal-instruct",
                deprecated=True,
                supports_structured_output=False,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # Phi 4 Mini
    KilnModel(
        family=ModelFamily.phi,
        name=ModelName.phi_4_mini,
        friendly_name="Phi 4 Mini - 3.8B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="phi4-mini",
                supports_function_calling=False,
            ),
        ],
    ),
    # Phi 3.5
    KilnModel(
        family=ModelFamily.phi,
        name=ModelName.phi_3_5,
        friendly_name="Phi 3.5",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_structured_output=False,
                supports_data_gen=False,
                model_id="phi3.5",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                supports_structured_output=False,
                supports_data_gen=False,
                model_id="microsoft/phi-3.5-mini-128k-instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
            ),
        ],
    ),
    # Gemma 4 31B
    KilnModel(
        family=ModelFamily.gemma,
        name=ModelName.gemma_4_31b,
        friendly_name="Gemma 4 31B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="gemma4:31b",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemma-4-31b-it",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                openrouter_reasoning_object=True,
                available_thinking_levels=GEMMA_4_THINKING_LEVELS,
                default_thinking_level="none",
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemma-4-31b-it",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                max_parallel_requests=2,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="google/gemma-4-31B-it",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
            ),
        ],
    ),
    # Gemma 4 27B (26B A4B MoE)
    KilnModel(
        family=ModelFamily.gemma,
        name=ModelName.gemma_4_27b,
        friendly_name="Gemma 4 27B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="gemma4:26b",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemma-4-26b-a4b-it",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                openrouter_reasoning_object=True,
                available_thinking_levels=GEMMA_4_THINKING_LEVELS,
                default_thinking_level="none",
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemma-4-26b-a4b-it",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                max_parallel_requests=2,
            ),
        ],
    ),
    # Gemma 4 E4B
    KilnModel(
        family=ModelFamily.gemma,
        name=ModelName.gemma_4_e4b,
        friendly_name="Gemma 4 4B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="gemma4:e4b",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # Gemma 4 E2B
    KilnModel(
        family=ModelFamily.gemma,
        name=ModelName.gemma_4_e2b,
        friendly_name="Gemma 4 2B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="gemma4:e2b",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # Gemma 3n 4B
    KilnModel(
        family=ModelFamily.gemma,
        name=ModelName.gemma_3n_4b,
        friendly_name="Gemma 3n 4B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="google/gemma-3n-e4b-it",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="gemma3n:e4b",
                supports_data_gen=False,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemma-3n-e4b-it",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                model_id="ai/gemma3n:4B-Q4_K_M",
                supports_data_gen=False,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
            ),
        ],
    ),
    # Gemma 3n 2B
    KilnModel(
        family=ModelFamily.gemma,
        name=ModelName.gemma_3n_2b,
        friendly_name="Gemma 3n 2B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="gemma3n:e2b",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.gemini_api,
                model_id="gemma-3n-e2b-it",
                deprecated=True,
                supports_structured_output=False,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # Gemma 3 27B
    KilnModel(
        family=ModelFamily.gemma,
        name=ModelName.gemma_3_27b,
        friendly_name="Gemma 3 27B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="gemma3:27b",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                model_id="google/gemma-3-27b-it",
                supports_function_calling=False,
            ),
        ],
    ),
    # Gemma 3 12B
    KilnModel(
        family=ModelFamily.gemma,
        name=ModelName.gemma_3_12b,
        friendly_name="Gemma 3 12B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="gemma3:12b",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                model_id="google/gemma-3-12b-it",
                supports_function_calling=False,
            ),
        ],
    ),
    # Gemma 3 4B
    KilnModel(
        family=ModelFamily.gemma,
        name=ModelName.gemma_3_4b,
        friendly_name="Gemma 3 4B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="gemma3:4b",
                ollama_model_aliases=["gemma3"],
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                model_id="google/gemma-3-4b-it",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                model_id="ai/gemma3:4B-Q4_K_M",
                supports_function_calling=False,
            ),
        ],
    ),
    # Gemma 3 1B
    KilnModel(
        family=ModelFamily.gemma,
        name=ModelName.gemma_3_1b,
        friendly_name="Gemma 3 1B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="gemma3:1b",
                supports_structured_output=False,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                model_id="ai/gemma3:1B-F16",
                supports_structured_output=False,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # Gemma 3 270M
    KilnModel(
        family=ModelFamily.gemma,
        name=ModelName.gemma_3_0p27b,
        friendly_name="Gemma 3 270M",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                model_id="ai/gemma3:270M-F16",
                supports_structured_output=False,
                supports_data_gen=False,
                supports_function_calling=False,
            )
        ],
    ),
    # Gemma 2 27b
    KilnModel(
        family=ModelFamily.gemma,
        name=ModelName.gemma_2_27b,
        friendly_name="Gemma 2 27B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                supports_data_gen=False,
                model_id="gemma2:27b",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=False,
                model_id="google/gemma-2-27b-it",
                supports_function_calling=False,
            ),
        ],
    ),
    # Gemma 2 9b
    KilnModel(
        family=ModelFamily.gemma,
        name=ModelName.gemma_2_9b,
        friendly_name="Gemma 2 9B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                supports_data_gen=False,
                model_id="gemma2:9b",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                # Best mode, but fails to often to enable without warning
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_structured_output=False,
                supports_data_gen=False,
                model_id="google/gemma-2-9b-it",
                deprecated=True,
                supports_function_calling=False,
            ),
            # fireworks AI errors - not allowing system role. Exclude until resolved.
        ],
    ),
    # Gemma 2 2.6b
    KilnModel(
        family=ModelFamily.gemma,
        name=ModelName.gemma_2_2b,
        friendly_name="Gemma 2 2B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                supports_data_gen=False,
                model_id="gemma2:2b",
                supports_function_calling=False,
            ),
        ],
    ),
    # Mixtral 8x7B
    KilnModel(
        family=ModelFamily.mixtral,
        name=ModelName.mixtral_8x7b,
        friendly_name="Mixtral 8x7B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="mistralai/mixtral-8x7b-instruct",
                deprecated=True,
                supports_data_gen=False,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="mixtral",
                supports_function_calling=False,
            ),
        ],
    ),
    # DeepSeek V4.1 Flash
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_4_1_flash,
        friendly_name="DeepSeek V4.1 Flash",
        featured_rank=11,
        editorial_notes="Latest V4.1 Flash variant. 1M context, configurable reasoning, and adds native image input.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="deepseek/deepseek-v4.1-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                available_thinking_levels=DEEPSEEK_V4_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="high",
                openrouter_reasoning_object=True,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/deepseek-v4p1-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="deepseek-ai/DeepSeek-V4.1-Flash",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # DeepSeek V4 Pro
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_4_pro,
        friendly_name="DeepSeek V4 Pro",
        featured_rank=8,
        editorial_notes="Open source flagship with 1.6T params (49B activated). 1M context, configurable reasoning.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="deepseek/deepseek-v4-pro",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                available_thinking_levels=DEEPSEEK_V4_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="high",
                openrouter_reasoning_object=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="accounts/fireworks/models/deepseek-v4-pro",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="deepseek-ai/DeepSeek-V4-Pro",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="deepseek-ai/DeepSeek-V4-Pro",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
            ),
        ],
    ),
    # DeepSeek V4 Flash
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_4_flash,
        friendly_name="DeepSeek V4 Flash",
        editorial_notes="Faster V4 variant with 284B params (13B activated). 1M context, same reasoning capabilities.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                suggested_for_synthetic_user=True,
                model_id="deepseek/deepseek-v4-flash-0731",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                available_thinking_levels=DEEPSEEK_V4_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="high",
                openrouter_reasoning_object=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                suggested_for_synthetic_user=True,
                model_id="accounts/fireworks/models/deepseek-v4-flash-0731",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                suggested_for_synthetic_user=True,
                model_id="deepseek-ai/DeepSeek-V4-Flash-0731",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
            ),
        ],
    ),
    # DeepSeek 3.2
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_3_2,
        friendly_name="DeepSeek 3.2",
        editorial_notes="Open and powerful. A fraction of the cost of other large models.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="deepseek/deepseek-v3.2",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/deepseek-v3p2",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Pro/deepseek-ai/DeepSeek-V3.2",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="deepseek-ai/DeepSeek-V3.2",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
        ],
    ),
    # DeepSeek 3.1 Terminus
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_3_1_terminus,
        friendly_name="DeepSeek 3.1 Terminus",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="deepseek/deepseek-v3.1-terminus:exacto",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/deepseek-v3p1-terminus",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                # the model page states it supports function calling, but our test fails
                # for this particular provider
                supports_function_calling=False,
            ),
        ],
    ),
    # DeepSeek 3.1
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_3_1,
        friendly_name="DeepSeek 3.1",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="deepseek/deepseek-chat-v3.1",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/deepseek-v3p1",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Pro/deepseek-ai/DeepSeek-V3.1",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="deepseek-ai/DeepSeek-V3.1",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
        ],
    ),
    # DeepSeek R1 0528
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_r1_0528,
        friendly_name="DeepSeek R1 0528",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="deepseek/deepseek-r1-0528",
                parser=ModelParserID.r1_thinking,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                r1_openrouter_options=True,
                require_openrouter_reasoning=True,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/deepseek-r1-0528",
                deprecated=True,
                parser=ModelParserID.r1_thinking,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="deepseek-ai/DeepSeek-R1",  # Note: Together remapped the R1 endpoint to this 0528 model
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
                reasoning_capable=True,
                supports_data_gen=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Pro/deepseek-ai/DeepSeek-R1",
                parser=ModelParserID.optional_r1_thinking,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                supports_data_gen=True,
                supports_function_calling=False,
            ),
        ],
    ),
    # DeepSeek R1 0528 Distill Qwen 3 8B
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_r1_0528_distill_qwen3_8b,
        friendly_name="DeepSeek R1 0528 Distill Qwen 3 8B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="deepseek/deepseek-r1-0528-qwen3-8b",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                r1_openrouter_options=True,
                require_openrouter_reasoning=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # DeepSeek 3
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_3,
        friendly_name="DeepSeek V3",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="deepseek/deepseek-chat",
                structured_output_mode=StructuredOutputMode.function_calling,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/deepseek-v3",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_structured_output=True,
                supports_data_gen=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="deepseek-ai/DeepSeek-V3",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Pro/deepseek-ai/DeepSeek-V3",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
        ],
    ),
    # DeepSeek R1
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_r1,
        friendly_name="DeepSeek R1 (Original)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="deepseek/deepseek-r1",
                parser=ModelParserID.r1_thinking,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                r1_openrouter_options=True,
                require_openrouter_reasoning=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/deepseek-r1",
                deprecated=True,
                parser=ModelParserID.r1_thinking,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                # I want your RAM
                name=ModelProviderName.ollama,
                model_id="deepseek-r1:671b",
                parser=ModelParserID.r1_thinking,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
        ],
    ),
    # DeepSeek R1 Distill Qwen 32B
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_r1_distill_qwen_32b,
        friendly_name="DeepSeek R1 Distill Qwen 32B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                model_id="deepseek/deepseek-r1-distill-qwen-32b",
                deprecated=True,
                r1_openrouter_options=True,
                parser=ModelParserID.r1_thinking,
                require_openrouter_reasoning=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                parser=ModelParserID.r1_thinking,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                model_id="deepseek-r1:32b",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_function_calling=False,
                reasoning_optional_for_structured_output=True,
            ),
        ],
    ),
    # DeepSeek R1 Distill Llama 70B
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_r1_distill_llama_70b,
        friendly_name="DeepSeek R1 Distill Llama 70B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                model_id="deepseek/deepseek-r1-distill-llama-70b",
                r1_openrouter_options=True,
                require_openrouter_reasoning=True,
                parser=ModelParserID.r1_thinking,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                supports_data_gen=False,
                parser=ModelParserID.r1_thinking,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                model_id="deepseek-r1:70b",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                supports_data_gen=False,
                parser=ModelParserID.r1_thinking,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                model_id="ai/deepseek-r1-distill-llama:70B-Q4_K_M",
                supports_function_calling=False,
            ),
        ],
    ),
    # DeepSeek R1 Distill Qwen 14B
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_r1_distill_qwen_14b,
        friendly_name="DeepSeek R1 Distill Qwen 14B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                supports_data_gen=False,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                model_id="deepseek/deepseek-r1-distill-qwen-14b",
                deprecated=True,
                r1_openrouter_options=True,
                require_openrouter_reasoning=True,
                openrouter_skip_required_parameters=True,
                parser=ModelParserID.r1_thinking,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                supports_data_gen=False,
                parser=ModelParserID.r1_thinking,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                model_id="deepseek-r1:14b",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # DeepSeek R1 Distill Llama 8B
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_r1_distill_llama_8b,
        friendly_name="DeepSeek R1 Distill Llama 8B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                supports_structured_output=False,
                supports_data_gen=False,
                reasoning_capable=True,
                # Best mode, but fails to often to enable without warning
                structured_output_mode=StructuredOutputMode.json_instructions,
                model_id="deepseek/deepseek-r1-distill-llama-8b",
                deprecated=True,
                r1_openrouter_options=True,
                require_openrouter_reasoning=True,
                openrouter_skip_required_parameters=True,
                parser=ModelParserID.r1_thinking,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                supports_structured_output=False,
                supports_data_gen=False,
                parser=ModelParserID.r1_thinking,
                reasoning_capable=True,
                # Best mode, but fails to often to enable without warning
                structured_output_mode=StructuredOutputMode.json_instructions,
                model_id="deepseek-r1:8b",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                supports_structured_output=False,
                supports_data_gen=False,
                parser=ModelParserID.r1_thinking,
                reasoning_capable=True,
                # Best mode, but fails to often to enable without warning
                structured_output_mode=StructuredOutputMode.json_instructions,
                model_id="ai/deepseek-r1-distill-llama:8B-Q4_K_M",
                supports_function_calling=False,
            ),
        ],
    ),
    # DeepSeek R1 Distill Qwen 7B
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_r1_distill_qwen_7b,
        friendly_name="DeepSeek R1 Distill Qwen 7B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                # Best mode, but fails to often to enable without warning
                supports_structured_output=False,
                supports_data_gen=False,
                model_id="deepseek/deepseek-r1-distill-qwen-7b",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                r1_openrouter_options=True,
                require_openrouter_reasoning=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                # Best mode, but fails to often to enable without warning
                supports_structured_output=False,
                supports_data_gen=False,
                parser=ModelParserID.r1_thinking,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                model_id="deepseek-r1:7b",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                # Best mode, but fails to often to enable without warning
                supports_structured_output=False,
                supports_data_gen=False,
                model_id="Pro/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
                supports_function_calling=False,
            ),
        ],
    ),
    # DeepSeek R1 Distill Qwen 1.5B
    KilnModel(
        family=ModelFamily.deepseek,
        name=ModelName.deepseek_r1_distill_qwen_1p5b,
        friendly_name="DeepSeek R1 Distill Qwen 1.5B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                supports_structured_output=False,
                supports_data_gen=False,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                model_id="deepseek/deepseek-r1-distill-qwen-1.5b",
                deprecated=True,
                r1_openrouter_options=True,
                require_openrouter_reasoning=True,
                openrouter_skip_required_parameters=True,
                parser=ModelParserID.r1_thinking,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                supports_structured_output=False,
                supports_data_gen=False,
                parser=ModelParserID.r1_thinking,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                model_id="deepseek-r1:1.5b",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
                supports_structured_output=False,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # Dolphin 2.9 Mixtral 8x22B
    KilnModel(
        family=ModelFamily.dolphin,
        name=ModelName.dolphin_2_9_8x22b,
        friendly_name="Dolphin 2.9 8x22B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                model_id="dolphin-mixtral:8x22b",
                uncensored=True,
                suggested_for_uncensored_data_gen=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                model_id="cognitivecomputations/dolphin-mixtral-8x22b",
                deprecated=True,
                uncensored=True,
                suggested_for_uncensored_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # Grok 4.5
    KilnModel(
        family=ModelFamily.grok,
        name=ModelName.grok_4_5,
        friendly_name="Grok 4.5",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="x-ai/grok-4.5",
                supports_structured_output=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GROK_4_5_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="low",
                openrouter_reasoning_object=True,
                uncensored=True,
                multimodal_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Grok 4.3
    KilnModel(
        family=ModelFamily.grok,
        name=ModelName.grok_4_3,
        friendly_name="Grok 4.3",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="x-ai/grok-4.3",
                supports_structured_output=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=GROK_4_3_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="low",
                openrouter_reasoning_object=True,
                uncensored=True,
                multimodal_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Grok 4.20
    KilnModel(
        family=ModelFamily.grok,
        name=ModelName.grok_4_20,
        friendly_name="Grok 4.20",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="x-ai/grok-4.20",
                supports_structured_output=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                uncensored=True,
                multimodal_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Grok 4.1 Fast
    KilnModel(
        family=ModelFamily.grok,
        name=ModelName.grok_4_1_fast,
        friendly_name="Grok 4.1 Fast",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="x-ai/grok-4.1-fast",
                deprecated=True,
                supports_structured_output=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                uncensored=True,
                multimodal_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Grok 4
    KilnModel(
        family=ModelFamily.grok,
        name=ModelName.grok_4,
        friendly_name="Grok 4",
        editorial_notes="xAI's flagship model. Less censorship and unfiltered by design.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="x-ai/grok-4",
                deprecated=True,
                supports_structured_output=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                uncensored=True,
                suggested_for_uncensored_data_gen=False,
            ),
        ],
    ),
    # Grok 3
    KilnModel(
        family=ModelFamily.grok,
        name=ModelName.grok_3,
        friendly_name="Grok 3",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="x-ai/grok-3",
                deprecated=True,
                supports_structured_output=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                uncensored=True,
            ),
        ],
    ),
    # Grok 3 Mini
    KilnModel(
        family=ModelFamily.grok,
        name=ModelName.grok_3_mini,
        friendly_name="Grok 3 Mini",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="x-ai/grok-3-mini",
                deprecated=True,
                supports_structured_output=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                uncensored=True,
            ),
        ],
    ),
    # Grok 2
    KilnModel(
        family=ModelFamily.grok,
        name=ModelName.grok_2,
        friendly_name="Grok 2",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="x-ai/grok-2-1212",
                deprecated=True,
                supports_structured_output=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
        ],
    ),
    # Qwen 3.8 Max
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p8_max,
        friendly_name="Qwen 3.8 Max",
        featured_rank=6,
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="qwen/qwen3.8-max",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
                supports_vision=True,
                multimodal_capable=True,
                supports_doc_extraction=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
            ),
            # Fireworks does not host a distinct Qwen 3.8 Max: the slug
            # accounts/fireworks/models/qwen3p8-max is an alias whose catalog
            # page is titled "Qwen3.8-2.4T-A95B" (canonical id
            # accounts/fireworks/models/qwen3p8-2p4t-a95b), i.e. it resolves to
            # the Qwen 3.8 2.4T A95B model, not Max. Omitted to avoid misrouting.
            # Together AI does not host Qwen3.8-Max: the API returns HTTP 404
            # "model_not_available" (unlike Qwen 3.7 Max, which Together hosts
            # streaming-only). Omitted until Together adds it.
        ],
    ),
    # Qwen 3.8 2.4T A95B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p8_2p4t_a95b,
        friendly_name="Qwen 3.8 2.4T A95B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.8-2.4t-a95b",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/qwen3p8-2p4t-a95b",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                supports_function_calling=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="Qwen/Qwen3.8-2.4T-A95B",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
                supports_function_calling=True,
            ),
        ],
    ),
    # Qwen 3.8 Flash
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p8_flash,
        friendly_name="Qwen 3.8 Flash",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.8-flash",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            # Together AI hosts Qwen/Qwen3.8-Flash but only in streaming-only mode
            # (returns HTTP 400 "This model only supports streaming", code
            # streaming_required), which Kiln's non-streaming adapter can't use.
            # Omitted until Together supports non-streaming.
        ],
    ),
    # Qwen 3.8 27B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p8_27b,
        friendly_name="Qwen 3.8 27B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.8-27b",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="Qwen/Qwen3.8-27B",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
        ],
    ),
    # Qwen 3.7 Max
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p7_max,
        friendly_name="Qwen 3.7 Max",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.7-max",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
            ),
            # Fireworks lists qwen3p7-max on its site but the API returns 404 (not deployed).
            # Together AI hosts Qwen/Qwen3.7-Max but only in streaming-only mode
            # (returns HTTP 400 "This model only supports streaming"), which Kiln's
            # non-streaming adapter can't use. Omitted until Together supports non-streaming.
        ],
    ),
    # Qwen 3.7 Plus
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p7_plus,
        friendly_name="Qwen 3.7 Plus",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.7-plus",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/qwen3p7-plus",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                supports_function_calling=True,
            ),
        ],
    ),
    # Qwen 3.7 Flash
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p7_flash,
        friendly_name="Qwen 3.7 Flash",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.7-flash",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3.6 Plus
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p6_plus,
        friendly_name="Qwen 3.6 Plus",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.6-plus",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/qwen3p6-plus",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
            ),
        ],
    ),
    # Qwen 3.6 Flash
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p6_flash,
        friendly_name="Qwen 3.6 Flash",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.6-flash",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3.6 35B A3B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p6_35b_a3b,
        friendly_name="Qwen 3.6 35B A3B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.6-35b-a3b",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3.6 27B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p6_27b,
        friendly_name="Qwen 3.6 27B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.6-27b",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.groq,
                model_id="qwen/qwen3.6-27b",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=False,
                supports_function_calling=True,
                available_thinking_levels=QWEN_3P6_GROQ_THINKING_LEVELS,
                default_thinking_level="default",
                # Groq serves this model with image input, but no Groq provider in
                # Kiln is wired for multimodal yet, so it stays text-only here.
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3.6:27b",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_capable=True,
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="Qwen/Qwen3.6-27B",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
        ],
    ),
    # Qwen 3.5 Plus
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p5_plus,
        friendly_name="Qwen 3.5 Plus",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.5-plus-02-15",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3.5 Flash
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p5_flash,
        friendly_name="Qwen 3.5 Flash",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.5-flash-02-23",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                # returns empty assistant messages instead of tool calls
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3.5 397B (17B Active)
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p5_397b_a17b,
        friendly_name="Qwen 3.5 397B (17B Active)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.5-397b-a17b",
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.optional_r1_thinking,  # thinking is not always present
                supports_data_gen=True,
                supports_function_calling=False,  # seemed to be some interference between tool calls, json, and thinking
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="Qwen/Qwen3.5-397B-A17B",
                deprecated=True,
                supports_data_gen=False,
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            # Featherless provider omitted: this model's Featherless deployment
            # returns degenerate output -- it rambles to the 4096 token cap with
            # unrelated word lists, failing even the plaintext test. Not a config
            # issue; the served weights appear broken.
        ],
    ),
    # Qwen 3.5 122B (10B Active)
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p5_122b_a10b,
        friendly_name="Qwen 3.5 122B (10B Active)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.5-122b-a10b",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=False,  # Flakey
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3.5 35B (3B Active)
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p5_35b_a3b,
        friendly_name="Qwen 3.5 35B (3B Active)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.5-35b-a3b",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                # 3B active params: echoes the JSON schema instead of generating data
                supports_structured_output=False,
                supports_data_gen=False,
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3.5 27B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p5_27b,
        friendly_name="Qwen 3.5 27B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.5-27b",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3.5 9B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3p5_9b,
        friendly_name="Qwen 3.5 9B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3.5-9b",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3 Next 80B A3B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_next_80b_a3b,
        friendly_name="Qwen 3 Next 80B A3B (Instruct)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-next-80b-a3b-instruct",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
                deprecated=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                supports_function_calling=False,
            ),
        ],
    ),
    # Qwen 3 Next 80B A3B (Thinking)
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_next_80b_a3b_thinking,
        friendly_name="Qwen 3 Next 80B A3B (Thinking)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-next-80b-a3b-thinking",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
                supports_function_calling=True,
                reasoning_capable=True,
                require_openrouter_reasoning=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-Next-80B-A3B-Thinking",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
                supports_function_calling=True,
                reasoning_capable=True,
                siliconflow_enable_thinking=True,
            ),
        ],
    ),
    # Qwen 3 Max Thinking
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_max_thinking,
        friendly_name="Qwen 3 Max Thinking",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-max-thinking",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
                supports_function_calling=True,
                # Adaptive reasoning: does not always emit reasoning, so keep
                # reasoning_capable=False to avoid "no reasoning returned" errors.
            ),
        ],
    ),
    # Qwen 3 Max
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_max,
        friendly_name="Qwen 3 Max",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-max",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_function_calling=True,
            ),
        ],
    ),
    # Qwen 3 235B (22B Active) 2507 Version
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_235b_a22b_2507,
        friendly_name="Qwen 3 235B (22B Active) 2507",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-235b-a22b-thinking-2507",
                reasoning_capable=True,
                require_openrouter_reasoning=True,
                supports_data_gen=True,
                r1_openrouter_options=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:235b-a22b-thinking-2507-q4_K_M",
                supports_data_gen=True,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/qwen3-235b-a22b-thinking-2507",
                deprecated=True,
                supports_data_gen=True,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
                deprecated=True,
                supports_data_gen=True,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.optional_r1_thinking,
            ),
        ],
    ),
    # Qwen 3 235B (22B Active)
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_235b_a22b,
        friendly_name="Qwen 3 235B (22B Active)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-235b-a22b",
                reasoning_capable=True,
                require_openrouter_reasoning=True,
                supports_data_gen=True,
                r1_openrouter_options=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:235b-a22b",
                supports_data_gen=True,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/qwen3-235b-a22b",
                deprecated=True,
                supports_data_gen=True,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="Qwen/Qwen3-235B-A22B-fp8-tput",
                deprecated=True,
                supports_data_gen=True,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-235B-A22B",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                siliconflow_enable_thinking=True,
                supports_data_gen=True,
            ),
        ],
    ),
    # Qwen 3 235B (22B Active) 2507 Version Non-Thinking
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_235b_a22b_2507_no_thinking,
        friendly_name="Qwen 3 235B (22B Active) 2507 Non-Thinking",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-235b-a22b-2507",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
                reasoning_capable=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:235b-a22b-instruct-2507-q4_K_M",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                reasoning_capable=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
                deprecated=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
                deprecated=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
        ],
    ),
    # Qwen 3 235B (22B Active) Non-Thinking
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_235b_a22b_no_thinking,
        friendly_name="Qwen 3 235B (22B Active) Non-Thinking",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-235b-a22b",
                structured_output_mode=StructuredOutputMode.json_instructions,
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=True,
                reasoning_capable=False,
                parser=ModelParserID.optional_r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:235b-a22b",
                structured_output_mode=StructuredOutputMode.json_schema,
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/qwen3-235b-a22b",
                supports_data_gen=True,
                formatter=ModelFormatterID.qwen3_style_no_think,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.optional_r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="Qwen/Qwen3-235B-A22B-fp8-tput",
                deprecated=True,
                supports_data_gen=True,
                formatter=ModelFormatterID.qwen3_style_no_think,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.optional_r1_thinking,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-235B-A22B",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                siliconflow_enable_thinking=False,
                supports_data_gen=True,
            ),
        ],
    ),
    # Qwen 3 32B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_32b,
        friendly_name="Qwen 3 32B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.groq,
                model_id="Qwen/Qwen3-32B",
                deprecated=True,
                supports_data_gen=True,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                # This model doesn't return reasoning content after a tool call so we need to allow optional reasoning.
                parser=ModelParserID.optional_r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-32b",
                reasoning_capable=True,
                require_openrouter_reasoning=True,
                r1_openrouter_options=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
                supports_data_gen=True,
                # Not reliable, even for simple functions
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:32b",
                supports_data_gen=True,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-32B",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.cerebras,
                model_id="qwen-3-32b",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
                reasoning_capable=True,
                # This model doesn't return reasoning content after a tool call so we need to allow optional reasoning.
                parser=ModelParserID.optional_r1_thinking,
            ),
        ],
    ),
    # Qwen 3 32B No Thinking
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_32b_no_thinking,
        friendly_name="Qwen 3 32B Non-Thinking",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-32b",
                structured_output_mode=StructuredOutputMode.json_instructions,
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=True,
                parser=ModelParserID.optional_r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:32b",
                structured_output_mode=StructuredOutputMode.json_schema,
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.cerebras,
                model_id="qwen-3-32b",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=True,
                parser=ModelParserID.optional_r1_thinking,
            ),
        ],
    ),
    # Qwen 3 30B (3B Active) 2507 Version
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_30b_a3b_2507,
        friendly_name="Qwen 3 30B (3B Active) 2507",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:30b-a3b-thinking-2507-q4_K_M",
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                model_id="ai/qwen3:30B-A3B-Q4_K_M",
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
        ],
    ),
    # Qwen 3 30B (3B Active)
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_30b_a3b,
        friendly_name="Qwen 3 30B (3B Active)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-30b-a3b",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                require_openrouter_reasoning=True,
                r1_openrouter_options=True,
                parser=ModelParserID.r1_thinking,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:30b-a3b",
                supports_data_gen=True,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/qwen3-30b-a3b",
                deprecated=True,
                supports_data_gen=True,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-30B-A3B",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
                supports_data_gen=True,
            ),
        ],
    ),
    # Qwen 3 30B (3B Active) 2507 Version Non-Thinking
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_30b_a3b_2507_no_thinking,
        friendly_name="Qwen 3 30B (3B Active) 2507 Non-Thinking",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-30b-a3b-instruct-2507",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:30b-a3b-instruct-2507-q8_0",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
        ],
    ),
    # Qwen 3 30B (3B Active) Non-Thinking
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_30b_a3b_no_thinking,
        friendly_name="Qwen 3 30B (3B Active) Non-Thinking",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-30b-a3b",
                structured_output_mode=StructuredOutputMode.json_instructions,
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=True,
                parser=ModelParserID.optional_r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:30b-a3b",
                structured_output_mode=StructuredOutputMode.json_schema,
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/qwen3-30b-a3b",
                supports_data_gen=True,
                formatter=ModelFormatterID.qwen3_style_no_think,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.optional_r1_thinking,
            ),
        ],
    ),
    # Qwen 3 14B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_14b,
        friendly_name="Qwen 3 14B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-14b",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                require_openrouter_reasoning=True,
                r1_openrouter_options=True,
                parser=ModelParserID.r1_thinking,
                supports_data_gen=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:14b",
                supports_data_gen=True,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-14B",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                reasoning_capable=True,
                siliconflow_enable_thinking=True,
                reasoning_optional_for_structured_output=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                model_id="ai/qwen3:14B-Q6_K",
                supports_data_gen=True,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
            ),
        ],
    ),
    # Qwen 3 14B Non-Thinking
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_14b_no_thinking,
        friendly_name="Qwen 3 14B Non-Thinking",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-14b",
                structured_output_mode=StructuredOutputMode.json_instructions,
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=True,
                parser=ModelParserID.optional_r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:14b",
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-14B",
                formatter=ModelFormatterID.qwen3_style_no_think,
                structured_output_mode=StructuredOutputMode.json_schema,
                siliconflow_enable_thinking=False,
                supports_data_gen=True,
            ),
        ],
    ),
    # Qwen 3 8B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_8b,
        friendly_name="Qwen 3 8B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-8b",
                supports_structured_output=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                require_openrouter_reasoning=True,
                r1_openrouter_options=True,
                parser=ModelParserID.r1_thinking,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:8b",
                supports_data_gen=False,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-8B",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                siliconflow_enable_thinking=True,
                reasoning_optional_for_structured_output=True,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                model_id="ai/qwen3:8B-Q4_K_M",
                supports_data_gen=False,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
            ),
        ],
    ),
    # Qwen 3 8B Non-Thinking
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_8b_no_thinking,
        friendly_name="Qwen 3 8B Non-Thinking",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-8b",
                structured_output_mode=StructuredOutputMode.json_instructions,
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=False,
                parser=ModelParserID.optional_r1_thinking,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:8b",
                structured_output_mode=StructuredOutputMode.json_schema,
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-8B",
                structured_output_mode=StructuredOutputMode.json_schema,
                siliconflow_enable_thinking=False,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # Qwen 3 4B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_4b,
        friendly_name="Qwen 3 4B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-4b:free",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                require_openrouter_reasoning=True,
                r1_openrouter_options=True,
                parser=ModelParserID.r1_thinking,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:4b",
                supports_data_gen=False,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
            ),
        ],
    ),
    # Qwen 3 4B Non-Thinking
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_4b_no_thinking,
        friendly_name="Qwen 3 4B Non-Thinking",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-4b:free",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=False,
                parser=ModelParserID.optional_r1_thinking,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:4b",
                structured_output_mode=StructuredOutputMode.json_schema,
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # Qwen 3 0.6B Non-Thinking -- not respecting /no_think tag, skipping
    # Qwen 3 1.7B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_1p7b,
        friendly_name="Qwen 3 1.7B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-1.7b:free",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                require_openrouter_reasoning=True,
                r1_openrouter_options=True,
                parser=ModelParserID.r1_thinking,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:1.7b",
                supports_data_gen=False,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
            ),
        ],
    ),
    # Qwen 3 1.7B Non-Thinking
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_1p7b_no_thinking,
        friendly_name="Qwen 3 1.7B Non-Thinking",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-1.7b:free",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=False,
                parser=ModelParserID.optional_r1_thinking,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:1.7b",
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=False,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
            ),
        ],
    ),
    # Qwen 3 0.6B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_0p6b,
        friendly_name="Qwen 3 0.6B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-0.6b-04-28:free",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                require_openrouter_reasoning=True,
                r1_openrouter_options=True,
                parser=ModelParserID.r1_thinking,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3:0.6b",
                supports_data_gen=False,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                model_id="ai/qwen3:0.6B-F16",
                supports_data_gen=False,
                reasoning_capable=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
            ),
        ],
    ),
    # Qwen 3 235B (22B Active) VL Instruct
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_vl_235b_a22b_no_thinking,
        friendly_name="Qwen 3 VL Instruct 235B / 22B Active (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen3-vl-235b-a22b-instruct",
                structured_output_mode=StructuredOutputMode.json_schema,
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=False,
                reasoning_capable=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/qwen3-vl-235b-a22b-instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                formatter=ModelFormatterID.qwen3_style_no_think,
                supports_data_gen=False,
                reasoning_capable=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-VL-235B-A22B-Instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=False,
                reasoning_capable=False,
                supports_doc_extraction=True,
                supports_vision=True,
                supports_function_calling=False,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3 VL 32B Instruct
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_vl_32b_no_thinking,
        friendly_name="Qwen 3 VL Instruct 32B (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-VL-32B-Instruct",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                reasoning_capable=False,
                supports_doc_extraction=True,
                supports_function_calling=False,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3 VL 30B (3B Active) Instruct
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_vl_30b_a3b_no_thinking,
        friendly_name="Qwen 3 VL Instruct 30B / 3B Active (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/qwen3-vl-30b-a3b-instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                reasoning_capable=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                reasoning_capable=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3 VL 8B Instruct
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_vl_8b_no_thinking,
        friendly_name="Qwen 3 VL Instruct 8B (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-VL-8B-Instruct",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                reasoning_capable=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3 VL Thinking 235B / 22B Active
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_vl_235b_a22b,
        friendly_name="Qwen 3 VL Thinking 235B / 22B Active (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3-vl:235b",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                reasoning_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
                max_parallel_requests=1,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/qwen3-vl-235b-a22b-thinking",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
                supports_data_gen=False,
                reasoning_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-VL-235B-A22B-Thinking",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=False,
                reasoning_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3 VL Thinking 32B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_vl_32b,
        friendly_name="Qwen 3 VL Thinking 32B (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3-vl:32b",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                reasoning_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
                max_parallel_requests=1,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-VL-32B-Thinking",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=False,
                reasoning_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3 VL Thinking 30B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_vl_30b,
        friendly_name="Qwen 3 VL Thinking 30B / 3B Active (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3-vl:30b",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                reasoning_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
                max_parallel_requests=1,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/qwen3-vl-30b-a3b-thinking",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
                supports_data_gen=False,
                reasoning_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-VL-30B-A3B-Thinking",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=False,
                reasoning_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3 VL Thinking 8B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_vl_8b,
        friendly_name="Qwen 3 VL Thinking 8B (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3-vl:8b",
                ollama_model_aliases=[
                    "qwen3-vl",
                ],
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                reasoning_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
                max_parallel_requests=1,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen3-VL-8B-Thinking",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=False,
                reasoning_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 3 VL Thinking 4B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_vl_4b,
        friendly_name="Qwen 3 VL Thinking 4B (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3-vl:4b",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                reasoning_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
                max_parallel_requests=1,
            ),
        ],
    ),
    # Qwen 3 VL Thinking 2B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_3_vl_2b,
        friendly_name="Qwen 3 VL Thinking 2B (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen3-vl:2b",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=False,
                reasoning_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
                max_parallel_requests=1,
            ),
        ],
    ),
    # Qwen Long L1 32B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_long_l1_32b,
        friendly_name="QwenLong L1 32B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Tongyi-Zhiwen/QwenLong-L1-32B",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
                supports_function_calling=False,
            ),
        ],
    ),
    # QwQ 32B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwq_32b,
        friendly_name="QwQ 32B (Qwen Reasoning)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwq-32b",
                deprecated=True,
                reasoning_capable=True,
                require_openrouter_reasoning=True,
                r1_openrouter_options=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwq",
                reasoning_capable=True,
                parser=ModelParserID.r1_thinking,
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="Qwen/QwQ-32B",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.r1_thinking,
                reasoning_capable=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/QwQ-32B",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                model_id="ai/qwq:32B-Q4_K_M",
                reasoning_capable=True,
                parser=ModelParserID.r1_thinking,
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_function_calling=False,
            ),
        ],
    ),
    # Qwen 2.5 VL 72B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_2p5_vl_72b,
        friendly_name="Qwen 2.5 VL 72B (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen2.5vl:72b",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
                max_parallel_requests=1,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen2.5-vl-72b-instruct",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen2.5-VL-72B-Instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="Qwen/Qwen2.5-VL-72B-Instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # supports video, but LiteLLM fails request validation
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 2.5 VL 32B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_2p5_vl_32b,
        friendly_name="Qwen 2.5 VL 32B (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen2.5vl:32b",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
                max_parallel_requests=1,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen2.5-vl-32b-instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Qwen/Qwen2.5-VL-32B-Instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/qwen2p5-vl-32b-instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 2.5 VL 7B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_2p5_vl_7b,
        friendly_name="Qwen 2.5 VL 7B (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen2.5vl:7b",
                supports_structured_output=False,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
                max_parallel_requests=1,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen-2.5-vl-7b-instruct",
                deprecated=True,
                supports_structured_output=False,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Pro/Qwen/Qwen2.5-VL-7B-Instruct",
                deprecated=True,
                supports_structured_output=False,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # Qwen 2.5 VL 3B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_2p5_vl_3b,
        friendly_name="Qwen 2.5 VL 3B (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen2.5vl:3b",
                supports_structured_output=False,
                supports_function_calling=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
                multimodal_requires_pdf_as_image=True,
                max_parallel_requests=1,
            ),
        ],
    ),
    # Qwen 2.5 72B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_2p5_72b,
        friendly_name="Qwen 2.5 72B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen-2.5-72b-instruct",
                # Not consistent with structure data. Works sometimes but not often
                supports_structured_output=False,
                supports_data_gen=False,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen2.5:72b",
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                provider_finetune_id="Qwen/Qwen2.5-72B-Instruct",
            ),
        ],
    ),
    # Qwen 2.5 14B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_2p5_14b,
        friendly_name="Qwen 2.5 14B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                provider_finetune_id="Qwen/Qwen2.5-14B-Instruct",
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen2.5:14b",
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # Qwen 2.5 7B
    KilnModel(
        family=ModelFamily.qwen,
        name=ModelName.qwen_2p5_7b,
        friendly_name="Qwen 2.5 7B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="qwen/qwen-2.5-7b-instruct",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.ollama,
                model_id="qwen2.5",
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.docker_model_runner,
                model_id="ai/qwen2.5:7B-Q4_K_M",
                supports_function_calling=False,
            ),
        ],
    ),
    # GLM 5.3
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_5_3,
        friendly_name="GLM 5.3",
        featured_rank=7,
        editorial_notes="Z.ai's newest flagship, with a 1M token context window and configurable reasoning. Strong long-horizon agentic and coding performance.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="z-ai/glm-5.3",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="zai-org/GLM-5.3",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="accounts/fireworks/models/glm-5p3",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=False,
            ),
        ],
    ),
    # GLM 5.3 Flash
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_5_3_flash,
        friendly_name="GLM 5.3 Flash",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                suggested_for_synthetic_user=True,
                model_id="z-ai/glm-5.3-flash",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                suggested_for_synthetic_user=True,
                model_id="zai-org/GLM-5.3-Flash",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/glm-5p3-flash",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # GLM 5.2
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_5_2,
        friendly_name="GLM 5.2",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="z-ai/glm-5.2",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/glm-5p2",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="zai-org/GLM-5.2",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="zai-org/GLM-5.2",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="zai-org/GLM-5.2",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
        ],
    ),
    # GLM 5.2 Fast — Fireworks speed-optimized serving of GLM 5.2 (routers/ slug, ~2x throughput)
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_5_2_fast,
        friendly_name="GLM 5.2 Fast",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/routers/glm-5p2-fast",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
        ],
    ),
    # GLM 5.1
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_5_1,
        friendly_name="GLM 5.1",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="z-ai/glm-5.1",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/glm-5p1",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="zai-org/GLM-5.1",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Pro/zai-org/GLM-5.1",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="zai-org/GLM-5.1",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
        ],
    ),
    # GLM 5V Turbo
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_5v_turbo,
        friendly_name="GLM 5V Turbo (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="z-ai/glm-5v-turbo",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
        ],
    ),
    # GLM 5 Turbo
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_5_turbo,
        friendly_name="GLM 5 Turbo",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="z-ai/glm-5-turbo",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
        ],
    ),
    # GLM 5
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_5,
        friendly_name="GLM 5",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="z-ai/glm-5",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/glm-5",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="zai-org/GLM-5",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Pro/zai-org/GLM-5",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="zai-org/GLM-5",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
        ],
    ),
    # GLM 4.7
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_4_7,
        friendly_name="GLM 4.7",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="z-ai/glm-4.7",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Pro/zai-org/GLM-4.7",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.cerebras,
                model_id="zai-glm-4.7",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="zai-org/GLM-4.7",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
        ],
    ),
    # GLM 4.7 Flash. Not available on siliconflow cn yet.
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_4_7_flash,
        friendly_name="GLM 4.7 Flash",
        editorial_notes="Cost-effective, fast, and open model from Z.ai.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="z-ai/glm-4.7-flash",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
        ],
    ),
    # GLM 4.6V
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_4_6v,
        friendly_name="GLM 4.6V (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="z-ai/glm-4.6v",
                structured_output_mode=StructuredOutputMode.function_calling,
                reasoning_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="zai-org/glm-4.6v",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.function_calling,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
                max_parallel_requests=1,
            ),
        ],
    ),
    # GLM 4.6
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_4_6,
        friendly_name="GLM 4.6",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="z-ai/glm-4.6:exacto",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="zai-org/GLM-4.6",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
            KilnModelProvider(
                name=ModelProviderName.cerebras,
                model_id="zai-glm-4.6",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
            ),
        ],
    ),
    # GLM 4.5V
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_4_5v,
        friendly_name="GLM 4.5V (Vision-Language)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="z-ai/glm-4.5v",
                supports_structured_output=False,
                reasoning_capable=True,
                supports_data_gen=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="zai-org/GLM-4.5V",
                supports_structured_output=False,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
                supports_data_gen=False,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
                max_parallel_requests=1,
            ),
            # fireworks currently has it but not serverless
        ],
    ),
    # GLM 4.5
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_4_5,
        friendly_name="GLM 4.5",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="z-ai/glm-4.5",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/glm-4p5",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="zai-org/GLM-4.5",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
            ),
        ],
    ),
    # GLM 4.5 AIR
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_4_5_air,
        friendly_name="GLM 4.5 AIR",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="z-ai/glm-4.5-air",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/glm-4p5-air",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="zai-org/GLM-4.5-Air-FP8",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                parser=ModelParserID.r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="zai-org/GLM-4.5-Air",
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
            ),
        ],
    ),
    # GLM 4.1V 9B
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_4_1v_9b_thinking,
        friendly_name="GLM-4.1V 9B Thinking",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Pro/THUDM/GLM-4.1V-9B-Thinking",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # GLM Z1 32B 0414
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_z1_32b_0414,
        friendly_name="GLM-Z1 32B 0414",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="THUDM/GLM-Z1-32B-0414",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # GLM Z1 9B 0414
    KilnModel(
        family=ModelFamily.glm,
        name=ModelName.glm_z1_9b_0414,
        friendly_name="GLM-Z1 9B 0414",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="THUDM/GLM-Z1-9B-0414",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # Kimi K3
    KilnModel(
        family=ModelFamily.kimi,
        name=ModelName.kimi_k3,
        friendly_name="Kimi K3",
        featured_rank=5,
        editorial_notes="Open, state-of-the-art model from Moonshot AI. 2.8T-parameter MoE with a 1M token context, configurable reasoning, and strong agentic performance.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="moonshotai/kimi-k3",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                available_thinking_levels=KIMI_K3_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="high",
                openrouter_reasoning_object=True,
                multimodal_capable=True,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="accounts/fireworks/models/kimi-k3",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                multimodal_capable=True,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="moonshotai/Kimi-K3",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
            # SiliconFlow provider omitted: moonshotai/Kimi-K3 returns HTTP 400
            # "Model does not exist" on the .cn endpoint Kiln uses (it appears live
            # on the .com site only).
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                suggested_for_evals=True,
                suggested_for_data_gen=True,
                model_id="moonshotai/Kimi-K3",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
        ],
    ),
    # Kimi K2.6
    # Not available on Together AI or SiliconFlow CN yet
    KilnModel(
        family=ModelFamily.kimi,
        name=ModelName.kimi_k2_6,
        friendly_name="Kimi K2.6",
        editorial_notes="Open, state-of-the-art model from Moonshot AI. Excellent price-to-performance ratio. Enhanced agent planning and reasoning capabilities.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/kimi-k2p6",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                multimodal_capable=True,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="moonshotai/kimi-k2.6",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_requires_pdf_as_image=True,
                supports_doc_extraction=True,
                # OpenRouter reports reasoning support via include_reasoning parameter
                # but similar to K2.5, it doesn't always return reasoning in the response
                # reasoning_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            # Together AI provider commented out - model not available yet
            # Check https://api.together.ai/models for availability
            # KilnModelProvider(
            #     name=ModelProviderName.together_ai,
            #     model_id="moonshotai/Kimi-K2.6",
            #     structured_output_mode=StructuredOutputMode.json_instructions,
            #     supports_data_gen=True,
            #     multimodal_capable=True,
            #     supports_vision=True,
            #     supports_doc_extraction=True,
            #     multimodal_mime_types=[
            #         KilnMimeType.PDF,
            #         KilnMimeType.TXT,
            #         KilnMimeType.MD,
            #         KilnMimeType.JPG,
            #         KilnMimeType.PNG,
            #     ],
            #     multimodal_requires_pdf_as_image=True,
            # ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="moonshotai/Kimi-K2.6",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
            ),
        ],
    ),
    # Kimi K2.5
    # Not available on SiliconFlow CN yet
    KilnModel(
        family=ModelFamily.kimi,
        name=ModelName.kimi_k2_5,
        friendly_name="Kimi K2.5",
        editorial_notes="Open, state-of-the-art model from Moonshot AI. Excellent price-to-performance ratio.",
        providers=[
            # Fireworks provider commented out due to immutable parameter requirements:
            # top_p must be exactly 0.95 (thinking mode) or temperature must be 0.6 (non-thinking mode)
            # We currently always send temperature=1.0 and top_p=1.0 from run_config, which causes errors.
            # In the future maybe Fireworks will make this easier to work with, or Kiln will add some sort of fixed params for thinking mode settings.
            # KilnModelProvider(
            #     name=ModelProviderName.fireworks_ai,
            #     model_id="accounts/fireworks/models/kimi-k2p5",
            #     structured_output_mode=StructuredOutputMode.json_schema,
            #     supports_data_gen=True,
            #     multimodal_capable=True,
            #     supports_vision=True,
            #     supports_doc_extraction=True,
            #     multimodal_mime_types=[
            #         KilnMimeType.JPG,
            #         KilnMimeType.PNG,
            #     ],
            # ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="moonshotai/kimi-k2.5",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                multimodal_capable=True,
                supports_vision=True,
                multimodal_requires_pdf_as_image=True,
                supports_doc_extraction=True,
                # while the model is capable of reasoning, it doesn't always return it in the response, so disabling it here
                # reasoning_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="moonshotai/Kimi-K2.5",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
                multimodal_capable=True,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                multimodal_requires_pdf_as_image=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="moonshotai/Kimi-K2.5",
                structured_output_mode=StructuredOutputMode.json_instructions,
            ),
        ],
    ),
    # Kimi K2 Thinking
    # Not hosted on Groq, and Together AI yet
    KilnModel(
        family=ModelFamily.kimi,
        name=ModelName.kimi_k2_thinking,
        friendly_name="Kimi K2 Thinking",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/kimi-k2-thinking",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="moonshotai/kimi-k2-thinking",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                require_openrouter_reasoning=True,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Pro/moonshotai/Kimi-K2-Thinking",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="moonshotai/Kimi-K2-Thinking",
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.optional_r1_thinking,
            ),
        ],
    ),
    # Kimi K2 Instruct 0905
    KilnModel(
        family=ModelFamily.kimi,
        name=ModelName.kimi_k2_0905,
        friendly_name="Kimi K2 0905",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="moonshotai/kimi-k2-0905:exacto",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/kimi-k2-instruct-0905",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="moonshotai/Kimi-K2-Instruct-0905",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                # this model on this provider currently fails the tool call test, but might work in the future
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.groq,
                model_id="moonshotai/kimi-k2-instruct-0905",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Pro/moonshotai/Kimi-K2-Instruct-0905",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
        ],
    ),
    # Kimi K2 Instruct
    KilnModel(
        family=ModelFamily.kimi,
        name=ModelName.kimi_k2,
        friendly_name="Kimi K2",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/kimi-k2-instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="moonshotai/kimi-k2",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="moonshotai/Kimi-K2-Instruct",
                deprecated=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
            ),
            KilnModelProvider(
                name=ModelProviderName.groq,
                model_id="moonshotai/kimi-k2-instruct",
                deprecated=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="Pro/moonshotai/Kimi-K2-Instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
        ],
    ),
    KilnModel(
        family=ModelFamily.kimi,
        name=ModelName.kimi_dev_72b,
        friendly_name="Kimi Dev 72B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="moonshotai/Kimi-Dev-72B",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                reasoning_optional_for_structured_output=True,
                supports_function_calling=False,
            ),
        ],
    ),
    # Ernie 4.5 300B A47B
    KilnModel(
        family=ModelFamily.ernie,
        name=ModelName.ernie_4_5_300b_a47b,
        friendly_name="Ernie 4.5 300B A47B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="baidu/ernie-4.5-300b-a47b",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
                r1_openrouter_options=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="baidu/ERNIE-4.5-300B-A47B",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
                supports_function_calling=False,
            ),
        ],
    ),
    # Hunyuan Hy3 — Tencent's 295B MoE (21B active), 256K context, text-only reasoning + tool-calling model.
    # reasoning_capable left False: Hy3 has adaptive/hybrid thinking (a no_think toggle) and can return no reasoning.
    KilnModel(
        family=ModelFamily.hunyuan,
        name=ModelName.hunyuan_hy3,
        friendly_name="Hunyuan Hy3",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="tencent/hy3",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
        ],
    ),
    # Hunyuan A13B Instruct
    KilnModel(
        family=ModelFamily.hunyuan,
        name=ModelName.hunyuan_a13b,
        friendly_name="Hunyuan A13B",
        providers=[
            # Openrouter provider for this model exists but currently wrongly parses the answer
            # it returns the reasoning at the right place, but wraps the answer (even JSON response)
            # between <answer> and </answer> tags
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="tencent/Hunyuan-A13B-Instruct",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                siliconflow_enable_thinking=True,
                reasoning_optional_for_structured_output=True,
                supports_data_gen=False,
                supports_function_calling=False,
            ),
        ],
    ),
    # Minimax M3
    # 1M context, native multimodal (text/image/video input), sparse attention
    KilnModel(
        family=ModelFamily.minimax,
        name=ModelName.minimax_m3,
        friendly_name="Minimax M3",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="minimax/minimax-m3",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                reasoning_capable=True,
                supports_data_gen=True,
                r1_openrouter_options=True,
                require_openrouter_reasoning=True,
                parser=ModelParserID.r1_thinking,
                multimodal_capable=True,
                supports_vision=True,
                supports_doc_extraction=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # documents (via PDF-as-image)
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
            ),
            # Fireworks exposes M3 as a text-only endpoint (no vision/video input).
            # Unlike OpenRouter, Fireworks does not emit reasoning unless a reasoning
            # effort is explicitly requested, so we treat it as a standard model here
            # and let Kiln drive chain-of-thought via the multi-turn COT strategy.
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/minimax-m3",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
            # Together serves the raw weights, which emit <think> inline (like
            # Featherless below). Use the optional parser so the tags are stripped
            # when present, and leave reasoning_capable False since M3 does not always
            # emit reasoning. Together advertises this as a text chat endpoint, so no
            # vision/multimodal here.
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="MiniMaxAI/MiniMax-M3",
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.optional_r1_thinking,
                supports_data_gen=True,
                supports_function_calling=False,
            ),
            # Featherless serves the raw weights, which emit <think> inline. Use the
            # optional parser so the tags are stripped when present, and leave
            # reasoning_capable False since (per the Fireworks note above) M3 does not
            # always emit reasoning -- requiring it would error on those responses.
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="MiniMaxAI/MiniMax-M3",
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.optional_r1_thinking,
                supports_data_gen=True,
                # Featherless doesn't advertise tool_use for this model
                supports_function_calling=False,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                ],
            ),
        ],
    ),
    # Minimax M2.7
    KilnModel(
        family=ModelFamily.minimax,
        name=ModelName.minimax_m2_7,
        friendly_name="Minimax M2.7",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="minimax/minimax-m2.7",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                reasoning_capable=True,
                supports_data_gen=True,
                r1_openrouter_options=True,
                require_openrouter_reasoning=True,
                parser=ModelParserID.r1_thinking,
            ),
            # Fireworks exposes M2.7 as a text-only endpoint and does not emit
            # reasoning unless a reasoning effort is explicitly requested, so we
            # treat it as a standard model here (mirroring the M3 Fireworks entry).
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/minimax-m2p7",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="MiniMaxAI/MiniMax-M2.7",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                reasoning_capable=True,
                supports_data_gen=True,
                reasoning_optional_for_structured_output=True,
                parser=ModelParserID.optional_r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="MiniMaxAI/MiniMax-M2.7",
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.optional_r1_thinking,
            ),
        ],
    ),
    # Minimax M2.5
    # OpenRouter accepts json_schema but M2.5 ignores the constraint;
    # json_instruction_and_object works because the simpler response_format:json_object
    # IS respected, and the schema is included in the prompt instructions.
    KilnModel(
        family=ModelFamily.minimax,
        name=ModelName.minimax_m2_5,
        friendly_name="Minimax M2.5",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="minimax/minimax-m2.5",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                reasoning_capable=True,
                supports_data_gen=True,
                r1_openrouter_options=True,
                require_openrouter_reasoning=True,
                parser=ModelParserID.r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/minimax-m2p5",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_data_gen=True,
                reasoning_optional_for_structured_output=True,
                parser=ModelParserID.optional_r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="MiniMaxAI/MiniMax-M2.5",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                reasoning_capable=True,
                supports_data_gen=True,
                reasoning_optional_for_structured_output=True,
                parser=ModelParserID.optional_r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.featherless_ai,
                model_id="MiniMaxAI/MiniMax-M2.5",
                structured_output_mode=StructuredOutputMode.json_instructions,
                parser=ModelParserID.optional_r1_thinking,
            ),
        ],
    ),
    # Minimax M2.1
    KilnModel(
        family=ModelFamily.minimax,
        name=ModelName.minimax_m2_1,
        friendly_name="Minimax M2.1",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="minimax/minimax-m2.1",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_data_gen=True,
                r1_openrouter_options=True,
                require_openrouter_reasoning=True,
                parser=ModelParserID.r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/minimax-m2p1",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_data_gen=True,
                reasoning_optional_for_structured_output=True,
            ),
        ],
    ),
    # Minimax M2
    KilnModel(
        family=ModelFamily.minimax,
        name=ModelName.minimax_m2,
        friendly_name="Minimax M2",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="minimax/minimax-m2",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_data_gen=True,
                r1_openrouter_options=True,
                require_openrouter_reasoning=True,
                parser=ModelParserID.r1_thinking,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="MiniMaxAI/MiniMax-M2",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                reasoning_capable=True,
                supports_data_gen=True,
                reasoning_optional_for_structured_output=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/minimax-m2",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_data_gen=True,
                reasoning_optional_for_structured_output=True,
            ),
        ],
    ),
    # Minimax M1 80K
    KilnModel(
        family=ModelFamily.minimax,
        name=ModelName.minimax_m1_80k,
        friendly_name="Minimax M1",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="minimax/minimax-m1",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
                r1_openrouter_options=True,
                require_openrouter_reasoning=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="MiniMaxAI/MiniMax-M1-80k",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                supports_data_gen=True,
                supports_function_calling=False,
            ),
        ],
    ),
    # Pangu Pro MOE
    KilnModel(
        family=ModelFamily.pangu,
        name=ModelName.pangu_pro_moe_72b_a16b,
        friendly_name="Pangu Pro MOE 72B A16B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="ascend-tribe/pangu-pro-moe",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                supports_data_gen=True,
                supports_function_calling=False,
            ),
        ],
    ),
    # Bytedance Seed OSS 36B
    KilnModel(
        family=ModelFamily.bytedance,
        name=ModelName.bytedance_seed_oss_36b,
        friendly_name="ByteDance Seed OSS 36B",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="bytedance/seed-oss-36b-instruct",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_data_gen=True,
                supports_function_calling=False,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="ByteDance-Seed/Seed-OSS-36B-Instruct",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_data_gen=True,
                supports_function_calling=False,
                reasoning_optional_for_structured_output=True,
            ),
        ],
    ),
    # Bytedance Seed 1.6
    KilnModel(
        family=ModelFamily.bytedance,
        name=ModelName.bytedance_seed_1_6,
        friendly_name="ByteDance Seed 1.6",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="bytedance-seed/seed-1.6",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_data_gen=True,
            ),
        ],
    ),
    # Bytedance Seed 1.6 Flash
    KilnModel(
        family=ModelFamily.bytedance,
        name=ModelName.bytedance_seed_1_6_flash,
        friendly_name="ByteDance Seed 1.6 Flash",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="bytedance-seed/seed-1.6-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_data_gen=True,
            ),
        ],
    ),
    # Arcee AI Trinity Large Thinking
    KilnModel(
        family=ModelFamily.arcee,
        name=ModelName.arcee_trinity_large_thinking,
        friendly_name="Arcee Trinity Large (Thinking)",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="arcee-ai/trinity-large-thinking",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
                supports_function_calling=True,
                reasoning_capable=True,
                require_openrouter_reasoning=True,
            ),
        ],
    ),
    # StepFun Step 3.7 Flash
    KilnModel(
        family=ModelFamily.stepfun,
        name=ModelName.stepfun_step3_7_flash,
        friendly_name="StepFun Step 3.7 Flash",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="stepfun/step-3.7-flash",
                structured_output_mode=StructuredOutputMode.json_schema,
                reasoning_capable=True,
                supports_data_gen=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                    # video
                    KilnMimeType.MP4,
                    KilnMimeType.MOV,
                ],
                supports_doc_extraction=True,
                multimodal_requires_pdf_as_image=True,
                supports_vision=True,
            ),
        ],
    ),
    # StepFun Step3 (deprecated)
    KilnModel(
        family=ModelFamily.stepfun,
        name=ModelName.stepfun_step3,
        friendly_name="StepFun Step3",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="stepfun-ai/step3",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                supports_function_calling=False,
                # image only is not sufficient for doc extraction
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                supports_doc_extraction=True,
                multimodal_requires_pdf_as_image=True,
                supports_vision=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.siliconflow_cn,
                model_id="stepfun-ai/step3",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instructions,
                reasoning_capable=True,
                supports_function_calling=False,
                # image only is not sufficient for doc extraction
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
                supports_doc_extraction=True,
                multimodal_requires_pdf_as_image=True,
                supports_vision=True,
            ),
        ],
    ),
    # MiMo-V2.5-Pro
    KilnModel(
        family=ModelFamily.mimo,
        name=ModelName.mimo_v2_5_pro,
        friendly_name="MiMo-V2.5-Pro",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="xiaomi/mimo-v2.5-pro",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
            ),
        ],
    ),
    # MiMo-V2.5
    KilnModel(
        family=ModelFamily.mimo,
        name=ModelName.mimo_v2_5,
        friendly_name="MiMo-V2.5",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="xiaomi/mimo-v2.5",
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_vision=True,
                multimodal_capable=True,
                supports_doc_extraction=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    # documents
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    # images
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # MiMo-V2-Pro
    KilnModel(
        family=ModelFamily.mimo,
        name=ModelName.mimo_v2_pro,
        friendly_name="MiMo-V2-Pro",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="xiaomi/mimo-v2-pro",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
            ),
        ],
    ),
    # MiMo-V2-Flash
    KilnModel(
        family=ModelFamily.mimo,
        name=ModelName.mimo_v2_flash,
        friendly_name="MiMo-V2-Flash",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="xiaomi/mimo-v2-flash",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
            ),
        ],
    ),
    # MiMo-V2-Omni
    KilnModel(
        family=ModelFamily.mimo,
        name=ModelName.mimo_v2_omni,
        friendly_name="MiMo-V2-Omni",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="xiaomi/mimo-v2-omni",
                deprecated=True,
                structured_output_mode=StructuredOutputMode.json_instruction_and_object,
                supports_data_gen=True,
                supports_vision=True,
                multimodal_capable=True,
                multimodal_mime_types=[
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Inkling (Thinking Machines)
    KilnModel(
        family=ModelFamily.thinking_machines,
        name=ModelName.inkling,
        friendly_name="Inkling",
        editorial_notes="Thinking Machines' first open-weights foundation model (Apache 2.0). A 975B-parameter MoE (41B active) with configurable reasoning. Hosted on Together AI.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="thinkingmachines/inkling",
                structured_output_mode=StructuredOutputMode.json_schema,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.together_ai,
                model_id="thinkingmachines/Inkling",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
            ),
            KilnModelProvider(
                name=ModelProviderName.fireworks_ai,
                model_id="accounts/fireworks/models/inkling",
                structured_output_mode=StructuredOutputMode.json_instructions,
                supports_data_gen=True,
            ),
        ],
    ),
    # Fugu Max
    KilnModel(
        family=ModelFamily.sakana,
        name=ModelName.fugu_max,
        friendly_name="Fugu Max",
        editorial_notes="Sakana's largest Fable-tier model from Japan. A learned multi-agent orchestrator that routes across a pool of models, including recursive instances of itself. 1M context, with vision and configurable reasoning.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="sakana/fugu-max",
                supports_structured_output=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=FUGU_ULTRA_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="xhigh",
                openrouter_reasoning_object=True,
                multimodal_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Fugu Ultra v2
    KilnModel(
        family=ModelFamily.sakana,
        name=ModelName.fugu_ultra_v2,
        friendly_name="Fugu Ultra v2",
        editorial_notes="The second generation of Sakana's Fugu Ultra, a Fable-tier model from Japan. A learned multi-agent orchestrator that routes across a pool of models, including recursive instances of itself. 1M context, with vision and configurable reasoning.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="sakana/fugu-ultra-v2",
                supports_structured_output=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=FUGU_ULTRA_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="xhigh",
                openrouter_reasoning_object=True,
                multimodal_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
    # Fugu Ultra
    KilnModel(
        family=ModelFamily.sakana,
        name=ModelName.fugu_ultra,
        friendly_name="Fugu Ultra",
        editorial_notes="Sakana's Fable-tier model from Japan. A learned multi-agent orchestrator that routes across a pool of models, including recursive instances of itself. 1M context, with vision and configurable reasoning.",
        providers=[
            KilnModelProvider(
                name=ModelProviderName.openrouter,
                model_id="sakana/fugu-ultra",
                supports_structured_output=True,
                supports_data_gen=True,
                structured_output_mode=StructuredOutputMode.json_schema,
                available_thinking_levels=FUGU_ULTRA_OPENROUTER_THINKING_LEVELS,
                default_thinking_level="xhigh",
                openrouter_reasoning_object=True,
                multimodal_capable=True,
                supports_doc_extraction=True,
                supports_vision=True,
                multimodal_requires_pdf_as_image=True,
                multimodal_mime_types=[
                    KilnMimeType.PDF,
                    KilnMimeType.TXT,
                    KilnMimeType.MD,
                    KilnMimeType.JPG,
                    KilnMimeType.PNG,
                ],
            ),
        ],
    ),
]


def get_model_by_name(name: ModelName) -> KilnModel:
    for model in built_in_models:
        if model.name == name:
            return model
    raise ValueError(f"Model {name} not found in the list of built-in models")


def built_in_models_from_provider(
    provider_name: ModelProviderName, model_name: str
) -> KilnModelProvider | None:
    for model in built_in_models:
        if model.name == model_name:
            for p in model.providers:
                if p.name == provider_name:
                    return p
    return None


def default_structured_output_mode_for_model_provider(
    model_name: str,
    provider: ModelProviderName,
    default: StructuredOutputMode = StructuredOutputMode.default,
    disallowed_modes: List[StructuredOutputMode] = [],
) -> StructuredOutputMode:
    """
    We don't expose setting this manually in the UI, so pull a recommended mode from ml_model_list
    """
    try:
        # Convert string to ModelName enum
        model_name_enum = ModelName(model_name)
        model = get_model_by_name(model_name_enum)
    except (ValueError, KeyError):
        # If model not found, return default
        return default

    # Find the provider within the model's providers
    for model_provider in model.providers:
        if model_provider.name == provider:
            mode = model_provider.structured_output_mode
            if mode not in disallowed_modes:
                return mode

    # If provider not found, return default
    return default


def default_thinking_level_for_model_provider(
    model_name: str,
    provider: ModelProviderName,
) -> str | None:
    try:
        model_name_enum = ModelName(model_name)
        model = get_model_by_name(model_name_enum)
    except (ValueError, KeyError):
        return None

    for model_provider in model.providers:
        if model_provider.name == provider:
            return model_provider.default_thinking_level

    return None
