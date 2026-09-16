from enum import Enum, IntEnum
from typing import Any, Dict, List, TypeAlias

StructuredInputType: TypeAlias = Dict[str, Any] | List[Any]
InputType: TypeAlias = StructuredInputType | str


class Priority(IntEnum):
    """Priority levels, where P0 is highest priority."""

    p0 = 0
    p1 = 1
    p2 = 2
    p3 = 3


class EvalStatus(str, Enum):
    """Lifecycle status of an eval (and, historically, of a spec)."""

    active = "active"
    future = "future"
    deprecated = "deprecated"
    archived = "archived"


# Only one rating type for now, but this allows for extensibility if we want to add more in the future
class TaskOutputRatingType(str, Enum):
    """Defines the types of rating systems available for task outputs."""

    five_star = "five_star"
    pass_fail = "pass_fail"
    pass_fail_critical = "pass_fail_critical"
    custom = "custom"


class StructuredOutputMode(str, Enum):
    """
    Enumeration of supported structured output modes.

    - json_schema: request json using API capabilities for json_schema
    - function_calling: request json using API capabilities for function calling
    - json_mode: request json using API's JSON mode, which should return valid JSON, but isn't checking/passing the schema
    - json_instructions: append instructions to the prompt to request json matching the schema. No API capabilities are used. You should have a custom parser on these models as they will be returning strings.
    - json_instruction_and_object: append instructions to the prompt to request json matching the schema. Also request the response as json_mode via API capabilities (returning dictionaries).
    - json_custom_instructions: The model should output JSON, but custom instructions are already included in the system prompt. Don't append additional JSON instructions.
    - default: let the adapter decide (legacy, do not use for new use cases)
    - unknown: used for cases where the structured output mode is not known (on old models where it wasn't saved). Should lookup best option at runtime.
    """

    default = "default"
    json_schema = "json_schema"
    function_calling_weak = "function_calling_weak"
    function_calling = "function_calling"
    json_mode = "json_mode"
    json_instructions = "json_instructions"
    json_instruction_and_object = "json_instruction_and_object"
    json_custom_instructions = "json_custom_instructions"
    unknown = "unknown"


class FineTuneStatusType(str, Enum):
    """
    The status type of a fine-tune job.
    """

    unknown = "unknown"
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class TurnMode(str, Enum):
    """Whether a Task runs as a single turn or as a multiturn conversation."""

    single_turn = "single_turn"
    multiturn = "multiturn"


class ChatStrategy(str, Enum):
    """Strategy for how a chat is structured."""

    # Single turn, immediately return the answer
    single_turn = "final_only"
    # Two turn, first turn is the thinking, second turn is the answer. Legacy format - used for old fine tunes but not new trains.
    two_message_cot_legacy = "final_and_intermediate"
    # Two turn, first turn is the thinking, second turn is the answer. New format - used for new trains.
    two_message_cot = "two_message_cot"
    # Single turn, with both the thinking and the answer in the same message, using R1-style thinking format in <think> tags
    single_turn_r1_thinking = "final_and_intermediate_r1_compatible"


THINKING_DATA_STRATEGIES: list[ChatStrategy] = [
    ChatStrategy.two_message_cot_legacy,
    ChatStrategy.single_turn_r1_thinking,
    ChatStrategy.two_message_cot,
]


class FeedbackSource(str, Enum):
    """Where a piece of feedback originated.

    This is an append-only enum: new sources can be added freely, but existing
    values must never be removed or renamed so that older persisted data
    continues to load.
    """

    run_page = "run-page"
    spec_feedback = "spec-feedback"


class ModelProviderName(str, Enum):
    """
    Enumeration of supported AI model providers.
    """

    openai = "openai"
    groq = "groq"
    amazon_bedrock = "amazon_bedrock"
    ollama = "ollama"
    openrouter = "openrouter"
    fireworks_ai = "fireworks_ai"
    kiln_fine_tune = "kiln_fine_tune"
    kiln_custom_registry = "kiln_custom_registry"
    openai_compatible = "openai_compatible"
    anthropic = "anthropic"
    gemini_api = "gemini_api"
    azure_openai = "azure_openai"
    huggingface = "huggingface"
    vertex = "vertex"
    together_ai = "together_ai"
    siliconflow_cn = "siliconflow_cn"
    cerebras = "cerebras"
    docker_model_runner = "docker_model_runner"
    featherless_ai = "featherless_ai"


class KilnMimeType(str, Enum):
    """
    Enumeration of supported mime types.
    """

    # documents
    PDF = "application/pdf"
    CSV = "text/csv"
    TXT = "text/plain"
    HTML = "text/html"
    MD = "text/markdown"

    # images
    PNG = "image/png"
    JPG = "image/jpeg"
    JPEG = "image/jpeg"

    # audio
    MP3 = "audio/mpeg"
    WAV = "audio/wav"
    OGG = "audio/ogg"

    # video
    MP4 = "video/mp4"
    MOV = "video/quicktime"
