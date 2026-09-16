import json
import math
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    GetJsonSchemaHandler,
    JsonValue,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    ValidationInfo,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema
from typing_extensions import Self

from kiln_ai.datamodel.basemodel import (
    ID_TYPE,
    FilenameString,
    FilenameStringShort,
    KilnParentedModel,
    KilnParentModel,
)
from kiln_ai.datamodel.code_file_storage import (
    read_code_from_sibling_file,
    write_code_to_sibling_file,
)
from kiln_ai.datamodel.datamodel_enums import (
    EvalStatus,
    Priority,
    TaskOutputRatingType,
)
from kiln_ai.datamodel.dataset_filters import DatasetFilterId, EvalInputFilterId
from kiln_ai.datamodel.json_schema import string_to_json_key
from kiln_ai.datamodel.task_run import Usage
from kiln_ai.datamodel.tool_id import ToolId, validate_tool_allowlist
from kiln_ai.utils.exhaustive_error import raise_exhaustive_enum_error

if TYPE_CHECKING:
    from kiln_ai.datamodel.spec import Spec
    from kiln_ai.datamodel.task import Task
    from kiln_ai.datamodel.task_run import TaskRun

EvalScores = Dict[str, float]

# Module-level set to track evals currently being migrated (to prevent recursion)
# Protected by _migration_lock to ensure thread-safe access
_migration_lock = Lock()
_currently_migrating_eval_ids: set[ID_TYPE] = set()

# Fixed name of the sibling file that holds a code judge's Python source, stored
# beside its eval_config.kiln. Fixed so authored tests can `from scorer import score`.
SCORER_CODE_FILENAME = "scorer.py"


class EvalTemplateId(str, Enum):
    """
    An eval template is a pre-defined eval that can be used as a starting point for a new eval.
    """

    kiln_requirements = "kiln_requirements"
    desired_behaviour = "desired_behaviour"
    issue = "kiln_issue"
    tool_call = "tool_call"
    toxicity = "toxicity"
    bias = "bias"
    maliciousness = "maliciousness"
    factual_correctness = "factual_correctness"
    jailbreak = "jailbreak"
    rag = "rag"


class EvalConfigType(str, Enum):
    """The type of eval configuration, determining how scores are generated."""

    g_eval = "g_eval"
    llm_as_judge = "llm_as_judge"
    v2 = "v2"


class V2EvalType(str, Enum):
    """V2-only eval type enum. Each value maps to a typed properties class
    and a V2 adapter."""

    llm_judge = "llm_judge"
    exact_match = "exact_match"
    pattern_match = "pattern_match"
    set_check = "set_check"
    tool_call_check = "tool_call_check"
    contains = "contains"
    step_count_check = "step_count_check"
    code_eval = "code_eval"


class LlmJudgeProperties(BaseModel):
    type: Literal[V2EvalType.llm_judge] = V2EvalType.llm_judge
    model_name: str
    model_provider: str
    system_prompt: str | None = None
    prompt_template: str
    reference_keys: list[str] = []
    thinking_instruction: str | None = None
    g_eval: bool = False
    # User-written evaluation steps, bound to {{ judge_instructions }} when the
    # prompt template is rendered. Used by evals with no spec or template to
    # derive default steps from.
    judge_instructions: list[str] | None = None


class ExactMatchProperties(BaseModel):
    type: Literal[V2EvalType.exact_match] = V2EvalType.exact_match
    value_expression: str | None = None
    expected_value: str | None = None
    reference_key: str | None = Field(default=None, min_length=1)
    case_sensitive: bool = True

    @model_validator(mode="after")
    def validate_value_source(self) -> Self:
        if (self.expected_value is None) == (self.reference_key is None):
            raise ValueError(
                "Exactly one of expected_value or reference_key must be set"
            )
        return self


class PatternMatchProperties(BaseModel):
    type: Literal[V2EvalType.pattern_match] = V2EvalType.pattern_match
    value_expression: str | None = None
    pattern: str
    mode: Literal["must_match", "must_not_match"] = "must_match"

    @model_validator(mode="after")
    def validate_pattern(self) -> Self:
        import re

        try:
            re.compile(self.pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{self.pattern}': {e}") from e
        return self


class ContainsProperties(BaseModel):
    type: Literal[V2EvalType.contains] = V2EvalType.contains
    value_expression: str | None = None
    substring: str | None = None
    reference_key: str | None = Field(default=None, min_length=1)
    case_sensitive: bool = True
    mode: Literal["must_contain", "must_not_contain"] = "must_contain"

    @model_validator(mode="after")
    def validate_value_source(self) -> Self:
        if (self.substring is None) == (self.reference_key is None):
            raise ValueError("Exactly one of substring or reference_key must be set")
        return self


class SetCheckProperties(BaseModel):
    type: Literal[V2EvalType.set_check] = V2EvalType.set_check
    value_expression: str | None = None
    expected_set: list[str] | None = None
    reference_key: str | None = Field(default=None, min_length=1)
    mode: Literal["subset", "superset", "equal"]

    @model_validator(mode="after")
    def validate_value_source(self) -> Self:
        if (self.expected_set is None) == (self.reference_key is None):
            raise ValueError("Exactly one of expected_set or reference_key must be set")
        return self


class ArgMatch(BaseModel):
    value: JsonValue
    match_mode: Literal["exact", "contains", "regex"] = "exact"

    @model_validator(mode="after")
    def validate_regex(self) -> Self:
        if self.match_mode == "regex":
            import re

            try:
                re.compile(str(self.value))
            except re.error as e:
                raise ValueError(f"Invalid regex value '{self.value}': {e}") from e
        return self


class ToolCallSpec(BaseModel):
    tool_name: str
    expected_args: dict[str, ArgMatch] | None = None


class ToolCallCheckProperties(BaseModel):
    type: Literal[V2EvalType.tool_call_check] = V2EvalType.tool_call_check
    expected_tools: list[ToolCallSpec] = Field(min_length=1)
    match_mode: Literal["any", "all", "ordered", "never"] = "all"
    on_unexpected_tools: Literal["ignore", "fail"] = "ignore"


class StepCountCheckProperties(BaseModel):
    type: Literal[V2EvalType.step_count_check] = V2EvalType.step_count_check
    count_type: Literal["tool_calls", "model_responses", "turns"]
    min_count: int | None = None
    max_count: int | None = None

    @model_validator(mode="after")
    def check_bounds(self) -> Self:
        if self.min_count is None and self.max_count is None:
            raise ValueError(
                "step_count_check requires at least one of min_count / max_count"
            )
        if (
            self.min_count is not None
            and self.max_count is not None
            and self.min_count > self.max_count
        ):
            raise ValueError("min_count must be <= max_count")
        return self


class CodeEvalProperties(BaseModel):
    type: Literal[V2EvalType.code_eval] = V2EvalType.code_eval
    code: str
    reference_keys: list[str] = []
    timeout_seconds: int = Field(default=180, ge=1, le=300)
    tool_allowlist: list[ToolId] = Field(
        default_factory=list,
        description="Explicit per-tool allowlist of tools the scorer code may call.",
    )

    @model_validator(mode="after")
    def validate_allowlist(self) -> Self:
        # No self-reference check: a code eval is not itself a tool.
        validate_tool_allowlist(self.tool_allowlist, caller="code evals")
        return self

    @model_validator(mode="before")
    @classmethod
    def _read_code_file(cls, data: Any, info: ValidationInfo) -> Any:
        """When loading from disk, inject `code` from the sibling scorer.py.

        The source is stored in scorer.py beside eval_config.kiln, not inline in
        the JSON. CodeEvalProperties is a nested member of the
        V2EvalConfigProperties discriminated union in EvalConfig.properties, so
        the load context set on the parent EvalConfig (`source_dir`) propagates
        down to this validator. The shared helper reads the file here, before
        field validation, so the existing validate_code trio runs against the
        loaded string unchanged.
        """
        # Explicit type-gate (defense-in-depth): this validator only ever runs
        # for code_eval properties — it lives on CodeEvalProperties, and both the
        # discriminated union and the eager parse route only code_eval dicts
        # here. Assert that gate so a future refactor can't quietly read
        # scorer.py for another eval type. None (type omitted, field defaults)
        # and the enum form both pass; only a present, mismatched type is
        # rejected, so valid-input behavior is unchanged.
        if isinstance(data, dict) and data.get("type") not in (
            None,
            V2EvalType.code_eval.value,
        ):
            raise ValueError(
                "CodeEvalProperties can only load code_eval properties, "
                f"got type: {data.get('type')!r}"
            )
        return read_code_from_sibling_file(
            data,
            info.context or {},
            filename=SCORER_CODE_FILENAME,
            kiln_filename="eval_config.kiln",
            model_label="CodeEvalProperties",
        )

    @model_serializer(mode="wrap")
    def _serialize(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, Any]:
        """On disk-save, write `code` to scorer.py and omit it from the .kiln JSON.

        Delegates to the shared sibling-file helper, which uses the same save
        context attachments use (`save_attachments` + `dest_path`); it propagates
        from the parent EvalConfig's save_to_file() down to this nested union
        member. Without that context — normal model_dump / API responses —
        `code` is left in the output and no file is written, so the API contract
        is unchanged. The default handler preserves `type` (needed by the
        discriminator), `reference_keys`, and `timeout_seconds`.

        Schema note: a custom model_serializer would otherwise collapse the
        *serialization-mode* JSON schema to an untyped object
        (`model_json_schema(mode="serialization")` loses per-field typing).
        Unlike CodeTool — which is never a FastAPI response_model — this model is
        nested in EvalConfig, and EvalConfig IS the declared `response_model` on
        several endpoints (eval_api.py). FastAPI generates response schemas in
        serialization mode, so a collapsed schema here would split
        CodeEvalProperties into an untyped `-Output` component and drift the
        checked-in api_schema.d.ts (breaking check_schema.sh and the web types
        that key off `components["schemas"]["CodeEvalProperties"]`). The
        `__get_pydantic_json_schema__` override below is therefore REQUIRED (not
        optional): it keeps the serialization-mode schema identical to
        validation mode. Do not remove either the serializer (runtime file
        storage) or the override (schema stability).
        """
        return write_code_to_sibling_file(
            handler(self),
            info.context or {},
            filename=SCORER_CODE_FILENAME,
            code=self.code,
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Keep the serialization-mode JSON schema identical to validation mode.

        The wrap serializer above returns an untyped `dict`, which would collapse
        this model's serialization-mode JSON schema to `{additionalProperties:
        true, type: object}` (dropping `code`, `type`, etc.). Because EvalConfig
        (which nests this model) is a FastAPI response_model, that collapse would
        drift the committed OpenAPI/api_schema.d.ts. Dropping the `serialization`
        core-schema entries makes JSON-schema generation use the field-based
        (validation) representation in both modes, so `code` stays present and
        typed and there is no `-Input`/`-Output` split. The custom serializer
        lives on the inner `model` core schema (the before/after validators wrap
        it in function schemas), so the strip must be recursive. This affects
        only schema generation, never runtime (de)serialization.
        """

        def strip_serialization(schema: Any) -> Any:
            if isinstance(schema, dict):
                return {
                    key: strip_serialization(value)
                    for key, value in schema.items()
                    if key != "serialization"
                }
            if isinstance(schema, list):
                return [strip_serialization(item) for item in schema]
            return schema

        return handler(strip_serialization(core_schema))

    @model_validator(mode="after")
    def validate_code(self) -> Self:
        code_bytes = self.code.encode("utf-8")
        if len(code_bytes) > 64 * 1024:
            raise ValueError(
                f"Code is too large ({len(code_bytes)} bytes). Maximum size is 64KB."
            )

        try:
            compile(self.code, "<code_eval>", "exec")
        except SyntaxError as e:
            raise ValueError(f"Code has a syntax error: {e}") from e

        import ast

        tree = ast.parse(self.code)
        # Both sync and async score functions are accepted here.
        # Async coroutines are transparently awaited in sandbox_worker.execute_scorer_bridged.
        has_score_fn = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "score"
            for node in ast.iter_child_nodes(tree)
        )
        if not has_score_fn:
            raise ValueError(
                "Code must define a module-level 'score' function (def score(...))."
            )

        return self


V2EvalConfigProperties = Annotated[
    Union[
        LlmJudgeProperties,
        ExactMatchProperties,
        PatternMatchProperties,
        SetCheckProperties,
        ToolCallCheckProperties,
        ContainsProperties,
        StepCountCheckProperties,
        CodeEvalProperties,
    ],
    Discriminator("type"),
]

# Parses a raw properties dict into its typed V2 class via the "type" discriminator.
_V2_PROPERTIES_ADAPTER: TypeAdapter[Any] = TypeAdapter(V2EvalConfigProperties)

# Explicit tuple of V2 property types for isinstance() checks.
# Must list exactly the same types as the V2EvalConfigProperties union above.
V2_PROPERTY_TYPES: tuple[type[BaseModel], ...] = (
    LlmJudgeProperties,
    ExactMatchProperties,
    PatternMatchProperties,
    SetCheckProperties,
    ToolCallCheckProperties,
    ContainsProperties,
    StepCountCheckProperties,
    CodeEvalProperties,
)


def reference_data_keys(props: V2EvalConfigProperties) -> list[str]:
    """Return the reference-data keys a single judge needs.

    Exhaustive match over the V2 properties union: adding a new V2 type
    without handling it here will fail ``ty`` type-checking.
    """
    match props:
        case ExactMatchProperties():
            return [props.reference_key] if props.reference_key else []
        case ContainsProperties():
            return [props.reference_key] if props.reference_key else []
        case SetCheckProperties():
            return [props.reference_key] if props.reference_key else []
        case LlmJudgeProperties():
            return list(props.reference_keys)
        case CodeEvalProperties():
            return list(props.reference_keys)
        case PatternMatchProperties():
            return []
        case ToolCallCheckProperties():
            return []
        case StepCountCheckProperties():
            return []
        case _:
            raise_exhaustive_enum_error(props)


def _eager_parse_code_eval_on_load(
    data: dict[str, Any], ctx: dict[str, Any]
) -> dict[str, Any]:
    """Eagerly parse a code_eval EvalConfig's `properties` on file load.

    V2 code judges store their score() source in a sibling scorer.py, not inline
    in the JSON. On load, parse a code_eval properties dict through
    CodeEvalProperties (which reads scorer.py via the propagated load context) so
    any error surfaces directly. Without this, the outer
    `V2EvalConfigProperties | dict | None` union would recover from the nested
    member's error by falling back to the dict branch, masking the real cause
    (e.g. a missing scorer.py or a bad score() function) behind a generic
    "V2 config requires typed properties". See functional spec §2.2 / §4.

    Only touches code_eval properties during a file load, gated explicitly on
    `type == code_eval`; every other input passes through unchanged. Lifted
    verbatim from EvalConfig.dispatch_properties_parsing so the code-eval load
    path is a clearly-named, code-eval-local step rather than smeared into the
    generic dispatcher.
    """
    if not ctx.get("loading_from_file"):
        return data
    props = data.get("properties")
    if isinstance(props, dict) and props.get("type") == V2EvalType.code_eval.value:
        data = dict(data)
        data["properties"] = CodeEvalProperties.model_validate(props, context=ctx)
    return data


def validate_scores_against_output_scores(
    scores: EvalScores,
    output_scores: list["EvalOutputScore"],
) -> list[str]:
    """Validate that *scores* fall within the expected range for each output score.

    Returns a list of human-readable problem strings (empty list means all OK).
    This is a pure function — it does NOT raise; callers decide how to surface errors.
    """

    def _is_numeric(v: object) -> bool:
        # NaN compares False against every range bound, so it passes every check
        # below, then serializes to null and makes the saved file fail on reload.
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False
        try:
            return math.isfinite(v)
        except OverflowError:
            # isfinite coerces int args to float, which an int like 10**400 can't be.
            return False

    problems: list[str] = []
    for output_score in output_scores:
        key = output_score.json_key()
        if key not in scores:
            continue
        value = scores[key]

        match output_score.type:
            case TaskOutputRatingType.five_star:
                if not _is_numeric(value) or value < 1.0 or value > 5.0:
                    problems.append(
                        f"Score {output_score.name} is a five_star rating and must be a number between 1.0 and 5.0 inclusive. Got: {value}"
                    )
            case TaskOutputRatingType.pass_fail:
                if not _is_numeric(value) or value < 0.0 or value > 1.0:
                    problems.append(
                        f"Score {output_score.name} is a pass_fail rating and must be a number between 0.0 and 1.0 inclusive. Got: {value}"
                    )
            case TaskOutputRatingType.pass_fail_critical:
                if not _is_numeric(value) or value < -1.0 or value > 1.0:
                    problems.append(
                        f"Score {output_score.name} is a pass_fail_critical rating and must be a number between -1.0 and 1.0 inclusive. Got: {value}"
                    )
            case TaskOutputRatingType.custom:
                problems.append(
                    f"Custom scores are not supported in evaluators. '{output_score.name}' was set to a custom score."
                )
            case _:
                raise_exhaustive_enum_error(output_score.type)
    return problems


class SkippedReason(str, Enum):
    """Terminal skip reasons stored as str for back/forward-compat."""

    missing_reference_key = "missing_reference_key"
    extraction_failed = "extraction_failed"
    missing_trace = "missing_trace"
    missing_drive_config = "missing_drive_config"
    incompatible_input_shape = "incompatible_input_shape"
    code_eval_not_trusted = "code_eval_not_trusted"
    type_not_available = "type_not_available"


class V2EvalResult(BaseModel):
    """Result of a single V2 eval ``evaluate()`` call."""

    scores: EvalScores = Field(default_factory=dict)
    skipped_reason: SkippedReason | None = None
    skipped_detail: str | None = None
    intermediate_outputs: Dict[str, str] | None = None
    usage: Usage | None = Field(
        default=None,
        description="What the judgment itself cost, if it called a model. None for the deterministic eval types, which call none. Stored on the resulting EvalRun as eval_usage.",
    )


class UserMessage(BaseModel):
    text: str


class SyntheticUserInfo(BaseModel):
    """The synthetic user's character sheet: who they are and what they want.

    This is both the persisted form on multi-turn synthetic eval inputs and
    the runtime shape the synthetic-user driver renders its system prompt
    from. The XML-tagged blob some wire formats carry is parsed into this at
    the wire boundary (kiln_ai.synthetic_user.parser) — it is never stored.

    extra="allow": unknown fields from newer generators survive load/save
    round-trips instead of being dropped.
    """

    model_config = ConfigDict(extra="allow")

    persona: str
    goal: str
    behavior_guidance: str | None = None


class SingleTurnEvalInputData(BaseModel):
    type: Literal["single_turn"] = "single_turn"
    user_message: UserMessage


class MultiTurnDriveConfig(BaseModel):
    """Settings for re-driving a multi-turn synthetic input at eval time.

    A multi-turn eval run regenerates each conversation: the agent under test
    comes from the run config being evaluated, while the synthetic user
    (customer) configured here is held constant across run configs — so a
    comparison varies only the agent. Stored per item, on
    MultiTurnSyntheticEvalInputData.drive_config.
    """

    model_name: str = Field(
        description="The model that plays the synthetic user during re-drives."
    )
    # A plain string rather than the provider enum so persisted items load on
    # builds that don't know the provider yet (same choice as LlmJudgeProperties).
    model_provider: str = Field(description="The provider of the synthetic-user model.")
    turns: int = Field(
        ge=1,
        le=20,
        description="Ceiling on the assistant turns per re-driven conversation.",
    )


class MultiTurnSyntheticEvalInputData(BaseModel):
    """A re-drivable multi-turn case: the opening user message, the synthetic
    user who continues the conversation at eval time, and the drive settings
    that synthetic user runs with.

    Together these make the item a self-contained replication recipe: with the
    persona, first_message, and drive_config it re-drives identically under any
    eval that references it, which is what makes conversation traces keyed to
    the item reusable across evals.

    first_message may be None; such items carry no seed to open a
    conversation with, so the eval runner skips them instead of re-driving.
    """

    type: Literal["multi_turn_synthetic"] = "multi_turn_synthetic"
    first_message: UserMessage | None = None
    synthetic_user_info: SyntheticUserInfo
    drive_config: MultiTurnDriveConfig | None = Field(
        default=None,
        description="How this item's conversation is re-driven: the "
        "synthetic-user model and turn count, stamped when the item is minted. "
        "This is the ONLY home for drive settings — no eval-level copy exists; "
        "displays and prefills derive from items. Held constant across run "
        "configs so a comparison varies only the agent under test. Immutable "
        "once minted: changing the synthetic-user setup means minting new "
        "items, which keeps traces keyed to this item valid. None only on "
        "items minted before drive settings were stamped; the eval runner "
        "skips such items with a clear reason rather than guessing a config.",
    )

    @model_serializer(mode="wrap")
    def _omit_unset_drive_config(
        self, handler: SerializerFunctionWrapHandler
    ) -> Dict[str, Any]:
        # Items that predate drive_config carry no key on disk, and absent and
        # null load identically — omit an unset config instead of churning
        # every legacy file with a null on resave.
        data: Dict[str, Any] = handler(self)
        if data.get("drive_config") is None:
            data.pop("drive_config", None)
        return data


EvalInputData = Annotated[
    Union[
        SingleTurnEvalInputData,
        MultiTurnSyntheticEvalInputData,
    ],
    Discriminator("type"),
]


class EvalInput(KilnParentedModel):
    """A single evaluation input item, stored as a child of a Task.

    Each EvalInput contains the data needed to run an evaluation (e.g. a user
    message) plus optional reference data for comparison and tags for filtering.
    """

    data: EvalInputData = Field(
        description="The input data for this eval item.",
    )
    reference: dict[str, JsonValue] | None = Field(
        default=None,
        description="Optional reference data (ground truth) for this eval input, keyed by reference name.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for filtering eval inputs.",
    )

    @model_validator(mode="after")
    def validate_tags(self) -> Self:
        # Empty or space-containing tags can't be selected by tag filters, so
        # reject them at creation instead of silently dropping the item later.
        for tag in self.tags:
            if not tag:
                raise ValueError("Tags cannot be empty strings")
            if " " in tag:
                raise ValueError("Tags cannot contain spaces. Try underscores.")
        return self


class EvalTaskInput(BaseModel):
    """The runtime data bundle passed to V2 evaluators.

    Assembled by the eval runner from the item being evaluated and the task run that
    was scored. The item is either an EvalInput or a TaskRun drawn from the dataset;
    which one it is determines where `reference_data` and `task_input` come from.
    """

    final_message: str = Field(
        description="The final model output (task output text).",
    )
    trace: list[dict[str, Any]] | None = Field(
        default=None,
        description="The full conversation trace, if available.",
    )
    reference_data: dict[str, JsonValue] | None = Field(
        default=None,
        description=(
            "Ground-truth data for the item being evaluated, keyed by reference name. "
            "Taken from EvalInput.reference for an EvalInput-backed item; for a "
            "TaskRun-backed dataset item it is the item's own stored output under the "
            "key 'reference_answer', since that output is the curated answer. None "
            "when a TaskRun is scored as itself (judge calibration), where the item "
            "and the scored run are the same record."
        ),
    )
    task_input: str | None = Field(
        default=None,
        description="The original task input text.",
    )

    @classmethod
    def from_trace(
        cls, trace: "TaskRun", source: "TaskRun | EvalInput"
    ) -> "EvalTaskInput":
        """What a judge sees: the trace that was produced, plus the item it came from.

        The two are separate arguments because they are separate records once eval traces
        live on their own TaskRun — the trace holds what the model said, and the source
        item holds the ground truth to compare it against. They are the same object only
        for calibration, where the golden dataset item is itself what gets scored.
        """
        from kiln_ai.datamodel.task_run import TaskRun as _TaskRun

        if not isinstance(trace, _TaskRun):
            raise TypeError("Expected a TaskRun instance for trace")

        trace_data: list[dict[str, Any]] | None = None
        if trace.trace is not None:
            trace_data = [dict(msg) for msg in trace.trace]

        if isinstance(source, EvalInput):
            reference_data = source.reference
            # The item's own text, not the trace's: an EvalInput is the canonical
            # statement of the input, and the adapter may have reserialized it.
            if isinstance(source.data, SingleTurnEvalInputData):
                task_input = source.data.user_message.text
            elif isinstance(source.data, MultiTurnSyntheticEvalInputData):
                # Multi-turn: the first message opened the conversation, and the
                # rest of the exchange lives in the trace. Items minted without a
                # first message have no canonical input text to offer.
                task_input = (
                    source.data.first_message.text
                    if source.data.first_message
                    else None
                )
            else:
                raise ValueError(
                    f"Unsupported EvalInput data type: {type(source.data).__name__}"
                )
        elif isinstance(source, _TaskRun):
            # A TaskRun-backed dataset item stores the curated answer as its output, so
            # that output is the ground truth to compare the trace against. Skipped when
            # source *is* trace (calibration, and `from_task_run`): there the golden item
            # is itself what gets scored, so a reference would be byte-identical to
            # `final_message` and every judge comparing them would pass.
            reference_data = (
                None if source is trace else {"reference_answer": source.output.output}
            )
            task_input = trace.input
        else:
            raise TypeError("Expected a TaskRun or EvalInput instance for source")

        return cls(
            final_message=trace.output.output,
            trace=trace_data,
            reference_data=reference_data,
            task_input=task_input,
        )

    @classmethod
    def from_task_run(cls, task_run: "TaskRun") -> "EvalTaskInput":
        """A TaskRun scored as itself, with no separate source item."""
        return cls.from_trace(task_run, task_run)

    @classmethod
    def from_eval_input(
        cls, eval_input: "EvalInput", run_output: "TaskRun"
    ) -> "EvalTaskInput":
        """A generated run scored against the EvalInput it was generated from.

        The argument-order counterpart of `from_trace` for callers holding the item
        first. The explicit type check stays: passed a TaskRun by mistake,
        `from_trace` would silently take its TaskRun branch instead of failing.
        """
        if not isinstance(eval_input, EvalInput):
            raise TypeError("Expected an EvalInput instance")
        return cls.from_trace(run_output, eval_input)


class EvalOutputScore(BaseModel):
    """
    A definition of a score that an evaluator will produce.

    Very similar to TaskRequirement, but conceptually different keeping in a separate models.
    """

    name: FilenameStringShort = Field(
        description="The name of the score. Will be provided to the model so use a descriptive name. Should align to the model's TaskRequirement name if you want to use human evals to evaluate the evaluator's performance."
    )
    instruction: str | None = Field(
        default=None,
        description="A description of the score, used to help the model understand the goal of the score. Will be provided to evaluator models, so should be written for the model, not the team/user.",
    )
    type: TaskOutputRatingType = Field(
        description="The type of rating to use ('five_star', 'pass_fail', 'pass_fail_critical').",
    )

    def json_key(self) -> str:
        """
        The JSON key for the score, used when running the evaluator with a LLM and we need JSON output.

        For example, "Overall Rating" -> "overall_rating"
        """
        return string_to_json_key(self.name)

    @model_validator(mode="after")
    def validate_type(self) -> Self:
        if self.type == TaskOutputRatingType.custom:
            raise ValueError(
                f"Custom scores are not supported in evaluators. Score '{self.name}' was set to a custom score."
            )
        return self


LEGACY_TRACE_FIELDS = ("output", "task_run_trace", "task_run_usage", "reference_answer")
"""The EvalRun fields that hold a copy of the trace a score was computed over.

Deprecated: new records point at a TaskRun with `scored_run_id` instead. Kept declared
and loadable forever - every record written before the split still carries them. A
pointer-mode record must leave all of them None, which `validate_record_mode` enforces.
`input` is deprecated alongside these but is not in this tuple: it is the one a legacy
record is *required* to have, so it is checked separately.

Three ways to mark a pydantic field deprecated; these fields use two of them:

1. A `DEPRECATED:` prefix in the `description`. Reaches a human reading the SDK docs or
   the OpenAPI schema, and nothing else. Used.
2. `json_schema_extra={"deprecated": True}`. Puts `"deprecated": true` in the JSON
   schema, which `openapi-typescript` turns into a `/** @deprecated */` JSDoc tag, so
   the TS compiler and editors strike through every web call site. No runtime effect.
   Used.
3. `Field(deprecated=True)`. Same schema output as (2), but pydantic also raises a
   DeprecationWarning on every attribute *read* — and reading these is the correct,
   permanent way to render a legacy record, so it would be a warning storm. Not used.
   Do not "fix" this to (3) without silencing that first; (2) already provides the
   tooling signal (3) would be reached for."""


class EvalRun(KilnParentedModel):
    """
    The scores an eval produced for a single dataset item.

    A run serves one of two purposes:
    - eval_config_eval=False (scoring): evaluating a task run config — the item's
      input was run through the task with task_run_config_id (which must be set)
      and the evaluator scored that output.
    - eval_config_eval=True (calibration): evaluating the eval config itself — an
      existing human-rated dataset item's input and output were scored so the
      evaluator can be compared against those human ratings. task_run_config_id
      must be None.

    A record is described by two independent facts — whether it points at a TaskRun, and
    whether it was skipped — which `validate_record_mode` constrains to three legal
    shapes. What is exclusive is where the trace lives: on the record, or on the TaskRun,
    never both.

    - **Pointer** (new): `scored_run_id` names the TaskRun that holds the trace. All
      inline trace fields must be None.
    - **Skipped**: `skipped_reason` set, so scores are not required. It also carries a
      `scored_run_id` if the trace existed and only scoring was skipped — so a skip can
      be a pointer record too — and none if the skip happened before generation.
    - **Legacy inline**: no `scored_run_id`; the trace lives on this record, and `input`
      is required unless the record was skipped. Every record written before the
      trace/score split is in this state, and it stays valid forever.
    """

    dataset_id: ID_TYPE | None = Field(
        default=None,
        description="The ID of the dataset item (TaskRun) that was used for this run. Mutually exclusive with eval_input_id.",
    )
    scored_run_id: ID_TYPE | None = Field(
        default=None,
        description="The ID of the TaskRun this score was computed over. None for legacy records that carry their trace inline. A dangling reference is tolerated: the score still renders and still aggregates, only the trace drill-through is unavailable.",
    )
    task_run_config_id: ID_TYPE | None = Field(
        description="The ID of the TaskRunConfig that was run, if this eval run was based on a task run. Must belong to the same Task as this eval. Can be None if this eval run is based on an eval config."
    )
    eval_config_eval: bool = Field(
        description="Whether this eval run to evaluate the parent eval config (evaluating the config using an existing dataset item). If true, task_run_config_id must be None, as we're not running the task.",
        default=False,
    )
    input: str | None = Field(
        default=None,
        json_schema_extra={"deprecated": True},
        description="DEPRECATED: the trace now lives on the TaskRun named by scored_run_id; read TaskRun.input instead. The input to the task. JSON formatted for structured input, plaintext for unstructured input. Required on legacy records (those with neither a scored_run_id nor a skipped_reason), never set on new ones.",
    )
    output: str | None = Field(
        default=None,
        json_schema_extra={"deprecated": True},
        description="DEPRECATED: the trace now lives on the TaskRun named by scored_run_id; read TaskRun.output.output instead. The output of the task. None for skipped-before-execution runs.",
    )
    reference_answer: str | None = Field(
        default=None,
        json_schema_extra={"deprecated": True},
        description="DEPRECATED: the trace now lives on the TaskRun named by scored_run_id. The reference answer for the input. JSON formatted for structured reference answer, plaintext for unstructured reference answer. Used for reference answer evals.",
    )
    intermediate_outputs: Dict[str, str] | None = Field(
        default=None,
        description="The intermediate outputs of the task (example, eval thinking).",
    )
    task_run_trace: str | None = Field(
        default=None,
        json_schema_extra={"deprecated": True},
        description="DEPRECATED: the trace now lives on the TaskRun named by scored_run_id; read TaskRun.trace instead. The JSON formatted trace of the task run that produced the output.",
    )
    scores: EvalScores = Field(
        default={},
        description="The output scores of the evaluator (aligning to those required by the grand-parent Eval this object is a child of).",
    )
    task_run_usage: Usage | None = Field(
        default=None,
        json_schema_extra={"deprecated": True},
        description="DEPRECATED: the trace now lives on the TaskRun named by scored_run_id; read TaskRun.usage instead. The usage of the task run that produced this eval run output (not the usage by the evaluation model).",
    )
    eval_usage: Usage | None = Field(
        default=None,
        description="The usage of the evaluation model (judge) that produced this eval run's scores, aggregated across every LLM call the judgment made. Distinct from task_run_usage, which is the evaluated task run's usage. None for non-LLM evals (e.g. code evals) and for records that predate this field.",
    )

    eval_input_id: ID_TYPE | None = Field(
        default=None,
        description="ID of the EvalInput used for this run (V2 evals). Mutually exclusive with dataset_id.",
    )
    skipped_reason: str | None = Field(
        default=None,
        description="If set, this run was skipped. Stored as str for back/forward-compat; conventionally a SkippedReason value.",
    )
    skipped_detail: str | None = Field(
        default=None,
        description="Case-specific detail for skipped runs (e.g. missing key name).",
    )

    def parent_eval_config(self) -> Union["EvalConfig", None]:
        if self.parent is not None and self.parent.__class__.__name__ != "EvalConfig":
            raise ValueError("parent must be an EvalConfig")
        return self.parent  # type: ignore

    @model_validator(mode="after")
    def validate_input_source(self) -> Self:
        if (self.dataset_id is None) == (self.eval_input_id is None):
            raise ValueError(
                "Exactly one of dataset_id (V1 TaskRun source) or "
                "eval_input_id (V2 EvalInput source) must be set"
            )
        return self

    @model_validator(mode="after")
    def validate_record_mode(self) -> Self:
        """Keep the three record states (pointer / skipped / legacy inline) exclusive.

        The forbidding half of the pointer rule is the one that earns its keep: a record
        that points at a TaskRun must never also carry a second copy of what it scored,
        which nothing would keep in sync.
        """
        inline_set = [f for f in LEGACY_TRACE_FIELDS if getattr(self, f) is not None]

        if self.scored_run_id is not None:
            # Checked before the skip branch on purpose: a record skipped at *scoring*
            # time still has a scored_run_id, and still must not carry inline data.
            if self.input is not None or inline_set:
                carried = (["input"] if self.input is not None else []) + inline_set
                raise ValueError(
                    "An EvalRun with scored_run_id must not carry inline trace data "
                    f"(set: {', '.join(carried)}). "
                    "The trace lives on the referenced TaskRun."
                )
            return self

        if self.skipped_reason is not None:
            # Skipped before generation: there is nothing to point at, and nothing to
            # require. Legacy skipped records that do carry inline data stay valid.
            return self

        if self.input is None:
            raise ValueError("A legacy EvalRun (no scored_run_id) requires input.")
        return self

    @model_validator(mode="after")
    def validate_output_fields(self, info: ValidationInfo) -> Self:
        parent_eval_config = self.parent_eval_config()
        if self.scored_run_id is not None:
            # Pointer mode: the trace lives on the referenced TaskRun, and
            # validate_record_mode already forbids inline copies here. The checks
            # below are about data carried on this record, so none of them apply.
            return self
        parent_eval = parent_eval_config.parent_eval() if parent_eval_config else None
        if not parent_eval:
            return self

        evaluation_data_type = parent_eval.evaluation_data_type

        # A full_trace eval scores the conversation trace, so a successful task
        # run must carry it. Both V1 and V2 writers attach the trace for exactly
        # this shape (a scored, non-skipped task-run eval of a full_trace eval),
        # so demanding it back makes a writer that drops the trace fail loudly
        # instead of persisting a record that can't be re-scored. Historical
        # files predating this gate are exempt so they still load; new writes
        # and rebuilds are held to it.
        if (
            not self.eval_config_eval
            and self.skipped_reason is None
            and evaluation_data_type == EvalDataType.full_trace
            and self.task_run_trace is None
            and not self.loaded_from_file(info)
        ):
            raise ValueError("full_trace task run eval runs should include trace")

        # Remaining checks are V1-only. V2 deliberately relaxes them: skipped
        # runs carry no output, and V2 writers never attach a trace to a
        # final_answer run in the first place.
        if parent_eval_config.config_type == EvalConfigType.v2:
            return self

        if self.output is None and self.skipped_reason is None:
            raise ValueError("V1 EvalRun requires output to be set")

        if (
            evaluation_data_type == EvalDataType.final_answer
            and self.task_run_trace is not None
        ):
            raise ValueError("final_answer runs should not set trace")

        return self

    @model_validator(mode="after")
    def validate_eval_run_types(self) -> Self:
        if self.eval_config_eval and self.task_run_config_id is not None:
            raise ValueError(
                "task_run_config_id must be None if eval_config_eval is true"
            )
        if not self.eval_config_eval and self.task_run_config_id is None:
            raise ValueError(
                "task_run_config_id must be set if eval_config_eval is false"
            )
        if self.eval_config_eval and self.dataset_id is None:
            raise ValueError(
                "eval_config_eval records must score a dataset item: judge "
                "calibration compares against human ratings, which only "
                "dataset items (TaskRuns) carry"
            )
        return self

    @model_validator(mode="after")
    def validate_scores(self) -> Self:
        if self.skipped_reason is not None:
            return self

        if self.scores is None or len(self.scores) == 0:
            raise ValueError("scores are required, and must have at least one score.")

        parent_eval_config = self.parent_eval_config()
        eval = parent_eval_config.parent_eval() if parent_eval_config else None
        if not eval:
            return self

        output_score_keys = [score.json_key() for score in eval.output_scores]
        if set(output_score_keys) != set(self.scores.keys()):
            raise ValueError(
                f"The scores produced by the evaluator must match the scores expected by the eval. Got: [{', '.join(self.scores.keys())}] and expected: [{', '.join(output_score_keys)}]"
            )

        problems = validate_scores_against_output_scores(
            self.scores, eval.output_scores
        )
        if problems:
            raise ValueError(problems[0])
        return self

    @model_validator(mode="after")
    def validate_reference_answer(self) -> Self:
        parent_eval_config = self.parent_eval_config()
        if parent_eval_config and parent_eval_config.config_type == EvalConfigType.v2:
            return self
        parent_eval = parent_eval_config.parent_eval() if parent_eval_config else None
        if not parent_eval:
            return self

        evaluation_data_type = parent_eval.evaluation_data_type
        if (
            self.reference_answer is not None
            and evaluation_data_type is not None
            and evaluation_data_type != EvalDataType.reference_answer
        ):
            raise ValueError(
                f"reference_answer is only valid for reference answer evals. Got: {evaluation_data_type.value}"
            )
        return self


class EvalConfig(KilnParentedModel, KilnParentModel, parent_of={"runs": EvalRun}):
    """
    A configuration for running an eval. This includes anything needed to run the eval on a dataset like the prompt, model, thresholds, etc.

    A eval might have many configs, example running the same eval with 2 different models. Comparing eval results is only valid within the scope of the same config.
    """

    name: FilenameString = Field(description="The name of the eval config.")
    model_name: str | None = Field(
        default=None,
        description="The name of the model to use for this eval config. Required for legacy configs, None for V2.",
    )
    model_provider: str | None = Field(
        default=None,
        description="The provider of the model to use for this eval config. Required for legacy configs, None for V2.",
    )
    config_type: EvalConfigType = Field(
        default=EvalConfigType.g_eval,
        description="This is used to determine the type of eval to run.",
    )
    properties: dict[str, Any] | V2EvalConfigProperties | None = Field(
        default=None,
        description="Properties to be used to execute the eval config. Legacy configs use a dict; V2 configs use typed properties.",
    )

    @model_validator(mode="before")
    @classmethod
    def dispatch_properties_parsing(cls, data: Any, info: ValidationInfo) -> Any:
        # The union lists dict first, so a raw dict always stays a plain dict —
        # even one whose keys happen to match a typed V2 shape (legacy configs
        # store arbitrary dicts). V2 configs persist properties as a dict too,
        # so parse those into the typed union here, before field validation.
        if not isinstance(data, dict):
            return data
        if data.get("config_type", EvalConfigType.g_eval) == EvalConfigType.v2:
            # code_eval stores its score() source in a sibling scorer.py: on file
            # load, the type-gated helper parses those props through
            # CodeEvalProperties (reading the sibling via the load context) so a
            # bad or missing scorer.py surfaces directly instead of being masked
            # by a union fallback. Other property types pass through unchanged.
            data = _eager_parse_code_eval_on_load(data, info.context or {})
            props = data.get("properties")
            if isinstance(props, dict):
                data = dict(data)
                data["properties"] = _V2_PROPERTIES_ADAPTER.validate_python(props)
        return data

    def parent_eval(self) -> Union["Eval", None]:
        if self.parent is not None and self.parent.__class__.__name__ != "Eval":
            raise ValueError("parent must be an Eval")
        return self.parent  # type: ignore

    def runs(self, readonly: bool = False) -> list[EvalRun]:
        return super().runs(readonly=readonly)  # type: ignore

    @model_validator(mode="after")
    def validate_properties(self) -> Self:
        if self.config_type in (EvalConfigType.g_eval, EvalConfigType.llm_as_judge):
            if not isinstance(self.properties, dict):
                raise ValueError("Legacy config properties must be a dict")
            if "eval_steps" not in self.properties or not isinstance(
                self.properties["eval_steps"], list
            ):
                raise ValueError("eval_steps is required and must be a list for g_eval")
            if "task_description" in self.properties and not isinstance(
                self.properties["task_description"], str
            ):
                raise ValueError(
                    "task_description is optional, but if provided must be a string"
                )
            if self.model_name is None or self.model_provider is None:
                raise ValueError(
                    "model_name and model_provider are required for legacy configs"
                )
            return self
        elif self.config_type == EvalConfigType.v2:
            if not isinstance(self.properties, BaseModel):
                raise ValueError("V2 config requires typed properties")
            if self.model_name is not None or self.model_provider is not None:
                raise ValueError(
                    "V2 configs must not set root-level model_name/model_provider"
                )
            return self
        else:
            raise ValueError(f"Invalid eval config type: {self.config_type}")

    @model_validator(mode="after")
    def validate_v2_templates_and_expressions(self) -> Self:
        if self.config_type != EvalConfigType.v2 or not isinstance(
            self.properties, BaseModel
        ):
            return self

        from kiln_ai.utils.jinja_engine import (
            compile_expression_or_raise,
            compile_template_or_raise,
        )

        props = self.properties
        if isinstance(props, LlmJudgeProperties):
            compile_template_or_raise(props.prompt_template)
            from jinja2 import meta

            from kiln_ai.utils.jinja_engine import _template_env

            referenced = meta.find_undeclared_variables(
                _template_env.parse(props.prompt_template)
            )
            meaningful = {"final_message", "trace", "task_input"}
            if not (referenced & meaningful):
                raise ValueError(
                    "prompt_template never references the model output. "
                    "A template that uses only reference_data (or no variables) "
                    "produces the same judge prompt for every run. "
                    "Reference the output, e.g. {{ final_message }}."
                )

        if isinstance(
            props,
            (
                ExactMatchProperties,
                PatternMatchProperties,
                ContainsProperties,
                SetCheckProperties,
            ),
        ):
            if props.value_expression is not None:
                compile_expression_or_raise(props.value_expression)

        return self

    @model_validator(mode="after")
    def validate_json_serializable(self) -> "EvalConfig":
        if self.config_type == EvalConfigType.v2:
            return self
        if self.properties is None:
            return self
        try:
            json.dumps(self.properties, ensure_ascii=False)
        except TypeError as e:
            raise ValueError(f"Properties must be JSON serializable: {e!s}")
        return self


class EvalDataType(str, Enum):
    """The type of task output data to evaluate."""

    final_answer = "final_answer"
    full_trace = "full_trace"
    reference_answer = "reference_answer"


class TaskRunSplit(BaseModel):
    """A split whose items are TaskRuns, selected by a dataset filter."""

    # Fields a future build adds are preserved rather than dropped, for the same reason
    # Eval.splits keeps unknown split names: these files sync between app versions. It is
    # also why the legacy-field migration never overwrites a split that `splits` already
    # describes — rebuilding one from a bare filter-id string would drop everything else
    # on it.
    model_config = ConfigDict(extra="allow")

    source: Literal["task_run"] = "task_run"
    filter_id: DatasetFilterId


class EvalInputSplit(BaseModel):
    """A split whose items are EvalInputs, selected by an eval-input filter."""

    model_config = ConfigDict(extra="allow")

    source: Literal["eval_input"] = "eval_input"
    filter_id: EvalInputFilterId


SplitRef = Annotated[
    Union[TaskRunSplit, EvalInputSplit],
    Discriminator("source"),
]
"""One of an eval's splits: which store its items come from, and which filter selects them.
Discriminated on `source`, so a split's backing is part of its value rather than a
convention a reader has to know."""

EvalSplitName = Literal["train", "val", "test"]
"""The split names the API exposes. `Eval.splits` is keyed by plain `str` so a file
written by a build that knows a fourth split still loads here (see Eval.splits)."""

LEGACY_SPLIT_FIELDS: Dict[str, str] = {
    "test": "eval_set_filter_id",
    "train": "train_set_filter_id",
}
"""Split name -> the deprecated flat `Eval` field a Kiln build predating `splits` stored it
in. These fields are an input format and nothing else: `Eval.migrate_legacy_split_fields`
reads each one once, on the way in, and clears it. Nothing else in the codebase reads or
writes them, and they are never written to disk again — see `Eval.splits`."""


class Eval(KilnParentedModel, KilnParentModel, parent_of={"configs": EvalConfig}):
    """An evaluator definition that specifies what to evaluate and how scores should be produced."""

    name: FilenameString = Field(description="The name of the eval.")
    description: str | None = Field(
        default=None, description="The description of the eval"
    )
    template: EvalTemplateId | None = Field(
        default=None,
        description="The template selected when creating this eval. Useful for suggesting eval steps and output scores.",
    )
    current_config_id: ID_TYPE = Field(
        default=None,
        description="The id of the current config to use for this eval. This can be changed over time to run the same eval with different configs.",
    )
    eval_set_filter_id: DatasetFilterId | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated, and neither read nor written. It exists only so evals written by a Kiln build that predates `splits` still load: on load its value is migrated into splits['test'] once, and the field is then cleared. It is always saved as null. Read splits['test'] instead.",
    )
    eval_configs_filter_id: DatasetFilterId | None = Field(
        default=None,
        description="The id of the dataset filter which defines which dataset items are included when comparing the quality of the eval configs under this eval. Should consist of dataset items with ratings.",
    )
    train_set_filter_id: DatasetFilterId | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated, and neither read nor written. It exists only so evals written by a Kiln build that predates `splits` still load: on load its value is migrated into splits['train'] once, and the field is then cleared. It is always saved as null. Read splits['train'] instead.",
    )
    splits: Dict[str, SplitRef] = Field(
        default_factory=dict,
        description="The eval's dataset splits, keyed by split name ('test', 'train', 'val'), and the only place they are stored. Each split names the store its items come from and the filter that selects them. Keys this build doesn't know are preserved but not exposed. 'golden' is not a split and does not belong here: the golden set must be dataset (TaskRun) based, because human ratings only exist on dataset items, so it is stored in eval_configs_filter_id instead. Nothing reads splits['golden'] — writing it is accepted and silently ignored. In Python, prefer Eval.set_split() to assigning into this dict: it refuses to mutate a readonly (cached) eval, and marks the field as set so exclude_unset dumps keep it.",
    )
    output_scores: List[EvalOutputScore] = Field(
        description="The scores this evaluator should produce."
    )
    favourite: bool = Field(
        default=False,
        description="Whether this eval is a favourite of the user. Rendered as a star icon in the UI.",
    )
    priority: Priority | None = Field(
        default=None,
        description="The priority of the eval. None on evals created before priority lived on evals; read through resolved_priority(), which falls back to the associated spec.",
    )
    status: EvalStatus | None = Field(
        default=None,
        description="The status of the eval. None on evals created before status lived on evals; read through resolved_status(), which falls back to the associated spec.",
    )
    template_properties: dict[str, str | int | bool | float] | None = Field(
        default=None,
        description="Properties to be used to execute the eval. This is template_type specific and should serialize to a json dict.",
    )
    evaluation_data_type: EvalDataType | None = Field(
        default=EvalDataType.final_answer,
        description="The output of the task run to evaluate. Can be final answer, full trace, or None for V2 evals.",
    )

    @model_validator(mode="after")
    def migrate_legacy_split_fields(self) -> Self:
        """Migrate the deprecated flat filter fields into `splits`, once, and clear them.

        `splits` is the only home a split has. These fields are an input format for evals
        written before it existed, so each one is read exactly once — here — and only for
        a split `splits` doesn't already describe. `splits` winning is what makes the
        migration one-way: once a value is in `splits` it is the eval's answer, and a
        legacy field left over beside it (a hand-edited file, or one an older build wrote
        after a newer one) is ignored rather than allowed to overwrite it. Overwriting
        would also drop any extra fields on the existing split object, which
        `TaskRunSplit`/`EvalInputSplit` keep on purpose (`extra="allow"`).

        Both fields are then cleared, unconditionally. That is what makes this a
        migration rather than a second home: nothing downstream can read a stale value,
        the eval saves with both fields null, and re-running the validator — which
        `validate_assignment` does on every attribute set, including `self.path = path`
        at the end of save_to_file — has nothing left to do. An older Kiln build reading
        the saved file sees no test set rather than the wrong one; that is the accepted
        cost of a single home, and the eval list surfaces the evals it can't read.

        Reads and writes go through `__dict__` because the fields are
        `deprecated=True`: attribute access on them emits a DeprecationWarning, which is
        meant for callers, not for the one place that is supposed to touch them.

        Must stay declared before validate_splits, which requires a test split: an eval
        that carries only legacy fields gets its test split from here.
        """
        for name, field_name in LEGACY_SPLIT_FIELDS.items():
            filter_id = self.__dict__.get(field_name)
            if filter_id is not None and name not in self.splits:
                self.splits[name] = TaskRunSplit(filter_id=filter_id)
                # The split now lives only in `splits`, so an exclude_unset dump has to
                # carry it: on a legacy eval `splits` was never explicitly set.
                self.__pydantic_fields_set__.add("splits")
            self.__dict__[field_name] = None
        return self

    @model_validator(mode="after")
    def validate_splits(self) -> Self:
        if "test" not in self.splits:
            raise ValueError("An eval must have a test split. Set splits['test'].")
        return self

    def set_split(self, name: str, split: SplitRef) -> None:
        """Set one of the eval's splits.

        Equivalent to `eval.splits[name] = split` plus the two things item assignment on
        a dict can't do for itself, because it never reaches `__setattr__`: refusing to
        mutate a readonly (cached) eval, and marking `splits` as set so an
        exclude_unset dump still carries it.
        """
        # Readonly instances are the cached ones, shared with every other holder of the
        # same file, so this check has to be explicit here.
        self._ensure_not_readonly("splits")
        self.splits[name] = split
        # Validated evals always have `splits` marked already (their test split came from
        # `splits` or from the legacy migration, which marks it), so this is for instances
        # built by model_construct, where nothing did.
        self.__pydantic_fields_set__.add("splits")

    # Workaround to return typed parent without importing Task
    def parent_task(self) -> Union["Task", None]:
        if self.parent is not None and self.parent.__class__.__name__ != "Task":
            raise ValueError("parent must be a Task")
        return self.parent  # type: ignore

    def configs(self, readonly: bool = False) -> list[EvalConfig]:
        return super().configs(readonly=readonly)  # type: ignore

    # Workaround to return typed parent without importing Spec
    def associated_spec(self, readonly: bool = False) -> Union["Spec", None]:
        """
        Get the spec associated with this eval, if any.
        Returns None for legacy evals that are not associated with a spec.
        """

        task = self.parent_task()
        if not task or not self.id:
            return None

        specs = task.specs(readonly=readonly)
        for spec in specs:
            if spec.eval_id == self.id:
                return spec
        return None

    def resolved_priority(self, spec: Union["Spec", None] = None) -> Priority:
        """
        The eval's effective priority. Priority lives on the eval; evals created
        before that (spec-backed legacy files) fall back to their spec's value.
        Pass *spec* when the caller already has it, to avoid a re-scan.
        """
        if self.priority is not None:
            return self.priority
        spec = spec or self.associated_spec(readonly=True)
        if spec is not None:
            return spec.priority
        return Priority.p1

    def resolved_status(self, spec: Union["Spec", None] = None) -> EvalStatus:
        """
        The eval's effective status, with the same spec fallthrough as
        resolved_priority().
        """
        if self.status is not None:
            return self.status
        spec = spec or self.associated_spec(readonly=True)
        if spec is not None:
            return spec.status
        return EvalStatus.active

    def eval_reference_data_keys(self) -> list[str]:
        """Union of reference-data keys across all of this eval's V2 configs.

        Returns deduplicated keys in stable insertion order.
        """
        seen: set[str] = set()
        result: list[str] = []
        for config in self.configs(readonly=True):
            if config.config_type != EvalConfigType.v2:
                continue
            if not isinstance(config.properties, V2_PROPERTY_TYPES):
                continue
            for key in reference_data_keys(config.properties):  # type: ignore[arg-type]
                if key not in seen:
                    seen.add(key)
                    result.append(key)
        return result

    @model_validator(mode="after")
    def upgrade_old_reference_answer_eval_config(self) -> Self:
        """
        Migration: Set the first judge config as the default for existing reference answer evals that don't have a current_config_id set.

        For reference_answer evals that don't have a current_config_id set, this migration
        will set the first config (by created_at) as the default.
        """
        if self.id is None:
            return self

        # Only run during file loading
        if not self._loaded_from_file:
            return self

        # Skip if already migrated (has a current_config_id set)
        if self.current_config_id is not None:
            return self

        # Only migrate reference_answer evals
        if self.evaluation_data_type != EvalDataType.reference_answer:
            return self

        # Prevent recursion: self.configs() loads child files, which re-loads this parent
        # (see basemodel.py where we iterate_children_paths_of_parent_path calls load_from_file)
        # This causes the validator to run again, creating an infinite loop without this guard.
        with _migration_lock:
            if self.id in _currently_migrating_eval_ids:
                return self
            _currently_migrating_eval_ids.add(self.id)

        try:
            # Get the configs - these are loaded from child files
            configs_list = self.configs(readonly=True)
            if configs_list and len(configs_list) > 0:
                # Sort by created_at to get the oldest (first created) config
                sorted_configs = sorted(configs_list, key=lambda c: c.created_at)
                self.current_config_id = sorted_configs[0].id
        finally:
            with _migration_lock:
                _currently_migrating_eval_ids.discard(self.id)

        return self

    @model_validator(mode="after")
    def validate_scores(self) -> Self:
        if self.output_scores is None or len(self.output_scores) == 0:
            raise ValueError(
                "output_scores are required, and must have at least one score."
            )

        # check for duplicate names (once transformed to JSON keys)
        output_score_keys = [score.json_key() for score in self.output_scores]
        if len(output_score_keys) != len(set(output_score_keys)):
            raise ValueError(
                f"output_scores must have unique names (once transformed to JSON keys). Got: [{', '.join(output_score_keys)}]"
            )
        return self

    @model_validator(mode="after")
    def validate_template_properties(self) -> Self:
        if self.template is None:
            return self

        if (
            self.template is not EvalTemplateId.rag
            and self.eval_configs_filter_id is None
        ):
            raise ValueError(
                "eval_configs_filter_id is required for all templates except 'rag'"
            )

        # For spec-based evals, template_properties will be None and validation happens in the spec
        # For legacy evals, template_properties contains the data and we validate here
        if self.template_properties is None:
            return self

        # Check for properties that are required for the issue template (legacy evals only)
        if self.template == EvalTemplateId.issue:
            if "issue_prompt" not in self.template_properties or not isinstance(
                self.template_properties["issue_prompt"], str
            ):
                raise ValueError("issue_prompt is required for issue template")
            if "failure_example" in self.template_properties and not isinstance(
                self.template_properties["failure_example"], str
            ):
                raise ValueError(
                    "failure_example is optional for issue template, but if provided must be a string"
                )
            if "pass_example" in self.template_properties and not isinstance(
                self.template_properties["pass_example"], str
            ):
                raise ValueError(
                    "pass_example is optional for issue template, but if provided must be a string"
                )

        if self.template == EvalTemplateId.tool_call:
            if self.evaluation_data_type != EvalDataType.full_trace:
                raise ValueError(
                    "tool_call template should have evaluation_data_type set to full_trace"
                )
            if (
                "tool" not in self.template_properties
                or not isinstance(self.template_properties["tool"], str)
                or not self.template_properties["tool"].strip()
            ):
                raise ValueError("tool is required for tool call template")
            if "tool_function_name" not in self.template_properties or not isinstance(
                self.template_properties["tool_function_name"], str
            ):
                raise ValueError(
                    "tool_function_name is required for tool call template"
                )
            if (
                "appropriate_tool_use_guidelines" not in self.template_properties
                or not isinstance(
                    self.template_properties["appropriate_tool_use_guidelines"], str
                )
                or not self.template_properties[
                    "appropriate_tool_use_guidelines"
                ].strip()
            ):
                raise ValueError(
                    "appropriate_tool_use_guidelines is required for tool call template"
                )
            if (
                "inappropriate_tool_use_guidelines" in self.template_properties
                and not isinstance(
                    self.template_properties["inappropriate_tool_use_guidelines"], str
                )
            ):
                raise ValueError(
                    "inappropriate_tool_use_guidelines is optional for tool call template, but if provided must be a string"
                )
        return self
