import json
import logging
from collections import defaultdict
from dataclasses import replace
from typing import Annotated, Any, Dict, List, Set, Tuple, Type, TypeVar

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse
from kiln_ai.adapters.adapter_registry import validate_run_config_tool_names
from kiln_ai.adapters.eval.base_eval import (
    DEFAULT_SYSTEM_PROMPT,
    build_default_llm_judge_prompt,
    derived_reference_keys,
    materialize_llm_judge_properties,
)
from kiln_ai.adapters.eval.eval_runner import EvalRunner, no_golden_set_message
from kiln_ai.adapters.eval.registry import v2_eval_adapter_from_config
from kiln_ai.adapters.eval.v2_eval_code_eval import (
    CodeEvalAdapter,
    add_code_trust,
    has_add_code_trust,
)
from kiln_ai.adapters.fine_tune.finetune_run_config_id import (
    finetune_from_finetune_run_config_id,
    finetune_run_config_id,
)
from kiln_ai.adapters.ml_model_list import ModelProviderName
from kiln_ai.adapters.prompt_builders import prompt_builder_from_id
from kiln_ai.datamodel import BasePrompt, Task, TaskRun
from kiln_ai.datamodel.basemodel import (
    ID_TYPE,
    FilenameStringShort,
    KilnParentedModel,
)
from kiln_ai.datamodel.datamodel_enums import (
    EvalStatus,
    Priority,
)
from kiln_ai.datamodel.dataset_filters import (
    DatasetFilterId,
    EvalInputFilterId,
    dataset_filter_from_id,
    eval_input_filter_from_id,
)
from kiln_ai.datamodel.eval import (
    V2_PROPERTY_TYPES,
    CodeEvalProperties,
    Eval,
    EvalConfig,
    EvalConfigType,
    EvalDataType,
    EvalInput,
    EvalInputData,
    EvalOutputScore,
    EvalRun,
    EvalScores,
    EvalSplitName,
    EvalTaskInput,
    EvalTemplateId,
    MultiTurnSyntheticEvalInputData,
    SingleTurnEvalInputData,
    SkippedReason,
    SplitRef,
    TaskRunSplit,
    V2EvalConfigProperties,
    reference_data_keys,
    validate_scores_against_output_scores,
)
from kiln_ai.datamodel.eval_splits import (
    ItemKey,
    ItemSource,
    ResolvedSplit,
    eval_run_item_key,
    resolve_split,
)
from kiln_ai.datamodel.json_schema import string_to_json_key
from kiln_ai.datamodel.prompt_id import is_frozen_prompt
from kiln_ai.datamodel.prompt_type import generator_label
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties
from kiln_ai.datamodel.spec import Spec
from kiln_ai.datamodel.task import RunConfigProperties, TaskRunConfig
from kiln_ai.datamodel.task_output import normalize_rating
from kiln_ai.datamodel.usage import Usage
from kiln_ai.tools.sandbox_bridge import ToolCallLogEntry
from kiln_ai.utils.exhaustive_error import raise_exhaustive_enum_error
from kiln_ai.utils.name_generator import generate_memorable_name
from kiln_ai.utils.open_ai_types import serialize_trace
from kiln_server.cancellable_streaming_response import CancellableStreamingResponse
from kiln_server.git_sync_decorators import build_save_context, no_write_lock
from kiln_server.project_api import project_from_id
from kiln_server.statistics_lib import percentile
from kiln_server.task_api import task_from_id
from kiln_server.utils.agent_checks.policy import (
    ALLOW_AGENT,
    DENY_AGENT,
    agent_policy_require_approval,
)
from kiln_server.utils.spec_utils import (
    eval_pass_fail_output_score,
    generate_spec_eval_tags,
    spec_eval_splits,
    tag_filter_id,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
)

from app.desktop.studio_server.code_tool_api import ToolCallLogEntryResponse

from .correlation_calculator import (
    CorrelationCalculator,
    CorrelationResult,
    CorrelationScore,
)

logger = logging.getLogger(__name__)


def reusable_frozen_prompt_id(
    task: Task,
    project_id: ID_TYPE,
    prompt_text: str,
    cot_instructions: str | None,
) -> str | None:
    """Find an existing run config whose frozen prompt exactly matches the given
    content, so we can reuse it instead of creating a duplicate frozen prompt.

    Returns the frozen prompt id (task_run_config::...) of the match, or None if
    there's no match. If multiple match (legacy data created before dedup), the
    most recently created run config is used.
    """
    # Treat empty-string and missing chain-of-thought instructions as equivalent
    # so prompts dedupe regardless of how "no instructions" is represented.
    normalized_cot = cot_instructions or None
    matches = [
        run_config
        for run_config in task.run_configs(readonly=True)
        if run_config.prompt is not None
        and run_config.prompt.prompt == prompt_text
        and (run_config.prompt.chain_of_thought_instructions or None) == normalized_cot
    ]
    if not matches:
        return None
    most_recent = max(matches, key=lambda run_config: run_config.created_at)
    return f"task_run_config::{project_id}::{task.id}::{most_recent.id}"


def eval_from_id(project_id: str, task_id: str, eval_id: str) -> Eval:
    task = task_from_id(project_id, task_id)
    eval = Eval.from_id_and_parent_path(eval_id, task.path)
    if eval is not None:
        return eval

    raise HTTPException(
        status_code=404,
        detail=f"Eval not found. ID: {eval_id}",
    )


def resolve_eval_display_fields(eval: Eval, spec: Spec | None = None) -> Eval:
    """Fill priority/status with their resolved values for API responses.

    Priority/status live on the eval, but evals created before that carry None
    and fall through to their spec. Responses always return concrete values so
    the UI never re-implements the fallthrough. Mutates the request-scoped
    instance without saving, so legacy files stay untouched.
    """
    eval.priority = eval.resolved_priority(spec)
    eval.status = eval.resolved_status(spec)
    return eval


def eval_config_from_id(
    project_id: str, task_id: str, eval_id: str, eval_config_id: str
) -> EvalConfig:
    eval = eval_from_id(project_id, task_id, eval_id)
    for config in eval.configs():
        if config.id == eval_config_id:
            return config

    raise HTTPException(
        status_code=404,
        detail=f"Eval config not found. ID: {eval_config_id}",
    )


def eval_input_from_id(project_id: str, task_id: str, eval_input_id: str) -> EvalInput:
    task = task_from_id(project_id, task_id)
    eval_input = EvalInput.from_id_and_parent_path(eval_input_id, task.path)
    if eval_input is not None:
        return eval_input

    raise HTTPException(
        status_code=404,
        detail=f"Eval input not found. ID: {eval_input_id}",
    )


class EvalInputReferences(BaseModel):
    """What still points at an eval input item, by id.

    Counts rather than the records themselves: the caller is deciding whether a delete is
    safe, and loading every trace to render a 409 body would make the guard cost more
    than the delete it is protecting.
    """

    trace_count: int = Field(
        description="Eval traces generated for this item (TaskRun.eval_source.source_id)."
    )
    score_count: int = Field(
        description="Stored score records naming this item (EvalRun.eval_input_id)."
    )

    def __bool__(self) -> bool:
        return self.trace_count > 0 or self.score_count > 0


def eval_input_references(task: Task, eval_input_id: str) -> EvalInputReferences:
    """Everything on disk that names `eval_input_id`.

    Two kinds, and both are id-only — neither copies the item's content, which is exactly
    why a delete has to be refused rather than cascaded:

    - Eval traces: `TaskRun.eval_source` is `("eval_input", item_id)`, and `TraceIndex`
      reuses a trace on that pair plus the run config. `include_eval_generated=True`
      because these runs are excluded from `task.runs()` by default.
    - Score records: `EvalRun.eval_input_id`, over every eval config of every eval on the
      task. Read-only: nothing here mutates them.
    """
    trace_count = sum(
        1
        for run in task.runs(readonly=True, include_eval_generated=True)
        if run.eval_source is not None
        and run.eval_source.source_type == "eval_input"
        and run.eval_source.source_id == eval_input_id
    )

    score_count = 0
    for eval in task.evals(readonly=True):
        for eval_config in eval.configs(readonly=True):
            score_count += sum(
                1
                for eval_run in eval_config.runs(readonly=True)
                if eval_run.eval_input_id == eval_input_id
            )

    return EvalInputReferences(trace_count=trace_count, score_count=score_count)


def references_conflict_detail(references: EvalInputReferences) -> str:
    """The 409 body for a delete that would orphan records.

    Names both counts even when one is zero, so the reader can tell "no traces" from "we
    didn't look at traces", and says what to do instead — retagging is the supported way
    to take an item out of an eval's scope.
    """
    return (
        f"Eval input is still referenced by {references.trace_count} eval trace(s) and "
        f"{references.score_count} score record(s), which name it by id and hold no copy "
        "of its content. Deleting it would leave those records describing an item that "
        "no longer exists. Retag the item to take it out of an eval's scope instead."
    )


def get_all_run_configs(project_id: str, task_id: str) -> list[TaskRunConfig]:
    """
    Returns all run configs for a task, including completed fine-tune run configs.
    Only includes fine-tunes that have a fine_tune_model_id (are completed and usable).
    """
    task = task_from_id(project_id, task_id)
    configs = task.run_configs()

    # Get run configs from finetunes and only include completed fine-tunes
    finetunes = task.finetunes()
    for finetune in finetunes:
        if finetune.run_config is not None and finetune.fine_tune_model_id is not None:
            configs.append(
                TaskRunConfig(
                    id=finetune_run_config_id(project_id, task_id, str(finetune.id)),
                    name=finetune.name,
                    description=finetune.description,
                    run_config_properties=finetune.run_config,
                    parent=task,  # special case, we need to reference the task model
                )
            )

    return configs


def task_run_config_from_id(
    project_id: str, task_id: str, run_config_id: str
) -> TaskRunConfig:
    task = task_from_id(project_id, task_id)
    for run_config in task.run_configs():
        if run_config.id == run_config_id:
            return run_config

    # special case for finetune run configs, it's inside the finetune model
    if run_config_id.startswith("finetune_run_config::"):
        finetune = finetune_from_finetune_run_config_id(run_config_id)
        if finetune.run_config is not None:
            return TaskRunConfig(
                id=finetune_run_config_id(project_id, task_id, str(finetune.id)),
                name=finetune.name,
                description=finetune.description,
                run_config_properties=finetune.run_config,
                parent=task,  # special case, we need to reference the task model
            )

    raise HTTPException(
        status_code=404,
        detail=f"Task run config not found. ID: {run_config_id}",
    )


async def run_eval_runner_with_status(eval_runner: EvalRunner) -> StreamingResponse:
    # Yields async messages designed to be used with server sent events (SSE)
    # https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
    async def event_generator():
        async for progress in eval_runner.run():
            data = {
                "progress": progress.complete,
                "total": progress.total,
                "errors": progress.errors,
            }
            yield f"data: {json.dumps(data)}\n\n"

        # Send the final complete message the app expects, and uses to stop listening
        yield "data: complete\n\n"

    return CancellableStreamingResponse(
        content=event_generator(),
        media_type="text/event-stream",
    )


class CreateEvaluatorRequest(BaseModel):
    """Request to create a new evaluator."""

    # FilenameStringShort (not the longer FilenameString): the generated
    # default output score is named after the eval and score names cap at 32
    # chars, so longer names must 422 here rather than 500 in the handler.
    name: FilenameStringShort = Field(description="The name of the evaluator.")
    description: str | None = Field(
        default=None, description="The description of the evaluator."
    )
    template: EvalTemplateId | None = Field(
        default=None, description="The eval template to use."
    )
    output_scores: list[EvalOutputScore] | None = Field(
        default=None,
        min_length=1,
        description="The scores this evaluator should produce. When omitted, a pass/fail score named after the eval is generated.",
    )
    eval_set_filter_id: DatasetFilterId | None = Field(
        default=None,
        description="The dataset filter for the eval set. When omitted, tag-based eval/train/golden filters are generated from the eval name, matching what spec-backed evals get.",
    )
    eval_configs_filter_id: DatasetFilterId | None = Field(
        default=None, description="The dataset filter for comparing eval configs."
    )
    template_properties: dict[str, str | float | int | bool] | None = Field(
        default=None, description="Template-specific properties."
    )
    evaluation_data_type: EvalDataType = Field(
        description="The type of task output to evaluate."
    )
    priority: Priority = Field(
        default=Priority.p1, description="The priority of the eval."
    )
    status: EvalStatus = Field(
        default=EvalStatus.active, description="The status of the eval."
    )


class CreateEvalConfigRequest(BaseModel):
    """Request to create a new eval configuration."""

    name: str | None = Field(default=None, description="The name of the eval config.")
    type: EvalConfigType = Field(description="The type of eval config.")
    properties: dict[str, Any] = Field(
        description="Properties for the eval config, specific to the type."
    )
    model_name: str | None = Field(
        default=None,
        description="The model to use for evaluation. Required for LLM-based eval types.",
    )
    provider: ModelProviderName | None = Field(
        default=None,
        description="The provider of the evaluation model. Required for LLM-based eval types.",
    )


class LlmJudgeBuilderInput(BaseModel):
    """Shared fields for llm_judge: model, provider, g_eval."""

    model_name: str = Field(description="The LLM model to use as judge.")
    provider: ModelProviderName = Field(description="The model provider.")
    g_eval: bool = Field(description="Whether to use G-Eval logprob scoring.")
    judge_prompt: str | None = Field(
        default=None,
        description="Override the judge prompt template. If unset, the server assembles a rich default from the eval's task and spec.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Override the judge system prompt. Defaults to 'You are an evaluator.'",
    )
    judge_instructions: list[str] | None = Field(
        default=None,
        description="User-written evaluation steps, bound to {{ judge_instructions }} when the judge prompt is rendered. Used by evals with no spec or template to derive default steps from.",
    )


class DefaultLlmJudgePromptResponse(BaseModel):
    """Response from the default LLM judge prompt endpoint."""

    judge_prompt: str
    system_prompt: str
    reference_keys: list[str] = Field(
        default_factory=list,
        description=(
            "Reference data keys the server will require of a judge for this eval, "
            "derived the same way `create_llm_judge_config` derives them. Returned so "
            "the builder can offer a place to supply them when testing, rather than "
            "re-deriving the rule client-side from the prompt's text."
        ),
    )


class CreateLlmJudgeConfigRequest(LlmJudgeBuilderInput):
    """Request to create a V2 llm_judge eval config with server-baked template."""

    name: str | None = Field(default=None, description="The name of the eval config.")


class TestV2EvalRequest(BaseModel):
    """Request to test-run a V2 eval config without persisting."""

    properties: V2EvalConfigProperties | None = Field(
        default=None,
        description="The V2 eval config properties to test. Required unless llm_judge_builder_input is set.",
    )
    eval_input: EvalTaskInput = Field(description="The input to evaluate.")
    llm_judge_builder_input: LlmJudgeBuilderInput | None = Field(
        default=None,
        description="Builder input for llm_judge; when set, the server bakes the full properties from the eval's output_scores.",
    )


class TestV2EvalResponse(BaseModel):
    """Response from a test-run of a V2 eval."""

    scores: EvalScores = Field(default_factory=dict)
    skipped_reason: str | None = None
    skipped_detail: str | None = None
    score_range_errors: list[str] | None = None
    intermediate_outputs: dict[str, str] | None = None
    tool_call_log: list[ToolCallLogEntryResponse] = Field(
        default_factory=list,
        description="Tools the scorer code called, in call order. Code evals only.",
    )


class TestV2EvalDraftRequest(BaseModel):
    """Request to test-run a V2 eval config for an eval that doesn't exist yet.

    Used by the creation flow, where the eval (and its scores) are still being
    drafted; the server builds a transient in-memory eval from output_scores.
    """

    properties: V2EvalConfigProperties = Field(
        description="The V2 eval config properties to test."
    )
    output_scores: list[EvalOutputScore] = Field(
        description="The scores the drafted eval will declare; returned scores are validated against them."
    )
    eval_input: EvalTaskInput = Field(description="The input to evaluate.")


async def run_v2_eval_test(
    project_id: str,
    eval_obj: Eval,
    properties: V2EvalConfigProperties,
    eval_input: EvalTaskInput,
) -> TestV2EvalResponse:
    """Run a transient (unsaved) V2 eval config against one input.

    Shared by the eval-scoped test endpoint and the creation-flow draft
    endpoint; *eval_obj* may be an unsaved in-memory eval.
    """
    transient_config = EvalConfig(
        name="test_run",
        config_type=EvalConfigType.v2,
        properties=properties,
        parent=eval_obj,
    )
    adapter = v2_eval_adapter_from_config(transient_config)

    tool_call_log: list[ToolCallLogEntry] = []

    # Trust-conferral gate: executing not-yet-saved code in the test pane
    # requires code trust for this session. Saved code needs no gate.
    if isinstance(adapter, CodeEvalAdapter):
        project = project_from_id(project_id)
        if not has_add_code_trust(str(project.path)):
            return TestV2EvalResponse(
                skipped_reason=SkippedReason.code_eval_not_trusted.value,
                skipped_detail="Project not trusted for code eval execution.",
            )
        # The author is iterating on this code right now, so show them what
        # it called -- nested LLM calls in particular are real spend.
        adapter.tool_call_recorder = tool_call_log.append

    result = await adapter.evaluate(eval_input)

    score_range_errors: list[str] | None = None
    if result.skipped_reason is None and result.scores:
        problems = validate_scores_against_output_scores(
            result.scores, eval_obj.output_scores
        )
        if problems:
            score_range_errors = problems

    return TestV2EvalResponse(
        scores=result.scores,
        skipped_reason=result.skipped_reason.value if result.skipped_reason else None,
        skipped_detail=result.skipped_detail,
        score_range_errors=score_range_errors,
        intermediate_outputs=result.intermediate_outputs,
        tool_call_log=ToolCallLogEntryResponse.from_log(tool_call_log),
    )


class CodeTrustResponse(BaseModel):
    """Response indicating whether code is trusted for a project in this session."""

    trusted: bool


class CreateTaskRunConfigRequest(BaseModel):
    """Request to create a new run config for eval."""

    name: str | None = Field(default=None, description="The name of the run config.")
    description: str | None = Field(
        default=None, description="The description of the run config."
    )
    run_config_properties: RunConfigProperties = Field(
        description="The run configuration properties."
    )


class UpdateRunConfigRequest(BaseModel):
    """Request to update a run config."""

    name: str | None = Field(default=None, description="The updated name.")
    starred: bool | None = Field(
        default=None, description="The updated starred status."
    )
    prompt_name: str | None = Field(
        default=None, description="The updated prompt name."
    )


class RunEvalConfigRequest(BaseModel):
    """Request to run an eval with specific run configs."""

    run_config_ids: list[str] = Field(description="The run config IDs to evaluate.")


class ScoreSummary(BaseModel):
    """Summary of scores for an eval run."""

    mean_score: float | None = Field(
        description="The mean score across all used runs. None when n_used == 0."
    )
    # Distribution fields. Optional/defaulted so older stored payloads and
    # existing API consumers keep working — the mean is unchanged. Numeric
    # (custom-type) scores like tool-call counts or per-turn latency are
    # heavily right-skewed, where the mean hides the tail that drives cost.
    min_score: float | None = Field(
        default=None,
        description="The lowest score across all used runs. None when n_used == 0.",
    )
    p25_score: float | None = Field(
        default=None,
        description="The 25th-percentile score across all used runs. None when n_used == 0.",
    )
    median_score: float | None = Field(
        default=None,
        description="The median (50th-percentile) score across all used runs. None when n_used == 0.",
    )
    p75_score: float | None = Field(
        default=None,
        description="The 75th-percentile score across all used runs. None when n_used == 0.",
    )
    p90_score: float | None = Field(
        default=None,
        description="The 90th-percentile score across all used runs. None when n_used == 0.",
    )
    max_score: float | None = Field(
        default=None,
        description="The highest score across all used runs. None when n_used == 0.",
    )
    n_used: int = Field(
        description="Number of EvalRuns with all expected scores and not skipped."
    )
    n_excluded: int = Field(
        description="Number of EvalRuns excluded due to skipped_reason."
    )


class MeanUsage(BaseModel):
    """Average token usage across eval runs."""

    mean_input_tokens: float | None = Field(
        default=None, description="Average input tokens per run."
    )
    mean_output_tokens: float | None = Field(
        default=None, description="Average output tokens per run."
    )
    mean_total_tokens: float | None = Field(
        default=None, description="Average total tokens per run."
    )
    mean_cost: float | None = Field(
        default=None, description="Average cost per run in USD."
    )
    mean_total_llm_latency_ms: float | None = Field(
        default=None,
        description="Average total LLM latency per run in milliseconds.",
    )


class EvalRunWithTrace(BaseModel):
    """An eval's scores for one item, plus the trace those scores were computed over.

    Where the trace lives depends on the record: on a TaskRun named by `scored_run_id`,
    inline on the EvalRun for records written before the trace/score split, or nowhere at
    all for a run that was skipped before anything was generated. This resolves whichever
    applies - falling back to the dataset item for the input whenever the record
    itself has none - so callers see one shape regardless of which it is.
    """

    eval_run: EvalRun = Field(description="The score record itself.")
    input: str | None = Field(
        description="The input the task was run on. From the scored TaskRun, from the "
        "EvalRun itself for legacy records, or from the dataset item whenever neither "
        "of those has it (pre-generation skips, and pointer records whose trace is "
        "missing)."
    )
    output: str | None = Field(
        description="What the task produced. Always the original output, never a "
        "repaired one: a repair can happen after scoring, so it is not what was scored. "
        "None when nothing was generated, or when the scored TaskRun is missing."
    )
    task_run_trace: str | None = Field(
        description="The JSON formatted trace of the task run that produced the output, "
        "if it recorded one."
    )
    task_run_usage: Usage | None = Field(
        description="The usage of the task run that produced the output. Not the "
        "judge's own usage, which is on the EvalRun as eval_usage."
    )

    @classmethod
    def from_own_fields(cls, eval_run: EvalRun) -> "EvalRunWithTrace":
        """A record with no `scored_run_id`, which has only its own fields to offer.

        Two kinds land here: legacy inline records, which carry a full copy of the trace,
        and pre-generation skips, which carry none of it - nothing was generated, so
        there is nothing to point at and nothing to copy. The latter get their input from
        the dataset item afterwards.
        """
        return cls(
            eval_run=eval_run,
            input=eval_run.input,
            output=eval_run.output,
            task_run_trace=eval_run.task_run_trace,
            task_run_usage=eval_run.task_run_usage,
        )

    @classmethod
    def from_scored_run(
        cls, eval_run: EvalRun, trace: TaskRun | None
    ) -> "EvalRunWithTrace":
        """A record that points at its trace, with `trace` None if it no longer exists.

        A dangling pointer degrades to nulls rather than raising: the score still renders
        and still aggregates, only the drill-through is unavailable.
        """
        return cls(
            eval_run=eval_run,
            input=trace.input if trace is not None else None,
            # Never repaired_output: a repair can happen after scoring, so it is not
            # what the score was computed over (functional spec 5.2).
            output=trace.output.output if trace is not None else None,
            task_run_trace=serialize_trace(trace.trace)
            if trace is not None and trace.trace
            else None,
            # Raw per-record usage, not the summary's blended figure: it omits
            # synthetic_user_usage and is last-turn-only for chain leaves. No UI
            # renders it today; a surface that reports cost should use the
            # summary's blend, not this field.
            task_run_usage=trace.usage if trace is not None else None,
        )


class EvalRunResult(BaseModel):
    """Results of an eval run including the eval and run config."""

    results: List[EvalRunWithTrace] = Field(
        description="The individual eval run results."
    )
    eval: Eval = Field(description="The parent eval.")
    eval_config: EvalConfig = Field(description="The eval config used.")
    run_config: TaskRunConfig = Field(description="The run config used.")


class UpdateFavouriteRequest(BaseModel):
    """Request to update the favourite status of an eval."""

    favourite: bool = Field(description="Whether the eval is a favourite.")


class UpdateEvalRequest(BaseModel):
    """Request to update an eval."""

    name: str | None = Field(default=None, description="The updated name.")
    description: str | None = Field(
        default=None, description="The updated description."
    )
    priority: Priority | None = Field(default=None, description="The updated priority.")
    status: EvalStatus | None = Field(default=None, description="The updated status.")
    # Typed so an invalid filter id is a 422 at request validation, not a 500
    # when TaskRunSplit rejects it inside the handler.
    train_set_filter_id: DatasetFilterId | None = Field(
        default=None, description="The updated train set filter ID."
    )


def eval_input_tags_must_be_filterable(tags: list[str] | None) -> list[str] | None:
    """Tag rule for the eval-input requests, mirroring EvalInput's own.

    A tag that is empty or contains a space can't be named by a `tag::`
    filter, so an item carrying one would silently never be selected by any
    eval. Same two rules and same wording as the datamodel, deliberately.

    Restated here because the datamodel enforces them while the item is being
    built or assigned: that failure is about the whole model, so the caller
    gets an error located nowhere and a body quoting the item being saved.
    Validating the request first points the 422 at `tags` and quotes what the
    caller actually sent.

    None is the update request's "leave tags alone" and carries nothing to
    check; the route rejects it separately for the PATCH.
    """
    for tag in tags or []:
        if not tag:
            raise ValueError("Tags cannot be empty strings")
        if " " in tag:
            raise ValueError("Tags cannot contain spaces. Try underscores.")
    return tags


def multi_turn_data_must_carry_a_drive_config(data: EvalInputData) -> EvalInputData:
    """Drive-config rule for the eval-input create request.

    A multi-turn item is only useful if it can be re-driven, and the drive
    settings live on the item alone: `data` is the immutable scenario, so no
    later PATCH can supply one. Without it the eval runner skips the item with
    missing_drive_config, and nothing can lift that, so accepting the create
    would mint a permanently unrunnable item. Reject it while the caller can
    still fix it.
    """
    if isinstance(data, MultiTurnSyntheticEvalInputData) and data.drive_config is None:
        raise ValueError(
            "drive_config is required for multi_turn_synthetic eval inputs. "
            "It sets the synthetic-user model and turn count the item is "
            "re-driven with, and cannot be added after the item is created."
        )
    return data


def multi_turn_data_must_carry_a_first_message(data: EvalInputData) -> EvalInputData:
    """First-message rule for the eval-input create request.

    The first message is the seed the synthetic user opens a re-driven
    conversation with. With no seed text there is nothing to send, so the eval
    runner skips the item with incompatible_input_shape instead of re-driving
    it. Like the drive config this lives on `data`, which is immutable, so a
    seedless item is permanently unrunnable and no PATCH can rescue it. Reject
    it while the caller can still fix it.

    The datamodel keeps `first_message` optional so items that already lack
    one still load. That is a reason to keep reading them, not to mint more.
    """
    if isinstance(data, MultiTurnSyntheticEvalInputData) and not (
        data.first_message and data.first_message.text
    ):
        raise ValueError(
            "first_message with non-empty text is required for "
            "multi_turn_synthetic eval inputs. It is the message the "
            "synthetic user opens each re-driven conversation with, and "
            "cannot be added after the item is created."
        )
    return data


class CreateEvalInputRequest(BaseModel):
    """Request to create an eval input item."""

    data: EvalInputData = Field(
        description="The input data for this eval item. A multi_turn_synthetic "
        "item must carry both a drive_config and a first_message with non-empty "
        "text: they are what make it re-drivable, and neither can be added "
        "after the item is created."
    )
    reference: dict[str, JsonValue] | None = Field(
        default=None,
        description="Optional reference data (ground truth) for this eval input, keyed by reference name.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for filtering eval inputs (matched by tag:: eval_input_filter_ids).",
    )

    _tags_must_be_filterable = field_validator("tags")(
        eval_input_tags_must_be_filterable
    )
    _data_must_carry_a_drive_config = field_validator("data")(
        multi_turn_data_must_carry_a_drive_config
    )
    _data_must_carry_a_first_message = field_validator("data")(
        multi_turn_data_must_carry_a_first_message
    )


class UpdateEvalInputRequest(BaseModel):
    """Partial update of an eval input item. Omitted fields are left unchanged.

    `data` is deliberately absent, and `extra="forbid"` turns an attempt to send it into
    a 422 rather than a silent no-op the caller reads as success. The scenario is the one
    thing that genuinely cannot be edited in place: trace reuse (`TraceIndex`) keys on
    `(source_type, item_id, run_config_id)`, so a later eval would hand a judge a
    conversation generated from the scenario this item *used to* have. Changing a
    scenario means POSTing a new item.

    `reference` does not have that problem and is editable. It keys nothing: stored
    scores snapshot the `reference_data` the judge actually saw (`_persist_judgment`)
    rather than pointing back at the item, and drive fingerprints hash the scenario, not
    the reference. So correcting ground truth invalidates nothing already on disk — it
    changes what future runs are graded against, which is the whole point of correcting
    it. Iterating on reference data is a normal part of authoring a corpus, and making it
    mint-a-new-item would leave one dead item behind per correction.

    The cost, stated: scores written either side of a `reference` edit hang off the same
    item id but were graded against different ground truth. Each EvalRun carries the
    reference it saw, so this is auditable, but a rollup that groups scores by item alone
    would mix the two.
    """

    model_config = ConfigDict(extra="forbid")

    tags: list[str] | None = Field(
        default=None,
        description="The item's tags, replacing the whole list. Send [] to clear them. Tags decide which eval_input_filter_id slices the item falls into, so this is how an item is added to or removed from an eval's scope.",
    )
    reference: dict[str, JsonValue] | None = Field(
        default=None,
        description="The item's reference data (ground truth), replacing the whole dict. Send null to clear it — omitting the field leaves it unchanged, which is a different request.",
    )

    _tags_must_be_filterable = field_validator("tags")(
        eval_input_tags_must_be_filterable
    )


class EvalsResponse(BaseModel):
    """The evals of a task, plus how many eval files this version of Kiln couldn't read."""

    evals: List[Eval] = Field(description="The evals which loaded successfully.")
    load_error_count: int = Field(
        description="How many eval files failed to load. Usually because they were written by a newer version of Kiln."
    )


class EvalInputsResponse(BaseModel):
    """A task's eval input items, plus how many item files this version of Kiln couldn't read."""

    eval_inputs: List[EvalInput] = Field(
        description="The eval input items which loaded successfully."
    )
    load_error_count: int = Field(
        description="How many eval input files failed to load. Usually because they were written by a newer version of Kiln."
    )


class EvalProgress(BaseModel):
    """Progress information for an eval."""

    dataset_size: int = Field(description="The total size of the eval dataset.")
    golden_dataset_size: int = Field(
        description="The total size of the golden dataset."
    )
    golden_dataset_not_rated_count: int = Field(
        description="Number of unrated golden dataset items."
    )
    golden_dataset_partially_rated_count: int = Field(
        description="Number of partially rated golden dataset items."
    )
    golden_dataset_fully_rated_count: int = Field(
        description="Number of fully rated golden dataset items."
    )
    train_dataset_size: int = Field(
        description="The total size of the train split. 0 when the eval has no train split."
    )
    val_dataset_size: int = Field(
        description="The total size of the val split. 0 when the eval has no val split."
    )
    current_eval_method: EvalConfig | None = Field(
        default=None, description="The currently selected eval config."
    )


class EvalResultSummary(BaseModel):
    """Summary of eval results across run configs."""

    results: Dict[ID_TYPE, Dict[str, ScoreSummary]] = Field(
        description="Scores keyed by run_config_id then output_score_id."
    )
    run_config_percent_complete: Dict[ID_TYPE, float] = Field(
        description="Percent of dataset processed per run config."
    )
    dataset_size: int = Field(description="Total size of the eval dataset.")
    multi_turn_item_count: int = Field(
        description="Items in the eval dataset that are stored multi-turn "
        "conversations. These are scored from their saved conversation, so "
        "every run config receives identical scores for them."
    )


class EvalResultsSummaryEvalInfo(BaseModel):
    """Metadata for a single eval within eval results summary."""

    name: str = Field(description="The eval name.")
    default_judge_config_id: ID_TYPE | None = Field(
        description="The default judge config ID for this eval, if any."
    )
    dataset_size: int = Field(description="Total size of the eval dataset.")
    output_score_keys: list[str] = Field(
        description="The output score keys for this eval."
    )


class EvalResultsSummaryRunConfigInfo(BaseModel):
    """Metadata for a run config within eval results summary."""

    name: str = Field(description="The run config name.")


class EvalResultsSummaryResultCell(BaseModel):
    """Results for a single (eval, run_config) cell."""

    mean_scores: Dict[str, float] = Field(
        description="Mean scores keyed by output_score_key."
    )
    percent_complete: float = Field(
        description="Percent of dataset processed for this run config."
    )


class EvalResultsSummaryResponse(BaseModel):
    """Aggregated eval results across all evals for a task."""

    evals_by_id: Dict[ID_TYPE, EvalResultsSummaryEvalInfo] = Field(
        description="Eval metadata keyed by eval ID."
    )
    run_configs_by_id: Dict[ID_TYPE, EvalResultsSummaryRunConfigInfo] = Field(
        description="Run config metadata keyed by run config ID."
    )
    scores_by_run_config_by_eval: Dict[
        ID_TYPE, Dict[ID_TYPE, EvalResultsSummaryResultCell]
    ] = Field(description="Results keyed by run config ID then eval ID.")


class EvalConfigCompareSummary(BaseModel):
    """Summary comparing eval configs against human ratings."""

    results: Dict[ID_TYPE, Dict[str, CorrelationResult]] = Field(
        description="Correlation results keyed by eval_config_id then output_score_id."
    )
    eval_config_percent_complete: Dict[ID_TYPE, float] = Field(
        description="Percent of dataset processed per eval config."
    )
    dataset_size: int = Field(
        description="Total size of the eval config comparison dataset."
    )
    fully_rated_count: int = Field(description="Number of fully rated dataset items.")
    partially_rated_count: int = Field(
        description="Number of partially rated dataset items."
    )
    not_rated_count: int = Field(description="Number of unrated dataset items.")


class EvalConfigResult(BaseModel):
    """Results for a single eval config."""

    eval_config_id: ID_TYPE = Field(description="The eval config ID.")
    results: Dict[str, ScoreSummary | None] = Field(
        description="Scores keyed by output_score_id. None when no data."
    )
    percent_complete: float = Field(description="Percent of the dataset processed.")
    n_excluded: int = Field(
        default=0,
        description="Number of EvalRuns excluded due to skipped_reason.",
    )


class RunConfigEvalResult(BaseModel):
    """Eval results for a specific run config."""

    eval_id: ID_TYPE = Field(description="The unique identifier of the eval.")
    eval_name: str = Field(description="The human-readable name of the eval.")
    dataset_size: int = Field(description="The dataset size for this eval.")
    eval_config_result: EvalConfigResult | None = Field(
        default=None, description="The eval config results, if available."
    )
    missing_default_eval_config: bool = Field(
        description="Whether the default eval config is missing."
    )
    spec_id: ID_TYPE | None = Field(
        default=None, description="The associated spec ID, if any."
    )


class RunConfigEvalScoresSummary(BaseModel):
    """Summary of all eval scores for a run config."""

    eval_results: List[RunConfigEvalResult] = Field(
        description="Eval results for each eval."
    )
    mean_usage: MeanUsage | None = Field(
        default=None, description="Average usage statistics across eval runs."
    )


def runs_in_filter(
    task: Task, filter_id: DatasetFilterId, readonly: bool
) -> list[TaskRun]:
    # Fetch all the dataset items IDs in a filter
    filter = dataset_filter_from_id(filter_id)
    return [run for run in task.runs(readonly=readonly) if filter(run)]


TaskChildT = TypeVar("TaskChildT", bound=KilnParentedModel)


def load_task_children_by_id(
    model_type: Type[TaskChildT], task: Task, ids: Set[str]
) -> Dict[str, TaskChildT]:
    """Bulk-load a task's children by id, without scanning the directory for nothing.

    The empty guard is not just a micro-optimization: `from_ids_and_parent_path` reads
    every child whose id isn't already cached in order to check it, so calling it with no
    ids to find would read the whole directory off disk. Every bulk load in this module
    goes through here so that guard can't be forgotten at one of them.

    Always readonly: every caller in this module only reads the loaded children, and
    readonly cache hits skip a deep copy per model - traces make those copies large.
    """
    if not ids:
        return {}
    return model_type.from_ids_and_parent_path(ids, task.path, readonly=True)


def summary_eval_config(eval: Eval) -> EvalConfig | None:
    """The eval config a summary reports on: the only one, or the one explicitly chosen.

    None when an eval has several judges and hasn't named a default - there is no
    non-arbitrary way to pick, and the caller reports that state rather than guessing.
    """
    eval_configs = eval.configs(readonly=True)
    if len(eval_configs) == 1:
        return eval_configs[0]
    if eval.current_config_id:
        return next(
            (config for config in eval_configs if config.id == eval.current_config_id),
            None,
        )
    return None


def scored_trace_usage_for_run_config(
    task: Task, evals: List[Eval], run_config_id: str
) -> Dict[str, Usage | None]:
    """Usage per scored TaskRun, for the traces this run config's summary will report on.

    Loaded in one pass for the whole request rather than per eval: each bulk load scans
    the task's `runs/` directory, which now holds every eval trace as well as the
    dataset corpus.

    Only the usage is kept, not the TaskRun. Traces are the large field, and this is the
    only thing read off them here - holding whole runs for the life of the request would
    keep every trace body in memory for a number the summary never prints. Skipped
    records are left out for the same reason: the consuming loop drops them before it
    reads usage at all.
    """
    scored_run_ids: Set[str] = set()
    for eval in evals:
        eval_config = summary_eval_config(eval)
        if eval_config is None:
            continue
        for eval_run in eval_config.runs(readonly=True):
            if (
                eval_run.task_run_config_id == run_config_id
                and eval_run.scored_run_id is not None
                and eval_run.skipped_reason is None
            ):
                scored_run_ids.add(eval_run.scored_run_id)
    traces = load_task_children_by_id(TaskRun, task, scored_run_ids)
    return {run_id: scored_trace_usage(trace) for run_id, trace in traces.items()}


def scored_trace_usage(trace: TaskRun) -> Usage | None:
    """The full generation spend of one scored TaskRun, as a summary reports it.

    Two record shapes need more than `trace.usage`:

    - A multi-turn chain leaf from the dataset (`parent_task_run_id` set) stores
      last-turn-only usage; its conversation totals live in `cumulative_usage`.
      Latency still reads from `usage` — `cumulative_usage` deliberately carries
      none, since per-message latencies don't aggregate meaningfully.
    - An eval-driven conversation stores the synthetic-user driver model's spend
      in `synthetic_user_usage`, beside the assistant-only `usage`. Only its
      **cost** is blended in here, deliberately, even though the field carries
      the driver's tokens and latency too:

      * Cost is total-spend semantics — what this trace cost to produce, both
        models included. That is what a run-config summary should report, and it
        matches migrated legacy traces, which have the blend fused inside
        `usage` with `synthetic_user_usage` None.
      * Tokens are not. The synthetic user is usually a different model on a
        different provider from the agent under test, so folding its ~3.5k input
        tokens per conversation into this figure would attribute them to the
        agent and make `cost / total_tokens` meaningless — the exact conflation
        `synthetic_user_usage` exists to undo.
      * Latency is not. This summary reports how responsive the agent is;
        driver-side wall clock is not the agent's, and summing it would make
        every driven run config look slower than it is.

      Blending everything would also make this field mean two different
      quantities depending on record age: legacy records blend cost only, since
      that is all the old field carried.

    None when the record has nothing to report, so it contributes nothing to an
    average instead of counting as a zero.
    """
    if trace.parent_task_run_id is not None:
        cumulative = trace.cumulative_usage
        base: Usage | None = Usage(
            input_tokens=cumulative.input_tokens if cumulative else None,
            output_tokens=cumulative.output_tokens if cumulative else None,
            total_tokens=cumulative.total_tokens if cumulative else None,
            cost=cumulative.cost if cumulative else None,
            cached_tokens=cumulative.cached_tokens if cumulative else None,
            total_llm_latency_ms=trace.usage.total_llm_latency_ms
            if trace.usage
            else None,
        )
    else:
        base = trace.usage

    if trace.synthetic_user_usage is not None:
        # Cost only — see the docstring. Adding the whole object would fold the
        # driver's tokens and latency into the agent's figures.
        base = (base or Usage()) + Usage(cost=trace.synthetic_user_usage.cost)
    if base is None or all(v is None for v in base.model_dump().values()):
        return None
    return base


def eval_run_task_usage(
    eval_run: EvalRun, usage_by_scored_run_id: Dict[str, Usage | None]
) -> Usage | None:
    """The evaluated task run's usage, from wherever this record keeps it.

    A pointer record whose trace is missing reports None rather than falling back to its
    own (always empty) `task_run_usage`: it contributes nothing to the average instead of
    counting as a zero. That is the same answer as a trace which recorded no usage, which
    is correct - both mean "no usage to report", and the summary treats them alike.
    """
    if eval_run.scored_run_id is None:
        return eval_run.task_run_usage
    return usage_by_scored_run_id.get(eval_run.scored_run_id)


def eval_item_input_text(item: TaskRun | EvalInput) -> str | None:
    """What a dataset item's input reads as, for display.

    Only used as the last resort when no trace can supply the input, so it answers
    "what was this item asking for" rather than "what exactly was sent to the model" —
    the adapter may reserialize a structured input on the way through.
    """
    if isinstance(item, TaskRun):
        return item.input
    if isinstance(item.data, SingleTurnEvalInputData):
        return item.data.user_message.text
    if isinstance(item.data, MultiTurnSyntheticEvalInputData):
        return item.data.first_message.text if item.data.first_message else None
    return None


def resolve_eval_run_traces(
    task: Task, eval_runs: List[EvalRun]
) -> List[EvalRunWithTrace]:
    """Join each score record to the trace it was computed over.

    Pointer records name a TaskRun; legacy records carry the trace inline. A dangling
    `scored_run_id` degrades to nulls rather than raising - the score still renders and
    still aggregates, only the drill-through is unavailable.

    `input` gets one extra fallback the other three fields don't: the dataset item the
    score was recorded against. A run skipped before anything was generated has neither
    an inline input nor a trace to join to, so without this it would render blank
    forever. The source item is a true statement of what was asked either way, while
    nothing but the trace can say what the model produced.
    """
    scored_run_ids = {
        eval_run.scored_run_id
        for eval_run in eval_runs
        if eval_run.scored_run_id is not None
    }
    traces = load_task_children_by_id(TaskRun, task, scored_run_ids)
    if len(traces) < len(scored_run_ids):
        # Delete protection means a trace shouldn't go missing while its score exists, so
        # this is either data lost out-of-band or a project imported with
        # `package-project --exclude-task-runs` (architecture 4.1). Counted rather than
        # logged per record: one line says the same thing without a flood.
        logger.warning(
            f"{len(scored_run_ids) - len(traces)} of {len(scored_run_ids)} eval traces "
            f"referenced by task {task.id} are missing. Their scores still render, "
            "without the run that produced them."
        )

    resolved = [
        EvalRunWithTrace.from_own_fields(eval_run)
        if eval_run.scored_run_id is None
        else EvalRunWithTrace.from_scored_run(
            eval_run, traces.get(eval_run.scored_run_id)
        )
        for eval_run in eval_runs
    ]

    _fill_missing_inputs_from_source_items(task, resolved)
    return resolved


def _fill_missing_inputs_from_source_items(
    task: Task, resolved: List[EvalRunWithTrace]
) -> None:
    """Fill in `input` for the records no trace could supply one for, in two bulk loads.

    Bulk rather than per record: each lookup scans the task's `runs/` directory, which
    now holds every eval trace alongside the dataset corpus.
    """
    needs_input = [result for result in resolved if result.input is None]
    if not needs_input:
        return

    dataset_ids = {
        result.eval_run.dataset_id
        for result in needs_input
        if result.eval_run.dataset_id is not None
    }
    eval_input_ids = {
        result.eval_run.eval_input_id
        for result in needs_input
        if result.eval_run.eval_input_id is not None
    }
    source_runs = load_task_children_by_id(TaskRun, task, dataset_ids)
    source_inputs = load_task_children_by_id(EvalInput, task, eval_input_ids)

    for result in needs_input:
        source: TaskRun | EvalInput | None = None
        if result.eval_run.dataset_id is not None:
            source = source_runs.get(result.eval_run.dataset_id)
        elif result.eval_run.eval_input_id is not None:
            source = source_inputs.get(result.eval_run.eval_input_id)
        if source is not None:
            result.input = eval_item_input_text(source)


def resolved_split_or_422(
    task: Task, eval: Eval, split: EvalSplitName
) -> ResolvedSplit:
    """The items of one of the eval's splits, or a 422 naming the split and the eval.

    An eval without the split asked for is a client error, not a server one: the caller
    named a split this eval doesn't have.

    On the SSE run endpoint, resolving before the StreamingResponse is what makes the 422
    reachable at all — see require_golden_set_or_422.
    """
    resolved = resolve_split(task, eval, split)
    if resolved is None:
        raise HTTPException(
            status_code=422,
            detail=f"Eval '{eval.id}' has no '{split}' split.",
        )
    return resolved


def split_size(split: ResolvedSplit | None) -> int:
    """How many items a split has, with an absent split reported as zero.

    Spelled out rather than written inline as `len(split) if split else 0`: ResolvedSplit
    defines __len__, so an empty split is falsy and truthiness cannot tell "this eval has
    no val split" from "its val split matched nothing". Both answer 0 here, but only
    because the absence check is `is None`.
    """
    return len(split) if split is not None else 0


def _cached_test_split(
    task: Task, eval: Eval, cache: Dict[Tuple[ItemSource, str], ResolvedSplit]
) -> ResolvedSplit | None:
    """The eval's test split, reusing an equivalent resolution from `cache`.

    Two evals resolve to the same items exactly when their splits name the same filter
    over the same store, so that pair is the cache key. A hit is re-stamped with this
    eval's id: ResolvedSplit carries the eval it was resolved from so a consumer can
    check a split belongs to the eval it is working on, and a cached value handed on
    unchanged would name whichever eval reached the filter first.

    `task` is deliberately NOT part of the key, so a cache must not outlive one task: the
    same (source, filter_id) selects different items in a different task. The one caller
    builds the cache inside a single request and passes the task it loaded, which holds
    that invariant positionally. A second caller has to keep it too, or key on the task.
    """
    split_ref = eval.splits.get("test")
    if split_ref is None:
        return None
    key = (split_ref.source, split_ref.filter_id)
    cached = cache.get(key)
    if cached is None:
        resolved = resolve_split(task, eval, "test")
        if resolved is None:
            return None
        cache[key] = resolved
        return resolved
    return cached if cached.eval_id == eval.id else replace(cached, eval_id=eval.id)


def require_golden_set_or_422(eval: Eval) -> DatasetFilterId:
    """The eval's golden filter id, or 422 when none is configured.

    Judge comparison scores against the golden set; returning the narrowed id
    lets callers use it without re-checking for None.

    Checked here rather than left to EvalRunner because these are SSE endpoints: the
    response is a StreamingResponse over a generator, so anything raised once the
    generator is running is emitted after a 200 status and an empty body, and the
    refusal never appears in the response at all. Refusing before the response is built
    is what makes the status code and detail part of the HTTP contract.

    This reaches the user's screen too, but only because the web UI reads these endpoints
    with `$lib/utils/sse_stream`'s fetch-based reader rather than a browser EventSource.
    EventSource cannot see the status or body of a non-200 — it fires `onerror` with a
    bare Event, which `createKilnError` renders as "Unknown error", so every refusal here
    was invisible until that client changed.
    """
    if eval.eval_configs_filter_id is None:
        raise HTTPException(status_code=422, detail=no_golden_set_message(eval))
    return eval.eval_configs_filter_id


def eval_grades_against_reference_data(data_type: EvalDataType | None) -> bool:
    """Whether an eval's data type means its V1 judges grade against ground truth.

    An exhaustive match rather than an `== reference_answer` equality test: a fourth
    `EvalDataType` that needs reference data would answer False silently, which is the
    same failure the "anything that isn't v2" spelling below exists to prevent, one level
    down. Adding a member fails `ty` here until it is classified.

    `None` means the eval never declared one, so nothing asks for a reference.
    """
    match data_type:
        case EvalDataType.reference_answer:
            return True
        case EvalDataType.final_answer | EvalDataType.full_trace:
            return False
        case None:
            return False
        case _:
            raise_exhaustive_enum_error(data_type)


def judge_requires_reference_data(eval: Eval, eval_config: EvalConfig) -> bool:
    """Whether this judge grades against reference data, which judge comparison has none of.

    Judge comparison scores each golden dataset item as itself, so `EvalTaskInput.from_trace`
    yields `reference_data = None` by design — populating it would make the reference
    byte-identical to the output being graded, and every judge would pass every item.

    Two judge kinds hit that, so one predicate covers both:

    - a V2 judge declaring reference keys: `check_reference_key` turns each item into a
      `missing_reference_key` skip, which `run_job` reports as success and `_persist_score`
      writes as a durable scoreless `EvalRun` — a "Complete" row with no scores, and a
      record that suppresses any later re-run.
    - a V1 judge on a `reference_answer` eval: `_run_legacy_job` calls `run_eval` without
      the item, so `GEval` raises for every job.

    The V1 arm is spelled "anything that isn't v2" for the same reason as
    `judge_scores_dataset_runs`: a judge type added to the enum later is refused here
    rather than reaching one of those two failures.

    Mirrored client-side by `compute_run_disallowed_missing_ref_data`
    (`app/web_ui/src/lib/utils/eval_types/judge_comparison_gate.ts`), which decides what
    the Compare Judges page shows. This is what decides whether the run happens.
    """
    if eval_config.config_type != EvalConfigType.v2:
        return eval_grades_against_reference_data(eval.evaluation_data_type)
    if not isinstance(eval_config.properties, V2_PROPERTY_TYPES):
        return False
    return len(reference_data_keys(eval_config.properties)) > 0


def no_comparable_judges_message(eval: Eval) -> str:
    """Why judge comparison has nothing to run for this eval. One wording, one raiser.

    Named for the eval-level fact it states, alongside `no_golden_set_message`. A
    per-judge refusal cannot share it: in a mixed set "every judge it has" is false, so
    such a caller needs its own per-judge wording rather than this one.
    """
    return (
        f"Eval '{eval.name}' has no judges that can be compared. Every judge it has grades "
        "against reference data, and comparing judges scores each golden dataset item as "
        "itself — there is no separate reference answer to compare against, so every item "
        "would be skipped without a score. Add a judge that grades the output on its own "
        "to compare judges for this eval."
    )


def comparable_eval_configs_or_422(eval: Eval) -> List[EvalConfig]:
    """The eval's judges minus the ones judge comparison can't score.

    A mixed set still runs: dropping the judges that can't be scored is what lets the
    others be compared. Only an eval where nothing is left is refused.

    Refused here rather than inside the runner for the reason spelled out on
    `require_golden_set_or_422`: this is an SSE endpoint, so anything raised once the
    StreamingResponse's generator is running arrives after a 200 with an empty body. It
    also has to beat the runner because the runner's first act is to write a durable
    scoreless `EvalRun` per item — records nothing in the UI clears, which then read as
    "already run" and suppress the re-run a later fix would need (functional spec 6.2).

    An eval with no judges at all is left to `EvalRunner`, which already names that case.
    """
    eval_configs = eval.configs()
    comparable = [
        eval_config
        for eval_config in eval_configs
        if not judge_requires_reference_data(eval, eval_config)
    ]
    if eval_configs and not comparable:
        raise HTTPException(status_code=422, detail=no_comparable_judges_message(eval))
    return comparable


def judge_scores_dataset_runs(config_type: EvalConfigType) -> bool:
    """Whether a judge of this type scores the output stored on a dataset run (TaskRun).

    Spelled as "anything that isn't v2" to match `EvalRunner.run_job`'s dispatch exactly:
    every type it sends to `_run_legacy_job` needs a TaskRun item, so a judge type added
    to the enum later is refused by the guards below rather than reaching that job's
    `ValueError` — which `AsyncJobRunner` turns into an unexplained per-job error count.
    """
    return config_type != EvalConfigType.v2


def judge_needs_dataset_runs_message(eval: Eval, config_type: EvalConfigType) -> str:
    """Why a V1 judge can't score this eval. One wording, two raisers.

    Shared between the creation guard and the run guard so the refusal a user gets when
    they attach the judge and the refusal they get when an already-attached one is run say
    the same thing, the way `no_golden_set_message` is shared with EvalRunner.

    Says "new eval dataset format" rather than naming EvalInputs and TaskRuns: those are
    model names, not words the UI uses anywhere, and the distinction between the two
    stores isn't one the user made or can act on. The UI's own refusals for the same
    situation were rewritten the same way.
    """
    return (
        f"Eval '{eval.name}' uses our new eval dataset format, which the "
        f"'{config_type.value}' judge type can't score. Choose a judge type that supports "
        "the new format."
    )


def require_dataset_run_test_split_or_400(
    eval: Eval, config_type: EvalConfigType
) -> None:
    """400 when a V1 judge is being attached to an eval whose test split isn't TaskRun-backed.

    Refusing at creation is the primary fix: without it the mismatch is only discovered
    once per job at run time, where every job raises inside `_run_legacy_job` and
    `AsyncJobRunner` reports them as a bare error count with no reason.

    Phrased as "present but not usable" rather than "is an EvalInputSplit" — the same
    choice, for the same reason, as `reject_unusable_train_splits` in
    prompt_optimization_job_api: a SplitRef variant added later is refused by default
    instead of slipping through into the failure this exists to prevent. A missing test
    split is left alone: `Eval.validate_splits` already requires one, and this guard is
    not where that is re-litigated.
    """
    if not judge_scores_dataset_runs(config_type):
        return
    test_split = eval.splits.get("test")
    if test_split is not None and not isinstance(test_split, TaskRunSplit):
        raise HTTPException(
            status_code=400,
            detail=judge_needs_dataset_runs_message(eval, config_type),
        )


def require_dataset_run_items_or_400(
    eval: Eval, config_type: EvalConfigType, split: ResolvedSplit
) -> None:
    """400 when the split about to be run holds items a V1 judge can't score.

    The creation guard can't help an eval that already carries such a config — written by
    a build predating that guard, or straight into a project file — so the run path checks
    the same thing again, against the items it actually resolved rather than the split ref
    it declared.

    Checked here rather than left to EvalRunner for the reason spelled out on
    require_golden_set_or_422: this is an SSE endpoint, so anything raised once the
    StreamingResponse's generator is running arrives after a 200 and never appears as a
    status code at all. Tested as "not task_run" so a new ItemSource fails closed.
    """
    if not judge_scores_dataset_runs(config_type):
        return
    if split.source != "task_run":
        raise HTTPException(
            status_code=400,
            detail=judge_needs_dataset_runs_message(eval, config_type),
        )


def build_score_key_to_task_requirement_id(task: Task) -> Dict[str, ID_TYPE]:
    # Create a map of score_key -> Task requirement ID
    score_key_to_task_requirement_id: Dict[str, ID_TYPE] = {}

    for task_requirement in task.requirements:
        score_key = string_to_json_key(task_requirement.name)
        score_key_to_task_requirement_id[score_key] = task_requirement.id
    return score_key_to_task_requirement_id


def human_score_from_task_run(
    task_run: TaskRun,
    score: EvalOutputScore,
    score_key_to_task_requirement_id: Dict[str, ID_TYPE],
) -> float | None:
    if not task_run.output.rating:
        return None
    score_key = score.json_key()

    # Overall rating
    if score_key == "overall_rating":
        return task_run.output.rating.value

    # Task requirement ratings. A requirement whose name matches the score
    # key may still be unrated — fall through to the named lookup rather
    # than letting the name collision hide a named rating for this score.
    req_id = score_key_to_task_requirement_id.get(score_key, None)
    if req_id:
        req_rating = task_run.output.rating.requirement_ratings.get(req_id, None)
        if req_rating is not None:
            return req_rating.value

    # Named ratings
    named_score_id = f"named::{score.name}"
    named_rating = task_run.output.rating.requirement_ratings.get(named_score_id, None)
    if named_rating is not None:
        return named_rating.value
    return None


def count_human_evals(
    items: List[TaskRun],
    eval: Eval,
    score_key_to_task_requirement_id: Dict[str, ID_TYPE],
) -> Tuple[int, int, int]:
    # Track how often we are missing human evals in dataset items
    fully_rated_count: int = 0
    partially_rated_count: int = 0
    not_rated_count: int = 0
    for dataset_item in items:
        has_all_scores = True
        has_any_scores = False
        for output_score in eval.output_scores:
            score = human_score_from_task_run(
                dataset_item, output_score, score_key_to_task_requirement_id
            )
            if score is None:
                has_all_scores = False
            else:
                has_any_scores = True

        if not has_any_scores:
            not_rated_count += 1
        elif has_all_scores:
            fully_rated_count += 1
        else:
            partially_rated_count += 1

    return fully_rated_count, partially_rated_count, not_rated_count


def score_summary_from_values(values: list[float], n_excluded: int) -> ScoreSummary:
    """Build a ScoreSummary from the raw per-run scores of one output score key.

    `values` must already exclude skipped runs and runs missing this score —
    the caller does that filtering, exactly as it always has for the mean.

    Percentiles use linear interpolation between the two nearest order
    statistics (`statistics_lib.percentile`), matching the numpy.percentile /
    statistics.quantiles(method="inclusive") default. So an even-length list's
    median is the average of the two middle values, and p90 of a short list is
    interpolated rather than snapped to an existing datum. This is the only
    percentile definition used anywhere in eval aggregation — do not mix in
    another.

    Empty `values` yields None for every statistic (never 0.0, which would read
    downstream as a real datum rather than "no data").
    """
    count = len(values)
    if count == 0:
        return ScoreSummary(mean_score=None, n_used=0, n_excluded=n_excluded)
    return ScoreSummary(
        mean_score=sum(values) / count,
        min_score=min(values),
        p25_score=percentile(values, 25),
        median_score=percentile(values, 50),
        p75_score=percentile(values, 75),
        p90_score=percentile(values, 90),
        max_score=max(values),
        n_used=count,
        n_excluded=n_excluded,
    )


def compute_score_summary(
    eval: Eval,
    eval_config: EvalConfig,
    task_run_configs: list[TaskRunConfig],
    split: ResolvedSplit,
) -> EvalResultSummary:
    """Aggregate an eval config's runs over exactly one of the eval's splits.

    Takes the resolved split rather than a set of ids so the aggregate is scoped to one
    store as well as one item set: a run is counted only when the item it scored is in
    this split, keyed on (source, id). A bare id would let an EvalInput's score be
    averaged into a TaskRun-backed split's mean, which no reader could then detect
    (functional spec 5.3).
    """
    # Stored multi-turn conversations (runs with parent_task_run_id set) are
    # judged on their saved trace, so their scores can't vary across run
    # configs; the UI calls this out per summary. Only a TaskRun-backed split
    # can contain them — EvalInput items are re-driven per run config.
    # getattr rather than direct access: split.items is a TaskRun/EvalInput
    # union (and test stubs), and only TaskRuns can be chain leaves. The
    # positive-case tests below pin the field name against renames.
    multi_turn_item_count = (
        sum(
            1
            for item in split.items
            if getattr(item, "parent_task_run_id", None) is not None
        )
        if split.source == "task_run"
        else 0
    )

    split_items = split.item_keys()
    if len(split_items) == 0:
        return EvalResultSummary(
            results={},
            run_config_percent_complete={},
            dataset_size=0,
            multi_turn_item_count=multi_turn_item_count,
        )

    remaining_expected_items: Dict[ID_TYPE, Set[ItemKey]] = {
        run_config.id: split_items.copy() for run_config in task_run_configs
    }
    partial_incomplete_counts: Dict[ID_TYPE, int] = {
        run_config.id: 0 for run_config in task_run_configs
    }

    # run_config_id -> output_score_json_key -> the individual scores, kept as a
    # list (not a running total) so the summary can report percentiles as well
    # as the mean.
    score_values: Dict[ID_TYPE, Dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    excluded_counts: Dict[ID_TYPE, int] = defaultdict(int)

    for eval_run in eval_config.runs(readonly=True):
        if eval_run.task_run_config_id is None:
            continue
        run_config_id = eval_run.task_run_config_id

        if run_config_id not in remaining_expected_items:
            continue
        item_key = eval_run_item_key(eval_run)
        if item_key not in remaining_expected_items[run_config_id]:
            continue
        else:
            remaining_expected_items[run_config_id].remove(item_key)

        if eval_run.skipped_reason is not None:
            excluded_counts[run_config_id] += 1
            _ = score_values[run_config_id]
            continue

        incomplete = False
        # Ensure this run_config_id has an entry even if no scores match
        _ = score_values[run_config_id]
        for output_score in eval.output_scores:
            score_key = output_score.json_key()
            if score_key in eval_run.scores:
                score_values[run_config_id][score_key].append(
                    eval_run.scores[score_key]
                )
            else:
                incomplete = True

        if incomplete:
            partial_incomplete_counts[run_config_id] += 1

    all_score_keys = [os.json_key() for os in eval.output_scores]

    results: Dict[ID_TYPE, Dict[str, ScoreSummary]] = {}
    for run_config_id, output_scores in score_values.items():
        results[run_config_id] = {}
        n_excluded = excluded_counts[run_config_id]
        for score_key in all_score_keys:
            values = output_scores.get(score_key, [])
            if len(values) > 0 or n_excluded > 0:
                results[run_config_id][score_key] = score_summary_from_values(
                    values, n_excluded
                )

    run_config_percent_complete: Dict[ID_TYPE, float] = {}
    for run_config in task_run_configs:
        n_excluded = excluded_counts[run_config.id]
        incomplete_count = partial_incomplete_counts[run_config.id] + len(
            remaining_expected_items[run_config.id]
        )
        n_processed = len(split_items) - incomplete_count
        percent_complete = (n_processed) / len(split_items)
        run_config_percent_complete[run_config.id] = percent_complete

    return EvalResultSummary(
        results=results,
        run_config_percent_complete=run_config_percent_complete,
        dataset_size=len(split_items),
        multi_turn_item_count=multi_turn_item_count,
    )


async def validate_run_config_tools(
    task: Task, run_config_properties: RunConfigProperties
) -> None:
    """Reject a run config whose tools or skills would collide at runtime.

    Tool and skill names may repeat across a project, but within one run
    config every tool must expose a unique function name and every skill a
    unique skill name. Validating here catches collisions at selection time,
    instead of at inference time; the runtime check remains the backstop for
    tools renamed after the run config was created.
    """
    try:
        await validate_run_config_tool_names(task, run_config_properties)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


def connect_evals_api(app: FastAPI):
    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/create_evaluator",
        summary="Create Evaluator",
        tags=["Evals"],
        openapi_extra=ALLOW_AGENT,
    )
    async def create_evaluator(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        request: CreateEvaluatorRequest,
    ) -> Eval:
        task = task_from_id(project_id, task_id)

        # When no eval set filter is provided, generate the same tag-based
        # splits a spec-backed eval would get, so downstream features (synth
        # data gen, fine-tuning, golden sets) work identically.
        eval_configs_filter_id = request.eval_configs_filter_id
        splits: dict[str, SplitRef]
        if request.eval_set_filter_id is not None:
            # Naming a filter is the caller opting out of generation: they get the test
            # split they asked for and nothing else. Train and val stay absent rather
            # than being minted from a name the caller never chose -- an unconfigured
            # split has no backing store to pick, and materializing an empty one turns
            # "this eval has no train set" into "this eval has an empty train set",
            # which reads as configured and isn't. See eval_splits_v1_v2 spec 3.2.
            splits = {"test": TaskRunSplit(filter_id=request.eval_set_filter_id)}
        else:
            tags = generate_spec_eval_tags(request.name)
            # Eval.splits is keyed by str, and dict key types are invariant, so the
            # narrower mapping has to be widened rather than passed through.
            splits = {
                split_name: split
                for split_name, split in spec_eval_splits(
                    test_tag=tags.test_tag,
                    train_tag=tags.train_tag,
                    val_tag=tags.val_tag,
                ).items()
            }
            if eval_configs_filter_id is None:
                eval_configs_filter_id = tag_filter_id(tags.golden_tag)

        output_scores = (
            request.output_scores
            if request.output_scores is not None
            else [eval_pass_fail_output_score(request.name)]
        )

        eval = Eval(
            name=request.name,
            description=request.description,
            template=request.template,
            output_scores=output_scores,
            splits=splits,
            eval_configs_filter_id=eval_configs_filter_id,
            template_properties=request.template_properties,
            evaluation_data_type=request.evaluation_data_type,
            priority=request.priority,
            status=request.status,
            parent=task,
        )
        eval.save_to_file()
        return eval

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/run_configs",
        summary="List Run Configs",
        tags=["Run Configs"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_run_configs(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
    ) -> list[TaskRunConfig]:
        return get_all_run_configs(project_id, task_id)

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}",
        summary="Get Eval",
        tags=["Evals"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_eval(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
    ) -> Eval:
        return resolve_eval_display_fields(eval_from_id(project_id, task_id, eval_id))

    @app.delete(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}",
        summary="Delete Eval",
        tags=["Evals"],
        openapi_extra=DENY_AGENT,
    )
    async def delete_eval(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
    ) -> None:
        eval = eval_from_id(project_id, task_id, eval_id)
        eval.delete()

    @app.patch(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}",
        summary="Update Eval",
        tags=["Evals"],
        openapi_extra=agent_policy_require_approval(
            "Allow agent to edit eval? Ensure you backup your project before allowing agentic edits."
        ),
    )
    async def update_eval(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
        request: UpdateEvalRequest,
    ) -> Eval:
        eval = eval_from_id(project_id, task_id, eval_id)

        if request.name is not None:
            eval.name = request.name
        if request.description is not None:
            eval.description = request.description
        if request.priority is not None:
            eval.priority = request.priority
        if request.status is not None:
            eval.status = request.status

        # legacy evals (not created with Specs) do not have a train split, but we need one
        # for some features such as prompt optimization
        if request.train_set_filter_id is not None:
            # if the eval already has a train split, we do not allow changing it because it
            # would make comparing results before and after the change very confusing
            if eval.splits.get("train") is not None:
                raise HTTPException(
                    status_code=400,
                    detail="Train set filter is already set and cannot be changed. Please create a new eval if you need a different train set.",
                )
            # set_split, not a direct splits write: it refuses to mutate a readonly
            # (cached) eval, which a direct `eval.splits[...] = ...` would do silently.
            eval.set_split("train", TaskRunSplit(filter_id=request.train_set_filter_id))

        eval.save_to_file()
        return resolve_eval_display_fields(eval)

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/evals",
        summary="List Evals",
        tags=["Evals"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_evals(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
    ) -> EvalsResponse:
        """List all evals for a task."""
        task = task_from_id(project_id, task_id)
        # Partial load: a project folder synced from a newer Kiln can contain an eval this
        # build can't parse. Return the readable evals rather than failing the whole list.
        evals, load_errors = Eval.all_children_of_parent_path_with_errors(task.path)
        for load_error in load_errors:
            # The response only carries a count, so log each failure with its path and
            # reason - it is the only way to tell a corrupt file from a version mismatch.
            logger.warning(
                f"Failed to load eval file {load_error.path}: {load_error.message}"
            )
        specs_by_eval_id = {
            spec.eval_id: spec for spec in task.specs(readonly=True) if spec.eval_id
        }
        for eval in evals:
            resolve_eval_display_fields(eval, specs_by_eval_id.get(eval.id))
        return EvalsResponse(evals=evals, load_error_count=len(load_errors))

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/eval_default_judge_types",
        summary="Get Default Judge Types",
        tags=["Evals"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_eval_default_judge_types(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
    ) -> dict[str, str]:
        """Map of eval ID to its default judge's type discriminator.

        V2 configs report their properties type (e.g. "code_eval",
        "llm_judge"); legacy configs report their config_type (e.g. "g_eval").
        Evals with no default judge are omitted. Used by the evals list to
        display each eval's type without fetching every config.
        """
        task = task_from_id(project_id, task_id)
        result: dict[str, str] = {}
        for eval in task.evals(readonly=True):
            if not eval.id or not eval.current_config_id:
                continue
            for config in eval.configs(readonly=True):
                if config.id != eval.current_config_id:
                    continue
                properties = config.properties
                if properties is None or isinstance(properties, dict):
                    result[eval.id] = config.config_type.value
                else:
                    result[eval.id] = properties.type.value
                break
        return result

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/eval_inputs",
        summary="List Eval Inputs",
        tags=["Eval Inputs"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_eval_inputs(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        filter_id: Annotated[
            EvalInputFilterId | None,
            Query(
                description="Optional eval-input filter to apply, e.g. 'all' or 'tag::my_tag' (the same IDs evals use as eval_input_filter_id)."
            ),
        ] = None,
    ) -> EvalInputsResponse:
        """List a task's eval input items, optionally restricted to a filter."""
        task = task_from_id(project_id, task_id)
        # Partial load: a project folder synced from a newer Kiln can contain an item
        # this build can't parse. Return the readable items rather than failing the
        # whole list, which would take the corpus away over one bad file.
        eval_inputs, load_errors = EvalInput.all_children_of_parent_path_with_errors(
            task.path, readonly=True
        )
        for load_error in load_errors:
            # The response only carries a count, so log each failure with its path and
            # reason - it is the only way to tell a corrupt file from a version mismatch.
            logger.warning(
                f"Failed to load eval input file {load_error.path}: {load_error.message}"
            )
        if filter_id is not None:
            try:
                filter = eval_input_filter_from_id(filter_id)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))
            eval_inputs = [
                eval_input for eval_input in eval_inputs if filter(eval_input)
            ]
        return EvalInputsResponse(
            eval_inputs=eval_inputs, load_error_count=len(load_errors)
        )

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/eval_inputs/{eval_input_id}",
        summary="Get Eval Input",
        tags=["Eval Inputs"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_eval_input(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_input_id: Annotated[
            str, Path(description="The unique identifier of the eval input.")
        ],
    ) -> EvalInput:
        return eval_input_from_id(project_id, task_id, eval_input_id)

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/eval_inputs",
        summary="Create Eval Input",
        tags=["Eval Inputs"],
        openapi_extra=ALLOW_AGENT,
    )
    async def create_eval_input(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        request: CreateEvalInputRequest,
    ) -> EvalInput:
        """Create an eval input item. Evals pick it up via their eval_input_filter_id, so tag it accordingly."""
        task = task_from_id(project_id, task_id)
        eval_input = EvalInput(
            data=request.data,
            reference=request.reference,
            tags=request.tags,
            parent=task,
        )
        eval_input.save_to_file()
        return eval_input

    @app.patch(
        "/api/projects/{project_id}/tasks/{task_id}/eval_inputs/{eval_input_id}",
        summary="Update Eval Input",
        tags=["Eval Inputs"],
        openapi_extra=agent_policy_require_approval(
            "Allow agent to edit eval inputs? Tags decide which slice an item falls into, and reference data is the ground truth an item is scored against, so both change what an eval reports. Ensure you backup your project before allowing agentic edits."
        ),
    )
    async def update_eval_input(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_input_id: Annotated[
            str, Path(description="The unique identifier of the eval input.")
        ],
        request: UpdateEvalInputRequest,
    ) -> EvalInput:
        """Update an eval input item's tags and/or reference data.

        `data` is not editable and sending it is a 422 — see UpdateEvalInputRequest for
        why the scenario is the one field that can't change in place.

        Reads `model_fields_set` rather than testing each field for None, because for
        `reference` the two are genuinely different requests: omitting it leaves ground
        truth alone, sending null clears it. Testing for None would make clearing
        impossible and silently look like a successful no-op.
        """
        eval_input = eval_input_from_id(project_id, task_id, eval_input_id)
        provided = request.model_fields_set

        if "tags" in provided:
            if request.tags is None:
                # Not silently ignored: a client sending null here means to remove the
                # tags, and an empty list is how that is spelled. Leaving it unchanged
                # would drop an intended edit.
                raise HTTPException(
                    status_code=422,
                    detail="tags cannot be null. Send [] to remove every tag, or omit the field to leave tags unchanged.",
                )
            eval_input.tags = request.tags
        if "reference" in provided:
            eval_input.reference = request.reference

        eval_input.save_to_file()
        return eval_input

    @app.delete(
        "/api/projects/{project_id}/tasks/{task_id}/eval_inputs/{eval_input_id}",
        summary="Delete Eval Input",
        tags=["Eval Inputs"],
        openapi_extra=DENY_AGENT,
    )
    async def delete_eval_input(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_input_id: Annotated[
            str, Path(description="The unique identifier of the eval input.")
        ],
    ) -> None:
        """Delete an eval input item, if nothing on disk still points at it.

        409 when anything does. Both kinds of reference name the item by id and hold no
        copy of it, so a delete that went through would leave records describing content
        that no longer exists — an eval trace whose scenario is gone, or a score whose
        input can't be read back. To take a referenced item out of an eval's scope,
        retag it with PATCH instead; to correct its ground truth, PATCH its reference.
        """
        task = task_from_id(project_id, task_id)
        eval_input = eval_input_from_id(project_id, task_id, eval_input_id)

        references = eval_input_references(task, eval_input_id)
        if references:
            raise HTTPException(
                status_code=409, detail=references_conflict_detail(references)
            )

        eval_input.delete()

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/eval_configs",
        summary="List Eval Configs",
        tags=["Evals"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_eval_configs(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
    ) -> list[EvalConfig]:
        eval = eval_from_id(project_id, task_id, eval_id)
        return eval.configs()

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/eval_config/{eval_config_id}",
        summary="Get Eval Config",
        tags=["Evals"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_eval_config(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
        eval_config_id: Annotated[
            str, Path(description="The unique identifier of the eval configuration.")
        ],
    ) -> EvalConfig:
        eval_config = eval_config_from_id(project_id, task_id, eval_id, eval_config_id)
        return eval_config

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/run_configs",
        summary="Create Run Config",
        tags=["Run Configs"],
        openapi_extra=ALLOW_AGENT,
    )
    async def create_task_run_config(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        request: CreateTaskRunConfigRequest,
    ) -> TaskRunConfig:
        task = task_from_id(project_id, task_id)
        name = request.name or generate_memorable_name()
        await validate_run_config_tools(task, request.run_config_properties)

        parent_project = task.parent_project()
        if parent_project is None:
            raise HTTPException(
                status_code=400,
                detail="Task must have a parent project.",
            )

        frozen_prompt: BasePrompt | None = None
        reused_frozen_prompt_id: str | None = None
        run_config_properties = request.run_config_properties
        if isinstance(run_config_properties, KilnAgentRunConfigProperties):
            prompt_id = run_config_properties.prompt_id
            if not is_frozen_prompt(prompt_id):
                # For dynamic prompts, we "freeze" a copy of this prompt into the task run config so we don't accidentally invalidate evals if the user changes something that impacts the prompt (example: changing data for multi-shot, or changing task for basic-prompt)
                # We then point the task_run_config.run_properties.prompt_id to this frozen prompt
                prompt_builder = prompt_builder_from_id(prompt_id, task)
                prompt_text = prompt_builder.build_base_prompt()
                cot_instructions = prompt_builder.chain_of_thought_prompt()
                # Reuse an existing frozen prompt with identical content instead of
                # creating a duplicate, to avoid cluttering the prompts list.
                reused_frozen_prompt_id = reusable_frozen_prompt_id(
                    task, parent_project.id, prompt_text, cot_instructions
                )
                if reused_frozen_prompt_id is None:
                    # Bake the source prompt's type into the name (e.g.
                    # "Gusty Forest - Basic (Zero Shot)") so it's identifiable
                    # without resolving the generator at render time.
                    label = generator_label(prompt_id)
                    memorable_name = generate_memorable_name()
                    prompt_name = (
                        f"{memorable_name} - {label}" if label else memorable_name
                    )
                    frozen_prompt = BasePrompt(
                        name=prompt_name,
                        description=f"Frozen copy of prompt '{prompt_id}'.",
                        generator_id=prompt_id,
                        prompt=prompt_text,
                        chain_of_thought_instructions=cot_instructions,
                    )
        task_run_config = TaskRunConfig(
            parent=task,
            name=name,
            run_config_properties=run_config_properties,
            description=request.description,
            prompt=frozen_prompt,
        )
        if isinstance(
            task_run_config.run_config_properties, KilnAgentRunConfigProperties
        ):
            if frozen_prompt is not None:
                # Set after, because the ID isn't known until the TaskRunConfig is created
                task_run_config.run_config_properties.prompt_id = f"task_run_config::{parent_project.id}::{task.id}::{task_run_config.id}"
            elif reused_frozen_prompt_id is not None:
                # Point at the existing frozen prompt we're reusing
                task_run_config.run_config_properties.prompt_id = (
                    reused_frozen_prompt_id
                )
        task_run_config.save_to_file()
        return task_run_config

    @app.patch(
        "/api/projects/{project_id}/tasks/{task_id}/run_configs/{run_config_id}",
        summary="Update Run Config",
        tags=["Run Configs"],
        openapi_extra=agent_policy_require_approval(
            "Allow agent to edit run config? Ensure you backup your project before allowing agentic edits."
        ),
    )
    async def update_run_config(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        run_config_id: Annotated[
            str, Path(description="The unique identifier of the run configuration.")
        ],
        request: UpdateRunConfigRequest,
    ) -> TaskRunConfig:
        run_config = task_run_config_from_id(project_id, task_id, run_config_id)
        if run_config.path is None:
            raise HTTPException(
                status_code=400,
                detail="Cannot update this run config.",
            )
        if request.name is not None:
            run_config.name = request.name
        if request.starred is not None:
            run_config.starred = request.starred
        if request.prompt_name is not None:
            if run_config.prompt is None:
                raise HTTPException(
                    status_code=400,
                    detail="Run config has no frozen prompt to rename.",
                )
            run_config.prompt = run_config.prompt.model_copy(
                update={"name": request.prompt_name}
            )
        run_config.save_to_file()
        return run_config

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/create_eval_config",
        summary="Create Eval Config",
        tags=["Evals"],
        openapi_extra=ALLOW_AGENT,
    )
    async def create_eval_config(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
        request: CreateEvalConfigRequest,
    ) -> EvalConfig:
        if request.type in (EvalConfigType.g_eval, EvalConfigType.llm_as_judge):
            if not request.model_name or not request.provider:
                raise HTTPException(
                    status_code=400,
                    detail="model_name and provider are required for LLM-based eval types.",
                )

        eval = eval_from_id(project_id, task_id, eval_id)
        require_dataset_run_test_split_or_400(eval, request.type)
        name = request.name or generate_memorable_name()

        try:
            eval_config = EvalConfig(
                name=name,
                config_type=request.type,
                properties=request.properties,
                model_name=request.model_name,
                model_provider=request.provider,
                parent=eval,
            )
        except ValidationError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid properties for eval config type '{request.type.value}'.",
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e),
            )

        # Trust-conferral gate: saving a code eval admits new code as
        # trusted-forever, so it requires code trust for this session.
        # (Defense-in-depth: the frontend pre-checks trust before calling.)
        if isinstance(eval_config.properties, CodeEvalProperties):
            project = project_from_id(project_id)
            if not has_add_code_trust(str(project.path)):
                raise HTTPException(
                    status_code=403,
                    detail="Project not trusted to add code evals. Grant code trust and retry.",
                )

        eval_config.save_to_file()
        return eval_config

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/create_llm_judge_config",
        summary="Create LLM Judge Eval Config",
        tags=["Evals"],
        openapi_extra=ALLOW_AGENT,
    )
    async def create_llm_judge_config(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
        request: CreateLlmJudgeConfigRequest,
    ) -> EvalConfig:
        eval = eval_from_id(project_id, task_id, eval_id)
        name = request.name or generate_memorable_name()

        try:
            properties = materialize_llm_judge_properties(
                eval=eval,
                model_name=request.model_name,
                model_provider=request.provider,
                g_eval=request.g_eval,
                judge_prompt=request.judge_prompt,
                system_prompt=request.system_prompt,
                judge_instructions=request.judge_instructions,
            )
            eval_config = EvalConfig(
                name=name,
                config_type=EvalConfigType.v2,
                properties=properties,
                parent=eval,
            )
        except (ValidationError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail=str(e),
            )
        eval_config.save_to_file()
        return eval_config

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/default_llm_judge_prompt",
        summary="Get Default LLM Judge Prompt",
        tags=["Evals"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_default_llm_judge_prompt(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
    ) -> DefaultLlmJudgePromptResponse:
        eval = eval_from_id(project_id, task_id, eval_id)
        # `reference_keys` comes from the same predicate `materialize_llm_judge_properties`
        # uses, so what the builder is told to collect is what the saved judge requires —
        # including when the user edits the reference block out of the prompt, which the
        # server ignores.
        return DefaultLlmJudgePromptResponse(
            judge_prompt=build_default_llm_judge_prompt(eval),
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            reference_keys=derived_reference_keys(eval),
        )

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/test_v2_eval",
        summary="Test V2 Eval Config",
        tags=["Evals"],
        openapi_extra=DENY_AGENT,
    )
    async def test_v2_eval(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
        request: TestV2EvalRequest,
    ) -> TestV2EvalResponse:
        try:
            eval_obj = eval_from_id(project_id, task_id, eval_id)

            if request.llm_judge_builder_input is not None:
                builder = request.llm_judge_builder_input
                properties = materialize_llm_judge_properties(
                    eval=eval_obj,
                    model_name=builder.model_name,
                    model_provider=builder.provider,
                    g_eval=builder.g_eval,
                    judge_prompt=builder.judge_prompt,
                    system_prompt=builder.system_prompt,
                    judge_instructions=builder.judge_instructions,
                )
            elif request.properties is not None:
                properties = request.properties
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either properties or llm_judge_builder_input must be provided.",
                )

            return await run_v2_eval_test(
                project_id, eval_obj, properties, request.eval_input
            )
        except (ValueError, NotImplementedError, ValidationError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/test_v2_eval_draft",
        summary="Test V2 Eval Config Draft",
        tags=["Evals"],
        openapi_extra=DENY_AGENT,
    )
    async def test_v2_eval_draft(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        request: TestV2EvalDraftRequest,
    ) -> TestV2EvalResponse:
        """Test a judge config for an eval that hasn't been created yet.

        Builds a transient in-memory eval from the drafted output_scores so
        the creation flow can test its judge before saving anything.
        """
        try:
            task = task_from_id(project_id, task_id)
            eval_obj = Eval(
                name="Draft Eval Test",
                output_scores=request.output_scores,
                eval_set_filter_id="tag::draft_test",
                eval_configs_filter_id="tag::draft_test",
                parent=task,
            )
            return await run_v2_eval_test(
                project_id, eval_obj, request.properties, request.eval_input
            )
        except (ValueError, NotImplementedError, ValidationError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    # GET for an operation that writes, per .agents/api_code_review.md's SSE exception.
    # The web client is no longer an EventSource — run_eval.svelte reads this with fetch
    # so it can see a 4xx refusal's body — but GET stays: it is the shape every SSE
    # consumer expects, and switching to POST would break any client that is one.
    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/eval_config/{eval_config_id}/run_comparison",
        summary="Run Run Config Comparison",
        tags=["Evals"],
        openapi_extra=agent_policy_require_approval("Run eval comparison?"),
    )
    @no_write_lock
    async def run_eval_config(
        request: Request,
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
        eval_config_id: Annotated[
            str, Path(description="The unique identifier of the eval configuration.")
        ],
        run_config_ids: Annotated[
            list[str],
            Query(description="The list of run configuration IDs to evaluate."),
        ] = [],
        all_run_configs: Annotated[
            bool,
            Query(
                description="Whether to evaluate all run configurations for the task."
            ),
        ] = False,
    ) -> StreamingResponse:
        """Run a specific eval config against one or more run configs and stream progress via SSE. Executes model runs and scores them."""
        eval_config = eval_config_from_id(project_id, task_id, eval_id, eval_config_id)

        # Load the list of run configs to use. Two options:
        run_configs: list[TaskRunConfig] = []
        if all_run_configs:
            # special case, we cannot directly load task.run_configs(), we need to also get all finetune run configs which live inside the finetune model
            run_configs = get_all_run_configs(project_id, task_id)
        else:
            if len(run_config_ids) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No run config ids provided. At least one run config id is required.",
                )
            run_configs = [
                task_run_config_from_id(project_id, task_id, run_config_id)
                for run_config_id in run_config_ids
            ]

        eval = eval_from_id(project_id, task_id, eval_id)
        task = task_from_id(project_id, task_id)

        split = resolved_split_or_422(task, eval, "test")
        require_dataset_run_items_or_400(eval, eval_config.config_type, split)

        eval_runner = EvalRunner(
            eval_configs=[eval_config],
            run_configs=run_configs,
            eval_run_type="task_run_eval",
            split=split,
            save_context=build_save_context(request),
        )

        # Surface drive-config/run-config incompatibilities as one 400 before
        # the SSE stream opens — otherwise each job fails individually and
        # clients only see an anonymous error count. (EventSource consumers
        # can't read a 400 body, but an up-front error state still beats N
        # silent job errors; API clients get the full message.) Run-config
        # strictness only applies to a hand-picked list — with
        # all_run_configs, one incompatible config shouldn't block the rest.
        try:
            eval_runner.validate_multi_turn_drive_readiness(
                check_run_configs=not all_run_configs
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return await run_eval_runner_with_status(eval_runner)

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/set_current_eval_config/{eval_config_id}",
        summary="Set Default Eval Config",
        tags=["Evals"],
        openapi_extra=ALLOW_AGENT,
    )
    async def set_default_eval_config(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
        eval_config_id: Annotated[
            str,
            Path(
                description="The unique identifier of the eval configuration to set as default, or 'None' to clear the default."
            ),
        ],
    ) -> Eval:
        eval = eval_from_id(project_id, task_id, eval_id)

        if eval_config_id == "None":
            eval.current_config_id = None
        else:
            eval_config = next(
                (
                    eval_config
                    for eval_config in eval.configs()
                    if eval_config.id == eval_config_id
                ),
                None,
            )
            if eval_config is None:
                raise HTTPException(
                    status_code=400,
                    detail="Eval config not found.",
                )
            eval.current_config_id = eval_config_id
        eval.save_to_file()

        return eval

    # GET for an operation that writes, per .agents/api_code_review.md's SSE exception.
    # The web client is no longer an EventSource — run_eval.svelte reads this with fetch
    # so it can see a 4xx refusal's body — but GET stays: it is the shape every SSE
    # consumer expects, and switching to POST would break any client that is one.
    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/run_calibration",
        summary="Run Calibration",
        tags=["Evals"],
        openapi_extra=agent_policy_require_approval(
            "Run eval calibration? This runs LLM calls across all eval configs and uses AI credits."
        ),
    )
    @no_write_lock
    async def run_eval_config_eval(
        request: Request,
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
    ) -> StreamingResponse:
        """Run all eval configs against each other for calibration and stream progress via SSE. Used to check that eval configs produce consistent scores."""
        eval = eval_from_id(project_id, task_id, eval_id)
        golden_filter_id = require_golden_set_or_422(eval)

        # An empty golden set would "complete" instantly with zero scores — a
        # vacuous calibration the UI reads as success. Refuse it up front.
        task = task_from_id(project_id, task_id)
        if not runs_in_filter(task, golden_filter_id, readonly=True):
            raise HTTPException(
                status_code=400,
                detail="This eval's golden dataset is empty, so there is "
                "nothing to calibrate the judge against. Add human-rated "
                "examples to the golden set first.",
            )

        eval_configs = comparable_eval_configs_or_422(eval)
        eval_runner = EvalRunner(
            eval_configs=eval_configs,
            run_configs=None,
            eval_run_type="eval_config_eval",
            save_context=build_save_context(request),
        )

        return await run_eval_runner_with_status(eval_runner)

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/eval_config/{eval_config_id}/run_config/{run_config_id}/results",
        summary="Get Eval Run Results",
        tags=["Evals"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_eval_run_results(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
        eval_config_id: Annotated[
            str, Path(description="The unique identifier of the eval configuration.")
        ],
        run_config_id: Annotated[
            str, Path(description="The unique identifier of the run configuration.")
        ],
        split: Annotated[
            EvalSplitName,
            Query(
                description="Which of the eval's dataset splits to return results for. "
                "Required: every response about eval results is scoped to exactly one "
                "split, and reading has no obvious default the way running does."
            ),
        ],
    ) -> EvalRunResult:
        """Results for one run config, scoped to one of the eval's splits."""
        task = task_from_id(project_id, task_id)
        eval = eval_from_id(project_id, task_id, eval_id)
        eval_config = eval_config_from_id(project_id, task_id, eval_id, eval_config_id)
        run_config = task_run_config_from_id(project_id, task_id, run_config_id)

        # Filtered at query time against stored runs, so this works retroactively on
        # results recorded before splits existed. Membership is keyed on (source, id):
        # an EvalRun records exactly one of dataset_id or eval_input_id, and a split
        # yields ids from one store only.
        resolved_split = resolved_split_or_422(task, eval, split)
        results = [
            run_result
            for run_result in eval_config.runs(readonly=True)
            if run_result.task_run_config_id == run_config_id
            and eval_run_item_key(run_result) in resolved_split
        ]
        return EvalRunResult(
            results=resolve_eval_run_traces(task, results),
            eval=eval,
            eval_config=eval_config,
            run_config=run_config,
        )

    # Overview of the eval progress
    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/progress",
        summary="Get Eval Progress",
        tags=["Evals"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_eval_progress(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
    ) -> EvalProgress:
        task = task_from_id(project_id, task_id)
        eval = eval_from_id(project_id, task_id, eval_id)

        # Every split size is resolved in its own store, so an EvalInput-backed eval
        # reports its real counts rather than the 400 that used to stand here. That 400
        # was never a policy about golden sets — it fired because this code could only
        # count TaskRuns (functional spec 6.1).
        test_split = resolved_split_or_422(task, eval, "test")
        train_split = resolve_split(task, eval, "train")
        val_split = resolve_split(task, eval, "val")

        # Golden stays TaskRun-typed by definition, and is legitimately 0 for a V2 eval
        # that has no golden set.
        golden_dataset_runs = (
            runs_in_filter(task, eval.eval_configs_filter_id, readonly=True)
            if eval.eval_configs_filter_id
            else []
        )

        # Count how many dataset items have human evals
        fully_rated_count, partially_rated_count, not_rated_count = count_human_evals(
            golden_dataset_runs,
            eval,
            build_score_key_to_task_requirement_id(task),
        )

        current_eval_method = next(
            (
                eval_config
                for eval_config in eval.configs()
                if eval_config.id == eval.current_config_id
            ),
            None,
        )

        return EvalProgress(
            dataset_size=len(test_split),
            golden_dataset_size=len(golden_dataset_runs),
            golden_dataset_not_rated_count=not_rated_count,
            golden_dataset_partially_rated_count=partially_rated_count,
            golden_dataset_fully_rated_count=fully_rated_count,
            train_dataset_size=split_size(train_split),
            val_dataset_size=split_size(val_split),
            current_eval_method=current_eval_method,
        )

    # This compares run_configs to each other on a given eval_config. Compare to below which compares eval_configs to each other.
    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/eval_config/{eval_config_id}/score_summary",
        summary="Get Run Config Score Summary",
        tags=["Evals"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_eval_config_score_summary(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
        eval_config_id: Annotated[
            str, Path(description="The unique identifier of the eval configuration.")
        ],
    ) -> EvalResultSummary:
        task = task_from_id(project_id, task_id)
        eval = eval_from_id(project_id, task_id, eval_id)
        eval_config = eval_config_from_id(project_id, task_id, eval_id, eval_config_id)
        task_run_configs = get_all_run_configs(project_id, task_id)

        test_split = resolved_split_or_422(task, eval, "test")
        if len(test_split) == 0:
            raise HTTPException(
                status_code=400,
                detail="This eval's test split is empty. Add items matching the test split's filter — dataset runs, or eval inputs, depending on which backs the split.",
            )

        return compute_score_summary(eval, eval_config, task_run_configs, test_split)

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/eval_results_summary",
        summary="Get Eval Results Summary",
        tags=["Evals"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_eval_results_summary(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
    ) -> EvalResultsSummaryResponse:
        task = task_from_id(project_id, task_id)
        task_run_configs = get_all_run_configs(project_id, task_id)

        run_configs_out: Dict[ID_TYPE, EvalResultsSummaryRunConfigInfo] = {
            rc.id: EvalResultsSummaryRunConfigInfo(name=rc.name)
            for rc in task_run_configs
        }

        # Resolving a split walks a whole store off disk, and sibling evals routinely
        # share a test filter, so cache it. Keyed on (source, filter_id) rather than the
        # filter id alone: the `tag::` grammar is shared across both stores, so
        # "tag::golden" over task.runs() and over task.eval_inputs() are different item
        # sets behind the same string (functional spec 5.3).
        split_cache: Dict[Tuple[ItemSource, str], ResolvedSplit] = {}
        evals_out: Dict[ID_TYPE, EvalResultsSummaryEvalInfo] = {}
        scores_out: Dict[ID_TYPE, Dict[ID_TYPE, EvalResultsSummaryResultCell]] = {}

        for eval in task.evals(readonly=True):
            test_split = _cached_test_split(task, eval, split_cache)
            if test_split is None:
                # Unreachable for an eval that loaded: Eval validates that it has a test
                # split. Skipping rather than 4xx-ing keeps one corrupt file from
                # emptying the whole task's results table.
                continue

            evals_out[eval.id] = EvalResultsSummaryEvalInfo(
                name=eval.name,
                default_judge_config_id=eval.current_config_id,
                dataset_size=len(test_split),
                output_score_keys=[s.json_key() for s in eval.output_scores],
            )

            if eval.current_config_id is None:
                continue

            default_config = None
            for eval_config in eval.configs(readonly=True):
                if eval_config.id == eval.current_config_id:
                    default_config = eval_config
                    break

            if default_config is None or len(test_split) == 0:
                continue

            summary = compute_score_summary(
                eval,
                default_config,
                task_run_configs,
                test_split,
            )

            for rc_id, scores_dict in summary.results.items():
                mean_scores = {
                    key: s.mean_score
                    for key, s in scores_dict.items()
                    if s.mean_score is not None
                }
                percent_complete = summary.run_config_percent_complete.get(rc_id, 0.0)
                cell = EvalResultsSummaryResultCell(
                    mean_scores=mean_scores,
                    percent_complete=percent_complete,
                )
                if rc_id not in scores_out:
                    scores_out[rc_id] = {}
                scores_out[rc_id][eval.id] = cell

        return EvalResultsSummaryResponse(
            evals_by_id=evals_out,
            run_configs_by_id=run_configs_out,
            scores_by_run_config_by_eval=scores_out,
        )

    # Compared to above, this is comparing all eval configs to each other, not looking at a single eval config
    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/eval_configs_score_summary",
        summary="Get Eval Config Comparison Summary",
        tags=["Evals"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_eval_configs_score_summary(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        eval_id: Annotated[str, Path(description="The unique identifier of the eval.")],
    ) -> EvalConfigCompareSummary:
        task = task_from_id(project_id, task_id)
        eval = eval_from_id(project_id, task_id, eval_id)
        eval_configs = eval.configs(readonly=True)

        score_key_to_task_requirement_id = build_score_key_to_task_requirement_id(task)

        # Build a set of all the dataset items IDs we expect to have scores for
        # Fetch all the dataset items in a filter, and return a map of dataset_id -> TaskRun
        if eval.eval_configs_filter_id is None:
            raise HTTPException(
                status_code=400,
                detail="No eval configs filter id set, cannot get eval configs score summary.",
            )

        filter = dataset_filter_from_id(eval.eval_configs_filter_id)
        expected_dataset_items = {run.id: run for run in task.runs() if filter(run)}
        expected_dataset_ids = set(expected_dataset_items.keys())
        if len(expected_dataset_ids) == 0:
            return EvalConfigCompareSummary(
                results={},
                eval_config_percent_complete={},
                dataset_size=0,
                fully_rated_count=0,
                partially_rated_count=0,
                not_rated_count=0,
            )

        # save a copy of the expected dataset ids for each eval config id, we'll update each as we process each eval run
        remaining_expected_dataset_ids: Dict[ID_TYPE, Set[ID_TYPE]] = {
            eval_config.id: set(expected_dataset_ids) for eval_config in eval_configs
        }

        # eval_config_id -> output_score_json_key -> correlation calculator
        correlation_calculators: Dict[ID_TYPE, Dict[str, CorrelationCalculator]] = {}

        for eval_config in eval_configs:
            for eval_run in eval_config.runs(readonly=True):
                # Only calibration records enter the judge-vs-human stats: the
                # same eval config also accumulates task_run_eval records, and a
                # golden item's fresh-generation score correlated against the
                # stored item's human rating would be a category error.
                if not eval_run.eval_config_eval:
                    continue
                dataset_item = expected_dataset_items.get(eval_run.dataset_id, None)
                if dataset_item is None:
                    # A dataset_id can be removed from the dataset filter (ran previously, then removed the tag to remove it from the eval config set filter)
                    # A dataset_id could be for an run_config, not for comparing eval at all
                    continue

                # Check if we should count this eval_run. Not every eval_run has to go into the stats:
                # Example: this dataset_id was already counted (not great there are dupes, but shouldn't be double counted if there are)
                if (
                    eval_run.dataset_id
                    not in remaining_expected_dataset_ids[eval_config.id]
                ):
                    continue
                else:
                    remaining_expected_dataset_ids[eval_config.id].remove(
                        eval_run.dataset_id
                    )

                for output_score in eval.output_scores:
                    score_key = output_score.json_key()
                    eval_score: float | None = eval_run.scores.get(score_key, None)

                    # Fetch the human eval score from the dataset item
                    human_score = human_score_from_task_run(
                        dataset_item, output_score, score_key_to_task_requirement_id
                    )

                    if human_score is None or eval_score is None:
                        # This score doesn't have both a human eval and eval score, so we can't compare
                        continue

                    if eval_config.id not in correlation_calculators:
                        correlation_calculators[eval_config.id] = {}

                    calculator = correlation_calculators[eval_config.id].get(
                        score_key, None
                    )
                    if calculator is None:
                        calculator = CorrelationCalculator()
                        correlation_calculators[eval_config.id][score_key] = calculator

                    normalized_eval_score = normalize_rating(
                        eval_score, output_score.type
                    )
                    normalized_human_score = normalize_rating(
                        human_score, output_score.type
                    )
                    calculator.add_score(
                        CorrelationScore(
                            measured_score=eval_score,
                            human_score=human_score,
                            normalized_measured_score=normalized_eval_score,
                            normalized_human_score=normalized_human_score,
                        )
                    )

        # Convert to score summaries
        results: Dict[ID_TYPE, Dict[str, CorrelationResult]] = {}
        for eval_config_id in correlation_calculators.keys():
            results[eval_config_id] = {}
            for score_key in correlation_calculators[eval_config_id].keys():
                calculator = correlation_calculators[eval_config_id].get(
                    score_key, None
                )
                if calculator is None:
                    # No scores to calculate correlation for this pair
                    continue

                correlation_result = calculator.calculate_correlation()
                results[eval_config_id][score_key] = correlation_result

        # Calculate the percent of the dataset that has been processed
        eval_config_percent_complete: Dict[ID_TYPE, float] = {}
        for eval_config in eval_configs:
            incomplete_count = len(remaining_expected_dataset_ids[eval_config.id])
            percent_incomplete = incomplete_count / len(expected_dataset_ids)
            eval_config_percent_complete[eval_config.id] = 1 - percent_incomplete

        # Count how many dataset items have human evals
        fully_rated_count, partially_rated_count, not_rated_count = count_human_evals(
            list(expected_dataset_items.values()),
            eval,
            score_key_to_task_requirement_id,
        )

        return EvalConfigCompareSummary(
            results=results,
            eval_config_percent_complete=eval_config_percent_complete,
            dataset_size=len(expected_dataset_ids),
            fully_rated_count=fully_rated_count,
            partially_rated_count=partially_rated_count,
            not_rated_count=not_rated_count,
        )

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/run_configs/{run_config_id}/eval_scores",
        summary="Get Run Config Eval Scores",
        tags=["Run Configs"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_run_config_eval_scores(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        run_config_id: Annotated[
            str, Path(description="The unique identifier of the run configuration.")
        ],
    ) -> RunConfigEvalScoresSummary:
        task = task_from_id(project_id, task_id)

        # Verify the run config exists
        task_run_config_from_id(project_id, task_id, run_config_id)

        # Build a mapping from eval_id to spec for evals that are associated with
        # specs. Used to attach spec ids to results and to resolve each eval's
        # status (status lives on the eval, falling back to the spec for legacy files).
        specs = task.specs()
        eval_id_to_spec: Dict[str, Spec] = {}
        for spec in specs:
            if spec.eval_id and spec.id:
                eval_id_to_spec[spec.eval_id] = spec

        # Archived evals are not reported at all, so they are filtered out once here
        # rather than skipped inside each pass over them. Status lives on the eval,
        # falling back to its spec for evals created before that.
        evals = [
            eval
            for eval in task.evals()
            if eval.resolved_status(eval_id_to_spec.get(eval.id) if eval.id else None)
            != EvalStatus.archived
        ]
        eval_results: List[RunConfigEvalResult] = []

        # The usage reported below is the evaluated task's, which lives on the scored
        # TaskRun for every record written since the trace/score split.
        usage_by_scored_run_id = scored_trace_usage_for_run_config(
            task, evals, run_config_id
        )

        # Usage tracking across all eval configs for this run config
        total_input_tokens = 0.0
        total_output_tokens = 0.0
        total_total_tokens = 0.0
        total_cost = 0.0
        total_llm_latency_ms_sum = 0.0
        input_tokens_count = 0
        output_tokens_count = 0
        total_tokens_count = 0
        cost_count = 0
        latency_ms_count = 0
        total_eval_runs = 0

        for eval in evals:
            associated_spec = eval_id_to_spec.get(eval.id) if eval.id else None

            # Get the dataset size for this eval, from whichever store backs its test
            # split. None is unreachable for an eval that loaded (Eval validates a test
            # split); skipping keeps one corrupt file from emptying the whole table.
            test_split = resolve_split(task, eval, "test")
            if test_split is None:
                continue
            dataset_size = len(test_split)

            default_eval_config = summary_eval_config(eval)
            if not default_eval_config:
                # No default eval config set, so we can't process this eval. Still return it so UI can show an error
                eval_results.append(
                    RunConfigEvalResult(
                        eval_id=eval.id,
                        eval_name=eval.name,
                        dataset_size=dataset_size,
                        eval_config_result=None,
                        missing_default_eval_config=True,
                        spec_id=associated_spec.id if associated_spec else None,
                    )
                )
                continue

            eval_config = default_eval_config
            # Track which split items we've seen for this eval_config, keyed on
            # (source, id) so an EvalInput's score can't be credited to a TaskRun.
            remaining_expected_items = test_split.item_keys()
            partial_incomplete_count = 0
            eval_config_n_excluded = 0

            # output_score_json_key -> the individual scores, for the mean and
            # the percentile summary (see score_summary_from_values)
            score_values: Dict[str, list[float]] = {}

            for eval_run in eval_config.runs(readonly=True):
                # Only include eval_runs for our specific run_config
                if eval_run.task_run_config_id != run_config_id:
                    continue

                # Check if this item is expected for this eval
                item_key = eval_run_item_key(eval_run)
                if item_key not in remaining_expected_items:
                    continue
                else:
                    remaining_expected_items.remove(item_key)

                if eval_run.skipped_reason is not None:
                    eval_config_n_excluded += 1
                    continue

                total_eval_runs += 1

                # The evaluated task's usage: on the scored TaskRun for pointer records
                # (as scored_trace_usage reports it - conversation totals for multi-turn
                # chain leaves, synthetic-user spend blended in for driven traces),
                # inline on legacy ones.
                usage = eval_run_task_usage(eval_run, usage_by_scored_run_id)
                if usage:
                    if usage.input_tokens is not None:
                        total_input_tokens += usage.input_tokens
                        input_tokens_count += 1
                    if usage.output_tokens is not None:
                        total_output_tokens += usage.output_tokens
                        output_tokens_count += 1
                    if usage.total_tokens is not None:
                        total_total_tokens += usage.total_tokens
                        total_tokens_count += 1
                    if usage.cost is not None:
                        total_cost += usage.cost
                        cost_count += 1
                    if usage.total_llm_latency_ms is not None:
                        total_llm_latency_ms_sum += usage.total_llm_latency_ms
                        latency_ms_count += 1

                incomplete = False
                for output_score in eval.output_scores:
                    score_key = output_score.json_key()
                    if score_key not in score_values:
                        score_values[score_key] = []

                    if score_key in eval_run.scores:
                        score_values[score_key].append(eval_run.scores[score_key])
                    else:
                        # We're missing a required score, so this eval_run is incomplete
                        incomplete = True

                if incomplete:
                    partial_incomplete_count += 1

            results: Dict[str, ScoreSummary | None] = {}
            for output_score in eval.output_scores:
                score_key = output_score.json_key()
                values = score_values.get(score_key, [])
                if len(values) > 0 or eval_config_n_excluded > 0:
                    results[score_key] = score_summary_from_values(
                        values, eval_config_n_excluded
                    )
                else:
                    results[score_key] = None

            # Calculate the percent of the dataset that has been processed
            incomplete_count = partial_incomplete_count + len(remaining_expected_items)
            if dataset_size > 0:
                percent_incomplete = incomplete_count / dataset_size
                percent_complete = 1 - percent_incomplete
            else:
                percent_complete = 0.0

            eval_results.append(
                RunConfigEvalResult(
                    eval_id=eval.id,
                    eval_name=eval.name,
                    dataset_size=dataset_size,
                    missing_default_eval_config=False,
                    spec_id=associated_spec.id if associated_spec else None,
                    eval_config_result=EvalConfigResult(
                        eval_config_id=eval_config.id,
                        results=results,
                        percent_complete=percent_complete,
                        n_excluded=eval_config_n_excluded,
                    ),
                )
            )

        # Calculate mean usage across all eval runs for this run config (only include values where >= 50% of samples have data)
        mean_usage = None
        if total_eval_runs > 0:
            threshold = total_eval_runs * 0.5
            mean_usage = MeanUsage(
                mean_input_tokens=total_input_tokens / input_tokens_count
                if input_tokens_count >= threshold
                else None,
                mean_output_tokens=total_output_tokens / output_tokens_count
                if output_tokens_count >= threshold
                else None,
                mean_total_tokens=total_total_tokens / total_tokens_count
                if total_tokens_count >= threshold
                else None,
                mean_cost=total_cost / cost_count if cost_count >= threshold else None,
                mean_total_llm_latency_ms=total_llm_latency_ms_sum / latency_ms_count
                if latency_ms_count >= threshold
                else None,
            )

        return RunConfigEvalScoresSummary(
            eval_results=eval_results,
            mean_usage=mean_usage,
        )

    @app.post(
        "/api/projects/{project_id}/add_code_trust",
        summary="Add code trust for a project",
        tags=["Evals"],
        openapi_extra=DENY_AGENT,
    )
    async def add_code_trust_endpoint(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
    ) -> CodeTrustResponse:
        project = project_from_id(project_id)
        add_code_trust(str(project.path))
        return CodeTrustResponse(trusted=True)

    @app.get(
        "/api/projects/{project_id}/add_code_trust",
        summary="Check code trust for a project",
        tags=["Evals"],
        openapi_extra=DENY_AGENT,
    )
    async def check_add_code_trust_endpoint(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
    ) -> CodeTrustResponse:
        project = project_from_id(project_id)
        return CodeTrustResponse(trusted=has_add_code_trust(str(project.path)))
