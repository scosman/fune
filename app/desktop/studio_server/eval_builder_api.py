"""Eval Builder review-pipeline API (studio side).

Three streams, one frame contract (see api_models/eval_builder_models.py):

  multi_turn_pipeline (multi-turn) — runs [drive → judge] as one unit of work
  per case. The await order inside each case's coroutine IS the stage
  dependency; per-stage semaphores bound the fan-out (DRIVE_CONCURRENCY
  drive loops, REVIEW_CONCURRENCY judge units). A case failing at any
  stage emits case_failed and the other cases keep flowing — completed
  results are never discarded. The judge receives the runner's REAL trace
  (tool calls and system turns included), rendered once into the canonical
  transcript, which is echoed on each case_judged frame. Claims are NOT
  built here: the client builds them lazily via build_claims for the
  traces a reviewer actually opens — under subset review most never are.

  single_turn_pipeline (single-turn) — the one-turn sibling of
  multi_turn_pipeline: runs [run → judge] per generated input. The task runs
  ONCE per input on the target run config — tools live, the user's keys —
  and each persisted, batch-tagged run pipes into the same judge unit.
  The judge scores the run's transcript, exactly what the saved eval
  judges; the same trace is echoed on the frame for the UI.

  judge_traces (both arms) — re-judge previously driven results: the same
  judge unit and frames as the driving streams, with the drive replaced by
  a disk reload of each stored run by id. Multi-turn judges the chain
  leaf's stored trace; single-turn judges the run's own. Nothing is
  driven or written; the judge input matches what the saved eval judges.

Only the claim builder reaches the remote kiln_server; the judge runs
locally via the Eval V2 llm_judge adapter (the user's keys).
Orchestration and concurrency live here so the UI stays a thin SSE consumer.
"""

import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Path, Request
from kiln_ai.adapters.adapter_registry import adapter_for_task, load_skills_for_task
from kiln_ai.adapters.model_adapters.base_adapter import AdapterConfig
from kiln_ai.adapters.retry_classification import (
    is_batch_fatal_error,
    is_retryable_error,
    unwrap_kiln_run_error,
)
from kiln_ai.datamodel.datamodel_enums import (
    ModelProviderName,
    StructuredOutputMode,
    TurnMode,
)
from kiln_ai.datamodel.prompt_id import PromptGenerators
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties
from kiln_ai.datamodel.task import Task
from kiln_ai.datamodel.task_output import DataSource, DataSourceType
from kiln_ai.datamodel.task_run import TaskRun
from kiln_ai.synthetic_user.case import SyntheticUserCase as RunnerCase
from kiln_ai.synthetic_user.runner import (
    NUM_CASES_MAX,
    BatchStartedEvent,
    CaseCompletedEvent,
    CaseFailedEvent,
    TurnCompletedEvent,
    run_cases_batch,
)
from kiln_ai.utils.async_job_runner import (
    AsyncJobRunner,
    AsyncJobRunnerObserver,
    RetryableError,
    compute_retry_delay,
)
from kiln_ai.utils.git_sync_protocols import SaveContext, default_save_context
from kiln_ai.utils.slow_operation import log_if_slow
from kiln_server.cancellable_streaming_response import CancellableStreamingResponse
from kiln_server.git_sync_decorators import build_save_context, no_write_lock
from kiln_server.task_api import task_from_id
from kiln_server.utils.agent_checks.policy import agent_policy_require_approval
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from app.desktop.studio_server.api_models.eval_builder_models import (
    AuthorJudgeApiInput,
    AuthorJudgeApiOutput,
    BuildClaimsApiInput,
    BuildClaimsApiOutput,
    JudgeConfig,
    PipelineBatchAbortedEvent,
    PipelineBatchCompletedEvent,
    PipelineBatchStartedEvent,
    PipelineCaseDrivenEvent,
    PipelineCaseFailedEvent,
    PipelineCaseJudgedEvent,
    PipelineTurnCompletedEvent,
    PreflightModelApiInput,
    PreflightModelApiOutput,
    RefineJudgeApiInput,
    RefineJudgeApiOutput,
)
from app.desktop.studio_server.multiturn_sdg_api import (
    RunCasesBatchApiInput,
    TargetRunConfigFields,
    guard_multiturn,
    resolve_target_run_config,
    to_su_driver_config,
)
from app.desktop.studio_server.utils.copilot_utils import (
    delete_multi_turn_batch_chains,
    delete_single_turn_batch_runs,
    get_copilot_api_key,
    single_turn_drive_tags,
    tag_single_turn_drive_run,
    task_capabilities_for_task,
)
from app.desktop.studio_server.utils.eval_builder_utils import (
    author_judge_prompt,
    build_claims_for_trace,
    refine_judge_prompt_from_grades,
    run_judge_for_trace,
    trace_or_echo,
    transcript_io_for_trace,
)

logger = logging.getLogger(__name__)

# The builder's two concurrency knobs, one per pipeline stage:
#   DRIVE_CONCURRENCY  — concurrent drive units: SU drive loops on the
#                        multi-turn pipeline, one-shot task runs on the
#                        single-turn one (the most expensive stage either way).
#   REVIEW_CONCURRENCY — concurrent review units: judge calls on the merged
#                        pipelines and the judge_traces re-judge.
DRIVE_CONCURRENCY = 10
REVIEW_CONCURRENCY = 8

# The single-turn run stage's knobs — the same posture as the multi-turn
# drive runner (shared retry classifier, retry count, backoff base). Like the
# multi-turn drive, a run has no app-level timeout: termination is
# guaranteed by structural bounds (the adapter's tool-call cap and the
# model client's per-request timeout), and a pathologically slow run is
# logged rather than killed.
RUN_MAX_RETRIES = 2
RUN_RETRY_DELAY_SECONDS = 1.0

# Identifies the single-turn pipeline in input_source.properties.adapter_name
# so a reader looking at a TaskRun can tell who created it.
_SINGLE_TURN_ADAPTER_NAME = "kiln_eval_builder_single_turn"

# The judge lane's retry policy — the same posture (shared classifier, same
# attempt count and backoff) as the drive runner in
# kiln_ai.synthetic_user.runner: transient provider failures retry,
# deterministic ones fail the case immediately. The judge is the one local
# leg without a runner-owned retry; the remote copilot legs already retry
# inside kiln_server (pipeline jobs, retries=3), so no client retry stacks
# on top of them.
JUDGE_MAX_RETRIES = 2
# Base of the shared exponential-backoff-with-jitter window, not a flat wait.
JUDGE_RETRY_DELAY_SECONDS = 1.0


async def run_judge_with_retry(*args, **kwargs):
    """run_judge_for_trace under the shared transient-retry policy.

    A thin wrapper rather than a judge-side AsyncJobRunner: the pool engine
    takes a fixed job list, but judge work arrives streaming as each drive
    completes."""
    attempt = 0
    while True:
        try:
            return await run_judge_for_trace(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt > JUDGE_MAX_RETRIES or not is_retryable_error(e):
                raise
            # attempt counts failures so far; the shared backoff windows are
            # indexed from zero, so the first retry draws from (0, base).
            await asyncio.sleep(
                compute_retry_delay(JUDGE_RETRY_DELAY_SECONDS, attempt - 1)
            )


def _sse(payload: dict | BaseModel) -> str:
    """Format one SSE `data:` frame (the shared eval_builder frame contract)."""
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(by_alias=True)  # by_alias → citations use `from`
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


SSE_TERMINATOR = "data: complete\n\n"


class ReplaceBatchTagsField(BaseModel):
    """The delete-on-redrive half of a drive request, shared by both driving
    streams (multi-turn multi_turn_pipeline, single_turn_pipeline) so the batch
    lifecycle contract can't drift between them. Subclasses declare their
    own `batch_tag`; the self-replacement guard reads it by name."""

    replace_batch_tags: list[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Batch tags of previous drives this one supersedes (aborted "
            "re-drives can leave several behind). Their runs are deleted "
            "once this drive has produced replacements "
            "(delete-on-redrive), so abandoned batches don't accumulate on "
            "disk — and a wholesale drive failure never destroys the only "
            "batch the user has."
        ),
    )

    @field_validator("replace_batch_tags")
    @classmethod
    def replace_batch_tags_must_be_valid(cls, value: list[str]) -> list[str]:
        for tag in value:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tag):
                raise ValueError(f"invalid batch tag: {tag!r}")
        return value

    @model_validator(mode="after")
    def batch_tag_cannot_be_replaced(self) -> Self:
        # Deleting the batch this drive is about to create would destroy the
        # results the moment they were produced.
        batch_tag = getattr(self, "batch_tag", None)
        if batch_tag is not None and batch_tag in self.replace_batch_tags:
            raise ValueError(
                "replace_batch_tags must not contain this drive's own batch_tag."
            )
        return self


class MultiTurnPipelineRequest(RunCasesBatchApiInput, ReplaceBatchTagsField):
    """The merged multi-turn pipeline's request: everything a drive takes
    (inherited — the two drive contracts can't drift) plus the judge that
    scores the results and the batch lifecycle fields.

    `judge.prompt` is also what the client later passes to build_claims as
    the eval_rubric — the claim builder pressure-tests the rubric the
    verdict was really produced under.
    """

    # forbid: a retired or misspelled field on this request must 422, not be
    # silently dropped (a dropped replace_batch_tags quietly disables the
    # batch cleanup with no signal anywhere).
    model_config = ConfigDict(extra="forbid")

    judge: JudgeConfig


class JudgeTracesRequest(BaseModel):
    """The re-judge request, both arms: score previously driven results with
    a (typically refined) judge. No drive fields — the runs already exist on
    disk, identified by the ids the pipeline streams echoed on their
    case_driven/case_judged frames (the chain leaf on multi-turn, the run
    itself on single-turn).
    """

    leaf_run_ids: list[str] = Field(
        min_length=1,
        max_length=NUM_CASES_MAX,
        description=(
            "TaskRun ids of the driven results to judge: chain-leaf ids on "
            "a multi-turn task, the pipeline's run ids on a single-turn "
            "one. Frames reference each case by its position in this list "
            "(case_index)."
        ),
    )
    judge: JudgeConfig

    # forbid: a retired or misspelled field on this request must 422, not be
    # silently dropped.
    model_config = ConfigDict(extra="forbid")

    @field_validator("leaf_run_ids")
    @classmethod
    def leaf_run_ids_must_be_non_blank(cls, value: list[str]) -> list[str]:
        # A blank id can never match a stored run; reject the request up
        # front instead of streaming a guaranteed per-case failure.
        for run_id in value:
            if not run_id.strip():
                raise ValueError("leaf_run_ids must not contain empty ids.")
        return value


class SingleTurnPipelineRequest(TargetRunConfigFields, ReplaceBatchTagsField):
    """The single-turn pipeline's request: the generated inputs to run the
    task on, the target config that runs them (inherited — the two drive
    contracts can't drift), the judge that scores each result, and the
    batch lifecycle fields.

    `judge.prompt` is also what the client later passes to build_claims as
    the eval_rubric — the claim builder pressure-tests the rubric the
    verdict was really produced under.
    """

    # forbid: a retired or misspelled field on this request must 422, not be
    # silently dropped (a dropped replace_batch_tags quietly disables the
    # batch cleanup with no signal anywhere).
    model_config = ConfigDict(extra="forbid")

    inputs: list[str] = Field(
        min_length=1,
        max_length=NUM_CASES_MAX,
        description=(
            "The generated task inputs, one run each — typically one per "
            "approved batch-plan prompt. For tasks with an input schema, "
            "each entry is the input as a JSON string (the same encoding "
            "the saved eval's inputs-only items store). Capped at the "
            "multi-turn batch size: the two arms share one batch budget."
        ),
    )
    input_model_name: str = Field(
        min_length=1,
        description=(
            "The model that generated the inputs (recorded on each run's "
            "input source, like the /generate output writer records it)."
        ),
    )
    input_provider: ModelProviderName = Field(
        description="The provider the inputs were generated with."
    )
    batch_tag: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]+$",
        min_length=1,
        max_length=64,
        description=(
            "Optional user-supplied batch label. Constrained to "
            "[A-Za-z0-9_-]{1,64} so it can safely be used as a tag on the "
            "driven TaskRuns. Auto-generated if not provided."
        ),
    )
    judge: JudgeConfig

    @field_validator("inputs")
    @classmethod
    def inputs_must_be_non_blank(cls, value: list[str]) -> list[str]:
        # A blank input would run the task on nothing; reject the request up
        # front instead of streaming a guaranteed per-case failure.
        for input_text in value:
            if not input_text.strip():
                raise ValueError("inputs must not contain empty entries.")
        return value


class JudgeStreamBase:
    """The judge unit + frame plumbing shared by the three pipeline streams
    (multi_turn_pipeline, judge_traces, single_turn_pipeline).

    Subclasses own the producer that feeds cases in (live drive, disk
    reload, or one-shot run — `_produce`) and how the judge reads a case
    (`_judge_view`); everything else — the `events()` drain loop, the judge
    semaphore, retries, per-case failure isolation, batch-fatal abort, and
    consumer-disconnect teardown — lives here so the streams cannot drift.
    """

    # Set by subclasses: the route name used in log lines.
    _stream_name: str
    # Set by subclasses: the batch_failed message for an unexpectedly
    # cancelled producer (a stray CancelledError killed it mid-stream).
    _producer_cancelled_message: str

    def __init__(
        self,
        *,
        project_id: str,
        task_id: str,
        judge: JudgeConfig,
    ) -> None:
        self._project_id = project_id
        self._task_id = task_id
        self._judge = judge
        # `None` is the end-of-stream sentinel.
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._review_sem = asyncio.Semaphore(REVIEW_CONCURRENCY)
        self._review_tasks: list[asyncio.Task] = []
        self._judged_count = 0
        self._failed_count = 0
        # batch_completed's tag and spend. Streams that drive (pipeline,
        # single-turn) set both; the disk-reload stream keeps the honest
        # neutral defaults — no new batch, no new drive spend.
        self._batch_tag = ""
        self._total_cost = 0.0
        # Set by the first batch-fatal failure; events() then skips
        # batch_completed and its finally tears everything down.
        self._aborted = False

    async def _produce(self) -> None:
        """The stage that feeds cases into the judge unit — a live drive, a
        disk reload, or a one-shot run per input. Emits its own frames and
        appends judge tasks to `_review_tasks`."""
        raise NotImplementedError

    async def events(self) -> AsyncIterator[str]:
        """The SSE stream body: drain frames until the producer and every
        judge task finished, then emit batch_completed (or batch_failed) and
        the terminator."""
        producer = asyncio.create_task(
            self._produce(), name=f"{self._stream_name}_producer"
        )

        async def close_when_done() -> None:
            # _review_tasks only grows while the producer runs, so once it is
            # done the list is final and the gather is complete.
            try:
                await producer
            finally:
                if self._review_tasks:
                    await asyncio.gather(*self._review_tasks, return_exceptions=True)
                await self._queue.put(None)

        closer = asyncio.create_task(
            close_when_done(), name=f"{self._stream_name}_closer"
        )

        try:
            while True:
                frame = await self._queue.get()
                if frame is None:
                    break
                yield frame
            # A batch-fatal abort already emitted batch_aborted in place of
            # batch_completed — fall through to the finally, which runs the
            # same teardown as a consumer disconnect (producer and in-flight
            # judges cancelled) and a doomed batch stops spending.
            if not self._aborted:
                # The closer swallows a producer-level crash (its finally
                # still closes the queue) — re-raise it here so the client
                # sees batch_failed, not a clean-looking batch_completed. Our
                # own teardown never reaches this line, so a cancelled
                # producer here means a stray CancelledError killed it: also
                # a failure.
                if producer.cancelled():
                    raise RuntimeError(self._producer_cancelled_message)
                producer_error = producer.exception() if producer.done() else None
                if producer_error is not None:
                    raise producer_error
                yield _sse(
                    PipelineBatchCompletedEvent(
                        judged=self._judged_count,
                        failed=self._failed_count,
                        batch_tag=self._batch_tag,
                        total_cost=self._total_cost,
                    )
                )
        except Exception as e:
            # Per-case failures never reach here (they become case_failed
            # frames); this catches orchestration bugs only.
            logger.exception("%s failed mid-stream", self._stream_name)
            yield _sse(
                {
                    "type": "batch_failed",
                    "code": "internal_error",
                    "message": f"{type(e).__name__}: {e}",
                }
            )
        finally:
            # Consumer disconnect (or any exit): stop the producer and any
            # in-flight judges so abandoned LLM calls stop spending. A
            # cancelled drive cancels its own workers, and each cancelled
            # case cleans up its own partial writes as it unwinds.
            producer.cancel()
            for t in self._review_tasks:
                t.cancel()
            closer.cancel()
            await asyncio.gather(
                producer, closer, *self._review_tasks, return_exceptions=True
            )
        yield SSE_TERMINATOR

    def _judge_view(
        self, case_index: int, trace: list[dict[str, Any]] | None
    ) -> tuple[str, str, list[dict[str, Any]] | None]:
        """What the judge scores for one case: (raw_input, raw_output, and
        the structured trace to judge over; the Optional is vestigial, as
        every arm now judges a transcript).

        The default is the multi-turn reading — transcript I/O plus the full
        trace — matching what the saved eval judges. The single-turn streams
        override only where raw_input comes from; both arms judge a
        transcript, so every override still returns one.
        """
        assert trace is not None  # multi-turn streams always carry a trace
        raw_input, raw_output = transcript_io_for_trace(trace)
        return raw_input, raw_output, trace

    async def _delete_superseded_batches(
        self,
        tags: list[str],
        delete_batch: Callable[[str], int],
        save_context: SaveContext,
    ) -> None:
        """Delete-on-redrive, AFTER the producer made replacement runs —
        deleting up front could leave the user with neither batch when a
        re-drive fails wholesale. Best-effort cleanup: a failure here must
        never cost the batch's results. `delete_batch` is the arm's
        task-bound tag -> deleted-count deleter.
        """
        for tag in tags:
            try:
                # The delete is sync file I/O over the task's run corpus —
                # run it off the event loop so other requests and streams
                # keep moving.
                async with save_context():
                    deleted = await asyncio.to_thread(delete_batch, tag)
                logger.info(
                    "%s: deleted %d runs of superseded batch %s",
                    self._stream_name,
                    deleted,
                    tag,
                )
            except Exception:
                logger.exception(
                    "%s: failed to delete superseded batch %s",
                    self._stream_name,
                    tag,
                )

    async def _judge_case(
        self,
        case_index: int,
        leaf_run_id: str,
        trace: list[dict[str, Any]] | None,
        drive_cost: float,
    ) -> None:
        """Judge one case (local). `trace` is the structured conversation
        echoed on the frame; how the judge reads the case is `_judge_view`.
        Claims are not built here — the client requests them per opened
        trace via build_claims."""
        async with self._review_sem:
            try:
                raw_input, raw_output, judge_trace = self._judge_view(case_index, trace)
                verdict = await run_judge_with_retry(
                    self._project_id,
                    self._task_id,
                    raw_input,
                    raw_output,
                    self._judge,
                    trace=judge_trace,
                )
                # Frame construction/serialization is inside the try: a
                # failure here must surface as case_failed too, or the batch
                # totals would claim a verdict the client never received.
                frame = _sse(
                    PipelineCaseJudgedEvent(
                        case_index=case_index,
                        leaf_run_id=leaf_run_id,
                        raw_input=raw_input,
                        raw_output=raw_output,
                        judge_score=verdict.judge_score,
                        judge_reasoning=verdict.judge_reasoning,
                        total_cost=drive_cost,
                        # Echo the structured trace alongside its flattened
                        # raw_output so the client can render the real chat UI
                        # and map citation spans back onto it.
                        trace=trace,
                    )
                )
            except Exception as e:
                # A config-scoped failure (bad key, deprecated model) will
                # kill every judgment identically — abort the whole batch on
                # the first one instead of failing every case one by one
                # (on the pipeline stream the drives would keep BILLING).
                if is_batch_fatal_error(e):
                    await self._abort_batch("judge", e)
                    return
                # The adapter's KilnRunError wrapper carries a genericized
                # message — surface the root provider error, like the abort path.
                root = unwrap_kiln_run_error(e)
                await self._fail_case(
                    case_index,
                    "judge",
                    "judge_failed",
                    f"{type(root).__name__}: {root}",
                    type(root).__name__,
                )
                return
            self._judged_count += 1
            await self._queue.put(frame)

    async def _abort_batch(
        self, stage: Literal["drive", "run", "judge"], error: BaseException
    ) -> None:
        """First batch-fatal failure wins: emit ONE batch_aborted frame and
        close the queue — events()' finally then runs the consumer-disconnect
        teardown (producer and in-flight judges cancelled). On the pipeline
        stream each cancelled case deletes its own partial chain as it
        unwinds (runner-side cleanup), so an abort leaves no orphan runs
        behind; the disk-reload stream has nothing to clean."""
        if self._aborted:
            return
        self._aborted = True
        root = unwrap_kiln_run_error(error)
        logger.error(
            "%s: batch-fatal %s failure, aborting the batch: %s",
            self._stream_name,
            stage,
            root,
        )
        await self._emit(
            PipelineBatchAbortedEvent(
                error=f"{type(root).__name__}: {root}", stage=stage
            )
        )
        await self._queue.put(None)

    async def _fail_case(
        self,
        case_index: int,
        stage: Literal["drive", "run", "judge"],
        code: str,
        message: str,
        error_type: str | None = None,
    ) -> None:
        """One case died at `stage`; the batch continues without it.

        `error_type` is the class name of the provider or unexpected exception
        behind the failure (None on deterministic failures, whose `code` already
        names them) — included in the log line as a grep-able `error_type=...`
        field so local logs can be grouped by failure kind, not just the frame.
        """
        logger.exception(
            "%s: %s failed for case %d, error_type=%s",
            self._stream_name,
            stage,
            case_index,
            error_type,
        )
        self._failed_count += 1
        await self._emit(
            PipelineCaseFailedEvent(
                case_index=case_index,
                stage=stage,
                code=code,
                message=message,
                error_type=error_type,
            )
        )

    async def _emit(self, payload: dict | BaseModel) -> None:
        await self._queue.put(_sse(payload))


class MultiTurnPipelineRun(JudgeStreamBase):
    """One merged-pipeline execution: [drive → judge] per case.

    Frames from the concurrently-running stages funnel through one queue and
    come out of the inherited `events()` drain loop; `_produce()` drives, and
    the inherited judge unit scores and isolates failures.
    """

    _stream_name = "multi_turn_pipeline"
    _producer_cancelled_message = "The drive was cancelled unexpectedly."

    def __init__(
        self,
        *,
        project_id: str,
        task_id: str,
        task: Task,
        cases: list[RunnerCase],
        input: MultiTurnPipelineRequest,
        save_context: SaveContext | None,
    ) -> None:
        super().__init__(
            project_id=project_id,
            task_id=task_id,
            judge=input.judge,
        )
        self._task = task
        self._cases = cases
        self._input = input
        # Resolved at construction — i.e. inside the endpoint, before the
        # stream opens — so an unknown/non-agent run config id is a clean
        # 4xx rather than a mid-stream error frame.
        self._target_run_config, self._target_run_config_id = resolve_target_run_config(
            input, project_id, task_id
        )
        # build_save_context returns None outside a git-synced request; fall
        # back the same way the runner does.
        self._save_context = save_context or default_save_context
        # Latest cumulative trace per case, captured from the runner's
        # in-process turn events — the REAL trace (tool calls, system turns),
        # not a wire projection. Popped when the case's review starts.
        # Values are the runner's typed message params, read as loose dicts by
        # the judge/claims layer (list[Any] because typing.cast is banned).
        self._latest_trace: dict[int, list[Any]] = {}
        # The base's _total_cost accumulates the batch's actual drive
        # billing — failed cases and discarded retry attempts included, not
        # just surviving conversations.
        self._batch_tag = input.batch_tag or ""

    async def _produce(self) -> None:
        """Consume the SU runner's events; each completed case pipelines
        straight into its own judge task — no stage barrier."""
        any_case_driven = False
        async for event in run_cases_batch(
            cases=self._cases,
            target_task=self._task,
            target_run_config=self._target_run_config,
            su_driver_config=to_su_driver_config(self._input.su_driver),
            turns=self._input.turns,
            concurrency=DRIVE_CONCURRENCY,
            batch_tag=self._input.batch_tag,
            save_context=self._save_context,
            task_run_config_id=self._target_run_config_id,
        ):
            if isinstance(event, BatchStartedEvent):
                self._batch_tag = event.batch_tag
                await self._emit(
                    PipelineBatchStartedEvent(
                        batch_tag=event.batch_tag,
                        total_cases=event.num_cases,
                    )
                )
            elif isinstance(event, TurnCompletedEvent):
                # The runner emits a fresh snapshot list per event; its typed
                # message params are plain dicts at runtime, which the
                # judge/claims layer treats loosely.
                self._latest_trace[event.case_index] = event.trace
                await self._emit(
                    PipelineTurnCompletedEvent(
                        case_index=event.case_index,
                        # The runner's per-attempt turn number, not an event
                        # count — a retried case restarts at 1, and counting
                        # events would overshoot the denominator.
                        turns_completed=event.turn_index,
                        total_turns=self._input.turns,
                    )
                )
            elif isinstance(event, CaseCompletedEvent):
                # Drive spend is real once billing happened: the surviving
                # conversation plus any retried attempts whose chains were
                # discarded. Per-case events carry conversation cost only.
                self._total_cost += event.total_cost + event.discarded_attempts_cost
                trace = self._latest_trace.pop(event.case_index, [])
                if not trace:
                    # turns >= 1 guarantees a turn event before the case
                    # completes; an empty trace means that invariant broke —
                    # fail the case, keep the batch.
                    await self._fail_case(
                        event.case_index,
                        "drive",
                        "missing_trace",
                        "The drive produced no trace for this case.",
                    )
                    continue
                any_case_driven = True
                await self._emit(
                    PipelineCaseDrivenEvent(
                        case_index=event.case_index,
                        leaf_run_id=event.leaf_run_id,
                    )
                )
                self._review_tasks.append(
                    asyncio.create_task(
                        self._judge_case(
                            event.case_index,
                            event.leaf_run_id,
                            trace,
                            event.total_cost,
                        ),
                        name=f"judge_case_{event.case_index}",
                    )
                )
            elif isinstance(event, CaseFailedEvent):
                # A dead case still billed for every attempt it made.
                self._total_cost += event.total_cost
                await self._fail_case(
                    event.case_index,
                    "drive",
                    event.error_code,
                    event.message,
                    event.error_type,
                )
            # The runner's BatchCompletedEvent is not forwarded: the
            # pipeline's own batch_completed fires after reviews drain.
        if any_case_driven:
            # Replacement chains exist on disk — now the superseded batches
            # can go. A drive that produced nothing keeps them untouched.
            await self._delete_superseded_batches(
                self._input.replace_batch_tags,
                lambda tag: delete_multi_turn_batch_chains(self._task, tag),
                self._save_context,
            )


class JudgeTracesRun(JudgeStreamBase):
    """One judge_traces execution: [reload → judge] per case.

    The calibration loop's re-judge stream, both arms: the same judge unit
    and frame contract as the driving streams, with the drive replaced by a
    disk reload of each stored run by id. Multi-turn judges the chain leaf's
    stored cumulative trace — exactly the conversation the drive-time judge
    saw and the saved eval will judge. Single-turn judges the run's own
    stored trace, the same reading as its pipeline and its saved eval.
    Nothing is driven and nothing is written.
    """

    _stream_name = "judge_traces"
    _producer_cancelled_message = "The trace reload was cancelled unexpectedly."

    def __init__(
        self,
        *,
        project_id: str,
        task_id: str,
        task: Task,
        input: JudgeTracesRequest,
    ) -> None:
        super().__init__(
            project_id=project_id,
            task_id=task_id,
            judge=input.judge,
        )
        # The base's neutral batch_tag ""/total_cost 0.0 are kept: no drive
        # ran, so there is no new batch tag and no new drive spend to report.
        self._task = task
        self._leaf_run_ids = input.leaf_run_ids
        self._single_turn = task.turn_mode != TurnMode.multiturn
        # Single-turn arm: each case's stored (input, output) pair. Only the
        # input is read back — the judge and the frames take raw_input from
        # here rather than from the transcript (the SingleTurnPipelineRun
        # convention); raw_output comes from the transcript rendering.
        self._case_io: dict[int, tuple[str, str]] = {}

    def _judge_view(
        self, case_index: int, trace: list[dict[str, Any]] | None
    ) -> tuple[str, str, list[dict[str, Any]] | None]:
        # Single-turn keeps the stored run's input verbatim (see the pipeline's
        # override); multi-turn takes the base's transcript reading whole.
        if self._single_turn:
            assert trace is not None  # the producer passes a trace or an echo
            _, raw_output = transcript_io_for_trace(trace)
            raw_input, _ = self._case_io[case_index]
            return raw_input, raw_output, trace
        return super()._judge_view(case_index, trace)

    async def _produce(self) -> None:
        """The producer: reload each stored run from disk and feed it
        straight into its own judge task — the drive stage of the merged
        pipeline, swapped for a disk read."""
        await self._emit(
            PipelineBatchStartedEvent(
                # No drive, no new batch: the runs keep their original tags.
                batch_tag="",
                total_cases=len(self._leaf_run_ids),
            )
        )
        # Bulk load: one pass over the run corpus instead of a per-id scan.
        # Sync file I/O, so off the event loop. The lambda keeps the
        # classmethod's TypeVar bound to TaskRun through to_thread.
        leaves: dict[str, TaskRun] = await asyncio.to_thread(
            lambda: TaskRun.from_ids_and_parent_path(
                set(self._leaf_run_ids), self._task.path
            )
        )
        for case_index, leaf_run_id in enumerate(self._leaf_run_ids):
            leaf = leaves.get(leaf_run_id)
            if leaf is None:
                # Runs can vanish between rounds (delete-on-redrive, manual
                # dataset edits) — fail the case, keep the batch.
                await self._fail_case(
                    case_index,
                    "judge",
                    "trace_not_found",
                    f"No saved result found for run id {leaf_run_id}.",
                )
                continue
            if self._single_turn:
                await self._produce_single_turn_case(case_index, leaf_run_id, leaf)
                continue
            if not leaf.trace:
                await self._fail_case(
                    case_index,
                    "judge",
                    "missing_trace",
                    "The saved conversation has no stored trace to judge.",
                )
                continue
            # The same projection the saved eval applies to a stored chain
            # leaf (EvalTaskInput.from_task_run), so the judge input is
            # identical to drive time and to eval time. drive_cost is 0.0:
            # the original drive already reported this chain's spend.
            trace = [dict(message) for message in leaf.trace]
            self._review_tasks.append(
                asyncio.create_task(
                    self._judge_case(case_index, leaf_run_id, trace, 0.0),
                    name=f"judge_case_{case_index}",
                )
            )

    async def _produce_single_turn_case(
        self, case_index: int, leaf_run_id: str, leaf: TaskRun
    ) -> None:
        """Feed one reloaded single-turn run into the judge unit: the run's
        stored transcript is what the judge scores, echoed to a two-message
        pair when the run recorded none. drive_cost is 0.0: the original
        pipeline already reported this run's spend."""
        output = leaf.output.output if leaf.output is not None else None
        if not output:
            await self._fail_case(
                case_index,
                "judge",
                "missing_output",
                "The saved run has no stored output to judge.",
            )
            return
        self._case_io[case_index] = (leaf.input, output)
        trace = trace_or_echo(
            [dict(message) for message in leaf.trace] if leaf.trace else None,
            leaf.input,
            output,
        )
        self._review_tasks.append(
            asyncio.create_task(
                self._judge_case(case_index, leaf_run_id, trace, 0.0),
                name=f"judge_case_{case_index}",
            )
        )


class _RunFailure(Exception):
    """A terminal per-case failure of the single-turn run stage — never
    retried. Deterministic input problems and provider errors the shared
    classifier calls permanent. `code` and `error_type` are surfaced on
    case_failed.
    """

    def __init__(self, code: str, message: str, error_type: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        # Class name of the provider or unexpected exception behind this
        # failure; None on deterministic failures, whose `code` already names
        # them.
        self.error_type = error_type


def _run_failure_details(error: Exception) -> tuple[str, str, str | None]:
    """Map a case's terminal exception to case_failed's code, message and
    error type."""
    if isinstance(error, _RunFailure):
        return error.code, str(error), error.error_type
    # A RetryableError whose attempts ran out (or an unexpected
    # orchestration error surfaced by the job runner). The retryable wrapper
    # is always raised `from` the provider error it classified, so its cause
    # names the real failure; an error that carries no cause falls back to None.
    cause = error.__cause__
    error_type = (
        type(unwrap_kiln_run_error(cause)).__name__ if cause is not None else None
    )
    return "unexpected_error", str(error), error_type


def _run_cost(run: TaskRun) -> float:
    """Read the rolled-up cost from a TaskRun, defaulting to 0 if usage is
    missing (defensive against fakes in unit tests that don't populate it).
    """
    usage = getattr(run, "cumulative_usage", None)
    if usage is None:
        return 0.0
    return float(getattr(usage, "cost", None) or 0.0)


def guard_single_turn(task: Task) -> None:
    """Reject early if the caller pointed the single-turn pipeline at a
    multi-turn task.

    This pipeline runs the task once per generated input. A multi-turn task
    needs a synthetic user to carry the conversation forward, which is a
    different drive entirely (multi_turn_pipeline).
    """
    if task.turn_mode == TurnMode.multiturn:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "task_not_single_turn",
                "message": (
                    "The single-turn pipeline requires a task with "
                    "turn_mode=single_turn."
                ),
            },
        )


class SingleTurnPipelineRun(JudgeStreamBase):
    """One single-turn pipeline execution: [run → judge] per input.

    The run stage is the one-turn sibling of the multi-turn drive: the task
    runs once per generated input on the target run config — tools live, the
    user's keys — through the same AsyncJobRunner fan-out and retry posture
    as the SU runner. Each run persists (adapter autosave) and is
    batch-tagged so save and delete-on-redrive can find it; a failed
    attempt deletes its own persisted run before retrying or dying, banking
    the spend first. Completed runs pipe straight into the inherited judge
    unit — no stage barrier.
    """

    _stream_name = "single_turn_pipeline"
    _producer_cancelled_message = "The run stage was cancelled unexpectedly."

    def __init__(
        self,
        *,
        project_id: str,
        task_id: str,
        task: Task,
        input: SingleTurnPipelineRequest,
        save_context: SaveContext | None,
    ) -> None:
        super().__init__(
            project_id=project_id,
            task_id=task_id,
            judge=input.judge,
        )
        self._task = task
        self._input = input
        # Resolved at construction — i.e. inside the endpoint, before the
        # stream opens — so an unknown/non-agent run config id is a clean
        # 4xx rather than a mid-stream error frame.
        self._target_run_config, self._target_run_config_id = resolve_target_run_config(
            input, project_id, task_id
        )
        # build_save_context returns None outside a git-synced request; fall
        # back the same way the runner does.
        self._save_context = save_context or default_save_context
        self._batch_tag = input.batch_tag or uuid.uuid4().hex[:12]
        # Each case's (input, output) pair, recorded by the producer for the
        # judge unit. Only the input is read back: raw_output and the judge's
        # trace both come from the transcript (see _judge_view).
        self._case_io: dict[int, tuple[str, str]] = {}
        self._any_case_driven = False

    def _judge_view(
        self, case_index: int, trace: list[dict[str, Any]] | None
    ) -> tuple[str, str, list[dict[str, Any]] | None]:
        # Single-turn keeps the REQUEST's input string verbatim: the saved
        # eval stores that same string on its inputs-only item and reads it
        # back from there (EvalTaskInput.from_trace takes task_input from the
        # item, not the trace, because the adapter may have reserialized it).
        # Taking the trace's opening user message instead would judge a
        # different string than the eval that ships.
        #
        # The OUTPUT and the judge trace come from the transcript, so the
        # judge sees what the agent actually did — tool calls and tool
        # results included — rather than only its closing message.
        assert trace is not None  # producers pass a trace or an echo of one
        _, raw_output = transcript_io_for_trace(trace)
        raw_input, _ = self._case_io[case_index]
        return raw_input, raw_output, trace

    async def _produce(self) -> None:
        """The producer: run the task once per input; each persisted run
        pipes straight into its own judge task — no stage barrier."""
        # Skills referenced by the run config load once for the whole batch
        # (the adapter raises on a skill tool id with no injected dict).
        # Before the first frame, matching the multi-turn runner's order —
        # a bad skill reference fails the stream without a batch_started.
        skills = load_skills_for_task(self._task, self._target_run_config)
        await self._emit(
            PipelineBatchStartedEvent(
                batch_tag=self._batch_tag,
                total_cases=len(self._input.inputs),
            )
        )

        fail_case = self._fail_case

        class _EmitRunFailed(AsyncJobRunnerObserver[tuple[int, str]]):
            """Emits case_failed exactly once per dead case — the job runner
            calls on_error only after retries are exhausted (or immediately
            for non-retryable failures)."""

            async def on_error(self, job: tuple[int, str], error: Exception) -> None:
                case_index, _input_text = job
                code, message, error_type = _run_failure_details(error)
                await fail_case(case_index, "run", code, message, error_type)

        async def _run_job(job: tuple[int, str]) -> bool:
            await self._run_one_input(job, skills)
            return True

        # AsyncJobRunner is the shared fan-out engine (same as the SU and
        # eval runners): a worker pool bounded by DRIVE_CONCURRENCY,
        # retrying cases whose run raised RetryableError before declaring
        # them dead.
        runner = AsyncJobRunner(
            jobs=list(enumerate(self._input.inputs)),
            run_job_fn=_run_job,
            concurrency=DRIVE_CONCURRENCY,
            max_retries=RUN_MAX_RETRIES,
            retry_delay=RUN_RETRY_DELAY_SECONDS,
            observers=[_EmitRunFailed()],
        )
        # The runner's coarse Progress stream goes unused — this stream's
        # protocol is the pipeline frames the jobs emit; draining it is what
        # drives the workers.
        async for _progress in runner.run():
            pass
        if self._any_case_driven:
            # Replacement runs exist on disk — now the superseded batches
            # can go. A run stage that produced nothing keeps them untouched.
            await self._delete_superseded_batches(
                self._input.replace_batch_tags,
                lambda tag: delete_single_turn_batch_runs(self._task, tag),
                self._save_context,
            )

    async def _run_one_input(self, job: tuple[int, str], skills: Any) -> None:
        """Run the task once on one input (one ATTEMPT), then hand the run
        to the judge unit.

        Failures RAISE instead of emitting: transient provider errors become
        RetryableError (the job runner re-runs the case), everything else
        becomes _RunFailure, and case_failed is emitted once — by the
        runner's on_error observer, after the last attempt. A failed or
        cancelled attempt deletes the run it persisted (banking its real
        spend first), so a retry starts clean and no untagged orphan
        outlives its case.
        """
        case_index, input_text = job
        run: TaskRun | None = None
        try:
            parsed_input: str | dict = input_text
            if self._task.input_json_schema is not None:
                # Structured tasks carry the input as a JSON string — the
                # same encoding base_eval.run_task parses at eval time.
                try:
                    parsed_input = json.loads(input_text)
                except json.JSONDecodeError as e:
                    # Deterministic: retrying replays the same parse on the
                    # same bytes.
                    raise _RunFailure(
                        "invalid_input",
                        "The generated input is not valid JSON for this "
                        f"task's input schema: {e}",
                    ) from e
            # A fresh adapter per attempt, like the drive runner's per-case
            # invoker; task_run_config_id stamps the run's output source
            # with the saved config it came from, exactly as a manual run
            # of that config would. default_tags lands the discovery tags in
            # the SAME save that persists the run, so a run orphaned by a
            # cancel mid-invoke stays discoverable (the next replace pass
            # sweeps it) instead of sitting untagged on disk forever.
            adapter = adapter_for_task(
                self._task,
                self._target_run_config,
                base_adapter_config=AdapterConfig(
                    skills=skills,
                    task_run_config_id=self._target_run_config_id,
                    default_tags=single_turn_drive_tags(self._batch_tag),
                ),
            )
            # No app-level timeout: a hung provider call is bounded by the
            # model client's per-request timeout and the tool loop by the
            # adapter's cap, so the invocation terminates structurally. The
            # watchdog makes a pathologically slow run visible in logs
            # without killing a healthy one.
            async with log_if_slow(f"single_turn_pipeline: case {case_index}"):
                run = await adapter.invoke(
                    input=parsed_input, input_source=self._input_source()
                )
            output = run.output.output if run.output is not None else None
            if not output:
                raise _RunFailure(
                    "missing_output", "The run produced no output to judge."
                )
            # Belt-and-braces tagging (normally a no-op — default_tags above
            # already landed the tags in the run's own save). Inside the
            # try: a failure here surfaces as case_failed, never a silent
            # drop.
            async with self._save_context():
                tag_single_turn_drive_run(run, self._batch_tag)
            case_cost = _run_cost(run)
            self._total_cost += case_cost
            # The judge scores the REQUEST's input string, verbatim: the
            # saved eval stores this same string on its inputs-only item and
            # the eval-time judge reads it from there (EvalTaskInput.
            # from_trace → user_message.text), so this is the byte-
            # identical pairing. The persisted run's own `input` can differ
            # in whitespace for structured tasks (the adapter re-serializes
            # the parsed dict) — that variant is never what a judge reads.
            self._case_io[case_index] = (input_text, output)
            self._any_case_driven = True
            run_id = str(run.id) if run.id is not None else ""
            await self._emit(
                PipelineCaseDrivenEvent(case_index=case_index, leaf_run_id=run_id)
            )
            # The run's structured trace (tool calls included) is what the
            # judge scores and what rides the case_judged frame. A run that
            # recorded none is judged on a two-message echo of its I/O pair,
            # which is lossless: the pair is everything that happened.
            trace = trace_or_echo(
                [dict(message) for message in run.trace] if run.trace else None,
                input_text,
                output,
            )
            self._review_tasks.append(
                asyncio.create_task(
                    self._judge_case(case_index, run_id, trace, case_cost),
                    name=f"judge_case_{case_index}",
                )
            )
        except _RunFailure:
            await self._delete_partial_run(run)
            raise
        except asyncio.CancelledError:
            # Stopping the batch cancels in-flight cases; a persisted run
            # must not outlive its case as an untagged orphan. Shield the
            # delete so the cancellation unwinding this task can't kill it
            # mid-write, then re-raise — cooperative cancellation must
            # always propagate.
            await asyncio.shield(self._delete_partial_run(run))
            raise
        except Exception as e:
            # Adapter network errors, model misconfig, save blow-ups,
            # anything unexpected. Log with full traceback; clean this
            # attempt's run, then classify: transient errors retry, the
            # rest fail the case.
            logger.exception(
                "single_turn_pipeline: unexpected error in case %d", case_index
            )
            await self._delete_partial_run(run)
            # The adapter's KilnRunError message is genericized user-facing
            # text — unwrap so failure events name the real provider failure
            # instead of the generic wrapper text.
            cause = unwrap_kiln_run_error(e)
            detail = str(cause).strip()
            if not detail and isinstance(cause, TimeoutError):
                # A raw provider timeout lands here (no app-level budget
                # remains) and carries no message; name it so the
                # case_failed frame isn't an empty string.
                detail = "The model provider request timed out."
            if is_retryable_error(e):
                raise RetryableError(f"{type(cause).__name__}: {detail}") from e
            raise _RunFailure(
                "unexpected_error",
                f"{type(cause).__name__}: {detail}",
                type(cause).__name__,
            ) from e

    async def _delete_partial_run(self, run: TaskRun | None) -> None:
        """Best-effort removal of a failed attempt's persisted run, banking
        its real spend first — the billing happened even though the run is
        discarded. Never raises: the terminal failure the caller is about
        to raise is the event that matters."""
        if run is None:
            return
        self._total_cost += _run_cost(run)
        try:
            async with self._save_context():
                run.delete()
        except Exception:
            logger.exception(
                "single_turn_pipeline: failed to clean up a failed case's run"
            )

    def _input_source(self) -> DataSource:
        """Attribute each run's input to the model that generated it (the
        input-generator lane) plus the batch tag — the same provenance
        shape the /generate output writer and the SU runner record."""
        return DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": self._input.input_model_name,
                "model_provider": self._input.input_provider.value,
                "adapter_name": _SINGLE_TURN_ADAPTER_NAME,
                "batch_tag": self._batch_tag,
            },
        )


def connect_eval_builder_api(app: FastAPI):
    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/eval_builder/multi_turn_pipeline",
        tags=["Eval Builder"],
        summary="Run Multi-Turn Pipeline",
        openapi_extra=agent_policy_require_approval(
            "Drive multi-turn synthetic-user conversations and judge each? "
            "Invokes the target model, SU driver, and judge (cost)."
        ),
    )
    @no_write_lock  # streaming route: lock would buffer the SSE and break cancel-on-disconnect
    async def multi_turn_pipeline(
        request: Request,
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[str, Path(description="The unique identifier of the task.")],
        input: MultiTurnPipelineRequest,
    ) -> CancellableStreamingResponse:
        """The merged multi-turn stream: [drive → judge] per case.

        Emits (all frames `type`-discriminated; errors carry {code, message}):
          - batch_started   { batch_tag, total_cases }
          - turn_completed  { case_index, turns_completed, total_turns }
          - case_driven     { case_index, leaf_run_id }
          - case_judged     { case_index, leaf_run_id, raw_input, raw_output,
                              judge_score, judge_reasoning, total_cost }
          - case_failed     { case_index, stage, code, message, error_type }
                              (batch continues)
          - batch_completed { judged, failed, batch_tag, total_cost }
          - batch_aborted   { error, stage }  (in place of batch_completed:
                              a config-scoped judge failure aborted the whole
                              batch; results already streamed remain valid)
          - batch_failed    { code, message }  (in place of batch_completed:
                              an orchestration-level crash ended the stream;
                              results already streamed remain valid)
        Terminated by `data: complete`. Claims are built afterwards, per
        opened trace, via build_claims.
        """
        # Guard + decode before the stream opens so the client sees a clean
        # 4xx rather than a half-open text/event-stream. The copilot key
        # check comes first: this stream runs entirely on the user's keys,
        # but the review that follows it builds claims through the remote
        # claim builder — discovering a missing key there would be AFTER the
        # user burned their own model spend driving and judging every case.
        get_copilot_api_key()
        task = task_from_id(project_id, task_id)
        guard_multiturn(task)
        try:
            runner_cases = [RunnerCase.model_validate(c) for c in input.cases]
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_case_shape",
                    "message": f"Could not parse cases against the runner shape: {exc}",
                },
            ) from exc

        run = MultiTurnPipelineRun(
            project_id=project_id,
            task_id=task_id,
            task=task,
            cases=runner_cases,
            input=input,
            save_context=build_save_context(request),
        )
        return CancellableStreamingResponse(
            content=run.events(),
            media_type="text/event-stream",
        )

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/eval_builder/single_turn_pipeline",
        tags=["Eval Builder"],
        summary="Run Single-Turn Review Pipeline",
        openapi_extra=agent_policy_require_approval(
            "Run the task once per generated input and judge each result? "
            "Invokes the target model (tools live) and the judge (cost)."
        ),
    )
    @no_write_lock  # streaming route: lock would buffer the SSE and break cancel-on-disconnect
    async def single_turn_pipeline(
        request: Request,
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[str, Path(description="The unique identifier of the task.")],
        input: SingleTurnPipelineRequest,
    ) -> CancellableStreamingResponse:
        """The single-turn stream: [run → judge] per generated input.

        The one-turn sibling of multi_turn_pipeline: the task runs ONCE per
        input on the target run config — tools live, the user's keys — and
        each persisted, batch-tagged run is judged locally.

        Emits (all frames `type`-discriminated; errors carry {code, message}):
          - batch_started   { batch_tag, total_cases }
          - case_driven     { case_index, leaf_run_id }
          - case_judged     { case_index, leaf_run_id, raw_input, raw_output,
                              judge_score, judge_reasoning, total_cost,
                              trace }
          - case_failed     { case_index, stage: "run" | "judge", code,
                              message, error_type }  (batch continues)
          - batch_completed { judged, failed, batch_tag, total_cost }
          - batch_aborted   { error, stage }  (in place of batch_completed:
                              a config-scoped judge failure aborted the whole
                              batch; results already streamed remain valid)
          - batch_failed    { code, message }  (in place of batch_completed:
                              an orchestration-level crash ended the stream;
                              results already streamed remain valid)
        Terminated by `data: complete`. No turn frames appear on this stream
        (each case is one run). raw_input is the run's own input string,
        kept verbatim because the saved eval reads that same string back;
        raw_output is the role-labelled transcript rendering, not the closing
        message. `trace` is the run's structured trace (tool calls included)
        and is what the judge scored, matching what the saved eval will
        score. Claims are built afterwards, per opened trace, via
        build_claims.
        """
        # Same fail-fast posture as multi_turn_pipeline: this stream runs
        # entirely on the user's keys, but the review that follows builds
        # claims through the remote claim builder — discovering a missing
        # key there would be AFTER the user burned their own model spend
        # running and judging every case.
        get_copilot_api_key()
        task = task_from_id(project_id, task_id)
        guard_single_turn(task)
        run = SingleTurnPipelineRun(
            project_id=project_id,
            task_id=task_id,
            task=task,
            input=input,
            save_context=build_save_context(request),
        )
        return CancellableStreamingResponse(
            content=run.events(),
            media_type="text/event-stream",
        )

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/eval_builder/judge_traces",
        tags=["Eval Builder"],
        summary="Judge Saved Eval-Builder Results",
        openapi_extra=agent_policy_require_approval(
            "Re-judge saved eval-builder results with a judge prompt? "
            "Invokes the judge model per result (cost)."
        ),
    )
    @no_write_lock  # streaming route: lock would buffer the SSE and break cancel-on-disconnect
    async def judge_traces(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[str, Path(description="The unique identifier of the task.")],
        input: JudgeTracesRequest,
    ) -> CancellableStreamingResponse:
        """Re-judge previously driven results: [reload → judge] per case.

        The judge calibration loop's re-score stream, both arms: after a
        refine produces a new judge prompt, this scores the SAME saved
        results again. Each run is reloaded from disk by id; multi-turn
        judges the chain leaf's stored trace, single-turn the run's own —
        either way the judge input matches what the saved eval will judge.
        Nothing is driven and nothing is written.

        Emits (all frames `type`-discriminated; errors carry {code, message}):
          - batch_started   { batch_tag: "", total_cases }
          - case_judged     { case_index, leaf_run_id, raw_input, raw_output,
                              judge_score, judge_reasoning, total_cost: 0,
                              trace }
          - case_failed     { case_index, stage: "judge", code, message,
                              error_type }
                              (batch continues; a run that cannot be
                              reloaded fails with code trace_not_found,
                              missing_trace, or missing_output)
          - batch_completed { judged, failed, batch_tag: "", total_cost: 0 }
          - batch_aborted   { error, stage: "judge" }  (in place of
                              batch_completed: a config-scoped judge failure
                              aborted the whole batch; results already
                              streamed remain valid)
          - batch_failed    { code, message }  (in place of batch_completed:
                              an orchestration-level crash ended the stream;
                              results already streamed remain valid)
        Terminated by `data: complete`. case_index is the position in
        leaf_run_ids; no drive or turn frames appear on this stream. Claims
        are built afterwards, per opened trace, via build_claims.
        """
        # Same fail-fast posture as multi_turn_pipeline: the judge runs on the
        # user's keys, but the review that follows builds claims through the
        # remote claim builder — surface a missing copilot key before the
        # user spends on judging every case.
        get_copilot_api_key()
        task = task_from_id(project_id, task_id)
        run = JudgeTracesRun(
            project_id=project_id,
            task_id=task_id,
            task=task,
            input=input,
        )
        return CancellableStreamingResponse(
            content=run.events(),
            media_type="text/event-stream",
        )

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/eval_builder/build_claims",
        tags=["Eval Builder"],
        openapi_extra=agent_policy_require_approval(
            "Build claim/evidence for a trace?"
        ),
    )
    async def build_claims(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[str, Path(description="The unique identifier of the task.")],
        input: BuildClaimsApiInput,
    ) -> BuildClaimsApiOutput:
        """Claims-only primitive: build claims for one trace given a known verdict.

        The multi-turn review's claims path: the pipeline stream stops at the
        judge, and the client calls this per trace the reviewer opens (under
        subset review most traces are never opened). Also used by the refine
        loop to regenerate claims without re-running the judge.
        """
        # The builder reads the task's own instruction as context for what the
        # task is; the endpoint resolves it here so the client never sends it.
        task = task_from_id(project_id, task_id)
        output = await build_claims_for_trace(
            task_instruction=task.instruction,
            raw_input=input.raw_input,
            raw_output=input.raw_output,
            eval_rubric=input.eval_rubric,
            judge_score=input.judge_score,
            judge_reasoning=input.judge_reasoning,
        )
        return output

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/eval_builder/preflight_model",
        tags=["Eval Builder"],
        summary="Preflight a Model Lane",
        openapi_extra=agent_policy_require_approval(
            "Send a one-word test completion to verify a model config works? "
            "(negligible cost)"
        ),
    )
    async def preflight_model(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[str, Path(description="The unique identifier of the task.")],
        input: PreflightModelApiInput,
    ) -> PreflightModelApiOutput:
        """One cheap completion through the SAME adapter model/provider
        resolution a real run uses (that resolution is where a dead model
        surfaces), on the user's same keys. Catches key/billing/deprecation/
        unreachable failures for a lane BEFORE the drive commits the
        plan/SU-gen minutes and the batch's model spend. Explicitly does NOT
        validate tools/MCP or mid-run rate limits. Nothing persists:
        allow_saving=False, so no TaskRun lands in the dataset — same as
        the transient review judge.
        """
        task_from_id(project_id, task_id)  # 404 on a bad path; not used further
        # A transient one-liner task, NOT the real task prompt: the check
        # verifies the lane responds at all, at the smallest possible spend.
        preflight_task = Task(
            name="preflight_check",
            instruction='Reply with exactly "OK".',
        )
        try:
            adapter = adapter_for_task(
                preflight_task,
                run_config_properties=KilnAgentRunConfigProperties(
                    model_name=input.model_name,
                    model_provider_name=input.model_provider,
                    prompt_id=PromptGenerators.SIMPLE,
                    structured_output_mode=StructuredOutputMode.default,
                ),
                base_adapter_config=AdapterConfig(allow_saving=False),
            )
            await adapter.invoke(input="Say OK")
        except Exception as e:
            root = unwrap_kiln_run_error(e)
            # litellm exceptions already lead with their class name
            # ("litellm.APIError: …") — prefixing the type again would
            # stutter; only add it when the message doesn't carry it.
            root_str = str(root)
            type_name = type(root).__name__
            message = (
                root_str
                if root_str.startswith((type_name, f"litellm.{type_name}"))
                else f"{type_name}: {root_str}"
            )
            raise HTTPException(
                status_code=400,
                detail={"code": "preflight_failed", "message": message},
            ) from e
        return PreflightModelApiOutput()

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/eval_builder/author_judge",
        tags=["Eval Builder"],
        openapi_extra=agent_policy_require_approval(
            "Author a judge prompt tailored to the spec?"
        ),
    )
    async def author_judge(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[str, Path(description="The unique identifier of the task.")],
        input: AuthorJudgeApiInput,
    ) -> AuthorJudgeApiOutput:
        """Author a spec-tailored judge prompt for the review — both arms.

        Returns the PROMPT only — the judge model is the user's pick. Both
        arms judge a transcript, so both rubrics are authored against one:
        the rubric arrives knowing the role labels and tool-call blocks its
        judge will meet, whatever the task's turn mode.
        Authoring is a REQUIRED step of the drive: an error here stops the
        drive on a retryable error client-side. There is no fallback judge.
        """
        # Fail fast on a missing copilot key before the remote authoring call:
        # a keyless caller gets a clean 401, not a deep upstream error.
        get_copilot_api_key()
        task = task_from_id(project_id, task_id)
        # The task is loaded for its tools and skills, so the rubric can grade
        # tool and skill use instead of guessing at it. The caller names the
        # run config the eval is about; without one the task default is read.
        task_tools, task_skills = await task_capabilities_for_task(
            task, input.run_config_id
        )
        return await author_judge_prompt(
            target_specification=input.target_specification,
            target_task_prompt=input.target_task_prompt,
            # Constant on purpose: both arms judge a transcript, so both
            # rubrics are authored against one. "multi_turn" names the
            # authoring prompt that teaches the transcript's vocabulary — the
            # role labels, the tool-call blocks, and that a missing tool call
            # is evidence — not the turn mode of the task being judged.
            trace_type="multi_turn",
            task_tools=task_tools,
            task_skills=task_skills,
        )

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/eval_builder/refine_judge",
        tags=["Eval Builder"],
        openapi_extra=agent_policy_require_approval(
            "Refine the judge prompt from the reviewer's grades?"
        ),
    )
    async def refine_judge(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[str, Path(description="The unique identifier of the task.")],
        input: RefineJudgeApiInput,
    ) -> RefineJudgeApiOutput:
        """Propose a judge-prompt revision from the human's per-claim grades.

        The refined prompt is a PROPOSAL — the UI validates it and shows the
        changes for approval; it is never auto-applied.
        """
        # Remote failures propagate as HTTPExceptions with the upstream's
        # message (custom_errors renders {"message": ...} for the UI), same as
        # the build_claims primitive.
        return await refine_judge_prompt_from_grades(
            judge_prompt=input.judge_prompt,
            graded_traces=input.graded_traces,
        )
