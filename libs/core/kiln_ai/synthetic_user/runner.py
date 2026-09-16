"""Batch runner — fans drive_case out across N cases, streams BatchEvents.

`run_cases_batch` is an async generator yielding typed events. Cases run
through a shared AsyncJobRunner worker pool (the same fan-out engine the
eval runner uses): transient provider failures retry up to DRIVE_MAX_RETRIES,
and a per-case failure surfaces as ONE `CaseFailedEvent` — after the last
attempt — without affecting other in-flight cases.
"""

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

from kiln_ai.adapters.adapter_registry import adapter_for_task, load_skills_for_task
from kiln_ai.adapters.model_adapters.base_adapter import AdapterConfig, SkillsDict
from kiln_ai.adapters.retry_classification import (
    is_retryable_error,
    unwrap_kiln_run_error,
)
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties
from kiln_ai.datamodel.task import Task
from kiln_ai.datamodel.task_output import DataSource, DataSourceType
from kiln_ai.datamodel.task_run import TaskRun
from kiln_ai.synthetic_user.case import SyntheticUserCase
from kiln_ai.synthetic_user.drive_loop import TargetInvoker, drive_case
from kiln_ai.synthetic_user.driver import SyntheticUserDriver
from kiln_ai.synthetic_user.models import (
    TAG_SU_ENDED_CONVERSATION,
    SyntheticUserDriverConfig,
)
from kiln_ai.synthetic_user.parser import (
    SyntheticUserInfoParseError,
    parse_synthetic_user_info,
)
from kiln_ai.utils.async_job_runner import (
    AsyncJobRunner,
    AsyncJobRunnerObserver,
    RetryableError,
)
from kiln_ai.utils.git_sync_protocols import SaveContext, default_save_context
from kiln_ai.utils.open_ai_types import ChatCompletionMessageParam
from kiln_ai.utils.slow_operation import log_if_slow

logger = logging.getLogger(__name__)

# Module constants.
# Largest supported batch size. The runner itself doesn't enforce it: it's the
# shared limit callers validate against, and it's mirrored anywhere a batch
# size is chosen, so this is the one place to change it. Each case is a full
# multi-turn conversation, so the cap bounds worst-case fan-out cost and wall
# time.
NUM_CASES_MAX = 200
MAX_TURNS_DEFAULT = 5
CONCURRENCY = 4
# Transient drive failures (see kiln_ai.adapters.retry_classification) retry
# up to this many times before the case is declared failed — the same retry
# posture as the eval runner. Timeouts and deterministic input problems are
# never retried.
DRIVE_MAX_RETRIES = 2
# Base of the job runner's exponential-backoff-with-jitter window, not a flat wait.
DRIVE_RETRY_DELAY_SECONDS = 1.0

# Tag scheme. `_TAG_SU_CASE` lets consumers filter to all SU-generated
# leaves; `_TAG_PREFIX_SU_BATCH` (+ batch_tag) groups one batch for review.
_TAG_SU_CASE = "synthetic_user_case"
_TAG_PREFIX_SU_BATCH = "synthetic_user_batch:"

# Identifies this code path in `input_source.properties.adapter_name` so a
# reader looking at a TaskRun can tell who created it.
_RUNNER_ADAPTER_NAME = "kiln_synthetic_user_runner"


# ───────────────────────── BatchEvent dataclasses ─────────────────────────


@dataclass(frozen=True)
class BatchStartedEvent:
    batch_tag: str
    num_cases: int


@dataclass(frozen=True)
class TurnCompletedEvent:
    case_index: int
    # 1-based turn number within the CURRENT drive attempt. Consumers must
    # read this rather than count events — a retried case restarts at 1.
    turn_index: int
    assistant_run_id: str
    # The SU reply that seeds the next turn. None on the case's last turn:
    # either the drive loop hit the turn ceiling and skipped the SU call
    # because no target turn would consume it, or the SU ended the
    # conversation and there is no next message to seed.
    su_next_message: str | None
    cumulative_cost: float
    # Cumulative OpenAI-format trace at this point (system + all turns so far).
    # Lets consumers observe the live conversation without reloading the run.
    trace: list[ChatCompletionMessageParam]


@dataclass(frozen=True)
class CaseCompletedEvent:
    case_index: int
    chain_run_ids: list[str]
    leaf_run_id: str
    total_turns: int
    # Target adapter cost + SU driver cost for THIS conversation (the
    # surviving chain only).
    total_cost: float
    # Real provider spend of this case's earlier failed attempts — their
    # chains were deleted, but the billing happened. Add to total_cost for
    # what the case actually cost end to end.
    discarded_attempts_cost: float = 0.0


@dataclass(frozen=True)
class CaseFailedEvent:
    case_index: int
    error_code: str
    message: str
    # Actual provider spend across ALL of this case's attempts. Nothing
    # survives on disk, but the billing was real.
    total_cost: float = 0.0
    # Class name of the provider or unexpected exception behind the failure, so
    # consumers can group failures by kind instead of parsing `message`. None on
    # deterministic failures — `error_code` already names those, even the ones an
    # exception triggered.
    error_type: str | None = None


@dataclass(frozen=True)
class BatchCompletedEvent:
    successful: int
    failed: int
    batch_tag: str
    # Actual provider spend for the batch: successful conversations plus
    # every failed or retried attempt's discarded spend.
    total_cost: float


BatchEvent = (
    BatchStartedEvent
    | TurnCompletedEvent
    | CaseCompletedEvent
    | CaseFailedEvent
    | BatchCompletedEvent
)


# ───────────────────────── public entry point ─────────────────────────


async def run_cases_batch(
    *,
    cases: list[SyntheticUserCase],
    target_task: Task,
    target_run_config: KilnAgentRunConfigProperties,
    su_driver_config: SyntheticUserDriverConfig,
    turns: int = MAX_TURNS_DEFAULT,
    concurrency: int = CONCURRENCY,
    batch_tag: str | None = None,
    save_context: SaveContext | None = None,
    task_run_config_id: str | None = None,
    case_timeout_seconds: float | None = None,
) -> AsyncIterator[BatchEvent]:
    """Drive `cases` concurrently against `target_task`, streaming progress.

    Cases run through an AsyncJobRunner worker pool bounded by `concurrency`,
    so the target-task LLM and the SU LLM aren't both hammered at unbounded
    fan-out. Events from different cases interleave; ordering WITHIN a case
    is `turn_completed`* → `case_completed | case_failed`.

    `case_timeout_seconds` optionally bounds each case's drive; a case that
    exceeds it fails with `case_timeout` and frees its concurrency slot while
    the batch continues. The default (None) is unbounded: termination is
    guaranteed by structural bounds instead — the turn ceiling, the
    adapter's per-turn tool-call cap, and the model client's per-request
    timeout — and a case running past a soft threshold logs a warning
    rather than being killed.

    Transient provider failures retry the whole case (up to
    DRIVE_MAX_RETRIES) — its turn events restart at 1; deterministic
    failures and caller-set timeout firings fail immediately.

    Yields:
      BatchStartedEvent — once, before any case runs.
      TurnCompletedEvent — one per assistant turn within a case (attempt).
      CaseCompletedEvent — one per case that ran to completion.
      CaseFailedEvent — one per case whose last attempt failed.
      BatchCompletedEvent — once, after all cases finish.
    """
    if not cases:
        raise ValueError("cases cannot be empty")
    if turns < 1:
        raise ValueError("turns must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if case_timeout_seconds is not None and case_timeout_seconds <= 0:
        raise ValueError("case_timeout_seconds must be > 0")

    resolved_batch_tag = batch_tag or _new_batch_tag()
    # Skills referenced by the run config must be pre-loaded at the
    # orchestration layer and injected via AdapterConfig — the adapter
    # raises if it meets a skill tool id with no injected dict. One
    # directory scan covers the whole batch.
    skills = load_skills_for_task(target_task, target_run_config)
    save_ctx: SaveContext = save_context or default_save_context
    # `None` is the end-of-stream sentinel pushed when all cases finish.
    queue: asyncio.Queue[BatchEvent | None] = asyncio.Queue()
    # Real spend of failed attempts, keyed by case index. Outlives retries
    # (each attempt banks its spend before its chain is deleted) so the
    # case's completion/failure event can report what was actually billed.
    failed_attempt_spend: dict[int, float] = {}

    async def _drive_job(job: tuple[int, SyntheticUserCase]) -> bool:
        case_index, case = job
        await _drive_one_case_and_emit(
            case_index=case_index,
            case=case,
            target_task=target_task,
            target_run_config=target_run_config,
            su_driver_config=su_driver_config,
            turns=turns,
            batch_tag=resolved_batch_tag,
            queue=queue,
            save_ctx=save_ctx,
            skills=skills,
            task_run_config_id=task_run_config_id,
            case_timeout_seconds=case_timeout_seconds,
            failed_attempt_spend=failed_attempt_spend,
        )
        return True

    class _EmitCaseFailed(AsyncJobRunnerObserver[tuple[int, SyntheticUserCase]]):
        """Emits case_failed exactly once per dead case — the job runner
        calls on_error only after retries are exhausted (or immediately
        for non-retryable failures)."""

        async def on_error(
            self, job: tuple[int, SyntheticUserCase], error: Exception
        ) -> None:
            case_index, _case = job
            code, message, error_type = _failure_details(error)
            await queue.put(
                CaseFailedEvent(
                    case_index=case_index,
                    error_code=code,
                    message=message,
                    total_cost=failed_attempt_spend.get(case_index, 0.0),
                    error_type=error_type,
                )
            )

    # AsyncJobRunner is the shared fan-out engine (same as the eval runner):
    # a worker pool bounded by `concurrency`, retrying cases whose drive
    # raised RetryableError (transient provider failures, classified inside
    # _drive_one_case_and_emit) before declaring them dead.
    runner = AsyncJobRunner(
        jobs=list(enumerate(cases)),
        run_job_fn=_drive_job,
        concurrency=concurrency,
        max_retries=DRIVE_MAX_RETRIES,
        retry_delay=DRIVE_RETRY_DELAY_SECONDS,
        observers=[_EmitCaseFailed()],
    )

    async def _run_jobs() -> None:
        # The runner's coarse Progress stream goes unused — this generator's
        # protocol is the typed BatchEvents the jobs put on `queue`; draining
        # it is what drives the workers. The sentinel closes the stream once
        # every case has completed or exhausted its attempts.
        try:
            async for _progress in runner.run():
                pass
        finally:
            queue.put_nowait(None)

    # Kick the batch off BEFORE the first yield so cases start running
    # concurrently with consumer setup; the `finally` below still tears it
    # down if the consumer disconnects immediately. Named so asyncio debug
    # dumps point at this code path.
    jobs_task = asyncio.create_task(
        _run_jobs(), name=f"su_batch_{resolved_batch_tag[:6]}"
    )

    successful = 0
    failed = 0
    total_cost = 0.0
    try:
        yield BatchStartedEvent(batch_tag=resolved_batch_tag, num_cases=len(cases))

        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
            if isinstance(event, CaseCompletedEvent):
                successful += 1
                total_cost += event.total_cost + event.discarded_attempts_cost
            elif isinstance(event, CaseFailedEvent):
                failed += 1
                total_cost += event.total_cost

        yield BatchCompletedEvent(
            successful=successful,
            failed=failed,
            batch_tag=resolved_batch_tag,
            total_cost=total_cost,
        )
    finally:
        # If the consumer stops iterating the generator early, cancel the job
        # runner — its own teardown cancels in-flight case workers, so
        # abandoned drives stop spending.
        jobs_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await jobs_task


# ───────────────────────── per-case orchestration ─────────────────────────


class _CaseFailure(Exception):
    """A terminal per-case failure — never retried.

    Deterministic input problems, caller-set per-case timeouts (a retry
    would pin a worker for another full drive budget), and provider errors
    the shared classifier calls permanent. `code` and `error_type` are
    surfaced on CaseFailedEvent.
    """

    def __init__(self, code: str, message: str, error_type: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        # Class name of the provider or unexpected exception behind this
        # failure; None on deterministic failures, whose `code` already names
        # them.
        self.error_type = error_type


def _failure_details(error: Exception) -> tuple[str, str, str | None]:
    """Map a case's terminal exception to CaseFailedEvent's code, message and
    error type."""
    if isinstance(error, _CaseFailure):
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


async def _drive_one_case_and_emit(
    *,
    case_index: int,
    case: SyntheticUserCase,
    target_task: Task,
    target_run_config: KilnAgentRunConfigProperties,
    su_driver_config: SyntheticUserDriverConfig,
    turns: int,
    batch_tag: str,
    queue: asyncio.Queue[BatchEvent | None],
    save_ctx: SaveContext,
    skills: SkillsDict,
    task_run_config_id: str | None,
    case_timeout_seconds: float | None,
    failed_attempt_spend: dict[int, float],
) -> None:
    """Run drive_case for one case (one ATTEMPT), emitting turn/completion
    events on `queue`.

    Failures RAISE instead of emitting: transient provider errors become
    RetryableError (the job runner re-runs the case), everything else
    becomes _CaseFailure, and case_failed is emitted once — by the runner's
    on_error observer, after the last attempt. Any turns a failed or
    cancelled attempt persisted are removed before raising, so a retry
    starts clean — but the attempt's real spend is banked in
    `failed_attempt_spend` first, so cost events stay honest about billing.
    """
    # Runs persist per turn (adapter autosave) but the batch tag only lands
    # on the leaf after a successful drive, so a mid-drive failure would
    # strand an untagged chain no downstream consumer can find — discovery
    # is tag-based. The invoker wrapper below tracks each run the moment it
    # persists so the failure arms can remove the full chain.
    persisted_runs: dict[str, TaskRun] = {}
    # SU spend of THIS attempt, accumulated per turn via the hook —
    # drive_case's own total is lost when it raises mid-case.
    attempt_su_cost = 0.0
    try:
        # Malformed blob fails this case without affecting others. Parsing
        # happens here — the wire boundary — so the driver only ever sees
        # the typed persona.
        try:
            su_info = parse_synthetic_user_info(case.synthetic_user_info)
        except SyntheticUserInfoParseError as e:
            # Deterministic: retrying replays the same parse on the same bytes.
            raise _CaseFailure("bad_synthetic_user_info", str(e)) from e
        su_driver = SyntheticUserDriver(su_info, su_driver_config)

        adapter_invoker = _make_target_invoker(
            case=case,
            target_task=target_task,
            target_run_config=target_run_config,
            su_driver_config=su_driver_config,
            batch_tag=batch_tag,
            skills=skills,
            task_run_config_id=task_run_config_id,
        )

        async def _target_invoker(
            *,
            input: str,
            prior_trace: list[ChatCompletionMessageParam] | None,
            parent_task_run: TaskRun | None,
        ) -> TaskRun:
            # Record each run the moment it exists on disk (the adapter
            # autosaves inside invoke) — cleanup must see a mid-turn persist
            # even when the turn's SU half never runs.
            run = await adapter_invoker(
                input=input,
                prior_trace=prior_trace,
                parent_task_run=parent_task_run,
            )
            if run.id is not None:
                persisted_runs[str(run.id)] = run
            return run

        turns_completed = 0

        async def _on_turn(
            *, run: TaskRun, su_message: str | None, su_cost: float
        ) -> None:
            nonlocal turns_completed, attempt_su_cost
            turns_completed += 1
            attempt_su_cost += su_cost
            await queue.put(
                TurnCompletedEvent(
                    case_index=case_index,
                    turn_index=turns_completed,
                    assistant_run_id=str(run.id) if run.id is not None else "",
                    su_next_message=su_message,
                    cumulative_cost=_cumulative_cost(run),
                    # Snapshot the cumulative trace so consumers see the
                    # conversation as of this turn.
                    trace=list(run.trace) if run.trace else [],
                )
            )

        # Unbounded by default (timeout=None): a hung provider call is
        # bounded by the model client's per-request timeout, and the drive
        # terminates structurally (turn ceiling, per-turn tool-call cap).
        # An explicit case_timeout_seconds still bounds the whole drive when
        # a caller sets one; either way the watchdog makes a pathologically
        # slow case visible in logs without killing a healthy run.
        async with log_if_slow(f"synthetic_user runner: case {case_index}"):
            result = await asyncio.wait_for(
                drive_case(
                    seed_prompt=case.seed_prompt,
                    target_invoker=_target_invoker,
                    su_driver=su_driver,
                    turns=turns,
                    on_turn=_on_turn,
                ),
                timeout=case_timeout_seconds,
            )

        ended_by_su = result.ended_early(turns)
        if len(result.chain) == 1 and turns > 1:
            # Not enforced: killing a paid drive over a rule the prompt states
            # but cannot guarantee would cost more than it saves. Logged
            # because a one-turn conversation is a single-turn trace wearing a
            # multi-turn label, which is worth knowing about when reading
            # results back.
            logger.warning(
                "synthetic_user runner: case %d ended on its first turn — the "
                "synthetic user ended the conversation before the target had "
                "answered a second message",
                case_index,
            )

        # Tag the leaf so eval-time loaders can find it. Inside the try
        # so a tag-save failure (full disk, validator rejection on a
        # malformed batch_tag) surfaces as case_failed, not a silent drop.
        leaf = result.chain[-1]
        async with save_ctx():
            _tag_leaf(leaf, batch_tag, ended_by_su=ended_by_su)

        await queue.put(
            CaseCompletedEvent(
                case_index=case_index,
                chain_run_ids=[
                    str(r.id) if r.id is not None else "" for r in result.chain
                ],
                leaf_run_id=str(leaf.id) if leaf.id is not None else "",
                total_turns=len(result.chain),
                total_cost=_cumulative_cost(leaf) + result.su_total_cost,
                discarded_attempts_cost=failed_attempt_spend.get(case_index, 0.0),
            )
        )
    except _CaseFailure:
        # Raised above before anything persisted (parse) — pass through.
        raise
    except asyncio.CancelledError:
        # Stopping the batch cancels in-flight cases mid-drive; their
        # persisted turns must not outlive the case as untagged orphans.
        # Shield the delete so the cancellation unwinding this task can't
        # kill it mid-chain, then re-raise — cooperative cancellation must
        # always propagate.
        _bank_attempt_spend(
            failed_attempt_spend, case_index, persisted_runs, attempt_su_cost
        )
        await asyncio.shield(_delete_partial_chain(persisted_runs, save_ctx))
        raise
    except asyncio.TimeoutError as e:
        _bank_attempt_spend(
            failed_attempt_spend, case_index, persisted_runs, attempt_su_cost
        )
        await _delete_partial_chain(persisted_runs, save_ctx)
        if case_timeout_seconds is None:
            # No case budget is set, so wait_for cannot have raised this:
            # it is a provider/network timeout surfacing raw. Classify it
            # like any other unexpected drive error (transient errors retry).
            logger.exception(
                "synthetic_user runner: unexpected error in case %d", case_index
            )
            cause = unwrap_kiln_run_error(e)
            # A bare TimeoutError carries no message; name the failure so
            # the case_failed frame isn't an empty string.
            detail = str(cause).strip() or "The model provider request timed out."
            if is_retryable_error(e):
                raise RetryableError(f"{type(cause).__name__}: {detail}") from e
            raise _CaseFailure(
                "unexpected_error",
                f"{type(cause).__name__}: {detail}",
                type(cause).__name__,
            ) from e
        # The drive exceeded its caller-set per-case budget; wait_for already
        # cancelled it. The partial chain was removed like any failed attempt.
        logger.warning(
            "synthetic_user runner: case %d timed out after %.0fs",
            case_index,
            case_timeout_seconds,
        )
        raise _CaseFailure(
            "case_timeout",
            f"The conversation did not finish within "
            f"{case_timeout_seconds:.0f}s and was cancelled.",
        ) from e
    except Exception as e:
        # Adapter network errors, model misconfig, save_to_file blow-up,
        # anything unexpected. Log with full traceback; clean this attempt's
        # partial chain, then classify: transient errors retry, the rest
        # fail the case.
        logger.exception(
            "synthetic_user runner: unexpected error in case %d", case_index
        )
        _bank_attempt_spend(
            failed_attempt_spend, case_index, persisted_runs, attempt_su_cost
        )
        await _delete_partial_chain(persisted_runs, save_ctx)
        # The adapter's KilnRunError message is genericized user-facing
        # text — unwrap so failure events name the real provider failure
        # instead of the generic wrapper text.
        cause = unwrap_kiln_run_error(e)
        if is_retryable_error(e):
            raise RetryableError(f"{type(cause).__name__}: {cause}") from e
        raise _CaseFailure(
            "unexpected_error",
            f"{type(cause).__name__}: {cause}",
            type(cause).__name__,
        ) from e


def _bank_attempt_spend(
    failed_attempt_spend: dict[int, float],
    case_index: int,
    persisted_runs: dict[str, TaskRun],
    attempt_su_cost: float,
) -> None:
    """Bank a failed attempt's real spend before its chain is deleted.

    The deepest persisted run's `cumulative_usage` already rolls up the
    whole chain's target cost; the SU side is accumulated per turn by the
    caller. Deleting the chain erases the only on-disk record, so this
    accumulator is how a discarded attempt's billing reaches cost events.
    """
    target_cost = 0.0
    if persisted_runs:
        target_cost = _cumulative_cost(next(reversed(persisted_runs.values())))
    failed_attempt_spend[case_index] = (
        failed_attempt_spend.get(case_index, 0.0) + target_cost + attempt_su_cost
    )


async def _delete_partial_chain(
    persisted_runs: dict[str, TaskRun], save_ctx: SaveContext
) -> None:
    """Best-effort removal of a failed attempt's partially-driven chain.

    A chain only becomes discoverable through the leaf's batch tag, applied
    after a successful drive — runs a failed attempt persisted would
    otherwise be permanent on-disk orphans (and a retry would drive on top
    of them). Never raises: the terminal failure the caller is about to
    raise is the event that matters.
    """
    if not persisted_runs:
        return
    try:
        async with save_ctx():
            # Newest first: remove the dangling end of the chain before its
            # ancestors so an interrupted cleanup can't orphan a child run.
            for run in reversed(list(persisted_runs.values())):
                run.delete()
    except Exception:
        logger.exception(
            "synthetic_user runner: failed to clean up a failed case's "
            "partial chain (%d runs)",
            len(persisted_runs),
        )


# ───────────────────────── target invoker construction ─────────────────────


def _make_target_invoker(
    *,
    case: SyntheticUserCase,
    target_task: Task,
    target_run_config: KilnAgentRunConfigProperties,
    su_driver_config: SyntheticUserDriverConfig,
    batch_tag: str,
    skills: SkillsDict,
    task_run_config_id: str | None,
) -> TargetInvoker:
    """Build a per-case TargetInvoker over the real adapter.

    The closure tracks `turn_index` so the root run carries the full SU
    attribution context in `input_source.properties` while subsequent
    runs carry only the slim `{batch_tag, turn_index}` — the case is
    recoverable by walking `parent_task_run_id` to the root.

    Concurrency: the returned closure is NOT safe to invoke concurrently.
    `nonlocal turn_index` is incremented per call; concurrent callers
    would race on the increment and the resulting `is_root` flag.
    `drive_case` calls it sequentially within a single case (the per-turn
    loop), which is the contract; cases are isolated by having their own
    closure with their own `turn_index`.
    """
    adapter = adapter_for_task(
        target_task,
        target_run_config,
        # task_run_config_id stamps each run's output source with the saved
        # config it came from, exactly as a manual run of that config would.
        base_adapter_config=AdapterConfig(
            skills=skills, task_run_config_id=task_run_config_id
        ),
    )
    turn_index = 0

    async def _invoker(
        *,
        input: str,
        prior_trace: list[ChatCompletionMessageParam] | None,
        parent_task_run: TaskRun | None,
    ) -> TaskRun:
        nonlocal turn_index
        turn_index += 1
        input_source = _build_input_source(
            case=case,
            su_driver_config=su_driver_config,
            batch_tag=batch_tag,
            turn_index=turn_index,
            is_root=(turn_index == 1),
        )
        return await adapter.invoke(
            input=input,
            input_source=input_source,
            prior_trace=prior_trace,
            parent_task_run=parent_task_run,
        )

    return _invoker


def _build_input_source(
    *,
    case: SyntheticUserCase,
    su_driver_config: SyntheticUserDriverConfig,
    batch_tag: str,
    turn_index: int,
    is_root: bool,
) -> DataSource:
    """Attribute the user-side input on this turn to the SU driver model.

    Root run carries the decomposed case context (persona / goal /
    behavior_guidance / seed_prompt). Subsequent runs carry only the slim
    batch_tag/turn_index pair — full context is recoverable by walking
    parent_task_run_id to the root.
    """
    props: dict[str, str | int | float] = {
        "model_name": su_driver_config.model_name,
        "model_provider": su_driver_config.model_provider_name.value,
        "adapter_name": _RUNNER_ADAPTER_NAME,
        "batch_tag": batch_tag,
        "turn_index": turn_index,
    }
    if is_root:
        # Parse is cheap (regex on a short string) and was already
        # validated when the SU driver was built — re-parsing here can't
        # surface a new error class.
        info = parse_synthetic_user_info(case.synthetic_user_info)
        props["persona"] = info.persona
        props["goal"] = info.goal
        if info.behavior_guidance:
            props["behavior_guidance"] = info.behavior_guidance
        props["seed_prompt"] = case.seed_prompt

    return DataSource(type=DataSourceType.synthetic, properties=props)


# ───────────────────────── small utilities ─────────────────────────


def _new_batch_tag() -> str:
    """12-char hex tag from uuid4 — short enough to read; long enough to
    avoid collisions across batches a user runs in the same session.
    """
    return uuid.uuid4().hex[:12]


def _cumulative_cost(run: TaskRun) -> float:
    """Read the rolled-up cost from a TaskRun, defaulting to 0 if usage is
    missing (defensive against fakes in unit tests that don't populate it).
    """
    usage = getattr(run, "cumulative_usage", None)
    if usage is None:
        return 0.0
    return float(getattr(usage, "cost", None) or 0.0)


def _tag_leaf(leaf: TaskRun, batch_tag: str, *, ended_by_su: bool) -> None:
    """Add the runner's discovery tags to the leaf TaskRun and persist.

    `ended_by_su` says the synthetic user ended this conversation before the
    turn ceiling, which earns the leaf TAG_SU_ENDED_CONVERSATION. The caller
    decides — a short chain is the only evidence, and only the drive knows why
    it is short.

    Tags are deduplicated (treated as a set then sorted) so re-runs
    against an already-tagged leaf are idempotent. A save_to_file
    exception surfaces to the caller (which converts to CaseFailedEvent).

    Reentrancy: the read-modify-write on `leaf.tags` assumes a single
    writer per leaf. The current call shape guarantees this (each case
    has its own leaf), so concurrent tagging across cases always hits
    distinct files. A future refactor that shares leaves across cases
    would need to re-introduce locking here.
    """
    tags = set(leaf.tags or [])
    tags.add(_TAG_SU_CASE)
    tags.add(f"{_TAG_PREFIX_SU_BATCH}{batch_tag}")
    if ended_by_su:
        tags.add(TAG_SU_ENDED_CONVERSATION)
    leaf.tags = sorted(tags)
    leaf.save_to_file()
