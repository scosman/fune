import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Set, Tuple

from kiln_ai.adapters.adapter_registry import load_skills_for_task
from kiln_ai.adapters.chat.chat_formatter import (
    chat_strategy_for_run,
    is_two_message_cot_strategy,
)
from kiln_ai.adapters.eval.base_eval import BaseEval, BaseV2EvalBridge
from kiln_ai.adapters.eval.registry import (
    legacy_eval_adapter_from_type,
    v2_eval_type_available,
)
from kiln_ai.adapters.eval.trace_index import TraceIndex, TraceKey, trace_key
from kiln_ai.adapters.ml_model_list import built_in_models_from_provider
from kiln_ai.adapters.model_adapters.base_adapter import SkillsDict
from kiln_ai.adapters.prompt_builders import prompt_builder_from_id
from kiln_ai.adapters.provider_tools import kiln_model_provider_from
from kiln_ai.adapters.retry_classification import (
    is_retryable_error,
    unwrap_kiln_run_error,
)
from kiln_ai.datamodel.basemodel import ID_TYPE, generate_model_id
from kiln_ai.datamodel.datamodel_enums import ModelProviderName
from kiln_ai.datamodel.dataset_filters import (
    DatasetFilterId,
    dataset_filter_from_id,
)
from kiln_ai.datamodel.eval import (
    Eval,
    EvalConfig,
    EvalConfigType,
    EvalDataType,
    EvalInput,
    EvalRun,
    EvalScores,
    EvalTaskInput,
    MultiTurnSyntheticEvalInputData,
    SkippedReason,
    V2EvalResult,
)
from kiln_ai.datamodel.eval_splits import (
    ItemKey,
    ResolvedSplit,
    eval_run_item_key,
    item_key,
)
from kiln_ai.datamodel.run_config import (
    KilnAgentRunConfigProperties,
    as_kiln_agent_run_config,
)
from kiln_ai.datamodel.task import Task, TaskRunConfig
from kiln_ai.datamodel.task_run import EvalItemSource, TaskRun, Usage
from kiln_ai.datamodel.usage import MessageUsage
from kiln_ai.synthetic_user import drive_case_for_eval
from kiln_ai.synthetic_user.drive_loop import DriveCaseResult
from kiln_ai.synthetic_user.models import (
    TAG_SU_ENDED_CONVERSATION,
    SyntheticUserDriverConfig,
)
from kiln_ai.utils.async_job_runner import AsyncJobRunner, Progress, RetryableError
from kiln_ai.utils.git_sync_protocols import SaveContext, default_save_context
from kiln_ai.utils.open_ai_types import ChatCompletionMessageParam, serialize_trace
from kiln_ai.utils.slow_operation import log_if_slow

logger = logging.getLogger(__name__)


@dataclass
class EvalJob:
    item: TaskRun | EvalInput
    type: Literal["task_run_eval", "eval_config_eval"]
    eval_config: EvalConfig
    task_run_config: TaskRunConfig | None = None
    # Recoverable-skip records this job replaces (their blocking condition
    # has lifted). Deleted after the job persists its record — leaving them
    # would put two records on one item and read paths take the first found.
    superseded_tombstones: List[EvalRun] = field(default_factory=list)


def _calibration_item(job: EvalJob) -> TaskRun | None:
    """The golden TaskRun a calibration job scores, or None if this isn't calibration.

    Calibration generates nothing: the human-rated dataset item *is* what gets scored
    (functional spec 4.5). So the trace is known before the job starts, and stays known
    even for a job that is skipped before it reaches the judge — which is why both the
    scoring path and the skip path ask here rather than each deciding for themselves.
    """
    if job.type != "eval_config_eval":
        return None
    if not isinstance(job.item, TaskRun):
        raise ValueError("Calibration items are always TaskRuns")
    return job.item


def _message_field(message: Any, key: str) -> Any:
    """One field of a trace message, or None for anything that isn't a message dict.

    A stored trace is a list of message dicts, but an in-memory one can hold provider
    objects too; reading through this keeps the health check from raising on them.
    """
    return message.get(key) if isinstance(message, dict) else None


def _has_text_content(content: Any) -> bool:
    """Whether a message's content carries any text.

    Content is either a plain string or a list of content parts, and both shapes reach
    here — traces are written by adapters and read back through pydantic.
    """
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            isinstance(part, dict) and bool(str(part.get("text") or "").strip())
            for part in content
        )
    return False


def conversation_health_problem(
    trace: list[ChatCompletionMessageParam] | None,
    required_turns: int,
    *,
    ended_by_su: bool = False,
) -> str | None:
    """Why `trace` is not a complete conversation for an item wanting `required_turns`
    turns, or None when it is complete.

    Structural completeness only, never error-freeness: a conversation whose tool calls
    failed is a legitimate thing to evaluate (judging how an agent handles errors is a
    first-class eval), so error-bearing tool messages say nothing about health here. What
    it does catch is a conversation that stopped short — a drive that died mid-way, or a
    partial record from an older writer — which would otherwise be judged as if the agent
    had simply finished.

    Health is a relationship between a trace and the item asking for it, not a property of
    the trace: the same conversation is complete for a two-turn item and short for a
    three-turn one, so the required count is always passed in by the caller.

    `ended_by_su` says the synthetic user chose to end this conversation, which makes
    `required_turns` a ceiling instead of an exact count: anything from one turn up to the
    ceiling is a finished conversation, and only an empty one or one somehow longer than
    the ceiling is a problem. It is for readers of a STORED trace, where all that survives
    of the drive is a tag — a short conversation the synthetic user chose and one a crash
    left behind look identical on disk, so the trace alone cannot say which it is. A caller
    that still holds the drive knows how many turns actually ran and should pass that count
    instead, keeping the exact check. The flag defaults to False so any caller that knows
    nothing about early ending keeps the strict gate it had before.
    """
    messages = trace or []
    user_turns = sum(
        1 for message in messages if _message_field(message, "role") == "user"
    )
    if ended_by_su:
        if user_turns < 1 or user_turns > required_turns:
            return f"expected 1 to {required_turns} user turns, found {user_turns}"
    elif user_turns != required_turns:
        return f"expected {required_turns} user turns, found {user_turns}"
    if not messages:
        return "the conversation is empty"
    last_role = _message_field(messages[-1], "role")
    if last_role != "assistant":
        return f"the conversation ends with a '{last_role}' message, not an assistant reply"
    if not _has_text_content(_message_field(messages[-1], "content")):
        return "the final assistant message has no text content"
    return None


def _splits_a_turn_into_two_messages(
    properties: KilnAgentRunConfigProperties, task: Task
) -> bool:
    """Whether this run config answers one turn with two user-role messages.

    A chain of thought prompt on a model with no reasoning step of its own is served by
    asking the model to think, then sending a second user message asking for the final
    answer. That extra message is indistinguishable from a real user turn, so a driven
    conversation both miscounts its turns and feeds the injected message back to the
    synthetic user as if the user had written it.

    The strategy is resolved from the same prompt builder and model provider the adapter
    reads, so the answer here is the one the drive would get. Built-in models are read
    straight from the model table rather than through the credentialed provider lookup:
    this preflight runs before any paid call and must give the same answer on a machine
    with no provider keys, and the two fields it needs are static entries in that table.
    Custom, fine-tuned and litellm models are not in the table, so they keep the
    credentialed lookup. A config that cannot be resolved answers False: the drive
    fails on the identical lookup before it spends anything, and one unresolvable
    config must not block the run configs beside it.
    """
    try:
        cot_prompt = prompt_builder_from_id(
            properties.prompt_id, task
        ).chain_of_thought_prompt()
        provider = None
        if cot_prompt:
            provider = built_in_models_from_provider(
                properties.model_provider_name, properties.model_name
            ) or kiln_model_provider_from(
                properties.model_name, properties.model_provider_name
            )
    except Exception as error:
        logger.warning(
            "Could not resolve the chat strategy for model '%s' with prompt '%s' (%s); "
            "leaving its multi-turn compatibility to the per-job checks",
            properties.model_name,
            properties.prompt_id,
            error,
        )
        return False
    return is_two_message_cot_strategy(
        chat_strategy_for_run(
            cot_prompt=cot_prompt,
            tuned_chat_strategy=provider.tuned_chat_strategy if provider else None,
            reasoning_capable=provider.reasoning_capable if provider else False,
        )
    )


def no_golden_set_message(eval: Eval) -> str:
    """Why judge comparison can't run without a golden set. One wording, two raisers.

    Shared with the API layer so the 4xx a user sees and the ValueError a library caller
    sees say the same thing (functional spec 9, architecture 4.3).
    """
    return (
        f"Eval '{eval.id}' has no golden set configured. Comparing judges scores them "
        "against a set of human-rated dataset items, so the eval needs one before a "
        "comparison can run."
    )


class EvalRunner:
    """
    Runs an eval. Async execution is supported to make it faster when using remote/fast model providers.

    Can run an eval in 2 modes:
    1) eval_config_eval: evaluate an eval config (judge quality) using existing dataset
       items. Scoped by the eval's golden filter, so its items are always TaskRuns.
    2) task_run_eval: evaluate a range of task run configs, generating new run output.
       Scoped by the `split` it is given, whose items may come from either store.
       Multi-turn synthetic EvalInputs are re-driven as a full conversation per run
       config using the drive config stamped on each item; stored multi-turn TaskRun
       chains are judged on their stored trace instead.
    """

    def __init__(
        self,
        eval_configs: List[EvalConfig],
        run_configs: List[TaskRunConfig] | None,
        eval_run_type: Literal["eval_config_eval", "task_run_eval"],
        split: ResolvedSplit | None = None,
        save_context: SaveContext | None = None,
    ):
        if len(eval_configs) == 0:
            raise ValueError("Eval runner requires at least one eval config")
        target_eval = eval_configs[0].parent_eval()
        if target_eval is None:
            raise ValueError("Eval config requires a parent eval")
        for eval_config in eval_configs:
            parent_eval = eval_config.parent_eval()
            if parent_eval is None:
                raise ValueError("Eval config requires a parent eval")
            if parent_eval.id != target_eval.id:
                raise ValueError("All eval configs must have the same parent eval")

        target_task = target_eval.parent_task()
        if target_task is None:
            raise ValueError("Eval config requires a (grand)parent task")

        # Both modes settle *what defines* their item scope here rather than at collect
        # time, but to different depths, and the difference is deliberate. task_run_eval
        # takes a ResolvedSplit: the items themselves, fixed now, so a split cannot be
        # accepted and then quietly re-resolved to something else. eval_config_eval takes
        # the golden filter id and still applies it to self.task.runs() at collect time,
        # so items added after construction are picked up — golden is TaskRun-only by
        # definition, so there is no source ambiguity for that to hide.
        self.golden_filter_id: DatasetFilterId | None = None
        if eval_run_type == "task_run_eval":
            if run_configs is None or len(run_configs) == 0:
                raise ValueError("Task run eval requires run configs")
            for run_config in run_configs:
                parent_task = run_config.parent_task()
                if parent_task is None:
                    raise ValueError("All run configs must have a parent task")
                if parent_task.id != target_task.id:
                    raise ValueError(
                        "Run config is not for the same task as the eval configs"
                    )
            if split is None:
                raise ValueError("Task run eval requires a resolved split")
            if split.eval_id != target_eval.id:
                raise ValueError(
                    f"Split '{split.name}' was resolved from eval '{split.eval_id}', not from "
                    f"eval '{target_eval.id}' whose configs are being run"
                )
        else:
            if run_configs is not None:
                raise ValueError("Mode 'eval_config_eval' does not support run configs")
            if split is not None:
                raise ValueError(
                    "Mode 'eval_config_eval' does not support a split: it is scoped by the eval's golden filter"
                )
            if target_eval.eval_configs_filter_id is None:
                raise ValueError(no_golden_set_message(target_eval))
            self.golden_filter_id = target_eval.eval_configs_filter_id

        self.eval_run_type = eval_run_type
        self.eval_configs = eval_configs
        self.run_configs = run_configs
        self.split = split
        self.task = target_task
        self.eval = target_eval
        self._skills: SkillsDict = self._preload_skills()
        self._save_context: SaveContext = save_context or default_save_context
        # Live, not precomputed like `already_run`: a trace persisted by one job has to be
        # visible to the next, whether that next job is running concurrently under a
        # different eval config or is this job's own retry (functional spec 4.2, 4.3).
        self._trace_index = TraceIndex(self.task, vet=self._build_trace_vet())

    def _build_trace_vet(self) -> Callable[[TraceKey, TaskRun], str | None] | None:
        """The completeness check the trace index applies to reuse candidates, or None
        when this run has no multi-turn conversations for it to check. Answers None for a
        usable candidate, or why it is unusable — the index logs the reason alongside the
        file it rejected.

        Required turn counts come from the split's own items, so every candidate is judged
        against the item asking for it. Anything the map doesn't name — single-turn
        generations, and items outside this run's split — is accepted: neither has a turn
        contract to fall short of. A candidate tagged as one the synthetic user chose to
        end is accepted at any length up to its item's turn count; the tag is the only
        record of that choice by the time a later run reads the file.
        """
        if self.split is None:
            return None
        required_turns: Dict[ItemKey, int] = {
            item_key(item): item.data.drive_config.turns
            for item in self.split.items
            if isinstance(item, EvalInput)
            and isinstance(item.data, MultiTurnSyntheticEvalInputData)
            and item.data.drive_config is not None
        }
        if not required_turns:
            return None

        def vet_conversation(key: TraceKey, trace: TaskRun) -> str | None:
            source_type, source_id, _ = key
            turns = required_turns.get((source_type, source_id))
            if turns is None:
                return None
            return conversation_health_problem(
                trace.trace,
                turns,
                ended_by_su=TAG_SU_ENDED_CONVERSATION in trace.tags,
            )

        return vet_conversation

    def collect_tasks(self) -> List[EvalJob]:
        if self.eval_run_type == "eval_config_eval":
            if self.golden_filter_id is None:
                raise ValueError(no_golden_set_message(self.eval))
            return self.collect_tasks_for_eval_config_eval(self.golden_filter_id)
        return self.collect_tasks_for_task_run_eval()

    def collect_tasks_for_eval_config_eval(
        self, eval_configs_filter_id: DatasetFilterId
    ) -> List[EvalJob]:
        """
        Collect all jobs for this run, excluding any that have already been run.

        This variant is used for mode "eval_config_eval", using existing dataset run data (input/output).

        The tasks:
        - should be in the eval config set filter
        - should not have already been run for this eval config + dataset item pair
        """
        filter = dataset_filter_from_id(eval_configs_filter_id)

        # already_run[eval_config_id][dataset_id]
        already_run: Dict[ID_TYPE, Set[ID_TYPE]] = {}
        superseded: Dict[Tuple[ID_TYPE, ID_TYPE], List[EvalRun]] = {}
        for eval_config in self.eval_configs:
            already_run[eval_config.id] = set()
            for run in eval_config.runs(readonly=True):
                # Only calibration records mark a golden item done (or supersede
                # its tombstones): the same eval config also accumulates
                # task_run_eval records, and a golden TaskRun that was scored as
                # a test item must still be calibrated.
                if not run.eval_config_eval:
                    continue
                if self._counts_as_already_run(run, eval_config):
                    already_run[eval_config.id].add(run.dataset_id)
                else:
                    superseded.setdefault((eval_config.id, run.dataset_id), []).append(
                        run
                    )

        return [
            EvalJob(
                item=task_run,
                eval_config=eval_config,
                type="eval_config_eval",
                superseded_tombstones=superseded.get((eval_config.id, task_run.id), []),
            )
            for task_run in self.task.runs(readonly=True)
            if filter(task_run)
            for eval_config in self.eval_configs
            if task_run.id not in already_run[eval_config.id]
        ]

    def collect_tasks_for_task_run_eval(self) -> List[EvalJob]:
        """
        Collect all jobs for this run, excluding any that have already been run.

        This variant is used for mode "task_run_eval", generating new run output using existing dataset item input.

        The tasks:
        - are the items of the split this runner was given, from whichever store backs it
        - should not have already been run for this eval config + run config + dataset item
        """
        if self.split is None:
            raise ValueError("Task run eval requires a resolved split")

        # already_run[eval_config_id][run_config_id][item_key]
        already_run: Dict[ID_TYPE, Dict[ID_TYPE, Set[ItemKey]]] = {}
        # superseded[(eval_config_id, run_config_id, item_key)] -> recoverable-skip
        # tombstones for that item, deleted after a successful re-run persists.
        # Keyed on ItemKey: the split's items may come from either store, and a
        # bare id could collide across stores.
        superseded: Dict[Tuple[ID_TYPE, ID_TYPE, ItemKey], List[EvalRun]] = {}
        for eval_config in self.eval_configs:
            already_run[eval_config.id] = {
                run_config.id: set() for run_config in self.run_configs or []
            }
            for run in eval_config.runs(readonly=True):
                # Scopes the dedupe to the run configs actually being evaluated: an eval
                # config accumulates runs for every run config ever compared against it.
                # The `is not None` is not redundant with the membership test — ID_TYPE is
                # `str | None`, so a run config whose file carries a null id would make
                # None a real key and fold every calibration record into its set.
                if (
                    run.task_run_config_id is None
                    or run.task_run_config_id not in already_run[eval_config.id]
                ):
                    continue
                if self._counts_as_already_run(run, eval_config):
                    already_run[eval_config.id][run.task_run_config_id].add(
                        eval_run_item_key(run)
                    )
                else:
                    superseded.setdefault(
                        (
                            eval_config.id,
                            run.task_run_config_id,
                            eval_run_item_key(run),
                        ),
                        [],
                    ).append(run)

        return [
            EvalJob(
                item=item,
                task_run_config=run_config,
                type="task_run_eval",
                eval_config=eval_config,
                superseded_tombstones=superseded.get(
                    (eval_config.id, run_config.id, (self.split.source, item.id)), []
                ),
            )
            for item in self.split.items
            for eval_config in self.eval_configs
            for run_config in self.run_configs or []
            if (self.split.source, item.id)
            not in already_run[eval_config.id][run_config.id]
        ]

    def _counts_as_already_run(self, run: EvalRun, eval_config: EvalConfig) -> bool:
        """Whether a persisted record makes its item "done" for dedup.

        Most records do — including skips, which are terminal verdicts about
        the input itself (missing_trace, extraction_failed, ...). The
        recoverable skips mark a blocked PRECONDITION instead: once the
        condition is lifted, treating the tombstone as done would freeze the
        item out forever, so it stops counting and the item is collected
        again. A still-blocked item keeps deduping, so re-triggering never
        piles up duplicate tombstones.
        """
        if run.skipped_reason == SkippedReason.missing_drive_config.value:
            # Blocked until the ITEM carries a drive config, which only a
            # replacement item can provide (items are immutable once minted).
            # eval_config_eval has no split and never re-drives, so a stray
            # record of this shape there stays a terminal verdict.
            if self.split is None:
                return True
            for item in self.split.items:
                if (
                    isinstance(item, EvalInput)
                    and item.id == run.eval_input_id
                    and isinstance(item.data, MultiTurnSyntheticEvalInputData)
                ):
                    return item.data.drive_config is None
            return True
        if run.skipped_reason == SkippedReason.incompatible_input_shape.value:
            # Two writers share this reason. The empty-seed guard's records are
            # terminal (items are immutable, a seedless item never gains one).
            # Earlier Kiln versions also skipped EVERY multi-turn synthetic item
            # with it before re-driving existed; a multi-turn item that carries
            # a seed can run now, so its tombstone must not freeze it out.
            if self.split is None:
                return True
            for item in self.split.items:
                if (
                    isinstance(item, EvalInput)
                    and item.id == run.eval_input_id
                    and isinstance(item.data, MultiTurnSyntheticEvalInputData)
                ):
                    return item.data.first_message is None
            return True
        if run.skipped_reason == SkippedReason.type_not_available.value:
            return not v2_eval_type_available(eval_config)
        return True

    def validate_multi_turn_drive_readiness(
        self, check_run_configs: bool = True
    ) -> None:
        """Fail fast on config problems every multi-turn re-drive job would
        hit: no item carrying a synthetic-user drive config, a stamped
        config with an unknown model provider, run configs that aren't
        Kiln agent configs, or run configs that answer in two messages per
        turn. Callers can invoke this before starting a batch
        so the user gets one clear error up front instead of one opaque
        error per job. The per-job checks stay as the backstop for
        standalone runner use.

        A partially stamped split is NOT an error here: the stamped items
        can still run, and each unstamped item records a per-item skip
        naming the fix. Only an entirely unstamped split fails up front,
        because nothing could run and one clear error beats a page of
        identical skips.

        `check_run_configs=False` limits validation to the drive configs —
        for callers running a fleet the user didn't hand-pick, where one
        incompatible run config shouldn't block every other config's jobs.

        Raises ValueError listing every problem; no-op unless re-drive jobs
        can actually occur (task_run_eval over a split containing multi-turn
        synthetic items — stored-TaskRun sources never re-drive, and a
        single-turn split has nothing to drive).
        """
        if self.eval_run_type != "task_run_eval" or self.split is None:
            return
        multi_turn_items = [
            item.data
            for item in self.split.items
            if isinstance(item, EvalInput)
            and isinstance(item.data, MultiTurnSyntheticEvalInputData)
        ]
        if not multi_turn_items:
            return
        problems: list[str] = []
        stamped_configs = [
            data.drive_config
            for data in multi_turn_items
            if data.drive_config is not None
        ]
        if not stamped_configs:
            problems.append(
                "none of this eval's multi-turn items has a synthetic user "
                "configuration (create a new batch to mint items that carry one)"
            )
        for provider in sorted({cfg.model_provider for cfg in stamped_configs}):
            try:
                ModelProviderName(provider)
            except ValueError:
                problems.append(
                    "a multi-turn item's synthetic-user drive config has "
                    f"unknown model provider '{provider}'"
                )
        if check_run_configs:
            for run_config in self.run_configs or []:
                try:
                    agent_properties = as_kiln_agent_run_config(
                        run_config.run_config_properties
                    )
                except ValueError:
                    problems.append(
                        f"run config '{run_config.name}' is not a Kiln agent "
                        "config, so it can't hold a multi-turn conversation"
                    )
                    continue
                if _splits_a_turn_into_two_messages(agent_properties, self.task):
                    problems.append(
                        f"run config '{run_config.name}' uses a chain of thought "
                        "prompt with a model that has no reasoning step of its own, "
                        "so it answers in two messages per turn and can't hold a "
                        "multi-turn conversation (pick a reasoning-capable model, or "
                        "a prompt without thinking instructions)"
                    )
        if problems:
            raise ValueError(
                "Cannot re-drive this eval's multi-turn conversations: "
                + "; ".join(problems)
                + "."
            )

    def _preload_skills(self) -> SkillsDict:
        """Collect all skill IDs from run configs and bulk-load them once."""
        if self.run_configs is None:
            return {}
        merged: SkillsDict = {}
        for rc in self.run_configs:
            skills = load_skills_for_task(self.task, rc.run_config_properties)
            merged.update(skills)
        return merged

    async def run(self, concurrency: int = 25) -> AsyncGenerator[Progress, None]:
        """
        Runs the configured eval run with parallel workers and yields progress updates.
        """
        jobs = self.collect_tasks()

        runner = AsyncJobRunner(
            concurrency=concurrency,
            jobs=jobs,
            run_job_fn=self.run_job,
            max_retries=2,
        )
        async for progress in runner.run():
            yield progress

    async def run_job(self, job: EvalJob) -> bool:
        try:
            if job.eval_config.config_type == EvalConfigType.v2:
                done = await self._run_v2_job(job)
            else:
                done = await self._run_legacy_job(job)
            if done:
                await self._delete_superseded_tombstones(job)
            return done
        except RetryableError as e:
            # Already classified by whoever raised it: this is a deliberate, expected
            # ask for another attempt (an unusable generation the retry replaces), not a
            # failure. Re-raised untouched, and logged without a stacktrace so a
            # self-healing event doesn't read as a crash in the logs.
            logger.warning(f"Retrying eval job for dataset item {job.item.id}: {e}")
            raise
        except Exception as e:
            if is_retryable_error(e):
                logger.error(
                    f"Transient error running eval job for dataset item {job.item.id}: {e}",
                    exc_info=True,
                )
                # KilnRunError's own message is genericized user-facing text; keep
                # the underlying provider detail for the developer-facing error log.
                raise RetryableError(str(unwrap_kiln_run_error(e))) from e
            logger.error(
                f"Error running eval job for dataset item {job.item.id}: {e}",
                exc_info=True,
            )
            raise

    async def _delete_superseded_tombstones(self, job: EvalJob) -> None:
        """Remove the recoverable-skip records this job just replaced. Runs
        only after the replacement record persisted, so an errored job leaves
        the tombstone in place (the item stays re-collectable). Deletion
        failures are logged, never raised: the fresh record is already the
        one read paths should prefer, a leftover duplicate is the lesser
        problem."""
        if not job.superseded_tombstones:
            return
        async with self._save_context():
            for run in job.superseded_tombstones:
                try:
                    run.delete()
                except Exception:
                    logger.warning(
                        f"Failed to delete superseded skip record {run.id} for "
                        f"dataset item {job.item.id}",
                        exc_info=True,
                    )

    async def _run_legacy_job(self, job: EvalJob) -> bool:
        if not isinstance(job.item, TaskRun):
            raise ValueError("Legacy eval jobs require a TaskRun item")

        evaluator = legacy_eval_adapter_from_type(job.eval_config)(
            job.eval_config,
            job.task_run_config.run_config_properties if job.task_run_config else None,
            skills=self._skills,
        )
        if not isinstance(evaluator, BaseEval):
            raise ValueError("Not able to create evaluator from eval config")

        task_output: str | None = None
        reference_answer: str | None = None
        trace: str | None = None
        scores: EvalScores | None = None
        intermediate_outputs: Dict[str, str] | None = None
        task_run_usage: Usage | None = None
        if job.type == "eval_config_eval":
            scores, intermediate_outputs = await evaluator.run_eval(job.item)
            task_output = job.item.output.output
            task_run_usage = job.item.usage
        else:
            (
                result_task_run,
                scores,
                intermediate_outputs,
            ) = await evaluator.run_task_and_eval(job.item)
            task_output = result_task_run.output.output
            task_run_usage = result_task_run.usage

            parent_eval = job.eval_config.parent_eval()
            if (
                parent_eval
                and parent_eval.evaluation_data_type == EvalDataType.full_trace
                and result_task_run.trace
            ):
                trace = serialize_trace(result_task_run.trace)

            if (
                parent_eval
                and parent_eval.evaluation_data_type == EvalDataType.reference_answer
            ):
                reference_answer = job.item.output.output

        async with self._save_context():
            eval_run = EvalRun(
                parent=job.eval_config,
                task_run_config_id=job.task_run_config.id
                if job.task_run_config
                else None,
                dataset_id=job.item.id,
                eval_config_eval=job.type == "eval_config_eval",
                scores=scores,
                input=job.item.input,
                output=task_output,
                reference_answer=reference_answer,
                intermediate_outputs=intermediate_outputs,
                task_run_trace=trace,
                task_run_usage=task_run_usage,
            )
            eval_run.save_to_file()

        return True

    async def _run_v2_job(self, job: EvalJob) -> bool:
        from kiln_ai.adapters.eval.registry import v2_eval_adapter_from_config

        try:
            rc_props = (
                job.task_run_config.run_config_properties
                if job.task_run_config
                else None
            )
            evaluator = v2_eval_adapter_from_config(
                job.eval_config, rc_props, self._skills
            )
        except NotImplementedError:
            return await self._persist_skip(
                job,
                SkippedReason.type_not_available,
                "V2 eval type not yet implemented",
            )

        if (
            isinstance(job.item, EvalInput)
            and isinstance(job.item.data, MultiTurnSyntheticEvalInputData)
            and job.type == "task_run_eval"
        ):
            # Multi-turn synthetic input: re-drive the conversation fresh
            # for this run config, then judge the new trace. The job.type
            # guard is defensive — collect_tasks never pairs eval_config_eval
            # with EvalInput items, because judge calibration is scoped by the
            # golden filter and golden filters only address TaskRuns. There is
            # no runtime handler for a hand-built job of that shape.
            seed = (
                job.item.data.first_message.text if job.item.data.first_message else ""
            )
            return await self._run_v2_multi_turn_synthetic_job(
                job, evaluator, job.item, job.item.data, seed
            )

        if isinstance(job.item, TaskRun) and job.item.parent_task_run_id is not None:
            # Multi-turn chain leaf: a conversation can't be regenerated in
            # a single model call, so both run modes evaluate the stored
            # trace. In task_run_eval mode the scores are therefore a property
            # of the stored conversation, identical across run configs —
            # re-driving per run config needs a synthetic-user seed + persona,
            # which EvalInput-sourced cases carry (branch above) but stored
            # TaskRun chains do not.
            #
            # The leaf is a curated dataset item that already holds its whole
            # conversation, so there is nothing to generate and nothing to
            # index — and no eval_source stamp, which would pull the item off
            # dataset surfaces. The score is a pointer record at the leaf.
            leaf = job.item
            if not leaf.trace:
                # The run exists but recorded no conversation, so the skip
                # still names what it could not score.
                return await self._persist_score(
                    job,
                    scored_run_id=leaf.id,
                    skipped_reason=SkippedReason.missing_trace.value,
                    skipped_detail="Multi-turn task run has no stored trace to evaluate",
                )

            eval_task_input = EvalTaskInput.from_task_run(leaf)
            result = await evaluator.evaluate(eval_task_input)
            return await self._persist_judgment(job, leaf, result)

        trace = await self._resolve_trace(job, evaluator)
        eval_task_input = EvalTaskInput.from_trace(trace, job.item)
        result = await evaluator.evaluate(eval_task_input)
        return await self._persist_judgment(job, trace, result)

    async def _resolve_trace(
        self, job: EvalJob, evaluator: BaseV2EvalBridge
    ) -> TaskRun:
        """The TaskRun this job scores, generating it only if the task has none."""
        golden = _calibration_item(job)
        if golden is not None:
            return golden

        if job.task_run_config is None:
            raise ValueError("A task_run_eval job requires a run config")

        key = trace_key(item_key(job.item), job.task_run_config.id)
        # `TraceIndex` logs both outcomes itself — debug on reuse, info on generation
        # (architecture 8) — so the was_generated bool has no reader here.
        trace, _ = await self._trace_index.get_or_create(
            key, lambda: self._generate_and_persist(job, evaluator, key)
        )
        return trace

    async def _generate_and_persist(
        self, job: EvalJob, evaluator: BaseV2EvalBridge, key: TraceKey
    ) -> TaskRun:
        """Run the task for this job's item, and make the result durable before scoring.

        Stamped from `key` rather than re-derived from the job, so the run files itself
        under exactly the key the index filed it under. A run that disagrees is never
        found again, and the eval regenerates it on every future run.
        """
        source_type, source_id, run_config_id = key
        trace = await evaluator.run_task(job.item, run_config_id=run_config_id)
        if trace.id is None:
            # `run_task` builds its adapter with allow_saving=False, and every adapter
            # clears the id of a run it did not persist (base_adapter.py:346). The runner
            # is the one persisting here, so it mints the id — the same thing
            # data_gen_api.py:474 does with an unsaved adapter run.
            #
            # Deliberately not allow_saving=True instead: the adapter would persist the
            # run before `eval_source` is stamped on it, so a crash in that window would
            # leave an eval trace permanently indistinguishable from a curated dataset
            # row — the contamination Task.runs()' default-exclude exists to prevent.
            trace.id = generate_model_id()
        trace.eval_source = EvalItemSource(source_type=source_type, source_id=source_id)
        async with self._save_context():
            trace.save_to_file()
        return trace

    async def _persist_score(
        self,
        job: EvalJob,
        *,
        scored_run_id: ID_TYPE = None,
        scores: EvalScores | None = None,
        skipped_reason: str | None = None,
        skipped_detail: str | None = None,
        intermediate_outputs: Dict[str, str] | None = None,
        eval_usage: Usage | None = None,
    ) -> bool:
        """Write one V2 score record.

        The single place the item-identity fields are filled in, because `collect_tasks`
        dedupes on exactly those: a skip record and a score record that disagreed would
        be two identities for one job. No inline trace field is ever set — they are
        deprecated, and the trace lives on the TaskRun (functional spec 3.2).
        """
        async with self._save_context():
            EvalRun(
                parent=job.eval_config,
                task_run_config_id=job.task_run_config.id
                if job.task_run_config
                else None,
                dataset_id=job.item.id if isinstance(job.item, TaskRun) else None,
                eval_input_id=job.item.id if isinstance(job.item, EvalInput) else None,
                eval_config_eval=job.type == "eval_config_eval",
                scored_run_id=scored_run_id,
                scores=scores or {},
                skipped_reason=skipped_reason,
                skipped_detail=skipped_detail,
                intermediate_outputs=intermediate_outputs,
                eval_usage=eval_usage,
            ).save_to_file()
        return True

    async def _persist_skip(
        self, job: EvalJob, reason: SkippedReason, detail: str
    ) -> bool:
        """A job skipped before it reached the judge.

        Calibration still names its trace: the golden item *is* what would have been
        scored, it is already on disk, and a skip that happens one step later — inside
        `evaluate()` — records it (functional spec 4.6, row 2). Where the failure
        happened shouldn't change the record's shape, and Phase 5 migrates old
        calibration skips to exactly this.

        A scoring job genuinely has nothing to point at. Generating a trace for a job
        that can never be scored is the spend these early skips exist to avoid.
        """
        golden = _calibration_item(job)
        return await self._persist_score(
            job,
            scored_run_id=golden.id if golden is not None else None,
            skipped_reason=reason.value,
            skipped_detail=detail,
        )

    async def _persist_judgment(
        self,
        job: EvalJob,
        trace: TaskRun,
        result: V2EvalResult,
    ) -> bool:
        """The score for one item, pointing at the trace it was computed over.

        Also the home of a scoring-time skip: the trace exists and this judge could not
        score it, so the record carries both `scored_run_id` and `skipped_reason`
        (functional spec 4.6, row 2).
        """
        return await self._persist_score(
            job,
            scored_run_id=trace.id,
            scores=result.scores,
            skipped_reason=result.skipped_reason.value
            if result.skipped_reason
            else None,
            skipped_detail=result.skipped_detail,
            intermediate_outputs=result.intermediate_outputs,
            eval_usage=result.usage,
        )

    async def _run_v2_multi_turn_synthetic_job(
        self,
        job: EvalJob,
        evaluator: BaseV2EvalBridge,
        eval_input: EvalInput,
        data: MultiTurnSyntheticEvalInputData,
        seed: str,
    ) -> bool:
        """task_run_eval over a multi-turn synthetic input.

        The run config under evaluation drives the agent while the drive
        config stamped on the item plays the synthetic user, so each run
        config gets its own fresh conversation — the property that makes
        run-config comparison meaningful for multi-turn. The conversation
        persists as one standalone TaskRun through the trace index, exactly
        like single-turn generations: durable before scoring, and reusable
        by every other judge over the same item and run config.
        """
        drive_config = data.drive_config
        if drive_config is None:
            return await self._persist_skip(
                job,
                SkippedReason.missing_drive_config,
                "This item has no synthetic user configuration. "
                "Create a new batch to replace it.",
            )

        if not seed:
            return await self._persist_skip(
                job,
                SkippedReason.incompatible_input_shape,
                "Multi-turn synthetic input has no first_message to open the conversation",
            )

        if job.task_run_config is None:
            raise ValueError("Task run eval requires a run config")
        try:
            agent_run_config = as_kiln_agent_run_config(
                job.task_run_config.run_config_properties
            )
        except ValueError as e:
            raise ValueError(
                "Multi-turn re-drive requires a Kiln agent run config; "
                f"run config '{job.task_run_config.name}' is a different type"
            ) from e
        try:
            su_provider = ModelProviderName(drive_config.model_provider)
        except ValueError as e:
            raise ValueError(
                "Invalid synthetic-user model provider on this item's "
                f"drive config: {drive_config.model_provider}"
            ) from e

        key = trace_key(item_key(eval_input), job.task_run_config.id)

        async def drive_and_persist() -> TaskRun:
            # No app-level timeout on the re-drive: it terminates
            # structurally (the turn ceiling, the adapter's tool-call cap,
            # the model client's per-request timeout). This path runs as a
            # background job, so the watchdog log is how a pathologically
            # slow drive gets noticed.
            async with log_if_slow(f"eval re-drive for eval item {eval_input.id}"):
                drive_result = await drive_case_for_eval(
                    seed_prompt=seed,
                    synthetic_user_info=data.synthetic_user_info,
                    target_task=self.task,
                    target_run_config=agent_run_config,
                    su_driver_config=SyntheticUserDriverConfig(
                        model_name=drive_config.model_name,
                        model_provider_name=su_provider,
                    ),
                    turns=drive_config.turns,
                    skills=self._skills,
                )
            turns_run = len(drive_result.chain)
            ended_by_su = drive_result.ended_early(drive_config.turns)
            if turns_run == 1 and drive_config.turns > 1:
                # Not enforced: discarding a paid drive over a rule the synthetic
                # user's prompt states but cannot guarantee would cost more than it
                # saves. Logged because a one-turn conversation is a single-turn
                # trace wearing a multi-turn label.
                logger.warning(
                    "The driven conversation for eval item %s ended on its first "
                    "turn — the synthetic user ended it before the agent had "
                    "answered a second message",
                    eval_input.id,
                )
            leaf_trace = drive_result.chain[-1].trace if drive_result.chain else None
            # Gated on the turns that actually ran, not the item's ceiling: the drive
            # is still in hand, so its own chain is the exact count the leaf's trace
            # has to match. A conversation the synthetic user ended is shorter than
            # the ceiling and still complete, while one whose trace lost turns the
            # chain says it ran is a broken record either way.
            problem = conversation_health_problem(leaf_trace, turns_run)
            if problem is not None:
                # Persisting it would index an incomplete conversation that every judge
                # of this item reuses from then on, so it is a failed generation rather
                # than a cheap result. Retryable, so the job re-drives.
                raise RetryableError(
                    f"The driven conversation for eval item {eval_input.id} is "
                    f"incomplete ({problem}), so it was not saved."
                )
            return await self._persist_driven_conversation(
                key, drive_result, seed=seed, ended_by_su=ended_by_su
            )

        # A raising drive persists nothing, so the job's retry re-drives; a
        # successful one is on disk before the judge sees it, so a scoring
        # failure re-scores without paying for the conversation again.
        trace, _ = await self._trace_index.get_or_create(key, drive_and_persist)

        eval_task_input = EvalTaskInput.from_trace(trace, eval_input)
        result = await evaluator.evaluate(eval_task_input)
        return await self._persist_judgment(job, trace, result)

    async def _persist_driven_conversation(
        self,
        key: TraceKey,
        drive_result: DriveCaseResult,
        *,
        seed: str,
        ended_by_su: bool,
    ) -> TaskRun:
        """One standalone TaskRun holding a freshly driven multi-turn conversation.

        The drive itself touches no disk, and its leaf's trace already holds the
        whole conversation — so a single childless run is the entire durable
        record. Persisting the chain's ancestors would store partial copies of
        the same conversation, and a child run would be invisible to the trace
        index's childless-only seed.

        Stamped from `key` rather than re-derived, for the same reason as
        `_generate_and_persist`: the run must file itself under exactly the key
        the index filed it under, or it is never found again.

        `ended_by_su` is written as a tag rather than kept in memory because the
        completeness gate that needs it runs again on every later reuse of this
        file, when the drive that produced it is long gone.
        """
        source_type, source_id, run_config_id = key
        leaf = drive_result.chain[-1]
        if leaf.output.source is None:
            # Fail before anything is saved: a run without an output source can't
            # carry the run_config_id half of its reuse key, so persisting it would
            # strand a paid conversation on disk that the index can never serve.
            raise ValueError(
                "Driven conversation has no output source; cannot stamp its reuse key"
            )
        # The drive's adapter runs detached from any TaskRunConfig, so the leaf's
        # output source has no run_config_id; the index requires it as half the key.
        output_source = leaf.output.source.model_copy(
            update={"run_config_id": run_config_id}
        )
        run = TaskRun(
            parent=self.task,
            # The seed, not the leaf's own input (the synthetic user's LAST message):
            # this run stands alone, so its input is what opened the conversation.
            input=seed,
            input_source=leaf.input_source,
            output=leaf.output.model_copy(update={"source": output_source}),
            trace=leaf.trace,
            usage=_conversation_usage(drive_result.chain),
            # The synthetic-user driver's spend rides its own field so `usage`
            # stays honestly assistant-only. The whole Usage, not a cost-only
            # stub: the SU is usually a different model on a different provider
            # from the agent, so its tokens are the half that makes the figure
            # reconcilable. `drive_case` already returns None for a drive whose
            # provider reported nothing.
            synthetic_user_usage=drive_result.su_usage,
            cumulative_usage=leaf.cumulative_usage
            or MessageUsage.from_trace(leaf.trace),
            eval_source=EvalItemSource(source_type=source_type, source_id=source_id),
            tags=[TAG_SU_ENDED_CONVERSATION] if ended_by_su else [],
        )
        # The drive runs with allow_saving=False, so nothing touched disk before
        # this fully-stamped run — no crash window in which a driven conversation
        # could persist without eval_source and pass for a curated dataset row.
        async with self._save_context():
            run.save_to_file()
        return run


def _conversation_usage(chain: list[TaskRun]) -> Usage | None:
    """Conversation-total assistant spend for one driven multi-turn conversation.

    The leaf's own `usage` covers only its final turn, so tokens and cost come
    from the cumulative recompute over the full trace, and latency sums each
    turn's in-flight accumulator. None when no turn reported anything — an
    all-None Usage would read as a report where there was none.
    """
    leaf = chain[-1]
    totals = leaf.cumulative_usage or MessageUsage.from_trace(leaf.trace)
    latencies = [
        run.usage.total_llm_latency_ms
        for run in chain
        if run.usage is not None and run.usage.total_llm_latency_ms is not None
    ]
    usage = Usage(
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        total_tokens=totals.total_tokens,
        cost=totals.cost,
        cached_tokens=totals.cached_tokens,
        total_llm_latency_ms=sum(latencies) if latencies else None,
    )
    if all(v is None for v in usage.model_dump().values()):
        return None
    return usage
