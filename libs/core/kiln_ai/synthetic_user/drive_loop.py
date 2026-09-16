"""Single-case drive loop for the multi-turn synthetic-user runner.

`drive_case` alternates the target task adapter with the local
`SyntheticUserDriver`, producing a chain of `TaskRun`s on the target side.
`turns` is a ceiling: the loop stops early when the SU driver replies with
the `EARLY_STOP_SENTINEL` message and nothing else, which is how the SU
says the conversation is over. Any other reply — including one that merely
mentions the sentinel inside a sentence — is an ordinary user message.

Persistence is fully delegated to `target_invoker(...)`: the batch runner's
invoker writes each TaskRun to disk (with `parent_task_run_id` chaining),
while the eval-time invoker keeps the chain in memory. Either way the
returned runs carry `trace` and `cumulative_usage`. The SU side is
in-memory only and produces no TaskRuns.
"""

from dataclasses import dataclass
from typing import Protocol

from kiln_ai.datamodel.task_run import TaskRun
from kiln_ai.datamodel.usage import Usage
from kiln_ai.synthetic_user.driver import SyntheticUserDriver
from kiln_ai.synthetic_user.models import EARLY_STOP_SENTINEL
from kiln_ai.utils.open_ai_types import ChatCompletionMessageParam


class TargetInvoker(Protocol):
    """Callable that invokes the target task for one turn. The runner
    wraps `adapter_for_task(task, run_config).invoke` to satisfy this;
    tests pass in a fake. Keeps the drive loop target-agnostic — it just
    cares about the TaskRun that comes back, persisted or in-memory per
    the invoker.
    """

    async def __call__(
        self,
        *,
        input: str,
        prior_trace: list[ChatCompletionMessageParam] | None,
        parent_task_run: TaskRun | None,
    ) -> TaskRun: ...


class TurnHook(Protocol):
    """Optional callback invoked once per turn.

    The runner uses this to translate per-turn outcomes into BatchEvents
    without coupling drive_case to the event shape. The hook fires after both
    the assistant turn is persisted AND the SU's next message is produced.
    `su_message` is None on two kinds of turn: the case's last allowed turn,
    where no SU call is made at all, and the turn where the SU ended the
    conversation, where the sentinel is swallowed rather than passed on (it
    is a control signal, not something anyone said). `su_cost` is the SU's
    LLM spend for this turn — 0.0 when no SU call was made or the provider
    reported no cost, but the real cost of the stopping call when the SU
    chose to end — surfaced per turn so callers can account spend as it
    happens, not only when the case completes.
    """

    async def __call__(
        self, *, run: TaskRun, su_message: str | None, su_cost: float
    ) -> None: ...


@dataclass(frozen=True)
class DriveCaseResult:
    """Outcome of one drive_case run.

    `chain` is the list of TaskRuns the adapter produced (leaf last);
    whether they were persisted is the target_invoker's choice. It is
    shorter than `turns` when the SU ended the conversation early, which
    `drive_case` treats as a normal completion. There is no stop_reason
    field — `ended_early` reads the two endings off the chain's length — and
    callers that persist a chain record the distinction as a tag so a reader
    of the stored trace can still tell an early ending from a truncated record.

    `su_usage` sums the SU driver model's usage across the case's turns.
    SU turns aren't persisted as TaskRuns, so this is the only place that
    spend surfaces at all — and it carries the driver's tokens, not just
    its cost, because the SU is usually a different model on a different
    provider from the agent under test. A cost alone can be reconciled
    against no invoice and split per model not at all.

    None when no turn reported usage, rather than a zeroed Usage: an
    unmeasured drive must not read as a genuinely free one.
    """

    chain: list[TaskRun]
    su_usage: Usage | None

    def ended_early(self, turns: int) -> bool:
        """Whether the SU ended this conversation before the `turns` ceiling.

        A completed drive appends exactly one TaskRun per turn it ran, and the
        loop's only early exit is the SU's stopping sentinel — every other way
        out raises, so no short chain is ever returned. A chain shorter than the
        ceiling therefore means the SU chose to stop, which is why no
        stop_reason field is needed. Pass the same `turns` the drive ran with;
        a different ceiling answers a different question.
        """
        return len(self.chain) < turns

    @property
    def su_total_cost(self) -> float:
        """The case's SU cost, or 0.0 when nothing was reported.

        Kept as a derived property so the interactive runner's
        `CaseCompletedEvent.total_cost` is unchanged in behaviour — it wants a
        float it can add, and "no usage reported" has always summed as zero
        there.
        """
        if self.su_usage is None or self.su_usage.cost is None:
            return 0.0
        return float(self.su_usage.cost)


async def drive_case(
    *,
    seed_prompt: str,
    target_invoker: TargetInvoker,
    su_driver: SyntheticUserDriver,
    turns: int,
    on_turn: TurnHook | None = None,
) -> DriveCaseResult:
    """Drive one synthetic-user case for up to `turns` turns.

    Args:
        seed_prompt: the opening user-side message sent into the target task.
        target_invoker: how to call the target task for one turn; returns
            the turn's TaskRun (persistence is the invoker's concern).
        su_driver: pre-built SU driver for this case. Caller is responsible
            for construction (so a malformed persona fails at the caller's
            layer, not here).
        turns: ceiling on the assistant turns to produce. The loop stops at
            `turns`, or earlier if the SU ends the conversation.
        on_turn: optional async hook called once per turn (see TurnHook for
            when `su_message` is None). The runner plugs in here to emit
            TurnCompletedEvent.

    Returns:
        DriveCaseResult with the chain of TaskRuns produced (leaf last).
        The chain holds one TaskRun per turn that ran, so it is shorter than
        `turns` when the SU ended the conversation early.
    """
    if turns < 1:
        raise ValueError(f"turns must be >= 1, got {turns}")
    # Assert-loud on missing seed. An empty string would silently flow
    # into the target adapter and surface as a confusing model-side error
    # rather than a clean "the case is malformed" signal.
    if not seed_prompt:
        raise ValueError("seed_prompt must be a non-empty string")

    user_msg: str = seed_prompt
    prev_run: TaskRun | None = None
    prev_trace: list[ChatCompletionMessageParam] | None = None
    chain: list[TaskRun] = []
    # Stays None until some turn actually reports usage, so a drive whose
    # provider reported nothing is distinguishable from a free one.
    su_usage: Usage | None = None

    for turn in range(1, turns + 1):
        new_run = await target_invoker(
            input=user_msg,
            prior_trace=prev_trace,
            parent_task_run=prev_run,
        )
        chain.append(new_run)

        # The SU only speaks when another target turn will consume its
        # reply — an SU call after the final turn would be paid LLM output
        # nobody reads. The SU driver does the role filtering / role swap /
        # invariant checks itself; we pass the cumulative trace as-is.
        su_message: str | None = None
        su_cost = 0.0
        su_ended_conversation = False
        if turn < turns:
            raw_su_message, turn_usage = await su_driver.respond(new_run.trace or [])
            if turn_usage is not None:
                # Usage.__add__ is None-graceful per field, so a turn that
                # reports only cost doesn't erase another turn's token counts.
                su_usage = turn_usage if su_usage is None else su_usage + turn_usage
                if turn_usage.cost is not None:
                    su_cost = float(turn_usage.cost)
            # A reply that is nothing but the sentinel is the SU saying it is
            # done. Whole-message match (after trimming the whitespace models
            # add) so a user who quotes the sentinel mid-sentence still gets a
            # real turn. The sentinel itself is dropped here: it is a control
            # signal, so it must never be spoken to the target or land in a
            # trace an eval later reads.
            if raw_su_message.strip() == EARLY_STOP_SENTINEL:
                su_ended_conversation = True
            else:
                su_message = raw_su_message

        # The hook fires before any early exit, so the completed turn — and
        # the spend of the SU call that ended it — is still reported.
        if on_turn is not None:
            await on_turn(run=new_run, su_message=su_message, su_cost=su_cost)

        if su_ended_conversation:
            break

        if su_message is not None:
            user_msg = su_message
        prev_run = new_run
        prev_trace = new_run.trace

    return DriveCaseResult(chain=chain, su_usage=su_usage)
