"""Unit tests for run_cases_batch.

`adapter_for_task` is monkeypatched to return a fake adapter whose
`.invoke` is an AsyncMock; `SyntheticUserDriver` is replaced with a
patched class returning a mocked instance whose `respond()` yields
canned strings. Tests collect the event stream into a list and inspect
ordering / per-case bookkeeping.
"""

import asyncio
import logging
import re
from typing import Any
from unittest.mock import AsyncMock, Mock

import litellm
import pytest

from kiln_ai.adapters.errors import KilnRunError, format_error_message
from kiln_ai.datamodel.datamodel_enums import (
    ModelProviderName,
    StructuredOutputMode,
)
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties, ToolsRunConfig
from kiln_ai.datamodel.task import Task
from kiln_ai.datamodel.task_run import TaskRun
from kiln_ai.datamodel.usage import Usage
from kiln_ai.synthetic_user import runner as runner_mod
from kiln_ai.synthetic_user.case import SyntheticUserCase
from kiln_ai.synthetic_user.driver import SyntheticUserDriver
from kiln_ai.synthetic_user.models import (
    EARLY_STOP_SENTINEL,
    TAG_SU_ENDED_CONVERSATION,
    SyntheticUserDriverConfig,
)
from kiln_ai.synthetic_user.runner import (
    NUM_CASES_MAX,
    BatchCompletedEvent,
    BatchEvent,
    BatchStartedEvent,
    CaseCompletedEvent,
    CaseFailedEvent,
    TurnCompletedEvent,
    run_cases_batch,
)
from kiln_ai.utils.git_sync_protocols import default_save_context

# ───────────────────────── helpers / fixtures ─────────────────────────


def _case(idx: int = 0) -> SyntheticUserCase:
    return SyntheticUserCase(
        seed_prompt=f"seed-{idx}",
        synthetic_user_info=(
            f"<persona>persona-{idx}</persona>"
            f"<goal>goal-{idx}</goal>"
            f"<behavior_guidance>guidance-{idx}</behavior_guidance>"
        ),
    )


def _su_driver_config() -> SyntheticUserDriverConfig:
    return SyntheticUserDriverConfig(
        model_name="claude_4_5_haiku",
        model_provider_name=ModelProviderName.openrouter,
    )


def _target_run_config() -> KilnAgentRunConfigProperties:
    return KilnAgentRunConfigProperties(
        model_name="gpt_5_5",
        model_provider_name=ModelProviderName.openrouter,
        prompt_id="simple_prompt_builder",
        structured_output_mode=StructuredOutputMode.default,
        tools_config=ToolsRunConfig(tools=[]),
    )


def _fake_run(run_id: str, cost: float = 0.0) -> Mock:
    """Stand-in for a persisted TaskRun. drive_case reads `.trace`; runner
    reads `.id` + `.cumulative_usage.cost`; `_tag_leaf` writes `.tags`
    and calls `.save_to_file()`.
    """
    run = Mock(spec=TaskRun)
    run.id = run_id
    run.trace = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": f"a-{run_id}"},
    ]
    run.cumulative_usage = Mock(cost=cost)
    run.tags = []
    run.save_to_file = Mock()
    return run


def _patch_adapter_for_task(
    monkeypatch: pytest.MonkeyPatch, invoke_side_effect: Any
) -> AsyncMock:
    """Make `adapter_for_task` return an adapter whose .invoke yields the
    given side_effect (list of TaskRuns or a callable).
    """
    adapter = Mock()
    adapter.invoke = AsyncMock(side_effect=invoke_side_effect)
    monkeypatch.setattr(
        runner_mod,
        "adapter_for_task",
        lambda task, run_config, base_adapter_config=None: adapter,
    )
    return adapter.invoke


def _patch_su_driver(
    monkeypatch: pytest.MonkeyPatch, replies_per_case: dict[int, list[str]] | list[str]
) -> None:
    """Replace SyntheticUserDriver with a stub that returns canned strings.

    Pass a list to give every case the same reply schedule; pass a dict
    keyed by case_index for per-case schedules.
    """
    call_counter = {"i": 0}

    def _ctor(info, config):
        idx = call_counter["i"]
        call_counter["i"] += 1
        replies = (
            replies_per_case[idx]
            if isinstance(replies_per_case, dict)
            else list(replies_per_case)
        )
        instance = Mock(spec=SyntheticUserDriver)
        # respond() returns (message, Usage | None). Tests that don't care about
        # the driver's spend hand back None — the shape a provider that reported
        # nothing produces, which the runner totals as zero.
        instance.respond = AsyncMock(side_effect=[(r, None) for r in replies])
        return instance

    monkeypatch.setattr(runner_mod, "SyntheticUserDriver", _ctor)


def _patch_su_driver_factory(monkeypatch: pytest.MonkeyPatch, factory: Any) -> None:
    """For when tests need full control (e.g., raise on construction)."""
    monkeypatch.setattr(runner_mod, "SyntheticUserDriver", factory)


@pytest.fixture
def fake_task() -> Mock:
    return Mock(spec=Task)


async def _collect(gen) -> list[BatchEvent]:
    out: list[BatchEvent] = []
    async for ev in gen:
        out.append(ev)
    return out


# ───────────────────────── input validation ─────────────────────────


def test_num_cases_max_is_pinned() -> None:
    """Callers mirror this cap in their own bounds, so a change here is a
    contract change for all of them — not a local tweak."""
    assert NUM_CASES_MAX == 200


@pytest.mark.asyncio
async def test_empty_cases_raises(fake_task: Mock) -> None:
    with pytest.raises(ValueError, match="cases cannot be empty"):
        async for _ in run_cases_batch(
            cases=[],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
        ):
            pass


@pytest.mark.asyncio
async def test_invalid_turns_raises(fake_task: Mock) -> None:
    with pytest.raises(ValueError, match="turns must be >= 1"):
        async for _ in run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=0,
        ):
            pass


@pytest.mark.asyncio
async def test_invalid_concurrency_raises(fake_task: Mock) -> None:
    with pytest.raises(ValueError, match="concurrency must be >= 1"):
        async for _ in run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            concurrency=0,
        ):
            pass


# ───────────────────────── happy path ─────────────────────────


@pytest.mark.asyncio
async def test_three_cases_produce_full_event_stream(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = [_case(0), _case(1), _case(2)]
    _patch_adapter_for_task(
        monkeypatch,
        [
            _fake_run("r0", cost=0.01),
            _fake_run("r1", cost=0.02),
            _fake_run("r2", cost=0.04),
        ],
    )
    _patch_su_driver(monkeypatch, replies_per_case=["u-next"])

    events = await _collect(
        run_cases_batch(
            cases=cases,
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=1,
            concurrency=4,
            batch_tag="testbatch",
        )
    )

    assert isinstance(events[0], BatchStartedEvent)
    assert events[0].batch_tag == "testbatch"
    assert events[0].num_cases == 3

    assert isinstance(events[-1], BatchCompletedEvent)
    assert events[-1].successful == 3
    assert events[-1].failed == 0
    assert events[-1].batch_tag == "testbatch"
    assert events[-1].total_cost == pytest.approx(0.07)

    turn_events = [e for e in events if isinstance(e, TurnCompletedEvent)]
    case_done = [e for e in events if isinstance(e, CaseCompletedEvent)]
    assert len(turn_events) == 3
    assert len(case_done) == 3
    assert {e.case_index for e in case_done} == {0, 1, 2}
    for ev in case_done:
        assert ev.total_turns == 1


@pytest.mark.asyncio
async def test_total_cost_sums_target_and_su_driver_spend(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CaseCompletedEvent.total_cost = leaf.cumulative_usage.cost + sum of
    SU driver's per-turn cost across the case. BatchCompletedEvent.total_cost
    sums those across successful cases. Locks in the honest-totals contract.
    """
    leaf_a = _fake_run("a-leaf", cost=0.10)
    leaf_b = _fake_run("b-leaf", cost=0.05)
    _patch_adapter_for_task(
        monkeypatch,
        [_fake_run("a-1"), leaf_a, _fake_run("b-1"), leaf_b],
    )

    # At turns=2 each case makes ONE SU call (none after the final turn),
    # at $0.01 → $0.01 SU per case.
    def _ctor(info, config):
        instance = Mock(spec=SyntheticUserDriver)
        instance.respond = AsyncMock(side_effect=[("u2", Usage(cost=0.01))])
        return instance

    monkeypatch.setattr(runner_mod, "SyntheticUserDriver", _ctor)

    events = await _collect(
        run_cases_batch(
            cases=[_case(0), _case(1)],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=2,
            concurrency=1,
        )
    )

    case_a, case_b = (e for e in events if isinstance(e, CaseCompletedEvent))
    assert case_a.total_cost == pytest.approx(0.10 + 0.01)
    assert case_b.total_cost == pytest.approx(0.05 + 0.01)

    batch = next(e for e in events if isinstance(e, BatchCompletedEvent))
    assert batch.total_cost == pytest.approx(0.15 + 0.02)


@pytest.mark.asyncio
async def test_turn_completed_event_carries_su_message_and_trace(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_adapter_for_task(
        monkeypatch, [_fake_run("r0", cost=0.01), _fake_run("r1", cost=0.02)]
    )
    _patch_su_driver(monkeypatch, replies_per_case=["the SU's reply"])

    events = await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=2,
        )
    )

    turns = [e for e in events if isinstance(e, TurnCompletedEvent)]
    assert turns[0].su_next_message == "the SU's reply"
    assert turns[0].cumulative_cost == pytest.approx(0.01)
    # Trace is whatever the fake run carried.
    assert any(m.get("role") == "assistant" for m in turns[0].trace)
    # The final turn has no SU reply — nothing would consume it.
    assert turns[1].su_next_message is None


@pytest.mark.asyncio
async def test_leaf_is_tagged_with_synthetic_user_case_and_batch_tag(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaf = _fake_run("leaf")
    _patch_adapter_for_task(monkeypatch, [leaf])
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=1,
            batch_tag="abc123",
        )
    )

    assert "synthetic_user_case" in leaf.tags
    assert "synthetic_user_batch:abc123" in leaf.tags
    leaf.save_to_file.assert_called_once()


@pytest.mark.asyncio
async def test_leaf_is_tagged_when_the_su_ends_the_conversation_early(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """The tag is the only durable record that this conversation is short because
    the synthetic user finished, not because the drive broke — the eval runner's
    completeness gate reads it back off disk long afterwards.
    """
    leaf = _fake_run("leaf")
    _patch_adapter_for_task(monkeypatch, [leaf])
    _patch_su_driver(monkeypatch, replies_per_case=[EARLY_STOP_SENTINEL])

    with caplog.at_level(logging.WARNING, logger="kiln_ai.synthetic_user.runner"):
        events = await _collect(
            run_cases_batch(
                cases=[_case()],
                target_task=fake_task,
                target_run_config=_target_run_config(),
                su_driver_config=_su_driver_config(),
                turns=3,
                batch_tag="abc123",
            )
        )

    completed = next(e for e in events if isinstance(e, CaseCompletedEvent))
    assert completed.total_turns == 1
    assert TAG_SU_ENDED_CONVERSATION in leaf.tags
    # Ending on the very first turn leaves a single-turn conversation under a
    # multi-turn batch. It is kept, but it is worth a line in the log.
    assert any(
        "ended on its first turn" in record.getMessage() for record in caplog.records
    )


@pytest.mark.asyncio
async def test_a_full_length_drive_is_not_tagged_as_ended_by_the_su(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """A conversation that used its whole turn ceiling must not carry the tag, or
    the tag would say nothing about any conversation."""
    runs = [_fake_run("r1"), _fake_run("leaf")]
    _patch_adapter_for_task(monkeypatch, runs)
    _patch_su_driver(monkeypatch, replies_per_case=["keep going"])

    with caplog.at_level(logging.WARNING, logger="kiln_ai.synthetic_user.runner"):
        await _collect(
            run_cases_batch(
                cases=[_case()],
                target_task=fake_task,
                target_run_config=_target_run_config(),
                su_driver_config=_su_driver_config(),
                turns=2,
                batch_tag="abc123",
            )
        )

    assert TAG_SU_ENDED_CONVERSATION not in runs[-1].tags
    assert caplog.records == []


@pytest.mark.asyncio
async def test_leaf_tagging_is_idempotent_on_pre_tagged_leaf(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-tagging a leaf that already has the runner's tags + an unrelated
    user tag preserves the user tag and dedupes the runner's tags. The
    spec calls this out as the property that makes re-runs safe.
    """
    leaf = _fake_run("leaf")
    leaf.tags = [
        "synthetic_user_case",
        "synthetic_user_batch:abc123",
        "user_kept_this",
    ]
    _patch_adapter_for_task(monkeypatch, [leaf])
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=1,
            batch_tag="abc123",
        )
    )

    assert leaf.tags.count("synthetic_user_case") == 1
    assert leaf.tags.count("synthetic_user_batch:abc123") == 1
    assert "user_kept_this" in leaf.tags
    assert leaf.tags == sorted(leaf.tags)


@pytest.mark.asyncio
async def test_auto_generates_batch_tag_when_not_provided(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_adapter_for_task(monkeypatch, [_fake_run("r")])
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    events = await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=1,
        )
    )

    # Couple the assertion to the public contract — the auto-generated
    # tag must satisfy the batch_tag input regex `[A-Za-z0-9_-]{1,64}` so
    # it can be passed back as `batch_tag` on a subsequent run. Avoids
    # restating the implementation (uuid4().hex[:12]), which would lock
    # us in to a specific length/charset just because that's what we picked.
    started = next(e for e in events if isinstance(e, BatchStartedEvent))
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", started.batch_tag) is not None


@pytest.mark.asyncio
async def test_skills_preloaded_once_and_injected_into_every_adapter(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skills load once per batch and reach each case's adapter via
    AdapterConfig — the adapter raises on skill tools without them.
    """
    sentinel_skills = {"skill_1": Mock()}
    load_calls: list[Any] = []

    def _fake_load(task, run_config):
        load_calls.append((task, run_config))
        return sentinel_skills

    monkeypatch.setattr(runner_mod, "load_skills_for_task", _fake_load)

    adapter = Mock()
    adapter.invoke = AsyncMock(side_effect=[_fake_run("r0"), _fake_run("r1")])
    adapter_configs: list[Any] = []

    def _fake_adapter_for_task(task, run_config, base_adapter_config=None):
        adapter_configs.append(base_adapter_config)
        return adapter

    monkeypatch.setattr(runner_mod, "adapter_for_task", _fake_adapter_for_task)
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    run_config = _target_run_config()
    await _collect(
        run_cases_batch(
            cases=[_case(0), _case(1)],
            target_task=fake_task,
            target_run_config=run_config,
            su_driver_config=_su_driver_config(),
            turns=1,
            task_run_config_id="rc-42",
        )
    )

    assert load_calls == [(fake_task, run_config)]
    assert len(adapter_configs) == 2
    assert all(cfg.skills is sentinel_skills for cfg in adapter_configs)
    # The saved config's id rides through to each adapter so persisted runs
    # attribute back to it, matching a manual run of that config.
    assert all(cfg.task_run_config_id == "rc-42" for cfg in adapter_configs)


# ───────────────────────── per-case failure isolation ─────────────────────────


@pytest.mark.asyncio
async def test_malformed_blob_surfaces_as_case_failed(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner parses each case's blob at its wire boundary — a bad
    blob fails that case alone; the others run normally.
    """

    def _ctor(info, config):
        instance = Mock(spec=SyntheticUserDriver)
        instance.respond = AsyncMock(return_value=("ok", None))
        return instance

    _patch_su_driver_factory(monkeypatch, _ctor)
    _patch_adapter_for_task(monkeypatch, [_fake_run("r0"), _fake_run("r2")])

    bad_case = SyntheticUserCase(
        seed_prompt="seed-1", synthetic_user_info="no tags here at all"
    )
    events = await _collect(
        run_cases_batch(
            cases=[_case(0), bad_case, _case(2)],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=1,
            concurrency=1,  # serialize so cases run in order
        )
    )

    case_done = [e for e in events if isinstance(e, CaseCompletedEvent)]
    case_failed = [e for e in events if isinstance(e, CaseFailedEvent)]
    assert {e.case_index for e in case_done} == {0, 2}
    assert {e.case_index for e in case_failed} == {1}
    assert case_failed[0].error_code == "bad_synthetic_user_info"


@pytest.mark.asyncio
async def test_target_invoke_failure_surfaces_as_case_failed(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If adapter.invoke raises, the case is marked failed — but other
    in-flight cases keep running.
    """
    _patch_adapter_for_task(monkeypatch, RuntimeError("kaboom"))
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    events = await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=1,
        )
    )

    failed = next(e for e in events if isinstance(e, CaseFailedEvent))
    assert failed.error_code == "unexpected_error"
    assert "RuntimeError" in failed.message
    assert "kaboom" in failed.message


@pytest.mark.asyncio
async def test_terminal_provider_error_names_its_class_on_the_event(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider error the shared classifier calls permanent carries its
    exception class on the event, so failures can be counted by kind without
    parsing the message."""
    _patch_adapter_for_task(
        monkeypatch,
        litellm.BadRequestError(
            message="max_tokens too large for this model",
            model="gpt_5_5",
            llm_provider="openrouter",
        ),
    )
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    events = await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=1,
        )
    )

    failed = next(e for e in events if isinstance(e, CaseFailedEvent))
    assert failed.error_code == "unexpected_error"
    assert failed.error_type == "BadRequestError"


@pytest.mark.asyncio
async def test_tag_leaf_failure_surfaces_as_case_failed(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If save_to_file fails (e.g., disk full, validator rejects the tag),
    the case becomes case_failed instead of silently vanishing.
    """
    bad_leaf = _fake_run("bad-leaf")
    bad_leaf.save_to_file = Mock(side_effect=OSError("disk full"))
    _patch_adapter_for_task(monkeypatch, [bad_leaf])
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    events = await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=1,
        )
    )

    failed = next(e for e in events if isinstance(e, CaseFailedEvent))
    assert failed.error_code == "unexpected_error"
    assert "disk full" in failed.message
    # The fully-driven chain never got its batch tag, so nothing could ever
    # find it again — the failure arm must remove it from disk.
    bad_leaf.delete.assert_called_once()


@pytest.mark.asyncio
async def test_failed_case_deletes_partial_chain(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-drive failure deletes the turns already persisted — an untagged
    partial chain would otherwise be invisible to eval loaders AND to
    delete-on-redrive, accumulating forever."""
    turn_one = _fake_run("turn-1")
    _patch_adapter_for_task(monkeypatch, [turn_one, RuntimeError("kaboom")])
    _patch_su_driver(monkeypatch, replies_per_case=["x", "y"])

    events = await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=2,
        )
    )

    failed = next(e for e in events if isinstance(e, CaseFailedEvent))
    assert failed.error_code == "unexpected_error"
    turn_one.delete.assert_called_once()


@pytest.mark.asyncio
async def test_su_failure_deletes_chain_including_just_persisted_run(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure in the SU half of turn N strikes AFTER run N persisted but
    BEFORE the turn hook fired — cleanup must delete runs 1..N, not just
    the turns that fully completed."""
    run_one = _fake_run("turn-1")
    run_two = _fake_run("turn-2")
    _patch_adapter_for_task(monkeypatch, [run_one, run_two])

    def _ctor(info, config):
        instance = Mock(spec=SyntheticUserDriver)
        # Turn 1's SU reply succeeds; turn 2's SU call dies mid-case.
        instance.respond = AsyncMock(
            side_effect=[("u2", None), ValueError("su blew up")]
        )
        return instance

    monkeypatch.setattr(runner_mod, "SyntheticUserDriver", _ctor)

    events = await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=3,
        )
    )

    failed = next(e for e in events if isinstance(e, CaseFailedEvent))
    assert failed.error_code == "unexpected_error"
    run_one.delete.assert_called_once()
    run_two.delete.assert_called_once()


@pytest.mark.asyncio
async def test_case_timeout_fails_case_and_deletes_partial_chain(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drive that exceeds a caller-set case_timeout_seconds fails with
    `case_timeout`, its already-persisted turns are removed, and the batch
    still completes — a caller's budget must free the slot it bounds."""
    turn_one = _fake_run("turn-1")
    calls = {"n": 0}

    async def invoke(**_kwargs: Any) -> Mock:
        calls["n"] += 1
        if calls["n"] == 1:
            return turn_one
        # Second turn hangs until cancelled by the timeout.
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    _patch_adapter_for_task(monkeypatch, invoke)
    _patch_su_driver(monkeypatch, replies_per_case=["x", "y"])

    events = await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=2,
            case_timeout_seconds=0.05,
        )
    )

    failed = next(e for e in events if isinstance(e, CaseFailedEvent))
    assert failed.error_code == "case_timeout"
    # A deterministic failure leaves error_type None even though a TimeoutError
    # triggered it: error_code already names the budget firing.
    assert failed.error_type is None
    turn_one.delete.assert_called_once()
    completed = next(e for e in events if isinstance(e, BatchCompletedEvent))
    assert completed.failed == 1
    assert completed.successful == 0
    # Timeouts are never retried — a retry would pin a worker for another
    # full drive budget. Two invokes = one attempt (turn 1 + the hang).
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_no_case_timeout_by_default(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no caller-set case_timeout_seconds the drive is unbounded —
    wait_for gets timeout=None. No budget may be derived from turn count:
    a slow-but-healthy case must never be killed by the runner."""
    captured_timeouts: list[float | None] = []
    real_wait_for = asyncio.wait_for

    async def spy_wait_for(awaitable: Any, timeout: float | None = None) -> Any:
        captured_timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", spy_wait_for)
    _patch_adapter_for_task(monkeypatch, [_fake_run("turn-1")])
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    events = await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=1,
        )
    )

    completed = next(e for e in events if isinstance(e, BatchCompletedEvent))
    assert completed.successful == 1
    # The drive's wait_for call is the only one made with timeout=None
    # (the job runner's internal polling uses a small float).
    assert captured_timeouts.count(None) == 1


@pytest.mark.asyncio
async def test_raw_provider_timeout_is_not_a_case_timeout(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no case budget set, an asyncio.TimeoutError surfacing raw from a
    provider call is classified as unexpected_error with a named message —
    never case_timeout (no budget exists to have fired), and never a crash
    from formatting a None budget."""
    turn_one = _fake_run("turn-1")
    calls = {"n": 0}

    async def invoke(**_kwargs: Any) -> Mock:
        calls["n"] += 1
        if calls["n"] == 1:
            return turn_one
        raise asyncio.TimeoutError()

    _patch_adapter_for_task(monkeypatch, invoke)
    _patch_su_driver(monkeypatch, replies_per_case=["x", "y"])

    events = await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=2,
        )
    )

    failed = next(e for e in events if isinstance(e, CaseFailedEvent))
    assert failed.error_code == "unexpected_error"
    # A bare TimeoutError has no str(); the message must still say what
    # happened instead of trailing off after the type name.
    assert failed.message == "TimeoutError: The model provider request timed out."
    # The same failure, structured: consumers group by class name instead of
    # parsing the prose above.
    assert failed.error_type == "TimeoutError"
    turn_one.delete.assert_called_once()
    # Raw timeouts are not classified transient: one attempt only.
    assert calls["n"] == 2


# ───────────────────────── retry behavior ─────────────────────────


@pytest.mark.asyncio
async def test_transient_error_retries_and_succeeds(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drive failing with a transient (classifier-approved) error is
    re-attempted by the job runner and can still complete — no case_failed."""
    monkeypatch.setattr(runner_mod, "DRIVE_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(
        runner_mod, "is_retryable_error", lambda e: isinstance(e, RuntimeError)
    )
    run = _fake_run("r-retry")
    invoke = _patch_adapter_for_task(monkeypatch, [RuntimeError("flaky 502"), run])
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    events = await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=1,
        )
    )

    assert not [e for e in events if isinstance(e, CaseFailedEvent)]
    completed = next(e for e in events if isinstance(e, CaseCompletedEvent))
    assert completed.case_index == 0
    assert invoke.call_count == 2
    batch = next(e for e in events if isinstance(e, BatchCompletedEvent))
    assert batch.successful == 1
    assert batch.failed == 0


@pytest.mark.asyncio
async def test_transient_error_exhausts_retries_then_fails_once(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retries exhausted → exactly ONE case_failed (never one per attempt),
    with the underlying error in the message and named on the event.

    Runs a real transient provider error through the shared classifier, wrapped
    the way the model adapter delivers it: both the KilnRunError wrapper and
    the retryable wrapper hide the original, so message and error_type must
    name the innermost provider error."""
    monkeypatch.setattr(runner_mod, "DRIVE_RETRY_DELAY_SECONDS", 0)

    def _raise_as_the_adapter_does(*args: Any, **kwargs: Any) -> None:
        original = litellm.RateLimitError(
            message="upstream rate limit",
            model="gpt_5_5",
            llm_provider="openrouter",
        )
        raise KilnRunError(
            message=format_error_message(original),
            partial_trace=None,
            original=original,
        ) from original

    invoke = _patch_adapter_for_task(monkeypatch, _raise_as_the_adapter_does)
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    events = await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=1,
        )
    )

    failed = [e for e in events if isinstance(e, CaseFailedEvent)]
    assert len(failed) == 1
    assert failed[0].error_code == "unexpected_error"
    assert "upstream rate limit" in failed[0].message
    assert failed[0].error_type == "RateLimitError"
    # One invoke per attempt at turns=1: the first try plus the retries.
    assert invoke.call_count == 1 + runner_mod.DRIVE_MAX_RETRIES


@pytest.mark.asyncio
async def test_each_failed_attempt_cleans_its_partial_chain(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """turns=2: every attempt persists turn 1 then dies on turn 2 — each
    attempt's orphan chain must be deleted, and turn events restart at 1
    per attempt (turn_index is per-attempt, not cumulative)."""
    monkeypatch.setattr(runner_mod, "DRIVE_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(
        runner_mod, "is_retryable_error", lambda e: isinstance(e, RuntimeError)
    )
    runs = [_fake_run(f"turn1-attempt{i}") for i in range(3)]
    _patch_adapter_for_task(
        monkeypatch,
        [
            runs[0],
            RuntimeError("flaky"),
            runs[1],
            RuntimeError("flaky"),
            runs[2],
            RuntimeError("flaky"),
        ],
    )
    _patch_su_driver(monkeypatch, replies_per_case=["x", "y"])

    events = await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=2,
        )
    )

    assert len([e for e in events if isinstance(e, CaseFailedEvent)]) == 1
    for run in runs:
        run.delete.assert_called_once()
    turn_indexes = [e.turn_index for e in events if isinstance(e, TurnCompletedEvent)]
    assert turn_indexes == [1, 1, 1]


@pytest.mark.asyncio
async def test_retried_case_batch_total_includes_both_attempts_costs(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retried case's discarded first attempt still billed the provider.
    total_cost stays per-conversation; discarded_attempts_cost carries the
    deleted attempt's spend; the batch total covers both."""
    monkeypatch.setattr(runner_mod, "DRIVE_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(
        runner_mod, "is_retryable_error", lambda e: isinstance(e, RuntimeError)
    )
    # Attempt 1: turn 1 persists ($0.03 target), turn 2 dies transiently.
    # Attempt 2: full chain; leaf's cumulative target cost $0.08.
    _patch_adapter_for_task(
        monkeypatch,
        [
            _fake_run("a1-t1", cost=0.03),
            RuntimeError("flaky 502"),
            _fake_run("a2-t1", cost=0.03),
            _fake_run("a2-leaf", cost=0.08),
        ],
    )

    # One SU call per attempt (turns=2), at $0.01.
    def _ctor(info, config):
        instance = Mock(spec=SyntheticUserDriver)
        instance.respond = AsyncMock(side_effect=[("u2", Usage(cost=0.01))])
        return instance

    monkeypatch.setattr(runner_mod, "SyntheticUserDriver", _ctor)

    events = await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=2,
        )
    )

    completed = next(e for e in events if isinstance(e, CaseCompletedEvent))
    # Surviving conversation: leaf target $0.08 + SU $0.01.
    assert completed.total_cost == pytest.approx(0.09)
    # Attempt 1's real spend: persisted turn $0.03 + SU $0.01.
    assert completed.discarded_attempts_cost == pytest.approx(0.04)
    batch = next(e for e in events if isinstance(e, BatchCompletedEvent))
    assert batch.total_cost == pytest.approx(0.13)


@pytest.mark.asyncio
async def test_dead_case_failed_event_reports_all_attempts_spend(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A case that exhausts retries billed on every attempt — case_failed
    carries the summed spend and the batch total includes it, even though
    nothing survives on disk."""
    monkeypatch.setattr(runner_mod, "DRIVE_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(
        runner_mod, "is_retryable_error", lambda e: isinstance(e, RuntimeError)
    )
    runs = [_fake_run(f"t1-a{i}", cost=0.02) for i in range(3)]
    _patch_adapter_for_task(
        monkeypatch,
        [
            runs[0],
            RuntimeError("flaky"),
            runs[1],
            RuntimeError("flaky"),
            runs[2],
            RuntimeError("flaky"),
        ],
    )

    def _ctor(info, config):
        instance = Mock(spec=SyntheticUserDriver)
        instance.respond = AsyncMock(side_effect=[("u2", Usage(cost=0.01))])
        return instance

    monkeypatch.setattr(runner_mod, "SyntheticUserDriver", _ctor)

    events = await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=2,
        )
    )

    failed = next(e for e in events if isinstance(e, CaseFailedEvent))
    # Three attempts, each $0.02 target + $0.01 SU before dying.
    assert failed.total_cost == pytest.approx(3 * 0.03)
    batch = next(e for e in events if isinstance(e, BatchCompletedEvent))
    assert batch.successful == 0
    assert batch.total_cost == pytest.approx(3 * 0.03)


@pytest.mark.asyncio
async def test_invalid_case_timeout_raises(fake_task: Mock) -> None:
    with pytest.raises(ValueError, match="case_timeout_seconds"):
        await _collect(
            run_cases_batch(
                cases=[_case()],
                target_task=fake_task,
                target_run_config=_target_run_config(),
                su_driver_config=_su_driver_config(),
                case_timeout_seconds=0,
            )
        )


# ───────────────────────── concurrency ─────────────────────────


@pytest.mark.asyncio
async def test_concurrency_semaphore_caps_in_flight_cases(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With concurrency=2 and 5 cases, max-in-flight should be exactly 2."""
    in_flight = 0
    max_seen = 0
    fan_out_event = asyncio.Event()
    lock = asyncio.Lock()

    async def slow_invoke(**_kwargs: Any) -> Mock:
        nonlocal in_flight, max_seen
        async with lock:
            in_flight += 1
            max_seen = max(max_seen, in_flight)
            if in_flight >= 2:
                fan_out_event.set()
        # Block until fan-out has been observed to actually happen.
        try:
            await asyncio.wait_for(fan_out_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        async with lock:
            in_flight -= 1
        return _fake_run("r")

    _patch_adapter_for_task(monkeypatch, slow_invoke)
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    await _collect(
        run_cases_batch(
            cases=[_case(i) for i in range(5)],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=1,
            concurrency=2,
        )
    )

    # Exactly 2 — not <=2 — so a regression that serialized everything
    # wouldn't pass vacuously.
    assert max_seen == 2


# ───────────────────────── input_source attribution ─────────────────────


@pytest.mark.asyncio
async def test_root_input_source_carries_decomposed_case_context(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First adapter.invoke call gets an input_source with the decomposed
    SU case context (persona/goal/behavior_guidance/seed_prompt) rather
    than the opaque blob. Lets dataset readers inspect attribution
    without re-parsing.
    """
    captured: list[dict[str, Any]] = []

    async def _capture(**kwargs: Any) -> Mock:
        captured.append(kwargs)
        return _fake_run(f"r-{len(captured)}")

    _patch_adapter_for_task(monkeypatch, _capture)
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    case = _case(0)
    await _collect(
        run_cases_batch(
            cases=[case],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=1,
            batch_tag="rb1",
        )
    )

    root_invoke = captured[0]
    props = root_invoke["input_source"].properties
    assert props["adapter_name"] == "kiln_synthetic_user_runner"
    assert props["model_name"] == "claude_4_5_haiku"
    assert props["model_provider"] == "openrouter"
    assert props["batch_tag"] == "rb1"
    assert props["turn_index"] == 1
    # Decomposed — no opaque blob persisted.
    assert "synthetic_user_info" not in props
    assert props["persona"] == "persona-0"
    assert props["goal"] == "goal-0"
    assert props["behavior_guidance"] == "guidance-0"
    assert props["seed_prompt"] == case.seed_prompt


@pytest.mark.asyncio
async def test_root_input_source_omits_behavior_guidance_when_absent(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`behavior_guidance` is the only optional SU info field — when the
    blob omits the tag (or has empty content), the property must be
    absent rather than empty-stringed. The DataSource validator rejects
    empty strings, so the filter in `_build_input_source` would otherwise
    save us here, but we want the wire shape predictable either way.
    """
    captured: list[dict[str, Any]] = []

    async def _capture(**kwargs: Any) -> Mock:
        captured.append(kwargs)
        return _fake_run(f"r-{len(captured)}")

    _patch_adapter_for_task(monkeypatch, _capture)
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    case_no_guidance = SyntheticUserCase(
        seed_prompt="hello",
        synthetic_user_info="<persona>p</persona><goal>g</goal>",
    )
    await _collect(
        run_cases_batch(
            cases=[case_no_guidance],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=1,
        )
    )

    props = captured[0]["input_source"].properties
    assert props["persona"] == "p"
    assert props["goal"] == "g"
    assert "behavior_guidance" not in props


@pytest.mark.asyncio
async def test_non_root_input_source_is_slim(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second turn's input_source carries only batch_tag + turn_index;
    the case context lives on the root.
    """
    captured: list[dict[str, Any]] = []

    async def _capture(**kwargs: Any) -> Mock:
        captured.append(kwargs)
        return _fake_run(f"r-{len(captured)}")

    _patch_adapter_for_task(monkeypatch, _capture)
    _patch_su_driver(monkeypatch, replies_per_case=["u2", "u3"])

    await _collect(
        run_cases_batch(
            cases=[_case()],
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=2,
            batch_tag="rb2",
        )
    )

    assert len(captured) == 2
    second_props = captured[1]["input_source"].properties
    assert "synthetic_user_info" not in second_props
    assert "seed_prompt" not in second_props
    assert second_props["batch_tag"] == "rb2"
    assert second_props["turn_index"] == 2


# ───────────────────────── consumer cancellation ─────────────────────────


@pytest.mark.asyncio
async def test_consumer_cancellation_cancels_in_flight_case_tasks(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the consumer breaks off mid-stream, the closer cancels in-flight
    case tasks rather than letting them run to completion writing to a
    dead queue.
    """
    case_started = asyncio.Event()
    case_can_proceed = asyncio.Event()
    saw_cancel = {"cancelled": False}

    async def _slow_invoke(**_kwargs: Any) -> Mock:
        case_started.set()
        try:
            await case_can_proceed.wait()
        except asyncio.CancelledError:
            saw_cancel["cancelled"] = True
            raise
        return _fake_run("r")

    _patch_adapter_for_task(monkeypatch, _slow_invoke)
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    gen = run_cases_batch(
        cases=[_case()],
        target_task=fake_task,
        target_run_config=_target_run_config(),
        su_driver_config=_su_driver_config(),
        turns=1,
    )

    # Drain BatchStartedEvent, then break.
    started = await gen.__anext__()
    assert isinstance(started, BatchStartedEvent)

    # Wait until the case task is actually running.
    await asyncio.wait_for(case_started.wait(), timeout=1.0)

    # Close the generator — simulates consumer disconnect. This should
    # cancel the in-flight case task via the finally block.
    await gen.aclose()

    # Give the cancellation a beat to propagate.
    await asyncio.sleep(0)

    assert saw_cancel["cancelled"] is True


@pytest.mark.asyncio
async def test_cancelled_case_deletes_partial_chain_and_reraises(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopping a batch cancels in-flight cases mid-drive; a cancelled case
    must remove its already-persisted turns AND re-raise the CancelledError —
    swallowing it would break cooperative teardown."""
    run_one = _fake_run("turn-1")
    reached_turn_two = asyncio.Event()
    calls = {"n": 0}

    async def invoke(**_kwargs: Any) -> Mock:
        calls["n"] += 1
        if calls["n"] == 1:
            return run_one
        reached_turn_two.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    _patch_adapter_for_task(monkeypatch, invoke)
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(
        runner_mod._drive_one_case_and_emit(
            case_index=0,
            case=_case(),
            target_task=fake_task,
            target_run_config=_target_run_config(),
            su_driver_config=_su_driver_config(),
            turns=2,
            batch_tag="tb",
            queue=queue,
            save_ctx=default_save_context,
            skills={},
            task_run_config_id=None,
            case_timeout_seconds=60.0,
            failed_attempt_spend={},
        )
    )
    await asyncio.wait_for(reached_turn_two.wait(), timeout=1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    run_one.delete.assert_called_once()


@pytest.mark.asyncio
async def test_consumer_disconnect_cleans_cancelled_cases_partial_chains(
    fake_task: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end teardown: closing the batch generator mid-case must leave
    no persisted turns behind — the cancelled case's cleanup runs (shielded)
    before the generator's teardown completes."""
    run_one = _fake_run("turn-1")
    reached_turn_two = asyncio.Event()
    calls = {"n": 0}

    async def invoke(**_kwargs: Any) -> Mock:
        calls["n"] += 1
        if calls["n"] == 1:
            return run_one
        reached_turn_two.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    _patch_adapter_for_task(monkeypatch, invoke)
    _patch_su_driver(monkeypatch, replies_per_case=["x"])

    gen = run_cases_batch(
        cases=[_case()],
        target_task=fake_task,
        target_run_config=_target_run_config(),
        su_driver_config=_su_driver_config(),
        turns=2,
    )
    started = await gen.__anext__()
    assert isinstance(started, BatchStartedEvent)
    await asyncio.wait_for(reached_turn_two.wait(), timeout=1.0)

    # Simulates the consumer disconnect / stop button.
    await gen.aclose()

    run_one.delete.assert_called_once()
