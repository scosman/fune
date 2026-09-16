"""Unit tests for SyntheticUserDriver.

`adapter_for_task` is monkeypatched to return a fake adapter whose
`invoke_returning_run_output` is an AsyncMock — no real LLM calls. Tests
focus on the driver's per-turn responsibilities: input shaping
(filtering, role swap, system prompt prepend) and output validation.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from kiln_ai.adapters.run_output import RunOutput
from kiln_ai.datamodel.datamodel_enums import ModelProviderName
from kiln_ai.datamodel.task_run import TaskRun
from kiln_ai.datamodel.usage import Usage
from kiln_ai.synthetic_user import driver as driver_mod
from kiln_ai.synthetic_user.driver import SyntheticUserDriver
from kiln_ai.synthetic_user.models import SyntheticUserDriverConfig
from kiln_ai.synthetic_user.parser import parse_synthetic_user_info
from kiln_ai.utils.open_ai_types import ChatCompletionMessageParam

_BLOB = (
    "<persona>A 30-something professional</persona>"
    "<goal>Find the 2016 RRSP limit</goal>"
    "<behavior_guidance>Press for specifics</behavior_guidance>"
)

_INFO = parse_synthetic_user_info(_BLOB)

_DRIVER_CONFIG = SyntheticUserDriverConfig(
    model_name="claude_4_5_haiku",
    model_provider_name=ModelProviderName.openrouter,
)


def _fake_run_output(text: str | dict = "hi from the SU") -> RunOutput:
    return RunOutput(output=text, intermediate_outputs=None)


def _patch_adapter(
    monkeypatch: pytest.MonkeyPatch,
    return_value: RunOutput,
    cost: float | None = None,
    usage: Usage | None = None,
) -> Mock:
    """Replace adapter_for_task with a stub returning a mock adapter whose
    invoke_returning_run_output yields (Mock(spec=TaskRun), return_value).
    Returns the mock adapter so tests can assert call args.

    Pass `usage` to set the in-memory TaskRun's `.usage` outright, or `cost`
    for the cost-only shorthand. With neither, `.usage` is None and
    `respond()` reports None.
    """
    task_run = Mock(spec=TaskRun)
    if usage is not None:
        task_run.usage = usage
    elif cost is not None:
        task_run.usage = Usage(
            input_tokens=0, output_tokens=0, total_tokens=0, cost=cost
        )
    else:
        task_run.usage = None
    adapter = Mock()
    adapter.invoke_returning_run_output = AsyncMock(
        return_value=(task_run, return_value)
    )
    monkeypatch.setattr(
        driver_mod, "adapter_for_task", lambda task, run_config: adapter
    )
    return adapter


# ───────────────────────── construction ─────────────────────────


def test_construction_stores_typed_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_adapter(monkeypatch, _fake_run_output())
    drv = SyntheticUserDriver(_INFO, _DRIVER_CONFIG)
    assert drv._info.persona == "A 30-something professional"
    assert drv._info.goal == "Find the 2016 RRSP limit"
    assert drv._info.behavior_guidance == "Press for specifics"


def test_construction_renders_system_prompt_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_adapter(monkeypatch, _fake_run_output())
    drv = SyntheticUserDriver(_INFO, _DRIVER_CONFIG)
    # System prompt is rendered on construction and reused.
    assert "A 30-something professional" in drv._system_prompt
    assert "Find the 2016 RRSP limit" in drv._system_prompt
    assert "Press for specifics" in drv._system_prompt


# ───────────────────────── respond — happy path ─────────────────────────


@pytest.mark.asyncio
async def test_respond_returns_adapter_output_and_zero_cost_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider that reports nothing yields None, not a zeroed Usage.

    The distinction is the point of returning Usage at all: a zeroed record
    would read downstream as a genuinely free call rather than an unmeasured
    one, and the drive's total would claim a precision it doesn't have.
    """
    adapter = _patch_adapter(monkeypatch, _fake_run_output("the SU's reply"))
    drv = SyntheticUserDriver(_INFO, _DRIVER_CONFIG)
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    message, usage = await drv.respond(conversation)

    assert message == "the SU's reply"
    assert usage is None
    adapter.invoke_returning_run_output.assert_awaited_once()


@pytest.mark.asyncio
async def test_respond_returns_usage_from_task_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole Usage comes back, tokens included — not just the cost.

    The SU's TaskRun is never persisted, so anything dropped here is gone for
    good: these token counts exist nowhere else on disk.
    """
    _patch_adapter(
        monkeypatch,
        _fake_run_output("hi"),
        usage=Usage(
            input_tokens=3548, output_tokens=61, total_tokens=3609, cost=0.0123
        ),
    )
    drv = SyntheticUserDriver(_INFO, _DRIVER_CONFIG)
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    _, usage = await drv.respond(conversation)

    assert usage is not None
    assert usage.input_tokens == 3548
    assert usage.output_tokens == 61
    assert usage.total_tokens == 3609
    assert usage.cost == pytest.approx(0.0123)


@pytest.mark.asyncio
async def test_respond_role_swaps_and_prepends_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end shape check: input is the role-swapped LAST user turn;
    prior_trace is [system, ...role-swapped earlier turns].
    """
    adapter = _patch_adapter(monkeypatch, _fake_run_output("ok"))
    drv = SyntheticUserDriver(_INFO, _DRIVER_CONFIG)
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "seed-user-1"},
        {"role": "assistant", "content": "target-reply-1"},
        {"role": "user", "content": "seed-user-2"},
        {"role": "assistant", "content": "target-reply-2"},
    ]

    await drv.respond(conversation)

    call = adapter.invoke_returning_run_output.await_args
    # `input` is the role-swapped LAST eval-frame turn — was "assistant" with
    # content "target-reply-2", becomes "user" content "target-reply-2".
    positional = call.args
    keyword = call.kwargs
    # invoke_returning_run_output(user_input, prior_trace=...)
    assert positional[0] == "target-reply-2"

    prior_trace = keyword["prior_trace"]
    # prior_trace = [system, role-swapped turns 0..-2]
    assert prior_trace[0]["role"] == "system"
    assert "A 30-something professional" in prior_trace[0]["content"]
    # The three earlier turns, all role-swapped.
    assert [m["role"] for m in prior_trace[1:]] == ["assistant", "user", "assistant"]
    assert prior_trace[1]["content"] == "seed-user-1"
    assert prior_trace[2]["content"] == "target-reply-1"
    assert prior_trace[3]["content"] == "seed-user-2"


@pytest.mark.asyncio
async def test_respond_filters_visible_message_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A system turn in the conversation must be dropped before role-swap."""
    adapter = _patch_adapter(monkeypatch, _fake_run_output("ok"))
    drv = SyntheticUserDriver(_INFO, _DRIVER_CONFIG)
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "should be dropped"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    await drv.respond(conversation)

    call = adapter.invoke_returning_run_output.await_args
    prior_trace = call.kwargs["prior_trace"]
    # prior_trace = [our system prompt, role-swapped user from u1]
    # The "system" eval-frame turn is filtered out.
    assert len(prior_trace) == 2
    assert prior_trace[0]["content"] == drv._system_prompt
    assert prior_trace[1] == {"role": "assistant", "content": "u1"}


@pytest.mark.asyncio
async def test_respond_drops_tool_dispatch_only_assistant_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool-using target produces assistant turns with content=None and
    tool_calls set. Those are actions the target took, not speech the SU
    should respond to. They must be filtered out before role_swap so the
    swap's strict-content invariant doesn't fire and so the SU's view of
    the conversation stays a coherent text-only dialog.
    """
    adapter = _patch_adapter(monkeypatch, _fake_run_output("ok"))
    drv = SyntheticUserDriver(_INFO, _DRIVER_CONFIG)
    # Shape that comes out of a tool-using target: user → tool-dispatch
    # assistant (content=None) → final text-response assistant.
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "what's the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        },  # type: ignore[typeddict-item]
        {"role": "assistant", "content": "It's 72F."},
    ]

    await drv.respond(conversation)

    call = adapter.invoke_returning_run_output.await_args
    # The tool-dispatch turn (None content) was dropped. After role-swap:
    #   eval [user, assistant("It's 72F.")] → LLM [assistant, user("It's 72F.")]
    # Last swapped turn becomes `input`; prior_trace is [sys, ...rest].
    assert call.args[0] == "It's 72F."
    prior_trace = call.kwargs["prior_trace"]
    assert len(prior_trace) == 2  # [sys, swapped user 'what's the weather']
    assert prior_trace[0]["content"] == drv._system_prompt
    assert prior_trace[1] == {"role": "assistant", "content": "what's the weather?"}


@pytest.mark.asyncio
async def test_respond_drops_empty_string_tool_dispatch_assistant_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter allows dispatch turns with content `''` alongside
    tool_calls, not just None — those must be filtered too, or the SU
    sees a blank user message after role-swap.
    """
    adapter = _patch_adapter(monkeypatch, _fake_run_output("ok"))
    drv = SyntheticUserDriver(_INFO, _DRIVER_CONFIG)
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "what's the weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        },  # type: ignore[typeddict-item]
        {"role": "assistant", "content": "It's 72F."},
    ]

    await drv.respond(conversation)

    call = adapter.invoke_returning_run_output.await_args
    # Same outcome as the content=None dispatch: the empty turn is dropped.
    assert call.args[0] == "It's 72F."
    prior_trace = call.kwargs["prior_trace"]
    assert len(prior_trace) == 2
    assert prior_trace[1] == {"role": "assistant", "content": "what's the weather?"}


@pytest.mark.asyncio
async def test_respond_keeps_assistant_turns_with_text_and_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some providers emit assistant turns with BOTH text content AND
    tool_calls in the same message ("Let me look that up." + dispatch).
    The text is user-facing — the SU should see it. Only turns with no
    text (None or empty content) get dropped.
    """
    adapter = _patch_adapter(monkeypatch, _fake_run_output("ok"))
    drv = SyntheticUserDriver(_INFO, _DRIVER_CONFIG)
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "what's the weather?"},
        {
            "role": "assistant",
            "content": "Let me look that up.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        },  # type: ignore[typeddict-item]
    ]

    await drv.respond(conversation)

    call = adapter.invoke_returning_run_output.await_args
    # The combined text+tool_calls assistant turn is NOT filtered out —
    # its text becomes the input the SU responds to.
    assert call.args[0] == "Let me look that up."


@pytest.mark.asyncio
async def test_respond_with_custom_visible_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom visibility set is honored — the driver doesn't hardcode the default."""
    adapter = _patch_adapter(monkeypatch, _fake_run_output("ok"))
    drv = SyntheticUserDriver(
        _INFO,
        SyntheticUserDriverConfig(
            model_name="x",
            model_provider_name=ModelProviderName.openrouter,
            visible_message_roles=["assistant"],
        ),
    )
    # Only assistant turns visible — the user turn gets filtered out, leaving
    # only the assistant for /respond. After filter, the conversation is a
    # single "assistant" message, which IS the required ends-on-assistant
    # shape: visible=[asst]; swap→[user]; last is input; prior_trace=[sys].
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    await drv.respond(conversation)

    call = adapter.invoke_returning_run_output.await_args
    assert call.args[0] == "a1"
    assert len(call.kwargs["prior_trace"]) == 1  # just the system prompt


# ───────────────────────── respond — invariants ─────────────────────────


@pytest.mark.asyncio
async def test_respond_raises_when_no_visible_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_adapter(monkeypatch, _fake_run_output())
    drv = SyntheticUserDriver(_INFO, _DRIVER_CONFIG)
    # All messages filtered out by visible_message_roles.
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "sys"},
    ]
    with pytest.raises(ValueError, match="No LLM-visible"):
        await drv.respond(conversation)


@pytest.mark.asyncio
async def test_respond_raises_when_empty_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_adapter(monkeypatch, _fake_run_output())
    drv = SyntheticUserDriver(_INFO, _DRIVER_CONFIG)
    with pytest.raises(ValueError, match="No LLM-visible"):
        await drv.respond([])


@pytest.mark.asyncio
async def test_respond_raises_when_ends_on_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_adapter(monkeypatch, _fake_run_output())
    drv = SyntheticUserDriver(_INFO, _DRIVER_CONFIG)
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u1"},
    ]
    with pytest.raises(ValueError, match="end on an assistant"):
        await drv.respond(conversation)


@pytest.mark.asyncio
async def test_respond_raises_on_non_string_adapter_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RunOutput.output is typed as Dict | str — only str is valid for the SU."""
    _patch_adapter(monkeypatch, _fake_run_output({"unexpected": "structured output"}))
    drv = SyntheticUserDriver(_INFO, _DRIVER_CONFIG)
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    with pytest.raises(RuntimeError, match="non-string output"):
        await drv.respond(conversation)


# ───────────────────────── respond — reuse ─────────────────────────


@pytest.mark.asyncio
async def test_respond_reuses_adapter_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter is built once in the constructor and reused per turn."""
    adapter = _patch_adapter(monkeypatch, _fake_run_output("reply"))
    drv = SyntheticUserDriver(_INFO, _DRIVER_CONFIG)

    conversation: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    await drv.respond(conversation)
    await drv.respond(conversation)
    await drv.respond(conversation)

    assert adapter.invoke_returning_run_output.await_count == 3
    # The driver's _adapter reference doesn't change between calls.
