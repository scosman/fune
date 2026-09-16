"""Unit tests for the eval-frame ↔ LLM-frame role swap."""

import pytest

from kiln_ai.synthetic_user.role_swap import role_swap
from kiln_ai.utils.open_ai_types import ChatCompletionMessageParam


def test_empty_conversation() -> None:
    assert role_swap([]) == []


def test_single_user_becomes_assistant() -> None:
    src: list[ChatCompletionMessageParam] = [{"role": "user", "content": "hi"}]
    assert role_swap(src) == [{"role": "assistant", "content": "hi"}]


def test_single_assistant_becomes_user() -> None:
    src: list[ChatCompletionMessageParam] = [{"role": "assistant", "content": "hello"}]
    assert role_swap(src) == [{"role": "user", "content": "hello"}]


def test_alternating_conversation_flips_both_directions() -> None:
    src: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    expected: list[ChatCompletionMessageParam] = [
        {"role": "assistant", "content": "u1"},
        {"role": "user", "content": "a1"},
        {"role": "assistant", "content": "u2"},
        {"role": "user", "content": "a2"},
    ]
    assert role_swap(src) == expected


def test_order_preserved() -> None:
    src: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": f"m{i}"} for i in range(5)
    ]
    out = role_swap(src)
    assert [m["content"] for m in out] == [f"m{i}" for i in range(5)]


def test_multiline_content_passes_through() -> None:
    src: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "line 1\nline 2\nline 3"}
    ]
    assert role_swap(src) == [
        {"role": "assistant", "content": "line 1\nline 2\nline 3"}
    ]


def test_system_role_raises() -> None:
    src: list[ChatCompletionMessageParam] = [{"role": "system", "content": "sys"}]
    with pytest.raises(ValueError, match="system"):
        role_swap(src)


def test_tool_role_raises() -> None:
    src: list[ChatCompletionMessageParam] = [
        {"role": "tool", "content": "{}", "tool_call_id": "x"}
    ]
    with pytest.raises(ValueError, match="tool"):
        role_swap(src)


def test_does_not_mutate_input() -> None:
    src: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    snapshot = [dict(m) for m in src]
    role_swap(src)
    assert [dict(m) for m in src] == snapshot
