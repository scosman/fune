"""Tests for ToolCallCheckEval adapter."""

import json

import pytest

from kiln_ai.adapters.eval.conftest import make_eval_task_input, make_v2_eval_config
from kiln_ai.adapters.eval.v2_eval_tool_call_check import ToolCallCheckEval
from kiln_ai.datamodel.eval import (
    ArgMatch,
    EvalTaskInput,
    SkippedReason,
    ToolCallCheckProperties,
    ToolCallSpec,
)

_make_config = make_v2_eval_config


def _inp(**overrides: object) -> EvalTaskInput:
    final_message = overrides.pop("final_message", "hello")
    return make_eval_task_input(final_message=str(final_message), **overrides)


def _trace_with_tool_calls(
    *calls: tuple[str, dict],
) -> list[dict]:
    """Build a trace with assistant tool calls. Each call is (name, args_dict)."""
    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        for i, (name, args) in enumerate(calls)
    ]
    return [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "tool_calls": tool_calls},
    ]


class TestToolCallCheckAllMode:
    @pytest.mark.asyncio
    async def test_pass(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[ToolCallSpec(tool_name="search")],
            )
        )
        trace = _trace_with_tool_calls(("search", {"q": "hi"}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 1.0}
        assert result.skipped_reason is None

    @pytest.mark.asyncio
    async def test_fail(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[
                    ToolCallSpec(tool_name="search"),
                    ToolCallSpec(tool_name="fetch"),
                ],
            )
        )
        trace = _trace_with_tool_calls(("search", {}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 0.0}
        assert result.skipped_reason is None


class TestToolCallCheckAnyMode:
    @pytest.mark.asyncio
    async def test_pass(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[
                    ToolCallSpec(tool_name="search"),
                    ToolCallSpec(tool_name="fetch"),
                ],
                match_mode="any",
            )
        )
        trace = _trace_with_tool_calls(("search", {}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 1.0}
        assert result.skipped_reason is None

    @pytest.mark.asyncio
    async def test_fail(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[
                    ToolCallSpec(tool_name="search"),
                    ToolCallSpec(tool_name="fetch"),
                ],
                match_mode="any",
            )
        )
        trace = _trace_with_tool_calls(("unrelated", {}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 0.0}
        assert result.skipped_reason is None


class TestToolCallCheckOrderedMode:
    @pytest.mark.asyncio
    async def test_pass(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[
                    ToolCallSpec(tool_name="search"),
                    ToolCallSpec(tool_name="fetch"),
                ],
                match_mode="ordered",
            )
        )
        trace = _trace_with_tool_calls(("search", {}), ("other", {}), ("fetch", {}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 1.0}
        assert result.skipped_reason is None

    @pytest.mark.asyncio
    async def test_fail(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[
                    ToolCallSpec(tool_name="search"),
                    ToolCallSpec(tool_name="fetch"),
                ],
                match_mode="ordered",
            )
        )
        trace = _trace_with_tool_calls(("fetch", {}), ("search", {}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 0.0}
        assert result.skipped_reason is None


class TestToolCallCheckNeverMode:
    @pytest.mark.asyncio
    async def test_pass(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[ToolCallSpec(tool_name="delete")],
                match_mode="never",
            )
        )
        trace = _trace_with_tool_calls(("search", {}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 1.0}
        assert result.skipped_reason is None

    @pytest.mark.asyncio
    async def test_fail(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[ToolCallSpec(tool_name="delete")],
                match_mode="never",
            )
        )
        trace = _trace_with_tool_calls(("delete", {}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 0.0}
        assert result.skipped_reason is None


class TestToolCallCheckNeverModeWithArgs:
    @pytest.mark.asyncio
    async def test_never_with_matching_args_fails(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[
                    ToolCallSpec(
                        tool_name="delete_database",
                        expected_args={
                            "force": ArgMatch(value=True, match_mode="exact")
                        },
                    )
                ],
                match_mode="never",
            )
        )
        trace = _trace_with_tool_calls(("delete_database", {"force": True}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 0.0}
        assert result.skipped_reason is None

    @pytest.mark.asyncio
    async def test_never_with_non_matching_args_passes(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[
                    ToolCallSpec(
                        tool_name="delete_database",
                        expected_args={
                            "force": ArgMatch(value=True, match_mode="exact")
                        },
                    )
                ],
                match_mode="never",
            )
        )
        trace = _trace_with_tool_calls(("delete_database", {"force": False}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 1.0}
        assert result.skipped_reason is None

    @pytest.mark.asyncio
    async def test_never_with_args_different_tool_passes(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[
                    ToolCallSpec(
                        tool_name="delete_database",
                        expected_args={
                            "force": ArgMatch(value=True, match_mode="exact")
                        },
                    )
                ],
                match_mode="never",
            )
        )
        trace = _trace_with_tool_calls(("search", {"q": "hi"}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 1.0}
        assert result.skipped_reason is None


class TestToolCallCheckUnexpectedTools:
    @pytest.mark.asyncio
    async def test_on_unexpected_fail(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[ToolCallSpec(tool_name="search")],
                on_unexpected_tools="fail",
            )
        )
        trace = _trace_with_tool_calls(("search", {}), ("other", {}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 0.0}
        assert result.skipped_reason is None

    @pytest.mark.asyncio
    async def test_on_unexpected_fail_with_args_mismatch(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[
                    ToolCallSpec(
                        tool_name="search",
                        expected_args={
                            "query": ArgMatch(value="foo", match_mode="exact")
                        },
                    )
                ],
                on_unexpected_tools="fail",
            )
        )
        trace = _trace_with_tool_calls(
            ("search", {"query": "foo"}), ("search", {"query": "bar"})
        )
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 0.0}
        assert result.skipped_reason is None

    @pytest.mark.asyncio
    async def test_on_unexpected_fail_all_args_match(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[
                    ToolCallSpec(
                        tool_name="search",
                        expected_args={
                            "query": ArgMatch(value="foo", match_mode="exact")
                        },
                    )
                ],
                on_unexpected_tools="fail",
            )
        )
        trace = _trace_with_tool_calls(("search", {"query": "foo"}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 1.0}
        assert result.skipped_reason is None

    @pytest.mark.asyncio
    async def test_on_unexpected_ignore(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[ToolCallSpec(tool_name="search")],
                on_unexpected_tools="ignore",
            )
        )
        trace = _trace_with_tool_calls(("search", {}), ("other", {}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 1.0}
        assert result.skipped_reason is None


class TestToolCallCheckArgMatch:
    @pytest.mark.asyncio
    async def test_exact(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[
                    ToolCallSpec(
                        tool_name="search",
                        expected_args={"query": ArgMatch(value="test")},
                    )
                ],
            )
        )
        trace = _trace_with_tool_calls(("search", {"query": "test"}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 1.0}

    @pytest.mark.asyncio
    async def test_contains(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[
                    ToolCallSpec(
                        tool_name="search",
                        expected_args={
                            "query": ArgMatch(value="test", match_mode="contains")
                        },
                    )
                ],
            )
        )
        trace = _trace_with_tool_calls(("search", {"query": "this is a test query"}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 1.0}

    @pytest.mark.asyncio
    async def test_regex(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[
                    ToolCallSpec(
                        tool_name="search",
                        expected_args={
                            "query": ArgMatch(value=r"\d+", match_mode="regex")
                        },
                    )
                ],
            )
        )
        trace = _trace_with_tool_calls(("search", {"query": "item 42"}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 1.0}

    @pytest.mark.asyncio
    async def test_fail(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[
                    ToolCallSpec(
                        tool_name="search",
                        expected_args={"query": ArgMatch(value="test")},
                    )
                ],
            )
        )
        trace = _trace_with_tool_calls(("search", {"query": "other"}))
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 0.0}


class TestToolCallCheckEdgeCases:
    @pytest.mark.asyncio
    async def test_missing_trace_skips(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[ToolCallSpec(tool_name="search")],
            )
        )
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=None))
        assert result.scores == {}
        assert result.skipped_reason == SkippedReason.missing_trace
        assert result.skipped_detail is not None

    @pytest.mark.asyncio
    async def test_empty_trace(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[ToolCallSpec(tool_name="search")],
            )
        )
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=[]))
        assert result.scores == {"score_a": 0.0}
        assert result.skipped_reason is None

    @pytest.mark.asyncio
    async def test_multiple_assistant_messages(self):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[
                    ToolCallSpec(tool_name="search"),
                    ToolCallSpec(tool_name="fetch"),
                ],
            )
        )
        trace = [
            {"role": "user", "content": "msg1"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({}),
                        },
                    }
                ],
            },
            {"role": "user", "content": "msg2"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {
                            "name": "fetch",
                            "arguments": json.dumps({}),
                        },
                    }
                ],
            },
        ]
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 1.0}
        assert result.skipped_reason is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "args_str",
        ["not valid json", "", None],
        ids=["invalid_json", "empty_string", "none"],
    )
    async def test_malformed_arguments_handled(self, args_str: str | None):
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[ToolCallSpec(tool_name="search")],
            )
        )
        trace = [
            {"role": "user", "content": "msg"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "search", "arguments": args_str},
                    }
                ],
            },
        ]
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 1.0}
        assert result.skipped_reason is None

    @pytest.mark.asyncio
    async def test_malformed_tool_call_entries_handled(self):
        """A null or non-dict "function", or a non-dict tool call, must degrade to
        an unnamed call rather than raise partway through extraction."""
        cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[ToolCallSpec(tool_name="search")],
            )
        )
        trace = [
            {"role": "user", "content": "msg"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "function": None},
                    {"id": "c2", "function": "oops"},
                    "not_a_dict",
                    {
                        "id": "c3",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"q": "cats"}),
                        },
                    },
                ],
            },
        ]
        calls = ToolCallCheckEval._extract_tool_calls(trace)
        assert [c["name"] for c in calls] == ["", "", "", "search"]
        assert [c["arguments"] for c in calls] == [{}, {}, {}, {"q": "cats"}]

        # The well-formed call is still found, so the eval scores instead of erroring.
        result = await ToolCallCheckEval(cfg).evaluate(_inp(trace=trace))
        assert result.scores == {"score_a": 1.0}
        assert result.skipped_reason is None

        # Degraded entries are unparseable calls: fail mode counts them as
        # unexpected tools rather than pretending they were not made.
        strict_cfg = _make_config(
            ToolCallCheckProperties(
                expected_tools=[ToolCallSpec(tool_name="search")],
                on_unexpected_tools="fail",
            )
        )
        strict = await ToolCallCheckEval(strict_cfg).evaluate(_inp(trace=trace))
        assert strict.scores == {"score_a": 0.0}
