import json
import re
from typing import Any

from kiln_ai.adapters.eval.base_eval import BaseV2EvalBridge
from kiln_ai.adapters.eval.eval_utils.v2_eval_helpers import build_binary_scores
from kiln_ai.datamodel.eval import (
    ArgMatch,
    EvalTaskInput,
    SkippedReason,
    ToolCallCheckProperties,
    ToolCallSpec,
    V2EvalResult,
)


class ToolCallCheckEval(BaseV2EvalBridge):
    """V2 adapter for tool_call_check: validates tool calls in the trace."""

    async def evaluate(self, eval_input: EvalTaskInput) -> V2EvalResult:
        props = self.properties
        assert isinstance(props, ToolCallCheckProperties)

        if eval_input.trace is None:
            return V2EvalResult(
                skipped_reason=SkippedReason.missing_trace,
                skipped_detail="tool_call_check requires a trace",
            )

        actual_calls = self._extract_tool_calls(eval_input.trace)

        passed = self._check(
            actual_calls,
            props.expected_tools,
            props.match_mode,
            props.on_unexpected_tools,
        )

        return V2EvalResult(
            scores=build_binary_scores(self._output_scores, passed),
        )

    @staticmethod
    def _extract_tool_calls(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract all tool calls from the trace as a flat list.

        Each entry is {"name": str, "arguments": dict}.
        """
        calls: list[dict[str, Any]] = []
        for msg in trace:
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                continue
            for tc in tool_calls:
                # .get("function", {}) isn't enough: the default only covers an
                # absent key, so a present-but-null value raises on the .get below.
                func = tc.get("function") if isinstance(tc, dict) else None
                if not isinstance(func, dict):
                    func = {}
                name = func.get("name", "")
                args_str = func.get("arguments", "{}")
                try:
                    args = (
                        json.loads(args_str) if isinstance(args_str, str) else args_str
                    )
                except (json.JSONDecodeError, TypeError):
                    args = {}
                calls.append(
                    {"name": name, "arguments": args if isinstance(args, dict) else {}}
                )
        return calls

    def _check(
        self,
        actual_calls: list[dict[str, Any]],
        expected_tools: list[ToolCallSpec],
        match_mode: str,
        on_unexpected: str,
    ) -> bool:
        """Check tool calls against expectations, returning pass/fail."""
        if match_mode == "never":
            for spec in expected_tools:
                for call in actual_calls:
                    if self._call_matches_spec(call, spec):
                        return False
            return True

        if match_mode == "any":
            found_any = False
            for spec in expected_tools:
                for call in actual_calls:
                    if self._call_matches_spec(call, spec):
                        found_any = True
                        break
                if found_any:
                    break
            passed = found_any
        elif match_mode == "all":
            passed = True
            for spec in expected_tools:
                found = False
                for call in actual_calls:
                    if self._call_matches_spec(call, spec):
                        found = True
                        break
                if not found:
                    passed = False
                    break
        elif match_mode == "ordered":
            passed = self._check_ordered(actual_calls, expected_tools)
        else:
            passed = False

        if passed and on_unexpected == "fail":
            for call in actual_calls:
                if not any(
                    self._call_matches_spec(call, spec) for spec in expected_tools
                ):
                    return False

        return passed

    def _check_ordered(
        self,
        actual_calls: list[dict[str, Any]],
        expected_tools: list[ToolCallSpec],
    ) -> bool:
        """Check that expected tools appear in order within actual calls."""
        expected_idx = 0
        for call in actual_calls:
            if expected_idx >= len(expected_tools):
                break
            if self._call_matches_spec(call, expected_tools[expected_idx]):
                expected_idx += 1
        return expected_idx == len(expected_tools)

    @staticmethod
    def _call_matches_spec(call: dict[str, Any], spec: ToolCallSpec) -> bool:
        """Check if a single actual call matches a ToolCallSpec."""
        if call["name"] != spec.tool_name:
            return False
        if spec.expected_args is None:
            return True
        actual_args = call.get("arguments", {})
        for arg_name, arg_match in spec.expected_args.items():
            if arg_name not in actual_args:
                return False
            if not _arg_value_matches(actual_args[arg_name], arg_match):
                return False
        return True


def _arg_value_matches(actual: Any, arg_match: ArgMatch) -> bool:
    """Check if an actual argument value matches an ArgMatch spec."""
    if arg_match.match_mode == "exact":
        return actual == arg_match.value  # type: ignore[no-any-return]
    elif arg_match.match_mode == "contains":
        return str(arg_match.value) in str(actual)
    elif arg_match.match_mode == "regex":
        try:
            return re.search(str(arg_match.value), str(actual)) is not None
        except re.error:
            return False
    return False
