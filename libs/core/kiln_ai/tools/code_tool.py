"""PythonCodeTool — parent-side runtime for user-authored code tools.

A thin caller over :mod:`kiln_ai.tools.sandbox_bridge`: builds a
:class:`~kiln_ai.tools.sandbox_bridge.NestedToolServer`, runs one bridged child via
:func:`~kiln_ai.tools.sandbox_bridge.run_bridged_child`, and maps the raw
:class:`~kiln_ai.tools.sandbox_bridge.BridgeResult` into a :class:`ToolCallResult`.
Depth caps, concurrency bounds, wall-clock timeouts, and nested-call IPC all live in
the shared bridge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from kiln_ai.datamodel.code_tool import CodeTool
from kiln_ai.datamodel.project import Project
from kiln_ai.datamodel.task import Task
from kiln_ai.datamodel.tool_id import ToolId, build_code_tool_id
from kiln_ai.sandbox.worker import child_main
from kiln_ai.tools.base_tool import (
    KilnToolInterface,
    ToolCallContext,
    ToolCallDefinition,
    ToolCallResult,
)
from kiln_ai.tools.sandbox_bridge import (
    BridgeResult,
    NestedToolServer,
    ToolCallLogEntry,
    describe_crash,
    run_bridged_child,
)

__all__ = [
    "ChildOutcome",
    "PythonCodeTool",
    "ToolCallLogEntry",
]

logger = logging.getLogger(__name__)


@dataclass
class ChildOutcome:
    ok: str | None = None
    error: str | None = None
    traceback_str: str | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    timed_out: bool = False
    crashed: bool = False
    exit_code: int | None = None

    def crash_description(self, subject: str) -> str:
        """Phrase this outcome's ``crashed`` state. Shared with the bridge."""
        assert self.crashed, "crash_description() is only meaningful for a crash"
        return describe_crash(subject, self.exit_code)


class PythonCodeTool(KilnToolInterface):
    """Wraps a :class:`CodeTool` artifact as a :class:`KilnToolInterface`."""

    def __init__(
        self,
        code_tool: CodeTool,
        project: Project,
        task: Task | None = None,
        tool_call_recorder: Callable[[ToolCallLogEntry], None] | None = None,
    ):
        self._code_tool = code_tool
        self._project = project
        self._task = task
        self._tool_call_recorder = tool_call_recorder

    async def id(self) -> ToolId:
        return build_code_tool_id(self._code_tool.id)

    async def name(self) -> str:
        return self._code_tool.tool_function_name

    async def description(self) -> str:
        return self._code_tool.tool_description

    async def toolcall_definition(self) -> ToolCallDefinition:
        return {
            "type": "function",
            "function": {
                "name": self._code_tool.tool_function_name,
                "description": self._code_tool.tool_description,
                "parameters": self._code_tool.parameters_schema,
            },
        }

    async def run(
        self, context: ToolCallContext | None = None, **kwargs: Any
    ) -> ToolCallResult:
        outcome = await self._invoke(context, kwargs)

        tool_name = self._code_tool.tool_function_name
        if outcome.timed_out:
            msg = f"Code tool '{tool_name}' timed out after {self._code_tool.timeout_seconds}s"
            return ToolCallResult(
                output=msg, is_error=True, error_message=msg, timed_out=True
            )
        if outcome.crashed:
            msg = outcome.crash_description(f"Code tool '{tool_name}'")
            return ToolCallResult(output=msg, is_error=True, error_message=msg)
        if outcome.error is not None:
            tb_text = outcome.traceback_str or ""
            output = f"Code tool '{tool_name}' failed: {outcome.error}"
            if tb_text:
                output += f"\n{tb_text}"
            return ToolCallResult(
                output=output,
                is_error=True,
                error_message=outcome.error,
            )

        assert outcome.ok is not None
        return ToolCallResult(output=outcome.ok)

    def _build_server(self, context: ToolCallContext | None) -> NestedToolServer:
        return NestedToolServer(
            allowlist=self._code_tool.tool_allowlist,
            project=self._project,
            task=self._task,
            context=context,
            recorder=self._tool_call_recorder,
        )

    async def _invoke(
        self, context: ToolCallContext | None, kwargs: dict[str, Any]
    ) -> ChildOutcome:
        server = self._build_server(context)
        result = await run_bridged_child(
            target=child_main,
            args=(self._code_tool.code, kwargs),
            timeout_s=float(self._code_tool.timeout_seconds),
            server=server,
        )
        return _bridge_result_to_outcome(result)


def _bridge_result_to_outcome(result: BridgeResult) -> ChildOutcome:
    if result.timed_out:
        return ChildOutcome(timed_out=True, duration_ms=result.duration_ms)
    if result.crashed:
        return ChildOutcome(
            crashed=True,
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
        )

    msg = result.result_msg
    assert msg is not None
    if "ok" in msg:
        return ChildOutcome(
            ok=msg["ok"],
            stdout=msg.get("stdout", ""),
            stderr=msg.get("stderr", ""),
            duration_ms=result.duration_ms,
        )
    return ChildOutcome(
        error=msg.get("error", "unknown error"),
        traceback_str=msg.get("traceback"),
        stdout=msg.get("stdout", ""),
        stderr=msg.get("stderr", ""),
        duration_ms=result.duration_ms,
    )
