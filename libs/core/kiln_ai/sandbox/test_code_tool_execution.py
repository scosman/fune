"""Tests for the code-tool execution engine.

Child/protocol tests spawn real child processes. Parent-side tests use
mock tool doubles. Shaped after the existing ``test_sandbox_worker.py`` suite.
"""

import asyncio
import json
import textwrap
from unittest.mock import patch

import pytest

from kiln_ai.datamodel.code_tool import CodeTool
from kiln_ai.datamodel.project import Project
from kiln_ai.sandbox.spawn import _spawn_lock
from kiln_ai.tools.base_tool import (
    KilnToolInterface,
    ToolCallDefinition,
    ToolCallResult,
)
from kiln_ai.tools.code_tool import (
    PythonCodeTool,
    ToolCallLogEntry,
)
from kiln_ai.tools.sandbox_bridge import (
    CODE_SANDBOX_MAX_CONCURRENCY,
    NestedToolServer,
    _depth,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "string"}},
}
EMPTY_SCHEMA = {"type": "object", "properties": {}}


def _make_code_tool(code: str, **overrides) -> CodeTool:
    defaults = {
        "name": "Test Tool",
        "tool_function_name": "test_tool",
        "tool_description": "A test tool",
        "parameters_schema": VALID_SCHEMA,
        "code": code,
        "timeout_seconds": 10,
    }
    defaults.update(overrides)
    return CodeTool(**defaults)


def _make_project(tmp_path) -> Project:
    p = Project(name="test_project", path=tmp_path / "project")
    p.save_to_file()
    return p


def _make_python_code_tool(
    tmp_path,
    code: str,
    tool_allowlist=None,
    tool_call_recorder=None,
    **overrides,
) -> PythonCodeTool:
    project = _make_project(tmp_path)
    ct = _make_code_tool(
        code,
        tool_allowlist=tool_allowlist or [],
        **overrides,
    )
    ct.parent = project
    return PythonCodeTool(
        ct,
        project,
        tool_call_recorder=tool_call_recorder,
    )


class FakeTool(KilnToolInterface):
    """Minimal tool double for testing nested calls.

    IMPORTANT: ``fn_name`` intentionally DIFFERS from the tool_id slug
    (e.g. ``fn_name="fake_add"`` for ``tool_id="kiln_tool::add_numbers"``).
    This ensures tests catch name-derivation bugs where the dispatch map
    would use the slug instead of ``tool.name()``.
    """

    def __init__(
        self,
        tool_id: str,
        fn_name: str,
        fn_desc: str = "fake",
        params: dict | None = None,
        result: ToolCallResult | None = None,
        delay: float = 0,
    ):
        self._id = tool_id
        self._name = fn_name
        self._desc = fn_desc
        self._params = params or EMPTY_SCHEMA
        self._result = result or ToolCallResult(output="ok")
        self._delay = delay

    async def id(self):
        return self._id

    async def name(self):
        return self._name

    async def description(self):
        return self._desc

    async def toolcall_definition(self) -> ToolCallDefinition:
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": self._desc,
                "parameters": self._params,
            },
        }

    async def run(self, context=None, **kwargs) -> ToolCallResult:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return self._result


# ---------------------------------------------------------------------------
# Child / protocol tests (real spawns)
# ---------------------------------------------------------------------------


class TestChildSyncRun:
    @pytest.mark.asyncio
    async def test_sync_run_returns_string(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            'def run(x):\n    return "hello " + x\n',
        )
        result = await tool.run(None, x="world")
        assert not result.is_error
        assert result.output == "hello world"

    @pytest.mark.asyncio
    async def test_sync_run_returns_dict(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            'def run(x):\n    return {"value": x}\n',
        )
        result = await tool.run(None, x="test")
        assert not result.is_error
        assert json.loads(result.output) == {"value": "test"}

    @pytest.mark.asyncio
    async def test_sync_run_returns_none(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            "def run(x):\n    pass\n",
        )
        result = await tool.run(None, x="test")
        assert not result.is_error
        assert result.output == "null"


class TestChildAsyncRun:
    @pytest.mark.asyncio
    async def test_async_run_returns_string(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            textwrap.dedent("""\
                import asyncio
                async def run(x):
                    async def greet(name):
                        return "hi " + name
                    results = await asyncio.gather(greet(x), greet(x + "!"))
                    return " ".join(results)
            """),
        )
        result = await tool.run(None, x="a")
        assert not result.is_error
        assert result.output == "hi a hi a!"

    @pytest.mark.asyncio
    async def test_asyncio_run_inside_async_errors(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            textwrap.dedent("""\
                import asyncio
                async def helper():
                    return 1
                async def run(x):
                    return asyncio.run(helper())
            """),
        )
        result = await tool.run(None, x="test")
        assert result.is_error
        assert (
            "cannot be called from a running event loop" in result.output.lower()
            or "cannot" in result.output.lower()
        )


class TestReturnSerialization:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code,expected",
        [
            ('def run(x):\n    return "raw"\n', "raw"),
            ("def run(x):\n    return 42\n", "42"),
            ("def run(x):\n    return 3.14\n", "3.14"),
            ("def run(x):\n    return True\n", "true"),
            ("def run(x):\n    return False\n", "false"),
            ("def run(x):\n    return None\n", "null"),
            ("def run(x):\n    return [1, 2]\n", "[1, 2]"),
            ('def run(x):\n    return {"k": "v"}\n', '{"k": "v"}'),
        ],
        ids=[
            "str",
            "int",
            "float",
            "bool_true",
            "bool_false",
            "none",
            "list",
            "dict",
        ],
    )
    async def test_serialization(self, tmp_path, code, expected):
        tool = _make_python_code_tool(tmp_path, code)
        result = await tool.run(None, x="test")
        assert not result.is_error
        assert result.output == expected

    @pytest.mark.asyncio
    async def test_non_serializable_type_errors(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            "def run(x):\n    return object()\n",
        )
        result = await tool.run(None, x="test")
        assert result.is_error
        assert "must return str or JSON-serializable" in result.output

    @pytest.mark.asyncio
    async def test_non_serializable_nested_value_errors(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            "def run(x):\n    return {'fn': lambda: None}\n",
        )
        result = await tool.run(None, x="test")
        assert result.is_error
        assert "non-JSON-serializable" in result.output

    @pytest.mark.asyncio
    async def test_string_passthrough_no_parsing(self, tmp_path):
        """JSON-shaped string returned by run() comes back as-is, not parsed."""
        tool = _make_python_code_tool(
            tmp_path,
            'def run(x):\n    return \'{"key": "value"}\'\n',
        )
        result = await tool.run(None, x="test")
        assert not result.is_error
        assert result.output == '{"key": "value"}'


class TestStdoutStderr:
    @pytest.mark.asyncio
    async def test_stdout_captured(self, tmp_path):
        project = _make_project(tmp_path)
        ct = _make_code_tool(
            'import sys\ndef run(x):\n    sys.stdout.write("debug")\n    return "ok"\n',
        )
        ct.parent = project
        pct = PythonCodeTool(ct, project)
        outcome = await pct._invoke(None, {"x": "test"})
        assert outcome.ok == "ok"
        assert "debug" in outcome.stdout

    @pytest.mark.asyncio
    async def test_stdout_truncation(self, tmp_path):
        project = _make_project(tmp_path)
        ct = _make_code_tool(
            'import sys\ndef run(x):\n    sys.stdout.write("A" * 100000)\n    return "ok"\n',
        )
        ct.parent = project
        pct = PythonCodeTool(ct, project)
        outcome = await pct._invoke(None, {"x": "test"})
        assert outcome.ok == "ok"
        assert len(outcome.stdout) <= 64 * 1024 + 50
        assert "truncated" in outcome.stdout


class TestTraceback:
    @pytest.mark.asyncio
    async def test_traceback_shows_code_tool_lines(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            textwrap.dedent("""\
                def helper():
                    raise ValueError("kaboom")
                def run(x):
                    helper()
            """),
        )
        result = await tool.run(None, x="test")
        assert result.is_error
        assert "kaboom" in result.output
        assert "<code_tool>" in result.output
        assert "worker.py" not in result.output


class TestMissingRun:
    @pytest.mark.asyncio
    async def test_missing_run_defense(self, tmp_path):
        """Even if save-time validation is bypassed, child handles missing run()."""
        project = _make_project(tmp_path)
        ct = CodeTool.__new__(CodeTool)
        object.__setattr__(
            ct,
            "__dict__",
            {
                "name": "bad",
                "tool_function_name": "bad",
                "tool_description": "bad",
                "parameters_schema": EMPTY_SCHEMA,
                "code": "x = 1\n",
                "timeout_seconds": 10,
                "tool_allowlist": [],
                "description": None,
                "is_archived": False,
                "id": "test123",
                "v": 1,
                "created_at": None,
                "created_by": None,
                "path": None,
            },
        )
        object.__setattr__(ct, "__pydantic_fields_set__", set())
        pct = PythonCodeTool(ct, project)
        result = await pct.run(None)
        assert result.is_error
        assert "run" in result.output.lower()


class TestImportForms:
    @pytest.mark.asyncio
    async def test_from_kiln_import_tools(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            textwrap.dedent("""\
                from kiln import tools
                def run(x):
                    return type(tools).__name__
            """),
        )
        result = await tool.run(None, x="test")
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_import_kiln_tools(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            textwrap.dedent("""\
                import kiln.tools
                def run(x):
                    return type(kiln.tools).__name__
            """),
        )
        result = await tool.run(None, x="test")
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_from_kiln_tools_import_exception(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            textwrap.dedent("""\
                from kiln.tools import ToolCallError
                def run(x):
                    return ToolCallError.__name__
            """),
        )
        result = await tool.run(None, x="test")
        assert not result.is_error
        assert result.output == "ToolCallError"

    @pytest.mark.asyncio
    async def test_from_kiln_import_async_tools(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            textwrap.dedent("""\
                from kiln import async_tools
                def run(x):
                    return type(async_tools).__name__
            """),
        )
        result = await tool.run(None, x="test")
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_exception_classes_identical_across_modules(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            textwrap.dedent("""\
                from kiln import tools, async_tools
                def run(x):
                    same_not_allowed = tools.ToolNotAllowed is async_tools.ToolNotAllowed
                    same_timeout = tools.ToolTimeout is async_tools.ToolTimeout
                    same_call_error = tools.ToolCallError is async_tools.ToolCallError
                    return str(same_not_allowed and same_timeout and same_call_error)
            """),
        )
        result = await tool.run(None, x="test")
        assert not result.is_error
        assert result.output == "True"


class TestJsonUnsafeKwargs:
    @pytest.mark.asyncio
    async def test_json_unsafe_kwargs_raise_in_frame(self, tmp_path):
        """Non-JSON-serializable tool kwargs raise ToolCallError inside child."""
        code = textwrap.dedent("""\
            from kiln.tools import ToolCallError
            from kiln import tools
            def run(x):
                try:
                    tools.some_tool(bad=object())
                except ToolCallError as e:
                    return f"caught: {e.tool}"
                return "no error"
        """)
        tool = _make_python_code_tool(tmp_path, code)
        result = await tool.run(None, x="test")
        assert not result.is_error
        assert "caught: some_tool" in result.output


# ---------------------------------------------------------------------------
# Parent-side tests (mock tools)
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_simple_run(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            'def run(x):\n    return "result_" + x\n',
        )
        result = await tool.run(None, x="abc")
        assert not result.is_error
        assert result.output == "result_abc"


class TestNestedToolCalls:
    @pytest.mark.asyncio
    async def test_nested_tool_success(self, tmp_path):
        fake = FakeTool(
            "kiln_tool::add_numbers",
            "fake_add",
            params=EMPTY_SCHEMA,
            result=ToolCallResult(output="42"),
        )
        code = textwrap.dedent("""\
            from kiln import tools
            def run(x):
                result = tools.fake_add()
                return "got: " + result
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=fake,
        ):
            result = await tool.run(None, x="test")
        assert not result.is_error
        assert result.output == "got: 42"

    @pytest.mark.asyncio
    async def test_nested_tool_is_error(self, tmp_path):
        fake = FakeTool(
            "kiln_tool::add_numbers",
            "fake_add",
            params=EMPTY_SCHEMA,
            result=ToolCallResult(
                output="tool failed", is_error=True, error_message="tool failed"
            ),
        )
        code = textwrap.dedent("""\
            from kiln.tools import ToolCallError
            from kiln import tools
            def run(x):
                try:
                    tools.fake_add()
                except ToolCallError as e:
                    return f"error: {e.message}"
                return "no error"
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=fake,
        ):
            result = await tool.run(None, x="test")
        assert not result.is_error
        assert "error: tool failed" in result.output

    @pytest.mark.asyncio
    async def test_nested_tool_not_allowed(self, tmp_path):
        fake = FakeTool(
            "kiln_tool::add_numbers",
            "fake_add",
            params=EMPTY_SCHEMA,
        )
        code = textwrap.dedent("""\
            from kiln.tools import ToolNotAllowed
            from kiln import tools
            def run(x):
                try:
                    tools.nonexistent_tool()
                except ToolNotAllowed as e:
                    return f"not allowed: {e.tool}"
                return "no error"
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=fake,
        ):
            result = await tool.run(None, x="test")
        assert not result.is_error
        assert "not allowed: nonexistent_tool" in result.output

    @pytest.mark.asyncio
    async def test_nested_tool_ambiguous(self, tmp_path):
        fake1 = FakeTool("mcp::remote::server1::search", "search")
        fake2 = FakeTool("mcp::remote::server2::search", "search")
        fakes = {
            "mcp::remote::server1::search": fake1,
            "mcp::remote::server2::search": fake2,
        }
        code = textwrap.dedent("""\
            from kiln.tools import ToolCallError
            from kiln import tools
            def run(x):
                try:
                    tools.search()
                except ToolCallError as e:
                    return f"ambiguous: {e.message}"
                return "no error"
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=[
                "mcp::remote::server1::search",
                "mcp::remote::server2::search",
            ],
        )
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            side_effect=lambda tid, **kw: fakes[tid],
        ):
            result = await tool.run(None, x="test")
        assert not result.is_error
        assert "ambiguous" in result.output.lower()

    @pytest.mark.asyncio
    async def test_nested_tool_invalid_kwargs(self, tmp_path):
        fake = FakeTool(
            "kiln_tool::add_numbers",
            "fake_add",
            params={
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
            },
            result=ToolCallResult(output="42"),
        )
        code = textwrap.dedent("""\
            from kiln.tools import ToolCallError
            from kiln import tools
            def run(x):
                try:
                    tools.fake_add(a="not_an_int")
                except ToolCallError as e:
                    return f"invalid: {e.tool}"
                return "no error"
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=fake,
        ):
            result = await tool.run(None, x="test")
        assert not result.is_error
        assert "invalid: fake_add" in result.output


class TestListTools:
    @pytest.mark.asyncio
    async def test_list_tools_returns_content(self, tmp_path):
        fake = FakeTool(
            "kiln_tool::add_numbers",
            "fake_add",
            fn_desc="Add two numbers",
            params={
                "type": "object",
                "properties": {"a": {"type": "integer"}},
            },
        )
        code = textwrap.dedent("""\
            import json
            from kiln import tools
            def run(x):
                tool_list = tools.list_tools()
                return json.dumps(tool_list)
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=fake,
        ):
            result = await tool.run(None, x="test")
        assert not result.is_error
        tool_list = json.loads(result.output)
        assert len(tool_list) == 1
        assert tool_list[0]["name"] == "fake_add"
        assert tool_list[0]["description"] == "Add two numbers"


class TestBrokenAllowlistEntry:
    """One unresolvable allowlist entry (e.g. a deleted RAG config) must not
    take down the whole nested-tool surface."""

    BROKEN_ID = "kiln_tool::rag::missing"

    def _patch_registry(self, healthy: FakeTool):
        def resolver(tool_id, project=None, task=None):
            if tool_id == self.BROKEN_ID:
                raise ValueError(f"RAG config not found: {tool_id}")
            return healthy

        return patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            side_effect=resolver,
        )

    @pytest.mark.asyncio
    async def test_list_tools_shows_healthy_and_unavailable(self, tmp_path):
        fake = FakeTool("kiln_tool::add_numbers", "fake_add", params=EMPTY_SCHEMA)
        code = textwrap.dedent("""\
            import json
            from kiln import tools
            def run(x):
                return json.dumps(tools.list_tools())
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers", self.BROKEN_ID],
        )
        with self._patch_registry(fake):
            result = await tool.run(None, x="test")
        assert not result.is_error
        tool_list = json.loads(result.output)
        assert len(tool_list) == 2
        by_name = {t["name"]: t for t in tool_list}
        assert by_name["fake_add"]["description"] == "fake"
        broken = by_name[self.BROKEN_ID]
        assert broken["description"].startswith("(unavailable:")
        assert "RAG config not found" in broken["description"]

    @pytest.mark.asyncio
    async def test_healthy_tool_still_callable(self, tmp_path):
        fake = FakeTool(
            "kiln_tool::add_numbers",
            "fake_add",
            params=EMPTY_SCHEMA,
            result=ToolCallResult(output="42"),
        )
        code = textwrap.dedent("""\
            from kiln import tools
            def run(x):
                return tools.fake_add()
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers", self.BROKEN_ID],
        )
        with self._patch_registry(fake):
            result = await tool.run(None, x="test")
        assert not result.is_error
        assert result.output == "42"

    @pytest.mark.asyncio
    async def test_calling_broken_tool_reports_unavailable(self, tmp_path):
        fake = FakeTool("kiln_tool::add_numbers", "fake_add", params=EMPTY_SCHEMA)
        code = textwrap.dedent("""\
            from kiln.tools import ToolNotAllowed
            from kiln import tools
            def run(x):
                try:
                    tools.missing_rag()
                except ToolNotAllowed as e:
                    return f"unavailable: {e}"
                return "no error"
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers", self.BROKEN_ID],
        )
        with self._patch_registry(fake):
            result = await tool.run(None, x="test")
        assert not result.is_error
        assert "unavailable:" in result.output
        assert "not available" in result.output
        assert "fake_add" in result.output


class TestTimeout:
    @pytest.mark.asyncio
    async def test_timeout_kills_child(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            "import time\ndef run(x):\n    time.sleep(30)\n    return 'done'\n",
            timeout_seconds=1,
        )
        result = await tool.run(None, x="test")
        assert result.is_error
        assert "timed out" in result.output

    @pytest.mark.asyncio
    async def test_timeout_during_nested_call(self, tmp_path):
        slow_fake = FakeTool(
            "kiln_tool::add_numbers",
            "fake_add",
            params=EMPTY_SCHEMA,
            result=ToolCallResult(output="42"),
            delay=30,
        )
        code = textwrap.dedent("""\
            from kiln import tools
            def run(x):
                return tools.fake_add()
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
            timeout_seconds=1,
        )
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=slow_fake,
        ):
            result = await tool.run(None, x="test")
        assert result.is_error
        assert "timed out" in result.output


class TestNestedTimeoutKind:
    """The dispatcher classifies a nested code tool's timeout from the typed
    ``timed_out`` flag on its result, never from the error text."""

    def _nested_code_tool(self, project, code: str, timeout_seconds: int = 10):
        ct = _make_code_tool(
            code,
            name="Nested Tool",
            tool_function_name="nested_tool",
            parameters_schema=EMPTY_SCHEMA,
            tool_allowlist=[],
            timeout_seconds=timeout_seconds,
        )
        ct.parent = project
        return PythonCodeTool(ct, project)

    OUTER_CODE = textwrap.dedent("""\
        from kiln.tools import ToolCallError, ToolTimeout
        from kiln import tools
        def run(x):
            try:
                tools.nested_tool()
            except ToolTimeout:
                return "timeout"
            except ToolCallError:
                return "call_error"
            return "no error"
    """)

    @pytest.mark.asyncio
    async def test_real_nested_timeout_raises_tool_timeout(self, tmp_path):
        project = _make_project(tmp_path)
        nested = self._nested_code_tool(
            project,
            "import time\ndef run():\n    time.sleep(30)\n",
            timeout_seconds=1,
        )
        outer_ct = _make_code_tool(
            self.OUTER_CODE, tool_allowlist=["kiln_tool::add_numbers"]
        )
        outer_ct.parent = project
        outer = PythonCodeTool(outer_ct, project)
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=nested,
        ):
            result = await outer.run(None, x="test")
        assert not result.is_error
        assert result.output == "timeout"

    @pytest.mark.asyncio
    async def test_failure_text_mentioning_timeout_raises_call_error(self, tmp_path):
        # An ordinary failure whose message merely contains "timed out" must
        # not spoof the timeout kind (which callers treat as retryable).
        project = _make_project(tmp_path)
        nested = self._nested_code_tool(
            project,
            'def run():\n    raise Exception("upstream request timed out after 3 retries")\n',
        )
        outer_ct = _make_code_tool(
            self.OUTER_CODE, tool_allowlist=["kiln_tool::add_numbers"]
        )
        outer_ct.parent = project
        outer = PythonCodeTool(outer_ct, project)
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=nested,
        ):
            result = await outer.run(None, x="test")
        assert not result.is_error
        assert result.output == "call_error"


class TestCrash:
    @pytest.mark.asyncio
    async def test_crash_via_os_exit(self, tmp_path):
        tool = _make_python_code_tool(
            tmp_path,
            "import os\ndef run(x):\n    os._exit(3)\n",
        )
        result = await tool.run(None, x="test")
        assert result.is_error
        assert "crashed" in result.output
        assert "exit code" in result.output


class TestDepthCap:
    @pytest.mark.asyncio
    async def test_depth_cap_at_10(self, tmp_path):
        """Depth >= 10 returns an error without spawning."""
        tool = _make_python_code_tool(tmp_path, 'def run(x):\n    return "ok"\n')
        token = _depth.set(10)
        try:
            result = await tool.run(None, x="test")
        finally:
            _depth.reset(token)
        assert result.is_error
        assert "maximum nested code execution depth exceeded" in result.output


class TestSemaphore:
    @pytest.mark.asyncio
    async def test_semaphore_top_level_only_no_deadlock(self, tmp_path):
        """Regression: nested code-tool calls bypass the semaphore.

        If nested calls counted against the semaphore, 8 parents each
        spawning a nested code-tool child would deadlock (parents hold
        all 8 slots, children wait forever).

        This test sets MAX_CONCURRENCY parents running concurrently,
        each at depth 1 (simulating nested calls). All should complete
        without deadlock because nested calls bypass the semaphore.
        """
        code = 'def run(x):\n    return "nested_ok"\n'
        results = []

        async def run_nested(i: int):
            tool = _make_python_code_tool(tmp_path, code)
            token = _depth.set(1)
            try:
                r = await tool.run(None, x=str(i))
                results.append(r)
            finally:
                _depth.reset(token)

        await asyncio.gather(
            *(run_nested(i) for i in range(CODE_SANDBOX_MAX_CONCURRENCY))
        )
        assert len(results) == CODE_SANDBOX_MAX_CONCURRENCY
        assert all(not r.is_error for r in results)


class TestToolCallRecorder:
    @pytest.mark.asyncio
    async def test_recorder_gets_entries(self, tmp_path):
        fake = FakeTool(
            "kiln_tool::add_numbers",
            "fake_add",
            params=EMPTY_SCHEMA,
            result=ToolCallResult(output="42"),
        )
        log: list[ToolCallLogEntry] = []
        code = textwrap.dedent("""\
            from kiln import tools
            def run(x):
                r = tools.fake_add()
                return r
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
            tool_call_recorder=log.append,
        )
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=fake,
        ):
            result = await tool.run(None, x="test")
        assert not result.is_error
        assert len(log) == 1
        assert log[0].tool_name == "fake_add"
        assert not log[0].is_error
        assert log[0].output_preview == "42"
        assert log[0].duration_ms >= 0


class TestAsyncToolsConcurrency:
    @pytest.mark.asyncio
    async def test_async_tools_gather_truly_concurrent(self, tmp_path):
        """async_tools + gather provides real parallelism via to_thread.

        Two fake tools each take ~0.3s. If sequential, wall clock >= 0.6s.
        With true concurrency via gather + to_thread, wall clock < 0.6s.
        """
        slow_fake = FakeTool(
            "kiln_tool::add_numbers",
            "fake_add",
            params=EMPTY_SCHEMA,
            result=ToolCallResult(output="done"),
            delay=0.3,
        )
        code = textwrap.dedent("""\
            import asyncio, time
            from kiln import async_tools
            async def run(x):
                start = time.monotonic()
                a, b = await asyncio.gather(
                    async_tools.fake_add(),
                    async_tools.fake_add(),
                )
                elapsed = time.monotonic() - start
                return f"{elapsed:.2f}"
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=slow_fake,
        ):
            result = await tool.run(None, x="test")
        assert not result.is_error
        elapsed = float(result.output)
        assert elapsed < 0.6, f"Expected < 0.6s (concurrent), got {elapsed:.2f}s"


class _EchoFakeTool(KilnToolInterface):
    """Fake tool that echoes its kwargs back, proving per-call routing."""

    def __init__(self, tool_id: str, fn_name: str, params: dict):
        self._id = tool_id
        self._name = fn_name
        self._params = params

    async def id(self):
        return self._id

    async def name(self):
        return self._name

    async def description(self):
        return "echo"

    async def toolcall_definition(self) -> ToolCallDefinition:
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": "echo",
                "parameters": self._params,
            },
        }

    async def run(self, context=None, **kwargs) -> ToolCallResult:
        return ToolCallResult(output=json.dumps(kwargs, sort_keys=True))


class TestCallIdRouting:
    @pytest.mark.asyncio
    async def test_concurrent_calls_routed_to_correct_caller(self, tmp_path):
        """4 threads each pass a unique idx kwarg; each gets its own value back.

        The echo tool returns the kwargs it received. Each thread asserts it
        got back the idx it sent, proving call_id routing maps the right
        response to the right waiting caller under concurrency.
        """
        idx_schema = {
            "type": "object",
            "properties": {"idx": {"type": "string"}},
            "required": ["idx"],
        }
        code = textwrap.dedent("""\
            import json, threading
            from kiln import tools
            def run(x):
                results = [None] * 4
                errors = []
                def call_tool(i):
                    try:
                        raw = tools.fake_echo(idx=str(i))
                        results[i] = json.loads(raw)["idx"]
                    except Exception as e:
                        errors.append(f"thread {i}: {e}")
                threads = [threading.Thread(target=call_tool, args=(i,)) for i in range(4)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                if errors:
                    return "errors: " + str(errors)
                return ",".join(results)
        """)
        fake = _EchoFakeTool(
            "kiln_tool::add_numbers",
            "fake_echo",
            params=idx_schema,
        )
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=fake,
        ):
            result = await tool.run(None, x="test")
        assert not result.is_error
        parts = result.output.split(",")
        assert len(parts) == 4
        assert parts == ["0", "1", "2", "3"]


class TestUnicodePassthrough:
    @pytest.mark.asyncio
    async def test_unicode_not_escaped(self, tmp_path):
        """ensure_ascii=False: non-ASCII chars pass through un-escaped."""
        tool = _make_python_code_tool(
            tmp_path,
            'def run(x):\n    return {"name": "\\u65e5\\u672c\\u8a9e"}\n',
        )
        result = await tool.run(None, x="test")
        assert not result.is_error
        parsed = json.loads(result.output)
        assert parsed["name"] == "日本語"
        assert "\\u" not in result.output


class TestSpawnLockIdentity:
    def test_spawn_lock_shared(self):
        """Code tools and code evals spawn through the same bridge / _spawn_lock."""
        from kiln_ai.tools import sandbox_bridge

        assert (
            sandbox_bridge.start_process_with_light_main.__module__
            == "kiln_ai.sandbox.spawn"
        )
        from kiln_ai.sandbox.spawn import _spawn_lock as shared_lock

        assert shared_lock is _spawn_lock


# ---------------------------------------------------------------------------
# End-to-end tests using REAL built-in tools (no mocking)
# ---------------------------------------------------------------------------


class TestRealBuiltInTools:
    """Tests that exercise real built-in tool dispatch without mocking
    ``tool_from_id_and_project``, ensuring the name-derivation path is
    exercised end-to-end.
    """

    @pytest.mark.asyncio
    async def test_keyword_call_by_canonical_name_succeeds(self, tmp_path):
        """tools.add(a=1, b=2) returns '3' — the canonical name from list_tools."""
        code = textwrap.dedent("""\
            from kiln import tools
            def run(x):
                return tools.add(a=1, b=2)
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        result = await tool.run(None, x="test")
        assert not result.is_error, f"Expected success, got: {result.output}"
        assert result.output == "3"

    @pytest.mark.asyncio
    async def test_list_tools_driven_call_succeeds(self, tmp_path):
        """Call using the name returned by list_tools() succeeds."""
        code = textwrap.dedent("""\
            from kiln import tools
            def run(x):
                tl = tools.list_tools()
                fn_name = tl[0]["name"]
                result = getattr(tools, fn_name)(a=5, b=3)
                return fn_name + ":" + result
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        result = await tool.run(None, x="test")
        assert not result.is_error, f"Expected success, got: {result.output}"
        assert result.output == "add:8"

    @pytest.mark.asyncio
    async def test_friendly_name_not_allowed(self, tmp_path):
        """tools.Addition(a=1,b=2) raises ToolNotAllowed listing canonical names."""
        code = textwrap.dedent("""\
            from kiln.tools import ToolNotAllowed
            from kiln import tools
            def run(x):
                try:
                    tools.Addition(a=1, b=2)
                except ToolNotAllowed as e:
                    return e.message
                return "no error"
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        result = await tool.run(None, x="test")
        assert not result.is_error
        assert "not available" in result.output
        assert "'add'" in result.output

    @pytest.mark.asyncio
    async def test_nonsense_name_not_allowed(self, tmp_path):
        """tools.bad_tool() raises ToolNotAllowed listing available names."""
        code = textwrap.dedent("""\
            from kiln.tools import ToolNotAllowed
            from kiln import tools
            def run(x):
                try:
                    tools.bad_tool(a=1)
                except ToolNotAllowed as e:
                    return e.message
                return "no error"
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        result = await tool.run(None, x="test")
        assert not result.is_error
        assert "not available" in result.output
        assert "'add'" in result.output

    @pytest.mark.asyncio
    async def test_positional_args_error_message(self, tmp_path):
        """tools.add(1, 2) raises ToolCallError mentioning keyword args and params."""
        code = textwrap.dedent("""\
            from kiln.tools import ToolCallError
            from kiln import tools
            def run(x):
                try:
                    tools.add(1, 2)
                except ToolCallError as e:
                    return e.message
                return "no error"
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        result = await tool.run(None, x="test")
        assert not result.is_error
        assert "keyword arguments" in result.output
        assert "tools.add(" in result.output
        assert "a: number (required)" in result.output
        assert "b: number (required)" in result.output

    @pytest.mark.asyncio
    async def test_wrong_kwargs_error_shows_schema(self, tmp_path):
        """tools.add(x=1) raises ToolCallError showing expected parameters."""
        code = textwrap.dedent("""\
            from kiln.tools import ToolCallError
            from kiln import tools
            def run(x):
                try:
                    tools.add(x=1)
                except ToolCallError as e:
                    return e.message
                return "no error"
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        result = await tool.run(None, x="test")
        assert not result.is_error
        assert "Expected parameters:" in result.output
        assert "a: number (required)" in result.output

    @pytest.mark.asyncio
    async def test_name_consistency_across_all_builtins(self, tmp_path):
        """For every KilnBuiltInToolId the dispatch-map name matches tool.name()
        AND matches what list_tools reports."""
        from kiln_ai.datamodel.tool_id import KilnBuiltInToolId
        from kiln_ai.tools.tool_registry import tool_from_id_and_project

        project = _make_project(tmp_path)

        math_ids = [
            KilnBuiltInToolId.ADD_NUMBERS,
            KilnBuiltInToolId.SUBTRACT_NUMBERS,
            KilnBuiltInToolId.MULTIPLY_NUMBERS,
            KilnBuiltInToolId.DIVIDE_NUMBERS,
        ]

        for builtin_id in math_ids:
            tool_id = builtin_id.value
            real_tool = tool_from_id_and_project(tool_id, project=project)
            real_name = await real_tool.name()

            ct = _make_code_tool(
                'def run(x): return "ok"',
                tool_allowlist=[tool_id],
            )
            ct.parent = project
            server = NestedToolServer(
                allowlist=ct.tool_allowlist, project=project, task=None, context=None
            )
            dispatch_names = list((await server.name_map()).keys())

            assert dispatch_names == [real_name], (
                f"For {builtin_id}: dispatch name {dispatch_names} != "
                f"tool.name() '{real_name}'"
            )

    @pytest.mark.asyncio
    async def test_async_proxy_keyword_call(self, tmp_path):
        """async_tools.subtract(a=5, b=3) returns '2'."""
        code = textwrap.dedent("""\
            from kiln import async_tools
            async def run(x):
                return await async_tools.subtract(a=5, b=3)
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::subtract_numbers"],
        )
        result = await tool.run(None, x="test")
        assert not result.is_error, f"Expected success, got: {result.output}"
        assert result.output == "2"

    @pytest.mark.asyncio
    async def test_async_proxy_positional_error(self, tmp_path):
        """async_tools.add(1, 2) raises ToolCallError with a helpful message."""
        code = textwrap.dedent("""\
            from kiln.tools import ToolCallError
            from kiln import async_tools
            async def run(x):
                try:
                    await async_tools.add(1, 2)
                except ToolCallError as e:
                    return e.message
                return "no error"
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        result = await tool.run(None, x="test")
        assert not result.is_error
        assert "keyword arguments" in result.output

    @pytest.mark.asyncio
    async def test_positional_on_nonsense_name_still_not_allowed(self, tmp_path):
        """tools.bad_tool(1) raises ToolNotAllowed (not TypeError), regardless of args."""
        code = textwrap.dedent("""\
            from kiln.tools import ToolNotAllowed
            from kiln import tools
            def run(x):
                try:
                    tools.bad_tool(1, 2)
                except ToolNotAllowed as e:
                    return e.message
                return "no error"
        """)
        tool = _make_python_code_tool(
            tmp_path,
            code,
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        result = await tool.run(None, x="test")
        assert not result.is_error
        assert "not available" in result.output


# ---------------------------------------------------------------------------
# UI example validation tests
#
# These tests execute the EXACT code strings shown in the "Code Tool Examples"
# modal (app/web_ui/src/lib/utils/code_tool_helpers.ts → generateExamples()).
# If you change those examples, you MUST update these tests to match.
# ---------------------------------------------------------------------------

# The example code strings are duplicated here intentionally so any drift
# between the UI and these tests causes a test failure during review.

EXAMPLE_PARALLEL_WITH_RETRIES = """\
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from kiln import tools

def run(urls: list[str], max_retries: int = 3) -> str:
    \"\"\"Fetch multiple URLs in parallel with retries.\"\"\"
    results = {}

    def fetch_with_retry(url):
        for attempt in range(max_retries):
            try:
                result = tools.fetch_url(url=url)
                return url, json.loads(result)
            except Exception as e:
                if attempt == max_retries - 1:
                    return url, {"error": str(e)}
                time.sleep(0.5 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(fetch_with_retry, u) for u in urls]
        for future in as_completed(futures):
            url, data = future.result()
            results[url] = data

    return json.dumps(results)
"""

EXAMPLE_ASYNC_FAN_OUT = """\
import json
import asyncio
from kiln import async_tools

async def run(user_ids: list[str]) -> str:
    \"\"\"Fetch user details concurrently using async_tools.\"\"\"
    async def fetch_user(uid):
        result = await async_tools.get_user(id=uid)
        return json.loads(result)

    users = await asyncio.gather(*(fetch_user(uid) for uid in user_ids))
    return json.dumps(users)
"""

EXAMPLE_FILTER_AND_TRANSFORM = """\
import json
from kiln import tools

def run(query: str, max_results: int = 10) -> str:
    \"\"\"Search and filter results, returning only relevant fields.\"\"\"
    raw = tools.search(query=query)
    results = json.loads(raw)

    filtered = [
        {"title": r["title"], "url": r["url"]}
        for r in results[:max_results]
        if "title" in r and "url" in r
    ]

    return json.dumps(filtered)
"""


class TestUIExampleParallelWithRetries:
    """Validate the 'Parallel with Retries' example from the Code Tool Examples modal."""

    @pytest.mark.asyncio
    async def test_parallel_with_retries_happy_path(self, tmp_path):
        fetch_url_responses = {
            "https://a.com": '{"status": "ok_a"}',
            "https://b.com": '{"status": "ok_b"}',
        }
        fake = FakeTool(
            "mcp::remote::test_server::fetch_url",
            "fetch_url",
            fn_desc="Fetch a URL",
            params={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        )

        async def route_fetch(context=None, **kwargs):
            url = kwargs["url"]
            return ToolCallResult(output=fetch_url_responses[url])

        fake.run = route_fetch  # type: ignore[assignment]

        tool = _make_python_code_tool(
            tmp_path,
            EXAMPLE_PARALLEL_WITH_RETRIES,
            tool_allowlist=["mcp::remote::test_server::fetch_url"],
            parameters_schema={
                "type": "object",
                "properties": {
                    "urls": {"type": "array", "items": {"type": "string"}},
                    "max_retries": {"type": "integer"},
                },
                "required": ["urls"],
            },
        )
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=fake,
        ):
            result = await tool.run(
                None, urls=["https://a.com", "https://b.com"], max_retries=1
            )
        assert not result.is_error, f"Expected success, got: {result.output}"
        parsed = json.loads(result.output)
        assert parsed["https://a.com"] == {"status": "ok_a"}
        assert parsed["https://b.com"] == {"status": "ok_b"}

    @pytest.mark.asyncio
    async def test_parallel_with_retries_error_fallback(self, tmp_path):
        """When a tool call fails, the retry logic catches the exception and
        returns an error dict after exhausting retries."""
        fake = FakeTool(
            "mcp::remote::test_server::fetch_url",
            "fetch_url",
            fn_desc="Fetch a URL",
            params={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            result=ToolCallResult(
                output="connection refused",
                is_error=True,
                error_message="connection refused",
            ),
        )
        tool = _make_python_code_tool(
            tmp_path,
            EXAMPLE_PARALLEL_WITH_RETRIES,
            tool_allowlist=["mcp::remote::test_server::fetch_url"],
            parameters_schema={
                "type": "object",
                "properties": {
                    "urls": {"type": "array", "items": {"type": "string"}},
                    "max_retries": {"type": "integer"},
                },
                "required": ["urls"],
            },
        )
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=fake,
        ):
            result = await tool.run(None, urls=["https://fail.com"], max_retries=1)
        assert not result.is_error, f"Expected success, got: {result.output}"
        parsed = json.loads(result.output)
        assert "error" in parsed["https://fail.com"]


class TestUIExampleAsyncFanOut:
    """Validate the 'Async Fan-Out' example from the Code Tool Examples modal."""

    @pytest.mark.asyncio
    async def test_async_fan_out_happy_path(self, tmp_path):
        user_data = {
            "u1": '{"name": "Alice", "id": "u1"}',
            "u2": '{"name": "Bob", "id": "u2"}',
        }
        fake = FakeTool(
            "mcp::remote::test_server::get_user",
            "get_user",
            fn_desc="Get user details",
            params={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        )

        async def route_user(context=None, **kwargs):
            uid = kwargs["id"]
            return ToolCallResult(output=user_data[uid])

        fake.run = route_user  # type: ignore[assignment]

        tool = _make_python_code_tool(
            tmp_path,
            EXAMPLE_ASYNC_FAN_OUT,
            tool_allowlist=["mcp::remote::test_server::get_user"],
            parameters_schema={
                "type": "object",
                "properties": {
                    "user_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["user_ids"],
            },
        )
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=fake,
        ):
            result = await tool.run(None, user_ids=["u1", "u2"])
        assert not result.is_error, f"Expected success, got: {result.output}"
        parsed = json.loads(result.output)
        assert len(parsed) == 2
        assert parsed[0] == {"name": "Alice", "id": "u1"}
        assert parsed[1] == {"name": "Bob", "id": "u2"}


class TestUIExampleFilterAndTransform:
    """Validate the 'Filter & Transform' example from the Code Tool Examples modal."""

    @pytest.mark.asyncio
    async def test_filter_and_transform_happy_path(self, tmp_path):
        search_results = json.dumps(
            [
                {"title": "Result 1", "url": "https://1.com", "score": 0.9},
                {"title": "Result 2", "url": "https://2.com", "score": 0.8},
                {"description": "no title or url"},
                {"title": "Result 3", "url": "https://3.com", "score": 0.7},
            ]
        )
        fake = FakeTool(
            "mcp::remote::test_server::search",
            "search",
            fn_desc="Search",
            params={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            result=ToolCallResult(output=search_results),
        )
        tool = _make_python_code_tool(
            tmp_path,
            EXAMPLE_FILTER_AND_TRANSFORM,
            tool_allowlist=["mcp::remote::test_server::search"],
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
        )
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=fake,
        ):
            result = await tool.run(None, query="test query")
        assert not result.is_error, f"Expected success, got: {result.output}"
        parsed = json.loads(result.output)
        assert len(parsed) == 3
        assert parsed[0] == {"title": "Result 1", "url": "https://1.com"}
        assert parsed[1] == {"title": "Result 2", "url": "https://2.com"}
        assert parsed[2] == {"title": "Result 3", "url": "https://3.com"}

    @pytest.mark.asyncio
    async def test_filter_and_transform_respects_max_results(self, tmp_path):
        search_results = json.dumps(
            [{"title": f"R{i}", "url": f"https://{i}.com"} for i in range(20)]
        )
        fake = FakeTool(
            "mcp::remote::test_server::search",
            "search",
            fn_desc="Search",
            params={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            result=ToolCallResult(output=search_results),
        )
        tool = _make_python_code_tool(
            tmp_path,
            EXAMPLE_FILTER_AND_TRANSFORM,
            tool_allowlist=["mcp::remote::test_server::search"],
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
        )
        with patch(
            "kiln_ai.tools.tool_registry.tool_from_id_and_project",
            return_value=fake,
        ):
            result = await tool.run(None, query="test", max_results=3)
        assert not result.is_error, f"Expected success, got: {result.output}"
        parsed = json.loads(result.output)
        assert len(parsed) == 3
