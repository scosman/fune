from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kiln_ai.datamodel.code_tool import CodeTool
from kiln_ai.datamodel.project import Project
from kiln_ai.tools.code_tool import ChildOutcome, PythonCodeTool
from kiln_ai.tools.sandbox_bridge import ToolCallLogEntry
from kiln_server.custom_errors import connect_custom_errors

from app.desktop.studio_server.code_tool_api import connect_code_tool_api
from app.desktop.studio_server.tool_api import connect_tool_servers_api

SIMPLE_CODE = "def run(x: int) -> str:\n    return str(x * 2)\n"
SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "integer"}},
    "required": ["x"],
}

TRUST_PATCH = "app.desktop.studio_server.code_tool_api.has_add_code_trust"


@pytest.fixture
def app():
    test_app = FastAPI()
    connect_custom_errors(test_app)
    connect_code_tool_api(test_app)
    connect_tool_servers_api(test_app)
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def test_project(tmp_path):
    project_path = tmp_path / "test_project" / "project.kiln"
    project_path.parent.mkdir()
    project = Project(name="Test Project", path=project_path)
    project.save_to_file()
    return project


@pytest.fixture
def mock_project_from_id(test_project):
    with (
        patch(
            "app.desktop.studio_server.code_tool_api.project_from_id",
            return_value=test_project,
        ),
        patch(
            "app.desktop.studio_server.tool_api.project_from_id",
            return_value=test_project,
        ),
    ):
        yield


@pytest.fixture
def create_request():
    return {
        "name": "double_it",
        "tool_function_name": "double_it",
        "tool_description": "Doubles a number",
        "parameters_schema": SIMPLE_SCHEMA,
        "code": SIMPLE_CODE,
    }


@pytest.fixture
def saved_code_tool(test_project):
    ct = CodeTool(
        name="my_tool",
        tool_function_name="my_tool",
        tool_description="Does something",
        parameters_schema=SIMPLE_SCHEMA,
        code=SIMPLE_CODE,
        parent=test_project,
    )
    ct.save_to_file()
    return ct


class TestCreateCodeTool:
    def test_create_success(
        self, client, test_project, mock_project_from_id, create_request
    ):
        with patch(TRUST_PATCH, return_value=True):
            response = client.post(
                f"/api/projects/{test_project.id}/code_tools",
                json=create_request,
            )
        assert response.status_code == 200
        result = response.json()
        assert result["tool_function_name"] == "double_it"
        assert result["tool_description"] == "Doubles a number"
        assert result["id"] is not None

        loaded = CodeTool.from_id_and_parent_path(result["id"], test_project.path)
        assert loaded is not None
        assert loaded.tool_function_name == "double_it"

    def test_create_not_trusted(
        self, client, test_project, mock_project_from_id, create_request
    ):
        with patch(TRUST_PATCH, return_value=False):
            response = client.post(
                f"/api/projects/{test_project.id}/code_tools",
                json=create_request,
            )
        assert response.status_code == 200
        result = response.json()
        assert result["not_trusted"] is True
        assert result["id"] is None

    def test_create_allows_duplicate_function_names(
        self,
        client,
        test_project,
        mock_project_from_id,
        saved_code_tool,
        create_request,
    ):
        # Duplicate function names are allowed within a project; collisions are
        # rejected per run config at run config creation time instead.
        create_request["tool_function_name"] = saved_code_tool.tool_function_name
        with patch(TRUST_PATCH, return_value=True):
            response = client.post(
                f"/api/projects/{test_project.id}/code_tools",
                json=create_request,
            )
        assert response.status_code == 200
        assert response.json()["id"] is not None

    def test_create_rejects_ambiguous_allowlist(
        self, client, test_project, mock_project_from_id, create_request
    ):
        # Two allowlisted tools sharing a function name would make calls from
        # sandboxed code ambiguous: rejected at creation.
        allowlisted = []
        for _ in range(2):
            ct = CodeTool(
                name="dup_tool",
                tool_function_name="dup_tool",
                tool_description="d",
                parameters_schema=SIMPLE_SCHEMA,
                code=SIMPLE_CODE,
                parent=test_project,
            )
            ct.save_to_file()
            allowlisted.append(f"kiln_tool::code::{ct.id}")

        create_request["tool_allowlist"] = allowlisted
        with patch(TRUST_PATCH, return_value=True):
            response = client.post(
                f"/api/projects/{test_project.id}/code_tools",
                json=create_request,
            )
        assert response.status_code == 400
        assert "share the same function name: dup_tool" in response.json()["message"]

        # A single one of them is fine.
        create_request["tool_allowlist"] = allowlisted[:1]
        with patch(TRUST_PATCH, return_value=True):
            response = client.post(
                f"/api/projects/{test_project.id}/code_tools",
                json=create_request,
            )
        assert response.status_code == 200

    def test_create_validation_error(self, client, test_project, mock_project_from_id):
        with patch(TRUST_PATCH, return_value=True):
            response = client.post(
                f"/api/projects/{test_project.id}/code_tools",
                json={
                    "name": "bad",
                    "tool_function_name": "bad",
                    "tool_description": "test",
                    "parameters_schema": SIMPLE_SCHEMA,
                    "code": "x = 1",
                },
            )
        assert response.status_code == 400
        assert "run" in response.json()["message"].lower()

    def test_create_reserved_word_param_rejected(
        self, client, test_project, mock_project_from_id, create_request
    ):
        create_request["parameters_schema"] = {
            "type": "object",
            "properties": {"from": {"type": "string"}},
        }
        with patch(TRUST_PATCH, return_value=True):
            response = client.post(
                f"/api/projects/{test_project.id}/code_tools",
                json=create_request,
            )
        assert response.status_code == 400
        message = response.json()["message"]
        assert "'from'" in message
        assert "reserved Python keyword" in message


class TestListCodeTools:
    def test_list_empty(self, client, test_project, mock_project_from_id):
        response = client.get(f"/api/projects/{test_project.id}/code_tools")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_archived_sorted_last(
        self, client, test_project, mock_project_from_id
    ):
        ct1 = CodeTool(
            name="active_tool",
            tool_function_name="active_tool",
            tool_description="Active",
            parameters_schema=SIMPLE_SCHEMA,
            code=SIMPLE_CODE,
            parent=test_project,
        )
        ct1.save_to_file()
        ct2 = CodeTool(
            name="archived_tool",
            tool_function_name="archived_tool",
            tool_description="Archived",
            parameters_schema=SIMPLE_SCHEMA,
            code=SIMPLE_CODE,
            is_archived=True,
            parent=test_project,
        )
        ct2.save_to_file()

        response = client.get(f"/api/projects/{test_project.id}/code_tools")
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 2
        assert results[0]["is_archived"] is False
        assert results[1]["is_archived"] is True


class TestGetCodeTool:
    def test_get_success(
        self, client, test_project, mock_project_from_id, saved_code_tool
    ):
        response = client.get(
            f"/api/projects/{test_project.id}/code_tools/{saved_code_tool.id}"
        )
        assert response.status_code == 200
        result = response.json()
        assert result["tool_function_name"] == "my_tool"
        assert result["code"] == SIMPLE_CODE

    def test_get_not_found(self, client, test_project, mock_project_from_id):
        response = client.get(f"/api/projects/{test_project.id}/code_tools/nonexistent")
        assert response.status_code == 404


class TestUpdateCodeTool:
    def test_patch_metadata(
        self, client, test_project, mock_project_from_id, saved_code_tool
    ):
        response = client.patch(
            f"/api/projects/{test_project.id}/code_tools/{saved_code_tool.id}",
            json={"name": "renamed_tool", "description": "New desc"},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "renamed_tool"
        assert result["description"] == "New desc"
        assert result["tool_function_name"] == "my_tool"
        assert result["code"] == SIMPLE_CODE

    def test_patch_preserves_functional_fields(
        self, client, test_project, mock_project_from_id, saved_code_tool
    ):
        response = client.patch(
            f"/api/projects/{test_project.id}/code_tools/{saved_code_tool.id}",
            json={"name": "new_name"},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["code"] == SIMPLE_CODE
        assert result["tool_function_name"] == "my_tool"
        assert result["parameters_schema"] == SIMPLE_SCHEMA
        assert result["timeout_seconds"] == 60

    def test_patch_no_changes(
        self, client, test_project, mock_project_from_id, saved_code_tool
    ):
        response = client.patch(
            f"/api/projects/{test_project.id}/code_tools/{saved_code_tool.id}",
            json={},
        )
        assert response.status_code == 200

    def test_patch_rejects_functional_fields(
        self, client, test_project, mock_project_from_id, saved_code_tool
    ):
        response = client.patch(
            f"/api/projects/{test_project.id}/code_tools/{saved_code_tool.id}",
            json={"name": "ok", "code": "def run(): pass"},
        )
        assert response.status_code == 422

        reloaded = CodeTool.from_id_and_parent_path(
            saved_code_tool.id, test_project.path
        )
        assert reloaded is not None
        assert reloaded.code == SIMPLE_CODE
        assert reloaded.name == "my_tool"


class TestArchiveCodeTool:
    def test_archive_unarchive(
        self, client, test_project, mock_project_from_id, saved_code_tool
    ):
        response = client.post(
            f"/api/projects/{test_project.id}/code_tools/{saved_code_tool.id}/archive",
            json={"archived": True},
        )
        assert response.status_code == 200
        assert response.json()["is_archived"] is True

        response = client.post(
            f"/api/projects/{test_project.id}/code_tools/{saved_code_tool.id}/archive",
            json={"archived": False},
        )
        assert response.status_code == 200
        assert response.json()["is_archived"] is False


class TestDeleteCodeTool:
    def test_delete_success(
        self, client, test_project, mock_project_from_id, saved_code_tool
    ):
        response = client.delete(
            f"/api/projects/{test_project.id}/code_tools/{saved_code_tool.id}"
        )
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        loaded = CodeTool.from_id_and_parent_path(saved_code_tool.id, test_project.path)
        assert loaded is None

    def test_delete_not_found(self, client, test_project, mock_project_from_id):
        response = client.delete(
            f"/api/projects/{test_project.id}/code_tools/nonexistent"
        )
        assert response.status_code == 404


class TestTestCodeTool:
    def _test_url(self, project_id: str) -> str:
        return f"/api/projects/{project_id}/test_code_tool"

    def _base_request(self) -> dict:
        return {
            "tool_function_name": "double_it",
            "parameters_schema": SIMPLE_SCHEMA,
            "code": SIMPLE_CODE,
            "params": {"x": 5},
        }

    def test_success(self, client, test_project, mock_project_from_id):
        outcome = ChildOutcome(ok="10", stdout="", stderr="", duration_ms=42)

        with (
            patch.object(
                PythonCodeTool,
                "_invoke",
                new_callable=AsyncMock,
                return_value=outcome,
            ),
            patch(TRUST_PATCH, return_value=True),
        ):
            response = client.post(
                self._test_url(test_project.id), json=self._base_request()
            )

        assert response.status_code == 200
        body = response.json()
        assert body["result"] == "10"
        assert body["error"] is None
        assert body["duration_ms"] == 42

    def test_validation_error_bad_code(
        self, client, test_project, mock_project_from_id
    ):
        req = self._base_request()
        req["code"] = "x = 1"

        response = client.post(self._test_url(test_project.id), json=req)

        assert response.status_code == 400
        assert "run" in response.json()["message"].lower()

    def test_params_validation_error(self, client, test_project, mock_project_from_id):
        req = self._base_request()
        req["params"] = {"x": "not_an_int"}

        with patch(TRUST_PATCH, return_value=True):
            response = client.post(self._test_url(test_project.id), json=req)

        assert response.status_code == 400
        assert "schema" in response.json()["message"].lower()

    def test_not_trusted(self, client, test_project, mock_project_from_id):
        with patch(TRUST_PATCH, return_value=False):
            response = client.post(
                self._test_url(test_project.id), json=self._base_request()
            )

        assert response.status_code == 200
        body = response.json()
        assert body["not_trusted"] is True
        assert body["result"] is None

    def test_error_result(self, client, test_project, mock_project_from_id):
        outcome = ChildOutcome(
            error="NameError: name 'foo' is not defined",
            traceback_str='File "<code_tool>", line 2\n    foo()',
            stdout="debug output",
            stderr="",
            duration_ms=10,
        )

        with (
            patch.object(
                PythonCodeTool,
                "_invoke",
                new_callable=AsyncMock,
                return_value=outcome,
            ),
            patch(TRUST_PATCH, return_value=True),
        ):
            response = client.post(
                self._test_url(test_project.id), json=self._base_request()
            )

        assert response.status_code == 200
        body = response.json()
        assert body["result"] is None
        assert "NameError" in body["error"]
        assert body["traceback"] is not None
        assert body["stdout"] == "debug output"

    def test_nothing_persisted(self, client, test_project, mock_project_from_id):
        project_dir = test_project.path.parent
        files_before = set(str(f) for f in project_dir.rglob("*"))

        outcome = ChildOutcome(ok="result", duration_ms=1)

        with (
            patch.object(
                PythonCodeTool,
                "_invoke",
                new_callable=AsyncMock,
                return_value=outcome,
            ),
            patch(TRUST_PATCH, return_value=True),
        ):
            client.post(self._test_url(test_project.id), json=self._base_request())

        files_after = set(str(f) for f in project_dir.rglob("*"))
        new_files = files_after - files_before
        assert len(new_files) == 0, f"Unexpected new files created: {new_files}"

    def test_mcp_cleanup(self, client, test_project, mock_project_from_id):
        outcome = ChildOutcome(ok="ok", duration_ms=1)
        cleanup_mock = AsyncMock()

        with (
            patch.object(
                PythonCodeTool,
                "_invoke",
                new_callable=AsyncMock,
                return_value=outcome,
            ),
            patch(TRUST_PATCH, return_value=True),
            patch(
                "kiln_ai.tools.mcp_session_manager.MCPSessionManager"
            ) as mock_manager_cls,
            patch("kiln_ai.tools.mcp_session_manager.clear_agent_run_id") as mock_clear,
        ):
            mock_manager_cls.shared.return_value.cleanup_session = cleanup_mock
            response = client.post(
                self._test_url(test_project.id), json=self._base_request()
            )

        assert response.status_code == 200
        cleanup_mock.assert_called_once()
        mock_clear.assert_called_once()

    def test_mcp_cleanup_on_error(self, app, test_project, mock_project_from_id):
        cleanup_mock = AsyncMock()
        error_client = TestClient(app, raise_server_exceptions=False)

        with (
            patch.object(
                PythonCodeTool,
                "_invoke",
                new_callable=AsyncMock,
                side_effect=RuntimeError("spawn failed"),
            ),
            patch(TRUST_PATCH, return_value=True),
            patch(
                "kiln_ai.tools.mcp_session_manager.MCPSessionManager"
            ) as mock_manager_cls,
            patch("kiln_ai.tools.mcp_session_manager.clear_agent_run_id") as mock_clear,
        ):
            mock_manager_cls.shared.return_value.cleanup_session = cleanup_mock
            response = error_client.post(
                self._test_url(test_project.id), json=self._base_request()
            )

        assert response.status_code == 500
        cleanup_mock.assert_called_once()
        mock_clear.assert_called_once()

    def test_tool_call_log(self, client, test_project, mock_project_from_id):
        outcome = ChildOutcome(ok="done", duration_ms=50)

        async def mock_invoke(self_ref, context, kwargs):
            if self_ref._tool_call_recorder:
                self_ref._tool_call_recorder(
                    ToolCallLogEntry(
                        tool_name="helper_tool",
                        arguments={"q": "test"},
                        output_preview="result preview",
                        is_error=False,
                        duration_ms=20,
                    )
                )
            return outcome

        with (
            patch.object(PythonCodeTool, "_invoke", mock_invoke),
            patch(TRUST_PATCH, return_value=True),
        ):
            response = client.post(
                self._test_url(test_project.id), json=self._base_request()
            )

        assert response.status_code == 200
        body = response.json()
        assert len(body["tool_call_log"]) == 1
        assert body["tool_call_log"][0]["tool_name"] == "helper_tool"
        assert body["tool_call_log"][0]["is_error"] is False


class TestAvailableToolsCodeGroup:
    def test_includes_code_tools(
        self, client, test_project, mock_project_from_id, saved_code_tool
    ):
        response = client.get(f"/api/projects/{test_project.id}/available_tools")
        assert response.status_code == 200
        tool_sets = response.json()
        code_sets = [ts for ts in tool_sets if ts["type"] == "code"]
        assert len(code_sets) == 1
        tools = code_sets[0]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "my_tool"

    def test_serves_display_name_and_function_name(
        self, client, test_project, mock_project_from_id
    ):
        # name carries the user-facing display name, function_name the
        # callable name — distinct values so a regression can't hide behind
        # a fixture where they coincide.
        ct = CodeTool(
            name="My Fancy Tool",
            tool_function_name="my_fancy_tool",
            tool_description="Does something",
            parameters_schema=SIMPLE_SCHEMA,
            code=SIMPLE_CODE,
            parent=test_project,
        )
        ct.save_to_file()

        response = client.get(f"/api/projects/{test_project.id}/available_tools")
        assert response.status_code == 200
        tool_sets = response.json()
        code_sets = [ts for ts in tool_sets if ts["type"] == "code"]
        assert len(code_sets) == 1
        [tool] = code_sets[0]["tools"]
        assert tool["name"] == "My Fancy Tool"
        assert tool["function_name"] == "my_fancy_tool"

    def test_excludes_archived_code_tools(
        self, client, test_project, mock_project_from_id, saved_code_tool
    ):
        saved_code_tool.is_archived = True
        saved_code_tool.save_to_file()

        response = client.get(f"/api/projects/{test_project.id}/available_tools")
        assert response.status_code == 200
        tool_sets = response.json()
        code_sets = [ts for ts in tool_sets if ts["type"] == "code"]
        assert len(code_sets) == 0
