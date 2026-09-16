import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from kiln_ai.datamodel.datamodel_enums import StructuredOutputMode
from kiln_ai.datamodel.external_tool_server import ExternalToolServer, ToolServerType
from kiln_ai.datamodel.project import Project
from kiln_ai.datamodel.prompt_id import PromptGenerators
from kiln_ai.datamodel.rag import RagConfig
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties
from kiln_ai.datamodel.task import Task, TaskRunConfig
from kiln_ai.datamodel.tool_id import (
    KILN_TASK_TOOL_ID_PREFIX,
    KilnBuiltInToolId,
)
from kiln_ai.tools.mcp_session_manager import KilnMCPError
from kiln_ai.utils.config import MCP_SECRETS_KEY
from kiln_ai.utils.open_ai_types import TASK_RESPONSE_TOOL_NAME
from kiln_server.custom_errors import connect_custom_errors
from mcp.types import ListToolsResult, Tool

from app.desktop.studio_server.tool_api import (
    ExternalToolApiDescription,
    ToolSetType,
    available_mcp_tools,
    connect_tool_servers_api,
    tool_server_from_id,
    validate_tool_server_connectivity,
)


@pytest.fixture
def mock_project_from_id(test_project):
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id",
        return_value=test_project,
    ) as mock:
        yield mock


def create_mcp_session_manager_patch(
    mock_tools=None, connection_error=None, list_tools_error=None
):
    """
    Creates a mock MCP session manager patch with various scenarios.

    Args:
        mock_tools: List of Tool objects to return from list_tools (defaults to empty list)
        connection_error: Exception to raise when creating MCP client connection
        list_tools_error: Exception to raise when calling list_tools

    Returns:
        Context manager that patches MCPSessionManager.shared
    """
    if mock_tools is None:
        mock_tools = []

    @asynccontextmanager
    async def mock_mcp_client(tool_server):
        if connection_error:
            raise connection_error

        mock_session = AsyncMock()
        if list_tools_error:
            mock_session.list_tools.side_effect = list_tools_error
        else:
            mock_session.list_tools.return_value = ListToolsResult(tools=mock_tools)

        yield mock_session

    return patch(
        "app.desktop.studio_server.tool_api.MCPSessionManager.shared"
    ), mock_mcp_client


@asynccontextmanager
async def mock_mcp_success(tools=None):
    """Context manager for successful MCP operations with optional tools."""
    tools = tools or []
    patch_obj, mock_client = create_mcp_session_manager_patch(mock_tools=tools)

    with patch_obj as mock_session_manager_shared:
        mock_session_manager = Mock()
        mock_session_manager.mcp_client = mock_client
        mock_session_manager_shared.return_value = mock_session_manager
        yield


@asynccontextmanager
async def mock_mcp_connection_error(error_message="Connection failed"):
    """Context manager for MCP connection errors."""
    error = KilnMCPError(error_message)
    patch_obj, mock_client = create_mcp_session_manager_patch(connection_error=error)

    with patch_obj as mock_session_manager_shared:
        mock_session_manager = Mock()
        mock_session_manager.mcp_client = mock_client
        mock_session_manager_shared.return_value = mock_session_manager
        yield


@asynccontextmanager
async def mock_mcp_list_tools_error(error_message="list_tools failed"):
    """Context manager for MCP list_tools errors."""
    error = KilnMCPError(error_message)
    patch_obj, mock_client = create_mcp_session_manager_patch(list_tools_error=error)

    with patch_obj as mock_session_manager_shared:
        mock_session_manager = Mock()
        mock_session_manager.mcp_client = mock_client
        mock_session_manager_shared.return_value = mock_session_manager
        yield


@pytest.fixture
def app():
    test_app = FastAPI()
    connect_custom_errors(test_app)
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
def mock_mcp_validation():
    """Fixture that provides a context manager for mocking MCP validation during tool creation"""
    return mock_mcp_success


async def test_create_tool_server_success(client, test_project):
    tool_data = {
        "name": "test_mcp_tool",
        "server_url": "https://example.com/mcp",
        "headers": {"Authorization": "Bearer test-token"},
        "description": "A test MCP tool",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        async with mock_mcp_success():
            response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )

            assert response.status_code == 200
            result = response.json()
            assert result["name"] == "test_mcp_tool"
            assert result["type"] == "remote_mcp"
            assert result["description"] == "A test MCP tool"
            assert result["properties"]["server_url"] == "https://example.com/mcp"
            assert (
                result["properties"]["headers"]["Authorization"] == "Bearer test-token"
            )
            assert "id" in result
            assert "created_at" in result


async def test_create_tool_server_validation_connection_failed(client, test_project):
    """Test tool server creation fails when MCP server is unreachable"""
    tool_data = {
        "name": "failing_tool",
        "server_url": "https://unreachable.example.com/mcp",
        "headers": {},
        "description": "Tool that will fail validation",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        async with mock_mcp_connection_error():
            # Unhandled exception is now raised instead of returning 422
            with pytest.raises(Exception, match="Connection failed"):
                client.post(
                    f"/api/projects/{test_project.id}/connect_remote_mcp",
                    json=tool_data,
                )


async def test_create_tool_server_validation_list_tools_failed(client, test_project):
    """Test tool server creation fails when MCP list_tools fails"""
    tool_data = {
        "name": "list_tools_failing",
        "server_url": "https://example.com/mcp",
        "headers": {},
        "description": "Tool where list_tools fails",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        async with mock_mcp_list_tools_error():
            # Unhandled exception is now raised instead of returning 422
            with pytest.raises(Exception, match="list_tools failed"):
                client.post(
                    f"/api/projects/{test_project.id}/connect_remote_mcp",
                    json=tool_data,
                )


def test_get_available_tool_servers_empty(client, test_project):
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        response = client.get(f"/api/projects/{test_project.id}/available_tool_servers")

        assert response.status_code == 200
        result = response.json()
        assert result == []


async def test_get_available_tool_servers_with_tool_server(client, test_project):
    # First create a tool server
    tool_data = {
        "name": "my_tool",
        "server_url": "https://api.example.com",
        "headers": {"X-API-Key": "secret"},
        "description": "My awesome tool",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        async with mock_mcp_success():
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
        created_tool = create_response.json()

        # Then get the list of tool servers
        response = client.get(f"/api/projects/{test_project.id}/available_tool_servers")

        assert response.status_code == 200
        result = response.json()
        assert len(result) == 1
        assert result[0]["name"] == "my_tool"
        assert result[0]["id"] == created_tool["id"]
        assert result[0]["description"] == "My awesome tool"
        assert "missing_secrets" in result[0]
        assert isinstance(result[0]["missing_secrets"], list)


@pytest.mark.parametrize(
    "server_type, tool_data, endpoint, expected_missing_secrets",
    [
        (
            ToolServerType.remote_mcp,
            {
                "name": "tool_with_missing_secrets",
                "server_url": "https://api.example.com",
                "headers": {"Authorization": "Bearer token", "X-API-Key": "secret"},
                "secret_header_keys": ["Authorization", "X-API-Key"],
                "description": "Tool with missing secrets",
                "is_archived": False,
            },
            "connect_remote_mcp",
            ["Authorization", "X-API-Key"],
        ),
        (
            ToolServerType.local_mcp,
            {
                "name": "local_tool_with_missing_secrets",
                "command": "python",
                "args": ["-m", "my_mcp_server"],
                "env_vars": {
                    "DATABASE_URL": "postgres://localhost",
                    "API_KEY": "secret",
                },
                "secret_env_var_keys": ["API_KEY"],
                "description": "Local tool with missing secrets",
                "is_archived": False,
            },
            "connect_local_mcp",
            ["API_KEY"],
        ),
    ],
)
async def test_get_available_tool_servers_with_missing_secrets(
    client,
    test_project,
    server_type,
    tool_data,
    endpoint,
    expected_missing_secrets,
):
    """Test that get_available_tool_servers includes missing_secrets field"""
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Mock MCPSessionManager for shell path cache clearing
        with patch(
            "app.desktop.studio_server.tool_api.MCPSessionManager.shared"
        ) as mock_session_manager_shared:
            mock_session_manager = Mock()
            mock_session_manager.clear_shell_path_cache = Mock()
            mock_session_manager_shared.return_value = mock_session_manager

            async with mock_mcp_success():
                create_response = client.post(
                    f"/api/projects/{test_project.id}/{endpoint}",
                    json=tool_data,
                )
                assert create_response.status_code == 200
            created_tool = create_response.json()

            mock_tool_server = Mock()
            mock_tool_server.id = created_tool["id"]
            mock_tool_server.name = tool_data["name"]
            mock_tool_server.type = server_type
            mock_tool_server.description = tool_data["description"]
            mock_tool_server.retrieve_secrets.return_value = (
                {},
                expected_missing_secrets,
            )
            mock_tool_server.properties = {"is_archived": False}

            # Mock the project's external_tool_servers method
            mock_project = Mock()
            mock_project.external_tool_servers.return_value = [mock_tool_server]
            mock_project_from_id.return_value = mock_project

            # Get the list of tool servers
            response = client.get(
                f"/api/projects/{test_project.id}/available_tool_servers"
            )

            assert response.status_code == 200
            result = response.json()
            assert len(result) == 1
            assert result[0]["id"] == created_tool["id"]
            assert "missing_secrets" in result[0]
            assert set(result[0]["missing_secrets"]) == set(expected_missing_secrets)


async def test_get_tool_server_success(client, test_project):
    # First create a tool server
    tool_data = {
        "name": "test_get_tool",
        "server_url": "https://example.com/api",
        "headers": {"Authorization": "Bearer token"},
        "description": "Tool for get test",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Create the tool with empty tools
        async with mock_mcp_success():
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
        created_tool = create_response.json()
        tool_server_id = created_tool["id"]

        # Mock retrieval with specific tools
        mock_tools = [
            Tool(name="test_tool_1", description="First test tool", inputSchema={}),
            Tool(name="calculator", description="Math calculations", inputSchema={}),
        ]

        async with mock_mcp_success(tools=mock_tools):
            # Now get the tool server
            response = client.get(
                f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
            )

            assert response.status_code == 200
            result = response.json()
            # Verify the tool server details match what we created
            assert result["name"] == "test_get_tool"
            assert result["type"] == "remote_mcp"
            assert result["description"] == "Tool for get test"
            assert result["properties"]["server_url"] == "https://example.com/api"
            assert result["properties"]["headers"]["Authorization"] == "Bearer token"
            assert "id" in result  # Just verify ID exists, don't check exact value

            # Verify available_tools is populated
            assert "available_tools" in result
            assert len(result["available_tools"]) == 2
            tool_names = [tool["name"] for tool in result["available_tools"]]
            assert "test_tool_1" in tool_names
            assert "calculator" in tool_names


async def test_get_tool_server_mcp_error_handling(client, test_project):
    """Test that MCP server errors are surfaced with HTTP 503 and the error message"""
    # First create a tool server
    tool_data = {
        "name": "failing_mcp_tool",
        "server_url": "https://example.com/api",
        "headers": {},
        "description": "MCP tool that will fail",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Create the tool with successful validation
        async with mock_mcp_success():
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
        created_tool = create_response.json()
        tool_server_id = created_tool["id"]

        # Mock retrieval with list_tools error
        async with mock_mcp_list_tools_error("Connection failed"):
            # The API should surface the error as HTTP 503 with the error message in detail
            response = client.get(
                f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
            )
            assert response.status_code == 503
            result = response.json()
            assert "Connection failed" in result["message"]


def test_get_tool_server_not_found(client, test_project):
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Try to get a non-existent tool server
        response = client.get(
            f"/api/projects/{test_project.id}/tool_servers/nonexistent-tool-server-id"
        )

        assert response.status_code == 404
        result = response.json()
        assert result["message"] == "Tool server not found"


async def test_get_tool_server_config_returns_file_data_without_connecting(
    client, test_project
):
    """The /config endpoint must return tool server data without attempting
    a connection, so the edit form loads even when the MCP server is unreachable."""
    tool_data = {
        "name": "config_test_tool",
        "server_url": "https://example.com/api",
        "headers": {},
        "description": "Tool for config test",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        async with mock_mcp_success():
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
        tool_server_id = create_response.json()["id"]

        # Call /config while the MCP connection is broken — must still succeed
        async with mock_mcp_list_tools_error("Connection failed"):
            response = client.get(
                f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}/config"
            )

        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "config_test_tool"
        assert result["type"] == "remote_mcp"
        assert result["available_tools"] == []
        assert result["missing_secrets"] == []


def test_get_tool_server_config_not_found(client, test_project):
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        response = client.get(
            f"/api/projects/{test_project.id}/tool_servers/nonexistent-id/config"
        )

        assert response.status_code == 404
        assert response.json()["message"] == "Tool server not found"


def _builtin_ai_models_set(set_result):
    """Return the always-present AI Models tool set (or None)."""
    return next(
        (
            s
            for s in set_result
            if s["type"] == "sandbox_code" and s["set_name"] == "AI Models"
        ),
        None,
    )


def test_get_available_tools_empty(client, test_project):
    """With no tool servers, only the always-available built-in AI Models set
    (llm + llm_judge) is returned. The Call Kiln API built-in stays hidden."""
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        response = client.get(f"/api/projects/{test_project.id}/available_tools")

        assert response.status_code == 200
        result = response.json()
        assert len(result) == 1
        ai_models = _builtin_ai_models_set(result)
        assert ai_models is not None
        tool_ids = {t["id"] for t in ai_models["tools"]}
        assert tool_ids == {"kiln_tool::llm", "kiln_tool::llm_judge"}
        # Call Kiln API is intentionally not listed.
        assert "kiln_tool::call_kiln_api" not in tool_ids


async def test_get_available_tools_success_single_server(client, test_project):
    """Test get_available_tools successfully retrieves tools from a single MCP server"""
    # First create a tool server
    tool_data = {
        "name": "test_available_tools",
        "server_url": "https://example.com/mcp",
        "headers": {"Authorization": "Bearer token"},
        "description": "Test MCP server",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Create the tool server
        async with mock_mcp_success():
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
        created_tool = create_response.json()
        server_id = created_tool["id"]

        # Mock retrieval with specific tools
        mock_tools = [
            Tool(name="echo", description="Echo tool", inputSchema={}),
            Tool(name="calculator", description="Math calculator", inputSchema={}),
            Tool(
                name="weather", description=None, inputSchema={}
            ),  # Test None description
        ]

        async with mock_mcp_success(tools=mock_tools):
            # Get available tools
            response = client.get(f"/api/projects/{test_project.id}/available_tools")

            assert response.status_code == 200
            set_result = response.json()
            # The built-in AI Models set plus the MCP set.
            assert len(set_result) == 2
            assert _builtin_ai_models_set(set_result) is not None
            mcp_set = next(
                s
                for s in set_result
                if s["set_name"] == "MCP Server: test_available_tools"
            )
            assert mcp_set["type"] == "mcp"
            result = mcp_set["tools"]
            assert len(result) == 3

            # Verify tool details
            tool_names = [tool["name"] for tool in result]
            assert "echo" in tool_names
            assert "calculator" in tool_names
            assert "weather" in tool_names

            # Verify tool IDs are properly formatted
            for tool in result:
                assert tool["id"].startswith(f"mcp::remote::{server_id}::")
                assert tool["name"] in ["echo", "calculator", "weather"]

            # Find specific tools and check their descriptions
            echo_tool = next(t for t in result if t["name"] == "echo")
            assert echo_tool["description"] == "Echo tool"

            weather_tool = next(t for t in result if t["name"] == "weather")
            assert weather_tool["description"] is None


async def test_get_available_tools_multiple_servers(client, test_project):
    """Test get_available_tools with multiple tool servers including kiln task tools"""

    # Create first MCP tool server
    tool_data_1 = {
        "name": "mcp_server_1",
        "server_url": "https://example1.com/mcp",
        "headers": {},
        "description": "First MCP server",
        "is_archived": False,
    }

    # Create second MCP tool server
    tool_data_2 = {
        "name": "mcp_server_2",
        "server_url": "https://example2.com/mcp",
        "headers": {},
        "description": "Second MCP server",
        "is_archived": False,
    }

    # Create kiln task tool servers
    kiln_task_server_1 = ExternalToolServer(
        name="kiln_task_server_1",
        type=ToolServerType.kiln_task,
        description="First kiln task server",
        properties={
            "name": "test_task_tool_1",
            "description": "First test task tool",
            "task_id": "task_1",
            "run_config_id": "run_config_1",
            "is_archived": False,
        },
        parent=test_project,
    )

    kiln_task_server_2 = ExternalToolServer(
        name="kiln_task_server_2",
        type=ToolServerType.kiln_task,
        description="Second kiln task server",
        properties={
            "name": "test_task_tool_2",
            "description": "Second test task tool",
            "task_id": "task_2",
            "run_config_id": "run_config_2",
            "is_archived": False,
        },
        parent=test_project,
    )

    # Save the kiln task tool servers
    kiln_task_server_1.save_to_file()
    kiln_task_server_2.save_to_file()

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Create both tool servers
        async with mock_mcp_success():
            response1 = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data_1,
            )
            assert response1.status_code == 200
            server1_id = response1.json()["id"]

            response2 = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data_2,
            )
            assert response2.status_code == 200
            server2_id = response2.json()["id"]

        # Mock tools for both servers
        mock_tools_1 = [
            Tool(name="tool_a", description="Tool A from server 1", inputSchema={}),
            Tool(name="tool_b", description="Tool B from server 1", inputSchema={}),
        ]
        mock_tools_2 = [
            Tool(name="tool_x", description="Tool X from server 2", inputSchema={}),
        ]

        # Create a mapping of server URLs to their tools
        tools_by_server = {
            "https://example1.com/mcp": ListToolsResult(tools=mock_tools_1),
            "https://example2.com/mcp": ListToolsResult(tools=mock_tools_2),
        }

        @asynccontextmanager
        async def mock_mcp_client(tool_server):
            mock_session = AsyncMock()
            # Use the server URL to determine which tools to return
            mock_session.list_tools.return_value = tools_by_server[
                tool_server.properties["server_url"]
            ]
            yield mock_session

        with patch(
            "app.desktop.studio_server.tool_api.MCPSessionManager.shared"
        ) as mock_session_manager_shared:
            mock_session_manager = AsyncMock()
            mock_session_manager.mcp_client = mock_mcp_client
            mock_session_manager_shared.return_value = mock_session_manager

            # Get available tools
            response = client.get(f"/api/projects/{test_project.id}/available_tools")

            assert response.status_code == 200
            set_result = response.json()
            # built-in AI Models + 2 MCP servers + 1 kiln task set
            assert len(set_result) == 4
            # The AI Models built-in set is always present; Call Kiln API is not.
            assert _builtin_ai_models_set(set_result) is not None

            # Find sets by name instead of assuming order
            server1_set = next(
                (s for s in set_result if s["set_name"] == "MCP Server: mcp_server_1"),
                None,
            )
            server2_set = next(
                (s for s in set_result if s["set_name"] == "MCP Server: mcp_server_2"),
                None,
            )
            kiln_task_set = next(
                (s for s in set_result if s["set_name"] == "Kiln Tasks as Tools"),
                None,
            )

            assert server1_set is not None, (
                "Could not find MCP Server: mcp_server_1 in results"
            )
            assert server2_set is not None, (
                "Could not find MCP Server: mcp_server_2 in results"
            )
            assert kiln_task_set is not None, (
                "Could not find Kiln Tasks as Tools in results"
            )

            assert len(server1_set["tools"]) == 2  # 2 from server1
            assert len(server2_set["tools"]) == 1  # 1 from server2
            assert len(kiln_task_set["tools"]) == 2  # 2 kiln task tools

            for tool in server1_set["tools"]:
                assert tool["id"].startswith(f"mcp::remote::{server1_id}::")
            for tool in server2_set["tools"]:
                assert tool["id"].startswith(f"mcp::remote::{server2_id}::")
            for tool in kiln_task_set["tools"]:
                assert tool["id"].startswith(KILN_TASK_TOOL_ID_PREFIX)

            # Verify MCP tools from both servers are present
            tool_names = [tool["name"] for tool in server1_set["tools"]]
            assert "tool_a" in tool_names
            assert "tool_b" in tool_names
            tool_names = [tool["name"] for tool in server2_set["tools"]]
            assert "tool_x" in tool_names

            # Verify kiln task tool names
            kiln_tool_names = [tool["name"] for tool in kiln_task_set["tools"]]
            assert "test_task_tool_1" in kiln_tool_names
            assert "test_task_tool_2" in kiln_tool_names

            # Verify kiln task tool IDs
            tool1 = next(
                t for t in kiln_task_set["tools"] if t["name"] == "test_task_tool_1"
            )
            assert tool1["id"] == f"{KILN_TASK_TOOL_ID_PREFIX}{kiln_task_server_1.id}"
            tool2 = next(
                t for t in kiln_task_set["tools"] if t["name"] == "test_task_tool_2"
            )
            assert tool2["id"] == f"{KILN_TASK_TOOL_ID_PREFIX}{kiln_task_server_2.id}"


async def test_get_available_tools_mcp_error_handling(client, test_project):
    """Test get_available_tools handles MCP connection errors gracefully"""
    # Create a tool server
    tool_data = {
        "name": "failing_mcp_server",
        "server_url": "https://failing.example.com/mcp",
        "headers": {},
        "description": "MCP server that will fail",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Create the tool server
        async with mock_mcp_success():
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200

        # Mock retrieval with error
        async with mock_mcp_list_tools_error("MCP connection failed"):
            # The API should handle the exception gracefully and return an empty list
            # The failing server should be skipped and not appear in the results
            response = client.get(f"/api/projects/{test_project.id}/available_tools")

            assert response.status_code == 200
            result = response.json()

            # Failing MCP server is skipped; only the always-present built-in
            # AI Models set remains (Call Kiln API is not listed).
            assert len(result) == 1
            assert _builtin_ai_models_set(result) is not None


def test_get_available_tools_demo_tools_enabled(client, test_project):
    """Test get_available_tools includes demo tools when enabled"""
    with (
        patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id,
        patch("app.desktop.studio_server.tool_api.Config.shared") as mock_config,
    ):
        mock_project_from_id.return_value = test_project

        # Mock config to enable demo tools
        mock_config_instance = Mock()
        mock_config_instance.enable_demo_tools = True
        mock_config_instance.user_id = "test_user"
        mock_config.return_value = mock_config_instance

        response = client.get(f"/api/projects/{test_project.id}/available_tools")

        assert response.status_code == 200
        result = response.json()

        # The always-present built-in AI Models set plus the demo set.
        assert len(result) == 2
        assert _builtin_ai_models_set(result) is not None

        demo_set = next(s for s in result if s["set_name"] == "Kiln Demo Tools")
        assert demo_set["type"] == "demo"
        assert len(demo_set["tools"]) == 4

        # Verify all demo tools are present with correct IDs and names
        tool_names = [tool["name"] for tool in demo_set["tools"]]
        tool_ids = [tool["id"] for tool in demo_set["tools"]]

        assert "Addition" in tool_names
        assert "Subtraction" in tool_names
        assert "Multiplication" in tool_names
        assert "Division" in tool_names

        assert "kiln_tool::add_numbers" in tool_ids
        assert "kiln_tool::subtract_numbers" in tool_ids
        assert "kiln_tool::multiply_numbers" in tool_ids
        assert "kiln_tool::divide_numbers" in tool_ids

        # Verify descriptions
        for tool in demo_set["tools"]:
            assert tool["description"] is not None
            if tool["name"] == "Addition":
                assert tool["description"] == "Add two numbers together"
            elif tool["name"] == "Subtraction":
                assert tool["description"] == "Subtract two numbers"
            elif tool["name"] == "Multiplication":
                assert tool["description"] == "Multiply two numbers"
            elif tool["name"] == "Division":
                assert tool["description"] == "Divide two numbers"


def test_get_available_tools_demo_tools_disabled(client, test_project):
    """Test get_available_tools excludes demo tools when disabled"""
    with (
        patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id,
        patch("app.desktop.studio_server.tool_api.Config.shared") as mock_config,
    ):
        mock_project_from_id.return_value = test_project

        # Mock config to disable demo tools (default behavior)
        mock_config_instance = Mock()
        mock_config_instance.enable_demo_tools = False
        mock_config_instance.user_id = "test_user"
        mock_config.return_value = mock_config_instance

        response = client.get(f"/api/projects/{test_project.id}/available_tools")

        assert response.status_code == 200
        result = response.json()

        # Demo tools disabled: only the always-present built-in AI Models set
        # remains (Call Kiln API is still not listed).
        assert len(result) == 1
        assert _builtin_ai_models_set(result) is not None


def test_get_available_tools_builtin_ai_models_always_present(client, test_project):
    """The AI Models built-in set (llm + llm_judge) is always available, and is
    not demo-gated (present regardless of enable_demo_tools)."""
    with (
        patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id,
        patch("app.desktop.studio_server.tool_api.Config.shared") as mock_config,
    ):
        mock_project_from_id.return_value = test_project
        mock_config_instance = Mock()
        mock_config_instance.enable_demo_tools = False
        mock_config_instance.user_id = "test_user"
        mock_config.return_value = mock_config_instance

        response = client.get(f"/api/projects/{test_project.id}/available_tools")

        assert response.status_code == 200
        ai_models = _builtin_ai_models_set(response.json())
        assert ai_models is not None
        by_id = {t["id"]: t for t in ai_models["tools"]}
        assert set(by_id) == {"kiln_tool::llm", "kiln_tool::llm_judge"}
        assert by_id["kiln_tool::llm"]["name"] == "LLM"
        assert by_id["kiln_tool::llm"]["function_name"] == "llm"
        assert by_id["kiln_tool::llm_judge"]["name"] == "LLM Judge"
        assert by_id["kiln_tool::llm_judge"]["function_name"] == "llm_judge"


def test_ai_models_set_is_typed_sandbox_code(client, test_project):
    """The AI Models set must carry SANDBOX_CODE, not a generic type.

    That type is the whole contract: it is what tells every tool picker (and any
    other API consumer) that these are not agent tools. Typed BUILTIN instead, they
    would silently become selectable in every agent picker.
    """
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        response = client.get(f"/api/projects/{test_project.id}/available_tools")

        assert response.status_code == 200
        sets_by_name = {s["set_name"]: s for s in response.json()}
        assert sets_by_name["AI Models"]["type"] == ToolSetType.SANDBOX_CODE.value
        # Every tool the sandbox-only set advertises really is one of the built-in
        # sandbox tools -- nothing agent-selectable may hide in here.
        assert {t["id"] for t in sets_by_name["AI Models"]["tools"]} == {
            KilnBuiltInToolId.LLM.value,
            KilnBuiltInToolId.LLM_JUDGE.value,
        }


def test_web_ui_built_in_tool_ids_match_the_enum():
    """Guard the built-in tool ids the web UI has to name literally.

    They are not in the generated OpenAPI client (the API types them as plain tool
    id strings), so built_in_tool_ids.ts hand-copies them. Two rules downstream key
    off those literals: `CODE_EVAL_ONLY_TOOL_IDS` keeps `llm_judge` out of the
    code-tool picker, and the code-judge examples grant themselves the tools their
    snippets call. Renaming an enum value without updating the .ts would silently
    break both, so fail here instead.
    """
    tool_ids_module = (
        Path(__file__).resolve().parents[2]
        / "web_ui"
        / "src"
        / "lib"
        / "utils"
        / "built_in_tool_ids.ts"
    )
    assert tool_ids_module.is_file(), f"expected the web UI module at {tool_ids_module}"

    # Exact ids, not a substring search: "kiln_tool::llm" is a prefix of
    # "kiln_tool::llm_judge" and of any renamed-but-not-updated value, so a loose
    # check would pass right through the drift this test exists to catch.
    declared = dict(
        re.findall(
            r'export const (\w+) = "([^"]+)"',
            tool_ids_module.read_text(),
        )
    )
    assert declared == {
        "LLM_TOOL_ID": KilnBuiltInToolId.LLM.value,
        "LLM_JUDGE_TOOL_ID": KilnBuiltInToolId.LLM_JUDGE.value,
    }, (
        f"built_in_tool_ids.ts declares {declared} -- update it to match "
        "KilnBuiltInToolId."
    )


def test_code_eval_only_tool_ids_uses_the_shared_constant():
    """`CODE_EVAL_ONLY_TOOL_IDS` must reference the guarded constant.

    test_web_ui_built_in_tool_ids_match_the_enum only guards built_in_tool_ids.ts.
    Re-typing the literal here instead of importing it would put the narrowing rule
    back outside that guard's reach.
    """
    tools_store = (
        Path(__file__).resolve().parents[2]
        / "web_ui"
        / "src"
        / "lib"
        / "stores"
        / "tools_store.ts"
    )
    assert tools_store.is_file(), f"expected the web UI store at {tools_store}"

    declaration = re.search(
        r"const CODE_EVAL_ONLY_TOOL_IDS = \[(.*?)\]",
        tools_store.read_text(),
        re.DOTALL,
    )
    assert declaration is not None, (
        "could not find CODE_EVAL_ONLY_TOOL_IDS in tools_store.ts"
    )
    assert declaration.group(1).strip() == "LLM_JUDGE_TOOL_ID", (
        f"CODE_EVAL_ONLY_TOOL_IDS is [{declaration.group(1).strip()}], expected "
        "[LLM_JUDGE_TOOL_ID] imported from built_in_tool_ids.ts."
    )


def test_web_ui_task_response_tool_name_matches_libs_core():
    """The web UI's copy of the structured-answer tool name must match libs/core.

    The chat trace and the claim evidence flattener both recognise the synthetic
    `task_response` call by name, to show its arguments as the model's answer
    rather than as a tool call. The name is not in the generated OpenAPI client,
    so task_response_tool.ts hand-copies it. Renaming the tool on one side
    without the other would silently show answers as tool calls, so fail here.
    """
    module = (
        Path(__file__).resolve().parents[2]
        / "web_ui"
        / "src"
        / "lib"
        / "utils"
        / "task_response_tool.ts"
    )
    assert module.is_file(), f"expected the web UI module at {module}"

    declared = dict(re.findall(r'export const (\w+) = "([^"]+)"', module.read_text()))
    assert declared == {"TASK_RESPONSE_TOOL_NAME": TASK_RESPONSE_TOOL_NAME}, (
        f"task_response_tool.ts declares {declared} -- update it to match "
        "TASK_RESPONSE_TOOL_NAME in kiln_ai.utils.open_ai_types."
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "lib/ui/trace/chat_trace.svelte",
        "routes/(app)/specs/[project_id]/[task_id]/builder/claim_evidence.ts",
    ],
)
def test_web_ui_task_response_consumers_use_the_shared_constant(relative_path):
    """Each web consumer must import the guarded constant, not retype the name.

    test_web_ui_task_response_tool_name_matches_libs_core only guards
    task_response_tool.ts; a literal typed again in a consumer would sit outside
    that guard.
    """
    source_file = Path(__file__).resolve().parents[2] / "web_ui" / "src" / relative_path
    assert source_file.is_file(), f"expected the web UI source at {source_file}"
    source = source_file.read_text()
    assert (
        'import { TASK_RESPONSE_TOOL_NAME } from "$lib/utils/task_response_tool"'
        in source
    ), f"{relative_path} does not import TASK_RESPONSE_TOOL_NAME"
    assert '"task_response"' not in source, (
        f"{relative_path} retypes the task_response literal instead of using "
        "TASK_RESPONSE_TOOL_NAME"
    )


async def test_create_tool_server_whitespace_handling(
    client, test_project, mock_mcp_validation
):
    """Test that whitespace is properly handled from inputs"""
    tool_data = {
        "name": "test_tool",  # Frontend trims whitespace before sending
        "server_url": "https://example.com/api",
        "headers": {"Authorization": "Bearer token"},
        "description": "A test tool",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        async with mock_mcp_validation():
            response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "test_tool"
        assert result["properties"]["server_url"] == "https://example.com/api"
        assert result["properties"]["headers"]["Authorization"] == "Bearer token"
        assert result["description"] == "A test tool"


async def test_create_tool_server_workflow(client, test_project):
    """Test the complete workflow of creating, listing, and retrieving a tool server"""
    # Step 1: Create a tool server
    tool_data = {
        "name": "workflow_test_tool",
        "server_url": "https://workflow.example.com/api",
        "headers": {"Authorization": "Bearer workflow-token"},
        "description": "Tool for testing complete workflow",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Step 1: Create the tool server
        async with mock_mcp_success():
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200

        created_tool = create_response.json()
        tool_server_id = created_tool["id"]

        # Step 2: Verify it appears in the list
        list_response = client.get(
            f"/api/projects/{test_project.id}/available_tool_servers"
        )
        assert list_response.status_code == 200
        tool_servers = list_response.json()
        assert len(tool_servers) == 1
        assert tool_servers[0]["id"] == tool_server_id
        assert tool_servers[0]["name"] == "workflow_test_tool"

        # Step 3 & 4: Get detailed view with mock tools
        mock_tools = [
            Tool(name=f"tool_{i}", description=f"Tool number {i}", inputSchema={})
            for i in range(20)
        ]

        async with mock_mcp_success(tools=mock_tools):
            detail_response = client.get(
                f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
            )
            assert detail_response.status_code == 200
            detailed_tool = detail_response.json()
            assert detailed_tool["id"] == tool_server_id
            assert detailed_tool["name"] == "workflow_test_tool"
            assert len(detailed_tool["available_tools"]) == 20

            tool_names = [tool["name"] for tool in detailed_tool["available_tools"]]
            expected_names = [f"tool_{i}" for i in range(20)]
            assert tool_names == expected_names


@pytest.mark.parametrize(
    "endpoint, tool_data_generator",
    [
        (
            "connect_remote_mcp",
            lambda i: {
                "name": f"concurrent_tool_{i}",
                "server_url": f"https://example{i}.com/api",
                "headers": {"X-Server-ID": str(i)},
                "description": f"Concurrent tool server {i}",
                "is_archived": False,
            },
        ),
        (
            "connect_local_mcp",
            lambda i: {
                "name": f"concurrent_tool_{i}",
                "command": "python",
                "args": ["-m", f"server_{i}"],
                "description": f"Concurrent tool {i}",
                "env_vars": {f"VAR_{i}": f"value_{i}"},
                "is_archived": False,
            },
        ),
    ],
)
async def test_create_tool_server_concurrent_creation(
    client, test_project, endpoint, tool_data_generator
):
    """Test creating multiple tool servers concurrently"""
    count = 3
    tool_servers = [tool_data_generator(i) for i in range(count)]

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        async with mock_mcp_success():
            created_tools = []
            for tool_data in tool_servers:
                response = client.post(
                    f"/api/projects/{test_project.id}/{endpoint}",
                    json=tool_data,
                )
                assert response.status_code == 200
                created_tools.append(response.json())

            # Verify all tools were created with unique IDs
            tool_ids = [tool["id"] for tool in created_tools]
            assert len(set(tool_ids)) == count

        # Verify they all appear in the list
        list_response = client.get(
            f"/api/projects/{test_project.id}/available_tool_servers"
        )
        assert list_response.status_code == 200
        tool_servers_list = list_response.json()
        assert len(tool_servers_list) == count

        # Verify correct names
        server_names = {server["name"] for server in tool_servers_list}
        expected_names = {f"concurrent_tool_{i}" for i in range(count)}
        assert server_names == expected_names


async def test_create_tool_server_duplicate_names_allowed(client, test_project):
    """Test that creating tool servers with duplicate names is allowed"""
    tool_data = {
        "name": "duplicate_name_tool",
        "server_url": "https://example1.com/api",
        "headers": {},
        "description": "First tool with this name",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        async with mock_mcp_success():
            # Create first tool
            response1 = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert response1.status_code == 200
            tool1 = response1.json()

            # Create second tool with same name but different URL
            tool_data["server_url"] = "https://example2.com/api"
            tool_data["description"] = "Second tool with this name"

            response2 = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert response2.status_code == 200
            tool2 = response2.json()

            # Verify they have different IDs
            assert tool1["id"] != tool2["id"]
        assert tool1["name"] == tool2["name"] == "duplicate_name_tool"

        # Verify both appear in list
        list_response = client.get(
            f"/api/projects/{test_project.id}/available_tool_servers"
        )
        assert list_response.status_code == 200
        tool_servers = list_response.json()
        assert len(tool_servers) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_type, properties, expected_prefix, mock_tools",
    [
        (
            ToolServerType.remote_mcp,
            {
                "server_url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer token"},
                "is_archived": False,
            },
            "mcp::remote::",
            [
                Tool(name="echo", description="Echo tool", inputSchema={}),
                Tool(name="calculator", description="Math calculator", inputSchema={}),
                Tool(name="weather", description=None, inputSchema={}),
            ],
        ),
        (
            ToolServerType.local_mcp,
            {
                "command": "python",
                "args": ["-m", "test_mcp_server"],
                "env_vars": {},
                "is_archived": False,
            },
            "mcp::local::",
            [
                Tool(name="local_echo", description="Local echo tool", inputSchema={}),
                Tool(name="local_calc", description=None, inputSchema={}),
            ],
        ),
    ],
)
async def test_available_mcp_tools_success(
    server_type, properties, expected_prefix, mock_tools
):
    """Test available_mcp_tools successfully retrieves tools from MCP servers"""

    # Create a mock ExternalToolServer
    server = ExternalToolServer(
        name="test_server",
        type=server_type,
        description="Test MCP server",
        properties=properties,
    )

    async with mock_mcp_success(tools=mock_tools):
        # Call the function
        result = await available_mcp_tools(server)

        # Verify the result
        assert len(result) == len(mock_tools)

        # Check tool details
        result_names = [tool.name for tool in result]
        expected_names = [tool.name for tool in mock_tools]
        assert result_names == expected_names

        # Check tool IDs are properly formatted
        for tool in result:
            assert tool.id.startswith(f"{expected_prefix}{server.id}::")

        # Check descriptions match
        for result_tool, expected_tool in zip(result, mock_tools):
            assert result_tool.description == expected_tool.description


@pytest.mark.asyncio
async def test_available_mcp_tools_connection_error():
    """Test available_mcp_tools throws exception on connection errors"""

    # Create a mock ExternalToolServer
    server = ExternalToolServer(
        name="failing_server",
        type=ToolServerType.remote_mcp,
        description="Failing MCP server",
        properties={
            "server_url": "https://failing.example.com/mcp",
            "headers": {},
            "is_archived": False,
        },
    )

    async with mock_mcp_connection_error():
        # Call the function - should throw exception on error
        with pytest.raises(Exception, match="Connection failed"):
            await available_mcp_tools(server)


@pytest.mark.asyncio
async def test_available_mcp_tools_list_tools_error():
    """Test available_mcp_tools throws exception on list_tools errors"""

    # Create a mock ExternalToolServer
    server = ExternalToolServer(
        name="error_server",
        type=ToolServerType.remote_mcp,
        description="MCP server with list_tools error",
        properties={
            "server_url": "https://error.example.com/mcp",
            "headers": {},
            "is_archived": False,
        },
    )

    async with mock_mcp_list_tools_error():
        # Call the function - should throw exception on error
        with pytest.raises(Exception, match="list_tools failed"):
            await available_mcp_tools(server)


@pytest.mark.asyncio
async def test_available_mcp_tools_empty_tools():
    """Test available_mcp_tools handles empty tools list"""

    # Create a mock ExternalToolServer
    server = ExternalToolServer(
        name="empty_server",
        type=ToolServerType.remote_mcp,
        description="MCP server with no tools",
        properties={
            "server_url": "https://empty.example.com/mcp",
            "headers": {},
            "is_archived": False,
        },
    )

    async with mock_mcp_success():  # Empty tools by default
        # Call the function
        result = await available_mcp_tools(server)

        # Verify empty list is returned
        assert result == []


@pytest.mark.asyncio
async def test_available_mcp_tools_kiln_task_raises_value_error():
    """Test available_mcp_tools raises ValueError when called with kiln_task server type"""

    # Create a mock ExternalToolServer with kiln_task type
    server = ExternalToolServer(
        name="test_kiln_task",
        type=ToolServerType.kiln_task,
        description="Test Kiln task server",
        properties={
            "name": "test_task",
            "description": "Test task description",
            "is_archived": False,
            "task_id": "test_task_id",
            "run_config_id": "test_run_config_id",
        },
    )

    # Test that ValueError is raised
    with pytest.raises(
        ValueError, match="Kiln task tools are not available from an MCP server"
    ):
        await available_mcp_tools(server)


# Unit tests for validate_tool_server_connectivity function
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_type, properties",
    [
        (
            ToolServerType.remote_mcp,
            {
                "server_url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer token"},
                "is_archived": False,
            },
        ),
        (
            ToolServerType.local_mcp,
            {
                "command": "python",
                "args": ["-m", "test_mcp_server"],
                "env_vars": {"DEBUG": "true"},
                "is_archived": False,
            },
        ),
    ],
)
async def test_validate_tool_server_connectivity_success(server_type, properties):
    """Test validate_tool_server_connectivity succeeds when MCP server is reachable"""
    tool_server = ExternalToolServer(
        name="test_server",
        type=server_type,
        description="Test MCP server",
        properties=properties,
    )

    async with mock_mcp_success():
        # Should not raise any exception
        await validate_tool_server_connectivity(tool_server)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_type, properties, error_context, error_message",
    [
        (
            ToolServerType.remote_mcp,
            {
                "server_url": "https://failing.example.com/mcp",
                "headers": {},
                "is_archived": False,
            },
            "connection_error",
            "Connection failed",
        ),
        (
            ToolServerType.remote_mcp,
            {
                "server_url": "https://example.com/mcp",
                "headers": {},
                "is_archived": False,
            },
            "list_tools_error",
            "list_tools failed",
        ),
        (
            ToolServerType.local_mcp,
            {
                "command": "python",
                "args": ["-m", "nonexistent_server"],
                "env_vars": {},
                "is_archived": False,
            },
            "list_tools_error",
            "Local MCP server failed",
        ),
    ],
)
async def test_validate_tool_server_connectivity_failed(
    server_type, properties, error_context, error_message
):
    """Test validate_tool_server_connectivity raises error when MCP server fails"""
    tool_server = ExternalToolServer(
        name="failing_server",
        type=server_type,
        description="Failing MCP server",
        properties=properties,
    )

    if error_context == "connection_error":
        context_manager = mock_mcp_connection_error()
    else:
        context_manager = mock_mcp_list_tools_error(error_message)

    async with context_manager:
        # Should raise the raw exception
        with pytest.raises(Exception, match=error_message):
            await validate_tool_server_connectivity(tool_server)


@pytest.mark.asyncio
async def test_validate_tool_server_connectivity_kiln_task():
    """Test validate_tool_server_connectivity does nothing for kiln_task type"""
    tool_server = ExternalToolServer(
        name="kiln_task_server",
        type=ToolServerType.kiln_task,
        description="Kiln task tool server",
        properties={
            "name": "kiln_task_server",
            "description": "Kiln task tool server",
            "task_id": "test_task_id",
            "run_config_id": "test_run_config_id",
            "is_archived": False,
        },
    )

    # Should complete without any exceptions or network calls
    await validate_tool_server_connectivity(tool_server)


# Tests for connect_local_mcp endpoint
async def test_create_local_tool_server_success(client, test_project):
    """Test successful local tool server creation"""
    tool_data = {
        "name": "test_local_mcp_tool",
        "command": "python",
        "args": ["-m", "test_mcp_server"],
        "env_vars": {"DEBUG": "true"},
        "description": "A test local MCP tool",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        async with mock_mcp_success():
            response = client.post(
                f"/api/projects/{test_project.id}/connect_local_mcp",
                json=tool_data,
            )

            assert response.status_code == 200
            result = response.json()
            assert result["name"] == "test_local_mcp_tool"
            assert result["type"] == "local_mcp"
            assert result["description"] == "A test local MCP tool"
            assert result["properties"]["command"] == "python"
            assert result["properties"]["args"] == ["-m", "test_mcp_server"]
            assert result["properties"]["env_vars"]["DEBUG"] == "true"
            assert "id" in result
            assert "created_at" in result


async def test_create_local_tool_server_validation_failed(client, test_project):
    """Test local tool server creation fails when MCP server validation fails"""
    tool_data = {
        "name": "failing_local_tool",
        "command": "python",
        "args": ["-m", "nonexistent_server"],
        "env_vars": {},
        "description": "Local tool that will fail validation",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        async with mock_mcp_connection_error():
            # Unhandled exception is now raised instead of returning 422
            with pytest.raises(Exception, match="Connection failed"):
                client.post(
                    f"/api/projects/{test_project.id}/connect_local_mcp",
                    json=tool_data,
                )


async def test_create_local_tool_server_duplicate_names_allowed(client, test_project):
    """Test local tool server creation allows duplicate names (like remote MCP)"""
    tool_data = {
        "name": "duplicate_name_tool",
        "command": "python",
        "args": ["-m", "server1"],
        "description": "First tool with this name",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        async with mock_mcp_success():
            # Create first tool
            response1 = client.post(
                f"/api/projects/{test_project.id}/connect_local_mcp",
                json=tool_data,
            )
            assert response1.status_code == 200
            tool1_id = response1.json()["id"]

            # Create second tool with same name but different properties
            tool_data["args"] = ["-m", "server2"]
            tool_data["description"] = "Second tool with this name"

            response2 = client.post(
                f"/api/projects/{test_project.id}/connect_local_mcp",
                json=tool_data,
            )
            assert response2.status_code == 200
            tool2_id = response2.json()["id"]

            # Both should exist with different IDs
            assert tool1_id != tool2_id


async def test_create_local_tool_server_list_tools_failed(client, test_project):
    """Test local tool server creation fails when MCP list_tools fails"""
    tool_data = {
        "name": "list_tools_failing_local",
        "command": "python",
        "args": ["-m", "failing_server"],
        "description": "Local tool where list_tools fails",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        async with mock_mcp_list_tools_error("list_tools failed"):
            # Unhandled exception is now raised instead of returning 422
            with pytest.raises(Exception, match="list_tools failed"):
                client.post(
                    f"/api/projects/{test_project.id}/connect_local_mcp",
                    json=tool_data,
                )


# Tests for tool_server_from_id function


def test_tool_server_from_id_success(test_project):
    """Test tool_server_from_id returns correct tool server when found"""

    # Create a tool server
    tool_server = ExternalToolServer(
        name="test_tool_server",
        type=ToolServerType.remote_mcp,
        description="Test tool server",
        properties={
            "server_url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer token"},
            "is_archived": False,
        },
        parent=test_project,
    )
    tool_server.save_to_file()

    # Test that we can find it with mocked project_from_id
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project
        found_server = tool_server_from_id(test_project.id, str(tool_server.id))
        assert found_server.id == tool_server.id
        assert found_server.name == "test_tool_server"
        assert found_server.type == ToolServerType.remote_mcp


def test_tool_server_from_id_not_found(test_project):
    """Test tool_server_from_id raises HTTPException when tool server not found"""

    # Try to find a non-existent tool server with mocked project_from_id
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project
        with pytest.raises(HTTPException) as exc_info:
            tool_server_from_id(test_project.id, "non-existent-id")

        assert exc_info.value.status_code == 404
        assert "Tool server not found" in str(exc_info.value.detail)


def test_tool_server_from_id_project_not_found():
    """Test tool_server_from_id raises HTTPException when project not found"""

    # Try to find a tool server in a non-existent project
    with pytest.raises(HTTPException) as exc_info:
        tool_server_from_id("non-existent-project", "some-tool-id")

    assert exc_info.value.status_code == 404
    assert "Project not found" in str(exc_info.value.detail)


def test_tool_server_from_id_multiple_servers(test_project):
    """Test tool_server_from_id finds correct server when multiple exist"""

    # Create multiple tool servers
    server1 = ExternalToolServer(
        name="server1",
        type=ToolServerType.remote_mcp,
        description="First server",
        properties={
            "server_url": "https://server1.com",
            "headers": {},
            "is_archived": False,
        },
        parent=test_project,
    )
    server1.save_to_file()

    server2 = ExternalToolServer(
        name="server2",
        type=ToolServerType.local_mcp,
        description="Second server",
        properties={
            "command": "python",
            "args": ["-m", "server"],
            "env_vars": {},
            "is_archived": False,
        },
        parent=test_project,
    )
    server2.save_to_file()

    # Test that we can find the correct one with mocked project_from_id
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project
        found_server1 = tool_server_from_id(test_project.id, str(server1.id))
        found_server2 = tool_server_from_id(test_project.id, str(server2.id))

        assert found_server1.id == server1.id
        assert found_server1.name == "server1"
        assert found_server1.type == ToolServerType.remote_mcp

        assert found_server2.id == server2.id
        assert found_server2.name == "server2"
        assert found_server2.type == ToolServerType.local_mcp


# Tests for DELETE /api/projects/{project_id}/tool_servers/{tool_server_id} endpoint
@pytest.mark.parametrize(
    "endpoint, tool_data, server_type",
    [
        (
            "connect_remote_mcp",
            {
                "name": "test_delete_tool",
                "server_url": "https://example.com/api",
                "headers": {"Authorization": "Bearer token"},
                "description": "Tool to be deleted",
                "is_archived": False,
            },
            ToolServerType.remote_mcp,
        ),
        (
            "connect_local_mcp",
            {
                "name": "test_delete_local",
                "command": "python",
                "args": ["-m", "test_server"],
                "env_vars": {"DEBUG": "true"},
                "description": "Local tool to be deleted",
                "is_archived": False,
            },
            ToolServerType.local_mcp,
        ),
    ],
)
async def test_delete_tool_server_success(
    client, test_project, endpoint, tool_data, server_type
):
    """Test successful deletion of a tool server"""
    async with mock_mcp_success():
        with patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id:
            mock_project_from_id.return_value = test_project

            # Create the tool server
            create_response = client.post(
                f"/api/projects/{test_project.id}/{endpoint}",
                json=tool_data,
            )
            assert create_response.status_code == 200
            created_tool = create_response.json()
            tool_server_id = created_tool["id"]

            # Verify it exists by getting it
            get_response = client.get(
                f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
            )
            assert get_response.status_code == 200
            assert get_response.json()["type"] == server_type.value

            # Now delete it
            delete_response = client.delete(
                f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
            )
            assert delete_response.status_code == 200

            # Verify it's been deleted by trying to get it again
            get_after_delete_response = client.get(
                f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
            )
            assert get_after_delete_response.status_code == 404


def test_delete_tool_server_not_found(client, test_project):
    """Test deletion of non-existent tool server returns 404"""
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Try to delete a non-existent tool server
        response = client.delete(
            f"/api/projects/{test_project.id}/tool_servers/non-existent-id"
        )
        assert response.status_code == 404
        assert "Tool server not found" in response.json()["message"]


def test_delete_tool_server_project_not_found(client):
    """Test deletion with non-existent project returns 404"""
    # Try to delete from a non-existent project
    response = client.delete(
        "/api/projects/non-existent-project/tool_servers/some-tool-id"
    )
    assert response.status_code == 404
    assert "Project not found" in response.json()["message"]


async def test_delete_tool_server_affects_available_servers_list(client, test_project):
    """Test that deleting a tool server removes it from the available servers list"""
    # Create two tool servers
    tool_data_1 = {
        "name": "tool_server_1",
        "server_url": "https://server1.com/api",
        "headers": {},
        "description": "First server",
        "is_archived": False,
    }

    tool_data_2 = {
        "name": "tool_server_2",
        "server_url": "https://server2.com/api",
        "headers": {},
        "description": "Second server",
        "is_archived": False,
    }

    async with mock_mcp_success():
        with patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id:
            mock_project_from_id.return_value = test_project

            # Create both servers
            create_response_1 = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data_1,
            )
            assert create_response_1.status_code == 200
            server_1_id = create_response_1.json()["id"]

            create_response_2 = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data_2,
            )
            assert create_response_2.status_code == 200
            server_2_id = create_response_2.json()["id"]

            # Verify both appear in available servers list
            list_response = client.get(
                f"/api/projects/{test_project.id}/available_tool_servers"
            )
            assert list_response.status_code == 200
            servers = list_response.json()
            assert len(servers) == 2
            server_ids = [server["id"] for server in servers]
            assert server_1_id in server_ids
            assert server_2_id in server_ids

            # Delete one server
            delete_response = client.delete(
                f"/api/projects/{test_project.id}/tool_servers/{server_1_id}"
            )
            assert delete_response.status_code == 200

            # Verify only one remains in the list
            list_after_delete_response = client.get(
                f"/api/projects/{test_project.id}/available_tool_servers"
            )
            assert list_after_delete_response.status_code == 200
            remaining_servers = list_after_delete_response.json()
            assert len(remaining_servers) == 1
            assert remaining_servers[0]["id"] == server_2_id
            assert remaining_servers[0]["name"] == "tool_server_2"


async def test_delete_tool_server_with_secret_headers(client, test_project):
    """Test that deleting a tool server removes secret headers from Config"""
    # Create a tool server with secret headers
    tool_data = {
        "name": "tool_with_secrets",
        "server_url": "https://example.com/api",
        "headers": {
            "Authorization": "Bearer secret-token-123",
            "X-API-Key": "api-key-456",
            "Content-Type": "application/json",
        },
        "secret_header_keys": ["Authorization", "X-API-Key"],
        "description": "Tool with secret headers",
        "is_archived": False,
    }

    async with mock_mcp_success():
        with patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id:
            mock_project_from_id.return_value = test_project

            # Create the tool server first (without mocking Config to avoid interference)
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
            created_tool = create_response.json()
            tool_server_id = created_tool["id"]

            # Verify the tool server was created with secret headers
            assert "secret_header_keys" in created_tool["properties"]
            assert created_tool["properties"]["secret_header_keys"] == [
                "Authorization",
                "X-API-Key",
            ]

            # Now mock Config for the delete operation
            with patch(
                "app.desktop.studio_server.tool_api.Config.shared"
            ) as mock_config_shared:
                mock_config = mock_config_shared.return_value
                mock_config.get_value.return_value = {
                    f"{tool_server_id}::Authorization": "Bearer secret-token-123",
                    f"{tool_server_id}::X-API-Key": "api-key-456",
                    "other_server::some_header": "other_value",  # Should not be deleted
                }
                mock_config.update_settings = Mock()

                # Delete the tool server
                delete_response = client.delete(
                    f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
                )
                assert delete_response.status_code == 200

                # Verify that secret headers were removed from config
                mock_config.update_settings.assert_called_once()
                call_args = mock_config.update_settings.call_args[0][0]

                # Check that the updated mcp_secrets no longer contains the deleted server's secrets
                expected_remaining_secrets = {
                    "other_server::some_header": "other_value"
                }
                assert call_args[MCP_SECRETS_KEY] == expected_remaining_secrets


@pytest.mark.parametrize(
    "endpoint, tool_data, server_type",
    [
        (
            "connect_remote_mcp",
            {
                "name": "tool_without_secrets",
                "server_url": "https://example.com/api",
                "headers": {"Content-Type": "application/json"},
                "secret_header_keys": [],
                "description": "Tool without secret headers",
                "is_archived": False,
            },
            ToolServerType.remote_mcp,
        ),
        (
            "connect_local_mcp",
            {
                "name": "local_tool_no_secrets",
                "command": "python",
                "args": ["-m", "test_server"],
                "env_vars": {"DEBUG": "true"},
                "secret_env_var_keys": [],
                "description": "Local tool without secret headers",
                "is_archived": False,
            },
            ToolServerType.local_mcp,
        ),
    ],
)
async def test_delete_tool_server_no_secrets(
    client, test_project, endpoint, tool_data, server_type
):
    """Test that deleting a tool server without secrets works correctly"""
    async with mock_mcp_success():
        with patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id:
            mock_project_from_id.return_value = test_project

            # Create the tool server first (without mocking Config to avoid interference)
            create_response = client.post(
                f"/api/projects/{test_project.id}/{endpoint}",
                json=tool_data,
            )
            assert create_response.status_code == 200
            created_tool = create_response.json()
            tool_server_id = created_tool["id"]

            # Verify the tool server was created
            assert created_tool["type"] == server_type.value
            if server_type == ToolServerType.remote_mcp:
                assert created_tool["properties"]["secret_header_keys"] == []

            # Now mock Config for the delete operation
            with patch(
                "app.desktop.studio_server.tool_api.Config.shared"
            ) as mock_config_shared:
                mock_config = mock_config_shared.return_value
                mock_config.get_value.return_value = {
                    "other_server::some_header": "other_value"
                }
                mock_config.update_settings = Mock()

                # Delete the tool server
                delete_response = client.delete(
                    f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
                )
                assert delete_response.status_code == 200

                # Verify that config update was still called
                mock_config.update_settings.assert_called_once()
                call_args = mock_config.update_settings.call_args[0][0]

                # Check that existing secrets remain unchanged
                expected_remaining_secrets = {
                    "other_server::some_header": "other_value"
                }
                assert call_args[MCP_SECRETS_KEY] == expected_remaining_secrets


async def test_delete_tool_server_missing_secret_header_keys_property(
    client, test_project
):
    """Test that deleting a tool server handles missing secret_header_keys property gracefully"""
    # Create a tool server first
    tool_data = {
        "name": "tool_without_secret_keys_prop",
        "server_url": "https://example.com/api",
        "headers": {"Content-Type": "application/json"},
        "secret_header_keys": [],
        "description": "Tool for testing missing property",
        "is_archived": False,
    }

    async with mock_mcp_success():
        with patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id:
            mock_project_from_id.return_value = test_project

            # Create the tool server first (without mocking Config to avoid interference)
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
            created_tool = create_response.json()
            tool_server_id = created_tool["id"]

            # Manually remove the secret_header_keys property to simulate old data
            # This requires directly modifying the tool server's properties
            tool_server = tool_server_from_id(test_project.id, tool_server_id)
            del tool_server.properties["secret_header_keys"]
            tool_server.save_to_file()

            # Now mock Config for the delete operation
            with patch(
                "app.desktop.studio_server.tool_api.Config.shared"
            ) as mock_config_shared:
                mock_config = mock_config_shared.return_value
                mock_config.get_value.return_value = {
                    "other_server::some_header": "other_value"
                }
                mock_config.update_settings = Mock()

                # Delete the tool server - this should handle the missing property gracefully
                delete_response = client.delete(
                    f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
                )
                assert delete_response.status_code == 200

                # Verify that config update was still called
                mock_config.update_settings.assert_called_once()
                call_args = mock_config.update_settings.call_args[0][0]

                # Check that existing secrets remain unchanged since no secret keys were found
                expected_remaining_secrets = {
                    "other_server::some_header": "other_value"
                }
                assert call_args[MCP_SECRETS_KEY] == expected_remaining_secrets


async def test_delete_tool_server_secret_key_not_in_config(client, test_project):
    """Test that deleting a tool server handles cases where secret keys are not in config"""
    # Create a tool server with secret headers
    tool_data = {
        "name": "tool_with_missing_secrets",
        "server_url": "https://example.com/api",
        "headers": {
            "Authorization": "Bearer secret-token-123",
            "X-API-Key": "api-key-456",
            "Content-Type": "application/json",
        },
        "secret_header_keys": ["Authorization", "X-API-Key"],
        "description": "Tool with secret headers not in config",
        "is_archived": False,
    }

    async with mock_mcp_success():
        with patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id:
            mock_project_from_id.return_value = test_project

            # Create the tool server first (without mocking Config to avoid interference)
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
            created_tool = create_response.json()
            tool_server_id = created_tool["id"]

            # Now mock Config for delete - but don't include this server's secrets
            with patch(
                "app.desktop.studio_server.tool_api.Config.shared"
            ) as mock_config_shared:
                mock_config = mock_config_shared.return_value
                # Config doesn't have the secret keys for this tool server
                mock_config.get_value.return_value = {
                    "other_server::some_header": "other_value"
                }
                mock_config.update_settings = Mock()

                # Delete the tool server - this should handle missing secrets gracefully
                delete_response = client.delete(
                    f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
                )
                assert delete_response.status_code == 200

                # Verify that config update was still called
                mock_config.update_settings.assert_called_once()
                call_args = mock_config.update_settings.call_args[0][0]

                # Check that existing secrets remain unchanged since the secret keys weren't in config
                expected_remaining_secrets = {
                    "other_server::some_header": "other_value"
                }
                assert call_args[MCP_SECRETS_KEY] == expected_remaining_secrets


# Tests for demo tools API endpoints
def test_get_demo_tools_enabled(client):
    """Test GET /api/demo_tools returns True when demo tools are enabled"""
    with patch("app.desktop.studio_server.tool_api.Config.shared") as mock_config:
        mock_config_instance = mock_config.return_value
        mock_config_instance.enable_demo_tools = True

        response = client.get("/api/demo_tools")

        assert response.status_code == 200
        assert response.json() is True


def test_get_demo_tools_disabled(client):
    """Test GET /api/demo_tools returns False when demo tools are disabled"""
    with patch("app.desktop.studio_server.tool_api.Config.shared") as mock_config:
        mock_config_instance = mock_config.return_value
        mock_config_instance.enable_demo_tools = False

        response = client.get("/api/demo_tools")

        assert response.status_code == 200
        assert response.json() is False


def test_set_demo_tools_enable(client):
    """Test POST /api/demo_tools enables demo tools"""
    with patch("app.desktop.studio_server.tool_api.Config.shared") as mock_config:
        mock_config_instance = mock_config.return_value
        mock_config_instance.enable_demo_tools = False  # Initially disabled

        response = client.post("/api/demo_tools?enable_demo_tools=true")

        assert response.status_code == 200
        assert response.json() is True
        # Verify the config was updated
        assert mock_config_instance.enable_demo_tools is True


def test_set_demo_tools_disable(client):
    """Test POST /api/demo_tools disables demo tools"""
    with patch("app.desktop.studio_server.tool_api.Config.shared") as mock_config:
        mock_config_instance = mock_config.return_value
        mock_config_instance.enable_demo_tools = True  # Initially enabled

        response = client.post("/api/demo_tools?enable_demo_tools=false")

        assert response.status_code == 200
        assert response.json() is False
        # Verify the config was updated
        assert mock_config_instance.enable_demo_tools is False


# Tests for secret headers functionality
async def test_connect_remote_mcp_with_secret_headers(client, test_project):
    """Test connect_remote_mcp endpoint stores secret headers correctly"""
    tool_data = {
        "name": "secret_tool",
        "server_url": "https://example.com/api",
        "headers": {
            "Authorization": "Bearer secret-token-123",
            "X-API-Key": "api-key-456",
            "Content-Type": "application/json",
        },
        "secret_header_keys": ["Authorization", "X-API-Key"],
        "description": "Tool with secret headers",
        "is_archived": False,
    }

    with (
        patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id,
        patch("app.desktop.studio_server.tool_api.Config.shared") as mock_config,
    ):
        mock_project_from_id.return_value = test_project

        # Mock config for storing secrets
        mock_config_instance = mock_config.return_value
        mock_config_instance.get_value.return_value = {}  # Empty mcp_secrets initially
        mock_config_instance.user_id = "test_user"

        async with mock_mcp_success():
            response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )

        assert response.status_code == 200
        result = response.json()

        # Verify the tool server was created correctly
        assert result["name"] == "secret_tool"
        assert result["properties"]["server_url"] == "https://example.com/api"

        # Verify secret headers were removed from properties and only non-secret headers remain
        stored_headers = result["properties"]["headers"]
        assert "Content-Type" in stored_headers
        assert stored_headers["Content-Type"] == "application/json"
        assert "Authorization" not in stored_headers
        assert "X-API-Key" not in stored_headers

        # Verify secret header keys were stored
        assert result["properties"]["secret_header_keys"] == [
            "Authorization",
            "X-API-Key",
        ]

        # Verify config.update_settings was called to store secrets
        mock_config_instance.update_settings.assert_called_once()
        call_args = mock_config_instance.update_settings.call_args[0][0]
        assert MCP_SECRETS_KEY in call_args

        # Verify secret values were stored with correct keys
        mcp_secrets = call_args[MCP_SECRETS_KEY]
        server_id = result["id"]
        assert f"{server_id}::Authorization" in mcp_secrets
        assert f"{server_id}::X-API-Key" in mcp_secrets
        assert mcp_secrets[f"{server_id}::Authorization"] == "Bearer secret-token-123"
        assert mcp_secrets[f"{server_id}::X-API-Key"] == "api-key-456"


async def test_connect_remote_mcp_no_secret_headers(client, test_project):
    """Test connect_remote_mcp endpoint without secret headers"""
    tool_data = {
        "name": "no_secret_tool",
        "server_url": "https://example.com/api",
        "headers": {
            "Content-Type": "application/json",
            "User-Agent": "Kiln-AI/1.0",
        },
        "secret_header_keys": [],  # Empty list
        "description": "Tool without secret headers",
        "is_archived": False,
    }

    with (
        patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id,
        patch("app.desktop.studio_server.tool_api.Config.shared") as mock_config,
    ):
        mock_project_from_id.return_value = test_project
        mock_config_instance = mock_config.return_value
        mock_config_instance.user_id = "test_user"

        async with mock_mcp_success():
            response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )

        assert response.status_code == 200
        result = response.json()

        # Verify all headers remain in properties
        stored_headers = result["properties"]["headers"]
        assert stored_headers["Content-Type"] == "application/json"
        assert stored_headers["User-Agent"] == "Kiln-AI/1.0"
        assert result["properties"]["secret_header_keys"] == []

        # Verify config.update_settings was NOT called since no secrets
        mock_config_instance.update_settings.assert_not_called()


async def test_connect_remote_mcp_existing_mcp_secrets(client, test_project):
    """Test connect_remote_mcp endpoint merges with existing mcp_secrets"""
    tool_data = {
        "name": "merge_secret_tool",
        "server_url": "https://example.com/api",
        "headers": {
            "Authorization": "Bearer new-token",
            "Content-Type": "application/json",
        },
        "secret_header_keys": ["Authorization"],
        "description": "Tool that merges with existing secrets",
        "is_archived": False,
    }

    existing_secrets = {
        "other_server_id::X-API-Key": "existing-api-key",
        "another_server::Token": "existing-token",
    }

    with (
        patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id,
        patch("app.desktop.studio_server.tool_api.Config.shared") as mock_config,
    ):
        mock_project_from_id.return_value = test_project

        # Mock config with existing secrets
        mock_config_instance = mock_config.return_value
        mock_config_instance.get_value.return_value = existing_secrets.copy()
        mock_config_instance.user_id = "test_user"

        async with mock_mcp_success():
            response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )

        assert response.status_code == 200
        result = response.json()

        # Verify config.update_settings was called to merge secrets
        mock_config_instance.update_settings.assert_called_once()
        call_args = mock_config_instance.update_settings.call_args[0][0]

        # Verify existing secrets are preserved and new ones added
        mcp_secrets = call_args[MCP_SECRETS_KEY]
        server_id = result["id"]

        # Existing secrets should still be there
        assert "other_server_id::X-API-Key" in mcp_secrets
        assert "another_server::Token" in mcp_secrets
        assert mcp_secrets["other_server_id::X-API-Key"] == "existing-api-key"
        assert mcp_secrets["another_server::Token"] == "existing-token"

        # New secret should be added
        assert f"{server_id}::Authorization" in mcp_secrets
        assert mcp_secrets[f"{server_id}::Authorization"] == "Bearer new-token"


# Tests for local MCP secret environment variables functionality
async def test_connect_local_mcp_with_secret_env_vars(client, test_project):
    """Test connect_local_mcp endpoint stores secret environment variables correctly"""
    tool_data = {
        "name": "secret_env_tool",
        "command": "python",
        "args": ["-m", "my_server"],
        "env_vars": {
            "PUBLIC_VAR": "public_value",
            "SECRET_API_KEY": "secret_key_123",
            "ANOTHER_SECRET": "another_secret_value",
            "DEBUG": "true",
        },
        "secret_env_var_keys": ["SECRET_API_KEY", "ANOTHER_SECRET"],
        "description": "Tool with secret environment variables",
        "is_archived": False,
    }

    with (
        patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id,
        patch("app.desktop.studio_server.tool_api.Config.shared") as mock_config,
    ):
        mock_project_from_id.return_value = test_project

        # Mock config for storing secrets
        mock_config_instance = mock_config.return_value
        mock_config_instance.get_value.return_value = {}  # Empty mcp_secrets initially
        mock_config_instance.user_id = "test_user"

        async with mock_mcp_success():
            response = client.post(
                f"/api/projects/{test_project.id}/connect_local_mcp",
                json=tool_data,
            )

        assert response.status_code == 200
        result = response.json()

        # Verify the tool server was created correctly
        assert result["name"] == "secret_env_tool"
        assert result["properties"]["command"] == "python"
        assert result["properties"]["args"] == ["-m", "my_server"]

        # Verify secret env vars were removed from properties and only non-secret env vars remain
        stored_env_vars = result["properties"]["env_vars"]
        assert "PUBLIC_VAR" in stored_env_vars
        assert stored_env_vars["PUBLIC_VAR"] == "public_value"
        assert "DEBUG" in stored_env_vars
        assert stored_env_vars["DEBUG"] == "true"
        assert "SECRET_API_KEY" not in stored_env_vars
        assert "ANOTHER_SECRET" not in stored_env_vars

        # Verify secret env var keys were stored
        assert result["properties"]["secret_env_var_keys"] == [
            "SECRET_API_KEY",
            "ANOTHER_SECRET",
        ]

        # Verify config.update_settings was called to store secrets
        mock_config_instance.update_settings.assert_called_once()
        call_args = mock_config_instance.update_settings.call_args[0][0]
        assert MCP_SECRETS_KEY in call_args

        # Verify secret values were stored with correct keys
        mcp_secrets = call_args[MCP_SECRETS_KEY]
        server_id = result["id"]
        assert f"{server_id}::SECRET_API_KEY" in mcp_secrets
        assert f"{server_id}::ANOTHER_SECRET" in mcp_secrets
        assert mcp_secrets[f"{server_id}::SECRET_API_KEY"] == "secret_key_123"
        assert mcp_secrets[f"{server_id}::ANOTHER_SECRET"] == "another_secret_value"


async def test_connect_local_mcp_no_secret_env_vars(client, test_project):
    """Test connect_local_mcp endpoint without secret environment variables"""
    tool_data = {
        "name": "no_secret_env_tool",
        "command": "python",
        "args": ["-m", "my_server"],
        "env_vars": {
            "PUBLIC_VAR": "public_value",
            "DEBUG": "true",
        },
        "secret_env_var_keys": [],  # Empty list
        "description": "Tool without secret environment variables",
        "is_archived": False,
    }

    with (
        patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id,
        patch("app.desktop.studio_server.tool_api.Config.shared") as mock_config,
    ):
        mock_project_from_id.return_value = test_project
        mock_config_instance = mock_config.return_value
        mock_config_instance.user_id = "test_user"

        async with mock_mcp_success():
            response = client.post(
                f"/api/projects/{test_project.id}/connect_local_mcp",
                json=tool_data,
            )

        assert response.status_code == 200
        result = response.json()

        # Verify all env vars remain in properties
        stored_env_vars = result["properties"]["env_vars"]
        assert stored_env_vars["PUBLIC_VAR"] == "public_value"
        assert stored_env_vars["DEBUG"] == "true"
        assert result["properties"]["secret_env_var_keys"] == []

        # Verify config.update_settings was NOT called since no secrets
        mock_config_instance.update_settings.assert_not_called()


async def test_connect_local_mcp_existing_mcp_secrets(client, test_project):
    """Test connect_local_mcp endpoint merges with existing mcp_secrets"""
    tool_data = {
        "name": "merge_secret_env_tool",
        "command": "python",
        "args": ["-m", "my_server"],
        "env_vars": {
            "PUBLIC_VAR": "public_value",
            "NEW_SECRET": "new_secret_value",
        },
        "secret_env_var_keys": ["NEW_SECRET"],
        "description": "Tool that merges with existing secrets",
        "is_archived": False,
    }

    existing_secrets = {
        "other_server_id::OLD_SECRET": "existing_secret",
        "another_server::TOKEN": "existing_token",
    }

    with (
        patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id,
        patch("app.desktop.studio_server.tool_api.Config.shared") as mock_config,
    ):
        mock_project_from_id.return_value = test_project

        # Mock config with existing secrets
        mock_config_instance = mock_config.return_value
        mock_config_instance.get_value.return_value = existing_secrets.copy()
        mock_config_instance.user_id = "test_user"

        async with mock_mcp_success():
            response = client.post(
                f"/api/projects/{test_project.id}/connect_local_mcp",
                json=tool_data,
            )

        assert response.status_code == 200
        result = response.json()

        # Verify config.update_settings was called to merge secrets
        mock_config_instance.update_settings.assert_called_once()
        call_args = mock_config_instance.update_settings.call_args[0][0]

        # Verify existing secrets are preserved and new ones added
        mcp_secrets = call_args[MCP_SECRETS_KEY]
        server_id = result["id"]

        # Existing secrets should still be there
        assert "other_server_id::OLD_SECRET" in mcp_secrets
        assert "another_server::TOKEN" in mcp_secrets
        assert mcp_secrets["other_server_id::OLD_SECRET"] == "existing_secret"
        assert mcp_secrets["another_server::TOKEN"] == "existing_token"

        # New secret should be added
        assert f"{server_id}::NEW_SECRET" in mcp_secrets
        assert mcp_secrets[f"{server_id}::NEW_SECRET"] == "new_secret_value"


async def test_delete_local_mcp_tool_server_with_secret_env_vars(client, test_project):
    """Test that deleting a local MCP tool server removes secret environment variables from Config"""
    # Create a local MCP tool server with secret env vars
    tool_data = {
        "name": "local_tool_with_secrets",
        "command": "python",
        "args": ["-m", "my_server"],
        "env_vars": {
            "PUBLIC_VAR": "public_value",
            "SECRET_API_KEY": "secret_key_123",
            "ANOTHER_SECRET": "another_secret_value",
        },
        "secret_env_var_keys": ["SECRET_API_KEY", "ANOTHER_SECRET"],
        "description": "Local tool with secret env vars",
        "is_archived": False,
    }

    async with mock_mcp_success():
        with patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id:
            mock_project_from_id.return_value = test_project

            # Create the tool server first (without mocking Config to avoid interference)
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_local_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
            created_tool = create_response.json()
            tool_server_id = created_tool["id"]

            # Verify the tool server was created with secret env vars
            assert "secret_env_var_keys" in created_tool["properties"]
            assert created_tool["properties"]["secret_env_var_keys"] == [
                "SECRET_API_KEY",
                "ANOTHER_SECRET",
            ]

            # Now mock Config for the delete operation
            with patch(
                "app.desktop.studio_server.tool_api.Config.shared"
            ) as mock_config_shared:
                mock_config = mock_config_shared.return_value
                mock_config.get_value.return_value = {
                    f"{tool_server_id}::SECRET_API_KEY": "secret_key_123",
                    f"{tool_server_id}::ANOTHER_SECRET": "another_secret_value",
                    "other_server::some_env_var": "other_value",  # Should not be deleted
                }
                mock_config.update_settings = Mock()

                # Delete the tool server
                delete_response = client.delete(
                    f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
                )
                assert delete_response.status_code == 200

                # Verify that secret env vars were removed from config
                mock_config.update_settings.assert_called_once()
                call_args = mock_config.update_settings.call_args[0][0]

                # Check that the updated mcp_secrets no longer contains the deleted server's secrets
                expected_remaining_secrets = {
                    "other_server::some_env_var": "other_value"
                }
                assert call_args[MCP_SECRETS_KEY] == expected_remaining_secrets


async def test_delete_local_mcp_tool_server_no_secret_env_vars(client, test_project):
    """Test that deleting a local MCP tool server without secret env vars works correctly"""
    # Create a local MCP tool server without secret env vars
    tool_data = {
        "name": "local_tool_without_secrets",
        "command": "python",
        "args": ["-m", "my_server"],
        "env_vars": {"PUBLIC_VAR": "public_value"},
        "secret_env_var_keys": [],
        "description": "Local tool without secret env vars",
        "is_archived": False,
    }

    async with mock_mcp_success():
        with patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id:
            mock_project_from_id.return_value = test_project

            # Create the tool server first (without mocking Config to avoid interference)
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_local_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
            created_tool = create_response.json()
            tool_server_id = created_tool["id"]

            # Verify the tool server was created without secret env vars
            assert created_tool["properties"]["secret_env_var_keys"] == []

            # Now mock Config for the delete operation
            with patch(
                "app.desktop.studio_server.tool_api.Config.shared"
            ) as mock_config_shared:
                mock_config = mock_config_shared.return_value
                mock_config.get_value.return_value = {
                    "other_server::some_env_var": "other_value"  # Should remain
                }
                mock_config.update_settings = Mock()

                # Delete the tool server
                delete_response = client.delete(
                    f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
                )
                assert delete_response.status_code == 200

                # Config should still be updated (even though no secrets were removed)
                mock_config.update_settings.assert_called_once()
                call_args = mock_config.update_settings.call_args[0][0]

                # Other secrets should remain unchanged
                expected_remaining_secrets = {
                    "other_server::some_env_var": "other_value"
                }
                assert call_args[MCP_SECRETS_KEY] == expected_remaining_secrets


async def test_delete_local_mcp_tool_server_secret_key_not_in_config(
    client, test_project
):
    """Test that deleting a local MCP tool server handles cases where secret keys are not in config"""
    # Create a local MCP tool server with secret env vars
    tool_data = {
        "name": "local_tool_with_missing_secrets",
        "command": "python",
        "args": ["-m", "my_server"],
        "env_vars": {
            "PUBLIC_VAR": "public_value",
            "SECRET_API_KEY": "secret_key_123",
            "ANOTHER_SECRET": "another_secret_value",
        },
        "secret_env_var_keys": ["SECRET_API_KEY", "ANOTHER_SECRET"],
        "description": "Local tool with secret env vars not in config",
        "is_archived": False,
    }

    async with mock_mcp_success():
        with patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id:
            mock_project_from_id.return_value = test_project

            # Create the tool server first (without mocking Config to avoid interference)
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_local_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
            created_tool = create_response.json()
            tool_server_id = created_tool["id"]

            # Now mock Config for the delete operation with missing secrets
            with patch(
                "app.desktop.studio_server.tool_api.Config.shared"
            ) as mock_config_shared:
                mock_config = mock_config_shared.return_value
                # Config doesn't contain the server's secrets (they were never saved or were deleted)
                mock_config.get_value.return_value = {
                    "other_server::some_env_var": "other_value"  # Should remain
                }
                mock_config.update_settings = Mock()

                # Delete the tool server - should not raise an exception
                delete_response = client.delete(
                    f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
                )
                assert delete_response.status_code == 200

                # Config should still be updated
                mock_config.update_settings.assert_called_once()
                call_args = mock_config.update_settings.call_args[0][0]

                # Other secrets should remain unchanged
                expected_remaining_secrets = {
                    "other_server::some_env_var": "other_value"
                }
                assert call_args[MCP_SECRETS_KEY] == expected_remaining_secrets


async def test_delete_local_mcp_tool_server_missing_secret_env_var_keys_property(
    client, test_project
):
    """Test that deleting a local MCP tool server handles missing secret_env_var_keys property gracefully"""
    # Create a local MCP tool server first
    tool_data = {
        "name": "local_tool_without_secret_keys_prop",
        "command": "python",
        "args": ["-m", "test_server"],
        "env_vars": {"DEBUG": "true"},
        "secret_env_var_keys": [],
        "description": "Local tool for testing missing property",
        "is_archived": False,
    }

    async with mock_mcp_success():
        with patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id:
            mock_project_from_id.return_value = test_project

            # Create the local tool server first (without mocking Config to avoid interference)
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_local_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
            created_tool = create_response.json()
            tool_server_id = created_tool["id"]

            # Manually remove the secret_env_var_keys property to simulate old data
            # This requires directly modifying the tool server's properties
            tool_server = tool_server_from_id(test_project.id, tool_server_id)
            del tool_server.properties["secret_env_var_keys"]
            tool_server.save_to_file()

            # Now mock Config for the delete operation
            with patch(
                "app.desktop.studio_server.tool_api.Config.shared"
            ) as mock_config_shared:
                mock_config = mock_config_shared.return_value
                mock_config.get_value.return_value = {
                    "other_server::some_env_var": "other_value"
                }
                mock_config.update_settings = Mock()

                # Delete the tool server - this should handle the missing property gracefully
                delete_response = client.delete(
                    f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
                )
                assert delete_response.status_code == 200

                # Verify that config update was still called
                mock_config.update_settings.assert_called_once()
                call_args = mock_config.update_settings.call_args[0][0]

                # Check that existing secrets remain unchanged since no secret keys were found
                expected_remaining_secrets = {
                    "other_server::some_env_var": "other_value"
                }
                assert call_args[MCP_SECRETS_KEY] == expected_remaining_secrets


async def test_get_tool_server_with_missing_secrets(client, test_project):
    """Test get_tool_server returns missing_secrets when secrets are not configured"""
    # First create a tool server with secret headers
    tool_data = {
        "name": "test_missing_secrets_tool",
        "server_url": "https://example.com/api",
        "headers": {"Authorization": "Bearer token", "X-API-Key": "key"},
        "secret_header_keys": ["Authorization", "X-API-Key"],
        "description": "Tool with missing secrets",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Create the tool with successful validation
        async with mock_mcp_success():
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
        created_tool = create_response.json()
        tool_server_id = created_tool["id"]

        # Mock the missing_secrets method to return missing secrets
        with patch(
            "app.desktop.studio_server.tool_api.tool_server_from_id"
        ) as mock_tool_server_from_id:
            mock_tool_server = Mock()
            mock_tool_server.id = tool_server_id
            mock_tool_server.name = "test_missing_secrets_tool"
            mock_tool_server.type = ToolServerType.remote_mcp
            mock_tool_server.description = "Tool with missing secrets"
            mock_tool_server.created_at = datetime.now().astimezone()
            mock_tool_server.created_by = None
            mock_tool_server.properties = {
                "server_url": "https://example.com/api",
                "headers": {"Authorization": "Bearer token", "X-API-Key": "key"},
                "secret_header_keys": ["Authorization", "X-API-Key"],
                "is_archived": False,
            }
            mock_tool_server.retrieve_secrets.return_value = (
                {},
                ["Authorization", "X-API-Key"],
            )
            mock_tool_server_from_id.return_value = mock_tool_server

            # Get the tool server - should return with missing_secrets and no available_tools
            response = client.get(
                f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
            )

            assert response.status_code == 200
            result = response.json()

            # Verify the tool server details
            assert result["name"] == "test_missing_secrets_tool"
            assert result["type"] == "remote_mcp"
            assert result["description"] == "Tool with missing secrets"

            # Verify missing_secrets is populated
            assert "missing_secrets" in result
            assert set(result["missing_secrets"]) == {"Authorization", "X-API-Key"}

            # Verify available_tools is empty when secrets are missing
            assert "available_tools" in result
            assert result["available_tools"] == []


async def test_get_tool_server_with_some_missing_secrets(client, test_project):
    """Test get_tool_server returns partial missing_secrets when some secrets are missing"""
    # First create a tool server with secret headers
    tool_data = {
        "name": "test_partial_missing_secrets_tool",
        "server_url": "https://example.com/api",
        "headers": {
            "Authorization": "Bearer token",
            "X-API-Key": "key",
            "Content-Type": "application/json",
        },
        "secret_header_keys": ["Authorization", "X-API-Key"],
        "description": "Tool with some missing secrets",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Create the tool with successful validation
        async with mock_mcp_success():
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
        created_tool = create_response.json()
        tool_server_id = created_tool["id"]

        # Mock the missing_secrets method to return only one missing secret
        with patch(
            "app.desktop.studio_server.tool_api.tool_server_from_id"
        ) as mock_tool_server_from_id:
            mock_tool_server = Mock()
            mock_tool_server.id = tool_server_id
            mock_tool_server.name = "test_partial_missing_secrets_tool"
            mock_tool_server.type = ToolServerType.remote_mcp
            mock_tool_server.description = "Tool with some missing secrets"
            mock_tool_server.created_at = datetime.now().astimezone()
            mock_tool_server.created_by = None
            mock_tool_server.properties = {
                "server_url": "https://example.com/api",
                "headers": {
                    "Authorization": "Bearer token",
                    "X-API-Key": "key",
                    "Content-Type": "application/json",
                },
                "secret_header_keys": ["Authorization", "X-API-Key"],
                "is_archived": False,
            }
            mock_tool_server.retrieve_secrets.return_value = (
                {"Authorization": "Bearer token"},
                ["X-API-Key"],
            )  # Only one missing
            mock_tool_server_from_id.return_value = mock_tool_server

            # Get the tool server - should return with missing_secrets and no available_tools
            response = client.get(
                f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
            )

            assert response.status_code == 200
            result = response.json()

            # Verify missing_secrets contains only the missing secret
            assert "missing_secrets" in result
            assert result["missing_secrets"] == ["X-API-Key"]

            # Verify available_tools is empty when any secrets are missing
            assert "available_tools" in result
            assert result["available_tools"] == []


async def test_get_tool_server_no_missing_secrets(client, test_project):
    """Test get_tool_server returns available_tools when no secrets are missing"""
    # First create a tool server with secret headers
    tool_data = {
        "name": "test_no_missing_secrets_tool",
        "server_url": "https://example.com/api",
        "headers": {"Authorization": "Bearer token", "X-API-Key": "key"},
        "secret_header_keys": ["Authorization", "X-API-Key"],
        "description": "Tool with no missing secrets",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Create the tool with successful validation
        async with mock_mcp_success():
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
        created_tool = create_response.json()
        tool_server_id = created_tool["id"]

        # Mock the missing_secrets method to return no missing secrets
        with patch(
            "app.desktop.studio_server.tool_api.tool_server_from_id"
        ) as mock_tool_server_from_id:
            mock_tool_server = Mock()
            mock_tool_server.id = tool_server_id
            mock_tool_server.name = "test_no_missing_secrets_tool"
            mock_tool_server.type = ToolServerType.remote_mcp
            mock_tool_server.description = "Tool with no missing secrets"
            mock_tool_server.created_at = datetime.now().astimezone()
            mock_tool_server.created_by = None
            mock_tool_server.properties = {
                "server_url": "https://example.com/api",
                "headers": {"Authorization": "Bearer token", "X-API-Key": "key"},
                "secret_header_keys": ["Authorization", "X-API-Key"],
                "is_archived": False,
            }
            mock_tool_server.retrieve_secrets.return_value = (
                {"Authorization": "Bearer token", "X-API-Key": "key"},
                [],
            )  # No missing secrets
            mock_tool_server_from_id.return_value = mock_tool_server

            # Mock successful tool retrieval
            mock_tools = [
                Tool(name="test_tool", description="Test tool", inputSchema={}),
                Tool(name="calculator", description="Math tool", inputSchema={}),
            ]

            async with mock_mcp_success(tools=mock_tools):
                # Get the tool server - should return available_tools when no secrets are missing
                response = client.get(
                    f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
                )

                assert response.status_code == 200
                result = response.json()

                # Verify missing_secrets is empty
                assert "missing_secrets" in result
                assert result["missing_secrets"] == []

                # Verify available_tools is populated when no secrets are missing
                assert "available_tools" in result
                assert len(result["available_tools"]) == 2
                tool_names = [tool["name"] for tool in result["available_tools"]]
                assert "test_tool" in tool_names
                assert "calculator" in tool_names


async def test_get_tool_server_local_mcp_with_missing_secrets(client, test_project):
    """Test get_tool_server returns missing_secrets for local MCP servers"""
    # First create a local MCP tool server
    tool_data = {
        "name": "test_local_missing_secrets",
        "command": "python",
        "args": ["-m", "my_server"],
        "env_vars": {
            "API_KEY": "secret_key",
            "PORT": "3000",
            "DATABASE_PASSWORD": "placeholder",
        },
        "secret_env_var_keys": ["API_KEY", "DATABASE_PASSWORD"],
        "description": "Local MCP tool with missing secrets",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Create the local MCP tool server
        async with mock_mcp_success():
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_local_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
        created_tool = create_response.json()
        tool_server_id = created_tool["id"]

        # Mock the missing_secrets method to return missing secrets
        with patch(
            "app.desktop.studio_server.tool_api.tool_server_from_id"
        ) as mock_tool_server_from_id:
            mock_tool_server = Mock()
            mock_tool_server.id = tool_server_id
            mock_tool_server.name = "test_local_missing_secrets"
            mock_tool_server.type = ToolServerType.local_mcp
            mock_tool_server.description = "Local MCP tool with missing secrets"
            mock_tool_server.created_at = datetime.now().astimezone()
            mock_tool_server.created_by = None
            mock_tool_server.properties = {
                "command": "python",
                "args": ["-m", "my_server"],
                "env_vars": {
                    "API_KEY": "secret_key",
                    "PORT": "3000",
                    "DATABASE_PASSWORD": "placeholder",
                },
                "secret_env_var_keys": ["API_KEY", "DATABASE_PASSWORD"],
                "is_archived": False,
            }
            mock_tool_server.retrieve_secrets.return_value = (
                {"API_KEY": "secret_key"},
                ["DATABASE_PASSWORD"],
            )  # Missing secret
            mock_tool_server_from_id.return_value = mock_tool_server

            # Get the tool server - should return with missing_secrets
            response = client.get(
                f"/api/projects/{test_project.id}/tool_servers/{tool_server_id}"
            )

            assert response.status_code == 200
            result = response.json()

            # Verify the tool server details
            assert result["name"] == "test_local_missing_secrets"
            assert result["type"] == "local_mcp"
            assert result["description"] == "Local MCP tool with missing secrets"

            # Verify missing_secrets is populated
            assert "missing_secrets" in result
            assert result["missing_secrets"] == ["DATABASE_PASSWORD"]

            # Verify available_tools is empty when secrets are missing
            assert "available_tools" in result
            assert result["available_tools"] == []


async def test_get_tool_server_kiln_task_success(client, test_project):
    """Test get_tool_server successfully retrieves a Kiln task tool server"""

    with (
        patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id,
        patch("app.desktop.studio_server.tool_api.task_from_id") as mock_task_from_id,
        patch(
            "kiln_ai.tools.kiln_task_tool.project_from_id"
        ) as mock_kiln_project_from_id,
    ):
        mock_project_from_id.return_value = test_project
        mock_kiln_project_from_id.return_value = test_project

        # Create a test task
        task = Task(
            name="Test Task",
            description="Test task for Kiln task tool",
            instruction="Complete the test task",
            parent=test_project,
        )
        task.save_to_file()

        # Set up task_from_id mock to return the task
        def mock_task_from_id_func(project_id, task_id):
            if task_id == str(task.id):
                return task
            else:
                raise HTTPException(status_code=404, detail="Task not found")

        mock_task_from_id.side_effect = mock_task_from_id_func

        # Create a Kiln task tool server
        kiln_task_server = ExternalToolServer(
            name="kiln_task_server",
            type=ToolServerType.kiln_task,
            description="Kiln task server for testing",
            properties={
                "name": "test_kiln_task_tool",
                "description": "Test Kiln task tool",
                "task_id": str(task.id),
                "run_config_id": "default",
                "is_archived": False,
            },
            parent=test_project,
        )
        kiln_task_server.save_to_file()

        # Get the tool server
        response = client.get(
            f"/api/projects/{test_project.id}/tool_servers/{kiln_task_server.id}"
        )

        assert response.status_code == 200
        result = response.json()

        # Verify the tool server details
        assert result["name"] == "kiln_task_server"
        assert result["type"] == "kiln_task"
        assert result["description"] == "Kiln task server for testing"
        assert result["properties"]["name"] == "test_kiln_task_tool"
        assert result["properties"]["description"] == "Test Kiln task tool"
        assert result["properties"]["task_id"] == str(task.id)
        assert result["properties"]["run_config_id"] == "default"
        assert "id" in result

        # Verify available_tools is populated with the Kiln task tool
        assert "available_tools" in result
        assert len(result["available_tools"]) == 1

        tool = result["available_tools"][0]
        assert tool["name"] == "test_kiln_task_tool"
        assert tool["description"] == "Test Kiln task tool"
        assert "inputSchema" in tool


@pytest.fixture
def edit_local_server_data():
    return {
        "name": "edited name",
        "command": "python",
        "args": ["-m", "my_server"],
        "env_vars": {
            "PORT": "3000",
            "DATABASE_PASSWORD": "1234",
        },
        "secret_env_var_keys": ["DATABASE_PASSWORD"],
        "description": "edited description",
        "is_archived": True,
    }


async def test_edit_local_mcp_404(client, test_project, edit_local_server_data):
    """Test edit_local_mcp returns 404 when the tool server does not exist"""
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project
        response = client.patch(
            f"/api/projects/{test_project.id}/edit_local_mcp/123",
            json=edit_local_server_data,
        )
        assert response.status_code == 404
        assert response.json() == {"message": "Tool server not found"}


@pytest.fixture
def existing_local_tool_server(test_project):
    existing_tool_server = ExternalToolServer(
        parent=test_project,
        type=ToolServerType.local_mcp,
        name="test_local_mcp",
        properties={
            "command": "echo",
            "args": ["hello"],
            "is_archived": False,
        },
    )
    existing_tool_server.save_to_file()
    return existing_tool_server


@pytest.fixture
def existing_remote_tool_server(test_project):
    existing_tool_server = ExternalToolServer(
        parent=test_project,
        type=ToolServerType.remote_mcp,
        name="test_remote_mcp",
        properties={
            "server_url": "https://example.com",
            "headers": {},
            "is_archived": False,
        },
    )
    existing_tool_server.save_to_file()
    return existing_tool_server


async def test_edit_local_mcp_wrong_type(
    client, test_project, edit_local_server_data, existing_remote_tool_server
):
    """Test edit_local_mcp returns 400 when the tool server is not a local MCP server"""

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project
        response = client.patch(
            f"/api/projects/{test_project.id}/edit_local_mcp/{existing_remote_tool_server.id}",
            json=edit_local_server_data,
        )
        assert response.status_code == 400
        assert response.json() == {
            "message": "Existing tool server is not a local MCP server. You can't edit a non-local MCP server with this endpoint."
        }


async def test_edit_local_mcp(
    client, test_project, edit_local_server_data, existing_local_tool_server
):
    """Test edit_local_mcp updates the tool server"""
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Create the local MCP tool server
        async with mock_mcp_success():
            edit_response = client.patch(
                f"/api/projects/{test_project.id}/edit_local_mcp/{existing_local_tool_server.id}",
                json=edit_local_server_data,
            )
            assert edit_response.status_code == 200
            response_json = edit_response.json()
            assert response_json["name"] == "edited name"
            assert response_json["type"] == ToolServerType.local_mcp
            assert response_json["description"] == "edited description"
            assert response_json["properties"]["command"] == "python"
            assert response_json["properties"]["args"] == ["-m", "my_server"]
            assert response_json["properties"]["env_vars"].keys() == {
                "PORT",
            }
            assert response_json["properties"]["env_vars"]["PORT"] == "3000"
            assert response_json["properties"]["secret_env_var_keys"] == [
                "DATABASE_PASSWORD",
            ]
            assert response_json["properties"]["is_archived"] is True

            # Verify the tool server changes were saved to file
            loaded_tool_server = ExternalToolServer.load_from_file(
                existing_local_tool_server.path
            )
            assert loaded_tool_server.name == "edited name"
            assert loaded_tool_server.type == ToolServerType.local_mcp
            assert loaded_tool_server.description == "edited description"
            assert loaded_tool_server.properties["command"] == "python"
            assert loaded_tool_server.properties["args"] == ["-m", "my_server"]
            assert loaded_tool_server.properties["env_vars"].keys() == {
                "PORT",
            }
            assert loaded_tool_server.properties["env_vars"]["PORT"] == "3000"
            assert loaded_tool_server.properties["secret_env_var_keys"] == [
                "DATABASE_PASSWORD",
            ]
            assert loaded_tool_server.properties["is_archived"] is True


@pytest.fixture
def edit_remote_server_data():
    return {
        "name": "edited name",
        "description": "edited description",
        "server_url": "https://example.com/edited",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer token",
        },
        "secret_header_keys": ["Authorization"],
        "is_archived": True,
    }


async def test_edit_remote_mcp_404(client, test_project, edit_remote_server_data):
    """Test edit_remote_mcp returns 404 when the tool server does not exist"""
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project
        response = client.patch(
            f"/api/projects/{test_project.id}/edit_remote_mcp/123",
            json=edit_remote_server_data,
        )
        assert response.status_code == 404
        assert response.json() == {"message": "Tool server not found"}


async def test_edit_remote_mcp_wrong_type(
    client, test_project, edit_remote_server_data, existing_local_tool_server
):
    """Test edit_local_mcp returns 400 when the tool server is not a local MCP server"""

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project
        response = client.patch(
            f"/api/projects/{test_project.id}/edit_remote_mcp/{existing_local_tool_server.id}",
            json=edit_remote_server_data,
        )
        assert response.status_code == 400
        assert response.json() == {
            "message": "Existing tool server is not a remote MCP server. You can't edit a non-remote MCP server with this endpoint."
        }


async def test_edit_remote_mcp(
    client, test_project, edit_remote_server_data, existing_remote_tool_server
):
    """Test edit_local_mcp updates the tool server"""
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Create the local MCP tool server
        async with mock_mcp_success():
            edit_response = client.patch(
                f"/api/projects/{test_project.id}/edit_remote_mcp/{existing_remote_tool_server.id}",
                json=edit_remote_server_data,
            )
            assert edit_response.status_code == 200
            response_json = edit_response.json()
            assert response_json["name"] == "edited name"
            assert response_json["type"] == ToolServerType.remote_mcp
            assert response_json["description"] == "edited description"
            assert (
                response_json["properties"]["server_url"]
                == "https://example.com/edited"
            )
            assert response_json["properties"]["headers"] == {
                "Content-Type": "application/json",
            }
            assert response_json["properties"]["secret_header_keys"] == [
                "Authorization"
            ]
            assert response_json["properties"]["is_archived"] is True

            # Verify the tool server changes were saved to file
            loaded_tool_server = ExternalToolServer.load_from_file(
                existing_remote_tool_server.path
            )
            assert loaded_tool_server.name == "edited name"
            assert loaded_tool_server.type == ToolServerType.remote_mcp
            assert loaded_tool_server.description == "edited description"
            assert (
                loaded_tool_server.properties["server_url"]
                == "https://example.com/edited"
            )
            assert loaded_tool_server.properties["headers"] == {
                "Content-Type": "application/json",
            }
            assert loaded_tool_server.properties["secret_header_keys"] == [
                "Authorization",
            ]
            assert loaded_tool_server.properties["is_archived"] is True


@pytest.mark.parametrize(
    "fixture_name, expected_type",
    [
        ("existing_remote_tool_server", ToolServerType.remote_mcp),
        ("existing_local_tool_server", ToolServerType.local_mcp),
    ],
)
async def test_archive_tool_server_skips_connectivity_validation(
    client,
    test_project,
    request,
    fixture_name,
    expected_type,
):
    tool_server = request.getfixturevalue(fixture_name)

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        with patch(
            "app.desktop.studio_server.tool_api.validate_tool_server_connectivity",
            new_callable=AsyncMock,
        ) as mock_validate:
            response = client.post(
                f"/api/projects/{test_project.id}/tool_servers/{tool_server.id}/archive",
                json={"is_archived": True},
            )

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["type"] == expected_type
        assert response_json["properties"]["is_archived"] is True
        mock_validate.assert_not_awaited()

        loaded_tool_server = ExternalToolServer.load_from_file(tool_server.path)
        assert loaded_tool_server.properties["is_archived"] is True


@pytest.mark.parametrize(
    "fixture_name, endpoint, property_key, bad_data",
    [
        # Test 1: Remote MCP with bad url
        (
            "existing_remote_tool_server",
            "edit_remote_mcp",
            "server_url",
            {"server_url": "http://invalid-url.com"},
        ),
        # Test 2: Local MCP with bad command
        (
            "existing_local_tool_server",
            "edit_local_mcp",
            "command",
            {"command": "invalid-command", "args": []},
        ),
    ],
    ids=["remote_mcp", "local_mcp"],
)
async def test_edit_mcp_does_not_keep_bad_data_in_memory(
    client,
    test_project,
    request,
    mock_project_from_id,
    fixture_name,
    endpoint,
    property_key,
    bad_data,
):
    """Test editing mcp servers with validation failure will not keep bad data in memory"""

    test_server = request.getfixturevalue(fixture_name)

    # Load ExternalToolServer in memory via tool_server_from_id and store the original value
    # tool_server_from_id needs project_from_id to be mocked to return properly
    mock_project_from_id(test_project)
    original_server = tool_server_from_id(test_project.id, test_server.id)
    original_value = original_server.properties[property_key]

    # Call patch endpoint with bad data and force validation failure
    bad_data = {
        "name": test_server.name,
        "description": test_server.description,
        "is_archived": False,
        **bad_data,
    }
    async with mock_mcp_connection_error():
        with pytest.raises(Exception, match="Connection failed"):
            await client.patch(
                f"/api/projects/{test_project.id}/{endpoint}/{test_server.id}",
                json=bad_data,
            )

    # Read the server from memory again and ensure it's not changed
    post_validation_server = tool_server_from_id(test_project.id, test_server.id)
    assert post_validation_server.properties[property_key] == original_value


def test_get_search_tools_success(client, test_project, mock_project_from_id):
    """Test get_search_tools returns RAG configs as search tools"""

    # Create some RAG configs
    rag_config_1 = RagConfig(
        parent=test_project,
        name="Test Search Tool 1",
        tool_name="test_search_tool_1",
        tool_description="First test search tool",
        description="First test search tool",
        extractor_config_id="extractor123",
        chunker_config_id="chunker456",
        embedding_config_id="embedding789",
        vector_store_config_id="vector_store123",
    )

    rag_config_2 = RagConfig(
        parent=test_project,
        name="Test Search Tool 2",
        description=None,
        tool_name="test_search_tool_2",
        tool_description="Second test search tool",
        extractor_config_id="extractor456",
        chunker_config_id="chunker789",
        embedding_config_id="embedding123",
        vector_store_config_id="vector_store456",
    )

    # Save the RAG configs
    rag_config_1.save_to_file()
    rag_config_2.save_to_file()

    # Get search tools
    response = client.get(f"/api/projects/{test_project.id}/search_tools")
    assert response.status_code == 200

    search_tools = response.json()
    assert len(search_tools) == 2

    # Check first search tool
    tool1 = next(t for t in search_tools if t["tool_name"] == "test_search_tool_1")
    assert tool1["id"] == str(rag_config_1.id)
    assert tool1["name"] == "Test Search Tool 1"
    assert tool1["description"] == "First test search tool"

    # Check second search tool
    tool2 = next(t for t in search_tools if t["tool_name"] == "test_search_tool_2")
    assert tool2["id"] == str(rag_config_2.id)
    assert tool2["description"] == "Second test search tool"
    assert tool2["name"] == "Test Search Tool 2"


# RAG-specific tests
async def test_get_available_tools_with_rag_configs(client, test_project):
    """Test get_available_tools includes RAG configs when there's an external tool server"""

    # Create some RAG configs
    rag_config_1 = RagConfig(
        parent=test_project,
        name="Test RAG Config 1",
        description="First test RAG configuration",
        tool_name="test_rag_config_1",
        tool_description="First test RAG configuration",
        extractor_config_id="extractor123",
        chunker_config_id="chunker456",
        embedding_config_id="embedding789",
        vector_store_config_id="vector_store123",
    )

    rag_config_2 = RagConfig(
        parent=test_project,
        name="Test RAG Config 2",
        description=None,  # Test None description
        tool_name="test_rag_config_2",
        tool_description="Second test RAG configuration",
        extractor_config_id="extractor123",
        chunker_config_id="chunker456",
        embedding_config_id="embedding789",
        vector_store_config_id="vector_store456",
    )

    # Save the RAG configs
    rag_config_1.save_to_file()
    rag_config_2.save_to_file()

    # Create an MCP tool server to trigger the RAG set inclusion
    tool_data = {
        "name": "test_server",
        "server_url": "https://example.com/mcp",
        "headers": {},
        "description": "Test server",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Create the MCP server
        async with mock_mcp_success():
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200

        # Mock no tools from MCP server
        async with mock_mcp_success():
            response = client.get(f"/api/projects/{test_project.id}/available_tools")

            assert response.status_code == 200
            result = response.json()

            # Built-in AI Models set + RAG set.
            assert len(result) == 2
            assert _builtin_ai_models_set(result) is not None
            rag_set = next(s for s in result if s["set_name"] == "Search Tools (RAG)")
            assert len(rag_set["tools"]) == 2

            # Verify RAG tool details: name is the display name, function_name
            # the callable tool name.
            tool_names = [tool["name"] for tool in rag_set["tools"]]

            assert "Test RAG Config 1" in tool_names
            assert "Test RAG Config 2" in tool_names

            # Verify tool IDs are properly formatted
            for tool in rag_set["tools"]:
                assert tool["id"].startswith("kiln_tool::rag::")

            # Find specific tools and check their fields
            config1_tool = next(
                t for t in rag_set["tools"] if t["name"] == "Test RAG Config 1"
            )
            assert config1_tool["function_name"] == "test_rag_config_1"
            assert config1_tool["description"] == "First test RAG configuration"

            config2_tool = next(
                t for t in rag_set["tools"] if t["name"] == "Test RAG Config 2"
            )
            assert config2_tool["function_name"] == "test_rag_config_2"
            assert config2_tool["description"] == "Second test RAG configuration"


async def test_get_available_tools_with_rag_and_mcp(client, test_project):
    """Test get_available_tools with both RAG configs and MCP servers"""

    # Create a RAG config
    rag_config = RagConfig(
        parent=test_project,
        name="mixed_test_rag",
        description="RAG config for mixed test",
        tool_name="mixed_test_rag",
        tool_description="RAG config for mixed test",
        extractor_config_id="extractor123",
        chunker_config_id="chunker456",
        embedding_config_id="embedding789",
        vector_store_config_id="vector_store123",
    )
    rag_config.save_to_file()

    # Create an MCP tool server
    tool_data = {
        "name": "mixed_test_server",
        "server_url": "https://example.com/mcp",
        "headers": {"Authorization": "Bearer token"},
        "description": "MCP server for mixed test",
        "is_archived": False,
    }

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Create the MCP server
        async with mock_mcp_success():
            create_response = client.post(
                f"/api/projects/{test_project.id}/connect_remote_mcp",
                json=tool_data,
            )
            assert create_response.status_code == 200
        created_tool = create_response.json()
        server_id = created_tool["id"]

        # Mock tools for the MCP server
        mock_tools = [
            Tool(name="mcp_tool", description="MCP tool", inputSchema={}),
        ]

        async with mock_mcp_success(tools=mock_tools):
            # Get available tools
            response = client.get(f"/api/projects/{test_project.id}/available_tools")

            assert response.status_code == 200
            result = response.json()

            # Built-in AI Models + RAG + MCP sets.
            assert len(result) == 3
            assert _builtin_ai_models_set(result) is not None

            # Find both sets
            mcp_set = next(
                (s for s in result if s["set_name"] == "MCP Server: mixed_test_server"),
                None,
            )
            rag_set = next(
                (s for s in result if s["set_name"] == "Search Tools (RAG)"),
                None,
            )

            assert mcp_set is not None
            assert rag_set is not None

            # Verify MCP tools
            assert len(mcp_set["tools"]) == 1
            assert mcp_set["tools"][0]["name"] == "mcp_tool"
            assert mcp_set["tools"][0]["id"].startswith(f"mcp::remote::{server_id}::")

            # Verify RAG tools
            assert len(rag_set["tools"]) == 1
            assert rag_set["tools"][0]["name"] == "mixed_test_rag"
            assert rag_set["tools"][0]["id"].startswith("kiln_tool::rag::")


def test_get_search_tools_excludes_archived(client, test_project, mock_project_from_id):
    """Archived RAG configs should not be returned by /search_tools."""

    active = RagConfig(
        parent=test_project,
        name="Active Search Tool",
        tool_name="active_tool",
        tool_description="Active",
        extractor_config_id="e1",
        chunker_config_id="c1",
        embedding_config_id="em1",
        vector_store_config_id="v1",
        is_archived=False,
    )
    archived = RagConfig(
        parent=test_project,
        name="Archived Search Tool",
        tool_name="archived_tool",
        tool_description="Archived",
        extractor_config_id="e2",
        chunker_config_id="c2",
        embedding_config_id="em2",
        vector_store_config_id="v2",
        is_archived=True,
    )
    active.save_to_file()
    archived.save_to_file()

    response = client.get(f"/api/projects/{test_project.id}/search_tools")
    assert response.status_code == 200
    tools = response.json()
    # Only the active one should be returned
    assert len(tools) == 1
    assert tools[0]["tool_name"] == "active_tool"
    assert tools[0]["name"] == "Active Search Tool"


async def test_available_tools_excludes_archived_rag_and_kiln_task_tools(
    client, test_project
):
    """Archived RAG configs and kiln task tools should be excluded from their respective sets in /available_tools."""

    # Create active and archived RAG configs
    active = RagConfig(
        parent=test_project,
        name="Active RAG",
        description="",
        tool_name="active_rag",
        tool_description="Active desc",
        extractor_config_id="e1",
        chunker_config_id="c1",
        embedding_config_id="em1",
        vector_store_config_id="v1",
        is_archived=False,
    )
    archived = RagConfig(
        parent=test_project,
        name="Archived RAG",
        description="",
        tool_name="archived_rag",
        tool_description="Archived desc",
        extractor_config_id="e2",
        chunker_config_id="c2",
        embedding_config_id="em2",
        vector_store_config_id="v2",
        is_archived=True,
    )
    active.save_to_file()
    archived.save_to_file()

    # Create active and archived kiln task tool servers
    active_kiln_task = ExternalToolServer(
        name="active_kiln_task_server",
        type=ToolServerType.kiln_task,
        description="Active kiln task server",
        properties={
            "name": "active_task_tool",
            "description": "Active task tool",
            "task_id": "task_1",
            "run_config_id": "run_config_1",
            "is_archived": False,
        },
        parent=test_project,
    )

    archived_kiln_task = ExternalToolServer(
        name="archived_kiln_task_server",
        type=ToolServerType.kiln_task,
        description="Archived kiln task server",
        properties={
            "name": "archived_task_tool",
            "description": "Archived task tool",
            "task_id": "task_2",
            "run_config_id": "run_config_2",
            "is_archived": True,
        },
        parent=test_project,
    )
    active_kiln_task.save_to_file()
    archived_kiln_task.save_to_file()

    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        response = client.get(f"/api/projects/{test_project.id}/available_tools")
        assert response.status_code == 200
        result = response.json()

        # Built-in AI Models + RAG + Kiln task sets.
        assert len(result) == 3
        assert _builtin_ai_models_set(result) is not None

        rag_set = next(
            (s for s in result if s["set_name"] == "Search Tools (RAG)"), None
        )
        kiln_task_set = next(
            (s for s in result if s["set_name"] == "Kiln Tasks as Tools"), None
        )

        assert rag_set is not None
        assert kiln_task_set is not None

        # Only the active RAG config should be present
        assert len(rag_set["tools"]) == 1
        assert rag_set["tools"][0]["name"] == "Active RAG"
        assert rag_set["tools"][0]["function_name"] == "active_rag"

        # Only the active kiln task tool should be present
        assert len(kiln_task_set["tools"]) == 1
        assert kiln_task_set["tools"][0]["name"] == "active_task_tool"


class TestExternalToolApiDescription:
    """Test cases for ExternalToolApiDescription class methods."""

    def test_tool_from_mcp_tool_with_all_fields(self):
        """Test creating ExternalToolApiDescription from MCP Tool with all fields."""
        mcp_tool = Tool(
            name="test_tool",
            description="A test tool description",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                    "param2": {"type": "number", "description": "Second parameter"},
                },
                "required": ["param1"],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "Output result"},
                },
                "required": ["result"],
            },
        )

        result = ExternalToolApiDescription.tool_from_mcp_tool(mcp_tool)

        assert result.name == "test_tool"
        assert result.description == "A test tool description"
        assert result.inputSchema == {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "number", "description": "Second parameter"},
            },
            "required": ["param1"],
        }
        assert result.outputSchema == {
            "type": "object",
            "properties": {
                "result": {"type": "string", "description": "Output result"},
            },
            "required": ["result"],
        }

    def test_tool_from_mcp_tool_with_minimal_fields(self):
        """Test creating ExternalToolApiDescription from MCP Tool with minimal fields."""
        mcp_tool = Tool(
            name="minimal_tool",
            description=None,
            inputSchema={},
        )

        result = ExternalToolApiDescription.tool_from_mcp_tool(mcp_tool)

        assert result.name == "minimal_tool"
        assert result.description is None
        assert result.inputSchema == {}
        assert result.outputSchema is None

    @pytest.mark.asyncio
    async def test_tool_from_kiln_task_tool_with_all_fields(self):
        """Test creating ExternalToolApiDescription from KilnTaskTool with all fields."""
        # Create a mock KilnTaskTool
        mock_tool = AsyncMock()
        mock_tool.name.return_value = "kiln_task_tool"
        mock_tool.description.return_value = "A Kiln task tool description"
        mock_tool.parameters_schema = {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input parameter"},
            },
            "required": ["input"],
        }

        result = await ExternalToolApiDescription.tool_from_kiln_task_tool(mock_tool)

        # Verify that the async methods were called
        mock_tool.name.assert_called_once()
        mock_tool.description.assert_called_once()

        assert result.name == "kiln_task_tool"
        assert result.description == "A Kiln task tool description"
        assert result.inputSchema == {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input parameter"},
            },
            "required": ["input"],
        }
        assert result.outputSchema is None

    @pytest.mark.asyncio
    async def test_tool_from_kiln_task_tool_with_minimal_fields(self):
        """Test creating ExternalToolApiDescription from KilnTaskTool with minimal fields."""
        # Create a mock KilnTaskTool with minimal fields
        mock_tool = AsyncMock()
        mock_tool.name.return_value = "minimal_kiln_tool"
        mock_tool.description.return_value = ""
        mock_tool.parameters_schema = None

        result = await ExternalToolApiDescription.tool_from_kiln_task_tool(mock_tool)

        assert result.name == "minimal_kiln_tool"
        assert result.description == ""
        assert result.inputSchema == {}
        assert result.outputSchema is None


@pytest.mark.asyncio
async def test_get_kiln_task_tools_success(client, test_project):
    """Test get_kiln_task_tools successfully retrieves kiln task tools"""

    with (
        patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id,
        patch("app.desktop.studio_server.tool_api.task_from_id") as mock_task_from_id,
    ):
        mock_project_from_id.return_value = test_project

        # Create test tasks
        task1 = Task(
            name="Test Task 1",
            description="First test task",
            instruction="Complete the first test task",
            parent=test_project,
        )
        task1.save_to_file()

        task2 = Task(
            name="Test Task 2",
            description="Second test task",
            instruction="Complete the second test task",
            parent=test_project,
        )
        task2.save_to_file()

        # Set up task_from_id mock to return the appropriate tasks
        def mock_task_from_id_func(project_id, task_id):
            if task_id == str(task1.id):
                return task1
            elif task_id == str(task2.id):
                return task2
            else:
                raise HTTPException(status_code=404, detail="Task not found")

        mock_task_from_id.side_effect = mock_task_from_id_func

        # Create kiln task tool servers
        kiln_task_server_1 = ExternalToolServer(
            name="kiln_task_server_1",
            type=ToolServerType.kiln_task,
            description="First kiln task server",
            properties={
                "name": "test_task_tool_1",
                "description": "First test task tool",
                "task_id": str(task1.id),
                "run_config_id": "run_config_1",
                "is_archived": False,
            },
            parent=test_project,
        )
        kiln_task_server_1.save_to_file()

        kiln_task_server_2 = ExternalToolServer(
            name="kiln_task_server_2",
            type=ToolServerType.kiln_task,
            description="Second kiln task server",
            properties={
                "name": "test_task_tool_2",
                "description": "Second test task tool",
                "task_id": str(task2.id),
                "run_config_id": "run_config_2",
                "is_archived": True,
            },
            parent=test_project,
        )
        kiln_task_server_2.save_to_file()

        # Create a non-kiln task server to ensure it's filtered out
        mcp_server = ExternalToolServer(
            name="mcp_server",
            type=ToolServerType.remote_mcp,
            description="MCP server",
            properties={
                "server_url": "https://example.com/mcp",
                "is_archived": False,
            },
            parent=test_project,
        )
        mcp_server.save_to_file()

        # Make the API call
        response = client.get(f"/api/projects/{test_project.id}/kiln_task_tools")

        assert response.status_code == 200
        results = response.json()

        # Should return 2 kiln task tools
        assert len(results) == 2

        # Find tools by their names since order may vary
        tool1_result = next(t for t in results if t["tool_name"] == "test_task_tool_1")
        tool2_result = next(t for t in results if t["tool_name"] == "test_task_tool_2")

        # Verify first tool
        assert tool1_result["tool_server_id"] == str(kiln_task_server_1.id)
        assert tool1_result["tool_name"] == "test_task_tool_1"
        assert tool1_result["tool_description"] == "First test task tool"
        assert tool1_result["task_id"] == str(task1.id)
        assert tool1_result["task_name"] == "Test Task 1"
        assert tool1_result["task_description"] == "First test task"
        assert tool1_result["is_archived"] is False
        assert "created_at" in tool1_result

        # Verify second tool
        assert tool2_result["tool_server_id"] == str(kiln_task_server_2.id)
        assert tool2_result["tool_name"] == "test_task_tool_2"
        assert tool2_result["tool_description"] == "Second test task tool"
        assert tool2_result["task_id"] == str(task2.id)
        assert tool2_result["task_name"] == "Test Task 2"
        assert tool2_result["task_description"] == "Second test task"
        assert tool2_result["is_archived"] is True
        assert "created_at" in tool2_result


@pytest.mark.asyncio
async def test_get_kiln_task_tools_no_tools(client, test_project):
    """Test get_kiln_task_tools returns empty list when no kiln task tools exist"""
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Create a non-kiln task server
        mcp_server = ExternalToolServer(
            name="mcp_server",
            type=ToolServerType.remote_mcp,
            description="MCP server",
            properties={
                "server_url": "https://example.com/mcp",
                "is_archived": False,
            },
            parent=test_project,
        )
        mcp_server.save_to_file()

        # Make the API call
        response = client.get(f"/api/projects/{test_project.id}/kiln_task_tools")

        assert response.status_code == 200
        results = response.json()
        assert len(results) == 0


@pytest.mark.asyncio
async def test_get_kiln_task_tools_invalid_task_reference(client, test_project):
    """Test get_kiln_task_tools handles invalid task references gracefully"""

    with (
        patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id,
        patch("app.desktop.studio_server.tool_api.task_from_id") as mock_task_from_id,
    ):
        mock_project_from_id.return_value = test_project

        # Create a valid task first
        valid_task = Task(
            name="Valid Task",
            description="Valid test task",
            instruction="Complete the valid test task",
            parent=test_project,
        )
        valid_task.save_to_file()

        # Set up task_from_id mock to raise HTTPException for invalid task_id
        def mock_task_from_id_func(project_id, task_id):
            if task_id == "non_existent_task_id":
                raise HTTPException(status_code=404, detail="Task not found")
            elif task_id == str(valid_task.id):
                return valid_task
            else:
                # This shouldn't be called in this test
                raise HTTPException(status_code=404, detail="Unexpected task_id")

        mock_task_from_id.side_effect = mock_task_from_id_func

        # Create kiln task server with invalid task_id
        kiln_task_server = ExternalToolServer(
            name="kiln_task_server_invalid",
            type=ToolServerType.kiln_task,
            description="Kiln task server with invalid task",
            properties={
                "name": "invalid_task_tool",
                "description": "Tool with invalid task reference",
                "task_id": "non_existent_task_id",
                "run_config_id": "run_config_invalid",
                "is_archived": False,
            },
            parent=test_project,
        )
        kiln_task_server.save_to_file()

        # Create a valid kiln task server
        valid_kiln_task_server = ExternalToolServer(
            name="kiln_task_server_valid",
            type=ToolServerType.kiln_task,
            description="Valid kiln task server",
            properties={
                "name": "valid_task_tool",
                "description": "Tool with valid task reference",
                "task_id": str(valid_task.id),
                "run_config_id": "run_config_valid",
                "is_archived": False,
            },
            parent=test_project,
        )
        valid_kiln_task_server.save_to_file()

        # Make the API call
        response = client.get(f"/api/projects/{test_project.id}/kiln_task_tools")

        assert response.status_code == 200
        results = response.json()

        # Should only return the valid tool, invalid one should be skipped
        assert len(results) == 1
        assert results[0]["tool_server_id"] == str(valid_kiln_task_server.id)
        assert results[0]["tool_name"] == "valid_task_tool"


@pytest.mark.asyncio
async def test_get_kiln_task_tools_empty_task_id(client, test_project):
    """Test get_kiln_task_tools handles empty task_id in properties"""
    with patch(
        "app.desktop.studio_server.tool_api.project_from_id"
    ) as mock_project_from_id:
        mock_project_from_id.return_value = test_project

        # Create kiln task server with empty task_id (this will pass validation but be skipped by endpoint)
        kiln_task_server = ExternalToolServer(
            name="kiln_task_server_empty_task_id",
            type=ToolServerType.kiln_task,
            description="Kiln task server with empty task_id",
            properties={
                "name": "empty_task_id_tool",
                "description": "Tool with empty task_id",
                "task_id": "",  # Empty string
                "run_config_id": "run_config_empty_task",
                "is_archived": False,
            },
            parent=test_project,
        )
        kiln_task_server.save_to_file()

        # Make the API call
        response = client.get(f"/api/projects/{test_project.id}/kiln_task_tools")

        assert response.status_code == 200
        results = response.json()

        # Should return empty list since empty task_id is falsy
        assert len(results) == 0


@pytest.mark.asyncio
async def test_add_kiln_task_tool_validation_success(client, test_project):
    """Test add_kiln_task_tool succeeds with valid task and run config (validates _validate_kiln_task_tool_task_and_run_config)"""

    # Create a test task
    task = Task(
        name="Test Task",
        description="Test task for validation",
        instruction="Complete the test task",
        parent=test_project,
    )
    task.save_to_file()

    # Create a run config with ID "default" for the task
    run_config = TaskRunConfig(
        name="default",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name="openai",
            prompt_id=PromptGenerators.SIMPLE,
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=task,
    )
    run_config.save_to_file()

    # Create tool data with valid task and run config
    tool_data = {
        "name": "test_tool",
        "description": "Test tool",
        "task_id": str(task.id),
        "run_config_id": str(run_config.id),
        "is_archived": False,
    }

    with (
        patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id,
        patch("app.desktop.studio_server.tool_api.task_from_id") as mock_task_from_id,
    ):
        mock_project_from_id.return_value = test_project
        mock_task_from_id.return_value = task

        # Should succeed without raising any exception
        response = client.post(
            f"/api/projects/{test_project.id}/kiln_task_tool", json=tool_data
        )

        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "test_tool"
        assert result["type"] == "kiln_task"

        # Verify task_from_id was called with correct parameters (validates the validation function)
        mock_task_from_id.assert_called_once_with(str(test_project.id), str(task.id))


@pytest.mark.asyncio
async def test_add_kiln_task_tool_validation_task_not_found(client, test_project):
    """Test add_kiln_task_tool raises exception when task is not found (validates _validate_kiln_task_tool_task_and_run_config)"""

    # Create tool data with non-existent task
    tool_data = {
        "name": "test_tool",
        "description": "Test tool",
        "task_id": "non_existent_task_id",
        "run_config_id": "default",
        "is_archived": False,
    }

    with patch("app.desktop.studio_server.tool_api.task_from_id") as mock_task_from_id:
        # Mock task_from_id to raise HTTPException
        mock_task_from_id.side_effect = HTTPException(
            status_code=404, detail="Task not found"
        )

        # Should raise HTTPException
        response = client.post(
            f"/api/projects/{test_project.id}/kiln_task_tool", json=tool_data
        )

        assert response.status_code == 404
        assert "Task not found" in response.json()["message"]


@pytest.mark.asyncio
async def test_add_kiln_task_tool_validation_run_config_not_found(client, test_project):
    """Test add_kiln_task_tool raises exception when run config is not found (validates _validate_kiln_task_tool_task_and_run_config)"""

    # Create a test task
    task = Task(
        name="Test Task",
        description="Test task for validation",
        instruction="Complete the test task",
        parent=test_project,
    )
    task.save_to_file()

    # Create a run config with ID "default" for the task
    run_config = TaskRunConfig(
        name="default",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name="openai",
            prompt_id=PromptGenerators.SIMPLE,
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=task,
    )
    run_config.save_to_file()

    # Create tool data with non-existent run config
    tool_data = {
        "name": "test_tool",
        "description": "Test tool",
        "task_id": str(task.id),
        "run_config_id": "non_existent_run_config",
        "is_archived": False,
    }

    with patch("app.desktop.studio_server.tool_api.task_from_id") as mock_task_from_id:
        # Mock task_from_id to return our test task
        mock_task_from_id.return_value = task

        # Should raise HTTPException for run config not found
        response = client.post(
            f"/api/projects/{test_project.id}/kiln_task_tool", json=tool_data
        )

        assert response.status_code == 400
        assert (
            "Run config not found for the specified task" in response.json()["message"]
        )


@pytest.mark.asyncio
async def test_edit_kiln_task_tool_validation_success(client, test_project):
    """Test edit_kiln_task_tool succeeds with valid task and run config (validates _validate_kiln_task_tool_task_and_run_config)"""

    # Create a test task
    task = Task(
        name="Test Task",
        description="Test task for validation",
        instruction="Complete the test task",
        parent=test_project,
    )
    task.save_to_file()

    # Create a run config with ID "default" for the task
    run_config = TaskRunConfig(
        name="default",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name="openai",
            prompt_id=PromptGenerators.SIMPLE,
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=task,
    )
    run_config.save_to_file()

    # Create an existing kiln task tool server
    existing_tool_server = ExternalToolServer(
        name="existing_tool",
        type=ToolServerType.kiln_task,
        description="Existing tool",
        properties={
            "name": "existing_tool",
            "description": "Existing tool",
            "task_id": str(task.id),
            "run_config_id": str(run_config.id),
            "is_archived": False,
        },
        parent=test_project,
    )
    existing_tool_server.save_to_file()

    # Create updated tool data with valid task and run config
    tool_data = {
        "name": "updated_tool",
        "description": "Updated tool",
        "task_id": str(task.id),
        "run_config_id": str(run_config.id),
        "is_archived": False,
    }

    with (
        patch(
            "app.desktop.studio_server.tool_api.project_from_id"
        ) as mock_project_from_id,
        patch("app.desktop.studio_server.tool_api.task_from_id") as mock_task_from_id,
    ):
        mock_project_from_id.return_value = test_project
        mock_task_from_id.return_value = task

        # Should succeed without raising any exception
        response = client.patch(
            f"/api/projects/{test_project.id}/edit_kiln_task_tool/{existing_tool_server.id}",
            json=tool_data,
        )

        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "updated_tool"
        assert result["type"] == "kiln_task"

        # Verify task_from_id was called with correct parameters (validates the validation function)
        mock_task_from_id.assert_called_once_with(str(test_project.id), str(task.id))


@pytest.mark.asyncio
async def test_edit_kiln_task_tool_validation_task_not_found(client, test_project):
    """Test edit_kiln_task_tool raises exception when task is not found (validates _validate_kiln_task_tool_task_and_run_config)"""

    # Create an existing kiln task tool server
    existing_tool_server = ExternalToolServer(
        name="existing_tool",
        type=ToolServerType.kiln_task,
        description="Existing tool",
        properties={
            "name": "existing_tool",
            "description": "Existing tool",
            "task_id": "old_task_id",
            "run_config_id": "default",
            "is_archived": False,
        },
        parent=test_project,
    )
    existing_tool_server.save_to_file()

    # Create tool data with non-existent task
    tool_data = {
        "name": "updated_tool",
        "description": "Updated tool",
        "task_id": "non_existent_task_id",
        "run_config_id": "default",
        "is_archived": False,
    }

    with patch("app.desktop.studio_server.tool_api.task_from_id") as mock_task_from_id:
        # Mock task_from_id to raise HTTPException
        mock_task_from_id.side_effect = HTTPException(
            status_code=404, detail="Task not found"
        )

        # Should raise HTTPException
        response = client.patch(
            f"/api/projects/{test_project.id}/edit_kiln_task_tool/{existing_tool_server.id}",
            json=tool_data,
        )

        assert response.status_code == 404
        assert "Task not found" in response.json()["message"]


@pytest.mark.asyncio
async def test_edit_kiln_task_tool_validation_run_config_not_found(
    client, test_project
):
    """Test edit_kiln_task_tool raises exception when run config is not found (validates _validate_kiln_task_tool_task_and_run_config)"""

    # Create a test task
    task = Task(
        name="Test Task",
        description="Test task for validation",
        instruction="Complete the test task",
        parent=test_project,
    )
    task.save_to_file()

    # Create a run config with ID "default" for the task
    run_config = TaskRunConfig(
        name="default",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name="openai",
            prompt_id=PromptGenerators.SIMPLE,
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=task,
    )
    run_config.save_to_file()

    # Create an existing kiln task tool server
    existing_tool_server = ExternalToolServer(
        name="existing_tool",
        type=ToolServerType.kiln_task,
        description="Existing tool",
        properties={
            "name": "existing_tool",
            "description": "Existing tool",
            "task_id": str(task.id),
            "run_config_id": str(run_config.id),
            "is_archived": False,
        },
        parent=test_project,
    )
    existing_tool_server.save_to_file()

    # Create tool data with non-existent run config
    tool_data = {
        "name": "updated_tool",
        "description": "Updated tool",
        "task_id": str(task.id),
        "run_config_id": "non_existent_run_config",
        "is_archived": False,
    }

    with patch("app.desktop.studio_server.tool_api.task_from_id") as mock_task_from_id:
        # Mock task_from_id to return our test task
        mock_task_from_id.return_value = task

        # Should raise HTTPException for run config not found
        response = client.patch(
            f"/api/projects/{test_project.id}/edit_kiln_task_tool/{existing_tool_server.id}",
            json=tool_data,
        )

        assert response.status_code == 400
        assert (
            "Run config not found for the specified task" in response.json()["message"]
        )


@pytest.mark.asyncio
async def test_get_tool_definition_success(client, test_project):
    """Test get_tool_definition returns correct tool definition for valid tool ID"""

    # Create a test task
    task = Task(
        name="Test Task",
        description="Test task for tool definition",
        instruction="Complete the test task",
        parent=test_project,
    )
    task.save_to_file()

    # Create a run config for the task
    run_config = TaskRunConfig(
        name="default",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name="openai",
            prompt_id=PromptGenerators.SIMPLE,
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=task,
    )
    run_config.save_to_file()

    # Create a kiln task tool server
    tool_server = ExternalToolServer(
        name="test_tool",
        type=ToolServerType.kiln_task,
        description="Test tool for definition",
        properties={
            "name": "test_tool",
            "description": "Test tool for definition",
            "task_id": str(task.id),
            "run_config_id": str(run_config.id),
            "is_archived": False,
        },
        parent=test_project,
    )
    tool_server.save_to_file()

    tool_id = f"{KILN_TASK_TOOL_ID_PREFIX}{tool_server.id}"

    with (
        patch("app.desktop.studio_server.tool_api.task_from_id") as mock_task_from_id,
        patch("app.desktop.studio_server.tool_api.tool_from_id") as mock_tool_from_id,
    ):
        mock_task_from_id.return_value = task

        # Mock the tool and its toolcall_definition method
        mock_tool = AsyncMock()
        mock_tool.toolcall_definition.return_value = {
            "type": "function",
            "function": {
                "name": "test_tool_function",
                "description": "Test tool function description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Input parameter"}
                    },
                    "required": ["input"],
                },
            },
        }
        mock_tool_from_id.return_value = mock_tool

        response = client.get(
            f"/api/projects/{test_project.id}/tasks/{task.id}/tools/{tool_id}/definition"
        )

        assert response.status_code == 200
        result = response.json()

        assert result["tool_id"] == tool_id
        assert result["function_name"] == "test_tool_function"
        assert result["description"] == "Test tool function description"
        assert "parameters" in result

        # Verify the tool was instantiated correctly
        mock_tool_from_id.assert_called_once_with(tool_id, task)
        mock_tool.toolcall_definition.assert_called_once()


@pytest.mark.asyncio
async def test_get_tool_definition_tool_not_found(client, test_project):
    """Test get_tool_definition returns 404 for invalid tool ID"""

    # Create a test task
    task = Task(
        name="Test Task",
        description="Test task for tool definition",
        instruction="Complete the test task",
        parent=test_project,
    )
    task.save_to_file()

    invalid_tool_id = "invalid_tool_id"

    with (
        patch("app.desktop.studio_server.tool_api.task_from_id") as mock_task_from_id,
        patch("app.desktop.studio_server.tool_api.tool_from_id") as mock_tool_from_id,
    ):
        mock_task_from_id.return_value = task
        mock_tool_from_id.side_effect = Exception("Tool not found")

        response = client.get(
            f"/api/projects/{test_project.id}/tasks/{task.id}/tools/{invalid_tool_id}/definition"
        )

        assert response.status_code == 404
        result = response.json()
        assert "Tool not found or could not be instantiated" in result["message"]
        assert invalid_tool_id in result["message"]


@pytest.mark.asyncio
async def test_get_tool_definition_task_not_found(client, test_project):
    """Test get_tool_definition returns 404 for invalid task ID"""

    invalid_task_id = "invalid_task_id"
    tool_id = "some_tool_id"

    with patch("app.desktop.studio_server.tool_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.side_effect = HTTPException(
            status_code=404, detail="Task not found"
        )

        response = client.get(
            f"/api/projects/{test_project.id}/tasks/{invalid_task_id}/tools/{tool_id}/definition"
        )

        assert response.status_code == 404
        assert "Task not found" in response.json()["message"]
