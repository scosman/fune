from http import HTTPStatus
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from kiln_ai.datamodel import Project, Task
from kiln_ai.synthetic_user.runner import NUM_CASES_MAX
from kiln_server.custom_errors import connect_custom_errors

from app.desktop.studio_server.api_client.kiln_ai_server_client.models import (
    BatchPlanOutput as BatchPlanOutputClient,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.types import (
    Response,
)
from app.desktop.studio_server.batch_plan_api import connect_batch_plan_api


def _ok_response() -> Response:
    """A 2xx upstream response. The proxy checks `.status_code` to catch an
    upstream 404, so these mocks need a real Response, not a sentinel string."""
    return Response(status_code=HTTPStatus.OK, content=b"{}", headers={}, parsed=None)


@pytest.fixture
def app():
    app = FastAPI()
    connect_custom_errors(app)
    connect_batch_plan_api(app)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def test_project(tmp_path) -> Project:
    project_path = tmp_path / "test_project" / "project.kiln"
    project_path.parent.mkdir()
    project = Project(name="Test Project", path=project_path)
    project.save_to_file()
    return project


@pytest.fixture
def test_task(test_project) -> Task:
    task = Task(
        name="Test Task",
        instruction="Test Instruction",
        parent=test_project,
        output_json_schema='{"type":"object","properties":{},"required":[]}',
    )
    task.save_to_file()
    return task


@pytest.fixture
def mock_copilot_key():
    with patch("app.desktop.studio_server.batch_plan_api.get_copilot_api_key") as mock:
        mock.return_value = "copilot-key"
        yield mock


@pytest.fixture
def mock_task_from_id(test_task):
    with patch("app.desktop.studio_server.batch_plan_api.task_from_id") as mock:
        mock.return_value = test_task
        yield mock


def test_batch_plan_requires_copilot_key(mock_task_from_id, client):
    with patch(
        "app.desktop.studio_server.batch_plan_api.get_copilot_api_key"
    ) as mock_key:
        mock_key.side_effect = HTTPException(status_code=401, detail="no key")
        resp = client.post(
            "/api/projects/proj-ID/tasks/task-ID/copilot/batch_plan",
            json={"guidance": "x", "count": 2},
        )
    assert resp.status_code == 401


def test_batch_plan_kiln_server_proxy(mock_copilot_key, mock_task_from_id, client):
    """The route proxies to kiln_server and maps the generated BatchPlanOutput
    back to the API output."""
    proxied = BatchPlanOutputClient(prompts=["x", "y"], summary="proxied")
    with (
        patch("app.desktop.studio_server.batch_plan_api.get_authenticated_client"),
        patch(
            "app.desktop.studio_server.batch_plan_api.batch_plan_v1_copilot_batch_plan_post.asyncio_detailed",
            new=AsyncMock(return_value=_ok_response()),
        ),
        patch(
            "app.desktop.studio_server.batch_plan_api.unwrap_response",
            return_value=proxied,
        ),
    ):
        resp = client.post(
            "/api/projects/proj-ID/tasks/task-ID/copilot/batch_plan",
            json={"guidance": "g", "count": 2},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["prompts"] == ["x", "y"]
    assert body["summary"] == "proxied"


def test_batch_plan_sends_task_context_from_server(
    mock_copilot_key, mock_task_from_id, client, test_task
):
    """Task prompt and schemas are derived server-side, not sent by the client."""
    proxied = BatchPlanOutputClient(prompts=["x"], summary="s")
    post = AsyncMock(return_value=_ok_response())
    with (
        patch("app.desktop.studio_server.batch_plan_api.get_authenticated_client"),
        patch(
            "app.desktop.studio_server.batch_plan_api.batch_plan_v1_copilot_batch_plan_post.asyncio_detailed",
            new=post,
        ),
        patch(
            "app.desktop.studio_server.batch_plan_api.unwrap_response",
            return_value=proxied,
        ),
    ):
        resp = client.post(
            "/api/projects/proj-ID/tasks/task-ID/copilot/batch_plan",
            json={"guidance": "g", "count": 1, "data_guide": "guide text"},
        )
    assert resp.status_code == 200, resp.text
    sent = post.call_args.kwargs["body"]
    assert sent.task_prompt == "Test Instruction"
    assert sent.count == 1
    assert sent.user_guidance == "g"
    assert sent.input_data_guide == "guide text"
    assert sent.task_input_schema is None
    assert sent.task_output_schema == test_task.output_json_schema


def test_batch_plan_accepts_count_at_the_cap(
    mock_copilot_key, mock_task_from_id, client
):
    """The cap is inclusive — a full-size batch plan is a valid request. Pairs
    with the over-cap test to pin both sides of the bound."""
    proxied = BatchPlanOutputClient(prompts=["x"], summary="s")
    post = AsyncMock(return_value=_ok_response())
    with (
        patch("app.desktop.studio_server.batch_plan_api.get_authenticated_client"),
        patch(
            "app.desktop.studio_server.batch_plan_api.batch_plan_v1_copilot_batch_plan_post.asyncio_detailed",
            new=post,
        ),
        patch(
            "app.desktop.studio_server.batch_plan_api.unwrap_response",
            return_value=proxied,
        ),
    ):
        resp = client.post(
            "/api/projects/proj-ID/tasks/task-ID/copilot/batch_plan",
            json={"guidance": "g", "count": NUM_CASES_MAX},
        )
    assert resp.status_code == 200, resp.text
    assert post.call_args.kwargs["body"].count == NUM_CASES_MAX


def test_batch_plan_rejects_count_over_the_cap(mock_task_from_id, client):
    """Over-cap counts are rejected by validation, before any upstream call."""
    resp = client.post(
        "/api/projects/proj-ID/tasks/task-ID/copilot/batch_plan",
        json={"guidance": "g", "count": NUM_CASES_MAX + 1},
    )
    assert resp.status_code == 422


def test_batch_plan_unknown_response_is_500(
    mock_copilot_key, mock_task_from_id, client
):
    with (
        patch("app.desktop.studio_server.batch_plan_api.get_authenticated_client"),
        patch(
            "app.desktop.studio_server.batch_plan_api.batch_plan_v1_copilot_batch_plan_post.asyncio_detailed",
            new=AsyncMock(return_value=_ok_response()),
        ),
        patch(
            "app.desktop.studio_server.batch_plan_api.unwrap_response",
            return_value="not a BatchPlanOutput",
        ),
    ):
        resp = client.post(
            "/api/projects/proj-ID/tasks/task-ID/copilot/batch_plan",
            json={"guidance": "g", "count": 2},
        )
    assert resp.status_code == 500


def test_batch_plan_kiln_server_unreachable_returns_502(
    mock_copilot_key, mock_task_from_id, client
):
    """A network failure reaching kiln_server surfaces as a clear 502, not a
    generic 500."""
    with (
        patch("app.desktop.studio_server.batch_plan_api.get_authenticated_client"),
        patch(
            "app.desktop.studio_server.batch_plan_api.batch_plan_v1_copilot_batch_plan_post.asyncio_detailed",
            new=AsyncMock(side_effect=httpx.ConnectError("boom")),
        ),
    ):
        resp = client.post(
            "/api/projects/proj-ID/tasks/task-ID/copilot/batch_plan",
            json={"guidance": "g", "count": 2},
        )
    assert resp.status_code == 502, resp.text
    assert "batch planning" in resp.json()["message"].lower()


def test_batch_plan_kiln_server_404_becomes_502(
    mock_copilot_key, mock_task_from_id, client
):
    """A kiln_server that doesn't serve batch planning (e.g. staging) 404s. That
    must NOT surface as our own 404 — a 404 on this route would read as "this
    studio endpoint doesn't exist". It's an upstream failure, so report 502."""
    not_found = Response(
        status_code=HTTPStatus.NOT_FOUND,
        content=b'{"detail":"Not Found"}',
        headers={},
        parsed=None,
    )
    with (
        patch("app.desktop.studio_server.batch_plan_api.get_authenticated_client"),
        patch(
            "app.desktop.studio_server.batch_plan_api.batch_plan_v1_copilot_batch_plan_post.asyncio_detailed",
            new=AsyncMock(return_value=not_found),
        ),
    ):
        resp = client.post(
            "/api/projects/proj-ID/tasks/task-ID/copilot/batch_plan",
            json={"guidance": "g", "count": 2},
        )
    assert resp.status_code == 502, resp.text
    assert "batch planning" in resp.json()["message"].lower()
