import csv
import io
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from kiln_ai.adapters.ml_model_list import ModelProviderName
from kiln_ai.adapters.model_adapters.litellm_adapter import LiteLlmAdapter
from kiln_ai.datamodel import (
    DataSource,
    DataSourceType,
    Project,
    Task,
    TaskOutput,
    TaskOutputRating,
    TaskOutputRatingType,
    TaskRun,
)
from kiln_ai.datamodel.datamodel_enums import TurnMode
from kiln_ai.datamodel.model_cache import ModelCache
from kiln_ai.datamodel.task_run import EvalItemSource
from kiln_ai.datamodel.tool_id import KilnBuiltInToolId
from kiln_ai.utils.config import Config

from kiln_server.custom_errors import connect_custom_errors
from kiln_server.run_api import (
    RunSummary,
    connect_run_api,
    deep_update,
    model_provider_from_string,
    parse_splits,
    run_from_id,
)


@pytest.fixture(autouse=True)
def _clear_model_cache_between_tests():
    """ModelCache is a process-global singleton. Several tests mutate run
    files on disk after creation; without clearing the cache between tests,
    a stale cached entry from a prior test could mask filesystem changes and
    create order-dependence. Clearing pre-test is cheap and removes the
    risk."""
    ModelCache.shared().clear()
    yield


@pytest.fixture
def app():
    app = FastAPI()
    connect_run_api(app)
    connect_custom_errors(app)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def mock_config():
    with patch("kiln_ai.utils.config.Config.shared") as MockConfig:
        # Mock the Config class
        mock_config_instance = MockConfig.return_value
        mock_config_instance.open_ai_api_key = "test_key"
        yield mock_config_instance


@pytest.fixture
def task_run_setup(tmp_path):
    project_path = tmp_path / "test_project" / "project.kiln"
    project_path.parent.mkdir()

    project = Project(name="Test Project", path=str(project_path))
    project.save_to_file()

    task = Task(
        name="Test Task",
        instruction="This is a test instruction",
        description="This is a test task",
        parent=project,
    )
    task.save_to_file()

    run_task_request = {
        "run_config_properties": {
            "model_name": "gpt_4o",
            "model_provider_name": "ollama",
            "prompt_id": "simple_prompt_builder",
            "structured_output_mode": "json_schema",
        },
        "plaintext_input": "Test input",
    }

    task_run = TaskRun(
        parent=task,
        input="Test input",
        input_source=DataSource(
            type=DataSourceType.human, properties={"created_by": "Test User"}
        ),
        output=TaskOutput(
            output="Test output",
            source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "gpt_4o",
                    "model_provider": "ollama",
                    "adapter_name": "kiln_langchain_adapter",
                    "prompt_id": "simple_prompt_builder",
                },
            ),
        ),
        trace=[
            {"role": "user", "content": "Test input"},
            {"role": "assistant", "content": "Test output"},
        ],
    )
    task_run.save_to_file()

    return {
        "project": project,
        "task": task,
        "run_task_request": run_task_request,
        "task_run": task_run,
    }


@pytest.fixture
def task_run_setup_multiturn(tmp_path):
    """Same shape as `task_run_setup` but the task is multi-turn so chains
    via parent_task_run_id are valid. Used by the leaf-only filter tests."""
    project_path = tmp_path / "test_project" / "project.kiln"
    project_path.parent.mkdir()

    project = Project(name="Test Project", path=str(project_path))
    project.save_to_file()

    task = Task(
        name="Test Task",
        instruction="This is a test instruction",
        description="This is a test task",
        parent=project,
        turn_mode=TurnMode.multiturn,
    )
    task.save_to_file()

    task_run = TaskRun(
        parent=task,
        input="Test input",
        input_source=DataSource(
            type=DataSourceType.human, properties={"created_by": "Test User"}
        ),
        output=TaskOutput(
            output="Test output",
            source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "gpt_4o",
                    "model_provider": "ollama",
                    "adapter_name": "kiln_langchain_adapter",
                    "prompt_id": "simple_prompt_builder",
                },
            ),
        ),
        trace=[
            {"role": "user", "content": "Test input"},
            {"role": "assistant", "content": "Test output"},
        ],
    )
    task_run.save_to_file()

    return {"project": project, "task": task, "task_run": task_run}


@pytest.mark.asyncio
async def test_run_task_success(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    run_task_request = task_run_setup["run_task_request"]

    with (
        patch("kiln_server.run_api.task_from_id") as mock_task_from_id,
        patch.object(LiteLlmAdapter, "invoke", new_callable=AsyncMock) as mock_invoke,
        patch("kiln_ai.utils.config.Config.shared") as MockConfig,
    ):
        mock_task_from_id.return_value = task
        mock_invoke.return_value = task_run_setup["task_run"]

        # Mock the Config class
        mock_config_instance = MockConfig.return_value
        mock_config_instance.ollama_base_url = "http://localhost:11434/v1"

        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/run", json=run_task_request
        )

    assert response.status_code == 200
    res = response.json()
    assert res["output"]["output"] == "Test output"
    # Checking that the ID is not None because it's saved to the disk
    assert res["id"] is not None


@pytest.mark.asyncio
async def test_run_task_structured_output(client, task_run_setup):
    task = task_run_setup["task"]
    run_task_request = task_run_setup["run_task_request"]

    with (
        patch("kiln_server.run_api.task_from_id") as mock_task_from_id,
        patch.object(LiteLlmAdapter, "invoke", new_callable=AsyncMock) as mock_invoke,
        patch("kiln_ai.utils.config.Config.shared") as MockConfig,
    ):
        mock_task_from_id.return_value = task
        task_run = task_run_setup["task_run"]
        task_run.output.output = '{"key": "value"}'
        mock_invoke.return_value = task_run

        # Mock the Config class
        mock_config_instance = MockConfig.return_value
        mock_config_instance.ollama_base_url = "http://localhost:11434/v1"

        response = client.post(
            f"/api/projects/project1-id/tasks/{task.id}/run", json=run_task_request
        )

    res = response.json()
    assert response.status_code == 200
    assert res["output"]["output"] == '{"key": "value"}'
    # ID set because it's saved to the disk
    assert res["id"] is not None


@pytest.mark.asyncio
async def test_run_task_no_input(client, task_run_setup, mock_config):
    task = task_run_setup["task"]

    # Missing input
    run_task_request = {
        "run_config_properties": {
            "model_name": "gpt_4o",
            "model_provider_name": "openai",
            "prompt_id": "simple_prompt_builder",
            "structured_output_mode": "json_schema",
        }
    }

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/project1-id/tasks/{task.id}/run", json=run_task_request
        )

    assert response.status_code == 400
    assert "No input provided" in response.json()["message"]


@pytest.mark.asyncio
async def test_run_task_structured_input(client, task_run_setup):
    task = task_run_setup["task"]

    with patch.object(
        Task,
        "input_schema",
        return_value={
            "type": "object",
            "properties": {"key": {"type": "string"}},
        },
    ):
        run_task_request = {
            "run_config_properties": {
                "model_name": "gpt_4o",
                "model_provider_name": "ollama",
                "prompt_id": "simple_prompt_builder",
                "structured_output_mode": "json_schema",
            },
            "structured_input": {"key": "value"},
        }

        with (
            patch("kiln_server.run_api.task_from_id") as mock_task_from_id,
            patch.object(
                LiteLlmAdapter, "invoke", new_callable=AsyncMock
            ) as mock_invoke,
            patch("kiln_ai.utils.config.Config.shared") as MockConfig,
        ):
            mock_task_from_id.return_value = task
            mock_invoke.return_value = task_run_setup["task_run"]

            # Mock the Config class
            mock_config_instance = MockConfig.return_value
            mock_config_instance.ollama_base_url = "http://localhost:11434/v1"

            response = client.post(
                f"/api/projects/project1-id/tasks/{task.id}/run", json=run_task_request
            )

    assert response.status_code == 200
    res = response.json()
    assert res["output"]["output"] == "Test output"
    assert res["id"] is not None


@pytest.mark.asyncio
async def test_run_task_forwards_task_run_config_id(client, task_run_setup):
    task = task_run_setup["task"]
    run_task_request = {
        **task_run_setup["run_task_request"],
        "task_run_config_id": "rc_abc123",
    }

    captured: dict = {}

    def fake_adapter_for_task(task_arg, run_config_properties, base_adapter_config):
        captured["task_run_config_id"] = base_adapter_config.task_run_config_id
        mock_adapter = MagicMock()
        mock_adapter.invoke = AsyncMock(return_value=task_run_setup["task_run"])
        return mock_adapter

    with (
        patch("kiln_server.run_api.task_from_id") as mock_task_from_id,
        patch(
            "kiln_server.run_api.adapter_for_task", side_effect=fake_adapter_for_task
        ),
        patch("kiln_ai.utils.config.Config.shared") as MockConfig,
    ):
        mock_task_from_id.return_value = task
        mock_config_instance = MockConfig.return_value
        mock_config_instance.ollama_base_url = "http://localhost:11434/v1"

        response = client.post(
            f"/api/projects/project-1/tasks/{task.id}/run", json=run_task_request
        )

    assert response.status_code == 200
    assert captured["task_run_config_id"] == "rc_abc123"


@pytest.mark.asyncio
async def test_run_task_task_run_config_id_defaults_to_none(client, task_run_setup):
    task = task_run_setup["task"]
    run_task_request = task_run_setup["run_task_request"]

    captured: dict = {}

    def fake_adapter_for_task(task_arg, run_config_properties, base_adapter_config):
        captured["task_run_config_id"] = base_adapter_config.task_run_config_id
        mock_adapter = MagicMock()
        mock_adapter.invoke = AsyncMock(return_value=task_run_setup["task_run"])
        return mock_adapter

    with (
        patch("kiln_server.run_api.task_from_id") as mock_task_from_id,
        patch(
            "kiln_server.run_api.adapter_for_task", side_effect=fake_adapter_for_task
        ),
        patch("kiln_ai.utils.config.Config.shared") as MockConfig,
    ):
        mock_task_from_id.return_value = task
        mock_config_instance = MockConfig.return_value
        mock_config_instance.ollama_base_url = "http://localhost:11434/v1"

        response = client.post(
            f"/api/projects/project-1/tasks/{task.id}/run", json=run_task_request
        )

    assert response.status_code == 200
    assert captured["task_run_config_id"] is None


def test_deep_update_with_empty_source():
    source = {}
    update = {"a": 1, "b": {"c": 2}}
    result = deep_update(source, update)
    assert result == {"a": 1, "b": {"c": 2}}


def test_deep_update_with_existing_keys():
    source = {"a": 0, "b": {"c": 1}}
    update = {"a": 1, "b": {"d": 2}}
    result = deep_update(source, update)
    assert result == {"a": 1, "b": {"c": 1, "d": 2}}


def test_deep_update_with_nested_dicts():
    source = {"a": {"b": {"c": 1}}}
    update = {"a": {"b": {"d": 2}, "e": 3}}
    result = deep_update(source, update)
    assert result == {"a": {"b": {"c": 1, "d": 2}, "e": 3}}


def test_deep_update_with_non_dict_values():
    source = {"a": 1, "b": [1, 2, 3]}
    update = {"a": 2, "b": [4, 5, 6], "c": "new"}
    result = deep_update(source, update)
    assert result == {"a": 2, "b": [4, 5, 6], "c": "new"}


def test_deep_update_with_mixed_types():
    source = {"a": 1, "b": {"c": [1, 2, 3]}}
    update = {"a": "new", "b": {"c": 4, "d": {"e": 5}}}
    result = deep_update(source, update)
    assert result == {"a": "new", "b": {"c": 4, "d": {"e": 5}}}


def test_deep_update_with_none_values():
    # Test case 1: Basic removal of keys
    source = {"a": 1, "b": 2, "c": 3}
    update = {"a": None, "b": 4}
    result = deep_update(source, update)
    assert result == {"b": 4, "c": 3}

    # Test case 2: Nested dictionaries
    source = {"x": 1, "y": {"y1": 10, "y2": 20, "y3": {"y3a": 100, "y3b": 200}}, "z": 3}
    update = {"y": {"y2": None, "y3": {"y3b": None, "y3c": 300}}, "z": None}
    result = deep_update(source, update)
    assert result == {"x": 1, "y": {"y1": 10, "y3": {"y3a": 100, "y3c": 300}}}

    # Test case 3: Update with empty dictionary
    source = {"a": 1, "b": 2}
    update = {}
    result = deep_update(source, update)
    assert result == {"a": 1, "b": 2}

    # Test case 4: Update missing with none elements
    source = {"a": 1, "b": {"d": 1}}
    update = {"b": {"e": {"f": {"h": 1, "j": None}, "g": None}}}
    result = deep_update(source, update)
    assert result == {"a": 1, "b": {"d": 1, "e": {"f": {"h": 1}}}}

    # Test case 5: Mixed types
    source = {"a": 1, "b": {"x": 10, "y": 20}, "c": [1, 2, 3]}
    update = {"b": {"y": None, "z": 30}, "c": None, "d": 4}
    result = deep_update(source, update)
    assert result == {"a": 1, "b": {"x": 10, "z": 30}, "d": 4}

    # Test case 6: Update with
    source = {}
    update = {"a": {"b": None, "c": None}}
    result = deep_update(source, update)
    assert result == {"a": {}}

    # Test case 7: Update with
    source = {
        "output": {
            "rating": None,
            "model_type": "task_output",
        },
    }
    update = {
        "output": {
            "rating": {
                "value": 2,
                "type": "five_star",
                "requirement_ratings": {
                    "148753630565": None,
                    "988847661375": 3,
                    "474350686960": None,
                },
            }
        }
    }
    result = deep_update(source, update)
    assert result["output"]["rating"]["value"] == 2
    assert result["output"]["rating"]["type"] == "five_star"
    assert result["output"]["rating"]["requirement_ratings"] == {
        # "148753630565": None,
        "988847661375": 3,
        # "474350686960": None,
    }


def test_update_run_method():
    run = TaskRun(
        input="Test input",
        input_source=DataSource(
            type=DataSourceType.human, properties={"created_by": "Jane Doe"}
        ),
        output=TaskOutput(
            output="Test output",
            source=DataSource(
                type=DataSourceType.human, properties={"created_by": "Jane Doe"}
            ),
        ),
    )

    dumped = run.model_dump()
    merged = deep_update(dumped, {"input": "Updated input"})
    updated_run = TaskRun.model_validate(merged)
    assert updated_run.input == "Updated input"

    update = {
        "output": {"rating": {"value": 4, "type": TaskOutputRatingType.five_star}}
    }
    dumped = run.model_dump()
    merged = deep_update(dumped, update)
    updated_run = TaskRun.model_validate(merged)
    assert updated_run.output.rating.value == 4
    assert updated_run.output.rating.type == TaskOutputRatingType.five_star


@pytest.mark.asyncio
async def test_update_run(client, tmp_path):
    project_path = tmp_path / "test_project" / "project.kiln"
    project_path.parent.mkdir()

    project = Project(name="Test Project", path=str(project_path))
    project.save_to_file()
    task = Task(
        name="Test Task",
        instruction="This is a test instruction",
        description="This is a test task",
        parent=project,
    )
    task.save_to_file()
    run = TaskRun(
        parent=task,
        input="Test input",
        input_source=DataSource(
            type=DataSourceType.human, properties={"created_by": "Jane Doe"}
        ),
        output=TaskOutput(
            output="Test output",
            source=DataSource(
                type=DataSourceType.human, properties={"created_by": "Jane Doe"}
            ),
        ),
    )
    run.save_to_file()

    test_cases = [
        {
            "name": "Update output rating",
            "patch": {
                "output": {
                    "rating": {"value": 4, "type": TaskOutputRatingType.five_star},
                }
            },
            "expected": {
                "output": {
                    "rating": {"value": 4, "type": TaskOutputRatingType.five_star},
                }
            },
        },
        {
            "name": "Update input",
            "patch": {
                "input": "Updated input",
            },
            "expected": {
                "input": "Updated input",
            },
        },
    ]

    for case in test_cases:
        with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
            mock_task_from_id.return_value = task

            response = client.patch(
                f"/api/projects/project1-id/tasks/{task.id}/runs/{run.id}",
                json=case["patch"],
            )

            assert response.status_code == 200, f"Failed on case: {case['name']}"

    # Test error cases, including deep validation
    error_cases = [
        {
            "name": "Task not found",
            "task_id": "non_existent_task_id",
            "run_id": run.id,
            "expected_status": 404,
            "expected_detail": "Task not found. ID: non_existent_task_id",
            "updates": {"input": "Updated input"},
        },
        {
            "name": "Run not found",
            "task_id": task.id,
            "run_id": "non_existent_run_id",
            "expected_status": 404,
            "expected_detail": "Run not found. ID: non_existent_run_id",
            "updates": {"input": "Updated input"},
        },
        {
            "name": "Invalid input",
            "task_id": task.id,
            "run_id": run.id,
            "expected_status": 422,
            "expected_detail": "Input: Input should be a valid string",
            "updates": {"input": 123},
        },
        {
            "name": "Invalid rating without value",
            "task_id": task.id,
            "run_id": run.id,
            "expected_status": 422,
            "expected_detail": "Output.Rating.Type: Input should be 'five_star', 'pass_fail', 'pass_fail_critical' or 'custom'",
            "updates": {
                "output": {
                    "rating": {"type": "invalid", "rating": 1},
                }
            },
        },
    ]

    for case in error_cases:
        with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
            mock_project_from_id.return_value = project

            response = client.patch(
                f"/api/projects/project1-id/tasks/{case['task_id']}/runs/{case['run_id']}",
                json=case["updates"],
            )

            assert response.status_code == case["expected_status"], (
                f"Failed on case: {case['name']}"
            )
            assert response.json()["message"] == case["expected_detail"], (
                f"Failed on case: {case['name']}"
            )


@pytest.fixture
def test_run(tmp_path) -> TaskRun:
    project_path = tmp_path / "test_project" / "project.kiln"
    project_path.parent.mkdir()
    project = Project(name="Test Project", path=str(project_path))
    project.save_to_file()
    task = Task(
        name="Test Task",
        instruction="This is a test instruction",
        description="This is a test task",
        parent=project,
    )
    task.save_to_file()
    run = TaskRun(
        parent=task,
        input="Test input",
        input_source=DataSource(
            type=DataSourceType.human, properties={"created_by": "Jane Doe"}
        ),
        output=TaskOutput(
            output="Test output",
            source=DataSource(
                type=DataSourceType.human, properties={"created_by": "Jane Doe"}
            ),
        ),
    )
    run.save_to_file()
    return run


def test_run_from_id_success(test_run):
    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = test_run.parent
        result = run_from_id(test_run.parent.parent.id, test_run.parent.id, test_run.id)
        assert result.id == test_run.id
        assert result.input == "Test input"
        assert result.output.output == "Test output"


def test_run_from_id_not_found(test_run):
    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = test_run.parent
        with pytest.raises(HTTPException) as exc_info:
            run_from_id(
                test_run.parent.parent.id, test_run.parent.id, "non_existent_run_id"
            )
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Run not found. ID: non_existent_run_id"


async def test_get_run_success(client, test_run):
    with patch("kiln_server.run_api.run_from_id") as mock_run_from_id:
        mock_run_from_id.return_value = test_run
        response = client.get(
            f"/api/projects/{test_run.parent.parent.id}/tasks/{test_run.parent.id}/runs/{test_run.id}"
        )

    assert response.status_code == 200
    result = response.json()
    assert result["id"] == test_run.id
    assert result["input"] == "Test input"
    assert result["output"]["output"] == "Test output"


async def test_get_run_not_found(client):
    with patch("kiln_server.run_api.run_from_id") as mock_run_from_id:
        mock_run_from_id.side_effect = HTTPException(
            status_code=404, detail="Run not found"
        )
        response = client.get(
            "/api/projects/project1-id/tasks/task1-id/runs/non_existent_run_id"
        )

    assert response.status_code == 404
    assert response.json()["message"] == "Run not found"


@pytest.mark.asyncio
async def test_get_runs_success(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task = MagicMock()
        mock_task.runs.return_value = [task_run]
        mock_task_from_id.return_value = mock_task

        response = client.get(f"/api/projects/{project.id}/tasks/{task.id}/runs")

    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["id"] == task_run.id
    assert result[0]["input"] == "Test input"
    assert result[0]["output"]["output"] == "Test output"


@pytest.mark.asyncio
async def test_get_runs_empty(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task = MagicMock()
        mock_task.runs.return_value = []
        mock_task_from_id.return_value = mock_task

        response = client.get(f"/api/projects/{project.id}/tasks/{task.id}/runs")

    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_get_runs_task_not_found(client):
    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.side_effect = HTTPException(
            status_code=404, detail="Task not found"
        )
        response = client.get(
            "/api/projects/project1-id/tasks/non_existent_task_id/runs"
        )

    assert response.status_code == 404
    assert response.json()["message"] == "Task not found"


def _make_run_with_date(task, created_at):
    """Create a TaskRun with a specific created_at timestamp."""
    run = TaskRun(
        parent=task,
        input="test",
        input_source=DataSource(
            type=DataSourceType.human,
            properties={"created_by": "test"},
        ),
        output=TaskOutput(
            output="test",
            source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "test",
                    "model_provider": "test",
                    "adapter_name": "test",
                },
            ),
        ),
        created_at=created_at,
    )
    run.save_to_file()
    return run


@pytest.mark.asyncio
async def test_get_runs_ordered_newest_first(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]

    old_run = _make_run_with_date(task, datetime(2024, 1, 1, tzinfo=timezone.utc))
    mid_run = _make_run_with_date(task, datetime(2024, 6, 1, tzinfo=timezone.utc))
    new_run = _make_run_with_date(task, datetime(2024, 12, 1, tzinfo=timezone.utc))

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task = MagicMock()
        mock_task.runs.return_value = [old_run, new_run, mid_run]
        mock_task_from_id.return_value = mock_task

        response = client.get(f"/api/projects/{project.id}/tasks/{task.id}/runs")

    assert response.status_code == 200
    result = response.json()
    ids = [r["id"] for r in result]
    assert ids == [new_run.id, mid_run.id, old_run.id]


@pytest.mark.asyncio
async def test_get_runs_with_limit(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]

    runs = [
        _make_run_with_date(task, datetime(2024, 1, i + 1, tzinfo=timezone.utc))
        for i in range(10)
    ]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task = MagicMock()
        mock_task.runs.return_value = runs
        mock_task_from_id.return_value = mock_task

        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs",
            params={"limit": 3},
        )

    assert response.status_code == 200
    result = response.json()
    assert len(result) == 3
    returned_ids = [r["id"] for r in result]
    assert returned_ids == [runs[9].id, runs[8].id, runs[7].id]


@pytest.mark.asyncio
async def test_get_runs_without_limit_returns_all(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]

    runs = [
        _make_run_with_date(task, datetime(2024, 1, i + 1, tzinfo=timezone.utc))
        for i in range(5)
    ]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task = MagicMock()
        mock_task.runs.return_value = runs
        mock_task_from_id.return_value = mock_task

        response = client.get(f"/api/projects/{project.id}/tasks/{task.id}/runs")

    assert response.status_code == 200
    result = response.json()
    assert len(result) == 5


@pytest.mark.asyncio
async def test_get_runs_limit_validation(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]

    response = client.get(
        f"/api/projects/{project.id}/tasks/{task.id}/runs",
        params={"limit": 0},
    )
    assert response.status_code == 422

    response = client.get(
        f"/api/projects/{project.id}/tasks/{task.id}/runs",
        params={"limit": -1},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_task_run_success(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task

        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs",
            json={
                "input": "Test input for direct creation",
                "output": "Test output for direct creation",
                "tags": ["golden_tag", "test_tag"],
                "rating": {
                    "type": "five_star",
                    "requirement_ratings": {
                        "named::desired_behaviour": {
                            "type": "pass_fail",
                            "value": 1.0,
                        }
                    },
                },
                "model_name": "copilot",
                "model_provider": "kiln",
                "adapter_name": "kiln-ai-adapter",
            },
        )

    assert response.status_code == 200
    result = response.json()
    assert result["input"] == "Test input for direct creation"
    assert result["output"]["output"] == "Test output for direct creation"
    assert result["tags"] == ["golden_tag", "test_tag"]
    assert result["output"]["rating"]["type"] == "five_star"
    assert (
        result["output"]["rating"]["requirement_ratings"]["named::desired_behaviour"][
            "value"
        ]
        == 1.0
    )
    assert result["output"]["source"]["properties"]["adapter_name"] == "kiln-ai-adapter"
    assert result["id"] is not None


@pytest.mark.asyncio
async def test_create_task_run_with_fail_rating(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task

        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs",
            json={
                "input": "Failing test input",
                "output": "Failing test output",
                "tags": ["golden_tag"],
                "rating": {
                    "type": "five_star",
                    "requirement_ratings": {
                        "named::desired_behaviour": {
                            "type": "pass_fail",
                            "value": 0.0,
                        }
                    },
                },
                "model_name": "copilot",
                "model_provider": "kiln",
                "adapter_name": "kiln-ai-adapter",
            },
        )

    assert response.status_code == 200
    result = response.json()
    assert (
        result["output"]["rating"]["requirement_ratings"]["named::desired_behaviour"][
            "value"
        ]
        == 0.0
    )


@pytest.mark.asyncio
async def test_create_task_run_minimal(client, task_run_setup):
    """Test creating a TaskRun with only required fields."""
    project = task_run_setup["project"]
    task = task_run_setup["task"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task

        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs",
            json={
                "input": "Minimal input",
                "output": "Minimal output",
                "model_name": "copilot",
                "model_provider": "kiln",
                "adapter_name": "kiln-ai-adapter",
            },
        )

    assert response.status_code == 200
    result = response.json()
    assert result["input"] == "Minimal input"
    assert result["output"]["output"] == "Minimal output"
    assert result["tags"] == []
    assert result["output"]["source"]["type"] == "synthetic"
    assert result["output"]["source"]["properties"]["model_name"] == "copilot"


@pytest.mark.asyncio
async def test_delete_run(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    # Verify the run file exists before deletion
    path = task_run.path
    assert path.exists()
    assert path.is_file()
    assert path.parent.exists()

    with patch("kiln_server.run_api.task_and_run_from_id") as mock_resolve:
        mock_resolve.return_value = (task, task_run)
        response = client.delete(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{task_run.id}"
        )

    assert response.status_code == 200
    # Verify the file was actually deleted
    assert not path.exists()
    assert not path.parent.exists()
    # Don't delete the task directory
    assert path.parent.parent.exists()


@pytest.mark.asyncio
async def test_delete_run_not_found(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]

    with patch("kiln_server.run_api.task_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = task

        response = client.delete(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/non_existent_run_id"
        )

    assert response.status_code == 404
    assert response.json()["message"] == "Run not found. ID: non_existent_run_id"


@pytest.mark.asyncio
async def test_update_run_clear_repair_fields(client, task_run_setup):
    task = task_run_setup["task"]

    run = TaskRun(
        parent=task,
        input="Test input",
        input_source=DataSource(
            type=DataSourceType.human, properties={"created_by": "Jane Doe"}
        ),
        output=TaskOutput(
            output="Test output",
            source=DataSource(
                type=DataSourceType.human, properties={"created_by": "Jane Doe"}
            ),
        ),
        repair_instructions="Fix this output",
        repaired_output=TaskOutput(
            output="Fixed output",
            source=DataSource(
                type=DataSourceType.human, properties={"created_by": "Jane Doe"}
            ),
        ),
    )

    run.save_to_file()
    assert run.repair_instructions is not None
    assert run.repaired_output is not None

    # Patch to clear repair fields
    patch_data = {"output": {"repair_instruction": None, "repaired_output": None}}

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task

        response = client.patch(
            f"/api/projects/project1-id/tasks/{task.id}/runs/{run.id}",
            json=patch_data,
        )

        assert response.status_code == 200
        result = response.json()

        # Verify repair fields are removed but other fields remain
        assert "repair_instruction" not in result["output"]
        assert "repaired_output" not in result["output"]
        assert result["output"]["output"] == "Test output"
        assert result["input"] == "Test input"


def test_run_summary_format_preview():
    assert RunSummary.format_preview(None) is None
    assert RunSummary.format_preview("short text") == "short text"
    assert RunSummary.format_preview("a" * 101) == "a" * 100 + "…"


def test_run_summary_repair_status_display_name():
    run = MagicMock()
    run.repair_instructions = None
    run.output = MagicMock()
    run.output.rating = None
    assert RunSummary.repair_status_display_name(run) == "NA"

    run.output.rating = TaskOutputRating(value=5.0, type=TaskOutputRatingType.five_star)
    assert RunSummary.repair_status_display_name(run) == "No repair needed"

    run.output.rating = TaskOutputRating(value=3.0, type=TaskOutputRatingType.custom)
    assert RunSummary.repair_status_display_name(run) == "Unknown"

    run.output.rating = TaskOutputRating(value=3.0, type=TaskOutputRatingType.five_star)
    run.output.output = "Some output"
    assert RunSummary.repair_status_display_name(run) == "Repair needed"

    run.output = None
    assert RunSummary.repair_status_display_name(run) == "No output"


def test_run_summary_from_run(task_run_setup):
    run = task_run_setup["task_run"]
    summary = RunSummary.from_run(run)
    assert summary.id == run.id
    assert summary.input_preview == RunSummary.format_preview(run.input)
    assert summary.output_preview == RunSummary.format_preview(run.output.output)
    assert summary.repair_state == RunSummary.repair_status_display_name(run)
    assert summary.model_name == run.output.source.properties.get("model_name")
    assert summary.input_source == run.input_source.type


@pytest.mark.asyncio
async def test_get_runs_summaries_success(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task = MagicMock()
        mock_task.runs.return_value = [task_run]
        mock_task_from_id.return_value = mock_task

        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs_summaries"
        )

    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["id"] == task_run.id
    assert result[0]["input_preview"] == RunSummary.format_preview(task_run.input)
    assert result[0]["output_preview"] == RunSummary.format_preview(
        task_run.output.output
    )
    assert result[0]["repair_state"] == RunSummary.repair_status_display_name(task_run)
    assert result[0]["model_name"] == task_run.output.source.properties.get(
        "model_name"
    )
    assert result[0]["input_source"] == task_run.input_source.type


@pytest.mark.asyncio
async def test_get_runs_summaries_returns_only_leaf_runs(
    client, task_run_setup_multiturn
):
    """The endpoint must hide intermediate (non-leaf) runs in multiturn chains.

    Setup: starting from the fixture's lone task_run (no parent), build a
    linear chain task_run -> mid -> leaf and a sibling branch task_run -> sib.
    Expected: only `leaf` and `sib` come back; task_run and mid are filtered
    out because they are parents of other runs.
    """
    project = task_run_setup_multiturn["project"]
    task = task_run_setup_multiturn["task"]
    root = task_run_setup_multiturn["task_run"]

    def _make_child(parent_run: TaskRun) -> TaskRun:
        run = TaskRun(
            parent=task,
            parent_task_run_id=parent_run.id,
            input="continuation",
            input_source=DataSource(
                type=DataSourceType.human, properties={"created_by": "Test User"}
            ),
            output=TaskOutput(
                output="continuation output",
                source=DataSource(
                    type=DataSourceType.synthetic,
                    properties={
                        "model_name": "gpt_4o",
                        "model_provider": "ollama",
                        "adapter_name": "kiln_langchain_adapter",
                        "prompt_id": "simple_prompt_builder",
                    },
                ),
            ),
        )
        run.save_to_file()
        return run

    mid = _make_child(root)
    leaf = _make_child(mid)
    sib = _make_child(root)

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs_summaries"
        )

    assert response.status_code == 200
    returned_ids = {entry["id"] for entry in response.json()}
    assert returned_ids == {leaf.id, sib.id}
    assert root.id not in returned_ids, "root must be filtered (has children)"
    assert mid.id not in returned_ids, "mid must be filtered (has children)"


@pytest.mark.asyncio
async def test_get_runs_summaries_task_not_found(client):
    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.side_effect = HTTPException(
            status_code=404, detail="Task not found"
        )
        response = client.get(
            "/api/projects/project1-id/tasks/non_existent_task_id/runs_summaries"
        )

    assert response.status_code == 404
    assert response.json()["message"] == "Task not found"


def _make_child_run(task, parent_run: TaskRun, **overrides) -> TaskRun:
    run = TaskRun(
        parent=task,
        parent_task_run_id=parent_run.id,
        input=overrides.get("input", "continuation"),
        input_source=DataSource(
            type=DataSourceType.human, properties={"created_by": "Test User"}
        ),
        output=TaskOutput(
            output=overrides.get("output", "continuation output"),
            source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "gpt_4o",
                    "model_provider": "ollama",
                    "adapter_name": "kiln_langchain_adapter",
                    "prompt_id": "simple_prompt_builder",
                },
            ),
        ),
        tags=overrides.get("tags", []),
    )
    run.save_to_file()
    return run


@pytest.mark.asyncio
async def test_get_runs_returns_only_leaf_runs(client, task_run_setup_multiturn):
    """GET /runs must hide intermediate runs for multiturn chains.

    Mirrors test_get_runs_summaries_returns_only_leaf_runs against the
    full-TaskRun endpoint (not the summaries view).
    """
    project = task_run_setup_multiturn["project"]
    task = task_run_setup_multiturn["task"]
    root = task_run_setup_multiturn["task_run"]

    mid = _make_child_run(task, root)
    leaf = _make_child_run(task, mid)
    sib = _make_child_run(task, root)

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(f"/api/projects/{project.id}/tasks/{task.id}/runs")

    assert response.status_code == 200
    returned_ids = {entry["id"] for entry in response.json()}
    assert returned_ids == {leaf.id, sib.id}
    assert root.id not in returned_ids, "root must be filtered (has children)"
    assert mid.id not in returned_ids, "mid must be filtered (has children)"


@pytest.mark.asyncio
async def test_get_tags_counts_only_leaf_runs(client, task_run_setup_multiturn):
    """GET /tags must not count tags attached to intermediate (non-leaf) runs.

    Setup: root has tag "intermediate" (will be hidden); mid has tag
    "intermediate" (also hidden); leaf has tags "shared" + "leaf_only"; sib
    has tag "shared".
    Expected: only "shared" (count 2) and "leaf_only" (count 1) are returned;
    "intermediate" is absent.
    """
    project = task_run_setup_multiturn["project"]
    task = task_run_setup_multiturn["task"]
    root = task_run_setup_multiturn["task_run"]

    # Tag the root in-place and re-save so the on-disk file reflects the tag.
    root.tags = ["intermediate"]
    root.save_to_file()

    mid = _make_child_run(task, root, tags=["intermediate"])
    _leaf = _make_child_run(task, mid, tags=["shared", "leaf_only"])
    _sib = _make_child_run(task, root, tags=["shared"])

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(f"/api/projects/{project.id}/tasks/{task.id}/tags")

    assert response.status_code == 200
    counts = response.json()
    assert counts == {"shared": 2, "leaf_only": 1}
    assert "intermediate" not in counts, (
        "tags on intermediate (filtered) runs must not appear in the counts"
    )


@pytest.mark.asyncio
async def test_delete_multiple_runs_success(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    # Create a second run
    second_run = TaskRun(
        parent=task,
        input="Test input 2",
        input_source=DataSource(
            type=DataSourceType.human, properties={"created_by": "Test User"}
        ),
        output=TaskOutput(
            output="Test output 2",
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "Test User"},
            ),
        ),
    )
    second_run.save_to_file()

    run_ids = [task_run.id, second_run.id]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/delete", json=run_ids
        )

    assert response.status_code == 200
    assert response.json() == {"success": True}
    # Verify files were deleted
    assert not task_run.path.exists()
    assert not second_run.path.exists()


@pytest.mark.asyncio
async def test_delete_multiple_runs_partial_failure(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    # Include one valid and one invalid run ID
    run_ids = [task_run.id, "non_existent_run_id"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/delete", json=run_ids
        )

    assert response.status_code == 500
    result = response.json()
    assert "failed_runs" in result["message"]
    assert "non_existent_run_id" in result["message"]["failed_runs"]
    assert "Run not found" in result["message"]["error"]


@pytest.mark.asyncio
async def test_delete_multiple_runs_task_not_found(client):
    run_ids = ["run1", "run2"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.side_effect = HTTPException(
            status_code=404, detail="Task not found"
        )
        response = client.post(
            "/api/projects/project1-id/tasks/non_existent_task_id/runs/delete",
            json=run_ids,
        )

    assert response.status_code == 404
    assert response.json()["message"] == "Task not found"


@pytest.mark.asyncio
async def test_delete_multiple_runs_with_exception(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    run_ids = [task_run.id]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        # Simulate an unexpected error during deletion
        with patch.object(TaskRun, "delete") as mock_delete:
            mock_delete.side_effect = Exception("Unexpected error")
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/runs/delete", json=run_ids
            )

    assert response.status_code == 500
    result = response.json()
    assert "failed_runs" in result["message"]
    assert task_run.id in result["message"]["failed_runs"]
    assert "Unexpected error" in result["message"]["error"]


def project_id_of(task: Task) -> str:
    parent = task.parent_project()
    assert parent is not None and parent.id is not None
    return parent.id


class TestEvalTraceProtection:
    """A run generated by an eval is the record an eval's scores were computed over.

    Deleting it would leave the score pointing at nothing, so it is refused - and the
    flag that marks it is refused as a client-settable field, since setting it on an
    ordinary run would make that run undeletable through this same guard.
    """

    def _eval_trace(self, task: Task) -> TaskRun:
        run = TaskRun(
            parent=task,
            input="Eval input",
            input_source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "gpt_4o",
                    "model_provider": "ollama",
                    "adapter_name": "kiln_langchain_adapter",
                },
            ),
            output=TaskOutput(output="Eval output"),
            eval_source=EvalItemSource(
                source_type="eval_input", source_id="500000000001"
            ),
        )
        run.save_to_file()
        return run

    def test_deleting_an_eval_generated_run_is_refused(self, client, task_run_setup):
        project = task_run_setup["project"]
        task = task_run_setup["task"]
        trace = self._eval_trace(task)

        with patch("kiln_server.run_api.task_and_run_from_id") as mock_resolve:
            mock_resolve.return_value = (task, trace)
            response = client.delete(
                f"/api/projects/{project.id}/tasks/{task.id}/runs/{trace.id}"
            )

        assert response.status_code == 409
        assert "needed for an eval" in response.json()["message"]
        assert trace.path.exists()

    def test_deleting_an_ordinary_run_still_works(self, client, task_run_setup):
        project = task_run_setup["project"]
        task = task_run_setup["task"]
        task_run = task_run_setup["task_run"]
        path = task_run.path

        with patch("kiln_server.run_api.task_and_run_from_id") as mock_resolve:
            mock_resolve.return_value = (task, task_run)
            response = client.delete(
                f"/api/projects/{project.id}/tasks/{task.id}/runs/{task_run.id}"
            )

        assert response.status_code == 200
        assert not path.exists()

    def test_bulk_delete_reports_eval_runs_and_deletes_the_rest(
        self, client, task_run_setup
    ):
        """A partially deleted selection has to say why, or the missing deletes read as
        a bug."""
        project = task_run_setup["project"]
        task = task_run_setup["task"]
        task_run = task_run_setup["task_run"]
        trace = self._eval_trace(task)

        with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
            mock_task_from_id.return_value = task
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/runs/delete",
                json=[task_run.id, trace.id],
            )

        assert response.status_code == 500
        result = response.json()["message"]
        assert result["failed_runs"] == [trace.id]
        assert "needed for an eval" in result["error"]
        assert not task_run.path.exists()
        assert trace.path.exists()

    def test_bulk_delete_reports_every_distinct_failure_reason(
        self, client, task_run_setup
    ):
        """Two protected runs and a missing one must not collapse to whichever the loop
        saw last - "Run not found" about a run that is on disk is actively misleading."""
        project = task_run_setup["project"]
        task = task_run_setup["task"]
        trace = self._eval_trace(task)

        with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
            mock_task_from_id.return_value = task
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/runs/delete",
                json=[trace.id, "non_existent_run_id"],
            )

        assert response.status_code == 500
        error = response.json()["message"]["error"]
        assert "needed for an eval" in error
        assert "Run not found" in error

    def test_patching_eval_source_onto_a_run_is_refused(self, client, task_run_setup):
        """No task_from_id patch: the guard has to refuse before the run is even loaded,
        which is also what makes it uniform across every future caller."""
        task = task_run_setup["task"]
        task_run = task_run_setup["task_run"]

        response = client.patch(
            f"/api/projects/{project_id_of(task)}/tasks/{task.id}/runs/{task_run.id}",
            json={
                "eval_source": {
                    "source_type": "eval_input",
                    "source_id": "500000000001",
                }
            },
        )

        assert response.status_code == 400
        assert "eval_source cannot be set by client" in response.json()["message"]
        assert TaskRun.load_from_file(task_run.path).eval_source is None

    def test_patching_eval_source_to_null_is_refused_too(self, client, task_run_setup):
        """Clearing is the half that matters: deep_update treats null as a delete, so
        allowing it would be a way out of the delete guard."""
        task = task_run_setup["task"]
        trace = self._eval_trace(task)

        response = client.patch(
            f"/api/projects/{project_id_of(task)}/tasks/{task.id}/runs/{trace.id}",
            json={"eval_source": None},
        )

        assert response.status_code == 400
        assert TaskRun.load_from_file(trace.path).eval_source is not None

    def test_patching_an_unrelated_field_on_an_eval_trace_still_works(
        self, client, task_run_setup
    ):
        """The rejection is about the field, not about the run."""
        task = task_run_setup["task"]
        trace = self._eval_trace(task)

        with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
            mock_task_from_id.return_value = task
            response = client.patch(
                f"/api/projects/{project_id_of(task)}/tasks/{task.id}/runs/{trace.id}",
                json={"tags": ["reviewed"]},
            )

        assert response.status_code == 200
        assert response.json()["tags"] == ["reviewed"]
        assert TaskRun.load_from_file(trace.path).eval_source is not None


@pytest.mark.asyncio
async def test_edit_tags_add_and_remove(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    # Initial tags
    task_run.tags = ["tag1", "tag2", "tag3"]
    task_run.save_to_file()

    run_ids = [task_run.id]
    add_tags = ["new_tag1", "new_tag2"]
    remove_tags = ["tag1", "tag3"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/edit_tags",
            json={"run_ids": run_ids, "add_tags": add_tags, "remove_tags": remove_tags},
        )

    assert response.status_code == 200
    assert response.json() == {"success": True}

    # Verify tags were both added and removed correctly
    updated_run = TaskRun.from_id_and_parent_path(task_run.id, task.path)
    assert set(updated_run.tags) == {"tag2", "new_tag1", "new_tag2"}


@pytest.mark.asyncio
async def test_add_tags_success(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    # Initial tags
    task_run.tags = ["existing_tag"]
    task_run.save_to_file()

    run_ids = [task_run.id]
    new_tags = ["new_tag1", "new_tag2"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/edit_tags",
            json={"run_ids": run_ids, "add_tags": new_tags, "remove_tags": []},
        )

    assert response.status_code == 200
    assert response.json() == {"success": True}

    # Verify tags were added
    updated_run = TaskRun.from_id_and_parent_path(task_run.id, task.path)
    assert set(updated_run.tags) == {"existing_tag", "new_tag1", "new_tag2"}


@pytest.mark.asyncio
async def test_add_tags_duplicate_tags(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    # Initial tags
    task_run.tags = ["existing_tag"]
    task_run.save_to_file()

    run_ids = [task_run.id]
    new_tags = ["existing_tag", "new_tag"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/edit_tags",
            json={"run_ids": run_ids, "add_tags": new_tags},
        )

    assert response.status_code == 200
    assert response.json() == {"success": True}

    # Verify no duplicate tags were added
    updated_run = TaskRun.from_id_and_parent_path(task_run.id, task.path)
    assert set(updated_run.tags) == {"existing_tag", "new_tag"}


@pytest.mark.asyncio
async def test_add_tags_run_not_found(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]

    run_ids = ["non_existent_run_id"]
    new_tags = ["new_tag"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/edit_tags",
            json={"run_ids": run_ids, "add_tags": new_tags},
        )

    assert response.status_code == 500
    result = response.json()
    assert "failed_runs" in result["message"]
    assert "non_existent_run_id" in result["message"]["failed_runs"]
    assert result["message"]["error"] == "Runs not found"


@pytest.mark.asyncio
async def test_add_tags_task_not_found(client):
    run_ids = ["run1"]
    new_tags = ["new_tag"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.side_effect = HTTPException(
            status_code=404, detail="Task not found"
        )
        response = client.post(
            "/api/projects/project1-id/tasks/non_existent_task_id/runs/edit_tags",
            json={"run_ids": run_ids, "add_tags": new_tags},
        )

    assert response.status_code == 404
    assert response.json()["message"] == "Task not found"


@pytest.mark.asyncio
async def test_add_tags_multiple_runs(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    # Create a second run
    second_run = TaskRun(
        parent=task,
        input="Test input 2",
        input_source=DataSource(
            type=DataSourceType.human, properties={"created_by": "Test User"}
        ),
        output=TaskOutput(
            output="Test output 2",
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "Test User"},
            ),
        ),
    )
    second_run.save_to_file()

    run_ids = [task_run.id, second_run.id]
    new_tags = ["new_tag"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/edit_tags",
            json={"run_ids": run_ids, "add_tags": new_tags},
        )

    assert response.status_code == 200
    assert response.json() == {"success": True}

    # Verify tags were added to both runs
    for run_id in run_ids:
        updated_run = TaskRun.from_id_and_parent_path(run_id, task.path)
        assert "new_tag" in updated_run.tags


@pytest.mark.asyncio
async def test_remove_tags_success(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    # Initial tags
    task_run.tags = ["tag1", "tag2", "tag3"]
    task_run.save_to_file()

    run_ids = [task_run.id]
    tags_to_remove = ["tag1", "tag3"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/edit_tags",
            json={"run_ids": run_ids, "add_tags": [], "remove_tags": tags_to_remove},
        )

    assert response.status_code == 200
    assert response.json() == {"success": True}

    # Verify tags were removed
    updated_run = TaskRun.from_id_and_parent_path(task_run.id, task.path)
    assert set(updated_run.tags) == {"tag2"}


@pytest.mark.asyncio
async def test_remove_tags_nonexistent_tags(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    # Initial tags
    task_run.tags = ["tag1", "tag2"]
    task_run.save_to_file()

    run_ids = [task_run.id]
    tags_to_remove = ["tag3", "tag4"]  # Tags that don't exist

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/edit_tags",
            json={"run_ids": run_ids, "add_tags": [], "remove_tags": tags_to_remove},
        )

    assert response.status_code == 200
    assert response.json() == {"success": True}

    # Verify original tags remain unchanged
    updated_run = TaskRun.from_id_and_parent_path(task_run.id, task.path)
    assert set(updated_run.tags) == {"tag1", "tag2"}


@pytest.mark.asyncio
async def test_remove_tags_run_not_found(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]

    run_ids = ["non_existent_run_id"]
    tags_to_remove = ["tag1"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/edit_tags",
            json={"run_ids": run_ids, "add_tags": [], "remove_tags": tags_to_remove},
        )

    assert response.status_code == 500
    result = response.json()
    assert "failed_runs" in result["message"]
    assert "non_existent_run_id" in result["message"]["failed_runs"]
    assert result["message"]["error"] == "Runs not found"


@pytest.mark.asyncio
async def test_remove_tags_task_not_found(client):
    run_ids = ["run1"]
    tags_to_remove = ["tag1"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.side_effect = HTTPException(
            status_code=404, detail="Task not found"
        )
        response = client.post(
            "/api/projects/project1-id/tasks/non_existent_task_id/runs/edit_tags",
            json={"run_ids": run_ids, "add_tags": [], "remove_tags": tags_to_remove},
        )

    assert response.status_code == 404
    assert response.json()["message"] == "Task not found"


@pytest.mark.asyncio
async def test_remove_tags_multiple_runs(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    # Create a second run
    second_run = TaskRun(
        parent=task,
        input="Test input 2",
        input_source=DataSource(
            type=DataSourceType.human, properties={"created_by": "Test User"}
        ),
        output=TaskOutput(
            output="Test output 2",
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "Test User"},
            ),
        ),
    )
    second_run.save_to_file()

    # Set initial tags for both runs
    task_run.tags = ["tag1", "tag2"]
    second_run.tags = ["tag1", "tag3"]
    task_run.save_to_file()
    second_run.save_to_file()

    run_ids = [task_run.id, second_run.id]
    tags_to_remove = ["tag1"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/edit_tags",
            json={"run_ids": run_ids, "add_tags": [], "remove_tags": tags_to_remove},
        )

    assert response.status_code == 200
    assert response.json() == {"success": True}

    # Verify tags were removed from both runs
    updated_run1 = TaskRun.from_id_and_parent_path(task_run.id, task.path)
    updated_run2 = TaskRun.from_id_and_parent_path(second_run.id, task.path)
    assert set(updated_run1.tags) == {"tag2"}
    assert set(updated_run2.tags) == {"tag3"}


def test_model_provider_from_string():
    assert model_provider_from_string("openai") == ModelProviderName.openai
    assert model_provider_from_string("ollama") == ModelProviderName.ollama

    with pytest.raises(ValueError, match="Unsupported provider: unknown"):
        model_provider_from_string("unknown")


@pytest.mark.asyncio
async def test_run_task_invalid_temperature_values(client, task_run_setup):
    """Test that invalid temperature values return 422 errors."""
    project = task_run_setup["project"]
    task = task_run_setup["task"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task

        # Test temperature below 0
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/run",
            json={
                "run_config_properties": {
                    "model_name": "gpt-4o",
                    "model_provider_name": "openai",
                    "prompt_id": "simple_prompt_builder",
                    "temperature": -0.1,
                    "structured_output_mode": "json_schema",
                },
                "plaintext_input": "Test input",
            },
        )
        assert response.status_code == 422
        error_detail = response.json()["message"]
        assert "temperature must be between 0 and 2" in str(error_detail)

        # Test temperature above 2
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/run",
            json={
                "run_config_properties": {
                    "model_name": "gpt-4o",
                    "model_provider_name": "openai",
                    "prompt_id": "simple_prompt_builder",
                    "temperature": 2.1,
                    "structured_output_mode": "json_schema",
                },
                "plaintext_input": "Test input",
            },
        )
        assert response.status_code == 422
        error_detail = response.json()["message"]
        assert "temperature must be between 0 and 2" in str(error_detail)


@pytest.mark.asyncio
async def test_run_task_invalid_top_p_values(client, task_run_setup):
    """Test that invalid top_p values return 422 errors."""
    project = task_run_setup["project"]
    task = task_run_setup["task"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task

        # Test top_p below 0
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/run",
            json={
                "run_config_properties": {
                    "model_name": "gpt-4o",
                    "model_provider_name": "openai",
                    "prompt_id": "simple_prompt_builder",
                    "top_p": -0.1,
                    "structured_output_mode": "json_schema",
                },
                "plaintext_input": "Test input",
            },
        )
        assert response.status_code == 422
        error_detail = response.json()["message"]
        assert "top_p must be between 0 and 1" in str(error_detail)

        # Test top_p above 1
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/run",
            json={
                "run_config_properties": {
                    "model_name": "gpt-4o",
                    "model_provider_name": "openai",
                    "prompt_id": "simple_prompt_builder",
                    "top_p": 1.1,
                    "structured_output_mode": "json_schema",
                },
                "plaintext_input": "Test input",
            },
        )
        assert response.status_code == 422
        error_detail = response.json()["message"]
        assert "top_p must be between 0 and 1" in str(error_detail)


@pytest.mark.asyncio
async def test_run_task_valid_boundary_values(client, task_run_setup):
    """Test that valid boundary values for temperature and top_p work correctly."""
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    with (
        patch("kiln_server.run_api.task_from_id") as mock_task_from_id,
        patch.object(LiteLlmAdapter, "invoke", new_callable=AsyncMock) as mock_invoke,
        patch("kiln_ai.utils.config.Config.shared") as MockConfig,
    ):
        mock_task_from_id.return_value = task
        mock_invoke.return_value = task_run

        # Mock the Config class
        mock_config_instance = MockConfig.return_value
        mock_config_instance.open_ai_api_key = "test_key"

        # Test valid boundary values - temperature = 0, top_p = 0
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/run",
            json={
                "run_config_properties": {
                    "model_name": "gpt-4o",
                    "model_provider_name": "openai",
                    "prompt_id": "simple_prompt_builder",
                    "temperature": 0.0,
                    "top_p": 0.0,
                    "structured_output_mode": "json_schema",
                },
                "plaintext_input": "Test input",
            },
        )
        assert response.status_code == 200

        # Test valid boundary values - temperature = 2, top_p = 1
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/run",
            json={
                "run_config_properties": {
                    "model_name": "gpt-4o",
                    "model_provider_name": "openai",
                    "prompt_id": "simple_prompt_builder",
                    "temperature": 2.0,
                    "top_p": 1.0,
                    "structured_output_mode": "json_schema",
                },
                "plaintext_input": "Test input",
            },
        )
        assert response.status_code == 200


@pytest.mark.parametrize(
    "input_str,expected",
    [
        (None, None),  # None input returns None
        ("", None),  # Empty string returns None
        ('{"train": 0.8, "test": 0.2}', {"train": 0.8, "test": 0.2}),  # Valid JSON dict
        ('{"train": 0.8}', {"train": 0.8}),  # Single key-value pair
        ('{"train": 1.0, "test": 0.0}', {"train": 1.0, "test": 0.0}),  # Boundary values
        (
            '{"train": 0.75, "test": 0.25}',
            {"train": 0.75, "test": 0.25},
        ),  # Decimal values
        ('{"train": 1, "test": 0}', {"train": 1, "test": 0}),  # Integer values
    ],
)
def test_parse_splits_valid(input_str, expected):
    assert parse_splits(input_str) == expected


@pytest.mark.parametrize(
    "input_str,expected_error",
    [
        # Invalid JSON
        (
            "{invalid json}",
            "Invalid splits format. Must be a valid JSON object with string keys and float values.",
        ),
        # JSON array instead of dict
        (
            "[1, 2, 3]",
            "Invalid splits format. Must be a valid JSON object with string keys and float values.",
        ),
        # JSON string instead of dict
        (
            '"not a dict"',
            "Invalid splits format. Must be a valid JSON object with string keys and float values.",
        ),
        # Non-string keys
        (
            '{"train": 0.8, 123: 0.2}',
            "Invalid splits format. Must be a valid JSON object with string keys and float values.",
        ),
        # Non-numeric values
        (
            '{"train": "0.8", "test": "0.2"}',
            "Invalid splits format. Must be a valid JSON object with string keys and float values.",
        ),
        # Values out of range (> 1)
        (
            '{"train": 1.2, "test": 0.2}',
            "Invalid splits format. Must be a valid JSON object with string keys and float values.",
        ),
        # Values out of range (< 0)
        (
            '{"train": -0.2, "test": 0.2}',
            "Invalid splits format. Must be a valid JSON object with string keys and float values.",
        ),
        # Mixed valid/invalid values
        (
            '{"train": 0.8, "test": 1.5}',
            "Invalid splits format. Must be a valid JSON object with string keys and float values.",
        ),
        # Boolean values
        (
            '{"train": true, "test": false}',
            "Invalid splits format. Must be a valid JSON object with string keys and float values.",
        ),
        # Null values
        (
            '{"train": null, "test": 0.5}',
            "Invalid splits format. Must be a valid JSON object with string keys and float values.",
        ),
    ],
)
def test_parse_splits_invalid(input_str, expected_error):
    with pytest.raises(HTTPException) as exc_info:
        parse_splits(input_str)
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == expected_error


@pytest.mark.asyncio
async def test_get_tags_success(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    # Set tags on the existing run
    task_run.tags = ["tag1", "tag2", "tag3"]
    task_run.save_to_file()

    # Create a second run with overlapping tags
    second_run = TaskRun(
        parent=task,
        input="Test input 2",
        input_source=DataSource(
            type=DataSourceType.human, properties={"created_by": "Test User"}
        ),
        output=TaskOutput(
            output="Test output 2",
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "Test User"},
            ),
        ),
        tags=["tag2", "tag3", "tag4"],
    )
    second_run.save_to_file()

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task = MagicMock()
        mock_task.runs.return_value = [task_run, second_run]
        mock_task_from_id.return_value = mock_task

        response = client.get(f"/api/projects/{project.id}/tasks/{task.id}/tags")

    assert response.status_code == 200
    result = response.json()

    # Verify tag counts: tag1(1), tag2(2), tag3(2), tag4(1)
    expected_counts = {"tag1": 1, "tag2": 2, "tag3": 2, "tag4": 1}
    assert result == expected_counts


def create_multiple_task_runs(task: Task, count: int) -> list[TaskRun]:
    """Helper function to create multiple TaskRuns for benchmarking."""
    runs = []
    for i in range(count):
        run = TaskRun(
            parent=task,
            input=f"Test input {i}",
            input_source=DataSource(
                type=DataSourceType.human, properties={"created_by": "Test User"}
            ),
            output=TaskOutput(
                output=f"Test output {i}",
                source=DataSource(
                    type=DataSourceType.synthetic,
                    properties={
                        "model_name": "gpt_4o",
                        "model_provider": "ollama",
                        "adapter_name": "kiln_langchain_adapter",
                        "prompt_id": "simple_prompt_builder",
                    },
                ),
            ),
        )
        run.save_to_file()
        runs.append(run)
    return runs


# Not actually paid, but we want the "must be run manually" feature of the paid marker as this is very slow
@pytest.mark.paid
@pytest.mark.parametrize("run_count", [100, 1000, 10000, 50000])
async def test_benchmark_tag_runs(client, task_run_setup, run_count):
    """Benchmark test for tagging TaskRuns with different counts."""
    project = task_run_setup["project"]
    task = task_run_setup["task"]

    # Create TaskRuns
    runs = create_multiple_task_runs(task, run_count)
    run_ids = [run.id for run in runs]

    # Benchmark the tagging operation
    start_time = time.perf_counter()

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/edit_tags",
            json={"run_ids": run_ids, "add_tags": ["benchmark_tag"], "remove_tags": []},
        )

    end_time = time.perf_counter()
    duration = end_time - start_time

    assert response.status_code == 200
    assert response.json() == {"success": True}

    # Calculate performance statistics
    runs_per_second = run_count / duration if duration > 0 else 0
    avg_time_per_run = duration / run_count if run_count > 0 else 0

    logger = logging.getLogger(__name__)
    logger.info(f"Tagging {run_count} runs took: {duration:.3f} seconds")
    logger.info(
        f"Performance: {runs_per_second:.1f} runs/second, {avg_time_per_run * 1000:.2f}ms per run"
    )


def _adapter_sanity_check_output_path() -> Path:
    return Path(__file__).resolve().parent / "adapter_sanity_check.txt"


def _append_to_sanity_check(content: str, output_path: Path) -> None:
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(content)
        f.write("\n")


@pytest.fixture
def adapter_sanity_check_setup(tmp_path):
    """Setup for paid adapter sanity check tests - real project/task, no adapter mocking."""
    # if project at the path does not exist, create it, otherwise reuse
    project_path = (
        Path("/Users/leonardmarcq/Downloads/")
        / "adapter_sanity_project"
        / "project.kiln"
    )
    if not project_path.exists():
        project_path.parent.mkdir()

        project = Project(name="Adapter Sanity Project", path=str(project_path))
        project.save_to_file()

        task = Task(
            name="Adapter Sanity Task",
            instruction="You are a helpful assistant. Respond concisely.",
            description="Task for adapter sanity checking",
            parent=project,
        )
        task.save_to_file()

    else:
        project = Project.load_from_file(project_path)
        task = next(
            (
                t
                for t in project.tasks(readonly=True)
                if t.name == "Adapter Sanity Task"
            ),
            None,
        )
        if task is None:
            raise ValueError("Task not found")

    config = Config.shared()
    original_projects = list(config.projects) if config.projects else []
    config._settings["projects"] = [*original_projects, str(project.path)]

    yield {"project": project, "task": task}

    config._settings["projects"] = original_projects


@pytest.fixture
def adapter_sanity_check_math_tools_setup(tmp_path):
    """Setup for paid math tools test - task with instructions to use add, multiply, etc."""
    project_path = tmp_path / "adapter_sanity_math_project" / "project.kiln"
    project_path.parent.mkdir()

    project = Project(name="Adapter Sanity Math Project", path=str(project_path))
    project.save_to_file()

    task = Task(
        name="Math Tools Task",
        instruction="You are an assistant that performs math using the provided tools. You MUST use the add, subtract, multiply, and divide tools for any arithmetic. For example, for 2+2 you must call the add tool with a=2 and b=2. End your response with the final answer in square brackets, e.g. [4].",
        description="Task for testing Kiln built-in math tools",
        parent=project,
    )
    task.save_to_file()

    config = Config.shared()
    original_projects = list(config.projects) if config.projects else []
    config._settings["projects"] = [*original_projects, str(project.path)]

    yield {"project": project, "task": task}

    config._settings["projects"] = original_projects


def _assert_math_tools_response(res: dict, expected_in_output: str) -> None:
    """Assert response has correct output, trace with tool calls, and output matches latest message."""
    assert res["id"] is not None

    output = res.get("output", {}).get("output", "")
    assert output is not None
    assert expected_in_output in output

    trace = res.get("trace") or []
    assistant_with_tool_calls = [
        m for m in trace if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert len(assistant_with_tool_calls) >= 1

    tool_messages = [m for m in trace if m.get("role") == "tool"]
    assert len(tool_messages) >= 1

    last_assistant = next(
        (m for m in reversed(trace) if m.get("role") == "assistant"), None
    )
    assert last_assistant is not None
    last_content = last_assistant.get("content") or ""
    assert expected_in_output in last_content

    intermediate_outputs = res.get("intermediate_outputs") or {}
    if intermediate_outputs:
        for key, value in intermediate_outputs.items():
            assert isinstance(value, str)


@pytest.mark.paid
@pytest.mark.asyncio
async def test_run_task_adapter_sanity_math_tools(
    client, adapter_sanity_check_math_tools_setup
):
    """Multiple runs with built-in Kiln math tools. Test that tools work across independent runs."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY required for this test")

    project = adapter_sanity_check_math_tools_setup["project"]
    task = adapter_sanity_check_math_tools_setup["task"]

    run_config = {
        "model_name": "gpt_5_nano",
        "model_provider_name": "openrouter",
        "prompt_id": "simple_prompt_builder",
        "structured_output_mode": "json_schema",
        "tools_config": {
            "tools": [
                KilnBuiltInToolId.ADD_NUMBERS.value,
                KilnBuiltInToolId.SUBTRACT_NUMBERS.value,
                KilnBuiltInToolId.MULTIPLY_NUMBERS.value,
                KilnBuiltInToolId.DIVIDE_NUMBERS.value,
            ]
        },
    }

    response1 = client.post(
        f"/api/projects/{project.id}/tasks/{task.id}/run",
        json={
            "run_config_properties": run_config,
            "plaintext_input": "What is 2 + 2? Use the tools to calculate.",
        },
    )
    assert response1.status_code == 200
    res1 = response1.json()
    _assert_math_tools_response(res1, "4")

    response2 = client.post(
        f"/api/projects/{project.id}/tasks/{task.id}/run",
        json={
            "run_config_properties": run_config,
            "plaintext_input": "What is 3 times 4? Use the tools to calculate.",
        },
    )
    assert response2.status_code == 200
    res2 = response2.json()
    _assert_math_tools_response(res2, "12")

    response3 = client.post(
        f"/api/projects/{project.id}/tasks/{task.id}/run",
        json={
            "run_config_properties": run_config,
            "plaintext_input": "What is 7 times 8 plus 3? Use the tools to calculate.",
        },
    )
    assert response3.status_code == 200
    res3 = response3.json()
    _assert_math_tools_response(res3, "59")

    response4 = client.post(
        f"/api/projects/{project.id}/tasks/{task.id}/run",
        json={
            "run_config_properties": run_config,
            "plaintext_input": "What is 10 minus 3? Use the tools to calculate.",
        },
    )
    assert response4.status_code == 200
    res4 = response4.json()
    _assert_math_tools_response(res4, "7")


def _make_task_run(
    task,
    *,
    input_text,
    output_text,
    parent_task_run_id=None,
    trace=None,
):
    run = TaskRun(
        parent=task,
        parent_task_run_id=parent_task_run_id,
        input=input_text,
        input_source=DataSource(
            type=DataSourceType.human, properties={"created_by": "Test User"}
        ),
        output=TaskOutput(
            output=output_text,
            source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "gpt_4o",
                    "model_provider": "ollama",
                    "adapter_name": "kiln_langchain_adapter",
                    "prompt_id": "simple_prompt_builder",
                },
            ),
        ),
        trace=trace,
    )
    run.save_to_file()
    return run


@pytest.fixture
def multiturn_task_run_setup(tmp_path):
    """Multiturn task with a single saved parent TaskRun whose trace has one user/assistant exchange."""
    project_path = tmp_path / "test_project" / "project.kiln"
    project_path.parent.mkdir()

    project = Project(name="Multiturn Project", path=str(project_path))
    project.save_to_file()

    task = Task(
        name="Multiturn Task",
        instruction="Have a conversation",
        description="Multiturn test task",
        parent=project,
        turn_mode="multiturn",
    )
    task.save_to_file()

    parent_run = _make_task_run(
        task,
        input_text="hi",
        output_text="hello",
        trace=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )

    return {"project": project, "task": task, "parent_run": parent_run}


@pytest.mark.asyncio
async def test_run_task_continuation_multiturn(client, multiturn_task_run_setup):
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]
    parent_run = multiturn_task_run_setup["parent_run"]

    follow_up_run = _make_task_run(
        task,
        input_text="how are you?",
        output_text="great, thanks!",
        parent_task_run_id=parent_run.id,
        trace=[
            *(parent_run.trace or []),
            {"role": "user", "content": "how are you?"},
            {"role": "assistant", "content": "great, thanks!"},
        ],
    )

    run_request = {
        "run_config_properties": {
            "model_name": "gpt_4o",
            "model_provider_name": "ollama",
            "prompt_id": "simple_prompt_builder",
            "structured_output_mode": "json_schema",
        },
        "plaintext_input": "how are you?",
        "parent_task_run_id": parent_run.id,
    }

    with (
        patch("kiln_server.run_api.task_from_id") as mock_task_from_id,
        patch.object(LiteLlmAdapter, "invoke", new_callable=AsyncMock) as mock_invoke,
        patch("kiln_ai.utils.config.Config.shared") as MockConfig,
    ):
        mock_task_from_id.return_value = task
        mock_invoke.return_value = follow_up_run

        mock_config_instance = MockConfig.return_value
        mock_config_instance.ollama_base_url = "http://localhost:11434/v1"

        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/run", json=run_request
        )

    assert response.status_code == 200

    # Plumbing check: API forwarded prior_trace and the parent run to the adapter.
    # (We don't re-assert the response body's trace/parent_task_run_id — those come
    # from the locally fabricated follow_up_run, not from real adapter behavior.)
    mock_invoke.assert_awaited_once()
    invoke_kwargs = mock_invoke.await_args.kwargs
    assert invoke_kwargs["prior_trace"] == parent_run.trace
    assert invoke_kwargs["parent_task_run"].id == parent_run.id


@pytest.mark.asyncio
async def test_run_task_continuation_single_turn_rejected(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    parent_run = task_run_setup["task_run"]

    run_request = {
        "run_config_properties": {
            "model_name": "gpt_4o",
            "model_provider_name": "ollama",
            "prompt_id": "simple_prompt_builder",
            "structured_output_mode": "json_schema",
        },
        "plaintext_input": "follow up",
        "parent_task_run_id": parent_run.id,
    }

    with (
        patch("kiln_server.run_api.task_from_id") as mock_task_from_id,
        patch.object(LiteLlmAdapter, "invoke", new_callable=AsyncMock) as mock_invoke,
    ):
        mock_task_from_id.return_value = task

        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/run", json=run_request
        )

    assert response.status_code == 400
    assert "multi-turn" in response.json()["message"].lower()
    assert "parent_task_run_id" in response.json()["message"]
    mock_invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_task_continuation_parent_not_found(client, multiturn_task_run_setup):
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]

    run_request = {
        "run_config_properties": {
            "model_name": "gpt_4o",
            "model_provider_name": "ollama",
            "prompt_id": "simple_prompt_builder",
            "structured_output_mode": "json_schema",
        },
        "plaintext_input": "follow up",
        "parent_task_run_id": "missing_parent_run_id",
    }

    with (
        patch("kiln_server.run_api.task_from_id") as mock_task_from_id,
        patch.object(LiteLlmAdapter, "invoke", new_callable=AsyncMock) as mock_invoke,
    ):
        mock_task_from_id.return_value = task

        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/run", json=run_request
        )

    assert response.status_code == 404
    assert "parent run not found" in response.json()["message"].lower()
    assert "missing_parent_run_id" in response.json()["message"]
    mock_invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_task_continuation_parent_deleted_mid_invoke(
    client, multiturn_task_run_setup
):
    """If the conversation is deleted while the model is generating, the
    continuation must not resurrect it: the just-saved child is removed from
    disk and the request fails with a clear conflict error."""
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]
    parent_run = multiturn_task_run_setup["parent_run"]

    run_request = {
        "run_config_properties": {
            "model_name": "gpt_4o",
            "model_provider_name": "ollama",
            "prompt_id": "simple_prompt_builder",
            "structured_output_mode": "json_schema",
        },
        "plaintext_input": "how are you?",
        "parent_task_run_id": parent_run.id,
    }

    saved_child_ids: list[str] = []

    async def delete_parent_then_return(*args, **kwargs):
        # Simulate a cascade delete landing mid-generation: the adapter saves
        # the new child run (as autosave does), but the parent is already gone
        # by the time invoke returns.
        follow_up = _make_task_run(
            task,
            input_text="how are you?",
            output_text="great, thanks!",
            parent_task_run_id=parent_run.id,
            trace=[
                *(parent_run.trace or []),
                {"role": "user", "content": "how are you?"},
                {"role": "assistant", "content": "great, thanks!"},
            ],
        )
        saved_child_ids.append(str(follow_up.id))
        parent_run.delete()
        return follow_up

    with (
        patch("kiln_server.run_api.task_from_id") as mock_task_from_id,
        patch.object(LiteLlmAdapter, "invoke", new_callable=AsyncMock) as mock_invoke,
        patch("kiln_ai.utils.config.Config.shared") as MockConfig,
    ):
        mock_task_from_id.return_value = task
        mock_invoke.side_effect = delete_parent_then_return

        mock_config_instance = MockConfig.return_value
        mock_config_instance.ollama_base_url = "http://localhost:11434/v1"
        # The child run is built while Config is mocked; created_by must be a
        # real string or TaskRun validation rejects it.
        mock_config_instance.user_id = "test_user"

        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/run", json=run_request
        )

    assert response.status_code == 409
    assert "deleted" in response.json()["message"].lower()
    # The orphaned child must not remain on disk.
    assert len(saved_child_ids) == 1
    ModelCache.shared().clear()
    assert TaskRun.from_id_and_parent_path(saved_child_ids[0], task.path) is None


@pytest.mark.asyncio
async def test_get_runs_summaries_multiturn_returns_leaf_only(
    client, multiturn_task_run_setup
):
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]
    run_a = multiturn_task_run_setup["parent_run"]

    run_b = _make_task_run(
        task,
        input_text="turn 2",
        output_text="reply 2",
        parent_task_run_id=run_a.id,
    )
    run_c = _make_task_run(
        task,
        input_text="turn 3",
        output_text="reply 3",
        parent_task_run_id=run_b.id,
    )

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs_summaries"
        )

    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert [r["id"] for r in result] == [run_c.id]


@pytest.mark.asyncio
async def test_get_runs_summaries_single_turn_returns_all_runs(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    first_run = task_run_setup["task_run"]

    second_run = _make_task_run(
        task, input_text="independent input 2", output_text="independent output 2"
    )
    third_run = _make_task_run(
        task, input_text="independent input 3", output_text="independent output 3"
    )

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs_summaries"
        )

    assert response.status_code == 200
    result = response.json()
    assert {r["id"] for r in result} == {first_run.id, second_run.id, third_run.id}


def _user_turn(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_turn(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _build_three_turn_chain(task):
    """Build root -> mid -> leaf with consistent traces (1, 2, 3 user messages)."""
    root = _make_task_run(
        task,
        input_text="turn 1",
        output_text="reply 1",
        trace=[_user_turn("turn 1"), _assistant_turn("reply 1")],
    )
    mid = _make_task_run(
        task,
        input_text="turn 2",
        output_text="reply 2",
        parent_task_run_id=root.id,
        trace=[
            _user_turn("turn 1"),
            _assistant_turn("reply 1"),
            _user_turn("turn 2"),
            _assistant_turn("reply 2"),
        ],
    )
    leaf = _make_task_run(
        task,
        input_text="turn 3",
        output_text="reply 3",
        parent_task_run_id=mid.id,
        trace=[
            _user_turn("turn 1"),
            _assistant_turn("reply 1"),
            _user_turn("turn 2"),
            _assistant_turn("reply 2"),
            _user_turn("turn 3"),
            _assistant_turn("reply 3"),
        ],
    )
    return root, mid, leaf


@pytest.mark.asyncio
async def test_get_chain_single_turn_task_returns_400(client, task_run_setup):
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    run = task_run_setup["task_run"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{run.id}/chain"
        )

    assert response.status_code == 400
    assert "multi-turn" in response.json()["message"].lower()


@pytest.mark.asyncio
async def test_get_chain_run_not_found_returns_404(client, multiturn_task_run_setup):
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/missing_run/chain"
        )

    assert response.status_code == 404
    assert "not found" in response.json()["message"].lower()
    assert "missing_run" in response.json()["message"]


@pytest.mark.asyncio
async def test_get_chain_single_turn_chain_returns_self_only(
    client, multiturn_task_run_setup
):
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]
    parent_run = multiturn_task_run_setup["parent_run"]

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{parent_run.id}/chain"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["chain_broken"] is False
    assert body["chain"] == [
        {"run_id": parent_run.id, "turn_index": 1, "trace_start_index": 0}
    ]


@pytest.mark.asyncio
async def test_get_chain_degenerate_trace_returns_empty_with_chain_broken(
    client, multiturn_task_run_setup
):
    """A leaf whose trace has no user messages is the turn_count == 0 branch:
    we can't position any run on a turn, so the endpoint surfaces empty
    chain + chain_broken=True (rather than 500-ing or returning a misaligned
    list). Without this test the branch is unexercised."""
    task = multiturn_task_run_setup["task"]
    project = multiturn_task_run_setup["project"]
    leaf = _make_task_run(
        task,
        input_text="solo",
        output_text="reply",
        trace=[
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "reply"},
        ],
    )

    ModelCache.shared().clear()

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{leaf.id}/chain"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["chain"] == []
    assert body["chain_broken"] is True
    assert body["has_children"] is False


@pytest.mark.asyncio
async def test_get_chain_three_turn_chain_returns_full_chain_root_to_leaf(
    client, multiturn_task_run_setup
):
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]
    # The fixture's parent_run is unrelated to the chain we build here.
    root, mid, leaf = _build_three_turn_chain(task)

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{leaf.id}/chain"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["chain_broken"] is False
    # Each turn starts where its parent's trace ended: root at 0, mid after
    # root's 2 messages, leaf after mid's 4.
    assert body["chain"] == [
        {"run_id": root.id, "turn_index": 1, "trace_start_index": 0},
        {"run_id": mid.id, "turn_index": 2, "trace_start_index": 2},
        {"run_id": leaf.id, "turn_index": 3, "trace_start_index": 4},
    ]


@pytest.mark.asyncio
async def test_get_chain_missing_parent_mid_chain_sets_chain_broken(
    client, multiturn_task_run_setup
):
    """If a parent run file is deleted from disk, the walk stops there.

    The endpoint returns the intact suffix (leaf only) with chain_broken=true.
    """
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]
    _root, mid, leaf = _build_three_turn_chain(task)

    # Delete the mid run's directory from disk. Also evict the cache so the
    # loader actually re-reads the filesystem.
    assert mid.path is not None
    shutil.rmtree(Path(mid.path).parent)
    ModelCache.shared().clear()

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{leaf.id}/chain"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["chain_broken"] is True
    # Only the leaf remains reachable. With its ancestors missing, its absolute
    # turn number and trace boundary are unknowable: turn_index is relative to
    # the returned suffix and trace_start_index is None.
    assert body["chain"] == [
        {"run_id": leaf.id, "turn_index": 1, "trace_start_index": None}
    ]


@pytest.mark.asyncio
async def test_get_chain_cycle_in_chain_is_handled(client, multiturn_task_run_setup):
    """Two runs whose parent_task_run_id point at each other terminate cleanly."""
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]

    # Build a normal pair, then mutate the root's parent_task_run_id to point at
    # the leaf to create a cycle.
    root = _make_task_run(
        task,
        input_text="a",
        output_text="A",
        trace=[_user_turn("a"), _assistant_turn("A")],
    )
    leaf = _make_task_run(
        task,
        input_text="b",
        output_text="B",
        parent_task_run_id=root.id,
        trace=[
            _user_turn("a"),
            _assistant_turn("A"),
            _user_turn("b"),
            _assistant_turn("B"),
        ],
    )
    root.parent_task_run_id = leaf.id
    root.save_to_file()

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{leaf.id}/chain"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["chain_broken"] is True
    # The walk visits leaf, then root (parent of leaf), then finds root's
    # parent (leaf) already visited → stops, returning both with broken=true.
    assert [a["run_id"] for a in body["chain"]] == [root.id, leaf.id]
    # The suffix root's boundary is unknowable (its recorded parent was not
    # resolved); the leaf still anchors after root's 2 trace messages.
    assert [a["trace_start_index"] for a in body["chain"]] == [None, 2]


@pytest.mark.asyncio
async def test_get_chain_run_trace_not_extending_parents_is_broken(
    client, multiturn_task_run_setup
):
    """Pathological: 3-run chain whose leaf trace (2 messages) is shorter than
    its parent's (4), so the leaf's turn can't be positioned after its
    ancestors'. Endpoint trims to the suffix (leaf only, boundary unknown) and
    reports chain_broken=true.
    """
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]

    root = _make_task_run(
        task,
        input_text="a",
        output_text="A",
        trace=[_user_turn("a"), _assistant_turn("A")],
    )
    mid = _make_task_run(
        task,
        input_text="b",
        output_text="B",
        parent_task_run_id=root.id,
        trace=[
            _user_turn("a"),
            _assistant_turn("A"),
            _user_turn("b"),
            _assistant_turn("B"),
        ],
    )
    leaf = _make_task_run(
        task,
        input_text="c",
        output_text="C",
        parent_task_run_id=mid.id,
        trace=[_user_turn("c"), _assistant_turn("C")],
    )

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{leaf.id}/chain"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["chain_broken"] is True
    assert body["chain"] == [
        {"run_id": leaf.id, "turn_index": 1, "trace_start_index": None}
    ]


@pytest.mark.asyncio
async def test_get_chain_cot_root_turn_positions_structurally(
    client, multiturn_task_run_setup
):
    """A chain-of-thought root emits TWO user messages for a single turn (the
    real input plus the final-answer prompt). Turn structure must come from the
    chain itself, not user-message counting: a 2-run chain with a CoT root is
    turns 1 and 2, with turn 2 anchored after the root's full 5-message trace.
    """
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]

    cot_root_trace = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "turn 1 + thinking instructions"},
        {"role": "assistant", "content": "thinking..."},
        {"role": "user", "content": "Considering the above, return a final result."},
        {"role": "assistant", "content": "reply 1"},
    ]
    root = _make_task_run(
        task,
        input_text="turn 1",
        output_text="reply 1",
        trace=cot_root_trace,
    )
    leaf = _make_task_run(
        task,
        input_text="turn 2",
        output_text="reply 2",
        parent_task_run_id=root.id,
        trace=[
            *cot_root_trace,
            _user_turn("turn 2"),
            _assistant_turn("reply 2"),
        ],
    )

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{leaf.id}/chain"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["chain_broken"] is False
    assert body["chain"] == [
        {"run_id": root.id, "turn_index": 1, "trace_start_index": 0},
        {"run_id": leaf.id, "turn_index": 2, "trace_start_index": 5},
    ]


@pytest.mark.asyncio
async def test_get_chain_has_children_true_for_intermediate_run(
    client, multiturn_task_run_setup
):
    """An intermediate node (one with at least one child) reports has_children=true."""
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]
    root, mid, _leaf = _build_three_turn_chain(task)

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        root_response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{root.id}/chain"
        )
        mid_response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{mid.id}/chain"
        )

    assert root_response.status_code == 200
    assert root_response.json()["has_children"] is True
    assert mid_response.status_code == 200
    assert mid_response.json()["has_children"] is True


@pytest.mark.asyncio
async def test_get_chain_has_children_false_for_leaf_run(
    client, multiturn_task_run_setup
):
    """A leaf node (no other run points at it) reports has_children=false."""
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]
    _root, _mid, leaf = _build_three_turn_chain(task)

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{leaf.id}/chain"
        )

    assert response.status_code == 200
    assert response.json()["has_children"] is False


@pytest.mark.asyncio
async def test_delete_leaf_cascades_full_three_turn_chain(
    client, multiturn_task_run_setup
):
    """Deleting the leaf of a 3-turn linear chain deletes every run in the chain."""
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]
    root, mid, leaf = _build_three_turn_chain(task)

    # Sanity: all three files exist on disk before we delete.
    assert root.path.exists()
    assert mid.path.exists()
    assert leaf.path.exists()

    ModelCache.shared().clear()

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.delete(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{leaf.id}"
        )

    assert response.status_code == 200
    assert not root.path.exists()
    assert not mid.path.exists()
    assert not leaf.path.exists()


@pytest.mark.asyncio
async def test_delete_leaf_stops_at_branching_root(client, multiturn_task_run_setup):
    """A root with two child branches must survive when one branch's leaf is deleted."""
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]

    root = _make_task_run(
        task,
        input_text="turn 1",
        output_text="reply 1",
        trace=[_user_turn("turn 1"), _assistant_turn("reply 1")],
    )
    branch_a_leaf = _make_task_run(
        task,
        input_text="branch a",
        output_text="reply a",
        parent_task_run_id=root.id,
    )
    branch_b_leaf = _make_task_run(
        task,
        input_text="branch b",
        output_text="reply b",
        parent_task_run_id=root.id,
    )

    ModelCache.shared().clear()

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.delete(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{branch_a_leaf.id}"
        )

    assert response.status_code == 200
    # Branch A's leaf is gone; the root and branch B's leaf both survive
    # because the root still has a live child.
    assert not branch_a_leaf.path.exists()
    assert root.path.exists()
    assert branch_b_leaf.path.exists()


@pytest.mark.asyncio
async def test_delete_interior_run_cascades_only_upward(
    client, multiturn_task_run_setup
):
    """Deleting a mid-chain run cascades to its ancestors but NOT its descendants."""
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]
    root, mid, leaf = _build_three_turn_chain(task)

    ModelCache.shared().clear()

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.delete(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{mid.id}"
        )

    assert response.status_code == 200
    # root had only one child (mid), and that child is now in the delete-set,
    # so root is removed too. The leaf (a descendant) is NOT touched.
    assert not root.path.exists()
    assert not mid.path.exists()
    assert leaf.path.exists()


@pytest.mark.asyncio
async def test_delete_with_cycle_in_parent_chain_stops_cleanly(
    client, multiturn_task_run_setup
):
    """A cycle in parent_task_run_id must not 500 the delete endpoint."""
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]

    a = _make_task_run(
        task,
        input_text="a",
        output_text="A",
        trace=[_user_turn("a"), _assistant_turn("A")],
    )
    b = _make_task_run(
        task,
        input_text="b",
        output_text="B",
        parent_task_run_id=a.id,
    )
    # Point a.parent_task_run_id at b to create a 2-node cycle.
    a.parent_task_run_id = b.id
    a.save_to_file()

    ModelCache.shared().clear()

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.delete(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{b.id}"
        )

    # Should not raise. b deleted; a's only remaining child (b) is deleted, so
    # a also gets deleted. The cycle check prevents revisiting b.
    assert response.status_code == 200
    assert not a.path.exists()
    assert not b.path.exists()


@pytest.mark.asyncio
async def test_delete_with_missing_parent_stops_cascade_cleanly(
    client, multiturn_task_run_setup
):
    """A broken parent reference stops the cascade without 500ing."""
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]
    _root, mid, leaf = _build_three_turn_chain(task)

    # Remove mid from disk while the leaf still references it.
    assert mid.path is not None
    shutil.rmtree(Path(mid.path).parent)
    ModelCache.shared().clear()

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.delete(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{leaf.id}"
        )

    assert response.status_code == 200
    # Leaf is gone. Cascade stops at the break — the still-present root is
    # NOT swept up because we couldn't verify mid (its only child) is dead.
    assert not leaf.path.exists()
    assert _root.path.exists()


@pytest.mark.asyncio
async def test_delete_single_turn_run_unchanged(client, task_run_setup):
    """Single-turn runs have no parent chain — behavior is the same as before."""
    project = task_run_setup["project"]
    task = task_run_setup["task"]
    task_run = task_run_setup["task_run"]

    path = task_run.path
    assert path.exists()

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.delete(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/{task_run.id}"
        )

    assert response.status_code == 200
    assert not path.exists()


@pytest.mark.asyncio
async def test_bulk_delete_leaf_cascades_full_chain(client, multiturn_task_run_setup):
    """POST /runs/delete with a leaf in a chain must cascade like single DELETE."""
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]
    root, mid, leaf = _build_three_turn_chain(task)

    assert root.path.exists()
    assert mid.path.exists()
    assert leaf.path.exists()

    ModelCache.shared().clear()

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/delete",
            json=[str(leaf.id)],
        )

    assert response.status_code == 200
    assert not root.path.exists()
    assert not mid.path.exists()
    assert not leaf.path.exists()


@pytest.mark.asyncio
async def test_bulk_delete_sibling_leaves_cascades_shared_root(
    client, multiturn_task_run_setup
):
    """Bulk-deleting both branches of a forked root sweeps the root up too.

    Single-delete of one branch would stop at the root (because the other
    branch keeps it alive). Bulk delete with both branches must see them as
    siblings of each other for the live-children check.
    """
    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]

    root = _make_task_run(
        task,
        input_text="turn 1",
        output_text="reply 1",
        trace=[_user_turn("turn 1"), _assistant_turn("reply 1")],
    )
    branch_a = _make_task_run(
        task,
        input_text="branch a",
        output_text="reply a",
        parent_task_run_id=root.id,
    )
    branch_b = _make_task_run(
        task,
        input_text="branch b",
        output_text="reply b",
        parent_task_run_id=root.id,
    )

    ModelCache.shared().clear()

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/delete",
            json=[str(branch_a.id), str(branch_b.id)],
        )

    assert response.status_code == 200
    assert not branch_a.path.exists()
    assert not branch_b.path.exists()
    assert not root.path.exists()


# --------------------------- Bulk upload tests ---------------------------


def _build_csv_bytes(rows: list[dict[str, str]]) -> bytes:
    """Serialize a list of dict rows to CSV bytes with the given header order.

    Used by both single-turn and multiturn upload tests.
    """

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


@pytest.mark.asyncio
async def test_bulk_upload_multiturn_success(client, multiturn_task_run_setup):
    """Uploading a multiturn CSV materializes one chain per row and reports both counts."""

    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]

    csv_bytes = _build_csv_bytes(
        [
            {
                "trace": json.dumps(
                    [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello"},
                        {"role": "user", "content": "What's 2+2?"},
                        {"role": "assistant", "content": "4"},
                    ]
                ),
                "tags": "",
            },
            {
                "trace": json.dumps(
                    [
                        {"role": "user", "content": "Hi again"},
                        {"role": "assistant", "content": "Hey!"},
                    ]
                ),
                "tags": "",
            },
        ]
    )

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/bulk_upload",
            files={"file": ("convs.csv", csv_bytes, "text/csv")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["filename"] == "convs.csv"
    # 2 conversations: 2 turns + 1 turn = 3 runs total
    assert body["imported_count"] == 3
    assert body["imported_conversation_count"] == 2


@pytest.mark.asyncio
async def test_bulk_upload_multiturn_invalid_trace_returns_422(
    client, multiturn_task_run_setup
):
    """A row with invalid JSON in `trace` surfaces as a 422 with a row-tagged detail."""

    project = multiturn_task_run_setup["project"]
    task = multiturn_task_run_setup["task"]

    csv_bytes = _build_csv_bytes(
        [
            {"trace": "not json", "tags": ""},
        ]
    )

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/bulk_upload",
            files={"file": ("bad.csv", csv_bytes, "text/csv")},
        )

    assert response.status_code == 422
    detail = response.json()["message"]
    assert "row 2" in detail
    assert "trace is not valid JSON" in detail


@pytest.mark.asyncio
async def test_bulk_upload_single_turn_response_has_null_conversation_count(
    client, task_run_setup
):
    """Existing single-turn upload still works and reports null conversation_count."""

    task = task_run_setup["task"]
    project = task_run_setup["project"]

    csv_bytes = _build_csv_bytes([{"input": "hi", "output": "hello", "tags": ""}])

    with patch("kiln_server.run_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/runs/bulk_upload",
            files={"file": ("single.csv", csv_bytes, "text/csv")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["imported_count"] == 1
    assert body["imported_conversation_count"] is None
