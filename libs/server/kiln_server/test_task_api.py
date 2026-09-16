from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from kiln_ai.datamodel import Project, Task, TaskRequirement
from kiln_ai.datamodel.external_tool_server import ToolServerType
from kiln_ai.utils.formatting import AGENT_TRUNCATION_SENTINEL

from kiln_server.custom_errors import connect_custom_errors
from kiln_server.task_api import connect_task_api, task_from_id


@pytest.fixture
def app():
    app = FastAPI()
    connect_task_api(app)
    connect_custom_errors(app)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def project_and_task(tmp_path):
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

    return project, task


def test_create_task_success(client, tmp_path):
    project_path = tmp_path / "test_project"
    project_path.mkdir()

    task_data = {
        "name": "Test Task 啊",
        "description": "This is a test task",
        "instruction": "This is a test instruction",
    }

    with (
        patch("kiln_server.task_api.project_from_id") as mock_project_from_id,
        patch("kiln_ai.datamodel.Task.save_to_file") as mock_save,
    ):
        mock_project_from_id.return_value = Project(
            name="Test Project", path=str(project_path)
        )
        mock_save.return_value = None

        response = client.post("/api/projects/project1-id/tasks", json=task_data)

    assert response.status_code == 200
    res = response.json()
    assert res["name"] == "Test Task 啊"
    assert res["description"] == "This is a test task"
    assert res["id"] is not None

    # Verify that project_from_id was called with the correct argument
    mock_project_from_id.assert_called_once_with("project1-id")


def test_create_task_project_not_found(client, tmp_path):
    task_data = {
        "name": "Test Task",
        "description": "This is a test task",
    }

    response = client.post("/api/projects/FAKEPROJECTID/tasks", json=task_data)

    assert response.status_code == 404
    assert response.json()["message"] == "Project not found. ID: FAKEPROJECTID"


def test_create_task_project_load_error(client, tmp_path):
    project_path = tmp_path / "test_project"
    project_path.mkdir()

    task_data = {
        "name": "Test Task",
        "description": "This is a test task",
    }

    with patch("kiln_server.task_api.project_from_id") as mock_load:
        mock_load.side_effect = HTTPException(
            status_code=404, detail="Project not found"
        )

    response = client.post("/api/projects/FAKEPROJECTID/tasks", json=task_data)

    assert response.status_code == 404
    assert "Project not found" in response.json()["message"]


def test_create_task_real_project(client, tmp_path):
    project_path = tmp_path / "real_project" / Project.base_filename()
    project_path.parent.mkdir()

    # Create a real Project
    project = Project(name="Real Project", path=str(project_path))
    project.save_to_file()

    task_data = {
        "name": "Real Task",
        "description": "This is a real task",
        "instruction": "Task instruction",
    }
    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = project

        response = client.post("/api/projects/project1-id/tasks", json=task_data)

        assert response.status_code == 200
        res = response.json()
        assert res["name"] == "Real Task"
        assert res["description"] == "This is a real task"
        assert res["instruction"] == "Task instruction"
        assert res["id"] is not None

        # Verify the task file on disk
        task_from_disk = project.tasks()[0]

        assert task_from_disk.name == "Real Task"
        assert task_from_disk.description == "This is a real task"
        assert task_from_disk.instruction == "Task instruction"
        assert task_from_disk.id == res["id"]

        # now post again, with an update
        update_data = {
            "description": "This is an updated task description",
        }
        response = client.patch(
            f"/api/projects/project1-id/tasks/{task_from_disk.id}",
            json=update_data,
        )
        assert response.status_code == 200
        res = response.json()
        assert res["description"] == "This is an updated task description"
        assert res["id"] == task_from_disk.id
        assert res["name"] == "Real Task"
        # Check disk
        task_from_disk_reloaded = project.tasks()[0]
        assert (
            task_from_disk_reloaded.description == "This is an updated task description"
        )
        assert task_from_disk_reloaded.id == task_from_disk.id
        assert task_from_disk_reloaded.instruction == "Task instruction"
        assert task_from_disk_reloaded.name == "Real Task"
        assert task_from_disk_reloaded.id == task_from_disk.id


def test_get_task_success(client, project_and_task):
    project, task = project_and_task

    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = project
        response = client.get(f"/api/projects/project1-id/tasks/{task.id}")

    assert response.status_code == 200
    res = response.json()
    assert res["name"] == "Test Task"
    assert res["description"] == "This is a test task"
    assert res["id"] == task.id
    assert res["instruction"] == "This is a test instruction"


def test_get_task_not_found(client, project_and_task):
    project, _ = project_and_task

    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = project
        response = client.get("/api/projects/project1-id/tasks/non_existent_task_id")

    assert response.status_code == 404
    assert response.json()["message"] == "Task not found. ID: non_existent_task_id"


def test_get_task_project_not_found(client):
    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.side_effect = HTTPException(
            status_code=404, detail="Project not found"
        )
        response = client.get("/api/projects/non_existent_project_id/tasks/task_id")

    assert response.status_code == 404
    assert "Project not found" in response.json()["message"]


def test_task_from_id_success(project_and_task):
    project, task = project_and_task

    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = project
        result = task_from_id("project1-id", task.id)

    assert isinstance(result, Task)
    assert result.id == task.id
    assert result.name == "Test Task"
    assert result.description == "This is a test task"


def test_task_from_id_not_found(project_and_task):
    project, _ = project_and_task

    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = project
        with pytest.raises(HTTPException) as exc_info:
            task_from_id("project1-id", "non_existent_task_id")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Task not found. ID: non_existent_task_id"


def test_update_task_input_schema_error(client, project_and_task):
    project, task = project_and_task

    update_data = {"input_json_schema": {"type": "object"}}

    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = project
        response = client.patch(
            f"/api/projects/{project.id}/tasks/{task.id}", json=update_data
        )

    assert response.status_code == 400
    assert (
        response.json()["message"] == "Input and output JSON schemas cannot be updated."
    )


def test_update_task_output_schema_error(client, project_and_task):
    project, task = project_and_task

    update_data = {"output_json_schema": {"type": "object"}}

    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = project
        response = client.patch(
            f"/api/projects/{project.id}/tasks/{task.id}", json=update_data
        )

    assert response.status_code == 400
    assert (
        response.json()["message"] == "Input and output JSON schemas cannot be updated."
    )


def test_update_task_id_mismatch_error(client, project_and_task):
    project, task = project_and_task

    update_data = {"id": "different_id"}

    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = project
        response = client.patch(
            f"/api/projects/{project.id}/tasks/{task.id}", json=update_data
        )

    assert response.status_code == 400
    assert (
        response.json()["message"] == "Task ID cannot be changed by client in a patch."
    )


def test_update_task_validation_error(client, project_and_task):
    project, task = project_and_task

    update_data = {"name": "Updated Task"}

    with (
        patch("kiln_server.task_api.project_from_id") as mock_project_from_id,
        patch(
            "kiln_server.task_api.Task.validate_and_save_with_subrelations"
        ) as mock_validate,
    ):
        mock_project_from_id.return_value = project
        mock_validate.return_value = None
        response = client.patch(
            f"/api/projects/{project.id}/tasks/{task.id}", json=update_data
        )

    assert response.status_code == 400
    assert response.json()["message"] == "Failed to update task."


def test_update_task_unexpected_return_type(client, project_and_task):
    project, task = project_and_task

    update_data = {"name": "Updated Task"}

    with (
        patch("kiln_server.task_api.project_from_id") as mock_project_from_id,
        patch(
            "kiln_server.task_api.Task.validate_and_save_with_subrelations"
        ) as mock_validate,
    ):
        mock_project_from_id.return_value = project
        mock_validate.return_value = MagicMock()  # Return a non-Task object
        response = client.patch(
            f"/api/projects/{project.id}/tasks/{task.id}", json=update_data
        )

    assert response.status_code == 500
    assert response.json()["message"] == "Failed to patch task."


def test_update_task_turn_mode_unchanged_succeeds(client, project_and_task):
    project, task = project_and_task

    update_data = {"turn_mode": "single_turn", "description": "Updated description"}

    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = project
        response = client.patch(
            f"/api/projects/{project.id}/tasks/{task.id}", json=update_data
        )

    assert response.status_code == 200
    res = response.json()
    assert res["turn_mode"] == "single_turn"
    assert res["description"] == "Updated description"


def test_update_task_turn_mode_change_rejected(client, project_and_task):
    project, task = project_and_task

    update_data = {"turn_mode": "multiturn"}

    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = project
        response = client.patch(
            f"/api/projects/{project.id}/tasks/{task.id}", json=update_data
        )

    assert response.status_code == 400
    assert (
        response.json()["message"] == "Task turn_mode cannot be changed after creation."
    )

    # Confirm on-disk state was not mutated by the rejected PATCH.
    reloaded = Task.from_id_and_parent_path(task.id, project.path)
    assert reloaded is not None
    assert reloaded.turn_mode.value == "single_turn"


def test_create_task_multiturn_with_structured_input_schema_rejected(client, tmp_path):
    project_path = tmp_path / "real_project" / Project.base_filename()
    project_path.parent.mkdir()

    project = Project(name="Real Project", path=str(project_path))
    project.save_to_file()

    task_data = {
        "name": "Bad Multiturn Task",
        "description": "Should fail validation",
        "instruction": "Task instruction",
        "turn_mode": "multiturn",
        "input_json_schema": (
            '{"type": "object", "properties": {"x": {"type": "string"}},'
            ' "required": ["x"]}'
        ),
    }

    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = project
        response = client.post(f"/api/projects/{project.id}/tasks", json=task_data)

    assert response.status_code == 422
    assert "structured input" in response.json()["message"].lower()


def test_create_task_multiturn_with_structured_output_schema_rejected(client, tmp_path):
    project_path = tmp_path / "real_project" / Project.base_filename()
    project_path.parent.mkdir()

    project = Project(name="Real Project", path=str(project_path))
    project.save_to_file()

    task_data = {
        "name": "Bad Multiturn Task",
        "description": "Should fail validation",
        "instruction": "Task instruction",
        "turn_mode": "multiturn",
        "output_json_schema": (
            '{"type": "object", "properties": {"y": {"type": "string"}},'
            ' "required": ["y"]}'
        ),
    }

    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = project
        response = client.post(f"/api/projects/{project.id}/tasks", json=task_data)

    assert response.status_code == 422
    assert "structured output" in response.json()["message"].lower()


def test_get_rating_options_empty_task(client, project_and_task):
    project, task = project_and_task

    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = project
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/rating_options"
        )

    assert response.status_code == 200
    res = response.json()
    assert res["options"] == []


def test_get_rating_options_with_requirements(client, project_and_task):
    project, task = project_and_task

    # Add a requirement to the task
    requirement = TaskRequirement(
        id="req1",
        name="Test Requirement",
        instruction="Test instruction",
        type="five_star",
    )
    task.requirements = [requirement]
    task.save_to_file()

    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = project
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/rating_options"
        )

    assert response.status_code == 200
    res = response.json()
    assert len(res["options"]) == 1
    option = res["options"][0]
    assert option["requirement"]["name"] == "Test Requirement"
    assert option["show_for_all"] is True
    assert option["show_for_tags"] == []


def test_get_rating_options_with_evals(client, project_and_task):
    project, task = project_and_task

    # Create a mock eval with output scores
    eval_mock = MagicMock()
    eval_mock.eval_configs_filter_id = "tag::golden_set"

    # Create score mocks with proper name attributes
    score1 = MagicMock()
    score1.name = "Score 1"
    score1.instruction = "Score 1 instruction"
    score1.type = "five_star"

    score2 = MagicMock()
    score2.name = "Overall Rating"
    score2.instruction = "Overall instruction"
    score2.type = "five_star"

    eval_mock.output_scores = [score1, score2]

    # Create secong mock eval with duplicate output scores
    eval_mock_2 = MagicMock()
    eval_mock_2.eval_configs_filter_id = "tag::golden_set"

    # Create score mocks with proper name attributes
    score3 = MagicMock()
    score3.name = "Score 1"
    score3.instruction = "Score 1 instruction"
    score3.type = "five_star"

    eval_mock_2.output_scores = [score3]

    with (
        patch("kiln_server.task_api.project_from_id") as mock_project_from_id,
        patch("kiln_ai.datamodel.Task.evals") as mock_evals,
    ):
        mock_project_from_id.return_value = project
        mock_evals.return_value = [eval_mock, eval_mock_2]

        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/rating_options"
        )

    assert response.status_code == 200
    res = response.json()
    assert len(res["options"]) == 1  # Only Score 1, Overall Rating is skipped
    option = res["options"][0]
    assert option["requirement"]["name"] == "Score 1"
    assert option["show_for_all"] is False
    # Note: we're checking it's not added twice with the dupe
    assert option["show_for_tags"] == ["golden_set"]


def test_get_rating_options_with_non_tag_filter(client, project_and_task, caplog):
    project, task = project_and_task

    # Create a mock eval with non-tag filter
    eval_mock = MagicMock()
    eval_mock.id = "test_eval"
    eval_mock.eval_configs_filter_id = "filter::some_filter"

    # Create score mock with proper attributes
    score = MagicMock()
    score.name = "Score 1"
    score.instruction = "Score 1 instruction"
    score.type = "five_star"

    eval_mock.output_scores = [score]

    with (
        patch("kiln_server.task_api.project_from_id") as mock_project_from_id,
        patch("kiln_ai.datamodel.Task.evals") as mock_evals,
    ):
        mock_project_from_id.return_value = project
        mock_evals.return_value = [eval_mock]

        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/rating_options"
        )

    assert response.status_code == 200
    res = response.json()
    assert res["options"] == []  # No options should be added for non-tag filter

    # Verify warning was logged
    assert "non-tag filter" in caplog.text
    assert "test_eval" in caplog.text


def test_get_rating_options_duplicate_requirements(client, project_and_task):
    project, task = project_and_task

    # Create two evals with the same requirement name
    eval1 = MagicMock()
    eval1.eval_configs_filter_id = "tag::golden_set1"

    score1 = MagicMock()
    score1.name = "Duplicate Score"
    score1.instruction = "Score 1 instruction"
    score1.type = "five_star"
    eval1.output_scores = [score1]

    eval2 = MagicMock()
    eval2.eval_configs_filter_id = "tag::golden_set2"

    score2 = MagicMock()
    score2.name = "Duplicate Score"  # Same name as eval1
    score2.instruction = "Score 2 instruction"
    score2.type = "five_star"
    eval2.output_scores = [score2]

    with (
        patch("kiln_server.task_api.project_from_id") as mock_project_from_id,
        patch("kiln_ai.datamodel.Task.evals") as mock_evals,
    ):
        mock_project_from_id.return_value = project
        mock_evals.return_value = [eval1, eval2]

        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/rating_options"
        )

    assert response.status_code == 200
    res = response.json()
    assert len(res["options"]) == 1  # Should be merged into one option
    option = res["options"][0]
    assert option["requirement"]["name"] == "Duplicate Score"
    assert option["show_for_all"] is False
    assert set(option["show_for_tags"]) == {"golden_set1", "golden_set2"}


def test_delete_task_success(client, project_and_task):
    project, task = project_and_task

    with patch("kiln_server.task_api.project_from_id") as mock_project_from_id:
        mock_project_from_id.return_value = project

        # First verify the task exists
        response = client.get(f"/api/projects/{project.id}/tasks/{task.id}")
        assert response.status_code == 200

        # Delete the task
        response = client.delete(f"/api/projects/{project.id}/tasks/{task.id}")
        assert response.status_code == 200

        # Verify the task was deleted
        response = client.get(f"/api/projects/{project.id}/tasks/{task.id}")
        assert response.status_code == 404
        assert response.json()["message"] == f"Task not found. ID: {task.id}"


def test_delete_task_archives_kiln_task_tools(client, project_and_task):
    """Test that deleting a task archives associated kiln_task tool servers."""
    project, task = project_and_task

    # Create mock kiln_task tool servers
    kiln_task_tool_1 = MagicMock()
    kiln_task_tool_1.type = ToolServerType.kiln_task
    kiln_task_tool_1.properties = {
        "task_id": task.id,
        "run_config_id": "config1",
        "name": "task_tool_1",
        "description": "First task tool",
        "is_archived": False,
    }
    kiln_task_tool_1.save_to_file = MagicMock()

    # Create a kiln_task tool for a different task (should not be archived)
    kiln_task_tool_2 = MagicMock()
    kiln_task_tool_2.type = ToolServerType.kiln_task
    kiln_task_tool_2.properties = {
        "task_id": "different_task_id",
        "run_config_id": "config2",
        "name": "task_tool_2",
        "description": "Second task tool",
        "is_archived": False,
    }
    kiln_task_tool_2.save_to_file = MagicMock()

    # Create a non-kiln_task tool (should not be affected)
    remote_mcp_tool = MagicMock()
    remote_mcp_tool.type = ToolServerType.remote_mcp
    remote_mcp_tool.properties = {
        "server_url": "https://example.com",
        "headers": {"Authorization": "Bearer token"},
    }
    remote_mcp_tool.save_to_file = MagicMock()

    with (
        patch("kiln_server.task_api.project_from_id") as mock_project_from_id,
        patch("kiln_server.task_api.task_from_id") as mock_task_from_id,
        patch(
            "kiln_ai.datamodel.project.Project.external_tool_servers",
            return_value=[kiln_task_tool_1, kiln_task_tool_2, remote_mcp_tool],
        ),
        patch("kiln_ai.datamodel.task.Task.parent_project", return_value=project),
    ):
        mock_project_from_id.return_value = project
        mock_task_from_id.return_value = task

        # Delete the task
        response = client.delete(f"/api/projects/{project.id}/tasks/{task.id}")

    assert response.status_code == 200

    # Verify that the matching kiln_task tool was archived
    assert kiln_task_tool_1.properties["is_archived"] is True
    kiln_task_tool_1.save_to_file.assert_called_once()

    # Verify that the non-matching kiln_task tool was not affected
    assert kiln_task_tool_2.properties["is_archived"] is False
    kiln_task_tool_2.save_to_file.assert_not_called()

    # Verify that the remote_mcp tool was not affected
    remote_mcp_tool.save_to_file.assert_not_called()


def test_delete_task_no_matching_kiln_task_tools(client, project_and_task):
    """Test that deleting a task works when no matching kiln_task tools exist."""
    project, task = project_and_task

    # Create a kiln_task tool for a different task
    kiln_task_tool = MagicMock()
    kiln_task_tool.type = ToolServerType.kiln_task
    kiln_task_tool.properties = {
        "task_id": "different_task_id",
        "run_config_id": "config1",
        "name": "task_tool",
        "description": "Task tool for different task",
        "is_archived": False,
    }
    kiln_task_tool.save_to_file = MagicMock()

    with (
        patch("kiln_server.task_api.project_from_id") as mock_project_from_id,
        patch("kiln_server.task_api.task_from_id") as mock_task_from_id,
        patch(
            "kiln_ai.datamodel.project.Project.external_tool_servers",
            return_value=[kiln_task_tool],
        ),
        patch("kiln_ai.datamodel.task.Task.parent_project", return_value=project),
    ):
        mock_project_from_id.return_value = project
        mock_task_from_id.return_value = task

        # Delete the task
        response = client.delete(f"/api/projects/{project.id}/tasks/{task.id}")

    assert response.status_code == 200

    # Verify that no tools were modified
    assert kiln_task_tool.properties["is_archived"] is False
    kiln_task_tool.save_to_file.assert_not_called()


def test_delete_task_no_external_tool_servers(client, project_and_task):
    """Test that deleting a task works when project has no external tool servers."""
    project, task = project_and_task

    with (
        patch("kiln_server.task_api.project_from_id") as mock_project_from_id,
        patch("kiln_server.task_api.task_from_id") as mock_task_from_id,
        patch(
            "kiln_ai.datamodel.project.Project.external_tool_servers",
            return_value=[],
        ),
        patch("kiln_ai.datamodel.task.Task.parent_project", return_value=project),
    ):
        mock_project_from_id.return_value = project
        mock_task_from_id.return_value = task

        # Delete the task
        response = client.delete(f"/api/projects/{project.id}/tasks/{task.id}")

    assert response.status_code == 200


def test_delete_task_archive_error_handling(client, project_and_task):
    """Test error handling when kiln_task tool properties are missing is_archived field."""
    project, task = project_and_task

    # Create a kiln_task tool without is_archived field
    kiln_task_tool = MagicMock()
    kiln_task_tool.type = ToolServerType.kiln_task
    kiln_task_tool.properties = {
        "task_id": task.id,
        "run_config_id": "config1",
        "name": "task_tool",
        "description": "Task tool without is_archived field",
        # Missing "is_archived" field
    }
    kiln_task_tool.save_to_file = MagicMock()

    with (
        patch("kiln_server.task_api.project_from_id") as mock_project_from_id,
        patch("kiln_server.task_api.task_from_id") as mock_task_from_id,
        patch(
            "kiln_ai.datamodel.project.Project.external_tool_servers",
            return_value=[kiln_task_tool],
        ),
        patch("kiln_ai.datamodel.task.Task.parent_project", return_value=project),
    ):
        mock_project_from_id.return_value = project
        mock_task_from_id.return_value = task

        # Delete the task - should raise TypeError
        with pytest.raises(TypeError, match="Expected archiveable tool task server"):
            client.delete(f"/api/projects/{project.id}/tasks/{task.id}")


def test_delete_task_parent_project_none(client, project_and_task):
    """Test that deleting a task works when parent_project() returns None."""
    project, task = project_and_task

    with (
        patch("kiln_server.task_api.project_from_id") as mock_project_from_id,
        patch("kiln_server.task_api.task_from_id") as mock_task_from_id,
        patch("kiln_ai.datamodel.task.Task.parent_project", return_value=None),
    ):
        mock_project_from_id.return_value = project
        mock_task_from_id.return_value = task

        # Delete the task
        response = client.delete(f"/api/projects/{project.id}/tasks/{task.id}")

    assert response.status_code == 200


# --- task_summaries endpoint tests ---


def test_task_summaries_happy_path(client, tmp_path):
    p1_path = tmp_path / "proj1" / "project.kiln"
    p1_path.parent.mkdir()
    p1 = Project(name="Project One", path=str(p1_path))
    p1.save_to_file()

    t1 = Task(name="Task A", instruction="Do A", parent=p1)
    t1.save_to_file()
    t2 = Task(name="Task B", instruction="Do B", description="desc B", parent=p1)
    t2.save_to_file()

    p2_path = tmp_path / "proj2" / "project.kiln"
    p2_path.parent.mkdir()
    p2 = Project(name="Project Two", description="Second project", path=str(p2_path))
    p2.save_to_file()

    t3 = Task(name="Task C", instruction="Do C", parent=p2)
    t3.save_to_file()
    t4 = Task(name="Task D", instruction="Do D", description="desc D", parent=p2)
    t4.save_to_file()

    with patch("kiln_server.task_api.Config.shared") as mock_config:
        mock_config.return_value.projects = [str(p1_path), str(p2_path)]
        response = client.get("/api/task_summaries")

    assert response.status_code == 200
    data = response.json()
    assert len(data["projects"]) == 2

    proj1 = data["projects"][0]
    assert proj1["name"] == "Project One"
    assert proj1["id"] == p1.id
    assert proj1["description"] is None
    assert "created_at" not in proj1
    assert len(proj1["tasks"]) == 2

    for task_entry in proj1["tasks"]:
        assert "created_at" not in task_entry
        assert "instruction_truncated" not in task_entry

    task_names = {t["name"] for t in proj1["tasks"]}
    assert task_names == {"Task A", "Task B"}

    proj2 = data["projects"][1]
    assert proj2["name"] == "Project Two"
    assert proj2["description"] == "Second project"
    assert "created_at" not in proj2
    assert len(proj2["tasks"]) == 2

    task_names_p2 = {t["name"] for t in proj2["tasks"]}
    assert task_names_p2 == {"Task C", "Task D"}


def test_task_summaries_empty_workspace(client):
    with patch("kiln_server.task_api.Config.shared") as mock_config:
        mock_config.return_value.projects = []
        response = client.get("/api/task_summaries")

    assert response.status_code == 200
    assert response.json() == {"projects": []}


def test_task_summaries_none_projects(client):
    with patch("kiln_server.task_api.Config.shared") as mock_config:
        mock_config.return_value.projects = None
        response = client.get("/api/task_summaries")

    assert response.status_code == 200
    assert response.json() == {"projects": []}


def test_task_summaries_instruction_truncation_over_limit(client, tmp_path):
    p_path = tmp_path / "proj" / "project.kiln"
    p_path.parent.mkdir()
    p = Project(name="Proj", path=str(p_path))
    p.save_to_file()

    long_instruction = " ".join(f"word{i}" for i in range(150))
    t = Task(name="Long Task", instruction=long_instruction, parent=p)
    t.save_to_file()

    with patch("kiln_server.task_api.Config.shared") as mock_config:
        mock_config.return_value.projects = [str(p_path)]
        response = client.get("/api/task_summaries")

    assert response.status_code == 200
    task_data = response.json()["projects"][0]["tasks"][0]
    assert "instruction_truncated" not in task_data
    assert task_data["instruction"].endswith(AGENT_TRUNCATION_SENTINEL)
    prefix = task_data["instruction"].removesuffix(AGENT_TRUNCATION_SENTINEL).rstrip()
    assert len(prefix.split()) == 100


def test_task_summaries_instruction_at_limit(client, tmp_path):
    p_path = tmp_path / "proj" / "project.kiln"
    p_path.parent.mkdir()
    p = Project(name="Proj", path=str(p_path))
    p.save_to_file()

    exact_instruction = " ".join(f"word{i}" for i in range(100))
    t = Task(name="Exact Task", instruction=exact_instruction, parent=p)
    t.save_to_file()

    with patch("kiln_server.task_api.Config.shared") as mock_config:
        mock_config.return_value.projects = [str(p_path)]
        response = client.get("/api/task_summaries")

    assert response.status_code == 200
    task_data = response.json()["projects"][0]["tasks"][0]
    assert task_data["instruction"] == exact_instruction
    assert AGENT_TRUNCATION_SENTINEL not in task_data["instruction"]


def test_task_summaries_instruction_under_limit(client, tmp_path):
    p_path = tmp_path / "proj" / "project.kiln"
    p_path.parent.mkdir()
    p = Project(name="Proj", path=str(p_path))
    p.save_to_file()

    t = Task(name="Short Task", instruction="Short instruction", parent=p)
    t.save_to_file()

    with patch("kiln_server.task_api.Config.shared") as mock_config:
        mock_config.return_value.projects = [str(p_path)]
        response = client.get("/api/task_summaries")

    assert response.status_code == 200
    task_data = response.json()["projects"][0]["tasks"][0]
    assert task_data["instruction"] == "Short instruction"
    assert AGENT_TRUNCATION_SENTINEL not in task_data["instruction"]


def test_task_summaries_null_description(client, tmp_path):
    p_path = tmp_path / "proj" / "project.kiln"
    p_path.parent.mkdir()
    p = Project(name="Proj", path=str(p_path))
    p.save_to_file()

    t = Task(name="No Desc Task", instruction="Do something", parent=p)
    t.save_to_file()

    with patch("kiln_server.task_api.Config.shared") as mock_config:
        mock_config.return_value.projects = [str(p_path)]
        response = client.get("/api/task_summaries")

    assert response.status_code == 200
    proj = response.json()["projects"][0]
    assert proj["description"] is None
    assert proj["tasks"][0]["description"] is None


def test_task_summaries_skips_corrupt_project(client, tmp_path):
    p_path = tmp_path / "good_proj" / "project.kiln"
    p_path.parent.mkdir()
    p = Project(name="Good", path=str(p_path))
    p.save_to_file()

    t = Task(name="Task", instruction="Do it", parent=p)
    t.save_to_file()

    bad_path = str(tmp_path / "nonexistent" / "project.kiln")

    with patch("kiln_server.task_api.Config.shared") as mock_config:
        mock_config.return_value.projects = [bad_path, str(p_path)]
        response = client.get("/api/task_summaries")

    assert response.status_code == 200
    data = response.json()
    assert len(data["projects"]) == 1
    assert data["projects"][0]["name"] == "Good"


def test_old_all_tasks_path_returns_404(client):
    response = client.get("/api/all_tasks")
    assert response.status_code == 404
