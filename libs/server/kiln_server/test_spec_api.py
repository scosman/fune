import json
import logging
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from kiln_ai.datamodel import Project, Task
from kiln_ai.datamodel.datamodel_enums import Priority
from kiln_ai.datamodel.eval import (
    Eval,
    EvalOutputScore,
    TaskOutputRatingType,
    TaskRunSplit,
)
from kiln_ai.datamodel.spec import Spec, SpecStatus, TaskSample
from kiln_ai.datamodel.spec_properties import (
    DesiredBehaviourProperties,
    HallucinationsProperties,
    SpecType,
    ToneProperties,
    ToxicityProperties,
)

from kiln_server.custom_errors import connect_custom_errors
from kiln_server.spec_api import connect_spec_api, resolve_available_spec_name


@pytest.fixture
def app():
    app = FastAPI()
    connect_spec_api(app)
    connect_custom_errors(app)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def project_and_task(tmp_path):
    project_path = tmp_path / "test_project" / "project.kiln"
    project_path.parent.mkdir()

    project = Project(name="Test Project", path=project_path)
    project.save_to_file()
    task = Task(
        name="Test Task",
        instruction="This is a test instruction",
        description="This is a test task",
        parent=project,
    )
    task.save_to_file()

    return project, task


def create_task_sample_dict():
    """Helper to create a valid task sample for API tests."""
    return {
        "input": "Example input",
        "output": "Example output",
    }


def create_tone_properties_dict():
    """Helper to create valid tone properties for API tests."""
    return {
        "spec_type": SpecType.tone.value,
        "core_requirement": "Test instruction",
        "tone_description": "Professional and friendly",
    }


def create_toxicity_properties_dict():
    """Helper to create valid toxicity properties for API tests."""
    return {
        "spec_type": SpecType.toxicity.value,
        "core_requirement": "Test instruction",
        "toxicity_examples": "Example toxicity content",
    }


def create_reference_answer_accuracy_properties_dict():
    """Helper to create valid reference answer accuracy properties for API tests."""
    return {
        "spec_type": SpecType.reference_answer_accuracy.value,
        "core_requirement": "Test instruction",
        "reference_answer_accuracy_description": "Must match reference",
        "accurate_examples": "Accurate example",
        "inaccurate_examples": "Inaccurate example",
    }


@pytest.fixture
def sample_tone_properties():
    """Fixture for creating complete ToneProperties objects for direct Spec creation."""
    return ToneProperties(
        spec_type=SpecType.tone,
        core_requirement="Test instruction",
        tone_description="Professional and friendly",
    )


@pytest.fixture
def sample_hallucinations_properties():
    """Fixture for creating complete HallucinationsProperties objects."""
    return HallucinationsProperties(
        spec_type=SpecType.hallucinations,
        core_requirement="Test instruction",
        hallucinations_examples="Example hallucination",
    )


@pytest.fixture
def sample_toxicity_properties():
    """Fixture for creating complete ToxicityProperties objects."""
    return ToxicityProperties(
        spec_type=SpecType.toxicity,
        core_requirement="Test instruction",
        toxicity_examples="Example toxicity",
    )


@pytest.fixture
def sample_desired_behaviour_properties():
    """Fixture for creating complete DesiredBehaviourProperties objects."""
    return DesiredBehaviourProperties(
        spec_type=SpecType.desired_behaviour,
        core_requirement="Test instruction",
        desired_behaviour_description="Avoid toxic content",
        correct_behaviour_examples=None,
        incorrect_behaviour_examples=None,
    )


def test_create_spec_success(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "name": "Test Spec",
        "definition": "The system should always respond politely",
        "priority": Priority.p1,
        "status": SpecStatus.active.value,
        "tags": ["test", "important"],
        "properties": create_tone_properties_dict(),
        "task_sample": create_task_sample_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 200
    res = response.json()
    assert res["name"] == "Test Spec"
    assert res["definition"] == "The system should always respond politely"
    assert res["properties"]["spec_type"] == SpecType.tone.value
    assert res["priority"] == 1
    assert res["status"] == "active"
    assert res["tags"] == ["test", "important"]
    # Verify eval_id was auto-generated
    assert res["eval_id"] is not None
    assert len(res["eval_id"]) > 0

    # Check that the spec was saved to the task/file
    specs = task.specs()
    assert len(specs) == 1
    assert specs[0].name == "Test Spec"
    assert specs[0].definition == "The system should always respond politely"
    assert specs[0].properties["spec_type"] == SpecType.tone
    assert specs[0].priority == Priority.p1
    assert specs[0].status == SpecStatus.active

    # Check that an eval was created with correct filter IDs
    evals = task.evals()
    assert len(evals) == 1
    assert evals[0].name == "Test Spec"
    assert evals[0].id == res["eval_id"]
    assert evals[0].eval_configs_filter_id == "tag::golden_test_spec"
    assert evals[0].splits == {
        "test": TaskRunSplit(filter_id="tag::test_test_spec"),
        "train": TaskRunSplit(filter_id="tag::train_test_spec"),
        "val": TaskRunSplit(filter_id="tag::val_test_spec"),
    }

    # Check the raw saved file, not the loaded model: what reaches the bytes is
    # invisible in eval.splits. All three splits go to `splits`, and the deprecated flat
    # filter fields are written null rather than left for an older build to read.
    saved_eval = json.loads(evals[0].path.read_text())
    assert saved_eval["eval_set_filter_id"] is None
    assert saved_eval["train_set_filter_id"] is None
    assert saved_eval["splits"] == {
        "test": {"source": "task_run", "filter_id": "tag::test_test_spec"},
        "train": {"source": "task_run", "filter_id": "tag::train_test_spec"},
        "val": {"source": "task_run", "filter_id": "tag::val_test_spec"},
    }


def test_create_spec_minimal(client, project_and_task):
    """Test creating a spec with minimal required fields (name, definition, properties only)."""
    project, task = project_and_task

    spec_data = {
        "name": "Minimal Spec",
        "definition": "No toxic content allowed",
        "properties": create_toxicity_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 200
    res = response.json()
    assert res["name"] == "Minimal Spec"
    assert res["definition"] == "No toxic content allowed"
    assert res["properties"]["spec_type"] == SpecType.toxicity.value
    # Check defaults were applied
    assert res["priority"] == 1  # Priority.p1
    assert res["status"] == "active"
    assert res["tags"] == []
    # Eval should be auto-created
    assert res["eval_id"] is not None


def test_create_spec_task_not_found(client):
    spec_data = {
        "name": "Test Spec",
        "definition": "System should behave correctly",
        "properties": create_tone_properties_dict(),
    }

    response = client.post(
        "/api/projects/project-id/tasks/fake-task-id/specs", json=spec_data
    )
    assert response.status_code == 404


def test_get_specs_success(
    client, project_and_task, sample_tone_properties, sample_toxicity_properties
):
    project, task = project_and_task

    spec1 = Spec(
        name="Spec 1",
        definition="System should respond appropriately",
        properties=sample_tone_properties,
        eval_id="test_eval_id_1",
        parent=task,
    )
    spec1.save_to_file()

    spec2 = Spec(
        name="Spec 2",
        definition="No toxic responses",
        priority=Priority.p3,
        properties=sample_toxicity_properties,
        eval_id="test_eval_id_2",
        parent=task,
    )
    spec2.save_to_file()

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(f"/api/projects/{project.id}/tasks/{task.id}/specs")

    assert response.status_code == 200
    res = response.json()
    assert isinstance(res, list)
    assert len(res) == 2
    spec_names = {spec["name"] for spec in res}
    assert spec_names == {"Spec 1", "Spec 2"}


def test_get_specs_empty(client, project_and_task):
    project, task = project_and_task

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(f"/api/projects/{project.id}/tasks/{task.id}/specs")

    assert response.status_code == 200
    res = response.json()
    assert isinstance(res, list)
    assert len(res) == 0


def test_get_specs_task_not_found(client):
    response = client.get("/api/projects/project-id/tasks/fake-task-id/specs")
    assert response.status_code == 404


def test_get_spec_success(client, project_and_task, sample_hallucinations_properties):
    project, task = project_and_task

    spec = Spec(
        name="Test Spec",
        definition="System should not hallucinate facts",
        priority=Priority.p2,
        status=SpecStatus.active,
        tags=["validation", "safety"],
        properties=sample_hallucinations_properties,
        eval_id="test_eval_id",
        parent=task,
    )
    spec.save_to_file()

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/{spec.id}"
        )

    assert response.status_code == 200
    res = response.json()
    assert res["name"] == "Test Spec"
    assert res["definition"] == "System should not hallucinate facts"
    assert res["properties"]["spec_type"] == SpecType.hallucinations.value
    assert res["priority"] == 2
    assert res["status"] == "active"
    assert res["tags"] == ["validation", "safety"]


def test_get_spec_not_found(client, project_and_task):
    project, task = project_and_task

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/nonexistent_id"
        )

    assert response.status_code == 404
    assert "Spec not found" in response.json()["message"]


def test_update_spec_success(client, project_and_task, sample_tone_properties):
    project, task = project_and_task

    spec = Spec(
        name="Original Name",
        definition="Original definition",
        priority=Priority.p3,
        status=SpecStatus.active,
        tags=["old_tag"],
        properties=sample_tone_properties,
        eval_id="test_eval_id",
        parent=task,
    )
    spec.save_to_file()

    update_data = {
        "name": "Updated Name",
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.patch(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/{spec.id}",
            json=update_data,
        )

    assert response.status_code == 200
    res = response.json()
    assert res["name"] == "Updated Name"
    # Verify other fields remain unchanged
    assert res["definition"] == "Original definition"
    assert res["priority"] == 3
    assert res["status"] == "active"
    assert res["tags"] == ["old_tag"]
    assert res["properties"]["spec_type"] == SpecType.tone.value

    # Verify the spec was updated in the task/file
    updated_spec = next((s for s in task.specs() if s.id == spec.id), None)
    assert updated_spec is not None
    assert updated_spec.name == "Updated Name"
    # Verify other fields remain unchanged
    assert updated_spec.definition == "Original definition"
    assert updated_spec.priority == Priority.p3
    assert updated_spec.status == SpecStatus.active
    assert updated_spec.tags == ["old_tag"]


def test_update_spec_with_existing_eval_id(
    client, project_and_task, sample_toxicity_properties
):
    """Test that updating a spec's name doesn't affect its eval_id."""
    project, task = project_and_task

    spec = Spec(
        name="Original Name",
        definition="Original definition",
        priority=Priority.p2,
        status=SpecStatus.active,
        tags=["old_tag"],
        eval_id="original_eval_id",
        properties=sample_toxicity_properties,
        parent=task,
    )
    spec.save_to_file()

    update_data = {
        "name": "Updated Name",
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.patch(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/{spec.id}",
            json=update_data,
        )

    assert response.status_code == 200
    res = response.json()
    assert res["name"] == "Updated Name"
    # Verify other fields including eval_id remain unchanged
    assert res["definition"] == "Original definition"
    assert res["properties"]["spec_type"] == SpecType.toxicity.value
    assert res["priority"] == 2
    assert res["status"] == "active"
    assert res["eval_id"] == "original_eval_id"


def test_update_spec_name_syncs_eval_name(
    client: TestClient,
    project_and_task: tuple[Project, Task],
    sample_tone_properties: ToneProperties,
) -> None:
    """Test that updating a spec's name also updates the associated eval's name."""
    project, task = project_and_task

    eval = Eval(
        name="Original Name",
        description="Test eval",
        template="rag",
        eval_set_filter_id="tag::test_eval",
        output_scores=[
            EvalOutputScore(
                name="Quality",
                type=TaskOutputRatingType.five_star,
            )
        ],
        parent=task,
    )
    eval.save_to_file()

    spec = Spec(
        name="Original Name",
        definition="Original definition",
        priority=Priority.p2,
        status=SpecStatus.active,
        tags=[],
        eval_id=eval.id,
        properties=sample_tone_properties,
        parent=task,
    )
    spec.save_to_file()

    update_data = {"name": "Updated Name"}

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.patch(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/{spec.id}",
            json=update_data,
        )

    assert response.status_code == 200
    assert response.json()["name"] == "Updated Name"

    # Verify the eval name was also updated
    updated_eval = Eval.from_id_and_parent_path(eval.id, task.path)
    assert updated_eval is not None
    assert updated_eval.name == "Updated Name"


def test_update_spec_name_rollback_eval_on_spec_save_failure(
    client: TestClient,
    project_and_task: tuple[Project, Task],
    sample_tone_properties: ToneProperties,
) -> None:
    """Test that eval name is rolled back if spec.save_to_file() fails."""
    project, task = project_and_task

    eval = Eval(
        name="Original Name",
        description="Test eval",
        template="rag",
        eval_set_filter_id="tag::test_eval",
        output_scores=[
            EvalOutputScore(
                name="Quality",
                type=TaskOutputRatingType.five_star,
            )
        ],
        parent=task,
    )
    eval.save_to_file()

    spec = Spec(
        name="Original Name",
        definition="Original definition",
        priority=Priority.p2,
        status=SpecStatus.active,
        tags=[],
        eval_id=eval.id,
        properties=sample_tone_properties,
        parent=task,
    )
    spec.save_to_file()

    update_data = {"name": "Updated Name"}

    def save_that_fails(self: Spec) -> None:
        raise Exception("disk error")

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        # Patch save_to_file after spec_from_id resolves so the lookup works
        with patch.object(Spec, "save_to_file", save_that_fails):
            with pytest.raises(Exception, match="disk error"):
                client.patch(
                    f"/api/projects/{project.id}/tasks/{task.id}/specs/{spec.id}",
                    json=update_data,
                )

    # Verify the eval name was rolled back
    rolled_back_eval = Eval.from_id_and_parent_path(eval.id, task.path)
    assert rolled_back_eval is not None
    assert rolled_back_eval.name == "Original Name"


def test_update_spec_tags_only(client, project_and_task, sample_tone_properties):
    """Test updating only tags field (save_tags use case)."""
    project, task = project_and_task

    spec = Spec(
        name="Original Name",
        definition="Original definition",
        priority=Priority.p3,
        status=SpecStatus.active,
        tags=["old_tag"],
        eval_id="original_eval_id",
        properties=sample_tone_properties,
        parent=task,
    )
    spec.save_to_file()

    # Simulate save_tags function sending only tags
    update_data = {
        "tags": ["new_tag", "updated_tag"],
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.patch(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/{spec.id}",
            json=update_data,
        )

    assert response.status_code == 200
    res = response.json()
    assert res["tags"] == ["new_tag", "updated_tag"]
    # Verify other fields remain unchanged
    assert res["name"] == "Original Name"
    assert res["definition"] == "Original definition"
    assert res["priority"] == 3
    assert res["status"] == "active"
    assert res["eval_id"] == "original_eval_id"

    # Verify the change persisted
    updated_spec = next((s for s in task.specs() if s.id == spec.id), None)
    assert updated_spec is not None
    assert updated_spec.tags == ["new_tag", "updated_tag"]


def test_update_spec_status_only(client, project_and_task, sample_tone_properties):
    """Test updating only status field (archive use case)."""
    project, task = project_and_task

    spec = Spec(
        name="Test Spec",
        definition="Test definition",
        priority=Priority.p2,
        status=SpecStatus.active,
        tags=["test"],
        properties=sample_tone_properties,
        eval_id="test_eval_id",
        parent=task,
    )
    spec.save_to_file()

    # Update only status to archived
    update_data = {
        "status": SpecStatus.archived.value,
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.patch(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/{spec.id}",
            json=update_data,
        )

    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "archived"
    # Verify other fields remain unchanged
    assert res["name"] == "Test Spec"
    assert res["definition"] == "Test definition"
    assert res["priority"] == 2
    assert res["tags"] == ["test"]

    # Verify the change persisted
    updated_spec = next((s for s in task.specs() if s.id == spec.id), None)
    assert updated_spec is not None
    assert updated_spec.status == SpecStatus.archived


def test_update_spec_not_found(client, project_and_task):
    project, task = project_and_task

    update_data = {
        "name": "Updated Name",
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.patch(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/nonexistent_id",
            json=update_data,
        )

    assert response.status_code == 404
    assert "Spec not found" in response.json()["message"]


def test_create_spec_creates_eval_with_correct_template(client, project_and_task):
    """Test that creating a spec auto-creates an eval with the correct template for the spec type."""
    project, task = project_and_task

    spec_data = {
        "name": "Reference Answer Spec",
        "definition": "Answers must match reference answers",
        "properties": create_reference_answer_accuracy_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 200
    res = response.json()
    assert res["eval_id"] is not None

    # Verify eval was created with correct properties
    evals = task.evals()
    assert len(evals) == 1
    assert evals[0].name == "Reference Answer Spec"
    # Reference answer spec should use 'rag' template
    assert evals[0].template == "rag"

    specs = task.specs()
    assert len(specs) == 1
    assert specs[0].eval_id == evals[0].id


def test_create_spec_with_properties(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "name": "Desired Behaviour Spec",
        "definition": "System should avoid toxic language",
        "properties": {
            "spec_type": SpecType.desired_behaviour.value,
            "core_requirement": "Test instruction",
            "desired_behaviour_description": "Avoid toxic language and offensive content",
            "incorrect_behaviour_examples": "Example 1: Don't use slurs\nExample 2: Don't be rude",
        },
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 200
    res = response.json()
    assert res["properties"] is not None
    assert res["properties"]["spec_type"] == SpecType.desired_behaviour.value
    assert (
        res["properties"]["desired_behaviour_description"]
        == "Avoid toxic language and offensive content"
    )
    assert (
        res["properties"]["incorrect_behaviour_examples"]
        == "Example 1: Don't use slurs\nExample 2: Don't be rude"
    )

    specs = task.specs()
    assert len(specs) == 1
    assert specs[0].properties is not None
    assert specs[0].properties["spec_type"] == SpecType.desired_behaviour


def test_create_spec_with_archived_status(client, project_and_task):
    """Test creating a spec with archived status."""
    project, task = project_and_task

    spec_data = {
        "name": "Archived Spec",
        "definition": "This spec is archived",
        "status": SpecStatus.archived.value,
        "properties": create_tone_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 200
    res = response.json()
    assert res["name"] == "Archived Spec"
    assert res["status"] == "archived"

    specs = task.specs()
    assert len(specs) == 1
    assert specs[0].status == SpecStatus.archived


def test_get_spec_with_archived_status(
    client, project_and_task, sample_tone_properties
):
    """Test getting a spec with archived status."""
    project, task = project_and_task

    spec = Spec(
        name="Archived Spec",
        definition="This spec is archived",
        status=SpecStatus.archived,
        properties=sample_tone_properties,
        eval_id="test_eval_id",
        parent=task,
    )
    spec.save_to_file()

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/{spec.id}"
        )

    assert response.status_code == 200
    res = response.json()
    assert res["name"] == "Archived Spec"
    assert res["status"] == "archived"


# Validation error tests (422 responses)


def test_create_spec_missing_name(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "definition": "The system should always respond politely",
        "properties": create_tone_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 422
    res = response.json()
    assert "source_errors" in res
    assert any(
        error["loc"] == ["body", "name"] and error["type"] == "missing"
        for error in res["source_errors"]
    )


def test_create_spec_missing_definition(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "name": "Test Spec",
        "properties": create_tone_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 422
    res = response.json()
    assert "source_errors" in res
    assert any(
        error["loc"] == ["body", "definition"] and error["type"] == "missing"
        for error in res["source_errors"]
    )


def test_create_spec_missing_properties(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "name": "Test Spec",
        "definition": "The system should always respond politely",
        "properties": None,
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 422
    res = response.json()
    assert "source_errors" in res
    # Properties is required now, so missing it gives model_attributes_type error
    assert any(error["loc"] == ["body", "properties"] for error in res["source_errors"])


def test_create_spec_priority_default(client, project_and_task):
    """Test that priority defaults to p1 when not provided."""
    project, task = project_and_task

    spec_data = {
        "name": "Test Spec",
        "definition": "The system should always respond politely",
        "properties": create_tone_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 200
    res = response.json()
    assert res["priority"] == 1  # Priority.p1


def test_create_spec_status_default(client, project_and_task):
    """Test that status defaults to active when not provided."""
    project, task = project_and_task

    spec_data = {
        "name": "Test Spec",
        "definition": "The system should always respond politely",
        "properties": create_tone_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "active"


def test_create_spec_tags_default(client, project_and_task):
    """Test that tags defaults to empty list when not provided."""
    project, task = project_and_task

    spec_data = {
        "name": "Test Spec",
        "definition": "The system should always respond politely",
        "properties": create_tone_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 200
    res = response.json()
    assert res["tags"] == []


def test_create_spec_invalid_spec_type_in_properties(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "name": "Test Spec",
        "definition": "The system should always respond politely",
        "properties": {"spec_type": "invalid_type_value"},
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 422
    res = response.json()
    assert "source_errors" in res
    # Check that properties validation failed with invalid spec_type
    assert any(
        "properties" in str(error.get("loc", [])) for error in res["source_errors"]
    )


def test_create_spec_invalid_priority_enum(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "name": "Test Spec",
        "definition": "The system should always respond politely",
        "priority": "p99",
        "properties": create_tone_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 422
    res = response.json()
    assert "source_errors" in res
    assert any(
        error["loc"] == ["body", "priority"] and error["type"] == "enum"
        for error in res["source_errors"]
    )


def test_create_spec_invalid_status_enum(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "name": "Test Spec",
        "definition": "The system should always respond politely",
        "status": "pending",
        "properties": create_tone_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 422
    res = response.json()
    assert "source_errors" in res
    assert any(
        error["loc"] == ["body", "status"] and error["type"] == "enum"
        for error in res["source_errors"]
    )


def test_create_spec_invalid_name_type(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "name": 12345,
        "definition": "The system should always respond politely",
        "properties": create_tone_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 422
    res = response.json()
    assert "source_errors" in res
    assert any(error["loc"] == ["body", "name"] for error in res["source_errors"])


def test_create_spec_invalid_tags_type(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "name": "Test Spec",
        "definition": "The system should always respond politely",
        "tags": "not_a_list",
        "properties": create_tone_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 422
    res = response.json()
    assert "source_errors" in res
    assert any(error["loc"] == ["body", "tags"] for error in res["source_errors"])


def test_create_spec_empty_string_in_tags(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "name": "Test Spec",
        "definition": "The system should always respond politely",
        "tags": [""],
        "properties": create_tone_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 422
    res = response.json()
    assert "source_errors" in res
    assert any(
        "empty strings" in error.get("msg", "").lower()
        for error in res["source_errors"]
    )


def test_create_spec_tag_with_space(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "name": "Test Spec",
        "definition": "The system should always respond politely",
        "tags": ["tag with space"],
        "properties": create_tone_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 422
    res = response.json()
    assert "source_errors" in res
    assert any(
        (
            "spaces" in error.get("msg", "").lower()
            or "underscores" in error.get("msg", "").lower()
        )
        for error in res["source_errors"]
    )


def test_update_spec_invalid_name_type(
    client, project_and_task, sample_tone_properties
):
    """Test that updating a spec with invalid name type fails."""
    project, task = project_and_task

    spec = Spec(
        name="Test Spec",
        definition="System should behave correctly",
        properties=sample_tone_properties,
        eval_id="test_eval_id",
        parent=task,
    )
    spec.save_to_file()

    update_data = {
        "name": 12345,
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.patch(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/{spec.id}",
            json=update_data,
        )

    assert response.status_code == 422
    res = response.json()
    assert "source_errors" in res
    assert any(error["loc"] == ["body", "name"] for error in res["source_errors"])


def test_create_spec_with_empty_tool_function_name(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "name": "Tool Use Spec",
        "definition": "Tool use validation test",
        "properties": {
            "spec_type": "appropriate_tool_use",
            "core_requirement": "Test instruction",
            "tool_id": "test_tool",
            "tool_function_name": "",
            "tool_use_guidelines": "Use this tool when needed",
            "appropriate_tool_use_examples": "examples",
            "inappropriate_tool_use_examples": "examples",
        },
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 422
    res = response.json()
    assert "source_errors" in res
    assert any(
        "tool_function_name" in error.get("msg", "").lower()
        for error in res["source_errors"]
    )


def test_create_spec_with_empty_tool_use_guidelines(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "name": "Tool Use Spec",
        "definition": "Tool use validation test",
        "properties": {
            "spec_type": "appropriate_tool_use",
            "core_requirement": "Test instruction",
            "tool_id": "test_tool",
            "tool_function_name": "test_tool_function",
            "tool_use_guidelines": "",
            "appropriate_tool_use_examples": "examples",
            "inappropriate_tool_use_examples": "examples",
        },
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 422
    res = response.json()
    assert "source_errors" in res
    assert any(
        "tool_use_guidelines" in error.get("msg", "").lower()
        for error in res["source_errors"]
    )


def test_create_spec_with_empty_behavior_description(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "name": "Desired Behaviour Spec",
        "definition": "Desired behaviour validation test",
        "properties": {
            "spec_type": "desired_behaviour",
            "core_requirement": "Test instruction",
            "desired_behaviour_description": "",
            "incorrect_behaviour_examples": "Example 1: Don't do this",
        },
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 422
    res = response.json()
    assert "source_errors" in res
    assert any(
        "desired_behaviour_description" in error.get("msg", "").lower()
        for error in res["source_errors"]
    )


def test_create_spec_with_empty_core_requirement(client, project_and_task):
    project, task = project_and_task

    spec_data = {
        "name": "Tone Spec",
        "definition": "Tone validation test",
        "properties": {
            "spec_type": "tone",
            "core_requirement": "",
            "tone_description": "Professional and friendly",
        },
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 422
    res = response.json()
    assert "source_errors" in res
    assert any(
        "core_requirement" in error.get("msg", "").lower()
        for error in res["source_errors"]
    )


def test_delete_spec_success(client, project_and_task, sample_tone_properties):
    project, task = project_and_task

    spec = Spec(
        name="Test Spec",
        definition="System should behave correctly",
        properties=sample_tone_properties,
        eval_id="test_eval_id",
        parent=task,
    )
    spec.save_to_file()

    specs = task.specs()
    assert len(specs) == 1

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.delete(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/{spec.id}"
        )

    assert response.status_code == 200

    specs = task.specs()
    assert len(specs) == 0


def test_delete_spec_not_found(client, project_and_task):
    project, task = project_and_task

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.delete(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/nonexistent_id"
        )

    assert response.status_code == 404
    assert "Spec not found" in response.json()["message"]


def test_delete_spec_with_associated_eval(
    client, project_and_task, sample_tone_properties
):
    """Test that deleting a spec also deletes its associated eval."""
    project, task = project_and_task

    # Create an eval with required fields (using 'rag' template to avoid needing eval_configs_filter_id)
    eval = Eval(
        name="Test Eval",
        description="Test eval description",
        template="rag",
        eval_set_filter_id="tag::test_eval",
        output_scores=[
            EvalOutputScore(
                name="Quality",
                type=TaskOutputRatingType.five_star,
            )
        ],
        parent=task,
    )
    eval.save_to_file()

    # Create a spec with the eval_id
    spec = Spec(
        name="Test Spec",
        definition="System should behave correctly",
        properties=sample_tone_properties,
        eval_id=eval.id,
        parent=task,
    )
    spec.save_to_file()

    # Verify both exist
    specs = task.specs()
    evals = task.evals()
    assert len(specs) == 1
    assert len(evals) == 1

    # Delete the spec
    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.delete(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/{spec.id}"
        )

    assert response.status_code == 200

    # Verify both spec and eval are deleted
    specs = task.specs()
    evals = task.evals()
    assert len(specs) == 0
    assert len(evals) == 0


def test_delete_spec_without_associated_eval(
    client, project_and_task, sample_tone_properties
):
    """Test that deleting a spec with a non-existent eval works correctly."""
    project, task = project_and_task

    spec = Spec(
        name="Test Spec",
        definition="System should behave correctly",
        properties=sample_tone_properties,
        eval_id="nonexistent_eval_id",
        parent=task,
    )
    spec.save_to_file()

    specs = task.specs()
    assert len(specs) == 1

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.delete(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/{spec.id}"
        )

    assert response.status_code == 200

    specs = task.specs()
    assert len(specs) == 0


def test_create_spec_with_task_sample(client, project_and_task):
    """Test creating a spec with a task sample."""
    project, task = project_and_task

    spec_data = {
        "name": "Spec With Sample",
        "definition": "Test spec with task sample",
        "properties": create_tone_properties_dict(),
        "task_sample": {
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris.",
        },
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 200
    res = response.json()
    assert res["name"] == "Spec With Sample"
    assert res["task_sample"] is not None
    assert res["task_sample"]["input"] == "What is the capital of France?"
    assert res["task_sample"]["output"] == "The capital of France is Paris."

    specs = task.specs()
    assert len(specs) == 1
    assert specs[0].task_sample is not None
    assert specs[0].task_sample.input == "What is the capital of France?"
    assert specs[0].task_sample.output == "The capital of France is Paris."


def test_create_spec_without_task_sample(client, project_and_task):
    """Test creating a spec without a task sample (should default to None)."""
    project, task = project_and_task

    spec_data = {
        "name": "Spec Without Sample",
        "definition": "Test spec without task sample",
        "properties": create_tone_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 200
    res = response.json()
    assert res["name"] == "Spec Without Sample"
    assert res["task_sample"] is None

    specs = task.specs()
    assert len(specs) == 1
    assert specs[0].task_sample is None


def test_get_spec_with_task_sample(client, project_and_task, sample_tone_properties):
    """Test getting a spec with a task sample."""
    project, task = project_and_task

    sample = TaskSample(
        input="Example input for get test",
        output="Example output for get test",
    )
    spec = Spec(
        name="Test Spec With Sample",
        definition="System should behave correctly",
        properties=sample_tone_properties,
        eval_id="test_eval_id",
        task_sample=sample,
        parent=task,
    )
    spec.save_to_file()

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.get(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/{spec.id}"
        )

    assert response.status_code == 200
    res = response.json()
    assert res["task_sample"] is not None
    assert res["task_sample"]["input"] == "Example input for get test"
    assert res["task_sample"]["output"] == "Example output for get test"


def test_create_spec_rolls_back_eval_on_spec_save_failure(
    client, project_and_task, sample_tone_properties
):
    """Test that eval is deleted when spec.save_to_file() fails during create."""
    project, task = project_and_task

    spec_data = {
        "name": "Test Spec",
        "definition": "The system should always respond politely",
        "properties": create_tone_properties_dict(),
    }

    def spec_save_fails(self):
        raise Exception("disk full")

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        with patch.object(Spec, "save_to_file", spec_save_fails):
            with pytest.raises(Exception, match="disk full"):
                client.post(
                    f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
                )

    # Eval should have been cleaned up
    assert len(task.evals()) == 0
    assert len(task.specs()) == 0


def test_update_spec_name_rollback_eval_fails_logs_error(
    client: TestClient,
    project_and_task: tuple[Project, Task],
    sample_tone_properties: ToneProperties,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that when spec save fails AND eval rollback also fails, the error is logged."""
    project, task = project_and_task

    eval = Eval(
        name="Original Name",
        description="Test eval",
        template="rag",
        eval_set_filter_id="tag::test_eval",
        output_scores=[
            EvalOutputScore(
                name="Quality",
                type=TaskOutputRatingType.five_star,
            )
        ],
        parent=task,
    )
    eval.save_to_file()

    spec = Spec(
        name="Original Name",
        definition="Original definition",
        priority=Priority.p2,
        status=SpecStatus.active,
        tags=[],
        eval_id=eval.id,
        properties=sample_tone_properties,
        parent=task,
    )
    spec.save_to_file()

    update_data = {"name": "Updated Name"}

    eval_save_count = 0
    original_eval_save = Eval.save_to_file

    def eval_save_fails_on_rollback(self):
        nonlocal eval_save_count
        eval_save_count += 1
        if eval_save_count == 1:
            # First call: name sync save succeeds
            return original_eval_save(self)
        # Second call: rollback save fails
        raise Exception("rollback error")

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        with patch.object(Spec, "save_to_file", side_effect=Exception("disk error")):
            with patch.object(Eval, "save_to_file", eval_save_fails_on_rollback):
                with caplog.at_level(logging.ERROR):
                    with pytest.raises(Exception, match="disk error"):
                        client.patch(
                            f"/api/projects/{project.id}/tasks/{task.id}/specs/{spec.id}",
                            json=update_data,
                        )

    assert "Failed to roll back eval after spec save failure" in caplog.text


class TestResolveAvailableSpecName:
    """The early-check resolver behind /available_spec_name — the same
    derived-tag comparison the spec-save guard enforces."""

    def test_free_name_returned_verbatim(self):
        result = resolve_available_spec_name("policy_adherence", ["other_spec"])
        assert result.name == "policy_adherence"
        assert result.was_taken is False

    def test_collision_gets_the_first_free_suffix(self):
        result = resolve_available_spec_name("Policy Adherence", ["Policy Adherence"])
        assert result.name == "Policy Adherence 2"
        assert result.was_taken is True

    def test_suffix_walks_past_taken_variants(self):
        result = resolve_available_spec_name(
            "Policy Adherence",
            ["Policy Adherence", "Policy Adherence 2", "Policy Adherence 3"],
        )
        assert result.name == "Policy Adherence 4"
        assert result.was_taken is True

    def test_collision_is_tag_derived_not_string_equality(self):
        # "Policy Adherence" and "policy_adherence" share a tag namespace —
        # the exact comparison that would 409 at save.
        result = resolve_available_spec_name("Policy Adherence", ["policy_adherence"])
        assert result.was_taken is True
        result = resolve_available_spec_name("policy_adherence", ["Policy Adherence"])
        assert result.was_taken is True

    def test_suffix_trims_to_the_short_name_limit(self):
        # 32-char candidate whose trim cut lands EXACTLY on a space: the base
        # is trimmed so base + " 2" still fits, and the trailing-space strip
        # is what keeps the join from fabricating a forbidden double space.
        long_name = "a" * 29 + " bb"  # 32 chars; [:30] ends with " "
        result = resolve_available_spec_name(long_name, [long_name])
        assert result.was_taken is True
        assert result.name == "a" * 29 + " 2"
        assert len(result.name) <= 32
        assert "  " not in result.name

    def test_suffix_trim_also_drops_a_trailing_underscore(self):
        # Same cut landing on a "_": stripping it keeps a "_ 2" seam out of
        # the name, which the validator allows but reads as a typo.
        long_name = "a" * 29 + "_bb"  # 32 chars; [:30] ends with "_"
        result = resolve_available_spec_name(long_name, [long_name])
        assert result.name == "a" * 29 + " 2"

    def test_old_underscore_suffixed_names_still_collide(self):
        # Evals named before the space suffix share the tag namespace with
        # the new form, so the walk skips them rather than duplicating.
        result = resolve_available_spec_name(
            "Policy Adherence", ["policy_adherence", "policy_adherence_2"]
        )
        assert result.name == "Policy Adherence 3"

    def test_exhausted_search_refuses(self):
        taken = ["name"] + [f"name {i}" for i in range(2, 100)]
        with pytest.raises(HTTPException) as exc:
            resolve_available_spec_name("name", taken)
        assert exc.value.status_code == 409


def test_available_spec_name_route(client, project_and_task):
    project, task = project_and_task
    url = f"/api/projects/{project.id}/tasks/{task.id}/available_spec_name"

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task

        # No specs yet: the candidate comes back untouched.
        response = client.get(url, params={"name": "policy_adherence"})
        assert response.status_code == 200
        assert response.json() == {"name": "policy_adherence", "was_taken": False}

        # Save a spec under that namespace, then the same candidate suffixes.
        spec = Spec(
            parent=task,
            name="Policy Adherence",  # differs by case/spacing — still collides
            definition="def",
            properties={
                "spec_type": "desired_behaviour",
                "core_requirement": "x",
                "desired_behaviour_description": "y",
            },
            priority=Priority.p1,
            status=SpecStatus.active,
            eval_id="12345",
        )
        spec.save_to_file()
        response = client.get(url, params={"name": "policy_adherence"})
        assert response.status_code == 200
        assert response.json() == {"name": "policy_adherence 2", "was_taken": True}


def test_available_spec_name_route_rejects_invalid_candidate(client, project_and_task):
    project, task = project_and_task
    url = f"/api/projects/{project.id}/tasks/{task.id}/available_spec_name"
    # Over the short-name limit — the query param carries the same validator
    # the save request enforces. Validation fires before task resolution.
    response = client.get(url, params={"name": "x" * 33})
    assert response.status_code == 422


def test_available_spec_name_route_task_not_found(client):
    response = client.get(
        "/api/projects/p/tasks/t/available_spec_name",
        params={"name": "policy_adherence"},
    )
    assert response.status_code == 404


def test_create_spec_sets_priority_and_status_on_eval(client, project_and_task):
    """Priority/status live on the eval going forward; spec creation writes them there."""
    project, task = project_and_task

    spec_data = {
        "name": "Eval Fields Spec",
        "definition": "The system should always respond politely",
        "priority": Priority.p2,
        "status": SpecStatus.future.value,
        "properties": create_tone_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )

    assert response.status_code == 200

    evals = task.evals()
    assert len(evals) == 1
    assert evals[0].priority == Priority.p2
    assert evals[0].status == SpecStatus.future
    assert evals[0].resolved_priority() == Priority.p2
    assert evals[0].resolved_status() == SpecStatus.future


def test_update_spec_priority_status_sync_to_eval(client, project_and_task):
    """PATCHing priority/status on a spec forwards them to the linked eval,
    which is the source of truth for reads."""
    project, task = project_and_task

    spec_data = {
        "name": "Sync Spec",
        "definition": "The system should always respond politely",
        "properties": create_tone_properties_dict(),
    }

    with patch("kiln_server.spec_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = task
        create_response = client.post(
            f"/api/projects/{project.id}/tasks/{task.id}/specs", json=spec_data
        )
        assert create_response.status_code == 200
        spec_id = create_response.json()["id"]

        update_response = client.patch(
            f"/api/projects/{project.id}/tasks/{task.id}/specs/{spec_id}",
            json={"priority": Priority.p3, "status": SpecStatus.archived.value},
        )

    assert update_response.status_code == 200

    evals = task.evals()
    assert len(evals) == 1
    assert evals[0].priority == Priority.p3
    assert evals[0].status == SpecStatus.archived
