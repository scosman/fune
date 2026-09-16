import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kiln_ai.datamodel import (
    DataSource,
    DataSourceType,
    Project,
    Task,
    TaskOutput,
    TaskRun,
)
from kiln_ai.datamodel.data_guide import DataGuide
from kiln_ai.datamodel.datamodel_enums import ModelProviderName, StructuredOutputMode
from kiln_ai.datamodel.extraction import Document
from kiln_ai.datamodel.prompt_id import PromptGenerators
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties
from kiln_server.custom_errors import connect_custom_errors

from app.desktop.studio_server.data_gen_api import (
    _MAX_BATCH_JOBS,
    DataGenCategoriesApiInput,
    DataGenQnaApiInput,
    DataGenSampleApiInput,
    DataGenSaveSamplesApiInput,
    GenerateOutputsBatchItem,
    SaveQnaPairInput,
    _batch_jobs,
    _BatchJob,
    _generate_one_input,
    _generate_one_output,
    _register_batch_job,
    _run_inputs_batch_job,
    _run_outputs_batch_job,
    connect_data_gen_api,
    topic_path_from_string,
    topic_path_to_string,
)


@pytest.fixture
def app():
    app = FastAPI()
    connect_custom_errors(app)
    connect_data_gen_api(app)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def data_source():
    return DataSource(
        type=DataSourceType.synthetic,
        properties={
            "model_name": "gpt-4",
            "model_provider": "openai",
            "adapter_name": "langchain_adapter",
        },
    )


@pytest.fixture
def test_project(tmp_path) -> Project:
    project_path = tmp_path / "test_project" / "project.kiln"
    project_path.parent.mkdir()
    project = Project(name="Test Project", path=project_path)
    project.save_to_file()
    return project


@pytest.fixture
def test_task(test_project) -> Task:
    task = Task(name="Test Task", instruction="Test Instruction", parent=test_project)
    task.save_to_file()
    return task


@pytest.fixture
def mock_task_run(data_source, test_task):
    return TaskRun(
        output=TaskOutput(
            output="Test Output",
            source=data_source,
        ),
        input="Test Input",
        input_source=data_source,
        parent=test_task,
    )


@pytest.fixture
def mock_task_adapter(mock_task_run):
    with patch("app.desktop.studio_server.data_gen_api.adapter_for_task") as mock:
        mock_adapter = AsyncMock()
        mock_adapter.invoke = AsyncMock()
        mock.return_value = mock_adapter

        mock_adapter.invoke.return_value = mock_task_run

        yield mock_adapter


@pytest.fixture
def mock_project_from_id(test_project):
    with patch("app.desktop.studio_server.data_gen_api.project_from_id") as mock:
        mock.return_value = test_project
        yield mock


@pytest.fixture
def mock_task_from_id(test_task):
    with patch("app.desktop.studio_server.data_gen_api.task_from_id") as mock:
        mock.return_value = test_task
        yield mock


def test_generate_categories_success(
    mock_task_from_id,
    mock_project_from_id,
    mock_task_adapter,
    client,
    mock_task_run,
):
    # Arrange
    input_data = DataGenCategoriesApiInput(
        node_path=["parent", "child"],
        num_subtopics=4,
        guidance="Generate tech categories",
        gen_type="eval",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE,
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
    )

    # Act
    response = client.post(
        "/api/projects/proj-ID/tasks/task-ID/generate_categories",
        json=input_data.model_dump(),
    )

    # Assert
    assert response.status_code == 200
    res = response.json()
    assert TaskOutput.model_validate(res["output"]) == mock_task_run.output
    mock_task_adapter.invoke.assert_awaited_once()


def test_generate_samples_success(
    mock_project_from_id,
    mock_task_from_id,
    mock_task_adapter,
    client,
    mock_task_run,
):
    # Arrange
    input_data = DataGenSampleApiInput(
        topic=["technology", "AI"],
        gen_type="training",
        guidance="Make long samples",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE,
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
    )

    # Act
    response = client.post(
        "/api/projects/proj-ID/tasks/task-ID/generate_inputs",
        json=input_data.model_dump(),
    )

    # Assert
    assert response.status_code == 200
    res = response.json()
    assert TaskOutput.model_validate(res["output"]) == mock_task_run.output
    mock_task_adapter.invoke.assert_awaited_once()


def test_generate_categories_rejects_mcp_run_config(
    mock_task_from_id,
    mock_project_from_id,
    client,
):
    input_data = {
        "node_path": ["parent", "child"],
        "num_subtopics": 4,
        "guidance": "Generate tech categories",
        "gen_type": "eval",
        "run_config_properties": {
            "type": "mcp",
            "tool_reference": {
                "tool_id": "mcp::local::server_id::tool_name",
            },
        },
    }

    response = client.post(
        "/api/projects/proj-ID/tasks/task-ID/generate_categories",
        json=input_data,
    )

    assert response.status_code == 422


def test_generate_samples_rejects_mcp_run_config(
    mock_task_from_id,
    mock_project_from_id,
    client,
):
    input_data = {
        "topic": ["technology", "AI"],
        "gen_type": "training",
        "guidance": "Make long samples",
        "run_config_properties": {
            "type": "mcp",
            "tool_reference": {
                "tool_id": "mcp::local::server_id::tool_name",
            },
        },
    }

    response = client.post(
        "/api/projects/proj-ID/tasks/task-ID/generate_inputs",
        json=input_data,
    )

    assert response.status_code == 422


@pytest.mark.paid
def test_save_sample_success_paid_run(
    mock_task_from_id,
    client,
    test_task,
):
    # Arrange
    input_data = DataGenSaveSamplesApiInput(
        input="Test sample input",
        input_model_name="gpt_4o",
        input_provider="openai",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt_4o_mini",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE,
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        topic_path=[],  # No topic path
        tags=["test_tag"],
    )

    # Act
    response = client.post(
        "/api/projects/proj-ID/tasks/task-ID/save_sample",
        json=input_data.model_dump(),
    )

    # Assert
    assert response.status_code == 200
    # Verify TaskRun was created with correct properties
    mock_task_from_id.assert_called_once_with("proj-ID", "task-ID")
    saved_runs = test_task.runs(include_intermediate_runs=True)
    assert len(saved_runs) == 1
    saved_run = saved_runs[0]

    assert saved_run.input == "Test sample input"
    assert saved_run.input_source.type == DataSourceType.synthetic
    properties = saved_run.input_source.properties
    assert properties == {
        "model_name": "gpt_4o",
        "model_provider": "openai",
        "adapter_name": "kiln_data_gen",
    }
    assert "topic_path" not in properties

    assert saved_run.output.source.type == DataSourceType.synthetic
    assert saved_run.output.source.properties["model_name"] == "gpt_4o_mini"
    assert saved_run.output.source.properties["model_provider"] == "openai"
    assert "test_tag" in saved_run.tags

    # Confirm the response contains same run
    assert response.json()["id"] == saved_run.id

    return


def test_generate_sample_success_with_mock_invoke(
    mock_task_from_id,
    mock_task_adapter,
    client,
    mock_task_run,
    test_task,
):
    # Arrange
    input_data = DataGenSaveSamplesApiInput(
        input="Test sample input",
        input_model_name="gpt_4o",
        input_provider="openai",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt_4o_mini",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE,
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        topic_path=["AI", "Machine Learning", "Deep Learning"],
        tags=["test_tag"],
    )

    # Act

    response = client.post(
        "/api/projects/proj-ID/tasks/task-ID/generate_sample?session_id=1234",
        json=input_data.model_dump(),
    )

    mock_task_adapter.invoke.assert_awaited_once()
    invoke_args = mock_task_adapter.invoke.await_args[1]
    assert invoke_args["input"] == "Test sample input"
    assert invoke_args["input_source"].type == DataSourceType.synthetic
    properties = invoke_args["input_source"].properties
    assert properties == {
        "model_name": "gpt_4o",
        "model_provider": "openai",
        "adapter_name": "kiln_data_gen",
        "topic_path": "AI>>>>>Machine Learning>>>>>Deep Learning",
    }

    # Assert
    assert response.status_code == 200
    # Verify TaskRun was created with correct properties
    mock_task_from_id.assert_called_once_with("proj-ID", "task-ID")

    # Check none are saved before calling save
    saved_runs = test_task.runs(include_intermediate_runs=True)
    assert len(saved_runs) == 0

    # Call save
    response = client.post(
        "/api/projects/proj-ID/tasks/task-ID/save_sample",
        json=response.json(),
    )
    assert response.status_code == 200

    # Check one is saved after calling save
    saved_runs = test_task.runs(include_intermediate_runs=True)
    assert len(saved_runs) == 1
    saved_run = saved_runs[0]
    assert saved_run.input_source.type == DataSourceType.synthetic
    # Not checking values since we mock them. We check it's called with correct properties above
    assert saved_run.output == mock_task_run.output
    assert saved_run.output.source.type == DataSourceType.synthetic
    assert "test_tag" in saved_run.tags
    assert "synthetic" in saved_run.tags
    assert "synthetic_session_1234" in saved_run.tags

    assert saved_run.output.source.properties == mock_task_run.input_source.properties

    # Confirm the response contains same run
    assert response.json()["id"] == saved_run.id


def test_generate_sample_success_with_topic_path(
    mock_task_from_id,
    mock_task_adapter,
    client,
    mock_task_run,
):
    # Arrange
    input_data = DataGenSaveSamplesApiInput(
        input="Test sample input",
        topic_path=["AI", "Machine Learning", "Deep Learning"],
        input_model_name="gpt_4o",
        input_provider="openai",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt_4o_mini",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE,
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
    )

    # Act
    response = client.post(
        "/api/projects/proj-ID/tasks/task-ID/generate_sample",
        json=input_data.model_dump(),
    )

    # Assert
    assert response.status_code == 200
    invoke_args = mock_task_adapter.invoke.await_args[1]
    assert (
        invoke_args["input_source"].properties["topic_path"]
        == "AI>>>>>Machine Learning>>>>>Deep Learning"
    )
    parsed_path = topic_path_from_string(
        invoke_args["input_source"].properties["topic_path"]
    )
    assert parsed_path == ["AI", "Machine Learning", "Deep Learning"]


def test_topic_path_conversions():
    # Test empty path
    assert topic_path_to_string([]) is None
    assert topic_path_from_string(None) == []
    assert topic_path_from_string("") == []

    # Test non-empty path
    test_path = ["AI", "Machine Learning", "Deep Learning"]
    path_string = topic_path_to_string(test_path)
    assert path_string == "AI>>>>>Machine Learning>>>>>Deep Learning"
    assert topic_path_from_string(path_string) == test_path

    # Test single item path
    assert topic_path_to_string(["AI"]) == "AI"
    assert topic_path_from_string("AI") == ["AI"]


@pytest.mark.parametrize(
    "guidance,topic_path,initial_instruction,expected_wrapped_call,expected_final_instruction,should_call_wrap",
    [
        # Test 1: Only guidance provided (no topic path)
        (
            "Custom guidance for generation",
            [],
            "Test Instruction",
            "Custom guidance for generation",
            "wrapped_instruction",
            True,
        ),
        # Test 2: Only topic path provided (no guidance). Topic path block is
        # appended after a leading newline to the empty guidance string.
        (
            None,
            ["AI", "Machine Learning"],
            "Original instruction",
            """
## Topic Path
The topic path for this sample is:
["AI", "Machine Learning"]""",
            "wrapped_instruction_with_topic",
            True,
        ),
        # Test 3: Both guidance and topic path provided. Topic path block is
        # appended directly after the guidance with a single newline separator.
        (
            "Focus on technical accuracy",
            ["Technology", "AI"],
            "Original instruction",
            """Focus on technical accuracy
## Topic Path
The topic path for this sample is:
["Technology", "AI"]""",
            "wrapped_instruction_combined",
            True,
        ),
        # Test 4: Neither guidance nor topic path provided
        (
            None,
            [],
            "Original instruction",
            None,  # Won't be called
            "Original instruction",  # Should remain unchanged
            False,
        ),
        # Test 5: Empty guidance string (should be treated as no guidance)
        (
            "",
            [],
            "Original instruction",
            None,  # Won't be called
            "Original instruction",  # Should remain unchanged
            False,
        ),
    ],
    ids=[
        "only_guidance",
        "only_topic_path",
        "both_guidance_and_topic_path",
        "neither_guidance_nor_topic_path",
        "empty_guidance",
    ],
)
def test_generate_sample_guidance_generation(
    mock_task_from_id,
    mock_task_adapter,
    client,
    mock_task_run,
    test_task,
    guidance,
    topic_path,
    initial_instruction,
    expected_wrapped_call,
    expected_final_instruction,
    should_call_wrap,
):
    """Test that guidance is properly generated and applied in save_sample function"""
    from unittest.mock import patch

    with patch(
        "app.desktop.studio_server.data_gen_api.wrap_task_with_guidance"
    ) as mock_wrap:
        # Set up the mock return value and initial task instruction
        mock_wrap.return_value = expected_final_instruction
        test_task.instruction = initial_instruction

        input_data = DataGenSaveSamplesApiInput(
            input="Test input",
            input_model_name="gpt-4",
            input_provider="openai",
            run_config_properties=KilnAgentRunConfigProperties(
                model_name="gpt-4",
                model_provider_name=ModelProviderName.openai,
                prompt_id=PromptGenerators.SIMPLE,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            topic_path=topic_path,
            guidance=guidance,
        )

        response = client.post(
            "/api/projects/proj-ID/tasks/task-ID/generate_sample",
            json=input_data.model_dump(),
        )

        assert response.status_code == 200

        if should_call_wrap:
            # Verify wrap_task_with_guidance was called with expected parameters
            mock_wrap.assert_called_once_with(
                initial_instruction, expected_wrapped_call
            )
        else:
            # Verify wrap_task_with_guidance was NOT called
            mock_wrap.assert_not_called()

        # Verify task instruction final state
        assert test_task.instruction == expected_final_instruction


def test_generate_qna_success_with_session_and_tags(
    mock_task_from_id,
    mock_task_adapter,
    client,
    mock_task_run,
    test_task,
):
    with (
        patch(
            "kiln_ai.datamodel.extraction.Document.from_id_and_parent_path"
        ) as mock_document,
        patch(
            "app.desktop.studio_server.data_gen_api.project_from_id"
        ) as mock_project_from_id,
    ):
        mock_document.return_value = MagicMock(friendly_name="doc1", spec=Document)
        mock_project_from_id.return_value = test_task.parent

        input_data = DataGenQnaApiInput(
            document_id="doc1",
            part_text=["section a", "section b"],
            num_samples=3,
            run_config_properties=KilnAgentRunConfigProperties(
                model_name="gpt_4o_mini",
                model_provider_name=ModelProviderName.openai,
                prompt_id=PromptGenerators.SIMPLE,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            guidance="Make concise QnA",
            tags=["custom_tag"],
        )

        response = client.post(
            "/api/projects/proj-ID/tasks/task-ID/generate_qna?session_id=abcd",
            json=input_data.model_dump(),
        )

        assert response.status_code == 200
        res = response.json()
        assert set(
            ["synthetic", "qna", "synthetic_qna_session_abcd", "custom_tag"]
        ).issubset(set(res.get("tags", [])))

        # Verify adapter was invoked with the expected positional dict payload
        called_args, called_kwargs = mock_task_adapter.invoke.await_args
        assert called_kwargs == {}
        payload = called_args[0]
        assert payload["kiln_data_gen_document_name"] == "doc1"
        assert payload["kiln_data_gen_part_text"] == ["section a", "section b"]
        assert payload["kiln_data_gen_num_samples"] == 3


def test_generate_qna_rejects_mcp_run_config(
    mock_task_from_id,
    client,
    test_task,
):
    with (
        patch(
            "kiln_ai.datamodel.extraction.Document.from_id_and_parent_path"
        ) as mock_document,
        patch(
            "app.desktop.studio_server.data_gen_api.project_from_id"
        ) as mock_project_from_id,
    ):
        mock_document.return_value = MagicMock(friendly_name="doc1", spec=Document)
        mock_project_from_id.return_value = test_task.parent

        input_data = {
            "document_id": "doc1",
            "part_text": ["section a", "section b"],
            "num_samples": 3,
            "run_config_properties": {
                "type": "mcp",
                "tool_reference": {
                    "tool_id": "mcp::local::server_id::tool_name",
                },
            },
        }

        response = client.post(
            "/api/projects/proj-ID/tasks/task-ID/generate_qna?session_id=abcd",
            json=input_data,
        )

        assert response.status_code == 422


def test_save_qna_pair_persists_task_run(
    mock_task_from_id,
    client,
    test_task,
):
    input_data = SaveQnaPairInput(
        query="What is Kiln?",
        answer="Kiln is an app for building AI systems.",
        model_name="gpt-4",
        model_provider="openai",
        tags=["my_tag"],
    )

    response = client.post(
        "/api/projects/proj-ID/tasks/task-ID/save_qna_pair?session_id=9876",
        json=input_data.model_dump(),
    )

    assert response.status_code == 200
    mock_task_from_id.assert_called_once_with("proj-ID", "task-ID")

    saved_runs = test_task.runs(include_intermediate_runs=True)
    assert len(saved_runs) == 1
    run = saved_runs[0]

    assert run.input == "What is Kiln?"
    assert run.output.output == "Kiln is an app for building AI systems."

    assert run.input_source.type == DataSourceType.synthetic
    assert run.input_source.properties == {
        "model_name": "gpt-4",
        "model_provider": "openai",
        "adapter_name": "kiln_qna_manual_save",
    }

    assert run.output.source.type == DataSourceType.synthetic
    assert run.output.source.properties == {
        "model_name": "gpt-4",
        "model_provider": "openai",
        "adapter_name": "kiln_qna_manual_save",
    }

    tags = set(run.tags)
    assert {"synthetic", "qna", "synthetic_qna_session_9876", "my_tag"}.issubset(tags)

    # Verify trace contains system, user, assistant messages
    assert isinstance(run.trace, list) and len(run.trace) == 3
    assert run.trace[0]["role"] == "system"
    assert run.trace[1]["role"] == "user" and run.trace[1]["content"] == "What is Kiln?"
    assert (
        run.trace[2]["role"] == "assistant"
        and run.trace[2]["content"] == "Kiln is an app for building AI systems."
    )


def test_get_data_gen_guide_none(
    mock_task_from_id,
    client,
):
    response = client.get("/api/projects/test_project/tasks/test_task/data_gen_guide")
    assert response.status_code == 200
    assert response.json() is None


def test_save_and_get_data_gen_guide(
    mock_task_from_id,
    test_task,
    client,
):
    body = (
        "# Reference Examples\n\n## Example 1\n```input\nx\n```\n\n```output\ny\n```\n\n"
        "# Guidelines & Rules\n\n<output_semantic>\n\n## Cholesterol\nIf cholesterol is high, never have low LDL.\n\n</output_semantic>"
    )
    response = client.put(
        "/api/projects/test_project/tasks/test_task/data_gen_guide",
        json={"guide": body},
    )
    assert response.status_code == 200
    result = response.json()
    assert "x" in result["guide"]
    assert "cholesterol" in result["guide"].lower()

    # Verify it's persisted via GET
    get_response = client.get(
        "/api/projects/test_project/tasks/test_task/data_gen_guide"
    )
    assert get_response.status_code == 200
    get_result = get_response.json()
    assert "x" in get_result["guide"]
    assert "cholesterol" in get_result["guide"].lower()

    # Verify task was actually saved to disk
    reloaded_task = Task.from_id_and_parent_path(test_task.id, test_task.parent.path)
    assert reloaded_task is not None
    current = reloaded_task.current_data_guide()
    assert current is not None
    assert "x" in current.guide
    assert "cholesterol" in current.guide.lower()
    # Only one DataGuide should exist on disk — saves overwrite in place
    # rather than accumulating new files.
    assert len(reloaded_task.data_guides()) == 1


def test_save_data_gen_guide_defaults_source_to_manual(
    mock_task_from_id,
    test_task,
    client,
):
    """First save without an explicit `source` defaults to 'manual' for
    back-compat. Existing guides on disk also default to 'manual'."""
    response = client.put(
        "/api/projects/test_project/tasks/test_task/data_gen_guide",
        json={"guide": "body"},
    )
    assert response.status_code == 200
    assert response.json()["source"] == "manual"


def test_save_data_gen_guide_persists_kiln_pro_source(
    mock_task_from_id,
    test_task,
    client,
):
    """When the copilot flow saves a guide it sends source='kiln_pro'; the
    backend persists it so refine can branch on it later."""
    response = client.put(
        "/api/projects/test_project/tasks/test_task/data_gen_guide",
        json={"guide": "body", "source": "kiln_pro"},
    )
    assert response.status_code == 200
    assert response.json()["source"] == "kiln_pro"


def test_save_data_gen_guide_edit_preserves_existing_source(
    mock_task_from_id,
    test_task,
    client,
):
    """Editing/refining a guide without sending `source` preserves whatever
    source the existing guide already had — otherwise editing a kiln_pro guide
    would silently flip it to manual and pick the wrong refine branch next time."""
    client.put(
        "/api/projects/test_project/tasks/test_task/data_gen_guide",
        json={"guide": "original", "source": "kiln_pro"},
    )
    response = client.put(
        "/api/projects/test_project/tasks/test_task/data_gen_guide",
        json={"guide": "edited"},
    )
    assert response.status_code == 200
    assert response.json()["source"] == "kiln_pro"
    assert response.json()["guide"] == "edited"


def test_save_data_gen_guide_rejects_blank(
    mock_task_from_id,
    test_task,
    client,
):
    """Blank/whitespace-only guides are rejected with 400 — DELETE is the
    correct way to remove a guide."""
    response = client.put(
        "/api/projects/test_project/tasks/test_task/data_gen_guide",
        json={"guide": "   \n  "},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["message"].lower()


def test_delete_data_gen_guide(
    mock_task_from_id,
    test_task,
    client,
):
    # First save a guide
    client.put(
        "/api/projects/test_project/tasks/test_task/data_gen_guide",
        json={"guide": "some guide body"},
    )

    # Delete it
    response = client.delete(
        "/api/projects/test_project/tasks/test_task/data_gen_guide"
    )
    assert response.status_code == 200

    # Verify it's gone
    get_response = client.get(
        "/api/projects/test_project/tasks/test_task/data_gen_guide"
    )
    assert get_response.json() is None


def test_save_data_gen_guide_overwrites_previous(
    mock_task_from_id,
    test_task,
    client,
):
    # Save first guide
    first_response = client.put(
        "/api/projects/test_project/tasks/test_task/data_gen_guide",
        json={"guide": "first body"},
    )
    first_id = first_response.json()["id"]

    # Overwrite with second
    response = client.put(
        "/api/projects/test_project/tasks/test_task/data_gen_guide",
        json={"guide": "second body"},
    )
    assert response.status_code == 200

    get_response = client.get(
        "/api/projects/test_project/tasks/test_task/data_gen_guide"
    )
    result = get_response.json()
    assert result["guide"] == "second body"

    # Same file — second save reuses the first DataGuide rather than creating
    # a new one. Keeps git history of the guide localized to one file.
    assert result["id"] == first_id
    reloaded_task = Task.from_id_and_parent_path(test_task.id, test_task.parent.path)
    assert reloaded_task is not None
    assert len(reloaded_task.data_guides()) == 1


# --- _resolve_data_guide / _combine_guidance helpers ---


def test_resolve_data_guide_override_replaces_persisted(test_task):
    """An explicit override (non-empty) replaces the persisted guide."""
    from kiln_ai.datamodel.data_guide import DataGuide

    from app.desktop.studio_server.data_gen_api import _resolve_data_guide

    saved = DataGuide(parent=test_task, guide="persisted body")
    saved.save_to_file()
    reloaded = Task.from_id_and_parent_path(test_task.id, test_task.parent.path)
    assert reloaded is not None

    assert _resolve_data_guide(reloaded, "override text") == "override text"


def test_resolve_data_guide_blank_override_returns_none(test_task):
    """A blank/whitespace override is treated as 'don't include any guide for
    this call' even if there's a persisted one."""
    from kiln_ai.datamodel.data_guide import DataGuide

    from app.desktop.studio_server.data_gen_api import _resolve_data_guide

    saved = DataGuide(parent=test_task, guide="persisted body")
    saved.save_to_file()
    reloaded = Task.from_id_and_parent_path(test_task.id, test_task.parent.path)
    assert reloaded is not None

    assert _resolve_data_guide(reloaded, "") is None
    assert _resolve_data_guide(reloaded, "   \n  ") is None


def test_resolve_data_guide_falls_back_to_persisted(test_task):
    """`None` means 'no override provided' → fall back to the saved guide."""
    from kiln_ai.datamodel.data_guide import DataGuide

    from app.desktop.studio_server.data_gen_api import _resolve_data_guide

    body = "# Reference Examples\n\nexamples body\n\n# Guidelines & Rules\n\nrules body"
    saved = DataGuide(parent=test_task, guide=body)
    saved.save_to_file()
    reloaded = Task.from_id_and_parent_path(test_task.id, test_task.parent.path)
    assert reloaded is not None

    out = _resolve_data_guide(reloaded, None)
    assert out == body


def test_resolve_data_guide_no_persisted_returns_none(test_task):
    from app.desktop.studio_server.data_gen_api import _resolve_data_guide

    assert _resolve_data_guide(test_task, None) is None


def test_combine_guidance_with_data_guide_only(test_task):
    """With a saved guide and no template guidance, the helper wraps
    the guide with the framing paragraph + stage hint."""
    from kiln_ai.datamodel.data_guide import DataGuide

    from app.desktop.studio_server.data_gen_api import _combine_guidance

    saved = DataGuide(parent=test_task, guide="GUIDE_BODY")
    saved.save_to_file()
    reloaded = Task.from_id_and_parent_path(test_task.id, test_task.parent.path)
    assert reloaded is not None

    out = _combine_guidance(reloaded, None, "inputs")
    assert out is not None
    assert "# Task Data Guide" in out
    assert "GUIDE_BODY" in out
    # New stage hint references the section shape (mirrors → "mirror their structure" still in inputs hint).
    assert "mirror their structure" in out
    # Authority cascade copy is present.
    assert "Authority cascade" in out
    # Template label appears in the cascade documentation, but the actual
    # labeled block is not rendered (no template was passed).
    assert "Per-stage guidance from the eval template" not in out
    # The new wrapper teaches the new section shape — old XML rule-grouping
    # framing must be gone. The migration help may still mention
    # `<input_structural>` by name so older guides can be read; the
    # assertion below targets the active *teaching*, not incidental references.
    assert "Rule grouping" not in out
    assert "two valid groups are" not in out


def test_combine_guidance_data_guide_and_template(test_task):
    """Both layers (data guide + template) appear in the right order with the
    authority cascade preserved."""
    from kiln_ai.datamodel.data_guide import DataGuide

    from app.desktop.studio_server.data_gen_api import _combine_guidance

    saved = DataGuide(parent=test_task, guide="GUIDE_BODY")
    saved.save_to_file()
    reloaded = Task.from_id_and_parent_path(test_task.id, test_task.parent.path)
    assert reloaded is not None

    out = _combine_guidance(reloaded, "TEMPLATE_GUIDANCE", "inputs")
    assert out is not None
    assert "GUIDE_BODY" in out
    assert "TEMPLATE_GUIDANCE" in out
    assert "# Task Data Guide" in out
    assert "Per-stage guidance from the eval template" in out
    assert out.index("GUIDE_BODY") < out.index("TEMPLATE_GUIDANCE")


def test_combine_guidance_session_only(test_task):
    """With no saved guide and only template guidance, the helper just
    returns the template-guidance block — no Data Guide framing."""
    from app.desktop.studio_server.data_gen_api import _combine_guidance

    out = _combine_guidance(test_task, "SESSION_ONLY", "topics")
    assert out is not None
    assert "SESSION_ONLY" in out
    assert "# Template Guidance" in out
    assert "# Task Data Guide" not in out


def test_combine_guidance_empty_returns_none(test_task):
    """No saved guide, no template, no override → None."""
    from app.desktop.studio_server.data_gen_api import _combine_guidance

    assert _combine_guidance(test_task, None, "topics") is None


def test_combine_guidance_override_wins_over_persisted(test_task):
    from kiln_ai.datamodel.data_guide import DataGuide

    from app.desktop.studio_server.data_gen_api import _combine_guidance

    saved = DataGuide(parent=test_task, guide="PERSISTED_GUIDE")
    saved.save_to_file()
    reloaded = Task.from_id_and_parent_path(test_task.id, test_task.parent.path)
    assert reloaded is not None

    out = _combine_guidance(
        reloaded,
        None,
        "inputs",
        data_guide_override="OVERRIDE_GUIDE",
    )
    assert out is not None
    assert "OVERRIDE_GUIDE" in out
    assert "PERSISTED_GUIDE" not in out


def test_combine_guidance_stage_hints_keys_are_input_only():
    """The stage hint dict must only have keys for the input-side stages —
    `topics` and `inputs`. The output stage no longer receives the Input Data
    Guide."""
    from app.desktop.studio_server.data_gen_api import _DATA_GUIDE_STAGE_HINTS

    assert set(_DATA_GUIDE_STAGE_HINTS.keys()) == {"topics", "inputs"}


def test_resolve_task_runtime_prompt_falls_back_when_no_default_run_config(
    test_task,
):
    """When the task has no default_run_config_id, the resolver returns
    `task.instruction` verbatim — that's still what the synthesis model sees
    at runtime in this case."""
    from app.desktop.studio_server.data_gen_api import _resolve_task_runtime_prompt

    test_task.default_run_config_id = None
    assert _resolve_task_runtime_prompt(test_task) == test_task.instruction


def test_resolve_task_runtime_prompt_uses_default_run_config_prompt(test_task):
    """When a default run config is set with a kiln_agent prompt_id, the
    resolver returns the prompt that prompt_builder_from_id resolves to —
    matching what the synthesis model actually sees at runtime."""
    from kiln_ai.datamodel.task import TaskRunConfig

    rc = TaskRunConfig(
        name="default",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE,
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=test_task,
    )
    rc.save_to_file()

    test_task.default_run_config_id = rc.id
    test_task.save_to_file()

    from app.desktop.studio_server.data_gen_api import _resolve_task_runtime_prompt

    resolved = _resolve_task_runtime_prompt(test_task)
    # SimplePromptBuilder builds from `task.instruction` plus framing — so the
    # resolved prompt must contain the instruction but isn't necessarily equal
    # to it (the builder may add output-format / system framing).
    assert test_task.instruction in resolved


def test_resolve_task_runtime_prompt_falls_back_when_default_rc_missing(
    test_task,
):
    """If default_run_config_id points to a non-existent run config, fall
    back to task.instruction rather than crashing."""
    test_task.default_run_config_id = "nonexistent-id"

    from app.desktop.studio_server.data_gen_api import _resolve_task_runtime_prompt

    assert _resolve_task_runtime_prompt(test_task) == test_task.instruction


def test_generate_sample_does_not_inject_data_guide(
    mock_task_from_id,
    mock_project_from_id,
    test_task,
    client,
):
    """The output-generation `/generate_sample` endpoint must NOT see the
    Input Data Guide in its prompt, even when one is persisted on the task.
    Output behavior is owned by the task's system prompt + output schema."""
    from kiln_ai.datamodel import DataSource, DataSourceType
    from kiln_ai.datamodel.data_guide import DataGuide
    from kiln_ai.datamodel.datamodel_enums import StructuredOutputMode
    from kiln_ai.datamodel.prompt_id import PromptGenerators
    from kiln_ai.datamodel.task_output import TaskOutput
    from kiln_ai.datamodel.task_run import TaskRun

    saved = DataGuide(parent=test_task, guide="DO_NOT_LEAK_TO_OUTPUT_STAGE")
    saved.save_to_file()

    captured: dict = {}

    def capturing_wrap(instruction: str, guidance: str) -> str:
        captured["wrapped_with"] = guidance
        return instruction

    fake_run = TaskRun(
        input="hi",
        input_source=DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": "x",
                "model_provider": "y",
                "adapter_name": "kiln_data_gen",
            },
        ),
        output=TaskOutput(
            output="ok",
            source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "x",
                    "model_provider": "y",
                    "adapter_name": "kiln_data_gen",
                },
            ),
        ),
    )

    with (
        patch(
            "app.desktop.studio_server.data_gen_api.adapter_for_task"
        ) as mock_adapter_for_task,
        patch(
            "app.desktop.studio_server.data_gen_api.load_skills_for_task",
            return_value=[],
        ),
        patch(
            "app.desktop.studio_server.data_gen_api.wrap_task_with_guidance",
            side_effect=capturing_wrap,
        ),
    ):
        mock_adapter = AsyncMock()
        mock_adapter.invoke = AsyncMock(return_value=fake_run)
        mock_adapter_for_task.return_value = mock_adapter

        response = client.post(
            "/api/projects/test_project/tasks/test_task/generate_sample",
            json={
                "input": "hi",
                "topic_path": ["greetings"],
                "input_model_name": "gpt-4",
                "input_provider": "openai",
                "guidance": None,
                "run_config_properties": {
                    "type": "kiln_agent",
                    "model_name": "gpt-4",
                    "model_provider_name": ModelProviderName.openai.value,
                    "prompt_id": PromptGenerators.SIMPLE.value,
                    "structured_output_mode": StructuredOutputMode.default.value,
                },
            },
        )

    assert response.status_code == 200
    # Either the wrap was never called (no guidance at all) or, if it was
    # called for the topic-path tail, the guidance string must NOT contain the
    # data guide body.
    wrapped = captured.get("wrapped_with", "")
    assert "DO_NOT_LEAK_TO_OUTPUT_STAGE" not in wrapped
    assert "Task Data Guide" not in wrapped


# --- /data_gen_guide_preview endpoint ---


def test_preview_data_gen_guide_success(
    mock_task_from_id,
    mock_project_from_id,
    test_task,
    client,
):
    """Happy path: each preview sample is generated by its own single-shot
    adapter call (num_samples=1) fanned out in parallel, and we get a flat list
    of GuidePreviewSample objects back. Output generation is no longer part of
    the preview — the Input Data Guide is input-only."""
    import json

    def _single_sample_run(value):
        run = MagicMock()
        run.output = MagicMock()
        run.output.output = json.dumps({"generated_samples": [value]})
        return run

    invoke_mock = AsyncMock(
        side_effect=[
            _single_sample_run("input one"),
            _single_sample_run("input two"),
        ]
    )

    with patch(
        "app.desktop.studio_server.data_gen_api.adapter_for_task"
    ) as mock_adapter_for_task:
        mock_adapter = AsyncMock()
        mock_adapter.invoke = invoke_mock
        mock_adapter_for_task.return_value = mock_adapter

        with patch(
            "app.desktop.studio_server.data_gen_api.load_skills_for_task",
            return_value=[],
        ):
            response = client.post(
                "/api/projects/test_project/tasks/test_task/data_gen_guide_preview",
                json={
                    "guide": "# Reference Inputs\n\nSome examples\n\n# Input Guidelines & Rules\n\nSome rules",
                    "run_config_properties": {
                        "type": "kiln_agent",
                        "model_name": "gpt-4",
                        "model_provider_name": ModelProviderName.openai.value,
                        "prompt_id": PromptGenerators.SIMPLE.value,
                        "structured_output_mode": StructuredOutputMode.default.value,
                    },
                    "num_samples": 2,
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body == [
        {"input": "input one"},
        {"input": "input two"},
    ]
    # One single-shot adapter invocation per requested sample (num_samples=2),
    # fanned out in parallel. No separate output pass.
    assert invoke_mock.call_count == 2


def test_preview_data_gen_guide_empty_output_returns_500(
    mock_task_from_id,
    mock_project_from_id,
    client,
):
    """If the input-generating adapter returns no output, surface a 500 with
    a helpful detail rather than crashing later in the pipeline."""
    sample_run = MagicMock()
    sample_run.output = None

    with patch(
        "app.desktop.studio_server.data_gen_api.adapter_for_task"
    ) as mock_adapter_for_task:
        mock_adapter = AsyncMock()
        mock_adapter.invoke = AsyncMock(return_value=sample_run)
        mock_adapter_for_task.return_value = mock_adapter

        with patch(
            "app.desktop.studio_server.data_gen_api.load_skills_for_task",
            return_value=[],
        ):
            response = client.post(
                "/api/projects/test_project/tasks/test_task/data_gen_guide_preview",
                json={
                    "guide": "# Reference Examples\n\nSome examples\n\n# Guidelines & Rules\n\nSome rules",
                    "run_config_properties": {
                        "type": "kiln_agent",
                        "model_name": "gpt-4",
                        "model_provider_name": ModelProviderName.openai.value,
                        "prompt_id": PromptGenerators.SIMPLE.value,
                        "structured_output_mode": StructuredOutputMode.default.value,
                    },
                    "num_samples": 1,
                },
            )

    assert response.status_code == 500
    assert "preview" in response.json()["message"].lower()


def test_preview_data_gen_guide_unparseable_samples_returns_500(
    mock_task_from_id,
    mock_project_from_id,
    client,
):
    """Every single-shot call returned non-JSON, so no sample parsed → 500
    rather than letting JSONDecodeError bubble up. Per-call parse failures are
    tolerated; we only error when nothing usable came back."""
    sample_run = MagicMock()
    sample_run.output = MagicMock()
    sample_run.output.output = "not json at all"

    with patch(
        "app.desktop.studio_server.data_gen_api.adapter_for_task"
    ) as mock_adapter_for_task:
        mock_adapter = AsyncMock()
        mock_adapter.invoke = AsyncMock(return_value=sample_run)
        mock_adapter_for_task.return_value = mock_adapter

        with patch(
            "app.desktop.studio_server.data_gen_api.load_skills_for_task",
            return_value=[],
        ):
            response = client.post(
                "/api/projects/test_project/tasks/test_task/data_gen_guide_preview",
                json={
                    "guide": "# Reference Examples\n\nSome examples\n\n# Guidelines & Rules\n\nSome rules",
                    "run_config_properties": {
                        "type": "kiln_agent",
                        "model_name": "gpt-4",
                        "model_provider_name": ModelProviderName.openai.value,
                        "prompt_id": PromptGenerators.SIMPLE.value,
                        "structured_output_mode": StructuredOutputMode.default.value,
                    },
                    "num_samples": 1,
                },
            )

    assert response.status_code == 500
    assert "preview" in response.json()["message"].lower()


def test_preview_data_gen_guide_surfaces_provider_error(
    mock_task_from_id,
    mock_project_from_id,
    client,
):
    """Sample calls are fanned out with return_exceptions=True, so a provider
    failure never propagates on its own. When every call fails, the provider's
    message (an exhausted key here) must reach the client — a generic "failed to
    generate" sends the user chasing model quality instead of their account."""

    class FakeKilnRunError(Exception):
        def __init__(self, message: str, original: Exception):
            super().__init__(message)
            self.original = original

    provider_error = RuntimeError(
        'OpenrouterException - {"error":{"message":"Key limit exceeded (total limit)","code":403}}'
    )

    with patch(
        "app.desktop.studio_server.data_gen_api.adapter_for_task"
    ) as mock_adapter_for_task:
        mock_adapter = AsyncMock()
        mock_adapter.invoke = AsyncMock(
            side_effect=FakeKilnRunError(
                "An unexpected error occurred.", provider_error
            )
        )
        mock_adapter_for_task.return_value = mock_adapter

        with patch(
            "app.desktop.studio_server.data_gen_api.load_skills_for_task",
            return_value=[],
        ):
            response = client.post(
                "/api/projects/test_project/tasks/test_task/data_gen_guide_preview",
                json={
                    "guide": "# Semantics\n\nSome rules",
                    "run_config_properties": {
                        "type": "kiln_agent",
                        "model_name": "gpt-4",
                        "model_provider_name": ModelProviderName.openai.value,
                        "prompt_id": PromptGenerators.SIMPLE.value,
                        "structured_output_mode": StructuredOutputMode.default.value,
                    },
                    "num_samples": 2,
                },
            )

    assert response.status_code == 500
    message = response.json()["message"]
    assert "Key limit exceeded (total limit)" in message
    # The KilnRunError wrapper's own text is the generic fallback — the useful
    # detail lives on `original`, which is what we must report.
    assert "An unexpected error occurred." not in message


# --- /data_gen_guide_refine endpoint ---


def test_refine_data_gen_guide_success(
    mock_task_from_id,
    client,
):
    """Happy path: the metaprompter LLM returns a structured JSON object with
    a `guide` key; the endpoint returns it as `refined_guide`."""
    import json

    refine_run = MagicMock()
    refine_run.output = MagicMock()
    refine_run.output.output = json.dumps({"guide": "REFINED FULL GUIDE BODY"})

    with patch(
        "app.desktop.studio_server.data_gen_api.adapter_for_task"
    ) as mock_adapter_for_task:
        mock_adapter = AsyncMock()
        mock_adapter.invoke = AsyncMock(return_value=refine_run)
        mock_adapter_for_task.return_value = mock_adapter

        response = client.post(
            "/api/projects/test_project/tasks/test_task/data_gen_guide_refine",
            json={
                "current_guide": (
                    "# Reference Inputs\n\n"
                    "## Example 1\n```input\nx\n```\n\n"
                    "# Input Guidelines & Rules\n\n"
                    "<input_semantic>\n\n## Old\nOld rule.\n\n</input_semantic>"
                ),
                "feedback": "Please make inputs more concise",
                "preview_samples": [
                    {"input": "i1", "looks_good": True},
                    {"input": "i2", "looks_good": False},
                ],
                "run_config_properties": {
                    "type": "kiln_agent",
                    "model_name": "gpt-4",
                    "model_provider_name": ModelProviderName.openai.value,
                    "prompt_id": PromptGenerators.SIMPLE.value,
                    "structured_output_mode": StructuredOutputMode.default.value,
                },
            },
        )

    assert response.status_code == 200
    assert response.json() == {"refined_guide": "REFINED FULL GUIDE BODY"}


def test_refine_data_gen_guide_falls_back_to_current_when_missing_key(
    mock_task_from_id,
    client,
):
    """If the LLM JSON doesn't include the `guide` key, fall back to the
    current guide body rather than returning an empty string."""
    import json

    refine_run = MagicMock()
    refine_run.output = MagicMock()
    refine_run.output.output = json.dumps({"unrelated": "field"})

    with patch(
        "app.desktop.studio_server.data_gen_api.adapter_for_task"
    ) as mock_adapter_for_task:
        mock_adapter = AsyncMock()
        mock_adapter.invoke = AsyncMock(return_value=refine_run)
        mock_adapter_for_task.return_value = mock_adapter

        response = client.post(
            "/api/projects/test_project/tasks/test_task/data_gen_guide_refine",
            json={
                "current_guide": "Original guide",
                "feedback": "fb",
                "preview_samples": [],
                "run_config_properties": {
                    "type": "kiln_agent",
                    "model_name": "gpt-4",
                    "model_provider_name": ModelProviderName.openai.value,
                    "prompt_id": PromptGenerators.SIMPLE.value,
                    "structured_output_mode": StructuredOutputMode.default.value,
                },
            },
        )

    assert response.status_code == 200
    assert response.json() == {"refined_guide": "Original guide"}


def test_refine_data_gen_guide_empty_output_returns_500(
    mock_task_from_id,
    client,
):
    refine_run = MagicMock()
    refine_run.output = None

    with patch(
        "app.desktop.studio_server.data_gen_api.adapter_for_task"
    ) as mock_adapter_for_task:
        mock_adapter = AsyncMock()
        mock_adapter.invoke = AsyncMock(return_value=refine_run)
        mock_adapter_for_task.return_value = mock_adapter

        response = client.post(
            "/api/projects/test_project/tasks/test_task/data_gen_guide_refine",
            json={
                "current_guide": "Original",
                "feedback": "fb",
                "preview_samples": [],
                "run_config_properties": {
                    "type": "kiln_agent",
                    "model_name": "gpt-4",
                    "model_provider_name": ModelProviderName.openai.value,
                    "prompt_id": PromptGenerators.SIMPLE.value,
                    "structured_output_mode": StructuredOutputMode.default.value,
                },
            },
        )

    assert response.status_code == 500
    assert "refine" in response.json()["message"].lower()


# --- Kiln Pro batch generation ----------------------------------------------


def _batch_rcp() -> KilnAgentRunConfigProperties:
    return KilnAgentRunConfigProperties(
        model_name="gpt-4",
        model_provider_name=ModelProviderName.openai,
        prompt_id=PromptGenerators.SIMPLE,
        structured_output_mode=StructuredOutputMode.json_schema,
    )


_RCP_JSON = {
    "type": "kiln_agent",
    "model_name": "gpt-4",
    "model_provider_name": ModelProviderName.openai.value,
    "prompt_id": PromptGenerators.SIMPLE.value,
    "structured_output_mode": StructuredOutputMode.json_schema.value,
}


def _noop_spawn(coro):
    """Drop-in for _spawn_batch_task that doesn't run the background job.

    Endpoint tests use this to test the synchronous responsibilities (job
    created, registered, status served, scoped) without racing the async job
    to completion — that path is covered deterministically by the
    _run_*_batch_job tests. Closing the coroutine avoids a
    'coroutine was never awaited' warning.
    """
    coro.close()


def _mk_job(job_id: str, status="complete") -> _BatchJob:
    return _BatchJob(
        job_id=job_id,
        project_id="p",
        task_id="t",
        kind="inputs",
        total=1,
        status=status,
    )


def test_register_batch_job_evicts_oldest_finished():
    _batch_jobs.clear()
    try:
        # Fill past the cap with finished jobs.
        for i in range(_MAX_BATCH_JOBS):
            _register_batch_job(_mk_job(f"done-{i}", status="complete"))
        assert len(_batch_jobs) == _MAX_BATCH_JOBS
        _register_batch_job(_mk_job("newest", status="complete"))
        # Back at the cap, oldest evicted, newest kept.
        assert len(_batch_jobs) == _MAX_BATCH_JOBS
        assert "done-0" not in _batch_jobs
        assert "newest" in _batch_jobs
    finally:
        _batch_jobs.clear()


def test_register_batch_job_never_evicts_running_jobs():
    _batch_jobs.clear()
    try:
        # A registry full of still-running jobs cannot be trimmed.
        for i in range(_MAX_BATCH_JOBS):
            _register_batch_job(_mk_job(f"run-{i}", status="running"))
        _register_batch_job(_mk_job("newest", status="running"))
        # No running job is dropped, even over the cap.
        assert len(_batch_jobs) == _MAX_BATCH_JOBS + 1
        assert all(k in _batch_jobs for k in ["run-0", "newest"])
    finally:
        _batch_jobs.clear()


async def test_run_inputs_batch_job_success(test_task):
    prompts = [f"p{i}" for i in range(5)]
    job = _BatchJob(
        job_id="j", project_id="p", task_id="t", kind="inputs", total=len(prompts)
    )

    def fake(project, task, rcp, data_guide, prompt):
        return f"in::{prompt}"

    with patch(
        "app.desktop.studio_server.data_gen_api._generate_one_input",
        new=AsyncMock(side_effect=fake),
    ):
        await _run_inputs_batch_job(
            job, MagicMock(), test_task, _batch_rcp(), None, prompts
        )

    assert job.status == "complete"
    assert job.completed == 5
    assert job.errors == 0
    for i, p in enumerate(prompts):
        assert job.results[i]["index"] == i
        assert job.results[i]["input"] == f"in::{p}"
        assert job.results[i]["error"] is None


async def test_run_inputs_batch_job_partial_failure(test_task):
    prompts = ["p0", "p1", "p2", "p3"]
    job = _BatchJob(
        job_id="j", project_id="p", task_id="t", kind="inputs", total=len(prompts)
    )

    def fake(project, task, rcp, data_guide, prompt):
        if prompt == "p2":
            raise ValueError("boom")
        return f"in::{prompt}"

    with patch(
        "app.desktop.studio_server.data_gen_api._generate_one_input",
        new=AsyncMock(side_effect=fake),
    ):
        await _run_inputs_batch_job(
            job, MagicMock(), test_task, _batch_rcp(), None, prompts
        )

    # One failure, but not all — still completes.
    assert job.status == "complete"
    assert job.errors == 1
    assert job.results[2]["input"] is None
    assert job.results[2]["error"] == "boom"
    assert job.results[0]["input"] == "in::p0"


async def test_run_inputs_batch_job_all_fail(test_task):
    prompts = ["p0", "p1"]
    job = _BatchJob(
        job_id="j", project_id="p", task_id="t", kind="inputs", total=len(prompts)
    )

    def fake(*args, **kwargs):
        raise ValueError("nope")

    with patch(
        "app.desktop.studio_server.data_gen_api._generate_one_input",
        new=AsyncMock(side_effect=fake),
    ):
        await _run_inputs_batch_job(
            job, MagicMock(), test_task, _batch_rcp(), None, prompts
        )

    assert job.status == "error"
    assert job.errors == 2


async def test_run_outputs_batch_job_success(test_task, mock_task_run):
    items = [GenerateOutputsBatchItem(index=i, input=f"input {i}") for i in range(4)]
    job = _BatchJob(
        job_id="o", project_id="p", task_id="t", kind="outputs", total=len(items)
    )

    def fake(*args, **kwargs):
        return mock_task_run

    with patch(
        "app.desktop.studio_server.data_gen_api._generate_one_output",
        new=AsyncMock(side_effect=fake),
    ):
        await _run_outputs_batch_job(
            job, "p", "t", items, "gpt-4", "openai", _batch_rcp(), None, None, None
        )

    assert job.status == "complete"
    assert job.completed == 4
    assert job.errors == 0
    for r in job.results:
        assert r["task_run"] is mock_task_run


async def test_generate_one_input_extracts_generated_input(
    test_project, test_task, data_source
):
    run = TaskRun(
        output=TaskOutput(
            output=json.dumps({"generated_input": "the generated input"}),
            source=data_source,
        ),
        input="x",
        input_source=data_source,
        parent=test_task,
    )
    with patch(
        "app.desktop.studio_server.data_gen_api.adapter_for_task"
    ) as mock_adapter_for_task:
        adapter = AsyncMock()
        adapter.invoke = AsyncMock(return_value=run)
        mock_adapter_for_task.return_value = adapter

        result = await _generate_one_input(
            test_project, test_task, _batch_rcp(), None, "a prompt"
        )

    assert result == "the generated input"


def test_generate_inputs_batch_endpoint(
    mock_project_from_id, mock_task_from_id, client, data_source, test_task
):
    run = TaskRun(
        output=TaskOutput(
            output=json.dumps({"generated_input": "gen input"}),
            source=data_source,
        ),
        input="x",
        input_source=data_source,
        parent=test_task,
    )
    with (
        patch(
            "app.desktop.studio_server.data_gen_api.adapter_for_task"
        ) as mock_adapter_for_task,
        patch(
            "app.desktop.studio_server.data_gen_api._spawn_batch_task",
            side_effect=_noop_spawn,
        ),
    ):
        adapter = AsyncMock()
        adapter.invoke = AsyncMock(return_value=run)
        mock_adapter_for_task.return_value = adapter

        start = client.post(
            "/api/projects/proj-ID/tasks/task-ID/generate_inputs_batch",
            json={
                "prompts": ["a", "b", "c"],
                "run_config_properties": _RCP_JSON,
            },
        )
        assert start.status_code == 200, start.text
        job_id = start.json()["job_id"]
        assert job_id

        # The job is registered synchronously; the status endpoint serves it.
        # Completion + results are covered by test_run_inputs_batch_job_success.
        status = client.get(
            f"/api/projects/proj-ID/tasks/task-ID/generate_inputs_batch/{job_id}"
        )
        assert status.status_code == 200, status.text
        data = status.json()

    assert data["total"] == 3
    assert data["model_name"] == "gpt-4"
    assert data["model_provider"] == "openai"


def test_inputs_batch_status_unknown_job_404(mock_task_from_id, client):
    resp = client.get(
        "/api/projects/proj-ID/tasks/task-ID/generate_inputs_batch/does-not-exist"
    )
    assert resp.status_code == 404


def test_inputs_batch_status_wrong_scope_404(
    mock_project_from_id, mock_task_from_id, client, data_source, test_task
):
    # Start a job under one task, then poll it under a different task id.
    run = TaskRun(
        output=TaskOutput(
            output=json.dumps({"generated_input": "gen input"}),
            source=data_source,
        ),
        input="x",
        input_source=data_source,
        parent=test_task,
    )
    with patch(
        "app.desktop.studio_server.data_gen_api.adapter_for_task"
    ) as mock_adapter_for_task:
        adapter = AsyncMock()
        adapter.invoke = AsyncMock(return_value=run)
        mock_adapter_for_task.return_value = adapter

        start = client.post(
            "/api/projects/proj-ID/tasks/task-ID/generate_inputs_batch",
            json={
                "prompts": ["a"],
                "run_config_properties": _RCP_JSON,
            },
        )
        job_id = start.json()["job_id"]

    # Same job_id, different task path → 404 (don't serve by job_id alone).
    resp = client.get(
        f"/api/projects/proj-ID/tasks/OTHER-task/generate_inputs_batch/{job_id}"
    )
    assert resp.status_code == 404


def test_generate_outputs_batch_endpoint(mock_task_from_id, client, mock_task_run):
    with (
        patch(
            "app.desktop.studio_server.data_gen_api.adapter_for_task"
        ) as mock_adapter_for_task,
        patch(
            "app.desktop.studio_server.data_gen_api._spawn_batch_task",
            side_effect=_noop_spawn,
        ),
    ):
        adapter = AsyncMock()
        adapter.invoke = AsyncMock(return_value=mock_task_run)
        mock_adapter_for_task.return_value = adapter

        start = client.post(
            "/api/projects/proj-ID/tasks/task-ID/generate_outputs_batch",
            json={
                "items": [
                    {"index": 0, "input": "i0"},
                    {"index": 1, "input": "i1"},
                ],
                "input_model_name": "gpt-4",
                "input_provider": "openai",
                "run_config_properties": _RCP_JSON,
            },
        )
        assert start.status_code == 200, start.text
        job_id = start.json()["job_id"]

        # Job registered synchronously; completion is covered by
        # test_run_outputs_batch_job_success.
        status = client.get(
            f"/api/projects/proj-ID/tasks/task-ID/generate_outputs_batch/{job_id}"
        )
        assert status.status_code == 200, status.text
        data = status.json()

    assert data["total"] == 2


def test_batch_jobs_registry_is_populated(
    mock_project_from_id, mock_task_from_id, client, data_source, test_task
):
    run = TaskRun(
        output=TaskOutput(
            output=json.dumps({"generated_input": "gen input"}),
            source=data_source,
        ),
        input="x",
        input_source=data_source,
        parent=test_task,
    )
    with (
        patch(
            "app.desktop.studio_server.data_gen_api.adapter_for_task"
        ) as mock_adapter_for_task,
        patch(
            "app.desktop.studio_server.data_gen_api._spawn_batch_task",
            side_effect=_noop_spawn,
        ),
    ):
        adapter = AsyncMock()
        adapter.invoke = AsyncMock(return_value=run)
        mock_adapter_for_task.return_value = adapter

        start = client.post(
            "/api/projects/proj-ID/tasks/task-ID/generate_inputs_batch",
            json={
                "prompts": ["a", "b"],
                "run_config_properties": _RCP_JSON,
            },
        )
        job_id = start.json()["job_id"]

    # Registration is synchronous in the handler, before the job is spawned.
    assert job_id in _batch_jobs
    assert _batch_jobs[job_id].kind == "inputs"


async def _instruction_for_one_input(project, task, data_guide, data_source):
    """Run _generate_one_input and return the instruction the generator task was
    built with, so we can assert on what the model actually reads."""
    run = TaskRun(
        output=TaskOutput(
            output=json.dumps({"generated_input": "x"}), source=data_source
        ),
        input="x",
        input_source=data_source,
        parent=task,
    )
    with patch(
        "app.desktop.studio_server.data_gen_api.adapter_for_task"
    ) as mock_adapter_for_task:
        adapter = AsyncMock()
        adapter.invoke = AsyncMock(return_value=run)
        mock_adapter_for_task.return_value = adapter

        await _generate_one_input(project, task, _batch_rcp(), data_guide, "a prompt")
        # First positional arg to adapter_for_task is the DataGenSingleInputTask.
        return mock_adapter_for_task.call_args.args[0].instruction


async def test_generate_one_input_omits_data_guide_when_toggle_off(
    test_project, test_task, data_source
):
    """The batch flow sends None when "Use Data Guide" is off. That must mean
    "don't use it" — not "fall back to the task's saved guide"."""
    # The task HAS a saved guide, so a fallback would silently include it.
    DataGuide(guide="Emails are terse.", parent=test_task).save_to_file()

    instruction = await _instruction_for_one_input(
        test_project, test_task, None, data_source
    )

    assert "<task_data_guide>" not in instruction
    assert "Emails are terse." not in instruction


async def test_generate_one_input_includes_data_guide_when_provided(
    test_project, test_task, data_source
):
    instruction = await _instruction_for_one_input(
        test_project, test_task, "Emails are terse.", data_source
    )

    assert "<task_data_guide>" in instruction
    assert "Emails are terse." in instruction
    # The guide arrives with its explanatory context, not as a bare blob.
    assert "# Task Data Guide" in instruction
    assert "Authority cascade" in instruction


async def _one_output_tags(session_id, mock_task_from_id, test_task, data_source):
    run = TaskRun(
        output=TaskOutput(output="out", source=data_source),
        input="in",
        input_source=data_source,
        parent=test_task,
    )
    with patch(
        "app.desktop.studio_server.data_gen_api.adapter_for_task"
    ) as mock_adapter_for_task:
        adapter = AsyncMock()
        adapter.invoke = AsyncMock(return_value=run)
        mock_adapter_for_task.return_value = adapter

        out = await _generate_one_output(
            "p",
            "t",
            "an input",
            "gpt-4",
            "openai",
            _batch_rcp(),
            None,
            session_id,
            None,
        )
    return out.tags


async def test_generate_one_output_tags_with_session(
    mock_task_from_id, test_task, data_source
):
    """Batch-generated runs carry the session tag, so a Kiln Pro batch is
    groupable in the dataset the same way a legacy synth session is."""
    tags = await _one_output_tags("abc123", mock_task_from_id, test_task, data_source)

    assert "synthetic" in tags
    assert "synthetic_session_abc123" in tags


async def test_generate_one_output_omits_session_tag_when_absent(
    mock_task_from_id, test_task, data_source
):
    tags = await _one_output_tags(None, mock_task_from_id, test_task, data_source)

    assert tags == ["synthetic"]
