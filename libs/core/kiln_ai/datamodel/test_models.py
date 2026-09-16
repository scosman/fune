import json
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from kiln_ai.datamodel import (
    DataSource,
    DataSourceType,
    Finetune,
    Project,
    Prompt,
    Task,
    TaskOutput,
    TaskRun,
)
from kiln_ai.datamodel.basemodel import generate_model_id, name_validator
from kiln_ai.datamodel.datamodel_enums import ChatStrategy, TurnMode
from kiln_ai.datamodel.test_json_schema import json_joke_schema


@pytest.fixture
def test_project_file(tmp_path):
    test_file_path = tmp_path / "project.kiln"
    data = {"v": 1, "name": "Test Project", "model_type": "project"}

    with open(test_file_path, "w") as file:
        json.dump(data, file, indent=4)

    return test_file_path


@pytest.fixture
def test_task_file(tmp_path):
    test_file_path = tmp_path / "task.kiln"
    data = {
        "v": 1,
        "name": "Test Task",
        "instruction": "Test Instruction",
        "model_type": "task",
    }

    with open(test_file_path, "w") as file:
        json.dump(data, file, indent=4)

    return test_file_path


def test_load_from_file(test_project_file):
    project = Project.load_from_file(test_project_file)
    assert project.v == 1
    assert project.name == "Test Project"
    assert project.path == test_project_file


def test_project_init():
    project = Project(name="test")
    assert project.name == "test"


def test_save_to_file(test_project_file):
    project = Project(
        name="Test Project", description="Test Description", path=test_project_file
    )
    project.save_to_file()

    with open(test_project_file, "r") as file:
        data = json.load(file)

    assert data["v"] == 1
    assert data["name"] == "Test Project"
    assert data["description"] == "Test Description"


def test_save_to_file_non_ascii(test_project_file):
    project = Project(
        name="Test Project", description="Chúc mừng!", path=test_project_file
    )
    project.save_to_file()

    with open(test_project_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["v"] == 1
    assert data["name"] == "Test Project"
    assert data["description"] == "Chúc mừng!"


def test_task_defaults():
    task = Task(name="Test Task", instruction="Test Instruction")
    assert task.description is None


def test_task_serialization(test_project_file):
    project = Project.load_from_file(test_project_file)
    task = Task(
        parent=project,
        name="Test Task",
        description="Test Description",
        instruction="Test Base Task Instruction",
        thinking_instruction="Test Thinking Instruction",
    )
    assert task._loaded_from_file is False

    task.save_to_file()

    parsed_task = Task.all_children_of_parent_path(test_project_file)[0]
    assert parsed_task.name == "Test Task"
    assert parsed_task.description == "Test Description"
    assert parsed_task.instruction == "Test Base Task Instruction"
    assert parsed_task.thinking_instruction == "Test Thinking Instruction"
    assert parsed_task._loaded_from_file is True

    # Confirm the local property is not persisted to disk
    json_data = json.loads(parsed_task.path.read_text())
    assert "_loaded_from_file" not in json_data


def test_save_to_file_without_path():
    project = Project(name="Test Project")
    with pytest.raises(ValueError):
        project.save_to_file()


def test_name_validation():
    Project(name="Test Project")
    Project(name="Te st_Proj- 1234567890")
    Project(name=("a" * 120))  # longest

    # a string with 120 characters

    with pytest.raises(ValueError):
        Project(name="Test Project!")
        Project(name="Test.Project")
        Project(name=("a" * 121))  # too long
        Project(name=("a"))  # too short


def test_auto_type_name():
    model = Project(name="Test Project")
    assert model.model_type == "project"


def test_load_tasks(test_project_file):
    # Set up a project model
    project = Project.load_from_file(test_project_file)

    # Set up multiple task models under the project
    task1 = Task(parent=project, name="Task1", instruction="Task 1 instruction")
    task2 = Task(parent=project, name="Task2", instruction="Task 2 instruction")
    task3 = Task(parent=project, name="Task3", instruction="Task 3 instruction")

    # Ensure the tasks are saved correctly
    task1.save_to_file()
    task2.save_to_file()
    task3.save_to_file()

    # Load tasks from the project
    tasks = project.tasks()

    # Verify that all tasks are loaded correctly
    assert len(tasks) == 3
    names = [task.name for task in tasks]
    assert "Task1" in names
    assert "Task2" in names
    assert "Task3" in names
    assert all(task.model_type == "task" for task in tasks)
    assert all(task.instruction != "" for task in tasks)


# verify no error on non-saved model
def test_load_children_no_path():
    project = Project(name="Test Project")
    assert len(project.tasks()) == 0


def test_check_model_type(test_project_file, test_task_file):
    project = Project.load_from_file(test_project_file)
    task = Task.load_from_file(test_task_file)
    assert project.model_type == "project"
    assert task.model_type == "task"
    assert task.instruction == "Test Instruction"

    with pytest.raises(ValueError):
        project = Project.load_from_file(test_task_file)

    with pytest.raises(ValueError):
        task = Task.load_from_file(test_project_file)


def test_task_output_schema(tmp_path):
    path = tmp_path / "task.kiln"
    task = Task(name="Test Task", path=path, instruction="Test Instruction")
    task.save_to_file()
    assert task.output_schema() is None
    task = Task(
        name="Test Task",
        instruction="Test Instruction",
        output_json_schema=json_joke_schema,
        input_json_schema=json_joke_schema,
        path=path,
    )
    task.save_to_file()
    schemas = [task.output_schema(), task.input_schema()]
    for schema in schemas:
        assert schema is not None
        assert schema["properties"]["setup"]["type"] == "string"
        assert schema["properties"]["punchline"]["type"] == "string"
        assert schema["properties"]["rating"] is not None

    # Not json schema
    with pytest.raises(ValidationError):
        task = Task(name="Test Task", output_json_schema="hello", path=path)
    with pytest.raises(ValidationError):
        task = Task(name="Test Task", output_json_schema='{"asdf":{}}', path=path)
    with pytest.raises(ValidationError):
        task = Task(name="Test Task", output_json_schema="{'asdf':{}}", path=path)
    with pytest.raises(ValidationError):
        task = Task(name="Test Task", input_json_schema="{asdf", path=path)


def test_task_run_intermediate_outputs():
    # Create a basic task output
    output = TaskOutput(
        output="test output",
        source=DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": "test-model",
                "model_provider": "test-provider",
                "adapter_name": "test-adapter",
            },
        ),
    )

    # Test valid intermediate outputs
    task_run = TaskRun(
        input="test input",
        input_source=DataSource(
            type=DataSourceType.human,
            properties={"created_by": "test-user"},
        ),
        output=output,
        intermediate_outputs={
            "cot": "chain of thought output",
            "draft": "draft output",
        },
    )
    assert task_run.intermediate_outputs == {
        "cot": "chain of thought output",
        "draft": "draft output",
    }


def test_finetune_basic():
    # Test basic initialization
    finetune = Finetune(
        name="test-finetune",
        provider="openai",
        base_model_id="gpt-3.5-turbo",
        dataset_split_id="dataset-123",
        train_split_name="train",
        system_message="Test system message",
    )
    assert finetune.name == "test-finetune"
    assert finetune.provider == "openai"
    assert finetune.base_model_id == "gpt-3.5-turbo"
    assert finetune.dataset_split_id == "dataset-123"
    assert finetune.train_split_name == "train"
    assert finetune.provider_id is None
    assert finetune.parameters == {}
    assert finetune.description is None


def test_finetune_full():
    # Test with all fields populated
    finetune = Finetune(
        name="test-finetune",
        description="Test description",
        provider="openai",
        base_model_id="gpt-3.5-turbo",
        provider_id="ft-abc123",
        dataset_split_id="dataset-123",
        train_split_name="train",
        system_message="Test system message",
        parameters={
            "epochs": 3,
            "learning_rate": 0.1,
            "batch_size": 4,
            "use_fp16": True,
            "model_suffix": "-v1",
        },
    )
    assert finetune.description == "Test description"
    assert finetune.provider_id == "ft-abc123"
    assert finetune.parameters == {
        "epochs": 3,
        "learning_rate": 0.1,
        "batch_size": 4,
        "use_fp16": True,
        "model_suffix": "-v1",
    }
    assert finetune.system_message == "Test system message"


def test_finetune_parent_task():
    # Test parent_task() method
    task = Task(name="Test Task", instruction="Test instruction")
    finetune = Finetune(
        name="test-finetune",
        provider="openai",
        base_model_id="gpt-3.5-turbo",
        parent=task,
        dataset_split_id="dataset-123",
        train_split_name="train",
        system_message="Test system message",
    )

    assert finetune.parent_task() == task

    # Test with no parent
    finetune_no_parent = Finetune(
        name="test-finetune",
        provider="openai",
        base_model_id="gpt-3.5-turbo",
        dataset_split_id="dataset-123",
        train_split_name="train",
        system_message="Test system message",
    )
    assert finetune_no_parent.parent_task() is None


def test_finetune_model_id(tmp_path):
    project_path = tmp_path / "project.kiln"
    project = Project(name="Test Project", path=str(project_path))
    project.save_to_file()

    task = Task(name="Test Task", instruction="Test instruction", parent=project)
    task.save_to_file()

    finetune = Finetune(
        name="test-finetune",
        provider="openai",
        base_model_id="gpt-3.5-turbo",
        parent=task,
        dataset_split_id="dataset-123",
        train_split_name="train",
        system_message="Test system message",
    )

    expected_id = f"{project.id}::{task.id}::{finetune.id}"
    assert finetune.nested_id() == expected_id

    finetune_no_task = Finetune(
        name="test-finetune",
        provider="openai",
        base_model_id="gpt-3.5-turbo",
        dataset_split_id="dataset-123",
        train_split_name="train",
        system_message="Test system message",
    )
    with pytest.raises(ValueError, match="Finetune must have a parent task"):
        finetune_no_task.nested_id()

    task_no_project = Task(name="Test Task", instruction="Test instruction")
    finetune_no_project = Finetune(
        name="test-finetune",
        provider="openai",
        base_model_id="gpt-3.5-turbo",
        parent=task_no_project,
        dataset_split_id="dataset-123",
        train_split_name="train",
        system_message="Test system message",
    )
    with pytest.raises(ValueError, match="Finetune must have a parent project"):
        finetune_no_project.nested_id()


def test_finetune_parameters_validation():
    # Test that parameters only accept valid types
    with pytest.raises(ValidationError):
        Finetune(
            name="test-finetune",
            provider="openai",
            base_model_id="gpt-3.5-turbo",
            parameters={"invalid": [1, 2, 3]},  # Lists are not allowed
        )


def test_task_run_input_source_validation(tmp_path):
    # Setup basic output for TaskRun creation
    output = TaskOutput(
        output="test output",
        source=DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": "test-model",
                "model_provider": "test-provider",
                "adapter_name": "test-adapter",
            },
        ),
    )

    project_path = tmp_path / "project.kiln"
    project = Project(name="Test Project", path=project_path)
    project.save_to_file()
    task = Task(name="Test Task", instruction="Test Instruction", parent=project)
    task.save_to_file()

    # Test 1: Creating without input_source should work when strict mode is off
    task_run = TaskRun(
        input="test input",
        output=output,
    )
    task_run.parent = task
    assert task_run.input_source is None

    # Save for later usage
    task_run.save_to_file()
    task_missing_input_source = task_run.path

    # Test 2: Creating with input_source should work when strict mode is off
    task_run = TaskRun(
        input="test input 2",
        input_source=DataSource(
            type=DataSourceType.human,
            properties={"created_by": "test-user"},
        ),
        output=output,
    )
    assert task_run.input_source is not None

    # Test 3: Creating without input_source should fail when strict mode is on
    with patch("kiln_ai.datamodel.task_run.strict_mode", return_value=True):
        with pytest.raises(ValueError) as exc_info:
            task_run = TaskRun(
                input="test input 3",
                output=output,
            )
        assert "input_source is required when strict mode is enabled" in str(
            exc_info.value
        )

        # Test 4: Loading from disk should work without input_source, even with strict mode on
        assert os.path.exists(task_missing_input_source)
        task_run = TaskRun.load_from_file(task_missing_input_source)
        assert task_run.input_source is None


def test_task_output_source_validation(tmp_path):
    # Setup basic output source for validation
    output_source = DataSource(
        type=DataSourceType.synthetic,
        properties={
            "model_name": "test-model",
            "model_provider": "test-provider",
            "adapter_name": "test-adapter",
        },
    )

    project_path = tmp_path / "project.kiln"
    project = Project(name="Test Project", path=project_path)
    project.save_to_file()
    task = Task(name="Test Task", instruction="Test Instruction", parent=project)
    task.save_to_file()

    # Test 1: Creating without source should work when strict mode is off
    task_output = TaskOutput(
        output="test output",
    )
    assert task_output.source is None

    # Save for later usage
    task_run = TaskRun(
        input="test input",
        input_source=output_source,
        output=task_output,
    )
    task_run.parent = task
    task_run.save_to_file()
    task_missing_output_source = task_run.path

    # Test 2: Creating with source should work when strict mode is off
    task_output = TaskOutput(
        output="test output 2",
        source=output_source,
    )
    assert task_output.source is not None

    # Test 3: Creating without source should fail when strict mode is on
    with patch("kiln_ai.datamodel.task_output.strict_mode", return_value=True):
        with pytest.raises(ValueError) as exc_info:
            task_output = TaskOutput(
                output="test output 3",
            )
        assert "Output source is required when strict mode is enabled" in str(
            exc_info.value
        )

        # Test 4: Loading from disk should work without source, even with strict mode on
        assert os.path.exists(task_missing_output_source)
        task_run = TaskRun.load_from_file(task_missing_output_source)
        assert task_run.output.source is None


def test_data_guide():
    from kiln_ai.datamodel.data_guide import DataGuide

    body = (
        "# Reference Examples\n\n"
        "## Example 1\n```input\ntest\n```\n\n```output\nresult\n```\n\n"
        "# Guidelines & Rules\n\n"
        "<output_semantic>\n\n## Cholesterol\nIf cholesterol is high, never have low LDL.\n\n</output_semantic>"
    )
    guide = DataGuide(guide=body)
    assert "test" in guide.guide
    assert "cholesterol" in guide.guide.lower()

    # Serializes correctly
    data = guide.model_dump()
    assert "test" in data["guide"]
    assert "cholesterol" in data["guide"].lower()

    # Deserializes correctly
    restored = DataGuide.model_validate(data)
    assert restored.guide == guide.guide

    # Field defaults to empty — a blank guide is permitted at the model layer
    # (validation lives at the API).
    blank = DataGuide()
    assert blank.guide == ""


def test_task_data_guide_accessors(tmp_path):
    """Task.data_guides() lists child guides; current_data_guide() returns
    the single one (or None) — the canonical accessor used by the data-gen
    APIs."""
    from kiln_ai.datamodel import Project, Task
    from kiln_ai.datamodel.data_guide import DataGuide

    project = Project(name="P", path=tmp_path / "p" / "project.kiln")
    project.path.parent.mkdir()
    project.save_to_file()

    task = Task(name="T", instruction="t", parent=project)
    task.save_to_file()

    # No guides yet
    assert task.data_guides() == []
    assert task.current_data_guide() is None

    # Save one DataGuide as a child of the task
    guide = DataGuide(parent=task, guide="some guide body")
    guide.save_to_file()

    reloaded = Task.from_id_and_parent_path(task.id, project.path)
    assert reloaded is not None
    guides = reloaded.data_guides()
    assert len(guides) == 1
    assert guides[0].guide == "some guide body"

    current = reloaded.current_data_guide()
    assert current is not None
    assert current.id == guide.id
    assert current.guide == "some guide body"


def test_task_run_tags_validation():
    # Setup basic output for TaskRun creation
    output = TaskOutput(
        output="test output",
        source=DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": "test-model",
                "model_provider": "test-provider",
                "adapter_name": "test-adapter",
            },
        ),
    )

    # Test 1: Valid tags should work
    task_run = TaskRun(
        input="test input",
        output=output,
        tags=["test_tag", "another_tag", "tag123"],
    )
    assert task_run.tags == ["test_tag", "another_tag", "tag123"]

    # Test 2: Empty list of tags should work
    task_run = TaskRun(
        input="test input",
        output=output,
        tags=[],
    )
    assert task_run.tags == []

    # Test 3: Empty string tag should fail
    with pytest.raises(ValueError) as exc_info:
        TaskRun(
            input="test input",
            output=output,
            tags=["valid_tag", ""],
        )
    assert "Tags cannot be empty strings" in str(exc_info.value)

    # Test 4: Tag with spaces should fail
    with pytest.raises(ValueError) as exc_info:
        TaskRun(
            input="test input",
            output=output,
            tags=["valid_tag", "invalid tag"],
        )
    assert "Tags cannot contain spaces. Try underscores." in str(exc_info.value)


def test_prompt_validation():
    prompt = Prompt(name="Test Prompt Name", prompt="Test Prompt")
    assert prompt.name == "Test Prompt Name"
    assert prompt.prompt == "Test Prompt"

    with pytest.raises(ValidationError):
        Prompt(name="Test Prompt")

    with pytest.raises(ValidationError):
        Prompt(name="Test Prompt", prompt=None)

    with pytest.raises(ValidationError):
        Prompt(name="Test Prompt", prompt="")

    with pytest.raises(ValidationError):
        Prompt(prompt="Test Prompt")


def test_prompt_parent_task():
    task = Task(name="Test Task", instruction="Test Instruction")
    prompt = Prompt(name="Test Prompt", prompt="Test Prompt", parent=task)
    assert prompt.parent == task


@pytest.mark.parametrize(
    "thinking_instructions,data_strategy,should_raise,expected_message",
    [
        # Test 1: Valid case - no thinking instructions with final_only
        (
            None,
            ChatStrategy.single_turn,
            False,
            None,
        ),
        # Test 2: Valid case - thinking instructions with final_and_intermediate
        (
            "Think step by step",
            ChatStrategy.two_message_cot_legacy,
            False,
            None,
        ),
        # Test 3: Valid case - no thinking instructions with final_and_intermediate_r1_compatible
        (
            None,
            ChatStrategy.single_turn_r1_thinking,
            False,
            None,
        ),
        # Test 4: Invalid case - thinking instructions with final_only
        (
            "Think step by step",
            ChatStrategy.single_turn,
            True,
            "Thinking instructions can only be used when data_strategy is",
        ),
        # Test 5: Invalid case - no thinking instructions with final_and_intermediate
        (
            None,
            ChatStrategy.two_message_cot_legacy,
            True,
            "Thinking instructions are required when data_strategy is",
        ),
        # Test 6: Invalid case - thinking instructions with final_and_intermediate_r1_compatible
        (
            "Think step by step",
            ChatStrategy.single_turn_r1_thinking,
            True,
            "Thinking instructions can only be used when data_strategy is",
        ),
        # Test 7: new COT format
        (
            "Think step by step",
            ChatStrategy.two_message_cot,
            False,
            None,
        ),
        # Test 8: new COT format
        (
            None,
            ChatStrategy.two_message_cot,
            True,
            "Thinking instructions are required when data_strategy is",
        ),
    ],
)
def test_finetune_thinking_instructions_validation(
    thinking_instructions, data_strategy, should_raise, expected_message
):
    base_params = {
        "name": "test-finetune",
        "provider": "openai",
        "base_model_id": "gpt-3.5-turbo",
        "dataset_split_id": "split1",
        "system_message": "test message",
        "data_strategy": data_strategy,
    }

    if thinking_instructions is not None:
        base_params["thinking_instructions"] = thinking_instructions

    if should_raise:
        with pytest.raises(ValueError) as exc_info:
            Finetune(**base_params)
        assert expected_message in str(exc_info.value)
    else:
        finetune = Finetune(**base_params)
        assert finetune.thinking_instructions == thinking_instructions
        assert finetune.data_strategy == data_strategy


@pytest.mark.parametrize(
    "intermediate_outputs,expected",
    [
        # No intermediate outputs
        (None, False),
        # Empty intermediate outputs
        ({}, False),
        # Only chain_of_thought
        ({"chain_of_thought": "thinking process"}, True),
        # Only reasoning
        ({"reasoning": "reasoning process"}, True),
        # Both chain_of_thought and reasoning
        (
            {"chain_of_thought": "thinking process", "reasoning": "reasoning process"},
            True,
        ),
        # Other intermediate outputs but no thinking data
        ({"other_output": "some data"}, False),
        # Mixed other outputs with thinking data
        ({"chain_of_thought": "thinking process", "other_output": "some data"}, True),
    ],
)
def test_task_run_has_thinking_training_data(intermediate_outputs, expected):
    task_run = TaskRun(
        input="test input",
        output=TaskOutput(output="test output"),
        intermediate_outputs=intermediate_outputs,
    )
    assert task_run.has_thinking_training_data() == expected


@pytest.mark.parametrize(
    "intermediate_outputs,expected",
    [
        # No intermediate outputs
        (None, None),
        # Empty intermediate outputs
        ({}, None),
        # Only chain_of_thought
        ({"chain_of_thought": "thinking process"}, "thinking process"),
        # Only reasoning
        ({"reasoning": "reasoning process"}, "reasoning process"),
        # Both chain_of_thought and reasoning (should return reasoning as it's checked first)
        (
            {"chain_of_thought": "thinking process", "reasoning": "reasoning process"},
            "reasoning process",
        ),
        # Other intermediate outputs but no thinking data
        ({"other_output": "some data"}, None),
        # Mixed other outputs with thinking data
        (
            {"chain_of_thought": "thinking process", "other_output": "some data"},
            "thinking process",
        ),
    ],
)
def test_task_run_thinking_training_data(intermediate_outputs, expected):
    task_run = TaskRun(
        input="test input",
        output=TaskOutput(output="test output"),
        intermediate_outputs=intermediate_outputs,
    )
    assert task_run.thinking_training_data() == expected


def test_chat_strategy_enum():
    # This has to align to the old FinetuneDataStrategy enum
    assert ChatStrategy.single_turn == "final_only"
    assert ChatStrategy.two_message_cot_legacy == "final_and_intermediate"
    assert (
        ChatStrategy.single_turn_r1_thinking == "final_and_intermediate_r1_compatible"
    )


def test_generate_model_id():
    model_id = generate_model_id()
    assert len(model_id) == 12
    # check it is a valid name - as we typically use model ids in filenames on FS
    validator = name_validator(min_length=1, max_length=12)
    validator(model_id)


# project and task fixture
@pytest.fixture
def task(tmp_path):
    project_path = tmp_path / "project.kiln"
    project = Project(name="P", path=project_path)
    project.save_to_file()
    task = Task(
        name="T", instruction="Do it", parent=project, turn_mode=TurnMode.multiturn
    )
    task.save_to_file()
    return task


def test_flat_task_run_folder_structure(task: Task):
    output = TaskOutput(output="out")

    parent_run = TaskRun(input="in", output=output, parent=task)
    parent_run.save_to_file()

    nested_run = TaskRun(
        input="nested in",
        output=output,
        parent=task,
        parent_task_run_id=parent_run.id,
    )
    nested_run.save_to_file()

    assert task.path is not None
    assert parent_run.path is not None
    assert nested_run.path is not None
    task_dir = task.path.parent
    runs_dir = task_dir / "runs"
    parent_run_dir = runs_dir / parent_run.build_child_dirname()
    nested_run_dir = runs_dir / nested_run.build_child_dirname()

    assert runs_dir.is_dir()
    assert (parent_run_dir / "task_run.kiln").is_file()
    assert (nested_run_dir / "task_run.kiln").is_file()
    assert nested_run.path.parent == nested_run_dir
    assert not (parent_run_dir / "runs").exists()

    assert parent_run.parent_task() == task
    assert parent_run.parent_task_run_id is None
    assert nested_run.parent_task() == task
    assert nested_run.parent_task_run_id == parent_run.id


def test_task_runs_multiple_levels_via_parent_task_run_id(task: Task):
    output = TaskOutput(output="out")

    run1 = TaskRun(input="in1", output=output, parent=task)
    run1.save_to_file()
    run2 = TaskRun(
        input="in2",
        output=output,
        parent=task,
        parent_task_run_id=run1.id,
    )
    run2.save_to_file()
    run3 = TaskRun(
        input="in3",
        output=output,
        parent=task,
        parent_task_run_id=run2.id,
    )
    run3.save_to_file()

    assert run1.parent_task_run_id is None
    assert run1.parent_task() == task

    assert run2.parent_task_run_id == run1.id
    assert run2.parent_task() == task

    assert run3.parent_task_run_id == run2.id
    assert run3.parent_task() == task

    assert task.path is not None
    loaded_task = Task.load_from_file(task.path)
    all_runs = {r.id: r for r in loaded_task.runs(include_intermediate_runs=True)}
    assert len(all_runs) == 3
    assert all_runs[run2.id].parent_task_run_id == run1.id
    assert all_runs[run3.id].parent_task_run_id == run2.id


def test_parent_task_for_chained_runs(task: Task):
    output = TaskOutput(output="out")
    run1 = TaskRun(input="in1", output=output, parent=task)
    run1.save_to_file()
    run2 = TaskRun(
        input="in2",
        output=output,
        parent=task,
        parent_task_run_id=run1.id,
    )
    run2.save_to_file()
    run3 = TaskRun(
        input="in3",
        output=output,
        parent=task,
        parent_task_run_id=run2.id,
    )
    run3.save_to_file()

    assert run3.parent_task() == task


def test_find_nested_task_run_by_parent_task_run_id(task: Task):
    assert task.path is not None

    output = TaskOutput(output="out")
    parent_run = TaskRun(input="in", output=output, parent=task)
    parent_run.save_to_file()
    nested_run = TaskRun(
        input="nested in",
        output=output,
        parent=task,
        parent_task_run_id=parent_run.id,
    )
    nested_run.save_to_file()
    target_id = nested_run.id

    loaded_task = Task.load_from_file(task.path)
    found = next(
        r
        for r in loaded_task.runs(include_intermediate_runs=True)
        if r.id == target_id and r.parent_task_run_id == parent_run.id
    )
    assert found is not None
    assert found.id == target_id
    assert found.input == "nested in"


def test_find_root_task_run_by_id_given_task(task: Task):
    output = TaskOutput(output="out")
    root_run = TaskRun(input="in", output=output, parent=task)
    root_run.save_to_file()
    target_id = root_run.id

    assert task.path is not None
    loaded_task = Task.load_from_file(task.path)
    found = next(
        r for r in loaded_task.runs(include_intermediate_runs=True) if r.id == target_id
    )
    assert found is not None
    assert found.id == target_id
    assert found.input == "in"


def test_data_source_default_run_config_id_is_none():
    source = DataSource(type=DataSourceType.human, properties={"created_by": "u"})
    assert source.run_config_id is None


def test_data_source_run_config_id_round_trips(task: Task):
    run = TaskRun(
        input="in",
        output=TaskOutput(
            output="out",
            source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "gpt_4o",
                    "model_provider": "openai",
                    "adapter_name": "kiln_langchain_adapter",
                },
                run_config_id="rc_abc123",
            ),
        ),
        parent=task,
    )
    run.save_to_file()

    assert task.path is not None
    loaded_task = Task.load_from_file(task.path)
    loaded = next(r for r in loaded_task.runs() if r.id == run.id)
    assert loaded.output.source.run_config_id == "rc_abc123"


def test_data_source_normalizes_empty_run_config_id_to_none():
    # Tool callers default missing IDs to "". Coerce empty strings to None so
    # the on-disk field is either a real ID or unset, never an ambiguous "".
    source = DataSource(
        type=DataSourceType.synthetic,
        properties={
            "model_name": "gpt_4o",
            "model_provider": "openai",
            "adapter_name": "kiln_langchain_adapter",
        },
        run_config_id="",
    )
    assert source.run_config_id is None


def test_data_source_does_not_validate_run_config_id_existence(task: Task):
    # Per the ticket: a TaskRunConfig may be manually deleted from disk without
    # invalidating historical TaskRuns that reference it.
    run = TaskRun(
        input="in",
        output=TaskOutput(
            output="out",
            source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "gpt_4o",
                    "model_provider": "openai",
                    "adapter_name": "kiln_langchain_adapter",
                },
                run_config_id="rc_does_not_exist",
            ),
        ),
        parent=task,
    )
    run.save_to_file()
    assert run.output.source.run_config_id == "rc_does_not_exist"


def test_comprehensive_flat_task_run_hierarchy(tmp_path):
    project_path = tmp_path / "project.kiln"
    project = Project(name="Test Project", path=project_path)
    project.save_to_file()
    task = Task(
        name="Test Task",
        instruction="Test instruction",
        parent=project,
        turn_mode=TurnMode.multiturn,
    )
    task.save_to_file()

    output = TaskOutput(output="test output")

    run1_l1 = TaskRun(input="level1_run1", output=output, parent=task)
    run1_l1.save_to_file()

    run2_l1 = TaskRun(input="level1_run2", output=output, parent=task)
    run2_l1.save_to_file()

    run1_l2 = TaskRun(
        input="level2_run1",
        output=output,
        parent=task,
        parent_task_run_id=run1_l1.id,
    )
    run1_l2.save_to_file()

    run2_l2 = TaskRun(
        input="level2_run2",
        output=output,
        parent=task,
        parent_task_run_id=run1_l1.id,
    )
    run2_l2.save_to_file()

    run1_l3 = TaskRun(
        input="level3_run1",
        output=output,
        parent=task,
        parent_task_run_id=run1_l2.id,
    )
    run1_l3.save_to_file()

    run2_l3 = TaskRun(
        input="level3_run2",
        output=output,
        parent=task,
        parent_task_run_id=run1_l2.id,
    )
    run2_l3.save_to_file()

    run1_l4 = TaskRun(
        input="level4_run1",
        output=output,
        parent=task,
        parent_task_run_id=run1_l3.id,
    )
    run1_l4.save_to_file()

    run2_l4 = TaskRun(
        input="level4_sibling",
        output=output,
        parent=task,
        parent_task_run_id=run2_l3.id,
    )
    run2_l4.save_to_file()

    loaded_project = Project.load_from_file(project_path)
    loaded_task = loaded_project.tasks()[0]

    all_runs = {r.id: r for r in loaded_task.runs(include_intermediate_runs=True)}
    assert len(all_runs) == 8

    assert all_runs[run1_l1.id].parent_task_run_id is None
    assert all_runs[run2_l1.id].parent_task_run_id is None
    assert all_runs[run1_l2.id].parent_task_run_id == run1_l1.id
    assert all_runs[run2_l2.id].parent_task_run_id == run1_l1.id
    assert all_runs[run1_l3.id].parent_task_run_id == run1_l2.id
    assert all_runs[run2_l3.id].parent_task_run_id == run1_l2.id
    assert all_runs[run1_l4.id].parent_task_run_id == run1_l3.id
    assert all_runs[run2_l4.id].parent_task_run_id == run2_l3.id

    for r in all_runs.values():
        assert r.parent_task() == loaded_task

    roots = [r for r in all_runs.values() if r.parent_task_run_id is None]
    assert len(roots) == 2
    chained = [r for r in all_runs.values() if r.parent_task_run_id is not None]
    assert len(chained) == 6


def test_task_run_wrong_parent_type_raises(tmp_path):
    project = Project(name="proj", path=tmp_path / "project.kiln")
    project.save_to_file()

    with pytest.raises(ValidationError, match="Parent must be of type"):
        TaskRun(
            input="bad parent",
            output=TaskOutput(
                output="x",
                source=DataSource(
                    type=DataSourceType.human, properties={"created_by": "test"}
                ),
            ),
            parent=project,
        )


def test_task_run_runs_on_disk(tmp_path):
    project = Project(name="proj", path=tmp_path / "project.kiln")
    project.save_to_file()
    task = Task(name="t", instruction="i", parent=project, turn_mode=TurnMode.multiturn)
    task.save_to_file()

    parent_run = TaskRun(
        input="parent",
        output=TaskOutput(
            output="parent out",
            source=DataSource(
                type=DataSourceType.human, properties={"created_by": "test"}
            ),
        ),
        parent=task,
    )
    parent_run.save_to_file()

    child_run = TaskRun(
        input="child",
        output=TaskOutput(
            output="child out",
            source=DataSource(
                type=DataSourceType.human, properties={"created_by": "test"}
            ),
        ),
        parent=task,
        parent_task_run_id=parent_run.id,
    )
    child_run.save_to_file()

    loaded_task = Task.load_from_file(task.path)
    children = loaded_task.runs(include_intermediate_runs=True)
    assert len(children) == 2
    by_id = {r.id: r for r in children}
    assert by_id[child_run.id].parent_task_run_id == parent_run.id
    assert by_id[parent_run.id].parent_task_run_id is None
