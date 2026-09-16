import json
import sys
from typing import get_args

import pytest
from pydantic import ValidationError

from kiln_ai.datamodel import (
    DatasetSplit,
    DataSource,
    DataSourceType,
    Finetune,
    MessageUsage,
    Project,
    Task,
    TaskOutput,
    TaskOutputRating,
    TaskOutputRatingType,
    TaskRequirement,
    TaskRun,
    Usage,
)
from kiln_ai.datamodel.eval import EvalRun
from kiln_ai.datamodel.eval_splits import ItemSource, eval_run_item_key
from kiln_ai.datamodel.task_run import EvalItemSource, eval_item_key


@pytest.fixture
def valid_task_run(tmp_path):
    task = Task(
        name="Test Task",
        instruction="test instruction",
        path=tmp_path / Task.base_filename(),
    )
    return TaskRun(
        parent=task,
        input="Test input",
        input_source=DataSource(
            type=DataSourceType.human,
            properties={"created_by": "John Doe"},
        ),
        output=TaskOutput(
            output="Test output",
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "John Doe"},
            ),
        ),
    )


def test_task_model_validation(valid_task_run):
    task_run = valid_task_run
    task_run.model_validate(task_run, strict=True)
    task_run.save_to_file()
    assert task_run.input == "Test input"
    assert task_run.input_source.type == DataSourceType.human
    assert task_run.input_source.properties == {"created_by": "John Doe"}
    assert task_run.output.output == "Test output"
    assert task_run.output.source.type == DataSourceType.human
    assert task_run.output.source.properties == {"created_by": "John Doe"}

    # Invalid source
    with pytest.raises(ValidationError, match="Input should be"):
        DataSource(type="invalid")

    if sys.version_info >= (3, 12):
        with pytest.raises(ValidationError, match="Invalid data source type"):
            task_run = valid_task_run.model_copy(deep=True)
            task_run.input_source.type = "invalid"
            DataSource.model_validate(task_run.input_source, strict=True)

    # Missing required field
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        task_run = valid_task_run.model_copy()
        task_run.input = None

    # Invalid source_properties type
    with pytest.raises(ValidationError):
        task_run = valid_task_run.model_copy()
        task_run.input_source.properties = "invalid"
        DataSource.model_validate(task_run.input_source, strict=True)

    # Test we catch nested validation errors
    with pytest.raises(ValidationError, match="'created_by' is required for"):
        task_run = TaskRun(
            input="Test input",
            input_source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "John Doe"},
            ),
            output=TaskOutput(
                output="Test output",
                source=DataSource(
                    type=DataSourceType.human,
                    properties={"wrong_key": "John Doe"},
                ),
            ),
        )


def test_task_run_relationship(valid_task_run):
    assert valid_task_run.__class__.relationship_name() == "runs"
    assert valid_task_run.__class__.parent_type().__name__ == "Task"


def test_dataset_split_relationship():
    assert DatasetSplit.relationship_name() == "dataset_splits"
    assert DatasetSplit.parent_type().__name__ == "Task"


def test_base_finetune_relationship():
    assert Finetune.relationship_name() == "finetunes"
    assert Finetune.parent_type().__name__ == "Task"


def test_structured_output_workflow(tmp_path):
    tmp_project_file = (
        tmp_path / "test_structured_output_runs" / Project.base_filename()
    )
    # Create project
    project = Project(name="Test Project", path=str(tmp_project_file))
    project.save_to_file()

    # Create task with requirements
    req1 = TaskRequirement(name="Req1", instruction="Name must be capitalized")
    req2 = TaskRequirement(name="Req2", instruction="Age must be positive")

    task = Task(
        name="Structured Output Task",
        parent=project,
        instruction="Generate a JSON object with name and age",
        output_json_schema=json.dumps(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            }
        ),
        requirements=[
            req1,
            req2,
        ],
    )
    task.save_to_file()

    # Create runs
    runs = []
    for source in [DataSourceType.human, DataSourceType.synthetic]:
        for _ in range(2):
            task_run = TaskRun(
                input="Generate info for John Doe",
                input_source=DataSource(
                    type=DataSourceType.human,
                    properties={"created_by": "john_doe"},
                )
                if source == DataSourceType.human
                else DataSource(
                    type=DataSourceType.synthetic,
                    properties={
                        "adapter_name": "TestAdapter",
                        "model_name": "GPT-4",
                        "model_provider": "OpenAI",
                        "prompt_id": "simple_prompt_builder",
                    },
                ),
                parent=task,
                output=TaskOutput(
                    output='{"name": "John Doe", "age": 30}',
                    source=DataSource(
                        type=DataSourceType.human,
                        properties={"created_by": "john_doe"},
                    ),
                ),
            )
            task_run.save_to_file()
            runs.append(task_run)

    # make a run with a repaired output
    repaired_run = TaskRun(
        input="Generate info for John Doe",
        input_source=DataSource(
            type=DataSourceType.human,
            properties={"created_by": "john_doe"},
        ),
        parent=task,
        output=TaskOutput(
            output='{"name": "John Doe", "age": 31}',
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
        ),
        repair_instructions="The age should be 31 instead of 30",
        repaired_output=TaskOutput(
            output='{"name": "John Doe", "age": 31}',
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
        ),
    )
    repaired_run.save_to_file()
    runs.append(repaired_run)

    # Update outputs with ratings
    for task_run in runs:
        task_run.output.rating = TaskOutputRating(
            value=4,
            requirement_ratings={
                req1.id: 5,
                req2.id: 5,
            },
        )
        task_run.save_to_file()

    # Load from disk and validate
    loaded_project = Project.load_from_file(tmp_project_file)
    loaded_task = loaded_project.tasks()[0]

    assert loaded_task.name == "Structured Output Task"
    assert len(loaded_task.requirements) == 2
    loaded_runs = loaded_task.runs(include_intermediate_runs=True)
    assert len(loaded_runs) == 5

    for task_run in loaded_runs:
        output = task_run.output
        assert output.rating is not None
        assert output.rating.value == 4
        assert len(output.rating.requirement_ratings) == 2

    # Find the run with the fixed output
    run_with_fixed_output = next(
        (task_run for task_run in loaded_runs if task_run.repaired_output is not None),
        None,
    )
    assert run_with_fixed_output is not None, "No run found with fixed output"
    assert (
        run_with_fixed_output.repaired_output.output
        == '{"name": "John Doe", "age": 31}'
    )


def test_task_output_requirement_rating_keys(tmp_path):
    # Create a project, task, and example hierarchy
    project = Project(name="Test Project", path=(tmp_path / "test_project"))
    project.save_to_file()

    # Create task requirements
    req1 = TaskRequirement(
        name="Requirement 1", instruction="Requirement 1 instruction"
    )
    req2 = TaskRequirement(
        name="Requirement 2", instruction="Requirement 2 instruction"
    )
    task = Task(
        name="Test Task",
        parent=project,
        instruction="Task instruction",
        requirements=[req1, req2],
    )
    task.save_to_file()

    # Valid case: all requirement IDs are valid
    task_run = TaskRun(
        input="Test input",
        input_source=DataSource(
            type=DataSourceType.human,
            properties={"created_by": "john_doe"},
        ),
        parent=task,
        output=TaskOutput(
            output="Test output",
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
            rating=TaskOutputRating(
                value=4,
                requirement_ratings={
                    req1.id: 5,
                    req2.id: 4,
                },
            ),
        ),
    )
    task_run.save_to_file()
    assert task_run.output.rating.requirement_ratings is not None


_schema_match = "This task requires a specific output schema. While the model produced JSON, that JSON didn't meet the schema."


def test_task_output_schema_validation(tmp_path):
    # Create a project, task, and example hierarchy
    project = Project(name="Test Project", path=(tmp_path / "test_project"))
    project.save_to_file()
    task = Task(
        name="Test Task",
        instruction="test instruction",
        parent=project,
        output_json_schema=json.dumps(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            }
        ),
    )
    task.save_to_file()

    # Create an run output with a valid schema
    task_output = TaskRun(
        input="Test input",
        input_source=DataSource(
            type=DataSourceType.human,
            properties={"created_by": "john_doe"},
        ),
        parent=task,
        output=TaskOutput(
            output='{"name": "John Doe", "age": 30}',
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
        ),
    )
    task_output.save_to_file()

    # changing to invalid output
    with pytest.raises(
        ValueError,
        match=_schema_match,
    ):
        task_output.output.output = '{"name": "John Doe", "age": "thirty"}'
        task_output.save_to_file()

    # changing to invalid output from loaded model
    loaded_task_output = TaskRun.load_from_file(task_output.path)
    with pytest.raises(
        ValueError,
        match=_schema_match,
    ):
        loaded_task_output.output.output = '{"name": "John Doe", "age": "forty"}'
        loaded_task_output.save_to_file()

    # Invalid case: output does not match task output schema
    with pytest.raises(ValueError, match=_schema_match):
        task_output = TaskRun(
            input="Test input",
            input_source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
            parent=task,
            output=TaskOutput(
                output='{"name": "John Doe", "age": "thirty"}',
                source=DataSource(
                    type=DataSourceType.human,
                    properties={"created_by": "john_doe"},
                ),
            ),
        )
        task_output.save_to_file()


_input_schema_match = "Input does not match task input schema"


def test_task_input_schema_validation(tmp_path):
    # Create a project and task hierarchy
    project = Project(name="Test Project", path=(tmp_path / "test_project"))
    project.save_to_file()
    task = Task(
        name="Test Task",
        parent=project,
        instruction="test instruction",
        input_json_schema=json.dumps(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            }
        ),
    )
    task.save_to_file()

    # Create an example with a valid input schema
    valid_task_output = TaskRun(
        input='{"name": "John Doe", "age": 30}',
        input_source=DataSource(
            type=DataSourceType.human,
            properties={"created_by": "john_doe"},
        ),
        parent=task,
        output=TaskOutput(
            output="Test output",
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
        ),
    )
    valid_task_output.save_to_file()

    # Changing to invalid input
    with pytest.raises(ValueError, match=_input_schema_match):
        valid_task_output.input = '{"name": "John Doe", "age": "thirty"}'
        valid_task_output.save_to_file()

    # loading from file, then changing to invalid input
    loaded_task_output = TaskRun.load_from_file(valid_task_output.path)
    with pytest.raises(ValueError, match=_input_schema_match):
        loaded_task_output.input = '{"name": "John Doe", "age": "thirty"}'
        loaded_task_output.save_to_file()

    # Invalid case: input does not match task input schema
    with pytest.raises(ValueError, match=_input_schema_match):
        task_output = TaskRun(
            input='{"name": "John Doe", "age": "thirty"}',
            input_source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
            parent=task,
            output=TaskOutput(
                output="Test output",
                source=DataSource(
                    type=DataSourceType.human,
                    properties={"created_by": "john_doe"},
                ),
            ),
        )
        task_output.save_to_file()


def test_valid_human_task_output():
    output = TaskOutput(
        output="Test output",
        source=DataSource(
            type=DataSourceType.human,
            properties={"created_by": "John Doe"},
        ),
    )
    assert output.source.type == DataSourceType.human
    assert output.source.properties["created_by"] == "John Doe"


def test_invalid_human_task_output_missing_created_by():
    with pytest.raises(ValidationError, match="'created_by' is required for"):
        TaskOutput(
            output="Test output",
            source=DataSource(
                type=DataSourceType.human,
                properties={},
            ),
        )


def test_invalid_human_task_output_empty_created_by():
    with pytest.raises(
        ValidationError, match="Property 'created_by' must be a non-empty string"
    ):
        TaskOutput(
            output="Test output",
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": ""},
            ),
        )


def test_valid_synthetic_task_output():
    output = TaskOutput(
        output="Test output",
        source=DataSource(
            type=DataSourceType.synthetic,
            properties={
                "adapter_name": "TestAdapter",
                "model_name": "GPT-4",
                "model_provider": "OpenAI",
                "prompt_id": "simple_prompt_builder",
            },
        ),
    )
    assert output.source.type == DataSourceType.synthetic
    assert output.source.properties["adapter_name"] == "TestAdapter"
    assert output.source.properties["model_name"] == "GPT-4"
    assert output.source.properties["model_provider"] == "OpenAI"
    assert output.source.properties["prompt_id"] == "simple_prompt_builder"


def test_invalid_synthetic_task_output_missing_keys():
    with pytest.raises(
        ValidationError,
        match="'model_provider' is required for",
    ):
        TaskOutput(
            output="Test output",
            source=DataSource(
                type=DataSourceType.synthetic,
                properties={"adapter_name": "TestAdapter", "model_name": "GPT-4"},
            ),
        )


def test_invalid_synthetic_task_output_empty_values():
    with pytest.raises(
        ValidationError, match="'model_name' must be a non-empty string"
    ):
        TaskOutput(
            output="Test output",
            source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "adapter_name": "TestAdapter",
                    "model_name": "",
                    "model_provider": "OpenAI",
                    "prompt_id": "simple_prompt_builder",
                },
            ),
        )


def test_invalid_synthetic_task_output_non_string_values():
    with pytest.raises(ValidationError, match="'prompt_id' must be of type str"):
        DataSource(
            type=DataSourceType.synthetic,
            properties={
                "adapter_name": "TestAdapter",
                "model_name": "GPT-4",
                "model_provider": "OpenAI",
                "prompt_id": 123,
            },
        )


def test_task_run_validate_repaired_output():
    # Test case 1: Valid TaskRun with no repaired_output
    valid_task_run = TaskRun(
        input="test input",
        input_source=DataSource(
            type=DataSourceType.human,
            properties={"created_by": "john_doe"},
        ),
        output=TaskOutput(
            output="test output",
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
        ),
    )
    assert valid_task_run.repaired_output is None

    # Test case 2: Valid TaskRun with repaired_output and no rating
    valid_task_run_with_repair = TaskRun(
        input="test input",
        input_source=DataSource(
            type=DataSourceType.human,
            properties={"created_by": "john_doe"},
        ),
        output=TaskOutput(
            output="test output",
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
        ),
        repair_instructions="Fix the output",
        repaired_output=TaskOutput(
            output="repaired output",
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
        ),
    )
    assert valid_task_run_with_repair.repaired_output is not None
    assert valid_task_run_with_repair.repaired_output.rating is None

    # test missing repair_instructions
    with pytest.raises(ValidationError) as exc_info:
        TaskRun(
            input="test input",
            input_source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
            output=TaskOutput(
                output="test output",
                source=DataSource(
                    type=DataSourceType.human,
                    properties={"created_by": "john_doe"},
                ),
            ),
            repaired_output=TaskOutput(
                output="repaired output",
                source=DataSource(
                    type=DataSourceType.human,
                    properties={"created_by": "john_doe"},
                ),
            ),
        )

    assert "Repair instructions are required" in str(exc_info.value)

    # test missing repaired_output
    with pytest.raises(ValidationError) as exc_info:
        TaskRun(
            input="test input",
            input_source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
            output=TaskOutput(
                output="test output",
                source=DataSource(
                    type=DataSourceType.human,
                    properties={"created_by": "john_doe"},
                ),
            ),
            repair_instructions="Fix the output",
        )

    assert "A repaired output is required" in str(exc_info.value)

    # Test case 3: Invalid TaskRun with repaired_output containing a rating
    with pytest.raises(ValidationError) as exc_info:
        TaskRun(
            input="test input",
            input_source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
            output=TaskOutput(
                output="test output",
                source=DataSource(
                    type=DataSourceType.human,
                    properties={"created_by": "john_doe"},
                ),
            ),
            repaired_output=TaskOutput(
                output="repaired output",
                source=DataSource(
                    type=DataSourceType.human,
                    properties={"created_by": "john_doe"},
                ),
                rating=TaskOutputRating(type=TaskOutputRatingType.five_star, value=5.0),
            ),
        )

    assert "Repaired output rating must be None" in str(exc_info.value)


def test_task_run_validate_repaired_output_structured(tmp_path):
    # Create a project, task, and example hierarchy
    project = Project(name="Test Project", path=(tmp_path / "test_project"))
    project.save_to_file()
    task = Task(
        name="Test Task",
        instruction="test instruction",
        parent=project,
        output_json_schema=json.dumps(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            }
        ),
    )
    task.save_to_file()

    # test valid repaired output schema
    task_run = TaskRun(
        parent=task,
        input="test input",
        input_source=DataSource(
            type=DataSourceType.human,
            properties={"created_by": "john_doe"},
        ),
        output=TaskOutput(
            output='{"name": "John Doe", "age": 30}',
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
        ),
        repair_instructions="Fix the output",
        repaired_output=TaskOutput(
            output='{"name": "John Doe", "age": 30}',
            source=DataSource(
                type=DataSourceType.human, properties={"created_by": "john_doe"}
            ),
        ),
    )

    assert task_run.repaired_output is not None
    assert task_run.repaired_output.rating is None

    # test invalid JSON
    with pytest.raises(ValueError):
        TaskRun(
            parent=task,
            input="test input",
            input_source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
            output=TaskOutput(
                output='{"name": "John Doe", "age": 30}',
                source=DataSource(
                    type=DataSourceType.human,
                    properties={"created_by": "john_doe"},
                ),
            ),
            repair_instructions="Fix the output",
            repaired_output=TaskOutput(
                output='{"name": "John Doe", "age": 30',  # missing closing brace
                source=DataSource(
                    type=DataSourceType.human,
                    properties={"created_by": "john_doe"},
                ),
            ),
        )

    # test invalid repaired output schema
    with pytest.raises(ValueError):
        TaskRun(
            parent=task,
            input="test input",
            input_source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "john_doe"},
            ),
            output=TaskOutput(
                output='{"name": "John Doe", "age": 30}',
                source=DataSource(
                    type=DataSourceType.human,
                    properties={"created_by": "john_doe"},
                ),
            ),
            repair_instructions="Fix the output",
            repaired_output=TaskOutput(
                output='{"name": "John Doe", "age": "thirty"}',  # invalid schema
                source=DataSource(
                    type=DataSourceType.human,
                    properties={"created_by": "john_doe"},
                ),
            ),
        )


@pytest.mark.parametrize(
    "input_tokens,output_tokens,total_tokens,cost,should_raise",
    [
        # Valid cases
        (100, 50, 150, 0.002, False),  # All fields
        (None, None, None, None, False),  # All None (defaults)
        # Invalid cases
        (-100, 50, 150, 0.002, True),  # Negative input_tokens
        (100, -50, 150, 0.002, True),  # Negative output_tokens
        (100, 50, -150, 0.002, True),  # Negative total_tokens
        (100, 50, 150, -0.002, True),  # Negative cost
    ],
)
def test_usage_model(input_tokens, output_tokens, total_tokens, cost, should_raise):
    """Test the Usage model with various input combinations."""
    if should_raise:
        with pytest.raises(ValidationError):
            Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost=cost,
            )
    else:
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )
        assert usage.input_tokens == input_tokens
        assert usage.output_tokens == output_tokens
        assert usage.total_tokens == total_tokens
        assert usage.cost == cost


def test_usage_model_in_task_run(valid_task_run):
    """Test that Usage can be properly set in a TaskRun."""
    usage = Usage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cost=0.002,
    )
    task_run = valid_task_run.model_copy(deep=True)
    task_run.usage = usage
    assert task_run.usage == usage
    assert task_run.usage.input_tokens == 100
    assert task_run.usage.output_tokens == 50
    assert task_run.usage.total_tokens == 150
    assert task_run.usage.cost == 0.002


@pytest.mark.parametrize(
    "usage1_data,usage2_data,expected_data",
    [
        # None + None = None
        (
            {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "cost": None,
            },
            {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "cost": None,
            },
            {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "cost": None,
            },
        ),
        # None + value = value
        (
            {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "cost": None,
            },
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cost": 0.005,
            },
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cost": 0.005,
            },
        ),
        # value + None = value
        (
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cost": 0.005,
            },
            {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "cost": None,
            },
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cost": 0.005,
            },
        ),
        # value1 + value2 = value1 + value2
        (
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cost": 0.005,
            },
            {
                "input_tokens": 200,
                "output_tokens": 75,
                "total_tokens": 275,
                "cost": 0.010,
            },
            {
                "input_tokens": 300,
                "output_tokens": 125,
                "total_tokens": 425,
                "cost": 0.015,
            },
        ),
        # Mixed scenarios
        (
            {
                "input_tokens": 100,
                "output_tokens": None,
                "total_tokens": 150,
                "cost": None,
            },
            {
                "input_tokens": None,
                "output_tokens": 75,
                "total_tokens": None,
                "cost": 0.010,
            },
            {
                "input_tokens": 100,
                "output_tokens": 75,
                "total_tokens": 150,
                "cost": 0.010,
            },
        ),
        # Edge case: zeros
        (
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost": 0.0},
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cost": 0.005,
            },
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cost": 0.005,
            },
        ),
    ],
)
def test_usage_addition(usage1_data, usage2_data, expected_data):
    """Test Usage addition with various combinations of None and numeric values."""
    usage1 = Usage(**usage1_data)
    usage2 = Usage(**usage2_data)
    result = usage1 + usage2

    assert result.input_tokens == expected_data["input_tokens"]
    assert result.output_tokens == expected_data["output_tokens"]
    assert result.total_tokens == expected_data["total_tokens"]
    assert result.cost == expected_data["cost"]


def test_usage_addition_type_error():
    """Test that adding Usage to non-Usage raises TypeError."""
    usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150, cost=0.005)

    with pytest.raises(TypeError, match="Cannot add Usage with"):
        usage + "not_a_usage"  # type: ignore

    with pytest.raises(TypeError, match="Cannot add Usage with"):
        usage + 42  # type: ignore

    with pytest.raises(TypeError, match="Cannot add Usage with"):
        usage + {"input_tokens": 100}  # type: ignore


def test_usage_addition_immutability():
    """Test that addition creates new Usage objects and doesn't mutate originals."""
    usage1 = Usage(input_tokens=100, output_tokens=50, total_tokens=150, cost=0.005)
    usage2 = Usage(input_tokens=200, output_tokens=75, total_tokens=275, cost=0.010)

    original_usage1_data = usage1.model_dump()
    original_usage2_data = usage2.model_dump()

    result = usage1 + usage2

    # Original objects should be unchanged
    assert usage1.model_dump() == original_usage1_data
    assert usage2.model_dump() == original_usage2_data

    # Result should be a new object
    assert result is not usage1
    assert result is not usage2
    assert result.input_tokens == 300
    assert result.output_tokens == 125
    assert result.total_tokens == 425
    assert result.cost == 0.015


def test_usage_total_llm_latency_ms_field():
    """Test total_llm_latency_ms field validation."""
    usage = Usage(total_llm_latency_ms=500)
    assert usage.total_llm_latency_ms == 500

    usage_none = Usage()
    assert usage_none.total_llm_latency_ms is None

    usage_zero = Usage(total_llm_latency_ms=0)
    assert usage_zero.total_llm_latency_ms == 0

    with pytest.raises(ValidationError):
        Usage(total_llm_latency_ms=-1)


@pytest.mark.parametrize(
    "latency1,latency2,expected",
    [
        (None, None, None),
        (None, 300, 300),
        (200, None, 200),
        (200, 300, 500),
        (0, 100, 100),
    ],
)
def test_usage_addition_with_latency(latency1, latency2, expected):
    """Test Usage.__add__ handles total_llm_latency_ms correctly."""
    u1 = Usage(total_llm_latency_ms=latency1)
    u2 = Usage(total_llm_latency_ms=latency2)
    result = u1 + u2
    assert result.total_llm_latency_ms == expected


def test_usage_backwards_compat_without_latency():
    """Test that old Usage JSON without total_llm_latency_ms deserializes as None."""
    old_json = (
        '{"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "cost": 0.01}'
    )
    usage = Usage.model_validate_json(old_json)
    assert usage.total_llm_latency_ms is None
    assert usage.input_tokens == 100


def test_task_run_default_cumulative_usage_is_none(valid_task_run):
    assert valid_task_run.cumulative_usage is None


def test_task_run_can_set_cumulative_usage(valid_task_run):
    cumulative = MessageUsage(
        input_tokens=300, output_tokens=120, total_tokens=420, cost=0.05
    )
    task_run = valid_task_run.model_copy(deep=True)
    task_run.cumulative_usage = cumulative
    assert task_run.cumulative_usage == cumulative
    # `usage` and `cumulative_usage` are independent fields.
    assert task_run.usage is None


def test_task_run_loads_old_json_without_cumulative_usage(valid_task_run):
    """JSON serialized before this field existed loads with cumulative_usage None."""
    payload = valid_task_run.model_dump(mode="json")
    payload.pop("cumulative_usage", None)

    reloaded = TaskRun.model_validate(payload)

    assert reloaded.cumulative_usage is None


def test_task_run_round_trip_cumulative_usage(valid_task_run):
    cumulative = MessageUsage(
        input_tokens=10, output_tokens=5, total_tokens=15, cost=0.001
    )
    task_run = valid_task_run.model_copy(deep=True)
    task_run.cumulative_usage = cumulative

    payload = task_run.model_dump(mode="json")
    reloaded = TaskRun.model_validate(payload)

    assert reloaded.cumulative_usage == cumulative


def test_task_run_per_message_usage_and_cumulative_have_no_latency_key(
    valid_task_run,
):
    """A persisted TaskRun's per-message ``usage`` and ``cumulative_usage``
    are typed ``MessageUsage`` and must NOT serialize ``total_llm_latency_ms``.
    The run-level ``usage`` field is still ``Usage`` and DOES carry the key."""
    task_run = valid_task_run.model_copy(deep=True)
    task_run.usage = Usage(input_tokens=5, total_llm_latency_ms=400)
    task_run.cumulative_usage = MessageUsage(input_tokens=12, output_tokens=8)
    task_run.trace = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "ok",
            "usage": MessageUsage(input_tokens=12, output_tokens=8, cost=0.42),
        },
    ]

    payload = task_run.model_dump(mode="json")

    cumulative_payload = payload["cumulative_usage"]
    assert cumulative_payload is not None
    assert "total_llm_latency_ms" not in cumulative_payload

    assistant_msg = payload["trace"][1]
    assert "total_llm_latency_ms" not in assistant_msg["usage"]

    # Run-level Usage still keeps the latency field.
    assert "total_llm_latency_ms" in payload["usage"]
    assert payload["usage"]["total_llm_latency_ms"] == 400


def test_task_run_loads_legacy_cumulative_usage_with_latency_key(valid_task_run):
    """A pre-split TaskRun JSON could have ``cumulative_usage.total_llm_latency_ms``
    set (typically to null). It must still load — Pydantic's default
    ``extra='ignore'`` drops the unknown field on the narrowed MessageUsage."""
    payload = valid_task_run.model_dump(mode="json")
    payload["cumulative_usage"] = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "cost": 0.001,
        "cached_tokens": None,
        "total_llm_latency_ms": None,  # legacy field — must be dropped
    }

    reloaded = TaskRun.model_validate(payload)

    assert reloaded.cumulative_usage is not None
    assert reloaded.cumulative_usage.input_tokens == 10
    assert reloaded.cumulative_usage.cost == 0.001
    # The narrowed MessageUsage doesn't carry the legacy field at all.
    assert not hasattr(reloaded.cumulative_usage, "total_llm_latency_ms")


def test_task_run_loads_legacy_cumulative_usage_with_populated_latency(valid_task_run):
    """Same as above but with a non-null legacy latency value — still loads."""
    payload = valid_task_run.model_dump(mode="json")
    payload["cumulative_usage"] = {
        "input_tokens": 7,
        "total_llm_latency_ms": 1234,  # non-null legacy value — must be dropped
    }

    reloaded = TaskRun.model_validate(payload)

    assert reloaded.cumulative_usage is not None
    assert reloaded.cumulative_usage.input_tokens == 7
    assert not hasattr(reloaded.cumulative_usage, "total_llm_latency_ms")


def test_task_run_eval_source_defaults_to_none(valid_task_run):
    assert valid_task_run.eval_source is None


@pytest.mark.parametrize("source_type", ["eval_input", "task_run"])
def test_task_run_eval_source_round_trip(valid_task_run, source_type):
    valid_task_run.eval_source = EvalItemSource(
        source_type=source_type, source_id="item123"
    )

    reloaded = TaskRun.model_validate(valid_task_run.model_dump(mode="json"))

    assert reloaded.eval_source == EvalItemSource(
        source_type=source_type, source_id="item123"
    )


def test_eval_item_source_rejects_unknown_source_type():
    with pytest.raises(ValidationError):
        EvalItemSource(source_type="something_else", source_id="item123")


@pytest.mark.parametrize("source_id", [None, ""])
def test_eval_item_source_requires_a_real_source_id(source_id):
    """An id-less source is a trace-index key that collides with every other id-less
    source. Absence is expressed by TaskRun.eval_source being None, not by an empty id."""
    with pytest.raises(ValidationError):
        EvalItemSource(source_type="eval_input", source_id=source_id)


def test_eval_item_source_type_tracks_item_source():
    """EvalItemSource.source_type restates eval_splits.ItemSource, because eval_splits
    imports task_run at runtime so the alias can't come back the other way. Nothing else
    fails if a third member is added to one and not the other."""
    assert set(get_args(ItemSource)) == set(
        get_args(EvalItemSource.model_fields["source_type"].annotation)
    )


@pytest.mark.parametrize("source_type", ["eval_input", "task_run"])
def test_eval_item_key_matches_eval_run_item_key(source_type):
    """A trace's source and a score's item must resolve to the same ItemKey, or the
    runner would index traces under one identity and look them up under another."""
    source = EvalItemSource(source_type=source_type, source_id="item123")

    id_field = "eval_input_id" if source_type == "eval_input" else "dataset_id"
    eval_run = EvalRun(
        **{id_field: "item123"},
        task_run_config_id="rc1",
        input="in",
        output="out",
        scores={"accuracy": 1.0},
    )

    assert eval_item_key(source) == eval_run_item_key(eval_run)
    assert eval_item_key(source) == (source_type, "item123")
