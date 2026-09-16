import json

import pytest
from pydantic import ValidationError

from kiln_ai.adapters.run_output import RunOutput
from kiln_ai.datamodel.datamodel_enums import (
    ModelProviderName,
    StructuredOutputMode,
    TaskOutputRatingType,
    TurnMode,
)
from kiln_ai.datamodel.model_cache import ModelCache
from kiln_ai.datamodel.project import Project
from kiln_ai.datamodel.prompt_id import PromptGenerators
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties
from kiln_ai.datamodel.spec import Spec
from kiln_ai.datamodel.spec_properties import (
    DesiredBehaviourProperties,
    SpecType,
    ToxicityProperties,
)
from kiln_ai.datamodel.task import Task, TaskRunConfig
from kiln_ai.datamodel.task_output import (
    TaskOutput,
    normalize_rating,
)
from kiln_ai.datamodel.task_run import EvalItemSource, TaskRun
from kiln_ai.datamodel.test_json_schema import json_joke_schema


def test_runconfig_valid_creation():
    config = KilnAgentRunConfigProperties(
        model_name="gpt-4",
        model_provider_name=ModelProviderName.openai,
        prompt_id=PromptGenerators.SIMPLE,
        structured_output_mode=StructuredOutputMode.json_schema,
    )

    assert config.model_name == "gpt-4"
    assert config.model_provider_name == ModelProviderName.openai
    assert config.prompt_id == PromptGenerators.SIMPLE  # Check default value


def test_runconfig_missing_required_fields():
    with pytest.raises(ValidationError) as exc_info:
        KilnAgentRunConfigProperties()  # type: ignore

    errors = exc_info.value.errors()
    assert (
        len(errors) == 4
    )  # task, model_name, model_provider_name, and prompt_id are required
    assert any(error["loc"][0] == "model_name" for error in errors)
    assert any(error["loc"][0] == "model_provider_name" for error in errors)
    assert any(error["loc"][0] == "prompt_id" for error in errors)
    assert any(error["loc"][0] == "structured_output_mode" for error in errors)


def test_runconfig_custom_prompt_id():
    config = KilnAgentRunConfigProperties(
        model_name="gpt-4",
        model_provider_name=ModelProviderName.openai,
        prompt_id=PromptGenerators.SIMPLE_CHAIN_OF_THOUGHT,
        structured_output_mode=StructuredOutputMode.json_schema,
    )

    assert config.prompt_id == PromptGenerators.SIMPLE_CHAIN_OF_THOUGHT


@pytest.fixture
def sample_task():
    return Task(name="Test Task", instruction="Test instruction")


@pytest.fixture
def sample_run_config_props(sample_task):
    return KilnAgentRunConfigProperties(
        model_name="gpt-4",
        model_provider_name=ModelProviderName.openai,
        prompt_id=PromptGenerators.SIMPLE,
        structured_output_mode=StructuredOutputMode.json_schema,
    )


def test_task_run_config_valid_creation(sample_task, sample_run_config_props):
    config = TaskRunConfig(
        name="Test Config",
        description="Test description",
        run_config_properties=sample_run_config_props,
        parent=sample_task,
    )

    assert config.name == "Test Config"
    assert config.description == "Test description"
    assert config.run_config_properties == sample_run_config_props
    assert config.parent_task() == sample_task


def test_task_run_config_minimal_creation(sample_task, sample_run_config_props):
    # Test creation with only required fields
    config = TaskRunConfig(
        name="Test Config",
        run_config_properties=sample_run_config_props,
        parent=sample_task,
    )

    assert config.name == "Test Config"
    assert config.description is None
    assert config.run_config_properties == sample_run_config_props


def test_task_run_config_missing_required_fields(sample_task):
    # Test missing name
    with pytest.raises(ValidationError) as exc_info:
        TaskRunConfig(
            run_config_properties=KilnAgentRunConfigProperties(
                model_name="gpt-4", model_provider_name="openai"
            ),  # type: ignore
            parent=sample_task,
        )  # type: ignore
    assert "Field required" in str(exc_info.value)

    # Test missing run_config
    with pytest.raises(ValidationError) as exc_info:
        TaskRunConfig(name="Test Config", parent=sample_task)  # type: ignore
    assert "Field required" in str(exc_info.value)


@pytest.mark.parametrize(
    "rating_type,rating,expected",
    [
        (TaskOutputRatingType.five_star, 1, 0),
        (TaskOutputRatingType.five_star, 2, 0.25),
        (TaskOutputRatingType.five_star, 3, 0.5),
        (TaskOutputRatingType.five_star, 4, 0.75),
        (TaskOutputRatingType.five_star, 5, 1),
        (TaskOutputRatingType.pass_fail, 0, 0),
        (TaskOutputRatingType.pass_fail, 1, 1),
        (TaskOutputRatingType.pass_fail, 0.5, 0.5),
        (TaskOutputRatingType.pass_fail_critical, -1, 0),
        (TaskOutputRatingType.pass_fail_critical, 0, 0.5),
        (TaskOutputRatingType.pass_fail_critical, 1, 1),
        (TaskOutputRatingType.pass_fail_critical, 0.5, 0.75),
    ],
)
def test_normalize_rating(rating_type, rating, expected):
    assert normalize_rating(rating, rating_type) == expected


@pytest.mark.parametrize(
    "rating_type,rating",
    [
        (TaskOutputRatingType.five_star, 0),
        (TaskOutputRatingType.five_star, 6),
        (TaskOutputRatingType.pass_fail, -0.5),
        (TaskOutputRatingType.pass_fail, 1.5),
        (TaskOutputRatingType.pass_fail_critical, -1.5),
        (TaskOutputRatingType.pass_fail_critical, 1.5),
        (TaskOutputRatingType.custom, 0),
        (TaskOutputRatingType.custom, 99),
    ],
)
def test_normalize_rating_errors(rating_type, rating):
    with pytest.raises(ValueError):
        normalize_rating(rating, rating_type)


def test_run_config_defaults():
    """RunConfig should require top_p, temperature, and structured_output_mode to be set."""

    config = KilnAgentRunConfigProperties(
        model_name="gpt-4",
        model_provider_name=ModelProviderName.openai,
        prompt_id=PromptGenerators.SIMPLE,
        structured_output_mode=StructuredOutputMode.json_schema,
    )
    assert config.top_p == 1.0
    assert config.temperature == 1.0


def test_run_config_valid_ranges():
    """RunConfig should accept valid ranges for top_p and temperature."""

    # Test valid values
    config = KilnAgentRunConfigProperties(
        model_name="gpt-4",
        model_provider_name=ModelProviderName.openai,
        prompt_id=PromptGenerators.SIMPLE,
        top_p=0.9,
        temperature=0.7,
        structured_output_mode=StructuredOutputMode.json_schema,
    )

    assert config.top_p == 0.9
    assert config.temperature == 0.7
    assert config.structured_output_mode == StructuredOutputMode.json_schema


@pytest.mark.parametrize("top_p", [0.0, 0.5, 1.0])
def test_run_config_valid_top_p(top_p):
    """Test that RunConfig accepts valid top_p values (0-1)."""

    config = KilnAgentRunConfigProperties(
        model_name="gpt-4",
        model_provider_name=ModelProviderName.openai,
        prompt_id=PromptGenerators.SIMPLE,
        top_p=top_p,
        temperature=1.0,
        structured_output_mode=StructuredOutputMode.json_schema,
    )

    assert config.top_p == top_p


@pytest.mark.parametrize("top_p", [-0.1, 1.1, 2.0])
def test_run_config_invalid_top_p(top_p):
    """Test that RunConfig rejects invalid top_p values."""

    with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
        KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE,
            top_p=top_p,
            temperature=1.0,
            structured_output_mode=StructuredOutputMode.json_schema,
        )


@pytest.mark.parametrize("temperature", [0.0, 1.0, 2.0])
def test_run_config_valid_temperature(temperature):
    """Test that RunConfig accepts valid temperature values (0-2)."""

    config = KilnAgentRunConfigProperties(
        model_name="gpt-4",
        model_provider_name=ModelProviderName.openai,
        prompt_id=PromptGenerators.SIMPLE,
        top_p=0.9,
        temperature=temperature,
        structured_output_mode=StructuredOutputMode.json_schema,
    )

    assert config.temperature == temperature


@pytest.mark.parametrize("temperature", [-0.1, 2.1, 3.0])
def test_run_config_invalid_temperature(temperature):
    """Test that RunConfig rejects invalid temperature values."""

    with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
        KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE,
            top_p=0.9,
            temperature=temperature,
            structured_output_mode=StructuredOutputMode.json_schema,
        )


def test_run_config_upgrade_old_entries():
    """Test that TaskRunConfig parses old entries correctly with nested objects, filling in defaults where needed."""

    data = {
        "v": 1,
        "name": "test name",
        "created_at": "2025-06-09T13:33:35.276927",
        "created_by": "scosman",
        "run_config_properties": {
            "model_name": "gpt_4_1_nano",
            "model_provider_name": "openai",
            "prompt_id": "task_run_config::189194447826::228174773209::244130257039",
            "top_p": 0.77,
            "temperature": 0.77,
            "structured_output_mode": "json_instruction_and_object",
        },
        "prompt": {
            "name": "Dazzling Unicorn",
            "description": "Frozen copy of prompt 'simple_prompt_builder'.",
            "generator_id": "simple_prompt_builder",
            "prompt": "Generate a joke, given a theme. The theme will be provided as a word or phrase as the input to the model. The assistant should output a joke that is funny and relevant to the theme. If a style is provided, the joke should be in that style. The output should include a setup and punchline.\n\nYour response should respect the following requirements:\n1) Keep the joke on topic. If the user specifies a theme, the joke must be related to that theme.\n2) Avoid any jokes that are offensive or inappropriate. Keep the joke clean and appropriate for all audiences.\n3) Make the joke funny and engaging. It should be something that someone would want to tell to their friends. Something clever, not just a simple pun.\n",
            "chain_of_thought_instructions": None,
        },
        "model_type": "task_run_config",
    }

    # Parse the data - this should be TaskRunConfig, not RunConfig
    parsed = TaskRunConfig.model_validate(data)
    assert parsed.name == "test name"
    assert parsed.created_by == "scosman"
    assert (
        parsed.run_config_properties.structured_output_mode
        == "json_instruction_and_object"
    )

    # should still work if loading from file
    parsed = TaskRunConfig.model_validate(data, context={"loading_from_file": True})
    assert parsed.name == "test name"
    assert parsed.created_by == "scosman"
    assert (
        parsed.run_config_properties.structured_output_mode
        == "json_instruction_and_object"
    )

    # Remove structured_output_mode from run_config_properties and parse again
    del data["run_config_properties"]["structured_output_mode"]

    with pytest.raises(ValidationError):
        # should error if not loading from file
        parsed = TaskRunConfig.model_validate(data)

    parsed = TaskRunConfig.model_validate(data, context={"loading_from_file": True})
    assert parsed.name == "test name"
    assert parsed.created_by == "scosman"
    assert parsed.run_config_properties.structured_output_mode == "unknown"


def test_task_name_unicode_name():
    task = Task(name="你好", instruction="Do something")
    assert task.name == "你好"


def test_task_run_config_long_name_folder_has_no_trailing_space(tmp_path):
    """Folder segment from name must not end with a trailing space (path/git tooling)."""
    project_path = tmp_path / "project.kiln"
    project = Project(name="Test Project", path=project_path)
    project.save_to_file()

    task = Task(
        name="run_agent_brand_mentions_feed_cl",
        instruction="Test instruction",
        parent=project,
    )
    task.save_to_file()

    long_name = "Deepseek 3p2 + KilnOptimized (3 xyz)"
    assert len(long_name[:32]) == 32 and long_name[:32].endswith(" ")

    run_config = TaskRunConfig(
        name=long_name,
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE,
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=task,
    )
    run_config.save_to_file()

    assert run_config.path is not None
    assert "run_configs" in run_config.path.parts
    folder_name = run_config.path.parent.name
    assert folder_name == folder_name.rstrip()
    assert not folder_name.endswith(" ")


def test_task_default_run_config_id_property(tmp_path):
    """Test that default_run_config_id can be set and retrieved."""

    # Create a task
    task = Task(
        name="Test Task", instruction="Test instruction", path=tmp_path / "task.kiln"
    )
    task.save_to_file()

    # Create a run config for the task
    run_config = TaskRunConfig(
        name="Test Config",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE,
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=task,
    )
    run_config.save_to_file()

    # Test None default (should be valid)
    assert task.default_run_config_id is None

    # Test setting a valid ID
    task.default_run_config_id = "123456789012"
    assert task.default_run_config_id == "123456789012"

    # Test setting back to None
    task.default_run_config_id = None
    assert task.default_run_config_id is None


def test_task_specs_relationship(tmp_path):
    """Test that specs can be created, saved, and retrieved through the task parent."""
    task = Task(
        name="Test Task", instruction="Test instruction", path=tmp_path / "task.kiln"
    )
    task.save_to_file()

    properties = DesiredBehaviourProperties(
        spec_type=SpecType.desired_behaviour,
        core_requirement="Test instruction",
        desired_behaviour_description="The system should behave correctly",
        correct_behaviour_examples="Example 1",
        incorrect_behaviour_examples="Example 1",
    )
    spec = Spec(
        name="Test Spec",
        definition="The system should behave correctly",
        properties=properties,
        eval_id="test_eval_id",
        parent=task,
    )
    spec.save_to_file()

    # Test specs can be retrieved from disk
    specs = task.specs()
    assert len(specs) == 1
    assert specs[0].name == "Test Spec"
    assert specs[0].definition == "The system should behave correctly"
    assert specs[0].properties["spec_type"] == SpecType.desired_behaviour


def test_task_specs_readonly(tmp_path):
    """Test that specs can be retrieved with readonly parameter."""
    task = Task(
        name="Test Task", instruction="Test instruction", path=tmp_path / "task.kiln"
    )
    task.save_to_file()

    properties = ToxicityProperties(
        spec_type=SpecType.toxicity,
        core_requirement="The system should avoid toxic language",
        toxicity_examples="Example 1",
    )
    spec = Spec(
        name="Readonly Spec",
        definition="System should handle readonly correctly",
        properties=properties,
        eval_id="test_eval_id",
        parent=task,
    )
    spec.save_to_file()

    specs_readonly = task.specs(readonly=True)
    assert len(specs_readonly) == 1
    assert specs_readonly[0].name == "Readonly Spec"

    specs_default = task.specs(readonly=False)
    assert len(specs_default) == 1
    assert specs_default[0].name == "Readonly Spec"


def test_task_prompt_optimization_jobs_relationship(tmp_path):
    """Test that prompt_optimization_jobs can be created, saved, and retrieved through the task parent."""
    from kiln_ai.datamodel import PromptOptimizationJob

    task = Task(
        name="Test Task", instruction="Test instruction", path=tmp_path / "task.kiln"
    )
    task.save_to_file()

    prompt_optimization_job = PromptOptimizationJob(
        name="Test Prompt Optimization Job",
        job_id="remote-job-123",
        target_run_config_id="config-123",
        latest_status="pending",
        parent=task,
    )
    prompt_optimization_job.save_to_file()

    prompt_optimization_jobs = task.prompt_optimization_jobs()
    assert len(prompt_optimization_jobs) == 1
    assert prompt_optimization_jobs[0].name == "Test Prompt Optimization Job"
    assert prompt_optimization_jobs[0].job_id == "remote-job-123"


def test_task_prompt_optimization_jobs_readonly(tmp_path):
    """Test that prompt_optimization_jobs can be retrieved with readonly parameter."""
    from kiln_ai.datamodel import PromptOptimizationJob

    task = Task(
        name="Test Task", instruction="Test instruction", path=tmp_path / "task.kiln"
    )
    task.save_to_file()

    prompt_optimization_job = PromptOptimizationJob(
        name="Readonly Prompt Optimization Job",
        job_id="remote-job-456",
        target_run_config_id="config-456",
        latest_status="succeeded",
        parent=task,
    )
    prompt_optimization_job.save_to_file()

    prompt_optimization_jobs_readonly = task.prompt_optimization_jobs(readonly=True)
    assert len(prompt_optimization_jobs_readonly) == 1
    assert (
        prompt_optimization_jobs_readonly[0].name == "Readonly Prompt Optimization Job"
    )

    prompt_optimization_jobs_default = task.prompt_optimization_jobs(readonly=False)
    assert len(prompt_optimization_jobs_default) == 1
    assert (
        prompt_optimization_jobs_default[0].name == "Readonly Prompt Optimization Job"
    )


def test_all_children_of_parent_path_task_runs(tmp_path):
    """all_children_of_parent_path lists TaskRun files under a Task."""
    task = Task(
        name="Test Task",
        instruction="Test instruction",
        path=tmp_path / "task.kiln",
    )
    task.save_to_file()

    output = TaskOutput(output="test output")

    run1 = TaskRun(input="input1", output=output, parent=task)
    run2 = TaskRun(input="input2", output=output, parent=task)
    run1.save_to_file()
    run2.save_to_file()

    children = TaskRun.all_children_of_parent_path(task.path)
    assert len(children) == 2
    inputs = {child.input for child in children}
    assert inputs == {"input1", "input2"}


def test_taskrun_parent_task_run_id_persists_on_load(tmp_path):
    output = TaskOutput(output="test output")

    task = Task(
        name="Test Task",
        instruction="Test instruction",
        path=tmp_path / "task.kiln",
        turn_mode=TurnMode.multiturn,
    )
    task.save_to_file()

    parent_run = TaskRun(input="parent input", output=output, parent=task)
    parent_run.save_to_file()

    nested_run = TaskRun(
        input="nested input",
        output=output,
        parent=task,
        parent_task_run_id=parent_run.id,
    )
    nested_run.save_to_file()

    loaded_run = TaskRun.load_from_file(nested_run.path)
    assert loaded_run is not None
    assert loaded_run.input == "nested input"
    assert loaded_run.parent_task_run_id == parent_run.id
    loaded_parent_task = loaded_run.parent_task()
    assert loaded_parent_task is not None
    assert loaded_parent_task.name == "Test Task"
    assert loaded_parent_task.instruction == "Test instruction"
    assert loaded_run.load_parent() is not None
    assert loaded_run.load_parent().id == task.id


def test_taskrun_loads_from_task_path(tmp_path):
    """TaskRun children can be loaded from Task path."""
    output = TaskOutput(output="test output")

    task = Task(
        name="Test Task",
        instruction="Test instruction",
        path=tmp_path / "task.kiln",
    )
    task.save_to_file()

    run = TaskRun(input="test input", output=output, parent=task)
    run.save_to_file()

    # Load children from task path - should succeed
    children = list(TaskRun.iterate_children_paths_of_parent_path(task.path))
    assert len(children) == 1
    assert children[0] == run.path


def test_taskrun_fails_to_load_from_project_path(tmp_path):
    project_path = tmp_path / "project.kiln"
    project = Project(name="Test Project", path=project_path)
    project.save_to_file()

    with pytest.raises(ValidationError, match="validation error for Task"):
        list(TaskRun.iterate_children_paths_of_parent_path(project_path))


def test_multiple_runs_flat_under_task_with_parent_task_run_chain(tmp_path):
    output = TaskOutput(output="test output")

    task = Task(
        name="Test Task",
        instruction="Test instruction",
        path=tmp_path / "task.kiln",
        turn_mode=TurnMode.multiturn,
    )
    task.save_to_file()

    run1 = TaskRun(input="input1", output=output, parent=task)
    run1.save_to_file()

    run2 = TaskRun(
        input="input2",
        output=output,
        parent=task,
        parent_task_run_id=run1.id,
    )
    run2.save_to_file()

    run3 = TaskRun(
        input="input3",
        output=output,
        parent=task,
        parent_task_run_id=run2.id,
    )
    run3.save_to_file()

    task_children = TaskRun.all_children_of_parent_path(task.path)
    assert len(task_children) == 3
    by_id = {r.id: r for r in task_children}
    assert by_id[run2.id].parent_task_run_id == run1.id
    assert by_id[run3.id].parent_task_run_id == run2.id
    assert by_id[run1.id].parent_task_run_id is None


def test_iterate_children_wrong_parent_model_type(tmp_path):
    invalid_parent_path = tmp_path / "invalid.kiln"
    invalid_data = {
        "model_type": "project",
        "extra_field": "x",
    }
    with open(invalid_parent_path, "w") as f:
        json.dump(invalid_data, f)

    with pytest.raises(ValidationError, match="validation errors for Task"):
        list(TaskRun.iterate_children_paths_of_parent_path(invalid_parent_path))


def test_is_toolcall_pending_false_when_no_trace():
    run = TaskRun(input="test", output=TaskOutput(output="response"))
    assert run.is_toolcall_pending is False


def test_is_toolcall_pending_false_when_last_message_has_no_tool_calls():
    run = TaskRun(
        input="test",
        output=TaskOutput(output="response"),
        trace=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ],
    )
    assert run.is_toolcall_pending is False


def test_is_toolcall_pending_true_when_last_message_has_tool_calls():
    run = TaskRun(
        input="test",
        output=TaskOutput(output=""),
        trace=[
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
                        "type": "function",
                    }
                ],
            },
        ],
    )
    assert run.is_toolcall_pending is True


def test_is_toolcall_pending_false_when_tool_calls_followed_by_tool_response():
    run = TaskRun(
        input="test",
        output=TaskOutput(output="3"),
        trace=[
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "content": "3", "tool_call_id": "call_1"},
        ],
    )
    assert run.is_toolcall_pending is False


def test_is_toolcall_pending_false_when_only_task_response_tool_calls():
    trace = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_tr",
                    "function": {
                        "name": "task_response",
                        "arguments": '{"answer": 42}',
                    },
                    "type": "function",
                }
            ],
        },
    ]
    run = TaskRun(input="test", output=TaskOutput(output='{"answer": 42}'), trace=trace)
    assert run.is_toolcall_pending is False
    assert (
        RunOutput(
            output='{"answer": 42}', intermediate_outputs=None, trace=trace
        ).is_toolcall_pending
        is False
    )


def test_is_toolcall_pending_true_when_task_response_mixed_with_unmanaged_tools():
    trace = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_tr",
                    "function": {
                        "name": "task_response",
                        "arguments": '{"x": 1}',
                    },
                    "type": "function",
                },
                {
                    "id": "call_unmanaged",
                    "function": {"name": "unmanaged_do", "arguments": "{}"},
                    "type": "function",
                },
            ],
        },
    ]
    run = TaskRun(input="test", output=TaskOutput(output=""), trace=trace)
    assert run.is_toolcall_pending is True
    assert (
        RunOutput(output="", intermediate_outputs=None, trace=trace).is_toolcall_pending
        is True
    )


def test_task_turn_mode_defaults_to_single_turn():
    task = Task(name="Test Task", instruction="Test instruction")
    assert task.turn_mode == TurnMode.single_turn


def test_task_loads_legacy_json_without_turn_mode_as_single_turn():
    legacy_data = {
        "v": 1,
        "name": "Legacy Task",
        "instruction": "Legacy instruction",
        "model_type": "task",
    }

    parsed = Task.model_validate(legacy_data, context={"loading_from_file": True})
    assert parsed.turn_mode == TurnMode.single_turn

    parsed_no_context = Task.model_validate(legacy_data)
    assert parsed_no_context.turn_mode == TurnMode.single_turn


def test_task_multiturn_rejects_input_json_schema():
    with pytest.raises(ValidationError, match="structured input"):
        Task(
            name="Multiturn Task",
            instruction="Test instruction",
            turn_mode=TurnMode.multiturn,
            input_json_schema=json_joke_schema,
        )


def test_task_multiturn_rejects_output_json_schema():
    with pytest.raises(ValidationError, match="structured output"):
        Task(
            name="Multiturn Task",
            instruction="Test instruction",
            turn_mode=TurnMode.multiturn,
            output_json_schema=json_joke_schema,
        )


def test_task_single_turn_allows_both_schemas():
    task = Task(
        name="Structured Task",
        instruction="Test instruction",
        turn_mode=TurnMode.single_turn,
        input_json_schema=json_joke_schema,
        output_json_schema=json_joke_schema,
    )
    assert task.turn_mode == TurnMode.single_turn
    assert task.input_json_schema == json_joke_schema
    assert task.output_json_schema == json_joke_schema


def test_task_turn_mode_is_frozen_after_construction(tmp_path):
    """turn_mode is immutable at the SDK level. Assignment after construction
    must raise — the field is the load-bearing invariant for
    parent_task_run_id and dataset/eval iteration semantics."""
    task = Task(
        name="Test",
        instruction="i",
        path=tmp_path / "task.kiln",
        turn_mode=TurnMode.single_turn,
    )
    with pytest.raises(ValidationError, match="frozen"):
        task.turn_mode = TurnMode.multiturn  # type: ignore[misc]


def test_task_turn_mode_immutable_after_load(tmp_path):
    """Loading a task from disk and trying to mutate turn_mode must raise."""
    task = Task(
        name="Test",
        instruction="i",
        path=tmp_path / "task.kiln",
        turn_mode=TurnMode.multiturn,
    )
    task.save_to_file()
    loaded = Task.load_from_file(task.path)
    assert loaded.turn_mode == TurnMode.multiturn
    with pytest.raises(ValidationError, match="frozen"):
        loaded.turn_mode = TurnMode.single_turn  # type: ignore[misc]


def test_taskrun_parent_task_run_id_rejected_on_single_turn_task(tmp_path):
    """A TaskRun pointing to a parent_task_run_id is invalid when the parent
    Task is single-turn — Task.runs() would silently drop the parent."""
    task = Task(
        name="Single-turn",
        instruction="i",
        path=tmp_path / "task.kiln",
        turn_mode=TurnMode.single_turn,
    )
    task.save_to_file()
    parent_run = TaskRun(input="p", output=TaskOutput(output="o"), parent=task)
    parent_run.save_to_file()
    with pytest.raises(ValidationError, match="multi-turn"):
        TaskRun(
            input="c",
            output=TaskOutput(output="o"),
            parent=task,
            parent_task_run_id=parent_run.id,
        )


def test_taskrun_parent_task_run_id_allowed_on_multiturn_task(tmp_path):
    """The same construction must succeed for a multi-turn parent task."""
    task = Task(
        name="Multi-turn",
        instruction="i",
        path=tmp_path / "task.kiln",
        turn_mode=TurnMode.multiturn,
    )
    task.save_to_file()
    parent_run = TaskRun(input="p", output=TaskOutput(output="o"), parent=task)
    parent_run.save_to_file()
    child = TaskRun(
        input="c",
        output=TaskOutput(output="o"),
        parent=task,
        parent_task_run_id=parent_run.id,
    )
    assert child.parent_task_run_id == parent_run.id


def _make_task_for_runs_tests(
    tmp_path, turn_mode: TurnMode = TurnMode.multiturn
) -> Task:
    task = Task(
        name="Test Task",
        instruction="Test instruction",
        path=tmp_path / "task.kiln",
        turn_mode=turn_mode,
    )
    task.save_to_file()
    return task


def test_task_runs_defaults_to_leaves_only_for_multiturn_chain(tmp_path):
    task = _make_task_for_runs_tests(tmp_path)
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

    loaded_task = Task.load_from_file(task.path)
    leaves = loaded_task.runs()
    assert len(leaves) == 1
    assert leaves[0].id == run3.id


def test_task_runs_with_include_intermediate_runs_returns_all(tmp_path):
    task = _make_task_for_runs_tests(tmp_path)
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

    loaded_task = Task.load_from_file(task.path)
    all_runs = loaded_task.runs(include_intermediate_runs=True)
    assert {r.id for r in all_runs} == {run1.id, run2.id, run3.id}


def test_task_runs_branching_chain_returns_multiple_leaves(tmp_path):
    task = _make_task_for_runs_tests(tmp_path)
    output = TaskOutput(output="out")

    root = TaskRun(input="root", output=output, parent=task)
    root.save_to_file()
    leaf_a = TaskRun(
        input="a",
        output=output,
        parent=task,
        parent_task_run_id=root.id,
    )
    leaf_a.save_to_file()
    leaf_b = TaskRun(
        input="b",
        output=output,
        parent=task,
        parent_task_run_id=root.id,
    )
    leaf_b.save_to_file()

    loaded_task = Task.load_from_file(task.path)
    leaves = loaded_task.runs()
    leaf_ids = {r.id for r in leaves}
    assert leaf_ids == {leaf_a.id, leaf_b.id}
    assert root.id not in leaf_ids


def test_task_runs_readonly_smoke(tmp_path):
    task = _make_task_for_runs_tests(tmp_path)
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

    loaded_task = Task.load_from_file(task.path)
    leaves = loaded_task.runs(readonly=True)
    assert len(leaves) == 1
    assert leaves[0].id == run2.id
    assert leaves[0]._readonly is True

    all_runs = loaded_task.runs(readonly=True, include_intermediate_runs=True)
    assert {r.id for r in all_runs} == {run1.id, run2.id}
    assert all(r._readonly for r in all_runs)


def test_task_runs_single_turn_modes_equivalent(tmp_path):
    task = _make_task_for_runs_tests(tmp_path)
    output = TaskOutput(output="out")

    r1 = TaskRun(input="in1", output=output, parent=task)
    r1.save_to_file()
    r2 = TaskRun(input="in2", output=output, parent=task)
    r2.save_to_file()
    r3 = TaskRun(input="in3", output=output, parent=task)
    r3.save_to_file()

    loaded_task = Task.load_from_file(task.path)
    leaves = loaded_task.runs()
    all_runs = loaded_task.runs(include_intermediate_runs=True)
    assert {r.id for r in leaves} == {r.id for r in all_runs}
    assert {r.id for r in leaves} == {r1.id, r2.id, r3.id}


def test_task_runs_persist_under_runs_folder_not_underscore_runs(tmp_path):
    task = _make_task_for_runs_tests(tmp_path)
    output = TaskOutput(output="out")

    run = TaskRun(input="in", output=output, parent=task)
    run.save_to_file()

    runs_folder = tmp_path / "runs"
    underscore_runs_folder = tmp_path / "_runs"
    assert runs_folder.is_dir(), (
        "TaskRun files must live under 'runs/' on disk (filesystem_name)"
    )
    assert not underscore_runs_folder.exists(), (
        "TaskRun files must NOT live under '_runs/' (the Python attribute name)"
    )
    assert run.path is not None
    assert run.path.parent.parent == runs_folder
    assert run.path.exists()

    # Loading via the auto-generated _runs accessor and the wrapped runs()
    # method must both find the file on disk.
    loaded_task = Task.load_from_file(task.path)
    via_underscore = loaded_task._runs()  # type: ignore[attr-defined]
    via_wrapped = loaded_task.runs()
    assert {r.id for r in via_underscore} == {run.id}
    assert {r.id for r in via_wrapped} == {run.id}
    assert TaskRun.relationship_name() == "runs"


def test_task_runs_excludes_eval_generated_by_default(tmp_path):
    task = _make_task_for_runs_tests(tmp_path)
    output = TaskOutput(output="out")

    dataset_run = TaskRun(input="dataset", output=output, parent=task)
    dataset_run.save_to_file()
    eval_trace = TaskRun(
        input="trace",
        output=output,
        parent=task,
        eval_source=EvalItemSource(source_type="eval_input", source_id="ei1"),
    )
    eval_trace.save_to_file()

    loaded_task = Task.load_from_file(task.path)
    assert {r.id for r in loaded_task.runs()} == {dataset_run.id}
    assert {r.id for r in loaded_task.runs(include_eval_generated=True)} == {
        dataset_run.id,
        eval_trace.id,
    }


def test_task_runs_filters_compose(tmp_path):
    """The leaf filter and the eval-generated filter are independent: a run must pass
    both to be returned, so neither flag alone surfaces an eval-generated intermediate."""
    task = _make_task_for_runs_tests(tmp_path)
    output = TaskOutput(output="out")
    eval_source = EvalItemSource(source_type="task_run", source_id="item1")

    dataset_leaf = TaskRun(input="dataset", output=output, parent=task)
    dataset_leaf.save_to_file()

    eval_intermediate = TaskRun(
        input="eval turn 1", output=output, parent=task, eval_source=eval_source
    )
    eval_intermediate.save_to_file()
    eval_leaf = TaskRun(
        input="eval turn 2",
        output=output,
        parent=task,
        eval_source=eval_source,
        parent_task_run_id=eval_intermediate.id,
    )
    eval_leaf.save_to_file()

    loaded_task = Task.load_from_file(task.path)

    assert {r.id for r in loaded_task.runs()} == {dataset_leaf.id}
    assert {r.id for r in loaded_task.runs(include_intermediate_runs=True)} == {
        dataset_leaf.id
    }
    assert {r.id for r in loaded_task.runs(include_eval_generated=True)} == {
        dataset_leaf.id,
        eval_leaf.id,
    }
    assert {
        r.id
        for r in loaded_task.runs(
            include_intermediate_runs=True, include_eval_generated=True
        )
    } == {dataset_leaf.id, eval_intermediate.id, eval_leaf.id}


def test_task_runs_eval_child_does_not_hide_dataset_parent(tmp_path):
    """A filtered-out eval-generated run must not count as a leaf-filter parent: an eval
    child chained onto a curated dataset run would otherwise remove that dataset run from
    the default view while itself being filtered, so the row vanishes entirely."""
    task = _make_task_for_runs_tests(tmp_path)
    output = TaskOutput(output="out")

    dataset_run = TaskRun(input="dataset", output=output, parent=task)
    dataset_run.save_to_file()
    eval_child = TaskRun(
        input="eval child",
        output=output,
        parent=task,
        eval_source=EvalItemSource(source_type="task_run", source_id="item1"),
        parent_task_run_id=dataset_run.id,
    )
    eval_child.save_to_file()

    loaded_task = Task.load_from_file(task.path)

    # Default view: the eval child is excluded, and the dataset run it chains onto
    # stays visible.
    assert {r.id for r in loaded_task.runs()} == {dataset_run.id}
    # With eval-generated runs included the leaf filter sees the whole chain, so the
    # dataset run is a parent and only the eval child is a leaf. Unchanged behavior.
    assert {r.id for r in loaded_task.runs(include_eval_generated=True)} == {
        eval_child.id
    }
    assert {
        r.id
        for r in loaded_task.runs(
            include_intermediate_runs=True, include_eval_generated=True
        )
    } == {dataset_run.id, eval_child.id}


def test_content_part_trace_bulk_load_after_readonly_scan(tmp_path):
    """Pydantic validates list-valued message content into a lazy, single-use iterator.
    A readonly scan caches the loaded instance; a later default (mutable) load deep-copies
    the cached model, which crashes unless the content was materialized at validation."""
    task = _make_task_for_runs_tests(tmp_path, turn_mode=TurnMode.single_turn)
    run = TaskRun(
        input="in",
        output=TaskOutput(output="out"),
        parent=task,
        trace=[
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": "hi"},
        ],
    )
    run.save_to_file()

    # Load from disk and populate the model cache, as any readonly runs scan does.
    ModelCache.shared().clear()
    assert {r.id for r in task.runs(readonly=True)} == {run.id}

    # Cache hit on the default (mutable) path: returns a deep copy of the cached model.
    loaded = TaskRun.from_ids_and_parent_path({run.id}, task.path)
    assert loaded[run.id].trace is not None
    assert loaded[run.id].trace[0]["content"] == [{"type": "text", "text": "hello"}]

    # The readonly passthrough returns the cached instance without copying.
    readonly_loaded = TaskRun.from_ids_and_parent_path(
        {run.id}, task.path, readonly=True
    )
    readonly_run = readonly_loaded[run.id]
    assert readonly_run.trace is not None
    assert readonly_run.trace[0]["content"] == [{"type": "text", "text": "hello"}]
