import json
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from kiln_ai.datamodel.basemodel import KilnParentModel, ReadOnlyMutationError
from kiln_ai.datamodel.datamodel_enums import EvalStatus, Priority
from kiln_ai.datamodel.eval import (
    LEGACY_TRACE_FIELDS,
    SCORER_CODE_FILENAME,
    ArgMatch,
    CodeEvalProperties,
    ContainsProperties,
    Eval,
    EvalConfig,
    EvalConfigType,
    EvalDataType,
    EvalInput,
    EvalInputSplit,
    EvalOutputScore,
    EvalRun,
    EvalTaskInput,
    EvalTemplateId,
    ExactMatchProperties,
    LlmJudgeProperties,
    MultiTurnDriveConfig,
    MultiTurnSyntheticEvalInputData,
    PatternMatchProperties,
    SetCheckProperties,
    SingleTurnEvalInputData,
    SkippedReason,
    StepCountCheckProperties,
    SyntheticUserInfo,
    TaskRunSplit,
    ToolCallCheckProperties,
    ToolCallSpec,
    UserMessage,
    V2EvalResult,
    V2EvalType,
    reference_data_keys,
    validate_scores_against_output_scores,
)
from kiln_ai.datamodel.spec import Spec
from kiln_ai.datamodel.spec_properties import DesiredBehaviourProperties, SpecType
from kiln_ai.datamodel.task import Task
from kiln_ai.datamodel.task_output import TaskOutput, TaskOutputRatingType
from kiln_ai.datamodel.task_run import TaskRun
from kiln_ai.datamodel.usage import Usage


@pytest.fixture
def mock_task():
    return Task(name="Test Task", instruction="Test instruction")


@pytest.fixture
def valid_eval_config_data():
    return {
        "name": "Test Eval Config",
        "config_type": EvalConfigType.g_eval,
        "properties": {"eval_steps": ["step1", "step2"]},
        "model_name": "gpt-4",
        "model_provider": "openai",
    }


@pytest.fixture
def valid_eval_config(valid_eval_config_data):
    return EvalConfig(**valid_eval_config_data)


def test_eval_config_valid(valid_eval_config):
    assert valid_eval_config.name == "Test Eval Config"
    assert valid_eval_config.config_type == EvalConfigType.g_eval
    assert valid_eval_config.properties["eval_steps"] == ["step1", "step2"]
    assert valid_eval_config.model_name == "gpt-4"
    assert valid_eval_config.model_provider == "openai"


def test_eval_config_missing_eval_steps(valid_eval_config):
    with pytest.raises(
        ValueError, match="eval_steps is required and must be a list for g_eval"
    ):
        valid_eval_config.properties = {}


def test_eval_config_missing_task_description(valid_eval_config):
    with pytest.raises(
        ValueError,
        match="task_description is optional, but if provided must be a string",
    ):
        valid_eval_config.properties = {"task_description": 123, "eval_steps": []}


def test_eval_config_invalid_json(valid_eval_config):
    class InvalidClass:
        pass

    with pytest.raises(ValueError, match="Properties must be JSON serializable"):
        valid_eval_config.properties = {
            "eval_steps": [],
            "invalid_key": InvalidClass(),
        }


def test_eval_config_invalid_eval_steps_type(valid_eval_config):
    with pytest.raises(
        ValueError, match="eval_steps is required and must be a list for g_eval"
    ):
        valid_eval_config.properties = {"eval_steps": "not a list"}


def test_eval_config_invalid_config_type(valid_eval_config):
    # Create an invalid config type using string
    with pytest.raises(ValueError):
        valid_eval_config.config_type = "invalid_type"


def test_eval_basic_properties():
    eval = Eval(
        name="Test Eval",
        description="Test Description",
        current_config_id="config123",
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.five_star,
            )
        ],
    )

    assert eval.name == "Test Eval"
    assert eval.description == "Test Description"
    assert eval.current_config_id == "config123"
    assert eval.output_scores[0].name == "accuracy"
    assert eval.output_scores[0].type == TaskOutputRatingType.five_star


def test_eval_with_train_set_filter_id():
    """The deprecated filter fields are accepted as input and land in `splits`."""
    eval = Eval(
        name="Test Eval",
        eval_set_filter_id="tag::eval_test",
        train_set_filter_id="tag::eval_train_test",
        eval_configs_filter_id="tag::eval_golden_test",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
    )

    assert eval.splits["test"] == TaskRunSplit(filter_id="tag::eval_test")
    assert eval.splits["train"] == TaskRunSplit(filter_id="tag::eval_train_test")
    # The golden set is not a split, and eval_configs_filter_id is not deprecated.
    assert eval.eval_configs_filter_id == "tag::eval_golden_test"


def test_eval_train_set_filter_id_defaults_to_none():
    """No train filter in, no train split out."""
    eval = Eval(
        name="Test Eval",
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="score",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
    )

    assert "train" not in eval.splits


def test_no_train_split_minted_on_load(mock_task, tmp_path):
    """An eval loaded without a train split has none. Nothing mints one."""
    task_path = tmp_path / "task.kiln"
    mock_task.path = task_path
    mock_task.save_to_file()

    eval = Eval(
        name="My Eval Name",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        train_set_filter_id=None,
        output_scores=[
            EvalOutputScore(
                name="score",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
    )
    eval.save_to_file()

    loaded_eval = Eval.load_from_file(str(eval.path))
    assert "train" not in loaded_eval.splits


def test_train_set_filter_id_survives_round_trip(mock_task, tmp_path):
    """An explicitly set train filter survives the save, as a split."""
    task_path = tmp_path / "task.kiln"
    mock_task.path = task_path
    mock_task.save_to_file()

    eval = Eval(
        name="My Eval",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        train_set_filter_id="tag::custom_train_tag",
        output_scores=[
            EvalOutputScore(
                name="score",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
    )
    eval.save_to_file()

    loaded_eval = Eval.load_from_file(str(eval.path))
    assert loaded_eval.splits["train"] == TaskRunSplit(
        filter_id="tag::custom_train_tag"
    )


def test_no_train_split_minted_on_new_eval():
    """A newly constructed eval without a train filter has no train split."""
    eval = Eval(
        name="New Eval",
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        train_set_filter_id=None,
        output_scores=[
            EvalOutputScore(
                name="score",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
    )
    assert "train" not in eval.splits


def test_eval_default_values():
    eval = Eval(
        name="Test Eval",
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="quality",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
    )

    assert eval.description is None
    assert eval.current_config_id is None


def test_eval_parent_task_relationship(mock_task, valid_eval_config_data):
    eval = Eval(
        name="Test Eval",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="score",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
    )
    config = EvalConfig(parent=eval, **valid_eval_config_data)

    assert eval.parent_task() == mock_task
    assert eval.parent == mock_task
    assert config.parent == eval
    assert config.parent_eval() == eval


def test_eval_parent_task_none():
    eval = Eval(
        name="Test Eval",
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="score",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
    )
    assert eval.parent_task() is None


def test_eval_parent_task_wrong_type():
    # Create a non-Task parent
    class DummyParent(KilnParentModel, parent_of={}):
        pass

    with pytest.raises(ValueError):
        Eval(name="Test Eval", parent=DummyParent())


def test_eval_with_persisted_children(mock_task, valid_eval_config_data, tmp_path):
    task_path = tmp_path / "task.kiln"
    mock_task.path = task_path
    mock_task.save_to_file()

    eval = Eval(
        name="Test Eval",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
    )
    eval.save_to_file()

    # Add config using the parent relationship
    config = EvalConfig(parent=eval, **valid_eval_config_data)
    config.save_to_file()

    run = EvalRun(
        parent=config,
        dataset_id="dataset123",
        task_run_config_id="config456",
        input='{"key": "value"}',
        output='{"result": "success"}',
        scores={"accuracy": 0.95},
    )
    run.save_to_file()

    # Test configs can be retrieved from disk
    evals = mock_task.evals()
    assert len(evals) == 1
    assert evals[0].name == "Test Eval"
    configs = evals[0].configs()
    assert len(configs) == 1
    assert configs[0].model_provider == "openai"
    assert configs[0].model_name == "gpt-4"

    # and back up
    assert configs[0].parent_eval().parent_task().path == task_path

    # Test runs can be retrieved from disk
    runs = configs[0].runs()
    assert len(runs) == 1
    assert runs[0].dataset_id == "dataset123"
    assert runs[0].task_run_config_id == "config456"
    assert runs[0].input == '{"key": "value"}'
    assert runs[0].output == '{"result": "success"}'
    assert runs[0].scores == {"accuracy": 0.95}

    # and back up
    assert runs[0].parent_eval_config().parent_eval().parent_task().path == task_path


def test_eval_run_valid_creation():
    """Test creating an EvalRun with valid data"""
    eval_run = EvalRun(
        dataset_id="dataset123",
        task_run_config_id="config456",
        input='{"key": "value"}',  # JSON formatted input
        output='{"result": "success"}',  # JSON formatted output
        scores={"accuracy": 0.95},
    )

    assert eval_run.dataset_id == "dataset123"
    assert eval_run.task_run_config_id == "config456"
    assert eval_run.input == '{"key": "value"}'
    assert eval_run.output == '{"result": "success"}'
    assert eval_run.scores == {"accuracy": 0.95}


def test_eval_run_plaintext():
    """Test creating an EvalRun with plaintext input/output"""
    eval_run = EvalRun(
        dataset_id="dataset123",
        task_run_config_id="config456",
        input="What is the capital of France?",
        output="The capital of France is Paris.",
        scores={"accuracy": 1.0},
    )

    assert eval_run.input == "What is the capital of France?"
    assert eval_run.output == "The capital of France is Paris."


def test_eval_run_missing_required_fields():
    """Test that omitting required fields raises ValidationError"""
    with pytest.raises(ValidationError) as exc_info:
        EvalRun(
            dataset_id="dataset123",
            # missing task_run_config_id
            input="test",
            output="test",
            scores={"score": 1.0},
        )

    assert "task_run_config_id" in str(exc_info.value)


def test_eval_run_invalid_scores():
    """Test that scores must be a dict of floats"""
    with pytest.raises(ValidationError):
        EvalRun(
            dataset_id="dataset123",
            task_run_config_id="config456",
            input="test",
            output="test",
            scores={"score": "not a float"},  # invalid score type
        )


def test_eval_missing_output_scores():
    """Test that eval creation fails when output_scores is missing"""
    with pytest.raises(ValidationError) as exc_info:
        Eval(
            name="Test Eval",
            eval_set_filter_id="tag::tag1",
            eval_configs_filter_id="tag::tag2",
        )
    assert "output_scores" in str(exc_info.value)


def test_eval_empty_output_scores():
    """Test that eval creation fails when output_scores is empty"""
    with pytest.raises(
        ValueError, match="output_scores are required, and must have at least one score"
    ):
        Eval(
            name="Test Eval",
            eval_set_filter_id="tag::tag1",
            eval_configs_filter_id="tag::tag2",
            output_scores=[],
        )


def test_eval_duplicate_output_scores():
    """Test that eval creation fails when output_scores has duplicate names"""
    with pytest.raises(
        ValueError,
        match="must have unique names",
    ):
        Eval(
            name="Test Eval",
            eval_set_filter_id="tag::tag1",
            eval_configs_filter_id="tag::tag2",
            output_scores=[
                EvalOutputScore(
                    name="score",
                    type=TaskOutputRatingType.five_star,
                ),
                EvalOutputScore(name="SCORE", type=TaskOutputRatingType.pass_fail),
            ],
        )


def test_eval_invalid_score_type():
    """Test that eval creation fails with invalid rating type in output_scores"""
    with pytest.raises(
        ValueError,
        match="Input should be 'five_star', 'pass_fail', 'pass_fail_critical'",
    ):
        Eval(
            name="Test Eval",
            eval_set_filter_id="tag::tag1",
            eval_configs_filter_id="tag::tag2",
            output_scores=[
                EvalOutputScore(
                    name="score",
                    type="invalid_type",
                )
            ],
        )


def test_eval_valid_output_scores():
    """Test that eval creation succeeds with valid output_scores"""
    eval = Eval(
        name="Test Eval",
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.five_star,
            ),
            EvalOutputScore(
                name="critical_check",
                type=TaskOutputRatingType.pass_fail_critical,
            ),
            EvalOutputScore(name="basic_check", type=TaskOutputRatingType.pass_fail),
        ],
    )
    assert len(eval.output_scores) == 3
    assert eval.output_scores[0].type == TaskOutputRatingType.five_star
    assert eval.output_scores[0].name == "accuracy"
    assert eval.output_scores[1].type == TaskOutputRatingType.pass_fail_critical
    assert eval.output_scores[1].name == "critical_check"
    assert eval.output_scores[2].type == TaskOutputRatingType.pass_fail
    assert eval.output_scores[2].name == "basic_check"


def test_eval_output_score_name_validation():
    """Test that EvalOutputScore validates score names properly"""

    with pytest.raises(
        ValueError,
        match="cannot contain any of the following characters",
    ):
        EvalOutputScore(
            name="Correctness ",
            type=TaskOutputRatingType.five_star,
        )

    with pytest.raises(
        ValueError,
        match="cannot contain any of the following characters",
    ):
        EvalOutputScore(
            name=" Leading Space",
            type=TaskOutputRatingType.five_star,
        )

    with pytest.raises(
        ValueError,
        match="cannot contain any of the following characters",
    ):
        EvalOutputScore(
            name="consecutive__underscores",
            type=TaskOutputRatingType.five_star,
        )

    with pytest.raises(
        ValueError,
        match="cannot contain any of the following characters",
    ):
        EvalOutputScore(
            name="invalid/slash",
            type=TaskOutputRatingType.five_star,
        )

    with pytest.raises(
        ValueError,
        match="cannot contain any of the following characters",
    ):
        EvalOutputScore(
            name="invalid.period",
            type=TaskOutputRatingType.five_star,
        )

    with pytest.raises(ValueError, match="too long"):
        EvalOutputScore(
            name="a" * 33,
            type=TaskOutputRatingType.five_star,
        )

    valid_score = EvalOutputScore(
        name="Valid Name With Spaces",
        type=TaskOutputRatingType.five_star,
    )
    assert valid_score.name == "Valid Name With Spaces"

    max_length_score = EvalOutputScore(
        name="a" * 32,
        type=TaskOutputRatingType.five_star,
    )
    assert max_length_score.name == "a" * 32


@pytest.fixture
def valid_eval_run_data():
    return {
        "dataset_id": "dataset123",
        "task_run_config_id": "config456",
        "input": "test input",
        "output": "test output",
        "scores": {"accuracy": 4.5},
    }


def test_eval_run_five_star_score_validation(valid_eval_config, valid_eval_run_data):
    # Setup eval with five_star rating
    eval = Eval(
        name="Test Eval",
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.five_star,
            )
        ],
    )
    valid_eval_config.parent = eval

    # Valid score
    run = EvalRun(parent=valid_eval_config, **valid_eval_run_data)
    assert run.scores["accuracy"] == 4.5

    # Invalid scores
    with pytest.raises(ValueError, match=r"must be a number between 1.0 and 5.0"):
        run = EvalRun(
            parent=valid_eval_config,
            **{**valid_eval_run_data, "scores": {"accuracy": 0.5}},
        )

    with pytest.raises(ValueError, match=r"must be a number between 1.0 and 5.0"):
        run = EvalRun(
            parent=valid_eval_config,
            **{**valid_eval_run_data, "scores": {"accuracy": 5.5}},
        )


def test_eval_run_pass_fail_score_validation(valid_eval_config, valid_eval_run_data):
    # Setup eval with pass_fail rating
    eval = Eval(
        name="Test Eval",
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="check",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
    )
    valid_eval_config.parent = eval

    # Valid scores
    run = EvalRun(
        parent=valid_eval_config, **{**valid_eval_run_data, "scores": {"check": 1.0}}
    )
    assert run.scores["check"] == 1.0

    run = EvalRun(
        parent=valid_eval_config, **{**valid_eval_run_data, "scores": {"check": 0.0}}
    )
    assert run.scores["check"] == 0.0

    # Invalid scores
    with pytest.raises(ValueError, match=r"must be a number between 0.0 and 1.0"):
        run = EvalRun(
            parent=valid_eval_config,
            **{**valid_eval_run_data, "scores": {"check": -0.1}},
        )

    with pytest.raises(ValueError, match=r"must be a number between 0.0 and 1.0"):
        run = EvalRun(
            parent=valid_eval_config,
            **{**valid_eval_run_data, "scores": {"check": 1.1}},
        )


def test_eval_run_pass_fail_critical_score_validation(
    valid_eval_config, valid_eval_run_data
):
    # Setup eval with pass_fail_critical rating
    eval = Eval(
        name="Test Eval",
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="critical",
                type=TaskOutputRatingType.pass_fail_critical,
            )
        ],
    )
    valid_eval_config.parent = eval

    # Valid scores
    run = EvalRun(
        parent=valid_eval_config, **{**valid_eval_run_data, "scores": {"critical": 1.0}}
    )
    assert run.scores["critical"] == 1.0

    run = EvalRun(
        parent=valid_eval_config,
        **{**valid_eval_run_data, "scores": {"critical": -1.0}},
    )
    assert run.scores["critical"] == -1.0

    # Invalid scores
    with pytest.raises(ValueError, match=r"must be a number between -1.0 and 1.0"):
        run = EvalRun(
            parent=valid_eval_config,
            **{**valid_eval_run_data, "scores": {"critical": -1.1}},
        )

    with pytest.raises(ValueError, match=r"must be a number between -1.0 and 1.0"):
        run = EvalRun(
            parent=valid_eval_config,
            **{**valid_eval_run_data, "scores": {"critical": 1.1}},
        )


def test_eval_run_score_keys_must_match(valid_eval_config, valid_eval_run_data):
    eval = Eval(
        name="Test Eval",
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.five_star,
            ),
            EvalOutputScore(
                name="critical",
                type=TaskOutputRatingType.pass_fail_critical,
            ),
        ],
    )
    valid_eval_config.parent = eval

    # Correct
    EvalRun(
        parent=valid_eval_config,
        **{**valid_eval_run_data, "scores": {"accuracy": 4.5, "critical": 1.0}},
    )

    # Correct but wrong order still okay
    EvalRun(
        parent=valid_eval_config,
        **{**valid_eval_run_data, "scores": {"critical": 1.0, "accuracy": 4.5}},
    )

    # Missing score
    with pytest.raises(
        ValueError,
        match="The scores produced by the evaluator must match the scores expected by the eval",
    ):
        EvalRun(
            parent=valid_eval_config,
            **{**valid_eval_run_data, "scores": {"accuracy": 4.5}},
        )

    # Extra score
    with pytest.raises(
        ValueError,
        match="The scores produced by the evaluator must match the scores expected by the eval",
    ):
        EvalRun(
            parent=valid_eval_config,
            **{
                **valid_eval_run_data,
                "scores": {"accuracy": 4.5, "critical": 1.0, "extra": 1.0},
            },
        )

    # Missing score w matching count
    with pytest.raises(
        ValueError,
        match="The scores produced by the evaluator must match the scores expected by the eval",
    ):
        EvalRun(
            parent=valid_eval_config,
            **{**valid_eval_run_data, "scores": {"accuracy": 4.5, "wrong": 1.0}},
        )


def test_eval_run_custom_scores_not_allowed(valid_eval_config, valid_eval_run_data):
    with pytest.raises(
        ValueError, match="Custom scores are not supported in evaluators"
    ):
        Eval(
            name="Test Eval",
            eval_set_filter_id="tag::tag1",
            eval_configs_filter_id="tag::tag2",
            output_scores=[
                EvalOutputScore(
                    name="custom",
                    type=TaskOutputRatingType.custom,
                )
            ],
        )


def test_eval_run_eval_config_eval_validation():
    """Test that eval_config_eval and task_run_config_id validation works correctly"""

    # Case 1: Valid configuration - eval_config_eval=True and task_run_config_id=None
    valid_run1 = EvalRun(
        dataset_id="dataset123",
        eval_config_eval=True,
        task_run_config_id=None,
        input="test input",
        output="test output",
        scores={"score": 1.0},
    )
    assert valid_run1.eval_config_eval is True
    assert valid_run1.task_run_config_id is None

    # Case 2: Valid configuration - eval_config_eval=False and task_run_config_id is set
    valid_run2 = EvalRun(
        dataset_id="dataset123",
        eval_config_eval=False,
        task_run_config_id="config456",
        input="test input",
        output="test output",
        scores={"score": 1.0},
    )
    assert valid_run2.eval_config_eval is False
    assert valid_run2.task_run_config_id == "config456"

    # Case 3: Invalid configuration - eval_config_eval=True but task_run_config_id is set
    with pytest.raises(
        ValueError, match="task_run_config_id must be None if eval_config_eval is true"
    ):
        EvalRun(
            dataset_id="dataset123",
            eval_config_eval=True,
            task_run_config_id="config456",
            input="test input",
            output="test output",
            scores={"score": 1.0},
        )

    # Case 4: Invalid configuration - eval_config_eval=False but task_run_config_id is None
    with pytest.raises(
        ValueError, match="task_run_config_id must be set if eval_config_eval is false"
    ):
        EvalRun(
            dataset_id="dataset123",
            eval_config_eval=False,
            task_run_config_id=None,
            input="test input",
            output="test output",
            scores={"score": 1.0},
        )


@pytest.mark.parametrize(
    "template_properties,should_raise,expected_error",
    [
        # Valid cases
        (
            {"issue_prompt": "Test issue prompt"},
            False,
            None,
        ),
        (
            {
                "issue_prompt": "Test issue prompt",
                "failure_example": "Test failure example",
            },
            False,
            None,
        ),
        (
            {
                "issue_prompt": "Test issue prompt",
                "failure_example": "Test failure example",
                "pass_example": "Test pass example",
            },
            False,
            None,
        ),
        (
            {
                "issue_prompt": "",
                "failure_example": "",
                "pass_example": "",
            },
            False,
            None,
        ),
        # Invalid cases
        (
            {},
            True,
            "issue_prompt is required for issue template",
        ),
        (
            {"failure_example": "Test failure example"},
            True,
            "issue_prompt is required for issue template",
        ),
        (
            {"issue_prompt": 123},
            True,
            "issue_prompt is required for issue template",
        ),
        (
            {
                "issue_prompt": "Test issue prompt",
                "failure_example": 456,
            },
            True,
            "failure_example is optional for issue template, but if provided must be a string",
        ),
        (
            {
                "issue_prompt": "Test issue prompt",
                "failure_example": "Test failure example",
                "pass_example": 789,
            },
            True,
            "pass_example is optional for issue template, but if provided must be a string",
        ),
    ],
)
def test_eval_template_properties_issue_template_validation(
    template_properties, should_raise, expected_error
):
    """Test issue template validation with various property combinations"""
    if should_raise:
        with pytest.raises(ValueError, match=expected_error):
            Eval(
                name="Test Eval",
                template=EvalTemplateId.issue,
                eval_set_filter_id="tag::tag1",
                eval_configs_filter_id="tag::tag2",
                output_scores=[
                    EvalOutputScore(
                        name="score",
                        type=TaskOutputRatingType.pass_fail,
                    )
                ],
                template_properties=template_properties,
            )
    else:
        eval = Eval(
            name="Test Eval",
            template=EvalTemplateId.issue,
            eval_set_filter_id="tag::tag1",
            eval_configs_filter_id="tag::tag2",
            output_scores=[
                EvalOutputScore(
                    name="score",
                    type=TaskOutputRatingType.pass_fail,
                )
            ],
            template_properties=template_properties,
        )
        assert eval.template == EvalTemplateId.issue
        for key, value in template_properties.items():
            assert (
                eval.template_properties is not None
                and eval.template_properties[key] == value
            )


@pytest.mark.parametrize(
    "template,template_properties",
    [
        (EvalTemplateId.kiln_requirements, {"random_property": "random_value"}),
        (EvalTemplateId.toxicity, {}),
        (EvalTemplateId.bias, {"some_property": 123}),
        (EvalTemplateId.maliciousness, {"test": True}),
        (EvalTemplateId.factual_correctness, {"score": 4.5}),
        (EvalTemplateId.jailbreak, {"prompt": "test"}),
        (
            None,
            {"issue_prompt": "This should not be validated", "failure_example": 123},
        ),
    ],
)
def test_eval_template_properties_non_validated_templates(
    template, template_properties
):
    """Test that templates without specific validation pass regardless of template_properties"""
    eval = Eval(
        name="Test Eval",
        template=template,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="score",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
        template_properties=template_properties,
    )
    assert eval.template == template
    for key, value in template_properties.items():
        assert (
            eval.template_properties is not None
            and eval.template_properties[key] == value
        )


@pytest.mark.parametrize(
    "template_properties,should_raise,expected_error",
    [
        # Valid cases
        (
            {
                "tool": "search_tool",
                "tool_function_name": "search",
                "appropriate_tool_use_guidelines": "Call the tool when user asks for search",
            },
            False,
            None,
        ),
        (
            {
                "tool": "calculator",
                "tool_function_name": "calculate",
                "appropriate_tool_use_guidelines": "Call the tool for math calculations",
                "inappropriate_tool_use_guidelines": "Don't call the tool for simple math",
            },
            False,
            None,
        ),
        (
            {
                "tool": "weather_api",
                "tool_function_name": "get_weather",
                "appropriate_tool_use_guidelines": "Call the tool when user asks about weather",
            },
            False,
            None,
        ),
        (
            {
                "tool": "database_query",
                "tool_function_name": "query_db",
                "appropriate_tool_use_guidelines": "Call for data retrieval requests",
                "inappropriate_tool_use_guidelines": "Don't call for personal questions",
            },
            False,
            None,
        ),
        (
            {
                "tool": "",
                "tool_function_name": "",
                "appropriate_tool_use_guidelines": "",
                "inappropriate_tool_use_guidelines": "",
            },
            True,
            "tool is required for tool call template",
        ),
        # Invalid cases - missing required fields
        (
            {},
            True,
            "tool is required for tool call template",
        ),
        (
            {"tool_function_name": "search"},
            True,
            "tool is required for tool call template",
        ),
        (
            {"tool": "search_tool"},
            True,
            "tool_function_name is required for tool call template",
        ),
        (
            {"tool": "search_tool", "tool_function_name": "search"},
            True,
            "appropriate_tool_use_guidelines is required for tool call template",
        ),
        # Invalid cases - wrong types
        (
            {"tool": 123, "tool_function_name": "search"},
            True,
            "tool is required for tool call template",
        ),
        (
            {"tool": "search_tool", "tool_function_name": 456},
            True,
            "tool_function_name is required for tool call template",
        ),
        (
            {
                "tool": "search_tool",
                "tool_function_name": "search",
                "appropriate_tool_use_guidelines": 123,
            },
            True,
            "appropriate_tool_use_guidelines is required for tool call template",
        ),
        (
            {
                "tool": "search_tool",
                "tool_function_name": "search",
                "appropriate_tool_use_guidelines": "Call for data retrieval requests",
                "inappropriate_tool_use_guidelines": 789,
            },
            True,
            "inappropriate_tool_use_guidelines is optional for tool call template, but if provided must be a string",
        ),
    ],
)
def test_eval_template_properties_tool_call_template_validation(
    template_properties, should_raise, expected_error
):
    """Test tool call template validation with various property combinations"""
    if should_raise:
        with pytest.raises(ValueError, match=expected_error):
            Eval(
                name="Test Eval",
                template=EvalTemplateId.tool_call,
                evaluation_data_type=EvalDataType.full_trace,
                eval_set_filter_id="tag::tag1",
                eval_configs_filter_id="tag::tag2",
                output_scores=[
                    EvalOutputScore(
                        name="score",
                        type=TaskOutputRatingType.pass_fail,
                    )
                ],
                template_properties=template_properties,
            )
    else:
        eval = Eval(
            name="Test Eval",
            template=EvalTemplateId.tool_call,
            evaluation_data_type=EvalDataType.full_trace,
            eval_set_filter_id="tag::tag1",
            eval_configs_filter_id="tag::tag2",
            output_scores=[
                EvalOutputScore(
                    name="score",
                    type=TaskOutputRatingType.pass_fail,
                )
            ],
            template_properties=template_properties,
        )
        assert eval.template == EvalTemplateId.tool_call
        for key, value in template_properties.items():
            assert (
                eval.template_properties is not None
                and eval.template_properties[key] == value
            )


def test_eval_tool_call_template_requires_full_trace_evaluation_data_type():
    """Test that tool_call template requires evaluation_data_type to be full_trace"""
    valid_template_properties: dict[str, str | int | bool | float] = {
        "tool": "search_tool",
        "tool_function_name": "search",
        "appropriate_tool_use_guidelines": "Call the tool when user asks for search",
    }

    # Valid case: tool_call template with full_trace
    eval = Eval(
        name="Test Eval",
        template=EvalTemplateId.tool_call,
        evaluation_data_type=EvalDataType.full_trace,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="score",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
        template_properties=valid_template_properties,
    )
    assert eval.template == EvalTemplateId.tool_call
    assert eval.evaluation_data_type == EvalDataType.full_trace

    # Invalid case: tool_call template with final_answer (default)
    with pytest.raises(
        ValueError,
        match="tool_call template should have evaluation_data_type set to full_trace",
    ):
        Eval(
            name="Test Eval",
            template=EvalTemplateId.tool_call,
            evaluation_data_type=EvalDataType.final_answer,
            eval_set_filter_id="tag::tag1",
            eval_configs_filter_id="tag::tag2",
            output_scores=[
                EvalOutputScore(
                    name="score",
                    type=TaskOutputRatingType.pass_fail,
                )
            ],
            template_properties=valid_template_properties,
        )

    # Invalid case: tool_call template with evaluation_data_type omitted (defaults to final_answer)
    with pytest.raises(
        ValueError,
        match="tool_call template should have evaluation_data_type set to full_trace",
    ):
        Eval(
            name="Test Eval",
            template=EvalTemplateId.tool_call,
            eval_set_filter_id="tag::tag1",
            eval_configs_filter_id="tag::tag2",
            output_scores=[
                EvalOutputScore(
                    name="score",
                    type=TaskOutputRatingType.pass_fail,
                )
            ],
            template_properties=valid_template_properties,
        )


@pytest.mark.parametrize(
    "template,eval_configs_filter_id,should_raise,expected_error",
    [
        # RAG template can have None
        (EvalTemplateId.rag, None, False, None),
        (EvalTemplateId.rag, "tag::tag2", False, None),
        # Other templates require eval_configs_filter_id
        (
            EvalTemplateId.issue,
            None,
            True,
            "eval_configs_filter_id is required for all templates except 'rag'",
        ),
        (
            EvalTemplateId.tool_call,
            None,
            True,
            "eval_configs_filter_id is required for all templates except 'rag'",
        ),
        (
            EvalTemplateId.kiln_requirements,
            None,
            True,
            "eval_configs_filter_id is required for all templates except 'rag'",
        ),
        (
            EvalTemplateId.toxicity,
            None,
            True,
            "eval_configs_filter_id is required for all templates except 'rag'",
        ),
        (
            EvalTemplateId.bias,
            None,
            True,
            "eval_configs_filter_id is required for all templates except 'rag'",
        ),
        (
            EvalTemplateId.maliciousness,
            None,
            True,
            "eval_configs_filter_id is required for all templates except 'rag'",
        ),
        (
            EvalTemplateId.factual_correctness,
            None,
            True,
            "eval_configs_filter_id is required for all templates except 'rag'",
        ),
        (
            EvalTemplateId.jailbreak,
            None,
            True,
            "eval_configs_filter_id is required for all templates except 'rag'",
        ),
        # None template skips template-specific validation
        (None, None, False, None),
        # Valid cases with eval_configs_filter_id provided
        (EvalTemplateId.issue, "tag::tag2", False, None),
        (EvalTemplateId.tool_call, "tag::tag2", False, None),
        (None, "tag::tag2", False, None),
    ],
)
def test_eval_configs_filter_id_validation(
    template, eval_configs_filter_id, should_raise, expected_error
):
    """Test that eval_configs_filter_id is required for all templates except 'rag'"""
    template_properties = {}
    if template == EvalTemplateId.issue:
        template_properties = {"issue_prompt": "Test issue prompt"}
    elif template == EvalTemplateId.tool_call:
        template_properties = {
            "tool": "search_tool",
            "tool_function_name": "search",
            "appropriate_tool_use_guidelines": "Call the tool when user asks for search",
        }

    eval_kwargs = {
        "name": "Test Eval",
        "template": template,
        "eval_set_filter_id": "tag::tag1",
        "eval_configs_filter_id": eval_configs_filter_id,
        "output_scores": [
            EvalOutputScore(
                name="score",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
        "template_properties": template_properties,
    }

    if template == EvalTemplateId.tool_call:
        eval_kwargs["evaluation_data_type"] = EvalDataType.full_trace

    if should_raise:
        with pytest.raises(ValueError, match=expected_error):
            Eval(**eval_kwargs)
    else:
        eval = Eval(**eval_kwargs)
        assert eval.template == template
        assert eval.eval_configs_filter_id == eval_configs_filter_id


def test_eval_run_trace_property(mock_task, valid_eval_config_data, tmp_path):
    """Test EvalRun with trace property"""
    task_path = tmp_path / "task.kiln"
    mock_task.path = task_path
    mock_task.save_to_file()

    eval = Eval(
        name="Test Eval",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
        evaluation_data_type=EvalDataType.full_trace,
    )
    eval.save_to_file()

    config = EvalConfig(parent=eval, **valid_eval_config_data)
    config.save_to_file()

    trace_data = '{"messages": [{"role": "user", "content": "test"}]}'
    eval_run = EvalRun(
        parent=config,
        dataset_id="dataset123",
        task_run_config_id="config456",
        input="test input",
        output="test output",
        scores={"accuracy": 0.95},
        task_run_trace=trace_data,
    )
    eval_run.save_to_file()

    # Verify the properties are saved correctly
    assert eval_run.task_run_trace == trace_data
    assert isinstance(eval_run.task_run_trace, str)

    # Verify persistence by reloading from disk
    runs = config.runs()
    assert len(runs) == 1
    assert runs[0].task_run_trace == trace_data


def test_eval_run_new_properties_default_none(
    mock_task, valid_eval_config_data, tmp_path
):
    """Test that new properties default to None when not provided"""
    task_path = tmp_path / "task.kiln"
    mock_task.path = task_path
    mock_task.save_to_file()

    eval = Eval(
        name="Test Eval",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
    )
    eval.save_to_file()

    config = EvalConfig(parent=eval, **valid_eval_config_data)
    config.save_to_file()

    eval_run = EvalRun(
        parent=config,
        dataset_id="dataset123",
        task_run_config_id="config456",
        input="test input",
        output="test output",
        scores={"accuracy": 0.95},
    )
    eval_run.save_to_file()

    # Verify the properties default to None
    assert eval_run.task_run_trace is None

    # Verify persistence by reloading from disk
    runs = config.runs()
    assert len(runs) == 1
    assert runs[0].task_run_trace is None


def test_eval_data_type_enum_values():
    """Test EvalDataType enum has correct values"""
    assert EvalDataType.final_answer == "final_answer"
    assert EvalDataType.full_trace == "full_trace"


def test_eval_default_evaluation_data_type():
    """Test that Eval defaults to final_answer for evaluation_data_type"""
    eval = Eval(
        name="Test Eval",
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="score",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
    )

    assert eval.evaluation_data_type == EvalDataType.final_answer


def test_eval_custom_evaluation_data_type():
    """Test Eval with custom evaluation_data_type"""
    eval = Eval(
        name="Test Eval",
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="score",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
        evaluation_data_type=EvalDataType.full_trace,
    )

    assert eval.evaluation_data_type == EvalDataType.full_trace


@pytest.mark.parametrize(
    "evaluation_data_type",
    [EvalDataType.final_answer, EvalDataType.full_trace],
)
def test_eval_all_evaluation_data_types(evaluation_data_type):
    """Test Eval with all possible evaluation_data_type values"""
    eval = Eval(
        name="Test Eval",
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="score",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
        evaluation_data_type=evaluation_data_type,
    )

    assert eval.evaluation_data_type == evaluation_data_type


def test_eval_run_eval_config_eval_data_type_validation(
    mock_task, valid_eval_config_data, tmp_path
):
    """Test that eval_config_eval works with all evaluation data types"""
    task_path = tmp_path / "task.kiln"
    mock_task.path = task_path
    mock_task.save_to_file()

    # Test with final_answer - should work
    eval_final_answer = Eval(
        name="Test Eval Final Answer",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
        evaluation_data_type=EvalDataType.final_answer,
    )
    eval_final_answer.save_to_file()

    config_final_answer = EvalConfig(parent=eval_final_answer, **valid_eval_config_data)
    config_final_answer.save_to_file()

    # This should work - eval_config_eval with final_answer
    EvalRun(
        parent=config_final_answer,
        dataset_id="dataset123",
        eval_config_eval=True,
        task_run_config_id=None,
        input="test input",
        output="test output",
        scores={"accuracy": 0.95},
    )

    # Test with full_trace - should work
    eval_full_trace = Eval(
        name="Test Eval Full Trace",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
        evaluation_data_type=EvalDataType.full_trace,
    )
    eval_full_trace.save_to_file()

    config_full_trace = EvalConfig(parent=eval_full_trace, **valid_eval_config_data)
    config_full_trace.save_to_file()

    # This should work - eval_config_eval with full_trace
    EvalRun(
        parent=config_full_trace,
        dataset_id="dataset123",
        eval_config_eval=True,
        task_run_config_id=None,
        input="test input",
        output="test output",
        scores={"accuracy": 0.95},
        task_run_trace='{"messages": [{"role": "user", "content": "test"}]}',
    )


def test_validate_output_fields_final_answer_valid_cases(
    mock_task, valid_eval_config_data
):
    """Test validate_output_fields with final_answer evaluation data type - valid cases"""
    eval = Eval(
        name="Test Eval",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
        evaluation_data_type=EvalDataType.final_answer,
    )
    config = EvalConfig(parent=eval, **valid_eval_config_data)

    # Valid case: no full_trace
    run = EvalRun(
        parent=config,
        dataset_id="dataset123",
        task_run_config_id="config456",
        input="test input",
        output="test output",
        scores={"accuracy": 0.95},
    )
    assert run.task_run_trace is None

    # Valid case: explicitly set to None
    run = EvalRun(
        parent=config,
        dataset_id="dataset123",
        task_run_config_id="config456",
        input="test input",
        output="test output",
        scores={"accuracy": 0.95},
        task_run_trace=None,
    )
    assert run.task_run_trace is None


def test_validate_output_fields_final_answer_invalid_cases(
    mock_task, valid_eval_config_data
):
    """Test validate_output_fields with final_answer evaluation data type - invalid cases"""
    eval = Eval(
        name="Test Eval",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
        evaluation_data_type=EvalDataType.final_answer,
    )
    config = EvalConfig(parent=eval, **valid_eval_config_data)

    # Invalid case: full_trace is set
    with pytest.raises(
        ValueError,
        match="final_answer runs should not set trace",
    ):
        EvalRun(
            parent=config,
            dataset_id="dataset123",
            task_run_config_id="config456",
            input="test input",
            output="test output",
            scores={"accuracy": 0.95},
            task_run_trace='{"messages": []}',
        )


def test_validate_output_fields_full_trace_valid_cases(
    mock_task, valid_eval_config_data
):
    """Test validate_output_fields with full_trace evaluation data type - valid cases"""
    eval = Eval(
        name="Test Eval",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
        evaluation_data_type=EvalDataType.full_trace,
    )
    config = EvalConfig(parent=eval, **valid_eval_config_data)

    # Valid case: full_trace is set
    run = EvalRun(
        parent=config,
        dataset_id="dataset123",
        task_run_config_id="config456",
        input="test input",
        output="test output",
        scores={"accuracy": 0.95},
        task_run_trace='{"messages": [{"role": "user", "content": "test"}]}',
    )
    assert run.task_run_trace == '{"messages": [{"role": "user", "content": "test"}]}'


def test_validate_output_fields_full_trace_invalid_cases(
    mock_task, valid_eval_config_data
):
    """Test validate_output_fields with full_trace evaluation data type - invalid cases"""
    eval = Eval(
        name="Test Eval",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
        evaluation_data_type=EvalDataType.full_trace,
    )
    config = EvalConfig(parent=eval, **valid_eval_config_data)

    # Invalid case: trace is omitted
    with pytest.raises(
        ValueError, match="full_trace task run eval runs should include trace"
    ):
        EvalRun(
            parent=config,
            dataset_id="dataset123",
            task_run_config_id="config456",
            input="test input",
            output="test output",
            scores={"accuracy": 0.95},
        )

    # Invalid case: trace is explicitly None
    with pytest.raises(
        ValueError, match="full_trace task run eval runs should include trace"
    ):
        EvalRun(
            parent=config,
            dataset_id="dataset123",
            task_run_config_id="config456",
            input="test input",
            output="test output",
            scores={"accuracy": 0.95},
            task_run_trace=None,
        )


def test_validate_output_fields_no_parent_eval(valid_eval_config_data):
    """Test validate_output_fields when there is no parent eval (should still validate mutual exclusivity)"""
    # Create a config without a parent eval
    config = EvalConfig(**valid_eval_config_data)

    # This should work - no parent eval means validation passes
    run = EvalRun(
        parent=config,
        dataset_id="dataset123",
        task_run_config_id="config456",
        input="test input",
        output="test output",
        scores={"accuracy": 0.95},
        task_run_trace='{"messages": []}',
    )
    assert run.task_run_trace == '{"messages": []}'


def test_validate_output_fields_no_parent_eval_config():
    """Test validate_output_fields when there is no parent eval config (should pass)"""
    # Create a run without a parent
    run = EvalRun(
        dataset_id="dataset123",
        task_run_config_id="config456",
        input="test input",
        output="test output",
        scores={"accuracy": 0.95},
        task_run_trace='{"messages": []}',
    )
    assert run.task_run_trace == '{"messages": []}'


@pytest.mark.parametrize(
    "evaluation_data_type,trace,should_raise,expected_error",
    [
        # final_answer cases
        (EvalDataType.final_answer, None, False, None),
        (
            EvalDataType.final_answer,
            '{"messages": []}',
            True,
            "final_answer runs should not set trace",
        ),
        # full_trace cases
        (EvalDataType.full_trace, '{"messages": []}', False, None),
        (
            EvalDataType.full_trace,
            None,
            True,
            "full_trace task run eval runs should include trace",
        ),
    ],
)
def test_validate_output_fields_parametrized(
    mock_task,
    valid_eval_config_data,
    evaluation_data_type,
    trace,
    should_raise,
    expected_error,
):
    """Test validate_output_fields with parametrized test cases"""
    eval = Eval(
        name="Test Eval",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
        evaluation_data_type=evaluation_data_type,
    )
    config = EvalConfig(parent=eval, **valid_eval_config_data)

    run_data = {
        "parent": config,
        "dataset_id": "dataset123",
        "task_run_config_id": "config456",
        "input": "test input",
        "output": "test output",
        "scores": {"accuracy": 0.95},
    }

    if trace is not None:
        run_data["task_run_trace"] = trace

    if should_raise:
        with pytest.raises(ValueError, match=expected_error):
            EvalRun(**run_data)
    else:
        run = EvalRun(**run_data)
        assert run.task_run_trace == trace


# ── V2 validate_output_fields: writer-shape matrix ─────────────────────────
#
# The V2 eval writers attach task_run_trace for exactly one shape: a scored
# (non-skipped), non-eval-config task run of a full_trace eval. These pin the
# datamodel gate to that shape so a writer dropping the trace is rejected, while
# every shape the writer legitimately leaves trace-less continues to pass.

V2_TRACE = '{"messages": [{"role": "user", "content": "test"}]}'


def _v2_eval_and_config(mock_task, data_type=EvalDataType.full_trace):
    """A V2 (typed-properties) config parented to an eval of the given type."""
    eval = Eval(
        name="V2 Eval",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(name="accuracy", type=TaskOutputRatingType.pass_fail)
        ],
        evaluation_data_type=data_type,
    )
    config = EvalConfig(
        parent=eval,
        name="V2 Config",
        config_type=EvalConfigType.v2,
        properties=LlmJudgeProperties(
            model_name="gpt-4o",
            model_provider="openai",
            prompt_template="Evaluate: {{ final_message }}",
        ),
    )
    return eval, config


def test_v2_full_trace_task_run_eval_with_trace_passes(mock_task):
    _, config = _v2_eval_and_config(mock_task)
    run = EvalRun(
        parent=config,
        eval_input_id="ei1",
        task_run_config_id="rc1",
        input="in",
        output="out",
        scores={"accuracy": 1.0},
        task_run_trace=V2_TRACE,
    )
    assert run.task_run_trace == V2_TRACE


def test_v2_full_trace_task_run_eval_without_trace_rejected(mock_task):
    _, config = _v2_eval_and_config(mock_task)
    with pytest.raises(
        ValueError, match="full_trace task run eval runs should include trace"
    ):
        EvalRun(
            parent=config,
            eval_input_id="ei1",
            task_run_config_id="rc1",
            input="in",
            output="out",
            scores={"accuracy": 1.0},
        )


def test_v2_full_trace_skipped_record_without_trace_passes(mock_task):
    _, config = _v2_eval_and_config(mock_task)
    run = EvalRun(
        parent=config,
        eval_input_id="ei1",
        task_run_config_id="rc1",
        input="in",
        output=None,
        scores={},
        skipped_reason=SkippedReason.missing_trace.value,
    )
    assert run.task_run_trace is None


def test_v2_full_trace_eval_config_eval_without_trace_passes(mock_task):
    _, config = _v2_eval_and_config(mock_task)
    run = EvalRun(
        parent=config,
        dataset_id="ds1",
        eval_config_eval=True,
        task_run_config_id=None,
        input="in",
        output="out",
        scores={"accuracy": 1.0},
    )
    assert run.task_run_trace is None


def test_v2_final_answer_record_without_trace_passes(mock_task):
    _, config = _v2_eval_and_config(mock_task, data_type=EvalDataType.final_answer)
    run = EvalRun(
        parent=config,
        eval_input_id="ei1",
        task_run_config_id="rc1",
        input="in",
        output="out",
        scores={"accuracy": 1.0},
    )
    assert run.task_run_trace is None


def test_v2_full_trace_trace_less_record_loads_from_file(mock_task, tmp_path):
    """Historical trace-less full_trace records must still load; the gate holds
    only new writes and rebuilds, not files already on disk."""
    mock_task.path = tmp_path / "task.kiln"
    mock_task.save_to_file()
    eval, config = _v2_eval_and_config(mock_task)
    eval.save_to_file()
    config.save_to_file()

    run = EvalRun(
        parent=config,
        eval_input_id="ei1",
        task_run_config_id="rc1",
        input="in",
        output="out",
        scores={"accuracy": 1.0},
        task_run_trace=V2_TRACE,
    )
    run.save_to_file()

    # Rewrite the persisted record to the trace-less shape a pre-gate writer
    # could have produced, then confirm it still loads rather than erroring.
    assert run.path is not None
    data = json.loads(run.path.read_text())
    data["task_run_trace"] = None
    run.path.write_text(json.dumps(data))

    loaded = EvalRun.load_from_file(str(run.path))
    assert loaded.task_run_trace is None


@pytest.mark.parametrize(
    "evaluation_data_type,reference_answer,should_raise,expected_error",
    [
        # reference_answer eval type - valid cases
        (EvalDataType.reference_answer, "answer text", False, None),
        (EvalDataType.reference_answer, None, False, None),
        # final_answer eval type
        (EvalDataType.final_answer, None, False, None),
        (
            EvalDataType.final_answer,
            "answer text",
            True,
            r"reference_answer is only valid for reference answer evals\. Got: final_answer",
        ),
        # full_trace eval type
        (EvalDataType.full_trace, None, False, None),
        (
            EvalDataType.full_trace,
            "answer text",
            True,
            r"reference_answer is only valid for reference answer evals\. Got: full_trace",
        ),
    ],
)
def test_validate_reference_answer_parametrized(
    mock_task,
    valid_eval_config_data,
    evaluation_data_type,
    reference_answer,
    should_raise,
    expected_error,
):
    """Test validate_reference_answer with parametrized test cases"""
    eval = Eval(
        name="Test Eval",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
        evaluation_data_type=evaluation_data_type,
    )
    config = EvalConfig(parent=eval, **valid_eval_config_data)

    run_data = {
        "parent": config,
        "dataset_id": "dataset123",
        "task_run_config_id": "config456",
        "input": "test input",
        "output": "test output",
        "scores": {"accuracy": 0.95},
    }

    if reference_answer is not None:
        run_data["reference_answer"] = reference_answer

    if evaluation_data_type == EvalDataType.full_trace:
        run_data["task_run_trace"] = (
            '{"messages": [{"role": "user", "content": "test"}]}'
        )

    if should_raise:
        with pytest.raises(ValueError, match=expected_error):
            EvalRun(**run_data)
    else:
        run = EvalRun(**run_data)
        assert run.reference_answer == reference_answer


def test_eval_upgrade_old_reference_answer_eval_config(mock_task, tmp_path):
    """Test that reference answer evals with no current_config_id get the first config set as default."""
    # Create an eval with reference_answer type and save to disk
    task = mock_task
    task.path = tmp_path / "task.kiln"
    task.save_to_file()

    eval = Eval(
        name="Test Eval",
        parent=task,
        evaluation_data_type=EvalDataType.reference_answer,
        eval_set_filter_id="all",
        eval_configs_filter_id="high_rating",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
    )
    eval.save_to_file()

    # Create two configs with different created_at times
    from datetime import datetime, timedelta

    config1 = EvalConfig(
        parent=eval,
        name="First Config",
        model_name="gpt-4",
        model_provider="openai",
        config_type=EvalConfigType.g_eval,
        properties={"eval_steps": ["step1"]},
    )
    config1.created_at = datetime.now().astimezone()
    config1.save_to_file()

    config2 = EvalConfig(
        parent=eval,
        name="Second Config",
        model_name="gpt-4",
        model_provider="openai",
        config_type=EvalConfigType.g_eval,
        properties={"eval_steps": ["step1"]},
    )
    config2.created_at = datetime.now().astimezone() + timedelta(seconds=1)
    config2.save_to_file()

    # Load from file - should set the first (oldest) config as default
    loaded_eval = Eval.load_from_file(str(eval.path))
    assert loaded_eval.current_config_id == config1.id  # First by created_at

    # Test with current_config_id already set - should not change it
    eval.current_config_id = config2.id
    eval.save_to_file()
    loaded_eval = Eval.load_from_file(str(eval.path))
    assert loaded_eval.current_config_id == config2.id  # Should keep existing value

    # Test with non-reference_answer type - should not set current_config_id
    eval.evaluation_data_type = EvalDataType.final_answer
    eval.current_config_id = None
    eval.save_to_file()
    loaded_eval = Eval.load_from_file(str(eval.path))
    assert (
        loaded_eval.current_config_id is None
    )  # Should not set for non-reference_answer

    # Test with no configs - should not error
    eval.evaluation_data_type = EvalDataType.reference_answer
    eval.current_config_id = None
    eval.save_to_file()
    # Delete config files
    if config1.path is not None:
        config1.path.unlink()
    if config2.path is not None:
        config2.path.unlink()
    loaded_eval = Eval.load_from_file(str(eval.path))
    assert loaded_eval.current_config_id is None  # No configs to set


# ── V1 Characterization Tests ──────────────────────────────────────────


def test_v1_eval_config_loads_from_disk(mock_task, tmp_path):
    """Characterization: V1 g_eval config round-trips through disk without corruption."""
    task_path = tmp_path / "task.kiln"
    mock_task.path = task_path
    mock_task.save_to_file()

    eval = Eval(
        name="Chartest",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(name="accuracy", type=TaskOutputRatingType.pass_fail)
        ],
    )
    eval.save_to_file()

    config = EvalConfig(
        name="GEval Config",
        parent=eval,
        config_type=EvalConfigType.g_eval,
        model_name="gpt-4",
        model_provider="openai",
        properties={"eval_steps": ["step1", "step2"], "task_description": "desc"},
    )
    config.save_to_file()

    loaded = EvalConfig.load_from_file(str(config.path))
    assert loaded.config_type == EvalConfigType.g_eval
    assert loaded.model_name == "gpt-4"
    assert loaded.model_provider == "openai"
    assert isinstance(loaded.properties, dict)
    assert loaded.properties["eval_steps"] == ["step1", "step2"]


def test_v1_eval_run_with_reference_answer(mock_task, tmp_path):
    """Characterization: V1 eval run with a reference_answer saves and loads."""
    task_path = tmp_path / "task.kiln"
    mock_task.path = task_path
    mock_task.save_to_file()

    eval = Eval(
        name="RefAnswer Eval",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        evaluation_data_type=EvalDataType.reference_answer,
        output_scores=[
            EvalOutputScore(name="score", type=TaskOutputRatingType.pass_fail)
        ],
    )
    eval.save_to_file()

    config = EvalConfig(
        name="Ref Config",
        parent=eval,
        config_type=EvalConfigType.g_eval,
        model_name="gpt-4",
        model_provider="openai",
        properties={"eval_steps": ["check ref"]},
    )
    config.save_to_file()

    run = EvalRun(
        parent=config,
        dataset_id="ds1",
        task_run_config_id="rc1",
        input="What?",
        output="Answer",
        reference_answer="Gold answer",
        scores={"score": 0.9},
    )
    run.save_to_file()

    loaded = EvalRun.load_from_file(str(run.path))
    assert loaded.reference_answer == "Gold answer"
    assert loaded.scores == {"score": 0.9}
    assert loaded.dataset_id == "ds1"


# ── V2 EvalConfig Tests ────────────────────────────────────────────────


def test_v2_eval_config_valid():
    """V2 config with typed LlmJudgeProperties is accepted."""
    config = EvalConfig(
        name="V2 Config",
        config_type=EvalConfigType.v2,
        properties=LlmJudgeProperties(
            model_name="gpt-4o",
            model_provider="openai",
            prompt_template="Evaluate: {{ final_message }}",
        ),
    )
    assert config.config_type == EvalConfigType.v2
    assert isinstance(config.properties, LlmJudgeProperties)
    assert config.model_name is None
    assert config.model_provider is None


def test_v2_eval_config_rejects_root_model_fields():
    """V2 config must NOT set root-level model_name / model_provider."""
    with pytest.raises(ValueError, match="must not set root-level model_name"):
        EvalConfig(
            name="Bad V2",
            config_type=EvalConfigType.v2,
            model_name="gpt-4o",
            model_provider="openai",
            properties=LlmJudgeProperties(
                model_name="gpt-4o",
                model_provider="openai",
                prompt_template="t",
            ),
        )


def test_v2_eval_config_rejects_undiscriminated_dict():
    """A V2 properties dict without a valid "type" discriminator is rejected."""
    with pytest.raises(ValidationError, match="type"):
        EvalConfig(
            name="Bad V2",
            config_type=EvalConfigType.v2,
            properties={"eval_steps": ["step"]},
        )


def test_v2_eval_config_requires_typed_properties():
    """V2 config rejects missing properties."""
    with pytest.raises(ValueError, match="V2 config requires typed properties"):
        EvalConfig(
            name="Bad V2",
            config_type=EvalConfigType.v2,
            properties=None,
        )


def test_legacy_config_unchanged():
    """Legacy g_eval config still validates the same as before."""
    config = EvalConfig(
        name="Legacy",
        config_type=EvalConfigType.g_eval,
        model_name="gpt-4",
        model_provider="openai",
        properties={"eval_steps": ["s1"]},
    )
    assert isinstance(config.properties, dict)


def test_legacy_config_requires_model_fields():
    """Legacy config rejects missing model_name / model_provider."""
    with pytest.raises(ValueError, match="model_name and model_provider are required"):
        EvalConfig(
            name="Legacy Missing",
            config_type=EvalConfigType.g_eval,
            properties={"eval_steps": ["s1"]},
        )


def test_v2_json_serializable_bypass():
    """V2 bypass of validate_json_serializable (which would fail for typed props)."""
    config = EvalConfig(
        name="V2 Bypass",
        config_type=EvalConfigType.v2,
        properties=ExactMatchProperties(expected_value="hello"),
    )
    assert config.config_type == EvalConfigType.v2


def test_v2_eval_config_discriminated_union_dispatch():
    """V2 properties discriminated union dispatches by type field."""
    config = EvalConfig(
        name="Pattern",
        config_type=EvalConfigType.v2,
        properties=PatternMatchProperties(pattern=r"\\d+"),
    )
    assert isinstance(config.properties, PatternMatchProperties)

    config2 = EvalConfig(
        name="Contains",
        config_type=EvalConfigType.v2,
        properties=ContainsProperties(substring="hello"),
    )
    assert isinstance(config2.properties, ContainsProperties)


# ── V2 EvalConfig Properties Validators ────────────────────────────────


def test_exact_match_xor_validator():
    """ExactMatchProperties requires exactly one of expected_value/reference_key."""
    with pytest.raises(
        ValueError, match="Exactly one of expected_value or reference_key"
    ):
        ExactMatchProperties(expected_value="a", reference_key="b")
    with pytest.raises(
        ValueError, match="Exactly one of expected_value or reference_key"
    ):
        ExactMatchProperties()

    assert ExactMatchProperties(expected_value="hello").expected_value == "hello"
    assert ExactMatchProperties(reference_key="key1").reference_key == "key1"


def test_contains_xor_validator():
    """ContainsProperties requires exactly one of substring/reference_key."""
    with pytest.raises(ValueError, match="Exactly one of substring or reference_key"):
        ContainsProperties(substring="a", reference_key="b")
    with pytest.raises(ValueError, match="Exactly one of substring or reference_key"):
        ContainsProperties()


def test_set_check_xor_validator():
    """SetCheckProperties requires exactly one of expected_set/reference_key."""
    with pytest.raises(
        ValueError, match="Exactly one of expected_set or reference_key"
    ):
        SetCheckProperties(expected_set=["a"], reference_key="b", mode="equal")
    with pytest.raises(
        ValueError, match="Exactly one of expected_set or reference_key"
    ):
        SetCheckProperties(mode="subset")

    assert SetCheckProperties(expected_set=["x"], mode="equal").expected_set == ["x"]


def test_set_check_mode_required():
    """SetCheckProperties.mode is required; omitting it raises ValidationError."""
    with pytest.raises(ValidationError):
        SetCheckProperties(expected_set=["a"])


def test_set_check_mode_explicit_values():
    """Each mode value works when explicitly provided."""
    for m in ("subset", "superset", "equal"):
        props = SetCheckProperties(expected_set=["a"], mode=m)
        assert props.mode == m


def test_step_count_check_bounds():
    """StepCountCheckProperties requires at least one of min/max, min <= max."""
    with pytest.raises(ValueError, match="at least one of min_count"):
        StepCountCheckProperties(count_type="tool_calls")
    with pytest.raises(ValueError, match="min_count must be <= max_count"):
        StepCountCheckProperties(count_type="turns", min_count=5, max_count=2)

    ok = StepCountCheckProperties(count_type="model_responses", min_count=1)
    assert ok.min_count == 1
    assert ok.max_count is None


# ── V2 Eval Tests ──────────────────────────────────────────────────────


def test_eval_v2_with_eval_input_split():
    """An EvalInput-backed test split is authored directly in `splits`."""
    eval = Eval(
        name="V2 Eval",
        splits={"test": EvalInputSplit(filter_id="all")},
        eval_configs_filter_id="tag::cfg",
        output_scores=[
            EvalOutputScore(name="score", type=TaskOutputRatingType.pass_fail)
        ],
    )
    assert eval.splits["test"] == EvalInputSplit(filter_id="all")
    assert eval.model_dump()["eval_set_filter_id"] is None


def test_eval_requires_a_test_split():
    """An eval with no test split, in either home, raises."""
    with pytest.raises(ValueError, match="must have a test split"):
        Eval(
            name="Neither",
            eval_configs_filter_id="tag::cfg",
            output_scores=[
                EvalOutputScore(name="s", type=TaskOutputRatingType.pass_fail)
            ],
        )


def test_eval_legacy_only_validates():
    """An eval carrying only legacy fields validates.

    Guards the declaration order of migrate_legacy_split_fields and validate_splits: if
    the migration stopped running first, this would raise 'must have a test split'.
    """
    eval = Eval(
        name="Legacy",
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::cfg",
        output_scores=[EvalOutputScore(name="s", type=TaskOutputRatingType.pass_fail)],
    )
    assert eval.splits["test"] == TaskRunSplit(filter_id="tag::tag1")


def test_eval_optional_evaluation_data_type():
    """evaluation_data_type defaults to final_answer."""
    eval = Eval(
        name="Default DT",
        eval_set_filter_id="tag::t",
        eval_configs_filter_id="tag::t2",
        output_scores=[EvalOutputScore(name="s", type=TaskOutputRatingType.pass_fail)],
    )
    assert eval.evaluation_data_type == EvalDataType.final_answer


def test_validate_template_properties_none_template():
    """When template is None, validate_template_properties returns early."""
    eval = Eval(
        name="No Template",
        eval_set_filter_id="tag::t",
        eval_configs_filter_id="tag::t2",
        template=None,
        output_scores=[EvalOutputScore(name="s", type=TaskOutputRatingType.pass_fail)],
    )
    assert eval.template is None


# ── V2 EvalRun Tests ───────────────────────────────────────────────────


def test_eval_run_v2_with_eval_input_id():
    """V2 eval run uses eval_input_id instead of dataset_id."""
    run = EvalRun(
        eval_input_id="ei_123",
        task_run_config_id="rc1",
        input="hi",
        output="hello",
        scores={"s": 1.0},
    )
    assert run.eval_input_id == "ei_123"
    assert run.dataset_id is None


def test_eval_run_input_source_xor():
    """Exactly one of dataset_id / eval_input_id must be set."""
    with pytest.raises(
        ValueError,
        match=r"Exactly one of dataset_id \(V1 TaskRun source\) or eval_input_id \(V2 EvalInput source\)",
    ):
        EvalRun(
            dataset_id="d1",
            eval_input_id="ei1",
            task_run_config_id="rc1",
            input="i",
            output="o",
            scores={"s": 1.0},
        )
    with pytest.raises(
        ValueError,
        match=r"Exactly one of dataset_id \(V1 TaskRun source\) or eval_input_id \(V2 EvalInput source\)",
    ):
        EvalRun(
            task_run_config_id="rc1",
            input="i",
            output="o",
            scores={"s": 1.0},
        )


def test_eval_run_skipped_allows_empty_scores():
    """When skipped_reason is set, empty scores are allowed."""
    run = EvalRun(
        eval_input_id="ei1",
        task_run_config_id="rc1",
        input="i",
        output="o",
        skipped_reason=SkippedReason.missing_reference_key.value,
        skipped_detail="key 'expected' not found",
        scores={},
    )
    assert run.skipped_reason == "missing_reference_key"
    assert run.scores == {}


def test_eval_run_skipped_allows_none_output():
    """Skipped runs can have None output."""
    run = EvalRun(
        eval_input_id="ei1",
        task_run_config_id="rc1",
        input="i",
        output=None,
        skipped_reason=SkippedReason.extraction_failed.value,
        scores={},
    )
    assert run.output is None


def test_eval_run_v2_bypass_output_fields():
    """V2 config_type bypasses validate_output_fields and validate_reference_answer."""
    eval = Eval(
        name="V2 Parent",
        splits={"test": EvalInputSplit(filter_id="all")},
        eval_configs_filter_id="tag::cfg",
        evaluation_data_type=EvalDataType.final_answer,
        output_scores=[
            EvalOutputScore(name="score", type=TaskOutputRatingType.pass_fail)
        ],
    )
    config = EvalConfig(
        name="V2 Config",
        parent=eval,
        config_type=EvalConfigType.v2,
        properties=ExactMatchProperties(expected_value="hello"),
    )
    run = EvalRun(
        parent=config,
        eval_input_id="ei1",
        task_run_config_id="rc1",
        input="i",
        output="hello",
        reference_answer="should be accepted in v2",
        scores={"score": 1.0},
    )
    assert run.reference_answer == "should be accepted in v2"


def test_eval_run_not_skipped_requires_scores():
    """Non-skipped runs with empty scores raise ValueError."""
    with pytest.raises(ValueError, match="scores are required"):
        EvalRun(
            eval_input_id="ei1",
            task_run_config_id="rc1",
            input="i",
            output="o",
            scores={},
        )


# ── EvalInput Tests ────────────────────────────────────────────────────


def test_eval_input_single_turn():
    """EvalInput with single_turn data."""
    ei = EvalInput(
        data=SingleTurnEvalInputData(user_message=UserMessage(text="What is 2+2?")),
    )
    assert ei.data.type == "single_turn"
    assert ei.data.user_message.text == "What is 2+2?"


def test_eval_input_multi_turn():
    """EvalInput with multi_turn_synthetic data carries the typed persona."""
    ei = EvalInput(
        data=MultiTurnSyntheticEvalInputData(
            first_message=UserMessage(text="Hello"),
            synthetic_user_info=SyntheticUserInfo(
                persona="student", goal="pass the exam"
            ),
        ),
    )
    assert ei.data.type == "multi_turn_synthetic"
    assert ei.data.first_message.text == "Hello"
    assert ei.data.synthetic_user_info.persona == "student"
    assert ei.data.synthetic_user_info.goal == "pass the exam"
    assert ei.data.synthetic_user_info.behavior_guidance is None


def test_eval_input_rejects_empty_tag():
    """An empty tag is rejected: tag filters can never select it."""
    with pytest.raises(ValidationError, match="Tags cannot be empty strings"):
        EvalInput(
            data=SingleTurnEvalInputData(user_message=UserMessage(text="hi")),
            tags=[""],
        )


def test_eval_input_rejects_tag_with_spaces():
    """A tag containing spaces is rejected, matching TaskRun tag rules."""
    with pytest.raises(ValidationError, match="Tags cannot contain spaces"):
        EvalInput(
            data=SingleTurnEvalInputData(user_message=UserMessage(text="hi")),
            tags=["bad tag"],
        )


def test_eval_input_valid_tags_round_trip():
    """Valid tags are accepted and survive a dump/validate cycle."""
    tags = ["eval_slice", "scenario:1", "synthetic_user_batch:b1"]
    ei = EvalInput(
        data=SingleTurnEvalInputData(user_message=UserMessage(text="hi")),
        tags=tags,
    )
    assert ei.tags == tags
    rebuilt = EvalInput.model_validate(ei.model_dump())
    assert rebuilt.tags == tags


def test_multi_turn_synthetic_requires_synthetic_user_info():
    """The persona is required — a case without one can't be re-driven."""
    with pytest.raises(ValidationError, match="synthetic_user_info"):
        MultiTurnSyntheticEvalInputData(first_message=UserMessage(text="Hello"))


def test_synthetic_user_info_preserves_unknown_fields():
    """extra="allow": fields from newer generators survive a load/dump cycle."""
    info = SyntheticUserInfo.model_validate(
        {"persona": "p", "goal": "g", "speaking_style": "terse"}
    )
    assert info.model_dump()["speaking_style"] == "terse"


def test_eval_input_multi_turn_persists_under_task(mock_task, tmp_path):
    """Multi-turn EvalInput round-trips through disk with the typed persona."""
    task_path = tmp_path / "task.kiln"
    mock_task.path = task_path
    mock_task.save_to_file()

    ei = EvalInput(
        parent=mock_task,
        data=MultiTurnSyntheticEvalInputData(
            first_message=UserMessage(text="opening message"),
            synthetic_user_info=SyntheticUserInfo(
                persona="frustrated customer",
                goal="get a refund",
                behavior_guidance="be polite then escalate",
            ),
        ),
        tags=["eval_slice", "scenario:2"],
    )
    ei.save_to_file()

    loaded_task = Task.load_from_file(str(task_path))
    inputs = loaded_task.eval_inputs(readonly=True)
    assert len(inputs) == 1
    data = inputs[0].data
    assert isinstance(data, MultiTurnSyntheticEvalInputData)
    assert data.first_message is not None
    assert data.first_message.text == "opening message"
    assert data.synthetic_user_info.persona == "frustrated customer"
    assert data.synthetic_user_info.goal == "get a refund"
    assert data.synthetic_user_info.behavior_guidance == "be polite then escalate"
    assert inputs[0].tags == ["eval_slice", "scenario:2"]


def test_eval_input_with_reference():
    """EvalInput with reference data."""
    ei = EvalInput(
        data=SingleTurnEvalInputData(user_message=UserMessage(text="Q")),
        reference={"expected_answer": "A", "source": "textbook"},
    )
    assert ei.reference == {"expected_answer": "A", "source": "textbook"}


def test_eval_input_with_tags():
    """EvalInput with tags."""
    ei = EvalInput(
        data=SingleTurnEvalInputData(user_message=UserMessage(text="Q")),
        tags=["math", "easy"],
    )
    assert ei.tags == ["math", "easy"]


def test_eval_input_persists_under_task(mock_task, tmp_path):
    """EvalInput saves as a child of Task and loads back."""
    task_path = tmp_path / "task.kiln"
    mock_task.path = task_path
    mock_task.save_to_file()

    ei = EvalInput(
        parent=mock_task,
        data=SingleTurnEvalInputData(user_message=UserMessage(text="Persist me")),
        reference={"key": "val"},
        tags=["t1"],
    )
    ei.save_to_file()

    loaded_task = Task.load_from_file(str(task_path))
    inputs = loaded_task.eval_inputs(readonly=True)
    assert len(inputs) == 1
    assert inputs[0].data.type == "single_turn"
    assert inputs[0].data.user_message.text == "Persist me"
    assert inputs[0].reference == {"key": "val"}
    assert inputs[0].tags == ["t1"]


# ── EvalTaskInput Tests ──────────────────────────────────────────────────


class TestEvalTaskInput:
    def test_minimal(self):
        """Only final_message is required."""
        eti = EvalTaskInput(final_message="Hello world")
        assert eti.final_message == "Hello world"
        assert eti.trace is None
        assert eti.reference_data is None
        assert eti.task_input is None

    def test_all_fields(self):
        """All fields populated."""
        trace = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
        ref = {"expected": "42", "source": "textbook"}
        eti = EvalTaskInput(
            final_message="hey",
            trace=trace,
            reference_data=ref,
            task_input="hi",
        )
        assert eti.final_message == "hey"
        assert eti.trace == trace
        assert eti.reference_data == ref
        assert eti.task_input == "hi"

    def test_round_trip(self):
        """model_dump / model_validate round-trip preserves all data."""
        eti = EvalTaskInput(
            final_message="answer",
            trace=[{"role": "user", "content": "q"}],
            reference_data={"k": 1},
            task_input="q",
        )
        data = eti.model_dump()
        rebuilt = EvalTaskInput.model_validate(data)
        assert rebuilt == eti

    def test_missing_final_message_raises(self):
        """final_message is required; omitting it raises ValidationError."""
        with pytest.raises(ValidationError, match="final_message"):
            EvalTaskInput()  # type: ignore[call-arg]


class TestEvalTaskInputFromEvalInput:
    def _run_output(self, mock_task):
        from kiln_ai.datamodel.task_output import (
            DataSource,
            DataSourceType,
            TaskOutput,
        )
        from kiln_ai.datamodel.task_run import TaskRun

        source = DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": "m",
                "model_provider": "p",
                "adapter_name": "a",
            },
        )
        return TaskRun(
            input="ignored",
            input_source=source,
            output=TaskOutput(output="final reply", source=source),
            trace=[
                {"role": "user", "content": "opening message"},
                {"role": "assistant", "content": "final reply"},
            ],
            parent=mock_task,
        )

    def test_single_turn(self, mock_task):
        ei = EvalInput(
            data=SingleTurnEvalInputData(user_message=UserMessage(text="Q")),
            reference={"expected": "A"},
        )
        eti = EvalTaskInput.from_eval_input(ei, self._run_output(mock_task))
        assert eti.task_input == "Q"
        assert eti.final_message == "final reply"
        assert eti.reference_data == {"expected": "A"}

    def test_multi_turn(self, mock_task):
        """Multi-turn: the seed message is the task_input; the conversation
        rides the trace."""
        ei = EvalInput(
            data=MultiTurnSyntheticEvalInputData(
                first_message=UserMessage(text="opening message"),
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
            ),
        )
        eti = EvalTaskInput.from_eval_input(ei, self._run_output(mock_task))
        assert eti.task_input == "opening message"
        assert eti.final_message == "final reply"
        assert eti.trace is not None
        assert len(eti.trace) == 2

    def test_multi_turn_without_first_message(self, mock_task):
        ei = EvalInput(
            data=MultiTurnSyntheticEvalInputData(
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
            ),
        )
        eti = EvalTaskInput.from_eval_input(ei, self._run_output(mock_task))
        assert eti.task_input is None


# ── MultiTurnDriveConfig Tests ───────────────────────────────────────────


class TestMultiTurnDriveConfig:
    def test_valid(self):
        cfg = MultiTurnDriveConfig(
            model_name="claude_4_5_haiku", model_provider="openrouter", turns=5
        )
        assert cfg.turns == 5

    @pytest.mark.parametrize("turns", [0, -1, 21])
    def test_turns_bounds(self, turns):
        with pytest.raises(ValidationError, match="turns"):
            MultiTurnDriveConfig(
                model_name="m", model_provider="openrouter", turns=turns
            )

    def test_unknown_provider_string_accepted(self):
        """model_provider is a plain string, not the enum — persisted items
        must load on builds that don't know the provider yet."""
        cfg = MultiTurnDriveConfig(
            model_name="m", model_provider="a_future_provider", turns=1
        )
        assert cfg.model_provider == "a_future_provider"

    def test_defaults_to_none_on_item(self):
        """Items minted before drive settings were stamped load with None."""
        data = MultiTurnSyntheticEvalInputData(
            synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
        )
        assert data.drive_config is None

    def test_persists_on_eval_input(self, mock_task, tmp_path):
        """The stamped drive config round-trips through disk on the item."""
        task_path = tmp_path / "task.kiln"
        mock_task.path = task_path
        mock_task.save_to_file()

        ei = EvalInput(
            parent=mock_task,
            data=MultiTurnSyntheticEvalInputData(
                first_message=UserMessage(text="opening message"),
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
                drive_config=MultiTurnDriveConfig(
                    model_name="claude_4_5_haiku",
                    model_provider="openrouter",
                    turns=5,
                ),
            ),
        )
        ei.save_to_file()

        loaded_task = Task.load_from_file(str(task_path))
        data = loaded_task.eval_inputs(readonly=True)[0].data
        assert isinstance(data, MultiTurnSyntheticEvalInputData)
        assert data.drive_config is not None
        assert data.drive_config.model_name == "claude_4_5_haiku"
        assert data.drive_config.model_provider == "openrouter"
        assert data.drive_config.turns == 5

    def test_eval_has_no_drive_config_field(self):
        """The item is the only home for drive settings. An eval-level copy
        would reintroduce a second read path that can drift from the items."""
        assert "multi_turn_drive_config" not in Eval.model_fields

    def test_unset_drive_config_is_omitted_from_saved_bytes(self, mock_task, tmp_path):
        """Items that predate the field must not gain a null key on resave."""
        task_path = tmp_path / "task.kiln"
        mock_task.path = task_path
        mock_task.save_to_file()

        ei = EvalInput(
            parent=mock_task,
            data=MultiTurnSyntheticEvalInputData(
                first_message=UserMessage(text="hi"),
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
            ),
        )
        ei.save_to_file()
        assert ei.path is not None
        on_disk = json.loads(ei.path.read_text())
        assert "drive_config" not in on_disk["data"]

    def test_stamped_drive_config_saved_bytes(self, mock_task, tmp_path):
        """A stamped config is written as a nested object under data."""
        task_path = tmp_path / "task.kiln"
        mock_task.path = task_path
        mock_task.save_to_file()

        ei = EvalInput(
            parent=mock_task,
            data=MultiTurnSyntheticEvalInputData(
                first_message=UserMessage(text="hi"),
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
                drive_config=MultiTurnDriveConfig(
                    model_name="m", model_provider="openrouter", turns=3
                ),
            ),
        )
        ei.save_to_file()
        assert ei.path is not None
        on_disk = json.loads(ei.path.read_text())
        assert on_disk["data"]["drive_config"] == {
            "model_name": "m",
            "model_provider": "openrouter",
            "turns": 3,
        }

    def test_explicit_null_drive_config_loads_as_absent(self):
        """A hand-written null means the same as no key at all."""
        data = MultiTurnSyntheticEvalInputData.model_validate(
            {
                "type": "multi_turn_synthetic",
                "first_message": {"text": "hi"},
                "synthetic_user_info": {"persona": "p", "goal": "g"},
                "drive_config": None,
            }
        )
        assert data.drive_config is None
        assert "drive_config" not in data.model_dump()


class TestEvalTaskInputFromTrace:
    """The trace and the item it was generated from are two records now."""

    @pytest.fixture
    def trace(self):
        return TaskRun(
            input="what the model saw",
            output=TaskOutput(output="what the model said"),
            trace=[{"role": "assistant", "content": "what the model said"}],
        )

    def test_from_an_eval_input_source(self, trace):
        eval_input = EvalInput(
            data=SingleTurnEvalInputData(user_message=UserMessage(text="2+2?")),
            reference={"answer": "4"},
        )

        result = EvalTaskInput.from_trace(trace, eval_input)

        assert result.final_message == "what the model said"
        assert result.trace == trace.trace
        assert result.reference_data == {"answer": "4"}
        # The item's own text, not the trace's: the item is the canonical input.
        assert result.task_input == "2+2?"

    def test_from_a_task_run_source(self, trace):
        item = TaskRun(
            input="the dataset input", output=TaskOutput(output="the curated answer")
        )

        result = EvalTaskInput.from_trace(trace, item)

        assert result.final_message == "what the model said"
        # A TaskRun-backed dataset item stores the curated answer as its output, so it
        # is the ground truth the judge compares the trace against.
        assert result.reference_data == {"reference_answer": "the curated answer"}
        # No separate statement of the input exists for a TaskRun-backed item, so the
        # trace's own input is what was actually scored.
        assert result.task_input == "what the model saw"

    def test_a_task_run_scored_as_itself_has_no_reference(self, trace):
        """Calibration scores the golden item as itself: a reference populated there
        would be byte-identical to final_message, and every judge would pass."""
        result = EvalTaskInput.from_trace(trace, trace)

        assert result.final_message == "what the model said"
        assert result.reference_data is None

    def test_from_task_run_has_no_reference(self, trace):
        """`from_task_run` is `from_trace(run, run)`, so one identity check covers it."""
        assert EvalTaskInput.from_task_run(trace).reference_data is None

    def test_an_equal_but_distinct_source_still_populates(self, trace):
        """The carve-out is identity, not equality: a separate item that happens to
        match the trace is still a real dataset item with a real reference answer."""
        twin = trace.model_copy(deep=True)

        assert EvalTaskInput.from_trace(trace, twin).reference_data == {
            "reference_answer": "what the model said"
        }

    def test_existing_constructors_are_from_trace(self, trace):
        """The two named constructors are the two shapes of `from_trace`."""
        eval_input = EvalInput(
            data=SingleTurnEvalInputData(user_message=UserMessage(text="2+2?")),
            reference={"answer": "4"},
        )
        assert EvalTaskInput.from_eval_input(
            eval_input, trace
        ) == EvalTaskInput.from_trace(trace, eval_input)
        assert EvalTaskInput.from_task_run(trace) == EvalTaskInput.from_trace(
            trace, trace
        )

    def test_from_a_multi_turn_eval_input_source(self, trace):
        """The item's first message is the canonical input; the conversation
        itself (and its final answer) comes from the trace TaskRun."""
        eval_input = EvalInput(
            data=MultiTurnSyntheticEvalInputData(
                first_message=UserMessage(text="opening message"),
                synthetic_user_info=SyntheticUserInfo(
                    persona="p", goal="g", behavior_guidance="b"
                ),
            ),
            reference={"expected": "resolution"},
        )

        result = EvalTaskInput.from_trace(trace, eval_input)

        assert result.final_message == "what the model said"
        assert result.trace == trace.trace
        assert result.reference_data == {"expected": "resolution"}
        assert result.task_input == "opening message"

    def test_from_a_multi_turn_eval_input_without_a_first_message(self, trace):
        """Items minted without a seed have no canonical input text; the judge
        still gets the trace and final answer."""
        eval_input = EvalInput(
            data=MultiTurnSyntheticEvalInputData(
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
            ),
        )

        result = EvalTaskInput.from_trace(trace, eval_input)

        assert result.task_input is None
        assert result.final_message == "what the model said"
        assert result.trace == trace.trace

    @pytest.mark.parametrize(
        "trace_arg, source, error",
        [
            ("not a run", TaskRun(input="i", output=TaskOutput(output="o")), TypeError),
            (
                TaskRun(input="i", output=TaskOutput(output="o")),
                "not an item",
                TypeError,
            ),
        ],
    )
    def test_rejects_shapes_it_cannot_describe(self, trace_arg, source, error):
        with pytest.raises(error):
            EvalTaskInput.from_trace(trace_arg, source)


# ── Save-time Jinja validation (validate_v2_templates_and_expressions) ───


def _make_v2_eval_config(**kwargs) -> EvalConfig:
    """Helper to build a V2 EvalConfig with minimal ceremony."""
    return EvalConfig(name="V2 Test", config_type=EvalConfigType.v2, **kwargs)


class TestV2TemplateValidation:
    def test_valid_prompt_template(self):
        """A prompt_template with a Jinja expression passes validation."""
        cfg = _make_v2_eval_config(
            properties=LlmJudgeProperties(
                model_name="m",
                model_provider="p",
                prompt_template="Evaluate: {{ final_message }}",
            ),
        )
        assert cfg.properties.prompt_template == "Evaluate: {{ final_message }}"

    def test_invalid_prompt_template_syntax(self):
        """Broken Jinja syntax in prompt_template is rejected."""
        with pytest.raises(ValidationError, match="Invalid Jinja2 template"):
            _make_v2_eval_config(
                properties=LlmJudgeProperties(
                    model_name="m",
                    model_provider="p",
                    prompt_template="Hello {{ broken",
                ),
            )

    def test_static_prompt_template_rejected(self):
        """A prompt_template with no Jinja expressions is rejected."""
        with pytest.raises(ValidationError, match="never references the model output"):
            _make_v2_eval_config(
                properties=LlmJudgeProperties(
                    model_name="m",
                    model_provider="p",
                    prompt_template="Just plain text, nothing dynamic.",
                ),
            )

    def test_comment_only_prompt_template_rejected(self):
        """A prompt_template with only Jinja comments is effectively static and rejected."""
        with pytest.raises(ValidationError, match="never references the model output"):
            _make_v2_eval_config(
                properties=LlmJudgeProperties(
                    model_name="m",
                    model_provider="p",
                    prompt_template="{# This is just a comment #} plain text",
                ),
            )

    def test_reference_data_only_prompt_template_rejected(self):
        """A prompt_template referencing only reference_data is rejected: it never varies with the model output."""
        with pytest.raises(ValidationError, match="never references the model output"):
            _make_v2_eval_config(
                properties=LlmJudgeProperties(
                    model_name="m",
                    model_provider="p",
                    prompt_template="{{ reference_data.expected_output }}",
                ),
            )

    def test_prompt_template_with_block_passes(self):
        """A prompt_template using {%% blocks referencing model output is not static."""
        cfg = _make_v2_eval_config(
            properties=LlmJudgeProperties(
                model_name="m",
                model_provider="p",
                prompt_template="{% if trace %}has trace{% endif %}",
            ),
        )
        assert cfg is not None

    def test_prompt_template_with_trace_passes(self):
        """A prompt_template referencing trace passes: trace counts as model output."""
        cfg = _make_v2_eval_config(
            properties=LlmJudgeProperties(
                model_name="m",
                model_provider="p",
                prompt_template="{{ trace[0].content }}",
            ),
        )
        assert cfg is not None

    def test_prompt_template_with_task_input_passes(self):
        """A prompt_template referencing task_input passes: it varies per run."""
        cfg = _make_v2_eval_config(
            properties=LlmJudgeProperties(
                model_name="m",
                model_provider="p",
                prompt_template="{{ task_input }}",
            ),
        )
        assert cfg is not None

    def test_valid_value_expression(self):
        """A valid value_expression compiles without error."""
        cfg = _make_v2_eval_config(
            properties=ExactMatchProperties(
                expected_value="yes",
                value_expression="final_message.strip()",
            ),
        )
        assert cfg.properties.value_expression == "final_message.strip()"

    def test_invalid_value_expression(self):
        """Bad Jinja syntax in value_expression is rejected."""
        with pytest.raises(ValidationError, match="Invalid Jinja2 expression"):
            _make_v2_eval_config(
                properties=ExactMatchProperties(
                    expected_value="yes",
                    value_expression="final_message[",
                ),
            )

    def test_none_value_expression_skipped(self):
        """value_expression=None (default) should not be validated."""
        cfg = _make_v2_eval_config(
            properties=ExactMatchProperties(expected_value="yes"),
        )
        assert isinstance(cfg.properties, ExactMatchProperties)
        assert cfg.properties.value_expression is None

    def test_reference_keys_stored(self):
        """reference_keys are stored on LlmJudgeProperties."""
        cfg = _make_v2_eval_config(
            properties=LlmJudgeProperties(
                model_name="m",
                model_provider="p",
                prompt_template="{{ final_message }}",
                reference_keys=["expected_answer", "context"],
            ),
        )
        assert isinstance(cfg.properties, LlmJudgeProperties)
        assert cfg.properties.reference_keys == ["expected_answer", "context"]

    def test_reference_keys_default_empty(self):
        """reference_keys defaults to empty list."""
        cfg = _make_v2_eval_config(
            properties=LlmJudgeProperties(
                model_name="m",
                model_provider="p",
                prompt_template="{{ final_message }}",
            ),
        )
        assert isinstance(cfg.properties, LlmJudgeProperties)
        assert cfg.properties.reference_keys == []

    def test_legacy_config_skips_jinja_validation(self):
        """Legacy (g_eval) configs bypass Jinja validation entirely."""
        cfg = EvalConfig(
            name="Legacy",
            config_type=EvalConfigType.g_eval,
            properties={"eval_steps": ["step1"]},
            model_name="gpt-4",
            model_provider="openai",
        )
        assert cfg.config_type == EvalConfigType.g_eval

    @pytest.mark.parametrize(
        "props",
        [
            PatternMatchProperties(pattern="ok", value_expression="final_message"),
            ContainsProperties(substring="yes", value_expression="final_message"),
            SetCheckProperties(
                expected_set=["a"], value_expression="final_message", mode="equal"
            ),
        ],
        ids=["pattern_match", "contains", "set_check"],
    )
    def test_value_expression_across_property_types(self, props):
        """value_expression validation works for all property types that support it."""
        cfg = _make_v2_eval_config(properties=props)
        assert hasattr(cfg.properties, "value_expression")
        assert cfg.properties.value_expression == "final_message"  # type: ignore[union-attr]


class TestCodeEvalPropertiesValidation:
    VALID_CODE = "def score(output, trace, reference_data, task_input):\n    return {'accuracy': 1.0}\n"

    def test_valid_code(self):
        props = CodeEvalProperties(code=self.VALID_CODE)
        assert props.code == self.VALID_CODE
        # 180, not 30: the default has to cover nested LLM calls from score().
        assert props.timeout_seconds == 180

    def test_custom_timeout(self):
        props = CodeEvalProperties(code=self.VALID_CODE, timeout_seconds=120)
        assert props.timeout_seconds == 120

    def test_timeout_min_boundary(self):
        with pytest.raises(ValidationError):
            CodeEvalProperties(code=self.VALID_CODE, timeout_seconds=0)

    def test_timeout_max_boundary(self):
        with pytest.raises(ValidationError):
            CodeEvalProperties(code=self.VALID_CODE, timeout_seconds=301)

    def test_syntax_error_rejected(self):
        with pytest.raises(ValidationError, match="syntax error"):
            CodeEvalProperties(code="def score(:\n")

    def test_missing_score_function_rejected(self):
        with pytest.raises(ValidationError, match="module-level 'score' function"):
            CodeEvalProperties(code="def not_score(output):\n    return {}\n")

    def test_code_too_large(self):
        big_code = (
            "def score(output, trace, reference_data, task_input):\n    return {'x': 1.0}\n"
            + ("# padding\n" * 10000)
        )
        if len(big_code.encode("utf-8")) <= 64 * 1024:
            big_code = big_code + " " * (64 * 1024 + 1)
        with pytest.raises(ValidationError, match="too large"):
            CodeEvalProperties(code=big_code)

    def test_nested_score_function_rejected(self):
        code = "def wrapper():\n    def score(output, trace, reference_data, task_input):\n        return {}\n"
        with pytest.raises(ValidationError, match="module-level 'score' function"):
            CodeEvalProperties(code=code)

    def test_async_score_function_accepted(self):
        code = "async def score(output, trace, reference_data, task_input):\n    return {'accuracy': 1.0}\n"
        props = CodeEvalProperties(code=code)
        assert props.code == code

    def test_default_tool_allowlist_is_empty(self):
        props = CodeEvalProperties(code=self.VALID_CODE)
        assert props.tool_allowlist == []

    def test_valid_tool_allowlist(self):
        props = CodeEvalProperties(
            code=self.VALID_CODE,
            tool_allowlist=[
                "kiln_tool::llm",
                "kiln_tool::llm_judge",
                "mcp::remote::server1::tool1",
            ],
        )
        assert len(props.tool_allowlist) == 3

    def test_tool_allowlist_rejects_skill_ids(self):
        with pytest.raises(ValidationError, match="Skill tool IDs cannot"):
            CodeEvalProperties(
                code=self.VALID_CODE,
                tool_allowlist=["kiln_tool::skill::some_skill"],
            )

    def test_tool_allowlist_rejects_unmanaged_ids(self):
        with pytest.raises(ValidationError, match="Unmanaged tool IDs cannot"):
            CodeEvalProperties(
                code=self.VALID_CODE,
                tool_allowlist=["kiln_unmanaged::some_tool"],
            )

    def test_tool_allowlist_rejects_duplicates(self):
        with pytest.raises(ValidationError, match="Duplicate tool ID"):
            CodeEvalProperties(
                code=self.VALID_CODE,
                tool_allowlist=["kiln_tool::llm", "kiln_tool::llm"],
            )

    def test_tool_allowlist_rejects_invalid_tool_id(self):
        with pytest.raises(ValidationError, match="Invalid tool ID"):
            CodeEvalProperties(
                code=self.VALID_CODE,
                tool_allowlist=["not_a_valid_tool_id"],
            )

    def test_tool_allowlist_allows_self_referential_code_tool_id(self):
        # A code eval is not itself a tool, so the CodeTool self-reference check
        # is intentionally omitted — any valid code tool ID is allowed.
        props = CodeEvalProperties(
            code=self.VALID_CODE,
            tool_allowlist=["kiln_tool::code::123456789012"],
        )
        assert props.tool_allowlist == ["kiln_tool::code::123456789012"]


# ── V1 Coexistence Regression Guards ─────────────────────────────────


class TestV1EvalRunCoexistence:
    def test_v1_eval_run_new_optional_fields_default_to_none(self):
        run = EvalRun(
            dataset_id="ds1",
            task_run_config_id="rc1",
            input="What is 2+2?",
            output="4",
            scores={"accuracy": 1.0},
        )
        assert run.eval_input_id is None
        assert run.skipped_reason is None
        assert run.skipped_detail is None

    def test_v1_eval_run_round_trip_preserves_none_defaults(self, mock_task, tmp_path):
        task_path = tmp_path / "task.kiln"
        mock_task.path = task_path
        mock_task.save_to_file()

        eval_obj = Eval(
            name="V1 Compat Eval",
            parent=mock_task,
            eval_set_filter_id="tag::v1set",
            eval_configs_filter_id="tag::golden",
            output_scores=[
                EvalOutputScore(name="acc", type=TaskOutputRatingType.pass_fail)
            ],
        )
        eval_obj.save_to_file()

        config = EvalConfig(
            name="V1 Config",
            parent=eval_obj,
            config_type=EvalConfigType.g_eval,
            model_name="gpt-4",
            model_provider="openai",
            properties={"eval_steps": ["check"]},
        )
        config.save_to_file()

        run = EvalRun(
            parent=config,
            dataset_id="ds1",
            task_run_config_id="rc1",
            input="hello",
            output="world",
            scores={"acc": 0.8},
        )
        run.save_to_file()

        loaded = EvalRun.load_from_file(str(run.path))
        assert loaded.dataset_id == "ds1"
        assert loaded.eval_input_id is None
        assert loaded.skipped_reason is None
        assert loaded.skipped_detail is None
        assert loaded.scores == {"acc": 0.8}

    def test_pre_split_eval_run_file_loads_unchanged(self, mock_task, tmp_path):
        """Regression guard for D15: a record written before scored_run_id existed still
        loads, with every inline field intact and no new field required. Read through
        config.runs(), the real path, not just load_from_file."""
        mock_task.path = tmp_path / "task.kiln"
        mock_task.save_to_file()

        eval_obj = Eval(
            name="Pre Split Eval",
            parent=mock_task,
            eval_set_filter_id="tag::tag1",
            eval_configs_filter_id="tag::tag2",
            output_scores=[
                EvalOutputScore(name="accuracy", type=TaskOutputRatingType.pass_fail)
            ],
        )
        eval_obj.save_to_file()
        config = EvalConfig(
            name="Pre Split Config",
            parent=eval_obj,
            config_type=EvalConfigType.g_eval,
            model_name="gpt-4",
            model_provider="openai",
            properties={"eval_steps": ["step1"]},
        )
        config.save_to_file()

        runs_dir = config.path.parent / "runs" / "legacy_run"
        runs_dir.mkdir(parents=True)
        on_disk = {
            "v": 1,
            "id": "123456789012",
            "model_type": "eval_run",
            "dataset_id": "ds1",
            "task_run_config_id": "rc1",
            "eval_config_eval": False,
            "input": "legacy input",
            "output": "legacy output",
            "reference_answer": None,
            "intermediate_outputs": {"chain_of_thought": "thinking"},
            "task_run_trace": None,
            "scores": {"accuracy": 1.0},
            "task_run_usage": {
                "input_tokens": 5,
                "output_tokens": 2,
                "total_tokens": 7,
            },
        }
        (runs_dir / "eval_run.kiln").write_text(json.dumps(on_disk))

        loaded_runs = config.runs()
        assert len(loaded_runs) == 1
        loaded = loaded_runs[0]
        assert loaded.scored_run_id is None
        assert loaded.eval_usage is None
        assert loaded.input == "legacy input"
        assert loaded.output == "legacy output"
        assert loaded.task_run_usage is not None
        assert loaded.task_run_usage.total_tokens == 7
        assert loaded.scores == {"accuracy": 1.0}

    def test_retired_reference_data_key_loads_and_is_dropped(self):
        """`reference_data` was declared on EvalRun on an unreleased branch and never
        shipped, so it was deleted outright rather than deprecated. Dev-build files
        that carry it must still load: EvalRun sets no `extra=` override, so pydantic's
        default `extra="ignore"` drops the key rather than rejecting the record."""
        run = EvalRun.model_validate(
            {
                "dataset_id": "ds1",
                "task_run_config_id": "rc1",
                "input": "What is 2+2?",
                "output": "4",
                "scores": {"accuracy": 1.0},
                "reference_data": {"expected": "4"},
            }
        )

        assert not hasattr(run, "reference_data")
        assert "reference_data" not in run.model_dump()
        assert run.scores == {"accuracy": 1.0}


class TestV1EvalConfigCoexistence:
    def test_v1_config_with_default_config_type(self):
        config = EvalConfig(
            name="Legacy Default",
            model_name="gpt-4",
            model_provider="openai",
            properties={"eval_steps": ["step1"]},
        )
        assert config.config_type == EvalConfigType.g_eval
        assert isinstance(config.properties, dict)

    def test_v1_config_from_dict_without_config_type_key(self):
        raw = {
            "name": "From Disk V1",
            "model_name": "gpt-4",
            "model_provider": "openai",
            "properties": {"eval_steps": ["step1", "step2"]},
        }
        config = EvalConfig.model_validate(raw)
        assert config.config_type == EvalConfigType.g_eval
        assert isinstance(config.properties, dict)
        assert config.properties["eval_steps"] == ["step1", "step2"]

    def test_v1_properties_with_type_key_not_misrouted(self):
        raw = {
            "name": "Type Key Collision",
            "config_type": "g_eval",
            "model_name": "gpt-4",
            "model_provider": "openai",
            "properties": {
                "eval_steps": ["step1"],
                "type": "exact_match",
            },
        }
        config = EvalConfig.model_validate(raw)
        assert config.config_type == EvalConfigType.g_eval
        assert isinstance(config.properties, dict)
        assert config.properties["type"] == "exact_match"
        assert config.properties["eval_steps"] == ["step1"]

    def test_v1_properties_fully_colliding_with_v2_shape_stay_dict(self):
        """A legacy properties dict that would parse cleanly as a typed V2 class must still load as a plain dict."""
        props = {
            "eval_steps": ["step1"],
            "type": "llm_judge",
            "model_name": "m",
            "model_provider": "p",
            "prompt_template": "{{ final_message }}",
        }
        # Premise: this dict is a valid LlmJudgeProperties payload, so only
        # explicit dispatch (not union fallback) keeps it untyped below.
        assert isinstance(LlmJudgeProperties.model_validate(props), LlmJudgeProperties)

        config = EvalConfig.model_validate(
            {
                "name": "Full Collision",
                "config_type": "g_eval",
                "model_name": "gpt-4",
                "model_provider": "openai",
                "properties": props,
            }
        )
        assert config.config_type == EvalConfigType.g_eval
        assert type(config.properties) is dict
        assert config.properties["type"] == "llm_judge"
        assert config.properties["eval_steps"] == ["step1"]

    def test_v2_config_from_dict_round_trips_typed(self):
        """V2 properties load from a raw dict into the typed class and survive dump/validate."""
        raw = {
            "name": "From Disk V2",
            "config_type": "v2",
            "properties": {
                "type": "llm_judge",
                "model_name": "m",
                "model_provider": "p",
                "prompt_template": "{{ final_message }}",
            },
        }
        config = EvalConfig.model_validate(raw)
        assert isinstance(config.properties, LlmJudgeProperties)

        reloaded = EvalConfig.model_validate(config.model_dump())
        assert isinstance(reloaded.properties, LlmJudgeProperties)
        assert reloaded.properties == config.properties

    def test_v1_llm_as_judge_config_type_preserved(self):
        config = EvalConfig(
            name="LLM Judge V1",
            config_type=EvalConfigType.llm_as_judge,
            model_name="gpt-4o",
            model_provider="openai",
            properties={"eval_steps": ["judge it"]},
        )
        assert config.config_type == EvalConfigType.llm_as_judge
        assert isinstance(config.properties, dict)

    def test_v1_config_round_trip_with_type_key_in_properties(
        self, mock_task, tmp_path
    ):
        task_path = tmp_path / "task.kiln"
        mock_task.path = task_path
        mock_task.save_to_file()

        eval_obj = Eval(
            name="Type Key Eval",
            parent=mock_task,
            eval_set_filter_id="tag::s",
            eval_configs_filter_id="tag::g",
            output_scores=[
                EvalOutputScore(name="s", type=TaskOutputRatingType.pass_fail)
            ],
        )
        eval_obj.save_to_file()

        config = EvalConfig(
            name="Type Key Config",
            parent=eval_obj,
            config_type=EvalConfigType.g_eval,
            model_name="gpt-4",
            model_provider="openai",
            properties={
                "eval_steps": ["s1"],
                "type": "some_value",
            },
        )
        config.save_to_file()

        loaded = EvalConfig.load_from_file(str(config.path))
        assert loaded.config_type == EvalConfigType.g_eval
        assert isinstance(loaded.properties, dict)
        assert loaded.properties["type"] == "some_value"
        assert loaded.properties["eval_steps"] == ["s1"]


# ---------------------------------------------------------------------------
# Legacy EvalRun output: may be None only when skipped_reason is set
# ---------------------------------------------------------------------------


class TestV1EvalRunOutputNoneGuard:
    """V1 EvalRun with output=None should raise unless skipped."""

    def test_v1_eval_run_output_none_raises(self, mock_task, valid_eval_config_data):
        eval_obj = Eval(
            name="Guard Test",
            parent=mock_task,
            eval_set_filter_id="tag::s",
            eval_configs_filter_id="tag::g",
            output_scores=[
                EvalOutputScore(name="score", type=TaskOutputRatingType.pass_fail)
            ],
        )
        config = EvalConfig(parent=eval_obj, **valid_eval_config_data)

        with pytest.raises(ValueError, match="V1 EvalRun requires output to be set"):
            EvalRun(
                parent=config,
                dataset_id="d1",
                task_run_config_id="c1",
                input="test",
                output=None,
                scores={"score": 1.0},
            )

    def test_v1_eval_run_output_none_skipped_allowed(
        self, mock_task, valid_eval_config_data
    ):
        eval_obj = Eval(
            name="Guard Skipped Test",
            parent=mock_task,
            eval_set_filter_id="tag::s",
            eval_configs_filter_id="tag::g",
            output_scores=[
                EvalOutputScore(name="score", type=TaskOutputRatingType.pass_fail)
            ],
        )
        config = EvalConfig(parent=eval_obj, **valid_eval_config_data)

        run = EvalRun(
            parent=config,
            dataset_id="d1",
            task_run_config_id="c1",
            input="test",
            output=None,
            scores={"score": 1.0},
            skipped_reason="missing_reference_key",
        )
        assert run.output is None
        assert run.skipped_reason == "missing_reference_key"

    def test_v1_eval_run_output_set_passes(self, mock_task, valid_eval_config_data):
        eval_obj = Eval(
            name="Guard Pass Test",
            parent=mock_task,
            eval_set_filter_id="tag::s",
            eval_configs_filter_id="tag::g",
            output_scores=[
                EvalOutputScore(name="score", type=TaskOutputRatingType.pass_fail)
            ],
        )
        config = EvalConfig(parent=eval_obj, **valid_eval_config_data)

        run = EvalRun(
            parent=config,
            dataset_id="d1",
            task_run_config_id="c1",
            input="test",
            output="some output",
            scores={"score": 1.0},
        )
        assert run.output == "some output"

    def test_v2_eval_run_output_none_allowed(self, mock_task):
        eval_obj = Eval(
            name="V2 Guard Test",
            parent=mock_task,
            splits={"test": EvalInputSplit(filter_id="tag::s")},
            eval_configs_filter_id="tag::g",
            evaluation_data_type=None,
            output_scores=[
                EvalOutputScore(name="score", type=TaskOutputRatingType.pass_fail)
            ],
        )
        config = EvalConfig(
            parent=eval_obj,
            name="V2 Config",
            config_type=EvalConfigType.v2,
            properties=ExactMatchProperties(
                expected_value="gold",
            ),
        )

        run = EvalRun(
            parent=config,
            eval_input_id="e1",
            task_run_config_id="c1",
            input="test",
            output=None,
            scores={"score": 1.0},
            skipped_reason="extraction_failed",
        )
        assert run.output is None


# ---------------------------------------------------------------------------
# CodeEvalProperties code validation: must parse and define a score function
# ---------------------------------------------------------------------------


class TestCodeEvalCodeValidation:
    """CodeEvalProperties.code must be parseable Python defining a module-level score function."""

    def test_valid_code_with_score_fn(self):
        props = CodeEvalProperties(
            code="def score(output, expected):\n    return 1.0\n"
        )
        assert props.code.startswith("def score")

    def test_code_missing_score_fn_raises(self):
        with pytest.raises(
            ValueError, match="must define a module-level 'score' function"
        ):
            CodeEvalProperties(code="def helper():\n    pass\n")

    def test_syntax_error_caught_by_compile(self):
        with pytest.raises(ValueError, match="syntax error"):
            CodeEvalProperties(code="def bad(:\n")

    def test_async_score_fn_valid(self):
        props = CodeEvalProperties(
            code="async def score(output, expected):\n    return 1.0\n"
        )
        assert "async def score" in props.code


class TestValidateScoresAgainstOutputScores:
    """Tests for the shared validate_scores_against_output_scores function."""

    def test_five_star_in_range(self):
        output_scores = [
            EvalOutputScore(name="quality", type=TaskOutputRatingType.five_star)
        ]
        assert (
            validate_scores_against_output_scores({"quality": 3.0}, output_scores) == []
        )
        assert (
            validate_scores_against_output_scores({"quality": 1.0}, output_scores) == []
        )
        assert (
            validate_scores_against_output_scores({"quality": 5.0}, output_scores) == []
        )

    def test_five_star_out_of_range(self):
        output_scores = [
            EvalOutputScore(name="quality", type=TaskOutputRatingType.five_star)
        ]
        problems = validate_scores_against_output_scores(
            {"quality": 6.0}, output_scores
        )
        assert len(problems) == 1
        assert "five_star" in problems[0]
        assert "6.0" in problems[0]

        problems_low = validate_scores_against_output_scores(
            {"quality": 0.5}, output_scores
        )
        assert len(problems_low) == 1
        assert "five_star" in problems_low[0]

    def test_pass_fail_in_range(self):
        output_scores = [
            EvalOutputScore(name="check", type=TaskOutputRatingType.pass_fail)
        ]
        assert (
            validate_scores_against_output_scores({"check": 0.0}, output_scores) == []
        )
        assert (
            validate_scores_against_output_scores({"check": 1.0}, output_scores) == []
        )
        assert (
            validate_scores_against_output_scores({"check": 0.5}, output_scores) == []
        )

    def test_pass_fail_out_of_range(self):
        output_scores = [
            EvalOutputScore(name="check", type=TaskOutputRatingType.pass_fail)
        ]
        problems = validate_scores_against_output_scores({"check": 1.5}, output_scores)
        assert len(problems) == 1
        assert "pass_fail" in problems[0]

        problems_neg = validate_scores_against_output_scores(
            {"check": -0.1}, output_scores
        )
        assert len(problems_neg) == 1

    def test_pass_fail_critical_in_range(self):
        output_scores = [
            EvalOutputScore(name="safety", type=TaskOutputRatingType.pass_fail_critical)
        ]
        assert (
            validate_scores_against_output_scores({"safety": -1.0}, output_scores) == []
        )
        assert (
            validate_scores_against_output_scores({"safety": 0.0}, output_scores) == []
        )
        assert (
            validate_scores_against_output_scores({"safety": 1.0}, output_scores) == []
        )

    def test_pass_fail_critical_out_of_range(self):
        output_scores = [
            EvalOutputScore(name="safety", type=TaskOutputRatingType.pass_fail_critical)
        ]
        problems = validate_scores_against_output_scores(
            {"safety": -1.5}, output_scores
        )
        assert len(problems) == 1
        assert "pass_fail_critical" in problems[0]

        problems_high = validate_scores_against_output_scores(
            {"safety": 1.1}, output_scores
        )
        assert len(problems_high) == 1

    def test_multiple_scores_multiple_errors(self):
        output_scores = [
            EvalOutputScore(name="quality", type=TaskOutputRatingType.five_star),
            EvalOutputScore(name="check", type=TaskOutputRatingType.pass_fail),
        ]
        problems = validate_scores_against_output_scores(
            {"quality": 10.0, "check": 2.0}, output_scores
        )
        assert len(problems) == 2

    def test_missing_score_key_ignored(self):
        output_scores = [
            EvalOutputScore(name="quality", type=TaskOutputRatingType.five_star)
        ]
        assert (
            validate_scores_against_output_scores({"other": 3.0}, output_scores) == []
        )

    def test_non_float_flagged(self):
        output_scores = [
            EvalOutputScore(name="check", type=TaskOutputRatingType.pass_fail)
        ]
        problems = validate_scores_against_output_scores(
            {"check": "not_a_float"}, output_scores
        )
        assert len(problems) == 1

    @pytest.mark.parametrize(
        "score_type",
        [
            TaskOutputRatingType.five_star,
            TaskOutputRatingType.pass_fail,
            TaskOutputRatingType.pass_fail_critical,
        ],
    )
    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
    )
    def test_non_finite_flagged(self, score_type, value):
        """NaN compares False against every range bound, so it passed all range
        checks; pydantic then serialized it as null, making the saved EvalRun file
        fail Dict[str, float] validation on the next load."""
        output_scores = [EvalOutputScore(name="metric", type=score_type)]
        problems = validate_scores_against_output_scores(
            {"metric": value}, output_scores
        )
        assert len(problems) == 1

    def test_overlarge_int_flagged_not_raised(self):
        """math.isfinite raises OverflowError on ints too large for float (10**400).
        This validator is documented as never raising, so it must report a problem."""
        output_scores = [
            EvalOutputScore(name="quality", type=TaskOutputRatingType.five_star)
        ]
        problems = validate_scores_against_output_scores(
            {"quality": 10**400}, output_scores
        )
        assert len(problems) == 1

    def test_integer_scores_accepted(self):
        output_scores = [
            EvalOutputScore(name="quality", type=TaskOutputRatingType.five_star),
            EvalOutputScore(name="check", type=TaskOutputRatingType.pass_fail),
            EvalOutputScore(
                name="safety", type=TaskOutputRatingType.pass_fail_critical
            ),
        ]
        assert (
            validate_scores_against_output_scores({"quality": 3}, output_scores) == []
        )
        assert validate_scores_against_output_scores({"check": 1}, output_scores) == []
        assert (
            validate_scores_against_output_scores({"safety": -1}, output_scores) == []
        )

    def test_boolean_scores_rejected(self):
        output_scores = [
            EvalOutputScore(name="check", type=TaskOutputRatingType.pass_fail)
        ]
        problems = validate_scores_against_output_scores({"check": True}, output_scores)
        assert len(problems) == 1

    def test_empty_scores_returns_empty(self):
        output_scores = [
            EvalOutputScore(name="check", type=TaskOutputRatingType.pass_fail)
        ]
        assert validate_scores_against_output_scores({}, output_scores) == []

    def test_eval_run_validate_scores_still_raises_on_out_of_range(self):
        """Confirm EvalRun.validate_scores still raises ValueError for out-of-range scores,
        ensuring the refactor to use validate_scores_against_output_scores is behavior-preserving."""
        eval_obj = Eval(
            name="Range Check Eval",
            eval_set_filter_id="tag::test",
            eval_configs_filter_id="tag::test2",
            output_scores=[
                EvalOutputScore(name="accuracy", type=TaskOutputRatingType.five_star),
            ],
        )
        eval_config = EvalConfig(
            name="Config",
            config_type=EvalConfigType.v2,
            properties=ExactMatchProperties(expected_value="hello"),
            parent=eval_obj,
        )
        with pytest.raises(
            ValueError,
            match=r"five_star rating and must be a number between 1\.0 and 5\.0",
        ):
            EvalRun(
                eval_input_id="inp1",
                task_run_config_id="rc1",
                eval_config_eval=False,
                input="test",
                output="test",
                scores={"accuracy": 6.0},
                parent=eval_config,
            )


# ---------------------------------------------------------------------------
# V2EvalResult model tests
# ---------------------------------------------------------------------------
class TestV2EvalResult:
    def test_default_construction(self):
        result = V2EvalResult()
        assert result.scores == {}
        assert result.skipped_reason is None
        assert result.skipped_detail is None
        assert result.intermediate_outputs is None

    def test_with_scores(self):
        result = V2EvalResult(scores={"quality": 4.0})
        assert result.scores == {"quality": 4.0}
        assert result.skipped_reason is None

    def test_with_skip(self):
        result = V2EvalResult(
            skipped_reason=SkippedReason.missing_trace,
            skipped_detail="no trace",
        )
        assert result.scores == {}
        assert result.skipped_reason == SkippedReason.missing_trace
        assert result.skipped_detail == "no trace"

    def test_with_intermediate_outputs(self):
        result = V2EvalResult(
            scores={"quality": 5.0},
            intermediate_outputs={"chain_of_thought": "reasoning text"},
        )
        assert result.intermediate_outputs == {"chain_of_thought": "reasoning text"}


# ---------------------------------------------------------------------------
# ToolCallCheckProperties.expected_tools must contain at least one tool
# ---------------------------------------------------------------------------
class TestToolCallCheckExpectedToolsValidator:
    def test_empty_expected_tools_rejected(self):
        with pytest.raises(ValidationError):
            ToolCallCheckProperties(expected_tools=[])

    def test_non_empty_expected_tools_accepted(self):
        props = ToolCallCheckProperties(
            expected_tools=[ToolCallSpec(tool_name="search")]
        )
        assert len(props.expected_tools) == 1


# ---------------------------------------------------------------------------
# ArgMatch.value must compile as a regex when match_mode is "regex"
# ---------------------------------------------------------------------------
class TestArgMatchRegexValidator:
    def test_bad_regex_rejected(self):
        with pytest.raises(ValidationError, match="Invalid regex"):
            ArgMatch(value="[invalid", match_mode="regex")

    def test_valid_regex_accepted(self):
        props = ArgMatch(value=r"^[a-z]+$", match_mode="regex")
        assert props.match_mode == "regex"

    def test_exact_mode_skips_regex_check(self):
        props = ArgMatch(value="[not regex", match_mode="exact")
        assert props.match_mode == "exact"

    def test_contains_mode_skips_regex_check(self):
        props = ArgMatch(value="[not regex", match_mode="contains")
        assert props.match_mode == "contains"


# ---------------------------------------------------------------------------
# reference_key must be a non-empty string when provided
# ---------------------------------------------------------------------------
class TestReferenceKeyMinLength:
    def test_exact_match_empty_reference_key_rejected(self):
        with pytest.raises(ValidationError):
            ExactMatchProperties(reference_key="")

    def test_exact_match_none_reference_key_accepted(self):
        props = ExactMatchProperties(expected_value="val", reference_key=None)
        assert props.reference_key is None

    def test_exact_match_valid_reference_key_accepted(self):
        props = ExactMatchProperties(reference_key="answer")
        assert props.reference_key == "answer"

    def test_contains_empty_reference_key_rejected(self):
        with pytest.raises(ValidationError):
            ContainsProperties(reference_key="")

    def test_contains_none_reference_key_accepted(self):
        props = ContainsProperties(substring="test", reference_key=None)
        assert props.reference_key is None

    def test_set_check_empty_reference_key_rejected(self):
        with pytest.raises(ValidationError):
            SetCheckProperties(reference_key="", mode="subset")

    def test_set_check_none_reference_key_accepted(self):
        props = SetCheckProperties(
            expected_set=["a"], reference_key=None, mode="subset"
        )
        assert props.reference_key is None


# ---------------------------------------------------------------------------
# reference_data_keys — exhaustive per-type accessor
# ---------------------------------------------------------------------------


class TestReferenceDataKeys:
    def test_exact_match_with_reference_key(self):
        props = ExactMatchProperties(reference_key="answer")
        assert reference_data_keys(props) == ["answer"]

    def test_exact_match_without_reference_key(self):
        props = ExactMatchProperties(expected_value="foo")
        assert reference_data_keys(props) == []

    def test_contains_with_reference_key(self):
        props = ContainsProperties(reference_key="expected")
        assert reference_data_keys(props) == ["expected"]

    def test_contains_without_reference_key(self):
        props = ContainsProperties(substring="hi")
        assert reference_data_keys(props) == []

    def test_set_check_with_reference_key(self):
        props = SetCheckProperties(reference_key="items", mode="equal")
        assert reference_data_keys(props) == ["items"]

    def test_set_check_without_reference_key(self):
        props = SetCheckProperties(expected_set=["a"], mode="equal")
        assert reference_data_keys(props) == []

    def test_llm_judge_with_keys(self):
        props = LlmJudgeProperties(
            model_name="gpt-4o",
            model_provider="openai",
            prompt_template="{{ final_message }}",
            reference_keys=["expected_answer", "context"],
        )
        assert reference_data_keys(props) == ["expected_answer", "context"]

    def test_llm_judge_empty_keys(self):
        props = LlmJudgeProperties(
            model_name="gpt-4o",
            model_provider="openai",
            prompt_template="{{ final_message }}",
        )
        assert reference_data_keys(props) == []

    def test_code_eval_with_keys(self):
        code = "def score(output, trace, reference_data, task_input):\n    return {'q': 1.0}"
        props = CodeEvalProperties(code=code, reference_keys=["gold"])
        assert reference_data_keys(props) == ["gold"]

    def test_code_eval_empty_keys(self):
        code = "def score(output, trace, reference_data, task_input):\n    return {'q': 1.0}"
        props = CodeEvalProperties(code=code)
        assert reference_data_keys(props) == []

    def test_pattern_match(self):
        props = PatternMatchProperties(pattern=".*")
        assert reference_data_keys(props) == []

    def test_tool_call_check(self):
        props = ToolCallCheckProperties(
            expected_tools=[ToolCallSpec(tool_name="search")]
        )
        assert reference_data_keys(props) == []

    def test_step_count_check(self):
        props = StepCountCheckProperties(count_type="tool_calls", min_count=1)
        assert reference_data_keys(props) == []

    def test_returns_copy_not_reference(self):
        props = LlmJudgeProperties(
            model_name="gpt-4o",
            model_provider="openai",
            prompt_template="{{ final_message }}",
            reference_keys=["a"],
        )
        result = reference_data_keys(props)
        result.append("mutated")
        assert reference_data_keys(props) == ["a"]


# ---------------------------------------------------------------------------
# eval_reference_data_keys — union across configs
# ---------------------------------------------------------------------------


class TestEvalReferenceDataKeys:
    def _make_eval_with_configs(self, props_list):
        """Build an Eval with V2 configs carrying the given properties."""
        from unittest.mock import Mock

        eval_obj = Mock(spec=Eval)
        configs = []
        for props in props_list:
            cfg = Mock(spec=EvalConfig)
            cfg.config_type = EvalConfigType.v2
            cfg.properties = props
            configs.append(cfg)
        eval_obj.configs = Mock(return_value=configs)
        eval_obj.eval_reference_data_keys = Eval.eval_reference_data_keys.__get__(
            eval_obj, Eval
        )
        return eval_obj

    def test_single_config(self):
        props = ExactMatchProperties(reference_key="answer")
        eval_obj = self._make_eval_with_configs([props])
        assert eval_obj.eval_reference_data_keys() == ["answer"]

    def test_union_across_configs(self):
        p1 = ExactMatchProperties(reference_key="answer")
        p2 = ContainsProperties(reference_key="context")
        p3 = LlmJudgeProperties(
            model_name="gpt-4o",
            model_provider="openai",
            prompt_template="{{ final_message }}",
            reference_keys=["answer", "extra"],
        )
        eval_obj = self._make_eval_with_configs([p1, p2, p3])
        keys = eval_obj.eval_reference_data_keys()
        assert keys == ["answer", "context", "extra"]

    def test_empty_configs(self):
        eval_obj = self._make_eval_with_configs([])
        assert eval_obj.eval_reference_data_keys() == []

    def test_no_reference_data_configs(self):
        props = PatternMatchProperties(pattern=".*")
        eval_obj = self._make_eval_with_configs([props])
        assert eval_obj.eval_reference_data_keys() == []

    def test_dedup_preserves_insertion_order(self):
        p1 = LlmJudgeProperties(
            model_name="gpt-4o",
            model_provider="openai",
            prompt_template="{{ final_message }}",
            reference_keys=["b", "a"],
        )
        p2 = LlmJudgeProperties(
            model_name="gpt-4o",
            model_provider="openai",
            prompt_template="{{ final_message }}",
            reference_keys=["a", "c"],
        )
        eval_obj = self._make_eval_with_configs([p1, p2])
        assert eval_obj.eval_reference_data_keys() == ["b", "a", "c"]


class TestEvalPriorityStatusResolution:
    """Priority/status live on the eval, falling back to the associated spec
    for evals created before that (legacy files), then to defaults."""

    def _make_eval(self, **kwargs) -> Eval:
        return Eval(
            name="Resolution Eval",
            eval_set_filter_id="tag::tag1",
            eval_configs_filter_id="tag::tag2",
            output_scores=[
                EvalOutputScore(name="score", type=TaskOutputRatingType.pass_fail)
            ],
            **kwargs,
        )

    def test_fields_default_to_none_and_resolve_to_defaults(self):
        eval = self._make_eval()
        assert eval.priority is None
        assert eval.status is None
        assert eval.resolved_priority() == Priority.p1
        assert eval.resolved_status() == EvalStatus.active

    def test_own_values_win(self, mock_task, tmp_path):
        mock_task.path = tmp_path / "task.kiln"
        mock_task.save_to_file()

        eval = self._make_eval(
            parent=mock_task, priority=Priority.p0, status=EvalStatus.deprecated
        )
        eval.save_to_file()
        spec = Spec(
            name="Backing Spec",
            definition="definition",
            properties=DesiredBehaviourProperties(
                spec_type=SpecType.desired_behaviour,
                desired_behaviour_description="be nice",
            ),
            priority=Priority.p3,
            status=EvalStatus.archived,
            eval_id=eval.id,
            parent=mock_task,
        )
        spec.save_to_file()

        assert eval.resolved_priority() == Priority.p0
        assert eval.resolved_status() == EvalStatus.deprecated

    def test_falls_back_to_spec(self, mock_task, tmp_path):
        mock_task.path = tmp_path / "task.kiln"
        mock_task.save_to_file()

        eval = self._make_eval(parent=mock_task)
        eval.save_to_file()
        spec = Spec(
            name="Backing Spec",
            definition="definition",
            properties=DesiredBehaviourProperties(
                spec_type=SpecType.desired_behaviour,
                desired_behaviour_description="be nice",
            ),
            priority=Priority.p2,
            status=EvalStatus.future,
            eval_id=eval.id,
            parent=mock_task,
        )
        spec.save_to_file()

        # Resolved via a task scan, and via an explicitly passed spec
        assert eval.resolved_priority() == Priority.p2
        assert eval.resolved_status() == EvalStatus.future
        assert eval.resolved_priority(spec) == Priority.p2
        assert eval.resolved_status(spec) == EvalStatus.future

    def test_round_trips_through_file(self, mock_task, tmp_path):
        mock_task.path = tmp_path / "task.kiln"
        mock_task.save_to_file()

        eval = self._make_eval(
            parent=mock_task, priority=Priority.p2, status=EvalStatus.future
        )
        eval.save_to_file()

        loaded = Eval.load_from_file(str(eval.path))
        assert loaded.priority == Priority.p2
        assert loaded.status == EvalStatus.future


class TestEvalSplits:
    """The splits dict, and the one-way migration of the deprecated flat filter fields."""

    @pytest.fixture
    def scores(self):
        return [EvalOutputScore(name="score", type=TaskOutputRatingType.pass_fail)]

    @pytest.fixture
    def saved_task(self, mock_task, tmp_path):
        mock_task.path = tmp_path / "task.kiln"
        mock_task.save_to_file()
        return mock_task

    def build_eval(self, scores, **kwargs) -> Eval:
        return Eval(name="Split Eval", output_scores=scores, **kwargs)

    def saved_json(self, eval: Eval) -> dict:
        eval.save_to_file()
        assert eval.path is not None
        return json.loads(Path(eval.path).read_text(encoding="utf-8"))

    def test_legacy_fields_migrate_into_splits(self, scores):
        eval = self.build_eval(
            scores,
            eval_set_filter_id="tag::test_x",
            train_set_filter_id="tag::train_x",
        )
        assert eval.splits == {
            "test": TaskRunSplit(filter_id="tag::test_x"),
            "train": TaskRunSplit(filter_id="tag::train_x"),
        }
        # Read once, then cleared: the fields are an input format, not a second home.
        data = eval.model_dump()
        assert data["eval_set_filter_id"] is None
        assert data["train_set_filter_id"] is None

    def test_fresh_eval_has_populated_splits(self, scores):
        """The migration is not gated on loading from a file."""
        eval = self.build_eval(scores, eval_set_filter_id="tag::test_x")
        assert eval._loaded_from_file is False
        assert eval.splits["test"] == TaskRunSplit(filter_id="tag::test_x")

    def test_splits_wins_over_a_legacy_field(self, saved_task, scores):
        """Both homes populated: `splits` is the answer and the legacy value is ignored.

        Once a split is in `splits` it is official, so a legacy field beside it — a
        hand-edited file, or one an older build wrote after a newer one — cannot
        overwrite it, and is dropped on the next save.
        """
        eval = self.build_eval(
            scores,
            parent=saved_task,
            eval_set_filter_id="tag::from_legacy",
            splits={"test": TaskRunSplit(filter_id="tag::from_splits")},
        )
        assert eval.splits["test"] == TaskRunSplit(filter_id="tag::from_splits")

        data = self.saved_json(eval)
        assert data["eval_set_filter_id"] is None
        assert data["splits"] == {
            "test": {"source": "task_run", "filter_id": "tag::from_splits"}
        }

    def test_a_legacy_field_does_not_rebuild_an_existing_split(
        self, saved_task, scores
    ):
        """The split object itself survives, not just its filter id.

        Splits are `extra="allow"` so a field a future build adds is not dropped. A
        migration that overwrote the existing entry would rebuild it from a bare filter-id
        string and lose everything else on it — and a legacy field is exactly the
        situation where that overwrite used to happen.
        """
        eval = Eval.model_validate(
            {
                "name": "Future Split",
                "parent": saved_task,
                "output_scores": [{"name": "score", "type": "pass_fail"}],
                "eval_set_filter_id": "tag::from_legacy",
                "train_set_filter_id": "tag::train_legacy",
                "splits": {
                    "test": {
                        "source": "task_run",
                        "filter_id": "tag::from_splits",
                        "weight": 0.5,
                    },
                    "train": {
                        "source": "eval_input",
                        "filter_id": "tag::train_inputs",
                        "weight": 0.25,
                    },
                },
            }
        )
        data = self.saved_json(eval)
        assert data["splits"]["test"] == {
            "source": "task_run",
            "filter_id": "tag::from_splits",
            "weight": 0.5,
        }
        assert data["splits"]["train"] == {
            "source": "eval_input",
            "filter_id": "tag::train_inputs",
            "weight": 0.25,
        }

        assert eval.path is not None
        reloaded = Eval.load_from_file(eval.path)
        assert getattr(reloaded.splits["test"], "weight") == 0.5
        assert getattr(reloaded.splits["train"], "weight") == 0.25

    def test_excluding_a_legacy_field_cannot_drop_a_split(self, scores):
        """With one home, no dump option can write a split nowhere at all.

        This was the worst failure the two-home serializer could have, so the property is
        kept as a test even though a single home makes it structural.
        """
        eval = self.build_eval(scores, eval_set_filter_id="tag::test_x")

        excluded = eval.model_dump(exclude={"eval_set_filter_id"})
        assert excluded["splits"] == {
            "test": {"source": "task_run", "filter_id": "tag::test_x"}
        }
        assert eval.model_dump(exclude_none=True)["splits"] == {
            "test": {"source": "task_run", "filter_id": "tag::test_x"}
        }

    def test_serialization_schema_describes_the_fields(self):
        """Eval has no model serializer, so both schema modes are field-derived.

        FastAPI generates response types from the serialization-mode schema; a model
        serializer would collapse it to a bare object and erase Eval from the generated
        web client. Pinned here because that failure would otherwise only show up as a
        schema diff in CI.
        """
        properties = Eval.model_json_schema(mode="serialization")["properties"]
        assert "splits" in properties
        assert properties["splits"]["additionalProperties"]["discriminator"]
        assert properties["name"]["type"] == "string"
        # The deprecation reaches the OpenAPI schema, and from there the generated
        # TypeScript, so a web caller reading a legacy field is flagged.
        assert properties["eval_set_filter_id"]["deprecated"] is True
        assert properties["train_set_filter_id"]["deprecated"] is True

    def test_legacy_eval_file_is_migrated_on_save(self, saved_task, tmp_path):
        """The gate: an existing-format eval file is rewritten into the new format.

        The file below is what a build predating `splits` writes. Loading it must produce
        the splits it describes, and saving it must write them to `splits` and null both
        legacy fields — a stale legacy value would let an older client evaluate against a
        dataset this one no longer uses.
        """
        eval_path = tmp_path / "existing_eval" / "eval.kiln"
        eval_path.parent.mkdir(parents=True)
        original = {
            "v": 1,
            "id": "123456789012",
            "created_at": "2025-01-01T00:00:00Z",
            "created_by": "someone",
            "name": "Existing Eval",
            "description": None,
            "template": None,
            "current_config_id": None,
            "eval_set_filter_id": "tag::eval_set_existing",
            "eval_configs_filter_id": "tag::golden_existing",
            "train_set_filter_id": "tag::train_existing",
            "output_scores": [
                {"name": "score", "instruction": None, "type": "pass_fail"}
            ],
            "favourite": False,
            "template_properties": None,
            "evaluation_data_type": "final_answer",
            "model_type": "eval",
        }
        eval_path.write_text(
            json.dumps(original, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        loaded = Eval.load_from_file(eval_path)
        assert loaded.splits == {
            "test": TaskRunSplit(filter_id="tag::eval_set_existing"),
            "train": TaskRunSplit(filter_id="tag::train_existing"),
        }

        loaded.save_to_file()
        saved = json.loads(eval_path.read_text(encoding="utf-8"))
        assert saved["eval_set_filter_id"] is None
        assert saved["train_set_filter_id"] is None
        assert saved["splits"] == {
            "test": {"source": "task_run", "filter_id": "tag::eval_set_existing"},
            "train": {"source": "task_run", "filter_id": "tag::train_existing"},
        }
        # Nothing else about the file changed, and the golden set is not a split.
        assert saved["eval_configs_filter_id"] == "tag::golden_existing"
        for key in ("v", "id", "created_at", "created_by", "name", "favourite"):
            assert saved[key] == original[key]

        # Migrated once: the second save is a no-op, and a reload agrees.
        Eval.load_from_file(eval_path).save_to_file()
        assert json.loads(eval_path.read_text(encoding="utf-8")) == saved

    def test_no_deprecation_warning_on_a_load_save_cycle(self, saved_task, tmp_path):
        """The migration reads the deprecated fields through `__dict__` for this reason.

        `deprecated=True` warns on attribute access. The warning is for callers; the one
        place that is supposed to touch these fields must not trip it, or every load of
        every legacy eval emits one.
        """
        eval_path = tmp_path / "warn_eval" / "eval.kiln"
        eval_path.parent.mkdir(parents=True)
        eval_path.write_text(
            json.dumps(
                {
                    "v": 1,
                    "id": "223456789012",
                    "name": "Warning Eval",
                    "eval_set_filter_id": "tag::test_x",
                    "train_set_filter_id": "tag::train_x",
                    "eval_configs_filter_id": "tag::golden_x",
                    "output_scores": [{"name": "score", "type": "pass_fail"}],
                    "model_type": "eval",
                }
            ),
            encoding="utf-8",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            loaded = Eval.load_from_file(eval_path)
            loaded.set_split("val", TaskRunSplit(filter_id="tag::val_x"))
            loaded.save_to_file()
            loaded.model_dump()
            loaded.model_dump_json()
            Eval.load_from_file(eval_path)

    def test_model_construct_is_not_migrated(self, scores):
        """model_construct skips validation, so the migration never runs.

        Nothing is lost — the fields keep the values they were given, and the dump is
        exactly what was constructed — but an unvalidated instance is not a migrated one.
        """
        eval = Eval.model_construct(
            name="Unvalidated",
            eval_set_filter_id="tag::eval_set_x",
            train_set_filter_id="tag::train_x",
            eval_configs_filter_id="tag::golden_x",
            output_scores=scores,
            splits={},
        )

        dumped = eval.model_dump()

        assert dumped["eval_set_filter_id"] == "tag::eval_set_x"
        assert dumped["train_set_filter_id"] == "tag::train_x"
        assert dumped["splits"] == {}

    def test_splits_native_eval_does_not_acquire_legacy_fields(
        self, saved_task, scores
    ):
        eval = self.build_eval(
            scores,
            parent=saved_task,
            splits={
                "test": TaskRunSplit(filter_id="tag::test_x"),
                "val": EvalInputSplit(filter_id="tag::val_x"),
            },
        )
        data = self.saved_json(eval)
        assert data["eval_set_filter_id"] is None
        assert data["train_set_filter_id"] is None
        assert data["splits"] == {
            "test": {"source": "task_run", "filter_id": "tag::test_x"},
            "val": {"source": "eval_input", "filter_id": "tag::val_x"},
        }

    def test_a_legacy_eval_gaining_a_val_split_saves_every_split_together(
        self, saved_task, scores
    ):
        """The mixed case: an existing eval gains a val split with the new tooling.

        All three end up in `splits`, rather than the migrated two staying behind in
        fields the val split could never share.
        """
        eval = self.build_eval(
            scores,
            parent=saved_task,
            eval_set_filter_id="tag::test_x",
            train_set_filter_id="tag::train_x",
        )
        eval.splits["val"] = EvalInputSplit(filter_id="tag::val_x")

        data = self.saved_json(eval)
        assert data["eval_set_filter_id"] is None
        assert data["train_set_filter_id"] is None
        assert data["splits"] == {
            "test": {"source": "task_run", "filter_id": "tag::test_x"},
            "train": {"source": "task_run", "filter_id": "tag::train_x"},
            "val": {"source": "eval_input", "filter_id": "tag::val_x"},
        }
        assert "val_set_filter_id" not in data

    def test_reserialized_eval_reloads_to_the_same_splits(self, saved_task, scores):
        eval = self.build_eval(
            scores,
            parent=saved_task,
            eval_set_filter_id="tag::test_x",
        )
        eval.splits["val"] = EvalInputSplit(filter_id="tag::val_x")
        eval.save_to_file()

        assert eval.path is not None
        reloaded = Eval.load_from_file(eval.path)
        assert reloaded.splits == eval.splits

    @pytest.mark.parametrize(
        "changed_to",
        [
            EvalInputSplit(filter_id="tag::inputs"),
            TaskRunSplit(filter_id="tag::other_runs"),
        ],
    )
    def test_repointing_a_migrated_test_split_writes_only_splits(
        self, saved_task, scores, changed_to
    ):
        """A migrated split is edited like any other: in `splits`, and only there.

        Saved twice, because saving is itself an assignment (`self.path = path`), which
        re-runs the validators: a split edit has to survive every later save of the same
        object, not just the first.
        """
        eval = self.build_eval(
            scores, parent=saved_task, eval_set_filter_id="tag::test_x"
        )
        eval.splits["test"] = changed_to

        for _ in range(2):
            data = self.saved_json(eval)
            assert data["eval_set_filter_id"] is None
            assert data["splits"] == {
                "test": {
                    "source": changed_to.source,
                    "filter_id": changed_to.filter_id,
                }
            }
            # The in-memory model still agrees with the file it just wrote.
            assert eval.splits["test"] == changed_to

    def test_a_split_edit_survives_unrelated_edits_and_saves(self, saved_task, scores):
        """The legacy fields are an input format, not state: nothing re-derives from them.

        Assignment re-runs the model validators, so the migration must not run a second
        time — otherwise an unrelated edit (here, a rename) would quietly revert the
        split. Clearing the fields as they are read is what guarantees that.
        """
        eval = self.build_eval(
            scores,
            parent=saved_task,
            eval_set_filter_id="tag::test_x",
            train_set_filter_id="tag::train_x",
        )
        eval.splits["train"] = EvalInputSplit(filter_id="tag::train_inputs")
        eval.name = "Renamed"
        eval.description = "unrelated edit"

        assert eval.splits["train"] == EvalInputSplit(filter_id="tag::train_inputs")
        data = self.saved_json(eval)
        assert data["train_set_filter_id"] is None
        assert data["splits"]["train"] == {
            "source": "eval_input",
            "filter_id": "tag::train_inputs",
        }

        assert eval.path is not None
        assert Eval.load_from_file(eval.path).splits["train"] == EvalInputSplit(
            filter_id="tag::train_inputs"
        )

    def test_whole_dict_assignment_replaces_splits(self, saved_task, scores):
        """`eval.splits = {...}` and `eval.splits[...] = ...` must not disagree."""
        eval = self.build_eval(
            scores, parent=saved_task, eval_set_filter_id="tag::legacy_test"
        )
        eval.splits = {"test": EvalInputSplit(filter_id="tag::inputs")}

        assert eval.splits == {"test": EvalInputSplit(filter_id="tag::inputs")}
        data = self.saved_json(eval)
        assert data["eval_set_filter_id"] is None
        assert data["splits"] == {
            "test": {"source": "eval_input", "filter_id": "tag::inputs"}
        }

    def test_whole_dict_assignment_dropping_a_split_removes_it(
        self, saved_task, scores
    ):
        """A key left out of the new dict is gone, and the migration cannot bring it back.

        Whole-dict assignment re-runs the validators, so a migration that had not already
        cleared the legacy fields would re-read them here and resurrect the dropped split.
        """
        eval = self.build_eval(
            scores,
            parent=saved_task,
            eval_set_filter_id="tag::test_x",
            train_set_filter_id="tag::train_x",
        )
        eval.splits = {"test": TaskRunSplit(filter_id="tag::test_x")}

        data = self.saved_json(eval)
        assert data["train_set_filter_id"] is None
        assert data["splits"] == {
            "test": {"source": "task_run", "filter_id": "tag::test_x"}
        }

        assert eval.path is not None
        assert "train" not in Eval.load_from_file(eval.path).splits

    def test_removing_a_split_removes_it_from_the_file(self, saved_task, scores):
        """Deleting a split deletes it from the file, not just from memory.

        Saving re-runs the validators, so this is the same guarantee as the whole-dict
        case: the migrated legacy field is already cleared, so nothing re-adds the split
        and the next load agrees.
        """
        eval = self.build_eval(
            scores,
            parent=saved_task,
            eval_set_filter_id="tag::test_x",
            train_set_filter_id="tag::train_x",
        )
        del eval.splits["train"]

        data = self.saved_json(eval)
        assert data["train_set_filter_id"] is None
        assert data["splits"] == {
            "test": {"source": "task_run", "filter_id": "tag::test_x"}
        }

        assert eval.path is not None
        reloaded = Eval.load_from_file(eval.path)
        assert "train" not in reloaded.splits
        assert reloaded.splits["test"] == TaskRunSplit(filter_id="tag::test_x")

    def test_assigning_a_legacy_field_does_not_edit_an_existing_split(
        self, saved_task, scores
    ):
        """Writing a legacy field is not how you edit a split, and it doesn't half-do it.

        `splits` wins over a legacy value wherever it comes from, including a late
        assignment, so the split is untouched and the assigned value is dropped rather
        than left on disk for an older client to read.
        """
        eval = self.build_eval(
            scores,
            parent=saved_task,
            eval_set_filter_id="tag::test_x",
            train_set_filter_id="tag::train_x",
        )
        eval.train_set_filter_id = "tag::sneak"

        assert eval.splits["train"] == TaskRunSplit(filter_id="tag::train_x")
        data = self.saved_json(eval)
        assert data["train_set_filter_id"] is None
        assert data["splits"]["train"] == {
            "source": "task_run",
            "filter_id": "tag::train_x",
        }

    def test_assigning_a_legacy_field_for_a_missing_split_is_migrated_once(
        self, saved_task, scores
    ):
        """A legacy value is migrated wherever it arrives from — and then it is gone.

        Assignment re-runs the validators, so a deprecated write behaves exactly like a
        deprecated constructor argument: it fills a split `splits` doesn't have, and the
        field is cleared. It never becomes a second home for it.
        """
        eval = self.build_eval(
            scores,
            parent=saved_task,
            splits={"test": TaskRunSplit(filter_id="tag::test_x")},
        )
        eval.train_set_filter_id = "tag::late"

        assert eval.splits["train"] == TaskRunSplit(filter_id="tag::late")
        data = self.saved_json(eval)
        assert data["train_set_filter_id"] is None
        assert data["splits"]["train"] == {
            "source": "task_run",
            "filter_id": "tag::late",
        }

    def test_clearing_a_legacy_field_does_not_remove_its_split(
        self, saved_task, scores
    ):
        """The mirror of the assignment case: writing None is not a deletion either."""
        eval = self.build_eval(
            scores,
            parent=saved_task,
            eval_set_filter_id="tag::test_x",
            train_set_filter_id="tag::train_x",
        )
        eval.train_set_filter_id = None

        assert eval.splits["train"] == TaskRunSplit(filter_id="tag::train_x")
        data = self.saved_json(eval)
        assert data["train_set_filter_id"] is None
        assert data["splits"]["train"] == {
            "source": "task_run",
            "filter_id": "tag::train_x",
        }

    def test_eval_input_backed_test_split_stays_in_splits(self, saved_task, scores):
        """An EvalInput-backed test split serializes into `splits`, never a legacy field."""
        eval = self.build_eval(
            scores,
            parent=saved_task,
            splits={"test": EvalInputSplit(filter_id="tag::inputs")},
        )
        assert eval.splits["test"] == EvalInputSplit(filter_id="tag::inputs")

        data = self.saved_json(eval)
        assert data["eval_set_filter_id"] is None
        assert data["splits"] == {
            "test": {"source": "eval_input", "filter_id": "tag::inputs"}
        }

    @pytest.mark.parametrize("source", ["task_run", "eval_input"])
    def test_unknown_field_inside_a_split_survives_a_round_trip(
        self, saved_task, scores, source
    ):
        """Forward compatibility one level below the split name.

        Both split types, because both are things a future build writes.
        """
        eval = Eval.model_validate(
            {
                "name": "Future Split",
                "parent": saved_task,
                "output_scores": [{"name": "score", "type": "pass_fail"}],
                "splits": {
                    "test": {
                        "source": source,
                        "filter_id": "tag::test_x",
                        "weight": 0.5,
                    }
                },
            }
        )
        data = self.saved_json(eval)
        assert data["splits"]["test"]["weight"] == 0.5

        assert eval.path is not None
        reloaded = Eval.load_from_file(eval.path)
        assert getattr(reloaded.splits["test"], "weight") == 0.5

    def test_unknown_split_source_fails_the_load(self, scores):
        """A deliberate limit: an unrecognized backing is a hard failure, not an opaque split.

        Unlike an unknown split *name*, which this build can ignore safely, an unknown
        *source* on a split this build addresses can't be resolved to items — accepting it
        would turn a loud load error into a silent "no items" at every reader, and would
        let an EvalInput-shaped split escape the filter-type guarantee EvalInputSplit
        exists to enforce. Recorded as a test so the trade is visible if a third source
        ever lands.
        """
        with pytest.raises(ValidationError):
            self.build_eval(
                scores, splits={"test": {"source": "warehouse", "filter_id": "tag::x"}}
            )

    def test_unknown_split_key_survives_a_round_trip(self, saved_task, scores):
        """A split name this build doesn't know loads, is ignored, and is not dropped."""
        eval = Eval.model_validate(
            {
                "name": "Future Eval",
                "parent": saved_task,
                "output_scores": [{"name": "score", "type": "pass_fail"}],
                "splits": {
                    "test": {"source": "task_run", "filter_id": "tag::test_x"},
                    "holdout": {"source": "eval_input", "filter_id": "tag::holdout_x"},
                },
            }
        )
        data = self.saved_json(eval)
        assert data["splits"]["holdout"] == {
            "source": "eval_input",
            "filter_id": "tag::holdout_x",
        }

        assert eval.path is not None
        assert Eval.load_from_file(eval.path).splits["holdout"] == EvalInputSplit(
            filter_id="tag::holdout_x"
        )

    @pytest.mark.parametrize(
        "filter_id",
        [
            "multi_filter::high_rating&all",
            "high_rating",
            # All four TaskRun-only forms functional spec 7 names, not just the two that
            # happen to differ in prefix: each is a rating or trace predicate over a
            # TaskRun, which an EvalInput has nothing to answer.
            "thinking_model",
            "thinking_model_high_rated",
        ],
    )
    def test_eval_input_split_rejects_task_run_only_filters(self, filter_id):
        """Functional spec 7, made structural: unrepresentable, not validator-enforced."""
        with pytest.raises(ValidationError):
            EvalInputSplit(filter_id=filter_id)
        assert TaskRunSplit(filter_id=filter_id).filter_id == filter_id

    def test_dict_round_trip_keeps_the_migrated_splits(self, scores):
        """A legacy eval rebuilt from its own dump keeps its splits, in the new format.

        The dump carries the splits and two nulls, so the rebuild has nothing left to
        migrate — the migration happened once, to the first instance.
        """
        eval = self.build_eval(
            scores,
            eval_set_filter_id="tag::test_x",
            train_set_filter_id="tag::train_x",
        )
        rebuilt = Eval(**eval.model_dump())
        rebuilt_data = rebuilt.model_dump()
        assert rebuilt_data["eval_set_filter_id"] is None
        assert rebuilt_data["train_set_filter_id"] is None
        assert rebuilt_data["splits"] == {
            "test": {"source": "task_run", "filter_id": "tag::test_x"},
            "train": {"source": "task_run", "filter_id": "tag::train_x"},
        }

    def test_model_copy_keeps_the_migrated_splits(self, scores):
        """A copy of a migrated eval is migrated too — there is no format to carry."""
        eval = self.build_eval(scores, eval_set_filter_id="tag::test_x")
        copied = eval.model_copy()
        assert copied.splits["test"] == TaskRunSplit(filter_id="tag::test_x")
        assert copied.model_dump()["eval_set_filter_id"] is None

    def test_a_migrated_split_survives_exclude_unset(self, scores):
        """A legacy eval never explicitly set `splits`, so the migration marks it set.

        Without the marking, an exclude_unset dump of a migrated eval carries neither the
        split nor a legacy field that could stand in for it.
        """
        eval = self.build_eval(scores, eval_set_filter_id="tag::test_x")

        # `source` is elided because it is a defaulted field the caller never set — that
        # is what exclude_unset does to any nested model, here and elsewhere in the repo.
        # What matters is that the split is present at all.
        data = eval.model_dump(exclude_unset=True)
        assert data["splits"]["test"]["filter_id"] == "tag::test_x"

    def test_writing_a_train_split_lands_in_splits(self, saved_task, scores):
        """What the eval-update endpoint does: same result whatever the eval arrived as."""
        migrated = self.build_eval(
            scores,
            parent=saved_task,
            eval_set_filter_id="tag::test_x",
            train_set_filter_id="tag::train_x",
        )
        migrated.splits["train"] = TaskRunSplit(filter_id="tag::train_updated")
        migrated_data = self.saved_json(migrated)
        assert migrated_data["train_set_filter_id"] is None
        assert migrated_data["splits"]["train"] == {
            "source": "task_run",
            "filter_id": "tag::train_updated",
        }

        native = self.build_eval(
            scores,
            parent=saved_task,
            splits={"test": TaskRunSplit(filter_id="tag::test_x")},
        )
        native.splits["train"] = TaskRunSplit(filter_id="tag::train_new")
        native_data = self.saved_json(native)
        assert native_data["train_set_filter_id"] is None
        assert native_data["splits"]["train"] == {
            "source": "task_run",
            "filter_id": "tag::train_new",
        }

    @pytest.mark.parametrize(
        "name,split",
        [
            ("train", TaskRunSplit(filter_id="tag::train_new")),
            ("train", EvalInputSplit(filter_id="tag::train_inputs")),
            ("val", TaskRunSplit(filter_id="tag::val_x")),
        ],
    )
    def test_set_split_stores_the_split_in_splits(
        self, saved_task, scores, name, split
    ):
        """Every split, whatever its name or backing, goes to the one home."""
        eval = self.build_eval(
            scores, parent=saved_task, eval_set_filter_id="tag::test_x"
        )
        eval.set_split(name, split)

        assert eval.splits[name] == split
        data = self.saved_json(eval)
        assert data["train_set_filter_id"] is None
        assert data["splits"][name] == {
            "source": split.source,
            "filter_id": split.filter_id,
        }

    def test_set_split_survives_exclude_unset(self, scores):
        """set_split marks `splits` set, which dict item assignment can't do for itself.

        Only reachable via model_construct: every validated eval has `splits` marked
        already, either because it was passed or because the legacy migration marked it.
        """
        eval = Eval.model_construct(name="Constructed", output_scores=scores)
        assert "splits" not in eval.model_fields_set

        eval.set_split("val", EvalInputSplit(filter_id="tag::val_x"))

        data = eval.model_dump(exclude_unset=True)
        assert data["splits"]["val"]["filter_id"] == "tag::val_x"

    def test_set_split_refuses_to_mutate_a_readonly_eval(self, saved_task, scores):
        """Readonly instances are the cached ones, shared with every other holder.

        `eval.splits[...] = ...` and set_split mutate a dict, which never reaches
        __setattr__ — so the readonly check has to be explicit, or a caller that took a
        readonly copy silently edits everyone else's.
        """
        eval = self.build_eval(
            scores, parent=saved_task, eval_set_filter_id="tag::test_x"
        )
        eval.save_to_file()
        assert eval.path is not None

        readonly = Eval.load_from_file(eval.path, readonly=True)
        with pytest.raises(ReadOnlyMutationError):
            readonly.set_split("train", TaskRunSplit(filter_id="tag::train_x"))
        assert "train" not in readonly.splits

    def test_set_split_and_direct_assignment_agree(self, saved_task, scores):
        """The two ways to write a split differ only in bookkeeping, not in outcome."""
        via_set_split = self.build_eval(
            scores, parent=saved_task, eval_set_filter_id="tag::test_x"
        )
        via_set_split.set_split("train", TaskRunSplit(filter_id="tag::train_x"))

        via_assignment = self.build_eval(
            scores, parent=saved_task, eval_set_filter_id="tag::test_x"
        )
        via_assignment.splits["train"] = TaskRunSplit(filter_id="tag::train_x")

        assert via_set_split.splits == via_assignment.splits
        assert (
            self.saved_json(via_set_split)["splits"]
            == self.saved_json(via_assignment)["splits"]
        )


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Code-as-file storage for code judges
#
# CodeEvalProperties.code lives in a sibling scorer.py, not inline in
# eval_config.kiln. CodeEvalProperties is a nested member of the
# V2EvalConfigProperties discriminated union in EvalConfig.properties, so the
# load/save context set on the parent EvalConfig must propagate down to it.
# ──────────────────────────────────────────────────────────────────────


VALID_SCORE = "def score(output, trace, reference_data, task_input):\n    return {'accuracy': 1.0}\n"
ASYNC_SCORE = "async def score(output, trace, reference_data, task_input):\n    return {'accuracy': 1.0}\n"


def _saved_task(tmp_path) -> Task:
    task = Task(name="Code Judge Task", instruction="Test instruction")
    task.path = tmp_path / "task" / "task.kiln"
    task.save_to_file()
    return task


def _saved_eval(task) -> Eval:
    eval_obj = Eval(
        name="Code Judge Eval",
        parent=task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(name="accuracy", type=TaskOutputRatingType.pass_fail),
        ],
    )
    eval_obj.save_to_file()
    return eval_obj


def _saved_code_eval_config(tmp_path, code=VALID_SCORE, **prop_overrides) -> EvalConfig:
    """Build and persist a Task -> Eval -> EvalConfig(v2, CodeEvalProperties)."""
    eval_obj = _saved_eval(_saved_task(tmp_path))
    config = EvalConfig(
        name="Code Judge Config",
        parent=eval_obj,
        config_type=EvalConfigType.v2,
        properties=CodeEvalProperties(code=code, **prop_overrides),
    )
    config.save_to_file()
    return config


class TestCodeEvalFileStorage:
    def test_context_propagates_to_nested_union_member(self, tmp_path):
        # The Phase-2 hinge: validation + serialization context must reach the
        # nested CodeEvalProperties inside the discriminated union. A full disk
        # round-trip through EvalConfig proves both directions at once.
        config = _saved_code_eval_config(
            tmp_path, code=VALID_SCORE, reference_keys=["gold"], timeout_seconds=90
        )

        # Serialization context reached the nested member: code is on disk in
        # scorer.py and absent from the serialized properties.
        scorer_py = config.path.parent / SCORER_CODE_FILENAME
        assert scorer_py.read_text(encoding="utf-8") == VALID_SCORE
        on_disk = json.loads(config.path.read_text(encoding="utf-8"))
        assert "code" not in on_disk["properties"]

        # Validation context reached the nested member: code is reconstructed on load.
        loaded = EvalConfig.load_from_file(config.path)
        assert isinstance(loaded.properties, CodeEvalProperties)
        assert loaded.properties.code == VALID_SCORE
        assert loaded.properties.reference_keys == ["gold"]
        assert loaded.properties.timeout_seconds == 90

    def test_save_writes_scorer_py_and_omits_code_from_properties(self, tmp_path):
        config = _saved_code_eval_config(
            tmp_path, code=VALID_SCORE, reference_keys=["gold"], timeout_seconds=42
        )

        scorer_py = config.path.parent / SCORER_CODE_FILENAME
        assert scorer_py.exists()
        assert scorer_py.read_text(encoding="utf-8") == VALID_SCORE

        on_disk_props = json.loads(config.path.read_text(encoding="utf-8"))[
            "properties"
        ]
        assert "code" not in on_disk_props
        # Discriminator + other functional fields still live in the .kiln JSON.
        assert on_disk_props["type"] == "code_eval"
        assert on_disk_props["reference_keys"] == ["gold"]
        assert on_disk_props["timeout_seconds"] == 42

    def test_load_reconstructs_code_from_scorer_py(self, tmp_path):
        config = _saved_code_eval_config(tmp_path, code=ASYNC_SCORE)

        loaded = EvalConfig.load_from_file(config.path)
        assert isinstance(loaded.properties, CodeEvalProperties)
        assert loaded.properties.code == ASYNC_SCORE

    def test_missing_scorer_py_fails_load(self, tmp_path):
        config = _saved_code_eval_config(tmp_path)
        (config.path.parent / SCORER_CODE_FILENAME).unlink()

        with pytest.raises(ValueError, match=SCORER_CODE_FILENAME):
            EvalConfig.load_from_file(config.path)

    def test_corrupted_scorer_py_fails_validator_on_load(self, tmp_path):
        config = _saved_code_eval_config(tmp_path)
        # Hand-edit scorer.py to source without a module-level `score` function.
        (config.path.parent / SCORER_CODE_FILENAME).write_text(
            "def helper():\n    pass\n", encoding="utf-8"
        )

        with pytest.raises(ValidationError, match="module-level 'score' function"):
            EvalConfig.load_from_file(config.path)

    def test_save_is_idempotent(self, tmp_path):
        config = _saved_code_eval_config(tmp_path, code=VALID_SCORE)
        scorer_py = config.path.parent / SCORER_CODE_FILENAME

        first_kiln = config.path.read_bytes()
        first_py = scorer_py.read_bytes()

        loaded = EvalConfig.load_from_file(config.path)
        loaded.save_to_file()

        assert config.path.read_bytes() == first_kiln
        assert scorer_py.read_bytes() == first_py

    def test_api_dump_keeps_code(self):
        # Without the save context, code stays in the dump and no file is written.
        props = CodeEvalProperties(code=VALID_SCORE)
        config = EvalConfig(
            name="Code Judge Config",
            config_type=EvalConfigType.v2,
            properties=props,
        )

        with patch.object(Path, "write_text") as mock_write:
            assert props.model_dump()["code"] == VALID_SCORE
            assert json.loads(props.model_dump_json())["code"] == VALID_SCORE
            # Also true when dumped as part of the parent EvalConfig (API shape).
            assert config.model_dump()["properties"]["code"] == VALID_SCORE
            assert (
                json.loads(config.model_dump_json())["properties"]["code"]
                == VALID_SCORE
            )
        mock_write.assert_not_called()

    def test_source_dir_missing_from_load_context_fails(self):
        # Defensive guard: loading_from_file set but source_dir absent (a future
        # base-model regression) fails clearly rather than silently skipping.
        with pytest.raises(
            ValidationError, match="source_dir missing from load context"
        ):
            CodeEvalProperties.model_validate(
                {"type": "code_eval"}, context={"loading_from_file": True}
            )

    def test_read_path_type_gate_rejects_mismatched_type(self):
        # Defense-in-depth: the CodeEvalProperties read path only handles
        # code_eval properties. A present, mismatched `type` is rejected before
        # any scorer.py read, rather than silently proceeding.
        with pytest.raises(ValidationError, match="can only load code_eval properties"):
            CodeEvalProperties.model_validate(
                {"type": "llm_judge", "code": VALID_SCORE}
            )

    def test_read_path_type_gate_allows_absent_and_enum_type(self):
        # None (type omitted, field defaults) and the enum form both pass the
        # gate — valid-input behavior is unchanged.
        assert CodeEvalProperties(code=VALID_SCORE).type == V2EvalType.code_eval
        assert (
            CodeEvalProperties.model_validate(
                {"type": V2EvalType.code_eval, "code": VALID_SCORE}
            ).code
            == VALID_SCORE
        )

    def test_serialize_rejects_non_directory_dest_path(self, tmp_path):
        props = CodeEvalProperties(code=VALID_SCORE)
        not_a_dir = tmp_path / "does_not_exist"
        with pytest.raises(ValueError, match="dest_path must be an existing directory"):
            props.model_dump(context={"save_attachments": True, "dest_path": not_a_dir})

    def test_other_eval_type_writes_no_sibling_file(self, tmp_path):
        # A v2 config with a non-code property type writes no scorer.py.
        eval_obj = _saved_eval(_saved_task(tmp_path))
        llm_config = EvalConfig(
            name="LLM Judge Config",
            parent=eval_obj,
            config_type=EvalConfigType.v2,
            properties=LlmJudgeProperties(
                model_name="gpt-4o",
                model_provider="openai",
                prompt_template="Evaluate: {{ final_message }}",
            ),
        )
        llm_config.save_to_file()
        assert not (llm_config.path.parent / SCORER_CODE_FILENAME).exists()

        # A legacy g_eval config likewise writes no scorer.py.
        legacy_config = EvalConfig(
            name="Legacy Config",
            parent=eval_obj,
            config_type=EvalConfigType.g_eval,
            model_name="gpt-4",
            model_provider="openai",
            properties={"eval_steps": ["s1"]},
        )
        legacy_config.save_to_file()
        assert not (legacy_config.path.parent / SCORER_CODE_FILENAME).exists()

    def test_inline_code_in_properties_is_lenient(self, tmp_path):
        # A properties dict that already carries `code` (e.g. an in-memory dict
        # passed to model_validate with load context) uses it as-is and does not
        # touch disk — the graceful-construction property (functional spec §7).
        with patch.object(Path, "read_text") as mock_read:
            props = CodeEvalProperties.model_validate(
                {"type": "code_eval", "code": VALID_SCORE},
                context={"loading_from_file": True, "source_dir": tmp_path},
            )
        assert props.code == VALID_SCORE
        mock_read.assert_not_called()

    def test_serialization_schema_matches_validation_schema(self):
        # The wrap serializer returns an untyped dict, which would collapse the
        # serialization-mode JSON schema to {additionalProperties: true, type:
        # object}. The __get_pydantic_json_schema__ override keeps it identical
        # to validation mode so EvalConfig's OpenAPI (EvalConfig is a FastAPI
        # response_model) does not drift the committed api_schema.d.ts.
        validation_schema = CodeEvalProperties.model_json_schema(mode="validation")
        serialization_schema = CodeEvalProperties.model_json_schema(
            mode="serialization"
        )
        assert serialization_schema == validation_schema
        # code stays a typed field (not lost to the collapse) in both modes.
        assert serialization_schema["properties"]["code"]["type"] == "string"

    def test_openapi_component_is_single_and_typed(self):
        # Reproduce the reviewer's check: a minimal FastAPI app whose
        # response_model is the real EvalConfig must emit a single, fully typed
        # CodeEvalProperties component — no -Input/-Output split, no collapse.
        fastapi = pytest.importorskip("fastapi")

        app = fastapi.FastAPI()

        @app.get("/config", response_model=EvalConfig)
        def _get_config():  # pragma: no cover - schema-only endpoint
            return None

        components = app.openapi()["components"]["schemas"]
        code_eval_names = [n for n in components if "CodeEvalProperties" in n]
        assert code_eval_names == ["CodeEvalProperties"]

        component = components["CodeEvalProperties"]
        assert "code" in component.get("properties", {})
        assert component["properties"]["code"]["type"] == "string"


# ── EvalRun record modes: pointer / skipped / legacy inline ────────────


def pointer_run_data(**overrides):
    data = {
        "eval_input_id": "ei1",
        "task_run_config_id": "rc1",
        "scored_run_id": "tr1",
        "scores": {"accuracy": 1.0},
    }
    data.update(overrides)
    return data


@pytest.fixture
def v2_eval_config(mock_task, tmp_path):
    """A saved V2 EvalConfig, for the record-mode tests that go to disk.

    validate_record_mode is parent-independent, so only the round-trip tests need this.
    """
    mock_task.path = tmp_path / "task.kiln"
    mock_task.save_to_file()

    eval = Eval(
        name="Record Mode Eval",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        output_scores=[
            EvalOutputScore(name="accuracy", type=TaskOutputRatingType.pass_fail)
        ],
    )
    eval.save_to_file()

    config = EvalConfig(
        name="V2 Judge",
        parent=eval,
        config_type=EvalConfigType.v2,
        properties=LlmJudgeProperties(
            model_name="gpt-4o",
            model_provider="openai",
            prompt_template="Evaluate: {{ final_message }}",
        ),
    )
    config.save_to_file()
    return config


def test_pointer_eval_run_is_valid_and_round_trips(v2_eval_config):
    run = EvalRun(parent=v2_eval_config, **pointer_run_data())
    run.save_to_file()

    loaded = EvalRun.load_from_file(str(run.path))
    assert loaded.scored_run_id == "tr1"
    assert loaded.input is None
    assert all(getattr(loaded, f) is None for f in LEGACY_TRACE_FIELDS)


INLINE_TRACE_CASES = [
    ("input", "some input"),
    ("output", "some output"),
    ("task_run_trace", '{"messages": []}'),
    ("task_run_usage", Usage(input_tokens=1)),
    ("reference_answer", "gold"),
]


def test_inline_trace_cases_cover_every_forbidden_field():
    """The values below can't be derived, but the field list can: without this, adding
    a field to LEGACY_TRACE_FIELDS would silently get a deprecation assertion and no
    rejection assertion."""
    assert {name for name, _ in INLINE_TRACE_CASES} == {"input", *LEGACY_TRACE_FIELDS}


@pytest.mark.parametrize("field,value", INLINE_TRACE_CASES)
def test_pointer_eval_run_rejects_inline_trace_data(field, value):
    with pytest.raises(ValidationError, match="must not carry inline trace data"):
        EvalRun(**pointer_run_data(**{field: value}))


def test_pointer_eval_run_error_names_every_field_carried():
    with pytest.raises(ValidationError) as exc_info:
        EvalRun(**pointer_run_data(input="in", output="out", reference_answer="gold"))
    message = str(exc_info.value)
    assert "input" in message
    assert "output" in message
    assert "reference_answer" in message


def test_legacy_eval_run_without_input_is_rejected():
    with pytest.raises(ValidationError, match="requires input"):
        EvalRun(
            eval_input_id="ei1",
            task_run_config_id="rc1",
            output="some output",
            scores={"accuracy": 1.0},
        )


def test_legacy_eval_run_with_input_is_valid():
    run = EvalRun(
        eval_input_id="ei1",
        task_run_config_id="rc1",
        input="some input",
        output="some output",
        scores={"accuracy": 1.0},
    )
    assert run.scored_run_id is None
    assert run.input == "some input"


def test_skipped_eval_run_needs_neither_input_nor_pointer():
    """A skip before generation has nothing to point at, and nothing to copy."""
    run = EvalRun(
        eval_input_id="ei1",
        task_run_config_id="rc1",
        skipped_reason=SkippedReason.incompatible_input_shape.value,
    )
    assert run.scored_run_id is None
    assert run.input is None


def test_skipped_at_scoring_time_keeps_its_pointer():
    """The trace existed; only scoring was skipped. The pointer is still meaningful."""
    run = EvalRun(
        eval_input_id="ei1",
        task_run_config_id="rc1",
        scored_run_id="tr1",
        skipped_reason=SkippedReason.missing_reference_key.value,
    )
    assert run.scored_run_id == "tr1"


def test_skipped_at_scoring_time_still_rejects_inline_data():
    """The pointer branch is checked before the skip branch on purpose."""
    with pytest.raises(ValidationError, match="must not carry inline trace data"):
        EvalRun(
            eval_input_id="ei1",
            task_run_config_id="rc1",
            scored_run_id="tr1",
            skipped_reason=SkippedReason.missing_reference_key.value,
            output="some output",
        )


def test_legacy_skipped_eval_run_with_inline_data_still_loads():
    """Records written before the split carry input on skips. They stay valid."""
    run = EvalRun(
        eval_input_id="ei1",
        task_run_config_id="rc1",
        input="some input",
        skipped_reason=SkippedReason.missing_trace.value,
    )
    assert run.input == "some input"


def test_pointer_eval_run_bypasses_v1_output_requirement(
    mock_task, valid_eval_config_data, tmp_path
):
    """validate_output_fields' 'V1 EvalRun requires output' rule must not fire for a
    pointer record: its output lives on the referenced TaskRun."""
    mock_task.path = tmp_path / "task.kiln"
    mock_task.save_to_file()

    eval = Eval(
        name="V1 Eval",
        parent=mock_task,
        eval_set_filter_id="tag::tag1",
        eval_configs_filter_id="tag::tag2",
        evaluation_data_type=EvalDataType.full_trace,
        output_scores=[
            EvalOutputScore(name="accuracy", type=TaskOutputRatingType.pass_fail)
        ],
    )
    eval.save_to_file()
    config = EvalConfig(parent=eval, **valid_eval_config_data)
    config.save_to_file()

    # Without the pointer, a V1 full_trace run with no output and no trace is rejected
    # twice over ("requires output", "should include trace").
    with pytest.raises(ValidationError):
        EvalRun(
            parent=config,
            dataset_id="ds1",
            task_run_config_id="rc1",
            input="in",
            scores={"accuracy": 1.0},
        )

    run = EvalRun(
        parent=config,
        dataset_id="ds1",
        task_run_config_id="rc1",
        scored_run_id="tr1",
        scores={"accuracy": 1.0},
    )
    assert run.output is None
    assert run.task_run_trace is None


def test_eval_usage_defaults_to_none_and_round_trips(v2_eval_config):
    run = EvalRun(parent=v2_eval_config, **pointer_run_data())
    assert run.eval_usage is None

    run.eval_usage = Usage(input_tokens=10, output_tokens=3, cost=0.002)
    run.save_to_file()

    loaded = EvalRun.load_from_file(str(run.path))
    assert loaded.eval_usage is not None
    assert loaded.eval_usage.input_tokens == 10
    assert loaded.eval_usage.cost == 0.002


@pytest.mark.parametrize("field_name", ["input", *LEGACY_TRACE_FIELDS])
def test_legacy_trace_fields_are_marked_deprecated(field_name):
    """Two signals, for two audiences: the description prefix for a human reading the
    SDK docs, and the schema flag, which openapi-typescript turns into a `@deprecated`
    JSDoc tag so the TS compiler strikes through every web call site."""
    field = EvalRun.model_fields[field_name]
    assert field.description is not None
    assert field.description.startswith("DEPRECATED:")

    schema_property = EvalRun.model_json_schema()["properties"][field_name]
    assert schema_property["deprecated"] is True


def test_reading_a_deprecated_field_does_not_warn():
    """Why the flag is json_schema_extra and not Field(deprecated=True): reading these
    is the correct, permanent way to render a legacy record, and Field(deprecated=True)
    would warn on every one of those reads. Switching to it fails here."""
    run = EvalRun(
        eval_input_id="ei1",
        task_run_config_id="rc1",
        input="legacy input",
        output="legacy output",
        scores={"accuracy": 1.0},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for field_name in ("input", *LEGACY_TRACE_FIELDS):
            getattr(run, field_name)


def test_live_eval_run_fields_are_not_marked_deprecated():
    """The flag has to be per-field, not smeared across the model."""
    schema_properties = EvalRun.model_json_schema()["properties"]
    for field_name in ("scored_run_id", "eval_usage", "scores", "intermediate_outputs"):
        assert "deprecated" not in schema_properties[field_name], field_name


def test_eval_config_eval_requires_a_dataset_item():
    """Judge calibration compares against human ratings, which only dataset
    items carry — a calibration record claiming an EvalInput is domain-invalid
    and must be rejected, not silently persisted."""
    with pytest.raises(
        ValidationError, match="eval_config_eval records must score a dataset item"
    ):
        EvalRun(
            eval_config_eval=True,
            task_run_config_id=None,
            dataset_id=None,
            eval_input_id="ei_1",
            input="in",
            output="out",
            scores={"accuracy": 1.0},
        )
