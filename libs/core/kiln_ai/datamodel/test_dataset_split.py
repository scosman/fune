from unittest.mock import Mock

import pytest
from pydantic import ValidationError

# import datamodel first or we get circular import errors
from kiln_ai.datamodel import (
    DatasetSplit,
    DatasetSplitDefinition,
    DataSource,
    DataSourceType,
    Task,
    TaskOutput,
    TaskOutputRating,
    TaskOutputRatingType,
    TaskRun,
)
from kiln_ai.datamodel.datamodel_enums import ModelProviderName, StructuredOutputMode
from kiln_ai.datamodel.dataset_split import (
    AllSplitDefinition,
    Train60Test20Val20SplitDefinition,
    Train80Test20SplitDefinition,
    Train80Val20SplitDefinition,
)
from kiln_ai.datamodel.prompt_id import PromptGenerators
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties, ToolsRunConfig
from kiln_ai.datamodel.test_dataset_filters import (
    AllDatasetFilter,
    HighRatingDatasetFilter,
    TagFilter,
    ThinkingModelDatasetFilter,
    ThinkingModelHighRatedFilter,
)


@pytest.fixture
def sample_task(tmp_path):
    task_path = tmp_path / "task.kiln"
    task = Task(
        name="Test Task",
        path=task_path,
        description="Test task for dataset splitting",
        instruction="Test instruction",
    )
    task.save_to_file()
    return task


@pytest.fixture
def sample_task_runs(sample_task):
    # Create 10 task runs with different ratings
    task_runs = []
    for i in range(10):
        rating = 5 if i < 6 else 1  # 6 high, 4 low ratings
        tags = ["tag1"] if i < 6 else []
        task_run = TaskRun(
            parent=sample_task,
            input=f"input_{i}",
            input_source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "test-user"},
            ),
            output=TaskOutput(
                output=f"output_{i}",
                source=DataSource(
                    type=DataSourceType.human,
                    properties={"created_by": "test-user"},
                ),
                rating=TaskOutputRating(
                    value=rating, type=TaskOutputRatingType.five_star
                ),
            ),
            tags=tags,
        )
        task_run.save_to_file()
        task_runs.append(task_run)
    return task_runs


@pytest.fixture
def task_run():
    return TaskRun(
        input="test input",
        input_source=DataSource(
            type=DataSourceType.human,
            properties={"created_by": "test-user"},
        ),
        output=TaskOutput(
            output="test output",
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "test-user"},
            ),
            rating=TaskOutputRating(value=5, type=TaskOutputRatingType.five_star),
        ),
    )


def test_dataset_split_definition():
    split = DatasetSplitDefinition(name="train", percentage=0.8)
    assert split.name == "train"
    assert split.percentage == 0.8
    assert split.description is None

    # Test validation
    with pytest.raises(ValidationError):
        DatasetSplitDefinition(name="train", percentage=1.5)


def test_dataset_split_validation():
    # Test valid percentages
    splits = [
        DatasetSplitDefinition(name="train", percentage=0.8),
        DatasetSplitDefinition(name="test", percentage=0.2),
    ]
    dataset = DatasetSplit(
        name="test_split",
        splits=splits,
        split_contents={"train": [], "test": []},
    )
    assert dataset.splits == splits

    # Test invalid percentages
    invalid_splits = [
        DatasetSplitDefinition(name="train", percentage=0.8),
        DatasetSplitDefinition(name="test", percentage=0.3),
    ]
    with pytest.raises(ValueError, match=r"sum of split percentages must be 1.0"):
        DatasetSplit(
            name="test_split",
            splits=invalid_splits,
            split_contents={"train": [], "test": []},
        )


def test_all_dataset_filter(task_run):
    assert AllDatasetFilter(task_run) is True


def test_high_rating_dataset_filter(sample_task_runs):
    num_high_quality = 0
    num_low_quality = 0
    for task_run in sample_task_runs:
        if HighRatingDatasetFilter(task_run):
            num_high_quality += 1
            assert task_run.output.rating.is_high_quality() is True
        else:
            num_low_quality += 1
            assert task_run.output.rating.is_high_quality() is False

        # Test repaired output always considered high quality
        task_run = task_run.model_copy(
            update={
                "repair_instructions": "repair instructions",
                "repaired_output": TaskOutput(
                    output="repaired output",
                    source=DataSource(
                        type=DataSourceType.human,
                        properties={"created_by": "test-user"},
                    ),
                ),
            }
        )
        assert HighRatingDatasetFilter(task_run) is True

    assert num_high_quality == 6
    assert num_low_quality == 4


@pytest.mark.parametrize(
    "splits,expected_sizes",
    [
        (Train80Test20SplitDefinition, {"train": 8, "test": 2}),
        (AllSplitDefinition, {"all": 10}),
        (Train80Val20SplitDefinition, {"train": 8, "val": 2}),
        (Train60Test20Val20SplitDefinition, {"train": 6, "test": 2, "val": 2}),
        (
            [
                DatasetSplitDefinition(name="train", percentage=0.7),
                DatasetSplitDefinition(name="validation", percentage=0.2),
                DatasetSplitDefinition(name="test", percentage=0.1),
            ],
            {"train": 7, "validation": 2, "test": 1},
        ),
    ],
)
def test_dataset_split_from_task(sample_task, sample_task_runs, splits, expected_sizes):
    assert sample_task_runs is not None
    dataset = DatasetSplit.from_task("Split Name", sample_task, splits)
    assert dataset.name == "Split Name"

    # Check split sizes match expected
    for split_name, expected_size in expected_sizes.items():
        assert len(dataset.split_contents[split_name]) == expected_size

    # Verify total size matches input size
    total_size = sum(len(ids) for ids in dataset.split_contents.values())
    assert total_size == len(sample_task_runs)


def test_dataset_split_with_high_rating_filter(sample_task, sample_task_runs):
    assert len(sample_task_runs) == 10
    dataset = DatasetSplit.from_task(
        "Split Name",
        sample_task,
        Train80Test20SplitDefinition,
        filter_id="high_rating",
    )

    assert dataset.filter == "high_rating"

    # Check that only high-rated task runs are included
    all_ids = []
    for ids in dataset.split_contents.values():
        all_ids.extend(ids)
    assert len(all_ids) == 6  # We created 6 high-rated task runs

    # Check split proportions
    train_size = len(dataset.split_contents["train"])
    test_size = len(dataset.split_contents["test"])
    assert train_size == 5  # ~80% of 6
    assert test_size == 1  # ~20% of 6


def test_dataset_split_with_single_split(sample_task, sample_task_runs):
    splits = [DatasetSplitDefinition(name="all", percentage=1.0)]
    dataset = DatasetSplit.from_task("Split Name", sample_task, splits)

    assert len(dataset.split_contents["all"]) == len(sample_task_runs)


def test_missing_count(sample_task, sample_task_runs):
    assert sample_task_runs is not None
    # Create a dataset split with all task runs
    dataset = DatasetSplit.from_task(
        "Split Name", sample_task, Train80Test20SplitDefinition
    )

    # Initially there should be no missing runs
    assert dataset.missing_count() == 0

    # Add some IDs to the split, that aren't on disk
    dataset.split_contents["test"].append("1")
    dataset.split_contents["test"].append("2")
    dataset.split_contents["test"].append("3")
    # shouldn't happen, but should not double count if it does
    dataset.split_contents["train"].append("3")

    # Now we should have 3 missing runs
    assert dataset.missing_count() == 3


def test_smaller_sample(sample_task, sample_task_runs):
    assert sample_task_runs is not None
    # Create a dataset split with all task runs
    dataset = DatasetSplit.from_task(
        "Split Name", sample_task, Train80Test20SplitDefinition
    )

    # Initially there should be no missing runs
    assert dataset.missing_count() == 0

    dataset.split_contents["test"].pop()
    dataset.split_contents["train"].pop()

    # Now we should have 0 missing runs. It's okay that dataset has newer data.
    assert dataset.missing_count() == 0


@pytest.mark.parametrize(
    "thinking_data,expected_result",
    [
        ({"reasoning": "Here's my answer"}, True),
        ({"chain_of_thought": "Here's my answer"}, True),
        ({"unknown": "Here's my answer"}, False),
        ({}, False),
        (None, False),
    ],
)
def test_thinking_model_dataset_filter(
    sample_task_runs, thinking_data, expected_result
):
    # Create a task run with thinking output
    task_run = sample_task_runs[0].model_copy(
        update={
            "output": TaskOutput(
                output="Let me think about this...\nHere's my answer",
                source=DataSource(
                    type=DataSourceType.human,
                    properties={"created_by": "test-user"},
                ),
                rating=TaskOutputRating(value=5, type=TaskOutputRatingType.five_star),
            ),
            "intermediate_outputs": thinking_data,
        }
    )

    assert ThinkingModelDatasetFilter(task_run) is expected_result


@pytest.mark.parametrize(
    "thinking_data,rating,expected_result",
    [
        ({"reasoning": "Here's my answer"}, 5, True),
        ({"chain_of_thought": "Here's my answer"}, 5, True),
        ({"unknown": "Here's my answer"}, 5, False),
        ({}, 5, False),
        (None, 5, False),
        ({"reasoning": "Here's my answer"}, 1, False),
        ({"chain_of_thought": "Here's my answer"}, 1, False),
        ({"unknown": "Here's my answer"}, 1, False),
        ({}, 1, False),
        (None, 1, False),
    ],
)
def test_thinking_model_dataset_filter_high_rated(
    sample_task_runs, thinking_data, rating, expected_result
):
    # Create a task run with thinking output
    task_run = sample_task_runs[0].model_copy(
        update={
            "output": TaskOutput(
                output="Let me think about this...\nHere's my answer",
                source=DataSource(
                    type=DataSourceType.human,
                    properties={"created_by": "test-user"},
                ),
                rating=TaskOutputRating(
                    value=rating, type=TaskOutputRatingType.five_star
                ),
            ),
            "intermediate_outputs": thinking_data,
        }
    )

    assert ThinkingModelHighRatedFilter(task_run) is expected_result


def test_tag_dataset_filter(sample_task_runs):
    num_tagged = 0
    num_untagged = 0
    filter = TagFilter("tag1")
    for task_run in sample_task_runs:
        if "tag1" in task_run.tags:
            num_tagged += 1
            assert "tag1" in task_run.tags
            assert filter(task_run) is True
        else:
            num_untagged += 1
            assert "tag1" not in task_run.tags
            assert filter(task_run) is False

    assert num_tagged == 6
    assert num_untagged == 4


def test_get_runs(sample_task, sample_task_runs):
    dataset = DatasetSplit.from_task(
        "Test Dataset", sample_task, Train80Test20SplitDefinition
    )

    runs = dataset._get_runs()

    assert len(runs) == len(sample_task_runs)
    run_ids = {run.id for run in runs}
    expected_ids = {run.id for run in sample_task_runs}
    assert run_ids == expected_ids


@pytest.fixture
def mock_run_with_tools():
    """Factory fixture to create mock TaskRun objects with specified tools."""

    def _create_mock_run(tools):
        run = Mock()
        tools_config = None
        if tools is not None:
            tools_config = ToolsRunConfig(tools=tools if tools else [])
        run.output.source.run_config = KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE,
            structured_output_mode=StructuredOutputMode.json_schema,
            tools_config=tools_config,
        )
        return run

    return _create_mock_run


@pytest.mark.parametrize(
    "tools_list, expected_mismatch, expected_tools",
    [
        (
            [
                ["kiln_tool::add_numbers", "kiln_tool::subtract_numbers"],
                ["kiln_tool::add_numbers", "kiln_tool::subtract_numbers"],
            ],
            False,
            ["kiln_tool::add_numbers", "kiln_tool::subtract_numbers"],
        ),
        (
            [
                ["kiln_tool::subtract_numbers", "kiln_tool::add_numbers"],
                ["kiln_tool::add_numbers", "kiln_tool::subtract_numbers"],
            ],
            False,
            ["kiln_tool::add_numbers", "kiln_tool::subtract_numbers"],
        ),
        (
            [["kiln_tool::add_numbers"], ["kiln_tool::multiply_numbers"]],
            True,
            None,
        ),
        # both runs have no tools
        ([None, None], False, []),
        # one run has tools, the other has none
        ([["kiln_tool::add_numbers"], None], True, None),
        ([], False, []),
        ([["kiln_tool::add_numbers"]], False, ["kiln_tool::add_numbers"]),
        ([None], False, []),
        # both runs have empty tools
        ([[], None], False, []),
    ],
    ids=[
        "matching_tools",
        "same_tools_different_order",
        "mismatched_tools",
        "no_tools_multiple_runs",
        "mixed_tools_and_no_tools",
        "empty_runs_list",
        "single_run_with_tools",
        "single_run_without_tools",
        "two_runs_without_tools",
    ],
)
def test_compute_tool_info(
    mock_run_with_tools, tools_list, expected_mismatch, expected_tools
):
    runs = [mock_run_with_tools(tools) for tools in tools_list]
    tool_info = DatasetSplit.compute_tool_info(runs)
    assert tool_info.has_tool_mismatch is expected_mismatch
    assert tool_info.tools == expected_tools


def test_compute_tool_info_treats_missing_config_as_empty_tools(mock_run_with_tools):
    run_with_tools = mock_run_with_tools(["kiln_tool::add_numbers"])
    run_no_source = Mock()
    run_no_source.output.source = None
    run_no_config = Mock()
    run_no_config.output.source.run_config = None

    tool_info = DatasetSplit.compute_tool_info(
        [run_no_source, run_with_tools, run_no_config]
    )
    assert tool_info.has_tool_mismatch is True
    assert tool_info.tools is None


def _make_task_run(task, *, input_text, output_text, parent_task_run_id=None):
    run = TaskRun(
        parent=task,
        parent_task_run_id=parent_task_run_id,
        input=input_text,
        input_source=DataSource(
            type=DataSourceType.human,
            properties={"created_by": "test-user"},
        ),
        output=TaskOutput(
            output=output_text,
            source=DataSource(
                type=DataSourceType.human,
                properties={"created_by": "test-user"},
            ),
        ),
    )
    run.save_to_file()
    return run


def test_build_split_contents_multiturn_excludes_interior_nodes(tmp_path):
    task_path = tmp_path / "multiturn_task.kiln"
    task = Task(
        name="Multiturn Task",
        path=task_path,
        description="Multiturn dataset filter test",
        instruction="Test instruction",
        turn_mode="multiturn",
    )
    task.save_to_file()

    run_a = _make_task_run(task, input_text="turn 1", output_text="reply 1")
    run_b = _make_task_run(
        task,
        input_text="turn 2",
        output_text="reply 2",
        parent_task_run_id=run_a.id,
    )
    leaf = _make_task_run(
        task,
        input_text="turn 3",
        output_text="reply 3",
        parent_task_run_id=run_b.id,
    )

    dataset = DatasetSplit.from_task("Multiturn Split", task, AllSplitDefinition)

    assert dataset.split_contents["all"] == [leaf.id]


def test_build_split_contents_single_turn_includes_all_runs(tmp_path):
    task_path = tmp_path / "single_turn_task.kiln"
    task = Task(
        name="Single-turn Task",
        path=task_path,
        description="Single-turn dataset filter regression",
        instruction="Test instruction",
    )
    task.save_to_file()

    runs = [
        _make_task_run(task, input_text=f"input {i}", output_text=f"output {i}")
        for i in range(3)
    ]

    dataset = DatasetSplit.from_task("Single-turn Split", task, AllSplitDefinition)

    assert set(dataset.split_contents["all"]) == {run.id for run in runs}
