import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from typing import ClassVar, Dict
from unittest.mock import AsyncMock, patch

import litellm
import pytest

from kiln_ai.adapters.errors import KilnRunError
from kiln_ai.adapters.eval.base_eval import BaseEval, BaseV2EvalBridge
from kiln_ai.adapters.eval.conftest import SkippingStubV2Eval, StubV2Eval
from kiln_ai.adapters.eval.eval_runner import (
    EvalJob,
    EvalRunner,
    _conversation_usage,
    conversation_health_problem,
)
from kiln_ai.adapters.ml_model_list import ModelProviderName
from kiln_ai.adapters.retry_classification import is_retryable_error
from kiln_ai.datamodel import (
    DataSource,
    DataSourceType,
    EvalItemSource,
    Task,
    TaskOutput,
    TaskOutputRatingType,
    TaskRun,
)
from kiln_ai.datamodel.datamodel_enums import TurnMode
from kiln_ai.datamodel.eval import (
    Eval,
    EvalConfig,
    EvalConfigType,
    EvalDataType,
    EvalInput,
    EvalInputSplit,
    EvalOutputScore,
    EvalRun,
    EvalScores,
    EvalSplitName,
    EvalTaskInput,
    ExactMatchProperties,
    MultiTurnDriveConfig,
    MultiTurnSyntheticEvalInputData,
    SingleTurnEvalInputData,
    SkippedReason,
    SyntheticUserInfo,
    TaskRunSplit,
    UserMessage,
    V2EvalResult,
)
from kiln_ai.datamodel.eval_splits import ResolvedSplit, resolve_split
from kiln_ai.datamodel.run_config import (
    KilnAgentRunConfigProperties,
    McpRunConfigProperties,
    MCPToolReference,
)
from kiln_ai.datamodel.task import StructuredOutputMode, TaskRunConfig
from kiln_ai.datamodel.task_output import TASK_OUTPUT_SCHEMA_ERROR_PREFIX
from kiln_ai.datamodel.usage import MessageUsage, Usage
from kiln_ai.synthetic_user.drive_loop import DriveCaseResult
from kiln_ai.synthetic_user.models import TAG_SU_ENDED_CONVERSATION
from kiln_ai.utils.async_job_runner import RetryableError
from kiln_ai.utils.git_sync_protocols import default_save_context
from kiln_ai.utils.open_ai_types import ChatCompletionMessageParam


def build_task_run_eval_runner(
    eval_configs: list[EvalConfig],
    run_configs: list[TaskRunConfig],
    *,
    split_name: EvalSplitName = "test",
    **kwargs,
) -> EvalRunner:
    """A task_run_eval runner over the named split, resolved from disk as of now.

    The runner is handed items rather than a filter, so its item set is a snapshot taken
    here. A test that creates dataset items or changes a split after calling this must
    build a second runner — which is what a real caller would have to do too.
    """
    eval = eval_configs[0].parent_eval()
    assert eval is not None
    task = eval.parent_task()
    assert task is not None
    split = resolve_split(task, eval, split_name)
    assert split is not None, f"eval has no '{split_name}' split"
    return EvalRunner(
        eval_configs=eval_configs,
        run_configs=run_configs,
        eval_run_type="task_run_eval",
        split=split,
        **kwargs,
    )


def _test_split(eval_configs) -> ResolvedSplit:
    """Resolve the test split the old constructor derived implicitly. The legacy
    filter fields on these fixture evals fold into `splits` on load, so the
    resolution exercises the same shim production evals use."""
    ev = eval_configs[0].parent_eval()
    assert ev is not None
    task = ev.parent_task()
    assert task is not None
    split = resolve_split(task, ev, "test")
    assert split is not None, "fixture eval has no resolvable test split"
    return split


class TraceGenerator:
    """Stands in for the model call inside `BaseEval.run_task`.

    Reproduces the exact shape production returns, which is load-bearing twice over:

    - **`id=None`.** `run_task` builds its adapter with `allow_saving=False`, and every
      adapter clears the id of a run it did not persist (`base_adapter.py:346`). A double
      that kept its default-factory id would let the runner `save_to_file()` a run that
      real code cannot, hiding a failure on every single job.
    - **`run_config_id` set on the output source.** Half the trace key. A double that
      left it unset would be rejected by the trace index.
    - **A real `trace` and `usage`.** These are the fields the split exists to move onto
      the TaskRun, and the reuse path hands back a run *reloaded from disk*. Leaving them
      None on the double would make the round-trip assert nothing: `full_trace` evals
      read `trace.trace` through `EvalTaskInput.from_trace`, and Phase 4's rollup reads
      `usage`.
    """

    def __init__(self, task: Task, output: str = "fresh output"):
        self.task = task
        self.output = output
        self.trace: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "the prompt the model saw"},
            {"role": "assistant", "content": output},
        ]
        self.usage = Usage(input_tokens=42, output_tokens=7, cost=0.001)
        self.calls: list[tuple[TaskRun | EvalInput, str | None]] = []

    async def __call__(
        self, item: TaskRun | EvalInput, run_config_id: str | None = None
    ) -> TaskRun:
        self.calls.append((item, run_config_id))
        # Real generation is a network call. Yielding here is what lets two jobs on one
        # key interleave — without it they run to completion in turn, and an unlocked
        # index would look correct.
        await asyncio.sleep(0)
        run = TaskRun(
            parent=self.task,
            input=item.input
            if isinstance(item, TaskRun)
            else item.data.user_message.text,
            output=TaskOutput(
                output=self.output,
                source=DataSource(
                    type=DataSourceType.synthetic,
                    properties={
                        "model_name": "gpt-4",
                        "model_provider": "openai",
                        "adapter_name": "test_adapter",
                    },
                    run_config_id=run_config_id,
                ),
            ),
            trace=self.trace,
            usage=self.usage,
        )
        run.id = None
        return run


def eval_traces(task: Task) -> list[TaskRun]:
    """The task's eval-generated runs, which `task.runs()` hides by default."""
    return [
        run
        for run in task.runs(readonly=True, include_eval_generated=True)
        if run.eval_source is not None
    ]


def trace_for(task: Task, run_id: str | None) -> TaskRun:
    return next(run for run in eval_traces(task) if run.id == run_id)


@contextmanager
def generating(generator: TraceGenerator, evaluator_factory=StubV2Eval):
    """Score V2 jobs with `evaluator_factory`'s judge, and generate with `generator`.

    The judge is built per eval config, as the registry does, so a test with two judges
    exercises two adapters rather than sharing one.
    """
    with (
        patch.object(BaseV2EvalBridge, "run_task", new=generator),
        patch(
            "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
            side_effect=lambda config, *args, **kwargs: evaluator_factory(config),
        ),
    ):
        yield


@pytest.fixture
def mock_task(tmp_path):
    task = Task(
        name="test",
        description="test",
        instruction="do the thing",
        path=tmp_path / "task.kiln",
    )
    task.save_to_file()
    return task


@pytest.fixture
def mock_eval(mock_task):
    eval = Eval(
        id="test",
        name="test",
        description="test",
        eval_set_filter_id="all",
        eval_configs_filter_id="all",
        output_scores=[
            EvalOutputScore(
                name="Accuracy",
                instruction="Check if the output is accurate",
                type=TaskOutputRatingType.pass_fail,
            ),
        ],
        parent=mock_task,
    )
    eval.save_to_file()
    return eval


@pytest.fixture
def data_source():
    return DataSource(
        type=DataSourceType.synthetic,
        properties={
            "model_name": "gpt-4",
            "model_provider": "openai",
            "adapter_name": "test_adapter",
        },
    )


@pytest.fixture
def mock_eval_config(mock_eval):
    eval_config = EvalConfig(
        name="test",
        model_name="gpt-4",
        model_provider="openai",
        parent=mock_eval,
        properties={
            "eval_steps": ["step1", "step2", "step3"],
        },
    )
    eval_config.save_to_file()
    return eval_config


@pytest.fixture
def mock_run_config(
    mock_task,
):
    rc = TaskRunConfig(
        name="test",
        description="test",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=mock_task,
    )
    rc.save_to_file()
    return rc


@pytest.fixture
def mock_eval_runner(mock_eval, mock_task, mock_eval_config, mock_run_config):
    return EvalRunner(
        eval_configs=[mock_eval_config],
        run_configs=[mock_run_config],
        eval_run_type="task_run_eval",
        split=_test_split([mock_eval_config]),
    )


# Test with and without concurrency
@pytest.mark.parametrize("concurrency", [1, 25])
@pytest.mark.asyncio
async def test_async_eval_runner_status_updates(mock_eval_runner, concurrency):
    # Real async testing!

    job_count = 50
    # Job objects are not the right type, but since we're mocking run_job, it doesn't matter
    jobs = [{} for _ in range(job_count)]

    # Mock collect_tasks to return our fake jobs
    mock_eval_runner.collect_tasks = lambda: jobs

    # Mock run_job to return True immediately
    mock_eval_runner.run_job = AsyncMock(return_value=True)

    # Expect the status updates in order, and 1 for each job
    expected_completed_count = 0
    async for progress in mock_eval_runner.run(concurrency=concurrency):
        assert progress.complete == expected_completed_count
        expected_completed_count += 1
        assert progress.errors == 0
        assert progress.total == job_count

    # Verify last status update was complete
    assert expected_completed_count == job_count + 1

    # Verify run_job was called for each job
    assert mock_eval_runner.run_job.call_count == job_count


def test_collect_tasks_filtering(
    mock_eval,
    mock_task,
    mock_eval_config,
    data_source,
    mock_run_config,
):
    """Test that tasks are properly filtered based on eval filters"""
    tags = ["tag1", "tag2", "tag3"]
    task_runs = []
    for tag in tags:
        # Create some task runs with different tags
        task_run = TaskRun(
            parent=mock_task,
            input="test1",
            input_source=data_source,
            output=TaskOutput(
                output="test1",
            ),
            tags=[tag],
        )
        task_run.save_to_file()
        task_runs.append(task_run)

    mock_eval.splits["test"] = TaskRunSplit(filter_id="tag::tag1")
    mock_eval.eval_configs_filter_id = "tag::tag2"

    # Create a new runner of type task run eval
    runner = build_task_run_eval_runner([mock_eval_config], [mock_run_config])
    jobs = runner.collect_tasks()

    # Should only get task_run1 jobs, the one with tag1
    assert len(jobs) == 1
    job = jobs[0]
    # job should be the tag1 item, and setup as a task run eval for mock_run_config
    assert job.item.tags == ["tag1"]
    assert job.task_run_config is not None
    assert job.task_run_config.id == mock_run_config.id
    assert job.eval_config.id == mock_eval_config.id

    # Change to an eval config set filter
    runner = EvalRunner(
        eval_configs=[mock_eval_config],
        run_configs=None,
        eval_run_type="eval_config_eval",
    )
    jobs = runner.collect_tasks()

    # Should only get eval_config1 jobs
    assert len(jobs) == 1
    job = jobs[0]
    # job should be the tag2 item, and setup as a eval config eval for mock_eval_config
    assert job.item.tags == ["tag2"]
    assert job.eval_config.id == mock_eval_config.id
    assert job.task_run_config is None

    # Add a second task run config, and call a new runner with multiple run configs
    rc = TaskRunConfig(
        name="test2",
        description="test2",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=mock_task,
    )
    rc.save_to_file()
    runner = build_task_run_eval_runner([mock_eval_config], [mock_run_config, rc])
    jobs = runner.collect_tasks()
    assert len(jobs) == 2
    for job in jobs:
        assert job.item.tags == ["tag1"]
        assert job.task_run_config is not None
        assert job.task_run_config.id in [mock_run_config.id, rc.id]
        assert job.eval_config.id == mock_eval_config.id
    assert jobs[0].task_run_config is not None
    assert jobs[1].task_run_config is not None
    assert jobs[0].task_run_config.id != jobs[1].task_run_config.id

    # add a second eval config, and call a new runner with multiple eval configs
    eval_config = EvalConfig(
        name="test2",
        model_name="gpt-4",
        model_provider="openai",
        parent=mock_eval,
        properties={
            "eval_steps": ["step1", "step2", "step3"],
        },
    )
    eval_config.save_to_file()
    runner = EvalRunner(
        eval_configs=[mock_eval_config, eval_config],
        run_configs=None,
        eval_run_type="eval_config_eval",
    )
    jobs = runner.collect_tasks()
    # Check we get 2 jobs, one for each eval config
    assert len(jobs) == 2
    for job in jobs:
        assert job.item.tags == ["tag2"]
        assert job.eval_config.id in [mock_eval_config.id, eval_config.id]
        assert job.task_run_config is None
    assert jobs[0].eval_config.id != jobs[1].eval_config.id


def test_validate_same_task(
    mock_eval_runner,
    mock_task,
    data_source,
    tmp_path,
    mock_eval_config,
    mock_run_config,
):
    # second eval config has a different task
    eval_config = EvalConfig(
        name="test2",
        model_name="gpt-4",
        model_provider="openai",
        properties={
            "eval_steps": ["step1", "step2", "step3"],
        },
        parent=Eval(
            name="test",
            description="test",
            eval_set_filter_id="all",
            eval_configs_filter_id="all",
            output_scores=[
                EvalOutputScore(
                    name="Accuracy",
                    instruction="Check if the output is accurate",
                    type=TaskOutputRatingType.pass_fail,
                ),
            ],
            parent=Task(
                name="test",
                description="test",
                instruction="do the thing",
            ),
        ),
    )

    with pytest.raises(
        ValueError, match="All eval configs must have the same parent eval"
    ):
        EvalRunner(
            eval_configs=[mock_eval_config, eval_config],
            run_configs=[mock_run_config],
            eval_run_type="eval_config_eval",
        )


def test_collect_tasks_excludes_already_run_task_run_eval(
    mock_eval, mock_task, data_source, mock_eval_config, mock_run_config
):
    """Test that already run tasks are excluded"""
    # Narrow the test split to the tag the runs carry, so "no jobs" below can only mean
    # "already run" — with a filter that matched nothing it would be zero either way.
    mock_eval.splits["test"] = TaskRunSplit(filter_id="tag::tag1")
    mock_eval.eval_configs_filter_id = "tag::nonexistent"

    # Create a task run
    task_run = TaskRun(
        parent=mock_task,
        input="test",
        input_source=data_source,
        tags=["tag1"],
        output=TaskOutput(
            output="test",
        ),
    )
    task_run.save_to_file()

    # Prior to any eval runs, we should get the task run
    jobs = build_task_run_eval_runner(
        [mock_eval_config], [mock_run_config]
    ).collect_tasks()
    assert len(jobs) == 1
    assert jobs[0].item.id == task_run.id
    assert jobs[0].task_run_config.id == mock_run_config.id
    assert jobs[0].eval_config.id == mock_eval_config.id

    # Create an eval run for this task
    EvalRun(
        parent=mock_eval_config,
        dataset_id=task_run.id,
        task_run_config_id=mock_run_config.id,
        input="test",
        output="test",
        scores={"accuracy": 1.0},
    ).save_to_file()

    jobs = build_task_run_eval_runner(
        [mock_eval_config], [mock_run_config]
    ).collect_tasks()

    # Should get no jobs since the task was already run
    assert len(jobs) == 0

    # Same split, a second item that has not been run: the split is really being read.
    second_run = TaskRun(
        parent=mock_task,
        input="test2",
        input_source=data_source,
        tags=["tag1"],
        output=TaskOutput(output="test2"),
    )
    second_run.save_to_file()
    jobs = build_task_run_eval_runner(
        [mock_eval_config], [mock_run_config]
    ).collect_tasks()
    assert [job.item.id for job in jobs] == [second_run.id]


def test_collect_tasks_excludes_already_run_eval_config_eval(
    mock_task, data_source, mock_eval_config, mock_eval, mock_run_config
):
    """Test that already run tasks are excluded"""
    # Create a task run
    task_run = TaskRun(
        parent=mock_task,
        input="test",
        input_source=data_source,
        tags=["tag1"],
        output=TaskOutput(
            output="test",
        ),
    )
    task_run.save_to_file()

    mock_eval.eval_set_filter_id = "tag::nonexistent"
    mock_eval.eval_configs_filter_id = "tag::tag1"
    mock_eval.save_to_file()

    # Prior to any eval runs, we should get 1 job for the eval config
    runner = EvalRunner(
        eval_configs=[mock_eval_config],
        run_configs=None,
        eval_run_type="eval_config_eval",
    )
    jobs = runner.collect_tasks()
    assert len(jobs) == 1
    assert jobs[0].item.id == task_run.id
    assert jobs[0].eval_config.id == mock_eval_config.id
    assert jobs[0].task_run_config is None

    # Create an eval run for this eval config task run pair, so now we should get no jobs (already run)
    EvalRun(
        parent=mock_eval_config,
        dataset_id=task_run.id,
        task_run_config_id=None,
        eval_config_eval=True,
        input="test",
        output="test",
        scores={
            "accuracy": 1.0,
        },
    ).save_to_file()

    jobs = runner.collect_tasks()

    # Should get no jobs since the task was already run
    assert len(jobs) == 0


def test_golden_item_scored_as_test_item_still_calibrates(
    mock_task, data_source, mock_eval_config, mock_eval, mock_run_config
):
    """Only calibration records mark a golden item done. The same eval config
    accumulates task_run_eval records too — a golden TaskRun that was scored as
    a test item (or tombstoned in the test lane) must still be calibrated, and
    the test lane's tombstones must not ride (or be deleted by) calibration."""
    golden_run = TaskRun(
        parent=mock_task,
        input="test",
        input_source=data_source,
        tags=["tag1"],
        output=TaskOutput(output="test"),
    )
    golden_run.save_to_file()

    mock_eval.eval_set_filter_id = "tag::nonexistent"
    mock_eval.eval_configs_filter_id = "tag::tag1"
    mock_eval.save_to_file()

    # Test-lane records on the same golden item: a real score and a tombstone
    # (terminal in the calibration runner, which has no split to recover
    # against). Neither is a calibration record.
    EvalRun(
        parent=mock_eval_config,
        dataset_id=golden_run.id,
        task_run_config_id=mock_run_config.id,
        eval_config_eval=False,
        input="test",
        output="test",
        scores={"accuracy": 1.0},
    ).save_to_file()
    EvalRun(
        parent=mock_eval_config,
        dataset_id=golden_run.id,
        task_run_config_id=mock_run_config.id,
        eval_config_eval=False,
        input="test",
        output=None,
        scores={},
        skipped_reason=SkippedReason.missing_drive_config.value,
        skipped_detail="test tombstone",
    ).save_to_file()

    runner = EvalRunner(
        eval_configs=[mock_eval_config],
        run_configs=None,
        eval_run_type="eval_config_eval",
    )
    jobs = runner.collect_tasks()
    assert [job.item.id for job in jobs] == [golden_run.id]
    assert jobs[0].superseded_tombstones == []

    # A real calibration record does mark it done.
    EvalRun(
        parent=mock_eval_config,
        dataset_id=golden_run.id,
        task_run_config_id=None,
        eval_config_eval=True,
        input="test",
        output="test",
        scores={"accuracy": 1.0},
    ).save_to_file()
    assert runner.collect_tasks() == []


def test_collect_tasks_multiple_run_configs(
    mock_eval, mock_eval_config, mock_task, data_source, mock_run_config
):
    """Test handling multiple run configs"""
    # Create a task run
    task_run = TaskRun(
        parent=mock_task,
        input="test",
        input_source=data_source,
        tags=["tag1"],
        output=TaskOutput(
            output="test",
        ),
    )
    task_run.save_to_file()

    # Add another run config
    second_config = TaskRunConfig(
        name="test2",
        description="test2",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-3.5",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=mock_task,
    )
    second_config.save_to_file()

    # Set filter to match the task
    mock_eval.splits["test"] = TaskRunSplit(filter_id="tag::tag1")

    jobs = build_task_run_eval_runner(
        [mock_eval_config], [mock_run_config, second_config]
    ).collect_tasks()

    # Should get 2 jobs, one for each config
    assert len(jobs) == 2
    assert {job.task_run_config.id for job in jobs} == {
        second_config.id,
        mock_run_config.id,
    }


def test_collect_tasks_empty_cases(mock_eval_runner, mock_task, data_source):
    """Test empty cases - no matching tasks or no tasks at all"""
    # Set filter that won't match anything
    mock_eval_runner.eval.eval_set_filter_id = "tag::nonexistent"
    mock_eval_runner.eval.eval_configs_filter_id = "tag::nonexistent"

    jobs = mock_eval_runner.collect_tasks()
    assert len(jobs) == 0

    # Create task run with non-matching tag
    task_run = TaskRun(
        parent=mock_task,
        input="test",
        input_source=data_source,
        tags=["other_tag"],
        output=TaskOutput(
            output="test",
        ),
    )
    task_run.save_to_file()

    jobs = mock_eval_runner.collect_tasks()
    assert len(jobs) == 0


@pytest.mark.asyncio
async def test_run_job_success_task_run_eval(
    mock_eval_runner, mock_task, data_source, mock_run_config, mock_eval_config
):
    # Create a task run to evaluate
    task_run = TaskRun(
        parent=mock_task,
        input="test input",
        input_source=data_source,
        output=TaskOutput(output="test output"),
    )
    task_run.save_to_file()

    # Create eval job
    job = EvalJob(
        item=task_run,
        task_run_config=mock_run_config,
        type="task_run_eval",
        eval_config=mock_eval_config,
    )

    # Mock the evaluator
    mock_scores = {"accuracy": 0.95}

    class MockEvaluator(BaseEval):
        async def run_task_and_eval(self, eval_job_item: TaskRun):
            return (
                TaskRun(
                    input=eval_job_item.input,
                    input_source=data_source,
                    output=TaskOutput(output="evaluated output"),
                    intermediate_outputs={"intermediate_output": "intermediate output"},
                ),
                mock_scores,
                {"intermediate_output": "intermediate output"},
            )

    with patch(
        "kiln_ai.adapters.eval.eval_runner.legacy_eval_adapter_from_type",
        return_value=lambda *args, **kwargs: MockEvaluator(*args, **kwargs),
    ):
        success = await mock_eval_runner.run_job(job)

    assert success is True

    # Verify eval run was saved
    eval_runs = mock_eval_config.runs()
    assert len(eval_runs) == 1
    saved_run = eval_runs[0]
    assert saved_run.dataset_id == task_run.id
    assert saved_run.task_run_config_id == mock_run_config.id
    assert saved_run.scores == mock_scores
    assert saved_run.input == "test input"
    assert saved_run.output == "evaluated output"
    assert saved_run.intermediate_outputs == {
        "intermediate_output": "intermediate output"
    }
    assert saved_run.parent_eval_config().id == mock_eval_config.id
    assert saved_run.eval_config_eval is False


@pytest.mark.asyncio
async def test_run_job_success_eval_config_eval(
    mock_eval_runner, mock_task, data_source, mock_run_config, mock_eval_config
):
    # Create a task run to evaluate
    task_run = TaskRun(
        parent=mock_task,
        input="test input",
        input_source=data_source,
        output=TaskOutput(output="test output"),
    )
    task_run.save_to_file()

    # Create eval job
    job = EvalJob(
        item=task_run,
        type="eval_config_eval",
        eval_config=mock_eval_config,
    )

    # Mock the evaluator
    mock_scores: EvalScores = {"accuracy": 0.95}

    class MockEvaluator(BaseEval):
        async def run_task_and_eval(self, eval_job_item: TaskRun):
            raise ValueError("Attempted to run task and eval for a config eval")

        async def run_eval(
            self, task_run: TaskRun, eval_job_item: TaskRun | None = None
        ) -> tuple[EvalScores, Dict[str, str] | None]:
            return mock_scores, {"intermediate_output": "intermediate output"}

    with patch(
        "kiln_ai.adapters.eval.eval_runner.legacy_eval_adapter_from_type",
        return_value=lambda *args, **kwargs: MockEvaluator(*args, **kwargs),
    ):
        success = await mock_eval_runner.run_job(job)

    assert success is True

    # Verify eval run was saved
    eval_runs = mock_eval_config.runs()
    assert len(eval_runs) == 1
    saved_run = eval_runs[0]
    assert saved_run.dataset_id == task_run.id
    assert saved_run.task_run_config_id is None
    assert saved_run.scores == mock_scores
    assert saved_run.input == "test input"
    assert saved_run.output == "test output"
    assert saved_run.parent_eval_config().id == mock_eval_config.id
    assert saved_run.eval_config_eval is True


@pytest.mark.asyncio
async def test_run_job_invalid_evaluator(
    mock_eval_runner, mock_task, data_source, mock_run_config, mock_eval_config
):
    task_run = TaskRun(
        parent=mock_task,
        input="test input",
        input_source=data_source,
        output=TaskOutput(output="test output"),
    )
    task_run.save_to_file()
    job = EvalJob(
        item=task_run,
        task_run_config=mock_run_config,
        type="task_run_eval",
        eval_config=mock_eval_config,
    )

    # Return an invalid evaluator type
    with patch(
        "kiln_ai.adapters.eval.eval_runner.legacy_eval_adapter_from_type",
        return_value=lambda *args, **kwargs: object(),
    ):
        with pytest.raises(ValueError):
            await mock_eval_runner.run_job(job)

    assert len(mock_eval_config.runs()) == 0


@pytest.mark.asyncio
async def test_run_job_evaluator_error(
    mock_eval_runner, mock_task, data_source, mock_run_config, mock_eval_config
):
    task_run = TaskRun(
        parent=mock_task,
        input="test input",
        input_source=data_source,
        output=TaskOutput(output="test output"),
    )
    task_run.save_to_file()
    job = EvalJob(
        item=task_run,
        task_run_config=mock_run_config,
        type="task_run_eval",
        eval_config=mock_eval_config,
    )

    class ErrorEvaluator(BaseEval):
        async def run_task_and_eval(self, eval_job_item: TaskRun):
            raise ValueError("Evaluation failed")

    with patch(
        "kiln_ai.adapters.eval.eval_runner.legacy_eval_adapter_from_type",
        return_value=lambda *args, **kwargs: ErrorEvaluator(*args, **kwargs),
    ):
        with pytest.raises(ValueError):
            await mock_eval_runner.run_job(job)

    assert len(mock_eval_config.runs()) == 0


@pytest.mark.asyncio
async def test_run_job_with_full_trace_evaluation_data_type(
    mock_eval_runner, mock_task, data_source, mock_run_config, mock_eval_config
):
    """Test EvalRunner with full_trace evaluation_data_type"""
    # Set the eval config to use full_trace evaluation data type
    mock_eval_config.parent.evaluation_data_type = EvalDataType.full_trace
    # Persist the change so validation on reload sees full_trace
    mock_eval_config.parent.save_to_file()

    # Create a task run to evaluate
    task_run = TaskRun(
        parent=mock_task,
        input="test input",
        input_source=data_source,
        output=TaskOutput(output="test output"),
    )
    task_run.save_to_file()

    # Create eval job
    job = EvalJob(
        item=task_run,
        task_run_config=mock_run_config,
        type="task_run_eval",
        eval_config=mock_eval_config,
    )

    # Mock the evaluator
    mock_scores = {"accuracy": 0.95}
    mock_trace: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "test input"},
        {"role": "assistant", "content": "test response"},
    ]

    class MockEvaluator(BaseEval):
        async def run_task_and_eval(self, eval_job_item: TaskRun):
            result_task_run = TaskRun(
                input=eval_job_item.input,
                input_source=data_source,
                output=TaskOutput(output="evaluated output"),
                intermediate_outputs={"intermediate_output": "intermediate output"},
                trace=mock_trace,
            )
            return (
                result_task_run,
                mock_scores,
                {"intermediate_output": "intermediate output"},
            )

    with patch(
        "kiln_ai.adapters.eval.eval_runner.legacy_eval_adapter_from_type",
        return_value=lambda *args, **kwargs: MockEvaluator(*args, **kwargs),
    ):
        success = await mock_eval_runner.run_job(job)

    assert success is True

    # Verify eval run was saved with trace
    eval_runs = mock_eval_config.runs()
    assert len(eval_runs) == 1
    saved_run = eval_runs[0]
    assert saved_run.task_run_trace is not None
    assert isinstance(saved_run.task_run_trace, str)
    # Verify the trace was JSON serialized
    parsed_trace = json.loads(saved_run.task_run_trace)
    assert parsed_trace == mock_trace


@pytest.mark.asyncio
async def test_run_job_full_trace_serializes_per_message_usage(
    mock_eval_runner, mock_task, data_source, mock_run_config, mock_eval_config
):
    """Regression: the V1 runner serialized the trace with a plain `json.dumps`.

    A real trace types its per-message `usage` as a `MessageUsage` model, which that
    encoder cannot handle - so the job died before it ever saved an EvalRun.
    """
    mock_eval_config.parent.evaluation_data_type = EvalDataType.full_trace
    mock_eval_config.parent.save_to_file()

    task_run = TaskRun(
        parent=mock_task,
        input="test input",
        input_source=data_source,
        output=TaskOutput(output="test output"),
    )
    task_run.save_to_file()

    job = EvalJob(
        item=task_run,
        task_run_config=mock_run_config,
        type="task_run_eval",
        eval_config=mock_eval_config,
    )

    class MockEvaluator(BaseEval):
        async def run_task_and_eval(self, eval_job_item: TaskRun):
            result_task_run = TaskRun(
                input=eval_job_item.input,
                input_source=data_source,
                output=TaskOutput(output="evaluated output"),
                trace=[
                    {"role": "user", "content": "test input"},
                    {
                        "role": "assistant",
                        "content": "test response",
                        "usage": MessageUsage(
                            input_tokens=42, output_tokens=7, cost=0.001
                        ),
                    },
                ],
            )
            return result_task_run, {"accuracy": 0.95}, {}

    with patch(
        "kiln_ai.adapters.eval.eval_runner.legacy_eval_adapter_from_type",
        return_value=lambda *args, **kwargs: MockEvaluator(*args, **kwargs),
    ):
        assert await mock_eval_runner.run_job(job) is True

    saved_run = mock_eval_config.runs()[0]
    assert saved_run.task_run_trace is not None
    assert json.loads(saved_run.task_run_trace)[1]["usage"] == {
        "input_tokens": 42,
        "output_tokens": 7,
        "total_tokens": None,
        "cost": 0.001,
        "cached_tokens": None,
    }


@pytest.mark.asyncio
async def test_run_job_with_final_answer_evaluation_data_type(
    mock_eval_runner, mock_task, data_source, mock_run_config, mock_eval_config
):
    """Test EvalRunner with final_answer evaluation_data_type (default)"""
    # Set the eval config to use final_answer evaluation data type (default)
    mock_eval_config.parent.evaluation_data_type = EvalDataType.final_answer

    # Create a task run to evaluate
    task_run = TaskRun(
        parent=mock_task,
        input="test input",
        input_source=data_source,
        output=TaskOutput(output="test output"),
    )
    task_run.save_to_file()

    # Create eval job
    job = EvalJob(
        item=task_run,
        task_run_config=mock_run_config,
        type="task_run_eval",
        eval_config=mock_eval_config,
    )

    # Mock the evaluator
    mock_scores = {"accuracy": 0.95}
    mock_trace: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "response"},
    ]

    class MockEvaluator(BaseEval):
        async def run_task_and_eval(self, eval_job_item: TaskRun):
            result_task_run = TaskRun(
                input=eval_job_item.input,
                input_source=data_source,
                output=TaskOutput(output="evaluated output"),
                intermediate_outputs={"intermediate_output": "intermediate output"},
                trace=mock_trace,
            )
            return (
                result_task_run,
                mock_scores,
                {"intermediate_output": "intermediate output"},
            )

    with patch(
        "kiln_ai.adapters.eval.eval_runner.legacy_eval_adapter_from_type",
        return_value=lambda *args, **kwargs: MockEvaluator(*args, **kwargs),
    ):
        success = await mock_eval_runner.run_job(job)

    assert success is True

    # Verify eval run was saved without trace
    eval_runs = mock_eval_config.runs()
    assert len(eval_runs) == 1
    saved_run = eval_runs[0]
    assert saved_run.task_run_trace is None


@pytest.mark.asyncio
async def test_run_job_with_none_trace(
    mock_eval_runner, mock_task, data_source, mock_run_config, mock_eval_config
):
    """Test EvalRunner with None trace"""
    # Set the eval config to use full_trace evaluation data type
    mock_eval_config.parent.evaluation_data_type = EvalDataType.full_trace

    # Create a task run to evaluate
    task_run = TaskRun(
        parent=mock_task,
        input="test input",
        input_source=data_source,
        output=TaskOutput(output="test output"),
    )
    task_run.save_to_file()

    # Create eval job
    job = EvalJob(
        item=task_run,
        task_run_config=mock_run_config,
        type="task_run_eval",
        eval_config=mock_eval_config,
    )

    # Mock the evaluator
    mock_scores = {"accuracy": 0.95}

    class MockEvaluator(BaseEval):
        async def run_task_and_eval(self, eval_job_item: TaskRun):
            result_task_run = TaskRun(
                input=eval_job_item.input,
                input_source=data_source,
                output=TaskOutput(output="evaluated output"),
                intermediate_outputs={"intermediate_output": "intermediate output"},
                trace=None,  # None trace
            )
            return (
                result_task_run,
                mock_scores,
                {"intermediate_output": "intermediate output"},
            )

    with patch(
        "kiln_ai.adapters.eval.eval_runner.legacy_eval_adapter_from_type",
        return_value=lambda *args, **kwargs: MockEvaluator(*args, **kwargs),
    ):
        with pytest.raises(ValueError):
            await mock_eval_runner.run_job(job)

    # For full_trace evals, None trace should fail and not save a run
    eval_runs = mock_eval_config.runs()
    assert len(eval_runs) == 0


@pytest.mark.parametrize(
    "error",
    [
        litellm.RateLimitError("rate limited", "provider", "model", None),
        litellm.APIConnectionError("connection failed", "provider", "model", None),
        # Timeout takes (message, model, llm_provider) and descends from
        # openai's connection error, so APIConnectionError above doesn't cover it.
        litellm.Timeout("timed out", "model", "provider"),
        litellm.InternalServerError("server error", "provider", "model", None),
        litellm.ServiceUnavailableError("unavailable", "provider", "model", None),
        litellm.BadGatewayError("bad gateway", "provider", "model", None),
        litellm.JSONSchemaValidationError("schema error", "provider", "model", None),
        ValueError(
            f"{TASK_OUTPUT_SCHEMA_ERROR_PREFIX} The error from the schema check was: ..."
        ),
    ],
)
def test_is_retryable_error_returns_true(error):
    assert is_retryable_error(error) is True


@pytest.mark.parametrize(
    "error",
    [
        ValueError("some other value error"),
        RuntimeError("runtime error"),
        KeyError("missing key"),
        TypeError("type error"),
    ],
)
def test_is_retryable_error_returns_false(error):
    assert is_retryable_error(error) is False


def wrapped_rate_limit_error(detail: str) -> KilnRunError:
    """A provider rate limit as the model adapter surfaces it: wrapped in
    KilnRunError whose own message is the genericized user-facing text, with
    the provider detail only on the inner error."""
    return KilnRunError(
        message="Rate limit exceeded. Wait a moment and try again.",
        partial_trace=None,
        original=litellm.RateLimitError(detail, "provider", "model", None),
    )


def test_is_retryable_error_unwraps_kiln_run_error():
    # The model adapter wraps provider exceptions in KilnRunError (to carry the
    # partial trace), so the classifier must look through the wrapper — otherwise
    # rate limits from a real adapter run would never be retried.
    assert is_retryable_error(wrapped_rate_limit_error("rate limited")) is True


def test_is_retryable_error_wrapped_non_transient_returns_false():
    wrapped = KilnRunError(
        message="An unexpected error occurred.",
        partial_trace=None,
        original=RuntimeError("boom"),
    )
    assert is_retryable_error(wrapped) is False


@pytest.mark.asyncio
async def test_run_job_wrapped_rate_limit_raises_retryable_with_detail(
    mock_eval_runner, mock_task, data_source, mock_run_config, mock_eval_config
):
    # Real adapter failures arrive wrapped in KilnRunError whose own message is the
    # genericized user-facing text. run_job must still classify the failure as
    # transient (RetryableError) and keep the underlying provider message for the
    # developer-facing logs.
    task_run = TaskRun(
        parent=mock_task,
        input="test input",
        input_source=data_source,
        output=TaskOutput(output="test output"),
    )
    task_run.save_to_file()
    job = EvalJob(
        item=task_run,
        task_run_config=mock_run_config,
        type="task_run_eval",
        eval_config=mock_eval_config,
    )

    class RateLimitedEvaluator(BaseEval):
        async def run_task_and_eval(self, eval_job_item: TaskRun):
            raise wrapped_rate_limit_error(
                "rate limit exceeded, please try again later"
            )

    with patch(
        "kiln_ai.adapters.eval.eval_runner.legacy_eval_adapter_from_type",
        return_value=lambda *args, **kwargs: RateLimitedEvaluator(*args, **kwargs),
    ):
        with pytest.raises(RetryableError) as exc_info:
            await mock_eval_runner.run_job(job)

    assert "rate limit exceeded, please try again later" in str(exc_info.value)
    assert "An unexpected error occurred" not in str(exc_info.value)
    assert len(mock_eval_config.runs()) == 0


def test_is_retryable_error_unwraps_nested_kiln_run_error():
    # Not produced by the current adapter chain (it passes through already-wrapped
    # errors), but the unwrap walks nested wrappers so classification can't
    # silently break if that ever changes.
    nested = KilnRunError(
        message="Rate limit exceeded. Wait a moment and try again.",
        partial_trace=None,
        original=wrapped_rate_limit_error("rate limited"),
    )
    assert is_retryable_error(nested) is True


# --- save_context tests ---


class _RecordingSaveContext:
    def __init__(self):
        self.enter_count = 0
        self.exit_count = 0
        self.last_exit_exc_type: type | None = None

    def __call__(self):
        @asynccontextmanager
        async def cm() -> AsyncIterator[None]:
            self.enter_count += 1
            try:
                yield
            except BaseException as exc:
                self.last_exit_exc_type = type(exc)
                self.exit_count += 1
                raise
            else:
                self.exit_count += 1

        return cm()


def test_eval_runner_defaults_to_default_save_context(
    mock_eval, mock_eval_config, mock_run_config
):
    runner = EvalRunner(
        eval_configs=[mock_eval_config],
        run_configs=[mock_run_config],
        eval_run_type="task_run_eval",
        split=_test_split([mock_eval_config]),
    )
    assert runner._save_context is default_save_context


def test_eval_runner_accepts_custom_save_context(
    mock_eval, mock_eval_config, mock_run_config
):
    recorder = _RecordingSaveContext()
    runner = EvalRunner(
        eval_configs=[mock_eval_config],
        run_configs=[mock_run_config],
        eval_run_type="task_run_eval",
        save_context=recorder,
        split=_test_split([mock_eval_config]),
    )
    assert runner._save_context is recorder


@pytest.mark.asyncio
async def test_run_job_custom_save_context_wraps_save(
    mock_task, data_source, mock_eval_config, mock_run_config
):
    recorder = _RecordingSaveContext()
    runner = EvalRunner(
        eval_configs=[mock_eval_config],
        run_configs=[mock_run_config],
        eval_run_type="task_run_eval",
        save_context=recorder,
        split=_test_split([mock_eval_config]),
    )

    task_run = TaskRun(
        parent=mock_task,
        input="test input",
        input_source=data_source,
        output=TaskOutput(output="test output"),
    )
    task_run.save_to_file()

    job = EvalJob(
        item=task_run,
        task_run_config=mock_run_config,
        type="task_run_eval",
        eval_config=mock_eval_config,
    )

    class MockEvaluator(BaseEval):
        async def run_task_and_eval(self, eval_job_item):
            return (
                TaskRun(
                    input=eval_job_item.input,
                    input_source=data_source,
                    output=TaskOutput(output="evaluated output"),
                ),
                {"accuracy": 1.0},
                {},
            )

    with patch(
        "kiln_ai.adapters.eval.eval_runner.legacy_eval_adapter_from_type",
        return_value=lambda *args, **kwargs: MockEvaluator(*args, **kwargs),
    ):
        success = await runner.run_job(job)

    assert success is True
    assert recorder.enter_count == 1
    assert recorder.exit_count == 1
    assert recorder.last_exit_exc_type is None


@pytest.mark.asyncio
async def test_run_job_save_context_sees_save_exception(
    mock_task, data_source, mock_eval_config, mock_run_config
):
    recorder = _RecordingSaveContext()
    runner = EvalRunner(
        eval_configs=[mock_eval_config],
        run_configs=[mock_run_config],
        eval_run_type="task_run_eval",
        save_context=recorder,
        split=_test_split([mock_eval_config]),
    )

    task_run = TaskRun(
        parent=mock_task,
        input="test input",
        input_source=data_source,
        output=TaskOutput(output="test output"),
    )
    task_run.save_to_file()

    job = EvalJob(
        item=task_run,
        task_run_config=mock_run_config,
        type="task_run_eval",
        eval_config=mock_eval_config,
    )

    class MockEvaluator(BaseEval):
        async def run_task_and_eval(self, eval_job_item):
            return (
                TaskRun(
                    input=eval_job_item.input,
                    input_source=data_source,
                    output=TaskOutput(output="evaluated output"),
                ),
                {"accuracy": 1.0},
                {},
            )

    with (
        patch(
            "kiln_ai.adapters.eval.eval_runner.legacy_eval_adapter_from_type",
            return_value=lambda *args, **kwargs: MockEvaluator(*args, **kwargs),
        ),
        patch.object(EvalRun, "save_to_file", side_effect=RuntimeError("disk full")),
    ):
        with pytest.raises(RuntimeError, match="disk full"):
            await runner.run_job(job)

    assert recorder.enter_count == 1
    assert recorder.exit_count == 1
    assert recorder.last_exit_exc_type is RuntimeError


@pytest.mark.asyncio
async def test_other_jobs_unaffected_by_save_context_rollback(
    mock_task, data_source, mock_eval_config, mock_run_config
):
    recorder = _RecordingSaveContext()
    runner = EvalRunner(
        eval_configs=[mock_eval_config],
        run_configs=[mock_run_config],
        eval_run_type="task_run_eval",
        save_context=recorder,
        split=_test_split([mock_eval_config]),
    )

    task_run = TaskRun(
        parent=mock_task,
        input="test input",
        input_source=data_source,
        output=TaskOutput(output="test output"),
    )
    task_run.save_to_file()

    def make_job():
        return EvalJob(
            item=task_run,
            task_run_config=mock_run_config,
            type="task_run_eval",
            eval_config=mock_eval_config,
        )

    class MockEvaluator(BaseEval):
        async def run_task_and_eval(self, eval_job_item):
            return (
                TaskRun(
                    input=eval_job_item.input,
                    input_source=data_source,
                    output=TaskOutput(output="evaluated output"),
                ),
                {"accuracy": 1.0},
                {},
            )

    call_count = {"n": 0}
    real_save_to_file = EvalRun.save_to_file

    def fail_first_save(self, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("disk full")
        return real_save_to_file(self, *args, **kwargs)

    with (
        patch(
            "kiln_ai.adapters.eval.eval_runner.legacy_eval_adapter_from_type",
            return_value=lambda *args, **kwargs: MockEvaluator(*args, **kwargs),
        ),
        patch.object(EvalRun, "save_to_file", fail_first_save),
    ):
        with pytest.raises(RuntimeError, match="disk full"):
            await runner.run_job(make_job())
        success = await runner.run_job(make_job())

    assert success is True
    # Two fresh contexts were opened and both closed; the second job's success
    # proves rollback from the first did not leak into the second's context.
    assert recorder.enter_count == 2
    assert recorder.exit_count == 2


# ===================================================================
# V2 Eval Runner Tests
# ===================================================================


@pytest.fixture
def mock_v2_eval(mock_task):
    """Eval with an EvalInput-backed test split."""
    eval = Eval(
        id="v2_eval",
        name="v2 test eval",
        description="v2 eval desc",
        splits={"test": EvalInputSplit(filter_id="all")},
        eval_configs_filter_id="all",
        output_scores=[
            EvalOutputScore(
                name="Accuracy",
                instruction="Check if the output is accurate",
                type=TaskOutputRatingType.pass_fail,
            ),
        ],
        parent=mock_task,
    )
    eval.save_to_file()
    return eval


@pytest.fixture
def mock_v2_eval_config(mock_v2_eval):
    """V2 EvalConfig with ExactMatchProperties."""
    eval_config = EvalConfig(
        name="v2 config",
        config_type=EvalConfigType.v2,
        properties=ExactMatchProperties(
            expected_value="hello",
        ),
        parent=mock_v2_eval,
    )
    eval_config.save_to_file()
    return eval_config


@pytest.fixture
def mock_eval_inputs(mock_task):
    """Create two EvalInput items under the task."""
    input1 = EvalInput(
        id="ei_1",
        data=SingleTurnEvalInputData(
            user_message=UserMessage(text="What is 2+2?"),
        ),
        reference={"answer": "4"},
        tags=["math"],
        parent=mock_task,
    )
    input1.save_to_file()
    input2 = EvalInput(
        id="ei_2",
        data=SingleTurnEvalInputData(
            user_message=UserMessage(text="Say hello"),
        ),
        reference={"answer": "hello"},
        tags=["greeting"],
        parent=mock_task,
    )
    input2.save_to_file()
    return [input1, input2]


@pytest.fixture
def mock_v2_runner(mock_v2_eval, mock_v2_eval_config):
    return EvalRunner(
        eval_configs=[mock_v2_eval_config],
        run_configs=None,
        eval_run_type="eval_config_eval",
    )


# -------------------------------------------------------------------
# Init / source mode tests
# -------------------------------------------------------------------
class TestEvalRunnerV2Init:
    def test_collects_all_inputs(
        self, mock_v2_eval_config, mock_run_config, mock_eval_inputs
    ):
        runner = EvalRunner(
            eval_configs=[mock_v2_eval_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_eval_config]),
        )
        jobs = runner.collect_tasks()
        assert len(jobs) == 2
        item_ids = {j.item.id for j in jobs}
        assert item_ids == {"ei_1", "ei_2"}
        for job in jobs:
            assert isinstance(job.item, EvalInput)
            assert job.type == "task_run_eval"
            assert job.task_run_config is mock_run_config

    def test_tag_filter(self, mock_task, mock_run_config, mock_eval_inputs):
        eval = Eval(
            id="tag_eval",
            name="tag eval",
            description="tag eval desc",
            splits={"test": EvalInputSplit(filter_id="tag::math")},
            eval_configs_filter_id="all",
            output_scores=[
                EvalOutputScore(
                    name="Accuracy",
                    instruction="Check",
                    type=TaskOutputRatingType.pass_fail,
                ),
            ],
            parent=mock_task,
        )
        eval.save_to_file()
        eval_config = EvalConfig(
            name="tag config",
            config_type=EvalConfigType.v2,
            properties=ExactMatchProperties(expected_value="4"),
            parent=eval,
        )
        eval_config.save_to_file()
        runner = EvalRunner(
            eval_configs=[eval_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([eval_config]),
        )
        jobs = runner.collect_tasks()
        assert len(jobs) == 1
        assert jobs[0].item.id == "ei_1"

    def test_dedup_already_run(
        self, mock_v2_eval_config, mock_run_config, mock_eval_inputs
    ):
        run = EvalRun(
            parent=mock_v2_eval_config,
            eval_input_id="ei_1",
            task_run_config_id=mock_run_config.id,
            eval_config_eval=False,
            scores={"accuracy": 1.0},
            input="What is 2+2?",
            output="4",
        )
        run.save_to_file()
        runner = EvalRunner(
            eval_configs=[mock_v2_eval_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_eval_config]),
        )
        jobs = runner.collect_tasks()
        assert len(jobs) == 1
        assert jobs[0].item.id == "ei_2"

    def test_eval_config_eval_collects_golden_task_runs(
        self, mock_v2_runner, mock_task, mock_eval_inputs, data_source
    ):
        """eval_config_eval on an EvalInput-sourced eval must still collect
        golden TASKRUNS (via eval_configs_filter_id) — judge validation needs
        stored, human-rated outputs, which EvalInput items don't carry."""
        golden_run = TaskRun(
            input="golden input",
            output=TaskOutput(output="golden output", source=data_source),
            parent=mock_task,
        )
        golden_run.save_to_file()
        jobs = mock_v2_runner.collect_tasks()
        assert len(jobs) == 1
        assert isinstance(jobs[0].item, TaskRun)
        assert jobs[0].item.id == golden_run.id
        assert jobs[0].type == "eval_config_eval"


# -------------------------------------------------------------------
# run_job V2 dispatch tests
# -------------------------------------------------------------------
# Three user turns, ending on an assistant reply: the shape a complete drive of the
# `multi_turn_eval_input` fixture produces, so a fake drive returning it passes the
# runner's completeness check the way a real one would.
MULTI_TURN_TRACE: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "turn 1"},
    {"role": "assistant", "content": "hi"},
    {"role": "user", "content": "turn 2"},
    {"role": "assistant", "content": "go on"},
    {"role": "user", "content": "turn 3"},
    {"role": "assistant", "content": "reply"},
]


def make_multi_turn_leaf(
    task: Task,
    data_source: DataSource,
    trace: list[ChatCompletionMessageParam] | None = MULTI_TURN_TRACE,
) -> TaskRun:
    """A saved multi-turn chain-leaf TaskRun (parent_task_run_id set).

    parent_task_run_id is only constructible when the task is multiturn;
    turn_mode is frozen, so use a multiturn copy (same path) as parent.
    """
    multiturn_task = task.model_copy(update={"turn_mode": TurnMode.multiturn})
    task_run = TaskRun(
        input="turn 1",
        output=TaskOutput(output="reply", source=data_source),
        parent=multiturn_task,
        parent_task_run_id="some_parent_id",
        trace=trace,
    )
    task_run.save_to_file()
    return task_run


class RecordingStubV2Eval(StubV2Eval):
    """StubV2Eval that records the EvalTaskInput it was asked to evaluate.

    A test that builds the judge itself reads its instance's ``seen_inputs``. When
    ``generating()`` builds the judge per eval config the instance is out of reach,
    so every input also lands in the class-level ``seen`` list that the
    ``recorded_judge_inputs`` fixture resets around each test.
    """

    seen: ClassVar[list[EvalTaskInput]] = []

    def __init__(self, eval_config: EvalConfig):
        super().__init__(eval_config)
        self.seen_inputs: list[EvalTaskInput] = []

    async def evaluate(self, eval_input: EvalTaskInput) -> V2EvalResult:
        self.seen_inputs.append(eval_input)
        RecordingStubV2Eval.seen.append(eval_input)
        return await super().evaluate(eval_input)


class TestRunV2Job:
    @pytest.mark.asyncio
    async def test_v2_dispatch_from_run_job(
        self, mock_v2_runner, mock_v2_eval_config, mock_eval_inputs, data_source
    ):
        task_run = TaskRun(
            input="test input",
            output=TaskOutput(output="hello", source=data_source),
            parent=mock_v2_runner.task,
        )
        task_run.save_to_file()
        job = EvalJob(
            item=task_run,
            eval_config=mock_v2_eval_config,
            type="eval_config_eval",
        )
        with patch(
            "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
            return_value=StubV2Eval(mock_v2_eval_config),
        ):
            result = await mock_v2_runner.run_job(job)
        assert result is True
        runs = mock_v2_eval_config.runs(readonly=True)
        assert len(runs) == 1
        saved = runs[0]
        assert saved.scores == {"accuracy": 1.0}
        assert saved.dataset_id == task_run.id
        assert saved.eval_input_id is None
        assert saved.skipped_reason is None
        assert saved.scored_run_id == task_run.id
        assert saved.output is None

    @pytest.mark.asyncio
    async def test_type_not_available_skip(
        self, mock_v2_runner, mock_v2_eval_config, mock_eval_inputs, data_source
    ):
        task_run = TaskRun(
            input="test input",
            output=TaskOutput(output="hello", source=data_source),
            parent=mock_v2_runner.task,
        )
        task_run.save_to_file()
        job = EvalJob(
            item=task_run,
            eval_config=mock_v2_eval_config,
            type="eval_config_eval",
        )
        with patch(
            "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
            side_effect=NotImplementedError("not yet implemented"),
        ):
            result = await mock_v2_runner.run_job(job)
        assert result is True
        runs = mock_v2_eval_config.runs(readonly=True)
        assert len(runs) == 1
        saved = runs[0]
        assert saved.scores == {}
        assert saved.skipped_reason == SkippedReason.type_not_available.value
        assert saved.skipped_detail == "V2 eval type not yet implemented"
        assert saved.dataset_id == task_run.id
        assert saved.eval_input_id is None
        # Calibration scores the golden item itself, so the trace exists no matter where
        # the skip happened — same shape as the scoring-time skip below.
        assert saved.scored_run_id == task_run.id
        assert eval_traces(mock_v2_runner.task) == []
        # No copy of the item on the record: the inline trace fields are deprecated and
        # new records never set them.
        assert saved.output is None
        assert saved.input is None

    @pytest.mark.asyncio
    async def test_adapter_skipped_reason(
        self, mock_v2_runner, mock_v2_eval_config, mock_eval_inputs, data_source
    ):
        task_run = TaskRun(
            input="test input",
            output=TaskOutput(output="hello", source=data_source),
            parent=mock_v2_runner.task,
        )
        task_run.save_to_file()
        job = EvalJob(
            item=task_run,
            eval_config=mock_v2_eval_config,
            type="eval_config_eval",
        )
        with patch(
            "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
            return_value=SkippingStubV2Eval(mock_v2_eval_config),
        ):
            result = await mock_v2_runner.run_job(job)
        assert result is True
        runs = mock_v2_eval_config.runs(readonly=True)
        assert len(runs) == 1
        saved = runs[0]
        assert saved.scores == {}
        assert saved.skipped_reason == SkippedReason.extraction_failed.value
        assert saved.skipped_detail == "test skip detail"
        # Skipped at scoring time, so the trace it could not score is still named.
        assert saved.scored_run_id == task_run.id
        assert saved.output is None

    @pytest.mark.asyncio
    async def test_type_not_available_skip_eval_input(
        self, mock_v2_runner, mock_v2_eval_config, mock_eval_inputs, mock_run_config
    ):
        # task_run_eval shape: an eval_config_eval job over an EvalInput is no
        # longer even recordable (EvalRun rejects it — calibration is
        # TaskRun-only), so the skip-writer is exercised on the legit lane.
        ei = mock_eval_inputs[1]
        job = EvalJob(
            item=ei,
            eval_config=mock_v2_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        with patch(
            "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
            side_effect=NotImplementedError("not yet implemented"),
        ):
            result = await mock_v2_runner.run_job(job)
        assert result is True
        runs = mock_v2_eval_config.runs(readonly=True)
        assert len(runs) == 1
        saved = runs[0]
        assert saved.eval_input_id == ei.id
        assert saved.dataset_id is None
        assert saved.eval_config_eval is False
        assert saved.task_run_config_id == mock_run_config.id
        assert saved.skipped_reason == SkippedReason.type_not_available.value
        # A scoring job that can never be scored has nothing to point at, and pays for
        # no generation to give itself one.
        assert saved.scored_run_id is None
        assert eval_traces(mock_v2_runner.task) == []
        assert saved.input is None

    @pytest.mark.asyncio
    async def test_multi_turn_task_run_eval_config_eval_scores_stored_trace(
        self, mock_v2_runner, mock_v2_eval_config, data_source
    ):
        task_run = make_multi_turn_leaf(mock_v2_runner.task, data_source)
        job = EvalJob(
            item=task_run,
            eval_config=mock_v2_eval_config,
            type="eval_config_eval",
        )
        stub = RecordingStubV2Eval(mock_v2_eval_config)
        with patch(
            "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
            return_value=stub,
        ):
            result = await mock_v2_runner.run_job(job)
        assert result is True

        # The adapter received the stored conversation, not a regeneration.
        assert len(stub.seen_inputs) == 1
        eval_task_input = stub.seen_inputs[0]
        assert eval_task_input.final_message == "reply"
        assert eval_task_input.task_input == "turn 1"
        assert eval_task_input.trace == MULTI_TURN_TRACE

        runs = mock_v2_eval_config.runs(readonly=True)
        assert len(runs) == 1
        saved = runs[0]
        assert saved.scores == {"accuracy": 1.0}
        assert saved.skipped_reason is None
        assert saved.dataset_id == task_run.id
        assert saved.eval_input_id is None
        assert saved.eval_config_eval is True
        assert saved.task_run_config_id is None
        # Pointer record at the stored leaf: no inline copy of the conversation.
        assert saved.scored_run_id == task_run.id
        assert saved.input is None
        assert saved.output is None
        assert saved.task_run_trace is None
        # The leaf stays a curated dataset item — no eval_source stamp, which
        # would pull it off dataset surfaces. Checked on the saved bytes.
        assert json.loads(task_run.path.read_text()).get("eval_source") is None

    @pytest.mark.asyncio
    async def test_multi_turn_eval_config_eval_full_trace_records_no_trace(
        self, mock_v2_runner, mock_v2_eval_config, data_source
    ):
        # A full_trace eval changes what the judge reads, not where the trace
        # lives: the record stays a pointer at the stored leaf.
        mock_v2_eval_config.parent.evaluation_data_type = EvalDataType.full_trace
        mock_v2_eval_config.parent.save_to_file()

        task_run = make_multi_turn_leaf(mock_v2_runner.task, data_source)
        job = EvalJob(
            item=task_run,
            eval_config=mock_v2_eval_config,
            type="eval_config_eval",
        )
        with patch(
            "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
            return_value=StubV2Eval(mock_v2_eval_config),
        ):
            result = await mock_v2_runner.run_job(job)
        assert result is True

        runs = mock_v2_eval_config.runs(readonly=True)
        assert len(runs) == 1
        saved = runs[0]
        assert saved.scores == {"accuracy": 1.0}
        assert saved.scored_run_id == task_run.id
        assert saved.task_run_trace is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("trace", [None, []])
    async def test_multi_turn_task_run_without_trace_skipped(
        self, mock_v2_runner, mock_v2_eval_config, data_source, trace
    ):
        task_run = make_multi_turn_leaf(mock_v2_runner.task, data_source, trace=trace)
        job = EvalJob(
            item=task_run,
            eval_config=mock_v2_eval_config,
            type="eval_config_eval",
        )
        with patch(
            "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
            return_value=StubV2Eval(mock_v2_eval_config),
        ):
            result = await mock_v2_runner.run_job(job)
        assert result is True
        runs = mock_v2_eval_config.runs(readonly=True)
        assert len(runs) == 1
        saved = runs[0]
        assert saved.scores == {}
        assert saved.skipped_reason == SkippedReason.missing_trace.value
        assert "no stored trace" in saved.skipped_detail
        # The run exists even though its trace doesn't, so the skip still
        # points at what it could not score.
        assert saved.scored_run_id == task_run.id
        assert saved.input is None
        assert saved.output is None

    @pytest.mark.asyncio
    async def test_multi_turn_task_run_adapter_skip_persists(
        self, mock_v2_runner, mock_v2_eval_config, data_source
    ):
        task_run = make_multi_turn_leaf(mock_v2_runner.task, data_source)
        job = EvalJob(
            item=task_run,
            eval_config=mock_v2_eval_config,
            type="eval_config_eval",
        )
        with patch(
            "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
            return_value=SkippingStubV2Eval(mock_v2_eval_config),
        ):
            result = await mock_v2_runner.run_job(job)
        assert result is True
        runs = mock_v2_eval_config.runs(readonly=True)
        assert len(runs) == 1
        saved = runs[0]
        assert saved.scores == {}
        assert saved.skipped_reason == SkippedReason.extraction_failed.value
        assert saved.skipped_detail == "test skip detail"
        assert saved.scored_run_id == task_run.id
        assert saved.output is None


@pytest.fixture
def mock_v2_task_run_eval(mock_task):
    eval = Eval(
        id="v2_eval_tr",
        name="v2 task run eval",
        description="v2 eval for task_run_eval mode",
        eval_set_filter_id="all",
        eval_configs_filter_id="all",
        output_scores=[
            EvalOutputScore(
                name="Accuracy",
                instruction="Check if the output is accurate",
                type=TaskOutputRatingType.pass_fail,
            ),
        ],
        parent=mock_task,
    )
    eval.save_to_file()
    return eval


@pytest.fixture
def mock_v2_task_run_eval_config(mock_v2_task_run_eval):
    eval_config = EvalConfig(
        name="v2 tr config",
        config_type=EvalConfigType.v2,
        properties=ExactMatchProperties(expected_value="hello"),
        parent=mock_v2_task_run_eval,
    )
    eval_config.save_to_file()
    return eval_config


@pytest.fixture
def mock_v2_task_run_eval_runner(
    mock_v2_task_run_eval, mock_v2_task_run_eval_config, mock_run_config
):
    return EvalRunner(
        eval_configs=[mock_v2_task_run_eval_config],
        run_configs=[mock_run_config],
        eval_run_type="task_run_eval",
        split=_test_split([mock_v2_task_run_eval_config]),
    )


class TestV2FreshGeneration:
    @pytest.mark.asyncio
    async def test_task_run_eval_generates_persists_and_scores_the_trace(
        self,
        mock_v2_task_run_eval_runner,
        mock_v2_task_run_eval_config,
        mock_run_config,
        data_source,
    ):
        """The trace is a TaskRun of its own, and the score only points at it."""
        item = TaskRun(
            input="test input",
            output=TaskOutput(output="stale output", source=data_source),
            parent=mock_v2_task_run_eval_runner.task,
        )
        item.save_to_file()

        job = EvalJob(
            item=item,
            eval_config=mock_v2_task_run_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )

        generator = TraceGenerator(mock_v2_task_run_eval_runner.task, "hello")
        with generating(generator):
            result = await mock_v2_task_run_eval_runner.run_job(job)

        assert result is True
        assert generator.calls == [(item, mock_run_config.id)]

        (saved,) = mock_v2_task_run_eval_config.runs(readonly=True)
        assert saved.scores == {"accuracy": 1.0}
        assert saved.dataset_id == item.id
        assert saved.eval_config_eval is False
        assert saved.skipped_reason is None
        assert saved.scored_run_id is not None
        assert saved.scored_run_id != item.id
        # No inline copy of what was scored: the trace has one home now.
        assert saved.input is None
        assert saved.output is None
        assert saved.task_run_trace is None
        assert saved.task_run_usage is None
        assert saved.reference_answer is None

        trace = trace_for(mock_v2_task_run_eval_runner.task, saved.scored_run_id)
        assert trace.output.output == "hello"
        assert trace.eval_source == EvalItemSource(
            source_type="task_run", source_id=item.id
        )
        assert trace.output.source.run_config_id == mock_run_config.id

    @pytest.mark.asyncio
    async def test_the_runner_persists_a_run_the_adapter_left_unsaved(
        self,
        mock_v2_task_run_eval_runner,
        mock_v2_task_run_eval_config,
        mock_run_config,
        data_source,
    ):
        """Production reproduction: `run_task` builds its adapter with
        `allow_saving=False`, and every adapter clears the id of a run it did not persist
        (`base_adapter.py:346`). The runner mints one before saving.

        Without that, `save_to_file()` raises "ID is not set - can not save or build
        path" and every V2 task_run_eval job fails. The score still records the *source*
        item as `dataset_id`, and the minted trace as `scored_run_id`.
        """
        task = mock_v2_task_run_eval_runner.task
        item = TaskRun(
            input="test input",
            output=TaskOutput(output="stale output", source=data_source),
            parent=task,
        )
        item.save_to_file()

        # The double is only worth anything if it reproduces the unsaved shape.
        assert (await TraceGenerator(task)(item, mock_run_config.id)).id is None

        job = EvalJob(
            item=item,
            eval_config=mock_v2_task_run_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        with generating(TraceGenerator(task)):
            assert await mock_v2_task_run_eval_runner.run_job(job) is True

        (trace,) = eval_traces(task)
        assert trace.id is not None
        assert trace.path is not None and trace.path.exists()
        assert TaskRun.load_from_file(trace.path).id == trace.id

        (saved,) = mock_v2_task_run_eval_config.runs(readonly=True)
        assert saved.dataset_id == item.id
        assert saved.scored_run_id == trace.id

    @pytest.mark.asyncio
    async def test_generated_trace_is_hidden_from_the_dataset(
        self,
        mock_v2_task_run_eval_runner,
        mock_v2_task_run_eval_config,
        mock_run_config,
        data_source,
    ):
        """Phase 1's default-exclude is what makes persisting traces safe at all."""
        task = mock_v2_task_run_eval_runner.task
        item = TaskRun(
            input="test input",
            output=TaskOutput(output="stale output", source=data_source),
            parent=task,
        )
        item.save_to_file()

        job = EvalJob(
            item=item,
            eval_config=mock_v2_task_run_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        with generating(TraceGenerator(task)):
            await mock_v2_task_run_eval_runner.run_job(job)

        assert [run.id for run in task.runs(readonly=True)] == [item.id]
        assert len(task.runs(readonly=True, include_eval_generated=True)) == 2

    @pytest.mark.asyncio
    async def test_judge_usage_is_recorded_on_the_score(
        self,
        mock_v2_task_run_eval_runner,
        mock_v2_task_run_eval_config,
        mock_run_config,
        data_source,
    ):
        """What the judgment cost, which nothing recorded before `eval_usage`.

        Distinct from the task's usage, which now lives on the trace's own TaskRun.
        """
        judge_usage = Usage(input_tokens=90, output_tokens=6, cost=0.002)

        class MeteredJudge(StubV2Eval):
            async def evaluate(self, eval_input):
                return V2EvalResult(scores={"accuracy": 1.0}, usage=judge_usage)

        item = TaskRun(
            input="test input",
            output=TaskOutput(output="stale output", source=data_source),
            parent=mock_v2_task_run_eval_runner.task,
        )
        item.save_to_file()

        job = EvalJob(
            item=item,
            eval_config=mock_v2_task_run_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        with generating(
            TraceGenerator(mock_v2_task_run_eval_runner.task), MeteredJudge
        ):
            await mock_v2_task_run_eval_runner.run_job(job)

        (saved,) = mock_v2_task_run_eval_config.runs(readonly=True)
        assert saved.eval_usage == judge_usage
        assert saved.task_run_usage is None

    @pytest.mark.asyncio
    async def test_scoring_skip_still_names_the_trace_it_could_not_score(
        self,
        mock_v2_task_run_eval_runner,
        mock_v2_task_run_eval_config,
        mock_run_config,
        data_source,
    ):
        item = TaskRun(
            input="test input",
            output=TaskOutput(output="stale output", source=data_source),
            parent=mock_v2_task_run_eval_runner.task,
        )
        item.save_to_file()

        job = EvalJob(
            item=item,
            eval_config=mock_v2_task_run_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )

        with generating(
            TraceGenerator(mock_v2_task_run_eval_runner.task), SkippingStubV2Eval
        ):
            result = await mock_v2_task_run_eval_runner.run_job(job)

        assert result is True
        (saved,) = mock_v2_task_run_eval_config.runs(readonly=True)
        assert saved.skipped_reason == SkippedReason.extraction_failed.value
        assert saved.skipped_detail == "test skip detail"
        assert saved.scores == {}
        assert saved.dataset_id == item.id
        # Generation succeeded and only the judge gave up, so the trace stays reusable —
        # and the record names the trace it could not score, not merely some trace.
        (trace,) = eval_traces(mock_v2_task_run_eval_runner.task)
        assert saved.scored_run_id == trace.id
        assert saved.output is None

    @pytest.mark.asyncio
    async def test_task_run_eval_multi_turn_scores_stored_trace_without_regen(
        self,
        mock_v2_task_run_eval_runner,
        mock_v2_task_run_eval_config,
        mock_run_config,
        data_source,
    ):
        """Multi-turn leaves can't be regenerated single-shot; task_run_eval
        mode scores the stored trace and never calls run_task."""
        leaf = make_multi_turn_leaf(mock_v2_task_run_eval_runner.task, data_source)
        job = EvalJob(
            item=leaf,
            eval_config=mock_v2_task_run_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )

        stub = RecordingStubV2Eval(mock_v2_task_run_eval_config)
        with (
            patch.object(
                stub, "run_task", side_effect=AssertionError("run_task called")
            ),
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=stub,
            ),
        ):
            result = await mock_v2_task_run_eval_runner.run_job(job)

        assert result is True
        assert len(stub.seen_inputs) == 1
        assert stub.seen_inputs[0].trace == MULTI_TURN_TRACE

        runs = mock_v2_task_run_eval_config.runs(readonly=True)
        assert len(runs) == 1
        saved = runs[0]
        assert saved.dataset_id == leaf.id
        assert saved.scores == {"accuracy": 1.0}
        assert saved.eval_config_eval is False
        assert saved.task_run_config_id == mock_run_config.id
        assert saved.skipped_reason is None
        # Pointer record at the stored leaf: no inline copy of what was scored.
        assert saved.scored_run_id == leaf.id
        assert saved.input is None
        assert saved.output is None
        assert saved.task_run_trace is None
        # And the leaf stays an unstamped dataset item, visible as before.
        # Checked on the saved bytes.
        assert json.loads(leaf.path.read_text()).get("eval_source") is None

    @pytest.mark.asyncio
    async def test_task_run_eval_multi_turn_full_trace_judges_in_memory_usage(
        self,
        mock_v2_task_run_eval_runner,
        mock_v2_task_run_eval_config,
        mock_run_config,
        data_source,
    ):
        """A stored trace whose assistant turns carry MessageUsage objects (not
        plain JSON) must reach the judge intact, and the record stays a pointer
        — nothing serializes the conversation onto it."""
        mock_v2_task_run_eval_config.parent.evaluation_data_type = (
            EvalDataType.full_trace
        )
        mock_v2_task_run_eval_config.parent.save_to_file()

        trace_with_usage: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "turn 1"},
            {
                "role": "assistant",
                "content": "reply",
                "usage": MessageUsage(input_tokens=5, output_tokens=7),
            },
        ]
        leaf = make_multi_turn_leaf(
            mock_v2_task_run_eval_runner.task, data_source, trace=trace_with_usage
        )
        job = EvalJob(
            item=leaf,
            eval_config=mock_v2_task_run_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        stub = RecordingStubV2Eval(mock_v2_task_run_eval_config)
        with patch(
            "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
            return_value=stub,
        ):
            result = await mock_v2_task_run_eval_runner.run_job(job)

        assert result is True
        assert len(stub.seen_inputs) == 1
        judged_trace = stub.seen_inputs[0].trace
        assert [m["role"] for m in judged_trace] == ["user", "assistant"]
        assert judged_trace[1]["content"] == "reply"

        runs = mock_v2_task_run_eval_config.runs(readonly=True)
        assert len(runs) == 1
        saved = runs[0]
        assert saved.scores == {"accuracy": 1.0}
        assert saved.scored_run_id == leaf.id
        assert saved.task_run_trace is None

    async def test_calibration_scores_the_golden_item_itself(
        self,
        mock_v2_runner,
        mock_v2_eval_config,
        data_source,
    ):
        """No generation, so no trace: `scored_run_id` is the dataset item (spec 4.5)."""
        task = mock_v2_runner.task
        golden = TaskRun(
            input="existing input",
            output=TaskOutput(output="existing output", source=data_source),
            parent=task,
        )
        golden.save_to_file()

        job = EvalJob(
            item=golden,
            eval_config=mock_v2_eval_config,
            type="eval_config_eval",
        )

        generator = TraceGenerator(task)
        with generating(generator):
            result = await mock_v2_runner.run_job(job)

        assert result is True
        assert generator.calls == []
        (saved,) = mock_v2_eval_config.runs(readonly=True)
        assert saved.scores == {"accuracy": 1.0}
        assert saved.dataset_id == golden.id
        assert saved.scored_run_id == golden.id
        assert saved.eval_config_eval is True
        assert saved.skipped_reason is None
        # The golden item is not flagged, so it stays a normal dataset run: visible,
        # and deletable.
        assert [
            run.id for run in task.runs(readonly=True, include_eval_generated=True)
        ] == [golden.id]
        assert TaskRun.load_from_file(golden.path).eval_source is None

    @pytest.mark.asyncio
    async def test_failed_generation_persists_nothing(
        self,
        mock_v2_task_run_eval_runner,
        mock_v2_task_run_eval_config,
        mock_run_config,
        data_source,
    ):
        """D14: a failed generation has no representation — not a trace, not a score."""
        task = mock_v2_task_run_eval_runner.task
        item = TaskRun(
            input="test input",
            output=TaskOutput(output="stale output", source=data_source),
            parent=task,
        )
        item.save_to_file()

        job = EvalJob(
            item=item,
            eval_config=mock_v2_task_run_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )

        stub = StubV2Eval(mock_v2_task_run_eval_config)
        with (
            patch.object(stub, "run_task", side_effect=RuntimeError("model exploded")),
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=stub,
            ),
            pytest.raises(RuntimeError, match="model exploded"),
        ):
            await mock_v2_task_run_eval_runner.run_job(job)

        assert mock_v2_task_run_eval_config.runs(readonly=True) == []
        assert len(task.runs(readonly=True, include_eval_generated=True)) == 1

    @pytest.mark.asyncio
    async def test_retry_after_a_scoring_failure_rescores_without_regenerating(
        self,
        mock_v2_task_run_eval_runner,
        mock_v2_task_run_eval_config,
        mock_run_config,
        data_source,
    ):
        """Functional spec 4.3: the trace survives the judge, so a retry is cheap.

        Two `run_job` calls on one job, which is what `AsyncJobRunner` does on a
        RetryableError — the live index is what makes the second one skip generation.
        """
        task = mock_v2_task_run_eval_runner.task
        item = TaskRun(
            input="test input",
            output=TaskOutput(output="stale output", source=data_source),
            parent=task,
        )
        item.save_to_file()

        job = EvalJob(
            item=item,
            eval_config=mock_v2_task_run_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )

        stub = StubV2Eval(mock_v2_task_run_eval_config)
        generator = TraceGenerator(task)
        with generating(generator, lambda config: stub):
            with patch.object(
                stub,
                "evaluate",
                side_effect=litellm.RateLimitError("slow down", "", ""),
            ):
                with pytest.raises(RetryableError):
                    await mock_v2_task_run_eval_runner.run_job(job)

            assert len(generator.calls) == 1
            traces = eval_traces(task)
            assert len(traces) == 1
            assert mock_v2_task_run_eval_config.runs(readonly=True) == []

            assert await mock_v2_task_run_eval_runner.run_job(job) is True

        assert len(generator.calls) == 1
        (saved,) = mock_v2_task_run_eval_config.runs(readonly=True)
        assert saved.scored_run_id == traces[0].id


# -------------------------------------------------------------------
# What the judge is handed: reference data reaching the evaluator
# -------------------------------------------------------------------
@pytest.fixture
def recorded_judge_inputs():
    RecordingStubV2Eval.seen = []
    yield RecordingStubV2Eval.seen
    RecordingStubV2Eval.seen = []


class TestReferenceDataReachesTheJudge:
    """A QnA dataset item stores the reference answer as its output. The judge is told
    to grade against it, so it has to arrive."""

    @pytest.mark.asyncio
    async def test_a_dataset_items_output_is_the_judges_reference_answer(
        self,
        mock_v2_task_run_eval_runner,
        mock_v2_task_run_eval_config,
        mock_run_config,
        data_source,
        recorded_judge_inputs,
    ):
        item = TaskRun(
            input="Who wrote Dune?",
            output=TaskOutput(output="Frank Herbert.", source=data_source),
            parent=mock_v2_task_run_eval_runner.task,
        )
        item.save_to_file()

        job = EvalJob(
            item=item,
            eval_config=mock_v2_task_run_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )

        generator = TraceGenerator(mock_v2_task_run_eval_runner.task, "Herbert, 1965.")
        with generating(generator, RecordingStubV2Eval):
            assert await mock_v2_task_run_eval_runner.run_job(job) is True

        (seen,) = recorded_judge_inputs
        assert seen.final_message == "Herbert, 1965."
        assert seen.reference_data == {"reference_answer": "Frank Herbert."}

        # The judge saw it; the record does not keep it. A pointer-mode record carries
        # no second copy of what it scored: the reference is derived from the item
        # `dataset_id` names, and `EvalRun` has no field to hold one.
        (saved,) = mock_v2_task_run_eval_config.runs(readonly=True)
        assert saved.dataset_id == item.id
        assert not hasattr(saved, "reference_data")

    @pytest.mark.asyncio
    async def test_calibration_gives_the_judge_no_reference_answer(
        self,
        mock_v2_runner,
        mock_v2_eval_config,
        data_source,
        recorded_judge_inputs,
    ):
        """Calibration scores the golden item as itself, so a reference here would be
        the answer key and the response — every item would pass."""
        golden = TaskRun(
            input="Who wrote Dune?",
            output=TaskOutput(output="Frank Herbert.", source=data_source),
            parent=mock_v2_runner.task,
        )
        golden.save_to_file()

        job = EvalJob(
            item=golden,
            eval_config=mock_v2_eval_config,
            type="eval_config_eval",
        )

        with generating(TraceGenerator(mock_v2_runner.task), RecordingStubV2Eval):
            assert await mock_v2_runner.run_job(job) is True

        (seen,) = recorded_judge_inputs
        assert seen.final_message == "Frank Herbert."
        assert seen.reference_data is None


# -------------------------------------------------------------------
# V2 EvalInput + task_run_eval (fresh generation from EvalInput)
# -------------------------------------------------------------------
@pytest.fixture
def mock_v2_eval_input_task_run_eval(mock_task):
    eval = Eval(
        id="v2_ei_tr_eval",
        name="v2 eval input task_run_eval",
        description="v2 eval for EvalInput + task_run_eval mode",
        splits={"test": EvalInputSplit(filter_id="all")},
        eval_configs_filter_id="all",
        output_scores=[
            EvalOutputScore(
                name="Accuracy",
                instruction="Check if the output is accurate",
                type=TaskOutputRatingType.pass_fail,
            ),
        ],
        parent=mock_task,
    )
    eval.save_to_file()
    return eval


@pytest.fixture
def mock_v2_ei_tr_eval_config(mock_v2_eval_input_task_run_eval):
    eval_config = EvalConfig(
        name="v2 ei tr config",
        config_type=EvalConfigType.v2,
        properties=ExactMatchProperties(expected_value="4"),
        parent=mock_v2_eval_input_task_run_eval,
    )
    eval_config.save_to_file()
    return eval_config


@pytest.fixture
def mock_v2_ei_tr_runner(
    mock_v2_eval_input_task_run_eval,
    mock_v2_ei_tr_eval_config,
    mock_run_config,
    mock_eval_inputs,
):
    # Depends on mock_eval_inputs: the split is a snapshot taken at construction,
    # so the inputs must exist on disk before the runner is built.
    return EvalRunner(
        eval_configs=[mock_v2_ei_tr_eval_config],
        run_configs=[mock_run_config],
        eval_run_type="task_run_eval",
        split=_test_split([mock_v2_ei_tr_eval_config]),
    )

    @pytest.mark.asyncio
    async def test_task_run_eval_uses_source_id_when_fresh_run_unsaved(
        self,
        mock_v2_task_run_eval_runner,
        mock_v2_task_run_eval_config,
        mock_run_config,
        data_source,
    ):
        """Production reproduction: run_task uses allow_saving=False, so the fresh
        TaskRun comes back with id=None. The EvalRun must still record the source
        dataset item's id."""
        stale_task_run = TaskRun(
            input="test input",
            output=TaskOutput(output="stale output", source=data_source),
            parent=mock_v2_task_run_eval_runner.task,
        )
        stale_task_run.save_to_file()

        fresh_task_run = TaskRun(
            input="test input",
            output=TaskOutput(output="hello", source=data_source),
            parent=mock_v2_task_run_eval_runner.task,
        )
        fresh_task_run.id = None

        job = EvalJob(
            item=stale_task_run,
            eval_config=mock_v2_task_run_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )

        stub = StubV2Eval(mock_v2_task_run_eval_config)
        with (
            patch.object(stub, "run_task", return_value=fresh_task_run),
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=stub,
            ),
        ):
            result = await mock_v2_task_run_eval_runner.run_job(job)

        assert result is True
        runs = mock_v2_task_run_eval_config.runs(readonly=True)
        assert len(runs) == 1
        saved = runs[0]
        assert saved.dataset_id == stale_task_run.id
        assert saved.eval_input_id is None
        assert saved.output == "hello"
        assert saved.scores == {"accuracy": 1.0}


class TestV2EvalInputFreshGeneration:
    @pytest.mark.asyncio
    async def test_eval_input_task_run_eval_generates_and_evaluates(
        self,
        mock_v2_ei_tr_runner,
        mock_v2_ei_tr_eval_config,
        mock_eval_inputs,
        mock_run_config,
    ):
        ei = mock_eval_inputs[0]
        job = EvalJob(
            item=ei,
            eval_config=mock_v2_ei_tr_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )

        generator = TraceGenerator(mock_v2_ei_tr_runner.task, "4")
        with generating(generator):
            result = await mock_v2_ei_tr_runner.run_job(job)

        assert result is True
        assert generator.calls == [(ei, mock_run_config.id)]

        (saved,) = mock_v2_ei_tr_eval_config.runs(readonly=True)
        assert saved.eval_input_id == ei.id
        assert saved.dataset_id is None
        assert saved.eval_config_eval is False
        assert saved.scores == {"accuracy": 1.0}
        assert saved.skipped_reason is None
        assert saved.input is None
        assert saved.output is None

        trace = trace_for(mock_v2_ei_tr_runner.task, saved.scored_run_id)
        assert trace.output.output == "4"
        assert trace.eval_source == EvalItemSource(
            source_type="eval_input", source_id=ei.id
        )
        assert trace.output.source.run_config_id == mock_run_config.id

    @pytest.mark.asyncio
    async def test_eval_input_task_run_eval_skip_persists_skipped(
        self,
        mock_v2_ei_tr_runner,
        mock_v2_ei_tr_eval_config,
        mock_eval_inputs,
        mock_run_config,
    ):
        ei = mock_eval_inputs[1]
        job = EvalJob(
            item=ei,
            eval_config=mock_v2_ei_tr_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )

        with generating(TraceGenerator(mock_v2_ei_tr_runner.task), SkippingStubV2Eval):
            result = await mock_v2_ei_tr_runner.run_job(job)

        assert result is True
        (saved,) = mock_v2_ei_tr_eval_config.runs(readonly=True)
        assert saved.eval_input_id == ei.id
        assert saved.dataset_id is None
        assert saved.eval_config_eval is False
        assert saved.skipped_reason == SkippedReason.extraction_failed.value
        assert saved.skipped_detail == "test skip detail"
        (trace,) = eval_traces(mock_v2_ei_tr_runner.task)
        assert saved.scored_run_id == trace.id
        assert saved.output is None
        assert saved.scores == {}

    @pytest.mark.asyncio
    async def test_eval_input_task_run_eval_no_reference(
        self,
        mock_task,
        mock_v2_ei_tr_eval_config,
        mock_run_config,
        recorded_judge_inputs,
    ):
        ei_no_ref = EvalInput(
            id="ei_no_ref",
            data=SingleTurnEvalInputData(
                user_message=UserMessage(text="no ref input"),
            ),
            reference=None,
            parent=mock_task,
        )
        ei_no_ref.save_to_file()

        runner = EvalRunner(
            eval_configs=[mock_v2_ei_tr_eval_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_ei_tr_eval_config]),
        )

        job = EvalJob(
            item=ei_no_ref,
            eval_config=mock_v2_ei_tr_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )

        with generating(TraceGenerator(runner.task), RecordingStubV2Eval):
            result = await runner.run_job(job)

        assert result is True
        (seen,) = recorded_judge_inputs
        assert seen.reference_data is None
        (saved,) = mock_v2_ei_tr_eval_config.runs(readonly=True)
        assert saved.eval_input_id == "ei_no_ref"


# -------------------------------------------------------------------
# Trace reuse: the point of the project
# -------------------------------------------------------------------
@pytest.fixture
def dataset_items(mock_task, data_source):
    items = []
    for i in range(2):
        run = TaskRun(
            input=f"input {i}",
            output=TaskOutput(output=f"dataset output {i}", source=data_source),
            parent=mock_task,
        )
        run.save_to_file()
        items.append(run)
    return items


@pytest.fixture
def second_judge(mock_v2_task_run_eval):
    """A second eval config on the same eval — the judge a user adds after the fact."""
    config = EvalConfig(
        name="second judge",
        config_type=EvalConfigType.v2,
        properties=ExactMatchProperties(expected_value="something else"),
        parent=mock_v2_task_run_eval,
    )
    config.save_to_file()
    return config


@pytest.fixture
def second_run_config(mock_task):
    rc = TaskRunConfig(
        name="second run config",
        description="a different way of running the task",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=mock_task,
    )
    rc.save_to_file()
    return rc


async def run_to_completion(runner) -> None:
    last = None
    async for progress in runner.run():
        last = progress
    assert last is not None
    assert last.errors == 0, f"{last.errors} job(s) errored"


def scored_run_ids(eval_config) -> dict:
    return {
        run.dataset_id or run.eval_input_id: run.scored_run_id
        for run in eval_config.runs(readonly=True)
    }


class TestTraceReuse:
    @pytest.mark.asyncio
    async def test_a_second_judge_reuses_the_first_judges_traces(
        self,
        mock_task,
        mock_v2_task_run_eval_config,
        second_judge,
        mock_run_config,
        dataset_items,
    ):
        """The project, in one test.

        Run an eval, add a judge, run again: the second judge scores what the first one
        already paid to generate. It also makes the comparison *paired* — both judges saw
        the same generation, so the delta between them is the judges.
        """
        generator = TraceGenerator(mock_task)

        first_run = build_task_run_eval_runner(
            [mock_v2_task_run_eval_config], [mock_run_config]
        )
        with generating(generator):
            await run_to_completion(first_run)

        assert len(generator.calls) == len(dataset_items)
        first_scores = scored_run_ids(mock_v2_task_run_eval_config)
        assert len(first_scores) == len(dataset_items)

        second_run = build_task_run_eval_runner(
            [mock_v2_task_run_eval_config, second_judge], [mock_run_config]
        )
        with generating(generator):
            await run_to_completion(second_run)

        assert len(generator.calls) == len(dataset_items), (
            "the second run generated something"
        )
        assert len(eval_traces(mock_task)) == len(dataset_items)
        assert len(mock_v2_task_run_eval_config.runs(readonly=True)) == len(
            dataset_items
        ), "the first judge was re-scored"
        assert scored_run_ids(second_judge) == first_scores

        # The second judge reads its trace back off disk, so the fields the split moved
        # onto the TaskRun have to survive the round-trip — the messages a full_trace
        # eval scores, and the usage Phase 4's rollup reports.
        for trace in eval_traces(mock_task):
            assert trace.trace == generator.trace
            assert trace.usage == generator.usage

    @pytest.mark.asyncio
    async def test_two_judges_in_one_run_generate_once_per_item(
        self,
        mock_task,
        mock_v2_task_run_eval_config,
        second_judge,
        mock_run_config,
        dataset_items,
    ):
        """The concurrent half, which a precomputed lookup could never have caught.

        `AsyncJobRunner` runs these four jobs at 25-way concurrency, so the two judges'
        jobs for one item overlap. Only the live per-key lock stops both from generating.
        """
        generator = TraceGenerator(mock_task)
        runner = build_task_run_eval_runner(
            [mock_v2_task_run_eval_config, second_judge], [mock_run_config]
        )
        with generating(generator):
            await run_to_completion(runner)

        assert len(generator.calls) == len(dataset_items)
        assert len(eval_traces(mock_task)) == len(dataset_items)
        assert scored_run_ids(mock_v2_task_run_eval_config) == scored_run_ids(
            second_judge
        )

    @pytest.mark.asyncio
    async def test_a_different_eval_reuses_the_same_traces(
        self,
        mock_task,
        mock_v2_task_run_eval,
        mock_v2_task_run_eval_config,
        mock_run_config,
        dataset_items,
    ):
        """Cross-*eval* reuse, which falls out of storing traces on the task (spec 10).

        The reuse key deliberately names no eval, so a second eval over the same items
        and run config scores what the first one generated. Nothing else pins this, and
        it is the one reuse axis a future change to the key would break silently.
        """
        other_eval = Eval(
            id="other_eval",
            name="a different eval",
            description="same items, different question",
            eval_set_filter_id="all",
            eval_configs_filter_id="all",
            output_scores=[
                EvalOutputScore(
                    name="Accuracy",
                    instruction="Check if the output is accurate",
                    type=TaskOutputRatingType.pass_fail,
                ),
            ],
            parent=mock_task,
        )
        other_eval.save_to_file()
        other_config = EvalConfig(
            name="other eval judge",
            config_type=EvalConfigType.v2,
            properties=ExactMatchProperties(expected_value="hello"),
            parent=other_eval,
        )
        other_config.save_to_file()

        generator = TraceGenerator(mock_task)
        with generating(generator):
            await run_to_completion(
                build_task_run_eval_runner(
                    [mock_v2_task_run_eval_config], [mock_run_config]
                )
            )
            first_scores = scored_run_ids(mock_v2_task_run_eval_config)

            await run_to_completion(
                build_task_run_eval_runner([other_config], [mock_run_config])
            )

        assert len(generator.calls) == len(dataset_items)
        assert len(eval_traces(mock_task)) == len(dataset_items)
        assert scored_run_ids(other_config) == first_scores

    @pytest.mark.asyncio
    async def test_a_second_run_config_generates_its_own_trace(
        self,
        mock_task,
        mock_v2_task_run_eval_config,
        mock_run_config,
        second_run_config,
        dataset_items,
    ):
        """Reuse is keyed on the run config, not around it: two ways of running the task
        are two generations, which is the comparison the eval exists to make."""
        generator = TraceGenerator(mock_task)
        runner = build_task_run_eval_runner(
            [mock_v2_task_run_eval_config], [mock_run_config, second_run_config]
        )
        with generating(generator):
            await run_to_completion(runner)

        assert len(generator.calls) == 2 * len(dataset_items)
        traces = eval_traces(mock_task)
        assert len(traces) == 2 * len(dataset_items)
        assert len({trace.id for trace in traces}) == len(traces)

        by_run_config = {
            rc.id: {
                run.scored_run_id
                for run in mock_v2_task_run_eval_config.runs(readonly=True)
                if run.task_run_config_id == rc.id
            }
            for rc in (mock_run_config, second_run_config)
        }
        assert by_run_config[mock_run_config.id].isdisjoint(
            by_run_config[second_run_config.id]
        )

    @pytest.mark.asyncio
    async def test_rerunning_one_judge_scores_nothing_new(
        self,
        mock_task,
        mock_v2_task_run_eval_config,
        mock_run_config,
        dataset_items,
    ):
        """Reuse must not have cost us the score-level dedupe: a second identical run is
        a no-op, not a re-score against the trace it just found."""
        generator = TraceGenerator(mock_task)
        with generating(generator):
            await run_to_completion(
                build_task_run_eval_runner(
                    [mock_v2_task_run_eval_config], [mock_run_config]
                )
            )
            before = scored_run_ids(mock_v2_task_run_eval_config)

            rerun = build_task_run_eval_runner(
                [mock_v2_task_run_eval_config], [mock_run_config]
            )
            assert rerun.collect_tasks() == []
            await run_to_completion(rerun)

        assert len(generator.calls) == len(dataset_items)
        assert scored_run_ids(mock_v2_task_run_eval_config) == before


class TestCollectTasksEvalInputTaskRunEval:
    def test_crosses_eval_inputs_x_eval_configs_x_run_configs(
        self,
        mock_v2_ei_tr_runner,
        mock_eval_inputs,
        mock_run_config,
    ):
        jobs = mock_v2_ei_tr_runner.collect_tasks()
        assert len(jobs) == 2
        for job in jobs:
            assert isinstance(job.item, EvalInput)
            assert job.type == "task_run_eval"
            assert job.task_run_config == mock_run_config

    def test_multiple_run_configs(
        self,
        mock_task,
        mock_v2_eval_input_task_run_eval,
        mock_v2_ei_tr_eval_config,
        mock_eval_inputs,
    ):
        rc1 = TaskRunConfig(
            name="config1",
            description="first",
            run_config_properties=KilnAgentRunConfigProperties(
                model_name="gpt-4",
                model_provider_name=ModelProviderName.openai,
                prompt_id="simple_prompt_builder",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            parent=mock_task,
        )
        rc1.save_to_file()
        rc2 = TaskRunConfig(
            name="config2",
            description="second",
            run_config_properties=KilnAgentRunConfigProperties(
                model_name="gpt-4o",
                model_provider_name=ModelProviderName.openai,
                prompt_id="simple_prompt_builder",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            parent=mock_task,
        )
        rc2.save_to_file()
        runner = EvalRunner(
            eval_configs=[mock_v2_ei_tr_eval_config],
            run_configs=[rc1, rc2],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_ei_tr_eval_config]),
        )
        jobs = runner.collect_tasks()
        assert len(jobs) == 4
        config_pairs = {(j.item.id, j.task_run_config.id) for j in jobs}
        for ei in mock_eval_inputs:
            assert (ei.id, rc1.id) in config_pairs
            assert (ei.id, rc2.id) in config_pairs

    def test_dedup_already_run_task_run_eval(
        self,
        mock_v2_ei_tr_runner,
        mock_v2_ei_tr_eval_config,
        mock_eval_inputs,
        mock_run_config,
    ):
        run = EvalRun(
            parent=mock_v2_ei_tr_eval_config,
            eval_input_id="ei_1",
            task_run_config_id=mock_run_config.id,
            eval_config_eval=False,
            scores={"accuracy": 1.0},
            input="What is 2+2?",
            output="4",
        )
        run.save_to_file()
        jobs = mock_v2_ei_tr_runner.collect_tasks()
        assert len(jobs) == 1
        assert jobs[0].item.id == "ei_2"

    def test_dedup_does_not_cross_run_configs(
        self,
        mock_task,
        mock_v2_eval_input_task_run_eval,
        mock_v2_ei_tr_eval_config,
        mock_eval_inputs,
    ):
        rc1 = TaskRunConfig(
            name="rc_a",
            description="a",
            run_config_properties=KilnAgentRunConfigProperties(
                model_name="gpt-4",
                model_provider_name=ModelProviderName.openai,
                prompt_id="simple_prompt_builder",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            parent=mock_task,
        )
        rc1.save_to_file()
        rc2 = TaskRunConfig(
            name="rc_b",
            description="b",
            run_config_properties=KilnAgentRunConfigProperties(
                model_name="gpt-4o",
                model_provider_name=ModelProviderName.openai,
                prompt_id="simple_prompt_builder",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            parent=mock_task,
        )
        rc2.save_to_file()
        run = EvalRun(
            parent=mock_v2_ei_tr_eval_config,
            eval_input_id="ei_1",
            task_run_config_id=rc1.id,
            eval_config_eval=False,
            scores={"accuracy": 1.0},
            input="What is 2+2?",
            output="4",
        )
        run.save_to_file()
        runner = EvalRunner(
            eval_configs=[mock_v2_ei_tr_eval_config],
            run_configs=[rc1, rc2],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_ei_tr_eval_config]),
        )
        jobs = runner.collect_tasks()
        assert len(jobs) == 3
        remaining = {(j.item.id, j.task_run_config.id) for j in jobs}
        assert (mock_eval_inputs[0].id, rc1.id) not in remaining
        assert (mock_eval_inputs[0].id, rc2.id) in remaining
        assert (mock_eval_inputs[1].id, rc1.id) in remaining
        assert (mock_eval_inputs[1].id, rc2.id) in remaining


class TestRunTaskFromEvalInput:
    @pytest.mark.asyncio
    async def test_run_task_extracts_eval_input_user_message(
        self,
        mock_v2_ei_tr_eval_config,
        mock_run_config,
    ):
        ei = EvalInput(
            id="ei_rt",
            data=SingleTurnEvalInputData(
                user_message=UserMessage(text="What is 2+2?"),
            ),
        )
        stub = StubV2Eval(
            mock_v2_ei_tr_eval_config,
            run_config=mock_run_config.run_config_properties,
        )
        mock_output = TaskRun(
            input="What is 2+2?",
            output=TaskOutput(
                output="4",
                source=DataSource(
                    type=DataSourceType.synthetic,
                    properties={
                        "model_name": "gpt-4",
                        "model_provider": "openai",
                        "adapter_name": "test",
                    },
                ),
            ),
        )
        with patch(
            "kiln_ai.adapters.eval.base_eval.adapter_for_task"
        ) as mock_adapter_for_task:
            mock_adapter = AsyncMock()
            mock_adapter.invoke = AsyncMock(return_value=mock_output)
            mock_adapter_for_task.return_value = mock_adapter
            result = await stub.run_task(ei)

        assert result == mock_output
        mock_adapter.invoke.assert_awaited_once_with("What is 2+2?")

    @pytest.mark.asyncio
    async def test_run_task_extracts_eval_input_with_json_schema(
        self,
        mock_task,
        mock_run_config,
    ):
        mock_task.input_json_schema = '{"type": "object"}'
        mock_task.save_to_file()

        eval = Eval(
            id="v2_json_eval",
            name="json eval",
            description="json eval desc",
            splits={"test": EvalInputSplit(filter_id="all")},
            eval_configs_filter_id="all",
            output_scores=[
                EvalOutputScore(
                    name="Accuracy",
                    instruction="Check",
                    type=TaskOutputRatingType.pass_fail,
                ),
            ],
            parent=mock_task,
        )
        eval.save_to_file()
        ec = EvalConfig(
            name="json config",
            config_type=EvalConfigType.v2,
            properties=ExactMatchProperties(expected_value="4"),
            parent=eval,
        )
        ec.save_to_file()

        ei = EvalInput(
            id="ei_json",
            data=SingleTurnEvalInputData(
                user_message=UserMessage(text='{"question": "2+2"}'),
            ),
        )
        stub = StubV2Eval(
            ec,
            run_config=mock_run_config.run_config_properties,
        )
        mock_output = TaskRun(
            input='{"question": "2+2"}',
            output=TaskOutput(
                output="4",
                source=DataSource(
                    type=DataSourceType.synthetic,
                    properties={
                        "model_name": "gpt-4",
                        "model_provider": "openai",
                        "adapter_name": "test",
                    },
                ),
            ),
        )
        with patch(
            "kiln_ai.adapters.eval.base_eval.adapter_for_task"
        ) as mock_adapter_for_task:
            mock_adapter = AsyncMock()
            mock_adapter.invoke = AsyncMock(return_value=mock_output)
            mock_adapter_for_task.return_value = mock_adapter
            result = await stub.run_task(ei)

        assert result == mock_output
        mock_adapter.invoke.assert_awaited_once_with({"question": "2+2"})

    @pytest.mark.asyncio
    async def test_run_task_still_works_with_task_run(
        self,
        mock_v2_ei_tr_eval_config,
        mock_run_config,
        data_source,
    ):
        tr = TaskRun(
            input="test input",
            output=TaskOutput(output="test output", source=data_source),
        )
        stub = StubV2Eval(
            mock_v2_ei_tr_eval_config,
            run_config=mock_run_config.run_config_properties,
        )
        mock_output = TaskRun(
            input="test input",
            output=TaskOutput(output="fresh output", source=data_source),
        )
        with patch(
            "kiln_ai.adapters.eval.base_eval.adapter_for_task"
        ) as mock_adapter_for_task:
            mock_adapter = AsyncMock()
            mock_adapter.invoke = AsyncMock(return_value=mock_output)
            mock_adapter_for_task.return_value = mock_adapter
            result = await stub.run_task(tr)

        assert result == mock_output
        mock_adapter.invoke.assert_awaited_once_with("test input")


# -------------------------------------------------------------------
# SkippedReason validity: all runner skip paths emit valid enum values
# -------------------------------------------------------------------
class TestRunnerSkipReasonsAreValidEnumMembers:
    """Verify every skip reason ``eval_runner.py`` writes is a valid enum member.

    ``EvalRun.skipped_reason`` is a plain ``str`` for forward-compat, so nothing at the
    model layer rejects a reason that is not a ``SkippedReason``. The runner's own skip
    helper takes a typed ``SkippedReason``, which covers its two call sites; this catches
    a future one that goes around the helper with a bare string.
    """

    def test_all_hardcoded_skip_reasons_are_valid(self):
        import ast
        import inspect

        from kiln_ai.adapters.eval import eval_runner

        tree = ast.parse(inspect.getsource(eval_runner))
        valid_values = {member.value for member in SkippedReason}

        enum_members = [
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "SkippedReason"
        ]
        assert enum_members, (
            "Expected eval_runner to name at least one SkippedReason member. "
            "Test may need updating if the runner was refactored."
        )
        for member in enum_members:
            assert member in SkippedReason.__members__, (
                f"eval_runner uses SkippedReason.{member}, which is not a member"
            )

        for node in ast.walk(tree):
            if not isinstance(node, ast.keyword) or node.arg != "skipped_reason":
                continue
            if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str
            ):
                assert node.value.value in valid_values, (
                    f'eval_runner uses raw string skipped_reason="{node.value.value}" '
                    "which is not a valid SkippedReason value — "
                    "use SkippedReason.<member>.value instead"
                )


# ── V1 Legacy Runner Coexistence Guards ──────────────────────────────


class TestV1LegacyRunnerCoexistence:
    """Verify V1 eval configs dispatch through the legacy runner path.

    Guards against V2 additions accidentally misrouting V1 configs.
    Complements the model-layer tests in TestV1EvalConfigCoexistence (42050a2).
    """

    @pytest.mark.asyncio
    async def test_v1_g_eval_dispatches_through_legacy_runner(
        self,
        mock_eval_runner,
        mock_task,
        mock_eval_config,
        mock_run_config,
        data_source,
    ):
        task_run = TaskRun(
            parent=mock_task,
            input="test input",
            input_source=data_source,
            output=TaskOutput(output="test output"),
        )
        task_run.save_to_file()

        job = EvalJob(
            item=task_run,
            task_run_config=mock_run_config,
            type="task_run_eval",
            eval_config=mock_eval_config,
        )

        assert mock_eval_config.config_type == EvalConfigType.g_eval

        mock_scores: EvalScores = {"accuracy": 0.9}

        class LegacyStubEval(BaseEval):
            async def run_task_and_eval(self, eval_job_item: TaskRun):
                return (
                    TaskRun(
                        input=eval_job_item.input,
                        input_source=data_source,
                        output=TaskOutput(output="legacy output"),
                    ),
                    mock_scores,
                    None,
                )

        with patch(
            "kiln_ai.adapters.eval.eval_runner.legacy_eval_adapter_from_type",
            return_value=lambda *args, **kwargs: LegacyStubEval(*args, **kwargs),
        ) as mock_dispatch:
            success = await mock_eval_runner.run_job(job)

        assert success is True
        mock_dispatch.assert_called_once_with(mock_eval_config)

        runs = mock_eval_config.runs()
        assert len(runs) == 1
        saved = runs[0]
        assert saved.dataset_id == task_run.id
        assert saved.eval_input_id is None
        assert saved.skipped_reason is None
        assert saved.scores == mock_scores
        assert saved.output == "legacy output"
        assert saved.eval_config_eval is False

    @pytest.mark.asyncio
    async def test_v1_config_without_config_type_key_runs_through_legacy_runner(
        self, mock_eval_runner, mock_task, mock_eval, mock_run_config, data_source
    ):
        raw = {
            "name": "No Config Type Key",
            "model_name": "gpt-4",
            "model_provider": "openai",
            "properties": {"eval_steps": ["step1"]},
        }
        config = EvalConfig.model_validate(raw)
        config.parent = mock_eval
        config.save_to_file()

        assert config.config_type == EvalConfigType.g_eval

        task_run = TaskRun(
            parent=mock_task,
            input="hello",
            input_source=data_source,
            output=TaskOutput(output="world"),
        )
        task_run.save_to_file()

        job = EvalJob(
            item=task_run,
            task_run_config=mock_run_config,
            type="task_run_eval",
            eval_config=config,
        )

        mock_scores: EvalScores = {"accuracy": 0.85}

        class LegacyStubEval(BaseEval):
            async def run_task_and_eval(self, eval_job_item: TaskRun):
                return (
                    TaskRun(
                        input=eval_job_item.input,
                        input_source=data_source,
                        output=TaskOutput(output="from default config_type"),
                    ),
                    mock_scores,
                    None,
                )

        with patch(
            "kiln_ai.adapters.eval.eval_runner.legacy_eval_adapter_from_type",
            return_value=lambda *args, **kwargs: LegacyStubEval(*args, **kwargs),
        ) as mock_dispatch:
            success = await mock_eval_runner.run_job(job)

        assert success is True
        mock_dispatch.assert_called_once_with(config)

        runs = config.runs()
        assert len(runs) == 1
        saved = runs[0]
        assert saved.dataset_id == task_run.id
        assert saved.eval_input_id is None
        assert saved.output == "from default config_type"
        assert saved.skipped_reason is None

    @pytest.mark.asyncio
    async def test_v1_config_with_type_key_in_properties_not_misrouted_at_runner(
        self, mock_eval_runner, mock_task, mock_eval, mock_run_config, data_source
    ):
        config = EvalConfig(
            name="Type Key Collision",
            config_type=EvalConfigType.g_eval,
            model_name="gpt-4",
            model_provider="openai",
            properties={"eval_steps": ["s1"], "type": "exact_match"},
            parent=mock_eval,
        )
        config.save_to_file()

        assert config.config_type == EvalConfigType.g_eval
        assert isinstance(config.properties, dict)
        assert config.properties["type"] == "exact_match"

        task_run = TaskRun(
            parent=mock_task,
            input="collision input",
            input_source=data_source,
            output=TaskOutput(output="collision output"),
        )
        task_run.save_to_file()

        job = EvalJob(
            item=task_run,
            task_run_config=mock_run_config,
            type="task_run_eval",
            eval_config=config,
        )

        mock_scores: EvalScores = {"accuracy": 0.75}

        class LegacyStubEval(BaseEval):
            async def run_task_and_eval(self, eval_job_item: TaskRun):
                return (
                    TaskRun(
                        input=eval_job_item.input,
                        input_source=data_source,
                        output=TaskOutput(output="legacy, not v2"),
                    ),
                    mock_scores,
                    None,
                )

        with patch(
            "kiln_ai.adapters.eval.eval_runner.legacy_eval_adapter_from_type",
            return_value=lambda *args, **kwargs: LegacyStubEval(*args, **kwargs),
        ) as mock_dispatch:
            success = await mock_eval_runner.run_job(job)

        assert success is True
        mock_dispatch.assert_called_once_with(config)

        runs = config.runs()
        assert len(runs) == 1
        saved = runs[0]
        assert saved.output == "legacy, not v2"
        assert saved.eval_input_id is None
        assert saved.skipped_reason is None


# -------------------------------------------------------------------
# V2 multi-turn synthetic re-drive tests
# -------------------------------------------------------------------


@pytest.fixture
def mock_v2_redrive_eval(mock_task):
    """EvalInput-sourced full_trace eval — the shape the builder saves for
    multi-turn. Drive settings live on the items, not the eval."""
    eval = Eval(
        id="v2_redrive_eval",
        name="v2 redrive eval",
        description="multi-turn re-drive eval",
        splits={"test": EvalInputSplit(filter_id="all")},
        eval_configs_filter_id="all",
        evaluation_data_type=EvalDataType.full_trace,
        output_scores=[
            EvalOutputScore(
                name="Accuracy",
                instruction="Check if the output is accurate",
                type=TaskOutputRatingType.pass_fail,
            ),
        ],
        parent=mock_task,
    )
    eval.save_to_file()
    return eval


@pytest.fixture
def mock_v2_redrive_config(mock_v2_redrive_eval):
    eval_config = EvalConfig(
        name="v2 redrive config",
        config_type=EvalConfigType.v2,
        properties=ExactMatchProperties(expected_value="reply"),
        parent=mock_v2_redrive_eval,
    )
    eval_config.save_to_file()
    return eval_config


@pytest.fixture
def multi_turn_eval_input(mock_task):
    """A stamped multi-turn item: persona, seed, and drive config together
    make it the self-contained recipe the runner re-drives from."""
    ei = EvalInput(
        id="ei_redrive",
        data=MultiTurnSyntheticEvalInputData(
            first_message=UserMessage(text="opening message"),
            synthetic_user_info=SyntheticUserInfo(
                persona="frustrated customer",
                goal="get a refund",
                behavior_guidance="be polite then escalate",
            ),
            drive_config=MultiTurnDriveConfig(
                model_name="claude_4_5_haiku",
                model_provider="openrouter",
                turns=3,
            ),
        ),
        parent=mock_task,
    )
    ei.save_to_file()
    return ei


def _fresh_leaf(
    task: Task,
    data_source: DataSource,
    su_usage: Usage | None = None,
    cumulative_usage: MessageUsage | None = None,
    trace: list[ChatCompletionMessageParam] | None = MULTI_TURN_TRACE,
    chain_length: int = 3,
) -> DriveCaseResult:
    """The in-memory DriveCaseResult drive_case_for_eval would return:
    an id-less, trace-carrying, never-saved leaf plus the SU-side spend.

    `chain_length` is how many turns the drive ran — the real loop appends one
    TaskRun per turn, leaf last. It defaults to the `multi_turn_eval_input`
    fixture's three, so the result reads as a drive that used its whole turn
    ceiling; a shorter chain is how the runner recognises a conversation the
    synthetic user chose to end. Only the leaf carries the trace and the usage,
    exactly as in a real drive.

    `su_usage` defaults to None — the shape a drive whose provider reported
    nothing produces, which is what the tests that don't care about SU spend
    want."""
    leaf = TaskRun(
        input="opening message",
        input_source=data_source,
        output=TaskOutput(output="fresh reply", source=data_source),
        trace=trace,
        cumulative_usage=cumulative_usage,
        parent=task,
    )
    leaf.id = None
    earlier = [leaf.model_copy(deep=True) for _ in range(max(chain_length - 1, 0))]
    return DriveCaseResult(chain=[*earlier, leaf], su_usage=su_usage)


class TestRunV2MultiTurnRedrive:
    @pytest.mark.asyncio
    async def test_redrives_persists_one_standalone_trace_and_points_at_it(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        job = EvalJob(
            item=multi_turn_eval_input,
            eval_config=mock_v2_redrive_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        stub = RecordingStubV2Eval(mock_v2_redrive_config)
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=stub,
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(return_value=_fresh_leaf(mock_task, data_source)),
            ) as mock_drive,
        ):
            result = await runner.run_job(job)

        assert result is True
        # The drive got the seed, the typed persona, the item's drive config
        # as the customer, and the job's run config as the agent.
        drive_kwargs = mock_drive.await_args.kwargs
        assert drive_kwargs["seed_prompt"] == "opening message"
        assert drive_kwargs["synthetic_user_info"].persona == "frustrated customer"
        assert drive_kwargs["turns"] == 3
        assert drive_kwargs["su_driver_config"].model_name == "claude_4_5_haiku"
        assert (
            drive_kwargs["target_run_config"].model_name
            == mock_run_config.run_config_properties.model_name
        )

        # The judge saw the FRESH conversation, not stored data.
        assert len(stub.seen_inputs) == 1
        assert stub.seen_inputs[0].final_message == "fresh reply"
        assert stub.seen_inputs[0].trace == MULTI_TURN_TRACE
        assert stub.seen_inputs[0].task_input == "opening message"

        # Exactly ONE TaskRun persisted for the whole drive: standalone
        # (childless), stamped as an eval trace, holding the full conversation
        # and filing itself under the item + run config it was driven for.
        all_runs = mock_task.runs(
            readonly=True, include_eval_generated=True, include_intermediate_runs=True
        )
        assert len(all_runs) == 1
        trace = all_runs[0]
        assert trace.parent_task_run_id is None
        assert trace.eval_source == EvalItemSource(
            source_type="eval_input", source_id="ei_redrive"
        )
        assert trace.output.source.run_config_id == mock_run_config.id
        assert trace.trace == MULTI_TURN_TRACE
        assert trace.input == "opening message"
        assert trace.output.output == "fresh reply"
        # Hidden from dataset surfaces, like every eval trace.
        assert mock_task.runs(readonly=True) == []

        runs = mock_v2_redrive_config.runs(readonly=True)
        assert len(runs) == 1
        saved = runs[0]
        assert saved.eval_input_id == "ei_redrive"
        assert saved.dataset_id is None
        assert saved.eval_config_eval is False
        assert saved.task_run_config_id == mock_run_config.id
        assert saved.scores == {"accuracy": 1.0}
        assert saved.skipped_reason is None
        # Pointer record: the conversation lives on the TaskRun, never inline.
        assert saved.scored_run_id == trace.id
        assert saved.input is None
        assert saved.output is None
        assert saved.task_run_trace is None
        assert saved.task_run_usage is None

    @pytest.mark.asyncio
    async def test_a_second_judge_reuses_the_driven_conversation(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_eval,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        """Two judges over one item + run config pay for ONE drive: the second
        scores the conversation the first persisted."""
        second_config = EvalConfig(
            name="v2 redrive config 2",
            config_type=EvalConfigType.v2,
            properties=ExactMatchProperties(expected_value="fresh reply"),
            parent=mock_v2_redrive_eval,
        )
        second_config.save_to_file()
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config, second_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                side_effect=lambda config, *args, **kwargs: StubV2Eval(config),
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(return_value=_fresh_leaf(mock_task, data_source)),
            ) as mock_drive,
        ):
            for eval_config in (mock_v2_redrive_config, second_config):
                job = EvalJob(
                    item=multi_turn_eval_input,
                    eval_config=eval_config,
                    type="task_run_eval",
                    task_run_config=mock_run_config,
                )
                assert await runner.run_job(job) is True

        assert mock_drive.await_count == 1
        (trace,) = eval_traces(mock_task)
        (first,) = mock_v2_redrive_config.runs(readonly=True)
        (second,) = second_config.runs(readonly=True)
        assert first.scored_run_id == trace.id
        assert second.scored_run_id == trace.id

    @pytest.mark.asyncio
    async def test_a_fresh_runner_finds_the_persisted_conversation(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        """The identity postcondition: the persisted run's own stored fields
        round-trip to the key it was driven for, so a runner built later (a
        fresh index seeded from disk) reuses it instead of re-driving."""

        def make_job():
            return EvalJob(
                item=multi_turn_eval_input,
                eval_config=mock_v2_redrive_config,
                type="task_run_eval",
                task_run_config=mock_run_config,
            )

        def make_runner():
            return EvalRunner(
                eval_configs=[mock_v2_redrive_config],
                run_configs=[mock_run_config],
                eval_run_type="task_run_eval",
                split=_test_split([mock_v2_redrive_config]),
            )

        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                side_effect=lambda config, *args, **kwargs: StubV2Eval(config),
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(return_value=_fresh_leaf(mock_task, data_source)),
            ) as mock_drive,
        ):
            assert await make_runner().run_job(make_job()) is True
            assert mock_drive.await_count == 1
            assert await make_runner().run_job(make_job()) is True
            assert mock_drive.await_count == 1

        (trace,) = eval_traces(mock_task)
        first, second = mock_v2_redrive_config.runs(readonly=True)
        assert first.scored_run_id == trace.id
        assert second.scored_run_id == trace.id

    @pytest.mark.asyncio
    async def test_retry_after_a_judge_failure_rescores_the_persisted_trace(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        """The conversation is durable before scoring, so a judge that blows up
        costs a re-score on retry, never a second drive."""
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )

        def make_job():
            return EvalJob(
                item=multi_turn_eval_input,
                eval_config=mock_v2_redrive_config,
                type="task_run_eval",
                task_run_config=mock_run_config,
            )

        class ExplodingJudge(StubV2Eval):
            async def evaluate(self, eval_input):
                raise RuntimeError("judge crashed")

        with patch(
            "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
            new=AsyncMock(return_value=_fresh_leaf(mock_task, data_source)),
        ) as mock_drive:
            with patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=ExplodingJudge(mock_v2_redrive_config),
            ):
                with pytest.raises(RuntimeError, match="judge crashed"):
                    await runner.run_job(make_job())
            # The failed attempt scored nothing but kept the conversation.
            assert mock_v2_redrive_config.runs(readonly=True) == []
            (trace,) = eval_traces(mock_task)

            with patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=StubV2Eval(mock_v2_redrive_config),
            ):
                assert await runner.run_job(make_job()) is True

        assert mock_drive.await_count == 1
        (saved,) = mock_v2_redrive_config.runs(readonly=True)
        assert saved.scored_run_id == trace.id
        assert saved.scores == {"accuracy": 1.0}

    @pytest.mark.asyncio
    async def test_unstamped_item_skips(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
    ):
        """An item without a stamped drive config has no customer to re-drive
        with — clean typed skip naming the fix, no drive attempted."""
        ei = EvalInput(
            id="ei_unstamped",
            data=MultiTurnSyntheticEvalInputData(
                first_message=UserMessage(text="opening message"),
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
            ),
            parent=mock_task,
        )
        ei.save_to_file()
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        job = EvalJob(
            item=ei,
            eval_config=mock_v2_redrive_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=StubV2Eval(mock_v2_redrive_config),
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(),
            ) as mock_drive,
        ):
            result = await runner.run_job(job)

        assert result is True
        mock_drive.assert_not_awaited()
        runs = mock_v2_redrive_config.runs(readonly=True)
        assert len(runs) == 1
        saved = runs[0]
        assert saved.skipped_reason == SkippedReason.missing_drive_config.value
        assert saved.eval_input_id == "ei_unstamped"
        assert saved.scores == {}
        assert saved.output is None
        assert saved.skipped_detail == (
            "This item has no synthetic user configuration. "
            "Create a new batch to replace it."
        )
        # Skipped before any drive: nothing to point at, no inline copy either.
        assert saved.scored_run_id is None
        assert saved.input is None
        assert eval_traces(mock_task) == []

    @pytest.mark.asyncio
    async def test_missing_first_message_skips(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
    ):
        """No seed message → nothing to open the conversation with."""
        ei = EvalInput(
            id="ei_no_seed",
            data=MultiTurnSyntheticEvalInputData(
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
                drive_config=MultiTurnDriveConfig(
                    model_name="claude_4_5_haiku",
                    model_provider="openrouter",
                    turns=3,
                ),
            ),
            parent=mock_task,
        )
        ei.save_to_file()
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        job = EvalJob(
            item=ei,
            eval_config=mock_v2_redrive_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=StubV2Eval(mock_v2_redrive_config),
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(),
            ) as mock_drive,
        ):
            result = await runner.run_job(job)

        assert result is True
        mock_drive.assert_not_awaited()
        runs = mock_v2_redrive_config.runs(readonly=True)
        saved = next(r for r in runs if r.eval_input_id == "ei_no_seed")
        assert saved.skipped_reason == SkippedReason.incompatible_input_shape.value
        assert "first_message" in saved.skipped_detail
        assert saved.scored_run_id is None
        assert eval_traces(mock_task) == []

    @pytest.mark.asyncio
    async def test_adapter_skip_still_names_the_driven_trace(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        """A judge-side skip after the drive: the conversation was paid for and
        persisted, so the skip record points at it for the next judge to reuse."""
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        job = EvalJob(
            item=multi_turn_eval_input,
            eval_config=mock_v2_redrive_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=SkippingStubV2Eval(mock_v2_redrive_config),
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(return_value=_fresh_leaf(mock_task, data_source)),
            ),
        ):
            result = await runner.run_job(job)

        assert result is True
        (trace,) = eval_traces(mock_task)
        runs = mock_v2_redrive_config.runs(readonly=True)
        assert len(runs) == 1
        saved = runs[0]
        assert saved.skipped_reason == SkippedReason.extraction_failed.value
        assert saved.scored_run_id == trace.id
        assert saved.output is None
        assert saved.task_run_trace is None

    @pytest.mark.asyncio
    async def test_transient_drive_error_classifies_retryable(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
    ):
        """A provider rate limit mid-conversation arrives wrapped in
        KilnRunError (whose own message is genericized user-facing text).
        run_job must unwrap it, classify the failure as transient so the
        job runner retries the re-drive, keep the provider detail for the
        error log, and persist nothing for the failed attempt."""
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        job = EvalJob(
            item=multi_turn_eval_input,
            eval_config=mock_v2_redrive_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        mid_drive_error = wrapped_rate_limit_error(
            "rate limit exceeded, please try again later"
        )
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=RecordingStubV2Eval(mock_v2_redrive_config),
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(side_effect=mid_drive_error),
            ),
        ):
            with pytest.raises(RetryableError) as exc_info:
                await runner.run_job(job)

        assert "rate limit exceeded, please try again later" in str(exc_info.value)
        assert "Wait a moment" not in str(exc_info.value)
        assert len(mock_v2_redrive_config.runs(readonly=True)) == 0
        # A drive that raised persisted nothing, so the retry drives fresh.
        assert eval_traces(mock_task) == []


# -------------------------------------------------------------------
# Recoverable-skip re-collection tests
# -------------------------------------------------------------------


def _skip_run(
    eval_config: EvalConfig,
    run_config_id: str | None,
    reason: SkippedReason,
    eval_input_id: str | None = None,
    dataset_id: str | None = None,
) -> EvalRun:
    run = EvalRun(
        parent=eval_config,
        eval_input_id=eval_input_id,
        dataset_id=dataset_id,
        task_run_config_id=run_config_id,
        eval_config_eval=False,
        scores={},
        input="input",
        output=None,
        skipped_reason=reason.value,
        skipped_detail="test tombstone",
    )
    run.save_to_file()
    return run


class TestRecoverableSkipRecollection:
    """Recoverable skips (missing_drive_config / type_not_available / the multi-turn incompatible_input_shape skips older builds wrote) stop
    deduping once their blocking condition is lifted; while still blocked
    they keep deduping so re-triggers never write duplicate tombstones."""

    def test_missing_drive_config_recollected_once_item_stamped(
        self,
        mock_v2_redrive_config,
        mock_run_config,
        mock_eval_inputs,
        multi_turn_eval_input,
    ):
        # Tombstone written while the item carried no drive config; the item
        # is stamped now, so it must be collected again.
        _skip_run(
            mock_v2_redrive_config,
            mock_run_config.id,
            SkippedReason.missing_drive_config,
            eval_input_id="ei_redrive",
        )
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        collected = {j.item.id for j in runner.collect_tasks()}
        assert collected == {"ei_1", "ei_2", "ei_redrive"}

    def test_missing_drive_config_still_deduped_while_item_unstamped(
        self, mock_task, mock_v2_eval_config, mock_run_config, mock_eval_inputs
    ):
        # The item is still unstamped: the condition holds, so the tombstone
        # keeps deduping (no duplicate skip records per trigger).
        EvalInput(
            id="ei_unstamped_dedupe",
            data=MultiTurnSyntheticEvalInputData(
                first_message=UserMessage(text="hi"),
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
            ),
            parent=mock_task,
        ).save_to_file()
        _skip_run(
            mock_v2_eval_config,
            mock_run_config.id,
            SkippedReason.missing_drive_config,
            eval_input_id="ei_unstamped_dedupe",
        )
        runner = EvalRunner(
            eval_configs=[mock_v2_eval_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_eval_config]),
        )
        collected = {j.item.id for j in runner.collect_tasks()}
        assert collected == {"ei_1", "ei_2"}

    @pytest.mark.parametrize("available_now", [True, False])
    def test_type_not_available_follows_adapter_availability(
        self, mock_v2_eval_config, mock_run_config, mock_eval_inputs, available_now
    ):
        _skip_run(
            mock_v2_eval_config,
            mock_run_config.id,
            SkippedReason.type_not_available,
            eval_input_id="ei_1",
        )
        runner = EvalRunner(
            eval_configs=[mock_v2_eval_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_eval_config]),
        )
        with patch(
            "kiln_ai.adapters.eval.eval_runner.v2_eval_type_available",
            return_value=available_now,
        ):
            collected = {j.item.id for j in runner.collect_tasks()}
        assert collected == ({"ei_1", "ei_2"} if available_now else {"ei_2"})

    def test_incompatible_shape_recollected_when_the_item_carries_a_seed(
        self,
        mock_v2_redrive_config,
        mock_run_config,
        mock_eval_inputs,
        multi_turn_eval_input,
    ):
        # Earlier Kiln versions skipped every multi-turn synthetic item with
        # incompatible_input_shape before re-driving existed. This item has a
        # seed, so it can run now — the tombstone must not freeze it out.
        _skip_run(
            mock_v2_redrive_config,
            mock_run_config.id,
            SkippedReason.incompatible_input_shape,
            eval_input_id="ei_redrive",
        )
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        collected = {j.item.id for j in runner.collect_tasks()}
        assert collected == {"ei_1", "ei_2", "ei_redrive"}

    def test_incompatible_shape_still_deduped_while_the_item_has_no_seed(
        self, mock_task, mock_v2_redrive_config, mock_run_config, mock_eval_inputs
    ):
        # A seedless item is genuinely incompatible (items are immutable, the
        # seed never appears), so its tombstone keeps deduping.
        EvalInput(
            id="ei_seedless",
            data=MultiTurnSyntheticEvalInputData(
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
                drive_config=MultiTurnDriveConfig(
                    model_name="claude_4_5_haiku",
                    model_provider="openrouter",
                    turns=3,
                ),
            ),
            parent=mock_task,
        ).save_to_file()
        _skip_run(
            mock_v2_redrive_config,
            mock_run_config.id,
            SkippedReason.incompatible_input_shape,
            eval_input_id="ei_seedless",
        )
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        collected = {j.item.id for j in runner.collect_tasks()}
        assert collected == {"ei_1", "ei_2"}

    def test_terminal_skips_still_dedupe(
        self,
        mock_v2_redrive_config,
        mock_run_config,
        mock_eval_inputs,
        multi_turn_eval_input,
    ):
        # Non-recoverable skips are verdicts about the input itself — a
        # runnable item must not resurrect them.
        _skip_run(
            mock_v2_redrive_config,
            mock_run_config.id,
            SkippedReason.extraction_failed,
            eval_input_id="ei_redrive",
        )
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        collected = {j.item.id for j in runner.collect_tasks()}
        assert collected == {"ei_1", "ei_2"}

    def test_task_run_lane_recollects_recoverable_skips(
        self, mock_task, mock_v2_task_run_eval_config, mock_run_config, data_source
    ):
        # Same semantics on the TaskRun-sourced lane (eval_set_filter_id).
        task_run = TaskRun(
            input="test input",
            output=TaskOutput(output="out", source=data_source),
            parent=mock_task,
        )
        task_run.save_to_file()
        tombstone = _skip_run(
            mock_v2_task_run_eval_config,
            mock_run_config.id,
            SkippedReason.type_not_available,
            dataset_id=task_run.id,
        )
        runner = EvalRunner(
            eval_configs=[mock_v2_task_run_eval_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_task_run_eval_config]),
        )
        with patch(
            "kiln_ai.adapters.eval.eval_runner.v2_eval_type_available",
            return_value=True,
        ):
            jobs = runner.collect_tasks()
        assert {j.item.id for j in jobs} == {task_run.id}
        # The recovered tombstone rides the job (parity with the EvalInput
        # lane) so the replacement record can delete it instead of leaving
        # two records on one item.
        assert [t.id for t in jobs[0].superseded_tombstones] == [tombstone.id]


# -------------------------------------------------------------------
# KIL-749 regression: fresh generations are keyed on the dataset item
# -------------------------------------------------------------------


class TestFreshGenerationDatasetId:
    @pytest.mark.asyncio
    async def test_unsaved_fresh_run_does_not_crash_record_keys_on_item(
        self,
        mock_v2_task_run_eval_runner,
        mock_v2_task_run_eval_config,
        mock_run_config,
        data_source,
    ):
        """KIL-749: run_task returns an UNSAVED TaskRun (id None). Keying the
        record on it crashed EvalRun validation for every TaskRun-backed item;
        the record must key on the dataset item instead."""
        item = TaskRun(
            input="test input",
            output=TaskOutput(output="stored", source=data_source),
            parent=mock_v2_task_run_eval_runner.task,
        )
        item.save_to_file()
        fresh_source = data_source.model_copy(
            update={"run_config_id": mock_run_config.id}
        )
        fresh = TaskRun(
            input="test input",
            input_source=data_source,
            output=TaskOutput(output="hello", source=fresh_source),
            parent=mock_v2_task_run_eval_runner.task,
        )
        fresh.id = None

        job = EvalJob(
            item=item,
            eval_config=mock_v2_task_run_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        stub = StubV2Eval(mock_v2_task_run_eval_config)
        with (
            patch.object(stub, "run_task", return_value=fresh),
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=stub,
            ),
        ):
            assert await mock_v2_task_run_eval_runner.run_job(job) is True

        runs = mock_v2_task_run_eval_config.runs(readonly=True)
        assert len(runs) == 1
        assert runs[0].dataset_id == item.id
        # Dedup now recognizes the item as done.
        assert mock_v2_task_run_eval_runner.collect_tasks() == []


# -------------------------------------------------------------------
# Cost recording tests
# -------------------------------------------------------------------


class TestEvalRunUsageRecording:
    @pytest.mark.asyncio
    async def test_drive_records_agent_usage_and_full_su_usage(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        drive_result = _fresh_leaf(
            mock_task,
            data_source,
            su_usage=Usage(
                input_tokens=3548, output_tokens=61, total_tokens=3609, cost=0.25
            ),
            cumulative_usage=MessageUsage(
                input_tokens=100, output_tokens=50, total_tokens=150, cost=1.0
            ),
        )
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        job = EvalJob(
            item=multi_turn_eval_input,
            eval_config=mock_v2_redrive_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=StubV2Eval(mock_v2_redrive_config),
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(return_value=drive_result),
            ) as mock_drive,
        ):
            await runner.run_job(job)

        # One job pays for exactly one drive.
        assert mock_drive.await_count == 1
        saved = mock_v2_redrive_config.runs(readonly=True)[0]
        # Pointer record: the spend lives on the persisted trace, never inline.
        assert saved.task_run_usage is None
        (trace,) = eval_traces(mock_task)
        assert saved.scored_run_id == trace.id
        # `usage` is honestly assistant-only, at conversation totals (the leaf's
        # cumulative recompute, not its last-turn-only usage).
        assert trace.usage is not None
        assert trace.usage.cost == pytest.approx(1.0)
        assert trace.usage.input_tokens == 100
        assert trace.usage.total_tokens == 150
        # The synthetic-user driver's spend rides its own field, with its own
        # tokens — the agent's counts above are a different model on a different
        # provider, so a cost-only SU record could be reconciled against neither
        # invoice and split per model not at all.
        assert trace.synthetic_user_usage is not None
        assert trace.synthetic_user_usage.cost == pytest.approx(0.25)
        assert trace.synthetic_user_usage.input_tokens == 3548
        assert trace.synthetic_user_usage.output_tokens == 61
        assert trace.synthetic_user_usage.total_tokens == 3609
        assert trace.cumulative_usage == MessageUsage(
            input_tokens=100, output_tokens=50, total_tokens=150, cost=1.0
        )

    @pytest.mark.asyncio
    async def test_zero_cost_drive_records_no_synthetic_user_usage(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        """None, not a zero-cost Usage: the rollup's null-tolerant blend would
        read a zero object as a real 0.0 cost and count it in averages."""
        drive_result = _fresh_leaf(mock_task, data_source, su_usage=None)
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        job = EvalJob(
            item=multi_turn_eval_input,
            eval_config=mock_v2_redrive_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=StubV2Eval(mock_v2_redrive_config),
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(return_value=drive_result),
            ),
        ):
            await runner.run_job(job)

        (trace,) = eval_traces(mock_task)
        assert trace.synthetic_user_usage is None

    @pytest.mark.asyncio
    async def test_eval_input_fresh_generation_records_usage(
        self,
        mock_v2_eval_config,
        mock_run_config,
        mock_eval_inputs,
        data_source,
    ):
        fresh_source = data_source.model_copy(
            update={"run_config_id": mock_run_config.id}
        )
        fresh = TaskRun(
            input="What is 2+2?",
            input_source=data_source,
            output=TaskOutput(output="hello", source=fresh_source),
            usage=Usage(cost=0.3, total_tokens=42),
            parent=mock_v2_eval_config.parent_eval().parent_task(),
        )
        fresh.id = None
        runner = EvalRunner(
            eval_configs=[mock_v2_eval_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_eval_config]),
        )
        job = EvalJob(
            item=mock_eval_inputs[0],
            eval_config=mock_v2_eval_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        stub = StubV2Eval(mock_v2_eval_config)
        with (
            patch.object(stub, "run_task", return_value=fresh),
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=stub,
            ),
        ):
            await runner.run_job(job)

        saved = mock_v2_eval_config.runs(readonly=True)[0]
        # Pointer record: the generation's usage lives on the persisted trace,
        # never inline on the score record.
        assert saved.task_run_usage is None
        assert saved.scored_run_id is not None
        task = mock_v2_eval_config.parent_eval().parent_task()
        trace = next(
            r
            for r in task.runs(readonly=True, include_eval_generated=True)
            if r.id == saved.scored_run_id
        )
        assert trace.usage is not None
        assert trace.usage.cost == pytest.approx(0.3)
        assert trace.usage.total_tokens == 42


class TestConversationUsage:
    def test_none_when_nothing_reported(self, mock_task, data_source):
        chain = _fresh_leaf(mock_task, data_source).chain
        assert _conversation_usage(chain) is None

    def test_tokens_and_cost_come_from_the_cumulative_recompute(
        self, mock_task, data_source
    ):
        chain = _fresh_leaf(
            mock_task,
            data_source,
            cumulative_usage=MessageUsage(cost=2.0, total_tokens=10),
        ).chain
        # The leaf's own usage is last-turn-only and must not win.
        chain[-1].usage = Usage(cost=0.5, total_tokens=3)
        usage = _conversation_usage(chain)
        assert usage is not None
        assert usage.cost == pytest.approx(2.0)
        assert usage.total_tokens == 10

    def test_falls_back_to_summing_the_trace_when_cumulative_is_missing(
        self, mock_task, data_source
    ):
        chain = _fresh_leaf(mock_task, data_source).chain
        chain[-1].trace = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "a",
                "usage": MessageUsage(cost=0.25, total_tokens=4),
            },
            {
                "role": "assistant",
                "content": "b",
                "usage": MessageUsage(cost=0.75, total_tokens=6),
            },
        ]
        usage = _conversation_usage(chain)
        assert usage is not None
        assert usage.cost == pytest.approx(1.0)
        assert usage.total_tokens == 10

    def test_latency_sums_every_turns_accumulator(self, mock_task, data_source):
        result = _fresh_leaf(
            mock_task,
            data_source,
            cumulative_usage=MessageUsage(cost=1.0),
        )
        leaf = result.chain[-1]
        leaf.usage = Usage(total_llm_latency_ms=300)
        earlier = leaf.model_copy(update={"usage": Usage(total_llm_latency_ms=200)})
        no_usage = leaf.model_copy(update={"usage": None})
        usage = _conversation_usage([earlier, no_usage, leaf])
        assert usage is not None
        assert usage.total_llm_latency_ms == 500
        assert usage.cost == pytest.approx(1.0)

    def test_latency_is_none_when_no_turn_reported_one(self, mock_task, data_source):
        chain = _fresh_leaf(
            mock_task,
            data_source,
            cumulative_usage=MessageUsage(cost=1.0),
        ).chain
        usage = _conversation_usage(chain)
        assert usage is not None
        assert usage.total_llm_latency_ms is None


# -------------------------------------------------------------------
# Up-front multi-turn drive validation tests
# -------------------------------------------------------------------


class TestValidateMultiTurnDriveReadiness:
    def test_single_turn_split_is_noop(
        self, mock_task, mock_v2_eval_config, mock_eval_inputs
    ):
        """A split with only single-turn items never re-drives, so it must
        not acquire agent-run-config checks — an MCP run config that would
        fail the multi-turn check is fine here."""
        mcp_rc = TaskRunConfig(
            name="mcp single turn config",
            description="not an agent",
            run_config_properties=McpRunConfigProperties(
                tool_reference=MCPToolReference(
                    tool_id="mcp::local::server1::tool1",
                ),
            ),
            parent=mock_task,
        )
        mcp_rc.save_to_file()
        runner = EvalRunner(
            eval_configs=[mock_v2_eval_config],
            run_configs=[mcp_rc],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_eval_config]),
        )
        runner.validate_multi_turn_drive_readiness()

    def test_valid_setup_passes(
        self, mock_v2_redrive_config, mock_run_config, multi_turn_eval_input
    ):
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        runner.validate_multi_turn_drive_readiness()

    def test_non_agent_run_config_rejected(
        self, mock_task, mock_v2_redrive_config, multi_turn_eval_input
    ):
        mcp_rc = TaskRunConfig(
            name="mcp config",
            description="not an agent",
            run_config_properties=McpRunConfigProperties(
                tool_reference=MCPToolReference(
                    tool_id="mcp::local::server1::tool1",
                ),
            ),
            parent=mock_task,
        )
        mcp_rc.save_to_file()
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mcp_rc],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        with pytest.raises(ValueError, match="mcp config"):
            runner.validate_multi_turn_drive_readiness()

    def test_bad_su_provider_rejected(
        self, mock_task, mock_run_config, mock_v2_redrive_config
    ):
        """A stamped item whose provider this build doesn't know fails every
        re-drive of that item — surfaced up front."""
        EvalInput(
            id="ei_bad_provider",
            data=MultiTurnSyntheticEvalInputData(
                first_message=UserMessage(text="hi"),
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
                drive_config=MultiTurnDriveConfig(
                    model_name="claude_4_5_haiku",
                    model_provider="not_a_real_provider",
                    turns=3,
                ),
            ),
            parent=mock_task,
        ).save_to_file()
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        with pytest.raises(ValueError, match="not_a_real_provider"):
            runner.validate_multi_turn_drive_readiness()

    def test_all_items_unstamped_fails_up_front(
        self, mock_task, mock_run_config, mock_v2_redrive_config
    ):
        """Nothing could run, so the user gets one clear error instead of a
        page of identical per-item skips."""
        for item_id in ("ei_bare_1", "ei_bare_2"):
            EvalInput(
                id=item_id,
                data=MultiTurnSyntheticEvalInputData(
                    first_message=UserMessage(text="hi"),
                    synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
                ),
                parent=mock_task,
            ).save_to_file()
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        with pytest.raises(ValueError, match="synthetic user configuration"):
            runner.validate_multi_turn_drive_readiness()

    def test_partially_stamped_split_passes(
        self, mock_task, mock_run_config, mock_v2_redrive_config, multi_turn_eval_input
    ):
        """One stamped item is enough to run; the unstamped one records its
        own per-item skip at execution instead of blocking the batch."""
        EvalInput(
            id="ei_bare",
            data=MultiTurnSyntheticEvalInputData(
                first_message=UserMessage(text="hi"),
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
            ),
            parent=mock_task,
        ).save_to_file()
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        runner.validate_multi_turn_drive_readiness()

    def test_two_message_chain_of_thought_run_config_rejected(
        self, mock_task, mock_v2_redrive_config, multi_turn_eval_input
    ):
        """A chain of thought prompt on a model with no reasoning step of its own is
        served by asking the model to think, then sending a second user message asking
        for the final answer. That message is indistinguishable from a real user turn,
        so the conversation can never be driven — refused before the first paid drive
        rather than after three of them."""
        cot_rc = TaskRunConfig(
            name="chain of thought config",
            description="thinking instructions on a model without native reasoning",
            run_config_properties=KilnAgentRunConfigProperties(
                model_name="gpt_4o",
                model_provider_name=ModelProviderName.openai,
                prompt_id="simple_chain_of_thought_prompt_builder",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            parent=mock_task,
        )
        cot_rc.save_to_file()
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[cot_rc],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        with patch(
            "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
            new=AsyncMock(),
        ) as drive:
            with pytest.raises(
                ValueError, match="no reasoning step of its own"
            ) as raised:
                runner.validate_multi_turn_drive_readiness()
        assert "chain of thought config" in str(raised.value)
        drive.assert_not_awaited()

    def test_two_message_chain_of_thought_rejected_without_provider_keys(
        self, mock_task, mock_v2_redrive_config, multi_turn_eval_input
    ):
        """The preflight reads the built-in model table, not the user's credentials,
        so a machine with no provider keys refuses the same config a keyed one does.
        A credentialed lookup would raise on the missing key, and swallowing that
        error would let the two-message config through to the paid drives."""
        cot_rc = TaskRunConfig(
            name="chain of thought config",
            description="thinking instructions on a model without native reasoning",
            run_config_properties=KilnAgentRunConfigProperties(
                model_name="gpt_4o",
                model_provider_name=ModelProviderName.openai,
                prompt_id="simple_chain_of_thought_prompt_builder",
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            parent=mock_task,
        )
        cot_rc.save_to_file()
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[cot_rc],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        with patch(
            "kiln_ai.adapters.provider_tools.get_config_value", return_value=None
        ):
            with pytest.raises(ValueError, match="no reasoning step of its own"):
                runner.validate_multi_turn_drive_readiness()

    @pytest.mark.parametrize(
        "model_name, provider, prompt_id",
        [
            # A reasoning model answers a chain of thought prompt in its own format, so
            # a turn is still one user message.
            (
                "gpt_oss_120b",
                ModelProviderName.cerebras,
                "simple_chain_of_thought_prompt_builder",
            ),
            # No thinking instructions at all: single turn whatever the model is.
            ("gpt_4o", ModelProviderName.openai, "simple_prompt_builder"),
        ],
    )
    def test_one_message_per_turn_run_configs_pass(
        self,
        mock_task,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        model_name: str,
        provider: ModelProviderName,
        prompt_id: str,
    ):
        """Only the two-message strategies are refused: blocking every chain of thought
        prompt would take reasoning models with it."""
        rc = TaskRunConfig(
            name="one message per turn config",
            description="drivable",
            run_config_properties=KilnAgentRunConfigProperties(
                model_name=model_name,
                model_provider_name=provider,
                prompt_id=prompt_id,
                structured_output_mode=StructuredOutputMode.json_schema,
            ),
            parent=mock_task,
        )
        rc.save_to_file()
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[rc],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        with patch(
            "kiln_ai.adapters.provider_tools.get_config_value", return_value=None
        ):
            runner.validate_multi_turn_drive_readiness()


class TestSupersededTombstoneDeletion:
    @pytest.mark.asyncio
    async def test_replacement_run_deletes_tombstone(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        """Once a re-collected item persists its fresh record, the old
        tombstone is deleted — two records on one item would race in
        first-found read paths (the fresh score could be masked)."""
        tombstone = _skip_run(
            mock_v2_redrive_config,
            mock_run_config.id,
            SkippedReason.missing_drive_config,
            eval_input_id="ei_redrive",
        )
        tombstone_path = tombstone.path
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        jobs = runner.collect_tasks()
        job = next(j for j in jobs if j.item.id == "ei_redrive")
        assert [t.id for t in job.superseded_tombstones] == [tombstone.id]

        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=StubV2Eval(mock_v2_redrive_config),
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(return_value=_fresh_leaf(mock_task, data_source)),
            ),
        ):
            assert await runner.run_job(job) is True

        assert not tombstone_path.exists()
        runs = mock_v2_redrive_config.runs(readonly=True)
        assert len(runs) == 1
        assert runs[0].skipped_reason is None
        assert runs[0].eval_input_id == "ei_redrive"

    def test_still_blocked_tombstone_not_marked_superseded(
        self, mock_task, mock_v2_eval_config, mock_run_config, mock_eval_inputs
    ):
        """While the item stays unstamped, the tombstone dedupes: the item is
        not re-collected and no job carries the tombstone for deletion."""
        EvalInput(
            id="ei_still_unstamped",
            data=MultiTurnSyntheticEvalInputData(
                first_message=UserMessage(text="hi"),
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
            ),
            parent=mock_task,
        ).save_to_file()
        _skip_run(
            mock_v2_eval_config,
            mock_run_config.id,
            SkippedReason.missing_drive_config,
            eval_input_id="ei_still_unstamped",
        )
        runner = EvalRunner(
            eval_configs=[mock_v2_eval_config],
            run_configs=[mock_run_config],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_eval_config]),
        )
        jobs = runner.collect_tasks()
        assert not any(j.item.id == "ei_still_unstamped" for j in jobs)
        assert all(j.superseded_tombstones == [] for j in jobs)


class TestValidateReadinessSourceGating:
    def test_task_run_source_is_noop(
        self, mock_task, multi_turn_eval_input, data_source
    ):
        """A stored-TaskRun-sourced split never re-drives (chain leaves judge
        their stored trace), so validation must not block the run — even with
        run-config problems it would otherwise flag, and even while stamped
        multi-turn items exist elsewhere under the task."""
        TaskRun(
            parent=mock_task,
            input="stored chain",
            input_source=data_source,
            output=TaskOutput(output="out"),
            tags=["stored_tag"],
        ).save_to_file()
        eval = Eval(
            id="tr_source_eval",
            name="task run sourced eval",
            description="stored-trace eval",
            eval_set_filter_id="tag::stored_tag",
            eval_configs_filter_id="all",
            evaluation_data_type=EvalDataType.full_trace,
            output_scores=[
                EvalOutputScore(
                    name="Accuracy",
                    instruction="Check",
                    type=TaskOutputRatingType.pass_fail,
                ),
            ],
            parent=mock_task,
        )
        eval.save_to_file()
        config = EvalConfig(
            name="tr source cfg",
            config_type=EvalConfigType.v2,
            properties=ExactMatchProperties(expected_value="x"),
            parent=eval,
        )
        config.save_to_file()
        mcp_rc = TaskRunConfig(
            name="mcp stored config",
            description="not an agent",
            run_config_properties=McpRunConfigProperties(
                tool_reference=MCPToolReference(
                    tool_id="mcp::local::server1::tool1",
                ),
            ),
            parent=mock_task,
        )
        mcp_rc.save_to_file()
        runner = EvalRunner(
            eval_configs=[config],
            run_configs=[mcp_rc],
            eval_run_type="task_run_eval",
            split=_test_split([config]),
        )
        runner.validate_multi_turn_drive_readiness()

    def test_check_run_configs_false_skips_run_config_problems(
        self, mock_task, mock_v2_redrive_config, multi_turn_eval_input
    ):
        """With an un-hand-picked fleet (all_run_configs), one incompatible
        run config must not block the others — only drive-config problems
        (which block every job) still raise."""
        mcp_rc = TaskRunConfig(
            name="mcp fleet member",
            description="not an agent",
            run_config_properties=McpRunConfigProperties(
                tool_reference=MCPToolReference(
                    tool_id="mcp::local::server1::tool1",
                ),
            ),
            parent=mock_task,
        )
        mcp_rc.save_to_file()
        runner = EvalRunner(
            eval_configs=[mock_v2_redrive_config],
            run_configs=[mcp_rc],
            eval_run_type="task_run_eval",
            split=_test_split([mock_v2_redrive_config]),
        )
        runner.validate_multi_turn_drive_readiness(check_run_configs=False)
        with pytest.raises(ValueError, match="mcp fleet member"):
            runner.validate_multi_turn_drive_readiness()


class TestEvalRunnerSplitArgument:
    def test_task_run_eval_requires_a_split(self, mock_eval_config, mock_run_config):
        with pytest.raises(ValueError, match="requires a resolved split"):
            EvalRunner(
                eval_configs=[mock_eval_config],
                run_configs=[mock_run_config],
                eval_run_type="task_run_eval",
            )

    def test_eval_config_eval_rejects_a_split(
        self, mock_eval, mock_task, mock_eval_config
    ):
        split = resolve_split(mock_task, mock_eval, "test")
        assert split is not None
        with pytest.raises(ValueError, match="does not support a split"):
            EvalRunner(
                eval_configs=[mock_eval_config],
                run_configs=None,
                eval_run_type="eval_config_eval",
                split=split,
            )

    def test_task_run_eval_accepts_an_eval_input_backed_split(
        self, mock_v2_eval_config, mock_run_config, mock_eval_inputs
    ):
        runner = build_task_run_eval_runner([mock_v2_eval_config], [mock_run_config])
        assert runner.split is not None
        assert runner.split.source == "eval_input"
        assert runner.eval_run_type == "task_run_eval"

    def test_rejects_a_split_resolved_from_a_different_eval(
        self, mock_task, mock_eval, mock_v2_eval, mock_eval_config, mock_run_config
    ):
        """Before the runner took items, the item set came from the eval and this was
        unconstructible. It stays unconstructible because the split remembers where it
        came from — otherwise one eval's judges would score another's items in silence."""
        other_evals_split = resolve_split(mock_task, mock_v2_eval, "test")
        assert other_evals_split is not None
        assert mock_v2_eval.id != mock_eval.id

        with pytest.raises(ValueError, match="was resolved from eval") as exc_info:
            EvalRunner(
                eval_configs=[mock_eval_config],
                run_configs=[mock_run_config],
                eval_run_type="task_run_eval",
                split=other_evals_split,
            )

        assert mock_v2_eval.id in str(exc_info.value)
        assert mock_eval.id in str(exc_info.value)


# -------------------------------------------------------------------
# eval_config_eval is golden-scoped, so its items are always TaskRuns
# -------------------------------------------------------------------


class TestCollectTasksEvalConfigEval:
    def test_collects_only_task_runs_on_an_eval_input_backed_eval(
        self, mock_v2_runner, mock_task, mock_eval_inputs, data_source
    ):
        """The eval's test split is EvalInput-backed, but calibration scopes by the golden
        filter, which can only address TaskRuns. The EvalInputs are present precisely so a
        source-mode branch would wrongly collect them."""
        task_run = TaskRun(
            parent=mock_task,
            input="golden input",
            input_source=data_source,
            output=TaskOutput(output="golden output"),
        )
        task_run.save_to_file()

        jobs = mock_v2_runner.collect_tasks()

        assert [job.item.id for job in jobs] == [task_run.id]
        assert all(isinstance(job.item, TaskRun) for job in jobs)
        assert all(job.type == "eval_config_eval" for job in jobs)

    @pytest.mark.asyncio
    async def test_writes_no_skipped_runs_for_an_eval_input_backed_eval(
        self, mock_v2_runner, mock_v2_eval_config, mock_eval_inputs
    ):
        """Architecture 4.3. This used to be a *successful* run that persisted one junk
        EvalRun per EvalInput, so only the absence of records can catch a regression."""
        async for _ in mock_v2_runner.run():
            pass

        assert mock_v2_eval_config.runs(readonly=True) == []

    def test_no_golden_set_raises_at_construction(self, mock_task, mock_eval_inputs):
        """Not at collect time. These runners are driven by SSE endpoints, so a failure
        raised once the response generator is running arrives after a 200 and reaches the
        client as a dead stream. Construction is the last point a caller can turn it into
        a real error (architecture 4.3, functional spec 9)."""
        eval = Eval(
            id="no_golden",
            name="no golden",
            description="EvalInput-backed eval with no golden set",
            splits={"test": EvalInputSplit(filter_id="all")},
            output_scores=[
                EvalOutputScore(
                    name="Accuracy",
                    instruction="Check",
                    type=TaskOutputRatingType.pass_fail,
                ),
            ],
            parent=mock_task,
        )
        eval.save_to_file()
        eval_config = EvalConfig(
            name="no golden config",
            config_type=EvalConfigType.v2,
            properties=ExactMatchProperties(expected_value="4"),
            parent=eval,
        )
        eval_config.save_to_file()

        with pytest.raises(
            ValueError, match="has no golden set configured"
        ) as exc_info:
            EvalRunner(
                eval_configs=[eval_config],
                run_configs=None,
                eval_run_type="eval_config_eval",
            )

        assert "no_golden" in str(exc_info.value)


# -------------------------------------------------------------------
# run_job V2 dispatch tests
# -------------------------------------------------------------------


class TestCollectTasksOverArbitrarySplits:
    def test_eval_input_backed_split_collects_exactly_the_matching_inputs(
        self, mock_task, mock_eval_inputs, mock_run_config
    ):
        """Asserted on which items, not on a job count: functional spec 4.2's failure mode
        is a run that succeeds over the wrong item set."""
        eval = Eval(
            id="tag_backed",
            name="tag backed",
            description="EvalInput-backed test split, narrowed by tag",
            splits={"test": EvalInputSplit(filter_id="tag::math")},
            eval_configs_filter_id="all",
            output_scores=[
                EvalOutputScore(
                    name="Accuracy",
                    instruction="Check",
                    type=TaskOutputRatingType.pass_fail,
                ),
            ],
            parent=mock_task,
        )
        eval.save_to_file()
        eval_config = EvalConfig(
            name="tag config",
            config_type=EvalConfigType.v2,
            properties=ExactMatchProperties(expected_value="4"),
            parent=eval,
        )
        eval_config.save_to_file()

        jobs = build_task_run_eval_runner(
            [eval_config], [mock_run_config]
        ).collect_tasks()

        assert [job.item.id for job in jobs] == ["ei_1"]
        assert all(isinstance(job.item, EvalInput) for job in jobs)

    def test_a_non_test_split_is_collected_the_same_way(
        self, mock_eval, mock_task, mock_eval_config, mock_run_config, data_source
    ):
        """Nothing in the runner names 'test' any more — it works whatever split it's given."""
        items = {}
        for tag in ["test_tag", "val_tag"]:
            run = TaskRun(
                parent=mock_task,
                input=tag,
                input_source=data_source,
                output=TaskOutput(output=tag),
                tags=[tag],
            )
            run.save_to_file()
            items[tag] = run

        mock_eval.splits["test"] = TaskRunSplit(filter_id="tag::test_tag")
        mock_eval.splits["val"] = TaskRunSplit(filter_id="tag::val_tag")

        jobs = build_task_run_eval_runner(
            [mock_eval_config], [mock_run_config], split_name="val"
        ).collect_tasks()

        assert [job.item.id for job in jobs] == [items["val_tag"].id]

    def test_dedupe_keys_on_the_item_source_not_the_bare_id(
        self, mock_task, mock_v2_ei_tr_eval_config, mock_run_config
    ):
        """Ids come from one generator shared by every model type (functional spec 5.3), so
        a TaskRun and an EvalInput can collide. A bare-id dedupe would drop this job."""
        shared_id = "collide_1"
        TaskRun(
            id=shared_id,
            parent=mock_task,
            input="task run with the colliding id",
            input_source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "gpt-4",
                    "model_provider": "openai",
                    "adapter_name": "test_adapter",
                },
            ),
            output=TaskOutput(output="out"),
        ).save_to_file()
        eval_input = EvalInput(
            id=shared_id,
            data=SingleTurnEvalInputData(user_message=UserMessage(text="eval input")),
            parent=mock_task,
        )
        eval_input.save_to_file()

        EvalRun(
            parent=mock_v2_ei_tr_eval_config,
            dataset_id=shared_id,
            task_run_config_id=mock_run_config.id,
            eval_config_eval=False,
            scores={"accuracy": 1.0},
            input="task run with the colliding id",
            output="out",
        ).save_to_file()

        jobs = build_task_run_eval_runner(
            [mock_v2_ei_tr_eval_config], [mock_run_config]
        ).collect_tasks()

        assert [job.item.id for job in jobs] == [shared_id]
        assert isinstance(jobs[0].item, EvalInput)

    def test_tombstones_key_on_the_item_source_not_the_bare_id(
        self,
        mock_task,
        mock_v2_eval_input_task_run_eval,
        mock_v2_ei_tr_eval_config,
        mock_run_config,
    ):
        """A tombstone recorded against a TaskRun must not attach to (or
        dedupe) a job for an EvalInput that shares the bare id."""
        shared_id = "collide_2"
        TaskRun(
            id=shared_id,
            parent=mock_task,
            input="task run with the colliding id",
            input_source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "gpt-4",
                    "model_provider": "openai",
                    "adapter_name": "test_adapter",
                },
            ),
            output=TaskOutput(output="out"),
        ).save_to_file()
        EvalInput(
            id=shared_id,
            data=SingleTurnEvalInputData(user_message=UserMessage(text="eval input")),
            parent=mock_task,
        ).save_to_file()

        # Tombstone keyed to the TASK RUN store (dataset_id, not eval_input_id).
        tombstone = EvalRun(
            parent=mock_v2_ei_tr_eval_config,
            dataset_id=shared_id,
            task_run_config_id=mock_run_config.id,
            eval_config_eval=False,
            input="task run with the colliding id",
            output=None,
            scores={},
            skipped_reason=SkippedReason.missing_drive_config.value,
            skipped_detail="test tombstone",
        )
        tombstone.save_to_file()

        jobs = build_task_run_eval_runner(
            [mock_v2_ei_tr_eval_config], [mock_run_config]
        ).collect_tasks()

        ei_jobs = [j for j in jobs if j.item.id == shared_id]
        assert len(ei_jobs) == 1
        assert isinstance(ei_jobs[0].item, EvalInput)
        # The TaskRun-store tombstone must not attach to the EvalInput's job.
        assert ei_jobs[0].superseded_tombstones == []

    def test_overlapping_splits_reuse_already_scored_items(
        self, mock_eval, mock_task, mock_eval_config, mock_run_config, data_source
    ):
        """Dedupe keys on the item, not on the split, so an item scored under one split is
        not re-scored when it turns up in another."""
        shared = TaskRun(
            parent=mock_task,
            input="in both splits",
            input_source=data_source,
            output=TaskOutput(output="out"),
            tags=["test_tag", "val_tag"],
        )
        shared.save_to_file()
        val_only = TaskRun(
            parent=mock_task,
            input="val only",
            input_source=data_source,
            output=TaskOutput(output="out"),
            tags=["val_tag"],
        )
        val_only.save_to_file()

        mock_eval.splits["test"] = TaskRunSplit(filter_id="tag::test_tag")
        mock_eval.splits["val"] = TaskRunSplit(filter_id="tag::val_tag")

        EvalRun(
            parent=mock_eval_config,
            dataset_id=shared.id,
            task_run_config_id=mock_run_config.id,
            input="in both splits",
            output="out",
            scores={"accuracy": 1.0},
        ).save_to_file()

        jobs = build_task_run_eval_runner(
            [mock_eval_config], [mock_run_config], split_name="val"
        ).collect_tasks()

        assert [job.item.id for job in jobs] == [val_only.id]


def test_collect_tasks_ignores_runs_from_run_configs_not_being_evaluated(
    mock_eval, mock_task, data_source, mock_eval_config, mock_run_config
):
    """Scoring an item under run config A must not exclude it when B is evaluated later.

    The eval config accumulates runs for every run config ever compared, so most of what
    collect_tasks reads belongs to run configs this runner was not given.
    """
    mock_eval.splits["test"] = TaskRunSplit(filter_id="tag::tag1")
    task_run = TaskRun(
        parent=mock_task,
        input="test",
        input_source=data_source,
        tags=["tag1"],
        output=TaskOutput(output="test"),
    )
    task_run.save_to_file()

    other_run_config = TaskRunConfig(
        name="other",
        description="a run config this runner was not given",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=mock_task,
    )
    other_run_config.save_to_file()
    EvalRun(
        parent=mock_eval_config,
        dataset_id=task_run.id,
        task_run_config_id=other_run_config.id,
        input="test",
        output="test",
        scores={"accuracy": 1.0},
    ).save_to_file()

    jobs = build_task_run_eval_runner(
        [mock_eval_config], [mock_run_config]
    ).collect_tasks()

    assert [job.item.id for job in jobs] == [task_run.id]


def test_collect_tasks_ignores_calibration_runs_when_a_run_config_has_no_id(
    mock_eval, mock_task, data_source, mock_eval_config, mock_run_config
):
    """`ID_TYPE` is `str | None`, so a run config file carrying a null id loads as one.

    That makes None a real key in the already-run map, and every calibration record —
    which is exactly the set that carries `task_run_config_id=None` — would be folded
    into it, silently skipping items that were never scored for this run config.
    """
    mock_eval.splits["test"] = TaskRunSplit(filter_id="tag::tag1")
    task_run = TaskRun(
        parent=mock_task,
        input="test",
        input_source=data_source,
        tags=["tag1"],
        output=TaskOutput(output="test"),
    )
    task_run.save_to_file()
    EvalRun(
        parent=mock_eval_config,
        dataset_id=task_run.id,
        task_run_config_id=None,
        eval_config_eval=True,
        input="test",
        output="test",
        scores={"accuracy": 1.0},
    ).save_to_file()

    id_less_run_config = TaskRunConfig(
        id=None,
        name="no id",
        description="a run config whose stored id is null",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=mock_task,
    )
    assert id_less_run_config.id is None

    jobs = build_task_run_eval_runner(
        [mock_eval_config], [id_less_run_config]
    ).collect_tasks()

    assert [job.item.id for job in jobs] == [task_run.id]


# -------------------------------------------------------------------
# Conversation completeness: the gate on saving and reusing a driven trace
# -------------------------------------------------------------------

TWO_TURN_CONVERSATION: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "turn 1"},
    {"role": "assistant", "content": "hi"},
    {"role": "user", "content": "turn 2"},
    {"role": "assistant", "content": "bye"},
]

# One user turn: what a three-turn item's conversation looks like when two of its
# turns are missing from the record.
STUMP_CONVERSATION: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "turn 1"},
    {"role": "assistant", "content": "hi"},
]


def _stump_drive(task: Task, data_source: DataSource) -> DriveCaseResult:
    """A drive that ran all three of its turns but whose leaf recorded only one.

    The full-length chain is what says the synthetic user did not end this
    conversation, so the short trace is a lost record rather than a finished
    conversation — the failure the completeness gate exists to catch.
    """
    return _fresh_leaf(task, data_source, trace=STUMP_CONVERSATION)


def _su_ended_drive(task: Task, data_source: DataSource) -> DriveCaseResult:
    """A three-turn item's drive the synthetic user ended after one turn.

    Chain and trace agree at one turn, which is what separates it from
    `_stump_drive`: the conversation is whole, it is just short.
    """
    return _fresh_leaf(task, data_source, trace=STUMP_CONVERSATION, chain_length=1)


class TestConversationHealthProblem:
    def test_the_exact_turn_count_ending_on_an_assistant_reply_is_complete(self):
        assert conversation_health_problem(TWO_TURN_CONVERSATION, 2) is None

    @pytest.mark.parametrize("required_turns", [1, 3])
    def test_a_turn_count_that_is_not_the_items_is_incomplete(self, required_turns):
        problem = conversation_health_problem(TWO_TURN_CONVERSATION, required_turns)
        assert problem == f"expected {required_turns} user turns, found 2"

    @pytest.mark.parametrize("trace", [None, []])
    def test_no_conversation_at_all_is_incomplete(self, trace):
        assert conversation_health_problem(trace, 2) == "expected 2 user turns, found 0"

    def test_only_user_messages_count_as_turns(self):
        """System and tool messages share the trace; counting them would let a
        conversation one reply short pass as complete."""
        trace: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "you are a helpful agent"},
            *TWO_TURN_CONVERSATION,
        ]
        assert conversation_health_problem(trace, 2) is None

    def test_a_conversation_ending_on_the_user_is_incomplete(self):
        """The agent never answered the last thing said to it, so there is nothing
        for a judge to score there."""
        trace: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "turn 2"},
        ]
        problem = conversation_health_problem(trace, 2)
        assert (
            problem
            == "the conversation ends with a 'user' message, not an assistant reply"
        )

    @pytest.mark.parametrize(
        "content", [None, "", "   ", [], [{"type": "text", "text": ""}]]
    )
    def test_a_final_assistant_message_with_no_text_is_incomplete(self, content):
        trace: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": content},
        ]
        assert (
            conversation_health_problem(trace, 1)
            == "the final assistant message has no text content"
        )

    def test_a_final_assistant_message_of_content_parts_is_complete(self):
        """Content is a string or a list of parts, and both shapes are stored."""
        trace: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "turn 1"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "here is your refund"}],
            },
        ]
        assert conversation_health_problem(trace, 1) is None

    def test_failed_tool_calls_do_not_make_a_conversation_incomplete(self):
        """Completeness is structural. How an agent handles a tool that errors is
        exactly the kind of thing an eval exists to judge, so an error-bearing tool
        message must not disqualify the conversation from being judged."""
        trace: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "refund me"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "refund", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "refund service unavailable",
                "is_error": True,
                "error_message": "502 from billing",
            },
            {"role": "assistant", "content": "I could not process that refund."},
        ]
        assert conversation_health_problem(trace, 1) is None


class TestAnSUEndedConversationIsCompleteBelowTheCeiling:
    """`required_turns` is a ceiling for a conversation the synthetic user chose to
    end, so a short one is finished rather than truncated. Every other check the gate
    makes still applies. This is the stored-trace path — the vet, where the drive is
    gone and the tag is all that is left; a caller still holding the drive passes the
    turns that ran and gets the exact check."""

    @pytest.mark.parametrize("required_turns", [3, 10])
    def test_fewer_turns_than_required_is_complete(self, required_turns):
        assert (
            conversation_health_problem(
                TWO_TURN_CONVERSATION, required_turns, ended_by_su=True
            )
            is None
        )

    @pytest.mark.parametrize("trace", [None, []])
    def test_no_conversation_at_all_is_still_incomplete(self, trace):
        """Ending the conversation still means saying something first; an empty
        trace is a drive that produced nothing, not a short one."""
        assert (
            conversation_health_problem(trace, 3, ended_by_su=True)
            == "expected 1 to 3 user turns, found 0"
        )

    def test_more_turns_than_required_is_still_incomplete(self):
        """The ceiling is enforced from above either way: a conversation the
        synthetic user ended cannot have outrun the limit it stopped short of, so a
        longer one means the trace is not the drive's."""
        assert (
            conversation_health_problem(TWO_TURN_CONVERSATION, 1, ended_by_su=True)
            == "expected 1 to 1 user turns, found 2"
        )

    def test_a_conversation_ending_on_the_user_is_still_incomplete(self):
        trace: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "turn 2"},
        ]
        assert conversation_health_problem(trace, 5, ended_by_su=True) == (
            "the conversation ends with a 'user' message, not an assistant reply"
        )

    def test_a_final_assistant_message_with_no_text_is_still_incomplete(self):
        trace: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": ""},
        ]
        assert conversation_health_problem(trace, 3, ended_by_su=True) == (
            "the final assistant message has no text content"
        )


class TestDrivenConversationIsVettedBeforeSaving:
    @pytest.mark.asyncio
    async def test_a_short_drive_saves_nothing_and_raises_retryably(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        """A conversation that stopped short is a failed generation: saving it would
        hand the stump to every judge of this item from then on."""
        runner = build_task_run_eval_runner([mock_v2_redrive_config], [mock_run_config])
        job = EvalJob(
            item=multi_turn_eval_input,
            eval_config=mock_v2_redrive_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=StubV2Eval(mock_v2_redrive_config),
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(return_value=_stump_drive(mock_task, data_source)),
            ),
        ):
            # RetryableError is what AsyncJobRunner re-drives on; a plain Exception here
            # would fail the job on its first short drive.
            with pytest.raises(RetryableError, match="expected 3 user turns, found 1"):
                await runner.run_job(job)

        assert eval_traces(mock_task) == []
        assert mock_v2_redrive_config.runs(readonly=True) == []

        # The key is still free: the retry drives again rather than finding a stump.
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=StubV2Eval(mock_v2_redrive_config),
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(return_value=_fresh_leaf(mock_task, data_source)),
            ),
        ):
            assert await runner.run_job(job) is True

        traces = eval_traces(mock_task)
        assert len(traces) == 1
        assert traces[0].trace == MULTI_TURN_TRACE

    @pytest.mark.asyncio
    async def test_a_conversation_the_su_ended_is_saved_and_tagged(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
        caplog,
    ):
        """A drive shorter than the ceiling because the synthetic user ended it is a
        finished conversation, so it is judged and kept — carrying the tag that lets
        a later run tell it apart from a truncated record."""
        runner = build_task_run_eval_runner([mock_v2_redrive_config], [mock_run_config])
        job = EvalJob(
            item=multi_turn_eval_input,
            eval_config=mock_v2_redrive_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=StubV2Eval(mock_v2_redrive_config),
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(return_value=_su_ended_drive(mock_task, data_source)),
            ),
            caplog.at_level(
                logging.WARNING, logger="kiln_ai.adapters.eval.eval_runner"
            ),
        ):
            assert await runner.run_job(job) is True

        traces = eval_traces(mock_task)
        assert len(traces) == 1
        assert traces[0].trace == STUMP_CONVERSATION
        assert traces[0].tags == [TAG_SU_ENDED_CONVERSATION]
        # A conversation that ended on its first turn is a single-turn trace under a
        # multi-turn item, which is worth a line in the log even though it is kept.
        assert any(
            "ended on its first turn" in record.getMessage()
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_an_early_ended_drive_whose_trace_lost_turns_is_still_rejected(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        """Ending early excuses a conversation from the item's ceiling, not from its
        own chain: two turns ran, so a leaf recording one lost a turn and is as broken
        as any other partial record."""
        runner = build_task_run_eval_runner([mock_v2_redrive_config], [mock_run_config])
        job = EvalJob(
            item=multi_turn_eval_input,
            eval_config=mock_v2_redrive_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=StubV2Eval(mock_v2_redrive_config),
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(
                    return_value=_fresh_leaf(
                        mock_task,
                        data_source,
                        trace=STUMP_CONVERSATION,
                        chain_length=2,
                    )
                ),
            ),
        ):
            with pytest.raises(RetryableError, match="expected 2 user turns, found 1"):
                await runner.run_job(job)

        assert eval_traces(mock_task) == []
        assert mock_v2_redrive_config.runs(readonly=True) == []

    @pytest.mark.asyncio
    async def test_a_full_length_drive_is_saved_without_the_early_ending_tag(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        """The tag has to mean something, so a drive that used its whole ceiling must
        not carry it — otherwise every stored trace would pass the gate."""
        runner = build_task_run_eval_runner([mock_v2_redrive_config], [mock_run_config])
        job = EvalJob(
            item=multi_turn_eval_input,
            eval_config=mock_v2_redrive_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=StubV2Eval(mock_v2_redrive_config),
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(return_value=_fresh_leaf(mock_task, data_source)),
            ),
        ):
            assert await runner.run_job(job) is True

        traces = eval_traces(mock_task)
        assert len(traces) == 1
        assert traces[0].tags == []

    @pytest.mark.asyncio
    async def test_the_job_retries_the_drive_and_the_whole_one_persists(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        """End to end through AsyncJobRunner's bounded retry."""
        runner = build_task_run_eval_runner([mock_v2_redrive_config], [mock_run_config])
        drive = AsyncMock(
            side_effect=[
                _stump_drive(mock_task, data_source),
                _fresh_leaf(mock_task, data_source),
            ]
        )
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=StubV2Eval(mock_v2_redrive_config),
            ),
            patch("kiln_ai.adapters.eval.eval_runner.drive_case_for_eval", new=drive),
            patch("kiln_ai.utils.async_job_runner.asyncio.sleep", new=AsyncMock()),
        ):
            await run_to_completion(runner)

        assert drive.await_count == 2
        traces = eval_traces(mock_task)
        assert len(traces) == 1
        assert traces[0].trace == MULTI_TURN_TRACE
        records = mock_v2_redrive_config.runs(readonly=True)
        assert len(records) == 1
        assert records[0].scored_run_id == traces[0].id
        assert records[0].skipped_reason is None

    @pytest.mark.asyncio
    async def test_a_re_driven_job_is_not_logged_as_an_error(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
        caplog,
    ):
        """A short drive is an expected, self-healing event. An ERROR with a stacktrace
        would make a run that recovered on its own read as a run that broke."""
        runner = build_task_run_eval_runner([mock_v2_redrive_config], [mock_run_config])
        drive = AsyncMock(
            side_effect=[
                _stump_drive(mock_task, data_source),
                _fresh_leaf(mock_task, data_source),
            ]
        )
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=StubV2Eval(mock_v2_redrive_config),
            ),
            patch("kiln_ai.adapters.eval.eval_runner.drive_case_for_eval", new=drive),
            patch("kiln_ai.utils.async_job_runner.asyncio.sleep", new=AsyncMock()),
            caplog.at_level(
                logging.WARNING, logger="kiln_ai.adapters.eval.eval_runner"
            ),
        ):
            await run_to_completion(runner)

        assert drive.await_count == 2
        logged = [
            record
            for record in caplog.records
            if record.name == "kiln_ai.adapters.eval.eval_runner"
        ]
        assert [record.levelno for record in logged] == [logging.WARNING]
        assert "expected 3 user turns, found 1" in logged[0].getMessage()
        assert logged[0].exc_info is None

    @pytest.mark.asyncio
    async def test_a_drive_that_never_completes_fails_the_job_visibly(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        """Retries are bounded: the job errors rather than looping, and still nothing
        broken reaches disk."""
        runner = build_task_run_eval_runner([mock_v2_redrive_config], [mock_run_config])
        drive = AsyncMock(
            side_effect=lambda **kwargs: _stump_drive(mock_task, data_source)
        )
        last = None
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=StubV2Eval(mock_v2_redrive_config),
            ),
            patch("kiln_ai.adapters.eval.eval_runner.drive_case_for_eval", new=drive),
            patch("kiln_ai.utils.async_job_runner.asyncio.sleep", new=AsyncMock()),
        ):
            async for progress in runner.run():
                last = progress

        assert last is not None
        assert last.errors == 1
        assert last.complete == 0
        # One attempt plus the runner's two retries.
        assert drive.await_count == 3
        assert eval_traces(mock_task) == []
        assert mock_v2_redrive_config.runs(readonly=True) == []


def _stored_conversation(
    task: Task,
    run_config: TaskRunConfig,
    trace: list[ChatCompletionMessageParam] | None,
    *,
    source_id: str = "ei_redrive",
    output: str = "stored reply",
    tags: list[str] | None = None,
) -> TaskRun:
    """An eval trace already on disk for an item and run config, as a previous eval
    run (or the migration) would have left it."""
    run = TaskRun(
        tags=tags or [],
        parent=task,
        input="opening message",
        output=TaskOutput(
            output=output,
            source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "gpt-4",
                    "model_provider": "openai",
                    "adapter_name": "test_adapter",
                },
                run_config_id=run_config.id,
            ),
        ),
        trace=trace,
        eval_source=EvalItemSource(source_type="eval_input", source_id=source_id),
    )
    run.save_to_file()
    return run


class TestStoredConversationsAreVettedBeforeReuse:
    @pytest.mark.asyncio
    async def test_a_stored_conversation_short_for_its_item_is_re_driven(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        """The reuse key says nothing about what is inside the file. A partial
        conversation at the right key must not be served to the judge."""
        stump = _stored_conversation(mock_task, mock_run_config, STUMP_CONVERSATION)
        assert stump.path is not None
        bytes_before = stump.path.read_bytes()

        runner = build_task_run_eval_runner([mock_v2_redrive_config], [mock_run_config])
        job = EvalJob(
            item=multi_turn_eval_input,
            eval_config=mock_v2_redrive_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        stub = RecordingStubV2Eval(mock_v2_redrive_config)
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=stub,
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(return_value=_fresh_leaf(mock_task, data_source)),
            ) as drive,
        ):
            assert await runner.run_job(job) is True

        drive.assert_awaited_once()
        assert stub.seen_inputs[0].trace == MULTI_TURN_TRACE
        fresh = next(t for t in eval_traces(mock_task) if t.id != stump.id)
        assert mock_v2_redrive_config.runs(readonly=True)[0].scored_run_id == fresh.id

        # The rejected conversation is left exactly where it was: a trace is never
        # deleted or rewritten on this path.
        assert stump.path.exists()
        assert stump.path.read_bytes() == bytes_before

    @pytest.mark.asyncio
    async def test_a_stored_conversation_complete_for_its_item_is_reused(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        """The other half: vetting must not cost a re-drive of a conversation that is
        whole, which is the entire point of reusing them."""
        stored = _stored_conversation(mock_task, mock_run_config, MULTI_TURN_TRACE)

        runner = build_task_run_eval_runner([mock_v2_redrive_config], [mock_run_config])
        job = EvalJob(
            item=multi_turn_eval_input,
            eval_config=mock_v2_redrive_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        with (
            patch(
                "kiln_ai.adapters.eval.registry.v2_eval_adapter_from_config",
                return_value=StubV2Eval(mock_v2_redrive_config),
            ),
            patch(
                "kiln_ai.adapters.eval.eval_runner.drive_case_for_eval",
                new=AsyncMock(),
            ) as drive,
        ):
            assert await runner.run_job(job) is True

        drive.assert_not_awaited()
        assert [t.id for t in eval_traces(mock_task)] == [stored.id]
        assert mock_v2_redrive_config.runs(readonly=True)[0].scored_run_id == stored.id

    @pytest.mark.asyncio
    async def test_single_turn_items_have_no_turn_contract_to_fail(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
    ):
        """A single-turn item's stored trace is reused whatever shape it has: the
        completeness check belongs to items that declare a turn count."""
        single_turn_item = EvalInput(
            id="ei_single",
            data=SingleTurnEvalInputData(user_message=UserMessage(text="hello")),
            parent=mock_task,
        )
        single_turn_item.save_to_file()
        stored = _stored_conversation(
            mock_task,
            mock_run_config,
            STUMP_CONVERSATION,
            source_id="ei_single",
        )

        runner = build_task_run_eval_runner([mock_v2_redrive_config], [mock_run_config])
        # The vet is installed — the split still holds a multi-turn item — so this is
        # permissiveness, not absence.
        assert runner._build_trace_vet() is not None

        job = EvalJob(
            item=single_turn_item,
            eval_config=mock_v2_redrive_config,
            type="task_run_eval",
            task_run_config=mock_run_config,
        )
        generator = TraceGenerator(mock_task)
        with generating(generator):
            assert await runner.run_job(job) is True

        assert generator.calls == []
        record = next(
            r
            for r in mock_v2_redrive_config.runs(readonly=True)
            if r.eval_input_id == "ei_single"
        )
        assert record.scored_run_id == stored.id

    def test_the_vet_judges_only_the_items_it_has_turn_counts_for(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        unstamped = EvalInput(
            id="ei_unstamped",
            data=MultiTurnSyntheticEvalInputData(
                first_message=UserMessage(text="opening message"),
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
            ),
            parent=mock_task,
        )
        unstamped.save_to_file()
        runner = build_task_run_eval_runner([mock_v2_redrive_config], [mock_run_config])
        vet = runner._build_trace_vet()
        assert vet is not None

        stump = _stored_conversation(mock_task, mock_run_config, STUMP_CONVERSATION)
        whole = _stored_conversation(mock_task, mock_run_config, MULTI_TURN_TRACE)
        rc_id = mock_run_config.id

        assert vet(("eval_input", "ei_redrive", rc_id), whole) is None
        # A rejection answers with the reason, which is what the index logs.
        assert vet(("eval_input", "ei_redrive", rc_id), stump) == (
            "expected 3 user turns, found 1"
        )
        # An item from another eval's split: this runner has no turn count for it, and
        # never looks it up either.
        assert vet(("eval_input", "ei_elsewhere", rc_id), stump) is None
        # Ids come from one generator shared by every model type, so a TaskRun can carry
        # an EvalInput's id. Matching on the id alone would check the wrong contract.
        assert vet(("task_run", "ei_redrive", rc_id), stump) is None
        # An item with no drive config is skipped by the readiness path before it ever
        # drives, so there is no turn count to hold its traces to.
        assert vet(("eval_input", "ei_unstamped", rc_id), stump) is None

    def test_the_vet_reads_the_tag_to_accept_a_short_stored_conversation(
        self,
        mock_task,
        mock_run_config,
        mock_v2_redrive_config,
        multi_turn_eval_input,
        data_source,
    ):
        """The tag is the only thing on disk that says a short conversation is one
        the synthetic user ended, so the same trace is reusable with it and a
        truncated record without it."""
        ended = _stored_conversation(
            mock_task,
            mock_run_config,
            STUMP_CONVERSATION,
            tags=[TAG_SU_ENDED_CONVERSATION],
        )
        truncated = _stored_conversation(mock_task, mock_run_config, STUMP_CONVERSATION)
        runner = build_task_run_eval_runner([mock_v2_redrive_config], [mock_run_config])
        vet = runner._build_trace_vet()
        assert vet is not None
        rc_id = mock_run_config.id

        assert vet(("eval_input", "ei_redrive", rc_id), ended) is None
        assert vet(("eval_input", "ei_redrive", rc_id), truncated) == (
            "expected 3 user turns, found 1"
        )

    def test_a_run_with_no_multi_turn_items_installs_no_vet(
        self, mock_v2_task_run_eval_runner, mock_v2_runner
    ):
        """Single-turn splits and calibration runs (which have no split at all) build
        their index exactly as before."""
        assert mock_v2_task_run_eval_runner._build_trace_vet() is None
        assert mock_v2_runner.split is None
        assert mock_v2_runner._build_trace_vet() is None
