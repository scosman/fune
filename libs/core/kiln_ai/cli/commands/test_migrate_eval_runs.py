import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pytest
import typer
from rich.console import Console

from kiln_ai.adapters.eval.trace_index import TraceIndex, trace_key
from kiln_ai.datamodel import Project, Task
from kiln_ai.datamodel.datamodel_enums import (
    ModelProviderName,
    StructuredOutputMode,
    TaskOutputRatingType,
)
from kiln_ai.datamodel.eval import (
    Eval,
    EvalConfig,
    EvalConfigType,
    EvalInput,
    EvalInputSplit,
    EvalOutputScore,
    EvalRun,
    ExactMatchProperties,
    SingleTurnEvalInputData,
    SkippedReason,
    TaskRunSplit,
    UserMessage,
)
from kiln_ai.datamodel.model_cache import ModelCache
from kiln_ai.datamodel.run_config import (
    KilnAgentRunConfigProperties,
    McpRunConfigProperties,
    MCPToolReference,
)
from kiln_ai.datamodel.task import TaskRunConfig
from kiln_ai.datamodel.task_output import DataSource, DataSourceType, TaskOutput
from kiln_ai.datamodel.task_run import EvalItemSource, TaskRun
from kiln_ai.datamodel.usage import Usage

from . import migrate_eval_runs as migration
from .migrate_eval_runs import (
    MIGRATION_ADAPTER_NAME,
    MigrationAction,
    apply_plan,
    migrate_eval_runs,
    plan_project,
    print_plan,
)


def build_project(
    tmp_path: Path,
    input_json_schema: str | None = None,
    task_name: str = "Migration Task",
) -> Project:
    project = Project(name="Migration Project", path=tmp_path / "project.kiln")
    project.save_to_file()
    task = Task(
        name=task_name,
        instruction="Answer the question",
        input_json_schema=input_json_schema,
        parent=project,
    )
    task.save_to_file()
    return project


def only_task(project: Project) -> Task:
    return project.tasks()[0]


def build_run_config(task: Task, name: str = "Run Config") -> TaskRunConfig:
    run_config = TaskRunConfig(
        name=name,
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=task,
    )
    run_config.save_to_file()
    return run_config


def build_eval(task: Task, name: str = "Eval", eval_input_backed: bool = False) -> Eval:
    eval = Eval(
        name=name,
        splits={
            "test": EvalInputSplit(filter_id="all")
            if eval_input_backed
            else TaskRunSplit(filter_id="all")
        },
        eval_configs_filter_id="all",
        output_scores=[
            EvalOutputScore(name="Accuracy", type=TaskOutputRatingType.pass_fail)
        ],
        parent=task,
    )
    eval.save_to_file()
    return eval


def build_v2_config(eval: Eval, name: str = "V2 Config") -> EvalConfig:
    config = EvalConfig(
        name=name,
        config_type=EvalConfigType.v2,
        properties=ExactMatchProperties(expected_value="expected"),
        parent=eval,
    )
    config.save_to_file()
    return config


def build_v1_config(eval: Eval, name: str = "V1 Config") -> EvalConfig:
    config = EvalConfig(
        name=name,
        config_type=EvalConfigType.g_eval,
        model_name="gpt-4o",
        model_provider="openai",
        properties={"eval_steps": ["check it"]},
        parent=eval,
    )
    config.save_to_file()
    return config


def build_dataset_item(task: Task, input: str = "the question") -> TaskRun:
    item = TaskRun(
        parent=task,
        input=input,
        output=TaskOutput(output="the golden answer"),
    )
    item.save_to_file()
    return item


def build_eval_input(task: Task, text: str = "the question") -> EvalInput:
    eval_input = EvalInput(
        parent=task,
        data=SingleTurnEvalInputData(user_message=UserMessage(text=text)),
    )
    eval_input.save_to_file()
    return eval_input


def save_eval_run(config: EvalConfig, **kwargs) -> EvalRun:
    eval_run = EvalRun(parent=config, **kwargs)
    eval_run.save_to_file()
    return eval_run


def snapshot(root: Path) -> Dict[Path, bytes]:
    """Every file under `root`, by content. The migration is judged on this."""
    return {
        path: path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()
    }


def changed_paths(before: Dict[Path, bytes], after: Dict[Path, bytes]) -> set[Path]:
    return {
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    }


@dataclass
class Fixture:
    """A project holding one of every record shape the migration has to tell apart."""

    project: Project
    task: Task
    run_config: TaskRunConfig
    dataset_item: TaskRun
    eval_input: EvalInput
    eval: Eval
    v2_config: EvalConfig
    v1_config: EvalConfig
    inline: EvalRun
    inline_eval_input: EvalRun
    pointer: EvalRun
    calibration: EvalRun
    pre_generation_skip: EvalRun
    v1_record: EvalRun

    def reload(self, eval_run: EvalRun) -> EvalRun:
        assert eval_run.path is not None
        return EvalRun.load_from_file(eval_run.path)

    def eval_traces(self) -> list[TaskRun]:
        return [
            run
            for run in self.task.runs(include_eval_generated=True)
            if run.eval_source is not None
        ]


@pytest.fixture
def fixture(tmp_path) -> Fixture:
    project = build_project(tmp_path)
    task = only_task(project)
    run_config = build_run_config(task)
    dataset_item = build_dataset_item(task)
    eval_input = build_eval_input(task)
    eval = build_eval(task)
    v2_config = build_v2_config(eval)
    v1_config = build_v1_config(eval)

    return Fixture(
        project=project,
        task=task,
        run_config=run_config,
        dataset_item=dataset_item,
        eval_input=eval_input,
        eval=eval,
        v2_config=v2_config,
        v1_config=v1_config,
        inline=save_eval_run(
            v2_config,
            task_run_config_id=run_config.id,
            dataset_id=dataset_item.id,
            scores={"accuracy": 1.0},
            input="the question",
            output="the generated answer",
            intermediate_outputs={"chain_of_thought": "thinking"},
            eval_usage=Usage(input_tokens=7, output_tokens=8),
        ),
        inline_eval_input=save_eval_run(
            v2_config,
            task_run_config_id=run_config.id,
            eval_input_id=eval_input.id,
            scores={"accuracy": 0.0},
            input="the question",
            output="a different generated answer",
        ),
        pointer=save_eval_run(
            v2_config,
            task_run_config_id=run_config.id,
            dataset_id=dataset_item.id,
            scored_run_id="already-a-trace",
            scores={"accuracy": 1.0},
        ),
        calibration=save_eval_run(
            v2_config,
            task_run_config_id=None,
            dataset_id=dataset_item.id,
            eval_config_eval=True,
            scores={"accuracy": 1.0},
            input="the question",
            output="the golden answer",
        ),
        pre_generation_skip=save_eval_run(
            v2_config,
            task_run_config_id=run_config.id,
            dataset_id=dataset_item.id,
            scores={},
            input="the question",
            output=None,
            skipped_reason=SkippedReason.incompatible_input_shape.value,
            skipped_detail="multi-turn",
        ),
        v1_record=save_eval_run(
            v1_config,
            task_run_config_id=run_config.id,
            dataset_id=dataset_item.id,
            scores={"accuracy": 1.0},
            input="the question",
            output="the v1 answer",
            task_run_usage=Usage(input_tokens=1, output_tokens=2),
        ),
    )


def migrate(fixture: Fixture) -> None:
    failures = apply_plan(plan_project(fixture.project))
    assert failures == []


def test_plan_classifies_every_record_shape(fixture):
    plan = plan_project(fixture.project)

    assert {change.eval_run.id: change.action for change in plan.changes} == {
        fixture.inline.id: MigrationAction.create_trace,
        fixture.inline_eval_input.id: MigrationAction.create_trace,
        fixture.pointer.id: MigrationAction.nothing_to_do,
        fixture.calibration.id: MigrationAction.link_calibration,
        fixture.pre_generation_skip.id: MigrationAction.clear_inline,
    }
    # The V1 config's records are not even loaded, let alone planned.
    assert plan.v1_configs_skipped == 1


def test_only_v2_inline_records_change(fixture, tmp_path):
    before = snapshot(tmp_path)
    migrate(fixture)
    after = snapshot(tmp_path)

    rewritten = {
        fixture.inline.path,
        fixture.inline_eval_input.path,
        fixture.calibration.path,
        fixture.pre_generation_skip.path,
    }
    created = {trace.path for trace in fixture.eval_traces()}
    assert changed_paths(before, after) == rewritten | created

    # Named explicitly, because "unchanged" is the whole promise for these two.
    assert after[fixture.v1_record.path] == before[fixture.v1_record.path]
    assert after[fixture.pointer.path] == before[fixture.pointer.path]


def test_v1_record_keeps_its_inline_trace(fixture):
    migrate(fixture)

    v1_record = fixture.reload(fixture.v1_record)
    assert v1_record.scored_run_id is None
    assert v1_record.input == "the question"
    assert v1_record.output == "the v1 answer"
    assert v1_record.task_run_usage == Usage(input_tokens=1, output_tokens=2)


def test_dry_run_writes_nothing(fixture, tmp_path):
    before = snapshot(tmp_path)

    with patch(
        "kiln_ai.cli.commands.migrate_eval_runs.typer.confirm", return_value=True
    ) as confirm:
        plan = migrate_eval_runs(tmp_path, dry_run=True)

    assert snapshot(tmp_path) == before
    confirm.assert_not_called()
    # The dry run reports exactly the work the real run then does.
    assert len(plan.pending_writes) == 4
    assert fixture.eval_traces() == []


def test_migration_is_idempotent(fixture, tmp_path):
    migrate(fixture)
    after_first = snapshot(tmp_path)

    second_plan = plan_project(fixture.project)
    assert [change.action for change in second_plan.changes] == [
        MigrationAction.nothing_to_do
    ] * 5
    assert second_plan.pending_writes == []

    assert apply_plan(second_plan) == []
    assert snapshot(tmp_path) == after_first


def test_inline_record_becomes_a_pointer(fixture):
    migrate(fixture)

    migrated = fixture.reload(fixture.inline)
    trace = next(
        run for run in fixture.eval_traces() if run.id == migrated.scored_run_id
    )

    assert migrated.scored_run_id == trace.id
    assert migrated.input is None
    assert migrated.output is None
    assert migrated.task_run_trace is None
    assert migrated.task_run_usage is None
    assert migrated.reference_answer is None

    # Everything the score record is *for* survives untouched.
    assert migrated.scores == {"accuracy": 1.0}
    assert migrated.dataset_id == fixture.dataset_item.id
    assert migrated.eval_input_id is None
    assert migrated.task_run_config_id == fixture.run_config.id
    assert migrated.intermediate_outputs == {"chain_of_thought": "thinking"}
    assert migrated.id == fixture.inline.id
    assert migrated.created_at == fixture.inline.created_at


def test_rewriting_a_record_changes_only_the_trace_fields(fixture):
    """The record is rebuilt from its own dump, so anything that failed to round-trip
    would be silently dropped from a file the user already has. Every field a record can
    carry is populated here, because that claim is only as good as the fields it covers —
    the skip fields are covered by the skip tests, which assert both survive."""
    record = save_eval_run(
        fixture.v2_config,
        task_run_config_id=fixture.run_config.id,
        dataset_id=fixture.dataset_item.id,
        scores={"accuracy": 1.0},
        input="the question",
        output="the generated answer",
        reference_answer="the golden answer",
        task_run_trace=json.dumps([{"role": "user", "content": "the question"}]),
        task_run_usage=Usage(input_tokens=1, output_tokens=2, cost=0.5),
        eval_usage=Usage(input_tokens=7, output_tokens=8),
        intermediate_outputs={"chain_of_thought": "thinking"},
    )
    before = record.model_dump()

    migrate(fixture)
    after = fixture.reload(record).model_dump()

    assert {key for key in before if before[key] != after[key]} == {
        "input",
        "output",
        "task_run_trace",
        "task_run_usage",
        "reference_answer",
        "scored_run_id",
    }
    assert after["eval_usage"] == before["eval_usage"] is not None


def test_eval_input_record_keeps_pointing_at_its_eval_input(fixture):
    """Migration moves the trace off the record; which item the record scored is not
    the runner's to change, so the EvalInput pointer survives untouched."""
    migrate(fixture)

    migrated = fixture.reload(fixture.inline_eval_input)
    assert migrated.scored_run_id is not None
    assert migrated.eval_input_id == fixture.eval_input.id
    assert migrated.dataset_id is None


@pytest.mark.parametrize(
    "record_name,expected_source_type,item_attr",
    [
        ("inline", "task_run", "dataset_item"),
        ("inline_eval_input", "eval_input", "eval_input"),
    ],
)
def test_synthesized_trace_is_filed_under_the_item_it_scored(
    fixture, record_name, expected_source_type, item_attr
):
    migrate(fixture)

    migrated = fixture.reload(getattr(fixture, record_name))
    trace = next(
        run for run in fixture.eval_traces() if run.id == migrated.scored_run_id
    )
    item = getattr(fixture, item_attr)

    assert trace.eval_source is not None
    assert trace.eval_source.source_type == expected_source_type
    assert trace.eval_source.source_id == item.id
    assert trace.output.source is not None
    assert trace.output.source.run_config_id == fixture.run_config.id


def test_synthesized_trace_carries_the_generation(fixture):
    migrate(fixture)

    trace = next(
        run
        for run in fixture.eval_traces()
        if run.id == fixture.reload(fixture.inline).scored_run_id
    )

    assert trace.input == "the question"
    assert trace.output.output == "the generated answer"
    assert trace.output.source is not None
    assert trace.output.source.type == DataSourceType.synthetic
    assert trace.output.source.properties["model_name"] == "gpt-4o"
    assert trace.output.source.properties["adapter_name"] == MIGRATION_ADAPTER_NAME
    assert trace.output.source.run_config == fixture.run_config.run_config_properties
    assert trace.input_source is not None
    assert trace.input_source.run_config_id == fixture.run_config.id


def test_synthesized_trace_carries_trace_and_usage_when_present(fixture):
    """The two fields the split moved off EvalRun, and the ones Phase 4's rollup reads."""
    messages = [
        {"role": "user", "content": "the question"},
        {
            "role": "assistant",
            "content": "the generated answer",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        },
    ]
    record = save_eval_run(
        fixture.v2_config,
        task_run_config_id=fixture.run_config.id,
        dataset_id=fixture.dataset_item.id,
        scores={"accuracy": 1.0},
        input="the question",
        output="the generated answer",
        task_run_trace=json.dumps(messages),
        task_run_usage=Usage(
            input_tokens=3, output_tokens=4, total_llm_latency_ms=12.0
        ),
    )

    migrate(fixture)

    trace = next(
        run
        for run in fixture.eval_traces()
        if run.id == fixture.reload(record).scored_run_id
    )
    assert trace.trace is not None
    assert [message["role"] for message in trace.trace] == ["user", "assistant"]
    assert [message["content"] for message in trace.trace] == [
        "the question",
        "the generated answer",
    ]
    assert trace.usage == Usage(
        input_tokens=3, output_tokens=4, total_llm_latency_ms=12.0
    )
    assert trace.cumulative_usage is not None
    assert trace.cumulative_usage.input_tokens == 3


def test_calibration_points_at_the_golden_item_and_creates_nothing(fixture):
    migrate(fixture)

    migrated = fixture.reload(fixture.calibration)
    assert migrated.scored_run_id == fixture.dataset_item.id
    assert migrated.input is None
    assert migrated.output is None

    # The golden item is the trace: it is never flagged, so it stays a dataset run.
    assert fixture.dataset_item.id not in {trace.id for trace in fixture.eval_traces()}
    assert fixture.dataset_item.id in {run.id for run in fixture.task.runs()}


def test_skipped_calibration_still_points_at_the_golden_item(fixture):
    """Phase 3 writes a live calibration skip with a scored_run_id, whether or not the
    judge was reached. An old one has to migrate to the same shape."""
    record = save_eval_run(
        fixture.v2_config,
        task_run_config_id=None,
        dataset_id=fixture.dataset_item.id,
        eval_config_eval=True,
        scores={},
        input="the question",
        output=None,
        skipped_reason=SkippedReason.type_not_available.value,
    )

    migrate(fixture)

    migrated = fixture.reload(record)
    assert migrated.scored_run_id == fixture.dataset_item.id
    assert migrated.input is None
    assert migrated.skipped_reason == SkippedReason.type_not_available.value


def test_pre_generation_skip_keeps_pointing_at_nothing(fixture):
    migrate(fixture)

    migrated = fixture.reload(fixture.pre_generation_skip)
    assert migrated.scored_run_id is None
    assert migrated.input is None
    assert migrated.skipped_reason == SkippedReason.incompatible_input_shape.value
    assert migrated.skipped_detail == "multi-turn"


def test_scoring_skip_is_a_skip_not_a_trace(fixture):
    """A scoring skip kept no output, so there is nothing to reconstruct — even though a
    trace did exist when it was written."""
    record = save_eval_run(
        fixture.v2_config,
        task_run_config_id=fixture.run_config.id,
        dataset_id=fixture.dataset_item.id,
        scores={},
        input="the question",
        output=None,
        skipped_reason=SkippedReason.missing_reference_key.value,
    )

    plan = plan_project(fixture.project)
    change = next(c for c in plan.changes if c.eval_run.id == record.id)
    assert change.action == MigrationAction.clear_inline
    assert change.trace is None


def test_already_clean_skip_needs_no_write(fixture):
    record = save_eval_run(
        fixture.v2_config,
        task_run_config_id=fixture.run_config.id,
        dataset_id=fixture.dataset_item.id,
        scores={},
        skipped_reason=SkippedReason.missing_trace.value,
    )

    change = next(
        c for c in plan_project(fixture.project).changes if c.eval_run.id == record.id
    )
    assert change.action == MigrationAction.nothing_to_do
    assert change.migrated is None


async def test_migrated_traces_are_reusable_by_the_runner(fixture):
    """The point of the whole project: the next eval config scores these instead of
    regenerating them. Asked of the index the way the runner asks it, so this keeps
    holding when the index changes how it stores what it knows."""
    migrate(fixture)
    index = TraceIndex(fixture.task)

    async def must_not_generate() -> TaskRun:
        raise AssertionError("the migrated trace should have been reused")

    for item in [
        ("task_run", fixture.dataset_item.id),
        ("eval_input", fixture.eval_input.id),
    ]:
        trace, generated = await index.get_or_create(
            trace_key(item, fixture.run_config.id), must_not_generate
        )
        assert generated is False
        assert trace.eval_source is not None
        assert trace.eval_source.source_id == item[1]


def test_migrated_traces_stay_off_dataset_surfaces(fixture):
    migrate(fixture)

    assert [run.id for run in fixture.task.runs()] == [fixture.dataset_item.id]
    assert len(fixture.task.runs(include_eval_generated=True)) == 3


def test_record_naming_a_deleted_run_config_is_left_alone(fixture, tmp_path):
    record = save_eval_run(
        fixture.v2_config,
        task_run_config_id="deleted-run-config",
        dataset_id=fixture.dataset_item.id,
        scores={"accuracy": 1.0},
        input="the question",
        output="the generated answer",
    )
    before = snapshot(tmp_path)

    change = next(
        c for c in plan_project(fixture.project).changes if c.eval_run.id == record.id
    )
    assert change.action == MigrationAction.unmigratable
    assert "deleted-run-config" in change.detail

    migrate(fixture)
    assert before[record.path] == snapshot(tmp_path)[record.path]


def test_calibration_without_a_dataset_item_is_left_alone(fixture):
    """A calibration record naming an eval input instead of a dataset item.

    The datamodel refuses to construct or load this shape (calibration requires
    dataset_id), so on disk it can only exist as a file this build can't read:
    it lands in load_errors, is stepped over, and its bytes are never touched.
    """
    runs_dir = fixture.v2_config.path.parent / "runs" / "999999999999"
    runs_dir.mkdir(parents=True)
    record_path = runs_dir / "eval_run.kiln"
    record_path.write_text(
        json.dumps(
            {
                "v": 1,
                "id": "999999999999",
                "model_type": "eval_run",
                "task_run_config_id": None,
                "eval_input_id": fixture.eval_input.id,
                "eval_config_eval": True,
                "scores": {"accuracy": 1.0},
                "input": "the question",
                "output": "the golden answer",
            }
        )
    )
    before = record_path.read_bytes()

    plan = plan_project(fixture.project)

    assert record_path in [error.path for error in plan.load_errors]
    assert all(c.eval_run.id != "999999999999" for c in plan.changes)
    assert apply_plan(plan) == []
    assert record_path.read_bytes() == before


def test_calibration_whose_golden_item_was_deleted_is_left_alone(fixture, tmp_path):
    """The record's own input/output is the last copy of what was calibrated. Nothing
    delete-protects a golden item (architecture §6 protects traces), so pointing at a
    deleted one and clearing the copy would destroy it."""
    golden = build_dataset_item(fixture.task, input="a deleted question")
    record = save_eval_run(
        fixture.v2_config,
        task_run_config_id=None,
        dataset_id=golden.id,
        eval_config_eval=True,
        scores={"accuracy": 1.0},
        input="a deleted question",
        output="a deleted answer",
    )
    golden.delete()
    before = snapshot(tmp_path)

    change = next(
        c for c in plan_project(fixture.project).changes if c.eval_run.id == record.id
    )
    assert change.action == MigrationAction.unmigratable
    assert "no longer exists" in change.detail

    migrate(fixture)
    after = fixture.reload(record)
    assert after.input == "a deleted question"
    assert after.output == "a deleted answer"
    assert before[record.path] == snapshot(tmp_path)[record.path]


def test_skip_whose_dataset_item_was_deleted_is_left_alone(fixture):
    """Same class, lower stakes: a live skip carries no input because the item supplies
    one for display. With the item gone, clearing it leaves nothing to display."""
    item = build_dataset_item(fixture.task, input="a deleted question")
    record = save_eval_run(
        fixture.v2_config,
        task_run_config_id=fixture.run_config.id,
        dataset_id=item.id,
        scores={},
        input="a deleted question",
        skipped_reason=SkippedReason.incompatible_input_shape.value,
    )
    item.delete()

    change = next(
        c for c in plan_project(fixture.project).changes if c.eval_run.id == record.id
    )
    assert change.action == MigrationAction.unmigratable
    assert "no longer exists" in change.detail

    migrate(fixture)
    assert fixture.reload(record).input == "a deleted question"


def test_skip_whose_eval_input_was_deleted_is_left_alone(fixture):
    eval_input = build_eval_input(fixture.task, text="a deleted question")
    record = save_eval_run(
        fixture.v2_config,
        task_run_config_id=fixture.run_config.id,
        eval_input_id=eval_input.id,
        scores={},
        input="a deleted question",
        skipped_reason=SkippedReason.incompatible_input_shape.value,
    )
    eval_input.delete()

    change = next(
        c for c in plan_project(fixture.project).changes if c.eval_run.id == record.id
    )
    assert change.action == MigrationAction.unmigratable


def corrupt_file_path(fixture: Fixture, level: str) -> Path:
    """Where to drop an unreadable file, for each kind of file the walk opens."""
    task_dir = fixture.task.path.parent
    locations = {
        "task": (fixture.project.path.parent / "tasks", "task.kiln"),
        "run_config": (task_dir / "run_configs", "task_run_config.kiln"),
        "task_run": (task_dir / "runs", "task_run.kiln"),
        "eval_input": (task_dir / "eval_inputs", "eval_input.kiln"),
        "eval": (task_dir / "evals", "eval.kiln"),
        "eval_config": (fixture.eval.path.parent / "configs", "eval_config.kiln"),
        "eval_run": (fixture.v2_config.path.parent / "runs", "eval_run.kiln"),
    }
    parent, filename = locations[level]
    return parent / "corrupt" / filename


@pytest.mark.parametrize(
    "level",
    ["task", "run_config", "task_run", "eval_input", "eval", "eval_config", "eval_run"],
)
def test_an_unreadable_file_is_reported_and_the_rest_migrate(fixture, level):
    """One file written by a newer Kiln must not stop a whole project from migrating —
    at any level of the walk, not just the records themselves."""
    corrupt = corrupt_file_path(fixture, level)
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text('{"truncated": ')

    plan = plan_project(fixture.project)

    assert [error.path for error in plan.load_errors] == [corrupt]
    assert len(plan.pending_writes) == 4
    assert apply_plan(plan) == []
    assert fixture.reload(fixture.inline).scored_run_id is not None


def test_mcp_run_config_traces_are_tool_calls(fixture):
    mcp_run_config = TaskRunConfig(
        name="MCP Run Config",
        run_config_properties=McpRunConfigProperties(
            tool_reference=MCPToolReference(tool_id="mcp::remote::server_id::tool_name")
        ),
        parent=fixture.task,
    )
    mcp_run_config.save_to_file()
    record = save_eval_run(
        fixture.v2_config,
        task_run_config_id=mcp_run_config.id,
        dataset_id=fixture.dataset_item.id,
        scores={"accuracy": 1.0},
        input="the question",
        output="the tool's answer",
    )

    migrate(fixture)

    trace = next(
        run
        for run in fixture.eval_traces()
        if run.id == fixture.reload(record).scored_run_id
    )
    assert trace.output.source is not None
    assert trace.output.source.type == DataSourceType.tool_call
    assert trace.output.source.properties == {}
    assert trace.output.source.run_config_id == mcp_run_config.id


@pytest.mark.parametrize(
    "written_elsewhere",
    [
        # Migrated by another writer...
        {"scored_run_id": "a-trace-someone-else-made", "input": None, "output": None},
        # ...or edited in any other way. Both would be reverted by a stale planned object.
        {"scores": {"accuracy": 0.0}, "skipped_detail": "edited by the app"},
    ],
)
def test_a_record_written_by_someone_else_is_not_clobbered(fixture, written_elsewhere):
    """The confirm prompt sits between planning and writing, so the app may have written
    the record in the meantime."""
    plan = plan_project(fixture.project)

    assert fixture.inline.path is not None
    edited = json.loads(fixture.inline.path.read_text()) | written_elsewhere
    fixture.inline.path.write_text(json.dumps(edited))

    failures = apply_plan(plan)

    assert len(failures) == 1
    assert failures[0].eval_run.id == fixture.inline.id
    assert "Changed on disk" in failures[0].message

    on_disk = json.loads(fixture.inline.path.read_text())
    assert {key: on_disk[key] for key in written_elsewhere} == written_elsewhere


def delete_item(item: TaskRun) -> None:
    item.delete()


def make_item_unreadable(item: TaskRun) -> None:
    """What a sync from a newer Kiln looks like — the file is there, the item isn't."""
    assert item.path is not None
    item.path.write_text('{"truncated": ')


def replace_item_with_another(item: TaskRun) -> None:
    assert item.path is not None
    swapped = json.loads(item.path.read_text()) | {"id": "a-different-run"}
    item.path.write_text(json.dumps(swapped))


@pytest.fixture
def live_model_cache():
    """Force `ModelCache` on for the test.

    It is disabled wherever the filesystem reports coarse timestamps, which is the case in
    CI — so a test of cache-related behavior is vacuous unless it turns the cache on
    itself. On macOS and Windows, which is what Kiln ships to, it is always live.
    """
    cache = ModelCache.shared()
    was_enabled = cache._enabled
    cache._enabled = True
    cache.clear()
    yield cache
    cache.clear()
    cache._enabled = was_enabled


def test_an_item_replaced_without_touching_its_mtime_is_not_deferred_to(
    fixture, live_model_cache
):
    """`ModelCache` validates on mtime alone, and the plan's own scan is what populates
    it. Every file-syncing tool preserves mtime by design, so a replacement that keeps it
    would be served the plan's own parse — the check would confirm the plan against the
    plan."""
    plan = plan_project(fixture.project)
    path = fixture.dataset_item.path
    assert path is not None
    planned_mtime = path.stat().st_mtime_ns

    replace_item_with_another(fixture.dataset_item)
    os.utime(path, ns=(planned_mtime, planned_mtime))

    # Non-vacuous: without invalidation this is exactly what the re-read would be handed.
    assert live_model_cache.get_model_id(path, TaskRun) == fixture.dataset_item.id

    failures = apply_plan(plan)

    assert {failure.eval_run.id for failure in failures} == {
        fixture.calibration.id,
        fixture.pre_generation_skip.id,
    }
    assert fixture.reload(fixture.calibration).output == "the golden answer"


@pytest.mark.parametrize(
    "lose_the_item",
    [delete_item, make_item_unreadable, replace_item_with_another],
    ids=["deleted", "unreadable", "replaced"],
)
def test_an_item_lost_after_planning_is_not_deferred_to(fixture, lose_the_item):
    """The plan's item check is a snapshot, and the confirm prompt can hold it open for
    minutes. Deferring the record's last copy to something lost in that window is the one
    way this command can destroy data, so the check is re-run at the write — and re-run as
    the same check, since a file that is present but unreadable, or present but holding a
    different item, is just as gone as a deleted one."""
    plan = plan_project(fixture.project)
    lose_the_item(fixture.dataset_item)

    failures = apply_plan(plan)

    # Both records that were going to defer to the golden item, and only those.
    assert {failure.eval_run.id for failure in failures} == {
        fixture.calibration.id,
        fixture.pre_generation_skip.id,
    }
    assert all("since the plan was built" in f.message for f in failures)
    for record in (fixture.calibration, fixture.pre_generation_skip):
        survivor = fixture.reload(record)
        assert survivor.scored_run_id is None
        assert survivor.input == "the question"

    # The record that writes its own replacement needs no such check, and still migrates.
    assert fixture.reload(fixture.inline).scored_run_id is not None


def test_unreadable_inline_trace_is_left_alone(fixture):
    record = save_eval_run(
        fixture.v2_config,
        task_run_config_id=fixture.run_config.id,
        dataset_id=fixture.dataset_item.id,
        scores={"accuracy": 1.0},
        input="the question",
        output="the generated answer",
        task_run_trace="not json",
    )

    change = next(
        c for c in plan_project(fixture.project).changes if c.eval_run.id == record.id
    )
    assert change.action == MigrationAction.unmigratable
    assert "not valid JSON" in change.detail


def test_the_refusal_report_prints_paths_and_reasons_verbatim(tmp_path):
    """The one surface that tells an operator what a data-safety command refused, and
    where. Task names may contain square brackets — so every path under them does — and a
    pydantic message always ends in `[type=..., input_value=...]`; rich would read both as
    markup and eat them, leaving a path that resolves to nothing and a reason that stops
    mid-sentence."""
    project = build_project(
        tmp_path,
        task_name="Sentiment [v2]",
        input_json_schema=json.dumps(
            {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            }
        ),
    )
    task = only_task(project)
    record = save_eval_run(
        build_v2_config(build_eval(task)),
        task_run_config_id=build_run_config(task).id,
        dataset_id=build_dataset_item(task, input='{"question": "why?"}').id,
        scores={"accuracy": 1.0},
        input="plain text, not the schema",
        output="an answer",
    )
    plan = plan_project(project)

    output = io.StringIO()
    with patch.object(migration, "console", Console(file=output, width=500)):
        print_plan(plan)
    printed = output.getvalue()

    assert str(record.path) in printed
    assert "Sentiment [v2]" in printed
    assert "[type=value_error" in printed


def test_input_that_no_longer_matches_the_task_schema_is_left_alone(tmp_path):
    project = build_project(
        tmp_path,
        input_json_schema=json.dumps(
            {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            }
        ),
    )
    task = only_task(project)
    run_config = build_run_config(task)
    item = build_dataset_item(task, input='{"question": "why?"}')
    config = build_v2_config(build_eval(task))
    record = save_eval_run(
        config,
        task_run_config_id=run_config.id,
        dataset_id=item.id,
        scores={"accuracy": 1.0},
        input="plain text, not the schema",
        output="an answer",
    )

    before = snapshot(tmp_path)
    plan = plan_project(project)
    change = next(c for c in plan.changes if c.eval_run.id == record.id)

    assert change.action == MigrationAction.unmigratable
    assert apply_plan(plan) == []
    assert snapshot(tmp_path) == before


def test_a_retried_migration_reuses_the_trace_it_already_wrote(fixture):
    """An interrupted apply leaves a trace with no record pointing at it. Since Phase 4 an
    eval-generated run can't be deleted from the app, so a retry that minted a second copy
    would leave a permanent duplicate behind every time."""
    plan = plan_project(fixture.project)
    with patch.object(EvalRun, "save_to_file", side_effect=RuntimeError("disk full")):
        assert len(apply_plan(plan)) == 4

    orphans = {trace.id for trace in fixture.eval_traces()}
    assert len(orphans) == 2
    assert fixture.reload(fixture.inline).scored_run_id is None

    retry = plan_project(fixture.project)
    reused = [c for c in retry.changes if c.action == MigrationAction.reuse_trace]
    assert {change.eval_run.id for change in reused} == {
        fixture.inline.id,
        fixture.inline_eval_input.id,
    }
    assert all(change.trace is None for change in reused)

    assert apply_plan(retry) == []
    assert {trace.id for trace in fixture.eval_traces()} == orphans
    assert fixture.reload(fixture.inline).scored_run_id in orphans


def test_a_different_generation_at_the_same_key_is_not_reused(fixture):
    """Reuse is on content, not on the trace key. A trace of a *different* generation for
    the same item and run config — what the runner writes after this project ships — would
    make the score say the judge saw something it never saw."""
    other_generation = TaskRun(
        parent=fixture.task,
        input="the question",
        output=TaskOutput(
            output="a different answer",
            source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "adapter_name": "kiln_openai_compatible_adapter",
                    "model_name": "gpt-4o",
                    "model_provider": "openai",
                },
                run_config_id=fixture.run_config.id,
            ),
        ),
        eval_source=EvalItemSource(
            source_type="task_run", source_id=fixture.dataset_item.id
        ),
    )
    other_generation.save_to_file()

    change = next(
        c
        for c in plan_project(fixture.project).changes
        if c.eval_run.id == fixture.inline.id
    )

    assert change.action == MigrationAction.create_trace
    assert change.trace is not None
    assert change.trace.id != other_generation.id
    assert change.migrated is not None
    assert change.migrated.scored_run_id == change.trace.id


def rewrite_the_output(raw: dict) -> None:
    raw["output"]["output"] = "an answer the judge never saw"


def rewrite_the_id(raw: dict) -> None:
    raw["id"] = "a-different-trace"


@pytest.mark.parametrize(
    "tamper,keep_mtime",
    [
        (rewrite_the_output, False),
        (rewrite_the_output, True),
        (rewrite_the_id, False),
    ],
    ids=["different output", "different output, same mtime", "different id"],
)
def test_a_trace_rewritten_after_planning_is_not_reused(
    fixture, live_model_cache, tamper, keep_mtime
):
    """Reuse picks a trace *because of its content*, so the write-time check re-asks the
    same question. Re-asking only the id would give that guarantee up at the one moment it
    can still refuse — while the record's own copy of what was scored is still on disk.

    The id case is the other half of that check, and not redundant with the content one:
    `_same_generation` ignores a record's own identity fields, so a same-content file
    carrying a new id passes it while leaving `scored_run_id` pointing at no file at all.
    """
    plan = plan_project(fixture.project)
    with patch.object(EvalRun, "save_to_file", side_effect=RuntimeError("disk full")):
        apply_plan(plan)

    retry = plan_project(fixture.project)
    reuse = next(c for c in retry.changes if c.eval_run.id == fixture.inline.id)
    assert reuse.action == MigrationAction.reuse_trace
    assert reuse.requires_item is not None

    # A sync replaces the trace with a version the plan did not match.
    trace_path = reuse.requires_item.path
    planned_mtime = trace_path.stat().st_mtime_ns
    tampered = json.loads(trace_path.read_text())
    tamper(tampered)
    trace_path.write_text(json.dumps(tampered))
    if keep_mtime:
        os.utime(trace_path, ns=(planned_mtime, planned_mtime))

    failures = apply_plan(retry)

    assert [failure.eval_run.id for failure in failures] == [fixture.inline.id]
    # The report has to describe *this* refusal: the trace is present and parses fine, so
    # sending the operator after a missing file would send them after nothing.
    assert "not the one the plan matched" in failures[0].message
    survivor = fixture.reload(fixture.inline)
    assert survivor.scored_run_id is None
    assert survivor.output == "the generated answer"


def test_a_failed_save_is_reported_and_the_rest_still_apply(fixture):
    real_save = EvalRun.save_to_file
    calls = {"n": 0}

    def fail_first(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk full")
        return real_save(self, *args, **kwargs)

    plan = plan_project(fixture.project)
    with patch.object(EvalRun, "save_to_file", fail_first):
        failures = apply_plan(plan)

    assert len(failures) == 1
    assert "disk full" in failures[0].message

    # The other three writes still landed: one bad record does not abandon a project
    # half-migrated.
    survivors = [
        change
        for change in plan.pending_writes
        if change.eval_run.id != failures[0].eval_run.id
    ]
    assert len(survivors) == 3
    for change in survivors:
        assert fixture.reload(change.eval_run).input is None


class TestCommand:
    def test_writes_after_confirmation(self, fixture, tmp_path):
        with patch(
            "kiln_ai.cli.commands.migrate_eval_runs.typer.confirm", return_value=True
        ):
            migrate_eval_runs(tmp_path)

        assert fixture.reload(fixture.inline).scored_run_id is not None

    def test_declining_the_prompt_writes_nothing(self, fixture, tmp_path):
        before = snapshot(tmp_path)
        with patch(
            "kiln_ai.cli.commands.migrate_eval_runs.typer.confirm", return_value=False
        ):
            with pytest.raises(typer.Exit) as exc_info:
                migrate_eval_runs(tmp_path)

        assert exc_info.value.exit_code == 0
        assert snapshot(tmp_path) == before

    def test_declining_the_prompt_still_signals_what_was_found(self, fixture, tmp_path):
        """Same exit code a --dry-run of this project gives: what the command found does
        not depend on how the operator got out of it."""
        save_eval_run(
            fixture.v2_config,
            task_run_config_id="deleted-run-config",
            dataset_id=fixture.dataset_item.id,
            scores={"accuracy": 1.0},
            input="the question",
            output="the generated answer",
        )
        before = snapshot(tmp_path)

        with patch(
            "kiln_ai.cli.commands.migrate_eval_runs.typer.confirm", return_value=False
        ):
            with pytest.raises(typer.Exit) as exc_info:
                migrate_eval_runs(tmp_path)

        assert exc_info.value.exit_code == 1
        assert snapshot(tmp_path) == before

    def test_the_prompt_names_both_writes(self, fixture, tmp_path):
        """The only gate before a one-way rewrite, so it counts the new files too."""
        with patch(
            "kiln_ai.cli.commands.migrate_eval_runs.typer.confirm", return_value=True
        ) as confirm:
            migrate_eval_runs(tmp_path)

        prompt = confirm.call_args.args[0]
        assert "Update 4 eval record(s)" in prompt
        assert "create 2 task run(s)" in prompt

    def test_yes_skips_the_prompt(self, fixture, tmp_path):
        with patch("kiln_ai.cli.commands.migrate_eval_runs.typer.confirm") as confirm:
            migrate_eval_runs(tmp_path, yes=True)

        confirm.assert_not_called()
        assert fixture.reload(fixture.inline).scored_run_id is not None

    def test_nothing_to_migrate(self, fixture, tmp_path):
        migrate_eval_runs(tmp_path, yes=True)
        # Second time through there is no work, and no prompt.
        with patch("kiln_ai.cli.commands.migrate_eval_runs.typer.confirm") as confirm:
            migrate_eval_runs(tmp_path, yes=False)
        confirm.assert_not_called()

    def test_exits_nonzero_when_a_record_could_not_be_migrated(self, fixture, tmp_path):
        save_eval_run(
            fixture.v2_config,
            task_run_config_id="deleted-run-config",
            dataset_id=fixture.dataset_item.id,
            scores={"accuracy": 1.0},
            input="the question",
            output="the generated answer",
        )

        with pytest.raises(typer.Exit) as exc_info:
            migrate_eval_runs(tmp_path, yes=True)

        assert exc_info.value.exit_code == 1
        # The migratable records still migrated.
        assert fixture.reload(fixture.inline).scored_run_id is not None

    def test_dry_run_exits_nonzero_when_a_record_could_not_be_migrated(
        self, fixture, tmp_path
    ):
        """A preflight is the more likely thing to script, so it signals too."""
        save_eval_run(
            fixture.v2_config,
            task_run_config_id="deleted-run-config",
            dataset_id=fixture.dataset_item.id,
            scores={"accuracy": 1.0},
            input="the question",
            output="the generated answer",
        )
        before = snapshot(tmp_path)

        with pytest.raises(typer.Exit) as exc_info:
            migrate_eval_runs(tmp_path, dry_run=True)

        assert exc_info.value.exit_code == 1
        assert snapshot(tmp_path) == before

    def test_no_project_path(self):
        with pytest.raises(typer.Exit) as exc_info:
            migrate_eval_runs(None)
        assert exc_info.value.exit_code == 1
