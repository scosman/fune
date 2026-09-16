from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiln_ai.adapters.model_adapters.base_adapter import (
    AdapterConfig,
    BaseAdapter,
    RunOutput,
)
from kiln_ai.datamodel import (
    DataSource,
    DataSourceType,
    MessageUsage,
    Project,
    Task,
    Usage,
)
from kiln_ai.datamodel.datamodel_enums import InputType, TurnMode
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties
from kiln_ai.utils.config import Config


class MockAdapter(BaseAdapter):
    async def _run(
        self, input: InputType, trace_ref, **kwargs
    ) -> tuple[RunOutput, Usage | None]:
        return RunOutput(output="Test output", intermediate_outputs=None), None

    def adapter_name(self) -> str:
        return "mock_adapter"


@pytest.fixture
def test_task(tmp_path):
    project = Project(name="test_project", path=tmp_path / "test_project.kiln")
    project.save_to_file()
    task = Task(
        parent=project,
        name="test_task",
        instruction="Task instruction",
        turn_mode=TurnMode.multiturn,
    )
    task.save_to_file()
    return task


@pytest.fixture
def adapter(test_task):
    return MockAdapter(
        task=test_task,
        run_config=KilnAgentRunConfigProperties(
            model_name="phi_3_5",
            model_provider_name="ollama",
            prompt_id="simple_chain_of_thought_prompt_builder",
            structured_output_mode="json_schema",
        ),
    )


def test_save_run_isolation(test_task, adapter):
    input_data = "Test input"
    output_data = "Test output"
    run_output = RunOutput(
        output=output_data,
        intermediate_outputs={"chain_of_thought": "Test chain of thought"},
    )

    task_run = adapter.generate_run(
        input=input_data,
        input_source=None,
        run_output=run_output,
    )
    task_run.save_to_file()

    # Check that the task input was saved correctly
    assert task_run.parent == test_task
    assert task_run.input == input_data
    assert task_run.input_source.type == DataSourceType.human
    assert task_run.intermediate_outputs == {
        "chain_of_thought": "Test chain of thought"
    }
    created_by = Config.shared().user_id
    if created_by and created_by != "":
        assert task_run.input_source.properties["created_by"] == created_by
    else:
        assert "created_by" not in task_run.input_source.properties

    # Check that the task output was saved correctly
    saved_output = task_run.output
    assert saved_output.output == output_data
    assert saved_output.source.type == DataSourceType.synthetic
    assert saved_output.rating is None

    # Verify that the data can be read back from disk
    reloaded_task = Task.load_from_file(test_task.path)
    reloaded_runs = reloaded_task.runs(include_intermediate_runs=True)
    assert len(reloaded_runs) == 1
    reloaded_run = reloaded_runs[0]
    assert reloaded_run.input == input_data
    assert reloaded_run.input_source.type == DataSourceType.human
    reloaded_output = reloaded_run.output

    reloaded_output = reloaded_run.output
    assert reloaded_output.output == output_data
    assert reloaded_output.source.type == DataSourceType.synthetic
    assert reloaded_output.rating is None
    assert reloaded_output.source.properties["adapter_name"] == "mock_adapter"
    assert reloaded_output.source.properties["model_name"] == "phi_3_5"
    assert reloaded_output.source.properties["model_provider"] == "ollama"
    assert (
        reloaded_output.source.properties["prompt_id"]
        == "simple_chain_of_thought_prompt_builder"
    )
    assert reloaded_output.source.properties["structured_output_mode"] == "json_schema"
    assert reloaded_output.source.properties["temperature"] == 1.0
    assert reloaded_output.source.properties["top_p"] == 1.0
    # Run again, with same input and different output. Should create a new TaskRun.
    different_run_output = RunOutput(
        output="Different output", intermediate_outputs=None
    )
    task_output = adapter.generate_run(input_data, None, different_run_output)
    task_output.save_to_file()
    assert len(test_task.runs(include_intermediate_runs=True)) == 2
    assert "Different output" in set(
        run.output.output for run in test_task.runs(include_intermediate_runs=True)
    )

    # run again with input of different type. Should create a new TaskRun and TaskOutput.
    task_output = adapter.generate_run(
        input_data,
        DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": "mock_model",
                "model_provider": "mock_provider",
                "prompt_id": "mock_prompt_builder",
                "adapter_name": "mock_adapter",
            },
        ),
        run_output,
    )
    task_output.save_to_file()
    assert len(test_task.runs(include_intermediate_runs=True)) == 3
    assert task_output.input == input_data
    assert task_output.input_source.type == DataSourceType.synthetic
    assert "Different output" in set(
        run.output.output for run in test_task.runs(include_intermediate_runs=True)
    )
    assert output_data in set(
        run.output.output for run in test_task.runs(include_intermediate_runs=True)
    )


def test_generate_run_non_ascii(test_task, adapter):
    input_data = {"key": "input with non-ascii character: 你好"}
    output_data = {"key": "output with non-ascii character: 你好"}
    run_output = RunOutput(
        output=output_data,
        intermediate_outputs=None,
    )

    task_run = adapter.generate_run(
        input=input_data,
        input_source=None,
        run_output=run_output,
    )
    task_run.save_to_file()

    # as these values are saved as strings, they should properly represent the non-ascii characters
    assert task_run.input == '{"key": "input with non-ascii character: 你好"}'
    assert task_run.output.output == '{"key": "output with non-ascii character: 你好"}'

    # check that the stringified unicode strings can be read back from the file
    reloaded_task = Task.load_from_file(test_task.path)
    reloaded_runs = reloaded_task.runs(include_intermediate_runs=True)
    assert len(reloaded_runs) == 1
    reloaded_run = reloaded_runs[0]
    assert reloaded_run.input == '{"key": "input with non-ascii character: 你好"}'
    assert (
        reloaded_run.output.output == '{"key": "output with non-ascii character: 你好"}'
    )


@pytest.mark.asyncio
async def test_autosave_false(test_task, adapter):
    with patch("kiln_ai.utils.config.Config.shared") as mock_shared:
        mock_config = mock_shared.return_value
        mock_config.autosave_runs = False
        mock_config.user_id = "test_user"

        input_data = "Test input"

        run = await adapter.invoke(input_data)

        # Check that no runs were saved
        assert len(test_task.runs(include_intermediate_runs=True)) == 0

        # Check that the run ID is not set
        assert run.id is None


@pytest.mark.asyncio
async def test_autosave_true_with_disabled(test_task, adapter):
    with patch("kiln_ai.utils.config.Config.shared") as mock_shared:
        mock_config = mock_shared.return_value
        mock_config.autosave_runs = True
        mock_config.user_id = "test_user"

        input_data = "Test input"

        adapter.base_adapter_config.allow_saving = False
        run = await adapter.invoke(input_data)

        # Check that no runs were saved
        assert len(test_task.runs(include_intermediate_runs=True)) == 0

        # Check that the run ID is not set
        assert run.id is None


@pytest.mark.asyncio
async def test_autosave_true(test_task, adapter):
    with patch("kiln_ai.utils.config.Config.shared") as mock_shared:
        mock_config = mock_shared.return_value
        mock_config.autosave_runs = True
        mock_config.user_id = "test_user"

        input_data = "Test input"

        run = await adapter.invoke(input_data)

        # Check that the run ID is set
        assert run.id is not None

        # Check that an task input was saved
        task_runs = test_task.runs(include_intermediate_runs=True)
        assert len(task_runs) == 1
        assert task_runs[0].input == input_data
        assert task_runs[0].input_source.type == DataSourceType.human

        output = task_runs[0].output
        assert output.output == "Test output"
        assert output.source.type == DataSourceType.synthetic
        assert output.source.properties["adapter_name"] == "mock_adapter"
        assert output.source.properties["model_name"] == "phi_3_5"
        assert output.source.properties["model_provider"] == "ollama"
        assert (
            output.source.properties["prompt_id"]
            == "simple_chain_of_thought_prompt_builder"
        )
        assert output.source.properties["structured_output_mode"] == "json_schema"
        assert output.source.properties["temperature"] == 1.0
        assert output.source.properties["top_p"] == 1.0


@pytest.mark.asyncio
async def test_invoke_continue_session(test_task, adapter):
    """Test that invoke with prior_trace continues a session and creates a new run."""
    with patch("kiln_ai.utils.config.Config.shared") as mock_shared:
        mock_config = mock_shared.return_value
        mock_config.autosave_runs = True
        mock_config.user_id = "test_user"

        trace = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        async def mock_run(input, trace_ref, **kwargs):
            prior_trace = kwargs.get("prior_trace")
            if prior_trace is not None:
                extended_trace = [
                    *prior_trace,
                    {"role": "user", "content": input},
                    {"role": "assistant", "content": "How can I help?"},
                ]
                return (
                    RunOutput(
                        output="How can I help?",
                        intermediate_outputs=None,
                        trace=extended_trace,
                    ),
                    None,
                )
            return RunOutput(output="Test output", intermediate_outputs=None), None

        adapter._run = mock_run

        with (
            patch.object(
                adapter,
                "model_provider",
                return_value=MagicMock(
                    parser="default",
                    formatter=None,
                    reasoning_capable=False,
                ),
            ),
            patch(
                "kiln_ai.adapters.model_adapters.base_adapter.model_parser_from_id"
            ) as mock_parser_from_id,
        ):
            mock_parser = MagicMock()
            mock_parser.parse_output.return_value = RunOutput(
                output="How can I help?",
                intermediate_outputs=None,
                trace=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "Tell me more"},
                    {"role": "assistant", "content": "How can I help?"},
                ],
            )
            mock_parser_from_id.return_value = mock_parser

            new_run = await adapter.invoke("Tell me more", prior_trace=trace)

        assert new_run.id is not None
        assert new_run.input == "Tell me more"
        assert new_run.output.output == "How can I help?"
        assert len(new_run.trace) == 4
        assert new_run.trace[-2]["content"] == "Tell me more"
        assert new_run.trace[-1]["content"] == "How can I help?"

        reloaded = Task.load_from_file(test_task.path)
        runs = reloaded.runs(include_intermediate_runs=True)
        assert len(runs) == 1
        assert runs[0].output.output == "How can I help?"


@pytest.mark.asyncio
async def test_invoke_with_empty_prior_trace_starts_fresh(test_task, adapter):
    """Test that invoke with prior_trace=[] starts a fresh conversation (no error)."""
    with patch("kiln_ai.utils.config.Config.shared") as mock_shared:
        mock_config = mock_shared.return_value
        mock_config.autosave_runs = True
        mock_config.user_id = "test_user"

        adapter._run = AsyncMock(
            return_value=(
                RunOutput(output="Fresh reply", intermediate_outputs=None, trace=None),
                None,
            )
        )
        with (
            patch.object(
                adapter,
                "model_provider",
                return_value=MagicMock(
                    parser="default",
                    formatter=None,
                    reasoning_capable=False,
                ),
            ),
            patch(
                "kiln_ai.adapters.model_adapters.base_adapter.model_parser_from_id",
                return_value=MagicMock(
                    parse_output=MagicMock(
                        return_value=RunOutput(
                            output="Fresh reply",
                            intermediate_outputs=None,
                            trace=None,
                        )
                    )
                ),
            ),
            patch(
                "kiln_ai.adapters.model_adapters.base_adapter.request_formatter_from_id",
            ),
        ):
            run = await adapter.invoke("Follow up", prior_trace=[])
        assert run.output.output == "Fresh reply"


def test_generate_run_always_creates_new_task_run(test_task, adapter):
    trace = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    run1 = adapter.generate_run(
        input="hi",
        input_source=None,
        run_output=RunOutput(
            output="hello",
            intermediate_outputs={"chain_of_thought": "old"},
            trace=trace,
        ),
        usage=Usage(input_tokens=10, output_tokens=20),
        trace=trace,
    )
    extended_trace = [
        *trace,
        {"role": "user", "content": "follow-up"},
        {"role": "assistant", "content": "ok"},
    ]
    run2 = adapter.generate_run(
        input="follow-up",
        input_source=None,
        run_output=RunOutput(
            output="ok",
            intermediate_outputs={"new_key": "new_val"},
            trace=extended_trace,
        ),
        usage=Usage(input_tokens=5, output_tokens=10),
        trace=extended_trace,
    )
    assert run2 is not run1
    assert run2.usage is not None and run2.usage.input_tokens == 5
    assert run2.usage.output_tokens == 10
    assert run2.intermediate_outputs == {"new_key": "new_val"}
    assert run2.output.output == "ok"


def test_properties_for_task_output_custom_values(test_task):
    """Test that _properties_for_task_output includes custom temperature, top_p, and structured_output_mode"""
    adapter = MockAdapter(
        task=test_task,
        run_config=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name="openai",
            prompt_id="simple_prompt_builder",
            temperature=0.7,
            top_p=0.9,
            structured_output_mode="json_schema",
        ),
    )

    input_data = "Test input"
    output_data = "Test output"
    run_output = RunOutput(output=output_data, intermediate_outputs=None)

    task_run = adapter.generate_run(
        input=input_data,
        input_source=None,
        run_output=run_output,
    )
    task_run.save_to_file()

    # Verify custom values are preserved in properties
    output = task_run.output
    assert output.source.properties["adapter_name"] == "mock_adapter"
    assert output.source.properties["model_name"] == "gpt-4"
    assert output.source.properties["model_provider"] == "openai"
    assert output.source.properties["prompt_id"] == "simple_prompt_builder"
    assert output.source.properties["structured_output_mode"] == "json_schema"
    assert output.source.properties["temperature"] == 0.7
    assert output.source.properties["top_p"] == 0.9


def test_generate_run_with_parent_task_run_sets_parent_task_run_id(test_task, adapter):
    prior_run = adapter.generate_run(
        input="prior input",
        input_source=None,
        run_output=RunOutput(output="prior output", intermediate_outputs=None),
    )
    prior_run.save_to_file()
    assert prior_run.id is not None
    assert prior_run.parent_task_run_id is None

    new_run = adapter.generate_run(
        input="new input",
        input_source=None,
        run_output=RunOutput(output="new output", intermediate_outputs=None),
        parent_task_run=prior_run,
    )

    assert new_run.parent_task_run_id == prior_run.id
    new_run.save_to_file()

    reloaded_task = Task.load_from_file(test_task.path)
    task_runs = reloaded_task.runs(include_intermediate_runs=True)
    assert len(task_runs) == 2
    by_id = {r.id: r for r in task_runs}
    assert by_id[prior_run.id].parent_task_run_id is None
    assert by_id[new_run.id].parent_task_run_id == prior_run.id
    assert by_id[new_run.id].output.output == "new output"


def test_generate_run_without_parent_task_run_defaults_to_task(test_task, adapter):
    """Test that generate_run without parent_task_run leaves parent_task_run_id unset."""
    run = adapter.generate_run(
        input="input",
        input_source=None,
        run_output=RunOutput(output="output", intermediate_outputs=None),
    )
    assert run.parent_task_run_id is None
    assert run.parent == test_task


def test_generate_run_records_task_run_config_id_from_adapter_config(
    test_task, adapter
):
    adapter.base_adapter_config = AdapterConfig(task_run_config_id="rc_abc123")
    run = adapter.generate_run(
        input="input",
        input_source=None,
        run_output=RunOutput(output="output", intermediate_outputs=None),
    )
    assert run.output.source.run_config_id == "rc_abc123"
    run.save_to_file()

    reloaded_task = Task.load_from_file(test_task.path)
    reloaded_run = reloaded_task.runs()[0]
    assert reloaded_run.output.source.run_config_id == "rc_abc123"


def test_generate_run_defaults_task_run_config_id_to_none(test_task, adapter):
    # AdapterConfig() defaults task_run_config_id to None; the output's source
    # inherits that and persists run_config_id=None.
    run = adapter.generate_run(
        input="input",
        input_source=None,
        run_output=RunOutput(output="output", intermediate_outputs=None),
    )
    assert run.output.source.run_config_id is None


def test_generate_run_with_unsaved_parent_task_run_raises(adapter):
    prior_run = adapter.generate_run(
        input="prior input",
        input_source=None,
        run_output=RunOutput(output="prior output", intermediate_outputs=None),
    )
    prior_run.id = None

    with pytest.raises(ValueError, match="parent_task_run must be persisted"):
        adapter.generate_run(
            input="new input",
            input_source=None,
            run_output=RunOutput(output="new output", intermediate_outputs=None),
            parent_task_run=prior_run,
        )


@pytest.mark.asyncio
async def test_invoke_with_parent_task_run_saves_under_task_with_link(
    test_task, adapter
):
    trace = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    prior_run = adapter.generate_run(
        input="Hello",
        input_source=None,
        run_output=RunOutput(
            output="Hi there!", intermediate_outputs=None, trace=trace
        ),
        trace=trace,
    )
    prior_run.save_to_file()
    assert prior_run.id is not None

    continuation_trace = [
        *trace,
        {"role": "user", "content": "Tell me more"},
        {"role": "assistant", "content": "More details!"},
    ]
    continuation_output = RunOutput(
        output="More details!",
        intermediate_outputs=None,
        trace=continuation_trace,
    )

    adapter._run = AsyncMock(return_value=(continuation_output, None))

    with (
        patch("kiln_ai.utils.config.Config.shared") as mock_shared,
        patch.object(
            adapter,
            "model_provider",
            return_value=MagicMock(
                parser="default",
                formatter=None,
                reasoning_capable=False,
            ),
        ),
        patch(
            "kiln_ai.adapters.model_adapters.base_adapter.model_parser_from_id",
            return_value=MagicMock(
                parse_output=MagicMock(return_value=continuation_output)
            ),
        ),
    ):
        mock_shared.return_value.autosave_runs = True
        mock_shared.return_value.user_id = "test_user"

        new_run = await adapter.invoke(
            "Tell me more",
            prior_trace=trace,
            parent_task_run=prior_run,
        )

    assert new_run.id is not None
    assert new_run.parent_task_run_id == prior_run.id

    reloaded_task = Task.load_from_file(test_task.path)
    task_runs = reloaded_task.runs(include_intermediate_runs=True)
    assert len(task_runs) == 2
    by_id = {r.id: r for r in task_runs}
    assert by_id[new_run.id].output.output == "More details!"
    assert by_id[new_run.id].parent_task_run_id == prior_run.id


def test_generate_run_sets_cumulative_usage_from_trace(test_task, adapter):
    """generate_run sums per-message usage across the trace into cumulative_usage."""
    trace = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "hello",
            "usage": MessageUsage(input_tokens=10, output_tokens=20, cost=0.1),
        },
        {"role": "user", "content": "more"},
        {
            "role": "assistant",
            "content": "ok",
            "usage": MessageUsage(input_tokens=5, output_tokens=15, cost=0.2),
        },
    ]

    task_run = adapter.generate_run(
        input="hi",
        input_source=None,
        run_output=RunOutput(output="ok", intermediate_outputs=None, trace=trace),
        usage=Usage(input_tokens=15, output_tokens=35, cost=0.3),
        trace=trace,
    )

    assert task_run.cumulative_usage is not None
    assert task_run.cumulative_usage.input_tokens == 15
    assert task_run.cumulative_usage.output_tokens == 35
    assert task_run.cumulative_usage.cost == pytest.approx(0.3)


def test_generate_run_sets_empty_cumulative_usage_when_trace_is_none(
    test_task, adapter
):
    """No trace → cumulative_usage is an empty MessageUsage (all-None fields), not None."""
    task_run = adapter.generate_run(
        input="hi",
        input_source=None,
        run_output=RunOutput(output="ok", intermediate_outputs=None),
    )

    assert task_run.cumulative_usage == MessageUsage()


def test_generate_run_fresh_run_cumulative_equals_usage(test_task, adapter):
    """For a fresh run (trace contains only this run's messages), cumulative_usage's
    aggregatable fields equal those of usage. cumulative_usage is a MessageUsage
    (no latency); usage is a Usage."""
    fresh_usage = Usage(input_tokens=12, output_tokens=8, cost=0.42)
    fresh_message_usage = MessageUsage(input_tokens=12, output_tokens=8, cost=0.42)
    trace = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "hello",
            "usage": fresh_message_usage,
        },
    ]

    task_run = adapter.generate_run(
        input="hi",
        input_source=None,
        run_output=RunOutput(output="hello", intermediate_outputs=None, trace=trace),
        usage=fresh_usage,
        trace=trace,
    )

    assert task_run.usage == fresh_usage
    assert task_run.cumulative_usage == fresh_message_usage


def test_generate_run_seeded_run_cumulative_includes_prior_trace_usage(
    test_task, adapter
):
    """Seeded run: cumulative_usage = this run's usage + prior trace's per-message usage."""
    seeded_usage = Usage(input_tokens=100, output_tokens=200, cost=0.5)
    new_usage = Usage(input_tokens=10, output_tokens=20, cost=0.05)

    full_trace = [
        {"role": "user", "content": "hi"},
        # Seeded prior turn (already had its own usage from a previous run).
        {
            "role": "assistant",
            "content": "hello",
            "usage": seeded_usage,
        },
        {"role": "user", "content": "follow-up"},
        # New turn produced by this run.
        {
            "role": "assistant",
            "content": "ack",
            "usage": new_usage,
        },
    ]

    # `usage` excludes the seeded portion (it's the new-turn-only running total).
    task_run = adapter.generate_run(
        input="follow-up",
        input_source=None,
        run_output=RunOutput(output="ack", intermediate_outputs=None, trace=full_trace),
        usage=new_usage,
        trace=full_trace,
    )

    assert task_run.usage == new_usage
    # cumulative_usage spans the whole trace.
    assert task_run.cumulative_usage is not None
    assert task_run.cumulative_usage.input_tokens == 110
    assert task_run.cumulative_usage.output_tokens == 220
    assert task_run.cumulative_usage.cost == pytest.approx(0.55)
    # And it strictly exceeds usage on every populated field.
    assert task_run.cumulative_usage.input_tokens > (task_run.usage.input_tokens or 0)
    assert (task_run.cumulative_usage.cost or 0) > (task_run.usage.cost or 0)
