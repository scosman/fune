import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
)

from kiln_ai.adapters.errors import StructuredOutputParseError
from kiln_ai.adapters.ml_model_list import KilnModelProvider, StructuredOutputMode
from kiln_ai.adapters.model_adapters.base_adapter import (
    AdapterConfig,
    BaseAdapter,
    RunOutput,
    assemble_unique_agent_tools,
)
from kiln_ai.adapters.model_adapters.stream_events import (
    AiSdkEventType,
    ToolCallEvent,
    ToolCallEventType,
)
from kiln_ai.adapters.prompt_builders import BasePromptBuilder
from kiln_ai.adapters.retry_classification import is_retryable_error
from kiln_ai.datamodel import Task, TaskRun, Usage
from kiln_ai.datamodel.datamodel_enums import ChatStrategy, ModelProviderName
from kiln_ai.datamodel.project import Project
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties, ToolsRunConfig
from kiln_ai.datamodel.skill import Skill
from kiln_ai.datamodel.tool_id import KilnBuiltInToolId
from kiln_ai.tools.base_tool import KilnToolInterface
from kiln_ai.utils.open_ai_types import ChatCompletionMessageParam


class MockAdapter(BaseAdapter):
    """Concrete implementation of BaseAdapter for testing"""

    async def _run(self, input, trace_ref, **kwargs):
        return None, None

    def adapter_name(self) -> str:
        return "test"


@pytest.fixture
def mock_provider():
    return KilnModelProvider(
        name="openai",
    )


@pytest.fixture
def base_project():
    return Project(name="test_project", description="test project description")


@pytest.fixture
def base_task(base_project):
    task = Task(name="test_task", instruction="test_instruction", parent=base_project)
    return task


@pytest.fixture
def adapter(base_task):
    return MockAdapter(
        task=base_task,
        run_config=KilnAgentRunConfigProperties(
            model_name="test_model",
            model_provider_name="openai",
            prompt_id="simple_prompt_builder",
            structured_output_mode="json_schema",
        ),
    )


@pytest.fixture
def mock_formatter():
    formatter = MagicMock()
    formatter.format_input.return_value = {"formatted": "input"}
    return formatter


@pytest.fixture
def mock_parser():
    parser = MagicMock()
    parser.parse_output.return_value = RunOutput(
        output="test output", intermediate_outputs={}
    )
    return parser


async def test_model_provider_uses_cache(adapter, mock_provider):
    """Test that cached provider is returned if it exists"""
    # Set up cached provider
    adapter._model_provider = mock_provider

    # Mock the provider loader to ensure it's not called
    with patch(
        "kiln_ai.adapters.model_adapters.base_adapter.kiln_model_provider_from"
    ) as mock_loader:
        provider = adapter.model_provider()

        assert provider == mock_provider
        mock_loader.assert_not_called()


async def test_model_provider_loads_and_caches(adapter, mock_provider):
    """Test that provider is loaded and cached if not present"""
    # Ensure no cached provider
    adapter._model_provider = None

    # Mock the provider loader
    with patch(
        "kiln_ai.adapters.model_adapters.base_adapter.kiln_model_provider_from"
    ) as mock_loader:
        mock_loader.return_value = mock_provider

        # First call should load and cache
        provider1 = adapter.model_provider()
        assert provider1 == mock_provider
        mock_loader.assert_called_once_with("test_model", "openai")

        # Second call should use cache
        mock_loader.reset_mock()
        provider2 = adapter.model_provider()
        assert provider2 == mock_provider
        mock_loader.assert_not_called()


async def test_model_provider_invalid_provider_model_name(base_project):
    """Test error when model or provider name is missing"""
    # Create a task with a parent project
    task = Task(name="test_task", instruction="test_instruction", parent=base_project)

    # Test with missing model name
    with pytest.raises(ValueError, match="Input should be"):
        MockAdapter(
            task=task,
            run_config=KilnAgentRunConfigProperties(
                model_name="test_model",
                model_provider_name="invalid",
                prompt_id="simple_prompt_builder",
            ),
        )


async def test_model_provider_missing_model_names(base_project):
    """Test error when model or provider name is missing"""
    # Create a task with a parent project
    task = Task(name="test_task", instruction="test_instruction", parent=base_project)

    # Test with missing model name
    adapter = MockAdapter(
        task=task,
        run_config=KilnAgentRunConfigProperties(
            model_name="",
            model_provider_name="openai",
            prompt_id="simple_prompt_builder",
            structured_output_mode="json_schema",
        ),
    )
    with pytest.raises(
        ValueError, match="model_name and model_provider_name must be provided"
    ):
        await adapter.model_provider()


async def test_model_provider_not_found(adapter):
    """Test error when provider loader returns None"""
    # Mock the provider loader to return None
    with patch(
        "kiln_ai.adapters.model_adapters.base_adapter.kiln_model_provider_from"
    ) as mock_loader:
        mock_loader.return_value = None

        with pytest.raises(
            ValueError,
            match="not found for model test_model",
        ):
            await adapter.model_provider()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output_schema,structured_output_mode,expected_json_instructions",
    [
        (False, StructuredOutputMode.json_instructions, False),
        (True, StructuredOutputMode.json_instructions, True),
        (False, StructuredOutputMode.json_instruction_and_object, False),
        (True, StructuredOutputMode.json_instruction_and_object, True),
        (True, StructuredOutputMode.json_mode, False),
        (False, StructuredOutputMode.json_mode, False),
    ],
)
async def test_prompt_builder_json_instructions(
    base_task,
    adapter,
    output_schema,
    structured_output_mode,
    expected_json_instructions,
):
    """Test that prompt builder is called with correct include_json_instructions value"""
    # Mock the prompt builder and has_structured_output method
    mock_prompt_builder = MagicMock()
    adapter.prompt_builder = mock_prompt_builder
    adapter.model_provider_name = "openai"
    adapter.has_structured_output = MagicMock(return_value=output_schema)
    adapter.run_config.structured_output_mode = structured_output_mode

    # Test
    adapter.build_prompt()
    mock_prompt_builder.build_prompt.assert_called_with(
        include_json_instructions=expected_json_instructions,
        skills=[],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "formatter_id,expected_input,expected_calls",
    [
        (None, {"original": "input"}, 0),  # No formatter
        ("test_formatter", {"formatted": "input"}, 1),  # With formatter
    ],
)
async def test_input_formatting(
    adapter, mock_formatter, mock_parser, formatter_id, expected_input, expected_calls
):
    """Test that input formatting is handled correctly based on formatter configuration"""
    # Mock the model provider to return our formatter ID and parser
    provider = MagicMock()
    provider.formatter = formatter_id
    provider.parser = "test_parser"
    provider.reasoning_capable = False
    adapter.model_provider = MagicMock(return_value=provider)

    # Mock the formatter factory and parser factory
    with (
        patch(
            "kiln_ai.adapters.model_adapters.base_adapter.request_formatter_from_id"
        ) as mock_factory,
        patch(
            "kiln_ai.adapters.model_adapters.base_adapter.model_parser_from_id"
        ) as mock_parser_factory,
    ):
        mock_factory.return_value = mock_formatter
        mock_parser_factory.return_value = mock_parser

        # Mock the _run method to capture the input
        captured_input = None

        async def mock_run(input, messages=None, **kwargs):
            nonlocal captured_input
            captured_input = input
            return RunOutput(output="test output", intermediate_outputs={}), None

        adapter._run = mock_run

        # Run the adapter
        original_input = {"original": "input"}
        await adapter.invoke_returning_run_output(original_input)

        # Verify formatter was called correctly
        assert captured_input == expected_input
        assert mock_factory.call_count == (1 if formatter_id else 0)
        assert mock_formatter.format_input.call_count == expected_calls

        # Verify original input was preserved in the run
        if formatter_id:
            mock_formatter.format_input.assert_called_once_with(original_input)


async def test_properties_for_task_output_includes_all_run_config_properties(adapter):
    """Test that all properties from KilnAgentRunConfigProperties are saved in task output properties"""
    # Get all field names from KilnAgentRunConfigProperties
    run_config_properties_fields = set(KilnAgentRunConfigProperties.model_fields.keys())

    # Get the properties saved by the adapter
    saved_properties = adapter._properties_for_task_output()
    saved_property_keys = set(saved_properties.keys())

    # Check which KilnAgentRunConfigProperties fields are missing from saved properties
    # Note: model_provider_name becomes model_provider in saved properties
    expected_mappings = {
        "model_name": "model_name",
        "model_provider_name": "model_provider",
        "prompt_id": "prompt_id",
        "temperature": "temperature",
        "top_p": "top_p",
        "structured_output_mode": "structured_output_mode",
        "thinking_level": None,
        "type": None,
        "tools_config": None,
        "input_transform": None,
    }

    missing_properties = []
    for field_name in run_config_properties_fields:
        expected_key = expected_mappings.get(field_name, field_name)
        if expected_key is not None and expected_key not in saved_property_keys:
            missing_properties.append(
                f"KilnAgentRunConfigProperties.{field_name} -> {expected_key}"
            )

    assert not missing_properties, (
        f"The following KilnAgentRunConfigProperties fields are not saved by _properties_for_task_output: {missing_properties}. Please update the method to include them."
    )


async def test_properties_for_task_output_catches_missing_new_property(adapter):
    """Test that demonstrates our test will catch when new properties are added to KilnAgentRunConfigProperties but not to _properties_for_task_output"""
    # Simulate what happens if a new property was added to KilnAgentRunConfigProperties
    # We'll mock the model_fields to include a fake new property
    original_fields = KilnAgentRunConfigProperties.model_fields.copy()

    # Create a mock field to simulate a new property being added
    from pydantic.fields import FieldInfo

    mock_field = FieldInfo(annotation=str, default="default_value")

    try:
        # Add a fake new field to simulate someone adding a property
        KilnAgentRunConfigProperties.model_fields["new_fake_property"] = mock_field

        # Get all field names from KilnAgentRunConfigProperties (now includes our fake property)
        run_config_properties_fields = set(
            KilnAgentRunConfigProperties.model_fields.keys()
        )

        # Get the properties saved by the adapter (won't include our fake property)
        saved_properties = adapter._properties_for_task_output()
        saved_property_keys = set(saved_properties.keys())

        # The mappings don't include our fake property
        expected_mappings = {
            "model_name": "model_name",
            "model_provider_name": "model_provider",
            "prompt_id": "prompt_id",
            "temperature": "temperature",
            "top_p": "top_p",
            "structured_output_mode": "structured_output_mode",
            "thinking_level": None,
            "type": None,
            "tools_config": None,
            "input_transform": None,
        }

        missing_properties = []
        for field_name in run_config_properties_fields:
            expected_key = expected_mappings.get(field_name, field_name)
            if expected_key is not None and expected_key not in saved_property_keys:
                missing_properties.append(
                    f"KilnAgentRunConfigProperties.{field_name} -> {expected_key}"
                )

        # This should find our missing fake property
        assert missing_properties == [
            "KilnAgentRunConfigProperties.new_fake_property -> new_fake_property"
        ], f"Expected to find missing fake property, but got: {missing_properties}"

    finally:
        # Restore the original fields
        KilnAgentRunConfigProperties.model_fields.clear()
        KilnAgentRunConfigProperties.model_fields.update(original_fields)


@pytest.mark.parametrize(
    "cot_prompt,tuned_strategy,reasoning_capable,expected_formatter_class",
    [
        # No COT prompt -> always single turn
        (None, None, False, "SingleTurnFormatter"),
        (None, ChatStrategy.two_message_cot, False, "SingleTurnFormatter"),
        (None, ChatStrategy.single_turn_r1_thinking, True, "SingleTurnFormatter"),
        # With COT prompt:
        # - Tuned strategy takes precedence (except single turn)
        (
            "think step by step",
            ChatStrategy.two_message_cot,
            False,
            "TwoMessageCotFormatter",
        ),
        (
            "think step by step",
            ChatStrategy.single_turn_r1_thinking,
            False,
            "SingleTurnR1ThinkingFormatter",
        ),
        # - Tuned single turn is ignored when COT exists
        (
            "think step by step",
            ChatStrategy.single_turn,
            True,
            "SingleTurnR1ThinkingFormatter",
        ),
        # - Reasoning capable -> single turn R1 thinking
        ("think step by step", None, True, "SingleTurnR1ThinkingFormatter"),
        # - Not reasoning capable -> two message COT
        ("think step by step", None, False, "TwoMessageCotFormatter"),
    ],
)
def test_build_chat_formatter(
    adapter,
    cot_prompt,
    tuned_strategy,
    reasoning_capable,
    expected_formatter_class,
):
    """Test chat formatter strategy selection based on COT prompt, tuned strategy, and model capabilities"""
    # Mock the prompt builder
    mock_prompt_builder = MagicMock()
    mock_prompt_builder.chain_of_thought_prompt.return_value = cot_prompt
    mock_prompt_builder.build_prompt.return_value = "system message"
    adapter.prompt_builder = mock_prompt_builder

    # Mock the model provider
    mock_provider = MagicMock()
    mock_provider.tuned_chat_strategy = tuned_strategy
    mock_provider.reasoning_capable = reasoning_capable
    adapter.model_provider = MagicMock(return_value=mock_provider)

    # Get the formatter
    formatter = adapter.build_chat_formatter("test input")

    # Verify the formatter type
    assert formatter.__class__.__name__ == expected_formatter_class

    # Verify the formatter was created with correct parameters
    assert formatter.system_message == "system message"
    assert formatter.user_input == "test input"
    # Only check thinking_instructions for formatters that use it
    if expected_formatter_class == "TwoMessageCotFormatter":
        if cot_prompt:
            assert formatter.thinking_instructions == cot_prompt
        else:
            assert formatter.thinking_instructions is None
    # For other formatters, don't assert thinking_instructions

    # Verify prompt builder was called correctly
    mock_prompt_builder.build_prompt.assert_called_once()
    mock_prompt_builder.chain_of_thought_prompt.assert_called_once()


def test_build_chat_formatter_with_prior_trace_returns_multiturn_formatter(adapter):
    prior_trace = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    formatter = adapter.build_chat_formatter("new input", prior_trace=prior_trace)
    assert formatter.__class__.__name__ == "MultiturnFormatter"
    assert formatter.initial_messages() == prior_trace


def test_build_chat_formatter_empty_prior_trace_matches_none(adapter):
    fmt_empty = adapter.build_chat_formatter("new input", prior_trace=[])
    fmt_none = adapter.build_chat_formatter("new input", prior_trace=None)
    assert type(fmt_empty) is type(fmt_none)
    assert fmt_empty.__class__.__name__ != "MultiturnFormatter"


@pytest.mark.asyncio
async def test_invoke_with_prior_trace_none_starts_fresh(base_project):
    task = Task(
        name="test_task",
        instruction="test_instruction",
        parent=base_project,
    )
    adapter = MockAdapter(
        task=task,
        run_config=KilnAgentRunConfigProperties(
            model_name="gpt_4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
    )
    adapter._run = AsyncMock(
        return_value=(
            RunOutput(output="ok", intermediate_outputs=None, trace=None),
            None,
        )
    )
    with (
        patch(
            "kiln_ai.adapters.model_adapters.base_adapter.model_parser_from_id",
            return_value=MagicMock(
                parse_output=MagicMock(
                    return_value=RunOutput(
                        output="ok", intermediate_outputs=None, trace=None
                    )
                )
            ),
        ),
        patch(
            "kiln_ai.adapters.model_adapters.base_adapter.request_formatter_from_id",
        ),
        patch.object(
            adapter,
            "model_provider",
            return_value=MagicMock(
                parser="default",
                formatter=None,
                reasoning_capable=False,
            ),
        ),
    ):
        run = await adapter.invoke("input", prior_trace=None)
    assert isinstance(run, TaskRun)
    assert run.output.output == "ok"
    adapter._run.assert_called_once()
    assert adapter._run.call_args[1].get("prior_trace") is None


@pytest.mark.asyncio
async def test_invoke_returning_run_output_passes_prior_trace_to_run(
    adapter, mock_parser, tmp_path
):
    project = Project(name="proj", path=tmp_path / "proj.kiln")
    project.save_to_file()
    task = Task(
        name="t",
        instruction="i",
        parent=project,
    )
    task.save_to_file()
    adapter.task = task

    trace = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    captured_prior_trace = None

    async def mock_run(input, messages=None, **kwargs):
        nonlocal captured_prior_trace
        captured_prior_trace = kwargs.get("prior_trace")
        return RunOutput(output="ok", intermediate_outputs=None, trace=trace), None

    adapter._run = mock_run

    provider = MagicMock()
    provider.parser = "test_parser"
    provider.formatter = None
    provider.reasoning_capable = False
    adapter.model_provider = MagicMock(return_value=provider)
    mock_parser.parse_output.return_value = RunOutput(
        output="ok", intermediate_outputs=None, trace=trace
    )

    with (
        patch(
            "kiln_ai.adapters.model_adapters.base_adapter.model_parser_from_id",
            return_value=mock_parser,
        ),
        patch(
            "kiln_ai.adapters.model_adapters.base_adapter.request_formatter_from_id",
        ),
    ):
        await adapter.invoke_returning_run_output("follow-up", prior_trace=trace)

    assert captured_prior_trace == trace


_INPUT_OBJECT_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"x": {"type": "number"}},
        "required": ["x"],
    }
)

_MULTITURN_STRUCTURED_ERROR = (
    "Cannot run multiturn execution with a task that has a structured input schema"
)


def test_normalize_prior_trace_empty_and_none():
    assert BaseAdapter._normalize_prior_trace(None) is None
    assert BaseAdapter._normalize_prior_trace([]) is None
    trace: list[ChatCompletionMessageParam] = [{"role": "user", "content": "h"}]
    assert BaseAdapter._normalize_prior_trace(trace) == trace


@pytest.mark.asyncio
async def test_invoke_rejects_multiturn_with_structured_input(tmp_path):
    project = Project(name="proj", path=tmp_path / "proj.kiln")
    project.save_to_file()
    task = Task(
        name="t",
        instruction="i",
        parent=project,
    )
    task.save_to_file()
    adapter = MockAdapter(
        task=task,
        run_config=KilnAgentRunConfigProperties(
            model_name="gpt_4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
    )
    adapter.input_schema = _INPUT_OBJECT_SCHEMA
    adapter._run = AsyncMock()
    prior_trace: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "hi"},
    ]

    with pytest.raises(ValueError, match=_MULTITURN_STRUCTURED_ERROR):
        await adapter.invoke({"x": 1}, prior_trace=prior_trace)

    adapter._run.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_trace", [None, []])
async def test_invoke_validates_input_schema_when_single_turn(
    tmp_path, prior_trace: list[ChatCompletionMessageParam] | None
):
    project = Project(name="proj", path=tmp_path / "proj.kiln")
    project.save_to_file()
    task = Task(
        name="t",
        instruction="i",
        parent=project,
    )
    task.save_to_file()
    adapter = MockAdapter(
        task=task,
        run_config=KilnAgentRunConfigProperties(
            model_name="gpt_4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
    )
    adapter.input_schema = _INPUT_OBJECT_SCHEMA
    adapter._run = AsyncMock()

    with pytest.raises(ValueError, match="input schema"):
        await adapter.invoke({}, prior_trace=prior_trace)

    adapter._run.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_trace", [None, []])
async def test_invoke_empty_prior_trace_like_none_allows_structured_input(
    tmp_path, prior_trace: list[ChatCompletionMessageParam] | None
):
    project = Project(name="proj", path=tmp_path / "proj.kiln")
    project.save_to_file()
    task = Task(
        name="t",
        instruction="i",
        parent=project,
    )
    task.save_to_file()
    adapter = MockAdapter(
        task=task,
        run_config=KilnAgentRunConfigProperties(
            model_name="gpt_4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
    )
    adapter.input_schema = _INPUT_OBJECT_SCHEMA
    adapter._run = AsyncMock(
        return_value=(
            RunOutput(output="ok", intermediate_outputs=None, trace=None),
            None,
        )
    )

    with (
        patch(
            "kiln_ai.adapters.model_adapters.base_adapter.model_parser_from_id",
            return_value=MagicMock(
                parse_output=MagicMock(
                    return_value=RunOutput(
                        output="ok", intermediate_outputs=None, trace=None
                    )
                )
            ),
        ),
        patch(
            "kiln_ai.adapters.model_adapters.base_adapter.request_formatter_from_id",
        ),
        patch.object(
            adapter,
            "model_provider",
            return_value=MagicMock(
                parser="default",
                formatter=None,
                reasoning_capable=False,
            ),
        ),
    ):
        await adapter.invoke({"x": 1}, prior_trace=prior_trace)

    adapter._run.assert_called_once()
    assert adapter._run.call_args[1].get("prior_trace") is None


def test_prepare_stream_rejects_multiturn_with_structured_input(tmp_path):
    project = Project(name="proj", path=tmp_path / "proj.kiln")
    project.save_to_file()
    task = Task(
        name="t",
        instruction="i",
        parent=project,
    )
    task.save_to_file()
    adapter = MockAdapter(
        task=task,
        run_config=KilnAgentRunConfigProperties(
            model_name="gpt_4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
    )
    adapter.input_schema = _INPUT_OBJECT_SCHEMA
    prior_trace: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "hi"},
    ]

    with (
        patch.object(
            adapter,
            "model_provider",
            return_value=MagicMock(formatter=None),
        ),
        pytest.raises(ValueError, match=_MULTITURN_STRUCTURED_ERROR),
    ):
        adapter._prepare_stream({"x": 1}, prior_trace=prior_trace)


@pytest.mark.parametrize("prior_trace", [None, []])
def test_prepare_stream_validates_input_schema_when_single_turn(
    tmp_path, prior_trace: list[ChatCompletionMessageParam] | None
):
    project = Project(name="proj", path=tmp_path / "proj.kiln")
    project.save_to_file()
    task = Task(
        name="t",
        instruction="i",
        parent=project,
    )
    task.save_to_file()
    adapter = MockAdapter(
        task=task,
        run_config=KilnAgentRunConfigProperties(
            model_name="gpt_4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
    )
    adapter.input_schema = _INPUT_OBJECT_SCHEMA
    invalid_input: dict = {}

    with (
        patch.object(
            adapter,
            "model_provider",
            return_value=MagicMock(formatter=None),
        ),
        pytest.raises(ValueError, match="input schema"),
    ):
        adapter._prepare_stream(invalid_input, prior_trace=prior_trace)


def test_build_chat_formatter_rejects_multiturn_with_structured_input(tmp_path):
    project = Project(name="proj", path=tmp_path / "proj.kiln")
    project.save_to_file()
    task = Task(
        name="t",
        instruction="i",
        parent=project,
    )
    task.save_to_file()
    adapter = MockAdapter(
        task=task,
        run_config=KilnAgentRunConfigProperties(
            model_name="gpt_4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
    )
    adapter.input_schema = _INPUT_OBJECT_SCHEMA
    prior_trace: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "hi"},
    ]

    with pytest.raises(ValueError, match=_MULTITURN_STRUCTURED_ERROR):
        adapter.build_chat_formatter("new input", prior_trace=prior_trace)


@pytest.mark.parametrize(
    "initial_mode,expected_mode",
    [
        (
            StructuredOutputMode.json_schema,
            StructuredOutputMode.json_schema,
        ),  # Should not change
        (
            StructuredOutputMode.unknown,
            StructuredOutputMode.json_mode,
        ),  # Should update to default
    ],
)
async def test_update_run_config_unknown_structured_output_mode(
    base_project, initial_mode, expected_mode
):
    """Test that unknown structured output mode is updated to the default for the model provider"""
    # Create a task with a parent project
    task = Task(name="test_task", instruction="test_instruction", parent=base_project)

    # Create a run config with the initial mode
    run_config = KilnAgentRunConfigProperties(
        model_name="test_model",
        model_provider_name="openai",
        prompt_id="simple_prompt_builder",
        structured_output_mode=initial_mode,
        temperature=0.7,  # Add some other properties to verify they're preserved
        top_p=0.9,
    )

    # Mock the default mode lookup
    with patch(
        "kiln_ai.adapters.model_adapters.base_adapter.default_structured_output_mode_for_model_provider"
    ) as mock_default:
        mock_default.return_value = StructuredOutputMode.json_mode

        # Create the adapter
        adapter = MockAdapter(task=task, run_config=run_config)

        # Verify the mode was updated correctly
        assert adapter.run_config.structured_output_mode == expected_mode

        # Verify other properties were preserved
        assert adapter.run_config.temperature == 0.7
        assert adapter.run_config.top_p == 0.9

        # Verify the default mode lookup was only called when needed
        if initial_mode == StructuredOutputMode.unknown:
            mock_default.assert_called_once_with("test_model", "openai")
        else:
            mock_default.assert_not_called()


@pytest.mark.parametrize(
    "tools_config,expected_tool_count,expected_tool_ids",
    [
        # No tools config
        (None, 0, []),
        # Empty tools config with None tools
        (ToolsRunConfig(tools=[]), 0, []),
        # Single tool
        ([KilnBuiltInToolId.ADD_NUMBERS], 1, [KilnBuiltInToolId.ADD_NUMBERS]),
        # Multiple tools
        (
            [KilnBuiltInToolId.ADD_NUMBERS, KilnBuiltInToolId.SUBTRACT_NUMBERS],
            2,
            [KilnBuiltInToolId.ADD_NUMBERS, KilnBuiltInToolId.SUBTRACT_NUMBERS],
        ),
        # All available built-in tools
        (
            [
                KilnBuiltInToolId.ADD_NUMBERS,
                KilnBuiltInToolId.SUBTRACT_NUMBERS,
                KilnBuiltInToolId.MULTIPLY_NUMBERS,
                KilnBuiltInToolId.DIVIDE_NUMBERS,
            ],
            4,
            [
                KilnBuiltInToolId.ADD_NUMBERS,
                KilnBuiltInToolId.SUBTRACT_NUMBERS,
                KilnBuiltInToolId.MULTIPLY_NUMBERS,
                KilnBuiltInToolId.DIVIDE_NUMBERS,
            ],
        ),
    ],
)
async def test_available_tools(
    base_project, tools_config, expected_tool_count, expected_tool_ids
):
    """Test that available_tools returns correct tools based on tools_config"""
    # Create a task with a parent project
    task = Task(name="test_task", instruction="test_instruction", parent=base_project)

    # Create tools config if we have tool IDs
    if tools_config is None:
        final_tools_config = None
    elif isinstance(tools_config, list):
        final_tools_config = ToolsRunConfig(tools=tools_config)
    else:
        final_tools_config = tools_config

    # Create adapter with tools config
    adapter = MockAdapter(
        task=task,
        run_config=KilnAgentRunConfigProperties(
            model_name="test_model",
            model_provider_name="openai",
            prompt_id="simple_prompt_builder",
            structured_output_mode="json_schema",
            tools_config=final_tools_config,
        ),
    )

    # Get available tools
    tools = await adapter.available_tools()

    # Verify tool count
    assert len(tools) == expected_tool_count

    # Verify all tools implement KilnToolInterface
    for tool in tools:
        assert isinstance(tool, KilnToolInterface)

    # Verify tool IDs match expected
    if expected_tool_ids:
        actual_tool_ids = [await tool.id() for tool in tools]
        assert actual_tool_ids == expected_tool_ids


async def test_available_tools_with_invalid_tool_id(base_project):
    """Test that available_tools raises ValueError for invalid tool ID"""
    # Create a task with a parent project
    task = Task(name="test_task", instruction="test_instruction", parent=base_project)

    # Create tools config with valid tool ID
    tools_config = ToolsRunConfig(tools=[KilnBuiltInToolId.ADD_NUMBERS])

    # Create adapter
    adapter = MockAdapter(
        task=task,
        run_config=KilnAgentRunConfigProperties(
            model_name="test_model",
            model_provider_name="openai",
            prompt_id="simple_prompt_builder",
            structured_output_mode="json_schema",
            tools_config=tools_config,
        ),
    )

    # Mock tool_from_id to raise ValueError for any tool ID
    with patch(
        "kiln_ai.adapters.model_adapters.base_adapter.tool_from_id"
    ) as mock_tool_from_id:
        mock_tool_from_id.side_effect = ValueError(
            "Tool ID test_id not found in tool registry"
        )

        # Should raise ValueError when trying to get tools
        with pytest.raises(
            ValueError, match="Tool ID test_id not found in tool registry"
        ):
            await adapter.available_tools()


async def test_available_tools_duplicate_names_raises_error(base_project):
    """Test that available_tools raises ValueError when tools have duplicate names"""
    # Create a task with a parent project
    task = Task(name="test_task", instruction="test_instruction", parent=base_project)

    # Create tools config with two different tool IDs
    tools_config = ToolsRunConfig(
        tools=[KilnBuiltInToolId.ADD_NUMBERS, KilnBuiltInToolId.SUBTRACT_NUMBERS]
    )

    # Create adapter
    adapter = MockAdapter(
        task=task,
        run_config=KilnAgentRunConfigProperties(
            model_name="test_model",
            model_provider_name="openai",
            prompt_id="simple_prompt_builder",
            structured_output_mode="json_schema",
            tools_config=tools_config,
        ),
    )

    # Create mock tools with duplicate names
    async def mock_name1():
        return "duplicate_name"

    async def mock_name2():
        return "duplicate_name"

    mock_tool1 = MagicMock(spec=KilnToolInterface)
    mock_tool1.name = mock_name1
    mock_tool2 = MagicMock(spec=KilnToolInterface)
    mock_tool2.name = mock_name2  # Same name as tool1

    # Mock tool_from_id to return our mock tools with duplicate names
    with patch(
        "kiln_ai.adapters.model_adapters.base_adapter.tool_from_id"
    ) as mock_tool_from_id:
        mock_tool_from_id.side_effect = [mock_tool1, mock_tool2]

        # Should raise ValueError when tools have duplicate names
        with pytest.raises(
            ValueError, match="share the same function name: duplicate_name"
        ):
            await adapter.available_tools()


async def test_assemble_unique_agent_tools_lists_colliding_names(base_project):
    task = Task(name="test_task", instruction="test_instruction", parent=base_project)

    def mock_tool(name: str) -> KilnToolInterface:
        tool = MagicMock(spec=KilnToolInterface)
        tool.name = AsyncMock(return_value=name)
        return tool

    with patch(
        "kiln_ai.adapters.model_adapters.base_adapter.tool_from_id"
    ) as mock_tool_from_id:
        mock_tool_from_id.side_effect = [
            mock_tool("dup_a"),
            mock_tool("dup_a"),
            mock_tool("dup_b"),
            mock_tool("dup_b"),
        ]
        with pytest.raises(
            ValueError, match="share the same function name: dup_a, dup_b"
        ):
            await assemble_unique_agent_tools(
                task, ["id_1", "id_2", "id_3", "id_4"], []
            )


async def test_assemble_unique_agent_tools_reserved_skill_name(base_project):
    task = Task(name="test_task", instruction="test_instruction", parent=base_project)
    skill = Skill(name="my-skill", description="d", parent=base_project)

    tool_named_skill = MagicMock(spec=KilnToolInterface)
    tool_named_skill.name = AsyncMock(return_value="skill")

    with patch(
        "kiln_ai.adapters.model_adapters.base_adapter.tool_from_id",
        return_value=tool_named_skill,
    ):
        with pytest.raises(ValueError) as exc_info:
            await assemble_unique_agent_tools(task, ["id_1"], [skill])
    assert "share the same function name: skill" in str(exc_info.value)
    assert "reserved" in str(exc_info.value)


async def test_custom_prompt_builder(base_task):
    """Test that custom prompt builder can be injected via AdapterConfig"""

    # Create a custom prompt builder
    class CustomPromptBuilder(BasePromptBuilder):
        def build_base_prompt(self) -> str:
            return "This is a custom prompt from injected builder"

    custom_builder = CustomPromptBuilder(base_task)

    adapter = MockAdapter(
        task=base_task,
        run_config=KilnAgentRunConfigProperties(
            model_name="test_model",
            model_provider_name="openai",
            prompt_id="simple_prompt_builder",
            structured_output_mode="json_schema",
        ),
        config=AdapterConfig(prompt_builder=custom_builder),
    )

    # Mock model provider
    provider = MagicMock()
    provider.reasoning_capable = False
    provider.tuned_chat_strategy = None
    adapter.model_provider = MagicMock(return_value=provider)

    # Test that the custom prompt builder is used
    formatter = adapter.build_chat_formatter(input="test input")
    assert formatter.system_message == "This is a custom prompt from injected builder"
    assert adapter.prompt_builder == custom_builder


class TestAgentRunContextLifecycle:
    """Unit tests for agent run context lifecycle in BaseAdapter."""

    @pytest.fixture
    def clear_context(self):
        """Clear the agent run context before each test."""
        from kiln_ai.run_context import clear_agent_run_id

        clear_agent_run_id()
        yield
        clear_agent_run_id()

    @pytest.mark.asyncio
    async def test_invoke_sets_run_context(self, adapter, clear_context):
        """Test that invoke sets the run context for root agent."""
        from kiln_ai.adapters.run_output import RunOutput
        from kiln_ai.run_context import get_agent_run_id

        # Mock the _run method
        async def mock_run(input, messages=None, **kwargs):
            # Check that run ID is set during _run
            run_id = get_agent_run_id()
            assert run_id is not None
            assert run_id.startswith("run_")
            return RunOutput(output="test output", intermediate_outputs={}), None

        adapter._run = mock_run

        # Mock the model provider and parser
        provider = MagicMock()
        provider.parser = "test_parser"
        provider.formatter = None
        provider.reasoning_capable = False
        adapter.model_provider = MagicMock(return_value=provider)

        parser = MagicMock()
        parser.parse_output.return_value = RunOutput(
            output="test output", intermediate_outputs={}
        )

        with (
            patch(
                "kiln_ai.adapters.model_adapters.base_adapter.model_parser_from_id"
            ) as mock_parser_factory,
            patch(
                "kiln_ai.adapters.model_adapters.base_adapter.request_formatter_from_id"
            ),
        ):
            mock_parser_factory.return_value = parser

            await adapter.invoke_returning_run_output({"test": "input"})

    @pytest.mark.asyncio
    async def test_invoke_clears_run_context_after(self, adapter, clear_context):
        """Test that invoke clears the run context after completion."""
        from kiln_ai.adapters.run_output import RunOutput
        from kiln_ai.run_context import get_agent_run_id

        # Mock the _run method
        async def mock_run(input, messages=None, **kwargs):
            return RunOutput(output="test output", intermediate_outputs={}), None

        adapter._run = mock_run

        # Mock the model provider and parser
        provider = MagicMock()
        provider.parser = "test_parser"
        provider.formatter = None
        provider.reasoning_capable = False
        adapter.model_provider = MagicMock(return_value=provider)

        parser = MagicMock()
        parser.parse_output.return_value = RunOutput(
            output="test output", intermediate_outputs={}
        )

        with (
            patch(
                "kiln_ai.adapters.model_adapters.base_adapter.model_parser_from_id"
            ) as mock_parser_factory,
            patch(
                "kiln_ai.adapters.model_adapters.base_adapter.request_formatter_from_id"
            ),
        ):
            mock_parser_factory.return_value = parser

            await adapter.invoke_returning_run_output({"test": "input"})

            # After invoke, run ID should be cleared
            assert get_agent_run_id() is None

    @pytest.mark.asyncio
    async def test_invoke_clears_run_context_on_error(self, adapter, clear_context):
        """Test that invoke clears the run context even on error."""
        from kiln_ai.run_context import get_agent_run_id

        # Mock the _run method to raise an error
        async def mock_run(input, messages=None, **kwargs):
            # Run ID should be set even when error occurs
            run_id = get_agent_run_id()
            assert run_id is not None
            raise ValueError("Test error")

        adapter._run = mock_run

        provider = MagicMock()
        provider.parser = "test_parser"
        provider.formatter = None
        provider.reasoning_capable = False
        adapter.model_provider = MagicMock(return_value=provider)

        with (
            patch("kiln_ai.adapters.model_adapters.base_adapter.model_parser_from_id"),
            patch(
                "kiln_ai.adapters.model_adapters.base_adapter.request_formatter_from_id"
            ),
        ):
            # Runtime failures from `_run` are wrapped in KilnRunError; the
            # underlying ValueError is available via `.original`.
            from kiln_ai.adapters.errors import KilnRunError

            with pytest.raises(KilnRunError) as ei:
                await adapter.invoke_returning_run_output({"test": "input"})
            assert isinstance(ei.value.original, ValueError)
            assert str(ei.value.original) == "Test error"

            # After error, run ID should be cleared
            assert get_agent_run_id() is None

    @pytest.mark.asyncio
    async def test_sub_agent_inherits_run(self, adapter, clear_context):
        """Test that sub-agent inherits parent's run ID."""
        from kiln_ai.adapters.run_output import RunOutput
        from kiln_ai.run_context import get_agent_run_id, set_agent_run_id

        # Simulate parent agent setting the run context
        parent_run_id = "parent_agent_run"
        set_agent_run_id(parent_run_id)

        # Mock the _run method to check inherited run ID
        async def mock_run(input, messages=None, **kwargs):
            # Sub-agent should see parent's run ID
            run_id = get_agent_run_id()
            assert run_id == parent_run_id
            return RunOutput(output="test output", intermediate_outputs={}), None

        adapter._run = mock_run

        # Mock the model provider and parser
        provider = MagicMock()
        provider.parser = "test_parser"
        provider.formatter = None
        provider.reasoning_capable = False
        adapter.model_provider = MagicMock(return_value=provider)

        parser = MagicMock()
        parser.parse_output.return_value = RunOutput(
            output="test output", intermediate_outputs={}
        )

        with (
            patch(
                "kiln_ai.adapters.model_adapters.base_adapter.model_parser_from_id"
            ) as mock_parser_factory,
            patch(
                "kiln_ai.adapters.model_adapters.base_adapter.request_formatter_from_id"
            ),
        ):
            mock_parser_factory.return_value = parser

            await adapter.invoke_returning_run_output({"test": "input"})

            # After invoke, the parent's run ID should still be set
            # (since we were acting as a sub-agent)
            assert get_agent_run_id() == parent_run_id

    @pytest.mark.asyncio
    async def test_sub_agent_does_not_create_new_run(self, adapter, clear_context):
        """Test that sub-agent doesn't create a new run ID."""
        from kiln_ai.adapters.run_output import RunOutput
        from kiln_ai.run_context import get_agent_run_id, set_agent_run_id

        # Simulate parent agent setting the run context
        parent_run_id = "parent_agent_run"
        set_agent_run_id(parent_run_id)

        run_id_during_run = None

        # Mock the _run method to capture run ID
        async def mock_run(input, messages=None, **kwargs):
            nonlocal run_id_during_run
            run_id_during_run = get_agent_run_id()
            return RunOutput(output="test output", intermediate_outputs={}), None

        adapter._run = mock_run

        # Mock the model provider and parser
        provider = MagicMock()
        provider.parser = "test_parser"
        provider.formatter = None
        provider.reasoning_capable = False
        adapter.model_provider = MagicMock(return_value=provider)

        parser = MagicMock()
        parser.parse_output.return_value = RunOutput(
            output="test output", intermediate_outputs={}
        )

        with (
            patch(
                "kiln_ai.adapters.model_adapters.base_adapter.model_parser_from_id"
            ) as mock_parser_factory,
            patch(
                "kiln_ai.adapters.model_adapters.base_adapter.request_formatter_from_id"
            ),
        ):
            mock_parser_factory.return_value = parser

            await adapter.invoke_returning_run_output({"test": "input"})

            # Sub-agent should have used the parent's run ID
            assert run_id_during_run == parent_run_id

    @pytest.mark.asyncio
    async def test_cleanup_session_called_on_completion(self, adapter, clear_context):
        """Test that cleanup_session is called when root agent completes."""
        from kiln_ai.adapters.run_output import RunOutput

        # Mock the _run method
        async def mock_run(input, messages=None, **kwargs):
            return RunOutput(output="test output", intermediate_outputs={}), None

        adapter._run = mock_run

        # Mock the model provider and parser
        provider = MagicMock()
        provider.parser = "test_parser"
        provider.formatter = None
        provider.reasoning_capable = False
        adapter.model_provider = MagicMock(return_value=provider)

        parser = MagicMock()
        parser.parse_output.return_value = RunOutput(
            output="test output", intermediate_outputs={}
        )

        with (
            patch(
                "kiln_ai.adapters.model_adapters.base_adapter.model_parser_from_id"
            ) as mock_parser_factory,
            patch(
                "kiln_ai.adapters.model_adapters.base_adapter.request_formatter_from_id"
            ),
            patch(
                "kiln_ai.tools.mcp_session_manager.MCPSessionManager"
            ) as mock_manager_class,
        ):
            mock_parser_factory.return_value = parser

            mock_manager = MagicMock()
            mock_manager_class.shared.return_value = mock_manager
            mock_manager.cleanup_session = AsyncMock()

            await adapter.invoke_returning_run_output({"test": "input"})

            # cleanup_session should have been called
            mock_manager.cleanup_session.assert_called_once()
            # The run ID should be a string that starts with "run_"
            run_id = mock_manager.cleanup_session.call_args[0][0]
            assert run_id.startswith("run_")

    @staticmethod
    def _fake_adapter_stream():
        """A one-chunk stand-in for the model stream, so these tests exercise
        the session scope around iteration rather than a real model path."""

        class FakeAdapterStream:
            async def __aiter__(self):
                yield ModelResponseStream(
                    id="test",
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(content="hi"),
                            finish_reason=None,
                        )
                    ],
                )

        return FakeAdapterStream()

    @pytest.mark.asyncio
    async def test_cleanup_session_called_after_openai_stream(
        self, adapter, clear_context
    ):
        """The streaming path opens the same session scope as invoke, so a
        consumed stream must close its sessions and release the run id too."""
        from kiln_ai.run_context import get_agent_run_id

        with (
            patch.object(
                adapter, "_prepare_stream", return_value=self._fake_adapter_stream()
            ),
            patch.object(
                adapter, "_finalize_stream", return_value=MagicMock(spec=TaskRun)
            ),
            patch(
                "kiln_ai.tools.mcp_session_manager.MCPSessionManager"
            ) as mock_manager_class,
        ):
            mock_manager = MagicMock()
            mock_manager_class.shared.return_value = mock_manager
            mock_manager.cleanup_session = AsyncMock()

            async for _chunk in adapter.invoke_openai_stream("test input"):
                pass

            mock_manager.cleanup_session.assert_called_once()
            run_id = mock_manager.cleanup_session.call_args[0][0]
            assert run_id.startswith("run_")
            assert get_agent_run_id() is None

    @pytest.mark.asyncio
    async def test_cleanup_session_called_after_ai_sdk_stream(
        self, adapter, clear_context
    ):
        """Same guarantee for the AI SDK stream: it owns a scope of its own."""
        from kiln_ai.run_context import get_agent_run_id

        finalized_run = MagicMock(spec=TaskRun)
        # Pins the finish branch: a bare mock is truthy here, which would
        # silently route the stream down the tool-calls-pending path instead.
        finalized_run.is_toolcall_pending = False

        with (
            patch.object(
                adapter, "_prepare_stream", return_value=self._fake_adapter_stream()
            ),
            patch.object(adapter, "_finalize_stream", return_value=finalized_run),
            patch(
                "kiln_ai.tools.mcp_session_manager.MCPSessionManager"
            ) as mock_manager_class,
        ):
            mock_manager = MagicMock()
            mock_manager_class.shared.return_value = mock_manager
            mock_manager.cleanup_session = AsyncMock()

            async for _event in adapter.invoke_ai_sdk_stream("test input"):
                pass

            mock_manager.cleanup_session.assert_called_once()
            run_id = mock_manager.cleanup_session.call_args[0][0]
            assert run_id.startswith("run_")
            assert get_agent_run_id() is None


class TestStreamMethods:
    """Tests for the streaming methods on BaseAdapter."""

    @pytest.fixture
    def stream_adapter(self, base_task):
        return MockAdapter(
            task=base_task,
            run_config=KilnAgentRunConfigProperties(
                model_name="test_model",
                model_provider_name="openai",
                prompt_id="simple_prompt_builder",
                structured_output_mode="json_schema",
            ),
        )

    @pytest.mark.asyncio
    async def test_invoke_openai_stream_raises_for_unsupported_adapter(
        self, stream_adapter
    ):
        """MockAdapter does not implement _create_run_stream."""
        provider = MagicMock()
        provider.formatter = None
        stream_adapter.model_provider = MagicMock(return_value=provider)

        with pytest.raises(NotImplementedError, match="Streaming is not supported"):
            async for _chunk in stream_adapter.invoke_openai_stream("test input"):
                pass

    @pytest.mark.asyncio
    async def test_invoke_ai_sdk_stream_raises_for_unsupported_adapter(
        self, stream_adapter
    ):
        """MockAdapter does not implement _create_run_stream."""
        provider = MagicMock()
        provider.formatter = None
        stream_adapter.model_provider = MagicMock(return_value=provider)

        with pytest.raises(NotImplementedError, match="Streaming is not supported"):
            async for _event in stream_adapter.invoke_ai_sdk_stream("test input"):
                pass

    @pytest.mark.asyncio
    async def test_invoke_ai_sdk_stream_resets_converter_between_tool_rounds(
        self, stream_adapter
    ):
        """tool-input-start must be emitted for a new tool call at index 0 after a tool round."""

        def _make_tool_chunk(call_id: str, name: str) -> ModelResponseStream:
            func = Function(name=name, arguments='{"x":1}')
            tc = ChatCompletionDeltaToolCall(index=0, function=func)
            tc.id = call_id
            delta = Delta(tool_calls=[tc])
            choice = StreamingChoices(index=0, delta=delta, finish_reason=None)
            return ModelResponseStream(id="test", choices=[choice])

        round1_chunk = _make_tool_chunk("call_r1", "tool_a")
        round2_chunk = _make_tool_chunk("call_r2", "tool_b")

        fake_events = [
            round1_chunk,
            ToolCallEvent(
                event_type=ToolCallEventType.INPUT_AVAILABLE,
                tool_call_id="call_r1",
                tool_name="tool_a",
                arguments={"x": 1},
            ),
            ToolCallEvent(
                event_type=ToolCallEventType.OUTPUT_AVAILABLE,
                tool_call_id="call_r1",
                tool_name="tool_a",
                result="done",
            ),
            round2_chunk,
        ]

        class FakeAdapterStream:
            result = MagicMock()

            async def __aiter__(self):
                for event in fake_events:
                    yield event

        with (
            patch.object(
                stream_adapter,
                "_prepare_stream",
                return_value=FakeAdapterStream(),
            ),
            patch.object(stream_adapter, "_finalize_stream"),
        ):
            events = []
            async for event in stream_adapter.invoke_ai_sdk_stream("test input"):
                events.append(event)

        tool_input_starts = [
            e for e in events if e.type == AiSdkEventType.TOOL_INPUT_START
        ]
        assert len(tool_input_starts) == 2, (
            "tool-input-start must fire once per tool-call round"
        )
        assert tool_input_starts[0].toolCallId == "call_r1"
        assert tool_input_starts[1].toolCallId == "call_r2"

    @pytest.mark.asyncio
    async def test_openai_stream_exposes_task_run_after_iteration(self, stream_adapter):
        fake_chunk = ModelResponseStream(
            id="test",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="hi"),
                    finish_reason=None,
                )
            ],
        )

        class FakeAdapterStream:
            result = MagicMock()

            async def __aiter__(self):
                yield fake_chunk

        expected_run = MagicMock(spec=TaskRun)

        with (
            patch.object(
                stream_adapter,
                "_prepare_stream",
                return_value=FakeAdapterStream(),
            ),
            patch.object(stream_adapter, "_finalize_stream", return_value=expected_run),
        ):
            stream = stream_adapter.invoke_openai_stream("test input")

            with pytest.raises(RuntimeError, match="not been fully consumed"):
                _ = stream.task_run

            async for _chunk in stream:
                pass

            assert stream.task_run is expected_run

    @pytest.mark.asyncio
    async def test_ai_sdk_stream_exposes_task_run_after_iteration(self, stream_adapter):
        fake_chunk = ModelResponseStream(
            id="test",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="hi"),
                    finish_reason=None,
                )
            ],
        )

        class FakeAdapterStream:
            result = MagicMock()

            async def __aiter__(self):
                yield fake_chunk

        expected_run = MagicMock(spec=TaskRun)

        with (
            patch.object(
                stream_adapter,
                "_prepare_stream",
                return_value=FakeAdapterStream(),
            ),
            patch.object(stream_adapter, "_finalize_stream", return_value=expected_run),
        ):
            stream = stream_adapter.invoke_ai_sdk_stream("test input")

            with pytest.raises(RuntimeError, match="not been fully consumed"):
                _ = stream.task_run

            async for _event in stream:
                pass

            assert stream.task_run is expected_run


class TestFinalizeStream:
    """Tests for _finalize_stream post-processing after streaming."""

    @pytest.fixture
    def finalize_adapter(self, base_task):
        return MockAdapter(
            task=base_task,
            run_config=KilnAgentRunConfigProperties(
                model_name="test_model",
                model_provider_name="openai",
                prompt_id="simple_prompt_builder",
                structured_output_mode="json_schema",
            ),
        )

    def _make_adapter_stream(self, output, usage=None, trace=None):
        from kiln_ai.adapters.model_adapters.adapter_stream import AdapterStreamResult

        stream = MagicMock()
        stream.result = AdapterStreamResult(
            run_output=RunOutput(
                output=output,
                intermediate_outputs={},
                trace=trace,
            ),
            usage=usage or Usage(),
        )
        return stream

    def test_finalize_stream_plain_text(self, finalize_adapter):
        provider = MagicMock()
        provider.parser = None
        provider.reasoning_capable = False
        finalize_adapter.model_provider = MagicMock(return_value=provider)

        adapter_stream = self._make_adapter_stream("Hello world")
        run = finalize_adapter._finalize_stream(adapter_stream, "test input", None)

        assert isinstance(run, TaskRun)
        assert run.output.output == "Hello world"
        assert run.id is None

    def _make_structured_adapter(self, base_task, schema):
        base_task.output_json_schema = schema
        adapter = MockAdapter(
            task=base_task,
            run_config=KilnAgentRunConfigProperties(
                model_name="test_model",
                model_provider_name="openai",
                prompt_id="simple_prompt_builder",
                structured_output_mode="json_schema",
            ),
        )
        return adapter

    def test_finalize_stream_structured_output(self, base_task):
        schema = '{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}'
        adapter = self._make_structured_adapter(base_task, schema)

        provider = MagicMock()
        provider.parser = None
        provider.reasoning_capable = False
        adapter.model_provider = MagicMock(return_value=provider)

        adapter_stream = self._make_adapter_stream({"name": "test"})
        run = adapter._finalize_stream(adapter_stream, "test input", None)

        assert isinstance(run, TaskRun)
        assert '"name"' in run.output.output

    def test_finalize_stream_structured_output_from_json_string(self, base_task):
        schema = '{"type": "object", "properties": {"val": {"type": "integer"}}, "required": ["val"]}'
        adapter = self._make_structured_adapter(base_task, schema)

        provider = MagicMock()
        provider.parser = None
        provider.reasoning_capable = False
        adapter.model_provider = MagicMock(return_value=provider)

        adapter_stream = self._make_adapter_stream('{"val": 42}')
        run = adapter._finalize_stream(adapter_stream, "test input", None)
        assert isinstance(run, TaskRun)

    def test_finalize_stream_structured_output_unparseable_json_is_retryable(
        self, base_task
    ):
        # A streamed response that isn't JSON is the same one-off model slip as
        # the non-streaming path, so it must carry the retryable type too.
        schema = '{"type": "object", "properties": {"val": {"type": "integer"}}, "required": ["val"]}'
        adapter = self._make_structured_adapter(base_task, schema)

        provider = MagicMock()
        provider.parser = None
        provider.reasoning_capable = False
        adapter.model_provider = MagicMock(return_value=provider)

        adapter_stream = self._make_adapter_stream("Sure! Here you go.")
        with pytest.raises(StructuredOutputParseError) as exc_info:
            adapter._finalize_stream(adapter_stream, "test input", None)
        assert is_retryable_error(exc_info.value) is True

    @pytest.mark.parametrize(
        "stream_output,expected_message",
        [
            # Already a non-dict value, so no JSON parsing is involved...
            (42, "structured response is not a dict: 42"),
            # ...and valid JSON that parses to a non-dict (the object wrapped in
            # a list, a common model slip). Both are wrong-shape output.
            ('[{"x": "y"}]', "structured response is not a dict: [{'x': 'y'}]"),
        ],
    )
    def test_finalize_stream_structured_output_not_dict_raises(
        self, base_task, stream_output, expected_message
    ):
        schema = '{"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}'
        adapter = self._make_structured_adapter(base_task, schema)

        provider = MagicMock()
        provider.parser = None
        provider.reasoning_capable = False
        adapter.model_provider = MagicMock(return_value=provider)

        adapter_stream = self._make_adapter_stream(stream_output)
        with pytest.raises(StructuredOutputParseError) as exc_info:
            adapter._finalize_stream(adapter_stream, "test input", None)
        # Retryable like a parse failure, and the message the user sees is
        # unchanged from when this raised RuntimeError.
        assert is_retryable_error(exc_info.value) is True
        assert str(exc_info.value) == expected_message

    def test_finalize_stream_non_structured_non_string_raises(self, finalize_adapter):
        provider = MagicMock()
        provider.parser = None
        provider.reasoning_capable = False
        finalize_adapter.model_provider = MagicMock(return_value=provider)

        adapter_stream = self._make_adapter_stream({"unexpected": "dict"})
        with pytest.raises(RuntimeError, match="not a string for non-structured"):
            finalize_adapter._finalize_stream(adapter_stream, "test input", None)

    def test_finalize_stream_reasoning_required_but_missing(self, finalize_adapter):
        provider = MagicMock()
        provider.parser = None
        provider.reasoning_capable = True
        provider.reasoning_optional_for_structured_output = False
        finalize_adapter.model_provider = MagicMock(return_value=provider)

        adapter_stream = self._make_adapter_stream("output")
        with pytest.raises(RuntimeError, match="Reasoning is required"):
            finalize_adapter._finalize_stream(adapter_stream, "test input", None)

    def test_finalize_stream_reasoning_not_required_with_tool_calls(
        self, finalize_adapter
    ):
        provider = MagicMock()
        provider.parser = None
        provider.reasoning_capable = True
        provider.reasoning_optional_for_structured_output = False
        finalize_adapter.model_provider = MagicMock(return_value=provider)

        trace = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "result", "tool_call_id": "call_1"},
        ]
        adapter_stream = self._make_adapter_stream("output", trace=trace)
        run = finalize_adapter._finalize_stream(adapter_stream, "test input", None)
        assert isinstance(run, TaskRun)

    def test_finalize_stream_saves_when_allowed(self, tmp_path):
        project_path = tmp_path / "proj" / "project.kiln"
        project_path.parent.mkdir()
        project = Project(name="test", path=project_path)
        project.save_to_file()
        task = Task(name="t", instruction="i", parent=project)
        task.save_to_file()

        adapter = MockAdapter(
            task=task,
            run_config=KilnAgentRunConfigProperties(
                model_name="test_model",
                model_provider_name="openai",
                prompt_id="simple_prompt_builder",
                structured_output_mode="json_schema",
            ),
            config=AdapterConfig(allow_saving=True),
        )

        provider = MagicMock()
        provider.parser = None
        provider.reasoning_capable = False
        adapter.model_provider = MagicMock(return_value=provider)

        adapter_stream = self._make_adapter_stream("result")
        with patch(
            "kiln_ai.adapters.model_adapters.base_adapter.Config"
        ) as mock_config:
            mock_config.shared.return_value.autosave_runs = True
            mock_config.shared.return_value.user_id = "test_user"
            run = adapter._finalize_stream(adapter_stream, "test input", None)
        assert run.id is not None


class TestResolveSkills:
    def test_returns_empty_for_non_kiln_agent(self, base_task):
        from kiln_ai.datamodel.run_config import (
            McpRunConfigProperties,
            MCPToolReference,
        )

        adapter = MockAdapter(
            task=base_task,
            run_config=McpRunConfigProperties(
                tool_reference=MCPToolReference(tool_id="mcp::local::s::t"),
            ),
        )
        assert adapter._resolve_skills() == []

    def test_returns_empty_when_no_tools_config(self, adapter):
        assert adapter._resolve_skills() == []

    @pytest.fixture
    def _run_config_with_tools(self):
        def _make(tools: list[str]) -> KilnAgentRunConfigProperties:
            return KilnAgentRunConfigProperties(
                model_name="test_model",
                model_provider_name="openai",
                prompt_id="simple_prompt_builder",
                structured_output_mode="json_schema",
                tools_config=ToolsRunConfig(tools=tools),
            )

        return _make

    def test_returns_empty_when_no_skill_ids(self, base_task, _run_config_with_tools):
        adapter = MockAdapter(
            task=base_task,
            run_config=_run_config_with_tools(["kiln_tool::add_numbers"]),
        )
        assert adapter._resolve_skills() == []

    def test_raises_when_skills_dict_not_provided(
        self, base_task, _run_config_with_tools
    ):
        adapter = MockAdapter(
            task=base_task,
            run_config=_run_config_with_tools(["kiln_tool::skill::skill_123"]),
        )
        with pytest.raises(ValueError, match="no skills dict was provided"):
            adapter._resolve_skills()

    def test_raises_when_skill_missing_from_dict(
        self, base_task, _run_config_with_tools
    ):
        adapter = MockAdapter(
            task=base_task,
            run_config=_run_config_with_tools(["kiln_tool::skill::skill_123"]),
            config=AdapterConfig(skills={}),
        )
        with pytest.raises(ValueError, match="not found in the injected skills dict"):
            adapter._resolve_skills()

    def test_resolves_skills_from_injected_dict(
        self, base_task, _run_config_with_tools
    ):
        skill = Skill(name="my-skill", description="A skill")
        adapter = MockAdapter(
            task=base_task,
            run_config=_run_config_with_tools([f"kiln_tool::skill::{skill.id}"]),
            config=AdapterConfig(skills={skill.id: skill}),
        )
        result = adapter._resolve_skills()
        assert len(result) == 1
        assert result[0].name == "my-skill"

    def test_deduplicates_skill_ids(self, base_task, _run_config_with_tools):
        skill = Skill(name="my-skill", description="A skill", body="do things")
        adapter = MockAdapter(
            task=base_task,
            run_config=_run_config_with_tools(
                [
                    f"kiln_tool::skill::{skill.id}",
                    f"kiln_tool::skill::{skill.id}",
                ]
            ),
            config=AdapterConfig(skills={skill.id: skill}),
        )
        result = adapter._resolve_skills()
        assert len(result) == 1
        assert result[0].name == "my-skill"

    def test_caches_result(self, base_task, _run_config_with_tools):
        skill = Skill(name="my-skill", description="A skill")
        adapter = MockAdapter(
            task=base_task,
            run_config=_run_config_with_tools([f"kiln_tool::skill::{skill.id}"]),
            config=AdapterConfig(skills={skill.id: skill}),
        )
        result1 = adapter._resolve_skills()
        result2 = adapter._resolve_skills()
        assert result1 is result2

    def test_build_prompt_includes_skills(self, base_task, _run_config_with_tools):
        skill = Skill(name="my-skill", description="A test skill")
        adapter = MockAdapter(
            task=base_task,
            run_config=_run_config_with_tools([f"kiln_tool::skill::{skill.id}"]),
            config=AdapterConfig(skills={skill.id: skill}),
        )
        prompt = adapter.build_prompt()
        assert "my-skill" in prompt
        assert "A test skill" in prompt

    def test_build_prompt_no_skills_section_without_skills(self, adapter):
        prompt = adapter.build_prompt()
        assert "## Skills" not in prompt


class TestInputTransformIntegration:
    """Adapter integration tests for input_transform per architecture section 7.4."""

    @pytest.fixture
    def object_schema(self):
        return json.dumps(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            }
        )

    def _make_adapter(self, base_task, input_transform=None, input_schema=None):
        run_config = KilnAgentRunConfigProperties(
            model_name="test_model",
            model_provider_name="openai",
            prompt_id="simple_prompt_builder",
            structured_output_mode="json_schema",
            input_transform=input_transform,
        )
        adapter = MockAdapter(task=base_task, run_config=run_config)
        if input_schema is not None:
            adapter.input_schema = input_schema
        return adapter

    async def _invoke_with_capture(self, adapter, task_input):
        captured_input = None

        async def mock_run(input, trace_ref, **kwargs):
            nonlocal captured_input
            captured_input = input
            return RunOutput(output="ok", intermediate_outputs={}, trace=None), None

        adapter._run = mock_run

        provider = MagicMock()
        provider.formatter = None
        provider.parser = "test_parser"
        provider.reasoning_capable = False
        adapter.model_provider = MagicMock(return_value=provider)

        with patch(
            "kiln_ai.adapters.model_adapters.base_adapter.model_parser_from_id",
            return_value=MagicMock(
                parse_output=MagicMock(
                    return_value=RunOutput(
                        output="ok", intermediate_outputs={}, trace=None
                    )
                )
            ),
        ):
            run, _ = await adapter.invoke_returning_run_output(task_input)

        return run, captured_input

    @pytest.mark.asyncio
    async def test_input_transform_object_schema(self, base_task, object_schema):
        """Object-schema task with JinjaInputTransform: rendered string passed to _run,
        raw dict preserved in TaskRun.input."""
        from kiln_ai.datamodel.input_transform import JinjaInputTransform

        transform = JinjaInputTransform(
            template="Hello {{ input.name }}, you are {{ input.age }}."
        )
        adapter = self._make_adapter(
            base_task, input_transform=transform, input_schema=object_schema
        )

        run, captured_input = await self._invoke_with_capture(
            adapter, {"name": "Alice", "age": 30}
        )

        assert captured_input == "Hello Alice, you are 30."
        assert run.input == json.dumps({"name": "Alice", "age": 30}, ensure_ascii=False)

    @pytest.mark.asyncio
    async def test_input_transform_plaintext_json(self, base_task):
        """Plaintext JSON input is parsed and templated correctly."""
        from kiln_ai.datamodel.input_transform import JinjaInputTransform

        transform = JinjaInputTransform(template="Key: {{ input.key }}")
        adapter = self._make_adapter(base_task, input_transform=transform)

        run, captured_input = await self._invoke_with_capture(
            adapter, '{"key": "value"}'
        )

        assert captured_input == "Key: value"
        assert run.input == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_input_transform_plaintext_non_json(self, base_task):
        """Non-JSON plaintext exposed via {{ input }}."""
        from kiln_ai.datamodel.input_transform import JinjaInputTransform

        transform = JinjaInputTransform(template="You said: {{ input }}")
        adapter = self._make_adapter(base_task, input_transform=transform)

        run, captured_input = await self._invoke_with_capture(adapter, "hello world")

        assert captured_input == "You said: hello world"
        assert run.input == "hello world"

    @pytest.mark.asyncio
    async def test_input_transform_array_schema(self, base_task):
        """List input exposed via {{ input[0] }}."""
        from kiln_ai.datamodel.input_transform import JinjaInputTransform

        array_schema = json.dumps({"type": "array", "items": {"type": "integer"}})
        transform = JinjaInputTransform(
            template="First: {{ input[0] }}, Count: {{ input | length }}"
        )
        adapter = self._make_adapter(
            base_task, input_transform=transform, input_schema=array_schema
        )

        run, captured_input = await self._invoke_with_capture(adapter, [10, 20, 30])

        assert captured_input == "First: 10, Count: 3"
        assert run.input == json.dumps([10, 20, 30])

    @pytest.mark.asyncio
    async def test_input_transform_none_identity(self, base_task):
        """RunConfig with input_transform=None: adapter behavior is identical to no-transform."""
        adapter = self._make_adapter(base_task, input_transform=None)

        run, captured_input = await self._invoke_with_capture(adapter, "raw input")

        assert captured_input == "raw input"
        assert run.input == "raw input"

    def test_input_transform_streaming_parity(self, base_task):
        """_prepare_stream applies the transform the same way as the sync path."""
        from kiln_ai.datamodel.input_transform import JinjaInputTransform

        transform = JinjaInputTransform(template="Rendered: {{ input.x }}")
        adapter = self._make_adapter(base_task, input_transform=transform)

        captured_stream_input = None

        def mock_create_run_stream(input, prior_trace=None):
            nonlocal captured_stream_input
            captured_stream_input = input
            return MagicMock()

        adapter._create_run_stream = mock_create_run_stream

        provider = MagicMock()
        provider.formatter = None
        adapter.model_provider = MagicMock(return_value=provider)

        adapter._prepare_stream({"x": 42}, prior_trace=None)

        assert captured_stream_input == "Rendered: 42"

    @pytest.mark.asyncio
    async def test_input_transform_undefined_error_pre_inference(self, base_task):
        """UndefinedError raised before inference — _run never called."""
        from jinja2.exceptions import UndefinedError

        from kiln_ai.datamodel.input_transform import JinjaInputTransform

        transform = JinjaInputTransform(template="{{ input.missing_key }}")
        adapter = self._make_adapter(base_task, input_transform=transform)

        with pytest.raises(ValueError, match="Input transform failed:") as exc_info:
            await self._invoke_with_capture(adapter, {"foo": "bar"})
        assert "missing_key" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, UndefinedError)

    def test_apply_input_transform_success(self, base_task):
        """Successful transform returns the rendered string unchanged."""
        from kiln_ai.datamodel.input_transform import JinjaInputTransform

        transform = JinjaInputTransform(template="Hello {{ input.name }}")
        adapter = self._make_adapter(base_task, input_transform=transform)
        result = adapter._apply_input_transform({"name": "world"})
        assert result == "Hello world"

    def test_apply_input_transform_error_wraps_with_prefix(self, base_task):
        """Runtime errors from the template are wrapped with a descriptive prefix."""
        from kiln_ai.datamodel.input_transform import JinjaInputTransform

        transform = JinjaInputTransform(template="{{ input.x.no_such_attr }}")
        adapter = self._make_adapter(base_task, input_transform=transform)

        with pytest.raises(ValueError, match="Input transform failed:"):
            adapter._apply_input_transform({"x": "a string"})

    @pytest.mark.asyncio
    async def test_input_transform_mcp_unchanged(self, base_task):
        """MCP run config: pipeline behavior unchanged (no input_transform field)."""
        from kiln_ai.datamodel.run_config import (
            McpRunConfigProperties,
            MCPToolReference,
        )

        mcp_config = McpRunConfigProperties(
            tool_reference=MCPToolReference(tool_id="mcp::local::server::tool"),
        )
        adapter = MockAdapter(task=base_task, run_config=mcp_config)

        result = adapter._apply_input_transform("some input")
        assert result == "some input"

        result_dict = adapter._apply_input_transform({"key": "val"})
        assert result_dict == {"key": "val"}
