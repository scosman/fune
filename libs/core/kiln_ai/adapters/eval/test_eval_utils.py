import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletionMessageToolCallParam

from kiln_ai.adapters.eval.eval_utils.eval_trace_formatter import EvalTraceFormatter
from kiln_ai.adapters.eval.eval_utils.eval_utils import EvalUtils
from kiln_ai.datamodel import DataSource, Task, TaskOutput, TaskRun
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties, ToolsRunConfig
from kiln_ai.utils.open_ai_types import ChatCompletionMessageParam


class TestEvalTraceFormatter:
    """Test cases for EvalTraceFormatter class"""

    def test_role_from_message(self):
        message: ChatCompletionMessageParam = {"role": "user", "content": "test"}  # type: ignore
        assert EvalTraceFormatter.role_from_message(message) == "user"

        message = {"role": "assistant", "content": "test"}  # type: ignore
        assert EvalTraceFormatter.role_from_message(message) == "assistant"

        message = {"role": "system", "content": "test"}  # type: ignore
        assert EvalTraceFormatter.role_from_message(message) == "system"

    def test_content_from_message_simple(self):
        message: ChatCompletionMessageParam = {"role": "user", "content": "Hello"}  # type: ignore
        assert EvalTraceFormatter.content_from_message(message) == "Hello"

    def test_content_from_message_none(self):
        message: ChatCompletionMessageParam = {"role": "user"}  # type: ignore
        assert EvalTraceFormatter.content_from_message(message) is None

        message = {"role": "user", "content": None}  # type: ignore
        assert EvalTraceFormatter.content_from_message(message) is None

    def test_content_from_message_tool_with_json_output(self):
        tool_output = json.dumps({"output": "tool result", "other": "ignored"})
        message: ChatCompletionMessageParam = {
            "role": "tool",
            "content": tool_output,
            "tool_call_id": "call_123",
        }  # type: ignore
        assert EvalTraceFormatter.content_from_message(message) == "tool result"

    def test_content_from_message_tool_with_plain_content(self):
        message: ChatCompletionMessageParam = {
            "role": "tool",
            "content": "plain tool result",
            "tool_call_id": "call_123",
        }  # type: ignore
        assert EvalTraceFormatter.content_from_message(message) == "plain tool result"

    def test_content_from_message_tool_with_invalid_json(self):
        message: ChatCompletionMessageParam = {
            "role": "tool",
            "content": "{invalid json}",
            "tool_call_id": "call_123",
        }  # type: ignore
        assert EvalTraceFormatter.content_from_message(message) == "{invalid json}"

    def test_content_from_message_tool_with_json_no_output(self):
        tool_output = json.dumps({"other": "field"})
        message: ChatCompletionMessageParam = {
            "role": "tool",
            "content": tool_output,
            "tool_call_id": "call_123",
        }  # type: ignore
        assert EvalTraceFormatter.content_from_message(message) == tool_output

    def test_reasoning_content_from_message_with_reasoning(self):
        message: ChatCompletionMessageParam = {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "I need to think about this...",
        }  # type: ignore
        assert (
            EvalTraceFormatter.reasoning_content_from_message(message)
            == "I need to think about this..."
        )

    def test_reasoning_content_from_message_without_reasoning(self):
        message: ChatCompletionMessageParam = {"role": "assistant", "content": "answer"}  # type: ignore
        assert EvalTraceFormatter.reasoning_content_from_message(message) is None

        message = {"role": "assistant", "content": "answer", "reasoning_content": None}  # type: ignore
        assert EvalTraceFormatter.reasoning_content_from_message(message) is None

    def test_tool_calls_from_message_with_tool_calls(self):
        tool_calls: list[ChatCompletionMessageToolCallParam] = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
            }
        ]
        message: ChatCompletionMessageParam = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }
        result = EvalTraceFormatter.tool_calls_from_message(message)
        assert result == tool_calls

    def test_tool_calls_from_message_without_tool_calls(self):
        message: ChatCompletionMessageParam = {"role": "assistant", "content": "test"}  # type: ignore
        assert EvalTraceFormatter.tool_calls_from_message(message) is None

    def test_formatted_tool_calls_from_message_single(self):
        tool_calls: list[ChatCompletionMessageToolCallParam] = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
            }
        ]
        message: ChatCompletionMessageParam = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }  # type: ignore
        result = EvalTraceFormatter.formatted_tool_calls_from_message(message)
        expected = """- Tool Name: test_tool
- Arguments: {"arg": "value"}"""
        assert result == expected

    def test_formatted_tool_calls_from_message_multiple(self):
        tool_calls: list[ChatCompletionMessageToolCallParam] = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "tool1", "arguments": '{"arg1": "value1"}'},
            },
            {
                "id": "call_124",
                "type": "function",
                "function": {"name": "tool2", "arguments": '{"arg2": "value2"}'},
            },
        ]
        message: ChatCompletionMessageParam = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }  # type: ignore
        result = EvalTraceFormatter.formatted_tool_calls_from_message(message)
        expected = """- Tool Name: tool1
- Arguments: {"arg1": "value1"}

- Tool Name: tool2
- Arguments: {"arg2": "value2"}"""
        assert result == expected

    def test_formatted_tool_calls_from_message_none(self):
        message: ChatCompletionMessageParam = {"role": "assistant", "content": "test"}  # type: ignore
        assert EvalTraceFormatter.formatted_tool_calls_from_message(message) is None

    def test_origin_tool_call_name_from_message_found(self):
        tool_calls: list[ChatCompletionMessageToolCallParam] = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
            }
        ]
        assistant_message: ChatCompletionMessageParam = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }
        tool_message: ChatCompletionMessageParam = {
            "role": "tool",
            "content": "result",
            "tool_call_id": "call_123",
        }
        trace = [assistant_message, tool_message]
        result = EvalTraceFormatter.origin_tool_call_name_from_message(
            tool_message, trace
        )
        assert result == "test_tool"

    def test_origin_tool_call_name_from_message_not_found(self):
        tool_message: ChatCompletionMessageParam = {
            "role": "tool",
            "content": "result",
            "tool_call_id": "call_999",
        }  # type: ignore
        trace: list[ChatCompletionMessageParam] = [
            {
                "role": "assistant",
                "content": "test",
            }  # type: ignore
        ]
        result = EvalTraceFormatter.origin_tool_call_name_from_message(
            tool_message, trace
        )
        assert result is None

    def test_origin_tool_call_name_from_message_no_tool_call_id(self):
        message: ChatCompletionMessageParam = {"role": "assistant", "content": "test"}  # type: ignore
        trace: list[ChatCompletionMessageParam] = [message]
        result = EvalTraceFormatter.origin_tool_call_name_from_message(message, trace)
        assert result is None

    def test_message_details_from_message_complete(self):
        tool_calls: list[ChatCompletionMessageToolCallParam] = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
            }
        ]
        message: ChatCompletionMessageParam = {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "thinking",
            "tool_calls": tool_calls,
        }  # type: ignore
        details = EvalTraceFormatter.message_details_from_message(message)
        assert details.role == "assistant"
        assert details.content == "answer"
        assert details.reasoning_content == "thinking"
        expected_tool_calls = """- Tool Name: test_tool
- Arguments: {"arg": "value"}"""
        assert details.tool_calls == expected_tool_calls

    def test_format_message(self):
        result = EvalTraceFormatter.format_message(
            "user", "user_message", "Hello world"
        )
        expected = """user:
<user_message>
Hello world
</user_message>"""
        assert result == expected

    def test_trace_to_formatted_conversation_history_simple(self):
        trace: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Hello"},  # type: ignore
            {"role": "assistant", "content": "Hi there"},  # type: ignore
        ]
        result = EvalTraceFormatter.trace_to_formatted_conversation_history(trace)
        expected = """user:
<user_message>
Hello
</user_message>

assistant:
<assistant_message>
Hi there
</assistant_message>"""
        assert result == expected

    def test_trace_to_formatted_conversation_history_with_reasoning(self):
        trace: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "What is 2+2?"},  # type: ignore
            {
                "role": "assistant",
                "reasoning_content": "I need to add 2 and 2 together.",
            },  # type: ignore
        ]
        result = EvalTraceFormatter.trace_to_formatted_conversation_history(trace)
        expected = """user:
<user_message>
What is 2+2?
</user_message>

assistant reasoning:
<assistant_reasoning_message>
I need to add 2 and 2 together.
</assistant_reasoning_message>"""
        assert result == expected

        trace_with_content: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "What is 2+2?"},  # type: ignore
            {
                "role": "assistant",
                "content": "4",
                "reasoning_content": "I need to add 2 and 2 together.",
            },  # type: ignore
        ]
        result_with_content = (
            EvalTraceFormatter.trace_to_formatted_conversation_history(
                trace_with_content
            )
        )
        expected_with_content = """user:
<user_message>
What is 2+2?
</user_message>

assistant reasoning:
<assistant_reasoning_message>
I need to add 2 and 2 together.
</assistant_reasoning_message>

assistant:
<assistant_message>
4
</assistant_message>"""
        assert result_with_content == expected_with_content

    def test_trace_to_formatted_conversation_history_with_tool_calls(self):
        tool_calls: list[ChatCompletionMessageToolCallParam] = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": '{"operation": "add", "a": 2, "b": 2}',
                },
            }
        ]
        trace: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Calculate 2+2"},  # type: ignore
            {"role": "assistant", "tool_calls": tool_calls},  # type: ignore
        ]
        result = EvalTraceFormatter.trace_to_formatted_conversation_history(trace)
        expected = """user:
<user_message>
Calculate 2+2
</user_message>

assistant requested tool calls:
<assistant_requested_tool_calls>
- Tool Name: calculator
- Arguments: {"operation": "add", "a": 2, "b": 2}
</assistant_requested_tool_calls>"""
        assert result == expected

    def test_trace_to_formatted_conversation_history_with_tool_message(self):
        tool_calls: list[ChatCompletionMessageToolCallParam] = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
            }
        ]
        tool_output = json.dumps({"output": "tool result"})
        trace: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Use a tool"},  # type: ignore
            {"role": "assistant", "tool_calls": tool_calls},  # type: ignore
            {
                "role": "tool",
                "content": tool_output,
                "tool_call_id": "call_123",
            },  # type: ignore
        ]
        result = EvalTraceFormatter.trace_to_formatted_conversation_history(trace)
        expected = """user:
<user_message>
Use a tool
</user_message>

assistant requested tool calls:
<assistant_requested_tool_calls>
- Tool Name: test_tool
- Arguments: {"arg": "value"}
</assistant_requested_tool_calls>

tool result from test_tool:
<tool_tool_message>
tool result
</tool_tool_message>"""
        assert result == expected

    def test_trace_to_formatted_conversation_history_tool_message_no_origin(self):
        trace: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Test"},  # type: ignore
            {
                "role": "tool",
                "content": "result",
                "tool_call_id": "call_999",
            },  # type: ignore
        ]
        result = EvalTraceFormatter.trace_to_formatted_conversation_history(trace)
        # The originating call is absent, so the tool cannot be named — but the
        # value it returned is still what a judge needs to see.
        expected = """user:
<user_message>
Test
</user_message>

tool result:
<tool_tool_message>
result
</tool_tool_message>"""
        assert result == expected

    def test_trace_to_formatted_conversation_history_empty(self):
        trace: list[ChatCompletionMessageParam] = []
        result = EvalTraceFormatter.trace_to_formatted_conversation_history(trace)
        assert result == ""

    def test_trace_to_formatted_conversation_history_empty_messages(self):
        trace: list[ChatCompletionMessageParam] = [
            {"role": "user"},  # type: ignore
            {"role": "assistant"},  # type: ignore
        ]
        result = EvalTraceFormatter.trace_to_formatted_conversation_history(trace)
        assert result == ""

    def test_trace_to_formatted_conversation_history_emits_every_block(
        self,
    ):
        tool_calls: list[ChatCompletionMessageToolCallParam] = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
            }
        ]
        trace_all: list[ChatCompletionMessageParam] = [
            {
                "role": "assistant",
                "content": "Final answer",
                "reasoning_content": "Thinking step",
                "tool_calls": tool_calls,
            },  # type: ignore
        ]
        result_all = EvalTraceFormatter.trace_to_formatted_conversation_history(
            trace_all
        )
        # One message carrying reasoning, text and a tool call emits all three,
        # in the order the model produced them: it thinks, says what it is about
        # to do, then does it.
        expected_all = """assistant reasoning:
<assistant_reasoning_message>
Thinking step
</assistant_reasoning_message>

assistant:
<assistant_message>
Final answer
</assistant_message>

assistant requested tool calls:
<assistant_requested_tool_calls>
- Tool Name: test_tool
- Arguments: {"arg": "value"}
</assistant_requested_tool_calls>"""
        assert result_all == expected_all

        trace_reasoning_only: list[ChatCompletionMessageParam] = [
            {
                "role": "assistant",
                "reasoning_content": "Thinking step",
            },  # type: ignore
        ]
        result_reasoning = EvalTraceFormatter.trace_to_formatted_conversation_history(
            trace_reasoning_only
        )
        expected_reasoning = """assistant reasoning:
<assistant_reasoning_message>
Thinking step
</assistant_reasoning_message>"""
        assert result_reasoning == expected_reasoning

        trace_tool_only: list[ChatCompletionMessageParam] = [
            {
                "role": "assistant",
                "tool_calls": tool_calls,
            },  # type: ignore
        ]
        result_tool = EvalTraceFormatter.trace_to_formatted_conversation_history(
            trace_tool_only
        )
        expected_tool = """assistant requested tool calls:
<assistant_requested_tool_calls>
- Tool Name: test_tool
- Arguments: {"arg": "value"}
</assistant_requested_tool_calls>"""
        assert result_tool == expected_tool


class TestEvalUtils:
    """Test cases for EvalUtils class"""

    @pytest.mark.asyncio
    async def test_formatted_available_tools_from_task_run_no_parent_task(self):
        task_run = MagicMock(spec=TaskRun)
        task_run.parent_task.return_value = None

        result = await EvalUtils.formatted_available_tools_from_task_run(task_run)
        assert result is None

    @pytest.mark.asyncio
    async def test_formatted_available_tools_from_task_run_no_source(self):
        task = MagicMock(spec=Task)
        task_run = MagicMock(spec=TaskRun)
        task_run.parent_task.return_value = task
        task_run.output = MagicMock(spec=TaskOutput)
        task_run.output.source = None

        result = await EvalUtils.formatted_available_tools_from_task_run(task_run)
        assert result is None

    @pytest.mark.asyncio
    async def test_formatted_available_tools_from_task_run_no_run_config(self):
        task = MagicMock(spec=Task)
        task_run = MagicMock(spec=TaskRun)
        task_run.parent_task.return_value = task
        task_run.output = MagicMock(spec=TaskOutput)
        task_run.output.source = MagicMock(spec=DataSource)
        task_run.output.source.run_config = None

        result = await EvalUtils.formatted_available_tools_from_task_run(task_run)
        assert result is None

    @pytest.mark.asyncio
    async def test_formatted_available_tools_from_task_run_no_tools_config(self):
        task = MagicMock(spec=Task)
        task_run = MagicMock(spec=TaskRun)
        task_run.parent_task.return_value = task
        task_run.output = MagicMock(spec=TaskOutput)
        task_run.output.source = MagicMock(spec=DataSource)
        task_run.output.source.run_config = MagicMock(spec=KilnAgentRunConfigProperties)
        task_run.output.source.run_config.tools_config = None

        result = await EvalUtils.formatted_available_tools_from_task_run(task_run)
        assert result is None

    @pytest.mark.asyncio
    async def test_formatted_available_tools_from_task_run_empty_tools(self):
        task = MagicMock(spec=Task)
        task_run = MagicMock(spec=TaskRun)
        task_run.parent_task.return_value = task
        task_run.output = MagicMock(spec=TaskOutput)
        task_run.output.source = MagicMock(spec=DataSource)
        task_run.output.source.run_config = MagicMock(spec=KilnAgentRunConfigProperties)
        task_run.output.source.run_config.tools_config = MagicMock(spec=ToolsRunConfig)
        task_run.output.source.run_config.tools_config.tools = []

        result = await EvalUtils.formatted_available_tools_from_task_run(task_run)
        assert result == ""

    @pytest.mark.asyncio
    async def test_formatted_available_tools_from_task_run_single_tool(self):
        task = MagicMock(spec=Task)
        task_run = MagicMock(spec=TaskRun)
        task_run.parent_task.return_value = task
        task_run.output = MagicMock(spec=TaskOutput)
        task_run.output.source = MagicMock(spec=DataSource)
        task_run.output.source.run_config = MagicMock(spec=KilnAgentRunConfigProperties)
        task_run.output.source.run_config.tools_config = MagicMock(spec=ToolsRunConfig)
        task_run.output.source.run_config.tools_config.tools = ["add_numbers"]

        mock_tool = AsyncMock()
        mock_tool.toolcall_definition = AsyncMock(
            return_value={
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers together and return the result",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "number",
                                "description": "The first number to add",
                            },
                            "b": {
                                "type": "number",
                                "description": "The second number to add",
                            },
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        )

        with patch(
            "kiln_ai.adapters.eval.eval_utils.eval_utils.tool_from_id",
            return_value=mock_tool,
        ):
            result = await EvalUtils.formatted_available_tools_from_task_run(task_run)

        expected = """<tool>
<tool_name>
add</tool_name>
<tool_description>
Add two numbers together and return the result</tool_description>
<tool_parameters>
{
  "type": "object",
  "properties": {
    "a": {
      "type": "number",
      "description": "The first number to add"
    },
    "b": {
      "type": "number",
      "description": "The second number to add"
    }
  },
  "required": [
    "a",
    "b"
  ]
}</tool_parameters>
</tool>"""
        assert result == expected

    @pytest.mark.asyncio
    async def test_formatted_available_tools_from_task_run_multiple_tools(self):
        task = MagicMock(spec=Task)
        task_run = MagicMock(spec=TaskRun)
        task_run.parent_task.return_value = task
        task_run.output = MagicMock(spec=TaskOutput)
        task_run.output.source = MagicMock(spec=DataSource)
        task_run.output.source.run_config = MagicMock(spec=KilnAgentRunConfigProperties)
        task_run.output.source.run_config.tools_config = MagicMock(spec=ToolsRunConfig)
        task_run.output.source.run_config.tools_config.tools = [
            "add_numbers",
            "subtract_numbers",
        ]

        def tool_definition_side_effect(tool_id):
            if tool_id == "add_numbers":
                return {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            elif tool_id == "subtract_numbers":
                return {
                    "type": "function",
                    "function": {
                        "name": "subtract",
                        "description": "Subtract two numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            return None

        mock_tool1 = AsyncMock()
        mock_tool1.toolcall_definition = AsyncMock(
            return_value=tool_definition_side_effect("add_numbers")
        )
        mock_tool2 = AsyncMock()
        mock_tool2.toolcall_definition = AsyncMock(
            return_value=tool_definition_side_effect("subtract_numbers")
        )

        def tool_from_id_side_effect(tool_id, parent_task):
            if tool_id == "add_numbers":
                return mock_tool1
            elif tool_id == "subtract_numbers":
                return mock_tool2
            return None

        with patch(
            "kiln_ai.adapters.eval.eval_utils.eval_utils.tool_from_id",
            side_effect=tool_from_id_side_effect,
        ):
            result = await EvalUtils.formatted_available_tools_from_task_run(task_run)

        assert result is not None
        assert "<tool>" in result
        assert "<tool_name>\nadd</tool_name>" in result
        assert "<tool_name>\nsubtract</tool_name>" in result
        assert "Add two numbers" in result
        assert "Subtract two numbers" in result
        assert "\n\n" in result

    @pytest.mark.asyncio
    async def test_formatted_available_tools_from_task_run_tool_error(self):
        task = MagicMock(spec=Task)
        task_run = MagicMock(spec=TaskRun)
        task_run.parent_task.return_value = task
        task_run.output = MagicMock(spec=TaskOutput)
        task_run.output.source = MagicMock(spec=DataSource)
        task_run.output.source.run_config = MagicMock(spec=KilnAgentRunConfigProperties)
        task_run.output.source.run_config.tools_config = MagicMock(spec=ToolsRunConfig)
        task_run.output.source.run_config.tools_config.tools = [
            "add_numbers",
            "broken_tool",
        ]

        mock_tool1 = AsyncMock()
        mock_tool1.toolcall_definition = AsyncMock(
            return_value={
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        )

        mock_tool2 = AsyncMock()
        mock_tool2.toolcall_definition = AsyncMock(
            side_effect=ValueError("Tool broken")
        )

        def tool_from_id_side_effect(tool_id, parent_task):
            if tool_id == "add_numbers":
                return mock_tool1
            elif tool_id == "broken_tool":
                return mock_tool2
            return None

        with patch(
            "kiln_ai.adapters.eval.eval_utils.eval_utils.tool_from_id",
            side_effect=tool_from_id_side_effect,
        ):
            result = await EvalUtils.formatted_available_tools_from_task_run(task_run)

        assert result is not None
        assert "<tool>" in result
        assert "<tool_name>\nadd</tool_name>" in result
        assert "broken_tool" not in result


class TestStructuredOutputRendering:
    """A structured answer arrives as arguments to the internal task_response
    tool. It is the model's answer, not a tool it chose to call."""

    def _task_response_call(self, arguments: str):
        return {
            "id": "call_tr",
            "type": "function",
            "function": {"name": "task_response", "arguments": arguments},
        }

    def test_renders_as_the_answer_not_a_tool_call(self):
        answer = '{"category": "refund_status"}'
        trace: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Where is my refund?"},  # type: ignore
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [self._task_response_call(answer)],
            },  # type: ignore
        ]
        result = EvalTraceFormatter.trace_to_formatted_conversation_history(trace)
        expected = """user:
<user_message>
Where is my refund?
</user_message>

assistant:
<assistant_message>
{"category": "refund_status"}
</assistant_message>"""
        assert result == expected
        assert "task_response" not in result

    def test_real_calls_are_still_listed_beside_it(self):
        """The wrapper is excluded from the tool-call listing; a genuine call in
        the same message is not. Without this the judge would be shown a tool
        the user never wrote, sitting alongside one they did."""
        answer = '{"total": 114.75}'
        trace: list[ChatCompletionMessageParam] = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "multiply",
                            "arguments": '{"a": 135, "b": 0.15}',
                        },
                    },
                    self._task_response_call(answer),
                ],
            },  # type: ignore
        ]
        result = EvalTraceFormatter.trace_to_formatted_conversation_history(trace)
        expected = """assistant:
<assistant_message>
{"total": 114.75}
</assistant_message>

assistant requested tool calls:
<assistant_requested_tool_calls>
- Tool Name: multiply
- Arguments: {"a": 135, "b": 0.15}
</assistant_requested_tool_calls>"""
        assert result == expected
        assert "task_response" not in result

    def test_wrapper_alone_emits_no_tool_call_block(self):
        message: ChatCompletionMessageParam = {
            "role": "assistant",
            "tool_calls": [self._task_response_call('{"a": 1}')],
        }  # type: ignore
        assert EvalTraceFormatter.formatted_tool_calls_from_message(message) is None

    def test_last_wrapper_wins(self):
        """Matches the adapter: the final task_response is the output the run
        was saved with."""
        message: ChatCompletionMessageParam = {
            "role": "assistant",
            "tool_calls": [
                self._task_response_call('{"first": true}'),
                self._task_response_call('{"second": true}'),
            ],
        }  # type: ignore
        assert (
            EvalTraceFormatter.structured_output_from_message(message)
            == '{"second": true}'
        )
