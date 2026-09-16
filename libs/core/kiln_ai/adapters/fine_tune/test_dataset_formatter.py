import json
import logging
import re
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from kiln_ai.adapters.chat.chat_formatter import (
    COT_FINAL_ANSWER_PROMPT,
    BasicChatMessage,
    ToolCallMessage,
    ToolResponseMessage,
)
from kiln_ai.adapters.fine_tune.dataset_formatter import (
    DatasetFormat,
    DatasetFormatter,
    build_training_chat,
    generate_chat_message_response,
    generate_chat_message_toolcall,
    generate_huggingface_chat_template,
    generate_huggingface_chat_template_toolcall,
    serialize_r1_style_message,
)
from kiln_ai.adapters.fine_tune.vertex_formatter import (
    VERTEX_GEMINI_ROLE_MAP,
    generate_vertex_gemini,
)
from kiln_ai.datamodel import (
    DatasetSplit,
    DataSource,
    DataSourceType,
    Task,
    TaskOutput,
    TaskRun,
)
from kiln_ai.datamodel.datamodel_enums import ChatStrategy
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties, ToolsRunConfig
from kiln_ai.datamodel.skill import Skill

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_task():
    task = Mock(spec=Task, thinking_instruction=None)
    task_runs = [
        Mock(
            spec=TaskRun,
            **{
                "id": f"run{i}",
                "input": '{"test": "input 你好"}',
                "repaired_output": None,
                "intermediate_outputs": {},
                "thinking_training_data": Mock(return_value=None),
                "input_source": Mock(
                    spec=DataSource,
                    **{
                        "type": DataSourceType.human,
                        "properties": {"created_by": "test"},
                    },
                ),
                "output": Mock(
                    spec=TaskOutput,
                    **{
                        "output": '{"test":   "output 你好"}',
                        "source": Mock(
                            spec=DataSource,
                            **{
                                "type": DataSourceType.synthetic,
                                "properties": {
                                    "model_name": "test",
                                    "model_provider": "test",
                                    "adapter_name": "test",
                                },
                                "run_config": None,
                            },
                        ),
                    },
                ),
                "trace": [
                    {
                        "content": "system message",
                        "role": "system",
                    },
                    {"content": '{"test": "input 你好"}', "role": "user"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call_tool{i}_1",
                                "function": {
                                    "arguments": '{"value": "intermediate"}',
                                    "name": "helper_tool",
                                },
                                "type": "function",
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "intermediate result",
                        "tool_call_id": f"call_tool{i}_1",
                    },
                    {
                        "role": "assistant",
                        "content": '{"test": "output 你好"}',
                    },
                ],
            },
        )
        for i in range(1, 4)
    ]

    # Set up parent_task reference for each TaskRun
    for run in task_runs:
        run.parent_task = Mock(return_value=task)

    task.runs.return_value = task_runs
    return task


@pytest.fixture
def mock_intermediate_outputs(mock_task):
    for run in mock_task.runs(include_intermediate_runs=True):
        run.intermediate_outputs = {"reasoning": "thinking output"}
        run.thinking_training_data.return_value = "thinking output"
    mock_task.thinking_instruction = "thinking instructions"
    return mock_task


@pytest.fixture
def mock_dataset(mock_task):
    dataset = Mock(spec=DatasetSplit)
    dataset.name = "test_dataset"
    dataset.parent_task.return_value = mock_task
    dataset.split_contents = {"train": ["run1", "run2"], "test": ["run3"]}
    return dataset


@pytest.fixture
def mock_training_chat_short():
    return [
        BasicChatMessage(role="system", content="system message"),
        BasicChatMessage(
            role="user",
            content="test input",
        ),
        BasicChatMessage(role="assistant", content="test output"),
    ]


@pytest.fixture
def mock_training_chat_two_step_plaintext():
    return [
        BasicChatMessage(role="system", content="system message"),
        BasicChatMessage(
            role="user",
            content="The input is:\n<user_input>\ntest input\n</user_input>\n\nthinking instructions",
        ),
        BasicChatMessage(role="assistant", content="thinking output"),
        BasicChatMessage(role="user", content="thinking final answer prompt"),
        BasicChatMessage(role="assistant", content="test output"),
    ]


@pytest.fixture
def mock_training_chat_two_step_json():
    return [
        BasicChatMessage(role="system", content="system message"),
        BasicChatMessage(
            role="user",
            content="The input is:\n<user_input>\ntest input\n</user_input>\n\nthinking instructions",
        ),
        BasicChatMessage(role="assistant", content="thinking output"),
        BasicChatMessage(role="user", content="thinking final answer prompt"),
        BasicChatMessage(role="assistant", content='{"a":"你好"}'),
    ]


def test_generate_chat_message_response(mock_training_chat_two_step_plaintext):
    result = generate_chat_message_response(mock_training_chat_two_step_plaintext)

    assert result == {
        "messages": [
            {"role": "system", "content": "system message"},
            {
                "role": "user",
                "content": "The input is:\n<user_input>\ntest input\n</user_input>\n\nthinking instructions",
            },
            {"role": "assistant", "content": "thinking output"},
            {"role": "user", "content": "thinking final answer prompt"},
            {"role": "assistant", "content": "test output"},
        ]
    }


def test_generate_chat_message_response_json(mock_training_chat_two_step_json):
    result = generate_chat_message_response(mock_training_chat_two_step_json)

    assert result == {
        "messages": [
            {"role": "system", "content": "system message"},
            {
                "role": "user",
                "content": "The input is:\n<user_input>\ntest input\n</user_input>\n\nthinking instructions",
            },
            {"role": "assistant", "content": "thinking output"},
            {"role": "user", "content": "thinking final answer prompt"},
            {"role": "assistant", "content": '{"a":"你好"}'},
        ]
    }


def test_generate_chat_message_toolcall(mock_training_chat_two_step_json):
    result = generate_chat_message_toolcall(mock_training_chat_two_step_json)

    assert result == {
        "messages": [
            {"role": "system", "content": "system message"},
            {
                "role": "user",
                "content": "The input is:\n<user_input>\ntest input\n</user_input>\n\nthinking instructions",
            },
            {"role": "assistant", "content": "thinking output"},
            {"role": "user", "content": "thinking final answer prompt"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "task_response",
                            "arguments": '{"a": "你好"}',
                        },
                    }
                ],
            },
        ]
    }


def test_generate_chat_message_toolcall_invalid_json(mock_training_chat_two_step_json):
    mock_training_chat_two_step_json[-1].content = "invalid json"
    with pytest.raises(ValueError, match=r"^Last message is not JSON"):
        generate_chat_message_toolcall(mock_training_chat_two_step_json)


async def test_dataset_formatter_dump_invalid_format(mock_dataset):
    formatter = DatasetFormatter(mock_dataset, "system message")

    with pytest.raises(ValueError, match="Unsupported format"):
        await formatter.dump_to_file(
            "train", "invalid_format", ChatStrategy.single_turn
        )


async def test_dataset_formatter_dump_invalid_split(mock_dataset):
    formatter = DatasetFormatter(mock_dataset, "system message")

    with pytest.raises(ValueError, match="Split invalid_split not found in dataset"):
        await formatter.dump_to_file(
            "invalid_split",
            DatasetFormat.OPENAI_CHAT_JSONL,
            ChatStrategy.single_turn,
        )


async def test_dataset_formatter_dump_to_file(mock_dataset, tmp_path):
    formatter = DatasetFormatter(mock_dataset, "system message")
    output_path = tmp_path / "output.jsonl"

    result_path = await formatter.dump_to_file(
        "train",
        DatasetFormat.OPENAI_CHAT_JSONL,
        path=output_path,
        data_strategy=ChatStrategy.single_turn,
    )

    assert result_path == output_path
    assert output_path.exists()

    # Verify file contents
    with open(output_path) as f:
        lines = f.readlines()
        assert len(lines) == 2  # Should have 2 entries for train split
        for line in lines:
            data = json.loads(line)
            assert "messages" in data
            assert len(data["messages"]) == 5
            assert data["messages"][0]["content"] == "system message"
            assert data["messages"][1]["content"] == '{"test": "input 你好"}'
            assert data["messages"][2]["role"] == "assistant"
            assert data["messages"][3]["role"] == "tool"
            # Raw chat doesn't fix json issues, like extra spaces
            assert data["messages"][4]["content"] == '{"test":   "output 你好"}'


async def test_dataset_formatter_dump_to_temp_file(mock_dataset):
    formatter = DatasetFormatter(mock_dataset, "system message 你好")

    result_path = await formatter.dump_to_file(
        "train",
        DatasetFormat.OPENAI_CHAT_JSONL,
        data_strategy=ChatStrategy.single_turn,
    )

    assert result_path.exists()
    assert result_path.parent == Path(tempfile.gettempdir())
    # Test our nice naming
    assert result_path.name.startswith(
        "test_dataset -- split-train -- format-openai_chat_jsonl -- no-cot.jsonl"
    )
    assert result_path.name.endswith(".jsonl")
    # Verify file contents
    with open(result_path) as f:
        lines = f.readlines()
        assert len(lines) == 2
        # check non-ascii characters are not escaped
        assert "你好" in lines[0]
        assert "你好" in lines[1]

        # confirm didn't use COT for final_only
        assert "thinking output" not in lines[0]
        assert "thinking instructions" not in lines[0]


async def test_dataset_formatter_dump_to_file_tool_format(mock_dataset, tmp_path):
    formatter = DatasetFormatter(mock_dataset, "system message")
    output_path = tmp_path / "output.jsonl"

    result_path = await formatter.dump_to_file(
        "train",
        DatasetFormat.OPENAI_CHAT_TOOLCALL_JSONL,
        path=output_path,
        data_strategy=ChatStrategy.single_turn,
    )

    assert result_path == output_path
    assert output_path.exists()

    # Verify file contents
    with open(output_path) as f:
        lines = f.readlines()
        assert len(lines) == 2  # Should have 2 entries for train split
        for line in lines:
            data = json.loads(line)
            assert "messages" in data
            assert len(data["messages"]) == 5
            # Check system and user messages
            assert data["messages"][0]["content"] == "system message"
            assert data["messages"][1]["content"] == '{"test": "input 你好"}'
            # Check trace tool calls (from trace)
            assert data["messages"][2]["role"] == "assistant"
            assert data["messages"][3]["role"] == "tool"
            # Check final tool call format (task_response)
            assistant_msg = data["messages"][4]
            assert assistant_msg["content"] is None
            assert "tool_calls" in assistant_msg
            assert len(assistant_msg["tool_calls"]) == 1
            tool_call = assistant_msg["tool_calls"][0]
            assert tool_call["type"] == "function"
            assert tool_call["function"]["name"] == "task_response"
            assert tool_call["function"]["arguments"] == '{"test": "output 你好"}'


async def test_dataset_formatter_dump_with_intermediate_data(
    mock_dataset, mock_intermediate_outputs
):
    formatter = DatasetFormatter(
        mock_dataset,
        "system message 你好",
        thinking_instructions="thinking instructions",
    )

    result_path = await formatter.dump_to_file(
        "train",
        DatasetFormat.OPENAI_CHAT_JSONL,
        data_strategy=ChatStrategy.two_message_cot_legacy,
    )

    assert result_path.exists()
    assert result_path.parent == Path(tempfile.gettempdir())
    # Test our nice naming, with cot
    assert (
        result_path.name
        == "test_dataset -- split-train -- format-openai_chat_jsonl -- cot.jsonl"
    )
    # Verify file contents
    with open(result_path) as f:
        lines = f.readlines()
        assert len(lines) == 2
        for line in lines:
            assert "thinking output" in line
            assert "thinking instructions" in line


async def test_dataset_formatter_dump_with_intermediate_data_r1_style(
    mock_dataset, mock_intermediate_outputs
):
    formatter = DatasetFormatter(
        mock_dataset,
        "system message 你好",
        thinking_instructions=None,
    )

    result_path = await formatter.dump_to_file(
        "train",
        DatasetFormat.OPENAI_CHAT_JSONL,
        data_strategy=ChatStrategy.single_turn_r1_thinking,
    )

    assert result_path.exists()
    assert result_path.parent == Path(tempfile.gettempdir())
    # Test our nice naming, with cot
    assert (
        result_path.name
        == "test_dataset -- split-train -- format-openai_chat_jsonl -- cot.jsonl"
    )
    # Verify file contents
    with open(result_path) as f:
        lines = f.readlines()
        assert len(lines) == 2
        for line in lines:
            assert "<think>" in line
            assert "</think>" in line


async def test_dataset_formatter_dump_with_intermediate_data_custom_instructions(
    mock_dataset, mock_intermediate_outputs
):
    formatter = DatasetFormatter(
        mock_dataset, "custom system message 你好", "custom thinking instructions"
    )

    result_path = await formatter.dump_to_file(
        "train",
        DatasetFormat.OPENAI_CHAT_JSONL,
        data_strategy=ChatStrategy.two_message_cot_legacy,
    )

    assert result_path.exists()
    assert result_path.parent == Path(tempfile.gettempdir())
    # Test our nice naming, with cot
    assert (
        result_path.name
        == "test_dataset -- split-train -- format-openai_chat_jsonl -- cot.jsonl"
    )
    # Verify file contents
    with open(result_path) as f:
        lines = f.readlines()
        assert len(lines) == 2
        for line in lines:
            assert "custom system message 你好" in line
            assert "custom thinking instructions" in line
            assert "thinking output" in line


def test_generate_huggingface_chat_template(mock_training_chat_two_step_plaintext):
    result = generate_huggingface_chat_template(mock_training_chat_two_step_plaintext)

    assert result == {
        "conversations": [
            {"role": "system", "content": "system message"},
            {
                "role": "user",
                "content": "The input is:\n<user_input>\ntest input\n</user_input>\n\nthinking instructions",
            },
            {"role": "assistant", "content": "thinking output"},
            {"role": "user", "content": "thinking final answer prompt"},
            {"role": "assistant", "content": "test output"},
        ]
    }


def test_generate_vertex_template(mock_training_chat_short):
    result = generate_vertex_gemini(mock_training_chat_short)

    assert result == {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": "system message",
                }
            ],
        },
        "contents": [
            {"role": "user", "parts": [{"text": "test input"}]},
            {"role": "model", "parts": [{"text": "test output"}]},
        ],
    }


def test_generate_vertex_template_thinking(mock_training_chat_two_step_plaintext):
    result = generate_vertex_gemini(mock_training_chat_two_step_plaintext)

    assert result == {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": "system message",
                }
            ],
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "The input is:\n<user_input>\ntest input\n</user_input>\n\nthinking instructions",
                    }
                ],
            },
            {"role": "model", "parts": [{"text": "thinking output"}]},
            {"role": "user", "parts": [{"text": "thinking final answer prompt"}]},
            {"role": "model", "parts": [{"text": "test output"}]},
        ],
    }


def test_generate_huggingface_chat_template_toolcall():
    messages = [
        BasicChatMessage("system", "system message"),
        BasicChatMessage("user", "test input"),
        BasicChatMessage("assistant", '{"key":"value"}'),
    ]

    result = generate_huggingface_chat_template_toolcall(messages)

    assert result["conversations"][0] == {"role": "system", "content": "system message"}
    assert result["conversations"][1] == {"role": "user", "content": "test input"}
    assistant_msg = result["conversations"][2]
    assert assistant_msg["role"] == "assistant"
    assert len(assistant_msg["tool_calls"]) == 1
    tool_call = assistant_msg["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "task_response"
    assert len(tool_call["function"]["id"]) == 9  # UUID is truncated to 9 chars
    assert tool_call["function"]["id"].isalnum()  # Check ID is alphanumeric
    assert tool_call["function"]["arguments"] == {"key": "value"}


def test_generate_huggingface_chat_template_toolcall_thinking(
    mock_training_chat_two_step_json,
):
    result = generate_huggingface_chat_template_toolcall(
        mock_training_chat_two_step_json
    )

    assert result["conversations"][0] == {"role": "system", "content": "system message"}
    assert result["conversations"][1] == {
        "role": "user",
        "content": "The input is:\n<user_input>\ntest input\n</user_input>\n\nthinking instructions",
    }
    assert result["conversations"][2] == {
        "role": "assistant",
        "content": "thinking output",
    }
    assert result["conversations"][3] == {
        "role": "user",
        "content": "thinking final answer prompt",
    }

    assistant_msg = result["conversations"][4]
    assert assistant_msg["role"] == "assistant"
    assert len(assistant_msg["tool_calls"]) == 1
    tool_call = assistant_msg["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "task_response"
    assert len(tool_call["function"]["id"]) == 9  # UUID is truncated to 9 chars
    assert tool_call["function"]["id"].isalnum()  # Check ID is alphanumeric
    assert tool_call["function"]["arguments"] == {"a": "你好"}


def test_generate_huggingface_chat_template_toolcall_invalid_json(
    mock_training_chat_two_step_json,
):
    mock_training_chat_two_step_json[-1].content = "invalid json"

    with pytest.raises(ValueError, match=r"^Last message is not JSON"):
        generate_huggingface_chat_template_toolcall(mock_training_chat_two_step_json)


def test_build_training_chat(mock_task):
    # Non repaired should use original output
    mock_task_run = mock_task.runs(include_intermediate_runs=True)[0]
    messages = build_training_chat(
        mock_task_run,
        "system message",
        data_strategy=ChatStrategy.single_turn,
    )

    assert len(messages) == 5
    system_msg = messages[0]
    assert system_msg.role == "system"
    assert system_msg.content == "system message"

    user_msg = messages[1]
    assert user_msg.role == "user"
    assert user_msg.content == '{"test": "input 你好"}'

    tool_call_msg = messages[2]
    assert tool_call_msg.role == "assistant"
    assert len(tool_call_msg.tool_calls) == 1

    tool_response_msg = messages[3]
    assert tool_response_msg.role == "tool"
    assert tool_response_msg.content == "intermediate result"

    final_msg = messages[4]
    assert final_msg.role == "assistant"
    assert final_msg.content == '{"test":   "output 你好"}'


def test_build_training_data_with_COT(mock_task):
    # Setup with needed fields for thinking
    mock_task_run = mock_task.runs(include_intermediate_runs=True)[0]
    assert mock_task_run.parent_task() == mock_task
    mock_task_run.intermediate_outputs = {"chain_of_thought": "cot output"}
    mock_task_run.thinking_training_data.return_value = "cot output"

    messages = build_training_chat(
        mock_task_run,
        "system message",
        data_strategy=ChatStrategy.two_message_cot,
        thinking_instructions="thinking instructions",
    )

    assert len(messages) == 7
    system_msg = messages[0]
    assert system_msg.role == "system"
    assert system_msg.content == "system message"

    user_msg = messages[1]
    assert user_msg.role == "user"
    assert (
        user_msg.content
        == 'The input is:\n<user_input>\n{"test": "input 你好"}\n</user_input>\n\nthinking instructions'
    )

    tool_call_msg = messages[2]
    assert tool_call_msg.role == "assistant"
    assert len(tool_call_msg.tool_calls) == 1

    tool_response_msg = messages[3]
    assert tool_response_msg.role == "tool"
    assert tool_response_msg.content == "intermediate result"

    assistant_msg = messages[4]
    assert assistant_msg.role == "assistant"
    assert assistant_msg.content == "cot output"

    final_answer_prompt_msg = messages[5]
    assert final_answer_prompt_msg.role == "user"
    assert final_answer_prompt_msg.content == COT_FINAL_ANSWER_PROMPT

    final_msg = messages[6]
    assert final_msg.role == "assistant"
    assert final_msg.content == '{"test":   "output 你好"}'


def test_build_training_data_with_COT_legacy(mock_task):
    # Setup with needed fields for thinking
    mock_task_run = mock_task.runs(include_intermediate_runs=True)[0]
    assert mock_task_run.parent_task() == mock_task
    mock_task_run.intermediate_outputs = {"chain_of_thought": "cot output"}
    mock_task_run.thinking_training_data.return_value = "cot output"

    messages = build_training_chat(
        mock_task_run,
        "system message",
        data_strategy=ChatStrategy.two_message_cot_legacy,
        thinking_instructions="thinking instructions",
    )

    assert len(messages) == 8
    system_msg = messages[0]
    assert system_msg.role == "system"
    assert system_msg.content == "system message"

    user_msg = messages[1]
    assert user_msg.role == "user"
    assert user_msg.content == '{"test": "input 你好"}'

    cot_msg = messages[2]
    assert cot_msg.role == "system"
    assert cot_msg.content == "thinking instructions"

    tool_call_msg = messages[3]
    assert tool_call_msg.role == "assistant"
    assert len(tool_call_msg.tool_calls) == 1

    tool_response_msg = messages[4]
    assert tool_response_msg.role == "tool"
    assert tool_response_msg.content == "intermediate result"

    assistant_msg = messages[5]
    assert assistant_msg.role == "assistant"
    assert assistant_msg.content == "cot output"

    final_answer_prompt_msg = messages[6]
    assert final_answer_prompt_msg.role == "user"
    assert final_answer_prompt_msg.content == COT_FINAL_ANSWER_PROMPT

    final_msg = messages[7]
    assert final_msg.role == "assistant"
    assert final_msg.content == '{"test":   "output 你好"}'


def test_build_training_data_with_COT_r1_style(mock_task):
    # Setup with needed fields for thinking
    mock_task_run = mock_task.runs(include_intermediate_runs=True)[0]
    assert mock_task_run.parent_task() == mock_task
    mock_task_run.intermediate_outputs = {"chain_of_thought": "cot output"}
    mock_task_run.thinking_training_data.return_value = "cot output"

    messages = build_training_chat(
        mock_task_run,
        "system message",
        data_strategy=ChatStrategy.single_turn_r1_thinking,
        thinking_instructions=None,
    )

    assert len(messages) == 5
    system_msg = messages[0]
    assert system_msg.role == "system"
    assert system_msg.content == "system message"

    user_msg = messages[1]
    assert user_msg.role == "user"
    assert user_msg.content == '{"test": "input 你好"}'

    tool_call_msg = messages[2]
    assert tool_call_msg.role == "assistant"
    assert len(tool_call_msg.tool_calls) == 1

    tool_response_msg = messages[3]
    assert tool_response_msg.role == "tool"
    assert tool_response_msg.content == "intermediate result"

    final_msg = messages[4]
    assert final_msg.role == "assistant"
    assert (
        final_msg.content
        == '<think>\ncot output\n</think>\n\n{"test":   "output 你好"}'
    )


def test_build_training_data_with_thinking(mock_task):
    # Setup with needed fields for thinking
    mock_task_run = mock_task.runs(include_intermediate_runs=True)[0]
    assert mock_task_run.parent_task() == mock_task
    # It should just use the reasoning output if both thinking and chain_of_thought are present
    mock_task_run.intermediate_outputs = {
        "reasoning": "thinking output",
        "chain_of_thought": "cot output",
    }
    mock_task_run.thinking_training_data.return_value = "thinking output"
    mock_task.thinking_instruction = "thinking instructions"
    assert mock_task.thinking_instruction == "thinking instructions"

    messages = build_training_chat(
        mock_task_run,
        "system message",
        ChatStrategy.two_message_cot,
        thinking_instructions="thinking instructions",
    )

    assert len(messages) == 7
    system_msg = messages[0]
    assert system_msg.role == "system"
    assert system_msg.content == "system message"

    user_msg = messages[1]
    assert user_msg.role == "user"
    assert (
        user_msg.content
        == 'The input is:\n<user_input>\n{"test": "input 你好"}\n</user_input>\n\nthinking instructions'
    )

    tool_call_msg = messages[2]
    assert tool_call_msg.role == "assistant"
    assert len(tool_call_msg.tool_calls) == 1

    tool_response_msg = messages[3]
    assert tool_response_msg.role == "tool"
    assert tool_response_msg.content == "intermediate result"

    assistant_msg = messages[4]
    assert assistant_msg.role == "assistant"
    assert assistant_msg.content == "thinking output"

    final_answer_prompt_msg = messages[5]
    assert final_answer_prompt_msg.role == "user"
    assert final_answer_prompt_msg.content == COT_FINAL_ANSWER_PROMPT

    final_msg = messages[6]
    assert final_msg.role == "assistant"
    assert final_msg.content == '{"test":   "output 你好"}'


def test_build_training_data_with_thinking_r1_style(mock_task):
    # Setup with needed fields for thinking
    mock_task_run = mock_task.runs(include_intermediate_runs=True)[0]
    assert mock_task_run.parent_task() == mock_task
    # It should just use the reasoning output if both thinking and chain_of_thought are present
    mock_task_run.intermediate_outputs = {
        "reasoning": "thinking output",
        "chain_of_thought": "cot output",
    }
    mock_task_run.thinking_training_data.return_value = "thinking output"
    mock_task.thinking_instruction = "thinking instructions"

    assert mock_task.thinking_instruction == "thinking instructions"

    messages = build_training_chat(
        mock_task_run,
        "system message",
        ChatStrategy.single_turn_r1_thinking,
        thinking_instructions=None,
    )

    assert len(messages) == 5
    system_msg = messages[0]
    assert system_msg.role == "system"
    assert system_msg.content == "system message"

    user_msg = messages[1]
    assert user_msg.role == "user"
    assert user_msg.content == '{"test": "input 你好"}'

    tool_call_msg = messages[2]
    assert tool_call_msg.role == "assistant"
    assert len(tool_call_msg.tool_calls) == 1

    tool_response_msg = messages[3]
    assert tool_response_msg.role == "tool"
    assert tool_response_msg.content == "intermediate result"

    final_msg = messages[4]
    assert final_msg.role == "assistant"
    assert (
        final_msg.content
        == '<think>\nthinking output\n</think>\n\n{"test":   "output 你好"}'
    )


def test_build_training_data_with_repaired_output(mock_task):
    # use repaired output if available
    mock_task_run = mock_task.runs(include_intermediate_runs=True)[0]
    mock_task_run.repair_instructions = "repair instructions"
    mock_task_run.repaired_output = TaskOutput(
        output='{"test": "repaired output"}',
        source=DataSource(
            type=DataSourceType.human,
            properties={"created_by": "test-user"},
        ),
    )

    messages = build_training_chat(
        mock_task_run,
        "system message",
        data_strategy=ChatStrategy.single_turn,
    )

    assert len(messages) == 5
    system_msg = messages[0]
    assert system_msg.role == "system"
    assert system_msg.content == "system message"

    user_msg = messages[1]
    assert user_msg.role == "user"
    assert user_msg.content == '{"test": "input 你好"}'

    tool_call_msg = messages[2]
    assert tool_call_msg.role == "assistant"
    assert len(tool_call_msg.tool_calls) == 1

    tool_response_msg = messages[3]
    assert tool_response_msg.role == "tool"
    assert tool_response_msg.content == "intermediate result"

    final_msg = messages[4]
    assert final_msg.role == "assistant"
    # Note we re-format the json
    assert final_msg.content == '{"test": "repaired output"}'


async def test_dataset_formatter_dump_to_file_json_schema_format(
    mock_dataset, tmp_path
):
    formatter = DatasetFormatter(mock_dataset, "system message")
    output_path = tmp_path / "output.jsonl"

    result_path = await formatter.dump_to_file(
        "train",
        DatasetFormat.OPENAI_CHAT_JSON_SCHEMA_JSONL,
        path=output_path,
        data_strategy=ChatStrategy.single_turn,
    )

    assert result_path == output_path
    assert output_path.exists()

    # Verify file contents
    with open(output_path) as f:
        lines = f.readlines()
        assert len(lines) == 2  # Should have 2 entries for train split
        for line in lines:
            data = json.loads(line)
            assert "messages" in data
            assert len(data["messages"]) == 5
            # Check system and user messages
            assert data["messages"][0]["content"] == "system message"
            assert data["messages"][1]["content"] == '{"test": "input 你好"}'
            # Check trace tool calls
            assert data["messages"][2]["role"] == "assistant"
            assert data["messages"][3]["role"] == "tool"
            # Check JSON format
            assistant_msg = data["messages"][4]
            assert assistant_msg["role"] == "assistant"
            # Verify the content is valid JSON
            assert assistant_msg["content"] == '{"test": "output 你好"}'
            json_content = json.loads(assistant_msg["content"])
            assert json_content == {"test": "output 你好"}


@pytest.mark.parametrize(
    "thinking,final_output,expected_output",
    [
        ("thinking", "final output", "<think>\nthinking\n</think>\n\nfinal output"),
        ("thinking", '{"name":"joe"}', '<think>\nthinking\n</think>\n\n{"name":"joe"}'),
    ],
)
def test_serialize_r1_style_message(thinking, final_output, expected_output):
    assert (
        serialize_r1_style_message(thinking=thinking, final_output=final_output)
        == expected_output
    )


@pytest.mark.parametrize(
    "thinking,final_output",
    [
        (None, "final output"),
        ("", "final output"),
        (" ", "final output"),
    ],
)
def test_serialize_r1_style_message_missing_thinking(thinking, final_output):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Thinking data is required when fine-tuning thinking models (R1, QwQ, etc). Please ensure your fine-tuning dataset contains reasoning or chain of thought output for every entry."
        ),
    ):
        serialize_r1_style_message(thinking=thinking, final_output=final_output)


def test_vertex_gemini_role_map_coverage():
    """Test that VERTEX_GEMINI_ROLE_MAP covers all possible BasicChatMessage.role values"""
    from typing import get_type_hints

    # Get the Literal type from BasicChatMessage.role
    role_type = get_type_hints(BasicChatMessage)["role"]
    # Extract the possible values from the Literal type
    possible_roles = role_type.__args__  # type: ignore

    # Check that every possible role is in the map
    for role in possible_roles:
        assert role in VERTEX_GEMINI_ROLE_MAP, (
            f"Role {role} is not mapped in VERTEX_GEMINI_ROLE_MAP"
        )

    # Check that there are no extra mappings
    assert set(VERTEX_GEMINI_ROLE_MAP.keys()) == set(possible_roles), (
        "VERTEX_GEMINI_ROLE_MAP has extra mappings"
    )


def mock_training_chat_two_step_with_tools(jsonOutput: bool = False):
    return [
        BasicChatMessage(
            role="system",
            content="You are a calculator, your task is to solve math equation by leveraging tools and standard conventions.",
        ),
        BasicChatMessage(role="user", content="Calculate 92 - (21+34)"),
        ToolCallMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_EeflFatFRBKui10Z23uTQQIN",
                    "function": {"arguments": '{"a":21,"b":34}', "name": "add"},
                    "type": "function",
                },
            ],
        ),
        ToolResponseMessage(
            role="tool", content="55", tool_call_id="call_EeflFatFRBKui10Z23uTQQIN"
        ),
        ToolCallMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_zzMBArMOdlDD0Pn3vWMliMmn",
                    "function": {"arguments": '{"a":92,"b":55}', "name": "subtract"},
                    "type": "function",
                }
            ],
        ),
        ToolResponseMessage(
            role="tool", content="37", tool_call_id="call_zzMBArMOdlDD0Pn3vWMliMmn"
        ),
        BasicChatMessage(
            role="assistant",
            content="The result of \\( 92 - (21 + 34) \\) is 37."
            if not jsonOutput
            else '{"answer": 37}',
        ),
    ]


@pytest.fixture
def mock_tool_definitions():
    return [
        {
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
        },
        {
            "type": "function",
            "function": {
                "name": "subtract",
                "description": "Subtract the second number from the first number and return the result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "The first number (minuend)",
                        },
                        "b": {
                            "type": "number",
                            "description": "The second number to subtract (subtrahend)",
                        },
                    },
                    "required": ["a", "b"],
                },
            },
        },
    ]


def test_generate_chat_message_response_with_tools(mock_tool_definitions):
    result = generate_chat_message_response(
        mock_training_chat_two_step_with_tools(),
        mock_tool_definitions,
    )
    assert result == {
        "messages": [
            {
                "role": "system",
                "content": "You are a calculator, your task is to solve math equation by leveraging tools and standard conventions.",
            },
            {"role": "user", "content": "Calculate 92 - (21+34)"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_EeflFatFRBKui10Z23uTQQIN",
                        "function": {"arguments": '{"a":21,"b":34}', "name": "add"},
                        "type": "function",
                    },
                ],
            },
            {
                "role": "tool",
                "content": "55",
                "tool_call_id": "call_EeflFatFRBKui10Z23uTQQIN",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_zzMBArMOdlDD0Pn3vWMliMmn",
                        "function": {
                            "arguments": '{"a":92,"b":55}',
                            "name": "subtract",
                        },
                        "type": "function",
                    },
                ],
            },
            {
                "role": "tool",
                "content": "37",
                "tool_call_id": "call_zzMBArMOdlDD0Pn3vWMliMmn",
            },
            {
                "role": "assistant",
                "content": "The result of \\( 92 - (21 + 34) \\) is 37.",
            },
        ],
        "tools": mock_tool_definitions,
    }


def test_generate_chat_message_response_with_tools_json(mock_tool_definitions):
    result = generate_chat_message_response(
        mock_training_chat_two_step_with_tools(jsonOutput=True),
        mock_tool_definitions,
    )
    assert result == {
        "messages": [
            {
                "role": "system",
                "content": "You are a calculator, your task is to solve math equation by leveraging tools and standard conventions.",
            },
            {"role": "user", "content": "Calculate 92 - (21+34)"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_EeflFatFRBKui10Z23uTQQIN",
                        "function": {"arguments": '{"a":21,"b":34}', "name": "add"},
                        "type": "function",
                    },
                ],
            },
            {
                "role": "tool",
                "content": "55",
                "tool_call_id": "call_EeflFatFRBKui10Z23uTQQIN",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_zzMBArMOdlDD0Pn3vWMliMmn",
                        "function": {
                            "arguments": '{"a":92,"b":55}',
                            "name": "subtract",
                        },
                        "type": "function",
                    },
                ],
            },
            {
                "role": "tool",
                "content": "37",
                "tool_call_id": "call_zzMBArMOdlDD0Pn3vWMliMmn",
            },
            {
                "role": "assistant",
                "content": '{"answer": 37}',
            },
        ],
        "tools": mock_tool_definitions,
    }


class TestGetToolDefinitionsWithSkills:
    @pytest.fixture
    def task_with_skills(self, tmp_path):
        from kiln_ai.datamodel import Project

        project = Project(name="test_project", path=tmp_path / "project.kiln")
        project.save_to_file()
        task = Task(
            name="test_task",
            instruction="do something",
            parent=project,
        )
        task.save_to_file()
        skill = Skill(name="my-skill", description="A test skill", parent=project)
        skill.save_to_file()
        return task, skill

    def _make_task_run_with_skill_tool(self, task, skill_id):
        run_config = KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name="openai",
            prompt_id="simple_prompt_builder",
            structured_output_mode="json_schema",
            tools_config=ToolsRunConfig(
                tools=[f"kiln_tool::skill::{skill_id}"],
            ),
        )
        return Mock(
            spec=TaskRun,
            output=Mock(
                source=Mock(run_config=run_config),
            ),
            parent_task=Mock(return_value=task),
        )

    async def test_skill_tool_ids_resolved(self, task_with_skills):
        task, skill = task_with_skills
        dataset = Mock(spec=DatasetSplit)
        dataset.parent_task.return_value = task
        dataset.split_contents = {"train": []}

        formatter = DatasetFormatter(dataset, "system message")

        task_run = self._make_task_run_with_skill_tool(task, skill.id)
        tool_defs = await formatter._get_tool_definitions_from_config(task_run)

        assert tool_defs is not None
        assert len(tool_defs) == 1
        assert tool_defs[0]["function"]["name"] == "skill"

    async def test_skill_tool_ids_mixed_with_builtin(self, task_with_skills):
        task, skill = task_with_skills
        run_config = KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name="openai",
            prompt_id="simple_prompt_builder",
            structured_output_mode="json_schema",
            tools_config=ToolsRunConfig(
                tools=[
                    "kiln_tool::add_numbers",
                    f"kiln_tool::skill::{skill.id}",
                ],
            ),
        )
        task_run = Mock(
            spec=TaskRun,
            output=Mock(source=Mock(run_config=run_config)),
            parent_task=Mock(return_value=task),
        )

        dataset = Mock(spec=DatasetSplit)
        dataset.parent_task.return_value = task
        dataset.split_contents = {"train": []}
        formatter = DatasetFormatter(dataset, "system message")

        tool_defs = await formatter._get_tool_definitions_from_config(task_run)

        assert tool_defs is not None
        assert len(tool_defs) == 2
        names = {d["function"]["name"] for d in tool_defs}
        assert "add" in names
        assert "skill" in names

    async def test_skill_tool_cached(self, task_with_skills):
        task, skill = task_with_skills
        dataset = Mock(spec=DatasetSplit)
        dataset.parent_task.return_value = task
        dataset.split_contents = {"train": []}
        formatter = DatasetFormatter(dataset, "system message")

        task_run = self._make_task_run_with_skill_tool(task, skill.id)
        defs1 = await formatter._get_tool_definitions_from_config(task_run)
        defs2 = await formatter._get_tool_definitions_from_config(task_run)

        assert defs1 == defs2
        assert len(formatter._tool_cache) == 1
