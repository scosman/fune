import json
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Protocol
from uuid import uuid4

from kiln_ai.adapters.chat.chat_formatter import (
    BasicChatMessage,
    ChatMessage,
    ToolCallMessage,
    ToolResponseMessage,
    get_chat_formatter,
)
from kiln_ai.adapters.chat.chat_utils import build_tool_call_messages
from kiln_ai.adapters.fine_tune.vertex_formatter import generate_vertex_gemini
from kiln_ai.datamodel import DatasetSplit, TaskRun
from kiln_ai.datamodel.datamodel_enums import THINKING_DATA_STRATEGIES, ChatStrategy
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties
from kiln_ai.datamodel.tool_id import SKILL_TOOL_ID_PREFIX
from kiln_ai.tools.base_tool import ToolCallDefinition
from kiln_ai.tools.tool_registry import tool_from_id
from kiln_ai.utils.exhaustive_error import raise_exhaustive_enum_error


class DatasetFormat(str, Enum):
    """Formats for dataset generation. Both for file format (like JSONL), and internal structure (like chat/toolcall)"""

    """OpenAI chat format with plaintext response"""
    OPENAI_CHAT_JSONL = "openai_chat_jsonl"

    """OpenAI chat format with json response_format"""
    OPENAI_CHAT_JSON_SCHEMA_JSONL = "openai_chat_json_schema_jsonl"

    """OpenAI chat format with tool call response"""
    OPENAI_CHAT_TOOLCALL_JSONL = "openai_chat_toolcall_jsonl"

    """HuggingFace chat template in JSONL"""
    HUGGINGFACE_CHAT_TEMPLATE_JSONL = "huggingface_chat_template_jsonl"

    """HuggingFace chat template with tool calls in JSONL"""
    HUGGINGFACE_CHAT_TEMPLATE_TOOLCALL_JSONL = (
        "huggingface_chat_template_toolcall_jsonl"
    )

    """Vertex Gemini format"""
    VERTEX_GEMINI = "vertex_gemini"


class FormatGenerator(Protocol):
    """Protocol for format generators"""

    def __call__(
        self,
        training_chat: list[ChatMessage],
        tools: list[ToolCallDefinition] | None = None,
    ) -> Dict[str, Any]: ...


def build_training_chat(
    task_run: TaskRun,
    system_message: str,
    data_strategy: ChatStrategy,
    thinking_instructions: str | None = None,
) -> list[ChatMessage]:
    """
    Generate chat message list for training.

    For final output, get the best task output from the task run, preferring repaired output if available.

    For thinking, get the intermediate output if it exists, otherwise return None.
    """
    final_output = task_run.output.output
    if task_run.repaired_output is not None:
        final_output = task_run.repaired_output.output

    thinking = None

    chat_formatter = get_chat_formatter(
        data_strategy,
        system_message,
        task_run.input,
        thinking_instructions,
    )
    # First turn already has its content (user message)
    chat_formatter.next_turn(None)

    # Extract the tool calls from the Traces and insert into the internal messages list
    tool_call_messages = build_tool_call_messages(task_run.trace)
    chat_formatter.append_messages(tool_call_messages)

    match data_strategy:
        case ChatStrategy.single_turn:
            chat_formatter.next_turn(final_output)
        case ChatStrategy.two_message_cot:
            thinking = get_thinking_data(task_run)
            chat_formatter.next_turn(thinking)
            chat_formatter.next_turn(final_output)
        case ChatStrategy.two_message_cot_legacy:
            thinking = get_thinking_data(task_run)
            chat_formatter.next_turn(thinking)
            chat_formatter.next_turn(final_output)
        case ChatStrategy.single_turn_r1_thinking:
            if thinking_instructions:
                raise ValueError(
                    "Thinking instructions are not supported when fine-tuning thinking models (R1, QwQ, etc). Please remove the thinking instructions."
                )

            thinking = get_thinking_data(task_run)
            response_msg = serialize_r1_style_message(thinking, final_output)
            chat_formatter.next_turn(response_msg)
        case _:
            raise_exhaustive_enum_error(data_strategy)

    return chat_formatter.messages


def get_thinking_data(task_run: TaskRun) -> str:
    """
    Raises an error if thinking data is not present.
    """
    thinking = task_run.thinking_training_data()
    if thinking is None:
        raise ValueError(
            "Thinking data is required when fine-tuning thinking models. Please ensure your fine-tuning dataset contains reasoning or chain of thought output for every entry."
        )

    return thinking


def serialize_r1_style_message(thinking: str | None, final_output: str):
    if thinking is None or len(thinking.strip()) == 0:
        raise ValueError(
            "Thinking data is required when fine-tuning thinking models (R1, QwQ, etc). Please ensure your fine-tuning dataset contains reasoning or chain of thought output for every entry."
        )

    return f"<think>\n{thinking}\n</think>\n\n{final_output}"


def generate_chat_message_list(
    training_chat: list[ChatMessage],
) -> list[dict[str, Any]]:
    """Generate OpenAI chat list. Not the full OpenAI body, just the list of messages."""

    messages: list[dict[str, Any]] = []

    for msg in training_chat:
        # if msg is BasicChatMessage, output as is.
        if isinstance(msg, BasicChatMessage):
            messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )
        # Assistant message with tool call
        elif isinstance(msg, ToolCallMessage):
            messages.append(
                {
                    "role": msg.role,
                    "content": msg.content
                    if msg.content is not None
                    else "",  # fireworks is more forgiving of empty strings than "null" in jsonl
                    "tool_calls": msg.tool_calls,
                }
            )
        # Tool call response message
        elif isinstance(msg, ToolResponseMessage):
            messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                }
            )
        else:
            # Should never happen
            raise ValueError(f"Unsupported message type: {type(msg)}")

    return messages


def generate_chat_message_response(
    training_chat: list[ChatMessage],
    tools: list[ToolCallDefinition] | None = None,
) -> Dict[str, Any]:
    """Generate OpenAI chat format with plaintext response"""

    messages: list[dict[str, Any]] = generate_chat_message_list(training_chat)

    result: Dict[str, Any] = {"messages": messages}

    if tools:
        result["tools"] = tools

    return result


def last_message_structured_content(training_chat: list[ChatMessage]) -> Dict:
    """Get the structured content of the last message"""
    if len(training_chat) < 1:
        raise ValueError("Training chat is empty")
    try:
        json_data = json.loads(training_chat[-1].content or "")
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Last message is not JSON (structured), and this format expects structured data: {e}"
        )
    if not isinstance(json_data, dict):
        raise ValueError(
            "Last message is not a JSON Dictionary (structured data), and this format expects structured_data."
        )
    return json_data


def generate_json_schema_message(
    training_chat: list[ChatMessage],
    tools: list[ToolCallDefinition] | None = None,
) -> Dict[str, Any]:
    """Generate OpenAI chat format with validated JSON response"""
    # Load and dump to ensure it's valid JSON and goes to 1 line
    last_msg_data = last_message_structured_content(training_chat)

    # re-format the json string in the last message for consistency
    json_string = json.dumps(last_msg_data, ensure_ascii=False)
    training_chat[-1].content = json_string

    return generate_chat_message_response(training_chat, tools=tools)


def generate_chat_message_toolcall(
    training_chat: list[ChatMessage],
    tools: list[ToolCallDefinition] | None = None,
) -> Dict[str, Any]:
    """Generate OpenAI chat format with tool call response"""
    last_message_data = last_message_structured_content(training_chat)

    messages: list[dict[str, Any]] = generate_chat_message_list(training_chat)

    # remove the last message, we're going to replace it with a toolcall
    messages = messages[:-1]

    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "task_response",
                        "arguments": json.dumps(last_message_data, ensure_ascii=False),
                    },
                }
            ],
        },
    )

    result: Dict[str, Any] = {"messages": messages}

    if tools:
        result["tools"] = tools

    return result


def generate_huggingface_chat_template(
    training_chat: list[ChatMessage],
    tools: list[ToolCallDefinition] | None = None,
) -> Dict[str, Any]:
    """Generate HuggingFace chat template"""

    conversations: list[dict[str, Any]] = generate_chat_message_list(training_chat)

    return {"conversations": conversations}


def generate_huggingface_chat_template_toolcall(
    training_chat: list[ChatMessage],
    tools: list[ToolCallDefinition] | None = None,
) -> Dict[str, Any]:
    """Generate HuggingFace chat template with tool calls"""
    last_message_data = last_message_structured_content(training_chat)

    # See https://huggingface.co/docs/transformers/en/chat_templating
    conversations: list[dict[str, Any]] = generate_chat_message_list(training_chat)

    # remove the last message, we're going to replace it with a toolcall
    conversations = conversations[:-1]

    conversations.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "task_response",
                        "id": str(uuid4()).replace("-", "")[:9],
                        "arguments": last_message_data,
                    },
                }
            ],
        },
    )

    return {"conversations": conversations}


FORMAT_GENERATORS: Dict[DatasetFormat, FormatGenerator] = {
    DatasetFormat.OPENAI_CHAT_JSONL: generate_chat_message_response,
    DatasetFormat.OPENAI_CHAT_JSON_SCHEMA_JSONL: generate_json_schema_message,
    DatasetFormat.OPENAI_CHAT_TOOLCALL_JSONL: generate_chat_message_toolcall,
    DatasetFormat.HUGGINGFACE_CHAT_TEMPLATE_JSONL: generate_huggingface_chat_template,
    DatasetFormat.HUGGINGFACE_CHAT_TEMPLATE_TOOLCALL_JSONL: generate_huggingface_chat_template_toolcall,
    DatasetFormat.VERTEX_GEMINI: generate_vertex_gemini,
}


class DatasetFormatter:
    """Handles formatting of datasets into various output formats"""

    def __init__(
        self,
        dataset: DatasetSplit,
        system_message: str,
        thinking_instructions: str | None = None,
    ):
        self.dataset = dataset
        self.system_message = system_message
        self.thinking_instructions = thinking_instructions
        self._tool_cache: dict[str, ToolCallDefinition] = {}

        task = dataset.parent_task()
        if task is None:
            raise ValueError("Dataset has no parent task")
        self.task = task

    async def dump_to_file(
        self,
        split_name: str,
        format_type: DatasetFormat,
        data_strategy: ChatStrategy,
        path: Path | None = None,
    ) -> Path:
        """
        Format the dataset into the specified format.

        Args:
            split_name: Name of the split to dump
            format_type: Format to generate the dataset in
            path: Optional path to write to. If None, writes to temp directory

        Returns:
            Path to the generated file

        Note:
            The output is written in UTF-8 encoding with ensure_ascii=False to properly
            support international text content while maintaining readability.
        """
        if format_type not in FORMAT_GENERATORS:
            raise ValueError(f"Unsupported format: {format_type}")
        if split_name not in self.dataset.split_contents:
            raise ValueError(f"Split {split_name} not found in dataset")

        generator = FORMAT_GENERATORS[format_type]

        include_cot = data_strategy in THINKING_DATA_STRATEGIES

        # Write to a temp file if no path is provided
        output_path = (
            path
            or Path(tempfile.gettempdir())
            / f"{self.dataset.name} -- split-{split_name} -- format-{format_type.value} -- {'cot' if include_cot else 'no-cot'}.jsonl"
        )

        # Include intermediate runs: a split snapshots run IDs at creation
        # time (leaves-only), but the user can later extend a multiturn
        # conversation - turning a run that was a leaf when the split was
        # built into an intermediate. We still need to resolve it here so
        # the export doesn't fail with "task run not found".
        runs = self.task.runs(include_intermediate_runs=True)
        runs_by_id = {run.id: run for run in runs}

        # Generate formatted output with UTF-8 encoding
        with open(output_path, "w", encoding="utf-8") as f:
            for run_id in self.dataset.split_contents[split_name]:
                task_run = runs_by_id.get(run_id)
                if task_run is None:
                    raise ValueError(
                        f"Task run {run_id} not found. This is required by this dataset."
                    )

                # Fetch tool definitions
                tools = await self._get_tool_definitions_from_config(task_run)

                # Training_chat is a list of ChatMessage
                training_chat = build_training_chat(
                    task_run=task_run,
                    system_message=self.system_message,
                    data_strategy=data_strategy,
                    thinking_instructions=self.thinking_instructions,
                )
                example = generator(training_chat, tools=tools)

                # Allow non-ascii characters in the dataset.
                # Better readability for non-English users. If you don't support UTF-8... you should.
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        return output_path

    # Helper

    async def _get_tool_definitions_from_config(
        self,
        task_run: TaskRun,
    ) -> list[ToolCallDefinition] | None:
        """Extract tool definitions from task run config, using cache to avoid repeated MCP server calls"""
        if not task_run.output.source:
            return None

        run_config = task_run.output.source.run_config
        if run_config is None or not isinstance(
            run_config, KilnAgentRunConfigProperties
        ):
            return None

        tools_config = run_config.tools_config
        if tools_config is None:
            return None

        task = task_run.parent_task()
        if task is None:
            return None

        tool_definitions: list[ToolCallDefinition] = []
        skill_tool_ids = [
            tid for tid in tools_config.tools if tid.startswith(SKILL_TOOL_ID_PREFIX)
        ]
        non_skill_tool_ids = [
            tid
            for tid in tools_config.tools
            if not tid.startswith(SKILL_TOOL_ID_PREFIX)
        ]

        for tool_id in non_skill_tool_ids:
            if tool_id in self._tool_cache:
                tool_definitions.append(self._tool_cache[tool_id])
                continue

            try:
                tool = tool_from_id(tool_id, task)
                tool_def = await tool.toolcall_definition()

                self._tool_cache[tool_id] = tool_def
                tool_definitions.append(tool_def)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to get tool definition for tool ID '{tool_id}': {e}"
                )

        if skill_tool_ids:
            cache_key = "::".join(sorted(skill_tool_ids))
            if cache_key in self._tool_cache:
                tool_definitions.append(self._tool_cache[cache_key])
            else:
                from kiln_ai.adapters.adapter_registry import (
                    load_skills_from_tool_ids,
                )
                from kiln_ai.tools.skill_tool import SkillTool

                skills_dict = load_skills_from_tool_ids(task, skill_tool_ids)
                if skills_dict:
                    skill_tool = SkillTool(
                        f"{SKILL_TOOL_ID_PREFIX}_combined",
                        list(skills_dict.values()),
                    )
                    skill_def = await skill_tool.toolcall_definition()
                    self._tool_cache[cache_key] = skill_def
                    tool_definitions.append(skill_def)

        return tool_definitions if tool_definitions else None
