from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, TypeAlias, Union

from litellm.types.utils import Message as LiteLLMMessage

from kiln_ai.datamodel.datamodel_enums import ChatStrategy, InputType
from kiln_ai.utils.exhaustive_error import raise_exhaustive_enum_error
from kiln_ai.utils.open_ai_types import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
)

COT_FINAL_ANSWER_PROMPT = "Considering the above, return a final result."


ChatCompletionMessageIncludingLiteLLM: TypeAlias = Union[
    ChatCompletionMessageParam, LiteLLMMessage
]


@dataclass
class BasicChatMessage:
    role: Literal["system", "assistant", "user"]
    content: Optional[str]


@dataclass
class ToolCallMessage:
    """Assistant message with tool calls for chat formatting"""

    role: Literal["assistant"]
    tool_calls: List[ChatCompletionMessageToolCallParam]
    content: Optional[str] = None


@dataclass
class ToolResponseMessage:
    """Tool response message for chat formatting"""

    role: Literal["tool"]
    content: str
    tool_call_id: str
    is_error: Optional[bool] = None
    error_message: Optional[str] = None
    kiln_task_tool_data: Optional[str] = None


ChatMessage = Union[
    BasicChatMessage,
    ToolCallMessage,
    ToolResponseMessage,
]


def chat_message_to_dict(message: ChatMessage) -> dict:
    """Convert a ChatMessage dataclass to an OpenAI-shaped message dict, including
    Kiln-specific fields (``is_error``, ``error_message``, ``kiln_task_tool_data``)
    when present on tool responses."""
    msg_dict: dict = {"role": message.role, "content": message.content}
    if isinstance(message, ToolCallMessage):
        msg_dict["tool_calls"] = message.tool_calls
    elif isinstance(message, ToolResponseMessage):
        msg_dict["tool_call_id"] = message.tool_call_id
        if message.is_error is not None:
            msg_dict["is_error"] = message.is_error
        if message.error_message is not None:
            msg_dict["error_message"] = message.error_message
        if message.kiln_task_tool_data is not None:
            msg_dict["kiln_task_tool_data"] = message.kiln_task_tool_data
    return msg_dict


@dataclass
class ChatTurn:
    """
    All data needed to send a chat turn to the model.
    """

    messages: Sequence[ChatMessage]
    final_call: bool


class ChatFormatter(ABC):
    def __init__(
        self,
        system_message: str,
        user_input: InputType,
        thinking_instructions: str | None = None,
    ) -> None:
        self.system_message = system_message
        self.user_input = user_input
        self.thinking_instructions = thinking_instructions
        self._messages: List[ChatMessage] = []
        self._state = "start"
        self._intermediate_outputs: Dict[str, str] = {}

    @property
    def messages(self) -> List[ChatMessage]:
        return list(self._messages)

    def append_messages(self, messages: Sequence[ChatMessage]) -> None:
        """Append messages to the internal messages list."""
        self._messages.extend(messages)

    def message_dicts(self) -> List[dict]:
        result = []
        for m in self._messages:
            msg_dict = {"role": m.role, "content": m.content}
            if isinstance(m, ToolCallMessage):
                msg_dict["tool_calls"] = m.tool_calls
            elif isinstance(m, ToolResponseMessage):
                msg_dict["tool_call_id"] = m.tool_call_id
            result.append(msg_dict)
        return result

    def intermediate_outputs(self) -> Dict[str, str]:
        """Get the intermediate outputs from the chat formatter."""
        return self._intermediate_outputs

    def initial_messages(self) -> list[ChatCompletionMessageIncludingLiteLLM]:
        """Messages to seed the conversation. Empty for fresh runs; prior trace for continuation."""
        return []

    @abstractmethod
    def next_turn(self, previous_output: str | None = None) -> Optional[ChatTurn]:
        """Advance the conversation and return the next messages if any."""
        raise NotImplementedError


class SingleTurnFormatter(ChatFormatter):
    def next_turn(self, previous_output: str | None = None) -> Optional[ChatTurn]:
        if self._state == "start":
            msgs = [
                BasicChatMessage("system", self.system_message),
                BasicChatMessage("user", format_user_message(self.user_input)),
            ]
            self._state = "awaiting_final"
            self._messages.extend(msgs)
            return ChatTurn(messages=msgs, final_call=True)

        if self._state == "awaiting_final":
            if previous_output is None:
                raise ValueError("previous_output required for final step")
            self._messages.append(BasicChatMessage("assistant", previous_output))
            self._state = "done"
            return None

        return None


class TwoMessageCotLegacyFormatter(ChatFormatter):
    def __init__(
        self,
        system_message: str,
        user_input: InputType,
        thinking_instructions: str | None,
    ) -> None:
        super().__init__(system_message, user_input, thinking_instructions)
        if self.thinking_instructions is None:
            raise ValueError(
                "thinking_instructions are required when strategy is final_and_intermediate"
            )

    def next_turn(self, previous_output: str | None = None) -> Optional[ChatTurn]:
        if self._state == "start":
            msgs = [
                BasicChatMessage("system", self.system_message),
                BasicChatMessage("user", format_user_message(self.user_input)),
                BasicChatMessage("system", self.thinking_instructions),
            ]
            self._state = "awaiting_thinking"
            self._messages.extend(msgs)
            return ChatTurn(messages=msgs, final_call=False)

        if self._state == "awaiting_thinking":
            if previous_output is None:
                raise ValueError("previous_output required for thinking step")
            self._intermediate_outputs["chain_of_thought"] = previous_output
            self._state = "awaiting_final"
            cot_message = BasicChatMessage("user", COT_FINAL_ANSWER_PROMPT)
            self._messages.append(BasicChatMessage("assistant", previous_output))
            self._messages.append(cot_message)
            return ChatTurn(messages=[cot_message], final_call=True)

        if self._state == "awaiting_final":
            if previous_output is None:
                raise ValueError("previous_output required for final step")
            self._messages.append(BasicChatMessage("assistant", previous_output))
            self._state = "done"
            return None

        return None


class TwoMessageCotFormatter(ChatFormatter):
    def __init__(
        self,
        system_message: str,
        user_input: InputType,
        thinking_instructions: str | None,
    ) -> None:
        super().__init__(system_message, user_input, thinking_instructions)
        if self.thinking_instructions is None:
            raise ValueError(
                "thinking_instructions are required when strategy is final_and_intermediate"
            )

    def next_turn(self, previous_output: str | None = None) -> Optional[ChatTurn]:
        if self._state == "start":
            # User message combines the input and the thinking instructions
            formatted_user_message = format_user_message(self.user_input)

            # If the input contains conversation_history, it's a full_trace evaluation description and formatted_user_message contains more than a single turn of user_input and shouldn't be wrapped in <user_input> tags to avoid confusing the judge models.
            if "<conversation_history>" in formatted_user_message:
                user_message = (
                    f"{formatted_user_message}\n\n{self.thinking_instructions}"
                )
            else:
                user_message = f"The input is:\n<user_input>\n{formatted_user_message}\n</user_input>\n\n{self.thinking_instructions}"

            msgs = [
                BasicChatMessage("system", self.system_message),
                BasicChatMessage("user", user_message),
            ]
            self._state = "awaiting_thinking"
            self._messages.extend(msgs)
            return ChatTurn(messages=msgs, final_call=False)

        if self._state == "awaiting_thinking":
            if previous_output is None:
                raise ValueError("previous_output required for thinking step")
            self._intermediate_outputs["chain_of_thought"] = previous_output
            self._state = "awaiting_final"
            self._messages.append(BasicChatMessage("assistant", previous_output))
            cot_message = BasicChatMessage("user", COT_FINAL_ANSWER_PROMPT)
            self._messages.append(cot_message)
            return ChatTurn(messages=[cot_message], final_call=True)

        if self._state == "awaiting_final":
            if previous_output is None:
                raise ValueError("previous_output required for final step")
            self._messages.append(BasicChatMessage("assistant", previous_output))
            self._state = "done"
            return None

        return None


class SingleTurnR1ThinkingFormatter(ChatFormatter):
    """Formatter for reasoning ("R1-style") models, which emit thinking
    natively: the prompt is a plain single turn with no thinking instructions.
    """

    def next_turn(self, previous_output: str | None = None) -> Optional[ChatTurn]:
        if self._state == "start":
            msgs = [
                BasicChatMessage("system", self.system_message),
                BasicChatMessage("user", format_user_message(self.user_input)),
            ]
            self._state = "awaiting_final"
            self._messages.extend(msgs)
            return ChatTurn(messages=msgs, final_call=True)

        if self._state == "awaiting_final":
            if previous_output is None:
                raise ValueError("previous_output required for final step")
            self._messages.append(BasicChatMessage("assistant", previous_output))
            self._state = "done"
            return None

        return None


class MultiturnFormatter(ChatFormatter):
    """
    Formatter for continuing a multi-turn conversation with prior trace.
    Takes prior_trace (existing conversation) and appends the new user message.
    Produces a single turn: the new user message. Tool calls and multi-turn
    model responses are handled by _run_model_turn's internal loop.

    When user_input is a dict or list with tool_call_id keys, the input is
    treated as tool call results (role "tool") rather than a user message.
    This supports resuming after a return_on_tool_call interrupt.
    """

    def __init__(
        self,
        prior_trace: list[ChatCompletionMessageParam],
        user_input: InputType,
    ) -> None:
        super().__init__(
            system_message="",
            user_input=user_input,
            thinking_instructions=None,
        )
        self._prior_trace = prior_trace

    def initial_messages(self) -> list[ChatCompletionMessageIncludingLiteLLM]:
        """Messages to seed the conversation (prior trace)."""
        return list(self._prior_trace)

    @property
    def _is_tool_result(self) -> bool:
        """Return True if user_input looks like one or more tool call results."""
        input = self.user_input
        if isinstance(input, dict):
            return "tool_call_id" in input
        if isinstance(input, list):
            return bool(input) and all(
                isinstance(item, dict) and "tool_call_id" in item for item in input
            )
        return False

    def next_turn(self, previous_output: str | None = None) -> Optional[ChatTurn]:
        if self._state == "start":
            self._state = "awaiting_final"
            if self._is_tool_result:
                if isinstance(self.user_input, dict):
                    raw_items: list[dict] = [self.user_input]
                else:
                    raw_items = list(self.user_input)  # type: ignore[arg-type]
                msgs: list[ChatMessage] = [
                    ToolResponseMessage(
                        role="tool",
                        content=str(item.get("content", "")),
                        tool_call_id=item["tool_call_id"],
                        is_error=item.get("is_error"),
                        error_message=item.get("error_message"),
                        kiln_task_tool_data=item.get("kiln_task_tool_data"),
                    )
                    for item in raw_items
                ]
                self._messages.extend(msgs)
                return ChatTurn(messages=msgs, final_call=True)
            else:
                # prior trace is already in the messages list and contains system and so on, we only need
                # to append the latest new user message
                user_msg = BasicChatMessage(
                    "user", format_user_message(self.user_input)
                )
                self._messages.append(user_msg)
                return ChatTurn(messages=[user_msg], final_call=True)

        if self._state == "awaiting_final":
            if previous_output is None:
                raise ValueError("previous_output required for final step")
            self._messages.append(BasicChatMessage("assistant", previous_output))
            self._state = "done"
            return None

        return None


def get_chat_formatter(
    strategy: ChatStrategy,
    system_message: str,
    user_input: InputType,
    thinking_instructions: str | None = None,
) -> ChatFormatter:
    match strategy:
        case ChatStrategy.single_turn:
            return SingleTurnFormatter(system_message, user_input)
        case ChatStrategy.two_message_cot_legacy:
            return TwoMessageCotLegacyFormatter(
                system_message, user_input, thinking_instructions
            )
        case ChatStrategy.two_message_cot:
            return TwoMessageCotFormatter(
                system_message, user_input, thinking_instructions
            )
        case ChatStrategy.single_turn_r1_thinking:
            return SingleTurnR1ThinkingFormatter(system_message, user_input)
        case _:
            raise_exhaustive_enum_error(strategy)


def chat_strategy_for_run(
    cot_prompt: str | None,
    tuned_chat_strategy: ChatStrategy | None,
    reasoning_capable: bool,
) -> ChatStrategy:
    """The chat strategy a run resolves to, from the prompt it uses and the model it
    runs on.

    Separate from formatter construction so callers that only need the *shape* of the
    conversation (how many messages a turn costs, whether thinking is its own call) can
    ask without building a formatter — and so there is only one copy of the branching to
    keep correct.
    """
    # Nothing to separate without thinking instructions, so one message per turn. True
    # even when a tuned strategy is set: those are either single turn already, or need
    # the thinking instructions this run doesn't have.
    if not cot_prompt:
        return ChatStrategy.single_turn

    # Some models (finetunes) are trained against a specific strategy, so honour it.
    # Except single turn: the user picked a prompt with thinking instructions, and
    # explicit prompt selection wins over the tuned default.
    if tuned_chat_strategy and tuned_chat_strategy != ChatStrategy.single_turn:
        return tuned_chat_strategy

    # A reasoning model emits its thinking in a structured format of its own, so one
    # call carries both the thinking and the answer. Any other model needs a second
    # call to separate them.
    if reasoning_capable:
        return ChatStrategy.single_turn_r1_thinking
    return ChatStrategy.two_message_cot


def is_two_message_cot_strategy(strategy: ChatStrategy) -> bool:
    """Whether one turn of this strategy costs two user-role messages.

    These strategies ask the model to think, then inject a second user message asking
    for the final answer. A turn is therefore not "one user message, one assistant
    reply", which anything counting turns in a trace has to know.
    """
    return strategy in (
        ChatStrategy.two_message_cot,
        ChatStrategy.two_message_cot_legacy,
    )


def format_user_message(input: InputType) -> str:
    """Build a user message from the input.

    Args:
        input (Union[Dict, str]): The input to format into a message.

    Returns:
        str: The formatted user message.
    """
    if not isinstance(input, str):
        return json.dumps(input, ensure_ascii=False)

    return input
