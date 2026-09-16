import json
import logging
from dataclasses import dataclass

from kiln_ai.utils.open_ai_types import (
    TASK_RESPONSE_TOOL_NAME,
    ChatCompletionMessageParam,
)
from openai.types.chat import ChatCompletionMessageToolCallParam

logger = logging.getLogger(__name__)


class EvalTraceFormatter:
    @dataclass
    class MessageDetails:
        role: str
        reasoning_content: str | None
        tool_calls: str | None
        content: str | None
        # The structured answer the model returned through the internal
        # task_response tool. Kept apart from tool_calls so it reads as the
        # answer rather than as a tool the model chose to call.
        structured_output: str | None

    @staticmethod
    def trace_to_formatted_conversation_history(
        trace: list[ChatCompletionMessageParam],
    ) -> str:
        """Convert a trace of chat completion messages to a formatted conversation history string.

        One message can carry several things at once — a model commonly narrates
        while it calls a tool — so each is emitted as its own block. Collecting
        blocks rather than assigning to shared variables is what keeps them: an
        earlier design assigned and emitted once per message, so the last branch
        taken silently replaced the others and a tool call accompanied by a
        sentence never reached the judge.
        """
        blocks: list[str] = []
        for message in trace:
            message_details = EvalTraceFormatter.message_details_from_message(message)
            role = message_details.role

            if role == "tool" and message_details.content:
                origin_tool_call_name = (
                    EvalTraceFormatter.origin_tool_call_name_from_message(
                        message, trace
                    )
                )
                # Named where the name is known, and still emitted where it is
                # not: an unresolvable origin means we cannot say which tool
                # answered, not that nothing did. Dropping the result would hide
                # the value itself, which is the part a judge needs.
                role_label = (
                    f"tool result from {origin_tool_call_name}"
                    if origin_tool_call_name
                    else "tool result"
                )
                blocks.append(
                    EvalTraceFormatter.format_message(
                        role_label,
                        f"{role}_tool_message",
                        message_details.content,
                    )
                )
                continue

            if message_details.reasoning_content:
                blocks.append(
                    EvalTraceFormatter.format_message(
                        f"{role} reasoning",
                        f"{role}_reasoning_message",
                        message_details.reasoning_content,
                    )
                )

            # Content before tool calls: the text in such a message is the
            # narration introducing the call ("Now I'll subtract that fee"), so
            # emitting the call first would read backwards.
            if message_details.content:
                blocks.append(
                    EvalTraceFormatter.format_message(
                        role, f"{role}_message", message_details.content
                    )
                )

            # Emitted under the same tag a plain message uses, because that is
            # what it is. A new tag would re-expose the plumbing under a
            # different name, and every judge prompt already knows what an
            # <assistant_message> is.
            if message_details.structured_output:
                blocks.append(
                    EvalTraceFormatter.format_message(
                        role,
                        f"{role}_message",
                        message_details.structured_output,
                    )
                )

            if message_details.tool_calls:
                blocks.append(
                    EvalTraceFormatter.format_message(
                        f"{role} requested tool calls",
                        f"{role}_requested_tool_calls",
                        message_details.tool_calls,
                    )
                )

        # Joined on what was emitted, not on position in the trace: a message
        # that renders to nothing must not leave a gap behind it.
        return "\n\n".join(blocks)

    @staticmethod
    def format_message(role_label: str, tag: str, content: str) -> str:
        return f"{role_label}:\n<{tag}>\n{content}\n</{tag}>"

    @staticmethod
    def message_details_from_message(
        message: ChatCompletionMessageParam,
    ) -> MessageDetails:
        return EvalTraceFormatter.MessageDetails(
            role=EvalTraceFormatter.role_from_message(message),
            reasoning_content=EvalTraceFormatter.reasoning_content_from_message(
                message
            ),
            tool_calls=EvalTraceFormatter.formatted_tool_calls_from_message(message),
            content=EvalTraceFormatter.content_from_message(message),
            structured_output=EvalTraceFormatter.structured_output_from_message(
                message
            ),
        )

    @staticmethod
    def role_from_message(message: ChatCompletionMessageParam) -> str:
        return message["role"]

    @staticmethod
    def content_from_message(message: ChatCompletionMessageParam) -> str | None:
        """Get the content of a message."""
        if (
            "content" not in message
            or message["content"] is None
            or not isinstance(message["content"], str)
        ):
            return None

        # For Kiln task tools, extract just the output field from the JSON response
        if message["role"] == "tool":
            try:
                parsed = json.loads(message["content"])
                if parsed and isinstance(parsed, dict) and "output" in parsed:
                    return parsed["output"]
            except Exception:
                # Content is not JSON, we will return as-is
                pass

        return message["content"]

    @staticmethod
    def reasoning_content_from_message(
        message: ChatCompletionMessageParam,
    ) -> str | None:
        if (
            "reasoning_content" not in message
            or message["reasoning_content"] is None
            or not isinstance(message["reasoning_content"], str)
        ):
            return None

        return message["reasoning_content"]

    @staticmethod
    def tool_calls_from_message(
        message: ChatCompletionMessageParam,
    ) -> list[ChatCompletionMessageToolCallParam] | None:
        tool_calls = message.get("tool_calls")
        return tool_calls if tool_calls else None

    @staticmethod
    def formatted_tool_calls_from_message(
        message: ChatCompletionMessageParam,
    ) -> str | None:
        tool_calls = EvalTraceFormatter.tool_calls_from_message(message)
        if tool_calls is None:
            return None

        # The task_response wrapper is not a tool the model chose to call, so it
        # is reported as the model's answer instead. See
        # structured_output_from_message.
        #
        # Blank line between calls: concatenating them ran the next call's name
        # onto the end of the previous call's arguments, so a message issuing
        # several calls read as one run-on block.
        described = "\n\n".join(
            f"- Tool Name: {tool_call['function']['name']}\n"
            f"- Arguments: {tool_call['function']['arguments']}"
            for tool_call in tool_calls
            if tool_call["function"]["name"] != TASK_RESPONSE_TOOL_NAME
        )
        # None rather than "" so a message whose only call was the wrapper
        # emits no tool-call block at all.
        return described or None

    @staticmethod
    def structured_output_from_message(
        message: ChatCompletionMessageParam,
    ) -> str | None:
        """The structured answer a model returned via the internal task_response tool.

        Function-calling structured output modes carry the answer as the
        arguments of a synthetic task_response call. Returns those arguments so
        the trace can show them as the answer rather than as tool use.
        """
        tool_calls = EvalTraceFormatter.tool_calls_from_message(message)
        if tool_calls is None:
            return None

        arguments = None
        for tool_call in tool_calls:
            if tool_call["function"]["name"] == TASK_RESPONSE_TOOL_NAME:
                # Last one wins, matching the adapter: when a model emits more
                # than one task_response, the final call is the output the run
                # was saved with.
                arguments = tool_call["function"]["arguments"]
        return arguments

    @staticmethod
    def origin_tool_call_name_from_message(
        message: ChatCompletionMessageParam,
        trace: list[ChatCompletionMessageParam],
    ) -> str | None:
        tool_call_id = message.get("tool_call_id")
        if not tool_call_id:
            return None
        for msg in trace:
            tool_calls = EvalTraceFormatter.tool_calls_from_message(msg)
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call["id"] == tool_call_id:
                        return tool_call["function"]["name"]
        logger.error(
            f"Origin tool call name not found for tool_call_id: {tool_call_id}"
        )
        return None
