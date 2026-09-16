"""Role-swap eval-frame conversation roles into LLM-frame roles.

Chat models are trained to generate `assistant`-labeled responses, so we
flip the eval-frame labels before the call: the LLM's "assistant" turn IS
the synthetic user.

    Eval frame:                       LLM frame (post-swap):
      user      = synthetic user        assistant = synthetic user
      assistant = target agent          user      = target agent

The driver filters `visible_message_roles` upstream, so a system or tool
turn reaching here is an internal invariant violation — fail loud rather
than silently drop.
"""

from kiln_ai.utils.open_ai_types import (
    ChatCompletionAssistantMessageParamWrapper,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)


def role_swap(
    conversation: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    """Flip eval-frame user/assistant labels into LLM-frame labels.

    Only `user` and `assistant` are handled. The driver is expected to
    have filtered out other roles before calling this.
    """
    result: list[ChatCompletionMessageParam] = []
    for msg in conversation:
        role = msg["role"]
        if role not in ("user", "assistant"):
            raise ValueError(
                f"role_swap received unsupported role {role!r}; "
                "the driver should have filtered it"
            )
        # The TypedDict union allows non-string content for multimodal /
        # tool turns, but the synthetic user only ever sees plain text from
        # the target. Narrowing here lets us assign into the swapped wrapper
        # type without a cast.
        content = msg["content"]
        if not isinstance(content, str):
            raise ValueError(
                f"role_swap requires string content for role {role!r}; "
                f"got {type(content).__name__}"
            )
        if role == "user":
            assistant_msg: ChatCompletionAssistantMessageParamWrapper = {
                "role": "assistant",
                "content": content,
            }
            result.append(assistant_msg)
        else:  # role == "assistant"
            user_msg: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": content,
            }
            result.append(user_msg)
    return result
