"""Per-turn synthetic-user driver.

Wraps a kiln_ai LiteLLM adapter and exposes a single async `respond()` that:
1. Filters the eval-frame conversation to `visible_message_roles`.
2. Role-swaps user/assistant so the LLM is generating the SU's reply.
3. Calls the adapter with the persona system prompt prepended as
   `prior_trace` and the latest swapped user turn as `input`.

The driver does NOT persist `TaskRun`s — it builds in-memory runs only.
The eval-dataset chain consists only of *target* TaskRuns.
"""

from kiln_ai.adapters.adapter_registry import adapter_for_task
from kiln_ai.datamodel.datamodel_enums import StructuredOutputMode
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties, ToolsRunConfig
from kiln_ai.datamodel.task import Task
from kiln_ai.datamodel.usage import Usage
from kiln_ai.synthetic_user.models import (
    SyntheticUserDriverConfig,
    SyntheticUserInfo,
)
from kiln_ai.synthetic_user.prompt import render_system_prompt
from kiln_ai.synthetic_user.role_swap import role_swap
from kiln_ai.utils.open_ai_types import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)


def _is_tool_dispatch_only(msg: ChatCompletionMessageParam) -> bool:
    """True if `msg` is an assistant turn with no user-facing text (i.e.,
    a pure tool-call dispatch). The SU LLM shouldn't see these — they're
    actions the target took, not speech the SU is reacting to. Content is
    checked falsy, not `is None`: the adapter emits dispatch turns with
    content `''` as well as None, and either would role-swap into a blank
    user message. Assistant turns that carry text alongside tool_calls are
    NOT filtered out (the text is the user-facing part).
    """
    return msg["role"] == "assistant" and not msg.get("content")


class SyntheticUserDriver:
    """Plays one synthetic user across multiple turns.

    Constructed once per case from the typed persona + driver config;
    `respond(conversation)` is called once per turn. Callers holding the
    tagged wire blob parse it first (kiln_ai.synthetic_user.parser) — blob
    parsing stays at the wire boundary. The adapter is built at
    construction time and reused across all turns of the case.
    """

    def __init__(
        self,
        synthetic_user_info: SyntheticUserInfo,
        driver_config: SyntheticUserDriverConfig,
    ):
        self._info = synthetic_user_info
        self._driver_config = driver_config
        self._system_prompt: str = render_system_prompt(self._info)

        # In-memory Task; nothing is persisted. The persona-playing system
        # prompt rides on `prior_trace[0]` each call, so this `instruction`
        # is effectively unused at runtime — kiln_ai's MultiturnFormatter
        # uses the first system message in `prior_trace` and skips the
        # task's instruction when `prior_trace` is non-empty. The Task
        # model requires a non-empty instruction, hence the placeholder.
        self._task = Task(
            name="synthetic_user_driver",
            description="In-memory SU player. Not persisted.",
            instruction=(
                "Placeholder — the persona-playing system prompt is supplied "
                "via prior_trace on every adapter call."
            ),
        )
        # Same RunConfigProperties shape used elsewhere; `structured_output_mode`
        # is `default` because SU output is free text (no JSON schema).
        self._run_config = KilnAgentRunConfigProperties(
            model_name=driver_config.model_name,
            model_provider_name=driver_config.model_provider_name,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.default,
            tools_config=ToolsRunConfig(tools=[]),
        )
        self._adapter = adapter_for_task(self._task, self._run_config)

    async def respond(
        self, conversation: list[ChatCompletionMessageParam]
    ) -> tuple[str, Usage | None]:
        """Return the SU's next message and the driver model's usage for the call.

        `conversation` is in the eval frame and must end on an `assistant`
        (target) turn. Drive-loop termination is the caller's concern.

        The whole `Usage` rather than just its cost: the SU's TaskRun is never
        persisted, so this in-memory value is the only place the driver model's
        tokens ever exist. A cost alone can neither be split per model against an
        invoice nor recomputed at a different price, and `cost / total_tokens`
        over a figure whose tokens are the agent's is meaningless.

        None when the provider reported nothing — distinct from a zeroed Usage,
        which would read as a genuinely free call rather than an unmeasured one.
        """
        # 1) Filter to visible roles (drop system/tool if present).
        visible = [
            m
            for m in conversation
            if m["role"] in self._driver_config.visible_message_roles
        ]
        # 2) Drop tool-dispatch-only assistant turns (falsy content).
        #    See _is_tool_dispatch_only for rationale.
        visible = [m for m in visible if not _is_tool_dispatch_only(m)]
        # 3) Invariants.
        if not visible:
            raise ValueError("No LLM-visible messages in conversation.")
        if visible[-1]["role"] != "assistant":
            raise ValueError(
                "Conversation must end on an assistant (target) turn — the SU "
                "is responding to that turn."
            )

        # 4) Role-swap then assemble prior_trace + input. The last swapped
        #    turn becomes the LLM `input`; everything before it goes into
        #    `prior_trace` with a system message prepended.
        swapped = role_swap(visible)
        last = swapped[-1]
        user_input = last["content"]
        if not isinstance(user_input, str):
            raise RuntimeError(
                "synthetic user input must be a plain string after role_swap"
            )

        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": self._system_prompt,
        }
        prior_trace: list[ChatCompletionMessageParam] = [system_msg, *swapped[:-1]]

        # 5) Adapter call (in-memory, no disk write).
        task_run, run_output = await self._adapter.invoke_returning_run_output(
            user_input, prior_trace=prior_trace
        )
        raw = run_output.output
        if not isinstance(raw, str):
            raise RuntimeError("synthetic user returned non-string output")

        return raw, task_run.usage
