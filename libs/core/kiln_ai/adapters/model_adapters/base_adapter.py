from __future__ import annotations

import json
import uuid
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator, Dict, Tuple

from litellm.types.utils import ModelResponseStream

from kiln_ai.adapters.chat.chat_formatter import (
    ChatFormatter,
    MultiturnFormatter,
    chat_strategy_for_run,
    get_chat_formatter,
)
from kiln_ai.adapters.errors import (
    KilnRunError,
    StructuredOutputParseError,
    format_error_message,
)
from kiln_ai.adapters.ml_model_list import (
    KilnModelProvider,
    StructuredOutputMode,
    default_structured_output_mode_for_model_provider,
)
from kiln_ai.adapters.model_adapters.adapter_stream import AdapterStreamResult
from kiln_ai.adapters.model_adapters.stream_events import (
    AiSdkStreamConverter,
    AiSdkStreamEvent,
    FinishEvent,
    FinishMessageMetadata,
    FinishStepEvent,
    StartEvent,
    StartStepEvent,
    ToolCallEvent,
)
from kiln_ai.adapters.parsers.json_parser import parse_json_string
from kiln_ai.adapters.parsers.parser_registry import model_parser_from_id
from kiln_ai.adapters.parsers.request_formatters import request_formatter_from_id
from kiln_ai.adapters.prompt_builders import BasePromptBuilder, prompt_builder_from_id
from kiln_ai.adapters.provider_tools import kiln_model_provider_from
from kiln_ai.adapters.run_output import RunOutput
from kiln_ai.datamodel import (
    DataSource,
    DataSourceType,
    MessageUsage,
    Task,
    TaskOutput,
    TaskRun,
    Usage,
)
from kiln_ai.datamodel.datamodel_enums import InputType
from kiln_ai.datamodel.json_schema import validate_schema_with_value_error
from kiln_ai.datamodel.run_config import (
    KilnAgentRunConfigProperties,
    as_kiln_agent_run_config,
)
from kiln_ai.datamodel.skill import Skill
from kiln_ai.datamodel.task import RunConfigProperties
from kiln_ai.datamodel.task_output import TASK_OUTPUT_SCHEMA_ERROR_PREFIX
from kiln_ai.datamodel.tool_id import SKILL_TOOL_ID_PREFIX, skill_id_from_tool_id
from kiln_ai.tools import KilnToolInterface
from kiln_ai.tools.mcp_session_manager import mcp_session_scope
from kiln_ai.tools.skill_tool import SkillTool
from kiln_ai.tools.tool_registry import tool_from_id
from kiln_ai.utils.config import Config
from kiln_ai.utils.exhaustive_error import raise_exhaustive_enum_error
from kiln_ai.utils.jinja_engine import render_input_transform
from kiln_ai.utils.open_ai_types import ChatCompletionMessageParam

if TYPE_CHECKING:
    from kiln_ai.adapters.model_adapters.adapter_stream import AdapterStream

SkillsDict = Dict[str, Skill]


@dataclass
class AdapterConfig:
    """
    An adapter config is config options that do NOT impact the output of the model.

    For example: if it's saved, of if we request additional data like logprobs.
    """

    allow_saving: bool = True
    top_logprobs: int | None = None
    default_tags: list[str] | None = None

    """
    The ID of the TaskRunConfig that originated this run, if any. Stored on the
    resulting TaskRun so the run can be traced back to its originating saved config
    (in addition to the inline run_config snapshot). None for ad-hoc/inline runs
    that were not initiated from a saved TaskRunConfig.
    """
    task_run_config_id: str | None = None

    """
    A custom prompt builder can be injected to override the system prompt building process.
    If not provided, the prompt builder will be created from the run_config.prompt_id which
    may load additional files from disk.
    """
    prompt_builder: BasePromptBuilder | None = None

    """
    Pre-loaded skills keyed by skill ID. When the run config references skills,
    they are looked up from this dict instead of reading from the filesystem.
    Use load_skills_for_task() to build this dict.
    """
    skills: SkillsDict | None = None

    """
    When True, the adapter will stop and return control to the caller when a tool call
    is invoked, instead of processing tool calls internally. Default is False (process
    tool calls internally).
    """
    return_on_tool_call: bool = False

    """
    Extra tools provided directly by the caller, in addition to tools resolved from the
    task's tool registry. These are sent to the model together with registry tools, and
    their names must not collide with registry tool names.

    If ``return_on_tool_call`` is False (the default), the adapter executes these tools
    itself just like registry tools. If True, the adapter returns as soon as the model
    requests a tool call and the caller is responsible for running the tool and passing
    results back via ``prior_trace``.
    """
    unmanaged_tools: list[KilnToolInterface] | None = None

    """
    When True, automatically inject prompt caching hints into completion
    requests. This is a cost optimization and does not affect model output.
    """
    automatic_prompt_caching: bool = False


class BaseAdapter(metaclass=ABCMeta):
    """Base class for AI model adapters that handle task execution.

    This abstract class provides the foundation for implementing model-specific adapters
    that can process tasks with structured or unstructured inputs/outputs. It handles
    input/output validation, prompt building, and run tracking.

    Prompt building is handled internally by the adapter, which uses a prompt builder
    based on the run config. To override the prompt building behavior, pass a custom prompt
    builder to the adapter config.
    """

    def __init__(
        self,
        task: Task,
        run_config: RunConfigProperties,
        config: AdapterConfig | None = None,
    ):
        self.task = task
        self.run_config: RunConfigProperties = run_config
        self.base_adapter_config = config or AdapterConfig()

        if isinstance(run_config, KilnAgentRunConfigProperties):
            self.update_run_config_unknown_structured_output_mode()
            self.prompt_builder = (
                self.base_adapter_config.prompt_builder
                or prompt_builder_from_id(run_config.prompt_id, task)
            )
        else:
            self.prompt_builder = None
        self._model_provider: KilnModelProvider | None = None
        self._resolved_skills: list[Skill] | None = None

        self.output_schema = task.output_json_schema
        self.input_schema = task.input_json_schema

    def model_provider(self) -> KilnModelProvider:
        """
        Lazy load the model provider for this adapter.
        """
        if self._model_provider is not None:
            return self._model_provider
        run_config = as_kiln_agent_run_config(self.run_config)
        if not run_config.model_name or not run_config.model_provider_name:
            raise ValueError("model_name and model_provider_name must be provided")
        self._model_provider = kiln_model_provider_from(
            run_config.model_name, run_config.model_provider_name
        )
        if not self._model_provider:
            raise ValueError(
                f"model_provider_name {run_config.model_provider_name} not found for model {run_config.model_name}"
            )
        return self._model_provider

    @staticmethod
    def _normalize_prior_trace(
        prior_trace: list[ChatCompletionMessageParam] | None,
    ) -> list[ChatCompletionMessageParam] | None:
        if not prior_trace:
            return None
        return prior_trace

    def _reject_multiturn_with_structured_input(
        self,
        prior_trace: list[ChatCompletionMessageParam] | None,
    ) -> None:
        if prior_trace is not None and self.input_schema is not None:
            raise ValueError(
                "Cannot run multiturn execution with a task that has a structured input schema. "
                "Use an unstructured task, or call without prior_trace."
            )

    async def invoke(
        self,
        input: InputType,
        input_source: DataSource | None = None,
        prior_trace: list[ChatCompletionMessageParam] | None = None,
        parent_task_run: TaskRun | None = None,
    ) -> TaskRun:
        task_run, _ = await self.invoke_returning_run_output(
            input, input_source, prior_trace, parent_task_run
        )
        return task_run

    async def _run_returning_run_output(
        self,
        input: InputType,
        input_source: DataSource | None = None,
        prior_trace: list[ChatCompletionMessageParam] | None = None,
        parent_task_run: TaskRun | None = None,
    ) -> Tuple[TaskRun, RunOutput]:
        # Pre-run validation: these checks run before any model call, so
        # there is no trace to preserve. They stay outside the
        # exception-wrapping block and surface as plain exceptions.
        prior_trace = self._normalize_prior_trace(prior_trace)
        self._reject_multiturn_with_structured_input(prior_trace)

        if self.input_schema is not None:
            validate_schema_with_value_error(
                input,
                self.input_schema,
                "This task requires a specific input schema. While the model produced JSON, that JSON didn't meet the schema. Search 'Troubleshooting Structured Data Issues' in our docs for more information.",
                require_object=False,
            )

        # Apply input transform if configured. The original `input` is preserved
        # for TaskRun.input persistence; `model_input` is what the model sees.
        model_input = self._apply_input_transform(input)

        # Format model input for model call (we save the original input in the
        # task without formatting). This runs in the adapter but before any
        # trace is built, so it also stays outside the wrapped region.
        formatted_input = model_input
        formatter_id = self.model_provider().formatter
        if formatter_id is not None:
            formatter = request_formatter_from_id(formatter_id)
            formatted_input = formatter.format_input(model_input)

        # Allocate the trace-so-far list here so the reference survives any
        # exception thrown from inside `_run` (or the post-processing that
        # follows). `_run` must mutate this list in place (extend/append,
        # or `list[:] = ...`) and never rebind the local name.
        trace_ref: list[ChatCompletionMessageParam] = []
        try:
            run_output, usage = await self._run(
                formatted_input, trace_ref, prior_trace=prior_trace
            )

            if not run_output.is_toolcall_pending:
                # Normal completion: parse and validate output
                provider = self.model_provider()
                parser = model_parser_from_id(provider.parser)
                parsed_output = parser.parse_output(original_output=run_output)

                # validate output
                if self.output_schema is not None:
                    # Parse json to dict if we have structured output
                    if isinstance(parsed_output.output, str):
                        try:
                            parsed_output.output = parse_json_string(
                                parsed_output.output
                            )
                        except ValueError as e:
                            # Re-type (message unchanged) so retries treat bad
                            # JSON like a schema mismatch: a one-off model slip.
                            raise StructuredOutputParseError(str(e)) from e

                    if not isinstance(parsed_output.output, dict):
                        # Valid JSON of the wrong shape (often the object wrapped
                        # in a list) is the same one-off model slip as bad JSON,
                        # so it carries the retryable type too.
                        raise StructuredOutputParseError(
                            f"structured response is not a dict: {parsed_output.output}"
                        )
                    validate_schema_with_value_error(
                        parsed_output.output,
                        self.output_schema,
                        TASK_OUTPUT_SCHEMA_ERROR_PREFIX,
                    )
                else:
                    if not isinstance(parsed_output.output, str):
                        raise RuntimeError(
                            f"response is not a string for non-structured task: {parsed_output.output}"
                        )

                trace_has_toolcalls = parsed_output.trace is not None and any(
                    message.get("role", None) == "tool"
                    for message in parsed_output.trace
                )

                # Validate reasoning content is present if required.
                # Models often skip reasoning on the final turn when tools are involved, so we don't require it then.
                if (
                    provider.reasoning_capable
                    and (
                        not parsed_output.intermediate_outputs
                        or "reasoning" not in parsed_output.intermediate_outputs
                    )
                    and not (
                        provider.reasoning_optional_for_structured_output
                        and self.has_structured_output()
                    )
                    and not trace_has_toolcalls
                ):
                    raise RuntimeError(
                        "Reasoning is required for this model, but no reasoning was returned."
                    )

                run_output = parsed_output

            run = self.generate_run(
                input,
                input_source,
                run_output,
                usage,
                run_output.trace,
                parent_task_run,
            )

            # Save the run if configured to do so, and we have a path to save to
            if (
                self.base_adapter_config.allow_saving
                and Config.shared().autosave_runs
                and self.task.path is not None
            ):
                run.save_to_file()
            else:
                # Clear the ID to indicate it's not persisted
                run.id = None

            return run, run_output
        except KilnRunError:
            # Already wrapped — pass through so we don't double-wrap.
            raise
        except Exception as e:
            # Trace conversion can itself throw (e.g., a malformed partial
            # assistant message was appended to `messages` just before the
            # real failure). Never let that swallow the original exception —
            # fall back to no trace so the user still sees the real error.
            partial_trace: list[ChatCompletionMessageParam] | None = None
            if trace_ref:
                try:
                    partial_trace = self._messages_to_trace(trace_ref)
                except Exception:
                    partial_trace = None
            raise KilnRunError(
                message=format_error_message(e),
                partial_trace=partial_trace,
                original=e,
            ) from e

    async def invoke_returning_run_output(
        self,
        input: InputType,
        input_source: DataSource | None = None,
        prior_trace: list[ChatCompletionMessageParam] | None = None,
        parent_task_run: TaskRun | None = None,
    ) -> Tuple[TaskRun, RunOutput]:
        async with mcp_session_scope():
            return await self._run_returning_run_output(
                input, input_source, prior_trace, parent_task_run
            )

    def invoke_openai_stream(
        self,
        input: InputType,
        input_source: DataSource | None = None,
        prior_trace: list[ChatCompletionMessageParam] | None = None,
        parent_task_run: TaskRun | None = None,
    ) -> OpenAIStreamResult:
        """Stream raw OpenAI-protocol chunks for the task execution.

        Returns an async-iterable that yields ``ModelResponseStream`` chunks
        as they arrive from the model.  After the iterator is exhausted the
        run has been validated and saved (when configured).  The resulting
        ``TaskRun`` is available via the ``.task_run`` property.

        Tool-call rounds happen internally and are not surfaced; use
        ``invoke_ai_sdk_stream`` if you need tool-call events.
        """
        return OpenAIStreamResult(
            self, input, input_source, prior_trace, parent_task_run
        )

    def invoke_ai_sdk_stream(
        self,
        input: InputType,
        input_source: DataSource | None = None,
        prior_trace: list[ChatCompletionMessageParam] | None = None,
        parent_task_run: TaskRun | None = None,
    ) -> AiSdkStreamResult:
        """Stream AI SDK protocol events for the task execution.

        Returns an async-iterable that yields ``AiSdkStreamEvent`` instances
        covering text, reasoning, tool-call lifecycle, step boundaries, and
        control events.  After the iterator is exhausted the resulting
        ``TaskRun`` is available via the ``.task_run`` property.
        """
        return AiSdkStreamResult(
            self, input, input_source, prior_trace, parent_task_run
        )

    def _prepare_stream(
        self,
        input: InputType,
        prior_trace: list[ChatCompletionMessageParam] | None,
    ) -> AdapterStream:
        prior_trace = self._normalize_prior_trace(prior_trace)
        self._reject_multiturn_with_structured_input(prior_trace)

        if self.input_schema is not None:
            validate_schema_with_value_error(
                input,
                self.input_schema,
                "This task requires a specific input schema. While the model produced JSON, that JSON didn't meet the schema. Search 'Troubleshooting Structured Data Issues' in our docs for more information.",
                require_object=False,
            )

        model_input = self._apply_input_transform(input)

        formatted_input = model_input
        formatter_id = self.model_provider().formatter
        if formatter_id is not None:
            formatter = request_formatter_from_id(formatter_id)
            formatted_input = formatter.format_input(model_input)

        return self._create_run_stream(formatted_input, prior_trace)

    def _finalize_stream(
        self,
        adapter_stream: AdapterStream,
        input: InputType,
        input_source: DataSource | None,
        parent_task_run: TaskRun | None = None,
    ) -> TaskRun:
        """Streaming invocations are only concerned with passing through events as they come in.
        At the end of the stream, we still need to validate the output, create a run and everything
        else that a non-streaming invocation would do.
        """

        result: AdapterStreamResult = adapter_stream.result
        run_output = result.run_output
        usage = result.usage

        if not run_output.is_toolcall_pending:
            # Normal completion: parse and validate output
            provider = self.model_provider()
            parser = model_parser_from_id(provider.parser)
            parsed_output = parser.parse_output(original_output=run_output)

            if self.output_schema is not None:
                if isinstance(parsed_output.output, str):
                    try:
                        parsed_output.output = parse_json_string(parsed_output.output)
                    except ValueError as e:
                        # Re-type (message unchanged) so retries treat bad JSON
                        # like a schema mismatch: a one-off model slip.
                        raise StructuredOutputParseError(str(e)) from e
                if not isinstance(parsed_output.output, dict):
                    # Valid JSON of the wrong shape (often the object wrapped in
                    # a list) is the same one-off model slip as bad JSON, so it
                    # carries the retryable type too.
                    raise StructuredOutputParseError(
                        f"structured response is not a dict: {parsed_output.output}"
                    )
                validate_schema_with_value_error(
                    parsed_output.output,
                    self.output_schema,
                    TASK_OUTPUT_SCHEMA_ERROR_PREFIX,
                )
            else:
                if not isinstance(parsed_output.output, str):
                    raise RuntimeError(
                        f"response is not a string for non-structured task: {parsed_output.output}"
                    )

            trace_has_toolcalls = parsed_output.trace is not None and any(
                message.get("role", None) == "tool" for message in parsed_output.trace
            )
            if (
                provider.reasoning_capable
                and (
                    not parsed_output.intermediate_outputs
                    or "reasoning" not in parsed_output.intermediate_outputs
                )
                and not (
                    provider.reasoning_optional_for_structured_output
                    and self.has_structured_output()
                )
                and not trace_has_toolcalls
            ):
                raise RuntimeError(
                    "Reasoning is required for this model, but no reasoning was returned."
                )

            run_output = parsed_output

        run = self.generate_run(
            input, input_source, run_output, usage, run_output.trace, parent_task_run
        )

        if (
            self.base_adapter_config.allow_saving
            and Config.shared().autosave_runs
            and self.task.path is not None
        ):
            run.save_to_file()
        else:
            run.id = None

        return run

    def _apply_input_transform(self, input: InputType) -> InputType:
        """If the run config has an input_transform, render it and return the
        resulting string. Otherwise return input unchanged.

        MCP run configs (no input_transform field) are a no-op.
        """
        if not isinstance(self.run_config, KilnAgentRunConfigProperties):
            return input
        transform = self.run_config.input_transform
        if transform is None:
            return input
        try:
            return render_input_transform(transform, input)
        except Exception as e:
            raise ValueError(f"Input transform failed: {e}") from e

    def has_structured_output(self) -> bool:
        return self.output_schema is not None

    @abstractmethod
    def adapter_name(self) -> str:
        pass

    @abstractmethod
    async def _run(
        self,
        input: InputType,
        trace_ref: list[ChatCompletionMessageParam],
        prior_trace: list[ChatCompletionMessageParam] | None = None,
    ) -> Tuple[RunOutput, Usage | None]:
        """Run the model. Implementations MUST mutate `trace_ref` in place
        (extend/append, or `trace_ref[:] = ...`) — never rebind it — so the
        caller keeps a live reference to the partial trace if an exception
        escapes.
        """
        pass

    def _messages_to_trace(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> list[ChatCompletionMessageParam]:
        """Convert the adapter's internal `messages` list to an API-safe trace.

        Default implementation returns the list as-is. Adapters that store
        internal message objects (e.g. LiteLLM's `Message`) should override
        this to normalize to `ChatCompletionMessageParam` shapes.
        """
        return messages

    def _create_run_stream(
        self,
        input: InputType,
        prior_trace: list[ChatCompletionMessageParam] | None = None,
    ) -> AdapterStream:
        """Create a stream for the adapter. Implementations must override this method to support streaming."""
        raise NotImplementedError("Streaming is not supported for this adapter type")

    def build_prompt(self) -> str:
        if self.prompt_builder is None:
            raise ValueError("Prompt builder is not available for MCP run config")
        # The prompt builder needs to know if we want to inject formatting instructions
        structured_output_mode = as_kiln_agent_run_config(
            self.run_config
        ).structured_output_mode
        add_json_instructions = self.has_structured_output() and (
            structured_output_mode == StructuredOutputMode.json_instructions
            or structured_output_mode
            == StructuredOutputMode.json_instruction_and_object
        )

        return self.prompt_builder.build_prompt(
            include_json_instructions=add_json_instructions,
            skills=self._resolve_skills(),
        )

    def _resolve_skills(self) -> list[Skill]:
        """Resolve skills from the injected skills dict.

        Uses the pre-loaded skills dict from AdapterConfig. Caches the result
        so that build_prompt and available_tools don't repeat
        the lookup. Raises ValueError if the run config references a skill
        that is not in the injected dict.
        """
        if self._resolved_skills is not None:
            return self._resolved_skills

        if self.run_config.type != "kiln_agent":
            self._resolved_skills = []
            return self._resolved_skills

        tool_config = as_kiln_agent_run_config(self.run_config).tools_config
        if tool_config is None or tool_config.tools is None:
            self._resolved_skills = []
            return self._resolved_skills

        skill_tool_ids = [
            tid for tid in tool_config.tools if tid.startswith(SKILL_TOOL_ID_PREFIX)
        ]
        if not skill_tool_ids:
            self._resolved_skills = []
            return self._resolved_skills

        injected = self.base_adapter_config.skills
        if injected is None:
            raise ValueError(
                "Run config references skills but no skills dict was provided via "
                "AdapterConfig(skills=...). Use load_skills_for_task() to pre-load "
                "skills and pass them to the adapter."
            )

        skills: list[Skill] = []
        seen: set[str] = set()
        for tool_id in skill_tool_ids:
            sid = skill_id_from_tool_id(tool_id)
            if sid not in injected:
                raise ValueError(
                    f"Skill {sid} referenced in run config but not found in the "
                    "injected skills dict."
                )
            if sid in seen:
                continue
            seen.add(sid)
            skills.append(injected[sid])

        self._resolved_skills = skills
        return self._resolved_skills

    def build_chat_formatter(
        self,
        input: InputType,
        prior_trace: list[ChatCompletionMessageParam] | None = None,
    ) -> ChatFormatter:
        prior_trace = self._normalize_prior_trace(prior_trace)
        self._reject_multiturn_with_structured_input(prior_trace)
        if prior_trace is not None:
            return MultiturnFormatter(prior_trace, input)
        if self.prompt_builder is None:
            raise ValueError("Prompt builder is not available for MCP run config")
        # Determine the chat strategy to use based on the prompt the user selected, the model's capabilities, and if the model was finetuned with a specific chat strategy.

        cot_prompt = self.prompt_builder.chain_of_thought_prompt()
        system_message = self.build_prompt()

        # The model only enters the decision once a COT prompt is in play, so keep the
        # provider lookup lazy: a plain prompt is single turn whatever the model is, and
        # resolving a provider can fail.
        provider = self.model_provider() if cot_prompt else None
        strategy = chat_strategy_for_run(
            cot_prompt=cot_prompt,
            tuned_chat_strategy=provider.tuned_chat_strategy if provider else None,
            reasoning_capable=provider.reasoning_capable if provider else False,
        )
        return get_chat_formatter(
            strategy=strategy,
            system_message=system_message,
            user_input=input,
            thinking_instructions=cot_prompt,
        )

    # create a run and task output
    def generate_run(
        self,
        input: InputType,
        input_source: DataSource | None,
        run_output: RunOutput,
        usage: Usage | None = None,
        trace: list[ChatCompletionMessageParam] | None = None,
        parent_task_run: TaskRun | None = None,
    ) -> TaskRun:
        output_str = (
            json.dumps(run_output.output, ensure_ascii=False)
            if isinstance(run_output.output, dict)
            else run_output.output
        )

        output_source_type = (
            DataSourceType.tool_call
            if self.run_config.type == "mcp"
            else DataSourceType.synthetic
        )

        new_output = TaskOutput(
            output=output_str,
            source=DataSource(
                type=output_source_type,
                properties=self._properties_for_task_output(),
                run_config_id=self.base_adapter_config.task_run_config_id,
                run_config=self.run_config,
            ),
        )

        # Convert input and output to JSON strings if they aren't strings
        input_str = (
            input if isinstance(input, str) else json.dumps(input, ensure_ascii=False)
        )

        if input_source is None:
            input_source = DataSource(
                type=DataSourceType.human,
                properties={"created_by": Config.shared().user_id},
            )

        parent_task_run_id: str | None = None
        if parent_task_run is not None:
            if parent_task_run.id is None:
                raise ValueError(
                    "parent_task_run must be persisted before using as parent: save the parent "
                    "TaskRun (e.g. save_to_file()) so it has a stable id."
                )
            parent_task_run_id = parent_task_run.id

        return TaskRun(
            parent=self.task,
            parent_task_run_id=parent_task_run_id,
            input=input_str,
            input_source=input_source,
            output=new_output,
            intermediate_outputs=run_output.intermediate_outputs,
            tags=self.base_adapter_config.default_tags or [],
            usage=usage,
            trace=trace,
            cumulative_usage=MessageUsage.from_trace(trace),
        )

    def _properties_for_task_output(self) -> Dict[str, str | int | float]:
        match self.run_config.type:
            case "mcp":
                return {}
            case "kiln_agent":
                if not isinstance(self.run_config, KilnAgentRunConfigProperties):
                    raise ValueError("Kiln agent run config is required")
                run_config = self.run_config

                props: Dict[str, str | int | float] = {}
                props["adapter_name"] = self.adapter_name()
                # Legacy properties where we save the run_config details into custom properties.
                # These are now also be saved in the run_config field.
                props["model_name"] = run_config.model_name
                props["model_provider"] = run_config.model_provider_name
                props["prompt_id"] = run_config.prompt_id
                props["structured_output_mode"] = run_config.structured_output_mode
                props["temperature"] = run_config.temperature
                props["top_p"] = run_config.top_p

                return props
            case _:
                raise_exhaustive_enum_error(self.run_config.type)

    def update_run_config_unknown_structured_output_mode(self) -> None:
        if self.run_config.type != "kiln_agent":
            return
        run_config = as_kiln_agent_run_config(self.run_config)
        structured_output_mode = run_config.structured_output_mode

        # Old datamodels didn't save the structured output mode. Some clients (tests, end users) might not set it.
        # Look up our recommended mode from ml_model_list if we have one
        if structured_output_mode == StructuredOutputMode.unknown:
            new_run_config = run_config.model_copy(deep=True)
            structured_output_mode = default_structured_output_mode_for_model_provider(
                run_config.model_name,
                run_config.model_provider_name,
            )
            new_run_config.structured_output_mode = structured_output_mode
            self.run_config = new_run_config

    async def available_tools(self) -> list[KilnToolInterface]:
        if self.run_config.type != "kiln_agent":
            return []
        tool_config = as_kiln_agent_run_config(self.run_config).tools_config
        if tool_config is None or tool_config.tools is None:
            return []

        return await assemble_unique_agent_tools(
            self.task, tool_config.tools, self._resolve_skills()
        )


async def assemble_unique_agent_tools(
    task: Task, tool_ids: list[str], skills: list[Skill]
) -> list[KilnToolInterface]:
    """Resolve a run config's tool IDs into tool instances, rejecting name collisions.

    Names may repeat across a project, but within one run config every tool the
    model sees must have a unique function name (and every attached skill a
    unique skill name — skills are loaded by name through the single "skill"
    tool). Shared by creation-time validation of run configs, which catches
    collisions at selection time, and by the runtime adapter, which stays the
    backstop for configs whose tools were renamed after creation (e.g. via
    edit_kiln_task_tool) or that were created outside the API.
    """
    non_skill_tool_ids = [
        tid for tid in tool_ids if not tid.startswith(SKILL_TOOL_ID_PREFIX)
    ]

    tools: list[KilnToolInterface] = [
        tool_from_id(tool_id, task) for tool_id in non_skill_tool_ids
    ]

    if skills:
        seen_names: set[str] = set()
        for skill in skills:
            if skill.name in seen_names:
                raise ValueError(
                    f"Duplicate skill name '{skill.name}'. Each skill must have a unique name."
                )
            seen_names.add(skill.name)
        tools.append(SkillTool(f"{SKILL_TOOL_ID_PREFIX}_combined", skills))

    tool_names = [await tool.name() for tool in tools]
    if len(tool_names) != len(set(tool_names)):
        duplicates = sorted({n for n in tool_names if tool_names.count(n) > 1})
        message = (
            f"Multiple selected tools share the same function name: {', '.join(duplicates)}. "
            "Each tool must have a unique name. Either de-select the duplicate tools, or "
            "modify their names to describe their unique purpose. The model will struggle "
            "if tools do not have descriptive names and tool execution will be undefined."
        )
        if skills and "skill" in duplicates:
            message += (
                " The name 'skill' is reserved for the tool that loads skills, "
                "so no other tool may use it while skills are selected."
            )
        raise ValueError(message)

    return tools


class OpenAIStreamResult:
    """Async-iterable wrapper around the OpenAI streaming flow.

    Yields ``ModelResponseStream`` chunks.  After iteration the resulting
    ``TaskRun`` is available via the ``.task_run`` property.

    When return_on_tool_call=True and the model requests tool calls, the stream
    will stop and ``task_run.is_toolcall_pending`` will be True.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        input: InputType,
        input_source: DataSource | None,
        prior_trace: list[ChatCompletionMessageParam] | None,
        parent_task_run: TaskRun | None = None,
    ) -> None:
        self._adapter = adapter
        self._input = input
        self._input_source = input_source
        self._prior_trace = prior_trace
        self._parent_task_run = parent_task_run
        self._task_run: TaskRun | None = None

    @property
    def task_run(self) -> TaskRun:
        if self._task_run is None:
            raise RuntimeError(
                "Stream has not been fully consumed yet. "
                "Iterate over the stream before accessing .task_run"
            )
        return self._task_run

    async def __aiter__(self) -> AsyncIterator[ModelResponseStream]:
        self._task_run = None
        async with mcp_session_scope():
            adapter_stream = self._adapter._prepare_stream(
                self._input, self._prior_trace
            )

            async for event in adapter_stream:
                if isinstance(event, ModelResponseStream):
                    yield event

            self._task_run = self._adapter._finalize_stream(
                adapter_stream, self._input, self._input_source, self._parent_task_run
            )


class AiSdkStreamResult:
    """Async-iterable wrapper around the AI SDK streaming flow.

    Yields ``AiSdkStreamEvent`` instances.  After iteration the resulting
    ``TaskRun`` is available via the ``.task_run`` property.

    When return_on_tool_call=True and the model requests tool calls, the FINISH
    event will have finishReason: "tool-calls" and ``task_run.is_toolcall_pending``
    will be True.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        input: InputType,
        input_source: DataSource | None,
        prior_trace: list[ChatCompletionMessageParam] | None,
        parent_task_run: TaskRun | None = None,
    ) -> None:
        self._adapter = adapter
        self._input = input
        self._input_source = input_source
        self._prior_trace = prior_trace
        self._parent_task_run = parent_task_run
        self._task_run: TaskRun | None = None

    @property
    def task_run(self) -> TaskRun:
        if self._task_run is None:
            raise RuntimeError(
                "Stream has not been fully consumed yet. "
                "Iterate over the stream before accessing .task_run"
            )
        return self._task_run

    async def __aiter__(self) -> AsyncIterator[AiSdkStreamEvent]:
        self._task_run = None
        async with mcp_session_scope():
            adapter_stream = self._adapter._prepare_stream(
                self._input, self._prior_trace
            )

            message_id = f"msg-{uuid.uuid4().hex}"
            converter = AiSdkStreamConverter()

            yield StartEvent(messageId=message_id)
            yield StartStepEvent()

            last_event_was_tool_call = False
            async for event in adapter_stream:
                if isinstance(event, ModelResponseStream):
                    if last_event_was_tool_call:
                        converter.reset_for_next_step()
                        last_event_was_tool_call = False
                    for ai_event in converter.convert_chunk(event):
                        yield ai_event
                elif isinstance(event, ToolCallEvent):
                    last_event_was_tool_call = True
                    for ai_event in converter.convert_tool_event(event):
                        yield ai_event

            for ai_event in converter.close_open_blocks():
                yield ai_event

            yield FinishStepEvent()

            self._task_run = self._adapter._finalize_stream(
                adapter_stream, self._input, self._input_source, self._parent_task_run
            )

            if self._task_run.is_toolcall_pending:
                yield FinishEvent(
                    messageMetadata=FinishMessageMetadata(finishReason="tool-calls"),
                )
            else:
                for ai_event in converter.finalize():
                    yield ai_event
