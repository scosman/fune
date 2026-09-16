import json
from typing import Tuple

from kiln_ai.adapters.errors import (
    KilnRunError,
    StructuredOutputParseError,
    format_error_message,
)
from kiln_ai.adapters.model_adapters.base_adapter import AdapterConfig, BaseAdapter
from kiln_ai.adapters.parsers.json_parser import parse_json_string
from kiln_ai.adapters.run_output import RunOutput
from kiln_ai.datamodel import DataSource, Task, TaskRun, Usage
from kiln_ai.datamodel.datamodel_enums import InputType
from kiln_ai.datamodel.json_schema import (
    single_string_field_name,
    validate_schema_with_value_error,
)
from kiln_ai.datamodel.run_config import McpRunConfigProperties
from kiln_ai.datamodel.task import RunConfigProperties
from kiln_ai.datamodel.task_output import TASK_OUTPUT_SCHEMA_ERROR_PREFIX
from kiln_ai.tools.mcp_session_manager import mcp_session_scope
from kiln_ai.tools.tool_registry import tool_from_id
from kiln_ai.utils.config import Config
from kiln_ai.utils.open_ai_types import (
    ChatCompletionAssistantMessageParamWrapper,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)


class MCPAdapter(BaseAdapter):
    def __init__(
        self,
        task: Task,
        run_config: RunConfigProperties,
        config: AdapterConfig | None = None,
    ):
        if run_config.type != "mcp":
            raise ValueError("MCPAdapter requires a run config with type mcp")
        super().__init__(task=task, run_config=run_config, config=config)

    def adapter_name(self) -> str:
        return "mcp_adapter"

    async def _run(
        self,
        input: InputType,
        trace_ref: list[ChatCompletionMessageParam],
        prior_trace: list[ChatCompletionMessageParam] | None = None,
    ) -> Tuple[RunOutput, Usage | None]:
        # MCP adapter is single-turn and does not build a conversation trace;
        # `trace_ref` is unused. Kept for signature compatibility with the
        # base adapter's exception-wrapping contract.
        _ = trace_ref
        if prior_trace is not None:
            raise NotImplementedError(
                "Session continuation is not supported for MCP adapter. "
                "MCP tools are single-turn and do not maintain conversation state."
            )

        run_config = self.run_config
        if not isinstance(run_config, McpRunConfigProperties):
            raise ValueError("MCPAdapter requires McpRunConfigProperties")

        # Get the actual tool from tool registry
        tool = tool_from_id(run_config.tool_reference.tool_id, self.task)

        tool_kwargs: dict[str, object]
        if self.input_schema is None:
            if not isinstance(input, str):
                raise ValueError("Plaintext task input must be a string")
            field_name = "input"
            tool_schema = run_config.tool_reference.input_schema
            if tool_schema is not None:
                field_name = single_string_field_name(tool_schema)
                if field_name is None:
                    raise ValueError(
                        "Plaintext task input requires MCP tool input schema with exactly one string field."
                    )
            tool_kwargs = {field_name: input}
        elif isinstance(input, dict):
            tool_kwargs = input
        else:
            tool_kwargs = {"input": input}

        result = await tool.run(context=None, **tool_kwargs)
        return RunOutput(output=result.output, intermediate_outputs=None), None

    async def invoke(
        self,
        input: InputType,
        input_source: DataSource | None = None,
        prior_trace: list[ChatCompletionMessageParam] | None = None,
        parent_task_run: TaskRun | None = None,
    ) -> TaskRun:
        if prior_trace or parent_task_run is not None:
            raise NotImplementedError(
                "Session continuation is not supported for MCP adapter. "
                "MCP tools are single-turn and do not maintain conversation state."
            )

        run_output, _ = await self.invoke_returning_run_output(
            input, input_source, prior_trace, parent_task_run
        )
        return run_output

    async def invoke_returning_run_output(
        self,
        input: InputType,
        input_source: DataSource | None = None,
        prior_trace: list[ChatCompletionMessageParam] | None = None,
        parent_task_run: TaskRun | None = None,
    ) -> Tuple[TaskRun, RunOutput]:
        """
        Runs the task and returns both the persisted TaskRun and raw RunOutput.
        The run executes inside an MCP session scope, so the tool call has a
        valid session to reuse. If this call opened the scope, the sessions it
        opened are cleaned up on completion; nested inside a caller's scope,
        that caller owns the teardown.
        """
        if prior_trace or parent_task_run is not None:
            raise NotImplementedError(
                "Session continuation is not supported for MCP adapter. "
                "MCP tools are single-turn and do not maintain conversation state."
            )

        async with mcp_session_scope():
            return await self._run_and_validate_output(
                input, input_source, parent_task_run
            )

    async def _run_and_validate_output(
        self,
        input: InputType,
        input_source: DataSource | None,
        parent_task_run: TaskRun | None = None,
    ) -> Tuple[TaskRun, RunOutput]:
        """
        Run the MCP task and validate the output.

        Runtime failures (tool invocation, JSON parsing, schema validation)
        are wrapped in `KilnRunError` so callers get a consistent error shape.
        MCP runs are single-turn, so there is no partial trace to preserve —
        `partial_trace` is always `None`.
        """
        # Pre-run input validation stays outside the wrap: these errors
        # happen before the tool is invoked and should surface as plain
        # exceptions (the API layer returns them as 4xx, not 500).
        if self.input_schema is not None:
            validate_schema_with_value_error(
                input,
                self.input_schema,
                "This task requires a specific input schema. While the model produced JSON, that JSON didn't meet the schema. Search 'Troubleshooting Structured Data Issues' in our docs for more information.",
                require_object=False,
            )

        try:
            # MCP adapter doesn't build a multi-message trace; pass an unused
            # `messages` list to satisfy the shared `_run` signature.
            run_output, usage = await self._run(input, [])

            if self.output_schema is not None:
                if isinstance(run_output.output, str):
                    try:
                        parsed_output = parse_json_string(run_output.output)
                    except ValueError as e:
                        # Re-type (message unchanged) so retries treat bad JSON
                        # like a schema mismatch: a one-off model slip.
                        raise StructuredOutputParseError(str(e)) from e
                else:
                    parsed_output = run_output.output
                if not isinstance(parsed_output, dict):
                    # Valid JSON of the wrong shape (often the object wrapped in
                    # a list) is the same one-off model slip as bad JSON, so it
                    # carries the retryable type too.
                    raise StructuredOutputParseError(
                        f"structured response is not a dict: {parsed_output}"
                    )
                validate_schema_with_value_error(
                    parsed_output,
                    self.output_schema,
                    TASK_OUTPUT_SCHEMA_ERROR_PREFIX,
                )
                run_output.output = parsed_output
            else:
                if not isinstance(run_output.output, str):
                    raise RuntimeError(
                        f"response is not a string for non-structured task: {run_output.output}"
                    )

            # Build single turn trace
            trace = self._build_single_turn_trace(input, run_output.output)

            run = self.generate_run(
                input, input_source, run_output, usage, trace, parent_task_run
            )

            if (
                self.base_adapter_config.allow_saving
                and Config.shared().autosave_runs
                and self.task.path is not None
            ):
                run.save_to_file()
            else:
                run.id = None

            return run, run_output
        except KilnRunError:
            # Already wrapped — pass through so we don't double-wrap.
            raise
        except Exception as e:
            raise KilnRunError(
                message=format_error_message(e),
                partial_trace=None,
                original=e,
            ) from e

    # Helpers

    @staticmethod
    def _build_single_turn_trace(
        input: InputType, output: str | dict
    ) -> list[ChatCompletionMessageParam]:
        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": input
            if isinstance(input, str)
            else json.dumps(input, ensure_ascii=False),
        }
        assistant_message: ChatCompletionAssistantMessageParamWrapper = {
            "role": "assistant",
            "content": output
            if isinstance(output, str)
            else json.dumps(output, ensure_ascii=False),
        }
        return [user_message, assistant_message]
