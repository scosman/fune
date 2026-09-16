import json

import httpx
import litellm
import pytest

from kiln_ai.adapters.errors import KilnRunError, StructuredOutputParseError
from kiln_ai.adapters.model_adapters.adapter_stream import (
    EMPTY_RESPONSE_ERROR_MESSAGE,
    raise_for_empty_model_response,
)
from kiln_ai.adapters.model_adapters.base_adapter import BaseAdapter, RunOutput
from kiln_ai.adapters.parsers.json_parser import parse_json_string
from kiln_ai.adapters.retry_classification import (
    is_batch_fatal_error,
    is_retryable_error,
)
from kiln_ai.datamodel import Task
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties
from kiln_ai.datamodel.task_output import TaskOutput

COUNT_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }
)

# Model output that json parsing must reject, shared so the test can ask the
# parser for the exact message it raises on this same input.
UNPARSEABLE_MODEL_OUTPUT = "Sure! Here you go."

# Valid JSON that still isn't the object a structured task needs: the model
# wrapped its answer in a list, one of the most common shape slips.
WRONG_SHAPE_MODEL_OUTPUT = '[{"count": 1}]'


def _provider_error(cls, message: str = "boom"):
    """litellm exception constructors want provider context (and some, like
    PermissionDeniedError, a raw response)."""
    try:
        return cls(message=message, llm_provider="openrouter", model="gpt_5_5")
    except TypeError:
        response = httpx.Response(
            status_code=403, request=httpx.Request("POST", "http://test")
        )
        return cls(
            message=message,
            llm_provider="openrouter",
            model="gpt_5_5",
            response=response,
        )


def _wrapped(error: Exception) -> KilnRunError:
    return KilnRunError("generic user-facing text", partial_trace=None, original=error)


class TestIsBatchFatalError:
    @pytest.mark.parametrize(
        "cls",
        [
            litellm.AuthenticationError,
            litellm.PermissionDeniedError,
            litellm.NotFoundError,
        ],
    )
    def test_config_scoped_provider_errors_are_batch_fatal(self, cls):
        assert is_batch_fatal_error(_provider_error(cls)) is True

    def test_budget_exceeded_is_batch_fatal(self):
        error = litellm.BudgetExceededError(current_cost=2.0, max_budget=1.0)
        assert is_batch_fatal_error(error) is True

    @pytest.mark.parametrize(
        "error",
        [
            # Transient classes must stay case-scoped (and retryable).
            _provider_error(litellm.RateLimitError),
            _provider_error(litellm.InternalServerError),
            _provider_error(litellm.ServiceUnavailableError),
            # Unrecognized errors must never abort a batch on a guess.
            ValueError("some judge parsing problem"),
            RuntimeError("anything else"),
            TimeoutError("slow provider"),
        ],
    )
    def test_everything_else_stays_case_scoped(self, error):
        assert is_batch_fatal_error(error) is False

    def test_unwraps_kiln_run_error(self):
        inner = _provider_error(litellm.AuthenticationError)
        assert is_batch_fatal_error(_wrapped(inner)) is True

    def test_batch_fatal_is_disjoint_from_retryable(self):
        # The abort path assumes a batch-fatal error is deterministic — a
        # class in both sets would retry twice and then abort, wasting the
        # retries and delaying the abort.
        for cls in (
            litellm.AuthenticationError,
            litellm.PermissionDeniedError,
            litellm.NotFoundError,
        ):
            error = _provider_error(cls)
            assert is_retryable_error(error) is False
            assert is_batch_fatal_error(error) is True


class TestUnwrapInteraction:
    def test_wrapped_transient_is_retryable_not_fatal(self):
        wrapped = _wrapped(_provider_error(litellm.RateLimitError))
        assert is_retryable_error(wrapped) is True
        assert is_batch_fatal_error(wrapped) is False


class TestTransientProviderClassification:
    def test_provider_timeout_is_retryable(self):
        # litellm.Timeout descends from openai's connection error, not
        # litellm's, so the classifier has to name it explicitly. Build a real
        # instance so a litellm upgrade that reshuffles the hierarchy fails here.
        error = litellm.Timeout(
            message="request timed out", model="gpt_5_5", llm_provider="openrouter"
        )
        assert isinstance(error, litellm.APIConnectionError) is False
        assert is_retryable_error(error) is True


class TestSchemaMismatchClassification:
    def test_real_schema_mismatch_raise_site_is_retryable(self):
        # Trigger an actual production raise site rather than a hand-built
        # ValueError, so the classifier can't drift from the raised message.
        task = Task(
            name="test task",
            instruction="test instruction",
            output_json_schema=COUNT_SCHEMA,
        )
        output = TaskOutput(output=json.dumps({"count": "not_an_int"}))
        with pytest.raises(ValueError) as exc_info:
            output.validate_output_format(task)
        assert is_retryable_error(exc_info.value) is True


class _CannedOutputAdapter(BaseAdapter):
    """Adapter whose model returns a fixed output, so tests exercise the real
    parse and shape checks in `invoke` instead of hand-built exceptions."""

    def __init__(self, model_output: str, **kwargs):
        super().__init__(**kwargs)
        self._model_output = model_output

    async def _run(self, input, trace_ref, **kwargs):
        output = RunOutput(output=self._model_output, intermediate_outputs=None)
        return output, None

    def adapter_name(self) -> str:
        return "test"


def _structured_adapter(model_output: str) -> _CannedOutputAdapter:
    task = Task(
        name="test task",
        instruction="test instruction",
        output_json_schema=COUNT_SCHEMA,
    )
    return _CannedOutputAdapter(
        model_output,
        task=task,
        run_config=KilnAgentRunConfigProperties(
            model_name="phi_3_5",
            model_provider_name="ollama",
            prompt_id="simple_prompt_builder",
            structured_output_mode="json_schema",
        ),
    )


class TestStructuredOutputParseClassification:
    async def test_real_json_parse_failure_raise_site_is_retryable(self):
        # A structured-output task whose model output can't be parsed is the
        # same one-off model slip as a schema mismatch, so it must retry too.
        adapter = _structured_adapter(UNPARSEABLE_MODEL_OUTPUT)
        with pytest.raises(KilnRunError) as exc_info:
            await adapter.invoke("test input")
        assert isinstance(exc_info.value.original, StructuredOutputParseError)
        assert is_retryable_error(exc_info.value) is True

        # Re-typing must not alter the message. Compare against what the parser
        # itself raises for the same output, so the two can't drift apart.
        with pytest.raises(ValueError) as parser_exc_info:
            parse_json_string(UNPARSEABLE_MODEL_OUTPUT)
        parser_message = str(parser_exc_info.value)
        assert str(exc_info.value.original) == parser_message
        # And the message the user sees survives the KilnRunError wrapping.
        assert str(exc_info.value) == parser_message

    async def test_real_wrong_shape_raise_site_is_retryable(self):
        # Valid JSON of the wrong shape (here, the object wrapped in a list)
        # never reaches the parse failure above, so it needs its own pin: it is
        # the same one-off model slip and must retry rather than fail the run.
        adapter = _structured_adapter(WRONG_SHAPE_MODEL_OUTPUT)
        with pytest.raises(KilnRunError) as exc_info:
            await adapter.invoke("test input")
        assert isinstance(exc_info.value.original, StructuredOutputParseError)
        assert is_retryable_error(exc_info.value) is True

        # Re-typing must not alter what the user sees: same text the raise site
        # produced when it was a RuntimeError.
        expected_message = "structured response is not a dict: [{'count': 1}]"
        assert str(exc_info.value.original) == expected_message
        assert str(exc_info.value) == expected_message

    def test_bare_value_error_is_not_retryable(self):
        # Only the ValueErrors the classifier names are retryable; a ValueError
        # stays terminal everywhere else (config errors, bad arguments, etc.),
        # which pins how narrow the recognized set is.
        assert is_retryable_error(ValueError("some judge parsing problem")) is False


class TestEmptyModelResponseClassification:
    """An assistant message with neither content nor tool calls: transient when
    the model simply returned nothing, permanent when a safety classifier
    refused. Both leave the same empty choice, so they are pinned together."""

    def test_real_empty_response_raise_site_is_retryable(self):
        # Trigger the production raise site rather than a hand-built ValueError,
        # so the classifier can't drift from the raised message.
        with pytest.raises(ValueError) as exc_info:
            raise_for_empty_model_response(
                {"finish_reason": "stop", "message": {"content": None}}
            )
        assert str(exc_info.value) == EMPTY_RESPONSE_ERROR_MESSAGE
        assert is_retryable_error(exc_info.value) is True

    def test_real_content_filter_raise_site_is_not_retryable(self):
        # Same empty choice, but the provider says it refused: repeating the
        # call returns the same refusal, so it must not be retried.
        with pytest.raises(ValueError) as exc_info:
            raise_for_empty_model_response(
                {"finish_reason": "content_filter", "message": {"content": None}}
            )
        assert not str(exc_info.value).startswith(EMPTY_RESPONSE_ERROR_MESSAGE)
        assert is_retryable_error(exc_info.value) is False
