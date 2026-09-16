"""Shared error types for adapter runs.

Provides:
- `ErrorWithTrace`: the API response body for run failures.
- `KilnRunError`: the exception thrown by the adapter that carries the partial
  conversation trace across the exception boundary so the API layer can return
  it to the client.
- `StructuredOutputParseError`: on a structured-output task, the model's
  response wasn't parseable JSON, or parsed to something other than an object.
- `format_error_message`: maps known exceptions to user-friendly text.
"""

from __future__ import annotations

import litellm
from pydantic import BaseModel

from kiln_ai.utils.open_ai_types import ChatCompletionMessageParam

_GENERIC_FALLBACK_MESSAGE = "An unexpected error occurred."


class ErrorWithTrace(BaseModel):
    """Structured error response pairing a user-friendly message with the
    partial conversation trace built up before the failure.

    Returned by endpoints that run a task adapter when the adapter throws
    after starting a run (LLM calls made, tools invoked, etc.).
    """

    message: str
    error_type: str
    trace: list[ChatCompletionMessageParam] | None = None


class KilnRunError(Exception):
    """Raised when an adapter run fails after the trace has started being built.

    Carries the partial trace so the API layer can return it to the client.
    The original exception chain is preserved via `__cause__`.
    """

    def __init__(
        self,
        message: str,
        partial_trace: list[ChatCompletionMessageParam] | None,
        original: Exception,
    ) -> None:
        super().__init__(message)
        self.partial_trace = partial_trace
        self.original = original
        self.error_type = type(original).__name__


class StructuredOutputParseError(ValueError):
    """Raised when a structured-output task's model response is the wrong
    shape: not valid JSON, or valid JSON that isn't the object the task needs
    (a list, number, or null — commonly the object wrapped in a list).

    Subclasses ValueError so existing `except ValueError` handling and the
    user-facing message are unchanged. The distinct type exists so retry
    classification can treat wrong-shape output like a schema mismatch: both
    mean the model flubbed its output shape once, and a retry may fix it.
    """


def _safe_str(exc: Exception) -> str:
    """Return str(exc); fall back to the exception class name if the string
    is empty, and to a generic message only if str() itself misbehaves."""
    try:
        result = str(exc)
    except Exception:
        return _GENERIC_FALLBACK_MESSAGE
    if not isinstance(result, str):
        return _GENERIC_FALLBACK_MESSAGE
    if not result:
        # An exception with an empty __str__ (e.g., RuntimeError("")) is more
        # useful to the user as the class name than a generic fallback.
        return type(exc).__name__
    return result


def format_error_message(exc: Exception) -> str:
    """Map an exception to a user-friendly message.

    Known exception types get custom messages. Unknown types get a generic
    fallback to avoid leaking provider internals to the client.
    """
    try:
        # Order matters: several litellm error classes inherit from each
        # other (e.g., RateLimitError is a subclass of InternalServerError in
        # some litellm versions), so we match the most specific types first
        # and fall through to the broader "provider unavailable" bucket last.
        if isinstance(exc, litellm.RateLimitError):
            return "Rate limit exceeded. Wait a moment and try again."
        if isinstance(exc, litellm.AuthenticationError):
            return "Authentication with the model provider failed. Check your API key."
        # Listed before the connection bucket: litellm.Timeout inherits from
        # openai's connection error, not litellm's, so it needs its own branch
        # (and a request that timed out did connect, so that message would lie).
        if isinstance(exc, litellm.Timeout):
            return "The model provider took too long to respond. Try again in a moment."
        if isinstance(exc, litellm.APIConnectionError):
            return "Could not connect to the model provider. Check your network connection."
        if isinstance(
            exc,
            (
                litellm.ServiceUnavailableError,
                litellm.BadGatewayError,
                litellm.InternalServerError,
            ),
        ):
            return "The model provider is currently unavailable. Try again in a moment."
        if isinstance(exc, litellm.JSONSchemaValidationError):
            return "The model's output didn't match the expected format."

        if isinstance(exc, RuntimeError):
            msg = _safe_str(exc)
            if msg.startswith("Too many turns"):
                return "The run exceeded the maximum number of turns."
            if msg.startswith("Too many tool calls"):
                return "The run exceeded the maximum number of tool calls in one turn."
            # Other RuntimeErrors (tool not found, arg parse/validate failures,
            # reasoning required) already have user-friendly messages with
            # useful context (e.g., tool names), so pass them through.
            return msg

        if isinstance(exc, ValueError):
            # ValueError messages from the adapter (schema mismatches, etc.)
            # are already user-readable and include helpful detail.
            return _safe_str(exc)

        return _GENERIC_FALLBACK_MESSAGE
    except Exception:
        return _GENERIC_FALLBACK_MESSAGE
