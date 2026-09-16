"""Shared retry classification for LLM-calling pipelines.

AsyncJobRunner retries only jobs that raise RetryableError; pipelines map
provider failures onto that contract through this classifier, so "transient"
means the same thing everywhere a model call can flake (the eval runner,
the synthetic-user batch runner).
"""

import litellm

from kiln_ai.adapters.errors import KilnRunError, StructuredOutputParseError
from kiln_ai.adapters.model_adapters.adapter_stream import (
    EMPTY_RESPONSE_ERROR_MESSAGE,
)
from kiln_ai.datamodel.task_output import TASK_OUTPUT_SCHEMA_ERROR_PREFIX


def unwrap_kiln_run_error(e: BaseException) -> BaseException:
    """The innermost non-wrapper error.

    The model adapter wraps provider exceptions in KilnRunError (to carry the
    partial trace), whose own message is genericized user-facing text — so both
    retry classification and error detail must use the underlying error. The
    isinstance guard on `original` keeps a (contract-violating) None from
    escaping as the result."""
    while isinstance(e, KilnRunError) and isinstance(e.original, BaseException):
        e = e.original
    return e


def is_retryable_error(e: BaseException) -> bool:
    """True for transient provider failures worth another attempt."""
    e = unwrap_kiln_run_error(e)

    if isinstance(
        e,
        (
            litellm.RateLimitError,
            litellm.APIConnectionError,
            # Listed separately: litellm.Timeout inherits from openai's
            # connection error, not litellm's, so APIConnectionError above
            # does not cover it.
            litellm.Timeout,
            litellm.InternalServerError,
            litellm.ServiceUnavailableError,
            litellm.BadGatewayError,
            litellm.JSONSchemaValidationError,
        ),
    ):
        return True

    # The model's output wasn't the JSON object a structured-output task
    # requires (unparseable, or valid JSON of the wrong shape). Same
    # transient event as a schema mismatch below.
    if isinstance(e, StructuredOutputParseError):
        return True

    # ValueError raised when structured output doesn't match the task's schema;
    # recognized by the shared prefix the raise sites prepend to the message.
    if isinstance(e, ValueError) and TASK_OUTPUT_SCHEMA_ERROR_PREFIX in str(e):
        return True

    # ValueError raised when the model returns an assistant message carrying
    # neither content nor tool calls; recognized by the message the raise sites
    # share. Transient: the same call repeated almost always comes back with
    # content. A content-filter refusal produces a different message from the
    # same raise site and is deliberately left out — a refusal is the model's
    # deterministic answer, so repeating the call returns the same refusal.
    if isinstance(e, ValueError) and str(e).startswith(EMPTY_RESPONSE_ERROR_MESSAGE):
        return True

    return False


def is_batch_fatal_error(e: BaseException) -> bool:
    """True for config-scoped failures guaranteed to kill every case in a
    batch identically: bad or revoked credentials, a deprecated/unknown
    model, a hard budget wall. On the first such error a batch owner should
    abort the whole batch rather than fail its cases one by one (and keep
    paying for the upstream work that feeds them).

    Disjoint from is_retryable_error — these are deterministic. Deliberately
    conservative: an unrecognized error stays case-scoped; never abort a
    batch on a guess.
    """
    e = unwrap_kiln_run_error(e)

    return isinstance(
        e,
        (
            litellm.AuthenticationError,  # bad/revoked key
            litellm.PermissionDeniedError,  # key lacks access
            litellm.NotFoundError,  # model deprecated/unknown
            litellm.BudgetExceededError,  # hard spend ceiling
        ),
    )
