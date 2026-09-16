"""Thin async wrapper over the vendored kiln_server SDK for `/generate`.

Owns two concerns the SDK leaves to callers:

1. Error classification. The SDK parses 200/401/422/500/502 into typed
   models; we translate those into the wrapper's typed exception hierarchy
   so callers never inspect raw HTTP status codes.
2. No retry. `/generate` is a once-per-batch authoring call, and
   kiln_server already retries transient provider failures internally
   before returning 502. A 502 reaching us is a genuine failure.
"""

import logging

from app.desktop.studio_server.api_client.kiln_ai_server_client.api.synthetic_user import (
    generate_v1_synthetic_user_generate_post,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.client import (
    AuthenticatedClient,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models import (
    GenerateSyntheticUsersRequest,
    GenerateSyntheticUsersResponse,
    GenerateV1SyntheticUserGeneratePostResponse500,
    GenerateV1SyntheticUserGeneratePostResponse502,
    HTTPValidationError,
    SyntheticUserCase,
    UnauthorizedResponse,
    ValidationError,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.types import (
    UNSET,
    Response,
)
from app.desktop.studio_server.api_client.kiln_server_client import (
    get_authenticated_client,
)

logger = logging.getLogger(__name__)


class SyntheticUserError(Exception):
    """Base class for SyntheticUserClient errors. Carries the kiln_server
    error code (e.g. `llm_unavailable`, `upstream_invalid_output`) when
    available, plus the HTTP status for debugging.
    """

    def __init__(self, code: str, message: str, status_code: int | None = None):
        prefix = f"{code}: " if code else ""
        super().__init__(f"{prefix}{message}")
        self.code = code
        self.message = message
        self.status_code = status_code


class SyntheticUserRequestError(SyntheticUserError):
    """Raised on 4xx. Either we sent a bad body (422 — runner bug) or the
    caller's credentials don't work (401). Not retryable — fix inputs.
    """


class SyntheticUserServerError(SyntheticUserError):
    """Raised on 5xx. 500 is a server bug; 502 is the kiln_server pipeline
    giving up on the upstream provider after its own internal retry. Not
    retryable at this layer — bubble up as a per-batch failure.
    """


class SyntheticUserClient:
    """Async client for kiln_server's `/v1/synthetic_user/generate`.

    Construction is cheap — no network until a method is called. Pass the
    same instance to as many coroutines as you want; the underlying httpx
    client is async-safe.
    """

    def __init__(self, *, api_key: str):
        self._client: AuthenticatedClient = get_authenticated_client(api_key=api_key)

    async def generate(
        self,
        *,
        target_task_prompt: str,
        target_specification: str,
        num_cases: int,
        case_scenarios: list[str] | None = None,
    ) -> list[SyntheticUserCase]:
        """POST /v1/synthetic_user/generate. Returns the SDK's case models
        as-is; each carries `seed_prompt` and `synthetic_user_info` (the
        tagged blob — see kiln_ai.synthetic_user.parser for the schema).

        `case_scenarios` (length must equal num_cases) makes it a scenario
        batch: one server pass generates case i around scenario i, and each
        returned case carries `scenario_index`. Under the server's salvage
        contract the response may be shorter than num_cases — scenario_index,
        not position, is the scenario mapping.
        """
        body = GenerateSyntheticUsersRequest(
            target_task_prompt=target_task_prompt,
            target_specification=target_specification,
            num_cases=num_cases,
            case_scenarios=case_scenarios if case_scenarios is not None else UNSET,
        )
        response = await generate_v1_synthetic_user_generate_post.asyncio_detailed(
            client=self._client, body=body
        )
        return self._extract_cases_or_raise(response)

    @staticmethod
    def _extract_cases_or_raise(response: Response) -> list[SyntheticUserCase]:
        """Translate the SDK's parsed response into either the case list
        (on 2xx) or a typed exception (on anything else).
        """
        parsed = response.parsed
        status = int(response.status_code)

        if isinstance(parsed, GenerateSyntheticUsersResponse):
            return list(parsed.cases)

        # Typed error bodies the SDK parses for us.
        if isinstance(parsed, GenerateV1SyntheticUserGeneratePostResponse502):
            # `code` is a typed enum on 502; surface its string value so
            # downstream callers can discriminate llm_unavailable from
            # upstream_invalid_output without importing the SDK type.
            raise SyntheticUserServerError(
                code=parsed.code.value,
                message=parsed.message,
                status_code=status,
            )
        if isinstance(parsed, GenerateV1SyntheticUserGeneratePostResponse500):
            raise SyntheticUserServerError(
                code=_code_or_default(parsed.code, f"http_{status}"),
                message=parsed.message,
                status_code=status,
            )
        if isinstance(parsed, UnauthorizedResponse):
            # The 401 comes from the server's API key dependency, which answers
            # every route with one shared body: it carries no code to
            # discriminate on, so the failure is always an auth failure.
            raise SyntheticUserRequestError(
                code="unauthorized",
                message=parsed.message,
                status_code=status,
            )
        if isinstance(parsed, HTTPValidationError):
            # 422 — we sent a body the server's pydantic validator rejected.
            # Indicates a runner bug; surface enough detail to debug.
            raise SyntheticUserRequestError(
                code="http_422",
                message=_format_validation_detail(parsed),
                status_code=status,
            )

        # Fallback: SDK couldn't parse a body, or the response is otherwise
        # unexpected. Pick a coarse classification by status range.
        if 400 <= status < 500:
            raise SyntheticUserRequestError(
                code=f"http_{status}",
                message="Unexpected client-error response from kiln_server.",
                status_code=status,
            )
        raise SyntheticUserServerError(
            code=f"http_{status}",
            message="Unexpected response from kiln_server.",
            status_code=status,
        )


def _code_or_default(code: object, default: str) -> str:
    """Resolve a typed-model `code` field that may be `UNSET` or a string."""
    return code if isinstance(code, str) and code else default


def _format_validation_detail(error: HTTPValidationError) -> str:
    """Render a FastAPI HTTPValidationError into a single-line message
    useful for debugging which field the runner sent wrong.
    """
    detail = error.detail
    if not isinstance(detail, list):
        return "Validation error (no detail)."
    parts: list[str] = []
    skipped = 0
    for item in detail:
        if not isinstance(item, ValidationError):
            skipped += 1
            continue
        loc = ".".join(str(x) for x in item.loc)
        parts.append(f"{loc}: {item.msg}")
    if not parts:
        # The SDK's HTTPValidationError.detail had items the SDK couldn't
        # parse as ValidationError — a shape we don't expect today. Log
        # so we can spot the discrepancy if it ever appears in the wild,
        # instead of silently returning the empty fallback.
        if skipped:
            logger.warning(
                "HTTPValidationError carried %d non-ValidationError detail item(s); "
                "raw detail repr: %r",
                skipped,
                detail,
            )
        return "Validation error (no detail)."
    return "Validation error: " + "; ".join(parts)
