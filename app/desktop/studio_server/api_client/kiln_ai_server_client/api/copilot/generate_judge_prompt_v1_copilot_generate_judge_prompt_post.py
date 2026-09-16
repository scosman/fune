from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.generate_judge_prompt_api_input import GenerateJudgePromptApiInput
from ...models.generate_judge_prompt_output import GenerateJudgePromptOutput
from ...models.http_validation_error import HTTPValidationError
from ...models.unauthorized_response import UnauthorizedResponse
from ...types import Response


def _get_kwargs(
    *,
    body: GenerateJudgePromptApiInput,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/copilot/generate_judge_prompt",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> GenerateJudgePromptOutput | HTTPValidationError | UnauthorizedResponse | None:
    if response.status_code == 200:
        response_200 = GenerateJudgePromptOutput.from_dict(response.json())

        return response_200

    if response.status_code == 401:
        response_401 = UnauthorizedResponse.from_dict(response.json())

        return response_401

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[GenerateJudgePromptOutput | HTTPValidationError | UnauthorizedResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: GenerateJudgePromptApiInput,
) -> Response[GenerateJudgePromptOutput | HTTPValidationError | UnauthorizedResponse]:
    """Generate Judge Prompt

     Author a judge prompt from a spec, for a declared trace shape.

    trace_type is required, not defaulted: the task authors different rubrics
    for single-turn pairs vs multi-turn transcripts, and a default would
    silently mis-author the shape the caller forgot to declare. Returns the
    prompt only — the judge model is the caller's choice, since the studio
    can offer models this server cannot reach.

    Args:
        body (GenerateJudgePromptApiInput): Request payload for the judge prompt authoring
            copilot.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GenerateJudgePromptOutput | HTTPValidationError | UnauthorizedResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    body: GenerateJudgePromptApiInput,
) -> GenerateJudgePromptOutput | HTTPValidationError | UnauthorizedResponse | None:
    """Generate Judge Prompt

     Author a judge prompt from a spec, for a declared trace shape.

    trace_type is required, not defaulted: the task authors different rubrics
    for single-turn pairs vs multi-turn transcripts, and a default would
    silently mis-author the shape the caller forgot to declare. Returns the
    prompt only — the judge model is the caller's choice, since the studio
    can offer models this server cannot reach.

    Args:
        body (GenerateJudgePromptApiInput): Request payload for the judge prompt authoring
            copilot.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GenerateJudgePromptOutput | HTTPValidationError | UnauthorizedResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: GenerateJudgePromptApiInput,
) -> Response[GenerateJudgePromptOutput | HTTPValidationError | UnauthorizedResponse]:
    """Generate Judge Prompt

     Author a judge prompt from a spec, for a declared trace shape.

    trace_type is required, not defaulted: the task authors different rubrics
    for single-turn pairs vs multi-turn transcripts, and a default would
    silently mis-author the shape the caller forgot to declare. Returns the
    prompt only — the judge model is the caller's choice, since the studio
    can offer models this server cannot reach.

    Args:
        body (GenerateJudgePromptApiInput): Request payload for the judge prompt authoring
            copilot.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GenerateJudgePromptOutput | HTTPValidationError | UnauthorizedResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: GenerateJudgePromptApiInput,
) -> GenerateJudgePromptOutput | HTTPValidationError | UnauthorizedResponse | None:
    """Generate Judge Prompt

     Author a judge prompt from a spec, for a declared trace shape.

    trace_type is required, not defaulted: the task authors different rubrics
    for single-turn pairs vs multi-turn transcripts, and a default would
    silently mis-author the shape the caller forgot to declare. Returns the
    prompt only — the judge model is the caller's choice, since the studio
    can offer models this server cannot reach.

    Args:
        body (GenerateJudgePromptApiInput): Request payload for the judge prompt authoring
            copilot.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GenerateJudgePromptOutput | HTTPValidationError | UnauthorizedResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
