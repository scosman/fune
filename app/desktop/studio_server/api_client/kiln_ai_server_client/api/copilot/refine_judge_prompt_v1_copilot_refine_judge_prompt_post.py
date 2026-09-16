from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.refine_judge_prompt_input import RefineJudgePromptInput
from ...models.refine_judge_prompt_output import RefineJudgePromptOutput
from ...models.unauthorized_response import UnauthorizedResponse
from ...types import Response


def _get_kwargs(
    *,
    body: RefineJudgePromptInput,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/copilot/refine_judge_prompt",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | RefineJudgePromptOutput | UnauthorizedResponse | None:
    if response.status_code == 200:
        response_200 = RefineJudgePromptOutput.from_dict(response.json())

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
) -> Response[HTTPValidationError | RefineJudgePromptOutput | UnauthorizedResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: RefineJudgePromptInput,
) -> Response[HTTPValidationError | RefineJudgePromptOutput | UnauthorizedResponse]:
    """Refine Judge Prompt

     Refine a judge prompt from human grades on reviewed traces.

    Returns a proposed revision plus a per-edit rationale; the caller shows it
    for approval and never auto-applies it.

    Args:
        body (RefineJudgePromptInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | RefineJudgePromptOutput | UnauthorizedResponse]
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
    body: RefineJudgePromptInput,
) -> HTTPValidationError | RefineJudgePromptOutput | UnauthorizedResponse | None:
    """Refine Judge Prompt

     Refine a judge prompt from human grades on reviewed traces.

    Returns a proposed revision plus a per-edit rationale; the caller shows it
    for approval and never auto-applies it.

    Args:
        body (RefineJudgePromptInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | RefineJudgePromptOutput | UnauthorizedResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: RefineJudgePromptInput,
) -> Response[HTTPValidationError | RefineJudgePromptOutput | UnauthorizedResponse]:
    """Refine Judge Prompt

     Refine a judge prompt from human grades on reviewed traces.

    Returns a proposed revision plus a per-edit rationale; the caller shows it
    for approval and never auto-applies it.

    Args:
        body (RefineJudgePromptInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | RefineJudgePromptOutput | UnauthorizedResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: RefineJudgePromptInput,
) -> HTTPValidationError | RefineJudgePromptOutput | UnauthorizedResponse | None:
    """Refine Judge Prompt

     Refine a judge prompt from human grades on reviewed traces.

    Returns a proposed revision plus a per-edit rationale; the caller shows it
    for approval and never auto-applies it.

    Args:
        body (RefineJudgePromptInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | RefineJudgePromptOutput | UnauthorizedResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
