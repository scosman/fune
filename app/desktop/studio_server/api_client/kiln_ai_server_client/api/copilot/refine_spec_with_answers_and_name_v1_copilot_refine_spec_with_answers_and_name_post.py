from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.refine_spec_from_answers_and_name_output import RefineSpecFromAnswersAndNameOutput
from ...models.submit_answers_request import SubmitAnswersRequest
from ...models.unauthorized_response import UnauthorizedResponse
from ...types import Response


def _get_kwargs(
    *,
    body: SubmitAnswersRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/copilot/refine_spec_with_answers_and_name",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | RefineSpecFromAnswersAndNameOutput | UnauthorizedResponse | None:
    if response.status_code == 200:
        response_200 = RefineSpecFromAnswersAndNameOutput.from_dict(response.json())

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
) -> Response[HTTPValidationError | RefineSpecFromAnswersAndNameOutput | UnauthorizedResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: SubmitAnswersRequest,
) -> Response[HTTPValidationError | RefineSpecFromAnswersAndNameOutput | UnauthorizedResponse]:
    """Refine Spec With Answers And Name

     Refine a specification with answers, also returning a suggested eval name.

    Returns the codegen'd refine-and-name task output directly (rather than a
    kiln-ai API model) so the suggested_name field ships without waiting on the
    pinned kiln-ai dependency. The plain refine route stays frozen for shipped
    clients.

    Args:
        body (SubmitAnswersRequest): Request to submit answers to a question set.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | RefineSpecFromAnswersAndNameOutput | UnauthorizedResponse]
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
    body: SubmitAnswersRequest,
) -> HTTPValidationError | RefineSpecFromAnswersAndNameOutput | UnauthorizedResponse | None:
    """Refine Spec With Answers And Name

     Refine a specification with answers, also returning a suggested eval name.

    Returns the codegen'd refine-and-name task output directly (rather than a
    kiln-ai API model) so the suggested_name field ships without waiting on the
    pinned kiln-ai dependency. The plain refine route stays frozen for shipped
    clients.

    Args:
        body (SubmitAnswersRequest): Request to submit answers to a question set.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | RefineSpecFromAnswersAndNameOutput | UnauthorizedResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: SubmitAnswersRequest,
) -> Response[HTTPValidationError | RefineSpecFromAnswersAndNameOutput | UnauthorizedResponse]:
    """Refine Spec With Answers And Name

     Refine a specification with answers, also returning a suggested eval name.

    Returns the codegen'd refine-and-name task output directly (rather than a
    kiln-ai API model) so the suggested_name field ships without waiting on the
    pinned kiln-ai dependency. The plain refine route stays frozen for shipped
    clients.

    Args:
        body (SubmitAnswersRequest): Request to submit answers to a question set.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | RefineSpecFromAnswersAndNameOutput | UnauthorizedResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: SubmitAnswersRequest,
) -> HTTPValidationError | RefineSpecFromAnswersAndNameOutput | UnauthorizedResponse | None:
    """Refine Spec With Answers And Name

     Refine a specification with answers, also returning a suggested eval name.

    Returns the codegen'd refine-and-name task output directly (rather than a
    kiln-ai API model) so the suggested_name field ships without waiting on the
    pinned kiln-ai dependency. The plain refine route stays frozen for shipped
    clients.

    Args:
        body (SubmitAnswersRequest): Request to submit answers to a question set.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | RefineSpecFromAnswersAndNameOutput | UnauthorizedResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
