from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.build_claim_evidence_input import BuildClaimEvidenceInput
from ...models.build_claim_evidence_output import BuildClaimEvidenceOutput
from ...models.http_validation_error import HTTPValidationError
from ...models.unauthorized_response import UnauthorizedResponse
from ...types import Response


def _get_kwargs(
    *,
    body: BuildClaimEvidenceInput,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/copilot/build_claim_evidence",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> BuildClaimEvidenceOutput | HTTPValidationError | UnauthorizedResponse | None:
    if response.status_code == 200:
        response_200 = BuildClaimEvidenceOutput.from_dict(response.json())

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
) -> Response[BuildClaimEvidenceOutput | HTTPValidationError | UnauthorizedResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: BuildClaimEvidenceInput,
) -> Response[BuildClaimEvidenceOutput | HTTPValidationError | UnauthorizedResponse]:
    """Build Claim Evidence

     Build a review card (overview plus claims) for one eval trace + judge decision.

    Args:
        body (BuildClaimEvidenceInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[BuildClaimEvidenceOutput | HTTPValidationError | UnauthorizedResponse]
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
    body: BuildClaimEvidenceInput,
) -> BuildClaimEvidenceOutput | HTTPValidationError | UnauthorizedResponse | None:
    """Build Claim Evidence

     Build a review card (overview plus claims) for one eval trace + judge decision.

    Args:
        body (BuildClaimEvidenceInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        BuildClaimEvidenceOutput | HTTPValidationError | UnauthorizedResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: BuildClaimEvidenceInput,
) -> Response[BuildClaimEvidenceOutput | HTTPValidationError | UnauthorizedResponse]:
    """Build Claim Evidence

     Build a review card (overview plus claims) for one eval trace + judge decision.

    Args:
        body (BuildClaimEvidenceInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[BuildClaimEvidenceOutput | HTTPValidationError | UnauthorizedResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: BuildClaimEvidenceInput,
) -> BuildClaimEvidenceOutput | HTTPValidationError | UnauthorizedResponse | None:
    """Build Claim Evidence

     Build a review card (overview plus claims) for one eval trace + judge decision.

    Args:
        body (BuildClaimEvidenceInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        BuildClaimEvidenceOutput | HTTPValidationError | UnauthorizedResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
