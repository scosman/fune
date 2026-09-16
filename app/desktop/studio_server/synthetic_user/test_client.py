"""Unit tests for SyntheticUserClient (the `/generate` wrapper).

The SDK's `asyncio_detailed` is patched per-test so no real network call
happens. Tests cover each status the SDK models (200, 401, 422, 500, 502)
+ the fallback paths for unparseable bodies.
"""

from http import HTTPStatus
from unittest.mock import AsyncMock

import pytest

from app.desktop.studio_server.api_client.kiln_ai_server_client.models import (
    GenerateSyntheticUsersResponse,
    GenerateV1SyntheticUserGeneratePostResponse500,
    GenerateV1SyntheticUserGeneratePostResponse502,
    GenerateV1SyntheticUserGeneratePostResponse502Code,
    HTTPValidationError,
    SyntheticUserCase,
    UnauthorizedResponse,
    ValidationError,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.types import (
    UNSET,
    Response,
)
from app.desktop.studio_server.synthetic_user import client as client_mod
from app.desktop.studio_server.synthetic_user.client import (
    SyntheticUserClient,
    SyntheticUserRequestError,
    SyntheticUserServerError,
)


def _make_client() -> SyntheticUserClient:
    return SyntheticUserClient(api_key="test-key")


def _patch_generate(monkeypatch: pytest.MonkeyPatch, mock: AsyncMock) -> None:
    monkeypatch.setattr(
        client_mod.generate_v1_synthetic_user_generate_post,
        "asyncio_detailed",
        mock,
    )


def _ok_response(num_cases: int = 1) -> Response:
    cases = [
        SyntheticUserCase(
            seed_prompt=f"seed-{i}",
            synthetic_user_info=(
                f"<persona>persona-{i}</persona>"
                f"<goal>goal-{i}</goal>"
                f"<behavior_guidance>guidance-{i}</behavior_guidance>"
            ),
        )
        for i in range(num_cases)
    ]
    return Response(
        status_code=HTTPStatus.OK,
        content=b"{}",
        headers={},
        parsed=GenerateSyntheticUsersResponse(cases=cases),
    )


def _err_response(status: int, parsed: object) -> Response:
    return Response(
        status_code=HTTPStatus(status),
        content=b"{}",
        headers={},
        parsed=parsed,  # type: ignore[arg-type]
    )


# ───────────────────────── happy path ─────────────────────────


@pytest.mark.asyncio
async def test_generate_happy_path_returns_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_generate(monkeypatch, AsyncMock(return_value=_ok_response(num_cases=3)))

    cases = await _make_client().generate(
        target_task_prompt="prompt",
        target_specification="spec",
        num_cases=3,
    )

    assert len(cases) == 3
    assert cases[0].seed_prompt == "seed-0"
    # synthetic_user_info is the tagged blob — opaque at this layer.
    assert "<persona>persona-0</persona>" in cases[0].synthetic_user_info


@pytest.mark.asyncio
async def test_generate_passes_request_body_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list = []

    async def _capture(*, client, body):
        captured.append(body)
        return _ok_response()

    _patch_generate(monkeypatch, AsyncMock(side_effect=_capture))

    await _make_client().generate(
        target_task_prompt="my task prompt",
        target_specification="my spec",
        num_cases=5,
    )

    assert len(captured) == 1
    sent = captured[0]
    assert sent.target_task_prompt == "my task prompt"
    assert sent.target_specification == "my spec"
    assert sent.num_cases == 5
    # No scenarios given → the field is omitted from the wire body entirely.
    assert "case_scenarios" not in sent.to_dict()


@pytest.mark.asyncio
async def test_generate_passes_case_scenarios_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list = []

    async def _capture(*, client, body):
        captured.append(body)
        return _ok_response()

    _patch_generate(monkeypatch, AsyncMock(side_effect=_capture))

    await _make_client().generate(
        target_task_prompt="prompt",
        target_specification="spec",
        num_cases=2,
        case_scenarios=["scenario A", "scenario B"],
    )

    assert captured[0].to_dict()["case_scenarios"] == ["scenario A", "scenario B"]


# ───────────────────────── 502 (typed code) ─────────────────────────


@pytest.mark.asyncio
async def test_generate_502_llm_unavailable_surfaces_typed_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = GenerateV1SyntheticUserGeneratePostResponse502(
        message="provider timed out",
        code=GenerateV1SyntheticUserGeneratePostResponse502Code.LLM_UNAVAILABLE,
    )
    _patch_generate(monkeypatch, AsyncMock(return_value=_err_response(502, parsed)))

    with pytest.raises(SyntheticUserServerError) as exc:
        await _make_client().generate(
            target_task_prompt="p", target_specification="s", num_cases=1
        )

    assert exc.value.code == "llm_unavailable"
    assert exc.value.message == "provider timed out"
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_generate_502_upstream_invalid_output_surfaces_typed_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = GenerateV1SyntheticUserGeneratePostResponse502(
        message="model returned unparseable output",
        code=GenerateV1SyntheticUserGeneratePostResponse502Code.UPSTREAM_INVALID_OUTPUT,
    )
    _patch_generate(monkeypatch, AsyncMock(return_value=_err_response(502, parsed)))

    with pytest.raises(SyntheticUserServerError) as exc:
        await _make_client().generate(
            target_task_prompt="p", target_specification="s", num_cases=1
        )

    assert exc.value.code == "upstream_invalid_output"


# ───────────────────────── 500 ─────────────────────────


@pytest.mark.asyncio
async def test_generate_500_with_code(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed = GenerateV1SyntheticUserGeneratePostResponse500(
        message="kaboom",
        code="internal_error",
    )
    _patch_generate(monkeypatch, AsyncMock(return_value=_err_response(500, parsed)))

    with pytest.raises(SyntheticUserServerError) as exc:
        await _make_client().generate(
            target_task_prompt="p", target_specification="s", num_cases=1
        )

    assert exc.value.code == "internal_error"
    assert exc.value.message == "kaboom"
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_generate_500_with_unset_code_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = GenerateV1SyntheticUserGeneratePostResponse500(
        message="kaboom",
        code=UNSET,  # type: ignore[arg-type]
    )
    _patch_generate(monkeypatch, AsyncMock(return_value=_err_response(500, parsed)))

    with pytest.raises(SyntheticUserServerError) as exc:
        await _make_client().generate(
            target_task_prompt="p", target_specification="s", num_cases=1
        )

    assert exc.value.code == "http_500"


# ───────────────────────── 401 ─────────────────────────


@pytest.mark.asyncio
async def test_generate_401_surfaces_as_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server answers every route's 401 with one shared body. It carries
    no code, so the wrapper supplies the classification itself."""
    parsed = UnauthorizedResponse(
        message="invalid api key",
        trace_id="trace-abc123",
    )
    _patch_generate(monkeypatch, AsyncMock(return_value=_err_response(401, parsed)))

    with pytest.raises(SyntheticUserRequestError) as exc:
        await _make_client().generate(
            target_task_prompt="p", target_specification="s", num_cases=1
        )

    assert exc.value.code == "unauthorized"
    assert exc.value.status_code == 401


# ───────────────────────── 422 (HTTPValidationError) ─────────────────────────


@pytest.mark.asyncio
async def test_generate_422_renders_validation_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = HTTPValidationError(
        detail=[
            ValidationError(
                loc=["body", "num_cases"],
                msg="value is greater than 50",
                type_="value_error",
            ),
            ValidationError(
                loc=["body", "target_specification"],
                msg="field required",
                type_="missing",
            ),
        ]
    )
    _patch_generate(monkeypatch, AsyncMock(return_value=_err_response(422, parsed)))

    with pytest.raises(SyntheticUserRequestError) as exc:
        await _make_client().generate(
            target_task_prompt="p", target_specification="s", num_cases=999
        )

    # Code is the http_422 sentinel; message carries the structured detail.
    assert exc.value.code == "http_422"
    assert "num_cases" in exc.value.message
    assert "value is greater than 50" in exc.value.message
    assert "target_specification" in exc.value.message


@pytest.mark.asyncio
async def test_generate_422_with_no_detail_returns_generic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SDK's HTTPValidationError accepts an Unset detail; we render a
    # generic message rather than crashing.
    parsed = HTTPValidationError(detail=UNSET)  # type: ignore[arg-type]
    _patch_generate(monkeypatch, AsyncMock(return_value=_err_response(422, parsed)))

    with pytest.raises(SyntheticUserRequestError) as exc:
        await _make_client().generate(
            target_task_prompt="p", target_specification="s", num_cases=1
        )

    assert exc.value.code == "http_422"
    assert "no detail" in exc.value.message.lower()


# ───────────────────────── unparseable / unexpected ─────────────────────────


@pytest.mark.asyncio
async def test_generate_unparseable_4xx_falls_back_to_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """parsed=None on a 4xx (SDK didn't recognize the body) — generic mapping."""
    _patch_generate(monkeypatch, AsyncMock(return_value=_err_response(400, None)))

    with pytest.raises(SyntheticUserRequestError) as exc:
        await _make_client().generate(
            target_task_prompt="p", target_specification="s", num_cases=1
        )

    assert exc.value.code == "http_400"
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_generate_unparseable_5xx_falls_back_to_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_generate(monkeypatch, AsyncMock(return_value=_err_response(503, None)))

    with pytest.raises(SyntheticUserServerError) as exc:
        await _make_client().generate(
            target_task_prompt="p", target_specification="s", num_cases=1
        )

    assert exc.value.code == "http_503"
    assert exc.value.status_code == 503


# ───────────────────────── no retry surface ─────────────────────────


@pytest.mark.asyncio
async def test_generate_does_not_retry_on_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unlike the v1 client, /generate has no retry loop — a 502 surfaces
    immediately as a per-batch failure.
    """
    parsed = GenerateV1SyntheticUserGeneratePostResponse502(
        message="boom",
        code=GenerateV1SyntheticUserGeneratePostResponse502Code.LLM_UNAVAILABLE,
    )
    mock = AsyncMock(return_value=_err_response(502, parsed))
    _patch_generate(monkeypatch, mock)

    with pytest.raises(SyntheticUserServerError):
        await _make_client().generate(
            target_task_prompt="p", target_specification="s", num_cases=1
        )

    # Exactly one call — no retry budget consumed.
    assert mock.await_count == 1
