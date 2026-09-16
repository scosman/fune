"""Tests for the multiturn_sdg FastAPI routes.

`task_from_id` / `get_copilot_api_key` / `SyntheticUserClient` /
`run_cases_batch` are patched per-test so no real network or filesystem
work happens. For SSE tests we patch `run_cases_batch` to yield canned
BatchEvents and assert the serialized `data:` frames match the expected
event schema.
"""

import asyncio
import json
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kiln_ai.datamodel.datamodel_enums import (
    ModelProviderName,
    StructuredOutputMode,
    TurnMode,
)
from kiln_ai.datamodel.run_config import (
    KilnAgentRunConfigProperties,
    McpRunConfigProperties,
    MCPToolReference,
    ToolsRunConfig,
)
from kiln_ai.datamodel.task import Task
from kiln_ai.datamodel.usage import MessageUsage
from kiln_ai.synthetic_user.runner import (
    NUM_CASES_MAX,
    BatchCompletedEvent,
    BatchStartedEvent,
    CaseCompletedEvent,
    CaseFailedEvent,
    TurnCompletedEvent,
)
from kiln_server.cancellable_streaming_response import CancellableStreamingResponse
from kiln_server.custom_errors import connect_custom_errors

from app.desktop.studio_server.api_client.kiln_ai_server_client.models import (
    SyntheticUserCase,
)
from app.desktop.studio_server.multiturn_sdg_api import (
    SU_CALLS_IN_FLIGHT,
    SU_CASES_PER_CALL,
    connect_multiturn_sdg_api,
)
from app.desktop.studio_server.synthetic_user.client import (
    SyntheticUserRequestError,
    SyntheticUserServerError,
)

# ───────────────────────── fixtures ─────────────────────────


@pytest.fixture
def app() -> FastAPI:
    # connect_custom_errors mirrors production: kiln_server.make_app() registers a
    # global HTTPException handler that rewraps `detail` as `{"message": detail}`,
    # so our structured `{"code", "message"}` detail dicts arrive on the wire as
    # `{"message": {"code": ..., "message": ...}}`. Tests must mount the same
    # handler or they assert a wire shape production never ships.
    app = FastAPI()
    connect_custom_errors(app)
    connect_multiturn_sdg_api(app)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _multiturn_task() -> Mock:
    task = Mock(spec=Task)
    task.name = "support_agent"
    task.instruction = "You are a customer support agent."
    task.turn_mode = TurnMode.multiturn
    return task


def _single_turn_task() -> Mock:
    task = Mock(spec=Task)
    task.name = "single_turn_task"
    task.instruction = "Do one thing."
    task.turn_mode = TurnMode.single_turn
    return task


@pytest.fixture
def patch_task_from_id():
    with patch("app.desktop.studio_server.multiturn_sdg_api.task_from_id") as m:
        yield m


@pytest.fixture
def patch_api_key():
    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.get_copilot_api_key",
        return_value="test-key",
    ):
        yield


@pytest.fixture
def patch_eval_api_task_from_id():
    """The saved-run-config resolver (task_run_config_from_id) loads the task
    through eval_api's own task_from_id import — patch it alongside ours."""
    with patch("app.desktop.studio_server.eval_api.task_from_id") as m:
        yield m


def _sdk_cases(n: int, with_indices: bool = False) -> list[SyntheticUserCase]:
    return [
        SyntheticUserCase(
            seed_prompt=f"seed-{i}",
            synthetic_user_info=(
                f"<persona>persona-{i}</persona>"
                f"<goal>goal-{i}</goal>"
                f"<behavior_guidance>guide-{i}</behavior_guidance>"
            ),
            scenario_index=i if with_indices else None,
        )
        for i in range(n)
    ]


def _chunk_sdk_cases(
    scenarios: list[str] | None, num_cases: int
) -> list[SyntheticUserCase]:
    """What kiln_server hands back for ONE chunk: cases numbered from 0 against
    the scenario slice that chunk was given, not against the caller's plan.
    Seed prompts echo the scenario so a stitched response can be checked for
    order as well as for index."""
    labels = scenarios if scenarios is not None else [str(i) for i in range(num_cases)]
    return [
        SyntheticUserCase(
            seed_prompt=f"seed-{label}",
            synthetic_user_info=(
                f"<persona>persona-{label}</persona>"
                f"<goal>goal-{label}</goal>"
                f"<behavior_guidance>guide-{label}</behavior_guidance>"
            ),
            scenario_index=i if scenarios is not None else None,
        )
        for i, label in enumerate(labels)
    ]


def _chunked_generate() -> AsyncMock:
    """A `client.generate` stub that answers every chunk in full."""

    async def _generate(
        *, num_cases: int, case_scenarios: list[str] | None = None, **_: Any
    ) -> list[SyntheticUserCase]:
        return _chunk_sdk_cases(case_scenarios, num_cases)

    return AsyncMock(side_effect=_generate)


def _generate_cases_body(num: int = 3) -> dict:
    return {
        "target_specification": "agent waives policy under pressure",
        "num_cases": num,
    }


def _plan_body(count: int) -> dict:
    """A generate_cases body carrying a `count`-item approved plan."""
    body = _generate_cases_body(num=count)
    body["case_prompts"] = [f"scenario-{i}" for i in range(count)]
    return body


def _run_cases_batch_body(num: int = 3) -> dict:
    return {
        "cases": [
            {
                "seed_prompt": f"seed-{i}",
                "synthetic_user_info": (
                    f"<persona>persona-{i}</persona>"
                    f"<goal>goal-{i}</goal>"
                    f"<behavior_guidance>guide-{i}</behavior_guidance>"
                ),
            }
            for i in range(num)
        ],
        "turns": 3,
        # The inline run config is the FULL properties shape a manual run
        # sends — tools and sampling ride along, nothing is rebuilt.
        "target_run_config": {
            "model_name": "gpt_5_5",
            "model_provider_name": "openrouter",
            "prompt_id": "simple_prompt_builder",
            "structured_output_mode": "default",
        },
        "su_driver": {
            "model_name": "claude_4_5_haiku",
            "model_provider": "openrouter",
        },
        "batch_tag": "testbatch",
    }


def _parse_sse(response_text: str) -> list[dict | str]:
    """SSE → list of decoded events. JSON frames → dict; `complete` → string."""
    events: list[dict | str] = []
    for line in response_text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "complete":
            events.append("complete")
            continue
        events.append(json.loads(payload))
    return events


# ───────────────────────── generate_cases ─────────────────────────


def test_generate_cases_happy_path(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    patch_task_from_id.return_value = _multiturn_task()
    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = AsyncMock(return_value=_sdk_cases(3))

        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=_generate_cases_body(num=3),
        )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["cases"]) == 3
    # Cases ride the wire as the SDK shape (seed_prompt + opaque blob).
    case0 = body["cases"][0]
    assert case0["seed_prompt"] == "seed-0"
    assert "<persona>persona-0</persona>" in case0["synthetic_user_info"]


def test_generate_cases_rejects_single_turn_task_with_400(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    patch_task_from_id.return_value = _single_turn_task()

    resp = client.post(
        "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
        json=_generate_cases_body(),
    )

    assert resp.status_code == 400
    assert resp.json()["message"]["code"] == "task_not_multiturn"


def test_generate_cases_does_not_call_upstream_when_guard_fails(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    patch_task_from_id.return_value = _single_turn_task()
    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=_generate_cases_body(),
        )
        MockClient.assert_not_called()


def test_generate_cases_server_error_surfaces_with_status(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """SyntheticUserServerError with status_code=502 → 502 response."""
    patch_task_from_id.return_value = _multiturn_task()
    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = AsyncMock(
            side_effect=SyntheticUserServerError(
                "llm_unavailable", "upstream timed out", status_code=502
            )
        )

        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=_generate_cases_body(),
        )

    assert resp.status_code == 502
    assert resp.json()["message"]["code"] == "llm_unavailable"


def test_generate_cases_request_error_surfaces_as_400(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    patch_task_from_id.return_value = _multiturn_task()
    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = AsyncMock(
            side_effect=SyntheticUserRequestError(
                "unsupported_model", "no such model", status_code=400
            )
        )

        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=_generate_cases_body(),
        )

    assert resp.status_code == 400
    assert resp.json()["message"]["code"] == "unsupported_model"


def test_generate_cases_validates_num_cases_upper_bound(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """Pydantic should reject NUM_CASES_MAX + 1 before reaching the body."""
    patch_task_from_id.return_value = _multiturn_task()
    resp = client.post(
        "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
        json={"target_specification": "spec", "num_cases": NUM_CASES_MAX + 1},
    )
    assert resp.status_code == 422


def test_generate_cases_accepts_num_cases_at_upper_bound(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """The bound is inclusive: a full-size batch is a valid request. It is far
    past what one upstream call takes, so it goes up as chunks like any other
    oversized ask — and the caller still gets the full count back."""
    patch_task_from_id.return_value = _multiturn_task()
    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = _chunked_generate()

        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=_generate_cases_body(num=NUM_CASES_MAX),
        )

    assert resp.status_code == 200, resp.text
    assert len(resp.json()["cases"]) == NUM_CASES_MAX
    assert [c.kwargs["num_cases"] for c in instance.generate.await_args_list] == [
        SU_CASES_PER_CALL
    ] * (NUM_CASES_MAX // SU_CASES_PER_CALL)


# ─────────────── generate_cases with per-case prompts (batch plan) ───────────────


def test_generate_cases_with_case_prompts_under_chunk_size_makes_one_call(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """A plan that fits in a single chunk rides as case_scenarios on one
    upstream call — the spec is passed through untouched (scenario composition
    happens server-side)."""
    patch_task_from_id.return_value = _multiturn_task()
    prompts = ["scenario A", "scenario B", "scenario C"]

    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = AsyncMock(return_value=_sdk_cases(3, with_indices=True))

        body = _generate_cases_body(num=3)
        body["case_prompts"] = prompts
        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=body,
        )

    assert resp.status_code == 200
    cases = resp.json()["cases"]
    assert [c["seed_prompt"] for c in cases] == ["seed-0", "seed-1", "seed-2"]
    assert [c["scenario_index"] for c in cases] == [0, 1, 2]
    assert instance.generate.await_count == 1
    call = instance.generate.await_args
    assert call.kwargs["case_scenarios"] == prompts
    assert call.kwargs["num_cases"] == 3
    assert call.kwargs["target_specification"] == "agent waives policy under pressure"


def test_generate_cases_salvaged_batch_keeps_scenario_index(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """A scenario batch may come back short (upstream salvage) — the response
    passes the survivors through with their scenario_index mapping intact."""
    patch_task_from_id.return_value = _multiturn_task()
    survivors = _sdk_cases(3, with_indices=True)
    del survivors[1]  # scenario 1's case degraded upstream

    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = AsyncMock(return_value=survivors)

        body = _generate_cases_body(num=3)
        body["case_prompts"] = ["a", "b", "c"]
        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=body,
        )

    assert resp.status_code == 200
    cases = resp.json()["cases"]
    assert [c["scenario_index"] for c in cases] == [0, 2]


def test_generate_cases_case_prompts_length_mismatch_is_422(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    patch_task_from_id.return_value = _multiturn_task()
    body = _generate_cases_body(num=3)
    body["case_prompts"] = ["only one prompt"]
    resp = client.post(
        "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
        json=body,
    )
    assert resp.status_code == 422


def test_generate_cases_blank_case_prompt_is_422(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    patch_task_from_id.return_value = _multiturn_task()
    body = _generate_cases_body(num=2)
    body["case_prompts"] = ["real scenario", "   "]
    resp = client.post(
        "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
        json=body,
    )
    assert resp.status_code == 422


def test_generate_cases_scenario_batch_upstream_error_passes_through_typed(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """An upstream chunk's typed failure IS the request's failure (no partial
    batch on the wire)."""
    patch_task_from_id.return_value = _multiturn_task()
    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = AsyncMock(
            side_effect=SyntheticUserServerError(
                "llm_unavailable", "upstream timed out", status_code=502
            )
        )

        body = _generate_cases_body(num=3)
        body["case_prompts"] = ["a", "b", "c"]
        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=body,
        )

    assert resp.status_code == 502
    assert resp.json()["message"]["code"] == "llm_unavailable"


# ─────────────── generate_cases chunking (plans over one call) ───────────────


def test_generate_cases_splits_a_full_plan_into_chunks(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """kiln_server caps a request at 50 cases, so the builder's default plan
    cannot go up whole: it is split into consecutive chunks, each asking for
    exactly the slice of the plan it carries."""
    patch_task_from_id.return_value = _multiturn_task()
    body = _plan_body(80)

    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = _chunked_generate()

        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=body,
        )

    assert resp.status_code == 200, resp.text
    calls = instance.generate.await_args_list
    assert len(calls) == 4
    assert [c.kwargs["num_cases"] for c in calls] == [20, 20, 20, 20]
    assert [c.kwargs["case_scenarios"] for c in calls] == [
        body["case_prompts"][0:20],
        body["case_prompts"][20:40],
        body["case_prompts"][40:60],
        body["case_prompts"][60:80],
    ]


def test_generate_cases_last_chunk_carries_the_remainder(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """A plan that isn't a multiple of the chunk size ends in a short chunk —
    the last call asks only for what's left, never for padding."""
    patch_task_from_id.return_value = _multiturn_task()
    body = _plan_body(45)

    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = _chunked_generate()

        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=body,
        )

    assert resp.status_code == 200, resp.text
    calls = instance.generate.await_args_list
    assert [c.kwargs["num_cases"] for c in calls] == [20, 20, 5]
    assert [len(c.kwargs["case_scenarios"]) for c in calls] == [20, 20, 5]
    assert calls[2].kwargs["case_scenarios"] == body["case_prompts"][40:45]


def test_generate_cases_offsets_chunk_scenario_indexes_back_to_the_plan(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """Upstream numbers each chunk's cases from 0 against that chunk's own
    scenario slice. The response must be plan-relative, so every chunk's index
    gets its plan offset added back — and the chunks stitch in plan order."""
    patch_task_from_id.return_value = _multiturn_task()
    body = _plan_body(45)

    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = _chunked_generate()

        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=body,
        )

    assert resp.status_code == 200, resp.text
    cases = resp.json()["cases"]
    # Chunk 1's first case came back as index 0; it is scenario 20 of the plan.
    assert [c["scenario_index"] for c in cases] == list(range(45))
    # Order is plan order, and each case still sits on the scenario it was
    # written for — the index isn't just a renumbered position.
    assert [c["seed_prompt"] for c in cases] == [
        f"seed-scenario-{i}" for i in range(45)
    ]


def test_generate_cases_salvaged_chunk_does_not_shift_later_chunks(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """A chunk may come back short (upstream salvage) — that is not an error,
    and it must not renumber anything: later chunks keep their own offsets, so
    every surviving case still points at the scenario it was written for."""
    patch_task_from_id.return_value = _multiturn_task()
    body = _plan_body(45)

    async def _generate(
        *, num_cases: int, case_scenarios: list[str] | None = None, **_: Any
    ) -> list[SyntheticUserCase]:
        chunk = _chunk_sdk_cases(case_scenarios, num_cases)
        if case_scenarios is not None and case_scenarios[0] == "scenario-20":
            del chunk[5]  # the middle chunk's 6th case degraded upstream
        return chunk

    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = AsyncMock(side_effect=_generate)

        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=body,
        )

    assert resp.status_code == 200, resp.text
    cases = resp.json()["cases"]
    assert len(cases) == 44
    assert [c["scenario_index"] for c in cases] == [i for i in range(45) if i != 25]


def test_generate_cases_one_failing_chunk_fails_the_request(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """Chunking is an implementation detail, not a new partial-success mode: a
    chunk's 422 maps exactly as a single call's 422 always has."""
    patch_task_from_id.return_value = _multiturn_task()

    async def _generate(
        *, num_cases: int, case_scenarios: list[str] | None = None, **_: Any
    ) -> list[SyntheticUserCase]:
        if case_scenarios is not None and case_scenarios[0] == "scenario-20":
            raise SyntheticUserRequestError(
                "http_422", "body.num_cases: less than or equal to 50", status_code=422
            )
        return _chunk_sdk_cases(case_scenarios, num_cases)

    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = AsyncMock(side_effect=_generate)

        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=_plan_body(45),
        )

    assert resp.status_code == 422
    assert resp.json()["message"]["code"] == "http_422"
    assert "less than or equal to 50" in resp.json()["message"]["message"]


def test_generate_cases_bounds_how_many_chunks_are_in_flight(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """The chunks overlap, but only so far: the plan below needs more calls
    than the limit allows at once, and the peak must land exactly on the
    limit — proof the calls run in parallel AND that the cap holds."""
    patch_task_from_id.return_value = _multiturn_task()
    plan_size = SU_CASES_PER_CALL * (SU_CALLS_IN_FLIGHT + 1)
    live = 0
    peak = 0

    async def _generate(
        *, num_cases: int, case_scenarios: list[str] | None = None, **_: Any
    ) -> list[SyntheticUserCase]:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        # Yield so every call the semaphore lets through is counted before any
        # of them finishes and frees a slot.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        live -= 1
        return _chunk_sdk_cases(case_scenarios, num_cases)

    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = AsyncMock(side_effect=_generate)

        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=_plan_body(plan_size),
        )

    assert resp.status_code == 200, resp.text
    assert len(resp.json()["cases"]) == plan_size
    assert peak == SU_CALLS_IN_FLIGHT


def test_generate_cases_empty_upstream_case_list_is_typed_502(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """Upstream promises >= 1 case or a 502; an empty 200 must surface as a
    typed 502, not an empty batch the UI fails on later."""
    patch_task_from_id.return_value = _multiturn_task()
    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = AsyncMock(return_value=[])

        body = _generate_cases_body(num=1)
        body["case_prompts"] = ["scenario A"]
        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=body,
        )

    assert resp.status_code == 502
    assert resp.json()["message"]["code"] == "upstream_invalid_output"


# ───────────────────────── run_cases_batch (SSE) ─────────────────────────


def _sse_get(client: TestClient, body: dict | None = None):
    body = body if body is not None else _run_cases_batch_body()
    return client.post(
        "/api/projects/proj-1/tasks/task-1/multiturn_sdg/run_cases_batch",
        json=body,
    )


def test_run_cases_batch_rejects_single_turn_task(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    patch_task_from_id.return_value = _single_turn_task()
    resp = _sse_get(client)
    assert resp.status_code == 400
    assert resp.json()["message"]["code"] == "task_not_multiturn"


def test_run_cases_batch_emits_full_sse_event_stream(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """Canned BatchEvent sequence → assert wire shape matches."""
    patch_task_from_id.return_value = _multiturn_task()

    canned: list = [
        BatchStartedEvent(batch_tag="testbatch", num_cases=2),
        TurnCompletedEvent(
            case_index=0,
            turn_index=1,
            assistant_run_id="r0a",
            su_next_message="next user msg",
            cumulative_cost=0.01,
            trace=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello back"},
            ],
        ),
        CaseCompletedEvent(
            case_index=0,
            chain_run_ids=["r0a"],
            leaf_run_id="r0a",
            total_turns=1,
            total_cost=0.01,
        ),
        CaseFailedEvent(
            case_index=1,
            error_code="bad_synthetic_user_info",
            message="missing required tag",
        ),
        BatchCompletedEvent(
            successful=1, failed=1, batch_tag="testbatch", total_cost=0.01
        ),
    ]

    async def _fake_runner(**_kwargs) -> AsyncIterator:
        for ev in canned:
            yield ev

    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.run_cases_batch",
        _fake_runner,
    ):
        resp = _sse_get(client)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    assert len(events) == 6  # 5 canned events + `complete` terminator
    assert events[-1] == "complete"

    assert events[0] == {
        "event": "batch_started",
        "batch_tag": "testbatch",
        "num_cases": 2,
    }
    assert events[1]["event"] == "turn_completed"
    assert events[1]["case_index"] == 0
    assert events[1]["su_next_message"] == "next user msg"
    assert events[1]["trace"][2]["content"] == "hello back"
    # No stop_signal field on TurnCompletedEvent anymore.
    assert "stop_signal" not in events[1]

    assert events[2]["event"] == "case_completed"
    # No stop_reason field on CaseCompletedEvent anymore.
    assert "stop_reason" not in events[2]

    assert events[3] == {
        "event": "case_failed",
        "case_index": 1,
        "error_code": "bad_synthetic_user_info",
        "message": "missing required tag",
        "total_cost": 0.0,
        # A deterministic failure leaves error_type None even though a parse
        # error triggered it: error_code already names the bad input.
        "error_type": None,
    }
    assert events[4] == {
        "event": "batch_completed",
        "successful": 1,
        "failed": 1,
        "batch_tag": "testbatch",
        "total_cost": 0.01,
    }


def test_run_cases_batch_jsonable_handles_message_usage_in_trace(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """A real-shape trace can carry MessageUsage Pydantic instances on
    assistant turns; `_jsonable` must turn those into JSON without
    crashing.
    """
    patch_task_from_id.return_value = _multiturn_task()

    usage = MessageUsage(input_tokens=10, output_tokens=20, total_tokens=30, cost=0.001)
    canned = [
        BatchStartedEvent(batch_tag="tb", num_cases=1),
        TurnCompletedEvent(
            case_index=0,
            turn_index=1,
            assistant_run_id="r0",
            su_next_message="hello",
            cumulative_cost=0.001,
            trace=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hi back", "usage": usage},  # type: ignore[typeddict-unknown-key]
            ],
        ),
        BatchCompletedEvent(successful=1, failed=0, batch_tag="tb", total_cost=0.001),
    ]

    async def _fake_runner(**_kwargs) -> AsyncIterator:
        for ev in canned:
            yield ev

    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.run_cases_batch",
        _fake_runner,
    ):
        resp = _sse_get(client)

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    turn = next(
        e for e in events if isinstance(e, dict) and e.get("event") == "turn_completed"
    )
    # MessageUsage went through .model_dump() — the assistant turn's
    # `usage` key is now a plain dict, not a Pydantic instance.
    usage_payload = turn["trace"][1]["usage"]
    assert isinstance(usage_payload, dict)
    assert usage_payload["cost"] == 0.001
    assert usage_payload["input_tokens"] == 10


def test_run_cases_batch_translates_runner_failure_to_batch_failed(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """If the runner raises mid-stream (developer bug), the stream still
    terminates cleanly with batch_failed → complete.
    """
    patch_task_from_id.return_value = _multiturn_task()

    async def _exploding_runner(**_kwargs) -> AsyncIterator:
        raise RuntimeError("upstream catastrophe")
        yield  # pragma: no cover — unreachable; marks this as a generator

    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.run_cases_batch",
        _exploding_runner,
    ):
        resp = _sse_get(client)

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[-1] == "complete"
    failed_evt = next(
        e for e in events if isinstance(e, dict) and e.get("event") == "batch_failed"
    )
    # Stable wire code; class name is in the message for debug, not on
    # the contract.
    assert failed_evt["error_code"] == "internal_error"
    assert "RuntimeError" in failed_evt["message"]
    assert "upstream catastrophe" in failed_evt["message"]


def test_event_to_payload_raises_on_unregistered_event_type() -> None:
    """If a new BatchEvent dataclass is added but not registered in
    `_EVENT_NAMES`, `_event_to_payload` must fail loud at test time, not
    silently emit a malformed SSE frame in production. Locks in the
    defensive RuntimeError so a contributor adding a new event without
    updating the map fails the build instead of shipping a wire bug.
    """
    from dataclasses import dataclass

    from app.desktop.studio_server.multiturn_sdg_api import _event_to_payload

    @dataclass(frozen=True)
    class _UnregisteredEvent:
        x: int = 1

    with pytest.raises(RuntimeError, match="Unregistered BatchEvent type"):
        _event_to_payload(_UnregisteredEvent())  # type: ignore[arg-type]


def test_run_cases_batch_jsonable_typeerror_surfaces_as_batch_failed(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """A non-serializable, non-Pydantic object in trace must not corrupt
    the stream — `_jsonable` raises TypeError, the outer except converts
    to `batch_failed`. Locks in the fail-loud branch so a future widening
    of `_jsonable` (e.g., a defensive `str(obj)` fallback) doesn't sneak
    arbitrary content onto the wire.
    """
    patch_task_from_id.return_value = _multiturn_task()

    class _NonSerializable:
        pass

    canned = [
        BatchStartedEvent(batch_tag="tb", num_cases=1),
        TurnCompletedEvent(
            case_index=0,
            turn_index=1,
            assistant_run_id="r0",
            su_next_message="x",
            cumulative_cost=0.0,
            trace=[
                {"role": "user", "content": "hi"},
                # Slip a non-Pydantic, non-JSON-native object into trace.
                {
                    "role": "assistant",
                    "content": "hi back",
                    "usage": _NonSerializable(),
                },  # type: ignore[typeddict-unknown-key]
            ],
        ),
    ]

    async def _fake_runner(**_kwargs) -> AsyncIterator:
        for ev in canned:
            yield ev

    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.run_cases_batch",
        _fake_runner,
    ):
        resp = _sse_get(client)

    events = _parse_sse(resp.text)
    failed_evt = next(
        e for e in events if isinstance(e, dict) and e.get("event") == "batch_failed"
    )
    assert failed_evt["error_code"] == "internal_error"
    assert "TypeError" in failed_evt["message"]


def test_run_cases_batch_validates_empty_cases(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    patch_task_from_id.return_value = _multiturn_task()
    body = _run_cases_batch_body()
    body["cases"] = []
    resp = client.post(
        "/api/projects/proj-1/tasks/task-1/multiturn_sdg/run_cases_batch",
        json=body,
    )
    assert resp.status_code == 422


def test_run_cases_batch_validates_too_many_cases(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    patch_task_from_id.return_value = _multiturn_task()
    body = _run_cases_batch_body(num=NUM_CASES_MAX + 1)
    resp = client.post(
        "/api/projects/proj-1/tasks/task-1/multiturn_sdg/run_cases_batch",
        json=body,
    )
    assert resp.status_code == 422


def test_run_cases_batch_rejects_malformed_case_shape_with_400(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """A case missing the synthetic_user_info field → 400, not a half-open stream."""
    patch_task_from_id.return_value = _multiturn_task()
    body = _run_cases_batch_body()
    body["cases"] = [{"seed_prompt": "hi"}]  # missing synthetic_user_info
    resp = client.post(
        "/api/projects/proj-1/tasks/task-1/multiturn_sdg/run_cases_batch",
        json=body,
    )
    assert resp.status_code == 400
    assert resp.json()["message"]["code"] == "invalid_case_shape"


@pytest.mark.parametrize(
    "bad_tag",
    [
        "",  # min_length=1
        "my run",  # space
        "tag:with:colons",  # colon — tag-unsafe (delimiter in our prefix scheme)
        "weird*chars",  # punctuation
        "x" * 65,  # max_length=64
    ],
)
def test_run_cases_batch_rejects_invalid_batch_tags(
    bad_tag: str,
    client: TestClient,
    patch_task_from_id,
    patch_api_key,
) -> None:
    """The batch_tag must be `[A-Za-z0-9_-]{1,64}` — character class and
    length boundaries both enforced. Locks in the rule so a future
    loosening (e.g., adding `:`) doesn't slip past test coverage.
    """
    patch_task_from_id.return_value = _multiturn_task()
    body = _run_cases_batch_body()
    body["batch_tag"] = bad_tag
    resp = client.post(
        "/api/projects/proj-1/tasks/task-1/multiturn_sdg/run_cases_batch",
        json=body,
    )
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "good_tag",
    [
        "a",  # min boundary
        "x" * 64,  # max boundary
        "abc-def_123",  # hyphen + underscore + alphanumerics
        "ABC123",  # uppercase
    ],
)
def test_run_cases_batch_accepts_valid_batch_tags(
    good_tag: str,
    client: TestClient,
    patch_task_from_id,
    patch_api_key,
) -> None:
    """Boundary chars that should pass — locks in the accept side of the
    pattern so the test pair fully fences the contract.
    """
    patch_task_from_id.return_value = _multiturn_task()

    async def _empty_runner(**_kwargs) -> AsyncIterator:
        # `if False: yield` keeps this an async generator without ever
        # emitting; the route still wraps it in a streaming response.
        if False:
            yield  # pragma: no cover

    body = _run_cases_batch_body()
    body["batch_tag"] = good_tag
    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.run_cases_batch",
        _empty_runner,
    ):
        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/run_cases_batch",
            json=body,
        )
    assert resp.status_code == 200


# ───────────────────── target run config resolution ─────────────────────


def _saved_agent_run_config(rc_id: str = "rc-1") -> Mock:
    """A saved TaskRunConfig stand-in whose properties carry everything the
    transient spec cannot — tools, sampling, structured output mode."""
    rc = Mock()
    rc.id = rc_id
    rc.run_config_properties = KilnAgentRunConfigProperties(
        model_name="gpt_5_5",
        model_provider_name=ModelProviderName.openrouter,
        prompt_id="simple_prompt_builder",
        structured_output_mode=StructuredOutputMode.json_schema,
        temperature=0.3,
        tools_config=ToolsRunConfig(tools=["kiln_tool::add_numbers"]),
    )
    return rc


def _multiturn_task_with_run_configs(run_configs: list[Mock]) -> Mock:
    task = _multiturn_task()
    task.run_configs.return_value = run_configs
    return task


def test_run_cases_batch_rejects_both_target_config_sources(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    patch_task_from_id.return_value = _multiturn_task()
    body = _run_cases_batch_body()
    body["target_run_config_id"] = "rc-1"
    resp = _sse_get(client, body)
    assert resp.status_code == 422
    assert "exactly one" in resp.text.lower()


def test_run_cases_batch_rejects_missing_target_config_source(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    patch_task_from_id.return_value = _multiturn_task()
    body = _run_cases_batch_body()
    del body["target_run_config"]
    resp = _sse_get(client, body)
    assert resp.status_code == 422
    assert "exactly one" in resp.text.lower()


def test_run_cases_batch_uses_saved_run_config_verbatim(
    client: TestClient, patch_task_from_id, patch_eval_api_task_from_id, patch_api_key
) -> None:
    """A referenced saved run config reaches the runner as-is — tools,
    temperature, and structured output mode included, nothing rebuilt — and
    the config's id rides along for run attribution."""
    rc = _saved_agent_run_config()
    task = _multiturn_task_with_run_configs([rc])
    patch_task_from_id.return_value = task
    patch_eval_api_task_from_id.return_value = task

    captured: dict = {}

    async def _fake_runner(**kwargs) -> AsyncIterator:
        captured.update(kwargs)
        yield BatchStartedEvent(batch_tag="t", num_cases=1)
        yield BatchCompletedEvent(successful=0, failed=0, batch_tag="t", total_cost=0.0)

    body = _run_cases_batch_body()
    del body["target_run_config"]
    body["target_run_config_id"] = "rc-1"
    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.run_cases_batch",
        _fake_runner,
    ):
        resp = _sse_get(client, body)

    assert resp.status_code == 200
    assert captured["target_run_config"] is rc.run_config_properties
    assert captured["task_run_config_id"] == "rc-1"


def test_run_cases_batch_inline_config_has_no_attribution_id(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """An inline config is an ad-hoc run — no saved config to attribute to."""
    patch_task_from_id.return_value = _multiturn_task()

    captured: dict = {}

    async def _fake_runner(**kwargs) -> AsyncIterator:
        captured.update(kwargs)
        yield BatchStartedEvent(batch_tag="t", num_cases=1)
        yield BatchCompletedEvent(successful=0, failed=0, batch_tag="t", total_cost=0.0)

    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.run_cases_batch",
        _fake_runner,
    ):
        resp = _sse_get(client)

    assert resp.status_code == 200
    assert captured["task_run_config_id"] is None


def test_run_cases_batch_inline_config_carries_full_properties(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """The inline mode is the FULL properties shape — tools and sampling
    reach the runner verbatim, same fidelity as a saved config."""
    patch_task_from_id.return_value = _multiturn_task()

    captured: dict = {}

    async def _fake_runner(**kwargs) -> AsyncIterator:
        captured.update(kwargs)
        yield BatchStartedEvent(batch_tag="t", num_cases=1)
        yield BatchCompletedEvent(successful=0, failed=0, batch_tag="t", total_cost=0.0)

    body = _run_cases_batch_body()
    body["target_run_config"] = {
        "model_name": "gpt_5_5",
        "model_provider_name": "openrouter",
        "prompt_id": "simple_prompt_builder",
        "structured_output_mode": "json_schema",
        "temperature": 0.3,
        "tools_config": {"tools": ["kiln_tool::add_numbers"]},
    }
    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.run_cases_batch",
        _fake_runner,
    ):
        resp = _sse_get(client, body)

    assert resp.status_code == 200
    config = captured["target_run_config"]
    assert isinstance(config, KilnAgentRunConfigProperties)
    assert config.tools_config is not None
    assert config.tools_config.tools == ["kiln_tool::add_numbers"]
    assert config.temperature == 0.3
    assert config.structured_output_mode == StructuredOutputMode.json_schema


def test_run_cases_batch_inline_mcp_config_is_400(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """An inline MCP-type config can't drive a conversation, same as the
    saved-config path — typed 400 before the stream opens."""
    patch_task_from_id.return_value = _multiturn_task()
    body = _run_cases_batch_body()
    body["target_run_config"] = {
        "type": "mcp",
        "tool_reference": {"tool_id": "mcp::local::srv::tool"},
    }
    resp = _sse_get(client, body)
    assert resp.status_code == 400
    assert resp.json()["message"]["code"] == "run_config_not_agent"


def test_run_cases_batch_unknown_run_config_id_is_404(
    client: TestClient, patch_task_from_id, patch_eval_api_task_from_id, patch_api_key
) -> None:
    task = _multiturn_task_with_run_configs([_saved_agent_run_config("other-rc")])
    patch_task_from_id.return_value = task
    patch_eval_api_task_from_id.return_value = task
    body = _run_cases_batch_body()
    del body["target_run_config"]
    body["target_run_config_id"] = "rc-1"
    resp = _sse_get(client, body)
    assert resp.status_code == 404
    assert resp.json()["message"]["code"] == "run_config_not_found"


def test_run_cases_batch_non_agent_run_config_is_400(
    client: TestClient, patch_task_from_id, patch_eval_api_task_from_id, patch_api_key
) -> None:
    """An MCP-type run config can't drive a conversation — the drive loop
    needs an agent-shaped invoker; surface a typed 400, not a crash."""
    rc = Mock()
    rc.id = "rc-1"
    rc.run_config_properties = McpRunConfigProperties(
        tool_reference=MCPToolReference(tool_id="mcp::local::srv::tool")
    )
    task = _multiturn_task_with_run_configs([rc])
    patch_task_from_id.return_value = task
    patch_eval_api_task_from_id.return_value = task
    body = _run_cases_batch_body()
    del body["target_run_config"]
    body["target_run_config_id"] = "rc-1"
    resp = _sse_get(client, body)
    assert resp.status_code == 400
    assert resp.json()["message"]["code"] == "run_config_not_agent"


def test_generate_cases_preserves_upstream_401_status(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """A 401 from kiln_server means our stored API key is bad — surface
    as 401, not collapsed to a generic 400. Operator-config problem
    surface vs caller-input problem surface should stay distinct.
    """
    patch_task_from_id.return_value = _multiturn_task()
    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = AsyncMock(
            side_effect=SyntheticUserRequestError(
                "unauthorized", "bad api key", status_code=401
            )
        )

        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=_generate_cases_body(),
        )

    assert resp.status_code == 401
    assert resp.json()["message"]["code"] == "unauthorized"


def test_generate_cases_preserves_upstream_422_status(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """A 422 from kiln_server (runner sent a body the validator rejected)
    is a different beast from a caller's-input 400 — preserve.
    """
    patch_task_from_id.return_value = _multiturn_task()
    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = AsyncMock(
            side_effect=SyntheticUserRequestError(
                "http_422", "body.target_specification: too long", status_code=422
            )
        )

        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=_generate_cases_body(),
        )

    assert resp.status_code == 422
    assert resp.json()["message"]["code"] == "http_422"


def test_run_cases_batch_uses_cancellable_streaming_response(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """Verifies the route wraps its generator in CancellableStreamingResponse.
    Without this, browser disconnects don't reach the runner and in-flight
    case tasks keep burning LLM calls until they finish. Matches the chat
    SSE route's `test_uses_cancellable_streaming_response` discipline.
    """
    patch_task_from_id.return_value = _multiturn_task()

    async def _empty_runner(**_kwargs) -> AsyncIterator:
        if False:
            yield  # pragma: no cover

    with (
        patch(
            "app.desktop.studio_server.multiturn_sdg_api.run_cases_batch",
            _empty_runner,
        ),
        patch(
            "app.desktop.studio_server.multiturn_sdg_api.CancellableStreamingResponse",
            wraps=CancellableStreamingResponse,
        ) as mock_cls,
    ):
        resp = _sse_get(client)
        _ = resp.content

    assert resp.status_code == 200
    mock_cls.assert_called_once()


def test_run_cases_batch_has_no_write_lock_decorator(app: FastAPI) -> None:
    """The SSE route must be @no_write_lock so the git_sync middleware
    doesn't wrap the entire streaming response in one atomic_write
    (which would block all other writes for the batch duration).
    """
    path = "/api/projects/{project_id}/tasks/{task_id}/multiturn_sdg/run_cases_batch"
    for route in app.routes:
        if getattr(route, "path", None) == path and "POST" in getattr(
            route, "methods", set()
        ):
            assert getattr(route.endpoint, "_git_sync_no_write_lock", False), (
                f"POST {path} must be @no_write_lock"
            )
            return
    raise AssertionError(f"POST {path} route not found")


def test_generate_cases_preserves_upstream_503_status(
    client: TestClient, patch_task_from_id, patch_api_key
) -> None:
    """An unexpected 5xx (503) should not silently collapse to 500."""
    patch_task_from_id.return_value = _multiturn_task()
    with patch(
        "app.desktop.studio_server.multiturn_sdg_api.SyntheticUserClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.generate = AsyncMock(
            side_effect=SyntheticUserServerError(
                "http_503", "upstream unavailable", status_code=503
            )
        )

        resp = client.post(
            "/api/projects/proj-1/tasks/task-1/multiturn_sdg/generate_cases",
            json=_generate_cases_body(),
        )

    assert resp.status_code == 503
    assert resp.json()["message"]["code"] == "http_503"
