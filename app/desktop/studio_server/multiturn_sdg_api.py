"""FastAPI routes for multi-turn synthetic data generation.

Two routes wrap the runner so the web UI can drive it without a
Python REPL:

  POST /api/projects/{project_id}/tasks/{task_id}/multiturn_sdg/generate_cases
       Synchronous JSON. Calls kiln_server `/generate` via the local
       SyntheticUserClient and returns the N cases as the SDK shape
       (`{seed_prompt, synthetic_user_info: <tagged blob>}` per case).

  POST /api/projects/{project_id}/tasks/{task_id}/multiturn_sdg/run_cases_batch
       SSE stream. Takes (possibly edited) cases + run config + SU driver
       config, runs the drive loop concurrently across cases, and emits
       BatchEvent frames as `data:` lines. Terminator is
       `data: complete\\n\\n`, matching the eval_api SSE convention.

Both routes guard `task.turn_mode == TurnMode.multiturn` before doing any
upstream work — the runner depends on multi-turn TaskRun chaining
(parent_task_run_id is rejected on single-turn tasks).

The kiln_server API key is read server-side (`get_copilot_api_key`) and
never crosses to the browser, matching the copilot pattern. The SU
driver model is exposed to the caller because the choice of model
affects probe quality and cost.
"""

import asyncio
import dataclasses
import json
import logging
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.responses import StreamingResponse
from kiln_ai.datamodel.datamodel_enums import (
    ModelProviderName,
    TurnMode,
)
from kiln_ai.datamodel.run_config import (
    KilnAgentRunConfigProperties,
    RunConfigProperties,
    as_kiln_agent_run_config,
)
from kiln_ai.datamodel.task import Task
from kiln_ai.datamodel.usage import MessageUsage
from kiln_ai.synthetic_user.case import SyntheticUserCase as RunnerCase
from kiln_ai.synthetic_user.models import SyntheticUserDriverConfig
from kiln_ai.synthetic_user.runner import (
    CONCURRENCY,
    MAX_TURNS_DEFAULT,
    NUM_CASES_MAX,
    BatchCompletedEvent,
    BatchEvent,
    BatchStartedEvent,
    CaseCompletedEvent,
    CaseFailedEvent,
    TurnCompletedEvent,
    run_cases_batch,
)
from kiln_server.cancellable_streaming_response import CancellableStreamingResponse
from kiln_server.git_sync_decorators import build_save_context, no_write_lock
from kiln_server.task_api import task_from_id
from kiln_server.utils.agent_checks.policy import agent_policy_require_approval
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from app.desktop.studio_server.api_client.kiln_ai_server_client.models import (
    SyntheticUserCase as SdkCase,
)
from app.desktop.studio_server.eval_api import task_run_config_from_id
from app.desktop.studio_server.synthetic_user.client import (
    SyntheticUserClient,
    SyntheticUserRequestError,
    SyntheticUserServerError,
)
from app.desktop.studio_server.utils.copilot_utils import get_copilot_api_key

logger = logging.getLogger(__name__)


# Persona cases asked for per kiln_server call. kiln_server accepts at most
# 50 cases per request and writes a request's whole batch in a single model
# call, so the generated output grows with the count: 20 keeps each call
# short (faster, and less of it lost to truncation and salvage) and well
# under the server's bound. If that bound ever changes this must stay at or
# below it.
SU_CASES_PER_CALL = 20

# How many chunk calls run at once. The chunks are independent, so they
# overlap rather than running end to end; four is deliberately conservative
# against provider rate limits.
SU_CALLS_IN_FLIGHT = 4


# ───────────────────────── Pydantic API models ─────────────────────────

# Cases ride the wire as `list[dict[str, Any]]`: the kiln_server SDK
# emits cases as attrs models with `to_dict()` (used by `/generate_cases`
# below) and the libs/core runner consumes `SyntheticUserCase` (Pydantic).
# Both are field-identical; this route validates dicts straight into the
# libs/core type via Pydantic. Trade-off: TS bindings type cases as
# `Record<string, unknown>` instead of getting per-field autocomplete.
SyntheticUserCaseDict = dict[str, Any]
_CASE_DICT_DESCRIPTION = (
    "A SyntheticUserCase. Shape: {seed_prompt: str, synthetic_user_info: str, "
    "scenario_index?: int | null}. The synthetic_user_info value is an "
    "XML-tagged blob: "
    "<persona>...</persona><goal>...</goal><behavior_guidance>...</behavior_guidance>. "
    "Parsed client-side by kiln_ai.synthetic_user.parser. scenario_index is "
    "set only on scenario batches (generate_cases with case_prompts) and maps "
    "the case back to its plan prompt."
)


class GenerateCasesApiInput(BaseModel):
    target_specification: str = Field(..., min_length=1)
    num_cases: int = Field(..., ge=1, le=NUM_CASES_MAX)
    case_prompts: list[str] | None = Field(
        default=None,
        description=(
            "Optional per-case scenario prompts (e.g. from an approved batch "
            "plan). When provided, case i is designed around prompt i and "
            "each returned case carries scenario_index. Under the upstream "
            "salvage contract a flaky case is dropped rather than failing "
            "the batch, so the response may hold fewer cases than prompts — "
            "scenario_index, not position, maps a case to its prompt. Length "
            "must equal num_cases."
        ),
    )

    @model_validator(mode="after")
    def _case_prompts_match_num_cases(self) -> "GenerateCasesApiInput":
        if self.case_prompts is not None:
            if len(self.case_prompts) != self.num_cases:
                raise ValueError(
                    "case_prompts length must equal num_cases "
                    f"({len(self.case_prompts)} != {self.num_cases})."
                )
            if any(not p.strip() for p in self.case_prompts):
                raise ValueError("case_prompts entries must be non-empty.")
        return self


class GenerateCasesApiOutput(BaseModel):
    cases: list[SyntheticUserCaseDict] = Field(..., description=_CASE_DICT_DESCRIPTION)


class SyntheticUserDriverSpec(BaseModel):
    """How to drive the synthetic user. Caller controls because probe
    quality and cost both depend on the model.
    """

    model_name: str = Field(..., min_length=1)
    model_provider: ModelProviderName


class TargetRunConfigFields(BaseModel):
    """The target-config half of every drive request — inherited by both the
    multi-turn batch/pipeline requests and the single-turn pipeline request,
    so the two drive contracts can't drift."""

    target_run_config: RunConfigProperties | None = Field(
        default=None,
        description=(
            "Inline run config for the target task, used verbatim — the "
            "same full properties shape a manual run sends, tools included. "
            "For driving a config that isn't worth saving (ad-hoc "
            "experiments, scripting). Must be a Kiln agent config. Exactly "
            "one of target_run_config / target_run_config_id is required."
        ),
    )
    target_run_config_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "ID of one of the target task's saved run configs. The drive "
            "uses the saved config verbatim — model, prompt, sampling, and "
            "tools — so the agent under test behaves exactly like a manual "
            "run, and driven runs attribute back to the config. Exactly one "
            "of target_run_config / target_run_config_id is required."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_target_config(self) -> Self:
        if (self.target_run_config is None) == (self.target_run_config_id is None):
            raise ValueError(
                "Provide exactly one of target_run_config or target_run_config_id."
            )
        return self


class RunCasesBatchApiInput(TargetRunConfigFields):
    cases: list[SyntheticUserCaseDict] = Field(
        ...,
        min_length=1,
        max_length=NUM_CASES_MAX,
        description=(
            f"Cases as returned by /generate_cases, optionally edited. "
            f"{_CASE_DICT_DESCRIPTION}"
        ),
    )
    turns: int = Field(
        default=MAX_TURNS_DEFAULT,
        ge=1,
        le=20,
        description="Ceiling on the assistant turns produced per case.",
    )
    su_driver: SyntheticUserDriverSpec
    batch_tag: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]+$",
        min_length=1,
        max_length=64,
        description=(
            "Optional user-supplied batch label. Constrained to "
            "[A-Za-z0-9_-]{1,64} so it can safely be used as a tag on leaf "
            "TaskRuns. Auto-generated if not provided."
        ),
    )


# ───────────────────────── helpers ─────────────────────────


def guard_multiturn(task: Task) -> None:
    """Reject early if the caller pointed us at a single-turn task. The
    runner's chained TaskRun shape (parent_task_run_id) is rejected on
    single-turn tasks by the datamodel validator — better to surface a
    clean 400 here than a mid-stream chain corruption.
    """
    if task.turn_mode != TurnMode.multiturn:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "task_not_multiturn",
                "message": (
                    "Multi-turn synthetic data generation requires a task with "
                    "turn_mode=multiturn."
                ),
            },
        )


def resolve_target_run_config(
    input: TargetRunConfigFields, project_id: str, task_id: str
) -> tuple[KilnAgentRunConfigProperties, str | None]:
    """The drive's target config, from whichever source the request used,
    plus the saved config's id for run attribution (None on the inline
    path — those runs are ad-hoc by definition).

    Both sources carry the FULL run config — tools included — so the driven
    task behaves exactly like a manual run of that config. Raises
    HTTPException, so callers must resolve BEFORE opening an SSE stream
    (clean 4xx, not a mid-stream error frame).
    """
    if input.target_run_config_id is not None:
        # task_run_config_from_id is the same resolver the run-config list
        # endpoint is built on, so every id the UI can offer resolves here —
        # including virtual fine-tune configs, which never appear under
        # task.run_configs().
        try:
            run_config = task_run_config_from_id(
                project_id, task_id, input.target_run_config_id
            )
        except HTTPException as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "run_config_not_found",
                    "message": (
                        "Task has no saved run config with ID "
                        f"'{input.target_run_config_id}'."
                    ),
                },
            ) from exc
        try:
            properties = as_kiln_agent_run_config(run_config.run_config_properties)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "run_config_not_agent",
                    "message": (
                        "Driving the task requires a Kiln agent run config; "
                        "the selected run config is a different type."
                    ),
                },
            ) from exc
        return properties, input.target_run_config_id
    if input.target_run_config is None:
        # Unreachable behind the request validator; a regression there is a
        # server bug, not a client error.
        raise RuntimeError("target_run_config missing despite request validation")
    try:
        return as_kiln_agent_run_config(input.target_run_config), None
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "run_config_not_agent",
                "message": (
                    "Driving the task requires a Kiln agent run config; "
                    "the inline run config is a different type."
                ),
            },
        ) from exc


def to_su_driver_config(spec: SyntheticUserDriverSpec) -> SyntheticUserDriverConfig:
    return SyntheticUserDriverConfig(
        model_name=spec.model_name,
        model_provider_name=spec.model_provider,
    )


# Maps each event dataclass to the snake_case `event` discriminator on the
# SSE frame. Keeps the wire shape stable even if the dataclass types are
# renamed later.
_EVENT_NAMES: dict[type, str] = {
    BatchStartedEvent: "batch_started",
    TurnCompletedEvent: "turn_completed",
    CaseCompletedEvent: "case_completed",
    CaseFailedEvent: "case_failed",
    BatchCompletedEvent: "batch_completed",
}


def _event_to_payload(event: BatchEvent) -> dict:
    name = _EVENT_NAMES.get(type(event))
    if name is None:
        # New dataclass added without registering it — fail loud rather
        # than silently swallowing.
        raise RuntimeError(f"Unregistered BatchEvent type: {type(event).__name__}")
    return {"event": name, **dataclasses.asdict(event)}


def _jsonable(obj: Any) -> Any:
    """json.dumps `default` handler. SSE trace frames embed `MessageUsage`
    (Pydantic) on assistant turns, which doesn't survive `dataclasses.asdict`
    recursion. The whitelist is intentionally narrow: any new Pydantic type
    on the wire must be added here explicitly, prompting a review for
    whether `model_dump()` exposes sensitive fields. Do NOT broaden to
    `hasattr(obj, "model_dump")` or `str(obj)` — silent leakage is worse
    than a loud TypeError that fails the stream.
    """
    if isinstance(obj, MessageUsage):
        return obj.model_dump()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def _to_http_exception(
    exc: SyntheticUserRequestError | SyntheticUserServerError,
) -> HTTPException:
    """Translate SyntheticUserClient's typed exceptions to HTTPExceptions.

    Returns the exception rather than raising so callers can `raise … from exc`
    at the call site — gives the type checker NoReturn semantics for free
    and avoids any chance of unbound-variable bugs after the try block.

    Status preservation: upstream's status is passed through faithfully for
    401/422 client errors and for any upstream 5xx; everything else collapses
    to a clean 400/500.
    Collapsing a 401 to 400 hides whether the operator's stored API key is
    bad vs the caller's body being malformed — both knowable distinctions
    that the consumer needs to act on.
    """
    if isinstance(exc, SyntheticUserRequestError):
        # 401 (kiln_server auth failed) and 422 (runner sent a bad body)
        # are both client-class errors but distinct causes; preserve.
        status = exc.status_code if exc.status_code in (401, 422) else 400
        return HTTPException(
            status_code=status,
            detail={"code": exc.code, "message": exc.message},
        )
    # SyntheticUserServerError: preserve the upstream 5xx (502 → 502,
    # 503 → 503, ...). Anything unrecognized falls to a clean 500.
    status = (
        exc.status_code if exc.status_code and 500 <= exc.status_code < 600 else 500
    )
    return HTTPException(
        status_code=status,
        detail={"code": exc.code, "message": exc.message},
    )


def _case_chunks(
    num_cases: int, case_prompts: list[str] | None
) -> list[tuple[int, int, list[str] | None]]:
    """Split a case request into `(plan offset, count, scenarios)` chunks of at
    most SU_CASES_PER_CALL, in plan order.

    The offset is where the chunk starts in the caller's plan — it's what turns
    the chunk-relative indexes upstream returns back into plan-relative ones.
    With no plan there are no scenarios to slice, so those chunks carry a count
    only.
    """
    chunks: list[tuple[int, int, list[str] | None]] = []
    for start in range(0, num_cases, SU_CASES_PER_CALL):
        if case_prompts is None:
            chunks.append((start, min(SU_CASES_PER_CALL, num_cases - start), None))
        else:
            scenarios = case_prompts[start : start + SU_CASES_PER_CALL]
            chunks.append((start, len(scenarios), scenarios))
    return chunks


def _case_dict_at_plan_offset(case: SdkCase, start: int) -> SyntheticUserCaseDict:
    """The case as a wire dict, with scenario_index moved from chunk-relative to
    plan-relative.

    Upstream numbers each case against the scenario list its own call was given,
    so the first case of every chunk comes back as index 0; adding the chunk's
    plan offset restores the plan numbering the caller sent. A case with no
    index (a batch generated without a plan) passes through untouched — under
    salvage, position is not a scenario mapping, so an index is never invented.
    """
    case_dict = case.to_dict()
    index = case_dict.get("scenario_index")
    if isinstance(index, int):
        case_dict["scenario_index"] = index + start
    return case_dict


# ───────────────────────── route registration ─────────────────────────


def connect_multiturn_sdg_api(app: FastAPI) -> None:
    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/multiturn_sdg/generate_cases",
        tags=["Multiturn SDG"],
        summary="Generate Multi-Turn SU Cases",
        openapi_extra=agent_policy_require_approval(
            "Generate synthetic-user cases? Uses an LLM call (cost)."
        ),
    )
    async def generate_cases(
        project_id: Annotated[
            str, Path(description="ID of the project containing the target task.")
        ],
        task_id: Annotated[
            str,
            Path(
                description=("ID of the target task. Must be a multi-turn task."),
            ),
        ],
        input: GenerateCasesApiInput,
    ) -> GenerateCasesApiOutput:
        task = task_from_id(project_id, task_id)
        guard_multiturn(task)

        api_key = get_copilot_api_key()
        client = SyntheticUserClient(api_key=api_key)

        # The batch goes upstream in chunks, not in one request. kiln_server
        # accepts at most 50 cases per call and writes a call's whole batch in
        # a single model call, so one big ask both breaks that bound and runs
        # long. The chunks are independent, so they run concurrently, and each
        # chunk's cases come back numbered against the slice that chunk was
        # given — the chunk's plan offset is added back so every index on the
        # wire is plan-relative. Plan prompts ride as case_scenarios (case i ←
        # prompt i, salvage drops a flaky case instead of failing the chunk).
        in_flight = asyncio.Semaphore(SU_CALLS_IN_FLIGHT)

        async def generate_chunk(
            start: int, count: int, scenarios: list[str] | None
        ) -> list[SyntheticUserCaseDict]:
            async with in_flight:
                chunk_cases = await client.generate(
                    target_task_prompt=task.instruction,
                    target_specification=input.target_specification,
                    num_cases=count,
                    case_scenarios=scenarios,
                )
            return [_case_dict_at_plan_offset(c, start) for c in chunk_cases]

        try:
            # gather keeps results in argument order, so the chunks stitch back
            # in plan order. A chunk's typed failure IS the request's failure:
            # it propagates out of gather and maps exactly as a single call's
            # did, rather than shipping a partial batch.
            by_chunk = await asyncio.gather(
                *(
                    generate_chunk(start, count, scenarios)
                    for start, count, scenarios in _case_chunks(
                        input.num_cases, input.case_prompts
                    )
                )
            )
        except (SyntheticUserRequestError, SyntheticUserServerError) as exc:
            raise _to_http_exception(exc) from exc

        cases = [case for chunk_cases in by_chunk for case in chunk_cases]

        if not cases:
            # Upstream promises >= 1 case or a 502; an empty 200 is a broken
            # contract. A short chunk is ordinary salvage, so the guard is on
            # the stitched batch: nothing at all came back. Surface it typed
            # rather than handing the UI an empty batch it would fail on later
            # with no visible cause.
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "upstream_invalid_output",
                    "message": "Synthetic-user generator returned no cases.",
                },
            )
        return GenerateCasesApiOutput(cases=cases)

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/multiturn_sdg/run_cases_batch",
        tags=["Multiturn SDG"],
        summary="Run Multi-Turn SU Cases Batch",
        openapi_extra=agent_policy_require_approval(
            "Run a multi-turn synthetic-user batch? Invokes the target model "
            "and the SU driver model for several turns per case (cost)."
        ),
    )
    @no_write_lock
    async def stream_run_cases_batch(
        request: Request,
        project_id: Annotated[
            str, Path(description="ID of the project containing the target task.")
        ],
        task_id: Annotated[
            str,
            Path(
                description=("ID of the target task. Must be a multi-turn task."),
            ),
        ],
        input: RunCasesBatchApiInput,
    ) -> StreamingResponse:
        # Guard + decode happen before the stream opens so the client sees
        # a clean 400 / 422 rather than a half-open text/event-stream on
        # bad input.
        task = task_from_id(project_id, task_id)
        guard_multiturn(task)

        # Parse dict → libs/core RunnerCase. Pydantic raises ValidationError
        # on missing keys or empty strings; surface as a clean 400 instead
        # of letting it explode inside the SSE generator. We go straight to
        # the libs/core type (skipping the SDK round-trip) because the two
        # shapes are field-identical and the runner only needs the libs/core
        # one.
        try:
            runner_cases = [RunnerCase.model_validate(c) for c in input.cases]
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_case_shape",
                    "message": f"Could not parse cases against the runner shape: {exc}",
                },
            ) from exc

        target_run_config, target_run_config_id = resolve_target_run_config(
            input, project_id, task_id
        )
        su_driver_config = to_su_driver_config(input.su_driver)
        save_context = build_save_context(request)

        async def event_generator():
            try:
                async for event in run_cases_batch(
                    cases=runner_cases,
                    target_task=task,
                    target_run_config=target_run_config,
                    su_driver_config=su_driver_config,
                    turns=input.turns,
                    concurrency=CONCURRENCY,
                    batch_tag=input.batch_tag,
                    save_context=save_context,
                    task_run_config_id=target_run_config_id,
                ):
                    yield (
                        "data: "
                        + json.dumps(
                            _event_to_payload(event),
                            default=_jsonable,
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )
            except Exception as e:
                # The catch is narrow in practice: run_cases_batch
                # swallows per-case failures into CaseFailedEvent, so the
                # only paths that escape here are developer bugs
                # (RuntimeError from _event_to_payload, TypeError from
                # _jsonable). asyncio.CancelledError is BaseException and
                # bypasses this except — correct, since cancellation
                # means the consumer is gone.
                logger.exception("multiturn_sdg run_cases_batch failed mid-stream")
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "event": "batch_failed",
                            # Stable wire code; class name goes in message
                            # so it stays useful for debug without leaking
                            # internal type names onto the wire contract.
                            "error_code": "internal_error",
                            "message": f"{type(e).__name__}: {e}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
            yield "data: complete\n\n"

        return CancellableStreamingResponse(
            content=event_generator(),
            media_type="text/event-stream",
        )
