import asyncio
import csv
import io
import json
import logging
import random
from http import HTTPStatus
from typing import Annotated

import httpx
import jsonschema
from fastapi import FastAPI, File, HTTPException, Path, UploadFile
from kiln_ai.datamodel import ClaimReview, Feedback, TaskRun
from kiln_ai.datamodel.basemodel import FilenameStringShort
from kiln_ai.datamodel.datamodel_enums import EvalStatus, Priority
from kiln_ai.datamodel.eval import (
    Eval,
    EvalConfig,
    EvalConfigType,
    EvalDataType,
    EvalInput,
    EvalInputSplit,
    LlmJudgeProperties,
    MultiTurnDriveConfig,
    TaskRunSplit,
)
from kiln_ai.datamodel.json_schema import validate_schema
from kiln_ai.datamodel.spec import (
    Spec,
    SpecStatus,
    SyntheticDataGenerationSessionConfig,
    SyntheticDataGenerationStepConfig,
    TaskSample,
)
from kiln_ai.datamodel.spec_properties import SpecProperties
from kiln_ai.datamodel.task_output import TaskOutputRating
from kiln_ai.utils.name_generator import generate_memorable_name
from kiln_server.task_api import task_from_id
from kiln_server.utils.agent_checks.policy import (
    ALLOW_AGENT,
    agent_policy_require_approval,
)
from kiln_server.utils.spec_utils import (
    generate_spec_eval_tags,
    spec_eval_data_type,
    spec_eval_output_score,
    spec_eval_template,
    tag_filter_id,
)
from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self

from app.desktop.studio_server.api_client.kiln_ai_server_client.api.copilot import (
    clarify_spec_v1_copilot_clarify_spec_post,
    generate_batch_v1_copilot_generate_batch_post,
    question_spec_v1_copilot_question_spec_post,
    refine_spec_v1_copilot_refine_spec_post,
    refine_spec_with_answers_and_name_v1_copilot_refine_spec_with_answers_and_name_post,
    refine_spec_with_answers_v1_copilot_refine_spec_with_answers_post,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.api.jobs import (
    get_data_guide_job_result_v1_jobs_data_guide_job_job_id_result_get,
    get_job_status_v1_jobs_job_type_job_id_status_get,
    start_data_guide_job_v1_jobs_data_guide_job_start_post,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.client import (
    AuthenticatedClient,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models import (
    ClarifySpecInput,
    ClarifySpecOutput,
    DraftInputDataGuideInput,
    GenerateBatchInput,
    GenerateBatchOutput,
    JobStatus,
    JobType,
    RefineSpecInput,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models import (
    QuestionSet as QuestionSetServerApi,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models import (
    RefineSpecApiOutput as RefineSpecApiOutputClient,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models import (
    RefineSpecFromAnswersAndNameOutput as RefineSpecFromAnswersAndNameOutputClient,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models import (
    SpecQuestionerApiInput as SpecQuestionerApiInputServerApi,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models import (
    SubmitAnswersRequest as SubmitAnswersRequestServerApi,
)
from app.desktop.studio_server.api_client.kiln_server_client import (
    get_authenticated_client,
)
from app.desktop.studio_server.api_models.copilot_models import (
    DRAFT_INPUT_DATA_GUIDE_MAX_EXAMPLE_LENGTH,
    ClarifySpecApiInput,
    ClarifySpecApiOutput,
    DataGuideJobResultApiOutput,
    DataGuideJobStatusApiOutput,
    DrivenSyntheticCaseApi,
    GenerateBatchApiInput,
    GenerateBatchApiOutput,
    ParseImportFileApiOutput,
    RefineSpecApiInput,
    ReviewedChainApi,
    ReviewedExample,
    SpecQuestionerApiInput,
    StartDataGuideJobApiInput,
    StartDataGuideJobApiOutput,
    SyntheticDataGenerationSessionConfigApi,
    TaskInfoApi,
)
from app.desktop.studio_server.api_models.eval_builder_models import (
    JudgeConfig,
    spec_name_must_have_a_json_key,
)
from app.desktop.studio_server.data_gen_api import (
    _resolve_task_runtime_prompt,
)
from app.desktop.studio_server.utils.copilot_utils import (
    SingleTurnDataset,
    build_multi_turn_eval_inputs,
    build_single_turn_batch_eval_inputs,
    create_single_turn_dataset,
    find_multi_turn_chain_leaves,
    find_single_turn_batch_runs,
    generate_copilot_examples,
    get_copilot_api_key,
    persist_eval_slice,
    rate_reviewed_batch_runs,
    split_and_tag_batch_runs,
    task_capabilities_for_task,
    task_info_payload,
    unrate_reviewed_batch_runs,
    untag_batch_runs_for_eval,
)
from app.desktop.studio_server.utils.eval_builder_utils import (
    build_judge_prompt_template,
)
from app.desktop.studio_server.utils.response_utils import (
    unwrap_response,
    upstream_route_missing,
    upstream_unreachable,
)
from libs.core.kiln_ai.datamodel.copilot_models.questions import (
    QuestionSet,
    RefineSpecApiOutput,
    SubmitAnswersRequest,
)

logger = logging.getLogger(__name__)


async def copilot_passthrough_payload(
    input: ClarifySpecApiInput | RefineSpecApiInput | SpecQuestionerApiInput,
) -> dict:
    """The kiln_server payload for a copilot route that forwards a client body.

    When the client names a project and task, the tools and skills of one of
    the task's run configs are read from local storage and attached to
    target_task_info so the copilot prompts can see what the target task can
    actually do. The client picks the config with run_config_id, or gets the
    task's default. A client that already sent capabilities keeps them. The
    ids are always stripped: they identify local storage and mean nothing to
    kiln_server.
    """
    task_info = input.target_task_info
    # Presence, not truthiness: the model already rejects a half-supplied pair,
    # so an empty id is a real (bad) id and belongs in the lookup below.
    if input.project_id is not None and input.task_id is not None:
        # A bad id 404s here, which is the honest answer: the caller asked for
        # this task's capabilities and we cannot produce them.
        task = task_from_id(input.project_id, input.task_id)
        task_tools, task_skills = await task_capabilities_for_task(
            task, input.run_config_id
        )
        task_info = task_info.model_copy(
            update={
                "task_tools": (
                    task_info.task_tools
                    if task_info.task_tools is not None
                    else task_tools
                ),
                "task_skills": (
                    task_info.task_skills
                    if task_info.task_skills is not None
                    else task_skills
                ),
            }
        )
    payload = input.model_dump(exclude={"project_id", "task_id", "run_config_id"})
    payload["target_task_info"] = task_info_payload(task_info)
    return payload


class MultiTurnSaveInfo(BaseModel):
    """Identifies an existing multi-turn synthetic-user batch to turn into an Eval.

    The endpoint splits the chains tagged with this batch_tag into golden and
    train slices, and mints the eval slice as EvalInput items from `cases` —
    the re-drivable inputs the eval runner regenerates conversations from,
    per run config, using `drive_config` as the synthetic user.
    """

    batch_tag: str = Field(
        description="The batch_tag emitted by the multi-turn synthetic-user runner "
        "(see kiln_ai.synthetic_user.runner). Identifies the set of conversation "
        "chains already persisted to disk that this Eval should evaluate."
    )
    reviewed_chains: list[ReviewedChainApi] = Field(
        default_factory=list,
        description="The human's review verdicts, one per reviewed chain keyed "
        "by leaf TaskRun id. Each becomes a golden RequirementRating on the "
        "chain leaf (plus Feedback / per-claim grades when present).",
    )
    cases: list[DrivenSyntheticCaseApi] = Field(
        min_length=1,
        description="The driven synthetic-user cases of this batch. Each is "
        "minted as an EvalInput — the eval slice the runner re-drives per "
        "run config at eval time.",
    )
    drive_config: MultiTurnDriveConfig = Field(
        description="The alignment-time drive settings (synthetic-user model "
        "+ turn count), stamped on each minted EvalInput so eval-time "
        "re-drives match the conversations the judge was calibrated on.",
    )


class SingleTurnSaveInfo(BaseModel):
    """Identifies an existing single-turn pipeline batch to turn into an Eval.

    The single-turn sibling of MultiTurnSaveInfo: the endpoint splits the
    runs tagged with this batch_tag into golden and train slices (reviewed →
    golden with ratings and claim reviews, unreviewed → train), and mints
    the eval slice as inputs-only EvalInput items from `inputs`. Nothing is
    generated at save time — the dataset is the runs the user just reviewed.
    """

    batch_tag: str = Field(
        description="The batch_tag emitted by the single-turn pipeline "
        "(eval_builder single_turn_pipeline). Identifies the set of "
        "batch-tagged TaskRuns already persisted to disk that this Eval's "
        "golden/train slices are split from."
    )
    reviewed_runs: list[ReviewedChainApi] = Field(
        default_factory=list,
        description="The human's review verdicts, one per reviewed run keyed "
        "by TaskRun id (the run itself is the leaf on this arm). Each "
        "becomes a golden RequirementRating on the run (plus Feedback / "
        "per-claim grades when present).",
    )
    inputs: list[str] = Field(
        min_length=1,
        description="The generated task inputs the batch actually ran — one "
        "EvalInput each, the eval slice the runner executes fresh per run "
        "config at eval time. For tasks with an input schema, each entry is "
        "the input as a JSON string (the same encoding the pipeline ran).",
    )

    @field_validator("inputs")
    @classmethod
    def inputs_must_be_non_blank(cls, value: list[str]) -> list[str]:
        # A blank input can never be run at eval time; reject the save up
        # front instead of persisting an eval item that fails every job.
        for input_text in value:
            if not input_text.strip():
                raise ValueError("inputs must not contain empty entries.")
        return value


class CreateSpecWithCopilotRequest(BaseModel):
    """Request model for creating a spec with Kiln Copilot.

    Three synthesis paths are supported, exactly one must be set per request:

    - **Single-turn (wizard):** caller supplies `single_turn` with a
      `batch_tag` pointing at runs already on disk (created by the eval
      builder's single_turn_pipeline) plus the review verdicts and the
      generated inputs. Endpoint tags the existing runs with golden/train
      filter tags (writing the verdicts onto the golden ones) and mints one
      EvalInput per input as the eval slice; no new TaskRuns are created and
      nothing is generated. `evaluate_full_trace` must be True — the
      pipeline judged the transcript, so the saved eval must too.

    - **Multi-turn (wizard):** caller supplies `multi_turn` with a `batch_tag`
      pointing at chains already on disk (created earlier by the
      synthetic-user runner) plus the driven cases and drive settings.
      Endpoint tags the existing chain leaves with golden/train filter tags
      and mints one EvalInput per driven case as the eval slice; no new
      TaskRuns are created. `evaluate_full_trace` must be True.

    - **Legacy single-turn (v1 manual flow):** caller supplies
      `sdg_session_config`. Endpoint calls `generate_copilot_examples` for
      fresh I/O pairs, splits them into eval/train/golden datasets, and tags
      new TaskRuns.

    If you don't want copilot at all, use POST /spec instead.

    The client is responsible for building:
    - definition: the spec definition string (buildSpecDefinition on client)
    - properties: the spec properties object (filtered, with spec_type included)
    """

    # Short limit: the name becomes the eval's EvalOutputScore.name (max 32)
    # — a longer name would fail deep inside Eval construction, not here.
    name: FilenameStringShort
    # Same up-front rule as the review requests: the judge's score key derives
    # from this name, and an empty key would persist an eval that can never
    # run (every job would fail inside the judge).
    _name_has_json_key = field_validator("name")(spec_name_must_have_a_json_key)
    definition: str = Field(
        description="The spec definition string, built by client using buildSpecDefinition()"
    )
    properties: SpecProperties = Field(
        discriminator="spec_type",
        description="The spec properties object, pre-built by client with spec_type included",
    )
    evaluate_full_trace: bool = False
    reviewed_examples: list[ReviewedExample] = Field(default_factory=list)
    judge_info: JudgeConfig = Field(
        description="The judge to persist as the eval's V2 config — the same "
        "shape (and, from the builder, the same values) the review step ran, "
        "so the calibrated judge is the one that ships."
    )
    sdg_session_config: SyntheticDataGenerationSessionConfigApi | None = None
    multi_turn: MultiTurnSaveInfo | None = None
    single_turn: SingleTurnSaveInfo | None = None
    # Legacy-arm generation context only; the wizard arms generate nothing at
    # save time and omit it.
    task_prompt_with_example: str | None = None
    task_sample: TaskSample | None = None
    run_config_id: str | None = Field(
        default=None,
        description="Legacy `sdg_session_config` path only: the run config "
        "whose tools and skills describe the target task while examples are "
        "generated. Omit to use the task's default run config. The wizard "
        "arms generate nothing here, so they read no capabilities and this "
        "field does not apply to them.",
    )

    @model_validator(mode="after")
    def validate_synthesis_path(self) -> Self:
        paths_set = [
            path
            for path in (self.multi_turn, self.single_turn, self.sdg_session_config)
            if path is not None
        ]
        if len(paths_set) != 1:
            raise ValueError(
                "Pass exactly one of `single_turn` (for single-turn runs "
                "already on disk), `multi_turn` (for multi-turn chains "
                "already on disk), or `sdg_session_config` (legacy: fresh "
                "single-turn synthesis)."
            )
        # One rule for both wizard arms: the pipeline judges the transcript, so
        # the saved eval must too, or the calibrated judge is not the judge that
        # ships. A single-turn run's transcript is its one exchange.
        if (
            self.multi_turn is not None or self.single_turn is not None
        ) and not self.evaluate_full_trace:
            raise ValueError(
                "A wizard save requires `evaluate_full_trace=True` — the "
                "pipeline judged full traces, so the saved eval must too."
            )
        return self


# --- Data Guide draft job plumbing -----------------------------------------
#
# The Data Guide draft runs as a kiln_server background job so the heavy
# summarize+aggregate work happens server-side and survives a flaky
# connection. The studio server proxies the job's start / status / result
# lifecycle so the web UI owns polling and the user can leave the page and
# come back (or get nudged back via the task-wide progress widget).
#
# These go through the generated kiln_ai_server_client like the other copilot
# and job endpoints. Note the start/result endpoints live under the
# `data_guide_job` path segment, while status goes through the shared
# `/{job_type}/{job_id}/status` route keyed by `JobType.DATA_GUIDE_JOB`.


async def _start_data_guide_job(
    client: AuthenticatedClient, body: DraftInputDataGuideInput
) -> str:
    """Start the Data Guide draft job on kiln_server and return its job id.
    Raises HTTPException on failure."""
    try:
        detailed = await start_data_guide_job_v1_jobs_data_guide_job_start_post.asyncio_detailed(
            client=client,
            body=body,
        )
    except httpx.HTTPError as e:
        raise upstream_unreachable("data guide") from e

    # This request names no resource, so an upstream 404 can only mean the route
    # isn't deployed — not "your job wasn't found".
    if detailed.status_code == HTTPStatus.NOT_FOUND:
        raise upstream_route_missing("data guide drafting")

    response = unwrap_response(
        detailed,
        default_detail="Failed to start the data guide job. Please try again.",
    )
    if not response.job_id:
        raise HTTPException(
            status_code=500,
            detail="Data guide job did not return a job id.",
        )
    return response.job_id


async def _get_data_guide_job_status(client: AuthenticatedClient, job_id: str) -> str:
    """Fetch the current status of a Data Guide draft job. Raises HTTPException
    on a transport/server error.

    The status endpoint can flip to `succeeded` slightly before the draft output
    is committed and retrievable. To keep the UI honest, we hold the reported
    status at `running` until the result is actually available — so the spinner
    and the task-wide indicator stay "in progress" and the client never tries to
    fetch (and error on) an unfinished result. A job that finished but produced
    an empty draft still reports `succeeded` here: that's a real failure the
    result fetch surfaces, not an in-flight state.
    """
    try:
        detailed = (
            await get_job_status_v1_jobs_job_type_job_id_status_get.asyncio_detailed(
                job_type=JobType.DATA_GUIDE_JOB,
                job_id=job_id,
                client=client,
            )
        )
    except httpx.HTTPError as e:
        raise upstream_unreachable("data guide") from e
    # This request names a job, so an upstream 404 genuinely means that job is
    # gone — propagate it as-is rather than reporting an upstream failure.
    response = unwrap_response(
        detailed,
        default_detail="Failed to check the data guide job status.",
    )
    if response.status == JobStatus.SUCCEEDED and not await _data_guide_result_ready(
        client, job_id
    ):
        return JobStatus.RUNNING.value
    return response.status.value


async def _data_guide_result_ready(client: AuthenticatedClient, job_id: str) -> bool:
    """Whether the draft job's result endpoint reports the job as finished
    (`succeeded`) — i.e. the draft is retrievable. False while the result
    endpoint still reads as in-progress (it can lag the status endpoint). A
    transport/server error is treated as "ready" so it surfaces through the
    normal result fetch instead of pinning the UI in-progress forever."""
    try:
        detailed = await get_data_guide_job_result_v1_jobs_data_guide_job_job_id_result_get.asyncio_detailed(
            job_id=job_id,
            client=client,
        )
        response = unwrap_response(
            detailed,
            default_detail="Failed to fetch the data guide result.",
        )
        return response.status == JobStatus.SUCCEEDED
    except (HTTPException, httpx.HTTPError):
        return True


async def _get_data_guide_job_result(client: AuthenticatedClient, job_id: str) -> str:
    """Fetch the draft guide markdown from a completed Data Guide draft job.
    Raises HTTPException on failure or an empty result."""
    try:
        detailed = await get_data_guide_job_result_v1_jobs_data_guide_job_job_id_result_get.asyncio_detailed(
            job_id=job_id,
            client=client,
        )
    except httpx.HTTPError as e:
        raise upstream_unreachable("data guide") from e
    # Job-addressed, so an upstream 404 really means the job is gone: propagate.
    response = unwrap_response(
        detailed,
        default_detail="Failed to fetch the data guide result.",
    )
    # The result endpoint can return before the draft is actually available —
    # the job's status can read `succeeded` slightly ahead of the output being
    # committed. Distinguish "not finished yet" (caller should keep polling)
    # from "finished but genuinely empty" so we don't surface a misleading
    # empty-draft error while the job is still wrapping up.
    if response.status != JobStatus.SUCCEEDED:
        # 409, not 425: 425 (Too Early) is specific to TLS early data — replayable
        # bytes sent before the handshake completes — and any proxy or SDK that
        # honours it may retry on different transport terms. This is a plain
        # resource-state conflict: the job isn't finished, so a result fetch isn't
        # a request it can serve yet.
        raise HTTPException(
            status_code=409,
            detail="The data guide draft is still being generated. Please wait.",
        )
    draft_guide = getattr(response.output, "draft_guide", "") or ""
    if not isinstance(draft_guide, str) or not draft_guide.strip():
        raise HTTPException(
            status_code=500,
            detail="Copilot returned an empty draft guide.",
        )
    return draft_guide


def _finalize_import_rows(rows: list[str], too_long: int) -> ParseImportFileApiOutput:
    """Shared tail for both parsers: turn accepted rows + a skipped-for-length
    count into the response, choosing a clear error when nothing remains."""
    if not rows:
        if too_long > 0:
            return ParseImportFileApiOutput(
                rows=[],
                error=(
                    "All examples were over the "
                    f"{DRAFT_INPUT_DATA_GUIDE_MAX_EXAMPLE_LENGTH:,} character limit."
                ),
            )
        return ParseImportFileApiOutput(rows=[], error="No examples found in the file.")
    warning = None
    if too_long > 0:
        warning = (
            f"{too_long} example{'' if too_long == 1 else 's'} over the "
            f"{DRAFT_INPUT_DATA_GUIDE_MAX_EXAMPLE_LENGTH:,} character limit "
            "will be skipped."
        )
    return ParseImportFileApiOutput(rows=rows, warning=warning)


def _parse_csv_import(
    content: str, input_json_schema: str | None
) -> ParseImportFileApiOutput:
    """Parse a single-column CSV of input examples with the stdlib csv reader
    (RFC 4180 — handles quoted commas/newlines/escaped quotes).

    Plaintext tasks take each cell as the raw input. Structured-input tasks take
    each cell as a JSON object, validated against the task's input schema (author
    these in a spreadsheet so the JSON's commas/quotes are escaped on export).

    Enforces a single column: any record that parses to more than one field
    means an unescaped separator (or a genuine multi-column file). Rather than
    silently keep column one and drop the rest, we reject and tell the user to
    quote values that contain commas.
    """
    rows_with_numbers: list[tuple[int, list[str]]] = []
    for row_number, fields in enumerate(csv.reader(io.StringIO(content)), start=1):
        if all(field.strip() == "" for field in fields):
            continue  # blank record
        rows_with_numbers.append((row_number, fields))

    if not rows_with_numbers:
        return ParseImportFileApiOutput(rows=[], error="No examples found in the file.")

    # Drop an optional single-cell "input" header.
    _, first_fields = rows_with_numbers[0]
    if len(first_fields) == 1 and first_fields[0].strip().lower() == "input":
        rows_with_numbers = rows_with_numbers[1:]

    if any(len(fields) > 1 for _, fields in rows_with_numbers):
        return ParseImportFileApiOutput(
            rows=[],
            error="Invalid CSV format. Expected only one column.",
        )

    accepted: list[str] = []
    too_long = 0
    for row_number, fields in rows_with_numbers:
        value = fields[0].strip()
        if value == "":
            continue
        if input_json_schema is not None:
            # Structured task: each cell must be a JSON object matching the schema.
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return ParseImportFileApiOutput(
                    rows=[], error=f"Row {row_number} is not valid JSON."
                )
            try:
                validate_schema(parsed, input_json_schema, require_object=False)
            except jsonschema.exceptions.ValidationError as e:
                return ParseImportFileApiOutput(
                    rows=[],
                    error=(
                        f"Row {row_number} does not match the task input schema: "
                        f"{e.message}"
                    ),
                )
            value = json.dumps(parsed)
        if len(value) > DRAFT_INPUT_DATA_GUIDE_MAX_EXAMPLE_LENGTH:
            too_long += 1
            continue
        accepted.append(value)
    return _finalize_import_rows(accepted, too_long)


def _validate_structured_examples(input_json_schema: str, examples: list[str]) -> None:
    """Validate each candidate input example against the task's input JSON
    schema, raising HTTPException(422) listing every example that fails.

    For structured-input tasks each example must be a JSON value matching the
    schema. The web UI does only a shallow check at import time; this is the
    authoritative gate (full schema validation, mirroring the dataset import
    path) run before the expensive draft job kicks off.
    """
    errors: list[str] = []
    for idx, raw in enumerate(examples):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            errors.append(f"Example {idx + 1} is not valid JSON.")
            continue
        try:
            validate_schema(parsed, input_json_schema, require_object=False)
        except jsonschema.exceptions.ValidationError as e:
            errors.append(
                f"Example {idx + 1} does not match the task input schema: {e.message}"
            )
    if errors:
        raise HTTPException(status_code=422, detail=" ".join(errors))


def validate_reviewed_refs(
    reviewed_refs: list[ReviewedChainApi],
    batch_leaves: list[TaskRun],
    batch_tag: str,
) -> set[str]:
    """The review must describe the batch being saved, on either arm: every
    reviewed ref must name a run of THIS batch, each at most once — checked
    up front so a stale or malformed review fails before any models are
    created (rate_reviewed_batch_runs re-checks membership as a backstop).
    Returns the reviewed run ids — the golden-eligible set that drives the
    split."""
    leaf_ids = {leaf.id for leaf in batch_leaves if leaf.id}
    reviewed_ids = [ref.leaf_run_id for ref in reviewed_refs]
    missing = [rid for rid in reviewed_ids if rid not in leaf_ids]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Reviewed runs not found in batch '{batch_tag}': {', '.join(missing)}."
            ),
        )
    duplicates = sorted({rid for rid in reviewed_ids if reviewed_ids.count(rid) > 1})
    if duplicates:
        raise HTTPException(
            status_code=422,
            detail=(
                "Each run can be reviewed at most once; "
                f"duplicated: {', '.join(duplicates)}."
            ),
        )
    return set(reviewed_ids)


def persist_spec_save(
    *,
    eval: Eval,
    eval_config: EvalConfig,
    single_turn_dataset: SingleTurnDataset | None,
    spec: Spec,
    batch_leaves: list[TaskRun],
    batch_eval_inputs: list[EvalInput],
    reviewed_refs: list[ReviewedChainApi],
    reviewed_leaf_ids: set[str],
    train_tag: str,
    golden_tag: str,
    val_tag: str,
    spec_name: str,
    rng: random.Random,
) -> None:
    """Persist all spec models as one unit of work, rolling back every mutation
    on failure.

    Both wizard arms ride the batch-runs path: `batch_leaves` are the runs
    already on disk (multi-turn chain leaves or single-turn pipeline runs)
    to split into golden / train / val and rate, and `batch_eval_inputs` is
    the arm's built eval slice. The legacy v1 manual flow instead passes
    `single_turn_dataset` (freshly generated runs plus its own eval slice)
    and empty batch args, so it never reaches the split and mints no val
    items.

    Owns three rollback ledgers: created models (Eval / EvalConfig / TaskRun /
    EvalInput / Spec), tagged batch runs, and rated batch runs. On any
    failure it reverses the run mutations (ratings first — applied last —
    then tags) and deletes the created models in reverse order, then re-raises
    so the caller sees the original error.

    Synchronous and file-I/O heavy; call via asyncio.to_thread so the event
    loop is not blocked.
    """
    saved_models: list[Eval | EvalConfig | TaskRun | EvalInput | Spec] = []
    tagged_leaves: list[tuple[TaskRun, set[str]]] = []
    rated_leaves: list[
        tuple[TaskRun, TaskOutputRating | None, list[Feedback | ClaimReview]]
    ] = []
    try:
        eval.save_to_file()
        saved_models.append(eval)

        eval_config.save_to_file()
        saved_models.append(eval_config)

        # Legacy v1 flow: the freshly generated golden and train runs (with
        # their review children), then the eval slice split out of the same
        # generated pool as train.
        if single_turn_dataset is not None:
            for run in single_turn_dataset.task_runs:
                run.save_to_file()
                saved_models.append(run)
                single_turn_dataset.save_pending_children(run)
            persist_eval_slice(single_turn_dataset.eval_inputs, saved_models)

        spec.save_to_file()
        saved_models.append(spec)

        # Wizard arms: persist the eval slice (EvalInput items minted from
        # the driven cases or the generated inputs) and split the batch runs
        # into disjoint golden/train/val slices, AFTER spec has saved so a
        # failure here triggers the rollback below. tagged_leaves captures
        # only the tags this call added, so untagging on rollback preserves
        # any tags the run already had.
        if batch_leaves:
            persist_eval_slice(batch_eval_inputs, saved_models)
            split_and_tag_batch_runs(
                batch_leaves,
                reviewed_leaf_ids,
                train_tag,
                golden_tag,
                val_tag,
                rng=rng,
                tagged_out=tagged_leaves,
            )
            # Then write the human's verdicts: golden ratings (+ feedback and
            # per-claim grades) on the reviewed (golden) runs.
            rate_reviewed_batch_runs(
                batch_leaves,
                reviewed_refs,
                spec_name=spec_name,
                rated_out=rated_leaves,
            )
    except Exception:
        # Reverse run mutations before deleting saved models, so a failed
        # save doesn't leave orphan ratings or tags pointing at a
        # now-deleted eval.
        if rated_leaves:
            unrate_reviewed_batch_runs(rated_leaves)
        if tagged_leaves:
            untag_batch_runs_for_eval(tagged_leaves)
        for model in reversed(saved_models):
            try:
                model.delete()
            except Exception:
                # Log cleanup error but continue; the original error matters more.
                logger.exception(
                    f"Failed to delete {type(model).__name__} during cleanup"
                )
        raise


def connect_copilot_api(app: FastAPI):
    @app.post(
        "/api/copilot/clarify_spec",
        tags=["Copilot"],
        openapi_extra=agent_policy_require_approval("Run Copilot spec clarification?"),
    )
    async def clarify_spec(input: ClarifySpecApiInput) -> ClarifySpecApiOutput:
        api_key = get_copilot_api_key()
        client = get_authenticated_client(api_key)

        clarify_input = ClarifySpecInput.from_dict(
            await copilot_passthrough_payload(input)
        )

        detailed_result = (
            await clarify_spec_v1_copilot_clarify_spec_post.asyncio_detailed(
                client=client,
                body=clarify_input,
            )
        )
        result = unwrap_response(
            detailed_result,
            none_detail="Failed to analyze spec. Please try again.",
        )

        if isinstance(result, ClarifySpecOutput):
            return ClarifySpecApiOutput.model_validate(result.to_dict())

        raise HTTPException(
            status_code=500,
            detail="Unknown error.",
        )

    @app.post(
        "/api/copilot/refine_spec",
        tags=["Copilot"],
        openapi_extra=agent_policy_require_approval("Run Copilot spec refinement?"),
    )
    async def refine_spec(input: RefineSpecApiInput) -> RefineSpecApiOutput:
        api_key = get_copilot_api_key()
        client = get_authenticated_client(api_key)

        refine_input = RefineSpecInput.from_dict(
            await copilot_passthrough_payload(input)
        )

        detailed_result = (
            await refine_spec_v1_copilot_refine_spec_post.asyncio_detailed(
                client=client,
                body=refine_input,
            )
        )
        result = unwrap_response(
            detailed_result,
            none_detail="Failed to refine spec with feedback. Please try again.",
        )

        if isinstance(result, RefineSpecApiOutputClient):
            return RefineSpecApiOutput.model_validate(result.to_dict())

        raise HTTPException(
            status_code=500,
            detail="Unknown error.",
        )

    @app.post(
        "/api/copilot/generate_batch",
        tags=["Copilot"],
        openapi_extra=agent_policy_require_approval("Run Copilot batch generation?"),
    )
    async def generate_batch(input: GenerateBatchApiInput) -> GenerateBatchApiOutput:
        api_key = get_copilot_api_key()
        client = get_authenticated_client(api_key)

        generate_input = GenerateBatchInput.from_dict(input.model_dump())

        detailed_result = (
            await generate_batch_v1_copilot_generate_batch_post.asyncio_detailed(
                client=client,
                body=generate_input,
            )
        )
        result = unwrap_response(
            detailed_result,
            none_detail="Failed to generate synthetic data for spec. Please try again.",
        )

        if isinstance(result, GenerateBatchOutput):
            return GenerateBatchApiOutput.model_validate(result.to_dict())

        raise HTTPException(
            status_code=500,
            detail="Unknown error.",
        )

    @app.post(
        "/api/copilot/question_spec",
        tags=["Copilot"],
        openapi_extra=agent_policy_require_approval("Run Copilot spec questioner?"),
    )
    async def question_spec(
        input: SpecQuestionerApiInput,
    ) -> QuestionSet:
        api_key = get_copilot_api_key()
        client = get_authenticated_client(api_key)

        questioner_input = SpecQuestionerApiInputServerApi.from_dict(
            await copilot_passthrough_payload(input)
        )

        detailed_result = (
            await question_spec_v1_copilot_question_spec_post.asyncio_detailed(
                client=client,
                body=questioner_input,
            )
        )
        result = unwrap_response(
            detailed_result,
            none_detail="Failed to generate clarifying questions for spec. Please try again.",
        )

        if isinstance(result, QuestionSetServerApi):
            return QuestionSet.model_validate(result.to_dict())

        raise HTTPException(
            status_code=500,
            detail="Unknown error.",
        )

    @app.post(
        "/api/copilot/refine_spec_with_question_answers",
        tags=["Copilot"],
        openapi_extra=agent_policy_require_approval(
            "Run Copilot spec refinement with question answers?"
        ),
    )
    async def submit_question_answers(
        request: SubmitAnswersRequest,
    ) -> RefineSpecApiOutput:
        api_key = get_copilot_api_key()
        client = get_authenticated_client(api_key)

        submit_input = SubmitAnswersRequestServerApi.from_dict(request.model_dump())

        # Prefer the newer route that also returns a model-suggested eval name.
        detailed_result = await refine_spec_with_answers_and_name_v1_copilot_refine_spec_with_answers_and_name_post.asyncio_detailed(
            client=client,
            body=submit_input,
        )

        # Transitional fallback: the deployed prod copilot won't serve the
        # *_and_name route until the server ships it. This request names no
        # resource, so a 404 can only mean the route isn't deployed (not a
        # missing resource) — fall back to the older route, which never carries
        # a suggested_name. Any other status (auth, 422, 500) still propagates
        # via unwrap_response below, so we don't widen the error gate.
        # Remove this fallback once the *_and_name route is universally deployed.
        if detailed_result.status_code == HTTPStatus.NOT_FOUND:
            logger.warning(
                "kiln_server refine_spec_with_answers_and_name route missing "
                "(404); falling back to refine_spec_with_answers without a "
                "suggested name."
            )
            fallback_result = await refine_spec_with_answers_v1_copilot_refine_spec_with_answers_post.asyncio_detailed(
                client=client,
                body=submit_input,
            )
            result = unwrap_response(
                fallback_result,
                none_detail="Failed to refine spec with question answers. Please try again.",
            )
            if isinstance(result, RefineSpecApiOutputClient):
                return RefineSpecApiOutput.model_validate(result.to_dict())

            raise HTTPException(
                status_code=500,
                detail="Unknown error.",
            )

        result = unwrap_response(
            detailed_result,
            none_detail="Failed to refine spec with question answers. Please try again.",
        )

        if isinstance(result, RefineSpecFromAnswersAndNameOutputClient):
            # The *_and_name output has no not_incorporated_feedback field; the
            # studio response requires it, so set it to None and carry the name.
            output = result.to_dict()
            return RefineSpecApiOutput.model_validate(
                {
                    "new_proposed_spec_edits": output["new_proposed_spec_edits"],
                    "not_incorporated_feedback": None,
                    "suggested_name": output["suggested_name"],
                }
            )

        raise HTTPException(
            status_code=500,
            detail="Unknown error.",
        )

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/copilot/data_guide_job/start",
        tags=["Copilot"],
        openapi_extra=agent_policy_require_approval(
            "Draft a data guide from input examples with Copilot?"
        ),
    )
    async def start_data_guide_job(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        input: StartDataGuideJobApiInput,
    ) -> StartDataGuideJobApiOutput:
        """Kick off the input data guide draft job on kiln_server and return its
        job id. The job summarizes and aggregates the heterogeneous list of
        input examples (manual entries, existing task runs, uploaded text
        documents) into a draft guide.

        The job runs in the background so the user can leave the page and come
        back. The web UI polls `.../data_guide_job/{job_id}/status` and, once
        the job succeeds, fetches `.../data_guide_job/{job_id}/result` and
        generates preview inputs locally via the existing
        `/data_gen_guide_preview` flow.
        """
        api_key = get_copilot_api_key()
        client = get_authenticated_client(api_key)

        task = task_from_id(project_id, task_id)

        # Structured-input tasks: validate every example against the real input
        # schema before kicking off the draft job. The client only does a
        # shallow check at import, so this is the authoritative gate.
        if task.input_json_schema is not None:
            _validate_structured_examples(task.input_json_schema, input.input_examples)

        resolved_task_prompt = _resolve_task_runtime_prompt(task)

        # Everything except the examples is derived server-side from the task:
        # the prompt is resolved (not trusted from the client) and the input
        # schema is read straight off the task. The output schema and
        # task.description are never forwarded — output policy must not reach the
        # guide LLM.
        body = DraftInputDataGuideInput(
            task_prompt=resolved_task_prompt,
            task_input_schema=task.input_json_schema,
            input_examples=input.input_examples,
        )

        job_id = await _start_data_guide_job(client, body)
        return StartDataGuideJobApiOutput(job_id=job_id)

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/copilot/parse_import_file",
        tags=["Copilot"],
        openapi_extra=ALLOW_AGENT,
    )
    async def parse_import_file(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        file: Annotated[
            UploadFile,
            File(description="The file of input examples to parse and validate."),
        ],
    ) -> ParseImportFileApiOutput:
        """Parse an uploaded bulk-import file of input examples for the data
        guide, server-side.

        Both task types use a single-column CSV (parsed with the stdlib csv
        reader). Plaintext tasks take each cell as the raw input; structured
        tasks take each cell as a JSON object, validated against the task's input
        schema. Returns the parsed example strings plus any whole-file `error` or
        partial-skip `warning` so the web UI just renders the result.
        """
        task = task_from_id(project_id, task_id)
        raw = await file.read()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=422,
                detail="The file must be UTF-8 encoded text.",
            )

        return _parse_csv_import(content, task.input_json_schema)

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/copilot/data_guide_job/{job_id}/status",
        tags=["Copilot"],
        openapi_extra=ALLOW_AGENT,
    )
    async def data_guide_job_status(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        job_id: Annotated[
            str, Path(description="The data guide draft job identifier.")
        ],
    ) -> DataGuideJobStatusApiOutput:
        """Return the current status of a data guide draft job (e.g. running,
        succeeded, failed, cancelled). The web UI polls this while showing the
        analyzing animation and the task-wide progress widget."""
        # Validate the route scope — 404 on an unknown project/task path rather
        # than serving by job_id alone under any task.
        task_from_id(project_id, task_id)
        api_key = get_copilot_api_key()
        client = get_authenticated_client(api_key)
        status = await _get_data_guide_job_status(client, job_id)
        return DataGuideJobStatusApiOutput(status=status)

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/copilot/data_guide_job/{job_id}/result",
        tags=["Copilot"],
        openapi_extra=ALLOW_AGENT,
    )
    async def data_guide_job_result(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        job_id: Annotated[
            str, Path(description="The data guide draft job identifier.")
        ],
    ) -> DataGuideJobResultApiOutput:
        """Return the draft guide markdown produced by a completed data guide
        draft job. The web UI calls this once the job status is `succeeded`."""
        # Validate the route scope — 404 on an unknown project/task path rather
        # than serving by job_id alone under any task.
        task_from_id(project_id, task_id)
        api_key = get_copilot_api_key()
        client = get_authenticated_client(api_key)
        draft_guide = await _get_data_guide_job_result(client, job_id)
        return DataGuideJobResultApiOutput(draft_guide=draft_guide)

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/spec_with_copilot",
        tags=["Copilot"],
        openapi_extra=agent_policy_require_approval("Create spec with Copilot?"),
    )
    async def create_spec_with_copilot(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        request: CreateSpecWithCopilotRequest,
    ) -> Spec:
        """Create a spec using Kiln Copilot.

        This endpoint uses Kiln Copilot to create:
        1. An Eval for the spec with the appropriate template
        2. A judge EvalConfig (LLM-as-judge)
        3. The Spec itself
        Plus, per synthesis path:
        - Wizard arms (`single_turn` / `multi_turn`): tag the batch's
          existing runs with the golden/train filter tags — reviewed runs
          become golden with the human's ratings and claim reviews,
          unreviewed runs become train — and mint the eval slice as one
          EvalInput per generated input (single-turn) or driven case
          (multi-turn). Nothing is generated at save time.
        - Legacy v1 flow (`sdg_session_config`): batch examples via the
          copilot API, split into the train dataset (persisted as TaskRuns)
          and the eval slice; the golden dataset is the request's
          human-reviewed examples.

        On every path the eval slice is EvalInput items, which the runner
        runs fresh per run config at eval time — nothing stored there is
        judged.

        If you don't need copilot, use POST /spec instead.

        All models are validated before any saves occur. If validation fails,
        no data is persisted.
        """
        task = task_from_id(project_id, task_id)

        # Idempotency guard against re-submits after a completed save (the
        # save is slow, so users retry). Compared via the derived eval tags,
        # not the raw name: tags come from the lowercased, space-normalized
        # name, so "My Spec" and "my_spec" would silently share a tag
        # namespace (and each other's datasets) if only exact names were
        # rejected. Two requests in flight at once can still race past this
        # check — acceptable for a single-user studio.
        requested_tags = generate_spec_eval_tags(request.name)
        if any(
            generate_spec_eval_tags(spec.name) == requested_tags
            for spec in task.specs(readonly=True)
        ):
            raise HTTPException(
                status_code=409,
                detail=f"A spec named '{request.name}' (or one differing only "
                "by case or spacing) already exists for this task.",
            )

        # Generate tags and filter IDs. The wizard arms deal their non-golden
        # batch runs train:val, so both tags address real items. The legacy v1
        # arm mints no val items and leaves its val split empty (0 items, not
        # an error) rather than giving the eval a different splits shape.
        tags = generate_spec_eval_tags(request.name)
        eval_tag, train_tag, val_tag, golden_tag = (
            tags.test_tag,
            tags.train_tag,
            tags.val_tag,
            tags.golden_tag,
        )
        train_set_filter_id = tag_filter_id(train_tag)
        val_set_filter_id = tag_filter_id(val_tag)
        eval_configs_filter_id = tag_filter_id(golden_tag)

        # Extract spec_type from properties (discriminated union)
        spec_type = request.properties["spec_type"]

        # Determine eval properties
        template = spec_eval_template(spec_type)
        output_scores = [spec_eval_output_score(request.name)]
        evaluation_data_type = spec_eval_data_type(
            spec_type, request.evaluate_full_trace
        )

        # The builder's judge template never renders a reference answer, so a
        # reference_answer eval would save fine and then mis-score every run.
        # Not reachable from the current UI flow — this guards direct
        # API/agent clients.
        if evaluation_data_type == EvalDataType.reference_answer:
            raise HTTPException(
                status_code=400,
                detail="Reference-answer specs are not supported by the spec "
                "builder yet: the saved judge would never see the reference "
                "answer. Create this eval from the Evals tab instead.",
            )

        # Batch arms: find the existing runs up front so we 404 before
        # creating any models if the batch_tag matches nothing. The reviewed
        # run ids drive the split — only rated runs are eligible for golden
        # (capped at the target fraction); the rest are dealt train:val with
        # their ratings kept.
        batch_leaves: list[TaskRun] = []
        reviewed_refs: list[ReviewedChainApi] = []
        reviewed_leaf_ids: set[str] = set()
        if request.multi_turn is not None:
            batch_leaves = find_multi_turn_chain_leaves(
                task, request.multi_turn.batch_tag
            )
            if not batch_leaves:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No multi-turn chains found for batch_tag "
                        f"'{request.multi_turn.batch_tag}'."
                    ),
                )
            reviewed_refs = request.multi_turn.reviewed_chains
            reviewed_leaf_ids = validate_reviewed_refs(
                reviewed_refs, batch_leaves, request.multi_turn.batch_tag
            )
        if request.single_turn is not None:
            batch_leaves = find_single_turn_batch_runs(
                task, request.single_turn.batch_tag
            )
            if not batch_leaves:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No single-turn runs found for batch_tag "
                        f"'{request.single_turn.batch_tag}'."
                    ),
                )
            reviewed_refs = request.single_turn.reviewed_runs
            reviewed_leaf_ids = validate_reviewed_refs(
                reviewed_refs, batch_leaves, request.single_turn.batch_tag
            )

        # Build and validate all models before saving any; persist_spec_save
        # commits them as one unit of work below.

        # The batch arms' eval slice (validated here, persisted in the unit
        # of work). Multi-turn: one EvalInput per driven case — 422s on a
        # malformed persona blob before anything is written. Single-turn: one
        # EvalInput per generated input; on a structured-input task each must
        # match the input schema, or the saved eval would fail every job at
        # run time — 422 up front instead.
        batch_eval_inputs: list[EvalInput] = []
        if request.multi_turn is not None:
            batch_eval_inputs = build_multi_turn_eval_inputs(
                request.multi_turn.cases,
                request.multi_turn.batch_tag,
                task,
                eval_tag,
                request.multi_turn.drive_config,
            )
        if request.single_turn is not None:
            if task.input_json_schema is not None:
                _validate_structured_examples(
                    str(task.input_json_schema), request.single_turn.inputs
                )
            batch_eval_inputs = build_single_turn_batch_eval_inputs(
                request.single_turn.inputs,
                request.single_turn.batch_tag,
                task,
                eval_tag,
            )

        # 1. Create the Eval. Golden, train and val are TaskRun slices on both
        # paths; the eval slice is EvalInput-tagged on both, re-run per run
        # config at eval time (multi-turn re-drives it, using the drive
        # config stamped on each item).
        eval = Eval(
            parent=task,
            name=request.name,
            description=None,
            template=template,
            output_scores=output_scores,
            # Priority and status live on the eval; the spec below mirrors
            # them at creation so the spec file stays truthful.
            priority=Priority.p1,
            status=EvalStatus.active,
            # `splits` is the single home for all three splits: the
            # EvalInput-backed test split and the TaskRun-backed train and val
            # splits. The deprecated flat filter fields are never written.
            splits={
                "test": EvalInputSplit(filter_id=f"tag::{eval_tag}"),
                "train": TaskRunSplit(filter_id=train_set_filter_id),
                "val": TaskRunSplit(filter_id=val_set_filter_id),
            },
            eval_configs_filter_id=eval_configs_filter_id,
            template_properties=None,
            evaluation_data_type=evaluation_data_type,
        )

        # 2. Create the judge eval config — V2 shape, the same judge the review
        # step ran transiently (one judge, persisted vs transient). V2 rails
        # give it an editable prompt_template the refine loop can write back
        # into, instead of the legacy llm_as_judge dispatch.
        eval_config = EvalConfig(
            parent=eval,
            name=generate_memorable_name(),
            config_type=EvalConfigType.v2,
            properties=LlmJudgeProperties(
                model_name=request.judge_info.model_name,
                model_provider=request.judge_info.model_provider,
                prompt_template=build_judge_prompt_template(
                    request.judge_info.prompt,
                    multi_turn=request.evaluate_full_trace,
                ),
            ),
        )

        # Set as default config after ID is assigned
        eval.current_config_id = eval_config.id

        # One RNG seam for every dataset split (the batch arms' run split and
        # the legacy generated pool) — injectable so tests are deterministic.
        rng = random.Random()

        # 3. Legacy v1 flow only: synthesise examples, then build the
        #    golden/train TaskRuns and the eval slice's EvalInputs from them.
        #    Both wizard arms skip this — their runs already exist on disk,
        #    and nothing is generated at save time.
        single_turn_dataset: SingleTurnDataset | None = None
        sdg_session_config_for_spec: SyntheticDataGenerationSessionConfig | None = None
        if request.sdg_session_config is not None:
            api_key = get_copilot_api_key()
            task_input_schema = (
                str(task.input_json_schema) if task.input_json_schema else ""
            )
            task_output_schema = (
                str(task.output_json_schema) if task.output_json_schema else ""
            )
            task_tools, task_skills = await task_capabilities_for_task(
                task, request.run_config_id
            )
            all_examples = await generate_copilot_examples(
                api_key=api_key,
                target_task_info=TaskInfoApi(
                    task_prompt=request.task_prompt_with_example or "",
                    task_input_schema=task_input_schema,
                    task_output_schema=task_output_schema,
                    task_tools=task_tools,
                    task_skills=task_skills,
                ),
                sdg_session_config=request.sdg_session_config,
                spec_definition=request.definition,
            )

            single_turn_dataset = create_single_turn_dataset(
                all_examples=all_examples,
                reviewed_examples=request.reviewed_examples,
                eval_tag=eval_tag,
                train_tag=train_tag,
                golden_tag=golden_tag,
                spec_name=request.name,
                rng=rng,
            )
            for run in single_turn_dataset.task_runs:
                run.parent = task
            for eval_input in single_turn_dataset.eval_inputs:
                eval_input.parent = task

            # Snapshot the generation config on the Spec (legacy flow only).
            topic_cfg = request.sdg_session_config.topic_generation_config
            input_cfg = request.sdg_session_config.input_generation_config
            output_cfg = request.sdg_session_config.output_generation_config
            sdg_session_config_for_spec = SyntheticDataGenerationSessionConfig(
                topic_generation_config=SyntheticDataGenerationStepConfig(
                    model_name=topic_cfg.task_metadata.model_name,
                    provider_name=topic_cfg.task_metadata.model_provider_name,
                    prompt=topic_cfg.prompt,
                ),
                input_generation_config=SyntheticDataGenerationStepConfig(
                    model_name=input_cfg.task_metadata.model_name,
                    provider_name=input_cfg.task_metadata.model_provider_name,
                    prompt=input_cfg.prompt,
                ),
                output_generation_config=SyntheticDataGenerationStepConfig(
                    model_name=output_cfg.task_metadata.model_name,
                    provider_name=output_cfg.task_metadata.model_provider_name,
                    prompt=output_cfg.prompt,
                ),
            )

        # 4. Create the Spec. The wizard arms leave sdg_session_config unset —
        # the operational state lives on the Eval (full_trace + filter_ids).
        spec = Spec(
            parent=task,
            name=request.name,
            definition=request.definition,
            properties=request.properties,
            priority=Priority.p1,
            status=SpecStatus.active,
            tags=[],
            eval_id=eval.id,
            task_sample=request.task_sample,
            synthetic_data_generation_session_config=sdg_session_config_for_spec,
        )

        # All models are now created and validated via Pydantic. Persist them
        # as one unit of work (all-or-nothing) off the event loop — the save is
        # dozens-to-hundreds of serial file writes plus the batch-run
        # mutations, all synchronous.
        await asyncio.to_thread(
            persist_spec_save,
            eval=eval,
            eval_config=eval_config,
            single_turn_dataset=single_turn_dataset,
            spec=spec,
            batch_leaves=batch_leaves,
            batch_eval_inputs=batch_eval_inputs,
            reviewed_refs=reviewed_refs,
            reviewed_leaf_ids=reviewed_leaf_ids,
            train_tag=train_tag,
            golden_tag=golden_tag,
            val_tag=val_tag,
            spec_name=request.name,
            rng=rng,
        )

        return spec
