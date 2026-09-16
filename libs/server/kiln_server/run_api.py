import asyncio
import json
import logging
import os
import tempfile
from asyncio import Lock
from datetime import datetime
from pathlib import Path as PathLibPath
from typing import Annotated, Any, Dict

from fastapi import Body, FastAPI, File, Form, HTTPException, Path, Query, UploadFile
from kiln_ai.adapters.adapter_registry import adapter_for_task, load_skills_for_task
from kiln_ai.adapters.errors import ErrorWithTrace
from kiln_ai.adapters.ml_model_list import ModelProviderName
from kiln_ai.adapters.model_adapters.base_adapter import AdapterConfig
from kiln_ai.datamodel import Task, TaskOutputRating, TaskOutputRatingType, TaskRun
from kiln_ai.datamodel.basemodel import ID_TYPE
from kiln_ai.datamodel.datamodel_enums import StructuredInputType, TurnMode
from kiln_ai.datamodel.task import RunConfigProperties
from kiln_ai.datamodel.task_output import DataSource, DataSourceType, TaskOutput
from kiln_ai.utils.dataset_import import (
    DatasetFileImporter,
    DatasetImportFormat,
    ImportConfig,
    KilnInvalidImportFormat,
)
from pydantic import BaseModel, ConfigDict, Field

from kiln_server.task_api import task_from_id
from kiln_server.utils.agent_checks.policy import (
    ALLOW_AGENT,
    DENY_AGENT,
    agent_policy_require_approval,
)

logger = logging.getLogger(__name__)

# Lock to prevent overwriting via concurrent updates. We use a load/update/write pattern that is not atomic.
update_run_lock = Lock()

EVAL_TRACE_DELETE_MESSAGE = "This run can't be deleted because it's needed for an eval."

# Defensive cap on parent-chain depth. Real multiturn conversations are tiny
# compared to this; the guard exists to terminate on disk corruption or cycles.
_MAX_ANCESTOR_DEPTH = 1000


def _walk_run_chain(
    leaf: TaskRun, task_path: PathLibPath | None
) -> tuple[list[TaskRun], bool]:
    """
    Walk parent_task_run_id from `leaf` upward, returning (chain_root_to_leaf, chain_broken).

    chain_broken is True if any parent failed to load, a cycle was detected, or
    the depth guard tripped. On break, the returned list is the intact suffix
    from `leaf` back to (but not including) the missing/cyclic node, reversed
    to root-to-leaf order.
    """
    chain: list[TaskRun] = [leaf]
    visited: set[str] = set()
    if leaf.id is not None:
        visited.add(str(leaf.id))
    current = leaf
    for _ in range(_MAX_ANCESTOR_DEPTH):
        if current.parent_task_run_id is None:
            chain.reverse()
            return chain, False
        if current.parent_task_run_id in visited:
            chain.reverse()
            return chain, True
        parent = TaskRun.from_id_and_parent_path(current.parent_task_run_id, task_path)
        if parent is None:
            chain.reverse()
            return chain, True
        chain.append(parent)
        if parent.id is not None:
            visited.add(str(parent.id))
        current = parent
    chain.reverse()
    return chain, True


def _collect_cascade_delete_runs(
    task: Task,
    target: TaskRun,
    already_queued: set[str] | None = None,
) -> list[TaskRun]:
    """Compute the full set of TaskRuns to delete when `target` is deleted.

    Walks `parent_task_run_id` upward from `target`. Each ancestor is included
    iff every one of its children has already been marked for deletion in this
    cascade (i.e. there's no sibling branch keeping it alive). Stops at the
    first ancestor with a live sibling child.

    Pass ``already_queued`` (set of run id strings) when running a sequence of
    cascades within a single batch (e.g. bulk delete). Runs in that set are
    treated as already-deleted for the live-children check, so an ancestor
    whose only live children are themselves queued in a prior cascade will be
    swept up here.
    """
    queued: set[str] = set(already_queued or ())
    target_id = str(target.id) if target.id is not None else None

    to_delete: list[TaskRun] = []
    if target_id is None or target_id not in queued:
        to_delete.append(target)
        if target_id is not None:
            queued.add(target_id)

    if target.parent_task_run_id is None:
        return to_delete

    # Pull every run on disk once so we can do child counts without re-hitting
    # the loader for each ancestor. We need the full chain view here, including
    # eval-generated runs: an eval trace hanging off an ancestor must count as a
    # live child, or the cascade deletes the ancestor out from under the eval.
    all_runs = task.runs(
        include_intermediate_runs=True, include_eval_generated=True, readonly=True
    )
    children_by_parent: Dict[str, list[str]] = {}
    runs_by_id: Dict[str, TaskRun] = {}
    for r in all_runs:
        if r.id is not None:
            runs_by_id[str(r.id)] = r
        if r.parent_task_run_id and r.id is not None:
            children_by_parent.setdefault(r.parent_task_run_id, []).append(str(r.id))

    visited: set[str] = set(queued)
    current = target
    for _ in range(_MAX_ANCESTOR_DEPTH):
        parent_id = current.parent_task_run_id
        if parent_id is None:
            break
        if parent_id in visited:
            # Cycle: stop here, but everything queued so far is still valid.
            break
        parent = runs_by_id.get(parent_id)
        if parent is None:
            # Chain broken: stop the cascade, don't 500.
            break
        live_children = [
            cid for cid in children_by_parent.get(parent_id, []) if cid not in queued
        ]
        if live_children:
            # A sibling branch survives — keep this parent.
            break
        if parent.eval_source is not None:
            # An eval still needs this ancestor: keep it, like a live sibling.
            break
        if parent.id is not None and str(parent.id) not in queued:
            to_delete.append(parent)
            queued.add(str(parent.id))
            visited.add(str(parent.id))
        current = parent
    return to_delete


def _count_user_messages(trace: list[Any] | None) -> int:
    if not trace:
        return 0
    return sum(1 for m in trace if isinstance(m, dict) and m.get("role") == "user")


def _position_chain_turns(
    chain_runs: list[TaskRun], chain_broken: bool
) -> tuple[list[TaskRun], list[int | None], bool]:
    """
    Map each run in a root-to-leaf chain to the index in the leaf's trace where
    its turn begins. A run's trace is its parent's trace plus its own turn's
    messages, so trace lengths strictly increase along the chain and each turn
    starts at its parent's trace length (0 for the true root). Message-role
    counting can't do this: chat strategies differ in how many user messages a
    turn emits (e.g. chain-of-thought adds a second one).

    A run whose trace fails to extend its predecessor's invalidates everything
    before it: keep the suffix from that run onward and flag the chain broken.
    Pass chain_broken=True when ancestors are already missing; the first kept
    run's boundary is then unknowable and reported as None.
    """
    kept: list[TaskRun] = []
    starts: list[int | None] = []
    prev_len = 0
    for run in chain_runs:
        trace_len = len(run.trace or [])
        if trace_len <= prev_len:
            kept, starts = [run], [None]
            chain_broken = True
        else:
            starts.append(None if not kept and chain_broken else prev_len)
            kept.append(run)
        prev_len = trace_len
    return kept, starts, chain_broken


def deep_update(
    source: Dict[str, Any] | None, update: Dict[str, Any | None]
) -> Dict[str, Any]:
    if source is None:
        return {k: v for k, v in update.items() if v is not None}
    for key, value in update.items():
        if value is None:
            source.pop(key, None)
        elif isinstance(value, dict):
            if key not in source or not isinstance(source[key], dict):
                source[key] = {}
            source[key] = deep_update(source[key], value)
        else:
            source[key] = value
    return {k: v for k, v in source.items() if v is not None}


class RunTaskRequest(BaseModel):
    """Request to invoke an AI model on a task."""

    run_config_properties: RunConfigProperties = Field(
        description="The run configuration specifying model, prompt, and generation parameters."
    )
    plaintext_input: str | None = Field(
        default=None,
        description="The task input as plaintext. Use for unstructured tasks.",
    )
    structured_input: StructuredInputType | None = Field(
        default=None,
        description="The task input as structured JSON. Use for tasks with an input schema.",
    )
    tags: list[str] | None = Field(
        default=None, description="Tags to apply to the resulting task run."
    )
    parent_task_run_id: str | None = Field(
        default=None,
        description=(
            "Continue the conversation started by this parent run. "
            "Multi-turn tasks only."
        ),
    )
    task_run_config_id: str | None = Field(
        default=None,
        description="The ID of the saved TaskRunConfig the caller used to populate run_config_properties, if any. Stored on the resulting TaskRun so the run can be traced back to its originating saved config. None for ad-hoc runs that were not initiated from a saved TaskRunConfig.",
    )

    # Allows use of the model_name field (usually pydantic will reserve model_*)
    model_config = ConfigDict(protected_namespaces=())


class RunChainEntry(BaseModel):
    """A single entry in a multi-turn run's conversation chain."""

    run_id: ID_TYPE = Field(
        description="The TaskRun id at this turn position in the chain."
    )
    turn_index: int = Field(
        description=(
            "1-based turn index within the returned chain (turn 1 = first "
            "entry, turn N = leaf). For an unbroken chain this is the absolute "
            "turn number in the conversation; for a broken chain it is relative "
            "to the returned suffix, since absolute positions are unknowable "
            "when ancestors are missing."
        )
    )
    trace_start_index: int | None = Field(
        description=(
            "Index into the leaf run's trace where this turn's messages begin. "
            "A run's trace is its parent's trace plus its own turn, so this is "
            "the parent run's trace length (0 for the conversation root). None "
            "when the boundary is unknowable (first entry of a broken chain)."
        )
    )


class RunChainResponse(BaseModel):
    """Ordered conversation chain for a multi-turn TaskRun.

    The chain is rooted at the conversation start and ends with the requested
    run itself (the requested run is always the final entry, even if it is the
    only entry).
    """

    chain: list[RunChainEntry] = Field(
        description=(
            "Ordered root-to-leaf, includes the requested run itself as the "
            "final entry. If chain_broken is true, the list contains only the "
            "intact suffix from the leaf back to (and excluding) the break "
            "point."
        )
    )
    chain_broken: bool = Field(
        description=(
            "True if while walking parents we encountered a parent_task_run_id "
            "that could not be loaded, a cycle, the depth guard, or a run whose "
            "trace does not extend its parent's (so it can't be positioned in "
            "the leaf's trace)."
        )
    )
    has_children: bool = Field(
        description=(
            "True if at least one other TaskRun in the task references the "
            "requested run via parent_task_run_id (i.e. the requested run is "
            "an intermediate node in the chain, not a leaf). Used by the UI "
            "to warn that sending a new message from this run will create a "
            "new branch rather than extending an existing one."
        )
    )


class RunSummary(BaseModel):
    """A summary of a task run for list views."""

    id: ID_TYPE = Field(description="The unique identifier of the task run.")
    rating: TaskOutputRating | None = Field(
        default=None, description="The rating of the task run output."
    )
    created_at: datetime = Field(description="When the run was created.")
    input_preview: str | None = Field(
        default=None, description="A truncated preview of the task input."
    )
    output_preview: str | None = Field(
        default=None, description="A truncated preview of the task output."
    )
    repair_state: str | None = Field(
        default=None,
        description="The repair state of the run (e.g., 'Repaired', 'No repair needed').",
    )
    model_name: str | None = Field(
        default=None, description="The model used for this run."
    )
    input_source: str | None = Field(
        default=None, description="The source of the input (human, synthetic, etc.)."
    )
    tags: list[str] | None = Field(default=None, description="Tags applied to the run.")

    @classmethod
    def format_preview(cls, text: str | None, max_length: int = 100) -> str | None:
        if text is None:
            return None
        if len(text) > max_length:
            return text[:max_length] + "…"
        return text

    @classmethod
    def repair_status_display_name(cls, run: TaskRun) -> str:
        if run.repair_instructions:
            return "Repaired"
        elif run.output and not run.output.rating:
            # A repair isn't requested until rated < 5 stars
            return "NA"
        elif not run.output or not run.output.output:
            return "No output"
        elif (
            run.output.rating
            and run.output.rating.value == 5.0
            and run.output.rating.type == TaskOutputRatingType.five_star
        ):
            return "No repair needed"
        elif (
            run.output.rating
            and run.output.rating.type != TaskOutputRatingType.five_star
        ):
            return "Unknown"
        elif run.output.output:
            return "Repair needed"
        return "Unknown"

    @classmethod
    def from_run(cls, run: TaskRun) -> "RunSummary":
        model_name = (
            run.output.source.properties.get("model_name")
            if run.output and run.output.source and run.output.source.properties
            else None
        )
        if not isinstance(model_name, str):
            model_name = None
        output = run.output.output if run.output and run.output.output else None

        return RunSummary(
            id=run.id,
            rating=run.output.rating,
            tags=run.tags,
            input_preview=RunSummary.format_preview(run.input),
            output_preview=RunSummary.format_preview(output),
            created_at=run.created_at,
            repair_state=RunSummary.repair_status_display_name(run),
            model_name=model_name,
            input_source=run.input_source.type if run.input_source else None,
        )


class BulkUploadResponse(BaseModel):
    """Response from a bulk import of task runs."""

    success: bool = Field(description="Whether the import succeeded.")
    filename: str = Field(description="The filename that was imported.")
    imported_count: int = Field(description="The number of task runs imported.")
    imported_conversation_count: int | None = Field(
        default=None,
        description=(
            "The number of conversations imported. None for single-turn uploads; "
            "set for multiturn uploads (where one row = one conversation that "
            "materializes as multiple TaskRuns linked via parent_task_run_id)."
        ),
    )


class CreateTaskRunRequest(BaseModel):
    """Request model for creating a synthetic TaskRun directly (without running a model)."""

    input: str = Field(description="The input for the task run")
    output: str = Field(description="The output for the task run")
    tags: list[str] = Field(default=[], description="Tags to apply to the task run")
    rating: TaskOutputRating | None = Field(
        default=None, description="Optional rating for the output"
    )
    model_name: str = Field(
        description="The name of the model used to generate the data",
    )
    model_provider: str = Field(
        description="The provider of the model used to generate the data",
    )
    adapter_name: str = Field(
        description="The name of the adapter used to generate the data",
    )


def run_from_id(project_id: str, task_id: str, run_id: str) -> TaskRun:
    _, run = task_and_run_from_id(project_id, task_id, run_id)
    return run


def task_and_run_from_id(
    project_id: str, task_id: str, run_id: str
) -> tuple[Task, TaskRun]:
    task = task_from_id(project_id, task_id)
    run = TaskRun.from_id_and_parent_path(run_id, task.path)
    if run:
        return task, run

    raise HTTPException(
        status_code=404,
        detail=f"Run not found. ID: {run_id}",
    )


def connect_run_api(app: FastAPI):
    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/runs/{run_id}",
        summary="Get Run",
        tags=["Runs"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_run(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        run_id: Annotated[
            str, Path(description="The unique identifier of the task run.")
        ],
    ) -> TaskRun:
        return run_from_id(project_id, task_id, run_id)

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/runs/{run_id}/chain",
        summary="Get Run Chain",
        tags=["Runs"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_run_chain(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        run_id: Annotated[
            str,
            Path(
                description="The unique identifier of the task run whose chain to return."
            ),
        ],
    ) -> RunChainResponse:
        task = task_from_id(project_id, task_id)
        leaf = TaskRun.from_id_and_parent_path(run_id, task.path)
        if leaf is None:
            raise HTTPException(
                status_code=404,
                detail=f"Run not found. ID: {run_id}",
            )
        if task.turn_mode != TurnMode.multiturn:
            raise HTTPException(
                status_code=400,
                detail="Run chain is only available for multi-turn tasks.",
            )
        has_children = any(
            r.parent_task_run_id == run_id
            for r in task.runs(
                include_intermediate_runs=True,
                include_eval_generated=True,
                readonly=True,
            )
        )
        chain_runs, chain_broken = _walk_run_chain(leaf, task.path)
        # Degenerate leaf trace (no user messages at all): not a conversation,
        # so we can't position any run on a turn. Surface as broken-chain with
        # an empty list.
        if _count_user_messages(leaf.trace) == 0:
            return RunChainResponse(
                chain=[], chain_broken=True, has_children=has_children
            )
        # Each run in the chain is one turn; anchor each turn to where it
        # begins in the leaf's trace, dropping any prefix that can't be
        # positioned (see _position_chain_turns).
        chain_runs, turn_starts, chain_broken = _position_chain_turns(
            chain_runs, chain_broken
        )
        chain = [
            RunChainEntry(
                run_id=r.id,
                turn_index=i + 1,
                trace_start_index=turn_starts[i],
            )
            for i, r in enumerate(chain_runs)
        ]
        return RunChainResponse(
            chain=chain,
            chain_broken=chain_broken,
            has_children=has_children,
        )

    @app.delete(
        "/api/projects/{project_id}/tasks/{task_id}/runs/{run_id}",
        summary="Delete Run",
        tags=["Runs"],
        openapi_extra=DENY_AGENT,
    )
    async def delete_run(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        run_id: Annotated[
            str, Path(description="The unique identifier of the task run.")
        ],
    ):
        task, run = task_and_run_from_id(project_id, task_id, run_id)
        # 409 not 400: the request is well formed, the resource state forbids it.
        if run.eval_source is not None:
            raise HTTPException(status_code=409, detail=EVAL_TRACE_DELETE_MESSAGE)
        # For multiturn chains, also delete ancestors whose only remaining child
        # is in our delete-set. Stop at the first ancestor that still has another
        # live child (a sibling branch) — or one an eval still needs.
        runs_to_delete = _collect_cascade_delete_runs(task, run)
        for r in runs_to_delete:
            r.delete()

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/runs",
        summary="List Runs",
        description=(
            "For multi-turn tasks, only leaf TaskRuns (those that are not the "
            "parent of another run via parent_task_run_id) are returned. "
            "Intermediate runs in a chain are filtered out. For single-turn "
            "tasks this is equivalent to listing every run."
        ),
        tags=["Runs"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_runs(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        limit: Annotated[
            int | None,
            Query(
                description="Maximum number of runs to return. When set, the most recent runs (by created_at) are returned. When omitted, all runs are returned.",
                ge=1,
            ),
        ] = None,
    ) -> list[TaskRun]:
        task = task_from_id(project_id, task_id)
        runs = list(task.runs(readonly=True))
        runs.sort(key=lambda r: r.created_at, reverse=True)
        if limit is not None:
            runs = runs[:limit]
        return runs

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/runs",
        summary="Create Run",
        tags=["Runs"],
        openapi_extra=ALLOW_AGENT,
    )
    async def create_task_run(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        request: CreateTaskRunRequest,
    ) -> TaskRun:
        """Create a TaskRun directly without running a model."""
        task = task_from_id(project_id, task_id)

        data_source = DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": request.model_name,
                "model_provider": request.model_provider,
                "adapter_name": request.adapter_name,
            },
        )

        output = TaskOutput(
            output=request.output,
            source=data_source,
            rating=request.rating,
        )

        run = TaskRun(
            parent=task,
            input=request.input,
            input_source=data_source,
            output=output,
            tags=request.tags,
        )
        run.save_to_file()
        return run

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/runs_summaries",
        summary="List Run Summaries",
        description=(
            "For multi-turn tasks, only leaf TaskRuns (those that are not the "
            "parent of another run via parent_task_run_id) are summarized. "
            "For single-turn tasks this is equivalent to summarizing every run."
        ),
        tags=["Runs"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_runs_summary(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
    ) -> list[RunSummary]:
        task = task_from_id(project_id, task_id)
        # Readonly since we are not mutating the runs. Faster as we don't need to copy them.
        # Summaries only need leaves.
        runs = task.runs(readonly=True)
        return [RunSummary.from_run(run) for run in runs]

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/runs/delete",
        summary="Delete Runs",
        tags=["Runs"],
        openapi_extra=DENY_AGENT,
    )
    async def delete_runs(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        run_ids: Annotated[list[str], Body(description="List of run IDs to delete.")],
    ):
        task = task_from_id(project_id, task_id)
        failed_runs: list[str] = []
        failure_reasons: list[str] = []

        def record_failure(run_id: str, reason: str) -> None:
            # Every distinct reason is kept, in first-seen order. A single "last error"
            # would report only whichever failure came last, which since eval traces
            # became undeletable can mean a mixed selection says "Run not found" about
            # runs that are on disk and merely protected.
            failed_runs.append(run_id)
            if reason and reason not in failure_reasons:
                failure_reasons.append(reason)

        # Cascade behavior matches single DELETE: sweep orphan ancestors. The
        # cumulative queued_ids set means an ancestor whose remaining children
        # are all in this batch also gets cascaded.
        queued_ids: set[str] = set()
        runs_to_delete: list[TaskRun] = []

        for run_id in run_ids:
            try:
                run = TaskRun.from_id_and_parent_path(run_id, task.path)
                if run is None:
                    record_failure(run_id, "Run not found")
                    continue
                if run.eval_source is not None:
                    # Reported rather than silently skipped, so a partially deleted
                    # selection says why. Collected like any other per-run failure
                    # instead of raising: the rest of the batch is still deletable.
                    record_failure(run_id, EVAL_TRACE_DELETE_MESSAGE)
                    continue
                cascade = _collect_cascade_delete_runs(task, run, queued_ids)
                for r in cascade:
                    if r.id is not None:
                        queued_ids.add(str(r.id))
                    runs_to_delete.append(r)
            except Exception as e:
                record_failure(run_id, str(e))

        for r in runs_to_delete:
            try:
                r.delete()
            except Exception as e:
                if r.id is not None:
                    record_failure(str(r.id), str(e))

        if failed_runs:
            raise HTTPException(
                status_code=500,
                detail={
                    "failed_runs": failed_runs,
                    "error": "; ".join(failure_reasons) or "Unknown error",
                },
            )
        return {"success": True}

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/run",
        summary="Execute Run",
        tags=["Runs"],
        openapi_extra=agent_policy_require_approval("Run task with LLM?"),
        responses={
            # Adapter failures (e.g., LLM rate limit, tool crash) return
            # ErrorWithTrace with the partial conversation trace so the UI
            # can show what happened before the error. Other 500s and 4xx
            # errors use the standard error shape.
            500: {"model": ErrorWithTrace},
        },
    )
    async def run_task(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        request: RunTaskRequest,
    ) -> TaskRun:
        """Invoke an AI model on a task and return the result. Unlike 'Create Run', this actually executes the model."""
        task = task_from_id(project_id, task_id)

        run_config_properties = request.run_config_properties
        skills = load_skills_for_task(task, run_config_properties)

        adapter = adapter_for_task(
            task,
            run_config_properties=run_config_properties,
            base_adapter_config=AdapterConfig(
                default_tags=request.tags,
                skills=skills,
                task_run_config_id=request.task_run_config_id,
            ),
        )

        input = request.plaintext_input
        if task.input_schema() is not None:
            input = request.structured_input

        if input is None:
            raise HTTPException(
                status_code=400,
                detail="No input provided. Ensure your provided the proper format (plaintext or structured).",
            )

        prior_trace = None
        parent_task_run = None
        if request.parent_task_run_id is not None:
            if task.turn_mode != TurnMode.multiturn:
                raise HTTPException(
                    status_code=400,
                    detail="parent_task_run_id is only valid for multi-turn tasks.",
                )
            parent_task_run = TaskRun.from_id_and_parent_path(
                request.parent_task_run_id, task.path
            )
            if parent_task_run is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Parent run not found. ID: {request.parent_task_run_id}",
                )
            if not parent_task_run.trace:
                raise HTTPException(
                    status_code=400,
                    detail="Parent run cannot be continued because it has no trace.",
                )
            prior_trace = parent_task_run.trace

        run = await adapter.invoke(
            input,
            prior_trace=prior_trace,
            parent_task_run=parent_task_run,
        )

        # The conversation may have been cascade-deleted while the model was
        # generating (invoke autosaves the new run before returning). Don't
        # resurrect a deleted conversation as an orphaned leaf: remove the
        # just-saved run and tell the caller what happened.
        if request.parent_task_run_id is not None:
            parent_still_exists = (
                TaskRun.from_id_and_parent_path(request.parent_task_run_id, task.path)
                is not None
            )
            if not parent_still_exists:
                if run.path is not None:
                    run.delete()
                raise HTTPException(
                    status_code=409,
                    detail="The conversation was deleted while the response was being generated, so the new message was discarded.",
                )

        return run

    @app.patch(
        "/api/projects/{project_id}/tasks/{task_id}/runs/{run_id}",
        summary="Update Run",
        tags=["Runs"],
        openapi_extra=agent_policy_require_approval(
            "Allow agent to edit run? Ensure you backup your project before allowing agentic edits."
        ),
    )
    async def update_run(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        run_id: Annotated[
            str, Path(description="The unique identifier of the task run.")
        ],
        run_data: Annotated[
            Dict[str, Any], Body(description="Fields to update on the run.")
        ],
    ) -> TaskRun:
        return await update_run_util(project_id, task_id, run_id, run_data)

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/runs/edit_tags",
        summary="Edit Run Tags",
        tags=["Runs"],
        openapi_extra=ALLOW_AGENT,
    )
    async def edit_tags(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        run_ids: Annotated[
            list[str], Body(description="The list of task run IDs to edit.")
        ],
        add_tags: Annotated[
            list[str] | None, Body(description="Tags to add to the task runs.")
        ] = None,
        remove_tags: Annotated[
            list[str] | None, Body(description="Tags to remove from the task runs.")
        ] = None,
    ):
        task = task_from_id(project_id, task_id)

        # all the runs we need to tag
        run_ids_set: set[str] = set(run_ids)
        runs_found_set: set[str] = set()

        batch_size = 500
        for i in range(0, len(run_ids), batch_size):
            # release the event loop to prevent blocking other operations for too long
            await asyncio.sleep(0)

            batch_run_ids = run_ids[i : i + batch_size]
            batch_runs = TaskRun.from_ids_and_parent_path(set(batch_run_ids), task.path)
            runs_found_set.update(batch_runs.keys())

            for run in batch_runs.values():
                modified = False
                if remove_tags and any(tag in (run.tags or []) for tag in remove_tags):
                    run.tags = list(
                        set(tag for tag in (run.tags or []) if tag not in remove_tags)
                    )
                    modified = True
                if add_tags and any(tag not in (run.tags or []) for tag in add_tags):
                    run.tags = list(set((run.tags or []) + add_tags))
                    modified = True
                if modified:
                    run.save_to_file()

        # all the runs we needed to tag minus the runs we did tag
        failed_runs = list(run_ids_set - runs_found_set)
        if failed_runs:
            raise HTTPException(
                status_code=500,
                detail={
                    "failed_runs": failed_runs,
                    "error": "Runs not found",
                },
            )
        return {"success": True}

    @app.post(
        "/api/projects/{project_id}/tasks/{task_id}/runs/bulk_upload",
        summary="Bulk Upload Runs",
        tags=["Runs"],
        openapi_extra=ALLOW_AGENT,
    )
    async def bulk_upload(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        file: Annotated[
            UploadFile, File(description="The CSV file containing run data to import.")
        ],
        splits: Annotated[
            str | None,
            Form(
                description="JSON string mapping split names to numeric proportions (0-1)."
            ),
        ] = None,
    ) -> BulkUploadResponse:
        task = task_from_id(project_id, task_id)

        # Parse splits from json form data
        splits_dict = parse_splits(splits)

        # store the file in temp directory
        file_name = file.filename if file.filename else "untitled"
        file_path = os.path.join(
            tempfile.gettempdir(),
            file_name,
        )
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            importer = DatasetFileImporter(
                task,
                ImportConfig(
                    dataset_type=DatasetImportFormat.CSV,
                    dataset_path=file_path,
                    dataset_name=file_name,
                    tag_splits=splits_dict,
                ),
            )
            import_result = importer.create_runs_from_file()
        except KilnInvalidImportFormat as e:
            logger.error(
                f"Invalid import format in {file_name}: {e!s}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=422,
                detail=str(e),
            )

        return BulkUploadResponse(
            success=True,
            filename=file_name,
            imported_count=import_result.imported_run_count,
            imported_conversation_count=import_result.imported_conversation_count,
        )

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/tags",
        summary="List Run Tags",
        description=(
            "Counts only include tags from leaf TaskRuns. For multi-turn tasks, "
            "tags attached to intermediate runs in a chain are not included."
        ),
        tags=["Runs"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_tags(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
    ) -> dict[str, int]:
        tags_count = {}
        task = task_from_id(project_id, task_id)
        # Not particularly efficient, but tasks are memory cached after first load so re-compute is fairly cheap
        # We also cache the result client side
        for run in task.runs(readonly=True):
            for tag in run.tags:
                tags_count[tag] = tags_count.get(tag, 0) + 1
        return tags_count


async def update_run_util(
    project_id: str, task_id: str, run_id: str, run_data: Dict[str, Any]
) -> TaskRun:
    # eval_source marks a run as a trace the eval runner generated. It hides the run from
    # the dataset and blocks deletion, so setting it here would let a client make an
    # ordinary run both invisible and permanently undeletable - the delete guard reads
    # the same field, so there'd be no way back through the API. It is written by the
    # eval runner in-process and by nothing else.
    #
    # Testing for the key rather than a set value also refuses `{"eval_source": null}`,
    # which deep_update treats as a clear. That is deliberate: an escape hatch out of the
    # delete guard would be the guard's own bypass. The accepted cost is that a trace
    # orphaned by deleting its eval - deleting an Eval removes its EvalRuns but not the
    # TaskRuns they scored - stays on disk, invisible to the dataset and removable only
    # from the filesystem.
    if "eval_source" in run_data:
        raise HTTPException(
            status_code=400,
            detail="eval_source cannot be set by client. It marks a run as generated by an eval.",
        )

    # Lock to prevent overwriting concurrent updates
    async with update_run_lock:
        task = task_from_id(project_id, task_id)

        run = TaskRun.from_id_and_parent_path(run_id, task.path)
        if run is None:
            raise HTTPException(
                status_code=404,
                detail=f"Run not found. ID: {run_id}",
            )

        # Update and save
        old_run_dumped = run.model_dump()
        merged = deep_update(old_run_dumped, run_data)
        updated_run = TaskRun.model_validate(merged)
        updated_run.path = run.path
        updated_run.save_to_file()
        return updated_run


def model_provider_from_string(provider: str) -> ModelProviderName:
    if not provider or provider not in ModelProviderName.__members__:
        raise ValueError(f"Unsupported provider: {provider}")
    return ModelProviderName(provider)


def parse_splits(splits: str | None) -> Dict[str, float] | None:
    # Parse splits from form data
    if not splits:
        return None
    try:
        splits_dict = json.loads(splits)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422,
            detail="Invalid splits format. Must be a valid JSON object with string keys and float values.",
        )

    if (
        not isinstance(splits_dict, dict)
        or not all(isinstance(k, str) for k in splits_dict.keys())
        or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in splits_dict.values()
        )
        or not all(0 <= float(v) <= 1 for v in splits_dict.values())
    ):
        raise HTTPException(
            status_code=422,
            detail="Invalid splits format. Must be a valid JSON object with string keys and float values.",
        )

    return splits_dict
