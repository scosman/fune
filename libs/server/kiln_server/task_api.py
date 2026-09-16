import logging
from typing import Annotated, Any, Dict, List

from fastapi import Body, FastAPI, HTTPException, Path
from kiln_ai.datamodel import Project, Task, TaskRequirement
from kiln_ai.datamodel.datamodel_enums import TurnMode
from kiln_ai.datamodel.external_tool_server import (
    ToolServerType,
)
from kiln_ai.utils.config import Config
from kiln_ai.utils.formatting import truncate_to_words_with_agent_sentinel
from pydantic import BaseModel, Field

from kiln_server.project_api import project_from_id
from kiln_server.utils.agent_checks.policy import (
    ALLOW_AGENT,
    DENY_AGENT,
    agent_policy_require_approval,
)

logger = logging.getLogger(__name__)


class TaskSummary(BaseModel):
    id: str
    name: str
    description: str | None
    instruction: str


class TaskSummariesProject(BaseModel):
    id: str
    name: str
    description: str | None
    tasks: list[TaskSummary]


class TaskSummariesResponse(BaseModel):
    projects: list[TaskSummariesProject]


def task_from_id(project_id: str, task_id: str) -> Task:
    parent_project = project_from_id(project_id)
    task = Task.from_id_and_parent_path(task_id, parent_project.path)
    if task:
        return task

    raise HTTPException(
        status_code=404,
        detail=f"Task not found. ID: {task_id}",
    )


class RatingOption(BaseModel):
    """A rating requirement with display rules."""

    requirement: TaskRequirement = Field(description="The task requirement to rate.")
    show_for_all: bool = Field(
        description="Whether this rating option is shown for all outputs."
    )
    show_for_tags: List[str] = Field(
        description="Tags for which this rating option is shown."
    )


class RatingOptionResponse(BaseModel):
    """The available rating options for a task."""

    options: List[RatingOption] = Field(description="The list of rating options.")


def connect_task_api(app: FastAPI):
    @app.post(
        "/api/projects/{project_id}/tasks",
        summary="Create Task",
        tags=["Tasks"],
        openapi_extra=ALLOW_AGENT,
    )
    async def create_task(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_data: Annotated[
            Dict[str, Any], Body(description="The task data to create.")
        ],
    ) -> Task:
        if "id" in task_data:
            raise HTTPException(
                status_code=400,
                detail="Task ID cannot be set by client.",
            )
        parent_project = project_from_id(project_id)

        task = Task.validate_and_save_with_subrelations(
            task_data, parent=parent_project
        )
        if task is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to create task.",
            )
        if not isinstance(task, Task):
            raise HTTPException(
                status_code=500,
                detail="Failed to create task.",
            )

        return task

    @app.patch(
        "/api/projects/{project_id}/tasks/{task_id}",
        summary="Update Task",
        tags=["Tasks"],
        openapi_extra=agent_policy_require_approval(
            "Allow agent to edit task? Ensure you backup your project before allowing agentic edits."
        ),
    )
    async def update_task(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
        task_updates: Annotated[
            Dict[str, Any], Body(description="Fields to update on the task.")
        ],
    ) -> Task:
        if "input_json_schema" in task_updates or "output_json_schema" in task_updates:
            raise HTTPException(
                status_code=400,
                detail="Input and output JSON schemas cannot be updated.",
            )
        if "id" in task_updates and task_updates["id"] != task_id:
            raise HTTPException(
                status_code=400,
                detail="Task ID cannot be changed by client in a patch.",
            )
        original_task = task_from_id(project_id, task_id)
        if (
            "turn_mode" in task_updates
            and task_updates["turn_mode"] != original_task.turn_mode.value
        ):
            raise HTTPException(
                status_code=400,
                detail="Task turn_mode cannot be changed after creation.",
            )
        if "turn_mode" in task_updates and isinstance(task_updates["turn_mode"], str):
            task_updates["turn_mode"] = TurnMode(task_updates["turn_mode"])
        updated_task_data = original_task.model_copy(update=task_updates)
        updated_task = Task.validate_and_save_with_subrelations(
            updated_task_data.model_dump(), parent=original_task.parent
        )
        if updated_task is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to update task.",
            )
        if not isinstance(updated_task, Task):
            raise HTTPException(
                status_code=500,
                detail="Failed to patch task.",
            )

        return updated_task

    @app.delete(
        "/api/projects/{project_id}/tasks/{task_id}",
        summary="Delete Task",
        tags=["Tasks"],
        openapi_extra=DENY_AGENT,
    )
    async def delete_task(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
    ) -> None:
        task = task_from_id(project_id, task_id)
        task.delete()

        # Archive any kiln task tools that have this task set as their task_id
        parent_project = task.parent_project()
        if parent_project is not None:
            for tool_server in parent_project.external_tool_servers():
                if (
                    tool_server.type == ToolServerType.kiln_task
                    and tool_server.properties.get("task_id") == task_id
                ):
                    # For kiln task tools, we know the properties are KilnTaskServerProperties
                    if "is_archived" in tool_server.properties:
                        tool_server.properties["is_archived"] = True
                    else:
                        raise TypeError("Expected archiveable tool task server")

                    tool_server.save_to_file()

    @app.get(
        "/api/projects/{project_id}/tasks",
        summary="List Tasks",
        tags=["Tasks"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_tasks(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
    ) -> List[Task]:
        parent_project = project_from_id(project_id)
        return parent_project.tasks()

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}",
        summary="Get Task",
        tags=["Tasks"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_task(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
    ) -> Task:
        return task_from_id(project_id, task_id)

    @app.get(
        "/api/projects/{project_id}/tasks/{task_id}/rating_options",
        summary="Get Rating Options",
        tags=["Tasks"],
        openapi_extra=ALLOW_AGENT,
    )
    async def get_rating_options(
        project_id: Annotated[
            str, Path(description="The unique identifier of the project.")
        ],
        task_id: Annotated[
            str,
            Path(description="The unique identifier of the task within the project."),
        ],
    ) -> RatingOptionResponse:
        """Determines which rating options should be shown for a given dataset item."""
        task = task_from_id(project_id, task_id)
        results: List[RatingOption] = []

        # First add all task requirements. We want these to be shown for all items.
        for requirement in task.requirements:
            results.append(
                RatingOption(
                    requirement=requirement,
                    show_for_all=True,
                    show_for_tags=[],
                )
            )

        # Then add eval requirements. We want these to be shown for all items in the eval's golden set filter.
        for eval in task.evals(readonly=True):
            if eval.eval_configs_filter_id is None:
                continue
            if not eval.eval_configs_filter_id.startswith("tag::"):
                logger.warning(
                    "Eval '%s' has non-tag filter '%s'. This isn't compatible with the web UI for automatic rating visibility.",
                    eval.id,
                    eval.eval_configs_filter_id,
                )
                continue
            golden_set_tag = eval.eval_configs_filter_id[len("tag::") :]

            for output_score in eval.output_scores:
                # Skip overall rating. It's added by default.
                if output_score.name == "Overall Rating":
                    continue

                # Check for existing requirement with this name
                existing_req = next(
                    (r for r in results if r.requirement.name == output_score.name),
                    None,
                )
                if existing_req:
                    # warn for type mismatch
                    if existing_req.requirement.type != output_score.type:
                        logger.warning(
                            "The rating option for '%s' has conflicting types: '%s' and '%s'. You shouldn't use the same name for goals of different rating types.",
                            output_score.name,
                            output_score.type,
                            existing_req.requirement.type,
                        )

                    if golden_set_tag not in existing_req.show_for_tags:
                        # Add the golden set tag to the existing requirement instead of creating a new one (unless that tag is already there)
                        existing_req.show_for_tags.append(golden_set_tag)
                    continue

                # Map eval requirements to task requirements
                requirement = TaskRequirement(
                    id="named::" + output_score.name,
                    name=output_score.name,
                    instruction=output_score.instruction or "No instructions provided",
                    type=output_score.type,
                )
                results.append(
                    RatingOption(
                        requirement=requirement,
                        show_for_all=False,
                        show_for_tags=[golden_set_tag],
                    )
                )

        return RatingOptionResponse(options=results)

    @app.get(
        "/api/task_summaries",
        summary="Task Summaries (agent-tuned)",
        tags=["Tasks"],
        openapi_extra=ALLOW_AGENT,
    )
    async def task_summaries() -> TaskSummariesResponse:
        """Return a workspace-wide list of projects and their tasks, with truncated
        task.instruction values. Unlike typical list endpoints, entries here are
        intentionally lossy — the shape is tuned for LLM-agent context efficiency,
        not for driving UIs that need the full Task model."""
        project_paths = Config.shared().projects or []
        projects: list[TaskSummariesProject] = []
        for project_path in project_paths:
            try:
                project = Project.load_from_file(project_path)
            except Exception:
                logger.warning(
                    "Failed to load project from path: %s", project_path, exc_info=True
                )
                continue
            if project.id is None:
                logger.warning("Project at %s has no ID, skipping", project_path)
                continue
            tasks: list[TaskSummary] = []
            for task in project.tasks(readonly=True):
                if task.id is None:
                    continue
                instruction = truncate_to_words_with_agent_sentinel(
                    task.instruction, 100
                )
                tasks.append(
                    TaskSummary(
                        id=task.id,
                        name=task.name,
                        description=task.description,
                        instruction=instruction or "",
                    )
                )
            projects.append(
                TaskSummariesProject(
                    id=project.id,
                    name=project.name,
                    description=project.description,
                    tasks=tasks,
                )
            )
        return TaskSummariesResponse(projects=projects)
