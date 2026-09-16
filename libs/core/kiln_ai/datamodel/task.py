from typing import TYPE_CHECKING, Dict, List, Union

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from kiln_ai.datamodel.basemodel import (
    ID_FIELD,
    ID_TYPE,
    FilenameString,
    FilenameStringShort,
    KilnParentedModel,
    KilnParentModel,
    ParentOfRelationship,
)
from kiln_ai.datamodel.data_guide import DataGuide
from kiln_ai.datamodel.datamodel_enums import (
    Priority,
    StructuredOutputMode,
    TaskOutputRatingType,
    TurnMode,
)
from kiln_ai.datamodel.dataset_split import DatasetSplit
from kiln_ai.datamodel.eval import Eval, EvalInput
from kiln_ai.datamodel.finetune import Finetune
from kiln_ai.datamodel.json_schema import (
    JsonObjectSchema,
    JsonSchema,
    schema_from_json_str,
)
from kiln_ai.datamodel.prompt import BasePrompt, Prompt
from kiln_ai.datamodel.prompt_optimization_job import PromptOptimizationJob
from kiln_ai.datamodel.run_config import RunConfigProperties
from kiln_ai.datamodel.spec import Spec
from kiln_ai.datamodel.task_run import TaskRun

if TYPE_CHECKING:
    from kiln_ai.datamodel.project import Project


class TaskRequirement(BaseModel):
    """
    Defines a specific requirement that should be met by task outputs.

    Includes an identifier, name, description, instruction for meeting the requirement,
    priority level, and rating type (five_star, pass_fail, pass_fail_critical, custom).
    """

    id: ID_TYPE = ID_FIELD
    name: FilenameStringShort = Field(description="The name of the task requirement.")
    description: str | None = Field(
        default=None,
        description="Optional elaboration on the requirement's purpose.",
    )
    instruction: str = Field(
        min_length=1, description="Instructions for meeting the requirement."
    )
    priority: Priority = Field(
        default=Priority.p2, description="The priority level of the requirement."
    )
    type: TaskOutputRatingType = Field(
        default=TaskOutputRatingType.five_star,
        description="The rating type used to evaluate this requirement.",
    )


class TaskRunConfig(KilnParentedModel):
    """
    A Kiln model for persisting a run config in a Kiln Project, nested under a task.

    Typically used to save a method of running a task for evaluation.

    A run config includes everything needed to run a task, except the input. Running the same RunConfig with the same input should make identical calls to the model (output may vary as models are non-deterministic).
    """

    name: FilenameString = Field(description="The name of the task run config.")
    description: str | None = Field(
        default=None, description="The description of the task run config."
    )
    run_config_properties: RunConfigProperties = Field(
        description="The run config properties to use for this task run."
    )
    # The prompt_id in the run_config_properties is the prompt ID to use for this task run.
    # However, we want the prompt to be perfectly consistent, and some prompt_ids are dynamic.
    # If we need to "freeze" a prompt, we can do so here (then point the prompt_id to this frozen prompt).
    prompt: BasePrompt | None = Field(
        default=None,
        description="A prompt to use for run config.",
    )
    starred: bool = Field(
        default=False,
        description="Whether this run config is starred/favourited by the user.",
    )

    # Workaround to return typed parent without importing Task
    def parent_task(self) -> Union["Task", None]:
        if self.parent is None or self.parent.__class__.__name__ != "Task":
            return None
        return self.parent  # type: ignore

    # Previously we didn't store structured_output_mode in the run_config_properties. Upgrade old models when loading from file.
    @model_validator(mode="before")
    def upgrade_old_entries(cls, data: dict, info: ValidationInfo) -> dict:
        if not info.context or not info.context.get("loading_from_file", False):
            # Not loading from file, so no need to upgrade
            return data

        if not isinstance(data, dict):
            return data

        run_config_properties = data.get("run_config_properties")
        if not isinstance(run_config_properties, dict):
            return data

        run_config_properties.setdefault("type", "kiln_agent")

        if run_config_properties.get("type") == "kiln_agent":
            structured_output_mode = run_config_properties.get(
                "structured_output_mode", None
            )
            if structured_output_mode is None:
                # Default to unknown. Adapter will have to guess at runtime.
                run_config_properties["structured_output_mode"] = (
                    StructuredOutputMode.unknown
                )

        return data


class Task(
    KilnParentedModel,
    KilnParentModel,
    parent_of={
        "_runs": ParentOfRelationship(model=TaskRun, filesystem_name="runs"),
        "dataset_splits": DatasetSplit,
        "finetunes": Finetune,
        "prompt_optimization_jobs": PromptOptimizationJob,
        "prompts": Prompt,
        "evals": Eval,
        "eval_inputs": EvalInput,
        "specs": Spec,
        "run_configs": TaskRunConfig,
        "data_guides": DataGuide,
    },
):
    """
    Represents a specific task to be performed, with associated requirements and validation rules.

    Contains the task definition, requirements, input/output schemas, and maintains
    a collection of task runs.
    """

    name: FilenameString = Field(description="The name of the task.")
    description: str | None = Field(
        default=None,
        description="A description of the task for you and your team. Will not be used in prompts/training/validation.",
    )
    instruction: str = Field(
        min_length=1,
        description="The instructions for the task. Will be used in prompts/training/validation.",
    )
    requirements: List[TaskRequirement] = Field(
        default=[],
        description="Deprecated: Use specs and prompts instead.",
    )
    output_json_schema: JsonObjectSchema | None = Field(
        default=None,
        description="JSON schema for structured task output. Must be an object schema.",
    )
    input_json_schema: JsonSchema | None = Field(
        default=None,
        description="JSON schema for structured task input. Can be an object or array schema.",
    )
    thinking_instruction: str | None = Field(
        default=None,
        description="Instructions for the model 'thinking' about the requirement prior to answering. Used for chain of thought style prompting.",
    )

    default_run_config_id: ID_TYPE | None = Field(
        default=None,
        description="ID of the run config to use for this task by default. Must exist in saved run configs for this task.",
    )

    turn_mode: TurnMode = Field(
        default=TurnMode.single_turn,
        frozen=True,
        description="Whether this task is single-turn (each run independent) or multi-turn (runs continue prior runs). Immutable after construction: changing it would invalidate existing TaskRuns. To change, clone the task.",
    )

    @model_validator(mode="after")
    def validate_turn_mode_compatibility(self) -> "Task":
        if self.turn_mode == TurnMode.multiturn:
            if self.input_json_schema is not None:
                raise ValueError(
                    "Multi-turn tasks cannot have a structured input schema. "
                    "Use plaintext input for multi-turn tasks."
                )
            if self.output_json_schema is not None:
                raise ValueError(
                    "Multi-turn tasks do not support structured output yet. "
                    "Use plaintext output, or set turn_mode to single_turn."
                )
        return self

    def output_schema(self) -> Dict | None:
        if self.output_json_schema is None:
            return None
        return schema_from_json_str(self.output_json_schema)

    def input_schema(self) -> Dict | None:
        if self.input_json_schema is None:
            return None
        # Allow arrays, not just objects
        return schema_from_json_str(self.input_json_schema, require_object=False)

    def runs(
        self,
        readonly: bool = False,
        include_intermediate_runs: bool = False,
        include_eval_generated: bool = False,
    ) -> list[TaskRun]:
        """Return TaskRuns for this task with leaf-only, dataset-only filtering by default.

        For multiturn tasks, child TaskRuns reference their parent via
        ``parent_task_run_id``. By default we return only the leaves of those
        chains - the runs that aren't a parent of any other run - because that
        is the right view for the vast majority of consumers iterating runs
        in-process (dataset/eval/finetune sample iteration, summary lists,
        statistics, tag counts, etc.).

        Pass ``include_intermediate_runs=True`` to get every run regardless of
        position in the chain. That is only correct for code that needs the
        complete on-disk set (e.g. walking ancestors, diagnostics). For
        single-turn tasks the two modes are equivalent.

        ``runs/`` also holds eval-generated traces (``eval_source`` set) alongside the
        dataset corpus. Those are excluded by default: they are a byproduct of running
        an eval, not data the user curated, and leaking them into fine-tune sets or
        few-shot prompts would feed a model its own eval outputs. Default-exclude rather
        than an opt-out filter is deliberate - forgetting to handle eval traces then
        fails visibly (missing data) instead of silently (contaminated data).
        ``include_eval_generated=True`` is meant for the eval runner alone, which needs
        to find the traces it can reuse; nothing else should pass it.

        The two filters compose: a run must pass both to be returned.

        Note: these filters only affect in-process iteration. Filesystem-level
        operations that copy the ``runs/`` directory (e.g. project export)
        copy every run regardless.
        """
        runs = self._runs(readonly=readonly)  # type: ignore[attr-defined]
        # Eval filter first: an excluded eval-generated run must not count as a parent
        # either, or an eval child chained onto a dataset run would hide that dataset
        # run from the default view while itself being filtered out.
        if not include_eval_generated:
            runs = [r for r in runs if r.eval_source is None]
        if not include_intermediate_runs:
            parent_ids = {r.parent_task_run_id for r in runs if r.parent_task_run_id}
            runs = [r for r in runs if r.id not in parent_ids]
        return runs

    # These wrappers help for typechecking. We should fix this in KilnParentModel
    def dataset_splits(self, readonly: bool = False) -> list[DatasetSplit]:
        return super().dataset_splits(readonly=readonly)  # type: ignore

    def finetunes(self, readonly: bool = False) -> list[Finetune]:
        return super().finetunes(readonly=readonly)  # type: ignore

    def prompts(self, readonly: bool = False) -> list[Prompt]:
        return super().prompts(readonly=readonly)  # type: ignore

    def evals(self, readonly: bool = False) -> list[Eval]:
        return super().evals(readonly=readonly)  # type: ignore

    def eval_inputs(self, readonly: bool = False) -> list[EvalInput]:
        return super().eval_inputs(readonly=readonly)  # type: ignore

    def run_configs(self, readonly: bool = False) -> list[TaskRunConfig]:
        return super().run_configs(readonly=readonly)  # type: ignore

    def specs(self, readonly: bool = False) -> list[Spec]:
        return super().specs(readonly=readonly)  # type: ignore

    def data_guides(self, readonly: bool = False) -> list[DataGuide]:
        return super().data_guides(readonly=readonly)  # type: ignore

    def current_data_guide(self, readonly: bool = False) -> DataGuide | None:
        # By design there is at most one DataGuide per task — saves overwrite
        # the existing one in place rather than creating a new file. If the
        # folder somehow ends up with multiple (e.g. an older import), return
        # the first; cleanup is up to the caller.
        guides = self.data_guides(readonly=readonly)
        return guides[0] if guides else None

    def prompt_optimization_jobs(
        self, readonly: bool = False
    ) -> list[PromptOptimizationJob]:
        return super().prompt_optimization_jobs(readonly=readonly)  # type: ignore

    # Workaround to return typed parent without importing Task
    def parent_project(self) -> Union["Project", None]:
        if self.parent is None or self.parent.__class__.__name__ != "Project":
            return None
        return self.parent  # type: ignore
