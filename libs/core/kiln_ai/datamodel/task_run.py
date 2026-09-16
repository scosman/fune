import json
from typing import TYPE_CHECKING, Dict, List, Literal, Union

from pydantic import BaseModel, Field, ValidationInfo, model_validator
from typing_extensions import Self

from kiln_ai.datamodel.basemodel import KilnParentedModel, KilnParentModel
from kiln_ai.datamodel.claim_review import ClaimReview
from kiln_ai.datamodel.feedback import Feedback
from kiln_ai.datamodel.json_schema import validate_schema_with_value_error
from kiln_ai.datamodel.strict_mode import strict_mode
from kiln_ai.datamodel.task_output import DataSource, TaskOutput
from kiln_ai.utils.open_ai_types import (
    ChatCompletionMessageParam,
    materialize_lazy_content,
    trace_has_pending_client_tool_calls,
)
from kiln_ai.utils.usage import MessageUsage as MessageUsage
from kiln_ai.utils.usage import Usage as Usage

if TYPE_CHECKING:
    from kiln_ai.datamodel.eval_splits import ItemKey
    from kiln_ai.datamodel.task import Task


# Lives here rather than in eval.py: eval.py imports TaskRun under TYPE_CHECKING only,
# so a TaskRun field typed from eval.py would make that a real import cycle.
# eval_splits.py already imports from this module, so building an ItemKey from one of
# these is a direction that already exists.
class EvalItemSource(BaseModel):
    """The eval dataset item a TaskRun was generated for.

    Not the run config — that lives on the same TaskRun at
    `output.source.run_config_id`.
    """

    source_type: Literal["eval_input", "task_run"] = Field(
        description="Which store the dataset item came from: an EvalInput (V2) or a TaskRun (V1-backed split)."
    )
    # `str`, not the usual `ID_TYPE` (`Optional[str]`): an id-less source is a trace-index
    # key that collides with every other id-less source. Absence is already expressed by
    # `TaskRun.eval_source` itself being None, so the inner id has no legitimate None
    # state. `ItemKey`'s own nullability is inherited from `KilnBaseModel.id` rather than
    # chosen, and a narrower id still satisfies it.
    source_id: str = Field(
        min_length=1,
        description="The id of the dataset item this run was generated for. Interpreted within the store named by source_type — ids are only unique within a store.",
    )


def eval_item_key(source: EvalItemSource) -> "ItemKey":
    """The dataset item an eval trace was generated for, as an ItemKey.

    The counterpart to `eval_splits.eval_run_item_key()`: both turn a record's stored
    source fields into the one vocabulary splits, the runner and the trace index share.
    """
    return (source.source_type, source.source_id)


class TaskRun(
    KilnParentedModel,
    KilnParentModel,
    parent_of={
        "feedback": Feedback,
        "claim_reviews": ClaimReview,
    },
):
    """
    Represents a single execution of a Task.

    Contains the input used, its source, the output produced, and optional
    repair information if the output needed correction.
    """

    input: str = Field(
        description="The inputs to the task. JSON formatted for structured input, plaintext for unstructured input."
    )
    input_source: DataSource | None = Field(
        default=None, description="The source of the input: human or synthetic."
    )

    output: TaskOutput = Field(description="The output of the task run.")
    repair_instructions: str | None = Field(
        default=None,
        description="Instructions for fixing the output. Should define what is wrong, and how to fix it. Will be used by models for both generating a fixed output, and evaluating future models.",
    )
    repaired_output: TaskOutput | None = Field(
        default=None,
        description="An version of the output with issues fixed. This must be a 'fixed' version of the existing output, and not an entirely new output. If you wish to generate an ideal curatorial output for this task unrelated to this output, generate a new TaskOutput with type 'human' instead of using this field.",
    )
    intermediate_outputs: Dict[str, str] | None = Field(
        default=None,
        description="Intermediate outputs from the task run. Keys are the names of the intermediate output steps (cot=chain of thought, etc), values are the output data.",
    )
    tags: List[str] = Field(
        default=[],
        description="Tags for the task run. Tags are used to categorize task runs for filtering and reporting.",
    )
    usage: Usage | None = Field(
        default=None,
        description="Usage information for the task run. This includes the number of input tokens, output tokens, and total tokens used.",
    )
    cumulative_usage: MessageUsage | None = Field(
        default=None,
        description=(
            "Sum of per-message token usage and cost across the entire trace, "
            "including any seeded prior trace. None on records created before "
            "this field existed. For a fresh (non-seeded) run, the token / "
            "cost fields equal those of `usage`."
        ),
    )
    synthetic_user_usage: Usage | None = Field(
        default=None,
        description=(
            "The synthetic-user driver model's spend for an eval-driven "
            "conversation, recorded beside the assistant's own usage so `usage` "
            "stays assistant-only. None for ordinary runs, and for migrated "
            "legacy traces whose driver cost is fused into `usage`."
        ),
    )
    trace: list[ChatCompletionMessageParam] | None = Field(
        default=None,
        description="The trace of the task run in OpenAI format. This is the list of messages that were sent to/from the model.",
    )
    parent_task_run_id: str | None = Field(
        default=None,
        description="The ID of the parent task run. This is the ID of the task run that contains this task run.",
    )
    eval_source: EvalItemSource | None = Field(
        default=None,
        description="Set when this run was generated by an eval. Names the eval dataset item it was generated for. None for ordinary dataset runs. Runs with this set are excluded from Task.runs() by default, so they do not appear on dataset surfaces.",
    )

    @model_validator(mode="after")
    def materialize_trace_content(self) -> Self:
        # Pydantic validates the trace wrappers' Iterable-typed `content` into a lazy,
        # single-use iterator that cannot be deepcopied. Materialize it here so any
        # later copy of the run (e.g. the model cache's mutable copies) is safe.
        if self.trace is not None:
            materialize_lazy_content(self.trace)
        return self

    @property
    def is_toolcall_pending(self) -> bool:
        """True if the trace ends with an assistant message awaiting client tool execution."""
        return trace_has_pending_client_tool_calls(self.trace)

    def thinking_training_data(self) -> str | None:
        """
        Get the thinking training data from the task run.
        """
        if self.intermediate_outputs is None:
            return None
        return self.intermediate_outputs.get(
            "reasoning"
        ) or self.intermediate_outputs.get("chain_of_thought")

    def has_thinking_training_data(self) -> bool:
        """
        Does this run have thinking data that we can use to train a thinking model?
        """
        return self.thinking_training_data() is not None

    def feedback(self, readonly: bool = False) -> list[Feedback]:
        return super().feedback(readonly=readonly)  # type: ignore

    def claim_reviews(self, readonly: bool = False) -> list[ClaimReview]:
        return super().claim_reviews(readonly=readonly)  # type: ignore

    # Workaround to return typed parent without importing Task
    def parent_task(self) -> Union["Task", None]:
        if self.parent is None or self.parent.__class__.__name__ != "Task":
            return None
        return self.parent  # type: ignore

    @model_validator(mode="after")
    def validate_input_format(self, info: ValidationInfo) -> Self:
        # Don't validate if loading from file (not new). Too slow.
        # We don't allow changing task schema, so this is redundant validation.
        # Note: we still validate if editing a loaded model
        if self.loading_from_file(info):
            # Consider loading an existing model as validated.
            self._last_validated_input = self.input
            return self

        # Don't validate if input has not changed. Too slow to run this every time.
        if (
            hasattr(self, "_last_validated_input")
            and self.input == self._last_validated_input
        ):
            return self

        task = self.parent_task()
        if task is None:
            # don't validate this relationship until we have a path or parent. Give them time to build it (but will catch it before saving)
            return self

        # validate input
        if task.input_json_schema is not None:
            try:
                input_parsed = json.loads(self.input)
            except json.JSONDecodeError:
                raise ValueError("Input is not a valid JSON object")

            validate_schema_with_value_error(
                input_parsed,
                task.input_json_schema,
                "Input does not match task input schema.",
                require_object=False,
            )

        self._last_validated_input = self.input
        return self

    @model_validator(mode="after")
    def validate_output_format(self, info: ValidationInfo) -> Self:
        # Don't validate if loading from file (not new). Too slow.
        # Note: we still validate if editing a loaded model's output.
        if self.loading_from_file(info):
            # Consider loading an existing model as validated.
            self._last_validated_output = self.output.output if self.output else None
            return self

        # Skip output validation when the run is waiting for tool call results.
        # The output field is empty/partial in this state.
        if self.is_toolcall_pending:
            self._last_validated_output = self.output.output if self.output else None
            return self

        # Don't validate unless output has changed since last validation.
        # The validator is slow and costly, don't want it running when setting other fields.
        if (
            hasattr(self, "_last_validated_output")
            and self.output is not None
            and self.output.output == self._last_validated_output
        ):
            return self

        task = self.parent_task()
        if task is None:
            return self

        self.output.validate_output_format(task)
        self._last_validated_output = self.output.output if self.output else None
        return self

    @model_validator(mode="after")
    def validate_parent_task_run_id_for_turn_mode(self, info: ValidationInfo) -> Self:
        # Single-turn tasks must not have a parent_task_run_id - the leaf
        # filter in Task.runs() would silently drop the parent from
        # iteration, producing phantom runs in evals/finetunes/dataset
        # splits. Skip on load so legacy files don't fail to deserialize.
        if self.loading_from_file(info):
            return self
        if self.parent_task_run_id is None:
            return self
        task = self.parent_task()
        if task is None:
            # Not yet attached - defer; revalidates when attached/saved.
            return self
        # Avoid circular import at module load.
        from kiln_ai.datamodel.datamodel_enums import TurnMode

        if task.turn_mode != TurnMode.multiturn:
            raise ValueError(
                "parent_task_run_id is only valid on multi-turn tasks. "
                "This task's turn_mode is single_turn."
            )
        return self

    @model_validator(mode="after")
    def validate_repaired_output(self) -> Self:
        if self.repaired_output is not None:
            if self.repaired_output.rating is not None:
                raise ValueError(
                    "Repaired output rating must be None. Repaired outputs are assumed to have a perfect rating, as they have been fixed."
                )

            task = self.parent_task()
            if (
                task is not None
                and self.repaired_output.output is not None
                and task.output_json_schema is not None
            ):
                try:
                    output_parsed = json.loads(self.repaired_output.output)
                except json.JSONDecodeError:
                    raise ValueError("Repaired output is not a valid JSON object")

                validate_schema_with_value_error(
                    output_parsed,
                    task.output_json_schema,
                    "Repaired output does not match task output schema.",
                )

        if self.repair_instructions is None and self.repaired_output is not None:
            raise ValueError(
                "Repair instructions are required if providing a repaired output."
            )
        if self.repair_instructions is not None and self.repaired_output is None:
            raise ValueError(
                "A repaired output is required if providing repair instructions."
            )

        return self

    @model_validator(mode="after")
    def validate_input_source(self, info: ValidationInfo) -> Self:
        # On strict mode and not loaded from file, we validate input_source is not None.
        # We want to be able to load any data, even if it's not perfect. But we want to create perfect data when adding new data.
        if not strict_mode():
            return self
        if self.loaded_from_file(info):
            return self
        if self.input_source is None:
            raise ValueError("input_source is required when strict mode is enabled")
        return self

    @model_validator(mode="after")
    def validate_tags(self) -> Self:
        for tag in self.tags:
            if not tag:
                raise ValueError("Tags cannot be empty strings")
            if " " in tag:
                raise ValueError("Tags cannot contain spaces. Try underscores.")

        return self
