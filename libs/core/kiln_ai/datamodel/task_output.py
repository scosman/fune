import json
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, ValidationInfo, model_validator
from typing_extensions import Self

from kiln_ai.datamodel.basemodel import ID_TYPE, KilnBaseModel
from kiln_ai.datamodel.datamodel_enums import TaskOutputRatingType
from kiln_ai.datamodel.json_schema import validate_schema_with_value_error
from kiln_ai.datamodel.run_config import RunConfigProperties
from kiln_ai.datamodel.strict_mode import strict_mode
from kiln_ai.utils.exhaustive_error import raise_exhaustive_enum_error

if TYPE_CHECKING:
    from kiln_ai.datamodel.task import Task


TASK_OUTPUT_SCHEMA_ERROR_PREFIX = (
    "This task requires a specific output schema. While the model produced JSON, "
    "that JSON didn't meet the schema. Search 'Troubleshooting Structured Data "
    "Issues' in our docs for more information."
)
"""User-facing prefix for a model output failing the task's output schema.
Single source of truth: retry classification recognizes these errors by it."""


class RequirementRating(BaseModel):
    """Rating for a specific requirement within a task output."""

    value: float = Field(
        description="The rating value. Interpretation depends on rating type"
    )
    type: TaskOutputRatingType = Field(description="The type of rating")


def normalize_rating(rating: float, rating_type: TaskOutputRatingType) -> float:
    """Normalize a rating to a 0-1 scale. Simple normalization, not z-score."""
    match rating_type:
        case TaskOutputRatingType.five_star:
            if rating < 1 or rating > 5:
                raise ValueError("Five star rating must be between 1 and 5")
            return (rating - 1) / 4
        case TaskOutputRatingType.pass_fail:
            if rating < 0 or rating > 1:
                raise ValueError("Pass fail rating must 0 to 1")
            return rating
        case TaskOutputRatingType.pass_fail_critical:
            if rating < -1 or rating > 1:
                raise ValueError("Pass fail critical rating must -1 to 1")
            return (rating + 1) / 2  # -1 to 1
        case TaskOutputRatingType.custom:
            raise ValueError("Custom rating type can not be normalized")
        case _:
            raise_exhaustive_enum_error(rating_type)


class TaskOutputRating(KilnBaseModel):
    """
    A rating for a task output, including an overall rating and ratings for each requirement.

    Supports:
    - five_star: 1-5 star ratings
    - pass_fail: boolean pass/fail (1.0 = pass, 0.0 = fail)
    - pass_fail_critical: tri-state (1.0 = pass, 0.0 = fail, -1.0 = critical fail)
    """

    type: TaskOutputRatingType = Field(
        default=TaskOutputRatingType.five_star,
        description="The rating system used for this rating.",
    )
    value: float | None = Field(
        description="The rating value. Interpretation depends on rating type:\n- five_star: 1-5 stars\n- pass_fail: 1.0 (pass) or 0.0 (fail)\n- pass_fail_critical: 1.0 (pass), 0.0 (fail), or -1.0 (critical fail)",
        default=None,
    )
    requirement_ratings: Dict[ID_TYPE, RequirementRating] = Field(
        default={},
        description="The ratings of the requirements of the task. The ID can be either a task_requirement_id or a named rating for an eval_output_score name (in format 'named::<name>').",
    )

    # Previously we stored rating values as a dict of floats, but now we store them as RequirementRating objects.
    @model_validator(mode="before")
    def upgrade_old_format(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data

        # Check if we have the old format (dict of floats)
        req_ratings = data.get("requirement_ratings", {})
        if req_ratings and all(
            isinstance(v, (int, float)) for v in req_ratings.values()
        ):
            # Convert each float to a RequirementRating object
            # all ratings are five star at the point we used this format
            data["requirement_ratings"] = {
                k: {"value": v, "type": TaskOutputRatingType.five_star}
                for k, v in req_ratings.items()
            }

        return data

    # Used to select high quality outputs for example selection (MultiShotPromptBuilder, etc)
    def is_high_quality(self) -> bool:
        if self.value is None:
            return False

        if self.type == TaskOutputRatingType.five_star:
            return self.value >= 4
        elif self.type == TaskOutputRatingType.pass_fail:
            return self.value == 1.0
        elif self.type == TaskOutputRatingType.pass_fail_critical:
            return self.value == 1.0
        return False

    @model_validator(mode="after")
    def validate_rating(self) -> Self:
        if self.type not in TaskOutputRatingType:
            raise ValueError(f"Invalid rating type: {self.type}")

        # Overall rating is optional
        if self.value is not None:
            self._validate_rating(self.type, self.value, "overall rating")

        for req_id, req_rating in self.requirement_ratings.items():
            self._validate_rating(
                req_rating.type,
                req_rating.value,
                f"requirement rating for req ID: {req_id}",
            )

        return self

    def _validate_rating(
        self, type: TaskOutputRatingType, rating: float | None, rating_name: str
    ) -> None:
        if type == TaskOutputRatingType.five_star:
            self._validate_five_star(rating, rating_name)
        elif type == TaskOutputRatingType.pass_fail:
            self._validate_pass_fail(rating, rating_name)
        elif type == TaskOutputRatingType.pass_fail_critical:
            self._validate_pass_fail_critical(rating, rating_name)

    def _validate_five_star(self, rating: float | None, rating_name: str) -> None:
        if rating is None or not isinstance(rating, float) or not rating.is_integer():
            raise ValueError(
                f"{rating_name.capitalize()} of type five_star must be an integer value (1-5)"
            )
        if rating < 1 or rating > 5:
            raise ValueError(
                f"{rating_name.capitalize()} of type five_star must be between 1 and 5 stars"
            )

    def _validate_pass_fail(self, rating: float | None, rating_name: str) -> None:
        if rating is None or not isinstance(rating, float) or not rating.is_integer():
            raise ValueError(
                f"{rating_name.capitalize()} of type pass_fail must be an integer value (0 or 1)"
            )
        if rating not in [0, 1]:
            raise ValueError(
                f"{rating_name.capitalize()} of type pass_fail must be 0 (fail) or 1 (pass)"
            )

    def _validate_pass_fail_critical(
        self, rating: float | None, rating_name: str
    ) -> None:
        if rating is None or not isinstance(rating, float) or not rating.is_integer():
            raise ValueError(
                f"{rating_name.capitalize()} of type pass_fail_critical must be an integer value (-1, 0, or 1)"
            )
        if rating not in [-1, 0, 1]:
            raise ValueError(
                f"{rating_name.capitalize()} of type pass_fail_critical must be -1 (critical fail), 0 (fail), or 1 (pass)"
            )


class DataSourceType(str, Enum):
    """
    The source type of a piece of data.

    Human: a human created the data
    Synthetic: a model created the data
    """

    human = "human"
    synthetic = "synthetic"
    file_import = "file_import"
    tool_call = "tool_call"


class DataSourceProperty(BaseModel):
    """
    Defines a property that can be associated with a data source.

    Includes validation rules for when properties are required or not allowed
    based on the data source type.
    """

    name: str
    type: Type[Union[str, int, float]]
    required_for: List[DataSourceType] = []
    not_allowed_for: List[DataSourceType] = []


class DataSource(BaseModel):
    """
    Represents the origin of data, either human, synthetic, file import, or tool call, with associated properties.

    Properties vary based on the source type - for synthetic/tool_call sources this includes
    model information, for human sources this includes creator information, for file imports
    this includes file information.
    """

    type: DataSourceType = Field(description="The type of data source.")
    properties: Dict[str, str | int | float] = Field(
        default={},
        description="Properties describing the data source. For synthetic things like model. For human: the human's name. For file_import: file information.",
    )
    run_config_id: Optional[str] = Field(
        default=None,
        description="The ID of the saved TaskRunConfig used to produce this data, if any. Only present when the run was initiated from a saved TaskRunConfig (e.g. via the saved-config dropdown or a tool call); runs configured ad-hoc from the run page leave this unset. Not validated against the file system: a TaskRunConfig may be deleted without invalidating historical runs that reference it.",
    )
    run_config: Optional[RunConfigProperties] = Field(
        default=None,
        description="The run config used to generate the data, if generated by a running a model in Kiln (only true for type=synthetic).",
    )

    @model_validator(mode="after")
    def normalize_empty_run_config_id(self) -> Self:
        # Some callers (e.g. tool wrappers reading from external properties)
        # default missing IDs to "". Coerce to None so this field is either a
        # real ID or unset — never an ambiguous empty string on disk.
        if self.run_config_id == "":
            self.run_config_id = None
        return self

    _data_source_properties = [
        DataSourceProperty(
            name="created_by",
            type=str,
            required_for=[DataSourceType.human],
            not_allowed_for=[
                DataSourceType.synthetic,
                DataSourceType.file_import,
                DataSourceType.tool_call,
            ],
        ),
        DataSourceProperty(
            name="model_name",
            type=str,
            required_for=[DataSourceType.synthetic],
            not_allowed_for=[
                DataSourceType.human,
                DataSourceType.file_import,
                DataSourceType.tool_call,
            ],
        ),
        DataSourceProperty(
            name="model_provider",
            type=str,
            required_for=[DataSourceType.synthetic],
            not_allowed_for=[
                DataSourceType.human,
                DataSourceType.file_import,
                DataSourceType.tool_call,
            ],
        ),
        DataSourceProperty(
            name="adapter_name",
            type=str,
            required_for=[DataSourceType.synthetic],
            not_allowed_for=[
                DataSourceType.human,
                DataSourceType.file_import,
                DataSourceType.tool_call,
            ],
        ),
        DataSourceProperty(
            # Legacy field -- allow loading from old runs, but we shouldn't be setting it.
            name="prompt_builder_name",
            type=str,
            not_allowed_for=[
                DataSourceType.human,
                DataSourceType.file_import,
                DataSourceType.tool_call,
            ],
        ),
        DataSourceProperty(
            # The PromptId of the prompt. Can be a saved prompt, fine-tune, generator name, etc. See PromptId type for more details.
            name="prompt_id",
            type=str,
            not_allowed_for=[
                DataSourceType.human,
                DataSourceType.file_import,
                DataSourceType.tool_call,
            ],
        ),
        DataSourceProperty(
            name="file_name",
            type=str,
            required_for=[DataSourceType.file_import],
            not_allowed_for=[
                DataSourceType.human,
                DataSourceType.synthetic,
                DataSourceType.tool_call,
            ],
        ),
    ]

    @model_validator(mode="after")
    def validate_type(self) -> "DataSource":
        if self.type not in DataSourceType:
            raise ValueError(f"Invalid data source type: {self.type}")
        return self

    @model_validator(mode="after")
    def validate_properties(self) -> "DataSource":
        for prop in self._data_source_properties:
            # Check the property type is correct
            if prop.name in self.properties:
                if not isinstance(self.properties[prop.name], prop.type):
                    raise ValueError(
                        f"'{prop.name}' must be of type {prop.type.__name__} for {self.type} data source"
                    )
            # Check the property is required for the data source type
            if self.type in prop.required_for:
                if prop.name not in self.properties:
                    raise ValueError(
                        f"'{prop.name}' is required for {self.type} data source"
                    )
            # Check the property is not allowed for the data source type
            elif self.type in prop.not_allowed_for and prop.name in self.properties:
                raise ValueError(
                    f"'{prop.name}' is not allowed for {self.type} data source"
                )
        return self

    @model_validator(mode="after")
    def validate_no_empty_properties(self) -> Self:
        for prop, value in self.properties.items():
            if isinstance(value, str) and value == "":
                raise ValueError(
                    f"Property '{prop}' must be a non-empty string for {self.type} data source"
                )
        return self


class TaskOutput(KilnBaseModel):
    """
    An output for a specific task run.

    Contains the actual output content, its source (human or synthetic),
    and optional rating information.
    """

    output: str = Field(
        description="The output of the task. JSON formatted for structured output, plaintext for unstructured output."
    )
    source: DataSource | None = Field(
        description="The source of the output: human or synthetic.",
        default=None,
    )
    rating: TaskOutputRating | None = Field(
        default=None, description="The rating of the output"
    )

    def validate_output_format(self, task: "Task") -> Self:
        # validate output
        if task.output_json_schema is not None:
            try:
                output_parsed = json.loads(self.output)
            except json.JSONDecodeError:
                raise ValueError("Output is not a valid JSON object")

            validate_schema_with_value_error(
                output_parsed,
                task.output_json_schema,
                TASK_OUTPUT_SCHEMA_ERROR_PREFIX,
            )
        return self

    @model_validator(mode="after")
    def validate_output_source(self, info: ValidationInfo) -> Self:
        # On strict mode and not loaded from file, we validate output_source is not None.
        # We want to be able to load any data, even if it's not perfect. But we want to create perfect data when adding new data.
        if not strict_mode():
            return self
        if self.loaded_from_file(info):
            return self
        if self.source is None:
            raise ValueError("Output source is required when strict mode is enabled")
        return self
