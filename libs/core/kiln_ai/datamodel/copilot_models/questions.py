"""
Data models for asking questions about a specification to refine it.

Copilot models are in /lib so they can be shared across lib, server, and client.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SpecQuestionerInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )
    task_prompt: str = Field(..., description="The task's prompt", title="task_prompt")
    task_input_schema: str | None = Field(
        None,
        description="If the task's input must conform to a specific input schema, it will be provided here",
        title="task_input_schema",
    )
    task_output_schema: str | None = Field(
        None,
        description="If the task's output must conform to a specific schema, it will be provided here",
        title="task_output_schema",
    )
    specification: str = Field(
        ..., description="The specification to analyze", title="specification"
    )


class AnswerOption(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )
    answer_title: str = Field(
        ...,
        description="A short title describing this answer option",
        title="answer_title",
    )
    answer_description: str = Field(
        ..., description="A description of this answer", title="answer_description"
    )


class Question(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )
    question_title: str = Field(
        ..., description="A short title for this question", title="question_title"
    )
    question_body: str = Field(
        ..., description="The full question text", title="question_body"
    )
    answer_options: list[AnswerOption] = Field(
        ...,
        description="A list of possible answers to this question for the user to select from",
        title="answer_options",
    )


class QuestionSet(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )
    questions: list[Question] = Field(
        ...,
        description="A set of questions to ask about the specification",
        title="questions",
    )


# Answer submission models
class AnswerOptionWithSelection(AnswerOption):
    """An answer option with user selection state."""

    selected: bool = Field(
        ...,
        description="Whether the user selected this answer option",
        title="selected",
    )


class QuestionWithAnswer(BaseModel):
    """A question with user-provided answer."""

    model_config = ConfigDict(extra="forbid")

    question_title: str = Field(
        ...,
        description="A short title for this question",
        title="question_title",
    )
    question_body: str = Field(
        ...,
        description="The full question text",
        title="question_body",
    )
    answer_options: list[AnswerOptionWithSelection] = Field(
        ...,
        description="Possible answers the user was asked to select from",
        title="answer_options",
    )
    custom_answer: Optional[str] = Field(
        None,
        description="User-provided text feedback when predefined answer options don't fit",
        title="custom_answer",
    )

    @model_validator(mode="after")
    def validate_answer(self) -> "QuestionWithAnswer":
        selected_count = sum(1 for opt in self.answer_options if opt.selected)
        has_custom = self.custom_answer is not None

        if selected_count > 1:
            raise ValueError("Only one answer option can be selected")
        if selected_count > 0 and has_custom:
            raise ValueError("Cannot have both a selected option and custom_answer")
        if selected_count == 0 and not has_custom:
            raise ValueError(
                "Must either select an answer option or provide custom_answer"
            )
        if (
            has_custom
            and self.custom_answer is not None
            and not self.custom_answer.strip()
        ):
            raise ValueError("custom_answer cannot be empty")
        return self


class SpecificationInput(BaseModel):
    """The specification to refine."""

    model_config = ConfigDict(extra="forbid")

    spec_fields: dict[str, str] = Field(
        ...,
        description="Dictionary mapping field names to their descriptions/purposes",
        title="spec_fields",
    )
    spec_field_current_values: dict[str, str] = Field(
        ...,
        description="Dictionary mapping field names to their current values",
        title="spec_field_current_values",
    )


class SubmitAnswersRequest(BaseModel):
    """Request to submit answers to a question set."""

    model_config = ConfigDict(extra="forbid")

    task_prompt: str = Field(
        ...,
        description="The task's prompt",
        title="task_prompt",
    )
    specification: SpecificationInput = Field(
        ...,
        description="The specification to refine",
        title="specification",
    )
    questions_and_answers: list[QuestionWithAnswer] = Field(
        ...,
        description="Questions about the specification with user-provided answers",
        title="questions_and_answers",
    )


class NewProposedSpecEditApi(BaseModel):
    """A proposed edit to a spec field."""

    spec_field_name: str
    proposed_edit: str
    reason_for_edit: str


class RefineSpecApiOutput(BaseModel):
    """Output from refining a spec."""

    new_proposed_spec_edits: list[NewProposedSpecEditApi]
    not_incorporated_feedback: str | None
    # Model-suggested, filename-safe name for the eval, derived from the issue.
    # Absent when the model doesn't provide one.
    suggested_name: str | None = None
