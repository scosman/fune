"""Shared Pydantic models for the Copilot API."""

from typing import Annotated, Literal

from kiln_ai.datamodel.claim_review import GradedClaim
from kiln_ai.datamodel.datamodel_enums import ModelProviderName
from pydantic import BaseModel, Field, StringConstraints, model_validator
from typing_extensions import Self


# Base models
class TaskToolInfoApi(BaseModel):
    """A tool the target task can call. Name and description only."""

    name: str = Field(description="The tool's name, as the model sees it.")
    description: str = Field(
        description="What the tool does. Never its parameter schema."
    )


class TaskSkillInfoApi(BaseModel):
    """A skill the target task can load. Name and description only."""

    name: str = Field(description="The skill's name, as the model sees it.")
    description: str = Field(description="What the skill does. Never the skill's body.")


class TaskInfoApi(BaseModel):
    """Task information for copilot API calls."""

    task_prompt: str = Field(description="The task's prompt.")
    task_input_schema: str = Field(description="The task's input JSON schema.")
    task_output_schema: str = Field(description="The task's output JSON schema.")
    # None and [] are different answers and must never be conflated: None means
    # the capabilities were not collected, so the copilot prompts render nothing
    # and stay exactly as they were before these fields existed; [] means the
    # task genuinely has none, which is worth telling the model explicitly.
    task_tools: list[TaskToolInfoApi] | None = Field(
        default=None,
        description="Tools available to the task. Omit if not collected; "
        "send [] if the task has none.",
    )
    task_skills: list[TaskSkillInfoApi] | None = Field(
        default=None,
        description="Skills available to the task. Omit if not collected; "
        "send [] if the task has none.",
    )


class TaskMetadataApi(BaseModel):
    """Metadata about the model used for a task."""

    model_name: str = Field(description="The name of the AI model used.")
    model_provider_name: ModelProviderName = Field(
        description="The provider hosting the model (e.g. OpenAI, Anthropic)."
    )


class SyntheticDataGenerationStepConfigApi(BaseModel):
    """Configuration for a synthetic data generation step."""

    task_metadata: TaskMetadataApi
    prompt: str


class SyntheticDataGenerationSessionConfigApi(BaseModel):
    """Configuration for a synthetic data generation session"""

    topic_generation_config: SyntheticDataGenerationStepConfigApi
    input_generation_config: SyntheticDataGenerationStepConfigApi
    output_generation_config: SyntheticDataGenerationStepConfigApi


class SampleApi(BaseModel):
    """A sample input/output pair."""

    input: str = Field(alias="input")
    output: str

    model_config = {"populate_by_name": True}


class ClaimReviewApi(BaseModel):
    """The reviewer's grades on one trace's claim summary.

    Mirrors the persisted ClaimReview shape (judge verdict, the overview,
    every claim with its agree/disagree and optional why, and the reviewer's
    overall call) so the save path can write it onto the golden TaskRun and
    judge refinement can consume it later.
    """

    judge_score: Literal["pass", "fail"]
    judge_reasoning: str
    overview: str
    claims: list[GradedClaim]
    human_verdict: Literal["pass", "fail"]


def verdicts_must_agree(
    user_says_meets_spec: bool, claim_review: ClaimReviewApi | None
) -> None:
    """The golden rating and the stored review record the same overall call.

    Both come from one derivation in the UI, so a mismatch is a corrupt
    payload; reject it up front rather than write an answer key that
    contradicts the review saved beside it.
    """
    if claim_review is None:
        return
    if (claim_review.human_verdict == "pass") != user_says_meets_spec:
        raise ValueError(
            "user_says_meets_spec must match claim_review.human_verdict "
            f"(got {user_says_meets_spec!r} vs {claim_review.human_verdict!r})"
        )


class ReviewedExample(BaseModel):
    """A reviewed example from the spec review process.

    Extends SampleApi with review-specific fields for tracking
    model and user judgments on spec compliance.
    """

    input: str = Field(alias="input")
    output: str
    model_says_meets_spec: bool
    user_says_meets_spec: bool
    feedback: str
    claim_review: ClaimReviewApi | None = Field(
        default=None,
        description="Per-claim grades from the claim review, when the example "
        "was reviewed that way (v2 builder).",
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def validate_verdicts_agree(self) -> Self:
        verdicts_must_agree(self.user_says_meets_spec, self.claim_review)
        return self


class ReviewedChainApi(BaseModel):
    """A reviewer's verdict on one multi-turn chain, keyed by its leaf run.

    The leaf TaskRun id is the durable identity that rides from the drive
    batch through review to save — the save path writes the golden rating
    (and the claim review) onto that leaf.
    """

    leaf_run_id: str
    user_says_meets_spec: bool
    feedback: str = ""
    claim_review: ClaimReviewApi | None = None

    @model_validator(mode="after")
    def validate_verdicts_agree(self) -> Self:
        verdicts_must_agree(self.user_says_meets_spec, self.claim_review)
        return self


class DrivenSyntheticCaseApi(BaseModel):
    """One driven synthetic-user case from the builder session.

    The save path mints an EvalInput from each — the re-drivable input the
    eval runner regenerates a conversation from, per run config.
    """

    seed_prompt: str = Field(
        min_length=1,
        description="The opening user-side message of the conversation.",
    )
    synthetic_user_info: str = Field(
        min_length=1,
        description="The XML-tagged persona blob as generated "
        "(persona/goal/behavior_guidance). Wire format only: the save path "
        "parses it into the structured submodel before anything persists.",
    )
    scenario_index: int | None = Field(
        default=None,
        description="Zero-based index into the builder's user-approved "
        "scenario plan identifying the scenario this case was generated "
        "from. Recorded on the minted EvalInput as a `scenario:{index}` "
        "provenance tag; omit when the case has no plan scenario.",
    )


# Input models
class SpecApi(BaseModel):
    """Spec field information for refinement."""

    spec_fields: dict[str, str]
    spec_field_current_values: dict[str, str]


class ExampleWithFeedbackApi(BaseModel):
    """An example with user feedback for spec refinement."""

    model_config = {"populate_by_name": True}

    user_agrees_with_judge: bool
    input: str = Field(alias="input")
    output: str
    fails_specification: bool
    user_feedback: str | None = None


class TaskScopedCopilotInput(BaseModel):
    """Base for copilot inputs the studio server can enrich from local storage.

    The ids let the studio server load the task and fill in target_task_info's
    capability fields before forwarding. They are studio-local identifiers and
    are always stripped from the outgoing payload. All are optional: a caller
    that omits them gets the plain passthrough it always got.
    """

    project_id: str | None = Field(
        default=None,
        description="The project holding the target task. Pair with task_id to "
        "have the server attach the task's tools and skills.",
    )
    task_id: str | None = Field(
        default=None,
        description="The target task. Pair with project_id to have the server "
        "attach the task's tools and skills.",
    )
    run_config_id: str | None = Field(
        default=None,
        description="The task run config whose tools and skills to attach — "
        "the one this request is about, such as the run config an eval is "
        "being written against. Omit to use the task's default run config.",
    )

    @model_validator(mode="after")
    def validate_ids_provided_together(self) -> Self:
        # Half a pair can only be a caller bug. Silently skipping enrichment
        # would ship a prompt quietly missing the task's capabilities, which
        # is far harder to notice than a rejected request.
        if (self.project_id is None) != (self.task_id is None):
            raise ValueError(
                "project_id and task_id must be provided together, or both omitted"
            )
        # Same reasoning one level down: a run config is only read once the
        # task is, so an id sent without the task would be dropped, and the
        # prompt would describe the wrong config's capabilities.
        if self.run_config_id is not None and self.task_id is None:
            raise ValueError("run_config_id requires project_id and task_id")
        return self


class ClarifySpecApiInput(TaskScopedCopilotInput):
    """Input for clarifying a spec with copilot."""

    target_task_info: TaskInfoApi
    target_specification: str
    num_samples_per_topic: int
    num_topics: int
    providers: list[ModelProviderName]
    num_exemplars: int = Field(default=10)


class RefineSpecApiInput(TaskScopedCopilotInput):
    """Input for refining a spec based on feedback."""

    target_task_info: TaskInfoApi
    target_specification: SpecApi
    examples_with_feedback: list[ExampleWithFeedbackApi]


class GenerateBatchApiInput(BaseModel):
    """Input for generating a batch of examples."""

    target_task_info: TaskInfoApi
    target_specification: str
    num_samples_per_topic: int
    num_topics: int
    sdg_session_config: SyntheticDataGenerationSessionConfigApi


# Output models
class SubsampleBatchOutputItemApi(BaseModel):
    """A single item from batch output for feedback."""

    input: str = Field(alias="input")
    output: str
    fails_specification: bool

    model_config = {"populate_by_name": True}


class ClarifySpecApiOutput(BaseModel):
    """Output from clarifying a spec."""

    examples_for_feedback: list[SubsampleBatchOutputItemApi]
    judge_result: SyntheticDataGenerationStepConfigApi
    sdg_session_config: SyntheticDataGenerationSessionConfigApi


class GenerateBatchApiOutput(BaseModel):
    """Output from generating a batch of examples."""

    data_by_topic: dict[str, list[SampleApi]]


class SpecQuestionerApiInput(TaskScopedCopilotInput):
    target_task_info: TaskInfoApi = Field(
        ...,
        description="The task info including prompt, input schema, and output schema",
        title="target_task_info",
    )
    target_specification: str = Field(
        ...,
        description="The specification to analyze",
        title="target_specification",
    )


# Input Data Guide draft job
#
# The draft runs as a kiln_server background job so the heavy
# summarize+aggregate work survives a flaky connection and the user can leave
# the page and come back. The studio server exposes the job's start / status /
# result lifecycle so the web UI owns polling. Preview inputs are no longer
# bundled here — once the draft is ready the UI generates them via the existing
# `/data_gen_guide_preview` endpoint.

DRAFT_INPUT_DATA_GUIDE_MAX_EXAMPLES = 200
# Per-example character ceiling. Each example becomes one summarize LLM call;
# 200k chars stays well under the model's context window even for prose. The
# client mirrors this and blocks before sending, so hitting it here is a guard.
DRAFT_INPUT_DATA_GUIDE_MAX_EXAMPLE_LENGTH = 200_000


class StartDataGuideJobApiInput(BaseModel):
    """Input to kick off the input data guide draft job.

    Carries only the input examples. All task info the job needs — the runtime
    prompt and the input JSON schema — is derived server-side from the task
    identified by the route, so the client can't supply a manipulated prompt or
    schema, and the output schema / description never reach the guide LLM.
    """

    input_examples: list[
        Annotated[
            str,
            StringConstraints(max_length=DRAFT_INPUT_DATA_GUIDE_MAX_EXAMPLE_LENGTH),
        ]
    ] = Field(
        ...,
        description=(
            "Heterogeneous list of input examples — short manual entries, the "
            "input portion of selected task runs, or full text of uploaded "
            "text documents (txt, md, csv). Every entry is a string and is "
            "treated as a candidate reference input regardless of source."
        ),
        min_length=1,
        max_length=DRAFT_INPUT_DATA_GUIDE_MAX_EXAMPLES,
    )


class StartDataGuideJobApiOutput(BaseModel):
    """Identifier for the started data guide draft job."""

    job_id: str = Field(description="Identifier for the started data guide draft job.")


class DataGuideJobStatusApiOutput(BaseModel):
    """Current status of a data guide draft job."""

    status: str = Field(
        description=(
            "Current job status (e.g. running, succeeded, failed, cancelled)."
        ),
    )


class DataGuideJobResultApiOutput(BaseModel):
    """Result of a completed data guide draft job."""

    draft_guide: str = Field(description="Full draft input data guide markdown.")


class ParseImportFileApiOutput(BaseModel):
    """Result of parsing an uploaded bulk-import file of input examples.

    Plaintext tasks parse a single-column CSV; structured-input tasks parse one
    JSON object per line, validated against the task's input schema. A non-null
    `error` means the whole file was rejected; `warning` means it was accepted
    but some examples were skipped (e.g. over the length limit).
    """

    rows: list[str] = Field(
        description="Parsed example input strings, ready to add. Empty when error is set.",
    )
    error: str | None = Field(
        default=None,
        description="Set when the whole file was rejected (invalid format/encoding).",
    )
    warning: str | None = Field(
        default=None,
        description="Set when the file was accepted but some examples were skipped.",
    )
