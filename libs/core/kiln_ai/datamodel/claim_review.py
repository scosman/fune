from typing import Literal

from pydantic import BaseModel, Field

from kiln_ai.datamodel.basemodel import KilnParentedModel


class GradedClaim(BaseModel):
    """One claim with a human grade on it.

    A claim is one decision the judge made, written so the reviewer can vote
    on it from the card alone. Grades have one direction on every claim:
    agree means the judge got that decision right, disagree means it got it
    wrong. The claim text carries its own evidence and citation markers.
    """

    text: str = Field(description="The claim as shown to the reviewer.")
    human_grade: Literal["agree", "disagree"] = Field(
        description="The human's grade on this claim."
    )
    human_feedback: str | None = Field(
        default=None,
        description="Optional plaintext reason for the grade.",
    )


class ClaimReview(KilnParentedModel):
    """A human's grades on the claim summary of one task run.

    Persisted alongside the run's rating so consumers (e.g. judge-prompt
    refinement) can use the full review, which claims were agreed or
    disagreed with and why, not just the final pass/fail. Every claim the
    reviewer saw is recorded with its grade, and the overview is kept so a
    stored review reads on its own.
    """

    judge_score: Literal["pass", "fail"] = Field(
        description="The judge's binary verdict on this run."
    )
    judge_reasoning: str = Field(description="The judge's explanation for its verdict.")
    overview: str = Field(
        description="The neutral summary of the run the reviewer read before "
        "grading the claims."
    )
    claims: list[GradedClaim] = Field(
        description="Every graded claim, in the order the reviewer saw them.",
    )
    human_verdict: Literal["pass", "fail"] = Field(
        description="The reviewer's overall call on this run. Derived from "
        "their grade on the verdict claim when the summary carried one, "
        "otherwise asked directly."
    )
