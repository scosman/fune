from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.judge_score import JudgeScore

T = TypeVar("T", bound="BuildClaimEvidenceInput")


@_attrs_define
class BuildClaimEvidenceInput:
    """
    Attributes:
        task_instruction (str): The client task's prompt or description. Says what the task is (a joke generator, a
            support assistant, an extractor). Context only: it never overrides the judge or the rubric.
        raw_input (str): The task's raw input, verbatim. Ground truth. For conversational tasks, the opening user
            message. Cite with source 'input'.
        raw_output (str): The task's raw output, verbatim. Ground truth. For conversational tasks, the full transcript
            as labelled turns. Cite with source 'output'.
        eval_rubric (str): The prompt the judge ran with. A hint about what matters; may be under-specified or wrong.
        judge_reasoning (str): The judge's explanation. May be rich, thin or a placeholder. Never shown to the reviewer;
            never describe it in the output.
        judge_score (JudgeScore):
    """

    task_instruction: str
    raw_input: str
    raw_output: str
    eval_rubric: str
    judge_reasoning: str
    judge_score: JudgeScore
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        task_instruction = self.task_instruction

        raw_input = self.raw_input

        raw_output = self.raw_output

        eval_rubric = self.eval_rubric

        judge_reasoning = self.judge_reasoning

        judge_score = self.judge_score.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "task_instruction": task_instruction,
                "raw_input": raw_input,
                "raw_output": raw_output,
                "eval_rubric": eval_rubric,
                "judge_reasoning": judge_reasoning,
                "judge_score": judge_score,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        task_instruction = d.pop("task_instruction")

        raw_input = d.pop("raw_input")

        raw_output = d.pop("raw_output")

        eval_rubric = d.pop("eval_rubric")

        judge_reasoning = d.pop("judge_reasoning")

        judge_score = JudgeScore(d.pop("judge_score"))

        build_claim_evidence_input = cls(
            task_instruction=task_instruction,
            raw_input=raw_input,
            raw_output=raw_output,
            eval_rubric=eval_rubric,
            judge_reasoning=judge_reasoning,
            judge_score=judge_score,
        )

        build_claim_evidence_input.additional_properties = d
        return build_claim_evidence_input

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
