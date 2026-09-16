from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="GenerateJudgePromptOutput")


@_attrs_define
class GenerateJudgePromptOutput:
    """
    Attributes:
        judge_evaluation_prompt (str):
    """

    judge_evaluation_prompt: str

    def to_dict(self) -> dict[str, Any]:
        judge_evaluation_prompt = self.judge_evaluation_prompt

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "judge_evaluation_prompt": judge_evaluation_prompt,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        judge_evaluation_prompt = d.pop("judge_evaluation_prompt")

        generate_judge_prompt_output = cls(
            judge_evaluation_prompt=judge_evaluation_prompt,
        )

        return generate_judge_prompt_output
