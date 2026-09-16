from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.graded_trace import GradedTrace


T = TypeVar("T", bound="RefineJudgePromptInput")


@_attrs_define
class RefineJudgePromptInput:
    """
    Attributes:
        judge_prompt (str): The current judge prompt / rubric being refined, verbatim. Your output is a revised version
            of this.
        graded_traces (list[GradedTrace]): One entry per human-reviewed trace: the judge's verdict, the claim builder's
            review card, the reviewer's grade on every claim, and the reviewer's overall verdict.
    """

    judge_prompt: str
    graded_traces: list[GradedTrace]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        judge_prompt = self.judge_prompt

        graded_traces = []
        for graded_traces_item_data in self.graded_traces:
            graded_traces_item = graded_traces_item_data.to_dict()
            graded_traces.append(graded_traces_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "judge_prompt": judge_prompt,
                "graded_traces": graded_traces,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.graded_trace import GradedTrace

        d = dict(src_dict)
        judge_prompt = d.pop("judge_prompt")

        graded_traces = []
        _graded_traces = d.pop("graded_traces")
        for graded_traces_item_data in _graded_traces:
            graded_traces_item = GradedTrace.from_dict(graded_traces_item_data)

            graded_traces.append(graded_traces_item)

        refine_judge_prompt_input = cls(
            judge_prompt=judge_prompt,
            graded_traces=graded_traces,
        )

        refine_judge_prompt_input.additional_properties = d
        return refine_judge_prompt_input

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
