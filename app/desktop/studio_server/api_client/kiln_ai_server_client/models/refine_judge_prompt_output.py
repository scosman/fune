from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.change import Change


T = TypeVar("T", bound="RefineJudgePromptOutput")


@_attrs_define
class RefineJudgePromptOutput:
    """
    Attributes:
        refined_judge_prompt (str): The complete revised judge prompt — a self-contained drop-in replacement preserving
            the original's output format and pass/fail contract. If no changes are warranted, the input judge_prompt
            unchanged.
        changes (list[Change]): One entry per distinct edit, ordered most-to-least important. Empty if no changes were
            warranted.
        not_incorporated_feedback (None | str): Actionable human feedback deliberately NOT folded into the judge prompt
            (out of scope for the judge, contradictory across traces, or one-off noise), with a brief reason. Null if
            everything actionable was incorporated.
    """

    refined_judge_prompt: str
    changes: list[Change]
    not_incorporated_feedback: None | str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        refined_judge_prompt = self.refined_judge_prompt

        changes = []
        for changes_item_data in self.changes:
            changes_item = changes_item_data.to_dict()
            changes.append(changes_item)

        not_incorporated_feedback: None | str
        not_incorporated_feedback = self.not_incorporated_feedback

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "refined_judge_prompt": refined_judge_prompt,
                "changes": changes,
                "not_incorporated_feedback": not_incorporated_feedback,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.change import Change

        d = dict(src_dict)
        refined_judge_prompt = d.pop("refined_judge_prompt")

        changes = []
        _changes = d.pop("changes")
        for changes_item_data in _changes:
            changes_item = Change.from_dict(changes_item_data)

            changes.append(changes_item)

        def _parse_not_incorporated_feedback(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        not_incorporated_feedback = _parse_not_incorporated_feedback(d.pop("not_incorporated_feedback"))

        refine_judge_prompt_output = cls(
            refined_judge_prompt=refined_judge_prompt,
            changes=changes,
            not_incorporated_feedback=not_incorporated_feedback,
        )

        refine_judge_prompt_output.additional_properties = d
        return refine_judge_prompt_output

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
