from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.human_grade import HumanGrade

T = TypeVar("T", bound="GradedClaim")


@_attrs_define
class GradedClaim:
    """
    Attributes:
        text (str): The claim exactly as shown to the reviewer. Every claim voices one decision the judge made. The text
            may carry a '(possible judge error)' tag, a 'Disagree if …' sentence, an 'Agree only if …' sentence, a trailing
            'Note: …' paragraph, or a closing "We suggest 'Agree' …" sentence; [n] citation markers may appear but the
            underlying trace is not provided.
        human_grade (HumanGrade):
        human_feedback (None | str): The reviewer's optional plaintext 'why' — the richest alignment signal when
            present. Null if left blank.
    """

    text: str
    human_grade: HumanGrade
    human_feedback: None | str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        text = self.text

        human_grade = self.human_grade.value

        human_feedback: None | str
        human_feedback = self.human_feedback

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "text": text,
                "human_grade": human_grade,
                "human_feedback": human_feedback,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        text = d.pop("text")

        human_grade = HumanGrade(d.pop("human_grade"))

        def _parse_human_feedback(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        human_feedback = _parse_human_feedback(d.pop("human_feedback"))

        graded_claim = cls(
            text=text,
            human_grade=human_grade,
            human_feedback=human_feedback,
        )

        graded_claim.additional_properties = d
        return graded_claim

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
