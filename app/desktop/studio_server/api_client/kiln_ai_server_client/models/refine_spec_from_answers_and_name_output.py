from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.new_proposed_spec_edit import NewProposedSpecEdit


T = TypeVar("T", bound="RefineSpecFromAnswersAndNameOutput")


@_attrs_define
class RefineSpecFromAnswersAndNameOutput:
    """
    Attributes:
        new_proposed_spec_edits (list[NewProposedSpecEdit]):
        suggested_name (str): short, human-readable name for the eval in Title Case with spaces, at most 32 characters,
            derived from the issue description
    """

    new_proposed_spec_edits: list[NewProposedSpecEdit]
    suggested_name: str

    def to_dict(self) -> dict[str, Any]:
        new_proposed_spec_edits = []
        for new_proposed_spec_edits_item_data in self.new_proposed_spec_edits:
            new_proposed_spec_edits_item = new_proposed_spec_edits_item_data.to_dict()
            new_proposed_spec_edits.append(new_proposed_spec_edits_item)

        suggested_name = self.suggested_name

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "new_proposed_spec_edits": new_proposed_spec_edits,
                "suggested_name": suggested_name,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.new_proposed_spec_edit import NewProposedSpecEdit

        d = dict(src_dict)
        new_proposed_spec_edits = []
        _new_proposed_spec_edits = d.pop("new_proposed_spec_edits")
        for new_proposed_spec_edits_item_data in _new_proposed_spec_edits:
            new_proposed_spec_edits_item = NewProposedSpecEdit.from_dict(new_proposed_spec_edits_item_data)

            new_proposed_spec_edits.append(new_proposed_spec_edits_item)

        suggested_name = d.pop("suggested_name")

        refine_spec_from_answers_and_name_output = cls(
            new_proposed_spec_edits=new_proposed_spec_edits,
            suggested_name=suggested_name,
        )

        return refine_spec_from_answers_and_name_output
