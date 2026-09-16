from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="NewProposedSpecEdit")


@_attrs_define
class NewProposedSpecEdit:
    """
    Attributes:
        spec_field_name (str): The name of the spec field that is being edited
        proposed_edit (str): A new value for this spec field incorporating the feedback
        reason_for_edit (str): The reason for editing this spec field
    """

    spec_field_name: str
    proposed_edit: str
    reason_for_edit: str

    def to_dict(self) -> dict[str, Any]:
        spec_field_name = self.spec_field_name

        proposed_edit = self.proposed_edit

        reason_for_edit = self.reason_for_edit

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "spec_field_name": spec_field_name,
                "proposed_edit": proposed_edit,
                "reason_for_edit": reason_for_edit,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        spec_field_name = d.pop("spec_field_name")

        proposed_edit = d.pop("proposed_edit")

        reason_for_edit = d.pop("reason_for_edit")

        new_proposed_spec_edit = cls(
            spec_field_name=spec_field_name,
            proposed_edit=proposed_edit,
            reason_for_edit=reason_for_edit,
        )

        return new_proposed_spec_edit
