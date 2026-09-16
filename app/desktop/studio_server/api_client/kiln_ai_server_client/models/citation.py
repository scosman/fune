from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.source import Source

T = TypeVar("T", bound="Citation")


@_attrs_define
class Citation:
    """
    Attributes:
        marker (int): The [n] used in the text.
        source (Source):
        from_ (str): Short verbatim snippet marking the start of the span. Located by first occurrence; must be unique
            in its source.
        to (str): Short verbatim snippet marking the end of the span. May equal from.
    """

    marker: int
    source: Source
    from_: str
    to: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        marker = self.marker

        source = self.source.value

        from_ = self.from_

        to = self.to

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "marker": marker,
                "source": source,
                "from": from_,
                "to": to,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        marker = d.pop("marker")

        source = Source(d.pop("source"))

        from_ = d.pop("from")

        to = d.pop("to")

        citation = cls(
            marker=marker,
            source=source,
            from_=from_,
            to=to,
        )

        citation.additional_properties = d
        return citation

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
