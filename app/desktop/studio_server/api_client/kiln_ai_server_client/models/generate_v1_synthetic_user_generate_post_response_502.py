from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.generate_v1_synthetic_user_generate_post_response_502_code import (
    GenerateV1SyntheticUserGeneratePostResponse502Code,
)

T = TypeVar("T", bound="GenerateV1SyntheticUserGeneratePostResponse502")


@_attrs_define
class GenerateV1SyntheticUserGeneratePostResponse502:
    """
    Attributes:
        message (str):
        code (GenerateV1SyntheticUserGeneratePostResponse502Code):
    """

    message: str
    code: GenerateV1SyntheticUserGeneratePostResponse502Code
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        message = self.message

        code = self.code.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "message": message,
                "code": code,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        message = d.pop("message")

        code = GenerateV1SyntheticUserGeneratePostResponse502Code(d.pop("code"))

        generate_v1_synthetic_user_generate_post_response_502 = cls(
            message=message,
            code=code,
        )

        generate_v1_synthetic_user_generate_post_response_502.additional_properties = d
        return generate_v1_synthetic_user_generate_post_response_502

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
