from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="GenerateSyntheticUsersRequest")


@_attrs_define
class GenerateSyntheticUsersRequest:
    """Request body for POST /v1/synthetic_user/generate.

    Generates `num_cases` synthetic-user cases designed to probe
    `target_specification` against the agent described by `target_task_prompt`,
    across multi-turn conversations.

        Attributes:
            target_task_prompt (str): Complete prompt of the target task (the AI assistant under evaluation). Used as
                material for designing realistic synthetic users; never executed by this endpoint.
            target_specification (str): Behavior or issue to investigate about the target task (e.g. 'hallucinates tax-year-
                specific rules for years before 2018'). Generated cases are designed so a multi-turn conversation will naturally
                surface this behavior.
            num_cases (int): Number of synthetic-user cases to generate (1-50).
            case_scenarios (list[str] | None | Unset): Optional per-case scenario briefs (e.g. from an approved batch plan).
                When provided, length must equal `num_cases` and the whole batch is generated in one pass with case i designed
                around scenario i. Each surviving case in the response carries `scenario_index` so a salvaged (shorter) batch
                stays mappable to its scenarios.
    """

    target_task_prompt: str
    target_specification: str
    num_cases: int
    case_scenarios: list[str] | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        target_task_prompt = self.target_task_prompt

        target_specification = self.target_specification

        num_cases = self.num_cases

        case_scenarios: list[str] | None | Unset
        if isinstance(self.case_scenarios, Unset):
            case_scenarios = UNSET
        elif isinstance(self.case_scenarios, list):
            case_scenarios = self.case_scenarios

        else:
            case_scenarios = self.case_scenarios

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "target_task_prompt": target_task_prompt,
                "target_specification": target_specification,
                "num_cases": num_cases,
            }
        )
        if case_scenarios is not UNSET:
            field_dict["case_scenarios"] = case_scenarios

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        target_task_prompt = d.pop("target_task_prompt")

        target_specification = d.pop("target_specification")

        num_cases = d.pop("num_cases")

        def _parse_case_scenarios(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                case_scenarios_type_0 = cast(list[str], data)

                return case_scenarios_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        case_scenarios = _parse_case_scenarios(d.pop("case_scenarios", UNSET))

        generate_synthetic_users_request = cls(
            target_task_prompt=target_task_prompt,
            target_specification=target_specification,
            num_cases=num_cases,
            case_scenarios=case_scenarios,
        )

        generate_synthetic_users_request.additional_properties = d
        return generate_synthetic_users_request

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
