from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.task_skill_info import TaskSkillInfo
    from ..models.task_tool_info import TaskToolInfo


T = TypeVar("T", bound="TaskInfo")


@_attrs_define
class TaskInfo:
    """Shared information about a task

    Attributes:
        task_prompt (str):
        task_input_schema (str):
        task_output_schema (str):
        task_tools (list[TaskToolInfo] | None | Unset):
        task_skills (list[TaskSkillInfo] | None | Unset):
    """

    task_prompt: str
    task_input_schema: str
    task_output_schema: str
    task_tools: list[TaskToolInfo] | None | Unset = UNSET
    task_skills: list[TaskSkillInfo] | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        task_prompt = self.task_prompt

        task_input_schema = self.task_input_schema

        task_output_schema = self.task_output_schema

        task_tools: list[dict[str, Any]] | None | Unset
        if isinstance(self.task_tools, Unset):
            task_tools = UNSET
        elif isinstance(self.task_tools, list):
            task_tools = []
            for task_tools_type_0_item_data in self.task_tools:
                task_tools_type_0_item = task_tools_type_0_item_data.to_dict()
                task_tools.append(task_tools_type_0_item)

        else:
            task_tools = self.task_tools

        task_skills: list[dict[str, Any]] | None | Unset
        if isinstance(self.task_skills, Unset):
            task_skills = UNSET
        elif isinstance(self.task_skills, list):
            task_skills = []
            for task_skills_type_0_item_data in self.task_skills:
                task_skills_type_0_item = task_skills_type_0_item_data.to_dict()
                task_skills.append(task_skills_type_0_item)

        else:
            task_skills = self.task_skills

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "task_prompt": task_prompt,
                "task_input_schema": task_input_schema,
                "task_output_schema": task_output_schema,
            }
        )
        if task_tools is not UNSET:
            field_dict["task_tools"] = task_tools
        if task_skills is not UNSET:
            field_dict["task_skills"] = task_skills

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.task_skill_info import TaskSkillInfo
        from ..models.task_tool_info import TaskToolInfo

        d = dict(src_dict)
        task_prompt = d.pop("task_prompt")

        task_input_schema = d.pop("task_input_schema")

        task_output_schema = d.pop("task_output_schema")

        def _parse_task_tools(data: object) -> list[TaskToolInfo] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                task_tools_type_0 = []
                _task_tools_type_0 = data
                for task_tools_type_0_item_data in _task_tools_type_0:
                    task_tools_type_0_item = TaskToolInfo.from_dict(task_tools_type_0_item_data)

                    task_tools_type_0.append(task_tools_type_0_item)

                return task_tools_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[TaskToolInfo] | None | Unset, data)

        task_tools = _parse_task_tools(d.pop("task_tools", UNSET))

        def _parse_task_skills(data: object) -> list[TaskSkillInfo] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                task_skills_type_0 = []
                _task_skills_type_0 = data
                for task_skills_type_0_item_data in _task_skills_type_0:
                    task_skills_type_0_item = TaskSkillInfo.from_dict(task_skills_type_0_item_data)

                    task_skills_type_0.append(task_skills_type_0_item)

                return task_skills_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[TaskSkillInfo] | None | Unset, data)

        task_skills = _parse_task_skills(d.pop("task_skills", UNSET))

        task_info = cls(
            task_prompt=task_prompt,
            task_input_schema=task_input_schema,
            task_output_schema=task_output_schema,
            task_tools=task_tools,
            task_skills=task_skills,
        )

        task_info.additional_properties = d
        return task_info

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
