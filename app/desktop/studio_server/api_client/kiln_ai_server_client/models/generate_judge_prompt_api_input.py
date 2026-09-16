from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.generate_judge_prompt_api_input_trace_type import GenerateJudgePromptApiInputTraceType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.task_skill_info import TaskSkillInfo
    from ..models.task_tool_info import TaskToolInfo


T = TypeVar("T", bound="GenerateJudgePromptApiInput")


@_attrs_define
class GenerateJudgePromptApiInput:
    """Request payload for the judge prompt authoring copilot.

    Attributes:
        target_specification (str): The specification describing what behavior the Target Task should exhibit or avoid
        target_task_prompt (str): Complete prompt for the Target Task including system instructions and few-shot
            examples
        trace_type (GenerateJudgePromptApiInputTraceType): Shape of the traces the judge will grade. Selects the
            authoring prompt: multi-turn rubrics reason over turn-labelled transcripts and tool activity, single-turn
            rubrics grade one input/output pair.
        task_tools (list[TaskToolInfo] | None | Unset): Tools available to the Target Task, rendered into the task
            prompt so the authored rubric can reason about tool use. Omit if the caller did not collect them; send [] if the
            task has none.
        task_skills (list[TaskSkillInfo] | None | Unset): Skills available to the Target Task, rendered into the task
            prompt so the authored rubric can reason about skill use. Omit if the caller did not collect them; send [] if
            the task has none.
    """

    target_specification: str
    target_task_prompt: str
    trace_type: GenerateJudgePromptApiInputTraceType
    task_tools: list[TaskToolInfo] | None | Unset = UNSET
    task_skills: list[TaskSkillInfo] | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        target_specification = self.target_specification

        target_task_prompt = self.target_task_prompt

        trace_type = self.trace_type.value

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
                "target_specification": target_specification,
                "target_task_prompt": target_task_prompt,
                "trace_type": trace_type,
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
        target_specification = d.pop("target_specification")

        target_task_prompt = d.pop("target_task_prompt")

        trace_type = GenerateJudgePromptApiInputTraceType(d.pop("trace_type"))

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

        generate_judge_prompt_api_input = cls(
            target_specification=target_specification,
            target_task_prompt=target_task_prompt,
            trace_type=trace_type,
            task_tools=task_tools,
            task_skills=task_skills,
        )

        generate_judge_prompt_api_input.additional_properties = d
        return generate_judge_prompt_api_input

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
