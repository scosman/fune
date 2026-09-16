"""Shared fixtures for studio server tests."""

import pytest
from kiln_ai.adapters.ml_model_list import ModelProviderName
from kiln_ai.datamodel import Project, Task
from kiln_ai.datamodel.datamodel_enums import StructuredOutputMode
from kiln_ai.datamodel.prompt_id import PromptGenerators
from kiln_ai.datamodel.run_config import (
    KilnAgentRunConfigProperties,
    RunConfigProperties,
    ToolsRunConfig,
)
from kiln_ai.datamodel.skill import Skill
from kiln_ai.datamodel.task import TaskRunConfig


@pytest.fixture
def agent_run_config_properties():
    """Factory for KilnAgentRunConfigProperties with throwaway model settings.

    Only the tools are parameterised: tests that care about a task's capability
    surface shouldn't have to restate model, prompt and output settings.
    """

    def _make(
        tools_config: ToolsRunConfig | None = None,
    ) -> KilnAgentRunConfigProperties:
        return KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE,
            structured_output_mode=StructuredOutputMode.json_schema,
            tools_config=tools_config,
        )

    return _make


@pytest.fixture
def save_skill():
    """Save a project skill together with its SKILL.md sidecar."""

    def _save(project: Project, name: str, description: str) -> Skill:
        skill = Skill(name=name, description=description, parent=project)
        skill.save_to_file()
        skill.save_skill_md(f"# {name}")
        return skill

    return _save


@pytest.fixture
def set_default_run_config():
    """Save a run config under a task and make it the task's default."""

    def _set(task: Task, properties: RunConfigProperties) -> TaskRunConfig:
        run_config = TaskRunConfig(
            name="default", run_config_properties=properties, parent=task
        )
        run_config.save_to_file()
        task.default_run_config_id = run_config.id
        task.save_to_file()
        return run_config

    return _set


@pytest.fixture
def give_task_one_tool_and_skill(
    agent_run_config_properties, save_skill, set_default_run_config
):
    """Give a task the standard test capability surface via its default run
    config: the built-in `add` tool plus a `refund-policy` project skill.

    One builder for every test that asserts on a populated capability payload,
    so the expected names and descriptions can't drift between them.
    """

    def _give(project: Project, task: Task) -> Skill:
        skill = save_skill(project, "refund-policy", "How and when refunds are issued.")
        set_default_run_config(
            task,
            agent_run_config_properties(
                tools_config=ToolsRunConfig(
                    tools=["kiln_tool::add_numbers", f"kiln_tool::skill::{skill.id}"]
                )
            ),
        )
        return skill

    return _give
