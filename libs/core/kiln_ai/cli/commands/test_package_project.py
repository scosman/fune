import json
import shutil
import zipfile
from pathlib import Path
from typing import TypedDict
from unittest.mock import patch

import pytest
import typer

from kiln_ai.datamodel import Project, Task
from kiln_ai.datamodel.code_tool import TOOL_CODE_FILENAME, CodeTool
from kiln_ai.datamodel.datamodel_enums import (
    ModelProviderName,
    StructuredOutputMode,
    TaskOutputRatingType,
)
from kiln_ai.datamodel.eval import (
    SCORER_CODE_FILENAME,
    CodeEvalProperties,
    Eval,
    EvalConfig,
    EvalConfigType,
    EvalOutputScore,
    EvalRun,
)
from kiln_ai.datamodel.external_tool_server import ExternalToolServer, ToolServerType
from kiln_ai.datamodel.prompt import Prompt
from kiln_ai.datamodel.prompt_id import PromptGenerators
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties, ToolsRunConfig
from kiln_ai.datamodel.skill import Skill
from kiln_ai.datamodel.task import TaskRunConfig
from kiln_ai.datamodel.task_output import TaskOutput
from kiln_ai.datamodel.task_run import TaskRun
from kiln_ai.datamodel.tool_id import (
    KILN_TASK_TOOL_ID_PREFIX,
    MCP_LOCAL_TOOL_ID_PREFIX,
    MCP_REMOTE_TOOL_ID_PREFIX,
    RAG_TOOL_ID_PREFIX,
    SKILL_TOOL_ID_PREFIX,
    KilnBuiltInToolId,
)

from .package_project import (
    PackageForTrainingConfig,
    build_prompt_from_builder,
    classify_tool_id,
    collect_required_skills,
    collect_required_tool_servers,
    collect_subtask_ids_from_tools,
    create_export_directory,
    create_zip,
    export_documents,
    export_eval_inputs,
    export_evals,
    export_skills,
    export_task,
    export_task_runs,
    export_tool_servers,
    get_default_run_config,
    get_tools_from_run_config,
    is_dynamic_prompt,
    load_project,
    package_project,
    package_project_for_training,
    parse_task_ids,
    print_tasks_table,
    save_prompt_to_task,
    validate_and_build_prompts,
    validate_and_build_prompts_noncli,
    validate_exported_prompts,
    validate_tasks,
    validate_tasks_noncli,
    validate_tools,
)


@pytest.fixture
def temp_project(tmp_path: Path):
    """Create a temporary project with tasks and run configs for testing."""
    project = Project(name="Test Project", path=tmp_path / "project.kiln")
    project.save_to_file()

    task = Task(
        name="Test Task",
        instruction="Test instruction for the task",
        parent=project,
    )
    task.save_to_file()

    run_config = TaskRunConfig(
        name="Test Run Config",
        parent=task,
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name="openai",
            prompt_id=PromptGenerators.SIMPLE.value,
            structured_output_mode=StructuredOutputMode.default,
        ),
    )
    run_config.save_to_file()

    task.default_run_config_id = run_config.id
    task.save_to_file()

    return {
        "project": project,
        "task": task,
        "run_config": run_config,
        "path": tmp_path,
    }


@pytest.fixture
def temp_project_with_multiple_tasks(tmp_path: Path):
    """Create a project with multiple tasks for testing."""
    project = Project(name="Multi Task Project", path=tmp_path / "project.kiln")
    project.save_to_file()

    tasks = []
    run_configs = []
    for i in range(3):
        task = Task(
            name=f"Task {i + 1}",
            instruction=f"Instruction for task {i + 1}",
            parent=project,
        )
        task.save_to_file()

        run_config = TaskRunConfig(
            name=f"Run Config {i + 1}",
            parent=task,
            run_config_properties=KilnAgentRunConfigProperties(
                model_name="gpt-4o",
                model_provider_name="openai",
                prompt_id=PromptGenerators.SIMPLE.value,
                structured_output_mode=StructuredOutputMode.default,
            ),
        )
        run_config.save_to_file()

        task.default_run_config_id = run_config.id
        task.save_to_file()

        tasks.append(task)
        run_configs.append(run_config)

    return {
        "project": project,
        "tasks": tasks,
        "run_configs": run_configs,
        "path": tmp_path,
    }


@pytest.fixture
def temp_project_no_run_config(tmp_path: Path):
    """Create a project with a task that has no default run config."""
    project = Project(name="No Config Project", path=tmp_path / "project.kiln")
    project.save_to_file()

    task = Task(
        name="Task Without Config",
        instruction="Test instruction",
        parent=project,
    )
    task.save_to_file()

    return {"project": project, "task": task, "path": tmp_path}


@pytest.fixture
def temp_project_with_tools(tmp_path: Path):
    """Create a project with a run config that uses tools."""
    project = Project(name="Tools Project", path=tmp_path / "project.kiln")
    project.save_to_file()

    task = Task(
        name="Task With Tools",
        instruction="Test instruction",
        parent=project,
    )
    task.save_to_file()

    run_config = TaskRunConfig(
        name="Config With Tools",
        parent=task,
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name="openai",
            prompt_id=PromptGenerators.SIMPLE.value,
            structured_output_mode=StructuredOutputMode.default,
            tools_config=ToolsRunConfig(tools=["kiln_tool::add_numbers"]),
        ),
    )
    run_config.save_to_file()

    task.default_run_config_id = run_config.id
    task.save_to_file()

    return {
        "project": project,
        "task": task,
        "run_config": run_config,
        "path": tmp_path,
    }


@pytest.fixture
def temp_project_with_dynamic_prompt(tmp_path: Path):
    """Create a project with a dynamic prompt generator."""
    project = Project(name="Dynamic Prompt Project", path=tmp_path / "project.kiln")
    project.save_to_file()

    task = Task(
        name="Task With Dynamic Prompt",
        instruction="Test instruction",
        parent=project,
    )
    task.save_to_file()

    run_config = TaskRunConfig(
        name="Config With Dynamic Prompt",
        parent=task,
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name="openai",
            prompt_id=PromptGenerators.FEW_SHOT.value,
            structured_output_mode=StructuredOutputMode.default,
        ),
    )
    run_config.save_to_file()

    task.default_run_config_id = run_config.id
    task.save_to_file()

    return {
        "project": project,
        "task": task,
        "run_config": run_config,
        "path": tmp_path,
    }


@pytest.fixture
def temp_project_with_kiln_task_tool(tmp_path: Path):
    """Create a project with a task that uses a kiln_task tool pointing to another task."""
    project = Project(name="Kiln Task Tool Project", path=tmp_path / "project.kiln")
    project.save_to_file()

    # Create the sub-task (the task that will be called as a tool)
    subtask = Task(
        name="Sub Task",
        instruction="This is a sub-task instruction",
        parent=project,
    )
    subtask.save_to_file()

    subtask_run_config = TaskRunConfig(
        name="Sub Task Run Config",
        parent=subtask,
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE.value,
            structured_output_mode=StructuredOutputMode.default,
        ),
    )
    subtask_run_config.save_to_file()

    subtask.default_run_config_id = subtask_run_config.id
    subtask.save_to_file()

    # Create external tool server for the kiln_task tool
    # Note: We must save subtask first before using subtask.id and subtask_run_config.id
    assert subtask.id is not None
    assert subtask_run_config.id is not None
    tool_server = ExternalToolServer(
        name="subtask_tool_server",
        type=ToolServerType.kiln_task,
        description="Tool server for subtask",
        properties={
            "name": "subtask_tool",
            "description": "Calls the subtask",
            "task_id": str(subtask.id),
            "run_config_id": str(subtask_run_config.id),
            "is_archived": False,
        },
        parent=project,
    )
    tool_server.save_to_file()

    # Create the main task that uses the kiln_task tool
    main_task = Task(
        name="Main Task",
        instruction="Main task instruction",
        parent=project,
    )
    main_task.save_to_file()

    main_run_config = TaskRunConfig(
        name="Main Run Config",
        parent=main_task,
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE.value,
            structured_output_mode=StructuredOutputMode.default,
            tools_config=ToolsRunConfig(
                tools=[f"{KILN_TASK_TOOL_ID_PREFIX}{tool_server.id}"]
            ),
        ),
    )
    main_run_config.save_to_file()

    main_task.default_run_config_id = main_run_config.id
    main_task.save_to_file()

    return {
        "project": project,
        "main_task": main_task,
        "main_run_config": main_run_config,
        "subtask": subtask,
        "subtask_run_config": subtask_run_config,
        "tool_server": tool_server,
        "path": tmp_path,
    }


@pytest.fixture
def temp_project_with_mcp_local_tool(tmp_path: Path):
    """Create a project with a task that uses a local MCP tool."""
    project = Project(name="MCP Local Project", path=tmp_path / "project.kiln")
    project.save_to_file()

    # Create external tool server for the local MCP tool
    tool_server = ExternalToolServer(
        name="local_mcp_server",
        type=ToolServerType.local_mcp,
        description="Local MCP server",
        properties={
            "command": "uvx",
            "args": ["some-mcp-server"],
            "is_archived": False,
        },
        parent=project,
    )
    tool_server.save_to_file()

    task = Task(
        name="Task With MCP Local",
        instruction="Test instruction",
        parent=project,
    )
    task.save_to_file()

    run_config = TaskRunConfig(
        name="Config With MCP Local",
        parent=task,
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE.value,
            structured_output_mode=StructuredOutputMode.default,
            tools_config=ToolsRunConfig(
                tools=[f"{MCP_LOCAL_TOOL_ID_PREFIX}{tool_server.id}::some_tool"]
            ),
        ),
    )
    run_config.save_to_file()

    task.default_run_config_id = run_config.id
    task.save_to_file()

    return {
        "project": project,
        "task": task,
        "run_config": run_config,
        "tool_server": tool_server,
        "path": tmp_path,
    }


@pytest.fixture
def temp_project_with_mcp_remote_tool(tmp_path: Path):
    """Create a project with a task that uses a remote MCP tool."""
    project = Project(name="MCP Remote Project", path=tmp_path / "project.kiln")
    project.save_to_file()

    # Create external tool server for the remote MCP tool
    tool_server = ExternalToolServer(
        name="remote_mcp_server",
        type=ToolServerType.remote_mcp,
        description="Remote MCP server",
        properties={
            "server_url": "https://example.com/mcp",
            "is_archived": False,
        },
        parent=project,
    )
    tool_server.save_to_file()

    task = Task(
        name="Task With MCP Remote",
        instruction="Test instruction",
        parent=project,
    )
    task.save_to_file()

    run_config = TaskRunConfig(
        name="Config With MCP Remote",
        parent=task,
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE.value,
            structured_output_mode=StructuredOutputMode.default,
            tools_config=ToolsRunConfig(
                tools=[f"{MCP_REMOTE_TOOL_ID_PREFIX}{tool_server.id}::some_tool"]
            ),
        ),
    )
    run_config.save_to_file()

    task.default_run_config_id = run_config.id
    task.save_to_file()

    return {
        "project": project,
        "task": task,
        "run_config": run_config,
        "tool_server": tool_server,
        "path": tmp_path,
    }


@pytest.fixture
def temp_project_with_rag_tool(tmp_path: Path):
    """Create a project with a task that uses a RAG tool."""
    project = Project(name="RAG Project", path=tmp_path / "project.kiln")
    project.save_to_file()

    task = Task(
        name="Task With RAG",
        instruction="Test instruction",
        parent=project,
    )
    task.save_to_file()

    run_config = TaskRunConfig(
        name="Config With RAG",
        parent=task,
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE.value,
            structured_output_mode=StructuredOutputMode.default,
            tools_config=ToolsRunConfig(tools=[f"{RAG_TOOL_ID_PREFIX}some_rag_config"]),
        ),
    )
    run_config.save_to_file()

    task.default_run_config_id = run_config.id
    task.save_to_file()

    return {
        "project": project,
        "task": task,
        "run_config": run_config,
        "path": tmp_path,
    }


@pytest.fixture
def temp_project_with_builtin_tool(tmp_path: Path):
    """Create a project with a task that uses a built-in tool."""
    project = Project(name="Builtin Tool Project", path=tmp_path / "project.kiln")
    project.save_to_file()

    task = Task(
        name="Task With Builtin",
        instruction="Test instruction",
        parent=project,
    )
    task.save_to_file()

    run_config = TaskRunConfig(
        name="Config With Builtin",
        parent=task,
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE.value,
            structured_output_mode=StructuredOutputMode.default,
            tools_config=ToolsRunConfig(tools=[KilnBuiltInToolId.ADD_NUMBERS.value]),
        ),
    )
    run_config.save_to_file()

    task.default_run_config_id = run_config.id
    task.save_to_file()

    return {
        "project": project,
        "task": task,
        "run_config": run_config,
        "path": tmp_path,
    }


class SkillToolProjectFixture(TypedDict):
    project: Project
    task: Task
    run_config: TaskRunConfig
    skill: Skill
    path: Path


@pytest.fixture
def temp_project_with_skill_tool(tmp_path: Path) -> SkillToolProjectFixture:
    """Create a project with a task that uses a skill tool."""
    project = Project(name="Skill Project", path=tmp_path / "project.kiln")
    project.save_to_file()

    skill = Skill(
        name="test-skill",
        description="A test skill for export testing",
        parent=project,
    )
    skill.save_to_file()
    skill.save_skill_md("This is the skill body with instructions.")

    task = Task(
        name="Task With Skill",
        instruction="Test instruction",
        parent=project,
    )
    task.save_to_file()

    run_config = TaskRunConfig(
        name="Config With Skill",
        parent=task,
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name=ModelProviderName.openai,
            prompt_id=PromptGenerators.SIMPLE.value,
            structured_output_mode=StructuredOutputMode.default,
            tools_config=ToolsRunConfig(tools=[f"{SKILL_TOOL_ID_PREFIX}{skill.id}"]),
        ),
    )
    run_config.save_to_file()

    task.default_run_config_id = run_config.id
    task.save_to_file()

    return {
        "project": project,
        "task": task,
        "run_config": run_config,
        "skill": skill,
        "path": tmp_path,
    }


class TestLoadProject:
    def test_load_project_from_file(self, temp_project):
        """Test loading a project from a direct file path."""
        project_file = temp_project["path"] / "project.kiln"
        loaded = load_project(project_file)
        assert loaded.name == "Test Project"

    def test_load_project_from_folder(self, temp_project):
        """Test loading a project from a folder containing project.kiln."""
        loaded = load_project(temp_project["path"])
        assert loaded.name == "Test Project"

    def test_load_project_missing_file(self, tmp_path: Path):
        """Test error when project file doesn't exist."""
        with pytest.raises(typer.Exit) as exc_info:
            load_project(tmp_path / "nonexistent.kiln")
        assert exc_info.value.exit_code == 1

    def test_load_project_missing_folder(self, tmp_path: Path):
        """Test error when folder has no project.kiln."""
        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()
        with pytest.raises(typer.Exit) as exc_info:
            load_project(empty_folder)
        assert exc_info.value.exit_code == 1


class TestParseTaskIds:
    def test_parse_all_tasks_flag(self, temp_project_with_multiple_tasks):
        """Test parsing with --all-tasks flag."""
        tasks = temp_project_with_multiple_tasks["tasks"]
        result = parse_task_ids("", True, tasks)
        assert len(result) == 3

    def test_parse_all_string(self, temp_project_with_multiple_tasks):
        """Test parsing with 'all' string."""
        tasks = temp_project_with_multiple_tasks["tasks"]
        result = parse_task_ids("all", False, tasks)
        assert len(result) == 3

    def test_parse_comma_separated(self, temp_project_with_multiple_tasks):
        """Test parsing comma-separated task IDs."""
        tasks = temp_project_with_multiple_tasks["tasks"]
        task_ids = f"{tasks[0].id},{tasks[1].id}"
        result = parse_task_ids(task_ids, False, tasks)
        assert len(result) == 2
        assert tasks[0].id in result
        assert tasks[1].id in result

    def test_parse_single_task(self, temp_project_with_multiple_tasks):
        """Test parsing a single task ID."""
        tasks = temp_project_with_multiple_tasks["tasks"]
        result = parse_task_ids(tasks[0].id, False, tasks)
        assert result == [tasks[0].id]

    def test_parse_no_tasks_error(self, temp_project_with_multiple_tasks):
        """Test error when no tasks specified."""
        tasks = temp_project_with_multiple_tasks["tasks"]
        with pytest.raises(typer.Exit) as exc_info:
            parse_task_ids("", False, tasks)
        assert exc_info.value.exit_code == 1


class TestValidateTasks:
    def test_validate_existing_tasks(self, temp_project_with_multiple_tasks):
        """Test validation of existing task IDs."""
        project = temp_project_with_multiple_tasks["project"]
        tasks = temp_project_with_multiple_tasks["tasks"]
        task_ids = [tasks[0].id, tasks[1].id]

        result = validate_tasks(task_ids, project)
        assert len(result) == 2

    def test_validate_missing_task(self, temp_project_with_multiple_tasks):
        """Test error when task ID doesn't exist."""
        project = temp_project_with_multiple_tasks["project"]
        with pytest.raises(typer.Exit) as exc_info:
            validate_tasks(["nonexistent_id"], project)
        assert exc_info.value.exit_code == 1


class TestValidateTasksNoncli:
    def test_validate_existing_tasks(self, temp_project_with_multiple_tasks):
        project = temp_project_with_multiple_tasks["project"]
        tasks = temp_project_with_multiple_tasks["tasks"]
        task_ids = [tasks[0].id, tasks[1].id]

        result = validate_tasks_noncli(task_ids, project)
        assert len(result) == 2

    def test_validate_missing_task_raises_value_error(
        self, temp_project_with_multiple_tasks
    ):
        project = temp_project_with_multiple_tasks["project"]
        with pytest.raises(ValueError, match="Task ID\\(s\\) not found"):
            validate_tasks_noncli(["nonexistent_id"], project)

    def test_validate_partial_missing_raises_value_error(
        self, temp_project_with_multiple_tasks
    ):
        project = temp_project_with_multiple_tasks["project"]
        tasks = temp_project_with_multiple_tasks["tasks"]
        with pytest.raises(ValueError, match="nonexistent"):
            validate_tasks_noncli([tasks[0].id, "nonexistent"], project)


class TestGetDefaultRunConfig:
    def test_get_valid_run_config(self, temp_project):
        """Test getting a valid default run config."""
        task = temp_project["task"]
        task_reloaded = Task.load_from_file(task.path)
        result = get_default_run_config(task_reloaded)
        assert result.name == "Test Run Config"

    def test_get_run_config_not_set(self, temp_project_no_run_config):
        """Test error when no default run config is set."""
        task = temp_project_no_run_config["task"]
        with pytest.raises(typer.Exit) as exc_info:
            get_default_run_config(task)
        assert exc_info.value.exit_code == 1

    def test_get_run_config_id_invalid(self, temp_project):
        """Test error when default run config ID doesn't exist."""
        task = temp_project["task"]
        task_reloaded = Task.load_from_file(task.path)
        task_reloaded.default_run_config_id = "nonexistent_id"

        with pytest.raises(typer.Exit) as exc_info:
            get_default_run_config(task_reloaded)
        assert exc_info.value.exit_code == 1


class TestIsDynamicPrompt:
    @pytest.mark.parametrize(
        "prompt_id,expected",
        [
            (PromptGenerators.SIMPLE.value, False),
            (PromptGenerators.SIMPLE_CHAIN_OF_THOUGHT.value, False),
            (PromptGenerators.FEW_SHOT.value, True),
            (PromptGenerators.MULTI_SHOT.value, True),
            (PromptGenerators.REPAIRS.value, True),
            (PromptGenerators.FEW_SHOT_CHAIN_OF_THOUGHT.value, True),
            (PromptGenerators.MULTI_SHOT_CHAIN_OF_THOUGHT.value, True),
            ("id::some_saved_prompt", False),
        ],
    )
    def test_is_dynamic_prompt(self, prompt_id, expected):
        """Test dynamic prompt detection."""
        assert is_dynamic_prompt(prompt_id) == expected


class TestValidateAndBuildPrompts:
    def test_build_static_prompts(self, temp_project):
        """Test building prompts with static generators."""
        task = Task.load_from_file(temp_project["task"].path)
        run_config = TaskRunConfig.load_from_file(temp_project["run_config"].path)

        result = validate_and_build_prompts([task], {task.id: run_config})

        assert task.id in result
        assert isinstance(result[task.id], Prompt)
        assert "Test instruction" in result[task.id].prompt

    def test_build_dynamic_prompts_confirm_yes(self, temp_project_with_dynamic_prompt):
        """Test building prompts with dynamic generators when user confirms."""
        task = Task.load_from_file(temp_project_with_dynamic_prompt["task"].path)
        run_config = TaskRunConfig.load_from_file(
            temp_project_with_dynamic_prompt["run_config"].path
        )

        with patch(
            "kiln_ai.cli.commands.package_project.typer.confirm", return_value=True
        ):
            result = validate_and_build_prompts([task], {task.id: run_config})

        assert task.id in result
        assert isinstance(result[task.id], Prompt)

    def test_build_dynamic_prompts_confirm_no(self, temp_project_with_dynamic_prompt):
        """Test that user can decline dynamic prompt export."""
        task = Task.load_from_file(temp_project_with_dynamic_prompt["task"].path)
        run_config = TaskRunConfig.load_from_file(
            temp_project_with_dynamic_prompt["run_config"].path
        )

        with patch(
            "kiln_ai.cli.commands.package_project.typer.confirm", return_value=False
        ):
            with pytest.raises(typer.Exit) as exc_info:
                validate_and_build_prompts([task], {task.id: run_config})
            assert exc_info.value.exit_code == 0


class TestValidateAndBuildPromptsNoncli:
    def test_build_static_prompts(self, temp_project):
        task = Task.load_from_file(temp_project["task"].path)
        run_config = TaskRunConfig.load_from_file(temp_project["run_config"].path)

        result = validate_and_build_prompts_noncli([task], {task.id: run_config})

        assert task.id in result
        assert isinstance(result[task.id], Prompt)
        assert "Test instruction" in result[task.id].prompt

    def test_build_dynamic_prompts_without_confirmation(
        self, temp_project_with_dynamic_prompt
    ):
        """Dynamic prompts are accepted silently without typer.confirm."""
        task = Task.load_from_file(temp_project_with_dynamic_prompt["task"].path)
        run_config = TaskRunConfig.load_from_file(
            temp_project_with_dynamic_prompt["run_config"].path
        )

        result = validate_and_build_prompts_noncli([task], {task.id: run_config})

        assert task.id in result
        assert isinstance(result[task.id], Prompt)

    def test_raises_value_error_on_build_failure(self, temp_project):
        task = Task.load_from_file(temp_project["task"].path)
        run_config = TaskRunConfig.load_from_file(temp_project["run_config"].path)

        with patch(
            "kiln_ai.cli.commands.package_project.prompt_builder_from_id",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(ValueError, match="Failed to build prompt"):
                validate_and_build_prompts_noncli([task], {task.id: run_config})


class TestPackageProjectCommand:
    def test_full_validation_flow(self, temp_project, tmp_path: Path):
        """Test the complete validation flow."""
        result = package_project(
            project_path=temp_project["path"],
            tasks=temp_project["task"].id,
            all_tasks=False,
            output=tmp_path / "test_output.zip",
        )

        assert len(result) == 1
        task_id = temp_project["task"].id
        assert task_id in result
        assert isinstance(result[task_id], Prompt)

    def test_all_tasks_flow(self, temp_project_with_multiple_tasks, tmp_path: Path):
        """Test validation with all tasks."""
        result = package_project(
            project_path=temp_project_with_multiple_tasks["path"],
            tasks="",
            all_tasks=True,
            output=tmp_path / "test_output.zip",
        )

        assert len(result) == 3

    def test_missing_project_path_error(self, tmp_path: Path):
        """Test error when no project path is provided."""
        with pytest.raises(typer.Exit) as exc_info:
            package_project(
                project_path=None,
                tasks="123",
                all_tasks=False,
                output=tmp_path / "test_output.zip",
            )
        assert exc_info.value.exit_code == 1

    def test_missing_project_error(self, tmp_path: Path):
        """Test error handling for missing project."""
        with pytest.raises(typer.Exit) as exc_info:
            package_project(
                project_path=tmp_path / "nonexistent",
                tasks="123",
                all_tasks=False,
                output=tmp_path / "test_output.zip",
            )
        assert exc_info.value.exit_code == 1

    def test_no_tasks_error(self, temp_project, tmp_path: Path):
        """Test error when no tasks are specified."""
        with pytest.raises(typer.Exit) as exc_info:
            package_project(
                project_path=temp_project["path"],
                tasks="",
                all_tasks=False,
                output=tmp_path / "test_output.zip",
            )
        assert exc_info.value.exit_code == 1


class TestPrintTasksTable:
    def test_prints_without_error(self, temp_project_with_multiple_tasks, capsys):
        """Test that print_tasks_table runs without errors."""
        tasks = temp_project_with_multiple_tasks["tasks"]
        print_tasks_table(tasks)


class TestBuildPromptFromBuilder:
    def test_builds_prompt_model(self, temp_project):
        """Test building a Prompt model from a builder."""
        from kiln_ai.adapters.prompt_builders import SimplePromptBuilder

        task = Task.load_from_file(temp_project["task"].path)
        builder = SimplePromptBuilder(task)

        result = build_prompt_from_builder(builder)

        assert isinstance(result, Prompt)
        assert result.name == "Exported Prompt"
        assert "Test instruction" in result.prompt


class TestCreateExportDirectory:
    def test_creates_directory_with_project(self, temp_project):
        """Test that export directory is created with project.kiln copied."""
        project = temp_project["project"]

        temp_dir, exported_project = create_export_directory(project)

        try:
            assert temp_dir.exists()
            assert (temp_dir / "project.kiln").exists()
            assert exported_project.name == project.name
            assert exported_project.path == temp_dir / "project.kiln"
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_raises_error_if_project_path_not_set(self):
        """Test that error is raised if project path is not set."""
        project = Project(name="No Path Project")

        with pytest.raises(ValueError, match="Project path is not set"):
            create_export_directory(project)


class TestExportTask:
    def test_exports_task_with_run_config(self, temp_project):
        """Test that task and run config are exported correctly."""
        project = temp_project["project"]
        task = Task.load_from_file(temp_project["task"].path)
        run_config = TaskRunConfig.load_from_file(temp_project["run_config"].path)

        temp_dir, exported_project = create_export_directory(project)

        try:
            exported_task, exported_run_config = export_task(
                task, run_config, exported_project
            )

            assert exported_task.name == task.name
            assert exported_task.instruction == task.instruction
            assert exported_task.id == task.id
            assert exported_task.path is not None
            assert exported_task.path.exists()

            assert exported_run_config.name == run_config.name
            assert exported_run_config.id == run_config.id
            assert exported_run_config.path is not None
            assert exported_run_config.path.exists()

            task_folder = exported_task.path.parent
            assert task_folder.name == temp_project["task"].path.parent.name
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_preserves_task_folder_name(self, temp_project):
        """Test that the original task folder name is preserved."""
        project = temp_project["project"]
        task = Task.load_from_file(temp_project["task"].path)
        run_config = TaskRunConfig.load_from_file(temp_project["run_config"].path)
        assert task.path is not None
        original_folder_name = task.path.parent.name

        temp_dir, exported_project = create_export_directory(project)

        try:
            exported_task, _ = export_task(task, run_config, exported_project)

            assert exported_task.path is not None
            assert exported_task.path.parent.name == original_folder_name
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSavePromptToTask:
    def test_saves_prompt_and_updates_run_config(self, temp_project):
        """Test that prompt is saved and run config is updated."""
        project = temp_project["project"]
        task = Task.load_from_file(temp_project["task"].path)
        run_config = TaskRunConfig.load_from_file(temp_project["run_config"].path)

        temp_dir, exported_project = create_export_directory(project)

        try:
            exported_task, exported_run_config = export_task(
                task, run_config, exported_project
            )

            prompt = Prompt(
                name="Test Prompt",
                prompt="Test prompt content",
                chain_of_thought_instructions="Think step by step",
            )

            saved_prompt = save_prompt_to_task(
                prompt, exported_task, exported_run_config
            )

            assert saved_prompt.path is not None
            assert saved_prompt.path.exists()
            assert saved_prompt.prompt == "Test prompt content"

            assert exported_run_config.path is not None
            reloaded_run_config = TaskRunConfig.load_from_file(exported_run_config.path)
            assert (
                reloaded_run_config.run_config_properties.prompt_id
                == f"id::{saved_prompt.id}"
            )

            assert exported_task.path is not None
            exported_task_reloaded = Task.load_from_file(exported_task.path)
            prompts = exported_task_reloaded.prompts()
            assert len(prompts) == 1
            assert prompts[0].id == saved_prompt.id
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


class TestValidateExportedPrompts:
    def test_validates_matching_prompts(self, temp_project):
        """Test that validation passes when prompts match."""
        project = temp_project["project"]
        task = Task.load_from_file(temp_project["task"].path)
        run_config = TaskRunConfig.load_from_file(temp_project["run_config"].path)

        temp_dir, exported_project = create_export_directory(project)

        try:
            exported_task, exported_run_config = export_task(
                task, run_config, exported_project
            )

            original_prompt = Prompt(
                name="Test Prompt",
                prompt="Test prompt content",
                chain_of_thought_instructions=None,
            )

            save_prompt_to_task(original_prompt, exported_task, exported_run_config)

            assert task.id is not None
            validate_exported_prompts(
                {task.id: original_prompt},
                {task.id: exported_task},
                {task.id: exported_run_config},
            )
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_raises_error_on_prompt_mismatch(self, temp_project):
        """Test that validation fails when prompts don't match."""
        project = temp_project["project"]
        task = Task.load_from_file(temp_project["task"].path)
        run_config = TaskRunConfig.load_from_file(temp_project["run_config"].path)

        temp_dir, exported_project = create_export_directory(project)

        try:
            exported_task, exported_run_config = export_task(
                task, run_config, exported_project
            )

            original_prompt = Prompt(
                name="Test Prompt",
                prompt="Original prompt content",
                chain_of_thought_instructions=None,
            )

            different_prompt = Prompt(
                name="Test Prompt",
                prompt="Different prompt content",
                chain_of_thought_instructions=None,
                parent=exported_task,
            )
            different_prompt.save_to_file()

            exported_run_config.run_config_properties.prompt_id = (
                f"id::{different_prompt.id}"
            )
            exported_run_config.save_to_file()

            assert task.id is not None
            with pytest.raises(ValueError, match="Prompt mismatch"):
                validate_exported_prompts(
                    {task.id: original_prompt},
                    {task.id: exported_task},
                    {task.id: exported_run_config},
                )
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


class TestCreateZip:
    def test_creates_valid_zip_file(self, tmp_path: Path):
        """Test that a valid zip file is created."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "file1.txt").write_text("content1")
        (source_dir / "subdir").mkdir()
        (source_dir / "subdir" / "file2.txt").write_text("content2")

        output_path = tmp_path / "output.zip"

        create_zip(source_dir, output_path)

        assert output_path.exists()

        with zipfile.ZipFile(output_path, "r") as zipf:
            names = zipf.namelist()
            assert "file1.txt" in names
            assert "subdir/file2.txt" in names

            assert zipf.read("file1.txt").decode() == "content1"
            assert zipf.read("subdir/file2.txt").decode() == "content2"

    def test_creates_parent_directories(self, tmp_path: Path):
        """Test that parent directories are created for output path."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("content")

        output_path = tmp_path / "nested" / "dirs" / "output.zip"

        create_zip(source_dir, output_path)

        assert output_path.exists()


class TestPackageProjectEndToEnd:
    def test_creates_zip_with_correct_structure(self, temp_project, tmp_path: Path):
        """Test full end-to-end export creates correct zip structure."""
        output_path = tmp_path / "export" / "output.zip"

        package_project(
            project_path=temp_project["path"],
            tasks=temp_project["task"].id,
            all_tasks=False,
            output=output_path,
        )

        assert output_path.exists()

        with zipfile.ZipFile(output_path, "r") as zipf:
            names = zipf.namelist()

            assert "project.kiln" in names

            task_files = [
                n for n in names if n.startswith("tasks/") and n.endswith("task.kiln")
            ]
            assert len(task_files) == 1

            run_config_files = [
                n for n in names if "run_configs" in n and n.endswith(".kiln")
            ]
            assert len(run_config_files) == 1

            prompt_files = [n for n in names if "prompts" in n and n.endswith(".kiln")]
            assert len(prompt_files) == 1

    def test_exported_project_is_loadable(self, temp_project, tmp_path: Path):
        """Test that the exported project can be loaded and used."""
        output_path = tmp_path / "output.zip"
        extract_path = tmp_path / "extracted"

        package_project(
            project_path=temp_project["path"],
            tasks=temp_project["task"].id,
            all_tasks=False,
            output=output_path,
        )

        with zipfile.ZipFile(output_path, "r") as zipf:
            zipf.extractall(extract_path)

        loaded_project = Project.load_from_file(extract_path / "project.kiln")
        assert loaded_project.name == temp_project["project"].name

        tasks = loaded_project.tasks()
        assert len(tasks) == 1
        assert tasks[0].id == temp_project["task"].id

        prompts = tasks[0].prompts()
        assert len(prompts) == 1

        run_configs = tasks[0].run_configs()
        assert len(run_configs) == 1
        assert run_configs[0].run_config_properties.prompt_id == f"id::{prompts[0].id}"

    def test_exports_multiple_tasks(
        self, temp_project_with_multiple_tasks, tmp_path: Path
    ):
        """Test exporting multiple tasks."""
        output_path = tmp_path / "output.zip"

        package_project(
            project_path=temp_project_with_multiple_tasks["path"],
            tasks="",
            all_tasks=True,
            output=output_path,
        )

        assert output_path.exists()

        extract_path = tmp_path / "extracted"
        with zipfile.ZipFile(output_path, "r") as zipf:
            zipf.extractall(extract_path)

        loaded_project = Project.load_from_file(extract_path / "project.kiln")
        tasks = loaded_project.tasks()
        assert len(tasks) == 3

        for task in tasks:
            prompts = task.prompts()
            assert len(prompts) == 1
            run_configs = task.run_configs()
            assert len(run_configs) == 1


class TestGetToolsFromRunConfig:
    def test_returns_tools_list(self, temp_project_with_tools):
        """Test that tools are returned from run config."""
        run_config = temp_project_with_tools["run_config"]
        tools = get_tools_from_run_config(run_config)
        assert tools == [KilnBuiltInToolId.ADD_NUMBERS.value]

    def test_returns_empty_for_no_tools(self, temp_project):
        """Test that empty list is returned when no tools configured."""
        run_config = temp_project["run_config"]
        tools = get_tools_from_run_config(run_config)
        assert tools == []


class TestClassifyToolId:
    @pytest.mark.parametrize(
        "tool_id,expected",
        [
            (KilnBuiltInToolId.ADD_NUMBERS.value, "builtin"),
            (KilnBuiltInToolId.SUBTRACT_NUMBERS.value, "builtin"),
            (KilnBuiltInToolId.CALL_KILN_API.value, "builtin"),
            (f"{KILN_TASK_TOOL_ID_PREFIX}some_server_id", "kiln_task"),
            (f"{MCP_REMOTE_TOOL_ID_PREFIX}server_id::tool_name", "mcp_remote"),
            (f"{MCP_LOCAL_TOOL_ID_PREFIX}server_id::tool_name", "mcp_local"),
            (f"{RAG_TOOL_ID_PREFIX}some_rag_config", "rag"),
            (f"{SKILL_TOOL_ID_PREFIX}some_skill_id", "skill"),
            ("unknown_prefix::something", "unknown"),
        ],
    )
    def test_classifies_tool_ids(self, tool_id, expected):
        """Test tool ID classification."""
        assert classify_tool_id(tool_id) == expected


class TestCollectSubtaskIds:
    def test_collects_subtask_from_kiln_task_tool(
        self, temp_project_with_kiln_task_tool
    ):
        """Test that subtask IDs are collected from kiln_task tools."""
        project = temp_project_with_kiln_task_tool["project"]
        main_task = temp_project_with_kiln_task_tool["main_task"]
        subtask = temp_project_with_kiln_task_tool["subtask"]

        additional_ids, names = collect_subtask_ids_from_tools([main_task.id], project)

        assert subtask.id in additional_ids
        assert subtask.name in names

    def test_no_subtasks_for_builtin_tools(self, temp_project_with_builtin_tool):
        """Test that built-in tools don't add subtasks."""
        project = temp_project_with_builtin_tool["project"]
        task = temp_project_with_builtin_tool["task"]

        additional_ids, names = collect_subtask_ids_from_tools([task.id], project)

        assert len(additional_ids) == 0
        assert len(names) == 0

    def test_no_subtasks_for_mcp_tools(self, temp_project_with_mcp_local_tool):
        """Test that MCP tools don't add subtasks."""
        project = temp_project_with_mcp_local_tool["project"]
        task = temp_project_with_mcp_local_tool["task"]

        additional_ids, names = collect_subtask_ids_from_tools([task.id], project)

        assert len(additional_ids) == 0
        assert len(names) == 0

    def test_skips_already_included_tasks(self, temp_project_with_kiln_task_tool):
        """Test that subtasks already in the list are not added again."""
        project = temp_project_with_kiln_task_tool["project"]
        main_task = temp_project_with_kiln_task_tool["main_task"]
        subtask = temp_project_with_kiln_task_tool["subtask"]

        # Include subtask already in the list
        additional_ids, _ = collect_subtask_ids_from_tools(
            [main_task.id, subtask.id], project
        )

        assert len(additional_ids) == 0


class TestValidateTools:
    def test_passes_with_builtin_tools(self, temp_project_with_builtin_tool):
        """Test that built-in tools pass validation."""
        task = Task.load_from_file(temp_project_with_builtin_tool["task"].path)
        run_config = TaskRunConfig.load_from_file(
            temp_project_with_builtin_tool["run_config"].path
        )

        assert task.id is not None
        # Should not raise
        validate_tools([task], {task.id: run_config})

    def test_passes_with_mcp_remote_tools(self, temp_project_with_mcp_remote_tool):
        """Test that remote MCP tools pass validation."""
        task = Task.load_from_file(temp_project_with_mcp_remote_tool["task"].path)
        run_config = TaskRunConfig.load_from_file(
            temp_project_with_mcp_remote_tool["run_config"].path
        )

        assert task.id is not None
        # Should not raise (with confirmation mocked)
        with patch(
            "kiln_ai.cli.commands.package_project.typer.confirm", return_value=True
        ):
            validate_tools([task], {task.id: run_config})

    def test_prompts_for_mcp_local_tools(self, temp_project_with_mcp_local_tool):
        """Test that local MCP tools prompt for confirmation."""
        task = Task.load_from_file(temp_project_with_mcp_local_tool["task"].path)
        run_config = TaskRunConfig.load_from_file(
            temp_project_with_mcp_local_tool["run_config"].path
        )

        assert task.id is not None
        with patch(
            "kiln_ai.cli.commands.package_project.typer.confirm", return_value=True
        ):
            validate_tools([task], {task.id: run_config})

    def test_exits_on_mcp_local_decline(self, temp_project_with_mcp_local_tool):
        """Test that declining MCP local prompt exits."""
        task = Task.load_from_file(temp_project_with_mcp_local_tool["task"].path)
        run_config = TaskRunConfig.load_from_file(
            temp_project_with_mcp_local_tool["run_config"].path
        )

        assert task.id is not None
        with patch(
            "kiln_ai.cli.commands.package_project.typer.confirm", return_value=False
        ):
            with pytest.raises(typer.Exit) as exc_info:
                validate_tools([task], {task.id: run_config})
            assert exc_info.value.exit_code == 0

    def test_errors_on_rag_tools(self, temp_project_with_rag_tool):
        """Test that RAG tools cause an error."""
        task = Task.load_from_file(temp_project_with_rag_tool["task"].path)
        run_config = TaskRunConfig.load_from_file(
            temp_project_with_rag_tool["run_config"].path
        )

        assert task.id is not None
        with pytest.raises(typer.Exit) as exc_info:
            validate_tools([task], {task.id: run_config})
        assert exc_info.value.exit_code == 1

    def test_passes_with_skill_tools(
        self, temp_project_with_skill_tool: SkillToolProjectFixture
    ) -> None:
        """Test that skill tools pass validation."""
        task = Task.load_from_file(temp_project_with_skill_tool["task"].path)
        run_config = TaskRunConfig.load_from_file(
            temp_project_with_skill_tool["run_config"].path
        )

        assert task.id is not None
        validate_tools([task], {task.id: run_config})

    def test_errors_on_unknown_tool_type(self, temp_project):
        """Test that unknown tool types cause an error.

        Note: Pydantic validates tool IDs at model creation time, so we need to
        bypass that validation to test our classify_tool_id function handles
        unknown types gracefully.
        """
        # The classify_tool_id function should return 'unknown' for unrecognized prefixes
        # This is tested directly rather than through validate_tools since Pydantic
        # validates tool IDs before they reach our code
        assert classify_tool_id("unknown_prefix::some_tool") == "unknown"

        # If an unknown tool ID somehow got past validation, validate_tools would error
        # We can test this by mocking get_tools_from_run_config
        task = Task.load_from_file(temp_project["task"].path)
        run_config = TaskRunConfig.load_from_file(temp_project["run_config"].path)

        assert task.id is not None
        with patch(
            "kiln_ai.cli.commands.package_project.get_tools_from_run_config",
            return_value=["unknown_prefix::some_tool"],
        ):
            with pytest.raises(typer.Exit) as exc_info:
                validate_tools([task], {task.id: run_config})
            assert exc_info.value.exit_code == 1


class TestCollectRequiredToolServers:
    def test_collects_kiln_task_server_id(self, temp_project_with_kiln_task_tool):
        """Test that kiln_task tool server IDs are collected."""
        main_task = Task.load_from_file(
            temp_project_with_kiln_task_tool["main_task"].path
        )
        run_config = TaskRunConfig.load_from_file(
            temp_project_with_kiln_task_tool["main_run_config"].path
        )
        tool_server = temp_project_with_kiln_task_tool["tool_server"]

        assert main_task.id is not None
        server_ids = collect_required_tool_servers(
            [main_task], {main_task.id: run_config}
        )

        assert tool_server.id in server_ids

    def test_collects_mcp_local_server_id(self, temp_project_with_mcp_local_tool):
        """Test that local MCP tool server IDs are collected."""
        task = Task.load_from_file(temp_project_with_mcp_local_tool["task"].path)
        run_config = TaskRunConfig.load_from_file(
            temp_project_with_mcp_local_tool["run_config"].path
        )
        tool_server = temp_project_with_mcp_local_tool["tool_server"]

        assert task.id is not None
        server_ids = collect_required_tool_servers([task], {task.id: run_config})

        assert tool_server.id in server_ids

    def test_collects_mcp_remote_server_id(self, temp_project_with_mcp_remote_tool):
        """Test that remote MCP tool server IDs are collected."""
        task = Task.load_from_file(temp_project_with_mcp_remote_tool["task"].path)
        run_config = TaskRunConfig.load_from_file(
            temp_project_with_mcp_remote_tool["run_config"].path
        )
        tool_server = temp_project_with_mcp_remote_tool["tool_server"]

        assert task.id is not None
        server_ids = collect_required_tool_servers([task], {task.id: run_config})

        assert tool_server.id in server_ids

    def test_no_server_for_builtin_tools(self, temp_project_with_builtin_tool):
        """Test that built-in tools don't add server IDs."""
        task = Task.load_from_file(temp_project_with_builtin_tool["task"].path)
        run_config = TaskRunConfig.load_from_file(
            temp_project_with_builtin_tool["run_config"].path
        )

        assert task.id is not None
        server_ids = collect_required_tool_servers([task], {task.id: run_config})

        assert len(server_ids) == 0


class TestCollectRequiredSkills:
    def test_collects_skill_id(
        self, temp_project_with_skill_tool: SkillToolProjectFixture
    ) -> None:
        """Test that skill IDs are collected from run configs."""
        task = Task.load_from_file(temp_project_with_skill_tool["task"].path)
        run_config = TaskRunConfig.load_from_file(
            temp_project_with_skill_tool["run_config"].path
        )
        skill = temp_project_with_skill_tool["skill"]

        assert task.id is not None
        skill_ids = collect_required_skills([task], {task.id: run_config})

        assert skill.id in skill_ids

    def test_no_skills_for_builtin_tools(self, temp_project_with_builtin_tool):
        """Test that built-in tools don't add skill IDs."""
        task = Task.load_from_file(temp_project_with_builtin_tool["task"].path)
        run_config = TaskRunConfig.load_from_file(
            temp_project_with_builtin_tool["run_config"].path
        )

        assert task.id is not None
        skill_ids = collect_required_skills([task], {task.id: run_config})

        assert len(skill_ids) == 0

    def test_skips_task_without_run_config(
        self, temp_project_with_skill_tool: SkillToolProjectFixture
    ) -> None:
        """Test that tasks without a matching run config are skipped."""
        task = Task.load_from_file(temp_project_with_skill_tool["task"].path)

        assert task.id is not None
        skill_ids = collect_required_skills([task], {})

        assert len(skill_ids) == 0


class TestExportSkills:
    def test_exports_skill(
        self, temp_project_with_skill_tool: SkillToolProjectFixture
    ) -> None:
        """Test that skills are exported with all files."""
        import shutil

        project = temp_project_with_skill_tool["project"]
        skill = temp_project_with_skill_tool["skill"]

        temp_dir, exported_project = create_export_directory(project)

        try:
            assert skill.id is not None
            export_skills({skill.id}, project, exported_project)

            exported_skills = exported_project.skills()
            assert len(exported_skills) == 1
            assert exported_skills[0].id == skill.id
            assert exported_skills[0].name == skill.name
            assert (
                exported_skills[0].body() == "This is the skill body with instructions."
            )
            assert exported_skills[0].references_dir().exists()
            assert exported_skills[0].assets_dir().exists()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_exports_no_skills_when_empty(self, temp_project):
        """Test that no skills are exported when set is empty."""
        import shutil

        project = temp_project["project"]
        temp_dir, exported_project = create_export_directory(project)

        try:
            export_skills(set(), project, exported_project)
            assert len(exported_project.skills()) == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_raises_when_exported_project_path_is_none(
        self, temp_project_with_skill_tool: SkillToolProjectFixture
    ) -> None:
        """Test that export_skills raises when exported project path is None."""
        project = temp_project_with_skill_tool["project"]
        skill = temp_project_with_skill_tool["skill"]
        exported_project = Project(name="No Path Project", path=None)

        assert skill.id is not None
        with pytest.raises(ValueError, match="Exported project path is not set"):
            export_skills({skill.id}, project, exported_project)

    def test_raises_when_skill_id_not_found(
        self, temp_project_with_skill_tool: SkillToolProjectFixture
    ) -> None:
        """Test that export_skills raises when a skill ID is not in the project."""
        import shutil

        project = temp_project_with_skill_tool["project"]
        temp_dir, exported_project = create_export_directory(project)

        try:
            with pytest.raises(ValueError, match=r"Skill ID.*not found in the project"):
                export_skills({"nonexistent_id"}, project, exported_project)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_raises_when_skill_path_is_none(
        self, temp_project_with_skill_tool: SkillToolProjectFixture
    ) -> None:
        """Test that export_skills raises when a skill's path is None."""
        import shutil
        from unittest.mock import MagicMock

        project = temp_project_with_skill_tool["project"]
        skill = temp_project_with_skill_tool["skill"]
        temp_dir, exported_project = create_export_directory(project)

        try:
            assert skill.id is not None
            fake_skill = MagicMock()
            fake_skill.id = skill.id
            fake_skill.name = skill.name
            fake_skill.path = None

            with patch.object(Project, "skills", return_value=[fake_skill]):
                with pytest.raises(ValueError, match="path is not set"):
                    export_skills({skill.id}, project, exported_project)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestExportToolServers:
    def test_exports_tool_server(self, temp_project_with_mcp_remote_tool):
        """Test that tool servers are exported correctly."""
        import shutil

        project = temp_project_with_mcp_remote_tool["project"]
        tool_server = temp_project_with_mcp_remote_tool["tool_server"]

        temp_dir, exported_project = create_export_directory(project)

        try:
            assert tool_server.id is not None
            export_tool_servers({tool_server.id}, project, exported_project)

            # Check the server was exported
            exported_servers = exported_project.external_tool_servers()
            assert len(exported_servers) == 1
            assert exported_servers[0].id == tool_server.id
            assert exported_servers[0].name == tool_server.name
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_exports_no_servers_when_empty(self, temp_project):
        """Test that no servers are exported when set is empty."""
        import shutil

        project = temp_project["project"]

        temp_dir, exported_project = create_export_directory(project)

        try:
            export_tool_servers(set(), project, exported_project)

            # Check no servers were exported
            exported_servers = exported_project.external_tool_servers()
            assert len(exported_servers) == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_exports_multiple_servers(self, tmp_path: Path):
        """Test exporting multiple tool servers."""
        import shutil

        project = Project(name="Multi Server Project", path=tmp_path / "project.kiln")
        project.save_to_file()

        server1 = ExternalToolServer(
            name="server1",
            type=ToolServerType.remote_mcp,
            properties={"server_url": "https://example1.com", "is_archived": False},
            parent=project,
        )
        server1.save_to_file()

        server2 = ExternalToolServer(
            name="server2",
            type=ToolServerType.remote_mcp,
            properties={"server_url": "https://example2.com", "is_archived": False},
            parent=project,
        )
        server2.save_to_file()

        temp_dir, exported_project = create_export_directory(project)

        try:
            assert server1.id is not None
            assert server2.id is not None
            export_tool_servers({server1.id, server2.id}, project, exported_project)

            exported_servers = exported_project.external_tool_servers()
            assert len(exported_servers) == 2
            exported_ids = {s.id for s in exported_servers}
            assert server1.id in exported_ids
            assert server2.id in exported_ids
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestPackageProjectWithTools:
    def test_exports_with_builtin_tools(
        self, temp_project_with_builtin_tool, tmp_path: Path
    ):
        """Test full export with built-in tools."""
        output_path = tmp_path / "output.zip"

        package_project(
            project_path=temp_project_with_builtin_tool["path"],
            tasks=temp_project_with_builtin_tool["task"].id,
            all_tasks=False,
            output=output_path,
        )

        assert output_path.exists()

    def test_exports_with_mcp_remote_tools(
        self, temp_project_with_mcp_remote_tool, tmp_path: Path
    ):
        """Test full export with remote MCP tools."""
        output_path = tmp_path / "output.zip"
        extract_path = tmp_path / "extracted"

        with patch(
            "kiln_ai.cli.commands.package_project.typer.confirm", return_value=True
        ):
            package_project(
                project_path=temp_project_with_mcp_remote_tool["path"],
                tasks=temp_project_with_mcp_remote_tool["task"].id,
                all_tasks=False,
                output=output_path,
            )

        assert output_path.exists()

        # Verify tool server was exported
        with zipfile.ZipFile(output_path, "r") as zipf:
            zipf.extractall(extract_path)

        loaded_project = Project.load_from_file(extract_path / "project.kiln")
        servers = loaded_project.external_tool_servers()
        assert len(servers) == 1
        assert servers[0].id == temp_project_with_mcp_remote_tool["tool_server"].id

    def test_exports_with_kiln_task_tools_includes_subtask(
        self, temp_project_with_kiln_task_tool, tmp_path: Path
    ):
        """Test that exporting a task with kiln_task tool includes the subtask."""
        output_path = tmp_path / "output.zip"
        extract_path = tmp_path / "extracted"

        # Only request the main task
        package_project(
            project_path=temp_project_with_kiln_task_tool["path"],
            tasks=temp_project_with_kiln_task_tool["main_task"].id,
            all_tasks=False,
            output=output_path,
        )

        assert output_path.exists()

        with zipfile.ZipFile(output_path, "r") as zipf:
            zipf.extractall(extract_path)

        loaded_project = Project.load_from_file(extract_path / "project.kiln")

        # Both main task and subtask should be exported
        tasks = loaded_project.tasks()
        assert len(tasks) == 2

        task_ids = {t.id for t in tasks}
        assert temp_project_with_kiln_task_tool["main_task"].id in task_ids
        assert temp_project_with_kiln_task_tool["subtask"].id in task_ids

        # Tool server should be exported
        servers = loaded_project.external_tool_servers()
        assert len(servers) == 1
        assert servers[0].id == temp_project_with_kiln_task_tool["tool_server"].id

    def test_exports_with_skill_tools(
        self, temp_project_with_skill_tool: SkillToolProjectFixture, tmp_path: Path
    ) -> None:
        """Test full export with skill tools includes the skill."""
        output_path = tmp_path / "output.zip"
        extract_path = tmp_path / "extracted"

        package_project(
            project_path=temp_project_with_skill_tool["path"],
            tasks=temp_project_with_skill_tool["task"].id,
            all_tasks=False,
            output=output_path,
        )

        assert output_path.exists()

        with zipfile.ZipFile(output_path, "r") as zipf:
            zipf.extractall(extract_path)

        loaded_project = Project.load_from_file(extract_path / "project.kiln")
        skills = loaded_project.skills()
        assert len(skills) == 1
        assert skills[0].id == temp_project_with_skill_tool["skill"].id
        assert skills[0].body() == "This is the skill body with instructions."

    def test_rejects_rag_tools(self, temp_project_with_rag_tool, tmp_path: Path):
        """Test that RAG tools cause export to fail."""
        output_path = tmp_path / "output.zip"

        with pytest.raises(typer.Exit) as exc_info:
            package_project(
                project_path=temp_project_with_rag_tool["path"],
                tasks=temp_project_with_rag_tool["task"].id,
                all_tasks=False,
                output=output_path,
            )
        assert exc_info.value.exit_code == 1

    def test_prompts_for_mcp_local_and_proceeds(
        self, temp_project_with_mcp_local_tool, tmp_path: Path
    ):
        """Test that MCP local tools prompt and can proceed."""
        output_path = tmp_path / "output.zip"

        with patch(
            "kiln_ai.cli.commands.package_project.typer.confirm", return_value=True
        ):
            package_project(
                project_path=temp_project_with_mcp_local_tool["path"],
                tasks=temp_project_with_mcp_local_tool["task"].id,
                all_tasks=False,
                output=output_path,
            )

        assert output_path.exists()


OUTPUT_TO_CWD = False


@pytest.fixture
def temp_project_with_evals(tmp_path: Path):
    """Create a project with a task, run config, evals (with configs and runs), and a document."""
    project = Project(name="Training Project", path=tmp_path / "project.kiln")
    project.save_to_file()

    task = Task(
        name="Eval Task",
        instruction="Task with evals",
        parent=project,
    )
    task.save_to_file()

    run_config = TaskRunConfig(
        name="Default RC",
        parent=task,
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name="openai",
            prompt_id=PromptGenerators.SIMPLE.value,
            structured_output_mode=StructuredOutputMode.default,
        ),
    )
    run_config.save_to_file()

    task.default_run_config_id = run_config.id
    task.save_to_file()

    output_score = EvalOutputScore(
        name="quality",
        instruction="Rate the quality",
        type=TaskOutputRatingType.five_star,
    )

    eval1 = Eval(
        name="Eval One",
        eval_set_filter_id="all",
        eval_configs_filter_id="all",
        output_scores=[output_score],
        parent=task,
    )
    eval1.save_to_file()

    eval1_config = EvalConfig(
        name="Config A",
        model_name="gpt-4o",
        model_provider="openai",
        config_type=EvalConfigType.g_eval,
        properties={"eval_steps": ["step1", "step2"]},
        parent=eval1,
    )
    eval1_config.save_to_file()

    eval1_run = EvalRun(
        dataset_id="ds_item_1",
        task_run_config_id=run_config.id,
        input="test input",
        output="test output",
        scores={"quality": 4.0},
        parent=eval1_config,
    )
    eval1_run.save_to_file()

    eval2 = Eval(
        name="Eval Two",
        eval_set_filter_id="all",
        eval_configs_filter_id="all",
        output_scores=[output_score],
        parent=task,
    )
    eval2.save_to_file()

    eval2_config = EvalConfig(
        name="Config B",
        model_name="gpt-4o",
        model_provider="openai",
        config_type=EvalConfigType.g_eval,
        properties={"eval_steps": ["step1"]},
        parent=eval2,
    )
    eval2_config.save_to_file()

    task_run = TaskRun(
        input="sample input",
        output=TaskOutput(output="sample output"),
        parent=task,
    )
    task_run.save_to_file()

    doc_dir = tmp_path / "documents" / "doc1 - TestDoc"
    doc_dir.mkdir(parents=True)
    (doc_dir / "document.kiln").write_text(
        '{"v": 1, "model_type": "document", "name": "TestDoc", "description": "A test doc", "original_file": {"name": "test.txt", "path": "test.txt", "size": 10, "mime_type": "text/plain"}, "kind": "document"}'
    )

    return {
        "project": project,
        "task": task,
        "run_config": run_config,
        "task_run": task_run,
        "eval1": eval1,
        "eval1_config": eval1_config,
        "eval1_run": eval1_run,
        "eval2": eval2,
        "eval2_config": eval2_config,
        "path": tmp_path,
    }


class TestExportEvals:
    def test_exports_selected_evals(self, temp_project_with_evals, tmp_path: Path):
        """Test that only selected eval IDs are exported with full subtree."""
        source = temp_project_with_evals
        _, exported_project = create_export_directory(source["project"])
        exported_task, _ = export_task(
            source["task"], source["run_config"], exported_project
        )

        exported_ids = export_evals(source["task"], [source["eval1"].id], exported_task)

        assert exported_ids == [source["eval1"].id]
        exported_evals = exported_task.evals(readonly=True)
        assert len(exported_evals) == 1
        assert exported_evals[0].id == source["eval1"].id

        configs = exported_evals[0].configs(readonly=True)
        assert len(configs) == 1
        assert configs[0].id == source["eval1_config"].id

    def test_exports_no_evals_when_ids_dont_match(
        self, temp_project_with_evals, tmp_path: Path
    ):
        source = temp_project_with_evals
        _, exported_project = create_export_directory(source["project"])
        exported_task, _ = export_task(
            source["task"], source["run_config"], exported_project
        )

        exported_ids = export_evals(source["task"], ["nonexistent_id"], exported_task)

        assert exported_ids == []
        assert len(exported_task.evals(readonly=True)) == 0


class TestExportDocuments:
    def test_exports_documents(self, temp_project_with_evals, tmp_path: Path):
        source = temp_project_with_evals
        _, exported_project = create_export_directory(source["project"])

        export_documents(source["project"], exported_project)

        assert exported_project.path is not None
        docs_dir = exported_project.path.parent / "documents"
        assert docs_dir.exists()
        doc_files = list(docs_dir.rglob("document.kiln"))
        assert len(doc_files) == 1

    def test_no_error_when_no_documents(self, temp_project, tmp_path: Path):
        source = temp_project
        _, exported_project = create_export_directory(source["project"])

        export_documents(source["project"], exported_project)

        assert exported_project.path is not None
        docs_dir = exported_project.path.parent / "documents"
        assert not docs_dir.exists()


class TestPackageProjectForTraining:
    def test_full_package_with_evals_and_docs(
        self, temp_project_with_evals, tmp_path: Path
    ):
        source = temp_project_with_evals
        output_path = (
            Path("./exported_training.zip")
            if OUTPUT_TO_CWD
            else tmp_path / "output" / "training.zip"
        )

        package_project_for_training(
            project=source["project"],
            task_ids=[source["task"].id],
            run_config_id=source["run_config"].id,
            eval_ids=[source["eval1"].id, source["eval2"].id],
            output=output_path,
        )

        assert output_path.exists()

        extract_path = tmp_path / "extracted"
        with zipfile.ZipFile(output_path, "r") as zipf:
            names = zipf.namelist()
            zipf.extractall(extract_path)

        loaded_project = Project.load_from_file(extract_path / "project.kiln")
        tasks = loaded_project.tasks()
        assert len(tasks) == 1

        loaded_task = tasks[0]
        loaded_evals = loaded_task.evals(readonly=True)
        assert len(loaded_evals) == 2
        loaded_eval_ids = {e.id for e in loaded_evals}
        assert source["eval1"].id in loaded_eval_ids
        assert source["eval2"].id in loaded_eval_ids

        for loaded_eval in loaded_evals:
            configs = loaded_eval.configs(readonly=True)
            assert len(configs) >= 1

        assert any("documents/" in n for n in names)

    def test_package_without_documents(self, temp_project_with_evals, tmp_path: Path):
        source = temp_project_with_evals
        output_path = tmp_path / "output" / "no_docs.zip"

        package_project_for_training(
            project=source["project"],
            task_ids=[source["task"].id],
            run_config_id=source["run_config"].id,
            eval_ids=[source["eval1"].id],
            output=output_path,
            config=PackageForTrainingConfig(include_documents=False),
        )

        assert output_path.exists()

        with zipfile.ZipFile(output_path, "r") as zipf:
            names = zipf.namelist()

        assert not any("documents/" in n for n in names)

    def test_package_with_specific_eval_ids(
        self, temp_project_with_evals, tmp_path: Path
    ):
        source = temp_project_with_evals
        output_path = tmp_path / "output" / "one_eval.zip"

        package_project_for_training(
            project=source["project"],
            task_ids=[source["task"].id],
            run_config_id=source["run_config"].id,
            eval_ids=[source["eval1"].id],
            output=output_path,
        )

        extract_path = tmp_path / "extracted_one"
        with zipfile.ZipFile(output_path, "r") as zipf:
            zipf.extractall(extract_path)

        loaded_project = Project.load_from_file(extract_path / "project.kiln")
        loaded_task = loaded_project.tasks()[0]
        loaded_evals = loaded_task.evals(readonly=True)
        assert len(loaded_evals) == 1
        assert loaded_evals[0].id == source["eval1"].id

    def test_eval_subtree_fully_included(self, temp_project_with_evals, tmp_path: Path):
        source = temp_project_with_evals
        output_path = tmp_path / "output" / "subtree.zip"

        package_project_for_training(
            project=source["project"],
            task_ids=[source["task"].id],
            run_config_id=source["run_config"].id,
            eval_ids=[source["eval1"].id],
            output=output_path,
        )

        extract_path = tmp_path / "extracted_subtree"
        with zipfile.ZipFile(output_path, "r") as zipf:
            zipf.extractall(extract_path)

        loaded_project = Project.load_from_file(extract_path / "project.kiln")
        loaded_task = loaded_project.tasks()[0]
        loaded_evals = loaded_task.evals(readonly=True)
        assert len(loaded_evals) == 1

        configs = loaded_evals[0].configs(readonly=True)
        assert len(configs) == 1
        assert configs[0].id == source["eval1_config"].id

        runs = configs[0].runs(readonly=True)
        assert len(runs) == 1
        assert runs[0].id == source["eval1_run"].id

    def test_default_config_includes_task_runs_and_eval_config_runs(
        self, temp_project_with_evals, tmp_path: Path
    ):
        source = temp_project_with_evals
        output_path = tmp_path / "output" / "default_runs.zip"

        package_project_for_training(
            project=source["project"],
            task_ids=[source["task"].id],
            run_config_id=source["run_config"].id,
            eval_ids=[source["eval1"].id],
            output=output_path,
        )

        extract_path = tmp_path / "extracted_default_runs"
        with zipfile.ZipFile(output_path, "r") as zipf:
            zipf.extractall(extract_path)

        loaded_project = Project.load_from_file(extract_path / "project.kiln")
        loaded_task = loaded_project.tasks()[0]

        task_runs = loaded_task.runs(include_intermediate_runs=True)
        assert len(task_runs) == 1
        assert task_runs[0].id == source["task_run"].id

        loaded_evals = loaded_task.evals(readonly=True)
        assert len(loaded_evals) == 1
        configs = loaded_evals[0].configs(readonly=True)
        assert len(configs) == 1
        eval_runs = configs[0].runs(readonly=True)
        assert len(eval_runs) == 1
        assert eval_runs[0].id == source["eval1_run"].id

    def test_exclude_task_runs(self, temp_project_with_evals, tmp_path: Path):
        source = temp_project_with_evals
        output_path = tmp_path / "output" / "no_task_runs.zip"

        package_project_for_training(
            project=source["project"],
            task_ids=[source["task"].id],
            run_config_id=source["run_config"].id,
            eval_ids=[source["eval1"].id],
            output=output_path,
            config=PackageForTrainingConfig(exclude_task_runs=True),
        )

        extract_path = tmp_path / "extracted_no_task_runs"
        with zipfile.ZipFile(output_path, "r") as zipf:
            names = zipf.namelist()
            zipf.extractall(extract_path)

        assert not any("/runs/" in n and "evals/" not in n for n in names)

        loaded_project = Project.load_from_file(extract_path / "project.kiln")
        loaded_task = loaded_project.tasks()[0]

        task_runs_dir = loaded_task.path.parent / "runs"
        assert not task_runs_dir.exists()

        loaded_evals = loaded_task.evals(readonly=True)
        configs = loaded_evals[0].configs(readonly=True)
        eval_runs = configs[0].runs(readonly=True)
        assert len(eval_runs) == 1

    def test_exclude_eval_config_runs(self, temp_project_with_evals, tmp_path: Path):
        source = temp_project_with_evals
        output_path = tmp_path / "output" / "no_eval_runs.zip"

        package_project_for_training(
            project=source["project"],
            task_ids=[source["task"].id],
            run_config_id=source["run_config"].id,
            eval_ids=[source["eval1"].id],
            output=output_path,
            config=PackageForTrainingConfig(exclude_eval_config_runs=True),
        )

        extract_path = tmp_path / "extracted_no_eval_runs"
        with zipfile.ZipFile(output_path, "r") as zipf:
            zipf.extractall(extract_path)

        loaded_project = Project.load_from_file(extract_path / "project.kiln")
        loaded_task = loaded_project.tasks()[0]

        task_runs = loaded_task.runs(include_intermediate_runs=True)
        assert len(task_runs) == 1
        assert task_runs[0].id == source["task_run"].id

        loaded_evals = loaded_task.evals(readonly=True)
        assert len(loaded_evals) == 1
        configs = loaded_evals[0].configs(readonly=True)
        assert len(configs) == 1
        assert configs[0].id == source["eval1_config"].id
        eval_runs = configs[0].runs(readonly=True)
        assert len(eval_runs) == 0


class TestPackageProjectForTrainingNoncli:
    def test_missing_task_raises_value_error(
        self, temp_project_with_evals, tmp_path: Path
    ):
        source = temp_project_with_evals
        output_path = tmp_path / "output" / "bad.zip"

        with pytest.raises(ValueError, match="Task ID\\(s\\) not found"):
            package_project_for_training(
                project=source["project"],
                task_ids=["nonexistent_id"],
                run_config_id=source["run_config"].id,
                eval_ids=[],
                output=output_path,
            )

    def test_bad_prompt_raises_value_error(
        self, temp_project_with_evals, tmp_path: Path
    ):
        source = temp_project_with_evals
        output_path = tmp_path / "output" / "bad_prompt.zip"

        with patch(
            "kiln_ai.cli.commands.package_project.prompt_builder_from_id",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(ValueError, match="Failed to build prompt"):
                package_project_for_training(
                    project=source["project"],
                    task_ids=[source["task"].id],
                    run_config_id=source["run_config"].id,
                    eval_ids=[],
                    output=output_path,
                )


class TestExportTaskRuns:
    def test_copies_runs_directory(self, temp_project_with_evals, tmp_path: Path):
        source = temp_project_with_evals
        _, exported_project = create_export_directory(source["project"])
        exported_task, _ = export_task(
            source["task"], source["run_config"], exported_project
        )

        export_task_runs(source["task"], exported_task)

        dest_runs_dir = exported_task.path.parent / "runs"
        assert dest_runs_dir.exists()
        run_files = list(dest_runs_dir.rglob("*.kiln"))
        assert len(run_files) == 1

    def test_noop_when_no_runs_directory(self, temp_project, tmp_path: Path):
        source = temp_project
        _, exported_project = create_export_directory(source["project"])
        exported_task, _ = export_task(
            source["task"], source["run_config"], exported_project
        )

        export_task_runs(source["task"], exported_task)

        dest_runs_dir = exported_task.path.parent / "runs"
        assert not dest_runs_dir.exists()


class TestExportEvalInputs:
    def test_copies_eval_inputs_directory(
        self, temp_project_with_evals, tmp_path: Path
    ):
        from kiln_ai.datamodel.eval import (
            EvalInput,
            SingleTurnEvalInputData,
            UserMessage,
        )

        source = temp_project_with_evals
        eval_input = EvalInput(
            parent=source["task"],
            data=SingleTurnEvalInputData(user_message=UserMessage(text="hello")),
            reference={"expected": "HELLO"},
            tags=["train_tag"],
        )
        eval_input.save_to_file()

        _, exported_project = create_export_directory(source["project"])
        exported_task, _ = export_task(
            source["task"], source["run_config"], exported_project
        )

        export_eval_inputs(source["task"], exported_task)

        dest_dir = exported_task.path.parent / "eval_inputs"
        assert dest_dir.exists()
        eval_input_files = list(dest_dir.rglob("*.kiln"))
        assert len(eval_input_files) == 1
        reloaded = EvalInput.load_from_file(eval_input_files[0])
        assert reloaded.reference == {"expected": "HELLO"}
        assert reloaded.tags == ["train_tag"]

    def test_noop_when_no_eval_inputs_directory(self, temp_project, tmp_path: Path):
        source = temp_project
        _, exported_project = create_export_directory(source["project"])
        exported_task, _ = export_task(
            source["task"], source["run_config"], exported_project
        )

        export_eval_inputs(source["task"], exported_task)

        dest_dir = exported_task.path.parent / "eval_inputs"
        assert not dest_dir.exists()

    def test_package_project_for_training_includes_eval_inputs(
        self, temp_project_with_evals, tmp_path: Path
    ):
        from kiln_ai.datamodel.eval import (
            EvalInput,
            SingleTurnEvalInputData,
            UserMessage,
        )

        source = temp_project_with_evals
        eval_input = EvalInput(
            parent=source["task"],
            data=SingleTurnEvalInputData(user_message=UserMessage(text="hello")),
        )
        eval_input.save_to_file()

        output_path = tmp_path / "output" / "with_eval_inputs.zip"
        package_project_for_training(
            project=source["project"],
            task_ids=[source["task"].id],
            run_config_id=source["run_config"].id,
            eval_ids=[],
            output=output_path,
        )

        import zipfile as _zipfile

        with _zipfile.ZipFile(output_path) as zf:
            assert any("eval_inputs" in name for name in zf.namelist())


# ──────────────────────────────────────────────────────────────────────
# Phase 4: code-as-file artifacts survive Kiln's export / copy paths
#
# tool.py / scorer.py are real files in the artifact folder (Phases 1-2), so
# they must travel with the artifact through the mechanisms that move artifacts
# between locations: the export/zip packager and folder-level copies (which is
# how git-sync tracks the project tree). These tests lock that in so it can't
# regress.
# ──────────────────────────────────────────────────────────────────────

SCORER_SOURCE = (
    "def score(output, reference_data):\n"
    "    return {'accuracy': 1.0 if output == reference_data.get('answer') else 0.0}\n"
)
TOOL_SOURCE = "def run(x):\n    return x\n"


@pytest.fixture
def temp_project_with_code_artifacts(tmp_path: Path):
    """Project with a code judge (scorer.py) and a code tool (tool.py) on disk."""
    project = Project(name="Code Artifacts Project", path=tmp_path / "project.kiln")
    project.save_to_file()

    task = Task(
        name="Code Artifacts Task",
        instruction="Task with code artifacts",
        parent=project,
    )
    task.save_to_file()

    run_config = TaskRunConfig(
        name="Default RC",
        parent=task,
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4o",
            model_provider_name="openai",
            prompt_id=PromptGenerators.SIMPLE.value,
            structured_output_mode=StructuredOutputMode.default,
        ),
    )
    run_config.save_to_file()

    task.default_run_config_id = run_config.id
    task.save_to_file()

    eval_obj = Eval(
        name="Code Judge Eval",
        eval_set_filter_id="all",
        eval_configs_filter_id="all",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                instruction="Exact match",
                type=TaskOutputRatingType.pass_fail,
            )
        ],
        parent=task,
    )
    eval_obj.save_to_file()

    # A v2 code_eval config: its CodeEvalProperties.code lands in scorer.py.
    code_eval_config = EvalConfig(
        name="Code Judge Config",
        config_type=EvalConfigType.v2,
        properties=CodeEvalProperties(code=SCORER_SOURCE, reference_keys=["answer"]),
        parent=eval_obj,
    )
    code_eval_config.save_to_file()

    # A project-level code tool: its code lands in tool.py.
    code_tool = CodeTool(
        name="My Tool",
        tool_function_name="my_tool",
        tool_description="Does things",
        parameters_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
        },
        code=TOOL_SOURCE,
        parent=project,
    )
    code_tool.save_to_file()

    return {
        "project": project,
        "task": task,
        "run_config": run_config,
        "eval": eval_obj,
        "code_eval_config": code_eval_config,
        "code_tool": code_tool,
    }


class TestCodeArtifactsSurviveExport:
    def test_code_judge_scorer_py_survives_training_export(
        self, temp_project_with_code_artifacts, tmp_path: Path
    ):
        """scorer.py travels with the eval config through the export/zip path.

        package_project_for_training exports evals via shutil.copytree, which
        carries the whole eval-config folder (eval_config.kiln + scorer.py) into
        the zip. Reloading the exported config must reconstruct `code` from the
        carried scorer.py.
        """
        source = temp_project_with_code_artifacts
        output_path = tmp_path / "output" / "training.zip"

        package_project_for_training(
            project=source["project"],
            task_ids=[source["task"].id],
            run_config_id=source["run_config"].id,
            eval_ids=[source["eval"].id],
            output=output_path,
        )

        assert output_path.exists()

        extract_path = tmp_path / "extracted"
        with zipfile.ZipFile(output_path, "r") as zipf:
            names = zipf.namelist()
            zipf.extractall(extract_path)

        # The sibling scorer.py is physically in the zip (not the .kiln JSON).
        assert any(n.endswith("/" + SCORER_CODE_FILENAME) for n in names)

        loaded_project = Project.load_from_file(extract_path / "project.kiln")
        loaded_task = loaded_project.tasks()[0]
        loaded_eval = next(
            e for e in loaded_task.evals(readonly=True) if e.id == source["eval"].id
        )
        loaded_config = next(
            c
            for c in loaded_eval.configs(readonly=True)
            if c.id == source["code_eval_config"].id
        )

        # code is reconstructed from scorer.py and absent from the on-disk JSON.
        assert isinstance(loaded_config.properties, CodeEvalProperties)
        assert loaded_config.properties.code == SCORER_SOURCE
        on_disk = json.loads(loaded_config.path.read_text(encoding="utf-8"))
        assert "code" not in on_disk["properties"]

    def test_code_tool_tool_py_survives_folder_copy(
        self, temp_project_with_code_artifacts, tmp_path: Path
    ):
        """tool.py travels when the artifact folder is copied elsewhere.

        The project packager (package_project) has no code-tool export path: it
        copies project.kiln, tasks/run-configs, tool servers, skills, evals, and
        documents, but never code_tools. (A code tool can be referenced from a
        run config's tool list via CODE_TOOL_ID_PREFIX, but classify_tool_id has
        no branch for it, so it classifies as "unknown" and validate_tools aborts
        packaging — a pre-existing packager limitation, out of scope here.)
        Either way there is no export path, so the mechanism that must carry a
        code tool between locations is the folder-level copy that git-sync (and
        any filesystem copy) performs. Copy the code-tool folder to a new
        location and confirm the reloaded tool reconstructs `code` from the
        carried tool.py.
        """
        source = temp_project_with_code_artifacts
        code_tool = source["code_tool"]

        src_folder = code_tool.path.parent
        # Sanity: the source really stores code as a sibling file, not inline.
        assert (src_folder / TOOL_CODE_FILENAME).read_text(encoding="utf-8") == (
            TOOL_SOURCE
        )
        assert "code" not in json.loads(code_tool.path.read_text(encoding="utf-8"))

        dest_folder = tmp_path / "copied_project" / "code_tools" / src_folder.name
        shutil.copytree(src_folder, dest_folder)

        reloaded = CodeTool.load_from_file(dest_folder / code_tool.path.name)
        assert reloaded.code == TOOL_SOURCE
        assert reloaded.tool_function_name == "my_tool"
