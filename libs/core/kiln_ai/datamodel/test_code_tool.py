"""Tests for CodeTool datamodel — validation, defaults, save/load, parent registration."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from kiln_ai.datamodel.code_tool import TOOL_CODE_FILENAME, CodeTool
from kiln_ai.datamodel.project import Project

VALID_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "string"}},
}
EMPTY_SCHEMA = {"type": "object", "properties": {}}

SYNC_RUN = "def run(x):\n    return x\n"
ASYNC_RUN = "async def run(x):\n    return x\n"


def _make_code_tool(**overrides):
    defaults = {
        "name": "My Tool",
        "tool_function_name": "my_tool",
        "tool_description": "Does things",
        "parameters_schema": VALID_SCHEMA,
        "code": SYNC_RUN,
    }
    defaults.update(overrides)
    return CodeTool(**defaults)


# ---------------------------------------------------------------------------
# Code validation trio
# ---------------------------------------------------------------------------


class TestCodeValidation:
    def test_valid_sync_run(self):
        ct = _make_code_tool(code=SYNC_RUN)
        assert ct.code == SYNC_RUN

    def test_valid_async_run(self):
        ct = _make_code_tool(code=ASYNC_RUN)
        assert ct.code == ASYNC_RUN

    def test_missing_run_rejected(self):
        with pytest.raises(ValidationError, match="module-level 'run' function"):
            _make_code_tool(code="def helper():\n    pass\n")

    def test_nested_run_rejected(self):
        code = "class Foo:\n    def run(self):\n        pass\n"
        with pytest.raises(ValidationError, match="module-level 'run' function"):
            _make_code_tool(code=code)

    def test_run_as_variable_rejected(self):
        with pytest.raises(ValidationError, match="module-level 'run' function"):
            _make_code_tool(code="run = 42\n")

    def test_syntax_error_rejected(self):
        with pytest.raises(ValidationError, match="syntax error"):
            _make_code_tool(code="def run(\n")

    def test_code_size_cap(self):
        big_code = "def run():\n    pass\n" + "# " + "x" * (64 * 1024)
        with pytest.raises(ValidationError, match="too large"):
            _make_code_tool(code=big_code)

    def test_empty_code_rejected(self):
        with pytest.raises(ValidationError, match="module-level 'run' function"):
            _make_code_tool(code="")


# ---------------------------------------------------------------------------
# Function name validation
# ---------------------------------------------------------------------------


class TestFunctionName:
    @pytest.mark.parametrize(
        "name",
        ["a", "my_tool", "tool123", "a" * 64],
    )
    def test_valid_names(self, name):
        ct = _make_code_tool(tool_function_name=name)
        assert ct.tool_function_name == name

    @pytest.mark.parametrize(
        "name",
        [
            "MyTool",  # uppercase
            "1tool",  # starts with digit
            "_tool",  # starts with underscore
            "tool-name",  # hyphen
            "a" * 65,  # too long
            "",  # empty
            "my_tool\n",  # trailing newline
        ],
    )
    def test_invalid_names(self, name):
        with pytest.raises(ValidationError, match="tool_function_name"):
            _make_code_tool(tool_function_name=name)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_valid_schema(self):
        ct = _make_code_tool(parameters_schema=VALID_SCHEMA)
        assert ct.parameters_schema == VALID_SCHEMA

    def test_empty_properties_allowed(self):
        ct = _make_code_tool(parameters_schema=EMPTY_SCHEMA)
        assert ct.parameters_schema == EMPTY_SCHEMA

    def test_non_object_schema_rejected(self):
        with pytest.raises(ValidationError, match="object with properties"):
            _make_code_tool(parameters_schema={"type": "string"})

    def test_missing_properties_rejected(self):
        with pytest.raises(ValidationError, match="object with properties"):
            _make_code_tool(parameters_schema={"type": "object"})

    @pytest.mark.parametrize(
        "param_name",
        ["from", "class", "return", "lambda", "None", "True", "import"],
    )
    def test_reserved_word_param_rejected(self, param_name):
        schema = {
            "type": "object",
            "properties": {param_name: {"type": "string"}},
        }
        with pytest.raises(ValidationError, match="reserved Python keyword"):
            _make_code_tool(parameters_schema=schema)

    @pytest.mark.parametrize("param_name", ["match", "case", "type", "_"])
    def test_soft_keyword_param_allowed(self, param_name):
        # Soft keywords are still valid identifiers, so def run(match) works fine.
        schema = {
            "type": "object",
            "properties": {param_name: {"type": "string"}},
        }
        ct = _make_code_tool(parameters_schema=schema)
        assert param_name in ct.parameters_schema["properties"]

    @pytest.mark.parametrize("param_name", ["from_date", "class_name", "query"])
    def test_normal_param_names_allowed(self, param_name):
        schema = {
            "type": "object",
            "properties": {param_name: {"type": "string"}},
        }
        ct = _make_code_tool(parameters_schema=schema)
        assert param_name in ct.parameters_schema["properties"]

    def test_reserved_word_param_still_loads_from_file(self, tmp_path):
        # Files saved before the reserved-word check may carry such a parameter;
        # they must still load, and loaded instances must still accept edits.
        project = Project(name="test_project", path=tmp_path / "project")
        project.save_to_file()

        ct = _make_code_tool()
        ct.parent = project
        ct.save_to_file()

        assert ct.path is not None
        data = json.loads(ct.path.read_text())
        data["parameters_schema"]["properties"] = {"from": {"type": "string"}}
        ct.path.write_text(json.dumps(data))

        loaded = CodeTool.from_id_and_parent_path(ct.id, project.path)
        assert loaded is not None
        assert "from" in loaded.parameters_schema["properties"]

        # Assignment re-runs model validators; archiving a legacy tool must not raise.
        loaded.is_archived = True
        loaded.save_to_file()


# ---------------------------------------------------------------------------
# Allowlist validation
# ---------------------------------------------------------------------------


class TestAllowlistValidation:
    def test_valid_allowlist(self):
        ct = _make_code_tool(
            tool_allowlist=[
                "kiln_tool::add_numbers",
                "mcp::remote::server1::tool1",
            ]
        )
        assert len(ct.tool_allowlist) == 2

    def test_empty_allowlist(self):
        ct = _make_code_tool(tool_allowlist=[])
        assert ct.tool_allowlist == []

    def test_rejects_skill_ids(self):
        with pytest.raises(ValidationError, match="Skill tool IDs cannot"):
            _make_code_tool(tool_allowlist=["kiln_tool::skill::some_skill"])

    def test_rejects_unmanaged_ids(self):
        with pytest.raises(ValidationError, match="Unmanaged tool IDs cannot"):
            _make_code_tool(tool_allowlist=["kiln_unmanaged::some_tool"])

    def test_rejects_duplicates(self):
        with pytest.raises(ValidationError, match="Duplicate tool ID"):
            _make_code_tool(
                tool_allowlist=[
                    "kiln_tool::add_numbers",
                    "kiln_tool::add_numbers",
                ]
            )

    def test_rejects_self_reference(self):
        ct_id = "123456789012"
        with pytest.raises(ValidationError, match="cannot reference itself"):
            _make_code_tool(
                id=ct_id,
                tool_allowlist=[f"kiln_tool::code::{ct_id}"],
            )

    def test_rejects_invalid_tool_id(self):
        with pytest.raises(ValidationError, match="Invalid tool ID"):
            _make_code_tool(tool_allowlist=["not_a_valid_tool_id"])


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_timeout(self):
        ct = _make_code_tool()
        assert ct.timeout_seconds == 60

    def test_default_allowlist(self):
        ct = _make_code_tool()
        assert ct.tool_allowlist == []

    def test_default_archived(self):
        ct = _make_code_tool()
        assert ct.is_archived is False

    def test_default_description(self):
        ct = _make_code_tool()
        assert ct.description is None

    def test_timeout_min_1(self):
        ct = _make_code_tool(timeout_seconds=1)
        assert ct.timeout_seconds == 1

    def test_timeout_below_min_rejected(self):
        with pytest.raises(ValidationError):
            _make_code_tool(timeout_seconds=0)

    def test_tool_description_required(self):
        with pytest.raises(ValidationError):
            _make_code_tool(tool_description="")


# ---------------------------------------------------------------------------
# Save/load round-trip and parent registration
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        project = Project(name="test_project", path=tmp_path / "project")
        project.save_to_file()

        ct = _make_code_tool(
            tool_allowlist=["kiln_tool::add_numbers"],
        )
        ct.parent = project
        ct.save_to_file()

        loaded = CodeTool.from_id_and_parent_path(ct.id, project.path)
        assert loaded is not None
        assert loaded.tool_function_name == ct.tool_function_name
        assert loaded.code == ct.code
        assert loaded.parameters_schema == ct.parameters_schema
        assert loaded.tool_description == ct.tool_description
        assert loaded.timeout_seconds == ct.timeout_seconds
        assert loaded.tool_allowlist == ct.tool_allowlist

    def test_project_code_tools_accessor(self, tmp_path):
        project = Project(name="test_project", path=tmp_path / "project")
        project.save_to_file()

        ct = _make_code_tool()
        ct.parent = project
        ct.save_to_file()

        tools = project.code_tools()
        assert len(tools) == 1
        assert tools[0].tool_function_name == "my_tool"


# ---------------------------------------------------------------------------
# Code-as-file storage: code lives in tool.py, not the .kiln JSON
# ---------------------------------------------------------------------------


@pytest.fixture
def saved_project(tmp_path):
    project = Project(name="test_project", path=tmp_path / "project")
    project.save_to_file()
    return project


def _save_tool(project, **overrides):
    ct = _make_code_tool(**overrides)
    ct.parent = project
    ct.save_to_file()
    return ct


class TestFileStorage:
    def test_save_writes_tool_py_and_omits_code_from_kiln(self, saved_project):
        ct = _save_tool(saved_project, code=SYNC_RUN)

        tool_py = ct.path.parent / TOOL_CODE_FILENAME
        assert tool_py.exists()
        assert tool_py.read_text(encoding="utf-8") == SYNC_RUN

        on_disk = json.loads(ct.path.read_text(encoding="utf-8"))
        assert "code" not in on_disk
        # Other functional fields still live in the .kiln JSON.
        assert on_disk["tool_function_name"] == ct.tool_function_name
        assert on_disk["parameters_schema"] == ct.parameters_schema

    def test_load_reconstructs_code_from_file(self, saved_project):
        ct = _save_tool(
            saved_project, code=ASYNC_RUN, tool_allowlist=["kiln_tool::add_numbers"]
        )

        loaded = CodeTool.load_from_file(ct.path)
        assert loaded.code == ASYNC_RUN
        assert loaded.tool_function_name == ct.tool_function_name
        assert loaded.parameters_schema == ct.parameters_schema
        assert loaded.tool_description == ct.tool_description
        assert loaded.timeout_seconds == ct.timeout_seconds
        assert loaded.tool_allowlist == ct.tool_allowlist

    def test_missing_tool_py_fails_load(self, saved_project):
        ct = _save_tool(saved_project)
        (ct.path.parent / TOOL_CODE_FILENAME).unlink()

        with pytest.raises(ValueError, match=TOOL_CODE_FILENAME):
            CodeTool.load_from_file(ct.path)

    def test_corrupted_tool_py_fails_validator_on_load(self, saved_project):
        ct = _save_tool(saved_project)
        # Hand-edit tool.py to source without a module-level `run` function.
        (ct.path.parent / TOOL_CODE_FILENAME).write_text(
            "def helper():\n    pass\n", encoding="utf-8"
        )

        with pytest.raises(ValidationError, match="module-level 'run' function"):
            CodeTool.load_from_file(ct.path)

    def test_save_is_idempotent(self, saved_project):
        ct = _save_tool(saved_project, code=SYNC_RUN)
        tool_py = ct.path.parent / TOOL_CODE_FILENAME

        first_kiln = ct.path.read_bytes()
        first_py = tool_py.read_bytes()

        loaded = CodeTool.load_from_file(ct.path)
        loaded.save_to_file()

        assert ct.path.read_bytes() == first_kiln
        assert tool_py.read_bytes() == first_py

    def test_clone_writes_fresh_tool_py(self, saved_project):
        original = _save_tool(saved_project, code=SYNC_RUN, name="Original")

        loaded = CodeTool.load_from_file(original.path)
        clone = CodeTool(
            name="Clone",
            tool_function_name=loaded.tool_function_name,
            tool_description=loaded.tool_description,
            parameters_schema=loaded.parameters_schema,
            code=loaded.code,
        )
        clone.parent = saved_project
        clone.save_to_file()

        assert clone.id != original.id
        clone_py = clone.path.parent / TOOL_CODE_FILENAME
        original_py = original.path.parent / TOOL_CODE_FILENAME
        assert clone_py != original_py
        assert clone_py.read_text(encoding="utf-8") == SYNC_RUN
        assert original_py.read_text(encoding="utf-8") == SYNC_RUN

    def test_delete_removes_folder_including_tool_py(self, saved_project):
        ct = _save_tool(saved_project)
        folder = ct.path.parent
        assert (folder / TOOL_CODE_FILENAME).exists()

        ct.delete()
        assert not folder.exists()

    def test_api_dump_keeps_code(self):
        ct = _make_code_tool(code=SYNC_RUN)

        # Without the save context, code stays in the dump and no file is written.
        with patch.object(Path, "write_text") as mock_write:
            assert ct.model_dump()["code"] == SYNC_RUN
            assert json.loads(ct.model_dump_json())["code"] == SYNC_RUN
        mock_write.assert_not_called()

    def test_source_dir_missing_from_load_context_fails(self):
        # Defensive guard: loading_from_file set but source_dir absent (a future
        # base-model regression) fails clearly rather than silently skipping.
        with pytest.raises(
            ValidationError, match="source_dir missing from load context"
        ):
            CodeTool.model_validate(
                {"name": "My Tool"}, context={"loading_from_file": True}
            )

    def test_serialize_rejects_non_directory_dest_path(self, tmp_path):
        ct = _make_code_tool(code=SYNC_RUN)
        not_a_dir = tmp_path / "does_not_exist"
        with pytest.raises(ValueError, match="dest_path must be an existing directory"):
            ct.model_dump(context={"save_attachments": True, "dest_path": not_a_dir})

    def test_other_model_loads_unaffected_by_source_dir(self, saved_project):
        # A non-code model loads normally even though load now passes the new
        # source_dir context key to every model's validation.
        loaded = Project.load_from_file(saved_project.path)
        assert loaded.name == saved_project.name
