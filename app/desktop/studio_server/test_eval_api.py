import json
import logging
from dataclasses import dataclass
from typing import ClassVar, Dict, Iterable, List, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from kiln_ai.adapters.eval.eval_runner import EvalRunner
from kiln_ai.adapters.ml_model_list import ModelProviderName
from kiln_ai.adapters.run_output import RunOutput
from kiln_ai.datamodel import (
    DataSource,
    DataSourceType,
    Finetune,
    Priority,
    Project,
    RequirementRating,
    Task,
    TaskOutput,
    TaskOutputRating,
    TaskOutputRatingType,
    TaskRequirement,
    TaskRun,
)
from kiln_ai.datamodel.basemodel import ID_TYPE
from kiln_ai.datamodel.datamodel_enums import (
    FineTuneStatusType,
    StructuredOutputMode,
    TurnMode,
)
from kiln_ai.datamodel.eval import (
    ContainsProperties,
    Eval,
    EvalConfig,
    EvalConfigType,
    EvalDataType,
    EvalInput,
    EvalInputSplit,
    EvalOutputScore,
    EvalRun,
    EvalTemplateId,
    ExactMatchProperties,
    LlmJudgeProperties,
    MultiTurnSyntheticEvalInputData,
    PatternMatchProperties,
    SingleTurnEvalInputData,
    SyntheticUserInfo,
    TaskRunSplit,
    UserMessage,
)
from kiln_ai.datamodel.eval_splits import ItemSource, ResolvedSplit
from kiln_ai.datamodel.prompt import BasePrompt
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties
from kiln_ai.datamodel.spec import Spec, SpecStatus
from kiln_ai.datamodel.spec_properties import DesiredBehaviourProperties, SpecType
from kiln_ai.datamodel.task import TaskRunConfig
from kiln_ai.datamodel.task_run import EvalItemSource, Usage
from kiln_ai.datamodel.usage import MessageUsage
from kiln_ai.tools.base_tool import ToolCallResult
from kiln_ai.tools.sandbox_bridge import BridgeResult
from kiln_server.custom_errors import connect_custom_errors

from app.desktop.studio_server.eval_api import (
    CreateEvalConfigRequest,
    CreateEvaluatorRequest,
    _cached_test_split,
    compute_score_summary,
    connect_evals_api,
    eval_config_from_id,
    eval_item_input_text,
    eval_run_task_usage,
    get_all_run_configs,
    judge_requires_reference_data,
    resolve_eval_run_traces,
    resolved_split_or_422,
    reusable_frozen_prompt_id,
    score_summary_from_values,
    scored_trace_usage,
    scored_trace_usage_for_run_config,
    split_size,
    summary_eval_config,
    task_run_config_from_id,
)


def stub_split(
    ids: Iterable[ID_TYPE],
    source: ItemSource = "task_run",
    name: str = "test",
    eval_id: ID_TYPE = "eval1",
) -> ResolvedSplit:
    """A ResolvedSplit over items with the given ids, for tests that patch resolve_split.

    The API layer asks a split only for its length and its item keys, both of which are
    derived from the items' ids, so stub items are enough — and building a real dataset
    would put the test's subject behind a filter-matching setup that isn't what it is
    testing.
    """
    spec = TaskRun if source == "task_run" else EvalInput
    return ResolvedSplit(
        name=name,
        source=source,
        items=[Mock(spec=spec, id=id) for id in ids],
        eval_id=eval_id,
    )


def patch_resolve_split(**by_split_name: ResolvedSplit | None):
    """Patch eval_api's resolve_split to answer per split name, None for anything else."""
    return patch(
        "app.desktop.studio_server.eval_api.resolve_split",
        side_effect=lambda task, eval, split: by_split_name.get(split),
    )


def patch_resolve_split_by_ref(items_by_ref: Dict[Tuple[ItemSource, str], set]):
    """Patch resolve_split to answer from each eval's own `splits`, per (source, filter).

    Keyed on the pair that actually determines an item set, so a test can give the same
    filter id different items in each store — which is the case source-aware caching
    exists for.
    """

    def resolve(task, eval, split_name):
        ref = eval.splits.get(split_name)
        if ref is None:
            return None
        return stub_split(
            items_by_ref.get((ref.source, ref.filter_id), set()),
            source=ref.source,
            name=split_name,
            eval_id=eval.id,
        )

    return patch(
        "app.desktop.studio_server.eval_api.resolve_split", side_effect=resolve
    )


class _CollectingResponses:
    """Stands in for the parent->child responses queue; the payload is not asserted."""

    def __init__(self):
        self.puts: list[dict] = []

    def put(self, msg: dict) -> None:
        self.puts.append(msg)


class _FakeLlmTool:
    """Minimal KilnToolInterface stand-in for the `llm` built-in."""

    async def name(self) -> str:
        return "llm"

    async def toolcall_definition(self):
        return {
            "type": "function",
            "function": {
                "name": "llm",
                "description": "Call a model",
                "parameters": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                },
            },
        }

    async def run(self, context=None, **kwargs):
        return ToolCallResult(output="a judgement")


@pytest.fixture
def app():
    app = FastAPI()
    connect_custom_errors(app)
    connect_evals_api(app)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def mock_task(tmp_path):
    project = Project(
        id="project1",
        name="Test Project",
        path=tmp_path / "project.kiln",
    )
    project.save_to_file()
    task = Task(
        id="task1",
        name="Test Task",
        description="Test Description",
        instruction="Test Instructions",
        path=tmp_path / "task.kiln",
        requirements=[
            TaskRequirement(
                name="score1",
                description="desc1",
                instruction="inst1",
                priority=Priority.p1,
                type=TaskOutputRatingType.five_star,
            ),
        ],
        parent=project,
    )
    task.save_to_file()
    return task


@pytest.fixture
def mock_eval(mock_task):
    eval = Eval(
        id="eval1",
        name="Test Eval",
        description="Test Description",
        template=EvalTemplateId.bias,
        output_scores=[
            EvalOutputScore(
                name="score1", instruction="desc1", type=TaskOutputRatingType.five_star
            ),
            EvalOutputScore(
                name="overall_rating",
                instruction="desc2",
                type=TaskOutputRatingType.five_star,
            ),
        ],
        eval_set_filter_id="tag::eval_set",
        eval_configs_filter_id="tag::golden",
        parent=mock_task,
    )
    eval.save_to_file()
    return eval


@pytest.fixture
def mock_eval_config(mock_eval):
    eval_config = EvalConfig(
        id="eval_config1",
        name="Test Eval Config",
        config_type=EvalConfigType.g_eval,
        properties={"eval_steps": ["step1", "step2"]},
        parent=mock_eval,
        model_name="gpt-4",
        model_provider="openai",
    )
    eval_config.save_to_file()
    return eval_config


@pytest.fixture
def mock_run_config(mock_task):
    run_config = TaskRunConfig(
        parent=mock_task,
        id="run_config1",
        name="Test Run Config",
        description="Test Description",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_chain_of_thought_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
    )
    run_config.save_to_file()
    return run_config


@pytest.fixture
def data_source():
    return DataSource(
        type=DataSourceType.synthetic,
        properties={
            "model_name": "gpt-4",
            "model_provider": "openai",
            "adapter_name": "test_adapter",
        },
    )


@pytest.fixture
def mock_task_from_id(mock_task):
    with patch("app.desktop.studio_server.eval_api.task_from_id") as mock:
        mock.return_value = mock_task
        yield mock


def test_get_evals_success(client, mock_task, mock_task_from_id, mock_eval):
    mock_task_from_id.return_value = mock_task

    response = client.get("/api/projects/project1/tasks/task1/evals")

    assert response.status_code == 200
    result = response.json()
    assert result["load_error_count"] == 0
    assert len(result["evals"]) == 1
    assert result["evals"][0]["id"] == "eval1"
    assert result["evals"][0]["name"] == "Test Eval"
    mock_task_from_id.assert_called_once_with("project1", "task1")


def test_get_evals_logs_each_load_error(
    client, mock_task, mock_task_from_id, mock_eval, caplog
):
    """The response only counts unreadable eval files; the log must name each one, or a
    permanently corrupt file is indistinguishable from a version mismatch."""
    mock_task_from_id.return_value = mock_task
    assert mock_eval.path is not None
    corrupt_dir = mock_eval.path.parent.parent / "corrupt_eval"
    corrupt_dir.mkdir()
    corrupt_file = corrupt_dir / "eval.kiln"
    corrupt_file.write_text('{"v": 1, ', encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="app.desktop.studio_server.eval_api"):
        response = client.get("/api/projects/project1/tasks/task1/evals")

    assert response.status_code == 200
    result = response.json()
    assert result["load_error_count"] == 1
    assert [e["id"] for e in result["evals"]] == [mock_eval.id]
    warning = next(
        r for r in caplog.records if "Failed to load eval file" in r.getMessage()
    )
    assert warning.levelno == logging.WARNING
    assert str(corrupt_file) in warning.getMessage()


def test_get_evals_partial_load(client, mock_task, mock_task_from_id, mock_eval):
    """Evals this build can't parse are counted, and the readable ones still load."""
    mock_task_from_id.return_value = mock_task

    readable = Eval(
        id="eval2",
        name="Readable Eval",
        output_scores=[
            EvalOutputScore(
                name="score1", instruction="desc1", type=TaskOutputRatingType.five_star
            )
        ],
        eval_set_filter_id="tag::eval_set",
        eval_configs_filter_id="tag::golden",
        parent=mock_task,
    )
    readable.save_to_file()

    # Two evals written by a hypothetical newer Kiln: this build refuses to load them.
    for unreadable_id in ["eval3", "eval4"]:
        unreadable_dir = mock_task.path.parent / "evals" / unreadable_id
        unreadable_dir.mkdir(parents=True)
        (unreadable_dir / Eval.base_filename()).write_text(
            json.dumps(
                {
                    "v": mock_eval.max_schema_version() + 1,
                    "id": unreadable_id,
                    "name": "Future Eval",
                    "model_type": "eval",
                    "output_scores": [],
                    "eval_set_filter_id": "tag::eval_set",
                    "eval_configs_filter_id": "tag::golden",
                }
            )
        )

    response = client.get("/api/projects/project1/tasks/task1/evals")

    assert response.status_code == 200
    result = response.json()
    assert result["load_error_count"] == 2
    assert {e["id"] for e in result["evals"]} == {"eval1", "eval2"}


def test_get_eval_success(client, mock_task, mock_task_from_id, mock_eval):
    mock_task_from_id.return_value = mock_task

    response = client.get("/api/projects/project1/tasks/task1/evals/eval1")

    assert response.status_code == 200
    result = response.json()
    assert result["id"] == "eval1"
    assert result["name"] == "Test Eval"
    mock_task_from_id.assert_called_once_with("project1", "task1")


def test_get_eval_not_found(client, mock_task, mock_task_from_id):
    mock_task_from_id.return_value = mock_task

    response = client.get("/api/projects/project1/tasks/task1/evals/non_existent")

    assert response.status_code == 404
    assert response.json()["message"] == "Eval not found. ID: non_existent"


@pytest.fixture
def valid_evaluator_request():
    return CreateEvaluatorRequest(
        name="Test Evaluator",
        description="Test Description",
        template=None,
        output_scores=[
            EvalOutputScore(name="score1", type=TaskOutputRatingType.five_star),
        ],
        eval_set_filter_id="tag::eval_set",
        eval_configs_filter_id="tag::golden",
        template_properties={"test_property": "test_value", "numeric_property": 42},
        evaluation_data_type=EvalDataType.final_answer,
    )


@pytest.fixture
def valid_eval_config_request():
    return CreateEvalConfigRequest(
        name="Test Eval Config",
        type=EvalConfigType.g_eval,
        properties={"eval_steps": ["step1", "step2"]},
        model_name="gpt-4",
        provider=ModelProviderName.openai,
    )


@pytest.mark.asyncio
async def test_create_evaluator(
    client, mock_task_from_id, valid_evaluator_request, mock_task
):
    mock_task_from_id.return_value = mock_task

    response = client.post(
        "/api/projects/project1/tasks/task1/create_evaluator",
        json=valid_evaluator_request.model_dump(),
    )

    assert response.status_code == 200
    result = response.json()
    assert result["name"] == valid_evaluator_request.name
    assert result["description"] == valid_evaluator_request.description
    assert result["template_properties"] == valid_evaluator_request.template_properties

    # Verify the eval was created with the correct template_properties on disk
    saved_eval = mock_task.evals()[0]
    assert saved_eval.template == valid_evaluator_request.template
    assert saved_eval.name == valid_evaluator_request.name
    assert saved_eval.description == valid_evaluator_request.description
    assert saved_eval.output_scores == valid_evaluator_request.output_scores
    assert saved_eval.splits["test"] == TaskRunSplit(
        filter_id=valid_evaluator_request.eval_set_filter_id
    )
    assert (
        saved_eval.eval_configs_filter_id
        == valid_evaluator_request.eval_configs_filter_id
    )
    assert saved_eval.template_properties == valid_evaluator_request.template_properties
    assert saved_eval.template_properties is not None
    assert saved_eval.template_properties["test_property"] == "test_value"
    assert saved_eval.template_properties["numeric_property"] == 42


@pytest.mark.asyncio
async def test_create_task_run_config_with_freezing(
    client, mock_task_from_id, mock_task
):
    mock_task_from_id.return_value = mock_task

    with (
        patch(
            "app.desktop.studio_server.eval_api.generate_memorable_name"
        ) as mock_generate_memorable_name,
    ):
        mock_generate_memorable_name.return_value = "Custom Name"

        response = client.post(
            "/api/projects/project1/tasks/task1/run_configs",
            json={
                "name": "Test Task Run Config",
                "description": "Test Description",
                "run_config_properties": {
                    "model_name": "gpt-4o",
                    "model_provider_name": "openai",
                    "prompt_id": "simple_chain_of_thought_prompt_builder",
                    "temperature": 0.5,
                    "structured_output_mode": "json_schema",
                },
                # top_p not included, should get default 1.0
            },
        )

    assert response.status_code == 200
    result = response.json()
    assert result["name"] == "Test Task Run Config"
    assert result["description"] == "Test Description"
    assert result["run_config_properties"]["model_name"] == "gpt-4o"
    assert result["run_config_properties"]["model_provider_name"] == "openai"
    assert (
        result["run_config_properties"]["prompt_id"]
        == "task_run_config::project1::task1::" + result["id"]
    )
    # Check temperature is set to custom value 0.5
    assert result["run_config_properties"]["temperature"] == 0.5
    # Check top_p gets default value 1.0 when not specified
    assert result["run_config_properties"]["top_p"] == 1.0
    assert result["prompt"]["name"] == "Custom Name - Chain of Thought"
    assert (
        result["prompt"]["description"]
        == "Frozen copy of prompt 'simple_chain_of_thought_prompt_builder'."
    )
    # Fetch it from API
    fetch_response = client.get("/api/projects/project1/tasks/task1/run_configs")
    assert fetch_response.status_code == 200
    configs = fetch_response.json()
    assert len(configs) == 1
    assert configs[0]["id"] == result["id"]
    assert configs[0]["name"] == result["name"]
    # Verify temperature and top_p persist on disk
    assert configs[0]["run_config_properties"]["temperature"] == 0.5
    assert configs[0]["run_config_properties"]["top_p"] == 1.0
    assert configs[0]["prompt"]["name"] == "Custom Name - Chain of Thought"
    assert configs[0]["prompt"]["description"] == (
        "Frozen copy of prompt 'simple_chain_of_thought_prompt_builder'."
    )
    assert configs[0]["run_config_properties"]["prompt_id"] == (
        "task_run_config::project1::task1::" + result["id"]
    )


@pytest.mark.asyncio
async def test_create_task_run_config_without_freezing(
    client, mock_task_from_id, mock_task
):
    mock_task_from_id.return_value = mock_task

    with (
        patch(
            "app.desktop.studio_server.eval_api.generate_memorable_name"
        ) as mock_generate_memorable_name,
    ):
        mock_generate_memorable_name.return_value = "Custom Name"

        response = client.post(
            "/api/projects/project1/tasks/task1/run_configs",
            json={
                "name": "Test Task Run Config",
                "description": "Test Description",
                "run_config_properties": {
                    "model_name": "gpt-4o",
                    "model_provider_name": "openai",
                    "prompt_id": "id::prompt_123",
                    "structured_output_mode": "json_schema",
                },
            },
        )

    assert response.status_code == 200
    result = response.json()
    assert result["name"] == "Test Task Run Config"
    assert result["description"] == "Test Description"
    assert result["run_config_properties"]["model_name"] == "gpt-4o"
    assert result["run_config_properties"]["model_provider_name"] == "openai"
    assert result["run_config_properties"]["prompt_id"] == "id::prompt_123"
    assert result["prompt"] is None


@pytest.mark.asyncio
async def test_create_task_run_config_reuses_existing_frozen_prompt(
    client, mock_task_from_id, mock_task
):
    mock_task_from_id.return_value = mock_task

    def create(name: str):
        return client.post(
            "/api/projects/project1/tasks/task1/run_configs",
            json={
                "name": name,
                "run_config_properties": {
                    "model_name": "gpt-4o",
                    "model_provider_name": "openai",
                    "prompt_id": "simple_chain_of_thought_prompt_builder",
                    "structured_output_mode": "json_schema",
                },
            },
        )

    first = create("First")
    assert first.status_code == 200
    first_result = first.json()
    # First config freezes a new prompt pointing at itself
    assert first_result["prompt"] is not None
    assert first_result["run_config_properties"]["prompt_id"] == (
        "task_run_config::project1::task1::" + first_result["id"]
    )

    second = create("Second")
    assert second.status_code == 200
    second_result = second.json()
    # Second config has identical content, so it reuses the first's frozen prompt
    # instead of creating a duplicate
    assert second_result["id"] != first_result["id"]
    assert second_result["prompt"] is None
    assert second_result["run_config_properties"]["prompt_id"] == (
        "task_run_config::project1::task1::" + first_result["id"]
    )

    # Only the first config contributes a frozen prompt to the task
    frozen_prompts = [rc.prompt for rc in mock_task.run_configs() if rc.prompt]
    assert len(frozen_prompts) == 1


def test_reusable_frozen_prompt_id_no_match(mock_task):
    assert (
        reusable_frozen_prompt_id(mock_task, "project1", "some prompt text", None)
        is None
    )


def test_reusable_frozen_prompt_id_picks_most_recent(mock_task):
    # Legacy case: multiple run configs with identical frozen content. The most
    # recently created one should be reused.
    older = TaskRunConfig(
        parent=mock_task,
        id="older",
        name="Older",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id="task_run_config::project1::task1::older",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        prompt=BasePrompt(name="Older Frozen", prompt="shared body"),
    )
    older.save_to_file()
    newer = TaskRunConfig(
        parent=mock_task,
        id="newer",
        name="Newer",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id="task_run_config::project1::task1::newer",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        prompt=BasePrompt(name="Newer Frozen", prompt="shared body"),
    )
    newer.save_to_file()

    # Force a deterministic ordering of created_at
    older.created_at = older.created_at.replace(year=2020)
    newer.created_at = newer.created_at.replace(year=2024)
    older.save_to_file()
    newer.save_to_file()

    result = reusable_frozen_prompt_id(mock_task, "project1", "shared body", None)
    assert result == "task_run_config::project1::task1::newer"

    # Content that doesn't match returns None
    assert (
        reusable_frozen_prompt_id(mock_task, "project1", "different body", None) is None
    )
    # cot mismatch is treated as a different prompt
    assert (
        reusable_frozen_prompt_id(mock_task, "project1", "shared body", "cot") is None
    )


def test_reusable_frozen_prompt_id_normalizes_empty_cot(mock_task):
    # An empty-string cot and a missing cot should be treated as equivalent.
    config = TaskRunConfig(
        parent=mock_task,
        id="rc_empty_cot",
        name="Empty CoT",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id="task_run_config::project1::task1::rc_empty_cot",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        prompt=BasePrompt(
            name="Empty CoT Frozen",
            prompt="body",
            chain_of_thought_instructions="",
        ),
    )
    config.save_to_file()

    assert reusable_frozen_prompt_id(mock_task, "project1", "body", None) == (
        "task_run_config::project1::task1::rc_empty_cot"
    )
    assert reusable_frozen_prompt_id(mock_task, "project1", "body", "") == (
        "task_run_config::project1::task1::rc_empty_cot"
    )


@pytest.mark.asyncio
async def test_create_eval_config(
    client, mock_task_from_id, valid_eval_config_request, mock_eval, mock_task
):
    mock_task_from_id.return_value = mock_task

    with (
        patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id,
    ):
        mock_eval_from_id.return_value = mock_eval

        response = client.post(
            "/api/projects/project1/tasks/task1/evals/eval1/create_eval_config",
            json=valid_eval_config_request.model_dump(),
        )

    assert response.status_code == 200
    result = response.json()
    assert result["name"] == valid_eval_config_request.name
    assert result["config_type"] == valid_eval_config_request.type
    assert result["properties"] == valid_eval_config_request.properties
    assert result["model_name"] == valid_eval_config_request.model_name
    assert result["model_provider"] == valid_eval_config_request.provider

    # Fetch disk
    assert len(mock_eval.configs()) == 1
    config = mock_eval.configs()[0]
    assert config.config_type == valid_eval_config_request.type
    assert config.properties == valid_eval_config_request.properties
    assert config.model_name == valid_eval_config_request.model_name
    assert config.model_provider == valid_eval_config_request.provider
    assert config.properties["eval_steps"][0] == "step1"
    assert config.properties["eval_steps"][1] == "step2"


@pytest.mark.asyncio
async def test_create_eval_config_missing_model_for_llm_type(
    client, mock_task_from_id, mock_eval, mock_task
):
    mock_task_from_id.return_value = mock_task

    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = mock_eval

        response = client.post(
            "/api/projects/project1/tasks/task1/evals/eval1/create_eval_config",
            json={
                "name": "Bad Config",
                "type": "g_eval",
                "properties": {"eval_steps": ["step1"]},
                "model_name": None,
                "provider": None,
            },
        )

    assert response.status_code == 400
    assert "model_name and provider are required" in response.json()["message"]


@pytest.mark.asyncio
async def test_create_eval_config_invalid_properties(
    client, mock_task_from_id, mock_eval, mock_task
):
    mock_task_from_id.return_value = mock_task

    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = mock_eval

        response = client.post(
            "/api/projects/project1/tasks/task1/evals/eval1/create_eval_config",
            json={
                "name": "Bad Config",
                "type": "g_eval",
                "properties": {"not_a_valid_field": True},
                "model_name": "gpt-4",
                "provider": "openai",
            },
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_create_eval_config_invalid_v2_properties(
    client, mock_task_from_id, mock_v2_eval, mock_task
):
    mock_task_from_id.return_value = mock_task

    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = mock_v2_eval

        response = client.post(
            "/api/projects/project1/tasks/task1/evals/eval_v2/create_eval_config",
            json={
                "name": "Bad V2 Config",
                "type": "v2",
                "properties": {"type": "not_a_real_type"},
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert "Invalid properties for eval config type" in body["message"]
    assert "v2" in body["message"]


CODE_EVAL_PROPERTIES = {
    "type": "code_eval",
    "code": "def score(output, **kwargs):\n    return {'accuracy': 1.0}\n",
}


@pytest.mark.asyncio
async def test_create_eval_config_code_eval_untrusted_403(
    client, mock_task_from_id, mock_v2_eval, mock_task
):
    mock_task_from_id.return_value = mock_task

    with (
        patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id,
        patch("app.desktop.studio_server.eval_api.project_from_id") as mock_proj,
        patch(
            "app.desktop.studio_server.eval_api.has_add_code_trust",
            return_value=False,
        ),
    ):
        mock_eval_from_id.return_value = mock_v2_eval
        mock_proj.return_value = Mock()

        response = client.post(
            "/api/projects/project1/tasks/task1/evals/eval_v2/create_eval_config",
            json={
                "name": "Code Eval Config",
                "type": "v2",
                "properties": CODE_EVAL_PROPERTIES,
            },
        )

    assert response.status_code == 403
    assert "not trusted" in response.json()["message"].lower()
    # Nothing should have been persisted.
    assert len(mock_v2_eval.configs()) == 0


@pytest.mark.asyncio
async def test_create_eval_config_code_eval_trusted_succeeds(
    client, mock_task_from_id, mock_v2_eval, mock_task
):
    mock_task_from_id.return_value = mock_task

    with (
        patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id,
        patch("app.desktop.studio_server.eval_api.project_from_id") as mock_proj,
        patch(
            "app.desktop.studio_server.eval_api.has_add_code_trust",
            return_value=True,
        ),
    ):
        mock_eval_from_id.return_value = mock_v2_eval
        mock_proj.return_value = Mock()

        response = client.post(
            "/api/projects/project1/tasks/task1/evals/eval_v2/create_eval_config",
            json={
                "name": "Code Eval Config",
                "type": "v2",
                "properties": CODE_EVAL_PROPERTIES,
            },
        )

    assert response.status_code == 200
    result = response.json()
    assert result["properties"]["type"] == "code_eval"
    # Persisted to disk.
    assert len(mock_v2_eval.configs()) == 1


def test_get_eval_config(
    client, mock_task_from_id, mock_eval, mock_task, mock_eval_config
):
    mock_task_from_id.return_value = mock_task

    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = mock_eval
        response = client.get(
            "/api/projects/project1/tasks/task1/evals/eval1/eval_config/eval_config1"
        )

    assert response.status_code == 200
    config = response.json()
    assert isinstance(config, dict)

    assert config["config_type"] == mock_eval_config.config_type
    assert config["properties"] == mock_eval_config.properties
    assert config["model_name"] == mock_eval_config.model_name
    assert config["model_provider"] == mock_eval_config.model_provider

    mock_eval_from_id.assert_called_once_with("project1", "task1", "eval1")


def test_get_eval_configs(
    client, mock_task_from_id, mock_eval, mock_task, mock_eval_config
):
    mock_task_from_id.return_value = mock_task

    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = mock_eval
        response = client.get(
            "/api/projects/project1/tasks/task1/evals/eval1/eval_configs"
        )

    assert response.status_code == 200
    configs = response.json()
    assert isinstance(configs, list)
    assert len(configs) == 1

    config = configs[0]
    assert config["config_type"] == mock_eval_config.config_type
    assert config["properties"] == mock_eval_config.properties
    assert config["model_name"] == mock_eval_config.model_name
    assert config["model_provider"] == mock_eval_config.model_provider

    mock_eval_from_id.assert_called_once_with("project1", "task1", "eval1")


@pytest.mark.asyncio
async def test_run_eval_config(
    client,
    mock_task_from_id,
    mock_task,
    mock_eval,
    mock_eval_config,
    mock_run_config,
    data_source,
):
    mock_task_from_id.return_value = mock_task

    in_test_split = TaskRun(
        parent=mock_task,
        input="in the test split",
        input_source=data_source,
        output=TaskOutput(output="out"),
        tags=["eval_set"],
    )
    in_test_split.save_to_file()
    TaskRun(
        parent=mock_task,
        input="golden only",
        input_source=data_source,
        output=TaskOutput(output="out"),
        tags=["golden"],
    ).save_to_file()

    # Mock progress updates
    progress_updates = [
        Mock(complete=1, total=3, errors=0),
        Mock(complete=2, total=3, errors=0),
        Mock(complete=3, total=3, errors=0),
    ]

    # Create async generator for mock progress
    async def mock_run():
        for progress in progress_updates:
            yield progress

    with (
        patch(
            "app.desktop.studio_server.eval_api.task_run_config_from_id"
        ) as mock_run_config_from_id,
        patch("app.desktop.studio_server.eval_api.EvalRunner") as MockEvalRunner,
    ):
        mock_run_config_from_id.return_value = mock_run_config
        mock_eval_runner = Mock()
        mock_eval_runner.run.return_value = mock_run()
        MockEvalRunner.return_value = mock_eval_runner

        # Make request with specific run_config_ids
        response = client.get(
            "/api/projects/project1/tasks/task1/evals/eval1/eval_config/eval_config1/run_comparison",
            params={"run_config_ids": ["run_config1", "run_config2"]},
        )

        assert response.status_code == 200

        # The runner is handed the eval's resolved test split, not a filter it re-reads.
        split = MockEvalRunner.call_args.kwargs["split"]
        assert split.name == "test"
        assert split.source == "task_run"
        assert [item.id for item in split.items] == [in_test_split.id]

        # Parse SSE messages
        messages = [msg for msg in response.iter_lines() if msg]

        # Should have 4 messages: 3 progress updates and 1 complete
        assert len(messages) == 4

        # Check progress messages
        for i, msg in enumerate(messages[:-1]):
            assert msg.startswith("data: ")
            data = json.loads(msg.split("data: ")[1])
            assert data["progress"] == i + 1
            assert data["total"] == 3
            assert data["errors"] == 0

        # Check complete message
        assert messages[-1] == "data: complete"


@pytest.mark.asyncio
async def test_run_eval_config_no_run_configs_error(
    client, mock_task_from_id, mock_task, mock_eval, mock_eval_config
):
    mock_task_from_id.return_value = mock_task

    with patch(
        "app.desktop.studio_server.eval_api.eval_config_from_id"
    ) as mock_eval_config_from_id:
        mock_eval_config_from_id.return_value = mock_eval_config

        # Make request with no run_config_ids and all_run_configs=False
        response = client.get(
            "/api/projects/project1/tasks/task1/evals/eval1/eval_config/eval_config1/run_comparison"
        )

        assert response.status_code == 400
        assert (
            response.json()["message"]
            == "No run config ids provided. At least one run config id is required."
        )


class TestResolvedSplitOr422:
    """The run endpoint's test split always exists (validate_splits enforces it at load),
    so the failure path is exercised here rather than through that endpoint. Phase 6's
    endpoints take a caller-supplied split name, where an absent split is reachable."""

    def test_returns_the_splits_items(self, mock_task, mock_eval, data_source):
        in_split = TaskRun(
            parent=mock_task,
            input="in the eval set",
            input_source=data_source,
            output=TaskOutput(output="out"),
            tags=["eval_set"],
        )
        in_split.save_to_file()
        TaskRun(
            parent=mock_task,
            input="out of the eval set",
            input_source=data_source,
            output=TaskOutput(output="out"),
            tags=["other"],
        ).save_to_file()

        resolved = resolved_split_or_422(mock_task, mock_eval, "test")

        assert resolved.name == "test"
        assert [item.id for item in resolved.items] == [in_split.id]

    def test_422s_naming_the_split_and_the_eval(self, mock_task, mock_eval):
        with pytest.raises(HTTPException) as exc_info:
            resolved_split_or_422(mock_task, mock_eval, "train")

        assert exc_info.value.status_code == 422
        assert "no 'train' split" in exc_info.value.detail
        assert mock_eval.id in exc_info.value.detail


# Asserted in full rather than by substring: this project has shipped user-facing strings
# that were wrong because nothing read them. 'V2 Test Eval' is mock_v2_eval's name.
V1_JUDGE_ON_EVAL_INPUTS_MESSAGE = (
    "Eval 'V2 Test Eval' uses our new eval dataset format, which the '{judge_type}' "
    "judge type can't score. Choose a judge type that supports the new format."
)


class TestLegacyJudgeNeedsDatasetRuns:
    """V1 judges (g_eval, llm_as_judge) score a stored task output, so they need TaskRuns.

    Without these refusals the mismatch only surfaces as `_run_legacy_job`'s ValueError,
    once per job, which AsyncJobRunner turns into a bare `errors: N` on an HTTP 200 SSE
    stream — no status code, no reason, nothing the UI can render.
    """

    CREATE_URL = "/api/projects/project1/tasks/task1/evals/eval_v2/create_eval_config"

    def _v1_request(self, judge_type: str) -> dict:
        return {
            "name": "V1 Judge",
            "type": judge_type,
            "properties": {"eval_steps": ["step1"]},
            "model_name": "gpt-4",
            "provider": "openai",
        }

    @pytest.mark.parametrize("judge_type", ["g_eval", "llm_as_judge"])
    def test_create_refuses_v1_judge_on_eval_input_backed_eval(
        self, client, mock_task_from_id, mock_task, mock_v2_eval, judge_type
    ):
        with patch(
            "app.desktop.studio_server.eval_api.eval_from_id"
        ) as mock_eval_from_id:
            mock_eval_from_id.return_value = mock_v2_eval
            response = client.post(self.CREATE_URL, json=self._v1_request(judge_type))

        assert response.status_code == 400
        assert response.json()["message"] == V1_JUDGE_ON_EVAL_INPUTS_MESSAGE.format(
            judge_type=judge_type
        )
        # Refused before anything was written.
        assert len(mock_v2_eval.configs()) == 0

    def test_create_allows_v2_judge_on_eval_input_backed_eval(
        self, client, mock_task_from_id, mock_task, mock_v2_eval
    ):
        with patch(
            "app.desktop.studio_server.eval_api.eval_from_id"
        ) as mock_eval_from_id:
            mock_eval_from_id.return_value = mock_v2_eval
            response = client.post(
                self.CREATE_URL,
                json={
                    "name": "V2 Judge",
                    "type": "v2",
                    "properties": {"type": "exact_match", "expected_value": "hello"},
                },
            )

        assert response.status_code == 200
        assert len(mock_v2_eval.configs()) == 1

    @pytest.mark.parametrize("judge_type", ["g_eval", "llm_as_judge"])
    def test_create_allows_v1_judge_on_task_run_backed_eval(
        self, client, mock_task_from_id, mock_task, mock_eval, judge_type
    ):
        assert isinstance(mock_eval.splits["test"], TaskRunSplit)

        with patch(
            "app.desktop.studio_server.eval_api.eval_from_id"
        ) as mock_eval_from_id:
            mock_eval_from_id.return_value = mock_eval
            response = client.post(
                "/api/projects/project1/tasks/task1/evals/eval1/create_eval_config",
                json=self._v1_request(judge_type),
            )

        assert response.status_code == 200
        assert len(mock_eval.configs()) == 1

    def _run_comparison(self, client, eval_id: str, eval_config_id: str):
        return client.get(
            f"/api/projects/project1/tasks/task1/evals/{eval_id}"
            f"/eval_config/{eval_config_id}/run_comparison",
            params={"run_config_ids": ["run_config1"]},
        )

    @pytest.mark.parametrize(
        "judge_type", [EvalConfigType.g_eval, EvalConfigType.llm_as_judge]
    )
    def test_run_refuses_a_config_created_before_the_creation_guard(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_v2_eval,
        mock_run_config,
        judge_type,
    ):
        """The creation guard can't reach an eval config already on disk."""
        EvalConfig(
            id="legacy_config",
            name="Legacy Judge",
            config_type=judge_type,
            properties={"eval_steps": ["step1"]},
            model_name="gpt-4",
            model_provider="openai",
            parent=mock_v2_eval,
        ).save_to_file()

        with (
            patch(
                "app.desktop.studio_server.eval_api.task_run_config_from_id"
            ) as mock_run_config_from_id,
            patch("app.desktop.studio_server.eval_api.EvalRunner") as MockEvalRunner,
        ):
            mock_run_config_from_id.return_value = mock_run_config
            response = self._run_comparison(client, "eval_v2", "legacy_config")

        assert response.status_code == 400
        assert response.json()["message"] == V1_JUDGE_ON_EVAL_INPUTS_MESSAGE.format(
            judge_type=judge_type.value
        )
        # Refused before the runner exists, so before the StreamingResponse starts.
        MockEvalRunner.assert_not_called()

    def test_run_allows_a_v2_config_on_an_eval_input_backed_eval(
        self, client, mock_task_from_id, mock_task, mock_v2_eval, mock_run_config
    ):
        EvalConfig(
            id="v2_config",
            name="V2 Judge",
            config_type=EvalConfigType.v2,
            properties={"type": "exact_match", "expected_value": "hello"},
            parent=mock_v2_eval,
        ).save_to_file()

        async def no_progress():
            return
            yield  # pragma: no cover — makes this an async generator; return runs first

        with (
            patch(
                "app.desktop.studio_server.eval_api.task_run_config_from_id"
            ) as mock_run_config_from_id,
            patch("app.desktop.studio_server.eval_api.EvalRunner") as MockEvalRunner,
        ):
            mock_run_config_from_id.return_value = mock_run_config
            mock_eval_runner = Mock()
            mock_eval_runner.run.return_value = no_progress()
            MockEvalRunner.return_value = mock_eval_runner

            response = self._run_comparison(client, "eval_v2", "v2_config")

            assert response.status_code == 200
            assert MockEvalRunner.call_args.kwargs["split"].source == "eval_input"


@pytest.mark.asyncio
async def test_eval_config_from_id(
    client, mock_task_from_id, mock_task, mock_eval, mock_eval_config
):
    mock_task_from_id.return_value = mock_task

    eval_config = eval_config_from_id("project1", "task1", "eval1", "eval_config1")

    assert eval_config.id == "eval_config1"
    assert eval_config.name == "Test Eval Config"
    assert eval_config.config_type == EvalConfigType.g_eval
    assert eval_config.properties == {"eval_steps": ["step1", "step2"]}

    with pytest.raises(HTTPException, match=r"Eval config not found. ID: non_existent"):
        eval_config_from_id("project1", "task1", "eval1", "non_existent")


@pytest.mark.asyncio
async def test_task_run_config_from_id(
    client, mock_task_from_id, mock_task, mock_run_config
):
    mock_task_from_id.return_value = mock_task

    run_config = task_run_config_from_id("project1", "task1", "run_config1")

    assert run_config.id == "run_config1"
    assert run_config.name == "Test Run Config"
    assert run_config.description == "Test Description"

    with pytest.raises(
        HTTPException, match=r"Task run config not found. ID: non_existent"
    ):
        task_run_config_from_id("project1", "task1", "non_existent")


@pytest.mark.asyncio
async def test_task_run_config_from_id_finetune(mock_task_from_id, mock_task):
    mock_task_from_id.return_value = mock_task

    run_config_props = KilnAgentRunConfigProperties(
        model_name="gpt-4",
        model_provider_name=ModelProviderName.openai,
        prompt_id="simple_chain_of_thought_prompt_builder",
        structured_output_mode=StructuredOutputMode.json_schema,
    )

    mock_finetune = Finetune(
        id="ft_test",
        name="Test Finetune",
        description="Test finetune description",
        provider="openai",
        base_model_id="model1",
        dataset_split_id="split1",
        system_message="System message",
        latest_status=FineTuneStatusType.completed,
        run_config=run_config_props,
        fine_tune_model_id="ft_model_123",
        parent=mock_task,
    )

    with patch(
        "app.desktop.studio_server.eval_api.finetune_from_finetune_run_config_id"
    ) as mock_finetune_from_id:
        mock_finetune_from_id.return_value = mock_finetune

        run_config = task_run_config_from_id(
            "project1", "task1", "finetune_run_config::project1::task1::ft_test"
        )

        assert run_config.id == "finetune_run_config::project1::task1::ft_test"
        assert run_config.name == "Test Finetune"
        assert run_config.description == "Test finetune description"
        assert run_config.run_config_properties == run_config_props
        assert run_config.parent == mock_task


@pytest.mark.asyncio
async def test_get_all_run_configs(mock_task_from_id, mock_task):
    """Test that get_all_run_configs returns regular run configs and completed finetune run configs."""
    mock_task_from_id.return_value = mock_task

    run_config_props = KilnAgentRunConfigProperties(
        model_name="gpt-4",
        model_provider_name=ModelProviderName.openai,
        prompt_id="simple_chain_of_thought_prompt_builder",
        structured_output_mode=StructuredOutputMode.json_schema,
    )

    regular_run_config = TaskRunConfig(
        id="regular_run_config1",
        name="Regular Run Config",
        description="A regular run config",
        run_config_properties=run_config_props,
        parent=mock_task,
    )
    regular_run_config.save_to_file()

    completed_finetune = Finetune(
        id="ft_completed",
        name="Completed Finetune",
        provider="openai",
        base_model_id="model1",
        dataset_split_id="split1",
        system_message="System message",
        latest_status=FineTuneStatusType.completed,
        run_config=run_config_props,
        fine_tune_model_id="ft_model_123",
        parent=mock_task,
    )
    completed_finetune.save_to_file()

    incomplete_finetune = Finetune(
        id="ft_incomplete",
        name="Incomplete Finetune",
        provider="openai",
        base_model_id="model2",
        dataset_split_id="split2",
        system_message="System message",
        latest_status=FineTuneStatusType.running,
        run_config=run_config_props,
        fine_tune_model_id=None,
        parent=mock_task,
    )
    incomplete_finetune.save_to_file()

    configs = get_all_run_configs("project1", "task1")

    config_ids = [config.id for config in configs]
    assert "regular_run_config1" in config_ids
    assert "finetune_run_config::project1::task1::ft_completed" in config_ids
    assert "finetune_run_config::project1::task1::ft_incomplete" not in config_ids


def test_run_config_starred_default(mock_task):
    """Test that starred defaults to False on TaskRunConfig."""
    run_config = TaskRunConfig(
        parent=mock_task,
        name="Starred Test Config",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_chain_of_thought_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
    )
    assert run_config.starred is False


def test_run_config_starred_persists(mock_task):
    """Test that starred field persists through save and load."""
    run_config = TaskRunConfig(
        parent=mock_task,
        name="Starred Persist Config",
        starred=True,
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_chain_of_thought_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
    )
    run_config.save_to_file()
    assert run_config.starred is True

    loaded = TaskRunConfig.load_from_file(run_config.path)
    assert loaded.starred is True


def test_update_run_config_starred(client, mock_task_from_id, mock_run_config):
    """Test the PATCH endpoint to star a run config."""
    assert mock_run_config.starred is False

    response = client.patch(
        "/api/projects/project1/tasks/task1/run_configs/run_config1",
        json={"starred": True},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["starred"] is True

    loaded = TaskRunConfig.load_from_file(mock_run_config.path)
    assert loaded.starred is True


def test_update_run_config_unstar(client, mock_task_from_id, mock_run_config):
    """Test the PATCH endpoint to unstar a previously starred run config."""
    mock_run_config.starred = True
    mock_run_config.save_to_file()

    response = client.patch(
        "/api/projects/project1/tasks/task1/run_configs/run_config1",
        json={"starred": False},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["starred"] is False

    loaded = TaskRunConfig.load_from_file(mock_run_config.path)
    assert loaded.starred is False


def test_update_run_config_not_found(client, mock_task_from_id, mock_task):
    """Test the PATCH endpoint returns 404 for non-existent run config."""
    response = client.patch(
        "/api/projects/project1/tasks/task1/run_configs/non_existent",
        json={"starred": True},
    )
    assert response.status_code == 404


def test_update_run_config_no_path(client, mock_task_from_id, mock_task):
    """Test that updating a run config without a path (e.g. finetune) returns 400."""
    finetune_run_config = TaskRunConfig(
        id="finetune_run_config::project1::task1::ft1",
        name="Finetune Config",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_chain_of_thought_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
        parent=mock_task,
    )

    with patch(
        "app.desktop.studio_server.eval_api.task_run_config_from_id"
    ) as mock_from_id:
        mock_from_id.return_value = finetune_run_config
        response = client.patch(
            "/api/projects/project1/tasks/task1/run_configs/finetune_run_config::project1::task1::ft1",
            json={"starred": True},
        )
    assert response.status_code == 400


def test_update_run_config_prompt_name(client, mock_task_from_id, mock_run_config):
    """Test the PATCH endpoint to update a frozen prompt's name."""
    mock_run_config.prompt = BasePrompt(
        name="Original Name",
        prompt="This is a frozen prompt",
    )
    mock_run_config.save_to_file()

    response = client.patch(
        "/api/projects/project1/tasks/task1/run_configs/run_config1",
        json={"prompt_name": "Updated Name"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["prompt"]["name"] == "Updated Name"

    loaded = TaskRunConfig.load_from_file(mock_run_config.path)
    assert loaded.prompt is not None
    assert loaded.prompt.name == "Updated Name"


def test_update_run_config_prompt_name_no_prompt(
    client, mock_task_from_id, mock_run_config
):
    """Test that updating prompt_name when no frozen prompt exists returns 400."""
    assert mock_run_config.prompt is None

    response = client.patch(
        "/api/projects/project1/tasks/task1/run_configs/run_config1",
        json={"prompt_name": "New Name"},
    )
    assert response.status_code == 400
    assert "no frozen prompt" in response.json()["message"].lower()


@pytest.fixture
def mock_eval_for_score_summary():
    eval = Mock(spec=Eval)
    eval.output_scores = [
        EvalOutputScore(
            name="accuracy",
            instruction="Test accuracy",
            type=TaskOutputRatingType.pass_fail,
        ),
        EvalOutputScore(
            name="relevance",
            instruction="Test relevance",
            type=TaskOutputRatingType.pass_fail,
        ),
    ]
    eval.eval_set_filter_id = "tag::eval_set"
    return eval


@pytest.fixture
def mock_eval_config_for_score_summary():
    config = Mock(spec=EvalConfig)

    scores: List[Tuple[str, str, Dict[str, float]]] = [
        # Run 1 - normal
        ("run1", "dataset_id_1", {"accuracy": 0.8, "relevance": 0.9}),
        ("run1", "dataset_id_2", {"accuracy": 0.6, "relevance": 0.7}),
        # Run 2 - only 1 score, should be 0.5 complete
        ("run2", "dataset_id_1", {"accuracy": 0.9, "relevance": 0.85}),
        # Run 3 - no valid scores, 0.0 complete
        ("run3", "dataset_id_1", {"other": 0.5}),
        # Run 4 - Partial incomplete doesn't divide by zero, still 0.0 complete
        ("run4", "dataset_id_1", {"accuracy": 0.5}),
        # Run 5 - duplicate dataset_id not double counted, item not in dataset filter ignored
        ("run5", "dataset_id_1", {"accuracy": 0.8, "relevance": 0.9}),
        ("run5", "dataset_id_1", {"accuracy": 0.8, "relevance": 0.9}),
        ("run5", "dataset_id_2", {"accuracy": 0.6, "relevance": 0.7}),
        ("run5", "not_in_filter", {"accuracy": 0.1, "relevance": 0.1}),
    ]
    runs = []

    id = 0
    for run_id, dataset_id, score in scores:
        id += 1
        runs.append(
            EvalRun(
                task_run_config_id=run_id,
                scores=score,
                input="input",
                output="output",
                dataset_id=dataset_id,
            )
        )

    config.runs.return_value = runs
    return config


@pytest.mark.asyncio
async def test_get_eval_config_score_summary(
    client, mock_eval_for_score_summary, mock_eval_config_for_score_summary
):
    with (
        patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id,
        patch_resolve_split(
            test=stub_split({"dataset_id_1", "dataset_id_2"})
        ) as mock_resolve_split,
        patch(
            "app.desktop.studio_server.eval_api.eval_config_from_id"
        ) as mock_eval_config_from_id,
        patch("app.desktop.studio_server.eval_api.task_from_id") as mock_task_from_id,
    ):
        mock_eval_from_id.return_value = mock_eval_for_score_summary
        mock_eval_config_from_id.return_value = mock_eval_config_for_score_summary

        mock_task = Mock(spec=Task)
        mock_task.run_configs.return_value = [
            Mock(spec=TaskRunConfig, id="run1"),
            Mock(spec=TaskRunConfig, id="run2"),
            Mock(spec=TaskRunConfig, id="run3"),
            Mock(spec=TaskRunConfig, id="run4"),
            Mock(spec=TaskRunConfig, id="run5"),
        ]
        mock_task.finetunes.return_value = []
        mock_task.runs.return_value = []
        mock_task_from_id.return_value = mock_task

        response = client.get(
            "/api/projects/project1/tasks/task1/evals/eval1/eval_config/eval_config1/score_summary"
        )

        assert response.status_code == 200
        top_level_result = response.json()

        # Verify the structure of the response
        assert "results" in top_level_result
        results = top_level_result["results"]
        assert "run_config_percent_complete" in top_level_result
        run_config_percent_complete = top_level_result["run_config_percent_complete"]
        assert "dataset_size" in top_level_result
        assert top_level_result["dataset_size"] == 2
        # No runs in the task store, so no stored multi-turn conversations
        assert top_level_result["multi_turn_item_count"] == 0

        # Check average scores for run1
        assert results["run1"]["accuracy"]["mean_score"] == 0.7  # (0.8 + 0.6) / 2
        assert results["run1"]["accuracy"]["n_used"] == 2
        assert results["run1"]["accuracy"]["n_excluded"] == 0
        assert results["run1"]["relevance"]["mean_score"] == 0.8  # Only one valid score
        assert run_config_percent_complete["run1"] == 1.0

        # Check average scores for run2
        assert results["run2"]["accuracy"]["mean_score"] == 0.9
        assert results["run2"]["accuracy"]["n_used"] == 1
        assert results["run2"]["accuracy"]["n_excluded"] == 0
        assert results["run2"]["relevance"]["mean_score"] == 0.85
        assert run_config_percent_complete["run2"] == 0.5

        # run 3 has non valid scores
        assert results["run3"] == {}
        assert run_config_percent_complete["run3"] == 0.0

        # run 4 has no scores
        assert results["run4"]["accuracy"]["mean_score"] == 0.5
        assert results["run4"]["accuracy"]["n_used"] == 1
        assert results["run4"]["accuracy"]["n_excluded"] == 0
        assert "relevance" not in results["run4"]
        assert run_config_percent_complete["run4"] == 0.0

        # Check average scores for run5 - duplicate dataset_id not double counted
        assert results["run5"]["accuracy"]["mean_score"] == 0.7  # (0.8 + 0.6) / 2
        assert results["run5"]["accuracy"]["n_used"] == 2
        assert results["run5"]["accuracy"]["n_excluded"] == 0
        assert results["run5"]["relevance"]["mean_score"] == 0.8  # Only one valid score
        assert run_config_percent_complete["run5"] == 1.0

        # Verify the mocks were called correctly
        mock_eval_from_id.assert_called_once_with("project1", "task1", "eval1")
        mock_eval_config_from_id.assert_called_once_with(
            "project1", "task1", "eval1", "eval_config1"
        )
        mock_eval_config_for_score_summary.runs.assert_called_once_with(readonly=True)
        mock_resolve_split.assert_called_once_with(
            mock_task, mock_eval_for_score_summary, "test"
        )


class TestScoreSummaryFromValues:
    """Distribution fields on ScoreSummary. Percentiles are linearly
    interpolated (numpy.percentile default) — see score_summary_from_values."""

    def test_empty_is_all_none(self):
        # Every statistic must be None, never 0.0 — a 0 would flow into
        # downstream aggregation as if it were a real datum.
        summary = score_summary_from_values([], 3)
        assert summary.mean_score is None
        assert summary.min_score is None
        assert summary.p25_score is None
        assert summary.median_score is None
        assert summary.p75_score is None
        assert summary.p90_score is None
        assert summary.max_score is None
        assert summary.n_used == 0
        assert summary.n_excluded == 3

    def test_single_value(self):
        summary = score_summary_from_values([0.4], 0)
        assert summary.mean_score == pytest.approx(0.4)
        assert summary.min_score == pytest.approx(0.4)
        assert summary.median_score == pytest.approx(0.4)
        assert summary.p90_score == pytest.approx(0.4)
        assert summary.max_score == pytest.approx(0.4)
        assert summary.n_used == 1

    def test_odd_length_median_is_middle_value(self):
        summary = score_summary_from_values([1.0, 3.0, 2.0], 0)
        assert summary.median_score == pytest.approx(2.0)
        assert summary.min_score == pytest.approx(1.0)
        assert summary.max_score == pytest.approx(3.0)
        assert summary.n_used == 3

    def test_even_length_median_interpolates(self):
        # Interpolated midpoint (2.5), not the lower middle value (2.0).
        summary = score_summary_from_values([1.0, 2.0, 3.0, 4.0], 0)
        assert summary.median_score == pytest.approx(2.5)
        assert summary.mean_score == pytest.approx(2.5)

    def test_right_skewed_tail_separates_mean_from_median(self):
        # The motivating case: one huge outlier drags the mean well above the
        # median, and p90 exposes the tail the mean alone would hide.
        values = [1.0] * 9 + [5.0, 100.0]
        summary = score_summary_from_values(values, 0)
        assert summary.mean_score == pytest.approx(114 / 11)
        assert summary.median_score == pytest.approx(1.0)
        assert summary.p90_score == pytest.approx(5.0)
        assert summary.max_score == pytest.approx(100.0)

    def test_quartiles(self):
        summary = score_summary_from_values([float(v) for v in range(1, 11)], 0)
        assert summary.p25_score == pytest.approx(3.25)
        assert summary.median_score == pytest.approx(5.5)
        assert summary.p75_score == pytest.approx(7.75)
        assert summary.p90_score == pytest.approx(9.1)


def test_score_summary_percentiles(mock_eval_for_score_summary):
    """compute_score_summary reports the distribution, not just the mean, and
    excludes skipped runs from it exactly as it does for the mean."""
    eval = mock_eval_for_score_summary
    config = Mock(spec=EvalConfig)

    accuracy_values = [0.1, 0.2, 0.3, 1.0]
    runs = [
        EvalRun(
            task_run_config_id="rc1",
            scores={"accuracy": value, "relevance": 0.5},
            input="input",
            output="output",
            dataset_id=f"ds{i}",
        )
        for i, value in enumerate(accuracy_values)
    ]
    # A skipped run with a wild score must not move any statistic.
    runs.append(
        EvalRun(
            task_run_config_id="rc1",
            scores={"accuracy": 99.0, "relevance": 99.0},
            input="input",
            output="output",
            dataset_id="ds_skipped",
            skipped_reason="extraction_failed",
        )
    )
    config.runs.return_value = runs

    task_run_configs = [Mock(spec=TaskRunConfig, id="rc1")]
    split = stub_split({f"ds{i}" for i in range(4)} | {"ds_skipped"})

    result = compute_score_summary(eval, config, task_run_configs, split)

    scores = result.results["rc1"]["accuracy"]
    assert scores.n_used == 4
    assert scores.n_excluded == 1
    assert scores.mean_score == pytest.approx(0.4)
    assert scores.min_score == pytest.approx(0.1)
    assert scores.p25_score == pytest.approx(0.175)
    assert scores.median_score == pytest.approx(0.25)
    assert scores.p75_score == pytest.approx(0.475)
    assert scores.p90_score == pytest.approx(0.79)
    assert scores.max_score == pytest.approx(1.0)


def test_score_summary_n_used_n_excluded(mock_eval_for_score_summary):
    eval = mock_eval_for_score_summary
    config = Mock(spec=EvalConfig)

    runs = [
        EvalRun(
            task_run_config_id="rc1",
            scores={"accuracy": 0.8, "relevance": 0.9},
            input="input",
            output="output",
            dataset_id="ds1",
        ),
        EvalRun(
            task_run_config_id="rc1",
            scores={},
            input="input",
            output="output",
            dataset_id="ds2",
            skipped_reason="missing_reference_key",
            skipped_detail="key foo",
        ),
        EvalRun(
            task_run_config_id="rc1",
            scores={"accuracy": 0.6, "relevance": 0.7},
            input="input",
            output="output",
            dataset_id="ds3",
        ),
    ]
    config.runs.return_value = runs

    task_run_configs = [Mock(spec=TaskRunConfig, id="rc1")]
    split = stub_split({"ds1", "ds2", "ds3"})

    result = compute_score_summary(eval, config, task_run_configs, split)

    assert result.dataset_size == 3
    scores = result.results["rc1"]
    assert scores["accuracy"].mean_score == pytest.approx(0.7)
    assert scores["accuracy"].n_used == 2
    assert scores["accuracy"].n_excluded == 1
    assert scores["relevance"].mean_score == pytest.approx(0.8)
    assert scores["relevance"].n_used == 2
    assert scores["relevance"].n_excluded == 1
    assert result.run_config_percent_complete["rc1"] == 1.0


def test_score_summary_all_skipped(mock_eval_for_score_summary):
    eval = mock_eval_for_score_summary
    config = Mock(spec=EvalConfig)

    runs = [
        EvalRun(
            task_run_config_id="rc1",
            scores={},
            input="input",
            output="output",
            dataset_id="ds1",
            skipped_reason="extraction_failed",
        ),
        EvalRun(
            task_run_config_id="rc1",
            scores={},
            input="input",
            output="output",
            dataset_id="ds2",
            skipped_reason="missing_trace",
        ),
    ]
    config.runs.return_value = runs

    task_run_configs = [Mock(spec=TaskRunConfig, id="rc1")]
    split = stub_split({"ds1", "ds2"})

    result = compute_score_summary(eval, config, task_run_configs, split)

    assert result.dataset_size == 2
    scores = result.results["rc1"]
    assert len(scores) == 2
    assert scores["accuracy"].mean_score is None
    assert scores["accuracy"].n_used == 0
    assert scores["accuracy"].n_excluded == 2
    assert scores["relevance"].mean_score is None
    assert scores["relevance"].n_used == 0
    assert scores["relevance"].n_excluded == 2
    # Percentiles follow the mean: None, not 0.0, when nothing was scored.
    for score_key in ("accuracy", "relevance"):
        assert scores[score_key].min_score is None
        assert scores[score_key].p25_score is None
        assert scores[score_key].median_score is None
        assert scores[score_key].p75_score is None
        assert scores[score_key].p90_score is None
        assert scores[score_key].max_score is None
    assert result.run_config_percent_complete["rc1"] == 1.0


RUN_RESULTS_PATH = (
    "/api/projects/project1/tasks/task1/evals/eval1"
    "/eval_config/eval_config1/run_config/run_config1/results"
)


def _tagged_task_run(task: Task, data_source: DataSource, *tags: str) -> TaskRun:
    run = TaskRun(
        parent=task,
        input="in",
        input_source=data_source,
        output=TaskOutput(output="out"),
        tags=list(tags),
    )
    run.save_to_file()
    return run


def _tagged_eval_input(task: Task, *tags: str, **overrides) -> EvalInput:
    eval_input = EvalInput(
        parent=task,
        data=SingleTurnEvalInputData(user_message=UserMessage(text="in")),
        tags=list(tags),
        **overrides,
    )
    eval_input.save_to_file()
    return eval_input


def _scored(eval_config: EvalConfig, **item) -> EvalRun:
    run = EvalRun(
        task_run_config_id="run_config1",
        scores={"score1": 3.0, "overall_rating": 1.0},
        input="input",
        output="output",
        parent=eval_config,
        **item,
    )
    run.save_to_file()
    return run


@pytest.mark.asyncio
async def test_get_eval_run_results(
    client,
    mock_task_from_id,
    mock_task,
    mock_eval,
    mock_eval_config,
    mock_run_config,
    data_source,
):
    mock_task_from_id.return_value = mock_task

    in_split = _tagged_task_run(mock_task, data_source, "eval_set")
    eval_run = _scored(mock_eval_config, dataset_id=in_split.id)

    # Test successful retrieval
    response = client.get(RUN_RESULTS_PATH, params={"split": "test"})

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "results" in data
    assert "eval" in data
    assert "eval_config" in data
    assert "run_config" in data

    # Verify results content
    assert len(data["results"]) == 1
    assert data["results"][0]["eval_run"]["id"] == eval_run.id
    assert data["results"][0]["eval_run"]["task_run_config_id"] == mock_run_config.id
    assert data["results"][0]["eval_run"]["scores"] == {
        "score1": 3.0,
        "overall_rating": 1.0,
    }

    # Test with invalid eval ID
    response = client.get(
        "/api/projects/project1/tasks/task1/evals/invalid_eval"
        "/eval_config/eval_config1/run_config/run_config1/results",
        params={"split": "test"},
    )
    assert response.status_code == 404

    # Test with invalid eval config ID
    response = client.get(
        "/api/projects/project1/tasks/task1/evals/eval1"
        "/eval_config/invalid_config/run_config/run_config1/results",
        params={"split": "test"},
    )
    assert response.status_code == 404

    # Test with invalid run config ID
    response = client.get(
        "/api/projects/project1/tasks/task1/evals/eval1"
        "/eval_config/eval_config1/run_config/invalid_run_config/results",
        params={"split": "test"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_eval_run_results_content_part_trace_after_readonly_scan(
    client,
    mock_task_from_id,
    mock_task,
    mock_eval,
    mock_eval_config,
    mock_run_config,
    data_source,
):
    """List-valued message content is validated into a lazy iterator by pydantic. A
    readonly runs scan caches that instance, and the results endpoint's bulk trace load
    then hits the cache - which must not fail on copying the lazy content."""
    item = _tagged_task_run(mock_task, data_source, "eval_set")
    trace_run = TaskRun(
        parent=mock_task,
        input="trace input",
        input_source=data_source,
        output=TaskOutput(output="trace output"),
        trace=[
            {"role": "user", "content": [{"type": "text", "text": "content part"}]},
            {"role": "assistant", "content": "answer"},
        ],
    )
    trace_run.save_to_file()
    eval_run = EvalRun(
        task_run_config_id="run_config1",
        scores={"score1": 3.0, "overall_rating": 1.0},
        dataset_id=item.id,
        scored_run_id=trace_run.id,
        parent=mock_eval_config,
    )
    eval_run.save_to_file()

    # Populate the model cache with readonly instances, as any runs scan does.
    for _ in mock_task.runs(readonly=True):
        pass

    response = client.get(RUN_RESULTS_PATH, params={"split": "test"})

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["eval_run"]["id"] == eval_run.id
    assert results[0]["input"] == "trace input"
    assert results[0]["output"] == "trace output"
    assert "content part" in results[0]["task_run_trace"]


class TestGetEvalRunResultsSplits:
    """Every response about eval results is scoped to exactly one split (spec 5)."""

    def test_requires_a_split(self, client, mock_task_from_id, mock_eval):
        response = client.get(RUN_RESULTS_PATH)

        assert response.status_code == 422
        assert "split" in response.text

    def test_unknown_split_name_is_rejected_before_anything_loads(
        self, client, mock_task_from_id, mock_eval
    ):
        """FastAPI's enum validation runs before the endpoint body, so no disk is read.

        The name claims "before anything loads", so assert it: a status code alone would
        also pass if the split name were validated after the task and eval were pulled
        off disk, which is a different — and slower, and 404-before-422 — contract.
        """
        response = client.get(RUN_RESULTS_PATH, params={"split": "holdout"})

        assert response.status_code == 422
        assert "holdout" in response.text
        mock_task_from_id.assert_not_called()

    def test_422s_for_a_split_this_eval_does_not_have(
        self, client, mock_task_from_id, mock_eval, mock_eval_config, mock_run_config
    ):
        response = client.get(RUN_RESULTS_PATH, params={"split": "val"})

        assert response.status_code == 422
        assert "no 'val' split" in response.json()["message"]
        assert mock_eval.id in response.json()["message"]

    def test_returns_only_the_requested_splits_results(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
        data_source,
    ):
        mock_eval.set_split("train", TaskRunSplit(filter_id="tag::train_set"))
        mock_eval.save_to_file()

        test_item = _tagged_task_run(mock_task, data_source, "eval_set")
        train_item = _tagged_task_run(mock_task, data_source, "train_set")
        untagged_item = _tagged_task_run(mock_task, data_source, "other")
        test_run = _scored(mock_eval_config, dataset_id=test_item.id)
        train_run = _scored(mock_eval_config, dataset_id=train_item.id)
        _scored(mock_eval_config, dataset_id=untagged_item.id)

        test_results = client.get(RUN_RESULTS_PATH, params={"split": "test"}).json()
        train_results = client.get(RUN_RESULTS_PATH, params={"split": "train"}).json()

        assert [r["eval_run"]["id"] for r in test_results["results"]] == [test_run.id]
        assert [r["eval_run"]["id"] for r in train_results["results"]] == [train_run.id]

    def test_returns_eval_input_backed_results(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
    ):
        mock_eval.splits["test"] = EvalInputSplit(filter_id="tag::inputs")
        mock_eval.eval_set_filter_id = None
        mock_eval.save_to_file()
        eval_input = _tagged_eval_input(mock_task, "inputs")
        eval_run = _scored(mock_eval_config, eval_input_id=eval_input.id)

        response = client.get(RUN_RESULTS_PATH, params={"split": "test"})

        assert response.status_code == 200
        assert [r["eval_run"]["id"] for r in response.json()["results"]] == [
            eval_run.id
        ]

    def test_does_not_credit_a_task_run_to_an_eval_inputs_id(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
        data_source,
    ):
        """Both stores draw ids from one 12-digit generator, so membership on a bare id
        would silently admit one store's result into the other store's split."""
        eval_input = _tagged_eval_input(mock_task, "inputs", id="500000000001")
        colliding_task_run = TaskRun(
            id=eval_input.id,
            parent=mock_task,
            input="in",
            input_source=DataSource(
                type=DataSourceType.human, properties={"created_by": "test"}
            ),
            output=TaskOutput(output="out"),
            tags=["inputs"],
        )
        colliding_task_run.save_to_file()
        mock_eval.splits["test"] = EvalInputSplit(filter_id="tag::inputs")
        mock_eval.eval_set_filter_id = None
        mock_eval.save_to_file()
        _scored(mock_eval_config, dataset_id=colliding_task_run.id)

        response = client.get(RUN_RESULTS_PATH, params={"split": "test"})

        assert response.status_code == 200
        assert response.json()["results"] == []


def _eval_trace(
    task: Task,
    data_source: DataSource,
    source: EvalItemSource,
    output: str = "traced output",
    **overrides,
) -> TaskRun:
    """A TaskRun the eval runner would have generated: flagged, with a trace and usage."""
    overrides.setdefault("trace", [{"role": "user", "content": "traced input"}])
    overrides.setdefault(
        "usage", Usage(input_tokens=11, output_tokens=7, total_tokens=18, cost=0.5)
    )
    run = TaskRun(
        parent=task,
        input="traced input",
        input_source=data_source,
        output=TaskOutput(output=output, source=data_source),
        eval_source=source,
        **overrides,
    )
    run.save_to_file()
    return run


def _pointer_scored(eval_config: EvalConfig, scored_run_id: str, **item) -> EvalRun:
    """A pointer-mode score record: it names a trace and carries no inline copy of it."""
    item.setdefault("scores", {"score1": 3.0, "overall_rating": 1.0})
    run = EvalRun(
        task_run_config_id="run_config1",
        scored_run_id=scored_run_id,
        parent=eval_config,
        **item,
    )
    run.save_to_file()
    return run


class TestResolveEvalRunTraces:
    """`resolve_eval_run_traces` and its helpers, called directly.

    The endpoint tests below cover the same join end to end; these pin the pieces the
    endpoint can't isolate - how many times the directory is read, and what each helper
    answers for an input the endpoint has no way to construct.
    """

    def test_reads_the_runs_directory_once_per_kind_of_lookup(
        self, mock_task, mock_eval, mock_eval_config, data_source
    ):
        """Bulk, not per record: this is the invariant that silently rots into an N+1,
        and the cost is a directory scan over every eval trace in the project."""
        traces = []
        skips = []
        for _ in range(3):
            item = _tagged_task_run(mock_task, data_source, "eval_set")
            traces.append(
                _pointer_scored(
                    mock_eval_config,
                    _eval_trace(
                        mock_task,
                        data_source,
                        EvalItemSource(source_type="task_run", source_id=item.id),
                    ).id,
                    dataset_id=item.id,
                )
            )
            skip_item = _tagged_task_run(mock_task, data_source, "eval_set")
            skips.append(
                EvalRun(
                    parent=mock_eval_config,
                    task_run_config_id="run_config1",
                    scores={},
                    skipped_reason="incompatible_input_shape",
                    dataset_id=skip_item.id,
                )
            )

        with patch.object(
            TaskRun, "from_ids_and_parent_path", wraps=TaskRun.from_ids_and_parent_path
        ) as bulk_load:
            resolve_eval_run_traces(mock_task, traces + skips)

        # One for the three pointer records' traces, one for the three skips' source
        # items. Six records, two reads.
        assert bulk_load.call_count == 2

    def test_reads_nothing_when_no_record_needs_a_lookup(
        self, mock_task, mock_eval, mock_eval_config
    ):
        """An all-legacy result set resolves entirely from its own fields. The empty
        guard matters because from_ids_and_parent_path reads uncached children to check
        them, so a no-op call would read the whole runs directory."""
        legacy = EvalRun(
            parent=mock_eval_config,
            task_run_config_id="run_config1",
            scores={"score1": 3.0, "overall_rating": 1.0},
            dataset_id="500000000001",
            input="in",
            output="out",
        )

        with (
            patch.object(TaskRun, "from_ids_and_parent_path") as load_runs,
            patch.object(EvalInput, "from_ids_and_parent_path") as load_inputs,
        ):
            resolved = resolve_eval_run_traces(mock_task, [legacy])

        load_runs.assert_not_called()
        load_inputs.assert_not_called()
        assert resolved[0].input == "in"

    @pytest.mark.parametrize(
        "item_factory,expected",
        [
            (
                lambda: TaskRun(
                    input="run input",
                    input_source=DataSource(
                        type=DataSourceType.human, properties={"created_by": "t"}
                    ),
                    output=TaskOutput(output="out"),
                ),
                "run input",
            ),
            (
                lambda: EvalInput(
                    data=SingleTurnEvalInputData(
                        user_message=UserMessage(text="single turn")
                    )
                ),
                "single turn",
            ),
            (
                lambda: EvalInput(
                    data=MultiTurnSyntheticEvalInputData(
                        first_message=UserMessage(text="first turn"),
                        synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
                    )
                ),
                "first turn",
            ),
            (
                lambda: EvalInput(
                    data=MultiTurnSyntheticEvalInputData(
                        synthetic_user_info=SyntheticUserInfo(persona="p", goal="g")
                    )
                ),
                None,
            ),
        ],
        ids=["task_run", "single_turn", "multi_turn", "multi_turn_no_first_message"],
    )
    def test_eval_item_input_text_reads_every_item_shape(self, item_factory, expected):
        """A multi-turn item with no first message is the one shape with nothing to show,
        and it is reachable: multi-turn is exactly what pre-generation skips are for."""
        assert eval_item_input_text(item_factory()) == expected

    def test_usage_comes_from_the_trace_for_a_pointer_and_inline_for_a_legacy_record(
        self, mock_eval_config
    ):
        inline_usage = Usage(input_tokens=1, total_tokens=1)
        trace_usage = Usage(input_tokens=2, total_tokens=2)
        legacy = EvalRun(
            parent=mock_eval_config,
            task_run_config_id="run_config1",
            scores={"score1": 3.0, "overall_rating": 1.0},
            dataset_id="500000000001",
            input="in",
            output="out",
            task_run_usage=inline_usage,
        )
        pointer = EvalRun(
            parent=mock_eval_config,
            task_run_config_id="run_config1",
            scores={"score1": 3.0, "overall_rating": 1.0},
            dataset_id="500000000002",
            scored_run_id="900000000001",
        )
        usage_by_id = {"900000000001": trace_usage}

        assert eval_run_task_usage(legacy, usage_by_id) == inline_usage
        assert eval_run_task_usage(pointer, usage_by_id) == trace_usage
        # A dangling pointer contributes nothing rather than counting as a zero.
        assert eval_run_task_usage(pointer, {}) is None

    def test_summary_eval_config_picks_the_only_config_or_the_named_one(
        self, mock_task, mock_eval, mock_eval_config
    ):
        only_config = summary_eval_config(mock_eval)
        assert only_config is not None
        assert only_config.id == mock_eval_config.id

        second = EvalConfig(
            id="eval_config2",
            name="Second",
            config_type=EvalConfigType.g_eval,
            properties={"eval_steps": ["step1"]},
            parent=mock_eval,
            model_name="gpt-4",
            model_provider="openai",
        )
        second.save_to_file()

        # Two configs and no default named: no non-arbitrary answer, so none is given.
        assert summary_eval_config(mock_eval) is None

        mock_eval.current_config_id = second.id
        named_config = summary_eval_config(mock_eval)
        assert named_config is not None
        assert named_config.id == second.id

    def test_a_dangling_trace_reference_is_logged_once_for_the_request(
        self, mock_task, mock_eval, mock_eval_config, caplog
    ):
        """Delete protection means a missing trace is data lost out of band, so it is
        worth a line - but one per request, not one per record."""
        dangling = [
            _pointer_scored(
                mock_eval_config, f"90000000000{i}", dataset_id=f"50000000000{i}"
            )
            for i in range(1, 3)
        ]

        with caplog.at_level(
            logging.WARNING, logger="app.desktop.studio_server.eval_api"
        ):
            resolve_eval_run_traces(mock_task, dangling)

        warnings = [
            record
            for record in caplog.records
            if record.name == "app.desktop.studio_server.eval_api"
        ]
        assert len(warnings) == 1
        assert "2 of 2 eval traces" in warnings[0].message

    def test_the_usage_pre_pass_reads_the_runs_directory_once_for_every_eval(
        self, mock_task, mock_eval, mock_eval_config, data_source
    ):
        """The whole reason this pre-pass exists: a per-eval load would be one scan of
        `runs/` per eval, over a directory that now holds every eval trace. Moving the
        load back inside the loop passes every other test."""
        second_eval = Eval(
            id="eval2",
            name="Second Eval",
            description="Second",
            template=EvalTemplateId.bias,
            output_scores=mock_eval.output_scores,
            eval_set_filter_id="tag::eval_set",
            eval_configs_filter_id="tag::golden",
            parent=mock_task,
        )
        second_eval.save_to_file()
        second_config = EvalConfig(
            name="Second Judge",
            config_type=EvalConfigType.g_eval,
            properties={"eval_steps": ["step1"]},
            parent=second_eval,
            model_name="gpt-4",
            model_provider="openai",
        )
        second_config.save_to_file()

        for eval_config in (mock_eval_config, second_config):
            item = _tagged_task_run(mock_task, data_source, "eval_set")
            trace = _eval_trace(
                mock_task,
                data_source,
                EvalItemSource(source_type="task_run", source_id=item.id),
            )
            _pointer_scored(eval_config, trace.id, dataset_id=item.id)

        with patch.object(
            TaskRun, "from_ids_and_parent_path", wraps=TaskRun.from_ids_and_parent_path
        ) as bulk_load:
            usage_by_id = scored_trace_usage_for_run_config(
                mock_task, [mock_eval, second_eval], "run_config1"
            )

        assert bulk_load.call_count == 1
        assert len(usage_by_id) == 2

    def test_the_usage_pre_pass_skips_records_that_never_reach_the_rollup(
        self, mock_task, mock_eval, mock_eval_config, data_source
    ):
        """A skipped record is dropped before usage is read, so loading its trace would
        be pure cost - and traces are the large field."""
        scored_item = _tagged_task_run(mock_task, data_source, "eval_set")
        scored_trace = _eval_trace(
            mock_task,
            data_source,
            EvalItemSource(source_type="task_run", source_id=scored_item.id),
        )
        _pointer_scored(mock_eval_config, scored_trace.id, dataset_id=scored_item.id)

        skipped_item = _tagged_task_run(mock_task, data_source, "eval_set")
        skipped_trace = _eval_trace(
            mock_task,
            data_source,
            EvalItemSource(source_type="task_run", source_id=skipped_item.id),
        )
        _pointer_scored(
            mock_eval_config,
            skipped_trace.id,
            dataset_id=skipped_item.id,
            scores={},
            skipped_reason="missing_reference_key",
        )

        usage_by_id = scored_trace_usage_for_run_config(
            mock_task, [mock_eval], "run_config1"
        )

        assert set(usage_by_id) == {scored_trace.id}


class TestScoredTraceUsage:
    """What one scored TaskRun's spend reads as in a summary.

    Three record generations share the read path: standalone driven traces
    (assistant usage + separate synthetic-user spend), dataset multi-turn chain
    leaves (last-turn usage, conversation totals in cumulative_usage), and
    migrated legacy traces (the blend fused into usage). One function must read
    all three correctly or a summary quietly misprices whole eval runs.
    """

    def test_driven_trace_blends_assistant_and_synthetic_user_spend(
        self, mock_task, mock_eval, mock_eval_config, data_source
    ):
        """End to end through the pre-pass: the reported cost is the assistant's
        plus the synthetic-user driver's, with tokens and latency untouched
        (the driver's record carries cost only)."""
        item = _tagged_task_run(mock_task, data_source, "eval_set")
        trace = _eval_trace(
            mock_task,
            data_source,
            EvalItemSource(source_type="task_run", source_id=item.id),
            usage=Usage(
                input_tokens=100,
                output_tokens=40,
                total_tokens=140,
                cost=0.5,
                total_llm_latency_ms=800,
            ),
            synthetic_user_usage=Usage(cost=0.25),
        )
        _pointer_scored(mock_eval_config, trace.id, dataset_id=item.id)

        usage_by_id = scored_trace_usage_for_run_config(
            mock_task, [mock_eval], "run_config1"
        )

        usage = usage_by_id[trace.id]
        assert usage is not None
        assert usage.cost == pytest.approx(0.75)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 40
        assert usage.total_tokens == 140
        assert usage.total_llm_latency_ms == 800

    def test_chain_leaf_reports_conversation_totals_not_its_last_turn(
        self, mock_task, data_source
    ):
        """A dataset chain leaf's `usage` covers only its final turn; the
        summary must report the conversation totals from `cumulative_usage`,
        keeping latency from `usage` (cumulative carries none)."""
        multiturn_task = mock_task.model_copy(update={"turn_mode": TurnMode.multiturn})
        leaf = TaskRun(
            parent=multiturn_task,
            parent_task_run_id="parent_run_id",
            input="turn 3",
            input_source=data_source,
            output=TaskOutput(output="reply", source=data_source),
            usage=Usage(
                input_tokens=10, total_tokens=12, cost=0.1, total_llm_latency_ms=250
            ),
            cumulative_usage=MessageUsage(
                input_tokens=300, output_tokens=90, total_tokens=390, cost=1.5
            ),
        )

        usage = scored_trace_usage(leaf)
        assert usage is not None
        assert usage.input_tokens == 300
        assert usage.output_tokens == 90
        assert usage.total_tokens == 390
        assert usage.cost == pytest.approx(1.5)
        assert usage.total_llm_latency_ms == 250

    def test_chain_leaf_without_cumulative_does_not_report_last_turn_as_totals(
        self, mock_task, data_source
    ):
        """A leaf that predates cumulative_usage has unknown conversation
        totals; reporting its last turn's tokens as the whole conversation
        would understate silently, so only the latency survives."""
        multiturn_task = mock_task.model_copy(update={"turn_mode": TurnMode.multiturn})
        leaf = TaskRun(
            parent=multiturn_task,
            parent_task_run_id="parent_run_id",
            input="turn 3",
            input_source=data_source,
            output=TaskOutput(output="reply", source=data_source),
            usage=Usage(input_tokens=10, cost=0.1, total_llm_latency_ms=250),
        )

        usage = scored_trace_usage(leaf)
        assert usage is not None
        assert usage.input_tokens is None
        assert usage.cost is None
        assert usage.total_llm_latency_ms == 250

    def test_migrated_legacy_trace_reads_unchanged(self, mock_task, data_source):
        """Migrated traces carry the blend fused inside `usage` with the
        synthetic-user field null, so the sum must be a no-op for them."""
        blended = Usage(input_tokens=100, total_tokens=140, cost=1.25)
        trace = TaskRun(
            parent=mock_task,
            input="in",
            input_source=data_source,
            output=TaskOutput(output="out", source=data_source),
            usage=blended,
        )
        assert trace.synthetic_user_usage is None
        assert scored_trace_usage(trace) == blended

    def test_synthetic_user_blends_cost_only(self, mock_task, data_source):
        """The driver's cost joins the total; its tokens and latency do not.

        The synthetic user is normally a different model on a different provider
        from the agent under test. Folding its tokens in would attribute them to
        the agent (~3.5k input per conversation) and make cost/token meaningless,
        and folding its latency in would make every driven run config look slower
        than it is. Cost alone is total-spend semantics, and it is what migrated
        legacy records already blend — so all three quantities keep one meaning
        across record generations.
        """
        trace = TaskRun(
            parent=mock_task,
            input="in",
            input_source=data_source,
            output=TaskOutput(output="out", source=data_source),
            usage=Usage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cost=1.0,
                total_llm_latency_ms=4000,
            ),
            synthetic_user_usage=Usage(
                input_tokens=3548,
                output_tokens=61,
                total_tokens=3609,
                cost=0.25,
                total_llm_latency_ms=9000,
            ),
        )

        usage = scored_trace_usage(trace)

        assert usage is not None
        # Cost blends: both models' spend produced this trace.
        assert usage.cost == pytest.approx(1.25)
        # Tokens and latency stay the agent's alone.
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.total_llm_latency_ms == 4000

    def test_synthetic_user_without_cost_leaves_the_total_alone(
        self, mock_task, data_source
    ):
        """A driver that reported tokens but no cost must not perturb anything —
        including not turning an all-agent figure into a different object."""
        agent = Usage(input_tokens=100, total_tokens=150, cost=1.0)
        trace = TaskRun(
            parent=mock_task,
            input="in",
            input_source=data_source,
            output=TaskOutput(output="out", source=data_source),
            usage=agent,
            synthetic_user_usage=Usage(input_tokens=3548, total_tokens=3609),
        )

        assert scored_trace_usage(trace) == agent

    def test_nothing_to_report_reads_as_none(self, mock_task, data_source):
        trace = TaskRun(
            parent=mock_task,
            input="in",
            input_source=data_source,
            output=TaskOutput(output="out", source=data_source),
        )
        assert scored_trace_usage(trace) is None


class TestEvalRunTraceJoin:
    """The results endpoint renders one shape whether the trace is inline or pointed at.

    Records written since the trace/score split keep the trace on a TaskRun, so without
    this join every new result would render blank input and output.
    """

    def _results(self, client) -> List[Dict]:
        response = client.get(RUN_RESULTS_PATH, params={"split": "test"})
        assert response.status_code == 200
        return response.json()["results"]

    def test_pointer_and_legacy_records_resolve_to_the_same_shape(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
        data_source,
    ):
        # A V1 eval only stores a trace when it evaluates one, and the point of this
        # test is that both records carry every field.
        mock_eval.evaluation_data_type = EvalDataType.full_trace
        mock_eval.save_to_file()
        legacy_item = _tagged_task_run(mock_task, data_source, "eval_set")
        pointer_item = _tagged_task_run(mock_task, data_source, "eval_set")
        legacy = EvalRun(
            parent=mock_eval_config,
            task_run_config_id="run_config1",
            scores={"score1": 3.0, "overall_rating": 1.0},
            dataset_id=legacy_item.id,
            input="traced input",
            output="traced output",
            task_run_trace=json.dumps(
                [{"role": "user", "content": "traced input"}], ensure_ascii=False
            ),
            task_run_usage=Usage(
                input_tokens=11, output_tokens=7, total_tokens=18, cost=0.5
            ),
        )
        legacy.save_to_file()
        trace = _eval_trace(
            mock_task,
            data_source,
            EvalItemSource(source_type="task_run", source_id=pointer_item.id),
        )
        pointer = _pointer_scored(
            mock_eval_config, trace.id, dataset_id=pointer_item.id
        )

        by_id = {r["eval_run"]["id"]: r for r in self._results(client)}

        assert by_id[legacy.id]["input"] == by_id[pointer.id]["input"] == "traced input"
        assert (
            by_id[legacy.id]["output"] == by_id[pointer.id]["output"] == "traced output"
        )
        assert by_id[legacy.id]["task_run_usage"] == by_id[pointer.id]["task_run_usage"]
        assert json.loads(by_id[pointer.id]["task_run_trace"]) == json.loads(
            by_id[legacy.id]["task_run_trace"]
        )

    def test_a_trace_carrying_per_message_usage_still_serializes(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
        data_source,
    ):
        """Regression: every trace the eval runner writes has per-message `usage`.

        Pydantic types that key as a `MessageUsage` model - on the TaskRun as built and
        again when it is read back off disk - so serializing the trace with a plain
        `json.dumps` raised `TypeError` and the endpoint 500'd on any real eval result.
        """
        item = _tagged_task_run(mock_task, data_source, "eval_set")
        trace = _eval_trace(
            mock_task,
            data_source,
            EvalItemSource(source_type="task_run", source_id=item.id),
            trace=[
                {"role": "user", "content": "traced input"},
                {
                    "role": "assistant",
                    "content": "traced output",
                    "usage": MessageUsage(
                        input_tokens=11, output_tokens=7, total_tokens=18, cost=0.5
                    ),
                },
            ],
        )
        _pointer_scored(mock_eval_config, trace.id, dataset_id=item.id)

        result = self._results(client)[0]

        assert json.loads(result["task_run_trace"])[1]["usage"] == {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "cost": 0.5,
            "cached_tokens": None,
        }

    def test_returns_the_output_that_was_scored_not_a_later_repair(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
        data_source,
    ):
        """A repair can happen after scoring, so it is not what the score describes."""
        item = _tagged_task_run(mock_task, data_source, "eval_set")
        trace = _eval_trace(
            mock_task,
            data_source,
            EvalItemSource(source_type="task_run", source_id=item.id),
            repaired_output=TaskOutput(output="repaired output", source=data_source),
            repair_instructions="fix it",
        )
        _pointer_scored(mock_eval_config, trace.id, dataset_id=item.id)

        assert self._results(client)[0]["output"] == "traced output"

    def test_a_dangling_scored_run_id_still_renders_its_scores(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
        data_source,
    ):
        """Only the drill-through is lost. Never raises, never drops the score."""
        item = _tagged_task_run(mock_task, data_source, "eval_set")
        _pointer_scored(mock_eval_config, "900000000001", dataset_id=item.id)

        result = self._results(client)[0]

        assert result["eval_run"]["scores"] == {"score1": 3.0, "overall_rating": 1.0}
        assert result["output"] is None
        assert result["task_run_trace"] is None
        assert result["task_run_usage"] is None
        # input alone falls back to the dataset item, which is still a true statement of
        # what was asked.
        assert result["input"] == "in"

    def test_a_scoring_skip_resolves_from_its_trace(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
        data_source,
    ):
        """A skip that happened after generation is a pointer record like any other."""
        item = _tagged_task_run(mock_task, data_source, "eval_set")
        trace = _eval_trace(
            mock_task,
            data_source,
            EvalItemSource(source_type="task_run", source_id=item.id),
        )
        _pointer_scored(
            mock_eval_config,
            trace.id,
            dataset_id=item.id,
            scores={},
            skipped_reason="missing_reference_key",
        )

        result = self._results(client)[0]

        assert result["input"] == "traced input"
        assert result["output"] == "traced output"


class TestPreGenerationSkipInput:
    """A run skipped before generation has no trace and no inline input.

    It has neither a `scored_run_id` to join through nor the `input` pre-split records
    carried, so without resolving the dataset item these rows render blank forever.
    """

    def _skip(self, eval_config: EvalConfig, **item) -> EvalRun:
        run = EvalRun(
            parent=eval_config,
            task_run_config_id="run_config1",
            scores={},
            skipped_reason="incompatible_input_shape",
            skipped_detail="V2 evals do not yet support multi-turn inputs",
            **item,
        )
        run.save_to_file()
        return run

    def _input_of_only_result(self, client) -> str | None:
        response = client.get(RUN_RESULTS_PATH, params={"split": "test"})
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 1
        return results[0]["input"]

    def test_resolves_input_from_a_task_run_item(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
        data_source,
    ):
        item = _tagged_task_run(mock_task, data_source, "eval_set")
        self._skip(mock_eval_config, dataset_id=item.id)

        assert self._input_of_only_result(client) == "in"

    def test_resolves_input_from_a_single_turn_eval_input(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
    ):
        mock_eval.splits["test"] = EvalInputSplit(filter_id="tag::inputs")
        mock_eval.eval_set_filter_id = None
        mock_eval.save_to_file()
        eval_input = _tagged_eval_input(mock_task, "inputs")
        self._skip(mock_eval_config, eval_input_id=eval_input.id)

        assert self._input_of_only_result(client) == "in"

    def test_resolves_input_from_a_multi_turn_eval_input(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
    ):
        """The case the runner actually writes: multi-turn items are skipped by V2."""
        mock_eval.splits["test"] = EvalInputSplit(filter_id="tag::inputs")
        mock_eval.eval_set_filter_id = None
        mock_eval.save_to_file()
        eval_input = EvalInput(
            parent=mock_task,
            data=MultiTurnSyntheticEvalInputData(
                first_message=UserMessage(text="first turn"),
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
            ),
            tags=["inputs"],
        )
        eval_input.save_to_file()
        self._skip(mock_eval_config, eval_input_id=eval_input.id)

        assert self._input_of_only_result(client) == "first turn"

    def test_a_missing_source_item_renders_no_input_rather_than_raising(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
    ):
        """The item is gone but its skip record isn't. The split is stubbed because a
        real one is resolved from the items themselves, so a deleted item leaves no
        split to be in."""
        self._skip(mock_eval_config, dataset_id="900000000002")

        with patch_resolve_split(test=stub_split({"900000000002"})):
            assert self._input_of_only_result(client) is None


@pytest.mark.asyncio
async def test_get_eval_config_compare_summary(
    client,
    mock_task_from_id,
    mock_task,
    mock_eval,
    mock_eval_config,
    mock_run_config,
):
    mock_task_from_id.return_value = mock_task

    # structured data to make it easier to generate test cases.
    @dataclass
    class EvalConfigSummaryTestData:
        human_overall_rating: float | None
        score1_overall_rating: float | None
        eval_overall_rating: float
        eval__score1_rating: float
        eval_config_id: str
        skip_eval_run: bool = False
        skip_golden_tag: bool = False

    test_data: List[EvalConfigSummaryTestData] = [
        # Test 1: ec1
        # Normal run, with some data to check calculations on a single run
        EvalConfigSummaryTestData(
            human_overall_rating=5.0,
            score1_overall_rating=2.0,
            eval_overall_rating=1.0,
            eval__score1_rating=3.5,
            eval_config_id="ec1",
        ),
        # Should be ignored as it's not in the eval set filter (golden tag). Would mess up the scores of eval_config1 if included
        EvalConfigSummaryTestData(
            human_overall_rating=5.0,
            score1_overall_rating=5.0,
            eval_overall_rating=4.0,
            eval__score1_rating=4.0,
            eval_config_id="ec1",
            skip_golden_tag=True,
        ),
        # Test 2: ec2 - Test multiple, and correct averaging
        EvalConfigSummaryTestData(
            human_overall_rating=5.0,
            score1_overall_rating=5.0,
            eval_overall_rating=4.0,
            eval__score1_rating=4.0,
            eval_config_id="ec2",
        ),
        EvalConfigSummaryTestData(
            human_overall_rating=5.0,
            score1_overall_rating=1.0,
            eval_overall_rating=3.0,
            eval__score1_rating=3.0,
            eval_config_id="ec2",
        ),
        # Test 3: Dataset item that has partial human rating
        EvalConfigSummaryTestData(
            human_overall_rating=5.0,
            score1_overall_rating=None,
            eval_overall_rating=3.0,
            eval__score1_rating=3.0,
            eval_config_id="ec3",
        ),
        # Test 4: Dataset item that has no human rating
        EvalConfigSummaryTestData(
            human_overall_rating=None,
            score1_overall_rating=None,
            eval_overall_rating=3.0,
            eval__score1_rating=3.0,
            eval_config_id="ec4",
        ),
        # Test 5: skipping eval run should lower the percent complete
        EvalConfigSummaryTestData(
            human_overall_rating=5.0,
            score1_overall_rating=5.0,
            eval_overall_rating=4.0,
            eval__score1_rating=4.0,
            eval_config_id="ec5",
            skip_eval_run=True,
        ),
    ]

    # Count items that don't have skip_golden_tag set to True
    total_in_dataset = sum(1 for x in test_data if not x.skip_golden_tag)

    eval_configs_by_id: Dict[str, EvalConfig] = {}

    assert len(mock_task.requirements) == 1
    assert mock_task.requirements[0].name == "score1"
    score1_requirement_id = mock_task.requirements[0].id
    for test_case in test_data:
        # create eval config if it doesn't exist
        eval_config = eval_configs_by_id.get(test_case.eval_config_id)
        if eval_config is None:
            eval_config = EvalConfig(
                id=test_case.eval_config_id,
                name="Test Eval Config",
                config_type=EvalConfigType.g_eval,
                properties={"eval_steps": ["step1", "step2"]},
                parent=mock_eval,
                model_name="gpt-4",
                model_provider="openai",
            )
            eval_config.save_to_file()
            eval_configs_by_id[test_case.eval_config_id] = eval_config

        tags = ["golden"]
        if test_case.skip_golden_tag:
            tags = []

        ratings = {}
        if test_case.score1_overall_rating is not None:
            ratings[score1_requirement_id] = RequirementRating(
                value=test_case.score1_overall_rating,
                type=TaskOutputRatingType.five_star,
            )

        task_run = TaskRun(
            output=TaskOutput(
                output="Test Output",
                source=DataSource(
                    type=DataSourceType.synthetic,
                    properties={
                        "model_name": "gpt-4",
                        "model_provider": "openai",
                        "adapter_name": "langchain_adapter",
                    },
                ),
                rating=TaskOutputRating(
                    value=test_case.human_overall_rating,
                    requirement_ratings=ratings,
                ),
            ),
            input="Test Input",
            input_source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "gpt-4",
                    "model_provider": "openai",
                    "adapter_name": "langchain_adapter",
                },
            ),
            tags=tags,
            parent=mock_task,
        )
        task_run.save_to_file()

        if test_case.skip_eval_run:
            continue

        eval_run = EvalRun(
            # Calibration records: the judge scored the stored golden output,
            # to be compared against the item's human rating. task_run_eval
            # records (fresh generations) are excluded from these stats.
            eval_config_eval=True,
            task_run_config_id=None,
            scores={
                "score1": test_case.eval__score1_rating,
                "overall_rating": test_case.eval_overall_rating,
            },
            input="input",
            output="output",
            dataset_id=task_run.id,
            parent=eval_config,
        )
        eval_run.save_to_file()

    # A task_run_eval record on a golden item (test/golden overlap is normal)
    # must NOT enter the calibration stats: its score is about a fresh
    # generation, not the stored output the human rated. Attached to test
    # case 5's item — the golden item with no calibration record — so if it
    # wrongly counted, ec5's percent-complete assertion below would fail.
    EvalRun(
        task_run_config_id="run_config1",
        scores={"score1": 1.0, "overall_rating": 1.0},
        input="stray input",
        output="fresh generation output",
        dataset_id=task_run.id,
        parent=eval_config,
    ).save_to_file()

    # Test successful retrieval
    response = client.get(
        "/api/projects/project1/tasks/task1/evals/eval1/eval_configs_score_summary"
    )

    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    results = data["results"]
    assert isinstance(results, dict)

    assert "eval_config_percent_complete" in data
    eval_config_percent_complete = data["eval_config_percent_complete"]
    assert isinstance(eval_config_percent_complete, dict)

    # check the counts
    assert data["fully_rated_count"] == 4
    assert data["partially_rated_count"] == 1
    assert data["not_rated_count"] == 1
    assert data["dataset_size"] == total_in_dataset

    # Test case 1: 1 item should be included, manually calculated scores, should exclude a second item that isn't in the eval config set filter
    assert results["ec1"] == {
        "overall_rating": {
            "mean_squared_error": 16.0,  # error 4.0^2
            "mean_absolute_error": 4.0,  # error 4.0
            "mean_normalized_squared_error": 1,  # max error: 1 v 5
            "mean_normalized_absolute_error": 1,  # max error: 1 v 5
            "spearman_correlation": None,  # Not enough data
            "pearson_correlation": None,
            "kendalltau_correlation": None,
        },
        "score1": {
            "mean_squared_error": 2.25,  # error (3.5-5.0)^2
            "mean_absolute_error": 1.5,  # error 1.5
            "mean_normalized_squared_error": 0.140625,  # hand calc
            "mean_normalized_absolute_error": 0.375,  # 1.5/4
            "spearman_correlation": None,  # Not enough data
            "pearson_correlation": None,  # Not enough data
            "kendalltau_correlation": None,  # Not enough data
        },
    }
    # 1 of total_in_dataset eval configs are are in ec1 test
    assert eval_config_percent_complete["ec1"] == pytest.approx(1 / total_in_dataset)

    # Test case 2: check proper averaging
    assert results["ec2"] == {
        "overall_rating": {
            "mean_squared_error": 2.5,  # error (1^2 + 2^2) / 2
            "mean_absolute_error": 1.5,  # (1+2)/2
            "mean_normalized_squared_error": 0.15625,  # (0.25^2 + 0.5^2) / 2
            "mean_normalized_absolute_error": 0.375,  # (0.25 + 0.5) / 2
            "spearman_correlation": None,
            "pearson_correlation": None,
            "kendalltau_correlation": None,
        },
        "score1": {
            "mean_squared_error": 2.5,  # (1^2+2^2)/2
            "mean_absolute_error": 1.5,  # (1+2)/2
            "mean_normalized_squared_error": 0.15625,  # (0.25^2 + 0.5^2) / 2
            "mean_normalized_absolute_error": 0.375,  # (0.25 + 0.5) / 2
            "spearman_correlation": 0.9999999999999999,
            "pearson_correlation": 1,
            "kendalltau_correlation": 1,
        },
    }
    # 2 of total_in_dataset eval configs are are in ec2 test
    assert eval_config_percent_complete["ec2"] == pytest.approx(2 / total_in_dataset)

    # Test case 3: Check partials still calculate available scores
    assert results["ec3"] == {
        "overall_rating": {
            "mean_squared_error": 4,
            "mean_absolute_error": 2,
            "mean_normalized_squared_error": 0.25,
            "mean_normalized_absolute_error": 0.5,
            "spearman_correlation": None,
            "pearson_correlation": None,
            "kendalltau_correlation": None,
        },
    }
    # 2 of total_in_dataset eval configs are are in ec2 test
    assert eval_config_percent_complete["ec3"] == pytest.approx(1 / total_in_dataset)

    # Test case 4: Check no rating is empty results
    assert results.get("ec4", {}) == {}
    assert eval_config_percent_complete["ec4"] == pytest.approx(1 / total_in_dataset)

    # Test case 5: Check skipping eval run lowers the percent complete
    assert eval_config_percent_complete["ec5"] == pytest.approx(0 / total_in_dataset)


def _seed_golden_run(mock_task) -> TaskRun:
    """One human-rated TaskRun in the golden set (tag::golden), so
    calibration has something to run against."""
    task_run = TaskRun(
        input="golden input",
        input_source=DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": "gpt-4",
                "model_provider": "openai",
                "adapter_name": "langchain_adapter",
            },
        ),
        output=TaskOutput(
            output="golden output",
            source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "gpt-4",
                    "model_provider": "openai",
                    "adapter_name": "langchain_adapter",
                },
            ),
        ),
        tags=["golden"],
        parent=mock_task,
    )
    task_run.save_to_file()
    return task_run


@pytest.mark.asyncio
async def test_run_eval_config_eval(
    client, mock_task_from_id, mock_task, mock_eval, mock_eval_config
):
    mock_task_from_id.return_value = mock_task
    _seed_golden_run(mock_task)

    # Create a mock response for run_eval_runner_with_status
    mock_response = StreamingResponse(
        content=iter([b"data: test\n\n"]), media_type="text/event-stream"
    )

    with patch(
        "app.desktop.studio_server.eval_api.run_eval_runner_with_status"
    ) as mock_run_eval:
        # Set up the mock to return our mock response
        mock_run_eval.return_value = mock_response

        # Call the endpoint
        response = client.get(
            "/api/projects/project1/tasks/task1/evals/eval1/run_calibration"
        )

        # Verify the response
        assert response.status_code == 200

        # Verify run_eval_runner_with_status was called with correct parameters
        mock_run_eval.assert_called_once()

        # Get the EvalRunner that was passed to run_eval_runner_with_status
        eval_runner = mock_run_eval.call_args[0][0]

        # Verify the EvalRunner was configured correctly
        assert len(eval_runner.eval_configs) == 1
        assert eval_runner.eval_configs[0].id == mock_eval_config.id
        assert eval_runner.run_configs is None
        assert eval_runner.eval_run_type == "eval_config_eval"


@pytest.mark.asyncio
async def test_run_eval_config_eval_422s_without_a_golden_set(
    client, mock_task_from_id, mock_task, mock_eval_config
):
    """The refusal has to beat the StreamingResponse.

    This is an SSE endpoint, so anything raised once the generator is running is emitted
    after a 200 status and an empty body. Asserting the status alone would not be enough
    either: this pins the whole HTTP contract functional spec 9 asks for, status and
    reason. The web UI reads these endpoints with `$lib/utils/sse_stream`'s fetch-based
    reader, so this reason is also what the user sees — the last mile is covered by
    `app/web_ui/src/lib/components/run_eval.component.test.ts`.
    """
    mock_task_from_id.return_value = mock_task
    no_golden_eval = Eval(
        id="eval_no_golden",
        name="No Golden Set",
        description="V2 eval that never had judges compared",
        output_scores=[
            EvalOutputScore(
                name="score1", instruction="desc1", type=TaskOutputRatingType.five_star
            ),
        ],
        splits={"test": EvalInputSplit(filter_id="tag::inputs")},
        parent=mock_task,
    )
    no_golden_eval.save_to_file()

    response = client.get(
        "/api/projects/project1/tasks/task1/evals/eval_no_golden/run_calibration"
    )

    assert response.status_code == 422
    message = response.json()["message"]
    assert "eval_no_golden" in message
    assert "no golden set configured" in message


def _llm_judge_properties(reference_keys: list[str]) -> LlmJudgeProperties:
    return LlmJudgeProperties(
        model_name="gpt-4",
        model_provider="openai",
        prompt_template="Grade {{ final_message }}",
        reference_keys=reference_keys,
    )


class TestJudgeRequiresReferenceData:
    """The predicate the Compare Judges page and `run_calibration` both decide from."""

    def test_v2_judge_declaring_a_reference_key(self, mock_eval):
        eval_config = EvalConfig(
            name="Reference judge",
            config_type=EvalConfigType.v2,
            properties=_llm_judge_properties(["reference_answer"]),
            parent=mock_eval,
        )
        assert judge_requires_reference_data(mock_eval, eval_config) is True

    def test_v2_judge_declaring_no_reference_key(self, mock_eval):
        eval_config = EvalConfig(
            name="Ordinary judge",
            config_type=EvalConfigType.v2,
            properties=_llm_judge_properties([]),
            parent=mock_eval,
        )
        assert judge_requires_reference_data(mock_eval, eval_config) is False

    @pytest.mark.parametrize(
        "properties,expected",
        [
            (ExactMatchProperties(reference_key="reference_answer"), True),
            (ExactMatchProperties(expected_value="yes"), False),
            (ContainsProperties(reference_key="reference_answer"), True),
            (ContainsProperties(substring="yes"), False),
            (PatternMatchProperties(pattern="^y"), False),
        ],
    )
    def test_deterministic_judges_follow_their_singular_reference_key(
        self, mock_eval, properties, expected
    ):
        eval_config = EvalConfig(
            name="Deterministic judge",
            config_type=EvalConfigType.v2,
            properties=properties,
            parent=mock_eval,
        )
        assert judge_requires_reference_data(mock_eval, eval_config) is expected

    @pytest.mark.parametrize(
        "data_type,expected",
        [
            (EvalDataType.reference_answer, True),
            (EvalDataType.final_answer, False),
            (EvalDataType.full_trace, False),
            (None, False),
        ],
    )
    def test_v1_judge_follows_the_evals_data_type(self, mock_task, data_type, expected):
        """V1 fails differently — `GEval` raises per job — but for the same reason."""
        reference_eval = Eval(
            id="eval_reference",
            name="Reference Eval",
            evaluation_data_type=data_type,
            output_scores=[
                EvalOutputScore(
                    name="score1",
                    instruction="desc1",
                    type=TaskOutputRatingType.five_star,
                ),
            ],
            eval_set_filter_id="tag::eval_set",
            eval_configs_filter_id="tag::golden",
            parent=mock_task,
        )
        eval_config = EvalConfig(
            name="V1 judge",
            config_type=EvalConfigType.g_eval,
            properties={"eval_steps": ["step1"]},
            model_name="gpt-4",
            model_provider="openai",
            parent=reference_eval,
        )
        assert judge_requires_reference_data(reference_eval, eval_config) is expected

    def test_v1_judge_on_an_ordinary_eval(self, mock_eval, mock_eval_config):
        assert mock_eval.evaluation_data_type != EvalDataType.reference_answer
        assert judge_requires_reference_data(mock_eval, mock_eval_config) is False

    def test_v2_judge_with_untyped_legacy_properties(self, mock_eval):
        """A dict-properties V2 config declares nothing, so it isn't blocked."""
        eval_config = EvalConfig.model_construct(
            name="Legacy shaped",
            config_type=EvalConfigType.v2,
            properties={"eval_steps": ["step1"]},
        )
        assert judge_requires_reference_data(mock_eval, eval_config) is False


@pytest.mark.asyncio
async def test_run_calibration_skips_judges_that_need_reference_data(
    client, mock_task_from_id, mock_task, mock_eval
):
    """A mixed table still compares the judges it can."""
    mock_task_from_id.return_value = mock_task
    # A golden item, so this reaches the judge filtering rather than stopping
    # at the empty-golden-set refusal that runs before it.
    _seed_golden_run(mock_task)
    comparable = EvalConfig(
        id="comparable_config",
        name="Ordinary judge",
        config_type=EvalConfigType.v2,
        properties=_llm_judge_properties([]),
        parent=mock_eval,
    )
    comparable.save_to_file()
    blocked = EvalConfig(
        id="reference_config",
        name="Reference judge",
        config_type=EvalConfigType.v2,
        properties=_llm_judge_properties(["reference_answer"]),
        parent=mock_eval,
    )
    blocked.save_to_file()

    with patch(
        "app.desktop.studio_server.eval_api.run_eval_runner_with_status"
    ) as mock_run_eval:
        mock_run_eval.return_value = StreamingResponse(
            content=iter([b"data: test\n\n"]), media_type="text/event-stream"
        )

        response = client.get(
            "/api/projects/project1/tasks/task1/evals/eval1/run_calibration"
        )

    assert response.status_code == 200
    eval_runner = mock_run_eval.call_args[0][0]
    assert [config.id for config in eval_runner.eval_configs] == ["comparable_config"]


@pytest.mark.asyncio
async def test_run_calibration_422s_when_every_judge_needs_reference_data(
    client, mock_task_from_id, mock_task, mock_eval
):
    """Refused before the StreamingResponse, so the status and reason are the response.

    Also before the runner, which is the point: the runner's first act would be to write
    a durable scoreless EvalRun per golden item, and nothing in the UI clears those.
    """
    mock_task_from_id.return_value = mock_task
    blocked = EvalConfig(
        id="reference_config",
        name="Reference judge",
        config_type=EvalConfigType.v2,
        properties=_llm_judge_properties(["reference_answer"]),
        parent=mock_eval,
    )
    blocked.save_to_file()
    # A golden item to skip, so "nothing was written" is a claim about the guard rather
    # than about an empty dataset.
    TaskRun(
        input="input1",
        output=TaskOutput(
            output="output1",
            rating=TaskOutputRating(value=4.0, requirement_ratings={}),
        ),
        tags=["golden"],
        parent=mock_task,
    ).save_to_file()

    with patch(
        "app.desktop.studio_server.eval_api.run_eval_runner_with_status"
    ) as mock_run_eval:
        response = client.get(
            "/api/projects/project1/tasks/task1/evals/eval1/run_calibration"
        )

    assert response.status_code == 422
    message = response.json()["message"]
    assert "Test Eval" in message
    assert "compared" in message
    assert "reference data" in message
    assert "each golden dataset item as itself" in message

    mock_run_eval.assert_not_called()
    assert list(blocked.runs()) == []


@pytest.mark.asyncio
async def test_set_current_eval_config(
    client, mock_task_from_id, mock_task, mock_eval, mock_eval_config
):
    """Test setting the current eval config for an evaluation."""
    mock_task_from_id.return_value = mock_task

    # Get the eval before updating to verify the change
    response = client.get("/api/projects/project1/tasks/task1/evals/eval1")
    assert response.status_code == 200
    eval_before = response.json()

    # The current_config_id might be None or different initially
    initial_config_id = eval_before.get("current_config_id")
    assert initial_config_id is None

    # Set the current eval config
    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = mock_eval
        response = client.post(
            "/api/projects/project1/tasks/task1/evals/eval1/set_current_eval_config/eval_config1"
        )
        assert response.status_code == 200
        updated_eval = response.json()

    # Verify the current_config_id was updated
    assert updated_eval["current_config_id"] == "eval_config1"
    assert updated_eval["id"] == "eval1"

    # Verify the change persists by fetching the eval again
    eval_from_disk = mock_task.evals()[0]
    assert eval_from_disk.current_config_id == "eval_config1"


def test_delete_eval_success(client, mock_task_from_id, mock_eval, mock_task):
    assert len(mock_task.evals()) == 1
    # Set up the mock eval to be returned by eval_from_id
    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = mock_eval

        # Make the delete request
        response = client.delete("/api/projects/project1/tasks/task1/evals/eval1")

    # Verify the response
    assert response.status_code == 200

    # Verify that eval_from_id was called with the correct parameters
    mock_eval_from_id.assert_called_once_with("project1", "task1", "eval1")

    # Verify that the eval was deleted
    assert len(mock_task.evals()) == 0


def test_delete_eval_not_found(client):
    # Set up the patch for eval_from_id to raise an HTTPException
    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.side_effect = HTTPException(
            status_code=404, detail="Eval not found. ID: nonexistent_eval"
        )

        # Make the delete request
        response = client.delete(
            "/api/projects/project1/tasks/task1/evals/nonexistent_eval"
        )

    # Verify the response
    assert response.status_code == 404
    assert response.json()["message"] == "Eval not found. ID: nonexistent_eval"


async def test_create_eval_then_delete_on_spec_failure(
    client, mock_task_from_id, mock_task
):
    create_request = {
        "name": "Test Eval for Spec",
        "description": "Test eval that will be cleaned up",
        "template": None,
        "output_scores": [
            {
                "name": "tone",
                "type": "pass_fail",
                "instruction": "Evaluate tone",
            }
        ],
        "eval_set_filter_id": "tag::test_tag",
        "eval_configs_filter_id": "tag::test_tag_golden",
        "template_properties": None,
        "evaluation_data_type": "final_answer",
    }

    response = client.post(
        "/api/projects/project1/tasks/task1/create_evaluator", json=create_request
    )

    assert response.status_code == 200
    eval_data = response.json()
    eval_id = eval_data["id"]

    assert len(mock_task.evals()) == 1

    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        created_eval = mock_task.evals()[0]
        mock_eval_from_id.return_value = created_eval

        delete_response = client.delete(
            f"/api/projects/project1/tasks/task1/evals/{eval_id}"
        )

    assert delete_response.status_code == 200
    assert len(mock_task.evals()) == 0


def test_update_eval_name_and_description(
    client, mock_task_from_id, mock_eval, mock_task
):
    """Test that update_eval successfully updates name and description."""
    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = mock_eval

        update_request = {
            "name": "Updated Eval Name",
            "description": "Updated description",
        }

        response = client.patch(
            "/api/projects/project1/tasks/task1/evals/eval1",
            json=update_request,
        )

    assert response.status_code == 200
    updated_eval = response.json()
    assert updated_eval["name"] == "Updated Eval Name"
    assert updated_eval["description"] == "Updated description"

    # Verify the eval was saved
    eval_from_disk = mock_task.evals()[0]
    assert eval_from_disk.name == "Updated Eval Name"
    assert eval_from_disk.description == "Updated description"


def test_update_eval_train_set_filter_id_when_none(
    client, mock_task_from_id, mock_eval, mock_task
):
    """update_eval sets a train split on an eval that has none."""
    assert "train" not in mock_eval.splits

    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = mock_eval

        update_request = {
            "train_set_filter_id": "tag::train_my_eval",
        }

        response = client.patch(
            "/api/projects/project1/tasks/task1/evals/eval1",
            json=update_request,
        )

    assert response.status_code == 200
    updated_eval = response.json()
    # `splits` is the only home: the response carries the new split there, and the
    # deprecated flat field stays null.
    assert updated_eval["splits"]["train"] == {
        "source": "task_run",
        "filter_id": "tag::train_my_eval",
    }
    assert updated_eval["train_set_filter_id"] is None

    # Verify the eval was saved
    eval_from_disk = mock_task.evals()[0]
    assert eval_from_disk.splits["train"] == TaskRunSplit(
        filter_id="tag::train_my_eval"
    )


def test_update_eval_train_set_filter_id_when_already_set(
    client, mock_task_from_id, mock_eval
):
    """Test that update_eval raises error when trying to change an existing train split."""
    mock_eval.splits["train"] = TaskRunSplit(filter_id="tag::existing_train_set")

    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = mock_eval

        update_request = {
            "train_set_filter_id": "tag::new_train_set",
        }

        response = client.patch(
            "/api/projects/project1/tasks/task1/evals/eval1",
            json=update_request,
        )

    assert response.status_code == 400
    assert (
        "Train set filter is already set and cannot be changed"
        in response.json()["message"]
    )


def test_update_eval_partial_update(client, mock_task_from_id, mock_eval, mock_task):
    """Test that update_eval only updates provided fields."""
    original_name = mock_eval.name
    original_description = mock_eval.description
    assert "train" not in mock_eval.splits

    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = mock_eval

        # Only update train_set_filter_id
        update_request = {
            "train_set_filter_id": "tag::train_set",
        }

        response = client.patch(
            "/api/projects/project1/tasks/task1/evals/eval1",
            json=update_request,
        )

    assert response.status_code == 200
    updated_eval = response.json()

    # Name and description should remain unchanged
    assert updated_eval["name"] == original_name
    assert updated_eval["description"] == original_description
    # the train split should be updated, in `splits` — the only home
    assert updated_eval["splits"]["train"] == {
        "source": "task_run",
        "filter_id": "tag::train_set",
    }
    assert updated_eval["train_set_filter_id"] is None


def test_update_eval_not_found(client):
    """Test that update_eval returns 404 when eval is not found."""
    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.side_effect = HTTPException(
            status_code=404, detail="Eval not found. ID: nonexistent_eval"
        )

        update_request = {
            "name": "Updated Name",
        }

        response = client.patch(
            "/api/projects/project1/tasks/task1/evals/nonexistent_eval",
            json=update_request,
        )

    assert response.status_code == 404
    assert "Eval not found" in response.json()["message"]


def test_update_eval_empty_request(client, mock_task_from_id, mock_eval, mock_task):
    """Test that update_eval succeeds with empty request (no fields to update)."""
    original_name = mock_eval.name
    original_description = mock_eval.description
    original_splits = dict(mock_eval.splits)

    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = mock_eval

        # Empty update request
        update_request = {}

        response = client.patch(
            "/api/projects/project1/tasks/task1/evals/eval1",
            json=update_request,
        )

    assert response.status_code == 200
    updated_eval = response.json()

    # All fields should remain unchanged
    assert updated_eval["name"] == original_name
    assert updated_eval["description"] == original_description
    assert updated_eval["splits"] == {
        name: split.model_dump() for name, split in original_splits.items()
    }


def test_update_eval_rejects_invalid_train_set_filter_id(
    client, mock_task_from_id, mock_eval, mock_task
):
    """train_set_filter_id is typed on the request, so a malformed filter id is
    a 422 at validation rather than a 500 when the split model rejects it
    inside the handler."""
    response = client.patch(
        "/api/projects/project1/tasks/task1/evals/eval1",
        json={"train_set_filter_id": "not_a_filter_id"},
    )

    assert response.status_code == 422


def test_runs_in_filter():
    # Create a mock task with runs
    mock_task = Mock(spec=Task)

    # Create task runs with different tags
    run1 = Mock(spec=TaskRun, id="run1")
    run2 = Mock(spec=TaskRun, id="run2")
    run3 = Mock(spec=TaskRun, id="run3")

    mock_task.runs.return_value = [run1, run2, run3]

    # Mock the dataset filter
    mock_filter = Mock()

    # Configure the filter to include only run1 and run3
    mock_filter.side_effect = lambda run: run.id in ["run1", "run3"]

    # Mock the dataset_filter_from_id function
    with patch(
        "app.desktop.studio_server.eval_api.dataset_filter_from_id"
    ) as mock_dataset_filter_from_id:
        mock_dataset_filter_from_id.return_value = mock_filter

        # Call the function under test
        from app.desktop.studio_server.eval_api import runs_in_filter

        result = runs_in_filter(mock_task, "tag::some_filter", readonly=True)

        # Verify the results
        assert len(result) == 2
        assert result[0].id == "run1"
        assert result[1].id == "run3"

        # Verify the filter was called for each run
        assert mock_filter.call_count == 3
        mock_dataset_filter_from_id.assert_called_once_with("tag::some_filter")


def test_build_score_key_to_task_requirement_id():
    # Create a mock task with requirements
    mock_task = Mock(spec=Task)

    # Create task requirements with different names
    req1 = Mock(spec=TaskRequirement)
    req1.id = "req_id_1"
    req1.name = "First Requirement"

    req2 = Mock(spec=TaskRequirement)
    req2.id = "req_id_2"
    req2.name = "Second Requirement"

    req3 = Mock(spec=TaskRequirement)
    req3.id = "req_id_3"
    req3.name = "Third-With-Hyphens"

    mock_task.requirements = [req1, req2, req3]

    # Mock the string_to_json_key function
    with patch(
        "app.desktop.studio_server.eval_api.string_to_json_key"
    ) as mock_string_to_json_key:
        # Configure the mock to convert spaces to underscores and lowercase
        mock_string_to_json_key.side_effect = lambda name: (
            name.lower().replace(" ", "_").replace("-", "_")
        )

        # Call the function under test
        from app.desktop.studio_server.eval_api import (
            build_score_key_to_task_requirement_id,
        )

        result = build_score_key_to_task_requirement_id(mock_task)

        # Verify the results
        assert len(result) == 3
        assert result["first_requirement"] == "req_id_1"
        assert result["second_requirement"] == "req_id_2"
        assert result["third_with_hyphens"] == "req_id_3"

        # Verify string_to_json_key was called for each requirement
        assert mock_string_to_json_key.call_count == 3
        mock_string_to_json_key.assert_any_call("First Requirement")
        mock_string_to_json_key.assert_any_call("Second Requirement")
        mock_string_to_json_key.assert_any_call("Third-With-Hyphens")


@pytest.mark.asyncio
async def test_get_eval_progress(client, mock_task_from_id, mock_task, mock_eval):
    mock_task_from_id.return_value = mock_task

    # Create runs for testing
    run1 = TaskRun(
        input="input1",
        output=TaskOutput(
            output="output1",
            rating=TaskOutputRating(
                value=4.0,  # Has overall rating
                requirement_ratings={
                    "req_id": RequirementRating(
                        value=3.0, type=TaskOutputRatingType.five_star
                    )
                },  # Has requirement rating
            ),
        ),
        tags=["golden"],
        parent=mock_task,
    )

    run2 = TaskRun(
        input="input2",
        output=TaskOutput(
            output="output2",
            rating=TaskOutputRating(
                value=5.0,  # Has overall rating
                requirement_ratings={},  # Missing requirement rating
            ),
        ),
        tags=["golden"],
        parent=mock_task,
    )

    run3 = TaskRun(
        input="input3",
        output=TaskOutput(
            output="output3",
            rating=None,  # No ratings at all
        ),
        tags=["golden"],
        parent=mock_task,
    )

    # Mock the necessary functions
    with (
        patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id,
        patch_resolve_split(
            test=stub_split({"run1", "run2", "run3", "run4"})
        ) as mock_resolve_split,
        patch(
            "app.desktop.studio_server.eval_api.runs_in_filter"
        ) as mock_runs_in_filter,
        patch(
            "app.desktop.studio_server.eval_api.build_score_key_to_task_requirement_id"
        ) as mock_build_score_key,
        patch(
            "app.desktop.studio_server.eval_api.count_human_evals"
        ) as mock_count_human_evals,
    ):
        mock_eval_from_id.return_value = mock_eval
        mock_runs_in_filter.return_value = [run1, run2, run3]
        mock_build_score_key.return_value = {"score1": "req_id"}
        mock_count_human_evals.return_value = (
            1,
            1,
            1,
        )  # fully_rated, partially_rated, not_rated

        # Call the endpoint
        response = client.get("/api/projects/project1/tasks/task1/evals/eval1/progress")

        # Verify the response
        assert response.status_code == 200
        result = response.json()

        assert result["dataset_size"] == 4
        assert result["golden_dataset_size"] == 3
        assert result["golden_dataset_fully_rated_count"] == 1
        assert result["golden_dataset_partially_rated_count"] == 1
        assert result["golden_dataset_not_rated_count"] == 1
        assert result["current_eval_method"] is None
        # No train or val split on this eval, so zero rather than an absent field.
        assert result["train_dataset_size"] == 0
        assert result["val_dataset_size"] == 0

        # Verify the function calls
        mock_eval_from_id.assert_called_once_with("project1", "task1", "eval1")
        # Which splits it asks for, not the order it asks in: resolving val before train
        # is not a behavior change, and TestEvalProgressSplitSizes already pins the
        # outcome against real data.
        assert {c.args[2] for c in mock_resolve_split.call_args_list} == {
            "test",
            "train",
            "val",
        }
        mock_runs_in_filter.assert_called_once_with(
            mock_task, mock_eval.eval_configs_filter_id, readonly=True
        )
        mock_build_score_key.assert_called_once_with(mock_task)
        mock_count_human_evals.assert_called_once_with(
            [run1, run2, run3], mock_eval, {"score1": "req_id"}
        )


PROGRESS_PATH = "/api/projects/project1/tasks/task1/evals/eval1/progress"


class TestEvalProgressSplitSizes:
    """Every split size is the real count in that split's own store (spec 6.1)."""

    def test_reports_each_splits_own_size(
        self, client, mock_task_from_id, mock_task, mock_eval, data_source
    ):
        mock_eval.set_split("train", TaskRunSplit(filter_id="tag::train_set"))
        mock_eval.splits["val"] = TaskRunSplit(filter_id="tag::val_set")
        mock_eval.save_to_file()
        for tag in ("eval_set", "eval_set", "train_set", "train_set", "val_set"):
            _tagged_task_run(mock_task, data_source, tag)

        result = client.get(PROGRESS_PATH).json()

        assert result["dataset_size"] == 2
        assert result["train_dataset_size"] == 2
        assert result["val_dataset_size"] == 1

    def test_absent_splits_are_zero_not_absent(
        self, client, mock_task_from_id, mock_task, mock_eval, data_source
    ):
        _tagged_task_run(mock_task, data_source, "eval_set")

        result = client.get(PROGRESS_PATH).json()

        assert result["dataset_size"] == 1
        assert result["train_dataset_size"] == 0
        assert result["val_dataset_size"] == 0

    def test_counts_an_eval_input_backed_eval_rather_than_refusing_it(
        self, client, mock_task_from_id, mock_task, mock_eval
    ):
        """The 400 that stood here was never about golden sets — it fired because this
        endpoint could only count TaskRuns. Golden is legitimately 0 for a V2 eval."""
        mock_eval.splits["test"] = EvalInputSplit(filter_id="tag::inputs")
        mock_eval.eval_set_filter_id = None
        # No golden set, the expected V2 state: template=None is what lets an eval
        # validate without one.
        mock_eval.template = None
        mock_eval.eval_configs_filter_id = None
        mock_eval.save_to_file()
        _tagged_eval_input(mock_task, "inputs")
        _tagged_eval_input(mock_task, "inputs")
        _tagged_eval_input(mock_task, "other")

        response = client.get(PROGRESS_PATH)

        assert response.status_code == 200
        result = response.json()
        assert result["dataset_size"] == 2
        assert result["golden_dataset_size"] == 0

    def test_an_eval_input_backed_split_matching_nothing_is_zero_not_an_error(
        self, client, mock_task_from_id, mock_task, mock_eval
    ):
        """The other backing's empty case. A configured split that matches nothing is a
        real, empty split — the same 0 an absent split reports, but reached by resolving
        rather than by not finding one."""
        mock_eval.set_split("test", EvalInputSplit(filter_id="tag::inputs"))
        mock_eval.save_to_file()
        _tagged_eval_input(mock_task, "some_other_tag")

        response = client.get(PROGRESS_PATH)

        assert response.status_code == 200
        assert response.json()["dataset_size"] == 0


class TestSplitSize:
    def test_absent_split_is_zero(self):
        assert split_size(None) == 0

    def test_an_empty_split_is_also_zero(self):
        """Both answer 0, but only because the absence check is `is None`:
        ResolvedSplit defines __len__, so an empty split is falsy."""
        assert split_size(stub_split(set())) == 0

    def test_a_populated_split_is_its_length(self):
        assert split_size(stub_split({"a", "b"})) == 2


class TestScoreSummarySplits:
    def test_summarizes_an_eval_input_backed_test_split(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
    ):
        mock_eval.splits["test"] = EvalInputSplit(filter_id="tag::inputs")
        mock_eval.eval_set_filter_id = None
        mock_eval.save_to_file()
        eval_input = _tagged_eval_input(mock_task, "inputs")
        _scored(mock_eval_config, eval_input_id=eval_input.id)

        response = client.get(
            "/api/projects/project1/tasks/task1/evals/eval1"
            "/eval_config/eval_config1/score_summary"
        )

        assert response.status_code == 200
        body = response.json()
        assert body["dataset_size"] == 1
        assert body["results"]["run_config1"]["score1"]["n_used"] == 1

    @pytest.mark.parametrize(
        "test_split_factory",
        [
            lambda: TaskRunSplit(filter_id="tag::eval_set"),
            lambda: EvalInputSplit(filter_id="tag::inputs"),
        ],
        ids=["task_run", "eval_input"],
    )
    def test_400s_on_an_empty_test_split(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        test_split_factory,
    ):
        """A split whose filter matches nothing is empty in either store, and refused the
        same way. The model layer already parametrizes this; the API layer only ever
        exercised the TaskRun backing, so a guard written as "no task runs" rather than
        "no items" would have passed. Nothing is tagged for either filter, so both splits
        resolve to zero items."""
        mock_eval.set_split("test", test_split_factory())
        mock_eval.save_to_file()

        response = client.get(
            "/api/projects/project1/tasks/task1/evals/eval1"
            "/eval_config/eval_config1/score_summary"
        )

        assert response.status_code == 400
        assert "test split is empty" in response.json()["message"]

    def test_does_not_average_another_stores_score_into_a_split(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
    ):
        eval_input = _tagged_eval_input(mock_task, "inputs", id="500000000002")
        colliding_task_run = TaskRun(
            id=eval_input.id,
            parent=mock_task,
            input="in",
            input_source=DataSource(
                type=DataSourceType.human, properties={"created_by": "test"}
            ),
            output=TaskOutput(output="out"),
            tags=["inputs"],
        )
        colliding_task_run.save_to_file()
        mock_eval.splits["test"] = EvalInputSplit(filter_id="tag::inputs")
        mock_eval.eval_set_filter_id = None
        mock_eval.save_to_file()
        _scored(mock_eval_config, dataset_id=colliding_task_run.id)

        body = client.get(
            "/api/projects/project1/tasks/task1/evals/eval1"
            "/eval_config/eval_config1/score_summary"
        ).json()

        assert body["dataset_size"] == 1
        assert body["results"] == {}


class TestRunConfigEvalScoresSplits:
    def test_scores_an_eval_input_backed_eval(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
    ):
        mock_eval.splits["test"] = EvalInputSplit(filter_id="tag::inputs")
        mock_eval.eval_set_filter_id = None
        mock_eval.current_config_id = mock_eval_config.id
        mock_eval.save_to_file()
        eval_input = _tagged_eval_input(mock_task, "inputs")
        _tagged_eval_input(mock_task, "inputs")
        _scored(mock_eval_config, eval_input_id=eval_input.id)

        response = client.get(
            "/api/projects/project1/tasks/task1/run_configs/run_config1/eval_scores"
        )

        assert response.status_code == 200
        eval_result = response.json()["eval_results"][0]
        assert eval_result["dataset_size"] == 2
        assert eval_result["eval_config_result"]["percent_complete"] == pytest.approx(
            0.5
        )
        assert eval_result["eval_config_result"]["results"]["score1"]["n_used"] == 1

    def test_does_not_credit_a_task_run_to_an_eval_inputs_id(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
    ):
        """A TaskRun-sourced result must not count toward an EvalInput-backed split's
        progress just because the two items happen to share an id."""
        eval_input = _tagged_eval_input(mock_task, "inputs", id="500000000003")
        colliding_task_run = TaskRun(
            id=eval_input.id,
            parent=mock_task,
            input="in",
            input_source=DataSource(
                type=DataSourceType.human, properties={"created_by": "test"}
            ),
            output=TaskOutput(output="out"),
            tags=["inputs"],
        )
        colliding_task_run.save_to_file()
        mock_eval.splits["test"] = EvalInputSplit(filter_id="tag::inputs")
        mock_eval.eval_set_filter_id = None
        mock_eval.current_config_id = mock_eval_config.id
        mock_eval.save_to_file()
        _scored(mock_eval_config, dataset_id=colliding_task_run.id)

        response = client.get(
            "/api/projects/project1/tasks/task1/run_configs/run_config1/eval_scores"
        )

        assert response.status_code == 200
        eval_result = response.json()["eval_results"][0]
        assert eval_result["dataset_size"] == 1
        assert eval_result["eval_config_result"]["percent_complete"] == 0.0
        assert eval_result["eval_config_result"]["results"]["score1"] is None


@pytest.mark.asyncio
async def test_get_eval_progress_eval_input_slice(client, mock_task_from_id, mock_task):
    """An EvalInput-typed eval reports its slice size from the matching
    EvalInput items — the spec page relies on this instead of a 400."""
    mock_task_from_id.return_value = mock_task

    eval = Eval(
        id="eval_input_eval",
        name="EvalInput Eval",
        output_scores=[
            EvalOutputScore(
                name="score1", instruction="desc1", type=TaskOutputRatingType.five_star
            ),
        ],
        splits={"test": EvalInputSplit(filter_id="tag::eval_slice")},
        eval_configs_filter_id="tag::golden",
        parent=mock_task,
    )
    eval.save_to_file()
    for i in range(3):
        EvalInput(
            data=MultiTurnSyntheticEvalInputData(
                first_message=UserMessage(text=f"seed {i}"),
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
            ),
            tags=["eval_slice"],
            parent=mock_task,
        ).save_to_file()
    # An input outside the slice tag is not counted.
    EvalInput(
        data=SingleTurnEvalInputData(user_message=UserMessage(text="other")),
        tags=["other"],
        parent=mock_task,
    ).save_to_file()

    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = eval
        response = client.get(
            "/api/projects/project1/tasks/task1/evals/eval_input_eval/progress"
        )

    assert response.status_code == 200
    result = response.json()
    assert result["dataset_size"] == 3
    assert result["golden_dataset_size"] == 0


@pytest.mark.asyncio
async def test_get_eval_progress_not_found(client, mock_task_from_id, mock_task):
    mock_task_from_id.return_value = mock_task

    # Mock eval_from_id to raise HTTPException
    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.side_effect = HTTPException(
            status_code=404,
            detail="Eval not found. ID: non_existent",
        )

        # Call the endpoint with non-existent eval ID
        response = client.get(
            "/api/projects/project1/tasks/task1/evals/non_existent/progress"
        )

        # Verify the response
        assert response.status_code == 404
        assert response.json()["message"] == "Eval not found. ID: non_existent"
        mock_eval_from_id.assert_called_once_with("project1", "task1", "non_existent")


@pytest.mark.asyncio
async def test_set_current_eval_config_none(
    client, mock_task_from_id, mock_task, mock_eval
):
    """Test clearing the current eval config for an evaluation by setting it to 'None'."""
    mock_task_from_id.return_value = mock_task

    # First set a non-null value to verify it can be cleared
    mock_eval.current_config_id = "some_existing_config_id"
    mock_eval.save_to_file()

    # Verify the current_config_id is set
    assert mock_task.evals()[0].current_config_id == "some_existing_config_id"

    # Clear the current eval config by setting it to "None"
    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = mock_eval
        response = client.post(
            "/api/projects/project1/tasks/task1/evals/eval1/set_current_eval_config/None"
        )
        assert response.status_code == 200
        updated_eval = response.json()

    # Verify the current_config_id was cleared (set to None)
    assert updated_eval["current_config_id"] is None
    assert updated_eval["id"] == "eval1"

    # Verify the change persists by fetching the eval again
    eval_from_disk = mock_task.evals()[0]
    assert eval_from_disk.current_config_id is None


@pytest.mark.asyncio
async def test_set_current_eval_config_not_found(
    client, mock_task_from_id, mock_task, mock_eval
):
    """Test 400 error when setting a non-existent eval config as default."""
    mock_task_from_id.return_value = mock_task

    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = mock_eval
        response = client.post(
            "/api/projects/project1/tasks/task1/evals/eval1/set_current_eval_config/non_existent_eval_config"
        )

    # Verify the response
    assert response.status_code == 400
    assert response.json()["message"] == "Eval config not found."


@pytest.mark.parametrize(
    "score_name,expected_score,has_overall_rating,has_requirement_rating,has_named_rating",
    [
        # Test overall rating
        ("overall_rating", 5.0, True, False, False),
        ("overall_rating", None, False, False, False),
        # Test task requirement rating
        ("score1", 3.0, False, True, False),
        ("score1", None, False, False, False),
        # Test named rating
        ("Named Score", 4.0, False, False, True),
        ("Named Score", None, False, False, False),
    ],
)
def test_human_score_from_task_run(
    score_name,
    expected_score,
    has_overall_rating,
    has_requirement_rating,
    has_named_rating,
):
    # Create a mock task run with the specified ratings
    task_run = Mock(spec=TaskRun)
    task_run.output = Mock(spec=TaskOutput)

    # Set up the rating object
    rating = Mock(spec=TaskOutputRating)
    rating.value = 5.0 if has_overall_rating else None

    # Set up requirement ratings
    requirement_ratings = {}
    if has_requirement_rating:
        requirement_ratings["req_id"] = RequirementRating(
            value=3.0, type=TaskOutputRatingType.five_star
        )
    if has_named_rating:
        requirement_ratings["named::Named Score"] = RequirementRating(
            value=4.0, type=TaskOutputRatingType.five_star
        )
    rating.requirement_ratings = requirement_ratings

    task_run.output.rating = (
        rating
        if (has_overall_rating or has_requirement_rating or has_named_rating)
        else None
    )

    # Create the score object
    score = EvalOutputScore(
        name=score_name, instruction="Test score", type=TaskOutputRatingType.five_star
    )

    # Create the score key to requirement ID mapping
    score_key_to_task_requirement_id: Dict[str, ID_TYPE] = {"score1": "req_id"}

    # Call the function
    from app.desktop.studio_server.eval_api import human_score_from_task_run

    result = human_score_from_task_run(
        task_run, score, score_key_to_task_requirement_id
    )

    # Verify the result
    assert result == expected_score


def test_human_score_named_rating_survives_requirement_name_collision():
    """A task requirement whose name maps to the same json_key as the score
    must not hide a named rating: spec-created evals store the human verdict
    under named::{score.name}, and the like-named requirement may be unrated."""
    task_run = Mock(spec=TaskRun)
    task_run.output = Mock(spec=TaskOutput)
    rating = Mock(spec=TaskOutputRating)
    rating.value = None
    rating.requirement_ratings = {
        # No rating under the colliding requirement's id ("req_id"); the
        # human verdict lives under the named key.
        "named::My Spec": RequirementRating(
            value=1.0, type=TaskOutputRatingType.pass_fail
        ),
    }
    task_run.output.rating = rating

    score = EvalOutputScore(
        name="My Spec", instruction="Test score", type=TaskOutputRatingType.pass_fail
    )
    # A requirement named like the score maps to the same json_key.
    score_key_to_task_requirement_id: Dict[str, ID_TYPE] = {"my_spec": "req_id"}

    from app.desktop.studio_server.eval_api import human_score_from_task_run

    result = human_score_from_task_run(
        task_run, score, score_key_to_task_requirement_id
    )

    assert result == 1.0

    # When the colliding requirement IS rated, its rating still wins.
    rating.requirement_ratings["req_id"] = RequirementRating(
        value=0.0, type=TaskOutputRatingType.pass_fail
    )
    assert (
        human_score_from_task_run(task_run, score, score_key_to_task_requirement_id)
        == 0.0
    )


@pytest.mark.asyncio
async def test_create_task_run_config_invalid_temperature_values(
    client, mock_task_from_id, mock_task
):
    """Test that invalid temperature values return 422 errors."""
    mock_task_from_id.return_value = mock_task

    # Test temperature below 0
    response = client.post(
        "/api/projects/project1/tasks/task1/run_configs",
        json={
            "name": "Test Task Run Config",
            "run_config_properties": {
                "model_name": "gpt-4o",
                "model_provider_name": "openai",
                "prompt_id": "simple_chain_of_thought_prompt_builder",
                "temperature": -0.1,
                "structured_output_mode": "json_schema",
            },
        },
    )
    assert response.status_code == 422
    error_detail = response.json()["message"]
    assert "temperature must be between 0 and 2" in str(error_detail)

    # Test temperature above 2
    response = client.post(
        "/api/projects/project1/tasks/task1/run_configs",
        json={
            "name": "Test Task Run Config",
            "run_config_properties": {
                "model_name": "gpt-4o",
                "model_provider_name": "openai",
                "prompt_id": "simple_chain_of_thought_prompt_builder",
                "temperature": 2.1,
                "structured_output_mode": "json_schema",
            },
        },
    )
    assert response.status_code == 422
    error_detail = response.json()["message"]
    assert "temperature must be between 0 and 2" in str(error_detail)


@pytest.mark.asyncio
async def test_create_task_run_config_invalid_top_p_values(
    client, mock_task_from_id, mock_task
):
    """Test that invalid top_p values return 422 errors."""
    mock_task_from_id.return_value = mock_task

    # Test top_p below 0
    response = client.post(
        "/api/projects/project1/tasks/task1/run_configs",
        json={
            "name": "Test Task Run Config",
            "run_config_properties": {
                "model_name": "gpt-4o",
                "model_provider_name": "openai",
                "prompt_id": "simple_chain_of_thought_prompt_builder",
                "top_p": -0.1,
                "structured_output_mode": "json_schema",
            },
        },
    )
    assert response.status_code == 422
    error_detail = response.json()["message"]
    assert "top_p must be between 0 and 1" in str(error_detail)

    # Test top_p above 1
    response = client.post(
        "/api/projects/project1/tasks/task1/run_configs",
        json={
            "name": "Test Task Run Config",
            "run_config_properties": {
                "model_name": "gpt-4o",
                "model_provider_name": "openai",
                "prompt_id": "simple_chain_of_thought_prompt_builder",
                "top_p": 1.1,
                "structured_output_mode": "json_schema",
            },
        },
    )
    assert response.status_code == 422
    error_detail = response.json()["message"]
    assert "top_p must be between 0 and 1" in str(error_detail)


@pytest.mark.asyncio
async def test_create_task_run_config_valid_boundary_values(
    client, mock_task_from_id, mock_task
):
    """Test that valid boundary values for temperature and top_p work correctly."""
    mock_task_from_id.return_value = mock_task

    # Test valid boundary values - temperature = 0, top_p = 0
    response = client.post(
        "/api/projects/project1/tasks/task1/run_configs",
        json={
            "name": "Test Task Run Config Min",
            "run_config_properties": {
                "model_name": "gpt-4o",
                "model_provider_name": "openai",
                "prompt_id": "simple_chain_of_thought_prompt_builder",
                "temperature": 0.0,
                "top_p": 0.0,
                "structured_output_mode": "json_schema",
            },
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["run_config_properties"]["temperature"] == 0.0
    assert result["run_config_properties"]["top_p"] == 0.0

    # Test valid boundary values - temperature = 2, top_p = 1
    response = client.post(
        "/api/projects/project1/tasks/task1/run_configs",
        json={
            "name": "Test Task Run Config Max",
            "run_config_properties": {
                "model_name": "gpt-4o",
                "model_provider_name": "openai",
                "prompt_id": "simple_chain_of_thought_prompt_builder",
                "temperature": 2.0,
                "top_p": 1.0,
                "structured_output_mode": "json_schema",
            },
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["run_config_properties"]["temperature"] == 2.0
    assert result["run_config_properties"]["top_p"] == 1.0


@pytest.mark.asyncio
async def test_get_run_config_eval_scores_with_usage(
    client, mock_task_from_id, mock_task, mock_eval, mock_eval_config, mock_run_config
):
    """Test that get_run_config_eval_scores correctly calculates mean usage statistics"""
    mock_task_from_id.return_value = mock_task

    # Create TaskRuns with usage data
    task_run_1 = TaskRun(
        input="test input 1",
        input_source=DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": "gpt-4",
                "model_provider": "openai",
                "adapter_name": "langchain_adapter",
            },
        ),
        output=TaskOutput(output="test output 1"),
        usage=Usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=0.005,
            total_llm_latency_ms=500,
        ),
        parent=mock_task,
    )
    task_run_1.save_to_file()

    task_run_2 = TaskRun(
        input="test input 2",
        input_source=DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": "gpt-4",
                "model_provider": "openai",
                "adapter_name": "langchain_adapter",
            },
        ),
        output=TaskOutput(output="test output 2"),
        usage=Usage(
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            cost=0.010,
            total_llm_latency_ms=1000,
        ),
        parent=mock_task,
    )
    task_run_2.save_to_file()

    # Create a TaskRun without usage data
    task_run_3 = TaskRun(
        input="test input 3",
        input_source=DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": "gpt-4",
                "model_provider": "openai",
                "adapter_name": "langchain_adapter",
            },
        ),
        output=TaskOutput(output="test output 3"),
        # No usage data
        parent=mock_task,
    )
    task_run_3.save_to_file()

    # Create EvalRuns for these TaskRuns
    eval_run_1 = EvalRun(
        task_run_config_id=mock_run_config.id,
        scores={"score1": 4.0, "overall_rating": 4.0},
        input="test input 1",
        output="test output 1",
        dataset_id=task_run_1.id,
        task_run_usage=task_run_1.usage,  # Copy usage from TaskRun
        parent=mock_eval_config,
    )
    eval_run_1.save_to_file()

    eval_run_2 = EvalRun(
        task_run_config_id=mock_run_config.id,
        scores={"score1": 4.5, "overall_rating": 4.5},
        input="test input 2",
        output="test output 2",
        dataset_id=task_run_2.id,
        task_run_usage=task_run_2.usage,  # Copy usage from TaskRun
        parent=mock_eval_config,
    )
    eval_run_2.save_to_file()

    eval_run_3 = EvalRun(
        task_run_config_id=mock_run_config.id,
        scores={"score1": 3.5, "overall_rating": 3.5},
        input="test input 3",
        output="test output 3",
        dataset_id=task_run_3.id,
        task_run_usage=task_run_3.usage,  # Copy usage from TaskRun (this will be None)
        parent=mock_eval_config,
    )
    eval_run_3.save_to_file()

    # Create mock objects instead of patching Pydantic models
    mock_task_for_api = MagicMock()
    mock_task_for_api.runs.return_value = [task_run_1, task_run_2, task_run_3]
    mock_task_for_api.evals.return_value = [mock_eval]

    mock_eval_config_for_api = MagicMock()
    mock_eval_config_for_api.runs.return_value = [eval_run_1, eval_run_2, eval_run_3]
    mock_eval_config_for_api.id = mock_eval_config.id

    mock_eval_for_api = MagicMock()
    mock_eval_for_api.configs.return_value = [mock_eval_config_for_api]
    mock_eval_for_api.id = mock_eval.id
    mock_eval_for_api.splits = mock_eval.splits
    mock_eval_for_api.output_scores = mock_eval.output_scores

    mock_eval.current_config_id = mock_eval_config.id

    # Patch the task_from_id to return our mock
    with (
        patch(
            "app.desktop.studio_server.eval_api.task_from_id"
        ) as mock_task_from_id_patch,
        patch(
            "app.desktop.studio_server.eval_api.eval_from_id"
        ) as mock_eval_from_id_patch,
        patch(
            "app.desktop.studio_server.eval_api.task_run_config_from_id"
        ) as mock_task_run_config_from_id_patch,
    ):
        mock_task_from_id_patch.return_value = mock_task_for_api
        mock_eval_from_id_patch.return_value = mock_eval_for_api
        mock_task_run_config_from_id_patch.return_value = mock_run_config

        with patch_resolve_split(
            test=stub_split({task_run_1.id, task_run_2.id, task_run_3.id})
        ):
            response = client.get(
                f"/api/projects/project1/tasks/task1/run_configs/{mock_run_config.id}/eval_scores"
            )

    assert response.status_code == 200
    data = response.json()

    # Verify the structure
    assert "eval_results" in data
    eval_results = data["eval_results"]
    assert len(eval_results) == 1

    eval_result = eval_results[0]
    assert "eval_config_result" in eval_result
    eval_config_result = eval_result["eval_config_result"]
    assert eval_config_result is not None
    assert eval_config_result["results"]["score1"]["mean_score"] == 4.0
    assert eval_config_result["results"]["overall_rating"]["mean_score"] == 4.0

    # Distribution over the three scores (3.5, 4.0, 4.5), linearly interpolated.
    # The mean alone cannot distinguish this from any other set summing to 12.0.
    for score_key in ("score1", "overall_rating"):
        summary = eval_config_result["results"][score_key]
        assert summary["n_used"] == 3
        assert summary["min_score"] == pytest.approx(3.5)
        assert summary["p25_score"] == pytest.approx(3.75)
        assert summary["median_score"] == pytest.approx(4.0)
        assert summary["p75_score"] == pytest.approx(4.25)
        assert summary["p90_score"] == pytest.approx(4.4)
        assert summary["max_score"] == pytest.approx(4.5)

    # Check that mean_usage is at the top level of the response
    assert "mean_usage" in data
    mean_usage = data["mean_usage"]
    assert mean_usage is not None

    # With 3 eval runs and 2 having usage data (2/3 = 66.7% > 50%),
    # all usage metrics should be included
    # Expected means: input_tokens=(100+200)/2=150, output_tokens=(50+100)/2=75,
    # total_tokens=(150+300)/2=225, cost=(0.005+0.010)/2=0.0075
    assert mean_usage["mean_input_tokens"] == 150.0
    assert mean_usage["mean_output_tokens"] == 75.0
    assert mean_usage["mean_total_tokens"] == 225.0
    assert mean_usage["mean_cost"] == 0.0075
    # Expected mean latency: (500+1000)/2 = 750.0 (2 of 3 runs have latency, 66.7% > 50%)
    assert mean_usage["mean_total_llm_latency_ms"] == 750.0


@pytest.mark.asyncio
async def test_get_run_config_eval_scores_latency_below_threshold(
    client, mock_task_from_id, mock_task, mock_eval, mock_eval_config, mock_run_config
):
    """Test that mean_total_llm_latency_ms is None when fewer than 50% of runs have latency data"""
    mock_task_from_id.return_value = mock_task

    # Create 3 TaskRuns, only 1 with latency data (1/3 = 33% < 50% threshold)
    task_run_1 = TaskRun(
        input="test input 1",
        input_source=DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": "gpt-4",
                "model_provider": "openai",
                "adapter_name": "langchain_adapter",
            },
        ),
        output=TaskOutput(output="test output 1"),
        usage=Usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=0.005,
            total_llm_latency_ms=500,
        ),
        parent=mock_task,
    )
    task_run_1.save_to_file()

    task_run_2 = TaskRun(
        input="test input 2",
        input_source=DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": "gpt-4",
                "model_provider": "openai",
                "adapter_name": "langchain_adapter",
            },
        ),
        output=TaskOutput(output="test output 2"),
        usage=Usage(
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            cost=0.010,
        ),
        parent=mock_task,
    )
    task_run_2.save_to_file()

    task_run_3 = TaskRun(
        input="test input 3",
        input_source=DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": "gpt-4",
                "model_provider": "openai",
                "adapter_name": "langchain_adapter",
            },
        ),
        output=TaskOutput(output="test output 3"),
        usage=Usage(
            input_tokens=150,
            output_tokens=75,
            total_tokens=225,
            cost=0.008,
        ),
        parent=mock_task,
    )
    task_run_3.save_to_file()

    eval_run_1 = EvalRun(
        task_run_config_id=mock_run_config.id,
        scores={"score1": 4.0, "overall_rating": 4.0},
        input="test input 1",
        output="test output 1",
        dataset_id=task_run_1.id,
        task_run_usage=task_run_1.usage,
        parent=mock_eval_config,
    )
    eval_run_1.save_to_file()

    eval_run_2 = EvalRun(
        task_run_config_id=mock_run_config.id,
        scores={"score1": 4.5, "overall_rating": 4.5},
        input="test input 2",
        output="test output 2",
        dataset_id=task_run_2.id,
        task_run_usage=task_run_2.usage,
        parent=mock_eval_config,
    )
    eval_run_2.save_to_file()

    eval_run_3 = EvalRun(
        task_run_config_id=mock_run_config.id,
        scores={"score1": 3.5, "overall_rating": 3.5},
        input="test input 3",
        output="test output 3",
        dataset_id=task_run_3.id,
        task_run_usage=task_run_3.usage,
        parent=mock_eval_config,
    )
    eval_run_3.save_to_file()

    mock_task_for_api = MagicMock()
    mock_task_for_api.runs.return_value = [task_run_1, task_run_2, task_run_3]
    mock_task_for_api.evals.return_value = [mock_eval]

    mock_eval_config_for_api = MagicMock()
    mock_eval_config_for_api.runs.return_value = [eval_run_1, eval_run_2, eval_run_3]
    mock_eval_config_for_api.id = mock_eval_config.id

    mock_eval_for_api = MagicMock()
    mock_eval_for_api.configs.return_value = [mock_eval_config_for_api]
    mock_eval_for_api.id = mock_eval.id
    mock_eval_for_api.splits = mock_eval.splits
    mock_eval_for_api.output_scores = mock_eval.output_scores

    mock_eval.current_config_id = mock_eval_config.id

    with (
        patch(
            "app.desktop.studio_server.eval_api.task_from_id"
        ) as mock_task_from_id_patch,
        patch(
            "app.desktop.studio_server.eval_api.eval_from_id"
        ) as mock_eval_from_id_patch,
        patch(
            "app.desktop.studio_server.eval_api.task_run_config_from_id"
        ) as mock_task_run_config_from_id_patch,
    ):
        mock_task_from_id_patch.return_value = mock_task_for_api
        mock_eval_from_id_patch.return_value = mock_eval_for_api
        mock_task_run_config_from_id_patch.return_value = mock_run_config

        with patch_resolve_split(
            test=stub_split({task_run_1.id, task_run_2.id, task_run_3.id})
        ):
            response = client.get(
                f"/api/projects/project1/tasks/task1/run_configs/{mock_run_config.id}/eval_scores"
            )

    assert response.status_code == 200
    data = response.json()
    mean_usage = data["mean_usage"]
    assert mean_usage is not None

    # Cost/tokens should be present (3/3 = 100% > 50%)
    assert mean_usage["mean_cost"] is not None
    # Latency should be None (only 1/3 = 33% < 50% threshold)
    assert mean_usage["mean_total_llm_latency_ms"] is None


EVAL_SCORES_PATH = (
    "/api/projects/project1/tasks/task1/run_configs/run_config1/eval_scores"
)


class TestRunConfigUsageRollup:
    """The evaluated task's usage moved onto the scored TaskRun.

    Reading `eval_run.task_run_usage` alone would report zero usage for every eval run
    since the split, because new records leave that field None.
    """

    def _mean_usage(self, client) -> Dict:
        response = client.get(EVAL_SCORES_PATH)
        assert response.status_code == 200
        return response.json()["mean_usage"]

    def test_reports_the_scored_task_runs_usage_for_a_pointer_record(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
        data_source,
    ):
        """Including latency - the field cumulative_usage would have dropped."""
        item = _tagged_task_run(mock_task, data_source, "eval_set")
        trace = _eval_trace(
            mock_task,
            data_source,
            EvalItemSource(source_type="task_run", source_id=item.id),
        )
        trace.usage = Usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=0.005,
            total_llm_latency_ms=500,
        )
        trace.save_to_file()
        _pointer_scored(mock_eval_config, trace.id, dataset_id=item.id)

        mean_usage = self._mean_usage(client)

        assert mean_usage["mean_input_tokens"] == 100.0
        assert mean_usage["mean_output_tokens"] == 50.0
        assert mean_usage["mean_total_tokens"] == 150.0
        assert mean_usage["mean_cost"] == 0.005
        assert mean_usage["mean_total_llm_latency_ms"] == 500.0

    def test_still_reports_task_run_usage_for_a_legacy_record(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
        data_source,
    ):
        item = _tagged_task_run(mock_task, data_source, "eval_set")
        legacy = EvalRun(
            parent=mock_eval_config,
            task_run_config_id="run_config1",
            scores={"score1": 3.0, "overall_rating": 1.0},
            dataset_id=item.id,
            input="in",
            output="out",
            task_run_usage=Usage(
                input_tokens=40,
                output_tokens=20,
                total_tokens=60,
                cost=0.001,
                total_llm_latency_ms=300,
            ),
        )
        legacy.save_to_file()

        mean_usage = self._mean_usage(client)

        assert mean_usage["mean_input_tokens"] == 40.0
        assert mean_usage["mean_total_llm_latency_ms"] == 300.0

    def test_a_pointer_record_with_a_missing_trace_contributes_nothing(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
        data_source,
    ):
        """It must not count as a zero, which would drag the average down."""
        scored_item = _tagged_task_run(mock_task, data_source, "eval_set")
        dangling_item = _tagged_task_run(mock_task, data_source, "eval_set")
        trace = _eval_trace(
            mock_task,
            data_source,
            EvalItemSource(source_type="task_run", source_id=scored_item.id),
        )
        trace.usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        trace.save_to_file()
        _pointer_scored(mock_eval_config, trace.id, dataset_id=scored_item.id)
        _pointer_scored(mock_eval_config, "900000000003", dataset_id=dangling_item.id)

        mean_usage = self._mean_usage(client)

        assert mean_usage["mean_input_tokens"] == 100.0
        assert mean_usage["mean_total_tokens"] == 150.0


@pytest.mark.asyncio
async def test_get_run_config_eval_scores_inline_aggregation(
    client, mock_task_from_id, mock_task, mock_eval, mock_eval_config, mock_run_config
):
    """Verify the inline aggregation path returns correct n_used, n_excluded, and percent_complete."""
    mock_task_from_id.return_value = mock_task

    task_runs = []
    for i in range(3):
        tr = TaskRun(
            input=f"input {i}",
            input_source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "gpt-4",
                    "model_provider": "openai",
                    "adapter_name": "test",
                },
            ),
            output=TaskOutput(output=f"output {i}"),
            parent=mock_task,
        )
        tr.save_to_file()
        task_runs.append(tr)

    scored_run = EvalRun(
        task_run_config_id=mock_run_config.id,
        scores={"score1": 3.0, "overall_rating": 4.0},
        input="input 0",
        output="output 0",
        dataset_id=task_runs[0].id,
        parent=mock_eval_config,
    )
    scored_run.save_to_file()

    skipped_run = EvalRun(
        task_run_config_id=mock_run_config.id,
        scores={},
        input="input 1",
        output="output 1",
        dataset_id=task_runs[1].id,
        skipped_reason="extraction_failed",
        parent=mock_eval_config,
    )
    skipped_run.save_to_file()

    mock_task_api = MagicMock()
    mock_task_api.runs.return_value = task_runs
    mock_task_api.evals.return_value = [mock_eval]
    mock_task_api.specs.return_value = []

    mock_ec_api = MagicMock()
    mock_ec_api.runs.return_value = [scored_run, skipped_run]
    mock_ec_api.id = mock_eval_config.id

    mock_eval_api = MagicMock()
    mock_eval_api.configs.return_value = [mock_ec_api]
    mock_eval_api.id = mock_eval.id
    mock_eval_api.splits = mock_eval.splits
    mock_eval_api.output_scores = mock_eval.output_scores
    mock_eval_api.name = mock_eval.name

    mock_eval.current_config_id = mock_eval_config.id

    with (
        patch("app.desktop.studio_server.eval_api.task_from_id") as p_task,
        patch("app.desktop.studio_server.eval_api.eval_from_id") as p_eval,
        patch("app.desktop.studio_server.eval_api.task_run_config_from_id") as p_rc,
        patch_resolve_split(test=stub_split({tr.id for tr in task_runs})),
    ):
        p_task.return_value = mock_task_api
        p_eval.return_value = mock_eval_api
        p_rc.return_value = mock_run_config

        response = client.get(
            f"/api/projects/project1/tasks/task1/run_configs/{mock_run_config.id}/eval_scores"
        )

    assert response.status_code == 200
    data = response.json()
    er = data["eval_results"][0]
    ecr = er["eval_config_result"]

    assert ecr["n_excluded"] == 1
    assert ecr["results"]["score1"]["n_used"] == 1
    assert ecr["results"]["score1"]["n_excluded"] == 1
    assert ecr["results"]["score1"]["mean_score"] == 3.0
    assert ecr["results"]["overall_rating"]["n_used"] == 1
    assert ecr["results"]["overall_rating"]["n_excluded"] == 1
    assert ecr["results"]["overall_rating"]["mean_score"] == 4.0
    assert ecr["percent_complete"] == pytest.approx(2.0 / 3.0)


@pytest.mark.asyncio
async def test_get_run_config_eval_scores_all_skipped(
    client, mock_task_from_id, mock_task, mock_eval, mock_eval_config, mock_run_config
):
    """When every EvalRun is skipped, mean_score should be None and n_used == 0."""
    mock_task_from_id.return_value = mock_task

    task_runs = []
    for i in range(2):
        tr = TaskRun(
            input=f"input {i}",
            input_source=DataSource(
                type=DataSourceType.synthetic,
                properties={
                    "model_name": "gpt-4",
                    "model_provider": "openai",
                    "adapter_name": "test",
                },
            ),
            output=TaskOutput(output=f"output {i}"),
            parent=mock_task,
        )
        tr.save_to_file()
        task_runs.append(tr)

    skipped_runs = [
        EvalRun(
            task_run_config_id=mock_run_config.id,
            scores={},
            input=f"input {i}",
            output=f"output {i}",
            dataset_id=task_runs[i].id,
            skipped_reason="incompatible_input_shape",
            parent=mock_eval_config,
        )
        for i in range(2)
    ]
    for sr in skipped_runs:
        sr.save_to_file()

    mock_task_api = MagicMock()
    mock_task_api.runs.return_value = task_runs
    mock_task_api.evals.return_value = [mock_eval]
    mock_task_api.specs.return_value = []

    mock_ec_api = MagicMock()
    mock_ec_api.runs.return_value = skipped_runs
    mock_ec_api.id = mock_eval_config.id

    mock_eval_api = MagicMock()
    mock_eval_api.configs.return_value = [mock_ec_api]
    mock_eval_api.id = mock_eval.id
    mock_eval_api.splits = mock_eval.splits
    mock_eval_api.output_scores = mock_eval.output_scores
    mock_eval_api.name = mock_eval.name

    mock_eval.current_config_id = mock_eval_config.id

    with (
        patch("app.desktop.studio_server.eval_api.task_from_id") as p_task,
        patch("app.desktop.studio_server.eval_api.eval_from_id") as p_eval,
        patch("app.desktop.studio_server.eval_api.task_run_config_from_id") as p_rc,
        patch_resolve_split(test=stub_split({tr.id for tr in task_runs})),
    ):
        p_task.return_value = mock_task_api
        p_eval.return_value = mock_eval_api
        p_rc.return_value = mock_run_config

        response = client.get(
            f"/api/projects/project1/tasks/task1/run_configs/{mock_run_config.id}/eval_scores"
        )

    assert response.status_code == 200
    data = response.json()
    er = data["eval_results"][0]
    ecr = er["eval_config_result"]

    assert ecr["n_excluded"] == 2
    assert ecr["results"]["score1"]["n_used"] == 0
    assert ecr["results"]["score1"]["n_excluded"] == 2
    assert ecr["results"]["score1"]["mean_score"] is None
    assert ecr["results"]["overall_rating"]["n_used"] == 0
    assert ecr["results"]["overall_rating"]["n_excluded"] == 2
    assert ecr["results"]["overall_rating"]["mean_score"] is None
    # Percentiles follow the mean: None, not 0.0, when nothing was scored.
    for score_key in ("score1", "overall_rating"):
        assert ecr["results"][score_key]["median_score"] is None
        assert ecr["results"][score_key]["p90_score"] is None
        assert ecr["results"][score_key]["min_score"] is None
        assert ecr["results"][score_key]["max_score"] is None
    assert ecr["percent_complete"] == 1.0


@pytest.mark.asyncio
async def test_get_run_config_eval_scores_includes_eval_input_evals(
    client, mock_task_from_id, mock_task
):
    """EvalInput-typed evals appear in a run config's eval scores with real
    sizing and completion instead of being silently omitted."""
    mock_task_from_id.return_value = mock_task

    eval = Eval(
        id="eval_input_eval",
        name="EvalInput Eval",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                instruction="Test accuracy",
                type=TaskOutputRatingType.pass_fail,
            ),
        ],
        splits={"test": EvalInputSplit(filter_id="tag::eval_slice")},
        eval_configs_filter_id="tag::golden",
        current_config_id="ec1",
        parent=mock_task,
    )
    eval.save_to_file()
    eval_config = EvalConfig(
        id="ec1",
        name="Judge",
        config_type=EvalConfigType.g_eval,
        properties={"eval_steps": ["step1"]},
        model_name="gpt-4",
        model_provider="openai",
        parent=eval,
    )
    eval_config.save_to_file()

    eval_input_ids = []
    for i in range(2):
        eval_input = EvalInput(
            data=MultiTurnSyntheticEvalInputData(
                first_message=UserMessage(text=f"seed {i}"),
                synthetic_user_info=SyntheticUserInfo(persona="p", goal="g"),
            ),
            tags=["eval_slice"],
            parent=mock_task,
        )
        eval_input.save_to_file()
        eval_input_ids.append(eval_input.id)

    run_config = TaskRunConfig(
        parent=mock_task,
        id="rc1",
        name="Run Config 1",
        run_config_properties=KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name=ModelProviderName.openai,
            prompt_id="simple_chain_of_thought_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
        ),
    )
    run_config.save_to_file()

    for eval_input_id, score in zip(eval_input_ids, [1.0, 0.0]):
        EvalRun(
            task_run_config_id="rc1",
            scores={"accuracy": score},
            input="input",
            output="output",
            eval_input_id=eval_input_id,
            parent=eval_config,
        ).save_to_file()

    response = client.get(
        "/api/projects/project1/tasks/task1/run_configs/rc1/eval_scores"
    )

    assert response.status_code == 200
    data = response.json()
    eval_result = next(
        (er for er in data["eval_results"] if er["eval_id"] == "eval_input_eval"),
        None,
    )
    assert eval_result is not None, "EvalInput eval missing from eval_scores"
    assert eval_result["dataset_size"] == 2
    ecr = eval_result["eval_config_result"]
    assert ecr["results"]["accuracy"]["mean_score"] == pytest.approx(0.5)
    assert ecr["results"]["accuracy"]["n_used"] == 2
    assert ecr["percent_complete"] == 1.0


def test_get_eval_configs_score_summary_no_filter_id(
    client, mock_task, mock_task_from_id
):
    """Test that get_eval_configs_score_summary returns 400 when eval_configs_filter_id is None"""
    mock_task_from_id.return_value = mock_task

    # Create an eval with eval_configs_filter_id set to None
    # Only RAG template allows eval_configs_filter_id to be None
    eval_without_filter = Eval(
        id="eval1",
        name="Test Eval",
        description="Test Description",
        template=EvalTemplateId.rag,
        output_scores=[
            EvalOutputScore(
                name="score1", instruction="desc1", type=TaskOutputRatingType.five_star
            ),
        ],
        eval_set_filter_id="tag::eval_set",
        eval_configs_filter_id=None,
        parent=mock_task,
    )
    eval_without_filter.save_to_file()

    with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id:
        mock_eval_from_id.return_value = eval_without_filter

        response = client.get(
            "/api/projects/project1/tasks/task1/evals/eval1/eval_configs_score_summary"
        )

        assert response.status_code == 400
        assert (
            response.json()["message"]
            == "No eval configs filter id set, cannot get eval configs score summary."
        )
        mock_eval_from_id.assert_called_once_with("project1", "task1", "eval1")


@pytest.mark.asyncio
async def test_get_run_config_eval_scores_includes_spec_id(
    client, mock_task, mock_eval, mock_eval_config, mock_run_config
):
    """Test that get_run_config_eval_scores includes spec_id for spec-associated evals and None for legacy evals"""

    # Create a spec that references the eval
    spec = Spec(
        id="spec1",
        name="Test Spec",
        definition="Test spec definition",
        properties=DesiredBehaviourProperties(
            spec_type=SpecType.desired_behaviour,
            core_requirement="test instruction",
            desired_behaviour_description="test desired behaviour",
        ),
        eval_id=mock_eval.id,  # Associate this spec with the eval
        parent=mock_task,
    )
    spec.save_to_file()

    # Create a second eval that is NOT associated with any spec (legacy eval)
    legacy_eval = Eval(
        id="legacy_eval1",
        name="Legacy Eval",
        description="Legacy eval without spec",
        template=None,
        eval_set_filter_id="tag::legacy_eval_set",
        eval_configs_filter_id="tag::legacy_golden",
        output_scores=[
            EvalOutputScore(
                name="score1",
                instruction="desc1",
                type=TaskOutputRatingType.five_star,
            ),
        ],
        parent=mock_task,
    )
    legacy_eval.save_to_file()

    # Create an eval config for the legacy eval
    legacy_eval_config = EvalConfig(
        id="legacy_eval_config1",
        name="Legacy Eval Config",
        config_type=EvalConfigType.g_eval,
        properties={"eval_steps": ["step1", "step2"]},
        parent=legacy_eval,
        model_name="gpt-4",
        model_provider="openai",
    )
    legacy_eval_config.save_to_file()
    legacy_eval.current_config_id = legacy_eval_config.id
    legacy_eval.save_to_file()

    # Create mock objects for the API
    mock_task_for_api = MagicMock()
    mock_task_for_api.evals.return_value = [mock_eval, legacy_eval]
    mock_task_for_api.specs.return_value = [spec]

    mock_eval_config_for_api = MagicMock()
    mock_eval_config_for_api.runs.return_value = []
    mock_eval_config_for_api.id = mock_eval_config.id

    mock_eval_for_api = MagicMock()
    mock_eval_for_api.configs.return_value = [mock_eval_config_for_api]
    mock_eval_for_api.id = mock_eval.id
    mock_eval_for_api.name = mock_eval.name
    mock_eval_for_api.splits = mock_eval.splits
    mock_eval_for_api.output_scores = mock_eval.output_scores
    mock_eval_for_api.current_config_id = mock_eval_config.id

    legacy_eval_config_for_api = MagicMock()
    legacy_eval_config_for_api.runs.return_value = []
    legacy_eval_config_for_api.id = legacy_eval_config.id

    legacy_eval_for_api = MagicMock()
    legacy_eval_for_api.configs.return_value = [legacy_eval_config_for_api]
    legacy_eval_for_api.id = legacy_eval.id
    legacy_eval_for_api.name = legacy_eval.name
    legacy_eval_for_api.splits = legacy_eval.splits
    legacy_eval_for_api.output_scores = legacy_eval.output_scores
    legacy_eval_for_api.current_config_id = legacy_eval_config.id

    # Patch the API dependencies
    with (
        patch(
            "app.desktop.studio_server.eval_api.task_from_id"
        ) as mock_task_from_id_patch,
        patch(
            "app.desktop.studio_server.eval_api.task_run_config_from_id"
        ) as mock_task_run_config_from_id_patch,
        patch_resolve_split(test=stub_split(set())),
    ):
        mock_task_from_id_patch.return_value = mock_task_for_api
        mock_task_run_config_from_id_patch.return_value = mock_run_config

        response = client.get(
            f"/api/projects/project1/tasks/task1/run_configs/{mock_run_config.id}/eval_scores"
        )

    assert response.status_code == 200
    data = response.json()

    # Verify the structure
    assert "eval_results" in data
    assert len(data["eval_results"]) == 2

    # Find the results by eval name
    spec_eval_result = next(
        (r for r in data["eval_results"] if r["eval_name"] == "Test Eval"), None
    )
    legacy_eval_result = next(
        (r for r in data["eval_results"] if r["eval_name"] == "Legacy Eval"), None
    )

    assert spec_eval_result is not None
    assert legacy_eval_result is not None

    # Verify spec_id is populated for spec-associated eval
    assert spec_eval_result["spec_id"] == "spec1"

    # Verify spec_id is None for legacy eval
    assert legacy_eval_result["spec_id"] is None


@pytest.mark.asyncio
async def test_get_run_config_eval_scores_excludes_archived_specs(
    client, mock_task, mock_eval, mock_eval_config, mock_run_config
):
    """Test that get_run_config_eval_scores excludes evals associated with archived specs"""

    # Create an active spec
    active_spec = Spec(
        id="active_spec1",
        name="Active Spec",
        definition="Active spec definition",
        properties=DesiredBehaviourProperties(
            spec_type=SpecType.desired_behaviour,
            core_requirement="test instruction",
            desired_behaviour_description="test desired behaviour",
        ),
        eval_id=mock_eval.id,
        status=SpecStatus.active,
        parent=mock_task,
    )
    active_spec.save_to_file()

    # Create an archived spec with its own eval
    archived_eval = Eval(
        id="archived_eval1",
        name="Archived Eval",
        description="Eval for archived spec",
        template=None,
        eval_set_filter_id="tag::archived_eval_set",
        eval_configs_filter_id="tag::archived_golden",
        output_scores=[
            EvalOutputScore(
                name="score1",
                instruction="desc1",
                type=TaskOutputRatingType.five_star,
            ),
        ],
        parent=mock_task,
    )
    archived_eval.save_to_file()

    archived_eval_config = EvalConfig(
        id="archived_eval_config1",
        name="Archived Eval Config",
        config_type=EvalConfigType.g_eval,
        properties={"eval_steps": ["step1"]},
        parent=archived_eval,
        model_name="gpt-4",
        model_provider="openai",
    )
    archived_eval_config.save_to_file()
    archived_eval.current_config_id = archived_eval_config.id
    archived_eval.save_to_file()

    archived_spec = Spec(
        id="archived_spec1",
        name="Archived Spec",
        definition="Archived spec definition",
        properties=DesiredBehaviourProperties(
            spec_type=SpecType.desired_behaviour,
            core_requirement="test instruction",
            desired_behaviour_description="test desired behaviour",
        ),
        eval_id=archived_eval.id,
        status=SpecStatus.archived,
        parent=mock_task,
    )
    archived_spec.save_to_file()

    # Build mock eval objects with explicit attributes
    mock_eval_config_for_api = MagicMock()
    mock_eval_config_for_api.id = mock_eval_config.id
    mock_eval_config_for_api.runs.return_value = []

    mock_eval_for_api = MagicMock()
    mock_eval_for_api.id = mock_eval.id
    mock_eval_for_api.name = mock_eval.name
    mock_eval_for_api.splits = mock_eval.splits
    mock_eval_for_api.output_scores = mock_eval.output_scores
    mock_eval_for_api.current_config_id = mock_eval_config.id
    mock_eval_for_api.configs.return_value = [mock_eval_config_for_api]
    # Use the real eval's status resolution so the spec fallthrough is exercised
    mock_eval_for_api.resolved_status.side_effect = mock_eval.resolved_status

    archived_eval_config_for_api = MagicMock()
    archived_eval_config_for_api.id = archived_eval_config.id
    archived_eval_config_for_api.runs.return_value = []

    archived_eval_for_api = MagicMock()
    archived_eval_for_api.id = archived_eval.id
    archived_eval_for_api.name = archived_eval.name
    archived_eval_for_api.splits = archived_eval.splits
    archived_eval_for_api.output_scores = archived_eval.output_scores
    archived_eval_for_api.current_config_id = archived_eval_config.id
    archived_eval_for_api.configs.return_value = [archived_eval_config_for_api]
    archived_eval_for_api.resolved_status.side_effect = archived_eval.resolved_status

    mock_task_for_api = MagicMock()
    mock_task_for_api.evals.return_value = [mock_eval_for_api, archived_eval_for_api]
    mock_task_for_api.specs.return_value = [active_spec, archived_spec]

    with (
        patch(
            "app.desktop.studio_server.eval_api.task_from_id"
        ) as mock_task_from_id_patch,
        patch(
            "app.desktop.studio_server.eval_api.task_run_config_from_id"
        ) as mock_task_run_config_from_id_patch,
        patch_resolve_split(test=stub_split(set())),
    ):
        mock_task_from_id_patch.return_value = mock_task_for_api
        mock_task_run_config_from_id_patch.return_value = mock_run_config

        response = client.get(
            f"/api/projects/project1/tasks/task1/run_configs/{mock_run_config.id}/eval_scores"
        )

    assert response.status_code == 200
    data = response.json()

    # Only the active spec's eval should be present, not the archived one
    assert len(data["eval_results"]) == 1
    assert data["eval_results"][0]["eval_name"] == "Test Eval"
    assert data["eval_results"][0]["spec_id"] == "active_spec1"


@pytest.mark.asyncio
async def test_get_run_configs_includes_finetunes_with_run_config(
    client, mock_task_from_id, mock_task
):
    """Test that finetunes are included in run configs only if they have a run_config set."""
    mock_task_from_id.return_value = mock_task

    run_config_props = KilnAgentRunConfigProperties(
        model_name="gpt-4",
        model_provider_name=ModelProviderName.openai,
        prompt_id="simple_chain_of_thought_prompt_builder",
        structured_output_mode=StructuredOutputMode.json_schema,
    )

    finetunes = [
        Finetune(
            id="ft_completed",
            name="Completed Finetune",
            provider="openai",
            base_model_id="model1",
            dataset_split_id="split1",
            system_message="System message",
            latest_status=FineTuneStatusType.completed,
            run_config=run_config_props,
            fine_tune_model_id="ft_model_123",
            parent=mock_task,
        ),
        Finetune(
            id="ft_running",
            name="Running Finetune",
            provider="openai",
            base_model_id="model2",
            dataset_split_id="split2",
            system_message="System message",
            latest_status=FineTuneStatusType.running,
            run_config=run_config_props,
            fine_tune_model_id=None,
            parent=mock_task,
        ),
        Finetune(
            id="ft_unknown",
            name="Unknown Finetune",
            provider="openai",
            base_model_id="model3",
            dataset_split_id="split3",
            system_message="System message",
            latest_status=FineTuneStatusType.unknown,
            run_config=run_config_props,
            fine_tune_model_id=None,
            parent=mock_task,
        ),
        Finetune(
            id="ft_failed",
            name="Failed Finetune",
            provider="openai",
            base_model_id="model4",
            dataset_split_id="split4",
            system_message="System message",
            latest_status=FineTuneStatusType.failed,
            run_config=run_config_props,
            fine_tune_model_id=None,
            parent=mock_task,
        ),
        Finetune(
            id="ft_no_run_config",
            name="No Run Config Finetune",
            provider="openai",
            base_model_id="model5",
            dataset_split_id="split5",
            system_message="System message",
            latest_status=FineTuneStatusType.completed,
            run_config=None,
            parent=mock_task,
        ),
    ]

    for finetune in finetunes:
        finetune.save_to_file()

    response = client.get("/api/projects/project1/tasks/task1/run_configs")

    assert response.status_code == 200
    configs = response.json()

    config_ids = [config["id"] for config in configs]

    assert "finetune_run_config::project1::task1::ft_completed" in config_ids
    assert "finetune_run_config::project1::task1::ft_running" not in config_ids
    assert "finetune_run_config::project1::task1::ft_failed" not in config_ids
    assert "finetune_run_config::project1::task1::ft_unknown" not in config_ids
    assert "finetune_run_config::project1::task1::ft_no_run_config" not in config_ids


# --- SSE endpoints must carry @no_write_lock ---


def _find_endpoint_by_path(app, path_suffix: str):
    """Locate the endpoint function for a route ending with path_suffix."""
    for route in app.routes:
        if getattr(route, "path", "").endswith(path_suffix):
            return route.endpoint  # type: ignore[attr-defined]
    raise AssertionError(f"Route ending in {path_suffix} not found")


def test_run_comparison_has_no_write_lock(app):
    endpoint = _find_endpoint_by_path(
        app, "/eval_config/{eval_config_id}/run_comparison"
    )
    assert getattr(endpoint, "_git_sync_no_write_lock", False) is True


def test_run_calibration_has_no_write_lock(app):
    endpoint = _find_endpoint_by_path(app, "/evals/{eval_id}/run_calibration")
    assert getattr(endpoint, "_git_sync_no_write_lock", False) is True


# --- eval_results_summary tests ---


def _build_mock_eval(
    eval_id: str,
    name: str,
    current_config_id: str | None,
    output_scores: list[EvalOutputScore],
    configs: list,
    test_split: TaskRunSplit | EvalInputSplit | None,
) -> Mock:
    mock = Mock(spec=Eval)
    mock.id = eval_id
    mock.name = name
    mock.current_config_id = current_config_id
    mock.splits = {} if test_split is None else {"test": test_split}
    mock.output_scores = output_scores
    mock.configs.return_value = configs
    return mock


def _build_mock_eval_config(
    config_id: str,
    name: str,
    eval_runs: list[EvalRun],
) -> Mock:
    mock = Mock(spec=EvalConfig)
    mock.id = config_id
    mock.name = name
    mock.runs.return_value = eval_runs
    return mock


@pytest.mark.asyncio
async def test_eval_results_summary_happy_path(client):
    output_scores_1 = [
        EvalOutputScore(
            name="accuracy",
            instruction="Test accuracy",
            type=TaskOutputRatingType.pass_fail,
        ),
    ]
    output_scores_2 = [
        EvalOutputScore(
            name="relevance",
            instruction="Test relevance",
            type=TaskOutputRatingType.pass_fail,
        ),
    ]

    # Eval 1 default config (ec1): rc1 has 2 runs, rc2 has 1 run
    eval1_runs_default = [
        EvalRun(
            task_run_config_id="rc1",
            scores={"accuracy": 0.8},
            input="i",
            output="o",
            dataset_id="ds1",
        ),
        EvalRun(
            task_run_config_id="rc1",
            scores={"accuracy": 0.6},
            input="i",
            output="o",
            dataset_id="ds2",
        ),
        EvalRun(
            task_run_config_id="rc2",
            scores={"accuracy": 0.9},
            input="i",
            output="o",
            dataset_id="ds1",
        ),
    ]

    # Eval 2 default config (ec4): rc2 has 1 run
    eval2_runs_default = [
        EvalRun(
            task_run_config_id="rc2",
            scores={"relevance": 0.3},
            input="i",
            output="o",
            dataset_id="ds3",
        ),
    ]

    e1c1 = _build_mock_eval_config("ec1", "Judge A", eval1_runs_default)
    e1c2 = _build_mock_eval_config("ec2", "Judge B", [])

    e2c1 = _build_mock_eval_config("ec3", "Judge C", [])
    e2c2 = _build_mock_eval_config("ec4", "Judge D", eval2_runs_default)

    eval1 = _build_mock_eval(
        eval_id="eval1",
        name="Eval One",
        current_config_id="ec1",
        output_scores=output_scores_1,
        configs=[e1c1, e1c2],
        test_split=TaskRunSplit(filter_id="tag::eval_set_1"),
    )
    eval2 = _build_mock_eval(
        eval_id="eval2",
        name="Eval Two",
        current_config_id="ec4",
        output_scores=output_scores_2,
        configs=[e2c1, e2c2],
        test_split=TaskRunSplit(filter_id="tag::eval_set_2"),
    )

    rc1_mock = Mock(spec=TaskRunConfig, id="rc1")
    rc1_mock.name = "Run Config 1"
    rc2_mock = Mock(spec=TaskRunConfig, id="rc2")
    rc2_mock.name = "Run Config 2"
    rc3_mock = Mock(spec=TaskRunConfig, id="rc3")
    rc3_mock.name = "Run Config 3"

    mock_task = Mock(spec=Task)
    mock_task.run_configs.return_value = [rc1_mock, rc2_mock, rc3_mock]
    mock_task.finetunes.return_value = []
    mock_task.runs.return_value = []
    mock_task.evals.return_value = [eval1, eval2]

    with (
        patch("app.desktop.studio_server.eval_api.task_from_id") as mock_task_from_id,
        patch_resolve_split_by_ref(
            {
                ("task_run", "tag::eval_set_1"): {"ds1", "ds2"},
                ("task_run", "tag::eval_set_2"): {"ds3"},
            }
        ),
    ):
        mock_task_from_id.return_value = mock_task

        response = client.get("/api/projects/p1/tasks/t1/eval_results_summary")

    assert response.status_code == 200
    data = response.json()

    # --- evals_by_id dict ---
    assert "eval1" in data["evals_by_id"]
    assert "eval2" in data["evals_by_id"]
    assert data["evals_by_id"]["eval1"]["name"] == "Eval One"
    assert data["evals_by_id"]["eval1"]["default_judge_config_id"] == "ec1"
    assert data["evals_by_id"]["eval1"]["dataset_size"] == 2
    assert data["evals_by_id"]["eval1"]["output_score_keys"] == ["accuracy"]
    assert data["evals_by_id"]["eval2"]["name"] == "Eval Two"
    assert data["evals_by_id"]["eval2"]["default_judge_config_id"] == "ec4"
    assert data["evals_by_id"]["eval2"]["dataset_size"] == 1
    assert data["evals_by_id"]["eval2"]["output_score_keys"] == ["relevance"]

    # --- run_configs_by_id dict ---
    assert data["run_configs_by_id"]["rc1"]["name"] == "Run Config 1"
    assert data["run_configs_by_id"]["rc2"]["name"] == "Run Config 2"
    assert data["run_configs_by_id"]["rc3"]["name"] == "Run Config 3"

    # --- scores_by_run_config_by_eval dict (run_config outer, eval inner) ---
    # Eval 1 default judge (ec1): rc1 mean=0.7, rc2 mean=0.9
    assert data["scores_by_run_config_by_eval"]["rc1"]["eval1"]["mean_scores"][
        "accuracy"
    ] == pytest.approx(0.7)
    assert (
        data["scores_by_run_config_by_eval"]["rc1"]["eval1"]["percent_complete"] == 1.0
    )
    assert data["scores_by_run_config_by_eval"]["rc2"]["eval1"]["mean_scores"][
        "accuracy"
    ] == pytest.approx(0.9)
    assert (
        data["scores_by_run_config_by_eval"]["rc2"]["eval1"]["percent_complete"] == 0.5
    )

    # Eval 2 default judge (ec4): rc2 mean=0.3
    assert data["scores_by_run_config_by_eval"]["rc2"]["eval2"]["mean_scores"][
        "relevance"
    ] == pytest.approx(0.3)
    assert (
        data["scores_by_run_config_by_eval"]["rc2"]["eval2"]["percent_complete"] == 1.0
    )


@pytest.mark.asyncio
async def test_eval_results_summary_behavioral_equivalence(client):
    """For the default judge of an eval, results in eval_results_summary match /score_summary."""
    output_scores = [
        EvalOutputScore(
            name="accuracy",
            instruction="Test accuracy",
            type=TaskOutputRatingType.pass_fail,
        ),
        EvalOutputScore(
            name="relevance",
            instruction="Test relevance",
            type=TaskOutputRatingType.pass_fail,
        ),
    ]

    eval_runs = [
        EvalRun(
            task_run_config_id="rc1",
            scores={"accuracy": 0.8, "relevance": 0.9},
            input="i",
            output="o",
            dataset_id="ds1",
        ),
        EvalRun(
            task_run_config_id="rc1",
            scores={"accuracy": 0.6, "relevance": 0.7},
            input="i",
            output="o",
            dataset_id="ds2",
        ),
    ]

    ec1 = _build_mock_eval_config("ec1", "Judge A", eval_runs)

    eval1 = _build_mock_eval(
        eval_id="eval1",
        name="Eval One",
        current_config_id="ec1",
        output_scores=output_scores,
        configs=[ec1],
        test_split=TaskRunSplit(filter_id="tag::eval_set"),
    )

    rc1_mock = Mock(spec=TaskRunConfig, id="rc1")
    rc1_mock.name = "Run Config 1"

    mock_task = Mock(spec=Task)
    mock_task.run_configs.return_value = [rc1_mock]
    mock_task.finetunes.return_value = []
    mock_task.runs.return_value = []
    mock_task.evals.return_value = [eval1]

    with (
        patch("app.desktop.studio_server.eval_api.task_from_id") as mock_task_from_id,
        patch_resolve_split_by_ref({("task_run", "tag::eval_set"): {"ds1", "ds2"}}),
        patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eval_from_id,
        patch(
            "app.desktop.studio_server.eval_api.eval_config_from_id"
        ) as mock_eval_config_from_id,
    ):
        mock_task_from_id.return_value = mock_task
        mock_eval_from_id.return_value = eval1
        mock_eval_config_from_id.return_value = ec1

        summary_response = client.get("/api/projects/p1/tasks/t1/eval_results_summary")
        score_response = client.get(
            "/api/projects/p1/tasks/t1/evals/eval1/eval_config/ec1/score_summary"
        )

    assert summary_response.status_code == 200
    assert score_response.status_code == 200

    summary_data = summary_response.json()
    score_data = score_response.json()

    # Compare per run_config cell: mean_scores should match score_summary results
    for rc_id, evals_dict in summary_data["scores_by_run_config_by_eval"].items():
        cell = evals_dict["eval1"]
        for score_key, mean_val in cell["mean_scores"].items():
            assert mean_val == pytest.approx(
                score_data["results"][rc_id][score_key]["mean_score"]
            )
        assert cell["percent_complete"] == pytest.approx(
            score_data["run_config_percent_complete"][rc_id]
        )


@pytest.mark.asyncio
async def test_eval_results_summary_empty_filter(client):
    """Empty dataset filter: eval appears in evals but not in results."""
    output_scores = [
        EvalOutputScore(
            name="accuracy",
            instruction="Test accuracy",
            type=TaskOutputRatingType.pass_fail,
        ),
    ]
    ec1 = _build_mock_eval_config("ec1", "Judge A", [])
    eval1 = _build_mock_eval(
        eval_id="eval1",
        name="Eval One",
        current_config_id="ec1",
        output_scores=output_scores,
        configs=[ec1],
        test_split=TaskRunSplit(filter_id="tag::empty"),
    )

    mock_task = Mock(spec=Task)
    mock_task.run_configs.return_value = []
    mock_task.finetunes.return_value = []
    mock_task.runs.return_value = []
    mock_task.evals.return_value = [eval1]

    with (
        patch("app.desktop.studio_server.eval_api.task_from_id") as mock_task_from_id,
        patch_resolve_split_by_ref({}),
    ):
        mock_task_from_id.return_value = mock_task

        response = client.get("/api/projects/p1/tasks/t1/eval_results_summary")

    assert response.status_code == 200
    data = response.json()
    assert "eval1" in data["evals_by_id"]
    assert data["evals_by_id"]["eval1"]["dataset_size"] == 0
    # No run_config should have an eval1 entry
    for rc_evals in data["scores_by_run_config_by_eval"].values():
        assert "eval1" not in rc_evals


@pytest.mark.asyncio
async def test_eval_results_summary_no_default_judge(client):
    """Eval with no current_config_id appears in evals but not in results."""
    output_scores = [
        EvalOutputScore(
            name="accuracy",
            instruction="Test accuracy",
            type=TaskOutputRatingType.pass_fail,
        ),
    ]
    ec1 = _build_mock_eval_config("ec1", "Judge A", [])
    eval1 = _build_mock_eval(
        eval_id="eval1",
        name="Eval One",
        current_config_id=None,
        output_scores=output_scores,
        configs=[ec1],
        test_split=TaskRunSplit(filter_id="tag::test"),
    )

    mock_task = Mock(spec=Task)
    mock_task.run_configs.return_value = []
    mock_task.finetunes.return_value = []
    mock_task.runs.return_value = []
    mock_task.evals.return_value = [eval1]

    with (
        patch("app.desktop.studio_server.eval_api.task_from_id") as mock_task_from_id,
        patch_resolve_split_by_ref({("task_run", "tag::test"): {"ds1"}}),
    ):
        mock_task_from_id.return_value = mock_task

        response = client.get("/api/projects/p1/tasks/t1/eval_results_summary")

    assert response.status_code == 200
    data = response.json()
    assert "eval1" in data["evals_by_id"]
    assert data["evals_by_id"]["eval1"]["default_judge_config_id"] is None
    # No run_config should have an eval1 entry
    for rc_evals in data["scores_by_run_config_by_eval"].values():
        assert "eval1" not in rc_evals


@pytest.mark.asyncio
async def test_eval_results_summary_no_evals(client):
    """Task with no evals returns empty dicts."""
    mock_task = Mock(spec=Task)
    mock_task.run_configs.return_value = []
    mock_task.finetunes.return_value = []
    mock_task.runs.return_value = []
    mock_task.evals.return_value = []

    with patch("app.desktop.studio_server.eval_api.task_from_id") as mock_task_from_id:
        mock_task_from_id.return_value = mock_task

        response = client.get("/api/projects/p1/tasks/t1/eval_results_summary")

    assert response.status_code == 200
    assert response.json() == {
        "evals_by_id": {},
        "run_configs_by_id": {},
        "scores_by_run_config_by_eval": {},
    }


class TestEvalResultsSummaryResolutionCaching:
    """A split is resolved once per unique (source, filter id), not once per eval.

    One case per scenario rather than one test that mutates `eval2.splits` between three
    requests: the source-awareness case is the newest and most valuable of the three, and
    sequencing it behind the other two means it only ever runs if they pass.
    """

    OUTPUT_SCORES: ClassVar[List[EvalOutputScore]] = [
        EvalOutputScore(
            name="accuracy",
            instruction="Test accuracy",
            type=TaskOutputRatingType.pass_fail,
        ),
    ]

    # Every (source, filter) pair used below resolves to the same single item, so a
    # resolution count is the only thing that varies between the cases.
    ALL_DS1: ClassVar[Dict[Tuple[ItemSource, str], set]] = {
        ("task_run", "tag::set1"): {"ds1"},
        ("task_run", "tag::set2"): {"ds1"},
        ("eval_input", "tag::set1"): {"ds1"},
    }

    def _task_with_two_evals(self, eval2_test_split) -> Mock:
        eval_runs = [
            EvalRun(
                task_run_config_id="rc1",
                scores={"accuracy": 0.8},
                input="i",
                output="o",
                dataset_id="ds1",
            ),
        ]
        eval1 = _build_mock_eval(
            eval_id="eval1",
            name="Eval One",
            current_config_id="ec1a",
            output_scores=self.OUTPUT_SCORES,
            configs=[_build_mock_eval_config("ec1a", "Judge A1", eval_runs)],
            test_split=TaskRunSplit(filter_id="tag::set1"),
        )
        eval2 = _build_mock_eval(
            eval_id="eval2",
            name="Eval Two",
            current_config_id="ec2a",
            output_scores=self.OUTPUT_SCORES,
            configs=[_build_mock_eval_config("ec2a", "Judge B1", eval_runs)],
            test_split=eval2_test_split,
        )

        rc1_mock = Mock(spec=TaskRunConfig, id="rc1")
        rc1_mock.name = "RC1"

        mock_task = Mock(spec=Task)
        mock_task.run_configs.return_value = [rc1_mock]
        mock_task.finetunes.return_value = []
        mock_task.evals.return_value = [eval1, eval2]
        return mock_task

    def _resolution_count(self, client, eval2_test_split) -> int:
        mock_task = self._task_with_two_evals(eval2_test_split)
        with (
            patch(
                "app.desktop.studio_server.eval_api.task_from_id"
            ) as mock_task_from_id,
            patch_resolve_split_by_ref(self.ALL_DS1) as mock_resolve,
        ):
            mock_task_from_id.return_value = mock_task
            response = client.get("/api/projects/p1/tasks/t1/eval_results_summary")

        assert response.status_code == 200
        return mock_resolve.call_count

    def test_different_filters_are_resolved_separately(self, client):
        assert self._resolution_count(client, TaskRunSplit(filter_id="tag::set2")) == 2

    def test_the_same_filter_over_the_same_store_is_resolved_once(self, client):
        assert self._resolution_count(client, TaskRunSplit(filter_id="tag::set1")) == 1

    def test_the_same_filter_over_a_different_store_is_resolved_again(self, client):
        """Same filter id over a DIFFERENT store is a different item set, so it must not
        share a resolution: the `tag::` grammar is identical in both (spec 5.3)."""
        assert (
            self._resolution_count(client, EvalInputSplit(filter_id="tag::set1")) == 2
        )


class TestEvalResultsSummaryRealEvals:
    """The endpoint against evals loaded off disk, with `resolve_split` left alone.

    Every other case in this block builds its evals with `_build_mock_eval` *and* patches
    `resolve_split`, so both halves of "eval -> its test split -> its items" are stubbed
    and only the aggregation arithmetic is real. That leaves the read the endpoint is
    built on untested: `_cached_test_split` asks for `eval.splits["test"]`, which on a
    pre-existing project is populated only because `Eval.migrate_legacy_split_fields` ran
    at load. Hand-setting `.splits` on a `Mock` manufactures that postcondition, so if the
    migration stopped running — or ran after `validate_splits` — this endpoint would
    return an empty results table for every shipped project with the whole suite green.
    """

    def _write_legacy_eval_file(
        self,
        task: Task,
        eval_id: str,
        name: str,
        eval_set_filter_id: str,
        current_config_id: str | None = None,
    ) -> Eval:
        """An eval file in the pre-`splits` shape a shipped project actually has on disk.

        Built through the model and then rewound, rather than hand-writing a whole eval
        dict: the point is the legacy split fields, and every other key should be
        whatever this build writes so the file doesn't rot into an unloadable shape when
        an unrelated field is added.
        """
        eval = Eval(
            id=eval_id,
            name=name,
            description="Legacy eval",
            output_scores=[
                EvalOutputScore(
                    name="accuracy",
                    instruction="Test accuracy",
                    type=TaskOutputRatingType.pass_fail,
                ),
            ],
            eval_set_filter_id=eval_set_filter_id,
            eval_configs_filter_id="tag::golden",
            current_config_id=current_config_id,
            parent=task,
        )
        eval.save_to_file()
        assert eval.path is not None
        # save_to_file writes the migrated shape (splits populated, legacy fields null).
        # Put the file back the way a build predating `splits` wrote it, which is what
        # every existing project's file still looks like until this app rewrites it.
        saved = json.loads(eval.path.read_text(encoding="utf-8"))
        saved.pop("splits", None)
        saved["eval_set_filter_id"] = eval_set_filter_id
        eval.path.write_text(json.dumps(saved), encoding="utf-8")
        return eval

    def test_summarizes_a_legacy_eval_read_from_disk(
        self, client, mock_task_from_id, mock_task, mock_run_config, data_source
    ):
        """The whole path, unmocked: legacy file -> migration -> splits['test'] -> items."""
        eval = self._write_legacy_eval_file(
            mock_task,
            eval_id="legacy_eval",
            name="Legacy Eval",
            eval_set_filter_id="tag::eval_set",
            current_config_id="legacy_config",
        )
        eval_config = EvalConfig(
            id="legacy_config",
            name="Legacy Judge",
            config_type=EvalConfigType.g_eval,
            properties={"eval_steps": ["step1"]},
            parent=eval,
            model_name="gpt-4",
            model_provider="openai",
        )
        eval_config.save_to_file()

        scored = _tagged_task_run(mock_task, data_source, "eval_set")
        _tagged_task_run(mock_task, data_source, "eval_set")
        _tagged_task_run(mock_task, data_source, "not_the_eval_set")
        EvalRun(
            task_run_config_id="run_config1",
            scores={"accuracy": 1.0},
            input="i",
            output="o",
            dataset_id=scored.id,
            parent=eval_config,
        ).save_to_file()

        response = client.get("/api/projects/project1/tasks/task1/eval_results_summary")

        assert response.status_code == 200
        data = response.json()
        # 2 of the 3 runs carry the tag the legacy `eval_set_filter_id` named. A
        # migration that stopped populating splits['test'] answers 0 here, and drops the
        # eval from the table entirely.
        assert data["evals_by_id"]["legacy_eval"]["dataset_size"] == 2
        assert data["evals_by_id"]["legacy_eval"]["name"] == "Legacy Eval"
        assert data["evals_by_id"]["legacy_eval"]["output_score_keys"] == ["accuracy"]
        cell = data["scores_by_run_config_by_eval"]["run_config1"]["legacy_eval"]
        assert cell["mean_scores"]["accuracy"] == pytest.approx(1.0)
        assert cell["percent_complete"] == pytest.approx(0.5)

    def test_summarizes_an_eval_input_backed_eval_read_from_disk(
        self, client, mock_task_from_id, mock_task, mock_eval, mock_run_config
    ):
        """The other backing, also off disk: the split's items come from `task.eval_inputs`."""
        mock_eval.set_split("test", EvalInputSplit(filter_id="tag::inputs"))
        mock_eval.current_config_id = "eval_config1"
        mock_eval.save_to_file()
        eval_config = EvalConfig(
            id="eval_config1",
            name="Test Eval Config",
            config_type=EvalConfigType.g_eval,
            properties={"eval_steps": ["step1"]},
            parent=mock_eval,
            model_name="gpt-4",
            model_provider="openai",
        )
        eval_config.save_to_file()

        scored_input = _tagged_eval_input(mock_task, "inputs")
        _tagged_eval_input(mock_task, "inputs")
        _tagged_eval_input(mock_task, "other")
        EvalRun(
            task_run_config_id="run_config1",
            scores={"score1": 4.0, "overall_rating": 5.0},
            input="i",
            output="o",
            eval_input_id=scored_input.id,
            parent=eval_config,
        ).save_to_file()

        response = client.get("/api/projects/project1/tasks/task1/eval_results_summary")

        assert response.status_code == 200
        data = response.json()
        assert data["evals_by_id"]["eval1"]["dataset_size"] == 2
        cell = data["scores_by_run_config_by_eval"]["run_config1"]["eval1"]
        assert cell["mean_scores"]["score1"] == pytest.approx(4.0)
        assert cell["percent_complete"] == pytest.approx(0.5)


class TestCachedTestSplit:
    def _eval(self, eval_id: str, split) -> Mock:
        return _build_mock_eval(
            eval_id=eval_id,
            name=eval_id,
            current_config_id=None,
            output_scores=[],
            configs=[],
            test_split=split,
        )

    def test_reuses_a_resolution_for_the_same_source_and_filter(self):
        cache = {}
        with patch_resolve_split_by_ref(
            {("task_run", "tag::shared"): {"ds1"}}
        ) as mock_resolve:
            first = _cached_test_split(
                Mock(spec=Task),
                self._eval("eval1", TaskRunSplit(filter_id="tag::shared")),
                cache,
            )
            second = _cached_test_split(
                Mock(spec=Task),
                self._eval("eval2", TaskRunSplit(filter_id="tag::shared")),
                cache,
            )

        assert mock_resolve.call_count == 1
        assert first.item_keys() == second.item_keys()

    def test_a_cache_hit_names_the_eval_it_was_asked_about(self):
        """ResolvedSplit carries the eval it came from so a consumer can check a split
        belongs to the eval it is working on. A cached value handed on unchanged would
        name whichever eval reached the filter first, quietly breaking that check."""
        cache = {}
        with patch_resolve_split_by_ref({("task_run", "tag::shared"): {"ds1"}}):
            _cached_test_split(
                Mock(spec=Task),
                self._eval("eval1", TaskRunSplit(filter_id="tag::shared")),
                cache,
            )
            second = _cached_test_split(
                Mock(spec=Task),
                self._eval("eval2", TaskRunSplit(filter_id="tag::shared")),
                cache,
            )

        assert second.eval_id == "eval2"

    def test_the_same_filter_over_a_different_store_is_a_different_entry(self):
        cache = {}
        with patch_resolve_split_by_ref(
            {
                ("task_run", "tag::shared"): {"ds1"},
                ("eval_input", "tag::shared"): {"ei1"},
            }
        ) as mock_resolve:
            task_run_backed = _cached_test_split(
                Mock(spec=Task),
                self._eval("eval1", TaskRunSplit(filter_id="tag::shared")),
                cache,
            )
            input_backed = _cached_test_split(
                Mock(spec=Task),
                self._eval("eval2", EvalInputSplit(filter_id="tag::shared")),
                cache,
            )

        assert mock_resolve.call_count == 2
        assert task_run_backed.item_keys() == {("task_run", "ds1")}
        assert input_backed.item_keys() == {("eval_input", "ei1")}

    def test_none_when_the_eval_has_no_test_split(self):
        with patch_resolve_split_by_ref({}) as mock_resolve:
            assert (
                _cached_test_split(Mock(spec=Task), self._eval("e", None), {}) is None
            )

        mock_resolve.assert_not_called()


class TestCodeEvalTrustEndpoints:
    @pytest.fixture(autouse=True)
    def _clear_trust(self):
        from kiln_ai.adapters.eval.v2_eval_code_eval import _reset_add_code_trust

        _reset_add_code_trust()
        yield
        _reset_add_code_trust()

    def test_add_trust(self, client):
        with patch("app.desktop.studio_server.eval_api.project_from_id") as mock_proj:
            mock_proj.return_value = Mock()
            response = client.post("/api/projects/proj-1/add_code_trust")

        assert response.status_code == 200
        assert response.json() == {"trusted": True}

    def test_add_trust_invalid_project(self, client):
        with patch("app.desktop.studio_server.eval_api.project_from_id") as mock_proj:
            mock_proj.side_effect = HTTPException(status_code=404, detail="Not found")
            response = client.post("/api/projects/bad-id/add_code_trust")

        assert response.status_code == 404

    def test_check_trust_untrusted(self, client):
        with patch("app.desktop.studio_server.eval_api.project_from_id") as mock_proj:
            mock_proj.return_value = Mock()
            response = client.get("/api/projects/proj-1/add_code_trust")
        assert response.status_code == 200
        assert response.json() == {"trusted": False}

    def test_check_trust_after_add(self, client):
        mock_project = Mock()
        with patch("app.desktop.studio_server.eval_api.project_from_id") as mock_proj:
            mock_proj.return_value = mock_project
            client.post("/api/projects/proj-1/add_code_trust")
            response = client.get("/api/projects/proj-1/add_code_trust")
        assert response.status_code == 200
        assert response.json() == {"trusted": True}


@pytest.fixture
def mock_v2_eval(mock_task):
    eval = Eval(
        id="eval_v2",
        name="V2 Test Eval",
        description="V2 eval for testing",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                instruction="Is the answer accurate?",
                type=TaskOutputRatingType.pass_fail,
            ),
        ],
        splits={"test": EvalInputSplit(filter_id="tag::v2_eval_set")},
        evaluation_data_type=None,
        parent=mock_task,
    )
    eval.save_to_file()
    return eval


@pytest.fixture
def mock_v2_reference_answer_eval(mock_task):
    eval = Eval(
        id="eval_v2_reference_answer",
        name="V2 Reference Answer Eval",
        description="V2 eval graded against a reference answer",
        output_scores=[
            EvalOutputScore(
                name="accuracy",
                instruction="Is the answer accurate?",
                type=TaskOutputRatingType.pass_fail,
            ),
        ],
        splits={"test": EvalInputSplit(filter_id="tag::v2_eval_set")},
        evaluation_data_type=EvalDataType.reference_answer,
        parent=mock_task,
    )
    eval.save_to_file()
    return eval


class TestTestV2Eval:
    def _url(self, eval_id: str = "eval_v2") -> str:
        return f"/api/projects/project1/tasks/task1/evals/{eval_id}/test_v2_eval"

    def _exact_match_payload(self) -> dict:
        return {
            "properties": {
                "type": "exact_match",
                "expected_value": "hello",
            },
            "eval_input": {
                "final_message": "hello",
            },
        }

    def test_exact_match_pass(self, client, mock_v2_eval):
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            response = client.post(
                self._url(),
                json=self._exact_match_payload(),
            )
        assert response.status_code == 200
        body = response.json()
        assert body["scores"]["accuracy"] == 1.0
        assert body["skipped_reason"] is None
        assert body["skipped_detail"] is None

    def test_exact_match_fail(self, client, mock_v2_eval):
        payload = self._exact_match_payload()
        payload["eval_input"]["final_message"] = "world"
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            response = client.post(self._url(), json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["scores"]["accuracy"] == 0.0
        assert body["skipped_reason"] is None

    def test_code_eval_untrusted_skip(self, client, mock_v2_eval):
        payload = {
            "properties": {
                "type": "code_eval",
                "code": "def score(output, **kwargs):\n    return {'accuracy': 1.0}\n",
            },
            "eval_input": {
                "final_message": "test",
            },
        }
        with (
            patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid,
            patch("app.desktop.studio_server.eval_api.project_from_id") as mock_proj,
            patch(
                "app.desktop.studio_server.eval_api.has_add_code_trust",
                return_value=False,
            ),
        ):
            mock_eid.return_value = mock_v2_eval
            mock_proj.return_value = Mock()
            response = client.post(self._url(), json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["scores"] == {}
        assert body["skipped_reason"] == "code_eval_not_trusted"
        assert body["skipped_detail"] == "Project not trusted for code eval execution."

    def test_code_eval_trusted_execution(self, client, mock_v2_eval):
        payload = {
            "properties": {
                "type": "code_eval",
                "code": "def score(output, **kwargs):\n    return {'accuracy': 1.0}\n",
            },
            "eval_input": {
                "final_message": "test",
            },
        }
        with (
            patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid,
            patch("app.desktop.studio_server.eval_api.project_from_id") as mock_proj,
            patch(
                "app.desktop.studio_server.eval_api.has_add_code_trust",
                return_value=True,
            ),
            patch(
                "kiln_ai.adapters.eval.v2_eval_code_eval.run_bridged_child",
                new=AsyncMock(
                    return_value=BridgeResult(
                        result_msg={"type": "result", "ok": {"accuracy": 0.75}}
                    )
                ),
            ),
        ):
            mock_eid.return_value = mock_v2_eval
            mock_proj.return_value = Mock()
            response = client.post(self._url(), json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["scores"]["accuracy"] == 0.75
        assert body["skipped_reason"] is None
        assert body["tool_call_log"] == []

    def test_code_eval_reports_nested_tool_calls(self, client, mock_v2_eval):
        """The test pane records what the scorer called, so nested LLM spend is visible."""
        payload = {
            "properties": {
                "type": "code_eval",
                "code": "def score(output, **kwargs):\n    return {'accuracy': 1.0}\n",
                "tool_allowlist": ["kiln_tool::llm"],
            },
            "eval_input": {"final_message": "test"},
        }

        responses = _CollectingResponses()

        async def serve_one_tool_call(**kwargs):
            """Stand in for the child: hand the server a real tool_call to serve.

            Going through the public ``serve()`` rather than poking the recorder
            exercises what the endpoint actually depends on -- allowlist resolution,
            the registry lookup, the tool run, and the recorder the endpoint
            installed -- end to end.
            """
            await kwargs["server"].serve(
                {
                    "type": "tool_call",
                    "call_id": "call-1",
                    "tool_name": "llm",
                    "arguments": {"prompt": "hi"},
                },
                responses,
            )
            return BridgeResult(result_msg={"type": "result", "ok": {"accuracy": 1.0}})

        with (
            patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid,
            patch("app.desktop.studio_server.eval_api.project_from_id") as mock_proj,
            patch(
                "app.desktop.studio_server.eval_api.has_add_code_trust",
                return_value=True,
            ),
            patch(
                "kiln_ai.tools.tool_registry.tool_from_id_and_project",
                return_value=_FakeLlmTool(),
            ),
            patch(
                "kiln_ai.adapters.eval.v2_eval_code_eval.run_bridged_child",
                new=serve_one_tool_call,
            ),
        ):
            mock_eid.return_value = mock_v2_eval
            mock_proj.return_value = Mock()
            response = client.post(self._url(), json=payload)

        assert response.status_code == 200
        log = response.json()["tool_call_log"]
        assert len(log) == 1
        assert log[0]["tool_name"] == "llm"
        assert log[0]["arguments"] == {"prompt": "hi"}
        assert log[0]["output_preview"] == "a judgement"
        assert log[0]["is_error"] is False
        # The parent also answered the child, which is what unblocks the call.
        assert responses.puts == [
            {"type": "tool_result", "call_id": "call-1", "ok": "a judgement"}
        ]

    def test_llm_judge_with_mocked_model(self, client, mock_v2_eval):
        payload = {
            "properties": {
                "type": "llm_judge",
                "model_name": "gpt-4o",
                "model_provider": "openai",
                "prompt_template": "Is this correct? Output: {{ final_message }}",
            },
            "eval_input": {
                "final_message": "test output",
            },
        }
        mock_run_output = RunOutput(
            output={"accuracy": 5},
            intermediate_outputs=None,
        )
        mock_adapter = MagicMock()
        # The adapter's first return value is the judge's own TaskRun, whose usage the
        # V2 result carries as eval_usage.
        mock_adapter.invoke_returning_run_output = AsyncMock(
            return_value=(Mock(usage=None), mock_run_output)
        )
        with (
            patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid,
            patch(
                "kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task",
                return_value=mock_adapter,
            ),
        ):
            mock_eid.return_value = mock_v2_eval
            response = client.post(self._url(), json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "accuracy" in body["scores"]
        assert body["skipped_reason"] is None

    def test_nothing_persisted(self, client, mock_v2_eval):
        eval_dir = mock_v2_eval.path.parent
        files_before = set(str(f) for f in eval_dir.rglob("*"))

        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            client.post(self._url(), json=self._exact_match_payload())

        files_after = set(str(f) for f in eval_dir.rglob("*"))
        new_files = files_after - files_before
        assert len(new_files) == 0, f"Unexpected new files created: {new_files}"

    def test_eval_not_found(self, client):
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.side_effect = HTTPException(
                status_code=404, detail="Eval not found. ID: bad_id"
            )
            response = client.post(
                self._url("bad_id"),
                json=self._exact_match_payload(),
            )
        assert response.status_code == 404

    def test_llm_judge_builder_input(self, client, mock_v2_eval):
        payload = {
            "llm_judge_builder_input": {
                "model_name": "gpt-4o",
                "provider": "openai",
                "g_eval": False,
            },
            "eval_input": {
                "final_message": "test output",
            },
        }
        mock_run_output = RunOutput(
            output={"accuracy": 5},
            intermediate_outputs=None,
        )
        mock_adapter = MagicMock()
        # The adapter's first return value is the judge's own TaskRun, whose usage the
        # V2 result carries as eval_usage.
        mock_adapter.invoke_returning_run_output = AsyncMock(
            return_value=(Mock(usage=None), mock_run_output)
        )
        with (
            patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid,
            patch(
                "app.desktop.studio_server.eval_api.materialize_llm_judge_properties"
            ) as mock_materialize,
            patch(
                "kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task",
                return_value=mock_adapter,
            ),
        ):
            mock_materialize.return_value = {
                "type": "llm_judge",
                "model_name": "gpt-4o",
                "model_provider": "openai",
                "g_eval": False,
                "prompt_template": "test template {{ final_message }}",
            }
            mock_eid.return_value = mock_v2_eval
            response = client.post(self._url(), json=payload)
        assert response.status_code == 200
        mock_materialize.assert_called_once_with(
            eval=mock_v2_eval,
            model_name="gpt-4o",
            model_provider="openai",
            g_eval=False,
            judge_prompt=None,
            system_prompt=None,
            judge_instructions=None,
        )
        body = response.json()
        assert "accuracy" in body["scores"]
        assert body["skipped_reason"] is None

    def test_400_when_no_properties_or_builder_input(self, client, mock_v2_eval):
        payload = {
            "eval_input": {
                "final_message": "test output",
            },
        }
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            response = client.post(self._url(), json=payload)
        assert response.status_code == 400
        body = response.json()
        msg = (body.get("message") or body.get("detail") or "").lower()
        assert "properties" in msg or "llm_judge" in msg

    def test_score_range_errors_none_for_in_range(self, client, mock_v2_eval):
        """In-range scores should NOT produce score_range_errors."""
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            response = client.post(
                self._url(),
                json=self._exact_match_payload(),
            )
        assert response.status_code == 200
        body = response.json()
        assert body["scores"]["accuracy"] == 1.0
        assert body["score_range_errors"] is None

    def test_score_range_errors_populated_for_out_of_range(self, client, mock_v2_eval):
        """Out-of-range scores should populate score_range_errors."""
        payload = {
            "properties": {
                "type": "code_eval",
                "code": "def score(output, **kwargs):\n    return {'accuracy': 5.0}\n",
            },
            "eval_input": {
                "final_message": "test",
            },
        }
        with (
            patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid,
            patch("app.desktop.studio_server.eval_api.project_from_id") as mock_proj,
            patch(
                "app.desktop.studio_server.eval_api.has_add_code_trust",
                return_value=True,
            ),
            patch(
                "kiln_ai.adapters.eval.v2_eval_code_eval.run_bridged_child",
                new=AsyncMock(
                    return_value=BridgeResult(
                        result_msg={"type": "result", "ok": {"accuracy": 5.0}}
                    )
                ),
            ),
        ):
            mock_eid.return_value = mock_v2_eval
            mock_proj.return_value = Mock()
            response = client.post(self._url(), json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["scores"]["accuracy"] == 5.0
        assert body["score_range_errors"] is not None
        assert len(body["score_range_errors"]) == 1
        assert "pass_fail" in body["score_range_errors"][0]

    def test_score_range_errors_none_when_skipped(self, client, mock_v2_eval):
        """Skipped results should not have score_range_errors."""
        payload = {
            "properties": {
                "type": "code_eval",
                "code": "def score(output, **kwargs):\n    return {'accuracy': 1.0}\n",
            },
            "eval_input": {
                "final_message": "test",
            },
        }
        with (
            patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid,
            patch("app.desktop.studio_server.eval_api.project_from_id") as mock_proj,
            patch(
                "app.desktop.studio_server.eval_api.has_add_code_trust",
                return_value=False,
            ),
        ):
            mock_eid.return_value = mock_v2_eval
            mock_proj.return_value = Mock()
            response = client.post(self._url(), json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["skipped_reason"] == "code_eval_not_trusted"
        assert body["score_range_errors"] is None


class TestCreateLlmJudgeConfig:
    def _url(self, eval_id: str = "eval_v2") -> str:
        return f"/api/projects/project1/tasks/task1/evals/{eval_id}/create_llm_judge_config"

    def test_success(self, client, mock_v2_eval):
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            response = client.post(
                self._url(),
                json={
                    "model_name": "gpt-4o",
                    "provider": "openai",
                    "g_eval": False,
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["config_type"] == "v2"
        props = body["properties"]
        assert props["type"] == "llm_judge"
        assert props["model_name"] == "gpt-4o"
        assert props["model_provider"] == "openai"
        assert props["g_eval"] is False
        assert "{{ task_input }}" in props["prompt_template"]
        assert "{{ final_message }}" in props["prompt_template"]
        assert props["system_prompt"] is not None
        assert props["thinking_instruction"] is not None
        assert props["reference_keys"] == []

    def test_g_eval_true(self, client, mock_v2_eval):
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            response = client.post(
                self._url(),
                json={
                    "model_name": "gpt-4o",
                    "provider": "openai",
                    "g_eval": True,
                },
            )
        assert response.status_code == 200
        assert response.json()["properties"]["g_eval"] is True

    def test_persisted_to_disk(self, client, mock_v2_eval):
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            client.post(
                self._url(),
                json={
                    "model_name": "gpt-4o",
                    "provider": "openai",
                    "g_eval": False,
                },
            )
        configs = mock_v2_eval.configs()
        assert len(configs) == 1
        cfg = configs[0]
        assert cfg.config_type.value == "v2"
        assert cfg.properties.type.value == "llm_judge"

    def test_eval_not_found(self, client):
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.side_effect = HTTPException(
                status_code=404, detail="Eval not found. ID: bad_id"
            )
            response = client.post(
                self._url("bad_id"),
                json={
                    "model_name": "gpt-4o",
                    "provider": "openai",
                    "g_eval": False,
                },
            )
        assert response.status_code == 404

    def test_missing_model(self, client, mock_v2_eval):
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            response = client.post(
                self._url(),
                json={
                    "g_eval": False,
                },
            )
        assert response.status_code == 422

    def test_with_custom_name(self, client, mock_v2_eval):
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            response = client.post(
                self._url(),
                json={
                    "model_name": "gpt-4o",
                    "provider": "openai",
                    "g_eval": False,
                    "name": "My Custom Judge",
                },
            )
        assert response.status_code == 200
        assert response.json()["name"] == "My Custom Judge"

    def test_reference_answer_eval_declares_the_key(
        self, client, mock_v2_reference_answer_eval
    ):
        """The baked prompt grades against a reference answer, so the saved config has
        to require one — otherwise the judge scores items that have none."""
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_reference_answer_eval
            response = client.post(
                self._url("eval_v2_reference_answer"),
                json={
                    "model_name": "gpt-4o",
                    "provider": "openai",
                    "g_eval": False,
                },
            )
        assert response.status_code == 200
        props = response.json()["properties"]
        assert props["reference_keys"] == ["reference_answer"]
        assert "<reference_answer>" in props["prompt_template"]

    def test_client_cannot_clear_the_server_derived_reference_keys(
        self, client, mock_v2_reference_answer_eval
    ):
        """The builder used to post its own `reference_keys`, which the endpoint wrote
        over the derived value — an empty list from a UI that cannot collect them turned
        the requirement off. The request field is gone; the server decides."""
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_reference_answer_eval
            response = client.post(
                self._url("eval_v2_reference_answer"),
                json={
                    "model_name": "gpt-4o",
                    "provider": "openai",
                    "g_eval": False,
                    "reference_keys": [],
                },
            )
        assert response.status_code == 200
        assert response.json()["properties"]["reference_keys"] == ["reference_answer"]


class TestV1CoexistenceAPI:
    """V1 coexistence regression guards at the API layer.

    Ensures that V1 g_eval / llm_as_judge configs and their runs continue to
    work through score-summary and eval-results endpoints after V2 additions.
    """

    def test_v1_g_eval_score_summary(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
    ):
        mock_task_from_id.return_value = mock_task

        run = EvalRun(
            parent=mock_eval_config,
            dataset_id="dataset_id1",
            task_run_config_id="run_config1",
            input="test input",
            output="test output",
            scores={"score1": 4.0, "overall_rating": 3.0},
        )
        run.save_to_file()

        run2 = EvalRun(
            parent=mock_eval_config,
            dataset_id="dataset_id2",
            task_run_config_id="run_config1",
            input="test input 2",
            output="test output 2",
            scores={"score1": 2.0, "overall_rating": 5.0},
        )
        run2.save_to_file()

        with patch_resolve_split(test=stub_split({"dataset_id1", "dataset_id2"})):
            response = client.get(
                "/api/projects/project1/tasks/task1/evals/eval1"
                "/eval_config/eval_config1/score_summary"
            )

        assert response.status_code == 200
        body = response.json()
        assert "results" in body
        assert "dataset_size" in body
        assert body["dataset_size"] == 2

        scores = body["results"]["run_config1"]
        assert scores["score1"]["mean_score"] == pytest.approx(3.0)
        assert scores["score1"]["n_used"] == 2
        assert scores["score1"]["n_excluded"] == 0
        assert scores["overall_rating"]["mean_score"] == pytest.approx(4.0)

        assert body["run_config_percent_complete"]["run_config1"] == 1.0

    def test_v1_g_eval_run_results(
        self,
        client,
        mock_task_from_id,
        mock_task,
        mock_eval,
        mock_eval_config,
        mock_run_config,
        data_source,
    ):
        mock_task_from_id.return_value = mock_task

        in_split = _tagged_task_run(mock_task, data_source, "eval_set")
        run = EvalRun(
            parent=mock_eval_config,
            task_run_config_id="run_config1",
            scores={"score1": 3.5, "overall_rating": 4.0},
            input="hello",
            output="world",
            dataset_id=in_split.id,
        )
        run.save_to_file()

        response = client.get(
            "/api/projects/project1/tasks/task1/evals/eval1"
            "/eval_config/eval_config1/run_config/run_config1/results",
            params={"split": "test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "eval" in data
        assert "eval_config" in data
        assert "run_config" in data

        assert len(data["results"]) == 1
        result = data["results"][0]
        assert result["eval_run"]["scores"] == {"score1": 3.5, "overall_rating": 4.0}
        assert result["eval_run"]["dataset_id"] == in_split.id
        assert result["eval_run"]["task_run_config_id"] == "run_config1"
        assert result["input"] == "hello"
        assert result["output"] == "world"

        for v2_field in (
            "eval_input_id",
            "skipped_reason",
            "skipped_detail",
        ):
            assert v2_field in result["eval_run"]
            assert result["eval_run"][v2_field] is None

    def test_v1_llm_as_judge_config_accepted(
        self,
        mock_eval,
    ):
        config = EvalConfig(
            name="LLM Judge V1",
            config_type=EvalConfigType.llm_as_judge,
            model_name="gpt-4",
            model_provider="openai",
            properties={"eval_steps": ["judge step"]},
            parent=mock_eval,
        )
        config.save_to_file()

        loaded = EvalConfig.load_from_file(str(config.path))
        assert loaded.config_type == EvalConfigType.llm_as_judge
        assert isinstance(loaded.properties, dict)
        assert loaded.properties["eval_steps"] == ["judge step"]
        assert loaded.model_name == "gpt-4"
        assert loaded.model_provider == "openai"

    def test_v1_score_summary_with_v2_optional_fields_on_runs(
        self,
        mock_eval_for_score_summary,
    ):
        config = Mock(spec=EvalConfig)

        runs = [
            EvalRun(
                task_run_config_id="rc1",
                scores={"accuracy": 0.9, "relevance": 0.8},
                input="input1",
                output="output1",
                dataset_id="ds1",
            ),
            EvalRun(
                task_run_config_id="rc1",
                scores={"accuracy": 0.7, "relevance": 0.6},
                input="input2",
                output="output2",
                dataset_id="ds2",
            ),
        ]
        for r in runs:
            assert r.eval_input_id is None
            assert r.skipped_reason is None

        config.runs.return_value = runs

        task_run_configs = [Mock(spec=TaskRunConfig, id="rc1")]

        result = compute_score_summary(
            mock_eval_for_score_summary,
            config,
            task_run_configs,
            stub_split({"ds1", "ds2"}),
        )

        assert result.dataset_size == 2
        scores = result.results["rc1"]
        assert scores["accuracy"].mean_score == pytest.approx(0.8)
        assert scores["accuracy"].n_used == 2
        assert scores["accuracy"].n_excluded == 0
        assert scores["relevance"].mean_score == pytest.approx(0.7)
        assert scores["relevance"].n_used == 2
        assert scores["relevance"].n_excluded == 0


class TestDefaultLlmJudgePrompt:
    def _url(self, eval_id: str = "eval_v2") -> str:
        return f"/api/projects/project1/tasks/task1/evals/{eval_id}/default_llm_judge_prompt"

    def test_returns_rich_prompt(self, client, mock_v2_eval):
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            response = client.get(self._url())
        assert response.status_code == 200
        body = response.json()
        assert "judge_prompt" in body
        assert "system_prompt" in body
        assert body["system_prompt"] == "You are an evaluator."
        assert "{{ task_input }}" in body["judge_prompt"]
        assert "{{ final_message }}" in body["judge_prompt"]
        # No spec or derivable template: the steps section defers to the
        # judge_instructions binding instead of baking score instructions.
        assert "{{ judge_instructions }}" in body["judge_prompt"]

    def test_reference_keys_match_what_create_will_require(
        self, client, mock_v2_reference_answer_eval
    ):
        """The builder is told what the saved judge will require so the Test Judge pane
        can offer a place to supply it. Derived server-side, not re-read from the
        prompt's text: a user who edits the reference block out has not made the eval
        stop requiring ground truth, and a pane reading only the prompt would then hide
        the one input that keeps its test runs from skipping."""
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_reference_answer_eval
            default = client.get(self._url("eval_v2_reference_answer"))
            created = client.post(
                "/api/projects/project1/tasks/task1/evals/eval_v2_reference_answer"
                "/create_llm_judge_config",
                json={
                    "model_name": "gpt-4o",
                    "provider": "openai",
                    "g_eval": False,
                    # The edited-down prompt the user saves: no reference block left.
                    "judge_prompt": "Score {{ final_message }} for accuracy.",
                },
            )

        assert default.status_code == 200
        assert created.status_code == 200
        assert default.json()["reference_keys"] == ["reference_answer"]
        assert (
            default.json()["reference_keys"]
            == created.json()["properties"]["reference_keys"]
        )

    def test_reference_keys_empty_for_an_ordinary_eval(self, client, mock_v2_eval):
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            response = client.get(self._url())
        assert response.status_code == 200
        assert response.json()["reference_keys"] == []

    def test_eval_not_found(self, client):
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.side_effect = HTTPException(
                status_code=404, detail="Eval not found. ID: bad_id"
            )
            response = client.get(self._url("bad_id"))
        assert response.status_code == 404


class TestCreateLlmJudgeConfigOverrides:
    def _url(self, eval_id: str = "eval_v2") -> str:
        return f"/api/projects/project1/tasks/task1/evals/{eval_id}/create_llm_judge_config"

    def test_with_judge_prompt_override(self, client, mock_v2_eval):
        custom_prompt = "Custom {{ task_input }} {{ final_message }}"
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            response = client.post(
                self._url(),
                json={
                    "model_name": "gpt-4o",
                    "provider": "openai",
                    "g_eval": False,
                    "judge_prompt": custom_prompt,
                },
            )
        assert response.status_code == 200
        props = response.json()["properties"]
        assert props["prompt_template"] == custom_prompt

    def test_with_system_prompt_override(self, client, mock_v2_eval):
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            response = client.post(
                self._url(),
                json={
                    "model_name": "gpt-4o",
                    "provider": "openai",
                    "g_eval": False,
                    "system_prompt": "Be very strict.",
                },
            )
        assert response.status_code == 200
        assert response.json()["properties"]["system_prompt"] == "Be very strict."

    def test_empty_judge_prompt_uses_default(self, client, mock_v2_eval):
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            response = client.post(
                self._url(),
                json={
                    "model_name": "gpt-4o",
                    "provider": "openai",
                    "g_eval": False,
                    "judge_prompt": "   ",
                },
            )
        assert response.status_code == 200
        props = response.json()["properties"]
        assert "<steps>" in props["prompt_template"]

    def test_invalid_jinja_returns_400(self, client, mock_v2_eval):
        with patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid:
            mock_eid.return_value = mock_v2_eval
            response = client.post(
                self._url(),
                json={
                    "model_name": "gpt-4o",
                    "provider": "openai",
                    "g_eval": False,
                    "judge_prompt": "{% invalid %}",
                },
            )
        assert response.status_code == 400


class TestTestV2EvalDraft:
    """The creation-flow endpoint: tests a judge for an eval that doesn't exist."""

    def _url(self) -> str:
        return "/api/projects/project1/tasks/task1/test_v2_eval_draft"

    def _payload(self) -> dict:
        return {
            "properties": {
                "type": "exact_match",
                "expected_value": "hello",
            },
            "output_scores": [
                {
                    "name": "accuracy",
                    "instruction": "Is the answer accurate?",
                    "type": "pass_fail",
                }
            ],
            "eval_input": {
                "final_message": "hello",
            },
        }

    def test_exact_match_pass(self, client, mock_task, mock_task_from_id):
        mock_task_from_id.return_value = mock_task
        response = client.post(self._url(), json=self._payload())
        assert response.status_code == 200
        body = response.json()
        assert body["scores"]["accuracy"] == 1.0
        assert body["skipped_reason"] is None

    def test_exact_match_fail(self, client, mock_task, mock_task_from_id):
        mock_task_from_id.return_value = mock_task
        payload = self._payload()
        payload["eval_input"]["final_message"] = "world"
        response = client.post(self._url(), json=payload)
        assert response.status_code == 200
        assert response.json()["scores"]["accuracy"] == 0.0

    def test_nothing_is_persisted(self, client, mock_task, mock_task_from_id):
        mock_task_from_id.return_value = mock_task
        response = client.post(self._url(), json=self._payload())
        assert response.status_code == 200
        # The transient eval must never be saved to the task.
        assert mock_task.evals() == []

    def test_code_eval_untrusted_skip(self, client, mock_task, mock_task_from_id):
        payload = self._payload()
        payload["properties"] = {
            "type": "code_eval",
            "code": "def score(output, **kwargs):\n    return {'accuracy': 1.0}\n",
        }
        with (
            patch("app.desktop.studio_server.eval_api.project_from_id") as mock_proj,
            patch(
                "app.desktop.studio_server.eval_api.has_add_code_trust",
                return_value=False,
            ),
        ):
            mock_task_from_id.return_value = mock_task
            mock_proj.return_value = Mock()
            response = client.post(self._url(), json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["scores"] == {}
        assert body["skipped_reason"] == "code_eval_not_trusted"

    def test_score_range_errors_reported(self, client, mock_task, mock_task_from_id):
        # A code judge returning an out-of-range value for a pass_fail score.
        payload = self._payload()
        payload["properties"] = {
            "type": "code_eval",
            "code": "def score(output, **kwargs):\n    return {'accuracy': 3.0}\n",
        }
        with (
            patch("app.desktop.studio_server.eval_api.project_from_id") as mock_proj,
            patch(
                "app.desktop.studio_server.eval_api.has_add_code_trust",
                return_value=True,
            ),
        ):
            mock_task_from_id.return_value = mock_task
            mock_proj.return_value = Mock()
            response = client.post(self._url(), json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["score_range_errors"]

    def test_empty_output_scores_400(self, client, mock_task, mock_task_from_id):
        mock_task_from_id.return_value = mock_task
        payload = self._payload()
        payload["output_scores"] = []
        response = client.post(self._url(), json=payload)
        assert response.status_code == 400

    def test_llm_judge_through_transient_eval(
        self, client, mock_task, mock_task_from_id
    ):
        # The transient eval has no path; the LLM judge adapter must still be
        # able to build its score schema from the drafted output_scores.
        payload = self._payload()
        payload["properties"] = {
            "type": "llm_judge",
            "model_name": "gpt-4o",
            "model_provider": "openai",
            "prompt_template": "Is this correct? Output: {{ final_message }}",
        }
        mock_run_output = RunOutput(
            output={"accuracy": "pass"},
            intermediate_outputs=None,
        )
        # The adapter always returns the judge's own TaskRun alongside the output;
        # only its usage is read, and it must be a real Usage or None because
        # V2EvalResult validates it.
        judge_run = MagicMock()
        judge_run.usage = None
        mock_adapter = MagicMock()
        mock_adapter.invoke_returning_run_output = AsyncMock(
            return_value=(judge_run, mock_run_output)
        )
        with patch(
            "kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task",
            return_value=mock_adapter,
        ):
            mock_task_from_id.return_value = mock_task
            response = client.post(self._url(), json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "accuracy" in body["scores"]
        assert body["skipped_reason"] is None


class TestTestV2EvalOverrides:
    def _url(self, eval_id: str = "eval_v2") -> str:
        return f"/api/projects/project1/tasks/task1/evals/{eval_id}/test_v2_eval"

    def test_llm_judge_builder_passes_overrides(self, client, mock_v2_eval):
        with (
            patch("app.desktop.studio_server.eval_api.eval_from_id") as mock_eid,
            patch(
                "app.desktop.studio_server.eval_api.materialize_llm_judge_properties"
            ) as mock_mat,
            patch(
                "app.desktop.studio_server.eval_api.v2_eval_adapter_from_config"
            ) as mock_adapter_factory,
        ):
            mock_eid.return_value = mock_v2_eval

            from kiln_ai.datamodel.eval import LlmJudgeProperties

            mock_mat.return_value = LlmJudgeProperties(
                model_name="gpt-4o",
                model_provider="openai",
                prompt_template="Custom {{ task_input }} {{ final_message }}",
                system_prompt="Be strict.",
                thinking_instruction="Think.",
                g_eval=False,
            )

            from kiln_ai.datamodel.eval import V2EvalResult

            mock_adapter = MagicMock()
            mock_adapter.evaluate = AsyncMock(
                return_value=V2EvalResult(scores={"accuracy": 1.0})
            )
            mock_adapter_factory.return_value = mock_adapter

            response = client.post(
                self._url(),
                json={
                    "eval_input": {
                        "final_message": "test output",
                        "task_input": "test input",
                    },
                    "llm_judge_builder_input": {
                        "model_name": "gpt-4o",
                        "provider": "openai",
                        "g_eval": False,
                        "judge_prompt": "Custom {{ task_input }} {{ final_message }}",
                        "system_prompt": "Be strict.",
                    },
                },
            )
        assert response.status_code == 200
        mock_mat.assert_called_once()
        call_kwargs = mock_mat.call_args
        assert (
            call_kwargs.kwargs["judge_prompt"]
            == "Custom {{ task_input }} {{ final_message }}"
        )
        assert call_kwargs.kwargs["system_prompt"] == "Be strict."


def make_multi_turn_eval_input(mock_task, tags: list[str], text: str = "seed"):
    eval_input = EvalInput(
        data=MultiTurnSyntheticEvalInputData(
            first_message=UserMessage(text=text),
            synthetic_user_info={"persona": "p", "goal": "g"},
        ),
        reference={"scenario": "s1", "expected_facts": ["fact one"]},
        tags=tags,
        parent=mock_task,
    )
    eval_input.save_to_file()
    return eval_input


def test_list_eval_inputs_empty(client, mock_task, mock_task_from_id):
    response = client.get("/api/projects/project1/tasks/task1/eval_inputs")

    assert response.status_code == 200
    assert response.json() == {"eval_inputs": [], "load_error_count": 0}


def test_list_eval_inputs_all_and_filtered(client, mock_task, mock_task_from_id):
    tagged = make_multi_turn_eval_input(mock_task, tags=["corpus", "nm_app_crit"])
    corpus_only = make_multi_turn_eval_input(mock_task, tags=["corpus"])

    response = client.get("/api/projects/project1/tasks/task1/eval_inputs")
    assert response.status_code == 200
    result = response.json()
    assert {item["id"] for item in result["eval_inputs"]} == {
        tagged.id,
        corpus_only.id,
    }
    # Every item read cleanly, so a caller has no reason to warn about a partial list.
    assert result["load_error_count"] == 0

    response = client.get(
        "/api/projects/project1/tasks/task1/eval_inputs",
        params={"filter_id": "tag::nm_app_crit"},
    )
    assert response.status_code == 200
    result = response.json()
    assert [item["id"] for item in result["eval_inputs"]] == [tagged.id]
    assert result["eval_inputs"][0]["reference"] == {
        "scenario": "s1",
        "expected_facts": ["fact one"],
    }

    response = client.get(
        "/api/projects/project1/tasks/task1/eval_inputs",
        params={"filter_id": "all"},
    )
    assert response.status_code == 200
    assert len(response.json()["eval_inputs"]) == 2


def test_list_eval_inputs_partial_load(client, mock_task, mock_task_from_id, caplog):
    """An item file this build can't parse is counted, not fatal: failing the whole list
    would hide a readable corpus behind one bad file. The count is all the response
    carries, so the log has to name the file that failed."""
    readable = make_multi_turn_eval_input(mock_task, tags=["corpus"])

    # An item written by a hypothetical newer Kiln: this build refuses to load it.
    unreadable_dir = mock_task.path.parent / "eval_inputs" / "future_item"
    unreadable_dir.mkdir(parents=True)
    unreadable_file = unreadable_dir / EvalInput.base_filename()
    unreadable_file.write_text(
        json.dumps(
            {
                "v": readable.max_schema_version() + 1,
                "id": "future_item",
                "model_type": "eval_input",
                "data": {"type": "single_turn", "user_message": {"text": "hi"}},
            }
        )
    )

    with caplog.at_level(logging.WARNING, logger="app.desktop.studio_server.eval_api"):
        response = client.get("/api/projects/project1/tasks/task1/eval_inputs")

    assert response.status_code == 200
    result = response.json()
    assert [item["id"] for item in result["eval_inputs"]] == [readable.id]
    assert result["load_error_count"] == 1
    warning = next(
        r for r in caplog.records if "Failed to load eval input file" in r.getMessage()
    )
    assert str(unreadable_file) in warning.getMessage()


def test_list_eval_inputs_partial_load_still_filters(
    client, mock_task, mock_task_from_id
):
    """The filter applies to what loaded, and the error count survives it: a caller
    asking for one slice still needs to know the corpus was read incompletely."""
    tagged = make_multi_turn_eval_input(mock_task, tags=["corpus", "nm_app_crit"])
    make_multi_turn_eval_input(mock_task, tags=["corpus"])

    unreadable_dir = mock_task.path.parent / "eval_inputs" / "future_item"
    unreadable_dir.mkdir(parents=True)
    (unreadable_dir / EvalInput.base_filename()).write_text(
        json.dumps(
            {
                "v": tagged.max_schema_version() + 1,
                "id": "future_item",
                "model_type": "eval_input",
                "data": {"type": "single_turn", "user_message": {"text": "hi"}},
            }
        )
    )

    response = client.get(
        "/api/projects/project1/tasks/task1/eval_inputs",
        params={"filter_id": "tag::nm_app_crit"},
    )

    assert response.status_code == 200
    result = response.json()
    assert [item["id"] for item in result["eval_inputs"]] == [tagged.id]
    assert result["load_error_count"] == 1


def test_list_eval_inputs_invalid_filter(client, mock_task, mock_task_from_id):
    response = client.get(
        "/api/projects/project1/tasks/task1/eval_inputs",
        params={"filter_id": "not_a_filter"},
    )

    assert response.status_code == 422
    assert "Invalid eval-input filter ID" in response.json()["message"]


def test_get_eval_input(client, mock_task, mock_task_from_id):
    eval_input = make_multi_turn_eval_input(mock_task, tags=["corpus"])

    response = client.get(
        f"/api/projects/project1/tasks/task1/eval_inputs/{eval_input.id}"
    )
    assert response.status_code == 200
    result = response.json()
    assert result["id"] == eval_input.id
    assert result["data"]["type"] == "multi_turn_synthetic"
    assert result["data"]["first_message"]["text"] == "seed"

    response = client.get("/api/projects/project1/tasks/task1/eval_inputs/999999")
    assert response.status_code == 404


def test_create_eval_input_multi_turn(client, mock_task, mock_task_from_id):
    response = client.post(
        "/api/projects/project1/tasks/task1/eval_inputs",
        json={
            "data": {
                "type": "multi_turn_synthetic",
                "first_message": {"text": "How many open work orders?"},
                "synthetic_user_info": {
                    "persona": "maintenance manager",
                    "goal": "get an overdue-WO count",
                    "behavior_guidance": "terse",
                },
                "drive_config": {
                    "model_name": "llama_3_1_8b",
                    "model_provider": "groq",
                    "turns": 4,
                },
            },
            "reference": {"scenario": "overdue_wos", "expected_facts": ["190 open"]},
            "tags": ["corpus", "nm_app_crit"],
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["data"]["type"] == "multi_turn_synthetic"
    assert result["reference"]["scenario"] == "overdue_wos"
    assert result["tags"] == ["corpus", "nm_app_crit"]

    on_disk = mock_task.eval_inputs(readonly=True)
    assert len(on_disk) == 1
    assert on_disk[0].id == result["id"]
    # synthetic_user_info is a typed SyntheticUserInfo on this base, not a bare dict,
    # so the posted JSON has to have been coerced into the model on the way in.
    assert on_disk[0].data.synthetic_user_info.persona == "maintenance manager"
    assert on_disk[0].data.synthetic_user_info.goal == "get an overdue-WO count"
    assert on_disk[0].data.synthetic_user_info.behavior_guidance == "terse"
    # The drive config is what makes the item re-drivable, and it can never be
    # added later, so it has to survive the save rather than only the response.
    drive_config = on_disk[0].data.drive_config
    assert drive_config is not None
    assert drive_config.model_name == "llama_3_1_8b"
    assert drive_config.model_provider == "groq"
    assert drive_config.turns == 4


def test_create_eval_input_multi_turn_requires_a_drive_config(
    client, mock_task, mock_task_from_id
):
    """Without one the runner skips the item and PATCH can't add it, so it is never
    runnable."""
    response = client.post(
        "/api/projects/project1/tasks/task1/eval_inputs",
        json={
            "data": {
                "type": "multi_turn_synthetic",
                "first_message": {"text": "How many open work orders?"},
                "synthetic_user_info": {"persona": "p", "goal": "g"},
            },
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert "drive_config is required" in body["message"]
    # Located on the field the caller sent, not reported against the whole item.
    assert body["source_errors"][0]["loc"] == ["body", "data"]
    assert mock_task.eval_inputs(readonly=True) == []


@pytest.mark.parametrize(
    "first_message",
    [
        pytest.param(None, id="omitted"),
        pytest.param({"text": ""}, id="empty_text"),
    ],
)
def test_create_eval_input_multi_turn_requires_a_first_message(
    client, mock_task, mock_task_from_id, first_message
):
    """No seed text means the runner has nothing to open the conversation with, so it
    skips the item, and `data` can't be edited to add one later."""
    data = {
        "type": "multi_turn_synthetic",
        "synthetic_user_info": {"persona": "p", "goal": "g"},
        "drive_config": {
            "model_name": "llama_3_1_8b",
            "model_provider": "groq",
            "turns": 4,
        },
    }
    if first_message is not None:
        data["first_message"] = first_message

    response = client.post(
        "/api/projects/project1/tasks/task1/eval_inputs",
        json={"data": data},
    )

    assert response.status_code == 422
    body = response.json()
    assert "first_message with non-empty text is required" in body["message"]
    # Located on the field the caller sent, not reported against the whole item.
    assert body["source_errors"][0]["loc"] == ["body", "data"]
    assert mock_task.eval_inputs(readonly=True) == []


@pytest.mark.parametrize(
    "tag,expected_message",
    [
        ("has space", "Tags cannot contain spaces. Try underscores."),
        ("", "Tags cannot be empty strings"),
    ],
)
def test_create_eval_input_rejects_unusable_tags(
    client, mock_task, mock_task_from_id, tag, expected_message
):
    """A tag no tag:: filter can name would make the item unselectable."""
    response = client.post(
        "/api/projects/project1/tasks/task1/eval_inputs",
        json={
            "data": {"type": "single_turn", "user_message": {"text": "hi"}},
            "tags": [tag],
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert expected_message in body["message"]
    assert body["source_errors"][0]["loc"] == ["body", "tags"]
    assert mock_task.eval_inputs(readonly=True) == []


@pytest.mark.parametrize(
    "tag,expected_message",
    [
        ("has space", "Tags cannot contain spaces. Try underscores."),
        ("", "Tags cannot be empty strings"),
    ],
)
def test_update_eval_input_rejects_unusable_tags(
    client, mock_task, mock_task_from_id, tag, expected_message
):
    eval_input = make_multi_turn_eval_input(mock_task, tags=["corpus"])

    response = client.patch(
        f"/api/projects/project1/tasks/task1/eval_inputs/{eval_input.id}",
        json={"tags": [tag]},
    )

    assert response.status_code == 422
    body = response.json()
    assert expected_message in body["message"]
    assert body["source_errors"][0]["loc"] == ["body", "tags"]
    # The rejected request quotes the tags sent, not the stored item — an error
    # body has no business carrying the item's contents or its path on disk.
    assert body["source_errors"][0]["input"] == str([tag])
    # The rejected write must not have half-applied: the item keeps its tags.
    on_disk = mock_task.eval_inputs(readonly=True)
    assert [item.tags for item in on_disk] == [["corpus"]]


def test_create_eval_input_single_turn_defaults(client, mock_task, mock_task_from_id):
    response = client.post(
        "/api/projects/project1/tasks/task1/eval_inputs",
        json={"data": {"type": "single_turn", "user_message": {"text": "hi"}}},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["data"]["type"] == "single_turn"
    assert result["reference"] is None
    assert result["tags"] == []


def test_create_eval_input_invalid_data(client, mock_task, mock_task_from_id):
    """A malformed submodel is rejected by the shape check, before any of the
    eval-input rules get a look in. Everything else here is valid so the 422 can only
    be about first_message's missing `text`, and the error points straight at it."""
    response = client.post(
        "/api/projects/project1/tasks/task1/eval_inputs",
        json={
            "data": {
                "type": "multi_turn_synthetic",
                "first_message": {},
                "synthetic_user_info": {"persona": "p", "goal": "g"},
                "drive_config": {
                    "model_name": "llama_3_1_8b",
                    "model_provider": "groq",
                    "turns": 4,
                },
            }
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["source_errors"][0]["loc"][-2:] == ["first_message", "text"]
    assert mock_task.eval_inputs(readonly=True) == []


def test_update_eval_input_tags(client, mock_task, mock_task_from_id):
    """A retag leaves content byte-identical."""
    eval_input = make_multi_turn_eval_input(mock_task, tags=["corpus"])

    response = client.patch(
        f"/api/projects/project1/tasks/task1/eval_inputs/{eval_input.id}",
        json={"tags": ["corpus", "val_split"]},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["tags"] == ["corpus", "val_split"]
    assert result["reference"] == {"scenario": "s1", "expected_facts": ["fact one"]}
    assert result["data"]["first_message"]["text"] == "seed"

    on_disk = mock_task.eval_inputs(readonly=True)[0]
    assert on_disk.tags == ["corpus", "val_split"]
    assert on_disk.reference == {"scenario": "s1", "expected_facts": ["fact one"]}
    assert on_disk.data.first_message.text == "seed"


def test_update_eval_input_tags_can_empty_the_list(
    client, mock_task, mock_task_from_id
):
    """Removing every tag takes the item out of every tag:: slice. It is a replacement,
    not a merge, so an empty list has to be accepted rather than read as 'unset'."""
    eval_input = make_multi_turn_eval_input(mock_task, tags=["corpus", "val_split"])

    response = client.patch(
        f"/api/projects/project1/tasks/task1/eval_inputs/{eval_input.id}",
        json={"tags": []},
    )

    assert response.status_code == 200
    assert response.json()["tags"] == []
    assert mock_task.eval_inputs(readonly=True)[0].tags == []


def test_update_eval_input_null_tags_is_rejected(client, mock_task, mock_task_from_id):
    """null is not the spelling for "remove every tag" — [] is. Accepting it as
    "unchanged" would drop an edit the caller believes landed."""
    eval_input = make_multi_turn_eval_input(mock_task, tags=["corpus"])

    response = client.patch(
        f"/api/projects/project1/tasks/task1/eval_inputs/{eval_input.id}",
        json={"tags": None},
    )

    assert response.status_code == 422
    assert "Send [] to remove every tag" in response.json()["message"]
    assert mock_task.eval_inputs(readonly=True)[0].tags == ["corpus"]


def test_update_eval_input_reference(client, mock_task, mock_task_from_id):
    """Correcting ground truth is an in-place edit.

    It keys nothing: stored scores snapshot the reference the judge actually saw, and
    drive fingerprints hash the scenario rather than the reference, so nothing already on
    disk is invalidated. Iterating on reference data is normal corpus authoring, and
    forcing it through a new item would leave one dead item behind per correction.
    """
    eval_input = make_multi_turn_eval_input(mock_task, tags=["corpus"])

    response = client.patch(
        f"/api/projects/project1/tasks/task1/eval_inputs/{eval_input.id}",
        json={
            "reference": {"scenario": "s1", "expected_facts": ["the corrected fact"]}
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["reference"] == {
        "scenario": "s1",
        "expected_facts": ["the corrected fact"],
    }
    # The whole dict is replaced, and the rest of the item is untouched.
    assert result["tags"] == ["corpus"]
    assert result["data"]["first_message"]["text"] == "seed"

    on_disk = mock_task.eval_inputs(readonly=True)[0]
    assert on_disk.reference == {
        "scenario": "s1",
        "expected_facts": ["the corrected fact"],
    }
    assert on_disk.data.first_message.text == "seed"


def test_update_eval_input_reference_and_tags_together(
    client, mock_task, mock_task_from_id
):
    eval_input = make_multi_turn_eval_input(mock_task, tags=["corpus"])

    response = client.patch(
        f"/api/projects/project1/tasks/task1/eval_inputs/{eval_input.id}",
        json={"tags": ["corpus", "fixed"], "reference": {"scenario": "s2"}},
    )

    assert response.status_code == 200
    on_disk = mock_task.eval_inputs(readonly=True)[0]
    assert on_disk.tags == ["corpus", "fixed"]
    assert on_disk.reference == {"scenario": "s2"}


def test_update_eval_input_null_reference_clears_it(
    client, mock_task, mock_task_from_id
):
    """Explicit null clears ground truth; omitting the field leaves it alone. The two
    are different requests, which is why the handler reads model_fields_set rather than
    testing for None — a None test would make clearing impossible."""
    eval_input = make_multi_turn_eval_input(mock_task, tags=["corpus"])

    response = client.patch(
        f"/api/projects/project1/tasks/task1/eval_inputs/{eval_input.id}",
        json={"reference": None},
    )

    assert response.status_code == 200
    assert response.json()["reference"] is None
    assert mock_task.eval_inputs(readonly=True)[0].reference is None


def test_update_eval_input_omitting_reference_leaves_it_unchanged(
    client, mock_task, mock_task_from_id
):
    """The other half of the pair above: a tags-only patch must not clear ground truth."""
    eval_input = make_multi_turn_eval_input(mock_task, tags=["corpus"])

    response = client.patch(
        f"/api/projects/project1/tasks/task1/eval_inputs/{eval_input.id}",
        json={"tags": ["corpus", "val_split"]},
    )

    assert response.status_code == 200
    assert response.json()["reference"] == {
        "scenario": "s1",
        "expected_facts": ["fact one"],
    }
    assert mock_task.eval_inputs(readonly=True)[0].reference == {
        "scenario": "s1",
        "expected_facts": ["fact one"],
    }


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            {
                "tags": ["corpus"],
                "data": {
                    "type": "multi_turn_synthetic",
                    "first_message": {"text": "new seed"},
                    "synthetic_user_info": {"persona": "p2", "goal": "g2"},
                },
            },
            id="data_alongside_tags",
        ),
        pytest.param(
            {"data": {"type": "single_turn", "user_message": {"text": "hi"}}},
            id="data_only",
        ),
    ],
)
def test_update_eval_input_rejects_scenario_edits(
    client, mock_task, mock_task_from_id, body
):
    """A scenario edit must fail loudly, not be silently dropped.

    Trace reuse keys on the item id, so editing `data` in place would let a later eval
    hand a judge a conversation generated from the scenario the item used to have.
    Changing a scenario is a POST of a new item. `extra="forbid"` is what makes the
    attempt a 422 instead of a no-op the caller reads as success — including when `data`
    rides along with an otherwise-valid tags edit, which must not half-apply.
    """
    eval_input = make_multi_turn_eval_input(mock_task, tags=["corpus"])

    response = client.patch(
        f"/api/projects/project1/tasks/task1/eval_inputs/{eval_input.id}",
        json=body,
    )

    assert response.status_code == 422

    on_disk = mock_task.eval_inputs(readonly=True)[0]
    assert on_disk.data.first_message.text == "seed"
    assert on_disk.reference == {"scenario": "s1", "expected_facts": ["fact one"]}
    assert on_disk.tags == ["corpus"]


def test_update_eval_input_404(client, mock_task, mock_task_from_id):
    response = client.patch(
        "/api/projects/project1/tasks/task1/eval_inputs/999999",
        json={"tags": ["x"]},
    )

    assert response.status_code == 404


def test_delete_eval_input(client, mock_task, mock_task_from_id):
    eval_input = make_multi_turn_eval_input(mock_task, tags=["corpus"])
    keep = make_multi_turn_eval_input(mock_task, tags=["corpus"], text="keep")

    response = client.delete(
        f"/api/projects/project1/tasks/task1/eval_inputs/{eval_input.id}"
    )
    assert response.status_code == 200

    on_disk = mock_task.eval_inputs(readonly=True)
    assert [item.id for item in on_disk] == [keep.id]

    response = client.get(
        f"/api/projects/project1/tasks/task1/eval_inputs/{eval_input.id}"
    )
    assert response.status_code == 404


def test_delete_eval_input_blocked_by_eval_trace(
    client, mock_task, mock_task_from_id, data_source
):
    """A trace names its item by id and holds no copy of it, so deleting the item would
    leave a conversation nothing can say the scenario for."""
    eval_input = make_multi_turn_eval_input(mock_task, tags=["corpus"])

    trace = TaskRun(
        parent=mock_task,
        input="seed",
        input_source=data_source,
        output=TaskOutput(output="response", source=data_source),
        eval_source=EvalItemSource(
            source_type="eval_input", source_id=str(eval_input.id)
        ),
    )
    trace.save_to_file()

    response = client.delete(
        f"/api/projects/project1/tasks/task1/eval_inputs/{eval_input.id}"
    )

    assert response.status_code == 409
    assert "1 eval trace(s)" in response.json()["message"]
    assert "0 score record(s)" in response.json()["message"]
    assert [item.id for item in mock_task.eval_inputs(readonly=True)] == [eval_input.id]


def test_delete_eval_input_blocked_by_score_record(
    client, mock_task, mock_task_from_id, mock_eval_config, data_source
):
    """A stored score names the item it scored. Deleting the item would leave the score
    describing an input that can no longer be read back."""
    eval_input = make_multi_turn_eval_input(mock_task, tags=["corpus"])

    # A scored run with no `eval_source`, so this pins the score-record half of the guard
    # on its own: the trace count stays 0 and only `EvalRun.eval_input_id` blocks.
    scored_run = TaskRun(
        parent=mock_task,
        input="seed",
        input_source=data_source,
        output=TaskOutput(output="response", source=data_source),
    )
    scored_run.save_to_file()

    EvalRun(
        parent=mock_eval_config,
        task_run_config_id="run_config1",
        eval_input_id=eval_input.id,
        scored_run_id=scored_run.id,
        scores={"score1": 4.0, "overall_rating": 4.0},
    ).save_to_file()

    response = client.delete(
        f"/api/projects/project1/tasks/task1/eval_inputs/{eval_input.id}"
    )

    assert response.status_code == 409
    assert "0 eval trace(s)" in response.json()["message"]
    assert "1 score record(s)" in response.json()["message"]
    assert [item.id for item in mock_task.eval_inputs(readonly=True)] == [eval_input.id]


def test_delete_eval_input_ignores_references_to_other_items(
    client, mock_task, mock_task_from_id, mock_eval_config, data_source
):
    """The guard is keyed on this item, not on 'the task has eval records at all'.

    Also pins that a `task_run`-sourced trace whose source_id happens to equal this
    EvalInput's id does not count: ids are only unique within a store, so matching on the
    id alone would block deletes for a record about a different item entirely.
    """
    target = make_multi_turn_eval_input(mock_task, tags=["corpus"])
    other = make_multi_turn_eval_input(mock_task, tags=["corpus"], text="other")

    TaskRun(
        parent=mock_task,
        input="seed",
        input_source=data_source,
        output=TaskOutput(output="response", source=data_source),
        eval_source=EvalItemSource(source_type="eval_input", source_id=str(other.id)),
    ).save_to_file()
    TaskRun(
        parent=mock_task,
        input="seed",
        input_source=data_source,
        output=TaskOutput(output="response", source=data_source),
        eval_source=EvalItemSource(source_type="task_run", source_id=str(target.id)),
    ).save_to_file()
    other_scored_run = TaskRun(
        parent=mock_task,
        input="seed",
        input_source=data_source,
        output=TaskOutput(output="response", source=data_source),
    )
    other_scored_run.save_to_file()
    EvalRun(
        parent=mock_eval_config,
        task_run_config_id="run_config1",
        eval_input_id=other.id,
        scored_run_id=other_scored_run.id,
        scores={"score1": 4.0, "overall_rating": 4.0},
    ).save_to_file()

    response = client.delete(
        f"/api/projects/project1/tasks/task1/eval_inputs/{target.id}"
    )

    assert response.status_code == 200
    assert [item.id for item in mock_task.eval_inputs(readonly=True)] == [other.id]


@pytest.mark.asyncio
async def test_run_calibration_empty_golden_set_400(
    client, mock_task_from_id, mock_task, mock_eval, mock_eval_config
):
    """No runs match the golden filter: calibration would complete vacuously
    (zero jobs, zero scores) and read as success — refuse it up front."""
    mock_task_from_id.return_value = mock_task

    response = client.get(
        "/api/projects/project1/tasks/task1/evals/eval1/run_calibration"
    )

    assert response.status_code == 400
    assert "golden dataset is empty" in response.json()["message"]


@pytest.mark.asyncio
async def test_run_comparison_multi_turn_drive_problems_400(
    client, mock_task_from_id, mock_task, mock_eval, mock_eval_config, mock_run_config
):
    """Multi-turn readiness problems (unstamped items, unknown synthetic-user
    providers, non-agent run configs) surface as one 400 before the SSE
    stream opens, not as N anonymous per-job errors."""
    mock_task_from_id.return_value = mock_task

    with (
        patch(
            "app.desktop.studio_server.eval_api.task_run_config_from_id"
        ) as mock_run_config_from_id,
        patch.object(
            EvalRunner,
            "validate_multi_turn_drive_readiness",
            side_effect=ValueError("run config 'MCP one' is not a Kiln agent config"),
        ),
    ):
        mock_run_config_from_id.return_value = mock_run_config
        response = client.get(
            "/api/projects/project1/tasks/task1/evals/eval1/eval_config/eval_config1/run_comparison",
            params={"run_config_ids": ["run_config1"]},
        )

    assert response.status_code == 400
    assert "MCP one" in response.json()["message"]


# ── Multi-turn item count: restored positive-case coverage for the count
# re-expressed from the resolved split (stored conversations = chain leaves). ──


def _multiturn_task_with_eval(tmp_path, evaluation_data_type: EvalDataType) -> Task:
    """A real on-disk multiturn task with an eval (id eval1) filtering on
    tag::eval_set and a judge config (id eval_config1)."""
    project = Project(
        id="project1", name="Test Project", path=tmp_path / "project.kiln"
    )
    project.save_to_file()
    task = Task(
        id="task1",
        name="Test Task",
        instruction="Test Instructions",
        path=tmp_path / "task.kiln",
        turn_mode=TurnMode.multiturn,
        parent=project,
    )
    task.save_to_file()

    eval = Eval(
        id="eval1",
        name="Eval",
        output_scores=[
            EvalOutputScore(
                name="score1", instruction="desc1", type=TaskOutputRatingType.pass_fail
            ),
        ],
        eval_set_filter_id="tag::eval_set",
        eval_configs_filter_id="tag::golden",
        evaluation_data_type=evaluation_data_type,
        parent=task,
    )
    eval.save_to_file()
    EvalConfig(
        id="eval_config1",
        name="Judge",
        config_type=EvalConfigType.g_eval,
        properties={"eval_steps": ["step1"]},
        model_name="gpt-4",
        model_provider="openai",
        parent=eval,
    ).save_to_file()
    return task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "evaluation_data_type",
    [EvalDataType.final_answer, EvalDataType.full_trace],
)
async def test_get_eval_config_score_summary_multi_turn_item_count(
    client, mock_task_from_id, tmp_path, evaluation_data_type
):
    """multi_turn_item_count counts the stored conversations (chain leaves) in
    the eval set. It's a property of the item set alone, so it must be the same
    for final_answer and full_trace evals."""
    task = _multiturn_task_with_eval(tmp_path, evaluation_data_type)
    mock_task_from_id.return_value = task

    output = TaskOutput(output="test output")
    # Single-turn item in the eval set: regenerated per run config.
    TaskRun(input="i1", output=output, tags=["eval_set"], parent=task).save_to_file()
    # Stored conversation in the eval set: only its leaf is an eval item.
    root = TaskRun(input="i2", output=output, parent=task)
    root.save_to_file()
    TaskRun(
        input="i3",
        output=output,
        tags=["eval_set"],
        parent=task,
        parent_task_run_id=root.id,
    ).save_to_file()
    # Stored conversation outside the eval set: must not count.
    other_root = TaskRun(input="i4", output=output, parent=task)
    other_root.save_to_file()
    TaskRun(
        input="i5",
        output=output,
        tags=["other"],
        parent=task,
        parent_task_run_id=other_root.id,
    ).save_to_file()

    response = client.get(
        "/api/projects/project1/tasks/task1/evals/eval1/eval_config/eval_config1/score_summary"
    )

    assert response.status_code == 200
    result = response.json()
    assert result["dataset_size"] == 2
    assert result["multi_turn_item_count"] == 1


@pytest.mark.asyncio
async def test_get_eval_config_score_summary_single_turn_only_set(
    client, mock_task_from_id, tmp_path
):
    """A full_trace eval whose set has no stored conversations reports zero
    multi-turn items — every item regenerates per run config."""
    task = _multiturn_task_with_eval(tmp_path, EvalDataType.full_trace)
    mock_task_from_id.return_value = task

    output = TaskOutput(output="test output")
    TaskRun(input="i1", output=output, tags=["eval_set"], parent=task).save_to_file()
    TaskRun(input="i2", output=output, tags=["eval_set"], parent=task).save_to_file()

    response = client.get(
        "/api/projects/project1/tasks/task1/evals/eval1/eval_config/eval_config1/score_summary"
    )

    assert response.status_code == 200
    result = response.json()
    assert result["dataset_size"] == 2
    assert result["multi_turn_item_count"] == 0


@pytest.mark.asyncio
async def test_eval_results_summary_emits_eval_input_backed_eval(client):
    """End-to-end wiring for an EvalInput-backed eval in the task-wide summary:
    its dataset size and scores must be emitted, not skipped."""
    output_scores = [
        EvalOutputScore(
            name="accuracy",
            instruction="Test accuracy",
            type=TaskOutputRatingType.pass_fail,
        ),
    ]
    eval_runs = [
        EvalRun(
            task_run_config_id="rc1",
            scores={"accuracy": 1.0},
            input="i",
            output="o",
            eval_input_id="ei1",
        ),
    ]
    ec = _build_mock_eval_config("ec1", "Judge", eval_runs)
    eval1 = _build_mock_eval(
        eval_id="eval1",
        name="Eval One",
        current_config_id="ec1",
        output_scores=output_scores,
        configs=[ec],
        test_split=EvalInputSplit(filter_id="tag::cases"),
    )

    rc1_mock = Mock(spec=TaskRunConfig, id="rc1")
    rc1_mock.name = "RC1"

    mock_task = Mock(spec=Task)
    mock_task.run_configs.return_value = [rc1_mock]
    mock_task.finetunes.return_value = []
    mock_task.runs.return_value = []
    mock_task.evals.return_value = [eval1]

    with (
        patch("app.desktop.studio_server.eval_api.task_from_id") as mock_task_from_id,
        patch_resolve_split_by_ref({("eval_input", "tag::cases"): {"ei1", "ei2"}}),
    ):
        mock_task_from_id.return_value = mock_task

        response = client.get("/api/projects/p1/tasks/t1/eval_results_summary")

    assert response.status_code == 200
    data = response.json()
    assert data["evals_by_id"]["eval1"]["dataset_size"] == 2
    rc_scores = data["scores_by_run_config_by_eval"]["rc1"]["eval1"]
    assert rc_scores["mean_scores"]["accuracy"] == 1.0
    assert rc_scores["percent_complete"] == 0.5


@pytest.mark.asyncio
async def test_create_evaluator_generates_filters_scores_priority_status(
    client, mock_task_from_id, mock_task
):
    """Omitting filters/scores generates the same tag-based setup a spec-backed
    eval gets, and priority/status are stored on the eval."""
    response = client.post(
        "/api/projects/project1/tasks/task1/create_evaluator",
        json={
            "name": "My Issue Eval",
            "template": "kiln_issue",
            "evaluation_data_type": "final_answer",
            "priority": 2,
            "status": "future",
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["splits"]["test"]["filter_id"] == "tag::test_my_issue_eval"
    assert result["splits"]["train"]["filter_id"] == "tag::train_my_issue_eval"
    assert result["splits"]["val"]["filter_id"] == "tag::val_my_issue_eval"
    assert result["eval_configs_filter_id"] == "tag::golden_my_issue_eval"
    assert result["priority"] == 2
    assert result["status"] == "future"
    assert len(result["output_scores"]) == 1
    assert result["output_scores"][0]["name"] == "My Issue Eval"
    assert result["output_scores"][0]["type"] == "pass_fail"

    saved_eval = mock_task.evals()[0]
    assert saved_eval.priority == Priority.p2
    assert saved_eval.status == SpecStatus.future
    assert saved_eval.splits["test"].filter_id == "tag::test_my_issue_eval"


@pytest.mark.asyncio
async def test_create_evaluator_defaults_priority_and_status(
    client, mock_task_from_id, mock_task, valid_evaluator_request
):
    response = client.post(
        "/api/projects/project1/tasks/task1/create_evaluator",
        json=valid_evaluator_request.model_dump(),
    )

    assert response.status_code == 200
    saved_eval = mock_task.evals()[0]
    assert saved_eval.priority == Priority.p1
    assert saved_eval.status == SpecStatus.active


def test_update_eval_priority_and_status(
    client, mock_task_from_id, mock_eval, mock_task
):
    response = client.patch(
        "/api/projects/project1/tasks/task1/evals/eval1",
        json={"priority": 0, "status": "archived"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["priority"] == 0
    assert result["status"] == "archived"

    saved_eval = mock_task.evals()[0]
    assert saved_eval.priority == Priority.p0
    assert saved_eval.status == SpecStatus.archived


def test_get_eval_resolves_priority_status_from_spec(
    client, mock_task_from_id, mock_eval, mock_task
):
    """A legacy spec-backed eval (no own priority/status) reads through to its spec."""
    spec = Spec(
        name="Backing Spec",
        definition="definition",
        properties=DesiredBehaviourProperties(
            spec_type=SpecType.desired_behaviour,
            desired_behaviour_description="be nice",
        ),
        priority=Priority.p3,
        status=SpecStatus.deprecated,
        eval_id=mock_eval.id,
        parent=mock_task,
    )
    spec.save_to_file()

    response = client.get("/api/projects/project1/tasks/task1/evals/eval1")
    assert response.status_code == 200
    result = response.json()
    assert result["priority"] == 3
    assert result["status"] == "deprecated"

    # The resolution is response-only: the eval file keeps None so the
    # fallthrough continues to track the spec.
    saved_eval = mock_task.evals()[0]
    assert saved_eval.priority is None
    assert saved_eval.status is None


def test_get_evals_resolves_priority_status(
    client, mock_task_from_id, mock_eval, mock_task
):
    """List endpoint resolves spec-backed evals via their spec, and evals with
    no spec to defaults."""
    spec = Spec(
        name="Backing Spec",
        definition="definition",
        properties=DesiredBehaviourProperties(
            spec_type=SpecType.desired_behaviour,
            desired_behaviour_description="be nice",
        ),
        priority=Priority.p0,
        status=SpecStatus.future,
        eval_id=mock_eval.id,
        parent=mock_task,
    )
    spec.save_to_file()

    legacy_eval = Eval(
        id="legacy_eval1",
        name="Legacy Eval",
        output_scores=[
            EvalOutputScore(name="score", type=TaskOutputRatingType.pass_fail)
        ],
        eval_set_filter_id="tag::eval_set",
        eval_configs_filter_id="tag::golden",
        parent=mock_task,
    )
    legacy_eval.save_to_file()

    response = client.get("/api/projects/project1/tasks/task1/evals")
    assert response.status_code == 200
    by_id = {e["id"]: e for e in response.json()["evals"]}
    assert by_id["eval1"]["priority"] == 0
    assert by_id["eval1"]["status"] == "future"
    assert by_id["legacy_eval1"]["priority"] == 1
    assert by_id["legacy_eval1"]["status"] == "active"


@pytest.mark.asyncio
async def test_create_evaluator_rejects_long_names(
    client, mock_task_from_id, mock_task
):
    """Score names cap at 32 chars, so a longer eval name must 422 rather than
    500 while generating the default score."""
    response = client.post(
        "/api/projects/project1/tasks/task1/create_evaluator",
        json={
            "name": "a" * 33,
            "evaluation_data_type": "final_answer",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_evaluator_rejects_empty_output_scores(
    client, mock_task_from_id, mock_task
):
    """An explicit empty score list is an error, not a request for defaults."""
    response = client.post(
        "/api/projects/project1/tasks/task1/create_evaluator",
        json={
            "name": "My Eval",
            "evaluation_data_type": "final_answer",
            "output_scores": [],
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_task_run_config_rejects_duplicate_skill_names(
    client, mock_task_from_id, mock_task
):
    from kiln_ai.datamodel.skill import Skill

    mock_task_from_id.return_value = mock_task
    project = mock_task.parent_project()
    duplicate_ids = []
    for _ in range(2):
        skill = Skill(name="dup-skill", description="d", parent=project)
        skill.save_to_file()
        skill.save_skill_md("# body")
        duplicate_ids.append(skill.id)
    unique = Skill(name="unique-skill", description="d", parent=project)
    unique.save_to_file()
    unique.save_skill_md("# body")

    def request_body(skill_ids):
        return {
            "name": "RC",
            "run_config_properties": {
                "model_name": "gpt-4o",
                "model_provider_name": "openai",
                "prompt_id": "simple_chain_of_thought_prompt_builder",
                "structured_output_mode": "json_schema",
                "tools_config": {
                    "tools": [f"kiln_tool::skill::{i}" for i in skill_ids]
                },
            },
        }

    # Two versions sharing a name in one run config: rejected (skills are
    # loaded by name at runtime, one would shadow the other).
    response = client.post(
        "/api/projects/project1/tasks/task1/run_configs",
        json=request_body(duplicate_ids),
    )
    assert response.status_code == 422
    assert "Duplicate skill name 'dup-skill'" in response.text

    # Distinct names: accepted.
    response = client.post(
        "/api/projects/project1/tasks/task1/run_configs",
        json=request_body([duplicate_ids[0], unique.id]),
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_create_task_run_config_rejects_duplicate_tool_function_names(
    client, mock_task_from_id, mock_task
):
    from kiln_ai.datamodel.code_tool import CodeTool

    mock_task_from_id.return_value = mock_task
    project = mock_task.parent_project()

    def make_code_tool(function_name: str) -> CodeTool:
        code_tool = CodeTool(
            name=function_name,
            tool_function_name=function_name,
            tool_description="d",
            parameters_schema={"type": "object", "properties": {}},
            code="def run() -> str:\n    return 'ok'\n",
            parent=project,
        )
        code_tool.save_to_file()
        return code_tool

    dup_a = make_code_tool("dup_tool")
    dup_b = make_code_tool("dup_tool")
    unique = make_code_tool("unique_tool")

    def request_body(tool_ids):
        return {
            "name": "RC",
            "run_config_properties": {
                "model_name": "gpt-4o",
                "model_provider_name": "openai",
                "prompt_id": "simple_chain_of_thought_prompt_builder",
                "structured_output_mode": "json_schema",
                "tools_config": {"tools": tool_ids},
            },
        }

    # Two tools resolving to the same function name in one run config: rejected.
    response = client.post(
        "/api/projects/project1/tasks/task1/run_configs",
        json=request_body(
            [f"kiln_tool::code::{dup_a.id}", f"kiln_tool::code::{dup_b.id}"]
        ),
    )
    assert response.status_code == 422
    assert "share the same function name: dup_tool" in response.text

    # Distinct function names: accepted.
    response = client.post(
        "/api/projects/project1/tasks/task1/run_configs",
        json=request_body(
            [f"kiln_tool::code::{dup_a.id}", f"kiln_tool::code::{unique.id}"]
        ),
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_create_task_run_config_rejects_tool_colliding_with_skill_loader(
    client, mock_task_from_id, mock_task
):
    from kiln_ai.datamodel.code_tool import CodeTool
    from kiln_ai.datamodel.skill import Skill

    mock_task_from_id.return_value = mock_task
    project = mock_task.parent_project()

    code_tool = CodeTool(
        name="skill",
        tool_function_name="skill",
        tool_description="d",
        parameters_schema={"type": "object", "properties": {}},
        code="def run() -> str:\n    return 'ok'\n",
        parent=project,
    )
    code_tool.save_to_file()
    skill = Skill(name="my-skill", description="d", parent=project)
    skill.save_to_file()
    skill.save_skill_md("# body")

    # A tool named "skill" collides with the skill loader tool when skills are
    # attached: rejected with a hint about the reserved name.
    response = client.post(
        "/api/projects/project1/tasks/task1/run_configs",
        json={
            "name": "RC",
            "run_config_properties": {
                "model_name": "gpt-4o",
                "model_provider_name": "openai",
                "prompt_id": "simple_chain_of_thought_prompt_builder",
                "structured_output_mode": "json_schema",
                "tools_config": {
                    "tools": [
                        f"kiln_tool::code::{code_tool.id}",
                        f"kiln_tool::skill::{skill.id}",
                    ]
                },
            },
        },
    )
    assert response.status_code == 422
    assert "share the same function name: skill" in response.text
    assert "reserved" in response.text


@pytest.mark.asyncio
async def test_create_task_run_config_rejects_missing_skill(
    client, mock_task_from_id, mock_task
):
    mock_task_from_id.return_value = mock_task

    # A run config referencing a skill that doesn't exist in the project (e.g.
    # deleted) is rejected at creation instead of failing at runtime.
    response = client.post(
        "/api/projects/project1/tasks/task1/run_configs",
        json={
            "name": "RC",
            "run_config_properties": {
                "model_name": "gpt-4o",
                "model_provider_name": "openai",
                "prompt_id": "simple_chain_of_thought_prompt_builder",
                "structured_output_mode": "json_schema",
                "tools_config": {"tools": ["kiln_tool::skill::missing_id"]},
            },
        },
    )
    assert response.status_code == 422
    assert "not found in the project: missing_id" in response.text
    assert len(mock_task.run_configs()) == 0
