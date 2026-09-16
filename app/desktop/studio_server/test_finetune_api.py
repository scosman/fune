import unittest.mock
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kiln_ai.adapters.fine_tune.base_finetune import FineTuneParameter
from kiln_ai.adapters.fine_tune.dataset_formatter import DatasetFormat
from kiln_ai.adapters.ml_model_list import (
    KilnModel,
    KilnModelProvider,
    ModelFamily,
    ModelName,
    ModelParserID,
    ModelProviderName,
)
from kiln_ai.datamodel import (
    DatasetSplit,
    Finetune,
    Project,
    Task,
    TaskOutput,
    TaskOutputRating,
    TaskOutputRatingType,
    TaskRun,
)
from kiln_ai.datamodel.datamodel_enums import ChatStrategy, TurnMode
from kiln_ai.datamodel.dataset_filters import DatasetFilterId
from kiln_ai.datamodel.dataset_split import (
    AllSplitDefinition,
    Train60Test20Val20SplitDefinition,
    Train80Test10Val10SplitDefinition,
    Train80Test20SplitDefinition,
    Train80Val20SplitDefinition,
)
from kiln_ai.datamodel.run_config import (
    KilnAgentRunConfigProperties,
    ToolsRunConfig,
)
from kiln_ai.datamodel.skill import Skill
from kiln_server.custom_errors import connect_custom_errors
from pydantic import BaseModel

from app.desktop.studio_server.finetune_api import (
    CreateDatasetSplitRequest,
    CreateFinetuneRequest,
    DatasetSplitType,
    FinetuneProviderModel,
    compute_finetune_tag_info,
    connect_fine_tune_api,
    data_strategies_from_finetune_id,
    fetch_fireworks_finetune_models,
    infer_data_strategies_for_model,
    system_message_from_request,
    thinking_instructions_from_request,
)


@pytest.fixture
def test_task(tmp_path):
    project_path = tmp_path / "project.kiln"

    project = Project(name="Test Project", path=str(project_path))
    project.save_to_file()

    task = Task(
        name="Test Task",
        instruction="This is a test instruction",
        description="This is a test task",
        parent=project,
    )
    task.save_to_file()

    tunes = [
        Finetune(
            id="ft1",
            name="Finetune 1",
            provider="openai",
            base_model_id="model1",
            dataset_split_id="split1",
            system_message="System prompt 1",
        ),
        Finetune(
            id="ft2",
            name="Finetune 2",
            provider="openai",
            base_model_id="model2",
            dataset_split_id="split2",
            system_message="System prompt 2",
        ),
    ]
    for tune in tunes:
        tune.parent = task
        tune.save_to_file()

    splits = [
        DatasetSplit(
            id="split1",
            name="Split 1",
            split_contents={"train": ["1", "2"]},
            splits=AllSplitDefinition,
        ),
        DatasetSplit(
            id="split2",
            name="Split 2",
            split_contents={"test": ["3"]},
            splits=AllSplitDefinition,
        ),
    ]
    for split in splits:
        split.parent = task
        split.save_to_file()

    # Add some runs with tags
    runs = [
        TaskRun(
            id="run1",
            name="Run 1",
            parent=task,
            tags=["fine_tune_1", "other_tag"],
            input="Test input 1",
            input_source={"type": "human", "properties": {"created_by": "user1"}},
            # Reasoning and high rated
            intermediate_outputs={"reasoning": "thinking output"},
            output=TaskOutput(
                output="Test output 1",
                source={"type": "human", "properties": {"created_by": "user1"}},
                rating=TaskOutputRating(
                    value=5.0,
                    type=TaskOutputRatingType.five_star,
                ),
            ),
        ),
        TaskRun(
            id="run2",
            name="Run 2",
            parent=task,
            tags=["fine_tune_1", "fine_tune_2"],
            input="Test input 2",
            input_source={"type": "human", "properties": {"created_by": "user2"}},
            output=TaskOutput(
                output="Test output 2",
                source={"type": "human", "properties": {"created_by": "user2"}},
            ),
        ),
        TaskRun(
            id="run3",
            name="Run 3",
            parent=task,
            tags=["other_tag"],
            input="Test input 3",
            input_source={"type": "human", "properties": {"created_by": "user3"}},
            output=TaskOutput(
                output="Test output 3",
                source={"type": "human", "properties": {"created_by": "user3"}},
            ),
        ),
    ]
    for run in runs:
        run.save_to_file()

    return task


@pytest.fixture
def mock_task_from_id_disk_backed(test_task, monkeypatch):
    mock_func = Mock(return_value=test_task)
    monkeypatch.setattr(
        "app.desktop.studio_server.finetune_api.task_from_id", mock_func
    )
    return mock_func


@pytest.fixture
def client():
    app = FastAPI()
    connect_custom_errors(app)
    connect_fine_tune_api(app)
    return TestClient(app)


@pytest.fixture
def empty_task(tmp_path):
    project = Project(name="Test Project", path=str(tmp_path / "project.kiln"))
    project.save_to_file()

    task = Task(
        name="Test Task",
        instruction="This is a test instruction",
        description="This is a test task",
        parent=project,
    )
    task.save_to_file()
    return task


def create_synthetic_run(
    task: Task,
    *,
    run_id: str,
    name: str,
    output_text: str,
    created_by: str,
    tags: list[str] | None = None,
    tool_ids: list[str] | None = None,
) -> TaskRun:
    output_source: dict[str, object] = {
        "type": "synthetic",
        "properties": {
            "model_name": "gpt-4",
            "model_provider": "openai",
            "adapter_name": "test",
            "prompt_id": "simple_prompt_builder",
        },
    }
    if tool_ids is not None:
        output_source["run_config"] = KilnAgentRunConfigProperties(
            model_name="gpt-4",
            model_provider_name="openai",
            prompt_id="simple_prompt_builder",
            structured_output_mode="default",
            tools_config=ToolsRunConfig(tools=tool_ids),
        )

    run = TaskRun(
        id=run_id,
        name=name,
        parent=task,
        tags=tags or ["fine_tune_tools"],
        input=f"Test input {run_id}",
        input_source={"type": "human", "properties": {"created_by": created_by}},
        output=TaskOutput(
            output=output_text,
            source=output_source,
        ),
    )
    run.save_to_file()
    return run


def create_dataset_split(
    task: Task, *, split_id: str, name: str, run_ids: list[str]
) -> DatasetSplit:
    split = DatasetSplit(
        id=split_id,
        name=name,
        split_contents={"train": run_ids},
        splits=AllSplitDefinition,
    )
    split.parent = task
    split.save_to_file()
    return split


def create_finetune(
    task: Task, *, finetune_id: str, name: str, split_id: str
) -> Finetune:
    finetune = Finetune(
        id=finetune_id,
        name=name,
        provider="together_ai",
        base_model_id="model1",
        dataset_split_id=split_id,
        system_message="System prompt",
    )
    finetune.parent = task
    finetune.save_to_file()
    return finetune


def test_finetune_provider_model_defaults():
    model = FinetuneProviderModel(
        name="Test Provider",
        id="test_provider",
    )

    assert model.data_strategies_supported == [
        ChatStrategy.single_turn,
        ChatStrategy.two_message_cot,
    ]


def test_get_dataset_splits(client, mock_task_from_id_disk_backed, test_task):
    response = client.get("/api/projects/project1/tasks/task1/dataset_splits")

    assert response.status_code == 200
    splits = response.json()
    assert len(splits) == 2

    assert splits[0]["id"] in ["split1", "split2"]
    assert splits[1]["id"] in ["split1", "split2"]
    assert splits[0]["id"] != splits[1]["id"]

    mock_task_from_id_disk_backed.assert_called_once_with("project1", "task1")


def test_get_finetunes(client, mock_task_from_id_disk_backed, test_task):
    response = client.get("/api/projects/project1/tasks/task1/finetunes")

    assert response.status_code == 200
    finetunes = response.json()
    assert len(finetunes) == 2
    assert finetunes[0]["id"] in ["ft1", "ft2"]
    assert finetunes[1]["id"] in ["ft1", "ft2"]
    assert finetunes[0]["id"] != finetunes[1]["id"]

    mock_task_from_id_disk_backed.assert_called_once_with("project1", "task1")


@pytest.fixture
def mock_built_in_models():
    models = [
        KilnModel(
            name="model1",
            family="family1",
            friendly_name="Model 1",
            providers=[
                KilnModelProvider(
                    name="groq",
                    provider_finetune_id="ft_model1",
                    supports_function_calling=True,
                ),
                KilnModelProvider(
                    name="together_ai",
                    provider_finetune_id="ft_model1_p2",
                    supports_function_calling=True,
                ),
            ],
        ),
        KilnModel(
            name="model2",
            family="family2",
            friendly_name="Model 2",
            providers=[
                KilnModelProvider(
                    name="groq",
                    provider_finetune_id="ft_model2",
                    supports_function_calling=False,
                ),
                KilnModelProvider(
                    name="together_ai",
                    provider_finetune_id=None,  # This one should be skipped
                ),
            ],
        ),
    ]
    with unittest.mock.patch(
        "app.desktop.studio_server.finetune_api.built_in_models", models
    ):
        yield models


@pytest.fixture
def mock_provider_enabled():
    async def mock_enabled(provider: str) -> bool:
        return provider == "groq"

    mock = Mock()
    mock.side_effect = mock_enabled

    with unittest.mock.patch(
        "app.desktop.studio_server.finetune_api.provider_enabled", mock
    ):
        yield mock


@pytest.fixture
def mock_provider_name_from_id():
    def mock_name(provider_id: str) -> str:
        return f"Provider {provider_id.replace('provider', '')}"

    with unittest.mock.patch(
        "app.desktop.studio_server.finetune_api.provider_name_from_id", mock_name
    ):
        yield mock_name


@pytest.mark.asyncio
async def test_get_finetune_providers(
    client, mock_built_in_models, mock_provider_name_from_id, mock_provider_enabled
):
    # Mock the Fireworks API call
    with patch(
        "app.desktop.studio_server.finetune_api.fetch_fireworks_finetune_models",
        new_callable=AsyncMock,
    ) as mock_fetch:
        # Set up mock return value with one model
        mock_fetch.return_value = [
            FinetuneProviderModel(
                name="Fireworks Model",
                id="fireworks/model-1",
                supports_function_calling=True,
            )
        ]

        response = client.get("/api/finetune_providers")

        # Verify the mock was called
        mock_fetch.assert_called_once()

        assert response.status_code == 200
        providers = response.json()
        assert len(providers) >= 3  # Now we expect at least 3 providers with Fireworks

        # Check provider1 (groq)
        provider1 = next(p for p in providers if p["id"] == "groq")
        assert provider1["name"] == "Provider groq"
        assert provider1["enabled"] is True
        assert len(provider1["models"]) == 2
        assert provider1["models"][0]["name"] == "Model 1"
        assert provider1["models"][0]["id"] == "ft_model1"
        assert provider1["models"][0]["supports_function_calling"] is True
        assert provider1["models"][1]["name"] == "Model 2"
        assert provider1["models"][1]["id"] == "ft_model2"
        assert provider1["models"][1]["supports_function_calling"] is False

        # Check provider2 (together_ai)
        provider2 = next(p for p in providers if p["id"] == "together_ai")
        assert provider2["name"] == "Provider together_ai"
        assert provider2["enabled"] is False
        assert len(provider2["models"]) == 1
        assert provider2["models"][0]["name"] == "Model 1"
        assert provider2["models"][0]["id"] == "ft_model1_p2"

        # Check Fireworks provider
        fireworks_provider = next(p for p in providers if p["id"] == "fireworks_ai")
        assert (
            fireworks_provider["name"] == "Provider fireworks_ai"
        )  # Using mock_provider_name_from_id
        assert len(fireworks_provider["models"]) == 1
        assert fireworks_provider["models"][0]["name"] == "Fireworks Model"
        assert fireworks_provider["models"][0]["id"] == "fireworks/model-1"
        assert fireworks_provider["models"][0]["supports_function_calling"] is True


@pytest.fixture
def mock_finetune_registry():
    mock_adapter = Mock()
    mock_adapter.available_parameters.return_value = [
        FineTuneParameter(
            name="learning_rate",
            type="float",
            description="Learning rate for training",
            optional=True,
        ),
        FineTuneParameter(
            name="epochs",
            type="int",
            description="Number of training epochs",
            optional=False,
        ),
    ]

    mock_registry = {"test_provider": mock_adapter}

    with unittest.mock.patch(
        "app.desktop.studio_server.finetune_api.finetune_registry", mock_registry
    ):
        yield mock_registry


def test_get_finetune_hyperparameters(client, mock_finetune_registry):
    response = client.get("/api/finetune/hyperparameters/test_provider")

    assert response.status_code == 200
    parameters = response.json()
    assert len(parameters) == 2

    assert parameters[0]["name"] == "learning_rate"
    assert parameters[0]["type"] == "float"
    assert parameters[0]["description"] == "Learning rate for training"
    assert parameters[0]["optional"] is True

    assert parameters[1]["name"] == "epochs"
    assert parameters[1]["type"] == "int"
    assert parameters[1]["description"] == "Number of training epochs"
    assert parameters[1]["optional"] is False


def test_get_finetune_hyperparameters_invalid_provider(client, mock_finetune_registry):
    response = client.get("/api/finetune/hyperparameters/invalid_provider")

    assert response.status_code == 400
    assert (
        response.json()["message"] == "Fine tune provider 'invalid_provider' not found"
    )


def test_dataset_split_type_enum():
    assert DatasetSplitType.TRAIN_TEST.value == "train_test"
    assert DatasetSplitType.TRAIN_TEST_VAL.value == "train_test_val"
    assert DatasetSplitType.TRAIN_TEST_VAL_80.value == "train_test_val_80"
    assert DatasetSplitType.ALL.value == "all"


class ModelTester(BaseModel):
    dataset_id: DatasetFilterId


# Check these strings from UI exist
@pytest.mark.parametrize(
    "id,expect_error",
    [
        ("all", False),
        ("high_rating", False),
        ("thinking_model", False),
        ("thinking_model_high_rated", False),
        ("invalid", True),
    ],
)
def test_dataset_filter_ids(id, expect_error):
    if expect_error:
        with pytest.raises(ValueError):
            ModelTester(dataset_id=id)
    else:
        model = ModelTester(dataset_id=id)
        assert model.dataset_id == id


def test_api_split_types_mapping():
    from app.desktop.studio_server.finetune_api import api_split_types

    assert api_split_types[DatasetSplitType.TRAIN_TEST] == Train80Test20SplitDefinition
    assert (
        api_split_types[DatasetSplitType.TRAIN_TEST_VAL]
        == Train60Test20Val20SplitDefinition
    )
    assert (
        api_split_types[DatasetSplitType.TRAIN_TEST_VAL_80]
        == Train80Test10Val10SplitDefinition
    )
    assert api_split_types[DatasetSplitType.ALL] == AllSplitDefinition
    assert api_split_types[DatasetSplitType.TRAIN_VAL] == Train80Val20SplitDefinition
    for split_type in DatasetSplitType:
        assert split_type in api_split_types


@pytest.fixture
def mock_dataset_split():
    split = DatasetSplit(
        id="new_split",
        name="Test Split",
        split_contents={"train": ["1", "2"], "test": ["3"]},
        splits=AllSplitDefinition,
    )
    return split


def test_create_dataset_split(
    client, mock_task_from_id_disk_backed, mock_dataset_split
):
    # Mock DatasetSplit.from_task and save_to_file
    mock_from_task = unittest.mock.patch.object(
        DatasetSplit, "from_task", return_value=mock_dataset_split
    )
    mock_save = unittest.mock.patch.object(DatasetSplit, "save_to_file")

    with mock_from_task as from_task_mock, mock_save as save_mock:
        request_data = {
            "dataset_split_type": "train_test",
            "filter_id": "high_rating",
            "name": "Test Split",
            "description": "Test description",
        }

        response = client.post(
            "/api/projects/project1/tasks/task1/dataset_splits", json=request_data
        )

        assert response.status_code == 200
        result = response.json()
        assert result["id"] == "new_split"
        assert result["name"] == "Test Split"

        # Verify the mocks were called correctly
        mock_task_from_id_disk_backed.assert_called_once_with("project1", "task1")
        from_task_mock.assert_called_once()
        _, kwargs = from_task_mock.call_args
        assert kwargs["filter_id"] == "high_rating"
        save_mock.assert_called_once()


def test_create_dataset_split_auto_name(
    client, mock_task_from_id_disk_backed, mock_dataset_split
):
    # Mock DatasetSplit.from_task and save_to_file
    mock_from_task = unittest.mock.patch.object(
        DatasetSplit, "from_task", return_value=mock_dataset_split
    )
    mock_save = unittest.mock.patch.object(DatasetSplit, "save_to_file")

    with mock_from_task as from_task_mock, mock_save as save_mock:
        request_data = {"dataset_split_type": "train_test", "filter_id": "all"}

        response = client.post(
            "/api/projects/project1/tasks/task1/dataset_splits", json=request_data
        )

        assert response.status_code == 200

        # Verify auto-generated name format
        from_task_mock.assert_called_once()
        args = from_task_mock.call_args[0]
        name = args[0]
        assert len(name.split()) == 2  # 2 word memorable name
        assert len(name) > 5  # Not too short
        save_mock.assert_called_once()


def test_create_dataset_split_request_validation():
    # Test valid request
    request = CreateDatasetSplitRequest(
        dataset_split_type=DatasetSplitType.TRAIN_TEST,
        filter_id="all",
        name="Test Split",
        description="Test description",
    )
    assert request.dataset_split_type == DatasetSplitType.TRAIN_TEST
    assert request.filter_id == "all"
    assert request.name == "Test Split"
    assert request.description == "Test description"

    # Test optional fields
    request = CreateDatasetSplitRequest(
        dataset_split_type=DatasetSplitType.TRAIN_TEST,
        filter_id="all",
    )
    assert request.name is None
    assert request.description is None

    # Test invalid dataset split type
    with pytest.raises(ValueError):
        CreateDatasetSplitRequest(dataset_split_type="invalid_type", filter_id="all")

    # Test invalid filter type
    with pytest.raises(ValueError):
        CreateDatasetSplitRequest(
            dataset_split_type=DatasetSplitType.TRAIN_TEST, filter_id="invalid_type"
        )


@pytest.fixture
def mock_finetune_adapter():
    adapter = Mock()
    adapter.create_and_start = AsyncMock(
        return_value=(
            None,  # First return value is ignored in the API
            Finetune(
                id="new_ft",
                name="New Finetune",
                provider="test_provider",
                base_model_id="base_model_1",
                dataset_split_id="split1",
                system_message="Test system message",
                thinking_instructions=None,
            ),
        )
    )
    return adapter


@pytest.mark.parametrize(
    "data_strategy,custom_thinking_instructions,expected_thinking_instructions,expect_error",
    [
        (ChatStrategy.single_turn, None, None, False),
        (
            ChatStrategy.two_message_cot,
            None,
            "Think step by step, explaining your reasoning.",
            False,
        ),  # Our default
        (ChatStrategy.two_message_cot, "CTI", "CTI", False),
        (ChatStrategy.two_message_cot_legacy, "CTI", "CTI", True),
    ],
)
async def test_create_finetune(
    client,
    mock_task_from_id_disk_backed,
    test_task,
    mock_finetune_registry,
    mock_finetune_adapter,
    data_strategy,
    custom_thinking_instructions,
    expected_thinking_instructions,
    expect_error,
):
    mock_finetune_registry["test_provider"] = mock_finetune_adapter

    request_data = {
        "name": "New Finetune",
        "description": "Test description",
        "dataset_id": "split1",
        "train_split_name": "train",
        "validation_split_name": "validation",
        "parameters": {"learning_rate": 0.001, "epochs": 10},
        "provider": "test_provider",
        "base_model_id": "base_model_1",
        "custom_system_message": "Test system message",
        "custom_thinking_instructions": custom_thinking_instructions,
        "data_strategy": data_strategy.value,
    }

    response = client.post(
        "/api/projects/project1/tasks/task1/finetunes", json=request_data
    )

    if expect_error:
        assert response.status_code == 422
        return

    assert response.status_code == 200
    result = response.json()
    assert result["id"] == "new_ft"
    assert result["name"] == "New Finetune"
    assert result["provider"] == "test_provider"
    assert result["base_model_id"] == "base_model_1"

    split1 = next(split for split in test_task.dataset_splits() if split.id == "split1")

    # Verify the adapter was called correctly
    mock_finetune_adapter.create_and_start.assert_awaited_once_with(
        dataset=split1,
        provider_id="test_provider",
        provider_base_model_id="base_model_1",
        train_split_name="train",
        system_message="Test system message",
        thinking_instructions=expected_thinking_instructions,
        parameters={"learning_rate": 0.001, "epochs": 10},
        name="New Finetune",
        description="Test description",
        validation_split_name="validation",
        data_strategy=data_strategy,
        run_config=None,
    )


def _make_multiturn_task(tmp_path) -> Task:
    project = Project(name="Test Project", path=str(tmp_path / "project.kiln"))
    project.save_to_file()
    task = Task(
        name="Multi-turn Test Task",
        instruction="Test instruction",
        parent=project,
        turn_mode=TurnMode.multiturn,
    )
    task.save_to_file()
    return task


def test_create_finetune_multiturn_task_rejected(client, tmp_path, monkeypatch):
    # Build a fresh multi-turn task locally rather than mutating the shared
    # test_task fixture (turn_mode is frozen post-construction).
    multiturn_task = _make_multiturn_task(tmp_path)
    monkeypatch.setattr(
        "app.desktop.studio_server.finetune_api.task_from_id",
        Mock(return_value=multiturn_task),
    )

    request_data = {
        "dataset_id": "split1",
        "train_split_name": "train",
        "parameters": {},
        "provider": "openai",
        "base_model_id": "base_model_1",
        "custom_system_message": "Test system message",
        "data_strategy": "final_only",
    }

    response = client.post(
        "/api/projects/project1/tasks/task1/finetunes", json=request_data
    )

    assert response.status_code == 400
    assert (
        response.json()["message"]
        == "Fine-tuning is not supported for multi-turn tasks."
    )


def test_download_dataset_jsonl_multiturn_task_rejected(client, tmp_path, monkeypatch):
    multiturn_task = _make_multiturn_task(tmp_path)
    monkeypatch.setattr(
        "app.desktop.studio_server.finetune_api.task_from_id",
        Mock(return_value=multiturn_task),
    )

    response = client.get(
        "/api/download_dataset_jsonl",
        params={
            "project_id": "project1",
            "task_id": "task1",
            "dataset_id": "split1",
            "split_name": "train",
            "format_type": "openai_chat_jsonl",
            "data_strategy": "final_only",
        },
    )

    assert response.status_code == 400
    assert (
        response.json()["message"]
        == "Fine-tuning is not supported for multi-turn tasks."
    )


def test_create_finetune_invalid_provider(client, mock_task_from_id_disk_backed):
    request_data = {
        "dataset_id": "split1",
        "train_split_name": "train",
        "parameters": {},
        "provider": "invalid_provider",
        "base_model_id": "base_model_1",
        "custom_system_message": "Test system message",
        "data_strategy": "final_only",
    }

    response = client.post(
        "/api/projects/project1/tasks/task1/finetunes", json=request_data
    )

    assert response.status_code == 400
    assert (
        response.json()["message"] == "Fine tune provider 'invalid_provider' not found"
    )


def test_create_finetune_invalid_dataset(
    client,
    mock_task_from_id_disk_backed,
    mock_finetune_registry,
    mock_finetune_adapter,
):
    mock_finetune_registry["test_provider"] = mock_finetune_adapter

    request_data = {
        "dataset_id": "invalid_split_id",
        "train_split_name": "train",
        "parameters": {},
        "provider": "test_provider",
        "base_model_id": "base_model_1",
        "custom_system_message": "Test system message",
        "data_strategy": "final_only",
    }

    response = client.post(
        "/api/projects/project1/tasks/task1/finetunes", json=request_data
    )

    assert response.status_code == 404
    assert (
        response.json()["message"]
        == "Dataset split with ID 'invalid_split_id' not found"
    )


def test_create_finetune_request_validation():
    # Test valid request with all fields
    request = CreateFinetuneRequest(
        name="Test Finetune",
        description="Test description",
        dataset_id="split1",
        train_split_name="train",
        validation_split_name="validation",
        parameters={"param1": "value1"},
        provider="test_provider",
        base_model_id="base_model_1",
        custom_system_message="Test system message",
        data_strategy=ChatStrategy.single_turn,
    )
    assert request.name == "Test Finetune"
    assert request.description == "Test description"
    assert request.dataset_id == "split1"
    assert request.validation_split_name == "validation"

    # Test valid request with only required fields
    request = CreateFinetuneRequest(
        dataset_id="split1",
        train_split_name="train",
        parameters={},
        provider="test_provider",
        base_model_id="base_model_1",
        custom_system_message="Test system message",
        data_strategy=ChatStrategy.single_turn,
    )
    assert request.name is None
    assert request.description is None
    assert request.validation_split_name is None

    # Test invalid request (missing required field)
    with pytest.raises(ValueError):
        CreateFinetuneRequest(
            dataset_id="split1",  # Missing other required fields
        )


def test_create_finetune_no_system_message(
    client,
    mock_task_from_id_disk_backed,
    mock_finetune_registry,
    mock_finetune_adapter,
):
    mock_finetune_registry["test_provider"] = mock_finetune_adapter

    request_data = {
        "dataset_id": "split1",
        "train_split_name": "train",
        "parameters": {},
        "provider": "test_provider",
        "base_model_id": "base_model_1",
        "data_strategy": "final_only",
    }

    response = client.post(
        "/api/projects/project1/tasks/task1/finetunes", json=request_data
    )

    assert response.status_code == 400
    assert (
        response.json()["message"]
        == "System message generator or custom system message is required"
    )


def test_create_finetune_no_data_strategy(
    client,
    mock_task_from_id_disk_backed,
    mock_finetune_registry,
    mock_finetune_adapter,
):
    mock_finetune_registry["test_provider"] = mock_finetune_adapter

    request_data = {
        "dataset_id": "split1",
        "train_split_name": "train",
        "parameters": {},
        "provider": "test_provider",
        "base_model_id": "base_model_1",
        "custom_system_message": "Test system message",
    }

    response = client.post(
        "/api/projects/project1/tasks/task1/finetunes", json=request_data
    )

    assert response.status_code == 422


@pytest.fixture
def mock_prompt_builder():
    builder = Mock()
    builder.build_prompt.return_value = "Generated system message"

    with unittest.mock.patch(
        "app.desktop.studio_server.finetune_api.prompt_builder_from_id",
        return_value=builder,
    ) as mock:
        yield mock, builder


async def test_create_finetune_with_prompt_builder(
    client,
    mock_task_from_id_disk_backed,
    mock_finetune_registry,
    mock_finetune_adapter,
    mock_prompt_builder,
):
    mock_finetune_registry["test_provider"] = mock_finetune_adapter
    prompt_builder_mock, builder = mock_prompt_builder

    request_data = {
        "dataset_id": "split1",
        "train_split_name": "train",
        "parameters": {},
        "provider": "test_provider",
        "base_model_id": "base_model_1",
        "system_message_generator": "test_prompt_builder",
        "data_strategy": "final_only",
    }

    response = client.post(
        "/api/projects/project1/tasks/task1/finetunes", json=request_data
    )

    assert response.status_code == 200
    result = response.json()
    assert result["id"] == "new_ft"

    # Verify prompt builder was called correctly
    prompt_builder_mock.assert_called_once()
    builder.build_prompt.assert_called_once()

    # Verify the adapter was called with the generated system message
    mock_finetune_adapter.create_and_start.assert_awaited_once()
    call_kwargs = mock_finetune_adapter.create_and_start.await_args[1]
    assert call_kwargs["system_message"] == "Generated system message"


def test_create_finetune_prompt_builder_error(
    client,
    mock_task_from_id_disk_backed,
    mock_finetune_registry,
    mock_finetune_adapter,
    mock_prompt_builder,
):
    mock_finetune_registry["test_provider"] = mock_finetune_adapter
    _, builder = mock_prompt_builder

    # Make the prompt builder raise an error
    builder.build_prompt.side_effect = ValueError("Invalid prompt configuration")

    request_data = {
        "dataset_id": "split1",
        "train_split_name": "train",
        "parameters": {},
        "provider": "test_provider",
        "base_model_id": "base_model_1",
        "system_message_generator": "test_prompt_builder",
        "data_strategy": "final_only",
    }

    response = client.post(
        "/api/projects/project1/tasks/task1/finetunes", json=request_data
    )

    assert response.status_code == 400
    assert (
        response.json()["message"]
        == "Error generating system message using generator: test_prompt_builder. Source error: Invalid prompt configuration"
    )


@pytest.fixture
def mock_dataset_formatter():
    formatter = Mock()
    formatter.dump_to_file = AsyncMock(return_value=Path("path/to/dataset.jsonl"))

    with unittest.mock.patch(
        "app.desktop.studio_server.finetune_api.DatasetFormatter",
        return_value=formatter,
    ) as mock_class:
        yield mock_class, formatter


@pytest.mark.parametrize(
    "data_strategy",
    [ChatStrategy.single_turn, ChatStrategy.two_message_cot_legacy],
)
def test_download_dataset_jsonl(
    client,
    mock_task_from_id_disk_backed,
    mock_dataset_formatter,
    tmp_path,
    data_strategy,
):
    mock_formatter_class, mock_formatter = mock_dataset_formatter

    # Create a temporary file to simulate the dataset
    test_file = tmp_path / "dataset.jsonl"
    test_file.write_text('{"test": "data"}')
    mock_formatter.dump_to_file = AsyncMock(return_value=test_file)

    response = client.get(
        "/api/download_dataset_jsonl",
        params={
            "project_id": "project1",
            "task_id": "task1",
            "dataset_id": "split1",
            "split_name": "train",
            "format_type": "openai_chat_jsonl",
            "custom_system_message": "Test system message",
            "data_strategy": data_strategy.value,
        },
    )

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/jsonl"
    assert (
        response.headers["Content-Disposition"]
        == f'attachment; filename="{test_file.name}"'
    )
    assert response.content == b'{"test": "data"}'

    # Verify the formatter was created and used correctly
    mock_formatter_class.assert_called_once()
    mock_formatter.dump_to_file.assert_called_once_with(
        "train",
        DatasetFormat.OPENAI_CHAT_JSONL,
        data_strategy,
    )


@pytest.fixture
def valid_download_params():
    return {
        "project_id": "project1",
        "task_id": "task1",
        "dataset_id": "split1",
        "split_name": "train",
        "format_type": "openai_chat_jsonl",
        "custom_system_message": "Test system message",
        "data_strategy": "final_only",
    }


def test_download_dataset_jsonl_invalid_format(
    client, mock_task_from_id_disk_backed, valid_download_params
):
    valid_download_params["format_type"] = "invalid_format"
    response = client.get(
        "/api/download_dataset_jsonl",
        params=valid_download_params,
    )

    assert response.status_code == 400
    assert response.json()["message"] == "Dataset format 'invalid_format' not found"


def test_download_dataset_jsonl_data_strategy_invalid(
    client, mock_task_from_id_disk_backed, valid_download_params
):
    valid_download_params["data_strategy"] = "invalid_data_strategy"
    response = client.get(
        "/api/download_dataset_jsonl",
        params=valid_download_params,
    )

    assert response.status_code == 400
    assert (
        response.json()["message"] == "Data strategy 'invalid_data_strategy' not found"
    )


def test_download_dataset_jsonl_invalid_dataset(
    client, mock_task_from_id_disk_backed, valid_download_params
):
    valid_download_params["dataset_id"] = "invalid_split"
    response = client.get(
        "/api/download_dataset_jsonl",
        params=valid_download_params,
    )

    assert response.status_code == 404
    assert (
        response.json()["message"] == "Dataset split with ID 'invalid_split' not found"
    )


def test_download_dataset_jsonl_invalid_split(
    client, mock_task_from_id_disk_backed, valid_download_params
):
    valid_download_params["split_name"] = "invalid_split"
    response = client.get(
        "/api/download_dataset_jsonl",
        params=valid_download_params,
    )

    assert response.status_code == 404
    assert (
        response.json()["message"]
        == "Dataset split with name 'invalid_split' not found"
    )


def test_download_dataset_jsonl_rejects_mismatched_tool_dataset(
    client, mock_task_from_id_disk_backed, valid_download_params, empty_task
):
    create_synthetic_run(
        empty_task,
        run_id="run_with_skill_a",
        name="Run With Skill A",
        output_text="Test output with skill A",
        created_by="user1",
        tool_ids=["kiln_tool::skill::skill_a"],
    )
    create_synthetic_run(
        empty_task,
        run_id="run_with_skill_b",
        name="Run With Skill B",
        output_text="Test output with skill B",
        created_by="user2",
        tool_ids=["kiln_tool::skill::skill_b"],
    )
    create_dataset_split(
        empty_task,
        split_id="split_mismatch",
        name="Split Mismatch",
        run_ids=["run_with_skill_a", "run_with_skill_b"],
    )
    mock_task_from_id_disk_backed.return_value = empty_task

    valid_download_params["dataset_id"] = "split_mismatch"
    response = client.get(
        "/api/download_dataset_jsonl",
        params=valid_download_params,
    )

    assert response.status_code == 400
    assert (
        response.json()["message"]
        == "Dataset contains mixed tool/skill selections and cannot be exported"
    )


def test_download_dataset_jsonl_with_prompt_builder(
    client,
    mock_task_from_id_disk_backed,
    test_task,
    mock_dataset_formatter,
    mock_prompt_builder,
    tmp_path,
):
    mock_formatter_class, mock_formatter = mock_dataset_formatter
    prompt_builder_mock, builder = mock_prompt_builder

    # Create a temporary file to simulate the dataset
    test_file = tmp_path / "dataset.jsonl"
    test_file.write_text('{"test": "data"}')
    mock_formatter.dump_to_file = AsyncMock(return_value=test_file)

    response = client.get(
        "/api/download_dataset_jsonl",
        params={
            "project_id": "project1",
            "task_id": "task1",
            "dataset_id": "split1",
            "split_name": "train",
            "format_type": "openai_chat_jsonl",
            "system_message_generator": "test_prompt_builder",
            "custom_thinking_instructions": "custom thinking instructions",
            "data_strategy": "final_only",
        },
    )

    assert response.status_code == 200

    # Verify prompt builder was used
    prompt_builder_mock.assert_called_once_with("test_prompt_builder", test_task)
    builder.build_prompt.assert_called_once()

    # Verify formatter was created with generated system message
    mock_formatter_class.assert_called_once()
    call_kwargs = mock_formatter_class.call_args[1]
    assert call_kwargs["dataset"].id == "split1"
    assert call_kwargs["system_message"] == "Generated system message"
    assert call_kwargs["thinking_instructions"] is None


async def test_get_finetune(client, mock_task_from_id_disk_backed):
    response = client.get("/api/projects/project1/tasks/task1/finetunes/ft1")

    assert response.status_code == 200
    finetune = response.json()["finetune"]
    assert finetune["id"] == "ft1"
    assert finetune["name"] == "Finetune 1"
    assert finetune["provider"] == "openai"
    assert finetune["base_model_id"] == "model1"
    assert finetune["dataset_split_id"] == "split1"
    assert finetune["system_message"] == "System prompt 1"

    status = response.json()["status"]
    assert status["status"] == "unknown"
    assert (
        status["message"]
        == "Provider 'openai' is not available for fine-tuning. Status cannot be refreshed."
    )

    mock_task_from_id_disk_backed.assert_called_once_with("project1", "task1")


def test_get_finetune_not_found(client, mock_task_from_id_disk_backed):
    response = client.get("/api/projects/project1/tasks/task1/finetunes/nonexistent")

    assert response.status_code == 404
    assert response.json()["message"] == "Finetune with ID 'nonexistent' not found"

    mock_task_from_id_disk_backed.assert_called_once_with("project1", "task1")


async def test_get_finetunes_with_status_update(
    client,
    mock_task_from_id_disk_backed,
    test_task,
    mock_finetune_registry,
    monkeypatch,
):
    # Create a mock enum class
    class MockModelProviderName:
        def __class_getitem__(cls, key):
            return "test_provider"

    monkeypatch.setattr(
        "app.desktop.studio_server.finetune_api.ModelProviderName",
        MockModelProviderName,
    )

    # Create mock adapter with status method
    mock_adapter = Mock()
    mock_adapter.status = AsyncMock(
        return_value={"status": "running", "message": "Training..."}
    )
    mock_adapter_class = Mock(return_value=mock_adapter)
    mock_finetune_registry["test_provider"] = mock_adapter_class

    # Add latest_status to mock finetunes
    tune1 = next(ft for ft in test_task.finetunes() if ft.id == "ft1")
    tune2 = next(ft for ft in test_task.finetunes() if ft.id == "ft2")
    tune1.latest_status = "pending"  # Should be updated
    tune1.save_to_file()
    tune2.latest_status = "completed"  # Should be skipped
    tune2.save_to_file()

    mock_adapter_class.assert_not_called()
    mock_adapter.status.assert_not_called()

    response = client.get(
        "/api/projects/project1/tasks/task1/finetunes?update_status=true"
    )

    assert response.status_code == 200
    finetunes = response.json()
    assert len(finetunes) == 2

    # Verify that status was only checked for the pending finetune
    mock_adapter_class.assert_called_once_with(tune1)
    mock_adapter.status.assert_called_once()


def test_thinking_instructions_non_cot_strategy():
    """Test that non-COT strategies return None regardless of other parameters"""
    task = Mock(spec=Task)
    result = thinking_instructions_from_request(
        task=task,
        data_strategy=ChatStrategy.single_turn,
        custom_thinking_instructions="custom instructions",
    )
    assert result is None


@pytest.mark.parametrize(
    "data_strategy",
    [
        ChatStrategy.two_message_cot_legacy,
        ChatStrategy.two_message_cot,
    ],
)
def test_thinking_instructions_custom(data_strategy):
    """Test that custom instructions are returned when provided"""
    task = Mock(spec=Task)
    custom_instructions = "My custom thinking instructions"
    result = thinking_instructions_from_request(
        task=task,
        data_strategy=data_strategy,
        custom_thinking_instructions=custom_instructions,
    )
    assert result == custom_instructions


@pytest.mark.parametrize(
    "data_strategy",
    [
        ChatStrategy.two_message_cot_legacy,
        ChatStrategy.two_message_cot,
    ],
)
@patch("app.desktop.studio_server.finetune_api.chain_of_thought_prompt")
def test_thinking_instructions_default(mock_cot, data_strategy):
    """Test that default chain of thought prompt is used when no custom instructions"""
    task = Mock(spec=Task)
    mock_cot.return_value = "Default COT instructions"

    result = thinking_instructions_from_request(
        task=task,
        data_strategy=data_strategy,
        custom_thinking_instructions=None,
    )

    mock_cot.assert_called_once_with(task)
    assert result == "Default COT instructions"


async def test_update_finetune(client, mock_task_from_id_disk_backed, test_task):
    """Test updating a finetune's name and description"""
    # Get the original finetune to verify changes later
    original_finetune = next(ft for ft in test_task.finetunes() if ft.id == "ft1")
    original_name = original_finetune.name

    # Prepare update data
    update_data = {
        "name": "Updated Finetune Name",
        "description": "Updated finetune description",
    }

    # Send PATCH request
    response = client.patch(
        "/api/projects/project1/tasks/task1/finetunes/ft1", json=update_data
    )

    # Verify response
    assert response.status_code == 200
    updated_finetune = response.json()
    assert updated_finetune["id"] == "ft1"
    assert updated_finetune["name"] == "Updated Finetune Name"
    assert updated_finetune["description"] == "Updated finetune description"

    mock_task_from_id_disk_backed.assert_called_with("project1", "task1")

    # Verify save_to_file was called by checking if the finetune in the task was updated
    updated_task_finetune = next(ft for ft in test_task.finetunes() if ft.id == "ft1")
    assert updated_task_finetune.name == "Updated Finetune Name"
    assert updated_task_finetune.description == "Updated finetune description"
    assert updated_task_finetune.name != original_name


@pytest.fixture
def mock_httpx_client():
    with patch("httpx.AsyncClient") as mock_client:
        client_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = client_instance
        yield client_instance


@pytest.fixture
def mock_config():
    with patch("app.desktop.studio_server.finetune_api.Config") as mock_config:
        config_instance = Mock()
        mock_config.shared.return_value = config_instance
        yield config_instance


@pytest.mark.asyncio
async def test_fetch_fireworks_finetune_models_no_api_key(mock_config):
    """Test that an empty list is returned when no API key is available"""
    mock_config.fireworks_api_key = None

    result = await fetch_fireworks_finetune_models()

    assert result == []
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_fetch_fireworks_finetune_models_success(mock_config, mock_httpx_client):
    """Test successful fetching of tunable models from Fireworks API"""
    mock_config.fireworks_api_key = "test-api-key"

    # Setup mock response for first page with next page token
    first_response = Mock()
    first_response.json.return_value = {
        "models": [
            {
                "name": "accounts/fireworks/models/qwen3-8b",
                "displayName": "Qwen3 8B",
                "supervisedLoraTunable": True,
                "supportsTools": True,
            },
            {
                "name": "accounts/fireworks/models/unsupported-model",
                "displayName": "Unsupported",
                "supervisedLoraTunable": True,  # supervised-tunable but not in allowlist
                "supportsTools": False,
            },
            {
                "name": "accounts/fireworks/models/llama-v3p3-70b-instruct",
                "displayName": "Llama 3.3 70B",
                "supervisedLoraTunable": False,  # not tunable, should be skipped
                "supportsTools": False,
            },
        ],
        "nextPageToken": "next-page-token",
    }

    # Setup mock response for second page with no next page token
    second_response = Mock()
    second_response.json.return_value = {
        "models": [
            {
                "name": "accounts/fireworks/models/gemma-4-26b-a4b-it",
                "displayName": "",  # Empty display name
                "supervisedFullParameterTunable": True,
                "supportsTools": False,
            },
            {
                "name": "accounts/fireworks/models/qwen3-32b",
                "displayName": "Qwen3 32B",
                "supervisedLoraTunable": True,
                "supervisedFullParameterTunable": True,
                "supportsTools": True,
            },
        ]
    }

    # Set up the client to return the responses in sequence
    mock_httpx_client.get.side_effect = [first_response, second_response]

    result = await fetch_fireworks_finetune_models()

    # Verify the API was called with the correct parameters
    assert mock_httpx_client.get.call_count == 2

    # First call should use initial parameters
    first_call_args = mock_httpx_client.get.call_args_list[0]
    assert (
        first_call_args[0][0] == "https://api.fireworks.ai/v1/accounts/fireworks/models"
    )
    assert first_call_args[1]["params"] == {"pageSize": 200}
    assert first_call_args[1]["headers"] == {"Authorization": "Bearer test-api-key"}

    # Second call should include the page token
    second_call_args = mock_httpx_client.get.call_args_list[1]
    assert (
        second_call_args[0][0]
        == "https://api.fireworks.ai/v1/accounts/fireworks/models"
    )
    assert second_call_args[1]["params"] == {
        "pageSize": 200,
        "pageToken": "next-page-token",
    }

    # 3 models: qwen3-8b (supported+supervised), gemma-4-26b-a4b-it (supported+supervised), qwen3-32b (supported+supervised)
    # Excluded: unsupported-model (not in allowlist), llama-v3p3-70b-instruct (not supervised-tunable)
    assert len(result) == 3

    assert result[0].name == "Qwen3 8B (qwen3-8b)"
    assert result[0].id == "accounts/fireworks/models/qwen3-8b"
    assert result[0].supports_function_calling is True

    # unsupported-model is supervised-tunable but not in allowlist
    assert all(
        model.id != "accounts/fireworks/models/unsupported-model" for model in result
    )

    # Empty display name falls back to the id tail
    gemma = next(
        model
        for model in result
        if model.id == "accounts/fireworks/models/gemma-4-26b-a4b-it"
    )
    assert gemma.name == "gemma-4-26b-a4b-it"
    assert gemma.supports_function_calling is False

    qwen32b = next(
        model for model in result if model.id == "accounts/fireworks/models/qwen3-32b"
    )
    assert qwen32b.supports_function_calling is True


@pytest.mark.asyncio
async def test_fetch_fireworks_finetune_models_empty_response(
    mock_config, mock_httpx_client
):
    """Test handling of empty model list from API"""
    mock_config.fireworks_api_key = "test-api-key"

    # Setup mock response with empty models list
    response = Mock()
    response.json.return_value = {"models": []}

    mock_httpx_client.get.return_value = response

    result = await fetch_fireworks_finetune_models()

    assert result == []
    mock_httpx_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_fireworks_finetune_models_invalid_response(
    mock_config, mock_httpx_client
):
    """Test handling of invalid response format from API"""
    mock_config.fireworks_api_key = "test-api-key"

    # Setup mock response with missing models key
    response = Mock()
    response.json.return_value = {"not_models": []}
    response.status_code = 200
    response.text = '{"not_models": []}'

    mock_httpx_client.get.return_value = response

    # Function should raise ValueError for invalid response
    with pytest.raises(ValueError) as excinfo:
        await fetch_fireworks_finetune_models()

    assert "Invalid response from Fireworks" in str(excinfo.value)
    assert "[200]" in str(excinfo.value)  # Should include status code
    mock_httpx_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_fireworks_finetune_models_http_error(
    mock_config, mock_httpx_client
):
    """Test handling of HTTP error from API"""
    mock_config.fireworks_api_key = "test-api-key"

    # Make the get request raise an exception
    mock_httpx_client.get.side_effect = httpx.HTTPError("Connection error")

    # Should propagate the error
    with pytest.raises(httpx.HTTPError):
        await fetch_fireworks_finetune_models()

    mock_httpx_client.get.assert_called_once()


@pytest.fixture
def mock_available_models():
    return [
        KilnModel(
            family=ModelFamily.gpt,
            name=ModelName.gpt_4_1,
            friendly_name="GPT 4.1",
            providers=[
                KilnModelProvider(
                    name=ModelProviderName.openai,
                    model_id="gpt-4.1",
                ),
                KilnModelProvider(
                    name=ModelProviderName.openrouter,
                    model_id="openai/gpt-4.1",
                ),
                KilnModelProvider(
                    name=ModelProviderName.azure_openai,
                    model_id="gpt-4.1",
                ),
                KilnModelProvider(
                    name=ModelProviderName.together_ai,
                    model_id="gpt-4.1-together",
                    provider_finetune_id="gpt-4.1-2025-04-14",
                ),
            ],
        ),
        KilnModel(
            family=ModelFamily.gpt,
            name=ModelName.gpt_4_1_mini,
            friendly_name="GPT 4.1 Mini",
            providers=[
                KilnModelProvider(
                    name=ModelProviderName.openai,
                    model_id="gpt-4.1-mini",
                ),
                KilnModelProvider(
                    name=ModelProviderName.openrouter,
                    model_id="openai/gpt-4.1-mini",
                ),
                KilnModelProvider(
                    name=ModelProviderName.azure_openai,
                    model_id="gpt-4.1-mini",
                ),
                KilnModelProvider(
                    name=ModelProviderName.together_ai,
                    model_id="gpt-4.1-mini-together",
                    provider_finetune_id="gpt-4.1-mini-2025-04-14",
                ),
            ],
        ),
        KilnModel(
            family=ModelFamily.qwen,
            name=ModelName.qwq_32b,
            friendly_name="QwQ 32B",
            providers=[
                KilnModelProvider(
                    name=ModelProviderName.huggingface,
                    model_id="qwen/qwq-32b",
                    provider_finetune_id="qwq-32b-xxx",
                    parser=ModelParserID.r1_thinking,
                )
            ],
        ),
    ]


@pytest.mark.parametrize(
    "model_id, provider, expected_data_strategies",
    [
        # for models where we have built-in models, we can infer the data strategies from the object itself
        (
            # does not have a parser, so should be defaults
            "gpt-4.1-2025-04-14",
            "together_ai",
            [
                ChatStrategy.single_turn,
                ChatStrategy.two_message_cot,
            ],
        ),
        (
            # does not have a parser, so should be defaults
            "gpt-4.1-mini-2025-04-14",
            "together_ai",
            [
                ChatStrategy.single_turn,
                ChatStrategy.two_message_cot,
            ],
        ),
        (
            # this model is not in any list, so should be defaults
            "fake-model-id",
            "fake-provider",
            [
                ChatStrategy.single_turn,
                ChatStrategy.two_message_cot,
            ],
        ),
        # this model has an R1 parser, should be r1 compatible
        (
            "qwq-32b-xxx",
            "huggingface",
            [
                ChatStrategy.single_turn_r1_thinking,
            ],
        ),
        # for fireworks_ai models, we infer the data strategies from the model name
        (
            # does not contain r1 or qwq in the id so it should be defaults
            "some-model-id",
            "fireworks_ai",
            [
                ChatStrategy.single_turn,
                ChatStrategy.two_message_cot,
            ],
        ),
        (
            # contains r1 in the id so it should be r1 compatible
            "some-model-with-r1-in-id",
            "fireworks_ai",
            [
                ChatStrategy.single_turn_r1_thinking,
            ],
        ),
        (
            # contains qwq in the id so it should be r1 compatible
            "some-model-with-qwq-in-id",
            "fireworks_ai",
            [
                ChatStrategy.single_turn_r1_thinking,
            ],
        ),
    ],
)
def test_infer_data_strategies(
    mock_available_models,
    model_id: str,
    provider: str,
    expected_data_strategies: list[ChatStrategy],
):
    assert (
        infer_data_strategies_for_model(mock_available_models, model_id, provider)
        == expected_data_strategies
    )


@pytest.mark.parametrize(
    "model_id, expected_data_strategies",
    [
        # R1 style models
        ("qwq-32b", [ChatStrategy.single_turn_r1_thinking]),
        (
            "deepseek-r1-distill-qwen-32b",
            [ChatStrategy.single_turn_r1_thinking],
        ),
        (
            "deepseek-r1",
            [ChatStrategy.single_turn_r1_thinking],
        ),
        (
            "deepseek-r1-basic",
            [ChatStrategy.single_turn_r1_thinking],
        ),
        (
            "deepseek-r1-distill-llama-70b",
            [ChatStrategy.single_turn_r1_thinking],
        ),
        (
            "deepseek-r1-distill-llama-8b",
            [ChatStrategy.single_turn_r1_thinking],
        ),
        (
            "deepseek-r1-distill-qwen-14b",
            [ChatStrategy.single_turn_r1_thinking],
        ),
        (
            "deepseek-r1-distill-qwen-1p5b",
            [ChatStrategy.single_turn_r1_thinking],
        ),
        (
            "deepseek-r1-distill-qwen-32b",
            [ChatStrategy.single_turn_r1_thinking],
        ),
        (
            "deepseek-r1-distill-qwen-7b",
            [ChatStrategy.single_turn_r1_thinking],
        ),
        # non-R1 style models
        (
            "deepseek-v3",
            [
                ChatStrategy.single_turn,
                ChatStrategy.two_message_cot,
            ],
        ),
        (
            "deepseek-v3-0324",
            [
                ChatStrategy.single_turn,
                ChatStrategy.two_message_cot,
            ],
        ),
        # optional R1 style
        (
            "qwen3-30b-a3b",
            [
                ChatStrategy.single_turn,
                ChatStrategy.single_turn_r1_thinking,
            ],
        ),
    ],
)
def test_data_strategies_from_finetune_id(
    model_id: str, expected_data_strategies: list[ChatStrategy]
):
    assert data_strategies_from_finetune_id(model_id) == expected_data_strategies


def test_finetune_dataset_info(client, mock_task_from_id_disk_backed, test_task):
    """Test the finetune_dataset_info endpoint returns correct data"""
    response = client.get("/api/projects/project1/tasks/task1/finetune_dataset_info")

    assert response.status_code == 200
    data = response.json()

    # Verify datasets
    assert len(data["existing_datasets"]) == 2
    dataset_ids = {ds["id"] for ds in data["existing_datasets"]}
    assert dataset_ids == {"split1", "split2"}

    # Verify finetunes
    assert len(data["existing_finetunes"]) == 2
    finetune_ids = {ft["id"] for ft in data["existing_finetunes"]}
    assert finetune_ids == {"ft1", "ft2"}

    # Verify fine_tune tags
    assert len(data["finetune_tags"]) == 2

    tag1 = next(x for x in data["finetune_tags"] if x["tag"] == "fine_tune_1")
    assert tag1["count"] == 2
    assert tag1["reasoning_count"] == 1
    assert tag1["high_quality_count"] == 1
    assert tag1["reasoning_and_high_quality_count"] == 1

    tag2 = next(x for x in data["finetune_tags"] if x["tag"] == "fine_tune_2")
    assert tag2["count"] == 1
    assert tag2["reasoning_count"] == 0
    assert tag2["high_quality_count"] == 0
    assert tag2["reasoning_and_high_quality_count"] == 0

    # Verify eligible_datasets (without tool filter, should be same as existing_datasets)
    assert "eligible_datasets" in data
    assert len(data["eligible_datasets"]) == 2
    eligible_dataset_ids = {ds["id"] for ds in data["eligible_datasets"]}
    assert eligible_dataset_ids == {"split1", "split2"}

    # Verify eligible_finetune_tags (without tool filter, should be same as finetune_tags)
    assert "eligible_finetune_tags" in data
    assert len(data["eligible_finetune_tags"]) == 2

    # Verify task_from_id was called correctly
    mock_task_from_id_disk_backed.assert_called_once_with("project1", "task1")


def test_finetune_dataset_info_no_tags(
    client, mock_task_from_id_disk_backed, test_task
):
    """Test finetune_dataset_info when there are no fine_tune tags"""
    # Remove all runs from the task
    for run in test_task.runs(include_intermediate_runs=True):
        run.delete()

    response = client.get("/api/projects/project1/tasks/task1/finetune_dataset_info")

    assert response.status_code == 200
    data = response.json()

    # Verify datasets and finetunes still exist
    assert len(data["existing_datasets"]) == 2
    assert len(data["existing_finetunes"]) == 2

    # Verify no fine_tune tags
    assert len(data["finetune_tags"]) == 0

    # Verify eligible properties still present
    assert "eligible_datasets" in data
    assert len(data["eligible_datasets"]) == 2
    assert "eligible_finetune_tags" in data
    assert len(data["eligible_finetune_tags"]) == 0


def test_finetune_dataset_info_excludes_orphan_datasets(
    client, mock_task_from_id_disk_backed, test_task
):
    """Test that finetune_dataset_info excludes orphan datasets (datasets not associated with any finetune)"""
    orphan_split = DatasetSplit(
        id="orphan_split",
        name="Orphan Split",
        split_contents={"train": ["4", "5"]},
        splits=AllSplitDefinition,
    )
    orphan_split.parent = test_task
    orphan_split.save_to_file()

    response = client.get("/api/projects/project1/tasks/task1/finetune_dataset_info")

    assert response.status_code == 200
    data = response.json()

    assert len(data["existing_datasets"]) == 2
    dataset_ids = {ds["id"] for ds in data["existing_datasets"]}
    assert dataset_ids == {"split1", "split2"}
    assert "orphan_split" not in dataset_ids

    assert len(data["existing_finetunes"]) == 2

    assert len(data["eligible_datasets"]) == 2
    eligible_dataset_ids = {ds["id"] for ds in data["eligible_datasets"]}
    assert eligible_dataset_ids == {"split1", "split2"}
    assert "orphan_split" not in eligible_dataset_ids


def test_finetune_dataset_info_no_datasets_or_finetunes(
    client, mock_task_from_id_disk_backed, tmp_path
):
    """Test finetune_dataset_info when there are no datasets or finetunes"""
    # Create a task with no datasets or finetunes
    project_path = tmp_path / "project.kiln"
    project = Project(name="Test Project", path=str(project_path))
    project.save_to_file()

    task = Task(
        name="Test Task",
        instruction="This is a test instruction",
        description="This is a test task",
        parent=project,
    )
    task.save_to_file()

    mock_task_from_id_disk_backed.return_value = task

    response = client.get("/api/projects/project1/tasks/task1/finetune_dataset_info")

    assert response.status_code == 200
    data = response.json()

    # Verify empty lists
    assert len(data["existing_datasets"]) == 0
    assert len(data["existing_finetunes"]) == 0
    assert len(data["finetune_tags"]) == 0

    # Verify empty eligible properties
    assert "eligible_datasets" in data
    assert len(data["eligible_datasets"]) == 0
    assert "eligible_finetune_tags" in data
    assert len(data["eligible_finetune_tags"]) == 0

    mock_task_from_id_disk_backed.assert_called_once_with("project1", "task1")


@pytest.mark.parametrize(
    "tool_filter, expected_count",
    [
        (None, 3),
        ([], 1),
        (["mcp::remote::server_a::tool_a"], 1),
        (["mcp::remote::server_a::tool_a", "mcp::remote::server_b::tool_b"], 1),
        (["mcp::remote::server_x::tool_x"], 0),
    ],
)
def test_compute_finetune_tag_info(empty_task, tool_filter, expected_count):
    create_synthetic_run(
        empty_task,
        run_id="run_with_tool_a",
        name="Run with tool A",
        output_text="Test output A",
        created_by="user1",
        tool_ids=["mcp::remote::server_a::tool_a"],
    )
    create_synthetic_run(
        empty_task,
        run_id="run_with_tool_a_b",
        name="Run with tool A and B",
        output_text="Test output AB",
        created_by="user2",
        tool_ids=[
            "mcp::remote::server_a::tool_a",
            "mcp::remote::server_b::tool_b",
        ],
    )
    create_synthetic_run(
        empty_task,
        run_id="run_no_tools",
        name="Run without tools",
        output_text="Test output no tools",
        created_by="user3",
    )

    result = compute_finetune_tag_info(empty_task, tool_filter=tool_filter)

    if expected_count == 0:
        assert len(result) == 0
    else:
        assert len(result) == 1
        assert result[0].tag == "fine_tune_tools"
        assert result[0].count == expected_count


@pytest.mark.parametrize(
    "tool_filter, expected_tag",
    [
        ([], "fine_tune_without_skill"),
        (["kiln_tool::skill::skill_a"], "fine_tune_with_skill"),
    ],
)
def test_compute_finetune_tag_info_with_skill_filters(
    empty_task, tool_filter, expected_tag
):
    skill_a = "kiln_tool::skill::skill_a"
    create_synthetic_run(
        empty_task,
        run_id="run_with_skill",
        name="Run With Skill",
        output_text="Test output with skill",
        created_by="user1",
        tags=["fine_tune_with_skill"],
        tool_ids=[skill_a],
    )
    create_synthetic_run(
        empty_task,
        run_id="run_without_skill",
        name="Run Without Skill",
        output_text="Test output without skill",
        created_by="user2",
        tags=["fine_tune_without_skill"],
    )

    result = compute_finetune_tag_info(empty_task, tool_filter=tool_filter)

    assert len(result) == 1
    assert result[0].tag == expected_tag
    assert result[0].count == 1


def test_finetune_dataset_info_filters_datasets_by_selected_tool(
    client,
    mock_task_from_id_disk_backed,
    empty_task,
):
    create_synthetic_run(
        empty_task,
        run_id="run_with_tool",
        name="Run With Tool",
        output_text="Test output with tool",
        created_by="user1",
        tool_ids=["mcp::remote::server_a::tool_a"],
    )
    create_synthetic_run(
        empty_task,
        run_id="run_without_tool",
        name="Run Without Tool",
        output_text="Test output without tool",
        created_by="user2",
    )

    create_dataset_split(
        empty_task,
        split_id="split_with_tool",
        name="Split With Tool",
        run_ids=["run_with_tool"],
    )
    create_dataset_split(
        empty_task,
        split_id="split_without_tool",
        name="Split Without Tool",
        run_ids=["run_without_tool"],
    )

    create_finetune(
        empty_task,
        finetune_id="ft_with_tool",
        name="ft_with_tool",
        split_id="split_with_tool",
    )
    create_finetune(
        empty_task,
        finetune_id="ft_without_tool",
        name="ft_without_tool",
        split_id="split_without_tool",
    )

    mock_task_from_id_disk_backed.return_value = empty_task

    response = client.get(
        "/api/projects/project1/tasks/task1/finetune_dataset_info",
        params={"tool_ids": ["mcp::remote::server_a::tool_a"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert {ds["id"] for ds in data["eligible_datasets"]} == {"split_with_tool"}
    assert len(data["eligible_finetune_tags"]) == 1
    assert data["eligible_finetune_tags"][0]["tag"] == "fine_tune_tools"
    assert data["eligible_finetune_tags"][0]["count"] == 1


def test_finetune_dataset_info_empty_tool_filter_excludes_mismatched_datasets(
    client,
    mock_task_from_id_disk_backed,
    empty_task,
):
    for run_id, skill_id in [
        ("run_with_skill_a", "kiln_tool::skill::skill_a"),
        ("run_with_skill_b", "kiln_tool::skill::skill_b"),
    ]:
        create_synthetic_run(
            empty_task,
            run_id=run_id,
            name=run_id,
            output_text=f"Test output {run_id}",
            created_by="user1",
            tool_ids=[skill_id],
        )

    create_dataset_split(
        empty_task,
        split_id="split_mismatch",
        name="Split Mismatch",
        run_ids=["run_with_skill_a", "run_with_skill_b"],
    )
    create_finetune(
        empty_task,
        finetune_id="ft_mismatch",
        name="Finetune Mismatch",
        split_id="split_mismatch",
    )

    mock_task_from_id_disk_backed.return_value = empty_task

    # Mixed-skill datasets should not appear reusable when the fine-tune UI is
    # filtering for no tools and no skills.
    response = client.get(
        "/api/projects/project1/tasks/task1/finetune_dataset_info",
        params={"empty_tool_filter": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["eligible_datasets"] == []


def test_system_message_from_request_with_skills(tmp_path):
    project = Project(name="Test Project", path=str(tmp_path / "project.kiln"))
    project.save_to_file()
    task = Task(
        name="Test Task",
        instruction="Do the thing",
        parent=project,
    )
    task.save_to_file()

    skills = [Skill(name="my-skill", description="Helps with things")]

    result = system_message_from_request(
        task=task,
        custom_system_message=None,
        system_message_generator="simple_prompt_builder",
        skills=skills,
    )

    assert "Do the thing" in result
    assert "# Skills" in result
    assert "my-skill" in result
    assert "Helps with things" in result


def test_system_message_from_request_without_skills(tmp_path):
    project = Project(name="Test Project", path=str(tmp_path / "project.kiln"))
    project.save_to_file()
    task = Task(
        name="Test Task",
        instruction="Do the thing",
        parent=project,
    )
    task.save_to_file()

    result = system_message_from_request(
        task=task,
        custom_system_message=None,
        system_message_generator="simple_prompt_builder",
    )

    assert "Do the thing" in result
    assert "## Skills" not in result


def test_system_message_custom_ignores_skills(tmp_path):
    """Custom system messages are used as-is; skills param is irrelevant."""
    project = Project(name="Test Project", path=str(tmp_path / "project.kiln"))
    project.save_to_file()
    task = Task(name="T", instruction="X", parent=project)
    task.save_to_file()

    skills = [Skill(name="my-skill", description="Desc")]

    result = system_message_from_request(
        task=task,
        custom_system_message="My custom message",
        system_message_generator=None,
        skills=skills,
    )

    assert result == "My custom message"
    assert "# Skills" not in result
