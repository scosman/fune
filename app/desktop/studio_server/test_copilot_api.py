import json
from http import HTTPStatus
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kiln_ai.datamodel import Project, Task, TaskRun
from kiln_ai.datamodel.datamodel_enums import (
    EvalStatus,
    Priority,
    TaskOutputRatingType,
)
from kiln_ai.datamodel.eval import (
    Eval,
    EvalConfigType,
    EvalDataType,
    EvalInputSplit,
    LlmJudgeProperties,
    TaskRunSplit,
)
from kiln_ai.datamodel.run_config import ToolsRunConfig
from kiln_ai.datamodel.spec_properties import SpecType
from kiln_ai.datamodel.task import TaskRunConfig
from kiln_ai.datamodel.task_output import DataSource, DataSourceType, TaskOutput
from kiln_server.custom_errors import connect_custom_errors

from app.desktop.studio_server.api_client.kiln_ai_server_client.models.clarify_spec_output import (
    ClarifySpecOutput,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models.data_guide_job_output import (
    DataGuideJobOutput,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models.data_guide_job_result_response import (
    DataGuideJobResultResponse,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models.generate_batch_output import (
    GenerateBatchOutput,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models.job_start_response import (
    JobStartResponse,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models.job_status import (
    JobStatus,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models.job_status_response import (
    JobStatusResponse,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models.job_type import (
    JobType,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models.question_set import (
    QuestionSet as QuestionSetServerApi,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models.refine_spec_api_output import (
    RefineSpecApiOutput,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models.refine_spec_from_answers_and_name_output import (
    RefineSpecFromAnswersAndNameOutput,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.types import (
    Response as SdkResponse,
)
from app.desktop.studio_server.api_models.copilot_models import (
    SampleApi,
    TaskSkillInfoApi,
    TaskToolInfoApi,
)
from app.desktop.studio_server.copilot_api import connect_copilot_api
from app.desktop.studio_server.utils.copilot_utils import SingleTurnDataset


@pytest.fixture
def app():
    app = FastAPI()
    connect_custom_errors(app)
    connect_copilot_api(app)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def mock_api_key():
    with patch(
        "app.desktop.studio_server.utils.copilot_utils.Config.shared"
    ) as mock_config_shared:
        mock_config = mock_config_shared.return_value
        mock_config.kiln_copilot_api_key = "test_api_key"
        yield mock_config


@pytest.fixture
def clarify_spec_input():
    return {
        "target_task_info": {
            "task_prompt": "Test task prompt",
            "task_input_schema": '{"type": "string"}',
            "task_output_schema": '{"type": "string"}',
        },
        "target_specification": "Test template",
        "num_samples_per_topic": 5,
        "num_topics": 3,
        "providers": ["openai"],
        "num_exemplars": 10,
    }


@pytest.fixture
def refine_spec_input():
    return {
        "target_task_info": {
            "task_prompt": "Test task prompt",
            "task_input_schema": '{"type": "string"}',
            "task_output_schema": '{"type": "string"}',
        },
        "target_specification": {
            "spec_fields": {},
            "spec_field_current_values": {},
        },
        "examples_with_feedback": [
            {
                "user_agrees_with_judge": True,
                "input": "test input",
                "output": "test output",
                "fails_specification": False,
            }
        ],
    }


@pytest.fixture
def submit_answers_input():
    return {
        "task_prompt": "Test task prompt",
        "specification": {
            "spec_fields": {"tone": "The desired tone"},
            "spec_field_current_values": {"tone": "friendly"},
        },
        "questions_and_answers": [
            {
                "question_title": "How formal?",
                "question_body": "Should the tone be formal or casual?",
                "answer_options": [
                    {
                        "answer_title": "Formal",
                        "answer_description": "Use a formal tone",
                        "selected": True,
                    },
                    {
                        "answer_title": "Casual",
                        "answer_description": "Use a casual tone",
                        "selected": False,
                    },
                ],
                "custom_answer": None,
            }
        ],
    }


@pytest.fixture
def generate_batch_input():
    return {
        "target_task_info": {
            "task_prompt": "Test task prompt",
            "task_input_schema": '{"type": "string"}',
            "task_output_schema": '{"type": "string"}',
        },
        "sdg_session_config": {
            "topic_generation_config": {
                "task_metadata": {
                    "model_name": "gpt-4",
                    "model_provider_name": "openai",
                },
                "prompt": "Test topic generation prompt",
            },
            "input_generation_config": {
                "task_metadata": {
                    "model_name": "gpt-4",
                    "model_provider_name": "openai",
                },
                "prompt": "Test input generation prompt",
            },
            "output_generation_config": {
                "task_metadata": {
                    "model_name": "gpt-4",
                    "model_provider_name": "openai",
                },
                "prompt": "Test output generation prompt",
            },
        },
        "target_specification": "Test template",
        "num_samples_per_topic": 5,
        "num_topics": 3,
    }


class TestClarifySpec:
    def test_clarify_spec_no_api_key(self, client, clarify_spec_input):
        with patch(
            "app.desktop.studio_server.utils.copilot_utils.Config.shared"
        ) as mock_config_shared:
            mock_config = mock_config_shared.return_value
            mock_config.kiln_copilot_api_key = None

            response = client.post("/api/copilot/clarify_spec", json=clarify_spec_input)
            assert response.status_code == 401
            assert "API key not configured" in response.json()["message"]

    def test_clarify_spec_success(self, client, clarify_spec_input, mock_api_key):
        mock_output = MagicMock(spec=ClarifySpecOutput)
        mock_output.to_dict.return_value = {
            "examples_for_feedback": [
                {
                    "input": "test input",
                    "output": "test output",
                    "fails_specification": False,
                }
            ],
            "judge_result": {
                "task_metadata": {
                    "model_name": "gpt-4",
                    "model_provider_name": "openai",
                },
                "prompt": "Test judge prompt",
            },
            "sdg_session_config": {
                "topic_generation_config": {
                    "task_metadata": {
                        "model_name": "gpt-4",
                        "model_provider_name": "openai",
                    },
                    "prompt": "Test topic generation prompt",
                },
                "input_generation_config": {
                    "task_metadata": {
                        "model_name": "gpt-4",
                        "model_provider_name": "openai",
                    },
                    "prompt": "Test input generation prompt",
                },
                "output_generation_config": {
                    "task_metadata": {
                        "model_name": "gpt-4",
                        "model_provider_name": "openai",
                    },
                    "prompt": "Test output generation prompt",
                },
            },
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = mock_output

        with patch(
            "app.desktop.studio_server.copilot_api.clarify_spec_v1_copilot_clarify_spec_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post("/api/copilot/clarify_spec", json=clarify_spec_input)
            assert response.status_code == 200
            result = response.json()
            assert "examples_for_feedback" in result
            assert result["judge_result"]["task_metadata"]["model_name"] == "gpt-4"

    def test_clarify_spec_no_response(self, client, clarify_spec_input, mock_api_key):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = None

        with patch(
            "app.desktop.studio_server.copilot_api.clarify_spec_v1_copilot_clarify_spec_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post("/api/copilot/clarify_spec", json=clarify_spec_input)
            assert response.status_code == 500
            assert "Failed to analyze spec" in response.json()["message"]

    def test_clarify_spec_validation_error(
        self, client, clarify_spec_input, mock_api_key
    ):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.content = b'{"message": "Validation error from server"}'

        with patch(
            "app.desktop.studio_server.copilot_api.clarify_spec_v1_copilot_clarify_spec_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post("/api/copilot/clarify_spec", json=clarify_spec_input)
            assert response.status_code == 422
            assert "Validation error from server" in response.json()["message"]


class TestRefineSpec:
    def test_refine_spec_no_api_key(self, client, refine_spec_input):
        with patch(
            "app.desktop.studio_server.utils.copilot_utils.Config.shared"
        ) as mock_config_shared:
            mock_config = mock_config_shared.return_value
            mock_config.kiln_copilot_api_key = None

            response = client.post("/api/copilot/refine_spec", json=refine_spec_input)
            assert response.status_code == 401
            assert "API key not configured" in response.json()["message"]

    def test_refine_spec_success(self, client, refine_spec_input, mock_api_key):
        mock_output = MagicMock(spec=RefineSpecApiOutput)
        mock_output.to_dict.return_value = {
            "new_proposed_spec_edits": [],
            "not_incorporated_feedback": None,
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = mock_output

        with patch(
            "app.desktop.studio_server.copilot_api.refine_spec_v1_copilot_refine_spec_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post("/api/copilot/refine_spec", json=refine_spec_input)
            assert response.status_code == 200
            result = response.json()
            assert "new_proposed_spec_edits" in result
            assert "not_incorporated_feedback" in result

    def test_refine_spec_no_response(self, client, refine_spec_input, mock_api_key):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = None

        with patch(
            "app.desktop.studio_server.copilot_api.refine_spec_v1_copilot_refine_spec_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post("/api/copilot/refine_spec", json=refine_spec_input)
            assert response.status_code == 500
            assert "Failed to refine spec" in response.json()["message"]

    def test_refine_spec_validation_error(
        self, client, refine_spec_input, mock_api_key
    ):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.content = b'{"message": "Validation error from server"}'

        with patch(
            "app.desktop.studio_server.copilot_api.refine_spec_v1_copilot_refine_spec_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post("/api/copilot/refine_spec", json=refine_spec_input)
            assert response.status_code == 422
            assert "Validation error from server" in response.json()["message"]


NEW_ROUTE = (
    "app.desktop.studio_server.copilot_api."
    "refine_spec_with_answers_and_name_v1_copilot_"
    "refine_spec_with_answers_and_name_post.asyncio_detailed"
)
OLD_ROUTE = (
    "app.desktop.studio_server.copilot_api."
    "refine_spec_with_answers_v1_copilot_refine_spec_with_answers_post.asyncio_detailed"
)


class TestSubmitQuestionAnswers:
    def test_no_api_key(self, client, submit_answers_input):
        with patch(
            "app.desktop.studio_server.utils.copilot_utils.Config.shared"
        ) as mock_config_shared:
            mock_config = mock_config_shared.return_value
            mock_config.kiln_copilot_api_key = None

            response = client.post(
                "/api/copilot/refine_spec_with_question_answers",
                json=submit_answers_input,
            )
            assert response.status_code == 401
            assert "API key not configured" in response.json()["message"]

    def test_new_route_success_maps_name(
        self, client, submit_answers_input, mock_api_key
    ):
        # The *_and_name route serves the suggested name; it must reach the caller.
        new_output = MagicMock(spec=RefineSpecFromAnswersAndNameOutput)
        new_output.to_dict.return_value = {
            "new_proposed_spec_edits": [
                {
                    "spec_field_name": "tone",
                    "proposed_edit": "Use a formal tone",
                    "reason_for_edit": "User chose formal",
                }
            ],
            "suggested_name": "headline_length",
        }
        new_response = MagicMock()
        new_response.status_code = 200
        new_response.parsed = new_output

        with (
            patch(NEW_ROUTE, new_callable=AsyncMock, return_value=new_response),
            patch(OLD_ROUTE, new_callable=AsyncMock) as old_route,
        ):
            response = client.post(
                "/api/copilot/refine_spec_with_question_answers",
                json=submit_answers_input,
            )
            assert response.status_code == 200
            result = response.json()
            assert result["suggested_name"] == "headline_length"
            assert result["not_incorporated_feedback"] is None
            assert len(result["new_proposed_spec_edits"]) == 1
            # The new route succeeded, so we never touch the fallback.
            old_route.assert_not_awaited()

    def test_missing_route_falls_back_without_name(
        self, client, submit_answers_input, mock_api_key
    ):
        # A 404 means the *_and_name route isn't deployed yet: fall back to the
        # older route, whose response carries no suggested_name.
        new_response = MagicMock()
        new_response.status_code = 404
        new_response.content = b""

        old_output = MagicMock(spec=RefineSpecApiOutput)
        old_output.to_dict.return_value = {
            "new_proposed_spec_edits": [],
            "not_incorporated_feedback": None,
        }
        old_response = MagicMock()
        old_response.status_code = 200
        old_response.parsed = old_output

        with (
            patch(NEW_ROUTE, new_callable=AsyncMock, return_value=new_response),
            patch(
                OLD_ROUTE, new_callable=AsyncMock, return_value=old_response
            ) as old_route,
        ):
            response = client.post(
                "/api/copilot/refine_spec_with_question_answers",
                json=submit_answers_input,
            )
            assert response.status_code == 200
            result = response.json()
            assert result["suggested_name"] is None
            old_route.assert_awaited_once()

    def test_other_error_propagates_without_fallback(
        self, client, submit_answers_input, mock_api_key
    ):
        # Non-404 upstream errors propagate as-is; we do not fall back on them.
        new_response = MagicMock()
        new_response.status_code = 500
        new_response.content = b'{"message": "Boom from server"}'

        with (
            patch(NEW_ROUTE, new_callable=AsyncMock, return_value=new_response),
            patch(OLD_ROUTE, new_callable=AsyncMock) as old_route,
        ):
            response = client.post(
                "/api/copilot/refine_spec_with_question_answers",
                json=submit_answers_input,
            )
            assert response.status_code == 500
            assert "Boom from server" in response.json()["message"]
            old_route.assert_not_awaited()

    def test_validation_error_propagates_without_fallback(
        self, client, submit_answers_input, mock_api_key
    ):
        new_response = MagicMock()
        new_response.status_code = 422
        new_response.content = b'{"message": "Validation error from server"}'

        with (
            patch(NEW_ROUTE, new_callable=AsyncMock, return_value=new_response),
            patch(OLD_ROUTE, new_callable=AsyncMock) as old_route,
        ):
            response = client.post(
                "/api/copilot/refine_spec_with_question_answers",
                json=submit_answers_input,
            )
            assert response.status_code == 422
            assert "Validation error from server" in response.json()["message"]
            old_route.assert_not_awaited()


class TestGenerateBatch:
    def test_generate_batch_no_api_key(self, client, generate_batch_input):
        with patch(
            "app.desktop.studio_server.utils.copilot_utils.Config.shared"
        ) as mock_config_shared:
            mock_config = mock_config_shared.return_value
            mock_config.kiln_copilot_api_key = None

            response = client.post(
                "/api/copilot/generate_batch", json=generate_batch_input
            )
            assert response.status_code == 401
            assert "API key not configured" in response.json()["message"]

    def test_generate_batch_success(self, client, generate_batch_input, mock_api_key):
        mock_output = MagicMock(spec=GenerateBatchOutput)
        mock_output.to_dict.return_value = {"data_by_topic": {}}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = mock_output

        with patch(
            "app.desktop.studio_server.copilot_api.generate_batch_v1_copilot_generate_batch_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(
                "/api/copilot/generate_batch", json=generate_batch_input
            )
            assert response.status_code == 200
            result = response.json()
            assert "data_by_topic" in result

    def test_generate_batch_no_response(
        self, client, generate_batch_input, mock_api_key
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = None

        with patch(
            "app.desktop.studio_server.copilot_api.generate_batch_v1_copilot_generate_batch_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(
                "/api/copilot/generate_batch", json=generate_batch_input
            )
            assert response.status_code == 500
            assert "Failed to generate synthetic data" in response.json()["message"]

    def test_generate_batch_validation_error(
        self, client, generate_batch_input, mock_api_key
    ):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.content = b'{"message": "Validation error from server"}'

        with patch(
            "app.desktop.studio_server.copilot_api.generate_batch_v1_copilot_generate_batch_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(
                "/api/copilot/generate_batch", json=generate_batch_input
            )
            assert response.status_code == 422
            assert "Validation error from server" in response.json()["message"]


class TestCreateSpecWithCopilot:
    @pytest.fixture
    def project_and_task(self, tmp_path):
        project_path = tmp_path / "test_project" / "project.kiln"
        project_path.parent.mkdir()
        project = Project(name="Test Project", path=project_path)
        project.save_to_file()
        task = Task(
            name="Test Task",
            instruction="Test instruction",
            description="Test task",
            parent=project,
        )
        task.save_to_file()
        return project, task

    @pytest.fixture
    def copilot_request_data(self):
        step_config = {
            "task_metadata": {
                "model_name": "gpt-4",
                "model_provider_name": "openai",
            },
            "prompt": "Test prompt",
        }
        return {
            "name": "Test Spec",
            "definition": "The system should respond politely",
            "properties": {
                "spec_type": SpecType.tone.value,
                "core_requirement": "Be polite",
                "tone_description": "Professional and friendly",
            },
            "judge_info": {
                "prompt": "Test prompt",
                "model_name": "gpt-4",
                "model_provider": "openai",
            },
            "sdg_session_config": {
                "topic_generation_config": step_config,
                "input_generation_config": step_config,
                "output_generation_config": step_config,
            },
            "task_prompt_with_example": "Test prompt",
        }

    def test_create_spec_with_copilot_reads_the_named_run_config(
        self, client, project_and_task, copilot_request_data
    ):
        """The legacy path generates examples against the target task's
        capability surface, so it must read the config the spec was written
        against rather than the task default."""
        project, task = project_and_task

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(
                "app.desktop.studio_server.copilot_api.get_copilot_api_key",
                return_value="test_key",
            ),
            patch(
                "app.desktop.studio_server.copilot_api.task_capabilities_for_task",
                new_callable=AsyncMock,
                return_value=([], []),
            ) as mock_capabilities,
            patch(
                "app.desktop.studio_server.copilot_api.generate_copilot_examples",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "app.desktop.studio_server.copilot_api.create_single_turn_dataset",
                return_value=SingleTurnDataset(),
            ),
            patch(
                "app.desktop.studio_server.copilot_api.generate_memorable_name",
                return_value="test-config-name",
            ),
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json={**copilot_request_data, "run_config_id": "rc-9"},
            )

        assert response.status_code == 200
        assert mock_capabilities.await_args.args[1] == "rc-9"

    def test_create_spec_with_copilot_success(
        self, client, project_and_task, copilot_request_data
    ):
        project, task = project_and_task

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(
                "app.desktop.studio_server.copilot_api.get_copilot_api_key",
                return_value="test_key",
            ),
            patch(
                "app.desktop.studio_server.copilot_api.generate_copilot_examples",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "app.desktop.studio_server.copilot_api.create_single_turn_dataset",
                return_value=SingleTurnDataset(),
            ),
            patch(
                "app.desktop.studio_server.copilot_api.generate_memorable_name",
                return_value="test-config-name",
            ),
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=copilot_request_data,
            )

        assert response.status_code == 200
        res = response.json()
        assert res["name"] == "Test Spec"
        assert res["definition"] == "The system should respond politely"
        assert res["eval_id"] is not None

        # Verify models were saved
        evals = task.evals()
        assert len(evals) == 1
        assert evals[0].name == "Test Spec"
        assert evals[0].current_config_id is not None

        # The saved judge is a V2 config: typed LlmJudgeProperties with the
        # judge prompt wrapped into a template (single-turn → I/O data blocks),
        # not the legacy llm_as_judge dict.
        configs = evals[0].configs()
        assert len(configs) == 1
        config = configs[0]
        assert config.config_type == EvalConfigType.v2
        assert isinstance(config.properties, LlmJudgeProperties)
        assert config.properties.model_name == "gpt-4"
        assert config.properties.model_provider == "openai"
        assert "Test prompt" in config.properties.prompt_template
        assert "{{ task_input }}" in config.properties.prompt_template
        assert config.model_name is None and config.model_provider is None

        # Every spec eval carries the same three splits: an EvalInput-backed
        # test split (re-run per run config at eval time) beside TaskRun-backed
        # train and val. This legacy arm deals no val items, so its val split
        # resolves to zero runs rather than to a different splits shape.
        assert evals[0].splits == {
            "test": EvalInputSplit(filter_id="tag::test_test_spec"),
            "train": TaskRunSplit(filter_id="tag::train_test_spec"),
            "val": TaskRunSplit(filter_id="tag::val_test_spec"),
        }
        # Golden is not a split, so splits above doesn't cover it: if it pointed at
        # the test tag, eval-config comparison would score against test items
        # instead of golden.
        assert evals[0].eval_configs_filter_id == "tag::golden_test_spec"

        # Priority and status live on the eval; the spec mirrors them at
        # creation, but reads and later edits go to the eval.
        assert evals[0].priority == Priority.p1
        assert evals[0].status == EvalStatus.active
        assert evals[0].resolved_priority() == Priority.p1
        assert evals[0].resolved_status() == EvalStatus.active

        specs = task.specs()
        assert len(specs) == 1
        assert specs[0].eval_id == evals[0].id

        # Single-turn on disk: the test split is EvalInput-backed (expressible
        # only in `splits`); train rides beside it. `splits` is the single home:
        # the deprecated flat fields are written null.
        on_disk = json.loads(evals[0].path.read_text())
        assert on_disk["splits"] == {
            "test": {"source": "eval_input", "filter_id": "tag::test_test_spec"},
            "train": {"source": "task_run", "filter_id": "tag::train_test_spec"},
            "val": {"source": "task_run", "filter_id": "tag::val_test_spec"},
        }
        assert on_disk["eval_set_filter_id"] is None
        assert on_disk["train_set_filter_id"] is None

    def test_generation_sees_the_tasks_capability_surface(
        self,
        client,
        project_and_task,
        copilot_request_data,
        give_task_one_tool_and_skill,
    ):
        """The generator is told what the target task can do, so the synthetic
        inputs it writes can exercise the tools and skills the task has."""
        project, task = project_and_task
        give_task_one_tool_and_skill(project, task)

        generate_mock = AsyncMock(return_value=[])
        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(
                "app.desktop.studio_server.copilot_api.get_copilot_api_key",
                return_value="test_key",
            ),
            patch(
                "app.desktop.studio_server.copilot_api.generate_copilot_examples",
                generate_mock,
            ),
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=copilot_request_data,
            )

        assert response.status_code == 200
        target_task_info = generate_mock.await_args.kwargs["target_task_info"]
        assert target_task_info.task_tools == [
            TaskToolInfoApi(
                name="add", description="Add two numbers together and return the result"
            )
        ]
        assert target_task_info.task_skills == [
            TaskSkillInfoApi(
                name="refund-policy", description="How and when refunds are issued."
            )
        ]

    def test_single_turn_save_writes_eval_inputs_and_splits_to_disk(
        self, client, project_and_task, copilot_request_data
    ):
        # Asserts on the SAVED BYTES, not the in-memory models: the eval slice
        # moving stores is only visible on disk (which files exist, and what
        # the eval's splits point at).
        project, task = project_and_task
        # 9 generated examples split 2:1 → 6 train runs + a 3-item eval slice;
        # the 2 reviewed examples become the golden answer key.
        generated = [
            SampleApi(input=f"generated input {i}", output=f"generated output {i}")
            for i in range(9)
        ]
        copilot_request_data["reviewed_examples"] = [
            {
                "input": "reviewed input 0",
                "output": "reviewed output 0",
                "model_says_meets_spec": False,
                "user_says_meets_spec": False,
                "feedback": "Fabricated a return window.",
                "claim_review": {
                    "judge_score": "fail",
                    "judge_reasoning": "Judge reasoning here.",
                    "overview": "The user asked about returns.",
                    "claims": [
                        {
                            "text": "The agent stated a return window [1].",
                            "human_grade": "agree",
                            "human_feedback": None,
                        },
                        {
                            "text": "It fails because the window was invented [1].",
                            "human_grade": "agree",
                            "human_feedback": None,
                        },
                    ],
                    "human_verdict": "fail",
                },
            },
            {
                "input": "reviewed input 1",
                "output": "reviewed output 1",
                "model_says_meets_spec": True,
                "user_says_meets_spec": True,
                "feedback": "",
            },
        ]

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(
                "app.desktop.studio_server.copilot_api.get_copilot_api_key",
                return_value="test_key",
            ),
            patch(
                "app.desktop.studio_server.copilot_api.generate_copilot_examples",
                new_callable=AsyncMock,
                return_value=generated,
            ),
            patch(
                "app.desktop.studio_server.copilot_api.generate_memorable_name",
                return_value="single-turn-judge",
            ),
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=copilot_request_data,
            )
        assert response.status_code == 200, response.text

        eval_path = task.evals()[0].path
        first_bytes = eval_path.read_text()
        on_disk = json.loads(first_bytes)

        # Test split EvalInput-backed, train TaskRun-backed, both in `splits` —
        # the single home. The deprecated flat fields are written null.
        assert on_disk["splits"] == {
            "test": {"source": "eval_input", "filter_id": "tag::test_test_spec"},
            "train": {"source": "task_run", "filter_id": "tag::train_test_spec"},
            "val": {"source": "task_run", "filter_id": "tag::val_test_spec"},
        }
        # Golden stays in the eval-configs filter the judge is calibrated against.
        assert on_disk["eval_set_filter_id"] is None
        assert on_disk["train_set_filter_id"] is None
        assert on_disk["eval_configs_filter_id"] == "tag::golden_test_spec"

        # Reload → save again is byte-stable: the splits survive a round trip
        # rather than living only in the freshly-built instance.
        reloaded = Eval.load_from_file(eval_path)
        reloaded.save_to_file()
        assert eval_path.read_text() == first_bytes

        # The eval slice on disk: one EvalInput per eval-slice example,
        # carrying the generated INPUT verbatim and nothing else.
        eval_inputs = task.eval_inputs()
        assert len(eval_inputs) == 3
        slice_inputs = {ei.data.user_message.text for ei in eval_inputs}
        assert slice_inputs <= {ex.input for ex in generated}
        for eval_input in eval_inputs:
            assert eval_input.data.type == "single_turn"
            assert "test_test_spec" in eval_input.tags
            # The generated output is discarded at mint — the runner writes a
            # fresh one per run config, so a stored one would never be judged.
            assert "generated output" not in eval_input.path.read_text()

        # No run carries the eval tag any more: the dataset is the 6 train
        # runs plus the 2 golden ones, and nothing is in two splits at once.
        # The legacy arm never reaches the train:val deal, so no run is val
        # either — its generated pool is split train:eval by the v1 math.
        runs = task.runs()
        assert len(runs) == 8
        by_tag = {
            tag: [run for run in runs if tag in run.tags]
            for tag in (
                "train_test_spec",
                "golden_test_spec",
                "test_test_spec",
                "val_test_spec",
            )
        }
        assert len(by_tag["train_test_spec"]) == 6
        assert len(by_tag["golden_test_spec"]) == 2
        assert by_tag["test_test_spec"] == []
        assert by_tag["val_test_spec"] == []

        # The golden answer key rides through untouched: human verdicts as
        # requirement ratings, plus the feedback and per-claim grades.
        golden_by_input = {run.input: run for run in by_tag["golden_test_spec"]}
        rating_key = "named::Test Spec"
        failed = golden_by_input["reviewed input 0"]
        assert failed.output.rating.requirement_ratings[rating_key].value == 0.0
        assert (
            failed.output.rating.requirement_ratings[rating_key].type
            == TaskOutputRatingType.pass_fail
        )
        assert [fb.feedback for fb in failed.feedback()] == [
            "Fabricated a return window."
        ]
        assert len(failed.claim_reviews()) == 1
        assert failed.claim_reviews()[0].judge_score == "fail"

        passed = golden_by_input["reviewed input 1"]
        assert passed.output.rating.requirement_ratings[rating_key].value == 1.0
        assert passed.feedback() == []
        assert passed.claim_reviews() == []

    def test_single_turn_save_failure_after_eval_slice_rolls_back(
        self, client, project_and_task, copilot_request_data
    ):
        # A failure AFTER the eval slice hit disk reverses everything: the
        # EvalInputs are Task children, so an incomplete save would otherwise
        # leave a tagged slice pointing at an eval that no longer exists.
        project, task = project_and_task
        generated = [
            SampleApi(input=f"generated input {i}", output=f"generated output {i}")
            for i in range(9)
        ]

        from app.desktop.studio_server.utils import copilot_utils

        def persist_then_boom(*args, **kwargs):
            copilot_utils.persist_eval_slice(*args, **kwargs)
            raise RuntimeError("disk full")

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(
                "app.desktop.studio_server.copilot_api.get_copilot_api_key",
                return_value="test_key",
            ),
            patch(
                "app.desktop.studio_server.copilot_api.generate_copilot_examples",
                new_callable=AsyncMock,
                return_value=generated,
            ),
            patch(
                "app.desktop.studio_server.copilot_api.persist_eval_slice",
                side_effect=persist_then_boom,
            ),
            # The endpoint re-raises after rollback; TestClient propagates it.
            pytest.raises(RuntimeError, match="disk full"),
        ):
            client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=copilot_request_data,
            )

        assert task.evals() == []
        assert task.specs() == []
        assert task.runs() == []
        assert task.eval_inputs() == []


class TestCreateSpecWithCopilotMultiTurn:
    """Multi-turn save path: tag existing chain leaves (golden/train) and mint
    EvalInputs from the driven cases instead of synthesising new examples.
    """

    BATCH_TAG = "abc123def456"

    @pytest.fixture
    def project_and_task(self, tmp_path):
        project_path = tmp_path / "test_project" / "project.kiln"
        project_path.parent.mkdir()
        project = Project(name="Test Project", path=project_path)
        project.save_to_file()
        task = Task(
            name="Test Task",
            instruction="Test instruction",
            description="Test task",
            parent=project,
        )
        task.save_to_file()
        return project, task

    @pytest.fixture
    def synthetic_chain_leaves(self, project_and_task):
        """Persist eight single-run "chains" tagged like the multi-turn runner
        leaves them. Single TaskRuns (no actual multi-turn parents) are
        sufficient: the endpoint only cares about the leaf tag. Eight leaves
        give the split room for a non-empty golden slice (caps at 25% = 2);
        the rest are train.
        """
        _, task = project_and_task
        source = DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": "haiku",
                "model_provider": "openrouter",
                "adapter_name": "kiln_synthetic_user_runner",
            },
        )
        leaves = []
        for i in range(8):
            run = TaskRun(
                parent=task,
                input=f"input {i}",
                input_source=source,
                output=TaskOutput(output=f"output {i}", source=source),
                tags=[
                    "synthetic_user_case",
                    f"synthetic_user_batch:{TestCreateSpecWithCopilotMultiTurn.BATCH_TAG}",
                ],
            )
            run.save_to_file()
            leaves.append(run)
        return leaves

    @staticmethod
    def _driven_case(idx: int) -> dict:
        return {
            "seed_prompt": f"seed prompt {idx}",
            "synthetic_user_info": (
                f"<persona>persona {idx}</persona>"
                f"<goal>goal {idx}</goal>"
                f"<behavior_guidance>guidance {idx}</behavior_guidance>"
            ),
            "scenario_index": idx,
        }

    @pytest.fixture
    def multi_turn_request_data(self):
        return {
            "name": "Multi Turn Spec",
            "definition": "The agent should not fabricate policies",
            "properties": {
                "spec_type": SpecType.issue.value,
                "issue_description": "Don't make stuff up",
            },
            "evaluate_full_trace": True,
            "judge_info": {
                "prompt": "Test prompt",
                "model_name": "gpt-4",
                "model_provider": "openai",
            },
            "multi_turn": {
                "batch_tag": TestCreateSpecWithCopilotMultiTurn.BATCH_TAG,
                "cases": [self._driven_case(i) for i in range(8)],
                "drive_config": {
                    "model_name": "claude_4_5_haiku",
                    "model_provider": "openrouter",
                    "turns": 5,
                },
            },
            "task_prompt_with_example": "Test prompt",
        }

    @staticmethod
    def _reviewed_chain(leaf_run_id: str, meets_spec: bool) -> dict:
        return {
            "leaf_run_id": leaf_run_id,
            "user_says_meets_spec": meets_spec,
            "feedback": "" if meets_spec else "Fabricated a return window.",
            "claim_review": {
                "judge_score": "pass" if meets_spec else "fail",
                "judge_reasoning": "Judge reasoning here.",
                "overview": "The user asked about returns.",
                "claims": [
                    {
                        "text": "The agent stated a return window [1].",
                        "human_grade": "agree",
                        "human_feedback": None,
                    }
                ],
                "human_verdict": "pass" if meets_spec else "fail",
            },
        }

    def test_multi_turn_save_success_tags_chains_and_creates_eval(
        self,
        client,
        project_and_task,
        synthetic_chain_leaves,
        multi_turn_request_data,
    ):
        project, task = project_and_task
        # Two of the three chains were reviewed: one pass, one fail.
        multi_turn_request_data["multi_turn"]["reviewed_chains"] = [
            self._reviewed_chain(synthetic_chain_leaves[0].id, meets_spec=False),
            self._reviewed_chain(synthetic_chain_leaves[1].id, meets_spec=True),
        ]

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(
                "app.desktop.studio_server.copilot_api.generate_memorable_name",
                return_value="multi-turn-judge",
            ),
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )

        assert response.status_code == 200, response.text
        res = response.json()
        assert res["name"] == "Multi Turn Spec"
        assert res["eval_id"] is not None
        # Multi-turn doesn't snapshot a generation config on the spec —
        # the operational state lives on the Eval.
        assert res["synthetic_data_generation_session_config"] is None

        # Eval: full_trace data type + judge config attached. The eval slice
        # is EvalInput-typed (re-driven per run config at eval time) with the
        # drive settings stamped on each item; golden and train stay TaskRun.
        evals = task.evals()
        assert len(evals) == 1
        eval_obj = evals[0]
        assert eval_obj.evaluation_data_type == EvalDataType.full_trace
        assert eval_obj.model_dump()["eval_set_filter_id"] is None
        # The save path writes the EvalInput-backed test split natively; the
        # on-disk shape is covered by the saved-bytes test below.
        assert eval_obj.splits["test"] == EvalInputSplit(
            filter_id="tag::test_multi_turn_spec"
        )
        assert eval_obj.splits["train"] == TaskRunSplit(
            filter_id="tag::train_multi_turn_spec"
        )
        assert eval_obj.splits["val"] == TaskRunSplit(
            filter_id="tag::val_multi_turn_spec"
        )
        assert eval_obj.model_dump()["train_set_filter_id"] is None
        assert eval_obj.current_config_id is not None

        # The saved judge is a V2 config with a multi-turn (trace) template.
        configs = eval_obj.configs()
        assert len(configs) == 1
        assert configs[0].config_type == EvalConfigType.v2
        assert isinstance(configs[0].properties, LlmJudgeProperties)
        assert "{{ trace | format_trace }}" in configs[0].properties.prompt_template

        # The eval slice: one EvalInput per driven case, carrying the seed,
        # the typed persona, the stamped drive config, and provenance tags
        # (batch + scenario).
        eval_inputs = task.eval_inputs()
        assert len(eval_inputs) == 8
        inputs_by_seed = {ei.data.first_message.text: ei for ei in eval_inputs}
        first = inputs_by_seed["seed prompt 0"]
        assert first.data.synthetic_user_info.persona == "persona 0"
        assert first.data.synthetic_user_info.goal == "goal 0"
        assert first.data.synthetic_user_info.behavior_guidance == "guidance 0"
        assert set(first.tags) == {
            "test_multi_turn_spec",
            f"synthetic_user_batch:{self.BATCH_TAG}",
            "scenario:0",
        }
        # Every item in the slice is stamped with the batch's drive settings.
        for ei in eval_inputs:
            assert ei.data.drive_config is not None
            assert ei.data.drive_config.model_name == "claude_4_5_haiku"
            assert ei.data.drive_config.model_provider == "openrouter"
            assert ei.data.drive_config.turns == 5

        # Chains split into DISJOINT slices: each leaf carries exactly one of
        # golden/train/val (on top of its synthetic_user_* tags) — the eval
        # slice lives on the EvalInputs above, not on chains. Golden caps at
        # 25% of 8 = 2, which here equals the two rated leaves — so both
        # become golden (the answer key). The six unreviewed leaves are dealt
        # 4 train / 2 val at the 40:25 ratio.
        split_tags = {
            "train_multi_turn_spec",
            "golden_multi_turn_spec",
            "val_multi_turn_spec",
        }
        runs_by_id = {run.id: run for run in task.runs()}
        for leaf in task.runs():
            assert len(split_tags & set(leaf.tags)) == 1
            assert "test_multi_turn_spec" not in leaf.tags
            assert "synthetic_user_case" in leaf.tags
        by_tag = {
            tag: {run.id for run in task.runs() if tag in run.tags}
            for tag in split_tags
        }
        assert len(by_tag["train_multi_turn_spec"]) == 4
        assert len(by_tag["val_multi_turn_spec"]) == 2
        # Golden == exactly the two reviewed leaves (rated count == the 25%
        # cap), so the deal only ever touched the unreviewed remainder.
        reviewed_ids = {
            synthetic_chain_leaves[0].id,
            synthetic_chain_leaves[1].id,
        }
        assert by_tag["golden_multi_turn_spec"] == reviewed_ids
        assert by_tag["val_multi_turn_spec"].isdisjoint(reviewed_ids)
        # An unreviewed leaf is held out in one of the dealt slices, never golden.
        unreviewed_tags = set(runs_by_id[synthetic_chain_leaves[2].id].tags)
        assert "golden_multi_turn_spec" not in unreviewed_tags

        # Reviewed leaves carry golden ratings matching the review clicks,
        # plus feedback + per-claim grades; the unreviewed leaf stays unrated.
        rating_key = "named::Multi Turn Spec"
        failed = runs_by_id[synthetic_chain_leaves[0].id]
        assert failed.output.rating.requirement_ratings[rating_key].value == 0.0
        assert (
            failed.output.rating.requirement_ratings[rating_key].type
            == TaskOutputRatingType.pass_fail
        )
        assert len(failed.feedback()) == 1
        assert failed.feedback()[0].feedback == "Fabricated a return window."
        assert len(failed.claim_reviews()) == 1
        assert failed.claim_reviews()[0].judge_score == "fail"
        assert failed.claim_reviews()[0].human_verdict == "fail"

        passed = runs_by_id[synthetic_chain_leaves[1].id]
        assert passed.output.rating.requirement_ratings[rating_key].value == 1.0
        assert passed.feedback() == []
        assert len(passed.claim_reviews()) == 1

        unreviewed = runs_by_id[synthetic_chain_leaves[2].id]
        assert unreviewed.output.rating is None
        assert unreviewed.claim_reviews() == []

    def test_multi_turn_save_reads_no_capabilities(
        self,
        client,
        project_and_task,
        synthetic_chain_leaves,
        multi_turn_request_data,
    ):
        """A wizard save tags runs already on disk and writes files; it
        generates nothing, so it describes no task to the copilot and reads no
        run config's tools or skills. Only the legacy generating path does."""
        project, task = project_and_task

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(
                "app.desktop.studio_server.copilot_api.generate_memorable_name",
                return_value="multi-turn-judge",
            ),
            patch(
                "app.desktop.studio_server.copilot_api.task_capabilities_for_task",
                new_callable=AsyncMock,
            ) as mock_capabilities,
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )

        assert response.status_code == 200, response.text
        mock_capabilities.assert_not_awaited()

    def test_multi_turn_save_writes_splits_natively_to_disk(
        self,
        client,
        project_and_task,
        synthetic_chain_leaves,
        multi_turn_request_data,
    ):
        # Asserts on the SAVED BYTES, not the in-memory model: a train split
        # homed by dict assignment instead of set_split would look identical
        # in memory and only diverge in the serialized file.
        project, task = project_and_task

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(
                "app.desktop.studio_server.copilot_api.generate_memorable_name",
                return_value="multi-turn-judge",
            ),
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )
        assert response.status_code == 200, response.text

        eval_path = task.evals()[0].path
        first_bytes = eval_path.read_text()
        on_disk = json.loads(first_bytes)

        # Test split EvalInput-backed, train and val TaskRun-backed, all three
        # in `splits` — the single home. The deprecated flat fields are written
        # null.
        assert on_disk["splits"] == {
            "test": {
                "source": "eval_input",
                "filter_id": "tag::test_multi_turn_spec",
            },
            "train": {
                "source": "task_run",
                "filter_id": "tag::train_multi_turn_spec",
            },
            "val": {
                "source": "task_run",
                "filter_id": "tag::val_multi_turn_spec",
            },
        }
        assert on_disk["train_set_filter_id"] is None
        assert on_disk["eval_set_filter_id"] is None
        # Golden slice rides along unchanged.
        assert on_disk["eval_configs_filter_id"] == "tag::golden_multi_turn_spec"
        # The retired pre-splits key never reaches disk, and drive settings
        # live on the eval items, not the eval.
        assert "eval_input_filter_id" not in first_bytes
        assert "multi_turn_drive_config" not in first_bytes

        # The stamped drive config is written per item, nested under data.
        eval_input_path = task.eval_inputs()[0].path
        on_disk_item = json.loads(eval_input_path.read_text())
        assert on_disk_item["data"]["drive_config"] == {
            "model_name": "claude_4_5_haiku",
            "model_provider": "openrouter",
            "turns": 5,
        }

        # Reload → save again is byte-stable: the split homing survives a
        # round trip rather than living only in the freshly-built instance.
        reloaded = Eval.load_from_file(eval_path)
        reloaded.save_to_file()
        assert eval_path.read_text() == first_bytes

    def test_multi_turn_save_unknown_leaf_fails_before_any_save(
        self,
        client,
        project_and_task,
        synthetic_chain_leaves,
        multi_turn_request_data,
    ):
        # A reviewed chain referencing a leaf outside the batch is rejected up
        # front — nothing is created and no leaf is mutated.
        project, task = project_and_task
        multi_turn_request_data["multi_turn"]["reviewed_chains"] = [
            self._reviewed_chain(synthetic_chain_leaves[0].id, meets_spec=False),
            self._reviewed_chain("not_a_real_leaf", meets_spec=True),
        ]

        with patch(
            "app.desktop.studio_server.copilot_api.task_from_id",
            return_value=task,
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )

        assert response.status_code == 404
        assert "not_a_real_leaf" in response.json()["message"]
        assert len(task.evals()) == 0
        assert len(task.specs()) == 0
        assert len(task.eval_inputs()) == 0
        for leaf in task.runs():
            assert leaf.output.rating is None
            assert leaf.feedback() == []
            assert leaf.claim_reviews() == []
            assert "train_multi_turn_spec" not in leaf.tags
            assert "golden_multi_turn_spec" not in leaf.tags

    def test_multi_turn_save_rejects_duplicate_reviewed_leaves(
        self,
        client,
        project_and_task,
        synthetic_chain_leaves,
        multi_turn_request_data,
    ):
        # The same leaf reviewed twice is a malformed request — rejected up
        # front (422) with nothing created or mutated.
        project, task = project_and_task
        multi_turn_request_data["multi_turn"]["reviewed_chains"] = [
            self._reviewed_chain(synthetic_chain_leaves[0].id, meets_spec=False),
            self._reviewed_chain(synthetic_chain_leaves[0].id, meets_spec=True),
        ]

        with patch(
            "app.desktop.studio_server.copilot_api.task_from_id",
            return_value=task,
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )

        assert response.status_code == 422
        assert "at most once" in response.json()["message"]
        assert len(task.evals()) == 0
        assert len(task.specs()) == 0
        for leaf in task.runs():
            assert leaf.output.rating is None

    def test_multi_turn_save_failure_mid_rating_rolls_back(
        self,
        client,
        project_and_task,
        synthetic_chain_leaves,
        multi_turn_request_data,
    ):
        # A failure AFTER tagging/rating started reverses everything: leaf
        # tags and ratings revert, created models are deleted.
        project, task = project_and_task
        multi_turn_request_data["multi_turn"]["reviewed_chains"] = [
            self._reviewed_chain(synthetic_chain_leaves[0].id, meets_spec=False),
        ]

        from app.desktop.studio_server.utils import copilot_utils

        # Tags are snapshotted at the moment of failure so the reversal
        # assertions below can't pass vacuously against a tag never written.
        tags_at_failure: set[str] = set()

        def rate_then_boom(*args, **kwargs):
            copilot_utils.rate_reviewed_batch_runs(*args, **kwargs)
            for leaf in args[0]:
                tags_at_failure.update(leaf.tags)
            raise RuntimeError("disk full")

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(
                "app.desktop.studio_server.copilot_api.rate_reviewed_batch_runs",
                side_effect=rate_then_boom,
            ),
            # The endpoint re-raises after rollback; TestClient propagates it.
            pytest.raises(RuntimeError, match="disk full"),
        ):
            client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )

        assert len(task.evals()) == 0
        assert len(task.specs()) == 0
        # The eval slice rolled back too: no orphan EvalInputs.
        assert len(task.eval_inputs()) == 0
        # All three split tags were on the leaves when the save blew up, and
        # rollback took every one back off — val is reversed like the others.
        assert {
            "train_multi_turn_spec",
            "golden_multi_turn_spec",
            "val_multi_turn_spec",
        } <= tags_at_failure
        for leaf in task.runs():
            assert leaf.output.rating is None
            assert leaf.feedback() == []
            assert leaf.claim_reviews() == []
            assert set(leaf.tags) == {
                "synthetic_user_case",
                f"synthetic_user_batch:{self.BATCH_TAG}",
            }

    def test_multi_turn_save_malformed_case_blob_is_422(
        self,
        client,
        project_and_task,
        synthetic_chain_leaves,
        multi_turn_request_data,
    ):
        # A case whose persona blob doesn't parse fails before anything is
        # created — the EvalInputs are built (and validated) up front.
        project, task = project_and_task
        multi_turn_request_data["multi_turn"]["cases"][3]["synthetic_user_info"] = (
            "no tags here at all"
        )

        with patch(
            "app.desktop.studio_server.copilot_api.task_from_id",
            return_value=task,
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )

        assert response.status_code == 422
        assert "Case 3" in response.json()["message"]
        assert len(task.evals()) == 0
        assert len(task.specs()) == 0
        assert len(task.eval_inputs()) == 0

    def test_duplicate_spec_name_is_409(
        self, client, project_and_task, multi_turn_request_data
    ):
        """A re-submitted save under an existing spec name (any casing) is
        rejected before any generation or model creation."""
        from kiln_ai.datamodel.spec import Spec, SpecStatus

        project, task = project_and_task
        existing = Spec(
            parent=task,
            name="MULTI TURN SPEC",
            definition="already here",
            properties={
                "spec_type": SpecType.issue.value,
                "issue_description": "existing",
            },
            status=SpecStatus.active,
            tags=[],
            eval_id="unused_eval_id",
        )
        existing.save_to_file()

        with patch(
            "app.desktop.studio_server.copilot_api.task_from_id",
            return_value=task,
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )

        assert response.status_code == 409
        assert "already exists" in response.json()["message"]
        assert len(task.evals()) == 0

    def test_tag_normalized_spec_name_collision_is_409(
        self, client, project_and_task, multi_turn_request_data
    ):
        """ "Multi_Turn_Spec" and "Multi Turn Spec" differ as names but produce
        identical eval tags (lowercase, spaces→underscores) — saving the
        second would silently share the first's datasets."""
        from kiln_ai.datamodel.spec import Spec, SpecStatus

        project, task = project_and_task
        existing = Spec(
            parent=task,
            name="Multi_Turn_Spec",
            definition="already here",
            properties={
                "spec_type": SpecType.issue.value,
                "issue_description": "existing",
            },
            status=SpecStatus.active,
            tags=[],
            eval_id="unused_eval_id",
        )
        existing.save_to_file()

        with patch(
            "app.desktop.studio_server.copilot_api.task_from_id",
            return_value=task,
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )

        assert response.status_code == 409
        assert len(task.evals()) == 0

    def test_spec_name_without_json_key_chars_is_422(
        self, client, project_and_task, multi_turn_request_data
    ):
        """A name with no [a-z0-9_] characters (e.g. fully non-ASCII) yields
        an empty judge score key — the save would persist an eval that can
        never run."""
        project, task = project_and_task
        multi_turn_request_data["name"] = "中文规格"

        with patch(
            "app.desktop.studio_server.copilot_api.task_from_id",
            return_value=task,
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )

        assert response.status_code == 422
        assert "score key" in response.json()["message"]
        assert len(task.evals()) == 0
        assert len(task.specs()) == 0

    def test_reference_answer_spec_type_is_400(
        self, client, project_and_task, multi_turn_request_data
    ):
        """The builder's judge template never renders a reference answer;
        saving one would mis-score every run. Guarded until supported."""
        project, task = project_and_task
        multi_turn_request_data["properties"] = {
            "spec_type": SpecType.reference_answer_accuracy.value,
            "core_requirement": "Answers must match the reference.",
            "reference_answer_accuracy_description": "Compare to reference.",
            "accurate_examples": "example a",
            "inaccurate_examples": "example b",
        }

        with patch(
            "app.desktop.studio_server.copilot_api.task_from_id",
            return_value=task,
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )

        assert response.status_code == 400
        assert "Reference-answer" in response.json()["message"]
        assert len(task.evals()) == 0
        assert len(task.specs()) == 0

    def test_spec_name_over_short_limit_is_422(
        self, client, project_and_task, multi_turn_request_data
    ):
        """The spec name becomes the eval's EvalOutputScore.name (max 32) —
        longer names must fail request validation, not 500 mid-save."""
        project, task = project_and_task
        multi_turn_request_data["name"] = "A Spec Name That Is Way Too Long For Scores"

        with patch(
            "app.desktop.studio_server.copilot_api.task_from_id",
            return_value=task,
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )

        assert response.status_code == 422
        assert len(task.evals()) == 0
        assert len(task.specs()) == 0

    def test_multi_turn_save_404_when_batch_tag_matches_nothing(
        self, client, project_and_task, multi_turn_request_data
    ):
        project, task = project_and_task

        with patch(
            "app.desktop.studio_server.copilot_api.task_from_id",
            return_value=task,
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )

        assert response.status_code == 404
        assert "batch_tag" in response.json()["message"]
        # No models created when the lookup fails up front.
        assert len(task.evals()) == 0
        assert len(task.specs()) == 0

    def test_validator_rejects_both_multi_turn_and_sdg_config(
        self, client, project_and_task, multi_turn_request_data
    ):
        project, task = project_and_task
        step_config = {
            "task_metadata": {
                "model_name": "gpt-4",
                "model_provider_name": "openai",
            },
            "prompt": "Test prompt",
        }
        multi_turn_request_data["sdg_session_config"] = {
            "topic_generation_config": step_config,
            "input_generation_config": step_config,
            "output_generation_config": step_config,
        }

        with patch(
            "app.desktop.studio_server.copilot_api.task_from_id",
            return_value=task,
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )

        assert response.status_code == 422
        body = response.json()
        # Pydantic surfaces the validator message somewhere in the response.
        assert "multi_turn" in str(body) and "sdg_session_config" in str(body)

    def test_validator_rejects_neither_multi_turn_nor_sdg_config(
        self, client, project_and_task, multi_turn_request_data
    ):
        project, task = project_and_task
        del multi_turn_request_data["multi_turn"]

        with patch(
            "app.desktop.studio_server.copilot_api.task_from_id",
            return_value=task,
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )

        assert response.status_code == 422
        assert "multi_turn" in str(response.json())

    def test_validator_rejects_multi_turn_without_evaluate_full_trace(
        self, client, project_and_task, multi_turn_request_data
    ):
        project, task = project_and_task
        multi_turn_request_data["evaluate_full_trace"] = False

        with patch(
            "app.desktop.studio_server.copilot_api.task_from_id",
            return_value=task,
        ):
            response = client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=multi_turn_request_data,
            )

        assert response.status_code == 422
        assert "evaluate_full_trace" in str(response.json())


class TestCreateSpecWithCopilotSingleTurnBatch:
    """The wizard single-turn save path: tag the pipeline's existing
    batch-tagged runs (golden/train, verdicts onto golden) and mint the eval
    slice from the generated inputs. Nothing is generated at save time —
    the sibling of the multi-turn path above, not of the legacy sdg one.
    """

    BATCH_TAG = "st1234abcd56"

    @pytest.fixture
    def project_and_task(self, tmp_path):
        project_path = tmp_path / "test_project" / "project.kiln"
        project_path.parent.mkdir()
        project = Project(name="Test Project", path=project_path)
        project.save_to_file()
        task = Task(
            name="Test Task",
            instruction="Test instruction",
            description="Test task",
            parent=project,
        )
        task.save_to_file()
        return project, task

    @pytest.fixture
    def batch_runs(self, project_and_task):
        """Persist eight runs tagged like the single-turn pipeline leaves
        them. Eight give the split room for a non-empty golden slice (caps
        at 25% = 2); the rest are train."""
        _, task = project_and_task
        source = DataSource(
            type=DataSourceType.synthetic,
            properties={
                "model_name": "haiku",
                "model_provider": "openrouter",
                "adapter_name": "kiln_eval_builder_single_turn",
            },
        )
        runs = []
        for i in range(8):
            run = TaskRun(
                parent=task,
                input=f"input {i}",
                input_source=source,
                output=TaskOutput(output=f"output {i}", source=source),
                tags=[
                    "single_turn_drive",
                    f"single_turn_drive_batch:{TestCreateSpecWithCopilotSingleTurnBatch.BATCH_TAG}",
                ],
            )
            run.save_to_file()
            runs.append(run)
        return runs

    @pytest.fixture
    def single_turn_request_data(self):
        return {
            "name": "Single Turn Spec",
            "definition": "The agent should not fabricate policies",
            "properties": {
                "spec_type": SpecType.issue.value,
                "issue_description": "Don't make stuff up",
            },
            "evaluate_full_trace": True,
            "judge_info": {
                "prompt": "Test prompt",
                "model_name": "gpt-4",
                "model_provider": "openai",
            },
            "single_turn": {
                "batch_tag": TestCreateSpecWithCopilotSingleTurnBatch.BATCH_TAG,
                "inputs": [f"input {i}" for i in range(8)],
            },
            "task_sample": {
                "input": "What's your return window?",
                "output": "Returns are accepted within 14 days.",
            },
        }

    @staticmethod
    def _reviewed_run(run_id: str, meets_spec: bool) -> dict:
        return {
            "leaf_run_id": run_id,
            "user_says_meets_spec": meets_spec,
            "feedback": "" if meets_spec else "Fabricated a return window.",
            "claim_review": {
                "judge_score": "pass" if meets_spec else "fail",
                "judge_reasoning": "Judge reasoning here.",
                "overview": "The user asked about returns.",
                "claims": [
                    {
                        "text": "The agent stated a return window [1].",
                        "human_grade": "agree",
                        "human_feedback": None,
                    }
                ],
                "human_verdict": "pass" if meets_spec else "fail",
            },
        }

    def _post(self, client, project, task, request_data):
        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(
                "app.desktop.studio_server.copilot_api.generate_memorable_name",
                return_value="single-turn-judge",
            ),
        ):
            return client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=request_data,
            )

    def test_single_turn_save_tags_runs_and_creates_eval(
        self, client, project_and_task, batch_runs, single_turn_request_data
    ):
        project, task = project_and_task
        # Two of the eight runs were reviewed: one pass, one fail.
        single_turn_request_data["single_turn"]["reviewed_runs"] = [
            self._reviewed_run(batch_runs[0].id, meets_spec=False),
            self._reviewed_run(batch_runs[1].id, meets_spec=True),
        ]

        response = self._post(client, project, task, single_turn_request_data)

        assert response.status_code == 200, response.text
        res = response.json()
        assert res["name"] == "Single Turn Spec"
        assert res["eval_id"] is not None
        # The wizard arm generates nothing, so no generation config snapshot
        # lands on the spec; the picked grounding sample does.
        assert res["synthetic_data_generation_session_config"] is None
        assert res["task_sample"]["input"] == "What's your return window?"

        # Eval: final_answer data type (the pipeline judged final answers),
        # EvalInput-backed test split, TaskRun-backed train in the legacy
        # field.
        evals = task.evals()
        assert len(evals) == 1
        eval_obj = evals[0]
        # Single-turn saves a full-trace eval now: the builder judged the
        # transcript, so the eval that ships judges the same thing.
        assert eval_obj.evaluation_data_type == EvalDataType.full_trace
        assert eval_obj.model_dump()["eval_set_filter_id"] is None
        assert eval_obj.splits["test"] == EvalInputSplit(
            filter_id="tag::test_single_turn_spec"
        )
        assert eval_obj.splits["train"] == TaskRunSplit(
            filter_id="tag::train_single_turn_spec"
        )
        assert eval_obj.splits["val"] == TaskRunSplit(
            filter_id="tag::val_single_turn_spec"
        )
        assert eval_obj.model_dump()["train_set_filter_id"] is None
        assert eval_obj.current_config_id is not None

        # The saved judge is a V2 config with the single-turn (I/O) template,
        # never the trace one.
        configs = eval_obj.configs()
        assert len(configs) == 1
        assert configs[0].config_type == EvalConfigType.v2
        assert isinstance(configs[0].properties, LlmJudgeProperties)
        # The judge template renders the transcript on both arms now, so the
        # saved judge reads what the builder's judge read.
        assert "format_trace" in configs[0].properties.prompt_template

        # The eval slice: one inputs-only EvalInput per generated input,
        # tagged with the eval slice + the drive batch it came from.
        eval_inputs = task.eval_inputs()
        assert len(eval_inputs) == 8
        assert {ei.data.type for ei in eval_inputs} == {"single_turn"}
        assert {ei.data.user_message.text for ei in eval_inputs} == {
            f"input {i}" for i in range(8)
        }
        assert all(
            set(ei.tags)
            == {
                "test_single_turn_spec",
                f"single_turn_drive_batch:{self.BATCH_TAG}",
            }
            for ei in eval_inputs
        )

        # Runs split into DISJOINT golden/train/val slices on top of their
        # pipeline tags. Golden caps at 25% of 8 = 2 = the reviewed runs; the
        # 6 unreviewed runs are dealt 4 train / 2 val at the 40:25 ratio.
        split_tags = {
            "train_single_turn_spec",
            "golden_single_turn_spec",
            "val_single_turn_spec",
        }
        runs_by_id = {run.id: run for run in task.runs()}
        for run in task.runs():
            assert len(split_tags & set(run.tags)) == 1
            assert "test_single_turn_spec" not in run.tags
            assert "single_turn_drive" in run.tags
        by_tag = {
            tag: {run.id for run in task.runs() if tag in run.tags}
            for tag in split_tags
        }
        assert len(by_tag["train_single_turn_spec"]) == 4
        assert len(by_tag["val_single_turn_spec"]) == 2
        reviewed_ids = {batch_runs[0].id, batch_runs[1].id}
        assert by_tag["golden_single_turn_spec"] == reviewed_ids
        assert by_tag["val_single_turn_spec"].isdisjoint(reviewed_ids)
        unreviewed_tags = set(runs_by_id[batch_runs[2].id].tags)
        assert "golden_single_turn_spec" not in unreviewed_tags

        # Reviewed runs carry golden ratings matching the review clicks, plus
        # feedback + per-claim grades; unreviewed runs stay unrated — REAL
        # runs fill train now, so no synthesized TaskRuns exist anywhere.
        rating_key = "named::Single Turn Spec"
        failed = runs_by_id[batch_runs[0].id]
        assert failed.output.rating.requirement_ratings[rating_key].value == 0.0
        assert (
            failed.output.rating.requirement_ratings[rating_key].type
            == TaskOutputRatingType.pass_fail
        )
        assert failed.feedback()[0].feedback == "Fabricated a return window."
        assert len(failed.claim_reviews()) == 1
        assert failed.claim_reviews()[0].judge_score == "fail"

        passed = runs_by_id[batch_runs[1].id]
        assert passed.output.rating.requirement_ratings[rating_key].value == 1.0
        assert passed.feedback() == []
        assert len(passed.claim_reviews()) == 1

        unreviewed = runs_by_id[batch_runs[2].id]
        assert unreviewed.output.rating is None
        assert unreviewed.claim_reviews() == []

        # Save-time generation is DEAD on this path: the task's runs are
        # exactly the eight the pipeline drove — nothing new was minted.
        assert len(task.runs()) == 8

    def test_single_turn_save_writes_splits_natively_to_disk(
        self, client, project_and_task, batch_runs, single_turn_request_data
    ):
        # Asserts on the SAVED BYTES, not the in-memory model, mirroring the
        # multi-turn saved-bytes test: the wire shape is the compatibility
        # contract older clients read.
        project, task = project_and_task
        response = self._post(client, project, task, single_turn_request_data)
        assert response.status_code == 200, response.text

        eval_path = task.evals()[0].path
        first_bytes = eval_path.read_text()
        on_disk = json.loads(first_bytes)

        assert on_disk["splits"] == {
            "test": {
                "source": "eval_input",
                "filter_id": "tag::test_single_turn_spec",
            },
            "train": {
                "source": "task_run",
                "filter_id": "tag::train_single_turn_spec",
            },
            "val": {
                "source": "task_run",
                "filter_id": "tag::val_single_turn_spec",
            },
        }
        assert on_disk["train_set_filter_id"] is None
        assert on_disk["eval_set_filter_id"] is None
        assert on_disk["eval_configs_filter_id"] == "tag::golden_single_turn_spec"
        assert "eval_input_filter_id" not in first_bytes

    def test_404_when_batch_tag_matches_nothing(
        self, client, project_and_task, single_turn_request_data
    ):
        project, task = project_and_task
        response = self._post(client, project, task, single_turn_request_data)
        assert response.status_code == 404
        assert "batch_tag" in response.json()["message"]
        assert len(task.evals()) == 0
        assert len(task.specs()) == 0

    def test_404_when_reviewed_run_not_in_batch(
        self, client, project_and_task, batch_runs, single_turn_request_data
    ):
        project, task = project_and_task
        single_turn_request_data["single_turn"]["reviewed_runs"] = [
            self._reviewed_run("no-such-run", meets_spec=True),
        ]
        response = self._post(client, project, task, single_turn_request_data)
        assert response.status_code == 404
        assert "no-such-run" in response.json()["message"]
        assert len(task.evals()) == 0

    def test_422_on_duplicate_reviewed_runs(
        self, client, project_and_task, batch_runs, single_turn_request_data
    ):
        project, task = project_and_task
        single_turn_request_data["single_turn"]["reviewed_runs"] = [
            self._reviewed_run(batch_runs[0].id, meets_spec=True),
            self._reviewed_run(batch_runs[0].id, meets_spec=False),
        ]
        response = self._post(client, project, task, single_turn_request_data)
        assert response.status_code == 422
        assert "at most once" in response.json()["message"]

    def test_validator_rejects_single_turn_without_full_trace(
        self, client, project_and_task, single_turn_request_data
    ):
        # Both wizard arms judge the transcript, so the saved eval must too —
        # otherwise the calibrated judge is not the judge that ships.
        project, task = project_and_task
        single_turn_request_data["evaluate_full_trace"] = False
        response = self._post(client, project, task, single_turn_request_data)
        assert response.status_code == 422
        assert "evaluate_full_trace" in str(response.json())

    def test_validator_rejects_blank_eval_slice_input(
        self, client, project_and_task, single_turn_request_data
    ):
        project, task = project_and_task
        single_turn_request_data["single_turn"]["inputs"][3] = "   "
        response = self._post(client, project, task, single_turn_request_data)
        assert response.status_code == 422

    def test_structured_task_rejects_non_schema_input(
        self, client, project_and_task, batch_runs, single_turn_request_data
    ):
        # A structured-input task's eval slice must parse against the input
        # schema, or the saved eval would fail every job at run time.
        project, task = project_and_task
        task.input_json_schema = json.dumps(
            {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            }
        )
        single_turn_request_data["single_turn"]["inputs"] = [
            json.dumps({"question": f"q {i}"}) for i in range(7)
        ] + ["not json"]
        response = self._post(client, project, task, single_turn_request_data)
        assert response.status_code == 422
        assert "not valid JSON" in response.json()["message"]
        assert len(task.evals()) == 0

    def test_save_failure_mid_rating_rolls_back(
        self, client, project_and_task, batch_runs, single_turn_request_data
    ):
        # A failure after the eval slice persisted, the runs were tagged, AND
        # the ratings were written must reverse EVERYTHING: created models
        # deleted, tags and ratings restored — the batch runs come out
        # exactly as the pipeline left them.
        project, task = project_and_task
        single_turn_request_data["single_turn"]["reviewed_runs"] = [
            self._reviewed_run(batch_runs[0].id, meets_spec=False),
        ]

        from app.desktop.studio_server.utils import copilot_utils

        def rate_then_boom(*args, **kwargs):
            copilot_utils.rate_reviewed_batch_runs(*args, **kwargs)
            raise RuntimeError("disk full")

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(
                "app.desktop.studio_server.copilot_api.rate_reviewed_batch_runs",
                side_effect=rate_then_boom,
            ),
            # The endpoint re-raises after rollback; TestClient propagates it.
            pytest.raises(RuntimeError, match="disk full"),
        ):
            client.post(
                f"/api/projects/{project.id}/tasks/{task.id}/spec_with_copilot",
                json=single_turn_request_data,
            )

        assert len(task.evals()) == 0
        assert len(task.specs()) == 0
        assert len(task.eval_inputs()) == 0
        for run in task.runs():
            assert set(run.tags) == {
                "single_turn_drive",
                f"single_turn_drive_batch:{self.BATCH_TAG}",
            }
            assert run.output.rating is None
            assert run.feedback() == []
            assert run.claim_reviews() == []


_JOBS_API = "app.desktop.studio_server.api_client.kiln_ai_server_client.api.jobs"
_START_FN = f"{_JOBS_API}.start_data_guide_job_v1_jobs_data_guide_job_start_post.asyncio_detailed"
_STATUS_FN = (
    f"{_JOBS_API}.get_job_status_v1_jobs_job_type_job_id_status_get.asyncio_detailed"
)
_RESULT_FN = f"{_JOBS_API}.get_data_guide_job_result_v1_jobs_data_guide_job_job_id_result_get.asyncio_detailed"


def _make_sdk_response(
    parsed=None,
    status_code: HTTPStatus = HTTPStatus.OK,
    content: bytes = b"{}",
) -> SdkResponse:
    return SdkResponse(
        status_code=status_code,
        content=content,
        headers={},
        parsed=parsed,
    )


class TestDataGuideJob:
    """Tests for the data guide draft job proxy endpoints.

    The draft runs as a kiln_server background job. The studio server proxies
    its lifecycle (via the generated kiln_ai_server_client) so the web UI owns
    polling:
      - POST .../copilot/data_guide_job/start → start_data_guide_job
      - GET  .../copilot/data_guide_job/{id}/status → get_job_status (DATA_GUIDE_JOB)
      - GET  .../copilot/data_guide_job/{id}/result → get_data_guide_job_result
    """

    START_URL = "/api/projects/proj_x/tasks/task_y/copilot/data_guide_job/start"
    STATUS_URL = (
        "/api/projects/proj_x/tasks/task_y/copilot/data_guide_job/job-123/status"
    )
    RESULT_URL = (
        "/api/projects/proj_x/tasks/task_y/copilot/data_guide_job/job-123/result"
    )

    DRAFT = (
        "# Semantics\n\n## Data Patterns\nShort greetings.\n\n"
        "# Style\n\n## Input-Level Metrics\nOne short sentence.\n\n"
        "# Presentation Defaults\n\nLowercase casual register.\n"
    )

    @staticmethod
    def _start_payload() -> dict:
        return {
            "input_examples": ["hello", "frog"],
        }

    @staticmethod
    def _make_task(instruction: str = "Translate the input to French.") -> Task:
        return Task(name="MockTask", instruction=instruction)

    def _patches(self, fn_path: str, sdk_mock: AsyncMock, *, task=None):
        """Patch the typed SDK endpoint plus the auth client and (for start)
        task lookup. The mocked endpoint never touches the auth client, so a
        bare MagicMock suffices."""
        return (
            patch(fn_path, sdk_mock),
            patch(
                "app.desktop.studio_server.copilot_api.get_authenticated_client",
                return_value=MagicMock(),
            ),
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task or self._make_task(),
            ),
        )

    # --- start --------------------------------------------------------------

    def test_start_no_api_key(self, client):
        with patch(
            "app.desktop.studio_server.utils.copilot_utils.Config.shared"
        ) as mock_config_shared:
            mock_config_shared.return_value.kiln_copilot_api_key = None
            response = client.post(self.START_URL, json=self._start_payload())
            assert response.status_code == 401
            assert "API key not configured" in response.json()["message"]

    def test_start_success(self, client, mock_api_key):
        sdk_mock = AsyncMock(
            return_value=_make_sdk_response(parsed=JobStartResponse(job_id="job-123"))
        )
        p1, p2, p3 = self._patches(
            _START_FN, sdk_mock, task=self._make_task("Translate the input to French.")
        )
        with p1, p2, p3:
            response = client.post(self.START_URL, json=self._start_payload())

        assert response.status_code == 200
        assert response.json() == {"job_id": "job-123"}
        # The typed request body carries the server-resolved runtime prompt
        # (task.instruction here, since this task has no default run config),
        # never a client-supplied prompt.
        body = sdk_mock.await_args.kwargs["body"]
        assert body.task_prompt == "Translate the input to French."
        assert body.input_examples == ["hello", "frog"]
        # Plaintext task → no input schema derived.
        assert not body.task_input_schema
        # The body model has no output schema or task description fields — output
        # policy must never reach the guide LLM.
        body_dict = body.to_dict()
        assert "task_output_schema" not in body_dict
        assert "task_description" not in body_dict

    def test_start_derives_input_schema_from_task(self, client, mock_api_key):
        """The input schema is read off the task server-side, not sent by the
        client (the payload has no schema field)."""
        schema = '{"type": "object", "properties": {"q": {"type": "string"}}}'
        structured_task = Task(
            name="MockTask",
            instruction="Answer the question.",
            input_json_schema=schema,
        )
        sdk_mock = AsyncMock(
            return_value=_make_sdk_response(parsed=JobStartResponse(job_id="job-123"))
        )
        p1, p2, p3 = self._patches(_START_FN, sdk_mock, task=structured_task)
        with p1, p2, p3:
            response = client.post(
                self.START_URL,
                json={"input_examples": ['{"q": "hello"}', '{"q": "frog"}']},
            )
        assert response.status_code == 200
        body = sdk_mock.await_args.kwargs["body"]
        assert body.task_input_schema == schema

    # --- structured-input validation ----------------------------------------

    STRUCTURED_SCHEMA = (
        '{"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}'
    )

    def _structured_task(self) -> Task:
        return Task(
            name="MockTask",
            instruction="Answer the question.",
            input_json_schema=self.STRUCTURED_SCHEMA,
        )

    def test_start_accepts_valid_structured_examples(self, client, mock_api_key):
        sdk_mock = AsyncMock(
            return_value=_make_sdk_response(parsed=JobStartResponse(job_id="job-123"))
        )
        p1, p2, p3 = self._patches(_START_FN, sdk_mock, task=self._structured_task())
        with p1, p2, p3:
            response = client.post(
                self.START_URL,
                json={"input_examples": ['{"q": "hello"}', '{"q": "frog"}']},
            )
        assert response.status_code == 200
        sdk_mock.assert_awaited_once()

    def test_start_rejects_structured_example_not_matching_schema(
        self, client, mock_api_key
    ):
        sdk_mock = AsyncMock()
        p1, p2, p3 = self._patches(_START_FN, sdk_mock, task=self._structured_task())
        with p1, p2, p3:
            response = client.post(
                self.START_URL,
                # second example is missing the required "q" field
                json={"input_examples": ['{"q": "ok"}', "{}"]},
            )
        assert response.status_code == 422
        assert "Example 2" in response.json()["message"]
        # validation must fail before the draft job is started
        sdk_mock.assert_not_awaited()

    def test_start_rejects_non_json_structured_example(self, client, mock_api_key):
        sdk_mock = AsyncMock()
        p1, p2, p3 = self._patches(_START_FN, sdk_mock, task=self._structured_task())
        with p1, p2, p3:
            response = client.post(
                self.START_URL,
                json={"input_examples": ["not valid json"]},
            )
        assert response.status_code == 422
        assert "not valid JSON" in response.json()["message"]
        sdk_mock.assert_not_awaited()

    def test_start_error_surfaces_message_field(self, client, mock_api_key):
        sdk_mock = AsyncMock(
            return_value=_make_sdk_response(
                status_code=HTTPStatus.PAYMENT_REQUIRED,
                content=b'{"message": "Your trial has expired."}',
            )
        )
        p1, p2, p3 = self._patches(_START_FN, sdk_mock)
        with p1, p2, p3:
            response = client.post(self.START_URL, json=self._start_payload())
        assert response.status_code == 402
        assert response.json()["message"] == "Your trial has expired."

    def test_start_error_non_json_falls_back_to_default(self, client, mock_api_key):
        sdk_mock = AsyncMock(
            return_value=_make_sdk_response(
                status_code=HTTPStatus.BAD_GATEWAY,
                content=b"upstream down",
            )
        )
        p1, p2, p3 = self._patches(_START_FN, sdk_mock)
        with p1, p2, p3:
            response = client.post(self.START_URL, json=self._start_payload())
        assert response.status_code == 502
        assert response.json()["message"] == (
            "Failed to start the data guide job. Please try again."
        )

    def test_start_missing_job_id_returns_500(self, client, mock_api_key):
        sdk_mock = AsyncMock(
            return_value=_make_sdk_response(parsed=JobStartResponse(job_id=""))
        )
        p1, p2, p3 = self._patches(_START_FN, sdk_mock)
        with p1, p2, p3:
            response = client.post(self.START_URL, json=self._start_payload())
        assert response.status_code == 500
        assert "job id" in response.json()["message"].lower()

    def test_start_kiln_server_unreachable_returns_502(self, client, mock_api_key):
        """A transport failure reaching kiln_server is an upstream problem, not
        ours: 502 with a message that says where to look, not a bare 500."""
        sdk_mock = AsyncMock(side_effect=httpx.ConnectError("boom"))
        p1, p2, p3 = self._patches(_START_FN, sdk_mock)
        with p1, p2, p3:
            response = client.post(self.START_URL, json=self._start_payload())
        assert response.status_code == 502
        assert "data guide" in response.json()["message"].lower()

    def test_start_kiln_server_404_becomes_502(self, client, mock_api_key):
        """Start names no resource, so an upstream 404 can only mean the route
        isn't deployed (an older kiln_server). Surfacing it as our own 404 would
        read as "this studio endpoint doesn't exist" — report 502 instead."""
        sdk_mock = AsyncMock(
            return_value=_make_sdk_response(
                status_code=HTTPStatus.NOT_FOUND,
                content=b'{"detail":"Not Found"}',
            )
        )
        p1, p2, p3 = self._patches(_START_FN, sdk_mock)
        with p1, p2, p3:
            response = client.post(self.START_URL, json=self._start_payload())
        assert response.status_code == 502
        assert "data guide" in response.json()["message"].lower()

    # --- status -------------------------------------------------------------

    def test_status_success(self, client, mock_api_key):
        sdk_mock = AsyncMock(
            return_value=_make_sdk_response(
                parsed=JobStatusResponse(job_id="job-123", status=JobStatus.RUNNING)
            )
        )
        p1, p2, p3 = self._patches(_STATUS_FN, sdk_mock)
        with p1, p2, p3:
            response = client.get(self.STATUS_URL)
        assert response.status_code == 200
        assert response.json() == {"status": "running"}
        # Status routes through the shared status endpoint keyed by job type.
        assert sdk_mock.await_args.kwargs["job_type"] == JobType.DATA_GUIDE_JOB
        assert sdk_mock.await_args.kwargs["job_id"] == "job-123"

    def test_status_error_surfaces(self, client, mock_api_key):
        sdk_mock = AsyncMock(
            return_value=_make_sdk_response(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR
            )
        )
        p1, p2, p3 = self._patches(_STATUS_FN, sdk_mock)
        with p1, p2, p3:
            response = client.get(self.STATUS_URL)
        assert response.status_code == 500
        assert response.json()["message"] == (
            "Failed to check the data guide job status."
        )

    def test_status_succeeded_held_until_result_ready(self, client, mock_api_key):
        """Status reads succeeded but the result endpoint still reports the job
        in-progress → hold at running so the UI stays in-progress instead of
        fetching an unfinished result."""
        status_mock = AsyncMock(
            return_value=_make_sdk_response(
                parsed=JobStatusResponse(job_id="job-123", status=JobStatus.SUCCEEDED)
            )
        )
        result_mock = AsyncMock(
            return_value=_make_sdk_response(
                parsed=DataGuideJobResultResponse(status=JobStatus.RUNNING, output=None)
            )
        )
        with (
            patch(_STATUS_FN, status_mock),
            patch(_RESULT_FN, result_mock),
            patch(
                "app.desktop.studio_server.copilot_api.get_authenticated_client",
                return_value=MagicMock(),
            ),
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=self._make_task(),
            ),
        ):
            response = client.get(self.STATUS_URL)
        assert response.status_code == 200
        assert response.json() == {"status": "running"}

    def test_status_succeeded_when_result_ready(self, client, mock_api_key):
        """When both status and result report succeeded, report succeeded."""
        status_mock = AsyncMock(
            return_value=_make_sdk_response(
                parsed=JobStatusResponse(job_id="job-123", status=JobStatus.SUCCEEDED)
            )
        )
        result_mock = AsyncMock(
            return_value=_make_sdk_response(
                parsed=DataGuideJobResultResponse(
                    status=JobStatus.SUCCEEDED,
                    output=DataGuideJobOutput(draft_guide="# Semantics"),
                )
            )
        )
        with (
            patch(_STATUS_FN, status_mock),
            patch(_RESULT_FN, result_mock),
            patch(
                "app.desktop.studio_server.copilot_api.get_authenticated_client",
                return_value=MagicMock(),
            ),
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=self._make_task(),
            ),
        ):
            response = client.get(self.STATUS_URL)
        assert response.status_code == 200
        assert response.json() == {"status": "succeeded"}

    # --- result -------------------------------------------------------------

    def test_result_success(self, client, mock_api_key):
        sdk_mock = AsyncMock(
            return_value=_make_sdk_response(
                parsed=DataGuideJobResultResponse(
                    status=JobStatus.SUCCEEDED,
                    output=DataGuideJobOutput(draft_guide=self.DRAFT),
                )
            )
        )
        p1, p2, p3 = self._patches(_RESULT_FN, sdk_mock)
        with p1, p2, p3:
            response = client.get(self.RESULT_URL)
        assert response.status_code == 200
        draft = response.json()["draft_guide"]
        # Copilot draft emits the canonical three-section guide shape.
        assert draft.startswith("# Semantics")
        assert "# Style" in draft
        assert "# Presentation Defaults" in draft
        assert "# Reference Inputs" not in draft
        assert sdk_mock.await_args.kwargs["job_id"] == "job-123"

    def test_result_empty_output_returns_500(self, client, mock_api_key):
        sdk_mock = AsyncMock(
            return_value=_make_sdk_response(
                parsed=DataGuideJobResultResponse(
                    status=JobStatus.SUCCEEDED, output=None
                )
            )
        )
        p1, p2, p3 = self._patches(_RESULT_FN, sdk_mock)
        with p1, p2, p3:
            response = client.get(self.RESULT_URL)
        assert response.status_code == 500
        assert "empty draft guide" in response.json()["message"].lower()

    def test_result_fetch_error_surfaces(self, client, mock_api_key):
        sdk_mock = AsyncMock(
            return_value=_make_sdk_response(status_code=HTTPStatus.BAD_GATEWAY)
        )
        p1, p2, p3 = self._patches(_RESULT_FN, sdk_mock)
        with p1, p2, p3:
            response = client.get(self.RESULT_URL)
        assert response.status_code == 502
        assert response.json()["message"] == ("Failed to fetch the data guide result.")

    def test_result_empty_draft_returns_500(self, client, mock_api_key):
        sdk_mock = AsyncMock(
            return_value=_make_sdk_response(
                parsed=DataGuideJobResultResponse(
                    status=JobStatus.SUCCEEDED,
                    output=DataGuideJobOutput(draft_guide="   "),
                )
            )
        )
        p1, p2, p3 = self._patches(_RESULT_FN, sdk_mock)
        with p1, p2, p3:
            response = client.get(self.RESULT_URL)
        assert response.status_code == 500
        assert "empty" in response.json()["message"].lower()

    def test_result_not_ready_returns_409(self, client, mock_api_key):
        """If the result endpoint returns before the job is actually succeeded
        (output not yet committed), surface a distinct "still generating" 409
        rather than mislabeling it as an empty draft."""
        sdk_mock = AsyncMock(
            return_value=_make_sdk_response(
                parsed=DataGuideJobResultResponse(status=JobStatus.RUNNING, output=None)
            )
        )
        p1, p2, p3 = self._patches(_RESULT_FN, sdk_mock)
        with p1, p2, p3:
            response = client.get(self.RESULT_URL)
        assert response.status_code == 409
        assert "still being generated" in response.json()["message"].lower()

    # --- job-addressed 404s stay 404s ---------------------------------------

    @pytest.mark.parametrize(
        "fn_path,url_attr",
        [(_STATUS_FN, "STATUS_URL"), (_RESULT_FN, "RESULT_URL")],
    )
    def test_job_addressed_404_is_propagated(
        self, client, mock_api_key, fn_path, url_attr
    ):
        """Unlike start, these requests name a job. An upstream 404 there really
        does mean "no such job", so it must reach the client as a 404 — not be
        rewritten into an upstream-failure 502."""
        sdk_mock = AsyncMock(
            return_value=_make_sdk_response(
                status_code=HTTPStatus.NOT_FOUND,
                content=b'{"message":"Job not found"}',
            )
        )
        p1, p2, p3 = self._patches(fn_path, sdk_mock)
        with p1, p2, p3:
            response = client.get(getattr(self, url_attr))
        assert response.status_code == 404
        assert response.json()["message"] == "Job not found"

    @pytest.mark.parametrize(
        "fn_path,url_attr",
        [(_STATUS_FN, "STATUS_URL"), (_RESULT_FN, "RESULT_URL")],
    )
    def test_job_kiln_server_unreachable_returns_502(
        self, client, mock_api_key, fn_path, url_attr
    ):
        sdk_mock = AsyncMock(side_effect=httpx.ConnectError("boom"))
        p1, p2, p3 = self._patches(fn_path, sdk_mock)
        with p1, p2, p3:
            response = client.get(getattr(self, url_attr))
        assert response.status_code == 502
        assert "data guide" in response.json()["message"].lower()


class TestParseImportFile:
    """Tests for the server-side bulk-import file parser.

    Plaintext tasks → single-column CSV (stdlib csv reader, single-column
    enforced). Structured tasks → one JSON object per line, schema-validated.
    """

    URL = "/api/projects/proj_x/tasks/task_y/copilot/parse_import_file"

    STRUCTURED_SCHEMA = (
        '{"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}'
    )

    def _post(self, client, content: bytes, name: str = "examples.csv"):
        return client.post(self.URL, files={"file": (name, content, "text/plain")})

    def _patch_task(self, task: Task):
        return patch(
            "app.desktop.studio_server.copilot_api.task_from_id",
            return_value=task,
        )

    def _plaintext_task(self) -> Task:
        return Task(name="MockTask", instruction="Echo the input.")

    def _structured_task(self) -> Task:
        return Task(
            name="MockTask",
            instruction="Answer.",
            input_json_schema=self.STRUCTURED_SCHEMA,
        )

    # --- plaintext ----------------------------------------------------------

    def test_plaintext_single_column(self, client):
        with self._patch_task(self._plaintext_task()):
            response = self._post(client, b"hello\nworld\n")
        assert response.status_code == 200
        body = response.json()
        assert body["rows"] == ["hello", "world"]
        assert body["error"] is None

    def test_plaintext_quoted_comma_and_newline_preserved(self, client):
        # A quoted field carries commas and newlines as a single value.
        content = b'"a, b\nc"\n"second"\n'
        with self._patch_task(self._plaintext_task()):
            response = self._post(client, content)
        assert response.status_code == 200
        assert response.json()["rows"] == ["a, b\nc", "second"]

    def test_plaintext_unescaped_comma_is_rejected(self, client):
        with self._patch_task(self._plaintext_task()):
            response = self._post(client, b"fine\nhas, an unescaped comma\n")
        assert response.status_code == 200
        body = response.json()
        assert body["rows"] == []
        assert body["error"] == "Invalid CSV format. Expected only one column."

    def test_plaintext_optional_input_header_dropped(self, client):
        with self._patch_task(self._plaintext_task()):
            response = self._post(client, b"input\nhello\nworld\n")
        assert response.status_code == 200
        assert response.json()["rows"] == ["hello", "world"]

    def test_plaintext_lone_quote_does_not_swallow_rows(self, client):
        # The stdlib parser keeps a mid-field quote literal — it must not merge
        # the following row (the bug the old hand-rolled parser had).
        with self._patch_task(self._plaintext_task()):
            response = self._post(client, b'5" nail\nnext row\n')
        assert response.status_code == 200
        assert response.json()["rows"] == ['5" nail', "next row"]

    def test_plaintext_empty_file(self, client):
        with self._patch_task(self._plaintext_task()):
            response = self._post(client, b"\n  \n")
        assert response.status_code == 200
        assert response.json()["error"] == "No examples found in the file."

    # --- structured (one JSON object per CSV cell) --------------------------

    MULTI_FIELD_SCHEMA = (
        '{"type": "object", "properties": {"q": {"type": "string"}, '
        '"n": {"type": "integer"}}, "required": ["q"]}'
    )

    def test_structured_single_field_cells(self, client):
        # A single-property JSON object has no comma, so it's a clean one-column
        # CSV without needing quotes.
        content = b'{"q": "hello"}\n{"q": "world"}\n'
        with self._patch_task(self._structured_task()):
            response = self._post(client, content, name="examples.csv")
        assert response.status_code == 200
        body = response.json()
        assert body["rows"] == ['{"q": "hello"}', '{"q": "world"}']
        assert body["error"] is None

    def test_structured_quoted_multi_field_cells(self, client):
        # Spreadsheet-export form: each JSON object is a quoted CSV cell with its
        # inner quotes doubled, so the JSON's commas don't split the columns.
        content = b'"{""q"": ""a"", ""n"": 1}"\n"{""q"": ""b"", ""n"": 2}"\n'
        task = Task(
            name="MockTask",
            instruction="Answer.",
            input_json_schema=self.MULTI_FIELD_SCHEMA,
        )
        with self._patch_task(task):
            response = self._post(client, content, name="examples.csv")
        assert response.status_code == 200
        assert response.json()["rows"] == ['{"q": "a", "n": 1}', '{"q": "b", "n": 2}']

    def test_structured_unquoted_multi_field_json_rejected(self, client):
        # Raw (unquoted) multi-field JSON: the comma splits it into columns, so
        # the single-column check rejects it and points the user at quoting.
        task = Task(
            name="MockTask",
            instruction="Answer.",
            input_json_schema=self.MULTI_FIELD_SCHEMA,
        )
        with self._patch_task(task):
            response = self._post(client, b'{"q": "a", "n": 1}\n', name="x.csv")
        assert response.status_code == 200
        assert (
            response.json()["error"] == "Invalid CSV format. Expected only one column."
        )

    def test_structured_non_json_cell(self, client):
        with self._patch_task(self._structured_task()):
            response = self._post(client, b'{"q": "ok"}\nnope\n', name="x.csv")
        assert response.status_code == 200
        assert response.json()["error"] == "Row 2 is not valid JSON."

    def test_structured_schema_mismatch(self, client):
        # Missing the required "q" field.
        with self._patch_task(self._structured_task()):
            response = self._post(client, b'{"q": "ok"}\n{}\n', name="x.csv")
        assert response.status_code == 200
        assert "Row 2 does not match the task input schema" in response.json()["error"]

    # --- encoding -----------------------------------------------------------

    def test_non_utf8_file_rejected(self, client):
        with self._patch_task(self._plaintext_task()):
            response = self._post(client, b"\xff\xfe invalid bytes")
        assert response.status_code == 422
        assert "UTF-8" in response.json()["message"]


def test_claim_review_api_requires_the_overall_call():
    """The request-model mirror of the persisted ClaimReview: the reviewer's
    overall call is what the golden rating is built from, so a payload
    without it is rejected before any model is written rather than defaulted
    to the judge's verdict."""
    import pydantic

    from app.desktop.studio_server.api_models.copilot_models import ClaimReviewApi

    with pytest.raises(pydantic.ValidationError, match="human_verdict"):
        ClaimReviewApi(
            judge_score="pass",
            judge_reasoning="Fine.",
            overview="Summary.",
            claims=[],
        )


@pytest.mark.parametrize("model_name", ["ReviewedChainApi", "ReviewedExample"])
def test_reviewed_item_rejects_a_rating_that_contradicts_its_review(model_name):
    """The golden rating and the stored review carry the same overall call;
    a payload where they differ is corrupt and 422s before anything is
    written."""
    import pydantic

    from app.desktop.studio_server.api_models import copilot_models

    model = getattr(copilot_models, model_name)
    base = (
        {"leaf_run_id": "run-1"}
        if model_name == "ReviewedChainApi"
        else {"input": "i", "output": "o", "model_says_meets_spec": True}
    )
    review = {
        "judge_score": "pass",
        "judge_reasoning": "Fine.",
        "overview": "Summary.",
        "claims": [],
        "human_verdict": "pass",
    }
    model(**base, user_says_meets_spec=True, feedback="", claim_review=review)
    with pytest.raises(pydantic.ValidationError, match="must match"):
        model(**base, user_says_meets_spec=False, feedback="", claim_review=review)


_QUESTION_SPEC_FN = (
    "app.desktop.studio_server.copilot_api."
    "question_spec_v1_copilot_question_spec_post.asyncio_detailed"
)
_CLARIFY_SPEC_FN = (
    "app.desktop.studio_server.copilot_api."
    "clarify_spec_v1_copilot_clarify_spec_post.asyncio_detailed"
)
_REFINE_SPEC_FN = (
    "app.desktop.studio_server.copilot_api."
    "refine_spec_v1_copilot_refine_spec_post.asyncio_detailed"
)


class TestPassthroughTaskCapabilities:
    """The passthrough routes attach the target task's capability surface when
    the caller names the task, and never leak the ids upstream."""

    QUESTION_SPEC_BODY: ClassVar[dict] = {
        "target_task_info": {
            "task_prompt": "Handle the support request.",
            "task_input_schema": "",
            "task_output_schema": "",
        },
        "target_specification": "The agent must never fabricate a refund.",
    }

    @pytest.fixture
    def capable_task(self, tmp_path, give_task_one_tool_and_skill):
        """A task whose default run config gives it one tool and one skill."""
        project = Project(name="Support", path=tmp_path / "project.kiln")
        project.save_to_file()
        task = Task(
            name="Support Agent",
            instruction="Handle the support request.",
            parent=project,
        )
        task.save_to_file()
        give_task_one_tool_and_skill(project, task)
        return project, task

    @staticmethod
    def _question_set_response():
        parsed = MagicMock(spec=QuestionSetServerApi)
        parsed.to_dict.return_value = {"questions": []}
        response = MagicMock()
        response.status_code = 200
        response.parsed = parsed
        return response

    def test_question_spec_wire_payload_snapshot(
        self, client, capable_task, mock_api_key
    ):
        """The exact bytes question_spec forwards for a task with a tool and a
        skill — capability names and descriptions only, ids stripped."""
        project, task = capable_task
        sdk_mock = AsyncMock(return_value=self._question_set_response())

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(_QUESTION_SPEC_FN, sdk_mock),
        ):
            response = client.post(
                "/api/copilot/question_spec",
                json={
                    **self.QUESTION_SPEC_BODY,
                    "project_id": str(project.id),
                    "task_id": str(task.id),
                },
            )

        assert response.status_code == 200
        assert sdk_mock.await_args.kwargs["body"].to_dict() == {
            "target_task_info": {
                "task_prompt": "Handle the support request.",
                "task_input_schema": "",
                "task_output_schema": "",
                "task_tools": [
                    {
                        "name": "add",
                        "description": "Add two numbers together and return the result",
                    }
                ],
                "task_skills": [
                    {
                        "name": "refund-policy",
                        "description": "How and when refunds are issued.",
                    }
                ],
            },
            "target_specification": "The agent must never fabricate a refund.",
        }

    def test_question_spec_without_ids_forwards_unchanged(self, client, mock_api_key):
        """A caller that names no task gets exactly the payload it always got:
        no capability keys at all, not explicit nulls."""
        sdk_mock = AsyncMock(return_value=self._question_set_response())

        with patch(_QUESTION_SPEC_FN, sdk_mock):
            response = client.post(
                "/api/copilot/question_spec", json=self.QUESTION_SPEC_BODY
            )

        assert response.status_code == 200
        assert sdk_mock.await_args.kwargs["body"].to_dict() == self.QUESTION_SPEC_BODY

    def test_question_spec_survives_unreadable_task_storage(
        self, client, capable_task, mock_api_key
    ):
        """Capability collection is best effort: a task whose storage cannot be
        read still gets its spec built, on the un-enriched payload."""
        project, task = capable_task
        sdk_mock = AsyncMock(return_value=self._question_set_response())

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch.object(
                type(task),
                "run_configs",
                side_effect=ValueError("corrupt run_config.kiln"),
            ),
            patch(_QUESTION_SPEC_FN, sdk_mock),
        ):
            response = client.post(
                "/api/copilot/question_spec",
                json={
                    **self.QUESTION_SPEC_BODY,
                    "project_id": str(project.id),
                    "task_id": str(task.id),
                },
            )

        assert response.status_code == 200
        assert sdk_mock.await_args.kwargs["body"].to_dict() == self.QUESTION_SPEC_BODY

    def test_question_spec_rejects_half_a_task_reference(self, client, mock_api_key):
        """One id without the other is a caller bug. Rejecting beats silently
        skipping enrichment, which would ship a prompt quietly missing the
        task's capabilities."""
        response = client.post(
            "/api/copilot/question_spec",
            json={**self.QUESTION_SPEC_BODY, "project_id": "project-1"},
        )

        assert response.status_code == 422
        assert "must be provided together" in response.json()["message"]

    @pytest.mark.parametrize(
        ("project_id", "task_id"),
        [("no-such-project", "no-such-task"), ("", "")],
        ids=["unknown_ids", "empty_ids"],
    )
    def test_question_spec_unresolvable_task_404s(
        self, client, mock_api_key, project_id, task_id
    ):
        """A bad id is fail-loud: the caller asked for this task's
        capabilities and the server cannot produce them. Empty strings are
        supplied ids too, so they 404 rather than quietly skipping
        enrichment."""
        sdk_mock = AsyncMock(return_value=self._question_set_response())

        with patch(_QUESTION_SPEC_FN, sdk_mock):
            response = client.post(
                "/api/copilot/question_spec",
                json={
                    **self.QUESTION_SPEC_BODY,
                    "project_id": project_id,
                    "task_id": task_id,
                },
            )

        assert response.status_code == 404
        sdk_mock.assert_not_awaited()

    def test_clarify_spec_attaches_capabilities_and_strips_ids(
        self, client, capable_task, mock_api_key, clarify_spec_input
    ):
        project, task = capable_task
        parsed = MagicMock(spec=ClarifySpecOutput)
        parsed.to_dict.return_value = {
            "examples_for_feedback": [],
            "judge_result": {
                "task_metadata": {
                    "model_name": "gpt-4",
                    "model_provider_name": "openai",
                },
                "prompt": "judge",
            },
            "sdg_session_config": {
                key: {
                    "task_metadata": {
                        "model_name": "gpt-4",
                        "model_provider_name": "openai",
                    },
                    "prompt": "Test prompt",
                }
                for key in (
                    "topic_generation_config",
                    "input_generation_config",
                    "output_generation_config",
                )
            },
        }
        sdk_response = MagicMock()
        sdk_response.status_code = 200
        sdk_response.parsed = parsed
        sdk_mock = AsyncMock(return_value=sdk_response)

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(_CLARIFY_SPEC_FN, sdk_mock),
        ):
            response = client.post(
                "/api/copilot/clarify_spec",
                json={
                    **clarify_spec_input,
                    "project_id": str(project.id),
                    "task_id": str(task.id),
                },
            )

        assert response.status_code == 200
        body = sdk_mock.await_args.kwargs["body"].to_dict()
        assert body["target_task_info"]["task_tools"] == [
            {
                "name": "add",
                "description": "Add two numbers together and return the result",
            }
        ]
        assert body["target_task_info"]["task_skills"] == [
            {"name": "refund-policy", "description": "How and when refunds are issued."}
        ]
        assert "project_id" not in body and "task_id" not in body

    def test_refine_spec_attaches_capabilities_and_strips_ids(
        self, client, capable_task, mock_api_key, refine_spec_input
    ):
        project, task = capable_task
        parsed = MagicMock(spec=RefineSpecApiOutput)
        parsed.to_dict.return_value = {
            "new_proposed_spec_edits": [],
            "not_incorporated_feedback": "",
        }
        sdk_response = MagicMock()
        sdk_response.status_code = 200
        sdk_response.parsed = parsed
        sdk_mock = AsyncMock(return_value=sdk_response)

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(_REFINE_SPEC_FN, sdk_mock),
        ):
            response = client.post(
                "/api/copilot/refine_spec",
                json={
                    **refine_spec_input,
                    "project_id": str(project.id),
                    "task_id": str(task.id),
                },
            )

        assert response.status_code == 200
        body = sdk_mock.await_args.kwargs["body"].to_dict()
        assert [t["name"] for t in body["target_task_info"]["task_tools"]] == ["add"]
        assert [s["name"] for s in body["target_task_info"]["task_skills"]] == [
            "refund-policy"
        ]
        assert "project_id" not in body and "task_id" not in body

    @pytest.fixture
    def second_run_config(self, capable_task, agent_run_config_properties):
        """A saved config on the same task that is NOT the default, giving the
        task `multiply` and no skills."""
        _, task = capable_task
        run_config = TaskRunConfig(
            name="other",
            run_config_properties=agent_run_config_properties(
                tools_config=ToolsRunConfig(tools=["kiln_tool::multiply_numbers"])
            ),
            parent=task,
        )
        run_config.save_to_file()
        return run_config

    def test_question_spec_reads_the_named_run_config(
        self, client, capable_task, second_run_config, mock_api_key
    ):
        """The eval is written about one run config, so the prompts must see
        that config's surface rather than the task's default. The id itself is
        studio-local and never reaches kiln_server."""
        project, task = capable_task
        sdk_mock = AsyncMock(return_value=self._question_set_response())

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(_QUESTION_SPEC_FN, sdk_mock),
        ):
            response = client.post(
                "/api/copilot/question_spec",
                json={
                    **self.QUESTION_SPEC_BODY,
                    "project_id": str(project.id),
                    "task_id": str(task.id),
                    "run_config_id": str(second_run_config.id),
                },
            )

        assert response.status_code == 200
        body = sdk_mock.await_args.kwargs["body"].to_dict()
        assert body["target_task_info"]["task_tools"] == [
            {
                "name": "multiply",
                "description": "Multiply two numbers together and return the result",
            }
        ]
        assert body["target_task_info"]["task_skills"] == []
        assert "run_config_id" not in body

    def test_question_spec_unresolvable_run_config_404s(
        self, client, capable_task, mock_api_key
    ):
        """Fail loud rather than quietly describing the default config, which
        is not the one the eval is being written against."""
        project, task = capable_task
        sdk_mock = AsyncMock(return_value=self._question_set_response())

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(_QUESTION_SPEC_FN, sdk_mock),
        ):
            response = client.post(
                "/api/copilot/question_spec",
                json={
                    **self.QUESTION_SPEC_BODY,
                    "project_id": str(project.id),
                    "task_id": str(task.id),
                    "run_config_id": "no-such-config",
                },
            )

        assert response.status_code == 404
        sdk_mock.assert_not_awaited()

    def test_question_spec_rejects_a_run_config_without_its_task(
        self, client, mock_api_key
    ):
        """A run config is only looked up once the task is, so an id sent
        alone would be dropped and the prompt would describe the wrong
        config."""
        response = client.post(
            "/api/copilot/question_spec",
            json={**self.QUESTION_SPEC_BODY, "run_config_id": "rc-1"},
        )

        assert response.status_code == 422
        assert "run_config_id requires" in response.json()["message"]

    def test_client_sent_capabilities_are_not_overwritten(
        self, client, capable_task, mock_api_key
    ):
        """A caller that already collected the surface keeps its own answer."""
        project, task = capable_task
        sdk_mock = AsyncMock(return_value=self._question_set_response())

        with (
            patch(
                "app.desktop.studio_server.copilot_api.task_from_id",
                return_value=task,
            ),
            patch(_QUESTION_SPEC_FN, sdk_mock),
        ):
            response = client.post(
                "/api/copilot/question_spec",
                json={
                    "target_task_info": {
                        **self.QUESTION_SPEC_BODY["target_task_info"],
                        "task_tools": [],
                        "task_skills": [],
                    },
                    "target_specification": self.QUESTION_SPEC_BODY[
                        "target_specification"
                    ],
                    "project_id": str(project.id),
                    "task_id": str(task.id),
                },
            )

        assert response.status_code == 200
        task_info = sdk_mock.await_args.kwargs["body"].to_dict()["target_task_info"]
        assert task_info["task_tools"] == []
        assert task_info["task_skills"] == []
