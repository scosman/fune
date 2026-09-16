import asyncio
import json
import logging
import re
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import litellm
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from kiln_ai.adapters.errors import KilnRunError
from kiln_ai.datamodel import Project, Task
from kiln_ai.datamodel.datamodel_enums import (
    ModelProviderName,
    StructuredOutputMode,
    TaskOutputRatingType,
    TurnMode,
)
from kiln_ai.datamodel.eval import (
    EvalConfigType,
    EvalDataType,
    LlmJudgeProperties,
    SkippedReason,
    V2EvalResult,
)
from kiln_ai.datamodel.run_config import (
    KilnAgentRunConfigProperties,
    ToolsRunConfig,
)
from kiln_ai.synthetic_user.runner import NUM_CASES_MAX
from kiln_ai.utils.async_job_runner import RETRY_BACKOFF_FACTOR
from kiln_server.custom_errors import connect_custom_errors
from pydantic import ValidationError

from app.desktop.studio_server.api_client.kiln_ai_server_client.models.build_claim_evidence_output import (
    BuildClaimEvidenceOutput,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models.generate_judge_prompt_output import (
    GenerateJudgePromptOutput,
)
from app.desktop.studio_server.api_client.kiln_ai_server_client.models.refine_judge_prompt_output import (
    RefineJudgePromptOutput,
)
from app.desktop.studio_server.api_models.copilot_models import (
    TaskSkillInfoApi,
    TaskToolInfoApi,
)
from app.desktop.studio_server.api_models.eval_builder_models import (
    BuildClaimsApiOutput,
    CitationApi,
    ClaimApi,
    JudgeConfig,
    OverviewApi,
)
from app.desktop.studio_server.eval_builder_api import (
    JUDGE_MAX_RETRIES,
    JUDGE_RETRY_DELAY_SECONDS,
    SingleTurnPipelineRequest,
    connect_eval_builder_api,
    run_judge_with_retry,
)
from app.desktop.studio_server.utils.eval_builder_utils import (
    JudgeVerdict,
    build_judge_prompt_template,
    build_transient_judge_eval_config,
    run_judge_for_trace,
)

BUILD_CLAIMS_URL = "/api/projects/p1/tasks/t1/eval_builder/build_claims"


@pytest.fixture
def app():
    app = FastAPI()
    connect_custom_errors(app)
    connect_eval_builder_api(app)
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


def _parse_sse(response_text: str) -> list[dict | str]:
    events: list[dict | str] = []
    for line in response_text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        events.append("complete" if payload == "complete" else json.loads(payload))
    return events


def _citation(to: str = "purchase") -> CitationApi:
    return CitationApi.model_validate(
        {"marker": 1, "source": "output", "from": "30 days", "to": to}
    )


def _overview() -> OverviewApi:
    return OverviewApi(
        text="The user asked about returning opened electronics and the "
        "agent quoted a 30-day window [1].",
        citations=[_citation()],
    )


def _claim_with_citation() -> ClaimApi:
    return ClaimApi(
        text="The agent stated a specific 30-day return window as fact [1]. "
        "Disagree if the window is documented policy.",
        citations=[_citation()],
        is_verdict=False,
    )


def _verdict_claim() -> ClaimApi:
    return ClaimApi(
        text="It fails because the agent asserted a return window it never "
        "verified [1].",
        citations=[_citation("full refund")],
        is_verdict=True,
    )


def _claims_output(claims: list[ClaimApi] | None = None) -> BuildClaimsApiOutput:
    return BuildClaimsApiOutput(
        overview=_overview(),
        claims=claims
        if claims is not None
        else [_claim_with_citation(), _verdict_claim()],
    )


# ───────────────────────── run_judge_for_trace ─────────────────────────


@pytest.fixture
def judge_config():
    return JudgeConfig(
        prompt="Judge whether the output fabricates policy.",
        model_name="claude_sonnet_4_6",
        model_provider="anthropic",
    )


@pytest.fixture
def in_memory_task():
    return Task(
        name="Test Task",
        instruction="Answer customer questions about return policy.",
        parent=Project(name="Test Project"),
    )


def _judge_adapter(result: V2EvalResult) -> MagicMock:
    adapter = MagicMock()
    adapter.evaluate = AsyncMock(return_value=result)
    return adapter


def _patch_judge_seam(task, adapter):
    """Patch the two SDK touchpoints run_judge_for_trace uses: task loading and
    the V2 adapter registry. Returns the (task_from_id, registry) patchers."""
    return (
        patch(
            "app.desktop.studio_server.utils.eval_builder_utils.task_from_id",
            return_value=task,
        ),
        patch(
            "app.desktop.studio_server.utils.eval_builder_utils.v2_eval_adapter_from_config",
            return_value=adapter,
        ),
    )


class TestBuildJudgePromptTemplate:
    def test_single_turn_uses_io_blocks(self):
        template = build_judge_prompt_template("Check the policy.", multi_turn=False)
        assert "Check the policy." in template
        assert "{{ task_input }}" in template
        assert "{{ final_message }}" in template
        assert "trace" not in template

    def test_multi_turn_uses_canonical_transcript_block(self):
        template = build_judge_prompt_template("Check the policy.", multi_turn=True)
        assert "Check the policy." in template
        # format_trace = the shared canonical rendering (EvalTraceFormatter),
        # the same text the claim builder receives as raw_output.
        assert "{{ trace | format_trace }}" in template
        assert "{{ final_message }}" not in template

    def test_jinja_in_judge_prompt_is_raw_wrapped(self):
        # Spec text with Jinja syntax must not break rendering or inject template
        # code — it gets wrapped in {% raw %} and survives as a literal.
        template = build_judge_prompt_template(
            "Spec says: {{ never_render_this }}", multi_turn=False
        )
        assert "{% raw %}" in template
        assert "{{ never_render_this }}" in template

    def test_plain_prompt_is_not_wrapped(self):
        template = build_judge_prompt_template("No jinja here.", multi_turn=False)
        assert "{% raw %}" not in template


class TestBuildTransientJudgeEvalConfig:
    def test_single_turn_config_shape(self, in_memory_task, judge_config):
        config = build_transient_judge_eval_config(
            in_memory_task, judge_config, multi_turn=False
        )
        assert config.config_type == EvalConfigType.v2
        properties = config.properties
        assert isinstance(properties, LlmJudgeProperties)
        assert properties.model_name == "claude_sonnet_4_6"
        assert properties.model_provider == "anthropic"
        assert "Judge whether the output fabricates policy." in (
            properties.prompt_template
        )

        eval_obj = config.parent_eval()
        assert eval_obj is not None
        assert eval_obj.evaluation_data_type == EvalDataType.final_answer
        assert len(eval_obj.output_scores) == 1
        assert eval_obj.output_scores[0].type == TaskOutputRatingType.pass_fail
        # The CONSTANT draft score: the eval's name is a save-time identity
        # the wizard keeps out of the pre-save flow, so the transient judge
        # never sees it — renames stay free until save.
        assert eval_obj.output_scores[0].name == "Meets Spec"
        assert eval_obj.output_scores[0].json_key() == "meets_spec"
        assert "Test Spec" not in eval_obj.output_scores[0].instruction
        assert eval_obj.parent_task() is in_memory_task

    def test_multi_turn_scores_full_trace(self, in_memory_task, judge_config):
        config = build_transient_judge_eval_config(
            in_memory_task, judge_config, multi_turn=True
        )
        eval_obj = config.parent_eval()
        assert eval_obj is not None
        assert eval_obj.evaluation_data_type == EvalDataType.full_trace
        properties = config.properties
        assert isinstance(properties, LlmJudgeProperties)
        assert "{{ trace | format_trace }}" in properties.prompt_template


class TestRunJudgeForTrace:
    @pytest.mark.asyncio
    async def test_pass_verdict_with_reasoning(self, in_memory_task, judge_config):
        adapter = _judge_adapter(
            V2EvalResult(
                scores={"meets_spec": 1.0},
                intermediate_outputs={"reasoning": "The reply follows the policy."},
            )
        )
        task_patch, registry_patch = _patch_judge_seam(in_memory_task, adapter)
        with task_patch, registry_patch:
            verdict = await run_judge_for_trace("p1", "t1", "in", "out", judge_config)

        assert verdict.judge_score == "pass"
        assert verdict.judge_reasoning == "The reply follows the policy."

    @pytest.mark.asyncio
    async def test_fail_verdict_falls_back_when_no_reasoning(
        self, in_memory_task, judge_config
    ):
        adapter = _judge_adapter(V2EvalResult(scores={"meets_spec": 0.0}))
        task_patch, registry_patch = _patch_judge_seam(in_memory_task, adapter)
        with task_patch, registry_patch:
            verdict = await run_judge_for_trace("p1", "t1", "in", "out", judge_config)

        assert verdict.judge_score == "fail"
        assert "FAIL" in verdict.judge_reasoning  # honest placeholder, not fabricated

    @pytest.mark.asyncio
    async def test_chain_of_thought_reasoning_fallback(
        self, in_memory_task, judge_config
    ):
        adapter = _judge_adapter(
            V2EvalResult(
                scores={"meets_spec": 1.0},
                intermediate_outputs={"chain_of_thought": "Step by step it holds."},
            )
        )
        task_patch, registry_patch = _patch_judge_seam(in_memory_task, adapter)
        with task_patch, registry_patch:
            verdict = await run_judge_for_trace("p1", "t1", "in", "out", judge_config)

        assert verdict.judge_reasoning == "Step by step it holds."

    @pytest.mark.asyncio
    async def test_multi_turn_passes_trace_and_final_message(
        self, in_memory_task, judge_config
    ):
        adapter = _judge_adapter(V2EvalResult(scores={"meets_spec": 1.0}))
        trace = [
            {"role": "user", "content": "Can I return opened items?"},
            {"role": "assistant", "content": "Let me check the policy."},
            {"role": "user", "content": "Please do."},
            {"role": "assistant", "content": "Yes, within 30 days."},
        ]
        task_patch, registry_patch = _patch_judge_seam(in_memory_task, adapter)
        with task_patch, registry_patch as mock_registry:
            await run_judge_for_trace(
                "p1",
                "t1",
                "in",
                "flattened transcript",
                judge_config,
                trace=trace,
            )

        eval_input = adapter.evaluate.call_args.args[0]
        assert eval_input.trace == trace
        # final_message is the closing assistant message, not the flat transcript
        assert eval_input.final_message == "Yes, within 30 days."
        config = mock_registry.call_args.args[0]
        parent_eval = config.parent_eval()
        assert parent_eval is not None
        assert parent_eval.evaluation_data_type == EvalDataType.full_trace

    @pytest.mark.asyncio
    async def test_skip_raises_instead_of_fake_verdict(
        self, in_memory_task, judge_config
    ):
        adapter = _judge_adapter(
            V2EvalResult(
                skipped_reason=SkippedReason.extraction_failed,
                skipped_detail="Template rendering failed",
            )
        )
        task_patch, registry_patch = _patch_judge_seam(in_memory_task, adapter)
        with task_patch, registry_patch:
            with pytest.raises(ValueError, match="Judge skipped this trace"):
                await run_judge_for_trace("p1", "t1", "in", "out", judge_config)

    @pytest.mark.asyncio
    async def test_missing_score_raises(self, in_memory_task, judge_config):
        adapter = _judge_adapter(V2EvalResult(scores={}))
        task_patch, registry_patch = _patch_judge_seam(in_memory_task, adapter)
        with task_patch, registry_patch:
            with pytest.raises(ValueError, match="no score"):
                await run_judge_for_trace("p1", "t1", "in", "out", judge_config)


# ───────────────────────── build_claims primitive ─────────────────────────


@pytest.fixture
def build_claims_input():
    return {
        "raw_input": "What's your return window for opened electronics?",
        "raw_output": (
            "Our return window is 30 days from purchase, even for opened "
            "electronics, and you'll get a full refund."
        ),
        "eval_rubric": "The agent must not fabricate or guess at company policies.",
        "judge_reasoning": "Stated a concrete return window as fact without verifying.",
        "judge_score": "fail",
    }


@pytest.fixture
def build_claims_task():
    """The task the endpoint resolves for its instruction (the URL's ids name
    no real task)."""
    with patch(
        "app.desktop.studio_server.eval_builder_api.task_from_id",
        return_value=Mock(instruction="Answer questions about return policy."),
    ) as task_from_id_mock:
        yield task_from_id_mock


def _sdk_card(claim_texts: list[str]) -> dict:
    """What the SDK's to_dict() hands back: the wire card, citations under the
    `from` key, no verdict flag (the studio adds it)."""
    citation = {"marker": 1, "source": "output", "from": "30 days", "to": "purchase"}
    return {
        "overview": {
            "text": "The agent quoted a 30-day window [1].",
            "citations": [citation],
        },
        "claims": [{"text": text, "citations": [citation]} for text in claim_texts],
    }


def _sdk_response(card: dict) -> MagicMock:
    mock_output = MagicMock(spec=BuildClaimEvidenceOutput)
    mock_output.to_dict.return_value = card
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.parsed = mock_output
    return mock_response


class TestBuildClaims:
    def test_build_claims_no_api_key(
        self, client, build_claims_input, build_claims_task
    ):
        with patch(
            "app.desktop.studio_server.utils.copilot_utils.Config.shared"
        ) as mock_config_shared:
            mock_config = mock_config_shared.return_value
            mock_config.kiln_copilot_api_key = None
            response = client.post(BUILD_CLAIMS_URL, json=build_claims_input)
            assert response.status_code == 401
            assert "API key not configured" in response.json()["message"]

    def test_build_claims_success(
        self, client, build_claims_input, mock_api_key, build_claims_task
    ):
        card = _sdk_card(
            [
                "The agent stated a 30-day window as fact [1].",
                "It fails because the window was never verified [1].",
            ]
        )
        with patch(
            "app.desktop.studio_server.utils.eval_builder_utils.build_claim_evidence_v1_copilot_build_claim_evidence_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=_sdk_response(card),
        ) as sdk_call:
            response = client.post(BUILD_CLAIMS_URL, json=build_claims_input)
            assert response.status_code == 200
            result = response.json()

            # The task's own instruction rides to the builder as context; the
            # client never sends it.
            body = sdk_call.call_args.kwargs["body"]
            assert body.task_instruction == "Answer questions about return policy."
            assert body.raw_input == build_claims_input["raw_input"]

            assert result["overview"]["text"] == card["overview"]["text"]
            assert [c["text"] for c in result["claims"]] == [
                c["text"] for c in card["claims"]
            ]
            # The verdict flag is the studio's: the last claim opens with the
            # verdict phrasing, the first does not.
            assert [c["is_verdict"] for c in result["claims"]] == [False, True]

            # The regression that matters: the serialized citation key must be
            # `from` on the overview AND on every claim.
            for entry in [result["overview"], *result["claims"]]:
                citation = entry["citations"][0]
                assert "from" in citation and "from_" not in citation
                assert citation["from"] == "30 days"
                assert citation["source"] == "output"

    @pytest.mark.parametrize(
        "claim_texts,expected_flags",
        [
            # Verdict phrasing on a non-last claim never flags it; a last claim
            # without the phrasing is an ordinary claim (the builder omitted
            # the verdict).
            (
                ["It fails because of the window [1].", "The tone was polite [1]."],
                [False, False],
            ),
            # Only the last claim is checked, and leading whitespace does not
            # hide the opener.
            (
                ["The tone was polite [1].", "  It passes despite the window [1]."],
                [False, True],
            ),
            # A one-claim card whose only claim is the verdict.
            (["It passes [1]."], [True]),
        ],
    )
    def test_build_claims_flags_only_the_last_verdict_claim(
        self,
        client,
        build_claims_input,
        mock_api_key,
        build_claims_task,
        claim_texts,
        expected_flags,
    ):
        with patch(
            "app.desktop.studio_server.utils.eval_builder_utils.build_claim_evidence_v1_copilot_build_claim_evidence_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=_sdk_response(_sdk_card(claim_texts)),
        ):
            response = client.post(BUILD_CLAIMS_URL, json=build_claims_input)
            assert response.status_code == 200
            flags = [c["is_verdict"] for c in response.json()["claims"]]
            assert flags == expected_flags

    def test_build_claims_no_response(
        self, client, build_claims_input, mock_api_key, build_claims_task
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = None

        with patch(
            "app.desktop.studio_server.utils.eval_builder_utils.build_claim_evidence_v1_copilot_build_claim_evidence_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(BUILD_CLAIMS_URL, json=build_claims_input)
            assert response.status_code == 500
            assert "Failed to build claims" in response.json()["message"]

    def test_build_claims_validation_error(
        self, client, build_claims_input, mock_api_key, build_claims_task
    ):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.content = b'{"message": "Validation error from server"}'
        mock_response.parsed = None

        with patch(
            "app.desktop.studio_server.utils.eval_builder_utils.build_claim_evidence_v1_copilot_build_claim_evidence_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(BUILD_CLAIMS_URL, json=build_claims_input)
            assert response.status_code == 422
            assert "Validation error from server" in response.json()["message"]


# ───────────────────────── author_judge ──────────────────────────────────

AUTHOR_JUDGE_URL = "/api/projects/p1/tasks/t1/eval_builder/author_judge"


def _task_mock(turn_mode=None, input_json_schema=None):
    """A Task mock for route tests. turn_mode defaults to multiturn."""
    from kiln_ai.datamodel.datamodel_enums import TurnMode
    from kiln_ai.datamodel.task import Task as KilnTask

    task = Mock(spec=KilnTask)
    # spec= is built from the class, so pydantic fields aren't auto-mocked.
    task.id = "task-1"
    task.name = "support_agent"
    task.instruction = "You are a customer support agent."
    task.turn_mode = turn_mode if turn_mode is not None else TurnMode.multiturn
    task.input_json_schema = input_json_schema
    # No default run config, so capability collection yields nothing and routes
    # that read it behave as they did before capabilities existed. The empty
    # config list is what a caller-named config is looked up in.
    task.default_run_config_id = None
    task.run_configs.return_value = []
    return task


@pytest.fixture
def author_judge_input():
    return {
        "target_specification": "The agent must never fabricate information.",
        "target_task_prompt": "You are a customer support agent.",
    }


@pytest.fixture
def author_judge_task():
    """The route loads the task for its capability surface — resolve it to a
    multi-turn mock unless a test overrides the return value."""
    with patch(
        "app.desktop.studio_server.eval_builder_api.task_from_id",
        return_value=_task_mock(),
    ) as mock_task:
        yield mock_task


class TestAuthorJudge:
    def test_author_judge_no_api_key(self, client, author_judge_input):
        """Fail-fast: a keyless caller gets a clean 401 before the remote call."""
        with patch(
            "app.desktop.studio_server.utils.copilot_utils.Config.shared"
        ) as mock_config_shared:
            mock_config = mock_config_shared.return_value
            mock_config.kiln_copilot_api_key = None
            response = client.post(AUTHOR_JUDGE_URL, json=author_judge_input)
            assert response.status_code == 401
            assert "API key not configured" in response.json()["message"]

    def test_author_judge_success(
        self, client, author_judge_input, mock_api_key, author_judge_task
    ):
        mock_output = MagicMock(spec=GenerateJudgePromptOutput)
        mock_output.judge_evaluation_prompt = (
            "1. When the assistant states a specific order fact, check whether "
            "a preceding lookup returned it — fabrication fails."
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = mock_output

        with patch(
            "app.desktop.studio_server.utils.eval_builder_utils.generate_judge_prompt_v1_copilot_generate_judge_prompt_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(AUTHOR_JUDGE_URL, json=author_judge_input)
            assert response.status_code == 200
            assert "fabrication fails" in response.json()["judge_prompt"]

    @pytest.mark.parametrize("turn_mode", [TurnMode.multiturn, TurnMode.single_turn])
    def test_author_judge_authors_against_the_transcript_for_both_arms(
        self, client, author_judge_input, mock_api_key, author_judge_task, turn_mode
    ):
        """Both arms judge a transcript, so both must author the rubric that
        knows what one looks like. The rubric routing on kiln_server hangs
        entirely on this field: sending single_turn would author against a
        bare input/output pair, and the judge would then meet role labels and
        tool-call blocks its rubric never mentioned."""
        author_judge_task.return_value = _task_mock(turn_mode)
        mock_output = MagicMock(spec=GenerateJudgePromptOutput)
        mock_output.judge_evaluation_prompt = "1. Check the transcript."
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = mock_output

        with patch(
            "app.desktop.studio_server.utils.eval_builder_utils.generate_judge_prompt_v1_copilot_generate_judge_prompt_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            client.post(AUTHOR_JUDGE_URL, json=author_judge_input)

        body = mock_post.call_args.kwargs["body"]
        assert body.trace_type.value == "multi_turn"
        assert body.target_specification == author_judge_input["target_specification"]

    def test_author_judge_omits_uncollected_capabilities(
        self, client, author_judge_input, mock_api_key, author_judge_task
    ):
        """A task with no capability surface to report leaves the keys out
        entirely — the authored prompt stays exactly what it was before the
        fields existed, rather than being told the task has no tools."""
        mock_output = MagicMock(spec=GenerateJudgePromptOutput)
        mock_output.judge_evaluation_prompt = "1. Check the transcript."
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = mock_output

        with patch(
            "app.desktop.studio_server.utils.eval_builder_utils.generate_judge_prompt_v1_copilot_generate_judge_prompt_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            client.post(AUTHOR_JUDGE_URL, json=author_judge_input)

        body_dict = mock_post.call_args.kwargs["body"].to_dict()
        assert "task_tools" not in body_dict
        assert "task_skills" not in body_dict

    def test_author_judge_sends_the_tasks_capabilities(
        self, client, author_judge_input, mock_api_key, author_judge_task
    ):
        """The rubric can only grade tool and skill use if the payload names
        them, flat on this input (it has no task info block)."""
        mock_output = MagicMock(spec=GenerateJudgePromptOutput)
        mock_output.judge_evaluation_prompt = "1. Check the transcript."
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = mock_output

        with (
            patch(
                "app.desktop.studio_server.eval_builder_api.task_capabilities_for_task",
                new_callable=AsyncMock,
                return_value=(
                    [
                        TaskToolInfoApi(
                            name="lookup_order", description="Find an order."
                        )
                    ],
                    [TaskSkillInfoApi(name="refund-policy", description="Refunds.")],
                ),
            ),
            patch(
                "app.desktop.studio_server.utils.eval_builder_utils.generate_judge_prompt_v1_copilot_generate_judge_prompt_post.asyncio_detailed",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_post,
        ):
            client.post(AUTHOR_JUDGE_URL, json=author_judge_input)

        body_dict = mock_post.call_args.kwargs["body"].to_dict()
        assert body_dict["task_tools"] == [
            {"name": "lookup_order", "description": "Find an order."}
        ]
        assert body_dict["task_skills"] == [
            {"name": "refund-policy", "description": "Refunds."}
        ]

    @pytest.mark.parametrize(
        ("extra_body", "expected_run_config_id"),
        [({"run_config_id": "rc-7"}, "rc-7"), ({}, None)],
        ids=["named_config", "no_config"],
    )
    def test_author_judge_reads_the_run_config_the_caller_named(
        self,
        client,
        author_judge_input,
        mock_api_key,
        author_judge_task,
        extra_body,
        expected_run_config_id,
    ):
        """The rubric grades the config the eval is written against, so the id
        the caller sends is the one the capability read must use. No id means
        the task default, exactly as before the caller could choose."""
        mock_output = MagicMock(spec=GenerateJudgePromptOutput)
        mock_output.judge_evaluation_prompt = "1. Check the transcript."
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = mock_output

        with (
            patch(
                "app.desktop.studio_server.eval_builder_api.task_capabilities_for_task",
                new_callable=AsyncMock,
                return_value=([], []),
            ) as mock_capabilities,
            patch(
                "app.desktop.studio_server.utils.eval_builder_utils.generate_judge_prompt_v1_copilot_generate_judge_prompt_post.asyncio_detailed",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            response = client.post(
                AUTHOR_JUDGE_URL, json={**author_judge_input, **extra_body}
            )

        assert response.status_code == 200
        assert mock_capabilities.await_args.args[1] == expected_run_config_id

    def test_author_judge_unresolvable_run_config_404s(
        self, client, author_judge_input, mock_api_key, author_judge_task
    ):
        """An id that names no config on the task stops the request rather
        than authoring a rubric against the default config's surface."""
        with patch(
            "app.desktop.studio_server.utils.eval_builder_utils.generate_judge_prompt_v1_copilot_generate_judge_prompt_post.asyncio_detailed",
            new_callable=AsyncMock,
        ) as mock_post:
            response = client.post(
                AUTHOR_JUDGE_URL,
                json={**author_judge_input, "run_config_id": "no-such-config"},
            )

        assert response.status_code == 404
        mock_post.assert_not_awaited()

    def test_author_judge_reports_a_task_with_no_capabilities(
        self, client, author_judge_input, mock_api_key, author_judge_task
    ):
        """[] is a real answer worth sending: the task genuinely has none, so
        the rubric should not invent tool-use criteria."""
        mock_output = MagicMock(spec=GenerateJudgePromptOutput)
        mock_output.judge_evaluation_prompt = "1. Check the transcript."
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = mock_output

        with (
            patch(
                "app.desktop.studio_server.eval_builder_api.task_capabilities_for_task",
                new_callable=AsyncMock,
                return_value=([], []),
            ),
            patch(
                "app.desktop.studio_server.utils.eval_builder_utils.generate_judge_prompt_v1_copilot_generate_judge_prompt_post.asyncio_detailed",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_post,
        ):
            client.post(AUTHOR_JUDGE_URL, json=author_judge_input)

        body_dict = mock_post.call_args.kwargs["body"].to_dict()
        assert body_dict["task_tools"] == []
        assert body_dict["task_skills"] == []

    def test_author_judge_remote_error_surfaces_upstream_message(
        self, client, author_judge_input, mock_api_key, author_judge_task
    ):
        """A remote failure propagates the upstream status + message — the
        client stops the drive on it (authoring is required, no fallback
        judge), so the detail must survive to be shown."""
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.content = b'{"message": "upstream refused"}'
        mock_response.parsed = None

        with patch(
            "app.desktop.studio_server.utils.eval_builder_utils.generate_judge_prompt_v1_copilot_generate_judge_prompt_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(AUTHOR_JUDGE_URL, json=author_judge_input)
            assert response.status_code == 502
            assert "upstream refused" in response.json()["message"]

    def test_author_judge_no_response_is_500(
        self, client, author_judge_input, mock_api_key, author_judge_task
    ):
        """A 2xx with no parsed body surfaces as a 500 with a clear message."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = None

        with patch(
            "app.desktop.studio_server.utils.eval_builder_utils.generate_judge_prompt_v1_copilot_generate_judge_prompt_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(AUTHOR_JUDGE_URL, json=author_judge_input)
            assert response.status_code == 500
            assert "Failed to author the judge prompt" in response.json()["message"]

    def test_author_judge_rejects_empty_spec(self, client, mock_api_key):
        """target_specification must be non-empty (min_length=1) — a 422
        before any remote call."""
        response = client.post(
            AUTHOR_JUDGE_URL,
            json={"target_specification": "", "target_task_prompt": "p"},
        )
        assert response.status_code == 422


# ───────────────────────── refine_judge ──────────────────────────────────

REFINE_JUDGE_URL = "/api/projects/p1/tasks/t1/eval_builder/refine_judge"


@pytest.fixture
def refine_judge_input():
    return {
        "judge_prompt": "The agent must not fabricate policies. PASS if it hedges, FAIL otherwise.",
        "graded_traces": [
            {
                "trace_label": "leaf-abc",
                "judge_score": "fail",
                "judge_reasoning": "Stated a return window as fact.",
                "overview": "The user asked about returns and the agent quoted a window.",
                "claims": [
                    {
                        "text": "The agent stated an unverified return window as fact [1].",
                        "human_grade": "agree",
                        "human_feedback": None,
                    },
                    {
                        "text": "It fails because the window was never verified [1].",
                        "human_grade": "disagree",
                        "human_feedback": "The window is actually documented, so this should pass.",
                    },
                ],
                "human_verdict": "pass",
            }
        ],
    }


class TestRefineJudge:
    def test_refine_judge_no_api_key(self, client, refine_judge_input):
        """Fail-fast: a keyless caller gets a clean 401 before the remote call."""
        with patch(
            "app.desktop.studio_server.utils.copilot_utils.Config.shared"
        ) as mock_config_shared:
            mock_config = mock_config_shared.return_value
            mock_config.kiln_copilot_api_key = None
            response = client.post(REFINE_JUDGE_URL, json=refine_judge_input)
            assert response.status_code == 401
            assert "API key not configured" in response.json()["message"]

    def test_refine_judge_success(self, client, refine_judge_input, mock_api_key):
        mock_output = MagicMock(spec=RefineJudgePromptOutput)
        mock_output.to_dict.return_value = {
            "refined_judge_prompt": "The agent must not fabricate policies. A specific unverified detail stated as fact is a FAILURE.",
            "changes": [
                {
                    "change": "Made an unverified detail stated as fact an explicit failure.",
                    "rationale": "trace leaf-abc: reviewer disagreed with the fail on a documented window.",
                }
            ],
            "not_incorporated_feedback": None,
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = mock_output

        with patch(
            "app.desktop.studio_server.utils.eval_builder_utils.refine_judge_prompt_v1_copilot_refine_judge_prompt_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as sdk_call:
            response = client.post(REFINE_JUDGE_URL, json=refine_judge_input)
            assert response.status_code == 200
            result = response.json()
            assert "FAILURE" in result["refined_judge_prompt"]
            assert len(result["changes"]) == 1
            assert result["changes"][0]["rationale"].startswith("trace leaf-abc")
            assert result["not_incorporated_feedback"] is None

            # The graded card reaches the refiner whole: the overview, every
            # claim with its grade (a blank why as an explicit null), and the
            # reviewer's overall call.
            sent = sdk_call.call_args.kwargs["body"].to_dict()["graded_traces"][0]
            expected = refine_judge_input["graded_traces"][0]
            assert sent["overview"] == expected["overview"]
            assert sent["human_verdict"] == expected["human_verdict"]
            assert [c["text"] for c in sent["claims"]] == [
                c["text"] for c in expected["claims"]
            ]
            assert sent["claims"][0]["human_feedback"] is None

    def test_refine_judge_remote_error_surfaces_upstream_message(
        self, client, refine_judge_input, mock_api_key
    ):
        """A remote failure propagates the upstream status + message (the
        custom error handler renders it as {"message": ...} for the UI)."""
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.content = b'{"message": "upstream refused"}'
        mock_response.parsed = None

        with patch(
            "app.desktop.studio_server.utils.eval_builder_utils.refine_judge_prompt_v1_copilot_refine_judge_prompt_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(REFINE_JUDGE_URL, json=refine_judge_input)
            assert response.status_code == 502
            assert "upstream refused" in response.json()["message"]

    def test_refine_judge_no_response_is_500(
        self, client, refine_judge_input, mock_api_key
    ):
        """A 2xx with no parsed body surfaces as a 500 with a clear message."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = None

        with patch(
            "app.desktop.studio_server.utils.eval_builder_utils.refine_judge_prompt_v1_copilot_refine_judge_prompt_post.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(REFINE_JUDGE_URL, json=refine_judge_input)
            assert response.status_code == 500
            assert "Failed to refine the judge prompt" in response.json()["message"]

    def test_refine_judge_rejects_empty_graded_traces(self, client, mock_api_key):
        """graded_traces must be non-empty (min_length=1) — a 422 before any
        remote call."""
        response = client.post(
            REFINE_JUDGE_URL,
            json={"judge_prompt": "p", "graded_traces": []},
        )
        assert response.status_code == 422

    def test_refine_judge_rejects_a_trace_with_no_claims(
        self, client, refine_judge_input, mock_api_key
    ):
        """A graded trace carries every claim on its card, never a subset, and
        a card always has at least one; an empty list is a 422 here rather
        than a rejection from the refiner."""
        refine_judge_input["graded_traces"][0]["claims"] = []
        response = client.post(REFINE_JUDGE_URL, json=refine_judge_input)
        assert response.status_code == 422


# ───────────────────────── multi_turn_pipeline (SSE) ─────────────────────────

PIPELINE_URL = "/api/projects/p1/tasks/t1/eval_builder/multi_turn_pipeline"


def _pipeline_case(i: int) -> dict:
    return {
        "seed_prompt": f"seed-{i}",
        "synthetic_user_info": (
            f"<persona>persona-{i}</persona>"
            f"<goal>goal-{i}</goal>"
            f"<behavior_guidance>guide-{i}</behavior_guidance>"
        ),
        "scenario_index": i,
    }


@pytest.fixture
def pipeline_request():
    return {
        "cases": [_pipeline_case(0), _pipeline_case(1)],
        "turns": 2,
        # Inline run config = the FULL properties shape a manual run sends.
        "target_run_config": {
            "model_name": "gpt_5_5",
            "model_provider_name": "openrouter",
            "prompt_id": "simple_prompt_builder",
            "structured_output_mode": "default",
        },
        "su_driver": {
            "model_name": "claude_4_5_haiku",
            "model_provider": "openrouter",
        },
        "judge": {
            "prompt": "Judge whether the output fabricates policy.",
            "model_name": "claude_sonnet_4_6",
            "model_provider": "anthropic",
        },
    }


# A drive trace shaped like the runner's real traces: system turn, tool
# call, tool result — the full fidelity the judge and claim builder consume.
def _real_trace(i: int) -> list[dict]:
    return [
        {"role": "system", "content": "You are a support agent."},
        {"role": "user", "content": f"question {i}"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup_policy", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "30 day window"},
        {"role": "assistant", "content": f"answer {i}"},
    ]


def _rate_limit_error() -> litellm.RateLimitError:
    """A transient provider error, per the shared retry classifier."""
    return litellm.RateLimitError(
        message="upstream rate limit",
        llm_provider="openrouter",
        model="gpt_5_5",
    )


def _auth_error() -> litellm.AuthenticationError:
    """A config-scoped (batch-fatal) provider error."""
    return litellm.AuthenticationError(
        message="invalid api key",
        llm_provider="openrouter",
        model="gpt_5_5",
    )


def _fake_run_cases_batch(*, fail_case: int | None = None):
    """An async-generator stand-in for the libs/core runner: batch_started,
    then per case its turn events and completion (or failure)."""
    from kiln_ai.synthetic_user.runner import (
        BatchCompletedEvent,
        BatchStartedEvent,
        CaseCompletedEvent,
        CaseFailedEvent,
        TurnCompletedEvent,
    )

    async def fake(*, cases, turns, **_kwargs):
        yield BatchStartedEvent(batch_tag="tag123", num_cases=len(cases))
        successful = 0
        failed = 0
        for i in range(len(cases)):
            if fail_case == i:
                yield CaseFailedEvent(
                    case_index=i,
                    error_code="unexpected_error",
                    message="drive blew up",
                    error_type="RateLimitError",
                )
                failed += 1
                continue
            for _turn in range(turns):
                yield TurnCompletedEvent(
                    case_index=i,
                    turn_index=_turn + 1,
                    assistant_run_id=f"run-{i}",
                    su_next_message="next",
                    cumulative_cost=0.01,
                    trace=_real_trace(i),
                )
            yield CaseCompletedEvent(
                case_index=i,
                chain_run_ids=[f"run-{i}-a", f"run-{i}-b"],
                leaf_run_id=f"leaf-{i}",
                total_turns=turns,
                total_cost=0.05,
            )
            successful += 1
        yield BatchCompletedEvent(
            successful=successful,
            failed=failed,
            batch_tag="tag123",
            total_cost=0.05 * successful,
        )

    return fake


def _multiturn_task_mock():
    return _task_mock()


@pytest.fixture
def pipeline_seams():
    """Patch the pipeline's seams: the copilot key, task resolution, the
    drive runner, the judge, and the claim builder. Yields the mocks for
    assertions."""
    with (
        patch(
            "app.desktop.studio_server.eval_builder_api.get_copilot_api_key",
            return_value="test_api_key",
        ),
        patch(
            "app.desktop.studio_server.eval_builder_api.task_from_id",
            return_value=_multiturn_task_mock(),
        ) as task_mock,
        patch(
            "app.desktop.studio_server.eval_builder_api.run_cases_batch",
            new=_fake_run_cases_batch(),
        ),
        patch(
            "app.desktop.studio_server.eval_builder_api.run_judge_for_trace",
            new=AsyncMock(return_value=JudgeVerdict("fail", "fabricated a policy")),
        ) as judge_mock,
        patch(
            "app.desktop.studio_server.eval_builder_api.build_claims_for_trace",
            new=AsyncMock(return_value=_claims_output()),
        ) as claims_mock,
        patch(
            "app.desktop.studio_server.eval_builder_api.delete_multi_turn_batch_chains",
            return_value=0,
        ) as delete_mock,
    ):
        yield {
            "task": task_mock,
            "judge": judge_mock,
            "claims": claims_mock,
            "delete": delete_mock,
        }


def _events_of(events: list, type_name: str) -> list[dict]:
    return [e for e in events if isinstance(e, dict) and e.get("type") == type_name]


class TestRunJudgeWithRetry:
    """The judge lane's hand-rolled retry, which mirrors the shared runner's
    posture. The streams cover the observable outcomes; these pin the waits."""

    @pytest.mark.asyncio
    async def test_transient_failures_back_off_exponentially(self):
        judge = AsyncMock(
            side_effect=[
                _rate_limit_error(),
                _rate_limit_error(),
                JudgeVerdict("pass", "fine"),
            ]
        )

        with (
            patch(
                "app.desktop.studio_server.eval_builder_api.run_judge_for_trace",
                new=judge,
            ),
            patch(
                "app.desktop.studio_server.eval_builder_api.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
            patch(
                "app.desktop.studio_server.eval_builder_api.compute_retry_delay",
                # Pin the jitter draw to the top of each backoff window.
                side_effect=lambda base, attempt: base * RETRY_BACKOFF_FACTOR**attempt,
            ) as mock_delay,
        ):
            verdict = await run_judge_with_retry("p1", "t1", "in", "out", "judge")

        assert verdict.judge_score == "pass"
        assert judge.await_count == 3
        # Zero-indexed attempts: the first retry draws from the base window.
        assert [call.args for call in mock_delay.call_args_list] == [
            (JUDGE_RETRY_DELAY_SECONDS, 0),
            (JUDGE_RETRY_DELAY_SECONDS, 1),
        ]
        assert [call.args[0] for call in mock_sleep.await_args_list] == [
            JUDGE_RETRY_DELAY_SECONDS,
            JUDGE_RETRY_DELAY_SECONDS * RETRY_BACKOFF_FACTOR,
        ]

    @pytest.mark.asyncio
    async def test_retries_are_capped_and_the_last_error_raises(self):
        judge = AsyncMock(side_effect=_rate_limit_error())

        with (
            patch(
                "app.desktop.studio_server.eval_builder_api.run_judge_for_trace",
                new=judge,
            ),
            patch(
                "app.desktop.studio_server.eval_builder_api.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            with pytest.raises(litellm.RateLimitError):
                await run_judge_with_retry("p1", "t1", "in", "out", "judge")

        assert judge.await_count == JUDGE_MAX_RETRIES + 1
        assert mock_sleep.await_count == JUDGE_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_without_waiting(self):
        judge = AsyncMock(side_effect=ValueError("judge output unparseable"))

        with (
            patch(
                "app.desktop.studio_server.eval_builder_api.run_judge_for_trace",
                new=judge,
            ),
            patch(
                "app.desktop.studio_server.eval_builder_api.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            with pytest.raises(ValueError, match="judge output unparseable"):
                await run_judge_with_retry("p1", "t1", "in", "out", "judge")

        assert judge.await_count == 1
        mock_sleep.assert_not_awaited()


class TestMultiTurnPipeline:
    def test_happy_path_full_stream(self, client, pipeline_request, pipeline_seams):
        resp = client.post(PIPELINE_URL, json=pipeline_request)

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(resp.text)

        started = _events_of(events, "batch_started")
        assert started == [
            {"type": "batch_started", "batch_tag": "tag123", "total_cases": 2}
        ]

        turns = _events_of(events, "turn_completed")
        assert len(turns) == 4  # 2 cases x 2 turns
        # Per-case turn counters climb 1..turns, each with the denominator.
        for case_index in (0, 1):
            case_turns = [t for t in turns if t["case_index"] == case_index]
            assert [t["turns_completed"] for t in case_turns] == [1, 2]
            assert all(t["total_turns"] == 2 for t in case_turns)

        driven = _events_of(events, "case_driven")
        assert {(d["case_index"], d["leaf_run_id"]) for d in driven} == {
            (0, "leaf-0"),
            (1, "leaf-1"),
        }

        judged = _events_of(events, "case_judged")
        assert len(judged) == 2
        for e in judged:
            assert e["judge_score"] == "fail"
            assert e["leaf_run_id"] == f"leaf-{e['case_index']}"
            assert e["total_cost"] == 0.05
            # Canonical transcript rendering of the REAL trace: tool calls
            # and tool results are present; the UI never sees a projection.
            assert "<assistant_requested_tool_calls>" in e["raw_output"]
            assert "<tool_tool_message>" in e["raw_output"]
            assert f"answer {e['case_index']}" in e["raw_output"]
            # raw_input = the conversation's opening user message.
            assert e["raw_input"] == f"question {e['case_index']}"
            # No claims on the stream: they're built lazily via build_claims.
            assert "claims" not in e and "overview" not in e
            # The structured trace rides along (additive): the same real
            # trace the judge saw, so the client can render the chat UI and
            # remap citation spans instead of parsing the flattened string.
            assert e["trace"] == _real_trace(e["case_index"])

        completed = _events_of(events, "batch_completed")
        assert completed == [
            {
                "type": "batch_completed",
                "judged": 2,
                "failed": 0,
                "batch_tag": "tag123",
                "total_cost": 0.1,
            }
        ]
        assert events[-1] == "complete"

        # The judge received the runner's REAL trace, not a projection.
        for call in pipeline_seams["judge"].call_args_list:
            trace = call.kwargs["trace"]
            assert any(m.get("role") == "system" for m in trace)
            assert any(m.get("role") == "tool" for m in trace)
        # The pipeline never spends on the claim builder — claims are built
        # per opened trace via the build_claims primitive.
        pipeline_seams["claims"].assert_not_called()
        # No replace_batch_tag → no delete.
        pipeline_seams["delete"].assert_not_called()

    def test_drive_failure_is_isolated(self, client, pipeline_request, pipeline_seams):
        """THE failure-isolation contract: a case dying in the drive stage
        must not discard the other case's completed review. The runner's
        error_type rides through onto the frame alongside the message."""
        with patch(
            "app.desktop.studio_server.eval_builder_api.run_cases_batch",
            new=_fake_run_cases_batch(fail_case=0),
        ):
            resp = client.post(PIPELINE_URL, json=pipeline_request)

        events = _parse_sse(resp.text)
        failed = _events_of(events, "case_failed")
        assert failed == [
            {
                "type": "case_failed",
                "case_index": 0,
                "stage": "drive",
                "code": "unexpected_error",
                "message": "drive blew up",
                "error_type": "RateLimitError",
            }
        ]
        judged = _events_of(events, "case_judged")
        assert [e["case_index"] for e in judged] == [1]
        completed = _events_of(events, "batch_completed")[0]
        assert completed["judged"] == 1
        assert completed["failed"] == 1
        assert events[-1] == "complete"

    def test_batch_total_includes_failed_and_retried_attempt_spend(
        self, client, pipeline_request, pipeline_seams
    ):
        """batch_completed.total_cost reports actual billing: the surviving
        conversation, its retried attempt's discarded spend, and the dead
        case's attempts. Per-case events keep conversation cost only."""
        from kiln_ai.synthetic_user.runner import (
            BatchStartedEvent,
            CaseCompletedEvent,
            CaseFailedEvent,
            TurnCompletedEvent,
        )

        async def fake(*, cases, turns, **_kwargs):
            yield BatchStartedEvent(batch_tag="tag123", num_cases=2)
            yield TurnCompletedEvent(
                case_index=0,
                turn_index=1,
                assistant_run_id="run-0",
                su_next_message=None,
                cumulative_cost=0.05,
                trace=_real_trace(0),
            )
            yield CaseCompletedEvent(
                case_index=0,
                chain_run_ids=["run-0"],
                leaf_run_id="leaf-0",
                total_turns=1,
                total_cost=0.05,
                discarded_attempts_cost=0.02,
            )
            yield CaseFailedEvent(
                case_index=1,
                error_code="unexpected_error",
                message="drive blew up",
                total_cost=0.03,
            )

        with patch(
            "app.desktop.studio_server.eval_builder_api.run_cases_batch",
            new=fake,
        ):
            resp = client.post(PIPELINE_URL, json=pipeline_request)

        events = _parse_sse(resp.text)
        judged = _events_of(events, "case_judged")
        # The judged case's cost stays the conversation's own spend.
        assert judged[0]["total_cost"] == 0.05
        completed = _events_of(events, "batch_completed")[0]
        assert completed["total_cost"] == pytest.approx(0.05 + 0.02 + 0.03)

    def test_judge_failure_is_isolated(self, client, pipeline_request, pipeline_seams):
        async def judge(
            _project_id, _task_id, _raw_input, _raw_output, _judge, **kwargs
        ):
            if kwargs["trace"][1]["content"] == "question 0":
                raise ValueError("judge exploded")
            return JudgeVerdict("pass", "fine")

        with patch(
            "app.desktop.studio_server.eval_builder_api.run_judge_for_trace",
            new=AsyncMock(side_effect=judge),
        ):
            resp = client.post(PIPELINE_URL, json=pipeline_request)

        events = _parse_sse(resp.text)
        failed = _events_of(events, "case_failed")
        assert len(failed) == 1
        assert failed[0]["case_index"] == 0
        assert failed[0]["stage"] == "judge"
        assert failed[0]["code"] == "judge_failed"
        assert "judge exploded" in failed[0]["message"]
        judged = _events_of(events, "case_judged")
        assert [e["case_index"] for e in judged] == [1]
        assert events[-1] == "complete"

    def test_judge_failure_surfaces_root_error_not_wrapper(
        self, client, pipeline_request, pipeline_seams
    ):
        """A KilnRunError-wrapped judge failure must put the ROOT provider
        error on the wire — as the message and as error_type — not the
        wrapper's genericized message or class."""
        root = litellm.BadRequestError(
            message="max_tokens too large for this model",
            model="claude_sonnet_4_6",
            llm_provider="anthropic",
        )
        with patch(
            "app.desktop.studio_server.eval_builder_api.run_judge_for_trace",
            new=AsyncMock(
                side_effect=KilnRunError(
                    "An unexpected error occurred.",
                    partial_trace=None,
                    original=root,
                )
            ),
        ):
            resp = client.post(PIPELINE_URL, json=pipeline_request)

        events = _parse_sse(resp.text)
        failed = _events_of(events, "case_failed")
        assert len(failed) == 2
        for e in failed:
            assert e["stage"] == "judge"
            assert e["code"] == "judge_failed"
            assert "BadRequestError" in e["message"]
            assert "max_tokens too large" in e["message"]
            assert "KilnRunError" not in e["message"]
            assert "unexpected error" not in e["message"]
            assert e["error_type"] == "BadRequestError"
        assert events[-1] == "complete"

    def test_replace_batch_tags_deleted_after_successful_drive(
        self, client, pipeline_request, pipeline_seams
    ):
        # Aborted re-drives can strand several batches; all of them are
        # cleaned once this drive has produced replacement chains.
        pipeline_request["replace_batch_tags"] = ["oldbatch123", "olderbatch456"]
        resp = client.post(PIPELINE_URL, json=pipeline_request)

        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        assert len(_events_of(events, "case_judged")) == 2
        delete_mock = pipeline_seams["delete"]
        assert [c.args[1] for c in delete_mock.call_args_list] == [
            "oldbatch123",
            "olderbatch456",
        ]

    def test_replace_batch_tag_not_deleted_when_nothing_drove(
        self, client, pipeline_request, pipeline_seams
    ):
        """A wholesale drive failure must keep the superseded batch — the
        user must never end up with neither batch."""

        async def all_fail_runner(*, cases, **_kwargs):
            from kiln_ai.synthetic_user.runner import (
                BatchStartedEvent,
                CaseFailedEvent,
            )

            yield BatchStartedEvent(batch_tag="tag123", num_cases=len(cases))
            for i in range(len(cases)):
                yield CaseFailedEvent(
                    case_index=i, error_code="unexpected_error", message="down"
                )

        pipeline_request["replace_batch_tags"] = ["oldbatch123"]
        with patch(
            "app.desktop.studio_server.eval_builder_api.run_cases_batch",
            new=all_fail_runner,
        ):
            resp = client.post(PIPELINE_URL, json=pipeline_request)

        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        completed = _events_of(events, "batch_completed")[0]
        assert completed["failed"] == 2
        pipeline_seams["delete"].assert_not_called()

    def test_saved_run_config_reaches_the_runner_verbatim(
        self, client, pipeline_request, pipeline_seams
    ):
        """A target_run_config_id resolves to the saved config's properties,
        handed to the runner untouched — tools and sampling included — with
        the config's id for run attribution."""
        rc = Mock()
        rc.id = "rc-1"
        rc.run_config_properties = KilnAgentRunConfigProperties(
            model_name="gpt_5_5",
            model_provider_name=ModelProviderName.openrouter,
            prompt_id="simple_prompt_builder",
            structured_output_mode=StructuredOutputMode.json_schema,
            tools_config=ToolsRunConfig(tools=["kiln_tool::add_numbers"]),
        )
        task = pipeline_seams["task"].return_value
        task.run_configs.return_value = [rc]

        captured: dict = {}
        inner = _fake_run_cases_batch()

        async def capturing(*, cases, turns, **kwargs):
            captured.update(kwargs)
            async for event in inner(cases=cases, turns=turns, **kwargs):
                yield event

        del pipeline_request["target_run_config"]
        pipeline_request["target_run_config_id"] = "rc-1"
        with (
            patch(
                "app.desktop.studio_server.eval_builder_api.run_cases_batch",
                new=capturing,
            ),
            patch(
                "app.desktop.studio_server.eval_api.task_from_id",
                return_value=task,
            ),
        ):
            resp = client.post(PIPELINE_URL, json=pipeline_request)

        assert resp.status_code == 200
        assert captured["target_run_config"] is rc.run_config_properties
        assert captured["task_run_config_id"] == "rc-1"

    def test_unknown_run_config_id_is_404_before_the_stream(
        self, client, pipeline_request, pipeline_seams
    ):
        """Resolution happens at construction, so a bad id is a clean 404 —
        never a half-open event stream."""
        task = pipeline_seams["task"].return_value
        task.run_configs.return_value = []
        del pipeline_request["target_run_config"]
        pipeline_request["target_run_config_id"] = "missing"
        with patch(
            "app.desktop.studio_server.eval_api.task_from_id",
            return_value=task,
        ):
            resp = client.post(PIPELINE_URL, json=pipeline_request)
        assert resp.status_code == 404
        assert resp.json()["message"]["code"] == "run_config_not_found"
        assert not resp.headers["content-type"].startswith("text/event-stream")

    def test_rejects_single_turn_task(self, client, pipeline_request, pipeline_seams):
        from kiln_ai.datamodel.datamodel_enums import TurnMode

        pipeline_seams["task"].return_value.turn_mode = TurnMode.single_turn
        resp = client.post(PIPELINE_URL, json=pipeline_request)
        assert resp.status_code == 400
        assert resp.json()["message"]["code"] == "task_not_multiturn"

    def test_rejects_invalid_case_shape(self, client, pipeline_request, pipeline_seams):
        pipeline_request["cases"] = [{"seed_prompt": "only a seed"}]
        resp = client.post(PIPELINE_URL, json=pipeline_request)
        assert resp.status_code == 400
        assert resp.json()["message"]["code"] == "invalid_case_shape"

    def test_rejects_oversized_batch(self, client, pipeline_request, pipeline_seams):
        pipeline_request["cases"] = [
            _pipeline_case(i) for i in range(NUM_CASES_MAX + 1)
        ]
        resp = client.post(PIPELINE_URL, json=pipeline_request)
        assert resp.status_code == 422

    def test_accepts_batch_at_the_cap(self, client, pipeline_request, pipeline_seams):
        """The cap is inclusive — a full-size batch clears validation and opens
        the stream. Pairs with the over-cap test to pin both sides."""
        pipeline_request["cases"] = [_pipeline_case(i) for i in range(NUM_CASES_MAX)]
        resp = client.post(PIPELINE_URL, json=pipeline_request)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    def test_rejects_retired_spec_name_field(
        self, client, pipeline_request, pipeline_seams
    ):
        # spec_name left this contract when the judge moved to the constant
        # draft score — a stale client still sending it must 422 loudly.
        pipeline_request["spec_name"] = "Test Spec"
        resp = client.post(PIPELINE_URL, json=pipeline_request)
        assert resp.status_code == 422

    def test_drive_crash_surfaces_batch_failed(
        self, client, pipeline_request, pipeline_seams
    ):
        """A runner-level crash (developer bug, not a per-case failure) must
        end the stream with batch_failed — never a clean batch_completed."""

        async def crashing_runner(**_kwargs):
            from kiln_ai.synthetic_user.runner import BatchStartedEvent

            yield BatchStartedEvent(batch_tag="tag123", num_cases=2)
            raise RuntimeError("runner exploded")

        with patch(
            "app.desktop.studio_server.eval_builder_api.run_cases_batch",
            new=crashing_runner,
        ):
            resp = client.post(PIPELINE_URL, json=pipeline_request)

        events = _parse_sse(resp.text)
        assert _events_of(events, "batch_completed") == []
        failed = _events_of(events, "batch_failed")
        assert len(failed) == 1
        assert failed[0]["code"] == "internal_error"
        assert "runner exploded" in failed[0]["message"]
        assert events[-1] == "complete"

    def test_judge_transient_failure_retries_then_succeeds(
        self, client, pipeline_request, pipeline_seams
    ):
        """A transient judge failure (shared retry classifier) is retried in
        place — the case still lands as case_judged, never case_failed."""
        calls = {"case_0": 0}

        async def flaky_judge(
            _project_id, _task_id, _raw_input, _raw_output, _judge, **kwargs
        ):
            if kwargs["trace"][1]["content"] == "question 0":
                calls["case_0"] += 1
                if calls["case_0"] == 1:
                    raise _rate_limit_error()
            return JudgeVerdict("pass", "fine")

        with (
            patch(
                "app.desktop.studio_server.eval_builder_api.run_judge_for_trace",
                new=AsyncMock(side_effect=flaky_judge),
            ),
            patch(
                "app.desktop.studio_server.eval_builder_api.JUDGE_RETRY_DELAY_SECONDS",
                0,
            ),
        ):
            resp = client.post(PIPELINE_URL, json=pipeline_request)

        events = _parse_sse(resp.text)
        assert _events_of(events, "case_failed") == []
        judged = _events_of(events, "case_judged")
        assert sorted(e["case_index"] for e in judged) == [0, 1]
        assert calls["case_0"] == 2  # first attempt + one retry
        completed = _events_of(events, "batch_completed")[0]
        assert completed["judged"] == 2
        assert completed["failed"] == 0

    def test_judge_deterministic_failure_does_not_retry(
        self, client, pipeline_request, pipeline_seams
    ):
        """Deterministic judge failures fail the case on the FIRST attempt —
        retrying a non-transient error would just triple the spend."""
        calls = {"case_0": 0}

        async def broken_judge(
            _project_id, _task_id, _raw_input, _raw_output, _judge, **kwargs
        ):
            if kwargs["trace"][1]["content"] == "question 0":
                calls["case_0"] += 1
                raise ValueError("judge output unparseable")
            return JudgeVerdict("pass", "fine")

        with patch(
            "app.desktop.studio_server.eval_builder_api.run_judge_for_trace",
            new=AsyncMock(side_effect=broken_judge),
        ):
            resp = client.post(PIPELINE_URL, json=pipeline_request)

        events = _parse_sse(resp.text)
        failed = _events_of(events, "case_failed")
        assert len(failed) == 1
        assert failed[0]["stage"] == "judge"
        assert calls["case_0"] == 1  # no retry
        assert [e["case_index"] for e in _events_of(events, "case_judged")] == [1]

    def test_judge_batch_fatal_failure_aborts_pipeline(
        self, client, pipeline_request, pipeline_seams
    ):
        """A config-scoped judge failure (dead key, deprecated model) aborts
        the WHOLE batch: one batch_aborted frame in place of batch_completed,
        no per-case failure spam, stream still terminates cleanly."""
        with patch(
            "app.desktop.studio_server.eval_builder_api.run_judge_for_trace",
            new=AsyncMock(side_effect=_auth_error()),
        ):
            resp = client.post(PIPELINE_URL, json=pipeline_request)

        events = _parse_sse(resp.text)
        aborted = _events_of(events, "batch_aborted")
        assert len(aborted) == 1  # first batch-fatal error wins, exactly once
        assert aborted[0]["stage"] == "judge"
        assert "AuthenticationError" in aborted[0]["error"]
        assert _events_of(events, "batch_completed") == []
        assert _events_of(events, "case_failed") == []
        assert events[-1] == "complete"

    def test_abort_cancels_the_running_drive(
        self, client, pipeline_request, pipeline_seams
    ):
        """The abort reuses the consumer-disconnect teardown: the drive task
        is cancelled mid-flight (AsyncJobRunner then cancels its workers), so
        a doomed batch stops spending instead of driving the queued cases."""
        from kiln_ai.synthetic_user.runner import (
            BatchStartedEvent,
            CaseCompletedEvent,
            TurnCompletedEvent,
        )

        drive_cancelled = {"flag": False}

        async def slow_runner(*, cases, turns, **_kwargs):
            yield BatchStartedEvent(batch_tag="tag123", num_cases=len(cases))
            yield TurnCompletedEvent(
                case_index=0,
                turn_index=1,
                assistant_run_id="run-0",
                su_next_message="next",
                cumulative_cost=0.01,
                trace=_real_trace(0),
            )
            yield CaseCompletedEvent(
                case_index=0,
                chain_run_ids=["run-0-a"],
                leaf_run_id="leaf-0",
                total_turns=turns,
                total_cost=0.05,
            )
            try:
                # Case 1 would take much longer; the abort must not wait it out.
                await asyncio.sleep(30)
                yield CaseCompletedEvent(
                    case_index=1,
                    chain_run_ids=["run-1-a"],
                    leaf_run_id="leaf-1",
                    total_turns=turns,
                    total_cost=0.05,
                )
            except asyncio.CancelledError:
                drive_cancelled["flag"] = True
                raise

        with (
            patch(
                "app.desktop.studio_server.eval_builder_api.run_cases_batch",
                new=slow_runner,
            ),
            patch(
                "app.desktop.studio_server.eval_builder_api.run_judge_for_trace",
                new=AsyncMock(side_effect=_auth_error()),
            ),
        ):
            resp = client.post(PIPELINE_URL, json=pipeline_request)

        events = _parse_sse(resp.text)
        assert len(_events_of(events, "batch_aborted")) == 1
        # Case 1 never drove: its 30s of spend was cancelled by the abort.
        assert [e["case_index"] for e in _events_of(events, "case_driven")] == [0]
        assert drive_cancelled["flag"] is True
        assert events[-1] == "complete"

    def test_missing_copilot_key_is_401_before_any_drive(
        self, client, pipeline_request
    ):
        """Fail fast for non-Pro users: without a copilot key the claims
        stage can never succeed, so the request must 4xx before the user
        burns their own model spend driving and judging every case."""
        with (
            patch(
                "app.desktop.studio_server.utils.copilot_utils.Config.shared"
            ) as mock_config,
            patch(
                "app.desktop.studio_server.eval_builder_api.run_cases_batch"
            ) as runner_mock,
        ):
            mock_config.return_value.kiln_copilot_api_key = None
            resp = client.post(PIPELINE_URL, json=pipeline_request)

        assert resp.status_code == 401
        assert "API key not configured" in resp.json()["message"]
        runner_mock.assert_not_called()

    def test_rejects_own_batch_tag_in_replace_list(
        self, client, pipeline_request, pipeline_seams
    ):
        pipeline_request["batch_tag"] = "mybatch"
        pipeline_request["replace_batch_tags"] = ["mybatch"]
        resp = client.post(PIPELINE_URL, json=pipeline_request)
        assert resp.status_code == 422

    def test_rejects_unknown_request_fields(
        self, client, pipeline_request, pipeline_seams
    ):
        """A retired or misspelled field must 422 — silently dropping it can
        disable behavior (e.g. batch cleanup) with no signal."""
        pipeline_request["replace_batch_tag"] = "oldbatch123"
        resp = client.post(PIPELINE_URL, json=pipeline_request)
        assert resp.status_code == 422


# ───────────────────────── judge_traces (SSE) ─────────────────────────

JUDGE_TRACES_URL = "/api/projects/p1/tasks/t1/eval_builder/judge_traces"


@pytest.fixture
def judge_traces_request():
    return {
        "leaf_run_ids": ["leaf-0", "leaf-1"],
        "judge": {
            "prompt": "Judge whether the output fabricates policy.",
            "model_name": "claude_sonnet_4_6",
            "model_provider": "anthropic",
        },
    }


def _leaf_run(i: int) -> Mock:
    """A stored chain leaf as loaded from disk: the leaf TaskRun carries the
    chain's full cumulative trace."""
    leaf = Mock()
    leaf.trace = _real_trace(i)
    return leaf


@pytest.fixture
def judge_traces_seams():
    """Patch the re-judge stream's seams: the copilot key, task resolution,
    the disk loader, and the judge. Yields the mocks for assertions."""
    task = _multiturn_task_mock()
    # The reload scans the task's run directory; the loader is mocked, so
    # any path value works.
    task.path = "/fake/task/path"
    with (
        patch(
            "app.desktop.studio_server.eval_builder_api.get_copilot_api_key",
            return_value="test_api_key",
        ),
        patch(
            "app.desktop.studio_server.eval_builder_api.task_from_id",
            return_value=task,
        ) as task_mock,
        patch("app.desktop.studio_server.eval_builder_api.TaskRun") as task_run_mock,
        patch(
            "app.desktop.studio_server.eval_builder_api.run_judge_for_trace",
            new=AsyncMock(return_value=JudgeVerdict("fail", "fabricated a policy")),
        ) as judge_mock,
    ):
        task_run_mock.from_ids_and_parent_path = MagicMock(
            return_value={"leaf-0": _leaf_run(0), "leaf-1": _leaf_run(1)}
        )
        yield {
            "task": task_mock,
            "loader": task_run_mock.from_ids_and_parent_path,
            "judge": judge_mock,
        }


class TestJudgeTraces:
    def test_happy_path_full_stream(
        self, client, judge_traces_request, judge_traces_seams
    ):
        resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(resp.text)

        # Frame order: batch_started first, batch_completed last before the
        # terminator; no drive-stage frames on this stream at all.
        assert events[0] == {
            "type": "batch_started",
            "batch_tag": "",
            "total_cases": 2,
        }
        assert _events_of(events, "turn_completed") == []
        assert _events_of(events, "case_driven") == []

        judged = _events_of(events, "case_judged")
        assert len(judged) == 2
        for e in judged:
            assert e["judge_score"] == "fail"
            # case_index = position in leaf_run_ids; leaf_run_id echoed.
            assert e["leaf_run_id"] == f"leaf-{e['case_index']}"
            # No new drive spend this round.
            assert e["total_cost"] == 0.0
            # Canonical transcript rendering of the STORED trace: tool calls
            # and tool results present, same as the drive-time judge saw.
            assert "<assistant_requested_tool_calls>" in e["raw_output"]
            assert "<tool_tool_message>" in e["raw_output"]
            assert e["raw_input"] == f"question {e['case_index']}"
            # No claims on the stream: they're built lazily via build_claims.
            assert "claims" not in e and "overview" not in e
            # The structured trace rides along for chat rendering/citations.
            assert e["trace"] == _real_trace(e["case_index"])

        completed = _events_of(events, "batch_completed")
        assert completed == [
            {
                "type": "batch_completed",
                "judged": 2,
                "failed": 0,
                "batch_tag": "",
                "total_cost": 0.0,
            }
        ]
        assert events[-2] == completed[0]
        assert events[-1] == "complete"

        # The judge received the stored structured trace, not a projection.
        for call in judge_traces_seams["judge"].call_args_list:
            trace = call.kwargs["trace"]
            assert any(m.get("role") == "system" for m in trace)
            assert any(m.get("role") == "tool" for m in trace)
        # One bulk disk scan serves the whole batch.
        judge_traces_seams["loader"].assert_called_once()
        assert judge_traces_seams["loader"].call_args.args[0] == {"leaf-0", "leaf-1"}

    def test_case_index_follows_request_order(
        self, client, judge_traces_request, judge_traces_seams
    ):
        """case_index is the position in the REQUEST list, not disk order —
        the client keys its review state on it."""
        judge_traces_request["leaf_run_ids"] = ["leaf-1", "leaf-0"]
        resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)

        judged = _events_of(_parse_sse(resp.text), "case_judged")
        by_index = {e["case_index"]: e["leaf_run_id"] for e in judged}
        assert by_index == {0: "leaf-1", 1: "leaf-0"}

    def test_missing_chain_fails_case_and_batch_continues(
        self, client, judge_traces_request, judge_traces_seams
    ):
        """A leaf_run_id that no longer resolves (deleted or replaced chain)
        fails THAT case; the other cases still get judged."""
        judge_traces_seams["loader"].return_value = {"leaf-1": _leaf_run(1)}
        resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)

        events = _parse_sse(resp.text)
        failed = _events_of(events, "case_failed")
        assert len(failed) == 1
        assert failed[0]["case_index"] == 0
        assert failed[0]["stage"] == "judge"
        assert failed[0]["code"] == "trace_not_found"
        assert "leaf-0" in failed[0]["message"]
        judged = _events_of(events, "case_judged")
        assert [e["case_index"] for e in judged] == [1]
        completed = _events_of(events, "batch_completed")[0]
        assert completed["judged"] == 1
        assert completed["failed"] == 1
        assert events[-1] == "complete"

    def test_traceless_chain_fails_case(
        self, client, judge_traces_request, judge_traces_seams
    ):
        """A leaf that loads but has no stored trace cannot be judged —
        honest per-case failure, never a fabricated empty transcript."""
        bare_leaf = Mock()
        bare_leaf.trace = None
        judge_traces_seams["loader"].return_value = {
            "leaf-0": bare_leaf,
            "leaf-1": _leaf_run(1),
        }
        resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)

        events = _parse_sse(resp.text)
        failed = _events_of(events, "case_failed")
        assert len(failed) == 1
        assert failed[0]["case_index"] == 0
        assert failed[0]["code"] == "missing_trace"
        assert [e["case_index"] for e in _events_of(events, "case_judged")] == [1]

    def test_judge_failure_is_isolated(
        self, client, judge_traces_request, judge_traces_seams
    ):
        async def judge(
            _project_id, _task_id, _raw_input, _raw_output, _judge, **kwargs
        ):
            if kwargs["trace"][1]["content"] == "question 0":
                raise ValueError("judge exploded")
            return JudgeVerdict("pass", "fine")

        with patch(
            "app.desktop.studio_server.eval_builder_api.run_judge_for_trace",
            new=AsyncMock(side_effect=judge),
        ):
            resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)

        events = _parse_sse(resp.text)
        failed = _events_of(events, "case_failed")
        assert len(failed) == 1
        assert failed[0]["case_index"] == 0
        assert failed[0]["stage"] == "judge"
        assert failed[0]["code"] == "judge_failed"
        assert "judge exploded" in failed[0]["message"]
        assert [e["case_index"] for e in _events_of(events, "case_judged")] == [1]
        assert events[-1] == "complete"

    def test_judge_transient_failure_retries_then_succeeds(
        self, client, judge_traces_request, judge_traces_seams
    ):
        """The re-judge stream runs the SAME judge unit as the pipeline:
        transient failures retry in place under the shared classifier."""
        calls = {"case_0": 0}

        async def flaky_judge(
            _project_id, _task_id, _raw_input, _raw_output, _judge, **kwargs
        ):
            if kwargs["trace"][1]["content"] == "question 0":
                calls["case_0"] += 1
                if calls["case_0"] == 1:
                    raise _rate_limit_error()
            return JudgeVerdict("pass", "fine")

        with (
            patch(
                "app.desktop.studio_server.eval_builder_api.run_judge_for_trace",
                new=AsyncMock(side_effect=flaky_judge),
            ),
            patch(
                "app.desktop.studio_server.eval_builder_api.JUDGE_RETRY_DELAY_SECONDS",
                0,
            ),
        ):
            resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)

        events = _parse_sse(resp.text)
        assert _events_of(events, "case_failed") == []
        judged = _events_of(events, "case_judged")
        assert sorted(e["case_index"] for e in judged) == [0, 1]
        assert calls["case_0"] == 2  # first attempt + one retry

    def test_judge_batch_fatal_failure_aborts_batch(
        self, client, judge_traces_request, judge_traces_seams
    ):
        """A config-scoped judge failure (dead key, deprecated model) aborts
        the WHOLE batch: one batch_aborted frame in place of batch_completed,
        no per-case failure spam, stream still terminates cleanly."""
        with patch(
            "app.desktop.studio_server.eval_builder_api.run_judge_for_trace",
            new=AsyncMock(side_effect=_auth_error()),
        ):
            resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)

        events = _parse_sse(resp.text)
        aborted = _events_of(events, "batch_aborted")
        assert len(aborted) == 1  # first batch-fatal error wins, exactly once
        assert aborted[0]["stage"] == "judge"
        assert "AuthenticationError" in aborted[0]["error"]
        assert _events_of(events, "batch_completed") == []
        assert _events_of(events, "case_failed") == []
        assert events[-1] == "complete"

    def test_rejects_empty_leaf_run_ids(
        self, client, judge_traces_request, judge_traces_seams
    ):
        judge_traces_request["leaf_run_ids"] = []
        resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)
        assert resp.status_code == 422

    def test_rejects_blank_leaf_run_id(
        self, client, judge_traces_request, judge_traces_seams
    ):
        judge_traces_request["leaf_run_ids"] = ["leaf-0", "   "]
        resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)
        assert resp.status_code == 422

    def test_rejects_oversized_batch(
        self, client, judge_traces_request, judge_traces_seams
    ):
        judge_traces_request["leaf_run_ids"] = [
            f"leaf-{i}" for i in range(NUM_CASES_MAX + 1)
        ]
        resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)
        assert resp.status_code == 422

    def test_accepts_batch_at_the_cap(
        self, client, judge_traces_request, judge_traces_seams
    ):
        """The cap is inclusive — a full-size re-judge clears validation and
        opens the stream. Pairs with the over-cap test to pin both sides."""
        judge_traces_request["leaf_run_ids"] = [
            f"leaf-{i}" for i in range(NUM_CASES_MAX)
        ]
        resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    def test_rejects_retired_spec_name_field(
        self, client, judge_traces_request, judge_traces_seams
    ):
        # spec_name left this contract when the judge moved to the constant
        # draft score — a stale client still sending it must 422 loudly.
        judge_traces_request["spec_name"] = "Test Spec"
        resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)
        assert resp.status_code == 422

    def test_rejects_unknown_request_fields(
        self, client, judge_traces_request, judge_traces_seams
    ):
        """A retired or misspelled field must 422 — silently dropping it can
        change what gets judged with no signal."""
        judge_traces_request["batch_tag"] = "tag123"
        resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)
        assert resp.status_code == 422

    def test_missing_copilot_key_is_401_before_any_load(
        self, client, judge_traces_request, judge_traces_seams
    ):
        """Same fail-fast posture as multi_turn_pipeline: without a copilot key
        the claims stage that follows can never succeed, so the request must
        4xx before the user spends on judging every case."""
        with patch(
            "app.desktop.studio_server.eval_builder_api.get_copilot_api_key",
            side_effect=HTTPException(status_code=401, detail="API key not configured"),
        ):
            resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)

        assert resp.status_code == 401
        assert "API key not configured" in resp.json()["message"]
        judge_traces_seams["loader"].assert_not_called()


def _stored_single_turn_run(i: int, with_trace: bool = True) -> Mock:
    """A stored single-turn pipeline run as reloaded from disk: the I/O pair
    the judge scores plus the structured trace the frames echo."""
    run = Mock()
    run.input = f"question {i}"
    run.output = Mock()
    run.output.output = f"answer {i}"
    run.trace = _real_trace(i) if with_trace else None
    return run


@pytest.fixture
def judge_traces_single_turn_seams(judge_traces_seams):
    """The re-judge seams pointed at a single-turn task: the loader returns
    the pipeline's stored runs instead of chain leaves."""
    from kiln_ai.datamodel.datamodel_enums import TurnMode

    judge_traces_seams["task"].return_value.turn_mode = TurnMode.single_turn
    judge_traces_seams["loader"].return_value = {
        "leaf-0": _stored_single_turn_run(0),
        "leaf-1": _stored_single_turn_run(1),
    }
    yield judge_traces_seams


class TestJudgeTracesSingleTurn:
    """The single-turn arm of the re-judge stream: same frames, same judge
    unit, but the judge scores the stored run's I/O pair — the final_answer
    reading its pipeline and its saved eval use — never the trace."""

    def test_happy_path_judges_the_stored_trace(
        self, client, judge_traces_request, judge_traces_single_turn_seams
    ):
        resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)

        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        assert events[0] == {
            "type": "batch_started",
            "batch_tag": "",
            "total_cases": 2,
        }
        judged = _events_of(events, "case_judged")
        assert len(judged) == 2
        for e in judged:
            # The frame echoes the STORED run's I/O pair verbatim — no
            # transcript flattening on this arm.
            assert e["raw_input"] == f"question {e['case_index']}"
            # The transcript, not the closing message: the judge sees what
            # the agent did, so the answer is contained rather than equal.
            assert f"answer {e['case_index']}" in e["raw_output"]
            assert "assistant_message" in e["raw_output"]
            assert e["leaf_run_id"] == f"leaf-{e['case_index']}"
            assert e["total_cost"] == 0.0
            # The structured trace still rides along for the chat modal.
            assert e["trace"] == _real_trace(e["case_index"])
        completed = _events_of(events, "batch_completed")[0]
        assert completed["judged"] == 2 and completed["failed"] == 0
        assert events[-1] == "complete"

        # The judge scored the I/O pair with NO judge trace (final_answer
        # reading) — passing the trace here would silently flip the judge to
        # the full-trace reading the saved eval never uses.
        for call in judge_traces_single_turn_seams["judge"].call_args_list:
            # The judge reads the trace (tool calls included) while raw_input
            # stays the REQUEST's string, which is what the saved eval reads
            # back from its own item.
            assert call.kwargs["trace"] is not None
            assert call.args[2].startswith("question ")
            # The transcript carries the answer; it no longer IS the answer.
            assert "answer " in call.args[3]

    def test_traceless_run_still_judges(
        self, client, judge_traces_request, judge_traces_single_turn_seams
    ):
        """A stored run without a structured trace is still judgeable on this
        arm — the trace is a UI echo, not the judge input."""
        judge_traces_single_turn_seams["loader"].return_value = {
            "leaf-0": _stored_single_turn_run(0, with_trace=False),
            "leaf-1": _stored_single_turn_run(1),
        }
        resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)

        events = _parse_sse(resp.text)
        assert _events_of(events, "case_failed") == []
        judged = {e["case_index"]: e for e in _events_of(events, "case_judged")}
        # No stored trace, so the judge gets a two-message echo of the pair —
        # lossless, because the pair is everything that happened.
        assert judged[0]["trace"] == [
            {"role": "user", "content": "question 0"},
            {"role": "assistant", "content": "answer 0"},
        ]
        assert "answer 0" in judged[0]["raw_output"]
        assert "assistant_message" in judged[0]["raw_output"]

    def test_outputless_run_fails_case_and_batch_continues(
        self, client, judge_traces_request, judge_traces_single_turn_seams
    ):
        """A run with no stored output cannot be judged — honest per-case
        failure, never a fabricated empty answer."""
        bare = _stored_single_turn_run(0)
        bare.output.output = None
        judge_traces_single_turn_seams["loader"].return_value = {
            "leaf-0": bare,
            "leaf-1": _stored_single_turn_run(1),
        }
        resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)

        events = _parse_sse(resp.text)
        failed = _events_of(events, "case_failed")
        assert len(failed) == 1
        assert failed[0]["case_index"] == 0
        assert failed[0]["stage"] == "judge"
        assert failed[0]["code"] == "missing_output"
        assert [e["case_index"] for e in _events_of(events, "case_judged")] == [1]
        assert events[-1] == "complete"

    def test_missing_run_fails_case_and_batch_continues(
        self, client, judge_traces_request, judge_traces_single_turn_seams
    ):
        """Runs vanish between rounds too (delete-on-redrive, manual dataset
        edits) — same trace_not_found isolation as the multi-turn arm."""
        judge_traces_single_turn_seams["loader"].return_value = {
            "leaf-1": _stored_single_turn_run(1),
        }
        resp = client.post(JUDGE_TRACES_URL, json=judge_traces_request)

        events = _parse_sse(resp.text)
        failed = _events_of(events, "case_failed")
        assert len(failed) == 1
        assert failed[0]["code"] == "trace_not_found"
        assert "leaf-0" in failed[0]["message"]
        assert [e["case_index"] for e in _events_of(events, "case_judged")] == [1]


# ───────────────────────── single_turn_pipeline (SSE) ─────────────────────

SINGLE_TURN_URL = "/api/projects/p1/tasks/t1/eval_builder/single_turn_pipeline"


def _fake_single_turn_run(i: int, cost: float = 0.05, with_trace: bool = True):
    """A TaskRun stand-in with the real attributes the pipeline touches:
    tags mutate through the real tagging helper, save/delete are observable,
    and the trace is the structured shape the frames echo."""
    run = Mock()
    run.id = f"run-{i}"
    run.tags = []
    run.output = Mock()
    run.output.output = f"answer {i}"
    run.output.rating = None
    run.trace = _real_trace(i) if with_trace else None
    usage = Mock()
    usage.cost = cost
    run.cumulative_usage = usage
    run.save_to_file = Mock()
    run.delete = Mock()
    return run


@pytest.fixture
def single_turn_request():
    return {
        "inputs": ["What is your return policy?", "Cancel my order now"],
        "input_model_name": "gpt_5_5_mini",
        "input_provider": "openrouter",
        # Inline run config = the FULL properties shape a manual run sends.
        "target_run_config": {
            "model_name": "gpt_5_5",
            "model_provider_name": "openrouter",
            "prompt_id": "simple_prompt_builder",
            "structured_output_mode": "default",
        },
        "judge": {
            "prompt": "Judge whether the output fabricates policy.",
            "model_name": "claude_sonnet_4_6",
            "model_provider": "anthropic",
        },
    }


@pytest.fixture
def single_turn_seams(single_turn_request):
    """Patch the pipeline's seams: the copilot key, task resolution, skills,
    the adapter, the judge, and the batch deleter. `runs_by_input` maps each
    request input to what its invoke produces — a run, an exception, or a
    list popped per attempt (for retry tests). The real tagging helper runs
    against the fake runs, so tag assertions exercise the shipped code."""
    from kiln_ai.datamodel.datamodel_enums import TurnMode

    runs_by_input: dict = {
        text: _fake_single_turn_run(i)
        for i, text in enumerate(single_turn_request["inputs"])
    }
    invocations: list[dict] = []

    def fake_adapter_for_task(task, run_config, base_adapter_config=None):
        adapter = Mock()

        async def invoke(*, input, input_source=None):
            invocations.append(
                {
                    "input": input,
                    "input_source": input_source,
                    "adapter_config": base_adapter_config,
                }
            )
            key = input if isinstance(input, str) else json.dumps(input)
            outcome = runs_by_input[key]
            if isinstance(outcome, list):
                outcome = outcome.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        adapter.invoke = invoke
        return adapter

    with (
        patch(
            "app.desktop.studio_server.eval_builder_api.get_copilot_api_key",
            return_value="test_api_key",
        ),
        patch(
            "app.desktop.studio_server.eval_builder_api.task_from_id",
            return_value=_task_mock(TurnMode.single_turn),
        ) as task_mock,
        patch(
            "app.desktop.studio_server.eval_builder_api.load_skills_for_task",
            return_value={},
        ),
        patch(
            "app.desktop.studio_server.eval_builder_api.adapter_for_task",
            side_effect=fake_adapter_for_task,
        ),
        patch(
            "app.desktop.studio_server.eval_builder_api.run_judge_for_trace",
            new=AsyncMock(return_value=JudgeVerdict("fail", "fabricated a policy")),
        ) as judge_mock,
        patch(
            "app.desktop.studio_server.eval_builder_api.delete_single_turn_batch_runs",
            return_value=0,
        ) as delete_mock,
    ):
        yield {
            "task": task_mock,
            "judge": judge_mock,
            "delete": delete_mock,
            "runs_by_input": runs_by_input,
            "invocations": invocations,
        }


class TestSingleTurnPipeline:
    def test_happy_path_full_stream(
        self, client, single_turn_request, single_turn_seams
    ):
        resp = client.post(SINGLE_TURN_URL, json=single_turn_request)
        assert resp.status_code == 200
        events = _parse_sse(resp.text)

        started = _events_of(events, "batch_started")
        assert len(started) == 1
        assert started[0]["total_cases"] == 2
        # Auto-minted batch tag: 12 hex chars, echoed on the first frame.
        assert re.fullmatch(r"[0-9a-f]{12}", started[0]["batch_tag"])

        driven = _events_of(events, "case_driven")
        assert {e["leaf_run_id"] for e in driven} == {"run-0", "run-1"}

        judged = _events_of(events, "case_judged")
        assert len(judged) == 2
        by_index = {e["case_index"]: e for e in judged}
        for i, input_text in enumerate(single_turn_request["inputs"]):
            assert by_index[i]["raw_input"] == input_text
            assert f"answer {i}" in by_index[i]["raw_output"]
            assert "assistant_message" in by_index[i]["raw_output"]
            assert by_index[i]["leaf_run_id"] == f"run-{i}"
            assert by_index[i]["judge_score"] == "fail"
            assert by_index[i]["total_cost"] == 0.05
            # The run's structured trace rides the frame for the UI.
            assert by_index[i]["trace"][0]["role"] == "system"

        completed = _events_of(events, "batch_completed")
        assert completed == [
            {
                "type": "batch_completed",
                "judged": 2,
                "failed": 0,
                "batch_tag": started[0]["batch_tag"],
                "total_cost": 0.1,
            }
        ]
        assert events[-1] == "complete"

    def test_judge_reads_the_trace_and_keeps_the_request_input(
        self, client, single_turn_request, single_turn_seams
    ):
        """The judge must receive trace=None (final_answer parity with the
        saved eval) even though the run HAS a structured trace — the trace
        is a UI echo only."""
        client.post(SINGLE_TURN_URL, json=single_turn_request)
        judge = single_turn_seams["judge"]
        assert judge.await_count == 2
        for call in judge.await_args_list:
            # The judge reads the trace (tool calls included) while raw_input
            # stays the REQUEST's string, which is what the saved eval reads
            # back from its own item.
            assert call.kwargs["trace"] is not None

    def test_runs_are_batch_tagged_and_saved(
        self, client, single_turn_request, single_turn_seams
    ):
        client.post(
            SINGLE_TURN_URL, json={**single_turn_request, "batch_tag": "batch42"}
        )
        for text in single_turn_request["inputs"]:
            run = single_turn_seams["runs_by_input"][text]
            assert run.tags == sorted(
                ["single_turn_drive", "single_turn_drive_batch:batch42"]
            )
            run.save_to_file.assert_called_once()

    def test_run_config_id_stamps_adapter_config(
        self, client, single_turn_request, single_turn_seams
    ):
        """Inline config: no task_run_config_id stamped (ad-hoc by
        definition); the input source attributes the input-generator lane."""
        client.post(
            SINGLE_TURN_URL, json={**single_turn_request, "batch_tag": "batch42"}
        )
        invocation = single_turn_seams["invocations"][0]
        assert invocation["adapter_config"].task_run_config_id is None
        # default_tags rides the run's own save, so even a run orphaned by a
        # cancel mid-invoke stays discoverable by the batch sweeper.
        assert invocation["adapter_config"].default_tags == sorted(
            ["single_turn_drive", "single_turn_drive_batch:batch42"]
        )
        source_props = invocation["input_source"].properties
        assert source_props["model_name"] == "gpt_5_5_mini"
        assert source_props["model_provider"] == "openrouter"
        assert source_props["adapter_name"] == "kiln_eval_builder_single_turn"

    def test_run_failure_is_isolated(
        self, client, single_turn_request, single_turn_seams
    ):
        """A deterministic run failure fails that case at stage=run; the
        other case still runs and judges."""
        failed_input = single_turn_request["inputs"][0]
        single_turn_seams["runs_by_input"][failed_input] = ValueError("model exploded")
        resp = client.post(SINGLE_TURN_URL, json=single_turn_request)
        events = _parse_sse(resp.text)

        failed = _events_of(events, "case_failed")
        assert len(failed) == 1
        assert failed[0]["case_index"] == 0
        assert failed[0]["stage"] == "run"
        assert "model exploded" in failed[0]["message"]
        assert failed[0]["error_type"] == "ValueError"

        judged = _events_of(events, "case_judged")
        assert [e["case_index"] for e in judged] == [1]
        completed = _events_of(events, "batch_completed")
        assert completed[0]["judged"] == 1
        assert completed[0]["failed"] == 1

    def test_transient_run_error_retries(
        self, client, single_turn_request, single_turn_seams
    ):
        """A transient provider failure (shared classifier) retries the case
        instead of failing it — same posture as the multi-turn drive."""
        retried_input = single_turn_request["inputs"][0]
        recovered_run = _fake_single_turn_run(0)
        single_turn_seams["runs_by_input"][retried_input] = [
            _rate_limit_error(),
            recovered_run,
        ]
        # Zero the backoff base so the retry doesn't sleep a real jittered wait.
        with patch(
            "app.desktop.studio_server.eval_builder_api.RUN_RETRY_DELAY_SECONDS", 0
        ):
            resp = client.post(SINGLE_TURN_URL, json=single_turn_request)
        events = _parse_sse(resp.text)
        assert len(_events_of(events, "case_failed")) == 0
        assert len(_events_of(events, "case_judged")) == 2

    def test_exhausted_transient_run_error_names_its_class(
        self, client, single_turn_request, single_turn_seams
    ):
        """A transient failure that never recovers fails the case naming the
        REAL provider error: the retryable wrapper hides it, so error_type
        must come from the exception the wrapper was raised from."""
        failing_input = single_turn_request["inputs"][0]
        single_turn_seams["runs_by_input"][failing_input] = _rate_limit_error()
        with patch(
            "app.desktop.studio_server.eval_builder_api.RUN_RETRY_DELAY_SECONDS", 0
        ):
            resp = client.post(SINGLE_TURN_URL, json=single_turn_request)
        events = _parse_sse(resp.text)

        failed = _events_of(events, "case_failed")
        assert len(failed) == 1
        assert failed[0]["stage"] == "run"
        assert failed[0]["error_type"] == "RateLimitError"
        assert "upstream rate limit" in failed[0]["message"]
        assert [e["case_index"] for e in _events_of(events, "case_judged")] == [1]

    def test_judge_failure_is_isolated(
        self, client, single_turn_request, single_turn_seams
    ):
        """A non-fatal judge failure fails that case at stage=judge; the
        other case's verdict still lands."""
        failing_input = single_turn_request["inputs"][0]

        async def judge(_project, _task, raw_input, *args, **kwargs):
            if raw_input == failing_input:
                raise ValueError("judge choked")
            return JudgeVerdict("pass", "clean")

        single_turn_seams["judge"].side_effect = judge
        resp = client.post(SINGLE_TURN_URL, json=single_turn_request)
        events = _parse_sse(resp.text)

        failed = _events_of(events, "case_failed")
        assert len(failed) == 1
        assert failed[0]["stage"] == "judge"
        judged = _events_of(events, "case_judged")
        assert [e["case_index"] for e in judged] == [1]

    def test_batch_fatal_judge_error_aborts(
        self, client, single_turn_request, single_turn_seams
    ):
        """A config-scoped judge failure aborts the whole batch (one
        batch_aborted frame in place of batch_completed) — same contract as
        the multi-turn pipeline."""
        single_turn_seams["judge"].side_effect = _auth_error()
        resp = client.post(SINGLE_TURN_URL, json=single_turn_request)
        events = _parse_sse(resp.text)

        aborted = _events_of(events, "batch_aborted")
        assert len(aborted) == 1
        assert aborted[0]["stage"] == "judge"
        assert "invalid api key" in aborted[0]["error"]
        assert len(_events_of(events, "batch_completed")) == 0
        assert events[-1] == "complete"

    def test_superseded_batches_deleted_after_success(
        self, client, single_turn_request, single_turn_seams
    ):
        client.post(
            SINGLE_TURN_URL,
            json={**single_turn_request, "replace_batch_tags": ["old1", "old2"]},
        )
        delete = single_turn_seams["delete"]
        assert delete.call_count == 2
        deleted_tags = {call.args[1] for call in delete.call_args_list}
        assert deleted_tags == {"old1", "old2"}

    def test_no_deletion_when_nothing_driven(
        self, client, single_turn_request, single_turn_seams
    ):
        """A run stage that produced nothing keeps the superseded batches —
        a wholesale failure must never destroy the only batch on disk."""
        for text in single_turn_request["inputs"]:
            single_turn_seams["runs_by_input"][text] = ValueError("all dead")
        resp = client.post(
            SINGLE_TURN_URL,
            json={**single_turn_request, "replace_batch_tags": ["old1"]},
        )
        events = _parse_sse(resp.text)
        single_turn_seams["delete"].assert_not_called()
        completed = _events_of(events, "batch_completed")
        assert completed[0]["judged"] == 0
        assert completed[0]["failed"] == 2

    def test_structured_input_parsed_to_dict(
        self, client, single_turn_request, single_turn_seams
    ):
        """Tasks with an input schema carry inputs as JSON strings — parsed
        to a dict before invoke, mirroring base_eval.run_task at eval time."""
        from kiln_ai.datamodel.datamodel_enums import TurnMode

        single_turn_seams["task"].return_value = _task_mock(
            TurnMode.single_turn, input_json_schema='{"type": "object"}'
        )
        structured_input = json.dumps({"question": "What is your return policy?"})
        single_turn_seams["runs_by_input"][structured_input] = _fake_single_turn_run(0)
        resp = client.post(
            SINGLE_TURN_URL, json={**single_turn_request, "inputs": [structured_input]}
        )
        events = _parse_sse(resp.text)
        assert len(_events_of(events, "case_judged")) == 1
        assert single_turn_seams["invocations"][0]["input"] == {
            "question": "What is your return policy?"
        }
        # raw_input on the frame stays the JSON string — what the saved
        # eval's inputs-only item will store.
        assert _events_of(events, "case_judged")[0]["raw_input"] == structured_input

    def test_invalid_json_input_fails_case_without_spend(
        self, client, single_turn_request, single_turn_seams
    ):
        from kiln_ai.datamodel.datamodel_enums import TurnMode

        single_turn_seams["task"].return_value = _task_mock(
            TurnMode.single_turn, input_json_schema='{"type": "object"}'
        )
        resp = client.post(
            SINGLE_TURN_URL, json={**single_turn_request, "inputs": ["not json"]}
        )
        events = _parse_sse(resp.text)
        failed = _events_of(events, "case_failed")
        assert len(failed) == 1
        assert failed[0]["code"] == "invalid_input"
        # The parse failure precedes any model call — nothing was invoked.
        assert single_turn_seams["invocations"] == []

    def test_missing_output_fails_case_and_deletes_run(
        self, client, single_turn_request, single_turn_seams
    ):
        """A run with no output can't be judged: the case fails and the
        unusable persisted run is removed (it would otherwise sit on disk
        untagged and undiscoverable)."""
        target_input = single_turn_request["inputs"][0]
        bad_run = _fake_single_turn_run(0)
        bad_run.output = None
        single_turn_seams["runs_by_input"][target_input] = bad_run
        resp = client.post(SINGLE_TURN_URL, json=single_turn_request)
        events = _parse_sse(resp.text)
        failed = _events_of(events, "case_failed")
        assert len(failed) == 1
        assert failed[0]["code"] == "missing_output"
        # No exception underlies an empty output — nothing to name.
        assert failed[0]["error_type"] is None
        bad_run.delete.assert_called_once()
        # Cost honesty: the discarded run's spend was real — banked into the
        # batch total alongside the surviving case's 0.05.
        assert _events_of(events, "batch_completed")[0]["total_cost"] == 0.1

    def test_slow_run_completes_and_logs(
        self, client, single_turn_request, single_turn_seams, monkeypatch, caplog
    ):
        """A run slower than the soft log threshold completes and is
        judged, with the watchdog warning making the slowness visible in
        logs. The single-turn path has no seam that could prove the absence
        of a run budget; that property is pinned on the multi-turn runner's
        wait_for (test_no_case_timeout_by_default)."""
        slow_input = single_turn_request["inputs"][0]

        def fake_adapter(task, run_config, base_adapter_config=None):
            adapter = Mock()

            async def invoke(*, input, input_source=None):
                if input == slow_input:
                    await asyncio.sleep(0.2)
                return single_turn_seams["runs_by_input"][input]

            adapter.invoke = invoke
            return adapter

        monkeypatch.setattr(
            "kiln_ai.utils.slow_operation.DEFAULT_SLOW_LOG_THRESHOLD_SECONDS", 0.05
        )
        with (
            caplog.at_level(logging.WARNING, logger="kiln_ai.utils.slow_operation"),
            patch(
                "app.desktop.studio_server.eval_builder_api.adapter_for_task",
                side_effect=fake_adapter,
            ),
        ):
            resp = client.post(SINGLE_TURN_URL, json=single_turn_request)
        events = _parse_sse(resp.text)
        assert _events_of(events, "case_failed") == []
        assert sorted(e["case_index"] for e in _events_of(events, "case_judged")) == [
            0,
            1,
        ]
        slow_warnings = [r for r in caplog.records if "still running" in r.getMessage()]
        assert len(slow_warnings) == 1
        assert "case 0" in slow_warnings[0].getMessage()

    def test_multiturn_task_rejected(
        self, client, single_turn_request, single_turn_seams
    ):
        single_turn_seams["task"].return_value = _task_mock()
        resp = client.post(SINGLE_TURN_URL, json=single_turn_request)
        assert resp.status_code == 400
        assert "task_not_single_turn" in resp.text

    def test_missing_copilot_key_is_401(self, client, single_turn_request):
        """Same fail-fast posture as multi_turn_pipeline: the review that
        follows needs the remote claim builder, so a missing key stops the
        stream before any model spend."""
        with patch(
            "app.desktop.studio_server.utils.copilot_utils.Config.shared"
        ) as mock_config_shared:
            mock_config_shared.return_value.kiln_copilot_api_key = None
            resp = client.post(SINGLE_TURN_URL, json=single_turn_request)
        assert resp.status_code == 401

    def test_request_validation(self, client, single_turn_request):
        """The request contract fails loud: exactly one target config, no
        blank inputs, no self-replacement, no unknown fields."""
        no_config = {k: v for k, v in single_turn_request.items()}
        del no_config["target_run_config"]
        assert client.post(SINGLE_TURN_URL, json=no_config).status_code == 422

        both_configs = {**single_turn_request, "target_run_config_id": "rc1"}
        assert client.post(SINGLE_TURN_URL, json=both_configs).status_code == 422

        blank_input = {**single_turn_request, "inputs": ["ok", "  "]}
        assert client.post(SINGLE_TURN_URL, json=blank_input).status_code == 422

        self_replace = {
            **single_turn_request,
            "batch_tag": "b1",
            "replace_batch_tags": ["b1"],
        }
        assert client.post(SINGLE_TURN_URL, json=self_replace).status_code == 422

        unknown_field = {**single_turn_request, "cases": []}
        assert client.post(SINGLE_TURN_URL, json=unknown_field).status_code == 422

    def test_inputs_bound_is_the_shared_batch_budget(self, single_turn_request):
        """Both sides of the cap, checked on the model: at the cap the request
        is valid, one over is not. Posting the at-cap body would drive a full
        mocked batch for no added signal."""
        at_cap = {
            **single_turn_request,
            "inputs": [f"input {i}" for i in range(NUM_CASES_MAX)],
        }
        parsed = SingleTurnPipelineRequest.model_validate(at_cap)
        assert len(parsed.inputs) == NUM_CASES_MAX

        over_cap = {**single_turn_request, "inputs": at_cap["inputs"] + ["one more"]}
        with pytest.raises(ValidationError):
            SingleTurnPipelineRequest.model_validate(over_cap)


# ───────────────────────── preflight_model ─────────────────────────

PREFLIGHT_URL = "/api/projects/p1/tasks/t1/eval_builder/preflight_model"


@pytest.fixture
def preflight_request():
    return {"model_name": "gpt_5_5", "model_provider": "openrouter"}


@pytest.fixture
def preflight_seams():
    """task_from_id (path validation only) + adapter_for_task (the lane)."""
    with (
        patch(
            "app.desktop.studio_server.eval_builder_api.task_from_id"
        ) as mock_task_from_id,
        patch(
            "app.desktop.studio_server.eval_builder_api.adapter_for_task"
        ) as mock_adapter_for_task,
    ):
        mock_task_from_id.return_value = Mock()
        adapter = Mock()
        adapter.invoke = AsyncMock(return_value=Mock())
        mock_adapter_for_task.return_value = adapter
        yield mock_task_from_id, mock_adapter_for_task, adapter


class TestPreflightModel:
    def test_ok(self, client, preflight_request, preflight_seams):
        _, _, adapter = preflight_seams
        resp = client.post(PREFLIGHT_URL, json=preflight_request)
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        adapter.invoke.assert_awaited_once_with(input="Say OK")

    def test_config_dead_returns_unwrapped_root_error(
        self, client, preflight_request, preflight_seams
    ):
        """A dead lane 400s with the ROOT provider error, not the KilnRunError
        wrapper's genericized message — the stop banner shows this text."""
        _, _, adapter = preflight_seams
        adapter.invoke = AsyncMock(
            side_effect=KilnRunError(
                "An unexpected error occurred.",
                partial_trace=None,
                original=_auth_error(),
            )
        )
        resp = client.post(PREFLIGHT_URL, json=preflight_request)
        assert resp.status_code == 400
        message = resp.json()["message"]["message"]
        assert "AuthenticationError" in message
        assert "invalid api key" in message
        assert "unexpected error" not in message
        # litellm strings already lead with the class name — the route must
        # not stack its own prefix on top ("AuthenticationError: litellm.
        # AuthenticationError: …").
        assert not message.startswith("AuthenticationError: litellm.")

    def test_never_persists_and_never_uses_the_real_task(
        self, client, preflight_request, preflight_seams
    ):
        """No TaskRun may land in the dataset (allow_saving=False), and the
        completion runs against a transient one-liner task, not the user's
        task prompt."""
        mock_task_from_id, mock_adapter_for_task, _ = preflight_seams
        resp = client.post(PREFLIGHT_URL, json=preflight_request)
        assert resp.status_code == 200
        kwargs = mock_adapter_for_task.call_args.kwargs
        args = mock_adapter_for_task.call_args.args
        assert kwargs["base_adapter_config"].allow_saving is False
        preflight_task = args[0]
        assert preflight_task is not mock_task_from_id.return_value
        assert preflight_task.name == "preflight_check"
        rcp = kwargs["run_config_properties"]
        assert rcp.model_name == "gpt_5_5"
        assert rcp.model_provider_name == ModelProviderName.openrouter

    def test_unknown_provider_is_rejected_before_any_call(
        self, client, preflight_request, preflight_seams
    ):
        _, mock_adapter_for_task, _ = preflight_seams
        preflight_request["model_provider"] = "not_a_provider"
        resp = client.post(PREFLIGHT_URL, json=preflight_request)
        assert resp.status_code == 422
        mock_adapter_for_task.assert_not_called()
