"""Tests for LlmJudgeEval adapter."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from litellm.types.utils import ModelResponse

from kiln_ai.adapters.eval.base_eval import materialize_llm_judge_properties
from kiln_ai.adapters.eval.v2_eval_llm_judge import (
    _DEFAULT_SYSTEM_PROMPT,
    LlmJudgeEval,
    _LlmJudgeTask,
)
from kiln_ai.adapters.run_output import RunOutput
from kiln_ai.datamodel.datamodel_enums import TaskOutputRatingType
from kiln_ai.datamodel.eval import (
    Eval,
    EvalConfig,
    EvalConfigType,
    EvalDataType,
    EvalOutputScore,
    EvalTaskInput,
    LlmJudgeProperties,
    SkippedReason,
)
from kiln_ai.datamodel.task import Task
from kiln_ai.datamodel.task_run import Usage


def _make_props(**overrides) -> LlmJudgeProperties:
    defaults = {
        "model_name": "gpt-4o",
        "model_provider": "openai",
        "prompt_template": "Rate this: {{ final_message }}",
    }
    defaults.update(overrides)
    return LlmJudgeProperties(**defaults)


def _judge_run(usage: Usage | None = None) -> Mock:
    """The judge's own TaskRun, the first half of what the adapter returns.

    Only `usage` is read from it — that is what becomes `V2EvalResult.usage`, and it must
    be a real `Usage` or None because the result model validates it.
    """
    run = Mock()
    run.usage = usage
    return run


def _make_config(props: LlmJudgeProperties | None = None) -> EvalConfig:
    if props is None:
        props = _make_props()
    parent = Mock()
    parent.output_scores = [
        EvalOutputScore(
            name="quality",
            instruction="Rate quality",
            type=TaskOutputRatingType.five_star,
        ),
    ]
    cfg = Mock(spec=EvalConfig)
    cfg.config_type = EvalConfigType.v2
    cfg.properties = props
    cfg.parent_eval.return_value = parent
    cfg.model_name = None
    cfg.model_provider = None
    return cfg


def _inp(**overrides) -> EvalTaskInput:
    defaults: dict = {
        "final_message": "Hello world",
        "trace": None,
        "reference_data": None,
        "task_input": None,
    }
    defaults.update(overrides)
    return EvalTaskInput(**defaults)


class TestLlmJudgeEvalInit:
    def test_valid_construction(self):
        cfg = _make_config()
        adapter = LlmJudgeEval(cfg)
        assert adapter.properties is cfg.properties

    def test_invalid_provider_raises(self):
        props = _make_props(model_provider="not_a_real_provider")
        cfg = _make_config(props)
        with pytest.raises(ValueError, match="Invalid model provider"):
            LlmJudgeEval(cfg)

    def test_non_llm_judge_properties_raises(self):
        cfg = _make_config()
        cfg.properties = "not_properties"
        with pytest.raises(ValueError):
            LlmJudgeEval(cfg)


_VALID_SCHEMA = '{"type": "object", "properties": {"quality": {"type": "string"}}, "required": ["quality"]}'


class TestLlmJudgeTask:
    def test_basic_creation(self):
        task = _LlmJudgeTask(
            system_prompt="You are a judge.",
            output_json_schema=_VALID_SCHEMA,
        )
        assert task.instruction == "You are a judge."
        assert task.output_json_schema == _VALID_SCHEMA


class TestLlmJudgeEvalLlmAsJudge:
    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_basic_scoring(self, mock_adapter_for_task):
        mock_adapter = AsyncMock()
        mock_run_output = RunOutput(
            output={"quality": "4"},
            intermediate_outputs=None,
        )
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            mock_run_output,
        )
        mock_adapter_for_task.return_value = mock_adapter

        cfg = _make_config()
        result = await LlmJudgeEval(cfg).evaluate(_inp())

        assert result.scores == {"quality": 4.0}
        assert result.skipped_reason is None
        assert result.skipped_detail is None
        assert result.usage is None

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_judge_usage_is_reported(self, mock_adapter_for_task):
        """What the judgment cost. Nothing recorded it before eval_usage existed."""
        judge_usage = Usage(input_tokens=120, output_tokens=8, cost=0.004)
        mock_adapter = AsyncMock()
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(judge_usage),
            RunOutput(output={"quality": "4"}, intermediate_outputs=None),
        )
        mock_adapter_for_task.return_value = mock_adapter

        result = await LlmJudgeEval(_make_config()).evaluate(_inp())

        assert result.usage == judge_usage

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_intermediate_outputs_propagated(self, mock_adapter_for_task):
        mock_adapter = AsyncMock()
        mock_run_output = RunOutput(
            output={"quality": "4"},
            intermediate_outputs={
                "chain_of_thought": "The answer is correct because it addresses all key points."
            },
        )
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            mock_run_output,
        )
        mock_adapter_for_task.return_value = mock_adapter

        cfg = _make_config()
        result = await LlmJudgeEval(cfg).evaluate(_inp())

        assert result.scores == {"quality": 4.0}
        assert result.intermediate_outputs == {
            "chain_of_thought": "The answer is correct because it addresses all key points."
        }

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_pass_fail_scoring(self, mock_adapter_for_task):
        mock_adapter = AsyncMock()
        mock_run_output = RunOutput(
            output={"quality": "pass"},
            intermediate_outputs=None,
        )
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            mock_run_output,
        )
        mock_adapter_for_task.return_value = mock_adapter

        cfg = _make_config()
        result = await LlmJudgeEval(cfg).evaluate(_inp())

        assert result.scores == {"quality": 1.0}
        assert result.skipped_reason is None

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_template_rendering(self, mock_adapter_for_task):
        mock_adapter = AsyncMock()
        mock_run_output = RunOutput(
            output={"quality": "3"},
            intermediate_outputs=None,
        )
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            mock_run_output,
        )
        mock_adapter_for_task.return_value = mock_adapter

        props = _make_props(
            prompt_template="Input: {{ task_input }}, Output: {{ final_message }}"
        )
        cfg = _make_config(props)
        await LlmJudgeEval(cfg).evaluate(
            _inp(final_message="answer", task_input="question")
        )

        call_args = mock_adapter.invoke_returning_run_output.call_args
        rendered = call_args[0][0]
        assert "Input: question" in rendered
        assert "Output: answer" in rendered

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_custom_system_prompt(self, mock_adapter_for_task):
        mock_adapter = AsyncMock()
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            RunOutput(output={"quality": "5"}, intermediate_outputs=None),
        )
        mock_adapter_for_task.return_value = mock_adapter

        props = _make_props(system_prompt="Custom instructions here.")
        cfg = _make_config(props)
        await LlmJudgeEval(cfg).evaluate(_inp())

        task_arg = mock_adapter_for_task.call_args[0][0]
        assert task_arg.instruction == "Custom instructions here."

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_default_system_prompt(self, mock_adapter_for_task):
        mock_adapter = AsyncMock()
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            RunOutput(output={"quality": "5"}, intermediate_outputs=None),
        )
        mock_adapter_for_task.return_value = mock_adapter

        props = _make_props(system_prompt=None)
        cfg = _make_config(props)
        await LlmJudgeEval(cfg).evaluate(_inp())

        task_arg = mock_adapter_for_task.call_args[0][0]
        assert task_arg.instruction == _DEFAULT_SYSTEM_PROMPT

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_adapter_config_llm_as_judge(self, mock_adapter_for_task):
        mock_adapter = AsyncMock()
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            RunOutput(output={"quality": "5"}, intermediate_outputs=None),
        )
        mock_adapter_for_task.return_value = mock_adapter

        props = _make_props(g_eval=False)
        cfg = _make_config(props)
        await LlmJudgeEval(cfg).evaluate(_inp())

        call_kwargs = mock_adapter_for_task.call_args[1]
        adapter_config = call_kwargs["base_adapter_config"]
        assert adapter_config.allow_saving is False
        assert adapter_config.top_logprobs is None

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_invalid_score_raises(self, mock_adapter_for_task):
        mock_adapter = AsyncMock()
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            RunOutput(output={"quality": "garbage"}, intermediate_outputs=None),
        )
        mock_adapter_for_task.return_value = mock_adapter

        cfg = _make_config()
        with pytest.raises(ValueError, match="No score found for metric"):
            await LlmJudgeEval(cfg).evaluate(_inp())

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_non_dict_output_raises(self, mock_adapter_for_task):
        mock_adapter = AsyncMock()
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            RunOutput(output="just a string", intermediate_outputs=None),
        )
        mock_adapter_for_task.return_value = mock_adapter

        cfg = _make_config()
        with pytest.raises(ValueError, match="must be a dictionary"):
            await LlmJudgeEval(cfg).evaluate(_inp())

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.BaseEval.build_score_schema")
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_discrete_score_schema(
        self, mock_adapter_for_task, mock_build_schema
    ):
        mock_build_schema.return_value = _VALID_SCHEMA
        mock_adapter = AsyncMock()
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            RunOutput(output={"quality": "3"}, intermediate_outputs=None),
        )
        mock_adapter_for_task.return_value = mock_adapter

        cfg = _make_config()
        await LlmJudgeEval(cfg).evaluate(_inp())

        discrete_calls = [
            c
            for c in mock_build_schema.call_args_list
            if c[1].get("allow_float_scores") is False
        ]
        assert len(discrete_calls) == 1


class TestLlmJudgeEvalGEval:
    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_adapter_config_g_eval(self, mock_adapter_for_task):
        mock_adapter = AsyncMock()
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            RunOutput(output={"quality": "5"}, intermediate_outputs=None),
        )
        mock_adapter_for_task.return_value = mock_adapter

        props = _make_props(g_eval=True)
        cfg = _make_config(props)

        with (
            patch(
                "kiln_ai.adapters.eval.v2_eval_llm_judge.built_in_models_from_provider",
                return_value=Mock(supports_logprobs=True),
            ),
            patch(
                "kiln_ai.adapters.eval.v2_eval_llm_judge.build_g_eval_score"
            ) as mock_g_eval_score,
        ):
            mock_g_eval_score.return_value = {"quality": 4.3}
            result = await LlmJudgeEval(cfg).evaluate(_inp())

        assert result.scores == {"quality": 4.3}
        assert result.skipped_reason is None

        call_kwargs = mock_adapter_for_task.call_args[1]
        adapter_config = call_kwargs["base_adapter_config"]
        assert adapter_config.top_logprobs == 10

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_g_eval_calls_scoring_functions(self, mock_adapter_for_task):
        mock_adapter = AsyncMock()
        mock_run_output = RunOutput(
            output={"quality": "5"},
            intermediate_outputs=None,
        )
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            mock_run_output,
        )
        mock_adapter_for_task.return_value = mock_adapter

        props = _make_props(g_eval=True)
        cfg = _make_config(props)

        with (
            patch(
                "kiln_ai.adapters.eval.v2_eval_llm_judge.built_in_models_from_provider",
                return_value=Mock(supports_logprobs=True),
            ),
            patch(
                "kiln_ai.adapters.eval.v2_eval_llm_judge.build_g_eval_score"
            ) as mock_g_eval_score,
        ):
            mock_g_eval_score.return_value = {"quality": 3.7}
            await LlmJudgeEval(cfg).evaluate(_inp())

        mock_g_eval_score.assert_called_once()
        call_args = mock_g_eval_score.call_args[0]
        assert call_args[0] == mock_run_output


class TestLlmJudgeEvalGEvalFailFast:
    @pytest.mark.asyncio
    async def test_g_eval_raises_when_provider_lacks_logprobs(self):
        mock_provider = Mock()
        mock_provider.supports_logprobs = False
        props = _make_props(g_eval=True)
        cfg = _make_config(props)
        adapter = LlmJudgeEval(cfg)

        with patch(
            "kiln_ai.adapters.eval.v2_eval_llm_judge.built_in_models_from_provider",
            return_value=mock_provider,
        ):
            with pytest.raises(ValueError, match="logprobs support"):
                await adapter.evaluate(_inp())

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_g_eval_proceeds_when_provider_supports_logprobs(
        self, mock_adapter_for_task
    ):
        mock_provider = Mock()
        mock_provider.supports_logprobs = True

        mock_adapter = AsyncMock()
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            RunOutput(output={"quality": "5"}, intermediate_outputs=None),
        )
        mock_adapter_for_task.return_value = mock_adapter

        props = _make_props(g_eval=True)
        cfg = _make_config(props)

        with (
            patch(
                "kiln_ai.adapters.eval.v2_eval_llm_judge.built_in_models_from_provider",
                return_value=mock_provider,
            ),
            patch(
                "kiln_ai.adapters.eval.v2_eval_llm_judge.build_g_eval_score"
            ) as mock_g_eval_score,
        ):
            mock_g_eval_score.return_value = {"quality": 4.3}
            result = await LlmJudgeEval(cfg).evaluate(_inp())

        assert result.scores == {"quality": 4.3}
        assert result.skipped_reason is None

    @pytest.mark.asyncio
    async def test_g_eval_raises_when_model_unknown(self):
        # Unknown model: logprobs support can't be verified, so the preflight
        # must fail loudly before spending on the judge call.
        props = _make_props(g_eval=True)
        cfg = _make_config(props)
        adapter = LlmJudgeEval(cfg)

        with patch(
            "kiln_ai.adapters.eval.v2_eval_llm_judge.built_in_models_from_provider",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="not a built-in model"):
                await adapter.evaluate(_inp())


class TestLlmJudgeEvalMissingReferenceData:
    @pytest.mark.asyncio
    async def test_undefined_reference_key_skips_cleanly(self):
        props = _make_props(
            prompt_template="Answer: {{ reference_data.answer }} Output: {{ final_message }}"
        )
        cfg = _make_config(props)
        result = await LlmJudgeEval(cfg).evaluate(_inp(reference_data=None))
        assert result.scores == {}
        assert result.skipped_reason == SkippedReason.missing_reference_key
        assert result.skipped_detail is not None
        assert "missing data" in result.skipped_detail.lower()

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_present_reference_data_proceeds(self, mock_adapter_for_task):
        mock_adapter = AsyncMock()
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            RunOutput(output={"quality": "3"}, intermediate_outputs=None),
        )
        mock_adapter_for_task.return_value = mock_adapter

        props = _make_props(
            prompt_template="Answer: {{ reference_data.answer }} Output: {{ final_message }}"
        )
        cfg = _make_config(props)
        result = await LlmJudgeEval(cfg).evaluate(
            _inp(reference_data={"answer": "correct"})
        )
        assert result.scores == {"quality": 3.0}
        assert result.skipped_reason is None

    @pytest.mark.asyncio
    async def test_unknown_template_variable_is_extraction_failure(self):
        # A typo'd variable is a template-authoring bug, not missing reference
        # data -- it must not masquerade as missing_reference_key.
        props = _make_props(prompt_template="Output: {{ final_mesage }}")
        cfg = _make_config(props)
        result = await LlmJudgeEval(cfg).evaluate(_inp())
        assert result.scores == {}
        assert result.skipped_reason == SkippedReason.extraction_failed
        assert result.skipped_detail is not None
        assert "unknown variable" in result.skipped_detail
        assert "final_mesage" in result.skipped_detail

    @pytest.mark.asyncio
    async def test_typo_alongside_reference_access_reports_unknown_variable(self):
        # When the template has both a typo and a missing reference key, the
        # authoring bug is the primary truth to surface.
        props = _make_props(
            prompt_template="{{ reference_data.answer }} vs {{ outpt }}"
        )
        cfg = _make_config(props)
        result = await LlmJudgeEval(cfg).evaluate(_inp(reference_data=None))
        assert result.skipped_reason == SkippedReason.extraction_failed
        assert result.skipped_detail is not None
        assert "outpt" in result.skipped_detail


class TestLlmJudgeEvalDeclaredReferenceKeys:
    """`reference_keys` is a declared requirement: a judge that cannot get one skips
    loudly instead of scoring against ground truth it never saw."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reference_data, expected_detail",
        [
            (None, "No reference_data"),
            ({"other": "x"}, "missing key 'reference_answer'"),
            ({"reference_answer": None}, "key 'reference_answer' is None"),
        ],
        ids=["no-reference-data", "wrong-key", "null-value"],
    )
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_missing_declared_key_skips_without_calling_the_model(
        self, mock_adapter_for_task, reference_data, expected_detail
    ):
        props = _make_props(reference_keys=["reference_answer"])
        cfg = _make_config(props)

        result = await LlmJudgeEval(cfg).evaluate(_inp(reference_data=reference_data))

        assert result.scores == {}
        assert result.skipped_reason == SkippedReason.missing_reference_key
        assert result.skipped_detail is not None
        assert expected_detail in result.skipped_detail
        mock_adapter_for_task.assert_not_called()

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_skips_on_the_first_unsatisfied_key(self, mock_adapter_for_task):
        props = _make_props(reference_keys=["reference_answer", "retrieved_context"])
        cfg = _make_config(props)

        result = await LlmJudgeEval(cfg).evaluate(
            _inp(reference_data={"reference_answer": "Frank Herbert."})
        )

        assert result.skipped_reason == SkippedReason.missing_reference_key
        assert result.skipped_detail is not None
        assert "retrieved_context" in result.skipped_detail
        mock_adapter_for_task.assert_not_called()

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_satisfied_keys_score_normally(self, mock_adapter_for_task):
        mock_adapter = AsyncMock()
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            RunOutput(output={"quality": "5"}, intermediate_outputs=None),
        )
        mock_adapter_for_task.return_value = mock_adapter

        props = _make_props(
            reference_keys=["reference_answer"],
            prompt_template=(
                "Reference: {{ reference_data.reference_answer }} "
                "Output: {{ final_message }}"
            ),
        )
        cfg = _make_config(props)

        result = await LlmJudgeEval(cfg).evaluate(
            _inp(reference_data={"reference_answer": "Frank Herbert."})
        )

        assert result.scores == {"quality": 5.0}
        assert result.skipped_reason is None
        rendered = mock_adapter.invoke_returning_run_output.call_args[0][0]
        assert "Frank Herbert." in rendered

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_no_declared_keys_does_not_require_reference_data(
        self, mock_adapter_for_task
    ):
        """A judge that declares nothing is not required to have anything: the
        requirement is the declaration, not the presence of reference data."""
        mock_adapter = AsyncMock()
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            RunOutput(output={"quality": "2"}, intermediate_outputs=None),
        )
        mock_adapter_for_task.return_value = mock_adapter

        cfg = _make_config(_make_props())
        result = await LlmJudgeEval(cfg).evaluate(_inp(reference_data=None))

        assert result.scores == {"quality": 2.0}
        assert result.skipped_reason is None


class TestBackendBakedReferenceAnswerJudge:
    """The declared-key guard is only worth having if a shipped path reaches it.
    `materialize_llm_judge_properties` is that path: it bakes both the prompt block and
    the declaration for a reference-answer eval, so these tests use the real properties
    rather than hand-written ones."""

    @staticmethod
    def _baked_props() -> LlmJudgeProperties:
        task = Task(name="QnA Task", instruction="Answer questions about the docs")
        eval_obj = Eval(
            name="Reference Answer Eval",
            evaluation_data_type=EvalDataType.reference_answer,
            output_scores=[
                EvalOutputScore(
                    name="quality",
                    instruction="Rate quality",
                    type=TaskOutputRatingType.five_star,
                )
            ],
            eval_set_filter_id="tag::test",
            parent=task,
        )
        return materialize_llm_judge_properties(
            eval_obj, "gpt-4o", "openai", g_eval=False
        )

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_item_without_a_reference_skips_before_the_model_call(
        self, mock_adapter_for_task
    ):
        """Judge calibration scores the golden item as itself, so it supplies no
        reference data by design — every calibration item of such a judge skips. That
        is the right answer for "does this match the reference" with no reference, and
        it costs no inference."""
        cfg = _make_config(self._baked_props())

        result = await LlmJudgeEval(cfg).evaluate(_inp(reference_data=None))

        assert result.scores == {}
        assert result.skipped_reason == SkippedReason.missing_reference_key
        mock_adapter_for_task.assert_not_called()

    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_the_reference_reaches_the_rendered_prompt(
        self, mock_adapter_for_task
    ):
        mock_adapter = AsyncMock()
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            RunOutput(output={"quality": "5"}, intermediate_outputs=None),
        )
        mock_adapter_for_task.return_value = mock_adapter
        cfg = _make_config(self._baked_props())

        result = await LlmJudgeEval(cfg).evaluate(
            _inp(
                final_message="Herbert, 1965.",
                task_input="Who wrote Dune?",
                reference_data={"reference_answer": "Frank Herbert."},
            )
        )

        assert result.skipped_reason is None
        rendered = mock_adapter.invoke_returning_run_output.call_args[0][0]
        assert "<reference_answer>\nFrank Herbert.\n</reference_answer>" in rendered


class TestLlmJudgeEvalNoParentEval:
    def test_no_parent_eval_raises(self):
        cfg = _make_config()
        cfg.parent_eval.return_value = None
        with pytest.raises(ValueError, match="parent eval"):
            LlmJudgeEval(cfg)


class TestLlmJudgeEvalMultipleScores:
    @pytest.mark.asyncio
    @patch("kiln_ai.adapters.eval.v2_eval_llm_judge.adapter_for_task")
    async def test_multiple_metrics(self, mock_adapter_for_task):
        parent = Mock()
        parent.output_scores = [
            EvalOutputScore(
                name="quality",
                instruction="Rate quality",
                type=TaskOutputRatingType.five_star,
            ),
            EvalOutputScore(
                name="relevance",
                instruction="Rate relevance",
                type=TaskOutputRatingType.pass_fail,
            ),
        ]
        cfg = _make_config()
        cfg.parent_eval.return_value = parent

        mock_adapter = AsyncMock()
        mock_adapter.invoke_returning_run_output.return_value = (
            _judge_run(),
            RunOutput(
                output={"quality": "4", "relevance": "pass"},
                intermediate_outputs=None,
            ),
        )
        mock_adapter_for_task.return_value = mock_adapter

        result = await LlmJudgeEval(cfg).evaluate(_inp())
        assert result.scores == {"quality": 4.0, "relevance": 1.0}
        assert result.skipped_reason is None


class TestLlmJudgeEvalTemplateRenderError:
    @pytest.mark.asyncio
    async def test_fromjson_on_non_json_skips_cleanly(self):
        props = _make_props(
            prompt_template="Result: {{ (final_message | fromjson).status }}"
        )
        cfg = _make_config(props)
        result = await LlmJudgeEval(cfg).evaluate(_inp(final_message="not valid json"))
        assert result.scores == {}
        assert result.skipped_reason == SkippedReason.extraction_failed
        assert result.skipped_detail is not None
        assert "not valid JSON" in result.skipped_detail


class TestLlmJudgeE2EMessageConstruction:
    """End-to-end test exercising the REAL adapter/chat-formatter pipeline.

    Mocks only at the model boundary (litellm.acompletion) to verify that
    the messages sent to the LLM are exactly the rendered prompt (no
    <user_input> wrapping, no appended 'Think step by step' instruction)
    and the system message equals the configured system_prompt.

    g_eval variant omitted: logprob mocking is non-trivial; g_eval scoring
    is covered by existing unit tests and shares the same message path.
    """

    _SYSTEM_PROMPT = "You are an expert code reviewer. Be strict."
    _PROMPT_TEMPLATE = (
        "Task: {{ task_input }}\n"
        "Response: {{ final_message }}\n"
        "Trace: {{ trace | tojson }}\n"
        "Ref: {{ reference_data.expected_answer }}"
    )
    _TASK_INPUT = "Implement a binary search function."
    _FINAL_MESSAGE = "def binary_search(arr, target): ..."

    @staticmethod
    def _trace() -> list[dict[str, str]]:
        return [
            {"role": "user", "content": "Implement binary search"},
            {"role": "assistant", "content": "def binary_search(arr, target): ..."},
        ]

    @staticmethod
    def _reference_data() -> dict[str, str]:
        return {"expected_answer": "A correct O(log n) implementation"}

    def _build_config(self) -> EvalConfig:
        props = LlmJudgeProperties(
            model_name="gpt-4o",
            model_provider="openai",
            prompt_template=self._PROMPT_TEMPLATE,
            system_prompt=self._SYSTEM_PROMPT,
        )
        parent = Mock()
        parent.output_scores = [
            EvalOutputScore(
                name="quality",
                instruction="Rate quality",
                type=TaskOutputRatingType.five_star,
            ),
            EvalOutputScore(
                name="correctness",
                instruction="Is the answer correct?",
                type=TaskOutputRatingType.pass_fail,
            ),
        ]
        cfg = Mock(spec=EvalConfig)
        cfg.config_type = EvalConfigType.v2
        cfg.properties = props
        cfg.parent_eval.return_value = parent
        cfg.model_name = None
        cfg.model_provider = None
        return cfg

    def _build_input(self) -> EvalTaskInput:
        return EvalTaskInput(
            task_input=self._TASK_INPUT,
            final_message=self._FINAL_MESSAGE,
            trace=self._trace(),
            reference_data=self._reference_data(),
        )

    def _expected_user_message(self) -> str:
        trace_json = json.dumps(self._trace(), sort_keys=True)
        ref = self._reference_data()
        return (
            f"Task: {self._TASK_INPUT}\n"
            f"Response: {self._FINAL_MESSAGE}\n"
            f"Trace: {trace_json}\n"
            f"Ref: {ref['expected_answer']}"
        )

    @pytest.mark.asyncio
    async def test_llm_as_judge_messages_sent_verbatim(self):
        """The user message must be the rendered prompt verbatim — no wrapping."""
        model_response = ModelResponse(
            model="gpt-4o",
            choices=[
                {
                    "message": {
                        "content": json.dumps({"quality": 4, "correctness": "pass"}),
                    }
                }
            ],
        )

        captured_kwargs: list[dict] = []

        async def fake_acompletion(**kwargs):
            captured_kwargs.append(kwargs)
            return model_response

        mock_config_obj = Mock()
        mock_config_obj.open_ai_api_key = "fake-key"
        mock_config_obj.user_id = "test_user"

        cfg = self._build_config()
        eval_input = self._build_input()

        with (
            patch("litellm.acompletion", side_effect=fake_acompletion),
            patch(
                "kiln_ai.utils.config.Config.shared",
                return_value=mock_config_obj,
            ),
        ):
            result = await LlmJudgeEval(cfg).evaluate(eval_input)

        assert len(captured_kwargs) == 1
        messages = captured_kwargs[0]["messages"]

        system_msgs = [m for m in messages if m["role"] == "system"]
        user_msgs = [m for m in messages if m["role"] == "user"]

        assert len(system_msgs) == 1
        assert len(user_msgs) == 1

        system_content = system_msgs[0]["content"]
        user_content = user_msgs[0]["content"]

        assert system_content == self._SYSTEM_PROMPT
        assert user_content == self._expected_user_message()

        assert "The input is:" not in user_content
        assert "<user_input>" not in user_content
        assert "Think step by step" not in user_content

        assert result.scores == {"quality": 4.0, "correctness": 1.0}
        assert result.skipped_reason is None
