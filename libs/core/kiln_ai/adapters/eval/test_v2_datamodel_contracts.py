"""Contract tests for the V2 eval datamodel and its scoring helpers.

Covers:
- V2_PROPERTY_TYPES explicit tuple stays in sync with the V2EvalConfigProperties union
- build_binary_scores accepts output_scores directly (no EvalConfig needed)
- V2 adapters cache output scores at init instead of calling parent_eval() per item
- EvalRun validate_input_source error message clarifies V1 vs V2 sources
"""

from typing import get_args
from unittest.mock import Mock

import pytest

from kiln_ai.adapters.eval.eval_utils.v2_eval_helpers import build_binary_scores
from kiln_ai.adapters.eval.v2_eval_exact_match import ExactMatchEval
from kiln_ai.datamodel.datamodel_enums import TaskOutputRatingType
from kiln_ai.datamodel.eval import (
    V2_PROPERTY_TYPES,
    EvalConfig,
    EvalConfigType,
    EvalOutputScore,
    EvalRun,
    EvalTaskInput,
    ExactMatchProperties,
    V2EvalConfigProperties,
)

# ---------------------------------------------------------------------------
# V2_PROPERTY_TYPES matches V2EvalConfigProperties union
# ---------------------------------------------------------------------------


class TestV2PropertyTypes:
    def test_contains_all_union_members(self):
        """The explicit tuple must list every type in the V2EvalConfigProperties union."""
        union_type = get_args(V2EvalConfigProperties)[0]
        union_members = set(get_args(union_type))
        explicit_members = set(V2_PROPERTY_TYPES)
        assert explicit_members == union_members

    def test_no_extra_members(self):
        """The explicit tuple must not contain types absent from the union."""
        union_type = get_args(V2EvalConfigProperties)[0]
        union_members = set(get_args(union_type))
        for t in V2_PROPERTY_TYPES:
            assert t in union_members

    def test_isinstance_works(self):
        """isinstance() checks against the tuple should work correctly."""
        props = ExactMatchProperties(expected_value="hello")
        assert isinstance(props, V2_PROPERTY_TYPES)


# ---------------------------------------------------------------------------
# build_binary_scores takes output_scores directly
# ---------------------------------------------------------------------------


class TestBuildBinaryScores:
    def test_pass_returns_ones(self):
        scores_def = [
            EvalOutputScore(
                name="score_a", instruction="a", type=TaskOutputRatingType.pass_fail
            ),
            EvalOutputScore(
                name="score_b", instruction="b", type=TaskOutputRatingType.pass_fail
            ),
        ]
        result = build_binary_scores(scores_def, passed=True)
        assert result == {"score_a": 1.0, "score_b": 1.0}

    def test_fail_returns_zeros(self):
        scores_def = [
            EvalOutputScore(
                name="score_a", instruction="a", type=TaskOutputRatingType.pass_fail
            ),
        ]
        result = build_binary_scores(scores_def, passed=False)
        assert result == {"score_a": 0.0}

    def test_empty_scores_returns_empty(self):
        result = build_binary_scores([], passed=True)
        assert result == {}

    def test_no_eval_config_needed(self):
        """build_binary_scores works from output_scores alone -- no EvalConfig or parent_eval() call."""
        scores_def = [
            EvalOutputScore(
                name="check", instruction="test", type=TaskOutputRatingType.pass_fail
            ),
        ]
        result = build_binary_scores(scores_def, passed=True)
        assert result == {"check": 1.0}


class TestCachedOutputScores:
    """Verify that V2 adapters cache _output_scores at init and don't call parent_eval() per item."""

    @pytest.mark.asyncio
    async def test_adapter_caches_output_scores(self):
        parent = Mock()
        parent.output_scores = [
            EvalOutputScore(
                name="s", instruction="i", type=TaskOutputRatingType.pass_fail
            ),
        ]
        cfg = Mock(spec=EvalConfig)
        cfg.config_type = EvalConfigType.v2
        cfg.properties = ExactMatchProperties(expected_value="hello")
        cfg.parent_eval.return_value = parent
        parent.parent_task.return_value = Mock()

        adapter = ExactMatchEval(cfg)
        assert adapter._output_scores is parent.output_scores

        cfg.parent_eval.reset_mock()

        inp = EvalTaskInput(final_message="hello")
        result = await adapter.evaluate(inp)
        assert result.scores == {"s": 1.0}
        assert result.skipped_reason is None

        cfg.parent_eval.assert_not_called()


# ---------------------------------------------------------------------------
# validate_input_source error message
# ---------------------------------------------------------------------------


class TestInputSourceErrorMessage:
    def test_both_set_mentions_v1_v2(self):
        with pytest.raises(ValueError, match=r"V1 TaskRun source") as exc_info:
            EvalRun(
                dataset_id="d1",
                eval_input_id="ei1",
                task_run_config_id="rc1",
                input="i",
                output="o",
                scores={"s": 1.0},
            )
        assert "V2 EvalInput source" in str(exc_info.value)

    def test_neither_set_mentions_v1_v2(self):
        with pytest.raises(ValueError, match=r"V1 TaskRun source") as exc_info:
            EvalRun(
                task_run_config_id="rc1",
                input="i",
                output="o",
                scores={"s": 1.0},
            )
        assert "V2 EvalInput source" in str(exc_info.value)
