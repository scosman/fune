from typing import ClassVar
from unittest.mock import Mock

import pytest

from kiln_ai.adapters.eval.g_eval import GEval
from kiln_ai.adapters.eval.registry import (
    legacy_eval_adapter_from_type,
    v2_eval_adapter_from_config,
    v2_eval_type_available,
)
from kiln_ai.datamodel.eval import (
    CodeEvalProperties,
    ContainsProperties,
    EvalConfig,
    EvalConfigType,
    EvalOutputScore,
    ExactMatchProperties,
    LlmJudgeProperties,
    PatternMatchProperties,
    SetCheckProperties,
    StepCountCheckProperties,
    ToolCallCheckProperties,
    ToolCallSpec,
    V2EvalType,
)
from kiln_ai.datamodel.task import TaskOutputRatingType

_V2_TYPE_TO_PROPS = {
    V2EvalType.llm_judge: LlmJudgeProperties(
        model_name="gpt-4o",
        model_provider="openai",
        prompt_template="{{ final_message }}",
    ),
    V2EvalType.exact_match: ExactMatchProperties(expected_value="test"),
    V2EvalType.pattern_match: PatternMatchProperties(pattern=".*"),
    V2EvalType.contains: ContainsProperties(substring="test"),
    V2EvalType.set_check: SetCheckProperties(expected_set=["a"], mode="subset"),
    V2EvalType.tool_call_check: ToolCallCheckProperties(
        expected_tools=[ToolCallSpec(tool_name="t")],
    ),
    V2EvalType.step_count_check: StepCountCheckProperties(
        count_type="tool_calls", min_count=1
    ),
    V2EvalType.code_eval: CodeEvalProperties(
        code="def score(output, trace, reference_data, task_input):\n    return {'score': 1.0}\n"
    ),
}


def _mock_eval_config(config_type: EvalConfigType) -> EvalConfig:
    cfg = Mock(spec=EvalConfig)
    cfg.config_type = config_type
    return cfg


def _mock_v2_eval_config(v2_type: V2EvalType) -> EvalConfig:
    props = _V2_TYPE_TO_PROPS[v2_type]
    parent = Mock()
    parent.output_scores = [
        EvalOutputScore(
            name="score", instruction="s", type=TaskOutputRatingType.pass_fail
        ),
    ]
    cfg = Mock(spec=EvalConfig)
    cfg.config_type = EvalConfigType.v2
    cfg.properties = props
    cfg.parent_eval.return_value = parent
    return cfg


class TestLegacyEvalAdapterFromType:
    @pytest.mark.parametrize(
        "config_type", [EvalConfigType.g_eval, EvalConfigType.llm_as_judge]
    )
    def test_legacy_types_return_geval(self, config_type: EvalConfigType):
        cfg = _mock_eval_config(config_type)
        assert legacy_eval_adapter_from_type(cfg) is GEval

    def test_v2_raises_not_implemented(self):
        cfg = _mock_eval_config(EvalConfigType.v2)
        with pytest.raises(
            NotImplementedError,
            match="V2 eval configs should use v2_eval_adapter_from_config",
        ):
            legacy_eval_adapter_from_type(cfg)


class TestV2EvalAdapterFromConfig:
    _IMPLEMENTED_V2_TYPES: ClassVar[list[V2EvalType]] = [
        V2EvalType.exact_match,
        V2EvalType.pattern_match,
        V2EvalType.contains,
        V2EvalType.set_check,
        V2EvalType.tool_call_check,
        V2EvalType.step_count_check,
        V2EvalType.llm_judge,
        V2EvalType.code_eval,
    ]

    @pytest.mark.parametrize("v2_type", _IMPLEMENTED_V2_TYPES)
    def test_v2_dispatch_implemented_types(self, v2_type: V2EvalType):
        cfg = _mock_v2_eval_config(v2_type)
        adapter = v2_eval_adapter_from_config(cfg)
        assert adapter is not None

    def test_v2_dispatch_rejects_legacy_config(self):
        cfg = _mock_eval_config(EvalConfigType.g_eval)
        with pytest.raises(ValueError, match="only accepts V2 configs"):
            v2_eval_adapter_from_config(cfg)


class TestV2EvalTypeAvailable:
    """The pure availability probe: True exactly when dispatch would succeed."""

    @pytest.mark.parametrize(
        "v2_type", TestV2EvalAdapterFromConfig._IMPLEMENTED_V2_TYPES
    )
    def test_registered_types_available(self, v2_type: V2EvalType):
        assert v2_eval_type_available(_mock_v2_eval_config(v2_type)) is True

    def test_legacy_config_not_available(self):
        assert v2_eval_type_available(_mock_eval_config(EvalConfigType.g_eval)) is False

    def test_untyped_properties_not_available(self):
        cfg = _mock_eval_config(EvalConfigType.v2)
        cfg.properties = {"type": "llm_judge"}
        assert v2_eval_type_available(cfg) is False
