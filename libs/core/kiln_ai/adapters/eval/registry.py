from typing import TYPE_CHECKING

from kiln_ai.adapters.eval.base_eval import BaseEval, BaseV2EvalBridge

if TYPE_CHECKING:
    from kiln_ai.adapters.model_adapters.base_adapter import SkillsDict
    from kiln_ai.datamodel.task import RunConfigProperties
from kiln_ai.adapters.eval.g_eval import GEval
from kiln_ai.adapters.eval.v2_eval_code_eval import CodeEvalAdapter
from kiln_ai.adapters.eval.v2_eval_contains import ContainsEval
from kiln_ai.adapters.eval.v2_eval_exact_match import ExactMatchEval
from kiln_ai.adapters.eval.v2_eval_llm_judge import LlmJudgeEval
from kiln_ai.adapters.eval.v2_eval_pattern_match import PatternMatchEval
from kiln_ai.adapters.eval.v2_eval_set_check import SetCheckEval
from kiln_ai.adapters.eval.v2_eval_step_count_check import StepCountCheckEval
from kiln_ai.adapters.eval.v2_eval_tool_call_check import ToolCallCheckEval
from kiln_ai.datamodel.eval import (
    V2_PROPERTY_TYPES,
    EvalConfig,
    EvalConfigType,
    V2EvalType,
)
from kiln_ai.utils.exhaustive_error import raise_exhaustive_enum_error

_V2_ADAPTER_MAP: dict[V2EvalType, type[BaseV2EvalBridge]] = {
    V2EvalType.exact_match: ExactMatchEval,
    V2EvalType.pattern_match: PatternMatchEval,
    V2EvalType.contains: ContainsEval,
    V2EvalType.set_check: SetCheckEval,
    V2EvalType.tool_call_check: ToolCallCheckEval,
    V2EvalType.step_count_check: StepCountCheckEval,
    V2EvalType.llm_judge: LlmJudgeEval,
    V2EvalType.code_eval: CodeEvalAdapter,
}


def v2_eval_type_available(eval_config: EvalConfig) -> bool:
    """Whether this config's V2 type has a registered adapter — the pure
    availability probe behind the type_not_available skip, safe to call
    during job collection (no adapter is instantiated)."""
    if eval_config.config_type != EvalConfigType.v2:
        return False
    if not isinstance(eval_config.properties, V2_PROPERTY_TYPES):
        return False
    return eval_config.properties.type in _V2_ADAPTER_MAP  # type: ignore[union-attr]


def legacy_eval_adapter_from_type(eval_config: EvalConfig) -> type[BaseEval]:
    """Legacy dispatch -- returns a BaseEval subclass for g_eval/llm_as_judge.

    For v2, raises NotImplementedError (v2 adapters use v2_eval_adapter_from_config).
    """
    match eval_config.config_type:
        case EvalConfigType.g_eval:
            return GEval
        case EvalConfigType.llm_as_judge:
            return GEval
        case EvalConfigType.v2:
            raise NotImplementedError(
                "V2 eval configs should use v2_eval_adapter_from_config(), not legacy_eval_adapter_from_type()"
            )
        case _:
            raise_exhaustive_enum_error(eval_config.config_type)


def v2_eval_adapter_from_config(
    eval_config: EvalConfig,
    run_config: "RunConfigProperties | None" = None,
    skills: "SkillsDict | None" = None,
) -> BaseV2EvalBridge:
    """V2 dispatch -- reads properties.type and looks up the adapter in _V2_ADAPTER_MAP.

    Returns an instantiated adapter, or raises NotImplementedError if the
    V2 type has no registered adapter (type_not_available skip path).
    """
    if eval_config.config_type != EvalConfigType.v2:
        raise ValueError("v2_eval_adapter_from_config only accepts V2 configs")
    if not isinstance(eval_config.properties, V2_PROPERTY_TYPES):
        raise ValueError("V2 config must have typed properties")
    v2_type = eval_config.properties.type  # type: ignore[union-attr]
    adapter_cls = _V2_ADAPTER_MAP.get(v2_type)
    if adapter_cls is None:
        raise NotImplementedError(f"V2 eval type '{v2_type}' is not yet implemented")
    return adapter_cls(eval_config, run_config, skills)
