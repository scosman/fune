"""V2 adapter for LLM Judge evaluations.

Supports two scoring modes:
- LLM-as-Judge (g_eval=False): uses structured output directly.
- G-Eval (g_eval=True): uses logprob-weighted scoring.

The prompt_template is rendered with Jinja2 using the EvalTaskInput fields
as the template namespace.
"""

from typing import TYPE_CHECKING, Any

from jinja2 import UndefinedError, meta

from kiln_ai.adapters.adapter_registry import adapter_for_task
from kiln_ai.adapters.eval.base_eval import (
    BaseEval,
    BaseV2EvalBridge,
    format_judge_instructions,
)

if TYPE_CHECKING:
    from kiln_ai.adapters.model_adapters.base_adapter import SkillsDict
    from kiln_ai.datamodel.task import RunConfigProperties

from kiln_ai.adapters.eval.eval_utils.scoring_utils import (
    build_g_eval_score,
    build_llm_as_judge_score,
    g_eval_single_metric,
    metric_offsets,
    raw_output_from_logprobs,
    score_from_token_string,
)
from kiln_ai.adapters.eval.eval_utils.v2_eval_helpers import check_reference_key
from kiln_ai.adapters.ml_model_list import (
    ModelProviderName,
    built_in_models_from_provider,
    default_structured_output_mode_for_model_provider,
)
from kiln_ai.adapters.model_adapters.base_adapter import AdapterConfig
from kiln_ai.datamodel.eval import (
    EvalConfig,
    EvalTaskInput,
    LlmJudgeProperties,
    SkippedReason,
    V2EvalResult,
)
from kiln_ai.datamodel.project import Project
from kiln_ai.datamodel.prompt_id import PromptGenerators
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties
from kiln_ai.datamodel.task import StructuredOutputMode, Task
from kiln_ai.utils.jinja_engine import JinjaExtractionError, _template_env

_DEFAULT_SYSTEM_PROMPT = (
    "Your job is to evaluate a model's performance on a task. "
    "Score the output according to the criteria provided."
)


def _unknown_template_variables(template: str, namespace: dict[str, Any]) -> set[str]:
    """Top-level variables the template uses that aren't EvalTaskInput fields.

    Distinguishes a template-authoring bug (typo'd variable name) from
    genuinely missing data (e.g. an absent reference_data key) when rendering
    raises UndefinedError.
    """
    ast = _template_env.parse(template)
    return meta.find_undeclared_variables(ast) - set(namespace)


class _LlmJudgeTask(Task, parent_of={}):
    """Ephemeral Task for invoking an LLM judge via adapter_for_task().

    Creates a temporary Project/Task with the system prompt and output
    JSON schema.
    """

    def __init__(
        self,
        system_prompt: str,
        output_json_schema: str,
    ):
        tmp_project = Project(name="LlmJudge")
        super().__init__(
            name="LlmJudge Task",
            parent=tmp_project,
            instruction=system_prompt,
            output_json_schema=output_json_schema,
        )


class LlmJudgeEval(BaseV2EvalBridge):
    """V2 adapter that invokes an LLM to score eval inputs.

    Uses LlmJudgeProperties from the eval config to configure the judge:
    model, prompt template, scoring mode, etc.
    """

    def __init__(
        self,
        eval_config: EvalConfig,
        run_config: "RunConfigProperties | None" = None,
        skills: "SkillsDict | None" = None,
    ) -> None:
        super().__init__(eval_config, run_config, skills)
        if not isinstance(self.properties, LlmJudgeProperties):
            raise ValueError(
                "LlmJudgeEval requires LlmJudgeProperties in the eval config"
            )
        if self.properties.model_provider not in ModelProviderName.__members__:
            raise ValueError(
                f"Invalid model provider: {self.properties.model_provider}"
            )

    async def evaluate(self, eval_input: EvalTaskInput) -> V2EvalResult:
        props = self.properties
        assert isinstance(props, LlmJudgeProperties)

        # Before the render and before any model call: a judge that declares a
        # reference key it cannot get must skip loudly, not score blind. A template can
        # be permissive (a hand-written one may guard its own lookups), so this declared
        # requirement is what refuses missing ground truth. It also runs first for the
        # backend-baked default, whose `<reference_answer>` block is unconditional —
        # that one would raise below and land on the same skip, one step later.
        #
        # Unlike the runner's early skips, this one cannot be hoisted ahead of trace
        # generation (`eval_runner.py` run_job): the requirement is a property of this
        # judge's config, and the runner scores one item against many judges. Reading it
        # there would put per-type properties back in the runner, which is exactly what
        # the `evaluate()` contract exists to keep out.
        for reference_key in props.reference_keys:
            _, skip_reason, skip_detail = check_reference_key(reference_key, eval_input)
            if skip_reason is not None:
                return V2EvalResult(
                    skipped_reason=skip_reason,
                    skipped_detail=skip_detail,
                )

        namespace = eval_input.model_dump()
        # Always bound (even when unset) so templates referencing it never hit
        # StrictUndefined; blank steps render as an empty <steps> body.
        namespace["judge_instructions"] = format_judge_instructions(
            props.judge_instructions
        )
        try:
            rendered_prompt = _template_env.from_string(props.prompt_template).render(
                **namespace
            )
        except JinjaExtractionError as e:
            return V2EvalResult(
                skipped_reason=SkippedReason.extraction_failed,
                skipped_detail=f"Template rendering failed: {e}",
            )
        except UndefinedError as e:
            # A typo'd template variable and genuinely missing reference data
            # both raise UndefinedError; only the latter is a data problem.
            unknown = _unknown_template_variables(props.prompt_template, namespace)
            if unknown:
                return V2EvalResult(
                    skipped_reason=SkippedReason.extraction_failed,
                    skipped_detail=(
                        "Template references unknown variable(s): "
                        + ", ".join(sorted(unknown))
                    ),
                )
            return V2EvalResult(
                skipped_reason=SkippedReason.missing_reference_key,
                skipped_detail=f"Template references missing data: {e}",
            )

        output_json_schema = BaseEval.build_score_schema(
            self.eval, allow_float_scores=False
        )

        system_prompt = props.system_prompt or _DEFAULT_SYSTEM_PROMPT

        judge_task = _LlmJudgeTask(
            system_prompt=system_prompt,
            output_json_schema=output_json_schema,
        )

        model_name = props.model_name
        provider = ModelProviderName(props.model_provider)

        if props.g_eval:
            model_provider = built_in_models_from_provider(provider, model_name)
            if model_provider is None:
                # Fail before spending on the judge call: without a built-in
                # entry, logprobs support can't be verified and the call would
                # fail deterministically anyway.
                raise ValueError(
                    f"g_eval=True requires logprobs support, but model "
                    f"'{model_name}' is not a built-in model for provider "
                    f"'{props.model_provider}', so logprobs support can't be "
                    f"verified. Use a built-in model that supports logprobs, "
                    f"or disable G-Eval for this judge."
                )
            if not model_provider.supports_logprobs:
                raise ValueError(
                    f"g_eval=True requires logprobs support, but provider "
                    f"'{props.model_provider}' for model '{model_name}' does not "
                    f"support logprobs"
                )

        structured_output_mode = default_structured_output_mode_for_model_provider(
            model_name,
            provider,
            default=StructuredOutputMode.json_schema,
            disallowed_modes=[
                StructuredOutputMode.function_calling,
                StructuredOutputMode.function_calling_weak,
            ],
        )

        top_logprobs = 10 if props.g_eval else None

        adapter = adapter_for_task(
            judge_task,
            run_config_properties=KilnAgentRunConfigProperties(
                model_name=model_name,
                model_provider_name=provider,
                prompt_id=PromptGenerators.SIMPLE,
                structured_output_mode=structured_output_mode,
            ),
            base_adapter_config=AdapterConfig(
                allow_saving=False,
                top_logprobs=top_logprobs,
            ),
        )

        judge_run, run_output = await adapter.invoke_returning_run_output(
            rendered_prompt
        )

        if props.g_eval:
            scores = build_g_eval_score(
                run_output,
                raw_output_from_logprobs,
                metric_offsets,
                g_eval_single_metric,
            )
        else:
            scores = build_llm_as_judge_score(
                run_output,
                score_from_token_string,
            )

        return V2EvalResult(
            scores=scores,
            intermediate_outputs=run_output.intermediate_outputs,
            # `usage`, not `cumulative_usage`: it already accumulates every call this
            # judgment made, and it is the one that carries latency.
            usage=judge_run.usage,
        )
