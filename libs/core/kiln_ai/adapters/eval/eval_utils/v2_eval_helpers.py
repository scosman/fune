import json
from typing import Any

from jinja2 import TemplateSyntaxError, Undefined, meta, nodes
from jinja2.parser import Parser
from kiln_ai.datamodel.eval import (
    EvalOutputScore,
    EvalScores,
    EvalTaskInput,
    SkippedReason,
    V2EvalResult,
)
from kiln_ai.utils.jinja_engine import JinjaExtractionError, _expression_env, extract


def build_binary_scores(
    output_scores: list[EvalOutputScore], passed: bool
) -> EvalScores:
    """Build an EvalScores dict with the same binary value for all declared score keys.

    Returns {} if output_scores is empty.
    """
    if not output_scores:
        return {}
    value = 1.0 if passed else 0.0
    return {score.json_key(): value for score in output_scores}


def references_trace(expression: str) -> bool:
    """True if a Jinja expression references the `trace` variable.

    Uses Jinja's own parser so only genuine variable references count --
    'trace' appearing as data (a quoted string, a dict key, an attribute of
    another variable) does not.
    """
    try:
        # Same parse compile_expression performs, wrapped into a template node
        # so meta can resolve the expression's undeclared (i.e. input) names.
        parser = Parser(_expression_env, expression, state="variable")
        template = nodes.Template([nodes.Output([parser.parse_expression()])], lineno=1)
    except TemplateSyntaxError:
        # Invalid expressions can't reference anything; the extraction path
        # owns reporting the syntax error.
        return False
    template.set_environment(_expression_env)
    return "trace" in meta.find_undeclared_variables(template)


def extract_value(
    expression: str | None,
    eval_input: EvalTaskInput,
) -> tuple[Any, SkippedReason | None, str | None]:
    """Extract a value from eval_input using a Jinja2 expression.

    If expression is None, defaults to eval_input.final_message.
    Returns (value, skip_reason, skip_detail).
    """
    if expression is None:
        return eval_input.final_message, None, None
    # An expression that needs the trace, but this run has none, is a
    # data-availability skip (missing_trace) -- not an extraction failure.
    if eval_input.trace is None and references_trace(expression):
        return (
            None,
            SkippedReason.missing_trace,
            f"Expression '{expression}' requires a trace, but this run has no trace",
        )
    data = eval_input.model_dump()
    try:
        result = extract(expression, data)
    except JinjaExtractionError as e:
        return (
            None,
            SkippedReason.extraction_failed,
            f"Expression '{expression}' failed: {e}",
        )
    if isinstance(result, Undefined) or result is None:
        return (
            None,
            SkippedReason.extraction_failed,
            f"Expression '{expression}' resolved to {'undefined' if isinstance(result, Undefined) else 'None'}",
        )
    return result, None, None


def extract_output_value(
    expression: str | None,
    eval_input: EvalTaskInput,
    output_scores: list[EvalOutputScore],
) -> tuple[Any, V2EvalResult | None]:
    """Extract an output value, returning a failing result on extraction failure.

    For deterministic checks (contains, exact_match, pattern_match, set_check),
    a missing or unparseable output field means the model didn't produce the
    expected structure -- that's a FAIL, not a skip.  Skips are reserved for
    missing ground-truth / reference data.

    Returns ``(value, None)`` on success, or ``(None, failing_result)`` when the
    extraction fails.
    """
    value, skip, detail = extract_value(expression, eval_input)
    if skip is not None:
        # A missing trace is a data-availability skip (like missing reference
        # data), not a model failure -- propagate it as a skip rather than
        # scoring it as a FAIL.
        if skip == SkippedReason.missing_trace:
            return None, V2EvalResult(skipped_reason=skip, skipped_detail=detail)
        return None, V2EvalResult(
            scores=build_binary_scores(output_scores, passed=False),
        )
    return value, None


def check_reference_key(
    reference_key: str,
    eval_input: EvalTaskInput,
) -> tuple[Any, SkippedReason | None, str | None]:
    """Look up a key in eval_input.reference_data.

    Returns (value, skip_reason, skip_detail).
    """
    if eval_input.reference_data is None:
        return (
            None,
            SkippedReason.missing_reference_key,
            f"No reference_data; need key '{reference_key}'",
        )
    if reference_key not in eval_input.reference_data:
        return (
            None,
            SkippedReason.missing_reference_key,
            f"reference_data missing key '{reference_key}'",
        )
    value = eval_input.reference_data[reference_key]
    if value is None:
        return (
            None,
            SkippedReason.missing_reference_key,
            f"reference_data key '{reference_key}' is None",
        )
    return value, None, None


def stringify_for_match(value: object) -> str:
    """Coerce a value to a string for comparison in deterministic checks.

    - str values pass through unchanged.
    - Other types are serialized via ``json.dumps`` (preserving insertion order,
      producing ``true``/``false`` for bools, double-quoted strings, etc.).
    - Falls back to ``str(value)`` if JSON serialization fails.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
