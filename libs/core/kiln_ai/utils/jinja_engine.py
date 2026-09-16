"""
Jinja2 template engine for Kiln input transforms.

Two SandboxedEnvironment instances:
  - Template env: StrictUndefined, used by compile_template_or_raise and render_input_transform.
  - Expression env: default Undefined, used by compile_expression_or_raise and extract.

Both envs share trim_blocks=True and lstrip_blocks=True for prompt-friendly output.

Public API:
  - compile_template_or_raise(template) -> None
  - compile_expression_or_raise(expression) -> None
  - render_input_transform(transform, task_input) -> str
  - extract(expression, data) -> Any
"""

from __future__ import annotations

import json
import types
from typing import TYPE_CHECKING, Any

from jinja2 import StrictUndefined, TemplateSyntaxError, Undefined
from jinja2.sandbox import SandboxedEnvironment

if TYPE_CHECKING:
    from kiln_ai.datamodel.input_transform import InputTransform


class JinjaExtractionError(ValueError):
    """Raised when a Jinja2 filter encounters invalid data during extraction.

    For example, the ``fromjson`` filter raises this when the input string is
    not valid JSON.
    """


def _fromjson(value: object) -> Any:
    """Jinja2 filter: parse a JSON string into a Python object.

    Raises ``JinjaExtractionError`` on non-string input or invalid JSON.
    """
    if not isinstance(value, str):
        raise JinjaExtractionError(
            f"fromjson filter expected a string, got {type(value).__name__}"
        )
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise JinjaExtractionError(f"value is not valid JSON: {e}") from e


def _format_trace(value: Any) -> str:
    """Jinja2 filter: render a conversation trace as the canonical
    role-labelled transcript (EvalTraceFormatter's format).

    Templates over ``EvalTaskInput`` use ``{{ trace | format_trace }}`` so an
    LLM sees the same human-readable transcript everywhere the trace is
    rendered, instead of raw message JSON.

    Raises ``JinjaExtractionError`` when the value isn't a message list.
    """
    # Local import: jinja_engine is a low-level util imported by datamodel
    # modules; importing the formatter at module scope would be circular.
    from kiln_ai.adapters.eval.eval_utils.eval_trace_formatter import (
        EvalTraceFormatter,
    )

    if not isinstance(value, list):
        raise JinjaExtractionError(
            f"format_trace filter expected a list of messages, got {type(value).__name__}"
        )
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            # Name the offending member, not the container — "got list" would
            # hide which message is malformed.
            raise JinjaExtractionError(
                f"format_trace filter expected a list of messages, but message "
                f"{index} is {type(message).__name__}, not a dict"
            )
    try:
        # Loose dicts are the runtime reality for traces; the formatter reads
        # them like its typed message params.
        return EvalTraceFormatter.trace_to_formatted_conversation_history(value)
    except Exception as e:
        # Traces arrive as loose dicts; a malformed message (e.g. missing
        # 'role') must surface as an extraction error the evaluator can treat
        # as a skip, not an unhandled crash mid-render.
        raise JinjaExtractionError(
            f"trace is not a renderable message list: {e}"
        ) from e


_template_env = SandboxedEnvironment(
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)
_template_env.filters["fromjson"] = _fromjson
_template_env.filters["format_trace"] = _format_trace

_expression_env = SandboxedEnvironment(
    undefined=Undefined,
    trim_blocks=True,
    lstrip_blocks=True,
)
_expression_env.filters["fromjson"] = _fromjson
_expression_env.filters["format_trace"] = _format_trace


def compile_template_or_raise(template: str) -> None:
    """Validate that the template compiles. Raises ValueError on syntax error."""
    try:
        _template_env.from_string(template)
    except TemplateSyntaxError as e:
        raise ValueError(
            f"Invalid Jinja2 template: {e.message} (line {e.lineno})"
        ) from e


def compile_expression_or_raise(expression: str) -> None:
    """Validate that a Jinja2 expression compiles. Raises ValueError on syntax error."""
    try:
        _expression_env.compile_expression(expression)
    except TemplateSyntaxError as e:
        raise ValueError(
            f"Invalid Jinja2 expression: {e.message} (line {e.lineno})"
        ) from e


def render_input_transform(
    transform: InputTransform,
    task_input: Any,
) -> str:
    """Render the transform against the task input, returning the first-user-message string.

    See specs/projects/templates/functional_spec.md section 5 for the input-handling rules.
    """
    from kiln_ai.datamodel.input_transform import JinjaInputTransform

    if isinstance(transform, JinjaInputTransform):
        namespace = _build_namespace(task_input)
        return _template_env.from_string(transform.template).render(**namespace)

    raise ValueError(f"Unknown InputTransform variant: {type(transform).__name__}")


def extract(expression: str, data: dict) -> Any:
    """Evaluate a Jinja2 expression against a dict; returns the Python value.

    - Missing keys return Undefined (not None, not a raise).
    - Explicit null values return None.
    - Generators are auto-materialized to lists.
    """
    try:
        compiled = _expression_env.compile_expression(
            expression, undefined_to_none=False
        )
    except TemplateSyntaxError as e:
        raise ValueError(
            f"Invalid Jinja2 expression: {e.message} (line {e.lineno})"
        ) from e
    result = compiled(**data)
    if isinstance(result, types.GeneratorType):
        result = list(result)
    return result


def _build_namespace(task_input: Any) -> dict:
    """Build the rendering namespace per functional_spec section 5.

    The template always sees a single variable ``input``.
    For plaintext (str) inputs, try json.loads(); fall back to the raw string.
    For dict/list inputs, expose them under ``input`` as-is.
    """
    if isinstance(task_input, str):
        try:
            parsed = json.loads(task_input)
        except (ValueError, TypeError):
            parsed = task_input
        return {"input": parsed}
    return {"input": task_input}
