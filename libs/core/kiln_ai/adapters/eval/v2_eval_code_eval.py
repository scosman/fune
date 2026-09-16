"""V2 adapter for code_eval: runs user-authored Python scorer in a sandboxed subprocess."""

import math
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable

from kiln_ai.adapters.eval.base_eval import BaseEval, BaseV2EvalBridge

if TYPE_CHECKING:
    from kiln_ai.adapters.model_adapters.base_adapter import SkillsDict
    from kiln_ai.datamodel.task import RunConfigProperties
from kiln_ai.adapters.eval.sandbox_worker import execute_scorer_bridged
from kiln_ai.datamodel.eval import (
    CodeEvalProperties,
    EvalConfig,
    EvalScores,
    EvalTaskInput,
    V2EvalResult,
)
from kiln_ai.tools.base_tool import ToolCallContext
from kiln_ai.tools.sandbox_bridge import (
    NestedToolServer,
    ToolCallLogEntry,
    run_bridged_child,
)

_trust_lock = Lock()
_trusted_projects: set[str] = set()


def add_code_trust(project_path: str) -> None:
    """Confer code trust on a project for the current session.

    Called when NEW/not-yet-saved code is admitted or executed (saving a code
    tool/eval, running not-yet-saved code in a test pane). Saved code is
    trusted to run without this — the flag governs the authoring session only.
    """
    with _trust_lock:
        _trusted_projects.add(project_path)


def has_add_code_trust(project_path: str) -> bool:
    with _trust_lock:
        return project_path in _trusted_projects


def _reset_add_code_trust() -> None:
    """Test-only reset of the in-memory trust set. Not part of the product API."""
    with _trust_lock:
        _trusted_projects.clear()


class CodeEvalAdapter(BaseV2EvalBridge):
    """V2 adapter that executes user-authored Python scorer code in a subprocess."""

    def __init__(
        self,
        eval_config: EvalConfig,
        run_config: "RunConfigProperties | None" = None,
        skills: "SkillsDict | None" = None,
    ) -> None:
        super().__init__(eval_config, run_config, skills)
        assert isinstance(self.properties, CodeEvalProperties)

        # Set by the test-pane endpoint, which is the one place a code judge's
        # nested tool calls (including LLM calls, which cost money) have somewhere
        # to be shown: the author is iterating on the code right there. An eval run
        # leaves it None -- per-item logs have no home in the run UI yet.
        self.tool_call_recorder: Callable[[ToolCallLogEntry], None] | None = None

    async def evaluate(self, eval_input: EvalTaskInput) -> V2EvalResult:
        props = self.properties
        assert isinstance(props, CodeEvalProperties)

        inputs: dict[str, Any] = {
            "output": eval_input.final_message,
            "trace": eval_input.trace,
            "reference_data": eval_input.reference_data,
            "task_input": eval_input.task_input,
        }

        server = NestedToolServer(
            allowlist=props.tool_allowlist,
            project=self.target_task.parent_project(),
            task=self.target_task,
            context=ToolCallContext(
                allow_saving=False,
                eval_output_schema=BaseEval.build_score_schema(
                    self.eval, allow_float_scores=False
                ),
            ),
            recorder=self.tool_call_recorder,
        )

        res = await run_bridged_child(
            target=execute_scorer_bridged,
            args=(props.code, inputs),
            timeout_s=float(props.timeout_seconds),
            server=server,
        )

        if res.timed_out:
            raise RuntimeError(
                f"Code eval scorer timed out after {props.timeout_seconds}s"
            )
        if res.crashed:
            raise RuntimeError(res.crash_description("Scorer"))

        result_msg = res.result_msg
        assert result_msg is not None
        if "error" in result_msg:
            raise RuntimeError(
                f"Code eval scorer failed: {result_msg['error']}\n"
                f"{result_msg.get('traceback', '')}"
            )

        raw_scores = result_msg["ok"]
        if not isinstance(raw_scores, dict):
            raise RuntimeError(
                f"Scorer must return a dict, got {type(raw_scores).__name__}"
            )

        scores = self._validate_scores(raw_scores)
        return V2EvalResult(scores=scores)

    def _validate_scores(self, raw: dict[str, Any]) -> EvalScores:
        expected_keys = {score.json_key() for score in self._output_scores}
        actual_keys = set(raw.keys())
        if actual_keys != expected_keys:
            raise RuntimeError(
                f"Score key mismatch: got {sorted(actual_keys)}, expected {sorted(expected_keys)}"
            )

        validated: EvalScores = {}
        for key, value in raw.items():
            if isinstance(value, bool):
                raise RuntimeError(
                    f"Score '{key}' returned a bool. Use a float (e.g. 1.0 for pass, 0.0 for fail)."
                )
            if isinstance(value, int):
                try:
                    value = float(value)
                except OverflowError:
                    # An int too large for a float, e.g. 10**400.
                    raise RuntimeError(
                        f"Score '{key}' must be a finite number, got {value}"
                    ) from None
            if not isinstance(value, float):
                raise RuntimeError(
                    f"Score '{key}' must be a float, got {type(value).__name__}"
                )
            # Fail here, in the scorer's own error surface, rather than at EvalRun
            # save time where the message loses the code eval context.
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Score '{key}' must be a finite number, got {value}"
                )
            validated[key] = value

        return validated
