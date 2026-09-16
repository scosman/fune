"""Child-process entry point for code-eval scorer execution.

Stdlib only — no Pydantic / Kiln-model / DB / UI imports. Imports only from
``kiln_ai.sandbox`` (itself stdlib-only).

The parent spawns this via ``multiprocessing`` (spawn context) through
:func:`kiln_ai.tools.sandbox_bridge.run_bridged_child`. Communication uses two
queues (requests child->parent, responses parent->child), so the user ``score()``
can call allowlisted tools (including ``tools.llm`` / ``tools.llm_judge``) via the
synthetic ``kiln.tools`` / ``kiln.async_tools`` modules.
"""

import inspect
import io
import sys
import traceback
from multiprocessing import Queue
from typing import Any

from kiln_ai.sandbox.entrypoint import call_entrypoint
from kiln_ai.sandbox.tools_api import install_tools_modules


def execute_scorer_bridged(
    code: str,
    inputs: dict[str, Any],
    requests: Queue,  # type: ignore[type-arg]
    responses: Queue,  # type: ignore[type-arg]
) -> None:
    """Entry point for the code-eval child process (two-queue bridge protocol).

    Runs user scorer code in an isolated namespace. Both ``def score`` and
    ``async def score`` are supported -- if the call returns a coroutine it is
    transparently awaited via :func:`call_entrypoint`.

    Puts exactly one ``result`` message on *requests* and then returns:
      - success:  {"type":"result","ok":<scores dict>,"stdout":...,"stderr":...}
      - error:    {"type":"result","error":str,"traceback":str,"stdout":...,"stderr":...}
      - missing:  {"type":"result","error":"...does not define score...","stdout":...,"stderr":...}

    ``ok`` carries the scorer's returned scores **dict verbatim** (a code eval returns
    a dict, not a passthrough string — no serialization step).
    """
    # Redirect stdout/stderr to capture user prints and avoid None-stdout
    # crashes in frozen (PyInstaller --windowed) builds.
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = captured_stdout  # type: ignore[assignment]
        sys.stderr = captured_stderr  # type: ignore[assignment]

        install_tools_modules(requests, responses)

        namespace: dict[str, Any] = {}
        exec(compile(code, "<code_eval>", "exec"), namespace)

        score_fn = namespace.get("score")
        if score_fn is None:
            _put_result_error(
                requests,
                "User code does not define a 'score()' function",
                captured_stdout,
                captured_stderr,
            )
            return
        if not callable(score_fn):
            _put_result_error(
                requests,
                "'score' is defined but is not callable",
                captured_stdout,
                captured_stderr,
            )
            return

        known_params = {
            "output": inputs["output"],
            "trace": inputs.get("trace"),
            "reference_data": inputs.get("reference_data"),
            "task_input": inputs["task_input"],
        }
        sig = inspect.signature(score_fn)
        declared = set(sig.parameters.keys())
        if not declared & {"output", "trace"}:
            _put_result_error(
                requests,
                "score() must accept at least 'output' or 'trace' as a parameter",
                captured_stdout,
                captured_stderr,
            )
            return
        call_kwargs = {k: v for k, v in known_params.items() if k in declared}
        result = call_entrypoint(score_fn, call_kwargs)

        requests.put(
            {
                "type": "result",
                "ok": result,
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
            }
        )
    except SystemExit as e:
        # sys.exit() in user code would otherwise kill the child before a
        # result is queued, surfacing as a generic crash with no explanation.
        # KeyboardInterrupt is deliberately NOT caught: it means the operator
        # interrupted the process group, not that the user's code failed.
        requests.put(
            {
                "type": "result",
                "error": f"User code exited via sys.exit({e.code!r}) instead of returning scores",
                "traceback": traceback.format_exc(),
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
            }
        )
    except Exception as exc:
        requests.put(
            {
                "type": "result",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
            }
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _put_result_error(
    requests: Queue,  # type: ignore[type-arg]
    error: str,
    stdout: io.StringIO,
    stderr: io.StringIO,
) -> None:
    requests.put(
        {
            "type": "result",
            "error": error,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
        }
    )
