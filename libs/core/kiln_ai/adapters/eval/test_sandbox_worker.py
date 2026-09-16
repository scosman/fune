"""Tests for the code-eval scorer worker -- multiprocessing scorer execution.

These tests SPAWN child processes via the shared bridge. Keep them fast. Scorer
code is passed as a string so nothing local needs pickling.
"""

import pytest

from kiln_ai.adapters.eval.conftest import run_scorer


def _inputs(output: str = "hello", **overrides: object) -> dict:
    defaults: dict = {
        "output": output,
        "trace": None,
        "reference_data": None,
        "task_input": "test input",
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_scorer_returns_dict(self):
        code = (
            "def score(output, trace, reference_data, task_input):\n"
            "    return {'accuracy': 1.0}\n"
        )
        result = run_scorer(code, _inputs(), timeout=10)
        assert "ok" in result
        assert result["ok"] == {"accuracy": 1.0}

    def test_scorer_uses_output(self):
        code = (
            "def score(output, trace, reference_data, task_input):\n"
            "    return {'has_hello': 1.0 if 'hello' in output else 0.0}\n"
        )
        result = run_scorer(code, _inputs(output="hello world"), timeout=10)
        assert result["ok"] == {"has_hello": 1.0}

        result2 = run_scorer(code, _inputs(output="goodbye"), timeout=10)
        assert result2["ok"] == {"has_hello": 0.0}

    def test_scorer_uses_stdlib(self):
        code = (
            "import math\n"
            "def score(output, trace, reference_data, task_input):\n"
            "    return {'sqrt4': math.sqrt(4)}\n"
        )
        result = run_scorer(code, _inputs(), timeout=10)
        assert result["ok"] == {"sqrt4": 2.0}

    def test_scorer_uses_kiln_helpers(self):
        code = (
            "from kiln_ai.adapters.eval.eval_helpers import KilnEvalHelpers\n"
            "def score(output, trace, reference_data, task_input):\n"
            "    return {'pass': KilnEvalHelpers.pass_fail(True)}\n"
        )
        result = run_scorer(code, _inputs(), timeout=10)
        assert result["ok"] == {"pass": 1.0}


# ---------------------------------------------------------------------------
# Async scorer support
# ---------------------------------------------------------------------------


class TestAsyncScorer:
    def test_async_score_returns_dict(self):
        code = (
            "async def score(output, trace, reference_data, task_input):\n"
            "    return {'accuracy': 1.0}\n"
        )
        result = run_scorer(code, _inputs(), timeout=10)
        assert "ok" in result
        assert result["ok"] == {"accuracy": 1.0}

    def test_async_score_with_gather(self):
        code = (
            "import asyncio\n"
            "\n"
            "async def _check_a(output):\n"
            "    return 1.0 if 'hello' in output else 0.0\n"
            "\n"
            "async def _check_b(output):\n"
            "    return 0.5\n"
            "\n"
            "async def score(output, trace, reference_data, task_input):\n"
            "    a, b = await asyncio.gather(_check_a(output), _check_b(output))\n"
            "    return {'check_a': a, 'check_b': b}\n"
        )
        result = run_scorer(code, _inputs(output="hello world"), timeout=10)
        assert "ok" in result
        assert result["ok"] == {"check_a": 1.0, "check_b": 0.5}

    def test_async_score_raising(self):
        code = (
            "async def score(output, trace, reference_data, task_input):\n"
            "    raise ValueError('async boom')\n"
        )
        result = run_scorer(code, _inputs(), timeout=10)
        assert "error" in result
        assert "async boom" in result["error"]
        assert "traceback" in result

    def test_async_score_timeout(self):
        code = (
            "import asyncio\n"
            "async def score(output, trace, reference_data, task_input):\n"
            "    await asyncio.sleep(10)\n"
            "    return {'x': 1.0}\n"
        )
        with pytest.raises(RuntimeError, match="timed out"):
            run_scorer(code, _inputs(), timeout=1)


# ---------------------------------------------------------------------------
# Stdout / stderr capture
# ---------------------------------------------------------------------------


class TestCapture:
    def test_stdout_captured(self):
        code = (
            "import sys\n"
            "def score(output, trace, reference_data, task_input):\n"
            "    sys.stdout.write('debug info')\n"
            "    return {'x': 1.0}\n"
        )
        result = run_scorer(code, _inputs(), timeout=10)
        assert "ok" in result
        assert "debug info" in result["stdout"]

    def test_stderr_captured(self):
        code = (
            "import sys\n"
            "def score(output, trace, reference_data, task_input):\n"
            "    sys.stderr.write('err msg')\n"
            "    return {'x': 1.0}\n"
        )
        result = run_scorer(code, _inputs(), timeout=10)
        assert "ok" in result
        assert "err msg" in result["stderr"]

    def test_large_captured_output_succeeds(self):
        # A result payload bigger than the OS pipe buffer (~64KB) keeps the
        # child alive until the parent drains the queue; the run must succeed
        # instead of being misreported as a timeout.
        code = (
            "def score(output, trace, reference_data, task_input):\n"
            "    print('x' * 70_000)\n"
            "    return {'x': 1.0}\n"
        )
        result = run_scorer(code, _inputs(), timeout=10)
        assert result["ok"] == {"x": 1.0}
        assert len(result["stdout"]) >= 70_000


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_runtime_exception(self):
        code = (
            "def score(output, trace, reference_data, task_input):\n"
            "    raise ValueError('boom')\n"
        )
        result = run_scorer(code, _inputs(), timeout=10)
        assert "error" in result
        assert "boom" in result["error"]
        assert "traceback" in result

    def test_missing_score_function(self):
        code = "x = 1\n"
        result = run_scorer(code, _inputs(), timeout=10)
        assert "error" in result
        assert "score" in result["error"].lower()

    def test_score_not_callable(self):
        code = "score = 42\n"
        result = run_scorer(code, _inputs(), timeout=10)
        assert "error" in result
        assert "not callable" in result["error"]

    def test_syntax_error_in_code(self):
        code = "def score(\n"
        result = run_scorer(code, _inputs(), timeout=10)
        assert "error" in result
        assert "traceback" in result

    def test_timeout(self):
        code = (
            "import time\n"
            "def score(output, trace, reference_data, task_input):\n"
            "    time.sleep(10)\n"
            "    return {'x': 1.0}\n"
        )
        with pytest.raises(RuntimeError, match="timed out"):
            run_scorer(code, _inputs(), timeout=1)

    def test_crash_via_os_exit(self):
        code = (
            "import os\n"
            "def score(output, trace, reference_data, task_input):\n"
            "    os._exit(1)\n"
        )
        with pytest.raises(RuntimeError, match="exit code"):
            run_scorer(code, _inputs(), timeout=10)

    def test_sys_exit_reported_as_user_error(self):
        # sys.exit() must surface as a user-code error, not a generic
        # no-result crash from the child dying silently.
        code = (
            "import sys\n"
            "def score(output, trace, reference_data, task_input):\n"
            "    sys.exit(2)\n"
        )
        result = run_scorer(code, _inputs(), timeout=10)
        assert "error" in result
        assert "sys.exit(2)" in result["error"]
        assert "traceback" in result

    def test_sys_exit_at_module_level(self):
        code = "import sys\nsys.exit()\n"
        result = run_scorer(code, _inputs(), timeout=10)
        assert "error" in result
        assert "sys.exit" in result["error"]


# ---------------------------------------------------------------------------
# Dynamic (named) score() signature
# ---------------------------------------------------------------------------


class TestDynamicSignature:
    """Verify that the harness introspects score() params and passes only declared ones."""

    def test_subset_output_only(self):
        code = (
            "def score(output):\n"
            "    return {'has_hello': 1.0 if 'hello' in output else 0.0}\n"
        )
        result = run_scorer(code, _inputs(output="hello"), timeout=10)
        assert result["ok"] == {"has_hello": 1.0}

    def test_subset_trace_only(self):
        code = "def score(trace):\n    return {'has_trace': 1.0 if trace else 0.0}\n"
        result = run_scorer(code, _inputs(trace=[{"role": "user"}]), timeout=10)
        assert result["ok"] == {"has_trace": 1.0}

    def test_subset_output_and_task_input(self):
        code = (
            "def score(output, task_input):\n"
            "    return {'match': 1.0 if task_input in output else 0.0}\n"
        )
        result = run_scorer(
            code,
            _inputs(output="echo: test input", task_input="test input"),
            timeout=10,
        )
        assert result["ok"] == {"match": 1.0}

    def test_reordered_params(self):
        code = "def score(task_input, trace):\n    return {'ok': 1.0}\n"
        result = run_scorer(code, _inputs(), timeout=10)
        assert result["ok"] == {"ok": 1.0}

    def test_full_four_params_backward_compat(self):
        code = (
            "def score(output, trace, reference_data, task_input):\n"
            "    return {'all': 1.0}\n"
        )
        result = run_scorer(code, _inputs(), timeout=10)
        assert result["ok"] == {"all": 1.0}

    def test_output_and_reference_data(self):
        code = (
            "def score(output, reference_data):\n"
            "    expected = (reference_data or {}).get('answer', '')\n"
            "    return {'match': 1.0 if expected in output else 0.0}\n"
        )
        result = run_scorer(
            code,
            _inputs(output="42 is the answer", reference_data={"answer": "42"}),
            timeout=10,
        )
        assert result["ok"] == {"match": 1.0}


class TestDynamicSignatureGuard:
    """Verify the guard: score() must accept at least 'output' or 'trace'."""

    def test_no_output_or_trace_error(self):
        code = "def score(reference_data):\n    return {'x': 1.0}\n"
        result = run_scorer(code, _inputs(), timeout=10)
        assert "error" in result
        assert "output" in result["error"]
        assert "trace" in result["error"]

    def test_empty_params_error(self):
        code = "def score():\n    return {'x': 1.0}\n"
        result = run_scorer(code, _inputs(), timeout=10)
        assert "error" in result
        assert "output" in result["error"]
        assert "trace" in result["error"]

    def test_only_task_input_error(self):
        code = "def score(task_input):\n    return {'x': 1.0}\n"
        result = run_scorer(code, _inputs(), timeout=10)
        assert "error" in result
        assert "output" in result["error"]
        assert "trace" in result["error"]


class TestDynamicSignatureAsync:
    """Verify dynamic signature works with async score functions too."""

    def test_async_subset_output_only(self):
        code = "async def score(output):\n    return {'length': float(len(output))}\n"
        result = run_scorer(code, _inputs(output="hi"), timeout=10)
        assert result["ok"] == {"length": 2.0}

    def test_async_reordered(self):
        code = "async def score(task_input, output):\n    return {'ok': 1.0}\n"
        result = run_scorer(code, _inputs(), timeout=10)
        assert result["ok"] == {"ok": 1.0}
