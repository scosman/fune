import asyncio
import logging

import pytest

from kiln_ai.utils.slow_operation import log_if_slow


@pytest.mark.asyncio
async def test_warns_when_block_exceeds_threshold_and_block_completes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The watchdog is observability, not enforcement: the slow block still
    runs to completion, with one warning naming the operation."""
    finished = False
    with caplog.at_level(logging.WARNING, logger="kiln_ai.utils.slow_operation"):
        async with log_if_slow("slow op", threshold_seconds=0.01):
            await asyncio.sleep(0.05)
            finished = True

    assert finished
    warnings = [r for r in caplog.records if "slow op" in r.getMessage()]
    assert len(warnings) == 1
    assert "still running" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_no_warning_when_block_finishes_within_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="kiln_ai.utils.slow_operation"):
        async with log_if_slow("fast op", threshold_seconds=5.0):
            await asyncio.sleep(0)
        # The watchdog is cancelled on exit; give a cancelled-too-late timer
        # a chance to fire if cancellation were broken.
        await asyncio.sleep(0.02)

    assert not [r for r in caplog.records if "fast op" in r.getMessage()]


@pytest.mark.asyncio
async def test_exception_propagates_and_watchdog_is_cancelled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="kiln_ai.utils.slow_operation"):
        with pytest.raises(RuntimeError, match="boom"):
            async with log_if_slow("failing op", threshold_seconds=0.01):
                raise RuntimeError("boom")
        await asyncio.sleep(0.02)

    assert not [r for r in caplog.records if "failing op" in r.getMessage()]


@pytest.mark.asyncio
async def test_default_threshold_resolved_at_call_time(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Call sites use the default threshold; patching the module constant
    must take effect (the default is resolved inside the call, not bound
    at import)."""
    monkeypatch.setattr(
        "kiln_ai.utils.slow_operation.DEFAULT_SLOW_LOG_THRESHOLD_SECONDS", 0.01
    )
    with caplog.at_level(logging.WARNING, logger="kiln_ai.utils.slow_operation"):
        async with log_if_slow("patched default op"):
            await asyncio.sleep(0.05)

    assert [r for r in caplog.records if "patched default op" in r.getMessage()]
