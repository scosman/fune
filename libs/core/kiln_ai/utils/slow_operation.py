"""Soft-threshold logging for long-running awaits.

The wrapped operation is never cancelled or timed: the watchdog only logs a
warning when the operation is still running at the threshold, so pathological
durations are visible in logs without killing healthy work.
"""

import asyncio
import contextlib
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)

# Well beyond a healthy conversation drive or single task run, so a firing
# means something worth investigating (a trickling stream, a stuck tool
# loop), not a slow-but-fine case. Logging only; nothing is cancelled.
DEFAULT_SLOW_LOG_THRESHOLD_SECONDS = 600.0


@contextlib.asynccontextmanager
async def log_if_slow(
    description: str,
    threshold_seconds: float | None = None,
) -> AsyncIterator[None]:
    """Warn once if the wrapped block is still running after
    `threshold_seconds` (default DEFAULT_SLOW_LOG_THRESHOLD_SECONDS,
    resolved at call time so tests can patch it); the block itself always
    runs to completion."""
    if threshold_seconds is None:
        threshold_seconds = DEFAULT_SLOW_LOG_THRESHOLD_SECONDS

    async def _watchdog() -> None:
        await asyncio.sleep(threshold_seconds)
        logger.warning(
            "%s is still running after %.0fs; letting it continue",
            description,
            threshold_seconds,
        )

    task = asyncio.create_task(_watchdog())
    try:
        yield
    finally:
        task.cancel()
        # gather(return_exceptions=True) absorbs the watchdog's own
        # CancelledError but still lets a cancellation aimed at the
        # ENCLOSING task propagate — a bare suppress() here would consume
        # it and keep a cancelled operation running (and spending).
        await asyncio.gather(task, return_exceptions=True)
