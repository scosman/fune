import asyncio
import logging
import random
from dataclasses import dataclass
from typing import AsyncGenerator, Awaitable, Callable, Generic, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Each retry waits in a window this many times wider than the last, so a short
# stall is absorbed by a single retry while a longer one still gets a real wait.
# Windows grow fast and are never capped, which suits the small retry counts
# callers use today (<= 2-3); raising max_retries much beyond that means
# lowering this factor or adding a ceiling.
RETRY_BACKOFF_FACTOR = 4.0


def compute_retry_delay(
    base_delay: float,
    attempt: int,
    random_uniform: Callable[[float, float], float] = random.uniform,
) -> float:
    """Seconds to wait before retry `attempt` (zero-indexed): a uniform draw
    from the whole exponentially growing window, i.e. "full jitter".

    Full jitter beats equal jitter and plain backoff under contention (AWS
    Architecture Blog, "Exponential Backoff and Jitter"): concurrent callers
    that failed together are spread across the window instead of retrying in
    lockstep and colliding again.
    """
    return random_uniform(0.0, base_delay * RETRY_BACKOFF_FACTOR**attempt)


@dataclass
class Progress:
    complete: int
    total: int
    errors: int


class RetryableError(Exception):
    """Raise from run_job_fn to signal a transient failure that should be retried."""

    pass


class AsyncJobRunnerObserver(Generic[T]):
    async def on_error(self, job: T, error: Exception):
        """
        Called when a job raises an unhandled exception.
        """
        pass

    async def on_success(self, job: T):
        """
        Called when a job completes successfully.
        """
        pass

    async def on_job_start(self, job: T):
        """
        Called when a job starts.
        """
        pass


class AsyncJobRunner(Generic[T]):
    def __init__(
        self,
        jobs: List[T],
        run_job_fn: Callable[[T], Awaitable[bool]],
        concurrency: int = 1,
        observers: List[AsyncJobRunnerObserver[T]] | None = None,
        max_retries: int = 0,
        # Width of the first backoff window, in seconds; later windows grow by
        # RETRY_BACKOFF_FACTOR. Each wait is a random draw from its window.
        retry_delay: float = 1.0,
        # Injectable so tests can pin the jitter draw.
        random_uniform: Callable[[float, float], float] = random.uniform,
    ):
        if concurrency < 1:
            raise ValueError("concurrency must be ≥ 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_delay < 0:
            raise ValueError("retry_delay must be >= 0")
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.random_uniform = random_uniform
        self.jobs = jobs
        self.run_job_fn = run_job_fn
        self.observers = observers or []

    def _compute_retry_delay(self, attempt: int) -> float:
        """Seconds to wait before this runner's retry `attempt` (zero-indexed):
        a jittered draw from that attempt's backoff window."""
        return compute_retry_delay(self.retry_delay, attempt, self.random_uniform)

    async def notify_error(self, job: T, error: Exception):
        for observer in self.observers:
            await observer.on_error(job, error)

    async def notify_success(self, job: T):
        for observer in self.observers:
            await observer.on_success(job)

    async def notify_job_start(self, job: T):
        for observer in self.observers:
            await observer.on_job_start(job)

    async def run(self) -> AsyncGenerator[Progress, None]:
        """
        Runs the jobs with parallel workers and yields progress updates.
        """
        complete = 0
        errors = 0
        total = len(self.jobs)

        # Send initial status
        yield Progress(complete=complete, total=total, errors=errors)

        worker_queue: asyncio.Queue[T] = asyncio.Queue()
        for job in self.jobs:
            worker_queue.put_nowait(job)

        # simple status queue to return progress. True=success, False=error
        status_queue: asyncio.Queue[bool] = asyncio.Queue()

        workers = []
        for _ in range(self.concurrency):
            task = asyncio.create_task(
                self._run_worker(worker_queue, status_queue, self.run_job_fn),
            )
            workers.append(task)

        try:
            # Send status updates until workers are done, and they are all sent
            while not status_queue.empty() or not all(
                worker.done() for worker in workers
            ):
                try:
                    # Use timeout to prevent hanging if all workers complete
                    # between our while condition check and get()
                    success = await asyncio.wait_for(status_queue.get(), timeout=0.1)
                    if success:
                        complete += 1
                    else:
                        errors += 1

                    yield Progress(
                        complete=complete,
                        total=total,
                        errors=errors,
                    )
                except asyncio.TimeoutError:
                    # Timeout is expected, just continue to recheck worker status
                    # Don't love this but beats sentinels for reliability
                    continue
        finally:
            # Cancel outstanding workers on early exit or error
            for w in workers:
                w.cancel()

            # These are redundant, but keeping them will catch async errors
            await asyncio.gather(*workers)
            await worker_queue.join()

    async def _run_worker(
        self,
        worker_queue: asyncio.Queue[T],
        status_queue: asyncio.Queue[bool],
        run_job_fn: Callable[[T], Awaitable[bool]],
    ):
        while True:
            try:
                job = worker_queue.get_nowait()
            except asyncio.QueueEmpty:
                # worker can end when the queue is empty
                break

            await self.notify_job_start(job)
            result = False
            last_error: Exception | None = None
            for attempt in range(1 + self.max_retries):
                is_last_attempt = attempt == self.max_retries
                try:
                    result = await run_job_fn(job)
                    last_error = None
                    break
                except RetryableError as e:
                    result = False
                    last_error = e
                    if is_last_attempt:
                        logger.error("Job failed to complete", exc_info=e)
                        break
                    await asyncio.sleep(self._compute_retry_delay(attempt))
                except Exception as e:
                    result = False
                    last_error = e
                    logger.error("Job failed to complete", exc_info=e)
                    break

            if result:
                await self.notify_success(job)
            elif last_error is not None:
                await self.notify_error(job, last_error)

            try:
                await status_queue.put(result)
            except Exception:
                logger.error("Failed to enqueue status for job", exc_info=True)
            finally:
                # Always mark the dequeued task as done, even on exceptions
                worker_queue.task_done()
