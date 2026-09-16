"""The live lookup that decides whether an eval job generates a trace or reuses one.

Reuse is keyed on `(source_type, source_id, run_config_id)` — the dataset item plus the
run config, and deliberately not the eval config. That is what lets a second judge score
generations the first judge already paid for (functional spec §2.1).

The lookup has to be live, not precomputed like the `already_run` set in
`EvalRunner.collect_tasks`. A set built before the first job runs cannot see a trace
persisted by another job in the same run, and `AsyncJobRunner` runs 25 of them at once:
two jobs sharing an item and a run config but differing in eval config would both miss
and both generate, spending exactly the money this exists to save. The same liveness is
what makes a retry after a scoring failure re-score rather than regenerate
(functional spec §4.2, §4.3).
"""

import json
import logging
from pathlib import Path
from typing import Awaitable, Callable, Dict, Tuple

from pydantic import ValidationError

from kiln_ai.datamodel.basemodel import ID_TYPE
from kiln_ai.datamodel.eval_splits import ItemKey, ItemSource
from kiln_ai.datamodel.task import Task
from kiln_ai.datamodel.task_run import TaskRun, eval_item_key
from kiln_ai.utils.lock import AsyncLockManager

logger = logging.getLogger(__name__)

TraceKey = Tuple[ItemSource, str, str]
"""What identifies a reusable eval trace: `(source_type, source_id, run_config_id)`.

`str` rather than the `ID_TYPE` (`Optional[str]`) the id fields carry, because this tuple
is a dict key: an id-less item and an id-less run config would produce one
`(source_type, None, None)` key that every id-less job collides on, handing them each
other's traces. `trace_key()` is where that impossibility is enforced."""


def trace_key(item: ItemKey, run_config_id: ID_TYPE) -> TraceKey:
    """The trace key for running `item` under `run_config_id`.

    The one place `ID_TYPE`'s nullability is resolved, so callers holding an `item.id` or
    a `task_run_config.id` don't each grow their own None check. Empty strings are
    rejected with None: they collide identically, and this key's whole purpose is that a
    collision is impossible.
    """
    source_type, source_id = item
    if not source_id or not run_config_id:
        raise ValueError(
            "A trace key needs both an item id and a run config id "
            f"(got item={item}, run_config_id={run_config_id}). Traces are looked up by "
            "the pair, so a missing half would match every other record missing it."
        )
    return (source_type, source_id, run_config_id)


def _stored_trace_key(run: TaskRun) -> TraceKey | None:
    """The key this run files itself under, from its own stored fields.

    Read from the run rather than from whatever the caller believes, because this is what
    the *next* process sees: a fresh index has only the record on disk to go on.
    """
    if run.eval_source is None:
        return None
    run_config_id = run.output.source.run_config_id if run.output.source else None
    if not run_config_id:
        return None
    return trace_key(eval_item_key(run.eval_source), run_config_id)


class TraceIndex:
    """The eval traces a task already has, and the gate for creating the ones it doesn't.

    Runner-owned and single-run-scoped: seeded once from disk at construction, then kept
    current by every trace it hands out.

    `vet` is the caller's veto over reuse: it decides whether a candidate record is fit to
    serve a key, which is knowledge the index does not have (a conversation's completeness
    depends on the item asking for it). It answers None for a fit record, or a reason
    string for an unfit one — the reason travels with the rejection so the index's own log
    line carries it, rather than stranding it in a separate line the caller emitted. A
    `vet` of None reuses on key identity alone.
    """

    def __init__(
        self, task: Task, vet: Callable[[TraceKey, TaskRun], str | None] | None = None
    ):
        self._task = task
        self._vet = vet
        # Paths, not TaskRuns. Reloading is what keeps a handed-out trace honest: it
        # re-reads the record (ModelCache validates on path + mtime), so an index built
        # early in a long eval never serves a stale object, and it stays correct across
        # cache invalidation. Cheap, because the reload is usually a cache hit.
        self._paths: Dict[TraceKey, Path] = {}
        # Already the two-level locking this needs: a manager mutex held only long enough
        # to create an entry, and a per-key lock held across generation, so jobs on
        # different keys never wait on each other. It also reclaims idle entries, which
        # matters over an eval with thousands of keys.
        self._locks = AsyncLockManager()
        self._seed()

    def _seed(self) -> None:
        # include_intermediate_runs stays False, so a trace that is the *parent* of
        # another run is invisible here. Safe because eval traces are always
        # childless: single-turn generations are single runs, and a driven
        # multi-turn conversation persists as one standalone run whose trace holds
        # the whole exchange (chain-leaf scoring reads stored dataset runs without
        # generating at all). Nothing may ever chain a child onto an eval trace
        # without revisiting this seed — the parent would turn invisible here, and
        # the eval would silently regenerate its trace on every run.
        for run in self._task.runs(readonly=True, include_eval_generated=True):
            if run.path is None:
                continue
            key = _stored_trace_key(run)
            if key is None:
                # Not an eval trace, or missing half its key — either way, no job can
                # ever match it.
                continue
            rejection = self._vet_rejection(key, run)
            if rejection is not None:
                # Skipped rather than indexed, so a fit candidate later in the scan can
                # still serve this key — an unfit file must not shadow a good one. The
                # file itself is left alone; this line is the only consequence. Info, not
                # warning: a rejected file stays on disk and is re-seen on every seed.
                logger.info(
                    "Not indexing eval trace at %s for %s: it failed vetting (%s)",
                    run.path,
                    key,
                    rejection,
                )
                continue
            # First fit wins (D8). Sync means two machines can each have generated a
            # trace for the same key; nothing here creates a second, and either is
            # correct.
            self._paths.setdefault(key, run.path)

    async def get_or_create(
        self, key: TraceKey, generate: Callable[[], Awaitable[TaskRun]]
    ) -> Tuple[TaskRun, bool]:
        """The trace for `key`, generating it if this task has none.

        Returns `(trace, was_generated)`. `generate` must persist the TaskRun before
        returning, stamped so the run files itself under `key`: the trace has to be
        durable before scoring is attempted (functional spec §4.1), and durable is only
        useful if the next run can find it.

        Callers racing on one key are serialized, and all but the first reuse the
        winner's trace. A `generate` that raises records nothing, leaving the key free
        for the job's retry.
        """
        async with self._locks.acquire(key):
            trace = self._load_indexed(key)
            if trace is not None:
                logger.debug("Reusing eval trace %s for %s", trace.id, key)
                return trace, False

            run = await generate()
            self._paths[key] = self._generated_path(key, run)
            logger.info("Generated eval trace %s for %s", run.id, key)
            return run, True

    def _load_indexed(self, key: TraceKey) -> TaskRun | None:
        path = self._paths.get(key)
        if path is None:
            return None
        try:
            trace = TaskRun.load_from_file(path)
        except FileNotFoundError:
            # The trace was deleted out from under us — sync, or an external delete that
            # never went through the API's 409 guard. Architecture §8's posture for a
            # missing trace is to degrade, not to cascade: drop the entry and regenerate,
            # rather than failing every job that wanted it.
            logger.warning(
                "Indexed eval trace for %s is gone from %s; regenerating", key, path
            )
            del self._paths[key]
            return None
        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            # The file exists but no longer parses as a TaskRun — truncated by a crash
            # mid-write, or rewritten by a newer schema. Same posture as a missing file:
            # drop the entry and regenerate, rather than failing every job on this key.
            logger.warning(
                "Indexed eval trace for %s at %s failed to load (%s); regenerating",
                key,
                path,
                error,
            )
            del self._paths[key]
            return None

        # The write path's check, applied on read. Entries only enter `_paths` verified,
        # so this fires only if the file changed identity after it was indexed — sync
        # replacing a record in place. Rare, and worse than the write-side mismatch it
        # mirrors: that one costs a regeneration, this one would score a judge against a
        # different item's output and produce numbers that look entirely normal.
        stored = _stored_trace_key(trace)
        if stored != key:
            logger.warning(
                "Indexed eval trace at %s now files itself under %s, not %s; regenerating",
                path,
                stored,
                key,
            )
            del self._paths[key]
            return None

        # Re-vetted on every serve, not just at seed: an indexed file can be replaced
        # after it was accepted (sync again), and an entry that was fit once is not
        # evidence it still is. Same degrade-don't-cascade posture as the checks above —
        # drop the entry and regenerate rather than failing the job.
        rejection = self._vet_rejection(key, trace)
        if rejection is not None:
            logger.warning(
                "Indexed eval trace at %s failed vetting for %s (%s); regenerating",
                path,
                key,
                rejection,
            )
            del self._paths[key]
            return None
        return trace

    def _vet_rejection(self, key: TraceKey, run: TaskRun) -> str | None:
        """Why the caller's vet refuses `run` as the trace for `key`, or None if it
        accepts.

        The reason comes back rather than a bare verdict so the index can put it in the
        line that names the file. The vet is the only thing that knows why, and a reason
        logged separately by the caller is invisible wherever the log level starts at
        warnings.

        A vet that raises counts as a rejection: a broken predicate must cost a
        regeneration, not take down every job that wanted this key.
        """
        if self._vet is None:
            return None
        try:
            return self._vet(key, run)
        except Exception as error:
            logger.warning(
                "Vetting the eval trace at %s for %s raised (%s); treating it as unusable",
                run.path,
                key,
                error,
            )
            return f"vetting raised {error}"

    def _generated_path(self, key: TraceKey, run: TaskRun) -> Path:
        """Where a freshly generated trace was persisted, once it has earned indexing.

        Enforces both halves of `generate`'s contract: persisted, and findable again.

        The index files the run under the key the *caller* asked for, but a later process
        rebuilds the index from the run's own `eval_source` and
        `output.source.run_config_id`. If those disagree, everything looks right for the
        rest of this run and the trace is never reused again — every future eval
        regenerates it, with no error to notice. So the disagreement is caught here,
        where the mistake was made.
        """
        if run.path is None:
            raise ValueError(
                f"Generating the eval trace for {key} returned an unsaved run. "
                "A trace must be persisted before it is scored, and the index holds "
                "paths rather than objects."
            )
        stored = _stored_trace_key(run)
        if stored != key:
            raise ValueError(
                f"Generating the eval trace for {key} persisted a run that files itself "
                f"under {stored}: its eval_source and output.source.run_config_id are "
                "what a later index reads. A run that disagrees with the key it was "
                "generated for is never found again, so the trace is regenerated on "
                "every eval, forever."
            )
        return run.path
