import asyncio
import logging
import os
from typing import Awaitable, Callable

import pytest

from kiln_ai.adapters.eval.eval_runner import conversation_health_problem
from kiln_ai.adapters.eval.trace_index import TraceIndex, TraceKey, trace_key
from kiln_ai.datamodel import (
    DataSource,
    DataSourceType,
    Task,
    TaskOutput,
    TaskRun,
)
from kiln_ai.datamodel.eval_splits import ItemSource
from kiln_ai.datamodel.task_run import EvalItemSource, eval_item_key
from kiln_ai.utils.open_ai_types import ChatCompletionMessageParam

KEY: TraceKey = ("eval_input", "item1", "rc1")


@pytest.fixture
def task(tmp_path):
    task = Task(
        name="test",
        description="test",
        instruction="do the thing",
        path=tmp_path / "task.kiln",
    )
    task.save_to_file()
    return task


def output_source(run_config_id: str | None) -> DataSource:
    return DataSource(
        type=DataSourceType.synthetic,
        properties={
            "model_name": "gpt-4",
            "model_provider": "openai",
            "adapter_name": "test_adapter",
        },
        run_config_id=run_config_id,
    )


def save_run(
    task: Task,
    *,
    run_config_id: str | None = "rc1",
    eval_source: EvalItemSource | None = None,
    output: str = "generated",
    trace: list[ChatCompletionMessageParam] | None = None,
) -> TaskRun:
    run = TaskRun(
        parent=task,
        input="the input",
        output=TaskOutput(output=output, source=output_source(run_config_id)),
        eval_source=eval_source,
        trace=trace,
    )
    run.save_to_file()
    return run


def save_trace(
    task: Task,
    *,
    source_type: ItemSource = "eval_input",
    source_id: str = "item1",
    run_config_id: str | None = "rc1",
    output: str = "generated",
    trace: list[ChatCompletionMessageParam] | None = None,
) -> TaskRun:
    return save_run(
        task,
        run_config_id=run_config_id,
        eval_source=EvalItemSource(source_type=source_type, source_id=source_id),
        output=output,
        trace=trace,
    )


WHOLE_CONVERSATION: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "turn 1"},
    {"role": "assistant", "content": "hello"},
    {"role": "user", "content": "turn 2"},
    {"role": "assistant", "content": "goodbye"},
]

STUMP_CONVERSATION: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "turn 1"},
    {"role": "assistant", "content": "hello"},
]


def two_turn_vet(key: TraceKey, run: TaskRun) -> str | None:
    """The vet the runner installs, for an item wanting two turns.

    The real predicate rather than a hand-rolled stand-in, so these tests fail if the
    index stops asking the question the runner actually asks.
    """
    return conversation_health_problem(run.trace, 2)


class Generator:
    """Builds `generate` callables, and records how many of them ran.

    `for_key` mirrors what Phase 3's runner does: it closes over the job it is generating
    for, and stamps the persisted run with that job's item and run config. Tests that
    want the mismatch case ask for a key the index was not asked about.
    """

    def __init__(self, task: Task, *, persist: bool = True, raises: bool = False):
        self.task = task
        self.persist = persist
        self.raises = raises
        self.calls = 0
        self.rendezvous: "Rendezvous | None" = None

    def for_key(self, key: TraceKey) -> Callable[[], Awaitable[TaskRun]]:
        source_type, source_id, run_config_id = key

        async def generate() -> TaskRun:
            self.calls += 1
            # Real generation is a network call. Yielding here is what makes the
            # concurrency tests able to fail: without it, a generator that never suspends
            # would let every caller run to completion in turn, and an unlocked index
            # would look correct.
            await asyncio.sleep(0)
            if self.rendezvous is not None:
                await self.rendezvous.wait()
            if self.raises:
                raise RuntimeError("generation failed")
            if not self.persist:
                return TaskRun(
                    input="the input",
                    output=TaskOutput(
                        output="unsaved", source=output_source(run_config_id)
                    ),
                    eval_source=EvalItemSource(
                        source_type=source_type, source_id=source_id
                    ),
                )
            return save_trace(
                self.task,
                source_type=source_type,
                source_id=source_id,
                run_config_id=run_config_id,
                output=f"generated {self.calls}",
            )

        return generate


class Rendezvous:
    """Blocks each arrival until `count` of them have arrived.

    Turns "these ran concurrently" into a pass/fail rather than a timing measurement: if
    the per-key lock were global, the first generator would wait for peers that can never
    start, and the wait times out.
    """

    def __init__(self, count: int, timeout: float = 5.0):
        self.count = count
        self.timeout = timeout
        self.arrived = 0
        self._all_here = asyncio.Event()

    async def wait(self) -> None:
        self.arrived += 1
        if self.arrived >= self.count:
            self._all_here.set()
        await asyncio.wait_for(self._all_here.wait(), timeout=self.timeout)


@pytest.mark.parametrize(
    "item, run_config_id",
    [
        (("task_run", None), "rc1"),
        (("eval_input", None), "rc1"),
        (("task_run", ""), "rc1"),
        (("task_run", "item1"), None),
        (("task_run", "item1"), ""),
        (("task_run", None), None),
    ],
)
def test_trace_key_rejects_missing_ids(item, run_config_id):
    with pytest.raises(ValueError, match="needs both an item id and a run config id"):
        trace_key(item, run_config_id)


@pytest.mark.parametrize("source_type", ["eval_input", "task_run"])
def test_trace_key_from_eval_item_source(source_type: ItemSource):
    source = EvalItemSource(source_type=source_type, source_id="item1")
    assert trace_key(eval_item_key(source), "rc1") == (source_type, "item1", "rc1")


@pytest.mark.asyncio
@pytest.mark.parametrize("source_type", ["eval_input", "task_run"])
async def test_seed_reuses_existing_trace(task, source_type: ItemSource):
    existing = save_trace(task, source_type=source_type, source_id="item1")
    generate = Generator(task)
    key: TraceKey = (source_type, "item1", "rc1")

    index = TraceIndex(task)
    trace, was_generated = await index.get_or_create(key, generate.for_key(key))

    assert was_generated is False
    assert trace.id == existing.id
    assert trace.output.output == "generated"
    assert generate.calls == 0


@pytest.mark.asyncio
async def test_seed_ignores_ordinary_dataset_runs(task):
    dataset_run = save_run(task)
    generate = Generator(task)
    key: TraceKey = ("task_run", str(dataset_run.id), "rc1")

    index = TraceIndex(task)
    trace, was_generated = await index.get_or_create(key, generate.for_key(key))

    assert was_generated is True
    assert trace.id != dataset_run.id
    assert generate.calls == 1


@pytest.mark.asyncio
async def test_seed_ignores_trace_without_run_config(task):
    """Half a key matches nothing: a trace with no run config can't be claimed by a job."""
    save_trace(task, run_config_id=None)
    generate = Generator(task)

    index = TraceIndex(task)
    _, was_generated = await index.get_or_create(KEY, generate.for_key(KEY))

    assert was_generated is True
    assert generate.calls == 1


@pytest.mark.asyncio
async def test_seed_duplicate_keys_first_wins(task):
    save_trace(task, output="first")
    save_trace(task, output="second")
    # Asserted against the order the seed itself iterates, not filesystem ordering: which
    # of two synced duplicates wins is arbitrary (D8), that it is stable is not.
    expected = next(
        run
        for run in task.runs(readonly=True, include_eval_generated=True)
        if run.eval_source is not None
    )
    generate = Generator(task)

    index = TraceIndex(task)
    trace, was_generated = await index.get_or_create(KEY, generate.for_key(KEY))

    assert was_generated is False
    assert trace.id == expected.id
    assert generate.calls == 0


@pytest.mark.asyncio
async def test_generated_trace_is_reused_without_reseeding(task):
    """The live half: a trace generated during a run is visible to the rest of it.

    This is what makes a retry after a scoring failure re-score rather than regenerate.
    """
    generate = Generator(task)
    index = TraceIndex(task)

    first, first_generated = await index.get_or_create(KEY, generate.for_key(KEY))
    second, second_generated = await index.get_or_create(KEY, generate.for_key(KEY))

    assert first_generated is True
    assert second_generated is False
    assert second.id == first.id
    assert generate.calls == 1


@pytest.mark.asyncio
async def test_generated_trace_is_found_by_a_fresh_index(task):
    """The cross-run half, and the whole point of the project.

    A later eval builds a new index from disk, and must find what this one generated —
    which only works if the run's own stored fields agree with the key it was generated
    for. Nothing in a single run's behavior would reveal it if they didn't.
    """
    generate = Generator(task)
    generated, _ = await TraceIndex(task).get_or_create(KEY, generate.for_key(KEY))

    next_run = Generator(task)
    trace, was_generated = await TraceIndex(task).get_or_create(
        KEY, next_run.for_key(KEY)
    )

    assert was_generated is False
    assert trace.id == generated.id
    assert next_run.calls == 0
    assert generate.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stamped_key",
    [
        ("eval_input", "item1", "rc_other"),
        ("eval_input", "item_other", "rc1"),
        ("task_run", "item1", "rc1"),
    ],
)
async def test_generated_run_stamped_with_another_key_raises(
    task, stamped_key: TraceKey
):
    """A trace that files itself elsewhere is a silent, permanent regeneration."""
    generate = Generator(task)
    index = TraceIndex(task)

    with pytest.raises(ValueError, match="files itself under"):
        await index.get_or_create(KEY, generate.for_key(stamped_key))


@pytest.mark.asyncio
async def test_generated_run_without_eval_source_raises(task):
    """The same failure by omission: an unflagged run is not a trace at all.

    It would also be a dataset run, visible in every fine-tune set and few-shot prompt.
    """
    index = TraceIndex(task)

    async def generate_unflagged() -> TaskRun:
        return save_run(task)

    with pytest.raises(ValueError, match="files itself under None"):
        await index.get_or_create(KEY, generate_unflagged)

    retry = Generator(task)
    _, was_generated = await index.get_or_create(KEY, retry.for_key(KEY))
    assert was_generated is True


@pytest.mark.asyncio
async def test_distinct_keys_do_not_share_a_trace(task):
    generate = Generator(task)
    index = TraceIndex(task)
    keys: list[TraceKey] = [
        ("eval_input", "item1", "rc1"),
        ("eval_input", "item1", "rc2"),
        ("task_run", "item1", "rc1"),
    ]

    traces = [
        (await index.get_or_create(key, generate.for_key(key)))[0] for key in keys
    ]

    assert generate.calls == 3
    assert len({trace.id for trace in traces}) == 3
    # Each is findable on its own key by a later index, not just distinguishable now.
    fresh = TraceIndex(task)
    reused = Generator(task)
    for key, trace in zip(keys, traces):
        found, was_generated = await fresh.get_or_create(key, reused.for_key(key))
        assert was_generated is False
        assert found.id == trace.id
    assert reused.calls == 0


@pytest.mark.asyncio
async def test_concurrent_callers_on_one_key_generate_once(task):
    """The test that protects the money.

    25 is `AsyncJobRunner`'s default concurrency, and jobs collide on one key whenever
    several eval configs score the same item under the same run config.
    """
    generate = Generator(task)
    index = TraceIndex(task)

    results = await asyncio.gather(
        *(index.get_or_create(KEY, generate.for_key(KEY)) for _ in range(25))
    )

    assert generate.calls == 1
    assert [was_generated for _, was_generated in results].count(True) == 1
    assert len({trace.id for trace, _ in results}) == 1


@pytest.mark.asyncio
async def test_distinct_keys_generate_concurrently(task):
    """The per-key lock must not have become a global one."""
    generate = Generator(task)
    generate.rendezvous = Rendezvous(count=5)
    index = TraceIndex(task)
    keys: list[TraceKey] = [("eval_input", f"item{i}", "rc1") for i in range(5)]

    results = await asyncio.gather(
        *(index.get_or_create(key, generate.for_key(key)) for key in keys)
    )

    assert generate.calls == 5
    assert all(was_generated for _, was_generated in results)


@pytest.mark.asyncio
async def test_unsaved_generated_run_raises_and_is_not_indexed(task):
    unsaved = Generator(task, persist=False)
    index = TraceIndex(task)

    with pytest.raises(ValueError, match="returned an unsaved run"):
        await index.get_or_create(KEY, unsaved.for_key(KEY))

    saved = Generator(task)
    _, was_generated = await index.get_or_create(KEY, saved.for_key(KEY))
    assert was_generated is True
    assert saved.calls == 1


@pytest.mark.asyncio
async def test_failed_generation_leaves_the_key_free(task):
    """Architecture §8: generation raises, nothing is persisted, the retry tries again."""
    failing = Generator(task, raises=True)
    index = TraceIndex(task)

    with pytest.raises(RuntimeError, match="generation failed"):
        await index.get_or_create(KEY, failing.for_key(KEY))

    retry = Generator(task)
    trace, was_generated = await index.get_or_create(KEY, retry.for_key(KEY))
    assert was_generated is True
    assert retry.calls == 1
    assert trace.path is not None


@pytest.mark.asyncio
async def test_restamped_trace_file_is_not_served(task):
    """A record that changed identity under an indexed path must not be handed out.

    Only reachable via sync replacing a file in place, and worse than the write-side
    mismatch it mirrors: serving it would score this judge against another item's output,
    and the scores would look entirely normal.
    """
    existing = save_trace(task)
    index = TraceIndex(task)
    assert existing.path is not None
    restamped = TaskRun.load_from_file(existing.path)
    restamped.eval_source = EvalItemSource(
        source_type="eval_input", source_id="item_other"
    )
    restamped.save_to_file()

    generate = Generator(task)
    trace, was_generated = await index.get_or_create(KEY, generate.for_key(KEY))

    assert was_generated is True
    assert trace.id != existing.id
    assert generate.calls == 1


@pytest.mark.asyncio
async def test_vanished_trace_file_regenerates(task):
    """A deleted trace degrades to a regeneration, never an error (architecture §8)."""
    existing = save_trace(task)
    index = TraceIndex(task)
    assert existing.path is not None
    existing.path.unlink()

    generate = Generator(task)
    trace, was_generated = await index.get_or_create(KEY, generate.for_key(KEY))

    assert was_generated is True
    assert trace.id != existing.id
    assert generate.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corrupt_content",
    [
        # Truncated JSON, as an in-place write interrupted mid-file leaves behind.
        '{"v": 1, "id": ',
        # Valid JSON that no longer validates as a TaskRun.
        '{"v": 1, "model_type": "task_run"}',
        # A schema version from a newer build, refused with a plain ValueError.
        '{"v": 999999, "model_type": "task_run", "input": "i"}',
    ],
)
async def test_corrupt_trace_file_regenerates(task, caplog, corrupt_content):
    """A trace file that no longer parses degrades like a deleted one: drop the entry,
    warn, and regenerate — rather than failing every job on this key forever."""
    existing = save_trace(task)
    index = TraceIndex(task)
    assert existing.path is not None
    existing.path.write_text(corrupt_content, encoding="utf-8")
    # Bump mtime so the model cache cannot serve the previously loaded instance.
    stat = existing.path.stat()
    os.utime(existing.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    generate = Generator(task)
    with caplog.at_level(logging.WARNING, logger="kiln_ai.adapters.eval.trace_index"):
        trace, was_generated = await index.get_or_create(KEY, generate.for_key(KEY))

    assert was_generated is True
    assert trace.id != existing.id
    assert generate.calls == 1
    warning = next(r for r in caplog.records if "failed to load" in r.getMessage())
    assert warning.levelno == logging.WARNING
    assert str(existing.path) in warning.getMessage()

    # The regenerated trace is indexed: the next lookup serves it without generating.
    again, was_generated_again = await index.get_or_create(KEY, generate.for_key(KEY))
    assert was_generated_again is False
    assert again.id == trace.id


class TestVettedReuse:
    """The index reuses on key identity; `vet` is where a caller adds "and it is fit"."""

    @pytest.mark.asyncio
    async def test_seed_skips_an_unfit_candidate_so_a_fit_one_wins(self, task, caplog):
        """An unfit record must not shadow a fit one at the same key, and rejecting it
        must not touch the file."""
        save_trace(task, trace=WHOLE_CONVERSATION, output="one")
        save_trace(task, trace=WHOLE_CONVERSATION, output="two")
        # The candidate the scan reaches FIRST is the broken one, so first-wins alone
        # would serve the stump to every judge of this key.
        scanned = [
            run
            for run in task.runs(readonly=True, include_eval_generated=True)
            if run.eval_source is not None
        ]
        unfit, fit = scanned[0], scanned[1]
        assert unfit.path is not None
        stumped = TaskRun.load_from_file(unfit.path)
        stumped.trace = STUMP_CONVERSATION
        stumped.save_to_file()
        bytes_before = unfit.path.read_bytes()
        generate = Generator(task)

        with caplog.at_level(logging.INFO, logger="kiln_ai.adapters.eval.trace_index"):
            index = TraceIndex(task, vet=two_turn_vet)
            trace, was_generated = await index.get_or_create(KEY, generate.for_key(KEY))

        assert was_generated is False
        assert trace.id == fit.id
        assert generate.calls == 0

        rejects = [r for r in caplog.records if "failed vetting" in r.getMessage()]
        assert len(rejects) == 1
        assert rejects[0].levelno == logging.INFO
        assert str(unfit.path) in rejects[0].getMessage()
        # The vet's reason rides the same line as the file it rejected.
        assert "expected 2 user turns, found 1" in rejects[0].getMessage()

        # Rejection is a log line and nothing else: the file stays exactly as it was.
        assert unfit.path.exists()
        assert unfit.path.read_bytes() == bytes_before

    @pytest.mark.asyncio
    async def test_every_candidate_at_a_key_can_be_unfit(self, task):
        """No fit candidate means no reuse at all, rather than serving the least bad."""
        save_trace(task, trace=STUMP_CONVERSATION, output="stump one")
        save_trace(task, trace=None, output="stump two")
        generate = Generator(task)

        index = TraceIndex(task, vet=two_turn_vet)
        trace, was_generated = await index.get_or_create(KEY, generate.for_key(KEY))

        assert was_generated is True
        assert generate.calls == 1
        assert trace.output.output == "generated 1"

    @pytest.mark.asyncio
    async def test_without_a_vet_an_unfit_trace_is_still_served(self, task):
        """The default is permissive: callers with no fitness question keep key-identity
        reuse exactly as it was."""
        existing = save_trace(task, trace=STUMP_CONVERSATION)
        generate = Generator(task)

        trace, was_generated = await TraceIndex(task).get_or_create(
            KEY, generate.for_key(KEY)
        )

        assert was_generated is False
        assert trace.id == existing.id
        assert generate.calls == 0

    @pytest.mark.asyncio
    async def test_indexed_trace_that_became_a_stump_is_re_vetted_at_serve_time(
        self, task, caplog
    ):
        """Passing the vet at seed is not a permanent pass: sync can replace the file
        with a different conversation after it was indexed."""
        existing = save_trace(task, trace=WHOLE_CONVERSATION)
        index = TraceIndex(task, vet=two_turn_vet)
        assert existing.path is not None

        replaced = TaskRun.load_from_file(existing.path)
        replaced.trace = STUMP_CONVERSATION
        replaced.save_to_file()
        # Bump mtime so the model cache cannot serve the previously loaded instance.
        stat = existing.path.stat()
        os.utime(existing.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        bytes_on_disk = existing.path.read_bytes()

        generate = Generator(task)
        with caplog.at_level(
            logging.WARNING, logger="kiln_ai.adapters.eval.trace_index"
        ):
            trace, was_generated = await index.get_or_create(KEY, generate.for_key(KEY))

        assert was_generated is True
        assert trace.id != existing.id
        assert generate.calls == 1
        warning = next(r for r in caplog.records if "failed vetting" in r.getMessage())
        assert warning.levelno == logging.WARNING
        assert str(existing.path) in warning.getMessage()
        # Serve-time rejection carries the reason too, so warnings-and-up logging is
        # enough to see why a trace was regenerated.
        assert "expected 2 user turns, found 1" in warning.getMessage()

        # The rejected file is left alone here too, even though it was indexed.
        assert existing.path.read_bytes() == bytes_on_disk

    @pytest.mark.asyncio
    async def test_a_raising_vet_counts_as_a_rejection(self, task, caplog):
        """A broken predicate costs a regeneration, never every job on the key."""
        save_trace(task, trace=WHOLE_CONVERSATION)

        def exploding_vet(key: TraceKey, run: TaskRun) -> str | None:
            raise RuntimeError("vet blew up")

        generate = Generator(task)
        with caplog.at_level(
            logging.WARNING, logger="kiln_ai.adapters.eval.trace_index"
        ):
            index = TraceIndex(task, vet=exploding_vet)
            trace, was_generated = await index.get_or_create(KEY, generate.for_key(KEY))

        assert was_generated is True
        assert generate.calls == 1
        assert trace.path is not None
        assert any("vet blew up" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "turns, expect_generated",
        [(2, False), (3, True)],
    )
    async def test_a_migrated_flat_trace_is_vetted_like_any_other(
        self, task, turns: int, expect_generated: bool
    ):
        """Migration synthesizes traces from legacy records and files them under the same
        keys, so they enter the index unvetted — a partial one must not be served."""
        migrated = save_run(
            task,
            eval_source=EvalItemSource(source_type="eval_input", source_id="item1"),
            output="goodbye",
            trace=WHOLE_CONVERSATION,
        )
        generate = Generator(task)

        def vet(key: TraceKey, run: TaskRun) -> str | None:
            return conversation_health_problem(run.trace, turns)

        index = TraceIndex(task, vet=vet)
        trace, was_generated = await index.get_or_create(KEY, generate.for_key(KEY))

        assert was_generated is expect_generated
        assert generate.calls == (1 if expect_generated else 0)
        if not expect_generated:
            assert trace.id == migrated.id
        assert migrated.path is not None and migrated.path.exists()
