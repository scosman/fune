---
status: draft
scope: Bring the eval-builder V2 line (eb-v2) onto the shipped train/val/test splits model
depends_on: specs/projects/eval_splits_v1_v2 (must ship first)
target_branches: review/eb-v2/base, review/eb-v2/eval-v2-core, review/eb-v2/sdg, dchiang/eb-v2-merge
---

# Project: Aligning eb-v2 to the shipped eval splits model

## Summary

`eval_splits_v1_v2` replaced the eval data-source model. An eval no longer has *a* source; each of
its three splits — `test`, `train`, `val` — carries its own backing (`TaskRun` or `EvalInput`), and
one accessor resolves "which items are in split X" and "which item did this `EvalRun` score" for
every reader. `Eval.eval_input_filter_id` is gone as a field, `Eval.validate_filter_fields`
("exactly one of `eval_set_filter_id` / `eval_input_filter_id`") is gone, the train lazy migration
is gone, and `EvalRunner` takes a resolved split instead of deriving items from the eval.

The **eb-v2 line** — the eval-builder V2 work, plus its synthetic-data-generation and multi-turn
re-drive machinery — was branched from `scosman/evals_v2` before any of that, and independently
built a substantial amount of code on top of the old model: a hand-rolled eval-level source mode in
the runner, hand-rolled per-call-site source branching in `eval_api.py`, an eval-creation path that
writes an `EvalInput`-backed test set alongside `TaskRun`-backed train and golden sets, a
tombstone/supersede dedupe scheme keyed on bare ids, and a per-eval multi-turn drive config gated on
the eval's source mode.

**This project brings eb-v2 onto the shipped model.** Nothing in eb-v2's *behavior* is meant to
change: multi-turn re-drive, tombstones, and the builder's dataset shape all survive. What changes is
what expresses them.

## A note to the spec author

This document is the **what**, not the **how**. The only thing genuinely fixed is the goal above
plus the scope decisions in the next section.

Everything under "Problems detected" is a set of observations from reading both trees at the commits
named below. They are **not a design, and not a checklist**. eb-v2 is unreviewed and still moving —
treat it as a *description of a problem*, not as an authority: where eb-v2 and the shipped model
disagree about how something should work, the shipped model wins by default, but eb-v2 is often
solving a real problem the shipped model never saw (multi-turn re-drive is the clearest case) and
that problem does not go away.

You are expected to think harder about this than this document does, to find problems it missed, and
to propose a better design than any sketched here. Where a note is marked *Rough guidance*, it
reflects something real that was learned, not a decision.

### What was read, and when

| Tree | Ref | Commit |
|---|---|---|
| Shipped splits model | `claude/eval-splits-v1-v2-q38412` (→ `scosman/evals_v2`) | `40e92193` (phase 6) |
| eb-v2 integration | `origin/dchiang/eb-v2-merge` | `78081dd6` (2026-08-05) |

Common ancestor of the two lines: **`0b34c87`**. eb-v2 is 279 commits past it; the splits line
carries the `scosman/evals_v2` merge plus six phases.

> **The splits line moved after this was written.** It was rebuilt directly on
> `origin/scosman/evals_v2` as **`claude/eval-splits-evals-v2`**, because
> `claude/eval-splits-v1-v2-q38412` was cut from the still-draft PR #1517
> (`leonard/kil-686-eval-job`) and carried 52 commits of that work, merge blockers included. Same
> project, same phases, one omission: **phase 5, the background eval job worker's split support, is
> not on the new branch** — its files are #1517's. Everything this document says about the shipped
> model still holds except where it describes that worker; those places are flagged inline. Read
> `claude/eval-splits-evals-v2` as the shipped model and treat `…-q38412` as the historical record.

The three `review/eb-v2/*` branches each had 1–3 commits not yet in `dchiang/eb-v2-merge` when this
was written, all cherry-pick duplicates of commits already present there under different hashes
(same subjects). Contents converge; hashes do not. **Do not assume a single ref** — re-derive
against whatever is current when this project starts, and expect line numbers below to have moved.
The function and file names are the stable reference.

---

## Scope

**In scope**

- eb-v2 code moves onto `Eval.splits`, `SplitRef`, `resolve_split`, `ResolvedSplit`, `ItemKey` and
  `eval_run_item_key`. No eval-level source mode survives; no reader branches on which filter field
  is set.
- eb-v2's eval-creation path (the copilot spec/eval builder) constructs splits explicitly, including
  a val split, and stores each where the shipped model says it belongs.
- eb-v2's own bookkeeping — tombstone/supersede dedupe, `_counts_as_already_run`, multi-turn drive
  readiness — is re-expressed against splits and `ItemKey`, preserving its behavior.
- eb-v2 picks up the surfaces that arrive with the merge: the required `split` on the results
  endpoint, `val_dataset_size` on progress, and the TypeScript split accessor. (The background eval
  job worker and its `split` parameter are **not** among them — see §10.) Consolidating the two SSE readers the merge leaves behind
  (§10) is post-merge cleanup — in scope if cheap, droppable if not.
- The `eval_input_filter_id` load shim is removed as part of this work (see §5) — it exists on the
  shipped line only to carry internal projects across, and eb-v2 is the internal project.

**Out of scope (decided, inherited from `eval_splits_v1_v2`)**

- **Golden sets for `EvalInput`.** `eval_configs_filter_id` stays `TaskRun`-only and outside the
  splits dict. eb-v2 already agrees with this — it routes `eval_config_eval` to the golden filter
  before consulting its source mode.
- **Prompt optimization over `EvalInput`-backed train splits.** The remote service resolves the train
  filter out of the project zip's `runs/` directory; that is a different repo.
- **Any change to eb-v2's judge, SDG, or multi-turn *behavior*.** This project is a re-expression, not
  a redesign. Where a behavior cannot be preserved, say so loudly rather than quietly dropping it.

  Note what this bullet does and does not claim. Preserving eb-v2's behavior is a **scoping
  decision — not an assertion that the behavior is correct**. That code is unreviewed, and alignment
  is very likely the first time anyone reads it closely against a written spec. A bug found while
  re-expressing it is worth reporting even though fixing it is out of scope here; silently carrying
  it forward because "no behavior changes" is the failure mode this bullet must not cause.

**Deliberately unresolved here** — see "Open questions": whether eb-v2's multi-turn builder path
gains a val split at all, and what backs it.

---

## Git / PR workflow

*Rough guidance, but spec process to confirm.*

1. `eval_splits_v1_v2` lands on `scosman/evals_v2` first. Nothing here starts before that.
2. eb-v2 merges (or rebases) `scosman/evals_v2` in. Given three review branches plus an integration
   branch, decide up front **which ref alignment happens on** and whether the review branches are
   rebased after or abandoned in favour of the integration branch. Doing this work three times is a
   real risk.
3. The alignment itself is one reviewed change against that ref.
4. The `eval_input_filter_id` shim is removed in the same change or immediately after it — it is
   marked `TODO: Remove before shipping` on the splits line and eb-v2 is what it is waiting for.

---

## Background: the model eb-v2 is aligning to

Read `specs/projects/eval_splits_v1_v2/functional_spec.md` and `architecture.md` for the full
design. The seam, in files:

| What | Where |
|---|---|
| `TaskRunSplit` / `EvalInputSplit` / `SplitRef` / `EvalSplitName` | `libs/core/kiln_ai/datamodel/eval.py:839-871` |
| `LEGACY_SPLIT_FIELDS` (which splits still have a flat on-disk home) | `eval.py:873` |
| `Eval.splits` | `eval.py:917` |
| `eval_input_filter_id` → `splits["test"]` before-validator shim | `eval.py:941` |
| Legacy flat fields fold into `splits`, once, on first validation | `eval.py:968` |
| `Eval.set_split()` — store a split where old readers can still see it | `eval.py:1014` |
| Provenance-preserving serializer | `eval.py:1048` |
| `ItemKey`, `ResolvedSplit`, `resolve_split`, `eval_run_item_key` | `libs/core/kiln_ai/datamodel/eval_splits.py` |
| `EvalRunner(split=ResolvedSplit)`, one collection path | `libs/core/kiln_ai/adapters/eval/eval_runner.py:81-86,151,191` |
| `resolved_split_or_422`, `split_size`, `_cached_test_split` | `app/desktop/studio_server/eval_api.py:559,579,590` |
| `compute_score_summary(split: ResolvedSplit)` | `eval_api.py:709` |
| Background eval job worker, `split` param, split pre-resolution | `app/desktop/studio_server/jobs/workers/eval.py`, `jobs/api.py:68` — **not on `claude/eval-splits-evals-v2`**; ships with #1517 |
| Spec-eval creation factory (`build_spec_eval`) | `libs/server/kiln_server/utils/spec_utils.py:112-195` |
| TypeScript mirror of the fold | `app/web_ui/src/lib/utils/eval_splits.ts` |

Three properties matter most for reading the rest of this document:

- **Source is a property of a split, not of an eval.** An eval may have a `TaskRun`-backed train set
  and an `EvalInput`-backed test set at the same time. This is a requirement, not an accident — and
  it is exactly the shape eb-v2's multi-turn builder already produces.
- **Identity is `(source, id)`, never a bare id.** Kiln ids are twelve decimal digits from one
  generator shared by every model type, so a `TaskRun` and an `EvalInput` can collide. A bare-id
  membership test admits one item's result into another item's split, silently.
- **The legacy flat fields still exist on disk, but their in-memory values are not state.** `splits`
  is the read surface. Assigning `eval.train_set_filter_id = x` compiles, does not raise, and does
  nothing.

## Background: what eb-v2 built on the old model

Not a full inventory of eb-v2 — only the parts the splits model touches.

- **An eval-level source mode.** `EvalRunner._source_mode`, set from `eval.eval_input_filter_id` in
  the constructor, selects between two whole collection paths.
- **A multi-turn re-drive path.** `Eval.multi_turn_drive_config` (a new `MultiTurnDriveConfig` model)
  plus `EvalRunner._run_v2_multi_turn_synthetic_job` regenerate each conversation per run config,
  holding the synthetic user constant. The shipped line does not have this at all — it writes a
  `SkippedReason.incompatible_input_shape` record with detail *"V2 evals do not yet support
  multi-turn inputs"* (`eval_runner.py:420-439`).
- **Tombstones and supersede.** `EvalJob.superseded_tombstones` plus `_counts_as_already_run`
  distinguish terminal skips from recoverable ones (`missing_drive_config`, `type_not_available`), so
  a blocked item is collected again once its precondition is lifted rather than frozen out forever.
  The shipped line has none of this.
- **A hand-rolled source-aware read layer in `eval_api.py`**: `expected_item_ids_for_eval`,
  `eval_run_item_id`, `eval_input_ids_in_filter`, and a `(source, filter_id)` summary cache key.
- **A multi-turn item counter**, `multi_turn_item_count_in_filter`, reported on `EvalResultSummary`.
- **An eval-creation path that already writes mixed backings** (`copilot_api.py:1095`).

**Both lines independently reached some of the same conclusions.** eb-v2's summary cache is already
keyed `(source, filter_id)` for exactly the reason the shipped model records; eb-v2 already routes
`eval_config_eval` to the golden filter before consulting source mode, which fixes the mis-routing
the shipped design deletes `collect_tasks_for_eval_input` to prevent. Where the two agree, alignment
is mechanical. Where they agree on the *problem* and differ on the *mechanism*, prefer the shipped
one — it is typed rather than conventional — but check that eb-v2 wasn't solving a strictly larger
problem first.

---

## Problems detected

Observations, not instructions. Line references are from the commits in the table above.

### 1. The runner's eval-level source mode has to become a per-split one

`EvalRunner.__init__` sets `self._source_mode` from `target_eval.eval_input_filter_id`
(`eval_runner.py:114-116`) and `collect_tasks` dispatches on it (`:126-143`) into
`collect_tasks_for_eval_input` (`:185`) or `collect_tasks_for_task_run_eval` (`:248`). The shipped
runner has neither: it takes a `ResolvedSplit`, has one `task_run_eval` collection path, and rejects
`task_run_eval` with no split and `eval_config_eval` with one.

Three consequences worth spelling out:

- `collect_tasks_for_eval_input` and `collect_tasks_for_task_run_eval` **merge into one method**.
  Their dedupe bookkeeping differs (`run.eval_input_id` vs `run.dataset_id`, plus eb-v2's superseded
  map) and has to be reconciled onto `ItemKey` — see §3.
- eb-v2's defensive `EvalInput` + `eval_config_eval` branch in `run_job` (`:635-653`), which writes a
  persisted skipped record per item, has nothing left to handle once calibration cannot reach an
  `EvalInput`. The shipped line deleted it. Whether eb-v2 keeps it is a real decision, not a
  formality — see §12.
- `_source_mode` also gates `validate_multi_turn_drive_readiness` (`:341`) — see §9.

The shipped `ResolvedSplit` also carries `eval_id`, and the runner rejects a split resolved from a
different eval. eb-v2 has no equivalent because its items came from the eval itself.

### 2. `eval_api.py`'s per-call-site source branching, and the bare ids underneath it

`expected_item_ids_for_eval` (`eval_api.py:589`) branches on which filter field is set and returns
`Set[ID_TYPE]`; `eval_run_item_id` (`:606`) returns whichever of `dataset_id` / `eval_input_id` is
set. Four call sites use them (`:1524` progress, `:1594` eval-config score summary, `:1663` results
summary, `:1930` run-config eval scores), and `compute_score_summary` (`:687`) takes
`expected_item_ids: set[ID_TYPE]`.

The accessor supersedes all of this: `resolve_split(task, eval, "test")` plus
`eval_run_item_key(run) in resolved_split`. But the substitution is **not** name-for-name, and the
difference is the point:

- `expected_item_ids_for_eval` returns **bare ids**, so membership is `id in expected_item_ids`. Once
  an eval can hold both backings at once, two ids from different stores can compare equal.
  `eval_run_item_key` returns `(source, id)`; the shipped `compute_score_summary` keys its
  `remaining_expected_items` on `ItemKey` for exactly this reason. This is not a hypothetical fixed by
  care at each site — it is fixed by the type.
- `expected_item_ids_for_eval` returns `None` for "neither filter set", which eb-v2's callers map to a
  400 or a `continue`. The shipped model distinguishes **absent** (`resolve_split` → `None`) from
  **empty** (a `ResolvedSplit` with no items), and callers map absence three different ways: 422 for
  test, `0` for train and val, skip in the summary loop. Do not collapse that back.
- The summary cache (`:1655-1660`) already keys `(source, filter_id)` — carry that forward. The
  shipped equivalent is `_cached_test_split` (`eval_api.py:590`), which additionally re-stamps a
  cache hit with the current eval's id.

`multi_turn_item_count_in_filter` (`:562`) is a fourth branch hiding in plain sight: it counts stored
multi-turn conversations and is called only when `eval.eval_set_filter_id is not None`
(`:1609`, `:1670`), i.e. "only when the test split is `TaskRun`-backed". Expressed against
splits that is a property of the resolved split's `source`, not of a filter field — and it becomes a
question the caller has to answer per split rather than per eval, once `?split=` can name a train or
val set.

### 3. eb-v2's dedupe and tombstone bookkeeping keys on bare ids

`collect_tasks_for_eval_input` builds `already_run[eval_config][run_config] -> {eval_input_id}` and
`superseded[(eval_config_id, run_config_id, eval_input_id)]`; `collect_tasks_for_task_run_eval` builds
the identical structures over `dataset_id`. Today that is safe by construction — each method sees one
store. **Collapsing them into one method removes that guarantee**, because the shipped
`collect_tasks_for_task_run_eval` builds `already_run` from *every* run on the eval config, which
spans both stores in one set.

`_counts_as_already_run` (`:302`) is the other half: it decides whether a persisted skip is terminal
or a recoverable precondition. It doesn't touch ids itself, but its result is what goes into those
sets, so it moves with them.

This is the single place where eb-v2's own feature is most at risk of quietly breaking during
alignment — the failure mode is an item silently never re-collected, or a tombstone matched to the
wrong item, neither of which shows up as an error.

### 4. The eval-creation path

`copilot_api.py`'s `create_spec_with_copilot` is the largest single conflict.

- **It unpacks a 3-tuple that is now a 4-field NamedTuple.**
  `eval_tag, train_tag, golden_tag = generate_spec_eval_tags(request.name)` (`:1003`) and
  `generate_spec_eval_filter_ids(...)` (`:1005`). On the shipped line `generate_spec_eval_tags`
  returns `SpecEvalTags(eval_tag, train_tag, val_tag, golden_tag)` and
  `generate_spec_eval_filter_ids` **is deleted**, replaced by `spec_eval_splits` returning
  `dict[EvalSplitName, SplitRef]`. This is a hard break, not a warning. The idempotency guard at
  `:991-993` compares whole tag tuples and still works, but reads differently once there are four.
- **It constructs `Eval(...)` inline with legacy kwargs** (`:1095-1114`), including
  `eval_input_filter_id=f"tag::{eval_tag}"` for the multi-turn path and `eval_set_filter_id` for the
  single-turn one. The shipped line collapsed both creation paths into one `build_spec_eval` factory
  (`spec_utils.py:155`) precisely so construction and split-homing cannot be separated by a caller
  who forgets the second step. eb-v2's path carries several things `build_spec_eval` does not know
  about — a 409 idempotency guard, a `reference_answer` refusal, `multi_turn_drive_config`, and the
  single-turn/multi-turn fork — so either the factory grows or this path stops using it. **That is a
  design decision, not a mechanical merge**, and it is the one most likely to be got wrong quietly.
- **`create_dataset_task_runs` has diverged in both directions.** The shipped signature gained
  `val_tag` and splits the non-test remainder two-thirds train / one-third val
  (`copilot_utils.py:246`); eb-v2's gained an injectable `rng` seam and no val. Both changed the same
  function's signature and body.
- **The multi-turn path already writes mixed backings** — an `EvalInput`-backed test set from
  `build_multi_turn_eval_inputs`, `TaskRun`-backed train and golden from tagged chain leaves. This is
  the shape the shipped model exists to represent, so the translation looks mechanical. *Rough
  guidance, but spec process to confirm this is the best approach:*
  `splits={"test": EvalInputSplit(...), "train": TaskRunSplit(...)}`, with `set_split` for the
  `TaskRun`-backed train split so the prompt-optimization zip reader still sees it. **But nobody has
  decided what a val split means on the multi-turn path**, and that decision may change the shape
  above rather than just adding a key to it — see Open questions.
- `eval_builder_utils.build_transient_judge_eval_config` (`:132-136`) constructs a throwaway `Eval`
  with `eval_set_filter_id="tag::transient_eval_builder_review"`. That still validates (the fold homes
  it), so it is a choice rather than a break — but it is a construction site, and construction sites
  are where provenance is decided.

### 5. `eval_input_filter_id` is a shim on a deadline, and eb-v2 is what it is waiting for

The shipped model does **not** declare `eval_input_filter_id` as a field. A `mode="before"` validator
(`eval.py:941`) reads it out of raw input, folds it into `splits["test"]` as an `EvalInputSplit`, and
drops it — carrying the TODO *"Remove before shipping. Only internal projects contain this key."*

So today eb-v2's `Eval(eval_input_filter_id=...)` construction and its on-disk internal projects still
load. That is the shim doing its job, and it is the reason alignment is not urgent-broken. It is also
why alignment cannot be deferred indefinitely: **the shim's removal is gated on this project**.
`eval_input_filter_id` appears on 50 lines across 13 eb-v2 files, 29 of them (58%) in test files.

Note the one thing the shim refuses: an input carrying **both** `eval_set_filter_id` and
`eval_input_filter_id` raises, since folding both would silently discard one. That is the only
surviving piece of `validate_filter_fields`, and it retires with the shim.

### 6. The train lazy migration is gone

`Eval.migrate_train_set_filter_id` (eb-v2 `eval.py:1065`) stamps `tag::train_{name_slug}` onto any
eval loaded without a train filter. It is **deleted** on the shipped line, and `#1621`'s
`migrate_val_set_filter_id` was never carried forward.

The visible consequences are inherited, not new to eb-v2: eval progress reports `0` for an absent
train or val split, `?split=train` 422s for evals that have none, and prompt optimization reports no
usable train set for legacy evals that previously relied on the minted (and always empty) filter.
Evals whose minted value was already persisted by an earlier save keep it.

What eb-v2 specifically has to check: it is the line that *creates* evals with train sets, and its
builder sets them explicitly, so no builder-created eval is affected. The exposure is entirely to
pre-existing evals in test fixtures and internal projects — where a fixture that omitted
`train_set_filter_id` and relied on the migration now has no train split at all, and any assertion
about a train size silently becomes an assertion about zero.

### 7. The exactly-one-source validator disappeared, and what replaces it fails differently

`Eval.validate_filter_fields` (eb-v2 `eval.py:1101`) raises *"Exactly one of eval_set_filter_id or
eval_input_filter_id must be set"*. On the shipped line that invariant is **structural** —
`splits["test"]` is a single value, so two backings for one split is not a representable state — and
what remains is `validate_splits`: *"An eval must have a test split."*

Anything that depends on the old error is affected, in both directions:

- Code that reasons "exactly one of the two is set, so checking one implies the other" — eb-v2's
  `expected_item_ids_for_eval` docstring says exactly this — no longer has a validator behind it.
- Tests that assert the old message, or that construct a deliberately-invalid eval to prove the
  validator fires, now get a different error or none.
- Constructing an `Eval` with **neither** filter now raises "must have a test split" at construction
  rather than "exactly one must be set". Same outcome, different message and different reason.

### 8. Legacy flat fields still exist, but writing to them does nothing

This is the sharpest edge in the shipped model, and eb-v2 has many construction and assignment sites.

`eval_set_filter_id` and `train_set_filter_id` remain declared, stored, and constructible — Kiln
project files sync between clients on different app versions, so removing them would break older
builds. But the fold runs **once**, on first validation, and after that `splits` is authoritative.
`eval.train_set_filter_id = x` on a loaded eval compiles, raises nothing, and changes nothing.

Two corollaries eb-v2 has to internalize:

- **Construction with legacy kwargs still works** and is recorded as legacy-homed, so it serializes
  back to the legacy field. That is why eb-v2 is not broken today.
- **`eval.set_split(name, ref)` and `eval.splits[name] = ref` produce the same in-memory model and
  different bytes on disk.** `set_split` homes a `TaskRun`-backed test or train split into its legacy
  field so older builds — and `package_project_for_training`'s zip, read by the closed-source remote
  prompt optimizer, which is in another repo and never migrates — still see it. Picking wrong is
  silent in both directions, and invisible to any test that asserts on `eval.splits`. **Tests over
  code that sets a split must assert on `model_dump()` or the saved file.**

`eval_set_filter_id` and `train_set_filter_id` appear on **235 lines across 24 eb-v2 files**
(`git grep -c "eval_set_filter_id\|train_set_filter_id" origin/dchiang/eb-v2-merge`, less `specs/`
and the generated `api_schema.d.ts`; 257 occurrences, since some lines name both). The seven
heaviest are 188 of those:
`test_eval_model.py` 83, `test_eval_api.py` 48, `eval_api.py` 18, `test_base_eval.py` 12,
`test_eval_runner.py` 10, `eval.py` 9, the eval detail page 8.

These counts deliberately **exclude** `eval_input_filter_id` (§5's separate 50 lines), because this
section is about the two fields that remain *declared* on the shipped model. Those keep working —
and the dangerous ones are the ones that keep working *vacuously*, since a test asserting on a
legacy field passes whether or not the split logic is right. `eval_input_filter_id` is a different
problem with a different deadline: it is not a field at all on the shipped line, so its call sites
must move rather than merely stop being authoritative.

### 9. Multi-turn is an eval-level property in a per-split world

`Eval.multi_turn_drive_config` is per-eval, and `validate_multi_turn_drive_readiness` no-ops unless
`eval_run_type == "task_run_eval"` **and** `self._source_mode == "eval_input"` (`eval_runner.py:341`).
With source now per split, the direct translation is `self.split.source == "eval_input"` — which is
almost certainly right, and worth stating deliberately rather than porting by reflex:

- An eval can now have a `TaskRun`-backed test split and an `EvalInput`-backed val split. Drive
  readiness is then a property of *the run*, not of the eval, and the same eval is ready for one
  split and not another.
- Conversely, an eval with a drive config whose current split is `TaskRun`-backed must not re-drive —
  which the per-split check gives for free, and an eval-level check does not.
- `SkippedReason.missing_drive_config` and the recoverable-tombstone logic that keys on it (§3) inherit
  the same question.

Worth confirming whether `multi_turn_drive_config` should stay eval-level at all, or become a property
of the split that needs it. *Rough guidance:* eval-level is probably right — the drive config exists to
hold the synthetic user constant across run configs so a comparison varies only the agent, and that
argument is about the eval, not the split. But nobody has actually asked the question.

**2026-08-14 update: the question was asked, and the answer is neither eval-level nor split-level —
it is per item.** The drive config now lives on `MultiTurnSyntheticEvalInputData.drive_config`,
stamped at mint, and `Eval.multi_turn_drive_config` is removed. The constancy argument above survives
intact: for any given item the synthetic user is still held constant across every run config and every
eval that references it — item-to-item variation was never the contamination concern (personas already
vary per item). What per-item placement adds is self-containedness: each item carries everything needed
to re-drive it identically (persona, first message, synthetic-user model, turns), which is what makes
conversation traces reusable across evals keyed by item id. The rough guidance's dichotomy was simply
too coarse. Consequences for the mechanics this section flagged: multi-turn detection now keys on the
split items' *type* rather than "eval has a drive config", so the per-split translation question
dissolves (a `TaskRun`-backed split contains no such items and no-ops for free); readiness reads the
split's items and fails up front only when *no* item carries a config; and
`SkippedReason.missing_drive_config` plus its recoverable-tombstone logic are per-item — the tombstone
recovers when the item it names carries a config, and items are immutable once minted, so recovery in
practice means a replacement batch.

### 10. Surfaces that arrive with the merge

Mostly not conflicts — code that does not exist on eb-v2 and lands with the merge. The last bullet is
the exception, and it is the one to read carefully: eb-v2 solved the same problem separately, so the
merge produces duplication rather than a gap.

- **The background eval job system.** `app/desktop/studio_server/jobs/workers/eval.py` does not exist
  on eb-v2 at all (it has only `noop.py`); it came from the `#1621` line. It brings `EvalJobParams.split`,
  progress measured against the split's own item universe, and `jobs/api.py`'s pre-resolution so a bad
  split is a 422 at request time rather than a doomed background job. eb-v2 has no opinion about any
  of it — but its multi-turn re-drive is the most expensive eval work in the product, and it is the
  obvious future consumer.

  **It does not arrive with this merge.** The worker belongs to the still-draft PR #1517, and
  `claude/eval-splits-evals-v2` does not carry it, so none of the above is in the tree that lands
  first — including the `split` parameter on `POST /api/jobs/evals/run`. The split support for it is
  built and reviewed (phase 5, commit `369a32ef8` on `claude/eval-splits-v1-v2-q38412`) and re-lands
  when #1517 does. Plan for it as a third arrival rather than as part of this one; the runner it
  drives, `EvalRunner(split=ResolvedSplit)`, *is* here and already refuses to run without a split.
- **`split` is required on the results endpoint.** `GET .../run_config/{id}/results` takes a
  **required** `split` query parameter; omitting it is a 422 and, after schema regeneration, a
  TypeScript build failure. eb-v2's `run_result/+page.svelte` does not pass it. This is meant to be
  caught at compile time — treat a build error there as the mechanism working.
- **`val` exists everywhere.** `EvalProgress.val_dataset_size`, a val tag and val split from spec-eval
  creation, `?split=val` on run and results. eb-v2 has no notion of val at any layer.
- **`app/web_ui/src/lib/utils/eval_splits.ts`.** The API sends the same two-homed format the serializer
  writes to disk — a legacy eval arrives as `{eval_set_filter_id: "tag::x", splits: {}}`, a native one
  as `{eval_set_filter_id: null, splits: {test: {...}}}` — so the web client folds too. It is an
  unenforced mirror of `Eval.fold_legacy_filter_fields` and `LEGACY_SPLIT_FIELDS`; a one-sided change
  fails silently in the UI.
- **A second fetch-based SSE reader** (`$lib/utils/sse_stream.ts`). This one is not a gap in eb-v2's
  thinking — eb-v2 reached the same conclusion first. It has its own fetch + `ReadableStream` reader,
  `$lib/utils/sse.ts` (`sse_data_payloads`, *"EventSource is GET-only, so streaming endpoints are
  consumed via fetch + ReadableStream"*), added in `5433e2935` and used by `streaming_chat.ts` and the
  builder page. What eb-v2 did **not** move off `EventSource` is the eval-run client
  (`run_eval.svelte`), which it has not touched since the common ancestor, so it merges cleanly and
  the shipped version wins.

  Two consequences, neither a conflict. First, `$lib/utils/` ends up with **two overlapping SSE
  readers** — `sse.ts` (line-splitting only, protocol left to the caller) and `sse_stream.ts` (adds
  non-2xx handling so a refusal body is surfaced). Consolidating them is post-merge cleanup worth
  scheduling, since leaving both means the next streaming endpoint picks one by coin flip. Second,
  eb-v2's `run_comparison` handler settles for "EventSource consumers can't read a 400 body"
  (`eval_api.py:1358-1364`) — true of the eval-run client it was written against, and no longer true
  after the merge. Its drive-readiness 400 becomes user-visible for free, wording written on the
  assumption nobody would read it included.

### 11. Web UI reads

Six eb-v2 pages read the legacy filter fields directly and need the TS accessor (`eval_split`,
`eval_split_filter_id`, `task_run_split_filter_id`):

- eval detail (`[eval_id]/+page.svelte:274-283`, `:302-315`, `:531-532`, and an
  `eval_input_filter_id` multi-turn badge at `:739`)
- spec detail (`[spec_id]/+page.svelte:634`)
- compare (`compare/+page.svelte:689-690`, and a multi-turn badge at `:1090`)
- data-gen intro (`data_gen_intro.svelte:117-139`)
- prompt-optimization create (`create_prompt_optimization_job/+page.svelte:522-523`, `:1083`, `:1120`)
- eval results (`run_result/+page.svelte`)

The shipped versions of all six already do this; the merge conflict is per-hunk, and eb-v2's two
multi-turn badges are the only genuinely new reads. The distinction the accessor draws that the legacy
fields could not: a **dataset tag or `/dataset` link** addresses `task.runs()` and is meaningful only
for a `TaskRun`-backed split (`task_run_split_filter_id`), while merely *displaying* a filter id is
backing-agnostic (`eval_split_filter_id`). eb-v2's badges want the third thing — the split's `source`
— which `eval_split(...)?.source === "eval_input"` gives directly.

### 12. `EvalJob` still cannot express which item types each run type can carry

Carried over from `eval_splits_v1_v2` — see the next section for why it lands here.

`EvalJob.item: TaskRun | EvalInput` and `EvalJob.type: Literal["task_run_eval", "eval_config_eval"]`
are independent fields, so `EvalJob(item=some_eval_input, type="eval_config_eval")` type-checks on both
lines. The shipped line made it *unreachable* — calibration collects from the golden filter, which is
`DatasetFilterId`-typed — but not *unrepresentable*, and deleted the `run_job` branch that used to
handle it. **eb-v2 still has that branch** (`eval_runner.py:635-653`), explicitly as a defensive
backstop, writing a persisted `incompatible_input_shape` record per item.

So alignment forces the question the shipped line deferred: keep a defensive branch for a state
nothing constructs, or make the state unrepresentable and delete the branch. eb-v2 also adds
`superseded_tombstones` to `EvalJob`, so the dataclass is being edited during this work regardless.

---

## Two items carried over from `eval_splits_v1_v2`

Both were raised during that project's phase 4, deferred to "whichever of phases 5 and 6 next edits
`run_job` / `collect_tasks_for_eval_config_eval`", and reached by neither. They are recorded in
`specs/projects/eval_splits_v1_v2/implementation_plan.md` `## Notes`, with full write-ups in
`phase_plans/phase_4.md` and `phase_plans/phase_5.md`.

**1. `EvalJob` does not express which item types each run type can carry → this project.** It is not a
loose end that happens to be nearby; it is the same decision as §12 above. eb-v2 has a live `run_job`
branch handling the impossible combination, this project decides that branch's fate, and `EvalJob`
gains a field (`superseded_tombstones`) in the same change. Fixing it anywhere else would mean editing
the dataclass and every `isinstance` in `run_job` twice.

**2. Calibration dedupe ignores what a run was *for* → in this project's scope, and worth landing
earlier if anyone can.** `collect_tasks_for_eval_config_eval` builds `already_run` from
`run.dataset_id` over **every** run on the eval config, including `task_run_eval` ones — so a golden
`TaskRun` that has been scored for some run config is treated as already-calibrated and silently never
calibrated. It is pre-existing, present identically on both lines (eb-v2 `eval_runner.py:145-183`,
shipped `eval_runner.py:158-190`), and has **nothing to do with splits or data sources** — golden is
`TaskRun`-only everywhere.

The concrete task, small enough to run as a standalone change:

> In `EvalRunner.collect_tasks_for_eval_config_eval`, exclude runs that were not calibration runs
> when building `already_run` — filter on `run.eval_config_eval` (equivalently
> `run.task_run_config_id is None`) before adding `run.dataset_id` to the set. Test that a golden
> `TaskRun` already scored for some run config is still collected for calibration, and that a golden
> `TaskRun` already calibrated for the same eval config is still excluded.

**Sequencing.** Ideally this lands on `scosman/evals_v2` on its own, before alignment starts, so both
lines inherit the fix and this project never arbitrates it. That is the cheaper path and someone
should take it. But it has been deferred once already, by two phases in a row, so it is given an
unconditional owner here rather than left as a dependency to watch: **if it has not landed by the
time alignment begins, it is this project's**, and it is cheap to take — the merged
`collect_tasks_for_task_run_eval` sits three lines away, and nobody wants to touch that method twice.

---

## Verification notes

> **Not a test plan, and not comprehensive.** These come from reading both trees, not from running
> them. The implementing agent owns producing a complete plan. Treat anything below that proves stale
> as evidence the rest needs the same scrutiny.

Areas most likely to break quietly:

- **Tombstone and dedupe behavior across the collapsed collection path** (§3). Assert on *which items
  are collected*, per backing, with and without a recoverable tombstone present. A wrong answer here
  is an item silently never re-run — a 200 with less work done.
- **Multi-turn re-drive over an `EvalInput`-backed split, with the eval's *other* splits
  `TaskRun`-backed** (§9). The mixed-backing eval is eb-v2's actual production shape, not a corner
  case.
- **Builder-created evals round-trip** (§4, §8). Create one through the copilot path on each of the
  single-turn and multi-turn forks, save, reload, save again, and assert on the **serialized bytes** —
  that the test split landed in the intended home, that a `TaskRun`-backed train split is visible to a
  reader of the flat field, and that nothing is invented or dropped.
- **Test fixtures that relied on the train lazy migration** (§6) — now silently assert zero.
- **Test fixtures that assert on legacy fields** (§8) — now pass vacuously. These need reading, not
  mechanical fixing.
- **Cross-store id collision.** Construct a `TaskRun` and an `EvalInput` with the same id, put the
  `EvalInput` in an `EvalInput`-backed split, and assert an `EvalRun` keyed on the `TaskRun` is not a
  member. This is the test that fails if anything reverts to comparing bare ids (§2, §3).
- **`multi_turn_item_count`** (§2) — that it stays honest once the split it is counted over can be
  `EvalInput`-backed or non-test.
- **The no-change path.** Every eb-v2 behavior this project does not intend to change: single-turn
  builder creation, judge authoring, SDG, calibration against golden. If any of these move, the
  re-expression was a redesign.

---

## Open questions

1. **Does the multi-turn builder path get a val split, and what backs it?** The single-turn path gains
   a `TaskRun`-backed val split from the shipped `create_dataset_task_runs`. The multi-turn path's test
   set is `EvalInput`s minted from driven cases and its train/golden sets are tagged chain leaves, so a
   val split could plausibly be either — or be omitted, which is a legitimate answer since train and val
   are optional. Nobody has decided. This is the largest genuinely-undesigned question in the project.
2. **Does `build_spec_eval` grow to cover the builder path, or does the builder construct splits
   directly?** (§4) The shipped factory exists so construction and split-homing cannot be separated;
   eb-v2's path carries four things the factory does not know about.
3. **Which ref does alignment happen on**, and what happens to the three review branches afterwards?
4. ~~**Does `multi_turn_drive_config` stay eval-level?**~~ — settled 2026-08-14: it becomes per-item
   on `MultiTurnSyntheticEvalInputData` and the eval-level field is removed (§9's dated update).
5. **Does eb-v2's defensive `EvalInput` + `eval_config_eval` branch survive**, or does `EvalJob` become
   unable to express the combination? (§12)
6. ~~**Is `specs/projects/eval_copilot_builder_v2` describing this same line?**~~ — checked, and it is
   **a predecessor on a different axis, not adjacent scope**. Its D03/D04/D05 are all about the
   *judge* axis (`EvalConfig.config_type`): make the copilot and questionnaire-builder creation paths
   emit V2 `llm_judge` configs instead of V1 ones. eb-v2's `create_spec_with_copilot` already
   constructs `EvalConfig(config_type=EvalConfigType.v2, properties=LlmJudgeProperties(...))`, so the
   copilot half (D03) is done and that overview is stale on it. `spec_api.py` on eb-v2 creates no
   `EvalConfig` at all, so the questionnaire-builder half (D05) is genuinely unbuilt — but it is
   judge-axis work and does not belong to this project. The `eval_set_filter_id` mention in D03 is
   incidental; it predates splits and the shipped model settles it.
