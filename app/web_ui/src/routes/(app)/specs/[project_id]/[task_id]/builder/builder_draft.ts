// Draft persistence for the eval-builder wizard (the SDG synth pattern:
// IndexedDB, per-task key, silent restore). Navigation or a reload can no
// longer destroy a session's authoring work: the spec fields and the
// approved batch plan restore silently; the batch-tag bookkeeping restores
// so chains a lost session left on disk are still cleaned up by the next
// drive (without it they'd be orphaned forever). Driven/review state is
// deliberately NOT persisted — it depends on disk state that can change
// under a saved draft, so restore never lands past the plan screen.

import type { ModelChoice } from "$lib/eval/default_judge"
import type { KilnAgentRunConfigProperties } from "$lib/types"
import type { SuggestedEdit } from "../spec_utils"

// A generated synthetic-user case as the wire carries it: the seed message,
// the persona blob, and the plan scenario it came from.
export type SyntheticUserCaseWire = {
  seed_prompt: string
  synthetic_user_info: string
  scenario_index?: number | null
}

// The generate_cases output, cached against exactly the inputs it depends
// on (the approved prompts + the spec text — the run config plays no part
// in persona generation). A re-drive with both unchanged reuses the cases
// instead of re-paying the multi-minute copilot generation, which makes the
// fix-config-then-drive-again recovery loop fast.
export type CachedSuCases = {
  prompts_json: string
  spec_text: string
  cases: SyntheticUserCaseWire[]
}

// The cached cases iff they were generated from EXACTLY this plan and spec
// (byte-compare); anything else means regenerate. Fresh variation on the
// same scenarios is deliberately not offered — that's what Refine Plan
// is for.
export function reusable_cached_cases(
  cache: CachedSuCases | null,
  approved_prompts: string[],
  spec: string,
): SyntheticUserCaseWire[] | null {
  if (!cache || cache.cases.length === 0) return null
  if (cache.spec_text !== spec) return null
  if (cache.prompts_json !== JSON.stringify(approved_prompts)) return null
  return cache.cases
}

// A run config as a cache key: object keys sorted and the tool list ordered,
// so two configs that would generate the same data key the same however they
// were assembled. Every field of the config counts, because every field
// reaches the generation call.
export function run_config_cache_key(
  properties: KilnAgentRunConfigProperties,
): string {
  const ordered_tools = {
    ...properties,
    tools_config: {
      ...properties.tools_config,
      tools: [...(properties.tools_config?.tools ?? [])].sort(),
    },
  }
  return JSON.stringify(ordered_tools, (_key, value) =>
    value && typeof value === "object" && !Array.isArray(value)
      ? Object.fromEntries(
          Object.entries(value as Record<string, unknown>).sort(([a], [b]) =>
            a < b ? -1 : a > b ? 1 : 0,
          ),
        )
      : value,
  )
}

// The single-turn arm's minted test inputs, cached against exactly what
// produced them: the approved plan prompts, the whole run config the input
// generator ran under, and the grounding data-guide the mint ran under (the
// spec plays no part — it already shaped the PLAN). A re-run with all three
// unchanged reuses the inputs instead of re-paying one generation call per
// prompt, which makes the fix-config-then-run-again recovery loop fast.
export type CachedMintedInputs = {
  prompts_json: string
  // The grounding guide passed to the mint (null = ungrounded). Pre-guide
  // drafts restore without the key and simply miss the cache.
  data_guide?: string | null
  // run_config_cache_key of the config the mint ran under. Model, tools,
  // skills and sampling all change what gets written, so the key is the whole
  // config rather than a few fields of it. Drafts written before the input
  // lane carried a config restore without the key and miss the cache, which
  // re-mints rather than serving inputs we can no longer describe.
  run_config_json?: string
  // Each input as the string the pipeline runs on (structured-task inputs
  // are JSON strings — the same encoding the saved eval's items store).
  inputs: string[]
}

export function reusable_minted_inputs(
  cache: CachedMintedInputs | null,
  approved_prompts: string[],
  data_guide: string | null,
  run_config_json: string,
): string[] | null {
  if (!cache || cache.inputs.length === 0) return null
  if ((cache.data_guide ?? null) !== data_guide) return null
  if (cache.run_config_json !== run_config_json) return null
  if (cache.prompts_json !== JSON.stringify(approved_prompts)) return null
  return cache.inputs
}

// Whether the clarify questions still fit the text the caller pairs them
// against: source is a snapshot of the exact text they were generated from,
// byte-compared because the copilot saw that raw string. Null means no set
// has landed, so regenerate.
export function questions_are_current(
  source: string | null,
  paired_against: string,
): boolean {
  return source !== null && source === paired_against
}

// Whether a suggested eval name may be written into the name field. A name
// the user typed is never overwritten; an untouched field is always safe to
// fill. Untouched means either empty/whitespace-only (nothing to lose, and a
// stray space is not a typed name) or still byte-equal to prefilled_name, the
// name the MACHINE last wrote — that one tracks the model's latest suggestion,
// so rewriting the description and continuing again renames the eval after the
// new one. Raw byte compare: a trailing space is a real edit.
export function should_prefill_suggested_name(
  current_name: string,
  prefilled_name: string | null,
): boolean {
  return current_name.trim() === "" || current_name === prefilled_name
}

// Whether a failed refine may discard the refine form's current values.
// Content the user may have edited is never discarded (current must still be
// byte-equal to what the code last wrote); programmatic content is discarded
// only when the source values it was derived from have changed underneath
// it. A transient failure with an unchanged source therefore keeps the good
// refinement, and a null snapshot (values restored from a saved draft, where
// authorship is unknowable) always means keep.
// The byte-compares rely on callers snapshotting the object they just built
// with the same construction on both sides, not on JSON canonicalization.
export function should_invalidate_refined_values(
  current: Record<string, string | null>,
  programmatic_json: string | null,
  derived_from_json: string | null,
  current_source_json: string,
): boolean {
  return (
    programmatic_json !== null &&
    JSON.stringify(current) === programmatic_json &&
    derived_from_json !== current_source_json
  )
}

export type BuilderDraft = {
  // Step 1-3 — spec authoring.
  description: string
  // Which text the last Continue processed. The clarify gate pairs questions
  // to this, so a reload must not swap the pairing target to an un-Continued
  // edit; drafts written before this key restore it as null.
  continued_description: string | null
  name: string
  // Which name the machine last wrote, so a reload keeps a still-untouched
  // suggestion replaceable; drafts from before this key restore null, which
  // safely treats the restored name as the user's own. Deliberately unlike
  // refined_values_programmatic_json, whose snapshot is NOT persisted so refine
  // text always restores as user-owned: discarding that text destroys content,
  // while a name suggestion is cheap to replace.
  prefilled_name: string | null
  property_values: Record<string, string | null>
  refined_property_values: Record<string, string | null>
  suggested_edits: Record<string, SuggestedEdit>
  // Step 4 — the approved plan (minutes of copilot work to recreate).
  batch_plan: { prompts: string[]; summary: string } | null
  batch_plan_edited: boolean
  // The plan's generated synthetic users (more minutes) — reused on drive
  // while plan+spec are byte-unchanged, revalidated in reusable_cached_cases.
  cached_su_cases: CachedSuCases | null
  // The single-turn arm's minted inputs — reused on a re-run while
  // plan+model are byte-unchanged, revalidated in reusable_minted_inputs.
  cached_minted_inputs: CachedMintedInputs | null
  // The auto-picked task sample grounding single-turn planning and input
  // minting — persisted beside the plan it grounded, so a restored session
  // mints with the same grounding (and saves the same provenance record).
  grounding_sample: { input: string; output: string } | null
  // The task's saved Data Guide as it read when Step 4 was entered (null when
  // the task had none, or the read failed), and whether the user keeps it in
  // the single-turn requests. Persisted beside the plan for the same reason as
  // the grounding sample: a restored session must mint under exactly what its
  // plan was drafted with, not under whatever the guide says by then. Drafts
  // written before guides were read here restore as no guide, off.
  data_guide_text: string | null
  use_data_guide: boolean
  // The user chose Continue Without Data Guide on this draft: Step 4 plans
  // without offering one again, on this session and on a reload.
  data_guide_skipped: boolean
  // The offer was on screen when the draft was last written: Step 4 had
  // been reached and was waiting on the user's choice, with no plan yet.
  // A restore lands such a draft back on Step 4, where the guide is read
  // again (the user may have just saved one) and the flow resumes. Every
  // other draft without a plan restores no further than the refine step,
  // as before.
  data_guide_offer_pending: boolean
  // Batch-tag bookkeeping — a CORRECTNESS carry, not convenience: these
  // name runs already on disk. The per-arm live-batch tag plus
  // undeleted_batch_tags, the delete-on-next-drive cleanup list (shared —
  // a task is one arm, so its tags never mix).
  multi_turn_batch_tag: string | null
  single_turn_batch_tag: string | null
  undeleted_batch_tags: string[]
  // The Drive Settings model lanes. Persisted so a reload or the
  // connect-a-provider round trip keeps the user's picks; pre-Drive-Settings
  // drafts restore these as null (?? below) and fall back to pre-population.
  su_driver: ModelChoice | null
  input_generator: ModelChoice | null
  // The whole run config the input generator was last committed with: model,
  // tools, skills and sampling. Kept beside the model lane because every part
  // of it changes what gets written, and because a restored session must be
  // able to run again without reopening Generation Settings. Drafts written
  // before the input lane carried a config restore this as null, which reads
  // as "nothing chosen".
  input_gen_run_config: KilnAgentRunConfigProperties | null
  judge_model: ModelChoice | null
  // The Generation Settings conversation length (multi-turn). Persisted for
  // the same reason as the lanes above: a reload should not silently put the
  // drive back on the default length. Null means no choice on record — drafts
  // written before this key restore that way — and the page falls back to its
  // default turn count.
  turns_per_case: number | null
  // The run config the entry page chose for this eval. Persisted so a reload
  // keeps evaluating the same thing; drafts written before the entry page
  // asked restore this as null, which reads as "nothing chosen" and leaves
  // the task default in charge.
  target_run_config_id: string | null
}

export const EMPTY_BUILDER_DRAFT: BuilderDraft = {
  description: "",
  continued_description: null,
  name: "",
  prefilled_name: null,
  property_values: {},
  refined_property_values: {},
  suggested_edits: {},
  batch_plan: null,
  batch_plan_edited: false,
  cached_su_cases: null,
  cached_minted_inputs: null,
  grounding_sample: null,
  data_guide_text: null,
  use_data_guide: false,
  data_guide_skipped: false,
  data_guide_offer_pending: false,
  multi_turn_batch_tag: null,
  single_turn_batch_tag: null,
  undeleted_batch_tags: [],
  su_driver: null,
  input_generator: null,
  input_gen_run_config: null,
  judge_model: null,
  turns_per_case: null,
  target_run_config_id: null,
}

export function builder_draft_key(project_id: string, task_id: string): string {
  return `eval_builder_draft_${project_id}_${task_id}_v1`
}

// A dev-only fetch mock may mark the window when installed. Drafts are
// gated on it in BOTH directions: canned mock state must never persist into
// a real session, and a real draft must never restore under the mock.
// Checking the flag rather than importing keeps mock code out of production
// builds.
export function builder_mock_active(): boolean {
  return (
    typeof window !== "undefined" && "__KILN_BUILDER_MOCK_ACTIVE__" in window
  )
}

function record_has_content(record: Record<string, string | null>): boolean {
  return Object.values(record).some((v) => (v ?? "").trim() !== "")
}

// Whether a stored draft carries anything worth restoring. Batch tags alone
// count: even a draft with no authoring content still names on-disk chains
// that the next drive must clean up.
export function draft_has_content(draft: BuilderDraft): boolean {
  return (
    draft.description.trim() !== "" ||
    draft.name.trim() !== "" ||
    record_has_content(draft.property_values) ||
    record_has_content(draft.refined_property_values) ||
    Object.keys(draft.suggested_edits).length > 0 ||
    draft.batch_plan !== null ||
    draft.multi_turn_batch_tag !== null ||
    draft.single_turn_batch_tag !== null ||
    draft.undeleted_batch_tags.length > 0
  )
}

// Whether a stored draft has authoring work to resume: the spec fields or a
// plan. Batch tags alone do not count here, unlike draft_has_content: a Reset
// keeps them for cleanup, and a draft that is only cleanup bookkeeping is
// nothing to continue. Drives the Evals page's button and the builder's
// Reset, both of which speak to the user about their work.
export function draft_is_resumable(draft: BuilderDraft): boolean {
  return (
    draft.description.trim() !== "" ||
    draft.name.trim() !== "" ||
    record_has_content(draft.property_values) ||
    record_has_content(draft.refined_property_values) ||
    Object.keys(draft.suggested_edits).length > 0 ||
    draft.batch_plan !== null
  )
}

// The furthest SAFE step a restored draft can land on — never past the plan
// screen (step 4), and never into review: review state isn't persisted, and
// presenting stale results would be worse than replaying a drive.
//   - a plan exists → the plan screen ("generate")
//   - the Data Guide offer was open → the plan screen, where the offer
//     re-opens (or, once a guide exists, the plan fires under it)
//   - refine output exists → "refine" (clarify Q&A isn't persisted, so a
//     draft that died mid-clarify restarts from the description)
//   - otherwise → "describe" with the description prefilled
export type RestoreStep = "describe" | "refine" | "generate"

export function restore_step(draft: BuilderDraft): RestoreStep {
  if (draft.batch_plan !== null && draft.batch_plan.prompts.length > 0) {
    return "generate"
  }
  if (draft.data_guide_offer_pending) {
    return "generate"
  }
  if (
    record_has_content(draft.refined_property_values) ||
    Object.keys(draft.suggested_edits).length > 0
  ) {
    return "refine"
  }
  return "describe"
}

// A fresh draft that KEEPS the batch-tag bookkeeping. Reset must not wipe
// the tags: they name runs already on disk, and delete-on-next-drive is
// the only thing that ever cleans those up — a full wipe would orphan them
// forever (the leak the draft's tag persistence exists to prevent).
export function reset_draft_keeping_tags(draft: BuilderDraft): BuilderDraft {
  return {
    ...EMPTY_BUILDER_DRAFT,
    multi_turn_batch_tag: draft.multi_turn_batch_tag,
    single_turn_batch_tag: draft.single_turn_batch_tag,
    undeleted_batch_tags: draft.undeleted_batch_tags,
  }
}

// After a multi-turn save, the just-saved batch's chains ARE the eval's data
// — never delete them. Earlier aborted drives can still have left superseded
// chains on disk (undeleted_batch_tags); wiping the whole draft would orphan
// those forever, since delete-on-next-drive is their only cleanup. Carry ONLY
// those leftover tags — with the saved batch's own tag excluded — into an
// otherwise-empty draft, so a later drive on this task cleans them up. The
// saved batch is dropped from both tag slots so no future replace_batch_tags
// can delete the eval's chains.
export function draft_after_save_keeping_stranded_tags(
  saved_batch_tag: string,
  undeleted_batch_tags: string[],
): BuilderDraft {
  return {
    ...EMPTY_BUILDER_DRAFT,
    undeleted_batch_tags: undeleted_batch_tags.filter(
      (tag) => tag !== saved_batch_tag,
    ),
  }
}

// The Evals page's create button advertises a resumable draft (the silent
// restore already happens on builder entry; this makes it discoverable).
// Copilot-gated: without copilot the button routes to the legacy flow,
// where no draft exists.
export function create_eval_button_label(
  has_copilot: boolean,
  has_draft: boolean,
): string {
  return has_copilot && has_draft ? "Continue Eval Draft" : "Create Eval"
}

// Where that button goes. A draft continues in the builder, which restores
// it on entry; everything else starts on the Create Eval page. Kept
// beside the label so the two cannot promise different things.
export function create_eval_destination(
  has_copilot: boolean,
  has_draft: boolean,
): "builder" | "select_template" {
  return has_copilot && has_draft ? "builder" : "select_template"
}
