// Step 4 plan-flow state logic, extracted pure so the stop screen and the
// preparing-review gate are unit-testable: the stop banner's wording, the
// three-tier destructive-action confirms, and the gate's resolved counting.

import type { ClaimsBuildState } from "./claim_evidence"

// The model lanes a drive spends through, checked before anything runs.
// Multi-turn preflights run config + synthetic-user driver + judge;
// single-turn preflights run config + input generator + judge. Lane order
// IS blame order: the target run config is the likeliest culprit and the
// one the user can fix in-app.
export type PreflightLane =
  | "run config"
  | "synthetic-user driver"
  | "input generator"
  | "judge"

export type PreflightOutcome = {
  lane: PreflightLane
  ok: boolean
  // The failure diagnosis (the route's unwrapped root provider error).
  message?: string | null
  // The lane's model ("gpt_4o via openrouter") — kept for telemetry;
  // Kiln-chosen lanes don't render it.
  model?: string | null
  // The lane's provider display name ("OpenRouter") — the one specific,
  // factual thing the generic lanes DO render: which key the step needs.
  provider?: string | null
}

export type PreflightFailure = {
  lane: PreflightLane
  message: string
  model: string | null
  provider: string | null
}

// The unified stop outcome: a drive that ended short of the approved plan.
// survivors = judged conversations; failed = post-retry case failures plus
// upstream salvage drops.
export type DriveStop = {
  survivors: number
  failed: number
  dominant_error: string | null
  // Set when a config-scoped failure aborted the whole batch server-side
  // (the batch_aborted frame) — the banner then leads with the abort
  // diagnosis instead of per-case counts.
  aborted_error?: string | null
  // Set when a lane failed its pre-drive check: NOTHING ran — no cases
  // generated, no spend. The banner leads with the lane's diagnosis.
  preflight?: PreflightFailure | null
}

// ── Conversation length ───────────────────────────────────────────────────

// The range the multi-turn drive route accepts for its turn count
// (multiturn_sdg_api's `turns` field is ge=1, le=20). Mirrored here so the
// stepper and every restored value stay inside what the route will take.
export const MIN_TURNS_PER_CASE = 1
export const MAX_TURNS_PER_CASE = 20

// The turn count a drive actually runs at. A saved draft can carry a value
// from an older range (or no number at all), so every reader goes through
// this instead of trusting the stored one — that way the quote, the request,
// and the stamp can never describe a length the route would reject. A
// non-numeric value falls back to the minimum, matching the stepper's own
// blank-entry behavior.
export function clamp_turns_per_case(turns: number): number {
  if (!Number.isFinite(turns)) return MIN_TURNS_PER_CASE
  return Math.min(
    MAX_TURNS_PER_CASE,
    Math.max(MIN_TURNS_PER_CASE, Math.round(turns)),
  )
}

// The length a restored draft starts at. The clamp above exists to pull a
// number from an older range back inside today's bounds, so only a genuine
// number is worth clamping: a draft with no value on record — or one whose
// stored value isn't a finite number at all — has expressed no choice, and
// restoring it as the clamp's minimum would quietly hand the user a
// one-turn run they never picked. Those restore the default instead.
export function restore_turns_per_case(
  stored: number | null | undefined,
  fallback: number,
): number {
  if (typeof stored !== "number" || !Number.isFinite(stored)) return fallback
  return clamp_turns_per_case(stored)
}

// ── Top-off drive planning ────────────────────────────────────────────────
//
// A retry after a partial drive must fill the batch, not replace it: the
// completed cases are paid results, and re-driving them re-bills every one
// while the previous batch's chains are deleted as superseded. The drive
// plan below decides, before anything is spent, whether the next drive can
// TOP OFF the current batch — drive only the missing slots, into the SAME
// batch tag, so the new results land beside the kept ones and review/save
// read one whole batch — or must start fresh (replace semantics).

export type DrivePlan<T> = {
  // What this drive sends the pipeline: every item on a fresh drive, only
  // the missing slots' items on a top-off.
  items: T[]
  // Maps the stream's case_index (a position in `items`) back to its slot
  // in the batch. Identity on a fresh drive.
  slot_of_stream_index: number[]
  // The batch tag to drive into: the current batch's tag on a top-off
  // (retried cases join the same box); null lets the server mint one.
  batch_tag: string | null
  // Tags superseded by this drive. A top-off never lists its own batch —
  // deleting the batch being topped off would destroy the paid successes
  // being kept.
  replace_batch_tags: string[]
  top_off: boolean
}

export function missing_slot_indices<R>(
  slots: readonly (R | null)[],
): number[] {
  return slots.flatMap((s, i) => (s === null ? [i] : []))
}

export function plan_drive<T, R>(args: {
  // The item list resolved for this attempt (cases or inputs).
  items: T[]
  // The current batch's item list, or null when no batch exists.
  batch_items: T[] | null
  // The current batch's per-slot results (null = missing).
  built_slots: readonly (R | null)[]
  batch_tag: string | null
  undeleted_batch_tags: string[]
}): DrivePlan<T> {
  const missing = missing_slot_indices(args.built_slots)
  // Top-off requires an existing batch with something to keep AND something
  // to fill, driven from byte-identical items — a changed plan or spec
  // resolves different items, and mixing them into the old batch would put
  // results the user never planned together under one tag. The slot/item
  // length agreement is a hard requirement: a divergence would map missing
  // slots onto the wrong (or no) items.
  const can_top_off =
    args.batch_tag !== null &&
    args.batch_items !== null &&
    args.built_slots.length === args.batch_items.length &&
    missing.length > 0 &&
    missing.length < args.built_slots.length &&
    JSON.stringify(args.items) === JSON.stringify(args.batch_items)
  if (!can_top_off) {
    return {
      items: args.items,
      slot_of_stream_index: args.items.map((_, i) => i),
      batch_tag: null,
      replace_batch_tags: [...args.undeleted_batch_tags],
      top_off: false,
    }
  }
  return {
    items: missing.map((i) => args.items[i]),
    slot_of_stream_index: missing,
    batch_tag: args.batch_tag,
    replace_batch_tags: args.undeleted_batch_tags.filter(
      (t) => t !== args.batch_tag,
    ),
    top_off: true,
  }
}

// Whether the current batch was produced under the same drive settings this
// attempt would run with. A changed judge — or, on multi-turn, a changed
// synthetic-user model or conversation-length ceiling — forces a fresh batch
// instead of a top-off: one review may not mix two judges' verdicts, and the
// saved drive stamp must describe the settings every conversation in the
// batch ran under. `su` and `turns` are omitted entirely on the single-turn
// arm, which has neither a synthetic-user lane nor conversations to length.
export function drive_lanes_unchanged(args: {
  judge: unknown
  batch_judge: unknown | null
  su?: unknown
  batch_su?: unknown | null
  turns?: number
  batch_turns?: number | null
}): boolean {
  if (args.batch_judge === null) return false
  if (JSON.stringify(args.judge) !== JSON.stringify(args.batch_judge)) {
    return false
  }
  if (args.su !== undefined) {
    if (args.batch_su === null || args.batch_su === undefined) return false
    if (JSON.stringify(args.su) !== JSON.stringify(args.batch_su)) return false
  }
  if (args.turns !== undefined) {
    // `turns` is a ceiling, so a batch legitimately holds conversations of
    // several lengths — the synthetic user can end one early. What has to
    // match is the ceiling they all ran under, which is what the stamp
    // records. Comparing the conversations' observed lengths instead would
    // wrongly reject a top-off onto any batch where a conversation ended
    // early.
    if (args.batch_turns === null || args.batch_turns === undefined) {
      return false
    }
    if (args.turns !== args.batch_turns) return false
  }
  return true
}

// Compact a batch's slots into the review list, preferring the live review
// entry where one exists for the same trace: claims built (or any later
// enrichment of) a kept case must survive a top-off's compaction — the
// slots hold each case as its drive produced it, not as review evolved it.
export function compact_batch_slots<T extends { trace_id: string }>(
  slots: readonly (T | null)[],
  current: readonly T[],
): T[] {
  const live = new Map(current.map((t) => [t.trace_id, t]))
  return slots
    .filter((t): t is T => t !== null)
    .map((t) => live.get(t.trace_id) ?? t)
}

// The one failure the banner reports, chosen in the given (blame) order —
// deterministic regardless of which lane's ping lost the race. All lanes
// are checked concurrently, so with several dead lanes the user fixes them
// front-to-back across re-drives.
export function first_preflight_failure(
  outcomes: PreflightOutcome[],
): PreflightFailure | null {
  const failed = outcomes.find((o) => !o.ok)
  if (!failed) return null
  return {
    lane: failed.lane,
    message: failed.message || "the model did not respond",
    model: failed.model ?? null,
    provider: failed.provider ?? null,
  }
}

// Dominant per-case error: the most frequent case_failed message.
// Config-class failures repeat identically, so the mode IS the diagnosis;
// ties resolve to the first seen. Blank messages are ignored.
export function dominant_failure_message(messages: string[]): string | null {
  const counts = new Map<string, number>()
  for (const m of messages) {
    if (!m) continue
    counts.set(m, (counts.get(m) ?? 0) + 1)
  }
  let best: string | null = null
  let best_count = 0
  for (const [m, c] of counts) {
    if (c > best_count) {
      best = m
      best_count = c
    }
  }
  return best
}

// The centred stop screen has room for a few wrapped lines of diagnosis (its
// column fits roughly 40 characters a line); the full text stays in the
// console/log. A provider message can run to hundreds of characters, which
// would push the screen's actions below the fold.
export const STOP_SCREEN_ERROR_EXCERPT_CHARS = 160

// A short excerpt of the diagnosis, cut to fit the stop screen: the message's
// first non-blank line without the trailing period the surrounding sentence
// supplies, shortened to the bound above. A shortened excerpt ends in an
// ellipsis, which stands in for the text that was dropped.
function error_excerpt(message: string): string {
  // Some messages arrive with their newlines still escaped as the two
  // characters backslash-n. The renderer turns those into real newlines before
  // it splits into lines, so unescaping here is what keeps the excerpt to one
  // rendered line instead of quoting the whole message as if it were one.
  const normalized = message.replace(/\\n/g, "\n")
  // The first line that has content, not simply the first: a message that opens
  // with a blank line would otherwise excerpt to nothing and drop the diagnosis
  // from the screen. The line is trimmed on its own, so an indented one is
  // quoted without its indent.
  const first_line = (normalized.split("\n").find((l) => l.trim() !== "") ?? "")
    .trim()
    .replace(/[.\s]+$/, "")
  // Cut by code points, not UTF-16 units: an emoji or other non-BMP character
  // sliced in half leaves a lone surrogate that renders as a broken glyph.
  const code_points = Array.from(first_line)
  return code_points.length > STOP_SCREEN_ERROR_EXCERPT_CHARS
    ? `${code_points.slice(0, STOP_SCREEN_ERROR_EXCERPT_CHARS).join("")}…`
    : first_line
}

// The stop banner's message. Rendered via Warning's markdown+trusted mode
// so the all-failed variant can carry the run-page deeplink in-message —
// markdown links open in a new tab by component design, so the wizard tab
// (and the plan behind it) survive the detour. The recovery sentence is
// its own PARAGRAPH (blank markdown line): raw error text can be long
// (full provider messages ride through unfiltered), and the action must
// never drown in the diagnosis.
export function drive_stop_banner(
  stop: DriveStop,
  run_config_name: string | null,
  run_config_model: string | null = null,
): string {
  const config_clause = run_config_name
    ? ` (run config: ${run_config_name})`
    : ""
  // Counts alone, no noun for the unit of work: the same sentence is true
  // whether a run drove conversations or generated single inputs.
  const total = stop.survivors + stop.failed
  if (stop.preflight) {
    // A lane failed its pre-drive test call: nothing ran and nothing was
    // spent. All lanes show the raw error (the SDG precedent — generation
    // failures render getMessage() verbatim), with a parenthetical naming
    // what was tested: the run config (name + model) on the user's lane,
    // the model on the other lanes. Recovery is the banner family's
    // deeplink formula; the model lanes also state the one requirement
    // fact we know (which provider's key the step runs on).
    const f = stop.preflight
    if (f.lane === "run config") {
      const clause = run_config_name
        ? ` (run config: ${run_config_name}${
            run_config_model ? `, ${run_config_model}` : ""
          })`
        : ""
      return `Could not create your eval data. Your run config failed a test call: ${f.message}${clause}.\n\nYou can [test your run config](/run), then start again.`
    }
    const subject =
      f.lane === "synthetic-user driver"
        ? "The model that plays the user"
        : f.lane === "input generator"
          ? "The input generation model"
          : "The judge model"
    const model_clause = f.model ? ` (${f.model})` : ""
    const requirement = f.provider
      ? `Creating your eval data requires your ${f.provider} API key. `
      : ""
    return `Could not create your eval data. ${subject} failed a test call: ${f.message}${model_clause}.\n\n${requirement}You can [check your model providers](/settings/providers), then try again.`
  }
  if (stop.aborted_error) {
    // A config-scoped failure aborted the batch mid-drive: the run config
    // (name + model) IS the diagnosis, and cases judged before the abort
    // remain valid survivors.
    const abort_config = run_config_name
      ? ` (run config: ${run_config_name}${
          run_config_model ? `, ${run_config_model}` : ""
        })`
      : ""
    const recovery =
      stop.survivors > 0
        ? `${stop.survivors} of ${total} completed before the stop. Continue with those, or [test your run config](/run) and run the batch again.`
        : `You can [test your run config](/run), then run the batch again.`
    return `The run was stopped: ${stop.aborted_error}${abort_config}.\n\n${recovery}`
  }
  if (stop.survivors === 0) {
    // Every case failed identically — a capability boundary of the run
    // config, not bad luck. Point at the one place it can be verified.
    return `All ${total} failed: ${
      stop.dominant_error ?? "no error details"
    }${config_clause}.\n\nYou can [test your run config](/run), then run the batch again.`
  }
  // The partial stop. It asks the question the screen exists to answer in the
  // first paragraph, and puts the diagnosis in the second: what was already
  // tried, and a bounded excerpt of the error behind most of the failures.
  // The excerpt ends the sentence with a period unless it already closes
  // itself: its own question or exclamation mark, or an ellipsis, whether the
  // provider's own or the one marking where the text was cut. The two
  // paragraphs are one newline apart:
  // a blank markdown line renders a spacer, which would double the gap the
  // stop screen's own layout already puts between them.
  const excerpt = stop.dominant_error ? error_excerpt(stop.dominant_error) : ""
  const common_clause = excerpt
    ? ` Most common error: ${excerpt}${/[!?…]$/.test(excerpt) ? "" : "."}`
    : ""
  return `${stop.failed} of ${total} failed. Continue with only ${stop.survivors}, or run this batch again?\nEach failure was retried.${common_clause}`
}

// A stop that left usable work behind: some cases failed, the batch was never
// aborted, and nothing failed before the drive started. Only this kind can be
// told in one counted sentence plus a line of diagnosis, so only this kind
// gets the centred screen; the others carry the full provider text and a
// recovery deeplink, and their way out runs through the plan.
export function is_partial_stop(stop: DriveStop): boolean {
  return stop.survivors > 0 && !stop.aborted_error && !stop.preflight
}

// SDG's confirm formula for the destructive tier that carries real work.
// The review-progress clause applies only once the user accepted the
// results (was in review) — on the stop screen no review exists yet.
export function driven_data_confirm(
  action: string,
  survivors: number,
  include_review_progress: boolean,
): string {
  const progress_clause = include_review_progress
    ? " and your review progress"
    : ""
  return `You have ${survivors} completed eval input${
    survivors === 1 ? "" : "s"
  }${progress_clause}. ${action} will discard them. This cannot be undone.`
}

// Regenerating the plan ALWAYS confirms — a plan alone costs minutes to
// make. Three-tier wording: what you lose scales the message — plan only /
// plan + row deletions / driven results. The first two tiers are SDG's
// exact formulas, and take plan_noun — the word for the plan's rows, which
// the eval builder passes as "items" on both arms so the confirm matches the
// plan surface's own header. The driven tier names the ACTION instead: SDG's
// formula puts the subject in front of a verb ("... will discard them"),
// where a plural row noun reads as broken grammar.
export function new_plan_confirm(state: {
  has_driven_results: boolean
  survivors: number
  include_review_progress: boolean
  plan_edited: boolean
  plan_noun?: string
}): string {
  const plan_noun = state.plan_noun ?? "items"
  if (state.has_driven_results) {
    return driven_data_confirm(
      "A new batch plan",
      state.survivors,
      state.include_review_progress,
    )
  }
  return state.plan_edited
    ? `Are you sure you want to discard the current ${plan_noun}, including the ones you removed? This cannot be undone.`
    : `Are you sure you want to discard the current ${plan_noun}? This cannot be undone.`
}

// What the drive is about to spend, shown on the button that spends it. The
// multi-turn arm bills per TURN (every case is a whole conversation), so it
// states the multiplication rather than the case count alone; the single-turn
// arm runs the task once per input and counts those.
export function drive_cost_warning(args: {
  is_multi_turn: boolean
  count: number
  turns_per_case: number
}): string {
  if (!args.is_multi_turn) {
    return `This will run your task on ${args.count} item${
      args.count === 1 ? "" : "s"
    } and may use considerable credits.`
  }
  // Spelled out rather than "N x M": the multiplication told the reader the
  // arithmetic but not what was being multiplied. "Up to", because the turn
  // count is a cap the conversation can finish under, not a quota it fills.
  return `This will run ${args.count} conversation${
    args.count === 1 ? "" : "s"
  } of up to ${args.turns_per_case} turn${
    args.turns_per_case === 1 ? "" : "s"
  } each. Running multi-turn data generation and evaluation can be expensive. Calculate your costs before proceeding.`
}

// ── The preparing-review gate ────────────────────────────────────────────

// A trace holds the gate only while its claims could still change: "built"
// and "error" are both RESOLVED (errored builds keep their in-review
// error+retry card and must not hold the door).
export function is_claims_resolved(state: ClaimsBuildState): boolean {
  return state === "built" || state === "error"
}

// How many of the selected traces are resolved — the gate advances (and the
// "Preparing Review (N/M)" title counts up) when this reaches
// selected_indices.length.
export function resolved_selected_count(
  traces: { claims_state: ClaimsBuildState }[],
  selected_indices: number[],
): number {
  return selected_indices.filter((i) => {
    const t = traces[i]
    return t !== undefined && is_claims_resolved(t.claims_state)
  }).length
}

// The share of the batch that has to survive for continuing with the
// survivors to lead. Below it too much of the approved plan is missing for
// the survivors to be the better offer.
const STOP_CONTINUE_LEADS_AT = 0.9

// Which of the stop screen's two actions is the primary. Continuing with the
// survivors leads only when almost the whole batch made it; below that, and
// whenever nothing survived, re-running the batch is the action worth leading
// with. A stop that never drove anything (preflight) or that was cut short by
// a failing config (aborted_error) leads with re-running whatever survived,
// because the survivors are not the whole story.
export function stop_primary_action(stop: DriveStop): "continue" | "rerun" {
  if (stop.preflight || stop.aborted_error) {
    return "rerun"
  }
  const total = stop.survivors + stop.failed
  if (stop.survivors <= 0 || total <= 0) {
    return "rerun"
  }
  return stop.survivors / total >= STOP_CONTINUE_LEADS_AT ? "continue" : "rerun"
}
