// Source assertions for the builder's wizard surfaces. The builder page is far
// too large to mount, but the strings and the wiring below are contractual: the
// ruled copy, and the rule that the Generation Settings dialog is the drive's
// ONLY entrance. Reading the source is the house precedent for pinning facts a
// render test can't reach (see lib/agent_coverage.test.ts).
import { describe, expect, it } from "vitest"
import * as fs from "fs"
import * as path from "path"

const page_source = fs.readFileSync(
  path.resolve(__dirname, "./+page.svelte"),
  "utf-8",
)

// Collapses runs of whitespace so an assertion survives Prettier rewrapping a
// long attribute across lines.
function normalize(source: string): string {
  return source.replace(/\s+/g, " ")
}

const normalized = normalize(page_source)

function contains(needle: string): boolean {
  return normalized.includes(normalize(needle))
}

// The slice of the page a claim is actually about. Negative assertions run
// against a region rather than the whole 4900-line file: "Advanced Settings"
// appearing on some unrelated future surface is not a regression of the drive
// dialog's title, and a whole-file not.toContain would say it was.
function region(start_anchor: string, end_anchor: string): string {
  const start = page_source.indexOf(start_anchor)
  if (start < 0) {
    throw new Error(`anchor not found in +page.svelte: ${start_anchor}`)
  }
  const end = page_source.indexOf(end_anchor, start + start_anchor.length)
  if (end < 0) {
    throw new Error(
      `end anchor not found after "${start_anchor}": ${end_anchor}`,
    )
  }
  return page_source.slice(start, end + end_anchor.length)
}

// A function body: from its signature to the closing brace at script indent.
function function_body(signature: string): string {
  return region(signature, "\n  }")
}

// How many times a symbol is named on the page — the entrance count for a
// drive function (its own definition plus its legitimate callers).
function mentions(symbol: string): number {
  return page_source.split(symbol).length - 1
}

// Mentions of a symbol as a whole word — so counting TURNS_PER_CASE isn't
// inflated by MIN_TURNS_PER_CASE / MAX_TURNS_PER_CASE.
function whole_word_mentions(symbol: string): number {
  const pattern = new RegExp(`(?<![A-Za-z0-9_])${symbol}(?![A-Za-z0-9_])`, "g")
  return (page_source.match(pattern) ?? []).length
}

const describe_step = region(
  '{:else if current_step === "describe"}',
  '{:else if current_step === "clarify"}',
)
const plan_surface = region("<KilnProBatchPlan", "</KilnProBatchPlan>")
const new_plan_dialog = region("bind:this={new_plan_dialog}", "</Dialog>")
const drive_settings_dialog = region(
  "bind:this={drive_settings_dialog}",
  "</Dialog>",
)

// The Generation Settings dialog's single-turn input-generator lane.
function input_gen_lane(): string {
  const start = drive_settings_dialog.indexOf("<RunConfigComponent")
  if (start < 0) throw new Error("input generator lane not found")
  return normalize(
    drive_settings_dialog.slice(
      start,
      drive_settings_dialog.indexOf("/>", start),
    ),
  )
}

describe("describe step action row", () => {
  it("offers one forward action, with no Cancel beside it", () => {
    // Leaving the wizard is the browser's Back. A Cancel button beside the
    // primary made the row read as a two-way choice.
    expect(describe_step).not.toContain(">Cancel<")
    expect(describe_step).not.toContain("Cancel</button")
  })

  it("names the forward action Continue, like every other step", () => {
    expect(normalize(describe_step)).toContain(
      "on:click={continue_from_describe} disabled={!description.trim()} > Continue",
    )
  })

  it("demotes the manual path to the secondary-action row", () => {
    expect(normalize(describe_step)).toContain(
      'class="link underline text-sm text-gray-500" on:click={create_manually} > Create Manually',
    )
  })
})

describe("plan surface copy", () => {
  it("names the plan surface for the eval dataset it proposes", () => {
    // The header is the one label this surface overrides: what it lists is a
    // proposed eval dataset, not the synthetic data flow's batch.
    expect(normalize(plan_surface)).toContain(
      'header_label="Eval Dataset Proposal"',
    )
  })

  it("renders one subheader, the same on both arms", () => {
    // What the next step does to each row belongs on that step, not in the
    // plan's sub-line: one sentence reads the same whether the run drives
    // conversations or generates single inputs.
    expect(normalize(plan_surface)).toContain(
      'subheader="Here\'s a plan for your eval dataset. Refine the plan if the coverage looks off."',
    )
    expect(plan_surface).not.toContain("is_multi_turn")
  })

  it("labels the primary button with the artifact noun and the count", () => {
    expect(
      contains(
        "generate_button_label={`Generate Dataset (${batch_plan.prompts.length} items)`}",
      ),
    ).toBe(true)
  })

  it("names the plan's rows items and drops the /generate sub-line", () => {
    // One noun prop, so the header reads "All Items (n)" and a screen reader
    // hears "items" too. The /generate sentence is about dataset samples, which
    // is not what this surface's rows become.
    expect(normalize(plan_surface)).toContain('items_label="Items"')
    expect(normalize(plan_surface)).toContain(
      'expanded_description="Each row will be used to seed one item of your eval dataset."',
    )
  })

  it("puts the data guide note on the sub-line, only when a guide was used", () => {
    // The note is a clause on the header's sub-line rather than a row of its
    // own, so the plan surface opens with one sentence. It is a claim about
    // how the plan was drafted, so it renders only when that is true.
    const normalized_surface = normalize(plan_surface)
    expect(normalized_surface).toContain(
      '<svelte:fragment slot="under_subheader"> {#if plan_drafted_with_data_guide}',
    )
    // The clause, normalized so Prettier's wrapping is not what is pinned. It
    // has to open with an explicit space, because Svelte drops whitespace at
    // the start of slot content and the sub-line adds no separator of its own.
    const note = normalize(region('<span id="data_guide_plan_note"', "</span"))
    expect(note).toContain('{" "}Planned using your')
    expect(note).toContain("data guide</button")
    expect(note.endsWith(".</span")).toBe(true)
  })

  it("names the rows' column for what this surface's rows hold", () => {
    // The shared table's default header is "Prompt", which is right for
    // /generate's generation prompts. These rows are per-item guidance, so
    // this surface must override it; losing the override silently mislabels
    // the column.
    expect(normalize(plan_surface)).toContain('column_label="Item Guidance"')
  })
})

describe("plan drafting screen", () => {
  const planning_copy = region(
    "$: generate_animation_title =",
    "$: generate_animation_warning =",
  )

  it("names what is being planned, with no arm-specific noun", () => {
    expect(planning_copy).toContain(`? "Planning Eval Dataset"`)
    expect(planning_copy).not.toContain("Drafting Scenarios")
    expect(planning_copy).not.toContain("Planning Test Inputs")
  })

  it("describes the batch with each arm's artifact noun and no count", () => {
    // The count is deliberately absent: the planner can return fewer lines
    // than asked for, so quoting the request here would promise a size the
    // plan screen then contradicts.
    expect(
      contains(
        '"Kiln is planning a diverse batch of conversations, tailored to your task and guidance."',
      ),
    ).toBe(true)
    expect(
      contains(
        '"Kiln is planning a diverse batch of eval data, tailored to your task and guidance."',
      ),
    ).toBe(true)
    expect(planning_copy).not.toContain("${eval_input_count}")
  })
})

// Drops both comment forms — Svelte markup comments and script line comments —
// so the vocabulary scan below reads only what ships. The `://` guard keeps a
// URL inside an attribute value from being mistaken for a comment, which would
// truncate that attribute mid-string.
function strip_comments(source: string): string {
  return source.replace(/<!--[\s\S]*?-->/g, "").replace(/(?<!:)\/\/.*$/gm, "")
}

// Every quoted span in the source. Run against whitespace-normalized text, so a
// string Prettier wrapped across lines is still read as one string.
const QUOTED_STRING = /"[^"]*"|'[^']*'|`[^`]*`/g

describe("ruled vocabulary", () => {
  it("leaves no user-facing scenario wording on the page", () => {
    // Rows are "items" and the artifact is "traces" / "eval data". Comments
    // may still say scenario (it is the planner's own term); a shipped string
    // may not.
    const shipped = normalize(strip_comments(page_source))
    const offenders = (shipped.match(QUOTED_STRING) ?? []).filter((s) =>
      /scenario/i.test(s),
    )
    expect(offenders).toEqual([])
  })

  it("uses one word for the plan's rows on both arms", () => {
    expect(contains('const plan_noun = "items"')).toBe(true)
  })

  it("sends both arms back to the same plan screen", () => {
    expect(page_source).not.toContain("Back to Scenarios")
    // One label for both arms, matching the plan surface it leads to.
    expect(contains("Plan Batch")).toBe(true)
    expect(page_source).not.toContain("Plan Traces")
    expect(page_source).not.toContain("Plan Test Inputs")
  })
})

describe("single entrance to the drive", () => {
  it("routes the plan's primary button into the settings dialog", () => {
    expect(normalize(plan_surface)).toContain(
      "on_generate_inputs={open_drive_settings}",
    )
  })

  it("has no run-immediately path left", () => {
    // start_drive_with_defaults was the second entrance; it is gone entirely,
    // not merely unreferenced.
    expect(mentions("start_drive_with_defaults")).toBe(0)
  })

  it("drops the advanced slot's model-choice link", () => {
    expect(normalize(plan_surface)).not.toContain('slot="advanced"')
    expect(plan_surface).not.toContain("choose which models to use")
  })

  it("starts a drive only from the settings dialog's submit", () => {
    const submit = function_body("async function submit_drive_settings() {")
    expect(submit).toContain("on_drive_multi_turn()")
    expect(submit).toContain("on_drive_single_turn()")
    // Each arm's drive function is named exactly twice on the page: its own
    // definition and that submit's call. A third mention is a second
    // entrance — a run the user was never shown the lanes or the cost of.
    expect(mentions("on_drive_multi_turn")).toBe(2)
    expect(mentions("on_drive_single_turn")).toBe(2)
  })

  it("opens the dialog from the error screen's Retry", () => {
    // Retrying a failed drive is still a drive: it re-states the lanes and the
    // cost rather than re-spending on the last attempt's settings silently.
    const retry = function_body("function on_continue_from_generate_step() {")
    expect(retry).toContain("open_drive_settings()")
    expect(retry).not.toContain("on_drive_")
  })

  it("opens the dialog for the no-results re-drive", () => {
    // Advancing to review with nothing to show (a Back aborted the pipeline)
    // re-drives — through the same entrance as the first attempt.
    const advance = function_body("function continue_to_review() {")
    expect(advance).toContain("open_drive_settings()")
    expect(advance).not.toContain("on_drive_")
  })

  it("derives every turn count from the one alias", () => {
    // The committed value reaches the drive only through the clamped alias, so
    // what runs, what is stamped, and what the progress bar counts cannot
    // drift apart or fall outside the range the route accepts.
    expect(
      contains(
        "$: drive_turns_per_case = clamp_turns_per_case(turns_per_case)",
      ),
    ).toBe(true)
    // TURNS_PER_CASE is read as a DEFAULT only: its definition, the knob's
    // seed, and the restore fallback. A fourth reader would be a drive value
    // that ignores the user's choice.
    expect(whole_word_mentions("TURNS_PER_CASE")).toBe(3)
    expect(contains("const TURNS_PER_CASE = 5")).toBe(true)
    expect(contains("let turns_per_case = TURNS_PER_CASE")).toBe(true)
  })
})

describe("the turns knob", () => {
  // The multi-turn half of the dialog's arm branch: the lane and the turns row
  // that only exist when there are conversations to run.
  const multi_turn_branch = (() => {
    const start = drive_settings_dialog.indexOf("{#if is_multi_turn}")
    const end = drive_settings_dialog.indexOf("{:else}", start)
    if (start < 0 || end < 0) {
      throw new Error("the drive dialog's arm branch was not found")
    }
    return drive_settings_dialog.slice(start, end)
  })()
  // The single-turn half of the same branch, so the negatives below are about
  // THIS dialog's other arm rather than the whole page.
  const single_turn_branch = (() => {
    const start = drive_settings_dialog.indexOf("{:else}")
    const end = drive_settings_dialog.indexOf("{/if}", start)
    if (start < 0 || end < 0) {
      throw new Error("the drive dialog's single-turn arm was not found")
    }
    return drive_settings_dialog.slice(start, end)
  })()
  // The multi-turn save block, from its own guard to the request it sends.
  const multi_turn_save = region(
    "if (multi_turn_batch_tag === null || driven_cases.length === 0) {",
    "signal: new_copilot_abort_signal(),",
  )

  it("renders the stepper row only on the multi-turn arm", () => {
    // A single-turn run is one shot per input — a length control there would
    // be a knob that changes nothing.
    expect(multi_turn_branch).toContain("Max turns per conversation")
    expect(single_turn_branch).not.toContain("Max turns per conversation")
    expect(single_turn_branch).not.toContain("<IncrementUi")
  })

  it("labels the row and pins its explanation as a tooltip", () => {
    expect(normalize(multi_turn_branch)).toContain(
      "<span>Max turns per conversation</span>",
    )
    expect(normalize(multi_turn_branch)).toContain("font-medium text-sm")
    expect(
      contains(
        'tooltip_text="One turn is one exchange: the user sends a message and your agent replies. A conversation stops early once the simulated user has what it came for, so this is a ceiling rather than a target. A higher ceiling tests deeper behavior and costs more."',
      ),
    ).toBe(true)
  })

  it("binds the stepper to the STAGED knob, bounded by the route's own range", () => {
    // Staged like the dialog's model lanes: the stepper never writes the
    // committed value directly, so leaving without submitting discards it.
    const stepper = normalize(
      multi_turn_branch.slice(multi_turn_branch.indexOf("<IncrementUi")),
    )
    expect(stepper).toContain("bind:value={staged_turns_per_case}")
    expect(stepper).not.toContain("bind:value={turns_per_case}")
    expect(stepper).toContain("min={MIN_TURNS_PER_CASE}")
    expect(stepper).toContain("max={MAX_TURNS_PER_CASE}")
  })

  it("discards a nudge when the dialog closes without submitting", () => {
    // Esc / X / backdrop leave the committed value alone; the next open
    // reseeds the staged copy from it. Without this reseed a cancelled nudge
    // would survive into the next drive — and silently disqualify a top-off,
    // which refuses to mix conversation lengths.
    const open = function_body("async function open_drive_settings() {")
    expect(open).toContain("staged_turns_per_case = turns_per_case")
  })

  it("commits the staged length on submit, clamped, before driving", () => {
    const submit = function_body("async function submit_drive_settings() {")
    const commit = submit.indexOf(
      "turns_per_case = clamp_turns_per_case(staged_turns_per_case)",
    )
    expect(commit).toBeGreaterThan(-1)
    expect(commit).toBeLessThan(submit.indexOf("on_drive_multi_turn()"))
    // The clamped alias is a reactive derivation, so it still holds the
    // pre-submit length during the synchronous handler — the drive has to
    // start after the commit has landed, or it runs the old number.
    const settle = submit.indexOf("await tick()")
    expect(settle).toBeGreaterThan(commit)
    expect(settle).toBeLessThan(submit.indexOf("on_drive_multi_turn()"))
  })

  it("puts the cost warning after the row, and quotes the staged length", () => {
    // Same form, and the warning reads the clamp of the STAGED value — moving
    // the stepper re-quotes the run before the user spends on it, even though
    // nothing is committed until submit.
    const dialog = normalize(drive_settings_dialog)
    expect(dialog.indexOf("Max turns per conversation")).toBeLessThan(
      dialog.indexOf("warning_message={drive_cost_message}"),
    )
    expect(contains("turns_per_case: staged_drive_turns_per_case,")).toBe(true)
    expect(
      contains(
        "$: staged_drive_turns_per_case = clamp_turns_per_case(staged_turns_per_case)",
      ),
    ).toBe(true)
  })

  it("sends one length per drive, read once at the top", () => {
    // Read once into a local, so a stepper moved mid-drive can't change what
    // the request, the top-off decision, and the stamp are talking about.
    const drive = function_body("async function on_drive_multi_turn() {")
    expect(drive).toContain("const chosen_turns = drive_turns_per_case")
    expect(drive).toContain("turns: chosen_turns,")
    expect(drive).not.toContain("turns: drive_turns_per_case,")
  })

  it("captures the driven length beside the synthetic-user model", () => {
    // Same capture/rollback pattern as driven_su_driver: stamped at batch
    // commit, restored when nothing drove so the previous batch's stamp
    // survives a failed re-drive.
    const drive = function_body("async function on_drive_multi_turn() {")
    expect(drive).toContain(
      "const previous_driven_turns_per_case = driven_turns_per_case",
    )
    expect(drive).toContain("driven_turns_per_case = chosen_turns")
    expect(drive).toContain(
      "driven_turns_per_case = previous_driven_turns_per_case",
    )
  })

  it("refuses a top-off that would mix conversation lengths", () => {
    const drive = function_body("async function on_drive_multi_turn() {")
    const guard = drive.slice(drive.indexOf("drive_lanes_unchanged({"))
    expect(normalize(guard)).toContain("turns: chosen_turns,")
    expect(normalize(guard)).toContain("batch_turns: driven_turns_per_case,")
  })

  it("stamps the driven length on save, never the live knob", () => {
    // The stepper can move after the drive; the saved eval must describe the
    // conversations that exist on disk. Save fails loud rather than guessing a
    // length for chains it has no record of.
    expect(multi_turn_save).toContain(
      "const saved_turns_per_case = driven_turns_per_case",
    )
    expect(multi_turn_save).toContain(
      "No conversation length was recorded for the driven conversations. Go back to Step 4.",
    )
    expect(normalize(multi_turn_save)).toContain("turns: saved_turns_per_case,")
    expect(multi_turn_save).not.toContain("drive_turns_per_case")
  })

  it("counts progress against the driven length", () => {
    // A stepper moved while the bar is on screen must not restate the
    // denominator under a batch already running at another length.
    expect(
      contains(
        "$: multi_turn_total_turns = pipeline_total_cases * (driven_turns_per_case ?? drive_turns_per_case)",
      ),
    ).toBe(true)
  })

  it("persists the chosen length and restores it through the clamp", () => {
    // The mirror is what puts the choice on disk; without it a reload silently
    // puts the drive back on the default. A stored value can predate today's
    // range, and a draft written before the knob existed carries no value.
    const mirror = region("$: current_draft = draft_ready", "    : null")
    expect(normalize(mirror)).toContain("turns_per_case,")
    const restore = normalize(function_body("async function restore_draft() {"))
    expect(restore).toContain(
      "turns_per_case = restore_turns_per_case( saved.turns_per_case, TURNS_PER_CASE, )",
    )
  })
})

describe("Generation Settings dialog", () => {
  it("is titled Generation Settings", () => {
    expect(
      contains(
        '<Dialog bind:this={drive_settings_dialog} title="Generation Settings">',
      ),
    ).toBe(true)
  })

  it("no longer carries the old Advanced Settings title", () => {
    expect(drive_settings_dialog).not.toContain("Advanced Settings")
  })

  it("submits with the artifact noun and the planned count", () => {
    expect(
      contains("submit_label={`Generate Dataset (${planned_total} items)`}"),
    ).toBe(true)
  })

  it("pins each lane's explanation as a tooltip, not a visible description", () => {
    expect(
      contains(
        'info_description="Stands in for a real user in each test conversation. Your agent replies to it."',
      ),
    ).toBe(true)
    expect(contains('label="Model that writes the user\'s messages"')).toBe(
      true,
    )
    expect(
      contains(
        'model_info_description="Writes the input for each dataset item. Your run config then produces the output that the judge scores."',
      ),
    ).toBe(true)
    // The lane writes the input half of each dataset item; the output comes
    // from the run config the eval is about, so the label says which of the
    // two this model is.
    expect(contains('model_label="Input Generation Model"')).toBe(true)
  })

  it("gives the input generator the same control synthetic data generation uses", () => {
    // Tools and skills change what the eval data looks like, so this lane is
    // the full run config picker rather than a bare model dropdown.
    const lane = input_gen_lane()
    expect(lane).toContain("hide_prompt_selector={true}")
    expect(lane).toContain("show_tools_selector_in_advanced={true}")
    expect(lane).toContain("show_name_field={false}")
  })

  it("gives the lane no task, so its tools stay out of the app-wide store", () => {
    // The tool and skill pickers seed from, and mirror every change into,
    // tools_store keyed by task id — a parent's write counts the same as a
    // user's click. With no task they skip that store entirely and are still
    // fully populated from the project. Passing a task here would let this
    // dialog's tools overwrite the ones chosen on Run and Synthetic Data, and
    // let those overwrite a restored draft's.
    const lane = input_gen_lane()
    expect(lane).not.toContain("current_task")
    expect(lane).not.toContain("bind:tools")
    expect(lane).not.toContain("bind:skills")
  })

  it("seeds the lane from the config it last committed", () => {
    // The component stays mounted between opens, so both a restored draft and
    // a visit abandoned by Cancel have to be seeded back to the committed
    // config rather than left showing stale edits.
    expect(input_gen_lane()).toContain(
      "initial_run_config_properties={input_gen_run_config}",
    )
    const open = normalize(
      function_body("async function open_drive_settings() {"),
    )
    expect(open).toContain(
      "input_gen_config_component?.apply_run_config_properties( input_gen_run_config, )",
    )
    // Nothing committed yet still has to reseed: the lane goes back to the
    // defaults a first open shows, or an abandoned visit's edits survive it.
    expect(open).toContain("input_gen_config_component?.reset_run_options()")
  })

  it("carries the committed config on the draft so a reload can still run", () => {
    // Without it a restored session would be bounced back into the dialog
    // before it could drive.
    const mirror = region("$: current_draft = draft_ready", "    : null")
    expect(normalize(mirror)).toContain("input_gen_run_config,")
    expect(
      normalize(function_body("async function restore_draft() {")),
    ).toContain("input_gen_run_config = saved.input_gen_run_config ?? null")
  })

  it("sends the committed run config to the minting route, not a rebuilt one", () => {
    // Tools and skills only reach the generator if the whole config the user
    // configured is what gets sent.
    const mint = normalize(
      function_body("async function mint_inputs_from_plan("),
    )
    expect(mint).toContain("run_config_properties: input_gen_config,")
    expect(mint).not.toContain('prompt_id: "simple_prompt_builder"')
  })

  it("keys the minted-inputs cache on the config it sends", () => {
    // Sampling and tools are both editable in this lane and both change what
    // is written, so the key is the whole config, derived from the very
    // object the request carries.
    const drive = normalize(
      function_body("async function on_drive_single_turn() {"),
    )
    expect(drive).toContain(
      "const input_gen_config_key = run_config_cache_key(chosen_input_config)",
    )
    expect(drive).toContain("run_config_json: input_gen_config_key,")
  })

  it("recommends the user-model lane's own tier, and filters on nothing", () => {
    // The simulated user writes one plain-text message a turn: no schema, no
    // tools. Recommending it from the data-gen flag pointed at the frontier
    // models, and requiring data-gen support pushed usable chat models into
    // "Not Recommended". It now reads its own registry flag and excludes
    // nothing.
    const su_lane = drive_settings_dialog.slice(
      drive_settings_dialog.indexOf("<AvailableModelsDropdown"),
    )
    const lane = normalize(su_lane.slice(0, su_lane.indexOf("/>")))
    expect(lane).toContain('suggested_mode: "synthetic_user"')
    expect(lane).not.toContain("requires_")
    // The lane's pre-selected default comes from the same set that badges it,
    // so the dialog does not open on a model its own advisory warns about.
    expect(
      normalize(function_body("async function fill_null_lanes() {")),
    ).toContain('build_suggested_models(models, "synthetic_user")[0]')
  })

  it("names the judge lane once and explains it per arm", () => {
    expect(contains('label="Judge Model"')).toBe(true)
    expect(
      contains(`"Checks each conversation against your eval's criteria."`),
    ).toBe(true)
    expect(contains(`"Checks each result against your eval's criteria."`)).toBe(
      true,
    )
  })

  it("puts the cost warning immediately before the submit", () => {
    // Last child of the FormContainer = directly above the submit row, which
    // is the one button that spends the credits.
    const dialog = normalize(drive_settings_dialog)
    const warning = dialog.indexOf("warning_message={drive_cost_message}")
    const close = dialog.indexOf("</FormContainer>")
    expect(warning).toBeGreaterThan(-1)
    expect(warning).toBeLessThan(close)
    // Nothing else renders between the warning and the form's close.
    expect(dialog.slice(warning, close)).not.toContain("<AvailableModels")
  })

  it("seeds every lane before the dialog is shown", () => {
    // Order is the whole fix: showing first let the user pick a model while
    // the defaults were still resolving, and the reseed then overwrote it.
    // Pinned by index, because only the sequence is wrong in the old version.
    const open = function_body("async function open_drive_settings() {")
    const awaited = open.indexOf("await prepopulate_lanes()")
    const reseed = open.indexOf("judge_model_combined = judge_model")
    const show = open.indexOf("drive_settings_dialog?.show()")
    expect(awaited).toBeGreaterThan(-1)
    expect(reseed).toBeGreaterThan(awaited)
    expect(show).toBeGreaterThan(reseed)
    // The pass is memoized as a promise, so a second caller awaits the one in
    // flight instead of returning early while the lanes are still null.
    expect(
      contains("let lanes_prepopulated: Promise<void> | null = null"),
    ).toBe(true)
    // A failed pass must still open the dialog on whatever is committed,
    // rather than swallow the click.
    expect(open).toContain('console.warn("Could not resolve the default')
  })

  it("clears a stale validation error when a lane changes", () => {
    // The error names a condition ("… to continue"); once the user fills the
    // lanes the condition is gone, so the sentence must go with it instead of
    // standing over two filled dropdowns. All four editable values are
    // dependencies, and the error itself is not read — raising it in submit
    // must not immediately re-run the clear.
    const statement = region(
      "$: drive_settings_error = cleared_on_lane_change(",
      ")",
    )
    const dependencies = normalize(
      statement.slice(statement.indexOf("(") + 1, statement.lastIndexOf(")")),
    )
    expect(dependencies).toContain("su_model_combined")
    expect(dependencies).toContain("input_gen_model_combined")
    expect(dependencies).toContain("judge_model_combined")
    expect(dependencies).toContain("staged_turns_per_case")
    expect(dependencies).not.toContain("drive_settings_error")
    expect(
      normalize(function_body("function cleared_on_lane_change(")),
    ).toContain("return null")
  })
})

describe("Refine Plan dialog", () => {
  it("replaces the native confirm on the regenerate button", () => {
    expect(normalize(plan_surface)).toContain(
      "on_regenerate={open_new_plan_dialog}",
    )
    expect(mentions("on_new_plan_with_confirm")).toBe(0)
  })

  it("names the action it performs, never the drive's verb", () => {
    // This dialog only re-plans — it generates nothing — so neither label may
    // borrow the drive's verb. The title says which plan it replaces; the
    // submit still echoes the shared regenerate button that opens it.
    expect(normalize(new_plan_dialog)).toContain('title="New Dataset Plan"')
    expect(normalize(new_plan_dialog)).toContain('submit_label="Refine Plan"')
    expect(new_plan_dialog).not.toContain("Generate Batch")
    expect(new_plan_dialog).not.toContain("Generate Trace Batch")
    expect(new_plan_dialog).not.toContain("Generate Dataset")
  })

  it("wraps the shared batch form with the ruled count label", () => {
    expect(normalize(new_plan_dialog)).toContain('count_label="Item Count"')
    expect(normalize(new_plan_dialog)).toContain('guidance_id="plan_steer"')
  })

  it("caps the stepper at the server's batch cap", () => {
    // The stepper must stop where the routes reject, so the user can't compose
    // a request that can only 422.
    expect(contains("const NUM_CASES_MAX = 200")).toBe(true)
    expect(normalize(new_plan_dialog)).toContain("count_max={NUM_CASES_MAX}")
  })

  it("binds the guidance box to the steer rather than prefilling a template", () => {
    expect(normalize(new_plan_dialog)).toContain("bind:guidance={plan_steer}")
    expect(new_plan_dialog).not.toContain("guidance_template=")
  })

  it("marks the guidance box optional so a blank steer submits", () => {
    // The box starts empty on purpose, so the default "just re-plan" click
    // sends nothing. Without this the shared field's required-validator
    // rejects the empty box and the regenerate path can never run.
    expect(normalize(new_plan_dialog)).toContain("guidance_optional={true}")
  })

  it("leaves the guidance example to the shared field's own description", () => {
    // The shared Guidance field already carries an example; a second one
    // stacked under it reads as noise.
    expect(new_plan_dialog).not.toContain("guidance_placeholder")
  })

  it("discards a typed steer when the dialog closes without submitting", () => {
    // The box is a DRAFT. Only a submit copies it into the steer the request
    // sends, so a steer the user typed and then abandoned cannot ride a later
    // plan request.
    expect(normalize(new_plan_dialog)).toContain(
      "on:close={discard_plan_steer_draft}",
    )
    expect(function_body("function discard_plan_steer_draft() {")).toContain(
      "plan_steer = pending_plan_steer",
    )
    expect(function_body("function submit_new_plan() {")).toContain(
      "pending_plan_steer = plan_steer",
    )
  })

  it("sends the committed steer, never the dialog's draft", () => {
    const request = region("compose_plan_guidance(", "count: eval_input_count,")
    expect(request).toMatch(/\bpending_plan_steer\b/)
    expect(request).not.toMatch(/(?<!pending_)\bplan_steer\b/)
  })

  it("keeps the steer through a failed attempt and drops it once a plan lands", () => {
    // Retention lives in on_plan_batch's early returns: every failure path
    // returns before the clear below, so Retry re-sends what was asked for.
    const plan = function_body("async function on_plan_batch() {")
    expect(plan).toContain('pending_plan_steer = ""')
    expect(
      plan.indexOf("batch_plan = { prompts, summary: data.summary }"),
    ).toBeLessThan(plan.indexOf('pending_plan_steer = ""'))
  })

  it("reseeds the count from the last request when no plan is on screen", () => {
    // After a failed regenerate there is no plan to read the size from;
    // eval_input_count still holds what that attempt asked for, so the dialog
    // and Retry agree instead of the stepper snapping back to the default.
    const open = function_body("function open_new_plan_dialog() {")
    expect(open).toContain(
      "if (batch_plan) eval_input_count = batch_plan.prompts.length",
    )
    expect(open).not.toContain("NUM_CASES")
  })
})

describe("Data Guide skip and Back", () => {
  // Back out of an unplanned Step 4 must re-offer, as it does on the
  // synthetic data page; the reset lives in the history handler.
  const handler = region(
    "function sync_step_from_history",
    "current_step = step",
  )

  it("clears the skip when Back leaves Step 4 without a plan", () => {
    expect(normalize(handler)).toContain(
      normalize(
        'if (current_step === "generate" && batch_plan === null) { data_guide_skipped = false }',
      ),
    )
  })
})

describe("Reset", () => {
  const reset = region("async function reset_draft_with_confirm", "\n  }\n")

  it("starts over on the Create Eval page, not by reloading a URL that may carry a description", () => {
    expect(normalize(reset)).toContain(
      normalize(
        "window.location.href = `/specs/${project_id}/${task_id}/select_template`",
      ),
    )
    expect(reset).not.toContain("window.location.reload()")
  })
})

describe("the run config the eval is written against", () => {
  // The signature's own return type closes at script indent, so the shared
  // function_body helper would stop there; take the whole function instead.
  const resolve = region(
    "async function resolve_drive_run_config(",
    "// ── Step 4 — both arms are plan-first over one pipeline shape.",
  )

  it("drives the config the entry page chose over the task default", () => {
    // The questions, the judge and the generated data were all authored
    // against that config, so driving the task default instead would produce
    // eval data for a different agent than the one the eval describes.
    const body = normalize(resolve)
    expect(body).toContain(
      "const chosen_config = chosen_by_user ?? default_match ?? run_configs[0]",
    )
    // The no-default notice names a config the user did not pick. An explicit
    // choice is not a fallback, so it must not raise it.
    expect(body).toContain(
      "fallback_run_config_name = chosen_by_user || default_match ? null : chosen_config.name",
    )
  })

  it("stops the drive when the chosen config is no longer on the task", () => {
    // Driving the task default instead would generate eval data for an agent
    // this eval does not describe, and say nothing about it.
    const body = normalize(resolve)
    expect(body).toContain("if (target_run_config_id && !chosen_by_user) {")
    expect(body).toContain("return null")
  })

  it("keeps the choice across a reload", () => {
    expect(
      normalize(region("$: current_draft = draft_ready", "    : null")),
    ).toContain("target_run_config_id,")
    expect(
      normalize(function_body("async function restore_draft() {")),
    ).toContain("target_run_config_id = saved.target_run_config_id ?? null")
  })

  it("tells the copilot calls that read a config which one to read", () => {
    // Server-side these calls read the config's tools and skills; without the
    // id they read the task default and describe the wrong agent. Only the
    // calls that read capabilities are wired — the save reads none, so
    // sending it there would be a field the server accepts and discards.
    expect(mentions("run_config_id: target_run_config_id")).toBe(2)
    expect(
      normalize(
        region("async function on_save() {", "function create_manually()"),
      ),
    ).not.toContain("run_config_id")
  })
})

// ── The review-preparation and drive progress screens ──────────────────────
//
// Every waiting screen on this step reads the same way: a static title and
// description, and one ProgressCount under the bar carrying every live number.
// A count inside a sentence re-lays that sentence out on every tick; under the
// bar it changes in place. These pin where each count lives, because the
// difference is invisible to a render test that only reads text.
describe("progress screens carry their counts under the bar, never in a string", () => {
  it("leaves the claims gate's title static on both the first round and a later one", () => {
    // Once for the first-round gate, once for the calibration round's.
    expect(normalized.split('title="Preparing Review"').length - 1).toBe(2)
    expect(normalized).toContain(
      'description="Finding the examples where your judgment is most useful."',
    )
    // The count it used to carry now sits under that screen's bar.
    expect(normalized.split("noun={`${judged_noun}s ready`}").length - 1).toBe(
      2,
    )
  })

  it("leaves the drive descriptions static, with the counts under the bar", () => {
    expect(normalized).toContain(
      'description="Simulating conversations with your agent and judging each one."',
    )
    expect(normalized).toContain(
      'description="Running your task on each item and judging the result."',
    )
    expect(normalized).toContain('noun="turns complete"')
    expect(normalized).toContain('noun="judged"')
    // No screen hand-rolls a bar any more: they all go through the component,
    // so the readouts cannot drift apart.
    expect(normalized).not.toContain('class="progress w-56 progress-success"')
  })

  it("routes every waiting screen's bar through the one readout component", () => {
    // Minting, both drive arms, both claims gates, and the re-check.
    expect(normalized.split("<ProgressCount").length - 1).toBe(6)
  })

  it("counts the cases dropped from the review on both claims gates", () => {
    // A case the builder wrote without a verdict claim is excluded from the
    // review; a prompt that starts dropping verdicts has to be visible.
    expect(normalized.split("num_no_verdict:").length - 1).toBe(2)
  })
})

// ── The eval description, opened from the page header ──────────────────────
//
// The review step has no control of its own for the eval's text: the header's
// sub-line is the way in, so the reviewer looks up rather than into the work.
// The line and the action it fires are two props that must agree, and the
// dialog they open lives in the review component.
describe("the eval description opens from the page header", () => {
  it("shows the line only on review, and only with text to show", () => {
    expect(
      contains(
        'review_can_show_spec = current_step === "review" && current_spec_text.trim().length > 0',
      ),
    ).toBe(true)
    expect(contains('review_can_show_spec ? "Eval Description" : ""')).toBe(
      true,
    )
  })

  it("fires the review's own dialog, under the same condition as the line", () => {
    // Same flag on both, so the header can never render a line that opens
    // nothing — nor an action with no line to fire it.
    expect(
      contains(
        "sub_subtitle_action={review_can_show_spec ? () => review_component?.show_spec_dialog() : undefined}",
      ),
    ).toBe(true)
    expect(contains("bind:this={review_component}")).toBe(true)
  })
})

// ── The empty review subset ────────────────────────────────────────────────
describe("a review with nothing to grade is not a dead end", () => {
  it("names both causes and offers the save on a calibration round", () => {
    expect(normalized).toContain(
      "None of these ${judged_noun}s could be reviewed. Analyzing them either failed or produced no verdict to check.",
    )
    // First round: create the data again. Later rounds: that would throw away
    // grades the reviewer already gave, so the opt-out is offered instead.
    expect(normalized).toContain(
      'calibration_rounds_completed > 0 ? "" : " Create your eval data again."',
    )
    const branch = region(
      "{:else if reviewable_trace_indices.length === 0}",
      "{:else}",
    )
    expect(normalize(branch)).toContain("Save Without Improving")
    expect(normalize(branch)).toContain("on:click={save_without_refining}")
  })
})

// ── The forward action on the last case ───────────────────────────────────
//
// The reviewer's feedback can either improve the judge or be kept as-is. The
// wizard asks rather than deciding, and the dialog's secondary is the only way
// out of the refine loop, so these strings and both wirings are contractual.
describe("the improve-judge dialog", () => {
  // One action button's own object literal, from its label to the brace that
  // closes it. Asserting over the whole dialog cannot tell the two buttons
  // apart: swap their bodies and every string is still somewhere in the
  // region. This is what makes a swap fail.
  function action_button(label: string): string {
    const dialog = region('title="Improve Judge with Feedback?"', "</Dialog>")
    const at = dialog.indexOf(`label: "${label}"`)
    if (at < 0) throw new Error(`no action button labelled ${label}`)
    const end = dialog.indexOf("\n    },", at)
    if (end < 0) throw new Error(`unterminated action button ${label}`)
    return normalize(dialog.slice(at, end))
  }

  it("asks before refining, in the words the reviewer was promised", () => {
    const d = normalize(
      region('title="Improve Judge with Feedback?"', "</Dialog>"),
    )
    expect(d).toContain(
      "You disagreed with the judge and gave feedback, which we can use to improve your Judge.",
    )
    expect(d).toContain('label: "Improve Judge"')
    expect(d).toContain('label: "Save Without Improving"')
  })

  it("wires each button to the path it names, and only that path", () => {
    const improve = action_button("Improve Judge")
    expect(improve).toContain("isPrimary: true")
    expect(improve).toContain("run_calibration_round()")
    expect(improve).not.toContain("save_without_refining()")

    const save = action_button("Save Without Improving")
    expect(save).toContain("save_without_refining()")
    expect(save).not.toContain("run_calibration_round()")
    expect(save).not.toContain("isPrimary")
  })

  it("is what the keyboard shortcut reaches too", () => {
    // The shortcut fires the review's forward action rather than the save, so
    // a round with feedback is asked about however the reviewer triggers it.
    const keys = normalize(
      region("function handle_global_keydown(", "function step_name_for("),
    )
    expect(keys).toContain("save_gate_met && review_on_last_trace")
    expect(keys).toContain("on_advance_to_save()")
    expect(keys).not.toContain("on_save()")
  })

  it("opens only where the review asks to go forward with feedback", () => {
    // The dialog replaces the automatic refine: the decision point is the
    // same one, so it is opened from the same branch that used to calibrate.
    const advance = normalize(
      region("function on_advance_to_save()", 'goto_step("save")'),
    )
    expect(advance).toContain('decision.action === "calibrate"')
    expect(advance).toContain("improve_judge_dialog?.show()")
  })

  it("keeps no second exit beside the review's own action", () => {
    // The dialog's secondary is the only way past a review that has feedback;
    // the quiet link that used to sit under the review is gone. The empty-
    // subset screen keeps its own exit, because there is no review to go
    // forward from there at all.
    const review = normalize(
      region("<ClaimEvidenceReview", '{:else if current_step === "save"}'),
    )
    expect(review).not.toContain("Save Without Improving")
    expect(normalized).not.toContain("Save Without Refining Further")
  })
})

// ── The save's success screen ─────────────────────────────────────────────
describe("the eval-created screen", () => {
  it("finishes on the house success control rather than a redirect", () => {
    const done = normalize(region('title="Eval Created"', "/>"))
    expect(done).toContain(
      'subtitle="You\'ve created a new eval, including an eval dataset and aligned judge!"',
    )
    expect(done).toContain('button_text={created_eval_href ? "View Eval"')
    // A save with no id has no eval page to offer, so it points at the list.
    expect(done).toContain(
      "link={created_eval_href ?? `/specs/${project_id}/${task_id}`}",
    )
  })

  it("suppresses the leave guards by state, not by a latch", () => {
    // A latch that is only ever set stays set: a wizard that somehow returned
    // to a live step would be unguarded. The guard reads where the wizard is,
    // so it comes back on by itself.
    expect(normalized).toContain(
      '$: leave_guard_suppressed = resetting || current_step === "done"',
    )
    expect(normalized).not.toContain("leave_guard_suppressed = true")
  })

  it("makes the finished state terminal", () => {
    // The wizard stays mounted behind the success screen, so Back would
    // otherwise land on the graded review with its save gate met — one click
    // from a second Spec with the same batch tag, or a paid round on an eval
    // that already shipped.
    const sync = normalize(
      region("function sync_step_from_history(", "abort_copilot_request()"),
    )
    expect(sync).toContain("if (saved_eval_created) {")
    expect(sync).toContain("goto(finished_destination)")

    // And the state those two actions would need is dropped on the way in, so
    // a bug that got past the guard still has nothing to act on.
    const finish = normalize(
      region("function finish_on_done_screen(", 'replace_step("done")'),
    )
    expect(finish).toContain("saved_eval_created = true")
    expect(finish).toContain("trace_claims = []")
    expect(finish).toContain("trace_reviews = []")
  })

  it("finishes both save branches the same way, after the draft is cleared", () => {
    // Multi-turn and single-turn each end in their own save; both must land
    // on the same screen, and neither may do it while a draft still points at
    // the work that just shipped.
    expect(normalized.split("finish_on_done_screen(saved.id)").length - 1).toBe(
      2,
    )
    const save_body = normalize(
      region("async function on_save() {", "function create_manually()"),
    )
    // The clear is awaited before the flip, on both paths.
    expect(
      save_body.split("await clear_builder_draft(").length - 1,
    ).toBeGreaterThanOrEqual(2)
    const [first, second] = save_body.split("finish_on_done_screen(saved.id)")
    expect(first).toContain("await clear_builder_draft(")
    expect(second).toContain("await clear_builder_draft(")
  })
})

// The wizard's own copy: what each step is called, and what the screens the
// steps open on say while they work.
describe("wizard step copy", () => {
  const step_names = function_body(
    'function step_name_for(step: Exclude<BuilderStep, "save" | "done">): string {',
  )

  it("names each step after what the user does on it", () => {
    const named = normalize(step_names)
    expect(named).toContain('case "clarify": return "Clarify Eval"')
    expect(named).toContain('case "refine": return "Review Updated Eval"')
    expect(named).toContain('case "generate": return "Create Eval Dataset"')
  })

  it("asks the refine step's question in the header's second line", () => {
    // The step shows the eval rewritten from the user's answers, so the
    // header asks what the step exists to answer — and only there. The review
    // step's own second line is pinned separately, below.
    expect(
      contains(
        '$: page_sub_subtitle = current_step === "refine" ' +
          '? "We\'ve integrated your feedback, does it look right?" ' +
          ': review_can_show_spec ? "Eval Description" : ""',
      ),
    ).toBe(true)
    expect(contains("sub_subtitle={page_sub_subtitle}")).toBe(true)
  })

  it("says what the minting screen is making, and leaves the count to the bar", () => {
    // The progress bar under the line already carries how far along the run
    // is, so the sentence says what is being made and stops there.
    expect(contains('? "Creating Eval Dataset"')).toBe(true)
    expect(contains("? `Creating ${planned_total} dataset items.`")).toBe(true)
  })

  it("carries that name into the drive screen that follows, on both arms", () => {
    // Minting and driving are one stretch of work to someone watching it, so
    // the screen that follows the minting screen says the same thing it did.
    expect(page_source.match(/title="Creating Eval Dataset"/g)).toHaveLength(2)
    expect(page_source).not.toContain('title="Creating Eval Data"')
  })
})

// A drive that stopped with usable work left behind takes over the step: how
// much failed and the two ways out are the whole decision there. Every other
// stop kind keeps the banner over the plan, because its text is raw provider
// output and its way out runs through the plan.
describe("the stopped-drive screen", () => {
  const stop_step = region(
    "{#if show_plan_approval && batch_plan}",
    "{:else if !generation_loading",
  )
  const partial_branch = stop_step.slice(0, stop_step.indexOf("{:else}"))
  const plan_branch = stop_step.slice(stop_step.indexOf("{:else}"))
  const stop_actions = region(
    "$: stop_lead = drive_stop",
    "[stop_rerun_action, stop_continue_action]",
  )

  it("takes over the step only for a stop that left usable work behind", () => {
    expect(normalize(partial_branch)).toContain(
      "{#if drive_stop && is_partial_stop(drive_stop)}",
    )
    expect(partial_branch).toContain("<Intro")
    expect(partial_branch).not.toContain("<Warning")
  })

  it("keeps the banner above the plan for every other stop kind", () => {
    // Preflight, abort and all-failed carry long raw provider text and a
    // recovery that runs through the plan, so the plan has to stay on screen.
    expect(plan_branch).toContain("<Warning")
    expect(plan_branch).toContain("<KilnProBatchPlan")
    // Always the error color here: what is left once the partial stop has
    // taken its own screen is a failed config or a failed batch.
    expect(normalize(plan_branch)).toContain('warning_color="error"')
    expect(plan_branch).not.toContain("is_partial_stop")
    expect(normalize(plan_branch)).toContain("markdown trusted")
  })

  it("demotes the plan's own primary while the Continue row is beside it", () => {
    // Two solid primaries on one screen is two leads. The plan's generate
    // button steps back to an outline whenever the survivors row co-renders.
    expect(normalize(plan_branch)).toContain(
      "generate_button_outline={has_driven_results && drive_stop !== null}",
    )
  })

  it("names the screen and hands it the two ways forward", () => {
    const screen = normalize(partial_branch)
    expect(screen).toContain('<Intro title="Errors During Dataset Creation"')
    expect(screen).toContain("action_buttons={stop_screen_actions}")
  })

  it("says what happened from the same source the banner uses", () => {
    // One function writes the sentence, so the screen and the banner can
    // never end up telling the same stop two different ways.
    expect(normalize(partial_branch)).toContain(
      "description_markdown={drive_stop_banner( drive_stop, " +
        "drive_run_config_name, drive_run_config_model, )}",
    )
    expect(partial_branch).not.toContain("description_paragraphs")
  })

  it("renders the leading action first and makes it the solid one", () => {
    // The house offer screen stacks its primary above the alternative, so the
    // order and the emphasis both come from the same decision.
    const actions = normalize(stop_actions)
    expect(actions).toContain(
      "label: `Continue With ${drive_stop?.survivors ?? 0}`, " +
        'onClick: on_continue_with_survivors, is_primary: stop_lead === "continue",',
    )
    expect(actions).toContain(
      'label: "Re-run Batch", onClick: open_drive_settings, ' +
        'is_primary: stop_lead === "rerun",',
    )
    expect(actions).toContain(
      '$: stop_screen_actions = stop_lead === "continue" ' +
        "? [stop_continue_action, stop_rerun_action] " +
        ": [stop_rerun_action, stop_continue_action]",
    )
  })

  it("swallows the keyboard shortcut instead of acting on it", () => {
    // The screen's buttons carry no shortcut hint, and which of them leads
    // changes with the batch, so the shortcut would fire an action the user
    // was never offered — on most of those batches a paid re-run. The
    // keystroke is consumed all the same, or a focused button takes the Enter.
    const shortcut = normalize(
      region(
        "if (drive_stop && is_partial_stop(drive_stop)) {",
        "if (has_driven_results) {",
      ),
    )
    expect(shortcut).toContain("event.preventDefault() return }")
    expect(shortcut).not.toContain("on_continue_with_survivors()")
    expect(shortcut).not.toContain("open_drive_settings()")
  })
})
