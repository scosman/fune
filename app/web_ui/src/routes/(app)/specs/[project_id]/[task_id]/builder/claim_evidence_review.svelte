<script lang="ts">
  // Claim review step — one trace at a time. The reviewer reads the Overview,
  // then votes Agree or Disagree on every claim the builder wrote, opening a
  // [n] citation into the trace modal only for the hard calls. Everything on
  // screen is the builder's text: the judge's score and reasoning never
  // render here, since the reviewer's calls are what calibrate the judge.
  //
  // Claims are answered where they are expanded: a case opens its first
  // undecided claim, and the others wait as one line each with their state and
  // Edit (open_claims_by_trace). The Overview sits beside the claims on wide
  // windows, and the numbered bar above them carries the decided count and a
  // button per case.
  //
  // The overall pass/fail call is the verdict claim's grade — the builder
  // writes the verdict as the last claim, and a case without one is not put in
  // front of the reviewer at all (reviewable_subset). Next is gated on the
  // whole trace being graded (is_trace_reviewed).
  //
  // Subset review: `selected_indices` is the judge-stratified sample the
  // reviewer grades (sized to the golden answer key) — the review shows
  // exactly these traces, mirroring the single-turn flow where the user
  // reviews exactly what's presented. Claims build lazily: opening a trace
  // triggers its build via `on_open_trace`, and the panel shows the build in
  // progress until they arrive. A trace whose build already failed is not in
  // this list at all: the claims gate resolves every selected trace before the
  // review opens, and the page drops the failures from the subset.
  import ClaimCard from "./claim_card.svelte"
  import ClaimText from "./claim_text.svelte"
  import ClaimTraceModal from "./claim_trace_modal.svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import Output from "$lib/ui/output.svelte"
  import SettingsHeader from "$lib/ui/settings_header.svelte"
  // The nav row hand-rolls FormContainer's submit button, so it renders the
  // same keyboard hint using the same platform check.
  import { isMacOS } from "$lib/utils/platform"
  import {
    is_trace_reviewed,
    type Citation,
    type ClaimVerdict,
    type TraceClaims,
    type TraceReview,
  } from "./claim_evidence"

  export let traces: TraceClaims[]
  // Two-way bound so the parent reads verdicts at save time.
  export let verdicts: TraceReview[]
  // Indices of the traces the reviewer grades; empty = all.
  export let selected_indices: number[] = []
  // Called with the trace index being shown — the parent builds its claims
  // if needed. Also the retry hook for a failed build.
  export let on_open_trace: (index: number) => void = () => {}
  // Called when the reviewer takes the forward action on the last case. What
  // happens next — a refine round or a straight save — is the parent's call,
  // and it asks the reviewer in a dialog when there is feedback to act on.
  export let on_save: () => void = () => {}
  // The review gate, computed by the parent (enough traces reviewed). Holds
  // the forward action on the last conversation disabled until enough of the
  // batch has been graded, the way every other form in the app holds a submit.
  export let save_disabled = true
  // What the judge judged, in the caller's vocabulary: "conversation" for
  // multi-turn, "example" for single-turn.
  export let judged_noun = "example"
  // True while the reviewer is on the last selected trace — the only position
  // where the primary action renders. Bound out (read-only for the parent) so
  // anything the parent stacks under that action appears only alongside it.
  export let on_last_trace = false
  // The eval's own description, read-only. The review shows what each
  // conversation did but never what the eval asks for, so a reviewer who
  // forgot it can reread it. Empty or null mounts no dialog at all, and
  // show_spec_dialog() below becomes a no-op.
  export let spec_text: string | null = null

  // Height at which an overview folds behind Show All. Output reads its cap in
  // pixels only, so this is a pixel value, never a viewport unit. Calibrated
  // on the captured corpus: the longest overview there is about 400 characters,
  // which renders around 260px tall in this column at a 1440px window and about
  // 320px on a narrower one; this cap is roughly twice that, so no real overview
  // folds and only a runaway of six or more sentences does.
  const OVERVIEW_MAX_HEIGHT = "480px"

  let current_index = 0
  let trace_modal: ClaimTraceModal | null = null
  let spec_dialog: Dialog | null = null
  let claim_cards: ClaimCard[] = []

  $: has_spec_text = (spec_text ?? "").trim().length > 0

  // Opens the eval description. The trigger is the page header's sub-line
  // rather than a control on this step: a reviewer who wants to reread what
  // the eval asks for looks up, not into the work. The dialog stays here, with
  // the text it renders; the page reaches in the same way this component
  // reaches into its own trace modal.
  export function show_spec_dialog() {
    spec_dialog?.show()
  }

  // The two section sub-lines. The claims section carries the step's purpose,
  // since the step no longer has a header of its own to carry it: answering
  // the claims IS confirming the judge. The overview's says what the panel
  // beside them holds.
  const CLAIMS_DESCRIPTION =
    "Confirm the judge is aligned to your expectations."
  const OVERVIEW_DESCRIPTION = "A summary of this task's run."

  $: selected =
    selected_indices.length > 0 ? selected_indices : traces.map((_, i) => i)
  $: current = traces[current_index]
  $: current_verdicts = verdicts[current_index]

  // Start on the first selected trace (a fresh mount has current_index 0,
  // which may be unselected under subset review).
  let started_on_selected = false
  $: if (!started_on_selected && selected.length > 0) {
    started_on_selected = true
    current_index = selected[0]
  }

  // Report every shown trace to the parent so lazily-built claims kick off
  // the moment the reviewer lands on a trace (idempotent parent-side).
  $: report_opened(current_index)
  function report_opened(index: number) {
    if (traces[index]) on_open_trace(index)
  }

  // Keep original indices, since verdicts are positional.
  $: visible = (current?.claims ?? []).map((claim, index) => ({ claim, index }))

  function open_citation(citation: Citation) {
    if (current) trace_modal?.open_citation(current, citation)
  }

  // Previous/Next walk the selected sequence.
  function go_prev() {
    const prior = selected.filter((i) => i < current_index)
    if (prior.length > 0) current_index = prior[prior.length - 1]
  }
  function go_next() {
    const later = selected.filter((i) => i > current_index)
    if (later.length > 0) current_index = later[0]
  }
  $: has_prev = selected.some((i) => i < current_index)
  $: has_next = selected.some((i) => i > current_index)
  $: on_last_trace = !has_next

  // Next is gated on the CURRENT conversation being fully answered, and says
  // so only by being disabled, as every other form in the app does. Save takes
  // the forward slot on the last conversation, but only once the save gate is
  // met.
  $: current_reviewed = is_trace_reviewed(current, current_verdicts)
  // The one gate on the one forward button: mid-review the current case has to
  // be answered, and on the last case the parent's review gate decides. Both
  // rules feed the same button, so the slot can never be enabled for one
  // meaning and disabled for another.
  $: forward_disabled = has_next ? !current_reviewed : save_disabled
  $: show_forward_hint = !has_next && !save_disabled

  // The reviewer's position in the graded sequence, which the line over the
  // pills states.
  $: case_position = selected.indexOf(current_index) + 1

  // Which claims are expanded, per case. On entry a case expands its first
  // undecided claim. Agree collapses the claim and expands the next undecided
  // one. Disagree keeps the claim expanded for its reason and also expands the
  // next undecided claim, so the reviewer can answer it straight away. A
  // disagree collapses when the reviewer answers or opens another claim, but
  // only once its reason is filled in: an empty reason is what holds Next, so
  // it stays in view.
  let open_claims_by_trace: Record<number, number[]> = {}
  // The claim the A / D shortcuts answer: the one the reviewer reached last.
  let active_claim_by_trace: Record<number, number> = {}
  $: pin_open_claims(current_index, current, current_verdicts)
  function pin_open_claims(
    trace_index: number,
    trace: TraceClaims | undefined,
    review: TraceReview | undefined,
  ) {
    if (open_claims_by_trace[trace_index] !== undefined) return
    // Lazily-built claims arrive later; wait until their slots exist.
    const slots = review?.claim_verdicts ?? []
    if (trace?.claims_state !== "built" || slots.length === 0) return
    // A disagree still missing its reason is what holds Next, so it opens
    // alongside the first undecided claim, as it was when the case was left.
    const first = first_undecided(slots)
    const waiting = unfinished(
      slots.map((_, i) => i),
      slots,
    )
    set_open(
      trace_index,
      [...waiting, first],
      first >= 0 ? first : waiting[0] ?? -1,
    )
  }
  $: open_claims = open_claims_by_trace[current_index] ?? []
  $: active_claim = active_claim_by_trace[current_index] ?? -1

  function first_undecided(slots: ClaimVerdict[]): number {
    return slots.findIndex((v) => v.agrees === null)
  }
  // The open claims that stay open whatever the reviewer does next: a
  // disagree whose reason is still empty.
  function unfinished(open: number[], slots: ClaimVerdict[]): number[] {
    return open.filter(
      (i) => slots[i]?.agrees === false && slots[i].why.trim().length === 0,
    )
  }
  function set_open(trace_index: number, open: number[], active: number) {
    const unique = [...new Set(open.filter((i) => i >= 0))]
    open_claims_by_trace = { ...open_claims_by_trace, [trace_index]: unique }
    active_claim_by_trace = { ...active_claim_by_trace, [trace_index]: active }
  }
  // Edit: expand this claim, collapsing every other claim that is finished.
  function open_claim(index: number) {
    const slots = current_verdicts?.claim_verdicts ?? []
    set_open(current_index, [...unfinished(open_claims, slots), index], index)
  }
  // A claim was answered. With no undecided claim left, only unfinished
  // disagrees stay open, so a finished case reads as a list of decided claims.
  function answered(index: number, agrees: boolean) {
    const slots = current_verdicts?.claim_verdicts ?? []
    const next = first_undecided(slots)
    const others = unfinished(
      open_claims.filter((i) => i !== index),
      slots,
    )
    if (agrees) set_open(current_index, [...others, next], next)
    else
      set_open(
        current_index,
        [...others, index, next],
        next >= 0 ? next : index,
      )
  }

  // Decided counts: for this case beside the Claims header, and for the whole
  // subset in the case header. A disagreement without its reason counts as
  // decided here; Next still waits for the reason.
  const count_decided = (review: TraceReview | undefined) =>
    (review?.claim_verdicts ?? []).filter((v) => v.agrees !== null).length
  $: case_decided = count_decided(current_verdicts)
  $: case_claims = current?.claims?.length ?? 0
  $: batch_counts = count_batch(selected, traces, verdicts)
  function count_batch(
    indices: number[],
    all_traces: TraceClaims[],
    reviews: TraceReview[],
  ) {
    let decided = 0
    let total = 0
    for (const i of indices) {
      decided += count_decided(reviews[i])
      total += all_traces[i]?.claims?.length ?? 0
    }
    return { decided, total }
  }

  // One button per case in the case header: the current case, cases already
  // reviewed, and the rest. Any case can be jumped to.
  $: case_steps = selected.map((trace_index, position) => ({
    trace_index,
    position,
    reviewed: is_trace_reviewed(
      traces[trace_index],
      trace_index === current_index ? current_verdicts : verdicts[trace_index],
    ),
  }))

  // A answers Agree and D Disagree on the claim reached last. Ignored while
  // typing, while a dialog is open, and with a modifier held so the browser's
  // own shortcuts still work.
  function handle_keydown(event: KeyboardEvent) {
    // A held key repeats, and would answer claim after claim.
    if (event.repeat) return
    if (event.metaKey || event.ctrlKey || event.altKey) return
    const key = event.key.toLowerCase()
    if (key !== "a" && key !== "d") return
    const target = event.target as HTMLElement | null
    if (
      target &&
      (target.isContentEditable ||
        ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
    )
      return
    if (document.querySelector("dialog[open]")) return
    if (current?.claims_state !== "built") return
    if (!open_claims.includes(active_claim)) return
    const card = claim_cards[active_claim]
    if (!card) return
    event.preventDefault()
    card.set_agrees(key === "a")
  }
</script>

<svelte:window on:keydown={handle_keydown} />

<!-- One vertical stack owns every gap on the step: gap-6 between sections
     (the form rhythm the rest of the app is built on), gap-3 inside one.
     Nothing below sets its own margin, so the spacing is read in one place
     and cannot drift element by element. -->
<div class="flex flex-col gap-6">
  {#if current && current_verdicts}
    <!-- The reviewer's position: one line saying where they are and how much
         of the batch is graded, over a pill per case. It is a progress readout
         first and a control second, so it reads as a row of segments rather
         than as numbered buttons — but every pill is still a jump to its case.
         The line is the only place the numbers are written; the pills carry
         state in colour alone. -->
    <div class="flex flex-col items-end gap-1">
      <span id="review-batch-count" class="text-sm text-gray-500"
        >Case {case_position} of {selected.length} · {batch_counts.decided}/{batch_counts.total}
        decided</span
      >
      <!-- Colour is the whole of a pill's meaning: primary for the case open
           now, success for one already graded, and the theme's inactive grey
           for one still to do. A graded case is green whatever the reviewer
           decided — the colour tracks that the work is done, not that the
           judge was right, so a disagreement is as complete as an agreement.
           The button is twice the bar's height with the bar centred in it, so
           the bar can be thin without the target being a sliver — margin would
           give the same spacing but none of the extra target. -->
      <div class="flex flex-wrap justify-end gap-2" id="review-case-steps">
        {#each case_steps as step (step.trace_index)}
          <button
            class="w-8 h-4 flex items-center"
            aria-label={`Case ${step.position + 1}`}
            aria-current={step.trace_index === current_index
              ? "step"
              : undefined}
            on:click={() => (current_index = step.trace_index)}
          >
            <span
              class="w-8 h-2 rounded-full {step.trace_index === current_index
                ? 'bg-primary'
                : step.reviewed
                  ? 'bg-success/60'
                  : 'bg-neutral'}"
            ></span>
          </button>
        {/each}
      </div>
    </div>

    <!-- Overview beside the claims on wide windows, in a grid that stacks to
         one column below xl the way the judge form's side pane does. The
         Overview holds still while the reviewer scrolls a long claim list. -->
    <div
      class="grid grid-cols-1 gap-y-6 xl:gap-x-16 xl:items-start xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]"
    >
      <!-- The Overview section renders in every state, so the step never
           changes shape between cases: the header and its two escape hatches
           hold still, and only the body differs — the overview when the build
           produced one, and the in-panel wait while it is still running. The
           [n] chips open the same trace view the claims do. -->
      <div
        id="review-overview"
        class="flex flex-col gap-3 min-w-0 xl:sticky xl:top-6"
      >
        <SettingsHeader
          title="Overview"
          subtitle={OVERVIEW_DESCRIPTION}
          show_divider={false}
        />

        {#if current.overview}
          <div class="flex flex-col gap-1">
            <!-- The read-only surface the rest of the app shows read-only
                 content on, with the chips rendered into its slot: Output
                 prints a string and cannot carry a clickable citation itself,
                 so the caller renders the body and Output keeps the panel and
                 the copy button (which copies the plain text passed as
                 raw_output). Capped in pixels so a runaway overview folds
                 behind Show All rather than pushing the buttons and the claims
                 down the page; the value and why it is a pixel value are on the
                 constant. -->
            <Output
              raw_output={current.overview.text}
              max_height={OVERVIEW_MAX_HEIGHT}
            >
              <p class="text-sm leading-relaxed">
                <ClaimText
                  text={current.overview.text}
                  citations={current.overview.citations}
                  on_cite={open_citation}
                />
              </p>
            </Output>
            <!-- The way out of the summary and into what it summarises, under
                 the panel's bottom right: the overview is the short form of the
                 trace, so the link to the long form belongs on it rather than
                 in a button row of its own. -->
            <div class="flex justify-end">
              <button
                id="view-full-trace"
                class="link underline text-sm text-gray-500"
                on:click={() => current && trace_modal?.open_trace(current)}
              >
                Full Trace
              </button>
            </div>
          </div>
        {:else if current.claims_state === "unbuilt" || current.claims_state === "building"}
          <!-- The build starts on open, so both render as in-progress, in the
               body the overview will fill. Named rather than written as "not
               built": a failed build never reaches this component, and if one
               ever did, an honest empty body beats a spinner that never stops. -->
          <div class="text-center py-12 text-gray-500">
            <div class="loading loading-dots loading-md mb-2"></div>
            <div class="text-sm">Analyzing this {judged_noun}…</div>
          </div>
        {/if}
      </div>

      <!-- The claims are a list of fields, so they sit at the form's
           field-to-field gap, not the tighter gap a claim uses inside itself.
           The section header is the list's first item and takes the same, and
           the nav row is its last, so the forward action sits under the work. -->
      <div class="flex flex-col gap-6 min-w-0">
        {#if current.claims_state === "built"}
          <SettingsHeader
            title="Claims"
            subtitle={CLAIMS_DESCRIPTION}
            show_divider={false}
          >
            <svelte:fragment slot="actions">
              <span id="review-case-count" class="text-sm text-gray-500">
                {case_decided}/{case_claims} decided
              </span>
            </svelte:fragment>
          </SettingsHeader>
          <!-- The house table: the claims are a numbered list of decisions to
               answer, which is what a table is for.
               Auto layout, sized by the outer two columns: the number hugs
               its digits (w-px), the decision hugs its two buttons with a
               floor that keeps short states from looking cramped, and the
               claim column takes everything left over (w-full). Its cells
               carry max-w-0, which is what lets a collapsed claim clamp to one
               line rather than widening the column to fit the sentence. -->
          <div class="rounded-lg border">
            <table class="table">
              <thead>
                <tr>
                  <th class="w-px">#</th>
                  <th class="w-full">Claim from Judge</th>
                  <th style="min-width: 250px">Decision</th>
                </tr>
              </thead>
              <tbody>
                {#each visible as { claim, index } (index)}
                  <ClaimCard
                    bind:this={claim_cards[index]}
                    {claim}
                    {index}
                    bind:verdict={current_verdicts.claim_verdicts[index]}
                    open={open_claims.includes(index)}
                    on_open={() => open_claim(index)}
                    on_answer={(agrees) => answered(index, agrees)}
                    on_cite={open_citation}
                  />
                {/each}
              </tbody>
            </table>
          </div>
        {/if}

        <!-- Previous on the left, the forward action on the right. Wizard-step
             navigation is the browser's Back/Forward. Previous is hidden
             rather than disabled where there is nothing to go back to. -->
        <div class="flex items-center justify-between gap-2">
          {#if has_prev}
            <button class="btn" on:click={go_prev}>Previous</button>
          {:else}
            <div></div>
          {/if}
          <!-- One forward button in every position, reading Next throughout.
               Where it leads changes with the position — the next case, or the
               parent's dialog for a finished review — but the word does not: a
               label that differs between the disabled and the enabled state
               reads as two buttons swapping places rather than one button
               opening. What the click leads to on the last case is settled in
               that dialog anyway, not in the word on the button. -->
          <button
            id="review-next"
            class="btn btn-primary"
            on:click={has_next ? go_next : on_save}
            disabled={forward_disabled}
          >
            Next
            {#if show_forward_hint}
              <!-- The hint rides the button only where the shortcut fires it:
                   the page binds ⌘↵ to the same save, and only on the last
                   case with the gate met. The house form shows the same hint
                   on its own submit. -->
              <span class="opacity-80 ml-2 text-xs font-light">
                {#if isMacOS()}
                  <span class="tracking-widest">⌘↵</span>
                {:else}
                  <span>ctrl ↵</span>
                {/if}
              </span>
            {/if}
          </button>
        </div>
      </div>
    </div>
  {/if}
</div>

<!-- One trace rendering for both arms: a single-turn run is a conversation of
     one turn, so the modal no longer needs to be told which arm it is on. -->
<ClaimTraceModal bind:this={trace_modal} />

<!-- One eval-level dialog, not one per conversation, so it lives outside the
     per-conversation markup. Wide, because the description is multi-paragraph
     prose that reads as a narrow ribbon at the default width. -->
{#if has_spec_text}
  <Dialog
    bind:this={spec_dialog}
    title="Eval Description"
    width="wide"
    action_buttons={[{ label: "Close", isCancel: true }]}
  >
    <div id="spec-text">
      <Output raw_output={spec_text ?? ""} show_border />
    </div>
  </Dialog>
{/if}
