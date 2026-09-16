<script lang="ts">
  // One claim in the claim review, rendered as a row of the caller's claims
  // table: one decision the judge made, written so the reviewer can vote on
  // it where they read it. The text carries its own evidence, with [n] chips
  // that open the trace at the cited span. The
  // reviewer answers Agree (the judge got this decision right) or Disagree
  // (it got it wrong); a disagreement needs a reason, which feeds judge
  // refinement. Every claim renders through this one component, the verdict
  // claim included: the builder writes the verdict as an ordinary last claim,
  // and the review derives the reviewer's overall call from its grade
  // (human_verdict in claim_evidence.ts). Nothing here names the judge or
  // its score: everything on screen is the builder's text, the verdict
  // claim's "It passes" / "It fails" included.
  import { onDestroy } from "svelte"
  import Warning from "$lib/ui/warning.svelte"
  import FormElement from "$lib/utils/form_element.svelte"
  import ClaimText from "./claim_text.svelte"
  import {
    split_claim_note,
    type Citation,
    type Claim,
    type ClaimVerdict,
  } from "./claim_evidence"

  export let claim: Claim
  // Position in the review's claim list, shown in the table's "#" column as
  // index + 1: the number the builder's own cross-references ("#1") use.
  export let index: number
  export let verdict: ClaimVerdict
  export let on_cite: (citation: Citation) => void = () => {}
  // The review decides which claims are expanded. A collapsed claim shows its
  // number, its state and one line of its text, with Edit to expand it again.
  export let open = true
  // Asks the review to expand this claim (Edit).
  export let on_open: () => void = () => {}
  // Tells the review the claim was answered, so it can collapse and expand
  // claims around it.
  export let on_answer: (agrees: boolean) => void = () => {}

  // The trailing "Note:" paragraph renders apart from the claim, muted; the
  // body is everything else.
  $: split = split_claim_note(claim.text)

  // Exported so the review's A / D shortcuts answer the expanded claim through
  // the same path a click does.
  export function set_agrees(value: boolean) {
    verdict.agrees = value
    // Agreeing hides the reason box — clear any text typed while disagreeing
    // so the user submits exactly what they see. A stale why would otherwise
    // ride the agree grade into the persisted review and judge refinement.
    if (value) verdict.why = ""
    verdict = verdict
    on_answer(value)
    // The reason box is inside FormElement, which exposes no element ref, so
    // the focus goes through the id this component owns and gives the field.
    if (!value)
      setTimeout(() => document.getElementById(why_id(index))?.focus(), 0)
  }

  // The collapsed row's state, in neutral words: a disagreement is not an
  // error, so neither answer is coloured.
  $: state_label =
    verdict.agrees === true
      ? "Agreed"
      : verdict.agrees === false
        ? "Disagreed"
        : "Not decided"

  // One plain line of the claim for the collapsed row. The [n] markers are
  // dropped: a chip cut off mid-line is not something to click.
  $: collapsed_text = split.body.replace(/\s*\[\d+\]/g, "")

  // The look of one answer button, given the side it answers. While the claim
  // is undecided both sides carry btn-outline btn-primary: that is the house
  // treatment for "pick one of these", and the ask is on both equally. Once
  // answered, the chosen side is filled secondary and the other drops to a
  // plain outline — the ask is spent, so the primary colour goes with it.
  $: answer_class = (side: boolean) =>
    verdict.agrees === side
      ? "btn-secondary"
      : verdict.agrees === null
        ? "btn-outline btn-primary"
        : "btn-outline"

  // The reason field and the hint that describes it, named once so the field,
  // its label and the aria wiring cannot drift apart.
  const why_id = (i: number) => `claim-why-${i}`
  const why_hint_id = (i: number) => `claim-why-hint-${i}`
  const WHY_LABEL =
    "What do you disagree with? What should the judge have done instead?"

  // Hint under the reason box. A one-line reason rarely gives judge refinement
  // enough to act on, so the hint nudges for more while the reviewer is still
  // typing. Length is the trimmed length; the tier changes only after a pause
  // so it never flips mid-keystroke, but it clears at once when the reason is
  // long enough or emptied. Length never gates saving: only an empty reason
  // does, and that stays the review's job.
  //
  // Both thresholds are the first trimmed length of their tier: under 20 reads
  // as a fragment, 40 and up is left alone.
  const BRIEF_REASON_MIN = 20
  const FULL_REASON_MIN = 40
  const REASON_HINT_DELAY_MS = 500

  type ReasonHint = "short" | "brief"
  const REASON_HINTS: Record<ReasonHint, string> = {
    short: "Likely too short to help improve the judge. What did it get wrong?",
    brief: "More details here would help the judge improve faster.",
  }

  let reason_hint: ReasonHint | null = null
  let hint_timer: ReturnType<typeof setTimeout> | null = null
  // The verdict the hint on screen belongs to, and the reason text it was
  // scheduled for. Both guard the reactive run below, which fires far more
  // often than the reviewer types.
  let hinted_verdict: ClaimVerdict | null = null
  let hinted_why: string | null = null

  function hint_for(why: string): ReasonHint | null {
    const length = why.trim().length
    if (length === 0 || length >= FULL_REASON_MIN) return null
    return length < BRIEF_REASON_MIN ? "short" : "brief"
  }

  // Runs on every keystroke, and on every other change to the verdict too:
  // Svelte treats the object as new whenever the parent writes it back.
  $: sync_reason_hint(verdict)

  function sync_reason_hint(next_verdict: ClaimVerdict) {
    // A different verdict object means a different claim on screen, so drop
    // the old claim's hint at once rather than leaving it over new text.
    if (next_verdict !== hinted_verdict) {
      hinted_verdict = next_verdict
      hinted_why = null
      clear_hint_timer()
      reason_hint = null
    }
    update_reason_hint(next_verdict.why)
  }

  function update_reason_hint(why: string) {
    // Only a change to the reason restarts the pause. Writebacks and unrelated
    // parent renders carry the same text and must leave the timer alone.
    if (why === hinted_why) return
    hinted_why = why
    clear_hint_timer()
    const next = hint_for(why)
    if (next === null) {
      reason_hint = null
      return
    }
    hint_timer = setTimeout(() => {
      reason_hint = next
      hint_timer = null
    }, REASON_HINT_DELAY_MS)
  }

  function clear_hint_timer() {
    if (hint_timer) clearTimeout(hint_timer)
    hint_timer = null
  }

  onDestroy(clear_hint_timer)
</script>

<!-- One row of the claims table (the caller owns the table and its header).
     Collapsed and expanded are the same three cells — the number, the claim,
     the decision — so a claim never changes shape, only height: collapsed the
     middle cell clamps to one line and the decision cell offers Edit;
     expanded it carries the evidence, the aside and the reason box, and the
     decision cell offers the two answers. -->
<tr id="claim-card-{index}">
  <td class="align-middle text-sm text-gray-500">{index + 1}</td>
  <!-- max-w-0 with the column's w-full: the cell takes the slack the other two
       columns leave, and nothing inside it can widen the column — which is
       what makes the collapsed clamp below a clamp rather than a long row. -->
  <td class="align-middle max-w-0">
    {#if !open}
      <div class="text-sm text-gray-500 truncate">{collapsed_text}</div>
    {:else}
      <!-- The claim owns the gaps between its own parts; the table owns the
           gap between claims. -->
      <div class="flex flex-col gap-3">
        <p class="text-sm leading-relaxed whitespace-normal">
          <ClaimText text={split.body} citations={claim.citations} {on_cite} />
        </p>

        {#if split.note !== null}
          <!-- The builder's aside, muted so it reads as context rather than as
               part of the decision being voted on. -->
          <p
            class="text-sm text-gray-500 leading-relaxed whitespace-normal"
            data-claim-note
          >
            <ClaimText
              text={split.note}
              citations={claim.citations}
              {on_cite}
            />
          </p>
        {/if}

        {#if verdict.agrees === false}
          <div class="flex flex-col gap-1 mt-2">
            <!-- A null validator, because an empty reason is not a form error to
               spell out: the control's own required treatment already says it,
               and the default validator would overwrite the placeholder with
               the message. -->
            <FormElement
              id={why_id(index)}
              inputType="textarea"
              label={WHY_LABEL}
              placeholder="This is wrong because…"
              aria_describedby={why_hint_id(index)}
              validator={() => null}
              bind:value={verdict.why}
            />
            <!-- Fixed height whether or not a hint is showing, so the row never
               shifts as the reviewer types. Announced politely so it reads out
               without taking focus; the icon and the words carry the meaning,
               the colour only ranks it. There is deliberately no "long
               enough" state. -->
            <div id={why_hint_id(index)} class="h-5" aria-live="polite">
              {#if reason_hint}
                <Warning
                  warning_color={reason_hint === "short" ? "warning" : "gray"}
                  warning_icon="exclaim"
                  inline
                  warning_message={REASON_HINTS[reason_hint]}
                />
              {/if}
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </td>
  <!-- The decision. The two answers sit side by side on one line and never
       wrap: they are what sizes this column, so shrink-0 keeps their own width
       the column's floor rather than letting the table squeeze them. -->
  <td class="align-middle">
    <div class="flex items-center gap-2">
      {#if !open}
        <span id="claim-state-{index}" class="text-sm text-gray-500 shrink-0">
          {state_label}
        </span>
        <button
          id="claim-edit-{index}"
          class="btn btn-sm shrink-0"
          on:click={on_open}
        >
          Edit
        </button>
      {:else}
        <!-- Agree / Disagree, in the words the payload stores, so nothing is
             translated between the click and the record. Both sides take their
             look from one place (answer_class), so the pair cannot drift. -->
        <button
          id="claim-agree-{index}"
          class="btn btn-sm shrink-0 {answer_class(true)}"
          on:click={() => set_agrees(true)}
        >
          Agree
          <span class="opacity-80 ml-2 text-xs font-light">A</span>
        </button>
        <button
          id="claim-disagree-{index}"
          class="btn btn-sm shrink-0 {answer_class(false)}"
          on:click={() => set_agrees(false)}
        >
          Disagree
          <span class="opacity-80 ml-2 text-xs font-light">D</span>
        </button>
      {/if}
    </div>
  </td>
</tr>
