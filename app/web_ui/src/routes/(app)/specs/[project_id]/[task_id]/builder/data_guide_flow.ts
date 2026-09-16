// The eval builder's Data Guide flow on Step 4 (single-turn only), as pure
// predicates so every condition that opens or shuts the offer, and the one
// that labels a plan as guide-drafted, is pinned by a test rather than
// spread across the page's render conditions.

// Where the Step 4 entry read of the task's guide stands. "failed" is an
// HTTP error or a thrown fetch: the task may well have a guide, so the step
// plans without one rather than offering to create one.
export type DataGuideRead = "unread" | "reading" | "none" | "found" | "failed"

export type DataGuideDecisionState = {
  on_generate_step: boolean
  is_multi_turn: boolean
  read: DataGuideRead
  // The user chose Continue Without Data Guide on this draft.
  skipped: boolean
  has_plan: boolean
  // Driven results exist for the current plan.
  has_results: boolean
  // A plan request (or any other Step 4 stage) is in flight.
  planning: boolean
}

// Whether Step 4 is waiting on the guide before it can plan: the read is in
// flight, or it found no guide and the user has not yet chosen. Persisted on
// the draft, so a reload or a return from setting the guide up lands back on
// this step. Only on Step 4 (Back to an earlier step ends the wait), only
// single-turn, never once the user skipped, and never over a plan, results,
// or a request in flight.
export function data_guide_decision_pending(
  state: DataGuideDecisionState,
): boolean {
  return (
    state.on_generate_step &&
    !state.is_multi_turn &&
    !state.skipped &&
    !state.has_plan &&
    !state.has_results &&
    !state.planning &&
    (state.read === "reading" || state.read === "none")
  )
}

// Whether the "Create a Data Guide" offer is on screen: the wait above, once
// the read has returned and found nothing. Never before it returns, because
// until then nothing knows the task has no guide.
export function data_guide_offer_open(state: DataGuideDecisionState): boolean {
  return data_guide_decision_pending(state) && state.read === "none"
}

// Whether the plan on screen was drafted with the guide on. The committed
// on/off state always describes the current plan, because every change to it
// re-plans; so a plan with the guide on and a guide to send is a guide-drafted
// plan. Multi-turn never sends the guide.
export function plan_used_data_guide(state: {
  is_multi_turn: boolean
  has_plan: boolean
  use_data_guide: boolean
  has_guide: boolean
}): boolean {
  return (
    !state.is_multi_turn &&
    state.has_plan &&
    state.use_data_guide &&
    state.has_guide
  )
}
