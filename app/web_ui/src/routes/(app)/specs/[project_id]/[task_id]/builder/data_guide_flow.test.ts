import { describe, expect, it } from "vitest"
import {
  data_guide_decision_pending,
  data_guide_offer_open,
  plan_used_data_guide,
  type DataGuideDecisionState,
} from "./data_guide_flow"

// The one state that opens the offer: Step 4 of a single-turn task whose
// guide read came back empty, not skipped, before any plan or results exist.
const OPEN: DataGuideDecisionState = {
  on_generate_step: true,
  is_multi_turn: false,
  read: "none",
  skipped: false,
  has_plan: false,
  has_results: false,
  planning: false,
}

describe("data_guide_offer_open — the Step 4 offer gate", () => {
  it("opens for a single-turn task with no guide, not skipped, before planning", () => {
    expect(data_guide_offer_open(OPEN)).toBe(true)
  })

  it("never opens off Step 4", () => {
    expect(data_guide_offer_open({ ...OPEN, on_generate_step: false })).toBe(
      false,
    )
  })

  it("never opens on the multi-turn arm", () => {
    expect(data_guide_offer_open({ ...OPEN, is_multi_turn: true })).toBe(false)
  })

  it("never opens before the guide read returns", () => {
    expect(data_guide_offer_open({ ...OPEN, read: "unread" })).toBe(false)
    expect(data_guide_offer_open({ ...OPEN, read: "reading" })).toBe(false)
  })

  it("shuts once the task has a guide", () => {
    expect(data_guide_offer_open({ ...OPEN, read: "found" })).toBe(false)
  })

  it("never opens after a failed read — the task may have a guide", () => {
    expect(data_guide_offer_open({ ...OPEN, read: "failed" })).toBe(false)
  })

  it("shuts once the user continued without one", () => {
    expect(data_guide_offer_open({ ...OPEN, skipped: true })).toBe(false)
  })

  it("shuts once a plan exists", () => {
    expect(data_guide_offer_open({ ...OPEN, has_plan: true })).toBe(false)
  })

  it("shuts once results exist", () => {
    expect(data_guide_offer_open({ ...OPEN, has_results: true })).toBe(false)
  })

  it("shuts while a plan request is in flight", () => {
    expect(data_guide_offer_open({ ...OPEN, planning: true })).toBe(false)
  })
})

describe("data_guide_decision_pending — what the draft persists", () => {
  it("is pending while the read is in flight, so a restore re-enters Step 4", () => {
    expect(data_guide_decision_pending({ ...OPEN, read: "reading" })).toBe(true)
  })

  it("is pending while the offer is open", () => {
    expect(data_guide_decision_pending(OPEN)).toBe(true)
  })

  it("is not pending before the read starts, nor once it found a guide or failed", () => {
    expect(data_guide_decision_pending({ ...OPEN, read: "unread" })).toBe(false)
    expect(data_guide_decision_pending({ ...OPEN, read: "found" })).toBe(false)
    expect(data_guide_decision_pending({ ...OPEN, read: "failed" })).toBe(false)
  })

  it("ends when the user leaves Step 4, skips, or a plan lands", () => {
    expect(
      data_guide_decision_pending({ ...OPEN, on_generate_step: false }),
    ).toBe(false)
    expect(data_guide_decision_pending({ ...OPEN, skipped: true })).toBe(false)
    expect(data_guide_decision_pending({ ...OPEN, has_plan: true })).toBe(false)
    expect(data_guide_decision_pending({ ...OPEN, planning: true })).toBe(false)
  })
})

describe("plan_used_data_guide — the proposal line's condition", () => {
  const USED = {
    is_multi_turn: false,
    has_plan: true,
    use_data_guide: true,
    has_guide: true,
  }

  it("is true for a single-turn plan drafted with the guide on", () => {
    expect(plan_used_data_guide(USED)).toBe(true)
  })

  it("is false with no plan on screen", () => {
    expect(plan_used_data_guide({ ...USED, has_plan: false })).toBe(false)
  })

  it("is false when the guide is off", () => {
    expect(plan_used_data_guide({ ...USED, use_data_guide: false })).toBe(false)
  })

  it("is false when the task has no guide, whatever the checkbox says", () => {
    expect(plan_used_data_guide({ ...USED, has_guide: false })).toBe(false)
  })

  it("is false on the multi-turn arm", () => {
    expect(plan_used_data_guide({ ...USED, is_multi_turn: true })).toBe(false)
  })
})
