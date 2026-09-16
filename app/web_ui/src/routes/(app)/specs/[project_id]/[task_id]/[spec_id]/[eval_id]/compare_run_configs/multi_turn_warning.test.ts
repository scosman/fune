import { describe, it, expect } from "vitest"
import { multiTurnStoredScoreWarning } from "./multi_turn_warning"

describe("multiTurnStoredScoreWarning", () => {
  it("warns when the eval set contains stored multi-turn conversations", () => {
    const message = multiTurnStoredScoreWarning(3)
    expect(message).toContain("3 stored multi-turn conversations")
    expect(message).toContain("identical scores")
  })

  it("uses singular copy for a single stored conversation", () => {
    const message = multiTurnStoredScoreWarning(1)
    expect(message).toContain("1 stored multi-turn conversation.")
  })

  it("is hidden for sets without stored conversations", () => {
    // The gate depends only on the item set: a single-turn-only set never
    // warns, whatever the eval's data type or filter shape.
    expect(multiTurnStoredScoreWarning(0)).toBeNull()
  })

  it("is hidden before the score summary has loaded", () => {
    expect(multiTurnStoredScoreWarning(undefined)).toBeNull()
    expect(multiTurnStoredScoreWarning(null)).toBeNull()
  })
})
