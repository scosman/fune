import { describe, it, expect } from "vitest"
import { show_suggested_advisory } from "./model_dropdown_settings"

describe("show_suggested_advisory", () => {
  it("renders in every state once the list is known", () => {
    for (const model_selected of [true, false]) {
      expect(show_suggested_advisory(model_selected, true)).toBe(true)
    }
  })

  it("waits for the model list before judging a chosen model", () => {
    // Until the list lands a chosen model reads as unsuggested whatever it is,
    // so rendering now would flash amber and turn green a moment later.
    expect(show_suggested_advisory(true, false)).toBe(false)
    // No model chosen: nothing to misjudge, the prompt to choose one shows.
    expect(show_suggested_advisory(false, false)).toBe(true)
  })
})
