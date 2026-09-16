import { describe, it, expect, vi } from "vitest"

// Pins the behavior restored by flipping SHOW_REFERENCE_DATA_UI back on, so the
// gated code paths don't rot while reference data is hidden from the UI.
vi.mock("$lib/utils/eval_types/reference_data_ui", () => ({
  SHOW_REFERENCE_DATA_UI: true,
}))

import {
  referenceDataUsageMode,
  NO_JUDGE_PROMPT,
  getV2EvalTypeMetadata,
  ALL_V2_EVAL_TYPES,
  type V2EvalType,
  type ReferenceDataUsageMode,
} from "./registry"

describe("referenceDataUsageMode (reference data shown)", () => {
  const EXPECTED_MODES: Record<V2EvalType, ReferenceDataUsageMode> = {
    llm_judge: "llm_judge",
    exact_match: "reference_field",
    contains: "reference_field",
    set_check: "reference_field",
    code_eval: "code",
    pattern_match: "optional",
    tool_call_check: "none",
    step_count_check: "none",
  }

  it("maps every V2EvalType to the correct mode", () => {
    for (const [type, expectedMode] of Object.entries(EXPECTED_MODES)) {
      // llm_judge resolves from its own signals in both flag states, so it needs some.
      expect(
        referenceDataUsageMode(type as V2EvalType, {
          ...NO_JUDGE_PROMPT,
          prompt_template: "{{ reference_data.reference_answer }}",
        }),
      ).toBe(expectedMode)
    }
  })

  it("still hides the input for an llm_judge that needs no reference data", () => {
    expect(
      referenceDataUsageMode("llm_judge", {
        ...NO_JUDGE_PROMPT,
        prompt_template: "Rate {{ final_message }}",
      }),
    ).toBe("none")
  })

  it("throws for an invalid type via assertNever", () => {
    expect(() =>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      referenceDataUsageMode("invalid_type" as any, NO_JUDGE_PROMPT),
    ).toThrow("Unexpected value")
  })

  it("covers every eval type without throwing", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      expect(() => referenceDataUsageMode(t, NO_JUDGE_PROMPT)).not.toThrow()
    }
  })
})

describe("metadata copy (reference data shown)", () => {
  it("mentions reference data in the types that support it", () => {
    expect(getV2EvalTypeMetadata("exact_match").description).toContain(
      "reference-data value",
    )
    expect(getV2EvalTypeMetadata("exact_match").explainer).toContain(
      "reference data",
    )
    expect(getV2EvalTypeMetadata("contains").description).toContain(
      "reference value",
    )
    expect(getV2EvalTypeMetadata("contains").explainer).toContain(
      "reference data",
    )
    expect(getV2EvalTypeMetadata("code_eval").explainer).toContain(
      "reference data",
    )
  })
})
