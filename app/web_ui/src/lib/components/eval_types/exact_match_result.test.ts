// @vitest-environment jsdom
import { describe, it, expect } from "vitest"
import { render } from "@testing-library/svelte"
import ExactMatchResult from "./exact_match_result.svelte"
import type { EvalConfig } from "$lib/types"

function makeConfig(overrides: Record<string, unknown> = {}): EvalConfig {
  return {
    v: 1,
    name: "test",
    config_type: "v2",
    model_type: "eval_config",
    properties: {
      type: "exact_match",
      expected_value: "hello",
      reference_key: null,
      value_expression: null,
      case_sensitive: true,
      ...overrides,
    },
  } as EvalConfig
}

describe("ExactMatchResult", () => {
  it("renders without errors", () => {
    const { container } = render(ExactMatchResult)
    expect(container).toBeTruthy()
  })

  it("delegates score display to EvalResultScores with toFixed(2)", () => {
    const { container } = render(ExactMatchResult, {
      props: { scores: { correct: 1 } },
    })
    expect(container.textContent).toContain("correct:")
    expect(container.textContent).toContain("1.00")
  })

  it("delegates skipped display to EvalResultScores", () => {
    const { container } = render(ExactMatchResult, {
      props: {
        skipped_reason: "missing_reference",
        skipped_detail: "No expected value",
      },
    })
    expect(container.textContent).toContain("Skipped")
    expect(container.textContent).toContain("missing reference")
    expect(container.textContent).toContain("No expected value")
  })

  it("shows empty state when no scores or skip reason", () => {
    const { container } = render(ExactMatchResult)
    expect(container.textContent).toContain("No scores available.")
  })

  it("shows Pass badge when the score is 1.0", () => {
    const { container } = render(ExactMatchResult, {
      props: { scores: { correct: 1.0 }, eval_config: makeConfig() },
    })
    expect(container.textContent).toContain("Pass")
    expect(container.querySelector(".badge-success")).toBeTruthy()
  })

  it("shows Fail badge when the score is 0.0", () => {
    const { container } = render(ExactMatchResult, {
      props: { scores: { correct: 0.0 }, eval_config: makeConfig() },
    })
    expect(container.textContent).toContain("Fail")
    expect(container.querySelector(".badge-error")).toBeTruthy()
  })

  it("does not show pass/fail badge when skipped", () => {
    const { container } = render(ExactMatchResult, {
      props: {
        scores: { correct: 0.0 },
        skipped_reason: "missing_reference",
        eval_config: makeConfig(),
      },
    })
    expect(container.querySelector(".badge-success")).toBeFalsy()
    expect(container.querySelector(".badge-error")).toBeFalsy()
  })

  it("shows expected_value from config", () => {
    const { container } = render(ExactMatchResult, {
      props: {
        scores: { correct: 1.0 },
        eval_config: makeConfig({ expected_value: "world" }),
      },
    })
    expect(container.textContent).toContain("Expected:")
    expect(container.textContent).toContain("world")
  })

  it("shows reference_key when no expected_value", () => {
    const { container } = render(ExactMatchResult, {
      props: {
        scores: { correct: 1.0 },
        eval_config: makeConfig({
          expected_value: null,
          reference_key: "answer",
        }),
      },
    })
    expect(container.textContent).toContain("Reference key:")
    expect(container.textContent).toContain("answer")
  })

  it("shows case insensitive label", () => {
    const { container } = render(ExactMatchResult, {
      props: {
        scores: { correct: 1.0 },
        eval_config: makeConfig({ case_sensitive: false }),
      },
    })
    expect(container.textContent).toContain("Case insensitive")
  })

  it("does not show case insensitive label when case_sensitive is true", () => {
    const { container } = render(ExactMatchResult, {
      props: {
        scores: { correct: 1.0 },
        eval_config: makeConfig({ case_sensitive: true }),
      },
    })
    expect(container.textContent).not.toContain("Case insensitive")
  })

  it("shows value_expression from config", () => {
    const { container } = render(ExactMatchResult, {
      props: {
        scores: { correct: 1.0 },
        eval_config: makeConfig({ value_expression: "$.result" }),
      },
    })
    expect(container.textContent).toContain("Expression:")
    expect(container.textContent).toContain("$.result")
  })
})
