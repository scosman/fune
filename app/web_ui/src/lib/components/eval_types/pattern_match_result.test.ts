// @vitest-environment jsdom
import { describe, it, expect } from "vitest"
import { render } from "@testing-library/svelte"
import PatternMatchResult from "./pattern_match_result.svelte"
import type { EvalConfig } from "$lib/types"

function makeConfig(overrides: Record<string, unknown> = {}): EvalConfig {
  return {
    v: 1,
    name: "test",
    config_type: "v2",
    model_type: "eval_config",
    properties: {
      type: "pattern_match",
      pattern: "\\d+",
      mode: "must_match",
      value_expression: null,
      ...overrides,
    },
  } as EvalConfig
}

describe("PatternMatchResult", () => {
  it("renders without errors", () => {
    const { container } = render(PatternMatchResult)
    expect(container).toBeTruthy()
  })

  it("shows Pass badge when the score is 1.0", () => {
    const { container } = render(PatternMatchResult, {
      props: { scores: { matches_pattern: 1.0 }, eval_config: makeConfig() },
    })
    expect(container.textContent).toContain("Pass")
    expect(container.querySelector(".badge-success")).toBeTruthy()
  })

  it("shows Fail badge when the score is 0.0", () => {
    const { container } = render(PatternMatchResult, {
      props: { scores: { matches_pattern: 0.0 }, eval_config: makeConfig() },
    })
    expect(container.textContent).toContain("Fail")
    expect(container.querySelector(".badge-error")).toBeTruthy()
  })

  it("delegates skipped display to EvalResultScores", () => {
    const { container } = render(PatternMatchResult, {
      props: {
        skipped_reason: "error",
        skipped_detail: "Regex failed",
      },
    })
    expect(container.textContent).toContain("Skipped")
    expect(container.textContent).toContain("Regex failed")
  })

  it("does not show pass/fail badge when skipped", () => {
    const { container } = render(PatternMatchResult, {
      props: {
        scores: { matches_pattern: 0.0 },
        skipped_reason: "error",
      },
    })
    expect(container.querySelector(".badge-success")).toBeFalsy()
    expect(container.querySelector(".badge-error")).toBeFalsy()
  })

  it("shows pattern from config", () => {
    const { container } = render(PatternMatchResult, {
      props: {
        scores: { matches_pattern: 1.0 },
        eval_config: makeConfig({ pattern: "^hello$" }),
      },
    })
    expect(container.textContent).toContain("Pattern:")
    expect(container.textContent).toContain("^hello$")
  })

  it("shows must_match mode label", () => {
    const { container } = render(PatternMatchResult, {
      props: {
        scores: { matches_pattern: 1.0 },
        eval_config: makeConfig({ mode: "must_match" }),
      },
    })
    expect(container.textContent).toContain("Must match")
  })

  it("shows must_not_match mode label", () => {
    const { container } = render(PatternMatchResult, {
      props: {
        scores: { matches_pattern: 1.0 },
        eval_config: makeConfig({ mode: "must_not_match" }),
      },
    })
    expect(container.textContent).toContain("Must not match")
  })

  it("shows value_expression from config", () => {
    const { container } = render(PatternMatchResult, {
      props: {
        scores: { matches_pattern: 1.0 },
        eval_config: makeConfig({ value_expression: "$.output" }),
      },
    })
    expect(container.textContent).toContain("Expression:")
    expect(container.textContent).toContain("$.output")
  })

  it("shows scores via EvalResultScores", () => {
    const { container } = render(PatternMatchResult, {
      props: { scores: { matches_pattern: 0.0 } },
    })
    expect(container.textContent).toContain("matches_pattern:")
    expect(container.textContent).toContain("0.00")
  })

  it("does not show config details when eval_config is null", () => {
    const { container } = render(PatternMatchResult, {
      props: { scores: { matches_pattern: 1.0 } },
    })
    expect(container.textContent).toContain("matches_pattern:")
    expect(container.textContent).not.toContain("Pattern:")
    expect(container.textContent).not.toContain("Mode:")
    expect(container.textContent).not.toContain("Expression:")
  })
})
