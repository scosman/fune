// @vitest-environment jsdom
import { describe, it, expect } from "vitest"
import { render } from "@testing-library/svelte"
import ContainsResult from "./contains_result.svelte"
import type { EvalConfig } from "$lib/types"

function makeConfig(overrides: Record<string, unknown> = {}): EvalConfig {
  return {
    v: 1,
    name: "test",
    config_type: "v2",
    model_type: "eval_config",
    properties: {
      type: "contains",
      substring: "expected text",
      reference_key: null,
      value_expression: null,
      case_sensitive: true,
      mode: "must_contain",
      ...overrides,
    },
  } as EvalConfig
}

describe("ContainsResult", () => {
  it("renders without errors", () => {
    const { container } = render(ContainsResult)
    expect(container).toBeTruthy()
  })

  it("shows Pass badge when the score is 1.0", () => {
    const { container } = render(ContainsResult, {
      props: { scores: { contains_expected: 1.0 }, eval_config: makeConfig() },
    })
    expect(container.textContent).toContain("Pass")
    expect(container.querySelector(".badge-success")).toBeTruthy()
  })

  it("shows Fail badge when the score is 0.0", () => {
    const { container } = render(ContainsResult, {
      props: { scores: { contains_expected: 0.0 }, eval_config: makeConfig() },
    })
    expect(container.textContent).toContain("Fail")
    expect(container.querySelector(".badge-error")).toBeTruthy()
  })

  it("delegates skipped display to EvalResultScores", () => {
    const { container } = render(ContainsResult, {
      props: {
        skipped_reason: "missing_reference",
        skipped_detail: "No value found",
      },
    })
    expect(container.textContent).toContain("Skipped")
    expect(container.textContent).toContain("No value found")
  })

  it("shows must_contain mode label", () => {
    const { container } = render(ContainsResult, {
      props: {
        scores: { contains_expected: 1.0 },
        eval_config: makeConfig({ mode: "must_contain" }),
      },
    })
    expect(container.textContent).toContain("Must contain")
  })

  it("shows must_not_contain mode label", () => {
    const { container } = render(ContainsResult, {
      props: {
        scores: { contains_expected: 1.0 },
        eval_config: makeConfig({ mode: "must_not_contain" }),
      },
    })
    expect(container.textContent).toContain("Must not contain")
  })

  it("shows substring from config", () => {
    const { container } = render(ContainsResult, {
      props: {
        scores: { contains_expected: 1.0 },
        eval_config: makeConfig({ substring: "foo bar" }),
      },
    })
    expect(container.textContent).toContain("Substring:")
    expect(container.textContent).toContain("foo bar")
  })

  it("shows reference_key when no substring", () => {
    const { container } = render(ContainsResult, {
      props: {
        scores: { contains_expected: 1.0 },
        eval_config: makeConfig({
          substring: null,
          reference_key: "answer",
        }),
      },
    })
    expect(container.textContent).toContain("Reference key:")
    expect(container.textContent).toContain("answer")
  })

  it("shows case insensitive label", () => {
    const { container } = render(ContainsResult, {
      props: {
        scores: { contains_expected: 1.0 },
        eval_config: makeConfig({ case_sensitive: false }),
      },
    })
    expect(container.textContent).toContain("Case insensitive")
  })

  it("shows value_expression from config", () => {
    const { container } = render(ContainsResult, {
      props: {
        scores: { contains_expected: 1.0 },
        eval_config: makeConfig({ value_expression: "$.data" }),
      },
    })
    expect(container.textContent).toContain("Expression:")
    expect(container.textContent).toContain("$.data")
  })

  it("shows scores via EvalResultScores", () => {
    const { container } = render(ContainsResult, {
      props: { scores: { contains_expected: 1.0 } },
    })
    expect(container.textContent).toContain("contains_expected:")
    expect(container.textContent).toContain("1.00")
  })

  it("does not show config details when eval_config is null", () => {
    const { container } = render(ContainsResult, {
      props: { scores: { contains_expected: 1.0 } },
    })
    expect(container.textContent).toContain("contains_expected:")
    expect(container.textContent).not.toContain("Substring:")
    expect(container.textContent).not.toContain("Mode:")
    expect(container.textContent).not.toContain("Reference key:")
    expect(container.textContent).not.toContain("Expression:")
  })
})
