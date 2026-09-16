// @vitest-environment jsdom
import { describe, it, expect } from "vitest"
import { render } from "@testing-library/svelte"
import SetCheckResult from "./set_check_result.svelte"
import type { EvalConfig } from "$lib/types"

function makeConfig(overrides: Record<string, unknown> = {}): EvalConfig {
  return {
    v: 1,
    name: "test",
    config_type: "v2",
    model_type: "eval_config",
    properties: {
      type: "set_check",
      expected_set: ["a", "b", "c"],
      reference_key: null,
      value_expression: null,
      mode: "equal",
      ...overrides,
    },
  } as EvalConfig
}

describe("SetCheckResult", () => {
  it("renders without errors", () => {
    const { container } = render(SetCheckResult)
    expect(container).toBeTruthy()
  })

  it("shows Pass badge when the score is 1.0", () => {
    const { container } = render(SetCheckResult, {
      props: { scores: { set_matches: 1.0 }, eval_config: makeConfig() },
    })
    expect(container.textContent).toContain("Pass")
    expect(container.querySelector(".badge-success")).toBeTruthy()
  })

  it("shows Fail badge when the score is 0.0", () => {
    const { container } = render(SetCheckResult, {
      props: { scores: { set_matches: 0.0 }, eval_config: makeConfig() },
    })
    expect(container.textContent).toContain("Fail")
    expect(container.querySelector(".badge-error")).toBeTruthy()
  })

  it("delegates skipped display to EvalResultScores", () => {
    const { container } = render(SetCheckResult, {
      props: {
        skipped_reason: "parse_error",
        skipped_detail: "Could not parse output",
      },
    })
    expect(container.textContent).toContain("Skipped")
    expect(container.textContent).toContain("Could not parse output")
  })

  it("shows subset mode label", () => {
    const { container } = render(SetCheckResult, {
      props: {
        scores: { set_matches: 1.0 },
        eval_config: makeConfig({ mode: "subset" }),
      },
    })
    expect(container.textContent).toContain("Output is subset of expected")
  })

  it("shows superset mode label", () => {
    const { container } = render(SetCheckResult, {
      props: {
        scores: { set_matches: 1.0 },
        eval_config: makeConfig({ mode: "superset" }),
      },
    })
    expect(container.textContent).toContain("Output is superset of expected")
  })

  it("shows equal mode label", () => {
    const { container } = render(SetCheckResult, {
      props: {
        scores: { set_matches: 1.0 },
        eval_config: makeConfig({ mode: "equal" }),
      },
    })
    expect(container.textContent).toContain("Output equals expected set")
  })

  it("shows expected_set from config", () => {
    const { container } = render(SetCheckResult, {
      props: {
        scores: { set_matches: 1.0 },
        eval_config: makeConfig({ expected_set: ["x", "y", "z"] }),
      },
    })
    expect(container.textContent).toContain("Expected:")
    expect(container.textContent).toContain("x, y, z")
  })

  it("shows reference_key when no expected_set", () => {
    const { container } = render(SetCheckResult, {
      props: {
        scores: { set_matches: 1.0 },
        eval_config: makeConfig({
          expected_set: null,
          reference_key: "tags",
        }),
      },
    })
    expect(container.textContent).toContain("Reference key:")
    expect(container.textContent).toContain("tags")
  })

  it("shows value_expression from config", () => {
    const { container } = render(SetCheckResult, {
      props: {
        scores: { set_matches: 1.0 },
        eval_config: makeConfig({ value_expression: "$.items" }),
      },
    })
    expect(container.textContent).toContain("Expression:")
    expect(container.textContent).toContain("$.items")
  })

  it("shows scores via EvalResultScores", () => {
    const { container } = render(SetCheckResult, {
      props: { scores: { set_matches: 0.0 } },
    })
    expect(container.textContent).toContain("set_matches:")
    expect(container.textContent).toContain("0.00")
  })

  it("does not show config details when eval_config is null", () => {
    const { container } = render(SetCheckResult, {
      props: { scores: { set_matches: 1.0 } },
    })
    expect(container.textContent).toContain("set_matches:")
    expect(container.textContent).not.toContain("Expected:")
    expect(container.textContent).not.toContain("Reference key:")
    expect(container.textContent).not.toContain("Expression:")
    expect(container.textContent).not.toContain("Output is subset")
    expect(container.textContent).not.toContain("Output is superset")
    expect(container.textContent).not.toContain("Output equals expected")
  })
})
