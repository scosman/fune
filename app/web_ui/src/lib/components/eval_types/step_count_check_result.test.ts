// @vitest-environment jsdom
import { describe, it, expect } from "vitest"
import { render } from "@testing-library/svelte"
import StepCountCheckResult from "./step_count_check_result.svelte"
import type { EvalConfig } from "$lib/types"

function makeConfig(overrides: Record<string, unknown> = {}): EvalConfig {
  return {
    v: 1,
    name: "test",
    config_type: "v2",
    model_type: "eval_config",
    properties: {
      type: "step_count_check",
      count_type: "tool_calls",
      min_count: 1,
      max_count: 5,
      ...overrides,
    },
  } as EvalConfig
}

describe("StepCountCheckResult", () => {
  it("renders without errors", () => {
    const { container } = render(StepCountCheckResult)
    expect(container).toBeTruthy()
  })

  it("shows Pass badge when the score is 1.0", () => {
    const { container } = render(StepCountCheckResult, {
      props: { scores: { within_bounds: 1.0 }, eval_config: makeConfig() },
    })
    expect(container.textContent).toContain("Pass")
    expect(container.querySelector(".badge-success")).toBeTruthy()
  })

  it("shows Fail badge when the score is 0.0", () => {
    const { container } = render(StepCountCheckResult, {
      props: { scores: { within_bounds: 0.0 }, eval_config: makeConfig() },
    })
    expect(container.textContent).toContain("Fail")
    expect(container.querySelector(".badge-error")).toBeTruthy()
  })

  it("delegates skipped display to EvalResultScores", () => {
    const { container } = render(StepCountCheckResult, {
      props: {
        skipped_reason: "no_trace",
        skipped_detail: "Missing conversation trace",
      },
    })
    expect(container.textContent).toContain("Skipped")
    expect(container.textContent).toContain("Missing conversation trace")
  })

  it("shows tool_calls count type label", () => {
    const { container } = render(StepCountCheckResult, {
      props: {
        scores: { within_bounds: 1.0 },
        eval_config: makeConfig({ count_type: "tool_calls" }),
      },
    })
    expect(container.textContent).toContain("Tool calls")
  })

  it("shows model_responses count type label", () => {
    const { container } = render(StepCountCheckResult, {
      props: {
        scores: { within_bounds: 1.0 },
        eval_config: makeConfig({ count_type: "model_responses" }),
      },
    })
    expect(container.textContent).toContain("Model responses")
  })

  it("shows turns count type label", () => {
    const { container } = render(StepCountCheckResult, {
      props: {
        scores: { within_bounds: 1.0 },
        eval_config: makeConfig({ count_type: "turns" }),
      },
    })
    expect(container.textContent).toContain("Turns")
  })

  it("shows range when both min and max are set", () => {
    const { container } = render(StepCountCheckResult, {
      props: {
        scores: { within_bounds: 1.0 },
        eval_config: makeConfig({ min_count: 2, max_count: 10 }),
      },
    })
    expect(container.textContent).toContain("2 - 10")
  })

  it("shows 'at least N' when only min is set", () => {
    const { container } = render(StepCountCheckResult, {
      props: {
        scores: { within_bounds: 1.0 },
        eval_config: makeConfig({ min_count: 3, max_count: null }),
      },
    })
    expect(container.textContent).toContain("at least 3")
  })

  it("shows 'at most N' when only max is set", () => {
    const { container } = render(StepCountCheckResult, {
      props: {
        scores: { within_bounds: 1.0 },
        eval_config: makeConfig({ min_count: null, max_count: 7 }),
      },
    })
    expect(container.textContent).toContain("at most 7")
  })

  it("shows 'any' when neither min nor max is set", () => {
    const { container } = render(StepCountCheckResult, {
      props: {
        scores: { within_bounds: 1.0 },
        eval_config: makeConfig({ min_count: null, max_count: null }),
      },
    })
    expect(container.textContent).toContain("any")
  })

  it("shows scores via EvalResultScores", () => {
    const { container } = render(StepCountCheckResult, {
      props: { scores: { within_bounds: 0.0 } },
    })
    expect(container.textContent).toContain("within_bounds:")
    expect(container.textContent).toContain("0.00")
  })

  it("does not show config details when eval_config is null", () => {
    const { container } = render(StepCountCheckResult, {
      props: { scores: { within_bounds: 1.0 } },
    })
    expect(container.textContent).toContain("within_bounds:")
    expect(container.textContent).not.toContain("Counting:")
    expect(container.textContent).not.toContain("Allowed range:")
  })
})
