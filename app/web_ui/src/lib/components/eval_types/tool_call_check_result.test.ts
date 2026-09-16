// @vitest-environment jsdom
import { describe, it, expect } from "vitest"
import { render } from "@testing-library/svelte"
import ToolCallCheckResult from "./tool_call_check_result.svelte"
import type { EvalConfig } from "$lib/types"

function makeConfig(overrides: Record<string, unknown> = {}): EvalConfig {
  return {
    v: 1,
    name: "test",
    config_type: "v2",
    model_type: "eval_config",
    properties: {
      type: "tool_call_check",
      expected_tools: [
        { tool_name: "search" },
        { tool_name: "calculate", expected_args: { x: 1 } },
      ],
      match_mode: "all",
      on_unexpected_tools: "ignore",
      ...overrides,
    },
  } as EvalConfig
}

describe("ToolCallCheckResult", () => {
  it("renders without errors", () => {
    const { container } = render(ToolCallCheckResult)
    expect(container).toBeTruthy()
  })

  it("shows Pass badge when the score is 1.0", () => {
    const { container } = render(ToolCallCheckResult, {
      props: {
        scores: { expected_tools_called: 1.0 },
        eval_config: makeConfig(),
      },
    })
    expect(container.textContent).toContain("Pass")
    expect(container.querySelector(".badge-success")).toBeTruthy()
  })

  it("shows Fail badge when the score is 0.0", () => {
    const { container } = render(ToolCallCheckResult, {
      props: {
        scores: { expected_tools_called: 0.0 },
        eval_config: makeConfig(),
      },
    })
    expect(container.textContent).toContain("Fail")
    expect(container.querySelector(".badge-error")).toBeTruthy()
  })

  it("delegates skipped display to EvalResultScores", () => {
    const { container } = render(ToolCallCheckResult, {
      props: {
        skipped_reason: "no_trace",
        skipped_detail: "No tool calls found",
      },
    })
    expect(container.textContent).toContain("Skipped")
    expect(container.textContent).toContain("No tool calls found")
  })

  it("shows 'any' match mode label", () => {
    const { container } = render(ToolCallCheckResult, {
      props: {
        scores: { expected_tools_called: 1.0 },
        eval_config: makeConfig({ match_mode: "any" }),
      },
    })
    expect(container.textContent).toContain("Any expected tool called")
  })

  it("shows 'all' match mode label", () => {
    const { container } = render(ToolCallCheckResult, {
      props: {
        scores: { expected_tools_called: 1.0 },
        eval_config: makeConfig({ match_mode: "all" }),
      },
    })
    expect(container.textContent).toContain("All expected tools called")
  })

  it("shows 'ordered' match mode label", () => {
    const { container } = render(ToolCallCheckResult, {
      props: {
        scores: { expected_tools_called: 1.0 },
        eval_config: makeConfig({ match_mode: "ordered" }),
      },
    })
    expect(container.textContent).toContain(
      "All expected tools called in order",
    )
  })

  it("shows 'never' match mode label", () => {
    const { container } = render(ToolCallCheckResult, {
      props: {
        scores: { expected_tools_called: 1.0 },
        eval_config: makeConfig({ match_mode: "never" }),
      },
    })
    expect(container.textContent).toContain("None of the listed tools called")
  })

  it("shows tool names from config", () => {
    const { container } = render(ToolCallCheckResult, {
      props: {
        scores: { expected_tools_called: 1.0 },
        eval_config: makeConfig({
          expected_tools: [{ tool_name: "alpha" }, { tool_name: "beta" }],
        }),
      },
    })
    expect(container.textContent).toContain("Tools:")
    expect(container.textContent).toContain("alpha, beta")
  })

  it("shows fail on unexpected tools message", () => {
    const { container } = render(ToolCallCheckResult, {
      props: {
        scores: { expected_tools_called: 1.0 },
        eval_config: makeConfig({ on_unexpected_tools: "fail" }),
      },
    })
    expect(container.textContent).toContain("Fails on unexpected tool calls")
  })

  it("does not show fail on unexpected tools when set to ignore", () => {
    const { container } = render(ToolCallCheckResult, {
      props: {
        scores: { expected_tools_called: 1.0 },
        eval_config: makeConfig({ on_unexpected_tools: "ignore" }),
      },
    })
    expect(container.textContent).not.toContain(
      "Fails on unexpected tool calls",
    )
  })

  it("shows scores via EvalResultScores", () => {
    const { container } = render(ToolCallCheckResult, {
      props: { scores: { expected_tools_called: 1.0 } },
    })
    expect(container.textContent).toContain("expected_tools_called:")
    expect(container.textContent).toContain("1.00")
  })

  it("does not show config details when eval_config is null", () => {
    const { container } = render(ToolCallCheckResult, {
      props: { scores: { expected_tools_called: 1.0 } },
    })
    expect(container.textContent).toContain("expected_tools_called:")
    expect(container.textContent).not.toContain("Tools:")
    expect(container.textContent).not.toContain("All expected tools")
    expect(container.textContent).not.toContain("Any expected tool")
    expect(container.textContent).not.toContain("Fails on unexpected")
  })
})
