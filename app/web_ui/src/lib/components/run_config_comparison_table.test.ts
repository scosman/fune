// @vitest-environment jsdom
import { describe, it, expect, vi } from "vitest"
import { render } from "@testing-library/svelte"
import type { Eval, Task, TaskRunConfig, EvalResultSummary } from "$lib/types"

vi.mock("$lib/ui/run_config_component/run_config_summary.svelte", async () => {
  const { default: Stub } = await import(
    "$lib/components/eval_types/__tests__/dialog_stub.svelte"
  )
  return { default: Stub }
})

vi.mock("$lib/components/output_type_table_preview.svelte", async () => {
  const { default: Stub } = await import(
    "$lib/components/eval_types/__tests__/dialog_stub.svelte"
  )
  return { default: Stub }
})

vi.mock("$lib/components/run_eval.svelte", async () => {
  const { default: Stub } = await import(
    "$lib/components/eval_types/__tests__/dialog_stub.svelte"
  )
  return { default: Stub }
})

vi.mock("$lib/ui/warning.svelte", async () => {
  const { default: Stub } = await import(
    "$lib/components/eval_types/__tests__/dialog_stub.svelte"
  )
  return { default: Stub }
})

const RunConfigComparisonTable = (
  await import("./run_config_comparison_table.svelte")
).default

function makeEval(overrides: Partial<Eval> = {}): Eval {
  return {
    v: 1,
    name: "Test Eval",
    model_type: "eval",
    output_scores: [{ name: "quality" }],
    ...overrides,
  } as Eval
}

function makeRunConfig(id: string, name: string = "Config"): TaskRunConfig {
  return {
    id,
    v: 1,
    name,
    model_type: "task_run_config",
    run_config_properties: {
      model_name: "gpt-4o",
      model_provider_name: "openai",
      prompt_id: "simple_prompt_builder",
    },
  } as TaskRunConfig
}

function makeSummary(
  rc_id: string,
  n_excluded: number,
  n_used: number,
  percent_complete: number = 1.0,
): EvalResultSummary {
  return {
    results: {
      [rc_id]: {
        quality: {
          mean_score: n_used > 0 ? 0.8 : null,
          n_used,
          n_excluded,
        },
      },
    },
    run_config_percent_complete: {
      [rc_id]: percent_complete,
    },
    dataset_size: n_used + n_excluded,
    multi_turn_item_count: 0,
  }
}

function baseProps(summary: EvalResultSummary | null = null) {
  return {
    project_id: "proj1",
    task_id: "task1",
    eval_id: "eval1",
    evaluator: makeEval(),
    task: { id: "task1", default_run_config_id: null } as unknown as Task,
    sorted_task_run_configs: [makeRunConfig("rc1", "Config A")],
    score_summary: summary,
    current_eval_config_id: "ec1",
    interactive: false,
    on_eval_complete: null,
    title: "Compare Run Configurations",
    subtitle: "Subtitle",
    eval_state: "complete" as const,
    on_add_run_config: null,
    current_eval_config_name: null,
    score_summary_error: null,
  }
}

describe("RunConfigComparisonTable n_excluded indicator", () => {
  it("shows info indicator when n_excluded > 0", () => {
    const summary = makeSummary("rc1", 3, 10)
    const { container } = render(RunConfigComparisonTable, {
      props: baseProps(summary),
    })
    const tooltipButtons = container.querySelectorAll(
      ".text-warning button, .text-error button",
    )
    expect(tooltipButtons.length).toBeGreaterThan(0)
  })

  it("hides info indicator when n_excluded == 0", () => {
    const summary = makeSummary("rc1", 0, 10)
    const { container } = render(RunConfigComparisonTable, {
      props: baseProps(summary),
    })
    const tooltipButtons = container.querySelectorAll(
      ".text-warning button, .text-error button",
    )
    expect(tooltipButtons.length).toBe(0)
  })

  it("shows text-warning when ratio <= 0.2", () => {
    // 2 excluded out of 10 total = 20% ratio -> yellow (not > 0.2)
    const summary = makeSummary("rc1", 2, 8)
    const { container } = render(RunConfigComparisonTable, {
      props: baseProps(summary),
    })
    const warningSpan = container.querySelector(".text-warning")
    expect(warningSpan).toBeTruthy()
    // No text-error indicator (percent_complete is 1.0 so no other .text-error)
    const errorIndicators = container.querySelectorAll(".text-error button")
    expect(errorIndicators.length).toBe(0)
  })

  it("shows text-error when ratio > 0.2", () => {
    // 3 excluded out of 10 total = 30% ratio -> red
    const summary = makeSummary("rc1", 3, 7)
    const { container } = render(RunConfigComparisonTable, {
      props: baseProps(summary),
    })
    const errorSpan = container.querySelector(".text-error button")
    expect(errorSpan).toBeTruthy()
  })

  it("coexists with incomplete-runs warning", () => {
    // Both: percent_complete < 1.0 AND n_excluded > 0
    const summary = makeSummary("rc1", 3, 7, 0.7)
    const { container } = render(RunConfigComparisonTable, {
      props: baseProps(summary),
    })
    // The incomplete warning should show (percent_complete < 1.0)
    expect(container.textContent).toContain("70% Complete")
    // The n_excluded indicator should also show (30% ratio -> text-error)
    const indicator = container.querySelector(
      ".text-error button, .text-warning button",
    )
    expect(indicator).toBeTruthy()
  })

  it("renders correct tooltip text with counts", () => {
    // 3 excluded, 10 used -> "3 of 13 cases were skipped..."
    const summary = makeSummary("rc1", 3, 10)
    const { container } = render(RunConfigComparisonTable, {
      props: baseProps(summary),
    })
    // The InfoTooltip renders a hidden tooltip div with the text
    expect(container.textContent).toContain(
      "3 of 13 cases were skipped and are not reflected in this score.",
    )
  })
})
