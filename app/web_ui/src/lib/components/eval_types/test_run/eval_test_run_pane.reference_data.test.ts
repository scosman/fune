// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from "vitest"
// Pins the behavior restored by flipping SHOW_REFERENCE_DATA_UI back on, so the
// gated code paths don't rot while reference data is hidden from the UI.
vi.mock("$lib/utils/eval_types/reference_data_ui", () => ({
  SHOW_REFERENCE_DATA_UI: true,
}))

import { render, cleanup } from "@testing-library/svelte"

// ---------------------------------------------------------------------------
// Module-level mocks
// ---------------------------------------------------------------------------

vi.mock("$lib/ui/dialog.svelte", async () => {
  const Stub = await import("../__tests__/dialog_stub.svelte")
  return { default: Stub.default }
})

vi.mock("$lib/utils/format_expanded_content", () => ({
  formatExpandedContent: (text: string) => ({ isJson: false, value: text }),
}))

vi.mock("$app/navigation", () => ({
  goto: vi.fn(),
}))

// Dynamic imports after mocks
const EvalTestRunPane = (await import("./eval_test_run_pane.svelte")).default

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeRun(
  id: string,
  input: string,
  output: string,
  created_at?: string,
) {
  return {
    v: 1,
    id,
    input,
    output: { output, source: { type: "human" as const } },
    tags: [],
    created_at: created_at || new Date().toISOString(),
  }
}

const run1 = makeRun("r1", "input one", "output one")

// An llm_judge's mode is decided by its own signals in both flag states, so the
// llm_judge cases below need a prompt that actually reads reference data.
const REFERENCE_SIGNALS = {
  prompt_template:
    "Grade {{ final_message }} against {{ reference_data.reference_answer }}",
  server_reference_keys: [] as string[],
  prompt_unavailable: false,
}

// ---------------------------------------------------------------------------
// Tests: EvalTestRunPane
// ---------------------------------------------------------------------------

describe("EvalTestRunPane", () => {
  afterEach(() => {
    cleanup()
  })

  describe("State 2: Ready (pick input)", () => {
    it("shows reference data field", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          judge_reference_signals: REFERENCE_SIGNALS,
        },
      })

      const refField = container.querySelector(
        '[data-testid="reference-data-field"]',
      )
      expect(refField).not.toBeNull()
    })
  })
})

// ---------------------------------------------------------------------------
// Tests: EvalTestRunPane hides reference data for "none" mode types
// ---------------------------------------------------------------------------

describe("EvalTestRunPane reference data visibility by eval type", () => {
  afterEach(() => {
    cleanup()
  })

  it("shows reference data field for pattern_match (optional mode) in ready state", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(EvalTestRunPane as any, {
      props: {
        available_runs: [run1],
        selected_run: run1,
        runs_loading: false,
        eval_config_type: "pattern_match",
      },
    })
    const refField = container.querySelector(
      '[data-testid="reference-data-field"]',
    )
    expect(refField).not.toBeNull()
  })

  it("hides reference data field for tool_call_check (none mode) in ready state", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(EvalTestRunPane as any, {
      props: {
        available_runs: [run1],
        selected_run: run1,
        runs_loading: false,
        eval_config_type: "tool_call_check",
      },
    })
    const refField = container.querySelector(
      '[data-testid="reference-data-field"]',
    )
    expect(refField).toBeNull()
  })

  it("hides reference data field for step_count_check (none mode) in ready state", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(EvalTestRunPane as any, {
      props: {
        available_runs: [run1],
        selected_run: run1,
        runs_loading: false,
        eval_config_type: "step_count_check",
      },
    })
    const refField = container.querySelector(
      '[data-testid="reference-data-field"]',
    )
    expect(refField).toBeNull()
  })

  it("shows reference data field for llm_judge in ready state", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(EvalTestRunPane as any, {
      props: {
        available_runs: [run1],
        selected_run: run1,
        runs_loading: false,
        eval_config_type: "llm_judge",
        judge_reference_signals: REFERENCE_SIGNALS,
      },
    })
    const refField = container.querySelector(
      '[data-testid="reference-data-field"]',
    )
    expect(refField).not.toBeNull()
  })

  it("shows reference data field for exact_match in ready state", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(EvalTestRunPane as any, {
      props: {
        available_runs: [run1],
        selected_run: run1,
        runs_loading: false,
        eval_config_type: "exact_match",
      },
    })
    const refField = container.querySelector(
      '[data-testid="reference-data-field"]',
    )
    expect(refField).not.toBeNull()
  })

  it("shows reference data field for code_eval in ready state", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(EvalTestRunPane as any, {
      props: {
        available_runs: [run1],
        selected_run: run1,
        runs_loading: false,
        eval_config_type: "code_eval",
      },
    })
    const refField = container.querySelector(
      '[data-testid="reference-data-field"]',
    )
    expect(refField).not.toBeNull()
  })

  it("shows reference data field for pattern_match in results state", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(EvalTestRunPane as any, {
      props: {
        available_runs: [run1],
        selected_run: run1,
        runs_loading: false,
        eval_config_type: "pattern_match",
        test_result: {
          scores: { match: 1.0 },
          skipped_reason: null,
        },
        test_has_valid_run: true,
      },
    })
    const refField = container.querySelector(
      '[data-testid="reference-data-field"]',
    )
    expect(refField).not.toBeNull()
  })

  it("shows reference data field for llm_judge in results state", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(EvalTestRunPane as any, {
      props: {
        available_runs: [run1],
        selected_run: run1,
        runs_loading: false,
        eval_config_type: "llm_judge",
        judge_reference_signals: REFERENCE_SIGNALS,
        test_result: {
          scores: { accuracy: 1.0 },
          skipped_reason: null,
        },
        test_has_valid_run: true,
      },
    })
    const refField = container.querySelector(
      '[data-testid="reference-data-field"]',
    )
    expect(refField).not.toBeNull()
  })
})
