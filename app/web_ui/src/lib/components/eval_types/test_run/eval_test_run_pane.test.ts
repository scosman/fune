// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from "vitest"
import { render, fireEvent, cleanup } from "@testing-library/svelte"
import { tick } from "svelte"

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

// ManualExampleDialog now wraps the shared AddExampleDialog. That dialog fetches
// the task schema in onMount, which never runs under the SSR-style render used
// here — so stub it. The stub emits the same `submit` event, letting us test
// the wrapper's prop wiring and submit→confirm conversion.
vi.mock("$lib/components/add_example_dialog.svelte", async () => {
  const Stub = await import("../__tests__/add_example_dialog_stub.svelte")
  return { default: Stub.default }
})

// Dynamic imports after mocks
const EvalTestRunPane = (await import("./eval_test_run_pane.svelte")).default
const TestRunInputCard = (await import("./test_run_input_card.svelte")).default
const TestRunBrowseDialog = (await import("./test_run_browse_dialog.svelte"))
  .default
const ManualExampleDialog = (await import("./manual_example_dialog.svelte"))
  .default
const ReferenceDataField = (await import("./reference_data_field.svelte"))
  .default
const { actionButtonsByTitle, resetActionButtons } = await import(
  "../__tests__/dialog_stub.svelte"
)
const { ALL_V2_EVAL_TYPES } = await import("$lib/utils/eval_types/registry")

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

function makeKilnError(msg: string) {
  return { getMessage: () => msg }
}

// Drive the stubbed AddExampleDialog: fill its input/output fields and click
// its Add button, which dispatches the shared `submit` event.
async function fillManualExample(
  container: HTMLElement,
  input: string,
  output: string,
) {
  const inputField = container.querySelector(
    '[data-testid="stub-input"]',
  ) as HTMLTextAreaElement
  const outputField = container.querySelector(
    '[data-testid="stub-output"]',
  ) as HTMLTextAreaElement
  await fireEvent.input(inputField, { target: { value: input } })
  await fireEvent.input(outputField, { target: { value: output } })
  await tick()
}

async function clickAdd(container: HTMLElement) {
  const addBtn = container.querySelector(
    '[data-testid="stub-add"]',
  ) as HTMLButtonElement
  expect(addBtn).not.toBeNull()
  await fireEvent.click(addBtn)
}

const run1 = makeRun("r1", "input one", "output one")
const run2 = makeRun("r2", "input two", "output two")
const run3 = makeRun("r3", "input three", "output three")
const run4 = makeRun("r4", "input four", "output four")

// ---------------------------------------------------------------------------
// Tests: EvalTestRunPane
// ---------------------------------------------------------------------------

describe("EvalTestRunPane", () => {
  afterEach(() => {
    cleanup()
  })

  describe("State 1: Empty dataset", () => {
    it("renders empty state when no runs available", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: { available_runs: [], runs_loading: false },
      })
      const emptyState = container.querySelector('[data-testid="empty-state"]')
      expect(emptyState).not.toBeNull()
      expect(container.textContent).toContain("No sample inputs yet")
      expect(container.textContent).toContain(
        "Run your task to generate inputs",
      )
    })

    it("shows Go to Run link", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: { available_runs: [], runs_loading: false },
      })
      const goToRunLink = container.querySelector("a.btn")
      expect(goToRunLink).not.toBeNull()
      expect(goToRunLink?.textContent?.trim()).toContain("Go to Run")
    })

    it("does NOT show Save Without Testing button", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: { available_runs: [], runs_loading: false },
      })
      const saveBtn = container.querySelector(
        '[data-testid="save-without-testing"]',
      )
      expect(saveBtn).toBeNull()
      expect(container.textContent).not.toContain("Save Without Testing")
    })
  })

  describe("State 2: Ready (pick input)", () => {
    it("renders selected run card without quick-picks", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1, run2, run3, run4],
          selected_run: run1,
          runs_loading: false,
        },
      })

      const selectedCard = container.querySelector(
        '[data-testid="selected-run-card"]',
      )
      expect(selectedCard).not.toBeNull()
      expect(selectedCard?.textContent).toContain("input one")

      const quickPicks = container.querySelectorAll(
        '[data-testid="quick-pick-card"]',
      )
      expect(quickPicks.length).toBe(0)
    })

    it("does NOT show Browse all dataset inputs link", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
        },
      })

      const browseLink = container.querySelector(
        '[data-testid="browse-all-link"]',
      )
      expect(browseLink).toBeNull()
    })

    it("shows Run button with btn-primary btn-outline style", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
        },
      })

      const runBtn = container.querySelector(
        '[data-testid="run-test-btn"]',
      ) as HTMLButtonElement
      expect(runBtn).not.toBeNull()
      expect(runBtn?.textContent?.trim()).toContain("Run")
      expect(runBtn?.classList.contains("btn-primary")).toBe(true)
      expect(runBtn?.classList.contains("btn-outline")).toBe(true)
    })

    it("does NOT show results placeholder", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
        },
      })

      const placeholder = container.querySelector(
        '[data-testid="results-placeholder"]',
      )
      expect(placeholder).toBeNull()
      expect(container.textContent).not.toContain("Run to see scores")
    })

    it("dispatches run event on Run button click", async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container, component } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
        },
      })
      const handler = vi.fn()
      component.$on("run", handler)

      const runBtn = container.querySelector(
        '[data-testid="run-test-btn"]',
      ) as HTMLButtonElement
      await fireEvent.click(runBtn)
      expect(handler).toHaveBeenCalled()
    })

    it("does not show reference data field for a judge that never reads one", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          judge_reference_signals: {
            prompt_template: "Rate {{ final_message }} for quality.",
            server_reference_keys: [],
            prompt_unavailable: false,
          },
        },
      })

      const refField = container.querySelector(
        '[data-testid="reference-data-field"]',
      )
      expect(refField).toBeNull()
    })

    it("shows reference data field for a judge whose prompt reads one", () => {
      // Not behind SHOW_REFERENCE_DATA_UI: this pane is the only place a reference
      // answer can be typed, and a reference-answer judge tested without one is
      // tested against a prompt missing the block the saved judge renders.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          judge_reference_signals: {
            prompt_template:
              "Grade {{ final_message }} against {{ reference_data.reference_answer }}",
            server_reference_keys: [],
            prompt_unavailable: false,
          },
        },
      })

      const refField = container.querySelector(
        '[data-testid="reference-data-field"]',
      )
      expect(refField).not.toBeNull()
    })

    it("names the key the saved judge will require", () => {
      // check_reference_key wants `reference_answer` exactly; without the hint the
      // tester learns that only by burning a run on a missing_reference_key skip.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          judge_reference_signals: {
            prompt_template:
              "Grade {{ final_message }} against {{ reference_data.reference_answer }}",
            server_reference_keys: ["reference_answer"],
            prompt_unavailable: false,
          },
        },
      })

      const required = container.querySelector(
        '[data-testid="ref-data-required-keys"]',
      )
      expect(required).not.toBeNull()
      expect(required?.textContent).toContain("reference_answer")
      // Already named as required; not repeated as a merely-read key.
      expect(
        container.querySelector('[data-testid="ref-data-prompt-keys"]'),
      ).toBeNull()
    })

    it("shows the input for a judge the server requires a key of, whatever the prompt says", () => {
      // The user can edit the reference block out; the server keeps requiring the key,
      // so every test run would skip with nowhere to supply one.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          judge_reference_signals: {
            prompt_template: "Rate {{ final_message }} for quality.",
            server_reference_keys: ["reference_answer"],
            prompt_unavailable: false,
          },
        },
      })

      expect(
        container.querySelector('[data-testid="reference-data-field"]'),
      ).not.toBeNull()
      expect(
        container.querySelector('[data-testid="ref-data-required-keys"]')
          ?.textContent,
      ).toContain("reference_answer")
    })

    it("shows the input when the default prompt could not be fetched", () => {
      // Nothing is known and the save path bakes the server default either way, so
      // fail open rather than leaving a judge the pane cannot test.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          judge_reference_signals: {
            prompt_template: "",
            server_reference_keys: [],
            prompt_unavailable: true,
          },
        },
      })

      expect(
        container.querySelector('[data-testid="reference-data-field"]'),
      ).not.toBeNull()
    })

    it("names a prompt-only key separately from a required one", () => {
      // A `.get()` lookup renders around a missing value instead of skipping, so it
      // must not be promised as required.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          judge_reference_signals: {
            prompt_template: "{{ reference_data.get('tone') }}",
            server_reference_keys: [],
            prompt_unavailable: false,
          },
        },
      })

      expect(
        container.querySelector('[data-testid="ref-data-required-keys"]'),
      ).toBeNull()
      expect(
        container.querySelector('[data-testid="ref-data-prompt-keys"]')
          ?.textContent,
      ).toContain("tone")
    })

    it("selected card shows Change button that opens browse dialog", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1, run2],
          selected_run: run1,
          runs_loading: false,
        },
      })

      const selectedCard = container.querySelector(
        '[data-testid="selected-run-card"]',
      )
      expect(selectedCard).not.toBeNull()
      const changeBtn = selectedCard?.querySelector("button")
      expect(changeBtn).not.toBeNull()
      expect(changeBtn?.textContent?.trim()).toBe("Change")
    })
  })

  describe("State 3: Running", () => {
    it("renders running state with spinner", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          test_loading: true,
          runs_loading: false,
        },
      })

      const runningState = container.querySelector(
        '[data-testid="running-state"]',
      )
      expect(runningState).not.toBeNull()
      expect(container.textContent).toContain("Running...")
      expect(container.textContent).toContain("Executing the scorer")
    })

    it("shows Cancel button", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          test_loading: true,
          runs_loading: false,
        },
      })

      const cancelBtn = container.querySelector('[data-testid="cancel-run"]')
      expect(cancelBtn).not.toBeNull()
    })

    it("dispatches cancel event on Cancel click", async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container, component } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          test_loading: true,
          runs_loading: false,
        },
      })
      const handler = vi.fn()
      component.$on("cancel", handler)

      const cancelBtn = container.querySelector(
        '[data-testid="cancel-run"]',
      ) as HTMLButtonElement
      await fireEvent.click(cancelBtn)
      expect(handler).toHaveBeenCalled()
    })

    it("shows selected run with Change disabled", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          test_loading: true,
          runs_loading: false,
        },
      })

      const selectedCard = container.querySelector(
        '[data-testid="selected-run-card"]',
      )
      expect(selectedCard).not.toBeNull()

      const changeBtn = selectedCard?.querySelector(
        "button",
      ) as HTMLButtonElement
      expect(changeBtn?.disabled).toBe(true)
    })
  })

  describe("State 4: Results", () => {
    it("renders scores when test result has scores", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          test_result: {
            scores: { accuracy: 0.95, helpfulness: 4.0 },
            skipped_reason: null,
            skipped_detail: null,
          },
          test_has_valid_run: true,
        },
      })

      const scoresSection = container.querySelector(
        '[data-testid="scores-section"]',
      )
      expect(scoresSection).not.toBeNull()
      expect(container.textContent).toContain("accuracy")
      expect(container.textContent).toContain("0.95")
      expect(container.textContent).toContain("helpfulness")
      expect(container.textContent).toContain("4")
      expect(container.textContent).toContain("Scores")
      expect(container.textContent).toContain("Preview only")
      expect(container.textContent).toContain("not saved")
    })

    it("lists the tool calls a code judge made", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          test_result: {
            scores: { accuracy: 1.0 },
            skipped_reason: null,
            skipped_detail: null,
            tool_call_log: [
              {
                tool_name: "llm_judge",
                arguments: { prompt: "hi" },
                output_preview: '{"accuracy": 1.0}',
                is_error: false,
                duration_ms: 412,
              },
            ],
          },
          test_has_valid_run: true,
        },
      })

      const log = container.querySelector('[data-testid="tool-call-log"]')
      expect(log).not.toBeNull()
      expect(log?.textContent).toContain("llm_judge")
      expect(log?.textContent).toContain("412ms")
      expect(log?.textContent).toContain("OK")
    })

    it("surfaces why a tool call failed, not just that it did", () => {
      // output_preview carries the error message; dropping it leaves an author
      // staring at a red row with no reason.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          test_result: {
            scores: { accuracy: 0.0 },
            skipped_reason: null,
            skipped_detail: null,
            tool_call_log: [
              {
                tool_name: "llm_judge",
                arguments: { prompt: "hi" },
                output_preview: "Invalid model provider: bogus",
                is_error: true,
                duration_ms: 412,
              },
            ],
          },
          test_has_valid_run: true,
        },
      })

      const log = container.querySelector('[data-testid="tool-call-log"]')
      expect(log?.textContent).toContain("Invalid model provider: bogus")
      expect(log?.textContent).toContain("Error")
    })

    it("hides the tool call section when the scorer called nothing", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          test_result: {
            scores: { accuracy: 1.0 },
            skipped_reason: null,
            skipped_detail: null,
            tool_call_log: [],
          },
          test_has_valid_run: true,
        },
      })

      expect(
        container.querySelector('[data-testid="tool-call-log"]'),
      ).toBeNull()
    })

    it("shows skipped result with reason", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          test_result: {
            scores: null,
            skipped_reason: "no_reference_data",
            skipped_detail: "No reference data available",
          },
        },
      })

      const skippedResult = container.querySelector(
        '[data-testid="skipped-result"]',
      )
      expect(skippedResult).not.toBeNull()
      expect(container.textContent).toContain("No reference data available")
    })

    it("shows shape warning when present", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          test_result: {
            scores: { wrong_key: 1.0 },
            skipped_reason: null,
          },
          test_shape_warning: "Missing expected scores: accuracy",
        },
      })

      const shapeWarning = container.querySelector(
        '[data-testid="shape-warning"]',
      )
      expect(shapeWarning).not.toBeNull()
      expect(container.textContent).toContain("Score Shape Mismatch")
      expect(container.textContent).toContain("Missing expected scores")
    })

    it("shows Run again button with btn-primary btn-outline style and no Save button", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          test_result: {
            scores: { accuracy: 1.0 },
            skipped_reason: null,
          },
          test_has_valid_run: true,
        },
      })

      const runAgainBtn = container.querySelector(
        '[data-testid="run-again"]',
      ) as HTMLButtonElement
      expect(runAgainBtn).not.toBeNull()
      expect(runAgainBtn?.classList.contains("btn-primary")).toBe(true)
      expect(runAgainBtn?.classList.contains("btn-outline")).toBe(true)

      const saveBtn = container.querySelector('[data-testid="save-eval"]')
      expect(saveBtn).toBeNull()
    })

    it("shows score-range warning when test_score_range_warning is set", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          test_result: {
            scores: { accuracy: 5.0 },
            skipped_reason: null,
          },
          test_score_range_warning:
            "Score accuracy is a pass_fail rating and must be a float between 0.0 and 1.0 inclusive. Got: 5.0",
          test_has_valid_run: false,
        },
      })

      const rangeWarning = container.querySelector(
        '[data-testid="score-range-warning"]',
      )
      expect(rangeWarning).not.toBeNull()
      expect(rangeWarning?.textContent).toContain("Score Out of Range")
      expect(rangeWarning?.textContent).toContain("pass_fail")
      expect(rangeWarning?.textContent).toContain("5.0")
    })

    it("does not show score-range warning when prop is null", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          test_result: {
            scores: { accuracy: 0.5 },
            skipped_reason: null,
          },
          test_score_range_warning: null,
          test_has_valid_run: true,
        },
      })

      const rangeWarning = container.querySelector(
        '[data-testid="score-range-warning"]',
      )
      expect(rangeWarning).toBeNull()
    })

    it("dispatches runAgain event on Run again click", async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container, component } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          test_result: {
            scores: { accuracy: 1.0 },
            skipped_reason: null,
          },
          test_has_valid_run: true,
        },
      })
      const handler = vi.fn()
      component.$on("runAgain", handler)

      const runAgainBtn = container.querySelector(
        '[data-testid="run-again"]',
      ) as HTMLButtonElement
      await fireEvent.click(runAgainBtn)
      expect(handler).toHaveBeenCalled()
    })
  })

  describe("Loading and error states", () => {
    it("renders loading spinner when runs_loading", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: { runs_loading: true },
      })

      const loading = container.querySelector('[data-testid="runs-loading"]')
      expect(loading).not.toBeNull()
      expect(container.textContent).toContain("Loading task runs")
    })

    it("renders error when runs_error set", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          runs_loading: false,
          runs_error: makeKilnError("Failed to load"),
        },
      })

      const error = container.querySelector('[data-testid="runs-error"]')
      expect(error).not.toBeNull()
      expect(container.textContent).toContain("Failed to load")
    })
  })

  describe("Test Run heading and subtitle", () => {
    it("renders Test Run heading", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: { available_runs: [], runs_loading: false },
      })

      const heading = container.querySelector(".text-xl.font-bold")
      expect(heading).not.toBeNull()
      expect(heading?.textContent).toContain("Test Judge")
    })

    it("renders updated subtitle text", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: { available_runs: [], runs_loading: false },
      })

      const subtitle = container.querySelector(
        '[data-testid="test-run-subtitle"]',
      )
      expect(subtitle).not.toBeNull()
      expect(subtitle?.textContent).toBe(
        "Test your judge on real data before saving.",
      )
    })

    it("subtitle uses standard secondary text style (text-sm text-gray-500)", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: { available_runs: [], runs_loading: false },
      })

      const subtitle = container.querySelector(
        '[data-testid="test-run-subtitle"]',
      )
      expect(subtitle).not.toBeNull()
      expect(subtitle?.classList.contains("text-sm")).toBe(true)
      expect(subtitle?.classList.contains("text-gray-500")).toBe(true)
      expect(subtitle?.classList.contains("text-xs")).toBe(false)
    })
  })
})

// ---------------------------------------------------------------------------
// Tests: TestRunInputCard
// ---------------------------------------------------------------------------

describe("TestRunInputCard", () => {
  afterEach(() => {
    cleanup()
  })

  it("renders selected variant with 'Selected Test Run' label in non-grey", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(TestRunInputCard as any, {
      props: {
        run: run1,
        variant: "selected",
        disabled: false,
      },
    })

    const card = container.querySelector('[data-testid="selected-run-card"]')
    expect(card).not.toBeNull()
    expect(card?.textContent).toContain("Selected Test Run")
    expect(card?.textContent).not.toContain("Selected Run\n")

    const label = card?.querySelector("span.font-medium")
    expect(label).not.toBeNull()
    expect(label?.classList.contains("text-base")).toBe(true)
    expect(label?.classList.contains("text-gray-500")).toBe(false)
  })

  it("renders input and output via the clamped viewer", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(TestRunInputCard as any, {
      props: {
        run: run1,
        variant: "selected",
        disabled: false,
      },
    })

    const card = container.querySelector('[data-testid="selected-run-card"]')
    expect(card).not.toBeNull()
    expect(card?.textContent).toContain("input one")
    expect(card?.textContent).toContain("output one")

    // ClampedText renders content in <pre> elements (clamping is visual/CSS).
    const clamped = card?.querySelectorAll("pre")
    expect(clamped?.length).toBeGreaterThanOrEqual(2)
  })

  it("renders pick variant as clickable", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(TestRunInputCard as any, {
      props: {
        run: run2,
        variant: "pick",
      },
    })

    const card = container.querySelector('[data-testid="quick-pick-card"]')
    expect(card).not.toBeNull()
    expect(card?.textContent).toContain("input two")
    expect(card?.textContent).toContain("output two")
  })

  it("dispatches select event when clicking pick card", async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container, component } = render(TestRunInputCard as any, {
      props: {
        run: run2,
        variant: "pick",
      },
    })
    const handler = vi.fn()
    component.$on("select", handler)

    const card = container.querySelector(
      '[data-testid="quick-pick-card"]',
    ) as HTMLButtonElement
    await fireEvent.click(card)
    expect(handler).toHaveBeenCalled()
  })

  it("Change button is disabled when disabled prop is true", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(TestRunInputCard as any, {
      props: {
        run: run1,
        variant: "selected",
        disabled: true,
      },
    })

    const changeBtn = container.querySelector("button") as HTMLButtonElement
    expect(changeBtn?.disabled).toBe(true)
  })

  it("Change button is enabled when disabled prop is false", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(TestRunInputCard as any, {
      props: {
        run: run1,
        variant: "selected",
        disabled: false,
      },
    })

    const changeBtn = container.querySelector("button") as HTMLButtonElement
    expect(changeBtn?.disabled).toBe(false)
  })

  it("keeps full input and output in the DOM without title tooltips", () => {
    const longInput = "a".repeat(200)
    const longOutput = "b".repeat(200)
    const longRun = makeRun("long", longInput, longOutput)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(TestRunInputCard as any, {
      props: {
        run: longRun,
        variant: "selected",
        disabled: false,
      },
    })

    // Full content stays in the DOM (ClampedText clamps visually, See all opens it).
    expect(container.textContent).toContain(longInput)
    expect(container.textContent).toContain(longOutput)
    // No raw title-attribute tooltips anymore.
    expect(container.querySelectorAll("p[title]").length).toBe(0)
  })

  it("dispatches change event when Change clicked in selected variant", async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container, component } = render(TestRunInputCard as any, {
      props: {
        run: run1,
        variant: "selected",
        disabled: false,
      },
    })
    const handler = vi.fn()
    component.$on("change", handler)

    const changeBtn = container.querySelector("button") as HTMLButtonElement
    await fireEvent.click(changeBtn)
    expect(handler).toHaveBeenCalledTimes(1)
  })

  it("pick variant passes run in select event detail", async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container, component } = render(TestRunInputCard as any, {
      props: {
        run: run3,
        variant: "pick",
      },
    })
    const handler = vi.fn()
    component.$on("select", handler)

    const card = container.querySelector(
      '[data-testid="quick-pick-card"]',
    ) as HTMLButtonElement
    await fireEvent.click(card)
    expect(handler.mock.calls[0][0].detail).toBe(run3)
  })

  it("pick variant renders as button element", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(TestRunInputCard as any, {
      props: {
        run: run2,
        variant: "pick",
      },
    })

    const card = container.querySelector('[data-testid="quick-pick-card"]')
    expect(card?.tagName).toBe("BUTTON")
  })
})

// ---------------------------------------------------------------------------
// Tests: TestRunBrowseDialog
// ---------------------------------------------------------------------------

describe("TestRunBrowseDialog", () => {
  afterEach(() => {
    cleanup()
  })

  it("renders wide dialog with correct title", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(TestRunBrowseDialog as any, {
      props: {
        available_runs: [run1, run2],
      },
    })

    const dialog = container.querySelector(
      '[data-title="Choose Dataset Sample"]',
    )
    expect(dialog).not.toBeNull()
    expect(dialog?.getAttribute("data-width")).toBe("wide")
  })

  it("renders table with Input, Output, Created At columns (no radio column)", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(TestRunBrowseDialog as any, {
      props: {
        available_runs: [run1, run2],
      },
    })

    const table = container.querySelector("table")
    expect(table).not.toBeNull()
    const headerTexts = Array.from(table?.querySelectorAll("th") || []).map(
      (h) => h.textContent?.trim(),
    )
    expect(headerTexts).toContain("Input")
    expect(headerTexts).toContain("Output")
    expect(headerTexts).toContain("Created At")
    expect(container.querySelectorAll('input[type="radio"]').length).toBe(0)
  })

  it("does not render a search field", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(TestRunBrowseDialog as any, {
      props: {
        available_runs: [run1],
      },
    })

    const searchInput = container.querySelector('input[type="search"]')
    expect(searchInput).toBeNull()
    const searchPlaceholder = container.querySelector(
      'input[placeholder*="Search"]',
    )
    expect(searchPlaceholder).toBeNull()
  })

  it("shows an 'Add Manual Example' link", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(TestRunBrowseDialog as any, {
      props: {
        available_runs: [run1],
      },
    })

    const addBtn = Array.from(container.querySelectorAll("button")).find((b) =>
      b.textContent?.trim().includes("Add Manual Example"),
    )
    expect(addBtn).toBeDefined()
  })

  it("renders rows with a Select button and no radio buttons", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(TestRunBrowseDialog as any, {
      props: {
        available_runs: [run1, run2],
      },
    })

    const radios = container.querySelectorAll('input[type="radio"]')
    expect(radios.length).toBe(0)

    const rows = container.querySelectorAll("tbody tr")
    expect(rows.length).toBe(2)
    const selectButtons = Array.from(
      container.querySelectorAll("button"),
    ).filter((b) => b.textContent?.trim() === "Select")
    expect(selectButtons.length).toBe(2)
  })

  it("dispatches select and closes dialog on Select click", async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container, component } = render(TestRunBrowseDialog as any, {
      props: {
        available_runs: [run1, run2],
      },
    })
    const handler = vi.fn()
    component.$on("select", handler)

    const selectButtons = Array.from(
      container.querySelectorAll("button"),
    ).filter((b) => b.textContent?.trim() === "Select")
    await fireEvent.click(selectButtons[1])
    expect(handler).toHaveBeenCalledTimes(1)
    expect(handler.mock.calls[0][0].detail).toBe(run2)
  })

  it("paginates with PAGE_SIZE of 5", () => {
    const manyRuns = Array.from({ length: 8 }, (_, i) =>
      makeRun(`r${i}`, `input ${i}`, `output ${i}`),
    )
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(TestRunBrowseDialog as any, {
      props: {
        available_runs: manyRuns,
      },
    })

    const rows = container.querySelectorAll("tbody tr")
    expect(rows.length).toBe(5)

    const nextBtn = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent?.trim() === "Next",
    )
    expect(nextBtn).not.toBeNull()
  })

  it("renders full cell text via ClampedText (clamping is visual, not truncated)", () => {
    const longRun = makeRun("long", "x".repeat(200), "y".repeat(200))
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(TestRunBrowseDialog as any, {
      props: {
        available_runs: [longRun],
      },
    })

    // ClampedText clamps with CSS, so the full content stays in the DOM.
    expect(container.textContent).toContain("x".repeat(200))
    expect(container.textContent).toContain("y".repeat(200))
  })

  it("shows subtitle text", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(TestRunBrowseDialog as any, {
      props: {
        available_runs: [run1],
      },
    })

    const dialogStub = container.querySelector('[data-testid="dialog-stub"]')
    expect(dialogStub).not.toBeNull()
  })

  it("handle_manual_confirm dispatches ephemeral run via confirm event", async () => {
    const { container, component } = render(ManualExampleDialog as any, {
      props: { project_id: "p1", task_id: "t1" },
    })
    const handler = vi.fn()
    component.$on("confirm", handler)

    await fillManualExample(container, "manual input", "manual output")
    await clickAdd(container)

    expect(handler).toHaveBeenCalledTimes(1)
    const detail = handler.mock.calls[0][0].detail
    expect(detail.v).toBe(1)
    expect(detail.input).toBe("manual input")
    expect(detail.output.output).toBe("manual output")
    expect(detail.output.source.type).toBe("human")
    expect(detail.output.source.properties).toEqual({})
    expect(detail.tags).toEqual([])
    expect(detail.model_type).toBe("manual")
    expect(detail.id).toBeUndefined()
    expect(detail.path).toBeUndefined()
  })
})

// ---------------------------------------------------------------------------
// Tests: ManualExampleDialog — ephemeral flow
// ---------------------------------------------------------------------------

describe("ManualExampleDialog ephemeral flow", () => {
  afterEach(() => {
    cleanup()
  })

  it("ephemeral run constructed by ManualExampleDialog has no id or path", async () => {
    const { container, component } = render(ManualExampleDialog as any, {
      props: { project_id: "p1", task_id: "t1" },
    })
    const handler = vi.fn()
    component.$on("confirm", handler)

    await fillManualExample(container, "ephemeral input", "ephemeral output")
    await clickAdd(container)

    expect(handler).toHaveBeenCalledTimes(1)
    const detail = handler.mock.calls[0][0].detail
    expect(detail.id).toBeUndefined()
    expect(detail.path).toBeUndefined()
    expect(detail.v).toBe(1)
    expect(detail.tags).toEqual([])
  })

  it("does not dispatch a confirm when both fields are empty", async () => {
    const { container, component } = render(ManualExampleDialog as any, {
      props: { project_id: "p1", task_id: "t1" },
    })
    const handler = vi.fn()
    component.$on("confirm", handler)

    await fillManualExample(container, "   ", "")
    await clickAdd(container)

    expect(handler).not.toHaveBeenCalled()
  })

  it("configures the shared dialog for manual-only entry", () => {
    const { container } = render(ManualExampleDialog as any, {
      props: { project_id: "p1", task_id: "t1" },
    })

    const stub = container.querySelector('[data-testid="add-example-stub"]')
    expect(stub).not.toBeNull()
    expect(stub?.getAttribute("data-title")).toBe("Add Manual Example")
    expect(stub?.getAttribute("data-sub-subtitle")).toContain("won't be saved")
    expect(stub?.getAttribute("data-manual-only")).toBe("true")
    expect(stub?.getAttribute("data-project-id")).toBe("p1")
    expect(stub?.getAttribute("data-task-id")).toBe("t1")
    expect(stub?.getAttribute("data-submit-label")).toBe("Use Example")
    expect(stub?.getAttribute("data-disable-when-empty")).toBe("true")
  })
})

// ---------------------------------------------------------------------------
// Tests: ManualExampleDialog
// ---------------------------------------------------------------------------

describe("ManualExampleDialog", () => {
  afterEach(() => {
    cleanup()
  })

  it("renders the shared add-example dialog", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ManualExampleDialog as any, {
      props: { project_id: "p1", task_id: "t1" },
    })

    expect(
      container.querySelector('[data-testid="add-example-stub"]'),
    ).not.toBeNull()
  })

  it("captures both input and output sides", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ManualExampleDialog as any, {
      props: { project_id: "p1", task_id: "t1" },
    })

    const stub = container.querySelector('[data-testid="add-example-stub"]')
    expect(stub?.getAttribute("data-include-input")).toBe("true")
    expect(stub?.getAttribute("data-include-output")).toBe("true")
  })

  it("passes manual-only, title and ephemeral note to the dialog", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ManualExampleDialog as any, {
      props: { project_id: "p1", task_id: "t1" },
    })

    const stub = container.querySelector('[data-testid="add-example-stub"]')
    expect(stub?.getAttribute("data-manual-only")).toBe("true")
    expect(stub?.getAttribute("data-title")).toBe("Add Manual Example")
    expect(stub?.getAttribute("data-sub-subtitle")).toContain("won't be saved")
  })

  it("forwards a submitted example as a confirm event", async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container, component } = render(ManualExampleDialog as any, {
      props: { project_id: "p1", task_id: "t1" },
    })
    const handler = vi.fn()
    component.$on("confirm", handler)

    await fillManualExample(container, "test input", "test output")
    await clickAdd(container)

    expect(handler).toHaveBeenCalledTimes(1)
    const detail = handler.mock.calls[0][0].detail
    expect(detail.input).toBe("test input")
    expect(detail.output.output).toBe("test output")
  })
})

// ---------------------------------------------------------------------------
// Tests: ReferenceDataField
// ---------------------------------------------------------------------------

describe("ReferenceDataField", () => {
  afterEach(() => {
    cleanup()
  })

  describe("display value (preview text)", () => {
    it("displays 'None' when reference_data is empty", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(ReferenceDataField as any, {
        props: { reference_data: "" },
      })

      const field = container.querySelector(
        '[data-testid="reference-data-field"]',
      )
      expect(field?.textContent).toContain("None")
      expect(field?.textContent).toContain("Reference Data")
    })

    it("displays key summary when valid JSON is provided", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(ReferenceDataField as any, {
        props: { reference_data: '{"expected": "hello", "key2": "world"}' },
      })

      const field = container.querySelector(
        '[data-testid="reference-data-field"]',
      )
      expect(field?.textContent).toContain("expected")
      expect(field?.textContent).toContain("key2")
    })

    it("displays 'Invalid JSON' for malformed reference_data", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(ReferenceDataField as any, {
        props: { reference_data: "not json" },
      })

      const field = container.querySelector(
        '[data-testid="reference-data-field"]',
      )
      expect(field?.textContent).toContain("Invalid JSON")
    })

    it("displays keys with '+N more' for many keys", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(ReferenceDataField as any, {
        props: {
          reference_data: '{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}',
        },
      })

      const field = container.querySelector(
        '[data-testid="reference-data-field"]',
      )
      expect(field?.textContent).toContain("+2 more")
    })

    it("displays 'Empty object' for empty JSON object", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(ReferenceDataField as any, {
        props: { reference_data: "{}" },
      })

      const field = container.querySelector(
        '[data-testid="reference-data-field"]',
      )
      expect(field?.textContent).toContain("Empty object")
    })

    it("displays 'Invalid: not an object' for non-object JSON values in preview", () => {
      const nonObjects = ['"asdf"', "false", "12", "[1,2]", "null"]
      for (const val of nonObjects) {
        cleanup()
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const { container } = render(ReferenceDataField as any, {
          props: { reference_data: val },
        })
        const field = container.querySelector(
          '[data-testid="reference-data-field"]',
        )
        expect(field?.textContent).toContain("Invalid: not an object")
      }
    })
  })

  describe("dialog structure", () => {
    it("opens a wide dialog with key-value editor on click", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(ReferenceDataField as any, {
        props: { reference_data: "" },
      })

      const editBtn = container.querySelector(
        '[data-testid="reference-data-edit"]',
      ) as HTMLButtonElement
      expect(editBtn).not.toBeNull()

      const dialog = container.querySelector('[data-title="Reference Data"]')
      expect(dialog).not.toBeNull()
      expect(dialog?.getAttribute("data-width")).toBe("wide")
    })

    it("shows key-value editor (not a textarea)", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(ReferenceDataField as any, {
        props: { reference_data: "" },
      })

      const editor = container.querySelector(
        '[data-testid="reference-data-editor"]',
      )
      expect(editor).not.toBeNull()

      const textarea = container.querySelector(
        '[data-testid="reference-data-textarea"]',
      )
      expect(textarea).toBeNull()
    })

    it("shows Add Value button", () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(ReferenceDataField as any, {
        props: { reference_data: "" },
      })

      const addBtn = container.querySelector(
        '[data-testid="reference-data-add"]',
      )
      expect(addBtn).not.toBeNull()
      expect(addBtn?.textContent).toContain("Add Value")
    })
  })

  describe("row management", () => {
    it("starts with one empty row when reference_data is empty", async () => {
      resetActionButtons()
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(ReferenceDataField as any, {
        props: { reference_data: "" },
      })

      const editBtn = container.querySelector(
        '[data-testid="reference-data-edit"]',
      ) as HTMLButtonElement
      await fireEvent.click(editBtn)
      await tick()

      const rows = container.querySelectorAll(
        '[data-testid="reference-data-row"]',
      )
      expect(rows.length).toBe(1)
    })

    it("populates rows from existing JSON data", async () => {
      resetActionButtons()
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(ReferenceDataField as any, {
        props: {
          reference_data: '{"name": "Alice", "age": 30}',
        },
      })

      const editBtn = container.querySelector(
        '[data-testid="reference-data-edit"]',
      ) as HTMLButtonElement
      await fireEvent.click(editBtn)
      await tick()

      const rows = container.querySelectorAll(
        '[data-testid="reference-data-row"]',
      )
      expect(rows.length).toBe(2)

      const keys = container.querySelectorAll(
        '[data-testid="reference-data-key"]',
      ) as NodeListOf<HTMLInputElement>
      const values = container.querySelectorAll(
        '[data-testid="reference-data-value"]',
      ) as NodeListOf<HTMLInputElement>
      expect(keys[0].value).toBe("name")
      expect(values[0].value).toBe('"Alice"')
      expect(keys[1].value).toBe("age")
      expect(values[1].value).toBe("30")
    })

    it("adds a new row when Add Value is clicked", async () => {
      resetActionButtons()
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(ReferenceDataField as any, {
        props: { reference_data: "" },
      })

      const editBtn = container.querySelector(
        '[data-testid="reference-data-edit"]',
      ) as HTMLButtonElement
      await fireEvent.click(editBtn)
      await tick()

      const addBtn = container.querySelector(
        '[data-testid="reference-data-add"]',
      ) as HTMLButtonElement
      await fireEvent.click(addBtn)
      await tick()

      const rows = container.querySelectorAll(
        '[data-testid="reference-data-row"]',
      )
      expect(rows.length).toBe(2)
    })

    it("removes a row when X is clicked", async () => {
      resetActionButtons()
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(ReferenceDataField as any, {
        props: {
          reference_data: '{"a": 1, "b": 2}',
        },
      })

      const editBtn = container.querySelector(
        '[data-testid="reference-data-edit"]',
      ) as HTMLButtonElement
      await fireEvent.click(editBtn)
      await tick()

      let rows = container.querySelectorAll(
        '[data-testid="reference-data-row"]',
      )
      expect(rows.length).toBe(2)

      const removeBtn = container.querySelector(
        '[data-testid="reference-data-remove"]',
      ) as HTMLButtonElement
      await fireEvent.click(removeBtn)
      await tick()

      rows = container.querySelectorAll('[data-testid="reference-data-row"]')
      expect(rows.length).toBe(1)
    })

    it("shows Name and Value column headers in a separate header row", async () => {
      resetActionButtons()
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(ReferenceDataField as any, {
        props: {
          reference_data: '{"a": 1, "b": 2}',
        },
      })

      const editBtn = container.querySelector(
        '[data-testid="reference-data-edit"]',
      ) as HTMLButtonElement
      await fireEvent.click(editBtn)
      await tick()

      const headerRow = container.querySelector(
        '[data-testid="reference-data-column-headers"]',
      )
      expect(headerRow).not.toBeNull()

      const labels = container.querySelectorAll(".label-text")
      const labelTexts = Array.from(labels).map((l) => l.textContent?.trim())
      expect(labelTexts).toContain("Name")
      expect(labelTexts).toContain("Value")
      expect(labels.length).toBe(2)
    })
  })

  describe("save validation", () => {
    async function renderAndGetSaveAction(reference_data = "") {
      resetActionButtons()
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const result = render(ReferenceDataField as any, {
        props: { reference_data },
      })
      const handler = vi.fn()
      result.component.$on("change", handler)

      const editBtn = result.container.querySelector(
        '[data-testid="reference-data-edit"]',
      ) as HTMLButtonElement
      await fireEvent.click(editBtn)
      await tick()

      const buttons = actionButtonsByTitle["Reference Data"]
      const saveBtn = buttons?.find((b) => b.label === "Save") as Record<
        string,
        unknown
      >
      return {
        ...result,
        handler,
        saveAction: saveBtn.action as () => boolean,
      }
    }

    it("saves valid key-value pairs and emits change", async () => {
      const { container, handler, saveAction } = await renderAndGetSaveAction()

      const keys = container.querySelectorAll(
        '[data-testid="reference-data-key"]',
      ) as NodeListOf<HTMLInputElement>
      const values = container.querySelectorAll(
        '[data-testid="reference-data-value"]',
      ) as NodeListOf<HTMLInputElement>

      await fireEvent.input(keys[0], { target: { value: "answer" } })
      await fireEvent.input(values[0], { target: { value: "hello" } })
      await tick()

      const result = saveAction()
      expect(result).toBe(true)
      expect(handler).toHaveBeenCalledTimes(1)
      const emitted = JSON.parse(handler.mock.calls[0][0].detail)
      expect(emitted).toEqual({ answer: "hello" })
    })

    it("emits empty string when all rows are empty", async () => {
      const { handler, saveAction } = await renderAndGetSaveAction()

      const result = saveAction()
      expect(result).toBe(true)
      expect(handler).toHaveBeenCalledTimes(1)
      expect(handler.mock.calls[0][0].detail).toBe("")
    })

    it("rejects rows with value but no name", async () => {
      const { container, handler, saveAction } = await renderAndGetSaveAction()

      const values = container.querySelectorAll(
        '[data-testid="reference-data-value"]',
      ) as NodeListOf<HTMLInputElement>
      await fireEvent.input(values[0], { target: { value: "some value" } })
      await tick()

      const result = saveAction()
      expect(result).toBe(false)
      expect(handler).not.toHaveBeenCalled()
      await tick()
      const error = container.querySelector(
        '[data-testid="reference-data-error"]',
      )
      expect(error?.textContent).toContain("must have a name")
    })

    it("rejects duplicate keys", async () => {
      const { container, handler, saveAction } =
        await renderAndGetSaveAction('{"a": 1}')

      // Add a second row
      const addBtn = container.querySelector(
        '[data-testid="reference-data-add"]',
      ) as HTMLButtonElement
      await fireEvent.click(addBtn)
      await tick()

      const keys = container.querySelectorAll(
        '[data-testid="reference-data-key"]',
      ) as NodeListOf<HTMLInputElement>
      const values = container.querySelectorAll(
        '[data-testid="reference-data-value"]',
      ) as NodeListOf<HTMLInputElement>
      await fireEvent.input(keys[1], { target: { value: "a" } })
      await fireEvent.input(values[1], { target: { value: "2" } })
      await tick()

      const result = saveAction()
      expect(result).toBe(false)
      expect(handler).not.toHaveBeenCalled()
      await tick()
      const error = container.querySelector(
        '[data-testid="reference-data-error"]',
      )
      expect(error?.textContent).toContain("Duplicate")
    })
  })

  describe("value parsing", () => {
    async function renderAndGetSaveAction() {
      resetActionButtons()
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const result = render(ReferenceDataField as any, {
        props: { reference_data: "" },
      })
      const handler = vi.fn()
      result.component.$on("change", handler)

      const editBtn = result.container.querySelector(
        '[data-testid="reference-data-edit"]',
      ) as HTMLButtonElement
      await fireEvent.click(editBtn)
      await tick()

      const buttons = actionButtonsByTitle["Reference Data"]
      const saveBtn = buttons?.find((b) => b.label === "Save") as Record<
        string,
        unknown
      >
      return {
        ...result,
        handler,
        saveAction: saveBtn.action as () => boolean,
      }
    }

    it("parses number values as numbers", async () => {
      const { container, handler, saveAction } = await renderAndGetSaveAction()
      const keys = container.querySelectorAll(
        '[data-testid="reference-data-key"]',
      ) as NodeListOf<HTMLInputElement>
      const values = container.querySelectorAll(
        '[data-testid="reference-data-value"]',
      ) as NodeListOf<HTMLInputElement>

      await fireEvent.input(keys[0], { target: { value: "count" } })
      await fireEvent.input(values[0], { target: { value: "12" } })
      await tick()

      saveAction()
      const emitted = JSON.parse(handler.mock.calls[0][0].detail)
      expect(emitted.count).toBe(12)
      expect(typeof emitted.count).toBe("number")
    })

    it("parses boolean true as boolean", async () => {
      const { container, handler, saveAction } = await renderAndGetSaveAction()
      const keys = container.querySelectorAll(
        '[data-testid="reference-data-key"]',
      ) as NodeListOf<HTMLInputElement>
      const values = container.querySelectorAll(
        '[data-testid="reference-data-value"]',
      ) as NodeListOf<HTMLInputElement>

      await fireEvent.input(keys[0], { target: { value: "flag" } })
      await fireEvent.input(values[0], { target: { value: "true" } })
      await tick()

      saveAction()
      const emitted = JSON.parse(handler.mock.calls[0][0].detail)
      expect(emitted.flag).toBe(true)
      expect(typeof emitted.flag).toBe("boolean")
    })

    it("parses boolean false as boolean", async () => {
      const { container, handler, saveAction } = await renderAndGetSaveAction()
      const keys = container.querySelectorAll(
        '[data-testid="reference-data-key"]',
      ) as NodeListOf<HTMLInputElement>
      const values = container.querySelectorAll(
        '[data-testid="reference-data-value"]',
      ) as NodeListOf<HTMLInputElement>

      await fireEvent.input(keys[0], { target: { value: "disabled" } })
      await fireEvent.input(values[0], { target: { value: "false" } })
      await tick()

      saveAction()
      const emitted = JSON.parse(handler.mock.calls[0][0].detail)
      expect(emitted.disabled).toBe(false)
      expect(typeof emitted.disabled).toBe("boolean")
    })

    it('parses quoted strings as strings (e.g. "12" becomes string "12")', async () => {
      const { container, handler, saveAction } = await renderAndGetSaveAction()
      const keys = container.querySelectorAll(
        '[data-testid="reference-data-key"]',
      ) as NodeListOf<HTMLInputElement>
      const values = container.querySelectorAll(
        '[data-testid="reference-data-value"]',
      ) as NodeListOf<HTMLInputElement>

      await fireEvent.input(keys[0], { target: { value: "code" } })
      await fireEvent.input(values[0], { target: { value: '"12"' } })
      await tick()

      saveAction()
      const emitted = JSON.parse(handler.mock.calls[0][0].detail)
      expect(emitted.code).toBe("12")
      expect(typeof emitted.code).toBe("string")
    })

    it("falls back to string for unparseable JSON values", async () => {
      const { container, handler, saveAction } = await renderAndGetSaveAction()
      const keys = container.querySelectorAll(
        '[data-testid="reference-data-key"]',
      ) as NodeListOf<HTMLInputElement>
      const values = container.querySelectorAll(
        '[data-testid="reference-data-value"]',
      ) as NodeListOf<HTMLInputElement>

      await fireEvent.input(keys[0], { target: { value: "text" } })
      await fireEvent.input(values[0], { target: { value: "hello world" } })
      await tick()

      saveAction()
      const emitted = JSON.parse(handler.mock.calls[0][0].detail)
      expect(emitted.text).toBe("hello world")
      expect(typeof emitted.text).toBe("string")
    })

    it("falls back to string for broken JSON (e.g. unclosed quote)", async () => {
      const { container, handler, saveAction } = await renderAndGetSaveAction()
      const keys = container.querySelectorAll(
        '[data-testid="reference-data-key"]',
      ) as NodeListOf<HTMLInputElement>
      const values = container.querySelectorAll(
        '[data-testid="reference-data-value"]',
      ) as NodeListOf<HTMLInputElement>

      await fireEvent.input(keys[0], { target: { value: "bad" } })
      await fireEvent.input(values[0], { target: { value: '"12' } })
      await tick()

      saveAction()
      const emitted = JSON.parse(handler.mock.calls[0][0].detail)
      expect(emitted.bad).toBe('"12')
      expect(typeof emitted.bad).toBe("string")
    })

    it("parses nested objects as objects", async () => {
      const { container, handler, saveAction } = await renderAndGetSaveAction()
      const keys = container.querySelectorAll(
        '[data-testid="reference-data-key"]',
      ) as NodeListOf<HTMLInputElement>
      const values = container.querySelectorAll(
        '[data-testid="reference-data-value"]',
      ) as NodeListOf<HTMLInputElement>

      await fireEvent.input(keys[0], { target: { value: "nested" } })
      await fireEvent.input(values[0], {
        target: { value: '{"a": 1}' },
      })
      await tick()

      saveAction()
      const emitted = JSON.parse(handler.mock.calls[0][0].detail)
      expect(emitted.nested).toEqual({ a: 1 })
    })

    it("parses null as null", async () => {
      const { container, handler, saveAction } = await renderAndGetSaveAction()
      const keys = container.querySelectorAll(
        '[data-testid="reference-data-key"]',
      ) as NodeListOf<HTMLInputElement>
      const values = container.querySelectorAll(
        '[data-testid="reference-data-value"]',
      ) as NodeListOf<HTMLInputElement>

      await fireEvent.input(keys[0], { target: { value: "empty" } })
      await fireEvent.input(values[0], { target: { value: "null" } })
      await tick()

      saveAction()
      const emitted = JSON.parse(handler.mock.calls[0][0].detail)
      expect(emitted.empty).toBeNull()
    })

    it("saves empty value as empty string when key is present", async () => {
      const { container, handler, saveAction } = await renderAndGetSaveAction()
      const keys = container.querySelectorAll(
        '[data-testid="reference-data-key"]',
      ) as NodeListOf<HTMLInputElement>

      await fireEvent.input(keys[0], { target: { value: "score" } })
      await tick()

      const result = saveAction()
      expect(result).toBe(true)
      expect(handler).toHaveBeenCalledTimes(1)
      const emitted = JSON.parse(handler.mock.calls[0][0].detail)
      expect(emitted.score).toBe("")
      expect(typeof emitted.score).toBe("string")
    })
  })
})

// ---------------------------------------------------------------------------
// Tests: Auto-select first run
// ---------------------------------------------------------------------------

describe("Auto-select integration", () => {
  afterEach(() => {
    cleanup()
  })

  it("pane renders ready state when selected_run is provided", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(EvalTestRunPane as any, {
      props: {
        available_runs: [run1, run2],
        selected_run: run1,
        runs_loading: false,
      },
    })

    const selectedCard = container.querySelector(
      '[data-testid="selected-run-card"]',
    )
    expect(selectedCard).not.toBeNull()
    expect(selectedCard?.textContent).toContain("input one")
  })

  it("shows fallback when has runs but no selected_run", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(EvalTestRunPane as any, {
      props: {
        available_runs: [run1],
        selected_run: null,
        runs_loading: false,
      },
    })

    expect(container.textContent).toContain("Select a run to get started")
  })

  it("does not show quick-picks when only 1 run", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(EvalTestRunPane as any, {
      props: {
        available_runs: [run1],
        selected_run: run1,
        runs_loading: false,
      },
    })

    const quickPicks = container.querySelectorAll(
      '[data-testid="quick-pick-card"]',
    )
    expect(quickPicks.length).toBe(0)

    const browseLink = container.querySelector(
      '[data-testid="browse-all-link"]',
    )
    expect(browseLink).toBeNull()
  })

  it("Go to Run link uses flat /run route", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(EvalTestRunPane as any, {
      props: {
        available_runs: [],
        runs_loading: false,
      },
    })

    const link = container.querySelector("a.btn") as HTMLAnchorElement
    expect(link?.getAttribute("href")).toBe("/run")
  })
})

// ---------------------------------------------------------------------------
// Tests: ReferenceDataField "Missing x" red-state
// ---------------------------------------------------------------------------

describe("ReferenceDataField missing-field red-state", () => {
  afterEach(() => {
    cleanup()
  })

  it("shows 'Missing x' when required field is not in reference data", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: {
        reference_data: "",
        required_reference_fields: ["expected_answer"],
      },
    })
    const editBtn = container.querySelector(
      '[data-testid="reference-data-edit"]',
    )
    expect(editBtn?.textContent).toContain("Missing expected_answer")
  })

  it("applies text-error class when field is missing", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: {
        reference_data: "",
        required_reference_fields: ["expected_answer"],
      },
    })
    const editBtn = container.querySelector(
      '[data-testid="reference-data-edit"]',
    )
    expect(editBtn?.classList.contains("text-error")).toBe(true)
    expect(editBtn?.classList.contains("text-gray-500")).toBe(false)
  })

  it("does not show missing state when required field is present", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: {
        reference_data: '{"expected_answer": "yes"}',
        required_reference_fields: ["expected_answer"],
      },
    })
    const editBtn = container.querySelector(
      '[data-testid="reference-data-edit"]',
    )
    expect(editBtn?.textContent).not.toContain("Missing")
    expect(editBtn?.classList.contains("text-error")).toBe(false)
    expect(editBtn?.classList.contains("text-gray-500")).toBe(true)
  })

  it("shows normal display when no required fields", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: {
        reference_data: "",
        required_reference_fields: [],
      },
    })
    const editBtn = container.querySelector(
      '[data-testid="reference-data-edit"]',
    )
    expect(editBtn?.textContent).toContain("None")
    expect(editBtn?.classList.contains("text-error")).toBe(false)
  })

  it("shows missing for field with null value in reference data", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: {
        reference_data: '{"expected_answer": null}',
        required_reference_fields: ["expected_answer"],
      },
    })
    const editBtn = container.querySelector(
      '[data-testid="reference-data-edit"]',
    )
    expect(editBtn?.textContent).toContain("Missing expected_answer")
    expect(editBtn?.classList.contains("text-error")).toBe(true)
  })

  it("shows missing for field with empty string value", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: {
        reference_data: '{"expected_answer": ""}',
        required_reference_fields: ["expected_answer"],
      },
    })
    const editBtn = container.querySelector(
      '[data-testid="reference-data-edit"]',
    )
    expect(editBtn?.textContent).toContain("Missing expected_answer")
  })

  it("comma-joins multiple missing fields", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: {
        reference_data: "",
        required_reference_fields: ["field_a", "field_b"],
      },
    })
    const editBtn = container.querySelector(
      '[data-testid="reference-data-edit"]',
    )
    expect(editBtn?.textContent).toContain("Missing field_a, field_b")
  })

  it("shows missing when reference data is invalid JSON", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: {
        reference_data: "not json",
        required_reference_fields: ["expected_answer"],
      },
    })
    const editBtn = container.querySelector(
      '[data-testid="reference-data-edit"]',
    )
    expect(editBtn?.textContent).toContain("Missing expected_answer")
    expect(editBtn?.classList.contains("text-error")).toBe(true)
  })

  it("shows normal display when field is present with truthy value", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: {
        reference_data: '{"expected_answer": "correct"}',
        required_reference_fields: ["expected_answer"],
      },
    })
    const editBtn = container.querySelector(
      '[data-testid="reference-data-edit"]',
    )
    expect(editBtn?.textContent).toContain("expected_answer")
    expect(editBtn?.textContent).not.toContain("Missing")
    expect(editBtn?.classList.contains("text-gray-500")).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// Tests: ReferenceDataField "How it works" callout per usage mode
// ---------------------------------------------------------------------------

describe("ReferenceDataField callout per usage mode", () => {
  afterEach(() => {
    cleanup()
  })

  it("renders llm_judge callout with Jinja example snippet", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: { reference_data: "", usage_mode: "llm_judge" },
    })
    const callout = container.querySelector('[data-testid="ref-data-callout"]')
    expect(callout).not.toBeNull()
    expect(callout?.textContent).toContain("expected values (ground truth)")
    expect(callout?.textContent).toContain("{{ reference_data.expected_type }}")
  })

  it("renders reference_field callout with From reference data wording", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: { reference_data: "", usage_mode: "reference_field" },
    })
    const callout = container.querySelector('[data-testid="ref-data-callout"]')
    expect(callout).not.toBeNull()
    expect(callout?.textContent).toContain("expected values (ground truth)")
    expect(callout?.textContent).toContain("From reference data")
    expect(callout?.textContent).toContain("select the field to compare")
  })

  it("renders code callout with reference_data argument and example", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: { reference_data: "", usage_mode: "code" },
    })
    const callout = container.querySelector('[data-testid="ref-data-callout"]')
    expect(callout).not.toBeNull()
    expect(callout?.textContent).toContain("expected values (ground truth)")
    expect(callout?.textContent).toContain("reference_data")
    expect(callout?.textContent).toContain(".get(")
  })

  it("renders optional callout pointing at the Output to Check expression", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: { reference_data: "", usage_mode: "optional" },
    })
    const callout = container.querySelector('[data-testid="ref-data-callout"]')
    expect(callout).not.toBeNull()
    expect(callout?.textContent).toContain("expected values (ground truth)")
    expect(callout?.textContent).toContain("Output to Check")
    expect(callout?.textContent).toContain("{{ reference_data.expected_type }}")
  })

  it("uses the shared CalloutCard component (blue style)", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: { reference_data: "", usage_mode: "llm_judge" },
    })
    const callout = container.querySelector('[data-testid="ref-data-callout"]')
    expect(callout).not.toBeNull()
    expect(callout?.classList.contains("border-primary/30")).toBe(true)
    expect(callout?.classList.contains("bg-primary/5")).toBe(true)
  })

  it("shows the JSON format note under Value column header", async () => {
    resetActionButtons()
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: {
        reference_data: '{"a": 1}',
        usage_mode: "llm_judge",
      },
    })

    const editBtn = container.querySelector(
      '[data-testid="reference-data-edit"]',
    ) as HTMLButtonElement
    await fireEvent.click(editBtn)
    await tick()

    const note = container.querySelector('[data-testid="json-format-note"]')
    expect(note).not.toBeNull()
    expect(note?.textContent).toContain("Any valid JSON")
    expect(note?.textContent).not.toContain("Values can be")
  })

  it("dialog has no subtitle (callout replaces it)", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { container } = render(ReferenceDataField as any, {
      props: { reference_data: "", usage_mode: "llm_judge" },
    })
    const dialogStub = container.querySelector('[data-title="Reference Data"]')
    expect(dialogStub).not.toBeNull()
    expect(dialogStub?.getAttribute("data-subtitle")).toBe("")
  })
})

// ---------------------------------------------------------------------------
// Tests: EvalTestRunPane hides reference data for every eval type while
// SHOW_REFERENCE_DATA_UI is off (the shipped default). The flag-on behavior is
// covered in eval_test_run_pane.reference_data.test.ts.
// ---------------------------------------------------------------------------

describe("EvalTestRunPane reference data visibility by eval type", () => {
  afterEach(() => {
    cleanup()
  })

  it("hides reference data field for every eval type in ready state", () => {
    for (const eval_config_type of ALL_V2_EVAL_TYPES) {
      cleanup()
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          eval_config_type,
        },
      })
      const refField = container.querySelector(
        '[data-testid="reference-data-field"]',
      )
      expect(refField).toBeNull()
    }
  })

  it("hides reference data field for every eval type in results state", () => {
    for (const eval_config_type of ALL_V2_EVAL_TYPES) {
      cleanup()
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { container } = render(EvalTestRunPane as any, {
        props: {
          available_runs: [run1],
          selected_run: run1,
          runs_loading: false,
          eval_config_type,
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
      expect(refField).toBeNull()
    }
  })
})
