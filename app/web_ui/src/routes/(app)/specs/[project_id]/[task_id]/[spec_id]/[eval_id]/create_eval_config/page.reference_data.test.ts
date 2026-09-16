// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
// Pins the behavior restored by flipping SHOW_REFERENCE_DATA_UI back on, so the
// gated code paths don't rot while reference data is hidden from the UI.
vi.mock("$lib/utils/eval_types/reference_data_ui", () => ({
  SHOW_REFERENCE_DATA_UI: true,
}))

import { render, fireEvent, cleanup } from "@testing-library/svelte"
import { tick } from "svelte"
import * as svelteMod from "svelte"

// ---------------------------------------------------------------------------
// Module-level mocks -- must come before the dynamic page import
// ---------------------------------------------------------------------------

const { mockPage, mockGoto, mockClientGET, mockLoadTask, mockLoadModels } =
  vi.hoisted(() => {
    type PageValue = {
      params: Record<string, string>
      url: URL
    }
    let pageValue: PageValue = {
      params: {
        project_id: "proj1",
        task_id: "task1",
        eval_id: "eval1",
        spec_id: "spec1",
      },
      url: new URL(
        "http://localhost/specs/proj1/task1/spec1/eval1/create_eval_config",
      ),
    }
    type Subscriber = (value: PageValue) => void
    const subscribers = new Set<Subscriber>()
    const mockPage = {
      subscribe(fn: Subscriber) {
        subscribers.add(fn)
        fn(pageValue)
        return () => subscribers.delete(fn)
      },
      set(v: PageValue) {
        pageValue = v
        subscribers.forEach((fn) => fn(v))
      },
    }

    const mockGoto = vi.fn()

    const mockClientGET = vi.fn().mockImplementation((path: string) => {
      if (path.includes("/evals/")) {
        return Promise.resolve({
          data: {
            id: "eval1",
            name: "Test Eval",
            eval_set_filter_id: null,
            splits: {
              test: { source: "eval_input", filter_id: "tag::inputs" },
            },
            eval_configs: [],
            eval_config_type: "v2",
            output_scores: [],
          },
          error: null,
        })
      }
      if (path.includes("/specs/")) {
        return Promise.resolve({
          data: { id: "spec1", name: "Test Spec" },
          error: null,
        })
      }
      return Promise.resolve({ data: null, error: null })
    })

    const mockLoadTask = vi.fn().mockResolvedValue({
      id: "task1",
      name: "Test Task",
      description: "desc",
      input_json_schema: "{}",
      output_json_schema: "{}",
    })

    const mockLoadModels = vi.fn().mockResolvedValue([])

    return {
      mockPage,
      mockGoto,
      mockClientGET,
      mockLoadTask,
      mockLoadModels,
    }
  })

const onMountCallbacks: Array<() => unknown> = []

vi.mock("$app/stores", () => ({
  page: mockPage,
}))

vi.mock("$app/navigation", () => ({
  goto: mockGoto,
  beforeNavigate: vi.fn(),
}))

vi.mock("$lib/api_client", () => ({
  client: {
    GET: mockClientGET,
  },
}))

vi.mock("$lib/stores", () => ({
  load_task: mockLoadTask,
  load_available_models: mockLoadModels,
}))

vi.mock("posthog-js", () => ({
  default: { capture: vi.fn() },
}))

vi.mock("$lib/stores/evals_store", () => ({
  set_current_eval_config: vi.fn().mockResolvedValue({}),
}))

vi.mock("$lib/agent", () => ({
  agentInfo: { set: vi.fn() },
}))

// Stub heavy Svelte components
vi.mock("$lib/utils/form_container.svelte", async () => {
  const Stub = await import("./__tests__/form_container_stub.svelte")
  return { default: Stub.default }
})

vi.mock("$lib/ui/collapse.svelte", async () => {
  const Stub = await import("./__tests__/collapse_stub.svelte")
  return { default: Stub.default }
})

vi.mock("$lib/ui/dialog.svelte", async () => {
  const Stub = await import("./__tests__/dialog_stub.svelte")
  return { default: Stub.default }
})

vi.mock("$lib/components/eval_types/llm_judge_form.svelte", async () => {
  const Stub = await import("./__tests__/llm_judge_form_stub.svelte")
  return { default: Stub.default }
})

// Stub TaskRunPicker
vi.mock("$lib/utils/task_run_picker.svelte", async () => {
  const Stub = await import("./__tests__/task_run_picker_stub.svelte")
  return { default: Stub.default }
})

// Mock the registry so that svelte:component uses our lightweight V2 form stub
const V2FormStubModule = await import("./__tests__/v2_form_stub.svelte")
vi.mock("$lib/utils/eval_types/registry", async (importOriginal) => {
  const original = (await importOriginal()) as Record<string, unknown>
  return {
    ...original,
    getV2EvalTypeMetadata: (type: string) => {
      const meta = (
        original.getV2EvalTypeMetadata as (t: string) => Record<string, unknown>
      )(type)
      return {
        ...meta,
        createFormComponent: V2FormStubModule.default,
      }
    },
  }
})

// Mock the v2_eval_api functions
const mockTestV2Eval = vi.fn()
const mockCreateEvalConfig = vi.fn()
const mockCreateLlmJudgeConfig = vi.fn()
const mockCheckAddCodeTrust = vi.fn()
const mockAddCodeTrust = vi.fn()
const mockFetchTaskRuns = vi.fn()
const mockTestV2EvalLlmJudge = vi.fn()

const sampleTaskRun = {
  v: 1,
  id: "run1",
  input: "test input",
  output: { output: "test output", source: { type: "human" as const } },
  tags: [],
  created_at: new Date().toISOString(),
}

vi.mock("$lib/api/v2_eval_api", async (importOriginal) => {
  const original = (await importOriginal()) as Record<string, unknown>
  return {
    ...original,
    testV2Eval: (...args: unknown[]) => mockTestV2Eval(...args),
    testV2EvalLlmJudge: (...args: unknown[]) => mockTestV2EvalLlmJudge(...args),
    createEvalConfig: (...args: unknown[]) => mockCreateEvalConfig(...args),
    createLlmJudgeConfig: (...args: unknown[]) =>
      mockCreateLlmJudgeConfig(...args),
    checkAddCodeTrust: (...args: unknown[]) => mockCheckAddCodeTrust(...args),
    addCodeTrust: (...args: unknown[]) => mockAddCodeTrust(...args),
    fetchTaskRuns: (...args: unknown[]) => mockFetchTaskRuns(...args),
  }
})

// Mock string_to_json_key
vi.mock("$lib/utils/json_schema_editor/json_schema_templates", () => ({
  string_to_json_key: (s: string) =>
    s
      .trim()
      .toLowerCase()
      .replace(/ /g, "_")
      .replace(/[^a-z0-9_.]/g, ""),
}))

// Mock format_expanded_content
vi.mock("$lib/utils/format_expanded_content", () => ({
  formatExpandedContent: (text: string) => ({ isJson: false, value: text }),
}))

// Dynamic imports after all mocks
const EvalConfigBuilder = (
  await import("$lib/components/eval_types/eval_config_builder.svelte")
).default
const { showCalls, resetCalls, actionButtonsByTitle } = await import(
  "./__tests__/dialog_stub.svelte"
)
const { setInitialCode, resetInitialCode } = await import(
  "./__tests__/v2_form_stub.svelte"
)
const { setInitialLlmJudgeValues, resetInitialLlmJudgeValues } = await import(
  "./__tests__/llm_judge_form_stub.svelte"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Render the EvalConfigBuilder component directly.
 */
async function renderBuilder(evalType: string = "code_eval") {
  onMountCallbacks.length = 0

  const spy = vi
    .spyOn(svelteMod, "onMount")
    .mockImplementation((fn: () => unknown) => {
      onMountCallbacks.push(fn)
    })

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const result = render(EvalConfigBuilder as any, {
    props: {
      eval_config_type: evalType,
      evaluator: {
        id: "eval1",
        name: "Test Eval",
        output_scores: [],
      },
      task: {
        id: "task1",
        name: "Test Task",
        instruction: "test instruction",
        input_json_schema: "{}",
        output_json_schema: "{}",
      },
      spec: { id: "spec1", name: "Test Spec" },
      project_id: "proj1",
      task_id: "task1",
      eval_id: "eval1",
      spec_id: "spec1",
    },
  })

  spy.mockRestore()

  for (const cb of onMountCallbacks) {
    await cb()
  }
  await tick()

  return result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("Reference data save gate", () => {
  beforeEach(() => {
    resetCalls()
    resetInitialCode()
    resetInitialLlmJudgeValues()
    mockTestV2Eval.mockReset()
    mockTestV2EvalLlmJudge.mockReset()
    mockCreateEvalConfig.mockReset()
    mockCreateLlmJudgeConfig.mockReset()
    mockCheckAddCodeTrust.mockReset()
    mockAddCodeTrust.mockReset()
    mockFetchTaskRuns.mockReset()
    mockFetchTaskRuns.mockResolvedValue([sampleTaskRun])
  })

  afterEach(() => {
    resetInitialCode()
    resetInitialLlmJudgeValues()
    cleanup()
  })

  describe("code_eval save gate", () => {
    it("blocks save when code body uses reference_data and no passing test", async () => {
      setInitialCode(
        'def score(output, reference_data=None):\n  val = reference_data["key"]\n  return {"score": 1.0}',
      )
      mockCheckAddCodeTrust.mockResolvedValueOnce({ trusted: true })

      const { container } = await renderBuilder("code_eval")

      // Wait for code_string binding to propagate from the stub to the builder
      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      const submitBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement
      await fireEvent.click(submitBtn)

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(showCalls).toContain("Test Required")
      expect(showCalls).not.toContain("Save Without Testing?")
      expect(mockCreateEvalConfig).not.toHaveBeenCalled()
      expect(mockCreateLlmJudgeConfig).not.toHaveBeenCalled()
    })
  })

  describe("llm_judge save gate", () => {
    afterEach(() => {
      resetInitialLlmJudgeValues()
    })

    it("blocks save when prompt contains reference_data and no passing test", async () => {
      setInitialLlmJudgeValues({
        selected_algo: "llm_as_judge",
        combined_model_name: "openai:gpt-4o",
        model_name: "gpt-4o",
        provider_name: "openai",
        judge_prompt: "Score based on {{ reference_data.expected_answer }}",
      })

      const { container } = await renderBuilder("llm_judge")

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      const submitBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement
      expect(submitBtn).not.toBeNull()
      await fireEvent.click(submitBtn)

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(showCalls).toContain("Test Required")
      expect(showCalls).not.toContain("Save Without Testing?")
      expect(mockCreateLlmJudgeConfig).not.toHaveBeenCalled()
      expect(mockCreateEvalConfig).not.toHaveBeenCalled()
    })

    it("an edit while a reference_data test is in flight does not count as tested", async () => {
      setInitialLlmJudgeValues({
        selected_algo: "llm_as_judge",
        combined_model_name: "openai:gpt-4o",
        model_name: "gpt-4o",
        provider_name: "openai",
        // reference_data in the prompt makes the test-before-save gate apply.
        judge_prompt: "Grade against {{ reference_data.expected }}.",
      })

      const { container } = await renderBuilder("llm_judge")

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      // Hold the test in flight so we can edit before it resolves.
      let resolveTest: (v: unknown) => void = () => {}
      mockTestV2EvalLlmJudge.mockReturnValueOnce(
        new Promise((r) => {
          resolveTest = r
        }),
      )

      await fireEvent.click(
        container.querySelector(
          '[data-testid="run-test-btn"]',
        ) as HTMLButtonElement,
      )
      await tick()

      // Edit while the request is in flight.
      const formStub = container.querySelector(
        '[data-testid="llm-judge-form-stub"]',
      ) as HTMLElement
      await fireEvent.input(formStub)
      await tick()

      resolveTest({ scores: { score: 1.0 }, skipped_reason: null })
      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      resetCalls()
      await fireEvent.click(
        container.querySelector(
          '[data-testid="column-save-button"]',
        ) as HTMLButtonElement,
      )
      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      // The edited config was never tested, so the reference-data gate blocks it.
      expect(showCalls).toContain("Test Required")
      expect(mockCreateLlmJudgeConfig).not.toHaveBeenCalled()
    })
  })

  describe("reference_keys on save", () => {
    /**
     * Helper: enter reference data via the test-pane's reference data editor.
     * Opens the editor, fills in rows, and invokes the dialog Save action
     * so the change event propagates to the builder's advanced_reference_data.
     */
    async function enterReferenceData(
      container: HTMLElement,
      entries: Array<{ key: string; value: string }>,
    ) {
      const editBtn = container.querySelector(
        '[data-testid="reference-data-edit"]',
      ) as HTMLButtonElement
      expect(editBtn).not.toBeNull()
      await fireEvent.click(editBtn)
      await tick()

      // The editor now has one empty row. Fill in the first entry.
      const keyInputs = container.querySelectorAll(
        '[data-testid="reference-data-key"]',
      )
      const valueInputs = container.querySelectorAll(
        '[data-testid="reference-data-value"]',
      )
      expect(keyInputs.length).toBeGreaterThanOrEqual(1)

      // Fill the first row
      await fireEvent.input(keyInputs[0], {
        target: { value: entries[0].key },
      })
      await fireEvent.input(valueInputs[0], {
        target: { value: entries[0].value },
      })
      await tick()

      // Add and fill additional rows
      for (let i = 1; i < entries.length; i++) {
        const addBtn = container.querySelector(
          '[data-testid="reference-data-add"]',
        ) as HTMLButtonElement
        await fireEvent.click(addBtn)
        await tick()

        const allKeys = container.querySelectorAll(
          '[data-testid="reference-data-key"]',
        )
        const allValues = container.querySelectorAll(
          '[data-testid="reference-data-value"]',
        )
        await fireEvent.input(allKeys[i], {
          target: { value: entries[i].key },
        })
        await fireEvent.input(allValues[i], {
          target: { value: entries[i].value },
        })
        await tick()
      }

      // Invoke the dialog's Save action to dispatch the change event
      const refDataButtons = actionButtonsByTitle["Reference Data"]
      expect(refDataButtons).toBeTruthy()
      const saveAction = refDataButtons.find(
        (b: Record<string, unknown>) => b.label === "Save",
      )
      expect(saveAction).toBeTruthy()
      const actionFn = saveAction!.action as () => boolean
      actionFn()
      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()
    }

    it("code_eval: saves non-empty reference_keys from entered reference data", async () => {
      setInitialCode(
        'def score(output, reference_data=None):\n  val = reference_data["key"]\n  return {"score": 1.0}',
      )

      const { container } = await renderBuilder("code_eval")

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      // Enter reference data with two keys
      await enterReferenceData(container, [
        { key: "expected", value: '"foo"' },
        { key: "context", value: '"bar"' },
      ])

      // Run a passing test (required by save gate)
      mockTestV2Eval.mockResolvedValueOnce({
        scores: { score: 1.0 },
        skipped_reason: null,
        skipped_detail: null,
      })

      const tryBtn = container.querySelector(
        '[data-testid="run-test-btn"]',
      ) as HTMLButtonElement
      expect(tryBtn).not.toBeNull()
      await fireEvent.click(tryBtn)

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      // Save
      resetCalls()
      mockCheckAddCodeTrust.mockResolvedValueOnce({ trusted: true })
      mockCreateEvalConfig.mockResolvedValueOnce({
        id: "config123",
        type: "v2",
        properties: {},
      })

      const submitBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement
      await fireEvent.click(submitBtn)

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(mockCreateEvalConfig).toHaveBeenCalledTimes(1)
      const savedProps = mockCreateEvalConfig.mock.calls[0][3].properties
      expect(savedProps.reference_keys).toEqual(["expected", "context"])
    })

    it("llm_judge: entered reference data never becomes the saved reference_keys", async () => {
      setInitialLlmJudgeValues({
        selected_algo: "llm_as_judge",
        combined_model_name: "openai:gpt-4o",
        model_name: "gpt-4o",
        provider_name: "openai",
        judge_prompt: "Score based on {{ reference_data.expected_answer }}",
      })

      const { container } = await renderBuilder("llm_judge")

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      // Enter reference data with two keys
      await enterReferenceData(container, [
        { key: "expected_answer", value: '"hello"' },
        { key: "topic", value: '"greetings"' },
      ])

      // Run a passing test (required by save gate)
      mockTestV2EvalLlmJudge.mockResolvedValueOnce({
        scores: { quality: 1.0 },
        skipped_reason: null,
        skipped_detail: null,
      })

      const tryBtn = container.querySelector(
        '[data-testid="run-test-btn"]',
      ) as HTMLButtonElement
      expect(tryBtn).not.toBeNull()
      await fireEvent.click(tryBtn)

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      // Save
      resetCalls()
      mockCreateLlmJudgeConfig.mockResolvedValueOnce({
        id: "config456",
        type: "v2",
        properties: {},
      })

      const submitBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement
      await fireEvent.click(submitBtn)

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(mockCreateLlmJudgeConfig).toHaveBeenCalledTimes(1)
      const savedPayload = mockCreateLlmJudgeConfig.mock.calls[0][3]
      // Whatever the tester typed describes one test case, not what the judge
      // requires. The server derives reference_keys from the eval instead.
      expect(savedPayload).not.toHaveProperty("reference_keys")
    })
  })
})
