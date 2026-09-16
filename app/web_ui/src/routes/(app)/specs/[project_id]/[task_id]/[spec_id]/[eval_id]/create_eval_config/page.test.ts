// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { render, fireEvent, cleanup } from "@testing-library/svelte"
import { tick } from "svelte"
import * as svelteMod from "svelte"

// ---------------------------------------------------------------------------
// Module-level mocks -- must come before the dynamic page import
// ---------------------------------------------------------------------------

const {
  mockPage,
  mockGoto,
  mockClientGET,
  mockLoadTask,
  mockLoadModels,
  onMountCallbacks,
} = vi.hoisted(() => {
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
          splits: { test: { source: "eval_input", filter_id: "tag::inputs" } },
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

  const onMountCallbacks: Array<() => unknown> = []

  return {
    mockPage,
    mockGoto,
    mockClientGET,
    mockLoadTask,
    mockLoadModels,
    onMountCallbacks,
  }
})

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
vi.mock("../../../../../../app_page.svelte", async () => {
  const Stub = await import("./__tests__/app_page_stub.svelte")
  return { default: Stub.default }
})

vi.mock("../../../../../../../app_page.svelte", async () => {
  const Stub = await import("./__tests__/app_page_stub.svelte")
  return { default: Stub.default }
})

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
const PickerPage = (await import("./+page.svelte")).default
const BuilderRoutePage = (await import("./[eval_config_type]/+page.svelte"))
  .default
const EvalConfigBuilder = (
  await import("$lib/components/eval_types/eval_config_builder.svelte")
).default
const { showCalls, resetCalls, actionButtonsByTitle } = await import(
  "./__tests__/dialog_stub.svelte"
)
const { ALL_V2_EVAL_TYPES, getV2EvalTypeMetadata } = await import(
  "$lib/utils/eval_types/registry"
)
const { setInitialCode, resetInitialCode } = await import(
  "./__tests__/v2_form_stub.svelte"
)
const { setInitialLlmJudgeValues, resetInitialLlmJudgeValues } = await import(
  "./__tests__/llm_judge_form_stub.svelte"
)
const { CREATE_EVAL_LAYOUT_KEY } = await import("./context")

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Render the picker page with context provided.
 * Pass spec_id "legacy" to simulate a legacy eval (no backing spec).
 */
async function renderPickerPage(spec_id: string = "spec1") {
  onMountCallbacks.length = 0

  // The picker page uses getContext("create_eval_layout"). We need to
  // set it up before rendering. Use Svelte's component-level context
  // via the `context` option of @testing-library/svelte.
  const { writable } = await import("svelte/store")

  const ctx = new Map()
  ctx.set(CREATE_EVAL_LAYOUT_KEY, {
    evaluator: writable({
      id: "eval1",
      name: "Test Eval",
      output_scores: [],
    }),
    task: writable({
      id: "task1",
      name: "Test Task",
      instruction: "test instruction",
    }),
    // The layout never loads a spec for legacy evals, so the store stays null
    spec: writable(
      spec_id === "legacy" ? null : { id: spec_id, name: "Test Spec" },
    ),
    project_id: writable("proj1"),
    task_id: writable("task1"),
    eval_id: writable("eval1"),
    spec_id: writable(spec_id),
  })

  const result = render(PickerPage, { context: ctx })
  await tick()
  return result
}

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

/**
 * Render the builder route page ([eval_config_type]/+page.svelte) with context.
 * Pass spec_id "legacy" to simulate a legacy eval (no backing spec).
 */
async function renderBuilderRoutePage(
  evalConfigType: string,
  spec_id: string = "spec1",
) {
  onMountCallbacks.length = 0

  mockPage.set({
    params: {
      project_id: "proj1",
      task_id: "task1",
      eval_id: "eval1",
      spec_id,
      eval_config_type: evalConfigType,
    },
    url: new URL(
      `http://localhost/specs/proj1/task1/${spec_id}/eval1/create_eval_config/${evalConfigType}`,
    ),
  })

  const { writable } = await import("svelte/store")

  const ctx = new Map()
  ctx.set(CREATE_EVAL_LAYOUT_KEY, {
    evaluator: writable({
      id: "eval1",
      name: "Test Eval",
      output_scores: [],
    }),
    task: writable({
      id: "task1",
      name: "Test Task",
      instruction: "test instruction",
      input_json_schema: "{}",
      output_json_schema: "{}",
    }),
    // The layout never loads a spec for legacy evals, so the store stays null
    spec: writable(
      spec_id === "legacy" ? null : { id: spec_id, name: "Test Spec" },
    ),
    project_id: writable("proj1"),
    task_id: writable("task1"),
    eval_id: writable("eval1"),
    spec_id: writable(spec_id),
  })

  const result = render(BuilderRoutePage, { context: ctx })
  await tick()
  return result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("create_eval_config picker page", () => {
  afterEach(() => {
    cleanup()
  })

  it("renders all eval types as rows in a single list", async () => {
    const { container } = await renderPickerPage()
    const rows = container.querySelectorAll("button.card")
    expect(rows.length).toBe(ALL_V2_EVAL_TYPES.length)
  })

  it("does not render an 'All judge types' section heading", async () => {
    const { container } = await renderPickerPage()
    expect(container.textContent).not.toContain("All judge types")
  })

  it("shows page title 'Select a Judge Type'", async () => {
    const { container } = await renderPickerPage()
    const appPage = container.querySelector("[data-testid='app-page-stub']")
    expect(appPage?.getAttribute("data-title")).toBe("Select a Judge Type")
  })

  it("shows updated subtitle copy", async () => {
    const { container } = await renderPickerPage()
    const appPage = container.querySelector("[data-testid='app-page-stub']")
    expect(appPage?.getAttribute("data-subtitle")).toBe(
      "Choose how each output gets scored.",
    )
  })

  it("does not show old subtitle copy", async () => {
    const { container } = await renderPickerPage()
    const appPage = container.querySelector("[data-testid='app-page-stub']")
    expect(appPage?.getAttribute("data-subtitle")).not.toContain(
      "Every type produces the same scores",
    )
  })

  it("recommended item is first and shows the Recommended badge", async () => {
    const { container } = await renderPickerPage()
    const rows = container.querySelectorAll("button.card")
    const firstRow = rows[0]
    expect(firstRow.textContent).toContain("LLM as Judge")
    expect(firstRow.textContent).toContain("Recommended")
    const badge = firstRow.querySelector(".badge-primary")
    expect(badge?.textContent).toContain("Recommended")
  })

  it("recommended item is a clickable button (no Continue button)", async () => {
    const { container } = await renderPickerPage()
    const rows = container.querySelectorAll("button.card")
    const firstRow = rows[0]
    expect(firstRow.tagName).toBe("BUTTON")
    expect(firstRow.textContent).not.toContain("Continue")
  })

  it("every row has a structurally right-aligned chevron (flex-none last child of w-full flex row)", async () => {
    const { container } = await renderPickerPage()
    const rows = container.querySelectorAll("button.card")
    expect(rows.length).toBeGreaterThan(0)
    for (const row of rows) {
      const flexRow = row.querySelector(".flex.w-full")
      expect(flexRow).not.toBeNull()

      const children = Array.from(flexRow!.children)
      const lastChild = children[children.length - 1]
      expect(lastChild.tagName).toBe("svg")
      expect(lastChild.classList.contains("flex-none")).toBe(true)

      const contentBlock = flexRow!.querySelector(".flex-1.min-w-0")
      expect(contentBlock).not.toBeNull()
    }
  })

  it("navigates to llm_judge on recommended row click", async () => {
    const { container } = await renderPickerPage()
    const rows = container.querySelectorAll("button.card")
    await fireEvent.click(rows[0])
    await tick()

    expect(mockGoto).toHaveBeenCalledWith(
      expect.stringContaining("/create_eval_config/llm_judge"),
    )
  })

  it("navigates to type on list row click", async () => {
    const { container } = await renderPickerPage()
    const rows = container.querySelectorAll("button.card")
    expect(rows.length).toBe(ALL_V2_EVAL_TYPES.length)
    await fireEvent.click(rows[1])
    await tick()

    expect(mockGoto).toHaveBeenCalledWith(
      expect.stringContaining("/create_eval_config/code_eval"),
    )
  })

  it("preserves query params when navigating", async () => {
    mockPage.set({
      params: {
        project_id: "proj1",
        task_id: "task1",
        eval_id: "eval1",
        spec_id: "spec1",
      },
      url: new URL(
        "http://localhost/specs/proj1/task1/spec1/eval1/create_eval_config?next_page=eval_configs&save_as_default=true",
      ),
    })

    const { container } = await renderPickerPage()
    const rows = container.querySelectorAll("button.card")
    await fireEvent.click(rows[0])
    await tick()

    expect(mockGoto).toHaveBeenCalledWith(
      expect.stringContaining("next_page=eval_configs"),
    )
    expect(mockGoto).toHaveBeenCalledWith(
      expect.stringContaining("save_as_default=true"),
    )

    // Reset page for other tests
    mockPage.set({
      params: {
        project_id: "proj1",
        task_id: "task1",
        eval_id: "eval1",
        spec_id: "spec1",
      },
      url: new URL(
        "http://localhost/specs/proj1/task1/spec1/eval1/create_eval_config",
      ),
    })
  })
})

describe("EvalConfigBuilder", () => {
  beforeEach(() => {
    resetCalls()
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
    cleanup()
  })

  describe("default test run selection", () => {
    function taskRunWithTrace(id: string, trace: unknown[] | null) {
      return {
        ...sampleTaskRun,
        id,
        input: `input ${id}`,
        output: { output: `output ${id}`, source: { type: "human" as const } },
        trace,
      }
    }

    async function selectedRunText(runs: unknown[]) {
      mockFetchTaskRuns.mockResolvedValue(runs)
      const { container } = await renderBuilder("step_count_check")
      const card = container.querySelector(
        "[data-testid='selected-run-card']",
      ) as HTMLElement
      expect(card).not.toBeNull()
      return card.textContent ?? ""
    }

    it("auto-selects the newest run that has a trace", async () => {
      const text = await selectedRunText([
        taskRunWithTrace("no_trace", null),
        taskRunWithTrace("traced", [{ role: "user", content: "hi" }]),
      ])
      expect(text).toContain("input traced")
      expect(text).not.toContain("input no_trace")
    })

    it("treats an empty trace as no trace", async () => {
      const text = await selectedRunText([
        taskRunWithTrace("empty_trace", []),
        taskRunWithTrace("traced", [{ role: "user", content: "hi" }]),
      ])
      expect(text).toContain("input traced")
    })

    it("falls back to the newest run when none have a trace", async () => {
      const text = await selectedRunText([
        taskRunWithTrace("newest", null),
        taskRunWithTrace("older", null),
      ])
      expect(text).toContain("input newest")
    })
  })

  describe("trust modal for code_eval", () => {
    it("shows trust dialog when test returns code_eval_not_trusted", async () => {
      const { container } = await renderBuilder("code_eval")

      // Wait for task runs to load (auto-selects first run)
      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      mockTestV2Eval.mockResolvedValueOnce({
        scores: {},
        skipped_reason: "code_eval_not_trusted",
        skipped_detail: "Code eval is not trusted for this project.",
      })

      const tryBtn = container.querySelector(
        '[data-testid="run-test-btn"]',
      ) as HTMLButtonElement
      expect(tryBtn).not.toBeNull()
      await fireEvent.click(tryBtn)

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(showCalls).toContain("Trust Code and Project?")
    })

    it("shows trust dialog when saving a code_eval without trust", async () => {
      const { container } = await renderBuilder("code_eval")

      mockCheckAddCodeTrust.mockResolvedValueOnce({ trusted: false })

      const submitBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement
      expect(submitBtn).not.toBeNull()
      await fireEvent.click(submitBtn)

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(showCalls).toContain("Trust Code and Project?")
      expect(mockCreateEvalConfig).not.toHaveBeenCalled()
    })

    it("skips trust check for non-requiresTrust types", async () => {
      const { container } = await renderBuilder("exact_match")

      const submitBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement
      expect(submitBtn).not.toBeNull()
      await fireEvent.click(submitBtn)

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(showCalls).not.toContain("Trust Code and Project?")
      expect(mockCheckAddCodeTrust).not.toHaveBeenCalled()
      expect(showCalls).toContain("Save Without Testing?")
    })
  })

  describe("save-without-testing confirm modal", () => {
    it("shows confirm dialog when saving V2 eval without running a test", async () => {
      const { container } = await renderBuilder("exact_match")

      const submitBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement
      await fireEvent.click(submitBtn)

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(showCalls).toContain("Save Without Testing?")
      expect(mockCreateEvalConfig).not.toHaveBeenCalled()
    })

    it("does not show confirm dialog after a successful test run", async () => {
      const { container } = await renderBuilder("exact_match")

      // Wait for task runs to load (auto-selects first run)
      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      mockTestV2Eval.mockResolvedValueOnce({
        scores: { accuracy: 1.0 },
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

      resetCalls()
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

      expect(showCalls).not.toContain("Save Without Testing?")
      expect(mockCreateEvalConfig).toHaveBeenCalledTimes(1)
    })

    it("re-shows the confirm dialog when the config is edited after a passing test", async () => {
      const { container } = await renderBuilder("exact_match")

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      mockTestV2Eval.mockResolvedValueOnce({
        scores: { accuracy: 1.0 },
        skipped_reason: null,
        skipped_detail: null,
      })

      await fireEvent.click(
        container.querySelector(
          '[data-testid="run-test-btn"]',
        ) as HTMLButtonElement,
      )
      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      // Edit the form after the passing test. The input bubbles to the wrapper's
      // on_config_edit, which invalidates the just-passed test.
      const formStub = container.querySelector(
        '[data-testid="v2-form-stub"]',
      ) as HTMLElement
      await fireEvent.input(formStub)
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

      expect(showCalls).toContain("Save Without Testing?")
      expect(mockCreateEvalConfig).not.toHaveBeenCalled()
    })

    it("shows confirm dialog for code_eval after trust is already granted", async () => {
      const { container } = await renderBuilder("code_eval")

      mockCheckAddCodeTrust.mockResolvedValueOnce({ trusted: true })

      const submitBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement
      await fireEvent.click(submitBtn)

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(showCalls).not.toContain("Trust Code and Project?")
      expect(showCalls).toContain("Save Without Testing?")
    })
  })
})

describe("builder route page ([eval_config_type])", () => {
  afterEach(() => {
    cleanup()
    mockPage.set({
      params: {
        project_id: "proj1",
        task_id: "task1",
        eval_id: "eval1",
        spec_id: "spec1",
      },
      url: new URL(
        "http://localhost/specs/proj1/task1/spec1/eval1/create_eval_config",
      ),
    })
  })

  it("shows error for unknown eval type", async () => {
    const { container } = await renderBuilderRoutePage("bogus_type")
    expect(container.textContent).toContain("Unknown Eval Type")
    expect(container.textContent).toContain('"bogus_type" is not a recognized')
  })

  it("renders builder for valid eval type", async () => {
    const { container } = await renderBuilderRoutePage("exact_match")
    expect(container.textContent).not.toContain("Unknown Eval Type")
  })

  it("sets per-type pageTitle on AppPage for each eval type", async () => {
    for (const evalType of ALL_V2_EVAL_TYPES) {
      const meta = getV2EvalTypeMetadata(evalType)
      const { container } = await renderBuilderRoutePage(evalType)
      const appPage = container.querySelector("[data-testid='app-page-stub']")
      expect(appPage?.getAttribute("data-title")).toBe(meta.pageTitle)
      cleanup()
    }
  })

  it("sets per-type pageSubtitle on AppPage", async () => {
    const meta = getV2EvalTypeMetadata("exact_match")
    const { container } = await renderBuilderRoutePage("exact_match")
    const appPage = container.querySelector("[data-testid='app-page-stub']")
    expect(appPage?.getAttribute("data-subtitle")).toBe(meta.pageSubtitle)
  })

  it("does not render sub_subtitle (Read the Docs link removed)", async () => {
    const { container } = await renderBuilderRoutePage("exact_match")
    expect(container.textContent).not.toContain("Read the Docs")
  })
})

describe("EvalConfigBuilder — container shell + intro", () => {
  beforeEach(() => {
    resetCalls()
    mockFetchTaskRuns.mockReset()
    mockFetchTaskRuns.mockResolvedValue([sampleTaskRun])
  })

  afterEach(() => {
    cleanup()
  })

  it("uses xl breakpoint two-column layout (not lg)", async () => {
    const { container } = await renderBuilder("exact_match")
    const grid = container.querySelector(".xl\\:items-start")
    expect(grid).not.toBeNull()
    expect(grid?.className).toContain("xl:grid-cols-")
    expect(grid?.className).not.toContain("lg:grid-cols-")
    expect(grid?.className).not.toContain("lg:flex-row")
    expect(grid?.className).not.toContain("xl:flex-row")
  })

  it("does not render secondary-title block (icon + label heading)", async () => {
    const { container } = await renderBuilder("exact_match")
    const secondaryTitle = container.querySelector(
      ".flex.items-center.gap-2.pt-4.mb-2",
    )
    expect(secondaryTitle).toBeNull()
  })

  it("right column has no bordered box wrapper", async () => {
    const { container } = await renderBuilder("exact_match")
    const borderedBox = container.querySelector(
      ".rounded-lg.border.bg-base-100",
    )
    expect(borderedBox).toBeNull()
  })

  it("renders Test Run heading with app standard font style", async () => {
    const { container } = await renderBuilder("exact_match")
    const headings = container.querySelectorAll(".text-xl.font-bold")
    const texts = Array.from(headings).map((h) => h.textContent?.trim())
    expect(texts).toContain("Test Judge")
  })

  it("renders eval_type_intro with explainer text", async () => {
    const { container } = await renderBuilder("exact_match")
    const intro = container.querySelector("[data-testid='eval-type-intro']")
    expect(intro).not.toBeNull()
    const meta = getV2EvalTypeMetadata("exact_match")
    expect(intro?.textContent).toContain(meta.explainer!)
  })

  it("renders eval_type_intro with type label", async () => {
    const { container } = await renderBuilder("exact_match")
    const intro = container.querySelector("[data-testid='eval-type-intro']")
    expect(intro?.textContent).toContain("Exact Match")
  })

  it("renders eval_type_intro with example when available", async () => {
    const { container } = await renderBuilder("exact_match")
    const intro = container.querySelector("[data-testid='eval-type-intro']")
    const meta = getV2EvalTypeMetadata("exact_match")
    expect(meta.example).toBeTruthy()
    expect(intro?.textContent).toContain(meta.example!)
  })

  it("renders eval_type_intro without example when not available", async () => {
    const { container } = await renderBuilder("contains")
    const intro = container.querySelector("[data-testid='eval-type-intro']")
    expect(intro).not.toBeNull()
    const meta = getV2EvalTypeMetadata("contains")
    expect(meta.example).toBeFalsy()
    expect(intro?.textContent).toContain(meta.explainer!)
  })

  it("B6: explainer is rendered inside a CalloutCard", async () => {
    const { container } = await renderBuilder("exact_match")
    const calloutCard = container.querySelector(
      "[data-testid='eval-type-intro-card']",
    )
    expect(calloutCard).not.toBeNull()
    expect(calloutCard?.classList.contains("card-bordered")).toBe(true)
    const intro = container.querySelector("[data-testid='eval-type-intro']")
    expect(intro).not.toBeNull()
    expect(
      intro?.querySelector("[data-testid='eval-type-intro-card']"),
    ).not.toBeNull()
  })

  it("B7: Save button is at the bottom of column 1 (not spanning both)", async () => {
    const { container } = await renderBuilder("exact_match")
    const saveBtn = container.querySelector(
      "[data-testid='column-save-button']",
    )
    expect(saveBtn).not.toBeNull()
    expect(saveBtn?.textContent?.trim()).toBe("Save")
    expect(saveBtn?.classList.contains("btn-primary")).toBe(true)

    const formContainerSubmit = container.querySelector(
      "[data-testid='form-submit-button']",
    )
    expect(formContainerSubmit).toBeNull()
  })

  it("B7: Save button still gates via handle_submit (trust check for code_eval)", async () => {
    const { container } = await renderBuilder("code_eval")
    mockCheckAddCodeTrust.mockResolvedValueOnce({ trusted: false })

    const saveBtn = container.querySelector(
      "[data-testid='column-save-button']",
    ) as HTMLButtonElement
    await fireEvent.click(saveBtn)

    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    expect(showCalls).toContain("Trust Code and Project?")
    expect(mockCreateEvalConfig).not.toHaveBeenCalled()
  })

  it("B7: Save button still gates via confirm modal when no valid test run", async () => {
    const { container } = await renderBuilder("exact_match")

    const saveBtn = container.querySelector(
      "[data-testid='column-save-button']",
    ) as HTMLButtonElement
    await fireEvent.click(saveBtn)

    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    expect(showCalls).toContain("Save Without Testing?")
    expect(mockCreateEvalConfig).not.toHaveBeenCalled()
  })

  it("has no Save button in the test run pane", async () => {
    const { container } = await renderBuilder("exact_match")
    const pane = container.querySelector("[data-testid='test-run-pane']")
    expect(pane).not.toBeNull()
    const paneSave = pane?.querySelector("[data-testid='save-eval']")
    expect(paneSave).toBeNull()
    const paneSaveWithout = pane?.querySelector(
      "[data-testid='save-without-testing']",
    )
    expect(paneSaveWithout).toBeNull()
  })

  it("B7: Cmd/Ctrl+Enter triggers gated save (confirm modal when no valid test run)", async () => {
    await renderBuilder("exact_match")

    await fireEvent.keyDown(window, {
      key: "Enter",
      metaKey: true,
    })

    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    expect(showCalls).toContain("Save Without Testing?")
    expect(mockCreateEvalConfig).not.toHaveBeenCalled()
  })
})

describe("EvalConfigBuilder — trust modal + save-flow behavior", () => {
  beforeEach(() => {
    resetCalls()
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
    cleanup()
  })

  describe("trust modal redesign", () => {
    it("trust dialog uses new title 'Trust Code and Project?'", async () => {
      const { container } = await renderBuilder("code_eval")

      const dialogs = container.querySelectorAll("[data-testid='dialog-stub']")
      const trustDialog = Array.from(dialogs).find(
        (d) => d.getAttribute("data-title") === "Trust Code and Project?",
      )
      expect(trustDialog).not.toBeNull()
    })

    it("trust dialog body contains new warning copy", async () => {
      const { container } = await renderBuilder("code_eval")

      expect(container.textContent).toContain(
        "This project wants to run Python code on your machine",
      )
      expect(container.textContent).toContain(
        "Never paste code from a stranger or the internet.",
      )
    })

    it("trust dialog has no yellow alert box", async () => {
      const { container } = await renderBuilder("code_eval")

      const dialogs = container.querySelectorAll("[data-testid='dialog-stub']")
      const trustDialog = Array.from(dialogs).find(
        (d) => d.getAttribute("data-title") === "Trust Code and Project?",
      )
      expect(trustDialog).not.toBeNull()
      const alertWarning = trustDialog!.querySelector(".alert-warning")
      expect(alertWarning).toBeNull()
    })

    it("trust dialog has a large warning icon", async () => {
      const { container } = await renderBuilder("code_eval")

      const icon = container.querySelector("[data-testid='trust-warning-icon']")
      expect(icon).not.toBeNull()
      expect(icon!.classList.contains("text-warning")).toBe(true)
      expect(icon!.classList.contains("w-10")).toBe(true)
      expect(icon!.classList.contains("h-10")).toBe(true)
    })

    it("trust dialog action button is labeled 'I Trust this Code'", async () => {
      const { container } = await renderBuilder("code_eval")

      const dialogs = container.querySelectorAll("[data-testid='dialog-stub']")
      const trustDialog = Array.from(dialogs).find(
        (d) => d.getAttribute("data-title") === "Trust Code and Project?",
      )
      expect(trustDialog).not.toBeNull()
      const buttons = JSON.parse(
        trustDialog!.getAttribute("data-action-buttons") || "[]",
      )
      const trustBtn = buttons.find(
        (b: Record<string, unknown>) => b.isWarning === true,
      )
      expect(trustBtn).toBeTruthy()
      expect(trustBtn.label).toBe("I Trust this Code")
    })

    it("C8/C9: trust dialog text is left-aligned with horizontal icon+text layout", async () => {
      const { container } = await renderBuilder("code_eval")

      const dialogs = container.querySelectorAll("[data-testid='dialog-stub']")
      const trustDialog = Array.from(dialogs).find(
        (d) => d.getAttribute("data-title") === "Trust Code and Project?",
      )
      expect(trustDialog).not.toBeNull()

      const layoutDiv = trustDialog!.querySelector(".flex.flex-row.items-start")
      expect(layoutDiv).not.toBeNull()

      const textDiv = layoutDiv!.querySelector(".text-left")
      expect(textDiv).not.toBeNull()

      const centeredText = layoutDiv!.querySelector(".text-center")
      expect(centeredText).toBeNull()

      const columnLayout = trustDialog!.querySelector(
        ".flex.flex-col.items-center.gap-4",
      )
      expect(columnLayout).toBeNull()
    })
  })

  describe("B1: loading state reset on modal defer", () => {
    it("resets create_evaluator_loading when deferring to trust dialog", async () => {
      let trustResolve: (v: unknown) => void
      const trustPromise = new Promise((resolve) => {
        trustResolve = resolve
      })
      mockCheckAddCodeTrust.mockImplementationOnce(() => trustPromise)

      const { container } = await renderBuilder("code_eval")

      const submitBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement

      await fireEvent.click(submitBtn)
      await tick()

      expect(submitBtn.disabled).toBe(true)

      trustResolve!({ trusted: false })
      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(showCalls).toContain("Trust Code and Project?")

      const saveBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement
      expect(saveBtn).not.toBeNull()
      expect(saveBtn.disabled).toBe(false)
    })

    it("resets create_evaluator_loading when deferring to confirm-save dialog", async () => {
      const { container } = await renderBuilder("exact_match")

      const submitBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement

      // For exact_match (no requiresTrust), handle_submit runs synchronously
      // up to the confirm_save_dialog.show() + create_evaluator_loading = false.
      // The stub sets submitting=true on click, but the handler resets it
      // in the same synchronous execution, so we verify the end state.
      await fireEvent.click(submitBtn)
      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(showCalls).toContain("Save Without Testing?")

      const saveBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement
      expect(saveBtn).not.toBeNull()
      expect(saveBtn.disabled).toBe(false)
    })

    it("resets loading when checkAddCodeTrust throws an error", async () => {
      let trustReject: (e: Error) => void
      const trustPromise = new Promise((_resolve, reject) => {
        trustReject = reject
      })
      mockCheckAddCodeTrust.mockImplementationOnce(() => trustPromise)

      const { container } = await renderBuilder("code_eval")

      const submitBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement

      await fireEvent.click(submitBtn)
      await tick()

      expect(submitBtn.disabled).toBe(true)

      trustReject!(new Error("Network error checking trust"))
      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      const saveBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement
      expect(saveBtn).not.toBeNull()
      expect(saveBtn.disabled).toBe(false)
    })
  })

  describe("dismiss-on-trust (fire-and-forget)", () => {
    it("grant_trust fires run_test without awaiting it", async () => {
      const { container } = await renderBuilder("code_eval")

      // Wait for task runs to load (auto-selects first run)
      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      mockTestV2Eval.mockResolvedValueOnce({
        scores: {},
        skipped_reason: "code_eval_not_trusted",
        skipped_detail: "Code eval is not trusted for this project.",
      })

      const tryBtn = container.querySelector(
        '[data-testid="run-test-btn"]',
      ) as HTMLButtonElement
      expect(tryBtn).not.toBeNull()
      await fireEvent.click(tryBtn)

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(showCalls).toContain("Trust Code and Project?")

      const buttons = actionButtonsByTitle["Trust Code and Project?"]
      expect(buttons).toBeTruthy()
      const trustAction = buttons.find(
        (b: Record<string, unknown>) => b.isWarning === true,
      )
      expect(trustAction).toBeTruthy()

      let testResolve: (v: unknown) => void
      const testPromise = new Promise((resolve) => {
        testResolve = resolve
      })
      mockAddCodeTrust.mockResolvedValueOnce({})
      mockTestV2Eval.mockImplementationOnce(() => testPromise)

      const asyncAction = trustAction!.asyncAction as () => Promise<boolean>
      const result = await asyncAction()

      expect(result).toBe(true)
      expect(mockAddCodeTrust).toHaveBeenCalled()

      testResolve!({
        scores: { accuracy: 1.0 },
        skipped_reason: null,
        skipped_detail: null,
      })
      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()
    })

    it("grant_trust fires do_save without awaiting it", async () => {
      const { container } = await renderBuilder("code_eval")

      mockCheckAddCodeTrust.mockResolvedValueOnce({ trusted: false })

      const submitBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement
      await fireEvent.click(submitBtn)

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(showCalls).toContain("Trust Code and Project?")

      const buttons = actionButtonsByTitle["Trust Code and Project?"]
      expect(buttons).toBeTruthy()
      const trustAction = buttons.find(
        (b: Record<string, unknown>) => b.isWarning === true,
      )
      expect(trustAction).toBeTruthy()

      let saveResolve: (v: unknown) => void
      const savePromise = new Promise((resolve) => {
        saveResolve = resolve
      })
      mockAddCodeTrust.mockResolvedValueOnce({})
      mockCreateEvalConfig.mockImplementationOnce(() => savePromise)

      const asyncAction = trustAction!.asyncAction as () => Promise<boolean>
      const result = await asyncAction()

      expect(result).toBe(true)
      expect(mockAddCodeTrust).toHaveBeenCalled()

      saveResolve!({
        id: "config123",
        type: "v2",
        properties: {},
      })
      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()
    })
  })
})

describe("Explainer placement + Judge Configuration header", () => {
  beforeEach(() => {
    resetCalls()
    mockFetchTaskRuns.mockReset()
    mockFetchTaskRuns.mockResolvedValue([sampleTaskRun])
  })

  afterEach(() => {
    cleanup()
  })

  it("renders eval_type_intro in its own grid row above the form column (not inside it)", async () => {
    const { container } = await renderBuilder("exact_match")
    const intro = container.querySelector("[data-testid='eval-type-intro']")
    expect(intro).not.toBeNull()
    // The form column sits in grid row 2, col 1; the intro sits in row 1, col 1.
    const formColumn = container.querySelector(
      ".xl\\:col-start-1.xl\\:row-start-2",
    )
    expect(formColumn).not.toBeNull()
    // intro should not be inside the form column
    expect(formColumn!.contains(intro)).toBe(false)
    // intro (row 1) should come before the form column (row 2) in document order
    const comparison = intro!.compareDocumentPosition(formColumn!)
    expect(comparison & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it("renders 'Judge Configuration' header at the top of the left column", async () => {
    const { container } = await renderBuilder("exact_match")
    const leftCol = container.querySelector(
      ".xl\\:col-start-1.xl\\:row-start-2",
    )
    expect(leftCol).not.toBeNull()
    const heading = leftCol!.querySelector(".text-xl.font-bold")
    expect(heading).not.toBeNull()
    expect(heading?.textContent).toContain("Judge Configuration")
  })

  it("'Judge Configuration' header matches Test Run header classes (text-xl font-bold)", async () => {
    const { container } = await renderBuilder("exact_match")
    const headings = container.querySelectorAll(".text-xl.font-bold")
    const texts = Array.from(headings).map((h) => h.textContent?.trim())
    expect(texts).toContain("Judge Configuration")
    expect(texts).toContain("Test Judge")
  })
})

describe("Unsaved-changes guard gated on typing", () => {
  beforeEach(() => {
    resetCalls()
    mockFetchTaskRuns.mockReset()
    mockFetchTaskRuns.mockResolvedValue([sampleTaskRun])
  })

  afterEach(() => {
    cleanup()
  })

  it("warn_before_unload is false initially (no guard until user types)", async () => {
    const { container } = await renderBuilder("exact_match")
    const formStub = container.querySelector(
      "[data-testid='form-container-stub']",
    )
    expect(formStub).not.toBeNull()
    expect(formStub!.getAttribute("data-warn-before-unload")).toBe("false")
  })

  it("warn_before_unload becomes true after a form input event", async () => {
    const { container } = await renderBuilder("exact_match")

    // Fire a bubbling input event on the v2 form stub inside the content wrapper.
    // This simulates a real user editing a form field.
    const v2Stub = container.querySelector("[data-testid='v2-form-stub']")
    expect(v2Stub).toBeTruthy()
    await fireEvent.input(v2Stub!)
    await tick()

    const formStub = container.querySelector(
      "[data-testid='form-container-stub']",
    )
    expect(formStub!.getAttribute("data-warn-before-unload")).toBe("true")
  })
})

describe("Breadcrumb — Add Judge", () => {
  afterEach(() => {
    cleanup()
    mockPage.set({
      params: {
        project_id: "proj1",
        task_id: "task1",
        eval_id: "eval1",
        spec_id: "spec1",
      },
      url: new URL(
        "http://localhost/specs/proj1/task1/spec1/eval1/create_eval_config",
      ),
    })
  })

  it("renders an 'Add Judge' breadcrumb on the builder route page", async () => {
    const { container } = await renderBuilderRoutePage("exact_match")
    const appPage = container.querySelector("[data-testid='app-page-stub']")
    expect(appPage).not.toBeNull()
    const breadcrumbs = JSON.parse(
      appPage!.getAttribute("data-breadcrumbs") || "[]",
    )
    const addJudge = breadcrumbs.find(
      (b: { label: string }) => b.label === "Add Judge",
    )
    expect(addJudge).toBeTruthy()
  })

  it("'Add Judge' breadcrumb links to the create_eval_config list page", async () => {
    const { container } = await renderBuilderRoutePage("exact_match")
    const appPage = container.querySelector("[data-testid='app-page-stub']")
    const breadcrumbs = JSON.parse(
      appPage!.getAttribute("data-breadcrumbs") || "[]",
    )
    const addJudge = breadcrumbs.find(
      (b: { label: string }) => b.label === "Add Judge",
    )
    expect(addJudge.href).toContain(
      "/specs/proj1/task1/spec1/eval1/create_eval_config",
    )
    // Should NOT point to the type-specific route
    expect(addJudge.href).not.toContain("/exact_match")
  })
})

describe("Breadcrumbs — legacy evals", () => {
  afterEach(() => {
    cleanup()
    mockPage.set({
      params: {
        project_id: "proj1",
        task_id: "task1",
        eval_id: "eval1",
        spec_id: "spec1",
      },
      url: new URL(
        "http://localhost/specs/proj1/task1/spec1/eval1/create_eval_config",
      ),
    })
  })

  function getBreadcrumbs(container: HTMLElement): Array<{
    label: string
    href: string
  }> {
    const appPage = container.querySelector("[data-testid='app-page-stub']")
    expect(appPage).not.toBeNull()
    return JSON.parse(appPage!.getAttribute("data-breadcrumbs") || "[]")
  }

  it("picker page renders the spec crumb for a real spec", async () => {
    const { container } = await renderPickerPage()
    const breadcrumbs = getBreadcrumbs(container)
    expect(breadcrumbs.map((b) => b.label)).toEqual([
      "Evals",
      "Test Spec",
      "Eval",
    ])
    expect(breadcrumbs[1].href).toBe("/specs/proj1/task1/spec1")
  })

  it("picker page drops the spec crumb for legacy evals, keeping the rest", async () => {
    const { container } = await renderPickerPage("legacy")
    const breadcrumbs = getBreadcrumbs(container)
    expect(breadcrumbs.map((b) => b.label)).toEqual(["Evals", "Eval"])
    expect(breadcrumbs[0].href).toBe("/specs/proj1/task1")
    expect(breadcrumbs[1].href).toBe("/specs/proj1/task1/legacy/eval1")
    // No crumb may link to the (nonexistent) legacy spec detail page
    expect(
      breadcrumbs.some((b) => b.href === "/specs/proj1/task1/legacy"),
    ).toBe(false)
  })

  it("builder route page drops the spec crumb for legacy evals, keeping the rest", async () => {
    const { container } = await renderBuilderRoutePage("exact_match", "legacy")
    const breadcrumbs = getBreadcrumbs(container)
    expect(breadcrumbs.map((b) => b.label)).toEqual([
      "Evals",
      "Eval",
      "Add Judge",
    ])
    expect(breadcrumbs[1].href).toBe("/specs/proj1/task1/legacy/eval1")
    expect(
      breadcrumbs.some((b) => b.href === "/specs/proj1/task1/legacy"),
    ).toBe(false)
  })
})

describe("EvalConfigBuilder — docs links + theme-aware colors", () => {
  beforeEach(() => {
    resetCalls()
    mockFetchTaskRuns.mockReset()
    mockFetchTaskRuns.mockResolvedValue([sampleTaskRun])
  })

  afterEach(() => {
    cleanup()
  })

  describe("docs-link audit", () => {
    it("select screen does not render Read the Docs or sub_subtitle", async () => {
      const { container } = await renderPickerPage()
      expect(container.textContent).not.toContain("Read the Docs")
      expect(container.textContent).not.toContain("Read the docs")
      expect(container.textContent).not.toContain("All judge types")
    })

    it("builder route does not render Read the Docs or sub_subtitle", async () => {
      const { container } = await renderBuilderRoutePage("exact_match")
      expect(container.textContent).not.toContain("Read the Docs")
      expect(container.textContent).not.toContain("Read the docs")
    })

    it("builder route does not pass sub_subtitle to AppPage", async () => {
      const { container } = await renderBuilderRoutePage("exact_match")
      const appPage = container.querySelector("[data-testid='app-page-stub']")
      expect(appPage).not.toBeNull()
      expect(appPage?.innerHTML).not.toContain("sub_subtitle")
    })
  })

  describe("design-guide color palette in create-flow components", () => {
    it("eval_type_intro uses design-guide secondary text color", async () => {
      const { container } = await renderBuilder("exact_match")
      const intro = container.querySelector("[data-testid='eval-type-intro']")
      expect(intro).not.toBeNull()
      expect(intro?.innerHTML).toContain("text-gray-500")
      expect(intro?.innerHTML).not.toContain("text-base-content")
    })

    it("test run pane uses design-guide palette, not non-standard colors", async () => {
      const { container } = await renderBuilder("exact_match")
      const pane = container.querySelector("[data-testid='test-run-pane']")
      expect(pane).not.toBeNull()
      expect(pane?.innerHTML).not.toContain("text-base-content")
      expect(pane?.innerHTML).not.toContain("text-gray-400")
      expect(pane?.innerHTML).not.toContain("text-gray-600")
      expect(pane?.innerHTML).not.toContain("text-gray-300")
    })

    it("confirm save dialog uses design-guide secondary text color", async () => {
      const { container } = await renderBuilder("exact_match")
      const dialogs = container.querySelectorAll("[data-testid='dialog-stub']")
      const confirmDialog = Array.from(dialogs).find(
        (d) => d.getAttribute("data-title") === "Save Without Testing?",
      )
      expect(confirmDialog).not.toBeNull()
      expect(confirmDialog?.innerHTML).toContain("text-gray-500")
      expect(confirmDialog?.innerHTML).not.toContain("text-base-content")
    })
  })
})

// Behavior with SHOW_REFERENCE_DATA_UI off (the shipped default): no reference
// data editor is rendered, the "Test Required" gate never fires, and saved
// reference_keys are always empty. The flag-on behavior is covered in
// page.reference_data.test.ts.
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
    it("does not block save when code body uses reference_data (no Test Required gate)", async () => {
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

      expect(showCalls).not.toContain("Test Required")
      expect(showCalls).toContain("Save Without Testing?")
    })

    it("does not render a reference data editor in the test pane", async () => {
      setInitialCode(
        'def score(output, reference_data=None):\n  val = reference_data["key"]\n  return {"score": 1.0}',
      )

      const { container } = await renderBuilder("code_eval")

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(
        container.querySelector('[data-testid="reference-data-field"]'),
      ).toBeNull()
      expect(
        container.querySelector('[data-testid="reference-data-edit"]'),
      ).toBeNull()
    })

    it("allows save when code only has reference_data in signature (no gate)", async () => {
      setInitialCode(
        "def score(output, reference_data=None):\n  return {'score': 1.0}",
      )
      mockCheckAddCodeTrust.mockResolvedValueOnce({ trusted: true })

      const { container } = await renderBuilder("code_eval")

      const submitBtn = container.querySelector(
        '[data-testid="column-save-button"]',
      ) as HTMLButtonElement
      await fireEvent.click(submitBtn)

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(showCalls).toContain("Save Without Testing?")
    })

    it("allows save after passing test when code uses reference_data", async () => {
      setInitialCode(
        'def score(output, reference_data=None):\n  val = reference_data["key"]\n  return {"score": 1.0}',
      )
      mockCheckAddCodeTrust.mockResolvedValueOnce({ trusted: true })

      const { container } = await renderBuilder("code_eval")

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

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

      expect(showCalls).not.toContain("Save Without Testing?")
      expect(mockCreateEvalConfig).toHaveBeenCalledTimes(1)
    })

    it("starter code with docstring + signature reference_data does NOT trigger Test Required", async () => {
      const starterCode = `def score(output, trace, reference_data, task_input):
    """Score the model output.

    Args:
        output: The model's final output string.
        trace: List of message dicts from the conversation.
        reference_data: Dict of reference/expected data (if any).
        task_input: The original task input string.

    Returns:
        no_dark_humour: return 0.0 for Fail or 1.0 for Pass
    """
    if not output:
        return {"no_dark_humour": 0.0}
    return {"no_dark_humour": 1.0}`
      setInitialCode(starterCode)
      mockCheckAddCodeTrust.mockResolvedValueOnce({ trusted: true })

      const { container } = await renderBuilder("code_eval")

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

      expect(showCalls).not.toContain("Test Required")
      expect(showCalls).toContain("Save Without Testing?")
    })
  })

  describe("llm_judge save gate", () => {
    afterEach(() => {
      resetInitialLlmJudgeValues()
    })

    it("does not block save when prompt contains reference_data (no Test Required gate)", async () => {
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

      expect(showCalls).not.toContain("Test Required")
      expect(showCalls).toContain("Save Without Testing?")
    })

    it("renders a reference data editor when the prompt uses reference data", async () => {
      // A reference-answer judge's baked prompt renders a <reference_answer> block.
      // Without an input here the pane can only test it with that block missing, and
      // the user then saves a judge that renders it.
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

      expect(
        container.querySelector('[data-testid="reference-data-field"]'),
      ).not.toBeNull()
    })

    it("renders a reference data editor when the server requires a key the prompt no longer shows", async () => {
      // The dead-end this closes: edit the <reference_answer> block out of a
      // reference-answer judge's prompt and the server still requires the key, so
      // every test run skips with missing_reference_key. Reading only the prompt
      // would hide the one input that can satisfy it.
      setInitialLlmJudgeValues({
        selected_algo: "llm_as_judge",
        combined_model_name: "openai:gpt-4o",
        model_name: "gpt-4o",
        provider_name: "openai",
        judge_prompt: "Score {{ final_message }} for accuracy.",
        default_reference_keys: ["reference_answer"],
      })

      const { container } = await renderBuilder("llm_judge")

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(
        container.querySelector('[data-testid="reference-data-field"]'),
      ).not.toBeNull()
    })

    it("renders a reference data editor when the default prompt could not be fetched", async () => {
      // Nothing is known about the judge, and save bakes the server default either
      // way. Fail open rather than leaving a judge the pane cannot test.
      setInitialLlmJudgeValues({
        selected_algo: "llm_as_judge",
        combined_model_name: "openai:gpt-4o",
        model_name: "gpt-4o",
        provider_name: "openai",
        default_prompt_unavailable: true,
      })

      const { container } = await renderBuilder("llm_judge")

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(
        container.querySelector('[data-testid="reference-data-field"]'),
      ).not.toBeNull()
    })

    it("does not render a reference data editor for a judge that never reads one", async () => {
      setInitialLlmJudgeValues({
        selected_algo: "llm_as_judge",
        combined_model_name: "openai:gpt-4o",
        model_name: "gpt-4o",
        provider_name: "openai",
        judge_prompt: "Score the output for quality.",
      })

      const { container } = await renderBuilder("llm_judge")

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(
        container.querySelector('[data-testid="reference-data-field"]'),
      ).toBeNull()
      expect(
        container.querySelector('[data-testid="reference-data-edit"]'),
      ).toBeNull()
    })

    it("sends a typed reference answer to the test run", async () => {
      setInitialLlmJudgeValues({
        selected_algo: "llm_as_judge",
        combined_model_name: "openai:gpt-4o",
        model_name: "gpt-4o",
        provider_name: "openai",
        judge_prompt: "Score based on {{ reference_data.reference_answer }}",
      })

      const { container } = await renderBuilder("llm_judge")

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      await fireEvent.click(
        container.querySelector(
          '[data-testid="reference-data-edit"]',
        ) as HTMLButtonElement,
      )
      await tick()

      await fireEvent.input(
        container.querySelector(
          '[data-testid="reference-data-key"]',
        ) as HTMLInputElement,
        { target: { value: "reference_answer" } },
      )
      await fireEvent.input(
        container.querySelector(
          '[data-testid="reference-data-value"]',
        ) as HTMLInputElement,
        { target: { value: '"Frank Herbert."' } },
      )
      await tick()
      // The dialog is stubbed, so invoke its Save action directly; that is what
      // dispatches the change back to the builder.
      const saveAction = actionButtonsByTitle["Reference Data"].find(
        (b: Record<string, unknown>) => b.label === "Save",
      )
      expect(saveAction).toBeTruthy()
      ;(saveAction!.action as () => boolean)()
      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      mockTestV2EvalLlmJudge.mockResolvedValueOnce({
        scores: { quality: 1.0 },
        skipped_reason: null,
        skipped_detail: null,
      })
      await fireEvent.click(
        container.querySelector(
          '[data-testid="run-test-btn"]',
        ) as HTMLButtonElement,
      )

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      expect(mockTestV2EvalLlmJudge).toHaveBeenCalledTimes(1)
      const eval_input = mockTestV2EvalLlmJudge.mock.calls[0][4]
      expect(eval_input.reference_data).toEqual({
        reference_answer: "Frank Herbert.",
      })
    })

    it("shows Save Without Testing when prompt does NOT contain reference_data", async () => {
      setInitialLlmJudgeValues({
        selected_algo: "llm_as_judge",
        combined_model_name: "openai:gpt-4o",
        model_name: "gpt-4o",
        provider_name: "openai",
        judge_prompt: "Score the output for quality.",
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

      expect(showCalls).toContain("Save Without Testing?")
    })
  })

  describe("reference_keys on save", () => {
    it("llm_judge: the request carries no reference_keys, the server derives them", async () => {
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

      // Run a passing test
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
      // The client used to post these and the endpoint wrote them over the value it
      // derived from the eval, so a UI that could not collect them turned the
      // requirement off. The field is gone from the request.
      expect(savedPayload).not.toHaveProperty("reference_keys")
    })

    it("code_eval: reference_keys are empty when no reference data is entered", async () => {
      setInitialCode(
        'def score(output, reference_data=None):\n  val = reference_data["key"]\n  return {"score": 1.0}',
      )

      const { container } = await renderBuilder("code_eval")

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      // Run a passing test (required by save gate) — no reference data entered
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
      expect(savedProps.reference_keys).toEqual([])
    })

    it("code_eval: reference_keys are empty when config does not use reference_data", async () => {
      mockCheckAddCodeTrust.mockResolvedValueOnce({ trusted: true })
      mockCreateEvalConfig.mockResolvedValueOnce({
        id: "config123",
        type: "v2",
        properties: {},
      })

      const { container } = await renderBuilder("code_eval")

      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()

      // Run a passing test
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
      expect(savedProps.reference_keys).toEqual([])
    })
  })
})

describe("Save flow — handle_submit logic", () => {
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

  it("contains: save after passing test calls createEvalConfig", async () => {
    const { container } = await renderBuilder("contains")

    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    // Run a passing test
    mockTestV2Eval.mockResolvedValueOnce({
      scores: { accuracy: 1.0 },
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

    // Now save
    resetCalls()
    mockCreateEvalConfig.mockResolvedValueOnce({
      id: "config_contains",
      type: "v2",
      properties: { type: "contains" },
    })

    const submitBtn = container.querySelector(
      '[data-testid="column-save-button"]',
    ) as HTMLButtonElement
    expect(submitBtn).not.toBeNull()
    await fireEvent.click(submitBtn)

    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    expect(mockCreateEvalConfig).toHaveBeenCalledTimes(1)
    expect(showCalls).not.toContain("Save Without Testing?")
  })

  it("code_eval: save after passing test calls createEvalConfig", async () => {
    const { container } = await renderBuilder("code_eval")

    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    // Run a passing test
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

    // Now save (trust granted)
    resetCalls()
    mockCheckAddCodeTrust.mockResolvedValueOnce({ trusted: true })
    mockCreateEvalConfig.mockResolvedValueOnce({
      id: "config_code",
      type: "v2",
      properties: { type: "code_eval" },
    })

    const submitBtn = container.querySelector(
      '[data-testid="column-save-button"]',
    ) as HTMLButtonElement
    expect(submitBtn).not.toBeNull()
    await fireEvent.click(submitBtn)

    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    expect(mockCreateEvalConfig).toHaveBeenCalledTimes(1)
    expect(showCalls).not.toContain("Save Without Testing?")
  })

  it("llm_judge: save after passing test calls createLlmJudgeConfig", async () => {
    setInitialLlmJudgeValues({
      selected_algo: "llm_as_judge",
      combined_model_name: "openai:gpt-4o",
      model_name: "gpt-4o",
      provider_name: "openai",
      judge_prompt: "Score the output for quality.",
    })

    const { container } = await renderBuilder("llm_judge")

    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    // Run a passing test
    mockTestV2EvalLlmJudge.mockResolvedValueOnce({
      scores: { quality: 4.0 },
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

    // Now save
    resetCalls()
    mockCreateLlmJudgeConfig.mockResolvedValueOnce({
      id: "config_llm",
      type: "v2",
      properties: { type: "llm_judge" },
    })

    const submitBtn = container.querySelector(
      '[data-testid="column-save-button"]',
    ) as HTMLButtonElement
    expect(submitBtn).not.toBeNull()
    await fireEvent.click(submitBtn)

    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    expect(mockCreateLlmJudgeConfig).toHaveBeenCalledTimes(1)
    expect(showCalls).not.toContain("Save Without Testing?")
  })

  it("contains: save without test shows confirm dialog and Save Anyway calls createEvalConfig", async () => {
    const { container } = await renderBuilder("contains")

    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    mockCreateEvalConfig.mockResolvedValueOnce({
      id: "config_contains",
      type: "v2",
      properties: { type: "contains" },
    })

    const submitBtn = container.querySelector(
      '[data-testid="column-save-button"]',
    ) as HTMLButtonElement
    expect(submitBtn).not.toBeNull()
    await fireEvent.click(submitBtn)

    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    expect(showCalls).toContain("Save Without Testing?")

    // Click Save Anyway via the dialog stub's action button
    const buttons = actionButtonsByTitle["Save Without Testing?"]
    expect(buttons).toBeTruthy()
    const saveAnyway = buttons.find(
      (b: Record<string, unknown>) => b.label === "Save Anyway",
    )
    expect(saveAnyway).toBeTruthy()

    const asyncAction = saveAnyway!.asyncAction as () => Promise<boolean>
    await asyncAction()

    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    expect(mockCreateEvalConfig).toHaveBeenCalledTimes(1)
  })
})
