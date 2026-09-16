// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from "vitest"
import { render, cleanup, fireEvent } from "@testing-library/svelte"
import { tick } from "svelte"
import { build_eval_generation_splits_param } from "$lib/utils/eval_generation_splits"

// ---------------------------------------------------------------------------
// Module-level mocks — must come before the dynamic page import
// ---------------------------------------------------------------------------

const {
  mockPage,
  mockGoto,
  mockClientGET,
  setEvalResponse,
  setProgressResponse,
} = vi.hoisted(() => {
  type PageValue = {
    params: Record<string, string>
    url: URL
  }
  let pageValue: PageValue = {
    params: {
      project_id: "proj1",
      task_id: "task1",
      spec_id: "spec1",
      eval_id: "eval1",
    },
    url: new URL("http://localhost/specs/proj1/task1/spec1/eval1"),
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

  const default_eval = {
    id: "eval1",
    name: "Test Eval",
    eval_set_filter_id: "tag::test",
    eval_configs_filter_id: "tag::golden",
    eval_configs: [],
    output_scores: [{ name: "accuracy", type: "five_star" }],
  }
  const default_progress = {
    dataset_size: 30,
    golden_dataset_size: 30,
    golden_dataset_not_rated_count: 0,
    golden_dataset_partially_rated_count: 0,
    current_eval_method: null,
    train_dataset_size: 0,
    val_dataset_size: 0,
  }
  let eval_response: Record<string, unknown> = default_eval
  let progress_response: Record<string, unknown> = default_progress
  const setEvalResponse = (v: Record<string, unknown> | null) => {
    eval_response = v ?? default_eval
  }
  const setProgressResponse = (v: Record<string, unknown> | null) => {
    progress_response = v ?? default_progress
  }

  const mockClientGET = vi.fn().mockImplementation((path: string) => {
    if (path.includes("/evals/") && path.includes("/progress")) {
      return Promise.resolve({ data: progress_response, error: null })
    }
    if (path.includes("/evals/")) {
      return Promise.resolve({ data: eval_response, error: null })
    }
    if (path.includes("/specs/")) {
      return Promise.resolve({
        data: { id: "spec1", name: "Test Spec" },
        error: null,
      })
    }
    return Promise.resolve({ data: null, error: null })
  })

  return {
    mockPage,
    mockGoto,
    mockClientGET,
    setEvalResponse,
    setProgressResponse,
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
  model_info: {
    subscribe: (fn: (v: null) => void) => {
      fn(null)
      return () => {}
    },
  },
  load_model_info: vi.fn(),
  model_name: () => "",
  load_available_models: vi.fn(),
  available_tools: {
    subscribe: (fn: (v: Record<string, never>) => void) => {
      fn({})
      return () => {}
    },
  },
  load_available_tools: vi.fn(),
}))

vi.mock("$lib/stores/progress_ui_store", () => ({
  progress_ui_state: {
    set: vi.fn(),
    subscribe: (fn: (v: null) => void) => {
      fn(null)
      return () => {}
    },
  },
}))

vi.mock("$lib/agent", () => ({
  agentInfo: { set: vi.fn() },
}))

// Stub heavy UI components with real .svelte stubs
vi.mock("../../../../../app_page.svelte", async () => {
  const Stub = await import("./__tests__/app_page_stub.svelte")
  return { default: Stub.default }
})

vi.mock("$lib/ui/info_tooltip.svelte", async () => {
  const Stub = await import("./__tests__/info_tooltip_stub.svelte")
  return { default: Stub.default }
})

vi.mock("$lib/ui/property_list.svelte", async () => {
  const Stub = await import("./__tests__/property_list_stub.svelte")
  return { default: Stub.default }
})

vi.mock("$lib/ui/edit_dialog.svelte", async () => {
  const Stub = await import("./__tests__/edit_dialog_stub.svelte")
  return { default: Stub.default }
})

const tagFromFilterId = (id: string) =>
  id.startsWith("tag::") ? id.replace("tag::", "") : undefined

vi.mock("../../spec_utils", () => ({
  tagFromFilterId,
  // The real helper links only for a tag filter id it was actually given, so the stub
  // does the same, through the same tag extraction: a stub that always returns undefined
  // cannot tell "we passed the TaskRun-only accessor and got nothing" from "we passed the
  // wrong accessor", and one that skipped tagFromFilterId would link a non-tag filter the
  // real helper refuses.
  linkFromFilterId: (
    project_id: string,
    task_id: string,
    filter_id: string | null | undefined,
  ) => {
    if (!filter_id) {
      return undefined
    }
    const tag = tagFromFilterId(filter_id)
    return tag ? `/dataset/${project_id}/${task_id}?tags=${tag}` : undefined
  },
}))

// Dynamic import after all mocks
const EvalDetailPage = (await import("./+page.svelte")).default

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function render_page(): Promise<HTMLElement> {
  const { container } = render(EvalDetailPage)
  await tick()
  await new Promise((r) => setTimeout(r, 0))
  await tick()
  return container
}

type RenderedProperty = Record<string, string | undefined>

async function rendered_properties(): Promise<RenderedProperty[]> {
  const container = await render_page()
  const lists = container.querySelectorAll("[data-testid='property-list-stub']")
  return Array.from(lists).flatMap((list) =>
    JSON.parse(list.getAttribute("data-properties") ?? "[]"),
  )
}

function row(
  properties: RenderedProperty[],
  name: string,
): RenderedProperty | undefined {
  return properties.find((property) => property.name === name)
}

// The page's copy is wrapped across source lines, so compare on collapsed whitespace.
function visible_text(container: HTMLElement): string {
  return (container.textContent ?? "").replace(/\s+/g, " ").trim()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("eval detail page — docs link removal (Phase 9)", () => {
  afterEach(() => {
    cleanup()
  })

  it("does not render 'Read the Docs' text anywhere on the page", async () => {
    const { container } = render(EvalDetailPage)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    expect(container.textContent).not.toContain("Read the Docs")
    expect(container.textContent).not.toContain("Read the docs")
  })

  it("does not pass sub_subtitle to AppPage", async () => {
    const { container } = render(EvalDetailPage)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    const appPage = container.querySelector("[data-testid='app-page-stub']")
    expect(appPage).not.toBeNull()
    expect(appPage?.getAttribute("data-sub-subtitle")).toBeNull()
  })

  it("does not pass sub_subtitle_link to AppPage", async () => {
    const { container } = render(EvalDetailPage)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    const appPage = container.querySelector("[data-testid='app-page-stub']")
    expect(appPage).not.toBeNull()
    expect(appPage?.getAttribute("data-sub-subtitle-link")).toBeNull()
  })

  it("does not contain any docs.kiln.tech URLs", async () => {
    const { container } = render(EvalDetailPage)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    expect(container.innerHTML).not.toContain("docs.kiln.tech")
  })
})

describe("eval detail page — dataset rows", () => {
  afterEach(() => {
    cleanup()
    setEvalResponse(null)
    setProgressResponse(null)
  })

  it("renders a legacy eval's dataset rows from its flat fields", async () => {
    setEvalResponse({
      id: "eval1",
      name: "Test Eval",
      eval_set_filter_id: "tag::test",
      train_set_filter_id: "tag::train",
      eval_configs_filter_id: "tag::golden",
      splits: {},
      eval_configs: [],
      output_scores: [{ name: "accuracy", type: "five_star" }],
    })
    setProgressResponse({
      dataset_size: 30,
      golden_dataset_size: 30,
      golden_dataset_not_rated_count: 0,
      golden_dataset_partially_rated_count: 0,
      current_eval_method: null,
      train_dataset_size: 7,
      val_dataset_size: 0,
    })

    const properties = await rendered_properties()

    expect(row(properties, "Test Dataset")?.value).toBe("tag::test (30 items)")
    expect(row(properties, "Test Dataset")?.link).toContain("tags=test")
    expect(row(properties, "Training Dataset")?.value).toBe(
      "tag::train (7 items)",
    )
    // A legacy eval has no val split, and nothing mints one on load. The row still
    // renders so "no val set" is visible rather than absent, with no filter id, no
    // item count and no dataset link to point at.
    expect(row(properties, "Validation Dataset")?.value).toBe("Not configured")
    expect(row(properties, "Validation Dataset")?.link).toBeUndefined()
  })

  it("renders a splits-only eval's dataset rows, including val", async () => {
    // The server omits a split from `splits` only when it wrote it to a legacy field,
    // so a V2 eval arrives with null flat fields and everything in `splits`. Reading
    // only the flat fields rendered "null (30 items)" here.
    setEvalResponse({
      id: "eval1",
      name: "Test Eval",
      eval_set_filter_id: null,
      train_set_filter_id: null,
      splits: {
        test: { source: "eval_input", filter_id: "tag::inputs" },
        train: { source: "task_run", filter_id: "tag::train_x" },
        val: { source: "task_run", filter_id: "tag::val_x" },
      },
      eval_configs: [],
      output_scores: [{ name: "accuracy", type: "five_star" }],
    })
    setProgressResponse({
      dataset_size: 30,
      golden_dataset_size: 0,
      golden_dataset_not_rated_count: 0,
      golden_dataset_partially_rated_count: 0,
      current_eval_method: null,
      train_dataset_size: 7,
      val_dataset_size: 3,
    })

    const properties = await rendered_properties()

    expect(row(properties, "Test Dataset")?.value).toBe(
      "tag::inputs (30 items)",
    )
    // The filter id is worth showing; a /dataset link built from it is not, because the
    // items are EvalInputs and that page lists task runs. This is the whole reason
    // task_run_split_filter_id exists next to eval_split_filter_id — swapping the two
    // would leave every value assertion here passing.
    expect(row(properties, "Test Dataset")?.link).toBeUndefined()
    expect(row(properties, "Training Dataset")?.value).toBe(
      "tag::train_x (7 items)",
    )
    expect(row(properties, "Training Dataset")?.link).toContain("tags=train_x")
    expect(row(properties, "Validation Dataset")?.value).toBe(
      "tag::val_x (3 items)",
    )
    expect(row(properties, "Validation Dataset")?.link).toContain("tags=val_x")
  })

  const DATASET_ROWS = [
    "Test Dataset",
    "Training Dataset",
    "Validation Dataset",
    "Golden Dataset",
  ]

  it("renders every dataset row when the eval has all four", async () => {
    setEvalResponse({
      id: "eval1",
      name: "Test Eval",
      eval_set_filter_id: null,
      train_set_filter_id: null,
      eval_configs_filter_id: "tag::golden_x",
      splits: {
        test: { source: "task_run", filter_id: "tag::test_x" },
        train: { source: "task_run", filter_id: "tag::train_x" },
        val: { source: "task_run", filter_id: "tag::val_x" },
      },
      eval_configs: [],
      output_scores: [{ name: "accuracy", type: "five_star" }],
    })
    setProgressResponse({
      dataset_size: 30,
      golden_dataset_size: 12,
      golden_dataset_not_rated_count: 0,
      golden_dataset_partially_rated_count: 0,
      current_eval_method: null,
      train_dataset_size: 7,
      val_dataset_size: 3,
    })

    const properties = await rendered_properties()

    expect(row(properties, "Test Dataset")?.value).toBe(
      "tag::test_x (30 items)",
    )
    expect(row(properties, "Test Dataset")?.link).toContain("tags=test_x")
    expect(row(properties, "Golden Dataset")?.value).toBe(
      "tag::golden_x (12 items)",
    )
    expect(row(properties, "Golden Dataset")?.link).toContain("tags=golden_x")
    expect(row(properties, "Training Dataset")?.value).toBe(
      "tag::train_x (7 items)",
    )
    expect(row(properties, "Training Dataset")?.link).toContain("tags=train_x")
    expect(row(properties, "Validation Dataset")?.value).toBe(
      "tag::val_x (3 items)",
    )
    expect(row(properties, "Validation Dataset")?.link).toContain("tags=val_x")
  })

  it("renders every dataset row as 'Not configured' when the eval has none", async () => {
    // A row that disappears when its dataset is absent is indistinguishable from a page
    // that never shows that dataset, so all four say so instead. Training is the reason
    // this is asserted: the migration that stamped a train tag onto every legacy eval was
    // removed, so a hidden row would silently vanish from every pre-existing eval.
    setEvalResponse({
      id: "eval1",
      name: "Test Eval",
      eval_set_filter_id: null,
      train_set_filter_id: null,
      eval_configs_filter_id: null,
      splits: {},
      eval_configs: [],
      output_scores: [{ name: "accuracy", type: "five_star" }],
    })

    const properties = await rendered_properties()

    for (const name of DATASET_ROWS) {
      expect(row(properties, name)?.value).toBe("Not configured")
      // Nothing to count and nothing to link to, but the tooltip still explains what the
      // dataset would be for.
      expect(row(properties, name)?.link).toBeUndefined()
      expect(row(properties, name)?.tooltip).toBeTruthy()
    }
  })

  it("orders the dataset rows test, training, validation, golden", async () => {
    // The order is a choice, not an accident: the three `splits` datasets read as a set
    // and golden — a different kind of thing — sits after them rather than between test
    // and training. Assert it so a reorder is a decision someone makes on purpose.
    const properties = await rendered_properties()

    const dataset_rows = properties
      .map((property) => property.name ?? "")
      .filter((name) => DATASET_ROWS.includes(name))
    expect(dataset_rows).toEqual(DATASET_ROWS)
  })

  it("keeps each dataset's tooltip distinct and specific", async () => {
    const properties = await rendered_properties()

    expect(row(properties, "Training Dataset")?.tooltip).toBe(
      "The training set used for optimization.",
    )
    expect(row(properties, "Validation Dataset")?.tooltip).toBe(
      "The validation set used for optimization.",
    )
    const tooltips = DATASET_ROWS.map((name) => row(properties, name)?.tooltip)
    expect(new Set(tooltips).size).toBe(DATASET_ROWS.length)
  })
})

describe("eval detail page — add eval data", () => {
  afterEach(() => {
    cleanup()
    mockGoto.mockClear()
    setEvalResponse(null)
    setProgressResponse(null)
  })

  async function splits_param_from_add_eval_data(): Promise<string | null> {
    const container = await render_page()
    const button = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent?.trim() === "Add Eval Data",
    )
    expect(button).toBeDefined()
    await fireEvent.click(button!)
    await tick()

    expect(mockGoto).toHaveBeenCalledTimes(1)
    const url = new URL(mockGoto.mock.calls[0][0], "http://localhost")
    return url.searchParams.get("splits")
  }

  it("allocates across every dataset the eval has, via the shared helper", async () => {
    const evaluator = {
      id: "eval1",
      name: "Test Eval",
      eval_set_filter_id: null,
      train_set_filter_id: null,
      eval_configs_filter_id: "tag::golden_x",
      splits: {
        test: { source: "task_run", filter_id: "tag::test_x" },
        train: { source: "task_run", filter_id: "tag::train_x" },
        val: { source: "task_run", filter_id: "tag::val_x" },
      },
      eval_configs: [],
      output_scores: [{ name: "accuracy", type: "five_star" }],
    }
    setEvalResponse(evaluator)

    const splits_param = await splits_param_from_add_eval_data()

    // Same allocation the synthetic data generation intro and the compare page produce:
    // all three ask this helper rather than hardcoding one.
    expect(splits_param).toBe(
      build_eval_generation_splits_param(evaluator as never),
    )
    expect(splits_param).toBe("train_x:0.4,val_x:0.25,test_x:0.25,golden_x:0.1")
  })

  it("sends a rag eval's whole allocation to test, skipping its golden tag", async () => {
    // Rag evals are minted with a golden tag like every other eval, but the rag flow has no
    // human-ratings step to read it. This page's old hardcoded param skipped golden for rag,
    // and the shared helper has to keep doing so — otherwise the share it allocates lands in
    // a tag the user has no way to consume.
    setEvalResponse({
      id: "eval1",
      name: "Test Eval",
      template: "rag",
      eval_set_filter_id: "tag::test_x",
      eval_configs_filter_id: "tag::golden_x",
      splits: {},
      eval_configs: [],
      output_scores: [{ name: "accuracy", type: "five_star" }],
    })

    expect(await splits_param_from_add_eval_data()).toBe("test_x:1")
  })

  async function alert_from_add_eval_data(): Promise<string[]> {
    const alerts: string[] = []
    vi.stubGlobal("alert", (message: string) => alerts.push(message))
    try {
      const container = await render_page()
      const button = Array.from(container.querySelectorAll("button")).find(
        (b) => b.textContent?.trim() === "Add Eval Data",
      )
      expect(button).toBeDefined()
      await fireEvent.click(button!)
      await tick()
    } finally {
      vi.unstubAllGlobals()
    }
    return alerts
  }

  it("hides the add-data button for an eval-input-backed test split and says why", async () => {
    // This flow adds TaskRuns, so an EvalInput-backed test split has nothing it can
    // add to. Rather than a button that only ever alerts, the page offers no button
    // and explains where the data comes from (the eval builder mints it at save).
    setEvalResponse({
      id: "eval1",
      name: "Test Eval",
      eval_set_filter_id: null,
      eval_configs_filter_id: "tag::golden_x",
      splits: { test: { source: "eval_input", filter_id: "tag::inputs" } },
      eval_configs: [],
      output_scores: [{ name: "accuracy", type: "five_star" }],
    })

    const container = await render_page()
    const button = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent?.trim() === "Add Eval Data",
    )
    expect(button).toBeUndefined()
    expect(container.textContent?.replace(/\s+/g, " ")).toContain(
      "created by the eval builder and can't be extended here",
    )
    expect(mockGoto).not.toHaveBeenCalled()
  })

  it("refuses a test split whose filter isn't a tag", async () => {
    setEvalResponse({
      id: "eval1",
      name: "Test Eval",
      eval_set_filter_id: null,
      eval_configs_filter_id: "tag::golden_x",
      splits: { test: { source: "task_run", filter_id: "high_rating" } },
      eval_configs: [],
      output_scores: [{ name: "accuracy", type: "five_star" }],
    })

    expect(await alert_from_add_eval_data()).toEqual([
      "No test or golden dataset tag found. If you're using a custom filter, please setup the dataset manually.",
    ])
    expect(mockGoto).not.toHaveBeenCalled()
  })
})

describe("eval detail page — eval data goals", () => {
  afterEach(() => {
    cleanup()
    setEvalResponse(null)
    setProgressResponse(null)
  })

  function progress(dataset_size: number, golden_dataset_size: number) {
    return {
      dataset_size,
      golden_dataset_size,
      golden_dataset_not_rated_count: 0,
      golden_dataset_partially_rated_count: 0,
      current_eval_method: null,
      train_dataset_size: 0,
      val_dataset_size: 0,
    }
  }

  it("is satisfied at 25 test dataset items and 12 golden items", async () => {
    // Golden's goal is lower than the test set's because golden takes the smallest share
    // of generated data — at the same 25 it, not the test set, gates this step.
    setProgressResponse(progress(25, 12))

    const text = visible_text(await render_page())

    expect(text).toContain(
      "You have 25 test dataset items and 12 golden items.",
    )
    expect(text).not.toContain("You require additional eval data")
  })

  it("asks for more golden data below 12", async () => {
    setProgressResponse(progress(25, 11))

    const text = visible_text(await render_page())

    expect(text).toContain(
      "You require additional eval data. You only have 11 golden items. We suggest at least 12 items.",
    )
  })

  it("holds golden to the same 12 once a default judge is set", async () => {
    // A default judge sends this step down a second branch, and that branch used to
    // measure golden against the test set's 25 while the copy under it named 12. Every
    // eval with a judge and 12-24 golden items read "You only have 18 golden items. We
    // suggest at least 12 golden items." -- a bar it had already cleared.
    setEvalResponse({
      id: "eval1",
      name: "Test Eval",
      eval_set_filter_id: "tag::test",
      eval_configs_filter_id: "tag::golden",
      eval_configs: [],
      output_scores: [{ name: "accuracy", type: "five_star" }],
      current_config_id: "eval_config1",
    })
    setProgressResponse(progress(25, 12))

    const text = visible_text(await render_page())

    expect(text).toContain(
      "You have 25 test dataset items and 12 golden items.",
    )
    expect(text).not.toContain("You require additional eval data")
  })

  it("still asks for more golden data below 12 once a default judge is set", async () => {
    // The pair that pins the branch above to the golden goal rather than to no check at
    // all.
    setEvalResponse({
      id: "eval1",
      name: "Test Eval",
      eval_set_filter_id: "tag::test",
      eval_configs_filter_id: "tag::golden",
      eval_configs: [],
      output_scores: [{ name: "accuracy", type: "five_star" }],
      current_config_id: "eval_config1",
    })
    setProgressResponse(progress(25, 11))

    const text = visible_text(await render_page())

    expect(text).toContain(
      "You require additional eval data. You only have 11 golden items. We suggest at least 12 items.",
    )
  })

  it("asks for more test dataset data below 25", async () => {
    setProgressResponse(progress(24, 30))

    const text = visible_text(await render_page())

    expect(text).toContain(
      "You require additional eval data. You only have 24 test dataset items. We suggest at least 25 items.",
    )
  })

  it("states both goals separately when both sets are short", async () => {
    // "at least 25 items in each set" was true when the goals matched and became wrong the
    // moment they didn't.
    setProgressResponse(progress(5, 2))

    const text = visible_text(await render_page())

    expect(text).toContain(
      "You only have 5 test dataset items and 2 golden items. We suggest at least 25 test dataset items and 12 golden items.",
    )
    expect(text).not.toContain("items in each set")
  })
})
