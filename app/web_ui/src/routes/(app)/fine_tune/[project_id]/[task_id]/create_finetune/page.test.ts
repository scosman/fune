// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { render, cleanup } from "@testing-library/svelte"
import { tick } from "svelte"
import * as svelteMod from "svelte"
import { writable } from "svelte/store"

// ---------------------------------------------------------------------------
// Module-level mocks -- must come before the dynamic page import
// ---------------------------------------------------------------------------

const { mockPage, mockGoto, mockLoadTask } = vi.hoisted(() => {
  type PageValue = { params: Record<string, string>; url: URL }
  const pageValue: PageValue = {
    params: { project_id: "proj1", task_id: "task1" },
    url: new URL("http://localhost/fine_tune/proj1/task1/create_finetune"),
  }
  const mockPage = {
    subscribe(fn: (v: PageValue) => void) {
      fn(pageValue)
      return () => {}
    },
  }
  return { mockPage, mockGoto: vi.fn(), mockLoadTask: vi.fn() }
})

vi.mock("$app/stores", () => ({ page: mockPage }))
vi.mock("$app/navigation", () => ({ goto: mockGoto }))
vi.mock("posthog-js", () => ({ default: { capture: vi.fn() } }))
vi.mock("$lib/api_client", () => ({
  client: { GET: vi.fn(), POST: vi.fn() },
  base_url: "http://test",
}))
vi.mock("$lib/agent", () => ({ agentInfo: { set: vi.fn() } }))
vi.mock("$lib/stores", () => ({
  fine_tune_target_model: writable<string | null>(null),
  load_task: mockLoadTask,
  current_task: writable(null),
}))
vi.mock("$lib/stores/fine_tune_store", () => ({
  available_tuning_models: writable([]),
  available_models_error: writable<unknown>(null),
  available_models_loading: writable(false),
  get_available_models: vi.fn(),
}))
vi.mock("$lib/stores/progress_ui_store", () => ({
  progress_ui_state: writable(null),
}))
vi.mock("$lib/stores/run_configs_store", () => ({
  load_task_run_configs: vi.fn().mockResolvedValue(undefined),
}))
vi.mock("$lib/stores/index_db_store", () => ({
  indexedDBStore: <T>(_key: string, initial: T) => ({
    store: writable(initial),
    initialized: Promise.resolve(),
  }),
}))

// Import the mocked store instances so tests can drive them.
const fineTuneStore = await import("$lib/stores/fine_tune_store")

const Page = (await import("./+page.svelte")).default

// onMount doesn't fire automatically under this test setup, so capture and run
// its callbacks manually (the page loads the task in onMount).
async function renderPage() {
  const onMountCallbacks: Array<() => unknown> = []
  const spy = vi
    .spyOn(svelteMod, "onMount")
    .mockImplementation((fn: () => unknown) => {
      onMountCallbacks.push(fn)
    })
  const result = render(Page)
  spy.mockRestore()
  for (const cb of onMountCallbacks) await cb()
  await tick()
  await new Promise((r) => setTimeout(r, 0))
  await tick()
  return result
}

describe("create-finetune +page.svelte", () => {
  beforeEach(() => {
    mockGoto.mockReset()
    mockLoadTask.mockReset()
    fineTuneStore.available_models_loading.set(false)
    fineTuneStore.available_models_error.set(null)
  })

  afterEach(() => cleanup())

  it("spins while available models are still loading", async () => {
    fineTuneStore.available_models_loading.set(true)
    mockLoadTask.mockResolvedValue({ id: "task1", name: "T" })
    const { container } = await renderPage()
    expect(container.querySelector(".loading-spinner")).not.toBeNull()
  })

  it("renders an error state (not an infinite spinner) when the task load fails", async () => {
    mockLoadTask.mockRejectedValue(new Error("task gone"))
    const { container } = await renderPage()
    expect(container.querySelector(".loading-spinner")).toBeNull()
    expect(container.textContent).toContain("Error Loading Task")
    expect(container.textContent).toContain("task gone")
  })

  it("shows the multi-turn notice for multi-turn tasks", async () => {
    mockLoadTask.mockResolvedValue({
      id: "task1",
      name: "T",
      turn_mode: "multiturn",
    })
    const { container } = await renderPage()
    expect(container.querySelector(".loading-spinner")).toBeNull()
    expect(container.textContent).toContain(
      "Fine-tuning is not supported for multi-turn tasks.",
    )
  })
})
