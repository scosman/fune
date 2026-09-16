// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { render, cleanup } from "@testing-library/svelte"
import { tick } from "svelte"

// ---------------------------------------------------------------------------
// Module-level mocks -- must come before the dynamic page import
// ---------------------------------------------------------------------------

const { mockPage, mockGoto, mockClientGET, mockLoadTask, defaultPageValue } =
  vi.hoisted(() => {
    type PageValue = {
      params: Record<string, string>
      url: URL
    }
    const defaultPageValue = (): PageValue => ({
      params: { project_id: "proj1", task_id: "task1" },
      url: new URL("http://localhost/generate/proj1/task1"),
    })
    let pageValue: PageValue = defaultPageValue()
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
    const mockClientGET = vi.fn()
    const mockLoadTask = vi.fn()

    return { mockPage, mockGoto, mockClientGET, mockLoadTask, defaultPageValue }
  })

vi.mock("$app/stores", () => ({
  page: mockPage,
}))

vi.mock("$app/navigation", () => ({
  goto: mockGoto,
}))

vi.mock("$lib/api_client", () => ({
  client: {
    GET: mockClientGET,
  },
}))

vi.mock("$lib/stores", () => ({
  load_task: mockLoadTask,
}))

vi.mock("$lib/agent", () => ({
  agentInfo: { set: vi.fn() },
}))

// IndexedDB isn't available in jsdom; back the saved-session store with an
// in-memory writable that reports "no ongoing session".
vi.mock("$lib/stores/index_db_store", async () => {
  const { writable } = await import("svelte/store")
  return {
    indexedDBStore: <T>(_key: string, initial: T) => ({
      store: writable(initial),
      initialized: Promise.resolve(),
    }),
  }
})

// Q&A store with no saved documents, so routing stays on this page.
vi.mock("./qna/qna_ui_store", () => ({
  createQnaStore: () => ({
    init: vi.fn().mockResolvedValue(undefined),
    subscribe: (fn: (value: { documents: unknown[] }) => void) => {
      fn({ documents: [] })
      return () => {}
    },
  }),
}))

// Stub the heavy intro component; these tests only care about routing state.
vi.mock("./data_gen_intro.svelte", async () => {
  const Stub = await import("./__tests__/data_gen_intro_stub.svelte")
  return { default: Stub.default }
})

// Dynamic import after all mocks
const Page = (await import("./+page.svelte")).default

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Flush pending promises and Svelte updates. */
async function settle() {
  await tick()
  await new Promise((r) => setTimeout(r, 0))
  await tick()
}

const single_turn_task = { id: "task1", name: "Test Task" }

describe("generate landing +page.svelte routing", () => {
  beforeEach(() => {
    mockGoto.mockReset()
    mockLoadTask.mockReset()
    mockClientGET.mockReset()
    // No saved data guide by default
    mockClientGET.mockResolvedValue({ data: null, error: null })
    mockPage.set(defaultPageValue())
  })

  afterEach(() => {
    cleanup()
  })

  it("shows a spinner while the task is loading", async () => {
    mockLoadTask.mockReturnValue(new Promise(() => {})) // never resolves
    const { container } = render(Page)
    await tick()
    expect(container.querySelector(".loading-spinner")).not.toBeNull()
  })

  it("shows the intro for a single-turn task with no saved session", async () => {
    mockLoadTask.mockResolvedValue(single_turn_task)
    const { container } = render(Page)
    await settle()
    expect(container.querySelector(".loading-spinner")).toBeNull()
    expect(
      container.querySelector("[data-testid='data-gen-intro-stub']"),
    ).not.toBeNull()
    expect(mockGoto).not.toHaveBeenCalled()
  })

  it("shows the multi-turn notice instead of the intro for multi-turn tasks", async () => {
    mockLoadTask.mockResolvedValue({
      ...single_turn_task,
      turn_mode: "multiturn",
    })
    const { container } = render(Page)
    await settle()
    expect(container.querySelector(".loading-spinner")).toBeNull()
    expect(container.textContent).toContain(
      "Synthetic data generation is not supported for multi-turn tasks.",
    )
    expect(
      container.querySelector("[data-testid='data-gen-intro-stub']"),
    ).toBeNull()
  })

  it("renders an error state (not an infinite spinner) when the task load fails", async () => {
    mockLoadTask.mockRejectedValue(new Error("network down"))
    const { container } = render(Page)
    await settle()
    expect(container.querySelector(".loading-spinner")).toBeNull()
    expect(container.textContent).toContain("Error Loading Task")
    expect(container.textContent).toContain("network down")
    expect(
      container.querySelector("[data-testid='data-gen-intro-stub']"),
    ).toBeNull()
  })

  it("does not hot-loop retrying a failing load", async () => {
    mockLoadTask.mockRejectedValue(new Error("network down"))
    render(Page)
    await settle()
    await settle()
    // The failed key stays marked as handled: exactly one load attempt.
    expect(mockLoadTask).toHaveBeenCalledTimes(1)
  })

  it("retries the load on a subsequent navigation after a failure", async () => {
    mockLoadTask.mockRejectedValueOnce(new Error("network down"))
    const { container } = render(Page)
    await settle()
    expect(container.textContent).toContain("Error Loading Task")

    // Navigating to a different routing key re-runs routing and recovers.
    mockLoadTask.mockResolvedValue(single_turn_task)
    mockPage.set({
      params: { project_id: "proj1", task_id: "task2" },
      url: new URL("http://localhost/generate/proj1/task2"),
    })
    await settle()
    expect(container.textContent).not.toContain("Error Loading Task")
    expect(
      container.querySelector("[data-testid='data-gen-intro-stub']"),
    ).not.toBeNull()
  })
})
