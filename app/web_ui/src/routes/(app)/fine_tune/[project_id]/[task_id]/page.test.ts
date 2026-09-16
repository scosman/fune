// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { render, cleanup } from "@testing-library/svelte"
import { tick } from "svelte"

// ---------------------------------------------------------------------------
// Module-level mocks -- must come before the dynamic page import
// ---------------------------------------------------------------------------

const { mockPage, mockGoto, mockClientGET, mockLoadTask } = vi.hoisted(() => {
  type PageValue = { params: Record<string, string>; url: URL }
  const pageValue: PageValue = {
    params: { project_id: "proj1", task_id: "task1" },
    url: new URL("http://localhost/fine_tune/proj1/task1"),
  }
  const mockPage = {
    subscribe(fn: (v: PageValue) => void) {
      fn(pageValue)
      return () => {}
    },
  }
  return {
    mockPage,
    mockGoto: vi.fn(),
    mockClientGET: vi.fn(),
    mockLoadTask: vi.fn(),
  }
})

vi.mock("$app/stores", () => ({ page: mockPage }))
vi.mock("$app/navigation", () => ({ goto: mockGoto }))
vi.mock("$lib/api_client", () => ({ client: { GET: mockClientGET } }))
vi.mock("$lib/stores", () => ({
  page: mockPage,
  load_task: mockLoadTask,
  load_available_models: vi.fn(),
  provider_name_from_id: (id: string) => id,
}))
vi.mock("$lib/agent", () => ({ agentInfo: { set: vi.fn() } }))
// EmptyFinetune pulls in heavier deps we don't need for these branch tests.
vi.mock("./empty_finetune.svelte", async () => {
  const Stub = await import("./__tests__/empty_finetune_stub.svelte")
  return { default: Stub.default }
})

const Page = (await import("./+page.svelte")).default

async function settle() {
  await tick()
  await new Promise((r) => setTimeout(r, 0))
  await tick()
}

const single_turn_task = { id: "task1", name: "Test Task" }

describe("fine-tune list +page.svelte", () => {
  beforeEach(() => {
    mockGoto.mockReset()
    mockLoadTask.mockReset()
    mockClientGET.mockReset()
    mockClientGET.mockResolvedValue({ data: [], error: null })
  })

  afterEach(() => cleanup())

  it("spins while the task is still loading", async () => {
    mockLoadTask.mockReturnValue(new Promise(() => {})) // never resolves
    const { container } = render(Page)
    await settle()
    expect(container.querySelector(".loading-spinner")).not.toBeNull()
  })

  it("renders an error state (not an infinite spinner) when the task load fails", async () => {
    mockLoadTask.mockRejectedValue(new Error("task gone"))
    const { container } = render(Page)
    await settle()
    expect(container.querySelector(".loading-spinner")).toBeNull()
    expect(container.textContent).toContain("Error Loading Fine Tunes")
    expect(container.textContent).toContain("task gone")
  })

  it("does not let a task-load failure shadow the finetunes error", async () => {
    mockLoadTask.mockRejectedValue(new Error("task gone"))
    mockClientGET.mockResolvedValue({
      data: null,
      error: new Error("finetunes boom"),
    })
    const { container } = render(Page)
    await settle()
    expect(container.querySelector(".loading-spinner")).toBeNull()
    expect(container.textContent).toContain("finetunes boom")
  })

  it("shows the multi-turn notice for multi-turn tasks", async () => {
    mockLoadTask.mockResolvedValue({
      ...single_turn_task,
      turn_mode: "multiturn",
    })
    const { container } = render(Page)
    await settle()
    expect(container.querySelector(".loading-spinner")).toBeNull()
    expect(container.textContent).toContain(
      "Fine-tuning is not supported for multi-turn tasks.",
    )
  })

  it("shows the empty state for a single-turn task with no fine-tunes", async () => {
    mockLoadTask.mockResolvedValue(single_turn_task)
    const { container } = render(Page)
    await settle()
    expect(container.querySelector(".loading-spinner")).toBeNull()
    expect(
      container.querySelector("[data-testid='empty-finetune-stub']"),
    ).not.toBeNull()
  })
})
