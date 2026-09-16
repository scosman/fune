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
    url: new URL("http://localhost/prompt_optimization/proj1/task1"),
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
vi.mock("$lib/stores", () => ({ load_task: mockLoadTask }))
vi.mock("$lib/agent", () => ({ agentInfo: { set: vi.fn() } }))
vi.mock("$lib/utils/copilot_utils", () => ({
  checkKilnCopilotAvailable: vi.fn().mockResolvedValue(true),
}))
vi.mock("$lib/utils/entitlement_utils", () => ({
  checkPromptOptimizationAccess: vi
    .fn()
    .mockResolvedValue({ has_access: true, error: null }),
}))

const Page = (await import("./+page.svelte")).default

async function settle() {
  await tick()
  await new Promise((r) => setTimeout(r, 0))
  await tick()
}

const one_job = [
  {
    id: "job1",
    name: "Job One",
    latest_status: "succeeded",
    created_at: "2020-01-01",
  },
]

describe("prompt-optimization list +page.svelte", () => {
  beforeEach(() => {
    mockGoto.mockReset()
    mockLoadTask.mockReset()
    mockClientGET.mockReset()
    mockClientGET.mockResolvedValue({ data: one_job, error: null })
  })

  afterEach(() => cleanup())

  it("renders an error state and never the single-turn UI when the task load fails", async () => {
    mockLoadTask.mockRejectedValue(new Error("task gone"))
    const { container } = render(Page)
    await settle()
    expect(container.querySelector(".loading-spinner")).toBeNull()
    expect(container.textContent).toContain("task gone")
    // The single-turn jobs table must not render on failure.
    expect(container.textContent).not.toContain("Job One")
    expect(container.textContent).not.toContain("Create Optimized Prompt")
  })

  it("shows the multi-turn warning (not the single-turn UI) for multi-turn tasks", async () => {
    mockLoadTask.mockResolvedValue({
      id: "task1",
      name: "T",
      turn_mode: "multiturn",
    })
    const { container } = render(Page)
    await settle()
    expect(container.querySelector(".loading-spinner")).toBeNull()
    expect(container.textContent).toContain(
      "Prompt optimization is not supported for multi-turn tasks.",
    )
    expect(container.textContent).not.toContain("Job One")
  })

  it("renders the jobs table for a single-turn task", async () => {
    mockLoadTask.mockResolvedValue({ id: "task1", name: "T" })
    const { container } = render(Page)
    await settle()
    expect(container.querySelector(".loading-spinner")).toBeNull()
    expect(container.textContent).toContain("Job One")
    expect(container.textContent).not.toContain(
      "Prompt optimization is not supported for multi-turn tasks.",
    )
  })
})
