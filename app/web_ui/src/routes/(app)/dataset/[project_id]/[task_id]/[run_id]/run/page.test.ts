// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { render, cleanup } from "@testing-library/svelte"
import { tick } from "svelte"
import { writable } from "svelte/store"
import { stubState, resetStubState } from "./__tests__/stub_state"

// ---------------------------------------------------------------------------
// Module-level mocks -- must come before the dynamic page import
// ---------------------------------------------------------------------------

const { mockPage, setRunId, mockGoto, mockClientGET, mockLoadTask } =
  vi.hoisted(() => {
    type PageValue = {
      params: Record<string, string>
      url: URL
      state: Record<string, unknown>
    }
    let pageValue: PageValue = {
      params: { project_id: "proj1", task_id: "task1", run_id: "leaf1" },
      url: new URL("http://localhost/dataset/proj1/task1/leaf1/run"),
      state: {},
    }
    const subs = new Set<(v: PageValue) => void>()
    const mockPage = {
      subscribe(fn: (v: PageValue) => void) {
        subs.add(fn)
        fn(pageValue)
        return () => subs.delete(fn)
      },
    }
    const setRunId = (run_id: string) => {
      pageValue = {
        ...pageValue,
        params: { ...pageValue.params, run_id },
        url: new URL(`http://localhost/dataset/proj1/task1/${run_id}/run`),
      }
      subs.forEach((fn) => fn(pageValue))
    }
    return {
      mockPage,
      setRunId,
      mockGoto: vi.fn(),
      mockClientGET: vi.fn(),
      mockLoadTask: vi.fn(),
    }
  })

vi.mock("$app/stores", () => ({ page: mockPage }))
vi.mock("$app/navigation", () => ({ goto: mockGoto }))
vi.mock("$lib/api_client", () => ({ client: { GET: mockClientGET } }))
vi.mock("$lib/agent", () => ({ agentInfo: { set: vi.fn() } }))
vi.mock("$lib/stores", () => ({
  get_task_composite_id: (p: string, t: string) => `${p}__${t}`,
  load_task: mockLoadTask,
  model_name: (id: string) => id,
  model_info: writable(null),
  load_model_info: vi.fn(),
  prompt_name_from_id: () => null,
  provider_name_from_id: (id: string) => id,
  load_available_models: vi.fn(),
  load_available_tools: vi.fn(),
  available_tools: writable({}),
  current_task: writable(null),
}))
vi.mock("$lib/stores/prompts_store", () => ({
  prompts_by_task_composite_id: writable({}),
  load_task_prompts: vi.fn(),
}))
vi.mock("$lib/stores/tools_store", () => ({
  get_tools_property_info: () => ({ value: "None", links: [] }),
  get_tool_names_from_ids: () => [],
  get_tool_server_name: () => null,
  split_tool_and_skill_ids: () => ({ tool_ids: [], skill_ids: [] }),
}))

// Stub the heavy child components. ChatTrace + the composer capture their
// props into stub_state so the test can inspect truncation / drive the send.
vi.mock("$lib/ui/trace/chat_trace.svelte", async () => ({
  default: (await import("./__tests__/chat_trace_stub.svelte")).default,
}))
vi.mock("$lib/ui/conversation/multiturn_composer.svelte", async () => ({
  default: (await import("./__tests__/composer_stub.svelte")).default,
}))
vi.mock("$lib/ui/conversation/chat_thinking_loading.svelte", async () => ({
  default: (await import("./__tests__/noop_stub.svelte")).default,
}))
vi.mock(
  "$lib/ui/run_config_component/run_config_component.svelte",
  async () => ({
    default: (await import("./__tests__/noop_stub.svelte")).default,
  }),
)
vi.mock(
  "$lib/ui/run_config_component/saved_run_configs_dropdown.svelte",
  async () => ({
    default: (await import("./__tests__/noop_stub.svelte")).default,
  }),
)
vi.mock("$lib/ui/run_sidebar.svelte", async () => ({
  default: (await import("./__tests__/noop_stub.svelte")).default,
}))
vi.mock("$lib/ui/property_list.svelte", async () => ({
  default: (await import("./__tests__/noop_stub.svelte")).default,
}))

const Page = (await import("./+page.svelte")).default

async function settle() {
  await tick()
  await new Promise((r) => setTimeout(r, 0))
  await tick()
}

const multiturn_task = { id: "task1", name: "T", turn_mode: "multiturn" }

const leaf1_run = {
  id: "leaf1",
  input: "hi",
  parent_task_run_id: "turn1run",
  trace: [
    { role: "user", content: "hi" },
    { role: "assistant", content: "hello" },
    { role: "user", content: "more" },
    { role: "assistant", content: "world" },
  ],
  output: { source: { run_config: null, properties: {} } },
}

const newleaf_run = {
  id: "newleaf",
  input: "hi",
  parent_task_run_id: "turn1run",
  trace: [
    { role: "user", content: "hi" },
    { role: "assistant", content: "hello" },
    { role: "user", content: "branch msg" },
    { role: "assistant", content: "branched reply" },
  ],
  output: { source: { run_config: null, properties: {} } },
}

const chain_data = {
  chain: [
    { turn_index: 1, run_id: "turn1run", trace_start_index: 0 },
    { turn_index: 2, run_id: "leaf1", trace_start_index: 2 },
  ],
  chain_broken: false,
  has_children: false,
}

describe("dataset run +page.svelte — fork send lifecycle", () => {
  beforeEach(() => {
    resetStubState()
    mockGoto.mockReset()
    mockLoadTask.mockReset()
    mockClientGET.mockReset()
    setRunId("leaf1")
    mockLoadTask.mockResolvedValue(multiturn_task)
    // jsdom lacks these; the transcript-scroll path calls them.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    globalThis.requestAnimationFrame = ((cb: (t: number) => void) => {
      cb(0)
      return 0
    }) as any
    window.scrollTo = vi.fn() as unknown as typeof window.scrollTo
  })

  afterEach(() => cleanup())

  it("holds the fork-point truncation (and disables the composer) until the forked run loads", async () => {
    // Defer the forked-run load so we can inspect the in-flight window.
    // A holder object keeps the resolver's type across the closure assignment.
    const deferred: { resolve: (() => void) | null } = { resolve: null }
    mockClientGET.mockImplementation(
      (path: string, opts: { params?: { path?: { run_id?: string } } }) => {
        if (path.includes("/chain")) {
          return Promise.resolve({ data: chain_data, error: null })
        }
        const rid = opts?.params?.path?.run_id
        if (rid === "leaf1") {
          return Promise.resolve({ data: leaf1_run, error: null })
        }
        if (rid === "newleaf") {
          return new Promise((res) => {
            deferred.resolve = () => res({ data: newleaf_run, error: null })
          })
        }
        return Promise.resolve({ data: null, error: null })
      },
    )
    mockGoto.mockImplementation(async (url: string) => {
      const m = String(url).match(/\/([^/]+)\/run$/)
      if (m) setRunId(m[1])
    })

    const { container } = render(Page)
    await settle()

    // Initial multiturn render: append composer, no truncation.
    expect(
      container.querySelector("[data-testid='multiturn-layout']"),
    ).not.toBeNull()
    expect(
      container.querySelector("[data-testid='composer-append']"),
    ).not.toBeNull()
    expect(stubState.chatTrace?.truncate_at_trace_index ?? null).toBeNull()

    // Open a fork on turn 1's assistant message (creates a branch of turn 2).
    stubState.chatTrace?.on_fork?.("leaf1", 1)
    await settle()
    expect(
      container.querySelector("[data-testid='composer-fork']"),
    ).not.toBeNull()
    expect(stubState.chatTrace?.truncate_at_trace_index).toBe(2)

    // Begin the send: the loader appears and truncation is held.
    stubState.composers.fork.on_send_start?.("branch msg")
    await tick()
    expect(
      container.querySelector("[data-testid='multiturn-pending-response']"),
    ).not.toBeNull()
    expect(stubState.chatTrace?.truncate_at_trace_index).toBe(2)

    // Send resolves -> navigate to the new run. The forked run hasn't loaded
    // yet: the view must NOT snap back to the full untruncated conversation,
    // and the (now append) composer must stay disabled.
    await stubState.composers.fork.on_success?.("newleaf")
    await settle()
    expect(
      container.querySelector("[data-testid='composer-append']"),
    ).not.toBeNull()
    expect(stubState.chatTrace?.truncate_at_trace_index).toBe(2)
    expect(stubState.composers.append.busy).toBe(true)
    expect(
      container.querySelector("[data-testid='multiturn-pending-response']"),
    ).not.toBeNull()

    // Forked run finishes loading: truncation released, composer re-enabled.
    deferred.resolve?.()
    await settle()
    expect(stubState.chatTrace?.truncate_at_trace_index ?? null).toBeNull()
    expect(stubState.composers.append.busy).toBe(false)
    expect(
      container.querySelector("[data-testid='multiturn-pending-response']"),
    ).toBeNull()
  })
})
