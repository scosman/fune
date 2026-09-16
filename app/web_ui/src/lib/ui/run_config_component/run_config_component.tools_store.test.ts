// @vitest-environment jsdom
// The tool and skill pickers inside this component mirror their selection into
// the app-wide tools/skills stores, keyed by task id — a parent's write counts
// the same as a user's click. Callers whose tools belong to something other
// than the task (a generation lane, say) pass no task so that round-trip never
// happens. These tests pin that: no task means the stores are neither read nor
// written, while the component itself still reports the tools it was given.
import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest"
import { render } from "@testing-library/svelte"
import { tick } from "svelte"
import { get } from "svelte/store"
import { available_tools } from "$lib/stores"
import { tools_store } from "$lib/stores/tools_store"
import { skills_store } from "$lib/stores/skills_store"
import type {
  KilnAgentRunConfigProperties,
  ToolSetApiDescription,
} from "$lib/types"
import { isKilnAgentRunConfig } from "$lib/types"

vi.mock("$app/navigation", () => ({ goto: vi.fn() }))

// The advanced panel is not what these tests are about, and its input-transform
// modal pulls a large dialog subtree into jsdom.
vi.mock("./advanced_run_options.svelte", async () => {
  const { default: Stub } = await import("./__tests__/empty_stub.svelte")
  return { default: Stub }
})

const RunConfigComponent = (await import("./run_config_component.svelte"))
  .default

beforeAll(() => {
  if (typeof globalThis.ResizeObserver === "undefined") {
    // eslint-disable-next-line @typescript-eslint/no-extraneous-class
    class ResizeObserverStub {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    ;(
      globalThis as unknown as { ResizeObserver: typeof ResizeObserverStub }
    ).ResizeObserver = ResizeObserverStub
  }
})

const project_id = "project_1"

const search_tools: ToolSetApiDescription = {
  type: "mcp",
  set_name: "Search",
  tools: [
    {
      id: "mcp::search",
      name: "Search",
      description: "Search the web",
      function_name: "search",
    },
  ],
}

// What another surface (Run, Synthetic Data) left in the stores for this task.
const OTHER_SURFACE_TOOLS = { task_1: ["mcp::search"] }
const OTHER_SURFACE_SKILLS = { task_1: ["skill::triage"] }

const config_with_tools: KilnAgentRunConfigProperties = {
  type: "kiln_agent",
  model_name: "gpt_5_4_mini",
  model_provider_name: "openai",
  prompt_id: "simple_prompt_builder",
  temperature: 1.0,
  top_p: 1.0,
  structured_output_mode: "default",
  thinking_level: null,
  input_transform: null,
  tools_config: { tools: ["mcp::search"] },
}

beforeEach(() => {
  available_tools.set({ [project_id]: [search_tools] })
  tools_store.set({ selected_tool_ids_by_task_id: { ...OTHER_SURFACE_TOOLS } })
  skills_store.set({
    selected_skill_ids_by_task_id: { ...OTHER_SURFACE_SKILLS },
  })
})

// Render the component the way a generation lane does: no task, so the tool
// pickers never touch the shared stores.
async function render_taskless_lane() {
  const { component } = render(RunConfigComponent, {
    props: {
      project_id,
      show_name_field: false,
      hide_prompt_selector: true,
      show_tools_selector_in_advanced: true,
    },
  })
  await new Promise((resolve) => setTimeout(resolve, 0))
  await tick()
  return component
}

describe("RunConfigComponent without a task", () => {
  it("leaves the shared stores alone when a config's tools are applied", async () => {
    const component = await render_taskless_lane()
    component.apply_run_config_properties(config_with_tools)
    await new Promise((resolve) => setTimeout(resolve, 0))
    await tick()
    expect(get(tools_store).selected_tool_ids_by_task_id).toEqual(
      OTHER_SURFACE_TOOLS,
    )
    expect(get(skills_store).selected_skill_ids_by_task_id).toEqual(
      OTHER_SURFACE_SKILLS,
    )
  })

  it("still reports the tools it was given", async () => {
    // The lane has to work, not just be harmless: what it hands back is what
    // the caller sends to the server.
    const component = await render_taskless_lane()
    component.apply_run_config_properties(config_with_tools)
    await tick()
    const properties = component.run_options_as_run_config_properties()
    expect(isKilnAgentRunConfig(properties)).toBe(true)
    if (!isKilnAgentRunConfig(properties)) return
    expect(properties.tools_config?.tools).toEqual(["mcp::search"])
  })

  it("does not seed itself from another surface's tools", async () => {
    // Without this the first open would silently inherit whatever Run last
    // used, and mint eval data under a config the user never chose.
    const component = await render_taskless_lane()
    const properties = component.run_options_as_run_config_properties()
    expect(isKilnAgentRunConfig(properties)).toBe(true)
    if (!isKilnAgentRunConfig(properties)) return
    expect(properties.tools_config?.tools).toEqual([])
  })

  it("clears an abandoned visit's tools when the options are reset", async () => {
    const component = await render_taskless_lane()
    component.apply_run_config_properties(config_with_tools)
    await tick()
    component.reset_run_options()
    await new Promise((resolve) => setTimeout(resolve, 0))
    await tick()
    const properties = component.run_options_as_run_config_properties()
    expect(isKilnAgentRunConfig(properties)).toBe(true)
    if (!isKilnAgentRunConfig(properties)) return
    expect(properties.tools_config?.tools).toEqual([])
    expect(properties.temperature).toBe(1.0)
    expect(properties.top_p).toBe(1.0)
    // The reset must not reach the shared stores either.
    expect(get(tools_store).selected_tool_ids_by_task_id).toEqual(
      OTHER_SURFACE_TOOLS,
    )
  })
})
