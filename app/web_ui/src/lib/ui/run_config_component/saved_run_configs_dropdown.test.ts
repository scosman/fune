// @vitest-environment jsdom
// The dropdown writes its own tooltip copy on the run page, and that write
// lands in the same prop a caller can supply. A caller that brings its own
// copy is explaining something the house text does not — which run config an
// eval is written against, say — so its text has to survive the options
// loading underneath it.
import { cleanup, render } from "@testing-library/svelte"
import { tick } from "svelte"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { model_info } from "$lib/stores"
import { run_configs_by_task_composite_id } from "$lib/stores/run_configs_store"
import { last_used_run_config_store } from "$lib/stores/last_used_run_config_store"
import type { ProviderModels, Task, TaskRunConfig } from "$lib/types"

vi.mock("$app/navigation", () => ({ goto: vi.fn() }))
// The control loads prompts and run configs on mount. Both are stubbed to
// no-ops so the test drives the stores directly and reaches no network — the
// stores themselves are the real ones, seeded below.
vi.mock("$lib/stores/prompts_store", async () => {
  const { writable } = await import("svelte/store")
  return {
    load_task_prompts: vi.fn(),
    prompts_by_task_composite_id: writable({}),
  }
})
vi.mock("$lib/stores/run_configs_store", async (import_original) => ({
  ...(await import_original<Record<string, unknown>>()),
  load_task_run_configs: vi.fn(),
}))

const SavedRunConfigsDropdownHarness = (
  await import("./__tests__/info_description_harness.svelte")
).default

const project_id = "project_1"
const current_task = { id: "task_1", name: "Task" } as Task
const composite_id = `${project_id}:${current_task.id}`

const CALLER_COPY = "The run config this eval tests."
const HOUSE_COPY_WITH_SAVED_CONFIGS =
  "Select a saved run configuration which includes model, prompt, tools and properties. Alternatively choose 'Custom' to manually configure this run."

function saved_config(id: string): TaskRunConfig {
  return {
    id,
    name: id,
    run_config_properties: {
      type: "kiln_agent",
      model_name: "gpt_5_4_mini",
      model_provider_name: "openai",
      prompt_id: "simple_prompt_builder",
      structured_output_mode: "default",
    },
  } as TaskRunConfig
}

beforeEach(() => {
  cleanup()
  // Set, so the mount's model-info load returns early rather than fetching.
  model_info.set({ models: [] } as unknown as ProviderModels)
  // No saved configs yet: the options below arrive after the first render,
  // which is the moment the default copy used to overwrite the caller's.
  run_configs_by_task_composite_id.set({})
  last_used_run_config_store.set({})
})

async function load_two_saved_configs() {
  run_configs_by_task_composite_id.set({
    [composite_id]: [saved_config("rc_1"), saved_config("rc_2")],
  })
  await tick()
  await tick()
}

describe("saved run configs dropdown tooltip", () => {
  it("keeps a caller's tooltip when the options load", async () => {
    const { getByTestId } = render(SavedRunConfigsDropdownHarness, {
      props: { project_id, current_task, info_description: CALLER_COPY },
    })
    await load_two_saved_configs()

    expect(getByTestId("tooltip_copy").textContent).toBe(CALLER_COPY)
  })

  it("still explains itself when the caller supplies no tooltip", async () => {
    const { getByTestId } = render(SavedRunConfigsDropdownHarness, {
      props: { project_id, current_task },
    })
    await load_two_saved_configs()

    expect(getByTestId("tooltip_copy").textContent).toBe(
      HOUSE_COPY_WITH_SAVED_CONFIGS,
    )
  })
})
