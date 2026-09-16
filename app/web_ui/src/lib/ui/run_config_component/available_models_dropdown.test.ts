// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest"
import { render, cleanup } from "@testing-library/svelte"
import { get, writable } from "svelte/store"
import { tick } from "svelte"
import type { AvailableModels, ModelDetails } from "$lib/types"

// The dropdown reaches into the model stores on mount. Stubbing them keeps the
// test on the one thing it is about: which suggested-model advisory renders.
const mock_available_models = writable<AvailableModels[]>([])
const mock_model_info = writable({})
const mock_ui_state = writable({ selected_model: null as string | null })
const mock_recent_model_store = writable([])

const { mock_available_model_details } = vi.hoisted(() => ({
  mock_available_model_details: vi.fn(),
}))

vi.mock("$lib/stores", () => ({
  available_models: mock_available_models,
  load_available_models: vi.fn().mockResolvedValue(undefined),
  available_model_details: mock_available_model_details,
  ui_state: mock_ui_state,
  provider_name_from_id: (id: string) => id,
  model_name: (id: string) => id,
  model_info: mock_model_info,
  load_model_info: vi.fn().mockResolvedValue(undefined),
}))

vi.mock("$lib/stores/recent_model_store", () => ({
  recent_model_store: mock_recent_model_store,
  addRecentModel: vi.fn(),
}))

const AvailableModelsDropdown = (
  await import("./available_models_dropdown.svelte")
).default

// Each suggested_mode, the ModelDetails flag that makes a model "suggested" in
// it, and the sentence that mode's advisory shows.
const MODES = [
  {
    suggested_mode: "data_gen",
    flag: "suggested_for_data_gen",
    message: `For data gen we suggest using one of the models marked "Recommended" in the dropdown.`,
  },
  {
    suggested_mode: "uncensored_data_gen",
    flag: "suggested_for_uncensored_data_gen",
    message: `For this data gen template we suggest using one of the models marked "Recommended" in the dropdown.`,
  },
  {
    suggested_mode: "evals",
    flag: "suggested_for_evals",
    message: `For evals we suggest using one of the models marked "Recommended" in the dropdown.`,
  },
  {
    suggested_mode: "doc_extraction",
    flag: "suggested_for_doc_extraction",
    message: `For doc extraction we suggest using one of the models marked "Recommended" in the dropdown.`,
  },
  {
    suggested_mode: "synthetic_user",
    flag: "suggested_for_synthetic_user",
    message: `For the simulated user we suggest using one of the models marked "Recommended" in the dropdown.`,
  },
] as const

// One provider entry: enough for the component to read the model list as
// loaded, which is what makes a model's suggested-for-X flags meaningful.
const LOADED_MODELS = [
  {
    provider_name: "OpenAI",
    provider_id: "openai",
    models: [{ id: "gpt-4o", name: "GPT-4o" }],
  },
] as unknown as AvailableModels[]

beforeEach(() => {
  mock_available_model_details.mockReset()
  // The default for these tests: the list has arrived.
  mock_available_models.set(LOADED_MODELS)
  mock_ui_state.set({ selected_model: null })
})

afterEach(cleanup)

// State of the world for one render: is a model chosen, and is it suggested for
// the mode under test.
function set_selection(suggested_flag: string, is_suggested: boolean) {
  mock_available_model_details.mockImplementation(
    (model_id: string | null): ModelDetails | null => {
      // No model chosen, or the list has not arrived yet: there are no details
      // to read, which is exactly why a loading dropdown cannot tell a
      // suggested model from an unsuggested one.
      if (!model_id || get(mock_available_models).length === 0) {
        return null
      }
      return { [suggested_flag]: is_suggested } as unknown as ModelDetails
    },
  )
}

async function render_dropdown(props: Record<string, unknown>) {
  const utils = render(AvailableModelsDropdown, { props })
  await tick()
  return utils
}

describe.each(MODES)(
  "suggested advisory — $suggested_mode",
  ({ suggested_mode, flag, message }) => {
    it("renders all three states", async () => {
      set_selection(flag, false)
      const no_model = await render_dropdown({
        model: null,
        settings: { suggested_mode },
      })
      expect(no_model.container.textContent).toContain(message)
      cleanup()

      set_selection(flag, true)
      const suggested = await render_dropdown({
        model: "openai/gpt-4o",
        settings: { suggested_mode },
      })
      expect(suggested.container.textContent).toContain(message)
      // The confirming state, identified by its own color rather than its text.
      expect(suggested.container.querySelector(".text-success")).not.toBeNull()
      cleanup()

      set_selection(flag, false)
      const not_suggested = await render_dropdown({
        model: "openai/gpt-4o",
        settings: { suggested_mode },
      })
      expect(not_suggested.container.textContent).toContain(message)
      expect(
        not_suggested.container.querySelector(".text-warning"),
      ).not.toBeNull()
    })

    it("shows nothing for a chosen model while the list is loading", async () => {
      // Before the list lands, a suggested model reads exactly like an
      // unsuggested one, so rendering the amber note here only to swap it for
      // the green check once the list arrives moves everything below it. A
      // restored model that IS suggested is the common case.
      mock_available_models.set([])
      set_selection(flag, true)
      const { container } = await render_dropdown({
        model: "openai/gpt-4o",
        settings: { suggested_mode },
      })
      expect(container.textContent).not.toContain(message)
      expect(container.querySelector(".text-warning")).toBeNull()
    })
  },
)
