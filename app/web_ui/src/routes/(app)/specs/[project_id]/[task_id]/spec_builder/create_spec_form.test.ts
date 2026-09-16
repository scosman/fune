// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from "vitest"
import { render, cleanup, fireEvent, waitFor } from "@testing-library/svelte"
import type { FieldConfig } from "../select_template/spec_templates"

vi.mock("$app/navigation", () => ({
  goto: vi.fn(),
  beforeNavigate: vi.fn(),
}))

// Import the component under test after the mocks are registered.
const CreateSpecForm = (await import("./create_spec_form.svelte")).default

const field_configs: FieldConfig[] = [
  {
    key: "behaviour",
    label: "Behaviour",
    description: "The behaviour to enforce.",
    required: true,
  },
]

function render_form(overrides: Record<string, unknown> = {}) {
  return render(CreateSpecForm, {
    props: {
      name: "My Eval",
      property_values: { behaviour: "Be concise" },
      initial_property_values: { behaviour: "Be concise" },
      field_configs,
      error: null,
      submitting: false,
      warn_before_unload: false,
      ...overrides,
    },
  })
}

afterEach(() => {
  cleanup()
})

describe("CreateSpecForm", () => {
  it("offers manual creation only", () => {
    const { getByText, queryByText } = render_form()
    expect(getByText("Create Eval")).toBeTruthy()
    // The Kiln Pro creation path was removed: neither its submit button nor
    // the "or Create Manually" escape hatch beside it should render.
    expect(queryByText("Create with Kiln Pro")).toBeNull()
    expect(queryByText("Create Manually")).toBeNull()
  })

  it("dispatches create_spec on submit", async () => {
    const on_create_spec = vi.fn()
    const { component, getByText } = render_form()
    component.$on("create_spec", on_create_spec)
    await fireEvent.click(getByText("Create Eval"))
    await waitFor(() => expect(on_create_spec).toHaveBeenCalled())
  })
})
