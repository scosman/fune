// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from "vitest"
import { render, cleanup, fireEvent } from "@testing-library/svelte"
import SynthDataGuide, {
  open_data_guide_in_new_tab,
} from "./synth_data_guide.svelte"

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function setup(props: Record<string, unknown> = {}) {
  return render(SynthDataGuide, {
    props: {
      project_id: "proj1",
      task_id: "task1",
      data_guide: "# Reference Inputs\n\nShort support questions.",
      use_data_guide: true,
      ...props,
    },
  })
}

// Svelte 4 keeps the live value of each prop in the instance context; this is
// what a parent's `bind:` reads back.
function bound_prop<T>(component: unknown, name: string): T {
  const instance = component as {
    $$: { ctx: unknown[]; props: Record<string, number> }
  }
  return instance.$$.ctx[instance.$$.props[name]] as T
}

describe("SynthDataGuide — the shared 'Use Data Guide' checkbox", () => {
  it("renders nothing when the task has no guide", () => {
    const { container } = setup({ data_guide: "" })
    expect(container.querySelector("#data_guide_toggle")).toBeNull()
    expect(container.textContent?.trim()).toBe("")
  })

  it("renders the checkbox with its label and description when a guide exists", () => {
    const { container } = setup()
    const box = container.querySelector(
      "#data_guide_toggle",
    ) as HTMLInputElement
    expect(box).not.toBeNull()
    expect(box.type).toBe("checkbox")
    expect(box.checked).toBe(true)
    expect(container.textContent).toContain("Use Data Guide")
    expect(container.textContent).toContain(
      "A saved description of what realistic inputs to this task look like",
    )
  })

  it("binds the on/off state outward", async () => {
    const { container, component } = setup()
    const box = container.querySelector(
      "#data_guide_toggle",
    ) as HTMLInputElement
    await fireEvent.click(box)
    expect(bound_prop<boolean>(component, "use_data_guide")).toBe(false)
  })

  it("View opens the saved guide in a new tab", async () => {
    const open = vi.spyOn(window, "open").mockImplementation(() => null)
    const { container } = setup()
    const view = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent?.trim() === "View",
    ) as HTMLButtonElement
    await fireEvent.click(view)
    expect(open).toHaveBeenCalledWith(
      "/generate/proj1/task1/data_guide",
      "_blank",
      "noopener,noreferrer",
    )
  })

  it("the exported opener refuses to open without both ids", () => {
    const open = vi.spyOn(window, "open").mockImplementation(() => null)
    open_data_guide_in_new_tab("", "task1")
    open_data_guide_in_new_tab("proj1", "")
    expect(open).not.toHaveBeenCalled()
  })
})
