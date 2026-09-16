// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from "vitest"
import { render, cleanup, fireEvent } from "@testing-library/svelte"
import DataGuideOffer from "./data_guide_offer.svelte"

const { mockCapture } = vi.hoisted(() => ({ mockCapture: vi.fn() }))
vi.mock("posthog-js", () => ({ default: { capture: mockCapture } }))

afterEach(() => {
  cleanup()
  mockCapture.mockReset()
})

function setup(surface: "synth" | "builder" = "synth") {
  const on_set_up = vi.fn()
  const on_skip = vi.fn()
  const utils = render(DataGuideOffer, {
    props: { surface, on_set_up, on_skip },
  })
  return { ...utils, on_set_up, on_skip }
}

function button(container: HTMLElement, label: string): HTMLButtonElement {
  const found = Array.from(container.querySelectorAll("button")).find(
    (b) => b.textContent?.trim() === label,
  )
  if (!found) throw new Error(`no button labelled ${label}`)
  return found
}

describe("DataGuideOffer — the shared 'Create a Data Guide' offer", () => {
  it("renders the offer copy both surfaces share", () => {
    const { container } = setup()
    expect(container.textContent).toContain("Create a Data Guide")
    expect(container.textContent).toContain(
      "A Data Guide tells us what realistic inputs to your task look like. Without one, the model might guess.",
    )
    expect(container.textContent).toContain(
      "Add examples, rate the data we generate, and we'll refine the guide from there.",
    )
  })

  it("offers setup as the primary action and skipping as the secondary one", () => {
    const { container } = setup()
    expect(button(container, "Set Up Data Guide").className).toContain(
      "btn-primary",
    )
    expect(
      button(container, "Continue Without Data Guide").className,
    ).not.toContain("btn-primary")
  })

  it("routes Set Up to the surface's handler and reports which surface", async () => {
    const { container, on_set_up, on_skip } = setup("builder")
    await fireEvent.click(button(container, "Set Up Data Guide"))
    expect(on_set_up).toHaveBeenCalledTimes(1)
    expect(on_skip).not.toHaveBeenCalled()
    expect(mockCapture).toHaveBeenCalledWith("data_guide_intro_clicked", {
      choice: "set_up",
      surface: "builder",
    })
  })

  it("routes Continue Without to the surface's skip handler", async () => {
    const { container, on_set_up, on_skip } = setup("synth")
    await fireEvent.click(button(container, "Continue Without Data Guide"))
    expect(on_skip).toHaveBeenCalledTimes(1)
    expect(on_set_up).not.toHaveBeenCalled()
    expect(mockCapture).toHaveBeenCalledWith("data_guide_intro_clicked", {
      choice: "skip",
      surface: "synth",
    })
  })
})
