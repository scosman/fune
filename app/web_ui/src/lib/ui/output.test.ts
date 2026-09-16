// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest"
import { render, cleanup } from "@testing-library/svelte"
import Output from "./output.svelte"
import OutputSlotHarness from "./output_slot_harness.test.svelte"

afterEach(() => cleanup())

// The surface Output draws when it prints its own text. Pinned so the slot
// branch added alongside it cannot change what every existing caller renders.
describe("output — no slot", () => {
  it("prints raw_output in its own pre, inside the panel", () => {
    const { container } = render(Output, {
      props: { raw_output: "hello world" },
    })

    const pre = container.querySelector("pre")
    expect(pre).not.toBeNull()
    expect(pre!.textContent).toBe("hello world")
    // The pre is the growing child of the panel, and the panel paints base-200.
    expect(pre!.className).toContain("grow")
    expect(pre!.className).toContain("text-xs")
    expect(pre!.parentElement!.className).toContain("bg-base-200")
    // One copy button, beside the content.
    expect(container.querySelectorAll("button").length).toBe(1)
  })

  it("marks a cited span in the printed text", () => {
    const { container } = render(Output, {
      props: { raw_output: "abcdef", mark: { start: 2, end: 4 } },
    })

    const marked = container.querySelector("[data-highlight-target]")
    expect(marked!.textContent).toBe("cd")
  })
})

describe("output — slotted content", () => {
  it("renders the slot inside the same surface, and keeps the copy button", () => {
    const { container } = render(OutputSlotHarness, {
      props: { raw_output: "the plain text" },
    })

    // The slot replaces the printed text: no pre on this path.
    expect(container.querySelector("pre")).toBeNull()
    const slotted = container.querySelector("[data-slotted]")
    expect(slotted).not.toBeNull()
    expect(slotted!.textContent).toBe("rendered content")
    // Still inside Output's panel, and the copy button is still there.
    expect(slotted!.closest(".bg-base-200")).not.toBeNull()
    expect(container.querySelectorAll("button").length).toBe(1)
  })
})

describe("output — the slot inside the control's chrome", () => {
  it("folds long slotted content the way it folds long text", async () => {
    const { container } = render(OutputSlotHarness, {
      props: { raw_output: "the plain text", max_height: "40px" },
    })
    // The fold measures the slot the same way it measures the printed text:
    // the content element is the slotted wrapper, so a caller's own rendering
    // gets the Show All affordance rather than silently overflowing.
    const panel = container.querySelector("[style*='max-height']")
    expect(panel).not.toBeNull()
    expect(
      container
        .querySelector("[data-slotted]")!
        .closest("[style*='max-height']"),
    ).toBe(panel)
  })

  it("honours no_padding on the slot", () => {
    const padded = render(OutputSlotHarness, {
      props: { raw_output: "t" },
    })
    const padded_box =
      padded.container.querySelector("[data-slotted]")!.parentElement!
    expect(padded_box.className).toContain("p-3")
    cleanup()

    const bare = render(OutputSlotHarness, {
      props: { raw_output: "t", no_padding: true },
    })
    const bare_box =
      bare.container.querySelector("[data-slotted]")!.parentElement!
    expect(bare_box.className).not.toContain("p-3")
  })
})
