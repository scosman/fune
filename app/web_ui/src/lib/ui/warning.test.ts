// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest"
import { render, cleanup } from "@testing-library/svelte"
import type { ComponentProps } from "svelte"
import Warning from "./warning.svelte"

afterEach(() => cleanup())

// The control owns its mark: there is one exclaim path, and callers colour it
// rather than redrawing it. Pinned so the variant that was removed cannot
// reappear as a prop.
describe("warning — the exclaim mark", () => {
  it("draws the ring even when a caller asks for the filled variant", () => {
    // The variant is gone, so the prop is inert: a caller still passing it
    // gets the one mark the control draws. Passing it is what makes this fail
    // if the variant ever comes back.
    // The cast is the assertion's point: the prop is gone from the type, and
    // a caller still passing it must get the ring rather than a second mark.
    const { container } = render(Warning, {
      props: {
        warning_message: "Something to say",
        warning_color: "error",
        filled_icon: true,
      } as unknown as ComponentProps<Warning>,
    })
    const paths = container.querySelectorAll("svg path")
    expect(paths).toHaveLength(1)
    // The ring: an outer circle with the bar and dot cut into it. A solid
    // disc would need fill-rule="evenodd" to knock them back out.
    expect(paths[0].getAttribute("fill-rule")).toBeNull()
  })

  it("colours the mark, not the text", () => {
    const { container } = render(Warning, {
      props: { warning_message: "Something to say", warning_color: "error" },
    })
    expect(container.querySelector("svg")!.className.baseVal).toContain(
      "text-error",
    )
    // The message is always the secondary text colour.
    expect(container.querySelector("div")!.className).toContain("text-gray-500")
  })
})
