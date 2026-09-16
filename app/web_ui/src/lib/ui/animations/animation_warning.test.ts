// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest"
import { render, cleanup } from "@testing-library/svelte"
import AnalyzingAnimation from "./analyzing_animation.svelte"
import ConversationAnimation from "./conversation_animation.svelte"

afterEach(() => cleanup())

const WARNING_TEXT = "This is taking longer than usual."

// The animations hand their warning line to the Warning control and let it
// decide how the line looks. Pinned at the control's defaults so a caller-side
// override cannot creep back in and make one warning read unlike the rest.
function warning_parts(container: HTMLElement) {
  const text_wrapper = Array.from(container.querySelectorAll("div")).find(
    (el) => el.textContent?.trim() === WARNING_TEXT && el.children.length === 0,
  )
  return { text_wrapper, outer: text_wrapper?.parentElement }
}

const animations = [
  ["analyzing_animation", AnalyzingAnimation],
  ["conversation_animation", ConversationAnimation],
] as const

describe.each(animations)("%s — the warning line", (_name, Animation) => {
  it("renders the warning at the Warning control's own defaults", () => {
    const { container } = render(Animation, {
      props: {
        title: "Working",
        description: "Doing the thing",
        warning: WARNING_TEXT,
      },
    })
    const { outer, text_wrapper } = warning_parts(container)
    expect(text_wrapper).toBeTruthy()
    // The control's default text size, not the caller's "base".
    expect(outer!.className).toContain("text-sm")
    expect(outer!.className).not.toContain("text-base")
    // The control's default icon-to-text gap, not the caller's inline 4px.
    expect(text_wrapper!.className).toContain("pl-4")
    expect(text_wrapper!.className).not.toContain("pl-1")
  })

  it("renders no warning line when there is no warning", () => {
    const { container } = render(Animation, {
      props: {
        title: "Working",
        description: "Doing the thing",
        warning: null,
      },
    })
    expect(container.textContent).not.toContain(WARNING_TEXT)
    expect(container.querySelector(".pl-4")).toBeNull()
  })
})
