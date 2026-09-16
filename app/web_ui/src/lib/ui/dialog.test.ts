// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest"
import { render, cleanup } from "@testing-library/svelte"
import Dialog from "./dialog.svelte"

afterEach(cleanup)

// The width prop is the dialog's only sizing control, and each value maps to
// one Tailwind max-width. Tailwind purges classes it can't see in the source,
// so the widths are also force-listed in a hidden div in the component.
function modal_box_class(width?: "normal" | "wide" | "extra_wide"): string {
  const props = width ? { title: "Trace", width } : { title: "Trace" }
  const { container } = render(Dialog, { props })
  const box = container.querySelector(".modal-box")
  if (!box) throw new Error("no modal-box rendered")
  return box.className
}

describe("Dialog width", () => {
  it("caps a normal dialog at the daisyUI default", () => {
    const className = modal_box_class()
    expect(className).not.toContain("w-11/12")
    expect(className).not.toContain("max-w-3xl")
    expect(className).not.toContain("max-w-7xl")
  })

  it("caps a wide dialog at 3xl", () => {
    const className = modal_box_class("wide")
    expect(className).toContain("w-11/12")
    expect(className).toContain("max-w-3xl")
  })

  it("caps an extra wide dialog at 7xl", () => {
    const className = modal_box_class("extra_wide")
    expect(className).toContain("w-11/12")
    expect(className).toContain("max-w-7xl")
    expect(className).not.toContain("max-w-3xl")
  })

  it("keeps every width class in the source for the compiler", () => {
    const { container } = render(Dialog, { props: { title: "Trace" } })
    const forced = container.querySelector(".modal-box > .hidden")
    expect(forced?.className).toContain("max-w-3xl")
    expect(forced?.className).toContain("max-w-7xl")
  })
})
