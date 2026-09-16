// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest"
import { render, cleanup, fireEvent } from "@testing-library/svelte"
import KilnProPromptsTable from "./kiln_pro_prompts_table.svelte"

afterEach(cleanup)

function setup(props: Record<string, unknown> = {}) {
  const utils = render(KilnProPromptsTable, {
    // The batch plan always passes these; its defaults are the values below.
    props: {
      prompts: ["first prompt", "second prompt"],
      items_label: "Dataset Items",
      expanded_description: null,
      column_label: "Prompt",
      ...props,
    },
  })
  const toggle = utils.container.querySelector("button") as HTMLButtonElement
  return { ...utils, toggle }
}

function header_text(container: HTMLElement): string {
  return (
    container.querySelector("span.text-sm.font-medium")?.textContent?.trim() ??
    ""
  )
}

// The sentence lives in the header's own text column, alongside the title —
// scoped so the chevron's wrapper (same text classes) can't be mistaken for it.
function description_text(container: HTMLElement): string | null {
  const node = container.querySelector(
    "div.flex.flex-col.gap-2 > div.text-sm.text-gray-500",
  )
  return node ? node.textContent?.trim() ?? "" : null
}

describe("toggle", () => {
  it("starts collapsed with the header and toggle label", () => {
    const { container, toggle } = setup()
    expect(header_text(container)).toBe("All Dataset Items (2)")
    expect(toggle.getAttribute("aria-label")).toBe("Show dataset items")
    expect(toggle.getAttribute("aria-expanded")).toBe("false")
  })

  it("shows the description only once expanded", async () => {
    const { container, toggle } = setup()
    expect(description_text(container)).toBeNull()
    await fireEvent.click(toggle)
    expect(description_text(container)).toBe(
      "Each prompt below will be used to guide one dataset sample.",
    )
    expect(toggle.getAttribute("aria-label")).toBe("Hide dataset items")
    expect(toggle.getAttribute("aria-expanded")).toBe("true")
  })
})

// The rows table only mounts once expanded, so its header is only readable
// after the toggle is clicked.
function column_header_text(container: HTMLElement): string {
  return container.querySelector("thead th")?.textContent?.trim() ?? ""
}

describe("column_label", () => {
  it("forwards a caller's header down to the rows table", async () => {
    const { container, toggle } = setup({ column_label: "Item Guidance" })
    await fireEvent.click(toggle)
    expect(column_header_text(container)).toBe("Item Guidance")
  })
})

describe("items_label", () => {
  it("drives the header and the aria-label from the one noun", async () => {
    // The point of a single prop: a screen reader can never announce a
    // different noun than the one on screen.
    const { container, toggle } = setup({ items_label: "Items" })
    expect(header_text(container)).toBe("All Items (2)")
    expect(toggle.getAttribute("aria-label")).toBe("Show items")
    await fireEvent.click(toggle)
    expect(toggle.getAttribute("aria-label")).toBe("Hide items")
  })

  it("lowercases a multi-word noun for the aria-label only", () => {
    const { container, toggle } = setup({ items_label: "Test Inputs" })
    expect(header_text(container)).toBe("All Test Inputs (2)")
    expect(toggle.getAttribute("aria-label")).toBe("Show test inputs")
  })
})

describe("expanded_description", () => {
  it("renders a caller's sentence in place of the default", async () => {
    const { container, toggle } = setup({
      expanded_description: "One line per test conversation.",
    })
    await fireEvent.click(toggle)
    expect(description_text(container)).toBe("One line per test conversation.")
  })

  it("renders no description at all when false", async () => {
    const { container, toggle } = setup({ expanded_description: false })
    await fireEvent.click(toggle)
    expect(description_text(container)).toBeNull()
    // The rows still expand — false suppresses the sentence, not the table.
    expect(container.textContent).toContain("first prompt")
  })
})
