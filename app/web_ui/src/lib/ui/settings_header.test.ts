// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest"
import { render, cleanup } from "@testing-library/svelte"
import SettingsHeader from "./settings_header.svelte"
import SettingsHeaderActionsHarness from "./settings_header_actions_harness.test.svelte"
import SettingsHeaderSubtitleHarness from "./settings_header_subtitle_harness.test.svelte"
import SettingsHeaderSubtitleActionsHarness from "./settings_header_subtitle_actions_harness.test.svelte"

afterEach(() => cleanup())

// What every caller without actions renders. Pinned so the actions slot added
// alongside it cannot change the header they already have.
describe("settings_header — no actions", () => {
  it("renders the title, and the subtitle only when given one", () => {
    const plain = render(SettingsHeader, { props: { title: "Overview" } })
    const rule = plain.container.querySelector("div")!
    expect(rule.className).toContain("border-b")
    expect(rule.querySelector("h2")!.textContent).toBe("Overview")
    expect(rule.querySelector("p")).toBeNull()
    // The title is the header's only child: no wrapper comes between them.
    expect(rule.firstElementChild!.tagName).toBe("H2")
    cleanup()

    const titled = render(SettingsHeader, {
      props: { title: "Overview", subtitle: "What happened here" },
    })
    expect(titled.container.querySelector("p")!.textContent).toBe(
      "What happened here",
    )
  })
})

// A screen that stacks several headers turns the rule off. The padding that
// cleared the rule goes with it, so the header does not leave a gap the
// parent stack did not ask for.
describe("settings_header — show_divider", () => {
  it("drops the rule and its padding, keeping everything else", () => {
    const { container } = render(SettingsHeader, {
      props: { title: "Overview", subtitle: "What happened here" },
    })
    const bare = container.querySelector("div")!
    expect(bare.className).toContain("border-b")
    expect(bare.className).toContain("pb-3")
    cleanup()

    const plain = render(SettingsHeader, {
      props: {
        title: "Overview",
        subtitle: "What happened here",
        show_divider: false,
      },
    })
    const rule = plain.container.querySelector("div")!
    expect(rule.className).not.toContain("border-b")
    expect(rule.className).not.toContain("pb-3")
    expect(rule.querySelector("h2")!.textContent).toBe("Overview")
    expect(rule.querySelector("p")!.textContent).toBe("What happened here")
  })
})

describe("settings_header — with actions", () => {
  it("puts the actions on the title's line, inside the header's own rule", () => {
    const { container } = render(SettingsHeaderActionsHarness, {
      props: { title: "Overview" },
    })
    const rule = container.querySelector(".border-b")!
    const action = container.querySelector("[data-action]")!
    // Inside the rule, so the rule still spans the full width.
    expect(action.closest(".border-b")).toBe(rule)
    // Beside the title, not under it: the title's box and the actions' box
    // are children of one row, and that row is the thing laying them out.
    const title_box = rule.querySelector("h2")!.parentElement!
    const action_box = action.parentElement!
    expect(action_box).not.toBe(title_box)
    expect(title_box.parentElement).toBe(action_box.parentElement)
    const row = title_box.parentElement!
    expect(row.className).toContain("flex")
    expect(row.className).toContain("justify-between")
  })
})

// The subtitle can also come in as a slot, for a caller whose line carries a
// link. Pinned so the string subtitle every other caller passes still renders
// the same paragraph, on both the plain and the actions branch.
describe("settings_header — the subtitle line", () => {
  it("renders a string subtitle in its own paragraph, with and without actions", () => {
    const plain = render(SettingsHeader, {
      props: { title: "Overview", subtitle: "What happened here" },
    })
    const plain_p = plain.container.querySelector("p")!
    expect(plain_p.className).toBe("text-sm text-gray-500")
    // A string subtitle is plain text: nothing is wrapped around it.
    expect(plain_p.children).toHaveLength(0)
    cleanup()

    const with_actions = render(SettingsHeaderActionsHarness, {
      props: { title: "Overview", subtitle: "What happened here" },
    })
    const actions_p = with_actions.container.querySelector("p")!
    expect(actions_p.className).toBe("text-sm text-gray-500")
    expect(actions_p.textContent).toBe("What happened here")
    expect(actions_p.children).toHaveLength(0)
    // Still the title's sibling inside the title box, not pushed elsewhere.
    expect(actions_p.previousElementSibling!.tagName).toBe("H2")
  })

  it("renders slot content inside that same paragraph, in place of the string", () => {
    const { container } = render(SettingsHeaderSubtitleHarness, {
      props: { title: "Overview", subtitle: "What happened here" },
    })
    const p = container.querySelector("p")!
    expect(p.className).toBe("text-sm text-gray-500")
    // The slot wins over the string, so a caller cannot end up with both.
    expect(p.textContent).not.toContain("What happened here")
    expect(p.textContent).toContain("Planned using your data guide.")
    expect(p.querySelector("[data-subtitle-action]")).not.toBeNull()
    expect(container.querySelectorAll("p")).toHaveLength(1)
  })

  it("renders slot content inside that same paragraph beside the actions", () => {
    const { container } = render(SettingsHeaderSubtitleActionsHarness, {
      props: { title: "Overview", subtitle: "What happened here" },
    })
    const p = container.querySelector("p")!
    expect(p.className).toBe("text-sm text-gray-500")
    expect(p.textContent).not.toContain("What happened here")
    expect(p.querySelector("[data-subtitle-action]")).not.toBeNull()
    // The actions still sit outside the subtitle, on the title's row.
    const action = container.querySelector("[data-action]")!
    expect(p.contains(action)).toBe(false)
    expect(action.closest(".border-b")).toBe(
      container.querySelector(".border-b"),
    )
  })
})
