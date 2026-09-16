// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest"
import { render, cleanup } from "@testing-library/svelte"
import ProgressCount from "./progress_count.svelte"

afterEach(() => cleanup())

function readout(container: HTMLElement) {
  return container.querySelector("[data-progress-count]")
}

// One line of squashed whitespace, so the assertions read as the user sees the
// line rather than as the template's newlines.
function line(container: HTMLElement): string {
  return (readout(container)?.textContent ?? "").replace(/\s+/g, " ").trim()
}

describe("ProgressCount", () => {
  it("reads N of M, in small grey text under the bar", () => {
    const { container } = render(ProgressCount, {
      props: { value: 3, max: 20, noun: "judged" },
    })

    expect(line(container)).toBe("3 of 20 judged")
    // Small and grey, the palette's secondary text — the count is context for
    // the bar, never the thing being read first.
    expect(readout(container)!.className).toContain("text-xs")
    expect(readout(container)!.className).toContain("text-gray-500")

    // Under the bar, not beside it: one centred column.
    const bar = container.querySelector("progress")!
    expect(bar.getAttribute("value")).toBe("3")
    expect(bar.getAttribute("max")).toBe("20")
    expect(bar.nextElementSibling).toBe(readout(container))
  })

  it("says 'up to' where the denominator is a ceiling", () => {
    // A conversation that ends early spends fewer turns, so the total is the
    // most the batch can spend, not what it will.
    const { container } = render(ProgressCount, {
      props: {
        value: 12,
        max: 60,
        noun: "turns complete",
        max_is_ceiling: true,
      },
    })
    expect(line(container)).toBe("12 of up to 60 turns complete")
  })

  it("reports failures on the same line, never in the sentence above", () => {
    const { container } = render(ProgressCount, {
      props: { value: 3, max: 20, noun: "judged", failed: 2 },
    })
    expect(line(container)).toBe("3 of 20 judged — 2 failed")
  })

  it("lets the bar run ahead of the count when failures still count as done", () => {
    // A failed case is finished, just not judged: the bar reaches full while
    // the count beside it stays short, rather than stalling on the failure.
    const { container } = render(ProgressCount, {
      props: {
        value: 18,
        bar_value: 20,
        max: 20,
        noun: "judged",
        failed: 2,
      },
    })
    expect(container.querySelector("progress")!.getAttribute("value")).toBe(
      "20",
    )
    expect(line(container)).toBe("18 of 20 judged — 2 failed")
  })

  it("shows the bar alone until a denominator arrives", () => {
    // The total lands with the first frame; "0 of 0" in the meantime would be
    // a worse answer than no line at all.
    const { container } = render(ProgressCount, {
      props: { value: 0, max: 0, noun: "judged" },
    })
    expect(readout(container)).toBeNull()
    expect(container.querySelector("progress")).not.toBeNull()
  })
})
