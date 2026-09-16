// @vitest-environment jsdom
import { describe, it, expect, afterEach, beforeEach, vi } from "vitest"
import { render, fireEvent, cleanup } from "@testing-library/svelte"
import { tick } from "svelte"
import ClaimCard from "./claim_card.svelte"
import type { Citation, Claim, ClaimVerdict } from "./claim_evidence"

afterEach(() => {
  cleanup()
})

function claim(overrides: Partial<Claim> = {}): Claim {
  return {
    text: "The agent stated a return window as fact [1]. Disagree if the window is documented.",
    citations: [
      { marker: 1, source: "output", from: "30 days", to: "30 days" },
    ],
    is_verdict: false,
    ...overrides,
  }
}

function fresh_verdict(): ClaimVerdict {
  return { agrees: null, why: "" }
}

function by_id<T extends HTMLElement>(container: HTMLElement, id: string): T {
  const found = container.querySelector<T>(`#${id}`)
  if (!found) throw new Error(`no element with id ${id}`)
  return found
}

describe("ClaimCard — Agree / Disagree", () => {
  it("asks with both answers in primary outline, and drops the colour once answered", async () => {
    const verdict = fresh_verdict()
    const { container } = render(ClaimCard, {
      props: { claim: claim(), index: 0, verdict },
    })

    // Undecided: the ask is on both sides equally, so both carry the house
    // pick-one-of-these treatment.
    const agree = by_id(container, "claim-agree-0")
    const disagree = by_id(container, "claim-disagree-0")
    for (const button of [agree, disagree]) {
      expect(button.className).toContain("btn-outline")
      expect(button.className).toContain("btn-primary")
    }

    // Answered: the chosen side fills in, and the side not taken keeps the
    // outline but loses the primary colour — there is nothing left to ask.
    await fireEvent.click(agree)
    expect(agree.className).toContain("btn-secondary")
    expect(agree.className).not.toContain("btn-outline")
    expect(disagree.className).toContain("btn-outline")
    expect(disagree.className).not.toContain("btn-primary")
  })

  it("is one table row of three cells, collapsed or expanded", async () => {
    // Collapsing changes a claim's height, never its shape: the same three
    // cells either way, so the columns line up down the table whatever state
    // each claim is in.
    for (const open of [true, false]) {
      const { container } = render(ClaimCard, {
        props: { claim: claim(), index: 0, verdict: fresh_verdict(), open },
      })
      const row = by_id(container, "claim-card-0")
      expect(row.tagName).toBe("TR")
      const cells = row.querySelectorAll("td")
      expect(cells).toHaveLength(3)
      for (const cell of cells) {
        expect(cell.className).toContain("align-middle")
      }
      cleanup()
    }
  })

  it("numbers the claim and records Agree without a reason box", async () => {
    const verdict = fresh_verdict()
    const { container } = render(ClaimCard, {
      props: { claim: claim(), index: 2, verdict },
    })

    // The number is the one the builder's own cross-references use, in the
    // row's first cell under the table's "#" header.
    expect(
      by_id(container, "claim-card-2").querySelector("td")!.textContent?.trim(),
    ).toBe("3")

    await fireEvent.click(by_id(container, "claim-agree-2"))
    expect(verdict.agrees).toBe(true)
    expect(by_id(container, "claim-agree-2").className).toContain(
      "btn-secondary",
    )
    expect(container.querySelector("#claim-why-2")).toBeNull()
  })

  it("Disagree opens the required reason box, and Agree drops the reason again", async () => {
    const verdict = fresh_verdict()
    const { container } = render(ClaimCard, {
      props: { claim: claim(), index: 0, verdict },
    })

    await fireEvent.click(by_id(container, "claim-disagree-0"))
    expect(verdict.agrees).toBe(false)
    expect(by_id(container, "claim-disagree-0").className).toContain(
      "btn-secondary",
    )
    const why = by_id<HTMLTextAreaElement>(container, "claim-why-0")
    expect(why.placeholder).toBe("This is wrong because…")
    // Required, and flagged as an error once the reviewer has touched the
    // field and left it empty. A box that has only just opened is not marked:
    // the form control holds its required state back until the first change,
    // so nothing is red before anyone has had a chance to type.
    expect(why.className).not.toContain("textarea-error")
    // Under 20 characters, so the card flags it as likely too short, and it
    // is still accepted: only an empty reason is ever held back.
    await fireEvent.input(why, { target: { value: "The window is real." } })
    expect(verdict.why).toBe("The window is real.")
    expect(why.className).not.toContain("textarea-error")
    await fireEvent.input(why, { target: { value: "" } })
    expect(why.className).toContain("textarea-error")

    // Switching to Agree hides the box and clears the reason typed under
    // Disagree, so nothing stale rides the agree grade into the record.
    await fireEvent.click(by_id(container, "claim-agree-0"))
    expect(verdict).toEqual({ agrees: true, why: "" })
    expect(container.querySelector("#claim-why-0")).toBeNull()
  })
})

describe("ClaimCard — the claim text", () => {
  it("chips a [n] that has a citation and leaves one without as plain text", async () => {
    let cited: Citation | undefined
    const { container, getAllByTitle } = render(ClaimCard, {
      props: {
        claim: claim({
          text: "The reply gives 30 days [1] and cites item [2] of the policy.",
        }),
        index: 0,
        verdict: fresh_verdict(),
        on_cite: (c: Citation) => (cited = c),
      },
    })

    // Exactly one chip: [1] resolves, [2] is a number the model quoted out
    // of the trace and must not become a dead button.
    const chips = getAllByTitle("View in trace")
    expect(chips.map((c) => c.textContent)).toEqual(["[1]"])
    expect(container.textContent).toContain("cites item [2] of the policy")

    await fireEvent.click(chips[0])
    expect(cited?.marker).toBe(1)
  })

  it("renders the Note paragraph apart and muted, with We suggest inline", () => {
    const { container } = render(ClaimCard, {
      props: {
        claim: claim({
          text: "The joke retells a known one [1]. We suggest 'Agree', keeping this eval focused on safety.\n\nNote: the rubric never mentions originality.",
        }),
        index: 0,
        verdict: fresh_verdict(),
      },
    })

    const note = container.querySelector("[data-claim-note]")
    expect(note?.textContent?.trim()).toBe(
      "Note: the rubric never mentions originality.",
    )
    expect(note?.className).toContain("text-gray-500")
    // The suggestion is part of the ask, so it stays in the claim body.
    const body = container.querySelector("p")
    expect(body?.textContent).toContain("We suggest 'Agree'")
    expect(body?.textContent).not.toContain("Note:")
  })
})

describe("ClaimCard — the disagree reason hint", () => {
  const TOO_SHORT =
    "Likely too short to help improve the judge. What did it get wrong?"
  const MORE_DETAIL = "More details here would help the judge improve faster."

  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  async function open_disagree() {
    const verdict = fresh_verdict()
    const rendered = render(ClaimCard, {
      props: { claim: claim(), index: 0, verdict },
    })
    await fireEvent.click(by_id(rendered.container, "claim-disagree-0"))
    return { ...rendered, verdict }
  }

  async function type_reason(container: HTMLElement, value: string) {
    await fireEvent.input(by_id(container, "claim-why-0"), {
      target: { value },
    })
  }

  // Let `ms` of the pause pass, then let Svelte render whatever that changed.
  async function wait(ms: number) {
    vi.advanceTimersByTime(ms)
    await tick()
  }

  function hint(container: HTMLElement): string {
    return by_id(container, "claim-why-hint-0").textContent?.trim() ?? ""
  }

  it("labels the reason box and points it at the hint slot", async () => {
    const { container } = await open_disagree()

    const label = container.querySelector('label[for="claim-why-0"]')
    expect(label?.textContent?.trim()).toBe(
      "What do you disagree with? What should the judge have done instead?",
    )

    const slot = by_id(container, "claim-why-hint-0")
    expect(
      by_id(container, "claim-why-0").getAttribute("aria-describedby"),
    ).toBe("claim-why-hint-0")
    expect(slot.getAttribute("aria-live")).toBe("polite")
    // Fixed height so an appearing hint never moves the card.
    expect(slot.className).toContain("h-5")
  })

  it("warns when the trimmed reason is under 20 characters", async () => {
    const { container } = await open_disagree()

    // 21 characters typed, 13 once trimmed: the tier reads the trimmed length.
    await type_reason(container, "  Wrong window.      ")
    await wait(500)
    expect(hint(container)).toBe(TOO_SHORT)
  })

  it("asks for more detail from 20 through 39 characters", async () => {
    const { container } = await open_disagree()

    await type_reason(container, "a".repeat(20))
    await wait(500)
    expect(hint(container)).toBe(MORE_DETAIL)

    await type_reason(container, "a".repeat(39))
    await wait(500)
    expect(hint(container)).toBe(MORE_DETAIL)
  })

  it("says nothing for an empty, blank, or long enough reason", async () => {
    const { container } = await open_disagree()
    await wait(500)
    expect(hint(container)).toBe("")

    await type_reason(container, "     ")
    await wait(500)
    expect(hint(container)).toBe("")

    await type_reason(container, "a".repeat(40))
    await wait(500)
    expect(hint(container)).toBe("")
  })

  it("holds the hint back until typing pauses", async () => {
    const { container } = await open_disagree()

    await type_reason(container, "Too short.")
    await wait(499)
    expect(hint(container)).toBe("")
    await wait(1)
    expect(hint(container)).toBe(TOO_SHORT)
  })

  it("restarts the pause on the next keystroke", async () => {
    const { container } = await open_disagree()

    await type_reason(container, "Too")
    await wait(400)
    await type_reason(container, "Too short.")
    await wait(400)
    expect(hint(container)).toBe("")
    await wait(100)
    expect(hint(container)).toBe(TOO_SHORT)
  })

  it("waits out the pause when the reason shrinks into the short tier", async () => {
    const { container } = await open_disagree()

    await type_reason(container, "a".repeat(30))
    await wait(500)
    expect(hint(container)).toBe(MORE_DETAIL)

    await type_reason(container, "Too short.")
    await wait(499)
    expect(hint(container)).toBe(MORE_DETAIL)
    await wait(1)
    expect(hint(container)).toBe(TOO_SHORT)
  })

  it("clears the hint the moment the reason gets long enough", async () => {
    const { container } = await open_disagree()

    await type_reason(container, "a".repeat(30))
    await wait(500)
    expect(hint(container)).toBe(MORE_DETAIL)

    await type_reason(container, "a".repeat(40))
    await tick()
    expect(hint(container)).toBe("")
  })

  it("drops the hint at once when the card moves to another verdict", async () => {
    const { container, component } = render(ClaimCard, {
      props: {
        claim: claim(),
        index: 0,
        verdict: { agrees: false, why: "Too short." },
      },
    })
    await wait(500)
    expect(hint(container)).toBe(TOO_SHORT)

    // The review reuses these cards by position, so a new verdict object is
    // how a different claim arrives. Its reason gets the usual pause.
    component.$set({ verdict: { agrees: false, why: "a".repeat(30) } })
    await tick()
    expect(hint(container)).toBe("")
    await wait(500)
    expect(hint(container)).toBe(MORE_DETAIL)
  })

  it("keeps the pause running when the parent re-sets the same verdict", async () => {
    const { container, component, verdict } = await open_disagree()

    await type_reason(container, "Too short.")
    await wait(400)
    // A writeback or an unrelated parent render: the same verdict object with
    // the same reason, so the pause must run out on schedule, not start over.
    component.$set({ verdict })
    await tick()
    await wait(100)
    expect(hint(container)).toBe(TOO_SHORT)
  })

  it("drops a pending hint when the card goes away", async () => {
    const { container, unmount } = await open_disagree()
    // Counted as a delta so the card's own focus timer, and anything the test
    // environment keeps running, stay out of it.
    const idle_timers = vi.getTimerCount()

    await type_reason(container, "Too short.")
    expect(vi.getTimerCount()).toBe(idle_timers + 1)
    unmount()
    expect(vi.getTimerCount()).toBe(idle_timers)
  })

  it("clears the hint the moment the reason is emptied", async () => {
    const { container } = await open_disagree()

    await type_reason(container, "Too short.")
    await wait(500)
    expect(hint(container)).toBe(TOO_SHORT)

    await type_reason(container, "")
    await tick()
    expect(hint(container)).toBe("")
  })
})

describe("ClaimCard — collapsed", () => {
  it("shows its state and one plain line, and Edit asks the review to open it", async () => {
    const on_open = vi.fn()
    const { container } = render(ClaimCard, {
      props: {
        claim: claim(),
        index: 1,
        verdict: { agrees: false, why: "The window is real." },
        open: false,
        on_open,
      },
    })

    expect(by_id(container, "claim-state-1").textContent?.trim()).toBe(
      "Disagreed",
    )
    expect(container.querySelector("#claim-agree-1")).toBeNull()
    expect(container.querySelector("#claim-why-1")).toBeNull()
    // The [1] marker is dropped, so a line cut short never ends in a chip.
    const card = by_id(container, "claim-card-1")
    expect(card.textContent).toContain(
      "The agent stated a return window as fact.",
    )
    expect(card.querySelector("button[title='View in trace']")).toBeNull()

    await fireEvent.click(by_id(container, "claim-edit-1"))
    expect(on_open).toHaveBeenCalledOnce()
  })
})

describe("ClaimCard — reporting answers", () => {
  it("tells the review each answer, Disagree and Agree alike", async () => {
    const on_answer = vi.fn()
    const { container } = render(ClaimCard, {
      props: { claim: claim(), index: 0, verdict: fresh_verdict(), on_answer },
    })

    await fireEvent.click(by_id(container, "claim-disagree-0"))
    await fireEvent.click(by_id(container, "claim-agree-0"))
    expect(on_answer.mock.calls).toEqual([[false], [true]])
    // Each answer carries its keyboard shortcut.
    expect(
      by_id(container, "claim-agree-0").querySelector("span")?.textContent,
    ).toBe("A")
    expect(
      by_id(container, "claim-disagree-0").querySelector("span")?.textContent,
    ).toBe("D")
  })
})
