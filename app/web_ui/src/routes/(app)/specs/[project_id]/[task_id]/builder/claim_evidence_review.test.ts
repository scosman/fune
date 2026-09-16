// @vitest-environment jsdom
import {
  describe,
  it,
  expect,
  afterAll,
  afterEach,
  beforeAll,
  vi,
} from "vitest"
import { render, fireEvent, cleanup } from "@testing-library/svelte"
import { tick } from "svelte"
import ClaimEvidenceReview from "./claim_evidence_review.svelte"
import {
  build_claim_review_payload,
  build_trace_reviews,
  is_trace_reviewed,
  user_says_meets_spec,
  type Citation,
  type Claim,
  type JudgeScore,
  type TraceClaims,
} from "./claim_evidence"
import type { TraceMessage } from "$lib/types"

// jsdom does not implement HTMLDialogElement.showModal/close. The trace
// modal's Dialog calls them, so polyfill with no-ops that just track the open
// state.
const original_show_modal = HTMLDialogElement.prototype.showModal
const original_close = HTMLDialogElement.prototype.close
const original_scroll_into_view = Element.prototype.scrollIntoView
beforeAll(() => {
  if (!HTMLDialogElement.prototype.showModal) {
    HTMLDialogElement.prototype.showModal = function () {
      this.open = true
    }
  }
  if (!HTMLDialogElement.prototype.close) {
    HTMLDialogElement.prototype.close = function () {
      this.open = false
    }
  }
  // Nor scrollIntoView, which the trace modal calls on the citation's <mark>
  // once it has rendered.
  if (!Element.prototype.scrollIntoView) {
    Element.prototype.scrollIntoView = function () {}
  }
})

afterAll(() => {
  HTMLDialogElement.prototype.showModal = original_show_modal
  HTMLDialogElement.prototype.close = original_close
  Element.prototype.scrollIntoView = original_scroll_into_view
})

afterEach(() => {
  cleanup()
})

// ── Fixtures ─────────────────────────────────────────────────────────────

const RAW_INPUT = "What is the return window on a mattress?"
const RAW_OUTPUT = "Our return window is 30 days."

function cite(from: string, to = from, source: "input" | "output" = "output") {
  return { marker: 1, source, from, to } satisfies Citation
}

function claim(text: string, overrides: Partial<Claim> = {}): Claim {
  return {
    text,
    citations: [cite("30 days")],
    is_verdict: false,
    ...overrides,
  }
}

const VERDICT_TEXT = "It fails because the window was never verified [1]."

// A built trace: an overview citing the input, one ordinary claim, and a
// second claim that is the verdict unless `verdict: false`.
function built_trace(
  id: string,
  opts: { judge_score?: JudgeScore; verdict?: boolean } = {},
): TraceClaims {
  const verdict = opts.verdict ?? true
  return {
    trace_id: id,
    leaf_run_id: `run_${id}`,
    raw_input: RAW_INPUT,
    raw_output: RAW_OUTPUT,
    judge_score: opts.judge_score ?? "fail",
    judge_reasoning: "The window was asserted with no source.",
    overview: {
      text: "The user asked about a mattress return window [1] and got a number.",
      citations: [cite("return window on", "mattress?", "input")],
    },
    claims: [
      claim("The agent stated a return window as fact [1]."),
      verdict
        ? claim(VERDICT_TEXT, { is_verdict: true })
        : claim("The agent named no policy page.", { citations: [] }),
    ],
    claims_state: "built",
    claims_error: null,
    trace: null,
  }
}

function by_id<T extends HTMLElement>(container: HTMLElement, id: string): T {
  const found = container.querySelector<T>(`#${id}`)
  if (!found) throw new Error(`no element with id ${id}`)
  return found
}

function render_review(
  traces: TraceClaims[],
  extra: Record<string, unknown> = {},
) {
  const verdicts = build_trace_reviews(traces)
  return {
    verdicts,
    ...render(ClaimEvidenceReview, {
      props: {
        traces,
        verdicts,
        selected_indices: traces.map((_, i) => i),
        judged_noun: "conversation",
        ...extra,
      },
    }),
  }
}

// Agree with every claim on the current trace.
async function agree_all(container: HTMLElement, count: number) {
  for (let i = 0; i < count; i++) {
    await fireEvent.click(by_id(container, `claim-agree-${i}`))
  }
}

function next_button(getByText: (t: string) => HTMLElement) {
  return getByText("Next") as HTMLButtonElement
}

// A mounted dialog, picked out by its title. Several are on the page at once,
// so the title is what tells them apart.
function dialog_titled(
  container: HTMLElement,
  title: string,
): HTMLDialogElement {
  const found = [...container.querySelectorAll("dialog")].find(
    (d) => d.querySelector("h3")?.textContent?.trim() === title,
  )
  if (!found) throw new Error(`no dialog titled ${title} rendered`)
  return found
}

function trace_dialog(container: HTMLElement): HTMLDialogElement {
  return dialog_titled(container, "Trace")
}

// The span the trace modal highlighted for the citation just clicked — only
// that view renders a <mark>, so its text identifies where it landed.
function cited_text(container: HTMLElement): string | null {
  return container.querySelector("mark")?.textContent ?? null
}

// Three claims, so a claim after the next one is there to stay collapsed.
function three_claim_trace(id: string): TraceClaims {
  return {
    ...built_trace(id),
    claims: [
      claim("The agent stated a return window as fact [1]."),
      claim("The agent named no policy page.", { citations: [] }),
      claim(VERDICT_TEXT, { is_verdict: true }),
    ],
  }
}

// The claims whose answer buttons are on screen, by index.
function open_claims(container: HTMLElement): number[] {
  return [...container.querySelectorAll("[id^=claim-agree-]")].map((el) =>
    Number(el.id.replace("claim-agree-", "")),
  )
}

// A collapsed claim's state label.
function state_of(container: HTMLElement, index: number): string {
  return by_id(container, `claim-state-${index}`).textContent?.trim() ?? ""
}

// The pill bar's buttons, and the one it marks as open. The pills carry their
// state in colour alone, so their position is read from the aria-label that
// names each for a screen reader.
function steps_of(container: HTMLElement): HTMLButtonElement[] {
  return [...by_id(container, "review-case-steps").querySelectorAll("button")]
}
function step_labels(container: HTMLElement): (string | null)[] {
  return steps_of(container).map((b) => b.getAttribute("aria-label"))
}
// The bar a pill draws. It sits inside a taller button, so the fill is on the
// inner element and the target is the outer one.
function fill_of(pill: HTMLButtonElement): string {
  return pill.querySelector("span")!.className
}
function current_step(container: HTMLElement): string | null {
  const label = by_id(container, "review-case-steps")
    .querySelector("[aria-current='step']")
    ?.getAttribute("aria-label")
  return label?.replace("Case ", "") ?? null
}

// The line over the pills, with its runs of whitespace squashed to one space.
function position_line(container: HTMLElement): string {
  return (by_id(container, "review-batch-count").textContent ?? "")
    .replace(/\s+/g, " ")
    .trim()
}

// ── The review surface ───────────────────────────────────────────────────

describe("ClaimEvidenceReview — overview and claims", () => {
  it("lists the claims in the house table, one row each", () => {
    const trace = built_trace("t0")
    const { container } = render_review([trace])

    // The house table style: a bordered, rounded wrapper around the table.
    const table = container.querySelector(".rounded-lg.border table")
    expect(table).not.toBeNull()
    expect(
      [...table!.querySelectorAll("thead th")].map((th) =>
        th.textContent?.trim(),
      ),
    ).toEqual(["#", "Claim from Judge", "Decision"])

    // Every claim is a row of that table's body — the claim list and the
    // table cannot drift out of step.
    const claim_count = trace.claims?.length ?? 0
    expect(claim_count).toBeGreaterThan(0)
    expect(
      [...table!.querySelectorAll("tbody tr")].map((row) => row.id),
    ).toEqual(Array.from({ length: claim_count }, (_, i) => `claim-card-${i}`))
  })

  it("shows the overview with its trace button, the first claim open and the rest collapsed", async () => {
    const { container, getByText } = render_review([built_trace("t0")])

    const overview = by_id(container, "review-overview")
    expect(overview.textContent).toContain(
      "The user asked about a mattress return window",
    )
    expect(overview.textContent).toContain("A summary of this task's run.")
    expect(overview.querySelector("#view-full-trace")).not.toBeNull()

    // The claims are the review, so the first is open on arrival with its
    // answers, and the rest wait as one line each with their state and Edit.
    // Their sub-line carries the step's purpose, since no step header does.
    expect(
      getByText("Confirm the judge is aligned to your expectations."),
    ).toBeTruthy()
    const first = by_id(container, "claim-card-0")
    // The number cell, under the table's "#" header.
    expect(first.querySelector("td")!.textContent?.trim()).toBe("1")
    expect(first.querySelector("#claim-agree-0")).not.toBeNull()
    const second = by_id(container, "claim-card-1")
    expect(second.querySelector("#claim-agree-1")).toBeNull()
    expect(state_of(container, 1)).toBe("Not decided")
    expect(second.querySelector("#claim-edit-1")).not.toBeNull()
    // One plain line: the [1] marker is dropped rather than cut into a chip.
    expect(second.textContent).toContain(
      "It fails because the window was never verified.",
    )
    expect(second.querySelector("button[title='View in trace']")).toBeNull()
    expect(
      [...container.querySelectorAll("button")].some((b) =>
        /show claims/i.test(b.textContent ?? ""),
      ),
    ).toBe(false)

    await fireEvent.click(by_id(container, "view-full-trace"))
    expect(trace_dialog(container).textContent).toContain(RAW_OUTPUT)
  })

  it("opens the trace at the span an overview chip cites", async () => {
    const { container } = render_review([built_trace("t0")])
    const chip = by_id(container, "review-overview").querySelector(
      "[title='View in trace']",
    ) as HTMLButtonElement
    expect(chip.textContent).toBe("[1]")

    await fireEvent.click(chip)
    // An input citation, so it lands in the Input panel of the raw view.
    expect(cited_text(trace_dialog(container))).toBe(
      "return window on a mattress?",
    )
  })

  it("caps the overview in pixels so only a runaway overview folds", () => {
    const { container } = render_review([built_trace("t0")])
    const overview = by_id(container, "review-overview")

    // Output measures overflow against a cap it reads in pixels, so the cap
    // has to be a pixel value: a viewport unit reads as its bare number and
    // folds every overview.
    const capped = [...overview.querySelectorAll("[style]")].filter((el) =>
      (el.getAttribute("style") ?? "").includes("max-height"),
    )
    expect(capped.length).toBe(1)
    expect(capped[0].getAttribute("style")).toContain("max-height: 480px")

    // jsdom measures every scrollHeight as 0, so nothing folds here whatever
    // the cap is. The assertion pins the intent: this overview is short.
    expect(
      [...overview.querySelectorAll("button")].some((b) =>
        /show all/i.test(b.textContent ?? ""),
      ),
    ).toBe(false)
  })
})

// Two paragraphs and a run of spaces: enough shape that a rendering which
// collapsed whitespace would read differently from the text as written.
const SPEC_TEXT =
  "The agent must not guess at policy details.\n\nIt may state  only the facts it was given."

const SPEC_DIALOG_TITLE = "Eval Description"

function spec_dialog(container: HTMLElement): HTMLDialogElement {
  return dialog_titled(container, SPEC_DIALOG_TITLE)
}

// The eval description has no control of its own on this step: the page header
// carries the line that opens it, and calls in through this exported function.
describe("ClaimEvidenceReview — the eval text", () => {
  it("opens the eval's description read-only, and offers no control of its own", async () => {
    const { container, component } = render_review([built_trace("t0")], {
      spec_text: SPEC_TEXT,
    })

    // Nothing in the work opens it — the trigger is the page header's.
    expect(container.querySelector("#view-eval")).toBeNull()

    const dialog = spec_dialog(container)
    expect(dialog.open).toBe(false)

    component.show_spec_dialog()
    await tick()
    expect(dialog.open).toBe(true)
    // The text verbatim, on the house read-only surface. Output prints into a
    // pre, which is what keeps the line breaks and runs of spaces on screen.
    const shown = by_id(dialog, "spec-text")
    expect(shown.textContent?.trim()).toBe(SPEC_TEXT)
    expect(shown.querySelector("pre")?.className).toContain(
      "whitespace-pre-wrap",
    )
    // Read-only: the description is shown, never edited here.
    expect(dialog.querySelector("input, textarea")).toBeNull()
  })

  it("keeps the eval text reachable while the claims are still building", async () => {
    // Every trace starts with no overview and builds lazily. The eval text
    // matters most in that state, where nothing on screen describes the
    // conversation yet.
    const unbuilt: TraceClaims = {
      ...built_trace("t0"),
      overview: null,
      claims: null,
      claims_state: "unbuilt",
    }
    const { container, component } = render_review([unbuilt], {
      spec_text: SPEC_TEXT,
    })
    // The Overview section renders in every state, so the header never goes
    // away: only the body under it changes.
    expect(container.querySelector("#review-overview")).not.toBeNull()

    component.show_spec_dialog()
    await tick()
    const dialog = spec_dialog(container)
    expect(dialog.open).toBe(true)
    expect(by_id(dialog, "spec-text").textContent?.trim()).toBe(SPEC_TEXT)
  })

  it("reopens the same dialog after moving to the next conversation", async () => {
    // One eval-level dialog for the whole review: advancing does not tear it
    // down and rebuild it, so the text is one call away on every trace.
    const { container, component, getByText } = render_review(
      [built_trace("t0"), built_trace("t1")],
      { spec_text: SPEC_TEXT },
    )

    component.show_spec_dialog()
    await tick()
    const opened = spec_dialog(container)
    expect(opened.open).toBe(true)
    // jsdom cannot submit the dialog's close form, so close it directly.
    opened.open = false

    await agree_all(container, 2)
    await fireEvent.click(next_button(getByText))
    expect(current_step(container)).toBe("2")

    component.show_spec_dialog()
    await tick()
    const reopened = spec_dialog(container)
    // The same node, not a fresh one mounted for this conversation.
    expect(reopened).toBe(opened)
    expect(reopened.open).toBe(true)
    expect(by_id(reopened, "spec-text").textContent?.trim()).toBe(SPEC_TEXT)
  })

  it("mounts no dialog when there is no eval text, and opening it is a no-op", async () => {
    for (const spec_text of [null, "", "   "]) {
      const { container, component } = render_review([built_trace("t0")], {
        spec_text,
      })

      // Nothing mounted: no dialog carrying the eval title.
      expect(
        [...container.querySelectorAll("dialog h3")].some(
          (h) => h.textContent?.trim() === "Eval Description",
        ),
      ).toBe(false)
      // The page header hides its line in this case, but the call must be safe
      // whatever the header does — it is the page's flag, not this one's.
      expect(() => component.show_spec_dialog()).not.toThrow()
      await tick()
      cleanup()
    }
  })
})

describe("ClaimEvidenceReview — the way into the full trace", () => {
  it("is a right-aligned link under the overview panel, not inside it", () => {
    const { container } = render_review([built_trace("t0")])

    const link = by_id(container, "view-full-trace")
    expect(link.textContent?.trim()).toBe("Full Trace")
    expect(link.className).toContain("link")
    expect(link.parentElement!.className).toContain("justify-end")

    // Under the panel, not in it: Output folds its body at max_height and
    // lays a gradient over the fold. Still its footer, though — next sibling,
    // closer than the gap the column gives its own sections.
    const panel = by_id(container, "review-overview").querySelector(
      ".relative",
    )!
    expect(panel.contains(link)).toBe(false)

    const footer = link.parentElement!
    expect(footer.previousElementSibling).toBe(panel)
    expect(footer.parentElement!.className).toContain("gap-1")
  })
})

describe("ClaimEvidenceReview — Next gating", () => {
  it("needs Agree or Disagree on every claim before Next opens", async () => {
    const { container, getByText } = render_review([
      built_trace("t0"),
      built_trace("t1"),
    ])

    expect(next_button(getByText).disabled).toBe(true)
    // Next is the screen's one primary, sized like Previous beside it, and
    // carries no keyboard hint: the shortcut fires Save, never Next.
    expect(next_button(getByText).className).toContain("btn-primary")
    expect(next_button(getByText).className).not.toContain("min-w-64")

    await fireEvent.click(by_id(container, "claim-agree-0"))
    expect(next_button(getByText).disabled).toBe(true)
    await fireEvent.click(by_id(container, "claim-agree-1"))
    expect(next_button(getByText).disabled).toBe(false)
  })

  it("holds Next until a Disagree carries a reason", async () => {
    const { container, getByText, verdicts } = render_review([
      built_trace("t0"),
      built_trace("t1"),
    ])

    await fireEvent.click(by_id(container, "claim-agree-0"))
    await fireEvent.click(by_id(container, "claim-disagree-1"))
    expect(next_button(getByText).disabled).toBe(true)

    await fireEvent.input(by_id(container, "claim-why-1"), {
      target: { value: "The window is published." },
    })
    expect(verdicts[0].claim_verdicts[1]).toEqual({
      agrees: false,
      why: "The window is published.",
    })
    expect(next_button(getByText).disabled).toBe(false)
  })
})

describe("ClaimEvidenceReview — the pass/fail call", () => {
  it("derives the call from the verdict claim's grade, and asks no Pass/Fail row", async () => {
    for (const judge_score of ["fail", "pass"] as const) {
      const traces = [built_trace("t0", { judge_score })]
      const { container, verdicts } = render_review(traces)
      expect(container.querySelector("#review-overall")).toBeNull()

      // Agreeing with the verdict claim keeps the judge's call.
      await agree_all(container, 2)
      expect(is_trace_reviewed(traces[0], verdicts[0])).toBe(true)
      expect(user_says_meets_spec(traces[0], verdicts[0])).toBe(
        judge_score === "pass",
      )

      // Disagreeing with it flips the call the other way. Every claim is
      // decided, so every claim is collapsed and the verdict reopens by Edit.
      await fireEvent.click(by_id(container, "claim-edit-1"))
      await fireEvent.click(by_id(container, "claim-disagree-1"))
      await fireEvent.input(by_id(container, "claim-why-1"), {
        target: { value: "The judge read the transcript wrong." },
      })
      expect(user_says_meets_spec(traces[0], verdicts[0])).toBe(
        judge_score !== "pass",
      )
      expect(build_claim_review_payload(traces[0], verdicts[0])).toMatchObject({
        overview: traces[0].overview?.text,
        human_verdict: judge_score === "pass" ? "fail" : "pass",
        claims: [
          { human_grade: "agree", human_feedback: null },
          {
            text: VERDICT_TEXT,
            human_grade: "disagree",
            human_feedback: "The judge read the transcript wrong.",
          },
        ],
      })
      cleanup()
    }
  })
})

describe("ClaimEvidenceReview — the forward action on the last conversation", () => {
  it("holds the slot disabled until the gate is met, with the label unchanged", async () => {
    const traces = [built_trace("only")]

    // Gate not met: the same button holds the slot, simply disabled, the way
    // every other form in the app holds a submit.
    const gated = render_review(traces, { save_disabled: true })
    const blocked = by_id<HTMLButtonElement>(gated.container, "review-next")
    expect(blocked.disabled).toBe(true)
    expect(gated.container.querySelector(".tooltip")).toBeNull()
    cleanup()

    // Gate met on the last conversation: the same slot, now enabled.
    const open = render_review(traces, { save_disabled: false })
    const live = by_id<HTMLButtonElement>(open.container, "review-next")
    expect(live.disabled).toBe(false)
    // The slot keeps one size across that flip, so it doesn't resize as the
    // gate completes — and that size is the same button Previous is.
    expect(live.className).toBe(blocked.className)
    expect(live.className).not.toContain("min-w-64")
  })

  // The label used to flip between Save and Refine Judge with the grades, then
  // between Continue and Next with the position. It no longer moves at all:
  // one button, one word, whatever the position and whatever the gate says.
  // What the click leads to on the last case is settled in the dialog it
  // opens, not in the word on the button.
  it("reads Next in every position, enabled or not", () => {
    const positions = [
      { traces: [built_trace("t0"), built_trace("t1")], save_disabled: true },
      { traces: [built_trace("only")], save_disabled: false },
      { traces: [built_trace("only")], save_disabled: true },
    ]
    for (const { traces, save_disabled } of positions) {
      const { container } = render_review(traces, { save_disabled })
      expect(by_id(container, "review-next").textContent?.trim()).toContain(
        "Next",
      )
      expect(container.textContent).not.toContain("Continue")
      cleanup()
    }
  })

  it("reports the forward action to the parent, which decides what it means", async () => {
    const on_save = vi.fn()
    const { container } = render_review([built_trace("only")], {
      save_disabled: false,
      on_save,
    })
    await fireEvent.click(by_id(container, "review-next"))
    expect(on_save).toHaveBeenCalledTimes(1)
  })
})

function echoing_trace(citation?: Citation): TraceClaims {
  return {
    ...built_trace("echo_0", { verdict: false }),
    claims: [
      claim(
        citation
          ? "The reply answers the question [1]."
          : "The reply gives 30 days.",
        { citations: citation ? [citation] : [] },
      ),
    ],
    trace: [
      { role: "system", content: "You are a support agent." },
      { role: "user", content: RAW_INPUT },
      { role: "assistant", content: RAW_OUTPUT },
    ],
  }
}

// A run whose stored trace is a tool loop: the model called a tool, read the
// result, then answered.
function tool_loop_trace(): TraceClaims {
  return {
    ...built_trace("tool_0", { verdict: false }),
    trace: [
      { role: "system", content: "You are a support agent." },
      { role: "user", content: RAW_INPUT },
      { role: "assistant", content: "Let me look up the policy." },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: {
              name: "lookup_policy",
              arguments: '{"topic": "returns"}',
            },
          },
        ],
      },
      {
        role: "tool",
        content: '{"output": "Returns accepted within 30 days."}',
        tool_call_id: "call_1",
      },
      { role: "assistant", content: RAW_OUTPUT },
    ] as TraceMessage[],
  }
}

// The first claim card's citation chip. The fixture's overview carries a chip
// of its own, so a title lookup over the whole page would find two.
function claim_chip(container: HTMLElement): HTMLButtonElement {
  const chip = by_id(
    container,
    "claim-card-0",
  ).querySelector<HTMLButtonElement>("[title='View in trace']")
  if (!chip) throw new Error("no citation chip on the first claim")
  return chip
}

function chat_bubbles(root: ParentNode): Element[] {
  return [...root.querySelectorAll("[data-testid^='chat-msg-']")]
}

function occurrences(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1
}

describe("the trace modal — one rendering for both arms", () => {
  it("renders a single-turn trace as a conversation, not as labelled panels", async () => {
    // A single-turn run is a conversation of one turn, so it opens in the same
    // chat view multi-turn does.
    const { container } = render_review([tool_loop_trace()])
    await fireEvent.click(by_id(container, "view-full-trace"))
    const dialog = trace_dialog(container)

    expect(dialog.querySelector("[data-testid='chat-msg-user']")).not.toBeNull()
    expect(chat_bubbles(dialog).length).toBeGreaterThan(0)
    expect(dialog.querySelector("[data-testid='review-input']")).toBeNull()
    expect(dialog.querySelector("[data-testid='review-output']")).toBeNull()
  })

  it("shows a single-turn tool loop's calls and results as trace nodes", async () => {
    // The reason the shared view is worth adopting: the chat surface renders
    // tool activity as first-class nodes, which the panels never did.
    const { container } = render_review([tool_loop_trace()])
    await fireEvent.click(by_id(container, "view-full-trace"))
    const dialog = trace_dialog(container)

    expect(
      dialog.querySelector("[data-testid='chat-msg-toolcall']"),
    ).not.toBeNull()
    expect(dialog.textContent ?? "").toContain("Toolcall")
  })
})

describe("the trace modal — multi-turn", () => {
  it("shows the conversation alone, with no panels around it", async () => {
    const traces = [echoing_trace()]
    const { container } = render_review(traces)
    await fireEvent.click(by_id(container, "view-full-trace"))
    const dialog = trace_dialog(container)
    const text = dialog.textContent ?? ""

    expect(dialog.querySelector("[data-testid='chat-msg-user']")).not.toBeNull()
    // No labelled panels, and the opening user message appears once: it IS
    // the input, so a panel above the conversation would print it twice.
    expect(text).not.toContain("Input")
    expect(text).not.toContain("Output")
    expect(occurrences(text, traces[0].raw_input)).toBe(1)
    expect(text).toContain(traces[0].raw_output)
    expect(dialog.querySelector("[data-testid='review-input']")).toBeNull()
    // The conversation gets the widest dialog the house chrome offers.
    expect(dialog.querySelector(".modal-box")?.className).toContain("max-w-7xl")
  })

  it("goes back to the plain conversation once the citation clears", async () => {
    const citation = cite("return window on", "mattress?", "input")
    const { container } = render_review([echoing_trace(citation)])
    await fireEvent.click(claim_chip(container))
    expect(trace_dialog(container).querySelector("mark")).not.toBeNull()

    // Browsing the same trace: the conversation renders unhighlighted.
    await fireEvent.click(by_id(container, "view-full-trace"))
    const dialog = trace_dialog(container)
    expect(dialog.querySelector("mark")).toBeNull()
    expect(dialog.querySelector("[data-testid='chat-msg-user']")).not.toBeNull()
  })

  it("keeps both raw panels when the run recorded no trace", async () => {
    const traces = [{ ...echoing_trace(), trace: null }]
    const { container } = render_review(traces)
    await fireEvent.click(by_id(container, "view-full-trace"))
    const dialog = trace_dialog(container)
    const text = dialog.textContent ?? ""

    expect(dialog.querySelector("[data-testid='chat-msg-user']")).toBeNull()
    expect(text).toContain("Input")
    expect(text).toContain("Output")
    expect(text).toContain(traces[0].raw_input)
    expect(text).toContain(traces[0].raw_output)
  })

  it("shows the raw panels on the house read-only surface", async () => {
    // The no-conversation fallback: each panel is an Output, so the trace
    // reads the way every other read-only block in the app does, and a JSON
    // input is printed rather than shown as a typed rendering.
    const raw_input = '{"question": "return window?"}'
    const traces = [{ ...echoing_trace(), raw_input, trace: null }]
    const { container } = render_review(traces)
    await fireEvent.click(by_id(container, "view-full-trace"))
    const dialog = trace_dialog(container)

    const panel = [...dialog.querySelectorAll("div")].find((d) =>
      d.className.includes("bg-base-200"),
    )
    expect(panel).not.toBeUndefined()
    const printed = panel?.querySelector("pre")
    expect(printed?.className).toContain("whitespace-pre-wrap")
    expect(printed?.textContent?.trim()).toBe(
      JSON.stringify(JSON.parse(raw_input), null, 2),
    )
  })

  it("marks an input citation on the opening user bubble", async () => {
    const citation = cite("return window on", "mattress?", "input")
    const { container } = render_review([echoing_trace(citation)])
    await fireEvent.click(claim_chip(container))
    const dialog = trace_dialog(container)
    const mark = dialog.querySelector("mark")

    // The span resolved against raw_input, not raw_output — both texts carry
    // "return window", only the input carries the rest of the anchor. On
    // multi-turn the input IS the conversation's opening message, so the
    // citation lands on that bubble.
    expect(mark?.textContent).toBe("return window on a mattress?")
    expect(mark?.closest("[data-testid='chat-msg-user']")).not.toBeNull()
    expect(dialog.querySelectorAll("mark").length).toBe(1)
  })

  it("marks an output citation on the chat node it came from", async () => {
    const { container } = render_review([echoing_trace(cite("30 days"))])
    await fireEvent.click(claim_chip(container))
    const dialog = trace_dialog(container)

    const mark = dialog.querySelector("mark")
    expect(mark?.textContent).toBe("30 days")
    expect(mark?.hasAttribute("data-highlight-target")).toBe(true)
  })
})

// The pill bar states the reviewer's position in the graded sequence: it is
// the review's only progress readout, so it must count the subset the reviewer
// actually walks rather than the whole batch.
describe("ClaimEvidenceReview — the case position", () => {
  it("counts the position within the selected subset, in the line and the pills", async () => {
    const traces = [
      built_trace("t0"),
      built_trace("t1"),
      built_trace("t2"),
      built_trace("t3"),
    ]
    // Two of the four are shown, and they are not the first two.
    const { container, getByText } = render_review(traces, {
      selected_indices: [1, 3],
    })

    // Position and progress in one line, and it is the only place either
    // number is written: the pills say the same thing in colour.
    expect(position_line(container)).toBe("Case 1 of 2 · 0/4 decided")
    // The line is not a heading — the step's only headings are its sections.
    expect(
      [...container.querySelectorAll("h2")].map((h) => h.textContent?.trim()),
    ).toEqual(["Overview", "Claims"])

    expect(steps_of(container)).toHaveLength(2)
    expect(current_step(container)).toBe("1")
    await agree_all(container, 2)
    await fireEvent.click(getByText("Next"))
    expect(position_line(container)).toBe("Case 2 of 2 · 2/4 decided")
    expect(current_step(container)).toBe("2")
  })

  it("names each pill for a screen reader, since colour is all it shows", () => {
    const { container } = render_review([
      built_trace("t0"),
      built_trace("t1"),
      built_trace("t2"),
    ])
    expect(step_labels(container)).toEqual(["Case 1", "Case 2", "Case 3"])
    // No text inside them: the pill is a segment, not a numbered button.
    expect(steps_of(container).every((b) => b.textContent?.trim() === "")).toBe(
      true,
    )
  })
})

describe("ClaimEvidenceReview — one claim at a time", () => {
  it("Agree collapses the claim and opens the next undecided one, and the last leaves none open", async () => {
    const { container } = render_review([three_claim_trace("t0")])
    expect(open_claims(container)).toEqual([0])

    await fireEvent.click(by_id(container, "claim-agree-0"))
    expect(open_claims(container)).toEqual([1])
    expect(state_of(container, 0)).toBe("Agreed")
    expect(state_of(container, 2)).toBe("Not decided")

    await fireEvent.click(by_id(container, "claim-agree-1"))
    await fireEvent.click(by_id(container, "claim-agree-2"))
    expect(open_claims(container)).toEqual([])
    expect([0, 1, 2].map((i) => state_of(container, i))).toEqual([
      "Agreed",
      "Agreed",
      "Agreed",
    ])
  })

  it("Disagree keeps the claim open for its reason and opens the next undecided claim too", async () => {
    const { container, verdicts } = render_review([three_claim_trace("t0")])
    await fireEvent.click(by_id(container, "claim-disagree-0"))
    expect(open_claims(container)).toEqual([0, 1])
    expect(container.querySelector("#claim-why-0")).not.toBeNull()
    expect(state_of(container, 2)).toBe("Not decided")

    // Answering the next claim settles the disagree above it once its reason
    // is in.
    await fireEvent.input(by_id(container, "claim-why-0"), {
      target: { value: "The window is documented." },
    })
    await fireEvent.click(by_id(container, "claim-agree-1"))
    expect(open_claims(container)).toEqual([2])
    expect(state_of(container, 0)).toBe("Disagreed")
    expect(verdicts[0].claim_verdicts[0]).toEqual({
      agrees: false,
      why: "The window is documented.",
    })
  })

  it("keeps a disagree with an empty reason open while the rest are answered", async () => {
    const traces = [three_claim_trace("t0")]
    const { container, verdicts } = render_review(traces)
    await fireEvent.click(by_id(container, "claim-disagree-0"))
    await fireEvent.click(by_id(container, "claim-agree-1"))
    await fireEvent.click(by_id(container, "claim-agree-2"))
    // The empty reason is what holds the case, so it stays in view.
    expect(open_claims(container)).toEqual([0])
    expect(container.querySelector("#claim-why-0")).not.toBeNull()
    expect(is_trace_reviewed(traces[0], verdicts[0])).toBe(false)

    // Opening another claim does not hide it either.
    await fireEvent.click(by_id(container, "claim-edit-1"))
    expect(open_claims(container)).toEqual([0, 1])
  })

  it("opens nothing more for a disagree on the last undecided claim", async () => {
    const { container } = render_review([built_trace("t0")])
    await fireEvent.click(by_id(container, "claim-agree-0"))
    await fireEvent.click(by_id(container, "claim-disagree-1"))
    expect(open_claims(container)).toEqual([1])
    expect(container.querySelector("#claim-why-1")).not.toBeNull()
  })

  it("Edit reopens a decided claim with its answer and collapses the finished one", async () => {
    const { container } = render_review([three_claim_trace("t0")])
    await agree_all(container, 3)

    await fireEvent.click(by_id(container, "claim-edit-1"))
    expect(open_claims(container)).toEqual([1])
    expect(by_id(container, "claim-agree-1").className).toContain(
      "btn-secondary",
    )

    await fireEvent.click(by_id(container, "claim-edit-2"))
    expect(open_claims(container)).toEqual([2])
  })

  it("opens a disagree still missing its reason and the first undecided claim when a case is shown again", () => {
    const traces = [three_claim_trace("t0")]
    const verdicts = build_trace_reviews(traces)
    verdicts[0].claim_verdicts[0] = { agrees: true, why: "" }
    verdicts[0].claim_verdicts[1] = { agrees: false, why: "" }
    const { container } = render_review(traces, { verdicts })
    expect(open_claims(container)).toEqual([1, 2])
  })
})

describe("ClaimEvidenceReview — progress", () => {
  it("counts decided claims for the case and across the subset", async () => {
    const { container } = render_review([built_trace("t0"), built_trace("t1")])
    const case_count = () =>
      by_id(container, "review-case-count").textContent?.trim()
    expect(case_count()).toBe("0/2 decided")
    expect(position_line(container)).toBe("Case 1 of 2 · 0/4 decided")

    // A disagree counts as decided before its reason is in; Next still waits
    // for the reason.
    await fireEvent.click(by_id(container, "claim-disagree-0"))
    expect(case_count()).toBe("1/2 decided")
    expect(position_line(container)).toBe("Case 1 of 2 · 1/4 decided")
  })

  it("jumps to a case from its pill and colours current, reviewed and pending", async () => {
    const on_open_trace = vi.fn()
    const traces = [built_trace("t0"), built_trace("t1"), built_trace("t2")]
    const { container } = render_review(traces, { on_open_trace })
    const steps = () => steps_of(container)
    expect(step_labels(container)).toEqual(["Case 1", "Case 2", "Case 3"])

    await agree_all(container, 2)
    await fireEvent.click(steps()[2])
    expect(current_step(container)).toBe("3")
    expect(on_open_trace).toHaveBeenLastCalledWith(2)

    // Three states, three fills, and each pill is the same pill shape.
    const [reviewed, pending, current] = steps()
    expect(current.getAttribute("aria-current")).toBe("step")
    expect(fill_of(current)).toContain("bg-primary")
    expect(fill_of(pending)).toContain("bg-neutral")
    expect(fill_of(reviewed)).toContain("bg-success/60")
    for (const pill of steps()) {
      // A thin bar, in a target twice its height — margin around the bar would
      // space it the same way but leave the target as thin as the bar.
      expect(fill_of(pill)).toContain("rounded-full")
      expect(fill_of(pill)).toContain("w-8")
      expect(fill_of(pill)).toContain("h-2")
      expect(pill.className).toContain("h-4")
    }
  })

  it("greens a reviewed case whether the reviewer agreed or disagreed", async () => {
    // The colour tracks that the case is done, not that the judge was right,
    // so a case graded with a disagreement is as green as one fully agreed.
    const traces = [three_claim_trace("t0"), built_trace("t1")]
    const { container, getByText } = render_review(traces)
    await fireEvent.click(by_id(container, "claim-disagree-0"))
    await fireEvent.input(by_id(container, "claim-why-0"), {
      target: { value: "The judge read the transcript wrong." },
    })
    await fireEvent.click(by_id(container, "claim-agree-1"))
    await fireEvent.click(by_id(container, "claim-agree-2"))

    // Still the open case, so it is primary: current outranks reviewed.
    expect(fill_of(steps_of(container)[0])).toContain("bg-primary")

    await fireEvent.click(next_button(getByText))
    expect(current_step(container)).toBe("2")
    expect(fill_of(steps_of(container)[0])).toContain("bg-success")
  })

  it("counts and numbers only the cases under review", () => {
    const traces = ["t0", "t1", "t2", "t3"].map((id) => built_trace(id))
    const verdicts = build_trace_reviews(traces)
    verdicts[0].claim_verdicts[0] = { agrees: true, why: "" }
    const { container } = render_review(traces, {
      verdicts,
      selected_indices: [1, 3],
    })
    expect(current_step(container)).toBe("1")
    // The batch count spans the whole batch; the position counts the subset.
    expect(position_line(container)).toBe("Case 1 of 2 · 0/4 decided")
    expect(step_labels(container)).toEqual(["Case 1", "Case 2"])
  })
})

describe("ClaimEvidenceReview — keyboard", () => {
  it("A and D answer the claim reached last", async () => {
    const { container, verdicts } = render_review([three_claim_trace("t0")])
    const slots = verdicts[0].claim_verdicts

    await fireEvent.keyDown(window, { key: "a" })
    expect(slots[0].agrees).toBe(true)

    // D disagrees with the second claim and opens the third, which the next
    // key answers rather than the disagree above it.
    await fireEvent.keyDown(window, { key: "d" })
    expect(slots[1].agrees).toBe(false)
    // D puts the focus in the reason box, where keys are typing; once the
    // reviewer leaves the box, the next key answers the claim below.
    await new Promise((resolve) => setTimeout(resolve, 0))
    const why = by_id(container, "claim-why-1")
    expect(document.activeElement).toBe(why)
    why.blur()
    await fireEvent.keyDown(window, { key: "A" })
    expect(slots[2].agrees).toBe(true)
    expect(slots[1].agrees).toBe(false)
  })

  it("leaves A and D alone while typing, with a modifier or a held key, or with a dialog open", async () => {
    const { container, verdicts } = render_review([three_claim_trace("t0")])
    const slots = verdicts[0].claim_verdicts
    await fireEvent.click(by_id(container, "claim-disagree-0"))

    await fireEvent.keyDown(by_id(container, "claim-why-0"), { key: "a" })
    expect(slots[1].agrees).toBeNull()
    for (const modifier of ["metaKey", "ctrlKey", "altKey"]) {
      await fireEvent.keyDown(window, { key: "a", [modifier]: true })
      expect(slots[1].agrees).toBeNull()
    }
    await fireEvent.keyDown(window, { key: "a", repeat: true })
    expect(slots[1].agrees).toBeNull()

    await fireEvent.click(by_id(container, "view-full-trace"))
    expect(trace_dialog(container).open).toBe(true)
    await fireEvent.keyDown(window, { key: "a" })
    expect(slots[1].agrees).toBeNull()
  })
})
