// @vitest-environment jsdom
import {
  describe,
  it,
  expect,
  afterEach,
  beforeAll,
  afterAll,
  vi,
} from "vitest"
import { render, cleanup } from "@testing-library/svelte"
import { tick } from "svelte"
import ClaimTraceModal from "./claim_trace_modal.svelte"
import type { Citation, TraceClaims } from "./claim_evidence"
import type { TraceMessage } from "$lib/types"

// The happy path — an input citation landing on the opening user bubble of a
// conversation shown on its own — is pinned end-to-end through the review
// component (claim_evidence_review.test.ts); this file covers the modal's
// fallback and observability behavior.

const original_scroll_into_view = Element.prototype.scrollIntoView
let original_show_modal: unknown
let original_close: unknown
beforeAll(() => {
  if (!Element.prototype.scrollIntoView) {
    Element.prototype.scrollIntoView = function () {}
  }
  // jsdom's <dialog> doesn't reliably implement showModal()/close(); minimal
  // stubs keep the `open` flag in sync so the dialog's content renders.
  const proto = HTMLDialogElement.prototype as unknown as Record<
    string,
    unknown
  >
  original_show_modal = proto.showModal
  original_close = proto.close
  proto.showModal = function () {
    ;(this as unknown as { open: boolean }).open = true
  }
  proto.close = function () {
    ;(this as unknown as { open: boolean }).open = false
  }
})
afterAll(() => {
  Element.prototype.scrollIntoView = original_scroll_into_view
  const proto = HTMLDialogElement.prototype as unknown as Record<
    string,
    unknown
  >
  proto.showModal = original_show_modal
  proto.close = original_close
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

const RAW_INPUT = "I want to return my order from last week."

// A multi-turn conversation whose opening user message IS raw_input (the
// server derives raw_input from it verbatim) plus the flattener's rendering
// of the whole conversation as raw_output.
const CONVERSATION: TraceMessage[] = [
  { role: "user", content: RAW_INPUT },
  { role: "assistant", content: "Happy to help with the return." },
] as unknown as TraceMessage[]

const RAW_OUTPUT = [
  `user:\n<user_message>\n${RAW_INPUT}\n</user_message>`,
  "assistant:\n<assistant_message>\nHappy to help with the return.\n</assistant_message>",
].join("\n\n")

function multi_turn_trace(overrides: Partial<TraceClaims> = {}): TraceClaims {
  return {
    trace_id: "batch1_case_0",
    leaf_run_id: "run_0",
    raw_input: RAW_INPUT,
    raw_output: RAW_OUTPUT,
    judge_score: "pass",
    judge_reasoning: "Helped with the return.",
    overview: null,
    claims: [],
    claims_state: "built",
    claims_error: null,
    trace: CONVERSATION,
    ...overrides,
  }
}

// The reason codes a run of the modal logged. The code travels in the warning
// message so a console filter can find it; the tests read it back the same way.
function warn_reasons(warn: { mock: { calls: unknown[][] } }): string[] {
  return warn.mock.calls
    .map((call) => String(call[0]).match(/\(([a-z_]+)\)/)?.[1] ?? "")
    .filter(Boolean)
}

function input_citation(): Citation {
  return {
    marker: 1,
    source: "input",
    from: "return my order",
    to: "return my order",
  }
}

describe("claim_trace_modal — citation mapping fallbacks", () => {
  it("shows an unmappable input citation without a highlight", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    // An opening message that differs from raw_input: the chat mapping
    // refuses. The conversation is the whole modal on this arm, so there is
    // no second copy of the input to mark instead.
    const diverged = multi_turn_trace({
      trace: [
        { role: "user", content: "A different opening entirely." },
        { role: "assistant", content: "Happy to help with the return." },
      ] as unknown as TraceMessage[],
    })
    const { container, component } = render(ClaimTraceModal, {
      props: {},
    })
    component.open_citation(diverged, input_citation())
    await tick()

    expect(
      container.querySelector("[data-testid='chat-msg-user']"),
    ).not.toBeNull()
    expect(container.querySelector("mark")).toBeNull()
    // The silent miss is observable. One code covers every mapping refusal;
    // the payload's `source` is what says this was the input path.
    expect(warn_reasons(warn)).toContain("flattener_drift")
  })

  it("logs a citation whose anchors are nowhere in the source", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    const { container, component } = render(ClaimTraceModal, {
      props: {},
    })
    component.open_citation(multi_turn_trace(), {
      marker: 1,
      source: "output",
      from: "a sentence this trace never contained",
      to: "nor this one",
    })
    await tick()

    expect(container.querySelector("mark")).toBeNull()
    expect(warn_reasons(warn)).toContain("anchor_not_found")
  })

  it("marks a citation the strict resolver would have missed", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    const { container, component } = render(ClaimTraceModal, {
      props: {},
    })
    // Anchors retyped with a line break where the trace has a space: the
    // tolerant resolver finds them, and the block mapper still adjudicates.
    component.open_citation(multi_turn_trace(), {
      marker: 1,
      source: "output",
      from: "Happy to help\nwith",
      to: "the return",
    })
    await tick()

    expect(container.querySelector("mark")?.textContent).toBe(
      "Happy to help with the return",
    )
    expect(warn).not.toHaveBeenCalled()
  })

  it("marks the from anchor of a citation that spans two turns", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    const { container, component } = render(ClaimTraceModal, {
      props: {},
    })
    // Both anchors exist in raw_output but in different blocks (user turn →
    // assistant turn). The anchor the model wrote still has a home, so the
    // citation highlights there instead of dying.
    component.open_citation(multi_turn_trace(), {
      marker: 1,
      source: "output",
      from: "return my order",
      to: "Happy to help",
    })
    await tick()

    expect(container.querySelector("mark")?.textContent).toBe("return my order")
    // Partial, and the model wrote a citation across two turns — both worth
    // knowing about.
    expect(warn_reasons(warn)).toContain("spans_two_turns")
  })

  it("rejects a highlight onto a row the chat never draws", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    // The flattener emits a block for the system prompt but ChatTrace drops
    // system rows — passing the highlight through would target nothing,
    // silently. It must be treated as a miss instead.
    const system_text = "You must never invent policy."
    const trace_with_system = [
      { role: "system", content: system_text },
      ...CONVERSATION,
    ] as unknown as TraceMessage[]
    const raw_output = [
      `system:\n<system_message>\n${system_text}\n</system_message>`,
      RAW_OUTPUT,
    ].join("\n\n")
    const { container, component } = render(ClaimTraceModal, {
      props: {},
    })
    component.open_citation(
      multi_turn_trace({ trace: trace_with_system, raw_output }),
      {
        marker: 1,
        source: "output",
        from: "never invent policy",
        to: "never invent policy",
      },
    )
    await tick()

    expect(container.querySelector("mark")).toBeNull()
    expect(warn_reasons(warn)).toContain("row_not_rendered")
  })

  it("does not warn on an empty trace — the raw panels carry the mark", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    const { container, component } = render(ClaimTraceModal, {
      props: {},
    })
    component.open_citation(multi_turn_trace({ trace: [] }), input_citation())
    await tick()

    // No chat renders; the Input panel still marks the citation, now through
    // Output, which tags its own mark.
    const panel_mark = container.querySelector("[data-highlight-target]")
    expect(panel_mark?.textContent).toBe("return my order")
    expect(warn).not.toHaveBeenCalled()
  })
})

describe("claim_trace_modal — scroll position", () => {
  it("every open starts at the top, not the last view's offset", async () => {
    const { container, component } = render(ClaimTraceModal, {
      props: {},
    })
    component.open_trace(multi_turn_trace())
    await tick()
    const content = container.querySelector(".overflow-y-auto") as HTMLElement
    expect(content).not.toBeNull()

    // The container survives the close, offset and all; neither entry point
    // may inherit it.
    content.scrollTop = 400
    component.open_citation(multi_turn_trace(), input_citation())
    await tick()
    expect(content.scrollTop).toBe(0)

    content.scrollTop = 400
    component.open_trace(multi_turn_trace())
    await tick()
    expect(content.scrollTop).toBe(0)
  })
})
