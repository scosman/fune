// @vitest-environment jsdom
import {
  describe,
  it,
  expect,
  afterEach,
  afterAll,
  beforeAll,
  vi,
} from "vitest"
import { render, cleanup, fireEvent } from "@testing-library/svelte"
import ChatTrace from "./chat_trace.svelte"
import type { Trace as TraceType, TraceMessage } from "$lib/types"

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

// jsdom's <dialog> doesn't keep the `open` flag in sync with
// showModal()/close() reliably (and didn't implement them at all before
// jsdom ~22). Force-install minimal stubs we control so we can observe
// open/close state. We restore originals in afterAll to avoid leaking.
let original_show_modal: unknown
let original_close: unknown
let installed_polyfill = false
beforeAll(() => {
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
  installed_polyfill = true
})
afterAll(() => {
  if (!installed_polyfill) return
  const proto = HTMLDialogElement.prototype as unknown as Record<
    string,
    unknown
  >
  proto.showModal = original_show_modal
  proto.close = original_close
  installed_polyfill = false
})

function userMsg(content: string): TraceMessage {
  return { role: "user", content } as TraceMessage
}

function assistantMsg(
  content: string | null,
  extras: Partial<{
    tool_calls: unknown[]
    reasoning_content: string
    usage: unknown
    latency_ms: number
  }> = {},
): TraceMessage {
  return {
    role: "assistant",
    content,
    ...extras,
  } as TraceMessage
}

function systemMsg(content: string): TraceMessage {
  return { role: "system", content } as TraceMessage
}

function toolMsg(
  content: string,
  tool_call_id = "call_1",
  extras: Partial<{ is_error: boolean; kiln_task_tool_data: string }> = {},
): TraceMessage {
  return { role: "tool", content, tool_call_id, ...extras } as TraceMessage
}

function makeToolCall(
  id: string,
  name: string,
  args: Record<string, unknown> = {},
) {
  return {
    id,
    type: "function" as const,
    function: { name, arguments: JSON.stringify(args) },
  }
}

// A highlight that only expands and scrolls to a bubble looks exactly like a
// broken citation, so every such case has to log. Tests that reach one stub
// console.warn (the warning is expected output, not suite noise) and read the
// reason code back from the message.
function stub_warn() {
  return vi.spyOn(console, "warn").mockImplementation(() => {})
}

function warned_no_mark(warn: ReturnType<typeof stub_warn>): boolean {
  return warn.mock.calls.some((call) =>
    String(call[0]).includes("no_mark_drawn"),
  )
}

describe("ChatTrace component — layout & roles", () => {
  it("right-aligns user messages and left-aligns assistant messages", () => {
    const trace: TraceType = [userMsg("hello"), assistantMsg("hi there")]
    const { container } = render(ChatTrace, { props: { trace } })
    const userWrap = container.querySelector(
      "[data-testid='chat-msg-user']",
    ) as HTMLElement
    const asstWrap = container.querySelector(
      "[data-testid='chat-msg-assistant']",
    ) as HTMLElement
    expect(userWrap).not.toBeNull()
    expect(asstWrap).not.toBeNull()
    expect(userWrap.className).toContain("items-end")
    expect(asstWrap.className).toContain("items-start")
  })

  it("does NOT render tool messages as their own bubble", () => {
    const trace: TraceType = [
      userMsg("u"),
      assistantMsg(null, { tool_calls: [makeToolCall("c1", "lookup")] }),
      toolMsg('{"output": "42"}', "c1"),
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    // No standalone tool bubble — tool results nest inside the assistant turn.
    const bubbles = container.querySelectorAll(
      "[data-testid='chat-msg-user'], [data-testid='chat-msg-assistant'], [data-testid='chat-msg-system']",
    )
    expect(bubbles.length).toBe(2)
  })

  it("renders user content as markdown by default", () => {
    const trace: TraceType = [userMsg("**bold**")]
    const { container } = render(ChatTrace, { props: { trace } })
    const strong = container.querySelector("strong")
    expect(strong).not.toBeNull()
    expect(strong?.textContent).toBe("bold")
  })

  it("renders assistant content as markdown by default", () => {
    const trace: TraceType = [assistantMsg("**bold answer**")]
    const { container } = render(ChatTrace, { props: { trace } })
    const strong = container.querySelector("strong")
    expect(strong).not.toBeNull()
    expect(strong?.textContent).toBe("bold answer")
  })

  it("splits an assistant message with reasoning + content into two bubbles (reasoning first)", () => {
    const trace: TraceType = [
      assistantMsg("the answer", { reasoning_content: "step-by-step" }),
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    const bubbles = container.querySelectorAll(
      "[data-testid='chat-msg-assistant']",
    )
    expect(bubbles.length).toBe(2)
    expect(
      bubbles[0].querySelector("[data-testid='chat-msg-thinking']"),
    ).not.toBeNull()
    expect(
      bubbles[0].querySelector("[data-testid='chat-msg-content']"),
    ).toBeNull()
    expect(
      bubbles[1].querySelector("[data-testid='chat-msg-content']"),
    ).not.toBeNull()
    expect(
      bubbles[1].querySelector("[data-testid='chat-msg-thinking']"),
    ).toBeNull()
  })

  it("renders a single reasoning-only bubble when the assistant has only reasoning", () => {
    const trace: TraceType = [
      assistantMsg(null, { reasoning_content: "thinking out loud" }),
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    const bubbles = container.querySelectorAll(
      "[data-testid='chat-msg-assistant']",
    )
    expect(bubbles.length).toBe(1)
    expect(
      bubbles[0].querySelector("[data-testid='chat-msg-thinking']"),
    ).not.toBeNull()
    expect(
      bubbles[0].querySelector("[data-testid='chat-msg-content']"),
    ).toBeNull()
  })

  it("splits reasoning + content + N tool calls into 2 + N bubbles", () => {
    const trace: TraceType = [
      assistantMsg("here's the plan", {
        reasoning_content: "thinking",
        tool_calls: [makeToolCall("c1", "lookup"), makeToolCall("c2", "fetch")],
      }),
      toolMsg('{"output": "r1"}', "c1"),
      toolMsg('{"output": "r2"}', "c2"),
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    const bubbles = container.querySelectorAll(
      "[data-testid='chat-msg-assistant']",
    )
    expect(bubbles.length).toBe(4)
    expect(
      bubbles[0].querySelector("[data-testid='chat-msg-thinking']"),
    ).not.toBeNull()
    expect(
      bubbles[1].querySelector("[data-testid='chat-msg-content']"),
    ).not.toBeNull()
    expect(
      bubbles[2].querySelector("[data-testid='chat-msg-toolcall']"),
    ).not.toBeNull()
    expect(
      bubbles[3].querySelector("[data-testid='chat-msg-toolcall']"),
    ).not.toBeNull()
  })

  it("splits an assistant message with content + N tool calls into 1 + N bubbles", () => {
    const trace: TraceType = [
      assistantMsg("here is what I'll do", {
        tool_calls: [makeToolCall("c1", "lookup"), makeToolCall("c2", "fetch")],
      }),
      toolMsg('{"output": "r1"}', "c1"),
      toolMsg('{"output": "r2"}', "c2"),
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    const bubbles = container.querySelectorAll(
      "[data-testid='chat-msg-assistant']",
    )
    expect(bubbles.length).toBe(3)
    // First bubble holds the content; following bubbles each hold one tool call.
    expect(
      bubbles[0].querySelector("[data-testid='chat-msg-content']"),
    ).not.toBeNull()
    expect(
      bubbles[0].querySelector("[data-testid='chat-msg-toolcall']"),
    ).toBeNull()
    expect(
      bubbles[1].querySelector("[data-testid='chat-msg-toolcall']"),
    ).not.toBeNull()
    expect(
      bubbles[2].querySelector("[data-testid='chat-msg-toolcall']"),
    ).not.toBeNull()
  })

  it("renders a single bubble when an assistant message has only content (no tool_calls)", () => {
    const trace: TraceType = [assistantMsg("just an answer")]
    const { container } = render(ChatTrace, { props: { trace } })
    const bubbles = container.querySelectorAll(
      "[data-testid='chat-msg-assistant']",
    )
    expect(bubbles.length).toBe(1)
  })

  it("renders one bubble per tool call when an assistant message has only tool_calls", () => {
    const trace: TraceType = [
      assistantMsg(null, {
        tool_calls: [makeToolCall("c1", "lookup"), makeToolCall("c2", "fetch")],
      }),
      toolMsg('{"output": "r1"}', "c1"),
      toolMsg('{"output": "r2"}', "c2"),
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    const bubbles = container.querySelectorAll(
      "[data-testid='chat-msg-assistant']",
    )
    expect(bubbles.length).toBe(2)
  })

  it("renders a usage actions row for the assistant turn, independent of thinking expansion", () => {
    const trace: TraceType = [
      assistantMsg(null, {
        reasoning_content: "deep thought",
        usage: { input_tokens: 10, output_tokens: 5 },
      }),
    ]
    const { container } = render(ChatTrace, {
      props: { trace, show_per_message_usage: true },
    })
    // One actions row for the turn, present without expanding (it's revealed
    // on hover via CSS, not gated on the thinking toggle).
    const metas = container.querySelectorAll("[data-testid='chat-msg-meta']")
    expect(metas.length).toBe(1)
    expect(
      metas[0].querySelector("button[aria-label='View turn usage']"),
    ).not.toBeNull()
  })

  it("renders a single usage actions row per assistant turn, outside the bubbles", () => {
    const trace: TraceType = [
      assistantMsg("answer", {
        tool_calls: [makeToolCall("c1", "lookup"), makeToolCall("c2", "fetch")],
        usage: { input_tokens: 10, output_tokens: 5 },
      }),
      toolMsg('{"output": "ok1"}', "c1"),
      toolMsg('{"output": "ok2"}', "c2"),
    ]
    const { container } = render(ChatTrace, {
      props: { trace, show_per_message_usage: true },
    })
    // Exactly one actions row for the whole turn — not one per bubble.
    const metas = container.querySelectorAll("[data-testid='chat-msg-meta']")
    expect(metas.length).toBe(1)
    // And it lives outside the individual message bubbles.
    const bubbles = container.querySelectorAll(
      "[data-testid='chat-msg-assistant']",
    )
    expect(bubbles.length).toBe(3)
    bubbles.forEach((b) => {
      expect(b.querySelector("[data-testid='chat-msg-meta']")).toBeNull()
    })
  })

  it("places the fork affordance on the assistant message, not the user bubble", () => {
    const trace: TraceType = [
      userMsg("hello"),
      assistantMsg("hi"),
      userMsg("again"),
    ]
    // The fork affordance is mapped onto the assistant message (index 1) that
    // precedes the forkable user turn.
    const forkable_run_ids = [null, "run-2", null]
    const { container } = render(ChatTrace, {
      props: { trace, forkable_run_ids, on_fork: vi.fn() },
    })
    const userBubbles = container.querySelectorAll(
      "[data-testid='chat-msg-user']",
    )
    expect(userBubbles.length).toBe(2)
    // No fork button inside either user bubble.
    userBubbles.forEach((b) => {
      expect(
        b.querySelector("button[aria-label='Fork from this turn']"),
      ).toBeNull()
    })
    // The fork button lives on the assistant turn.
    const assistantTurns = container.querySelectorAll(
      "[data-testid='chat-msg-assistant-turn']",
    )
    expect(
      assistantTurns[0].querySelector(
        "button[aria-label='Fork from this turn']",
      ),
    ).not.toBeNull()
  })
})

describe("ChatTrace component — JSON content rendering", () => {
  // Output renders into a <pre> with hljs spans; ChatMarkdown renders a
  // .chat-markdown wrapper. Which one a bubble used is the assertion here.
  function contentEl(container: HTMLElement): HTMLElement {
    return container.querySelector(
      "[data-testid='chat-msg-content'], [data-testid='chat-msg-user']",
    ) as HTMLElement
  }
  function expectJsonRender(container: HTMLElement, expected_text: string) {
    const el = contentEl(container)
    const pre = el.querySelector("pre")
    expect(pre).not.toBeNull()
    expect(el.querySelector(".chat-markdown")).toBeNull()
    expect(pre?.textContent).toBe(expected_text)
    // Syntax highlighting applied, matching the other JSON surfaces.
    expect(pre?.innerHTML).toContain("hljs-")
  }
  function expectMarkdownRender(container: HTMLElement) {
    const el = contentEl(container)
    expect(el.querySelector(".chat-markdown")).not.toBeNull()
    expect(el.querySelector("pre")).toBeNull()
  }

  it("renders a pretty-printed JSON object as a highlighted code block", () => {
    const raw = '{\n  "answer": "yes",\n  "score": 3\n}'
    const { container } = render(ChatTrace, {
      props: { trace: [assistantMsg(raw)] as TraceType },
    })
    expectJsonRender(container, raw)
  })

  it("pretty-prints a minified JSON object", () => {
    const { container } = render(ChatTrace, {
      props: {
        trace: [assistantMsg('{"answer":"yes","score":3}')] as TraceType,
      },
    })
    expectJsonRender(container, '{\n  "answer": "yes",\n  "score": 3\n}')
  })

  it("renders a JSON array as a highlighted code block", () => {
    const { container } = render(ChatTrace, {
      props: { trace: [assistantMsg("[1,2,3]")] as TraceType },
    })
    expectJsonRender(container, "[\n  1,\n  2,\n  3\n]")
  })

  it("renders a bare JSON number as a highlighted code block", () => {
    const { container } = render(ChatTrace, {
      props: { trace: [assistantMsg("42")] as TraceType },
    })
    expectJsonRender(container, "42")
  })

  it("keeps a quoted JSON string scalar on the markdown path", () => {
    const { container } = render(ChatTrace, {
      props: { trace: [assistantMsg('"just some text"')] as TraceType },
    })
    expectMarkdownRender(container)
  })

  it("keeps plain prose on the markdown path", () => {
    const { container } = render(ChatTrace, {
      props: {
        trace: [assistantMsg("**bold** and a list:\n- one")] as TraceType,
      },
    })
    expectMarkdownRender(container)
    expect(container.querySelector("strong")?.textContent).toBe("bold")
  })

  it("keeps prose with an inline JSON snippet on the markdown path", () => {
    const { container } = render(ChatTrace, {
      props: {
        trace: [
          assistantMsg('The tool returned {"ok": true} which means success.'),
        ] as TraceType,
      },
    })
    expectMarkdownRender(container)
  })

  it("falls back to markdown for malformed / truncated JSON", () => {
    const { container } = render(ChatTrace, {
      props: { trace: [assistantMsg('{"answer": "yes", "sco')] as TraceType },
    })
    expectMarkdownRender(container)
  })

  it("renders JSON in user bubbles too", () => {
    const { container } = render(ChatTrace, {
      props: { trace: [userMsg('{"city":"Paris"}')] as TraceType },
    })
    expectJsonRender(container, '{\n  "city": "Paris"\n}')
  })

  it("renders JSON reasoning through Output, matching the legacy trace view", async () => {
    const { container } = render(ChatTrace, {
      props: {
        trace: [
          assistantMsg("done", { reasoning_content: '{"plan":["a","b"]}' }),
        ] as TraceType,
      },
    })
    await fireEvent.click(
      container.querySelector(
        "[data-testid='chat-msg-thinking'] button",
      ) as HTMLElement,
    )
    const thinking = container.querySelector(
      "[data-testid='chat-msg-thinking']",
    ) as HTMLElement
    const pre = thinking.querySelector("pre")
    expect(pre).not.toBeNull()
    expect(thinking.querySelector(".chat-markdown")).toBeNull()
    expect(pre?.textContent).toBe('{\n  "plan": [\n    "a",\n    "b"\n  ]\n}')
  })

  it("keeps prose reasoning on the markdown path", async () => {
    const { container } = render(ChatTrace, {
      props: {
        trace: [
          assistantMsg("done", { reasoning_content: "**weighing** options" }),
        ] as TraceType,
      },
    })
    await fireEvent.click(
      container.querySelector(
        "[data-testid='chat-msg-thinking'] button",
      ) as HTMLElement,
    )
    const thinking = container.querySelector(
      "[data-testid='chat-msg-thinking']",
    ) as HTMLElement
    expect(thinking.querySelector(".chat-markdown")).not.toBeNull()
    expect(thinking.querySelector("pre")).toBeNull()
  })

  // The user tint identifies the sender, so it survives both content shapes:
  // the JSON panel mounts transparent rather than repainting the bubble.
  it.each([
    ["prose", "hello there"],
    ["JSON", '{"city":"Paris"}'],
  ])("keeps the user tint for %s turns", (_label, content) => {
    const { container } = render(ChatTrace, {
      props: { trace: [userMsg(content)] as TraceType },
    })
    const bubble = container.querySelector(
      "[data-testid='chat-msg-user'] > div",
    ) as HTMLElement
    expect(bubble.className).toContain("bg-primary/10")
    // The mounted Output must not paint its own panel over the tint.
    expect(
      bubble.querySelector("[class*='bg-base-200'], [class*='bg-white']"),
    ).toBeNull()
  })

  it("does not route tool results through the content JSON branch", async () => {
    const trace: TraceType = [
      assistantMsg(null, { tool_calls: [makeToolCall("c1", "lookup")] }),
      toolMsg('{"output": {"temp_c": 21}}', "c1"),
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    await fireEvent.click(
      container.querySelector(
        "[data-testid='chat-msg-toolcall'] button",
      ) as HTMLElement,
    )
    // The tool result keeps mounting Output from inside the tool-call bubble,
    // pretty-printed, exactly as before.
    const toolCall = container.querySelector(
      "[data-testid='chat-tool-call']",
    ) as HTMLElement
    const pres = [...toolCall.querySelectorAll("pre")].map((p) => p.textContent)
    expect(pres).toContain('{\n  "temp_c": 21\n}')
  })

  it("renders a JSON bubble via Output when a DIFFERENT message is cited", () => {
    const trace: TraceType = [
      assistantMsg("some prose to cite"),
      assistantMsg('{"answer": "yes"}'),
    ]
    const { container } = render(ChatTrace, {
      props: {
        trace,
        highlight: {
          trace_index: 0,
          kind: "content" as const,
          start: 0,
          end: 4,
        },
      },
    })
    const bubbles = [
      ...container.querySelectorAll("[data-testid='chat-msg-content']"),
    ]
    expect(bubbles.length).toBe(2)
    expect(
      bubbles[0].querySelector("mark[data-highlight-target]"),
    ).not.toBeNull()
    const pre = bubbles[1].querySelector("pre")
    expect(pre).not.toBeNull()
    expect(pre?.textContent).toBe('{\n  "answer": "yes"\n}')
  })

  it("citation highlight still wins over the JSON path", () => {
    const raw = '{"answer": "yes"}'
    const { container } = render(ChatTrace, {
      props: {
        trace: [assistantMsg(raw)] as TraceType,
        highlight: {
          trace_index: 0,
          kind: "content" as const,
          start: 0,
          end: 5,
        },
      },
    })
    const el = contentEl(container)
    expect(el.querySelector("mark[data-highlight-target]")).not.toBeNull()
    expect(el.querySelector("pre")).toBeNull()
  })
})

describe("ChatTrace component — thinking", () => {
  it("starts with thinking collapsed", () => {
    const trace: TraceType = [
      assistantMsg("done", { reasoning_content: "step-by-step plan" }),
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    // Header is visible.
    expect(container.textContent).toContain("Thinking")
    // Body text is NOT visible until expanded.
    expect(container.textContent).not.toContain("step-by-step plan")
  })

  it("expands thinking when the user clicks anywhere on the bubble (not just the toggle)", async () => {
    const trace: TraceType = [
      assistantMsg("done", { reasoning_content: "secret thought" }),
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    const thinkingTestid = container.querySelector(
      "[data-testid='chat-msg-thinking']",
    ) as HTMLElement
    // The bubble is the parent of the chat-msg-thinking marker.
    const bubble = thinkingTestid.parentElement as HTMLElement
    expect(container.textContent).not.toContain("secret thought")
    await fireEvent.click(bubble)
    expect(container.textContent).toContain("secret thought")
  })

  it("clicking the toggle inside the bubble while expanded collapses it (does not re-expand via container click)", async () => {
    const trace: TraceType = [
      assistantMsg("done", { reasoning_content: "secret thought" }),
    ]
    const { container, getByText } = render(ChatTrace, { props: { trace } })
    // Expand via container click.
    const thinkingTestid = container.querySelector(
      "[data-testid='chat-msg-thinking']",
    ) as HTMLElement
    const bubble = thinkingTestid.parentElement as HTMLElement
    await fireEvent.click(bubble)
    expect(container.textContent).toContain("secret thought")
    // Now click the toggle button. The button's |stopPropagation must keep
    // the container's expand-only handler from re-expanding.
    await fireEvent.click(getByText("Thinking"))
    expect(container.textContent).not.toContain("secret thought")
  })

  it("expands thinking when the toggle is clicked", async () => {
    const trace: TraceType = [
      assistantMsg("done", { reasoning_content: "deep thought" }),
    ]
    const { container, getByText } = render(ChatTrace, { props: { trace } })
    await fireEvent.click(getByText("Thinking"))
    expect(container.textContent).toContain("deep thought")
  })

  it("renders thinking content as markdown when expanded", async () => {
    const trace: TraceType = [
      assistantMsg("done", { reasoning_content: "**emphasis**" }),
    ]
    const { container, getByText } = render(ChatTrace, { props: { trace } })
    await fireEvent.click(getByText("Thinking"))
    const strong = container.querySelector(
      "[data-testid='chat-msg-thinking'] strong",
    )
    expect(strong).not.toBeNull()
    expect(strong?.textContent).toBe("emphasis")
  })

  it("omits the thinking section when reasoning_content is empty", () => {
    const trace: TraceType = [assistantMsg("just an answer")]
    const { container } = render(ChatTrace, { props: { trace } })
    expect(
      container.querySelector("[data-testid='chat-msg-thinking']"),
    ).toBeNull()
  })
})

describe("ChatTrace component — citations inside a tool call", () => {
  // The flattened block a tool-call citation's offsets index, exactly as the
  // server renders it: `- Tool Name: {name}\n- Arguments: {args}`.
  const ARGS = '{"a": 135.0, "b": 0.15}'
  const PREFIX = `- Tool Name: multiply\n- Arguments: `

  function toolCallTrace(): TraceType {
    return [
      userMsg("Work out the refund."),
      assistantMsg(null, {
        tool_calls: [makeToolCallRaw("c1", "multiply", ARGS)],
      }),
      toolMsg('{"output": "20.25"}', "c1"),
    ]
  }

  function makeToolCallRaw(id: string, name: string, args: string) {
    return {
      id,
      type: "function" as const,
      function: { name, arguments: args },
    }
  }

  it("marks the cited arguments without disturbing the JSON rendering", () => {
    const { container } = render(ChatTrace, {
      props: {
        trace: toolCallTrace(),
        highlight: {
          trace_index: 1,
          kind: "tool_calls",
          start: PREFIX.length,
          end: PREFIX.length + ARGS.length,
        },
      },
    })

    const marks = container.querySelectorAll("mark")
    expect(marks.length).toBeGreaterThan(0)
    // The JSON is still pretty-printed and syntax-highlighted, which is the
    // point: a cited call looks like an uncited one, plus the mark.
    const card = container.querySelector("[data-testid='chat-tool-call']")
    expect(card?.querySelector(".hljs-attr")).not.toBeNull()
    expect(card?.textContent).toContain('"a"')
  })

  it("marks a single value when only that value is cited", () => {
    const value_start = PREFIX.length + ARGS.indexOf("0.15")
    const { container } = render(ChatTrace, {
      props: {
        trace: toolCallTrace(),
        highlight: {
          trace_index: 1,
          kind: "tool_calls",
          start: value_start,
          end: value_start + "0.15".length,
        },
      },
    })
    const marked = [...container.querySelectorAll("mark")]
      .map((m) => m.textContent)
      .join("")
    expect(marked).toBe("0.15")
  })

  it("draws no mark when the citation covers the tool NAME, which the card does not show as text", () => {
    const { container } = render(ChatTrace, {
      props: {
        trace: toolCallTrace(),
        highlight: {
          trace_index: 1,
          kind: "tool_calls",
          start: 0,
          end: "- Tool Name: multiply".length,
        },
      },
    })
    expect(container.querySelectorAll("mark").length).toBe(0)
  })

  it("keeps a cited tool RESULT's formatting instead of falling back to raw text", () => {
    const { container } = render(ChatTrace, {
      props: {
        trace: toolCallTrace(),
        highlight: {
          trace_index: 2,
          kind: "tool_result",
          start: 0,
          end: "20.25".length,
        },
      },
    })
    const marked = [...container.querySelectorAll("mark")]
      .map((m) => m.textContent)
      .join("")
    expect(marked).toBe("20.25")
    // Rendered through Output, not as a plain-text fallback.
    expect(container.querySelector(".hljs-number")).not.toBeNull()
  })
})

describe("ChatTrace component — structured output", () => {
  // A structured-output task returns its answer as arguments to the internal
  // `task_response` tool. The message shape here is copied from a real run:
  // no content, one tool call, the answer as its JSON arguments.
  const ANSWER =
    '{"category": "refund_status", "priority": "normal", "summary": "Refund status for order #987654."}'

  function structuredOutputTrace(): TraceType {
    return [
      userMsg("Hi, my order number is #987654. When will my refund arrive?"),
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "toolu_01",
            type: "function" as const,
            function: { name: "task_response", arguments: ANSWER },
          },
        ],
      } as TraceMessage,
    ]
  }

  it("renders the answer as Output, not as a tool call the agent chose to make", () => {
    const { container } = render(ChatTrace, {
      props: { trace: structuredOutputTrace() },
    })

    expect(
      container.querySelector("[data-testid='chat-msg-structured-output']"),
    ).not.toBeNull()
    // The user never wrote a `task_response` tool, so it must not be presented
    // as one — neither the name nor the tool-call chrome may appear.
    expect(container.textContent).not.toContain("task_response")
    expect(container.textContent).not.toContain("Toolcall:")
    expect(
      container.querySelector("[data-testid='chat-msg-toolcall']"),
    ).toBeNull()
    expect(container.textContent).toContain("refund_status")
  })

  it("still renders a real tool call as a tool call in the same message", () => {
    // Guards the unwrap from over-reaching: only `task_response` is plumbing.
    const trace: TraceType = [
      userMsg("Work out the refund."),
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "c1",
            type: "function" as const,
            function: { name: "multiply", arguments: '{"a": 135, "b": 0.15}' },
          },
          {
            id: "toolu_01",
            type: "function" as const,
            function: { name: "task_response", arguments: ANSWER },
          },
        ],
      } as TraceMessage,
    ]
    const { container } = render(ChatTrace, { props: { trace } })

    const toolcalls = container.querySelectorAll(
      "[data-testid='chat-msg-toolcall']",
    )
    expect(toolcalls.length).toBe(1)
    expect(container.textContent).toContain("multiply")
    expect(
      container.querySelector("[data-testid='chat-msg-structured-output']"),
    ).not.toBeNull()
    expect(container.textContent).not.toContain("task_response")
  })
})

describe("ChatTrace component — tool calls", () => {
  it("renders one bubble per tool call with a 'Toolcall: {name}' label, collapsed by default", () => {
    const trace: TraceType = [
      assistantMsg(null, {
        tool_calls: [
          makeToolCall("c1", "lookup"),
          makeToolCall("c2", "fetch"),
          makeToolCall("c3", "save"),
        ],
      }),
      toolMsg('{"output": "r1"}', "c1"),
      toolMsg('{"output": "r2"}', "c2"),
      toolMsg('{"output": "r3"}', "c3"),
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    const tcBubbles = container.querySelectorAll(
      "[data-testid='chat-msg-toolcall']",
    )
    expect(tcBubbles.length).toBe(3)
    expect(container.textContent).toContain("Toolcall:")
    expect(container.textContent).toContain("lookup")
    expect(container.textContent).toContain("fetch")
    expect(container.textContent).toContain("save")
    // No expanded card visible yet.
    expect(
      container.querySelectorAll("[data-testid='chat-tool-call']").length,
    ).toBe(0)
    // No grouping summary like "3 tool calls" anymore.
    expect(container.textContent).not.toContain("3 tool calls")
  })

  it("expands a tool-call bubble when the user clicks anywhere on it (not just the toggle)", async () => {
    const trace: TraceType = [
      assistantMsg(null, {
        tool_calls: [makeToolCall("c1", "lookup", { q: "secret-arg" })],
      }),
      toolMsg('{"output": "ok"}', "c1"),
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    const tcTestid = container.querySelector(
      "[data-testid='chat-msg-toolcall']",
    ) as HTMLElement
    const bubble = tcTestid.parentElement as HTMLElement
    // No args visible while collapsed.
    expect(container.textContent).not.toContain("secret-arg")
    await fireEvent.click(bubble)
    expect(container.textContent).toContain("secret-arg")
  })

  it("expands a tool-call bubble and reveals args + matching tool result", async () => {
    const trace: TraceType = [
      assistantMsg(null, {
        tool_calls: [makeToolCall("c1", "lookup", { q: "kiln" })],
      }),
      toolMsg('{"output": "answer-from-tool"}', "c1"),
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    const tcBubble = container.querySelector(
      "[data-testid='chat-msg-toolcall']",
    ) as HTMLElement
    const toggle = tcBubble.querySelector("button") as HTMLButtonElement
    await fireEvent.click(toggle)
    const cards = container.querySelectorAll("[data-testid='chat-tool-call']")
    expect(cards.length).toBe(1)
    const card = cards[0] as HTMLElement
    expect(card.textContent).toContain("kiln")
    expect(card.textContent).toContain("Tool Result")
    expect(card.textContent).toContain("answer-from-tool")
  })

  it("expands tool-call bubbles independently", async () => {
    const trace: TraceType = [
      assistantMsg(null, {
        tool_calls: [
          makeToolCall("c1", "lookup", { q: "first" }),
          makeToolCall("c2", "fetch", { q: "second" }),
        ],
      }),
      toolMsg('{"output": "r1"}', "c1"),
      toolMsg('{"output": "r2"}', "c2"),
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    const tcBubbles = container.querySelectorAll(
      "[data-testid='chat-msg-toolcall']",
    )
    expect(tcBubbles.length).toBe(2)
    // Expand only the second one.
    const secondToggle = tcBubbles[1].querySelector(
      "button",
    ) as HTMLButtonElement
    await fireEvent.click(secondToggle)
    expect(
      container.querySelectorAll("[data-testid='chat-tool-call']").length,
    ).toBe(1)
    expect(container.textContent).toContain("second")
    expect(container.textContent).not.toContain("first")
  })

  it("marks the tool result as an error when the tool message reports is_error", async () => {
    const trace: TraceType = [
      assistantMsg(null, { tool_calls: [makeToolCall("c1", "lookup")] }),
      toolMsg("boom", "c1", { is_error: true }),
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    const toggle = container.querySelector(
      "[data-testid='chat-msg-toolcall'] button",
    ) as HTMLButtonElement
    await fireEvent.click(toggle)
    expect(container.textContent).toContain("Tool Error")
  })

  it("indicates when a tool call has no matching result", async () => {
    const trace: TraceType = [
      assistantMsg(null, { tool_calls: [makeToolCall("c1", "lookup")] }),
      // no matching tool message
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    const toggle = container.querySelector(
      "[data-testid='chat-msg-toolcall'] button",
    ) as HTMLButtonElement
    await fireEvent.click(toggle)
    expect(container.textContent).toContain("No tool result recorded")
  })
})

describe("ChatTrace component — system prompt", () => {
  it("does not render system messages at all (no toggle, no content)", () => {
    const trace: TraceType = [systemMsg("you are helpful"), userMsg("hi")]
    const { container } = render(ChatTrace, { props: { trace } })
    expect(
      container.querySelector("[data-testid='chat-msg-system']"),
    ).toBeNull()
    expect(container.textContent).not.toContain("you are helpful")
    expect(container.textContent).not.toContain("system prompt")
  })

  it("does not render developer messages either", () => {
    const trace: TraceType = [
      { role: "developer", content: "internal directive" } as TraceMessage,
      userMsg("hi"),
    ]
    const { container } = render(ChatTrace, { props: { trace } })
    expect(
      container.querySelector("[data-testid='chat-msg-system']"),
    ).toBeNull()
    expect(container.textContent).not.toContain("internal directive")
  })
})

describe("ChatTrace component — fork affordance", () => {
  function fork_button(container: HTMLElement) {
    return container.querySelectorAll<HTMLButtonElement>(
      'button[aria-label="Fork from this turn"]',
    )
  }

  it("renders a fork button on assistant messages that are forkable", () => {
    const trace: TraceType = [
      systemMsg("s"),
      userMsg("u1"),
      assistantMsg("a1"),
      userMsg("u2"),
      assistantMsg("a2"),
    ]
    // Mapped onto the assistant message (index 2) preceding the forkable turn.
    const forkable_run_ids = [null, null, "run-2", null, null]
    const { container } = render(ChatTrace, {
      props: { trace, forkable_run_ids, on_fork: vi.fn() },
    })
    expect(fork_button(container).length).toBe(1)
  })

  it("does NOT render a fork button on user messages even if forkable_run_ids[i] is set", () => {
    const trace: TraceType = [userMsg("u1"), assistantMsg("a1")]
    const forkable_run_ids = ["run-u", "run-a"]
    const { container } = render(ChatTrace, {
      props: { trace, forkable_run_ids, on_fork: vi.fn() },
    })
    // Only the assistant msg gets a fork button (the user-mapped id is ignored).
    expect(fork_button(container).length).toBe(1)
  })

  it("invokes on_fork with the mapped run id and assistant trace index", async () => {
    const trace: TraceType = [
      systemMsg("s"),
      userMsg("u1"),
      assistantMsg("a1"),
      userMsg("u2"),
    ]
    // Forking the new turn 2 is offered on the assistant message at index 2.
    const forkable_run_ids = [null, null, "run-leaf", null]
    const on_fork = vi.fn()
    const { container } = render(ChatTrace, {
      props: { trace, forkable_run_ids, on_fork },
    })
    const button = fork_button(container)[0]
    expect(button).toBeDefined()
    await fireEvent.click(button)
    expect(on_fork).toHaveBeenCalledTimes(1)
    expect(on_fork).toHaveBeenCalledWith("run-leaf", 2)
  })

  it("does NOT render any fork button when on_fork is not provided", () => {
    const trace: TraceType = [userMsg("u1"), assistantMsg("a1")]
    const forkable_run_ids = [null, "run-1"]
    const { container } = render(ChatTrace, {
      props: { trace, forkable_run_ids },
    })
    expect(fork_button(container).length).toBe(0)
  })

  it("hides messages at or after truncate_at_trace_index", () => {
    const trace: TraceType = [
      userMsg("u1"),
      assistantMsg("a1"),
      userMsg("u2"),
      assistantMsg("a2"),
    ]
    const { container } = render(ChatTrace, {
      props: { trace, truncate_at_trace_index: 2 },
    })
    // Only indices 0..1 render.
    const bubbles = container.querySelectorAll(
      "[data-testid='chat-msg-user'], [data-testid='chat-msg-assistant']",
    )
    expect(bubbles.length).toBe(2)
  })
})

describe("ChatTrace component — citation highlight", () => {
  it("renders unchanged (markdown, no <mark>) when highlight is null", () => {
    const trace: TraceType = [assistantMsg("**bold answer**")]
    const { container } = render(ChatTrace, {
      props: { trace, highlight: null },
    })
    // The dataset run page shares this component with no highlight: markdown
    // still renders and nothing is marked.
    expect(container.querySelector("strong")?.textContent).toBe("bold answer")
    expect(container.querySelector("mark")).toBeNull()
    expect(container.querySelector("[data-highlight-target]")).toBeNull()
  })

  it("wraps the cited span in <mark> in the matching content node", () => {
    const trace: TraceType = [
      userMsg("What is the return window?"),
      assistantMsg("Our return window is 30 days."),
    ]
    // Offsets index the raw content "Our return window is 30 days." — mark
    // "return window".
    const content = "Our return window is 30 days."
    const start = content.indexOf("return window")
    const end = start + "return window".length
    const { container } = render(ChatTrace, {
      props: {
        trace,
        highlight: { trace_index: 1, kind: "content", start, end },
      },
    })
    const contentNode = container.querySelector(
      "[data-testid='chat-msg-content']",
    ) as HTMLElement
    const mark = contentNode.querySelector("mark")
    expect(mark).not.toBeNull()
    expect(mark?.textContent).toBe("return window")
    // The marked node is the scroll target.
    expect(mark?.hasAttribute("data-highlight-target")).toBe(true)
  })

  it("auto-expands and marks a reasoning highlight", () => {
    const reasoning = "Let me think about policy."
    const trace: TraceType = [
      assistantMsg(null, { reasoning_content: reasoning }),
    ]
    const start = reasoning.indexOf("think")
    const end = start + "think about policy".length
    const { container } = render(ChatTrace, {
      props: {
        trace,
        highlight: { trace_index: 0, kind: "reasoning", start, end },
      },
    })
    // The thinking bubble expands itself so the cited moment is visible.
    const mark = container.querySelector(
      "[data-testid='chat-msg-thinking'] mark",
    )
    expect(mark?.textContent).toBe("think about policy")
  })

  it("marks the cited span in a user bubble", () => {
    // Input citations map onto the opening user message on multi-turn, so
    // the user bubble must be able to carry a mark like assistant content.
    const content = "I want to return my order from last week."
    const trace: TraceType = [userMsg(content), assistantMsg("Happy to help.")]
    const start = content.indexOf("return my order")
    const end = start + "return my order".length
    const { container } = render(ChatTrace, {
      props: {
        trace,
        highlight: { trace_index: 0, kind: "content", start, end },
      },
    })
    const userNode = container.querySelector(
      "[data-testid='chat-msg-user']",
    ) as HTMLElement
    const mark = userNode.querySelector("mark")
    expect(mark).not.toBeNull()
    expect(mark?.textContent).toBe("return my order")
    expect(mark?.hasAttribute("data-highlight-target")).toBe(true)
  })

  it("marks the cited span inside a tool result and targets the mark, not the bubble", () => {
    const output =
      "line one\nThe skill file loaded fine_tuning.md successfully.\nline three"
    const trace: TraceType = [
      assistantMsg(null, {
        tool_calls: [makeToolCall("call_1", "load_skill")],
      }),
      toolMsg(JSON.stringify({ output }), "call_1"),
      assistantMsg("Done."),
    ]
    const start = output.indexOf("loaded fine_tuning.md")
    const end = start + "loaded fine_tuning.md".length
    const { container } = render(ChatTrace, {
      props: {
        trace,
        highlight: { trace_index: 1, kind: "tool_result", start, end },
      },
    })
    // The owning tool-call bubble auto-expands and the exact span is marked
    // in the raw result text (pretty-print is dropped for the marked node).
    const toolCall = container.querySelector(
      "[data-testid='chat-tool-call']",
    ) as HTMLElement
    expect(toolCall).not.toBeNull()
    const mark = toolCall.querySelector("mark")
    expect(mark).not.toBeNull()
    expect(mark?.textContent).toBe("loaded fine_tuning.md")
    // The mark is the one scroll target; the bubble no longer is.
    const targets = container.querySelectorAll("[data-highlight-target]")
    expect(targets.length).toBe(1)
    expect(targets[0].tagName).toBe("MARK")
  })

  it("falls back to the bubble target when the displayed result diverges from the cited text", () => {
    // isError envelope without an output field: the chat displays the
    // unwrapped error while citation offsets index the raw JSON — a mark in
    // either string would be misplaced, so the bubble stays the target.
    const warn = stub_warn()
    const content = JSON.stringify({ isError: true, error: "boom" })
    const trace: TraceType = [
      assistantMsg(null, {
        tool_calls: [makeToolCall("call_1", "load_skill")],
      }),
      toolMsg(content, "call_1"),
    ]
    const { container } = render(ChatTrace, {
      props: {
        trace,
        highlight: { trace_index: 1, kind: "tool_result", start: 0, end: 4 },
      },
    })
    expect(container.querySelector("mark")).toBeNull()
    const target = container.querySelector("[data-highlight-target]")
    expect(target).not.toBeNull()
    expect(target?.tagName).not.toBe("MARK")
    expect(warned_no_mark(warn)).toBe(true)
  })

  it("marks an isError envelope that still carries a string output", () => {
    // `output` wins over `isError` in both the flattener and the display,
    // so the mark must be sliced from the output string, not suppressed.
    const output = "partial data before the failure"
    const content = JSON.stringify({ isError: true, output })
    const trace: TraceType = [
      assistantMsg(null, {
        tool_calls: [makeToolCall("call_1", "load_skill")],
      }),
      toolMsg(content, "call_1"),
    ]
    const start = output.indexOf("before the failure")
    const end = start + "before the failure".length
    const { container } = render(ChatTrace, {
      props: {
        trace,
        highlight: { trace_index: 1, kind: "tool_result", start, end },
      },
    })
    const mark = container.querySelector("mark")
    expect(mark?.textContent).toBe("before the failure")
  })

  it("marks a plain non-JSON tool result", () => {
    // The commonest shape after the Kiln envelope: the content IS the raw
    // text the citation offsets index.
    const warn = stub_warn()
    const content = "HTTP 200\nbody: return window is 30 days"
    const trace: TraceType = [
      assistantMsg(null, {
        tool_calls: [makeToolCall("call_1", "fetch_policy")],
      }),
      toolMsg(content, "call_1"),
    ]
    const start = content.indexOf("30 days")
    const end = start + "30 days".length
    const { container } = render(ChatTrace, {
      props: {
        trace,
        highlight: { trace_index: 1, kind: "tool_result", start, end },
      },
    })
    const mark = container.querySelector("mark")
    expect(mark?.textContent).toBe("30 days")
    // A citation that really is marked has nothing to report.
    expect(warned_no_mark(warn)).toBe(false)
  })

  it("suppresses the mark when the envelope's output is not a string", () => {
    // The display pretty-prints the object while the offsets could only
    // index a raw string — no mark can be honest, so the bubble stays the
    // target.
    const warn = stub_warn()
    const content = JSON.stringify({ output: { rows: [1, 2, 3] } })
    const trace: TraceType = [
      assistantMsg(null, {
        tool_calls: [makeToolCall("call_1", "query_db")],
      }),
      toolMsg(content, "call_1"),
    ]
    const { container } = render(ChatTrace, {
      props: {
        trace,
        highlight: { trace_index: 1, kind: "tool_result", start: 0, end: 4 },
      },
    })
    expect(container.querySelector("mark")).toBeNull()
    expect(
      container.querySelector("[data-highlight-target]")?.tagName,
    ).not.toBe("MARK")
    expect(warned_no_mark(warn)).toBe(true)
  })

  it("falls back to the bubble when duplicate call ids point the display at a different message", () => {
    // Two tool messages share one call id (a retried call): the bubble
    // renders the LAST result, but the citation names the FIRST message's
    // trace index — slicing the displayed text would mark the wrong bytes.
    const warn = stub_warn()
    const first = "attempt one: timed out"
    const second = "attempt two: return window is 30 days"
    const trace: TraceType = [
      assistantMsg(null, {
        tool_calls: [makeToolCall("call_1", "fetch_policy")],
      }),
      toolMsg(JSON.stringify({ output: first }), "call_1"),
      toolMsg(JSON.stringify({ output: second }), "call_1"),
    ]
    const { container } = render(ChatTrace, {
      props: {
        trace,
        // trace_index 1 = the FIRST tool message.
        highlight: { trace_index: 1, kind: "tool_result", start: 0, end: 7 },
      },
    })
    expect(container.querySelector("mark")).toBeNull()
    const target = container.querySelector("[data-highlight-target]")
    expect(target).not.toBeNull()
    expect(target?.tagName).not.toBe("MARK")
    expect(warned_no_mark(warn)).toBe(true)
  })

  it("logs a tool-call citation, which lands on the whole bubble", () => {
    // A tool CALL renders through a component rather than plain text, so the
    // bubble only expands and scrolls — the one highlight kind that never
    // marks, whatever the result beside it looks like.
    const warn = stub_warn()
    const trace: TraceType = [
      assistantMsg(null, {
        tool_calls: [makeToolCall("call_1", "load_skill")],
      }),
    ]
    const { container } = render(ChatTrace, {
      props: {
        trace,
        highlight: { trace_index: 0, kind: "tool_calls", start: 0, end: 9 },
      },
    })
    expect(container.querySelector("mark")).toBeNull()
    expect(warned_no_mark(warn)).toBe(true)
  })
})

describe("ChatTrace component — usage info row", () => {
  it("renders a usage info button when show_per_message_usage and message has usage", () => {
    const trace: TraceType = [
      assistantMsg("answer", {
        usage: { input_tokens: 10, output_tokens: 5 },
      }),
    ]
    const { container } = render(ChatTrace, {
      props: { trace, show_per_message_usage: true },
    })
    expect(
      container.querySelector("button[aria-label='View turn usage']"),
    ).not.toBeNull()
  })

  it("labels the usage button so users know what it does", () => {
    const trace: TraceType = [
      assistantMsg("answer", {
        usage: { input_tokens: 10, output_tokens: 5 },
      }),
    ]
    const { container } = render(ChatTrace, {
      props: { trace, show_per_message_usage: true },
    })
    const button = container.querySelector(
      "button[aria-label='View turn usage']",
    ) as HTMLButtonElement
    expect(button).not.toBeNull()
    expect(button.textContent).toContain("Usage")
  })

  it("does not render usage button when show_per_message_usage is false", () => {
    const trace: TraceType = [
      assistantMsg("answer", {
        usage: { input_tokens: 10, output_tokens: 5 },
      }),
    ]
    const { container } = render(ChatTrace, {
      props: { trace, show_per_message_usage: false },
    })
    expect(
      container.querySelector("button[aria-label='View turn usage']"),
    ).toBeNull()
  })

  it("opens a usage breakdown dialog when the usage info button is clicked", async () => {
    const trace: TraceType = [
      assistantMsg("answer", {
        usage: {
          input_tokens: 10,
          output_tokens: 5,
          total_tokens: 15,
          cost: 0.0001234,
        },
        latency_ms: 1234,
      }),
    ]
    const { container, baseElement } = render(ChatTrace, {
      props: { trace, show_per_message_usage: true },
    })
    const button = container.querySelector(
      "button[aria-label='View turn usage']",
    ) as HTMLButtonElement
    await fireEvent.click(button)
    // The page mounts two Dialogs (subtask trace + usage info). Pick the
    // one whose heading is "Usage".
    const dialogs = Array.from(
      baseElement.querySelectorAll<HTMLDialogElement>("dialog"),
    )
    const usage_dialog = dialogs.find((d) =>
      (d.textContent ?? "").includes("Usage"),
    )
    expect(usage_dialog).toBeDefined()
    const text = usage_dialog!.textContent ?? ""
    expect(text).toContain("Cost")
    expect(text).toContain("$0.000123")
    expect(text).toContain("Total tokens")
    expect(text).toContain("15")
    expect(text).toContain("Input tokens")
    expect(text).toContain("Output tokens")
    expect(text).toContain("Latency")
  })
})
