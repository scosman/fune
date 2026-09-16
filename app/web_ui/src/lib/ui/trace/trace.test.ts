// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest"
import { render, cleanup, fireEvent } from "@testing-library/svelte"
import Trace from "./trace.svelte"
import type { Trace as TraceType, TraceMessage } from "$lib/types"

afterEach(() => cleanup())

function userMsg(content: string): TraceMessage {
  return { role: "user", content } as TraceMessage
}

function assistantMsg(
  content: string | null,
  extras: Partial<{
    tool_calls: unknown[]
    reasoning_content: string
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

function toolMsg(content: string, tool_call_id = "call_1"): TraceMessage {
  return { role: "tool", content, tool_call_id } as TraceMessage
}

async function expandAll(container: HTMLElement): Promise<void> {
  const inputs = container.querySelectorAll<HTMLInputElement>(
    "input[type=checkbox]",
  )
  for (const input of Array.from(inputs)) {
    await fireEvent.click(input)
  }
}

describe("Trace component", () => {
  it("renders one collapsible block per message in the trace", () => {
    const trace: TraceType = [
      systemMsg("you are helpful"),
      userMsg("hi"),
      assistantMsg("hello there"),
    ]
    const { container } = render(Trace, { props: { trace } })
    const collapses = container.querySelectorAll(".collapse")
    expect(collapses.length).toBe(3)
  })

  it("shows the role label for each message block", () => {
    const trace: TraceType = [
      systemMsg("s"),
      userMsg("u"),
      assistantMsg("a"),
      toolMsg('{"r":1}'),
    ]
    const { container } = render(Trace, { props: { trace } })
    const text = container.textContent || ""
    // Role labels use CSS uppercase, so the DOM has display-case strings.
    expect(text).toContain("System")
    expect(text).toContain("User")
    expect(text).toContain("Assistant")
    expect(text).toContain("Tool")
  })

  it("starts with all blocks collapsed (no expanded content rendered)", () => {
    const trace: TraceType = [
      userMsg("u1"),
      assistantMsg("a1", { reasoning_content: "thinking" }),
    ]
    const { container } = render(Trace, { props: { trace } })
    // Reasoning content header should not appear until expanded.
    expect(container.textContent).not.toContain("Reasoning")
  })

  it("renders content with raw Output (pre) by default for non-tool messages", async () => {
    const trace: TraceType = [assistantMsg("**bold markdown**")]
    const { container } = render(Trace, { props: { trace } })
    await expandAll(container)
    // Output renders into a <pre> with the raw text, NOT a <strong>.
    const pre = container.querySelector("pre")
    expect(pre).not.toBeNull()
    expect(pre?.textContent).toContain("**bold markdown**")
    expect(container.querySelector("strong")).toBeNull()
  })

  it("renders reasoning content as raw Output by default (no markdown)", async () => {
    const trace: TraceType = [
      assistantMsg("done", { reasoning_content: "**deep thought**" }),
    ]
    const { container } = render(Trace, { props: { trace } })
    await expandAll(container)
    expect(container.textContent).toContain("Reasoning")
    expect(container.querySelector("strong")).toBeNull()
  })

  it("renders an empty trace without errors", () => {
    const { container } = render(Trace, { props: { trace: [] as TraceType } })
    const collapses = container.querySelectorAll(".collapse")
    expect(collapses.length).toBe(0)
  })
})

describe("Trace component — auto_expand_indices", () => {
  const trace: TraceType = [
    systemMsg("you are helpful"),
    userMsg("find the window"),
    assistantMsg("Our return window is 30 days.", {
      reasoning_content: "checking the docs",
    }),
    toolMsg('{"output": "30 days"}'),
  ]

  function expandedRoles(container: HTMLElement): string[] {
    return [...container.querySelectorAll(".collapse")]
      .filter(
        (block) =>
          block.querySelector<HTMLInputElement>("input[type=checkbox]")
            ?.checked,
      )
      .map(
        (block) =>
          block.querySelector(".collapse-title span")?.textContent ?? "",
      )
  }

  it("leaves every block collapsed when the prop is not passed", () => {
    // The default is what every existing caller gets (the run page, the error
    // trace view), so it has to stay the all-collapsed list they render today.
    const { container } = render(Trace, { props: { trace } })
    expect(expandedRoles(container)).toEqual([])
    expect(container.textContent).not.toContain("Reasoning")
  })

  it("expands exactly the listed index and nothing else", () => {
    const { container } = render(Trace, {
      props: { trace, auto_expand_indices: [2] },
    })
    expect(expandedRoles(container)).toEqual(["Assistant"])
    // Expanded means the content is really rendered, not just checked.
    expect(container.textContent).toContain("Our return window is 30 days.")
    expect(container.textContent).toContain("Reasoning")
    // The unlisted rows are still one click away, not open.
    expect(container.textContent).not.toContain("Tool Result")
  })

  it("picks one message of a role without opening its siblings", () => {
    // The rule the review surface needs: two assistant messages, only the
    // final one open. A role-keyed prop could not express this.
    const two_answers: TraceType = [
      assistantMsg("first pass"),
      toolMsg('{"output": "30 days"}'),
      assistantMsg("Our return window is 30 days."),
    ]
    const { container } = render(Trace, {
      props: { trace: two_answers, auto_expand_indices: [2] },
    })
    const open = [...container.querySelectorAll(".collapse")].map(
      (block) =>
        !!block.querySelector<HTMLInputElement>("input[type=checkbox]")
          ?.checked,
    )
    expect(open).toEqual([false, false, true])
    expect(container.querySelector("pre")?.textContent).toContain(
      "Our return window is 30 days.",
    )
  })

  it("expands every listed index", () => {
    const { container } = render(Trace, {
      props: { trace, auto_expand_indices: [2, 3] },
    })
    expect(expandedRoles(container)).toEqual(["Assistant", "Tool"])
    expect(container.textContent).toContain("Tool Result")
  })

  it("ignores an index that is not in the trace", () => {
    const { container } = render(Trace, {
      props: { trace, auto_expand_indices: [42] },
    })
    expect(expandedRoles(container)).toEqual([])
  })

  it("does not rebuild when the parent reassigns an equal trace", async () => {
    // The expansion map is built ONCE. Parents reassign a structurally
    // identical trace on unrelated state changes (the run page does on save),
    // and rebuilding there would close every row the reader had opened.
    const { container, component } = render(Trace, {
      props: { trace, auto_expand_indices: [2] },
    })
    const tool_block = [...container.querySelectorAll(".collapse")].find(
      (block) =>
        block.querySelector(".collapse-title span")?.textContent === "Tool",
    )!
    await fireEvent.click(
      tool_block.querySelector<HTMLInputElement>("input[type=checkbox]")!,
    )
    expect(expandedRoles(container)).toEqual(["Assistant", "Tool"])

    // A new array holding the same messages, and a different index list.
    await component.$set({
      trace: [...trace] as TraceType,
      auto_expand_indices: [0],
    })
    expect(expandedRoles(container)).toEqual(["Assistant", "Tool"])
  })

  it("keeps the reader's clicks while the trace stays the same", async () => {
    // The rebuild is keyed on the trace, so an unrelated prop change must not
    // slam a block the reader just opened back shut.
    const { container, component } = render(Trace, {
      props: { trace, auto_expand_indices: [2] },
    })
    const tool_block = [...container.querySelectorAll(".collapse")].find(
      (block) =>
        block.querySelector(".collapse-title span")?.textContent === "Tool",
    )!
    await fireEvent.click(
      tool_block.querySelector<HTMLInputElement>("input[type=checkbox]")!,
    )
    expect(expandedRoles(container)).toEqual(["Assistant", "Tool"])

    await component.$set({ project_id: "proj_1" })
    expect(expandedRoles(container)).toEqual(["Assistant", "Tool"])
  })

  it("lets the reader collapse a block that started expanded", async () => {
    const { container } = render(Trace, {
      props: { trace, auto_expand_indices: [2] },
    })
    const assistant_block = [...container.querySelectorAll(".collapse")].find(
      (block) =>
        block.querySelector(".collapse-title span")?.textContent ===
        "Assistant",
    )!
    await fireEvent.click(
      assistant_block.querySelector<HTMLInputElement>("input[type=checkbox]")!,
    )
    expect(expandedRoles(container)).toEqual([])
  })
})
