// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest"
import { render, cleanup, fireEvent } from "@testing-library/svelte"
import { tick } from "svelte"

vi.mock("$lib/api_client", () => ({
  client: {
    GET: vi.fn(),
    POST: vi.fn(),
  },
}))

vi.mock("posthog-js", () => ({
  default: { capture: vi.fn() },
}))

// Stub the dialog so its slot (the edit-inputs form) always renders and the
// action buttons are exposed for direct invocation.
vi.mock("$lib/ui/dialog.svelte", async () => {
  const Stub = await import("../eval_types/__tests__/dialog_stub.svelte")
  return { default: Stub.default }
})

import CodeToolTestPanel from "./code_tool_test_panel.svelte"
import { client } from "$lib/api_client"
import {
  actionButtonsByTitle,
  resetActionButtons,
} from "../eval_types/__tests__/dialog_stub.svelte"

const parameters_schema = {
  type: "object",
  properties: {
    name: { type: "string", title: "Name" },
    note: { type: "string", title: "Note" },
  },
  required: ["name"],
}

const baseProps = {
  project_id: "p1",
  tool_function_name: "my_tool",
  tool_description: "desc",
  parameters_schema,
  code: "print(1)",
  timeout_seconds: 5,
  tool_allowlist: [] as string[],
}

function doneAction(): () => boolean {
  const buttons = actionButtonsByTitle["Edit Test Input"]
  const done = buttons?.find((b) => b.label === "Done")
  return done?.action as () => boolean
}

afterEach(() => {
  cleanup()
  resetActionButtons()
  vi.restoreAllMocks()
})

beforeEach(() => {
  vi.mocked(client.POST).mockReset()
})

describe("CodeToolTestPanel edit-inputs validation", () => {
  it("keeps the dialog open and shows an error when a required field is missing, preserving typed values", async () => {
    const { container } = render(CodeToolTestPanel, { props: baseProps })
    await tick()

    // Type into the optional field; leave the required "name" blank.
    const textareas = container.querySelectorAll("textarea")
    expect(textareas.length).toBe(2)
    await fireEvent.input(textareas[1], { target: { value: "keep me" } })
    await tick()

    // Click "Done" -> build fails on the missing required field.
    const shouldClose = doneAction()()
    await tick()

    // Dialog stays open (action returned false) and the error is surfaced.
    expect(shouldClose).toBe(false)
    expect(container.textContent).toContain("Required property not set")

    // The typed optional value must not be discarded.
    const textareasAfter = container.querySelectorAll("textarea")
    expect((textareasAfter[1] as HTMLTextAreaElement).value).toBe("keep me")
  })

  it("saves and closes when all required fields are provided", async () => {
    const { container } = render(CodeToolTestPanel, { props: baseProps })
    await tick()

    const textareas = container.querySelectorAll("textarea")
    await fireEvent.input(textareas[0], { target: { value: "Alice" } })
    await fireEvent.input(textareas[1], { target: { value: "hi" } })
    await tick()

    const shouldClose = doneAction()()
    await tick()

    expect(shouldClose).toBe(true)
    expect(container.textContent).not.toContain("Required property not set")

    // The built values feed the input preview shown on the panel.
    expect(container.textContent).toContain("Alice")
  })
})
