// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { render, fireEvent, cleanup, waitFor } from "@testing-library/svelte"
import { writable } from "svelte/store"
import { tick } from "svelte"

vi.mock("$app/navigation", () => ({
  goto: vi.fn(),
  beforeNavigate: vi.fn(),
}))

vi.mock("posthog-js", () => ({
  default: { capture: vi.fn() },
}))

const postMock = vi.fn()
const patchMock = vi.fn()
vi.mock("$lib/api_client", () => ({
  client: {
    POST: (...args: unknown[]) => postMock(...args),
    PATCH: (...args: unknown[]) => patchMock(...args),
  },
  base_url: "http://test:8000",
}))

const current_project = writable<{ id: string; name: string } | null>({
  id: "proj-1",
  name: "Test Project",
})
const projects = writable<{
  projects: Array<{ id: string; name: string }>
  error: string | null
} | null>({
  projects: [{ id: "proj-1", name: "Test Project" }],
  error: null,
})
const ui_state = writable({
  current_project_id: "proj-1",
  current_task_id: null,
  selected_model: null,
})
vi.mock("$lib/stores", () => ({
  current_project,
  projects,
  ui_state,
  load_current_task: vi.fn(),
  load_rating_options: vi.fn(),
  get_task_composite_id: (project_id: string, task_id: string) =>
    `${project_id}__${task_id}`,
}))

const EditTask = (await import("./edit_task.svelte")).default

beforeEach(() => {
  postMock.mockReset()
  patchMock.mockReset()
  postMock.mockResolvedValue({ data: { id: "task-new" }, error: null })
  patchMock.mockResolvedValue({ data: { id: "task-new" }, error: null })
})

afterEach(() => {
  cleanup()
})

function findButtonByText(
  container: HTMLElement,
  label: string,
): HTMLButtonElement | null {
  const buttons = Array.from(container.querySelectorAll("button"))
  return (
    (buttons.find((b) => b.textContent?.trim() === label) as
      | HTMLButtonElement
      | undefined) ?? null
  )
}

async function fillRequiredFields(container: HTMLElement) {
  const nameInput = container.querySelector(
    "#task_name",
  ) as HTMLInputElement | null
  const instructionInput = container.querySelector(
    "#task_instructions",
  ) as HTMLTextAreaElement | null
  expect(nameInput).not.toBeNull()
  expect(instructionInput).not.toBeNull()
  await fireEvent.input(nameInput!, { target: { value: "My Task" } })
  await fireEvent.input(instructionInput!, {
    target: { value: "Do the thing." },
  })
  await tick()
}

async function submitForm(container: HTMLElement) {
  const submitBtn = container.querySelector(
    'button[type="submit"]',
  ) as HTMLButtonElement | null
  expect(submitBtn).not.toBeNull()
  await fireEvent.click(submitBtn!)
  await waitFor(() => expect(postMock).toHaveBeenCalled())
}

describe("EditTask — turn_mode selector", () => {
  it("defaults to single_turn and shows the input + output schema sections", () => {
    const { container, queryByTestId } = render(EditTask, {
      props: { redirect_on_created: null },
    })

    const single = queryByTestId("turn-mode-single-turn") as HTMLInputElement
    const multi = queryByTestId("turn-mode-multiturn") as HTMLInputElement
    expect(single).not.toBeNull()
    expect(multi).not.toBeNull()
    expect(single.type).toBe("radio")
    expect(multi.type).toBe("radio")
    expect(single.checked).toBe(true)
    expect(multi.checked).toBe(false)

    expect(container.textContent).toContain("Part 3: Input Schema")
    expect(container.textContent).toContain("Part 4: Output Schema")

    // Both schema editors are actually mounted (not just their headings): the
    // input and output SchemaSections each render a Plain Text / Structured
    // JSON format toggle.
    const structured_labels = Array.from(
      container.querySelectorAll(".label-text"),
    ).filter((el) => el.textContent?.trim() === "Structured JSON")
    expect(structured_labels.length).toBe(2)
  })

  it("hides input + output schema sections when multiturn is selected", async () => {
    const { container, queryByTestId, getByTestId } = render(EditTask, {
      props: { redirect_on_created: null },
    })

    const multi = getByTestId("turn-mode-multiturn") as HTMLInputElement
    await fireEvent.click(multi)
    await tick()

    expect(container.textContent).not.toContain("Part 3: Input Schema")
    expect(container.textContent).not.toContain("Part 4: Output Schema")

    expect(multi.checked).toBe(true)
    expect(
      (queryByTestId("turn-mode-single-turn") as HTMLInputElement).checked,
    ).toBe(false)
  })

  it("re-shows the schema sections when switching back to single_turn", async () => {
    const { container, getByTestId } = render(EditTask, {
      props: { redirect_on_created: null },
    })

    await fireEvent.click(getByTestId("turn-mode-multiturn"))
    await tick()
    expect(container.textContent).not.toContain("Part 3: Input Schema")

    await fireEvent.click(getByTestId("turn-mode-single-turn"))
    await tick()
    expect(container.textContent).toContain("Part 3: Input Schema")
    expect(container.textContent).toContain("Part 4: Output Schema")
  })

  it("renders read-only label and no toggle when read_only_turn_mode is true", () => {
    const task = {
      id: "task-1",
      name: "Existing",
      description: "",
      instruction: "Hi",
      requirements: [],
      turn_mode: "multiturn" as const,
    }
    const { queryByTestId, getByTestId } = render(EditTask, {
      props: {
        // @ts-expect-error partial Task is fine for the form
        task,
        read_only_turn_mode: true,
        redirect_on_created: null,
      },
    })

    const readonly = getByTestId("turn-mode-readonly")
    expect(readonly.textContent).toContain("Task type:")
    expect(readonly.textContent).toContain("Multi-turn")
    expect(readonly.textContent).toContain(
      "This setting can't be changed after the task is created.",
    )

    expect(queryByTestId("turn-mode-single-turn")).toBeNull()
    expect(queryByTestId("turn-mode-multiturn")).toBeNull()
  })

  it("renders single-turn read-only when task is single-turn", () => {
    const task = {
      id: "task-1",
      name: "Existing",
      description: "",
      instruction: "Hi",
      requirements: [],
      turn_mode: "single_turn" as const,
    }
    const { getByTestId } = render(EditTask, {
      props: {
        // @ts-expect-error partial Task is fine for the form
        task,
        read_only_turn_mode: true,
        redirect_on_created: null,
      },
    })
    const readonly = getByTestId("turn-mode-readonly")
    expect(readonly.textContent).toContain("Single-turn")
  })
})

describe("EditTask — create payload", () => {
  it("posts turn_mode: single_turn by default", async () => {
    const { container } = render(EditTask, {
      props: { redirect_on_created: null },
    })
    await fillRequiredFields(container)
    await submitForm(container)

    const call = postMock.mock.calls[0]
    expect(call[0]).toBe("/api/projects/{project_id}/tasks")
    const body = call[1].body as Record<string, unknown>
    expect(body.turn_mode).toBe("single_turn")
    expect(body.name).toBe("My Task")
    expect(body.instruction).toBe("Do the thing.")
  })

  it("posts turn_mode: multiturn without schema fields when multiturn is selected", async () => {
    const { container, getByTestId } = render(EditTask, {
      props: { redirect_on_created: null },
    })
    await fireEvent.click(getByTestId("turn-mode-multiturn"))
    await tick()

    await fillRequiredFields(container)
    await submitForm(container)

    const call = postMock.mock.calls[0]
    const body = call[1].body as Record<string, unknown>
    expect(body.turn_mode).toBe("multiturn")
    expect(body.input_json_schema ?? null).toBeNull()
    expect(body.output_json_schema ?? null).toBeNull()
  })
})

describe("EditTask — Try an example adapts to turn mode", () => {
  it("clicking 'Try an example' in single-turn loads the structured joke example", async () => {
    const { container, getByTestId } = render(EditTask, {
      props: { redirect_on_created: null },
    })

    const exampleBtn = findButtonByText(
      container as HTMLElement,
      "Try an example.",
    )
    expect(exampleBtn).not.toBeNull()
    await fireEvent.click(exampleBtn!)
    await tick()

    const nameInput = container.querySelector("#task_name") as HTMLInputElement
    expect(nameInput.value).toBe("Joke Generator")
    expect(
      (getByTestId("turn-mode-single-turn") as HTMLInputElement).checked,
    ).toBe(true)
  })

  it("clicking 'Try an example' in multiturn loads a generic chat example and stays multiturn", async () => {
    const { container, getByTestId } = render(EditTask, {
      props: { redirect_on_created: null },
    })

    await fireEvent.click(getByTestId("turn-mode-multiturn"))
    await tick()

    const exampleBtn = findButtonByText(
      container as HTMLElement,
      "Try an example.",
    )
    expect(exampleBtn).not.toBeNull()
    await fireEvent.click(exampleBtn!)
    await tick()

    const nameInput = container.querySelector("#task_name") as HTMLInputElement
    expect(nameInput.value).toBe("Chat Assistant")
    const instructionInput = container.querySelector(
      "#task_instructions",
    ) as HTMLTextAreaElement
    expect(instructionInput.value.toLowerCase()).toContain("helpful assistant")
    expect(
      (getByTestId("turn-mode-multiturn") as HTMLInputElement).checked,
    ).toBe(true)
  })
})
