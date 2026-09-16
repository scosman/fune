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
import { render, cleanup, fireEvent, waitFor } from "@testing-library/svelte"
import MultiturnComposer from "./multiturn_composer.svelte"

// Mock the api client at the source. send_multiturn imports it from
// $lib/api_client; the composer's submit path runs send_multiturn which
// POSTs through this client. Tests don't trigger Send so we just provide
// a safe no-op mock.
vi.mock("$lib/api_client", () => ({
  client: {
    POST: vi.fn(async () => ({ data: { id: "new-id" }, error: null })),
  },
  base_url: "http://test",
}))

// jsdom does not implement <dialog>.showModal()/close(); the composer's
// discard-confirmation Dialog uses them. Polyfill minimally so the test
// can observe open/close state. We track which methods we installed so we
// can tear them down in afterAll and not leak globals to other suites.
let installed_show_modal = false
let installed_close = false
beforeAll(() => {
  const proto = HTMLDialogElement.prototype as unknown as Record<
    string,
    unknown
  >
  if (typeof proto.showModal !== "function") {
    proto.showModal = function () {
      ;(this as unknown as { open: boolean }).open = true
    }
    installed_show_modal = true
  }
  if (typeof proto.close !== "function") {
    proto.close = function () {
      ;(this as unknown as { open: boolean }).open = false
    }
    installed_close = true
  }
})

afterAll(() => {
  const proto = HTMLDialogElement.prototype as unknown as Record<
    string,
    unknown
  >
  if (installed_show_modal) {
    delete proto.showModal
    installed_show_modal = false
  }
  if (installed_close) {
    delete proto.close
    installed_close = false
  }
})

afterEach(() => cleanup())

const base_props = {
  project_id: "p1",
  task_id: "t1",
  parent_task_run_id: "leaf-42",
  run_config_component: null,
  on_success: vi.fn(),
}

describe("MultiturnComposer", () => {
  it("append mode renders no Cancel button and no fork context strip", () => {
    const { container, queryByTestId } = render(MultiturnComposer, {
      props: { ...base_props, mode: "append" },
    })
    expect(queryByTestId("multiturn-composer-cancel")).toBeNull()
    expect(queryByTestId("multiturn-fork-context-strip")).toBeNull()
    // Sanity: still renders the input form area and a Send button.
    expect(queryByTestId("multiturn-composer-input")).not.toBeNull()
    expect(container.querySelector("button[type=submit]")).not.toBeNull()
  })

  it("fork mode renders the context strip with the forked turn number", () => {
    const { getByTestId } = render(MultiturnComposer, {
      props: {
        ...base_props,
        mode: "fork",
        forked_turn_index: 3,
        on_cancel: vi.fn(),
      },
    })
    const strip = getByTestId("multiturn-fork-context-strip")
    expect(strip.textContent || "").toContain("Forking turn 3")
  })

  it("fork mode prefills the textarea with prefill_text", async () => {
    const { container } = render(MultiturnComposer, {
      props: {
        ...base_props,
        mode: "fork",
        forked_turn_index: 2,
        prefill_text: "original turn text",
        on_cancel: vi.fn(),
      },
    })
    await waitFor(() => {
      const textarea = container.querySelector<HTMLTextAreaElement>("textarea")
      expect(textarea?.value).toBe("original turn text")
    })
  })

  it("Cancel with unchanged input calls on_cancel without a discard dialog", async () => {
    const on_cancel = vi.fn()
    const { getByTestId, container } = render(MultiturnComposer, {
      props: {
        ...base_props,
        mode: "fork",
        forked_turn_index: 2,
        prefill_text: "same text",
        on_cancel,
      },
    })
    // Wait for prefill to complete.
    await waitFor(() => {
      const textarea = container.querySelector<HTMLTextAreaElement>("textarea")
      expect(textarea?.value).toBe("same text")
    })
    const cancel = getByTestId("multiturn-composer-cancel")
    await fireEvent.click(cancel)
    expect(on_cancel).toHaveBeenCalledTimes(1)
    // No open dialog.
    const dialog = container.querySelector<HTMLDialogElement>("dialog.modal")
    expect(dialog?.open).not.toBe(true)
  })

  it("Cancel with dirty input opens a discard dialog and only fires on_cancel after confirm", async () => {
    const on_cancel = vi.fn()
    const { getByTestId, container } = render(MultiturnComposer, {
      props: {
        ...base_props,
        mode: "fork",
        forked_turn_index: 2,
        prefill_text: "original",
        on_cancel,
      },
    })
    await waitFor(() => {
      const textarea = container.querySelector<HTMLTextAreaElement>("textarea")
      expect(textarea?.value).toBe("original")
    })
    // Edit textarea to make it dirty.
    const textarea = container.querySelector<HTMLTextAreaElement>("textarea")!
    await fireEvent.input(textarea, { target: { value: "edited" } })

    const cancel = getByTestId("multiturn-composer-cancel")
    await fireEvent.click(cancel)
    expect(on_cancel).not.toHaveBeenCalled()

    // Dialog should be open. Click the Discard button.
    const dialog = container.querySelector<HTMLDialogElement>("dialog.modal")
    expect(dialog).not.toBeNull()
    const discard = Array.from(
      dialog!.querySelectorAll<HTMLButtonElement>("button"),
    ).find((b) => (b.textContent || "").trim() === "Discard")
    expect(discard).toBeDefined()
    await fireEvent.click(discard!)
    expect(on_cancel).toHaveBeenCalledTimes(1)
  })

  it("request_swap with dirty input opens the discard dialog and fires on_proceed only on Discard", async () => {
    // Render the composer and dirty the input. We then call request_swap
    // through the exported instance method (Svelte 4 attaches script
    // `export function` declarations to the component instance).
    const on_cancel = vi.fn()
    const on_proceed = vi.fn()
    const { container, component } = render(MultiturnComposer, {
      props: {
        ...base_props,
        mode: "fork",
        forked_turn_index: 2,
        prefill_text: "original",
        on_cancel,
      },
    })
    await waitFor(() => {
      const textarea = container.querySelector<HTMLTextAreaElement>("textarea")
      expect(textarea?.value).toBe("original")
    })
    const textarea = container.querySelector<HTMLTextAreaElement>("textarea")!
    await fireEvent.input(textarea, { target: { value: "edited" } })

    // The composer exports request_swap; testing-library returns the
    // component instance via `component`.
    const instance = component as unknown as {
      request_swap: (cb: () => void) => void
      is_dirty: () => boolean
    }
    expect(instance.is_dirty()).toBe(true)
    instance.request_swap(on_proceed)

    // Dialog should be open; on_proceed not yet called.
    const dialog = container.querySelector<HTMLDialogElement>("dialog.modal")
    expect(dialog?.open).toBe(true)
    expect(on_proceed).not.toHaveBeenCalled()

    // Click Discard — the swap callback should fire, NOT on_cancel.
    const discard = Array.from(
      dialog!.querySelectorAll<HTMLButtonElement>("button"),
    ).find((b) => (b.textContent || "").trim() === "Discard")
    await fireEvent.click(discard!)
    expect(on_proceed).toHaveBeenCalledTimes(1)
    expect(on_cancel).not.toHaveBeenCalled()
  })

  it("request_swap with clean input fires on_proceed without opening the dialog", async () => {
    const on_cancel = vi.fn()
    const on_proceed = vi.fn()
    const { container, component } = render(MultiturnComposer, {
      props: {
        ...base_props,
        mode: "fork",
        forked_turn_index: 2,
        prefill_text: "untouched",
        on_cancel,
      },
    })
    await waitFor(() => {
      const textarea = container.querySelector<HTMLTextAreaElement>("textarea")
      expect(textarea?.value).toBe("untouched")
    })
    const instance = component as unknown as {
      request_swap: (cb: () => void) => void
      is_dirty: () => boolean
    }
    expect(instance.is_dirty()).toBe(false)
    instance.request_swap(on_proceed)
    expect(on_proceed).toHaveBeenCalledTimes(1)
    expect(on_cancel).not.toHaveBeenCalled()
    const dialog = container.querySelector<HTMLDialogElement>("dialog.modal")
    expect(dialog?.open).not.toBe(true)
  })
})
