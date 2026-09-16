// @vitest-environment jsdom
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest"
import { render, cleanup } from "@testing-library/svelte"
import { writable } from "svelte/store"
import { current_task } from "$lib/stores"
import UploadDatasetDialog from "./upload_dataset_dialog.svelte"

// upload_dataset_dialog reads page params from $app/stores. In a vitest
// environment there's no live SvelteKit request, so stub the page store with
// a minimal writable producing the params shape the component expects.
vi.mock("$app/stores", () => ({
  page: writable({ params: { project_id: "p1", task_id: "t1" } }),
}))

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyTask = any

const NOOP = () => {}

describe("UploadDatasetDialog", () => {
  beforeEach(() => {
    current_task.set(null)
  })

  afterEach(() => {
    cleanup()
    current_task.set(null)
  })

  it("renders single-turn help when turn_mode is single_turn", () => {
    const { container } = render(UploadDatasetDialog, {
      props: { onImportCompleted: NOOP, turn_mode: "single_turn" },
    })

    const titleNode = container.querySelector("h3")
    expect(titleNode?.textContent?.trim()).toBe("Add CSV to Dataset")

    const codeNodes = Array.from(container.querySelectorAll("code")).map(
      (n) => n.textContent,
    )
    expect(codeNodes).toContain("input")
    expect(codeNodes).toContain("output")
    expect(codeNodes).not.toContain("trace")

    const downloadLink = container.querySelector(
      'a[href="/sample_multiturn.csv"]',
    )
    expect(downloadLink).toBeNull()
  })

  it("renders multiturn help when turn_mode is multiturn", () => {
    const { container } = render(UploadDatasetDialog, {
      props: { onImportCompleted: NOOP, turn_mode: "multiturn" },
    })

    const titleNode = container.querySelector("h3")
    expect(titleNode?.textContent?.trim()).toBe("Add Multiturn CSV to Dataset")

    const codeNodes = Array.from(container.querySelectorAll("code")).map(
      (n) => n.textContent,
    )
    expect(codeNodes).toContain("trace")
    expect(codeNodes).toContain("tags")
    expect(codeNodes).not.toContain("input")
    expect(codeNodes).not.toContain("output")

    expect(container.textContent).toContain("System messages are not supported")

    const downloadLink = container.querySelector(
      'a[href="/sample_multiturn.csv"]',
    ) as HTMLAnchorElement | null
    expect(downloadLink).not.toBeNull()
    expect(downloadLink?.hasAttribute("download")).toBe(true)
    expect(downloadLink?.textContent).toContain("Download sample CSV")
  })

  it("uses the route's turn_mode prop, ignoring a mismatched current_task store", () => {
    // The global store holds a different task than the route: the docs must
    // follow the prop (the route's task), not the stale store.
    current_task.set({ turn_mode: "single_turn" } as AnyTask)
    const { container } = render(UploadDatasetDialog, {
      props: { onImportCompleted: NOOP, turn_mode: "multiturn" },
    })

    const titleNode = container.querySelector("h3")
    expect(titleNode?.textContent?.trim()).toBe("Add Multiturn CSV to Dataset")

    const codeNodes = Array.from(container.querySelectorAll("code")).map(
      (n) => n.textContent,
    )
    expect(codeNodes).toContain("trace")
    expect(codeNodes).not.toContain("input")
  })

  it("falls back to single-turn help when turn_mode is null", () => {
    const { container } = render(UploadDatasetDialog, {
      props: { onImportCompleted: NOOP, turn_mode: null },
    })

    const titleNode = container.querySelector("h3")
    expect(titleNode?.textContent?.trim()).toBe("Add CSV to Dataset")

    const codeNodes = Array.from(container.querySelectorAll("code")).map(
      (n) => n.textContent,
    )
    expect(codeNodes).toContain("input")
    expect(codeNodes).toContain("output")
  })
})
