// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest"
import { render, cleanup, fireEvent, waitFor } from "@testing-library/svelte"
import { tick } from "svelte"
import { get } from "svelte/store"

vi.mock("$lib/api_client", () => ({
  client: {
    GET: vi.fn(),
    POST: vi.fn(),
  },
}))

vi.mock("$lib/stores", () => ({
  load_projects: vi.fn().mockResolvedValue(undefined),
}))

vi.mock("$app/navigation", () => ({
  replaceState: vi.fn(),
  beforeNavigate: vi.fn(),
}))

vi.mock("posthog-js", () => ({
  default: { capture: vi.fn() },
}))

vi.mock("$lib/git_sync/url_utils", () => ({
  sync_url_query_param: vi.fn(),
  read_url_query_param: vi.fn().mockReturnValue(null),
}))

vi.mock("$lib/stores/git_import_wizard_store", () => {
  const { writable } = require("svelte/store")
  return {
    git_import_wizard_store: writable({
      git_url: "",
      pat_token: null,
      oauth_token: null,
      auth_mode: "system_keys",
      clone_path: "",
      selected_branch: "",
      selected_project_path: "",
      selected_project_id: "",
      selected_project_name: "",
    }),
    clear_wizard_store: vi.fn(),
    validate_step_requirements: vi.fn().mockReturnValue(true),
  }
})

// Stub the git step components so the trust-binding flow can be driven via the
// captured callbacks, with no real network or OAuth.
vi.mock("./step_url.svelte", async () => {
  const Stub = await import("./__tests__/step_url_stub.svelte")
  return { default: Stub.default }
})
vi.mock("./step_credentials.svelte", async () => {
  const Stub = await import("./__tests__/step_credentials_stub.svelte")
  return { default: Stub.default }
})
vi.mock("./step_branch.svelte", async () => {
  const Stub = await import("./__tests__/step_branch_stub.svelte")
  return { default: Stub.default }
})

import ImportProject from "./import_project.svelte"
import { client } from "$lib/api_client"
import { git_import_wizard_store } from "$lib/stores/git_import_wizard_store"
import { stepUrlProps } from "./__tests__/step_url_stub.svelte"
import { stepCredentialsProps } from "./__tests__/step_credentials_stub.svelte"

type WizardState = Parameters<typeof git_import_wizard_store.set>[0]

function setWizardState(overrides: Partial<WizardState>) {
  git_import_wizard_store.set({
    git_url: "",
    pat_token: null,
    oauth_token: null,
    auth_mode: "system_keys",
    clone_path: "",
    selected_branch: "",
    selected_project_path: "",
    selected_project_id: "",
    selected_project_name: "",
    ...overrides,
  })
}

const baseProps = {
  create_link: "/create",
  on_complete: vi.fn(),
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  window.location.hash = ""
})

beforeEach(() => {
  vi.mocked(client.POST).mockReset()
  vi.mocked(client.GET).mockReset()
  vi.mocked(baseProps.on_complete).mockReset()
  window.location.hash = ""
})

describe("ImportProject local_file conflict handling", () => {
  async function renderAtLocalStep() {
    const result = render(ImportProject, { props: baseProps })
    await tick()

    // Click "Import from Local Folder" button to navigate to local_file step
    const localBtn = result.getByText("Import from Local Folder")
    await fireEvent.click(localBtn)
    await tick()

    // Now trigger the file selector error to reveal the manual path input
    vi.mocked(client.GET).mockRejectedValue(new Error("No file selector"))
    const selectBtn = result.container.querySelector(
      "button.btn-primary",
    ) as HTMLButtonElement
    if (selectBtn?.textContent?.includes("Select Project File")) {
      await fireEvent.click(selectBtn)
      await tick()
      await new Promise((r) => setTimeout(r, 0))
      await tick()
    }

    return result
  }

  // After form submit, the UI now navigates to a trust confirmation page.
  // This helper clicks "Trust Project" to proceed to the actual import.
  async function confirmTrustPage(container: HTMLElement) {
    await waitFor(() => {
      expect(container.textContent).toContain("Trust this Project?")
    })
    const trustBtn = container.querySelector(
      "button.btn-warning",
    ) as HTMLButtonElement
    expect(trustBtn?.textContent?.trim()).toBe("Trust Project")
    await fireEvent.click(trustBtn)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()
  }

  it("shows conflict button on 409 response", async () => {
    vi.mocked(client.POST).mockResolvedValue({
      data: undefined,
      error: { message: "Duplicate project ID" },
      response: new Response(null, { status: 409 }),
    } as never)

    const { container } = await renderAtLocalStep()

    const input = container.querySelector(
      "#import_project_path",
    ) as HTMLInputElement
    expect(input).toBeTruthy()
    await fireEvent.input(input, {
      target: { value: "/path/to/project.kiln" },
    })
    await tick()

    const submitBtn = container.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement
    expect(submitBtn).toBeTruthy()
    await fireEvent.click(submitBtn)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    // Confirm the trust page to proceed to the actual import
    await confirmTrustPage(container)

    await waitFor(() => {
      expect(container.textContent).toContain("Remove existing and re-import")
    })

    // Normal submit button should be hidden in conflict state
    const hiddenSubmit = container.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement
    expect(hiddenSubmit?.classList.contains("hidden")).toBe(true)
  })

  it("conflict button is type=button so it cannot submit the form", async () => {
    vi.mocked(client.POST).mockResolvedValue({
      data: undefined,
      error: { message: "Duplicate project ID" },
      response: new Response(null, { status: 409 }),
    } as never)

    const { container, getByText } = await renderAtLocalStep()

    const input = container.querySelector(
      "#import_project_path",
    ) as HTMLInputElement
    await fireEvent.input(input, {
      target: { value: "/path/to/project.kiln" },
    })
    await tick()

    const submitBtn = container.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement
    await fireEvent.click(submitBtn)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    // Confirm the trust page to proceed to the actual import
    await confirmTrustPage(container)

    await waitFor(() => {
      expect(container.textContent).toContain("Remove existing and re-import")
    })

    // type="button" keeps the conflict button out of implicit form submission.
    // Without it the button defaults to type="submit" and, as the first submit
    // button in tree order, would become the form's default button -- so Enter
    // in the path field could fire the destructive remove-and-re-import action.
    const conflictBtn = getByText(
      "Remove existing and re-import",
    ) as HTMLButtonElement
    expect(conflictBtn.getAttribute("type")).toBe("button")
  })

  it("does not show conflict button on non-409 error", async () => {
    vi.mocked(client.POST).mockResolvedValue({
      data: undefined,
      error: { message: "Server error" },
      response: new Response(null, { status: 500 }),
    } as never)

    const { container } = await renderAtLocalStep()

    const input = container.querySelector(
      "#import_project_path",
    ) as HTMLInputElement
    expect(input).toBeTruthy()
    await fireEvent.input(input, {
      target: { value: "/path/to/project.kiln" },
    })
    await tick()

    const submitBtn = container.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement
    await fireEvent.click(submitBtn)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    // Confirm the trust page to proceed to the actual import
    await confirmTrustPage(container)

    await waitFor(() => {
      expect(container.textContent).toContain("Server error")
    })
    expect(container.textContent).not.toContain("Remove existing and re-import")
  })

  it("clicking conflict button retries with remove_conflicting_id=true", async () => {
    // First call: 409 conflict
    vi.mocked(client.POST).mockResolvedValueOnce({
      data: undefined,
      error: { message: "Duplicate project ID" },
      response: new Response(null, { status: 409 }),
    } as never)

    const { container, getByText } = await renderAtLocalStep()

    const input = container.querySelector(
      "#import_project_path",
    ) as HTMLInputElement
    expect(input).toBeTruthy()
    await fireEvent.input(input, {
      target: { value: "/path/to/project.kiln" },
    })
    await tick()

    const submitBtn = container.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement
    await fireEvent.click(submitBtn)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    // Confirm the trust page to proceed to the actual import
    await confirmTrustPage(container)

    await waitFor(() => {
      expect(container.textContent).toContain("Remove existing and re-import")
    })

    // Second call: success
    vi.mocked(client.POST).mockResolvedValueOnce({
      data: { id: "proj_1", name: "Imported" },
      error: undefined,
      response: new Response(null, { status: 200 }),
    } as never)

    const conflictBtn = getByText("Remove existing and re-import")
    await fireEvent.click(conflictBtn)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    // Verify the second call included remove_conflicting_id
    const calls = vi.mocked(client.POST).mock.calls
    expect(calls.length).toBe(2)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const secondCallOpts = calls[1][1] as any
    expect(secondCallOpts?.params?.query?.remove_conflicting_id).toBe(true)
  })

  it("editing path after 409 clears conflict state and restores submit button", async () => {
    vi.mocked(client.POST).mockResolvedValue({
      data: undefined,
      error: { message: "Duplicate project ID" },
      response: new Response(null, { status: 409 }),
    } as never)

    const { container } = await renderAtLocalStep()

    const input = container.querySelector(
      "#import_project_path",
    ) as HTMLInputElement
    expect(input).toBeTruthy()
    await fireEvent.input(input, {
      target: { value: "/path/to/project.kiln" },
    })
    await tick()

    const submitBtn = container.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement
    await fireEvent.click(submitBtn)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    // Confirm the trust page to proceed to the actual import
    await confirmTrustPage(container)

    // Confirm conflict state is active
    await waitFor(() => {
      expect(container.textContent).toContain("Remove existing and re-import")
    })
    // Re-query submit button after step navigation
    const submitBtnAfterConflict = container.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement
    expect(submitBtnAfterConflict?.classList.contains("hidden")).toBe(true)

    // Now edit the path via typing - should clear conflict and restore normal submit
    const inputAfterConflict = container.querySelector(
      "#import_project_path",
    ) as HTMLInputElement
    await fireEvent.input(inputAfterConflict, {
      target: { value: "/different/path/project.kiln" },
    })
    await tick()

    await waitFor(() => {
      expect(container.textContent).not.toContain(
        "Remove existing and re-import",
      )
    })

    // Normal submit button should be visible again
    const restoredSubmit = container.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement
    expect(restoredSubmit?.classList.contains("hidden")).toBe(false)
  })

  it("file picker programmatic path change clears conflict via reactive block", async () => {
    // Validates the reactive clearing mechanism via the real file-picker path.
    // select_project_file() sets import_project_path programmatically (no DOM
    // input event), which the old on:input wrapper div would have missed.
    const result = render(ImportProject, { props: baseProps })
    await tick()

    const localBtn = result.getByText("Import from Local Folder")
    await fireEvent.click(localBtn)
    await tick()

    const { container } = result

    // Use the file picker (not manual input) to set the initial path
    vi.mocked(client.GET).mockResolvedValueOnce({
      data: { file_path: "/path/to/project.kiln" },
      error: undefined,
      response: new Response(null, { status: 200 }),
    } as never)

    const selectBtn = container.querySelector(
      "button.btn-primary",
    ) as HTMLButtonElement
    expect(selectBtn?.textContent).toContain("Select Project File")
    await fireEvent.click(selectBtn)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    // Path was set programmatically by the file picker; input field should show
    const input = container.querySelector(
      "#import_project_path",
    ) as HTMLInputElement
    expect(input).toBeTruthy()
    expect(input.value).toBe("/path/to/project.kiln")

    // Submit -> trust page -> 409 conflict
    vi.mocked(client.POST).mockResolvedValueOnce({
      data: undefined,
      error: { message: "Duplicate project ID" },
      response: new Response(null, { status: 409 }),
    } as never)

    const submitBtn = container.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement
    await fireEvent.click(submitBtn)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    // Confirm the trust page to proceed to the actual import
    await confirmTrustPage(container)

    // Conflict button appears (reactive did NOT self-wipe on the 409)
    await waitFor(() => {
      expect(container.textContent).toContain("Remove existing and re-import")
    })
    const submitBtnAfterConflict = container.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement
    expect(submitBtnAfterConflict?.classList.contains("hidden")).toBe(true)

    // Clear the path to make the file picker button reappear.
    // This also clears the conflict (expected: path changed from the conflict path).
    const inputAfterConflict = container.querySelector(
      "#import_project_path",
    ) as HTMLInputElement
    await fireEvent.input(inputAfterConflict, { target: { value: "" } })
    await tick()

    await waitFor(() => {
      expect(container.textContent).not.toContain(
        "Remove existing and re-import",
      )
    })

    // File picker button is visible again (select_file_unavailable is still false)
    const selectBtn2 = container.querySelector(
      "button.btn-primary",
    ) as HTMLButtonElement
    expect(selectBtn2?.textContent).toContain("Select Project File")

    // Use the file picker to set a DIFFERENT path programmatically
    vi.mocked(client.GET).mockResolvedValueOnce({
      data: { file_path: "/different/programmatic/path.kiln" },
      error: undefined,
      response: new Response(null, { status: 200 }),
    } as never)

    await fireEvent.click(selectBtn2)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    // Verify the file picker set the new path programmatically
    const updatedInput = container.querySelector(
      "#import_project_path",
    ) as HTMLInputElement
    expect(updatedInput?.value).toBe("/different/programmatic/path.kiln")

    // Submit again -> trust page -> 409
    vi.mocked(client.POST).mockResolvedValueOnce({
      data: undefined,
      error: { message: "Duplicate project ID" },
      response: new Response(null, { status: 409 }),
    } as never)

    const submitBtn2 = container.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement
    await fireEvent.click(submitBtn2)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    // Confirm the trust page again
    await confirmTrustPage(container)

    // Conflict button appears again -- the reactive block correctly handled the
    // programmatic path set from the file picker without self-wiping on the 409
    await waitFor(() => {
      expect(container.textContent).toContain("Remove existing and re-import")
    })

    const hiddenSubmit = container.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement
    expect(hiddenSubmit?.classList.contains("hidden")).toBe(true)
  })
})

describe("ImportProject trust binding per repo", () => {
  // Reach the git url step through the UI (Svelte 4 onMount does not run under
  // jsdom/vitest, so hash-driven entry can't be tested here). The store is
  // pre-seeded to simulate a repo already carried through the wizard.
  async function renderAtUrlStep() {
    const result = render(ImportProject, { props: baseProps })
    await tick()
    await fireEvent.click(result.getByText("Git Auto Sync"))
    await tick()
    return result
  }

  it("re-entering credentials for the same repo skips trust and returns to branch", async () => {
    // Repo A has already reached the branch step (clone_path set), which is
    // only reachable after passing trust for it. A branch that needs
    // credentials bounces back here; re-verifying the same repo must not
    // re-prompt trust.
    setWizardState({
      git_url: "https://example.com/a.git",
      clone_path: "/clone/a",
      auth_mode: "pat_token",
      pat_token: "tokA",
    })

    const { container } = await renderAtUrlStep()

    // Same URL: adopt_git_url keeps the existing clone_path.
    stepUrlProps.on_auth_required?.("https://example.com/a.git")
    await tick()
    expect(get(git_import_wizard_store).clone_path).toBe("/clone/a")

    stepCredentialsProps.on_success?.("tokA", "pat_token")
    await tick()

    expect(
      container.querySelector('[data-testid="step-branch-stub"]'),
    ).not.toBe(null)
    expect(container.textContent).not.toContain("Trust this Project?")
  })

  it("entering a different repo after Back clears stale trust and lands on trust_confirm", async () => {
    // Repo A fully entered (clone_path set). User goes Back to the url step and
    // enters a different, private repo B. Because clone_path belonged to A, the
    // credentials success must NOT skip trust for B.
    setWizardState({
      git_url: "https://example.com/a.git",
      clone_path: "/clone/a",
      auth_mode: "pat_token",
      pat_token: "tokA",
    })

    const { container } = await renderAtUrlStep()

    // Enter repo B, which requires authentication.
    stepUrlProps.on_auth_required?.("https://example.com/b.git")
    await tick()

    // clone_path from A must be cleared by the URL change.
    expect(get(git_import_wizard_store).clone_path).toBe("")

    // Verify credentials for B.
    stepCredentialsProps.on_success?.("tokB", "pat_token")
    await tick()

    expect(container.textContent).toContain("Trust this Project?")
    expect(container.querySelector('[data-testid="step-branch-stub"]')).toBe(
      null,
    )
  })

  it("re-confirming the same repo url on the url step keeps its downstream state", async () => {
    // Returning to the url step and submitting the same repo must not wipe the
    // trust/clone_path already granted for it.
    setWizardState({
      git_url: "https://example.com/a.git",
      clone_path: "/clone/a",
      auth_mode: "system_keys",
    })

    await renderAtUrlStep()

    stepUrlProps.on_success?.("https://example.com/a.git", "system_keys")
    await tick()

    expect(get(git_import_wizard_store).clone_path).toBe("/clone/a")
  })
})

describe("ImportProject local trust guard", () => {
  it("confirming trust with no selected path returns to file selection and does not import", async () => {
    // Reach the local trust page with an empty path (the file picker was
    // unavailable, so the manual step showed Continue). Trust Project must not
    // POST an empty project path.
    const result = render(ImportProject, { props: baseProps })
    await tick()
    await fireEvent.click(result.getByText("Import from Local Folder"))
    await tick()

    // Make the file picker fail so the Continue button is shown with no path.
    vi.mocked(client.GET).mockRejectedValue(new Error("No file selector"))
    const { container } = result
    const selectBtn = container.querySelector(
      "button.btn-primary",
    ) as HTMLButtonElement
    await fireEvent.click(selectBtn)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    // Continue -> trust page (path is still empty).
    const continueBtn = container.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement
    await fireEvent.click(continueBtn)
    await tick()

    await waitFor(() => {
      expect(container.textContent).toContain("Trust this Project?")
    })

    // Trust Project with no path must redirect back, not import.
    const trustBtn = container.querySelector(
      "button.btn-warning",
    ) as HTMLButtonElement
    await fireEvent.click(trustBtn)
    await tick()
    await new Promise((r) => setTimeout(r, 0))
    await tick()

    expect(container.textContent).not.toContain("Trust this Project?")
    expect(container.textContent).toContain(
      "Select or enter the path to a project.kiln file",
    )
    expect(vi.mocked(client.POST)).not.toHaveBeenCalled()
  })
})
