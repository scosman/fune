// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from "vitest"
import { render, cleanup } from "@testing-library/svelte"
import { tick } from "svelte"

// ---------------------------------------------------------------------------
// Module-level mocks — must come before the dynamic page import
// ---------------------------------------------------------------------------

const {
  mockPage,
  mockClientGET,
  setEvalsResponse,
  holdCopilotCheck,
  holdDraftCheck,
  isDraftHeld,
} = vi.hoisted(() => {
  type PageValue = {
    params: Record<string, string>
    url: URL
  }
  const pageValue: PageValue = {
    params: { project_id: "proj1", task_id: "task1" },
    url: new URL("http://localhost/specs/proj1/task1"),
  }
  const mockPage = {
    subscribe(fn: (value: PageValue) => void) {
      fn(pageValue)
      return () => {}
    },
  }

  const default_evals_response = {
    evals: [
      {
        id: "eval1",
        name: "Test Eval",
        created_at: "2024-01-01T00:00:00Z",
        output_scores: [],
      },
    ],
    load_error_count: 0,
  }
  let evals_response: Record<string, unknown> = default_evals_response
  const setEvalsResponse = (v: Record<string, unknown> | null) => {
    evals_response = v ?? default_evals_response
  }

  // Keeps the copilot check pending, so the page sits with one loader
  // outstanding and every other one settled.
  let copilot_held = false
  const holdCopilotCheck = (held: boolean) => {
    copilot_held = held
  }

  const mockClientGET = vi.fn().mockImplementation((path: string) => {
    if (path.endsWith("/evals")) {
      return Promise.resolve({ data: evals_response, error: null })
    }
    if (path.endsWith("/specs")) {
      return Promise.resolve({ data: [], error: null })
    }
    if (path.includes("verify_kiln_copilot_api_key")) {
      return copilot_held
        ? new Promise(() => {})
        : Promise.resolve({ data: { is_valid: false }, error: null })
    }
    return Promise.resolve({ data: null, error: null })
  })

  // Keeps the IndexedDB draft peek pending forever, which is what a real
  // browser does when open() fires `blocked` — index_db_store resolves on
  // success and error but not on that, so the promise never settles.
  let draft_held = false
  const holdDraftCheck = (held: boolean) => {
    draft_held = held
  }
  const isDraftHeld = () => draft_held

  return {
    mockPage,
    mockClientGET,
    setEvalsResponse,
    holdCopilotCheck,
    holdDraftCheck,
    isDraftHeld,
  }
})

// jsdom has no IndexedDB, so the real store takes its non-browser branch and
// hands back an already-resolved promise — the async path a real browser takes
// is never exercised. This mock restores it, and can hold it open.
vi.mock("$lib/stores/index_db_store", async () => {
  const { writable } = await import("svelte/store")
  return {
    indexedDBStore: <T>(_key: string, initial: T) => ({
      store: writable(initial),
      initialized: isDraftHeld()
        ? new Promise<void>(() => {})
        : Promise.resolve(),
      persist: () => Promise.resolve(),
    }),
  }
})

// Svelte 4 async onMount callbacks do not execute in jsdom/vitest, and the page keeps
// its loading state until the onMount copilot check settles. Run the callback inline.
vi.mock("svelte", async (importOriginal) => {
  const actual = await importOriginal<typeof import("svelte")>()
  return {
    ...actual,
    // @testing-library/svelte probes this Svelte 5 only export to pick a mount strategy.
    mount: undefined,
    onMount: (fn: () => unknown) => void fn(),
  }
})

vi.mock("$app/stores", () => ({
  page: mockPage,
}))

vi.mock("$app/navigation", () => ({
  goto: vi.fn(),
  replaceState: vi.fn(),
}))

vi.mock("$lib/api_client", () => ({
  client: {
    GET: mockClientGET,
  },
}))

vi.mock("$lib/agent", () => ({
  agentInfo: { set: vi.fn() },
}))

vi.mock("posthog-js", () => ({
  default: { capture: vi.fn() },
}))

vi.mock("../../../app_page.svelte", async () => {
  const Stub = await import("./__tests__/app_page_stub.svelte")
  return { default: Stub.default }
})

// Dynamic import after all mocks
const SpecsPage = (await import("./+page.svelte")).default

const EXPECTED_TOOLTIP =
  "You may need to update Kiln. Some evals could not be opened by this version of Kiln."

// Mirrors the page's own bound on the draft peek. Kept in step by the test
// below, which fails if the page waits any longer than this.
const DRAFT_CHECK_TIMEOUT_MS = 2000

async function render_page() {
  const result = render(SpecsPage)
  // The page has three independent loaders (specs, evals, copilot settings) and only
  // renders its body once all three settle, so flush a few macrotask turns.
  for (let i = 0; i < 5; i++) {
    await new Promise((r) => setTimeout(r, 0))
    await tick()
  }
  return result
}

// The create button is LABELLED from loaders that settle at different times
// ("Create Eval" vs "Continue Eval Draft"), so a button painted before they
// all land rewrites itself a few ms later — a flash on every page load. The
// button is held with the body instead: one spinner, then both together.
describe("specs page — the create button waits for the body", () => {
  afterEach(() => {
    cleanup()
    holdCopilotCheck(false)
    holdDraftCheck(false)
    setEvalsResponse(null)
  })

  function app_page(container: HTMLElement) {
    return container.querySelector("[data-testid='app-page-stub']")!
  }

  it("renders no action button while any loader is still outstanding", async () => {
    // Specs and evals land; the copilot check does not. The evals list is
    // non-empty, so the button would render on its own were it gated only on
    // there being something to show.
    holdCopilotCheck(true)
    const { container } = await render_page()

    expect(app_page(container).getAttribute("data-action-button-count")).toBe(
      "0",
    )
    // And the body is still the spinner, so nothing has arrived early.
    expect(container.querySelector(".loading-spinner")).not.toBeNull()
  })

  it("renders the button once everything has landed, labelled and ready", async () => {
    const { container } = await render_page()

    expect(app_page(container).getAttribute("data-action-button-count")).toBe(
      "1",
    )
    expect(app_page(container).getAttribute("data-action-button-labels")).toBe(
      "Create Eval",
    )
    expect(container.querySelector(".loading-spinner")).toBeNull()
  })

  // The draft peek is the one loader with no network timeout behind it, and
  // IndexedDB can hang outright rather than fail. Holding the page on it would
  // trade a mislabelled button for a screen that never arrives, so the peek is
  // bounded and gives up.
  it("gives up on a draft peek that never settles instead of holding the page", async () => {
    holdDraftCheck(true)
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const { container } = render(SpecsPage)

      // The API loaders land; the peek does not, so the page is still waiting.
      await vi.advanceTimersByTimeAsync(50)
      expect(container.querySelector(".loading-spinner")).not.toBeNull()

      // Past its deadline the peek is abandoned, and the page arrives with the
      // default label — the honest answer when no draft could be read.
      await vi.advanceTimersByTimeAsync(DRAFT_CHECK_TIMEOUT_MS)
      await tick()
      expect(container.querySelector(".loading-spinner")).toBeNull()
      expect(
        app_page(container).getAttribute("data-action-button-labels"),
      ).toBe("Create Eval")
    } finally {
      vi.useRealTimers()
    }
  })
})

describe("specs page — eval load error line", () => {
  afterEach(() => {
    cleanup()
    setEvalsResponse(null)
  })

  it("renders the plural error line and tooltip when several evals fail to load", async () => {
    setEvalsResponse({
      evals: [{ id: "eval1", name: "Test Eval", output_scores: [] }],
      load_error_count: 2,
    })

    const { container } = await render_page()

    const error_line = container.querySelector(".text-error")
    expect(error_line).not.toBeNull()
    expect(error_line?.textContent).toContain("2 evals failed to load")
    expect(error_line?.textContent).toContain(EXPECTED_TOOLTIP)
    expect(container.querySelector("[role='tooltip']")?.textContent).toContain(
      EXPECTED_TOOLTIP,
    )
  })

  it("renders the singular error line when exactly one eval fails to load", async () => {
    setEvalsResponse({
      evals: [{ id: "eval1", name: "Test Eval", output_scores: [] }],
      load_error_count: 1,
    })

    const { container } = await render_page()

    expect(container.textContent).toContain("1 eval failed to load")
    expect(container.textContent).not.toContain("1 evals failed to load")
    expect(container.textContent).toContain(EXPECTED_TOOLTIP)
  })

  it("renders nothing when no evals fail to load", async () => {
    setEvalsResponse({
      evals: [{ id: "eval1", name: "Test Eval", output_scores: [] }],
      load_error_count: 0,
    })

    const { container } = await render_page()

    expect(container.textContent).not.toContain("failed to load")
    expect(container.textContent).not.toContain(EXPECTED_TOOLTIP)
    expect(container.querySelector(".text-error")).toBeNull()
  })
})
