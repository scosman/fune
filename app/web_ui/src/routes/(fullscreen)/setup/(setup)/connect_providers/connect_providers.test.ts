// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest"
import {
  render,
  cleanup,
  fireEvent,
  screen,
  waitFor,
  within,
} from "@testing-library/svelte"
import * as svelteMod from "svelte"

vi.mock("$lib/api_client", () => ({
  client: {
    GET: vi.fn(),
    POST: vi.fn().mockResolvedValue({
      data: null,
      error: { message: "not running" },
    }),
  },
  base_url: "http://localhost:8757",
}))

vi.mock("$lib/stores", () => ({
  clear_available_models_cache: vi.fn(),
}))

vi.mock("$lib/stores/copilot_connection_store", () => ({
  setCopilotConnected: vi.fn(),
}))

vi.mock("$app/navigation", () => ({
  goto: vi.fn(),
  beforeNavigate: vi.fn(),
}))

vi.mock("posthog-js", () => ({
  default: { capture: vi.fn() },
}))

import ConnectProviders from "./connect_providers.svelte"

const invalid_key_message = "Failed to connect to OpenAI. Invalid API key."

const mock_fetch = vi.fn(async (url: string) => {
  if (url.includes("/api/settings")) {
    return { status: 200, json: async () => ({}) } as unknown as Response
  }
  if (url.includes("/api/provider/connect_api_key")) {
    return {
      status: 401,
      json: async () => ({ message: invalid_key_message }),
    } as unknown as Response
  }
  throw new Error("Unexpected fetch: " + url)
})

beforeEach(() => {
  vi.stubGlobal("fetch", mock_fetch)
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

// Svelte 4 async onMount callbacks do not execute in jsdom/vitest,
// so we capture onMount callbacks via vi.spyOn and invoke them manually.
async function render_connect_providers() {
  const on_mount_callbacks: Array<() => unknown> = []
  const spy = vi
    .spyOn(svelteMod, "onMount")
    .mockImplementation((fn: () => unknown) => {
      on_mount_callbacks.push(fn)
    })

  render(ConnectProviders)

  spy.mockRestore()

  for (const callback of on_mount_callbacks) {
    await callback()
  }
  await svelteMod.tick()
}

async function open_provider_dialog(provider_name: string) {
  const provider_row = screen.getByAltText(provider_name)
    .parentElement as HTMLElement
  await fireEvent.click(
    within(provider_row).getByRole("button", { name: "Connect" }),
  )
}

async function submit_api_key(key: string) {
  await fireEvent.input(screen.getByPlaceholderText("API Key"), {
    target: { value: key },
  })
  await fireEvent.click(screen.getByRole("button", { name: "Connect" }))
}

async function fail_openai_connection() {
  await open_provider_dialog("OpenAI")
  await submit_api_key("bad-key")
  await waitFor(() =>
    expect(screen.getByText(invalid_key_message)).toBeTruthy(),
  )
  await fireEvent.click(
    screen.getByRole("button", { name: "Cancel setting up OpenAI" }),
  )
}

describe("ConnectProviders API key dialog", () => {
  it("does not leak a failed provider's error into another provider's dialog", async () => {
    await render_connect_providers()
    await fail_openai_connection()

    await open_provider_dialog("OpenRouter.ai")

    expect(screen.getByText("Connect OpenRouter.ai")).toBeTruthy()
    expect(screen.queryByText(invalid_key_message)).toBeNull()
  })

  it("does not show a stale error when reopening the same provider's dialog", async () => {
    await render_connect_providers()
    await fail_openai_connection()

    await open_provider_dialog("OpenAI")

    expect(screen.queryByText(invalid_key_message)).toBeNull()
  })

  it("clears the missing field highlight when a dialog is reopened", async () => {
    await render_connect_providers()

    await open_provider_dialog("OpenAI")
    await fireEvent.click(screen.getByRole("button", { name: "Connect" }))
    expect(
      screen.getByPlaceholderText("API Key").classList.contains("input-error"),
    ).toBe(true)

    await fireEvent.click(
      screen.getByRole("button", { name: "Cancel setting up OpenAI" }),
    )
    await open_provider_dialog("OpenAI")

    expect(
      screen.getByPlaceholderText("API Key").classList.contains("input-error"),
    ).toBe(false)
  })

  it("closes the dialog and clears the error after a successful connection", async () => {
    await render_connect_providers()

    await open_provider_dialog("OpenAI")
    await submit_api_key("bad-key")
    await waitFor(() =>
      expect(screen.getByText(invalid_key_message)).toBeTruthy(),
    )

    mock_fetch.mockImplementationOnce(
      async () =>
        ({ status: 200, json: async () => ({}) }) as unknown as Response,
    )
    await submit_api_key("good-key")

    await waitFor(() => expect(screen.queryByText("Connect OpenAI")).toBeNull())
    expect(screen.queryByText(invalid_key_message)).toBeNull()
  })
})
