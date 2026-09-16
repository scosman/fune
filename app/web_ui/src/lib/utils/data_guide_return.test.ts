// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  data_guide_return,
  read_data_guide_caller,
  stash_data_guide_caller,
  stashed_data_guide_caller,
  with_data_guide_caller,
} from "./data_guide_return"

function params(query: string): URLSearchParams {
  return new URLSearchParams(query)
}

describe("read_data_guide_caller — the allowlist", () => {
  it("reads the two known callers", () => {
    expect(read_data_guide_caller(params("return_to=synth"))).toBe("synth")
    expect(read_data_guide_caller(params("return_to=builder"))).toBe("builder")
  })

  it("is null with no key, which means synthetic data generation", () => {
    expect(read_data_guide_caller(params(""))).toBeNull()
    expect(read_data_guide_caller(null)).toBeNull()
    expect(read_data_guide_caller(undefined)).toBeNull()
  })

  it("rejects anything that is not a known key — the param is user-editable", () => {
    for (const value of [
      "https://evil.example",
      "//evil.example",
      "/specs/p/t/builder",
      "Builder",
      "builder ",
      "evals",
      "",
    ]) {
      expect(read_data_guide_caller(params(`return_to=${value}`))).toBeNull()
    }
  })
})

describe("with_data_guide_caller — forwarding", () => {
  it("leaves the path untouched with no caller", () => {
    expect(with_data_guide_caller("/generate/p/t/data_guide", null)).toBe(
      "/generate/p/t/data_guide",
    )
  })

  it("appends the caller to a bare path", () => {
    expect(with_data_guide_caller("/generate/p/t/data_guide", "builder")).toBe(
      "/generate/p/t/data_guide?return_to=builder",
    )
  })

  it("keeps existing params and adds the caller", () => {
    const url = with_data_guide_caller("/a/b?draft_failed=1", "builder")
    const query = new URL(url, "http://localhost").searchParams
    expect(query.get("draft_failed")).toBe("1")
    expect(query.get("return_to")).toBe("builder")
  })

  it("round-trips through the reader", () => {
    const url = new URL(with_data_guide_caller("/a/b", "synth"), "http://x")
    expect(read_data_guide_caller(url.searchParams)).toBe("synth")
  })
})

describe("data_guide_return — where each caller lands", () => {
  it("no key returns to synthetic data generation, exactly as before", () => {
    expect(data_guide_return(null, "p1", "t1")).toEqual({
      label: "Synthetic Data Generation",
      href: "/generate/p1/t1/synth?session_continued=true",
      view_href: "/generate/p1/t1/synth",
    })
  })

  it("synth is the same as no key", () => {
    expect(data_guide_return("synth", "p1", "t1")).toEqual(
      data_guide_return(null, "p1", "t1"),
    )
  })

  it("builder returns to the eval builder under its Evals breadcrumb", () => {
    expect(data_guide_return("builder", "p1", "t1")).toEqual({
      label: "Evals",
      href: "/specs/p1/t1/builder",
      view_href: "/specs/p1/t1/builder",
    })
  })
})

describe("the connect-page stash", () => {
  beforeEach(() => {
    sessionStorage.clear()
  })

  it("is empty by default", () => {
    expect(stashed_data_guide_caller()).toBeNull()
  })

  it("stashes a caller for the OAuth hop and reads it back without clearing", () => {
    stash_data_guide_caller("builder")
    expect(stashed_data_guide_caller()).toBe("builder")
    expect(stashed_data_guide_caller()).toBe("builder")
  })

  it("stashing null clears an earlier caller", () => {
    stash_data_guide_caller("builder")
    stash_data_guide_caller(null)
    expect(stashed_data_guide_caller()).toBeNull()
  })

  it("ignores a stash that is not a known key — storage is as editable as a URL", async () => {
    // The stash is read from storage when the module loads, so load a fresh
    // copy over a hand-edited value.
    sessionStorage.setItem(
      "kiln_data_guide_return_to",
      JSON.stringify("https://evil.example"),
    )
    vi.resetModules()
    const fresh = await import("./data_guide_return")
    expect(fresh.stashed_data_guide_caller()).toBeNull()
  })

  it("reads a stash left by an earlier page load", async () => {
    sessionStorage.setItem(
      "kiln_data_guide_return_to",
      JSON.stringify("builder"),
    )
    vi.resetModules()
    const fresh = await import("./data_guide_return")
    expect(fresh.stashed_data_guide_caller()).toBe("builder")
  })
})
