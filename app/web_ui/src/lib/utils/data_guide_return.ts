// Where the Data Guide setup chain returns to. The chain (chooser, manual
// setup, Kiln Pro draft, the saved-guide view and its refine loop) is opened
// from synthetic data generation and from the eval builder, and every page
// in it needs to know which, for its breadcrumb and for where its finish
// action lands. The caller travels as a `return_to` search param that each
// page forwards to the next. It is an allowlisted key, never a URL: the param
// is user-editable, and a key can only ever resolve to one of the two known
// pages. No key means synthetic data generation.

import { get } from "svelte/store"
import { sessionStorageStore } from "$lib/stores/local_storage_store"

export type DataGuideCaller = "synth" | "builder"

const DATA_GUIDE_RETURN_PARAM = "return_to"

const CALLERS: readonly DataGuideCaller[] = ["synth", "builder"]

function is_caller(value: unknown): value is DataGuideCaller {
  return typeof value === "string" && CALLERS.includes(value as DataGuideCaller)
}

// The caller named in a page's URL, or null when there is none or the value
// is not one of the allowed keys.
export function read_data_guide_caller(
  params: URLSearchParams | null | undefined,
): DataGuideCaller | null {
  const value = params?.get(DATA_GUIDE_RETURN_PARAM)
  return is_caller(value) ? value : null
}

// `path` with the caller forwarded on it. A null caller returns the path
// untouched, so a page opened without a key hands the next page no key.
export function with_data_guide_caller(
  path: string,
  caller: DataGuideCaller | null,
): string {
  if (caller === null) return path
  const [base, query = ""] = path.split("?", 2)
  const params = new URLSearchParams(query)
  params.set(DATA_GUIDE_RETURN_PARAM, caller)
  return `${base}?${params.toString()}`
}

export type DataGuideReturn = {
  // The breadcrumb root every page in the chain shows.
  label: string
  // Where the setup pages return: their breadcrumb, and the saved screen's
  // Continue. Synthetic data generation resumes its session here.
  href: string
  // Where the saved-guide view returns: its breadcrumb, and its redirects
  // when there is no guide to show.
  view_href: string
}

export function data_guide_return(
  caller: DataGuideCaller | null,
  project_id: string,
  task_id: string,
): DataGuideReturn {
  if (caller === "builder") {
    const href = `/specs/${project_id}/${task_id}/builder`
    return { label: "Evals", href, view_href: href }
  }
  return {
    label: "Synthetic Data Generation",
    href: `/generate/${project_id}/${task_id}/synth?session_continued=true`,
    view_href: `/generate/${project_id}/${task_id}/synth`,
  }
}

// The Kiln Pro connect page is a static route: it leaves for OAuth and comes
// back through a fixed URL, so a search param cannot survive the hop. The
// caller is stashed in sessionStorage on the way out and read on the way
// back; the stash lives as long as the tab and is overwritten (or cleared)
// every time the chain reaches the connect page. Read through the allowlist
// like the param, since storage is as editable as a URL.
const caller_stash = sessionStorageStore<string | null>(
  "kiln_data_guide_return_to",
  null,
)

export function stash_data_guide_caller(caller: DataGuideCaller | null): void {
  caller_stash.set(caller)
}

export function stashed_data_guide_caller(): DataGuideCaller | null {
  const value = get(caller_stash)
  return is_caller(value) ? value : null
}
