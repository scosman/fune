// Source assertions for Step 5's entry screen. Same house precedent as
// generate_step_surface.test.ts: the builder page is too large to mount, and
// these are contractual — the reviewer supplied this copy word for word, and
// asked for the shared Intro control by name rather than a bespoke screen.
import { describe, expect, it } from "vitest"
import * as fs from "fs"
import * as path from "path"

const intro_source = fs.readFileSync(
  path.resolve(__dirname, "./review_intro.svelte"),
  "utf-8",
)
const page_source = fs.readFileSync(
  path.resolve(__dirname, "./+page.svelte"),
  "utf-8",
)

// Collapses whitespace so an assertion survives Prettier rewrapping a long
// attribute across lines.
function normalize(source: string): string {
  return source.replace(/\s+/g, " ")
}

describe("Step 5 entry screen", () => {
  it("is built on the shared Intro control, not a bespoke screen", () => {
    // Asked for by name. A hand-rolled version drifted on width, type scale,
    // icon treatment and button style, and read as foreign on sight.
    expect(intro_source).toContain('import Intro from "$lib/ui/intro.svelte"')
    expect(normalize(intro_source)).toContain("<Intro")
  })

  it("keeps the reviewer's copy word for word", () => {
    const normalized = normalize(intro_source)
    expect(normalized).toContain('title="Validating the Judge"')
    expect(normalized).toContain(
      "Let's confirm your judge is aligned to your expectations.",
    )
    // The noun is the one substitution: he wrote "examples", which is wrong on
    // the multi-turn arm where every item is a whole conversation.
    expect(normalized).toContain(
      "We'll show a set of ${judged_noun}s, and you tell us if you agree with its judgement.",
    )
  })

  it("offers one primary action, labelled Start", () => {
    const normalized = normalize(intro_source)
    expect(normalized).toContain('label: "Start"')
    expect(normalized).toContain("is_primary: true")
  })

  it("fronts the review rather than replacing it", () => {
    // The intro is shown once per arrival and then hands over to the grading
    // surface. If this inverts, the reviewer never reaches the claims.
    const normalized = normalize(page_source)
    expect(normalized).toContain("{#if !review_intro_dismissed} <ReviewIntro")
    expect(normalized).toContain("on_start={() => (review_intro_dismissed =")
  })

  it("does not persist the dismissal", () => {
    // Per arrival, not per user: a reviewer who reloads has lost the context
    // along with the page, so the screen earns its place again.
    expect(intro_source).not.toContain("localStorage")
    expect(intro_source).not.toContain("sessionStorage")
  })
})
