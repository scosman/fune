import { describe, expect, it } from "vitest"
import type { TraceMessage } from "$lib/types"
import {
  apply_rejudge_results,
  build_claim_review_payload,
  build_graded_traces,
  build_trace_reviews,
  calibration_gate_target,
  reviewable_subset,
  disagreed_trace_indices,
  disagreement_feedback,
  flipped_indices,
  fold_with_offsets,
  grade_disagreement_count,
  has_grade_disagreement,
  human_verdict,
  is_trace_reviewed,
  map_input_span_to_trace,
  map_output_span_to_trace,
  MAX_JUDGE_PROMPT_CHARS,
  plan_save_action,
  rejudge_shortfall_notice,
  resolve_citation_span,
  resolve_citation_span_whitespace_tolerant,
  review_target,
  reviewed_trace_count,
  select_calibration_subset,
  select_review_subset,
  split_claim_note,
  strip_wrapping_code_fence,
  tokenize_claim_text,
  user_says_meets_spec,
  validate_refined_judge_prompt,
  WHITESPACE_FREE_FOLD,
  WHITESPACE_TOLERANT_FOLD,
  type Claim,
  type RejudgeCaseResult,
  type TraceClaims,
  type TraceReview,
} from "./claim_evidence"

function claim(overrides: Partial<Claim> = {}): Claim {
  return {
    text: "The agent stated a return window as fact [1].",
    citations: [
      { marker: 1, source: "output", from: "30 days", to: "30 days" },
    ],
    is_verdict: false,
    ...overrides,
  }
}

// The verdict claim the builder writes last, when it writes one.
function verdict_claim(overrides: Partial<Claim> = {}): Claim {
  return claim({
    text: "It fails because the window was never verified [1].",
    is_verdict: true,
    ...overrides,
  })
}

// A built trace with one ordinary claim and the verdict claim.
function trace(overrides: Partial<TraceClaims> = {}): TraceClaims {
  return {
    trace_id: "trace_0",
    leaf_run_id: null,
    raw_input: "What's the return window?",
    raw_output: "Our return window is 30 days.",
    judge_score: "fail",
    judge_reasoning: "Fabricated the window.",
    overview: {
      text: "The user asked about the return window.",
      citations: [],
    },
    claims: [claim(), verdict_claim()],
    claims_state: "built",
    claims_error: null,
    ...overrides,
  }
}

// A trace whose claims build FAILED: no overview, no claims.
function errored(overrides: Partial<TraceClaims> = {}): TraceClaims {
  return trace({
    overview: null,
    claims: null,
    claims_state: "error",
    claims_error: "boom",
    ...overrides,
  })
}

// A review with every claim agreed.
function all_agreed(t: TraceClaims): TraceReview {
  return {
    trace_id: t.trace_id,
    claim_verdicts: (t.claims ?? []).map(() => ({ agrees: true, why: "" })),
  }
}

describe("build_trace_reviews", () => {
  it("creates one positional verdict per claim", () => {
    const reviews = build_trace_reviews([trace({ claims: [claim(), claim()] })])
    expect(reviews[0].claim_verdicts).toEqual([
      { agrees: null, why: "" },
      { agrees: null, why: "" },
    ])
  })

  it("starts with no slots for a trace whose claims have not arrived", () => {
    const reviews = build_trace_reviews([trace({ claims: null })])
    expect(reviews[0].claim_verdicts).toHaveLength(0)
  })
})

describe("is_trace_reviewed", () => {
  it("needs a grade on every claim", () => {
    const t = trace()
    const review = build_trace_reviews([t])[0]
    expect(is_trace_reviewed(t, review)).toBe(false)
    review.claim_verdicts[0].agrees = true
    expect(is_trace_reviewed(t, review)).toBe(false)
    review.claim_verdicts[1].agrees = true
    expect(is_trace_reviewed(t, review)).toBe(true)
  })

  it("needs a why on any disagreement", () => {
    const t = trace()
    const review = all_agreed(t)
    review.claim_verdicts[0].agrees = false
    expect(is_trace_reviewed(t, review)).toBe(false)
    review.claim_verdicts[0].why = "Wrong claim."
    expect(is_trace_reviewed(t, review)).toBe(true)
  })

  it("is reviewed only when a verdict claim carries the call", () => {
    const with_verdict = trace()
    expect(is_trace_reviewed(with_verdict, all_agreed(with_verdict))).toBe(true)

    // Nothing on screen records pass/fail, so grading every claim is not a
    // finished review. Such a trace never reaches the reviewer anyway.
    const without = trace({ claims: [claim(), claim()] })
    expect(is_trace_reviewed(without, all_agreed(without))).toBe(false)
  })

  it("is never reviewed before the claim slots are sized to the claims", () => {
    // The slots are sized when the build lands; an empty slot list must not
    // read as "every claim graded".
    const t = trace()
    const review: TraceReview = {
      trace_id: t.trace_id,
      claim_verdicts: [],
    }
    expect(is_trace_reviewed(t, review)).toBe(false)
  })

  it("is never reviewed while the build is unbuilt or in flight", () => {
    for (const claims_state of ["unbuilt", "building"] as const) {
      const t = trace({ overview: null, claims: null, claims_state })
      const review: TraceReview = {
        trace_id: t.trace_id,
        claim_verdicts: [],
      }
      expect(is_trace_reviewed(t, review)).toBe(false)
    }
  })
})

describe("human_verdict / user_says_meets_spec", () => {
  it("derives the call from the verdict claim: agree keeps the judge's, disagree flips it", () => {
    for (const judge_score of ["pass", "fail"] as const) {
      const flipped = judge_score === "pass" ? "fail" : "pass"
      const t = trace({ judge_score })
      const review = all_agreed(t)
      expect(human_verdict(t, review)).toBe(judge_score)
      expect(user_says_meets_spec(t, review)).toBe(judge_score === "pass")

      review.claim_verdicts[1] = { agrees: false, why: "Judge was wrong." }
      expect(human_verdict(t, review)).toBe(flipped)
      expect(user_says_meets_spec(t, review)).toBe(flipped === "pass")
    }
  })

  it("ignores grades on ordinary claims: only the verdict claim moves the call", () => {
    const t = trace({ judge_score: "fail" })
    const review = all_agreed(t)
    review.claim_verdicts[0] = { agrees: false, why: "Not a fact." }
    expect(human_verdict(t, review)).toBe("fail")
  })

  it("has no call at all when the builder wrote no verdict claim", () => {
    const t = trace({ claims: [claim(), claim()], judge_score: "fail" })
    const review = all_agreed(t)
    expect(human_verdict(t, review)).toBeNull()
    expect(() => user_says_meets_spec(t, review)).toThrow(/graded/)
  })

  it("refuses to guess an unanswered call", () => {
    const t = trace()
    const review = build_trace_reviews([t])[0]
    expect(human_verdict(t, review)).toBeNull()
    expect(() => user_says_meets_spec(t, review)).toThrow(/graded/)
  })
})

describe("review_target", () => {
  it("is N//4 with a floor of one", () => {
    expect(review_target(40)).toBe(10)
    expect(review_target(38)).toBe(9)
    expect(review_target(4)).toBe(1)
    expect(review_target(3)).toBe(1)
    expect(review_target(1)).toBe(1)
    expect(review_target(0)).toBe(0)
  })

  it("stops growing past ten, so a bigger batch is not more review", () => {
    // The default batch is 80, where a quarter would be 20. Reviewing is human
    // work that does not scale with the batch, so the ask stays at ten and the
    // answer key is simply shorter than the server's 25% ceiling.
    expect(review_target(80)).toBe(10)
    expect(review_target(200)).toBe(10)
  })
})

describe("select_review_subset", () => {
  function batch(
    scores: ("pass" | "fail")[],
  ): Pick<TraceClaims, "judge_score">[] {
    return scores.map((judge_score) => ({ judge_score }))
  }

  it("selects everything when the target covers the batch", () => {
    expect(select_review_subset(batch(["pass"]))).toEqual([0])
  })

  it("prefers a judge-fail for a tiny batch's single slot", () => {
    expect(select_review_subset(batch(["pass", "fail", "pass"]))).toEqual([1])
  })

  it("stratifies ~50/50 across judge verdicts at 40", () => {
    // 20 passes then 20 fails: 10 selected, 5 from each bucket.
    const traces = batch([
      ...Array(20).fill("pass"),
      ...Array(20).fill("fail"),
    ] as ("pass" | "fail")[])
    const picked = select_review_subset(traces)
    expect(picked).toHaveLength(10)
    const fails = picked.filter((i) => traces[i].judge_score === "fail")
    expect(fails).toHaveLength(5)
    // Spread across plan order, not clustered at the front.
    expect(picked.some((i) => i >= 30)).toBe(true)
    expect(picked.some((i) => i < 10)).toBe(true)
  })

  it("tops up from the other bucket when one is short", () => {
    // 39 passes, 1 fail: the fail is always picked, passes fill to 10.
    const traces = batch([...Array(39).fill("pass"), "fail"] as (
      | "pass"
      | "fail"
    )[])
    const picked = select_review_subset(traces)
    expect(picked).toHaveLength(10)
    expect(picked).toContain(39)
  })

  it("handles single-verdict batches (all pass)", () => {
    const traces = batch(Array(12).fill("pass") as ("pass" | "fail")[])
    const picked = select_review_subset(traces)
    expect(picked).toHaveLength(3)
    expect(new Set(picked).size).toBe(3)
  })

  it("is deterministic", () => {
    const traces = batch([
      "pass",
      "fail",
      "pass",
      "fail",
      "pass",
      "fail",
      "pass",
      "fail",
    ])
    expect(select_review_subset(traces)).toEqual(select_review_subset(traces))
  })
})

describe("reviewed_trace_count", () => {
  it("counts reviewed traces only (the save-gate numerator)", () => {
    const traces = [trace(), trace({ trace_id: "trace_1" })]
    const reviews = build_trace_reviews(traces)
    expect(reviewed_trace_count(traces, reviews)).toBe(0)
    reviews[0] = all_agreed(traces[0])
    expect(reviewed_trace_count(traces, reviews)).toBe(1)
  })
})

describe("errored-build trace is not gradable at all", () => {
  // A trace whose claims build failed has nothing on screen to grade, and
  // nothing records its pass/fail call. It never counts toward the save gate;
  // the retry is the only way forward.
  it("is never reviewed, however the review is filled in", () => {
    const t = errored()
    const reviews = build_trace_reviews([t])
    expect(reviews[0].claim_verdicts).toHaveLength(0)
    expect(reviewed_trace_count([t], reviews)).toBe(0)
    expect(() => user_says_meets_spec(t, reviews[0])).toThrow(/graded/)
  })
})

describe("build_claim_review_payload", () => {
  it("throws before claims are built (unbuilt and errored traces have no claims to grade)", () => {
    const unbuilt = trace({
      overview: null,
      claims: null,
      claims_state: "unbuilt",
    })
    expect(() =>
      build_claim_review_payload(unbuilt, build_trace_reviews([unbuilt])[0]),
    ).toThrow(/built/)
    const failed = errored()
    const review = build_trace_reviews([failed])[0]
    expect(() => build_claim_review_payload(failed, review)).toThrow(/built/)
  })

  it("throws on a half-graded trace rather than inventing a grade", () => {
    // The answer key must never carry a grade the reviewer never gave.
    // Reachable only if a caller ever skips the is_trace_reviewed gate — most
    // pressingly after a calibration round, which resets every grade to null.
    const t = trace()
    const review = all_agreed(t)
    review.claim_verdicts[1].agrees = null
    expect(() => build_claim_review_payload(t, review)).toThrow(/graded/)
  })

  it("writes the overview, every claim's grade and the derived call", () => {
    const t = trace()
    const review: TraceReview = {
      trace_id: "trace_0",
      claim_verdicts: [
        { agrees: true, why: "" },
        { agrees: false, why: "  Policy is real.  " },
      ],
    }
    expect(build_claim_review_payload(t, review)).toEqual({
      judge_score: "fail",
      judge_reasoning: "Fabricated the window.",
      overview: "The user asked about the return window.",
      claims: [
        {
          text: "The agent stated a return window as fact [1].",
          human_grade: "agree",
          human_feedback: null,
        },
        {
          text: "It fails because the window was never verified [1].",
          human_grade: "disagree",
          human_feedback: "Policy is real.",
        },
      ],
      human_verdict: "pass",
    })
  })

  it("throws when the builder wrote no verdict claim: nothing records the call", () => {
    const t = trace({ claims: [claim()] })
    expect(() => build_claim_review_payload(t, all_agreed(t))).toThrow(/graded/)
  })
})

describe("disagreement_feedback", () => {
  it("concatenates the disagree whys across the claims", () => {
    const review: TraceReview = {
      trace_id: "trace_0",
      claim_verdicts: [
        { agrees: false, why: "claim why" },
        { agrees: true, why: "ignored" },
        { agrees: false, why: "verdict why" },
      ],
    }
    expect(disagreement_feedback(review)).toBe("claim why verdict why")
  })
})

describe("build_graded_traces", () => {
  it("includes only fully graded traces and labels them by run id, else trace id", () => {
    const reviewed_t = trace({ leaf_run_id: "leaf-abc" })
    const reviewed_review = all_agreed(reviewed_t)
    reviewed_review.claim_verdicts[1] = {
      agrees: false,
      why: "policy is real",
    }
    const half_t = trace({ trace_id: "trace_1", leaf_run_id: null })
    const half_review = build_trace_reviews([half_t])[0] // ungraded → excluded

    const graded = build_graded_traces(
      [reviewed_t, half_t],
      [reviewed_review, half_review],
    )
    expect(graded).toHaveLength(1)
    expect(graded[0]).toEqual({
      trace_label: "leaf-abc",
      ...build_claim_review_payload(reviewed_t, reviewed_review),
    })
    // Falls back to the client trace id when no durable run id exists.
    const single = build_graded_traces([half_t], [all_agreed(half_t)])
    expect(single[0].trace_label).toBe("trace_1")
  })

  it("leaves out a trace whose claims build failed", () => {
    // No claims, so no claim grades to hand the refiner.
    const t = errored()
    expect(build_graded_traces([t], build_trace_reviews([t]))).toEqual([])
  })

  it("excludes a trace whose claims build is still in flight", () => {
    const in_flight = trace({
      overview: null,
      claims: null,
      claims_state: "building",
    })
    const review: TraceReview = {
      trace_id: "trace_0",
      claim_verdicts: [],
    }
    expect(build_graded_traces([in_flight], [review])).toHaveLength(0)
  })
})

describe("validate_refined_judge_prompt", () => {
  it("accepts a plain-text prompt", () => {
    expect(
      validate_refined_judge_prompt(
        "PASS if the reply is polite, FAIL otherwise.",
      ),
    ).toBeNull()
  })

  it("rejects empty / whitespace-only prompts", () => {
    expect(validate_refined_judge_prompt("")).toMatch(/empty/)
    expect(validate_refined_judge_prompt("   \n ")).toMatch(/empty/)
  })

  it("rejects Jinja/template braces and code fences", () => {
    expect(validate_refined_judge_prompt("Score {{ trace }}")).toMatch(/Jinja/)
    expect(validate_refined_judge_prompt("{% if x %}pass{% endif %}")).toMatch(
      /Jinja/,
    )
    expect(validate_refined_judge_prompt("{# comment #} judge")).toMatch(
      /Jinja/,
    )
    expect(validate_refined_judge_prompt("```\nrubric\n```")).toMatch(/fences/)
  })

  it("rejects an oversized prompt", () => {
    expect(
      validate_refined_judge_prompt("a".repeat(MAX_JUDGE_PROMPT_CHARS + 1)),
    ).toMatch(/too long/)
  })
})

describe("strip_wrapping_code_fence", () => {
  it("unwraps a fenced prompt, with or without a language tag", () => {
    expect(strip_wrapping_code_fence("```\nPASS if polite.\n```")).toBe(
      "PASS if polite.",
    )
    expect(strip_wrapping_code_fence("```markdown\nPASS if polite.\n```")).toBe(
      "PASS if polite.",
    )
    expect(
      validate_refined_judge_prompt(
        strip_wrapping_code_fence("```markdown\nPASS if polite.\n```"),
      ),
    ).toBeNull()
  })

  it("leaves a prompt without a wrapping fence unchanged", () => {
    const plain = "PASS if the reply is polite, FAIL otherwise."
    expect(strip_wrapping_code_fence(plain)).toBe(plain)
  })

  it("leaves interior and one-sided fences unchanged", () => {
    const interior = "Judge this:\n```\nexample\n```\nThen decide."
    expect(strip_wrapping_code_fence(interior)).toBe(interior)
    const open_only = "```markdown\nPASS if polite."
    expect(strip_wrapping_code_fence(open_only)).toBe(open_only)
    const close_only = "PASS if polite.\n```"
    expect(strip_wrapping_code_fence(close_only)).toBe(close_only)
  })

  it("strips only one wrapping pair, so a doubly wrapped prompt still fails", () => {
    const doubled = "```\n```\nPASS if polite.\n```\n```"
    expect(strip_wrapping_code_fence(doubled)).toBe("```\nPASS if polite.\n```")
    expect(
      validate_refined_judge_prompt(strip_wrapping_code_fence(doubled)),
    ).toMatch(/fences/)
  })

  it("recovers the wrapping only, never the content", () => {
    // Jinja braces inside the fence survive the unwrap and still fail.
    expect(
      validate_refined_judge_prompt(
        strip_wrapping_code_fence("```\nScore {{ trace }}\n```"),
      ),
    ).toMatch(/Jinja/)
    // An empty fence unwraps to nothing, which is still not a judge prompt.
    expect(
      validate_refined_judge_prompt(strip_wrapping_code_fence("```\n\n```")),
    ).toMatch(/empty/)
    // Runaway output stays runaway once the fence is off.
    expect(
      validate_refined_judge_prompt(
        strip_wrapping_code_fence(
          `\`\`\`\n${"a".repeat(MAX_JUDGE_PROMPT_CHARS + 1)}\n\`\`\``,
        ),
      ),
    ).toMatch(/too long/)
  })
})

describe("resolve_citation_span", () => {
  it("resolves from/to anchors in order", () => {
    const span = resolve_citation_span(
      "return window is 30 days from purchase",
      {
        from: "30 days",
        to: "purchase",
      },
    )
    // from_end closes the `from` anchor alone, for callers that cannot place
    // the whole span.
    expect(span).toEqual({ start: 17, from_end: 24, end: 38 })
  })

  it("returns null when an anchor is missing", () => {
    expect(
      resolve_citation_span("no anchors here", { from: "30 days", to: "x" }),
    ).toBeNull()
  })
})

describe("resolve_citation_span — curly vs straight punctuation", () => {
  // The captured shape: the model's output carries typographic quotes, the
  // anchors it retyped for the citation carry straight ones.
  const curly = "Please confirm whether you’d like a “refund” or store credit."

  it("resolves straight-quote anchors against curly source text", () => {
    const span = resolve_citation_span(curly, {
      from: "you'd like",
      to: "“refund”",
    })
    expect(span).not.toBeNull()
    // The offsets index the ORIGINAL text, curly characters and all — they
    // slice the highlight the reviewer sees.
    expect(curly.slice(span!.start, span!.end)).toBe("you’d like a “refund”")
  })

  it("resolves curly anchors against straight source text", () => {
    const straight = "Please confirm whether you'd like a refund."
    const span = resolve_citation_span(straight, {
      from: "you’d like",
      to: "refund",
    })
    expect(span).not.toBeNull()
    expect(straight.slice(span!.start, span!.end)).toBe("you'd like a refund")
  })

  it("still resolves when both sides are curly", () => {
    // Straight-on-straight is the suite above; this is the other unchanged
    // pairing, which one-sided folding would break.
    const span = resolve_citation_span(curly, {
      from: "you’d like",
      to: "store credit",
    })
    expect(span).not.toBeNull()
    expect(curly.slice(span!.start, span!.end)).toBe(
      "you’d like a “refund” or store credit",
    )
  })

  it("keeps offsets exact around every folded and unfolded character", () => {
    // Folding may only ever swap one code unit for one. Text mixing all four
    // folded characters with punctuation that is NOT folded (the en dash) puts
    // any length-changing substitution straight into the slice.
    const mixed = "Policy — the “30 day” window — you’d like a ‘refund’ now."
    const span = resolve_citation_span(mixed, {
      from: '"30 day"',
      to: "'refund'",
    })
    expect(span).not.toBeNull()
    expect(mixed.slice(span!.start, span!.end)).toBe(
      "“30 day” window — you’d like a ‘refund’",
    )
  })

  it("still returns null when the cited text genuinely isn't there", () => {
    // Folding punctuation must not turn a real miss into a match.
    expect(
      resolve_citation_span(curly, {
        from: "you'd like",
        to: "a restocking fee",
      }),
    ).toBeNull()
  })
})

// The tool-trace case's shape at its size: one JSON object whose
// `updated_mention_sources` array carries the sources a discovery run seeded.
// Built here rather than imported from the dev-only preview corpus, so the test
// owns its fixture, but sized past 5000 characters like the run the bug was
// reported on — the mark being lost in an unformatted blob is a length problem.
function jumbo_json_output(): string {
  const kinds = ["web_search", "news_outlet", "rss_feed", "twitter_query"]
  return JSON.stringify({
    summary:
      "Initial mention-source curation for Acme Construction Tech: seeded 15 probationary agent sources across web, news, RSS, X, and Reddit tuned to Construction PM SaaS.",
    updated_mention_sources: Array.from({ length: 15 }, (_, i) => ({
      kind: kinds[i % kinds.length],
      value: `"Acme Construction Tech" source ${i} -site:acmeconstructiontech.example`,
      notes: `Seeded source ${i}: exact-name brand mentions scoped to the construction context and excluding the owned domain, so unrelated collisions stay out of the feed.`,
      addedBy: "agent",
      priority: "probationary",
    })),
    brand_memory_record: {
      type: "discovery_run",
      ran_at: "2026-07-07T22:15:00Z",
      candidates_evaluated: 22,
      rationale:
        "First discovery run - the Brand Memory feed is empty and no sources existed. For B2B Construction PM SaaS, I biased toward X and Reddit plus web and news, adding US construction trade outlets, matching RSS feeds, and exact/alias web queries scoped away from the known exclusions. All new sources start probationary pending signal validation.",
    },
  })
}

// The anchors the rig's claim synthesizer writes for a case: the first three
// whitespace-separated words of the raw output, then the next three. Retyped
// from the RAW string, which is exactly why they miss the pretty-printed one.
function leading_anchors(raw_output: string): { from: string; to: string } {
  const words = raw_output.split(/\s+/).filter(Boolean)
  return { from: words.slice(0, 3).join(" "), to: words.slice(3, 6).join(" ") }
}

describe("fold_with_offsets", () => {
  it("collapses whitespace runs and maps each fold back over the whole run", () => {
    const pretty = '{\n  "window": "30 days"\n}'
    const { folded, map, map_end } = fold_with_offsets(
      pretty,
      WHITESPACE_TOLERANT_FOLD,
    )
    expect(folded).toBe('{ "window": "30 days" }')
    // The offsets index the ORIGINAL, so slicing with them brings the newline
    // and indent the fold hid back with the text.
    const space = folded.indexOf(" ")
    expect(pretty.slice(map[space], map_end[space])).toBe("\n  ")
    // Every folded character round-trips: a fold that lost a character, or
    // attributed one to the wrong run, shows up as a gap here.
    expect(
      [...folded].map((_, i) => pretty.slice(map[i], map_end[i])).join(""),
    ).toBe(pretty)
  })

  it("folds typography 1:1 while collapsing whitespace around it", () => {
    const text = "Policy:\n  the “30 day”\n  window."
    const { folded, map, map_end } = fold_with_offsets(
      text,
      WHITESPACE_TOLERANT_FOLD,
    )
    expect(folded).toBe('Policy: the "30 day" window.')
    const start = folded.indexOf('"30 day"')
    const end = folded.indexOf("window.") + "window.".length
    // A length-changing typographic fold would shift these offsets and slice
    // the wrong characters out of the original.
    expect(text.slice(map[start], map_end[end - 1])).toBe("“30 day”\n  window.")
  })

  it("copies a run through when no rule owns it", () => {
    // The dispatch asks each rule to match its own run ANCHORED. A pattern
    // whose match depends on what follows it cannot answer that, and the run
    // is copied through as ordinary text — which keeps the offsets exact,
    // where indexing past the end of the rule table would throw and cost the
    // reviewer the citation entirely.
    const { folded, map, map_end } = fold_with_offsets("axyb", [
      { pattern: "x(?=y)", replace: () => "!" },
    ])
    expect(folded).toBe("axyb")
    expect(map).toEqual([0, 1, 2, 3])
    expect(map_end).toEqual([1, 2, 3, 4])
  })

  it("drops whitespace runs entirely under the free fold", () => {
    // The pass that rescues tokens pretty-printing pushed APART: no collapse
    // can match a space against the nothing the raw string had there.
    const { folded, map, map_end } = fold_with_offsets(
      '{\n  "a": 1\n}',
      WHITESPACE_FREE_FOLD,
    )
    expect(folded).toBe('{"a":1}')
    // A dropped run belongs to no folded character, so a span ending before
    // one stops at the last character it really covered.
    expect(map_end[folded.indexOf("1")]).toBe('{\n  "a": 1'.length)
    // …and the character after a dropped run still points past it, or every
    // offset from there on would be short by the run's length.
    expect(map[folded.indexOf('"a"')]).toBe("{\n  ".length)
  })
})

describe("resolve_citation_span_whitespace_tolerant", () => {
  it("resolves anchors across a pretty-printed key/value boundary", () => {
    // The exact bug shape: the model retyped its anchors off the raw string,
    // where the colon is flush; the reviewer is looking at the pretty one,
    // where it is followed by a space and the entries by newline + indent.
    const raw = '{"window":"30 days","refund":"14 days"}'
    const pretty = JSON.stringify(JSON.parse(raw), null, 2)
    const span = resolve_citation_span_whitespace_tolerant(pretty, {
      from: '"window":"30 days"',
      to: '"refund":"14 days"',
    })
    expect(span).not.toBeNull()
    expect(pretty.slice(span!.start, span!.end)).toBe(
      '"window": "30 days",\n  "refund": "14 days"',
    )
  })

  it("resolves anchors whose only difference is newline for space", () => {
    // The collapsing pass on its own: both sides separate the tokens, the
    // rendering just separates them differently.
    const text = "Our return window\nis 30 days\nfrom purchase."
    const span = resolve_citation_span_whitespace_tolerant(text, {
      from: "window is 30 days",
      to: "purchase",
    })
    expect(span).not.toBeNull()
    expect(text.slice(span!.start, span!.end)).toBe(
      "window\nis 30 days\nfrom purchase",
    )
  })

  it("prefers the match that respects word boundaries", () => {
    // Why the collapsing pass answers first: dropping whitespace outright lets
    // `a b` match the earlier `ab`, and the reviewer would be sent to a span
    // the citation never pointed at while the real one sat further down.
    const pretty = JSON.stringify({ tag: "ab", name: "a b" }, null, 2)
    const span = resolve_citation_span_whitespace_tolerant(pretty, {
      from: "a b",
      to: "a b",
    })
    expect(span).not.toBeNull()
    expect(pretty.slice(0, span!.start)).toContain('"tag": "ab"')
  })

  it("keeps a JSON whitespace collision on the right value", () => {
    // The JSON shape the fix exists for only ever reaches the whitespace-FREE
    // pass (a flush `"key":"value"` cannot survive the collapsing one), so the
    // wider pass has to stay honest on a haystack where dropping whitespace
    // makes two different values look alike.
    const pretty = JSON.stringify({ a: "30days", b: "30 days" }, null, 2)
    const span = resolve_citation_span_whitespace_tolerant(pretty, {
      from: '"b":"30 days"',
      to: '"b":"30 days"',
    })
    expect(span).not.toBeNull()
    expect(pretty.slice(span!.start, span!.end)).toBe('"b": "30 days"')
  })

  it("starts a span after the whitespace its first anchor begins on", () => {
    // Symmetric with the trailing clamp below: an anchor retyped with a leading
    // space must not drag the newline and indent pretty-printing inserted into
    // the front of the highlight.
    const pretty = JSON.stringify({ note: "see below", next: 1 }, null, 2)
    const span = resolve_citation_span_whitespace_tolerant(pretty, {
      from: ' "next"',
      to: "1",
    })
    expect(span).not.toBeNull()
    expect(pretty.slice(span!.start, span!.end)).toBe('"next": 1')
  })

  it("stops a span before the whitespace its last anchor ends on", () => {
    // A `to` ending in whitespace must not drag the newline and indent
    // pretty-printing inserted into the mark: on screen that is a highlighted
    // line break, which reads as a rendering bug.
    const pretty = JSON.stringify({ note: "see below", next: 1 }, null, 2)
    const span = resolve_citation_span_whitespace_tolerant(pretty, {
      from: '"note"',
      to: '"see below", ',
    })
    expect(span).not.toBeNull()
    expect(pretty.slice(span!.start, span!.end)).toBe('"note": "see below",')
  })

  it("returns null when the cited text genuinely isn't in the rendering", () => {
    // Whitespace tolerance must not turn a real miss into a match, or the
    // caller would mark a span the citation never pointed at.
    expect(
      resolve_citation_span_whitespace_tolerant('{\n  "a": 1\n}', {
        from: '"restocking fee"',
        to: '"a"',
      }),
    ).toBeNull()
    // A present `from` with an absent `to` is the same miss from the other
    // side: half an anchor pair locates nothing.
    expect(
      resolve_citation_span_whitespace_tolerant('{\n  "a": 1\n}', {
        from: '"a"',
        to: '"restocking fee"',
      }),
    ).toBeNull()
  })

  it("returns null for an anchor that is nothing but whitespace", () => {
    // The fold that drops whitespace empties such an anchor out, and an empty
    // needle matches at every position while covering none of them. Reporting
    // that as a resolution would cost the caller the raw mark it falls back to.
    expect(
      resolve_citation_span_whitespace_tolerant('{\n  "a": 1\n}', {
        from: '"a"',
        to: " ",
      }),
    ).toBeNull()
    expect(
      resolve_citation_span_whitespace_tolerant('{\n  "a": 1\n}', {
        from: "\n  ",
        to: '"a"',
      }),
    ).toBeNull()
  })

  it("anchors the jumbo tool-trace citation in its pretty-printed output", () => {
    const raw = jumbo_json_output()
    expect(raw.length).toBeGreaterThan(5000)
    const anchors = leading_anchors(raw)
    const pretty = JSON.stringify(JSON.parse(raw), null, 2)
    // The anchors resolve against the raw string — that is the mark the
    // reviewer clicked, and today's fallback still shows it there.
    expect(resolve_citation_span(raw, anchors)).not.toBeNull()
    // The 1:1 fold cannot find them in the pretty rendering, which is the bug.
    expect(resolve_citation_span(pretty, anchors)).toBeNull()
    const span = resolve_citation_span_whitespace_tolerant(pretty, anchors)
    expect(span).not.toBeNull()
    expect(pretty.slice(span!.start, span!.end)).toBe(
      '{\n  "summary": "Initial mention-source curation for Acme Construction',
    )
  })
})

describe("map_output_span_to_trace — flattener block layout port", () => {
  // A trace exercising every block kind the formatter emits: a plain user
  // turn, a reasoning-only assistant turn, a tool-call turn, its tool result,
  // and a final content turn.
  const trace = [
    { role: "user", content: "What is the return window?" },
    { role: "assistant", reasoning_content: "Let me think about policy." },
    {
      role: "assistant",
      content: null,
      tool_calls: [
        {
          id: "call_1",
          type: "function",
          function: { name: "lookup_policy", arguments: '{"q":1}' },
        },
      ],
    },
    { role: "tool", tool_call_id: "call_1", content: "30 day window" },
    { role: "assistant", content: "Our return window is 30 days." },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ] as any[]

  // Build raw_output exactly the way the server flattener does: every block a
  // message carries, joined by a blank line. Each message here carries one, so
  // the count matches; a message carrying both text and a call emits two.
  const raw_output = [
    "user:\n<user_message>\nWhat is the return window?\n</user_message>",
    "assistant reasoning:\n<assistant_reasoning_message>\nLet me think about policy.\n</assistant_reasoning_message>",
    'assistant requested tool calls:\n<assistant_requested_tool_calls>\n- Tool Name: lookup_policy\n- Arguments: {"q":1}\n</assistant_requested_tool_calls>',
    "tool result from lookup_policy:\n<tool_tool_message>\n30 day window\n</tool_tool_message>",
    "assistant:\n<assistant_message>\nOur return window is 30 days.\n</assistant_message>",
  ].join("\n\n")

  function map(from: string, to: string) {
    const span = resolve_citation_span(raw_output, { from, to })
    expect(span).not.toBeNull()
    return map_output_span_to_trace(trace, raw_output, span!)
  }

  it("maps a span in a chat (content) turn", () => {
    // `from` unique to the final turn — "return window" also appears in the
    // user turn, and resolve_citation_span anchors on the FIRST occurrence.
    const h = map("Our return", "30 days")
    expect(h).not.toBeNull()
    expect(h!.trace_index).toBe(4)
    expect(h!.kind).toBe("content")
    expect("Our return window is 30 days.".slice(h!.start, h!.end)).toBe(
      "Our return window is 30 days",
    )
    // Both anchors sit in this turn, so the whole span is marked.
    expect(h!.from_anchor_only).toBe(false)
  })

  it("maps a span in a reasoning block", () => {
    const h = map("think", "policy")
    expect(h).not.toBeNull()
    expect(h!.trace_index).toBe(1)
    expect(h!.kind).toBe("reasoning")
    expect("Let me think about policy.".slice(h!.start, h!.end)).toBe(
      "think about policy",
    )
  })

  it("maps a span in a tool-call block", () => {
    const h = map("Tool Name", "lookup_policy")
    expect(h).not.toBeNull()
    expect(h!.trace_index).toBe(2)
    expect(h!.kind).toBe("tool_calls")
  })

  it("marks the from anchor alone when the span crosses two turns", () => {
    // from lands in the user turn, to in the reasoning turn. No block holds
    // the whole span, so the anchor the model wrote is what gets marked.
    const span = resolve_citation_span(raw_output, {
      from: "What is the return",
      to: "think about",
    })
    expect(span).not.toBeNull()
    const h = map_output_span_to_trace(trace, raw_output, span!)
    expect(h).not.toBeNull()
    expect(h!.trace_index).toBe(0)
    expect(h!.kind).toBe("content")
    expect(h!.from_anchor_only).toBe(true)
    expect("What is the return window?".slice(h!.start, h!.end)).toBe(
      "What is the return",
    )
  })

  it("returns null when the span starts in the flattener's tag chrome", () => {
    // The recovery is scoped to the block holding `from`; an anchor that is
    // part of the chrome belongs to no block and has nothing to mark.
    const start = raw_output.indexOf("<assistant_reasoning_message>")
    expect(start).toBeGreaterThan(0)
    expect(
      map_output_span_to_trace(trace, raw_output, {
        start,
        from_end: start + 5,
        end: start + 5,
      }),
    ).toBeNull()
  })

  it("refuses a block whose text has drifted from raw_output", () => {
    // The byte-identity guard: the recomputed layout says this block holds the
    // span, but raw_output disagrees at that offset, so the port has drifted
    // and no highlight is honest — recovery included.
    const drifted = raw_output.replace(
      "Let me think about policy.",
      "Let me think about POLICY!",
    )
    const span = resolve_citation_span(drifted, {
      from: "think about",
      to: "POLICY",
    })
    expect(span).not.toBeNull()
    expect(map_output_span_to_trace(trace, drifted, span!)).toBeNull()
    // And with a cross-turn span, so the from-anchor path hits it too.
    const crossing = resolve_citation_span(drifted, {
      from: "think about",
      to: "Our return window",
    })
    expect(crossing).not.toBeNull()
    expect(map_output_span_to_trace(trace, drifted, crossing!)).toBeNull()
  })

  it("recovers a whitespace-drifted cross-turn citation", () => {
    // The strict resolver misses anchors the model retyped with different
    // whitespace; the tolerant one finds them, and the block mapper still
    // adjudicates the result — here down to the from anchor.
    const anchors = {
      from: "Let me think\nabout policy",
      to: "Our return window",
    }
    expect(resolve_citation_span(raw_output, anchors)).toBeNull()
    const span = resolve_citation_span_whitespace_tolerant(raw_output, anchors)
    expect(span).not.toBeNull()
    const h = map_output_span_to_trace(trace, raw_output, span!)
    expect(h).not.toBeNull()
    expect(h!.trace_index).toBe(1)
    expect(h!.kind).toBe("reasoning")
    expect(h!.from_anchor_only).toBe(true)
    expect("Let me think about policy.".slice(h!.start, h!.end)).toBe(
      "Let me think about policy",
    )
  })
})

describe("map_output_span_to_trace — single-turn output", () => {
  // Single-turn raw_output has none of the flattener's headers or tags: it IS
  // the assistant message's content.
  const raw_output = "Our return window is 30 days from delivery."
  const trace = [
    { role: "system", content: "You are a support agent." },
    { role: "user", content: "What is the return window?" },
    { role: "assistant", content: raw_output },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ] as any[]

  it("maps a span onto the message whose content is the whole output", () => {
    const span = resolve_citation_span(raw_output, {
      from: "30 days",
      to: "delivery",
    })
    expect(span).not.toBeNull()
    const h = map_output_span_to_trace(trace, raw_output, span!)
    expect(h).not.toBeNull()
    expect(h!.trace_index).toBe(2)
    expect(h!.kind).toBe("content")
    // Offsets carry over unchanged — there is no block chrome to subtract.
    expect(h!.start).toBe(span!.start)
    expect(h!.end).toBe(span!.end)
    expect(raw_output.slice(h!.start, h!.end)).toBe("30 days from delivery")
  })

  it("returns null when no message carries the whole output", () => {
    const other = [
      { role: "user", content: "What is the return window?" },
      { role: "assistant", content: "A different answer entirely." },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ] as any[]
    expect(
      map_output_span_to_trace(other, raw_output, { start: 4, end: 10 }),
    ).toBeNull()
  })

  it("returns null when two messages both carry the whole output", () => {
    const echoed = [
      { role: "user", content: raw_output },
      { role: "assistant", content: raw_output },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ] as any[]
    // Which one the citation meant is unknowable, so nothing is highlighted.
    expect(
      map_output_span_to_trace(echoed, raw_output, { start: 4, end: 10 }),
    ).toBeNull()
  })
})

describe("map_input_span_to_trace — input citations land on the opening user bubble", () => {
  // On multi-turn, raw_input is the first user message's content verbatim
  // (the server's transcript_io_for_trace pick), so the offsets carry over.
  const raw_input = "I want to return my order from last week."
  const trace = [
    { role: "system", content: "You are a support agent." },
    { role: "user", content: raw_input },
    { role: "assistant", content: "Happy to help with that." },
    { role: "user", content: "It arrived damaged." },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ] as any[]

  it("maps a span onto the first user message, not a later one", () => {
    const span = resolve_citation_span(raw_input, {
      from: "return my order",
      to: "last week",
    })
    expect(span).not.toBeNull()
    const h = map_input_span_to_trace(trace, raw_input, span!)
    expect(h).not.toBeNull()
    expect(h!.trace_index).toBe(1)
    expect(h!.kind).toBe("content")
    expect(raw_input.slice(h!.start, h!.end)).toBe(
      "return my order from last week",
    )
  })

  it("skips an empty first user message, mirroring the server's pick", () => {
    const with_empty = [
      { role: "user", content: "" },
      { role: "user", content: raw_input },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ] as any[]
    const h = map_input_span_to_trace(with_empty, raw_input, {
      start: 0,
      end: 6,
    })
    expect(h).not.toBeNull()
    expect(h!.trace_index).toBe(1)
  })

  it("returns null when the opening message differs from raw_input", () => {
    // raw_input that did not come from this trace's opening message must not
    // put a mark on it — the offsets would slice a different string.
    const other = [
      { role: "user", content: "A different opening entirely." },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ] as any[]
    expect(
      map_input_span_to_trace(other, raw_input, { start: 0, end: 6 }),
    ).toBeNull()
  })

  it("returns null when the trace has no user message", () => {
    const no_user = [
      { role: "assistant", content: "Hello." },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ] as any[]
    expect(
      map_input_span_to_trace(no_user, raw_input, { start: 0, end: 6 }),
    ).toBeNull()
  })

  it("returns null when the span overruns the input", () => {
    expect(
      map_input_span_to_trace(trace, raw_input, {
        start: 0,
        end: raw_input.length + 1,
      }),
    ).toBeNull()
    expect(
      map_input_span_to_trace(trace, raw_input, { start: -1, end: 6 }),
    ).toBeNull()
  })
})

// ── Calibration loop ──────────────────────────────────────────────────────

function score_traces(
  scores: ("pass" | "fail")[],
): Pick<TraceClaims, "judge_score">[] {
  return scores.map((judge_score) => ({ judge_score }))
}

function rejudge_result(judge_score: "pass" | "fail"): RejudgeCaseResult {
  return {
    judge_score,
    judge_reasoning: `Re-checked: ${judge_score}.`,
    raw_input: "input",
    raw_output: "output",
    trace: null,
  }
}

describe("select_calibration_subset", () => {
  // 16 traces, alternating fail/pass → target floor(16/4) = 4.
  const sixteen = score_traces(
    Array.from({ length: 16 }, (_, i) => (i % 2 === 0 ? "fail" : "pass")),
  )
  const all_judged = sixteen.map((_, i) => i)

  it("fills disagreed first, then flips, then a fresh top-up", () => {
    const subset = select_calibration_subset(sixteen, {
      disagreed: [10],
      flipped: [12, 2],
      reviewed: [10, 11, 12],
      judged: all_judged,
    })
    expect(subset).toHaveLength(4)
    expect(subset).toContain(10)
    expect(subset).toContain(2)
    expect(subset).toContain(12)
    // The remaining slot is a fresh never-reviewed trace.
    const fresh = subset.filter((i) => ![2, 10, 12].includes(i))
    expect(fresh).toHaveLength(1)
    expect([10, 11, 12]).not.toContain(fresh[0])
  })

  it("overflow: disagreed beat flips beat fresh, in stable plan order", () => {
    // 8 traces → target 2; three disagreements overflow the target.
    const eight = score_traces(Array.from({ length: 8 }, () => "pass"))
    expect(
      select_calibration_subset(eight, {
        disagreed: [5, 1, 7],
        flipped: [0, 2],
        reviewed: [1, 5, 7],
        judged: eight.map((_, i) => i),
      }),
    ).toEqual([1, 5])
    // One disagreement + overflowing flips: the flip slot goes to the
    // earliest flipped index.
    expect(
      select_calibration_subset(eight, {
        disagreed: [3],
        flipped: [6, 0, 4],
        reviewed: [3],
        judged: eight.map((_, i) => i),
      }),
    ).toEqual([0, 3])
  })

  it("excludes cases without a fresh verdict from every stratum", () => {
    const eight = score_traces(Array.from({ length: 8 }, () => "pass"))
    const judged = [0, 3, 4, 6, 7] // 1, 2, 5 failed to re-judge
    const subset = select_calibration_subset(eight, {
      disagreed: [1],
      flipped: [2],
      reviewed: [1, 2],
      judged,
    })
    expect(subset).toHaveLength(2)
    for (const i of subset) expect(judged).toContain(i)
  })

  it("tops up fresh picks stratified across pass and fail", () => {
    const subset = select_calibration_subset(sixteen, {
      disagreed: [],
      flipped: [],
      reviewed: [0, 1],
      judged: all_judged,
    })
    expect(subset).toHaveLength(4)
    // Never-reviewed only.
    expect(subset).not.toContain(0)
    expect(subset).not.toContain(1)
    // Both verdict classes present (alternating scores make this checkable).
    const scores = subset.map((i) => sixteen[i].judge_score)
    expect(scores).toContain("fail")
    expect(scores).toContain("pass")
  })

  it("uses the same target math as the first round (floor(N/4), min 1)", () => {
    const three = score_traces(["pass", "fail", "pass"])
    const subset = select_calibration_subset(three, {
      disagreed: [],
      flipped: [],
      reviewed: [],
      judged: [0, 1, 2],
    })
    expect(subset).toHaveLength(1)
  })

  it("returns fewer than target when the eligible pool is smaller", () => {
    const subset = select_calibration_subset(sixteen, {
      disagreed: [2],
      flipped: [],
      reviewed: all_judged,
      judged: [2, 4],
    })
    // Target is 4 but only two traces re-judged, and every trace was
    // already reviewed — no fresh candidates exist.
    expect(subset).toEqual([2])
  })

  it("returns empty when no re-judged trace is eligible (the round-failure trigger)", () => {
    // Disagreed/flipped traces failed to re-judge and every fresh verdict
    // belongs to an already-reviewed trace. The wizard must treat this as a
    // failed round on the retryable surface — a 0-of-0 review would wedge
    // the save gate.
    const eight = score_traces(Array.from({ length: 8 }, () => "pass"))
    expect(
      select_calibration_subset(eight, {
        disagreed: [1],
        flipped: [2],
        reviewed: [0, 3, 4, 5, 6, 7],
        judged: [0, 3, 4],
      }),
    ).toEqual([])
  })
})

describe("reviewable_subset — what the reviewer is really shown", () => {
  // Claims state and the claims themselves are all this reads, so the
  // fixtures say just that. A "built" fixture carries a verdict claim, which
  // is the builder's contract; "no_verdict" is one that broke it.
  function states(...states: ("built" | "error" | "no_verdict")[]) {
    return states.map((state) => ({
      claims_state: state === "error" ? ("error" as const) : ("built" as const),
      claims:
        state === "error"
          ? null
          : state === "no_verdict"
            ? [claim()]
            : [claim(), claim({ is_verdict: true })],
    }))
  }

  it("drops the conversations whose claims never built", () => {
    // Ten selected out of forty, two of which failed to analyze.
    const traces = states(...Array<"built">(38).fill("built"), "error", "error")
    const selected = [0, 1, 2, 3, 4, 5, 6, 7, 38, 39]

    expect(reviewable_subset(traces, selected)).toEqual([
      0, 1, 2, 3, 4, 5, 6, 7,
    ])
  })

  it("keeps a fully built selection exactly as it was", () => {
    const traces = states("built", "built", "built")
    expect(reviewable_subset(traces, [0, 2])).toEqual([0, 2])
  })

  it("drops a built conversation whose claims carry no verdict", () => {
    // The builder writes the verdict as the last claim. A build that omits it
    // has claims to grade but nothing that records the pass/fail call, so the
    // reviewer is never asked to supply one; the count shrinks from that same
    // number the failed builds shrink it from.
    const traces = states("built", "no_verdict", "built", "no_verdict")
    const selected = [0, 1, 2, 3]

    expect(reviewable_subset(traces, selected)).toEqual([0, 2])
    expect(reviewable_subset(traces, selected)).toHaveLength(
      selected.length - 2,
    )
  })

  it("leaves a conversation whose claims carry a verdict untouched", () => {
    const traces = states("built", "built", "built")
    expect(reviewable_subset(traces, [0, 1, 2])).toEqual([0, 1, 2])
  })

  it("empties when nothing could be analyzed", () => {
    // The all-errored case: the review has nothing to show, and the page owes
    // the reviewer a message rather than an empty walk.
    expect(reviewable_subset(states("error", "error"), [0, 1])).toEqual([])
  })

  it("makes the gate, the header count and the review counter agree", () => {
    // Three numbers are drawn from these two values: the save gate's target,
    // the step header's "reviewing N of M", and the review's own "1 of N".
    const traces = states(...Array<"built">(38).fill("built"), "error", "error")
    const walked = reviewable_subset(traces, [0, 1, 2, 3, 4, 5, 6, 7, 38, 39])
    const target = calibration_gate_target(traces.length, walked.length)

    expect(target).toBe(walked.length)
    // Uncapped, the target is a pure function of the batch size, so it would
    // still demand ten reviews of a walk that only offers eight.
    expect(review_target(traces.length)).toBe(10)
    expect(target).toBe(8)
  })
})

describe("calibration_gate_target — save gate during rounds", () => {
  it("keeps the standard target when the round surfaced a full subset", () => {
    expect(calibration_gate_target(40, 10)).toBe(10)
  })

  it("caps the demand at the subset size on a re-judge shortfall", () => {
    // The gate must never demand reviews of traces the round didn't show —
    // otherwise a large shortfall makes the save unreachable.
    expect(calibration_gate_target(40, 3)).toBe(3)
    expect(calibration_gate_target(40, 1)).toBe(1)
  })

  it("keeps the floor-1 target when the subset covers it", () => {
    expect(calibration_gate_target(3, 3)).toBe(1)
  })

  it("is zero only for the empty subset the wizard never lets reach review", () => {
    expect(calibration_gate_target(40, 0)).toBe(0)
  })
})

describe("flipped_indices", () => {
  it("reports only re-judged cases whose verdict changed", () => {
    const traces = score_traces(["pass", "fail", "pass", "fail"])
    const results = new Map([
      [0, rejudge_result("fail")], // flip
      [1, rejudge_result("fail")], // unchanged
      [3, rejudge_result("pass")], // flip
      // 2 failed to re-judge: unknown, not unchanged — never a flip.
    ])
    expect(flipped_indices(traces, results)).toEqual([0, 3])
  })
})

describe("apply_rejudge_results", () => {
  it("folds fresh verdicts in and resets claims for rebuild", () => {
    const t = trace({ trace_id: "batch_case_0", leaf_run_id: "leaf_0" })
    const applied = apply_rejudge_results(
      [t],
      new Map([[0, rejudge_result("pass")]]),
      "batch_r1",
    )
    expect(applied[0].judge_score).toBe("pass")
    expect(applied[0].judge_reasoning).toBe("Re-checked: pass.")
    expect(applied[0].claims).toBeNull()
    expect(applied[0].overview).toBeNull()
    expect(applied[0].claims_state).toBe("unbuilt")
    // New per-round trace_id: an in-flight claim build from the previous
    // round must miss the identity guard, not corrupt the fresh state.
    expect(applied[0].trace_id).toBe("batch_r1_case_0")
    expect(applied[0].leaf_run_id).toBe("leaf_0")
  })

  it("leaves cases that failed to re-judge untouched", () => {
    const t = trace({ trace_id: "batch_case_0" })
    const applied = apply_rejudge_results([t], new Map(), "batch_r1")
    expect(applied[0]).toEqual(t)
  })

  it("grades reset: reviews rebuilt from the applied traces start empty", () => {
    const t = trace()
    const applied = apply_rejudge_results(
      [t],
      new Map([[0, rejudge_result("pass")]]),
      "batch_r1",
    )
    const reviews = build_trace_reviews(applied)
    expect(reviews[0].claim_verdicts).toHaveLength(0)
  })
})

describe("plan_save_action — loop entry and exit", () => {
  it("disagreement enters a calibration round", () => {
    expect(plan_save_action({ has_disagreement: true })).toEqual({
      action: "calibrate",
    })
  })

  it("converged reviews save (with the last refined judge)", () => {
    expect(plan_save_action({ has_disagreement: false })).toEqual({
      action: "save",
    })
  })

  // The disagreement flag is the ONLY input: both arms calibrate the same
  // way, and no round count caps the loop. Both properties are enforced by
  // the signature, so the two tests above are the whole surface; the only
  // other ways out are convergence and the save-without-refining link, which
  // bypasses this planner entirely.
})

// A graded trace's claims by grade, for the loop-entry predicates.
function graded(...grades: ("agree" | "disagree")[]) {
  return {
    claims: grades.map((human_grade) => ({
      text: "c",
      human_grade,
      human_feedback: null,
    })),
  }
}

describe("has_grade_disagreement / disagreed_trace_indices", () => {
  it("flags a disagreement on any claim", () => {
    expect(has_grade_disagreement([graded("agree", "agree")])).toBe(false)
    expect(has_grade_disagreement([graded("agree", "disagree")])).toBe(true)
  })

  it("finds trace indices carrying any explicit disagree verdict", () => {
    const agree: TraceReview = {
      trace_id: "t0",
      claim_verdicts: [{ agrees: true, why: "" }],
    }
    const claim_disagree: TraceReview = {
      trace_id: "t1",
      claim_verdicts: [
        { agrees: true, why: "" },
        { agrees: false, why: "off" },
      ],
    }
    const unreviewed: TraceReview = {
      trace_id: "t2",
      claim_verdicts: [{ agrees: null, why: "" }],
    }
    expect(
      disagreed_trace_indices([agree, claim_disagree, unreviewed]),
    ).toEqual([1])
  })
})

describe("rejudge_shortfall_notice", () => {
  it("silent when everything re-judged", () => {
    expect(rejudge_shortfall_notice(0, 20)).toBeNull()
  })

  it("counts the stale eval data against what was put up for re-checking", () => {
    // Counts rather than a noun: the same sentence is true whether the round
    // re-checked conversations or single test runs.
    expect(rejudge_shortfall_notice(1, 20)).toContain("1 of 20 couldn't be")
    expect(rejudge_shortfall_notice(3, 20)).toContain("3 of 20 couldn't be")
    expect(rejudge_shortfall_notice(3, 20)).not.toContain("conversation")
    expect(rejudge_shortfall_notice(3, 20)).not.toContain("test run")
    expect(rejudge_shortfall_notice(3, 20)).toContain(
      "left out of this review round",
    )
    expect(rejudge_shortfall_notice(3, 20)).not.toMatch(/—/)
  })
})

describe("grade_disagreement_count — the refine trigger as a count", () => {
  it("counts traces carrying a disagreement, matching the loop's predicate", () => {
    // Continue opens the improve-judge dialog exactly when the count is
    // non-zero — the same condition under which a round would start.
    expect(grade_disagreement_count([])).toBe(0)
    expect(grade_disagreement_count([graded("agree")])).toBe(0)
    const set = [
      graded("agree"),
      graded("disagree"),
      graded("agree", "agree", "disagree"),
    ]
    expect(grade_disagreement_count(set)).toBe(2)
  })

  it("flips back to zero the moment the last disagreement clears", () => {
    // Convergence signal: an all-agree set counts zero, so Continue stops
    // asking about the judge and saves.
    expect(
      grade_disagreement_count([graded("agree"), graded("agree", "agree")]),
    ).toBe(0)
  })
})

describe("tokenize_claim_text — inline [n] markers", () => {
  it("chips every marker with a citation and folds the rest into the text", () => {
    const citations = [
      { marker: 1, source: "output" as const, from: "a", to: "a" },
      { marker: 12, source: "input" as const, from: "b", to: "b" },
    ]
    expect(
      tokenize_claim_text(
        "[1] leads, item [3] is quoted, [12] ends",
        citations,
      ),
    ).toEqual([
      { kind: "cite", n: 1, citation: citations[0] },
      { kind: "text", value: " leads, item [3] is quoted, " },
      { kind: "cite", n: 12, citation: citations[1] },
      { kind: "text", value: " ends" },
    ])
  })

  it("keeps text without markers as one token", () => {
    expect(tokenize_claim_text("Plain.\nTwo lines.", [])).toEqual([
      { kind: "text", value: "Plain.\nTwo lines." },
    ])
  })
})

describe("split_claim_note — the trailing Note paragraph", () => {
  it("splits a Note paragraph off the claim body", () => {
    expect(
      split_claim_note(
        "The joke retells a known one [1]. We suggest 'Agree'.\n\nNote: the rubric never mentions originality.",
      ),
    ).toEqual({
      body: "The joke retells a known one [1]. We suggest 'Agree'.",
      note: "Note: the rubric never mentions originality.",
    })
  })

  it("keeps text with no Note paragraph whole, Note mid-sentence included", () => {
    const text = "Note that the reply cites [1]. Disagree if the note is wrong."
    expect(split_claim_note(text)).toEqual({ body: text, note: null })
  })
})

describe("flattener port — every block a message carries", () => {
  // The port recomputes the server's layout to place citations, so these pin
  // the cases where a message emits more than one block. Offsets are written
  // out longhand rather than computed, so a port change that shifts them fails
  // here instead of silently mis-placing a highlight.
  const CALL = {
    id: "c1",
    type: "function" as const,
    function: { name: "multiply", arguments: '{"a": 135.0, "b": 0.15}' },
  }

  it("maps a citation onto the tool call of a message that also has text", () => {
    const trace = [
      {
        role: "assistant",
        content: "Let me calculate that.",
        tool_calls: [CALL],
      },
    ] as unknown as TraceMessage[]
    const raw =
      "assistant:\n<assistant_message>\nLet me calculate that.\n</assistant_message>\n\n" +
      "assistant requested tool calls:\n<assistant_requested_tool_calls>\n" +
      '- Tool Name: multiply\n- Arguments: {"a": 135.0, "b": 0.15}\n' +
      "</assistant_requested_tool_calls>"
    const start = raw.indexOf('{"a": 135.0')
    const hit = map_output_span_to_trace(trace, raw, {
      start,
      end: start + '{"a": 135.0, "b": 0.15}'.length,
    })
    expect(hit).not.toBeNull()
    expect(hit?.kind).toBe("tool_calls")
    expect(hit?.trace_index).toBe(0)
  })

  it("maps a citation onto the text of that same message", () => {
    const trace = [
      {
        role: "assistant",
        content: "Let me calculate that.",
        tool_calls: [CALL],
      },
    ] as unknown as TraceMessage[]
    const raw =
      "assistant:\n<assistant_message>\nLet me calculate that.\n</assistant_message>\n\n" +
      "assistant requested tool calls:\n<assistant_requested_tool_calls>\n" +
      '- Tool Name: multiply\n- Arguments: {"a": 135.0, "b": 0.15}\n' +
      "</assistant_requested_tool_calls>"
    const start = raw.indexOf("Let me calculate")
    const hit = map_output_span_to_trace(trace, raw, {
      start,
      end: start + "Let me calculate that.".length,
    })
    expect(hit?.kind).toBe("content")
    expect(hit?.trace_index).toBe(0)
  })

  it("maps a citation onto a tool result whose label names its tool", () => {
    const trace = [
      { role: "assistant", content: null, tool_calls: [CALL] },
      { role: "tool", tool_call_id: "c1", content: "20.25" },
    ] as unknown as TraceMessage[]
    const raw =
      "assistant requested tool calls:\n<assistant_requested_tool_calls>\n" +
      '- Tool Name: multiply\n- Arguments: {"a": 135.0, "b": 0.15}\n' +
      "</assistant_requested_tool_calls>\n\n" +
      "tool result from multiply:\n<tool_tool_message>\n20.25\n</tool_tool_message>"
    const start = raw.lastIndexOf("20.25")
    const hit = map_output_span_to_trace(trace, raw, {
      start,
      end: start + "20.25".length,
    })
    expect(hit?.kind).toBe("tool_result")
    expect(hit?.trace_index).toBe(1)
  })

  it("draws nothing when the recomputed layout disagrees with raw_output", () => {
    const trace = [
      {
        role: "assistant",
        content: "Let me calculate that.",
        tool_calls: [CALL],
      },
    ] as unknown as TraceMessage[]
    // A transcript that does not match what this trace renders to.
    const raw =
      "assistant:\n<assistant_message>\nsomething else\n</assistant_message>"
    expect(
      map_output_span_to_trace(trace, raw, { start: 30, end: 39 }),
    ).toBeNull()
  })
})

describe("flattener port — the structured-output wrapper", () => {
  const TR = {
    id: "call_tr",
    type: "function" as const,
    function: {
      name: "task_response",
      arguments: '{"category": "refund_status"}',
    },
  }
  const REAL = {
    id: "c1",
    type: "function" as const,
    function: { name: "multiply", arguments: '{"a": 135, "b": 0.15}' },
  }

  it("maps a citation onto the structured answer, rendered as a message", () => {
    const trace = [
      { role: "assistant", content: null, tool_calls: [TR] },
    ] as unknown as TraceMessage[]
    const raw =
      'assistant:\n<assistant_message>\n{"category": "refund_status"}\n</assistant_message>'
    const start = raw.indexOf('"refund_status"')
    const hit = map_output_span_to_trace(trace, raw, {
      start,
      end: start + '"refund_status"'.length,
    })
    expect(hit?.kind).toBe("content")
    expect(hit?.trace_index).toBe(0)
  })

  it("keeps a real call listed beside the wrapper, and the wrapper out of it", () => {
    // The wrapper sitting before a real call is what makes this worth pinning:
    // if the port stopped excluding it, every offset after it would shift.
    const trace = [
      { role: "assistant", content: null, tool_calls: [REAL, TR] },
      { role: "user", content: "thanks" },
    ] as unknown as TraceMessage[]
    const raw =
      'assistant:\n<assistant_message>\n{"category": "refund_status"}\n</assistant_message>\n\n' +
      "assistant requested tool calls:\n<assistant_requested_tool_calls>\n" +
      '- Tool Name: multiply\n- Arguments: {"a": 135, "b": 0.15}\n' +
      "</assistant_requested_tool_calls>\n\n" +
      "user:\n<user_message>\nthanks\n</user_message>"
    const call_start = raw.indexOf('{"a": 135')
    const on_call = map_output_span_to_trace(trace, raw, {
      start: call_start,
      end: call_start + '{"a": 135, "b": 0.15}'.length,
    })
    expect(on_call?.kind).toBe("tool_calls")
    expect(on_call?.trace_index).toBe(0)

    // A later turn still resolves, which is what proves no extra block was
    // emitted for the wrapper and shifted everything after it.
    const later = raw.lastIndexOf("thanks")
    const on_later = map_output_span_to_trace(trace, raw, {
      start: later,
      end: later + "thanks".length,
    })
    expect(on_later?.kind).toBe("content")
    expect(on_later?.trace_index).toBe(1)
  })
})
