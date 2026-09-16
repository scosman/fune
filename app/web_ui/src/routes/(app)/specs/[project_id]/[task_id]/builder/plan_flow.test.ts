import { describe, expect, it } from "vitest"
import {
  clamp_turns_per_case,
  compact_batch_slots,
  dominant_failure_message,
  drive_cost_warning,
  drive_lanes_unchanged,
  drive_stop_banner,
  driven_data_confirm,
  first_preflight_failure,
  is_claims_resolved,
  is_partial_stop,
  missing_slot_indices,
  new_plan_confirm,
  plan_drive,
  resolved_selected_count,
  restore_turns_per_case,
  stop_primary_action,
  MAX_TURNS_PER_CASE,
  MIN_TURNS_PER_CASE,
  STOP_SCREEN_ERROR_EXCERPT_CHARS,
  type DriveStop,
} from "./plan_flow"
import type { ClaimsBuildState } from "./claim_evidence"

describe("dominant_failure_message", () => {
  it("returns null for no messages", () => {
    expect(dominant_failure_message([])).toBeNull()
  })

  it("ignores blank messages", () => {
    expect(dominant_failure_message(["", "", ""])).toBeNull()
  })

  it("returns the most frequent message", () => {
    expect(
      dominant_failure_message([
        "RateLimitError",
        "NotFoundError: model x",
        "RateLimitError",
      ]),
    ).toBe("RateLimitError")
  })

  it("resolves ties to the first seen", () => {
    expect(dominant_failure_message(["a", "b"])).toBe("a")
  })
})

describe("drive_stop_banner", () => {
  const partial: DriveStop = {
    survivors: 38,
    failed: 2,
    dominant_error: "RateLimitError from OpenRouter",
  }

  it("partial failure: the question first, then what was already tried", () => {
    const msg = drive_stop_banner(partial, "Polite Hawk")
    expect(msg).toBe(
      "2 of 40 failed. Continue with only 38, or run this batch again?\nEach failure was retried. Most common error: RateLimitError from OpenRouter.",
    )
  })

  it("partial failure: a dominant error ending in a period gets one period, not two", () => {
    const msg = drive_stop_banner(
      { ...partial, dominant_error: "Request timed out." },
      "Polite Hawk",
    )
    expect(msg.endsWith("Most common error: Request timed out.")).toBe(true)
  })

  it("partial failure without a dominant error omits that clause", () => {
    const msg = drive_stop_banner(
      { ...partial, dominant_error: null },
      "Polite Hawk",
    )
    expect(msg).toBe(
      "2 of 40 failed. Continue with only 38, or run this batch again?\nEach failure was retried.",
    )
  })

  it("all-failed: error color content — dominant error, run config name, and the /run deeplink", () => {
    const msg = drive_stop_banner(
      {
        survivors: 0,
        failed: 40,
        dominant_error: "NotFoundError: model gpt_5_5 is unavailable",
      },
      "Polite Hawk",
    )
    expect(msg).toBe(
      "All 40 failed: NotFoundError: model gpt_5_5 is unavailable (run config: Polite Hawk).\n\nYou can [test your run config](/run), then run the batch again.",
    )
  })

  it("all-failed without a run config name omits the clause", () => {
    const msg = drive_stop_banner(
      { survivors: 0, failed: 5, dominant_error: "boom" },
      null,
    )
    expect(msg).not.toContain("run config:")
    expect(msg).toContain("[test your run config](/run)")
  })

  it("abort with survivors: diagnosis with name+model, continue-or-test recovery", () => {
    const msg = drive_stop_banner(
      {
        survivors: 12,
        failed: 28,
        dominant_error: null,
        aborted_error: "AuthenticationError: invalid api key",
      },
      "Polite Hawk",
      "gpt_5_5",
    )
    expect(msg).toBe(
      "The run was stopped: AuthenticationError: invalid api key (run config: Polite Hawk, gpt_5_5).\n\n12 of 40 completed before the stop. Continue with those, or [test your run config](/run) and run the batch again.",
    )
  })

  it("abort with no survivors: test-then-drive recovery only", () => {
    const msg = drive_stop_banner(
      {
        survivors: 0,
        failed: 40,
        dominant_error: null,
        aborted_error: "NotFoundError: model gpt_5_5 is deprecated",
      },
      "Polite Hawk",
      "gpt_5_5",
    )
    expect(msg).toBe(
      "The run was stopped: NotFoundError: model gpt_5_5 is deprecated (run config: Polite Hawk, gpt_5_5).\n\nYou can [test your run config](/run), then run the batch again.",
    )
  })

  it("abort outranks the all-failed wording even at zero survivors", () => {
    const msg = drive_stop_banner(
      { survivors: 0, failed: 3, dominant_error: "x", aborted_error: "boom" },
      null,
    )
    expect(msg).toMatch(/^The run was stopped: boom\./)
  })
})

describe("driven_data_confirm", () => {
  it("has-data wording includes the review-progress clause", () => {
    expect(driven_data_confirm("New scenarios", 38, true)).toBe(
      "You have 38 completed eval inputs and your review progress. New scenarios will discard them. This cannot be undone.",
    )
  })

  it("stop-screen wording omits the review-progress clause", () => {
    expect(driven_data_confirm("Editing the scenarios", 38, false)).toBe(
      "You have 38 completed eval inputs. Editing the scenarios will discard them. This cannot be undone.",
    )
  })

  it("singular survivor", () => {
    expect(driven_data_confirm("New scenarios", 1, false)).toContain(
      "1 completed eval input.",
    )
  })
})

describe("new_plan_confirm", () => {
  it("pristine plan uses SDG's exact formula", () => {
    expect(
      new_plan_confirm({
        has_driven_results: false,
        survivors: 0,
        include_review_progress: false,
        plan_edited: false,
      }),
    ).toBe(
      "Are you sure you want to discard the current items? This cannot be undone.",
    )
  })

  it("edited plan uses SDG's removed-items variant", () => {
    expect(
      new_plan_confirm({
        has_driven_results: false,
        survivors: 0,
        include_review_progress: false,
        plan_edited: true,
      }),
    ).toBe(
      "Are you sure you want to discard the current items, including the ones you removed? This cannot be undone.",
    )
  })

  it("driven results outrank the edited tier", () => {
    expect(
      new_plan_confirm({
        has_driven_results: true,
        survivors: 40,
        include_review_progress: true,
        plan_edited: true,
      }),
    ).toBe(
      "You have 40 completed eval inputs and your review progress. A new batch plan will discard them. This cannot be undone.",
    )
  })

  it("names the action, not the plan's rows, in the driven tier", () => {
    // "New items will discard them" is broken grammar; the driven tier states
    // the action the user is about to take instead, so the caller's row noun
    // must not reach this sentence.
    const msg = new_plan_confirm({
      has_driven_results: true,
      survivors: 3,
      include_review_progress: false,
      plan_edited: false,
      plan_noun: "items",
    })
    expect(msg).toBe(
      "You have 3 completed eval inputs. A new batch plan will discard them. This cannot be undone.",
    )
  })
})

describe("claims gate resolution", () => {
  it("built and error are resolved; unbuilt and building are not", () => {
    expect(is_claims_resolved("built")).toBe(true)
    expect(is_claims_resolved("error")).toBe(true)
    expect(is_claims_resolved("unbuilt")).toBe(false)
    expect(is_claims_resolved("building")).toBe(false)
  })

  const traces = (states: ClaimsBuildState[]) =>
    states.map((claims_state) => ({ claims_state }))

  it("counts only the selected traces", () => {
    const t = traces(["built", "unbuilt", "error", "building", "built"])
    expect(resolved_selected_count(t, [0, 2, 3])).toBe(2)
    expect(resolved_selected_count(t, [1, 3])).toBe(0)
    expect(resolved_selected_count(t, [0, 2, 4])).toBe(3)
  })

  it("out-of-range indices never count as resolved", () => {
    expect(resolved_selected_count(traces(["built"]), [0, 7])).toBe(1)
  })

  it("empty selection resolves to zero", () => {
    expect(resolved_selected_count(traces(["built"]), [])).toBe(0)
  })
})

describe("first_preflight_failure", () => {
  it("returns null when every lane passed", () => {
    expect(
      first_preflight_failure([
        { lane: "run config", ok: true },
        { lane: "synthetic-user driver", ok: true },
        { lane: "judge", ok: true },
      ]),
    ).toBeNull()
  })

  it("picks the first failure in blame order, not race order", () => {
    const failure = first_preflight_failure([
      { lane: "run config", ok: false, message: "config dead" },
      { lane: "synthetic-user driver", ok: true },
      { lane: "judge", ok: false, message: "judge dead" },
    ])
    expect(failure?.lane).toBe("run config")
    expect(failure?.message).toBe("config dead")
  })

  it("falls back to a generic message when the lane gave none", () => {
    const failure = first_preflight_failure([
      { lane: "judge", ok: false, message: null },
    ])
    expect(failure?.message).toBe("the model did not respond")
    expect(failure?.model).toBeNull()
  })
})

describe("drive_stop_banner — preflight stop", () => {
  it("run-config lane names the config and deep-links /run", () => {
    const banner = drive_stop_banner(
      {
        survivors: 0,
        failed: 0,
        dominant_error: null,
        preflight: {
          lane: "run config",
          message: "AuthenticationError: invalid api key",
          model: null,
          provider: null,
        },
      },
      "Polite Hawk",
      "gpt_5_5",
    )
    expect(banner).toContain(
      "Could not create your eval data. Your run config failed a test call",
    )
    expect(banner).toContain("AuthenticationError: invalid api key")
    expect(banner).toContain("(run config: Polite Hawk, gpt_5_5)")
    expect(banner).toContain(
      "You can [test your run config](/run), then start again.",
    )
    expect(banner).toContain("[test your run config](/run)")
    // The recovery action is its own paragraph, never drowned in the error.
    expect(banner).toContain("\n\n")
  })

  it("judge lane names the model and points at Settings", () => {
    const banner = drive_stop_banner(
      {
        survivors: 0,
        failed: 0,
        dominant_error: null,
        preflight: {
          lane: "judge",
          message: "NotFoundError: model retired",
          model: "gpt_4o via openrouter",
          provider: "OpenRouter",
        },
      },
      "Polite Hawk",
      "gpt_5_5",
    )
    // Kiln-chosen lanes stay generic: neither the raw error nor the model
    // renders — the house generic, unadorned.
    // Kiln-chosen lanes show the raw error (the SDG precedent) plus the
    // requirement fact and the providers deeplink.
    expect(banner).toContain(
      "The judge model failed a test call: NotFoundError: model retired (gpt_4o via openrouter).",
    )
    expect(banner).toContain(
      "Creating your eval data requires your OpenRouter API key.",
    )
    expect(banner).toContain(
      "You can [check your model providers](/settings/providers), then try again.",
    )
    expect(banner).not.toContain("Polite Hawk")
  })

  it("preflight wins over the other banner variants", () => {
    // A preflight stop can coexist with survivors from a PREVIOUS drive —
    // the banner must report the check failure, not paid case counts.
    const banner = drive_stop_banner(
      {
        survivors: 12,
        failed: 0,
        dominant_error: "old error",
        preflight: {
          lane: "synthetic-user driver",
          message: "BudgetExceededError: quota",
          model: "claude_4_5_haiku via openrouter",
          provider: "OpenRouter",
        },
      },
      "Polite Hawk",
    )
    expect(banner).toContain("Could not create your eval data.")
    expect(banner).not.toContain("12 of")
    expect(banner).not.toContain("old error")
    expect(banner).toContain("BudgetExceededError")
  })

  it("input-generator lane names its role, model, and key requirement", () => {
    const banner = drive_stop_banner(
      {
        survivors: 0,
        failed: 0,
        dominant_error: null,
        preflight: {
          lane: "input generator",
          message: "NotFoundError: model retired",
          model: "gpt_5_4_mini via openrouter",
          provider: "OpenRouter",
        },
      },
      "Polite Hawk",
    )
    expect(banner).toContain(
      "The input generation model failed a test call: NotFoundError: model retired (gpt_5_4_mini via openrouter).",
    )
    expect(banner).toContain(
      "Creating your eval data requires your OpenRouter API key.",
    )
    expect(banner).toContain(
      "[check your model providers](/settings/providers)",
    )
  })
})

// One banner serves both arms. Naming the unit of work meant the sentence had
// to be told which arm it was on, and it read wrong whenever it was told
// nothing; counts alone are true either way.
describe("drive_stop_banner — counts, not nouns", () => {
  it("counts rather than naming what failed", () => {
    const banner = drive_stop_banner(
      {
        survivors: 38,
        failed: 2,
        dominant_error: "RateLimitError from OpenRouter",
      },
      "Polite Hawk",
    )
    expect(banner).toContain("2 of 40 failed.")
    expect(banner).toContain("Continue with only 38")
    expect(banner).not.toContain("conversation")
    expect(banner).not.toContain("test run")
  })

  it("counts in the all-failed and abort variants too", () => {
    const all_failed = drive_stop_banner(
      { survivors: 0, failed: 40, dominant_error: "boom" },
      "Polite Hawk",
    )
    expect(all_failed).toContain("All 40 failed: boom")
    const aborted = drive_stop_banner(
      {
        survivors: 12,
        failed: 28,
        dominant_error: null,
        aborted_error: "AuthenticationError: invalid api key",
      },
      "Polite Hawk",
      "gpt_5_5",
    )
    expect(aborted).toContain("12 of 40 completed before the stop.")
    expect(aborted).not.toContain("conversation")
    expect(aborted).not.toContain("test run")
  })
})

// The stop screen is a 300px column with its actions below the text, so the
// provider error it quotes has to be a short bounded excerpt: a raw message
// runs to hundreds of characters and pushes the actions off the screen.
describe("the partial stop's error excerpt", () => {
  const partial: DriveStop = {
    survivors: 38,
    failed: 2,
    dominant_error: "RateLimitError from OpenRouter",
  }
  const banner_of = (dominant_error: string): string =>
    drive_stop_banner({ ...partial, dominant_error }, null)
  // The clause the excerpt is quoted in, so each case reads as the sentence the
  // screen shows. It throws rather than returning "" when the clause is
  // missing: an absent clause is its own outcome, asserted on the whole banner.
  const excerpt_of = (dominant_error: string): string => {
    const msg = banner_of(dominant_error)
    const at = msg.indexOf("Most common error: ")
    if (at === -1) {
      throw new Error(`banner carries no error clause: ${msg}`)
    }
    return msg.slice(at)
  }

  it("trims the message and ends it with a single period, no gap", () => {
    expect(excerpt_of("   RateLimitError from OpenRouter")).toBe(
      "Most common error: RateLimitError from OpenRouter.",
    )
  })

  it("collapses several trailing periods to one", () => {
    expect(excerpt_of("Request timed out..")).toBe(
      "Most common error: Request timed out.",
    )
  })

  it("quotes only the first line of a multi-line message", () => {
    expect(
      excerpt_of(
        "RateLimitError from OpenRouter\nRetry after 30s\nrequest id 7",
      ),
    ).toBe("Most common error: RateLimitError from OpenRouter.")
  })

  it("skips a blank first line and quotes the first line with content", () => {
    expect(
      excerpt_of("\n   \nRateLimitError from OpenRouter\nRetry after 30s"),
    ).toBe("Most common error: RateLimitError from OpenRouter.")
  })

  it("splits on escaped newlines the way the renderer does", () => {
    expect(excerpt_of("RateLimitError from OpenRouter\\nRetry after 30s")).toBe(
      "Most common error: RateLimitError from OpenRouter.",
    )
  })

  it("cuts a long message at the bound and ends it with an ellipsis", () => {
    const excerpt = excerpt_of("x".repeat(400))
    expect(excerpt).toBe(`Most common error: ${"x".repeat(160)}…`)
    expect(excerpt.endsWith(".")).toBe(false)
  })

  it("bounds the excerpt at 160 characters", () => {
    // Pinned so the number in the case above stays honest: changing the bound
    // has to be a deliberate edit here, not a silently passing suite.
    expect(STOP_SCREEN_ERROR_EXCERPT_CHARS).toBe(160)
  })

  it("keeps a message that is exactly at the bound whole", () => {
    const at_bound = "x".repeat(160)
    expect(excerpt_of(at_bound)).toBe(`Most common error: ${at_bound}.`)
  })

  it("never cuts a surrogate pair in half", () => {
    // 201 UTF-16 units but only 101 code points, and the odd leading character
    // puts every pair on an odd offset: cutting by UTF-16 units would both
    // shorten a message that fits and leave half a character on screen.
    const astral = `x${"\u{1F642}".repeat(100)}`
    expect(astral.length).toBe(201)
    const excerpt = excerpt_of(astral)
    expect(excerpt).toBe(`Most common error: ${astral}.`)
    // The u flag is what makes this a well-formedness check: without it the
    // class matches the code units inside a perfectly good pair as well.
    expect([...excerpt].every((ch) => !/[\uD800-\uDFFF]/u.test(ch))).toBe(true)
  })

  it("adds no period after a message that asks a question", () => {
    expect(excerpt_of("Connection reset?")).toBe(
      "Most common error: Connection reset?",
    )
  })

  it("keeps a provider's own ellipsis and adds no period after it", () => {
    expect(excerpt_of("Request timed out…")).toBe(
      "Most common error: Request timed out…",
    )
  })

  it("drops the clause when the message is nothing but punctuation", () => {
    // Stripping the trailing periods leaves nothing to quote, and half a
    // sentence reads as a bug on the screen.
    expect(banner_of("...").endsWith("Each failure was retried.")).toBe(true)
    expect(banner_of(" . . . ").endsWith("Each failure was retried.")).toBe(
      true,
    )
  })
})

// Which of the stop screen's two actions leads. Continuing is only worth
// leading with when almost everything survived; otherwise the batch is worth
// running again.
describe("stop_primary_action", () => {
  const stop = (survivors: number, failed: number): DriveStop => ({
    survivors,
    failed,
    dominant_error: null,
  })

  it("leads with continuing when almost everything survived", () => {
    expect(stop_primary_action(stop(19, 1))).toBe("continue")
  })

  it("leads with re-running when enough was lost to be worth redoing", () => {
    expect(stop_primary_action(stop(8, 2))).toBe("rerun")
    expect(stop_primary_action(stop(89, 11))).toBe("rerun")
  })

  it("leads with re-running when nothing survived", () => {
    expect(stop_primary_action(stop(0, 10))).toBe("rerun")
  })

  it("treats exactly the threshold as good enough to continue", () => {
    expect(stop_primary_action(stop(9, 1))).toBe("continue")
    expect(stop_primary_action(stop(90, 10))).toBe("continue")
  })

  it("leads with re-running when there was nothing to run", () => {
    // A zero total is a stop before anything started, so there are no
    // survivors to continue with and the ratio is undefined.
    expect(stop_primary_action(stop(0, 0))).toBe("rerun")
  })

  it("leads with re-running when the stop was not about how much survived", () => {
    // A preflight stop never drove the batch, and an abort cut it short on a
    // config fault, so whatever count came back is not the thing to decide on.
    expect(
      stop_primary_action({
        ...stop(19, 1),
        preflight: {
          lane: "judge",
          message: "model retired",
          model: "gpt_5_4_mini",
          provider: "openrouter",
        },
      }),
    ).toBe("rerun")
    expect(
      stop_primary_action({
        ...stop(19, 1),
        aborted_error: "AuthenticationError: invalid api key",
      }),
    ).toBe("rerun")
  })
})

// Which stop kinds the centred screen is for: only the ones that can be told
// in one counted sentence, with usable work left behind.
describe("is_partial_stop", () => {
  const stop = (extra: Partial<DriveStop> = {}): DriveStop => ({
    survivors: 38,
    failed: 2,
    dominant_error: null,
    ...extra,
  })

  it("is a partial stop when cases failed and the batch itself did not", () => {
    expect(is_partial_stop(stop())).toBe(true)
  })

  it("is not a partial stop when nothing survived", () => {
    expect(is_partial_stop(stop({ survivors: 0, failed: 40 }))).toBe(false)
  })

  it("is not a partial stop when the batch was aborted", () => {
    expect(is_partial_stop(stop({ aborted_error: "boom" }))).toBe(false)
  })

  it("is not a partial stop when a lane failed before the drive", () => {
    expect(
      is_partial_stop(
        stop({
          preflight: {
            lane: "judge",
            message: "model retired",
            model: null,
            provider: null,
          },
        }),
      ),
    ).toBe(false)
  })
})

describe("plan_drive — top-off vs fresh batch", () => {
  const cases = ["c0", "c1", "c2", "c3"]
  // Slots 1 and 3 failed; 0 and 2 are paid successes.
  const partial = ["r0", null, "r2", null]

  it("tops off: drives only the missing slots into the same batch tag", () => {
    // The bug this pins: a retry after a partial drive used to re-drive
    // EVERY case under a fresh tag, deleting the previous batch — the paid
    // successes were re-billed and any run outside the one tag was
    // invisible to review, counts, and save.
    const plan = plan_drive({
      items: cases,
      batch_items: cases,
      built_slots: partial,
      batch_tag: "batchA",
      undeleted_batch_tags: ["batchA", "stale1"],
    })
    expect(plan.top_off).toBe(true)
    expect(plan.items).toEqual(["c1", "c3"])
    expect(plan.slot_of_stream_index).toEqual([1, 3])
    expect(plan.batch_tag).toBe("batchA")
  })

  it("never lists its own batch in replace_batch_tags", () => {
    // Deleting the batch being topped off would destroy the paid
    // successes being kept; stale tags from earlier aborted drives still
    // get cleaned.
    const plan = plan_drive({
      items: cases,
      batch_items: cases,
      built_slots: partial,
      batch_tag: "batchA",
      undeleted_batch_tags: ["batchA", "stale1"],
    })
    expect(plan.replace_batch_tags).toEqual(["stale1"])
  })

  it("recurses: a partially-failed top-off plans the still-missing slots", () => {
    const after_first_top_off = ["r0", "r1", "r2", null]
    const plan = plan_drive({
      items: cases,
      batch_items: cases,
      built_slots: after_first_top_off,
      batch_tag: "batchA",
      undeleted_batch_tags: ["batchA"],
    })
    expect(plan.top_off).toBe(true)
    expect(plan.items).toEqual(["c3"])
    expect(plan.slot_of_stream_index).toEqual([3])
  })

  it("goes fresh when the items changed", () => {
    // Changed plan or spec resolves different cases; mixing them into the
    // old batch would put results the user never planned together under
    // one tag.
    const plan = plan_drive({
      items: ["c0", "c1", "c2", "different"],
      batch_items: cases,
      built_slots: partial,
      batch_tag: "batchA",
      undeleted_batch_tags: ["batchA"],
    })
    expect(plan.top_off).toBe(false)
    expect(plan.items).toEqual(["c0", "c1", "c2", "different"])
    expect(plan.slot_of_stream_index).toEqual([0, 1, 2, 3])
    expect(plan.batch_tag).toBeNull()
    expect(plan.replace_batch_tags).toEqual(["batchA"])
  })

  it("goes fresh when there is no batch to top off", () => {
    const plan = plan_drive({
      items: cases,
      batch_items: null,
      built_slots: [],
      batch_tag: null,
      undeleted_batch_tags: [],
    })
    expect(plan.top_off).toBe(false)
    expect(plan.items).toEqual(cases)
  })

  it("goes fresh when nothing succeeded", () => {
    // With zero paid successes there is nothing to keep; replace semantics
    // clean up the failed batch's tag.
    const plan = plan_drive({
      items: cases,
      batch_items: cases,
      built_slots: [null, null, null, null],
      batch_tag: "batchA",
      undeleted_batch_tags: ["batchA"],
    })
    expect(plan.top_off).toBe(false)
    expect(plan.replace_batch_tags).toEqual(["batchA"])
  })

  it("goes fresh when nothing is missing", () => {
    const plan = plan_drive({
      items: cases,
      batch_items: cases,
      built_slots: ["r0", "r1", "r2", "r3"],
      batch_tag: "batchA",
      undeleted_batch_tags: ["batchA"],
    })
    expect(plan.top_off).toBe(false)
  })
})

describe("missing_slot_indices", () => {
  it("returns the null slots in order", () => {
    expect(missing_slot_indices(["a", null, "b", null])).toEqual([1, 3])
    expect(missing_slot_indices([])).toEqual([])
    expect(missing_slot_indices([null])).toEqual([0])
  })
})

describe("plan_drive — slot/item length agreement", () => {
  it("goes fresh when the slots and batch items diverge in length", () => {
    // A divergence would map missing slots onto the wrong (or no) items;
    // fresh replace semantics are the only safe plan.
    const plan = plan_drive({
      items: ["c0", "c1", "c2"],
      batch_items: ["c0", "c1", "c2"],
      built_slots: ["r0", null],
      batch_tag: "batchA",
      undeleted_batch_tags: ["batchA"],
    })
    expect(plan.top_off).toBe(false)
    expect(plan.items).toEqual(["c0", "c1", "c2"])
  })
})

describe("clamp_turns_per_case", () => {
  it("mirrors the drive route's accepted range", () => {
    // multiturn_sdg_api declares turns as ge=1, le=20. A stepper that could
    // exceed either end would compose a request that can only 422.
    expect(MIN_TURNS_PER_CASE).toBe(1)
    expect(MAX_TURNS_PER_CASE).toBe(20)
  })

  it("passes an in-range value through untouched", () => {
    // The default (5) above all: an untouched knob must drive exactly what the
    // builder drove before the knob existed.
    expect(clamp_turns_per_case(5)).toBe(5)
    expect(clamp_turns_per_case(1)).toBe(1)
    expect(clamp_turns_per_case(20)).toBe(20)
    expect(clamp_turns_per_case(12)).toBe(12)
  })

  it("clamps to both ends of the range", () => {
    expect(clamp_turns_per_case(0)).toBe(1)
    expect(clamp_turns_per_case(-4)).toBe(1)
    expect(clamp_turns_per_case(21)).toBe(20)
    expect(clamp_turns_per_case(500)).toBe(20)
  })

  it("rounds a fractional value to a whole turn", () => {
    expect(clamp_turns_per_case(3.4)).toBe(3)
    expect(clamp_turns_per_case(3.6)).toBe(4)
  })

  it("falls back to the minimum for a non-numeric value", () => {
    // A corrupt or hand-edited draft can carry anything; the drive still has
    // to send a number the route accepts.
    expect(clamp_turns_per_case(NaN)).toBe(1)
    expect(clamp_turns_per_case(Infinity)).toBe(1)
    expect(clamp_turns_per_case(undefined as unknown as number)).toBe(1)
  })
})

describe("restore_turns_per_case", () => {
  // The builder's page default; passed in so the helper stays free of the
  // page's constants.
  const PAGE_DEFAULT = 5

  it("clamps a genuine number", () => {
    expect(restore_turns_per_case(8, PAGE_DEFAULT)).toBe(8)
    expect(restore_turns_per_case(99, PAGE_DEFAULT)).toBe(20)
    expect(restore_turns_per_case(0, PAGE_DEFAULT)).toBe(1)
  })

  it("restores the default when no choice is on record", () => {
    expect(restore_turns_per_case(null, PAGE_DEFAULT)).toBe(PAGE_DEFAULT)
    expect(restore_turns_per_case(undefined, PAGE_DEFAULT)).toBe(PAGE_DEFAULT)
  })

  it("restores the default for a value that is not a finite number", () => {
    // The clamp's minimum is for numbers that fell out of range, not for
    // garbage: a corrupt draft must not hand the user a one-turn run they
    // never picked.
    expect(restore_turns_per_case(NaN, PAGE_DEFAULT)).toBe(PAGE_DEFAULT)
    expect(restore_turns_per_case(Infinity, PAGE_DEFAULT)).toBe(PAGE_DEFAULT)
    expect(restore_turns_per_case("8" as unknown as number, PAGE_DEFAULT)).toBe(
      PAGE_DEFAULT,
    )
  })
})

describe("drive_lanes_unchanged", () => {
  const judge = { prompt: "p", model_name: "m", model_provider: "openai" }

  it("is unchanged when the judge matches (single-turn shape, no su)", () => {
    expect(drive_lanes_unchanged({ judge, batch_judge: { ...judge } })).toBe(
      true,
    )
  })

  it("a changed judge forces a fresh batch", () => {
    // One review may not mix two judges' verdicts.
    expect(
      drive_lanes_unchanged({
        judge,
        batch_judge: { ...judge, model_name: "other" },
      }),
    ).toBe(false)
    expect(
      drive_lanes_unchanged({
        judge,
        batch_judge: { ...judge, prompt: "refined" },
      }),
    ).toBe(false)
  })

  it("no batch judge means no batch to reuse", () => {
    expect(drive_lanes_unchanged({ judge, batch_judge: null })).toBe(false)
  })

  it("a changed synthetic-user model forces a fresh batch", () => {
    // The saved drive stamp must describe every conversation in the batch.
    const su = { model_name: "su", model_provider: "openai" }
    expect(
      drive_lanes_unchanged({
        judge,
        batch_judge: { ...judge },
        su,
        batch_su: { ...su },
      }),
    ).toBe(true)
    expect(
      drive_lanes_unchanged({
        judge,
        batch_judge: { ...judge },
        su,
        batch_su: { ...su, model_name: "different" },
      }),
    ).toBe(false)
    expect(
      drive_lanes_unchanged({
        judge,
        batch_judge: { ...judge },
        su,
        batch_su: null,
      }),
    ).toBe(false)
  })

  it("a changed conversation length forces a fresh batch", () => {
    // A top-off at a different length would put 5-turn and 10-turn
    // conversations in one batch under a stamp that can name only one length.
    const su = { model_name: "su", model_provider: "openai" }
    const same_lanes = { judge, batch_judge: { ...judge }, su, batch_su: su }
    expect(
      drive_lanes_unchanged({ ...same_lanes, turns: 5, batch_turns: 5 }),
    ).toBe(true)
    expect(
      drive_lanes_unchanged({ ...same_lanes, turns: 10, batch_turns: 5 }),
    ).toBe(false)
    expect(
      drive_lanes_unchanged({ ...same_lanes, turns: 5, batch_turns: 10 }),
    ).toBe(false)
  })

  it("no recorded batch length means no batch to top off", () => {
    const su = { model_name: "su", model_provider: "openai" }
    expect(
      drive_lanes_unchanged({
        judge,
        batch_judge: { ...judge },
        su,
        batch_su: su,
        turns: 5,
        batch_turns: null,
      }),
    ).toBe(false)
  })
})

describe("compact_batch_slots", () => {
  const t = (trace_id: string, extra = "") => ({ trace_id, extra })

  it("compacts non-null slots in order", () => {
    expect(compact_batch_slots([t("a"), null, t("b")], [])).toEqual([
      t("a"),
      t("b"),
    ])
  })

  it("prefers the live review entry for the same trace", () => {
    // Claims built (or any later enrichment of) a kept case must survive a
    // top-off's compaction; the slots hold the drive-time copies.
    const enriched = t("a", "claims built")
    expect(compact_batch_slots([t("a"), null, t("b")], [enriched])).toEqual([
      enriched,
      t("b"),
    ])
  })
})

describe("drive_cost_warning", () => {
  it("states the conversation count and the turn ceiling on the multi-turn arm", () => {
    expect(
      drive_cost_warning({
        is_multi_turn: true,
        count: 40,
        turns_per_case: 5,
      }),
    ).toBe(
      "This will run 40 conversations of up to 5 turns each. Running multi-turn data generation and evaluation can be expensive. Calculate your costs before proceeding.",
    )
  })

  it("counts items on the single-turn arm", () => {
    expect(
      drive_cost_warning({
        is_multi_turn: false,
        count: 40,
        turns_per_case: 5,
      }),
    ).toBe(
      "This will run your task on 40 items and may use considerable credits.",
    )
  })

  it("says one item, not 1 items", () => {
    // A one-row plan is reachable: the user can delete rows down to one.
    expect(
      drive_cost_warning({
        is_multi_turn: false,
        count: 1,
        turns_per_case: 5,
      }),
    ).toBe(
      "This will run your task on 1 item and may use considerable credits.",
    )
  })

  it("ignores turns_per_case on the single-turn arm", () => {
    // Single-turn runs the task once per input; a turns value must not leak
    // into its wording or its arithmetic.
    const a = drive_cost_warning({
      is_multi_turn: false,
      count: 12,
      turns_per_case: 5,
    })
    const b = drive_cost_warning({
      is_multi_turn: false,
      count: 12,
      turns_per_case: 9,
    })
    expect(a).toBe(b)
  })

  it("restates a changed turns_per_case", () => {
    expect(
      drive_cost_warning({ is_multi_turn: true, count: 7, turns_per_case: 3 }),
    ).toBe(
      "This will run 7 conversations of up to 3 turns each. Running multi-turn data generation and evaluation can be expensive. Calculate your costs before proceeding.",
    )
  })

  it("says one conversation of one turn, not 1 conversations of 1 turns", () => {
    expect(
      drive_cost_warning({ is_multi_turn: true, count: 1, turns_per_case: 1 }),
    ).toBe(
      "This will run 1 conversation of up to 1 turn each. Running multi-turn data generation and evaluation can be expensive. Calculate your costs before proceeding.",
    )
  })
})
