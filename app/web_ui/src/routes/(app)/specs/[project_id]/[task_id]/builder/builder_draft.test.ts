import { afterEach, describe, expect, it } from "vitest"
import {
  builder_draft_key,
  builder_mock_active,
  create_eval_button_label,
  create_eval_destination,
  draft_after_save_keeping_stranded_tags,
  draft_has_content,
  draft_is_resumable,
  questions_are_current,
  reset_draft_keeping_tags,
  restore_step,
  reusable_cached_cases,
  reusable_minted_inputs,
  run_config_cache_key,
  should_invalidate_refined_values,
  should_prefill_suggested_name,
  EMPTY_BUILDER_DRAFT,
  type BuilderDraft,
  type CachedSuCases,
} from "./builder_draft"
import type { KilnAgentRunConfigProperties } from "$lib/types"
import { restore_turns_per_case } from "./plan_flow"

// The input generator's committed run config, as the component hands it over.
const INPUT_GEN_RUN_CONFIG: KilnAgentRunConfigProperties = {
  type: "kiln_agent",
  model_name: "gpt_5_4_mini",
  model_provider_name: "openai",
  prompt_id: "simple_prompt_builder",
  temperature: 1.0,
  top_p: 1.0,
  structured_output_mode: "json_schema",
  thinking_level: "medium",
  input_transform: null,
  tools_config: { tools: ["kiln_tool::search", "skill::triage"] },
}

// A draft with every field populated — the round-trip and resolution
// fixtures below carve it down.
const full_draft: BuilderDraft = {
  description: "The agent must not fabricate policies.",
  // Deliberately NOT equal to description: the live text drifts past the last
  // Continue whenever the user edits without continuing. The round trip below
  // only serializes, so it can't catch the two being crossed — distinctness
  // keeps the fixture discriminating for any future test that drives the real
  // save and restore paths.
  continued_description: "The agent must not fabricate refund policies.",
  name: "no-fabrication",
  // Deliberately NOT equal to name: this fixture is a user who renamed. Same
  // reason as continued_description above — distinctness is what would let a
  // test of the real save/restore paths catch the two fields being crossed.
  prefilled_name: "no-fabricated-policies",
  property_values: {
    issue_description: "The agent must not fabricate policies.",
    issue_examples: "Invented a 90-day return window.",
    non_issue_examples: null,
  },
  refined_property_values: {
    issue_description: "The agent must not fabricate or guess at policies.",
  },
  suggested_edits: {
    issue_description: {
      proposed_value: "The agent must not fabricate or guess at policies.",
      reason_for_edit: "Broadened to cover guessing.",
    },
  },
  batch_plan: {
    prompts: ["Customer asks about a return window.", "Warranty question."],
    summary: "Two fabrication-bait scenarios.",
  },
  batch_plan_edited: true,
  cached_su_cases: {
    prompts_json: JSON.stringify([
      "Customer asks about a return window.",
      "Warranty question.",
    ]),
    spec_text: "The agent must not fabricate or guess at policies.",
    cases: [
      {
        seed_prompt: "What's your return window?",
        synthetic_user_info: "<persona>impatient</persona>",
        scenario_index: 0,
      },
      {
        seed_prompt: "Is my laptop still under warranty?",
        synthetic_user_info: "<persona>polite</persona>",
        scenario_index: 1,
      },
    ],
  },
  cached_minted_inputs: {
    prompts_json: JSON.stringify([
      "Customer asks about a return window.",
      "Warranty question.",
    ]),
    run_config_json: run_config_cache_key(INPUT_GEN_RUN_CONFIG),
    inputs: ["What's your return window?", "Is my laptop under warranty?"],
  },
  grounding_sample: {
    input: "What's your return policy on opened electronics?",
    output: "Opened electronics can be returned within 14 days.",
  },
  data_guide_text: "# Reference Inputs\n\nShort support questions.",
  use_data_guide: true,
  data_guide_skipped: false,
  data_guide_offer_pending: false,
  multi_turn_batch_tag: "multi_turn_batch_1234",
  single_turn_batch_tag: "single_turn_batch_5678",
  undeleted_batch_tags: ["multi_turn_batch_1200", "multi_turn_batch_1234"],
  su_driver: {
    model_name: "gpt_5_4_mini",
    model_provider: "openai",
  },
  input_generator: {
    model_name: "gpt_5_4_mini",
    model_provider: "openai",
  },
  input_gen_run_config: INPUT_GEN_RUN_CONFIG,
  judge_model: {
    model_name: "gpt_5_4",
    model_provider: "openai",
  },
  // Deliberately NOT the default 5: a fixture on the default couldn't tell a
  // restored choice from the fallback.
  turns_per_case: 8,
  target_run_config_id: "run_config_1",
}

describe("draft round-trip", () => {
  it("survives serialization with every field intact", () => {
    // IndexedDB structured-clones the draft; JSON is a strictly harsher
    // proxy (drops functions/undefined), so surviving it guarantees the
    // stored shape restores exactly.
    const restored = JSON.parse(JSON.stringify(full_draft)) as BuilderDraft
    expect(restored).toEqual(full_draft)
    expect(restore_step(restored)).toBe(restore_step(full_draft))
  })

  it("empty draft round-trips to no-content", () => {
    const restored = JSON.parse(
      JSON.stringify(EMPTY_BUILDER_DRAFT),
    ) as BuilderDraft
    expect(restored).toEqual(EMPTY_BUILDER_DRAFT)
    expect(draft_has_content(restored)).toBe(false)
  })

  it("keys drafts per task", () => {
    expect(builder_draft_key("p1", "t1")).not.toBe(
      builder_draft_key("p1", "t2"),
    )
    expect(builder_draft_key("p1", "t1")).toBe(builder_draft_key("p1", "t1"))
  })
})

describe("restore_step resolution", () => {
  it("empty draft restores to describe", () => {
    expect(restore_step(EMPTY_BUILDER_DRAFT)).toBe("describe")
  })

  it("description alone restores to describe (clarify is never a target)", () => {
    expect(
      restore_step({
        ...EMPTY_BUILDER_DRAFT,
        description: "Some spec",
        property_values: { issue_description: "Some spec" },
      }),
    ).toBe("describe")
  })

  it("refined content restores to refine", () => {
    expect(
      restore_step({
        ...EMPTY_BUILDER_DRAFT,
        description: "Some spec",
        refined_property_values: { issue_description: "Refined spec" },
      }),
    ).toBe("refine")
  })

  it("suggested edits alone restore to refine", () => {
    expect(
      restore_step({
        ...EMPTY_BUILDER_DRAFT,
        suggested_edits: {
          issue_description: { proposed_value: "x", reason_for_edit: "" },
        },
      }),
    ).toBe("refine")
  })

  it("whitespace-only refined values do not count as refine content", () => {
    expect(
      restore_step({
        ...EMPTY_BUILDER_DRAFT,
        description: "Some spec",
        refined_property_values: { issue_description: "   " },
      }),
    ).toBe("describe")
  })

  it("a plan restores to the plan screen", () => {
    expect(restore_step(full_draft)).toBe("generate")
  })

  it("never restores past step 4", () => {
    // Even a draft that carries batch tags (a drive happened) resolves to
    // the plan screen at furthest — review state is never persisted.
    expect(["describe", "refine", "generate"]).toContain(
      restore_step(full_draft),
    )
  })

  it("an open Data Guide offer restores to the plan screen with no plan", () => {
    expect(
      restore_step({
        ...EMPTY_BUILDER_DRAFT,
        description: "d",
        refined_property_values: { issue_description: "x" },
        data_guide_offer_pending: true,
      }),
    ).toBe("generate")
  })

  it("a skipped guide alone does not restore past refine — the plan fires from Continue", () => {
    expect(
      restore_step({
        ...EMPTY_BUILDER_DRAFT,
        refined_property_values: { issue_description: "x" },
        data_guide_skipped: true,
      }),
    ).toBe("refine")
  })

  it("the Data Guide fields survive a round trip on a pending draft", () => {
    const pending: BuilderDraft = {
      ...full_draft,
      batch_plan: null,
      data_guide_text: null,
      use_data_guide: false,
      data_guide_skipped: true,
      data_guide_offer_pending: true,
    }
    const restored = JSON.parse(JSON.stringify(pending)) as BuilderDraft
    expect(restored.data_guide_skipped).toBe(true)
    expect(restored.data_guide_offer_pending).toBe(true)
    expect(restored.data_guide_text).toBeNull()
    expect(restore_step(restored)).toBe("generate")
  })

  it("a plan with no prompts falls back to the earlier steps", () => {
    expect(
      restore_step({
        ...full_draft,
        batch_plan: { prompts: [], summary: "" },
      }),
    ).toBe("refine")
  })
})

describe("draft_has_content", () => {
  it("is false for the empty draft", () => {
    expect(draft_has_content(EMPTY_BUILDER_DRAFT)).toBe(false)
  })

  it("is false for whitespace-only fields", () => {
    expect(
      draft_has_content({
        ...EMPTY_BUILDER_DRAFT,
        description: "   ",
        property_values: { issue_description: "  ", other: null },
      }),
    ).toBe(false)
  })

  it("is true for a description", () => {
    expect(
      draft_has_content({ ...EMPTY_BUILDER_DRAFT, description: "spec" }),
    ).toBe(true)
  })

  it("is true for batch tags alone — the orphaned-chain correctness carry", () => {
    // Tags name chains already on disk; a draft carrying only them must
    // still restore so the next drive can clean those chains up.
    expect(
      draft_has_content({
        ...EMPTY_BUILDER_DRAFT,
        undeleted_batch_tags: ["multi_turn_batch_1200"],
      }),
    ).toBe(true)
    expect(
      draft_has_content({
        ...EMPTY_BUILDER_DRAFT,
        multi_turn_batch_tag: "multi_turn_batch_1234",
      }),
    ).toBe(true)
    expect(
      draft_has_content({
        ...EMPTY_BUILDER_DRAFT,
        single_turn_batch_tag: "single_turn_batch_5678",
      }),
    ).toBe(true)
  })

  it("is true for a plan", () => {
    expect(
      draft_has_content({
        ...EMPTY_BUILDER_DRAFT,
        batch_plan: { prompts: ["p"], summary: "s" },
      }),
    ).toBe(true)
  })
})

describe("builder_mock_active — the mock gate", () => {
  afterEach(() => {
    delete (globalThis as { window?: unknown }).window
  })

  it("is false when no window exists (SSR / plain node)", () => {
    expect(builder_mock_active()).toBe(false)
  })

  it("is false in a browser without the mock installed", () => {
    ;(globalThis as { window?: unknown }).window = {}
    expect(builder_mock_active()).toBe(false)
  })

  it("is true once the mock marks the window", () => {
    ;(globalThis as { window?: unknown }).window = {
      __KILN_BUILDER_MOCK_ACTIVE__: true,
    }
    expect(builder_mock_active()).toBe(true)
  })
})

describe("reusable_cached_cases — SU-case reuse", () => {
  const prompts = ["scenario a", "scenario b"]
  const spec = "no fabrication"
  const cache: CachedSuCases = {
    prompts_json: JSON.stringify(prompts),
    spec_text: spec,
    cases: [
      { seed_prompt: "hi", synthetic_user_info: "<persona>x</persona>" },
      { seed_prompt: "yo", synthetic_user_info: "<persona>y</persona>" },
    ],
  }

  it("reuses when plan and spec are byte-unchanged", () => {
    expect(
      reusable_cached_cases(cache, ["scenario a", "scenario b"], spec),
    ).toBe(cache.cases)
  })

  it("misses with no cache", () => {
    expect(reusable_cached_cases(null, prompts, spec)).toBeNull()
  })

  it("misses when a prompt was edited", () => {
    expect(
      reusable_cached_cases(cache, ["scenario a EDITED", "scenario b"], spec),
    ).toBeNull()
  })

  it("misses when a prompt was deleted", () => {
    expect(reusable_cached_cases(cache, ["scenario a"], spec)).toBeNull()
  })

  it("misses when the spec text changed", () => {
    expect(reusable_cached_cases(cache, prompts, "different spec")).toBeNull()
  })

  it("misses on an empty cached case list", () => {
    expect(
      reusable_cached_cases({ ...cache, cases: [] }, prompts, spec),
    ).toBeNull()
  })

  it("prompt order matters (case i maps to prompt i)", () => {
    expect(
      reusable_cached_cases(cache, ["scenario b", "scenario a"], spec),
    ).toBeNull()
  })
})

describe("questions_are_current — clarify staleness gate", () => {
  const description = "The agent must not fabricate policies."

  // An aborted or failed request must leave the source unset: recording one
  // for a set that never landed would let stale questions read as current.
  it("is false with no source — nothing has landed yet", () => {
    expect(questions_are_current(null, description)).toBe(false)
  })

  it("is true for an empty description once an empty set landed", () => {
    // The null guard means "no set has landed yet", not "the text is empty".
    expect(questions_are_current("", "")).toBe(true)
  })

  it("is true when the description is unchanged", () => {
    // A no-op Back-then-forward must not burn another paid copilot call.
    expect(questions_are_current(description, description)).toBe(true)
  })

  it("is false after a one-byte edit", () => {
    expect(
      questions_are_current(
        description,
        "The agent must not fabricate policy.",
      ),
    ).toBe(false)
  })

  it("is false for a whitespace-only difference — raw byte compare", () => {
    expect(questions_are_current(description, description + "\n")).toBe(false)
    expect(questions_are_current(description + " ", description)).toBe(false)
  })
})

describe("reset_draft_keeping_tags", () => {
  it("wipes all authoring state but carries the batch tags", () => {
    const fresh = reset_draft_keeping_tags(full_draft)
    expect(fresh.multi_turn_batch_tag).toBe("multi_turn_batch_1234")
    expect(fresh.single_turn_batch_tag).toBe("single_turn_batch_5678")
    expect(fresh.undeleted_batch_tags).toEqual([
      "multi_turn_batch_1200",
      "multi_turn_batch_1234",
    ])
    expect(fresh.description).toBe("")
    expect(fresh.name).toBe("")
    expect(fresh.batch_plan).toBeNull()
    expect(fresh.cached_su_cases).toBeNull()
    expect(fresh.cached_minted_inputs).toBeNull()
    expect(fresh.refined_property_values).toEqual({})
  })

  it("resolves to the describe step after reset", () => {
    expect(restore_step(reset_draft_keeping_tags(full_draft))).toBe("describe")
  })

  it("still counts as content when tags exist, so cleanup survives another restore", () => {
    expect(draft_has_content(reset_draft_keeping_tags(full_draft))).toBe(true)
  })

  it("is a full empty draft when there were no tags", () => {
    const no_tags = {
      ...full_draft,
      multi_turn_batch_tag: null,
      single_turn_batch_tag: null,
      undeleted_batch_tags: [],
    }
    expect(reset_draft_keeping_tags(no_tags)).toEqual(EMPTY_BUILDER_DRAFT)
  })
})

describe("run_config_cache_key", () => {
  it("keys the same config the same however it was assembled", () => {
    // Object key order and tool selection order are assembly detail, not a
    // difference in what gets generated.
    const reordered: KilnAgentRunConfigProperties = {
      tools_config: { tools: ["skill::triage", "kiln_tool::search"] },
      top_p: 1.0,
      temperature: 1.0,
      input_transform: null,
      thinking_level: "medium",
      structured_output_mode: "json_schema",
      prompt_id: "simple_prompt_builder",
      model_provider_name: "openai",
      model_name: "gpt_5_4_mini",
      type: "kiln_agent",
    }
    expect(run_config_cache_key(reordered)).toBe(
      run_config_cache_key(INPUT_GEN_RUN_CONFIG),
    )
  })

  it("separates configs that differ in any field that reaches the call", () => {
    const fields: Partial<KilnAgentRunConfigProperties>[] = [
      { model_name: "gpt_5_4" },
      { model_provider_name: "openrouter" },
      { temperature: 0.7 },
      { top_p: 0.9 },
      { structured_output_mode: "json_mode" },
      { thinking_level: "high" },
      { tools_config: { tools: ["kiln_tool::search"] } },
    ]
    for (const field of fields) {
      expect(
        run_config_cache_key({ ...INPUT_GEN_RUN_CONFIG, ...field }),
      ).not.toBe(run_config_cache_key(INPUT_GEN_RUN_CONFIG))
    }
  })

  it("reads a config with no tools block as one with no tools", () => {
    const { tools_config: _dropped, ...no_tools_block } = INPUT_GEN_RUN_CONFIG
    expect(run_config_cache_key(no_tools_block)).toBe(
      run_config_cache_key({ ...no_tools_block, tools_config: { tools: [] } }),
    )
  })
})

describe("reusable_minted_inputs — single-turn input reuse", () => {
  const prompts = ["input plan a", "input plan b"]
  const guide = "A real example input from this task's dataset..."
  const config_key = run_config_cache_key(INPUT_GEN_RUN_CONFIG)
  const cache = {
    prompts_json: JSON.stringify(prompts),
    data_guide: guide,
    run_config_json: config_key,
    inputs: ["What's your return window?", "Is my laptop under warranty?"],
  }

  it("reuses when plan, run config, and guide are byte-unchanged", () => {
    expect(
      reusable_minted_inputs(
        cache,
        ["input plan a", "input plan b"],
        guide,
        config_key,
      ),
    ).toBe(cache.inputs)
  })

  it("misses with no cache", () => {
    expect(reusable_minted_inputs(null, prompts, guide, config_key)).toBeNull()
  })

  it("misses when a prompt was edited or deleted", () => {
    expect(
      reusable_minted_inputs(
        cache,
        ["input plan a EDITED", "input plan b"],
        guide,
        config_key,
      ),
    ).toBeNull()
    expect(
      reusable_minted_inputs(cache, ["input plan a"], guide, config_key),
    ).toBeNull()
  })

  it("misses when any part of the run config changed", () => {
    // Model, tools and sampling all change what the generator writes, so each
    // has to re-mint rather than serve the old inputs.
    const changed: Partial<KilnAgentRunConfigProperties>[] = [
      { model_name: "gpt_5_4" },
      { model_provider_name: "openrouter" },
      { temperature: 0.7 },
      { structured_output_mode: "json_mode" },
      { thinking_level: "high" },
      { tools_config: { tools: [] } },
      { tools_config: { tools: ["kiln_tool::search", "skill::escalate"] } },
    ]
    for (const field of changed) {
      expect(
        reusable_minted_inputs(
          cache,
          prompts,
          guide,
          run_config_cache_key({ ...INPUT_GEN_RUN_CONFIG, ...field }),
        ),
      ).toBeNull()
    }
  })

  it("misses when the grounding guide changed (a different sample minted these)", () => {
    expect(reusable_minted_inputs(cache, prompts, null, config_key)).toBeNull()
    // An ungrounded cache reuses only for an ungrounded mint; a pre-guide
    // draft (no key at all) behaves the same via the ?? null read.
    const ungrounded = { ...cache, data_guide: null }
    expect(reusable_minted_inputs(ungrounded, prompts, null, config_key)).toBe(
      ungrounded.inputs,
    )
    expect(
      reusable_minted_inputs(ungrounded, prompts, guide, config_key),
    ).toBeNull()
  })

  it("misses on a cache from before the input lane carried a config", () => {
    // We cannot say what those inputs were generated with, so they are not
    // reusable under any config.
    const { run_config_json: _dropped, ...pre_config } = cache
    expect(
      reusable_minted_inputs(pre_config, prompts, guide, config_key),
    ).toBeNull()
  })

  it("misses on an empty cached input list", () => {
    expect(
      reusable_minted_inputs(
        { ...cache, inputs: [] },
        prompts,
        guide,
        config_key,
      ),
    ).toBeNull()
  })
})

describe("draft_after_save_keeping_stranded_tags", () => {
  it("carries stranded cleanup tags but drops the just-saved batch's own", () => {
    const residual = draft_after_save_keeping_stranded_tags(
      "multi_turn_batch_1234",
      ["multi_turn_batch_1200", "multi_turn_batch_1234"],
    )
    // The saved batch is the eval's data — never carried (a later
    // replace_batch_tags would delete the eval's chains). The older aborted
    // batch is a genuine orphan, so it rides forward for cleanup.
    expect(residual.undeleted_batch_tags).toEqual(["multi_turn_batch_1200"])
    expect(residual.multi_turn_batch_tag).toBeNull()
    expect(residual.description).toBe("")
    expect(residual.batch_plan).toBeNull()
  })

  it("is a full empty draft when nothing was stranded", () => {
    expect(
      draft_after_save_keeping_stranded_tags("only_batch", ["only_batch"]),
    ).toEqual(EMPTY_BUILDER_DRAFT)
  })
})

describe("should_prefill_suggested_name", () => {
  it("prefills an empty field", () => {
    expect(should_prefill_suggested_name("", null)).toBe(true)
  })

  it("prefills a whitespace-only field — a stray space is not a typed name", () => {
    expect(should_prefill_suggested_name("   ", null)).toBe(true)
  })

  it("leaves a typed name alone when no machine name is on record — a restored draft keeps the user's name", () => {
    expect(should_prefill_suggested_name("no-fabrication", null)).toBe(false)
  })

  it("replaces an untouched machine name — it tracks the newest suggestion", () => {
    expect(
      should_prefill_suggested_name("no-fabrication", "no-fabrication"),
    ).toBe(true)
  })

  // The compare is raw bytes, not trimmed: a trailing space is a real edit.
  it("treats a trailing space on the machine's name as a user edit", () => {
    expect(
      should_prefill_suggested_name("no-fabrication ", "no-fabrication"),
    ).toBe(false)
  })

  // Clearing a typed name hands ownership back, so the next Continue refills.
  it("prefills a cleared field even with a machine name on record", () => {
    expect(should_prefill_suggested_name("   ", "no-fabrication")).toBe(true)
  })

  it("keeps a name the user edited away from the machine's — it survives every later Continue", () => {
    expect(
      should_prefill_suggested_name("refund-accuracy", "no-fabrication"),
    ).toBe(false)
  })
})

describe("should_invalidate_refined_values", () => {
  // The refined text a successful refine wrote, and the classified values it
  // was derived from — plus what those classified values became after the user
  // went back and rewrote the description.
  const refined = {
    issue_description: "The agent must not fabricate or guess at policies.",
  }
  const refined_json = JSON.stringify(refined)
  const old_source = JSON.stringify({
    issue_description: "The agent must not fabricate policies.",
  })
  const new_source = JSON.stringify({
    issue_description: "The agent must not invent warranty terms.",
  })

  it("discards programmatic values whose source moved underneath them", () => {
    // The whole point: stale refined text must not outrank the description
    // the user just rewrote.
    expect(
      should_invalidate_refined_values(
        refined,
        refined_json,
        old_source,
        new_source,
      ),
    ).toBe(true)
  })

  it("keeps programmatic values when the source is unchanged", () => {
    // A transient refine failure must not destroy an accepted refinement.
    expect(
      should_invalidate_refined_values(
        refined,
        refined_json,
        old_source,
        old_source,
      ),
    ).toBe(false)
  })

  it("keeps user-edited values even when the source moved", () => {
    expect(
      should_invalidate_refined_values(
        { issue_description: "My own wording." },
        refined_json,
        old_source,
        new_source,
      ),
    ).toBe(false)
  })

  it("keeps values with no snapshot", () => {
    // Draft-restored values have unknowable authorship, so they are kept.
    expect(
      should_invalidate_refined_values(refined, null, null, new_source),
    ).toBe(false)
  })

  it("reads reordered keys as user-edited — compares are byte-exact", () => {
    // No canonicalization: both sides must be stringified from the same
    // construction. A future second rendered field inherits this constraint.
    const current = { issue_description: "a", issue_examples: "b" }
    const reordered = JSON.stringify({
      issue_examples: "b",
      issue_description: "a",
    })
    expect(
      should_invalidate_refined_values(
        current,
        reordered,
        old_source,
        new_source,
      ),
    ).toBe(false)
  })

  it("round-trips a null-valued field", () => {
    // Unanswered fields ride as null, not "" — the compare must survive them.
    const with_null = { issue_description: "Kept.", non_issue_examples: null }
    expect(
      should_invalidate_refined_values(
        with_null,
        JSON.stringify(with_null),
        old_source,
        new_source,
      ),
    ).toBe(true)
  })
})

describe("create_eval_button_label", () => {
  it("advertises the draft only with copilot AND content", () => {
    expect(create_eval_button_label(true, true)).toBe("Continue Eval Draft")
    expect(create_eval_button_label(true, false)).toBe("Create Eval")
    expect(create_eval_button_label(false, true)).toBe("Create Eval")
    expect(create_eval_button_label(false, false)).toBe("Create Eval")
  })
})

describe("model lanes (su_driver / judge_model)", () => {
  it("round-trip through serialization", () => {
    const restored = JSON.parse(JSON.stringify(full_draft)) as BuilderDraft
    expect(restored.su_driver).toEqual({
      model_name: "gpt_5_4_mini",
      model_provider: "openai",
    })
    expect(restored.judge_model).toEqual({
      model_name: "gpt_5_4",
      model_provider: "openai",
    })
  })

  it("a pre-Drive-Settings draft restores lanes as null via ??", () => {
    // Drafts written before the lanes existed have no such keys. The
    // builder restores with `saved.su_driver ?? null` — mirror that here
    // against a legacy-shaped blob.
    const { su_driver: _su, judge_model: _judge, ...legacy } = full_draft
    const restored = JSON.parse(JSON.stringify(legacy)) as BuilderDraft
    expect(restored.su_driver ?? null).toBeNull()
    expect(restored.judge_model ?? null).toBeNull()
  })

  it("reset wipes the lanes — pre-population re-fills them", () => {
    const reset = reset_draft_keeping_tags(full_draft)
    expect(reset.su_driver).toBeNull()
    expect(reset.judge_model).toBeNull()
  })

  it("lanes alone do not make a draft restorable", () => {
    expect(
      draft_has_content({
        ...EMPTY_BUILDER_DRAFT,
        su_driver: { model_name: "m", model_provider: "openai" },
      }),
    ).toBe(false)
  })
})

describe("conversation length (turns_per_case)", () => {
  // The builder restores with `restore_turns_per_case(saved.turns_per_case,
  // TURNS_PER_CASE)` — mirrored here so the cases that matter (a stored
  // choice, no choice, an out-of-range choice, a corrupt one) are pinned
  // against the real helper.
  const PAGE_DEFAULT = 5
  const restore = (saved: BuilderDraft) =>
    restore_turns_per_case(saved.turns_per_case, PAGE_DEFAULT)

  it("round-trips the chosen length", () => {
    const restored = JSON.parse(JSON.stringify(full_draft)) as BuilderDraft
    expect(restored.turns_per_case).toBe(8)
    expect(restore(restored)).toBe(8)
  })

  it("restores the default when no choice is on record", () => {
    // Both shapes reach here: a draft from before this key existed (no key at
    // all) and a reset draft (explicit null).
    const { turns_per_case: _turns, ...legacy } = full_draft
    const legacy_restored = JSON.parse(JSON.stringify(legacy)) as BuilderDraft
    expect(restore(legacy_restored)).toBe(PAGE_DEFAULT)
    expect(EMPTY_BUILDER_DRAFT.turns_per_case).toBeNull()
    expect(restore(EMPTY_BUILDER_DRAFT)).toBe(PAGE_DEFAULT)
  })

  it("clamps a stored length that falls outside today's range", () => {
    expect(restore({ ...full_draft, turns_per_case: 99 })).toBe(20)
    expect(restore({ ...full_draft, turns_per_case: 0 })).toBe(1)
  })

  it("restores the default when the stored length isn't a number", () => {
    // A draft round-trips through JSON in IndexedDB, so a corrupt or
    // hand-edited record can carry anything. That is no choice on record, not
    // a choice of one turn.
    expect(restore({ ...full_draft, turns_per_case: NaN })).toBe(PAGE_DEFAULT)
    expect(
      restore({
        ...full_draft,
        turns_per_case: "8" as unknown as number,
      }),
    ).toBe(PAGE_DEFAULT)
  })

  it("reset drops the choice back to the default", () => {
    expect(reset_draft_keeping_tags(full_draft).turns_per_case).toBeNull()
  })
})

describe("create_eval_destination — where the Evals page's button goes", () => {
  it("continues a draft in the builder, which restores it", () => {
    expect(create_eval_destination(true, true)).toBe("builder")
  })

  it("starts on the Create Eval page otherwise", () => {
    expect(create_eval_destination(true, false)).toBe("select_template")
    expect(create_eval_destination(false, true)).toBe("select_template")
    expect(create_eval_destination(false, false)).toBe("select_template")
  })

  it("agrees with the label: 'Continue Eval Draft' always means the builder", () => {
    for (const has_copilot of [true, false]) {
      for (const has_draft of [true, false]) {
        const continues =
          create_eval_button_label(has_copilot, has_draft) ===
          "Continue Eval Draft"
        expect(
          create_eval_destination(has_copilot, has_draft) === "builder",
        ).toBe(continues)
      }
    }
  })
})

describe("draft_is_resumable — what the Evals page offers to continue", () => {
  it("is false for the empty draft", () => {
    expect(draft_is_resumable(EMPTY_BUILDER_DRAFT)).toBe(false)
  })

  it("is false for cleanup tags alone — a Reset leaves exactly this behind", () => {
    const after_reset = reset_draft_keeping_tags(full_draft)
    expect(draft_has_content(after_reset)).toBe(true)
    expect(draft_is_resumable(after_reset)).toBe(false)
  })

  it("is true for a description, a name, refined values, or a plan", () => {
    expect(
      draft_is_resumable({ ...EMPTY_BUILDER_DRAFT, description: "d" }),
    ).toBe(true)
    expect(draft_is_resumable({ ...EMPTY_BUILDER_DRAFT, name: "n" })).toBe(true)
    expect(
      draft_is_resumable({
        ...EMPTY_BUILDER_DRAFT,
        refined_property_values: { issue_description: "x" },
      }),
    ).toBe(true)
    expect(
      draft_is_resumable({
        ...EMPTY_BUILDER_DRAFT,
        batch_plan: { prompts: ["p"], summary: "s" },
      }),
    ).toBe(true)
  })

  it("never claims more than draft_has_content does", () => {
    for (const draft of [EMPTY_BUILDER_DRAFT, full_draft]) {
      if (draft_is_resumable(draft)) expect(draft_has_content(draft)).toBe(true)
    }
  })
})
