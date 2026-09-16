import { describe, it, expect } from "vitest"
import {
  ALL_V2_EVAL_TYPES,
  getV2EvalTypeMetadata,
  getV2TypeFromEvalConfig,
  buildV2EvalTypeRegistry,
  extractV2Props,
  evalTypeJudgeLabel,
  manualExampleSupport,
  referenceDataKeys,
  referenceDataUsageMode,
  NO_JUDGE_PROMPT,
  type JudgeReferenceSignals,
  type V2EvalType,
  type V2EvalTypeMetadata,
  type EvalTypeTag,
  type ReferenceDataUsageMode,
} from "./registry"
import type { EvalConfig } from "$lib/types"
import type { V2EvalConfigProperties } from "$lib/api/v2_eval_api"

const EXPECTED_TYPES: V2EvalType[] = [
  "exact_match",
  "pattern_match",
  "contains",
  "set_check",
  "tool_call_check",
  "step_count_check",
  "llm_judge",
  "code_eval",
]

describe("ALL_V2_EVAL_TYPES", () => {
  it("contains exactly the expected entries", () => {
    expect(ALL_V2_EVAL_TYPES).toHaveLength(EXPECTED_TYPES.length)
  })

  it("contains all expected type values", () => {
    for (const t of EXPECTED_TYPES) {
      expect(ALL_V2_EVAL_TYPES).toContain(t)
    }
  })

  it("has no duplicates", () => {
    const unique = new Set(ALL_V2_EVAL_TYPES)
    expect(unique.size).toBe(ALL_V2_EVAL_TYPES.length)
  })
})

describe("getV2EvalTypeMetadata", () => {
  it("returns metadata with all required fields for every type", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      const meta = getV2EvalTypeMetadata(t)
      expect(meta.label).toBeTruthy()
      expect(meta.description).toBeTruthy()
      expect(typeof meta.requiresTrust).toBe("boolean")
      expect(meta.createFormComponent).toBeTruthy()
      expect(meta.resultRendererComponent).toBeTruthy()
    }
  })

  it("returns correct label for each type", () => {
    const expectedLabels: Record<V2EvalType, string> = {
      exact_match: "Exact Match",
      pattern_match: "Pattern Match",
      contains: "Contains",
      set_check: "Set Check",
      tool_call_check: "Tool Call Check",
      step_count_check: "Step Count Check",
      llm_judge: "LLM as Judge",
      code_eval: "Code",
    }
    for (const [type, label] of Object.entries(expectedLabels)) {
      expect(getV2EvalTypeMetadata(type as V2EvalType).label).toBe(label)
    }
  })

  it("only code_eval requires trust", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      const meta = getV2EvalTypeMetadata(t)
      if (t === "code_eval") {
        expect(meta.requiresTrust).toBe(true)
      } else {
        expect(meta.requiresTrust).toBe(false)
      }
    }
  })

  it("does not have an icon string field (icons are now SVG components)", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      const meta = getV2EvalTypeMetadata(t)
      expect("icon" in meta).toBe(false)
    }
  })

  it("throws for an invalid type via assertNever", () => {
    expect(() =>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getV2EvalTypeMetadata("invalid_type" as any),
    ).toThrow("Unexpected value")
  })

  it("returns distinct form components for different types", () => {
    const components = new Set<V2EvalTypeMetadata["createFormComponent"]>()
    for (const t of ALL_V2_EVAL_TYPES) {
      components.add(getV2EvalTypeMetadata(t).createFormComponent)
    }
    expect(components.size).toBe(ALL_V2_EVAL_TYPES.length)
  })

  it("every type has a non-empty pageTitle", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      expect(getV2EvalTypeMetadata(t).pageTitle).toBeTruthy()
    }
  })

  it("every type has a non-empty pageSubtitle", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      expect(getV2EvalTypeMetadata(t).pageSubtitle).toBeTruthy()
    }
  })

  it("every type has a non-empty explainer", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      expect(getV2EvalTypeMetadata(t).explainer).toBeTruthy()
    }
  })

  it("example is non-empty when present", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      const meta = getV2EvalTypeMetadata(t)
      if (meta.example !== undefined) {
        expect(meta.example).toBeTruthy()
      }
    }
  })

  it("only code_eval has tags (Beta); all other types have empty tags", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      const meta = getV2EvalTypeMetadata(t)
      if (t === "code_eval") {
        expect(meta.tags).toHaveLength(1)
        expect(meta.tags[0]).toEqual({ label: "Beta", tone: "beta" })
      } else {
        expect(meta.tags).toHaveLength(0)
      }
    }
  })

  it("every tag has a valid tone", () => {
    const validTones: EvalTypeTag["tone"][] = ["default", "beta"]
    for (const t of ALL_V2_EVAL_TYPES) {
      const meta = getV2EvalTypeMetadata(t)
      for (const tag of meta.tags) {
        expect(tag.label).toBeTruthy()
        expect(validTones).toContain(tag.tone)
      }
    }
  })

  it("exactly one type has recommended set to true", () => {
    const recommendedTypes = ALL_V2_EVAL_TYPES.filter(
      (t) => getV2EvalTypeMetadata(t).recommended === true,
    )
    expect(recommendedTypes).toHaveLength(1)
  })

  it("recommended type is the first entry in ALL_V2_EVAL_TYPES", () => {
    const first = ALL_V2_EVAL_TYPES[0]
    expect(getV2EvalTypeMetadata(first).recommended).toBe(true)
  })

  it("non-recommended types do not have recommended set to true", () => {
    for (const t of ALL_V2_EVAL_TYPES.slice(1)) {
      const meta = getV2EvalTypeMetadata(t)
      expect(meta.recommended).not.toBe(true)
    }
  })

  it("returns correct pageTitle for each type", () => {
    const expectedTitles: Record<V2EvalType, string> = {
      llm_judge: "Add an LLM Judge",
      code_eval: "Add a Code Judge",
      exact_match: "Add an Exact Match Check",
      pattern_match: "Add a Pattern Match Check",
      contains: "Add a Contains Check",
      set_check: "Add a Set Check",
      tool_call_check: "Add a Tool Call Check",
      step_count_check: "Add a Step Count Check",
    }
    for (const [type, title] of Object.entries(expectedTitles)) {
      expect(getV2EvalTypeMetadata(type as V2EvalType).pageTitle).toBe(title)
    }
  })

  it("code_eval has a beta-toned tag", () => {
    const meta = getV2EvalTypeMetadata("code_eval")
    const betaTags = meta.tags.filter((tag) => tag.tone === "beta")
    expect(betaTags).toHaveLength(1)
    expect(betaTags[0].label).toBe("Beta")
  })

  it("llm_judge has no tags", () => {
    const meta = getV2EvalTypeMetadata("llm_judge")
    expect(meta.tags).toEqual([])
  })

  it("agent types have no tags", () => {
    const agentTypes: V2EvalType[] = ["tool_call_check", "step_count_check"]
    for (const t of agentTypes) {
      const meta = getV2EvalTypeMetadata(t)
      expect(meta.tags).toEqual([])
    }
  })

  it("deterministic types have no tags", () => {
    const deterministicTypes: V2EvalType[] = [
      "exact_match",
      "pattern_match",
      "contains",
      "set_check",
    ]
    for (const t of deterministicTypes) {
      const meta = getV2EvalTypeMetadata(t)
      expect(meta.tags).toEqual([])
    }
  })

  it("no type has any default-tone tags", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      const meta = getV2EvalTypeMetadata(t)
      const defaultTags = meta.tags.filter((tag) => tag.tone === "default")
      expect(defaultTags).toHaveLength(0)
    }
  })
})

describe("evalTypeJudgeLabel", () => {
  it('returns "LLM as Judge" for llm_judge (no double Judge)', () => {
    expect(evalTypeJudgeLabel("llm_judge")).toBe("LLM as Judge")
  })

  it('returns "Code Judge" for code_eval', () => {
    expect(evalTypeJudgeLabel("code_eval")).toBe("Code Judge")
  })

  it('returns "Exact Match Judge" for exact_match', () => {
    expect(evalTypeJudgeLabel("exact_match")).toBe("Exact Match Judge")
  })

  it('returns "Pattern Match Judge" for pattern_match', () => {
    expect(evalTypeJudgeLabel("pattern_match")).toBe("Pattern Match Judge")
  })

  it('returns "Contains Judge" for contains', () => {
    expect(evalTypeJudgeLabel("contains")).toBe("Contains Judge")
  })

  it('returns "Set Check Judge" for set_check', () => {
    expect(evalTypeJudgeLabel("set_check")).toBe("Set Check Judge")
  })

  it('returns "Tool Call Check Judge" for tool_call_check', () => {
    expect(evalTypeJudgeLabel("tool_call_check")).toBe("Tool Call Check Judge")
  })

  it('returns "Step Count Check Judge" for step_count_check', () => {
    expect(evalTypeJudgeLabel("step_count_check")).toBe(
      "Step Count Check Judge",
    )
  })
})

describe("getV2TypeFromEvalConfig", () => {
  it("returns the V2 type for a valid v2 config", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      const result = getV2TypeFromEvalConfig({
        config_type: "v2",
        properties: { type: t },
      })
      expect(result).toBe(t)
    }
  })

  it("returns null for a legacy g_eval config", () => {
    expect(
      getV2TypeFromEvalConfig({
        config_type: "g_eval",
        properties: { type: "exact_match" },
      }),
    ).toBeNull()
  })

  it("returns null for a legacy llm_as_judge config", () => {
    expect(
      getV2TypeFromEvalConfig({
        config_type: "llm_as_judge",
        properties: null,
      }),
    ).toBeNull()
  })

  it("returns null when properties is null", () => {
    expect(
      getV2TypeFromEvalConfig({
        config_type: "v2",
        properties: null,
      }),
    ).toBeNull()
  })

  it("returns null when properties is undefined", () => {
    expect(
      getV2TypeFromEvalConfig({
        config_type: "v2",
      }),
    ).toBeNull()
  })

  it("returns null when properties has no type field", () => {
    expect(
      getV2TypeFromEvalConfig({
        config_type: "v2",
        properties: {},
      }),
    ).toBeNull()
  })

  it("returns null for an unrecognized V2 type string", () => {
    expect(
      getV2TypeFromEvalConfig({
        config_type: "v2",
        properties: { type: "not_a_real_type" },
      }),
    ).toBeNull()
  })

  it("returns null when type is a non-string value", () => {
    expect(
      getV2TypeFromEvalConfig({
        config_type: "v2",
        properties: { type: 42 as unknown as string },
      }),
    ).toBeNull()
  })
})

describe("buildV2EvalTypeRegistry", () => {
  it("returns a Map with an entry for every type", () => {
    const registry = buildV2EvalTypeRegistry()
    expect(registry.size).toBe(ALL_V2_EVAL_TYPES.length)
    for (const t of ALL_V2_EVAL_TYPES) {
      expect(registry.has(t)).toBe(true)
    }
  })

  it("registry values match getV2EvalTypeMetadata", () => {
    const registry = buildV2EvalTypeRegistry()
    for (const t of ALL_V2_EVAL_TYPES) {
      const fromRegistry = registry.get(t)!
      const fromFn = getV2EvalTypeMetadata(t)
      expect(fromRegistry.label).toBe(fromFn.label)
      expect(fromRegistry.description).toBe(fromFn.description)
      expect(fromRegistry.requiresTrust).toBe(fromFn.requiresTrust)
      expect(fromRegistry.recommended).toBe(fromFn.recommended)
      expect(fromRegistry.tags).toEqual(fromFn.tags)
      expect(fromRegistry.pageTitle).toBe(fromFn.pageTitle)
      expect(fromRegistry.pageSubtitle).toBe(fromFn.pageSubtitle)
      expect(fromRegistry.createFormComponent).toBe(fromFn.createFormComponent)
      expect(fromRegistry.resultRendererComponent).toBe(
        fromFn.resultRendererComponent,
      )
    }
  })
})

describe("extractV2Props", () => {
  function makeEvalConfig(
    properties: Record<string, unknown> | null,
  ): EvalConfig {
    return {
      v: 1,
      name: "test",
      config_type: "v2",
      model_type: "eval_config",
      properties,
    } as EvalConfig
  }

  it("returns typed properties when type matches", () => {
    const config = makeEvalConfig({
      type: "exact_match",
      case_sensitive: true,
      expected_value: "hello",
      reference_key: null,
      value_expression: null,
    })
    const result = extractV2Props(config, "exact_match")
    expect(result).not.toBeNull()
    expect(result!.type).toBe("exact_match")
    expect(result!.case_sensitive).toBe(true)
    expect(result!.expected_value).toBe("hello")
  })

  it("returns null when type discriminator does not match", () => {
    const config = makeEvalConfig({
      type: "contains",
      case_sensitive: false,
      mode: "must_contain",
    })
    const result = extractV2Props(config, "exact_match")
    expect(result).toBeNull()
  })

  it("returns null when eval_config is null", () => {
    expect(extractV2Props(null, "exact_match")).toBeNull()
  })

  it("returns null when properties is null", () => {
    const config = makeEvalConfig(null)
    expect(extractV2Props(config, "exact_match")).toBeNull()
  })

  it("returns null when properties has no type field", () => {
    const config = makeEvalConfig({ case_sensitive: true })
    expect(extractV2Props(config, "exact_match")).toBeNull()
  })

  it("works for each V2 type", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      const config = makeEvalConfig({ type: t })
      const result = extractV2Props(config, t)
      expect(result).not.toBeNull()
      expect(result!.type).toBe(t)
    }
  })
})

describe("manualExampleSupport", () => {
  // Only the types that inherently operate on the trace (no output-only mode)
  // block manual examples.
  const TRACE_ONLY_TYPES: V2EvalType[] = ["tool_call_check", "step_count_check"]

  const MANUAL_SUPPORTED_TYPES: V2EvalType[] = [
    "exact_match",
    "pattern_match",
    "contains",
    "set_check",
    "llm_judge",
    "code_eval",
  ]

  it("blocks manual examples for trace-only judge types", () => {
    for (const t of TRACE_ONLY_TYPES) {
      const result = manualExampleSupport(t)
      expect(result.supported).toBe(false)
      expect(result.reason).toBeTruthy()
    }
  })

  it("allows manual examples for every other judge type", () => {
    for (const t of MANUAL_SUPPORTED_TYPES) {
      expect(manualExampleSupport(t)).toEqual({ supported: true, reason: null })
    }
  })

  it("covers every eval type", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      expect(() => manualExampleSupport(t)).not.toThrow()
    }
  })
})

function signals(
  overrides: Partial<JudgeReferenceSignals>,
): JudgeReferenceSignals {
  return { ...NO_JUDGE_PROMPT, ...overrides }
}

// Behavior with SHOW_REFERENCE_DATA_UI off (the shipped default). The flag-on
// behavior is covered in registry.reference_data.test.ts.
describe("referenceDataUsageMode", () => {
  it("reports 'none' for every V2EvalType while reference data is hidden", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      expect(referenceDataUsageMode(t, NO_JUDGE_PROMPT)).toBe("none")
    }
  })

  it("shows the input for an llm_judge whose prompt uses reference data", () => {
    // Not behind the flag: the Test Judge pane is the only place a reference answer
    // can be supplied, and without it a reference-answer judge is validated against a
    // prompt with no reference block and then saved as one that renders it.
    expect(
      referenceDataUsageMode(
        "llm_judge",
        signals({
          prompt_template:
            "Grade {{ final_message }} against {{ reference_data.reference_answer }}",
        }),
      ),
    ).toBe("llm_judge")
  })

  it("shows the input when the server declared a key, whatever the prompt says", () => {
    // The user can edit the reference block out of the prompt; the server keeps
    // requiring the key, so every test run would skip with no way to supply one.
    expect(
      referenceDataUsageMode(
        "llm_judge",
        signals({
          prompt_template: "Rate {{ final_message }}",
          server_reference_keys: ["reference_answer"],
        }),
      ),
    ).toBe("llm_judge")
  })

  it("shows the input when the prompt could not be fetched", () => {
    // Nothing is known, and the save path bakes the server default either way.
    // Offering an unused input is the harmless direction.
    expect(
      referenceDataUsageMode(
        "llm_judge",
        signals({ prompt_unavailable: true }),
      ),
    ).toBe("llm_judge")
  })

  it("hides the input for an llm_judge that needs no reference data", () => {
    expect(
      referenceDataUsageMode(
        "llm_judge",
        signals({ prompt_template: "Rate {{ final_message }}" }),
      ),
    ).toBe("none")
  })

  it("hides the input for an llm_judge with no prompt yet", () => {
    expect(referenceDataUsageMode("llm_judge", NO_JUDGE_PROMPT)).toBe("none")
  })

  it("keeps every other type behind the flag, signals or not", () => {
    for (const t of ALL_V2_EVAL_TYPES.filter((t) => t !== "llm_judge")) {
      expect(
        referenceDataUsageMode(
          t,
          signals({
            prompt_template: "{{ reference_data.anything }}",
            server_reference_keys: ["anything"],
            prompt_unavailable: true,
          }),
        ),
      ).toBe("none")
    }
  })

  it("covers every eval type without throwing", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      expect(() => referenceDataUsageMode(t, NO_JUDGE_PROMPT)).not.toThrow()
    }
  })

  it("still throws for an invalid type via assertNever", () => {
    expect(() =>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      referenceDataUsageMode("invalid_type" as any, NO_JUDGE_PROMPT),
    ).toThrow("Unexpected value")
  })

  it("returns one of the valid mode strings for every type", () => {
    const validModes: ReferenceDataUsageMode[] = [
      "llm_judge",
      "reference_field",
      "code",
      "optional",
      "none",
    ]
    for (const t of ALL_V2_EVAL_TYPES) {
      expect(validModes).toContain(referenceDataUsageMode(t, NO_JUDGE_PROMPT))
    }
  })

  it("keeps reference data out of every type's user-facing copy", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      const meta = getV2EvalTypeMetadata(t)
      const copy = [
        meta.label,
        meta.description,
        meta.pageTitle,
        meta.pageSubtitle,
        meta.explainer ?? "",
        meta.example ?? "",
      ]
        .join(" ")
        .toLowerCase()
      expect(copy).not.toContain("reference")
    }
  })
})

describe("referenceDataKeys", () => {
  it.each([
    [{ type: "exact_match", reference_key: "answer" }, ["answer"]],
    [{ type: "exact_match", expected_value: "yes" }, []],
    [{ type: "contains", reference_key: "answer" }, ["answer"]],
    [{ type: "contains", substring: "yes" }, []],
    [{ type: "set_check", reference_key: "answer" }, ["answer"]],
    [{ type: "set_check", expected_set: ["a"] }, []],
    [{ type: "llm_judge", reference_keys: ["a", "b"] }, ["a", "b"]],
    [{ type: "llm_judge", reference_keys: [] }, []],
    [{ type: "code_eval", reference_keys: ["a"] }, ["a"]],
    [{ type: "pattern_match", pattern: "^y" }, []],
    [{ type: "tool_call_check" }, []],
    [{ type: "step_count_check" }, []],
  ])("mirrors the backend keys for %o", (props, expected) => {
    expect(
      referenceDataKeys(props as unknown as V2EvalConfigProperties),
    ).toEqual(expected)
  })

  it("covers every V2 eval type without throwing", () => {
    for (const t of ALL_V2_EVAL_TYPES) {
      expect(() =>
        referenceDataKeys({
          type: t,
          reference_keys: [],
        } as unknown as V2EvalConfigProperties),
      ).not.toThrow()
    }
  })

  it("throws via assertNever for a type it does not know", () => {
    expect(() =>
      referenceDataKeys({
        type: "invented_type",
      } as unknown as V2EvalConfigProperties),
    ).toThrow("Unexpected value")
  })
})
