import { describe, it, expect } from "vitest"
import { build_suggested_models } from "./suggested_models"
import type { AvailableModels, ModelDetails } from "$lib/types"

function model(
  id: string,
  flags: Partial<
    Pick<
      ModelDetails,
      | "suggested_for_evals"
      | "suggested_for_data_gen"
      | "suggested_for_synthetic_user"
    >
  > = {},
): ModelDetails {
  return {
    id,
    name: `Name of ${id}`,
    supports_structured_output: true,
    supports_data_gen: true,
    supports_logprobs: false,
    suggested_for_data_gen: false,
    suggested_for_evals: false,
    suggested_for_synthetic_user: false,
    uncensored: false,
    suggested_for_uncensored_data_gen: false,
    task_filter: null,
    supports_doc_extraction: false,
    suggested_for_doc_extraction: false,
    multimodal_capable: false,
    multimodal_mime_types: null,
    supports_function_calling: false,
    untested_model: false,
    deprecated: false,
    ...flags,
  } as ModelDetails
}

function provider(
  provider_id: string,
  models: ModelDetails[],
): AvailableModels {
  return {
    provider_id,
    provider_name: `Display ${provider_id}`,
    models,
  } as AvailableModels
}

describe("build_suggested_models", () => {
  it("filters by the requested mode's registry flag", () => {
    const providers = [
      provider("openrouter", [
        model("judge_only", { suggested_for_evals: true }),
        model("sdg_only", { suggested_for_data_gen: true }),
        model("su_only", { suggested_for_synthetic_user: true }),
        model("neither"),
      ]),
    ]
    expect(
      build_suggested_models(providers, "evals").map((m) => m.model_id),
    ).toEqual(["judge_only"])
    expect(
      build_suggested_models(providers, "data_gen").map((m) => m.model_id),
    ).toEqual(["sdg_only"])
    expect(
      build_suggested_models(providers, "synthetic_user").map(
        (m) => m.model_id,
      ),
    ).toEqual(["su_only"])
  })

  it("keeps first-appearance order — the first entry is the pre-selection", () => {
    const providers = [
      provider("openrouter", [
        model("first", { suggested_for_evals: true }),
        model("second", { suggested_for_evals: true }),
      ]),
    ]
    const result = build_suggested_models(providers, "evals")
    expect(result[0].model_id).toBe("first")
    expect(result[0].provider_id).toBe("openrouter")
  })

  it("dedupes a model across providers, preferring official over aggregator", () => {
    const providers = [
      provider("openrouter", [model("shared", { suggested_for_evals: true })]),
      provider("openai", [model("shared", { suggested_for_evals: true })]),
    ]
    const result = build_suggested_models(providers, "evals")
    expect(result).toHaveLength(1)
    expect(result[0].provider_id).toBe("openai")
  })

  it("keeps the earlier provider when the later one is lower priority", () => {
    const providers = [
      provider("openai", [model("shared", { suggested_for_evals: true })]),
      provider("openrouter", [model("shared", { suggested_for_evals: true })]),
    ]
    const result = build_suggested_models(providers, "evals")
    expect(result).toHaveLength(1)
    expect(result[0].provider_id).toBe("openai")
  })

  it("ranks providers missing from the preference order last", () => {
    const providers = [
      provider("some_custom", [model("shared", { suggested_for_evals: true })]),
      provider("openrouter", [model("shared", { suggested_for_evals: true })]),
    ]
    const result = build_suggested_models(providers, "evals")
    expect(result[0].provider_id).toBe("openrouter")
  })

  it("returns empty for no providers", () => {
    expect(build_suggested_models([], "evals")).toEqual([])
  })
})
