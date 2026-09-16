import type { AvailableModels, ModelDetails } from "$lib/types"

// A model the registry suggests for a given role, resolved to the best
// provider the user has connected. model_id/provider_id are the wire ids
// (e.g. "gpt_5_4" / "openai"); the *_name fields are display names.
export type SuggestedModel = {
  model_name: string
  provider_name: string
  model_id: string
  provider_id: string
}

// Which registry suggestion flag to read. The registry (SDK ml_model_list)
// maintains per-provider suggested_for_* booleans; these are the roles the
// eval builder and judge pickers care about.
export type SuggestedModelMode = "evals" | "data_gen" | "synthetic_user"

// Only the boolean fields of ModelDetails, so a role can be mapped to a real
// suggestion flag and nothing else — a string field would read as always
// suggested rather than failing to compile.
type ModelDetailsBooleanField = {
  [K in keyof ModelDetails]-?: NonNullable<ModelDetails[K]> extends boolean
    ? K
    : never
}[keyof ModelDetails]

// The registry flag each role reads, so adding a role is one line here
// rather than another branch inside the loop below.
const mode_flags: Record<SuggestedModelMode, ModelDetailsBooleanField> = {
  evals: "suggested_for_evals",
  data_gen: "suggested_for_data_gen",
  synthetic_user: "suggested_for_synthetic_user",
}

// Tie-break when the same model is available via several providers:
// official/native APIs before aggregators. (v1 judge form's order.)
const provider_id_preferred_order = [
  "openai",
  "gemini_api",
  "vertex",
  "anthropic",
  "groq",
  "openrouter",
  "ollama",
]

function provider_priority(provider_id: string): number {
  const idx = provider_id_preferred_order.indexOf(provider_id)
  return idx === -1 ? Infinity : idx
}

// The registry-suggested models for a role, across the user's connected
// providers, deduped by model with the preferred provider winning. Order is
// first-appearance in the available-models payload (registry order within
// each provider) — the first entry is the pre-selection default.
export function build_suggested_models(
  providers: AvailableModels[],
  mode: SuggestedModelMode,
): SuggestedModel[] {
  const suggested: SuggestedModel[] = []

  for (const provider of providers) {
    for (const model of provider.models) {
      if (!model[mode_flags[mode]]) {
        continue
      }
      const existing_index = suggested.findIndex((s) => s.model_id === model.id)
      const candidate: SuggestedModel = {
        model_name: model.name,
        model_id: model.id,
        provider_id: provider.provider_id,
        provider_name: provider.provider_name,
      }
      if (existing_index === -1) {
        suggested.push(candidate)
      } else if (
        provider_priority(provider.provider_id) <
        provider_priority(suggested[existing_index].provider_id)
      ) {
        suggested[existing_index] = candidate
      }
    }
  }

  return suggested
}
