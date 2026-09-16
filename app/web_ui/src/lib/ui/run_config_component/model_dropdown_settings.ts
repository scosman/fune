import type { ModelDetails } from "$lib/types"

export interface ModelDropdownSettings {
  // Filter out all the models that do not match the predicate
  filter_models_predicate: (model: ModelDetails) => boolean
  requires_structured_output: boolean
  requires_data_gen: boolean
  requires_logprobs: boolean
  requires_uncensored_data_gen: boolean
  requires_doc_extraction: boolean
  requires_tool_support: boolean
  suggested_mode:
    | "data_gen"
    | "evals"
    | "uncensored_data_gen"
    | "doc_extraction"
    | "synthetic_user"
    | null
}

// Whether the "we suggest a Recommended model" advisory renders under a model
// dropdown.
//
// `suggestion_known` is false while the model list is still loading: until it
// arrives, a chosen model reads as "not suggested" whatever it really is. Every
// caller waits that out rather than flash an amber warning that turns green a
// moment later and shifts the rows under it.
export function show_suggested_advisory(
  model_selected: boolean,
  suggestion_known: boolean,
): boolean {
  if (model_selected && !suggestion_known) {
    return false
  }
  return true
}
