// The judge-config shapes shared across the eval builder. The judge MODEL
// is always chosen by the user (the builder's Drive Settings pickers,
// pre-populated from the task's last saved eval or the registry's
// suggested-for-evals models) — nothing here hardcodes a model or provider,
// so the builder carries no dependency on any particular provider being
// connected.

import type { components } from "$lib/api_schema"

// The ONE judge shape across the builder: the review step runs this judge
// and the save path persists it, so the calibrated judge is the shipped one.
export type JudgeConfig = components["schemas"]["JudgeConfig"]

// The provider registry's enum — a lane's provider is always one of these
// (the picks come from the models registry), and JudgeConfig now validates
// it, so ModelChoice carries the same enum end-to-end rather than a bare
// string.
type ModelProviderName = components["schemas"]["ModelProviderName"]

// A bare model choice for one of the builder's lanes (synthetic-user driver
// or judge), as the wire carries it.
export type ModelChoice = {
  model_name: string
  model_provider: ModelProviderName
}

// Construct a lane choice from registry-sourced ids (a dropdown pick, a
// suggested model, or a persisted eval config). The provider always
// originates from the models registry — a real ModelProviderName — so this is
// the single honest wire→domain boundary where the loose string is asserted,
// keeping every downstream lane (and the JudgeConfig built from it) enum-typed
// without scattering casts at each construction.
export function model_choice(
  model_name: string,
  model_provider: string,
): ModelChoice {
  return { model_name, model_provider: model_provider as ModelProviderName }
}
