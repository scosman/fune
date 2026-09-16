import { get } from "svelte/store"
import { sessionStorageStore } from "./local_storage_store"

export interface GitImportWizardState {
  git_url: string
  pat_token: string | null
  oauth_token: string | null
  auth_mode: string
  clone_path: string
  selected_branch: string
  selected_project_path: string
  selected_project_id: string
  selected_project_name: string
}

const INITIAL_STATE: GitImportWizardState = {
  git_url: "",
  pat_token: null,
  oauth_token: null,
  auth_mode: "system_keys",
  clone_path: "",
  selected_branch: "",
  selected_project_path: "",
  selected_project_id: "",
  selected_project_name: "",
}

export const git_import_wizard_store =
  sessionStorageStore<GitImportWizardState>("git_import_wizard", INITIAL_STATE)

export function clear_wizard_store() {
  git_import_wizard_store.set({ ...INITIAL_STATE })
}

export type WizardStep =
  | "method"
  | "local_file"
  | "local_trust_confirm"
  | "url"
  | "credentials"
  | "branch"
  | "project"
  | "trust_confirm"
  | "complete"

export function validate_step_requirements(step: WizardStep): boolean {
  const state = get(git_import_wizard_store)

  switch (step) {
    case "method":
    case "local_file":
    // local_trust_confirm depends on the component-local selected file path,
    // which this store-only check can't see; the component gates it directly.
    case "local_trust_confirm":
    case "url":
      return true
    case "credentials":
      return !!state.git_url
    case "branch":
      if (!state.git_url) return false
      if (state.auth_mode === "pat_token" && !state.pat_token) return false
      if (state.auth_mode === "github_oauth" && !state.oauth_token) return false
      return true
    case "project":
      return !!state.clone_path
    case "trust_confirm":
      if (!state.git_url) return false
      if (state.auth_mode === "pat_token" && !state.pat_token) return false
      if (state.auth_mode === "github_oauth" && !state.oauth_token) return false
      return true
    case "complete":
      return (
        !!state.clone_path &&
        !!state.selected_branch &&
        !!state.selected_project_path
      )
    default:
      return false
  }
}
