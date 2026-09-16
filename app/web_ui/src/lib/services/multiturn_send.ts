import { client } from "$lib/api_client"
import {
  isMcpRunConfig,
  type RunConfigProperties,
  type TaskRun,
} from "$lib/types"

export type RunConfigController = {
  clear_run_options_errors: () => void
  clear_model_dropdown_error: () => void
  run_options_as_run_config_properties: () => RunConfigProperties
  get_selected_model: () => string | null
  set_model_dropdown_error: (msg: string) => void
}

export type InputFormController = {
  get_plaintext_input_data: () => string | null
  clear_input: () => void
}

export type SendMultiturnArgs = {
  project_id: string
  task_id: string
  parent_task_run_id: string | null | undefined
  run_config_component: RunConfigController | null | undefined
  input_form: InputFormController | null | undefined
  // Receives the new run id and the full created run (the POST response,
  // which already contains the completed trace). Callers can hand the run
  // straight to the next page to avoid a redundant load / loading flash.
  on_success: (new_run_id: string, created_run: TaskRun) => Promise<void> | void
  tags?: string[]
  // The message text to send. When omitted, it's read from input_form. Pass
  // it explicitly when the caller has already cleared the input (so the text
  // isn't lost from the in-flight request).
  plaintext?: string
  // When true, a missing parent_task_run_id is allowed and creates a new
  // root conversation (the first turn). Used by the /run page; the in-chat
  // composer leaves this false so it can't fire before its run has loaded.
  allow_root_turn?: boolean
}

export type SendMultiturnResult =
  | { ok: true; new_run_id: string }
  | { ok: false; error: unknown }

// Pure-ish, side-effect-light helper that performs the multiturn Send flow.
// On success the caller-provided on_success runs first (so navigation happens),
// and the input form is only cleared after on_success resolves. This way an
// error in on_success does not silently drop the user's typed text.
export async function send_multiturn(
  args: SendMultiturnArgs,
): Promise<SendMultiturnResult> {
  const {
    project_id,
    task_id,
    parent_task_run_id,
    run_config_component,
    input_form,
    on_success,
    // Multiturn turns are manual runs, tagged the same as the /run page so
    // they aren't singled out from other manually-created runs.
    tags = ["manual_run"],
    allow_root_turn = false,
    plaintext,
  } = args

  if (!parent_task_run_id && !allow_root_turn) {
    return {
      ok: false,
      error: new Error(
        "Cannot send a multiturn message: the current run is not loaded yet.",
      ),
    }
  }

  if (!run_config_component) {
    return {
      ok: false,
      error: new Error(
        "Task configuration is still loading. Please wait a moment and try again.",
      ),
    }
  }

  run_config_component.clear_run_options_errors()
  run_config_component.clear_model_dropdown_error()
  const run_config_properties =
    run_config_component.run_options_as_run_config_properties()
  const is_mcp = isMcpRunConfig(run_config_properties)
  if (!is_mcp && !run_config_component.get_selected_model()) {
    run_config_component.set_model_dropdown_error("Required")
    return {
      ok: false,
      error: new Error("You must select a model before sending"),
    }
  }

  const text = plaintext ?? input_form?.get_plaintext_input_data() ?? ""
  const { data, error: fetch_error } = await client.POST(
    "/api/projects/{project_id}/tasks/{task_id}/run",
    {
      params: { path: { project_id, task_id } },
      body: {
        run_config_properties,
        plaintext_input: text,
        structured_input: null,
        tags,
        parent_task_run_id: parent_task_run_id ?? null,
      },
    },
  )

  if (fetch_error) {
    return { ok: false, error: fetch_error }
  }
  if (!data?.id) {
    return {
      ok: false,
      error: new Error("Server did not return a new run id."),
    }
  }

  await on_success(data.id, data)
  // Only clear input after on_success resolves. If on_success throws, the
  // caller catches it and the textarea contents are preserved. on_success
  // may also have already unmounted the form (e.g. by clearing `run`), in
  // which case the bound reference points at a destroyed component with no
  // method on it — guard against that.
  if (typeof input_form?.clear_input === "function") {
    input_form.clear_input()
  }
  return { ok: true, new_run_id: data.id }
}
