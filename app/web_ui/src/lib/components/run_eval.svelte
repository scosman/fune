<script lang="ts">
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import Dialog from "$lib/ui/dialog.svelte"
  import Warning from "$lib/ui/warning.svelte"
  import { base_url } from "$lib/api_client"
  import { stream_sse } from "$lib/utils/sse_stream"
  import posthog from "posthog-js"

  export let btn_size: "normal" | "mid" | "small" | "xs" = "mid"
  export let btn_primary: boolean = true
  export let btn_class: string = ""
  export let on_run_complete: () => void = () => {}
  export let eval_state:
    | "not_started"
    | "running"
    | "complete"
    | "complete_with_errors" = "not_started"

  export let eval_type: "eval_config" | "run_config"
  export let project_id: string
  export let task_id: string
  export let eval_id: string
  export let current_eval_config_id: string | null = null
  export let run_all: boolean = false
  export let run_config_ids: string[] = []
  let run_url: string = ""
  $: {
    if (eval_type === "run_config") {
      const params = new URLSearchParams()

      if (run_all) {
        params.append("all_run_configs", "true")
      } else {
        params.append("all_run_configs", "false")
        // FastAPI expects multiple query parameters with the same name for list[str]
        run_config_ids.forEach((id) => {
          params.append("run_config_ids", id)
        })
      }

      run_url = `${base_url}/api/projects/${encodeURIComponent(project_id)}/tasks/${encodeURIComponent(task_id)}/evals/${encodeURIComponent(eval_id)}/eval_config/${encodeURIComponent(current_eval_config_id!)}/run_comparison?${params.toString()}`
    } else if (eval_type === "eval_config") {
      // Eval config only supports running all evals for now
      run_all = true
      run_url = `${base_url}/api/projects/${encodeURIComponent(project_id)}/tasks/${encodeURIComponent(task_id)}/evals/${encodeURIComponent(eval_id)}/run_calibration`
    }
  }

  let run_dialog: Dialog | null = null
  let running_progress_dialog: Dialog | null = null
  let eval_run_error: KilnError | null = null

  let eval_complete_count = 0
  let eval_total_count = 0
  let eval_error_count = 0

  function run_eval(): boolean {
    if (eval_type === "run_config" && !current_eval_config_id) {
      eval_run_error = new KilnError(
        "Select all options needed to run the eval.",
        null,
      )
      eval_state = "complete_with_errors"
      // True to close the run dialog, and then show the error in the progress dialog
      running_progress_dialog?.show()
      return true
    }

    if (
      eval_type === "run_config" &&
      !run_all &&
      (!run_config_ids || run_config_ids.length === 0)
    ) {
      eval_run_error = new KilnError(
        "Select at least one run config to run the eval.",
        null,
      )
      eval_state = "complete_with_errors"
      // True to close the run dialog, and then show the error in the progress dialog
      running_progress_dialog?.show()
      return true
    }

    eval_state = "running"
    eval_complete_count = 0
    eval_total_count = 0
    eval_error_count = 0

    eval_run_error = null

    posthog.capture("run_eval", {
      eval_type: eval_type,
      run_all: run_all,
    })

    // fetch rather than EventSource: these endpoints refuse up front with a 4xx that
    // names the reason (no golden set, no such split), and EventSource cannot read the
    // status or body of a non-200 — every refusal reached the user as "Unknown error".
    const stream = stream_sse(run_url, {
      subject: "eval stream",
      on_message: (data) => {
        try {
          if (data === "complete") {
            // Special end message
            stream.close()
            eval_state =
              eval_error_count > 0 ? "complete_with_errors" : "complete"

            on_run_complete()
          } else {
            const parsed = JSON.parse(data)
            eval_complete_count = parsed.progress
            eval_total_count = parsed.total
            eval_error_count = parsed.errors
            eval_state = "running"
          }
        } catch (error) {
          stream.close()
          eval_run_error = createKilnError(error)
          console.error(eval_run_error)
          eval_state = "complete_with_errors"
          on_run_complete()
        }
      },
      on_error: (error) => {
        eval_state = "complete_with_errors"
        eval_run_error = createKilnError(error)
        on_run_complete()
      },
    })

    // Switch over to the progress dialog, closing the run dialog
    running_progress_dialog?.show()
    return true
  }

  // Returns false so the dialog isn't closed
  function re_run_eval(): boolean {
    run_eval()
    return false
  }

  function run_dialog_buttons(eval_state: string) {
    let buttons = []

    if (eval_state === "complete" || eval_state === "complete_with_errors") {
      buttons.push({
        label: "Close",
        isCancel: true,
        isPrimary: false,
      })
    }

    if (eval_state === "complete_with_errors") {
      buttons.push({
        label: "Re-run Eval",
        isPrimary: true,
        action: re_run_eval,
      })
    }

    return buttons
  }

  function run_button_style_class() {
    if (btn_size === "small") {
      return "btn-sm rounded-full"
    } else if (btn_size === "mid") {
      return "btn-mid"
    } else if (btn_size === "xs") {
      return "btn-xs rounded-full"
    }
    return ""
  }
</script>

{#if eval_state === "not_started"}
  <button
    class="btn {run_button_style_class()} {btn_primary
      ? 'btn-primary'
      : 'btn-outline'} whitespace-nowrap {btn_class}"
    on:click={() => {
      run_dialog?.show()
    }}>Run {run_all ? "All Evals" : "Eval"}</button
  >
{:else}
  <button
    class="btn {run_button_style_class()} whitespace-nowrap {btn_class}"
    on:click={() => {
      running_progress_dialog?.show()
    }}
  >
    {#if eval_state === "running"}
      <div class="loading loading-spinner loading-xs"></div>
      Running...
    {:else if eval_state === "complete"}
      Eval Complete
    {:else if eval_state === "complete_with_errors"}
      Eval Errors
    {:else}
      Eval Status
    {/if}
  </button>
{/if}

<Dialog
  bind:this={running_progress_dialog}
  title=""
  action_buttons={run_dialog_buttons(eval_state)}
>
  <div
    class="mt-12 mb-6 flex flex-col items-center justify-center min-h-[100px] text-center"
  >
    {#if eval_state === "complete" && eval_complete_count == 0}
      <div class="font-medium">No Data Needed to be Evaluated</div>
      <div class="text-gray-500 text-sm mt-2 flex flex-col gap-2">
        <div>
          If you want to add more data to your eval,
          <a
            href="https://docs.kiln.tech/docs/evaluations#create-your-eval-datasets"
            target="_blank"
            class="link">read the docs</a
          > for instructions.
        </div>
      </div>
    {:else if eval_state === "complete"}
      <div class="font-medium">Eval Complete 🎉</div>
    {:else if eval_state === "complete_with_errors"}
      <div class="font-medium">Eval Complete with Errors</div>
    {:else if eval_state === "running"}
      <div class="loading loading-spinner loading-lg text-success"></div>
      <div class="font-medium mt-4">Running...</div>
    {/if}
    <div class="text-sm font-light min-w-[120px]">
      {#if eval_total_count > 0}
        <div>
          {eval_complete_count + eval_error_count} of {eval_total_count}
        </div>
      {/if}
      {#if eval_error_count > 0}
        <div class="text-error font-light text-xs">
          {eval_error_count} error{eval_error_count === 1 ? "" : "s"}
        </div>
      {/if}
      {#if eval_run_error}
        <div class="text-error font-light text-xs mt-2">
          {eval_run_error.getMessage() || "An unknown error occurred"}
        </div>
      {/if}
    </div>
  </div>
</Dialog>

<Dialog
  bind:this={run_dialog}
  title="Run Eval"
  action_buttons={[
    {
      label: "Cancel",
      isCancel: true,
    },
    {
      label: "Run Eval",
      action: run_eval,
      isPrimary: true,
    },
  ]}
>
  <div class="flex flex-col gap-2 font-light mt-4">
    <div>Run this eval with the selected configuration?</div>
    <div>Don't close this page if you want to monitor progress.</div>
    <Warning
      warning_color="warning"
      warning_message="This may use considerable compute/credits."
      inline={true}
    />
  </div>
</Dialog>
