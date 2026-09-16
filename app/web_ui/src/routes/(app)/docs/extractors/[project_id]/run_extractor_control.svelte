<script lang="ts">
  import Dialog from "$lib/ui/dialog.svelte"
  import { extractorProgressStore } from "$lib/stores/extractor_progress_store"
  import type { ExtractorConfig } from "$lib/types"
  import Warning from "$lib/ui/warning.svelte"

  export let btn_size: "normal" | "mid" = "mid"
  export let project_id: string
  export let extractor_config: ExtractorConfig

  $: extractor_config_id = extractor_config.id || ""

  let run_confirm_dialog: Dialog | null = null

  let progress_dialog: Dialog | null = null
  export function show() {
    progress_dialog?.show()
  }

  export function close() {
    progress_dialog?.close()
    return true
  }

  function run_dialog_buttons(state: string) {
    let buttons = []

    if (state === "complete") {
      buttons.push({
        label: "Close",
        isCancel: true,
        isPrimary: false,
      })
    }

    if (state === "incomplete" || state === "completed_with_errors") {
      buttons.push({
        label: "Re-run",
        isPrimary: true,
        action: () => {
          extractorProgressStore.run_extractor(project_id, extractor_config_id)
          return false
        },
      })
    }

    return buttons
  }

  $: error_count =
    $extractorProgressStore.progress[extractor_config_id]?.error || 0
  $: total_count =
    $extractorProgressStore.progress[extractor_config_id]?.total || 0
  $: success_count =
    $extractorProgressStore.progress[extractor_config_id]?.success || 0

  function get_state_to_label(status: string) {
    const state_to_label: Record<
      string,
      {
        label: string
        icon: "play" | "loading" | null
        action: () => void
        isPrimary: boolean
      } | null
    > = {
      not_started: {
        label: "Run",
        icon: "play",
        isPrimary: true,
        action: () => {
          run_confirm_dialog?.show()
        },
      },
      running: {
        label: "Running",
        icon: "loading",
        isPrimary: false,
        action: () => {
          progress_dialog?.show()
        },
      },
      completed_with_errors: {
        label: "View Progress",
        icon: null,
        isPrimary: false,
        action: () => {
          progress_dialog?.show()
        },
      },
      complete: null,
      incomplete: {
        label: "Retry",
        icon: "play",
        isPrimary: true,
        action: () => {
          extractorProgressStore.run_extractor(project_id, extractor_config_id)
          return false
        },
      },
    }

    return state_to_label[status]
  }

  $: status = $extractorProgressStore.status[extractor_config_id]
  $: disabled =
    extractor_config.is_archived ||
    $extractorProgressStore.status[extractor_config_id] === "complete"
  $: button_state = get_state_to_label(status)
</script>

{#if button_state}
  <button
    class="btn {btn_size === 'mid'
      ? 'btn-mid'
      : ''} whitespace-nowrap {button_state?.isPrimary ? 'btn-primary' : ''}"
    on:click={(event) => {
      event.stopPropagation()
      button_state?.action()
    }}
    {disabled}
  >
    {#if button_state?.icon === "loading"}
      <div class="loading loading-spinner loading-xs"></div>
    {:else if button_state?.icon === "play"}
      <!-- Attribution: https://www.svgrepo.com/svg/526106/play -->
      <svg
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        class="w-4 h-4"
        ><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g
          id="SVGRepo_tracerCarrier"
          stroke-linecap="round"
          stroke-linejoin="round"
        ></g><g id="SVGRepo_iconCarrier">
          <path
            d="M21.4086 9.35258C23.5305 10.5065 23.5305 13.4935 21.4086 14.6474L8.59662 21.6145C6.53435 22.736 4 21.2763 4 18.9671L4 5.0329C4 2.72368 6.53435 1.26402 8.59661 2.38548L21.4086 9.35258Z"
            fill="currentColor"
          ></path>
        </g></svg
      >
    {/if}
    {button_state?.label}
  </button>
{/if}

<Dialog
  bind:this={run_confirm_dialog}
  title="Extract All Documents?"
  action_buttons={[
    { label: "Cancel", isCancel: true },
    {
      label: "Extract All",
      action: () => {
        extractorProgressStore.run_extractor(project_id, extractor_config_id)
        run_confirm_dialog?.close()
        return false
      },
      isPrimary: true,
    },
  ]}
>
  <div class="flex flex-col gap-2 font-light mt-4">
    <div>This may take a while, depending on the number of documents.</div>
    <div>Leave the page open while extraction is running.</div>
    <Warning
      warning_color="warning"
      warning_message="This may use considerable compute/credits."
      inline={true}
    />
  </div>
</Dialog>

<Dialog
  bind:this={progress_dialog}
  title=""
  action_buttons={run_dialog_buttons(status)}
>
  <div
    class="mt-12 mb-6 flex flex-col items-center justify-center min-h-[100px] text-center"
  >
    {#if status === "complete"}
      <div class="font-medium">Extraction Complete 🎉</div>
    {:else if status === "incomplete"}
      <div class="font-medium">Extraction Incomplete</div>
    {:else if status === "running"}
      <div class="loading loading-spinner loading-lg text-success"></div>
      <div class="font-medium mt-4">Running...</div>
    {:else if status === "completed_with_errors"}
      <div class="font-medium">Extraction Completed with Errors</div>
    {:else if status === "not_started"}
      <div class="font-medium">Extraction Not Started</div>
    {/if}
    <div class="text-sm font-light min-w-[120px]">
      {#if total_count > 0}
        <div>
          Completed {success_count} of {total_count}
        </div>
      {/if}
      {#if error_count > 0}
        <div class="text-error font-light text-xs">
          {error_count} error{error_count === 1 ? "" : "s"}
        </div>
      {/if}
    </div>
  </div>
</Dialog>
