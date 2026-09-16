<script lang="ts">
  import Dialog from "$lib/ui/dialog.svelte"
  import Warning from "$lib/ui/warning.svelte"
  import { createKilnError, type KilnError } from "$lib/utils/error_handlers"
  import { addCodeTrust } from "$lib/api/v2_eval_api"

  export let project_id: string
  export let on_trust_granted: () => void

  let dialog: Dialog
  let trust_error: KilnError | null = null

  export function show() {
    trust_error = null
    dialog.show()
  }

  async function grant_trust_and_proceed(): Promise<boolean> {
    try {
      await addCodeTrust(project_id)
    } catch (e) {
      trust_error = createKilnError(e)
      return false
    }
    on_trust_granted()
    return true
  }
</script>

<Dialog
  bind:this={dialog}
  title="Trust Code and Project?"
  action_buttons={[
    {
      label: "Run — I Trust This Code",
      isWarning: true,
      asyncAction: grant_trust_and_proceed,
    },
  ]}
>
  <div class="flex flex-row items-start gap-4">
    <svg
      class="w-10 h-10 text-warning flex-none"
      fill="currentColor"
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M128,20.00012a108,108,0,1,0,108,108A108.12217,108.12217,0,0,0,128,20.00012Zm0,192a84,84,0,1,1,84-84A84.0953,84.0953,0,0,1,128,212.00012Zm-12-80v-52a12,12,0,1,1,24,0v52a12,12,0,1,1-24,0Zm28,40a16,16,0,1,1-16-16A16.018,16.018,0,0,1,144,172.00012Z"
      />
    </svg>
    <div class="flex flex-col gap-2 text-sm text-left">
      <p>
        This project wants to run Python code on your machine. Only proceed if
        you trust the code and this project.
      </p>
      <p class="font-bold">Never paste code from a stranger or the internet.</p>
    </div>
  </div>
  {#if trust_error}
    <div class="mt-3">
      <Warning
        warning_color="error"
        inline={true}
        warning_message={trust_error.getMessage()}
      />
    </div>
  {/if}
</Dialog>
