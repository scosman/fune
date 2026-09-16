<script lang="ts">
  import { client } from "$lib/api_client"
  import { page } from "$app/stores"
  import type { TurnMode } from "$lib/types"
  import Dialog from "$lib/ui/dialog.svelte"

  export let onImportCompleted: () => void
  export let tag_splits: Record<string, number> | null = null
  // The route's task turn mode, threaded in by the owning page. The CSV
  // contract must match the task the import POSTs to (the route's task_id),
  // not the global current-task store, which can be stale or for a different
  // task and would document the wrong columns.
  export let turn_mode: TurnMode | null = null

  let selected_file: File | null = null

  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  $: is_multiturn = turn_mode === "multiturn"
  $: dialog_title = is_multiturn
    ? "Add Multiturn CSV to Dataset"
    : "Add CSV to Dataset"

  function handleFileSelect(event: Event) {
    const input = event.target as HTMLInputElement
    if (input.files && input.files[0]) {
      selected_file = input.files[0]
    }
  }

  async function handleUpload(): Promise<boolean> {
    if (!selected_file) {
      return false
    }

    const formData = new FormData()
    formData.append("file", selected_file)
    if (tag_splits) {
      formData.append("splits", JSON.stringify(tag_splits))
    }

    const { error } = await client.POST(
      "/api/projects/{project_id}/tasks/{task_id}/runs/bulk_upload",
      {
        params: {
          path: { project_id, task_id },
        },
        // Unknown as multipart file uploads aren't supported by the openapi-typescript library
        // see: https://github.com/openapi-ts/openapi-typescript/issues/1214
        body: formData as unknown as { file: string },
      },
    )

    if (error) {
      throw error
    }

    selected_file = null

    onImportCompleted()

    return true
  }

  let dialog: Dialog | null = null

  export function show() {
    dialog?.show()
    selected_file = null
  }

  export function close() {
    dialog?.close()
    selected_file = null
    return true
  }

  function handleCancel() {
    close()
    return true
  }
</script>

<Dialog
  bind:this={dialog}
  title={dialog_title}
  action_buttons={[
    { label: "Cancel", isCancel: true, action: () => handleCancel() },
    {
      label: "Upload",
      asyncAction: () => handleUpload(),
      disabled: !selected_file,
      isPrimary: true,
    },
  ]}
>
  <div class="font-light text-sm">
    <div class="space-y-2">
      {#if is_multiturn}
        <div>
          <p>
            Upload a CSV where each row describes one conversation. The CSV must
            have a header row. The following columns are supported:
          </p>
          <ul class="mb-3 ml-4 mt-3 list-disc">
            <li>
              <code>trace</code> - Required. JSON-encoded list of OpenAI chat
              messages. Assistant messages may include an optional
              <code>reasoning_content</code> field.
            </li>
            <li><code>tags</code> - Optional, comma-separated string.</li>
          </ul>
          <p class="mb-2">Each <code>trace</code> entry is one message:</p>
          <pre class="text-xs bg-base-200 p-2 rounded mb-3 overflow-x-auto">{`[
  {"role": "user", "content": "What is the capital of France?"},
  {"role": "assistant", "content": "Paris."},
  {"role": "user", "content": "And of Germany?"},
  {"role": "assistant", "content": "Berlin."}
]`}</pre>
          <p class="mb-3 text-xs opacity-70">
            Multiturn traces must alternate user/assistant and end with
            assistant. System messages are not supported — set the system prompt
            on the task instead.
          </p>
          <p class="mb-6">
            <a href="/sample_multiturn.csv" download class="link"
              >Download sample CSV</a
            >
          </p>
        </div>
      {:else}
        <div>
          <p>
            Upload a CSV to add each row to your dataset. The CSV must have a
            header row (<a
              href="https://docs.kiln.tech/docs/organizing-datasets"
              target="_blank"
              class="link">see docs</a
            >). The following columns are supported:
          </p>
          <ul class="mb-6 ml-4 mt-3 list-disc">
            <li><code>input</code> - Required</li>
            <li><code>output</code> - Required</li>
            <li><code>reasoning</code> - Optional</li>
            <li><code>chain_of_thought</code> - Optional</li>
            <li><code>tags</code> - Optional, comma-separated string.</li>
          </ul>
        </div>
      {/if}
    </div>
    <input
      type="file"
      class="file-input file-input-bordered w-full"
      on:change={handleFileSelect}
      accept=".csv"
    />
  </div>
</Dialog>
