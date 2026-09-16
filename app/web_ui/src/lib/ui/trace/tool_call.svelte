<script lang="ts">
  import type { ToolCallMessageParam } from "$lib/types"
  import Output from "$lib/ui/output.svelte"
  import { kiln_task_tool_server_id } from "$lib/stores/tools_store"

  export let tool_call: ToolCallMessageParam
  export let nameTag: string = "Tool Name"
  export let project_id: string | undefined = undefined
  export let persistent_tool_id: string | undefined = undefined
  // A citation's span within the arguments, in the coordinates of the raw
  // argument string. Output translates it onto the printed JSON.
  export let arguments_mark: { start: number; end: number } | null = null

  function get_tool_link(): string | null {
    if (!project_id) return null

    // If we have a persistent_tool_id, try to create a specific link
    const tool_server_id = persistent_tool_id
      ? kiln_task_tool_server_id(persistent_tool_id)
      : null
    if (tool_server_id) {
      return `/tools/${project_id}/kiln_task/${tool_server_id}`
    }

    return null
  }
</script>

<div class="grid grid-cols-[auto,1fr] gap-x-3 gap-y-0 text-xs">
  <div class="font-medium text-gray-500">{nameTag}:</div>
  <div class="font-mono">
    {#if get_tool_link()}
      <a href={get_tool_link()} class="text-gray-500 link" target="_blank">
        {tool_call.function.name}
      </a>
    {:else}
      {tool_call.function.name}
    {/if}
  </div>
  <div class="font-medium text-gray-500">Arguments:</div>
  <Output
    raw_output={tool_call.function.arguments}
    no_padding={true}
    mark={arguments_mark}
  />
</div>
