<script lang="ts">
  // Stub for $lib/ui/trace/chat_trace.svelte. ChatTrace has its own unit tests;
  // these tests only need to see which trace the page handed it.
  import type { Trace } from "$lib/types"

  export let trace: Trace = []
  export let project_id: string | undefined = undefined
  // Referenced so svelte-check does not flag it unused; the page passes it.
  void project_id
</script>

<div data-testid="chat-trace-stub">
  {#each trace as message}
    {@const content =
      "content" in message && typeof message.content === "string"
        ? message.content
        : ""}
    <div>{message.role}: {content}</div>
  {/each}
</div>
