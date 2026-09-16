<script lang="ts">
  import { stubState } from "./stub_state"
  import type { Trace } from "$lib/types"

  export let trace: Trace = []
  export let project_id: string | undefined = undefined
  export let forkable_run_ids: (string | null)[] | undefined = undefined
  export let truncate_at_trace_index: number | null = null
  export let on_fork:
    | ((run_id: string, trace_index: number) => void)
    | undefined = undefined
  export let show_per_message_usage: boolean = false

  // Surface the props the test cares about on every render.
  $: stubState.chatTrace = { truncate_at_trace_index, on_fork }
  $: void [trace, project_id, forkable_run_ids, show_per_message_usage]
</script>

<div
  data-testid="chat-trace-stub"
  data-truncate={truncate_at_trace_index ?? ""}
></div>
