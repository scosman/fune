<script lang="ts">
  import { stubState } from "./stub_state"

  export let mode: string = "append"
  export let busy: boolean = false
  export let parent_task_run_id: string | null = null
  export let on_send_start: ((text: string) => void) | undefined = undefined
  export let on_success:
    | ((new_run_id: string) => void | Promise<void>)
    | undefined = undefined
  export let on_send_settled: ((ok: boolean) => void) | undefined = undefined
  export let on_cancel: (() => void) | undefined = undefined
  // Accept and ignore the remaining props the page passes.
  export let project_id: string = ""
  export let task_id: string = ""
  export let run_config_component: unknown = null
  export let prefill_text: string = ""
  export let forked_turn_index: number | undefined = undefined

  $: stubState.composers[mode] = {
    mode,
    busy,
    parent_task_run_id,
    on_send_start,
    on_success,
    on_send_settled,
  }
  $: void [
    on_cancel,
    project_id,
    task_id,
    run_config_component,
    prefill_text,
    forked_turn_index,
  ]
</script>

<div data-testid={"composer-" + mode} data-busy={busy}></div>
