<script lang="ts">
  // Binds info_description so a test can read what the control writes into
  // it. The control writes its own default copy into that prop, which is the
  // write a caller's text has to survive.
  import type { Task } from "$lib/types"
  import SavedRunConfigsDropdown from "../saved_run_configs_dropdown.svelte"

  export let project_id: string
  export let current_task: Task
  export let info_description: string = ""
</script>

<SavedRunConfigsDropdown {project_id} {current_task} bind:info_description />
<div data-testid="tooltip_copy">{info_description}</div>
