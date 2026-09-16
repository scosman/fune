<script context="module" lang="ts">
  // The saved guide opens in a new tab wherever it is linked from a form:
  // the form's state (a batch being planned, a dialog mid-edit) would not
  // survive navigating away from it.
  export function open_data_guide_in_new_tab(
    project_id: string,
    task_id: string,
  ) {
    if (!project_id || !task_id) return
    window.open(
      `/generate/${project_id}/${task_id}/data_guide`,
      "_blank",
      "noopener,noreferrer",
    )
  }
</script>

<script lang="ts">
  // The "Use Data Guide" checkbox: renders only when the task has a saved
  // guide, and binds the on/off state the surrounding form sends with its
  // generation requests. Shared by synthetic data generation and the eval
  // builder, so it takes the guide text and the state directly rather than
  // either surface's data model.
  import FormElement from "$lib/utils/form_element.svelte"

  export let project_id: string
  export let task_id: string
  export let data_guide: string
  export let use_data_guide: boolean
</script>

{#if data_guide}
  <div class="flex flex-col gap-1">
    <FormElement
      id="data_guide_toggle"
      label="Use Data Guide"
      description="A saved description of what realistic inputs to this task look like, so generated inputs match the structure and style of your real data."
      inputType="checkbox"
      bind:value={use_data_guide}
      inline_action={{
        handler: () => open_data_guide_in_new_tab(project_id, task_id),
        label: "View",
      }}
    />
  </div>
{/if}
