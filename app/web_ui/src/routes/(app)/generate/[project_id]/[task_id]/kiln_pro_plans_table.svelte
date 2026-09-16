<script lang="ts">
  import TableActionMenu from "$lib/ui/table_action_menu.svelte"

  export let prompts: string[]
  // When provided, each row gets a delete action.
  export let on_delete: ((index: number) => void) | null = null
  // When provided, each row shows its status (e.g. what became of the prompt
  // once inputs were generated). Parallel to `prompts`.
  export let statuses: string[] | null = null
  // Header for the first column.
  export let column_label: string

  $: reserved_width = (statuses ? 140 : 0) + (on_delete ? 40 : 0)
</script>

<table class="table table-fixed">
  <thead>
    <tr>
      <!-- The first column's header is caller-named (column_label); the rest
           are fixed: status is 140 and the action menu 40, matching the
           samples table. -->
      <th style="width: calc(100% - {reserved_width}px)">{column_label}</th>
      {#if statuses}
        <th style="width: 140px">Status</th>
      {/if}
      {#if on_delete}
        <th style="width: 40px"></th>
      {/if}
    </tr>
  </thead>
  <tbody>
    {#each prompts as prompt, i}
      <tr>
        <td class="py-2 align-top whitespace-normal">{prompt}</td>
        {#if statuses}
          <td class="py-2 align-top">{statuses[i] ?? ""}</td>
        {/if}
        {#if on_delete}
          <td class="p-0 align-top">
            <TableActionMenu
              width="w-40"
              items={[
                {
                  label: "Remove",
                  onclick: () => on_delete?.(i),
                },
              ]}
            />
          </td>
        {/if}
      </tr>
    {/each}
  </tbody>
</table>
