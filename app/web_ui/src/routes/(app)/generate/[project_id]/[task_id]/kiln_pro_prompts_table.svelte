<script lang="ts">
  import KilnProPlansTable from "./kiln_pro_plans_table.svelte"

  export let prompts: string[]
  // When provided, each row gets a delete action.
  export let on_delete: ((index: number) => void) | null = null
  // The noun for the rows. One prop feeds both the visible header and the
  // toggle's aria-label, so what a screen reader announces can never drift
  // from what is on screen.
  export let items_label: string
  // The sentence under the header while expanded: null keeps the /generate
  // sentence, false renders no description, a string renders that string.
  // The render gate is falsy, so an empty string behaves the same as false.
  export let expanded_description: string | null | false
  // Passed straight to the rows table: the header over its first column.
  export let column_label: string

  const DEFAULT_EXPANDED_DESCRIPTION =
    "Each prompt below will be used to guide one dataset sample."

  let show_prompts = false
  $: count = prompts.length
  $: description =
    expanded_description === null
      ? DEFAULT_EXPANDED_DESCRIPTION
      : expanded_description
  // Assistive text uses the same noun as the header, lowercased mid-sentence.
  $: toggle_label = `${show_prompts ? "Hide" : "Show"} ${items_label.toLowerCase()}`
</script>

<div class="rounded-lg border">
  <button
    class="w-full flex items-center justify-between px-4 py-3 text-left"
    aria-label={toggle_label}
    aria-expanded={show_prompts}
    on:click={() => (show_prompts = !show_prompts)}
  >
    <div class="flex flex-col gap-2">
      <span class="text-sm font-medium">All {items_label} ({count})</span>
      {#if show_prompts && description}
        <div class="text-sm text-gray-500">
          {description}
        </div>
      {/if}
    </div>
    <div class="flex items-center text-sm text-gray-500">
      <svg
        class="w-4 h-4 transition-transform {show_prompts ? 'rotate-180' : ''}"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <polyline points="6 9 12 15 18 9" />
      </svg>
    </div>
  </button>

  {#if show_prompts}
    <KilnProPlansTable {prompts} {on_delete} {column_label} />
  {/if}
</div>
