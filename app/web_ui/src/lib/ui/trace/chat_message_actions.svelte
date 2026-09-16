<script lang="ts">
  import ForkIcon from "../icons/fork_icon.svelte"
  import DollarIcon from "../icons/dollar_icon.svelte"

  // Per-message action row (usage, fork). The row always reserves its space
  // below the message so hovering never shifts the conversation around; the
  // icons themselves only fade in on hover/focus of the parent (which must
  // carry the `group` class) — like the Claude desktop app.
  export let align: "start" | "end" = "start"
  export let show_usage: boolean = false
  export let show_fork: boolean = false
  export let on_usage: () => void = () => {}
  export let on_fork: () => void = () => {}
</script>

<!-- Space is reserved (no height animation); only opacity fades the icons in.
     Full width so the align/justify actually positions the icons. -->
<div
  class="flex w-full flex-row items-center gap-1 pt-1 opacity-0 transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100 {align ===
  'end'
    ? 'justify-end'
    : 'justify-start'}"
  data-testid="chat-msg-meta"
>
  {#if show_usage}
    <button
      type="button"
      class="btn btn-xs btn-ghost gap-1 text-gray-400 hover:text-gray-700"
      aria-label="View turn usage"
      on:click={on_usage}
    >
      <span class="block h-4 w-4"><DollarIcon /></span>
      Usage
    </button>
  {/if}
  {#if show_fork}
    <button
      type="button"
      class="btn btn-xs btn-ghost gap-1 text-gray-400 hover:text-gray-700"
      aria-label="Fork from this turn"
      on:click={on_fork}
    >
      <span class="block h-4 w-4"><ForkIcon /></span>
      Fork
    </button>
  {/if}
</div>
