<script lang="ts">
  import ChatMarkdown from "$lib/ui/chat/chat_markdown.svelte"
  import StarsIcon from "$lib/ui/icons/stars_icon.svelte"
  import Warning from "$lib/ui/warning.svelte"

  export let summary: string
  // Set once the user edits the plan (e.g. deletes prompts) — the summary is
  // not regenerated, so we flag that it may no longer match.
  export let out_of_sync = false
</script>

<div class="rounded-lg border p-6">
  <div class="flex items-center justify-between gap-2 mb-3">
    <div
      class="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-primary"
    >
      <span
        class="flex items-center justify-center rounded-[10px] flex-none w-7 h-7 p-[6px] bg-blue-50 text-[#628BD9]"
      >
        <StarsIcon />
      </span>
      Overview
    </div>
    {#if out_of_sync}
      <Warning
        warning_message="Dataset items edited. Overview may be out of date."
        warning_color="warning"
        warning_icon="exclaim"
        text_size="xs"
        inline={true}
      />
    {/if}
  </div>
  <ChatMarkdown text={summary} />
</div>
