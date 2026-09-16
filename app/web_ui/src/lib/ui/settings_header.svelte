<script lang="ts">
  export let title: string
  // A caller that needs a link or other markup on the subtitle line fills the
  // `subtitle` slot instead of this string; with no slot the render is
  // unchanged.
  export let subtitle: string | null = null
  // The rule under the header. A screen that stacks many headers reads as a
  // ladder of lines rather than as sections, so those callers turn it off. The
  // padding goes with it: with no rule to clear, the gap below the header is
  // the parent stack's to own.
  export let show_divider = true
</script>

<div class={show_divider ? "pb-3 border-b border-gray-200" : ""}>
  {#if $$slots.actions}
    <!-- Actions ride on the title's line, right-aligned and bottom-aligned
         with it, inside the header's own rule so the rule still runs the full
         width. The title markup is repeated rather than wrapped for every
         caller: a header with no actions renders exactly what it always did. -->
    <div class="flex items-end justify-between gap-3">
      <div class="min-w-0">
        <h2 class="text-lg font-medium text-gray-900">{title}</h2>
        {#if $$slots.subtitle}
          <p class="text-sm text-gray-500"><slot name="subtitle" /></p>
        {:else if subtitle}
          <p class="text-sm text-gray-500">{subtitle}</p>
        {/if}
      </div>
      <!-- The control spaces the actions, so two headers with two buttons
           never drift apart at their call sites. -->
      <div class="flex items-center gap-2 flex-none">
        <slot name="actions" />
      </div>
    </div>
  {:else}
    <h2 class="text-lg font-medium text-gray-900">{title}</h2>
    {#if $$slots.subtitle}
      <p class="text-sm text-gray-500"><slot name="subtitle" /></p>
    {:else if subtitle}
      <p class="text-sm text-gray-500">{subtitle}</p>
    {/if}
  {/if}
</div>
