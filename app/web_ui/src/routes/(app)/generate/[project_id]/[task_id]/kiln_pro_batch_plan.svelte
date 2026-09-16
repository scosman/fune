<script lang="ts">
  import KilnProPlanSummary from "./kiln_pro_plan_summary.svelte"
  import KilnProPromptsTable from "./kiln_pro_prompts_table.svelte"
  import SettingsHeader from "$lib/ui/settings_header.svelte"

  export let plan: { prompts: string[]; summary: string }
  export let on_generate_inputs: () => void
  export let on_regenerate: () => void
  export let on_delete_prompt: (index: number) => void
  export let summary_out_of_sync = false
  // Optional override for the generate button's label. The eval builder's
  // click starts a full conversation drive (long, paid), not quick sample
  // generation — its label must say so. Default keeps /generate unchanged.
  export let generate_button_label: string | null = null
  // The eval builder hides the generate button once the exact current plan
  // already has driven results — continuing to those results is the only
  // forward action there. Default keeps /generate unchanged.
  export let hide_generate_button = false
  // Renders the generate button as an outline primary. The eval builder turns
  // this on for the screens where another solid primary is already on the
  // page, so only one solid primary shows at a time. Default keeps /generate
  // unchanged.
  export let generate_button_outline = false
  // Header and its sub-line. The eval builder overrides the header (what it
  // lists is a proposed eval dataset) and passes its own one-line sub-line;
  // what the next step does to each row belongs on that step, not here. It
  // renames the rows through the items_label / expanded_description props
  // below.
  export let header_label = "Batch Plan"
  export let subheader =
    "Review the plan for generating your synthetic data batch."
  // The rule under the header. The eval builder's screens stack headers and
  // read as a ladder of lines with it on, so that flow turns it off. Default
  // keeps /generate unchanged.
  export let show_header_divider = true
  // Passed straight to the prompts table: the noun for the plan's rows (which
  // drives its header and aria-label together), the sentence it shows when
  // expanded, and the header over the rows' first column. The defaults are
  // /generate's strings, so that flow passes nothing.
  export let items_label = "Dataset Items"
  export let expanded_description: string | null | false = null
  export let column_label = "Prompt"

  $: count = plan.prompts.length

  // No confirm here — each parent decides whether and when regenerating
  // needs confirmation (e.g. before discarding driven results).
</script>

<div class="flex flex-col gap-4 mt-12">
  <!-- The house section header carries the title, the sub-line and the two
  actions, so this surface reads like every other section header in the app.
  The optional per-consumer clause (the eval builder's note that the plan used
  the task's Data Guide) rides on the same subtitle line; nothing renders with
  no consumer content. That line is a paragraph, so the slot's content has to
  be phrasing content (spans, links, plain text) and never a block element. -->
  <SettingsHeader title={header_label} show_divider={show_header_divider}>
    <svelte:fragment slot="subtitle"
      >{subheader}<slot name="under_subheader" /></svelte:fragment
    >
    <svelte:fragment slot="actions">
      <button class="btn btn-md" on:click={on_regenerate}>Refine Plan</button>
      {#if !hide_generate_button}
        <button
          class="btn btn-md {generate_button_outline
            ? 'btn-outline btn-primary'
            : 'btn-primary'}"
          disabled={count === 0}
          on:click={on_generate_inputs}
        >
          {generate_button_label ?? `Generate Batch (${count})`}
        </button>
      {/if}
    </svelte:fragment>
  </SettingsHeader>
  <!-- Optional per-consumer secondary action (the eval builder's
  model-settings link), right-aligned under the primary-button cluster.
  Guarded so with no consumer filling the slot NOTHING renders and
  /generate's output stays byte-identical. -->
  {#if $$slots.advanced}
    <div class="flex justify-end -mt-3">
      <slot name="advanced" />
    </div>
  {/if}

  <KilnProPlanSummary
    summary={plan.summary}
    out_of_sync={summary_out_of_sync}
  />
  <KilnProPromptsTable
    prompts={plan.prompts}
    on_delete={on_delete_prompt}
    {items_label}
    {expanded_description}
    {column_label}
  />
</div>
