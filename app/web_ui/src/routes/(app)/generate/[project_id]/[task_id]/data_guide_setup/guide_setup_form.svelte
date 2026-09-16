<script lang="ts">
  import { createEventDispatcher } from "svelte"
  import FormContainer from "$lib/utils/form_container.svelte"
  import { KilnError } from "$lib/utils/error_handlers"
  import type { KilnAgentRunConfigProperties } from "$lib/types"
  import { isKilnAgentRunConfig } from "$lib/types"
  import TableActionMenu from "$lib/ui/table_action_menu.svelte"
  import RunOptionsTiles from "./run_options_tiles.svelte"
  import GenerationSettingsTrigger from "./generation_settings_trigger.svelte"
  import AddExampleDialog, {
    type GuideSample,
  } from "$lib/components/add_example_dialog.svelte"
  import Warning from "$lib/ui/warning.svelte"
  import ClampedText from "$lib/ui/clamped_text.svelte"
  import SeeAllDialog from "$lib/ui/see_all_dialog.svelte"
  import { formatExpandedContent } from "$lib/utils/format_expanded_content"

  export let project_id: string
  export let task_id: string

  // page_error is exported so the parent can surface async errors (e.g. a
  // failed preview API call) inline above the submit button instead of in a
  // separate top-level banner. Cleared by handle_continue on each new attempt.
  export let page_error: KilnError | null = null
  let page_submitting: boolean = false

  // Unified examples list (manual + existing + saved golden)
  export let guide_examples: GuideSample[] = []

  // Build the full input data guide markdown from the user's examples — only
  // the `# Reference Inputs` section. Rules are intentionally not collected
  // from the user here — the metaprompter generates them from refine
  // feedback on the first refine pass.
  function build_guide_md(): string {
    const valid_examples = guide_examples.filter((e) => (e.input ?? "").trim())
    if (valid_examples.length === 0) {
      return ""
    }
    const examples_body = valid_examples
      .map(
        (e, i) => `## Example ${i + 1}\n\`\`\`input\n${e.input ?? ""}\n\`\`\``,
      )
      .join("\n\n")
    return `# Reference Inputs\n\n${examples_body}`
  }

  // --- Example management ---
  let add_example_dialog: AddExampleDialog
  function open_add_example_dialog() {
    add_example_dialog?.open_add()
  }
  function open_edit_example_dialog(index: number) {
    add_example_dialog?.open_edit(guide_examples[index], index)
  }
  function handle_example_submit(
    event: CustomEvent<{
      sample: GuideSample
      index: number
      mode: "add" | "edit"
    }>,
  ) {
    const { sample, index, mode } = event.detail
    if (mode === "edit" && index >= 0) {
      guide_examples[index] = sample
      guide_examples = guide_examples
    } else {
      guide_examples = [...guide_examples, sample]
    }
  }
  function remove_example(index: number) {
    guide_examples = guide_examples.filter((_, i) => i !== index)
  }

  $: has_examples = guide_examples.length > 0

  function open_generation_options() {
    run_options_tiles?.open_combined_dialog()
  }

  // --- Run options (per-stage) ---
  // Bound to the shared RunOptionsTiles instance so we can pull the configured
  // run configs at submit time.
  let run_options_tiles: RunOptionsTiles | null = null
  // Friendly names of the selected generation model + provider, for the trigger.
  let generation_model_name = ""
  let generation_provider = ""

  function handle_continue() {
    // FormContainer flips submitting=true before dispatching submit, and expects
    // us to flip it back if we don't proceed. Otherwise the Continue button
    // keeps spinning after a synchronous validation failure.
    page_error = null
    try {
      const valid_examples = guide_examples.filter((e) =>
        (e.input ?? "").trim(),
      )
      if (valid_examples.length === 0) {
        page_error = new KilnError("At least one example is required.")
        return
      }
      const input_run_config = run_options_tiles?.get_input_run_config()
      if (!input_run_config) {
        page_error = new KilnError(
          "Please select a model for input generation.",
          null,
        )
        return
      }
      if (!isKilnAgentRunConfig(input_run_config)) {
        page_error = new KilnError(
          "Data Guide requires a kiln_agent run config.",
          null,
        )
        return
      }
      dispatch("generate_preview", {
        guide: build_guide_md(),
        input_run_config,
      })
    } finally {
      page_submitting = false
    }
  }

  // --- Events ---
  const dispatch = createEventDispatcher<{
    generate_preview: {
      guide: string
      input_run_config: KilnAgentRunConfigProperties
    }
  }>()

  let see_all_dialog: SeeAllDialog
</script>

<FormContainer
  submit_label="Continue"
  on:submit={handle_continue}
  bind:error={page_error}
  bind:submitting={page_submitting}
  submit_disabled={!has_examples}
  compact_button={true}
  warn_before_unload={has_examples}
>
  <!-- Example Inputs Section -->
  <div class="flex flex-col gap-2">
    <div class="flex items-center justify-between">
      <div>
        <div class="font-medium">Example Inputs</div>
        <div class="text-sm text-gray-500">
          Example inputs to guide synthetic input generation.
        </div>
      </div>
      <button
        class="btn btn-sm {has_examples
          ? 'btn-outline btn-primary'
          : 'btn-primary'}"
        on:click={open_add_example_dialog}
        type="button">+ Add Example</button
      >
    </div>

    {#if guide_examples.length > 0}
      <div class="rounded-lg border">
        <table class="table table-fixed">
          <thead>
            <tr>
              <th>Input</th>
              <th style="width: 50px"></th>
            </tr>
          </thead>
          <tbody>
            {#each guide_examples as example, i}
              {@const input_content = formatExpandedContent(
                example.input ?? "",
              )}
              <tr>
                <td class="py-2">
                  <ClampedText
                    content={input_content.isJson ? "" : input_content.value}
                    html_content={input_content.isJson
                      ? input_content.value
                      : null}
                    on:see_all={() =>
                      see_all_dialog.show("Input", example.input ?? "")}
                  />
                </td>
                <td class="py-2 p-0">
                  <div class="dropdown dropdown-end dropdown-hover">
                    <TableActionMenu
                      items={[
                        {
                          label: "Edit",
                          onclick: () => open_edit_example_dialog(i),
                        },
                        { label: "Remove", onclick: () => remove_example(i) },
                      ]}
                    />
                  </div>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {:else}
      <div
        class="rounded-lg border border-dashed border-gray-300 p-8 text-center text-sm text-gray-400"
      >
        No example inputs
      </div>
    {/if}
  </div>

  <RunOptionsTiles
    bind:this={run_options_tiles}
    bind:selected_model_name_display={generation_model_name}
    bind:selected_provider_display={generation_provider}
    {project_id}
  />
  {#if !has_examples}
    <div class="flex justify-end">
      <Warning
        warning_message="Please provide at least one example."
        warning_color="error"
        warning_icon="exclaim"
        inline={true}
      />
    </div>
  {/if}
  <GenerationSettingsTrigger
    model_name={generation_model_name}
    provider={generation_provider}
    open={open_generation_options}
  />
</FormContainer>

<AddExampleDialog
  bind:this={add_example_dialog}
  {project_id}
  {task_id}
  include_output={false}
  existing_examples={guide_examples}
  on:submit={handle_example_submit}
/>

<SeeAllDialog bind:this={see_all_dialog} />
