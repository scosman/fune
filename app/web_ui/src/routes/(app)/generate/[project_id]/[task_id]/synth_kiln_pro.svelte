<script lang="ts">
  import { onMount } from "svelte"
  import { get } from "svelte/store"
  import { goto, pushState } from "$app/navigation"
  import { page } from "$app/stores"
  import posthog from "posthog-js"
  import { client } from "$lib/api_client"
  import { checkKilnCopilotAvailable } from "$lib/utils/copilot_utils"
  import RefiningAnimation from "$lib/ui/animations/refining_animation.svelte"
  import FormContainer from "$lib/utils/form_container.svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import RunConfigComponent from "$lib/ui/run_config_component/run_config_component.svelte"
  import { isKilnAgentRunConfig } from "$lib/types"
  import type { KilnAgentRunConfigProperties } from "$lib/types"
  import { createKilnError, KilnError } from "$lib/utils/error_handlers"
  import SynthDataGuide from "./synth_data_guide.svelte"
  import KilnProBatchForm from "./kiln_pro_batch_form.svelte"
  import KilnProBatchPlan from "./kiln_pro_batch_plan.svelte"
  import KilnProInputs from "./kiln_pro_inputs.svelte"
  import { SynthDataGuidanceDataModel } from "./synth_data_guidance_datamodel"

  export let project_id: string
  export let task_id: string
  export let guidance_data: SynthDataGuidanceDataModel
  // Groups every sample from this synth session under one dataset tag.
  export let session_id: string | null = null

  const selected_template = guidance_data.selected_template
  const data_guide_store = guidance_data.data_guide
  const use_data_guide_store = guidance_data.use_data_guide
  $: task = guidance_data.task

  let inputs_dialog: Dialog | null = null

  let current_state: "loading" | "ready" = "loading"

  // Stage within the connected flow.
  let stage: "intro" | "planning" | "plan" | "inputs" = "intro"

  let num_inputs = 50
  // Saved for the session so Regenerate reopens the modal with the user's edits.
  let batch_guidance = ""
  let guidance_initialized = false
  // The dialog stays mounted across reopens, so we own this flag and reset it
  // on submit — otherwise the spinner persists into the next Regenerate.
  let batch_submitting = false

  type BatchPlan = { prompts: string[]; summary: string }
  let plan: BatchPlan | null = null
  // True once the user edits the plan (deletes prompts); the summary isn't
  // regenerated so we flag it as possibly out of date.
  let plan_edited = false
  let plan_error: KilnError | null = null

  // Generate Inputs modal lives here (not in the inputs view) so it opens over
  // the plan page — the page only transitions once the user submits it.
  let inputs_run_config: RunConfigComponent | null = null
  let inputs_submitting = false
  let inputs_error: KilnError | null = null
  let inputs_rcp: KilnAgentRunConfigProperties | null = null
  let inputs_data_guide: string | null = null
  // Set by KilnProInputs: generated samples not yet written to the dataset.
  let unsaved_samples = false

  onMount(async () => {
    let pro_available = false
    try {
      pro_available = await checkKilnCopilotAvailable()
    } catch {
      pro_available = false
    }
    // Pro is required for the batch planner. If the key isn't connected, send
    // the user to the static connect route, which returns here once connected.
    // The static path keeps the Kinde OAuth redirect_uri a fixed, allow-listed
    // URL (the project/task can't be in the path).
    if (!pro_available) {
      goto(`/generate/synth_data_pro_auth`, { replaceState: true })
      return
    }
    current_state = "ready"
  })

  // The template the guidance box started from, so edits can be undone.
  let batch_guidance_template = ""

  // Prefill the guidance box once, as soon as the Generate Batch page shows.
  $: if (current_state === "ready" && !guidance_initialized) {
    batch_guidance_template = guidance_data.kiln_pro_batch_plan_prefill()
    batch_guidance = batch_guidance_template
    guidance_initialized = true
  }

  // Each forward step pushes a history entry tagged with the stage, so the
  // browser back button steps back through the flow. SvelteKit restores
  // $page.state on back; we mirror it into `stage`. The in-app back buttons
  // just call history.back() so both paths go through the same mechanism and
  // the history stack stays consistent.
  function advance_stage(next: "plan" | "inputs") {
    stage = next
    pushState("", { kiln_pro_stage: next })
  }

  function go_back() {
    history.back()
  }

  $: sync_stage_from_history(
    ($page.state as Record<string, unknown>)?.kiln_pro_stage as
      | string
      | undefined,
  )

  function sync_stage_from_history(hist_stage: string | undefined) {
    // "planning" is transient and owned by submit_batch — don't fight it.
    if (stage === "planning") return

    // Going back is destructive at both stages, and every route out of them —
    // the in-app Back button, "Refine Plan", and the browser's own back
    // button — pops history. So confirm here, once, at the point the pop
    // actually happens. popstate can't be cancelled, so on "stay" we push the
    // entry back to re-sync history with the UI the user is still looking at.
    const leaving = (from: string) => stage === from && hist_stage !== from
    let confirm_msg: string | null = null
    if (leaving("inputs") && unsaved_samples) {
      confirm_msg =
        "You have generated samples that aren't saved to your dataset. Going back will discard them. This cannot be undone."
    } else if (leaving("plan")) {
      confirm_msg = plan_edited
        ? "Are you sure you want to discard the current batch plan, including the dataset items you removed? This cannot be undone."
        : "Are you sure you want to discard the current batch plan? This cannot be undone."
    }
    if (confirm_msg && !confirm(confirm_msg)) {
      pushState("", { kiln_pro_stage: stage })
      return
    }

    if (!hist_stage && (stage === "plan" || stage === "inputs")) {
      plan_error = null
      stage = "intro"
    } else if (hist_stage === "plan" && stage === "inputs") {
      stage = "plan"
    }
  }

  async function submit_batch() {
    batch_submitting = false
    const requested = num_inputs
    const data_guide = get(guidance_data.use_data_guide)
      ? get(guidance_data.data_guide)
      : null
    plan_error = null
    stage = "planning"
    posthog.capture("kiln_pro_batch_plan", {
      count: requested,
      gen_type: guidance_data.gen_type,
      template: get(selected_template),
    })
    try {
      const { data, error } = await client.POST(
        "/api/projects/{project_id}/tasks/{task_id}/copilot/batch_plan",
        {
          params: { path: { project_id, task_id } },
          body: { guidance: batch_guidance, count: requested, data_guide },
        },
      )
      if (error) throw error
      if (!data) throw new Error("Batch planner returned no plan.")
      plan = { prompts: data.prompts, summary: data.summary }
      plan_edited = false
      advance_stage("plan")
    } catch (e) {
      plan_error = createKilnError(e)
      // Drop back to the intro so the user can adjust and retry.
      stage = "intro"
    }
  }

  function delete_prompt(index: number) {
    if (!plan) return
    plan = {
      ...plan,
      prompts: plan.prompts.filter((_, i) => i !== index),
    }
    plan_edited = true
  }

  function open_inputs_modal() {
    inputs_error = null
    inputs_dialog?.show()
  }

  // Capture the model config, then transition to the inputs view (which starts
  // generation). The plan page stays put until this submit fires.
  function submit_inputs() {
    inputs_submitting = false
    const rcp =
      inputs_run_config?.run_options_as_run_config_properties() ?? null
    if (!rcp || !isKilnAgentRunConfig(rcp)) {
      inputs_error = new KilnError(
        "Synthetic data generation requires a model with a kiln_agent run config.",
        null,
      )
      return
    }
    inputs_rcp = rcp
    inputs_data_guide = get(guidance_data.use_data_guide)
      ? get(guidance_data.data_guide)
      : null
    inputs_dialog?.close()
    advance_stage("inputs")
  }
</script>

{#if current_state === "loading"}
  <div class="flex justify-center items-center min-h-[50vh]">
    <div class="loading loading-spinner loading-lg"></div>
  </div>
{:else if stage === "planning"}
  <div class="flex flex-col items-center justify-center min-h-[50vh] mt-12">
    <RefiningAnimation
      title="Planning Batch"
      description={`Kiln is planning a diverse batch of ${num_inputs} dataset items, tailored to your task and guidance.`}
    />
  </div>
{:else if stage === "plan" && plan}
  <KilnProBatchPlan
    {plan}
    on_generate_inputs={open_inputs_modal}
    on_regenerate={go_back}
    on_delete_prompt={delete_prompt}
    summary_out_of_sync={plan_edited}
  />
{:else if stage === "inputs" && plan && inputs_rcp}
  <KilnProInputs
    {plan}
    {project_id}
    {task_id}
    {guidance_data}
    run_config_properties={inputs_rcp}
    data_guide={inputs_data_guide}
    {session_id}
    on_back={go_back}
    bind:unsaved_samples
  />
{:else}
  <div class="max-w-2xl mx-auto mt-8 md:mt-16 mb-8">
    <div class="flex items-center gap-2 mb-6">
      <div class="h-8 w-8 flex-none">
        <img src="/images/animated_logo.svg" alt="Kiln Pro" />
      </div>
      <h2 class="text-xl font-medium">Generate Synthetic Data Batch</h2>
    </div>
    <FormContainer
      submit_label="Generate Batch"
      compact_button={true}
      focus_on_mount={false}
      bind:submitting={batch_submitting}
      on:submit={submit_batch}
    >
      <KilnProBatchForm
        bind:count={num_inputs}
        bind:guidance={batch_guidance}
        guidance_template={batch_guidance_template}
      />
      <SynthDataGuide
        {project_id}
        {task_id}
        data_guide={$data_guide_store}
        bind:use_data_guide={$use_data_guide_store}
      />
    </FormContainer>
    {#if plan_error}
      <div class="text-error text-sm mt-4">
        {plan_error.getMessage()}
      </div>
    {/if}
  </div>
{/if}

<Dialog bind:this={inputs_dialog} title="Generation Settings">
  <FormContainer
    submit_label={plan ? `Generate Batch (${plan.prompts.length})` : "Generate"}
    bind:submitting={inputs_submitting}
    on:submit={submit_inputs}
    keyboard_submit={false}
  >
    {#if task}
      <RunConfigComponent
        bind:this={inputs_run_config}
        {project_id}
        current_task={task}
        requires_structured_output={true}
        hide_prompt_selector={true}
        show_tools_selector_in_advanced={true}
        show_name_field={false}
        model_dropdown_settings={{
          requires_data_gen: true,
          requires_uncensored_data_gen:
            guidance_data.suggest_uncensored($selected_template),
          suggested_mode: guidance_data.suggest_uncensored($selected_template)
            ? "uncensored_data_gen"
            : "data_gen",
        }}
      />
    {/if}
    {#if inputs_error}
      <div class="text-error text-sm">{inputs_error.getMessage()}</div>
    {/if}
  </FormContainer>
</Dialog>
