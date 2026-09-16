<script lang="ts">
  import AppPage from "../../../../app_page.svelte"
  import { client } from "$lib/api_client"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { onMount, onDestroy } from "svelte"
  import { page } from "$app/stores"
  import { goto } from "$app/navigation"
  import GuideSetupForm from "./guide_setup_form.svelte"
  import type { GuideSample } from "$lib/components/add_example_dialog.svelte"
  import { pending_data_guide_example } from "./pending_example_store"
  import { get } from "svelte/store"
  import GuidePreview from "./guide_preview.svelte"
  import DataGenDescription from "../data_gen_description.svelte"
  import { SynthDataGuidanceDataModel } from "../synth_data_guidance_datamodel"
  import type { KilnAgentRunConfigProperties } from "$lib/types"
  import { agentInfo } from "$lib/agent"
  import AnalyzingAnimation from "$lib/ui/animations/analyzing_animation.svelte"
  import RefiningAnimation from "$lib/ui/animations/refining_animation.svelte"
  import Completed from "$lib/ui/completed.svelte"
  import { dedupe_by_input } from "$lib/utils/dedupe_by_input"
  import posthog from "posthog-js"
  import {
    data_guide_return,
    read_data_guide_caller,
    with_data_guide_caller,
  } from "$lib/utils/data_guide_return"

  type GuideBuilderState =
    | "loading"
    | "setup"
    | "generating"
    | "preview"
    | "refining"
    | "regenerating"
    | "load_error"

  // Start in "loading" so we can redirect away if a saved guide already exists
  // (the refine flow lives at /data_guide). Without this we'd briefly flash
  // the setup form before the GET resolves.
  let current_state: GuideBuilderState = "loading"
  let error: KilnError | null = null
  let submitting = false
  // Flipped true once the guide is saved and we're navigating away. Suppresses
  // GuidePreview's unsaved-changes warn so the post-save goto doesn't prompt.
  let saved = false

  // The full input data guide markdown. Setup builds this from the user's
  // examples; refine rewrites it wholesale to incorporate generated rules.
  let guide: string = ""

  type PreviewSample = { input: string }
  type ReviewedSample = {
    input: string
    looks_good: boolean | undefined
  }
  let preview_samples: PreviewSample[] = []
  let preview_initial_guide: string = ""
  let reviewed_samples: ReviewedSample[] = []
  let general_feedback: string = ""

  // Captured from the setup form so refine/regenerate can reuse them
  let captured_input_run_config: KilnAgentRunConfigProperties | null = null

  // Number of successful refine/regenerate cycles before save, for analytics.
  let refine_iterations = 0

  // Lifted out of GuideSetupForm so the user's examples survive the
  // setup → generating → setup unmount cycle that happens when a preview
  // request fails.
  let guide_examples: GuideSample[] = []

  let guidance_data: SynthDataGuidanceDataModel =
    new SynthDataGuidanceDataModel()
  onDestroy(() => {
    guidance_data.destroy()
  })

  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  // Which page opened the setup chain (no key means synthetic data
  // generation): every link out of here forwards it, and the breadcrumb and
  // the finish action return to it.
  $: caller = read_data_guide_caller($page.url.searchParams)
  $: return_target = data_guide_return(caller, project_id, task_id)
  $: agentInfo.set({
    name: "Set Up Data Guide",
    description: `Setup the task input data guide for project ${project_id}, task ${task_id}. The input data guide describes the structure, rules, and examples for synthetic input generation.`,
  })

  onMount(async () => {
    // If a saved guide already exists, the user shouldn't be in the setup
    // flow — that's for first-time creation only. Edit/delete/re-verify
    // happens on the main /data_guide page.
    //
    // Distinguish "GET returned no guide" from "GET failed". A backend or
    // network error must NOT silently land the user on setup, where they
    // could overwrite an existing-but-currently-unreachable guide once the
    // server recovers.
    try {
      const { data, error: api_error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/data_gen_guide",
        { params: { path: { project_id, task_id } } },
      )
      if (api_error) {
        error = createKilnError(api_error)
        current_state = "load_error"
        return
      }
      if (data) {
        goto(
          with_data_guide_caller(
            `/generate/${project_id}/${task_id}/data_guide`,
            caller,
          ),
          { replaceState: true },
        )
        return
      }
    } catch (e) {
      error = createKilnError(e)
      current_state = "load_error"
      return
    }

    // Seed from the synth-page handoff: if the user clicked "Set Up Data
    // Guide" and added their first example via the dialog before navigating
    // here, that sample is sitting on a writable store. Pull it once and
    // clear so a hard refresh doesn't re-seed it.
    const seeded = get(pending_data_guide_example)
    if (seeded) {
      guide_examples = [seeded]
      pending_data_guide_example.set(null)
    }

    current_state = "setup"
  })

  async function handle_generate_preview(
    event: CustomEvent<{
      guide: string
      input_run_config: KilnAgentRunConfigProperties
    }>,
  ) {
    error = null
    submitting = true
    current_state = "generating"

    try {
      captured_input_run_config = event.detail.input_run_config
      guide = event.detail.guide

      const { data, error: api_error } = await client.POST(
        "/api/projects/{project_id}/tasks/{task_id}/data_gen_guide_preview",
        {
          params: { path: { project_id, task_id } },
          body: {
            guide,
            run_config_properties: captured_input_run_config,
            num_samples: 5,
          },
        },
      )

      if (api_error) throw api_error
      if (!data) throw new KilnError("No preview inputs returned", null)

      preview_samples = dedupe_by_input(data as PreviewSample[])
      reviewed_samples = preview_samples.map((s) => ({
        input: s.input,
        looks_good: undefined,
      }))
      general_feedback = ""
      preview_initial_guide = guide
      current_state = "preview"
    } catch (e) {
      error = createKilnError(e)
      current_state = "setup"
    } finally {
      submitting = false
    }
  }

  async function handle_refine(
    event: CustomEvent<{
      feedback: string
      rated_samples: { input: string; looks_good: boolean }[]
    }>,
  ) {
    error = null
    submitting = true

    const has_negative_feedback =
      event.detail.rated_samples.some((s) => !s.looks_good) ||
      event.detail.feedback.trim().length > 0

    current_state = has_negative_feedback ? "refining" : "regenerating"

    try {
      if (!captured_input_run_config) {
        throw new KilnError("No model configuration available", null)
      }

      let refined_guide = guide
      if (has_negative_feedback) {
        const { data, error: api_error } = await client.POST(
          "/api/projects/{project_id}/tasks/{task_id}/data_gen_guide_refine",
          {
            params: { path: { project_id, task_id } },
            body: {
              current_guide: guide,
              source: "manual",
              feedback: event.detail.feedback,
              preview_samples: event.detail.rated_samples,
              run_config_properties: captured_input_run_config,
            },
          },
        )

        if (api_error) throw api_error
        if (!data) throw new KilnError("No refinement returned", null)

        refined_guide = data.refined_guide
      }

      const { data: preview_data, error: preview_error } = await client.POST(
        "/api/projects/{project_id}/tasks/{task_id}/data_gen_guide_preview",
        {
          params: { path: { project_id, task_id } },
          body: {
            guide: refined_guide,
            run_config_properties: captured_input_run_config,
            num_samples: 5,
          },
        },
      )

      if (preview_error) throw preview_error
      if (!preview_data) throw new KilnError("No preview inputs returned", null)

      guide = refined_guide
      preview_samples = dedupe_by_input(preview_data as PreviewSample[])
      reviewed_samples = preview_samples.map((s) => ({
        input: s.input,
        looks_good: undefined,
      }))
      general_feedback = ""
      preview_initial_guide = guide
      current_state = "preview"
      refine_iterations++
    } catch (e) {
      error = createKilnError(e)
      current_state = "preview"
    } finally {
      submitting = false
    }
  }

  async function handle_save() {
    error = null
    submitting = true

    try {
      const { error: api_error } = await client.PUT(
        "/api/projects/{project_id}/tasks/{task_id}/data_gen_guide",
        {
          params: { path: { project_id, task_id } },
          body: { guide, source: "manual" },
        },
      )

      if (api_error) throw api_error

      // `saved` both disables the unsaved-changes warn and flips the page to
      // the success screen (Completed), which returns the user to synth via a
      // button rather than auto-navigating.
      saved = true

      posthog.capture("data_guide_saved", {
        method: "after_preview",
        source: "setup",
        refine_iterations,
      })
    } catch (e) {
      error = createKilnError(e)
    } finally {
      submitting = false
    }
  }
</script>

<div class="max-w-[1400px]">
  <AppPage
    title="Set Up Data Guide"
    subtitle="Your Data Guide will help us generate better synthetic inputs."
    sub_subtitle="Read the Docs"
    sub_subtitle_link="https://docs.kiln.tech/docs/synthetic-data-generation"
    breadcrumbs={[
      {
        label: return_target.label,
        href: return_target.href,
        // This page is a sub-flow of its caller — replace rather than push
        // so back from the caller returns to wherever the user originally
        // came from (cards page, spec page, etc.) instead of bouncing here.
        replace_state: true,
      },
    ]}
  >
    <DataGenDescription bind:guidance_data />

    {#if saved}
      <Completed
        title="Data Guide Saved"
        subtitle={`Your Data Guide is saved. Click Continue to return to ${return_target.label}.`}
        link={return_target.href}
        button_text="Continue"
      />
    {:else if current_state === "loading"}
      <div class="flex flex-col items-center justify-center py-24 gap-4">
        <span class="loading loading-spinner loading-lg" />
      </div>
    {:else if current_state === "setup"}
      <GuideSetupForm
        {project_id}
        {task_id}
        bind:guide_examples
        bind:page_error={error}
        on:generate_preview={handle_generate_preview}
      />
    {:else if current_state === "generating"}
      <AnalyzingAnimation
        title="Generating Inputs"
        description="Generating synthetic inputs to test your data guide."
      />
    {:else if current_state === "preview"}
      <GuidePreview
        initial_guide={preview_initial_guide}
        bind:guide
        bind:error
        bind:submitting
        bind:reviewed_samples
        bind:general_feedback
        {saved}
        on:refine={handle_refine}
        on:save={handle_save}
      />
    {:else if current_state === "refining"}
      <RefiningAnimation
        title="Refining Data Guide"
        description="Refining your data guide with the feedback you provided and generating fresh inputs to review."
      />
    {:else if current_state === "regenerating"}
      <AnalyzingAnimation
        title="Regenerating Inputs"
        description="Regenerating synthetic inputs to test your edited data guide."
      />
    {:else if current_state === "load_error"}
      <div
        class="w-full min-h-[50vh] flex flex-col justify-center items-center gap-2"
      >
        <div class="text-error text-sm">
          {error?.getMessage() ?? "An unknown error occurred"}
        </div>
      </div>
    {/if}
  </AppPage>
</div>
