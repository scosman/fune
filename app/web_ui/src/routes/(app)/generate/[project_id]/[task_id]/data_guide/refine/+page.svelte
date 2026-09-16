<script lang="ts">
  // Refine flow: previewing + iterating on a saved input data guide. Reached
  // from /data_guide via a handoff store carrying the chosen run configs and
  // the (possibly edited) guide text. The saved-guide read-only view lives at
  // /data_guide; this page only handles the active refine loop.
  import AppPage from "../../../../../app_page.svelte"
  import { client } from "$lib/api_client"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { onMount, onDestroy } from "svelte"
  import { page } from "$app/stores"
  import { goto } from "$app/navigation"
  import { get } from "svelte/store"
  import GuidePreview from "../../data_guide_setup/guide_preview.svelte"
  import DataGenDescription from "../../data_gen_description.svelte"
  import { SynthDataGuidanceDataModel } from "../../synth_data_guidance_datamodel"
  import type { KilnAgentRunConfigProperties } from "$lib/types"
  import { agentInfo } from "$lib/agent"
  import AnalyzingAnimation from "$lib/ui/animations/analyzing_animation.svelte"
  import RefiningAnimation from "$lib/ui/animations/refining_animation.svelte"
  import { pending_data_guide_refine_handoff } from "../refine_handoff_store"
  import posthog from "posthog-js"
  import {
    data_guide_return,
    read_data_guide_caller,
    with_data_guide_caller,
  } from "$lib/utils/data_guide_return"

  type RefineState =
    | "loading"
    | "generating"
    | "preview"
    | "refining"
    | "regenerating"
    | "load_error"

  let current_state: RefineState = "loading"
  let error: KilnError | null = null
  let submitting = false
  // Flipped true once the guide is saved and we're navigating away. Suppresses
  // GuidePreview's unsaved-changes warn so the post-save goto doesn't prompt.
  let saved = false

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

  let captured_input_run_config: KilnAgentRunConfigProperties | null = null
  // Snapshot of the guide as actually-saved-on-server at handoff time. Drives
  // the GuidePreview submit button label/behavior: when the working `guide`
  // matches this, there's nothing to persist — the button becomes a plain
  // "Back to Input Data Guide" navigation rather than a save.
  let saved_guide_snapshot: string = ""
  $: requires_save = guide !== saved_guide_snapshot

  // Number of successful refine/regenerate cycles before save, for analytics.
  let refine_iterations = 0

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
  $: saved_guide_url = with_data_guide_caller(
    `/generate/${project_id}/${task_id}/data_guide`,
    caller,
  )
  $: agentInfo.set({
    name: "Refine Data Guide",
    description: `Refine the saved task input data guide for project ${project_id}, task ${task_id}.`,
  })

  onMount(async () => {
    const handoff = get(pending_data_guide_refine_handoff)
    if (!handoff) {
      // No seed (direct URL hit / hard refresh) — the saved-guide view is
      // where this flow starts.
      goto(saved_guide_url, { replaceState: true })
      return
    }
    pending_data_guide_refine_handoff.set(null)

    guide = handoff.guide
    saved_guide_snapshot = handoff.saved_guide
    captured_input_run_config = handoff.input_run_config

    await run_initial_preview()
  })

  async function run_initial_preview() {
    error = null
    submitting = true
    current_state = "generating"

    try {
      if (!captured_input_run_config) {
        throw new KilnError("No model configuration available", null)
      }

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

      preview_samples = data as PreviewSample[]
      reviewed_samples = preview_samples.map((s) => ({
        input: s.input,
        looks_good: undefined,
      }))
      general_feedback = ""
      preview_initial_guide = guide
      current_state = "preview"
    } catch (e) {
      // Surface the failure on this page rather than silently bouncing back
      // to /data_guide, which would lose the error and leave the user with
      // no idea why the refine flow didn't start.
      error = createKilnError(e)
      current_state = "load_error"
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
      preview_samples = preview_data as PreviewSample[]
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

  function handle_back() {
    // Suppress the unsaved-changes warn — there's nothing the user could lose
    // (guide already matches what's on the server, samples were just for
    // verification).
    saved = true
    goto(saved_guide_url, { replaceState: true })
  }

  async function handle_save() {
    error = null
    submitting = true

    try {
      const { error: api_error } = await client.PUT(
        "/api/projects/{project_id}/tasks/{task_id}/data_gen_guide",
        {
          params: { path: { project_id, task_id } },
          body: { guide },
        },
      )

      if (api_error) throw api_error

      // Disable the unsaved-changes warn before goto fires beforeNavigate.
      saved = true

      posthog.capture("data_guide_saved", {
        method: "after_preview",
        source: "refine",
        refine_iterations,
      })

      // Replace state so the user can't back-navigate into the now-stale
      // refine flow they just exited. /data_guide will refetch the saved
      // guide on its own.
      goto(saved_guide_url, { replaceState: true })
    } catch (e) {
      error = createKilnError(e)
    } finally {
      submitting = false
    }
  }
</script>

<div class="max-w-[1400px]">
  <AppPage
    title="Refine Data Guide"
    sub_subtitle="Read the Docs"
    sub_subtitle_link="https://docs.kiln.tech/docs/synthetic-data-generation"
    breadcrumbs={[
      {
        label: return_target.label,
        href: return_target.view_href,
      },
      {
        label: "Data Guide",
        href: saved_guide_url,
      },
    ]}
  >
    <DataGenDescription bind:guidance_data />

    {#if current_state === "loading" || current_state === "generating"}
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
        {requires_save}
        on:refine={handle_refine}
        on:save={handle_save}
        on:back={handle_back}
      />
    {:else if current_state === "refining"}
      <RefiningAnimation
        title="Refining Data Guide"
        description="Kiln is refining your data guide with the feedback you provided and generating fresh inputs to review. Hold tight!"
      />
    {:else if current_state === "regenerating"}
      <AnalyzingAnimation
        title="Regenerating Inputs"
        description="Regenerating inputs to review with your edited data guide. Hold tight!"
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
