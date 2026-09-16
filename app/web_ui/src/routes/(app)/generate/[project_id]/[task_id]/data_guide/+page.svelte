<script lang="ts">
  // Saved-guide view: shown when the user already has a saved data guide and
  // revisits to read it. The active refine/preview loop lives at
  // /data_guide/refine; this page hands off to it via a writable store when
  // the user clicks "Test Data Guide" or submits the edit dialog.
  import AppPage from "../../../../app_page.svelte"
  import { client } from "$lib/api_client"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { onMount, onDestroy } from "svelte"
  import { page } from "$app/stores"
  import { goto } from "$app/navigation"
  import GuideRefineView from "../data_guide_setup/guide_refine_view.svelte"
  import DataGenDescription from "../data_gen_description.svelte"
  import { SynthDataGuidanceDataModel } from "../synth_data_guidance_datamodel"
  import type { KilnAgentRunConfigProperties, DataGuide } from "$lib/types"
  import { agentInfo } from "$lib/agent"
  import DeleteDialog from "$lib/ui/delete_dialog.svelte"
  import { isMacOS } from "$lib/utils/platform"
  import { pending_data_guide_refine_handoff } from "./refine_handoff_store"
  import posthog from "posthog-js"
  import {
    data_guide_return,
    read_data_guide_caller,
    with_data_guide_caller,
  } from "$lib/utils/data_guide_return"

  type ViewState = "loading" | "saved"

  let current_state: ViewState = "loading"
  let error: KilnError | null = null

  let guide: string = ""
  let saved_data_guide: DataGuide | null = null
  // Bound so the AppPage's "Edit" action button can drive the dialog inside
  // GuideRefineView without having to lift the dialog state up here.
  let refine_view: GuideRefineView | null = null

  // Surface "Save Without Verifying" pending + error state inside the edit
  // dialog (instead of dropping the failure into the parent-level `error`,
  // which renders behind the still-open dialog).
  let save_submitting: boolean = false
  let save_error: KilnError | null = null

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
    name: "Data Guide",
    description: `View and refine the saved task input data guide for project ${project_id}, task ${task_id}.`,
  })

  // Distinguish "load failed" from "no guide saved". Without this, a 5xx
  // would silently send the user into setup and risk overwriting a guide
  // that's actually present but unreachable due to a transient error.
  let load_failed = false
  onMount(async () => {
    try {
      const { data, error: api_error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/data_gen_guide",
        { params: { path: { project_id, task_id } } },
      )
      if (api_error) {
        load_failed = true
        error = createKilnError(api_error)
      } else if (data) {
        guide = data.guide
        saved_data_guide = data
      }
    } catch (e) {
      load_failed = true
      error = createKilnError(e)
    }

    if (load_failed) {
      // Surface the load error in-page rather than redirecting away, so the
      // user can see what went wrong instead of being silently sent to setup.
      current_state = "saved"
      return
    }

    // No saved guide content → send them back to the caller, whose intro
    // is the entry point for creating a new guide (manual or Kiln Pro).
    if (!guide.trim()) {
      goto(return_target.view_href)
      return
    }

    current_state = "saved"
  })

  function handle_generate_preview(
    event: CustomEvent<{
      guide: string
      input_run_config: KilnAgentRunConfigProperties
    }>,
  ) {
    pending_data_guide_refine_handoff.set({
      guide: event.detail.guide,
      saved_guide: guide,
      input_run_config: event.detail.input_run_config,
    })
    goto(
      with_data_guide_caller(
        `/generate/${project_id}/${task_id}/data_guide/refine`,
        caller,
      ),
    )
  }

  async function handle_save_with_guide(event: CustomEvent<{ guide: string }>) {
    save_error = null
    save_submitting = true

    try {
      const { data: saved, error: api_error } = await client.PUT(
        "/api/projects/{project_id}/tasks/{task_id}/data_gen_guide",
        {
          params: { path: { project_id, task_id } },
          body: { guide: event.detail.guide },
        },
      )

      if (api_error) throw api_error

      if (saved) {
        saved_data_guide = saved
        guide = saved.guide
      }

      posthog.capture("data_guide_saved", {
        method: "save_without_verifying",
        source: "saved_view",
        refine_iterations: 0,
      })
    } catch (e) {
      save_error = createKilnError(e)
    } finally {
      save_submitting = false
    }
  }

  let delete_dialog: DeleteDialog | null = null
  $: delete_url = `/api/projects/${project_id}/tasks/${task_id}/data_gen_guide`
  function after_delete() {
    goto(return_target.view_href)
  }
</script>

<div class="max-w-[1400px]">
  <AppPage
    title="Data Guide"
    sub_subtitle="Read the Docs"
    sub_subtitle_link="https://docs.kiln.tech/docs/synthetic-data-generation"
    breadcrumbs={[
      {
        label: return_target.label,
        href: return_target.view_href,
        // This page is a sub-flow of its caller — replace rather than push
        // so back from the caller returns to wherever the user originally
        // came from (cards page, spec page, etc.) instead of bouncing here.
        replace_state: true,
      },
    ]}
    action_buttons={current_state === "saved"
      ? [
          {
            icon: "/images/delete.svg",
            handler: () => delete_dialog?.show(),
            shortcut: isMacOS() ? "Backspace" : "Delete",
          },
          {
            label: "Edit",
            handler: () => refine_view?.open_edit_chooser(),
          },
        ]
      : []}
  >
    <DataGenDescription bind:guidance_data />

    {#if current_state === "loading"}
      <div class="flex flex-col items-center justify-center py-24 gap-4">
        <span class="loading loading-spinner loading-lg" />
      </div>
    {:else if current_state === "saved"}
      <GuideRefineView
        bind:this={refine_view}
        {project_id}
        {guide}
        data_guide={saved_data_guide}
        bind:page_error={error}
        bind:save_error
        bind:save_submitting
        on:generate_preview={handle_generate_preview}
        on:save={handle_save_with_guide}
      />
    {/if}
  </AppPage>
</div>

<DeleteDialog
  name="Data Guide"
  bind:this={delete_dialog}
  {delete_url}
  {after_delete}
/>
