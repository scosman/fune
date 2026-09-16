<script lang="ts">
  // Spinner page for an in-flight Data Guide draft job. The draft runs as a
  // kiln_server background job; this page polls it (via the shared job store)
  // and, on success, fetches the draft and hands off to the base copilot page
  // to generate preview inputs. On failure it drops the user back on the input
  // page with their examples and an error. The user can safely leave and come
  // back — the job keeps running server-side and the store keeps polling.
  import AppPage from "../../../../../app_page.svelte"
  import { client } from "$lib/api_client"
  import { page } from "$app/stores"
  import { goto } from "$app/navigation"
  import { onMount } from "svelte"
  import RefiningAnimation from "$lib/ui/animations/refining_animation.svelte"
  import { agentInfo } from "$lib/agent"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import {
    data_guide_jobs,
    getDataGuideJob,
    clearDataGuideJob,
  } from "$lib/stores/data_guide_job_store"
  import { pending_data_guide_draft } from "../pending_draft_store"
  import {
    data_guide_return,
    read_data_guide_caller,
    with_data_guide_caller,
  } from "$lib/utils/data_guide_return"

  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  $: job_id = $page.params.job_id!
  // Which page opened the setup chain (no key means synthetic data
  // generation): every link out of here forwards it, and the breadcrumb and
  // the finish action return to it.
  $: caller = read_data_guide_caller($page.url.searchParams)
  $: return_target = data_guide_return(caller, project_id, task_id)
  $: agentInfo.set({
    name: "Set Up Data Guide",
    description: `Drafting the input data guide for project ${project_id}, task ${task_id}.`,
  })

  const base_path = () =>
    `/generate/${project_id}/${task_id}/data_guide_setup_copilot`
  const base_url = () => with_data_guide_caller(base_path(), caller)

  // Guards against double-handling a terminal status (the reactive block can
  // fire more than once before navigation completes).
  let handled = false
  let error: KilnError | null = null

  $: record = $data_guide_jobs[`${project_id}/${task_id}`]
  // Only show the "drafting" animation while the job is genuinely still
  // running. If the user arrives here on an already-complete job (e.g. clicking
  // the Complete indicator), show a neutral spinner during the brief
  // result-fetch + redirect rather than flashing the drafting animation.
  $: drafting = record?.status === "running"

  onMount(() => {
    // No record for this task means there's nothing to resume here — either it
    // was cleared (saved/restarted) or this is a stale/direct link. Send the
    // user back to the setup entry point.
    if (!getDataGuideJob(project_id, task_id)) {
      goto(base_url(), { replaceState: true })
    }
  })

  // React to status transitions reported by the shared store's poller.
  $: if (!handled && record && record.job_id === job_id) {
    if (record.status === "succeeded") {
      handled = true
      handle_success()
    } else if (record.status === "failed" || record.status === "cancelled") {
      handled = true
      // Drop back to the input page; it re-seeds the examples from the record
      // and shows the error, then clears the record.
      goto(with_data_guide_caller(`${base_path()}?draft_failed=1`, caller), {
        replaceState: true,
      })
    }
  }

  async function handle_success() {
    try {
      const { data, error: api_error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/copilot/data_guide_job/{job_id}/result",
        { params: { path: { project_id, task_id, job_id } } },
      )
      if (api_error) throw api_error
      if (!data?.draft_guide)
        throw new KilnError("No draft guide returned", null)

      const current = getDataGuideJob(project_id, task_id)
      if (!current)
        throw new KilnError("Data guide job is no longer tracked", null)
      pending_data_guide_draft.set({
        job_id,
        project_id,
        task_id,
        draft_guide: data.draft_guide,
        run_config_properties: current.run_config_properties,
      })
      goto(base_url(), { replaceState: true })
    } catch (e) {
      // Fetching the finished result failed (e.g. transient network). Surface
      // the error and let the user retry rather than losing the completed
      // draft. Keep `handled` true: resetting it here would re-fire the
      // reactive block above (status is still "succeeded") and auto-retry in a
      // tight loop instead of waiting for the user's Retry click.
      error = createKilnError(e)
    }
  }

  function retry_result() {
    error = null
    // `handled` stays true across the failed fetch (see handle_success catch),
    // so don't gate on it — just re-run the result fetch.
    handle_success()
  }

  function back_to_setup() {
    clearDataGuideJob(project_id, task_id)
    goto(base_url(), { replaceState: true })
  }
</script>

<div class="max-w-[1400px]">
  <AppPage
    title="Set Up Data Guide"
    subtitle="Your Data Guide will help us generate better synthetic inputs."
    breadcrumbs={[
      {
        label: return_target.label,
        href: return_target.href,
        replace_state: true,
      },
    ]}
  >
    {#if error}
      <div
        class="w-full min-h-[50vh] flex flex-col justify-center items-center gap-4"
      >
        <div class="text-error text-sm text-center max-w-[480px]">
          {error.getMessage() ?? "An unknown error occurred"}
        </div>
        <div class="flex gap-2">
          <button class="btn btn-primary btn-sm" on:click={retry_result}>
            Retry
          </button>
          <button class="btn btn-outline btn-sm" on:click={back_to_setup}>
            Back to Setup
          </button>
        </div>
      </div>
    {:else if drafting}
      <RefiningAnimation
        title="Analyzing Inputs"
        description_paragraphs={[
          "Kiln Pro is analyzing your examples and drafting your data guide. This may take a while, depending on the number of examples.",
          "You can leave this page and come back; we'll keep working in the background.",
        ]}
      />
    {:else}
      <div class="flex flex-col items-center justify-center py-24 gap-4">
        <span class="loading loading-spinner loading-lg" />
      </div>
    {/if}
  </AppPage>
</div>
