<script lang="ts">
  import { goto } from "$app/navigation"
  import { page } from "$app/stores"
  import { onMount } from "svelte"
  import AppPage from "../../../../app_page.svelte"
  import { agentInfo } from "$lib/agent"
  import { getDataGuideJob } from "$lib/stores/data_guide_job_store"
  import AddExampleDialog, {
    type GuideSample,
  } from "$lib/components/add_example_dialog.svelte"
  import { pending_data_guide_example } from "../data_guide_setup/pending_example_store"
  import posthog from "posthog-js"
  import {
    data_guide_return,
    read_data_guide_caller,
    stash_data_guide_caller,
    with_data_guide_caller,
  } from "$lib/utils/data_guide_return"

  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  // Which page opened the setup chain (no key means synthetic data
  // generation): every link out of here forwards it, and the breadcrumb and
  // the finish action return to it.
  $: caller = read_data_guide_caller($page.url.searchParams)
  $: return_target = data_guide_return(caller, project_id, task_id)
  $: agentInfo.set({
    name: "Choose Data Guide Workflow",
    description: `Choose between manual and Kiln Pro Data Guide creation for project ${project_id}, task ${task_id}.`,
  })

  // If a Kiln Pro draft job is already in progress for this task, skip the
  // chooser and drop the user straight back onto its spinner page.
  onMount(() => {
    const job = getDataGuideJob(project_id, task_id)
    if (job) {
      goto(
        with_data_guide_caller(
          `/generate/${project_id}/${task_id}/data_guide_setup_copilot/${job.job_id}`,
          caller,
        ),
        { replaceState: true },
      )
    }
  })

  // Add-example dialog reused from the data_guide_setup flow. Picking the
  // manual path opens it first so the user fills out their first example here;
  // after they submit, we stash the sample on a pending store and navigate to
  // /data_guide_setup, where it gets seeded into the form.
  let add_data_guide_example_dialog: AddExampleDialog

  function pick_manual() {
    posthog.capture("data_guide_chooser_picked", { choice: "manual" })
    add_data_guide_example_dialog?.open_add()
  }

  function handle_data_guide_example_added(
    event: CustomEvent<{
      sample: GuideSample
      index: number
      mode: "add" | "edit"
    }>,
  ) {
    pending_data_guide_example.set(event.detail.sample)
    goto(
      with_data_guide_caller(
        `/generate/${project_id}/${task_id}/data_guide_setup`,
        caller,
      ),
    )
  }

  function pick_kiln_pro() {
    posthog.capture("data_guide_chooser_picked", { choice: "kiln_pro" })
    const job = getDataGuideJob(project_id, task_id)
    if (job) {
      goto(
        with_data_guide_caller(
          `/generate/${project_id}/${task_id}/data_guide_setup_copilot/${job.job_id}`,
          caller,
        ),
      )
      return
    }
    // Static auth route: it forwards to the setup page once connected (and
    // straight through if the user already is). Its URL is fixed for OAuth,
    // so the caller rides a stash across that hop instead of the URL.
    stash_data_guide_caller(caller)
    goto(`/generate/data_guide_pro_auth`, { replaceState: true })
  }
</script>

<div class="max-w-[900px]">
  <AppPage
    title="Set Up Data Guide"
    subtitle="Your Data Guide will help us generate better synthetic inputs."
    sub_subtitle="Read the Docs"
    sub_subtitle_link="https://docs.kiln.tech/docs/synthetic-data-generation"
    breadcrumbs={[
      {
        label: return_target.label,
        href: return_target.href,
        replace_state: true,
      },
    ]}
  >
    <div class="my-4 max-w-[680px] mx-auto">
      <div class="overflow-x-auto">
        <table class="table w-full mt-4">
          <colgroup>
            <col class="w-[50%]" />
            <col class="w-[25%]" />
            <col class="w-[25%]" />
          </colgroup>
          <thead>
            <tr class="border-b-0">
              <th></th>
              <th class="text-center text-lg">Manual</th>
              <th class="text-center text-lg">
                <div class="flex items-center justify-center gap-2">
                  <img
                    src="/images/animated_logo.svg"
                    alt="Kiln Pro"
                    class="size-4"
                  />
                  <span>Kiln Pro</span>
                </div>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th class="font-bold text-xs text-gray-500"
                >AI Guided Authoring</th
              >
              <td class="text-center">Manual</td>
              <td class="text-center border-l">Automatic</td>
            </tr>
            <tr>
              <th class="font-bold text-xs text-gray-500"
                >Style & Constraints Discovery</th
              >
              <td class="text-center">Manual</td>
              <td class="text-center border-l">Automatic</td>
            </tr>
            <tr>
              <th class="font-bold text-xs text-gray-500"
                >Learn From Documents</th
              >
              <td class="text-center">—</td>
              <td class="text-center border-l">Supported</td>
            </tr>
            <tr>
              <th class="font-bold text-xs text-gray-500">Approx. Effort</th>
              <td class="text-center">~15 min</td>
              <td class="text-center border-l">~5 min</td>
            </tr>
            <tr class="border-b">
              <th class="font-bold text-xs text-base-content/60"
                >Kiln Account</th
              >
              <td class="text-center">Optional</td>
              <td class="text-center border-l">Required</td>
            </tr>
            <tr>
              <th></th>
              <td class="text-center pt-4">
                <button
                  class="btn btn-outline btn-sm whitespace-nowrap"
                  on:click={pick_manual}
                >
                  Create Manually
                </button>
              </td>
              <td class="text-center pt-4">
                <button
                  class="btn btn-primary btn-sm whitespace-nowrap"
                  on:click={pick_kiln_pro}
                >
                  Use Kiln Pro
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </AppPage>
</div>

<AddExampleDialog
  bind:this={add_data_guide_example_dialog}
  {project_id}
  {task_id}
  include_output={false}
  sub_subtitle="To start, add an example input to guide synthetic input generation."
  on:submit={handle_data_guide_example_added}
/>
