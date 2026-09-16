<script lang="ts">
  import AppPage from "../../../app_page.svelte"
  import EmptyFinetune from "./empty_finetune.svelte"
  import { client } from "$lib/api_client"
  import type { Finetune, Task } from "$lib/types"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { goto } from "$app/navigation"
  import { page } from "$app/stores"
  import { formatDate } from "$lib/utils/formatters"
  import {
    provider_name_from_id,
    load_available_models,
    load_task,
  } from "$lib/stores"
  import { data_strategy_name } from "$lib/utils/formatters"
  import Warning from "$lib/ui/warning.svelte"

  import { agentInfo } from "$lib/agent"
  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  $: agentInfo.set({
    name: "Fine-Tuning",
    description: `Fine-tuning list for project ID ${project_id}, task ID ${task_id}. Shows all fine-tune jobs and their status.`,
  })
  $: is_empty = !finetunes || finetunes.length == 0
  $: is_multiturn = task?.turn_mode === "multiturn"

  let task: Task | null = null
  let task_error: KilnError | null = null
  let finetunes: Finetune[] | null = null
  let finetunes_error: KilnError | null = null
  let finetunes_loading = true

  $: if (project_id && task_id) {
    finetunes_error = null
    finetunes = null
    task = null
    task_error = null
    load_available_models()
    load_task_for_page(project_id, task_id)
    get_finetunes(project_id, task_id)
  }

  async function load_task_for_page(
    req_project_id: string,
    req_task_id: string,
  ) {
    try {
      const loaded = await load_task(req_project_id, req_task_id)
      if (req_project_id !== project_id || req_task_id !== task_id) return
      task = loaded
    } catch (e) {
      // Without this the "task === null" spinner would spin forever on a
      // failed load (e.g. a deleted task or a project you can't access).
      if (req_project_id !== project_id || req_task_id !== task_id) return
      if (e instanceof Error && e.message.includes("Load failed")) {
        task_error = new KilnError(
          "Could not load task. This task may belong to a project you don't have access to.",
          null,
        )
      } else {
        task_error = createKilnError(e)
      }
    }
  }

  async function get_finetunes(req_project_id: string, req_task_id: string) {
    try {
      finetunes_loading = true
      const { data: finetunes_response, error: get_error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/finetunes",
        {
          params: {
            path: {
              project_id: req_project_id,
              task_id: req_task_id,
            },
            query: {
              update_status: true,
            },
          },
        },
      )
      if (req_project_id !== project_id || req_task_id !== task_id) return
      if (get_error) {
        throw get_error
      }
      const sorted_finetunes = finetunes_response.sort((a, b) => {
        return (
          new Date(b.created_at || "").getTime() -
          new Date(a.created_at || "").getTime()
        )
      })
      finetunes = sorted_finetunes
    } catch (e) {
      if (req_project_id !== project_id || req_task_id !== task_id) return
      if (e instanceof Error && e.message.includes("Load failed")) {
        finetunes_error = new KilnError(
          "Could not load finetunes. This task may belong to a project you don't have access to.",
          null,
        )
      } else {
        finetunes_error = createKilnError(e)
      }
    } finally {
      if (req_project_id === project_id && req_task_id === task_id) {
        finetunes_loading = false
      }
    }
  }

  const status_map: Record<string, string> = {
    pending: "Pending",
    running: "Running",
    completed: "Completed",
    failed: "Failed",
    unknown: "Unknown",
  }
  function format_status(status: string) {
    return status_map[status] || status
  }
</script>

<div class="max-w-[1400px]">
  <AppPage
    title="Fine Tunes"
    subtitle="Fine-tune models for the current task."
    sub_subtitle="Read the Docs"
    sub_subtitle_link="https://docs.kiln.tech/docs/fine-tuning-guide"
    breadcrumbs={[
      {
        label: "Optimize",
        href: `/optimize/${project_id}/${task_id}`,
      },
    ]}
    action_buttons={task === null || is_multiturn || is_empty
      ? []
      : [
          {
            label: "Create Fine Tune",
            href: `/fine_tune/${project_id}/${task_id}/create_finetune`,
            primary: true,
          },
        ]}
  >
    {#if finetunes_loading || (task === null && !task_error)}
      <div class="w-full min-h-[50vh] flex justify-center items-center">
        <div class="loading loading-spinner loading-lg"></div>
      </div>
    {:else if finetunes_error || task_error}
      <div
        class="w-full min-h-[50vh] flex flex-col justify-center items-center gap-2"
      >
        <div class="font-medium">Error Loading Fine Tunes</div>
        <div class="text-error text-sm">
          {(finetunes_error || task_error)?.getMessage() ||
            "An unknown error occurred"}
        </div>
      </div>
    {:else if is_multiturn}
      <div class="flex flex-col items-center justify-center min-h-[60vh]">
        <Warning
          warning_message="Fine-tuning is not supported for multi-turn tasks."
          warning_color="warning"
          warning_icon="info"
        />
      </div>
    {:else if is_empty}
      <div class="flex flex-col items-center justify-center min-h-[60vh]">
        <EmptyFinetune {project_id} {task_id} />
      </div>
    {:else if finetunes}
      <div class="overflow-x-auto rounded-lg border">
        <table class="table">
          <thead>
            <tr>
              <th> Name </th>
              <th> Type </th>
              <th> Provider</th>
              <th> Base Model</th>
              <th> Status </th>
              <th> Created At </th>
            </tr>
          </thead>
          <tbody>
            {#each finetunes as finetune}
              <tr
                class="hover cursor-pointer"
                on:click={() => {
                  goto(
                    `/fine_tune/${project_id}/${task_id}/fine_tune/${finetune.id}`,
                  )
                }}
              >
                <td> {finetune.name} </td>
                <td>
                  {data_strategy_name(finetune.data_strategy)}
                </td>
                <td> {provider_name_from_id(finetune.provider)} </td>
                <td> {finetune.base_model_id} </td>
                <td> {format_status(finetune.latest_status)} </td>
                <td> {formatDate(finetune.created_at)} </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </AppPage>
</div>
