<script lang="ts">
  import AppPage from "../../../app_page.svelte"
  import { client } from "$lib/api_client"
  import type { PromptOptimizationJob, Task } from "$lib/types"
  import { createKilnError, KilnError } from "$lib/utils/error_handlers"
  import { goto } from "$app/navigation"
  import { page } from "$app/stores"
  import { load_task } from "$lib/stores"
  import { formatDate } from "$lib/utils/formatters"
  import Warning from "$lib/ui/warning.svelte"
  import Intro from "$lib/ui/intro.svelte"
  import OptimizeIcon from "$lib/ui/icons/optimize_icon.svelte"
  import { checkKilnCopilotAvailable } from "$lib/utils/copilot_utils"
  import { checkPromptOptimizationAccess } from "$lib/utils/entitlement_utils"
  import PromptOptimizationCopilotRequired from "./prompt_optimization_copilot_required.svelte"
  import EntitlementRequiredCard from "$lib/ui/kiln_copilot/entitlement_required_card.svelte"

  import { agentInfo } from "$lib/agent"
  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  $: agentInfo.set({
    name: "Prompt Optimization",
    description: `Prompt optimization jobs list for project ID ${project_id}, task ID ${task_id}. Shows optimization job history and results.`,
  })

  let loading = true

  let task: Task | null = null
  let task_error: KilnError | null = null
  let prompt_optimization_jobs: PromptOptimizationJob[] | null = null
  let prompt_optimization_jobs_error: KilnError | null = null

  let kiln_copilot_connected: boolean | null = null
  let has_prompt_optimization_entitlement: boolean | null = null
  let copilot_check_error: KilnError | null = null

  $: error = copilot_check_error || prompt_optimization_jobs_error || task_error
  $: is_empty =
    !prompt_optimization_jobs || prompt_optimization_jobs.length === 0
  $: is_multiturn = task?.turn_mode === "multiturn"

  $: if (project_id && task_id) {
    task = null
    task_error = null
    load_task_for_page(project_id, task_id)
    load_jobs_and_copilot_status(project_id, task_id)
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
      if (req_project_id !== project_id || req_task_id !== task_id) return
      task_error = createKilnError(e)
    }
  }

  async function load_jobs_and_copilot_status(
    req_project_id: string,
    req_task_id: string,
  ) {
    loading = true
    copilot_check_error = null
    await get_prompt_optimization_jobs(req_project_id, req_task_id)

    if (req_project_id !== project_id || req_task_id !== task_id) return

    if (!prompt_optimization_jobs || prompt_optimization_jobs.length === 0) {
      try {
        kiln_copilot_connected = await checkKilnCopilotAvailable()
      } catch (e) {
        if (req_project_id !== project_id || req_task_id !== task_id) return
        copilot_check_error = createKilnError(e)
        kiln_copilot_connected = false
      }

      if (req_project_id !== project_id || req_task_id !== task_id) return

      if (kiln_copilot_connected) {
        const { has_access, error: entitlement_error } =
          await checkPromptOptimizationAccess()
        if (req_project_id !== project_id || req_task_id !== task_id) return
        has_prompt_optimization_entitlement = has_access
        if (entitlement_error) {
          copilot_check_error = entitlement_error
        }
      }
    }

    if (req_project_id === project_id && req_task_id === task_id) {
      loading = false
    }
  }

  async function get_prompt_optimization_jobs(
    req_project_id: string,
    req_task_id: string,
  ) {
    try {
      prompt_optimization_jobs_error = null
      const { data: prompt_optimization_jobs_response, error: get_error } =
        await client.GET(
          "/api/projects/{project_id}/tasks/{task_id}/prompt_optimization_jobs",
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
      const sorted_prompt_optimization_jobs =
        prompt_optimization_jobs_response.sort((a, b) => {
          return (
            new Date(b.created_at || "").getTime() -
            new Date(a.created_at || "").getTime()
          )
        })
      prompt_optimization_jobs = sorted_prompt_optimization_jobs
    } catch (e) {
      if (req_project_id !== project_id || req_task_id !== task_id) return
      if (e instanceof Error && e.message.includes("Load failed")) {
        prompt_optimization_jobs_error = new KilnError(
          "Could not load Prompt Optimization jobs. This task may belong to a project you don't have access to.",
          null,
        )
      } else {
        prompt_optimization_jobs_error = createKilnError(e)
      }
    }
  }

  type StatusBadge = { label: string; badge_class: string }
  const status_badge_map: Record<string, StatusBadge> = {
    pending: { label: "Pending", badge_class: "badge-outline" },
    running: { label: "Running", badge_class: "badge-outline" },
    succeeded: {
      label: "Complete",
      badge_class: "badge-outline badge-primary",
    },
    failed: { label: "Failed", badge_class: "badge-outline badge-error" },
    cancelled: {
      label: "Cancelled",
      badge_class: "badge-outline badge-warning",
    },
  }
  function get_status_badge(status: string): StatusBadge {
    return (
      status_badge_map[status] || {
        label: status,
        badge_class: "badge-outline",
      }
    )
  }
</script>

<AppPage
  title="Prompt Optimizer Jobs"
  sub_subtitle="Read the Docs"
  sub_subtitle_link="https://docs.kiln.tech/docs/prompts/automatic-prompt-optimizer"
  breadcrumbs={[
    {
      label: "Optimize",
      href: `/optimize/${project_id}/${task_id}`,
    },
    {
      label: "Prompts",
      href: `/prompts/${project_id}/${task_id}`,
    },
  ]}
  action_buttons={task === null || is_multiturn || is_empty || error
    ? []
    : [
        {
          label: "Create Optimized Prompt",
          href: `/prompt_optimization/${project_id}/${task_id}/create_prompt_optimization_job`,
          primary: true,
        },
      ]}
>
  {#if loading || (task === null && !task_error)}
    <!-- Wait for the task to resolve before choosing a content branch: the
         multi-turn gate must come from a loaded task, never a null default, or
         a multi-turn task would flash the single-turn intro/jobs UI. -->
    <div class="w-full min-h-[50vh] flex justify-center items-center">
      <div class="loading loading-spinner loading-lg"></div>
    </div>
  {:else if error}
    <div
      class="w-full min-h-[50vh] flex flex-col justify-center items-center gap-2"
    >
      <div class="text-error text-sm">
        {error.getMessage() || "An unknown error occurred"}
      </div>
    </div>
  {:else if is_multiturn}
    <div class="flex flex-col items-center justify-center min-h-[60vh]">
      <Warning
        warning_message="Prompt optimization is not supported for multi-turn tasks."
        warning_color="warning"
        warning_icon="info"
      />
    </div>
  {:else if is_empty}
    {#if kiln_copilot_connected === false}
      <PromptOptimizationCopilotRequired />
    {:else if has_prompt_optimization_entitlement === false}
      <EntitlementRequiredCard feature_name="Prompt Optimization" />
    {:else}
      <div class="flex flex-col items-center justify-center min-h-[60vh]">
        <Intro
          title="Kiln Prompt Optimizer automatically optimizes your prompts."
          description_paragraphs={[
            "Improve eval performance by optimizing your prompts using training data.",
          ]}
          align_title_left={true}
          action_buttons={[
            {
              label: "Create Optimized Prompt",
              href: `/prompt_optimization/${project_id}/${task_id}/create_prompt_optimization_job`,
              is_primary: true,
            },
          ]}
        >
          <div slot="icon">
            <div class="h-12 w-12">
              <OptimizeIcon />
            </div>
          </div>
        </Intro>
      </div>
    {/if}
  {:else if prompt_optimization_jobs}
    <div class="overflow-x-auto rounded-lg border">
      <table class="table">
        <thead>
          <tr>
            <th> Name </th>
            <th> Status </th>
            <th> Created At </th>
          </tr>
        </thead>
        <tbody>
          {#each prompt_optimization_jobs as prompt_optimization_job}
            {@const badge = get_status_badge(
              prompt_optimization_job.latest_status,
            )}
            <tr
              class="hover cursor-pointer"
              on:click={() => {
                goto(
                  `/prompt_optimization/${project_id}/${task_id}/prompt_optimization_job/${prompt_optimization_job.id}`,
                )
              }}
            >
              <td> {prompt_optimization_job.name} </td>
              <td>
                <div class="badge px-3 py-1 gap-1 {badge.badge_class}">
                  {#if prompt_optimization_job.latest_status === "pending" || prompt_optimization_job.latest_status === "running"}
                    <span class="loading loading-spinner h-[12px] w-[12px]"
                    ></span>
                  {/if}
                  {badge.label}
                </div>
              </td>
              <td> {formatDate(prompt_optimization_job.created_at)} </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</AppPage>
