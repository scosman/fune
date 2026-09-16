<script lang="ts">
  import AppPage from "../../../app_page.svelte"
  import { client } from "$lib/api_client"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { onMount } from "svelte"
  import { page } from "$app/stores"
  import { goto } from "$app/navigation"
  import type { KilnTaskToolDescription } from "$lib/types"
  import Warning from "$lib/ui/warning.svelte"
  import { formatDate } from "$lib/utils/formatters"
  import { ui_state } from "$lib/stores"

  import { agentInfo } from "$lib/agent"
  $: project_id = $page.params.project_id!
  $: agentInfo.set({
    name: "Kiln Task Tools",
    description: `Kiln task tools list for project ID ${project_id}. Shows tools that run Kiln tasks as tool calls.`,
  })

  let kiln_task_tools: KilnTaskToolDescription[] | null = null
  let loading = true
  let error: KilnError | null = null

  onMount(async () => {
    await fetch_kiln_task_tools()
    loading = false
  })

  async function fetch_kiln_task_tools() {
    try {
      error = null

      if (!project_id) {
        throw new Error("No project ID provided")
      }

      const { data, error: fetch_error } = await client.GET(
        "/api/projects/{project_id}/kiln_task_tools",
        {
          params: {
            path: {
              project_id,
            },
          },
        },
      )

      if (fetch_error) {
        throw fetch_error
      }

      // Sort by most recently created first
      kiln_task_tools =
        data?.sort(
          (a, b) =>
            new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
        ) || null
    } catch (err) {
      error = createKilnError(err)
    }
  }

  function navigateToToolDetails(tool: KilnTaskToolDescription) {
    goto(`/tools/${project_id}/kiln_task/${tool.tool_server_id}`)
  }
</script>

<div class="max-w-[1400px]">
  <AppPage
    title="Manage Kiln Tasks as Tools"
    subtitle="Allow your tasks to call another Kiln task, as a tool call."
    sub_subtitle="Read the Docs"
    sub_subtitle_link="https://docs.kiln.tech/docs/agents#multi-actor-interaction-aka-subtasks"
    breadcrumbs={[
      {
        label: "Optimize",
        href: `/optimize/${project_id}/${$ui_state.current_task_id}`,
      },
      {
        label: "Tools",
        href: `/tools/${project_id}`,
      },
    ]}
    action_buttons={[
      {
        label: "Create New",
        href: `/tools/${project_id}/add_tools/kiln_task`,
        primary: true,
      },
    ]}
  >
    {#if loading}
      <div class="w-full min-h-[50vh] flex justify-center items-center">
        <div class="loading loading-spinner loading-lg"></div>
      </div>
    {:else if error}
      <div
        class="w-full min-h-[50vh] flex flex-col justify-center items-center gap-2"
      >
        <div class="font-medium">Error Loading Kiln Task Tools</div>
        <div class="text-error text-sm">
          {error.getMessage() || "An unknown error occurred"}
        </div>
      </div>
    {:else if !kiln_task_tools || kiln_task_tools.length === 0}
      <div class="flex flex-col items-center justify-center min-h-[60vh]">
        <div class="text-center">
          <div class="text-2xl font-bold mb-2">No Kiln Task Tools</div>
          <div class="text-gray-500 mb-6">
            You haven't created any Kiln Task tools yet.
          </div>
        </div>
      </div>
    {:else}
      <div class="overflow-x-auto rounded-lg border mt-4">
        <table class="table table-fixed w-full">
          <thead>
            <tr>
              <th class="w-48">Tool Name</th>
              <th class="w-96">Description</th>
              <th class="w-28">Created At</th>
              <th class="w-20">Status</th>
            </tr>
          </thead>
          <tbody>
            {#each kiln_task_tools as tool}
              <tr
                class="hover:bg-base-200 cursor-pointer"
                on:click={() => navigateToToolDetails(tool)}
                on:keydown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault()
                    navigateToToolDetails(tool)
                  }
                }}
                role="button"
                tabindex={0}
              >
                <td class="align-top">
                  <div class="font-medium truncate" title={tool.tool_name}>
                    {tool.tool_name}
                  </div>
                  <!-- Tool names are user-typed and not unique, so the ID is what
                       tells two same-named tools apart. Shown on every row: an ID
                       that appears only on the ambiguous ones would make the rows
                       it is meant to clarify the odd ones out. -->
                  <div
                    class="text-xs text-gray-500 font-mono truncate"
                    title={tool.tool_server_id}
                  >
                    {tool.tool_server_id}
                  </div>
                </td>
                <td
                  class="text-sm truncate align-top"
                  title={tool.tool_description || "N/A"}
                  >{tool.tool_description || "N/A"}</td
                >
                <td class="text-sm whitespace-nowrap align-top"
                  >{formatDate(tool.created_at)}</td
                >
                <td class="text-sm align-top">
                  {#if tool.is_archived}
                    <Warning
                      warning_message="Archived"
                      warning_color="warning"
                      inline={true}
                    />
                  {:else}
                    <Warning
                      warning_message="Ready"
                      warning_color="success"
                      warning_icon="check"
                      inline={true}
                    />
                  {/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </AppPage>
</div>
