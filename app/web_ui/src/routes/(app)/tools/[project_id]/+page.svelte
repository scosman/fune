<script lang="ts">
  import AppPage from "../../app_page.svelte"
  import { client } from "$lib/api_client"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { onMount } from "svelte"
  import { page } from "$app/stores"
  import { goto } from "$app/navigation"
  import type {
    KilnTaskToolDescription,
    KilnToolServerDescription,
    CodeToolResponse,
  } from "$lib/types"
  import { toolServerTypeToString } from "$lib/utils/formatters"
  import EmptyTools from "../empty_tools.svelte"
  import Warning from "$lib/ui/warning.svelte"
  import { uncache_available_tools, ui_state } from "$lib/stores"
  import InfoTooltip from "$lib/ui/info_tooltip.svelte"
  import type { SearchToolApiDescription } from "$lib/types"

  import { agentInfo } from "$lib/agent"
  $: project_id = $page.params.project_id!
  $: agentInfo.set({
    name: "Tools",
    description: `Tools list for project ID ${project_id}. Shows connected tools including RAG search tools, Kiln task tools, and MCP servers.`,
  })
  $: is_empty =
    !demo_tools_enabled &&
    (!tools || tools.length == 0) &&
    (!kiln_task_tools || kiln_task_tools.length === 0) &&
    (!search_tools || search_tools.length === 0) &&
    (!code_tools || code_tools.length === 0)

  let tools: KilnToolServerDescription[] | null = null
  let demo_tools_enabled: boolean | null = null
  let search_tools: SearchToolApiDescription[] | null = null
  let kiln_task_tools: KilnTaskToolDescription[] | null = null
  let code_tools: CodeToolResponse[] | null = null
  let loading = true
  let error: KilnError | null = null

  onMount(async () => {
    await Promise.allSettled([
      fetch_available_tool_servers(),
      load_demo_tools(),
      load_rag_configs(),
      load_kiln_task_tools(),
      load_code_tools(),
    ])
    loading = false
  })

  $: unarchived_kiln_task_tools = kiln_task_tools?.filter(
    (tool) => !tool.is_archived,
  )

  $: unarchived_code_tools = (code_tools || []).filter((ct) => !ct.is_archived)
  $: archived_code_tools = (code_tools || []).filter((ct) => ct.is_archived)

  async function fetch_available_tool_servers() {
    try {
      error = null

      if (!project_id) {
        throw new Error("No project ID provided")
      }

      const { data, error: fetch_error } = await client.GET(
        "/api/projects/{project_id}/available_tool_servers",
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

      tools = data
    } catch (err) {
      error = createKilnError(err)
    }
  }

  async function load_demo_tools() {
    try {
      const { data, error } = await client.GET("/api/demo_tools")
      if (error) {
        throw error
      }
      demo_tools_enabled = data
    } catch (err) {
      console.error("Error loading demo tools", err)
      error = createKilnError(err)
    }
  }

  async function load_rag_configs() {
    try {
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/search_tools",
        {
          params: {
            path: {
              project_id,
            },
          },
        },
      )
      if (error) {
        throw error
      }
      search_tools = data
    } catch (err) {
      error = createKilnError(err)
      console.error("Error loading search tools", err)
    }
  }

  async function load_kiln_task_tools() {
    try {
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/kiln_task_tools",
        {
          params: {
            path: {
              project_id,
            },
          },
        },
      )
      if (error) {
        throw error
      }
      kiln_task_tools = data
    } catch (err) {
      error = createKilnError(err)
      console.error("Error loading kiln task tools count", err)
    }
  }

  async function load_code_tools() {
    try {
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/code_tools",
        {
          params: {
            path: {
              project_id,
            },
          },
        },
      )
      if (error) {
        throw error
      }
      code_tools = data
    } catch (err) {
      error = createKilnError(err)
      console.error("Error loading code tools", err)
    }
  }

  function navigateToToolServer(tool_server: KilnToolServerDescription) {
    if (tool_server.id) {
      if (tool_server.type === "kiln_task") {
        goto(`/tools/${project_id}/kiln_task/${tool_server.id}`)
      } else {
        goto(`/tools/${project_id}/tool_servers/${tool_server.id}`)
      }
    }
  }

  async function disable_demo_tools() {
    try {
      demo_tools_enabled = false
      const { error } = await client.POST("/api/demo_tools", {
        params: {
          query: {
            enable_demo_tools: false,
          },
        },
      })
      if (error) {
        throw error
      }
      // Delete the project_id from the available_tools, so next load it loads the updated list.
      uncache_available_tools(project_id)
    } catch (error) {
      console.error(error)
    }
  }
</script>

<div class="max-w-[1400px]">
  <AppPage
    title="Tools"
    subtitle="Connect your project to tools such as RAG systems, Kiln Tasks, and MCP servers"
    sub_subtitle="Read the Docs"
    sub_subtitle_link="https://docs.kiln.tech/docs/tools-and-mcp"
    breadcrumbs={[
      {
        label: "Optimize",
        href: `/optimize/${project_id}/${$ui_state.current_task_id}`,
      },
    ]}
    action_buttons={is_empty
      ? []
      : [
          {
            label: "Add Tools",
            href: `/tools/${project_id}/add_tools`,
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
        <div class="font-medium">Error Loading Tools</div>
        <div class="text-error text-sm">
          {error.getMessage() || "An unknown error occurred"}
        </div>
      </div>
    {:else if !is_empty}
      <div class="overflow-x-auto rounded-lg border mt-4">
        <table class="table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Description</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {#if search_tools && search_tools.length > 0}
              <tr
                class="hover:bg-base-200 cursor-pointer"
                on:click={() => goto(`/docs/rag_configs/${project_id}`)}
                on:keydown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault()
                    goto(`/docs/rag_configs/${project_id}`)
                  }
                }}
                role="button"
                tabindex="0"
              >
                <td class="font-medium">Search Tools (RAG)</td>
                <td class="text-sm">Kiln Search Tools</td>
                <td class="text-sm"
                  >Search tools retrieve information from your Kiln document
                  library.
                  {#if search_tools.length == 1}
                    One search tool available
                  {:else}
                    {search_tools.length} search tools available
                  {/if}
                  <InfoTooltip
                    tooltip_text="Tools: {search_tools
                      .map((tool) => tool.name + ' (' + tool.tool_name + ')')
                      .join(', ')}"
                    no_pad={true}
                  />
                </td>
                <td class="text-sm">
                  <Warning
                    warning_message="Ready"
                    warning_color="success"
                    warning_icon="check"
                    inline={true}
                  />
                </td>
              </tr>
            {/if}
            {#if kiln_task_tools && kiln_task_tools.length > 0 && unarchived_kiln_task_tools}
              <tr
                class="hover:bg-base-200 cursor-pointer"
                on:click={() => goto(`/tools/${project_id}/kiln_task_tools`)}
                on:keydown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault()
                    goto(`/tools/${project_id}/kiln_task_tools`)
                  }
                }}
                role="button"
                tabindex="0"
              >
                <td class="font-medium">Kiln Tasks as Tools</td>
                <td class="text-sm">Kiln Tasks</td>
                <td class="text-sm"
                  >Tools that run Kiln tasks using specified configurations.
                  {#if unarchived_kiln_task_tools.length == 0}
                    No Kiln task tools available
                  {:else if unarchived_kiln_task_tools.length == 1}
                    One Kiln task tool available
                  {:else}
                    {unarchived_kiln_task_tools.length} Kiln task tools available
                  {/if}
                  {#if unarchived_kiln_task_tools.length > 0}
                    <InfoTooltip
                      tooltip_text="Tools: {unarchived_kiln_task_tools
                        .map((tool) => tool.tool_name)
                        .join(', ')}"
                      no_pad={true}
                    />
                  {/if}
                </td>
                <td class="text-sm">
                  {#if unarchived_kiln_task_tools.length == 0}
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
            {/if}
            {#each [...unarchived_code_tools, ...archived_code_tools] as ct}
              <tr
                class="hover:bg-base-200 cursor-pointer"
                on:click={() =>
                  goto(`/tools/${project_id}/code_tools/${ct.id}`)}
                on:keydown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault()
                    goto(`/tools/${project_id}/code_tools/${ct.id}`)
                  }
                }}
                role="button"
                tabindex="0"
              >
                <td class="font-medium">{ct.name}</td>
                <td class="text-sm">Code Tool</td>
                <td class="text-sm">{ct.description || ct.tool_description}</td>
                <td class="text-sm">
                  {#if ct.is_archived}
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
            {#each (tools || []).filter((tool) => tool.type !== "kiln_task") as tool}
              {@const missing_secrets =
                tool.missing_secrets && tool.missing_secrets.length > 0}
              <tr
                class="hover:bg-base-200 cursor-pointer"
                on:click={() => navigateToToolServer(tool)}
                on:keydown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault()
                    navigateToToolServer(tool)
                  }
                }}
                role="button"
                tabindex={0}
              >
                <td class="font-medium">{tool.name}</td>
                <td class="text-sm">{toolServerTypeToString(tool.type)}</td>
                <td class="text-sm">{tool.description || "N/A"}</td>
                <td class="text-sm">
                  {#if missing_secrets}
                    <Warning
                      warning_message="Action Required"
                      warning_color="warning"
                      inline={true}
                    />
                  {:else if tool.is_archived}
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
            {#if demo_tools_enabled}
              <tr>
                <td class="font-medium">Math Demo Tools</td>
                <td class="text-sm">Built-in Tools</td>
                <td class="text-sm"
                  >Basic math tools: add, subtract, multiply, divide.
                  <button
                    class="link text-gray-500"
                    on:click={disable_demo_tools}
                  >
                    Disable
                  </button>
                </td>
                <td class="text-sm">
                  <Warning
                    warning_message="Ready"
                    warning_color="success"
                    warning_icon="check"
                    inline={true}
                  />
                </td>
              </tr>
            {/if}
          </tbody>
        </table>
      </div>
    {:else}
      <div class="flex flex-col items-center justify-center min-h-[50vh]">
        <EmptyTools {project_id} />
      </div>
    {/if}
  </AppPage>
</div>
