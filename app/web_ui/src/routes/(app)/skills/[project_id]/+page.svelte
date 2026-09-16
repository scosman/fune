<script lang="ts">
  import AppPage from "../../app_page.svelte"
  import Intro from "$lib/ui/intro.svelte"
  import Warning from "$lib/ui/warning.svelte"
  import { client } from "$lib/api_client"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { page } from "$app/stores"
  import { goto } from "$app/navigation"
  import { formatDate } from "$lib/utils/formatters"
  import type { Skill } from "$lib/types"
  import SkillsIcon from "$lib/ui/icons/skills_icon.svelte"
  import TableActionMenu from "$lib/ui/table_action_menu.svelte"
  import { ui_state } from "$lib/stores"

  import { agentInfo } from "$lib/agent"
  $: project_id = $page.params.project_id!
  $: agentInfo.set({
    name: "Skills",
    description: `Skills list for project ID ${project_id}. Reusable instructions for agents, loaded into context when needed.`,
  })

  let skills: Skill[] = []
  let loading = true
  let error: KilnError | null = null

  $: if (project_id) {
    fetch_skills(project_id)
  }

  async function fetch_skills(req_project_id: string) {
    try {
      loading = true
      error = null
      const { data, error: fetch_error } = await client.GET(
        "/api/projects/{project_id}/skills",
        {
          params: { path: { project_id: req_project_id } },
        },
      )
      if (req_project_id !== project_id) return
      if (fetch_error) {
        throw fetch_error
      }
      skills = data
    } catch (err) {
      if (req_project_id !== project_id) return
      error = createKilnError(err)
    } finally {
      if (req_project_id === project_id) {
        loading = false
      }
    }
  }

  $: sorted_skills = skills
    .slice()
    .sort(
      (a, b) =>
        (b.created_at ? new Date(b.created_at).getTime() : 0) -
        (a.created_at ? new Date(a.created_at).getTime() : 0),
    )
</script>

<div class="max-w-[1400px]">
  <AppPage
    title="Skills"
    subtitle="Reusable instructions for your agents, loaded into context only when needed."
    sub_subtitle="Read the Docs"
    sub_subtitle_link="https://docs.kiln.tech/docs/skills"
    breadcrumbs={[
      {
        label: "Optimize",
        href: `/optimize/${project_id}/${$ui_state.current_task_id}`,
      },
    ]}
    action_buttons={!loading && skills.length === 0
      ? []
      : [
          {
            label: "Add Skill",
            href: `/skills/${project_id}/create`,
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
        <div class="font-medium">Error Loading Skills</div>
        <div class="text-error text-sm">
          {error.getMessage() || "An unknown error occurred"}
        </div>
      </div>
    {:else if !loading && skills.length === 0}
      <div class="flex flex-col items-center justify-center min-h-[50vh]">
        <Intro
          title="Extend your agent with Skills"
          description_paragraphs={[
            "Skills provide domain knowledge and reusable workflows.",
            "They load on demand so your agent stays focused and efficient.",
          ]}
          action_buttons={[
            {
              label: "Add Skill",
              href: `/skills/${project_id}/create`,
              is_primary: true,
            },
            {
              label: "Docs & Guide",
              href: "https://docs.kiln.tech/docs/skills",
              is_primary: false,
              new_tab: true,
            },
          ]}
        >
          <div slot="icon">
            <div class="h-12 w-12">
              <SkillsIcon />
            </div>
          </div>
        </Intro>
      </div>
    {:else}
      <div class="overflow-x-auto rounded-lg border mt-4">
        <table class="table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Description</th>
              <th>Created At</th>
              <th>Status</th>
              <th class="w-16">Actions</th>
            </tr>
          </thead>
          <tbody>
            {#each sorted_skills as skill}
              <tr
                class="hover:bg-base-200 cursor-pointer"
                class:opacity-60={skill.is_archived}
                on:click={() => goto(`/skills/${project_id}/${skill.id}`)}
                on:keydown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault()
                    goto(`/skills/${project_id}/${skill.id}`)
                  }
                }}
                role="button"
                tabindex="0"
              >
                <td class="font-medium">{skill.name}</td>
                <td class="text-sm max-w-[400px] truncate"
                  >{skill.description}</td
                >
                <td class="text-sm whitespace-nowrap"
                  >{formatDate(skill.created_at ?? undefined)}</td
                >
                <td class="text-sm">
                  {#if skill.is_archived}
                    <Warning
                      warning_message="Archived"
                      warning_color="warning"
                      inline={true}
                    />
                  {:else}
                    <Warning
                      warning_message="Active"
                      warning_color="success"
                      warning_icon="check"
                      inline={true}
                    />
                  {/if}
                </td>
                <td>
                  <TableActionMenu
                    width="w-48"
                    items={[
                      {
                        label: "Clone",
                        onclick: () =>
                          goto(`/skills/${project_id}/clone/${skill.id}`),
                      },
                    ]}
                  />
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </AppPage>
</div>
