<script lang="ts">
  import AppPage from "../../../app_page.svelte"
  import { page } from "$app/stores"
  import { goto } from "$app/navigation"
  import DataGenIntro from "./data_gen_intro.svelte"
  import { indexedDBStore } from "$lib/stores/index_db_store"
  import { get, writable, type Writable } from "svelte/store"
  import { createQnaStore, type QnaStore } from "./qna/qna_ui_store"
  import { DEFAULT_QNA_GUIDANCE } from "./qna/guidance"
  import { agentInfo } from "$lib/agent"
  import { client } from "$lib/api_client"
  import { load_task } from "$lib/stores"
  import type { Task } from "$lib/types"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import Warning from "$lib/ui/warning.svelte"

  // watch out because query param value is not the same as gen_type
  type SynthReasonQueryParam = "eval" | "fine_tune" | "qna"

  let loading = true
  let load_error: KilnError | null = null
  let task: Task | null = null
  $: is_multiturn = task?.turn_mode === "multiturn"
  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  $: agentInfo.set({
    name: "Generate Data",
    description: `Data generation home for project ID ${project_id}, task ID ${task_id}. Choose between synthetic data generation and Q&A generation.`,
  })

  let cachedQnaStore: QnaStore | null = null
  let cachedQnaProjectId: string | null = null
  let cachedQnaTaskId: string | null = null
  let cachedQnaInitialized = false

  // Show the Data Guide action button only when one is already saved.
  // For first-time users (no guide yet), the entry point is the "Set Up Data
  // Guide" Intro inside the synth flow, not a top-bar shortcut.
  let has_data_guide: boolean = false

  // we only need gen_type to do the routing, the type-specific data is handled by the
  // mode-specific pages we redirect to
  type SavedDataGenState = {
    gen_type?: "training" | "eval" | "qna" | null
  }

  let saved_state: Writable<SavedDataGenState> = writable({
    gen_type: null,
  })

  let last_handled_key: string | null = null

  $: if (project_id && task_id) {
    // Refresh the Data Guide top-bar button state on every project/task
    // change. SvelteKit reuses this component across param-only navs, so
    // onMount won't refire and has_data_guide would otherwise go stale.
    void check_data_guide()
    const key = `${project_id}/${task_id}?${$page.url.searchParams.toString()}`
    if (last_handled_key !== key) {
      last_handled_key = key
      handle_routing(project_id, task_id)
    }
  }

  async function check_data_guide() {
    try {
      const { data } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/data_gen_guide",
        { params: { path: { project_id, task_id } } },
      )
      has_data_guide = !!data?.guide?.trim()
    } catch {
      has_data_guide = false
    }
  }

  async function handle_routing(req_project_id: string, req_task_id: string) {
    loading = true
    load_error = null

    // Synthetic data generation is single-turn only. For multi-turn tasks
    // don't redirect into the synth/qna flows — show the disabled notice.
    let loaded_task: Task | null
    try {
      loaded_task = await load_task(req_project_id, req_task_id)
    } catch (e) {
      if (req_project_id !== project_id || req_task_id !== task_id) return
      // Show the error instead of spinning forever. The key stays marked as
      // handled so we don't hot-loop retrying a failing load; navigating to a
      // different key (or remounting the route) retries naturally.
      load_error = createKilnError(e)
      loading = false
      return
    }
    if (req_project_id !== project_id || req_task_id !== task_id) return
    task = loaded_task
    if (task?.turn_mode === "multiturn") {
      loading = false
      return
    }

    const reason_param = $page.url.searchParams.get(
      "reason",
    ) as SynthReasonQueryParam | null

    // Eval/Fine-tuning modes redirect to synth page - this is when we explicitly link back
    // to the synth page (e.g. via toast UI)
    if (reason_param === "eval" || reason_param === "fine_tune") {
      const params = $page.url.searchParams
      await goto(
        `/generate/${req_project_id}/${req_task_id}/synth?${params.toString()}`,
      )
      return
    } else if (reason_param === "qna") {
      const params = $page.url.searchParams
      await goto(
        `/generate/${req_project_id}/${req_task_id}/qna?${params.toString()}`,
      )
      return
    } else if (reason_param) {
      //typecheck will flag this if we add a new case that we don't handle
      const invalid_reason: never = reason_param
      console.error(`Invalid reason: ${invalid_reason}`)
    }

    // user lands on this page without any specific state in the URL, we want to redirect
    // them to wherever they last were (i.e. synth page) if they have any ongoing session
    try {
      const currentSessionGenType = await getCurrentSessionGenType(
        req_project_id,
        req_task_id,
      )
      if (req_project_id !== project_id || req_task_id !== task_id) return
      switch (currentSessionGenType) {
        case "training":
          await goto(`/generate/${req_project_id}/${req_task_id}/synth`)
          return
        case "eval":
          await goto(`/generate/${req_project_id}/${req_task_id}/synth`)
          return
        case "qna":
          await goto(`/generate/${req_project_id}/${req_task_id}/qna`)
          return
        case null:
          // no ongoing session, stay on this page and show intro
          break
        default: {
          // invalid gen type - typecheck will flag if upstream typing adds a new case
          const value: never = currentSessionGenType
          console.error(`Invalid gen type: ${value}`)
          break
        }
      }
    } catch (error) {
      console.error("Error checking for ongoing session:", error)
    }

    if (req_project_id === project_id && req_task_id === task_id) {
      loading = false
    }
  }

  async function getCurrentSessionGenType(
    req_project_id: string,
    req_task_id: string,
  ): Promise<"training" | "eval" | "qna" | null> {
    // Check for saved Q&A session first
    if (
      !cachedQnaStore ||
      cachedQnaProjectId !== req_project_id ||
      cachedQnaTaskId !== req_task_id
    ) {
      cachedQnaStore = createQnaStore(req_project_id, req_task_id)
      cachedQnaProjectId = req_project_id
      cachedQnaTaskId = req_task_id
      cachedQnaInitialized = false
    }
    if (!cachedQnaInitialized) {
      await cachedQnaStore.init(DEFAULT_QNA_GUIDANCE)
      cachedQnaInitialized = true
    }
    const qna = cachedQnaStore
    if (get(qna).documents.length > 0) {
      return "qna"
    }
    // Check for saved synthetic data session
    const synth_data_key = `synth_data_${req_project_id}_${req_task_id}_v2`
    const { store, initialized } = indexedDBStore(synth_data_key, {
      gen_type: null,
      template_id: null,
      eval_id: null,
      splits: {},
      root_node: { topic: "", samples: [], sub_topics: [] },
    })
    // Wait for the store to be initialized, then set the state
    await initialized
    saved_state = store
    return $saved_state.gen_type || null
  }
</script>

<div class="max-w-[1400px]">
  <AppPage
    title="Synthetic Data Generation"
    no_y_padding
    sub_subtitle="Read the Docs"
    sub_subtitle_link="https://docs.kiln.tech/docs/synthetic-data-generation"
    action_buttons={has_data_guide && !is_multiturn
      ? [
          {
            label: "Data Guide",
            href: `/generate/${project_id}/${task_id}/data_guide`,
          },
        ]
      : []}
  >
    {#if loading}
      <div class="w-full min-h-[50vh] flex justify-center items-center">
        <div class="loading loading-spinner loading-lg"></div>
      </div>
    {:else if load_error}
      <div
        class="w-full min-h-[50vh] flex flex-col justify-center items-center gap-2"
      >
        <div class="font-medium">Error Loading Task</div>
        <div class="text-error text-sm">
          {load_error.getMessage()}
        </div>
      </div>
    {:else if is_multiturn}
      <div class="flex flex-col items-center justify-center min-h-[60vh]">
        <Warning
          warning_message="Synthetic data generation is not supported for multi-turn tasks."
          warning_color="warning"
          warning_icon="info"
        />
      </div>
    {:else}
      <DataGenIntro
        generate_subtopics={() => {}}
        generate_samples={() => {}}
        {project_id}
        {task_id}
        is_setup={false}
      />
    {/if}
  </AppPage>
</div>
