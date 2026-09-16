<script lang="ts">
  import AppPage from "../../../../../app_page.svelte"
  import type { ActionButton } from "$lib/types"
  import Run from "../../../../../run/run.svelte"
  import Output from "$lib/ui/output.svelte"
  import {
    get_task_composite_id,
    load_task,
    model_name,
    model_info,
    load_model_info,
    prompt_name_from_id,
    provider_name_from_id,
    load_available_models,
    load_available_tools,
    available_tools,
    current_task,
  } from "$lib/stores"
  import {
    prompts_by_task_composite_id,
    load_task_prompts,
  } from "$lib/stores/prompts_store"
  import { page } from "$app/stores"
  import { getContext, onDestroy, tick } from "svelte"
  import { client } from "$lib/api_client"
  import { createKilnError, KilnError } from "$lib/utils/error_handlers"
  import type {
    Task,
    TaskRun,
    StructuredOutputMode,
    RunChainEntry,
  } from "$lib/types"
  import { isMcpRunConfig } from "$lib/types"
  import {
    formatDate,
    structuredOutputModeToString,
  } from "$lib/utils/formatters"
  import { goto } from "$app/navigation"
  import DeleteDialog from "$lib/ui/delete_dialog.svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import PropertyList from "$lib/ui/property_list.svelte"
  import type { UiProperty } from "$lib/ui/property_list"
  import { dataset_item_link, prompt_link } from "$lib/utils/link_builder"
  import type { ProviderModels, PromptResponse, TraceMessage } from "$lib/types"
  import { isMacOS } from "$lib/utils/platform"
  import type { Writable } from "svelte/store"
  import {
    get_tools_property_info,
    get_tool_names_from_ids,
    get_tool_server_name,
    split_tool_and_skill_ids,
  } from "$lib/stores/tools_store"
  import { agentInfo } from "$lib/agent"
  import ChatTrace from "$lib/ui/trace/chat_trace.svelte"
  import ChatLoading from "$lib/ui/conversation/chat_thinking_loading.svelte"
  import MultiturnComposer from "$lib/ui/conversation/multiturn_composer.svelte"
  import RunConfigComponent from "$lib/ui/run_config_component/run_config_component.svelte"
  import SavedRunConfigurationsDropdown from "$lib/ui/run_config_component/saved_run_configs_dropdown.svelte"
  import { isKilnAgentRunConfig, type TaskRunConfig } from "$lib/types"
  import Warning from "$lib/ui/warning.svelte"
  import RunSidebar from "$lib/ui/run_sidebar.svelte"
  import {
    compute_forkable_run_ids,
    fork_target_from_assistant_block,
    type ForkTarget,
  } from "./fork_helpers"

  $: run_id = $page.params.run_id!
  $: task_id = $page.params.task_id!
  $: project_id = $page.params.project_id!

  $: agentInfo.set({
    name: "Dataset Run Detail",
    description: `Detail view for run ID ${run_id} in project ID ${project_id}, task ID ${task_id}. Shows run input, output, rating, model info, and repair options.`,
  })
  // @ts-expect-error list_page is not a property of PageState
  $: list_page = ($page.state.list_page || []) as string[]

  let task: Task | null = null
  let run: TaskRun | null = null
  let loading = true
  let load_error: KilnError | null = null

  // The multiturn view is a chat layout where the composer sits at the bottom
  // of the conversation with its own padding, so it needs the app shell's
  // bottom padding removed. Single-turn keeps the normal padded document flow.
  $: is_multiturn = task?.turn_mode === "multiturn"
  const noLayoutBottomPadding = getContext<Writable<boolean> | undefined>(
    "noLayoutBottomPadding",
  )
  $: noLayoutBottomPadding?.set(!!is_multiturn)
  onDestroy(() => noLayoutBottomPadding?.set(false))
  let see_all_properties = false
  let raw_data_dialog: Dialog | null = null
  let tools_property_value: string | string[] = "Loading..."
  let tool_links: (string | null)[] | undefined
  let skills_property_value: string | string[] = "None"
  let skill_links: (string | null)[] | undefined

  $: {
    const run_config = run?.output?.source?.run_config
    let all_ids: string[] = []
    if (!run_config) {
      all_ids = []
    } else {
      const run_config_type = run_config.type
      switch (run_config_type) {
        case "mcp":
          all_ids = []
          break
        case "kiln_agent":
          all_ids = run_config.tools_config?.tools ?? []
          break
        default: {
          const _exhaustive: never = run_config_type
          throw new Error(`Unknown run config type: ${_exhaustive}`)
        }
      }
    }
    const { tool_ids, skill_ids } = split_tool_and_skill_ids(all_ids)
    const tools_property_info = get_tools_property_info(
      tool_ids,
      project_id,
      $available_tools,
    )
    tools_property_value = tools_property_info.value
    tool_links = tools_property_info.links
    const skills_property_info = get_tools_property_info(
      skill_ids,
      project_id,
      $available_tools,
    )
    skills_property_value = skills_property_info.value
    skill_links = skills_property_info.links
  }

  function get_kiln_agent_properties(
    run: TaskRun | null,
    current_task_prompts: PromptResponse | null,
    model_info: ProviderModels | null,
  ): UiProperty[] {
    const properties: UiProperty[] = []
    const model_id = run?.output?.source?.properties?.model_name
    if (model_id && typeof model_id === "string") {
      properties.push({
        name: "Output Model",
        value: model_name(model_id, model_info),
      })
    }

    // Prompt ID previously was stored in the prompt_builder_name field
    let prompt_id = (
      run?.output?.source?.properties?.prompt_id ||
      run?.output?.source?.properties?.prompt_builder_name ||
      ""
    ).toString()
    if (prompt_id) {
      const prompt_name = prompt_name_from_id(prompt_id, current_task_prompts)
      if (prompt_name) {
        let link = prompt_link(project_id, task_id, prompt_id)
        properties.push({
          name: "Prompt",
          value: prompt_name,
          link: link,
        })
      }
    }

    properties.push({
      name: "Available Tools",
      value: tools_property_value,
      links: tool_links,
      badge: Array.isArray(tools_property_value) ? true : false,
      collapse_badges: true,
    })
    properties.push({
      name: "Available Skills",
      value: skills_property_value,
      links: skill_links,
      badge: Array.isArray(skills_property_value) ? true : false,
      collapse_badges: true,
    })
    return properties
  }

  function get_mcp_properties(run: TaskRun | null): UiProperty[] {
    const run_config = run?.output?.source?.run_config
    const tool_id =
      (isMcpRunConfig(run_config)
        ? run_config.tool_reference?.tool_id
        : null) || run?.output?.source?.properties?.tool_id

    const tool_name =
      typeof tool_id === "string"
        ? get_tool_names_from_ids(
            [tool_id],
            $available_tools[project_id] || [],
          )[0]
        : null
    const tool_server_name =
      typeof tool_id === "string"
        ? get_tool_server_name($available_tools, project_id, tool_id)
        : null

    const properties: UiProperty[] = [
      {
        name: "MCP Tool",
        value: tool_name || "Unknown",
      },
    ]

    if (tool_server_name) {
      properties.push({
        name: "Tool Server",
        value: tool_server_name,
      })
    }

    return properties
  }

  function get_kiln_agent_advanced_properties(
    run: TaskRun | null,
  ): UiProperty[] {
    const properties: UiProperty[] = []
    if (run?.output?.source?.properties?.model_provider) {
      properties.push({
        name: "Model Provider",
        value: provider_name_from_id(
          String(run.output.source.properties.model_provider),
        ),
      })
    }

    if (run?.output?.source?.properties?.temperature !== undefined) {
      properties.push({
        name: "Temperature",
        value: run.output.source.properties.temperature,
      })
    }

    if (run?.output?.source?.properties?.top_p !== undefined) {
      properties.push({
        name: "Top P",
        value: run.output.source.properties.top_p,
      })
    }

    if (run?.output?.source?.properties?.structured_output_mode) {
      let mode = run.output.source.properties.structured_output_mode
      if (typeof mode === "string") {
        const json_mode = structuredOutputModeToString(
          mode as StructuredOutputMode,
        )
        if (json_mode) {
          properties.push({
            name: "JSON Mode",
            value: json_mode,
          })
        }
      }
    }
    return properties
  }

  function get_properties(
    run: TaskRun | null,
    current_task_prompts: PromptResponse | null,
    model_info: ProviderModels | null,
  ) {
    let properties: UiProperty[] = []
    const run_config = run?.output?.source?.run_config

    if (run?.id) {
      properties.push({
        name: "ID",
        value: run.id,
      })
    }

    if (run?.parent_task_run_id) {
      const parent_link = dataset_item_link(
        project_id,
        task_id,
        run.parent_task_run_id,
      )
      properties.push({
        name: "Parent ID",
        value: run.parent_task_run_id,
        link: parent_link ?? undefined,
      })
    }

    if (run?.input_source?.type) {
      properties.push({
        name: "Input Source",
        value:
          run.input_source.type.charAt(0).toUpperCase() +
          run.input_source.type.slice(1),
      })
    }

    if (!run_config) {
      // if run_config is null, render the kiln agent properties
      properties.push(
        ...get_kiln_agent_properties(run, current_task_prompts, model_info),
      )
    } else {
      const run_config_type = run_config.type
      switch (run_config_type) {
        case "mcp": {
          properties.push(...get_mcp_properties(run))
          break
        }
        case "kiln_agent":
          // if run_config is kiln_agent, render the kiln agent properties
          properties.push(
            ...get_kiln_agent_properties(run, current_task_prompts, model_info),
          )
          break
        default: {
          const _exhaustive: never = run_config_type
          throw new Error(`Unknown run config type: ${_exhaustive}`)
        }
      }
    }

    if (run?.created_at) {
      properties.push({
        name: "Created At",
        value: formatDate(run.created_at),
      })
    }

    let topic_path: string | undefined = undefined
    if (
      run?.input_source?.properties?.topic_path &&
      typeof run?.input_source?.properties?.topic_path === "string"
    ) {
      topic_path = run?.input_source?.properties?.topic_path?.replaceAll(
        ">>>>>",
        " > ",
      )
    }
    if (topic_path) {
      properties.push({
        name: "Topic",
        value: topic_path,
      })
    }

    return properties
  }

  function get_advanced_properties(run: TaskRun | null) {
    let properties: UiProperty[] = []
    const run_config = run?.output?.source?.run_config
    if (!run_config) {
      properties.push(...get_kiln_agent_advanced_properties(run))
    } else {
      const run_config_type = run_config.type
      switch (run_config_type) {
        case "mcp":
          break
        case "kiln_agent": {
          properties.push(...get_kiln_agent_advanced_properties(run))
          break
        }
        default: {
          const _exhaustive: never = run_config_type
          throw new Error(`Unknown run config type: ${_exhaustive}`)
        }
      }
    }

    if (run?.input_source?.properties?.created_by) {
      properties.push({
        name: "Created By",
        value: run.input_source.properties.created_by,
      })
    }

    return properties
  }

  let properties_for_list: UiProperty[] = []
  $: {
    void tools_property_value
    void skills_property_value
    properties_for_list = [
      ...get_properties(
        run,
        $prompts_by_task_composite_id[
          get_task_composite_id(project_id, task_id)
        ] ?? null,
        $model_info,
      ),
      ...(see_all_properties ? get_advanced_properties(run) : []),
    ]
  }

  // The /run page hands the just-created first-turn run over via navigation
  // state. Seed it (plus the task, from the current-task store set on /run) so
  // we render the conversation immediately instead of flashing the full-page
  // loading spinner while load_run / load_task re-fetch them (both still run
  // below to refresh). We only drop `loading` once both are present, so we
  // never briefly render the "Run not found" branch (which needs run && task).
  $: {
    // @ts-expect-error created_run is not a declared property of PageState
    const seeded_run = $page.state?.created_run as TaskRun | undefined
    if (seeded_run?.id && seeded_run.id === run_id && run === null) {
      run = seeded_run
      if (!task && $current_task?.id === task_id) {
        task = $current_task
      }
      if (run && task) {
        loading = false
      }
    }
  }

  $: if (project_id && task_id && run_id) {
    load_run(project_id, task_id, run_id)
    load_task_for_page(project_id, task_id)
    load_task_prompts(project_id, task_id)
    load_available_tools(project_id)
    load_model_info()
    load_available_models()
  }

  async function load_task_for_page(
    req_project_id: string,
    req_task_id: string,
  ) {
    const loaded = await load_task(req_project_id, req_task_id)
    if (req_project_id !== project_id || req_task_id !== task_id) return
    task = loaded
  }

  async function load_run(
    req_project_id: string,
    req_task_id: string,
    req_run_id: string,
  ) {
    try {
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/runs/{run_id}",
        {
          params: {
            path: {
              project_id: req_project_id,
              task_id: req_task_id,
              run_id: req_run_id,
            },
          },
        },
      )
      if (
        req_project_id !== project_id ||
        req_task_id !== task_id ||
        req_run_id !== run_id
      )
        return
      if (error) {
        throw error
      }
      run = data
    } catch (error) {
      if (
        req_project_id !== project_id ||
        req_task_id !== task_id ||
        req_run_id !== run_id
      )
        return
      if (error instanceof Error && error.message.includes("Load failed")) {
        load_error = new KilnError(
          "Could not load run. It may belong to a project you don't have access to.",
          null,
        )
      } else {
        load_error = createKilnError(error)
      }
    } finally {
      if (
        req_project_id === project_id &&
        req_task_id === task_id &&
        req_run_id === run_id
      ) {
        loading = false
      }
    }
  }

  let delete_dialog: DeleteDialog | null = null
  let deleted: Record<string, boolean> = {}
  $: delete_url = `/api/projects/${project_id}/tasks/${task_id}/runs/${run_id}`
  function after_delete() {
    deleted[run_id] = true
  }

  function next_run() {
    const index = list_page.indexOf(run_id)
    if (index < list_page.length - 1) {
      const next_run_id = list_page[index + 1]
      load_run_by_id(next_run_id)
    }
  }

  function prev_run() {
    const index = list_page.indexOf(run_id)
    if (index > 0) {
      const prev_run_id = list_page[index - 1]
      load_run_by_id(prev_run_id)
    }
  }

  function load_run_by_id(new_run_id: string) {
    load_error = null
    run = null
    loading = true
    goto(`/dataset/${project_id}/${task_id}/${new_run_id}/run`, {
      state: { list_page: list_page },
    })
  }

  async function handle_send(new_run_id: string) {
    load_error = null
    // Deliberately do NOT clear `run` or flip `loading` here: keeping the
    // current transcript (with its optimistic message + loading indicator)
    // mounted while we navigate avoids the blank full-page spinner flash.
    // `awaiting_response` keeps the composer disabled until the new run
    // renders, and the run-load reactive clears the optimistic state then.
    // noScroll: SvelteKit otherwise jumps to the top of the page on navigate,
    // which would flash the top before our pin scrolls back to the latest turn.
    await goto(`/dataset/${project_id}/${task_id}/${new_run_id}/run`, {
      replaceState: true,
      noScroll: true,
    })
  }

  // The transcript content element. The whole page scrolls (no inner scroll
  // region) — we observe this element for content mutations and pin the window
  // scroll to the bottom while things settle.
  let transcript_scroll_el: HTMLElement | null = null
  // The composer block (the last message's composer + "Show Raw Data"). We
  // scroll to the bottom of THIS element rather than the document's or the
  // chat column's: the document height is dominated by the Options sidebar
  // when the conversation is short, and the chat column itself stretches to
  // the sidebar's height (flex-row align-items: stretch), so neither of their
  // bottoms tracks the conversation. The composer block's bottom always lands
  // just past "Show Raw Data", keeping the latest turn and composer in view.
  let composer_block_el: HTMLElement | null = null
  // Scroll the page to the latest turn whenever a run renders — both on
  // initial load and after sending a new turn.
  let scrolled_for_run_id: string | null = null
  $: if (
    run &&
    run.id === run_id &&
    task?.turn_mode === "multiturn" &&
    run.id !== scrolled_for_run_id
  ) {
    scrolled_for_run_id = run.id ?? null
    // The freshly-loaded run's trace already contains the just-sent turn, so
    // drop the optimistic placeholder + loader to avoid showing them twice.
    // For a fork, this is also where the held truncation is released — the new
    // branch is now rendered, so we show it in full.
    optimistic_sent_message = null
    awaiting_response = false
    fork_send_trace_index = null
    apply_transcript_scroll()
  }

  // While a send is in flight we append the user's message to the transcript
  // optimistically and show a loading indicator below it, then redirect once
  // the run lands. `awaiting_response` stays true from send-start until the
  // new run renders (cleared above), so the loader shows continuously across
  // the navigation and the composer stays disabled the whole time.
  let optimistic_sent_message: string | null = null
  let awaiting_response = false
  $: display_trace = build_display_trace(
    run?.trace ?? [],
    optimistic_sent_message,
  )
  function build_display_trace(
    trace: TraceMessage[],
    optimistic: string | null,
  ): TraceMessage[] {
    if (optimistic === null) return trace
    return [...trace, { role: "user", content: optimistic } as TraceMessage]
  }

  function handle_send_start(text: string) {
    optimistic_sent_message = text
    awaiting_response = true
    apply_transcript_scroll()
  }

  function handle_send_settled(ok: boolean) {
    // On error no new run loads, so clear the optimistic state here. On
    // success we leave it: the run-load reactive clears it once the new run
    // (with the real response) renders, keeping the loader up until then.
    if (!ok) {
      optimistic_sent_message = null
      awaiting_response = false
      // Fork failed: release the held truncation. fork_target is still set (no
      // navigation happened), so the fork composer stays open at the same point.
      fork_send_trace_index = null
    }
  }

  // Safety net: if loading the new run fails after a successful send, don't
  // leave the loader spinning or the composer disabled forever.
  $: if (load_error) {
    awaiting_response = false
    optimistic_sent_message = null
    fork_send_trace_index = null
  }

  async function apply_transcript_scroll() {
    // Wait for ChatTrace (keyed on run.id) to render the new trace, then a
    // frame so layout settles before measuring/scrolling.
    await tick()
    requestAnimationFrame(() => {
      pin_transcript_to_bottom()
    })
  }

  // Keep the transcript pinned to the bottom for a short window after load.
  // A single scroll isn't enough: ChatMarkdown (and any images) reflow after
  // our initial frames, growing scrollHeight, which is why it lands "almost"
  // at the bottom. We re-pin on every mutation until things settle, then stop
  // so we don't fight the user's own scrolling.
  let settle_observer: MutationObserver | null = null
  let settle_timeout: ReturnType<typeof setTimeout> | null = null

  function pin_transcript_to_bottom() {
    const el = transcript_scroll_el
    if (!el || typeof MutationObserver === "undefined") return
    stop_pinning_transcript()
    const stick = () => {
      const block = composer_block_el
      if (!block) {
        window.scrollTo({ top: document.documentElement.scrollHeight })
        return
      }
      // Scroll so the bottom of the composer block ("Show Raw Data") sits at
      // the bottom of the viewport, regardless of how tall the Options sidebar
      // is or whether it wrapped below in the mobile layout.
      const target =
        window.scrollY +
        block.getBoundingClientRect().bottom -
        window.innerHeight
      window.scrollTo({ top: Math.max(0, target) })
    }
    stick()
    settle_observer = new MutationObserver(() => requestAnimationFrame(stick))
    settle_observer.observe(el, {
      childList: true,
      subtree: true,
      characterData: true,
    })
    settle_timeout = setTimeout(stop_pinning_transcript, 1000)
  }

  function stop_pinning_transcript() {
    settle_observer?.disconnect()
    settle_observer = null
    if (settle_timeout) {
      clearTimeout(settle_timeout)
      settle_timeout = null
    }
  }

  onDestroy(stop_pinning_transcript)

  // ---- Multiturn composer state ----
  let multiturn_run_config_component: RunConfigComponent
  let multiturn_save_config_error: KilnError | null = null
  let multiturn_set_default_error: KilnError | null = null
  let multiturn_selected_run_config_id: string | null = null
  let multiturn_selected_model_specific_run_config_id: string | null = null

  function multiturn_initial_model(r: TaskRun | null): string {
    const cfg = r?.output?.source?.run_config ?? null
    if (cfg && isKilnAgentRunConfig(cfg)) {
      return `${cfg.model_provider_name}/${cfg.model_name}`
    }
    return ""
  }
  function multiturn_initial_prompt(r: TaskRun | null): string {
    const cfg = r?.output?.source?.run_config ?? null
    if (cfg && isKilnAgentRunConfig(cfg)) {
      return cfg.prompt_id
    }
    return "simple_prompt_builder"
  }

  async function handle_save_new_multiturn_run_config(): Promise<TaskRunConfig> {
    if (!multiturn_run_config_component) {
      throw new Error("Run configuration component is not loaded")
    }
    return await multiturn_run_config_component.save_new_run_config()
  }

  // ---- Run chain / fork state ----
  let run_chain: RunChainEntry[] = []
  let chain_broken = false
  let chain_load_failed = false
  let run_has_children = false
  let chain_loaded_for_run_id: string | null = null
  let fork_target: ForkTarget | null = null
  // Holds the fork point's truncation across the navigation to the forked run.
  // fork_target is reset the instant run_id changes (below), so on its own the
  // transcript would snap back to the full untruncated conversation mid-send;
  // this survives the navigation and is cleared only once the forked run
  // renders, mirroring how awaiting_response gates the append flow.
  let fork_send_trace_index: number | null = null

  // Reset fork + chain state whenever the run id changes so we don't surface
  // stale data (banners, suffix-aligned mappings) from the previous run
  // before the new fetch resolves.
  $: if (run_id) {
    fork_target = null
    run_chain = []
    chain_broken = false
    chain_load_failed = false
    run_has_children = false
  }

  $: if (
    task &&
    run &&
    task.turn_mode === "multiturn" &&
    chain_loaded_for_run_id !== run_id
  ) {
    load_run_chain(project_id, task_id, run_id)
  }

  async function load_run_chain(
    req_project_id: string,
    req_task_id: string,
    req_run_id: string,
  ) {
    chain_loaded_for_run_id = req_run_id
    try {
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/runs/{run_id}/chain",
        {
          params: {
            path: {
              project_id: req_project_id,
              task_id: req_task_id,
              run_id: req_run_id,
            },
          },
        },
      )
      if (
        req_project_id !== project_id ||
        req_task_id !== task_id ||
        req_run_id !== run_id
      )
        return
      if (error) {
        throw error
      }
      run_chain = data?.chain ?? []
      chain_broken = !!data?.chain_broken
      run_has_children = !!data?.has_children
      chain_load_failed = false
    } catch (_) {
      if (
        req_project_id !== project_id ||
        req_task_id !== task_id ||
        req_run_id !== run_id
      )
        return
      run_chain = []
      chain_broken = false
      run_has_children = false
      chain_load_failed = true
    }
  }

  $: forkable_run_ids = compute_forkable_run_ids(run?.trace ?? [], run_chain)

  // Bound to the fork-mode MultiturnComposer so we can consult is_dirty()
  // / request_swap() when the user clicks fork on a different turn while
  // a composer is already open.
  let fork_composer: MultiturnComposer | null = null

  function on_fork(clicked_run_id: string, trace_index: number) {
    const target = fork_target_from_assistant_block(
      clicked_run_id,
      trace_index,
      run_chain,
    )
    if (!target) return
    const apply = () => {
      fork_target = target
    }
    // No active fork composer (or it's not the same turn we're already on):
    // if one is open, route through it so it can prompt on dirty edits.
    if (fork_target && fork_composer) {
      if (fork_target.trace_index === target.trace_index) {
        // Clicking fork on the already-active turn is a no-op.
        return
      }
      fork_composer.request_swap(apply)
      return
    }
    apply()
  }

  function cancel_fork() {
    fork_target = null
  }

  // Fork send begins: hold the fork point's truncation and raise the loader
  // before navigation, so the transcript stays truncated (with a pending-
  // response indicator) instead of flashing the full original conversation
  // while the forked run is created and loaded.
  function handle_fork_send_start() {
    fork_send_trace_index = fork_target?.trace_index ?? null
    awaiting_response = true
    apply_transcript_scroll()
  }

  async function handle_fork_success(new_run_id: string) {
    // Don't clear fork_target here: navigation resets it (see the run_id
    // reactive), and fork_send_trace_index / awaiting_response keep the view
    // truncated + loading until the forked run renders.
    await handle_send(new_run_id)
  }

  function new_chat() {
    goto(`/run`)
  }

  let buttons: ActionButton[] = []
  $: {
    buttons = []
    // Multiturn: start a fresh conversation (the /run page is the new-chat
    // entry point), mirroring the Kiln Assistant's "New Chat" button.
    if (is_multiturn) {
      buttons.push({
        label: "New Chat",
        icon: "/images/new_chat.svg",
        handler: new_chat,
      })
    }
    if (!deleted[run_id]) {
      buttons.push({
        icon: "/images/delete.svg",
        handler: () => delete_dialog?.show(),
        shortcut: isMacOS() ? "Backspace" : "Delete",
      })
    }
    if (list_page.length > 1) {
      const index = list_page.indexOf(run_id)
      if (index !== -1) {
        buttons.push({
          icon: "/images/previous.svg",
          handler: prev_run,
          shortcut: "ArrowLeft",
          disabled: index === 0,
        })
        buttons.push({
          icon: "/images/next.svg",
          handler: next_run,
          shortcut: "ArrowRight",
          disabled: index === list_page.length - 1,
        })
      }
    }
  }

  // Fancy logic to maintain the search string when navigating back to the dataset page (filters, sorting, etc.)
  const lastPageUrl = getContext<Writable<URL | undefined>>("lastPageUrl")
  function get_breadcrumbs() {
    if (!$lastPageUrl) {
      return []
    }

    try {
      const referrerPath = $lastPageUrl.pathname

      // Check if the referrer path is /dataset/{project_id}/{task_id}
      // since we only want to breadcrumb back to that page
      const expectedPath = `/dataset/${$page.params.project_id}/${$page.params.task_id}`

      if (referrerPath === expectedPath) {
        return [
          {
            label: "Dataset",
            // Include the full URL with search params to a
            href: $lastPageUrl.pathname + $lastPageUrl.search,
          },
        ]
      }
    } catch (error) {
      console.warn("Failed to parse referrer URL:", error)
    }

    return []
  }
</script>

<!-- Both layouts use the same capped width with the chat/input on the left and
     the Options sidebar on the right. -->
<div class="max-w-[1400px]">
  <AppPage
    title="Dataset Run"
    subtitle={run?.id ? `Run ID: ${run.id}` : undefined}
    action_buttons={buttons}
    breadcrumbs={get_breadcrumbs()}
    no_y_padding={is_multiturn}
  >
    {#if loading}
      <div class="w-full min-h-[50vh] flex justify-center items-center">
        <div class="loading loading-spinner loading-lg"></div>
      </div>
    {:else if deleted[run_id] === true}
      <div class="badge badge-error badge-lg p-4">Run Deleted</div>
    {:else if load_error}
      <div class="text-error">{load_error.getMessage()}</div>
    {:else if run && task}
      {#if task.turn_mode === "multiturn" && task.id}
        {@const multiturn_task_id = task.id}
        <!-- Chat-style layout: the whole page scrolls (no inner scroll
             regions). The conversation flows top-to-bottom with the composer
             sitting in normal flow directly below the last message; the
             Options sidebar sits at the top of the page in normal flow. -->
        <div data-testid="multiturn-layout">
          <div class="flex flex-col xl:flex-row gap-8 xl:gap-16">
            <div class="grow flex flex-col min-w-0">
              <div bind:this={transcript_scroll_el} class="min-w-0">
                <div class="flex w-full flex-col gap-6">
                  {#if run_has_children}
                    <div role="alert" data-testid="run-has-children-banner">
                      <Warning
                        warning_color="warning"
                        warning_icon="info"
                        warning_message="This run already has follow-up turns. Sending a new message here will start a new conversation branch — the existing continuations will be preserved."
                        outline={true}
                      />
                    </div>
                  {/if}
                  {#if chain_broken}
                    <div role="alert" data-testid="fork-chain-broken-banner">
                      <Warning
                        warning_color="warning"
                        warning_icon="exclaim"
                        warning_message="Some earlier turns can't be forked because their run data is missing. Forking is still available for later turns."
                        outline={true}
                      />
                    </div>
                  {/if}
                  {#if chain_load_failed}
                    <div role="alert" data-testid="fork-load-failed-banner">
                      <Warning
                        warning_color="warning"
                        warning_icon="exclaim"
                        warning_message="Couldn't load conversation history. Forking is unavailable."
                        outline={true}
                      />
                    </div>
                  {/if}
                  <!-- Intentionally NOT keyed on run.id: each turn loads a new
                       leaf run whose trace is a superset of the previous one,
                       so letting ChatTrace diff (append the new messages)
                       avoids tearing down and rebuilding the whole transcript
                       on every send — which caused a visible flash. -->
                  <ChatTrace
                    trace={display_trace}
                    {project_id}
                    {forkable_run_ids}
                    truncate_at_trace_index={fork_target?.trace_index ??
                      fork_send_trace_index}
                    {on_fork}
                    show_per_message_usage={task?.turn_mode === "multiturn"}
                  />
                  {#if awaiting_response}
                    <div data-testid="multiturn-pending-response">
                      <ChatLoading />
                    </div>
                  {/if}
                </div>
              </div>
              <div bind:this={composer_block_el} class="bg-base-100 pb-6 pt-2">
                <div class="flex w-full flex-col gap-2">
                  {#if fork_target}
                    <MultiturnComposer
                      bind:this={fork_composer}
                      mode="fork"
                      {project_id}
                      task_id={multiturn_task_id}
                      parent_task_run_id={fork_target.parent_run_id}
                      run_config_component={multiturn_run_config_component}
                      prefill_text={fork_target.prefill}
                      forked_turn_index={fork_target.turn_index}
                      {chain_broken}
                      on_success={handle_fork_success}
                      on_cancel={cancel_fork}
                      on_send_start={handle_fork_send_start}
                      on_send_settled={handle_send_settled}
                    />
                  {:else}
                    <MultiturnComposer
                      mode="append"
                      {project_id}
                      task_id={multiturn_task_id}
                      parent_task_run_id={run.id ?? null}
                      run_config_component={multiturn_run_config_component}
                      busy={awaiting_response}
                      on_success={handle_send}
                      on_send_start={handle_send_start}
                      on_send_settled={handle_send_settled}
                    />
                  {/if}
                  <!-- Raw data opens in a modal so it doesn't reflow the chat. -->
                  <div>
                    <button
                      class="text-xs link"
                      on:click={() => raw_data_dialog?.show()}
                    >
                      Show Raw Data
                    </button>
                  </div>
                </div>
              </div>
            </div>
            <div class="w-72 2xl:w-96 flex-none flex flex-col xl:pl-4 pb-8">
              <div class="text-xl font-bold mb-4">Options</div>
              <div class="flex flex-col gap-4">
                {#key run.id}
                  <SavedRunConfigurationsDropdown
                    {project_id}
                    current_task={task}
                    bind:selected_run_config_id={multiturn_selected_run_config_id}
                    bind:save_config_error={multiturn_save_config_error}
                    bind:set_default_error={multiturn_set_default_error}
                    save_new_run_config={handle_save_new_multiturn_run_config}
                    selected_model_specific_run_config_id={multiturn_selected_model_specific_run_config_id}
                  />
                  <RunConfigComponent
                    model={multiturn_initial_model(run)}
                    prompt_method={multiturn_initial_prompt(run)}
                    bind:this={multiturn_run_config_component}
                    {project_id}
                    current_task={task}
                    requires_structured_output={false}
                    bind:selected_run_config_id={multiturn_selected_run_config_id}
                    bind:set_default_error={multiturn_set_default_error}
                    bind:selected_model_specific_run_config_id={multiturn_selected_model_specific_run_config_id}
                    show_name_field={false}
                  />
                {/key}
              </div>
              <div class="mt-8">
                <PropertyList
                  properties={properties_for_list}
                  title="Properties"
                />
                <button
                  class="text-xs text-gray-500 underline text-left cursor-pointer bg-transparent border-none p-0 mt-4"
                  on:click={() => (see_all_properties = !see_all_properties)}
                >
                  {see_all_properties ? "See Less" : "See All"}
                </button>
              </div>
              <div class="mt-8">
                <RunSidebar
                  {project_id}
                  {task}
                  {run}
                  on_run_updated={(updated) => (run = updated)}
                />
              </div>
            </div>
          </div>
        </div>
      {:else}
        <div data-testid="single-turn-layout">
          <div class="flex flex-col xl:flex-row gap-8 xl:gap-16 mb-8">
            <div class="grow">
              <div class="text-xl font-bold mb-4">Input</div>
              <Output raw_output={run.input} />
            </div>
            <div class="w-72 2xl:w-96 flex-none flex flex-col">
              <PropertyList
                properties={properties_for_list}
                title="Properties"
              />
              <button
                class="text-xs text-gray-500 underline text-left cursor-pointer bg-transparent border-none p-0 mt-4"
                on:click={() => (see_all_properties = !see_all_properties)}
              >
                {see_all_properties ? "See Less" : "See All"}
              </button>
            </div>
          </div>
          <Run initial_run={run} {task} {project_id} />
        </div>
      {/if}
    {:else}
      <div class="text-gray-500 text-lg">Run not found</div>
    {/if}
  </AppPage>
</div>

<DeleteDialog
  name="Dataset Run"
  bind:this={delete_dialog}
  {delete_url}
  {after_delete}
/>

<Dialog bind:this={raw_data_dialog} title="Raw Data" width="wide">
  {#if run}
    <div class="text-sm max-h-[70vh] overflow-auto">
      <Output raw_output={JSON.stringify(run, null, 2)} />
    </div>
  {/if}
</Dialog>
