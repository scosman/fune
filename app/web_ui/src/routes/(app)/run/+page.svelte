<script lang="ts">
  import AppPage from "../app_page.svelte"
  import { agentInfo } from "$lib/agent"
  import { current_task, current_project, ui_state } from "$lib/stores"
  import { createKilnError } from "$lib/utils/error_handlers"
  import FormContainer from "$lib/utils/form_container.svelte"
  import { KilnError } from "$lib/utils/error_handlers"
  import Run from "./run.svelte"
  import { client } from "$lib/api_client"
  import type { TaskRun, TaskRunConfig, TraceMessage } from "$lib/types"
  import RunInputForm from "./run_input_form.svelte"
  import posthog from "posthog-js"
  import { getContext, onDestroy, onMount, tick } from "svelte"
  import type { Writable } from "svelte/store"
  import RunConfigComponent from "$lib/ui/run_config_component/run_config_component.svelte"
  import SavedRunConfigurationsDropdown from "$lib/ui/run_config_component/saved_run_configs_dropdown.svelte"
  import MultiturnComposer from "$lib/ui/conversation/multiturn_composer.svelte"
  import ChatTrace from "$lib/ui/trace/chat_trace.svelte"
  import ChatLoading from "$lib/ui/conversation/chat_thinking_loading.svelte"
  import ChatIcon from "$lib/ui/icons/chat_icon.svelte"
  import { isMcpRunConfig } from "$lib/types"
  import { page } from "$app/stores"
  import { goto } from "$app/navigation"
  import ErrorWithTraceComponent from "$lib/ui/error_with_trace.svelte"
  import type { ErrorWithTrace } from "$lib/types"
  import { is_error_with_trace } from "./error_with_trace_detection"

  $: agentInfo.set({
    name: "Run",
    description: `Run a task with a selected model and configuration.${$current_task ? ` Current task: ${$current_task.name}.` : ""}`,
  })

  let run_error: KilnError | null = null
  let error_with_trace: ErrorWithTrace | null = null
  let submitting = false
  let run_complete = false

  let input_form: RunInputForm
  let output_section: HTMLElement | null = null
  let model_name: string = ""
  let provider: string = ""
  let selected_run_config_id: string | null = null
  // Some models have a model-specific suggested run config, such as fine-tuned models. If a model like that is selected, this will be set to the run config ID.
  let selected_model_specific_run_config_id: string | null = null
  let model: string = $ui_state.selected_model || ""

  let run_config_component: RunConfigComponent
  let save_config_error: KilnError | null = null
  let set_default_error: KilnError | null = null

  let response: TaskRun | null = null
  $: run_focus = !response && !error_with_trace

  $: project_id = $current_project?.id ?? ""
  $: task_id = $current_task?.id ?? ""
  $: input_schema = $current_task?.input_json_schema
  $: pending_tool_id = $page.url.searchParams.get("tool_id")
  $: pending_run_config_id = $page.url.searchParams.get("run_config_id")

  $: subtitle = $current_task ? "Task: " + $current_task.name : ""

  // For multi-turn tasks the /run page mirrors the in-conversation chat UI:
  // an (empty) transcript area with the composer pinned at the bottom. The
  // first message starts a new root conversation, then we redirect to the
  // dataset run page where the conversation continues.
  $: is_multiturn = $current_task?.turn_mode === "multiturn"
  const noLayoutBottomPadding = getContext<Writable<boolean> | undefined>(
    "noLayoutBottomPadding",
  )
  $: noLayoutBottomPadding?.set(!!is_multiturn)
  onDestroy(() => noLayoutBottomPadding?.set(false))

  // Optimistic state for the first turn: show the just-sent message + a
  // loading indicator while the root run is created, then redirect.
  let mt_optimistic_message: string | null = null
  let mt_awaiting_response = false
  $: mt_display_trace = (
    mt_optimistic_message === null
      ? []
      : [{ role: "user", content: mt_optimistic_message }]
  ) as TraceMessage[]

  function handle_mt_send_start(text: string) {
    mt_optimistic_message = text
    mt_awaiting_response = true
  }
  function handle_mt_send_settled(ok: boolean) {
    // On error the redirect never happens, so reset; on success we navigate
    // away and this component unmounts.
    if (!ok) {
      mt_optimistic_message = null
      mt_awaiting_response = false
    }
  }
  async function handle_first_turn_success(
    new_run_id: string,
    created_run?: TaskRun,
  ) {
    // Hand the freshly-created run to the dataset run page via navigation
    // state so it can render the conversation immediately instead of flashing
    // the full-page loading spinner while it re-fetches the run. noScroll keeps
    // the page from jumping to the top before the transcript pins to the
    // latest turn.
    await goto(`/dataset/${project_id}/${task_id}/${new_run_id}/run`, {
      state: created_run ? { created_run } : {},
      noScroll: true,
    })
  }

  onMount(() => {
    const model_override = $page.url.searchParams.get("model")
    if (model_override) {
      model = model_override
      selected_run_config_id = "custom"
    }
  })

  async function run_task() {
    try {
      submitting = true
      run_error = null
      error_with_trace = null
      response = null
      run_complete = false
      if (!run_config_component) {
        throw new Error(
          "Task configuration is still loading. Please wait a moment and try again.",
        )
      }
      run_config_component.clear_run_options_errors()
      run_config_component.clear_model_dropdown_error()
      const run_config_properties =
        run_config_component.run_options_as_run_config_properties()
      const is_mcp_run = isMcpRunConfig(run_config_properties)
      // mcp run configs don't need a model
      if (!is_mcp_run && !run_config_component.get_selected_model()) {
        run_config_component.set_model_dropdown_error("Required")
        throw new Error("You must select a model before running")
      }
      // If the user picked a saved run config (rather than configuring inline),
      // forward its ID so it gets recorded on the resulting TaskRun.
      const task_run_config_id =
        selected_run_config_id && selected_run_config_id !== "custom"
          ? selected_run_config_id
          : null
      const {
        data, // only present if 2XX response
        error: fetch_error, // only present if 4XX or 5XX response
      } = await client.POST("/api/projects/{project_id}/tasks/{task_id}/run", {
        params: {
          path: {
            project_id: project_id,
            task_id: task_id,
          },
        },
        body: {
          run_config_properties: run_config_properties,
          plaintext_input: input_form.get_plaintext_input_data(),
          // @ts-expect-error - let the server verify the type. TS isn't ideal for runtime type checking.
          structured_input: input_form.get_structured_input_data(),
          tags: ["manual_run"],
          task_run_config_id: task_run_config_id,
        },
      })
      if (fetch_error) {
        // openapi-fetch already parses the error body into fetch_error, so we
        // can inspect it directly to decide whether this is the new structured
        // ErrorWithTrace shape (HTTP 500 from an adapter failure) or a plain
        // HTTPException. Calling response.json() here would throw because the
        // body stream has already been consumed during parsing.
        if (is_error_with_trace(fetch_error)) {
          error_with_trace = fetch_error
          return
        }
        throw fetch_error
      }
      if (is_mcp_run) {
        posthog.capture("run_mcp_tool_directly")
      } else {
        const tools = run_config_component.get_tools()
        posthog.capture("run_task", {
          model_name: model_name,
          provider: provider,
          prompt_method: run_config_component.get_prompt_method(),
          tool_count: tools.length,
          search_tools: tools.filter((tool) =>
            tool.startsWith("kiln_tool::rag::"),
          ).length,
          mcp_tools: tools.filter((tool) => tool.startsWith("mcp::")).length,
          kiln_task_tools: tools.filter((tool) =>
            tool.startsWith("kiln_task::"),
          ).length,
        })
      }
      response = data
    } catch (e) {
      run_error = createKilnError(e)
    } finally {
      submitting = false
      await tick() // ensure {#if !submitting && response} has rendered
      if (response) scroll_to_output_if_needed()
      run_complete = true
    }
  }

  function clear_all() {
    input_form.clear_input()
    response = null
    error_with_trace = null
    run_complete = false
  }

  // Check if the Output section headers are visible in the viewport
  // We only care about the top portion being visible (headers + some buffer)
  function is_element_partially_visible(element: HTMLElement): boolean {
    const rect = element.getBoundingClientRect()
    const viewportHeight =
      window.innerHeight || document.documentElement.clientHeight

    // Check if the top of the element is visible and there's enough buffer
    // We want to see the headers (roughly 100px from top) plus some buffer
    // If the element is smaller than 100px, just check if it's fully visible
    const bufferSize = Math.min(100, rect.height)
    return rect.top >= 0 && rect.top <= viewportHeight - bufferSize
  }

  // Smooth scroll to output section if it's not visible
  function scroll_to_output_if_needed() {
    if (output_section && !is_element_partially_visible(output_section)) {
      // Calculate the target scroll position to show just the headers + buffer
      const rect = output_section.getBoundingClientRect()
      const currentScrollTop =
        window.pageYOffset || document.documentElement.scrollTop
      const viewportHeight =
        window.innerHeight || document.documentElement.clientHeight

      // Position the Output section so that 200px of it is visible from the top
      // This shows the headers and some buffer, but not the entire section
      // If the element is smaller than 200px, show the entire element
      const visibleHeight = Math.min(200, rect.height)
      const targetScrollTop =
        currentScrollTop + rect.top - (viewportHeight - visibleHeight)

      window.scrollTo({
        top: targetScrollTop,
        behavior: "smooth",
      })
    }
  }

  async function handle_save_new_run_config(): Promise<TaskRunConfig> {
    if (!run_config_component) {
      throw new Error("Run configuration component is not loaded")
    }
    return await run_config_component.save_new_run_config()
  }

  function handle_input_change() {
    if (response) {
      response = null
    }
    if (error_with_trace) {
      error_with_trace = null
    }
  }
</script>

<!-- Both layouts use the same capped width with the chat/input on the left and
     the Options sidebar on the right. -->
<div class="max-w-[1400px]">
  <AppPage
    title="Run"
    bind:subtitle
    no_y_padding={is_multiturn}
    action_buttons={is_multiturn
      ? []
      : [{ label: "Clear All", handler: clear_all }]}
  >
    {#if is_multiturn && $current_task}
      <!-- Multi-turn: a chat-style new conversation that mirrors the in-run
           multiturn page. The first message starts a root run, then we
           redirect to the dataset run page to continue the conversation. -->
      <div class="flex flex-col xl:flex-row gap-8 xl:gap-16">
        <!-- The whole page scrolls; the composer is pinned to the bottom of the
             viewport via position:sticky. The min-height keeps the composer at
             the bottom of the screen for the (empty) new-conversation state. -->
        <div class="grow flex flex-col min-w-0 xl:min-h-[calc(100vh-11rem)]">
          <div class="min-w-0 xl:flex-1 xl:flex xl:flex-col">
            <div class="flex w-full flex-col gap-6 xl:flex-1">
              {#if mt_awaiting_response}
                <ChatTrace
                  trace={mt_display_trace}
                  {project_id}
                  show_per_message_usage={false}
                />
                <div data-testid="run-multiturn-pending">
                  <ChatLoading />
                </div>
              {:else}
                <div class="flex flex-1 items-center justify-center px-4 py-8">
                  <div
                    class="flex max-w-[340px] flex-col items-center gap-3 rounded-xl px-6 py-8 text-center"
                  >
                    <div class="h-10 w-10 text-gray-400"><ChatIcon /></div>
                    <div class="text-base font-medium">
                      Start a conversation
                    </div>
                    <div class="text-sm font-light text-gray-500">
                      Send a message below to begin a multi-turn conversation
                      for your task.
                    </div>
                  </div>
                </div>
              {/if}
            </div>
          </div>
          <div class="sticky bottom-0 z-10 mt-6 bg-base-100 pb-6 pt-4">
            <div class="w-full">
              <MultiturnComposer
                mode="append"
                {project_id}
                {task_id}
                parent_task_run_id={null}
                allow_root_turn={true}
                {run_config_component}
                busy={mt_awaiting_response}
                on_success={handle_first_turn_success}
                on_send_start={handle_mt_send_start}
                on_send_settled={handle_mt_send_settled}
              />
            </div>
          </div>
        </div>
        <div class="w-72 2xl:w-96 flex-none flex flex-col gap-4">
          <div class="text-xl font-bold">Options</div>
          <SavedRunConfigurationsDropdown
            {project_id}
            current_task={$current_task}
            bind:selected_run_config_id
            bind:save_config_error
            bind:set_default_error
            save_new_run_config={handle_save_new_run_config}
            {selected_model_specific_run_config_id}
          />
          <RunConfigComponent
            {model}
            bind:this={run_config_component}
            {project_id}
            current_task={$current_task}
            requires_structured_output={false}
            bind:selected_run_config_id
            bind:set_default_error
            bind:selected_model_specific_run_config_id
            {pending_tool_id}
            {pending_run_config_id}
            show_name_field={false}
          />
        </div>
      </div>
    {:else}
      <div class="flex flex-col xl:flex-row gap-8 xl:gap-16">
        <div class="grow">
          <div class="text-xl font-bold mb-4">Input</div>
          <FormContainer
            submit_label="Run"
            on:submit={run_task}
            bind:error={run_error}
            bind:submitting
            bind:primary={run_focus}
            bind:keyboard_submit={run_focus}
          >
            <RunInputForm
              bind:input_schema
              bind:this={input_form}
              onInputChange={handle_input_change}
            />
          </FormContainer>
        </div>
        {#if $current_task}
          <div class="w-72 2xl:w-96 flex-none flex flex-col gap-4">
            <div class="text-xl font-bold">Options</div>
            <SavedRunConfigurationsDropdown
              {project_id}
              current_task={$current_task}
              bind:selected_run_config_id
              bind:save_config_error
              bind:set_default_error
              save_new_run_config={handle_save_new_run_config}
              {selected_model_specific_run_config_id}
            />
            <RunConfigComponent
              {model}
              bind:this={run_config_component}
              {project_id}
              current_task={$current_task}
              requires_structured_output={!!$current_task.output_json_schema}
              bind:selected_run_config_id
              bind:set_default_error
              bind:selected_model_specific_run_config_id
              {pending_tool_id}
              {pending_run_config_id}
              show_name_field={false}
            />
          </div>
        {/if}
      </div>
      {#if $current_task && !submitting && project_id}
        {#if error_with_trace}
          <div class="mt-8 xl:mt-12">
            <ErrorWithTraceComponent
              error={error_with_trace}
              error_title="Run Failed"
            />
          </div>
        {:else if response != null}
          <div
            class="mt-8 xl:mt-12"
            bind:this={output_section}
            id="output-section"
          >
            <Run
              initial_run={response}
              task={$current_task}
              {project_id}
              bind:model_name
              bind:provider
              bind:run_complete
              focus_repair_on_appear={true}
            />
          </div>
        {/if}
      {/if}
    {/if}
  </AppPage>
</div>
