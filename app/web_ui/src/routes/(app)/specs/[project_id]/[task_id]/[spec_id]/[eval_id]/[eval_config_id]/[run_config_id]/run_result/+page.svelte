<script lang="ts">
  import AppPage from "../../../../../../../../app_page.svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import Warning from "$lib/ui/warning.svelte"
  import type {
    EvalRunResult,
    Eval,
    EvalConfig,
    EvalRunWithTrace,
    TaskRunConfig,
    Trace,
  } from "$lib/types"
  import ChatTrace from "$lib/ui/trace/chat_trace.svelte"
  import { isKilnAgentRunConfig } from "$lib/types"
  import { client } from "$lib/api_client"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { onMount, tick } from "svelte"
  import { page } from "$app/stores"
  import { string_to_json_key } from "$lib/utils/json_schema_editor/json_schema_templates"
  import { eval_split_filter_id } from "$lib/utils/eval_splits"
  import {
    eval_config_to_ui_name,
    eval_config_to_detailed_ui_name,
  } from "$lib/utils/formatters"
  import {
    getV2TypeFromEvalConfig,
    getV2EvalTypeMetadata,
  } from "$lib/utils/eval_types/registry"
  import {
    get_task_composite_id,
    model_info,
    load_model_info,
    model_name,
    provider_name_from_id,
    prompt_name_from_id,
    load_available_models,
  } from "$lib/stores"
  import {
    prompts_by_task_composite_id,
    load_task_prompts,
  } from "$lib/stores/prompts_store"
  import OutputTypeTablePreview from "$lib/components/output_type_table_preview.svelte"

  import { agentInfo } from "$lib/agent"
  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  $: eval_id = $page.params.eval_id!
  $: eval_config_id = $page.params.eval_config_id!
  $: run_config_id = $page.params.run_config_id!
  $: agentInfo.set({
    name: "Eval Run Result",
    description: `Eval run result for eval ID ${eval_id}, eval config ID ${eval_config_id}, run config ID ${run_config_id} in project ID ${project_id}, task ID ${task_id}. Shows individual eval run outputs and scores.`,
  })

  let results: EvalRunResult | null = null
  let results_error: KilnError | null = null
  let results_loading = true
  let peek_dialog: Dialog | null = null
  let thinking_dialog: Dialog | null = null
  let displayed_result: EvalRunWithTrace | null = null

  let trace_dialog: Dialog | null = null
  let displayed_trace: Trace | null = null

  // Keyed on the row object, so a reload of results drops the old entries.
  const parsed_traces = new WeakMap<EvalRunWithTrace, Trace | null>()

  // The roles ChatTrace understands: it draws user and assistant bubbles and
  // filters the rest. Any other role would silently speak as the assistant.
  const renderable_roles = new Set([
    "user",
    "assistant",
    "system",
    "developer",
    "tool",
  ])

  // ChatTrace only renders string message content, and only as user and
  // assistant bubbles (it filters system, developer and tool roles). So a trace
  // counts as renderable only if every element is a message object with a role
  // ChatTrace knows, every user/assistant content is a string or absent, every
  // tool call carries the id and function name ChatTrace dereferences, and at
  // least one user or assistant message exists. Anything else — a stored array
  // that is not a message list, content-part lists it would show as an empty
  // message, a role it would mislabel as the assistant, a tool call it would
  // throw on mid-render, a trace with no bubbles to show — falls back to the
  // flat view instead of throwing or rendering a lossy cell.
  function is_renderable_trace(raw: unknown): raw is Trace {
    if (!Array.isArray(raw)) {
      return false
    }
    let has_bubble = false
    for (const message of raw) {
      if (typeof message !== "object" || message === null) {
        return false
      }
      const role = (message as { role?: unknown }).role
      if (typeof role !== "string" || !renderable_roles.has(role)) {
        return false
      }
      if (role === "user" || role === "assistant") {
        const content = (message as { content?: unknown }).content
        if (
          content !== undefined &&
          content !== null &&
          typeof content !== "string"
        ) {
          return false
        }
        has_bubble = true
      }
      const tool_calls = (message as { tool_calls?: unknown }).tool_calls
      if (tool_calls && !are_renderable_tool_calls(tool_calls)) {
        return false
      }
    }
    return has_bubble
  }

  // ChatTrace reads `id` and `function.name` off every tool call without
  // guarding either, so a malformed one throws while the dialog is opening.
  // Tool results are safe: ChatTrace type-checks `tool_call_id` before use.
  function are_renderable_tool_calls(tool_calls: unknown): boolean {
    if (!Array.isArray(tool_calls)) {
      return false
    }
    for (const tool_call of tool_calls) {
      if (typeof tool_call !== "object" || tool_call === null) {
        return false
      }
      if (typeof (tool_call as { id?: unknown }).id !== "string") {
        return false
      }
      const fn = (tool_call as { function?: unknown }).function
      if (typeof fn !== "object" || fn === null) {
        return false
      }
      if (typeof (fn as { name?: unknown }).name !== "string") {
        return false
      }
    }
    return true
  }

  // The row's conversation, or null when it has none we can render as one. Null
  // sends the row down the untouched Input/Output path.
  function parsed_trace(result: EvalRunWithTrace): Trace | null {
    const cached = parsed_traces.get(result)
    if (cached !== undefined) {
      return cached
    }
    let trace: Trace | null = null
    try {
      const raw = result.task_run_trace
        ? JSON.parse(result.task_run_trace)
        : null
      trace = is_renderable_trace(raw) ? raw : null
    } catch (_) {
      trace = null
    }
    parsed_traces.set(result, trace)
    return trace
  }

  // The row's preview text. Falls back to the row's input when the conversation
  // has no plain-text user message; a row with neither renders only the link.
  function first_user_message(trace: Trace, fallback: string | null): string {
    for (const message of trace) {
      if (
        message.role === "user" &&
        "content" in message &&
        typeof message.content === "string" &&
        message.content
      ) {
        return message.content
      }
    }
    return fallback ?? ""
  }

  // Whether any row actually reads as a conversation. The column header and the
  // shared trace dialog both hinge on this, so a page whose rows all fall back
  // to the flat view renders exactly as it always has. `parsed_trace` memoizes
  // per row, so re-running this on a results change is cheap.
  $: has_conversation_row = !!results?.results.some((result) =>
    parsed_trace(result),
  )

  onMount(() => {
    peek_dialog?.show()
  })

  $: if (project_id && task_id && eval_id && eval_config_id && run_config_id) {
    load_all(project_id, task_id, eval_id, eval_config_id, run_config_id)
  }

  async function load_all(
    req_project_id: string,
    req_task_id: string,
    req_eval_id: string,
    req_eval_config_id: string,
    req_run_config_id: string,
  ) {
    await tick()
    load_model_info()
    load_task_prompts(req_project_id, req_task_id)
    load_available_models()
    get_evals(
      req_project_id,
      req_task_id,
      req_eval_id,
      req_eval_config_id,
      req_run_config_id,
    )
  }

  async function get_evals(
    req_project_id: string,
    req_task_id: string,
    req_eval_id: string,
    req_eval_config_id: string,
    req_run_config_id: string,
  ) {
    try {
      results = null
      results_error = null
      results_loading = true
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/eval_config/{eval_config_id}/run_config/{run_config_id}/results",
        {
          params: {
            path: {
              project_id: req_project_id,
              task_id: req_task_id,
              eval_id: req_eval_id,
              eval_config_id: req_eval_config_id,
              run_config_id: req_run_config_id,
            },
            // This page renders the eval's test split, which is what it has always
            // shown. Train and val are not surfaced in the UI (functional spec 4.4).
            query: { split: "test" },
          },
        },
      )
      if (
        req_project_id !== project_id ||
        req_task_id !== task_id ||
        req_eval_id !== eval_id ||
        req_eval_config_id !== eval_config_id ||
        req_run_config_id !== run_config_id
      )
        return
      if (error) {
        throw error
      }
      results = data
    } catch (error) {
      if (
        req_project_id !== project_id ||
        req_task_id !== task_id ||
        req_eval_id !== eval_id ||
        req_eval_config_id !== eval_config_id ||
        req_run_config_id !== run_config_id
      )
        return
      results_error = createKilnError(error)
    } finally {
      if (
        req_project_id === project_id &&
        req_task_id === task_id &&
        req_eval_id === eval_id &&
        req_eval_config_id === eval_config_id &&
        req_run_config_id === run_config_id
      ) {
        results_loading = false
      }
    }
  }

  function get_run_config_properties(
    run_config: TaskRunConfig | null,
    evaluator: Eval | null,
  ): Record<string, string> {
    if (!run_config || !evaluator) {
      return {}
    }
    const base: Record<string, string> = {
      "Run Configuration Name": run_config.name,
    }
    const test_filter_id = eval_split_filter_id(evaluator, "test")
    if (test_filter_id) {
      base["Task Inputs From Dataset"] = test_filter_id
    }
    if (!isKilnAgentRunConfig(run_config.run_config_properties)) {
      return {
        ...base,
        Type: "MCP Tool (No Agent)",
      }
    }

    return {
      ...base,
      Model: model_name(
        run_config.run_config_properties.model_name,
        $model_info,
      ),
      Provider: provider_name_from_id(
        run_config.run_config_properties.model_provider_name,
      ),
      Prompt: prompt_name_from_id(
        run_config.run_config_properties.prompt_id,
        $prompts_by_task_composite_id[
          get_task_composite_id(project_id, task_id)
        ] ?? null,
      ),
    }
  }

  function get_eval_properties(
    evaluator: Eval | null,
    eval_config: EvalConfig | null,
  ): Record<string, string> {
    if (!evaluator || !eval_config) {
      return {}
    }
    if (eval_config.config_type === "v2") {
      return {
        "Judge Name": eval_config.name,
        "Judge Type": eval_config_to_detailed_ui_name(eval_config),
      }
    }
    return {
      "Judge Name": eval_config.name,
      "Judge Algorithm": eval_config_to_ui_name(eval_config.config_type),
      "Judge Model": model_name(
        eval_config.model_name ?? undefined,
        $model_info,
      ),
      "Model Provider": provider_name_from_id(eval_config.model_provider ?? ""),
    }
  }

  $: is_v2_config = results?.eval_config?.config_type === "v2"

  $: v2_result_component = (() => {
    if (!results?.eval_config) return null
    const v2type = getV2TypeFromEvalConfig(results.eval_config)
    if (!v2type) return null
    return getV2EvalTypeMetadata(v2type).resultRendererComponent
  })()
</script>

<AppPage
  title="Eval Results"
  subtitle="Evaluating a task run configuration with a judge."
>
  {#if results_loading}
    <div class="w-full min-h-[50vh] flex justify-center items-center">
      <div class="loading loading-spinner loading-lg"></div>
    </div>
  {:else if results_error}
    <div
      class="w-full min-h-[50vh] flex flex-col justify-center items-center gap-2"
    >
      <div class="font-medium">Error Loading Eval Results</div>
      <div class="text-error text-sm">
        {results_error.getMessage() || "An unknown error occurred"}
      </div>
    </div>
  {:else if results && results.results.length === 0}
    <div
      class="w-full min-h-[50vh] flex flex-col justify-center items-center gap-2"
    >
      <div class="font-medium">Eval Results Empty</div>
      <div class="text-error text-sm">
        No results found for this run config.
      </div>
    </div>
  {:else if results}
    <div class="flex flex-col xl:flex-row gap-8 xl:gap-16 mb-8">
      <div class="grow basis-1/2">
        <div class="text-xl font-bold">Task Run Config</div>
        <div class="text-sm text-gray-500 mb-4">
          How the task outputs were generated.
        </div>
        <div
          class="grid grid-cols-[auto,1fr] gap-y-2 gap-x-4 text-sm 2xl:text-base"
        >
          {#each Object.entries(get_run_config_properties(results.run_config, results.eval)) as [prop_name, prop_value]}
            <div class="flex items-center">{prop_name}</div>
            <div class="flex items-center text-gray-500 overflow-x-hidden">
              {prop_value}
            </div>
          {/each}
        </div>
      </div>
      <div class="grow basis-1/2">
        <div class="text-xl font-bold">Judge</div>
        <div class="text-sm text-gray-500 mb-4">
          The judge used to evaluate the task outputs.
        </div>
        <div
          class="grid grid-cols-[auto,1fr] gap-y-2 gap-x-4 text-sm 2xl:text-base"
        >
          {#each Object.entries(get_eval_properties(results.eval, results.eval_config)) as [prop_name, prop_value]}
            <div class="flex items-center">{prop_name}</div>
            <div class="flex items-center text-gray-500 overflow-x-hidden">
              {prop_value}
            </div>
          {/each}
        </div>
      </div>
    </div>
    <div class="overflow-x-auto rounded-lg border">
      <table class="table">
        <thead>
          <tr>
            {#if has_conversation_row}
              <th>Interaction</th>
            {:else}
              <th>Input & Output</th>
            {/if}
            {#if !is_v2_config}
              <th>Thinking</th>
            {/if}
            {#if is_v2_config && v2_result_component}
              <th>Result</th>
            {:else}
              {#each results.eval.output_scores as score}
                <th class="text-center">
                  {score.name}
                  {#if score.type}
                    <OutputTypeTablePreview output_score_type={score.type} />
                  {/if}
                </th>
              {/each}
            {/if}
          </tr>
        </thead>
        <tbody>
          {#each results.results as result}
            <!-- A row reads as a conversation when it carries one we can
                 render, whatever the task's turn mode: a single-turn run
                 records a transcript too, tool calls and all. -->
            {@const row_trace = parsed_trace(result)}
            <tr>
              <td>
                {#if row_trace}
                  <!-- The first user message stands in for the row, and the
                       full conversation opens in a dialog. -->
                  <!-- Input, then reference answer, mirroring the flat view's
                       reading order below. -->
                  <div class="min-w-[280px] flex flex-col gap-3">
                    <div class="line-clamp-3 whitespace-pre-line">
                      {first_user_message(row_trace, result.input)}
                    </div>
                    {#if result.eval_run.reference_answer}
                      <div>
                        <div class="font-medium">Reference Answer:</div>
                        <div>
                          {result.eval_run.reference_answer}
                        </div>
                      </div>
                    {/if}
                    <div>
                      <!-- Same quiet-link affordance the builder's claim cards
                           use to open their trace modal. -->
                      <button
                        type="button"
                        class="text-xs text-primary hover:underline"
                        on:click={() => {
                          displayed_trace = row_trace
                          trace_dialog?.show()
                        }}
                      >
                        View Full Trace
                      </button>
                    </div>
                  </div>
                {:else}
                  <div class="font-medium">Input:</div>
                  <div>
                    <!-- Nullable: a skipped run whose dataset item is gone.
                         Svelte stringifies null to "null". -->
                    {result.input ?? ""}
                  </div>
                  {#if result.eval_run.reference_answer}
                    <div class="font-medium mt-4">Reference Answer:</div>
                    <div>
                      {result.eval_run.reference_answer}
                    </div>
                  {/if}
                  <div class="font-medium mt-4">Output:</div>
                  <div>
                    {#if result.eval_run.scored_run_id && result.output == null}
                      <!-- A resolved trace always has a string output, so null
                           here means the referenced trace could not be loaded.
                           Say so rather than rendering a blank that looks like
                           an empty output. The input above may still resolve
                           from the dataset item. -->
                      <div class="text-sm text-gray-500">
                        Trace unavailable. It may have been deleted or not
                        included in an import.
                      </div>
                    {:else}
                      {result.output ?? ""}
                    {/if}
                  </div>
                {/if}
              </td>
              {#if !is_v2_config}
                <td>
                  {#if result.eval_run.intermediate_outputs?.reasoning || result.eval_run.intermediate_outputs?.chain_of_thought}
                    <div class="max-w-[600px] min-w-[200px]">
                      <div class="max-h-[140px] overflow-y-hidden relative">
                        {result.eval_run.intermediate_outputs?.reasoning ||
                          result.eval_run.intermediate_outputs
                            ?.chain_of_thought ||
                          "N/A"}
                        <div class="absolute bottom-0 left-0 w-full">
                          <div
                            class="h-36 bg-gradient-to-t from-white to-transparent"
                          ></div>
                          <div
                            class="text-center bg-white font-medium font-sm text-gray-500"
                          >
                            <button
                              class="text-gray-500"
                              on:click={() => {
                                displayed_result = result
                                thinking_dialog?.show()
                              }}
                            >
                              See all
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  {:else}
                    N/A
                  {/if}
                </td>
              {/if}
              {#if is_v2_config && v2_result_component}
                <td>
                  <svelte:component
                    this={v2_result_component}
                    scores={result.eval_run.scores}
                    skipped_reason={result.eval_run.skipped_reason ?? null}
                    skipped_detail={result.eval_run.skipped_detail ?? null}
                    eval_config={results.eval_config}
                    intermediate_outputs={result.eval_run
                      .intermediate_outputs ?? null}
                  />
                </td>
              {:else}
                {#each results.eval.output_scores as score}
                  {@const score_value =
                    result.eval_run.scores[string_to_json_key(score.name)]}
                  <td class="text-center">
                    {score_value != null ? score_value.toFixed(2) : "N/A"}
                  </td>
                {/each}
              {/if}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</AppPage>

<Dialog
  title="Are you sure you want to peek?"
  bind:this={peek_dialog}
  blur_background={true}
  action_buttons={[
    {
      label: "Look Anyways",
      isError: true,
    },
    {
      label: "Go Back",
      isPrimary: true,
      action: () => {
        window.history.back()
        return true
      },
    },
  ]}
>
  <div class="font-light flex flex-col gap-4">
    <Warning
      warning_message="We strongly suggest you don't look at these results! Looking at these results can bias future iteration."
    />
    <div>
      Viewing these evaluation results may lead to data leakage - a fundamental
      issue in machine learning where information from your test set
      inadvertently influences your development process. When you examine
      specific examples, you're likely to optimize for those particular cases
      rather than developing solutions that generalize well to unseen data.
    </div>
    <div>
      Use our "Run" screen or fresh synthetic dataset generation if you want to
      explore what type of content a run configuration is generating.
    </div>
  </div>
</Dialog>

<Dialog bind:this={thinking_dialog} title="Thinking Output">
  <div class="font-light text-sm whitespace-pre-wrap">
    {displayed_result?.eval_run.intermediate_outputs?.reasoning ||
      displayed_result?.eval_run.intermediate_outputs?.chain_of_thought ||
      "N/A"}
  </div>
</Dialog>

{#if has_conversation_row}
  <!-- One shared dialog for every row: `displayed_trace` picks the row, and
       every open assigns it fresh. -->
  <Dialog bind:this={trace_dialog} title="Trace" width="extra_wide">
    {#if displayed_trace}
      <ChatTrace trace={displayed_trace} {project_id} />
    {/if}
  </Dialog>
{/if}
