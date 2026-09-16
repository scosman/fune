<script lang="ts">
  import AppPage from "../../../../../../app_page.svelte"
  import type { Eval, Task, Spec } from "$lib/types"
  import { client } from "$lib/api_client"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { onMount, tick } from "svelte"
  import { page } from "$app/stores"
  import FormElement, {
    type InlineAction,
  } from "$lib/utils/form_element.svelte"
  import type { EvalConfig, TaskRunConfig, EvalResultSummary } from "$lib/types"
  import { goto } from "$app/navigation"
  import {
    model_info,
    load_model_info,
    load_available_models,
    load_task,
    get_task_composite_id,
  } from "$lib/stores"
  import {
    load_task_run_configs,
    run_configs_by_task_composite_id,
  } from "$lib/stores/run_configs_store"
  import { set_current_eval_config } from "$lib/stores/evals_store"
  import Warning from "$lib/ui/warning.svelte"
  import { string_to_json_key } from "$lib/utils/json_schema_editor/json_schema_templates"
  import { formatEvalConfigName } from "$lib/utils/formatters"
  import CreateNewRunConfigDialog from "$lib/ui/run_config_component/create_new_run_config_dialog.svelte"
  import type { OptionGroup } from "$lib/ui/fancy_select_types"
  import Dialog from "$lib/ui/dialog.svelte"
  import type { ActionButton } from "$lib/types"
  import EvalConfigInstruction from "../eval_configs/eval_config_instruction.svelte"
  import Intro from "$lib/ui/intro.svelte"
  import RunConfigComparisonTable from "$lib/components/run_config_comparison_table.svelte"
  import { load_task_prompts } from "$lib/stores/prompts_store"
  import { multiTurnStoredScoreWarning } from "./multi_turn_warning"

  import { agentInfo } from "$lib/agent"
  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  $: spec_id = $page.params.spec_id!
  $: eval_id = $page.params.eval_id!
  let spec: Spec | null = null
  let spec_loading = true
  let spec_error: KilnError | null = null

  let task: Task | null = null
  let loading_task = true
  let task_error: KilnError | null = null

  let evaluator: Eval | null = null
  let eval_error: KilnError | null = null
  let eval_loading = true

  let eval_configs: EvalConfig[] | null = null
  let eval_configs_error: KilnError | null = null
  let eval_configs_loading = true
  let current_eval_config_id: string | null = null

  let run_configs_error: KilnError | null = null
  let run_configs_loading = true

  let score_summary: EvalResultSummary | null = null
  let score_summary_error: KilnError | null = null

  $: multi_turn_warning = multiTurnStoredScoreWarning(
    score_summary?.multi_turn_item_count,
  )

  // Note: not including score_summary_error, because it's not a critical error we should block the UI for
  $: loading =
    eval_loading ||
    eval_configs_loading ||
    run_configs_loading ||
    loading_task ||
    spec_loading
  $: error =
    eval_error ||
    eval_configs_error ||
    run_configs_error ||
    task_error ||
    spec_error

  $: current_task_run_configs =
    $run_configs_by_task_composite_id[
      get_task_composite_id(project_id, task_id)
    ] || null

  $: agentInfo.set({
    name: "Compare Run Configs",
    description: `Compare run configurations for a specific eval: eval ID ${eval_id}, spec ID ${spec_id} in project ID ${project_id}, task ID ${task_id}. Eval name: ${evaluator?.name ?? "[loading]"}. Table of eval results across configurations.`,
  })

  onMount(async () => {
    // Wait for page params to load
    await tick()
    // Wait for these to load, as they are needed for better labels. Usually already cached and instant.
    await Promise.all([
      load_model_info(),
      load_task_prompts(project_id, task_id),
      load_available_models(),
      get_task(),
      get_spec(),
    ])
    // Get the eval first (want it to set the current config id before the other two load)
    await get_eval()
    // These two can be parallel
    await Promise.all([get_eval_configs(), get_task_run_configs()])
    // This needs the selected eval config id, set from above requests
    get_score_summary()
  })

  let create_new_run_config_dialog: CreateNewRunConfigDialog | null = null

  async function get_spec() {
    if (spec_id === "legacy") {
      spec_loading = false
      return
    }
    try {
      spec_loading = true
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/specs/{spec_id}",
        {
          params: {
            path: {
              project_id,
              task_id,
              spec_id,
            },
          },
        },
      )
      if (error) {
        throw error
      }
      spec = data
    } catch (error) {
      spec_error = createKilnError(error)
    } finally {
      spec_loading = false
    }
  }

  async function get_task() {
    loading_task = true
    try {
      task = await load_task(project_id, task_id)
      if (!task) {
        throw new Error("Task not found")
      }
    } catch (err) {
      task_error = createKilnError(err)
    } finally {
      loading_task = false
    }
  }

  async function get_eval() {
    try {
      eval_loading = true
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}",
        {
          params: {
            path: {
              project_id,
              task_id,
              eval_id,
            },
          },
        },
      )
      if (error) {
        throw error
      }
      evaluator = data
      // Set the selected eval config: prefer query params, then eval's default, then first eval config (set below in load_eval_configs)
      current_eval_config_id =
        $page.url.searchParams.get("selected_eval_config") ||
        evaluator.current_config_id ||
        null
    } catch (error) {
      eval_error = createKilnError(error)
    } finally {
      eval_loading = false
    }
  }

  async function get_eval_configs() {
    try {
      eval_configs_loading = true
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/eval_configs",
        {
          params: {
            path: {
              project_id,
              task_id,
              eval_id,
            },
          },
        },
      )
      if (error) {
        throw error
      }
      eval_configs = data
      // Fallback to first eval config if no current eval config id is set from load_eval()
      if (
        !current_eval_config_id &&
        eval_configs.length > 0 &&
        eval_configs[0].id
      ) {
        current_eval_config_id = eval_configs[0].id
      }
    } catch (error) {
      eval_configs_error = createKilnError(error)
    } finally {
      eval_configs_loading = false
    }
  }

  async function get_task_run_configs() {
    run_configs_loading = true
    try {
      await load_task_run_configs(project_id, task_id)
    } catch (err) {
      run_configs_error = createKilnError(err)
    } finally {
      run_configs_loading = false
    }
  }

  async function get_score_summary() {
    score_summary = null
    if (!current_eval_config_id) {
      score_summary_error = new KilnError("No judge selected", null)
      return
    }
    try {
      score_summary_error = null
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/eval_config/{eval_config_id}/score_summary",
        {
          params: {
            path: {
              project_id,
              task_id,
              eval_id,
              eval_config_id: current_eval_config_id,
            },
          },
        },
      )
      if (error) {
        throw error
      }
      score_summary = data
    } catch (error) {
      score_summary_error = createKilnError(error)
    }
  }

  // Watches the current eval config id, performing actions based on it
  $: watch_selected_eval_config(current_eval_config_id)
  function watch_selected_eval_config(selected_id: string | null) {
    if (selected_id === "__create_new_judge__") {
      // if it's the "create new judge" special value, navigate to the create eval config page
      const params = new URLSearchParams()
      params.set("next_page", "compare_run_configs")
      goto(
        `/specs/${project_id}/${task_id}/${spec_id}/${eval_id}/create_eval_config?${params.toString()}`,
      )
      return
    }
    // If the selected id is not null, then get the score summary
    score_summary = null
    if (selected_id) {
      get_score_summary()
    }
  }

  $: current_eval_config = eval_configs?.find(
    (config) => config.id === current_eval_config_id,
  )

  // Sort task run configs - default first, then by last output score
  $: sorted_task_run_configs = current_task_run_configs
    ? sortTaskRunConfigs(current_task_run_configs, evaluator, score_summary)
    : []

  function sortTaskRunConfigs(
    configs: TaskRunConfig[] | null,
    evaluator: Eval | null,
    score_summary: EvalResultSummary | null,
  ): TaskRunConfig[] {
    if (!configs || !configs.length) return []

    return [...configs].sort((a, b) => {
      // Default run config always comes first
      if (a.id === task?.default_run_config_id) return -1
      if (b.id === task?.default_run_config_id) return 1

      // If we have evaluator and score summary, sort by the last output score
      if (evaluator?.output_scores?.length && score_summary?.results) {
        const lastScoreKey = string_to_json_key(
          evaluator.output_scores[evaluator.output_scores.length - 1].name,
        )

        const scoreA =
          score_summary.results["" + a.id]?.[lastScoreKey]?.mean_score
        const scoreB =
          score_summary.results["" + b.id]?.[lastScoreKey]?.mean_score

        // If both have scores, sort by score (higher first)
        if (
          scoreA !== null &&
          scoreA !== undefined &&
          scoreB !== null &&
          scoreB !== undefined
        ) {
          return scoreB - scoreA
        }

        // If only one has a score, it comes first
        if (scoreA !== null && scoreA !== undefined) return -1
        if (scoreB !== null && scoreB !== undefined) return 1
      }

      // Fallback to sort by name
      return a.name.localeCompare(b.name)
    })
  }

  let judge_instructions_dialog: Dialog | null = null

  function get_eval_config_select_options(
    configs: EvalConfig[] | null,
  ): OptionGroup[] {
    let options: OptionGroup[] = []

    options.push({
      label: "",
      options: [
        {
          value: "__create_new_judge__",
          label: "New Judge",
          badge: "＋",
          badge_color: "primary",
        },
      ],
    })

    if (configs && configs.length > 0) {
      const sortedConfigs = [...configs].sort((a, b) => {
        if (a.id === evaluator?.current_config_id) return -1
        if (b.id === evaluator?.current_config_id) return 1
        return 0
      })

      options.push({
        label: "Judges",
        options: sortedConfigs.map((config) => ({
          value: config.id,
          label: formatEvalConfigName(config, $model_info),
          badge:
            config.id === evaluator?.current_config_id ? "Default" : undefined,
        })),
      })
    }

    return options
  }

  let eval_state:
    | "not_started"
    | "running"
    | "complete"
    | "complete_with_errors" = "not_started"

  $: has_default_eval_config = evaluator && evaluator.current_config_id

  function action_buttons(evaluator: Eval | null): ActionButton[] {
    if (!evaluator) {
      return []
    }
    if (evaluator.template !== "rag") {
      return [
        {
          label: "Compare Judges",
          href: `/specs/${project_id}/${task_id}/${spec_id}/${eval_id}/eval_configs`,
          primary: !has_default_eval_config,
        },
      ]
    } else {
      return [
        {
          label: "Add Judge",
          href: `/specs/${project_id}/${task_id}/${spec_id}/${eval_id}/create_eval_config?next_page=compare_run_configs`,
          primary: false,
        },
      ]
    }
  }

  let inline_judge_action: InlineAction | null = null
  let set_current_eval_config_error: KilnError | null = null

  async function handle_set_current_eval_config() {
    if (!current_eval_config_id) {
      return
    }
    try {
      set_current_eval_config_error = null
      evaluator = await set_current_eval_config(
        project_id,
        task_id,
        eval_id,
        current_eval_config_id,
      )
    } catch (error) {
      set_current_eval_config_error = createKilnError(error)
    }
  }

  $: void (current_eval_config_id,
  evaluator?.current_config_id,
  update_inline_judge_action())

  function update_inline_judge_action() {
    if (
      evaluator?.template === "rag" &&
      current_eval_config_id !== evaluator?.current_config_id
    ) {
      inline_judge_action = {
        handler: handle_set_current_eval_config,
        label: "Set as default",
      }
    } else {
      inline_judge_action = null
    }
  }
</script>

<div class="max-w-[1400px]">
  <AppPage
    title="Compare Run Configurations"
    subtitle="Find the best configuration for running your task."
    sub_subtitle="Read the Docs"
    sub_subtitle_link="https://docs.kiln.tech/docs/evaluations#finding-the-ideal-run-method"
    breadcrumbs={spec_id === "legacy"
      ? [
          {
            label: "Evals",
            href: `/specs/${project_id}/${task_id}`,
          },
          {
            label: "Eval",
            href: `/specs/${project_id}/${task_id}/${spec_id}/${eval_id}`,
          },
        ]
      : [
          {
            label: "Evals",
            href: `/specs/${project_id}/${task_id}`,
          },
          {
            label: spec?.name || "Eval",
            href: `/specs/${project_id}/${task_id}/${spec_id}`,
          },
          {
            label: "Eval",
            href: `/specs/${project_id}/${task_id}/${spec_id}/${eval_id}`,
          },
        ]}
    action_buttons={action_buttons(evaluator)}
  >
    <!-- Stored multi-turn conversations can't be regenerated per run config,
      so the eval runner judges their saved messages and those rows score
      identically across configs. Gate on the actual item set: the score
      summary reports how many such items the eval set contains. -->
    {#if multi_turn_warning}
      <Warning
        warning_color="warning"
        warning_icon="info"
        warning_message={multi_turn_warning}
      />
    {/if}
    {#if loading}
      <div class="w-full min-h-[50vh] flex justify-center items-center">
        <div class="loading loading-spinner loading-lg"></div>
      </div>
    {:else if error}
      <div
        class="w-full min-h-[50vh] flex flex-col justify-center items-center gap-2"
      >
        <div class="font-medium">Error Loading Evaluator</div>
        <div class="text-error text-sm">
          {error.getMessage() || "An unknown error occurred"}
        </div>
      </div>
    {:else if evaluator}
      {#if evaluator.template === "rag" && eval_configs !== null && eval_configs.length === 0}
        <div class="max-w-[300px] mx-auto flex flex-col gap-2 mt-[10vh]">
          <Intro
            title="Create a Judge to Get Started"
            description_paragraphs={[
              "Create a judge to use when comparing run configurations.",
              "A judge specifies how an eval is run (algorithm, model, instructions, etc).",
            ]}
            action_buttons={[
              {
                label: "Create Judge",
                href: `/specs/${project_id}/${task_id}/${spec_id}/${eval_id}/create_eval_config?next_page=compare_run_configs&save_as_default=true`,
                is_primary: true,
              },
            ]}
          />
        </div>
      {:else}
        <div class="flex flex-col xl:flex-row gap-8 xl:gap-16 mb-8">
          <div class="grow flex flex-col gap-4">
            <div>
              <div class="text-xl font-bold">Judge</div>
              <div class="text-sm text-gray-500 mb-2">
                Select the judge to use when comparing run configurations.
              </div>
              <FormElement
                id="eval_config_select"
                inputType="fancy_select"
                bind:value={current_eval_config_id}
                fancy_select_options={get_eval_config_select_options(
                  eval_configs,
                )}
                inline_action={inline_judge_action}
              />
              {#if set_current_eval_config_error}
                <div class="text-error text-sm mt-2">
                  {set_current_eval_config_error.getMessage() ||
                    "An unknown error occurred"}
                </div>
              {/if}
              {#if !has_default_eval_config && evaluator?.template !== "rag"}
                <div class="mt-2">
                  <Warning
                    warning_message="No default judge selected. We recommend using 'Compare Judges' and selecting the best as the default."
                    warning_color="warning"
                    inline={true}
                  />
                </div>
              {:else if has_default_eval_config && evaluator.current_config_id != current_eval_config_id}
                <div class="mt-2">
                  <Warning
                    warning_message={evaluator.template === "rag"
                      ? "The currently selected judge is not the default. You can change the default with 'Set as default' above."
                      : "The currently selected judge is not the default. You can change the default in 'Compare Judges'."}
                    warning_color="warning"
                    inline={true}
                  />
                </div>
              {/if}
            </div>
            {#if current_eval_config_id && evaluator}
              {#if evaluator.template === "rag"}
                <button
                  class="flex link text-gray-500 text-sm 2xl:text-base"
                  on:click={() => {
                    judge_instructions_dialog?.show()
                  }}
                >
                  Judge Instructions
                </button>
              {:else}
                <div class="flex gap-x-4 text-sm 2xl:text-base items-center">
                  <span>Judge Quality</span>
                  <a
                    href={`/specs/${project_id}/${task_id}/${spec_id}/${eval_id}/eval_configs`}
                    class="link text-gray-500"
                  >
                    Compare and optimize
                  </a>
                </div>
              {/if}
            {/if}
          </div>
        </div>
        <div class="mt-16">
          <RunConfigComparisonTable
            {project_id}
            {task_id}
            {spec_id}
            {eval_id}
            {evaluator}
            {task}
            {sorted_task_run_configs}
            {score_summary}
            {score_summary_error}
            {current_eval_config_id}
            bind:eval_state
            interactive={true}
            current_eval_config_name={current_eval_config?.name ||
              "select above"}
            on_add_run_config={() => {
              create_new_run_config_dialog?.show()
            }}
            on_eval_complete={() => {
              get_score_summary()
            }}
          />
        </div>
      {/if}
    {/if}
  </AppPage>
</div>

<CreateNewRunConfigDialog
  bind:this={create_new_run_config_dialog}
  subtitle="Your evaluator can compare multiple run configurations to find which one produces
    the highest scores on your eval's test dataset."
  {project_id}
  {task}
  new_run_config_created={async () => {
    await load_task_prompts(project_id, task_id, true)
  }}
/>

<Dialog
  bind:this={judge_instructions_dialog}
  title="Instructions for Judge '{current_eval_config?.name}'"
  action_buttons={[
    {
      label: "Close",
      isCancel: true,
    },
  ]}
>
  <EvalConfigInstruction bind:eval_config={current_eval_config} />
</Dialog>
