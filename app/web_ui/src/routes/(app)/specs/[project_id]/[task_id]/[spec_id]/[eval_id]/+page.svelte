<script lang="ts">
  import AppPage from "../../../../../app_page.svelte"
  import type { Eval, Spec } from "$lib/types"
  import { client } from "$lib/api_client"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { tick } from "svelte"
  import { page } from "$app/stores"
  import type { EvalProgress } from "$lib/types"
  import InfoTooltip from "$lib/ui/info_tooltip.svelte"
  import {
    eval_config_to_ui_name,
    eval_config_to_detailed_ui_name,
  } from "$lib/utils/formatters"
  import {
    model_info,
    load_model_info,
    model_name,
    load_available_models,
  } from "$lib/stores"
  import type { ProviderModels } from "$lib/types"
  import { goto } from "$app/navigation"
  import { progress_ui_state } from "$lib/stores/progress_ui_store"
  import PropertyList from "$lib/ui/property_list.svelte"
  import type { UiProperty } from "$lib/ui/property_list"
  import { getDetailedModelNameFromParts } from "$lib/utils/run_config_formatters"
  import {
    eval_type_display,
    judge_type_from_config,
  } from "$lib/utils/eval_types/eval_type_display"
  import EditDialog from "$lib/ui/edit_dialog.svelte"
  import { tagFromFilterId, linkFromFilterId } from "../../spec_utils"
  import { available_tools, load_available_tools } from "$lib/stores"
  import {
    eval_split,
    eval_split_filter_id,
    task_run_split_filter_id,
  } from "$lib/utils/eval_splits"
  import { build_eval_generation_splits_param } from "$lib/utils/eval_generation_splits"

  import { agentInfo } from "$lib/agent"
  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  $: spec_id = $page.params.spec_id!
  $: eval_id = $page.params.eval_id!
  $: agentInfo.set({
    name: "Eval Detail",
    description: `Eval detail for eval ID ${eval_id}, spec ID ${spec_id} in project ID ${project_id}, task ID ${task_id}. Eval name: ${evaluator?.name ?? "[loading]"}. Shows eval configurations and run results.`,
  })
  $: is_legacy_eval = spec_id === "legacy"

  let spec: Spec | null = null
  let spec_error: KilnError | null = null
  let spec_loading = true

  let evaluator: Eval | null = null
  let eval_error: KilnError | null = null
  let eval_loading = true

  let eval_progress_loading = true
  let eval_progress: EvalProgress | null = null
  let eval_progress_error: KilnError | null = null

  $: loading = spec_loading || eval_loading || eval_progress_loading
  $: error = spec_error || eval_error || eval_progress_error

  $: if (project_id && task_id && spec_id && eval_id) {
    load_all(project_id, task_id, spec_id, eval_id)
  }

  async function load_all(
    req_project_id: string,
    req_task_id: string,
    req_spec_id: string,
    req_eval_id: string,
  ) {
    await tick()
    load_model_info()
    load_available_models()
    // Used to map a tool_call_check judge's tool names to a tool id when
    // launching the add-data flow.
    load_available_tools(req_project_id)
    await Promise.all([
      get_spec(req_project_id, req_task_id, req_spec_id),
      get_eval(req_project_id, req_task_id, req_eval_id),
      get_eval_progress(req_project_id, req_task_id, req_eval_id),
    ])
  }

  async function get_spec(
    req_project_id: string,
    req_task_id: string,
    req_spec_id: string,
  ) {
    if (req_spec_id === "legacy") {
      if (
        req_project_id === project_id &&
        req_task_id === task_id &&
        req_spec_id === spec_id
      ) {
        spec_loading = false
      }
      return
    }
    try {
      spec_loading = true
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/specs/{spec_id}",
        {
          params: {
            path: {
              project_id: req_project_id,
              task_id: req_task_id,
              spec_id: req_spec_id,
            },
          },
        },
      )
      if (
        req_project_id !== project_id ||
        req_task_id !== task_id ||
        req_spec_id !== spec_id
      )
        return
      if (error) {
        throw error
      }
      spec = data
    } catch (error) {
      if (
        req_project_id !== project_id ||
        req_task_id !== task_id ||
        req_spec_id !== spec_id
      )
        return
      spec_error = createKilnError(error)
    } finally {
      if (
        req_project_id === project_id &&
        req_task_id === task_id &&
        req_spec_id === spec_id
      ) {
        spec_loading = false
      }
    }
  }

  async function get_eval(
    req_project_id: string,
    req_task_id: string,
    req_eval_id: string,
  ) {
    try {
      eval_loading = true
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}",
        {
          params: {
            path: {
              project_id: req_project_id,
              task_id: req_task_id,
              eval_id: req_eval_id,
            },
          },
        },
      )
      if (
        req_project_id !== project_id ||
        req_task_id !== task_id ||
        req_eval_id !== eval_id
      )
        return
      if (error) {
        throw error
      }
      evaluator = data
    } catch (error) {
      if (
        req_project_id !== project_id ||
        req_task_id !== task_id ||
        req_eval_id !== eval_id
      )
        return
      eval_error = createKilnError(error)
    } finally {
      if (
        req_project_id === project_id &&
        req_task_id === task_id &&
        req_eval_id === eval_id
      ) {
        eval_loading = false
      }
    }
  }

  async function get_eval_progress(
    req_project_id: string,
    req_task_id: string,
    req_eval_id: string,
  ) {
    try {
      eval_progress_loading = true
      eval_progress = null
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/progress",
        {
          params: {
            path: {
              project_id: req_project_id,
              task_id: req_task_id,
              eval_id: req_eval_id,
            },
          },
        },
      )
      if (
        req_project_id !== project_id ||
        req_task_id !== task_id ||
        req_eval_id !== eval_id
      )
        return
      if (error) {
        throw error
      }
      eval_progress = data
    } catch (error) {
      if (
        req_project_id !== project_id ||
        req_task_id !== task_id ||
        req_eval_id !== eval_id
      )
        return
      eval_progress_error = createKilnError(error)
    } finally {
      if (
        req_project_id === project_id &&
        req_task_id === task_id &&
        req_eval_id === eval_id
      ) {
        eval_progress_loading = false
      }
    }
  }

  // Progress is a separate request from the eval itself, so until it lands there is no
  // count to append.
  function item_count_suffix(count: number | null | undefined): string {
    return count === null || count === undefined ? "" : ` (${count} items)`
  }

  function get_eval_properties(
    evaluator: Eval | null,
    spec: Spec | null,
    eval_progress: EvalProgress | null,
    modelInfo: ProviderModels | null,
  ): UiProperty[] {
    if (!evaluator) {
      return []
    }
    const properties: UiProperty[] = []

    properties.push({
      name: "Name",
      value: evaluator.name,
    })
    if (evaluator.description) {
      properties.push({
        name: "Description",
        value: evaluator.description,
      })
    }
    properties.push({
      name: "Type",
      value: eval_type_display(
        spec,
        evaluator,
        judge_type_from_config(eval_progress?.current_eval_method),
      ),
      tooltip: "The judge type scoring this eval, and what it checks.",
    })
    if (evaluator.evaluation_data_type === "full_trace") {
      properties.push({
        name: "Conversation History",
        value: "Included",
        tooltip:
          "When included, your task runs will be evaluated on their full conversation history including intermediate steps and tool calls. When disabled, only the final answer is evaluated.",
      })
    }
    properties.push({
      name: "ID",
      value: evaluator.id || "unknown",
    })

    // Every dataset row renders whether or not the eval has that dataset. A dataset is
    // only there if something explicitly wrote one (functional spec 3.2 — unconfigured
    // splits stay unconfigured), so most pre-existing evals are missing several. Hiding a
    // row would make "this eval has no val set" indistinguishable from "this page doesn't
    // show val sets". With no dataset there is no filter to show, no items to count and
    // nothing to link to, so the row says so instead.
    const NOT_CONFIGURED = "Not configured"

    const test_filter_id = eval_split_filter_id(evaluator, "test")
    properties.push({
      name: "Test Dataset",
      tooltip:
        "Held-out data for measuring final quality. Not used for training or tuning. Shown in the 'Compare' view metrics.",
      value: test_filter_id
        ? test_filter_id + item_count_suffix(eval_progress?.dataset_size)
        : NOT_CONFIGURED,
      link: test_filter_id
        ? linkFromFilterId(
            project_id,
            task_id,
            task_run_split_filter_id(evaluator, "test"),
          )
        : undefined,
    })

    const train_filter_id = eval_split_filter_id(evaluator, "train")
    properties.push({
      name: "Training Dataset",
      value: train_filter_id
        ? train_filter_id + item_count_suffix(eval_progress?.train_dataset_size)
        : NOT_CONFIGURED,
      tooltip: "The training set used for optimization.",
      link: train_filter_id
        ? linkFromFilterId(
            project_id,
            task_id,
            task_run_split_filter_id(evaluator, "train"),
          )
        : undefined,
    })

    const val_filter_id = eval_split_filter_id(evaluator, "val")
    properties.push({
      name: "Validation Dataset",
      value: val_filter_id
        ? val_filter_id + item_count_suffix(eval_progress?.val_dataset_size)
        : NOT_CONFIGURED,
      tooltip: "The validation set used for optimization.",
      link: val_filter_id
        ? linkFromFilterId(
            project_id,
            task_id,
            task_run_split_filter_id(evaluator, "val"),
          )
        : undefined,
    })

    // Golden comes after the three `splits` datasets rather than beside the test one.
    // Test/training/validation are the eval's own splits and read as a set; golden is a
    // different kind of thing (human-rated items for judging the judge), so it sits at
    // the end instead of interrupting them.
    const golden_filter_id = evaluator.eval_configs_filter_id
    properties.push({
      name: "Golden Dataset",
      value: golden_filter_id
        ? golden_filter_id +
          item_count_suffix(eval_progress?.golden_dataset_size)
        : NOT_CONFIGURED,
      tooltip:
        "This is the dataset that we use to evaluate the quality of judge models. Items in this set need human ratings so we can compare judge ratings to human ratings.",
      link: golden_filter_id
        ? linkFromFilterId(project_id, task_id, golden_filter_id)
        : undefined,
    })

    if (eval_progress?.current_eval_method) {
      if (eval_progress.current_eval_method.config_type === "v2") {
        properties.push({
          name: "Judge Type",
          value: eval_config_to_detailed_ui_name(
            eval_progress.current_eval_method,
          ),
          tooltip: "The type of judge used for evaluation.",
        })
      } else {
        properties.push({
          name: "Judge Algorithm",
          value: eval_config_to_ui_name(
            eval_progress.current_eval_method.config_type,
          ),
          tooltip: "The evaluation algorithm used by your selected judge.",
        })
        properties.push({
          name: "Judge Model",
          value: getDetailedModelNameFromParts(
            eval_progress.current_eval_method.model_name ?? "",
            eval_progress.current_eval_method.model_provider ?? "",
            modelInfo,
          ),
          tooltip: "The model used by your selected judge.",
        })
      }
    }

    return properties
  }

  $: has_default_eval_config = evaluator && evaluator.current_config_id

  let edit_dialog: EditDialog | null = null

  const MIN_DATASET_SIZE = 25
  // Lower than the test goal because golden takes the smallest share of generated data
  // (GOLDEN_SPLIT_WEIGHT is 10 of 100). Held to the same 25, golden — not the test set —
  // is what this step waits on.
  const MIN_GOLDEN_DATASET_SIZE = 12
  let current_step: 0 | 1 | 2 | 3 | 4 | 5 | 6 = 0
  let current_step_id:
    | "goals"
    | "eval_data"
    | "human_ratings"
    | "compare_judges"
    | "compare_run_configs"
    | "unknown" = "unknown"
  let required_more_eval_data = false
  let required_more_golden_data = false
  // Steps jumped over by picking a default judge without completing them.
  // Rendered distinctly in the stepper so they don't read as completed.
  let skipped_steps: Set<number> = new Set()
  // Steps done but below the recommended amount (e.g. some eval data, fewer
  // than the suggested minimum). Rendered like the current step (dot instead
  // of a check) to signal there's more worth doing.
  let partial_steps: Set<number> = new Set()
  let goals: string[] = []
  let golden_dataset_explanation = ""

  function number_of_steps(evaluator: Eval | null): number {
    if (evaluator?.template === "rag") {
      return 3
    } else {
      return 5
    }
  }

  function step_id_from_title(
    title: string,
  ):
    | "goals"
    | "eval_data"
    | "human_ratings"
    | "compare_judges"
    | "compare_run_configs"
    | "unknown" {
    switch (title) {
      case "Define Goals":
        return "goals"
      case "Create Eval Data":
        return "eval_data"
      case "Human Ratings":
        return "human_ratings"
      case "Find the Best Judge":
        return "compare_judges"
      case "Find the Best Way to Run this Task":
        return "compare_run_configs"
      default:
        return "unknown"
    }
  }

  function step_titles(evaluator: Eval | null): string[] {
    if (evaluator?.template === "rag") {
      return [
        "Define Goals",
        "Create Eval Data",
        "Find the Best Way to Run this Task",
      ]
    } else {
      return [
        "Define Goals",
        "Create Eval Data",
        "Human Ratings",
        "Find the Best Judge",
        "Find the Best Way to Run this Task",
      ]
    }
  }

  function step_tooltips(evaluator: Eval | null): Record<number, string> {
    if (evaluator?.template === "rag") {
      return {
        1: "Each eval needs a set of quality goals to measure (aka 'eval scores'). You can add separate evals for different goals, or multiple goals to the same eval.",
        2: "Each eval needs a Q&A dataset to help find the best way of running your task by comparing outputs against reference answers. We'll help you create it with synthetic data!",
        3: "This tool will help you compare a variety of options for running this task and find the best one for your eval's goals. You can compare different models, prompts, tools, or fine-tunes.",
      }
    } else {
      return {
        1: "Each eval needs a set of quality goals to measure (aka 'eval scores'). You can add separate evals for different goals, or multiple goals to the same eval.",
        2: "Evals need datasets for training, validating and evaluating. Import a dataset, or use synthetic data generation.",
        3: "A 'golden' dataset is a dataset of items that are rated by humans. Rating a 'golden' dataset lets us determine if the judge is working by checking how well it aligns to human preferences. ",
        4: "Benchmark various judge methods (model+prompt+algorithm). We'll compare judges to your golden dataset to find the judge which best matches your human preferences.",
        5: "This tool will help you compare a variety of options for running this task and find the best one for your eval's goals. You can compare different models, prompts, tools, or fine-tunes.",
      }
    }
  }

  function update_eval_progress(
    progress: EvalProgress | null,
    evaluator: Eval | null,
  ) {
    update_golden_dataset_explanation(progress)
    current_step = 1
    current_step_id = "goals"
    skipped_steps = new Set()
    partial_steps = new Set()
    if (!progress || !evaluator) {
      return
    }

    // Goals are setup. Generate friendly names for them.
    goals = []
    for (const output of evaluator.output_scores) {
      goals.push(output.name + " (" + output.type + ")")
    }

    if (has_default_eval_config && progress.dataset_size > 0) {
      // The user bypassed the recommended steps by selecting a default judge,
      // so the final step is available — but any bypassed step renders as
      // "skipped", not completed, so it's clear the judge was never validated
      // against human ratings. An empty eval set still means there is nothing
      // to run, so that case falls through to the eval-data step instead
      // (common for non-LLM judges, which are set as the default judge at
      // creation time).
      const skipped = new Set<number>()
      const partial = new Set<number>()
      // The dataset minimum is a recommendation: with at least one item the
      // step was genuinely done, so below-minimum renders as partial (hollow
      // circle), never skipped.
      required_more_eval_data = progress.dataset_size < MIN_DATASET_SIZE
      required_more_golden_data =
        evaluator?.template !== "rag" &&
        progress.golden_dataset_size < MIN_GOLDEN_DATASET_SIZE
      if (required_more_eval_data || required_more_golden_data) {
        partial.add(2)
      }
      if (evaluator?.template === "rag") {
        current_step = 3
      } else {
        if (golden_dataset_explanation) {
          // Human ratings are incomplete. The judge step still shows as done:
          // a default judge is set, which is what that step establishes.
          skipped.add(3)
        }
        current_step = 5
      }
      skipped_steps = skipped
      partial_steps = partial
      current_step_id = "compare_run_configs"
      return
    }

    current_step = 2
    current_step_id = "eval_data"
    required_more_eval_data = progress.dataset_size < MIN_DATASET_SIZE
    required_more_golden_data =
      evaluator?.template !== "rag" &&
      progress.golden_dataset_size < MIN_GOLDEN_DATASET_SIZE
    if (required_more_eval_data || required_more_golden_data) {
      return
    }

    if (evaluator?.template === "rag") {
      // RAG evals don't have a golden dataset or compare judges step. Everything is setup!
      current_step = 3
      current_step_id = "compare_run_configs"
    } else {
      current_step = 3
      current_step_id = "human_ratings"
      if (golden_dataset_explanation) {
        return
      }

      current_step = 4
      current_step_id = "compare_judges"
      if (!has_default_eval_config) {
        return
      }

      // Everything is setup!
      current_step = 5
      current_step_id = "compare_run_configs"
    }
  }
  $: update_eval_progress(eval_progress, evaluator)

  function update_golden_dataset_explanation(progress: EvalProgress | null) {
    if (!progress) {
      return
    }
    if (progress.golden_dataset_size == 0) {
      golden_dataset_explanation =
        "Your golden dataset is empty. Add data to your golden dataset to get started."
      return
    }
    let golden_dataset_rating_issues: string[] = []
    if (progress.golden_dataset_not_rated_count > 0) {
      golden_dataset_rating_issues.push(
        `${progress.golden_dataset_not_rated_count} item${progress.golden_dataset_not_rated_count == 1 ? " is" : "s are"} unrated`,
      )
    }
    if (progress.golden_dataset_partially_rated_count > 0) {
      golden_dataset_rating_issues.push(
        `${progress.golden_dataset_partially_rated_count} item${progress.golden_dataset_partially_rated_count == 1 ? " is" : "s are"} partially unrated`,
      )
    }
    if (golden_dataset_rating_issues.length > 0) {
      // Some golden dataset items are not fully rated.
      golden_dataset_explanation = `In your golden dataset ${golden_dataset_rating_issues.join(" and ")}. Fully rate all items to to get the best results from your eval.`
    } else {
      golden_dataset_explanation = ""
    }
  }

  function add_eval_data() {
    if (!evaluator) {
      alert("Unable to add eval data. Please try again later.")
      return
    }
    // Adding eval data writes TaskRuns, so only a TaskRun-backed test split has a tag
    // it can add under. An EvalInput-backed one gets its own message: "use a tag filter
    // instead" is not advice that helps when the store, not the filter's form, is what
    // this flow can't reach. The message diagnoses and stops there — nothing in this app
    // creates eval inputs, so there is no action to point at.
    if (eval_split(evaluator, "test")?.source === "eval_input") {
      alert(
        "This eval uses our new eval dataset format, which can't be generated from this UI.",
      )
      return
    }
    const test_filter_id = task_run_split_filter_id(evaluator, "test")
    const test_tag = test_filter_id
      ? tagFromFilterId(test_filter_id)
      : undefined
    let golden_tag: string | undefined = undefined
    if (evaluator?.eval_configs_filter_id) {
      golden_tag = tagFromFilterId(evaluator.eval_configs_filter_id)
    }
    if (!test_tag || (evaluator.template !== "rag" && !golden_tag)) {
      alert(
        "No test or golden dataset tag found. If you're using a custom filter, please setup the dataset manually.",
      )
      return
    }

    const params = new URLSearchParams()
    if (evaluator.template === "rag") {
      params.set("reason", "qna")
    } else {
      params.set("reason", "eval")
    }
    if (evaluator.template) {
      params.set("template_id", evaluator.template)
    }
    params.set("eval_id", `${project_id}::${task_id}::${eval_id}`)
    params.set("eval_link", window.location.pathname)

    // Every entry into this flow allocates generated data the same way, so the eval's
    // splits decide the allocation rather than which button reached here.
    const splits_param = build_eval_generation_splits_param(evaluator)
    if (splits_param) {
      params.set("splits", splits_param)
    }

    // Add tool_id for tool call evals so generation can enable the tool.
    // Each era keeps it somewhere different: appropriate_tool_use specs store
    // tool_id, legacy tool_call-template evals store it in
    // template_properties, and new-style tool evals (template-less, scored by
    // the tool_call_check judge) list tool function names on the judge.
    let tool_id: string | undefined = undefined
    if (evaluator.template === "tool_call") {
      const spec_properties = spec?.properties
      if (spec_properties?.spec_type === "appropriate_tool_use") {
        tool_id = spec_properties?.tool_id
      } else {
        tool_id = evaluator.template_properties?.tool_id as string | undefined
      }
    } else {
      tool_id = tool_id_from_judge()
    }
    if (tool_id) {
      params.set("tool_id", String(tool_id))
    }

    const url = `/dataset/${project_id}/${task_id}/add_data?${params.toString()}`
    show_progress_ui("When you're done adding data, ", 2)
    goto(url)
  }

  // The tool id for the first expected tool of a tool_call_check default
  // judge, resolved by function name from the project's available tools.
  // Undefined for other judge types, unknown tools, or before tools load —
  // the add-data flow works without it, the tool just isn't pre-enabled.
  function tool_id_from_judge(): string | undefined {
    const judge_properties = eval_progress?.current_eval_method?.properties as
      | Record<string, unknown>
      | null
      | undefined
    if (judge_properties?.["type"] !== "tool_call_check") {
      return undefined
    }
    const expected_tools = judge_properties["expected_tools"]
    if (!Array.isArray(expected_tools) || expected_tools.length === 0) {
      return undefined
    }
    const first = expected_tools[0]
    const function_name =
      first && typeof first === "object"
        ? (first as { tool_name?: unknown }).tool_name
        : undefined
    if (typeof function_name !== "string" || !function_name) {
      return undefined
    }
    for (const tool_set of $available_tools[project_id] ?? []) {
      for (const tool of tool_set.tools) {
        if ((tool.function_name ?? tool.name) === function_name) {
          return tool.id
        }
      }
    }
    return undefined
  }

  function show_progress_ui(body: string, step: number) {
    progress_ui_state.set({
      title: "Creating Eval",
      body,
      link: $page.url.pathname,
      cta: "return to the eval",
      progress: null,
      step_count: number_of_steps(evaluator),
      current_step: step,
    })
  }

  function show_golden_dataset() {
    if (!evaluator || !evaluator.eval_configs_filter_id) {
      return
    }
    const url = linkFromFilterId(
      project_id,
      task_id,
      evaluator.eval_configs_filter_id,
    )
    if (!url) {
      return
    }

    show_progress_ui("When you're done rating, ", 3)
    goto(url)
  }

  function compare_eval_methods() {
    let url = `/specs/${project_id}/${task_id}/${spec_id}/${eval_id}/eval_configs`
    show_progress_ui("When you're done comparing judges, ", 4)
    goto(url)
  }

  function compare_run_configs() {
    let url = `/specs/${project_id}/${task_id}/${spec_id}/${eval_id}/compare_run_configs`
    goto(url)
  }
</script>

<div class="max-w-[1400px]">
  <AppPage
    title="Eval: {evaluator?.name || ''}"
    subtitle="Follow these steps to find the best way to evaluate and run your task"
    breadcrumbs={spec_id === "legacy"
      ? [
          {
            label: "Evals",
            href: `/specs/${project_id}/${task_id}`,
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
        ]}
    action_buttons={is_legacy_eval
      ? [
          {
            label: "Edit",
            disabled: loading || error !== null,
            handler: () => {
              edit_dialog?.show()
            },
          },
        ]
      : []}
  >
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
      <div class="flex flex-col xl:flex-row gap-8 xl:gap-16 mb-8">
        <div class="grow">
          <ul class="steps steps-vertical ml-4 overflow-x-hidden">
            {#each Array.from({ length: number_of_steps(evaluator) }, (_, i) => i + 1) as step}
              {@const step_title = step_titles(evaluator)[step - 1]}
              {@const step_id = step_id_from_title(step_title)}
              <li
                class="step {skipped_steps.has(step)
                  ? ''
                  : current_step >= step
                    ? 'step-primary'
                    : ''}"
                data-content={skipped_steps.has(step)
                  ? "–"
                  : partial_steps.has(step) || current_step == step
                    ? "●"
                    : current_step > step
                      ? "✓"
                      : ""}
              >
                <div
                  class="text-left py-3 min-h-[100px] flex flex-col place-content-center pl-4"
                >
                  <div class="font-medium">
                    {step_title}
                    {#if step_tooltips(evaluator)[step]}
                      <InfoTooltip
                        tooltip_text={step_tooltips(evaluator)[step]}
                        position={step < 4 ? "bottom" : "top"}
                        no_pad={true}
                      />
                    {/if}
                  </div>
                  <div class="text-sm text-gray-500">
                    {#if step_id == "goals" && goals.length > 0}
                      This eval has {goals.length} goal{goals.length == 1
                        ? ""
                        : "s"}: {goals.join(", ")}.
                    {:else if step_id == "eval_data"}
                      <div>
                        <div class="mb-1">
                          {#if eval_progress && !required_more_eval_data && !required_more_golden_data}
                            {#if evaluator?.template === "rag"}
                              You have {eval_progress?.dataset_size} test dataset
                              items.
                            {:else}
                              You have {eval_progress?.dataset_size} test dataset
                              items and {eval_progress?.golden_dataset_size} golden
                              items.
                            {/if}
                          {:else if eval_progress && eval_progress.dataset_size == 0 && eval_progress.golden_dataset_size == 0 && evaluator.template === "rag"}
                            Create a query & answer dataset for this eval.
                          {:else if eval_progress && eval_progress.dataset_size == 0 && eval_progress.golden_dataset_size == 0}
                            Create data for this eval.
                          {:else if eval_progress && (required_more_eval_data || required_more_golden_data)}
                            You require additional eval data. You only have
                            {#if required_more_eval_data && required_more_golden_data}
                              {eval_progress?.dataset_size} test dataset items and
                              {eval_progress?.golden_dataset_size}
                              golden items. We suggest at least {MIN_DATASET_SIZE}
                              test dataset items and {MIN_GOLDEN_DATASET_SIZE} golden
                              items.
                            {:else if required_more_eval_data}
                              {eval_progress?.dataset_size} test dataset items. We
                              suggest at least {MIN_DATASET_SIZE} items.
                            {:else if required_more_golden_data}
                              {eval_progress?.golden_dataset_size} golden items.
                              We suggest at least {MIN_GOLDEN_DATASET_SIZE} items.
                            {/if}
                          {/if}
                        </div>
                        {#if eval_split(evaluator, "test")?.source === "eval_input"}
                          <!-- EvalInput-typed slice: items are minted by the eval
                            builder at save; the add-data flow tags TaskRuns, which
                            doesn't apply, so offer no dead-end button. -->
                          This eval's data was created by the eval builder and can't
                          be extended here.
                        {:else}
                          <button
                            class="btn btn-sm {current_step_id == 'eval_data'
                              ? 'btn-primary'
                              : ''}"
                            on:click={add_eval_data}
                          >
                            Add Eval Data
                          </button>
                        {/if}
                      </div>
                    {:else if step_id == "human_ratings"}
                      <div class="mb-1">
                        {#if golden_dataset_explanation}
                          {golden_dataset_explanation}
                        {:else}
                          All items in your golden dataset are fully rated.
                        {/if}
                      </div>
                      <div>
                        {#if evaluator.eval_configs_filter_id && linkFromFilterId(project_id, task_id, evaluator.eval_configs_filter_id)}
                          <button
                            class="btn btn-sm {current_step_id ==
                            'human_ratings'
                              ? 'btn-primary'
                              : ''}"
                            on:click={show_golden_dataset}
                          >
                            {golden_dataset_explanation ? "Rate" : "View"} Golden
                            Dataset
                          </button>
                        {:else}
                          <!-- We always use "tag::" so this shouldn't happen unless it's created by code. -->
                          Your golden dataset is filtered by
                          <span class="font-mono bg-gray-100 p-1"
                            >{evaluator.eval_configs_filter_id}</span
                          >. Please rate these entries in the
                          <a
                            class="link"
                            href={`/dataset/${project_id}/${task_id}`}
                            >dataset tab</a
                          >.
                        {/if}
                      </div>
                    {:else if step_id == "compare_judges"}
                      <div class="mb-1">
                        {#if eval_progress?.current_eval_method}
                          You selected the judge '{eval_config_to_detailed_ui_name(
                            eval_progress.current_eval_method,
                          )}'{#if eval_progress.current_eval_method.model_name}
                            using the model '{model_name(
                              eval_progress.current_eval_method.model_name,
                              $model_info,
                            )}'{/if}.
                        {:else}
                          Compare automated evals to find one that aligns with
                          your human preferences.
                        {/if}
                      </div>
                      <div>
                        <button
                          class="btn btn-sm {current_step_id == 'compare_judges'
                            ? 'btn-primary'
                            : ''}"
                          on:click={compare_eval_methods}
                        >
                          Compare Judges
                        </button>
                      </div>
                    {:else if step_id == "compare_run_configs"}
                      <div class="mb-1">
                        Compare models, prompts, tools and fine-tunes to find
                        the most effective.
                      </div>
                      <div>
                        <button
                          class="btn btn-sm {current_step_id ==
                          'compare_run_configs'
                            ? 'btn-primary'
                            : ''}"
                          on:click={compare_run_configs}
                        >
                          Compare Run Configurations
                        </button>
                      </div>
                    {/if}
                  </div>
                </div>
              </li>
            {/each}
          </ul>
        </div>

        <div class="w-72 2xl:w-96 flex-none">
          <PropertyList
            properties={get_eval_properties(
              evaluator,
              spec,
              eval_progress,
              $model_info,
            )}
            title="Evaluator Properties"
          />
        </div>
      </div>
    {/if}
  </AppPage>
</div>

{#if is_legacy_eval}
  <EditDialog
    bind:this={edit_dialog}
    name="Eval"
    patch_url={`/api/projects/${project_id}/tasks/${task_id}/evals/${eval_id}`}
    delete_url={`/api/projects/${project_id}/tasks/${task_id}/evals/${eval_id}`}
    after_delete={() => {
      goto(`/specs/${project_id}/${task_id}`)
    }}
    fields={[
      {
        label: "Eval Name",
        description: "A name to identify this eval.",
        api_name: "name",
        value: evaluator?.name || "",
        input_type: "input",
        max_length: 120,
      },
      {
        label: "Description",
        description: "A description of the eval for you and your team.",
        api_name: "description",
        value: evaluator?.description || "",
        input_type: "textarea",
        optional: true,
      },
    ]}
  />
{/if}
