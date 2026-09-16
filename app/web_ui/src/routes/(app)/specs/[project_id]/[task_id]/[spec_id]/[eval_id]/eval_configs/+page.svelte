<script lang="ts">
  import AppPage from "../../../../../../app_page.svelte"
  import type { Eval, Spec } from "$lib/types"
  import { client } from "$lib/api_client"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { tick } from "svelte"
  import { page } from "$app/stores"
  import RunEval from "$lib/components/run_eval.svelte"
  import type { EvalConfig, EvalConfigCompareSummary } from "$lib/types"
  import FormElement from "$lib/utils/form_element.svelte"
  import { load_model_info, load_available_models } from "$lib/stores"
  import { load_task_prompts } from "$lib/stores/prompts_store"
  import { set_current_eval_config } from "$lib/stores/evals_store"
  import Warning from "$lib/ui/warning.svelte"
  import InfoTooltip from "$lib/ui/info_tooltip.svelte"
  import { string_to_json_key } from "$lib/utils/json_schema_editor/json_schema_templates"
  import Dialog from "$lib/ui/dialog.svelte"
  import { eval_config_to_detailed_ui_name } from "$lib/utils/formatters"
  import type { TaskOutputRatingType } from "$lib/types"
  import type { UiProperty } from "$lib/ui/property_list"
  import Intro from "$lib/ui/intro.svelte"
  import EvalConfigInstruction from "./eval_config_instruction.svelte"
  import ClampedText from "$lib/ui/clamped_text.svelte"
  import {
    comparable_eval_configs,
    compute_run_disallowed_missing_ref_data,
  } from "$lib/utils/eval_types/judge_comparison_gate"

  import { agentInfo } from "$lib/agent"
  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  $: eval_id = $page.params.eval_id!
  $: spec_id = $page.params.spec_id!
  $: agentInfo.set({
    name: "Eval Configs",
    description: `Eval configurations for eval ID ${eval_id}, spec ID ${spec_id} in project ID ${project_id}, task ID ${task_id}. Eval name: ${evaluator?.name ?? "[loading]"}. Lists available evaluation configurations.`,
  })

  let spec: Spec | null = null
  let spec_loading = true
  let spec_error: KilnError | null = null

  let score_legend_dialog: Dialog | null = null

  let evaluator: Eval | null = null
  let eval_error: KilnError | null = null
  let eval_loading = true

  let eval_config_instructions_dialog: Dialog | null = null
  let displayed_eval_config: EvalConfig | null = null
  $: displayed_eval_config_dialog_title = displayed_eval_config
    ? `Details for Judge '${displayed_eval_config.name}'`
    : "Details for Judge"
  $: displayed_eval_config_dialog_subtitle = displayed_eval_config
    ? `${eval_config_type_label(displayed_eval_config)}`
    : undefined

  let eval_configs: EvalConfig[] | null = null
  let eval_configs_error: KilnError | null = null
  let eval_configs_loading = true

  let score_summary: EvalConfigCompareSummary | null = null
  let score_summary_error: KilnError | null = null

  type ScoreType =
    | "mse"
    | "mae"
    | "norm_mse"
    | "norm_mae"
    | "spearman"
    | "pearson"
    | "kendalltau"

  let score_type: ScoreType = "kendalltau"

  $: loading = eval_loading || eval_configs_loading || spec_loading // Score summary not blocking whole UI
  $: error =
    eval_error || eval_configs_error || score_summary_error || spec_error

  let eval_state:
    | "not_started"
    | "running"
    | "complete"
    | "complete_with_errors" = "not_started"
  // The judges this page can actually compare. A judge that grades against reference
  // data is excluded everywhere the page reasons about completion or about picking a
  // winner: judge comparison never scores it, so counting it would either report a
  // completion it can't reach or nudge the user to crown a judge with no scores.
  $: comparable_configs = comparable_eval_configs(eval_configs, evaluator)
  $: should_select_eval_config = !!(
    comparable_configs.length && !evaluator?.current_config_id
  )
  $: focus_select_eval_config = !!(
    should_select_eval_config &&
    (eval_state?.includes("complete") ||
      (score_summary &&
        comparable_configs.every(
          (config) =>
            (score_summary!.eval_config_percent_complete?.["" + config.id] ||
              0.0) >= 1.0,
        )))
  )

  // Sort eval_configs whenever score_type or score_summary changes
  $: if (score_type && eval_configs && score_summary && evaluator) {
    sortEvalConfigs()
  }

  function sortEvalConfigs() {
    if (!eval_configs) return
    if (!evaluator) return
    const nonNullEvaluator = evaluator

    const sorted = [...eval_configs].sort((a, b) => {
      // Always put default (current) config on top
      if (a.id === nonNullEvaluator.current_config_id) return -1
      if (b.id === nonNullEvaluator.current_config_id) return 1

      // If no score summary, keep original order
      if (
        !score_summary ||
        !nonNullEvaluator.output_scores ||
        nonNullEvaluator.output_scores.length === 0
      ) {
        return 0
      }

      // Get the last output score for sorting
      const lastOutputScore =
        nonNullEvaluator.output_scores[
          nonNullEvaluator.output_scores.length - 1
        ]
      const scoreNameKey = string_to_json_key(lastOutputScore.name)

      const aScores = score_summary?.results?.["" + a.id]?.[scoreNameKey]
      const bScores = score_summary?.results?.["" + b.id]?.[scoreNameKey]

      // Handle missing scores (put them at the end)
      if (!aScores && !bScores) return 0
      if (!aScores) return 1
      if (!bScores) return -1

      let aValue, bValue

      // Get the appropriate score based on score_type
      if (score_type === "mae") {
        aValue = aScores.mean_absolute_error
        bValue = bScores.mean_absolute_error
        // Lower is better for MAE, so sort ascending
        return (aValue ?? Infinity) - (bValue ?? Infinity)
      } else if (score_type === "mse") {
        aValue = aScores.mean_squared_error
        bValue = bScores.mean_squared_error
        // Lower is better for MSE, so sort ascending
        return (aValue ?? Infinity) - (bValue ?? Infinity)
      } else if (score_type === "norm_mse") {
        aValue = aScores.mean_normalized_squared_error
        bValue = bScores.mean_normalized_squared_error
        // Lower is better for normalized MSE, so sort ascending
        return (aValue ?? Infinity) - (bValue ?? Infinity)
      } else if (score_type === "norm_mae") {
        aValue = aScores.mean_normalized_absolute_error
        bValue = bScores.mean_normalized_absolute_error
        // Lower is better for normalized MAE, so sort ascending
        return (aValue ?? Infinity) - (bValue ?? Infinity)
      } else if (score_type === "spearman") {
        aValue = aScores.spearman_correlation
        bValue = bScores.spearman_correlation
        // Higher is better for correlation, so sort descending
        // Handle null/undefined values
        if (aValue === null || aValue === undefined) return 1
        if (bValue === null || bValue === undefined) return -1
        return bValue - aValue
      } else if (score_type === "pearson") {
        aValue = aScores.pearson_correlation
        bValue = bScores.pearson_correlation
        // Higher is better for correlation, so sort descending
        // Handle null/undefined values
        if (aValue === null || aValue === undefined) return 1
        if (bValue === null || bValue === undefined) return -1
        return bValue - aValue
      } else if (score_type === "kendalltau") {
        aValue = aScores.kendalltau_correlation
        bValue = bScores.kendalltau_correlation
        // Higher is better for correlation, so sort descending
        // Handle null/undefined values
        if (aValue === null || aValue === undefined) return 1
        if (bValue === null || bValue === undefined) return -1
        return bValue - aValue
      }

      return 0
    })

    // Only assign when the ordering really changed
    if (!sorted.every((v, i) => v === (eval_configs || [])[i])) {
      eval_configs = sorted
    }
  }

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
    await Promise.all([
      load_model_info(),
      load_task_prompts(req_project_id, req_task_id),
      load_available_models(),
      get_spec(req_project_id, req_task_id, req_spec_id),
      get_eval(req_project_id, req_task_id, req_eval_id),
    ])
    if (
      req_project_id !== project_id ||
      req_task_id !== task_id ||
      req_eval_id !== eval_id
    )
      return
    get_eval_config(req_project_id, req_task_id, req_eval_id)
    get_score_summary(req_project_id, req_task_id, req_eval_id)
  }

  async function get_spec(
    req_project_id: string,
    req_task_id: string,
    req_spec_id: string,
  ) {
    if (req_spec_id === "legacy") {
      if (req_project_id === project_id && req_task_id === task_id) {
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
      // Override default score_type to "mse" when eval rating type is pass_fail or pass_fail_critical
      // We have different scores to compare human preference to LLM-as-Judge in evals. Kendall Tau (the current default) is a great score for comparing ranked data, but kinda lousy for pass/fail data
      const hasPassFailRating = evaluator.output_scores.some(
        (score) =>
          score.type === "pass_fail" || score.type === "pass_fail_critical",
      )
      if (hasPassFailRating) {
        score_type = "mse"
      }
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

  async function get_eval_config(
    req_project_id: string,
    req_task_id: string,
    req_eval_id: string,
  ) {
    try {
      eval_configs_loading = true
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/eval_configs",
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

      eval_configs = data
      // Initial sort will be handled by the reactive statement
    } catch (error) {
      if (
        req_project_id !== project_id ||
        req_task_id !== task_id ||
        req_eval_id !== eval_id
      )
        return
      eval_configs_error = createKilnError(error)
    } finally {
      if (
        req_project_id === project_id &&
        req_task_id === task_id &&
        req_eval_id === eval_id
      ) {
        eval_configs_loading = false
      }
    }
  }

  async function get_score_summary(
    req_project_id: string,
    req_task_id: string,
    req_eval_id: string,
  ) {
    try {
      score_summary = null
      score_summary_error = null
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/eval_configs_score_summary",
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
      score_summary = data
    } catch (error) {
      if (
        req_project_id !== project_id ||
        req_task_id !== task_id ||
        req_eval_id !== eval_id
      )
        return
      score_summary_error = createKilnError(error)
    }
  }

  function get_eval_properties(
    evaluator: Eval,
    score_summary: EvalConfigCompareSummary | null,
  ): UiProperty[] {
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

    let eval_configs_set_size = ""
    if (score_summary) {
      eval_configs_set_size = " (" + score_summary.dataset_size + " items)"
    }
    if (evaluator.eval_configs_filter_id) {
      properties.push({
        name: "Golden Dataset",
        value: evaluator.eval_configs_filter_id + eval_configs_set_size,
      })
    }
    return properties
  }

  function incomplete_warning(
    score_summary: EvalConfigCompareSummary | null,
    comparable_configs: EvalConfig[],
  ): string[] {
    if (!score_summary) {
      return []
    }

    const warnings: string[] = []
    if (score_summary.dataset_size === 0) {
      warnings.push(
        "There are zero items in your golden dataset. You can add and tag data from page for this eval.",
      )
    }
    if (score_summary.not_rated_count > 0) {
      warnings.push(
        `${score_summary.not_rated_count} item(s) in your golden dataset are not rated at all. Add human ratings to these items in the dataset tab.`,
      )
    }
    if (score_summary.partially_rated_count > 0) {
      warnings.push(
        `${score_summary.partially_rated_count} item(s) in your golden dataset are only partially rated. Add human ratings for each score in the dataset tab.`,
      )
    }

    // Only the judges "Run All Evals" will actually run. A judge that grades against
    // reference data never completes here, so counting it would leave this warning
    // permanently on screen telling the user to click a button that skips it.
    //
    // Skipped entirely for an empty golden dataset: the server returns
    // `eval_config_percent_complete: {}` in that case, and reading a missing entry as 0%
    // would stack "click Run All Evals to generate the missing scores" on top of the
    // "there are zero items in your golden dataset" warning above, where there is
    // nothing to run.
    if (score_summary.dataset_size > 0) {
      const completion_values = comparable_configs.map(
        (config) =>
          score_summary.eval_config_percent_complete?.["" + config.id] || 0.0,
      )
      const minComplete =
        completion_values.length > 0
          ? completion_values.reduce((min, val) => Math.min(min, val), 1.0)
          : 1.0
      if (minComplete < 1.0) {
        warnings.push(
          "You evals are incomplete. Click 'Run All Evals' to generate scores for the missing items.",
        )
      }
    }

    return warnings
  }

  async function handle_set_current_eval_config(eval_config_id: string | null) {
    try {
      evaluator = await set_current_eval_config(
        project_id,
        task_id,
        eval_id,
        eval_config_id,
      )
    } catch (error) {
      eval_error = createKilnError(error)
    }
  }

  function eval_config_type_label(eval_config: EvalConfig): string {
    if (eval_config.config_type === "v2") {
      return eval_config_to_detailed_ui_name(eval_config)
    }
    // V1 g_eval and llm_as_judge both show as "LLM as Judge"
    return "LLM as Judge"
  }

  function info_tooltip_text(
    rating_type: TaskOutputRatingType,
    score_type: ScoreType,
  ) {
    let label = ""
    if (score_type === "mae") {
      label = "Mean absolute error. Lower is better."
    } else if (score_type === "mse") {
      label = "Mean squared error. Lower is better."
    } else if (score_type === "norm_mse") {
      label = "Normalized mean squared error. Lower is better."
    } else if (score_type === "norm_mae") {
      label = "Normalized mean absolute error. Lower is better."
    } else if (score_type === "spearman") {
      label = "Spearman's rank correlation. Higher is better."
    } else if (score_type === "pearson") {
      label = "Pearson's correlation. Higher is better."
    } else if (score_type === "kendalltau") {
      label = "Kendall's Tau correlation. Higher is better."
    }
    label += " For "
    if (rating_type === "five_star") {
      label += "1 to 5 star rating."
    } else if (rating_type === "pass_fail") {
      label += "pass/fail rating."
    } else if (rating_type === "pass_fail_critical") {
      label += "pass/fail/critical rating."
    }
    return label
  }
</script>

<AppPage
  title="Compare Judges"
  subtitle="Find the judge that best matches human preferences"
  sub_subtitle="Read the docs"
  sub_subtitle_link="https://docs.kiln.tech/docs/evaluations#finding-the-ideal-eval-method"
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
  action_buttons={eval_configs?.length
    ? [
        {
          label: "Instructions",
          handler: () => {
            score_legend_dialog?.show()
          },
        },
        {
          label: "Add Judge",
          href: `/specs/${project_id}/${task_id}/${spec_id}/${eval_id}/create_eval_config?next_page=eval_configs`,
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
      <div class="font-medium">Error Loading</div>
      <div class="text-error text-sm">
        {error.getMessage() || "An unknown error occurred"}
      </div>
    </div>
  {:else if evaluator}
    {#if eval_configs?.length}
      <div class="flex flex-col xl:flex-row gap-8 xl:gap-16 mb-8">
        <div class="grow">
          <div class="text-xl font-bold mb-4">Evaluator Properties</div>
          <div
            class="grid grid-cols-[auto,1fr] gap-y-2 gap-x-4 text-sm 2xl:text-base"
          >
            {#each get_eval_properties(evaluator, score_summary) as property}
              <div class="flex items-center">{property.name}</div>
              <div class="flex items-center text-gray-500 overflow-x-hidden">
                {property.value}
              </div>
            {/each}
          </div>
          {#if score_summary && score_summary.dataset_size > 0 && score_summary.dataset_size < 25}
            <div class="mt-4">
              <Warning
                warning_message={`There are only ${score_summary.dataset_size} item(s) in your Golden Dataset. This is generally too small to get a sense of how judges perform.`}
                warning_color="warning"
                inline={true}
              />
            </div>
          {/if}
        </div>
      </div>
      <div class="mt-16">
        <div class="flex flex-col lg:flex-row gap-4 lg:gap-8 mb-6">
          <div class="grow">
            <div class="text-xl font-bold">
              Correlation to Human Preferences
            </div>
            <div class="text-xs text-gray-500">
              Each score in this table is a measure for how much each judge
              correlates to human preferences, using the selected scoring
              metric.
            </div>
            {#if score_summary_error}
              <div class="text-error text-sm">
                {score_summary_error.getMessage() ||
                  "An unknown error occurred fetching scores."}
              </div>
            {/if}
          </div>
          <div class="flex flex-row gap-2">
            <FormElement
              id="score-type"
              label=""
              aria_label="Score"
              inputType="select"
              select_options={[
                ["kendalltau", "Kendall's Tau Correlation"],
                ["spearman", "Spearman Rank Correlation"],
                ["norm_mse", "Normalized Mean Squared Error"],
                ["mse", "Mean Squared Error"],
                ["norm_mae", "Normalized Mean Absolute Error"],
                ["mae", "Mean Absolute Error"],
                ["pearson", "Pearson Correlation"],
              ]}
              bind:value={score_type}
            />
            <div class="mt-1">
              <RunEval
                btn_size="normal"
                bind:eval_state
                {project_id}
                {task_id}
                {eval_id}
                run_all={true}
                btn_primary={!focus_select_eval_config}
                eval_type="eval_config"
                on_run_complete={() => {
                  get_score_summary(project_id, task_id, eval_id)
                }}
              />
            </div>
          </div>
        </div>

        <!-- Warn the user if some evals are incomplete -->

        {#if incomplete_warning(score_summary, comparable_configs).length}
          <div class="mt-6 mb-4">
            <Warning
              warning_message={`There are issues you should resolve before analyzing this data.`}
              inline={true}
            />
            <ul class="list-disc list-inside text-sm text-gray-500 pl-2 pt-2">
              {#each incomplete_warning(score_summary, comparable_configs) as warning}
                <li>{warning}</li>
              {/each}
            </ul>
          </div>
        {:else if should_select_eval_config}
          <div class="mb-4 mt-2">
            <Warning
              warning_message="Click 'Set as Default' below to select a winner."
              warning_color={focus_select_eval_config ? "primary" : "gray"}
              warning_icon={focus_select_eval_config ? "exclaim" : "info"}
              large_icon={focus_select_eval_config}
              inline={true}
            />
          </div>
        {/if}

        <div class="overflow-x-auto rounded-lg border">
          <table class="table table-fixed">
            <thead>
              <tr>
                <th class="w-[200px] max-w-[250px]">
                  <div>Judge</div>
                  <div class="font-normal">How task output is evaluated</div>
                </th>
                <th class="text-center w-[160px]">Status</th>
                <th class="w-auto">Eval Details</th>
                {#each evaluator.output_scores as output_score}
                  <th class="text-center w-[130px]">
                    {output_score.name}
                    {#if output_score.type}
                      <InfoTooltip
                        tooltip_text={info_tooltip_text(
                          output_score.type,
                          score_type,
                        )}
                        no_pad={true}
                      />
                    {/if}
                  </th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each eval_configs || [] as eval_config}
                {@const percent_complete =
                  score_summary?.eval_config_percent_complete?.[
                    "" + eval_config.id
                  ] || 0.0}
                {@const run_disallowed_missing_ref_data =
                  compute_run_disallowed_missing_ref_data(
                    eval_config,
                    evaluator,
                  )}
                <tr>
                  <td class="max-w-[250px]">
                    <div class="font-medium break-words">
                      {eval_config.name}
                    </div>
                    <div class="text-sm text-gray-500">
                      Type: {eval_config_type_label(eval_config)}
                    </div>
                    {#if run_disallowed_missing_ref_data}
                      <div class="mt-2">
                        <Warning
                          warning_message="This judge requires reference data, which is not populated. It can't be run or compared here."
                          warning_color="warning"
                          inline={true}
                          text_size="xs"
                        />
                      </div>
                    {/if}
                  </td>
                  <td class="text-center text-sm">
                    {#if run_disallowed_missing_ref_data}
                      <!-- Neither complete nor behind: a percentage in error red would
                           read as something to fix, and the row's warning already says
                           this judge is one the comparison skips. -->
                      <div class="text-gray-500">Not comparable</div>
                    {:else if percent_complete < 1.0}
                      <div class="text-error">
                        {(percent_complete * 100.0).toFixed(0)}% Complete
                      </div>
                    {:else}
                      <div>Complete</div>
                    {/if}
                    {#if eval_config.id == evaluator.current_config_id}
                      <button
                        class="btn btn-xs rounded-full btn-primary mt-1 min-w-[120px]"
                        on:click={() => {
                          handle_set_current_eval_config(null)
                        }}
                      >
                        Default <span class="pl-[1px]">&#x2715;</span>
                      </button>
                    {:else}
                      <button
                        class="btn btn-xs rounded-full mt-1 min-w-[120px] {focus_select_eval_config &&
                        !run_disallowed_missing_ref_data
                          ? 'btn-primary'
                          : 'btn-secondary btn-outline'}"
                        on:click={() => {
                          if (eval_config.id) {
                            handle_set_current_eval_config(eval_config.id)
                          }
                        }}
                      >
                        Set as Default
                      </button>
                    {/if}
                  </td>
                  <td>
                    <div class="min-w-[300px]">
                      <ClampedText
                        max_lines={5}
                        on:see_all={() => {
                          displayed_eval_config = eval_config
                          eval_config_instructions_dialog?.show()
                        }}
                      >
                        <EvalConfigInstruction {eval_config} />
                      </ClampedText>
                    </div>
                  </td>
                  {#each evaluator.output_scores as output_score}
                    {@const scores =
                      score_summary?.results?.["" + eval_config.id]?.[
                        string_to_json_key(output_score.name)
                      ]}
                    <td class="text-center min-w-[115px]">
                      {#if scores}
                        {#if score_type === "mae"}
                          {scores.mean_absolute_error.toFixed(2)}
                        {:else if score_type === "mse"}
                          {scores.mean_squared_error.toFixed(2)}
                        {:else if score_type === "norm_mse"}
                          {scores.mean_normalized_squared_error.toFixed(3)}
                        {:else if score_type === "norm_mae"}
                          {scores.mean_normalized_absolute_error.toFixed(3)}
                        {:else if score_type === "spearman"}
                          {#if scores.spearman_correlation != null}
                            {scores.spearman_correlation.toFixed(3)}
                          {:else}
                            N/A <InfoTooltip
                              tooltip_text="There wasn't enough data, or variation in the data, to calculate a Spearman correlation. Add more data to your golden dataset, focusing on values which are missing (for example, if all current items pass, add some which fail)."
                              no_pad={true}
                            />
                          {/if}
                        {:else if score_type === "pearson"}
                          {#if scores.pearson_correlation != null}
                            {scores.pearson_correlation.toFixed(3)}
                          {:else}
                            N/A <InfoTooltip
                              tooltip_text="There wasn't enough data, or variation in the data, to calculate a Pearson correlation. Add more data to your golden dataset, focusing on values which are missing (for example, if all current items pass, add some which fail)."
                              no_pad={true}
                            />
                          {/if}
                        {:else if score_type === "kendalltau"}
                          {#if scores.kendalltau_correlation != null}
                            {scores.kendalltau_correlation.toFixed(3)}
                          {:else}
                            N/A <InfoTooltip
                              tooltip_text="There wasn't enough data, or variation in the data, to calculate a Kendall's Tau correlation. Add more data to your golden dataset, focusing on values which are missing (for example, if all current items pass, add some which fail)."
                              no_pad={true}
                            />
                          {/if}
                        {/if}
                      {:else}
                        None
                        {#if !run_disallowed_missing_ref_data}
                          <InfoTooltip
                            tooltip_text="No scores were found for this judge. Click 'Run All Evals' to generate scores and ensure your golden dataset has human ratings."
                            no_pad={true}
                          />
                        {/if}
                      {/if}
                    </td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </div>
    {:else}
      <div class="max-w-[300px] mx-auto flex flex-col gap-2 mt-[10vh]">
        <Intro
          title="Create a Judge to Get Started"
          description_paragraphs={[
            "A judge specifies how an eval is run (algorithm, model, instructions, etc).",
          ]}
          action_buttons={[
            {
              label: "Add Judge",
              href: `/specs/${project_id}/${task_id}/${spec_id}/${eval_id}/create_eval_config?next_page=eval_configs`,
              is_primary: true,
            },
          ]}
        />
      </div>
    {/if}
  {/if}
</AppPage>

<Dialog
  bind:this={eval_config_instructions_dialog}
  width="wide"
  title={displayed_eval_config_dialog_title}
  subtitle={displayed_eval_config_dialog_subtitle}
>
  {#if displayed_eval_config}
    <EvalConfigInstruction eval_config={displayed_eval_config} />
  {/if}
</Dialog>

<Dialog bind:this={score_legend_dialog} title="How to Compare Judges">
  <div class="font-medium text-sm text-gray-500">
    Each score is a correlation score between human ratings and the automated
    judge's scores. Use these scores to find the judge which best matches human
    preferences, and set it as your default judge.
  </div>
  <div class="m-8 font-light text-sm flex flex-col gap-2">
    <div class="font-bold text-xl">Quick Start</div>
    <div>
      Add a variety of judges with different options (model, algorithm,
      instructions). Then click 'Run All Evals' to generate scores from each
      judge on your golden dataset.
    </div>
    <div>
      We suggest you use Kendall's Tau correlation scores to compare results.
      Kendall's Tau scores range from -1.0 to 1. Higher values indicate higher
      correlation between the human ratings and the automated judge's scores.
      The absolute value of Kendall's Tau scores will vary depending on how
      subjective your task is.
    </div>
    <div>
      Finally, set the judge with the highest Kendall's Tau score as your
      default judge.
    </div>

    <div class="font-bold text-xl mt-6">Detailed Instructions</div>
    <div>
      <a
        href="https://docs.kiln.tech/docs/evaluations#finding-the-ideal-eval-method"
        target="_blank"
        class="link">Read the docs</a
      > for more information, a detailed walkthrough, and technical details about
      each scoring metric.
    </div>
  </div>
</Dialog>
