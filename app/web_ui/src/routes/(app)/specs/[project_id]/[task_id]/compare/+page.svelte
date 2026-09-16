<script lang="ts">
  import AppPage from "../../../../app_page.svelte"
  import { onMount, tick } from "svelte"
  import { page } from "$app/stores"
  import { goto } from "$app/navigation"
  import { client } from "$lib/api_client"
  import { createKilnError, KilnError } from "$lib/utils/error_handlers"
  import { formatLatency } from "$lib/utils/formatters"
  import type { Task, TaskRunConfig, Eval } from "$lib/types"
  import type { components } from "$lib/api_schema"
  import CompareChart from "$lib/components/compare_chart.svelte"
  import CompareRadarChart from "$lib/components/compare_radar_chart.svelte"
  type RunConfigEvalScoresSummary =
    components["schemas"]["RunConfigEvalScoresSummary"]
  type ScoreSummary = components["schemas"]["ScoreSummary"]
  import {
    model_info,
    load_model_info,
    load_available_models,
    get_task_composite_id,
    load_task,
  } from "$lib/stores"
  import {
    load_task_run_configs,
    run_configs_by_task_composite_id,
  } from "$lib/stores/run_configs_store"
  import {
    load_task_prompts,
    prompts_by_task_composite_id,
  } from "$lib/stores/prompts_store"
  import {
    getRunConfigModelDisplayName,
    getRunConfigPromptDisplayName,
    getRunConfigPromptInfoText,
    getRunConfigInputTransformSummaryLabel,
  } from "$lib/utils/run_config_formatters"
  import { isKilnAgentRunConfig, isMcpRunConfig } from "$lib/types"
  import InfoTooltip from "$lib/ui/info_tooltip.svelte"
  import { prompt_link } from "$lib/utils/link_builder"
  import { tagFromFilterId } from "../spec_utils"
  import { eval_split, task_run_split_filter_id } from "$lib/utils/eval_splits"
  import { build_eval_generation_splits_param } from "$lib/utils/eval_generation_splits"
  import CreateNewRunConfigDialog from "$lib/ui/run_config_component/create_new_run_config_dialog.svelte"
  import SavedRunConfigurationsDropdown from "$lib/ui/run_config_component/saved_run_configs_dropdown.svelte"
  import RunEval from "$lib/components/run_eval.svelte"
  import FloatingMenu from "$lib/ui/floating_menu.svelte"
  import type { FloatingMenuItem } from "$lib/ui/floating_menu_types"

  import { agentInfo } from "$lib/agent"
  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  // agentInfo.set is below, after validSelectedModels is defined
  $: fromOptimize = $page.url.searchParams.get("from") === "optimize"
  $: breadcrumbs = fromOptimize
    ? [{ label: "Optimize", href: `/optimize/${project_id}/${task_id}` }]
    : [{ label: "Evals", href: `/specs/${project_id}/${task_id}` }]

  // State management
  let columns = 2 // Start with 2 columns
  let selectedModels: (string | null)[] = [null, null] // Track selected model for each column
  let hiddenEvalIds: string[] = [] // Eval IDs hidden by the user (kiln_cost_section is never hideable)

  // Run configs state
  let loading_run_configs = true
  let loading_run_configs_error: KilnError | null = null

  // Task state
  let task: Task | null = null
  let loading_task = true
  let task_error: KilnError | null = null

  $: loading = loading_run_configs || loading_task
  $: error = loading_run_configs_error || task_error

  // Eval scores cache and state
  let eval_scores_cache: Record<string, RunConfigEvalScoresSummary> = {}
  let eval_scores_loading: Record<string, boolean> = {}
  let eval_scores_errors: Record<string, string> = {}

  // Eval data cache (full eval objects, keyed by eval_id)
  let eval_data_cache: Record<string, Eval> = {}
  let eval_templates_loading: Record<string, boolean> = {}
  let eval_templates_errors: Record<string, string> = {}

  // Track if we're initializing from URL to avoid updating URL during initial load
  let isInitializing = true

  // Initialize basic state from URL parameters (columns only)
  function initializeFromURL() {
    const urlParams = new URLSearchParams($page.url.search)

    // Get columns from URL
    const urlColumns = urlParams.get("columns")
    if (urlColumns) {
      const parsedColumns = parseInt(urlColumns, 10)
      if (parsedColumns >= 2 && parsedColumns <= MAX_COLUMNS) {
        columns = parsedColumns
      }
    }

    // Initialize selectedModels array with correct length
    selectedModels = new Array(columns).fill(null)

    // Hidden evals can be restored before run configs are loaded - they are just IDs.
    // Defensive: drop the cost section ID + dedupe in case a hand-edited URL is messy.
    const urlHidden = urlParams.get("hidden_evals")
    if (urlHidden) {
      hiddenEvalIds = [
        ...new Set(
          urlHidden
            .split(",")
            .map((id) => id.trim())
            .filter((id) => id.length > 0 && id !== "kiln_cost_section"),
        ),
      ]
    }
  }

  // Restore model selections from URL after data is loaded
  function restoreStateFromURL() {
    if (!current_task_run_configs) return

    const urlParams = new URLSearchParams($page.url.search)
    const urlModels = urlParams.get("models")

    if (urlModels) {
      const modelIds = urlModels.split(",").map((id) => (id === "" ? null : id))

      // Validate each model ID exists in task_run_configs
      for (let i = 0; i < Math.min(modelIds.length, columns); i++) {
        const modelId = modelIds[i]
        if (
          modelId &&
          current_task_run_configs.find((config) => config.id === modelId)
        ) {
          selectedModels[i] = modelId
        }
      }

      // Trigger reactivity
      selectedModels = [...selectedModels]
    }
  }

  // Update URL with current state
  function updateURL() {
    if (isInitializing) return

    const urlParams = new URLSearchParams($page.url.search)

    // Update columns
    urlParams.set("columns", columns.toString())

    // Update models (empty string for null values)
    const modelIds = selectedModels.map((id) =>
      id && id !== "__create_new_run_config__" ? id : "",
    )
    urlParams.set("models", modelIds.join(","))

    // Update hidden evals (omit param when none are hidden to keep URL clean)
    if (hiddenEvalIds.length > 0) {
      urlParams.set("hidden_evals", hiddenEvalIds.join(","))
    } else {
      urlParams.delete("hidden_evals")
    }

    // Use replace to avoid creating new history entries
    const newURL = `${$page.url.pathname}?${urlParams.toString()}`
    goto(newURL, { replaceState: true })
  }

  // Reactive statements to update URL when state changes
  $: if (!isInitializing && (columns || selectedModels || hiddenEvalIds)) {
    updateURL()
  }

  onMount(async () => {
    // Wait for page params to load
    await tick()

    // Initialize basic state from URL first (columns)
    initializeFromURL()

    // Load data needed for the page
    await Promise.all([
      load_model_info(),
      load_task_prompts(project_id, task_id),
      load_available_models(),
      get_task(),
    ])
    await get_task_run_configs()

    // Now that data is loaded, restore full state from URL
    restoreStateFromURL()

    // Mark initialization as complete
    isInitializing = false
  })

  async function get_task_run_configs() {
    loading_run_configs = true
    try {
      await load_task_run_configs(project_id, task_id)
    } catch (err) {
      loading_run_configs_error = createKilnError(err)
    } finally {
      loading_run_configs = false
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

  async function fetch_eval_scores(run_config_id: string) {
    if (
      eval_scores_cache[run_config_id] ||
      eval_scores_loading[run_config_id] ||
      eval_scores_errors[run_config_id]
    ) {
      return // Already cached, loading, or errored
    }

    try {
      eval_scores_loading[run_config_id] = true
      const { data, error: fetch_error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/run_configs/{run_config_id}/eval_scores",
        {
          params: {
            path: {
              project_id,
              task_id,
              run_config_id,
            },
          },
        },
      )
      if (fetch_error) {
        throw fetch_error
      }
      eval_scores_cache[run_config_id] = data
      delete eval_scores_errors[run_config_id]
    } catch (err) {
      const kilnError = createKilnError(err)
      eval_scores_errors[run_config_id] =
        kilnError.getMessage() || "Failed to fetch eval scores"
    } finally {
      eval_scores_loading[run_config_id] = false
    }
  }

  // Reactively fetch eval scores when models are selected (for table)
  $: {
    selectedModels.forEach((modelId) => {
      if (
        modelId &&
        modelId !== "__create_new_run_config__" &&
        !eval_scores_cache[modelId] &&
        !eval_scores_loading[modelId] &&
        !eval_scores_errors[modelId]
      ) {
        fetch_eval_scores(modelId)
      }
    })
  }

  // Fetch eval scores for ALL run configs (for chart)
  $: if (current_task_run_configs) {
    current_task_run_configs.forEach((config) => {
      if (
        config.id &&
        !eval_scores_cache[config.id] &&
        !eval_scores_loading[config.id] &&
        !eval_scores_errors[config.id]
      ) {
        fetch_eval_scores(config.id)
      }
    })
  }

  // Check if chart data is still loading
  $: chartLoading =
    !current_task_run_configs ||
    current_task_run_configs.some(
      (config) =>
        config.id &&
        !eval_scores_cache[config.id] &&
        !eval_scores_errors[config.id],
    )

  async function fetch_eval_data(eval_id: string) {
    if (
      eval_data_cache[eval_id] !== undefined ||
      eval_templates_loading[eval_id] ||
      eval_templates_errors[eval_id]
    ) {
      return // Already cached, loading, or errored
    }

    try {
      eval_templates_loading[eval_id] = true
      const { data, error: fetch_error } = await client.GET(
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
      if (fetch_error) {
        throw fetch_error
      }
      eval_data_cache[eval_id] = data
      delete eval_templates_errors[eval_id]
    } catch (err) {
      const kilnError = createKilnError(err)
      eval_templates_errors[eval_id] =
        kilnError.getMessage() || "Failed to fetch eval template"
    } finally {
      eval_templates_loading[eval_id] = false
    }
  }

  // Generate comparison features dynamically from eval_scores
  $: comparisonFeatures = generateComparisonFeatures(
    selectedModels,
    eval_scores_cache,
  )

  // Generate comparison features for ALL run configs (for chart)
  $: allRunConfigIds = (current_task_run_configs || [])
    .map((c) => c.id)
    .filter((id): id is string => id !== null && id !== undefined)
  $: chartComparisonFeatures = generateComparisonFeatures(
    allRunConfigIds,
    eval_scores_cache,
  )

  // Filter out user-hidden evals (cost section is never hideable). hiddenEvalIds is
  // passed in as a parameter (rather than read via closure) so that Svelte's reactive
  // `$:` statements track it as a dependency and re-run when it changes.
  function filterVisibleFeatures<T extends { eval_id: string }>(
    features: T[],
    hidden: string[],
  ): T[] {
    if (hidden.length === 0) return features
    return features.filter(
      (section) =>
        section.eval_id === "kiln_cost_section" ||
        !hidden.includes(section.eval_id),
    )
  }

  $: visibleComparisonFeatures = filterVisibleFeatures(
    comparisonFeatures,
    hiddenEvalIds,
  )
  $: visibleChartComparisonFeatures = filterVisibleFeatures(
    chartComparisonFeatures,
    hiddenEvalIds,
  )

  // Names of currently-hidden evals (used for the "show hidden" dropdown).
  // chartComparisonFeatures is built from ALL run configs for the task and is a
  // superset of comparisonFeatures (which only covers selected models), so it
  // alone is enough to resolve display names.
  $: hiddenEvalsInfo = hiddenEvalIds
    .filter((id) => id !== "kiln_cost_section")
    .map((evalId) => {
      const feature = chartComparisonFeatures.find((s) => s.eval_id === evalId)
      return { eval_id: evalId, category: feature?.category ?? "Unknown eval" }
    })

  function hideEval(evalId: string) {
    if (evalId === "kiln_cost_section") return
    if (hiddenEvalIds.includes(evalId)) return
    hiddenEvalIds = [...hiddenEvalIds, evalId]
  }

  function showEval(evalId: string) {
    hiddenEvalIds = hiddenEvalIds.filter((id) => id !== evalId)
  }

  function showAllHiddenEvals() {
    hiddenEvalIds = []
  }

  $: hiddenEvalsMenuItems = [
    { label: "Show Eval", header: true },
    ...hiddenEvalsInfo.map(
      (info): FloatingMenuItem => ({
        label: info.category,
        onclick: () => showEval(info.eval_id),
      }),
    ),
    ...(hiddenEvalsInfo.length > 1
      ? [{ label: "Show All", onclick: showAllHiddenEvals }]
      : []),
  ] as FloatingMenuItem[]

  // Reactively fetch eval templates for sections
  $: {
    comparisonFeatures.forEach((section) => {
      if (section.eval_id !== "kiln_cost_section") {
        fetch_eval_data(section.eval_id)
      }
    })
  }

  function generateComparisonFeatures(
    models: (string | null)[],
    scores_cache: Record<string, RunConfigEvalScoresSummary>,
  ) {
    const features: {
      category: string
      items: { label: string; key: string }[]
      has_default_eval_config: boolean | undefined
      eval_id: string
      spec_id: string | null
    }[] = []
    const evalCategories: Record<string, Set<string>> = {}
    const hasDefaultEvalConfig: Record<string, boolean> = {}
    const evalNames: Record<string, string> = {}
    const specIds: Record<string, string | null> = {}

    // Collect all evals and their scores from selected models
    models.forEach((modelId) => {
      if (!modelId || !scores_cache[modelId]) return

      const evalScores = scores_cache[modelId]
      evalScores.eval_results.forEach((evalResult) => {
        const evalId = evalResult.eval_id || ""
        hasDefaultEvalConfig[evalId] = !evalResult.missing_default_eval_config
        evalNames[evalId] = evalResult.eval_name
        specIds[evalId] = evalResult.spec_id || null

        if (!evalCategories[evalId]) {
          evalCategories[evalId] = new Set()
        }

        Object.keys(evalResult.eval_config_result?.results || {}).forEach(
          (scoreKey) => {
            evalCategories[evalId].add(scoreKey)
          },
        )
      })
    })

    // Convert to comparison features format
    Object.entries(evalCategories).forEach(([evalId, scoreKeys]) => {
      const items = Array.from(scoreKeys).map((scoreKey) => ({
        label: scoreKey
          .replace(/_/g, " ")
          .replace(/\b\w/g, (l) => l.toUpperCase()),
        key: `${evalId}::${scoreKey}`,
      }))

      features.push({
        category: evalNames[evalId] || evalId,
        items,
        has_default_eval_config: hasDefaultEvalConfig[evalId],
        eval_id: evalId,
        spec_id: specIds[evalId],
      })
    })

    // Add Cost section (always last)
    const costItems = [
      { label: "Input Tokens", key: "cost::mean_input_tokens" },
      { label: "Output Tokens", key: "cost::mean_output_tokens" },
      { label: "Total Tokens", key: "cost::mean_total_tokens" },
      { label: "Cost (USD)", key: "cost::mean_cost" },
      { label: "Latency", key: "cost::mean_total_llm_latency_ms" },
    ]

    features.push({
      category: "Average Usage, Cost & Latency",
      items: costItems,
      has_default_eval_config: undefined,
      eval_id: "kiln_cost_section",
      spec_id: null,
    })

    return features
  }

  // Generate dropdown options from run configs
  $: current_task_run_configs =
    $run_configs_by_task_composite_id[
      get_task_composite_id(project_id, task_id)
    ] || null

  let target_new_run_config_col: number | null = null
  let create_new_run_config_dialog: CreateNewRunConfigDialog | null = null
  $: if (selectedModels.includes("__create_new_run_config__")) {
    target_new_run_config_col = selectedModels.indexOf(
      "__create_new_run_config__",
    )
    create_new_run_config_dialog?.show()
  }

  const MAX_COLUMNS = 10
  function addColumn() {
    if (columns < MAX_COLUMNS) {
      columns++
      selectedModels = [...selectedModels, null]
    }
  }

  function removeColumn(index: number) {
    if (columns > 2) {
      columns--
      selectedModels = selectedModels.filter((_, i) => i !== index)
    }
  }

  // Core value extraction - returns raw number, used by both table and chart
  function getModelValueRaw(
    modelKey: string | null,
    dataKey: string,
  ): number | null {
    if (!modelKey || !eval_scores_cache[modelKey]) return null

    const [category, scoreKey] = dataKey.split("::")
    if (!category || !scoreKey) return null

    const evalScores = eval_scores_cache[modelKey]

    // Handle cost metrics
    if (category === "cost") {
      if (!evalScores.mean_usage) return null

      const meanUsage = evalScores.mean_usage
      switch (scoreKey) {
        case "mean_input_tokens":
          return meanUsage.mean_input_tokens ?? null
        case "mean_output_tokens":
          return meanUsage.mean_output_tokens ?? null
        case "mean_total_tokens":
          return meanUsage.mean_total_tokens ?? null
        case "mean_cost":
          return meanUsage.mean_cost ?? null
        case "mean_total_llm_latency_ms":
          return meanUsage.mean_total_llm_latency_ms ?? null
      }
      return null
    }

    // Handle eval metrics (category is eval_id)
    const evalResult = evalScores.eval_results.find(
      (e) => e.eval_id === category,
    )
    if (!evalResult) return null

    const score: ScoreSummary | null | undefined =
      evalResult.eval_config_result?.results[scoreKey]
    if (score) {
      return score.mean_score
    }

    return null
  }

  // Formatted value for table display
  function getModelValue(modelKey: string | null, dataKey: string): string {
    const value = getModelValueRaw(modelKey, dataKey)
    if (value === null) return "—"

    const [category, scoreKey] = dataKey.split("::")

    // Format cost with currency symbol, latency with ms/s, tokens as whole numbers, others as decimals
    if (category === "cost") {
      if (scoreKey === "mean_cost") {
        return `$${value.toFixed(7)}`
      } else if (scoreKey === "mean_total_llm_latency_ms") {
        return formatLatency(value)
      } else {
        return value.toFixed(1)
      }
    }

    return value.toFixed(2)
  }

  function getModelExcluded(
    modelKey: string | null,
    evalID: string | null,
  ): { n_excluded: number; n_used: number } {
    if (
      evalID === "kiln_cost_section" ||
      !modelKey ||
      !eval_scores_cache[modelKey]
    )
      return { n_excluded: 0, n_used: 0 }

    const evalScores = eval_scores_cache[modelKey]
    const evalResult = evalScores.eval_results.find((e) => e.eval_id === evalID)
    if (!evalResult?.eval_config_result) return { n_excluded: 0, n_used: 0 }

    // Read both n_excluded and n_used from the first available ScoreSummary
    // to keep the data source consistent (same approach as run_config_comparison_table)
    const results = evalResult.eval_config_result.results
    for (const key of Object.keys(results)) {
      const ss = results[key]
      if (ss) return { n_excluded: ss.n_excluded ?? 0, n_used: ss.n_used ?? 0 }
    }
    return { n_excluded: 0, n_used: 0 }
  }

  function getModelPercentComplete(
    modelKey: string | null,
    evalID: string | null,
  ): number {
    if (evalID === "kiln_cost_section") return 1.0
    if (!modelKey || !eval_scores_cache[modelKey]) return 0.0

    const evalScores = eval_scores_cache[modelKey]
    const evalResult = evalScores.eval_results.find((e) => e.eval_id === evalID)

    if (!evalResult) return 0.0

    return evalResult.eval_config_result?.percent_complete || 0.0
  }

  function getSelectedRunConfig(modelKey: string | null): TaskRunConfig | null {
    if (!modelKey || !current_task_run_configs) return null
    return (
      current_task_run_configs.find((config) => config.id === modelKey) || null
    )
  }

  function getModelDefaultEvalConfigID(
    modelKey: string | null,
    evalID: string | null,
  ): string | null | undefined {
    if (evalID === "kiln_cost_section") return null
    if (!modelKey || !eval_scores_cache[modelKey]) return null

    const evalScores = eval_scores_cache[modelKey]
    const evalResult = evalScores.eval_results.find((e) => e.eval_id === evalID)

    if (!evalResult) return null

    return evalResult.eval_config_result?.eval_config_id
  }

  function getEvalDatasetSize(evalID: string | null): number {
    if (!evalID) return 0
    for (const modelKey of selectedModels) {
      if (!modelKey || !eval_scores_cache[modelKey]) continue
      const evalResult = eval_scores_cache[modelKey].eval_results.find(
        (e) => e.eval_id === evalID,
      )
      if (evalResult) return evalResult.dataset_size
    }
    return 0
  }

  function navigateToAddData(eval_id: string) {
    const evalData = eval_data_cache[eval_id]
    if (!evalData) return

    // Same refusal the eval detail page and the data-gen dialog make: this flow adds
    // TaskRuns, and an EvalInput-backed test dataset can't receive them.
    if (eval_split(evalData, "test")?.source === "eval_input") {
      alert(
        "This eval uses our new eval dataset format, which can't be generated from this UI.",
      )
      return
    }

    // Adding data means adding TaskRuns, so only a TaskRun-backed test split has a
    // tag to add them under.
    const test_filter_id = task_run_split_filter_id(evalData, "test")
    const test_tag = test_filter_id
      ? tagFromFilterId(test_filter_id)
      : undefined
    const golden_tag = evalData.eval_configs_filter_id
      ? tagFromFilterId(evalData.eval_configs_filter_id)
      : undefined

    // Refuse before navigating, matching the eval detail page's identical guard. Without
    // it, a missing tag falls through to a navigation with no `splits` param at all, and
    // the user adds rows that silently never join the eval's set. The golden half is
    // reachable in ordinary use: a non-rag eval with no golden set is the expected V2
    // state (functional spec 6.1), and this button appears precisely when the eval's
    // test split is empty.
    if (!test_tag || (evalData.template !== "rag" && !golden_tag)) {
      alert(
        "No test or golden dataset tag found. If you're using a custom filter, please setup the dataset manually.",
      )
      return
    }

    const params = new URLSearchParams()
    params.set("reason", "eval")
    if (evalData.template) {
      params.set("template_id", evalData.template)
    }
    params.set("eval_id", `${project_id}::${task_id}::${eval_id}`)

    const spec_id =
      comparisonFeatures.find((s) => s.eval_id === eval_id)?.spec_id ?? "legacy"
    params.set(
      "eval_link",
      `/specs/${project_id}/${task_id}/${spec_id}/${eval_id}`,
    )

    // Every entry into this flow allocates generated data the same way, so the eval's
    // splits decide the allocation rather than which button reached here.
    const splits_param = build_eval_generation_splits_param(evalData)
    if (splits_param) {
      params.set("splits", splits_param)
    }

    if (evalData.template === "tool_call") {
      const tool_id = evalData.template_properties?.tool_id
      if (tool_id) {
        params.set("tool_id", String(tool_id))
      }
    }

    goto(`/dataset/${project_id}/${task_id}/add_data?${params.toString()}`)
  }

  function getPercentageDifferenceRaw(
    base: number | null,
    compare: number | null,
  ): string {
    if (base === null || compare === null) return ""

    // Handle division by zero
    if (base === 0) {
      if (compare === 0) return "even"
      return "N/A"
    }

    const percentDiff = ((compare - base) / base) * 100

    // Handle very small differences (less than 0.01%)
    if (Math.abs(percentDiff) < 0.01) return "even"

    // Format the percentage
    const formatted =
      Math.abs(percentDiff) < 1
        ? percentDiff.toFixed(2)
        : Math.abs(percentDiff) < 10
          ? percentDiff.toFixed(1)
          : percentDiff.toFixed(0)

    return percentDiff >= 0 ? `+${formatted}%` : `${formatted}%`
  }

  // Reactive valid selected models - must be reactive ($:) for template to update
  $: validSelectedModels = selectedModels.filter(
    (m): m is string => m !== null && m !== "__create_new_run_config__",
  )

  $: agentInfo.set({
    name: "Compare Evals",
    description: `Compare evals for project ID ${project_id}, task ID ${task_id}. Side-by-side comparison of eval results. ${validSelectedModels.length > 0 ? `The following run configs are selected: ${validSelectedModels.join(", ")}.` : "No run configs are selected."}`,
  })

  $: allSelectedLoading = validSelectedModels.every(
    (modelId) =>
      eval_scores_loading[modelId] ||
      (!eval_scores_cache[modelId] && !eval_scores_errors[modelId]),
  )

  $: anyLoadedData = validSelectedModels.some(
    (modelId) => eval_scores_cache[modelId],
  )

  function navigateToEvalPage(spec_id: string | null, eval_id: string) {
    if (!spec_id) return
    const template = eval_data_cache[eval_id]?.template ?? null
    if (template === "rag") {
      goto(
        `/specs/${project_id}/${task_id}/${spec_id}/${eval_id}/compare_run_configs`,
      )
    } else {
      goto(`/specs/${project_id}/${task_id}/${spec_id}/${eval_id}/eval_configs`)
    }
  }
</script>

<AppPage
  title="Compare Run Configurations"
  subtitle="Find the optimal run configuration for your task using evals"
  {breadcrumbs}
>
  {#if loading}
    <div class="w-full min-h-[50vh] flex justify-center items-center">
      <div class="loading loading-spinner loading-lg"></div>
    </div>
  {:else if error}
    <div
      class="w-full min-h-[50vh] flex flex-col justify-center items-center gap-2"
    >
      <div class="font-medium">Error Loading Run Configurations</div>
      <div class="text-error text-sm">
        {error.getMessage() || "An unknown error occurred"}
      </div>
    </div>
  {:else}
    <div class="max-w-[1900px] mx-auto">
      {#if allSelectedLoading && !anyLoadedData && validSelectedModels.length > 0}
        <!-- Big centered loading spinner when no data is loaded yet -->
        <div
          class="bg-white border border-gray-200 rounded-lg p-12 flex flex-col items-center justify-center min-h-[400px]"
        >
          <div class="loading loading-spinner loading-lg mb-4"></div>
          <div class="text-gray-600">Loading evaluation scores...</div>
        </div>
      {:else}
        <!-- Table action buttons - positioned above table on the right -->
        <div class="flex justify-end gap-2 mb-4">
          {#if hiddenEvalsInfo.length > 0}
            <div class="hidden-evals-dropdown">
              <FloatingMenu items={hiddenEvalsMenuItems} width="w-72">
                <button
                  slot="trigger"
                  type="button"
                  class="btn btn-sm btn-outline"
                >
                  Hidden Evals ({hiddenEvalsInfo.length})
                </button>
              </FloatingMenu>
            </div>
          {/if}
          {#if columns < MAX_COLUMNS}
            <button
              on:click={addColumn}
              class="btn btn-sm btn-outline gap-2"
              title="Add comparison column"
            >
              <svg
                class="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                />
              </svg>
              Add Column
            </button>
          {/if}
        </div>

        <div class="bg-white border border-gray-200 rounded-lg overflow-hidden">
          <!-- Model Selection Header Row -->
          <div
            class="grid border-b border-gray-200 bg-gray-50"
            style="grid-template-columns: 200px repeat({columns}, 1fr);"
          >
            <div class="px-6 py-4 font-semibold text-gray-900"></div>
            {#each Array(columns) as _, i}
              <div class="px-6 py-4 relative min-w-0">
                {#if task}
                  <div class="flex items-center gap-1 xl:flex-row">
                    {#if columns > 2}
                      <button
                        on:click={() => removeColumn(i)}
                        class="w-6 h-6 rounded-full flex items-center justify-center hover:bg-gray-200 transition-colors flex-shrink-0"
                        title="Remove column"
                      >
                        ✕
                      </button>
                    {/if}
                    <div class="flex-1 min-w-0">
                      <SavedRunConfigurationsDropdown
                        title=""
                        {project_id}
                        current_task={task}
                        bind:selected_run_config_id={selectedModels[i]}
                        run_page={false}
                        auto_select_default={i === 0}
                      />
                    </div>
                  </div>
                {/if}

                <!-- Show selected model info below dropdown -->
                {#if selectedModels[i]}
                  {@const selectedConfig = getSelectedRunConfig(
                    selectedModels[i],
                  )}
                  {#if selectedConfig}
                    {@const current_task_prompts =
                      $prompts_by_task_composite_id[
                        get_task_composite_id(project_id, task_id)
                      ] || null}
                    {@const prompt_info_text = getRunConfigPromptInfoText(
                      selectedConfig,
                      current_task_prompts,
                    )}
                    {@const config_prompt_id = isKilnAgentRunConfig(
                      selectedConfig.run_config_properties,
                    )
                      ? selectedConfig.run_config_properties.prompt_id
                      : undefined}
                    {@const prompt_link_url = config_prompt_id
                      ? prompt_link(project_id, task_id, config_prompt_id)
                      : undefined}
                    {@const prompt_display_name = getRunConfigPromptDisplayName(
                      selectedConfig,
                      current_task_prompts,
                    )}
                    {@const transformLabel =
                      getRunConfigInputTransformSummaryLabel(selectedConfig)}
                    <div class="mt-3 text-center">
                      <div class="font-semibold text-gray-900 text-sm">
                        {#if isMcpRunConfig(selectedConfig.run_config_properties)}
                          {selectedConfig.run_config_properties.tool_reference
                            .tool_name ?? "MCP Tool"}
                        {:else}
                          {getRunConfigModelDisplayName(
                            selectedConfig,
                            $model_info,
                          ) ?? "Unknown Model"}
                        {/if}
                      </div>
                      {#if prompt_display_name}
                        <div class="text-xs text-gray-500 font-normal mt-1">
                          {#if prompt_link_url}
                            Prompt: <a
                              href={prompt_link_url}
                              class="text-gray-500 font-normal link"
                            >
                              {prompt_display_name}
                            </a>
                          {:else}
                            Prompt: {prompt_display_name}
                          {/if}
                          {#if prompt_info_text}
                            <InfoTooltip
                              tooltip_text={prompt_info_text}
                              position="bottom"
                              no_pad={true}
                            />
                          {/if}
                        </div>
                      {/if}
                      {#if transformLabel}
                        <div class="text-xs text-gray-500 font-normal mt-1">
                          Input Transform: {transformLabel}
                        </div>
                      {/if}
                    </div>
                  {/if}
                {/if}
              </div>
            {/each}
          </div>

          <!-- Comparison Data - only show if models are selected -->
          {#if validSelectedModels.length > 0}
            {#each visibleComparisonFeatures as section}
              <!-- Section Header -->
              <div
                class="bg-gray-50 px-6 py-3 border-b border-gray-200 flex items-center justify-between gap-2"
              >
                <h4
                  class="text-sm font-semibold text-gray-900 uppercase tracking-wide"
                >
                  {section.category}
                </h4>
                {#if section.eval_id !== "kiln_cost_section"}
                  <button
                    on:click={() => hideEval(section.eval_id)}
                    class="w-6 h-6 rounded-full flex items-center justify-center text-gray-500 hover:bg-gray-200 hover:text-gray-900 transition-colors"
                    title="Hide this eval"
                  >
                    ✕
                  </button>
                {/if}
              </div>

              {#if section.items.length == 0}
                <div
                  class="grid gap-4 border-b border-gray-100 last:border-b-0"
                  style="grid-template-columns: 200px repeat(1, 1fr);"
                >
                  <!-- Empty section for visual consistency -->
                  <div
                    class="px-6 py-4 bg-gray-50 font-medium text-gray-700 flex items-center"
                  ></div>
                  <div class="px-6 py-4 text-center">
                    {#if section.has_default_eval_config === false}
                      <div>Select a default eval config to compare scores.</div>
                      {#if eval_templates_loading[section.eval_id]}
                        <div class="mt-2">
                          <div class="loading loading-spinner loading-xs"></div>
                        </div>
                      {:else if eval_templates_errors[section.eval_id]}
                        <div class="mt-2 text-error text-xs">
                          {eval_templates_errors[section.eval_id]}
                        </div>
                      {:else if eval_data_cache[section.eval_id] !== undefined}
                        <button
                          on:click={() =>
                            navigateToEvalPage(
                              section.spec_id ?? "legacy",
                              section.eval_id,
                            )}
                          class="btn btn-xs rounded-full mt-2"
                        >
                          Manage Eval Configs
                        </button>
                      {/if}
                    {:else}
                      Unknown issue - no scores found
                    {/if}
                  </div>
                </div>
              {/if}

              <!-- Section Rows -->
              {#if section.items.length > 0}
                <div
                  class="grid"
                  style="grid-template-columns: 200px repeat({columns}, 1fr);"
                >
                  {#each section.items as item, item_index}
                    <!-- Feature Label -->
                    <div
                      class="px-6 py-4 bg-gray-50 font-medium text-gray-700 flex items-center border-b border-gray-100"
                    >
                      {item.label}
                    </div>

                    <!-- Model Values -->
                    {#each Array(columns) as _, i}
                      {@const loading =
                        selectedModels[i] &&
                        eval_scores_loading[selectedModels[i]]}
                      {@const error =
                        selectedModels[i] &&
                        eval_scores_errors[selectedModels[i]]}
                      {@const percentComplete =
                        (selectedModels[i] &&
                          getModelPercentComplete(
                            selectedModels[i],
                            section.eval_id,
                          )) ||
                        0.0}
                      {#if selectedModels[i] && (loading || error || percentComplete < 1.0)}
                        <!-- These cells merge vertically for error states -->
                        {#if item_index === 0}
                          <div
                            class="px-6 py-4 text-center flex items-center justify-center border-b border-gray-100"
                            style="grid-row: span {section.items.length};"
                          >
                            {#if loading}
                              <!-- Column loading spinner -->
                              <div
                                class="loading loading-spinner loading-sm"
                              ></div>
                            {:else if error}
                              <!-- Error state -->
                              <span class="text-error text-sm">Error</span>
                            {:else if percentComplete < 1.0}
                              {@const runConfigId = selectedModels[i]}
                              {@const defaultEvalConfigId =
                                getModelDefaultEvalConfigID(
                                  runConfigId,
                                  section.eval_id,
                                )}
                              {@const incomplete_excluded = getModelExcluded(
                                selectedModels[i],
                                section.eval_id,
                              )}
                              <div class="flex flex-col items-center gap-1">
                                <div class="text-warning text-sm font-medium">
                                  Eval Incomplete
                                </div>
                                {#if incomplete_excluded.n_excluded > 0}
                                  {@const incomplete_ratio =
                                    incomplete_excluded.n_excluded /
                                    (incomplete_excluded.n_used +
                                      incomplete_excluded.n_excluded)}
                                  <span
                                    class={incomplete_ratio > 0.2
                                      ? "text-error"
                                      : "text-warning"}
                                  >
                                    <InfoTooltip
                                      symbol="info"
                                      position="top"
                                      tooltip_text={`${incomplete_excluded.n_excluded} of ${incomplete_excluded.n_used + incomplete_excluded.n_excluded} cases were skipped and are not reflected in this score.`}
                                    />
                                  </span>
                                {/if}
                                <div class="text-left">
                                  {#if getEvalDatasetSize(section.eval_id) === 0}
                                    {#if eval_split(eval_data_cache[section.eval_id], "test")?.source === "eval_input"}
                                      <!-- EvalInput-typed slice: data is minted by the
                                        eval builder; the add-data flow tags TaskRuns,
                                        which doesn't apply. -->
                                      <div class="text-xs text-gray-500">
                                        No eval data. This eval's data is
                                        created by the eval builder.
                                      </div>
                                    {:else}
                                      <button
                                        class="btn btn-xs mt-1"
                                        on:click={() =>
                                          navigateToAddData(section.eval_id)}
                                      >
                                        Add Eval Data
                                      </button>
                                    {/if}
                                  {:else if defaultEvalConfigId && runConfigId}
                                    <RunEval
                                      eval_id={section.eval_id}
                                      run_config_ids={[runConfigId]}
                                      {project_id}
                                      {task_id}
                                      current_eval_config_id={defaultEvalConfigId}
                                      eval_type="run_config"
                                      btn_size="xs"
                                      btn_primary={false}
                                      on_run_complete={() => {
                                        if (runConfigId) {
                                          delete eval_scores_cache[runConfigId]
                                          delete eval_scores_errors[runConfigId]
                                          fetch_eval_scores(runConfigId)
                                        }
                                      }}
                                    />
                                  {:else}
                                    <div class="text-xs text-gray-500">
                                      Select a default judge to run evals
                                    </div>
                                    {#if section.spec_id}
                                      <button
                                        class="btn btn-xs mt-1"
                                        on:click={() =>
                                          navigateToEvalPage(
                                            section.spec_id,
                                            section.eval_id,
                                          )}
                                      >
                                        Select Judge
                                      </button>
                                    {/if}
                                  {/if}
                                </div>
                              </div>
                            {/if}
                          </div>
                        {/if}
                      {:else}
                        {@const excluded_info =
                          item_index === 0
                            ? getModelExcluded(
                                selectedModels[i],
                                section.eval_id,
                              )
                            : null}
                        <div
                          class="px-6 py-4 text-center flex items-center justify-center border-b border-gray-100"
                        >
                          <!-- Normal value -->
                          <div class="flex flex-col items-center">
                            <span class="text-gray-900">
                              {getModelValue(selectedModels[i], item.key)}
                              {#if excluded_info && excluded_info.n_excluded > 0}
                                {@const ratio =
                                  excluded_info.n_excluded /
                                  (excluded_info.n_used +
                                    excluded_info.n_excluded)}
                                <span
                                  class={ratio > 0.2
                                    ? "text-error"
                                    : "text-warning"}
                                >
                                  <InfoTooltip
                                    symbol="info"
                                    position="top"
                                    tooltip_text={`${excluded_info.n_excluded} of ${excluded_info.n_used + excluded_info.n_excluded} cases were skipped and are not reflected in this score.`}
                                  />
                                </span>
                              {/if}
                            </span>
                            {#if i > 0 && selectedModels[0] !== null}
                              {@const baseRaw = getModelValueRaw(
                                selectedModels[0],
                                item.key,
                              )}
                              {@const currentRaw = getModelValueRaw(
                                selectedModels[i],
                                item.key,
                              )}
                              {@const percentDiff = getPercentageDifferenceRaw(
                                baseRaw,
                                currentRaw,
                              )}
                              {#if percentDiff}
                                <span class="text-xs text-gray-500 mt-1">
                                  {percentDiff}
                                </span>
                              {/if}
                            {/if}
                          </div>
                        </div>
                      {/if}
                    {/each}
                  {/each}
                </div>
              {/if}
            {/each}
          {:else}
            <!-- Empty state message within the table -->
            <div class="px-6 py-12 text-center text-gray-500">
              <svg
                class="mx-auto h-12 w-24 text-gray-400 mb-4"
                viewBox="0 0 1730 800"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M140 400C140 283.438 140 225.159 158.148 180.638C174.11 141.477 199.582 109.638 230.911 89.6844C266.527 67 313.151 67 406.4 67H539.6C632.85 67 679.473 67 715.091 89.6844C746.42 109.638 771.891 141.477 787.852 180.638C806 225.159 806 283.438 806 400C806 516.562 806 574.842 787.852 619.364C771.891 658.525 746.42 690.364 715.091 710.314C679.473 733 632.85 733 539.6 733H406.4C313.151 733 266.527 733 230.911 710.314C199.582 690.364 174.11 658.525 158.148 619.364C140 574.842 140 516.562 140 400Z"
                  stroke="currentColor"
                  stroke-width="50"
                />
                <path
                  d="M924 400C924 283.438 924 225.159 942.148 180.638C958.11 141.477 983.582 109.638 1014.91 89.6844C1050.53 67 1097.15 67 1190.4 67H1323.6C1416.85 67 1463.47 67 1499.09 89.6844C1530.42 109.638 1555.89 141.477 1571.85 180.638C1590 225.159 1590 283.438 1590 400C1590 516.562 1590 574.842 1571.85 619.364C1555.89 658.525 1530.42 690.364 1499.09 710.314C1463.47 733 1416.85 733 1323.6 733H1190.4C1097.15 733 1050.53 733 1014.91 710.314C983.582 690.364 958.11 658.525 942.148 619.364C924 574.842 924 516.562 924 400Z"
                  stroke="currentColor"
                  stroke-width="50"
                  stroke-dasharray="100 100"
                />
              </svg>
              <div class="text-lg font-medium text-gray-900 mb-2">
                Select run configurations to compare
              </div>
              <div class="text-gray-500">
                Choose run configurations from the dropdowns above to see a
                detailed comparison
              </div>
            </div>
          {/if}
        </div>

        {#if validSelectedModels.length > 0}
          <div class="mt-16">
            <CompareRadarChart
              comparisonFeatures={visibleComparisonFeatures}
              {getModelValueRaw}
              run_configs={current_task_run_configs || []}
              model_info={$model_info}
              selectedRunConfigIds={validSelectedModels}
              prompts={$prompts_by_task_composite_id[
                get_task_composite_id(project_id, task_id)
              ] || null}
            />
          </div>
        {/if}

        <div class="mt-16">
          <CompareChart
            comparisonFeatures={visibleChartComparisonFeatures}
            {getModelValueRaw}
            run_configs={current_task_run_configs || []}
            model_info={$model_info}
            prompts={$prompts_by_task_composite_id[
              get_task_composite_id(project_id, task_id)
            ] || null}
            loading={loading || chartLoading}
          />
        </div>
      {/if}
    </div>
  {/if}
</AppPage>

<CreateNewRunConfigDialog
  bind:this={create_new_run_config_dialog}
  {project_id}
  {task}
  new_run_config_created={(run_config) => {
    if (target_new_run_config_col !== null) {
      if (target_new_run_config_col < columns) {
        selectedModels[target_new_run_config_col] = run_config.id || null
        selectedModels = [...selectedModels] // Trigger reactivity
      }
      target_new_run_config_col = null
    }
  }}
  on:close={() => {
    if (target_new_run_config_col !== null) {
      if (target_new_run_config_col < columns) {
        selectedModels[target_new_run_config_col] = null
        selectedModels = [...selectedModels]
      }
      target_new_run_config_col = null // Trigger reactivity
    }
  }}
/>

<style>
  .hidden-evals-dropdown :global(ul.menu li > button),
  .hidden-evals-dropdown :global(ul.menu li > a) {
    font-size: 0.875rem;
    font-weight: 500;
    color: rgb(17 24 39);
  }

  /* Render a gray "+" prefix on eval rows only. Excludes the header
     (first-child) and the "Restore All" footer (last-child at position 4+). */
  .hidden-evals-dropdown
    :global(
      ul.menu
        li:not(:first-child):not(:last-child:nth-child(n + 4))
        > button::before
    ) {
    content: "+";
    color: rgb(107 114 128);
    margin-right: 0.375rem;
    font-weight: 400;
  }

  /* "Restore All" footer styling — gray-500. */
  .hidden-evals-dropdown
    :global(ul.menu li:last-child:nth-child(n + 4) > button) {
    color: rgb(107 114 128);
  }

  /* Divider before the "Show all hidden" footer. nth-child(n+4) ensures we
     only render it when the list has header + 2+ evals + show-all footer. */
  .hidden-evals-dropdown :global(ul.menu li:last-child:nth-child(n + 4)) {
    border-top: 1px solid rgb(209 213 219);
    margin-top: 0.5rem;
    padding-top: 0.5rem;
  }
</style>
