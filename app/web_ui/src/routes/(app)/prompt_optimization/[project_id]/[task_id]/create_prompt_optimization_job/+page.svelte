<script lang="ts">
  import AppPage from "../../../../app_page.svelte"
  import FormContainer from "$lib/utils/form_container.svelte"
  import InfoTooltip from "$lib/ui/info_tooltip.svelte"
  import { page } from "$app/stores"
  import { client } from "$lib/api_client"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { formatDate } from "$lib/utils/formatters"
  import { onMount } from "svelte"
  import Completed from "$lib/ui/completed.svelte"
  import SavedRunConfigurationsDropdown from "$lib/ui/run_config_component/saved_run_configs_dropdown.svelte"
  import { isKilnAgentRunConfig } from "$lib/types"
  import type { Task, TaskRunConfig, Eval, EvalConfig } from "$lib/types"
  import {
    load_task_prompts,
    prompts_by_task_composite_id,
  } from "$lib/stores/prompts_store"
  import {
    available_tools,
    get_task_composite_id,
    load_available_models,
    load_available_tools,
    load_model_info,
    model_info,
  } from "$lib/stores"
  import {
    load_task_run_configs,
    run_configs_by_task_composite_id,
  } from "$lib/stores/run_configs_store"
  import {
    getRunConfigModelDisplayName,
    getDetailedModelNameFromParts,
    getRunConfigUiProperties,
    getRunConfigInputTransform,
  } from "$lib/utils/run_config_formatters"
  import InputTransformModal from "$lib/ui/run_config_component/input_transform_modal.svelte"
  import CreateNewRunConfigDialog from "$lib/ui/run_config_component/create_new_run_config_dialog.svelte"
  import Output from "$lib/ui/output.svelte"
  import Warning from "$lib/ui/warning.svelte"
  import { checkKilnCopilotAvailable } from "$lib/utils/copilot_utils"
  import { checkPromptOptimizationAccess } from "$lib/utils/entitlement_utils"
  import PromptOptimizationCopilotRequired from "../prompt_optimization_copilot_required.svelte"
  import EntitlementRequiredCard from "$lib/ui/kiln_copilot/entitlement_required_card.svelte"
  import PropertyList from "$lib/ui/property_list.svelte"
  import TableActionMenu from "$lib/ui/table_action_menu.svelte"
  import posthog from "posthog-js"
  import { agentInfo } from "$lib/agent"
  import { task_run_split_filter_id } from "$lib/utils/eval_splits"

  function tagFromFilterId(filter_id: string): string | undefined {
    if (filter_id.startsWith("tag::")) {
      return filter_id.replace("tag::", "")
    }
    return undefined
  }

  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  $: agentInfo.set({
    name: "Create Prompt Optimization Job",
    description: `Create a new prompt optimization job for project ID ${project_id}, task ID ${task_id}. Configure target run config and optimization parameters.`,
  })

  let target_run_config_id: string | null = null

  let create_new_run_config_dialog: CreateNewRunConfigDialog | null = null

  $: if (target_run_config_id === "__create_new_run_config__") {
    create_new_run_config_dialog?.show()
  }

  function get_prompt_text_from_id(
    prompt_id: string | null | undefined,
  ): string | null {
    if (!prompt_id || !task_prompts) {
      return null
    }

    const saved_prompt = task_prompts.prompts.find((p) => p.id === prompt_id)
    if (saved_prompt?.prompt) {
      return saved_prompt.prompt
    }

    const is_generator = task_prompts.generators.some((g) => g.id === prompt_id)
    if (is_generator) {
      return null
    }

    return null
  }

  $: prompt_text =
    selected_run_config?.prompt?.prompt ||
    get_prompt_text_from_id(
      isKilnAgentRunConfig(selected_run_config?.run_config_properties)
        ? selected_run_config.run_config_properties.prompt_id
        : undefined,
    )

  let create_job_error: KilnError | null = null
  let create_job_loading = false
  let created_job: { id: string } | null = null

  let current_task: Task | null = null
  let task_load_error: KilnError | null = null
  let kiln_copilot_connected: boolean | null = null
  let has_prompt_optimization_entitlement: boolean | null = null
  let copilot_check_error: KilnError | null = null

  let loading = true
  $: error = task_load_error || copilot_check_error
  $: is_multiturn = current_task?.turn_mode === "multiturn"

  type EvalWithConfig = {
    eval: Eval
    configs: EvalConfig[]
    current_config: EvalConfig | null
    has_default_config: boolean
    has_train_set: boolean
    train_set_size: number | null
    model_is_supported: boolean
    validation_status: "unchecked" | "checking" | "valid" | "invalid"
    judge_error: string | null
    train_error: string | null
    other_error: string | null
  }

  let evals_with_configs: EvalWithConfig[] = []
  let evals_loading = false
  let evals_error: KilnError | null = null
  let selected_eval_ids: Set<string> = new Set()

  type EvalSortableColumn = "name" | "status" | "created_at"
  let evalSortColumn: EvalSortableColumn = "created_at"
  let evalSortDirection: "asc" | "desc" = "desc"

  function handleEvalSort(column: EvalSortableColumn) {
    if (evalSortColumn === column) {
      evalSortDirection = evalSortDirection === "asc" ? "desc" : "asc"
    } else {
      evalSortColumn = column
      evalSortDirection = "desc"
    }
  }

  function evalStatusOrder(
    status: "unchecked" | "checking" | "valid" | "invalid",
  ): number {
    switch (status) {
      case "valid":
        return 0
      case "invalid":
        return 1
      case "checking":
        return 2
      case "unchecked":
        return 3
    }
  }

  $: sorted_evals_with_configs = (() => {
    if (evals_with_configs.length === 0) return []
    return [...evals_with_configs].sort((a, b) => {
      let aValue: string | number
      let bValue: string | number
      switch (evalSortColumn) {
        case "name":
          aValue = (a.eval.name || "").toLowerCase()
          bValue = (b.eval.name || "").toLowerCase()
          break
        case "status":
          aValue = evalStatusOrder(a.validation_status)
          bValue = evalStatusOrder(b.validation_status)
          break
        case "created_at":
          aValue = a.eval.created_at ? new Date(a.eval.created_at).getTime() : 0
          bValue = b.eval.created_at ? new Date(b.eval.created_at).getTime() : 0
          break
        default:
          return 0
      }
      if (aValue < bValue) return evalSortDirection === "asc" ? -1 : 1
      if (aValue > bValue) return evalSortDirection === "asc" ? 1 : -1
      return 0
    })
  })()

  let run_config_validation_status:
    | "unchecked"
    | "checking"
    | "valid"
    | "invalid" = "unchecked"
  let run_config_validation_message: string | null = null
  let run_config_blocking_reason:
    | "has_tools"
    | "unsupported_model"
    | "other"
    | null = null

  $: has_evals_without_config = evals_with_configs.some(
    (item) =>
      item.eval.id &&
      selected_eval_ids.has(item.eval.id) &&
      !item.has_default_config,
  )
  $: has_unsupported_models =
    run_config_validation_status === "invalid" ||
    evals_with_configs.some(
      (item) =>
        item.eval.id &&
        selected_eval_ids.has(item.eval.id) &&
        item.has_default_config &&
        !item.model_is_supported,
    )
  $: is_validating =
    run_config_validation_status === "checking" ||
    evals_with_configs.some(
      (item) =>
        item.eval.id &&
        selected_eval_ids.has(item.eval.id) &&
        item.validation_status === "checking",
    )
  $: submit_disabled =
    selected_eval_ids.size === 0 ||
    has_evals_without_config ||
    evals_loading ||
    has_unsupported_models ||
    is_validating ||
    run_config_validation_status === "unchecked" ||
    evals_with_configs.some(
      (item) =>
        item.eval.id &&
        selected_eval_ids.has(item.eval.id) &&
        item.validation_status === "unchecked",
    )

  $: review_visible = run_config_validation_status === "valid"

  $: selected_run_config = get_selected_run_config(
    target_run_config_id,
    $run_configs_by_task_composite_id,
    project_id,
    task_id,
  )

  $: task_prompts =
    $prompts_by_task_composite_id[get_task_composite_id(project_id, task_id)] ||
    null

  function get_selected_run_config(
    run_config_id: string | null,
    configs_by_task: Record<string, TaskRunConfig[]>,
    proj_id: string,
    tsk_id: string,
  ): TaskRunConfig | null {
    if (
      !run_config_id ||
      run_config_id === "custom" ||
      run_config_id === "__create_new_run_config__"
    ) {
      return null
    }
    const configs =
      configs_by_task[get_task_composite_id(proj_id, tsk_id)] || []
    return configs.find((config) => config.id === run_config_id) || null
  }

  onMount(async () => {
    try {
      kiln_copilot_connected = await checkKilnCopilotAvailable()
    } catch (e) {
      copilot_check_error = createKilnError(e)
      kiln_copilot_connected = false
    }

    if (kiln_copilot_connected) {
      const { has_access, error: entitlement_error } =
        await checkPromptOptimizationAccess()
      has_prompt_optimization_entitlement = has_access
      if (entitlement_error) {
        copilot_check_error = entitlement_error
      }
    }

    if (kiln_copilot_connected && has_prompt_optimization_entitlement) {
      load_available_tools(project_id)
      await Promise.all([load_model_info(), load_available_models()])
      await Promise.all([
        load_task(),
        load_task_prompts(project_id, task_id),
        load_task_run_configs(project_id, task_id),
        load_evals_and_configs(),
      ])
    }
    loading = false
  })

  async function load_task() {
    try {
      task_load_error = null
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}",
        {
          params: {
            path: {
              project_id,
              task_id,
            },
          },
        },
      )
      if (error) {
        throw error
      }
      if (!data) {
        throw new KilnError("Task not found")
      }
      current_task = data
    } catch (e) {
      task_load_error = createKilnError(e)
    }
  }

  async function load_evals_and_configs() {
    try {
      evals_loading = true
      evals_error = null

      const { data: evals_response, error: evals_fetch_error } =
        await client.GET("/api/projects/{project_id}/tasks/{task_id}/evals", {
          params: {
            path: {
              project_id,
              task_id,
            },
          },
        })

      if (evals_fetch_error) {
        throw evals_fetch_error
      }

      if (!evals_response) {
        evals_with_configs = []
        return
      }
      const evals_data = evals_response.evals

      const evals_with_configs_promises = evals_data.map(async (evalItem) => {
        if (!evalItem.id) {
          throw new Error("Eval ID is missing")
        }

        const { data: configs_data, error: configs_error } = await client.GET(
          "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}/eval_configs",
          {
            params: {
              path: {
                project_id,
                task_id,
                eval_id: evalItem.id,
              },
            },
          },
        )

        if (configs_error) {
          throw configs_error
        }

        const current_config =
          (configs_data || []).find(
            (c) => c.id === evalItem.current_config_id,
          ) || null

        return {
          eval: evalItem,
          configs: configs_data || [],
          current_config,
          has_default_config: false,
          has_train_set: false,
          train_set_size: null,
          model_is_supported: false,
          validation_status: "unchecked" as const,
          judge_error: null,
          train_error: null,
          other_error: null,
        }
      })

      evals_with_configs = await Promise.all(evals_with_configs_promises)

      selected_eval_ids = new Set(
        evals_with_configs
          .map((item) => item.eval.id)
          .filter((id): id is string => id !== null && id !== undefined),
      )
    } catch (e) {
      evals_error = createKilnError(e)
    } finally {
      evals_loading = false
    }
  }

  async function check_run_config_validation() {
    if (!target_run_config_id) return

    try {
      run_config_validation_status = "checking"
      run_config_validation_message = null
      run_config_blocking_reason = null

      const run_config = get_selected_run_config(
        target_run_config_id,
        $run_configs_by_task_composite_id,
        project_id,
        task_id,
      )

      if (!isKilnAgentRunConfig(run_config?.run_config_properties)) {
        run_config_validation_status = "invalid"
        run_config_validation_message =
          "Only Kiln Agent run configurations are supported for Prompt Optimization"
        run_config_blocking_reason = "has_tools"
        return
      }

      if (
        run_config.run_config_properties.tools_config &&
        run_config.run_config_properties.tools_config.tools.length > 0
      ) {
        run_config_validation_status = "invalid"
        run_config_validation_message =
          "Tools and skills aren't supported for Kiln Prompt Optimization"
        run_config_blocking_reason = "has_tools"
        return
      }

      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/prompt_optimization_jobs/check_run_config",
        {
          params: {
            path: {
              project_id,
              task_id,
            },
            query: {
              run_config_id: target_run_config_id,
            },
          },
        },
      )

      if (error) {
        throw error
      }

      if (data && !data.is_supported) {
        run_config_validation_status = "invalid"
        run_config_blocking_reason = "unsupported_model"
        if (run_config) {
          run_config_validation_message = `${getRunConfigModelDisplayName(run_config, $model_info) ?? "This model"} is not supported for Prompt Optimization`
        } else {
          run_config_validation_message =
            "Model is not supported for Kiln Prompt Optimization"
        }
      } else {
        run_config_validation_status = "valid"
        run_config_validation_message = null
        run_config_blocking_reason = null
      }
    } catch (e) {
      run_config_validation_status = "invalid"
      run_config_validation_message = createKilnError(e).getMessage()
      run_config_blocking_reason = "other"
    }
  }

  async function check_eval_validation(
    index: number,
    run_even_if_unselected = false,
  ) {
    const item = evals_with_configs[index]
    if (!item.eval.id) return

    if (!run_even_if_unselected && !selected_eval_ids.has(item.eval.id)) {
      return
    }

    try {
      evals_with_configs[index].validation_status = "checking"
      evals_with_configs[index].judge_error = null
      evals_with_configs[index].train_error = null
      evals_with_configs[index].other_error = null
      evals_with_configs = [...evals_with_configs]

      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/prompt_optimization_jobs/check_eval",
        {
          params: {
            path: {
              project_id,
              task_id,
            },
            query: {
              eval_id: item.eval.id,
            },
          },
        },
      )

      if (error) {
        throw error
      }

      if (data) {
        evals_with_configs[index].has_default_config = data.has_default_config
        evals_with_configs[index].has_train_set = data.has_train_set
        evals_with_configs[index].model_is_supported = data.model_is_supported
        if (data.unsupported_reason) {
          evals_with_configs[index].other_error = data.unsupported_reason
        }

        // Reset before the conditional size fetch below: a stale count from a
        // previous TaskRun-backed split must not survive a switch to an
        // EvalInput-backed split, where the size is unknown.
        evals_with_configs[index].train_set_size = null

        // If has train set, fetch the size. The size fetch reads tag counts over
        // dataset runs, so it only applies to TaskRun-backed train splits. For an
        // EvalInput-backed train split (now valid for optimization) the size stays
        // unknown (null) and the empty-set error below does not apply.
        const train_filter_id = task_run_split_filter_id(item.eval, "train")
        if (data.has_train_set && train_filter_id) {
          const train_tag = tagFromFilterId(train_filter_id)
          if (train_tag) {
            try {
              const { data: tag_counts, error: tag_error } = await client.GET(
                "/api/projects/{project_id}/tasks/{task_id}/tags",
                {
                  params: {
                    path: {
                      project_id,
                      task_id,
                    },
                  },
                },
              )
              if (tag_error) {
                throw tag_error
              }
              evals_with_configs[index].train_set_size =
                tag_counts?.[train_tag] || 0
            } catch (_) {
              evals_with_configs[index].train_set_size = 0
            }
          }
        }

        if (!data.has_default_config) {
          evals_with_configs[index].judge_error = "No judge configured"
        } else if (!data.model_is_supported) {
          evals_with_configs[index].judge_error = "Model not supported"
        }

        if (!data.has_train_set) {
          evals_with_configs[index].train_error = "Training set required"
        } else if (
          evals_with_configs[index].train_set_size !== null &&
          evals_with_configs[index].train_set_size === 0
        ) {
          evals_with_configs[index].train_error = "Training set is empty"
        }

        const has_errors =
          evals_with_configs[index].judge_error !== null ||
          evals_with_configs[index].train_error !== null ||
          evals_with_configs[index].other_error !== null
        evals_with_configs[index].validation_status = has_errors
          ? "invalid"
          : "valid"
      }
    } catch (e) {
      evals_with_configs[index].validation_status = "invalid"
      evals_with_configs[index].other_error = createKilnError(e).getMessage()
    }
    evals_with_configs = [...evals_with_configs]
    if (
      evals_with_configs[index].validation_status === "invalid" &&
      item.eval.id
    ) {
      selected_eval_ids = new Set(
        [...selected_eval_ids].filter((id) => id !== item.eval.id),
      )
    }
  }

  $: if (
    target_run_config_id &&
    target_run_config_id !== "custom" &&
    target_run_config_id !== "__create_new_run_config__"
  ) {
    check_run_config_validation()
  } else {
    run_config_validation_status = "unchecked"
    run_config_validation_message = null
    run_config_blocking_reason = null
  }

  $: if (evals_with_configs.length > 0 && !evals_loading) {
    evals_with_configs.forEach((_, index) => {
      const item = evals_with_configs[index]
      if (
        item.eval.id &&
        selected_eval_ids.has(item.eval.id) &&
        item.validation_status === "unchecked"
      ) {
        check_eval_validation(index)
      }
    })
  }

  async function refresh_evaluators() {
    const { data: evals_data } = await client.GET(
      "/api/projects/{project_id}/tasks/{task_id}/evals",
      {
        params: {
          path: {
            project_id,
            task_id,
          },
        },
      },
    )

    if (evals_data) {
      const evals_by_id = new Map(evals_data.evals.map((e) => [e.id, e]))
      const configs_by_eval_id = await Promise.all(
        evals_with_configs.map(async (item) => {
          const eval_id = item.eval.id
          if (!eval_id) return { eval_id: null as string | null, configs: [] }
          const { data: configs_data } = await client.GET(
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
          const configs = configs_data || []
          return { eval_id, configs }
        }),
      )
      evals_with_configs = evals_with_configs.map((item, i) => {
        const fresh_eval = item.eval.id ? evals_by_id.get(item.eval.id) : null
        const { configs } = configs_by_eval_id[i] ?? { configs: [] }
        const eval_obj = fresh_eval ?? item.eval
        const current_config =
          (configs.find((c) => c.id === eval_obj.current_config_id) as
            | EvalConfig
            | undefined) ?? null
        return {
          ...item,
          eval: eval_obj,
          configs,
          current_config,
          validation_status: "unchecked" as const,
          judge_error: null,
          train_error: null,
          other_error: null,
        }
      })
    } else {
      evals_with_configs = evals_with_configs.map((item) => ({
        ...item,
        validation_status: "unchecked" as const,
        judge_error: null,
        train_error: null,
        other_error: null,
      }))
    }

    await Promise.all(
      evals_with_configs.map((_, i) => check_eval_validation(i, true)),
    )
  }

  async function create_prompt_optimization_job() {
    try {
      create_job_loading = true
      created_job = null
      create_job_error = null

      if (
        !target_run_config_id ||
        target_run_config_id === "custom" ||
        target_run_config_id === "__create_new_run_config__"
      ) {
        throw new Error("Please select a saved run configuration")
      }

      const { data: response, error: post_error } = await client.POST(
        "/api/projects/{project_id}/tasks/{task_id}/prompt_optimization_jobs/start",
        {
          params: {
            path: {
              project_id,
              task_id,
            },
          },
          body: {
            target_run_config_id,
            eval_ids: Array.from(selected_eval_ids),
          },
        },
      )

      if (post_error) {
        throw post_error
      }
      if (
        !response ||
        typeof response !== "object" ||
        !("id" in response) ||
        typeof response.id !== "string"
      ) {
        throw new Error("Invalid response from server")
      }
      posthog.capture("create_prompt", {
        from_generator: "prompt_optimization",
        eval_count: selected_eval_ids.size,
      })
      created_job = { id: response.id }
    } catch (e) {
      if (e instanceof Error && e.message.includes("Load failed")) {
        create_job_error = new KilnError(
          "Could not create a Kiln Prompt Optimization job.",
          null,
        )
      } else {
        create_job_error = createKilnError(e)
      }
    } finally {
      create_job_loading = false
    }
  }

  let input_transform_modal: InputTransformModal | null = null

  $: input_transform = selected_run_config
    ? getRunConfigInputTransform(selected_run_config)
    : null

  $: run_config_properties = selected_run_config
    ? getRunConfigUiProperties(
        project_id,
        task_id,
        selected_run_config,
        $model_info,
        task_prompts,
        $available_tools,
        () => input_transform_modal?.show(),
      )
    : null
</script>

<div class="max-w-[1400px]">
  <AppPage
    title="Create Optimized Prompt"
    subtitle="Automatically optimize your prompt, maximizing its quality using evals."
    sub_subtitle="Read the Docs"
    sub_subtitle_link="https://docs.kiln.tech/docs/prompts/automatic-prompt-optimizer"
    breadcrumbs={[
      {
        label: "Optimize",
        href: `/optimize/${project_id}/${task_id}`,
      },
      {
        label: "Prompts",
        href: `/prompts/${project_id}/${task_id}`,
      },
    ]}
  >
    {#if loading}
      <div class="w-full min-h-[50vh] flex justify-center items-center">
        <div class="loading loading-spinner loading-lg"></div>
      </div>
    {:else if error}
      <div class="w-full min-h-[50vh] flex justify-center items-center">
        <div class="text-error text-sm">
          {error.getMessage() || "An unknown error occurred"}
        </div>
      </div>
    {:else if kiln_copilot_connected === false}
      <PromptOptimizationCopilotRequired />
    {:else if has_prompt_optimization_entitlement === false}
      <EntitlementRequiredCard feature_name="Prompt Optimization" />
    {:else if created_job}
      <Completed
        title="Prompt Optimization Job Started"
        subtitle="Optimization may take up to several hours depending on the number of evaluators and training examples."
        button_text="View Optimizer Jobs"
        link={`/prompt_optimization/${project_id}/${task_id}/prompt_optimization_job/${created_job.id}`}
      />
    {:else if is_multiturn}
      <div class="flex flex-col items-center justify-center min-h-[60vh]">
        <Warning
          warning_message="Prompt optimization is not supported for multi-turn tasks."
          warning_color="warning"
          warning_icon="info"
        />
      </div>
    {:else if current_task}
      <FormContainer
        submit_visible={true}
        submit_label="Run Optimization"
        {submit_disabled}
        on:submit={create_prompt_optimization_job}
        bind:error={create_job_error}
        bind:submitting={create_job_loading}
        compact_button={true}
        warn_before_unload={selected_run_config !== null}
      >
        <div class="flex flex-col gap-8">
          <div>
            <div class="flex flex-col gap-1">
              <div class="text-xl font-bold flex justify-between items-center">
                <div class="text-xl font-bold">
                  Step 1: Select Target Run Configuration
                </div>
                <span class="font-normal">
                  <InfoTooltip
                    tooltip_text="The selected run configuration is used for two things:\n\n**1. Target model** — the model we'll optimize the prompt for. Different models respond differently, so the best prompt is model-specific.\n\n**2. Starting prompt** — we'll begin from this prompt and iterate. Choose your best prompt so the optimizer can build on your work.\n\nKiln Prompt Optimization only supports OpenRouter, OpenAI, Gemini, and Anthropic providers."
                  />
                </span>
              </div>
              <div class="text-xs text-gray-500 font-medium">
                We'll tune the prompt for this model. Different models respond
                differently, so the best prompt is model-specific.
              </div>
            </div>
            <SavedRunConfigurationsDropdown
              title=""
              {project_id}
              {current_task}
              bind:selected_run_config_id={target_run_config_id}
              run_page={false}
              auto_select_default={false}
              filter_run_configs={(config) =>
                isKilnAgentRunConfig(config.run_config_properties)}
            />

            {#if selected_run_config}
              {#if run_config_validation_status === "checking"}
                <div class="flex items-center gap-2 text-sm text-gray-500 mt-4">
                  <span class="loading loading-spinner loading-xs"></span>
                  <span>Validating run configuration...</span>
                </div>
              {:else if run_config_validation_status === "invalid"}
                <div class="mt-3">
                  <Warning
                    warning_color="error"
                    outline={true}
                    warning_message={run_config_blocking_reason === "has_tools"
                      ? `**${run_config_validation_message}**\nPlease select a different run configuration, or create a new one without tools or skills.`
                      : run_config_blocking_reason === "unsupported_model"
                        ? `**${run_config_validation_message}**\nPrompt Optimization only supports OpenRouter, OpenAI, Gemini, and Anthropic. See the [models page](/models) for supported models. Choose another run configuration or create one that uses a supported model and provider.`
                        : run_config_validation_message}
                    markdown={true}
                    trusted={true}
                  />
                </div>
              {:else if run_config_properties}
                <div
                  class="mt-6 grid grid-cols-1 xl:grid-cols-[1fr,auto] gap-6 items-start"
                >
                  <div class="flex-1 min-w-0">
                    <div class="text-md font-semibold text-left">Prompt</div>
                    <div class="text-xs text-gray-500 font-medium mt-1 mb-1">
                      Your selected run configuration's prompt. Will be used as
                      the starting point for optimization.
                    </div>
                    <Output raw_output={prompt_text || ""} max_height="220px" />
                  </div>
                  <div class="flex-shrink-0 flex-row max-w-[400px]">
                    <div class="text-md font-semibold text-left mb-4">
                      Details
                    </div>
                    <PropertyList
                      properties={run_config_properties}
                      open_links_in_new_tab={true}
                    />
                  </div>
                </div>
              {/if}
            {/if}
          </div>

          {#if review_visible && selected_run_config}
            <div>
              <div>
                <div class="flex flex-col gap-1">
                  <div
                    class="text-xl font-bold flex justify-between items-center"
                  >
                    <div>Step 2: Select Optimization Evals</div>
                    <span class="font-normal">
                      <InfoTooltip
                        tooltip_text="Our prompt optimizer will iteratively update your prompt.\n\n**Evals** — We use these evals to measure whether each new prompt is an improvement or regression.\n\n**Training Data** — We use the training dataset associated with each eval during optimization, so eval data is never seen by the optimizer.\n\n[Learn more](https://docs.kiln.tech/docs/prompts/automatic-prompt-optimizer)"
                      />
                    </span>
                  </div>
                  <div class="text-xs text-gray-500 font-medium">
                    Your prompt will be optimized to maximize performance across
                    these evals.
                  </div>
                </div>
              </div>

              <div>
                <div
                  class="text-sm font-medium text-gray-700 mt-2 mb-2 flex items-center justify-between"
                >
                  <div class="flex items-center gap-2">
                    {#if evals_with_configs.length > 0}
                      <span
                        >{selected_eval_ids.size} of {evals_with_configs.length}
                        selected</span
                      >
                    {/if}
                  </div>
                  <button
                    type="button"
                    class="btn btn-xs btn-outline"
                    on:click={refresh_evaluators}
                    disabled={is_validating}
                  >
                    {#if is_validating}
                      <span class="loading loading-spinner loading-xs"></span>
                    {:else}
                      ↻
                    {/if}
                    Refresh
                  </button>
                </div>

                {#if evals_loading}
                  <div class="flex justify-center items-center py-8">
                    <div class="loading loading-spinner loading-md"></div>
                  </div>
                {:else if evals_error}
                  <div
                    class="bg-error/10 border border-error/20 rounded-lg p-4"
                  >
                    <div class="text-error text-sm">
                      {evals_error.getMessage() || "Failed to load Evals"}
                    </div>
                  </div>
                {:else if evals_with_configs.length === 0}
                  <div
                    class="bg-base-200 rounded-lg p-4 text-center text-gray-500 flex flex-col items-center gap-2"
                  >
                    No evals configured for this task.
                    <a
                      class="btn btn-sm btn-primary"
                      href={`/specs/${project_id}/${task_id}/`}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Create Eval
                    </a>
                  </div>
                {:else}
                  <div
                    class="overflow-x-auto overflow-y-hidden rounded-lg border"
                  >
                    <table class="table">
                      <thead>
                        <tr>
                          <th style="width: 40px;"></th>
                          <th
                            on:click={() => handleEvalSort("name")}
                            class="hover:bg-base-200 cursor-pointer"
                          >
                            Name
                            <span class="inline-block w-3 text-center">
                              {evalSortColumn === "name"
                                ? evalSortDirection === "asc"
                                  ? "▲"
                                  : "▼"
                                : "\u200B"}
                            </span>
                          </th>
                          <th>Judge</th>
                          <th>Training Dataset</th>
                          <th
                            on:click={() => handleEvalSort("status")}
                            class="hover:bg-base-200 cursor-pointer"
                          >
                            Status
                            <span class="inline-block w-3 text-center">
                              {evalSortColumn === "status"
                                ? evalSortDirection === "asc"
                                  ? "▲"
                                  : "▼"
                                : "\u200B"}
                            </span>
                          </th>
                          <th
                            on:click={() => handleEvalSort("created_at")}
                            class="hover:bg-base-200 cursor-pointer"
                          >
                            Created
                            <span class="inline-block w-3 text-center">
                              {evalSortColumn === "created_at"
                                ? evalSortDirection === "asc"
                                  ? "▲"
                                  : "▼"
                                : "\u200B"}
                            </span>
                          </th>
                          <th class="w-[100px]"></th>
                        </tr>
                      </thead>
                      <tbody>
                        {#each sorted_evals_with_configs as { eval: evalItem, current_config, train_set_size, validation_status, judge_error, train_error, other_error }}
                          {@const spec_id = "legacy"}
                          {@const eval_url = evalItem.id
                            ? `/specs/${project_id}/${task_id}/${spec_id}/${evalItem.id}`
                            : undefined}
                          {@const is_selected = evalItem.id
                            ? selected_eval_ids.has(evalItem.id)
                            : false}
                          <tr
                            class="hover:bg-base-200 cursor-pointer"
                            on:click={() => {
                              if (
                                evalItem.id &&
                                validation_status !== "invalid"
                              ) {
                                const newSet = new Set(selected_eval_ids)
                                if (is_selected) {
                                  newSet.delete(evalItem.id)
                                } else {
                                  newSet.add(evalItem.id)
                                }
                                selected_eval_ids = newSet
                              }
                            }}
                          >
                            <td on:click|stopPropagation>
                              <input
                                type="checkbox"
                                class="checkbox checkbox-sm"
                                checked={is_selected}
                                disabled={validation_status === "invalid"}
                                on:change={(e) => {
                                  if (evalItem.id) {
                                    const newSet = new Set(selected_eval_ids)
                                    if (e.currentTarget.checked) {
                                      newSet.add(evalItem.id)
                                    } else {
                                      newSet.delete(evalItem.id)
                                    }
                                    selected_eval_ids = newSet
                                  }
                                }}
                              />
                            </td>
                            <td class="font-medium">
                              {evalItem.name}
                            </td>
                            <td class="text-sm">
                              {#if current_config}
                                <div class="text-gray-600">
                                  {current_config.name}
                                </div>
                                <div class="text-xs text-gray-500">
                                  {getDetailedModelNameFromParts(
                                    current_config.model_name ?? "",
                                    current_config.model_provider ?? "",
                                    $model_info,
                                  )}
                                </div>
                              {:else}
                                <span class="text-gray-400">—</span>
                              {/if}
                            </td>
                            <td class="text-sm whitespace-nowrap">
                              {#if train_set_size !== null}
                                {@const train_tag = tagFromFilterId(
                                  task_run_split_filter_id(evalItem, "train") ||
                                    "",
                                )}
                                {@const dataset_link = train_tag
                                  ? `/dataset/${project_id}/${task_id}?tags=${train_tag}`
                                  : undefined}
                                {#if dataset_link}
                                  <a
                                    href={dataset_link}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    class="link text-gray-600 hover:text-secondary"
                                    on:click|stopPropagation
                                  >
                                    {train_set_size}
                                    {train_set_size === 1 ? "item" : "items"}
                                  </a>
                                {:else}
                                  <span class="text-gray-500">
                                    {train_set_size}
                                    {train_set_size === 1 ? "item" : "items"}
                                  </span>
                                {/if}
                              {/if}
                            </td>
                            <td>
                              {#if validation_status === "checking"}
                                <span class="loading loading-spinner loading-xs"
                                ></span>
                              {:else if validation_status === "valid"}
                                <div
                                  class="badge badge-outline badge-sm badge-success"
                                >
                                  Ready
                                </div>
                              {:else if validation_status === "invalid"}
                                {@const eval_configs_link = `/specs/${project_id}/${task_id}/${spec_id}/${evalItem.id}/eval_configs`}
                                {@const train_tag = tagFromFilterId(
                                  task_run_split_filter_id(evalItem, "train") ||
                                    "",
                                )}
                                {@const dataset_add_link =
                                  train_tag &&
                                  `/dataset/${project_id}/${task_id}/add_data?reason=eval&splits=${train_tag}:1.0`}
                                {@const tooltip_parts = [
                                  "Fix the following issues and click Refresh to update this eval's status.",
                                  judge_error === "No judge configured"
                                    ? `**No judge configured** — This eval doesn't have a default judge. To fix, [set default judge](${eval_configs_link}) for this eval.`
                                    : null,
                                  judge_error === "Model not supported"
                                    ? `**Model not supported** — This eval's default judge model is not supported. Kiln Prompt Optimization only supports OpenRouter, OpenAI, Gemini, and Anthropic providers. To fix, [change your default judge](${eval_configs_link}) for this eval.`
                                    : null,
                                  train_error === "Training set required" ||
                                  train_error === "Training set is empty"
                                    ? `**Training set is empty** — This eval doesn't have any training data. To fix, ${dataset_add_link ? `[add samples to your dataset](${dataset_add_link})` : `add samples to your dataset`} with the tag '${train_tag}'.`
                                    : null,
                                  other_error
                                    ? "**Error**\n\n" + other_error
                                    : null,
                                ].filter((x) => x !== null)}
                                {@const combined_tooltip =
                                  tooltip_parts.join("\n\n")}
                                <div class="flex flex-row items-center gap-1">
                                  <div
                                    class="badge badge-outline badge-sm badge-error whitespace-nowrap"
                                  >
                                    Not Ready
                                  </div>
                                  <div class="text-gray-500">
                                    <InfoTooltip
                                      tooltip_text={combined_tooltip}
                                      position="right"
                                    />
                                  </div>
                                </div>
                              {/if}
                            </td>
                            <td class="text-sm text-gray-500 whitespace-nowrap">
                              {formatDate(evalItem.created_at || undefined)}
                            </td>
                            <td class="p-0" on:click|stopPropagation>
                              <TableActionMenu
                                items={[
                                  {
                                    label: "View Eval",
                                    href: eval_url,
                                    target: "_blank",
                                    rel: "noopener noreferrer",
                                    hidden: !eval_url,
                                  },
                                ]}
                              />
                            </td>
                          </tr>
                        {/each}
                      </tbody>
                    </table>
                  </div>

                  {#if selected_eval_ids.size === 0}
                    <div class="mt-4 flex justify-end">
                      <Warning
                        warning_color="error"
                        warning_icon="exclaim"
                        inline={true}
                        warning_message="No evaluators selected. Please select at least one evaluator."
                      />
                    </div>
                  {/if}
                {/if}
              </div>
            </div>
          {/if}
        </div>
      </FormContainer>
    {/if}
  </AppPage>

  <CreateNewRunConfigDialog
    bind:this={create_new_run_config_dialog}
    {project_id}
    task={current_task}
    new_run_config_created={(run_config) => {
      target_run_config_id = run_config.id || null
    }}
    hide_tools_selector={true}
    on:close={() => {
      if (target_run_config_id === "__create_new_run_config__") {
        target_run_config_id = null
      }
    }}
  />

  {#if input_transform}
    <InputTransformModal
      bind:this={input_transform_modal}
      transform={input_transform}
    />
  {/if}
</div>
