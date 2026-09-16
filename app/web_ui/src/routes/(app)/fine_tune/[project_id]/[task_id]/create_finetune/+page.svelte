<script lang="ts">
  import AppPage from "../../../../app_page.svelte"
  import FormContainer from "$lib/utils/form_container.svelte"
  import FormElement from "$lib/utils/form_element.svelte"
  import { page } from "$app/stores"
  import { client, base_url } from "$lib/api_client"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { onMount } from "svelte"
  import type { ChatStrategy } from "$lib/types"
  import Warning from "$lib/ui/warning.svelte"
  import Completed from "$lib/ui/completed.svelte"
  import PromptTypeSelector from "$lib/ui/run_config_component/prompt_type_selector.svelte"
  import {
    fine_tune_target_model as model_provider,
    load_task,
    current_task,
  } from "$lib/stores"
  import {
    available_tuning_models,
    available_models_error,
    available_models_loading,
    get_available_models,
  } from "$lib/stores/fine_tune_store"
  import { progress_ui_state } from "$lib/stores/progress_ui_store"
  import { goto } from "$app/navigation"
  import posthog from "posthog-js"

  import type {
    FinetuneProvider,
    DatasetSplit,
    Finetune,
    FineTuneParameter,
    KilnAgentRunConfigProperties,
    ModelProviderName,
    Task,
  } from "$lib/types"
  import { isKilnAgentRunConfig } from "$lib/types"
  import SelectFinetuneDataset from "./select_finetune_dataset.svelte"
  import InfoTooltip from "$lib/ui/info_tooltip.svelte"
  import RunConfigComponent from "$lib/ui/run_config_component/run_config_component.svelte"
  import type { OptionGroup, Option } from "$lib/ui/fancy_select_types"
  import Collapse from "$lib/ui/collapse.svelte"
  import { indexedDBStore } from "$lib/stores/index_db_store"
  import { writable, type Writable } from "svelte/store"
  import { load_task_run_configs } from "$lib/stores/run_configs_store"
  import { agentInfo } from "$lib/agent"
  let finetune_description = ""
  let finetune_name = ""
  const disabled_header = "disabled_header"
  let data_strategy: ChatStrategy = "final_only"
  let finetune_custom_system_prompt = ""
  let finetune_custom_thinking_instructions =
    "Think step by step, explaining your reasoning."
  let system_prompt_method = "simple_prompt_builder"

  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  $: agentInfo.set({
    name: "Create Fine-Tune",
    description: `Create a new fine-tuning job for project ID ${project_id}, task ID ${task_id}. Configure model, data strategy, and training parameters.`,
  })

  let task: Task | null = null
  let task_error: KilnError | null = null
  $: is_multiturn = task?.turn_mode === "multiturn"

  let run_config_component: RunConfigComponent | null = null

  let provider_id: ModelProviderName | null = null
  $: provider_id = $model_provider?.includes("/")
    ? ($model_provider?.split("/")[0] as ModelProviderName)
    : null
  $: base_model_id = $model_provider?.includes("/")
    ? $model_provider?.split("/").slice(1).join("/")
    : null

  $: selected_model = $available_tuning_models
    ?.flatMap((provider) =>
      provider.models.map((model) => ({
        provider,
        model,
      })),
    )
    .find(
      ({ provider, model }) =>
        provider.id === provider_id && model.id === base_model_id,
    )

  $: if (
    selected_model &&
    !selected_model.model.supports_function_calling &&
    run_config_component
  ) {
    // Clear tools and skills if the model doesn't support function calling
    run_config_component.clear_tools()
    run_config_component.clear_skills()
  }

  $: disabled_tools_selector =
    selected_model && !selected_model.model.supports_function_calling

  let selected_tool_ids: string[] = []
  let selected_skill_ids: string[] = []

  let available_model_select: OptionGroup[] = []

  let selected_dataset: DatasetSplit | null = null

  interface SavedFinetuneState {
    name?: string
    description?: string
    provider?: string
    base_model_id?: string
    dataset_split_id?: string
    parameters?: Record<string, string>
    system_message?: string
    thinking_instructions?: string
    data_strategy?: ChatStrategy
    system_prompt_method?: string
    tools?: string[]
    skills?: string[]
  }

  // IndexedDB-backed store for persisting form state
  let saved_state: Writable<SavedFinetuneState> = writable({})
  let state_initialized = false
  let saved_dataset_id: string | null = null

  // Track saved dataset ID reactively, this is used to restore the selected dataset when user leaves the page.
  $: saved_dataset_id = $saved_state.dataset_split_id || null

  // Initialize form from saved state
  export function initialize_from_finetune(state: SavedFinetuneState) {
    // Model selection (via store)
    if (state.provider && state.base_model_id) {
      $model_provider = `${state.provider}/${state.base_model_id}`
    }

    // Name and description
    finetune_name = state.name || ""
    finetune_description = state.description || ""

    // System prompt - restore the method that was selected
    if (state.system_prompt_method) {
      system_prompt_method = state.system_prompt_method
    }
    if (state.system_message) {
      finetune_custom_system_prompt = state.system_message
    }
    if (state.thinking_instructions) {
      finetune_custom_thinking_instructions = state.thinking_instructions
    }

    // Data strategy
    data_strategy = state.data_strategy || "final_only"

    // Hyperparameters
    hyperparameter_values = {}
    if (state.parameters) {
      for (const [key, value] of Object.entries(state.parameters)) {
        hyperparameter_values[key] = String(value)
      }
    }

    // Tools and skills
    selected_tool_ids = state.tools || []
    selected_skill_ids = state.skills || []
  }

  // Reactively update saved_state when form values change
  $: if (state_initialized) {
    saved_state.set({
      name: finetune_name || undefined,
      description: finetune_description || undefined,
      provider: provider_id || undefined,
      base_model_id: base_model_id || undefined,
      dataset_split_id: selected_dataset?.id || undefined,
      parameters:
        Object.keys(hyperparameter_values).length > 0
          ? hyperparameter_values
          : undefined,
      system_message: finetune_custom_system_prompt || undefined,
      thinking_instructions: finetune_custom_thinking_instructions || undefined,
      system_prompt_method: system_prompt_method,
      data_strategy: data_strategy,
      tools: selected_tool_ids.length > 0 ? selected_tool_ids : undefined,
      skills: selected_skill_ids.length > 0 ? selected_skill_ids : undefined,
    })
  }

  function clear_and_reload() {
    let msg =
      "Are you sure you want to clear current selections? This cannot be undone."

    if (confirm(msg)) {
      clear_saved_state()
      // reload the window keeping the same URL
      window.location.reload()
    }
  }

  function clear_saved_state() {
    // Prevent the reactive block from immediately repopulating the store
    state_initialized = false

    // Clear the saved state in IndexedDB by saving the defaults
    saved_state.update((s) => ({
      ...s,
      name: undefined,
      description: undefined,
      provider: undefined,
      base_model_id: undefined,
      dataset_split_id: undefined,
      parameters: undefined,
      system_message: undefined,
      thinking_instructions: undefined,
      data_strategy: "final_only",
      system_prompt_method: "simple_prompt_builder",
      tools: undefined,
      skills: undefined,
    }))
  }

  $: selecting_thinking_dataset =
    selected_dataset?.filter?.includes("thinking_model")
  $: selected_dataset_has_val = selected_dataset?.splits?.find(
    (s) => s.name === "val",
  )
  $: selected_dataset_training_set_name = selected_dataset?.split_contents[
    "train"
  ]
    ? "train"
    : selected_dataset?.split_contents["all"]
      ? "all"
      : null

  $: step_3_visible =
    ($model_provider && $model_provider !== disabled_header) ||
    !!selected_dataset ||
    !!saved_dataset_id
  $: step_4_visible =
    $model_provider && $model_provider !== disabled_header && !!selected_dataset
  $: is_download = !!$model_provider?.startsWith("download_")
  $: step_5_download_visible = step_4_visible && is_download
  $: submit_visible = !!(step_4_visible && !is_download)

  onMount(async () => {
    get_available_models()
    // Catch the load: an uncaught rejection would leave task === null and spin
    // the "task === null" loader forever (e.g. a deleted or inaccessible task).
    load_task(project_id, task_id)
      .then((loaded) => {
        task = loaded
      })
      .catch((e) => {
        if (e instanceof Error && e.message.includes("Load failed")) {
          task_error = new KilnError(
            "Could not load task. This task may belong to a project you don't have access to.",
            null,
          )
        } else {
          task_error = createKilnError(e)
        }
      })

    // Initialize IndexedDB-backed store for state persistence
    const state_key = `create_finetune_state_${project_id}_${task_id}`
    const { store, initialized } = indexedDBStore<SavedFinetuneState>(
      state_key,
      {},
    )
    await initialized
    saved_state = store

    // Load saved state if it exists
    if ($saved_state && Object.keys($saved_state).length > 0) {
      initialize_from_finetune($saved_state)
    }

    state_initialized = true
  })

  $: build_available_model_select($available_tuning_models, $current_task)

  function build_available_model_select(
    models: FinetuneProvider[] | null,
    task: Task | null,
  ) {
    if (!models) {
      return
    }
    available_model_select = []

    const models_with_tools: Option[] = []
    const models_without_tools: Option[] = []
    const disabled_providers: Option[] = []

    for (const provider of models) {
      for (const model of provider.models) {
        const model_key =
          (provider.enabled ? "" : "disabled_") + provider.id + "/" + model.id
        const model_label = provider.name + ": " + model.name

        const option: Option = {
          value: model_key,
          label: model_label,
        }

        if (!provider.enabled) {
          // if the provider is disabled, add a badge
          option.badge = "Requires API Key"
          option.badge_color = "primary"
          option.disabled = true
          disabled_providers.push(option)
        } else if (!model.supports_function_calling) {
          models_without_tools.push(option)
        } else {
          models_with_tools.push(option)
        }
      }

      if (!provider.enabled && provider.models.length === 0) {
        disabled_providers.push({
          value: "disabled_" + provider.id,
          label: provider.name,
          badge: "Requires API Key",
          badge_color: "primary",
          disabled: true,
        })
      }
    }

    if (models_with_tools.length > 0) {
      available_model_select.push({
        label: "Models with Tool Calling Support",
        options: models_with_tools,
      })
    }

    if (models_without_tools.length > 0) {
      available_model_select.push({
        label: "Models without Tool Calling Support",
        options: models_without_tools,
      })
    }

    if (disabled_providers.length > 0) {
      available_model_select.push({
        label: "Requires API Key Configuration",
        options: disabled_providers,
      })
    }

    const has_structured_output = !!task?.output_json_schema

    available_model_select.push({
      label: "Download Dataset",
      options: has_structured_output
        ? download_options
        : download_options.filter(
            (o) => !structured_only_formats.has(o.value as string),
          ),
    })

    // Check if the model provider is in the available model select
    // If not, reset to disabled header. The list can change over time.
    const all_values = available_model_select.flatMap((g) =>
      g.options.map((o) => o.value),
    )
    if (!all_values.includes($model_provider)) {
      $model_provider = disabled_header
    }
  }

  const download_model_select_options: Record<string, string> = {
    download_jsonl_msg: "openai_chat_jsonl",
    download_jsonl_json_schema_msg: "openai_chat_json_schema_jsonl",
    download_jsonl_toolcall: "openai_chat_toolcall_jsonl",
    download_huggingface_chat_template: "huggingface_chat_template_jsonl",
    download_huggingface_chat_template_toolcall:
      "huggingface_chat_template_toolcall_jsonl",
    download_vertex_gemini: "vertex_gemini",
  }

  const download_options: Option[] = [
    {
      value: "download_jsonl_msg",
      label: "OpenAI chat format (JSONL)",
    },
    {
      value: "download_jsonl_json_schema_msg",
      label: "OpenAI chat format with JSON response (JSONL)",
    },
    {
      value: "download_jsonl_toolcall",
      label: "OpenAI chat format with tool call response (JSONL)",
    },
    {
      value: "download_huggingface_chat_template",
      label: "HuggingFace chat template (JSONL)",
    },
    {
      value: "download_huggingface_chat_template_toolcall",
      label: "HuggingFace chat template with tool calls (JSONL)",
    },
    {
      value: "download_vertex_gemini",
      label: "Google Vertex-AI Gemini format (JSONL)",
    },
  ]

  const structured_only_formats = new Set([
    "download_jsonl_json_schema_msg",
    "download_jsonl_toolcall",
    "download_huggingface_chat_template_toolcall",
  ])

  $: get_hyperparameters(provider_id)

  let hyperparameters: FineTuneParameter[] | null = null
  let hyperparameters_error: KilnError | null = null
  let hyperparameters_loading = true
  let hyperparameter_values: Record<string, string> = {}
  async function get_hyperparameters(provider_id: string | null) {
    if (!provider_id || provider_id === disabled_header) {
      return
    }
    try {
      hyperparameters_loading = true
      hyperparameters = null
      hyperparameter_values = {}
      if (is_download) {
        // No hyperparameters for download options
        return
      }
      const { data: hyperparameters_response, error: get_error } =
        await client.GET("/api/finetune/hyperparameters/{provider_id}", {
          params: {
            path: {
              provider_id,
            },
          },
        })
      if (get_error) {
        throw get_error
      }
      if (!hyperparameters_response) {
        throw new Error("Invalid response from server")
      }
      hyperparameters = hyperparameters_response
    } catch (e) {
      if (e instanceof Error && e.message.includes("Load failed")) {
        hyperparameters_error = new KilnError(
          "Could not load hyperparameters for fine-tuning.",
          null,
        )
      } else {
        hyperparameters_error = createKilnError(e)
      }
    } finally {
      hyperparameters_loading = false
    }
  }

  const type_strings: Record<FineTuneParameter["type"], string> = {
    int: "Integer",
    float: "Float",
    bool: "Boolean - 'true' or 'false'",
    string: "String",
  }

  function get_system_prompt_method_param(): string | undefined {
    return system_prompt_method === "custom" ? undefined : system_prompt_method
  }
  function get_custom_system_prompt_param(): string | undefined {
    return system_prompt_method === "custom"
      ? finetune_custom_system_prompt
      : undefined
  }
  function get_custom_thinking_instructions_param(): string | undefined {
    return system_prompt_method === "custom" &&
      data_strategy === "two_message_cot"
      ? finetune_custom_thinking_instructions
      : undefined
  }

  let create_finetune_error: KilnError | null = null
  let create_finetune_loading = false
  let created_finetune: Finetune | null = null
  async function create_finetune() {
    try {
      create_finetune_loading = true
      created_finetune = null
      if (!provider_id || !base_model_id) {
        throw new Error("Invalid model or provider")
      }

      // Filter out empty strings from hyperparameter_values, and parse/validate types
      const hyperparameter_values = build_parsed_hyperparameters()

      // Create a run config object based on the UI
      // Extract just the model name from the full path (e.g., "accounts/fireworks/models/qwen3-1p7b" -> "qwen3-1p7b")
      const model_name = base_model_id?.split("/").pop() || base_model_id
      const base_run_config =
        run_config_component?.run_options_as_run_config_properties()
      const run_config_properties: KilnAgentRunConfigProperties | undefined =
        base_run_config && isKilnAgentRunConfig(base_run_config) && model_name
          ? {
              ...base_run_config,
              model_name: model_name,
              model_provider_name: provider_id,
            }
          : undefined

      const { data: create_finetune_response, error: post_error } =
        await client.POST(
          "/api/projects/{project_id}/tasks/{task_id}/finetunes",
          {
            params: {
              path: {
                project_id,
                task_id,
              },
            },
            body: {
              dataset_id: selected_dataset?.id || "",
              provider: provider_id,
              base_model_id: base_model_id,
              train_split_name: selected_dataset_training_set_name || "",
              name: finetune_name ? finetune_name : undefined,
              description: finetune_description
                ? finetune_description
                : undefined,
              system_message_generator: get_system_prompt_method_param(),
              custom_system_message: get_custom_system_prompt_param(),
              custom_thinking_instructions:
                get_custom_thinking_instructions_param(),
              parameters: hyperparameter_values,
              data_strategy: data_strategy,
              validation_split_name: selected_dataset_has_val
                ? "val"
                : undefined,
              run_config_properties: run_config_properties,
            },
          },
        )
      if (post_error) {
        throw post_error
      }
      if (!create_finetune_response || !create_finetune_response.id) {
        throw new Error("Invalid response from server")
      }
      posthog.capture("create_finetune", {
        base_model: base_model_id,
        provider: provider_id,
        prompt_method: system_prompt_method,
        supports_tools: disabled_tools_selector ? "no" : "yes",
        tool_count: selected_tool_ids.length,
        skill_count: selected_skill_ids.length,
      })
      created_finetune = create_finetune_response

      // Reload run configs to include the new finetune run config
      load_task_run_configs(project_id, task_id, true).catch((err) => {
        console.warn(
          "Failed to reload run configs store after finetune creation",
          err,
        )
      })

      // Clear the saved state now that fine-tune is created
      clear_saved_state()
      progress_ui_state.set({
        title: "Creating Fine-Tune",
        body: "In progress,  ",
        link: `/fine_tune/${project_id}/${task_id}/fine_tune/${created_finetune?.id}`,
        cta: "view job status",
        progress: null,
        step_count: 4,
        current_step: 3,
      })
    } catch (e) {
      if (e instanceof Error && e.message.includes("Load failed")) {
        create_finetune_error = new KilnError(
          "Could not create a dataset split for fine-tuning.",
          null,
        )
      } else {
        create_finetune_error = createKilnError(e)
      }
    } finally {
      create_finetune_loading = false
    }
  }

  function build_parsed_hyperparameters() {
    let parsed_hyperparameters: Record<string, string | number | boolean> = {}
    for (const hyperparameter of hyperparameters || []) {
      let raw_value = hyperparameter_values[hyperparameter.name]
      // remove empty strings
      if (!raw_value) {
        continue
      }
      let value = undefined
      if (hyperparameter.type === "int") {
        const parsed = parseInt(raw_value)
        if (
          isNaN(parsed) ||
          !Number.isInteger(parsed) ||
          parsed.toString() !== raw_value // checks it didn't parse 1.1 to 1
        ) {
          throw new Error(
            `Invalid integer value for ${hyperparameter.name}: ${raw_value}`,
          )
        }
        value = parsed
      } else if (hyperparameter.type === "float") {
        const parsed = parseFloat(raw_value)
        if (isNaN(parsed)) {
          throw new Error(
            `Invalid float value for ${hyperparameter.name}: ${raw_value}`,
          )
        }
        value = parsed
      } else if (hyperparameter.type === "bool") {
        if (raw_value !== "true" && raw_value !== "false") {
          throw new Error("Invalid boolean value: " + raw_value)
        }
        value = raw_value === "true"
      } else if (hyperparameter.type === "string") {
        value = raw_value
      } else {
        throw new Error("Invalid hyperparameter type: " + hyperparameter.type)
      }
      parsed_hyperparameters[hyperparameter.name] = value
    }
    return parsed_hyperparameters
  }

  async function download_dataset_jsonl(split_name: string) {
    const params = {
      dataset_id: selected_dataset?.id || "",
      project_id: project_id,
      task_id: task_id,
      split_name: split_name,
      data_strategy: data_strategy,
      format_type: $model_provider
        ? download_model_select_options[$model_provider]
        : undefined,
      system_message_generator: get_system_prompt_method_param(),
      custom_system_message: get_custom_system_prompt_param(),
      custom_thinking_instructions: get_custom_thinking_instructions_param(),
    }

    // Format params as query string, including escaping values and filtering undefined
    const query_string = Object.entries(params)
      .filter(([_, value]) => value !== undefined)
      .map(([key, value]) => `${key}=${encodeURIComponent(value || "")}`)
      .join("&")

    window.open(base_url + "/api/download_dataset_jsonl?" + query_string)
  }

  let data_strategy_select_options: [ChatStrategy, string][] = []

  function update_data_strategies_supported(
    model_provider: string | null,
    base_model_id: string | null,
    is_download: boolean,
    available_models: FinetuneProvider[] | null,
  ) {
    if (!model_provider || (!base_model_id && !is_download)) {
      return
    }

    const data_strategies_labels: Record<ChatStrategy, string> = {
      final_only: "Final Response Only (Recommended)",
      two_message_cot: "Thinking - Learn both thinking and final response",
      final_and_intermediate:
        "Thinking - Learn both thinking and final response (Legacy Format)",
      final_and_intermediate_r1_compatible: is_download
        ? "Thinking (R1 compatible) - Learn both thinking and final response"
        : "Thinking - Learn both thinking and final response",
    }

    const r1_disabled_for_downloads = [
      // R1 data strategy currently disabled for toolcall downloads
      // because unclear how to use in the best way
      "download_huggingface_chat_template_toolcall",
      "download_jsonl_toolcall",

      // R1 currently not supported by Vertex models
      "download_vertex_gemini",
    ]
    if (r1_disabled_for_downloads.includes(model_provider)) {
      return ["final_only", "two_message_cot"]
    }

    const compatible_data_strategies: ChatStrategy[] = is_download
      ? [
          "final_only",
          "two_message_cot",
          "final_and_intermediate_r1_compatible",
        ]
      : available_models
          ?.map((model) => model.models)
          .flat()
          .find((model) => model.id === base_model_id)
          ?.data_strategies_supported ?? []

    data_strategy_select_options = compatible_data_strategies.map(
      (strategy) => [strategy, data_strategies_labels[strategy]],
    ) as [ChatStrategy, string][]

    data_strategy = compatible_data_strategies[0]
  }

  $: update_data_strategies_supported(
    $model_provider,
    base_model_id,
    is_download,
    $available_tuning_models,
  )

  function go_to_providers_settings() {
    progress_ui_state.set({
      title: "Creating Fine-Tune",
      body: "When you're done connecting providers, ",
      link: $page.url.pathname,
      cta: "return to fine-tuning",
      progress: null,
      step_count: 4,
      current_step: 1,
    })
    goto("/settings/providers?highlight=finetune")
  }
</script>

<div class="max-w-[900px]">
  <AppPage
    title="Create a New Fine Tune"
    subtitle="Fine-tuned models learn from your dataset."
    sub_subtitle="Read the Docs"
    sub_subtitle_link="https://docs.kiln.tech/docs/fine-tuning-guide"
    breadcrumbs={[
      { label: "Fine Tunes", href: `/fine_tune/${project_id}/${task_id}` },
    ]}
    action_buttons={[
      {
        label: "Reset",
        handler: clear_and_reload,
      },
      {
        label: "Docs & Guide",
        href: "https://docs.kiln.tech/docs/fine-tuning-guide",
      },
    ]}
  >
    {#if $available_models_loading || (task === null && !task_error)}
      <div class="w-full min-h-[50vh] flex justify-center items-center">
        <div class="loading loading-spinner loading-lg"></div>
      </div>
    {:else if task_error}
      <div
        class="w-full min-h-[50vh] flex flex-col justify-center items-center gap-2"
      >
        <div class="font-medium">Error Loading Task</div>
        <div class="text-error text-sm">
          {task_error.getMessage() || "An unknown error occurred"}
        </div>
      </div>
    {:else if is_multiturn}
      <div class="flex flex-col items-center justify-center min-h-[60vh]">
        <Warning
          warning_message="Fine-tuning is not supported for multi-turn tasks."
          warning_color="warning"
          warning_icon="info"
        />
      </div>
    {:else if created_finetune}
      <Completed
        title="Fine Tune Created"
        subtitle="It will take a while to complete training."
        link={`/fine_tune/${project_id}/${task_id}/fine_tune/${created_finetune?.id}`}
        button_text="View Fine Tune Job"
      />
    {:else if $available_models_error}
      <div
        class="w-full min-h-[50vh] flex flex-col justify-center items-center gap-2"
      >
        <div class="font-medium">
          Error Loading Available Models and Datasets
        </div>
        <div class="text-error text-sm">
          {$available_models_error?.getMessage() || "An unknown error occurred"}
        </div>
      </div>
    {:else}
      <FormContainer
        {submit_visible}
        submit_label="Start Fine-Tune Job"
        on:submit={create_finetune}
        bind:error={create_finetune_error}
        bind:submitting={create_finetune_loading}
        gap={4}
      >
        <div class="text-xl font-bold">
          Step 1: Select Base Model to Fine-Tune
        </div>
        <div>
          <FormElement
            label="Model & Provider"
            description="Select which model to fine-tune. Alternatively, download a JSONL file to fine-tune using any infrastructure."
            inputType="fancy_select"
            id="provider"
            fancy_select_options={available_model_select}
            bind:value={$model_provider}
          />
          <button
            class="mt-2 underline decoration-gray-400"
            on:click={go_to_providers_settings}
          >
            <Warning
              warning_message="For 1-click fine-tuning connect Fireworks, Together, or Google Vertex."
              warning_icon="info"
              warning_color="success"
              inline={true}
            />
          </button>
        </div>
        <div class="text-xl font-bold">
          Step 2: Configure Fine-Tuning Run Configuration
        </div>
        <div>
          <PromptTypeSelector
            bind:prompt_method={system_prompt_method}
            description="The system message to use for fine-tuning. Choose the prompt you want to use with your fine-tuned model."
            info_description="There are tradeoffs to consider when choosing a system prompt for fine-tuning. Read more: [OpenAI Docs](https://platform.openai.com/docs/guides/fine-tuning/#crafting-prompts)."
            exclude_cot={true}
            custom_prompt_name="Custom Fine Tuning Prompt"
            {project_id}
            {task_id}
          />
          {#if system_prompt_method === "custom"}
            <div class="p-4 border-l-4 border-gray-300">
              <FormElement
                label="Custom System Prompt"
                description="Enter a custom system prompt to use during fine-tuning."
                info_description="There are tradeoffs to consider when choosing a system prompt for fine-tuning. Read more: [OpenAI Docs](https://platform.openai.com/docs/guides/fine-tuning/#crafting-prompts)."
                inputType="textarea"
                id="finetune_custom_system_prompt"
                bind:value={finetune_custom_system_prompt}
              />
              {#if data_strategy === "two_message_cot"}
                <div class="mt-4"></div>
                <FormElement
                  label="Custom Thinking Instructions"
                  description="Instructions for the model's 'thinking' stage, before returning the final response."
                  info_description="When training with intermediate results (reasoning, chain of thought, etc.), this prompt will be used to ask the model to 'think' before returning the final response."
                  inputType="textarea"
                  id="finetune_custom_thinking_instructions"
                  bind:value={finetune_custom_thinking_instructions}
                />
              {/if}
            </div>
          {/if}
        </div>
        <div>
          <FormElement
            label="Reasoning"
            description="Should the model be trained on reasoning/thinking content?"
            info_description="If you select 'Thinking', the model training will include thinking such as reasoning or chain of thought. Use this if you want to call the tuned model with a chain-of-thought prompt for additional inference time compute."
            inputType="select"
            id="data_strategy"
            select_options={data_strategy_select_options}
            bind:value={data_strategy}
          />
        </div>
        <div>
          <RunConfigComponent
            bind:this={run_config_component}
            bind:tools={selected_tool_ids}
            bind:skills={selected_skill_ids}
            {project_id}
            tools_selector_settings={{
              hide_create_kiln_task_tool_button: true,
              hide_info_description: true,
              disabled: disabled_tools_selector,
              empty_label: disabled_tools_selector
                ? "Tool calling not supported on this model"
                : undefined,
              description:
                "Choose which tools the model should learn to call during fine-tuning.",
            }}
            skills_selector_settings={{
              hide_info_description: true,
              disabled: !!disabled_tools_selector,
              empty_label: disabled_tools_selector
                ? "Skills not supported with this model"
                : undefined,
              description:
                "Choose which skills the model should learn to use during fine-tuning.",
            }}
            hide_model_selector={true}
            hide_prompt_selector={true}
          />
        </div>

        {#if step_3_visible}
          <div>
            <div class="text-xl font-bold">
              Step 3: Select Fine-Tuning Dataset
            </div>
            <div class="font-light">
              Select a dataset to use for this fine-tune.
              <InfoTooltip
                tooltip_text="A fine-tuning dataset is a subset of your dataset which is used to train and validate the fine-tuned model. This is typically a subset of your dataset, which is intentionally kept separate from your eval data."
                position="bottom"
                no_pad={true}
              />
            </div>
          </div>
          {#if state_initialized}
            <SelectFinetuneDataset
              {project_id}
              {task_id}
              required_tool_ids={[...selected_tool_ids, ...selected_skill_ids]}
              {saved_dataset_id}
              bind:selected_dataset
            />
            {#if selected_dataset && data_strategy === "two_message_cot" && !selecting_thinking_dataset}
              <Warning
                warning_message="You are training a model for inference-time thinking, but are not using a dataset filtered to samples with reasoning or chain-of-thought training data. This is not recommended, as it may lead to poor performance. We suggest creating a new dataset with a thinking filter."
                large_icon={true}
              />
            {/if}
            {#if selected_dataset && data_strategy === "final_and_intermediate_r1_compatible" && !selecting_thinking_dataset}
              <Warning
                warning_message="You are training a 'thinking' model, but did not explicitly select a dataset filtered to samples with reasoning or chain-of-thought training data. If any of your training samples are missing reasoning data, it will error. If your data contains reasoning, you can ignore this warning."
                large_icon={true}
              />
            {/if}
          {/if}
        {/if}

        {#if step_4_visible}
          <div class="text-xl font-bold">Step 4: Advanced Options</div>
          {#if !is_download}
            <div>
              <Collapse title="Advanced Options">
                <FormElement
                  label="Name"
                  description="A name to identify this fine-tune. Leave blank and we'll generate one for you."
                  optional={true}
                  inputType="input"
                  id="finetune_name"
                  bind:value={finetune_name}
                />
                <FormElement
                  label="Description"
                  description="An optional description of this fine-tune."
                  optional={true}
                  inputType="textarea"
                  id="finetune_description"
                  bind:value={finetune_description}
                />
                {#if hyperparameters_loading}
                  <div class="w-full flex justify-center items-center">
                    <div class="loading loading-spinner loading-lg"></div>
                  </div>
                {:else if hyperparameters_error || !hyperparameters}
                  <div class="text-error text-sm">
                    {hyperparameters_error?.getMessage() ||
                      "An unknown error occurred"}
                  </div>
                {:else if hyperparameters.length > 0}
                  {#each hyperparameters as hyperparameter}
                    <FormElement
                      label={hyperparameter.name +
                        " (" +
                        type_strings[hyperparameter.type] +
                        ")"}
                      description={hyperparameter.description}
                      info_description="If you aren't sure, leave blank for default/recommended value. Ensure your value is valid for the type (e.g. an integer can't have decimals)."
                      inputType="input"
                      optional={hyperparameter.optional}
                      id={hyperparameter.name}
                      bind:value={hyperparameter_values[hyperparameter.name]}
                    />
                  {/each}
                {/if}
              </Collapse>
            </div>
          {/if}
        {/if}
      </FormContainer>
    {/if}
    {#if step_5_download_visible}
      <div>
        <div class="text-xl font-bold">Step 5: Download JSONL</div>
        <div class="text-sm">
          Download JSONL files to fine-tune using any infrastructure, such as
          <a
            href="https://github.com/unslothai/unsloth"
            class="link"
            target="_blank">Unsloth</a
          >
          or
          <a
            href="https://github.com/axolotl-ai-cloud/axolotl"
            class="link"
            target="_blank">Axolotl</a
          >.
        </div>
        <div class="flex flex-col gap-4 mt-6">
          {#each Object.keys(selected_dataset?.split_contents || {}) as split_name}
            <button
              class="btn {Object.keys(selected_dataset?.split_contents || {})
                .length > 1
                ? 'btn-secondary btn-outline'
                : 'btn-primary'} max-w-[400px]"
              on:click={() => download_dataset_jsonl(split_name)}
            >
              Download Split: {split_name} ({selected_dataset?.split_contents[
                split_name
              ]?.length} examples)
            </button>
          {/each}
        </div>
      </div>
    {/if}
  </AppPage>
</div>
