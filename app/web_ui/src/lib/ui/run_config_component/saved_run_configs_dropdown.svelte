<script lang="ts">
  import FormElement, {
    type InlineAction,
  } from "$lib/utils/form_element.svelte"
  import type { OptionGroup, Option } from "$lib/ui/fancy_select_types"
  import type {
    PromptResponse,
    ProviderModels,
    Task,
    TaskRunConfig,
  } from "$lib/types"
  import {
    model_info,
    load_model_info,
    get_task_composite_id,
  } from "$lib/stores"
  import {
    load_task_prompts,
    prompts_by_task_composite_id,
  } from "$lib/stores/prompts_store"
  import {
    getRunConfigModelDisplayName,
    getRunConfigPromptDisplayName,
    getRunConfigInputTransformSummaryLabel,
  } from "$lib/utils/run_config_formatters"
  import { isMcpRunConfig } from "$lib/types"
  import { onMount } from "svelte"
  import {
    load_task_run_configs,
    run_configs_by_task_composite_id,
    update_task_default_run_config,
  } from "$lib/stores/run_configs_store"
  import {
    get_last_used_run_config,
    set_last_used_run_config,
  } from "$lib/stores/last_used_run_config_store"
  import { createKilnError, type KilnError } from "$lib/utils/error_handlers"
  import Warning from "$lib/ui/warning.svelte"

  export let title: string = "Run Configuration"
  export let project_id: string
  export let current_task: Task
  export let selected_run_config_id: string | null = null // This will be null until the default_run_config_id is set
  export let save_new_run_config: (() => Promise<TaskRunConfig>) | null = null
  export let save_config_error: KilnError | null = null
  export let set_default_error: KilnError | null = null
  export let info_description: string = ""
  export let description: string = ""
  export let run_page: boolean = true
  export let auto_select_default: boolean = true
  export let selected_model_specific_run_config_id: string | null = null
  export let filter_run_configs: ((config: TaskRunConfig) => boolean) | null =
    null

  $: show_save_button = run_page && selected_run_config_id === "custom"
  $: show_set_default_button =
    run_page && selected_run_config_id !== default_run_config_id

  onMount(async () => {
    load_model_info()
  })

  $: default_run_config_id = current_task.default_run_config_id ?? null

  $: if (project_id && current_task.id) {
    load_task_run_configs(project_id, current_task.id)
    load_task_prompts(project_id, current_task.id)
  }

  // Initialization of selected_run_config_id.
  // Start with "custom" immediately (avoids showing "Select an option" while configs load),
  // then upgrade to the preferred config once it's available in the loaded options.
  // Preference order: user's persisted last-used selection for this task, then the task default.
  // Last-used wins so a "Custom" selection (and any sticky model tied to it) survives revisits
  // even when the task has a default configured.
  let pending_preferred_run_config_id: string | null = null
  let initialized_for_task_id: string | null = null

  // Reset selection when the task changes so init re-runs for the new task. The parent's
  // bound selected_run_config_id survives task switches (dropdown remounts, parent does not),
  // so without this we'd persist the old task's id against the new task's storage key.
  $: if (
    initialized_for_task_id !== null &&
    initialized_for_task_id !== (current_task?.id ?? null)
  ) {
    selected_run_config_id = null
    pending_preferred_run_config_id = null
    initialized_for_task_id = null
  }

  $: if (auto_select_default && selected_run_config_id === null) {
    if (run_page && current_task?.id) {
      const composite_key = get_task_composite_id(project_id, current_task.id)
      const last_used = get_last_used_run_config(composite_key)
      selected_run_config_id = "custom"
      initialized_for_task_id = current_task.id
      if (last_used === "custom") {
        pending_preferred_run_config_id = null
      } else if (last_used) {
        pending_preferred_run_config_id = last_used
      } else if (default_run_config_id) {
        pending_preferred_run_config_id = default_run_config_id
      } else {
        pending_preferred_run_config_id = null
      }
    } else {
      selected_run_config_id = null
    }
  }
  $: if (
    pending_preferred_run_config_id &&
    auto_select_default &&
    run_page &&
    selected_run_config_id === "custom" &&
    current_task?.id
  ) {
    const composite_key = get_task_composite_id(project_id, current_task.id)
    const loaded_configs = $run_configs_by_task_composite_id[composite_key]
    if (loaded_configs !== undefined) {
      if (
        loaded_configs.some((c) => c.id === pending_preferred_run_config_id)
      ) {
        selected_run_config_id = pending_preferred_run_config_id
      } else if (
        default_run_config_id &&
        loaded_configs.some((c) => c.id === default_run_config_id)
      ) {
        // Persisted last-used id no longer exists; fall back to the task default.
        selected_run_config_id = default_run_config_id
      }
      pending_preferred_run_config_id = null
    }
  }

  // Persist the current selection so returning visits prefer it over the task default.
  // Gate on initialized_for_task_id matching current_task.id so we never write a stale selection
  // (e.g., leftover from a previous task) against the wrong storage key.
  $: if (
    run_page &&
    !pending_preferred_run_config_id &&
    selected_run_config_id &&
    current_task?.id &&
    initialized_for_task_id === current_task.id
  ) {
    set_last_used_run_config(
      get_task_composite_id(project_id, current_task.id),
      selected_run_config_id,
    )
  }

  let cold_start_info_description =
    "You can save your run configuration including model, prompt, tools and properties. This makes it easy to return to later."
  let saved_configs_info_description =
    "Select a saved run configuration which includes model, prompt, tools and properties. Alternatively choose 'Custom' to manually configure this run."

  // Whether the caller gave the control its own tooltip copy. Captured once at
  // init, because the default below writes to the same prop: without the
  // snapshot a caller's text would be overwritten as soon as the options load.
  const caller_supplied_info_description = info_description !== ""

  $: if (!caller_supplied_info_description && run_page && options.length > 0) {
    info_description =
      options.length === 1
        ? cold_start_info_description
        : saved_configs_info_description
  }

  let options: OptionGroup[] = []

  $: options = build_options(
    default_run_config_id,
    $run_configs_by_task_composite_id,
    $model_info,
    $prompts_by_task_composite_id[
      get_task_composite_id(project_id, current_task.id ?? "")
    ] ?? { generators: [], prompts: [] },
    run_page,
    selected_model_specific_run_config_id,
    filter_run_configs,
  )

  // Build the options for the dropdown
  function build_options(
    default_run_config_id: string | null | undefined,
    run_configs_by_task_composite_id: Record<string, TaskRunConfig[]>,
    model_info: ProviderModels | null,
    current_task_prompts: PromptResponse | null,
    run_page: boolean,
    selected_model_specific_run_config_id: string | null,
    filter_run_configs: ((config: TaskRunConfig) => boolean) | null,
  ): OptionGroup[] {
    const options: OptionGroup[] = []

    if (run_page) {
      // Add new custom configuration option
      options.push({
        label: "",
        options: [
          {
            value: "custom",
            label: "Custom",
            description: "Run with the options specified below.",
          },
        ],
      })
    } else {
      options.push({
        label: "",
        options: [
          {
            value: "__create_new_run_config__",
            label: "New Run Configuration",
            badge: "＋",
            badge_color: "primary",
          },
        ],
      })
    }

    const all_run_configs = (
      run_configs_by_task_composite_id[
        get_task_composite_id(project_id, current_task.id ?? "")
      ] ?? []
    ).filter((config) => !filter_run_configs || filter_run_configs(config))

    // Add model-specific run config first if it is specified and exists
    if (selected_model_specific_run_config_id) {
      const model_specific_run_config = all_run_configs.find(
        (config) => config.id === selected_model_specific_run_config_id,
      )

      if (model_specific_run_config) {
        const is_finetune_run_config =
          selected_model_specific_run_config_id.startsWith(
            "finetune_run_config::",
          )
        const model_specific_section_label = is_finetune_run_config
          ? "Fine-Tune Configurations"
          : "Model Specific Configurations"
        const model_specific_config_label = is_finetune_run_config
          ? "Fine-Tune Config"
          : "Model Specific Run Config"
        const model_specific_config_description = is_finetune_run_config
          ? "The run configuration used to fine-tune the selected model."
          : "The run configuration suggested for the selected model."

        options.push({
          label: model_specific_section_label,
          options: [
            {
              value: model_specific_run_config.id,
              label: model_specific_config_label,
              description: model_specific_config_description,
              badge: "Recommended",
              badge_color: "primary",
            },
          ],
        })
      }
    }

    // Add saved configurations if they exist
    let saved_configuration_options: Option[] = []

    // Add default configuration first if it exists
    if (default_run_config_id) {
      const default_config = all_run_configs.find(
        (config) => config.id === default_run_config_id,
      )

      if (default_config) {
        const mcp_props = isMcpRunConfig(default_config.run_config_properties)
          ? default_config.run_config_properties
          : null
        let description = mcp_props
          ? `MCP Tool: ${mcp_props.tool_reference?.tool_name ?? "Unknown"}`
          : `Model: ${getRunConfigModelDisplayName(default_config, model_info)}
            Prompt: ${getRunConfigPromptDisplayName(default_config, current_task_prompts)}`
        if (!mcp_props) {
          const transformLabel =
            getRunConfigInputTransformSummaryLabel(default_config)
          if (transformLabel) {
            description += `\nInput Transform: ${transformLabel}`
          }
        }
        saved_configuration_options.push({
          value: default_run_config_id,
          label: `${default_config.name} (Default)`,
          description,
        })
      }
    }

    // Exclude finetune run configs
    const other_task_run_configs = all_run_configs.filter(
      (config) =>
        config.id !== default_run_config_id &&
        !config.id?.startsWith("finetune_run_config::"),
    )
    if (other_task_run_configs.length > 0) {
      saved_configuration_options.push(
        ...other_task_run_configs.map((config) => {
          const mcp_props = isMcpRunConfig(config.run_config_properties)
            ? config.run_config_properties
            : null
          let description = mcp_props
            ? `MCP Tool: ${mcp_props.tool_reference?.tool_name ?? "Unknown"}`
            : `Model: ${getRunConfigModelDisplayName(config, model_info)}
            Prompt: ${getRunConfigPromptDisplayName(config, current_task_prompts)}`
          if (!mcp_props) {
            const transformLabel =
              getRunConfigInputTransformSummaryLabel(config)
            if (transformLabel) {
              description += `\nInput Transform: ${transformLabel}`
            }
          }
          return {
            value: config.id ?? "",
            label: config.name,
            description,
          }
        }),
      )
    }

    if (saved_configuration_options.length > 0) {
      options.push({
        label: "Saved Configurations",
        options: saved_configuration_options,
      })
    }

    return options
  }

  $: void (selected_run_config_id, clear_run_options_errors())

  function clear_run_options_errors() {
    save_config_error = null
    set_default_error = null
  }

  let inline_action_loading = false

  async function handle_save() {
    if (!save_new_run_config) {
      return
    }
    try {
      inline_action_loading = true
      save_config_error = null
      const saved_run_config = await save_new_run_config()
      if (saved_run_config.id) {
        selected_run_config_id = saved_run_config.id
      }
    } catch (e) {
      save_config_error = createKilnError(e)
    } finally {
      inline_action_loading = false
    }
  }

  async function set_run_config_as_default() {
    if (!project_id || !current_task.id || !selected_run_config_id) {
      return
    }
    // Update task default run config
    try {
      inline_action_loading = true
      set_default_error = null
      await update_task_default_run_config(
        project_id,
        current_task.id,
        selected_run_config_id,
      )
    } catch (e) {
      set_default_error = createKilnError(e)
    } finally {
      inline_action_loading = false
    }
  }

  let inline_action: InlineAction | null = null

  $: void (show_save_button,
  show_set_default_button,
  inline_action_loading,
  update_inline_action())

  function update_inline_action() {
    if (show_save_button) {
      inline_action = {
        handler: handle_save,
        label: "Save current options",
        loading: inline_action_loading,
        loading_text: "Saving...",
      }
    } else if (show_set_default_button) {
      inline_action = {
        handler: set_run_config_as_default,
        label: "Set as task default",
        loading: inline_action_loading,
        loading_text: "Saving...",
      }
    } else {
      inline_action = null
    }
  }
</script>

<div class="flex flex-col gap-2">
  <FormElement
    label={title}
    {description}
    {info_description}
    inputType="fancy_select"
    bind:value={selected_run_config_id}
    id="run_config"
    bind:fancy_select_options={options}
    {inline_action}
  />
  {#if save_config_error}
    <div class="text-error text-sm text-right">
      {save_config_error.getMessage() || "An unknown error occurred"}
    </div>
  {/if}
  {#if set_default_error}
    <div class="text-error text-sm text-right">
      {set_default_error.getMessage() || "An unknown error occurred"}
    </div>
  {/if}
  {#if selected_model_specific_run_config_id && selected_model_specific_run_config_id !== selected_run_config_id}
    <Warning
      warning_icon="exclaim"
      warning_color="warning"
      warning_message="You are not using the run configuration recommended for the selected model."
    />
  {/if}
</div>
