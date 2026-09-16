<script lang="ts">
  import {
    available_models,
    available_model_details,
    load_available_models,
    get_task_composite_id,
  } from "$lib/stores"
  import {
    load_task_run_configs,
    run_configs_by_task_composite_id,
    save_new_task_run_config,
  } from "$lib/stores/run_configs_store"
  import { KilnError } from "$lib/utils/error_handlers"
  import type {
    RunConfigProperties,
    KilnAgentRunConfigProperties,
    StructuredOutputMode,
    AvailableModels,
    Task,
    TaskRunConfig,
    InputTransform,
  } from "$lib/types"
  import { isKilnAgentRunConfig, isMcpRunConfig } from "$lib/types"
  import { inputTransformsEqual } from "$lib/utils/run_config_formatters"
  import AvailableModelsDropdown from "./available_models_dropdown.svelte"
  import PromptTypeSelector from "./prompt_type_selector.svelte"
  import ToolsSelector from "./tools_selector.svelte"
  import SkillsSelector from "./skills_selector.svelte"
  import AdvancedRunOptions from "./advanced_run_options.svelte"
  import Collapse from "$lib/ui/collapse.svelte"
  import McpRunConfigPanel from "$lib/ui/run_config_component/mcp_run_config_panel.svelte"
  import { tick, onMount } from "svelte"
  import { ui_state } from "$lib/stores"
  import { load_task_prompts } from "$lib/stores/prompts_store"
  import type { ModelDropdownSettings } from "./model_dropdown_settings"
  import FormElement from "$lib/utils/form_element.svelte"
  import { arrays_equal } from "$lib/utils/collections"
  import type { ToolsSelectorSettings } from "./tools_selector_settings"
  import type { SkillsSelectorSettings } from "./skills_selector_settings"
  import { generate_memorable_name } from "$lib/utils/name_generator"
  import {
    filename_string_validator_default,
    normalize_filename_string,
  } from "$lib/utils/input_validators"
  import { split_tool_and_skill_ids } from "$lib/stores/tools_store"

  // Props
  export let project_id: string
  export let current_task: Task | null = null // When task is null, certain functionality is disabled such as saving a new run config
  export let model_name: string = ""
  export let provider: string = ""
  export let model_dropdown_settings: Partial<ModelDropdownSettings> = {}
  // Name and explain the model dropdown when the caller's model plays a
  // specific role. Defaults keep the plain "Model" label with no tooltip.
  export let model_label: string = "Model"
  export let model_info_description: string | undefined = undefined
  export let tools_selector_settings: Partial<ToolsSelectorSettings> = {}
  export let skills_selector_settings: Partial<SkillsSelectorSettings> = {}
  export let selected_run_config_id: string | null = null
  export let set_default_error: KilnError | null = null
  export let hide_prompt_selector: boolean = false
  export let hide_tools_selector: boolean = false
  export let show_tools_selector_in_advanced: boolean = false
  export let requires_structured_output: boolean = false
  export let hide_model_selector: boolean = false
  export let pending_tool_id: string | null = null
  export let pending_skill_id: string | null = null
  export let pending_run_config_id: string | null = null
  // Model-specific suggested run config, such as fine-tuned models. If a model like that is selected, this will be set to the run config ID.
  export let selected_model_specific_run_config_id: string | null = null
  export let run_config_name: string = generate_memorable_name()
  export let show_name_field: boolean = true

  export let model: string | null = $ui_state.selected_model
  // Optional: seed every run option (model, prompt, tools, temperature, top_p,
  // structured output mode, thinking level) from an existing config. Takes
  // precedence over `model` and the model defaults. See below.
  export let initial_run_config_properties: KilnAgentRunConfigProperties | null =
    null
  export let prompt_method: string = "simple_prompt_builder"
  export let tools: string[] = []
  export let skills: string[] = []
  let requires_tool_support: boolean = false

  // The run options this component starts on, before a model's own defaults or
  // a caller's config override them. One object so reset_run_options() below
  // cannot drift from the values used here.
  // The sampling defaults are used by every provider I checked (OpenRouter,
  // Fireworks, Together, etc)
  const DEFAULT_RUN_OPTIONS: {
    temperature: number
    top_p: number
    structured_output_mode: StructuredOutputMode
    thinking_level: string | null
    input_transform: InputTransform | null
  } = {
    temperature: 1.0,
    top_p: 1.0,
    structured_output_mode: "default",
    thinking_level: null,
    input_transform: null,
  }

  let temperature: number = DEFAULT_RUN_OPTIONS.temperature
  let top_p: number = DEFAULT_RUN_OPTIONS.top_p

  let structured_output_mode: StructuredOutputMode =
    DEFAULT_RUN_OPTIONS.structured_output_mode
  let thinking_level: string | null = DEFAULT_RUN_OPTIONS.thinking_level
  let input_transform: InputTransform | null =
    DEFAULT_RUN_OPTIONS.input_transform
  $: current_model_details = available_model_details(
    model_name,
    provider,
    $available_models,
  )

  $: model_name = model ? model.split("/").slice(1).join("/") : ""
  $: provider = model ? model.split("/")[0] : ""
  $: requires_tool_support = tools.length > 0 || skills.length > 0

  $: updated_model_dropdown_settings = {
    ...model_dropdown_settings,
    requires_tool_support: requires_tool_support,
    requires_structured_output: requires_structured_output,
  }

  let model_dropdown: AvailableModelsDropdown
  let model_dropdown_error_message: string | null = null

  onMount(async () => {
    await load_available_models()
  })

  $: if (project_id && current_task?.id) {
    load_task_prompts(project_id, current_task.id)
  }

  $: is_mcp = (() => {
    if (!selected_run_config_id || !current_task?.id) return false
    const all_configs =
      $run_configs_by_task_composite_id[
        get_task_composite_id(project_id, current_task.id)
      ] ?? []
    const config = all_configs.find((c) => c.id === selected_run_config_id)
    return isMcpRunConfig(config?.run_config_properties)
  })()

  $: selected_mcp_config = (() => {
    if (!is_mcp || !selected_run_config_id || !current_task?.id) return null
    const all_configs =
      $run_configs_by_task_composite_id[
        get_task_composite_id(project_id, current_task.id)
      ] ?? []
    return all_configs.find((c) => c.id === selected_run_config_id) ?? null
  })()

  // If requires_structured_output, update structured_output_mode when model changes
  // We test each model in our known model list, so a smart default is selected automatically.
  function update_structured_output_mode_if_needed(
    model_name: string,
    provider: string,
    available_models: AvailableModels[],
  ) {
    if (requires_structured_output) {
      const model_details = available_model_details(
        model_name,
        provider,
        available_models,
      )
      const new_mode = model_details?.structured_output_mode || "default"
      if (new_mode !== structured_output_mode) {
        structured_output_mode = new_mode
        return true
      }
    }
  }

  function update_thinking_level_if_needed() {
    const new_level = current_model_details?.default_thinking_level ?? null
    if (new_level !== thinking_level) {
      thinking_level = new_level
      return true
    }
    return false
  }

  // When a run config is selected, update the current run options to match the selected config
  let prior_selected_run_config_id: string | null = null
  let run_config_just_loaded = false

  // Set every run option from a config's properties. `run_config_just_loaded`
  // stops the next reactive pass from re-applying model defaults over the values
  // we just set (see update_for_state_changes).
  // Exported as well: initial_run_config_properties below seeds only once, so a
  // caller whose dialog stays mounted between opens calls this to show what it
  // last committed rather than the edits an abandoned visit left behind.
  export function apply_run_config_properties(
    config_properties: KilnAgentRunConfigProperties,
  ) {
    model =
      config_properties.model_provider_name + "/" + config_properties.model_name
    prompt_method = config_properties.prompt_id
    const split = split_tool_and_skill_ids(
      config_properties.tools_config?.tools ?? [],
    )
    tools = split.tool_ids
    skills = split.skill_ids
    temperature = config_properties.temperature
    top_p = config_properties.top_p
    structured_output_mode = config_properties.structured_output_mode
    thinking_level = config_properties.thinking_level ?? null
    input_transform = config_properties.input_transform ?? null
    run_config_just_loaded = true
  }

  // Seed the run options from a config the caller already ran with, rather than
  // from the model defaults — e.g. a multi-step flow that remounts this
  // component between steps and wants the user's earlier picks (including
  // advanced options) to survive. Applied once, when it first arrives: callers
  // typically resolve it asynchronously, after this component has mounted.
  let applied_initial_run_config = false
  $: if (initial_run_config_properties && !applied_initial_run_config) {
    applied_initial_run_config = true
    apply_run_config_properties(initial_run_config_properties)
  }

  async function update_current_run_options_for_selected_run_config() {
    // Only run once immediately after a run config selection, not every reactive update
    if (prior_selected_run_config_id === selected_run_config_id) {
      return
    }
    prior_selected_run_config_id = selected_run_config_id

    const selected_run_config = await get_selected_run_config()
    if (!selected_run_config || selected_run_config === "custom") {
      // No need to update selected_run_config_id, it's already custom or unset
      return
    }
    if (isMcpRunConfig(selected_run_config.run_config_properties)) {
      return
    }

    const config_properties = selected_run_config.run_config_properties
    if (!isKilnAgentRunConfig(config_properties)) {
      return
    }
    apply_run_config_properties(config_properties)
  }

  // Main reactive statement. This class is a bit wild, as many changes are circular.
  // Example: changing the run config will update model, but selecting model will jump back to "custom" run config (or finetune run config).
  // These are legit desired behaviour: respect the user's last selection, and make the rest consistent. But it makes updating the state a bit tricky.
  // Make 1 big reactive statement to update the state. Then we debounce it to avoid excessive updates.
  // Test cases if you edit (including a page reload version of each test):
  // 1. Select a fine-tune model, it's run config should be automatically selected and the RC's values filled
  // 2. Select a legacy fine-tune model (no run config baked in), it's prompt should be selected and RC stays custom
  // 3. Select a saved run config, should set all fields to the saved config's values
  // 4. Change any field after setting a run config, should deselect the run config to "custom"
  $: void (model,
  prompt_method,
  temperature,
  top_p,
  structured_output_mode,
  thinking_level,
  input_transform,
  tools,
  skills,
  $available_models,
  selected_run_config_id,
  debounce_update_for_state_changes())

  // Since some changes can make many other fields change (eg run config), we debounce the updates to avoid excessive updates.
  // Just mark as dirty, and run again only once, after the update is done.
  // Knowing only 1 is called in parallel also makes it simpler to reason about.
  let running: boolean = false
  let run_again: boolean = false
  async function debounce_update_for_state_changes() {
    if (running) {
      run_again = true
      return
    }
    running = true
    await tick()
    try {
      await update_for_state_changes()
    } finally {
      running = false
      if (run_again) {
        run_again = false
        debounce_update_for_state_changes()
      }
    }
  }

  // Progress step by step, stopping if any step asks to. It could be missing data, and the remaining steps aren't valid.
  async function update_for_state_changes() {
    // Apply URL-driven intent first so it is not blocked by model loading.
    apply_pending_tool_selection_if_needed()
    await apply_pending_run_config_if_needed()

    // All steps below need available_models to be loaded. Don't set run_again as it would be tight loop, we're reactive to $available_models.
    if ($available_models.length === 0) {
      return
    }

    // Check if they selected a new model, in which case we want to update the run config to the finetune run config if needed
    const model_changed = process_model_change()

    // Only apply model-default overrides for user-initiated model changes.
    // When a run config was just loaded (which may have changed the model),
    // the run config's values should take priority.
    if (model_changed && !run_config_just_loaded) {
      update_structured_output_mode_if_needed(
        model_name,
        provider,
        $available_models,
      )
      update_thinking_level_if_needed()
    }
    run_config_just_loaded = false

    // Update all the run options if they have changed the run config
    await update_current_run_options_for_selected_run_config()

    // deselect the run config if they have changed any run options to not match the selected run config
    await reset_to_custom_options_if_needed()
  }

  let last_applied_pending_run_config_id: string | null = null
  async function apply_pending_run_config_if_needed() {
    if (pending_run_config_id === last_applied_pending_run_config_id) {
      return
    }
    if (!pending_run_config_id || !current_task?.id) {
      return
    }
    await load_task_run_configs(project_id, current_task.id)
    const all_configs =
      $run_configs_by_task_composite_id[
        get_task_composite_id(project_id, current_task.id)
      ] ?? []
    const exists = all_configs.find((c) => c.id === pending_run_config_id)
    if (exists) {
      selected_run_config_id = pending_run_config_id
    }
    // Apply each deep-linked run config only once (tracked by ID). Otherwise it
    // keeps re-forcing selected_run_config_id back on every reactive pass,
    // fighting the user's manual changes (e.g. changing the model) and causing
    // an infinite loop. Tracking the ID (rather than a boolean) still lets a new
    // deep link applied within the same session take effect.
    last_applied_pending_run_config_id = pending_run_config_id
  }

  let pending_tool_selection_applied = false
  function apply_pending_tool_selection_if_needed() {
    if (pending_tool_selection_applied) {
      return
    }
    if (!pending_tool_id) {
      return
    }

    if (selected_run_config_id !== "custom") {
      selected_run_config_id = "custom"
      selected_model_specific_run_config_id = null
    }
    pending_tool_selection_applied = true
  }

  let prior_model: string | null = null
  function process_model_change(): boolean {
    // only run once immediately after a model change, not every reactive update
    if (prior_model === model) {
      return false
    }
    prior_model = model

    // Special case on model change: if the model says it has a model-specific run config, select that run config.
    // Currently used by fine-tuned models which need to be called like they are trained.
    const model_details = available_model_details(
      model_name,
      provider,
      $available_models,
    )
    if (model_details?.model_specific_run_config) {
      if (!run_config_just_loaded) {
        selected_run_config_id = model_details.model_specific_run_config
      }
      selected_model_specific_run_config_id =
        model_details.model_specific_run_config
    } else {
      selected_model_specific_run_config_id = null
    }
    return true
  }

  async function reset_to_custom_options_if_needed() {
    const selected_run_config = await get_selected_run_config()
    if (!selected_run_config || selected_run_config === "custom") {
      return
    }
    if (isMcpRunConfig(selected_run_config.run_config_properties)) {
      return
    }

    const config_properties = selected_run_config.run_config_properties
    if (!isKilnAgentRunConfig(config_properties)) {
      return
    }

    // Check if any values have changed from the saved config properties
    let model_changed = false
    let provider_changed = false
    let prompt_changed = false
    const current_model_name = model ? model.split("/").slice(1).join("/") : ""
    const current_provider_name = model ? model.split("/")[0] : ""
    model_changed = config_properties.model_name !== current_model_name
    provider_changed =
      config_properties.model_provider_name !== current_provider_name
    prompt_changed = config_properties.prompt_id !== prompt_method

    // Legacy models can be "unknown". Don't consider those as mismatches.
    const output_mode_mismatch =
      config_properties.structured_output_mode !== "unknown" &&
      config_properties.structured_output_mode !== structured_output_mode

    if (
      model_changed ||
      provider_changed ||
      prompt_changed ||
      config_properties.temperature !== temperature ||
      config_properties.top_p !== top_p ||
      (config_properties.thinking_level ?? null) !== thinking_level ||
      output_mode_mismatch ||
      !arrays_equal(config_properties.tools_config?.tools ?? [], [
        ...tools,
        ...skills,
      ]) ||
      !inputTransformsEqual(
        config_properties.input_transform ?? null,
        input_transform,
      )
    ) {
      // The user has changed something, so deselect the run config - it no longer matches the selected run config
      selected_run_config_id = "custom"
    }
  }

  // Helper function to convert run options to server run_config_properties format
  export function run_options_as_run_config_properties(): RunConfigProperties {
    if (selected_mcp_config?.run_config_properties) {
      return selected_mcp_config.run_config_properties
    }
    const all_tool_ids = [...tools, ...skills]
    return {
      type: "kiln_agent",
      model_name: model_name,
      // @ts-expect-error server will catch if enum is not valid
      model_provider_name: provider,
      prompt_id: prompt_method,
      temperature: temperature,
      top_p: top_p,
      structured_output_mode: structured_output_mode,
      thinking_level: thinking_level,
      input_transform: input_transform,
      tools_config: {
        tools: all_tool_ids,
      },
    }
  }

  export async function save_new_run_config(): Promise<TaskRunConfig> {
    if (!current_task?.id) {
      throw new Error("Cannot save run config: no task selected")
    }
    // Regenerate if the name field is hidden, to avoid collisions on repeated saves.
    // If visible, we respect the current value to avoid overwriting user input.
    if (!show_name_field) {
      run_config_name = generate_memorable_name()
    }
    run_config_name = normalize_filename_string(run_config_name)
    const saved_config = await save_new_task_run_config(
      project_id,
      current_task.id,
      run_options_as_run_config_properties(),
      run_config_name,
    )
    // Reload prompts to update the dropdown with the new static prompt that is made from saving a new run config
    await load_task_prompts(project_id, current_task.id, true)
    if (!saved_config || !saved_config.id) {
      throw new Error("Saved config id not found")
    }
    return saved_config
  }

  async function get_selected_run_config(): Promise<
    TaskRunConfig | "custom" | null
  > {
    if (!current_task?.id) {
      return null
    }
    // Make sure the task run configs are loaded, will be quick if they already are
    await load_task_run_configs(project_id, current_task.id)

    // Map selected ID back to TaskRunConfig object
    if (!selected_run_config_id) {
      return null
    } else if (selected_run_config_id === "custom") {
      return "custom"
    } else {
      // Find the config by ID
      const all_configs =
        $run_configs_by_task_composite_id[
          get_task_composite_id(project_id, current_task.id)
        ] ?? []
      let run_config = all_configs.find(
        (config) => config.id === selected_run_config_id,
      )
      return run_config ?? "custom"
    }
  }

  // Expose methods for run parent component
  export function get_selected_model(): string | null {
    return model_dropdown ? model_dropdown.get_selected_model() : null
  }

  export function clear_run_options_errors() {
    set_default_error = null
  }

  export function clear_model_dropdown_error() {
    model_dropdown_error_message = null
  }

  export function set_model_dropdown_error(message: string) {
    model_dropdown_error_message = message
  }

  export function get_prompt_method(): string {
    return prompt_method
  }

  export function get_tools(): string[] {
    return [...tools]
  }

  export function clear_tools() {
    tools = []
  }

  export function get_skills(): string[] {
    return [...skills]
  }

  export function clear_skills() {
    skills = []
  }

  // Put the options a user edits here back to where a fresh component would
  // have them for the model now selected. For a caller whose dialog stays
  // mounted between opens with no config to reseed from: an abandoned visit
  // must not leave its edits on the next one. The model and the prompt are the
  // caller's to set, so they are left alone.
  export function reset_run_options() {
    tools = []
    skills = []
    temperature = DEFAULT_RUN_OPTIONS.temperature
    top_p = DEFAULT_RUN_OPTIONS.top_p
    input_transform = DEFAULT_RUN_OPTIONS.input_transform
    structured_output_mode = DEFAULT_RUN_OPTIONS.structured_output_mode
    // The selected model's own defaults, not the pre-model placeholders: the
    // model is not what the user abandoned, so it keeps the settings it earned.
    update_structured_output_mode_if_needed(
      model_name,
      provider,
      $available_models,
    )
    update_thinking_level_if_needed()
  }
</script>

<div class="w-full flex flex-col gap-4">
  {#if show_name_field}
    <FormElement
      label="Name"
      id="run_config_name"
      bind:value={run_config_name}
      max_length={120}
      validator={filename_string_validator_default}
    />
  {/if}
  {#if is_mcp}
    {#if selected_mcp_config}
      <McpRunConfigPanel run_config={selected_mcp_config} {project_id} />
    {/if}
  {:else}
    {#if !hide_model_selector}
      <AvailableModelsDropdown
        task_id={current_task?.id ?? null}
        label={model_label}
        info_description={model_info_description}
        bind:model
        settings={updated_model_dropdown_settings}
        bind:error_message={model_dropdown_error_message}
        bind:this={model_dropdown}
      />
    {/if}
    {#if !hide_prompt_selector}
      <PromptTypeSelector
        bind:prompt_method
        info_description="Choose a prompt. Learn more on the 'Prompts' tab."
        bind:linked_model_selection={model}
        {project_id}
        task_id={current_task?.id ?? null}
      />
    {/if}
    {#if !show_tools_selector_in_advanced}
      {#if !hide_tools_selector}
        <ToolsSelector
          bind:tools
          {project_id}
          task_id={current_task?.id ?? null}
          settings={tools_selector_settings}
          {pending_tool_id}
        />
        <SkillsSelector
          bind:skills
          {project_id}
          task_id={current_task?.id ?? null}
          settings={skills_selector_settings}
          {pending_skill_id}
        />
      {/if}
      <Collapse title="Advanced Options">
        <slot name="advanced" />
        <AdvancedRunOptions
          bind:temperature
          bind:top_p
          bind:structured_output_mode
          bind:thinking_level
          bind:input_transform
          available_thinking_levels={current_model_details?.available_thinking_levels ??
            null}
          has_structured_output={requires_structured_output}
        />
      </Collapse>
    {:else}
      <Collapse title="Advanced Options">
        <slot name="advanced" />
        {#if !hide_tools_selector}
          <ToolsSelector
            bind:tools
            {project_id}
            task_id={current_task?.id ?? null}
            settings={tools_selector_settings}
            {pending_tool_id}
          />
          <SkillsSelector
            bind:skills
            {project_id}
            task_id={current_task?.id ?? null}
            settings={skills_selector_settings}
          />
        {/if}
        <AdvancedRunOptions
          bind:temperature
          bind:top_p
          bind:structured_output_mode
          bind:thinking_level
          bind:input_transform
          available_thinking_levels={current_model_details?.available_thinking_levels ??
            null}
          has_structured_output={requires_structured_output}
        />
      </Collapse>
    {/if}
  {/if}
</div>
