<script lang="ts">
  import {
    available_models,
    load_available_models,
    available_model_details,
    ui_state,
    provider_name_from_id,
    model_name as model_name_from_id,
    model_info,
    load_model_info,
  } from "$lib/stores"
  import {
    addRecentModel,
    type RecentModel,
    recent_model_store,
  } from "$lib/stores/recent_model_store"
  import type { AvailableModels, ProviderModels } from "$lib/types"
  import { onMount } from "svelte"
  import FormElement, {
    type InlineAction,
  } from "$lib/utils/form_element.svelte"
  import Warning from "$lib/ui/warning.svelte"
  import type { OptionGroup, Option } from "$lib/ui/fancy_select_types"
  import { mime_type_to_string } from "$lib/utils/formatters"
  import {
    show_suggested_advisory,
    type ModelDropdownSettings,
  } from "./model_dropdown_settings"

  const LOGPROBS_WARNING =
    "This model does not support logprobs. It will likely fail when running a G-eval or other logprob queries."

  export let task_id: string | null = null
  export let model: string | null = $ui_state.selected_model
  export let label: string = "Model"
  export let description: string | undefined = undefined
  export let info_description: string | undefined = undefined
  export let settings: Partial<ModelDropdownSettings> = {}
  export let error_message: string | null = null
  export let inline_action: InlineAction | null = null
  export let optional: boolean = false
  export let hide_optional_badge: boolean = false
  export let empty_label: string = "Select a model"
  // Empty-dropdown affordance (fancy_select's built-in empty state). An
  // empty model list means no connected provider offers a usable model, so
  // every picker names the way out — a same-tab link, because
  // connecting clears the models cache in this tab and the return trip
  // refetches it (a new tab would strand this tab's stale cache).
  const empty_state_message = "No models available"
  const empty_state_subtitle = "Connect an AI provider"
  const empty_state_link = "/settings/providers"

  let default_model_dropdown_settings: ModelDropdownSettings = {
    filter_models_predicate: (_) => true,
    requires_structured_output: false,
    requires_data_gen: false,
    requires_logprobs: false,
    requires_uncensored_data_gen: false,
    requires_doc_extraction: false,
    requires_tool_support: false,
    suggested_mode: null,
  }
  $: model_dropdown_settings = {
    ...default_model_dropdown_settings,
    ...settings,
  }
  $: $ui_state.selected_model = model
  $: model_options = format_model_options(
    $available_models || [],
    model_dropdown_settings,
    task_id,
    $recent_model_store,
    $model_info,
  )

  // Export the parsed model name and provider name
  export let model_name: string | null = null
  export let provider_name: string | null = null
  let provider_display_name: string | null = null

  // Map of model value to provider display name, to correctly identify custom providers
  let model_value_to_provider_name: Map<string, string> = new Map()
  $: get_model_provider(model, model_value_to_provider_name)

  function get_model_provider(
    model_provider: string | null,
    model_value_to_provider_name_map: Map<string, string>,
  ) {
    if (!model_provider) {
      model_name = null
      provider_name = null
      provider_display_name = null
      return
    }
    model_name = model_provider.split("/").slice(1).join("/")
    provider_name = model_provider.split("/")[0]
    // Use the map to get the provider display name (for custom providers)
    provider_display_name =
      model_value_to_provider_name_map.get(model_provider) || null
  }

  $: addRecentModel(model_name, provider_name, provider_display_name)

  onMount(async () => {
    await load_available_models()
    await load_model_info()
  })

  let unsupported_models: Option[] = []
  let untested_models: Option[] = []
  let deprecated_models: Option[] = []
  let previous_model: string | null = model

  function get_model_warning(selected: string): string | null {
    if (
      unsupported_models.some(
        (m) => m.value === selected && settings.requires_logprobs,
      )
    ) {
      return LOGPROBS_WARNING
    }

    return null
  }

  function confirm_model_select(event: Event) {
    const select = event.target as HTMLSelectElement
    const selected = select.value
    const warning = get_model_warning(selected)
    if (warning && !confirm(warning)) {
      select.value = previous_model ?? ""
      model = previous_model
      return
    }
    previous_model = selected
  }

  function format_model_options(
    providers: AvailableModels[],
    model_dropdown_settings: ModelDropdownSettings,
    task_id: string | null,
    recent_models: RecentModel[],
    model_data: ProviderModels | null,
  ): OptionGroup[] {
    let options: OptionGroup[] = []
    unsupported_models = []
    untested_models = []
    deprecated_models = []
    // Clear and rebuild the map
    model_value_to_provider_name = new Map()

    // Recent models section
    let recent_model_list: Option[] = []
    for (const recent_model of recent_models) {
      const model_details = available_model_details(
        recent_model.model_id,
        recent_model.model_provider,
        providers,
        recent_model.provider_display_name,
      )
      if (
        !model_details ||
        !model_dropdown_settings.filter_models_predicate(model_details) ||
        model_details.deprecated
      ) {
        continue
      }
      // Use the stored provider_display_name for custom providers, otherwise fall back to provider_name_from_id
      const display_name =
        recent_model.provider_display_name ||
        provider_name_from_id(recent_model.model_provider)
      recent_model_list.push({
        value: recent_model.model_provider + "/" + recent_model.model_id,
        label:
          model_name_from_id(recent_model.model_id, model_data) +
          " (" +
          display_name +
          ")",
      })
    }
    if (recent_model_list.length > 0) {
      options.push({
        label: "Recent Models",
        options: recent_model_list,
      })
    }

    for (const provider of providers) {
      let model_list: Option[] = []
      for (const model of provider.models) {
        if (!model_dropdown_settings.filter_models_predicate(model)) {
          continue
        }
        // Exclude models that are not available for the current task
        if (
          model &&
          model.task_filter &&
          task_id &&
          !model.task_filter.includes(task_id)
        ) {
          continue
        }

        let id = provider.provider_id + "/" + model.id
        let long_label = provider.provider_name + " / " + model.name
        if (model.untested_model) {
          untested_models.push({
            value: id,
            label: long_label,
          })
          model_value_to_provider_name.set(id, provider.provider_name)
          continue
        }
        if (model.deprecated) {
          deprecated_models.push({
            value: id,
            label: long_label,
            disabled: true,
          })
          model_value_to_provider_name.set(id, provider.provider_name)
          continue
        }

        const unsupported =
          (model_dropdown_settings.requires_data_gen &&
            !model.supports_data_gen) ||
          (model_dropdown_settings.requires_structured_output &&
            !model.supports_structured_output) ||
          (model_dropdown_settings.requires_logprobs &&
            !model.supports_logprobs) ||
          (model_dropdown_settings.requires_uncensored_data_gen &&
            !model.uncensored) ||
          (model_dropdown_settings.requires_doc_extraction &&
            !model.supports_doc_extraction) ||
          (model_dropdown_settings.requires_tool_support &&
            !model.supports_function_calling)
        if (unsupported) {
          unsupported_models.push({
            value: id,
            label: long_label,
          })
          model_value_to_provider_name.set(id, provider.provider_name)
          continue
        }

        let badge: string | undefined = undefined
        if (
          (settings.suggested_mode === "data_gen" &&
            model.suggested_for_data_gen) ||
          (settings.suggested_mode === "evals" && model.suggested_for_evals) ||
          (settings.suggested_mode === "uncensored_data_gen" &&
            model.suggested_for_uncensored_data_gen) ||
          (settings.suggested_mode === "doc_extraction" &&
            model.suggested_for_doc_extraction) ||
          (settings.suggested_mode === "synthetic_user" &&
            model.suggested_for_synthetic_user)
        ) {
          badge = "Recommended"
        }
        model_list.push({
          value: id,
          label: model.name,
          badge: badge,
        })
        model_value_to_provider_name.set(id, provider.provider_name)
      }
      if (model_list.length > 0) {
        options.push({
          label: provider.provider_name,
          options: model_list,
        })
      }
    }

    if (untested_models.length > 0) {
      options.push({
        label: "Untested Models",
        options: untested_models,
      })
    }

    if (unsupported_models.length > 0) {
      let not_recommended_label = "Not Recommended"
      if (model_dropdown_settings.requires_uncensored_data_gen) {
        not_recommended_label =
          "Not Recommended - Uncensored Data Gen Not Supported"
      } else if (model_dropdown_settings.requires_tool_support) {
        not_recommended_label = "Not Recommended - Tool Calling Not Supported"
      } else if (model_dropdown_settings.requires_data_gen) {
        not_recommended_label = "Not Recommended - Data Gen Not Supported"
      } else if (model_dropdown_settings.requires_structured_output) {
        not_recommended_label = "Not Recommended - Structured Output Fails"
      } else if (model_dropdown_settings.requires_logprobs) {
        not_recommended_label = "Not Recommended - Logprobs Not Supported"
      }

      options.push({
        label: not_recommended_label,
        options: unsupported_models,
      })
    }

    if (deprecated_models.length > 0) {
      options.push({
        label: "Deprecated Models",
        options: deprecated_models,
      })
    }

    if (settings.suggested_mode === "doc_extraction") {
      for (const option_group of options) {
        for (const option of option_group.options) {
          if (typeof option.value !== "string") {
            continue
          }
          const slash_index = option.value.indexOf("/")
          const option_provider_name = option.value.substring(0, slash_index)
          const option_model_name = option.value.substring(slash_index + 1)
          const mime_types =
            available_model_details(
              option_model_name,
              option_provider_name,
              providers,
            )?.multimodal_mime_types || []

          if (mime_types.length) {
            const formatted_mime_types = mime_types.map((mime_type) =>
              mime_type_to_string(mime_type),
            )
            option.description = "Supports " + formatted_mime_types.join(", ")
          }
        }
      }
    }

    return options
  }

  // Extra check to make sure the model is available to use
  export function get_selected_model(): string | null {
    for (const provider of model_options) {
      if (provider.options.find((m) => m.value === model && !m.disabled)) {
        return model
      }
    }
    return null
  }

  $: selected_model_untested = untested_models.find((m) => m.value === model)
  $: selected_model_unsupported = unsupported_models.find(
    (m) => m.value === model,
  )
  $: selected_model_details = available_model_details(
    model_name,
    provider_name,
    $available_models,
  )
  $: selected_model_deprecated = selected_model_details?.deprecated ?? false

  // The model list is what makes the suggested-for-X flags below knowable:
  // while it is still loading every model reads as not-suggested, so a
  // suggested one cannot be told apart from a genuinely unsuggested one. An
  // empty list stays unknown — the fetch found or returned nothing, and the
  // dropdown shows its own empty state instead.
  $: suggestion_known = ($available_models || []).length > 0

  $: selected_model_suggested_data_gen =
    selected_model_details?.suggested_for_data_gen || false

  $: selected_model_suggested_uncensored_data_gen =
    selected_model_details?.suggested_for_uncensored_data_gen || false

  $: selected_model_suggested_evals =
    selected_model_details?.suggested_for_evals || false

  $: selected_model_suggested_doc_extraction =
    selected_model_details?.suggested_for_doc_extraction || false

  $: selected_model_suggested_synthetic_user =
    selected_model_details?.suggested_for_synthetic_user || false
</script>

<div class="flex flex-col gap-2">
  <FormElement
    {label}
    {description}
    {info_description}
    {inline_action}
    {optional}
    {hide_optional_badge}
    {empty_label}
    {empty_state_message}
    {empty_state_subtitle}
    {empty_state_link}
    bind:value={model}
    id="model"
    inputType="fancy_select"
    on_select={confirm_model_select}
    bind:error_message
    fancy_select_options={model_options}
  />

  {#if selected_model_untested}
    <Warning
      warning_message="This model has not been tested with Kiln. It may not work as expected."
    />
  {:else if selected_model_deprecated}
    <Warning
      warning_message="This model is deprecated and can no longer be used. Please select a different model or provider."
    />
  {:else if selected_model_unsupported}
    {#if model_dropdown_settings.requires_uncensored_data_gen}
      <Warning
        warning_message="The current data gen template works best with uncensored models like Grok. This model may refuse to generate data for sensitive topics."
      />
    {:else if model_dropdown_settings.requires_tool_support}
      <Warning warning_message="This model does not support tool calling." />
    {:else if model_dropdown_settings.requires_data_gen}
      <Warning
        warning_message="This model is not recommended for use with data generation. It's known to generate incorrect data."
      />
    {:else if model_dropdown_settings.requires_logprobs}
      <Warning large_icon warning_message={LOGPROBS_WARNING} />
    {:else if model_dropdown_settings.requires_structured_output}
      <Warning
        warning_message="This model is not recommended for use with tasks requiring structured output. It fails to consistently return structured data."
      />
    {/if}
  {:else if settings.suggested_mode === "data_gen"}
    {#if show_suggested_advisory(!!model, suggestion_known)}
      <Warning
        warning_icon={!model
          ? "info"
          : selected_model_suggested_data_gen
            ? "check"
            : "exclaim"}
        warning_color={!model
          ? "gray"
          : selected_model_suggested_data_gen
            ? "success"
            : "warning"}
        warning_message={`For data gen we suggest using one of the models marked "Recommended" in the dropdown.`}
      />
    {/if}
  {:else if settings.suggested_mode === "uncensored_data_gen"}
    {#if show_suggested_advisory(!!model, suggestion_known)}
      <Warning
        warning_icon={!model
          ? "info"
          : selected_model_suggested_uncensored_data_gen
            ? "check"
            : "exclaim"}
        warning_color={!model
          ? "gray"
          : selected_model_suggested_uncensored_data_gen
            ? "success"
            : "warning"}
        warning_message={`For this data gen template we suggest using one of the models marked "Recommended" in the dropdown.`}
      />
    {/if}
  {:else if settings.suggested_mode === "evals"}
    {#if show_suggested_advisory(!!model, suggestion_known)}
      <Warning
        warning_icon={!model
          ? "info"
          : selected_model_suggested_evals
            ? "check"
            : "exclaim"}
        warning_color={!model
          ? "gray"
          : selected_model_suggested_evals
            ? "success"
            : "warning"}
        warning_message={`For evals we suggest using one of the models marked "Recommended" in the dropdown.`}
      />
    {/if}
  {:else if settings.suggested_mode === "doc_extraction"}
    {#if show_suggested_advisory(!!model, suggestion_known)}
      <Warning
        warning_icon={!model
          ? "info"
          : selected_model_suggested_doc_extraction
            ? "check"
            : "exclaim"}
        warning_color={!model
          ? "gray"
          : selected_model_suggested_doc_extraction
            ? "success"
            : "warning"}
        warning_message={`For doc extraction we suggest using one of the models marked "Recommended" in the dropdown.`}
      />
    {/if}
  {:else if settings.suggested_mode === "synthetic_user"}
    {#if show_suggested_advisory(!!model, suggestion_known)}
      <Warning
        warning_icon={!model
          ? "info"
          : selected_model_suggested_synthetic_user
            ? "check"
            : "exclaim"}
        warning_color={!model
          ? "gray"
          : selected_model_suggested_synthetic_user
            ? "success"
            : "warning"}
        warning_message={`For the simulated user we suggest using one of the models marked "Recommended" in the dropdown.`}
      />
    {/if}
  {/if}
</div>
