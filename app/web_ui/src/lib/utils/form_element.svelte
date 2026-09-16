<script context="module" lang="ts">
  export type InlineAction = {
    handler: () => void
    label: string
    loading?: boolean
    loading_text?: string
    disabled?: boolean
  }

  export type RadioOption = {
    value: string
    label: string
    description?: string
  }
</script>

<script lang="ts">
  import { onMount, onDestroy, getContext } from "svelte"
  import InfoTooltip from "$lib/ui/info_tooltip.svelte"
  import FancySelect from "$lib/ui/fancy_select.svelte"
  import type { OptionGroup } from "$lib/ui/fancy_select_types"

  export let inputType:
    | "input"
    | "input_number"
    | "textarea"
    | "select"
    | "fancy_select"
    | "multi_select"
    | "header_only"
    | "checkbox"
    | "radio" = "input"
  export let id: string
  export let label: string = ""
  export let value: unknown
  export let description: string = ""
  export let info_description: string = ""
  export let placeholder: string | null = null
  export let optional: boolean = false
  export let hide_optional_badge: boolean = false
  export let max_length: number | null = null
  export let error_message: string | null = null // start null because they haven't had a chance to edit it yet
  export let light_label: boolean = false // styling
  export let select_options: [unknown, string][] = []
  export let select_options_grouped: [string, [unknown, string][]][] = []
  export let fancy_select_options: OptionGroup[] = []
  export let radio_options: RadioOption[] = []
  export let on_select: (e: Event) => void = () => {}
  export let disabled: boolean = false
  export let info_msg: string | null = null
  export let height: "base" | "compact" | "medium" | "large" | "xl" = "base"
  export let empty_label: string = "Select an option"
  export let empty_state_message: string = "No options available"
  export let empty_state_subtitle: string | null = null
  export let empty_state_link: string | null = null
  export let inline_action: InlineAction | null = null
  export let aria_label: string | null = null
  // The id of an element elsewhere on the page that describes this field — a
  // live hint or an advisory the control itself does not own. Null for every
  // caller that has nothing to point at, which leaves the attribute off.
  export let aria_describedby: string | null = null
  export let hide_label: boolean = false
  export let min: number | null = null
  export let max: number | null = null
  export let on_radio_change: (() => void) | null = null

  function is_empty(value: unknown): boolean {
    if (value === null || value === undefined) {
      return true
    }
    if (typeof value === "string") {
      return value.length === 0
    }
    return false
  }

  // Export to let parent redefine this. This is a basic "Optional" and max length check
  export let validator: (value: unknown) => string | null = () => {
    if (inputType === "header_only") {
      return null
    }
    if (!optional && is_empty(value)) {
      return '"' + label + '" is required'
    }
    if (max_length && typeof value === "string" && value.length > max_length) {
      return (
        '"' +
        label +
        '" is too long. Max length is ' +
        max_length +
        " characters."
      )
    }
    return null
  }
  // Shorter error message that appears in a badge over the input
  let inline_error: string | null = null
  let initial_run = true
  $: {
    if (inputType === "header_only") {
      inline_error = null
    } else if (initial_run) {
      initial_run = false
    } else if (!optional && is_empty(value)) {
      inline_error = "Required"
    } else if (
      max_length &&
      typeof value === "string" &&
      value.length > max_length
    ) {
      inline_error = "" + value.length + "/" + max_length
    } else {
      inline_error = null
    }
  }

  export function run_validator() {
    const error = validator(value)
    error_message = error
  }

  // run validator after value change
  let initialized = false
  function run_validator_on_change(_: unknown) {
    if (initialized) {
      run_validator()
    }
  }
  $: run_validator_on_change(value)

  const formContainer = getContext<{
    registerFormElement: (validator: {
      run_validator: () => void
    }) => () => void
  } | null>("form_container")

  let unregister: (() => void) | null = null

  onMount(() => {
    initialized = true
    if (formContainer) {
      unregister = formContainer.registerFormElement({ run_validator })
    }
  })

  onDestroy(() => {
    if (unregister) {
      unregister()
    }
  })

  // Little dance to keep type checker happy
  function handleCheckboxChange(event: Event) {
    const target = event.target as HTMLInputElement
    if (target) value = target.checked
  }

  const height_class = {
    base: "h-18",
    compact: "h-24",
    medium: "h-36",
    large: "h-60",
    xl: "h-96",
  }
</script>

<div>
  <div class="flex flex-row items-center gap-2 pb-[4px]">
    {#if inputType === "checkbox"}
      <input
        type="checkbox"
        {id}
        class="checkbox"
        checked={value ? true : false}
        on:change={handleCheckboxChange}
        aria-label={aria_label || label}
        {disabled}
      />
    {/if}
    {#if label || inline_action || info_description || error_message || description}
      <label
        for={id}
        class="text-sm font-medium text-left flex flex-col gap-1 w-full"
      >
        {#if label || inline_action || info_description || error_message}
          <div class="flex flex-row items-center">
            {#if label && !hide_label}
              <span class={light_label ? "text-xs text-gray-500 h-4" : ""}
                >{label}</span
              >
              <slot name="label_suffix" />
              <span class="grow"></span>
              <span class="pl-1 text-xs text-gray-500 flex-none"
                >{info_msg ||
                  (optional && !hide_optional_badge ? "Optional" : "")}</span
              >
            {:else}
              <span class="grow"></span>
            {/if}
            {#if inline_action}
              <button
                type="button"
                class="link ml-4 text-xs text-gray-500 hover:text-gray-700 disabled:opacity-60 disabled:no-underline disabled:pointer-events-none"
                disabled={inline_action.loading || inline_action.disabled}
                on:click|stopPropagation={inline_action.handler}
                >{inline_action.loading
                  ? inline_action.loading_text ?? "Loading..."
                  : inline_action.label}</button
              >
            {/if}
            {#if info_description}
              <div class="text-gray-500 {light_label ? 'h-4 mt-[-4px]' : ''}">
                <InfoTooltip tooltip_text={info_description} />
              </div>
            {/if}
            {#if error_message}
              <span class="text-error">
                <InfoTooltip tooltip_text={error_message} symbol="exclaim" />
              </span>
            {/if}
          </div>
        {/if}
        {#if description}
          <div class="text-xs text-gray-500">
            {description}
          </div>
        {/if}
      </label>
    {/if}
  </div>
  <div class="relative">
    {#if inputType === "textarea"}
      <textarea
        aria-label={aria_label || label}
        aria-describedby={aria_describedby}
        placeholder={error_message || placeholder || label}
        {id}
        class="textarea text-base textarea-bordered w-full {height_class[
          height
        ]} whitespace-pre-wrap break-words text-left align-top
       {error_message || inline_error ? 'textarea-error' : ''}"
        bind:value
        autocomplete="off"
        data-op-ignore="true"
        {disabled}
      />
    {:else if inputType === "input"}
      <input
        aria-label={aria_label || label}
        aria-describedby={aria_describedby}
        type="text"
        placeholder={error_message || placeholder || label}
        {id}
        class="input text-base input-bordered w-full font-base {error_message ||
        inline_error
          ? 'input-error'
          : ''}"
        bind:value
        autocomplete="off"
        data-op-ignore="true"
        {disabled}
      />
    {:else if inputType === "input_number"}
      <input
        aria-label={aria_label || label}
        aria-describedby={aria_describedby}
        type="number"
        placeholder={error_message || placeholder || label}
        {id}
        class="input text-base input-bordered w-full font-base {error_message ||
        inline_error
          ? 'input-error'
          : ''}"
        bind:value
        autocomplete="off"
        data-op-ignore="true"
        {disabled}
        {min}
        {max}
      />
    {:else if inputType === "select"}
      <select
        aria-label={aria_label || label}
        aria-describedby={aria_describedby}
        {id}
        class="select select-bordered w-full {error_message || inline_error
          ? 'select-error'
          : ''}"
        bind:value
        on:input={on_select}
        {disabled}
      >
        {#if select_options_grouped.length > 0}
          {#each select_options_grouped as group}
            <optgroup label={group[0]}>
              {#each group[1] as option}
                <option
                  value={option[0]}
                  disabled={("" + option[0]).startsWith("disabled")}
                  selected={option[0] === value}>{option[1]}</option
                >
              {/each}
            </optgroup>
          {/each}
        {:else}
          {#each select_options as option}
            <option
              value={option[0]}
              disabled={("" + option[0]).startsWith("disabled")}
              selected={option[0] === value}>{option[1]}</option
            >
          {/each}
        {/if}
      </select>
    {:else if inputType === "fancy_select" || inputType === "multi_select"}
      <FancySelect
        aria_label={aria_label || label}
        bind:options={fancy_select_options}
        bind:selected={value}
        multi_select={inputType === "multi_select"}
        error_outline={!!error_message}
        {disabled}
        {empty_label}
        {empty_state_message}
        {empty_state_subtitle}
        {empty_state_link}
      />
    {:else if inputType === "radio"}
      <div
        role="radiogroup"
        aria-label={aria_label || label}
        class="flex flex-col gap-2 {error_message ? 'input-error' : ''}"
        {id}
        tabindex="-1"
        data-testid="radio-group-{id}"
      >
        {#each radio_options as option}
          <label
            class="flex items-start gap-2.5 cursor-pointer rounded-lg border px-3 py-2.5 transition-colors
              {value === option.value
              ? 'border-primary/30 bg-primary/[0.03]'
              : 'border-base-300 hover:border-base-content/20'}
              {disabled ? 'opacity-50 pointer-events-none' : ''}
              {error_message ? 'border-error/50' : ''}"
          >
            <input
              type="radio"
              name={id}
              class="radio radio-sm radio-primary mt-0.5 flex-none"
              value={option.value}
              checked={value === option.value}
              on:change={() => {
                value = option.value
                if (on_radio_change) on_radio_change()
              }}
              {disabled}
            />
            <span class="flex flex-col gap-0.5">
              <span class="text-sm font-medium">{option.label}</span>
              {#if option.description}
                <span class="text-xs text-gray-500">{option.description}</span>
              {/if}
            </span>
          </label>
        {/each}
      </div>
    {/if}
    {#if inline_error || (inputType === "select" && error_message)}
      <span
        class="absolute right-3 bottom-4 badge badge-error badge-sm badge-outline text-xs bg-base-100"
      >
        {inline_error || error_message}
      </span>
    {/if}
  </div>
</div>
