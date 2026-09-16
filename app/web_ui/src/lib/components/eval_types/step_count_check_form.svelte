<script lang="ts">
  import type { components } from "$lib/api_schema"
  import FormElement from "$lib/utils/form_element.svelte"

  export let properties: components["schemas"]["StepCountCheckProperties"] = {
    type: "step_count_check",
    count_type: "tool_calls",
    min_count: null,
    max_count: null,
  }

  export function getProperties(): components["schemas"]["StepCountCheckProperties"] {
    return properties
  }

  export function validate(): string | null {
    if (properties.min_count == null && properties.max_count == null) {
      return "At least one bound must be set."
    }
    if (
      properties.min_count != null &&
      properties.max_count != null &&
      properties.min_count > properties.max_count
    ) {
      return "Minimum must be less than or equal to maximum."
    }
    return null
  }

  let bounds_error: string | null = null
  let bounds_touched = false

  function on_bounds_focusout() {
    bounds_touched = true
    check_bounds(properties.min_count, properties.max_count)
  }

  function check_bounds(
    min: number | null | undefined,
    max: number | null | undefined,
  ) {
    if (min != null && max != null && min > max) {
      bounds_error = "Minimum must be ≤ maximum."
    } else {
      bounds_error = null
    }
  }

  $: if (bounds_touched) {
    check_bounds(properties.min_count, properties.max_count)
  }
</script>

<div class="flex flex-col gap-6">
  <FormElement
    id="step_count_check_count_type"
    label="What to Count"
    description="Choose what counts as one step in the agent's trace."
    inputType="radio"
    radio_options={[
      {
        value: "tool_calls",
        label: "Tool calls",
        description:
          "Each individual tool invocation. A single response that calls three tools counts as three.",
      },
      {
        value: "model_responses",
        label: "Model responses",
        description:
          "Each message the model generates, including intermediate messages that only call tools.",
      },
      {
        value: "turns",
        label: "Conversation turns",
        description:
          "Each user message and everything the agent does to answer it. Tool calls and model responses within a turn don't add to the count.",
      },
    ]}
    bind:value={properties.count_type}
  />

  <div class="flex flex-col gap-3">
    <FormElement
      id="step_count_bounds_header"
      inputType="header_only"
      label="Bounds"
      description="Set a minimum and/or maximum. At least one is required."
      value=""
    />
    <div class="ml-4 border-l border-base-300 pl-4">
      <!-- focusout (not blur) so the event bubbles up from the inner inputs -->
      <!-- svelte-ignore a11y-no-static-element-interactions -->
      <div on:focusout={on_bounds_focusout}>
        <div class="flex gap-4" data-testid="bounds-row">
          <div class="flex-1">
            <FormElement
              id="step_count_check_min"
              label="Minimum"
              inputType="input_number"
              optional={true}
              placeholder="No minimum"
              bind:value={properties.min_count}
              min={0}
            />
          </div>
          <div class="flex-1">
            <FormElement
              id="step_count_check_max"
              label="Maximum"
              inputType="input_number"
              optional={true}
              placeholder="No maximum"
              bind:value={properties.max_count}
              min={0}
            />
          </div>
        </div>
        {#if bounds_error}
          <p class="text-error text-xs mt-1" data-testid="bounds-error">
            {bounds_error}
          </p>
        {/if}
      </div>
    </div>
  </div>
</div>
