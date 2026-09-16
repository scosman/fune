<svelte:options accessors />

<script lang="ts">
  import type { components } from "$lib/api_schema"
  import FormElement from "$lib/utils/form_element.svelte"
  import OutputValueField from "./form_parts/output_value_field.svelte"
  import ReferenceFieldSelect from "./form_parts/reference_field_select.svelte"
  import { SHOW_REFERENCE_DATA_UI } from "$lib/utils/eval_types/reference_data_ui"

  export let properties: components["schemas"]["ExactMatchProperties"] = {
    type: "exact_match",
    case_sensitive: true,
    value_expression: null,
    expected_value: null,
    reference_key: null,
  }

  export let reference_candidate_keys: string[] = []
  export let required_reference_fields: string[] = []

  export function getProperties(): components["schemas"]["ExactMatchProperties"] {
    if (source === "reference_key") {
      return { ...properties, expected_value: null }
    }
    return { ...properties, reference_key: null }
  }

  export function validate(): string | null {
    if (source === "expected_value" && !properties.expected_value) {
      return "Expected value is required."
    }
    if (source === "reference_key" && !properties.reference_key) {
      return "Reference key is required."
    }
    return null
  }

  let source: "expected_value" | "reference_key" = !SHOW_REFERENCE_DATA_UI
    ? "expected_value"
    : properties.expected_value
      ? "expected_value"
      : properties.reference_key
        ? "reference_key"
        : "expected_value"

  $: required_reference_fields =
    source === "reference_key" && properties.reference_key
      ? [properties.reference_key]
      : []

  function on_source_change() {
    if (source === "expected_value") {
      properties.reference_key = null
    } else {
      properties.expected_value = null
    }
  }
</script>

<div class="flex flex-col gap-6">
  <div class="flex flex-col gap-3">
    {#if SHOW_REFERENCE_DATA_UI}
      <FormElement
        id="exact_match_source"
        label="Expected Value"
        inputType="radio"
        radio_options={[
          {
            value: "expected_value",
            label: "Fixed value",
            description: "Enter the exact value the output should match.",
          },
          {
            value: "reference_key",
            label: "From reference data",
            description:
              "Look up the expected value from your dataset's reference fields.",
          },
        ]}
        bind:value={source}
        on_radio_change={on_source_change}
      />

      {#if source === "expected_value"}
        <div class="ml-4 border-l border-base-300 pl-4">
          <FormElement
            id="exact_match_expected_value"
            label="Value"
            inputType="input"
            placeholder="e.g. yes"
            bind:value={properties.expected_value}
          />
        </div>
      {:else}
        <div class="ml-4 border-l border-base-300 pl-4">
          <ReferenceFieldSelect
            id_prefix="exact_match"
            candidate_keys={reference_candidate_keys}
            bind:value={properties.reference_key}
          />
        </div>
      {/if}
    {:else}
      <FormElement
        id="exact_match_expected_value"
        label="Expected Value"
        description="The exact value the output should match."
        inputType="input"
        placeholder="e.g. yes"
        bind:value={properties.expected_value}
      />
    {/if}
  </div>

  <FormElement
    id="exact_match_case_sensitive"
    label="Case Sensitive"
    inputType="checkbox"
    bind:value={properties.case_sensitive}
  />

  <OutputValueField
    id_prefix="exact_match"
    bind:value={properties.value_expression}
  />
</div>
