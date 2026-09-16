<svelte:options accessors />

<script lang="ts">
  import type { components } from "$lib/api_schema"
  import FormElement from "$lib/utils/form_element.svelte"
  import OutputValueField from "./form_parts/output_value_field.svelte"
  import ReferenceFieldSelect from "./form_parts/reference_field_select.svelte"
  import { SHOW_REFERENCE_DATA_UI } from "$lib/utils/eval_types/reference_data_ui"

  export let properties: components["schemas"]["ContainsProperties"] = {
    type: "contains",
    case_sensitive: true,
    mode: "must_contain",
    value_expression: null,
    substring: null,
    reference_key: null,
  }

  export let reference_candidate_keys: string[] = []
  export let required_reference_fields: string[] = []

  export function getProperties(): components["schemas"]["ContainsProperties"] {
    if (source === "reference_key") {
      return { ...properties, substring: null }
    }
    return { ...properties, reference_key: null }
  }

  export function validate(): string | null {
    if (source === "substring" && !properties.substring) {
      return "Substring is required."
    }
    if (source === "reference_key" && !properties.reference_key) {
      return "Reference key is required."
    }
    return null
  }

  let source: "substring" | "reference_key" =
    SHOW_REFERENCE_DATA_UI && properties.reference_key
      ? "reference_key"
      : "substring"

  $: required_reference_fields =
    source === "reference_key" && properties.reference_key
      ? [properties.reference_key]
      : []

  function on_source_change() {
    if (source === "substring") {
      properties.reference_key = null
    } else {
      properties.substring = null
    }
  }
</script>

<div class="flex flex-col gap-6">
  <div class="flex flex-col gap-3">
    {#if SHOW_REFERENCE_DATA_UI}
      <FormElement
        id="contains_source"
        label="Expected Substring"
        description="The text to search for in the output."
        inputType="radio"
        radio_options={[
          {
            value: "substring",
            label: "Fixed value",
            description: "Enter the text to search for in the output.",
          },
          {
            value: "reference_key",
            label: "From reference data",
            description:
              "Look up the search text from your dataset's reference fields.",
          },
        ]}
        bind:value={source}
        on_radio_change={on_source_change}
      />

      {#if source === "substring"}
        <div class="ml-4 border-l border-base-300 pl-4">
          <FormElement
            id="contains_substring"
            label="Value"
            inputType="input"
            placeholder="e.g. success"
            bind:value={properties.substring}
          />
        </div>
      {:else}
        <div class="ml-4 border-l border-base-300 pl-4">
          <ReferenceFieldSelect
            id_prefix="contains"
            candidate_keys={reference_candidate_keys}
            bind:value={properties.reference_key}
          />
        </div>
      {/if}
    {:else}
      <FormElement
        id="contains_substring"
        label="Expected Substring"
        description="The text to search for in the output."
        inputType="input"
        placeholder="e.g. success"
        bind:value={properties.substring}
      />
    {/if}
  </div>

  <FormElement
    id="contains_mode"
    label="Match Mode"
    inputType="radio"
    radio_options={[
      {
        value: "must_contain",
        label: "Must contain",
        description: "The output must contain the value to pass.",
      },
      {
        value: "must_not_contain",
        label: "Must not contain",
        description: "The output must NOT contain the value to pass.",
      },
    ]}
    bind:value={properties.mode}
  />

  <FormElement
    id="contains_case_sensitive"
    label="Case Sensitive"
    inputType="checkbox"
    bind:value={properties.case_sensitive}
  />

  <OutputValueField
    id_prefix="contains"
    bind:value={properties.value_expression}
  />
</div>
