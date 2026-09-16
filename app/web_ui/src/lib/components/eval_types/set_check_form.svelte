<svelte:options accessors />

<script lang="ts">
  import type { components } from "$lib/api_schema"
  import FormElement from "$lib/utils/form_element.svelte"
  import OutputValueField from "./form_parts/output_value_field.svelte"
  import ReferenceFieldSelect from "./form_parts/reference_field_select.svelte"
  import TagInput from "./tag_input.svelte"
  import { SHOW_REFERENCE_DATA_UI } from "$lib/utils/eval_types/reference_data_ui"

  export let properties: components["schemas"]["SetCheckProperties"] = {
    type: "set_check",
    mode: "equal",
    value_expression: null,
    expected_set: [],
    reference_key: null,
  }

  export let reference_candidate_keys: string[] = []
  export let required_reference_fields: string[] = []

  export function getProperties(): components["schemas"]["SetCheckProperties"] {
    if (source === "reference_key") {
      return { ...properties, expected_set: null }
    }
    return { ...properties, reference_key: null }
  }

  export function validate(): string | null {
    if (
      source === "expected_set" &&
      (!properties.expected_set || properties.expected_set.length === 0)
    ) {
      return "Expected set must contain at least one value."
    }
    if (source === "reference_key" && !properties.reference_key) {
      return "Reference key is required."
    }
    return null
  }

  let source: "expected_set" | "reference_key" =
    SHOW_REFERENCE_DATA_UI && properties.reference_key
      ? "reference_key"
      : "expected_set"

  $: required_reference_fields =
    source === "reference_key" && properties.reference_key
      ? [properties.reference_key]
      : []

  function on_source_change() {
    if (source === "expected_set") {
      properties.reference_key = null
    } else {
      properties.expected_set = null
      expected_set_tags = []
    }
  }

  let expected_set_tags: string[] = properties.expected_set ?? []
  $: if (source === "expected_set") {
    properties.expected_set = expected_set_tags
  }
</script>

<div class="flex flex-col gap-6">
  <div class="flex flex-col gap-3">
    {#if SHOW_REFERENCE_DATA_UI}
      <FormElement
        id="set_check_source"
        label="Expected Values"
        description="Define what the output should contain."
        inputType="radio"
        radio_options={[
          {
            value: "expected_set",
            label: "Fixed values",
            description: "Enter the expected values directly.",
          },
          {
            value: "reference_key",
            label: "From reference data",
            description:
              "Look up the expected values from your dataset's reference fields.",
          },
        ]}
        bind:value={source}
        on_radio_change={on_source_change}
      />

      {#if source === "expected_set"}
        <div class="ml-4 border-l border-base-300 pl-4">
          <FormElement
            id="set_check_expected_values_header"
            inputType="header_only"
            label="Values"
            description="Add items by typing and pressing Enter or comma."
            value=""
          />
          <TagInput
            id="set_check_expected_set"
            bind:tags={expected_set_tags}
            placeholder="Type a value and press Enter"
          />
        </div>
      {:else}
        <div class="ml-4 border-l border-base-300 pl-4">
          <ReferenceFieldSelect
            id_prefix="set_check"
            candidate_keys={reference_candidate_keys}
            bind:value={properties.reference_key}
          />
        </div>
      {/if}
    {:else}
      <FormElement
        id="set_check_expected_values_header"
        inputType="header_only"
        label="Expected Values"
        description="Define what the output should contain. Add items by typing and pressing Enter or comma."
        value=""
      />
      <TagInput
        id="set_check_expected_set"
        bind:tags={expected_set_tags}
        placeholder="Type a value and press Enter"
      />
    {/if}
  </div>

  <FormElement
    id="set_check_mode"
    label="Comparison Mode"
    inputType="radio"
    radio_options={[
      {
        value: "equal",
        label: "Equal",
        description:
          "The output must contain exactly the expected values, with no extras and nothing missing.",
      },
      {
        value: "subset",
        label: "Subset",
        description:
          "Every output value must appear in the expected values (extras in expected are OK).",
      },
      {
        value: "superset",
        label: "Superset",
        description:
          "Every expected value must appear in the output (extra output values are OK).",
      },
    ]}
    bind:value={properties.mode}
  />

  <OutputValueField
    id_prefix="set_check"
    bind:value={properties.value_expression}
    extra_description="Must be an array."
  />
</div>
