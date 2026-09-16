<script lang="ts">
  import { createEventDispatcher } from "svelte"
  import FormContainer from "$lib/utils/form_container.svelte"
  import FormElement from "$lib/utils/form_element.svelte"
  import type { KilnError } from "$lib/utils/error_handlers"
  import type { FieldConfig } from "../select_template/spec_templates"
  import { filename_string_short_validator } from "$lib/utils/input_validators"
  import type { Priority } from "$lib/types"

  export let name: string
  export let property_values: Record<string, string | null>
  export let initial_property_values: Record<string, string | null>
  export let priority: Priority = 1
  export let field_configs: FieldConfig[]
  export let error: KilnError | null
  export let submitting: boolean
  export let warn_before_unload: boolean

  const dispatch = createEventDispatcher<{
    create_spec: void
  }>()

  function reset_field(key: string) {
    property_values[key] = initial_property_values[key] ?? null
    property_values = { ...property_values }
  }

  function has_form_changes(
    current: Record<string, string | null>,
    initial: Record<string, string | null>,
  ): boolean {
    for (const key of Object.keys(current)) {
      if (current[key] !== initial[key]) return true
    }
    return false
  }

  $: computed_warn_before_unload =
    warn_before_unload &&
    has_form_changes(property_values, initial_property_values)

  function handle_submit() {
    dispatch("create_spec")
  }
</script>

<FormContainer
  submit_label="Create Eval"
  on:submit={handle_submit}
  bind:error
  bind:submitting
  compact_button={true}
  warn_before_unload={computed_warn_before_unload}
>
  <FormElement
    label="Eval Name"
    description="A short name for your own reference."
    id="spec_name"
    bind:value={name}
    validator={filename_string_short_validator}
  />

  <FormElement
    label="Priority"
    id="priority"
    inputType="select"
    bind:value={priority}
    description="The priority level for this eval."
    select_options={[
      [0, "P0 - Critical"],
      [1, "P1 - High"],
      [2, "P2 - Medium"],
      [3, "P3 - Low"],
    ]}
  />

  {#each field_configs as field (field.key)}
    <FormElement
      label={field.label}
      id={field.key}
      inputType="textarea"
      disabled={field.disabled || false}
      description={field.description}
      info_description={field.info_description}
      height={field.height || "base"}
      bind:value={property_values[field.key]}
      optional={!field.required}
      inline_action={initial_property_values[field.key] &&
      property_values[field.key] !== initial_property_values[field.key]
        ? {
            handler: () => reset_field(field.key),
            label: "Reset",
          }
        : undefined}
    />
  {/each}
</FormContainer>
