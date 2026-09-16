<script lang="ts">
  import type { components } from "$lib/api_schema"
  import FormElement from "$lib/utils/form_element.svelte"
  import OutputValueField from "./form_parts/output_value_field.svelte"

  export let properties: components["schemas"]["PatternMatchProperties"] = {
    type: "pattern_match",
    pattern: "",
    mode: "must_match",
    value_expression: null,
  }

  export function getProperties(): components["schemas"]["PatternMatchProperties"] {
    return properties
  }

  let regex_error: string | null = null
  let pattern_touched = false

  function on_pattern_focusout() {
    pattern_touched = true
    validate_regex()
  }

  function validate_regex() {
    if (!properties.pattern) {
      regex_error = null
      return
    }
    try {
      new RegExp(properties.pattern)
      regex_error = null
    } catch (e) {
      regex_error = `Invalid regex: ${e instanceof SyntaxError ? e.message : String(e)}`
    }
  }

  $: if (pattern_touched && properties.pattern !== undefined) {
    validate_regex()
  }

  export function validate(): string | null {
    if (!properties.pattern) {
      return "Regular expression is required."
    }
    try {
      new RegExp(properties.pattern)
    } catch {
      return "Invalid regular expression pattern."
    }
    return null
  }
</script>

<div class="flex flex-col gap-6">
  <!-- focusout (not blur) so the event bubbles up from the inner input -->
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div
    on:focusout={on_pattern_focusout}
    data-testid="pattern-match-pattern-section"
  >
    <FormElement
      id="pattern_match_pattern"
      label="Expected Pattern (Regex)"
      description="The pattern to test against the output."
      info_description="A regular expression (regex) is a sequence of characters that defines a search pattern. For example, ^yes$ matches only the exact string 'yes', while \\d+ matches one or more digits."
      inputType="input"
      placeholder="e.g. ^(yes|no)$"
      bind:value={properties.pattern}
      error_message={regex_error}
    />
  </div>

  <FormElement
    id="pattern_match_mode"
    label="Match Mode"
    inputType="radio"
    radio_options={[
      {
        value: "must_match",
        label: "Must match",
        description: "The output must match the pattern to pass.",
      },
      {
        value: "must_not_match",
        label: "Must not match",
        description: "The output must NOT match the pattern to pass.",
      },
    ]}
    bind:value={properties.mode}
  />

  <OutputValueField
    id_prefix="pattern_match"
    bind:value={properties.value_expression}
  />
</div>
