<script lang="ts">
  import FormElement from "$lib/utils/form_element.svelte"
  import RunInputFormElement from "$lib/components/run_input_form_element.svelte"
  import {
    model_from_schema_string,
    type SchemaModelProperty,
  } from "$lib/utils/json_schema_editor/json_schema_templates"

  let id = "plaintext_input_" + Math.random().toString(36).substring(2, 15)

  export let input_schema: string | null | undefined
  export let onInputChange: (() => void) | null = null
  // Plaintext-mode label/placeholder. The label doubles as the validation
  // message and aria-label, so it stays meaningful even when visually hidden
  // (e.g. the multiturn composer hides the header and uses a custom hint).
  export let label: string = "Plaintext Input"
  export let placeholder: string | null = null
  export let hide_label: boolean = false
  // Disables the plaintext textarea (e.g. while a multiturn send is in flight).
  export let disabled: boolean = false
  // Height of the plaintext textarea. Defaults to the tall input used on the
  // /run page; the multiturn composer overrides it with a shorter box.
  export let height: "base" | "compact" | "medium" | "large" | "xl" = "large"
  let plaintext_input: string = ""
  $: void (plaintext_input, onInputChange?.())

  // Store ref to the root form element
  let rootFormElement: { buildValue(): unknown } | null = null

  // Key to force component remount when needed
  let formKey = 0

  let structured_input_model: SchemaModelProperty | null = null
  $: structured_input_model = input_schema
    ? model_from_schema_string(input_schema)
    : null

  // These two are mutually exclusive. One returns null if the other is not null.
  export function get_plaintext_input_data(): string | null {
    if (input_schema) {
      return null
    }
    return plaintext_input
  }

  export function get_structured_input_data(): unknown | null {
    if (!input_schema) {
      return null
    }
    return rootFormElement?.buildValue()
  }

  export function clear_input() {
    plaintext_input = ""
    // resets the form to its initial state
    formKey += 1
  }

  // Set the plaintext input value. No-op when the form is in structured mode
  // (input_schema is present). Used by the multiturn fork composer to seed
  // the textarea with the original turn's text.
  export function set_plaintext_input(value: string) {
    if (input_schema) return
    plaintext_input = value
  }

  // Focus the plaintext textarea and place the caret at the end. No-op when
  // the form is in structured mode or the textarea isn't mounted yet.
  export function focus_plaintext_input() {
    if (input_schema) return
    if (typeof document === "undefined") return
    const textarea = document.getElementById(id) as HTMLTextAreaElement | null
    if (!textarea) return
    textarea.focus()
    textarea.setSelectionRange(textarea.value.length, textarea.value.length)
  }
</script>

{#if !input_schema}
  <!-- Keyed on formKey so clear_input() remounts the field and resets its
       validation state — otherwise the now-empty textarea immediately flags
       "Required" right after a successful send. -->
  {#key formKey}
    <FormElement
      {label}
      {placeholder}
      {hide_label}
      {disabled}
      {height}
      inputType="textarea"
      {id}
      bind:value={plaintext_input}
    />
  {/key}
{:else if structured_input_model}
  {#key formKey}
    <RunInputFormElement
      property={structured_input_model}
      {onInputChange}
      level={0}
      path="root"
      hideHeaderAndIndent={true}
      bind:this={rootFormElement}
    />
  {/key}
{:else}
  <p>Invalid or unsupported input schema</p>
{/if}
