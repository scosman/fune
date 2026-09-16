<script lang="ts">
  import { createEventDispatcher } from "svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import CloseIcon from "$lib/ui/icons/close_icon.svelte"
  import CalloutCard from "$lib/ui/callout_card.svelte"
  import BookIcon from "$lib/ui/icons/book_icon.svelte"
  import type { ReferenceDataUsageMode } from "$lib/utils/eval_types/registry"

  export let reference_data: string = ""
  export let required_reference_fields: string[] = []
  export let usage_mode: ReferenceDataUsageMode = "llm_judge"
  /**
   * Names the saved judge will require. A reference-answer judge requires
   * `reference_answer` exactly and the run is refused without it, so the name is worth
   * saying rather than leaving the tester to find it in a skipped result.
   */
  export let judge_required_keys: string[] = []
  /**
   * Names the prompt reads that are not required. Worth naming too, but separately:
   * a `reference_data.get("x")` lookup renders around a missing value rather than
   * skipping, so promising a skip here would be wrong.
   */
  export let judge_prompt_keys: string[] = []

  $: prompt_keys_lead =
    judge_required_keys.length > 0
      ? "Its prompt also reads:"
      : "This judge's prompt reads:"

  let dialog: Dialog
  let rows: Array<{ key: string; value: string }> = []
  let save_error: string | null = null

  const dispatch = createEventDispatcher<{
    change: string
  }>()

  $: missing_fields = get_missing_fields(
    reference_data,
    required_reference_fields,
  )
  $: display_value =
    missing_fields.length > 0
      ? "Missing " + missing_fields.join(", ")
      : get_display_value(reference_data)
  $: display_is_error = missing_fields.length > 0

  function is_plain_object(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value)
  }

  function get_missing_fields(data: string, required: string[]): string[] {
    if (required.length === 0) return []
    if (!data.trim()) return required
    try {
      const parsed = JSON.parse(data.trim())
      if (!is_plain_object(parsed)) return required
      return required.filter(
        (k) =>
          !(k in parsed) ||
          parsed[k] === undefined ||
          parsed[k] === null ||
          parsed[k] === "",
      )
    } catch {
      return required
    }
  }

  function get_display_value(data: string): string {
    if (!data.trim()) return "None"
    try {
      const parsed = JSON.parse(data.trim())
      if (!is_plain_object(parsed)) return "Invalid: not an object"
      const keys = Object.keys(parsed)
      if (keys.length === 0) return "Empty object"
      if (keys.length <= 3) return keys.join(", ")
      return `${keys.slice(0, 3).join(", ")} +${keys.length - 3} more`
    } catch {
      return "Invalid JSON"
    }
  }

  function parse_value(raw: string): unknown {
    try {
      return JSON.parse(raw)
    } catch {
      return raw
    }
  }

  function object_to_rows(data: string): Array<{ key: string; value: string }> {
    if (!data.trim()) return []
    try {
      const parsed = JSON.parse(data.trim())
      if (!is_plain_object(parsed)) return []
      return Object.entries(parsed).map(([key, val]) => ({
        key,
        value: JSON.stringify(val),
      }))
    } catch {
      return []
    }
  }

  function open_editor() {
    rows = object_to_rows(reference_data)
    if (rows.length === 0) {
      rows = [{ key: "", value: "" }]
    }
    save_error = null
    dialog?.show()
  }

  function add_row() {
    rows = [...rows, { key: "", value: "" }]
  }

  function remove_row(index: number) {
    rows = rows.filter((_, i) => i !== index)
  }

  function handle_save(): boolean {
    // Filter out completely empty rows
    const filled = rows.filter((r) => r.key.trim() || r.value.trim())

    if (filled.length === 0) {
      dispatch("change", "")
      return true
    }

    // Check for empty keys on rows that have values
    const empty_key = filled.find((r) => !r.key.trim())
    if (empty_key) {
      save_error = "Each row must have a name."
      return false
    }

    // Check for duplicate keys
    const keys = filled.map((r) => r.key.trim())
    const unique_keys = new Set(keys)
    if (unique_keys.size !== keys.length) {
      save_error = "Duplicate names are not allowed."
      return false
    }

    const obj: Record<string, unknown> = {}
    for (const row of filled) {
      obj[row.key.trim()] = parse_value(row.value)
    }

    save_error = null
    dispatch("change", JSON.stringify(obj))
    return true
  }
</script>

<div
  class="flex items-center justify-between text-sm py-1"
  data-testid="reference-data-field"
>
  <span>Reference Data</span>
  <button
    type="button"
    class="link text-xs {display_is_error
      ? 'text-error'
      : 'text-gray-500 hover:text-primary'}"
    on:click={open_editor}
    data-testid="reference-data-edit"
  >
    {display_value}
  </button>
</div>

<Dialog
  bind:this={dialog}
  title="Reference Data"
  width="wide"
  action_buttons={[
    {
      label: "Save",
      isPrimary: true,
      action: handle_save,
    },
  ]}
>
  <div class="flex flex-col gap-3" data-testid="reference-data-editor">
    {#if usage_mode === "llm_judge"}
      <CalloutCard testid="ref-data-callout">
        <svelte:fragment slot="icon"><BookIcon /></svelte:fragment>
        {#if judge_required_keys.length > 0 || judge_prompt_keys.length > 0}
          <p class="text-sm text-gray-500">
            Reference data is the expected values (ground truth) for this test
            case.
          </p>
          {#if judge_required_keys.length > 0}
            <p class="text-sm text-gray-500 mt-1">
              This judge requires {judge_required_keys.length === 1
                ? "this name"
                : "these names"}, spelled exactly &mdash; the test run is
              skipped without {judge_required_keys.length === 1
                ? "it"
                : "them"}:
            </p>
            <p
              class="mt-1 flex flex-wrap gap-1"
              data-testid="ref-data-required-keys"
            >
              {#each judge_required_keys as key}
                <code
                  class="font-mono text-xs bg-base-200 px-1.5 py-0.5 rounded"
                  >{key}</code
                >
              {/each}
            </p>
          {/if}
          {#if judge_prompt_keys.length > 0}
            <p class="text-sm text-gray-500 mt-1">
              {prompt_keys_lead}
            </p>
            <p
              class="mt-1 flex flex-wrap gap-1"
              data-testid="ref-data-prompt-keys"
            >
              {#each judge_prompt_keys as key}
                <code
                  class="font-mono text-xs bg-base-200 px-1.5 py-0.5 rounded"
                  >{key}</code
                >
              {/each}
            </p>
          {/if}
        {:else}
          <p class="text-sm text-gray-500">
            Reference data is the expected values (ground truth) for this test
            case. Use it in your judge prompt, like so:
          </p>
          <p class="mt-1">
            <code class="font-mono text-xs bg-base-200 px-1.5 py-0.5 rounded"
              >{'The output type should equal "{{ reference_data.expected_type }}"'}</code
            >
          </p>
        {/if}
      </CalloutCard>
    {:else if usage_mode === "reference_field"}
      <CalloutCard testid="ref-data-callout">
        <svelte:fragment slot="icon"><BookIcon /></svelte:fragment>
        <p class="text-sm text-gray-500">
          Reference data is the expected values (ground truth) for this test
          case. In the judge config, choose <span class="font-medium"
            >From reference data</span
          >, then select the field to compare against.
        </p>
      </CalloutCard>
    {:else if usage_mode === "optional"}
      <CalloutCard testid="ref-data-callout">
        <svelte:fragment slot="icon"><BookIcon /></svelte:fragment>
        <p class="text-sm text-gray-500">
          Reference data is the expected values (ground truth) for this test
          case. It's optional here, but if your <span class="font-medium"
            >Output to Check</span
          > expression reads it via Jinja, provide it so the test can run:
        </p>
        <p class="mt-1">
          <code class="font-mono text-xs bg-base-200 px-1.5 py-0.5 rounded"
            >{"{{ reference_data.expected_type }}"}</code
          >
        </p>
      </CalloutCard>
    {:else if usage_mode === "code"}
      <CalloutCard testid="ref-data-callout">
        <svelte:fragment slot="icon"><BookIcon /></svelte:fragment>
        <p class="text-sm text-gray-500">
          Reference data is the expected values (ground truth) for this test
          case. Your scoring code receives it as the <code
            class="font-mono text-xs bg-base-200 px-1.5 py-0.5 rounded"
            >reference_data</code
          > argument, like so:
        </p>
        <p class="mt-1">
          <code class="font-mono text-xs bg-base-200 px-1.5 py-0.5 rounded"
            >expected = (reference_data or {"{}"}).get("expected_answer")</code
          >
        </p>
      </CalloutCard>
    {/if}

    {#if rows.length === 0}
      <p class="text-sm text-gray-500">
        No values yet. Add one to get started.
      </p>
    {/if}

    <!-- Column headers rendered once above all rows -->
    {#if rows.length > 0}
      <div
        class="flex items-end gap-2"
        data-testid="reference-data-column-headers"
      >
        <div class="flex-1">
          <span class="label-text text-xs">Name</span>
        </div>
        <div class="flex-[2] flex flex-col">
          <span class="label-text text-xs">Value</span>
          <span class="text-xs text-gray-500" data-testid="json-format-note"
            >Any valid JSON: strings, numbers, booleans, arrays, or objects.</span
          >
        </div>
        <!-- Spacer matching the remove button width -->
        <div class="w-8 flex-shrink-0"></div>
      </div>
    {/if}

    {#each rows as row, index}
      <div class="flex items-center gap-2" data-testid="reference-data-row">
        <div class="flex-1">
          <input
            type="text"
            class="input input-bordered input-sm w-full font-mono"
            placeholder="name"
            bind:value={row.key}
            data-testid="reference-data-key"
          />
        </div>
        <div class="flex-[2]">
          <input
            type="text"
            class="input input-bordered input-sm w-full font-mono"
            placeholder="value"
            bind:value={row.value}
            data-testid="reference-data-value"
          />
        </div>
        <div class="flex-shrink-0">
          <button
            type="button"
            class="btn btn-ghost btn-sm btn-square"
            on:click={() => remove_row(index)}
            aria-label="Remove row"
            data-testid="reference-data-remove"
          >
            <span class="w-4 h-4 block text-gray-400"><CloseIcon /></span>
          </button>
        </div>
      </div>
    {/each}

    <div>
      <button
        type="button"
        class="btn btn-ghost btn-xs text-primary"
        on:click={add_row}
        data-testid="reference-data-add"
      >
        ＋ Add Value
      </button>
    </div>

    {#if save_error}
      <div class="text-error text-xs" data-testid="reference-data-error">
        {save_error}
      </div>
    {/if}
  </div>
</Dialog>
