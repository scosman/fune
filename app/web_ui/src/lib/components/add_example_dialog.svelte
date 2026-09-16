<script lang="ts" context="module">
  // A captured example. Which of input/output are populated depends on the
  // dialog's `include_input` / `include_output` props — either or both.
  export type GuideSample = {
    input?: string
    output?: string
    task_run_id?: string
  }
</script>

<script lang="ts">
  // Reusable Add/Edit Example dialog. Captures an input, an output, or both
  // (configurable via include_input/include_output). For structured tasks it
  // renders a schema-aware field-by-field form for whichever side has a JSON
  // schema; plaintext tasks fall back to a free-text textarea. Exposes
  // open_add()/open_edit() and dispatches `submit` with the resulting sample.
  import { createEventDispatcher, onMount } from "svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import FormElement from "$lib/utils/form_element.svelte"
  import TaskRunPicker from "$lib/utils/task_run_picker.svelte"
  import { client } from "$lib/api_client"
  import RunInputFormElement from "$lib/components/run_input_form_element.svelte"
  import {
    model_from_schema_string,
    type SchemaModelProperty,
  } from "$lib/utils/json_schema_editor/json_schema_templates"
  import {
    fetch_task_sample_candidates,
    task_run_to_example,
  } from "$lib/utils/task_sample_example"
  import type { TaskRun } from "$lib/types"

  export let project_id: string
  export let task_id: string
  // Existing examples already added. Used to filter the "Select Existing"
  // picker so the user can't double-add the same run.
  export let existing_examples: GuideSample[] = []
  // Which sides to capture. Callers that only want inputs (e.g. an input data
  // guide) pass include_output={false}; the default captures both.
  export let include_input: boolean = true
  export let include_output: boolean = true
  // When true, skip the "Select Existing" run picker and method-selection
  // screen entirely — the dialog opens straight into the manual entry form.
  // Used by callers that already provide their own run picker upstream (e.g.
  // the scorer test-run flow).
  export let manual_only: boolean = false
  // Dialog chrome. Defaults suit the data-guide flow; other callers pass their
  // own copy. When title is null the header reflects add/edit mode.
  export let title: string | null = null
  // Callers override the sub-subtitle to frame the specific flow — e.g. the
  // data-guide chooser uses "To start, ..." for the very first example.
  export let sub_subtitle: string =
    "Add a task data example to guide generation."
  // Label for the submit button when adding (edit mode always shows "Save").
  export let submit_label: string = "Add"
  // When true, the submit button is disabled until at least one captured side
  // has content — matching the older manual-example UX for the scorer flow.
  export let disable_when_empty: boolean = false

  let example_dialog: Dialog
  let mode: "add" | "edit" = "add"
  let add_method: "manual" | "existing" | null = null
  let editing_index: number = -1
  let editing_input: string = ""
  let editing_output: string = ""
  let editing_task_run_id: string | undefined = undefined

  let available_runs: TaskRun[] = []
  let loading_runs: boolean = true

  // Input/output JSON schemas for structured tasks, fetched on mount. When
  // present (and the side is enabled), new manual entry is captured
  // field-by-field instead of via a free-text textarea.
  let input_json_schema: string | null = null
  let output_json_schema: string | null = null
  // Gates the manual form until the schema fetch resolves, so a user can't
  // start typing in the plaintext fallback only to have it swapped out for
  // the structured form (discarding their text) once the schema arrives.
  let schema_loading: boolean = true
  let rootInputFormElement: RunInputFormElement | null = null
  let rootOutputFormElement: RunInputFormElement | null = null
  let manual_error: string | null = null
  // Bumped on each open so the structured forms re-mount fresh.
  let formKey = 0

  function model_from_schema_or_null(
    schema: string | null,
  ): SchemaModelProperty | null {
    if (!schema) return null
    try {
      return model_from_schema_string(schema)
    } catch {
      return null
    }
  }

  $: input_structured_model = model_from_schema_or_null(input_json_schema)
  $: output_structured_model = model_from_schema_or_null(output_json_schema)
  // Field-by-field entry only when adding from scratch (the schema form can't
  // be pre-filled, so editing an existing example stays on the textarea).
  $: use_structured_input =
    mode === "add" && include_input && !!input_structured_model
  $: use_structured_output =
    mode === "add" && include_output && !!output_structured_model

  // A captured side has content if it's included and either uses the structured
  // form (validated on submit) or has non-blank text. When disable_when_empty
  // is set, the submit button stays disabled until at least one side qualifies.
  $: input_has_content =
    include_input && (use_structured_input || editing_input.trim().length > 0)
  $: output_has_content =
    include_output &&
    (use_structured_output || editing_output.trim().length > 0)
  $: submit_disabled =
    disable_when_empty && !input_has_content && !output_has_content

  $: added_run_ids = new Set(
    existing_examples
      .map((e) => e.task_run_id)
      .filter((id): id is string => !!id),
  )
  $: filtered_available_runs = available_runs.filter(
    (r) => !r.id || !added_run_ids.has(r.id),
  )

  const dispatch = createEventDispatcher<{
    submit: { sample: GuideSample; index: number; mode: "add" | "edit" }
  }>()

  onMount(async () => {
    if (manual_only) {
      loading_runs = false
    } else {
      try {
        const result = await fetch_task_sample_candidates(project_id, task_id)
        available_runs = result.available_runs.filter(
          (r) => r.input_source?.type !== "synthetic",
        )
      } catch {
        // Non-critical
      } finally {
        loading_runs = false
      }
    }

    try {
      const { data } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}",
        { params: { path: { project_id, task_id } } },
      )
      input_json_schema = data?.input_json_schema ?? null
      output_json_schema = data?.output_json_schema ?? null
    } catch {
      // Non-critical — falls back to the free-text textarea.
    } finally {
      schema_loading = false
    }
  })

  export function open_add() {
    mode = "add"
    add_method = manual_only ? "manual" : null
    editing_index = -1
    editing_input = ""
    editing_output = ""
    editing_task_run_id = undefined
    manual_error = null
    formKey += 1
    example_dialog?.show()
  }

  export function open_edit(sample: GuideSample, index: number) {
    mode = "edit"
    add_method = "manual"
    editing_index = index
    editing_input = sample.input ?? ""
    editing_output = sample.output ?? ""
    editing_task_run_id = sample.task_run_id
    manual_error = null
    example_dialog?.show()
  }

  // Resolve one side's value, building JSON from the structured form when
  // active. Returns { ok, value } so callers can abort on a build error.
  function resolve_side(
    use_structured: boolean,
    form: RunInputFormElement | null,
    text_value: string,
  ): { ok: true; value: string } | { ok: false } {
    if (!use_structured) return { ok: true, value: text_value }
    if (!form) {
      manual_error = "Form not ready."
      return { ok: false }
    }
    try {
      // buildValue() returns undefined when every field is optional and left
      // blank; JSON.stringify(undefined) is undefined, so coerce to "".
      const built = form.buildValue()
      const value = built === undefined ? "" : JSON.stringify(built, null, 2)
      return { ok: true, value }
    } catch (e) {
      manual_error = e instanceof Error ? e.message : String(e)
      return { ok: false }
    }
  }

  function save_manual() {
    manual_error = null

    const sample: GuideSample = {
      task_run_id: mode === "edit" ? editing_task_run_id : undefined,
    }

    if (include_input) {
      const resolved = resolve_side(
        use_structured_input,
        rootInputFormElement,
        editing_input,
      )
      if (!resolved.ok) return
      sample.input = resolved.value
    }
    if (include_output) {
      const resolved = resolve_side(
        use_structured_output,
        rootOutputFormElement,
        editing_output,
      )
      if (!resolved.ok) return
      sample.output = resolved.value
    }

    dispatch("submit", { sample, index: editing_index, mode })
    example_dialog?.close()
  }

  function select_existing_run(run: TaskRun) {
    const ex = task_run_to_example(run)
    const sample: GuideSample = {
      ...(include_input ? { input: ex.input } : {}),
      ...(include_output ? { output: ex.output } : {}),
      task_run_id: run.id ?? undefined,
    }
    dispatch("submit", { sample, index: -1, mode: "add" })
    example_dialog?.close()
  }
</script>

<Dialog
  bind:this={example_dialog}
  width="wide"
  title={title ?? (mode === "edit" ? "Edit Example" : "Add Example")}
  {sub_subtitle}
>
  {#if mode === "add" && (loading_runs || schema_loading)}
    <div class="flex justify-center items-center py-8">
      <span class="loading loading-spinner loading-md text-primary" />
    </div>
  {:else}
    {#if mode === "add" && add_method === null && filtered_available_runs.length > 0}
      <!-- Method selection -->
      <div class="flex flex-col gap-4 mt-8">
        <button
          class="btn btn-outline mb-2"
          on:click={() => (add_method = "manual")}
          type="button"
        >
          Add Manually
        </button>

        <div class="flex items-center gap-2">
          <div class="flex-1 border-t border-base-300"></div>
          <span class="text-sm text-gray-400">or Select Existing</span>
          <div class="flex-1 border-t border-base-300"></div>
        </div>

        <div class="flex flex-col mt-2 gap-2">
          <TaskRunPicker
            available_runs={filtered_available_runs}
            inputs_only={!include_output}
            on:select={(e) => select_existing_run(e.detail)}
          />
        </div>
      </div>
    {/if}
    {#if mode === "edit" || filtered_available_runs.length === 0 || add_method === "manual"}
      <!-- Manual input/output form -->
      <div class="flex flex-col gap-3">
        {#if include_input}
          {#if use_structured_input && input_structured_model}
            <div class="font-medium text-sm">Input</div>
            {#key formKey}
              <RunInputFormElement
                property={input_structured_model}
                level={0}
                path="input_root"
                hideHeaderAndIndent={true}
                bind:this={rootInputFormElement}
              />
            {/key}
          {:else}
            <FormElement
              label="Input"
              id="example_input"
              inputType="textarea"
              height="medium"
              bind:value={editing_input}
              optional={true}
              hide_optional_badge={true}
            />
          {/if}
        {/if}

        {#if include_output}
          {#if use_structured_output && output_structured_model}
            <div class="font-medium text-sm">Output</div>
            {#key formKey}
              <RunInputFormElement
                property={output_structured_model}
                level={0}
                path="output_root"
                hideHeaderAndIndent={true}
                bind:this={rootOutputFormElement}
              />
            {/key}
          {:else}
            <FormElement
              label="Output"
              id="example_output"
              inputType="textarea"
              height="medium"
              bind:value={editing_output}
              optional={true}
              hide_optional_badge={true}
            />
          {/if}
        {/if}

        {#if manual_error}
          <div class="text-error text-sm">{manual_error}</div>
        {/if}
        <div class="flex flex-row gap-2 justify-end mt-2">
          {#if mode === "add" && filtered_available_runs.length > 0 && available_runs.length > 0}
            <button
              type="button"
              class="btn btn-sm h-10 btn-outline min-w-24"
              on:click={() => (add_method = null)}
            >
              Back
            </button>
          {/if}
          <button
            type="button"
            class="btn btn-sm h-10 btn-primary min-w-24"
            disabled={submit_disabled}
            on:click={save_manual}
          >
            {mode === "edit" ? "Save" : submit_label}
          </button>
        </div>
      </div>
    {/if}
  {/if}
</Dialog>
