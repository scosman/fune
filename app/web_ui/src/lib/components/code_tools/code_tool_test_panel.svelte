<script lang="ts">
  import { client } from "$lib/api_client"
  import { createKilnError, type KilnError } from "$lib/utils/error_handlers"
  import RunInputFormElement from "$lib/components/run_input_form_element.svelte"
  import {
    model_from_schema,
    type SchemaModelProperty,
  } from "$lib/utils/json_schema_editor/json_schema_templates"
  import { formatParamPreview } from "$lib/utils/code_tool_helpers"
  import Output from "$lib/ui/output.svelte"
  import Warning from "$lib/ui/warning.svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import CodeTrustDialog from "$lib/components/code_tools/code_trust_dialog.svelte"
  import type { TestCodeToolResponse } from "$lib/types"
  import posthog from "posthog-js"
  import ToolCallLogTable from "$lib/components/tool_calls/tool_call_log_table.svelte"

  export let project_id: string
  export let tool_function_name: string
  export let tool_description: string
  export let parameters_schema: { [key: string]: unknown }
  export let code: string
  export let timeout_seconds: number
  export let tool_allowlist: string[]

  let test_loading = false
  let test_result: TestCodeToolResponse | null = null
  let test_error: KilnError | null = null
  let trust_dialog: CodeTrustDialog
  let edit_inputs_dialog: Dialog
  let abort_controller: AbortController | null = null

  // Track whether a successful test has run since last edit
  export let has_tested = false

  $: schema_model = parse_schema(parameters_schema)
  $: has_params = schema_model?.properties && schema_model.properties.length > 0

  let input_components: Record<string, RunInputFormElement> = {}

  // Stores the last-built param values for preview display
  let param_values: Record<string, unknown> = {}
  // Validation error from building the edit-inputs form. Shown inside the edit
  // dialog so a bad input keeps the dialog open instead of silently closing.
  let input_error: KilnError | null = null

  function parse_schema(
    schema: { [key: string]: unknown } | undefined,
  ): SchemaModelProperty | null {
    if (!schema) return null
    try {
      const model = model_from_schema(
        schema as Parameters<typeof model_from_schema>[0],
      )
      return model
    } catch {
      return null
    }
  }

  function build_params(): { [key: string]: unknown } {
    if (!has_params) return {}
    const params: Record<string, unknown> = {}
    for (const [id, component] of Object.entries(input_components)) {
      if (!component) continue
      const val = component.buildValue()
      if (val !== undefined) {
        params[id] = val
      }
    }
    return params
  }

  function save_and_close_inputs(): boolean {
    try {
      param_values = build_params()
      input_error = null
      return true
    } catch (e) {
      // A required field is missing or an object param is invalid JSON. Keep
      // the dialog open and surface the error so the typed inputs aren't
      // discarded and Run Test can't post stale values.
      input_error = createKilnError(e)
      return false
    }
  }

  function open_edit_inputs() {
    input_error = null
    edit_inputs_dialog.show()
  }

  async function run_test() {
    test_loading = true
    test_result = null
    test_error = null

    const params = { ...param_values }

    const controller = new AbortController()
    abort_controller = controller

    try {
      const { data, error } = await client.POST(
        "/api/projects/{project_id}/test_code_tool",
        {
          params: {
            path: { project_id },
          },
          body: {
            tool_function_name,
            tool_description: tool_description || "test",
            parameters_schema,
            code,
            timeout_seconds,
            tool_allowlist: tool_allowlist || [],
            params,
          },
          signal: controller.signal,
        },
      )

      if (error) {
        throw error
      }

      if (data.not_trusted) {
        trust_dialog.show()
        test_loading = false
        return
      }

      test_result = data
      has_tested = true
      posthog.capture("test_code_tool", {
        project_id,
        tool_function_name,
        success: !data.error,
      })
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        // User cancelled
      } else {
        test_error = createKilnError(e)
      }
    } finally {
      if (abort_controller === controller) {
        test_loading = false
        abort_controller = null
      }
    }
  }

  function cancel_test() {
    if (abort_controller) {
      abort_controller.abort()
    }
  }

  function on_trust_granted() {
    run_test()
  }
</script>

<div class="flex flex-col gap-3" data-testid="code-tool-test-panel">
  <div>
    <h3 class="text-xl font-bold">Test</h3>
    <p class="text-sm text-gray-500 mt-0.5">
      Runs your code live against real tools — side effects included.
    </p>
  </div>

  {#if has_params && schema_model?.properties}
    <div
      class="rounded-lg border border-base-300 bg-base-200/50 p-3 flex flex-col gap-2"
      data-testid="test-input-preview"
    >
      <div class="flex items-center justify-between">
        <span class="font-medium text-sm">Test Input</span>
        <button
          class="link underline text-xs text-gray-500"
          on:click={open_edit_inputs}
          data-testid="edit-inputs-btn"
        >
          Edit
        </button>
      </div>
      <div class="flex flex-col gap-1.5">
        {#each schema_model.properties as prop}
          <div class="text-xs">
            <span class="font-medium text-gray-500">{prop.id}</span>
            <p
              class="text-gray-500 break-words mt-0.5 line-clamp-2"
              title={formatParamPreview(param_values[prop.id])}
            >
              {formatParamPreview(param_values[prop.id]) || "—"}
            </p>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  {#if test_loading}
    <div
      class="flex flex-col items-center gap-3 py-8"
      data-testid="running-state"
    >
      <span class="loading loading-spinner loading-md text-primary"></span>
      <div class="text-sm font-medium">Running...</div>
      <p class="text-xs text-gray-500">Executing your code tool</p>
      <button
        type="button"
        class="btn btn-sm btn-outline mt-1"
        on:click={cancel_test}
        data-testid="cancel-run"
      >
        Cancel
      </button>
    </div>
  {:else if test_result}
    {#if test_result.result !== null && test_result.result !== undefined}
      <div class="flex flex-col gap-1" data-testid="test-output">
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium">Output</span>
          {#if test_result.duration_ms}
            <span class="text-xs text-gray-400"
              >{(test_result.duration_ms / 1000).toFixed(1)}s</span
            >
          {/if}
        </div>
        <Output raw_output={test_result.result} max_height="200px" />
      </div>
    {/if}

    {#if test_result.error}
      <div class="flex flex-col gap-1" data-testid="test-error-result">
        <span class="text-sm font-medium text-error">Error</span>
        <div class="bg-error/10 rounded-lg p-3 text-sm">
          <p class="font-mono text-xs whitespace-pre-wrap">
            {test_result.error}
          </p>
          {#if test_result.traceback}
            <details class="mt-2">
              <summary class="text-xs text-gray-500 cursor-pointer"
                >Traceback</summary
              >
              <pre
                class="text-xs font-mono whitespace-pre-wrap mt-1 text-gray-600">{test_result.traceback}</pre>
            </details>
          {/if}
        </div>
        {#if test_result.duration_ms}
          <span class="text-xs text-gray-400"
            >{(test_result.duration_ms / 1000).toFixed(1)}s</span
          >
        {/if}
      </div>
    {/if}

    <ToolCallLogTable
      entries={test_result.tool_call_log ?? []}
      title="Internal Tool Calls"
      tooltip_text="Calls made by the code tool to other tools. These are not visible to the agent"
    />

    {#if test_result.stdout}
      <details class="text-sm" data-testid="test-stdout">
        <summary class="cursor-pointer text-gray-500 text-xs font-medium"
          >stdout</summary
        >
        <pre
          class="bg-base-200 rounded-lg p-3 text-xs font-mono whitespace-pre-wrap mt-1 max-h-40 overflow-y-auto">{test_result.stdout}</pre>
      </details>
    {/if}

    {#if test_result.stderr}
      <details class="text-sm" data-testid="test-stderr">
        <summary class="cursor-pointer text-gray-500 text-xs font-medium"
          >stderr</summary
        >
        <pre
          class="bg-base-200 rounded-lg p-3 text-xs font-mono whitespace-pre-wrap mt-1 max-h-40 overflow-y-auto">{test_result.stderr}</pre>
      </details>
    {/if}

    <button
      type="button"
      class="btn btn-primary btn-outline btn-sm w-full"
      on:click={run_test}
      data-testid="run-again"
    >
      Run Again
    </button>
  {:else}
    <button
      type="button"
      class="btn btn-primary btn-outline btn-sm w-full"
      on:click={run_test}
      data-testid="run-test-btn"
    >
      Run Test
    </button>
  {/if}

  {#if test_error}
    <div data-testid="test-error">
      <Warning
        warning_color="error"
        inline={true}
        warning_message={test_error.getMessage()}
      />
    </div>
  {/if}
</div>

<!-- Edit inputs modal (wide, mirrors the code-evals browse dialog pattern) -->
<Dialog
  bind:this={edit_inputs_dialog}
  title="Edit Test Input"
  width="wide"
  action_buttons={[
    {
      label: "Done",
      isPrimary: true,
      action: save_and_close_inputs,
    },
  ]}
>
  {#if has_params && schema_model?.properties}
    <div class="flex flex-col gap-4">
      {#each schema_model.properties as prop}
        <RunInputFormElement
          property={prop}
          path={prop.id}
          bind:this={input_components[prop.id]}
        />
      {/each}
      {#if input_error}
        <Warning
          warning_color="error"
          inline={true}
          warning_message={input_error.getMessage()}
        />
      {/if}
    </div>
  {/if}
</Dialog>

<CodeTrustDialog bind:this={trust_dialog} {project_id} {on_trust_granted} />
