<script lang="ts">
  import { goto, pushState } from "$app/navigation"
  import { page } from "$app/stores"
  import AppPage from "../../../../app_page.svelte"
  import FormContainer from "$lib/utils/form_container.svelte"
  import FormElement from "$lib/utils/form_element.svelte"
  import { tool_name_validator } from "$lib/utils/input_validators"
  import SchemaSection from "../../../../../(fullscreen)/setup/(setup)/create_task/schema_section.svelte"
  import CodeEditor from "$lib/components/code_editor.svelte"
  import CodeToolTestPanel from "$lib/components/code_tools/code_tool_test_panel.svelte"
  import CodeTrustDialog from "$lib/components/code_tools/code_trust_dialog.svelte"
  import ToolsSelector from "$lib/ui/run_config_component/tools_selector.svelte"
  import Collapse from "$lib/ui/collapse.svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import { client } from "$lib/api_client"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { onMount } from "svelte"
  import {
    uncache_available_tools,
    available_tools,
    ui_state,
  } from "$lib/stores"
  import posthog from "posthog-js"
  import {
    generateCodeToolPlaceholder,
    generateImportHelper,
    shouldInsertImport,
    generateExamples,
    plainTextParamsSchema,
    resolveStep2Code,
  } from "$lib/utils/code_tool_helpers"

  import { agentInfo } from "$lib/agent"
  $: project_id = $page.params.project_id!
  $: agentInfo.set({
    name: "New Code Tool",
    description: `Create a new Code Tool for project ID ${project_id}. A Python function that runs as a tool.`,
  })

  // Wizard step via pushState shallow routing so browser Back returns to step 1
  $: current_step =
    ($page.state as Record<string, unknown>)?.wizard_step === "code"
      ? "code"
      : "define"

  // Step 1 — Define
  let name = ""
  let tool_function_name = ""
  let tool_description = ""
  let schema_section: SchemaSection

  // Use pre-filled state if coming from a clone (only on initial mount)
  const init_state = $page.state as Record<string, unknown> | undefined
  if (init_state) {
    if (typeof init_state.name === "string") name = init_state.name
    if (typeof init_state.tool_function_name === "string")
      tool_function_name = init_state.tool_function_name
    if (typeof init_state.tool_description === "string")
      tool_description = init_state.tool_description
  }
  let clone_schema_string =
    init_state && typeof init_state.parameters_schema_string === "string"
      ? init_state.parameters_schema_string
      : null
  let clone_code =
    init_state && typeof init_state.code === "string"
      ? (init_state.code as string)
      : null
  let clone_timeout =
    init_state && typeof init_state.timeout_seconds === "number"
      ? (init_state.timeout_seconds as number)
      : 60
  let clone_allowlist =
    init_state && Array.isArray(init_state.tool_allowlist)
      ? (init_state.tool_allowlist as string[])
      : []

  // Step 2 — Code & Test
  let code = ""
  let code_editor: CodeEditor
  let timeout_seconds: number = clone_timeout
  let tool_allowlist: string[] = clone_allowlist
  let parameters_schema: { [key: string]: unknown } = {}
  let generated_placeholder = ""
  let schema_changed_hint = false
  let create_error: KilnError | null = null
  let create_loading = false
  let test_panel_has_tested = false
  let confirm_save_dialog: Dialog
  let create_trust_dialog: CodeTrustDialog
  let examples_dialog: Dialog
  let active_example_tab = 0

  $: examples = generateExamples()

  // warn before unload covers the entire flow (both steps)
  $: has_started =
    name !== "" ||
    tool_function_name !== "" ||
    code !== "" ||
    current_step === "code"

  // Page-level beforeunload guard (covers step 2 which has no FormContainer)
  onMount(() => {
    function handle_before_unload(e: BeforeUnloadEvent) {
      if (has_started) {
        e.preventDefault()
      }
    }
    window.addEventListener("beforeunload", handle_before_unload)
    return () =>
      window.removeEventListener("beforeunload", handle_before_unload)
  })

  function continue_to_code() {
    // Build parameters_schema from SchemaSection
    if (schema_section.is_plaintext()) {
      // Plain text mode: use a static schema with a single `input` string param
      parameters_schema = plainTextParamsSchema()
    } else {
      const schema_str = schema_section.get_schema_string("parameters")
      if (schema_str) {
        try {
          parameters_schema = JSON.parse(schema_str)
        } catch {
          create_error = new KilnError(
            "Parameter schema is invalid JSON.",
            null,
          )
          return
        }
      } else {
        // Structured mode but get_schema_string returned null (e.g. empty)
        parameters_schema = {
          type: "object",
          properties: {},
          required: [],
          additionalProperties: false,
        }
      }
    }

    // Generate placeholder code
    const new_placeholder = generateCodeToolPlaceholder(
      parameters_schema,
      tool_description,
    )
    // Decide the editor contents from the code state itself, not current_step:
    // browser Back pops the shallow-routing step, so current_step reads
    // "define" on a return and would wrongly wipe user code with a placeholder.
    const resolved = resolveStep2Code({
      code,
      newPlaceholder: new_placeholder,
      generatedPlaceholder: generated_placeholder,
      schemaChangedHint: schema_changed_hint,
      cloneCode: clone_code,
    })
    code = resolved.code
    generated_placeholder = resolved.generatedPlaceholder
    schema_changed_hint = resolved.schemaChangedHint
    if (resolved.cloneConsumed) {
      clone_code = null
    }

    // Push a shallow-routing history entry so browser Back returns to step 1
    pushState("", { wizard_step: "code" })
    // Update editor value after switching step
    setTimeout(() => {
      code_editor?.setValue(code)
    }, 0)
  }

  function on_code_change(e: CustomEvent<string>) {
    code = e.detail
    schema_changed_hint = false
    test_panel_has_tested = false
  }

  // Resolve the callable function name (available_tools carries it in
  // function_name, separate from the display name) so the import comment
  // matches what the sandbox dispatches on.
  function resolve_tool_function_name(tool_id: string): string {
    const tool_sets = $available_tools[project_id]
    if (!tool_sets) return tool_id
    for (const ts of tool_sets) {
      for (const t of ts.tools) {
        if (t.id === tool_id) return t.function_name ?? t.name
      }
    }
    const segments = tool_id.split("::")
    return segments[segments.length - 1]
  }

  // Import helper: when tools are selected and code lacks the import
  let prev_allowlist_length = clone_allowlist.length
  $: if (
    tool_allowlist.length > 0 &&
    tool_allowlist.length > prev_allowlist_length
  ) {
    const added_tool_id = tool_allowlist[tool_allowlist.length - 1]
    prev_allowlist_length = tool_allowlist.length
    if (shouldInsertImport(code)) {
      const fn_name = resolve_tool_function_name(added_tool_id)
      const import_block = generateImportHelper(fn_name)
      code = import_block + code
      code_editor?.setValue(code)
    }
  } else {
    prev_allowlist_length = tool_allowlist.length
  }

  function show_examples() {
    active_example_tab = 0
    examples_dialog.show()
  }

  function use_example(): boolean {
    code = examples[active_example_tab].code
    code_editor?.setValue(code)
    return true
  }

  async function handle_create() {
    if (!test_panel_has_tested) {
      confirm_save_dialog.show()
      return
    }
    await do_create()
  }

  async function confirm_save_without_test(): Promise<boolean> {
    await do_create()
    return true
  }

  async function do_create() {
    create_loading = true
    create_error = null

    try {
      const { data, error } = await client.POST(
        "/api/projects/{project_id}/code_tools",
        {
          params: {
            path: { project_id },
          },
          body: {
            name,
            tool_function_name,
            tool_description,
            parameters_schema,
            code,
            timeout_seconds,
            tool_allowlist,
          },
        },
      )

      if (error) {
        throw error
      }

      if (data.not_trusted) {
        create_trust_dialog.show()
        create_loading = false
        return
      }

      posthog.capture("create_code_tool", {
        project_id,
        tool_function_name,
      })

      uncache_available_tools(project_id)
      goto(`/tools/${project_id}/code_tools/${data.id}`)
    } catch (e) {
      create_error = createKilnError(e)
    } finally {
      create_loading = false
    }
  }

  function on_create_trust_granted() {
    do_create()
  }
</script>

<div class="max-w-[1400px]">
  {#if current_step === "define"}
    <AppPage
      title="New Code Tool"
      subtitle="Define the tool's identity and parameters"
      breadcrumbs={[
        {
          label: "Optimize",
          href: `/optimize/${project_id}/${$ui_state.current_task_id}`,
        },
        {
          label: "Tools",
          href: `/tools/${project_id}`,
        },
        {
          label: "Add Tools",
          href: `/tools/${project_id}/add_tools`,
        },
      ]}
    >
      <div class="max-w-2xl">
        <FormContainer
          submit_label="Continue"
          on:submit={continue_to_code}
          warn_before_unload={has_started}
        >
          <FormElement
            id="name"
            label="Display Name"
            description="User-facing name for this tool."
            inputType="input"
            bind:value={name}
            placeholder="e.g. User Lookup"
          />

          <FormElement
            id="tool_function_name"
            label="Tool Name"
            description="The function name exposed to the model. Must be lowercase with underscores."
            inputType="input"
            bind:value={tool_function_name}
            placeholder="e.g. get_user"
            validator={tool_name_validator}
          />

          <FormElement
            id="tool_description"
            label="Description"
            description="Shown to the model — describe what this tool does and when to use it."
            inputType="textarea"
            bind:value={tool_description}
            placeholder="e.g. Look up a user by ID and return their profile information."
          />

          <FormElement
            id="parameters"
            label="Parameters"
            description="Define the parameters the model will pass to this tool."
            info_description="Parameters use JSON Schema. Each parameter has a name, type, and optional description. The model uses these to know what arguments to provide."
            inputType="header_only"
            value=""
          />
          <SchemaSection
            bind:this={schema_section}
            schema_string={clone_schema_string}
            warn_about_required={true}
            structured_label="Structured Parameter List"
          />
        </FormContainer>
      </div>
    </AppPage>
  {:else}
    <AppPage
      title="New Code Tool"
      subtitle="Write a tool that invokes a Python function, optionally calling other tools."
      breadcrumbs={[
        {
          label: "Optimize",
          href: `/optimize/${project_id}/${$ui_state.current_task_id}`,
        },
        {
          label: "Tools",
          href: `/tools/${project_id}`,
        },
        {
          label: "Add Tools",
          href: `/tools/${project_id}/add_tools`,
        },
      ]}
    >
      <div class="flex flex-col lg:flex-row gap-8 lg:gap-16">
        <!-- Left: Code Editor -->
        <div class="flex-1 min-w-0 flex flex-col gap-4">
          <h3 class="text-xl font-bold">Code</h3>
          <FormElement
            id="code_editor_label"
            label="Python Function"
            description="Write a Python function named 'run' that implements your tool."
            info_description="Must include either `def run(...):` or `async def run(...):`."
            inputType="header_only"
            inline_action={{
              handler: show_examples,
              label: "Examples",
            }}
            value=""
          />

          {#if schema_changed_hint}
            <div
              class="text-xs text-warning bg-warning/10 rounded px-3 py-1.5"
              data-testid="schema-changed-hint"
            >
              Schema changed — check <code>run()</code>'s parameters.
            </div>
          {/if}

          <CodeEditor
            bind:this={code_editor}
            value={code}
            min_height="400px"
            on:change={on_code_change}
          />

          <Collapse title="Advanced Options">
            <div class="flex flex-col gap-4 py-2">
              <FormElement
                id="timeout_seconds"
                label="Timeout (seconds)"
                description="Maximum time allowed for the tool to execute."
                inputType="input_number"
                bind:value={timeout_seconds}
                placeholder="60"
                min={1}
              />
            </div>
          </Collapse>

          {#if create_error}
            <div class="mt-2">
              <div
                class="bg-error/10 text-error rounded-lg p-3 text-sm"
                data-testid="create-error"
              >
                {create_error.getMessage()}
              </div>
            </div>
          {/if}

          <button
            class="btn btn-primary w-full mt-2"
            on:click={handle_create}
            disabled={create_loading}
            data-testid="create-btn"
          >
            {#if create_loading}
              <span class="loading loading-spinner loading-sm"></span>
            {/if}
            Create Code Tool
          </button>
        </div>

        <!-- Right: Tools + Test Panel -->
        <div class="w-full lg:w-[420px] flex flex-col gap-6">
          <div>
            <h3 class="text-xl font-bold mb-3">Tool Access</h3>
            <ToolsSelector
              {project_id}
              label="Tools"
              settings={{
                description: "The code can only call tools listed here.",
                info_description:
                  "Select the tools this code tool is allowed to call. Tool calls use the kiln.tools or kiln.async_tools module.",
                hide_create_kiln_task_tool_button: true,
                optional: true,
                empty_label: "None (no tool access)",
                sandbox_code_context: "code_tool",
              }}
              bind:tools={tool_allowlist}
            />
          </div>

          <div class="border-t pt-4">
            <CodeToolTestPanel
              {project_id}
              {tool_function_name}
              {tool_description}
              {parameters_schema}
              {code}
              {timeout_seconds}
              {tool_allowlist}
              bind:has_tested={test_panel_has_tested}
            />
          </div>
        </div>
      </div>
    </AppPage>

    <Dialog
      bind:this={confirm_save_dialog}
      title="Save Without Testing?"
      action_buttons={[
        {
          label: "Save Anyway",
          isWarning: true,
          asyncAction: confirm_save_without_test,
        },
      ]}
    >
      <p class="text-sm">
        You haven't run a successful test since your last edit. It's recommended
        to test your code before saving.
      </p>
    </Dialog>

    <CodeTrustDialog
      bind:this={create_trust_dialog}
      {project_id}
      on_trust_granted={on_create_trust_granted}
    />

    <Dialog
      bind:this={examples_dialog}
      title="Code Tool Examples"
      width="wide"
      action_buttons={[
        {
          label: "Use This Example",
          isPrimary: true,
          action: use_example,
        },
      ]}
    >
      <div class="flex flex-col gap-4">
        <div class="tabs tabs-bordered">
          {#each examples as example, i}
            <button
              type="button"
              class="tab {active_example_tab === i ? 'tab-active' : ''}"
              on:click={() => (active_example_tab = i)}
            >
              {example.label}
            </button>
          {/each}
        </div>
        <div
          class="bg-base-200 rounded-lg p-4 overflow-x-auto font-mono text-sm whitespace-pre"
        >
          {examples[active_example_tab].code}
        </div>
      </div>
    </Dialog>
  {/if}
</div>
