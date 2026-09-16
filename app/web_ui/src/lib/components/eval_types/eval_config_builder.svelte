<script lang="ts">
  import FormContainer from "$lib/utils/form_container.svelte"
  import { page } from "$app/stores"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import type { Eval, Task, EvalConfigType, TaskRunOutput } from "$lib/types"
  import type { components } from "$lib/api_schema"
  import { goto } from "$app/navigation"
  import posthog from "posthog-js"
  import { set_current_eval_config } from "$lib/stores/evals_store"
  import LlmJudgeForm from "$lib/components/eval_types/llm_judge_form.svelte"
  import JudgeConfigFields from "$lib/components/eval_types/judge_config_fields.svelte"
  import {
    getV2EvalTypeMetadata,
    manualExampleSupport,
    type V2EvalType,
    type EvalTypeFormApi,
  } from "$lib/utils/eval_types/registry"
  import {
    createEvalConfig,
    createLlmJudgeConfig,
    testV2Eval,
    testV2EvalLlmJudge,
    fetchTaskRuns,
    checkAddCodeTrust,
    addCodeTrust,
    type EvalTaskInput,
    type TestV2EvalResponse,
  } from "$lib/api/v2_eval_api"
  import { validate_result_shape } from "$lib/utils/eval_types/test_run_shape"
  import { select_default_test_run } from "$lib/utils/eval_types/test_run_selection"
  import {
    parse_reference_data,
    parse_reference_keys,
  } from "$lib/utils/eval_types/reference_data_input"
  import Dialog from "$lib/ui/dialog.svelte"
  import TrustCodeDialog from "$lib/components/eval_types/trust_code_dialog.svelte"
  import { onMount } from "svelte"
  import EvalTypeIntro from "$lib/components/eval_types/eval_type_intro.svelte"
  import EvalTestRunPane from "$lib/components/eval_types/test_run/eval_test_run_pane.svelte"
  import {
    uses_reference_data_llm_judge,
    uses_reference_data_code_eval,
  } from "$lib/utils/eval_types/reference_data_gate"
  import { SHOW_REFERENCE_DATA_UI } from "$lib/utils/eval_types/reference_data_ui"

  export let eval_config_type: V2EvalType
  export let evaluator: Eval
  export let task: Task
  export let project_id: string
  export let task_id: string
  export let eval_id: string
  export let spec_id: string

  $: metadata = getV2EvalTypeMetadata(eval_config_type)

  // LLM judge form bindings
  let llm_model_name: string | undefined = undefined
  let llm_provider_name: string | undefined = undefined
  let llm_combined_model_name: string | undefined = undefined
  let llm_selected_algo: EvalConfigType | undefined = undefined
  let llm_judge_prompt: string | undefined = undefined
  let llm_system_prompt: string | undefined = undefined
  // The Evaluation Instructions steps for evals with no derivable prompt.
  // Create-only today: this builder is only reached from create_eval_config,
  // so there's no persisted value to load. If a judge-edit path is ever
  // added, this must be initialized from the config's judge_instructions.
  let llm_judge_instructions: string[] = [""]
  // Filled by LlmJudgeForm from the default-prompt endpoint. The server decides what a
  // judge for this eval requires; the Test Judge pane offers a place to supply it
  // rather than re-deriving the rule from the prompt's text.
  let llm_server_reference_keys: string[] = []
  let llm_default_prompt_unavailable = false

  $: judge_reference_signals = {
    prompt_template: llm_judge_prompt ?? "",
    server_reference_keys: llm_server_reference_keys,
    prompt_unavailable: llm_default_prompt_unavailable,
  }

  function cleaned_judge_instructions(): string[] | null {
    const cleaned = llm_judge_instructions
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
    return cleaned.length > 0 ? cleaned : null
  }

  // Code eval form binding — tracks code edits reactively for the save gate.
  // Starts undefined so the child's initial value flows up via bind:.
  let code_eval_code: string | undefined = undefined

  // Form component references -- bind:this on svelte:component yields a
  // generic component instance, so we keep a loose ref and cast it.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let v2FormComponentRef: any
  $: v2FormComponent = v2FormComponentRef as EvalTypeFormApi | undefined

  // Save state
  let create_evaluator_error: KilnError | null = null
  let create_evaluator_loading = false
  let complete = false

  // Test-run panel state
  let available_runs: TaskRunOutput[] = []
  let runs_loading = true
  let runs_error: KilnError | null = null
  let selected_task_run: TaskRunOutput | null = null
  let advanced_reference_data = ""
  let test_loading = false
  let test_error: KilnError | null = null
  let test_result: TestV2EvalResponse | null = null
  let test_shape_warning: string | null = null
  let test_score_range_warning: string | null = null
  let test_abort_controller: AbortController | null = null

  // Trust, confirm, and test-required modal refs
  let trust_dialog: TrustCodeDialog
  let confirm_save_dialog: Dialog
  let test_required_dialog: Dialog
  let form_container: FormContainer

  // Pending action after trust grant: "test" or "save"
  let pending_trust_action: "test" | "save" | null = null

  // Reference data candidate keys for dropdown (parsed from test run panel)
  let reference_candidate_keys: string[] = []
  $: reference_candidate_keys = parse_reference_keys(advanced_reference_data)

  // Whether the current config uses reference_data (drives the test-before-save gate).
  // Both llm_judge_prompt and code_eval_code are direct reactive dependencies so
  // Svelte's $: tracking re-evaluates when the user edits either one.
  $: config_uses_reference_data = compute_uses_reference_data(
    eval_config_type,
    llm_judge_prompt,
    code_eval_code,
  )

  function compute_uses_reference_data(
    type: V2EvalType,
    judge_prompt: string | undefined,
    code: string | undefined,
  ): boolean {
    // While reference data is hidden from the UI there's no way to supply it in
    // the Test Judge pane, so the test-before-save gate can never be satisfied.
    // Report "unused" to keep the normal save flow, even if the user hand-wrote
    // a reference_data lookup into their prompt or code.
    if (!SHOW_REFERENCE_DATA_UI) {
      return false
    }
    if (type === "llm_judge") {
      return uses_reference_data_llm_judge(judge_prompt ?? "")
    }
    if (type === "code_eval") {
      return uses_reference_data_code_eval(code ?? "")
    }
    return false
  }

  // Unified tested-state. A passing test only counts for the exact config that
  // produced it: `config_version` bumps on every edit, and each run records the
  // version it tested against (captured before the request's await, so an edit
  // while a test is in flight invalidates it on arrival). The result counts only
  // while the version is unchanged, so any later edit re-arms the
  // test-before-save gates.
  let config_version = 0
  let last_tested: { version: number; passed: boolean } | null = null

  $: test_valid_for_current_config =
    !!last_tested &&
    last_tested.passed &&
    last_tested.version === config_version

  // Required reference fields surfaced by the active form.
  // Only the deterministic forms (exact_match, contains, set_check) bind
  // this; reset when switching to any other eval type so stale values
  // from a previous form don't leak.
  let required_reference_fields: string[] = []
  $: if (
    eval_config_type !== "exact_match" &&
    eval_config_type !== "contains" &&
    eval_config_type !== "set_check"
  ) {
    required_reference_fields = []
  }

  $: manual_example_support = manualExampleSupport(eval_config_type)

  // Unsaved-changes guard + tested-state invalidation. Every real edit arms the
  // guard and bumps config_version, so a prior passing test no longer counts.
  let has_typed = false

  function on_config_edit() {
    has_typed = true
    config_version++
  }

  // Model/algo selection uses callback props rather than DOM events, so the
  // wrapper's on:input/on:change can't see them; watch the bound values to
  // register those edits too.
  $: if (llm_combined_model_name || llm_selected_algo) on_config_edit()

  $: is_llm_judge = eval_config_type === "llm_judge"
  $: can_submit_v2 = !!eval_config_type && !is_llm_judge
  $: can_submit_llm =
    is_llm_judge && !!llm_selected_algo && !!llm_combined_model_name
  $: can_submit = can_submit_v2 || can_submit_llm

  function handleKeydown(event: KeyboardEvent) {
    if (!can_submit || create_evaluator_loading) return
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault()
      form_container.validate_and_submit()
    }
  }

  onMount(async () => {
    await load_task_runs()
  })

  async function load_task_runs() {
    try {
      runs_loading = true
      runs_error = null
      available_runs = await fetchTaskRuns(project_id, task_id)
      selected_task_run = select_default_test_run(available_runs)
    } catch (e) {
      runs_error = createKilnError(e)
    } finally {
      runs_loading = false
    }
  }

  function build_eval_input(): EvalTaskInput | null {
    if (!selected_task_run) return null

    const eval_input: EvalTaskInput = {
      final_message: selected_task_run.output?.output ?? "",
    }

    if (selected_task_run.input) {
      eval_input.task_input = selected_task_run.input
    }

    if (selected_task_run.trace) {
      eval_input.trace = selected_task_run.trace as {
        [key: string]: unknown
      }[]
    }

    const reference = parse_reference_data(advanced_reference_data)
    if (!reference.ok) {
      test_error = createKilnError(new Error(reference.error))
      return null
    }
    if (reference.data) {
      eval_input.reference_data = reference.data
    }

    return eval_input
  }

  async function run_test() {
    // Clear all prior test state up front so a re-run never leaves a stale
    // result/warning/error on screen — even when an early-return validation
    // path fires before the test runs.
    test_error = null
    test_result = null
    last_tested = null
    test_shape_warning = null
    test_score_range_warning = null

    const eval_input = build_eval_input()
    if (!eval_input) return

    if (is_llm_judge) {
      if (!llm_model_name || !llm_provider_name || !llm_selected_algo) {
        test_error = createKilnError(
          new Error("Please select a model and algorithm first."),
        )
        return
      }
    } else {
      if (!v2FormComponent) return
      if (v2FormComponent.validate) {
        const validation_error = v2FormComponent.validate()
        if (validation_error) {
          test_error = createKilnError(new Error(validation_error))
          return
        }
      }
    }

    const controller = new AbortController()
    test_abort_controller = controller

    // Capture the config version BEFORE the await so an edit made while the
    // request is in flight can't be mis-stamped as tested-and-passed.
    const tested_version = config_version

    try {
      test_loading = true

      let result: TestV2EvalResponse

      if (is_llm_judge) {
        const g_eval = llm_selected_algo === "g_eval"
        result = await testV2EvalLlmJudge(
          project_id,
          task_id,
          eval_id,
          {
            model_name: llm_model_name!,
            provider:
              llm_provider_name! as components["schemas"]["ModelProviderName"],
            g_eval,
            judge_prompt: llm_judge_prompt ?? null,
            system_prompt: llm_system_prompt ?? null,
            judge_instructions: cleaned_judge_instructions(),
          },
          eval_input,
          controller.signal,
        )
      } else {
        const properties = v2FormComponent!.getProperties()
        result = await testV2Eval(
          project_id,
          task_id,
          eval_id,
          {
            properties,
            eval_input,
          },
          controller.signal,
        )
      }

      if (
        result.skipped_reason === "code_eval_not_trusted" &&
        metadata?.requiresTrust
      ) {
        pending_trust_action = "test"
        trust_dialog.show()
        test_loading = false
        return
      }

      test_result = result

      if (result.scores && !result.skipped_reason) {
        const shape = validate_result_shape(
          result.scores,
          evaluator?.output_scores,
        )
        test_shape_warning = shape.message

        let passed = shape.valid
        if (result.score_range_errors && result.score_range_errors.length > 0) {
          test_score_range_warning = result.score_range_errors.join("; ")
          passed = false
        }

        if (passed) {
          last_tested = { version: tested_version, passed: true }
        }
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        // User cancelled -- not an error
      } else {
        test_error = createKilnError(e)
      }
    } finally {
      if (test_abort_controller === controller) {
        test_loading = false
        test_abort_controller = null
      }
    }
  }

  function cancel_test() {
    if (test_abort_controller) {
      test_abort_controller.abort()
      test_abort_controller = null
    }
    test_loading = false
  }

  async function grant_trust_and_retry(): Promise<boolean> {
    try {
      await addCodeTrust(project_id)
    } catch (e) {
      test_error = createKilnError(e)
      return false
    }
    const action = pending_trust_action
    pending_trust_action = null
    if (action === "test") {
      run_test()
    } else if (action === "save") {
      do_save()
    }
    return true
  }

  async function handle_submit() {
    // Clear any prior error so a re-submit after a fix starts clean.
    create_evaluator_error = null

    if (metadata?.requiresTrust) {
      try {
        const trust_response = await checkAddCodeTrust(project_id)
        if (!trust_response.trusted) {
          pending_trust_action = "save"
          create_evaluator_loading = false
          trust_dialog.show()
          return
        }
      } catch (e) {
        create_evaluator_loading = false
        create_evaluator_error = createKilnError(e)
        return
      }
    }

    // Surface form validation errors before prompting to save without testing.
    // A hard error means the config can't be saved at all, so the
    // "Save Without Testing?" confirmation shouldn't appear yet.
    if (!is_llm_judge && v2FormComponent?.validate) {
      const validation_error = v2FormComponent.validate()
      if (validation_error) {
        create_evaluator_loading = false
        create_evaluator_error = createKilnError(new Error(validation_error))
        return
      }
    }

    // When the config uses reference_data, require a passing test with
    // current prompt/code AND reference data before allowing save.
    if (config_uses_reference_data) {
      if (!test_valid_for_current_config) {
        create_evaluator_loading = false
        test_required_dialog.show()
        return
      }
    } else if (!test_valid_for_current_config) {
      create_evaluator_loading = false
      confirm_save_dialog.show()
      return
    }

    await do_save()
  }

  async function do_save() {
    try {
      create_evaluator_loading = true
      create_evaluator_error = null

      let data: components["schemas"]["EvalConfig"]

      if (is_llm_judge) {
        if (!llm_model_name || !llm_provider_name || !llm_selected_algo) {
          throw new Error("No model or algorithm selected")
        }
        const g_eval = llm_selected_algo === "g_eval"
        data = await createLlmJudgeConfig(project_id, task_id, eval_id, {
          model_name: llm_model_name,
          provider:
            llm_provider_name as components["schemas"]["ModelProviderName"],
          g_eval,
          judge_prompt: llm_judge_prompt ?? null,
          system_prompt: llm_system_prompt ?? null,
          judge_instructions: cleaned_judge_instructions(),
        })
      } else if (eval_config_type && v2FormComponent) {
        if (v2FormComponent.validate) {
          const validation_error = v2FormComponent.validate()
          if (validation_error) {
            throw new Error(validation_error)
          }
        }
        const properties = v2FormComponent.getProperties() as Record<
          string,
          unknown
        >
        if (eval_config_type === "code_eval") {
          // code_eval is the one type whose reference_keys the client owns: they name
          // what the user's scoring code reads, which only this page knows. An
          // llm_judge's are derived server-side from the eval, so its request above
          // carries none.
          properties.reference_keys = config_uses_reference_data
            ? reference_candidate_keys
            : []
        }
        data = await createEvalConfig(project_id, task_id, eval_id, {
          type: "v2",
          properties,
          model_name: null,
          provider: null,
        })
      } else {
        throw new Error("No eval type selected")
      }

      posthog.capture("create_eval_config", {
        config_type: "v2",
        v2_type: eval_config_type,
        ...(is_llm_judge
          ? { model_name: llm_model_name, provider_name: llm_provider_name }
          : {}),
      })

      const save_as_default = $page.url.searchParams.get("save_as_default")
      if (data.id && save_as_default === "true") {
        try {
          await set_current_eval_config(project_id, task_id, eval_id, data.id)
        } catch (e) {
          console.error("Failed to set as default:", e)
        }
      }

      complete = true
      const next_page = $page.url.searchParams.get("next_page")
      if (next_page === "eval_configs") {
        goto(
          `/specs/${project_id}/${task_id}/${spec_id}/${eval_id}/eval_configs`,
        )
      } else if (next_page === "compare_run_configs") {
        goto(
          `/specs/${project_id}/${task_id}/${spec_id}/${eval_id}/compare_run_configs`,
        )
      } else {
        goto(
          `/specs/${project_id}/${task_id}/${spec_id}/${eval_id}?selected_eval_config=${data.id}`,
        )
      }
    } catch (e) {
      create_evaluator_error = createKilnError(e)
    } finally {
      create_evaluator_loading = false
    }
  }

  function select_task_run(run: TaskRunOutput) {
    selected_task_run = run
    test_result = null
    last_tested = null
    test_shape_warning = null
    test_score_range_warning = null
    test_error = null
  }
</script>

<svelte:window on:keydown={handleKeydown} />

<!--
  Grid so the intro spans only the form column (row 1, col 1) while the
  Judge Configuration and Test Judge panes sit side-by-side on row 2 at the
  same level. Collapses to a single column below xl.

  FormContainer wraps only the left column so its <form> boundary and
  FormElement validators don't span into the test-run pane.
-->
<div
  class="grid grid-cols-1 gap-y-6 xl:gap-x-16 xl:items-start xl:grid-cols-[minmax(0,1fr)_18rem] 2xl:grid-cols-[minmax(0,1fr)_24rem]"
>
  {#if metadata}
    <div class="min-w-0 xl:col-start-1 xl:row-start-1">
      <EvalTypeIntro evalType={eval_config_type} {metadata} />
    </div>
  {/if}

  <!-- Left: form (inside FormContainer so validation scopes to config fields only) -->
  <div class="min-w-0 xl:col-start-1 xl:row-start-2">
    <FormContainer
      bind:this={form_container}
      submit_visible={false}
      keyboard_submit={false}
      focus_on_mount={false}
      submit_label="Save"
      on:submit={handle_submit}
      bind:error={create_evaluator_error}
      bind:submitting={create_evaluator_loading}
      warn_before_unload={!complete && !!eval_config_type && has_typed}
    >
      <!--
        Catch every real form interaction for the unsaved-changes guard and to
        invalidate a prior test. on:input covers typing; on:change covers native
        selects, checkboxes, radios, and fancy_select dropdowns (which emit a
        bubbling change on pick).
      -->
      <div
        class="flex flex-col gap-6"
        on:input={on_config_edit}
        on:change={on_config_edit}
      >
        <div>
          <div class="text-xl font-bold">Judge Configuration</div>
        </div>

        {#if is_llm_judge}
          <LlmJudgeForm
            {task_id}
            {project_id}
            {eval_id}
            bind:model_name={llm_model_name}
            bind:provider_name={llm_provider_name}
            bind:combined_model_name={llm_combined_model_name}
            bind:selected_algo={llm_selected_algo}
            bind:judge_prompt={llm_judge_prompt}
            bind:system_prompt={llm_system_prompt}
            bind:judge_instructions={llm_judge_instructions}
            bind:default_reference_keys={llm_server_reference_keys}
            bind:default_prompt_unavailable={llm_default_prompt_unavailable}
          />
        {:else}
          <JudgeConfigFields
            bind:this={v2FormComponentRef}
            {eval_config_type}
            {project_id}
            {reference_candidate_keys}
            output_scores={evaluator?.output_scores}
            bind:code_string={code_eval_code}
            bind:required_reference_fields
          />
        {/if}

        {#if can_submit}
          <button
            type="button"
            class="btn btn-primary w-full"
            data-testid="column-save-button"
            disabled={create_evaluator_loading}
            on:click={() => form_container.validate_and_submit()}
          >
            {#if create_evaluator_loading}
              <span class="loading loading-spinner loading-md"></span>
            {:else}
              Save
            {/if}
          </button>
        {/if}
      </div>
    </FormContainer>
  </div>

  <!-- Right: test run pane (outside FormContainer — not validated on save) -->
  <div class="min-w-0 xl:col-start-2 xl:row-start-2">
    <EvalTestRunPane
      {project_id}
      {task_id}
      {eval_config_type}
      {runs_loading}
      {runs_error}
      {available_runs}
      selected_run={selected_task_run}
      reference_data={advanced_reference_data}
      {required_reference_fields}
      {test_loading}
      {test_result}
      {test_error}
      {test_shape_warning}
      {test_score_range_warning}
      test_has_valid_run={test_valid_for_current_config}
      {is_llm_judge}
      {can_submit_llm}
      {judge_reference_signals}
      manual_example_supported={manual_example_support.supported}
      on:select={(e) => select_task_run(e.detail)}
      on:run={run_test}
      on:cancel={cancel_test}
      on:updateReferenceData={(e) => {
        advanced_reference_data = e.detail
        // A reference-data edit is a config edit: invalidate any prior test.
        on_config_edit()
      }}
      on:runAgain={run_test}
    />
  </div>
</div>

<TrustCodeDialog bind:this={trust_dialog} on_trust={grant_trust_and_retry} />

<Dialog
  bind:this={confirm_save_dialog}
  title="Save Without Testing?"
  action_buttons={[
    {
      label: "Save Anyway",
      isError: true,
      asyncAction: async () => {
        await do_save()
        return true
      },
    },
  ]}
>
  <p class="text-sm text-gray-500">
    You haven't tested this judge yet. Running a quick test helps catch issues
    before saving. Are you sure you want to save without testing?
  </p>
</Dialog>

<Dialog
  bind:this={test_required_dialog}
  title="Test Required"
  action_buttons={[
    {
      label: "OK",
      isPrimary: true,
      action: () => true,
    },
  ]}
>
  <p class="text-sm text-gray-500">
    You must successfully run your judge once in the <code
      class="bg-base-200 px-1 py-0.5 rounded text-xs font-mono">Test Judge</code
    >
    panel before saving. We must check your
    <code class="bg-base-200 px-1 py-0.5 rounded text-xs font-mono"
      >reference_data</code
    > code works properly.
  </p>
</Dialog>
