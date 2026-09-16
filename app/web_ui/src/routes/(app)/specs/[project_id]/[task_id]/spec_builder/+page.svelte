<script lang="ts">
  import AppPage from "../../../../app_page.svelte"
  import { page } from "$app/stores"
  import { onMount, tick } from "svelte"
  import { autofillSpecName } from "$lib/utils/formatters"
  import { normalize_filename_string } from "$lib/utils/input_validators"
  import { createKilnError, KilnError } from "$lib/utils/error_handlers"
  import type { SpecType, SpecProperties, Priority } from "$lib/types"
  import { goto } from "$app/navigation"
  import { spec_field_configs } from "../select_template/spec_templates"
  import {
    buildSpecDefinition,
    implied_judge_for_spec_type,
    eval_template_for_spec_type,
  } from "../spec_utils"
  import { client } from "$lib/api_client"
  import { load_task } from "$lib/stores"
  import { set_current_eval_config } from "$lib/stores/evals_store"
  import {
    createEvalConfig,
    createEvaluator,
    checkAddCodeTrust,
    addCodeTrust,
  } from "$lib/api/v2_eval_api"
  import {
    ALL_V2_EVAL_TYPES,
    getV2EvalTypeMetadata,
    type V2EvalType,
  } from "$lib/utils/eval_types/registry"
  import TrustCodeDialog from "$lib/components/eval_types/trust_code_dialog.svelte"
  import CreateSpecForm from "./create_spec_form.svelte"
  import CreateSpecJudgeForm from "./create_spec_judge_form.svelte"
  import posthog from "posthog-js"
  import { agentInfo } from "$lib/agent"

  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  $: agentInfo.set({
    name: "Evals",
    description: `Guided eval builder for project ID ${project_id}, task ID ${task_id}. Step-by-step creation of evals with requirements and test cases.`,
  })

  // Form state
  let spec_type: SpecType = "desired_behaviour"
  // Judge-only mode: no template was picked (programmatic checks route here
  // with just a judge param); the eval is created spec-less and template-less.
  let judge_only = false
  let name = ""
  let property_values: Record<string, string | null> = {}
  let initial_property_values: Record<string, string | null> = {}
  let evaluate_full_trace = false
  let priority: Priority = 1

  // The judge chosen on the previous screen. Null means the LLM-judge flow (the
  // full template form, saved as a spec); any other type means a judge that
  // never reads a written rubric, so no spec is created at all -- just an eval
  // with the judge attached.
  let judge_type: V2EvalType | null = null
  let judge_form: CreateSpecJudgeForm
  let trust_dialog: TrustCodeDialog

  $: is_non_llm_judge = judge_type !== null && judge_type !== "llm_judge"

  // Tool-call and step-count judges score the agent's execution trace, so their
  // evals have to be created against the full history, not just the final answer.
  $: judge_requires_full_trace =
    judge_type === "tool_call_check" || judge_type === "step_count_check"

  // Loading/error state
  let loading = true
  let loading_error: KilnError | null = null
  let error: KilnError | null = null

  // Submission state
  let submitting = false
  let saving_spec = false
  let complete = false

  // Get field configs for the current spec_type
  $: field_configs = spec_field_configs[spec_type] || []

  $: is_tool_use_spec = spec_type === "appropriate_tool_use"
  // No UI toggles evaluation_data_type for v2 judges (their runtime input
  // always carries the trace; deterministic judges pick their source via
  // "Output to Check"). It's still recorded as full_trace for trace-reading
  // judges so their eval runs store the trace snapshot.
  $: if (is_tool_use_spec || judge_requires_full_trace)
    evaluate_full_trace = true

  // Initialize form from URL params
  async function initialize() {
    loading = true
    loading_error = null

    try {
      // Confirm the route's task exists before showing the form, so a stale
      // link surfaces an error here instead of failing on save.
      const task = await load_task(project_id, task_id)
      if (!task) {
        throw new KilnError("Failed to load task.")
      }

      // Get spec type from URL params. Without one, a valid non-LLM judge
      // param enters judge-only mode: a template-less eval built from just a
      // name and the judge's config (the programmatic checks on the template
      // picker route here). Anything else restarts at the evals page.
      const spec_type_param = $page.url.searchParams.get("type")
      if (!spec_type_param) {
        const judge_param = $page.url.searchParams.get("judge")
        const judge = ALL_V2_EVAL_TYPES.includes(judge_param as V2EvalType)
          ? (judge_param as V2EvalType)
          : null
        if (judge !== null && judge !== "llm_judge") {
          judge_only = true
          judge_type = judge
          return
        }
        complete = true
        goto(`/specs/${project_id}/${task_id}`)
        return
      }

      spec_type = spec_type_param as SpecType
      name = autofillSpecName(spec_type)

      // Every template's judge is implied; the URL's judge param is ignored
      // for templated flows so a hand-edited param can't turn a rubric-only
      // template into a spec-less deterministic eval (or bypass the pinned
      // tool call judge).
      judge_type = implied_judge_for_spec_type(spec_type)

      // Initialize property values from field configs. Fields the user never sees
      // (because a non-LLM judge is scoring) still save their template default, so
      // the spec stays valid and reads sensibly on the eval's detail page.
      const fieldConfigs = spec_field_configs[spec_type] || []
      const values: Record<string, string | null> = {}

      for (const field of fieldConfigs) {
        if (field.default_value !== undefined) {
          values[field.key] = field.default_value
        }
      }

      property_values = values
      initial_property_values = { ...values }
    } catch (e) {
      loading_error = createKilnError(e)
    } finally {
      loading = false
    }
  }

  onMount(() => {
    initialize()
  })

  // Handler for creating a spec from the template form
  async function handle_create_spec() {
    error = null
    try {
      saving_spec = true

      // Normalize the spec name before saving
      name = normalize_filename_string(name)

      // Build definition and properties on the client side
      const definition = buildSpecDefinition(spec_type, property_values)

      // Build properties object with spec_type, filtering out null and empty values
      const filteredValues = Object.fromEntries(
        Object.entries(property_values).filter(
          ([_, value]) => value !== null && value.trim() !== "",
        ),
      )
      const properties = {
        spec_type: spec_type,
        ...filteredValues,
      } as SpecProperties

      const { data, error: api_error } = await client.POST(
        "/api/projects/{project_id}/tasks/{task_id}/specs",
        {
          params: { path: { project_id, task_id } },
          body: {
            name,
            definition,
            properties,
            evaluate_full_trace,
            priority: Number(priority) as 0 | 1 | 2 | 3,
            status: "active",
            task_sample: null,
          },
        },
      )
      if (api_error) throw api_error

      const spec_id = data?.id
      if (!spec_id) {
        throw new KilnError("Failed to create eval. Please try again.")
      }

      posthog.capture("create_spec", {
        spec_type: spec_type,
        with_copilot: false,
      })

      complete = true
      goto(`/specs/${project_id}/${task_id}/${spec_id}`)
    } catch (e) {
      error = createKilnError(e)
    } finally {
      submitting = false
      saving_spec = false
    }
  }

  // Handler for creating an eval together with a non-LLM judge. No spec is
  // created: the judge never reads a written rubric, so there are no spec
  // fields worth asking for. The eval carries the template, priority, and the
  // judge config, and reads/edits work exactly like any other spec-less eval.
  async function handle_create_eval_with_judge() {
    error = null
    try {
      // Validate the judge before writing anything: an eval saved with a judge
      // we then fail to create would strand the user on a half-built eval.
      const judge_error = judge_form.validateJudge()
      if (judge_error) {
        error = new KilnError(judge_error)
        await tick() // Let FormContainer apply submitting=true before we clear it
        return
      }

      // Code judges execute Python locally, so get consent before creating anything.
      if (judge_type && getV2EvalTypeMetadata(judge_type).requiresTrust) {
        const trust = await checkAddCodeTrust(project_id)
        if (!trust.trusted) {
          trust_dialog.show()
          return
        }
      }

      // Read the judge's values while its form is still on screen: the saving
      // spinner replaces it, and a destroyed component can't be read.
      const judge_properties = judge_form.getJudgeProperties()

      saving_spec = true
      name = normalize_filename_string(name)

      const evaluator = await createEvaluator(project_id, task_id, {
        name,
        template: judge_only ? null : eval_template_for_spec_type(spec_type),
        evaluation_data_type: evaluate_full_trace
          ? "full_trace"
          : "final_answer",
        priority: Number(priority) as 0 | 1 | 2 | 3,
        status: "active",
      })
      if (!evaluator.id) {
        throw new KilnError("Failed to create eval. Please try again.")
      }

      // Attach the judge and make it the default, so the eval is ready to run
      // instead of landing on "Not Ready - Configure". If this fails, delete
      // the eval again: a judge-less orphan would also squat on the tag
      // filters generated from its name, breaking a retry under the same name.
      try {
        const eval_config = await createEvalConfig(
          project_id,
          task_id,
          evaluator.id,
          {
            type: "v2",
            properties: judge_properties,
            model_name: null,
            provider: null,
          },
        )
        if (eval_config.id) {
          await set_current_eval_config(
            project_id,
            task_id,
            evaluator.id,
            eval_config.id,
          )
        }
      } catch (e) {
        try {
          await client.DELETE(
            "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}",
            {
              params: {
                path: { project_id, task_id, eval_id: evaluator.id },
              },
            },
          )
        } catch {
          // Roll-back is best effort; surface the original failure.
        }
        throw e
      }

      posthog.capture("create_eval_with_judge", {
        spec_type: judge_only ? undefined : spec_type,
        judge_type: judge_type ?? undefined,
      })

      complete = true
      goto(`/specs/${project_id}/${task_id}/legacy/${evaluator.id}`)
    } catch (e) {
      error = createKilnError(e)
    } finally {
      submitting = false
      saving_spec = false
    }
  }

  async function grant_trust_and_save(): Promise<boolean> {
    try {
      await addCodeTrust(project_id)
    } catch (e) {
      error = createKilnError(e)
      return false
    }
    await handle_create_eval_with_judge()
    return true
  }

  // Non-LLM judge creation shows the Test Judge pane beside the form, so it
  // needs the wide layout.
  $: page_class = is_non_llm_judge ? "max-w-[1400px]" : "max-w-[900px]"

  $: breadcrumbs = [
    {
      label: "Evals",
      href: `/specs/${project_id}/${task_id}`,
    },
    {
      label: "Eval Types",
      href: `/specs/${project_id}/${task_id}/select_template`,
    },
  ]

  // Warn before unload when there are unsaved changes
  $: warn_before_unload = !complete && !loading
</script>

<div class={page_class}>
  <AppPage
    title="Create Eval"
    subtitle="Define a behaviour to enforce or avoid for your task, and automatically measure quality."
    sub_subtitle="Read the Docs"
    sub_subtitle_link="https://docs.kiln.tech/docs/evals-and-specs"
    {breadcrumbs}
  >
    {#if loading || saving_spec}
      <div class="w-full min-h-[50vh] flex justify-center items-center">
        <div class="loading loading-spinner loading-lg"></div>
      </div>
    {:else if loading_error}
      <div class="text-error text-sm">
        {loading_error.getMessage() || "An unknown error occurred"}
      </div>
    {:else if is_non_llm_judge && judge_type}
      <CreateSpecJudgeForm
        bind:this={judge_form}
        bind:name
        bind:priority
        {judge_type}
        {project_id}
        {task_id}
        bind:error
        bind:submitting
        {warn_before_unload}
        on:save={handle_create_eval_with_judge}
      />
    {:else}
      <CreateSpecForm
        bind:name
        bind:property_values
        {initial_property_values}
        bind:priority
        {field_configs}
        bind:error
        bind:submitting
        {warn_before_unload}
        on:create_spec={handle_create_spec}
      />
    {/if}
  </AppPage>
</div>

<TrustCodeDialog bind:this={trust_dialog} on_trust={grant_trust_and_save} />
