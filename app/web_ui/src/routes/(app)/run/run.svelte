<script lang="ts">
  import type { TaskRun, Task } from "$lib/types"
  import { client } from "$lib/api_client"
  import Output from "$lib/ui/output.svelte"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { model_info, model_name as model_name_from_id } from "$lib/stores"
  import TraceComponent from "$lib/ui/trace/trace.svelte"
  import Warning from "$lib/ui/warning.svelte"
  import OutputRepairEditForm from "./output_repair_edit_form.svelte"
  import AvailableModelsDropdown from "$lib/ui/run_config_component/available_models_dropdown.svelte"
  import RunSidebar from "$lib/ui/run_sidebar.svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import FormContainer from "$lib/utils/form_container.svelte"
  import FormElement from "$lib/utils/form_element.svelte"
  import InfoTooltip from "$lib/ui/info_tooltip.svelte"
  import type { components } from "$lib/api_schema"

  const REPAIR_ENABLED_FOR_SOURCES: Array<
    components["schemas"]["DataSourceType"]
  > = ["human", "synthetic"]

  export let project_id: string
  export let task: Task
  export let initial_run: TaskRun
  let updated_run: TaskRun | null = null
  $: run = updated_run || initial_run
  export let model_name: string | null = null
  export let provider: string | null = null
  export let run_complete: boolean = false
  export let focus_repair_on_appear: boolean = false

  // note: this run is NOT the main run, but a repair run TaskRun
  let repair_run: TaskRun | null = null
  let repair_instructions: string | null = null
  // Seed repair_instructions from the persisted run so tooltips/UI on historical repairs show the original feedback.
  // Track the last seeded run id so switching to a different run reseeds rather than leaking prior text.
  let seeded_for_run_id: string | null = null
  $: if (run?.id && run.id !== seeded_for_run_id) {
    repair_instructions = run.repair_instructions ?? null
    seeded_for_run_id = run.id
  }

  // Mirror the overall rating from the persisted run so repair / run-complete
  // logic can react without owning the rating UI itself (the sidebar component
  // owns the actual rating + tags + feedback UI now).
  $: overall_rating = (run?.output?.rating?.value ?? null) as number | null
  // True if this "Run" has everything we want: a rating and a repaired output (or 5-star rating and no repair is needed)
  $: run_complete = overall_rating === 5 || !!run?.repaired_output?.output

  // Repair is available if the run has an overall rating but it's not 5 stars, and it doesn't yet have a repaired output
  $: should_offer_repair =
    run &&
    overall_rating !== null &&
    overall_rating !== 5 &&
    !run?.repaired_output?.output && // model already repaired
    !repair_run // repair generated, should show repair evaluation instead
  $: repair_review_available = !!repair_run && !run?.repaired_output?.output
  $: repair_complete = !!run?.repaired_output?.output
  $: repair_enabled_for_source = REPAIR_ENABLED_FOR_SOURCES.some(
    (s) => s === run?.output?.source?.type,
  )

  $: repair_source =
    run?.repaired_output?.source?.type === "human"
      ? {
          type: "user",
          name: run.repaired_output.source.properties?.created_by ?? "unknown",
        }
      : run?.repaired_output?.source?.type === "synthetic"
        ? { type: "synthetic" }
        : null

  let show_raw_data = false

  async function patch_run(
    patch_body: Record<string, unknown>,
  ): Promise<TaskRun> {
    const {
      data, // only present if 2XX response
      error: fetch_error, // only present if 4XX or 5XX response
    } = await client.PATCH(
      "/api/projects/{project_id}/tasks/{task_id}/runs/{run_id}",
      {
        params: {
          path: {
            project_id: project_id,
            task_id: task.id || "",
            run_id: run?.id || "",
          },
        },
        body: patch_body,
      },
    )
    if (fetch_error) {
      throw fetch_error
    }
    return data
  }

  // Optional per-task override for which model generates the repair. Persisted in
  // localStorage so the choice survives reloads. Useful when the original run's
  // model can't be rehydrated (e.g. pre-fix legacy openai_compatible runs whose
  // persisted model_name lost its "{provider}::" prefix on disk).
  type RepairModelOverride = {
    model_name: string
    provider: string
  }
  let repair_model_override: RepairModelOverride | null = null
  $: repair_model_storage_key = task?.id
    ? `kiln_repair_model_override:${project_id}:${task.id}`
    : null
  $: if (repair_model_storage_key) {
    repair_model_override = load_repair_model_override(repair_model_storage_key)
  } else {
    repair_model_override = null
  }

  function load_repair_model_override(key: string): RepairModelOverride | null {
    try {
      const raw = localStorage.getItem(key)
      if (!raw) return null
      const parsed = JSON.parse(raw)
      if (
        parsed &&
        typeof parsed.model_name === "string" &&
        typeof parsed.provider === "string"
      ) {
        return { model_name: parsed.model_name, provider: parsed.provider }
      }
    } catch {
      // ignore corrupted entries
    }
    return null
  }

  // The original run's model (what produced the saved output). Treated as the
  // canonical "default" for repair: the dialog preselects it and the Reset
  // button snaps the selection back to it.
  $: original_run_model_name =
    typeof run?.output?.source?.properties?.model_name === "string"
      ? (run.output.source.properties.model_name as string)
      : null
  $: original_run_provider =
    typeof run?.output?.source?.properties?.model_provider === "string"
      ? (run.output.source.properties.model_provider as string)
      : null

  // The model that will actually be used to generate the repair: override
  // wins, otherwise we fall back to the original run's model.
  $: effective_repair_model_name =
    repair_model_override?.model_name ?? original_run_model_name
  $: effective_repair_provider =
    repair_model_override?.provider ?? original_run_provider
  $: effective_repair_model_display = effective_repair_model_name
    ? model_name_from_id(effective_repair_model_name, $model_info)
    : null

  let repair_submitting = false
  let repair_error: KilnError | null = null
  async function attempt_repair() {
    try {
      repair_submitting = true
      const trimmed_instructions = repair_instructions?.trim()
      if (!trimmed_instructions) {
        throw new KilnError("Repair instructions are required", null)
      }
      if (!task.id || !run?.id) {
        throw new KilnError(
          "This task run isn't saved. Enable Auto-save. You can't repair unsaved runs.",
          null,
        )
      }
      // Only send the override on the wire — when no override is set, the
      // server reads the original run's persisted model from source_properties.
      const {
        data: repair_data, // only present if 2XX response
        error: fetch_error, // only present if 4XX or 5XX response
      } = await client.POST(
        "/api/projects/{project_id}/tasks/{task_id}/runs/{run_id}/generate_repair",
        {
          params: {
            path: {
              project_id: project_id,
              task_id: task.id,
              run_id: run?.id,
            },
          },
          body: repair_model_override
            ? {
                evaluator_feedback: trimmed_instructions,
                model_name: repair_model_override.model_name,
                provider: repair_model_override.provider,
              }
            : {
                evaluator_feedback: trimmed_instructions,
              },
        },
      )
      if (fetch_error) {
        throw fetch_error
      }
      repair_run = repair_data
      repair_error = null
    } catch (err) {
      repair_error = createKilnError(err)
    } finally {
      repair_submitting = false
    }
  }

  let accept_repair_error: KilnError | null = null
  let accept_repair_submitting = false
  async function accept_repair() {
    try {
      accept_repair_error = null
      accept_repair_submitting = true
      if (!repair_run) {
        throw new KilnError("No repair to accept", null)
      }
      if (!task.id || !run?.id) {
        throw new KilnError(
          "This task run isn't saved. Enable Auto-save. You can't accept repairs for unsaved runs.",
          null,
        )
      }
      const {
        data, // only present if 2XX response
        error: fetch_error, // only present if 4XX or 5XX response
      } = await client.POST(
        "/api/projects/{project_id}/tasks/{task_id}/runs/{run_id}/save_repair",
        {
          params: {
            path: {
              project_id: project_id,
              task_id: task.id,
              run_id: run?.id,
            },
          },
          body: {
            repair_run: repair_run,
            evaluator_feedback: repair_instructions || "",
          },
        },
      )
      if (fetch_error) {
        throw fetch_error
      }
      updated_run = data
      repair_run = null
    } catch (err) {
      accept_repair_error = createKilnError(err)
    } finally {
      accept_repair_submitting = false
    }
  }

  let delete_repair_error: KilnError | null = null
  let delete_repair_submitting = false
  async function delete_repair() {
    if (
      !confirm(
        "Are you sure you want to delete this repair?\n\nThis action cannot be undone.",
      )
    ) {
      return
    }
    try {
      delete_repair_error = null
      delete_repair_submitting = true
      let original_repair_instructions = run?.repair_instructions
      let patch_body = {
        repair_instructions: null,
        repaired_output: null,
      }
      updated_run = await patch_run(patch_body)
      repair_run = null

      // Pull in the instructions from the original repair, so they can edit them if wanted
      if (!repair_instructions && original_repair_instructions) {
        repair_instructions = original_repair_instructions
      }
    } catch (err) {
      delete_repair_error = createKilnError(err)
    } finally {
      delete_repair_submitting = false
    }
  }

  function retry_repair() {
    repair_run = null
    accept_repair_error = null
  }

  // Repair-model override dialog state
  let repair_model_dialog: Dialog | null = null
  let dialog_combined_model: string | null = null
  let dialog_model_name: string | null = null
  let dialog_provider_name: string | null = null

  function open_repair_model_dialog() {
    // Preselect the currently effective repair model (override > original run's model).
    dialog_combined_model =
      effective_repair_model_name && effective_repair_provider
        ? `${effective_repair_provider}/${effective_repair_model_name}`
        : null
    repair_model_dialog?.show()
  }

  function reset_repair_model_dialog() {
    // Reset = no override. Each run will fall back to whichever model originally
    // produced it (which can vary across runs in the same task). Clear the dropdown
    // visually; the actual override is cleared on Save (or stays as-is on Cancel).
    dialog_combined_model = null
  }

  function save_repair_model_override(): boolean {
    // Persist whatever is currently selected at the time Save is clicked.
    // Empty selection = clear the override; per-run defaults take over.
    if (!dialog_model_name || !dialog_provider_name) {
      clear_repair_model_override()
      return true
    }
    const next: RepairModelOverride = {
      model_name: dialog_model_name,
      provider: dialog_provider_name,
    }
    repair_model_override = next
    if (repair_model_storage_key) {
      try {
        localStorage.setItem(repair_model_storage_key, JSON.stringify(next))
      } catch {
        // ignore quota/availability errors; in-memory state still applies
      }
    }
    return true
  }

  function clear_repair_model_override() {
    repair_model_override = null
    if (repair_model_storage_key) {
      try {
        localStorage.removeItem(repair_model_storage_key)
      } catch {
        // ignore
      }
    }
  }

  let repair_edit_mode = false
  function show_repair_edit() {
    repair_edit_mode = true
  }

  function handle_manual_edit_cancel() {
    repair_edit_mode = false
  }

  function handle_manual_edit_submit(repair_run_edited: TaskRun) {
    repair_edit_mode = false
    updated_run = repair_run_edited
    repair_run = null
  }

  function toggle_raw_data() {
    show_raw_data = !show_raw_data
    if (show_raw_data) {
      // Scroll to the raw data section when it's shown
      setTimeout(() => {
        const rawDataElement = document.getElementById("raw_data")
        if (rawDataElement) {
          rawDataElement.scrollIntoView({ behavior: "smooth", block: "start" })
        }
      }, 100)
    }
  }

  function get_intermediate_output_title(name: string): string {
    return (
      {
        reasoning: "Model Reasoning Output",
        chain_of_thought: "Chain of Thought Output",
      }[name] || name
    )
  }
</script>

<div>
  <div class="flex flex-col xl:flex-row gap-8 xl:gap-16">
    <!-- px/-mx pair: overflow-hidden contains wide trace content, but would
         otherwise clip the 4px focus ring (outline-offset: 2px) of inputs like
         the repair textarea. The padding gives the ring room; the matching
         negative margin keeps content aligned with the rest of the page. -->
    <div class="grow min-w-0 overflow-hidden px-1.5 -mx-1.5">
      <div class="text-xl font-bold mb-1">Output</div>
      {#if task.output_json_schema}
        <div class="text-xs font-medium text-gray-500 flex flex-row mb-2">
          <svg
            fill="currentColor"
            class="w-4 h-4 mr-[2px]"
            viewBox="0 0 56 56"
            xmlns="http://www.w3.org/2000/svg"
            ><path
              d="M 27.9999 51.9063 C 41.0546 51.9063 51.9063 41.0781 51.9063 28 C 51.9063 14.9453 41.0312 4.0937 27.9765 4.0937 C 14.8983 4.0937 4.0937 14.9453 4.0937 28 C 4.0937 41.0781 14.9218 51.9063 27.9999 51.9063 Z M 24.7655 40.0234 C 23.9687 40.0234 23.3593 39.6719 22.6796 38.8750 L 15.9296 30.5312 C 15.5780 30.0859 15.3671 29.5234 15.3671 29.0078 C 15.3671 27.9063 16.2343 27.0625 17.2655 27.0625 C 17.9452 27.0625 18.5077 27.3203 19.0702 28.0469 L 24.6718 35.2890 L 35.5702 17.8281 C 36.0155 17.1016 36.6249 16.75 37.2343 16.75 C 38.2655 16.75 39.2733 17.4297 39.2733 18.5547 C 39.2733 19.0703 38.9687 19.6328 38.6640 20.1016 L 26.7577 38.8750 C 26.2421 39.6484 25.5858 40.0234 24.7655 40.0234 Z"
            /></svg
          >
          Structure Valid
        </div>
      {/if}
      {#if run.trace}
        <!-- Render the output, but leave the COT and other intermediate output rendering to the trace -->
        <Output raw_output={run.output.output} />
        <div>
          <div class="font-bold mt-6 mb-2">Message Trace</div>
          <TraceComponent trace={run.trace} {project_id} />
        </div>
      {:else}
        <Output raw_output={run.output.output} />
        {#if run.intermediate_outputs}
          {#each Object.entries(run.intermediate_outputs) as [name, intermediate_output]}
            <div
              class="text-xs font-bold text-gray-500 mt-4 mb-1 flex flex-row items-center gap-1"
            >
              {get_intermediate_output_title(name)}
              <InfoTooltip
                tooltip_text={`This is intermediate output from the model, and not considered part of the final answer. This thinking helped formulate the final answer above. This is known as 'chain of thought', 'thinking output', or 'inference time compute'.`}
              />
            </div>
            <Output raw_output={intermediate_output} />
          {/each}
        {/if}
      {/if}
      <div>
        <div class="mt-2">
          <button class="text-xs link" on:click={toggle_raw_data}
            >{show_raw_data ? "Hide" : "Show"} Raw Data</button
          >
        </div>

        <div class={show_raw_data ? "" : "hidden"}>
          <h1 class="text-xl font-bold mt-2 mb-2" id="raw_data">Raw Data</h1>
          <div class="text-sm">
            <Output raw_output={JSON.stringify(run, null, 2)} />
          </div>
        </div>
      </div>

      {#if !repair_enabled_for_source && (should_offer_repair || repair_review_available || repair_complete)}
        <div class="grow mt-10">
          <Warning
            warning_message="Repair is not available for runs from {run.output
              .source?.type || 'unknown'} sources."
            warning_color="warning"
            inline={true}
          />
        </div>
      {/if}

      {#if repair_enabled_for_source && (should_offer_repair || repair_review_available || repair_complete)}
        <div class="grow mt-10">
          <div class="flex items-baseline justify-between gap-4 flex-wrap mb-2">
            <div class="text-xl font-bold">Repair Output</div>
            {#if should_offer_repair}
              <div class="text-xs text-gray-500">
                {#if effective_repair_model_display}
                  Repairing with <span class="font-medium text-gray-700"
                    >{effective_repair_model_display}</span
                  >
                  ·
                {/if}
                <button
                  type="button"
                  class="link"
                  on:click={open_repair_model_dialog}
                  >Select different model</button
                >
              </div>
            {/if}
          </div>
          {#if should_offer_repair}
            <p class="text-sm text-gray-500 mb-4">
              Since the output isn't 5-star, provide instructions for the model
              on how to fix it.
            </p>
            <FormContainer
              submit_label="Attempt Repair"
              on:submit={attempt_repair}
              bind:submitting={repair_submitting}
              bind:error={repair_error}
              focus_on_mount={focus_repair_on_appear}
            >
              <FormElement
                id="repair_instructions"
                label="Repair Instructions"
                inputType="textarea"
                bind:value={repair_instructions}
              />
            </FormContainer>
          {:else if repair_edit_mode && repair_run}
            <p class="text-sm text-gray-500 mb-4">
              Manually improve or correct the response.
            </p>
            <OutputRepairEditForm
              {task}
              {run}
              {repair_run}
              {project_id}
              repair_instructions={repair_instructions || ""}
              on_submit={handle_manual_edit_submit}
              on_cancel={handle_manual_edit_cancel}
            />
          {:else if repair_review_available}
            <p class="text-sm text-gray-500 mb-4">
              The model has attempted to fix the output given <span
                class="tooltip link"
                data-tip="The instructions you provided to the model: {repair_instructions ||
                  'No instruction provided'}">your instructions</span
              >. Review the result.
            </p>
            <Output raw_output={repair_run?.output.output || ""} />
          {:else if repair_complete}
            {#if repair_source?.type === "user"}
              <p class="text-sm text-gray-500 mb-4">
                This repaired output was provided by {repair_source.name}.
              </p>
            {:else}
              <p class="text-sm text-gray-500 mb-4">
                The model has fixed the output given <span
                  class="tooltip link"
                  data-tip="The instructions you provided to the model: {repair_instructions ||
                    'No instruction provided'}">your instructions</span
                >.
              </p>
            {/if}
            <Output raw_output={run?.repaired_output?.output || ""} />
            <div class="mt-2 text-xs text-gray-500 text-right">
              {#if delete_repair_submitting}
                <span class="loading loading-spinner loading-sm"></span>
              {:else if delete_repair_error}
                <p class="text-error">
                  Error Deleting Repair:
                  {delete_repair_error.getMessage()}
                </p>
              {:else}
                <button class="link" on:click={delete_repair}
                  >Delete Repair</button
                >
              {/if}
            </div>
          {/if}
        </div>
        {#if repair_review_available && !repair_edit_mode}
          <div class="mt-4">
            <div class="flex flex-row gap-4 justify-between">
              <button class="btn" on:click={show_repair_edit}>Edit</button>
              <div class="flex flex-row gap-4">
                <button class="btn" on:click={retry_repair}>Retry Repair</button
                >
                <button
                  class="btn btn-primary"
                  on:click={accept_repair}
                  disabled={accept_repair_submitting}
                >
                  {#if accept_repair_submitting}
                    <span class="loading loading-spinner loading-sm"></span>
                  {:else}
                    Accept Repair (5 Stars)
                  {/if}
                </button>
              </div>
            </div>

            {#if accept_repair_error}
              <p class="mt-2 text-error font-medium text-sm">
                Error Accepting Repair<br />
                <span class="text-error text-xs font-normal">
                  {accept_repair_error.getMessage()}</span
                >
              </p>
            {/if}
          </div>
        {/if}
      {/if}
    </div>

    <div class="w-72 2xl:w-96 flex-none">
      <RunSidebar
        {project_id}
        {task}
        {run}
        on_run_updated={(updated) => (updated_run = updated)}
      />
    </div>
  </div>
</div>

<Dialog
  bind:this={repair_model_dialog}
  title="Repair Model"
  sub_subtitle="Override the model used to generate repairs for this task. Reset to use each run's original model."
  action_buttons={[
    { label: "Cancel", isCancel: true },
    {
      label: "Save",
      isPrimary: true,
      action: save_repair_model_override,
    },
  ]}
>
  <div class="pt-2">
    <AvailableModelsDropdown
      label="Repair Model"
      description="Leave empty to use each run's original model. Pick a specific model to override."
      bind:model={dialog_combined_model}
      bind:model_name={dialog_model_name}
      bind:provider_name={dialog_provider_name}
      optional
      hide_optional_badge
      empty_label="Use the same model as the original run"
      inline_action={dialog_combined_model
        ? {
            handler: reset_repair_model_dialog,
            label: "Reset",
          }
        : null}
    />
  </div>
</Dialog>
