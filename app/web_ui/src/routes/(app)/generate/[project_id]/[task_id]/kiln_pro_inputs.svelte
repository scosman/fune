<script lang="ts">
  import { onDestroy, onMount } from "svelte"
  import { get } from "svelte/store"
  import { client } from "$lib/api_client"
  import posthog from "posthog-js"
  import type { components } from "$lib/api_schema"
  import { isKilnAgentRunConfig } from "$lib/types"
  import type {
    KilnAgentRunConfigProperties,
    RunConfigProperties,
  } from "$lib/types"
  import { createKilnError, KilnError } from "$lib/utils/error_handlers"
  import RunConfigComponent from "$lib/ui/run_config_component/run_config_component.svelte"
  import FormContainer from "$lib/utils/form_container.svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import InfoTooltip from "$lib/ui/info_tooltip.svelte"
  import Warning from "$lib/ui/warning.svelte"
  import TableActionMenu from "$lib/ui/table_action_menu.svelte"
  import { SynthDataGuidanceDataModel } from "./synth_data_guidance_datamodel"
  import {
    runInputsBatch,
    runOutputsBatch,
    type BatchRun,
    type InputsBatchStatus,
    type OutputsBatchStatus,
  } from "$lib/stores/kiln_pro_batch_store"

  export let plan: { prompts: string[]; summary: string }
  export let project_id: string
  export let task_id: string
  export let guidance_data: SynthDataGuidanceDataModel
  // The user already chose the model/data-guide in the modal on the plan page;
  // generation starts on mount with these.
  export let run_config_properties: KilnAgentRunConfigProperties
  export let data_guide: string | null
  // Tags every generated run with synthetic_session_<id>, matching legacy synth.
  export let session_id: string | null = null
  // Returns to the plan — the only way out once every sample is removed.
  export let on_back: () => void
  // Generated samples that aren't in the dataset yet. Leaving this screen throws
  // them away, so the parent reads this to guard navigation (bound).
  export let unsaved_samples: boolean = false

  type TaskRun = components["schemas"]["TaskRun-Output"]
  type Row = {
    prompt: string
    input: string | Record<string, unknown> | null
    output: string | null
    task_run: TaskRun | null
    input_error: string | null
    output_error: string | null
    // Set once persisted; also the dataset id the "Saved" status links to.
    saved_id: string | null
  }

  let rows: Row[] = plan.prompts.map((p) => ({
    prompt: p,
    input: null,
    output: null,
    task_run: null,
    input_error: null,
    output_error: null,
    saved_id: null,
  }))

  $: task = guidance_data.task

  // Covers the whole save: spinner while it runs, success state when done.
  let save_dialog: Dialog | null = null

  let inputs_status: "running" | "complete" | "error" | null = null
  let outputs_started = false
  let outputs_status: "running" | "complete" | "error" | null = null
  let batch_error: KilnError | null = null

  // Rows the current output batch is still working on. Without this, a row
  // whose output was removed after a batch finished would spin forever.
  let outputs_pending = new Set<number>()

  let inputs_run: BatchRun<InputsBatchStatus> | null = null
  let outputs_run: BatchRun<OutputsBatchStatus> | null = null
  let inputs_unsub: (() => void)[] = []
  let outputs_unsub: (() => void)[] = []

  let saving = false
  let save_progress = 0
  let save_target = 0
  let save_errors: KilnError[] = []
  let total_saved = 0
  let show_save_errors = false

  $: total = rows.length
  $: generated_inputs = rows.filter((r) => r.input !== null).length
  $: generated_outputs = rows.filter((r) => r.output !== null).length
  $: input_errors = rows.filter((r) => r.input_error !== null).length
  $: inputs_done = inputs_status !== null && inputs_status !== "running"
  $: outputs_done = outputs_status !== null && outputs_status !== "running"
  $: outputs_savable = rows.filter(
    (r) => r.task_run !== null && !r.saved_id,
  ).length
  // Rows that have an input but no output — the targets of Generate Outputs,
  // whether they've never been run, failed, or had their output removed.
  $: outputs_missing = rows.filter(
    (r) => r.input !== null && r.task_run === null,
  ).length
  $: generating = !inputs_done || (outputs_started && !outputs_done)
  // A full reset would orphan anything already written to the dataset.
  $: any_saved = rows.some((r) => r.saved_id)
  // A row with an input or an output that never made it to the dataset is work
  // the user would lose by navigating away. Rows whose input failed hold nothing
  // worth keeping, so they don't count.
  $: unsaved_samples = rows.some(
    (r) => !r.saved_id && (r.input !== null || r.task_run !== null),
  )

  // Before outputs run, warn about inputs that failed to generate — the retry
  // action lives on the warning itself. Once the user has clicked Generate
  // Outputs they've accepted those failures, and the useful warning becomes the
  // samples still missing an output.
  type WarningState = {
    message: string
    color: "warning" | "error" | "success"
    icon: "exclaim" | "check"
    // Rendered as a button beside the warning when present.
    retry?: boolean
  }
  // One warning at a time, like legacy: whatever the user still has to do beats
  // what they've already done. Saving 4 of 5 shouldn't stack a green "all saved"
  // under an amber "1 missing output" — the missing output is the live task.
  function build_warning(
    started_outputs: boolean,
    failed_inputs: number,
    ok_inputs: number,
    missing_outputs: number,
    savable: number,
    saved_any: boolean,
  ): WarningState | null {
    if (!started_outputs) {
      if (failed_inputs === 0) return null
      return {
        message: `${failed_inputs} ${failed_inputs === 1 ? "input" : "inputs"} failed to generate.`,
        color: ok_inputs > 0 ? "warning" : "error",
        icon: "exclaim",
        retry: true,
      }
    }
    if (missing_outputs > 0) {
      return {
        message: `${missing_outputs} ${missing_outputs === 1 ? "item is" : "items are"} missing outputs.`,
        color: savable > 0 ? "warning" : "error",
        icon: "exclaim",
      }
    }
    // Nothing left to generate and nothing left to save: the batch is done.
    if (saved_any && savable === 0) {
      return {
        message: "All items saved into your Dataset.",
        color: "success",
        icon: "check",
      }
    }
    return null
  }
  $: active_warning = build_warning(
    outputs_started,
    input_errors,
    generated_inputs,
    outputs_missing,
    outputs_savable,
    any_saved,
  )

  // Once there's something to save, regenerating is no longer the job.
  $: show_regenerate =
    inputs_done && !generating && !saving && !any_saved && outputs_savable === 0
  $: show_generate_outputs = inputs_done && !generating && outputs_missing > 0
  $: show_save = !generating && outputs_savable > 0
  // With every sample saved there's nothing left to act on, so the row has no
  // right-hand side. Let the warning take it rather than leaving a dead gap.
  $: has_actions =
    total > 0 && (show_regenerate || show_generate_outputs || show_save)

  let outputs_dialog: Dialog | null = null
  let outputs_run_config: RunConfigComponent | null = null
  let outputs_submitting = false

  // Re-running inputs re-asks for the model rather than silently reusing the
  // one picked on the plan page. `active_rcp` is whatever was chosen last.
  let inputs_dialog: Dialog | null = null
  const selected_template = guidance_data.selected_template
  let inputs_run_config: RunConfigComponent | null = null
  let inputs_submitting = false
  let inputs_action: "retry" | "regenerate" = "regenerate"
  let active_rcp: KilnAgentRunConfigProperties = run_config_properties

  function open_inputs_dialog(action: "retry" | "regenerate") {
    inputs_action = action
    batch_error = null
    inputs_dialog?.show()
  }

  function submit_inputs_config() {
    inputs_submitting = false
    const rcp =
      inputs_run_config?.run_options_as_run_config_properties() ?? null
    if (!rcp || !isKilnAgentRunConfig(rcp)) {
      batch_error = new KilnError(
        "Synthetic data generation requires a model with a kiln_agent run config.",
        null,
      )
      return
    }
    // Regenerating discards every input (and any outputs made from them), so
    // confirm — same native confirm the Refine Plan button uses. Retry is
    // additive, so it just runs.
    if (inputs_action === "regenerate") {
      const msg = `Are you sure you want to regenerate all ${total} inputs? This discards the current inputs${
        generated_outputs > 0 ? " and the outputs generated from them" : ""
      }. This cannot be undone.`
      if (!confirm(msg)) {
        return
      }
    }
    active_rcp = rcp
    inputs_dialog?.close()
    if (inputs_action === "retry") {
      retry_failed()
    } else {
      regenerate_inputs()
    }
  }

  let expanded: boolean[] = new Array(rows.length).fill(false)

  function toggle_expand(index: number) {
    expanded[index] = !expanded[index]
  }

  function format_expanded(data: string): string {
    // If JSON, pretty format it
    try {
      const json = JSON.parse(data)
      return JSON.stringify(json, null, 2)
    } catch (_) {
      // Not JSON
    }

    return data
  }

  function input_text(input: string | Record<string, unknown>): string {
    return typeof input === "string" ? input : JSON.stringify(input)
  }

  function delete_row(index: number) {
    rows = rows.filter((_, i) => i !== index)
    expanded = expanded.filter((_, i) => i !== index)
  }

  // Dataset splits (e.g. train 80% / test 20%) are assigned per sample by
  // drawing from the configured distribution — the same thing the legacy synth
  // flow does at save time. Without this, Kiln Pro samples land in the dataset
  // with no split tag at all.
  function random_split_tag(): string | undefined {
    const splits = get(guidance_data.splits)
    const tags = Object.keys(splits)
    if (tags.length === 0) return undefined

    const roll = Math.random()
    let cumulative = 0
    for (const [tag, probability] of Object.entries(splits)) {
      cumulative += probability
      if (roll <= cumulative) return tag
    }
    // Only reachable through floating-point drift when the splits sum to 1.
    return tags[tags.length - 1]
  }

  // Drops the output but keeps the input, so Generate Outputs can run it again.
  function remove_output(index: number) {
    rows[index].output = null
    rows[index].task_run = null
    rows[index].output_error = null
    rows = rows
  }

  onMount(() => {
    start_inputs()
  })

  onDestroy(() => {
    inputs_run?.cancel()
    outputs_run?.cancel()
    inputs_unsub.forEach((u) => u())
    outputs_unsub.forEach((u) => u())
  })

  function output_preview(run: TaskRun): string {
    const out = run.output?.output
    return typeof out === "string" ? out : JSON.stringify(out)
  }

  // Run config in the shape the legacy synth analytics use, so Kiln Pro events
  // line up with generate_synthetic_inputs / save_synthetic_data in PostHog.
  // Synth always uses kiln_agent configs; guard so an MCP config doesn't throw.
  function run_config_analytics(rcp: RunConfigProperties) {
    if (!isKilnAgentRunConfig(rcp)) return {}
    return {
      model_name: rcp.model_name,
      provider: rcp.model_provider_name,
      prompt_method: rcp.prompt_id,
      tools: rcp.tools_config?.tools ?? [],
      structured_output_mode: rcp.structured_output_mode,
    }
  }

  // Config used for the last output batch, so save analytics can report what
  // produced the runs being saved.
  let outputs_rcp: RunConfigProperties | null = null

  // Generates inputs for a subset of rows. The batch indexes its results by
  // position within the prompts it was given, so `row_indices` maps them back.
  function run_inputs(
    row_indices: number[],
    action: "initial" | "retry" | "regenerate",
  ) {
    if (row_indices.length === 0) return
    // Still reported in analytics, but no longer sent to the API.
    const gen_type = guidance_data.gen_type
    const prompts = row_indices.map((i) => rows[i].prompt)

    posthog.capture("kiln_pro_generate_inputs", {
      count: row_indices.length,
      action,
      gen_type,
      ...run_config_analytics(active_rcp),
    })

    inputs_run?.cancel()
    inputs_unsub.forEach((u) => u())
    inputs_unsub = []
    batch_error = null

    inputs_status = "running"
    // No gen_type: the batch plan's prompt already says what each input is for.
    // `data_guide` is null when the user has "Use Data Guide" off, and the
    // server treats null as "don't use one" (no fallback to the saved guide).
    inputs_run = runInputsBatch(project_id, task_id, {
      prompts,
      data_guide: data_guide || null,
      run_config_properties: active_rcp,
    })
    inputs_unsub.push(
      inputs_run.status.subscribe((s) => {
        if (!s) return
        for (const r of s.results) {
          const row = rows[row_indices[r.index]]
          if (!row) continue
          if (r.input !== null && r.input !== undefined) {
            row.input = r.input
          }
          if (r.error) row.input_error = r.error
        }
        rows = rows
        inputs_status = s.status
      }),
    )
    inputs_unsub.push(
      inputs_run.error.subscribe((e) => {
        if (e) {
          batch_error = e
          inputs_status = "error"
        }
      }),
    )
  }

  function start_inputs() {
    run_inputs(
      rows.map((_, i) => i),
      "initial",
    )
  }

  // Re-runs only the descriptions whose input generation failed. Non-destructive:
  // successful rows are untouched.
  function retry_failed() {
    const failed = rows
      .map((r, i) => (r.input_error ? i : -1))
      .filter((i) => i >= 0)
    for (const i of failed) {
      rows[i].input_error = null
      rows[i].input = null
    }
    rows = rows
    run_inputs(failed, "retry")
  }

  // Discards every input and generates again from the same descriptions. Only
  // offered before outputs exist, so there's no generated work to destroy.
  function regenerate_inputs() {
    rows = rows.map((r) => ({
      ...r,
      input: null,
      output: null,
      task_run: null,
      input_error: null,
      output_error: null,
    }))
    expanded = new Array(rows.length).fill(false)
    outputs_pending = new Set()
    run_inputs(
      rows.map((_, i) => i),
      "regenerate",
    )
  }

  function start_outputs() {
    outputs_submitting = false
    const rcp =
      outputs_run_config?.run_options_as_run_config_properties() ?? null
    if (!rcp) {
      batch_error = new KilnError("No run config selected.", null)
      return
    }
    outputs_dialog?.close()

    // Only rows still missing an output: never run, failed, or output removed.
    const items = rows
      .map((r, i) => ({ index: i, input: r.input, task_run: r.task_run }))
      .filter(
        (
          it,
        ): it is {
          index: number
          input: string | Record<string, unknown>
          task_run: null
        } => it.input !== null && it.task_run === null,
      )
      .map(({ index, input }) => ({ index, input }))
    if (items.length === 0) return

    outputs_rcp = rcp
    posthog.capture("kiln_pro_generate_outputs", {
      count: items.length,
      ...run_config_analytics(rcp),
    })

    outputs_run?.cancel()
    outputs_unsub.forEach((u) => u())
    outputs_unsub = []
    batch_error = null
    outputs_started = true
    outputs_status = "running"
    for (const it of items) {
      rows[it.index].output_error = null
    }
    outputs_pending = new Set(items.map((it) => it.index))
    rows = rows

    outputs_run = runOutputsBatch(project_id, task_id, {
      items,
      input_model_name: active_rcp.model_name,
      input_provider: active_rcp.model_provider_name,
      run_config_properties: rcp,
      session_id,
    })
    outputs_unsub.push(
      outputs_run.status.subscribe((s) => {
        if (!s) return
        for (const r of s.results) {
          if (r.task_run) {
            rows[r.index].task_run = r.task_run
            rows[r.index].output = output_preview(r.task_run)
          }
          if (r.error) rows[r.index].output_error = r.error
          if (r.task_run || r.error) outputs_pending.delete(r.index)
        }
        outputs_pending = outputs_pending
        rows = rows
        outputs_status = s.status
      }),
    )
    outputs_unsub.push(
      outputs_run.error.subscribe((e) => {
        if (e) {
          batch_error = e
          outputs_status = "error"
        }
      }),
    )
  }

  // Saving converts each generated run to a persisted TaskRun and can fail, so
  // mirror the legacy flow: show progress, collect per-item errors, and leave
  // failed rows unsaved so a re-click retries just those.
  async function save_all() {
    const to_save = rows.filter((r) => r.task_run && !r.saved_id)
    if (to_save.length === 0) return
    posthog.capture("kiln_pro_save_data", {
      count: to_save.length,
      ...(outputs_rcp ? run_config_analytics(outputs_rcp) : {}),
    })
    saving = true
    save_errors = []
    save_progress = 0
    // Per-save, not cumulative: the dialog reports what THIS save wrote. Rows
    // already persisted carry a saved_id and never re-post, so a later save of
    // one straggler must say "1", not the running total.
    total_saved = 0
    save_target = to_save.length
    // Open straight away — the dialog shows the spinner, then flips to the
    // success state when the save finishes.
    save_dialog?.show()
    for (const row of to_save) {
      try {
        const split_tag = random_split_tag()
        const to_post = split_tag
          ? {
              ...row.task_run,
              tags: [...(row.task_run?.tags ?? []), split_tag],
            }
          : row.task_run
        const { data, error } = await client.POST(
          "/api/projects/{project_id}/tasks/{task_id}/save_sample",
          {
            params: { path: { project_id, task_id } },
            // The batch status returns the Output variant; save_sample accepts
            // the Input variant. They round-trip the same object on the server.
            body: to_post as unknown as components["schemas"]["TaskRun-Input"],
          },
        )
        if (error) throw error
        if (!data?.id) throw new KilnError("Save failed: no id returned.", null)
        row.saved_id = data.id
        total_saved++
      } catch (e) {
        save_errors = [...save_errors, createKilnError(e)]
      }
      save_progress++
      rows = rows
    }
    saving = false
    // The dialog stays open either way and reports what happened, matching
    // legacy: partial failures are a section of the same result dialog, not a
    // separate surface. Failed rows keep no saved_id, so Save All retries them.
  }
</script>

<div class="flex flex-col gap-4 mt-12">
  <div class="flex flex-row items-center justify-between gap-4">
    <div
      class="min-w-0 flex flex-row items-center gap-3 {has_actions
        ? ''
        : 'ml-auto'}"
    >
      {#if !generating && active_warning}
        <Warning
          warning_message={active_warning.message}
          warning_color={active_warning.color}
          warning_icon={active_warning.icon}
          inline={true}
        />
        {#if active_warning.retry && !saving}
          <button
            class="btn btn-link btn-sm shrink-0 p-0"
            on:click={() => open_inputs_dialog("retry")}
          >
            Retry
          </button>
        {/if}
      {/if}
    </div>
    <!-- With no rows left, the empty state owns the only action: Back to Plan. -->
    {#if has_actions}
      <div class="flex flex-row gap-2 shrink-0">
        {#if show_regenerate}
          <button
            class="btn btn-md"
            on:click={() => open_inputs_dialog("regenerate")}
          >
            Regenerate Inputs
          </button>
        {/if}
        {#if show_generate_outputs}
          <button
            class="btn btn-md {outputs_savable > 0 ? '' : 'btn-primary'}"
            disabled={saving}
            on:click={() => outputs_dialog?.show()}
          >
            Generate Outputs ({outputs_missing})
          </button>
        {/if}
        {#if show_save}
          <button
            class="btn btn-md btn-primary"
            disabled={saving}
            on:click={save_all}
          >
            {saving ? "Saving…" : `Save All (${outputs_savable})`}
          </button>
        {/if}
      </div>
    {/if}
  </div>

  {#if batch_error}
    <div class="text-error text-sm">{batch_error.getMessage()}</div>
  {/if}

  {#if generating}
    {@const progress = inputs_done ? generated_outputs : generated_inputs}
    <div class="flex flex-row items-center justify-center gap-3 py-2">
      <span class="loading loading-spinner loading-sm text-primary"></span>
      <span class="text-sm font-medium">
        {inputs_done ? "Generating Outputs" : "Generating Inputs"}
      </span>
      <span class="text-gray-400">·</span>
      <span class="text-sm font-light text-gray-500">
        {progress} of {total}
      </span>
    </div>
  {/if}

  {#if total === 0}
    <div
      class="rounded-lg border px-4 py-12 flex flex-col items-center gap-3 text-center"
    >
      <div class="text-sm text-gray-500">
        You've removed every sample from this batch.
      </div>
      <button class="btn btn-md btn-primary" on:click={on_back}>
        Back to Plan
      </button>
    </div>
  {:else}
    <div class="rounded-lg border">
      <table class="table table-fixed">
        <thead>
          <tr>
            <!-- Three equal content columns; the action menu takes the last 40px. -->
            <th style="width: calc((100% - 40px) / 3)">
              Guidance <InfoTooltip
                tooltip_text="The prompt from your batch plan that guided this sample."
                position="bottom"
              />
            </th>
            <th style="width: calc((100% - 40px) / 3)">
              Input <InfoTooltip
                tooltip_text="The input to the task, generated from your batch plan."
                position="bottom"
              />
            </th>
            <th style="width: calc((100% - 40px) / 3)">
              Output <InfoTooltip
                tooltip_text="The output from running the task on the input."
                position="bottom"
              />
            </th>
            <th style="width: 40px"></th>
          </tr>
        </thead>
        <tbody>
          {#each rows as row, i}
            <tr on:click={() => toggle_expand(i)} class="cursor-pointer">
              <td class="py-2 align-top">
                {#if expanded[i]}
                  <pre
                    class="whitespace-pre-wrap break-words">{row.prompt}</pre>
                {:else}
                  <div class="truncate w-0 min-w-full text-gray-500">
                    {row.prompt}
                  </div>
                {/if}
              </td>
              <td class="py-2 align-top">
                {#if row.input !== null}
                  {#if expanded[i]}
                    <pre
                      class="whitespace-pre-wrap break-words">{format_expanded(
                        input_text(row.input),
                      )}</pre>
                  {:else}
                    <div class="truncate w-0 min-w-full">
                      {input_text(row.input)}
                    </div>
                  {/if}
                {:else if row.input_error}
                  <span class="text-error text-sm">Failed</span>
                {:else}
                  <span class="flex items-center gap-2 text-gray-500">
                    <span class="loading loading-spinner loading-xs"></span>
                    Generating…
                  </span>
                {/if}
              </td>
              <td class="py-2 align-top">
                {#if row.output !== null}
                  {#if expanded[i]}
                    <pre
                      class="whitespace-pre-wrap break-words">{format_expanded(
                        row.output,
                      )}</pre>
                  {:else}
                    <div class="truncate w-0 min-w-full">{row.output}</div>
                  {/if}
                {:else if row.output_error}
                  <span class="text-error text-sm">Failed</span>
                {:else if outputs_pending.has(i)}
                  <span class="flex items-center gap-2 text-gray-500">
                    <span class="loading loading-spinner loading-xs"></span>
                    Generating…
                  </span>
                {:else}
                  <span class="text-gray-400">Not Generated</span>
                {/if}
              </td>
              <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
              <td class="p-0 align-top" on:click|stopPropagation>
                <TableActionMenu
                  items={[
                    {
                      label: "Remove Output",
                      onclick: () => remove_output(i),
                      hidden: generating || !!row.saved_id || !row.task_run,
                    },
                    {
                      label: "Remove Sample",
                      onclick: () => delete_row(i),
                      hidden: generating || !!row.saved_id,
                    },
                  ]}
                />
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>

<!-- Save: spinner while it runs, then the result — what saved, where it went,
     and anything that failed. Mirrors the legacy synth save dialog. -->
<Dialog bind:this={save_dialog} title="" center_content={true}>
  {#if saving}
    <div class="text-center flex flex-col items-center justify-center py-6">
      <span class="loading loading-spinner loading-lg text-primary mb-3"></span>
      <div class="text-lg font-medium">Saving</div>
      <div class="font-light text-sm text-gray-500 mt-1">
        {save_progress} of {save_target}
      </div>
    </div>
  {:else}
    <div
      class="text-center flex flex-col items-center justify-center pt-2 pb-2"
    >
      {#if total_saved > 0}
        <svg
          fill="currentColor"
          class="size-12 text-success mb-3"
          viewBox="0 0 56 56"
          xmlns="http://www.w3.org/2000/svg"
          ><path
            d="M 27.9999 51.9063 C 41.0546 51.9063 51.9063 41.0781 51.9063 28 C 51.9063 14.9453 41.0312 4.0937 27.9765 4.0937 C 14.8983 4.0937 4.0937 14.9453 4.0937 28 C 4.0937 41.0781 14.9218 51.9063 27.9999 51.9063 Z M 27.9999 47.9219 C 16.9374 47.9219 8.1014 39.0625 8.1014 28 C 8.1014 16.9609 16.9140 8.0781 27.9765 8.0781 C 39.0155 8.0781 47.8983 16.9609 47.9219 28 C 47.9454 39.0625 39.0390 47.9219 27.9999 47.9219 Z M 25.0468 39.7188 C 25.8202 39.7188 26.4530 39.3437 26.9452 38.6172 L 38.5234 20.4063 C 38.8046 19.9375 39.0858 19.3984 39.0858 18.8828 C 39.0858 17.8047 38.1483 17.1484 37.1640 17.1484 C 36.5312 17.1484 35.9452 17.5 35.5234 18.2031 L 24.9296 35.1484 L 19.4921 28.1172 C 18.9765 27.4141 18.4140 27.1563 17.7812 27.1563 C 16.7499 27.1563 15.9296 28 15.9296 29.0547 C 15.9296 29.5703 16.1405 30.0625 16.4687 30.5078 L 23.0312 38.6172 C 23.6640 39.3906 24.2733 39.7188 25.0468 39.7188 Z"
          /></svg
        >
      {/if}
      <div class="font-medium">
        Saved {total_saved} new {total_saved === 1 ? "item" : "items"}.
      </div>
      <div class="font-light text-sm">
        These are now available in the <a
          href={`/dataset/${project_id}/${task_id}`}
          class="link">dataset tab</a
        >.
      </div>
      {#if session_id}
        <div class="font-light text-xs mt-4 text-gray-500">
          All items are tagged with &quot;synthetic_session_{session_id}&quot;
        </div>
      {/if}
      {#if save_errors.length > 0}
        <div class="text-error font-light text-sm mt-4">
          {save_errors.length}
          {save_errors.length === 1 ? "item" : "items"} failed to save. Running again
          may resolve transient issues.
          <button
            class="link"
            on:click={() => (show_save_errors = !show_save_errors)}
          >
            {show_save_errors ? "Hide errors" : "Show errors"}
          </button>
          {#if show_save_errors}
            <div class="flex flex-col gap-1 mt-2 text-xs text-left">
              {#each save_errors as e}
                <div>{e.getMessage()}</div>
              {/each}
            </div>
          {/if}
        </div>
      {/if}
    </div>
  {/if}
</Dialog>

<!-- One dialog, reused for both retry and regenerate -->
<Dialog bind:this={inputs_dialog} title="Generation Settings">
  <FormContainer
    submit_label={inputs_action === "retry"
      ? `Retry (${input_errors})`
      : `Regenerate Inputs (${total})`}
    bind:submitting={inputs_submitting}
    on:submit={submit_inputs_config}
    keyboard_submit={false}
  >
    {#if task}
      <RunConfigComponent
        bind:this={inputs_run_config}
        {project_id}
        current_task={task}
        requires_structured_output={true}
        hide_prompt_selector={true}
        show_tools_selector_in_advanced={true}
        show_name_field={false}
        model_dropdown_settings={{
          requires_data_gen: true,
          requires_uncensored_data_gen:
            guidance_data.suggest_uncensored($selected_template),
          suggested_mode: guidance_data.suggest_uncensored($selected_template)
            ? "uncensored_data_gen"
            : "data_gen",
        }}
      />
    {/if}
  </FormContainer>
</Dialog>

<!-- Generate Outputs modal -->
<Dialog bind:this={outputs_dialog} title="Generation Settings">
  <FormContainer
    submit_label="Generate Outputs"
    bind:submitting={outputs_submitting}
    on:submit={start_outputs}
    keyboard_submit={false}
  >
    <div>
      <div class="font-medium text-sm">Status</div>
      <div class="font-light">
        {outputs_missing}
        {outputs_missing === 1 ? "item" : "items"} pending
        {#if generated_outputs > 0}
          / {generated_outputs} already generated
        {/if}
      </div>
    </div>
    {#if task}
      <RunConfigComponent
        bind:this={outputs_run_config}
        {project_id}
        current_task={task}
        requires_structured_output={!!task.output_json_schema}
        show_name_field={false}
        model_dropdown_settings={{
          requires_structured_output: !!task.output_json_schema,
        }}
      />
    {/if}
  </FormContainer>
</Dialog>
