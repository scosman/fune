<script lang="ts">
  import Rating from "../../routes/(app)/run/rating.svelte"
  import type {
    TaskRun,
    Task,
    Feedback,
    RequirementRating,
    TaskRequirement,
    Trace,
  } from "$lib/types"
  import { client } from "$lib/api_client"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { formatDate, formatLatency } from "$lib/utils/formatters"
  import { bounceOut } from "svelte/easing"
  import { fly } from "svelte/transition"
  import { onMount } from "svelte"
  import TagPicker from "$lib/ui/tag_picker.svelte"
  import InfoTooltip from "$lib/ui/info_tooltip.svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import FormContainer from "$lib/utils/form_container.svelte"
  import FormElement from "$lib/utils/form_element.svelte"
  import { rating_options_for_sample, get_task_composite_id } from "$lib/stores"
  import {
    rating_options_by_task_composite_id,
    load_rating_options,
  } from "$lib/stores/rating_options_store"
  import posthog from "posthog-js"
  import PropertyList from "$lib/ui/property_list.svelte"
  import TableActionMenu from "$lib/ui/table_action_menu.svelte"

  export let project_id: string
  export let task: Task
  export let run: TaskRun
  export let on_run_updated: ((updated: TaskRun) => void) | null = null

  // URL-scoped rating options keyed by project/task composite id.
  $: task_rating_options =
    project_id && task.id
      ? $rating_options_by_task_composite_id[
          get_task_composite_id(project_id, task.id)
        ] ?? null
      : null
  $: if (project_id && task.id) {
    load_rating_options(project_id, task.id).catch((e: unknown) => {
      console.warn("Failed to load rating options", e)
    })
  }

  // Dynamic rating requirements based on tags
  $: rating_requirements = rating_options_for_sample(
    task_rating_options,
    run?.tags || [],
  )

  let save_rating_error: KilnError | null = null

  type RatingValue = number | null
  let overall_rating: RatingValue = null
  let requirement_ratings: RatingValue[] = []

  $: rate_focus = run && overall_rating === null

  // Use for some animations on first mount
  let mounted = false
  onMount(() => {
    setTimeout(() => {
      mounted = true
    }, 50) // Short delay to ensure component is fully mounted
  })

  function load_server_ratings(
    new_run: TaskRun | null,
    reqs: TaskRequirement[],
  ) {
    // Skip if run or requirements are missing
    if (!reqs || !new_run) {
      return
    }
    // Fill ratings with nulls
    requirement_ratings = Array(reqs.length).fill(null)
    if (!new_run) {
      return
    }
    overall_rating = (new_run.output.rating?.value || null) as RatingValue
    Object.entries(new_run.output.rating?.requirement_ratings || {}).forEach(
      ([req_id, rating]) => {
        let index = reqs.findIndex((req) => req.id === req_id)
        if (index !== -1) {
          const task_req = reqs[index]
          // Only load if the task requirement type matches the rating type. Technically users can switch the rating type, and we don't want to assume a 1 star rating is a "pass"
          if (task_req.type === rating.type) {
            requirement_ratings[index] = rating.value
          }
        }
      },
    )
  }

  // Seed the ratings from the server run on navigation (new run.id), and once
  // more when the task's rating options finish loading so requirement_ratings
  // can be populated - we should not re-load on every Run mutation because if a
  // PATCH is slow (as is the case for git-sync projects), it would reset the
  // user's in progress input
  let seeded_ratings_for_run_id: string | null = null
  let seeded_ratings_with_options = false
  $: {
    const options_loaded = !!task_rating_options
    const on_new_run = !!run?.id && run.id !== seeded_ratings_for_run_id
    const options_just_loaded = options_loaded && !seeded_ratings_with_options
    if (run?.id && (on_new_run || options_just_loaded)) {
      load_server_ratings(run, rating_requirements)
      seeded_ratings_for_run_id = run.id
      seeded_ratings_with_options = options_loaded
    }
  }

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

  let tags_error: KilnError | null = null

  async function save_tags(tags: string[]) {
    try {
      let patch_body = {
        tags: tags,
      }
      const updated = await patch_run(patch_body)
      tags_error = null
      on_run_updated?.(updated)
    } catch (err) {
      tags_error = createKilnError(err)
    }
  }

  async function save_ratings() {
    try {
      let requirement_ratings_obj: Record<string, RequirementRating | null> = {}
      rating_requirements.forEach((req, index) => {
        if (!req.id) {
          return
        }
        if (
          requirement_ratings[index] !== null &&
          requirement_ratings[index] !== undefined
        ) {
          requirement_ratings_obj[req.id] = {
            value: requirement_ratings[index],
            type: req.type,
          }
        } else {
          requirement_ratings_obj[req.id] = null
        }
      })
      let patch_body = {
        output: {
          rating: {
            value: overall_rating,
            type: "five_star",
            requirement_ratings: requirement_ratings_obj,
          },
        },
      }
      const updated = await patch_run(patch_body)
      save_rating_error = null
      posthog.capture("save_ratings", {})
      on_run_updated?.(updated)
    } catch (err) {
      save_rating_error = createKilnError(err)
    }
  }

  // ---- Subtask usage ----
  type SubtaskReference = {
    project_id: string
    task_id: string
    run_id: string
  }

  function extract_subtask_references(trace: Trace): SubtaskReference[] {
    const references: SubtaskReference[] = []
    for (const message of trace) {
      if (
        "kiln_task_tool_data" in message &&
        message.kiln_task_tool_data &&
        typeof message.kiln_task_tool_data === "string"
      ) {
        const [project_id, , task_id, run_id] =
          message.kiln_task_tool_data.split(":::")
        if (project_id && task_id && run_id) {
          references.push({ project_id, task_id, run_id })
        }
      }
    }
    return references
  }

  async function calculate_subtask_usage(
    trace: Trace | null | undefined,
    visited: Set<string> = new Set(),
  ): Promise<{ cost: number; tokens: number; latency_ms: number }> {
    if (!trace) return { cost: 0, tokens: 0, latency_ms: 0 }

    const references = extract_subtask_references(trace)
    let total_cost = 0
    let total_tokens = 0
    let total_llm_latency_ms = 0

    for (const ref of references) {
      const key = `${ref.project_id}:${ref.task_id}:${ref.run_id}`
      if (visited.has(key)) continue
      visited.add(key)

      try {
        const response = await client.GET(
          "/api/projects/{project_id}/tasks/{task_id}/runs/{run_id}",
          {
            params: {
              path: {
                project_id: ref.project_id,
                task_id: ref.task_id,
                run_id: ref.run_id,
              },
            },
          },
        )

        if (!response.error && response.data) {
          total_cost += response.data.usage?.cost ?? 0
          total_tokens += response.data.usage?.total_tokens ?? 0
          total_llm_latency_ms += response.data.usage?.total_llm_latency_ms ?? 0
          const subtask_usage = await calculate_subtask_usage(
            response.data.trace,
            visited,
          )
          total_cost += subtask_usage.cost
          total_tokens += subtask_usage.tokens
          total_llm_latency_ms += subtask_usage.latency_ms
        }
      } catch (error) {
        console.warn(
          "Failed to fetch subtask usage, continuing on error as subtask run may have been deleted.",
          {
            project_id: ref.project_id,
            task_id: ref.task_id,
            run_id: ref.run_id,
            error,
          },
        )
      }
    }

    return {
      cost: total_cost,
      tokens: total_tokens,
      latency_ms: total_llm_latency_ms,
    }
  }

  let subtask_cost: number | null = null
  let subtask_tokens: number | null = null
  let subtask_latency_ms: number | null = null
  let subtask_usage_loading = false
  // Counter to prevent race conditions: when run changes rapidly, multiple async requests
  // may be in flight. We only update state if this request is still the latest one.
  let subtask_usage_request_id = 0

  async function load_subtask_usage(trace: Trace | null | undefined) {
    const request_id = ++subtask_usage_request_id

    if (!trace || extract_subtask_references(trace).length === 0) {
      if (request_id === subtask_usage_request_id) {
        subtask_cost = null
        subtask_tokens = null
        subtask_latency_ms = null
        subtask_usage_loading = false
      }
      return
    }

    subtask_usage_loading = true
    try {
      const usage = await calculate_subtask_usage(trace)
      if (request_id === subtask_usage_request_id) {
        subtask_cost = usage.cost
        subtask_tokens = usage.tokens
        subtask_latency_ms = usage.latency_ms
      }
    } catch {
      if (request_id === subtask_usage_request_id) {
        subtask_cost = null
        subtask_tokens = null
        subtask_latency_ms = null
      }
    } finally {
      if (request_id === subtask_usage_request_id) {
        subtask_usage_loading = false
      }
    }
  }

  function get_usage_properties(
    base_cost: number,
    base_tokens: number,
    base_latency_ms: number,
    subtask_cost: number | null,
    subtask_usage_loading: boolean,
    subtask_tokens: number | null,
    subtask_latency_ms: number | null,
  ) {
    let properties = []

    const label = (metric: string) => "Total " + metric

    const run_cost = base_cost
    const run_tokens = base_tokens
    const run_latency = base_latency_ms

    if (subtask_usage_loading) {
      properties.push({
        name: label("Cost"),
        value: "Loading...",
      })
    } else {
      const total_cost = run_cost + (subtask_cost ?? 0)
      if (total_cost > 0) {
        properties.push({
          name: label("Cost"),
          value: `$${total_cost.toFixed(6)}`,
        })
      }
    }

    if (subtask_usage_loading) {
      properties.push({
        name: "Subtasks Cost",
        value: "Loading...",
      })
    } else if (subtask_cost && subtask_cost > 0) {
      properties.push({
        name: "Subtasks Cost",
        value: `$${subtask_cost.toFixed(6)}`,
      })
    }

    if (subtask_usage_loading) {
      properties.push({
        name: label("Tokens"),
        value: "Loading...",
      })
    } else {
      const total_tokens = run_tokens + (subtask_tokens ?? 0)
      if (total_tokens > 0) {
        properties.push({
          name: label("Tokens"),
          value: total_tokens,
        })
      }
    }

    if (subtask_usage_loading) {
      properties.push({
        name: "Subtasks Tokens",
        value: "Loading...",
      })
    } else if (subtask_tokens && subtask_tokens > 0) {
      properties.push({
        name: "Subtasks Tokens",
        value: subtask_tokens,
      })
    }

    if (subtask_usage_loading) {
      properties.push({
        name: label("Latency"),
        value: "Loading...",
      })
    } else {
      const total_latency = run_latency + (subtask_latency_ms ?? 0)
      if (total_latency > 0) {
        properties.push({
          name: label("Latency"),
          value: formatLatency(total_latency),
        })
      }
    }

    if (subtask_usage_loading) {
      properties.push({
        name: "Subtasks Latency",
        value: "Loading...",
      })
    } else if (subtask_latency_ms && subtask_latency_ms > 0) {
      properties.push({
        name: "Subtasks Latency",
        value: formatLatency(subtask_latency_ms),
      })
    }

    return properties
  }

  $: is_multiturn = task?.turn_mode === "multiturn"
  $: load_subtask_usage(run?.trace)

  // Total latency across the whole conversation. cumulative_usage carries
  // cost/tokens but not latency, so sum each message's latency_ms from the
  // trace (the leaf run's trace is the full transcript).
  function sum_trace_latency_ms(trace: Trace | null | undefined): number {
    if (!trace) return 0
    let total = 0
    for (const message of trace) {
      if (
        "latency_ms" in message &&
        typeof message.latency_ms === "number" &&
        message.latency_ms > 0
      ) {
        total += message.latency_ms
      }
    }
    return total
  }

  // Base (non-subtask) usage for the Usage panel. Multiturn shows the whole
  // conversation via the run's cumulative_usage (sum across every turn in the
  // trace, including seeded prior turns); single-turn shows the run's own
  // usage. cumulative_usage can be null on records predating the field, so
  // fall back to the run's usage.
  $: base_cost = is_multiturn
    ? run?.cumulative_usage?.cost ?? run?.usage?.cost ?? 0
    : run?.usage?.cost ?? 0
  $: base_tokens = is_multiturn
    ? run?.cumulative_usage?.total_tokens ?? run?.usage?.total_tokens ?? 0
    : run?.usage?.total_tokens ?? 0
  $: base_latency_ms = is_multiturn
    ? sum_trace_latency_ms(run?.trace)
    : run?.usage?.total_llm_latency_ms ?? 0

  $: usage_properties = get_usage_properties(
    base_cost,
    base_tokens,
    base_latency_ms,
    subtask_cost,
    subtask_usage_loading,
    subtask_tokens,
    subtask_latency_ms,
  )

  // ---- Feedback ----
  let feedbacks: Feedback[] = []
  let feedback_loading = false
  let feedback_error: KilnError | null = null
  let add_feedback_dialog: Dialog
  let add_feedback_open = false
  let view_feedback_dialog: Dialog
  let new_feedback_text = ""
  let add_feedback_submitting = false

  const MAX_VISIBLE_FEEDBACKS = 3

  type FeedbackSortColumn = "created_by" | "created_at"
  let feedback_sort_column: FeedbackSortColumn = "created_at"
  let feedback_sort_direction: "asc" | "desc" = "desc"

  function handle_feedback_sort(column: FeedbackSortColumn) {
    if (feedback_sort_column === column) {
      feedback_sort_direction =
        feedback_sort_direction === "asc" ? "desc" : "asc"
    } else {
      feedback_sort_column = column
      feedback_sort_direction = column === "created_at" ? "desc" : "asc"
    }
  }

  $: sorted_feedbacks = [...feedbacks].sort((a, b) => {
    const dir = feedback_sort_direction === "asc" ? 1 : -1
    if (feedback_sort_column === "created_by") {
      return dir * (a.created_by ?? "").localeCompare(b.created_by ?? "")
    }
    return dir * (a.created_at ?? "").localeCompare(b.created_at ?? "")
  })

  let feedback_request_id = 0
  let last_loaded_run_id: string | null = null

  async function load_feedback() {
    if (!task.id || !run?.id) {
      feedbacks = []
      last_loaded_run_id = null
      return
    }
    if (run.id === last_loaded_run_id) return

    const request_id = ++feedback_request_id
    feedback_loading = true
    feedbacks = []
    try {
      const { data, error: fetch_error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/runs/{run_id}/feedback",
        {
          params: {
            path: {
              project_id,
              task_id: task.id,
              run_id: run.id,
            },
          },
        },
      )
      if (request_id !== feedback_request_id) return
      if (fetch_error) throw fetch_error
      feedbacks = data
      feedback_error = null
      last_loaded_run_id = run.id
    } catch (err) {
      if (request_id !== feedback_request_id) return
      feedback_error = createKilnError(err)
    } finally {
      if (request_id === feedback_request_id) {
        feedback_loading = false
      }
    }
  }

  let add_feedback_error: KilnError | null = null

  async function submit_feedback() {
    if (!task.id || !run?.id || !new_feedback_text.trim()) return
    add_feedback_submitting = true
    try {
      const { data, error: fetch_error } = await client.POST(
        "/api/projects/{project_id}/tasks/{task_id}/runs/{run_id}/feedback",
        {
          params: {
            path: {
              project_id,
              task_id: task.id,
              run_id: run.id,
            },
          },
          body: {
            feedback: new_feedback_text.trim(),
            source: "run-page",
          },
        },
      )
      if (fetch_error) throw fetch_error
      ++feedback_request_id
      feedbacks = [...feedbacks, data]
      new_feedback_text = ""
      add_feedback_dialog.close()
      add_feedback_error = null
    } catch (err) {
      add_feedback_error = createKilnError(err)
    } finally {
      add_feedback_submitting = false
    }
  }

  async function delete_feedback(fb: Feedback) {
    if (!task.id || !run?.id || !fb.id) return
    try {
      const { error: fetch_error } = await client.DELETE(
        "/api/projects/{project_id}/tasks/{task_id}/runs/{run_id}/feedback/{feedback_id}",
        {
          params: {
            path: {
              project_id,
              task_id: task.id,
              run_id: run.id,
              feedback_id: fb.id,
            },
          },
        },
      )
      if (fetch_error) throw fetch_error
      feedbacks = feedbacks.filter((f) => f.id !== fb.id)
    } catch (err) {
      feedback_error = createKilnError(err)
    }
  }

  $: if (run?.id && task.id) load_feedback()
</script>

<div>
  <div class="text-xl font-bold mt-10 lg:mt-0 mb-6">
    Rating and Feedback
    {#if save_rating_error}
      <button class="tooltip" data-tip={save_rating_error.getMessage()}>
        <svg
          class="w-5 h-5 ml-1 text-error inline"
          viewBox="0 0 1024 1024"
          xmlns="http://www.w3.org/2000/svg"
          ><path
            fill="currentColor"
            d="M512 64a448 448 0 1 1 0 896 448 448 0 0 1 0-896zm0 192a58.432 58.432 0 0 0-58.24 63.744l23.36 256.384a35.072 35.072 0 0 0 69.76 0l23.296-256.384A58.432 58.432 0 0 0 512 256zm0 512a51.2 51.2 0 1 0 0-102.4 51.2 51.2 0 0 0 0 102.4z"
          /></svg
        >
      </button>
    {:else if rate_focus && mounted}
      <div class="w-7 h-7 ml-3 inline text-primary">
        <svg
          in:fly={{
            delay: 50,
            duration: 1000,
            easing: bounceOut,
            y: "-20px",
            opacity: 1,
          }}
          fill="currentColor"
          class="w-7 h-7 inline"
          viewBox="0 0 512 512"
          xmlns="http://www.w3.org/2000/svg"
          ><path
            d="M256,464c114.87,0,208-93.13,208-208S370.87,48,256,48,48,141.13,48,256,141.13,464,256,464ZM164.64,251.35a16,16,0,0,1,22.63-.09L240,303.58V170a16,16,0,0,1,32,0V303.58l52.73-52.32A16,16,0,1,1,347.27,274l-80,79.39a16,16,0,0,1-22.54,0l-80-79.39A16,16,0,0,1,164.64,251.35Z"
          /></svg
        >
      </div>
    {/if}
  </div>

  <div class="grid grid-cols-[auto,1fr] gap-4 text-sm 2xl:text-base">
    {#if rating_requirements}
      {#each rating_requirements as requirement, index}
        <div class="flex items-center">
          {requirement.name}:
          <InfoTooltip
            tooltip_text={`Requirement #${index + 1} - ${requirement.instruction || "No instruction provided"}${requirement.type === "pass_fail_critical" ? " Use 'critical' rating for responses which are never tolerable, beyond a typical failure." : ""}`}
          />
        </div>
        <div class="flex items-center">
          <Rating
            bind:rating={requirement_ratings[index]}
            type={requirement.type}
            size={6}
            on:rating_changed={save_ratings}
          />
        </div>
      {/each}
    {/if}
    <div class="flex items-center text-nowrap 2xl:min-w-32">
      <div class="font-medium">Overall Rating:</div>
      <div class="text-gray-500">
        <InfoTooltip tooltip_text="The overall rating of the output." />
      </div>
    </div>
    <div class="flex items-center">
      <Rating
        bind:rating={overall_rating}
        type="five_star"
        size={7}
        on:rating_changed={save_ratings}
      />
    </div>
    <div class="font-medium flex items-center text-nowrap">Feedback:</div>
    <div class="flex items-center">
      <button
        type="button"
        class="btn btn-outline btn-primary btn-xs"
        on:click={() => {
          new_feedback_text = ""
          add_feedback_error = null
          add_feedback_open = true
          add_feedback_dialog.show()
        }}>Add Feedback</button
      >
    </div>
    {#if feedback_error}
      <div></div>
      <p class="text-error text-xs">{feedback_error.getMessage()}</p>
    {/if}
    {#if feedback_loading}
      <div></div>
      <div>
        <span class="loading loading-spinner loading-xs"></span>
      </div>
    {:else if feedbacks.length > 0}
      <div></div>
      <!-- svelte-ignore a11y-no-static-element-interactions -->
      <div
        tabindex="0"
        role="button"
        class="text-left cursor-pointer hover:outline hover:outline-1 hover:outline-base-300 focus-visible:outline focus-visible:outline-1 focus-visible:outline-base-300 rounded px-1.5 py-1 -ml-1.5 transition-all outline-none"
        on:click={(e) => {
          const el = e.currentTarget
          if (el instanceof HTMLElement) el.blur()
          view_feedback_dialog.show()
        }}
        on:keydown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault()
            const el = e.currentTarget
            if (el instanceof HTMLElement) el.blur()
            view_feedback_dialog.show()
          }
        }}
      >
        {#each feedbacks.slice(-MAX_VISIBLE_FEEDBACKS).reverse() as fb}
          <div class="mb-2 last:mb-0">
            <div class="text-xs text-gray-500">
              {fb.created_by || "Unknown"}
            </div>
            <div class="text-sm line-clamp-2 whitespace-pre-line">
              {fb.feedback}
            </div>
          </div>
        {/each}
        {#if feedbacks.length > MAX_VISIBLE_FEEDBACKS}
          <div class="text-xs text-gray-400 mt-1">
            {feedbacks.length - MAX_VISIBLE_FEEDBACKS} more…
          </div>
        {/if}
      </div>
    {/if}
  </div>
  <div class="mt-8 mb-4">
    <div class="text-xl font-bold">Tags</div>
    {#if tags_error}
      <p class="text-error text-sm">
        {tags_error.getMessage()}
      </p>
    {/if}
    <TagPicker
      tags={run.tags}
      tag_type="task_run"
      {project_id}
      task_id={task.id || null}
      initial_expanded={false}
      on:tags_changed={(event) => {
        const { current } = event.detail
        run.tags = current
        save_tags(current)
      }}
    />
  </div>
  <div>
    {#if usage_properties && usage_properties.length > 0}
      <PropertyList
        properties={usage_properties}
        title={is_multiturn ? "Total Usage" : "Usage"}
      />
    {/if}
  </div>
</div>

<Dialog
  bind:this={add_feedback_dialog}
  title="Add Feedback"
  width="wide"
  sub_subtitle="What worked well or fell short — the more specific, the more useful for improving results."
  on:close={() => (add_feedback_open = false)}
>
  {#if add_feedback_open}
    <FormContainer
      submit_label="Save"
      on:submit={submit_feedback}
      bind:submitting={add_feedback_submitting}
      bind:error={add_feedback_error}
    >
      <FormElement
        id="new_feedback_input"
        hide_label={true}
        label="Feedback"
        inputType="textarea"
        height="medium"
        placeholder="e.g., Tone was off — too casual; the second paragraph contradicts the first; factually wrong — it confused X with Y"
        bind:value={new_feedback_text}
      />
    </FormContainer>
  {/if}
</Dialog>

<Dialog bind:this={view_feedback_dialog} title="All Feedback" width="wide">
  {#if feedbacks.length === 0}
    <p class="text-sm text-gray-500">No feedback yet.</p>
  {:else}
    <div class="rounded-lg border overflow-hidden">
      <table class="table table-sm w-full table-fixed">
        <thead>
          <tr>
            <th
              class="w-[15%] hover:bg-base-200 cursor-pointer"
              on:click={() => handle_feedback_sort("created_by")}
            >
              Created By
              <span class="inline-block w-3 text-center">
                {feedback_sort_column === "created_by"
                  ? feedback_sort_direction === "asc"
                    ? "▲"
                    : "▼"
                  : "​"}
              </span>
            </th>
            <th>Feedback</th>
            <th
              class="w-[25%] hover:bg-base-200 cursor-pointer"
              on:click={() => handle_feedback_sort("created_at")}
            >
              Created At
              <span class="inline-block w-3 text-center">
                {feedback_sort_column === "created_at"
                  ? feedback_sort_direction === "asc"
                    ? "▲"
                    : "▼"
                  : "​"}
              </span>
            </th>
            <th class="w-14"></th>
          </tr>
        </thead>
        <tbody>
          {#each sorted_feedbacks as fb}
            <tr>
              <td class="whitespace-nowrap align-top"
                >{fb.created_by || "Unknown"}</td
              >
              <td class="whitespace-pre-wrap break-words align-top"
                >{fb.feedback}</td
              >
              <td class="align-top">{formatDate(fb.created_at)}</td>
              <td class="align-top pr-3">
                <TableActionMenu
                  items={[
                    {
                      label: "Delete",
                      onclick: () => delete_feedback(fb),
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
</Dialog>
