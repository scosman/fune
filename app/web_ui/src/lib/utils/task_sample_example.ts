import { client } from "$lib/api_client"
import type { TaskRun, TaskRunOutput } from "$lib/types"

/**
 * A task sample example consisting of an input/output pair.
 */
export type TaskSampleExample = {
  input: string
  output: string
}

/**
 * How the example was auto-selected (data label, not UI state).
 */
export type AutoSelectType = "highly_rated" | "most_recent"

/**
 * Result of fetching task sample examples from the dataset.
 */
export type TaskSampleFetchResult = {
  auto_select_type: AutoSelectType | null
  selected_example: TaskSampleExample | null
  available_runs: TaskRun[]
}

/**
 * Checks if a task run has a 5-star rating.
 */
export function is_five_star_rated(run: TaskRun): boolean {
  const rating = run.output?.rating
  if (!rating) return false
  return rating.type === "five_star" && rating.value === 5
}

/**
 * Extract the output string from a task run, preferring repaired output.
 */
function get_output_string(run: TaskRun | TaskRunOutput): string {
  if (run.repaired_output?.output) {
    return run.repaired_output.output
  }
  return run.output?.output ?? ""
}

/**
 * Converts a TaskRun to a TaskSampleExample.
 */
export function task_run_to_example(
  run: TaskRun | TaskRunOutput,
): TaskSampleExample {
  return {
    input: run.input ?? "",
    output: get_output_string(run),
  }
}

/**
 * Fetches task runs and determines the task sample selection status.
 *
 * Priority (over the runs `candidate_filter` keeps; no filter = all runs):
 * 1. If a 5-star rated sample exists, auto-select it (confident)
 * 2. If samples exist but none are 5-star, auto-select the most recent
 * 3. If no samples exist, indicate manual entry is needed
 *
 * @throws Error if fetching fails
 */
export async function fetch_task_sample_candidates(
  project_id: string,
  task_id: string,
  candidate_filter?: (run: TaskRun) => boolean,
): Promise<TaskSampleFetchResult> {
  const { data: runs, error } = await client.GET(
    "/api/projects/{project_id}/tasks/{task_id}/runs",
    {
      params: {
        path: { project_id, task_id },
      },
    },
  )

  if (error) {
    throw new Error(
      typeof error === "string"
        ? error
        : (error as { message?: string }).message ?? "Failed to fetch runs",
    )
  }

  const candidates = candidate_filter ? runs?.filter(candidate_filter) : runs
  if (!candidates || candidates.length === 0) {
    return {
      auto_select_type: null,
      selected_example: null,
      available_runs: [],
    }
  }

  // Sort runs by recency (most recent first)
  const sorted_runs = [...candidates].sort((a, b) => {
    const a_date = a.created_at ? new Date(a.created_at).getTime() : 0
    const b_date = b.created_at ? new Date(b.created_at).getTime() : 0
    return b_date - a_date
  })

  // First, look for 5-star rated samples (most recent 5-star)
  const five_star_run = sorted_runs.find(is_five_star_rated)
  if (five_star_run) {
    return {
      auto_select_type: "highly_rated",
      selected_example: task_run_to_example(five_star_run),
      available_runs: sorted_runs,
    }
  }

  // No 5-star samples, but samples exist - auto-select the most recent
  const most_recent = sorted_runs[0]

  return {
    auto_select_type: "most_recent",
    selected_example: task_run_to_example(most_recent),
    available_runs: sorted_runs,
  }
}

/**
 * Builds a task prompt with instruction, requirements, and optional task sample examples.
 * Uses the backend prompt builder for consistent formatting.
 */
export async function build_prompt_with_task_sample(
  project_id: string,
  task_id: string,
  examples: TaskSampleExample[],
): Promise<string> {
  const { data, error } = await client.POST(
    "/api/projects/{project_id}/tasks/{task_id}/build_prompt_with_examples",
    {
      params: {
        path: { project_id, task_id },
      },
      body: {
        examples: examples,
      },
    },
  )

  if (error) {
    throw new Error(
      typeof error === "string"
        ? error
        : (error as { message?: string }).message ?? "Failed to build prompt",
    )
  }

  return data?.prompt ?? ""
}
