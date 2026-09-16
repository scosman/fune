import { client } from "$lib/api_client"
import type { Eval, EvalStatus, SpecType } from "$lib/types"
import type { components } from "$lib/api_schema"
import type { V2EvalType } from "$lib/utils/eval_types/registry"
import { spec_field_configs } from "./select_template/spec_templates"
import posthog from "posthog-js"

export type SuggestedEdit = {
  proposed_value: string
  reason_for_edit: string
}

/**
 * The creation workflow the spec builder runs, carried as a `workflow` query
 * param. Manual is the only one: the builder fills the template form by hand.
 */
export type SpecWorkflow = "manual"

/**
 * The judge each template implies. Programmatic judges are chosen directly on
 * the template picker (not via a template), so every template's judge is
 * implied: an LLM judge for the written-rubric templates, and the Tool Call
 * Check judge for tool call, which reads the trace directly and collects its
 * own tool list.
 */
export function implied_judge_for_spec_type(spec_type: SpecType): V2EvalType {
  switch (spec_type) {
    case "appropriate_tool_use":
      return "tool_call_check"
    default:
      return "llm_judge"
  }
}

/**
 * The EvalTemplateId recorded on evals created without a spec. The open-ended
 * behaviour mappings are kept for flows that record a template after the fact.
 * Tool use deliberately maps to none: the legacy "tool_call" template means a
 * pre-spec LLM judge over the trace, while new tool evals are the Tool Call
 * Check programmatic judge — recording the legacy template on them would
 * conflate the two. Mirrors the server's spec_eval_template.
 */
export function eval_template_for_spec_type(
  spec_type: SpecType,
): components["schemas"]["EvalTemplateId"] | null {
  switch (spec_type) {
    case "desired_behaviour":
      return "desired_behaviour"
    case "issue":
      return "kiln_issue"
    default:
      return null
  }
}

/**
 * The next screen after a template is picked: the spec builder, with the
 * template's implied judge.
 */
export function next_page_after_template(
  project_id: string,
  task_id: string,
  spec_type: SpecType,
): string {
  return next_page_after_judge(
    project_id,
    task_id,
    spec_type,
    implied_judge_for_spec_type(spec_type),
  )
}

/**
 * The spec builder URL for a judge picked directly from the template screen's
 * programmatic checks section. No template is involved: the eval is created
 * spec-less and template-less, with just a name and the judge's own config.
 */
export function judge_only_builder_url(
  project_id: string,
  task_id: string,
  judge: V2EvalType,
): string {
  const params = new URLSearchParams({ judge, workflow: "manual" })
  return `/specs/${project_id}/${task_id}/spec_builder?${params.toString()}`
}

/**
 * The next screen once both template and judge are known: always the spec
 * builder. Every template's judge is implied, so there is nothing left to ask
 * between the picker and the builder.
 */
export function next_page_after_judge(
  project_id: string,
  task_id: string,
  spec_type: SpecType,
  judge: V2EvalType,
): string {
  return spec_builder_url(project_id, task_id, spec_type, "manual", judge)
}

export function spec_builder_url(
  project_id: string,
  task_id: string,
  spec_type: SpecType,
  workflow: SpecWorkflow,
  judge: V2EvalType,
): string {
  const params = new URLSearchParams({
    type: spec_type,
    workflow,
    judge,
  })
  return `/specs/${project_id}/${task_id}/spec_builder?${params.toString()}`
}

/**
 * Build a definition string from properties
 * @param specType - The type of spec
 * @param properties - The properties of the spec
 * @returns The definition string
 */
export function buildSpecDefinition(
  specType: SpecType,
  properties: Record<string, string | null>,
): string {
  const fieldConfigs = spec_field_configs[specType]
  const parts: string[] = []

  for (const field of fieldConfigs) {
    const value = properties[field.key]
    if (value && value.trim()) {
      parts.push(`## ${field.label}\n${value}`)
    }
  }

  return parts.join("\n\n")
}

/**
 * Update an eval's priority via the API. Priority lives on the eval (spec-backed
 * or not); the server resolves legacy files through their spec on read.
 * @returns The updated eval
 * @throws Error if the API call fails
 */
export async function updateEvalPriority(
  project_id: string,
  task_id: string,
  evaluator: Eval,
  newPriority: number,
): Promise<Eval> {
  if (!evaluator.id) {
    throw new Error("Eval ID is required")
  }

  const { data, error } = await client.PATCH(
    "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}",
    {
      params: {
        path: { project_id, task_id, eval_id: evaluator.id },
      },
      body: {
        priority: newPriority as 0 | 1 | 2 | 3,
      },
    },
  )

  if (error) {
    throw error
  }

  posthog.capture("update_eval_priority", {
    old_priority: evaluator.priority,
    new_priority: newPriority,
  })

  return data
}

/**
 * Update an eval's status via the API. Status lives on the eval (spec-backed
 * or not); the server resolves legacy files through their spec on read.
 * @returns The updated eval
 * @throws Error if the API call fails
 */
export async function updateEvalStatus(
  project_id: string,
  task_id: string,
  evaluator: Eval,
  newStatus: EvalStatus,
): Promise<Eval> {
  if (!evaluator.id) {
    throw new Error("Eval ID is required")
  }

  const { data, error } = await client.PATCH(
    "/api/projects/{project_id}/tasks/{task_id}/evals/{eval_id}",
    {
      params: {
        path: { project_id, task_id, eval_id: evaluator.id },
      },
      body: {
        status: newStatus,
      },
    },
  )

  if (error) {
    throw error
  }

  posthog.capture("update_eval_status", {
    old_status: evaluator.status,
    new_status: newStatus,
  })

  return data
}

/**
 * Extract a tag from a filter_id string (e.g., "tag::my_tag" -> "my_tag")
 * @param filter_id - The filter ID to extract the tag from
 * @returns The tag if the filter_id is a tag filter, undefined otherwise
 */
export function tagFromFilterId(filter_id: string): string | undefined {
  if (filter_id.startsWith("tag::")) {
    return filter_id.replace("tag::", "")
  }
  return undefined
}

/**
 * Generate a dataset link from a filter_id
 * @param project_id - The project ID
 * @param task_id - The task ID
 * @param filter_id - The filter ID to generate a link from
 * @returns The dataset URL if the filter_id is a tag filter, undefined otherwise
 */
export function linkFromFilterId(
  project_id: string,
  task_id: string,
  filter_id: string | null | undefined,
): string | undefined {
  if (!filter_id) {
    return undefined
  }
  const tag = tagFromFilterId(filter_id)
  if (tag) {
    return `/dataset/${project_id}/${task_id}?tags=${tag}`
  }
  return undefined
}
