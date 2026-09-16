import type { ToolSetType } from "$lib/types"
import { assertNever } from "$lib/utils/exhaustive"

const FINE_TUNE_PROMPT_PREFIX = "fine_tune_prompt::"

export function prompt_link(
  project_id: string,
  task_id: string,
  prompt_id: string,
): string | undefined {
  if (!project_id || !task_id || !prompt_id) {
    return undefined
  }
  // Special case for fine-tuned prompts
  if (prompt_id.startsWith(FINE_TUNE_PROMPT_PREFIX)) {
    // Get the last component of the prompt ID. fine_tune_prompt::[project_id]::[task_id]::fine_tune_id
    const trimmed_prompt_id = prompt_id.replace(FINE_TUNE_PROMPT_PREFIX, "")
    const fine_tune_id =
      trimmed_prompt_id.split("::").pop() || trimmed_prompt_id
    return `/fine_tune/${project_id}/${task_id}/fine_tune/${encodeURIComponent(fine_tune_id)}`
  }
  // ID style prompt, link to saved
  return `/prompts/${project_id}/${task_id}/saved/${encodeURIComponent(prompt_id)}`
}

export function dataset_item_link(
  project_id: string,
  task_id: string,
  run_id: string,
): string | null {
  if (!project_id || !task_id || !run_id) {
    return null
  }
  return `/dataset/${project_id}/${task_id}/${run_id}/run`
}

export function tool_link(project_id: string, tool_id: string): string | null {
  if (!project_id || !tool_id) {
    return null
  }
  if (
    tool_id.startsWith("mcp::remote::") ||
    tool_id.startsWith("mcp::local::")
  ) {
    return `/tools/${project_id}/tool_servers/${tool_id.split("::")[2]}`
  } else if (tool_id.startsWith("kiln_task::")) {
    return `/tools/${project_id}/kiln_task/${tool_id.split("::")[1]}`
  } else if (tool_id.startsWith("kiln_tool::skill::")) {
    return `/skills/${project_id}/${tool_id.split("::")[2]}`
  } else if (tool_id.startsWith("kiln_tool::rag::")) {
    return `/docs/rag_configs/${project_id}/${tool_id.split("::")[2]}/rag_config`
  } else if (tool_id.startsWith("kiln_tool::code::")) {
    return `/tools/${project_id}/code_tools/${tool_id.split("::")[2]}`
  } else if (tool_id.startsWith("kiln_tool::")) {
    return `/tools/${project_id}`
  } else {
    return null
  }
}

export function tool_set_type_label(type: ToolSetType): string {
  switch (type) {
    case "code":
      return "Code Tool"
    case "mcp":
      return "MCP"
    case "search":
      return "Search"
    case "kiln_task":
      return "Kiln Task"
    case "demo":
      return "Demo"
    case "skill":
      return "Skill"
    case "builtin":
      return "Built-in"
    case "sandbox_code":
      return "Sandbox Code"
    default:
      // Compile error here means a new ToolSetType needs a label above.
      return assertNever(type)
  }
}
