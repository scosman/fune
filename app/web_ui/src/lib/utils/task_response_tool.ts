// The synthetic tool a model calls to hand back a structured answer. The user
// never defined it, so the UI shows its arguments as the answer rather than as
// a tool call. Must match TASK_RESPONSE_TOOL_NAME in
// libs/core/kiln_ai/utils/open_ai_types.py; a desktop test fails if they drift.
export const TASK_RESPONSE_TOOL_NAME = "task_response"
