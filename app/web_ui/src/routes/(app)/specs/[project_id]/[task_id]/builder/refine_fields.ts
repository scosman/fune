import type { SuggestedEdit } from "../spec_utils"

// One proposed edit from the refine call's response.
export type ProposedSpecEdit = {
  spec_field_name: string
  proposed_edit: string
  reason_for_edit?: string
}

export type RefineEditSplit = {
  // Proposed values for fields the form renders, keyed by field name.
  refined_edits: Record<string, string>
  // The same edits with their reasons, for the per-field refinement notes.
  suggested_edits: Record<string, SuggestedEdit>
  // Field names of discarded edits, for telemetry. The request declares
  // only the rendered fields, so any edit here means the refine contract
  // drifted and is worth knowing about, even though the user is not shown
  // anything (there is nothing they could do with it).
  dropped_fields: string[]
}

// Split the refine response into edits the form can show and edits it
// cannot. The form is the only surface the user reviews before save, so an
// edit for a field it does not render must not merge into the saved values.
export function split_refine_edits(
  proposed_edits: ProposedSpecEdit[],
  rendered_fields: readonly string[],
): RefineEditSplit {
  const rendered = new Set(rendered_fields)
  const refined_edits: Record<string, string> = {}
  const suggested_edits: Record<string, SuggestedEdit> = {}
  const dropped_fields: string[] = []
  for (const edit of proposed_edits) {
    if (rendered.has(edit.spec_field_name)) {
      refined_edits[edit.spec_field_name] = edit.proposed_edit
      suggested_edits[edit.spec_field_name] = {
        proposed_value: edit.proposed_edit,
        reason_for_edit: edit.reason_for_edit ?? "",
      }
    } else {
      dropped_fields.push(edit.spec_field_name)
    }
  }
  return { refined_edits, suggested_edits, dropped_fields }
}

// Drop record keys for fields the refine form does not render. Drafts
// written before the lean form can carry example-field values that no
// longer have a surface; restoring them would silently reach the saved spec.
export function keep_rendered_fields<T>(
  record: Record<string, T>,
  rendered_fields: readonly string[],
): Record<string, T> {
  const rendered = new Set(rendered_fields)
  return Object.fromEntries(
    Object.entries(record).filter(([key]) => rendered.has(key)),
  )
}
