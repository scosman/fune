import { describe, expect, it } from "vitest"
import {
  keep_rendered_fields,
  split_refine_edits,
  type ProposedSpecEdit,
} from "./refine_fields"

const RENDERED = ["issue_description"]

describe("split_refine_edits", () => {
  it("passes rendered-field edits through untouched", () => {
    const edits: ProposedSpecEdit[] = [
      {
        spec_field_name: "issue_description",
        proposed_edit: "No clickbait headlines.",
        reason_for_edit: "Tightened wording.",
      },
    ]
    const result = split_refine_edits(edits, RENDERED)
    expect(result.refined_edits).toEqual({
      issue_description: "No clickbait headlines.",
    })
    expect(result.suggested_edits).toEqual({
      issue_description: {
        proposed_value: "No clickbait headlines.",
        reason_for_edit: "Tightened wording.",
      },
    })
    expect(result.dropped_fields).toEqual([])
  })

  it("splits a mixed response and reports the dropped field names", () => {
    const edits: ProposedSpecEdit[] = [
      {
        spec_field_name: "issue_description",
        proposed_edit: "No clickbait headlines.",
      },
      {
        spec_field_name: "issue_examples",
        proposed_edit: "One simple trick!",
      },
      {
        spec_field_name: "mystery_field",
        proposed_edit: "text",
      },
    ]
    const result = split_refine_edits(edits, RENDERED)
    expect(Object.keys(result.refined_edits)).toEqual(["issue_description"])
    expect(Object.keys(result.suggested_edits)).toEqual(["issue_description"])
    expect(result.dropped_fields).toEqual(["issue_examples", "mystery_field"])
  })

  it("returns empty results for an empty edits array", () => {
    const result = split_refine_edits([], RENDERED)
    expect(result.refined_edits).toEqual({})
    expect(result.suggested_edits).toEqual({})
    expect(result.dropped_fields).toEqual([])
  })

  it("defaults a missing edit reason to an empty string", () => {
    const edits: ProposedSpecEdit[] = [
      { spec_field_name: "issue_description", proposed_edit: "Tighter." },
    ]
    const result = split_refine_edits(edits, RENDERED)
    expect(result.suggested_edits.issue_description.reason_for_edit).toBe("")
  })

  it("keeps the last edit when a rendered field name repeats", () => {
    const edits: ProposedSpecEdit[] = [
      { spec_field_name: "issue_description", proposed_edit: "First." },
      { spec_field_name: "issue_description", proposed_edit: "Second." },
    ]
    const result = split_refine_edits(edits, RENDERED)
    expect(result.refined_edits).toEqual({ issue_description: "Second." })
    expect(result.suggested_edits.issue_description.proposed_value).toBe(
      "Second.",
    )
  })
})

describe("keep_rendered_fields", () => {
  it("drops keys for fields that are not rendered", () => {
    const record = {
      issue_description: "keep",
      issue_examples: "drop",
      non_issue_examples: "drop too",
    }
    expect(keep_rendered_fields(record, RENDERED)).toEqual({
      issue_description: "keep",
    })
  })

  it("returns an empty record when nothing is rendered", () => {
    expect(keep_rendered_fields({ a: 1 }, [])).toEqual({})
  })
})
