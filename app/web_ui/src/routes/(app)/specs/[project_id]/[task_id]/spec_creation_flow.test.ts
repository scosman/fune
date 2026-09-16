import { describe, it, expect } from "vitest"
import {
  implied_judge_for_spec_type,
  eval_template_for_spec_type,
  next_page_after_template,
  next_page_after_judge,
  judge_only_builder_url,
  spec_builder_url,
  buildSpecDefinition,
} from "./spec_utils"
import {
  spec_field_configs,
  core_field_config,
} from "./select_template/spec_templates"
import type { SpecType } from "$lib/types"

const ALL_SPEC_TYPES = Object.keys(spec_field_configs) as SpecType[]

describe("core field invariant", () => {
  // The non-LLM judge flow only shows the core field and saves every other field
  // from its template default. If a template ever lacks a core field, or has a
  // required field with no default, that flow would POST a spec the backend
  // property validators reject.
  it.each(ALL_SPEC_TYPES)(
    "%s has exactly one core field, and every other required field has a default",
    (spec_type) => {
      const fields = spec_field_configs[spec_type]
      const core_fields = fields.filter((field) => field.core)
      expect(core_fields).toHaveLength(1)

      const unfillable = fields.filter(
        (field) =>
          !field.core && field.required && field.default_value === undefined,
      )
      expect(unfillable).toEqual([])
    },
  )

  it.each(ALL_SPEC_TYPES)(
    "%s builds a non-empty definition from just its core field",
    (spec_type) => {
      // Spec.definition is min_length=1 on the backend. Reproduce what the spec
      // builder saves for a non-LLM judge: template defaults + the core field.
      const values: Record<string, string | null> = {}
      for (const field of spec_field_configs[spec_type]) {
        if (field.default_value !== undefined) {
          values[field.key] = field.default_value
        }
      }
      values[core_field_config(spec_type).key] = "The user's description."

      expect(buildSpecDefinition(spec_type, values).trim()).not.toBe("")
    },
  )
})

describe("implied_judge_for_spec_type", () => {
  it("pins tool call evals to the tool call check judge", () => {
    expect(implied_judge_for_spec_type("appropriate_tool_use")).toBe(
      "tool_call_check",
    )
  })

  it("pins every other template to an LLM judge", () => {
    const llm_judged = ALL_SPEC_TYPES.filter(
      (spec_type) => implied_judge_for_spec_type(spec_type) === "llm_judge",
    )
    expect(llm_judged).toContain("desired_behaviour")
    expect(llm_judged).toContain("issue")
    expect(llm_judged).toContain("reference_answer_accuracy")
    expect(llm_judged).toContain("toxicity")
    expect(llm_judged).not.toContain("appropriate_tool_use")
  })
})

describe("eval_template_for_spec_type", () => {
  it("maps the open behaviour templates", () => {
    expect(eval_template_for_spec_type("issue")).toBe("kiln_issue")
    expect(eval_template_for_spec_type("desired_behaviour")).toBe(
      "desired_behaviour",
    )
  })

  // The legacy "tool_call" template means a pre-spec LLM judge over the
  // trace; new tool evals are the template-less Tool Call Check judge, so
  // recording that template on them would conflate the two.
  it("never records the legacy tool_call template on new tool evals", () => {
    expect(eval_template_for_spec_type("appropriate_tool_use")).toBe(null)
  })
})

describe("next_page_after_template", () => {
  // The template picker is the last question in the flow: every template goes
  // straight to the manual spec builder, with no workflow screen in between.
  it.each(ALL_SPEC_TYPES)(
    "sends %s to the manual spec builder",
    (spec_type) => {
      const url = new URL(
        next_page_after_template("p1", "t1", spec_type),
        "http://localhost",
      )
      expect(url.pathname).toBe("/specs/p1/t1/spec_builder")
      expect(url.searchParams.get("type")).toBe(spec_type)
      expect(url.searchParams.get("workflow")).toBe("manual")
      expect(url.searchParams.get("judge")).toBe(
        implied_judge_for_spec_type(spec_type),
      )
    },
  )

  it("prefills the tool call judge for tool call evals", () => {
    const url = next_page_after_template("p1", "t1", "appropriate_tool_use")
    expect(url).toContain("judge=tool_call_check")
  })
})

describe("next_page_after_judge", () => {
  // Both the judge combos Kiln Pro used to claim and the ones it never
  // supported now land in the same place.
  it.each([["llm_judge"], ["code_eval"], ["exact_match"]] as const)(
    "sends %s straight to the manual builder",
    (judge) => {
      const url = new URL(
        next_page_after_judge("p1", "t1", "issue", judge),
        "http://localhost",
      )
      expect(url.pathname).toBe("/specs/p1/t1/spec_builder")
      expect(url.searchParams.get("judge")).toBe(judge)
      expect(url.searchParams.get("workflow")).toBe("manual")
    },
  )
})

describe("judge_only_builder_url", () => {
  it("routes to the spec builder with judge and manual workflow, no type", () => {
    const url = new URL(
      judge_only_builder_url("p1", "t1", "code_eval"),
      "http://localhost",
    )
    expect(url.pathname).toBe("/specs/p1/t1/spec_builder")
    expect(url.searchParams.get("judge")).toBe("code_eval")
    expect(url.searchParams.get("workflow")).toBe("manual")
    expect(url.searchParams.get("type")).toBeNull()
  })
})

describe("spec_builder_url", () => {
  it("encodes the template, workflow, and judge", () => {
    const url = new URL(
      spec_builder_url("p1", "t1", "issue", "manual", "pattern_match"),
      "http://localhost",
    )
    expect(url.pathname).toBe("/specs/p1/t1/spec_builder")
    expect(url.searchParams.get("type")).toBe("issue")
    expect(url.searchParams.get("workflow")).toBe("manual")
    expect(url.searchParams.get("judge")).toBe("pattern_match")
  })
})

describe("the removed Kiln Pro creation path", () => {
  // The workflow screen and the builder's pro mode are gone. No entry point
  // into eval creation may route to either, whatever the template or judge.
  it.each(ALL_SPEC_TYPES)(
    "is unreachable from the %s template",
    (spec_type) => {
      const urls = [
        next_page_after_template("p1", "t1", spec_type),
        next_page_after_judge(
          "p1",
          "t1",
          spec_type,
          implied_judge_for_spec_type(spec_type),
        ),
        judge_only_builder_url("p1", "t1", "code_eval"),
      ]
      for (const url of urls) {
        expect(url).not.toContain("select_workflow")
        expect(
          new URL(url, "http://localhost").searchParams.get("workflow"),
        ).toBe("manual")
      }
    },
  )
})
