// @vitest-environment jsdom
import { describe, it, expect, vi, beforeAll } from "vitest"
import { render, fireEvent } from "@testing-library/svelte"
import { tick } from "svelte"
import { available_tools } from "$lib/stores"
import type { ToolSetApiDescription } from "$lib/types"

vi.mock("$lib/utils/form_element.svelte", async () => {
  const { default: Stub } = await import("./__tests__/form_element_stub.svelte")
  return { default: Stub }
})

vi.mock("$lib/utils/form_list.svelte", async () => {
  const { default: Stub } = await import("./__tests__/form_list_stub.svelte")
  return { default: Stub }
})

vi.mock("$lib/ui/collapse.svelte", async () => {
  const { default: Stub } = await import("./__tests__/collapse_stub.svelte")
  return { default: Stub }
})

vi.mock("./tag_input.svelte", async () => {
  const { default: Stub } = await import("./__tests__/tag_input_stub.svelte")
  return { default: Stub }
})

vi.mock("$lib/ui/dialog.svelte", async () => {
  const { default: Stub } = await import("./__tests__/dialog_stub.svelte")
  return { default: Stub }
})

const PatternMatchForm = (await import("./pattern_match_form.svelte")).default
const ExactMatchForm = (await import("./exact_match_form.svelte")).default
const ContainsForm = (await import("./contains_form.svelte")).default
const SetCheckForm = (await import("./set_check_form.svelte")).default
const StepCountCheckForm = (await import("./step_count_check_form.svelte"))
  .default
const ToolCallCheckForm = (await import("./tool_call_check_form.svelte"))
  .default

beforeAll(() => {
  if (typeof globalThis.ResizeObserver === "undefined") {
    // eslint-disable-next-line @typescript-eslint/no-extraneous-class
    class ResizeObserverStub {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    ;(
      globalThis as unknown as { ResizeObserver: typeof ResizeObserverStub }
    ).ResizeObserver = ResizeObserverStub
  }
})

describe("PatternMatchForm", () => {
  it("validate returns error for empty pattern", () => {
    const { component } = render(PatternMatchForm, {
      props: {
        properties: {
          type: "pattern_match" as const,
          pattern: "",
          mode: "must_match" as const,
          value_expression: null,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBe(
      "Regular expression is required.",
    )
  })

  it("validate returns error for invalid regex", () => {
    const { component } = render(PatternMatchForm, {
      props: {
        properties: {
          type: "pattern_match" as const,
          pattern: "[invalid",
          mode: "must_match" as const,
          value_expression: null,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBe(
      "Invalid regular expression pattern.",
    )
  })

  it("validate returns null for valid regex", () => {
    const { component } = render(PatternMatchForm, {
      props: {
        properties: {
          type: "pattern_match" as const,
          pattern: "^hello.*world$",
          mode: "must_match" as const,
          value_expression: null,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBeNull()
  })

  it("getProperties returns current properties", () => {
    const { component } = render(PatternMatchForm, {
      props: {
        properties: {
          type: "pattern_match" as const,
          pattern: "test",
          mode: "must_not_match" as const,
          value_expression: null,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const props = (component as any).getProperties()
    expect(props.type).toBe("pattern_match")
    expect(props.pattern).toBe("test")
    expect(props.mode).toBe("must_not_match")
  })
})

describe("ExactMatchForm", () => {
  it("validate returns error when expected_value source is selected but empty", () => {
    const { component } = render(ExactMatchForm, {
      props: {
        properties: {
          type: "exact_match" as const,
          case_sensitive: true,
          value_expression: null,
          expected_value: null,
          reference_key: null,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBe("Expected value is required.")
  })

  it("validate returns null when expected_value is set", () => {
    const { component } = render(ExactMatchForm, {
      props: {
        properties: {
          type: "exact_match" as const,
          case_sensitive: true,
          value_expression: null,
          expected_value: "hello",
          reference_key: null,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBeNull()
  })
})

describe("ContainsForm", () => {
  it("validate returns error when substring source is selected but empty", () => {
    const { component } = render(ContainsForm, {
      props: {
        properties: {
          type: "contains" as const,
          case_sensitive: true,
          mode: "must_contain" as const,
          value_expression: null,
          substring: null,
          reference_key: null,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBe("Substring is required.")
  })

  it("validate returns null when substring is set", () => {
    const { component } = render(ContainsForm, {
      props: {
        properties: {
          type: "contains" as const,
          case_sensitive: true,
          mode: "must_contain" as const,
          value_expression: null,
          substring: "hello",
          reference_key: null,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBeNull()
  })
})

describe("SetCheckForm", () => {
  it("validate returns error when expected_set is empty", () => {
    const { component } = render(SetCheckForm, {
      props: {
        properties: {
          type: "set_check" as const,
          mode: "equal" as const,
          value_expression: null,
          expected_set: [],
          reference_key: null,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBe(
      "Expected set must contain at least one value.",
    )
  })

  it("validate returns null when expected_set has values", () => {
    const { component } = render(SetCheckForm, {
      props: {
        properties: {
          type: "set_check" as const,
          mode: "equal" as const,
          value_expression: null,
          expected_set: ["a", "b"],
          reference_key: null,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBeNull()
  })

  it("getProperties always sends explicit mode", () => {
    const { component } = render(SetCheckForm, {
      props: {
        properties: {
          type: "set_check" as const,
          mode: "superset" as const,
          value_expression: null,
          expected_set: ["a"],
          reference_key: null,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const props = (component as any).getProperties()
    expect(props.mode).toBe("superset")
    expect(props.type).toBe("set_check")
  })
})

describe("StepCountCheckForm", () => {
  it("validate returns error when neither min nor max is set", () => {
    const { component } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "tool_calls" as const,
          min_count: null,
          max_count: null,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBe(
      "At least one bound must be set.",
    )
  })

  it("validate returns error when min > max", () => {
    const { component } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "tool_calls" as const,
          min_count: 10,
          max_count: 5,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBe(
      "Minimum must be less than or equal to maximum.",
    )
  })

  it("validate returns null when only min is set", () => {
    const { component } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "model_responses" as const,
          min_count: 1,
          max_count: null,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBeNull()
  })

  it("validate returns null when both min and max are set correctly", () => {
    const { component } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "turns" as const,
          min_count: 2,
          max_count: 10,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBeNull()
  })
})

describe("SetCheckForm tag input", () => {
  it("renders tag input stub for expected_set source", () => {
    const { container } = render(SetCheckForm, {
      props: {
        properties: {
          type: "set_check" as const,
          mode: "equal" as const,
          value_expression: null,
          expected_set: ["a", "b"],
          reference_key: null,
        },
      },
    })
    const tagInput = container.querySelector(
      '[data-testid="tag-input-set_check_expected_set"]',
    )
    expect(tagInput).toBeTruthy()
  })
})

// A stale reference_key must never survive into saved properties while the
// reference data UI is hidden: the source is forced to the fixed value, so the
// key would be invisible in the form yet still persisted on the config.
describe("Reference data UI is hidden: getProperties drops any stale key", () => {
  it("ExactMatchForm nulls a pre-set reference_key", () => {
    const { component } = render(ExactMatchForm, {
      props: {
        properties: {
          type: "exact_match" as const,
          case_sensitive: true,
          value_expression: null,
          expected_value: "hello",
          reference_key: "expected_answer",
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const props = (component as any).getProperties()
    expect(props.reference_key).toBeNull()
    expect(props.expected_value).toBe("hello")
  })

  it("ContainsForm nulls a pre-set reference_key", () => {
    const { component } = render(ContainsForm, {
      props: {
        properties: {
          type: "contains" as const,
          case_sensitive: true,
          mode: "must_contain" as const,
          value_expression: null,
          substring: "success",
          reference_key: "expected_answer",
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const props = (component as any).getProperties()
    expect(props.reference_key).toBeNull()
    expect(props.substring).toBe("success")
  })

  it("SetCheckForm nulls a pre-set reference_key", () => {
    const { component } = render(SetCheckForm, {
      props: {
        properties: {
          type: "set_check" as const,
          mode: "equal" as const,
          value_expression: null,
          expected_set: ["a", "b"],
          reference_key: "expected_answer",
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const props = (component as any).getProperties()
    expect(props.reference_key).toBeNull()
    expect(props.expected_set).toEqual(["a", "b"])
  })
})

describe("Reference data UI is hidden", () => {
  it("ExactMatchForm renders no source radio group and no reference copy", () => {
    const { container } = render(ExactMatchForm, {
      props: {
        properties: {
          type: "exact_match" as const,
          case_sensitive: true,
          value_expression: null,
          expected_value: "hello",
          reference_key: null,
        },
      },
    })
    expect(
      container.querySelectorAll(
        'input[type="radio"][name="exact_match_source"]',
      ),
    ).toHaveLength(0)
    expect(
      container.querySelector('[data-testid="radio-group-exact_match_source"]'),
    ).toBeNull()
    expect(
      container.querySelector(
        '[data-testid="form-element-exact_match_reference_key"]',
      ),
    ).toBeNull()
    expect(container.textContent || "").not.toMatch(/reference/i)
  })

  it("ContainsForm renders no source radio group and no reference copy", () => {
    const { container } = render(ContainsForm, {
      props: {
        properties: {
          type: "contains" as const,
          case_sensitive: true,
          mode: "must_contain" as const,
          value_expression: null,
          substring: "hello",
          reference_key: null,
        },
      },
    })
    expect(
      container.querySelectorAll('input[type="radio"][name="contains_source"]'),
    ).toHaveLength(0)
    expect(
      container.querySelector('[data-testid="radio-group-contains_source"]'),
    ).toBeNull()
    expect(
      container.querySelector(
        '[data-testid="form-element-contains_reference_key"]',
      ),
    ).toBeNull()
    expect(container.textContent || "").not.toMatch(/reference/i)
  })

  it("SetCheckForm renders no source radio group and no reference copy", () => {
    const { container } = render(SetCheckForm, {
      props: {
        properties: {
          type: "set_check" as const,
          mode: "equal" as const,
          value_expression: null,
          expected_set: ["a"],
          reference_key: null,
        },
      },
    })
    expect(
      container.querySelectorAll(
        'input[type="radio"][name="set_check_source"]',
      ),
    ).toHaveLength(0)
    expect(
      container.querySelector('[data-testid="radio-group-set_check_source"]'),
    ).toBeNull()
    expect(
      container.querySelector(
        '[data-testid="form-element-set_check_reference_key"]',
      ),
    ).toBeNull()
    expect(container.textContent || "").not.toMatch(/reference/i)
  })
})

describe("ToolCallCheckForm", () => {
  it("validate returns error when no tools are defined", () => {
    const { component } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBe(
      "At least one expected tool must be defined.",
    )
  })

  it("validate returns error when a tool has empty name, with tool index", () => {
    const { component } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBe(
      "Expected Tool #1 is missing a name.",
    )
  })

  it("validate error identifies the correct tool index", () => {
    const { component } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [
            { tool_name: "search", expected_args: null },
            { tool_name: "", expected_args: null },
          ],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBe(
      "Expected Tool #2 is missing a name.",
    )
  })

  it("validate returns null when tools are properly defined", () => {
    const { component } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search_web", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBeNull()
  })

  it("an argument value with no name blocks save and shows an inline error", async () => {
    const { component, container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [
            {
              tool_name: "search",
              expected_args: { q: { value: "x", match_mode: "exact" } },
            },
          ],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })

    // Clear the arg name while keeping its value. Previously this row was
    // silently dropped; now it must error and gate save.
    const nameInput = container.querySelector(
      '[data-testid="input-arg_name_0_0"]',
    ) as HTMLInputElement
    await fireEvent.input(nameInput, { target: { value: "" } })
    await tick()

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBe(
      "Expected Tool #1, argument #1: Add a name, or clear the value.",
    )
    const nameField = container.querySelector(
      '[data-testid="form-element-arg_name_0_0"]',
    )
    expect(nameField?.getAttribute("data-error-message")).toContain(
      "Add a name",
    )
  })

  it("an invalid-JSON argument value blocks save and shows an inline error", async () => {
    const { component, container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [
            {
              tool_name: "search",
              expected_args: { q: { value: "x", match_mode: "exact" } },
            },
          ],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })

    // Previously invalid JSON was silently coerced to a raw string.
    const valueInput = container.querySelector(
      '[data-testid="input-arg_value_0_0"]',
    ) as HTMLInputElement
    await fireEvent.input(valueInput, { target: { value: "{unclosed" } })
    await tick()

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBe(
      "Expected Tool #1, argument #1: Must be valid JSON.",
    )
    const valueField = container.querySelector(
      '[data-testid="form-element-arg_value_0_0"]',
    )
    expect(valueField?.getAttribute("data-error-message")).toContain(
      "valid JSON",
    )
  })

  it("a named, valid-JSON argument passes validation and parses on save", () => {
    const { component } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [
            {
              tool_name: "search",
              expected_args: { limit: { value: 5, match_mode: "exact" } },
            },
          ],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((component as any).validate()).toBeNull()
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const props = (component as any).getProperties()
    expect(props.expected_tools[0].expected_args.limit.value).toBe(5)
  })
})

// Redesigned-form coverage: relabels, tooltips, progressive disclosure, section
// structure, and the reference_key validation path. Pre-existing contract tests
// (getProperties shape, validate pass/fail for expected_value/substring sources,
// radio switching) are NOT duplicated here.

describe("Relabeled fields and Jinja tooltips", () => {
  it("ExactMatch has 'Jinja Expression' label when custom value is set", () => {
    const { container } = render(ExactMatchForm, {
      props: {
        properties: {
          type: "exact_match" as const,
          case_sensitive: true,
          value_expression: "final_message | upper",
          expected_value: "hello",
          reference_key: null,
        },
      },
    })
    const formElement = container.querySelector(
      '[data-testid="form-element-exact_match_value_expression"]',
    )
    expect(formElement).toBeTruthy()
    expect(formElement?.getAttribute("data-label")).toBe("Jinja Expression")
  })

  it("PatternMatch has 'Jinja Expression' label when custom value is set", () => {
    const { container } = render(PatternMatchForm, {
      props: {
        properties: {
          type: "pattern_match" as const,
          pattern: "test",
          mode: "must_match" as const,
          value_expression: "final_message | upper",
        },
      },
    })
    const formElement = container.querySelector(
      '[data-testid="form-element-pattern_match_value_expression"]',
    )
    expect(formElement).toBeTruthy()
    expect(formElement?.getAttribute("data-label")).toBe("Jinja Expression")
  })

  it("Contains has 'Jinja Expression' label when custom value is set", () => {
    const { container } = render(ContainsForm, {
      props: {
        properties: {
          type: "contains" as const,
          case_sensitive: true,
          mode: "must_contain" as const,
          value_expression: "final_message | upper",
          substring: "hello",
          reference_key: null,
        },
      },
    })
    const formElement = container.querySelector(
      '[data-testid="form-element-contains_value_expression"]',
    )
    expect(formElement).toBeTruthy()
    expect(formElement?.getAttribute("data-label")).toBe("Jinja Expression")
  })
})

describe("Progressive disclosure and section structure", () => {
  it("ExactMatch shows only expected_value input when fixed value selected", () => {
    const { container } = render(ExactMatchForm, {
      props: {
        properties: {
          type: "exact_match" as const,
          case_sensitive: true,
          value_expression: null,
          expected_value: "hello",
          reference_key: null,
        },
      },
    })
    expect(
      container.querySelector(
        '[data-testid="form-element-exact_match_expected_value"]',
      ),
    ).toBeTruthy()
    expect(
      container.querySelector(
        '[data-testid="form-element-exact_match_reference_key"]',
      ),
    ).toBeNull()
    expect(
      container.querySelector('[data-testid="radio-group-exact_match_source"]'),
    ).toBeNull()
    expect(
      container.querySelector(
        '[data-testid="form-element-exact_match_source"]',
      ),
    ).toBeNull()
  })

  it("Contains shows the substring input and no source radio group", () => {
    const { container } = render(ContainsForm, {
      props: {
        properties: {
          type: "contains" as const,
          case_sensitive: true,
          mode: "must_contain" as const,
          value_expression: null,
          substring: "hello",
          reference_key: null,
        },
      },
    })
    expect(
      container.querySelector(
        '[data-testid="form-element-contains_substring"]',
      ),
    ).toBeTruthy()
    expect(
      container.querySelector(
        '[data-testid="form-element-contains_reference_key"]',
      ),
    ).toBeNull()
    expect(
      container.querySelector('[data-testid="radio-group-contains_source"]'),
    ).toBeNull()
    expect(
      container.querySelector('[data-testid="form-element-contains_source"]'),
    ).toBeNull()
  })

  it("PatternMatch renders match mode radio group and pattern field", () => {
    const { container } = render(PatternMatchForm, {
      props: {
        properties: {
          type: "pattern_match" as const,
          pattern: "test",
          mode: "must_match" as const,
          value_expression: null,
        },
      },
    })
    expect(
      container.querySelector('[data-testid="radio-group-pattern_match_mode"]'),
    ).toBeTruthy()
    expect(
      container.querySelector('[data-testid="pattern-match-pattern-section"]'),
    ).toBeTruthy()
    expect(
      container.querySelector(
        '[data-testid="form-element-pattern_match_mode"]',
      ),
    ).toBeTruthy()
  })

  it("Contains renders match mode radio group", () => {
    const { container } = render(ContainsForm, {
      props: {
        properties: {
          type: "contains" as const,
          case_sensitive: true,
          mode: "must_contain" as const,
          value_expression: null,
          substring: "hello",
          reference_key: null,
        },
      },
    })
    expect(
      container.querySelector('[data-testid="radio-group-contains_mode"]'),
    ).toBeTruthy()
    expect(
      container.querySelector('[data-testid="form-element-contains_mode"]'),
    ).toBeTruthy()
  })

  it("PatternMatch has regex tooltip", () => {
    const { container } = render(PatternMatchForm, {
      props: {
        properties: {
          type: "pattern_match" as const,
          pattern: "test",
          mode: "must_match" as const,
          value_expression: null,
        },
      },
    })
    const formElement = container.querySelector(
      '[data-testid="form-element-pattern_match_pattern"]',
    )
    expect(formElement?.getAttribute("data-info-description")).toContain(
      "regular expression",
    )
  })
})

// Redesigned set_check, tool_call_check, step_count_check forms.
// Tests verify new section structure, radio groups, progressive disclosure,
// and that getProperties()/validate() contracts are preserved exactly.

describe("SetCheckForm section structure and progressive disclosure", () => {
  it("renders Comparison Mode radio with label", () => {
    const { container } = render(SetCheckForm, {
      props: {
        properties: {
          type: "set_check" as const,
          mode: "equal" as const,
          value_expression: null,
          expected_set: ["a"],
          reference_key: null,
        },
      },
    })
    const el = container.querySelector(
      '[data-testid="form-element-set_check_mode"]',
    )
    expect(el).toBeTruthy()
    expect(el?.getAttribute("data-label")).toBe("Comparison Mode")
    expect(
      container.querySelector('[data-testid="radio-group-set_check_mode"]'),
    ).toBeTruthy()
  })

  it("shows tag input and no reference key input", () => {
    const { container } = render(SetCheckForm, {
      props: {
        properties: {
          type: "set_check" as const,
          mode: "equal" as const,
          value_expression: null,
          expected_set: ["a", "b"],
          reference_key: null,
        },
      },
    })
    expect(
      container.querySelector(
        '[data-testid="tag-input-set_check_expected_set"]',
      ),
    ).toBeTruthy()
    expect(
      container.querySelector(
        '[data-testid="form-element-set_check_reference_key"]',
      ),
    ).toBeNull()
  })

  it("has 'Jinja Expression' label when custom value is set", () => {
    const { container } = render(SetCheckForm, {
      props: {
        properties: {
          type: "set_check" as const,
          mode: "equal" as const,
          value_expression: "final_message | upper",
          expected_set: ["a"],
          reference_key: null,
        },
      },
    })
    const formElement = container.querySelector(
      '[data-testid="form-element-set_check_value_expression"]',
    )
    expect(formElement).toBeTruthy()
    expect(formElement?.getAttribute("data-label")).toBe("Jinja Expression")
  })

  it("Jinja Expression description includes 'Must be an array.' via extra_description", () => {
    const { container } = render(SetCheckForm, {
      props: {
        properties: {
          type: "set_check" as const,
          mode: "equal" as const,
          value_expression: "final_message | upper",
          expected_set: ["a"],
          reference_key: null,
        },
      },
    })
    const formElement = container.querySelector(
      '[data-testid="form-element-set_check_value_expression"]',
    )
    const desc = formElement?.getAttribute("data-description") || ""
    expect(desc).toContain("Jinja syntax")
    expect(desc).toContain("Must be an array.")
  })

  it("renders all three comparison mode options", () => {
    const { getAllByText } = render(SetCheckForm, {
      props: {
        properties: {
          type: "set_check" as const,
          mode: "equal" as const,
          value_expression: null,
          expected_set: ["a"],
          reference_key: null,
        },
      },
    })
    expect(getAllByText("Equal").length).toBeGreaterThan(0)
    expect(getAllByText("Subset").length).toBeGreaterThan(0)
    expect(getAllByText("Superset").length).toBeGreaterThan(0)
  })

  it("getProperties nulls reference_key when expected_set source selected", () => {
    const { component } = render(SetCheckForm, {
      props: {
        properties: {
          type: "set_check" as const,
          mode: "equal" as const,
          value_expression: null,
          expected_set: ["a", "b"],
          reference_key: null,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const props = (component as any).getProperties()
    expect(props.reference_key).toBeNull()
    expect(props.expected_set).toEqual(["a", "b"])
  })
})

describe("ToolCallCheckForm section structure and progressive disclosure", () => {
  it("renders Match Mode radio with label", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    const el = container.querySelector(
      '[data-testid="form-element-tool_call_check_match_mode"]',
    )
    expect(el).toBeTruthy()
    expect(el?.getAttribute("data-label")).toBe("Match Mode")
    expect(
      container.querySelector(
        '[data-testid="radio-group-tool_call_check_match_mode"]',
      ),
    ).toBeTruthy()
  })

  it("renders all four match mode options", () => {
    const { getAllByText } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    expect(getAllByText("Any").length).toBeGreaterThan(0)
    expect(getAllByText("All (any order)").length).toBeGreaterThan(0)
    expect(getAllByText("Ordered (in list order)").length).toBeGreaterThan(0)
    expect(getAllByText("Never").length).toBeGreaterThan(0)
  })

  it("shows Unlisted Tool Calls radio when match_mode is not 'never'", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    expect(
      container.querySelector(
        '[data-testid="form-element-tool_call_check_on_unexpected"]',
      ),
    ).toBeTruthy()
    expect(
      container.querySelector(
        '[data-testid="radio-group-tool_call_check_on_unexpected"]',
      ),
    ).toBeTruthy()
  })

  it("hides Unlisted Tool Calls radio when match_mode is 'never'", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "never" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    expect(
      container.querySelector(
        '[data-testid="form-element-tool_call_check_on_unexpected"]',
      ),
    ).toBeNull()
  })

  it("renders Expected Tools header_only FormElement", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    const header = container.querySelector(
      '[data-testid="form-element-tool_call_expected_tools_header"]',
    )
    expect(header).toBeTruthy()
    expect(header?.getAttribute("data-label")).toBe("Expected Tools")
  })

  it("getProperties returns current match_mode", () => {
    const { component } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "ordered" as const,
          on_unexpected_tools: "fail" as const,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const props = (component as any).getProperties()
    expect(props.match_mode).toBe("ordered")
    expect(props.on_unexpected_tools).toBe("fail")
  })

  it("getProperties syncs arg rows to properties", async () => {
    const initProps = {
      type: "tool_call_check" as const,
      expected_tools: [
        {
          tool_name: "search",
          expected_args: {
            query: { value: "hello world", match_mode: "exact" as const },
          },
        },
      ],
      match_mode: "all" as const,
      on_unexpected_tools: "ignore" as const,
    }
    const { component } = render(ToolCallCheckForm, {
      props: { properties: initProps },
    })
    // Null out expected_args on properties directly, diverging from
    // the internal arg_rows (which still hold the deserialized "query" row).
    // If sync_args_to_properties is unwired, getProperties returns this null.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ;(initProps.expected_tools[0] as any).expected_args = null
    await tick()

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const props = (component as any).getProperties()
    // sync_args_to_properties must have restored args from internal arg_rows
    expect(props.expected_tools[0].expected_args).not.toBeNull()
    expect(props.expected_tools[0].expected_args.query).toEqual({
      value: "hello world",
      match_mode: "exact",
    })
  })
})

describe("StepCountCheckForm section structure and progressive disclosure", () => {
  it("renders What to Count radio with label", () => {
    const { container } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "tool_calls" as const,
          min_count: 1,
          max_count: null,
        },
      },
    })
    const el = container.querySelector(
      '[data-testid="form-element-step_count_check_count_type"]',
    )
    expect(el).toBeTruthy()
    expect(el?.getAttribute("data-label")).toBe("What to Count")
    expect(
      container.querySelector(
        '[data-testid="radio-group-step_count_check_count_type"]',
      ),
    ).toBeTruthy()
  })

  it("renders all three count type options with descriptions", () => {
    const { getAllByText } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "tool_calls" as const,
          min_count: null,
          max_count: null,
        },
      },
    })
    expect(getAllByText("Tool calls").length).toBeGreaterThan(0)
    expect(getAllByText("Model responses").length).toBeGreaterThan(0)
    expect(getAllByText("Conversation turns").length).toBeGreaterThan(0)
  })

  it("renders Bounds header_only FormElement", () => {
    const { container } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "tool_calls" as const,
          min_count: null,
          max_count: null,
        },
      },
    })
    const header = container.querySelector(
      '[data-testid="form-element-step_count_bounds_header"]',
    )
    expect(header).toBeTruthy()
    expect(header?.getAttribute("data-label")).toBe("Bounds")
  })

  it("getProperties returns correct count_type", () => {
    const { component } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "model_responses" as const,
          min_count: 2,
          max_count: 10,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const props = (component as any).getProperties()
    expect(props.count_type).toBe("model_responses")
    expect(props.min_count).toBe(2)
    expect(props.max_count).toBe(10)
  })

  it("getProperties returns properties directly (not a copy with nulled fields)", () => {
    const { component } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "turns" as const,
          min_count: 5,
          max_count: null,
        },
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const props = (component as any).getProperties()
    expect(props.type).toBe("step_count_check")
    expect(props.count_type).toBe("turns")
    expect(props.min_count).toBe(5)
    expect(props.max_count).toBeNull()
  })

  it("renders min and max count form elements", () => {
    const { container } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "tool_calls" as const,
          min_count: null,
          max_count: null,
        },
      },
    })
    expect(
      container.querySelector(
        '[data-testid="form-element-step_count_check_min"]',
      ),
    ).toBeTruthy()
    expect(
      container.querySelector(
        '[data-testid="form-element-step_count_check_max"]',
      ),
    ).toBeTruthy()
  })

  it("renders count type descriptions", () => {
    const { getAllByText } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "tool_calls" as const,
          min_count: null,
          max_count: null,
        },
      },
    })
    expect(
      getAllByText(
        "Each individual tool invocation. A single response that calls three tools counts as three.",
      ).length,
    ).toBeGreaterThan(0)
    expect(
      getAllByText(
        "Each message the model generates, including intermediate messages that only call tools.",
      ).length,
    ).toBeGreaterThan(0)
    expect(
      getAllByText(
        "Each user message and everything the agent does to answer it. Tool calls and model responses within a turn don't add to the count.",
      ).length,
    ).toBeGreaterThan(0)
  })
})

// UI polish tests: hidden duplicate labels, indent wrappers, moved controls,
// placeholders, renamed sections, and regex tooltip content.

describe("Standard controls: visible labels (no hidden labels)", () => {
  it("OutputValueField shows visible 'Jinja Expression' label when custom", () => {
    const { container } = render(ExactMatchForm, {
      props: {
        properties: {
          type: "exact_match" as const,
          case_sensitive: true,
          value_expression: "final_message | upper",
          expected_value: "hello",
          reference_key: null,
        },
      },
    })
    const valueExprField = container.querySelector(
      '[data-testid="form-element-exact_match_value_expression"]',
    )
    expect(valueExprField?.getAttribute("data-hide-label")).toBe("false")
    expect(valueExprField?.getAttribute("data-label")).toBe("Jinja Expression")
  })

  it("ExactMatch expected_value field shows visible 'Expected Value' label", () => {
    const { container } = render(ExactMatchForm, {
      props: {
        properties: {
          type: "exact_match" as const,
          case_sensitive: true,
          value_expression: null,
          expected_value: "hello",
          reference_key: null,
        },
      },
    })
    const field = container.querySelector(
      '[data-testid="form-element-exact_match_expected_value"]',
    )
    expect(field?.getAttribute("data-hide-label")).toBe("false")
    expect(field?.getAttribute("data-label")).toBe("Expected Value")
    expect(field?.getAttribute("data-description")).toBe(
      "The exact value the output should match.",
    )
  })

  it("PatternMatch regex field shows visible 'Expected Pattern (Regex)' label", () => {
    const { container } = render(PatternMatchForm, {
      props: {
        properties: {
          type: "pattern_match" as const,
          pattern: "test",
          mode: "must_match" as const,
          value_expression: null,
        },
      },
    })
    const field = container.querySelector(
      '[data-testid="form-element-pattern_match_pattern"]',
    )
    expect(field?.getAttribute("data-hide-label")).toBe("false")
    expect(field?.getAttribute("data-label")).toBe("Expected Pattern (Regex)")
  })

  it("Contains substring field shows visible 'Expected Substring' label", () => {
    const { container } = render(ContainsForm, {
      props: {
        properties: {
          type: "contains" as const,
          case_sensitive: true,
          mode: "must_contain" as const,
          value_expression: null,
          substring: "hello",
          reference_key: null,
        },
      },
    })
    const field = container.querySelector(
      '[data-testid="form-element-contains_substring"]',
    )
    expect(field?.getAttribute("data-hide-label")).toBe("false")
    expect(field?.getAttribute("data-label")).toBe("Expected Substring")
    expect(field?.getAttribute("data-description")).toBe(
      "The text to search for in the output.",
    )
  })
})

describe("UI polish: case-sensitive moved out of OutputValueField", () => {
  it("ExactMatch has case-sensitive checkbox as direct child of form", () => {
    const { container } = render(ExactMatchForm, {
      props: {
        properties: {
          type: "exact_match" as const,
          case_sensitive: true,
          value_expression: null,
          expected_value: "hello",
          reference_key: null,
        },
      },
    })
    const caseSensitive = container.querySelector(
      '[data-testid="form-element-exact_match_case_sensitive"]',
    )
    expect(caseSensitive).toBeTruthy()
    expect(caseSensitive?.getAttribute("data-label")).toBe("Case Sensitive")
  })

  it("ExactMatch case-sensitive is separate from OutputValueField", () => {
    const { container } = render(ExactMatchForm, {
      props: {
        properties: {
          type: "exact_match" as const,
          case_sensitive: true,
          value_expression: null,
          expected_value: "hello",
          reference_key: null,
        },
      },
    })
    const caseSensitive = container.querySelector(
      '[data-testid="form-element-exact_match_case_sensitive"]',
    )
    expect(caseSensitive).toBeTruthy()
  })

  it("Contains has case-sensitive checkbox as direct child of form", () => {
    const { container } = render(ContainsForm, {
      props: {
        properties: {
          type: "contains" as const,
          case_sensitive: true,
          mode: "must_contain" as const,
          value_expression: null,
          substring: "hello",
          reference_key: null,
        },
      },
    })
    const caseSensitive = container.querySelector(
      '[data-testid="form-element-contains_case_sensitive"]',
    )
    expect(caseSensitive).toBeTruthy()
    expect(caseSensitive?.getAttribute("data-label")).toBe("Case Sensitive")
  })

  it("Contains case-sensitive is separate from OutputValueField", () => {
    const { container } = render(ContainsForm, {
      props: {
        properties: {
          type: "contains" as const,
          case_sensitive: true,
          mode: "must_contain" as const,
          value_expression: null,
          substring: "hello",
          reference_key: null,
        },
      },
    })
    const caseSensitive = container.querySelector(
      '[data-testid="form-element-contains_case_sensitive"]',
    )
    expect(caseSensitive).toBeTruthy()
  })
})

describe("UI polish: placeholders on inputs", () => {
  it("ExactMatch expected_value has placeholder", () => {
    const { container } = render(ExactMatchForm, {
      props: {
        properties: {
          type: "exact_match" as const,
          case_sensitive: true,
          value_expression: null,
          expected_value: "hello",
          reference_key: null,
        },
      },
    })
    const field = container.querySelector(
      '[data-testid="form-element-exact_match_expected_value"]',
    )
    expect(field?.getAttribute("data-placeholder")).toBe("e.g. yes")
  })

  it("PatternMatch regex field has placeholder", () => {
    const { container } = render(PatternMatchForm, {
      props: {
        properties: {
          type: "pattern_match" as const,
          pattern: "",
          mode: "must_match" as const,
          value_expression: null,
        },
      },
    })
    const field = container.querySelector(
      '[data-testid="form-element-pattern_match_pattern"]',
    )
    expect(field?.getAttribute("data-placeholder")).toBe("e.g. ^(yes|no)$")
  })

  it("Contains substring has placeholder", () => {
    const { container } = render(ContainsForm, {
      props: {
        properties: {
          type: "contains" as const,
          case_sensitive: true,
          mode: "must_contain" as const,
          value_expression: null,
          substring: "hello",
          reference_key: null,
        },
      },
    })
    const field = container.querySelector(
      '[data-testid="form-element-contains_substring"]',
    )
    expect(field?.getAttribute("data-placeholder")).toBe("e.g. success")
  })
})

describe("Standard controls: PatternMatch uses single FormElement with 'Expected Pattern (Regex)' label", () => {
  it("Pattern FormElement has label 'Expected Pattern (Regex)' (no duplicate section title)", () => {
    const { container } = render(PatternMatchForm, {
      props: {
        properties: {
          type: "pattern_match" as const,
          pattern: "test",
          mode: "must_match" as const,
          value_expression: null,
        },
      },
    })
    const section = container.querySelector(
      '[data-testid="pattern-match-pattern-section"]',
    )
    expect(section).toBeTruthy()
    const formElement = section?.querySelector(
      '[data-testid="form-element-pattern_match_pattern"]',
    )
    expect(formElement?.getAttribute("data-label")).toBe(
      "Expected Pattern (Regex)",
    )
    const heading = section?.querySelector("h3")
    expect(heading).toBeNull()
  })
})

describe("UI polish: regex tooltip is educational", () => {
  it("PatternMatch regex tooltip explains what regex is with an example", () => {
    const { container } = render(PatternMatchForm, {
      props: {
        properties: {
          type: "pattern_match" as const,
          pattern: "test",
          mode: "must_match" as const,
          value_expression: null,
        },
      },
    })
    const field = container.querySelector(
      '[data-testid="form-element-pattern_match_pattern"]',
    )
    const tooltip = field?.getAttribute("data-info-description") || ""
    expect(tooltip).toContain("regular expression")
    expect(tooltip).toContain("regex")
    expect(tooltip).toContain("^yes$")
  })
})

describe("PatternMatch regex validation on focusout", () => {
  it("shows an inline regex error after the field loses focus", async () => {
    const { container } = render(PatternMatchForm, {
      props: {
        properties: {
          type: "pattern_match" as const,
          pattern: "[invalid(",
          mode: "must_match" as const,
          value_expression: null,
        },
      },
    })
    const field = container.querySelector(
      '[data-testid="form-element-pattern_match_pattern"]',
    )
    // Untouched: no error yet.
    expect(field?.getAttribute("data-error-message")).toBe("")

    // A real browser fires a bubbling focusout from the input on blur, which
    // reaches the section wrapper's on:focusout handler.
    const input = container.querySelector(
      '[data-testid="input-pattern_match_pattern"]',
    ) as HTMLInputElement
    await fireEvent.focusOut(input)
    await tick()

    expect(field?.getAttribute("data-error-message")).toContain("Invalid regex")
  })
})

describe("Standard controls: OutputValueField renders labeled dropdown", () => {
  it("OutputValueField renders a labeled fancy_select dropdown", () => {
    const { container } = render(ExactMatchForm, {
      props: {
        properties: {
          type: "exact_match" as const,
          case_sensitive: true,
          value_expression: null,
          expected_value: "hello",
          reference_key: null,
        },
      },
    })
    const dropdown = container.querySelector(
      '[data-testid="form-element-exact_match_output_source"]',
    )
    expect(dropdown).toBeTruthy()
    expect(dropdown?.getAttribute("data-type")).toBe("fancy_select")
    expect(dropdown?.getAttribute("data-label")).toBe("Output to Check")
    expect(dropdown?.getAttribute("data-hide-label")).toBe("false")
  })
})

describe("Standard controls: description and tooltip on visible-label fields", () => {
  it("OutputValueField has Jinja description with visible label when custom", () => {
    const { container } = render(ExactMatchForm, {
      props: {
        properties: {
          type: "exact_match" as const,
          case_sensitive: true,
          value_expression: "final_message | upper",
          expected_value: "hello",
          reference_key: null,
        },
      },
    })
    const field = container.querySelector(
      '[data-testid="form-element-exact_match_value_expression"]',
    )
    expect(field?.getAttribute("data-hide-label")).toBe("false")
    expect(field?.getAttribute("data-description")).toContain("Jinja syntax")
    expect(field?.getAttribute("data-label")).toBe("Jinja Expression")
  })

  it("PatternMatch regex field has description and tooltip with visible label", () => {
    const { container } = render(PatternMatchForm, {
      props: {
        properties: {
          type: "pattern_match" as const,
          pattern: "test",
          mode: "must_match" as const,
          value_expression: null,
        },
      },
    })
    const field = container.querySelector(
      '[data-testid="form-element-pattern_match_pattern"]',
    )
    expect(field?.getAttribute("data-hide-label")).toBe("false")
    expect(field?.getAttribute("data-description")).toContain("pattern")
    expect(field?.getAttribute("data-info-description")).toContain(
      "regular expression",
    )
  })
})

// ──────────────────────────────────────────────────────────────────
// set_check, tool_call_check, step_count_check UI polish
// ──────────────────────────────────────────────────────────────────

describe("SetCheckForm UI polish", () => {
  it("tag input has a visible 'Expected Values' header", () => {
    const { container } = render(SetCheckForm, {
      props: {
        properties: {
          type: "set_check" as const,
          mode: "equal" as const,
          value_expression: null,
          expected_set: ["a"],
          reference_key: null,
        },
      },
    })
    const header = container.querySelector(
      '[data-testid="form-element-set_check_expected_values_header"]',
    )
    expect(header).toBeTruthy()
    expect(header?.getAttribute("data-type")).toBe("header_only")
    expect(header?.getAttribute("data-label")).toBe("Expected Values")
    expect(header?.getAttribute("data-description")).toBe(
      "Define what the output should contain. Add items by typing and pressing Enter or comma.",
    )
  })

  it("comparison mode descriptions use plain language", () => {
    const { getAllByText } = render(SetCheckForm, {
      props: {
        properties: {
          type: "set_check" as const,
          mode: "equal" as const,
          value_expression: null,
          expected_set: ["a"],
          reference_key: null,
        },
      },
    })
    expect(
      getAllByText(
        "The output must contain exactly the expected values, with no extras and nothing missing.",
      ).length,
    ).toBeGreaterThan(0)
    expect(
      getAllByText(
        "Every output value must appear in the expected values (extras in expected are OK).",
      ).length,
    ).toBeGreaterThan(0)
    expect(
      getAllByText(
        "Every expected value must appear in the output (extra output values are OK).",
      ).length,
    ).toBeGreaterThan(0)
  })

  it("tag input is not wrapped in an indent container", () => {
    const { container } = render(SetCheckForm, {
      props: {
        properties: {
          type: "set_check" as const,
          mode: "equal" as const,
          value_expression: null,
          expected_set: ["a"],
          reference_key: null,
        },
      },
    })
    const tagInput = container.querySelector(
      '[data-testid="tag-input-set_check_expected_set"]',
    )
    expect(tagInput).toBeTruthy()
    expect(tagInput?.closest(".ml-4.border-l.pl-4")).toBeNull()
  })
})

describe("ToolCallCheckForm UI polish", () => {
  it("Expected Tools header appears before Match Mode in DOM", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    const sections = container.querySelectorAll("[data-testid]")
    const sectionIds = Array.from(sections).map((s) =>
      s.getAttribute("data-testid"),
    )
    const toolsIdx = sectionIds.indexOf(
      "form-element-tool_call_expected_tools_header",
    )
    const matchIdx = sectionIds.indexOf(
      "form-element-tool_call_check_match_mode",
    )
    expect(toolsIdx).toBeGreaterThanOrEqual(0)
    expect(matchIdx).toBeGreaterThanOrEqual(0)
    expect(toolsIdx).toBeLessThan(matchIdx)
  })

  it("renders 'Expected Tools' header label exactly once (no duplication)", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    const header = container.querySelector(
      '[data-testid="form-element-tool_call_expected_tools_header"]',
    )
    expect(header).toBeTruthy()
    expect(header?.getAttribute("data-label")).toBe("Expected Tools")
    const allWithLabel = container.querySelectorAll(
      '[data-label="Expected Tools"]',
    )
    expect(allWithLabel).toHaveLength(1)
  })

  it("arg row header says 'Comparison' not 'Match'", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [
            {
              tool_name: "search",
              expected_args: {
                query: { value: "test", match_mode: "exact" as const },
              },
            },
          ],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    const argMatchField = container.querySelector(
      '[data-testid="form-element-arg_match_0_0"]',
    )
    expect(argMatchField?.getAttribute("data-label")).toBe("Comparison")
  })

  it("tool fields are nested with indent pattern", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    const toolNameField = container.querySelector(
      '[data-testid="form-element-tool_name_0"]',
    )
    const indent = toolNameField?.closest(".ml-4.border-l.pl-4")
    expect(indent).toBeTruthy()
  })

  it("match mode descriptions distinguish 'any order' vs 'in list order'", () => {
    const { getAllByText } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    expect(getAllByText("All (any order)").length).toBeGreaterThan(0)
    expect(getAllByText("Ordered (in list order)").length).toBeGreaterThan(0)
  })

  it("description contextualizes for 'never' mode", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "never" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    const header = container.querySelector(
      '[data-testid="form-element-tool_call_expected_tools_header"]',
    )
    expect(header?.getAttribute("data-description")).toContain("must NOT call")
  })

  it("description for non-never mode says 'expected to call'", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    const header = container.querySelector(
      '[data-testid="form-element-tool_call_expected_tools_header"]',
    )
    expect(header?.getAttribute("data-description")).toContain(
      "expected to call",
    )
  })

  it("label is 'Unlisted Tool Calls' not 'On Unexpected Tools'", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    const el = container.querySelector(
      '[data-testid="form-element-tool_call_check_on_unexpected"]',
    )
    expect(el?.getAttribute("data-label")).toBe("Unlisted Tool Calls")
  })

  it("Expected Arguments collapse has a description", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    const collapse = container.querySelector('[data-testid="collapse-stub"]')
    expect(collapse?.getAttribute("data-title")).toBe("Expected Arguments")
  })

  it("arg value field has info_description about JSON format", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [
            {
              tool_name: "search",
              expected_args: {
                query: { value: "test", match_mode: "exact" as const },
              },
            },
          ],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    const argValueField = container.querySelector(
      '[data-testid="form-element-arg_value_0_0"]',
    )
    const tooltip = argValueField?.getAttribute("data-info-description") || ""
    expect(tooltip).toContain("JSON")
    expect(tooltip).toContain("quoted")
  })
})

describe("ToolCallCheckForm arg remove icon button", () => {
  it("renders a remove button with aria-label 'Remove argument'", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [
            {
              tool_name: "search",
              expected_args: {
                query: { value: "test", match_mode: "exact" as const },
              },
            },
          ],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    const removeBtn = container.querySelector(
      'button[aria-label="Remove argument"]',
    )
    expect(removeBtn).toBeTruthy()
    expect(removeBtn?.querySelector("svg")).toBeTruthy()
  })

  it("clicking the remove icon button removes the arg row", async () => {
    const { container, component } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [
            {
              tool_name: "search",
              expected_args: {
                query: { value: "test", match_mode: "exact" as const },
                limit: { value: 10, match_mode: "exact" as const },
              },
            },
          ],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    // Should have two arg rows initially
    const removeBtns = container.querySelectorAll(
      'button[aria-label="Remove argument"]',
    )
    expect(removeBtns).toHaveLength(2)

    // Click the first remove button
    await fireEvent.click(removeBtns[0])
    await tick()

    // Should now have one arg row
    const remainingBtns = container.querySelectorAll(
      'button[aria-label="Remove argument"]',
    )
    expect(remainingBtns).toHaveLength(1)

    // getProperties should reflect the removal
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const props = (component as any).getProperties()
    const argKeys = Object.keys(props.expected_tools[0].expected_args || {})
    expect(argKeys).toHaveLength(1)
  })
})

describe("ToolCallCheckForm arg-name placeholder", () => {
  it("arg name input has example placeholder 'e.g. query'", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [
            {
              tool_name: "search",
              expected_args: {
                query: { value: "test", match_mode: "exact" as const },
              },
            },
          ],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    const argNameField = container.querySelector(
      '[data-testid="form-element-arg_name_0_0"]',
    )
    expect(argNameField?.getAttribute("data-placeholder")).toBe("e.g. query")
  })
})

describe("ToolCallCheckForm Tool field copy", () => {
  it("Tool field renders as a dropdown (fancy_select)", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    const toolNameField = container.querySelector(
      '[data-testid="form-element-tool_name_0"]',
    )
    expect(toolNameField?.getAttribute("data-type")).toBe("fancy_select")
    expect(toolNameField?.getAttribute("data-description")).toBe(
      "The tool that should be called.",
    )
  })

  it("Tool field info_description explains the function name differs from the display name", () => {
    const { container } = render(ToolCallCheckForm, {
      props: {
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "search", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })
    const toolNameField = container.querySelector(
      '[data-testid="form-element-tool_name_0"]',
    )
    const tooltip = toolNameField?.getAttribute("data-info-description") || ""
    expect(tooltip).toContain("function name")
    expect(tooltip).toContain("trace")
  })
})

describe("StepCountCheckForm UI polish", () => {
  it("min and max inputs are in a side-by-side flex row", () => {
    const { container } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "tool_calls" as const,
          min_count: null,
          max_count: null,
        },
      },
    })
    const boundsRow = container.querySelector('[data-testid="bounds-row"]')
    expect(boundsRow).toBeTruthy()
    expect(boundsRow?.classList.contains("flex")).toBe(true)
    const minField = boundsRow?.querySelector(
      '[data-testid="form-element-step_count_check_min"]',
    )
    const maxField = boundsRow?.querySelector(
      '[data-testid="form-element-step_count_check_max"]',
    )
    expect(minField).toBeTruthy()
    expect(maxField).toBeTruthy()
  })

  it("bounds error is shown once, not duplicated on both inputs", async () => {
    const { container } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "tool_calls" as const,
          min_count: 10,
          max_count: 5,
        },
      },
    })
    // Before blur, no error is shown (bounds_touched is false)
    expect(
      container.querySelectorAll('[data-testid="bounds-error"]'),
    ).toHaveLength(0)

    // A real browser fires a bubbling focusout from the input when it loses
    // focus; that bubbles up to the wrapper's on:focusout handler. (A plain
    // blur does not bubble, so the wrapper would never see it.)
    const boundsRow = container.querySelector('[data-testid="bounds-row"]')
    const minField = boundsRow?.querySelector(
      '[data-testid="form-element-step_count_check_min"]',
    )
    expect(minField).toBeTruthy()
    await fireEvent.focusOut(minField!)
    await tick()

    // Error should appear exactly once (not on each input individually)
    const errorElements = container.querySelectorAll(
      '[data-testid="bounds-error"]',
    )
    expect(errorElements).toHaveLength(1)
    expect(errorElements[0].textContent).toContain("Minimum must be")
  })

  it("min field has placeholder 'No minimum'", () => {
    const { container } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "tool_calls" as const,
          min_count: null,
          max_count: null,
        },
      },
    })
    const minField = container.querySelector(
      '[data-testid="form-element-step_count_check_min"]',
    )
    expect(minField?.getAttribute("data-placeholder")).toBe("No minimum")
  })

  it("max field has placeholder 'No maximum'", () => {
    const { container } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "tool_calls" as const,
          min_count: null,
          max_count: null,
        },
      },
    })
    const maxField = container.querySelector(
      '[data-testid="form-element-step_count_check_max"]',
    )
    expect(maxField?.getAttribute("data-placeholder")).toBe("No maximum")
  })

  it("min label is 'Minimum' not 'Minimum Count'", () => {
    const { container } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "tool_calls" as const,
          min_count: null,
          max_count: null,
        },
      },
    })
    const minField = container.querySelector(
      '[data-testid="form-element-step_count_check_min"]',
    )
    expect(minField?.getAttribute("data-label")).toBe("Minimum")
  })

  it("max label is 'Maximum' not 'Maximum Count'", () => {
    const { container } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "tool_calls" as const,
          min_count: null,
          max_count: null,
        },
      },
    })
    const maxField = container.querySelector(
      '[data-testid="form-element-step_count_check_max"]',
    )
    expect(maxField?.getAttribute("data-label")).toBe("Maximum")
  })

  it("bounds are nested with indent pattern", () => {
    const { container } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "tool_calls" as const,
          min_count: null,
          max_count: null,
        },
      },
    })
    const boundsRow = container.querySelector('[data-testid="bounds-row"]')
    const indent = boundsRow?.closest(".ml-4.border-l.pl-4")
    expect(indent).toBeTruthy()
  })

  it("turns description explains what a turn spans", () => {
    const { getAllByText } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "turns" as const,
          min_count: null,
          max_count: null,
        },
      },
    })
    expect(
      getAllByText(
        "Each user message and everything the agent does to answer it. Tool calls and model responses within a turn don't add to the count.",
      ).length,
    ).toBeGreaterThan(0)
  })

  it("model_responses description clarifies tool-call-only messages count", () => {
    const { getAllByText } = render(StepCountCheckForm, {
      props: {
        properties: {
          type: "step_count_check" as const,
          count_type: "model_responses" as const,
          min_count: null,
          max_count: null,
        },
      },
    })
    expect(
      getAllByText(
        "Each message the model generates, including intermediate messages that only call tools.",
      ).length,
    ).toBeGreaterThan(0)
  })
})

describe("ToolCallCheckForm tool option filtering", () => {
  const mcp_set: ToolSetApiDescription = {
    type: "mcp",
    set_name: "MCP Server: demo",
    tools: [
      {
        id: "mcp::remote::demo::search",
        name: "Search",
        description: "Search the web",
        function_name: "search",
      },
    ],
  }

  const ai_models_set: ToolSetApiDescription = {
    type: "sandbox_code",
    set_name: "AI Models",
    tools: [
      {
        id: "kiln_tool::llm",
        name: "LLM",
        description: "Call a model",
        function_name: "llm",
      },
      {
        id: "kiln_tool::llm_judge",
        name: "LLM Judge",
        description: "Judge with the eval schema",
        function_name: "llm_judge",
      },
    ],
  }

  it("excludes the sandbox-only built-ins, keeping real agent tools", async () => {
    available_tools.set({ proj_tcc: [ai_models_set, mcp_set] })

    const { container } = render(ToolCallCheckForm, {
      props: {
        project_id: "proj_tcc",
        properties: {
          type: "tool_call_check" as const,
          expected_tools: [{ tool_name: "", expected_args: null }],
          match_mode: "all" as const,
          on_unexpected_tools: "ignore" as const,
        },
      },
    })

    // Let the store subscription fire, then flush reactivity.
    await new Promise((resolve) => setTimeout(resolve, 0))
    await tick()

    // Neither sandbox built-in is an agent tool, so neither can appear in a trace.
    expect(
      container.querySelector('[data-testid="fancy-option-llm"]'),
    ).toBeNull()
    expect(
      container.querySelector('[data-testid="fancy-option-llm_judge"]'),
    ).toBeNull()
    // A real agent tool is still offered.
    expect(
      container.querySelector('[data-testid="fancy-option-search"]'),
    ).not.toBeNull()
  })
})
