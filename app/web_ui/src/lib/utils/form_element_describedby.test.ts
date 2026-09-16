// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest"
import { render, cleanup } from "@testing-library/svelte"
import FormElement from "./form_element.svelte"

afterEach(() => cleanup())

// aria_describedby is additive: a caller that does not pass it must render the
// field exactly as before, with no empty attribute left behind.
describe("form_element — aria_describedby", () => {
  it("leaves the attribute off when no caller names a description", () => {
    for (const inputType of ["input", "textarea", "select"] as const) {
      const { container } = render(FormElement, {
        props: { id: "field", inputType, label: "Label", value: "" },
      })
      const field = container.querySelector("#field")!
      expect(field.hasAttribute("aria-describedby")).toBe(false)
      // The label wiring the control already had is untouched.
      expect(field.getAttribute("aria-label")).toBe("Label")
      cleanup()
    }
  })

  it("names the element when a caller passes one, whatever it renders", () => {
    for (const inputType of ["input", "textarea", "select"] as const) {
      const { container } = render(FormElement, {
        props: {
          id: "field",
          inputType,
          label: "Label",
          value: "",
          aria_describedby: "field-hint",
        },
      })
      expect(
        container.querySelector("#field")!.getAttribute("aria-describedby"),
      ).toBe("field-hint")
      cleanup()
    }
  })
})
