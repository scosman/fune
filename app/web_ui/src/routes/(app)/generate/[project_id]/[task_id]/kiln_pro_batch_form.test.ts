// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from "vitest"
import { render, cleanup, fireEvent } from "@testing-library/svelte"
import { tick } from "svelte"
import KilnProBatchForm from "./kiln_pro_batch_form.svelte"
import KilnProBatchFormInFormContainer from "./__tests__/kiln_pro_batch_form_in_form_container.svelte"

// FormContainer (used by the submit tests below) registers a navigation guard
// on mount, which has no router behind it under jsdom.
vi.mock("$app/navigation", () => ({
  goto: vi.fn(),
  beforeNavigate: vi.fn(),
}))

const GUIDANCE_DESCRIPTION = `This allows you to control the dataset you are generating. For example, "10% of the dataset should be in Spanish."`

afterEach(cleanup)

function setup(props: Record<string, unknown> = {}) {
  const utils = render(KilnProBatchForm, {
    props: { count: 50, guidance: "", ...props },
  })
  const count_input = utils.container.querySelector(
    'input[aria-label="Count"]',
  ) as HTMLInputElement
  const guidance_box = utils.container.querySelector(
    "textarea",
  ) as HTMLTextAreaElement
  return { ...utils, count_input, guidance_box }
}

// The badge FormElement puts in a field's label row when it is optional.
function optional_badge(container: HTMLElement): HTMLElement | null {
  return (
    Array.from(container.querySelectorAll("span")).find(
      (s) => s.textContent?.trim() === "Optional",
    ) ?? null
  )
}

function reset_button(container: HTMLElement): HTMLButtonElement | null {
  return (
    Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent?.trim() === "Reset",
    ) ?? null
  )
}

// Svelte 4 keeps the live value of each prop in the instance context; this is
// what a parent's `bind:` reads back. Lets the tests check outward flow
// without a wrapper component.
function bound_prop<T>(component: unknown, name: string): T {
  const instance = component as {
    $$: { ctx: unknown[]; props: Record<string, number> }
  }
  return instance.$$.ctx[instance.$$.props[name]] as T
}

describe("KilnProBatchForm", () => {
  it("renders the count row with the shipped layout and label", () => {
    const { container, count_input } = setup()
    const row = container.firstElementChild as HTMLElement
    expect(row.className).toBe("flex flex-row items-center gap-4")
    const label = row.firstElementChild as HTMLElement
    expect(label.className).toBe("flex-grow font-medium text-sm")
    expect(label.textContent).toBe("Sample Count")
    expect(count_input).not.toBeNull()
    expect(count_input.value).toBe("50")
  })

  it("caps the count at 200 by default", async () => {
    const { count_input } = setup()
    await fireEvent.input(count_input, { target: { value: "9999" } })
    expect(count_input.value).toBe("200")
  })

  it("honours a count_max override", async () => {
    const { count_input } = setup({ count_max: 20 })
    await fireEvent.input(count_input, { target: { value: "9999" } })
    expect(count_input.value).toBe("20")
  })

  it("renders the guidance field with the shipped label and description", () => {
    const { container, guidance_box } = setup()
    expect(guidance_box).not.toBeNull()
    expect(guidance_box.getAttribute("aria-label")).toBe("Guidance")
    expect(container.textContent).toContain(GUIDANCE_DESCRIPTION)
    // The id the generate page's label and autofill hooks rely on, and the
    // medium box height: tall enough for a few lines of steer, short enough
    // that the plan dialog keeps its submit button on screen.
    expect(guidance_box.id).toBe("batch_guidance")
    expect(guidance_box.className).toContain("h-36")
  })

  it("leaves the guidance box on FormElement's label placeholder by default", () => {
    // Unset, the prop changes nothing: FormElement keeps falling back to the
    // field's own label, which is what every existing caller renders today.
    const { guidance_box } = setup()
    expect(guidance_box.getAttribute("placeholder")).toBe("Guidance")
  })

  it("passes guidance_placeholder through to the guidance box", () => {
    // Surfaces that start the box empty show the shape of a good answer here
    // instead of prefilling text the user didn't write.
    const { guidance_box } = setup({
      guidance_placeholder: `For example, "Fewer refund scenarios."`,
    })
    expect(guidance_box.getAttribute("placeholder")).toBe(
      `For example, "Fewer refund scenarios."`,
    )
  })

  it("uses a guidance_id override for the field's id", () => {
    const { guidance_box } = setup({ guidance_id: "eval_batch_guidance" })
    expect(guidance_box.id).toBe("eval_batch_guidance")
  })

  it("uses count_label as the noun in the count row", () => {
    const { container } = setup({ count_label: "Trace Count" })
    const label = container.firstElementChild?.firstElementChild as HTMLElement
    expect(label.textContent).toBe("Trace Count")
  })

  it("hides Reset when there is no guidance template", () => {
    const { container } = setup({ guidance: "edited by hand" })
    expect(reset_button(container)).toBeNull()
  })

  it("hides Reset when the guidance still matches the template", () => {
    const { container } = setup({
      guidance: "start here",
      guidance_template: "start here",
    })
    expect(reset_button(container)).toBeNull()
  })

  it("shows Reset once guidance differs from the template, and restores it", async () => {
    const { container, guidance_box } = setup({
      guidance: "edited",
      guidance_template: "start here",
    })
    const reset = reset_button(container)
    expect(reset).not.toBeNull()
    await fireEvent.click(reset as HTMLButtonElement)
    expect(guidance_box.value).toBe("start here")
    // Back on the template, so the action disappears again.
    expect(reset_button(container)).toBeNull()
  })

  it("renders no warning by default", () => {
    const { container } = setup()
    expect(container.querySelector("svg.text-warning")).toBeNull()
  })

  it("renders an amber warning after the rows when warning_message is set", () => {
    const { container } = setup({ warning_message: "This costs money." })
    const warning = container.lastElementChild as HTMLElement
    expect(warning.textContent).toContain("This costs money.")
    expect(warning.querySelector("svg.text-warning")).not.toBeNull()
    // Warning carries no margin of its own: the caller's form column spaces
    // it above the submit button. Its text sits at the default indent so it
    // lines up with the field labels above it.
    expect(warning.className).not.toMatch(/\bm[tby]-\d/)
    expect((warning.lastElementChild as HTMLElement).className).toContain(
      "pl-4",
    )
  })

  it("propagates count and guidance edits to the bound props", async () => {
    const { container, component, guidance_box } = setup()
    const increase = container.querySelector(
      'button[aria-label="Increase"]',
    ) as HTMLButtonElement
    await fireEvent.click(increase)
    expect(bound_prop<number>(component, "count")).toBe(51)

    await fireEvent.input(guidance_box, { target: { value: "be terse" } })
    expect(bound_prop<string>(component, "guidance")).toBe("be terse")
  })

  it("treats guidance as required by default, with no Optional badge", () => {
    // Pins the shipped /generate rendering: that page prefills the box, so a
    // blank one there means the user cleared a prompt rather than opted out.
    const { container } = setup()
    expect(optional_badge(container)).toBeNull()
  })

  it("badges guidance Optional when guidance_optional is set", () => {
    // The empty box has no placeholder of its own, so this badge is the only
    // cue that leaving it blank is a valid answer.
    const { container } = setup({ guidance_optional: true })
    expect(optional_badge(container)).not.toBeNull()
  })
})

describe("KilnProBatchForm inside a form", () => {
  function submit_setup(props: Record<string, unknown> = {}) {
    const on_submit = vi.fn()
    const utils = render(KilnProBatchFormInFormContainer, {
      props: { count: 50, guidance: "", ...props },
    })
    utils.component.$on("submit", on_submit)
    const submit = utils.container.querySelector(
      'button[type="submit"]',
    ) as HTMLButtonElement
    const guidance_box = utils.container.querySelector(
      "textarea",
    ) as HTMLTextAreaElement
    return { ...utils, on_submit, submit, guidance_box }
  }

  // Leaves the guidance box empty the way a user does: type, then clear. This
  // is what makes FormElement mark the field, which is the state FormContainer
  // reads when it decides whether the submit may go through.
  async function empty_the_box(guidance_box: HTMLTextAreaElement) {
    await fireEvent.input(guidance_box, { target: { value: "x" } })
    await fireEvent.input(guidance_box, { target: { value: "" } })
    await tick()
  }

  // FormContainer validates asynchronously and flushes the DOM before it
  // decides, so let the whole chain settle before asserting.
  async function click_submit(button: HTMLButtonElement) {
    await fireEvent.click(button)
    await new Promise((resolve) => setTimeout(resolve, 20))
    await tick()
  }

  it("blocks a submit on an empty guidance box by default", async () => {
    const { container, on_submit, submit, guidance_box } = submit_setup()
    await empty_the_box(guidance_box)
    expect(guidance_box.className).toContain("textarea-error")
    await click_submit(submit)
    expect(on_submit).not.toHaveBeenCalled()
    expect(container.textContent).toContain("Please correct the errors above")
  })

  it("submits an empty guidance box when guidance_optional is set", async () => {
    // The regression this guards: a surface whose guidance box is empty by
    // design could never submit, because the shared field refused the blank.
    const { container, on_submit, submit, guidance_box } = submit_setup({
      guidance_optional: true,
    })
    await empty_the_box(guidance_box)
    expect(guidance_box.className).not.toContain("textarea-error")
    await click_submit(submit)
    expect(on_submit).toHaveBeenCalledTimes(1)
    expect(container.textContent).not.toContain(
      "Please correct the errors above",
    )
  })
})
