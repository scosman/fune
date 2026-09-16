// Source assertions for the entry page: its copy, its section order, and its
// run-config dialog. The page mounts the whole run-config picker (model list,
// tool and skill pickers, task stores), which a render test would have to stub
// down to nothing; the rules below are contractual either way. Reading the source is the house precedent
// for pinning facts a render test can't reach (see builder/generate_step_surface.test.ts).
import * as fs from "fs"
import * as path from "path"
import { describe, expect, it } from "vitest"

const page_source = fs.readFileSync(
  path.resolve(__dirname, "./+page.svelte"),
  "utf-8",
)

// Collapses runs of whitespace so an assertion survives Prettier rewrapping a
// long attribute across lines.
function normalize(source: string): string {
  return source.replace(/\s+/g, " ")
}

const normalized = normalize(page_source)

function contains(needle: string): boolean {
  return normalized.includes(normalize(needle))
}

// The slice of the page a claim is about, so a negative assertion is not
// answered by some unrelated part of the file.
function region(start_anchor: string, end_anchor: string): string {
  const start = page_source.indexOf(start_anchor)
  if (start < 0) {
    throw new Error(`anchor not found in +page.svelte: ${start_anchor}`)
  }
  const end = page_source.indexOf(end_anchor, start + start_anchor.length)
  if (end < 0) {
    throw new Error(
      `end anchor not found after "${start_anchor}": ${end_anchor}`,
    )
  }
  return page_source.slice(start, end + end_anchor.length)
}

const dialog = region("bind:this={run_config_dialog}", "</Dialog>")
const continue_fn = region(
  "async function continue_with_description() {",
  "\n  }",
)
const goto_builder_fn = region(
  "function goto_builder(run_config_id: string | null) {",
  "\n  }",
)
const open_dialog_fn = region("function open_run_config_dialog() {", "\n  }")

// The page body below the offer, and the two arms inside it. Section order is
// read as source order within an arm, which is the order the sections stack.
const page_body = region(
  '<div class="pt-6 max-w-5xl flex flex-col gap-10">',
  "</AppPage>",
)
const pro_arm_start = page_body.indexOf("{#if $kilnCopilotConnected === true}")
const manual_arm_start = page_body.indexOf("{:else}")
const assistant_arm = page_body.slice(pro_arm_start, manual_arm_start)
const manual_arm = page_body.slice(manual_arm_start, page_body.indexOf("{/if}"))

// Where a section sits in the body, by the anchor that identifies it.
function position_of(anchor: string): number {
  const at = page_body.indexOf(anchor)
  if (at < 0) {
    throw new Error(`section anchor not found in the page body: ${anchor}`)
  }
  return at
}

describe("the run config dialog", () => {
  it("opens on Continue instead of going straight to the builder", () => {
    // The choice is asked before the wizard starts, because everything the
    // wizard does afterwards reads the chosen config.
    expect(contains("on:click={open_run_config_dialog}")).toBe(true)
    expect(normalize(open_dialog_fn)).toContain("run_config_dialog?.show()")
  })

  it("hands over without a config when there is no task to pick from", () => {
    // A task that failed to load leaves the dialog with no picker in it, so
    // opening would show an empty box. The builder resolves the task default
    // when nothing is chosen, which is what this page did before it asked.
    expect(normalize(open_dialog_fn)).toContain(
      "if (!task) { goto_builder(null) return }",
    )
  })

  it("picks a config with the same control the run page uses", () => {
    // Same dropdown over the same component, so a config is chosen the same
    // way in both places and its tools and skills are visible while choosing.
    expect(dialog).toContain("<SavedRunConfigsDropdown")
    expect(dialog).toContain("<RunConfigComponent")
    expect(normalize(dialog)).toContain(
      "save_new_run_config={handle_save_new_run_config}",
    )
    expect(normalize(dialog)).toContain("hide_prompt_selector={true}")
    expect(normalize(dialog)).toContain(
      "show_tools_selector_in_advanced={true}",
    )
    expect(normalize(dialog)).toContain("show_name_field={false}")
  })

  it("names the choice in its title and says what it is for", () => {
    // The title names the thing being chosen, and the subtitle says what the
    // choice is used for, so the dialog is readable without the page behind
    // it.
    expect(normalize(dialog)).toContain('title="Choose Run Config"')
    expect(normalize(dialog)).toContain(
      'subtitle="Choose how your task will be run for generating examples in the eval builder."',
    )
  })

  it("explains what the choice decides", () => {
    expect(
      contains(
        'info_description="The run config your task runs with in the eval builder. Kiln uses its tools and skills to write the questions and the judge, then runs it to create the eval dataset."',
      ),
    ).toBe(true)
  })
})

describe("continuing into the builder", () => {
  it("saves an edited config before proceeding", () => {
    // "Custom" is only in the picker's local state, while the wizard, the
    // drive and the saved eval all refer to the config by id. Navigating on
    // "custom" would hand the builder an id that resolves to nothing.
    const body = normalize(continue_fn)
    expect(body).toContain('if (!run_config_id || run_config_id === "custom")')
    expect(body).toContain("await handle_save_new_run_config()")
    expect(body.indexOf("handle_save_new_run_config")).toBeLessThan(
      body.indexOf("goto_builder("),
    )
  })

  it("hands the chosen config over beside the description", () => {
    expect(normalize(goto_builder_fn)).toContain(
      "`&run_config_id=${encodeURIComponent(run_config_id)}`",
    )
  })

  it("never navigates without an id", () => {
    // A saved config with no id is a broken save, not a reason to start a
    // wizard that cannot resolve its target.
    expect(normalize(continue_fn)).toContain(
      'if (!run_config_id) { throw new Error("The saved run config has no id.") }',
    )
  })
})

describe("the entry page's copy", () => {
  it("names the page for what the user is here to do", () => {
    expect(normalize(page_source)).toContain('<AppPage title="Create Eval"')
  })

  it("counts the ways to create an eval, per state", () => {
    // The subtitle counts what is actually on the screen: the offer screen
    // shows two buttons and no template list, so it keeps its own line.
    expect(normalize(page_source)).toContain(
      'subtitle={show_offer ? "Kiln Pro drafts the eval for you, or set one up yourself." ' +
        ': $kilnCopilotConnected === true ? "Three ways to create an eval" ' +
        ': "Two ways to create an eval"}',
    )
  })

  it("describes the assistant in its section header, not on the field", () => {
    // What the assistant does belongs to the section, so the field keeps the
    // one question it asks and nothing else.
    expect(normalize(assistant_arm)).toContain(
      '<SettingsHeader title="LLM Judge Assistant" subtitle="Describe what to evaluate in plain language. ' +
        'Kiln Pro writes the eval and generates the dataset." />',
    )
    expect(normalize(assistant_arm)).toContain(
      'label="What should this eval check?"',
    )
    // Scoped to the field itself, and to the prop's own name: the arm is full
    // of other props that end in "description=".
    const description_field = assistant_arm.slice(
      assistant_arm.indexOf("<FormElement"),
      assistant_arm.indexOf("/>", assistant_arm.indexOf("<FormElement")),
    )
    expect(description_field).toContain('label="What should this eval check?"')
    expect(description_field).not.toContain(' description="')
  })

  it("labels the primary for what it produces", () => {
    expect(normalize(assistant_arm)).toContain(
      "on:click={open_run_config_dialog} > Write My Eval",
    )
  })

  it("offers the templates as a quiet alternative that stays on the page", () => {
    // Someone who already knows the template they want moves down the page
    // rather than navigating, so a typed description is not lost.
    expect(normalize(assistant_arm)).toContain(
      'class="link underline text-sm text-gray-500" on:click={scroll_to_templates} > or use templates',
    )
    expect(normalize(page_body)).toContain('<div id="llm_judge_templates"')
    expect(normalize(page_source)).toContain(
      'document .getElementById("llm_judge_templates")',
    )
  })
})

describe("the entry page's sections", () => {
  it("leads with the assistant and puts the templates last when Kiln Pro is on", () => {
    expect(position_of('title="LLM Judge Assistant"')).toBeLessThan(
      position_of('title="Programmatic Checks"'),
    )
    expect(position_of('title="Programmatic Checks"')).toBeLessThan(
      position_of('<div id="llm_judge_templates"'),
    )
  })

  it("leads with the templates when there is no assistant to lead with", () => {
    // The manual arm opens on the list it always opened on; only its title
    // changed, so the section still says what it holds.
    expect(normalize(manual_arm)).toContain(
      '<SettingsHeader title="LLM Judge Templates" />',
    )
  })

  it("renders the templates as a section rather than a disclosure", () => {
    // A list behind a disclosure read like opting out of the assistant. It is
    // a section now, so the Collapse and its import are gone.
    expect(page_source).not.toContain("<Collapse")
    expect(page_source).not.toContain("collapse.svelte")
  })
})
