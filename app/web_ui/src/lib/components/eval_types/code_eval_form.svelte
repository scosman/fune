<script lang="ts">
  import type { components } from "$lib/api_schema"
  import FormElement from "$lib/utils/form_element.svelte"
  import type { InlineAction } from "$lib/utils/form_element.svelte"
  import CodeEditor from "$lib/components/code_editor.svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import Warning from "$lib/ui/warning.svelte"
  import ToolsSelector from "$lib/ui/run_config_component/tools_selector.svelte"
  import type { EvalOutputScore } from "$lib/types"
  import { generate_default_code, generate_examples } from "./code_eval_helpers"
  import { SHOW_REFERENCE_DATA_UI } from "$lib/utils/eval_types/reference_data_ui"
  import { string_to_json_key } from "$lib/utils/json_schema_editor/json_schema_templates"

  export let output_scores: EvalOutputScore[] | undefined = undefined
  export let project_id: string = ""

  // Creation flow: the score is named after the eval, which the user is still
  // typing. The starter code begins with a static "score_name_placeholder" key
  // and regenerates with the real key live as the eval is named, until the
  // user's first manual edit freezes it. Running the judge validates the
  // returned keys against the eval's scores.
  export let placeholder_score_key: boolean = false

  function initial_code(): string {
    if (placeholder_score_key) {
      return generate_default_code(placeholder_scores(output_scores))
    }
    return generate_default_code(output_scores)
  }

  function placeholder_scores(
    scores: EvalOutputScore[] | undefined,
  ): EvalOutputScore[] {
    return [
      {
        name: "score_name_placeholder",
        type: scores?.[0]?.type ?? "pass_fail",
      } as EvalOutputScore,
    ]
  }

  export let properties: components["schemas"]["CodeEvalProperties"] & {
    timeout_seconds?: number
  } = {
    type: "code_eval",
    code: initial_code(),
    reference_keys: [],
    timeout_seconds: 180,
    tool_allowlist: [],
  }

  // Bindable code string so the parent can track code edits reactively.
  export let code_string: string = properties.code

  let user_has_edited = false
  // The editor dispatches `change` for programmatic setValue too, so remember
  // what we generated ourselves: only a change that differs from our own
  // generation counts as a user edit. Without this, a programmatic
  // regeneration would mark the form as edited and freeze all further
  // regeneration.
  let last_generated_code = properties.code

  // The keys the eval expects the score function to return.
  $: expected_score_keys = (output_scores ?? [])
    .map((score) => string_to_json_key(score.name))
    .filter((key) => key.length > 0)

  // Until the user's first manual edit, the starter tracks the eval's scores:
  // in placeholder mode the real score key replaces the placeholder live as
  // the eval is named (falling back to the placeholder while there is no
  // valid name). The first genuine edit freezes the code permanently.
  $: if (output_scores && !user_has_edited) {
    const scores =
      placeholder_score_key && expected_score_keys.length === 0
        ? placeholder_scores(output_scores)
        : output_scores
    const new_code = generate_default_code(scores)
    if (new_code !== properties.code) {
      last_generated_code = new_code
      properties.code = new_code
      code_string = new_code
      code_editor?.setValue(new_code)
    }
  }

  // The starter was never edited if the generated placeholder key is still in
  // the code — a string this codebase generated, so blocking on its presence
  // at save time carries none of the false-positive risk of asserting the
  // real key appears. (Guard against an eval genuinely named to produce this
  // key, however unlikely.)
  export function validate(): string | null {
    const code = properties.code ?? ""
    if (
      /\bscore_name_placeholder\b/.test(code) &&
      !expected_score_keys.includes("score_name_placeholder")
    ) {
      const key_hint =
        expected_score_keys.length > 0
          ? ` with the eval score key ${expected_score_keys
              .map((key) => `"${key}"`)
              .join(", ")}`
          : ""
      return `Replace "score_name_placeholder" in your score function${key_hint} before saving.`
    }
    return null
  }

  // Hidden while the eval name is empty or invalid (no keys to show yet).
  // The replace-the-placeholder wording only applies while the placeholder is
  // actually in the code (the user edited before naming the eval); otherwise
  // the note simply states the required key.
  $: code_has_placeholder = /\bscore_name_placeholder\b/.test(code_string ?? "")
  $: score_key_note =
    expected_score_keys.length > 0
      ? code_has_placeholder
        ? `Replace "score_name_placeholder" in the code below with the eval score key ${expected_score_keys
            .map((key) => `"${key}"`)
            .join(", ")}.`
        : `Your function must return the score key${
            expected_score_keys.length > 1 ? "s" : ""
          } ${expected_score_keys.map((key) => `"${key}"`).join(", ")}.`
      : null

  let timeout_seconds: number = properties.timeout_seconds ?? 180

  $: properties.timeout_seconds = timeout_seconds

  let tool_allowlist: string[] = properties.tool_allowlist ?? []

  $: properties.tool_allowlist = tool_allowlist

  export function getProperties(): Omit<
    components["schemas"]["CodeEvalProperties"],
    "reference_keys"
  > & {
    timeout_seconds?: number
  } {
    return {
      type: "code_eval",
      code: properties.code,
      timeout_seconds,
      tool_allowlist,
    }
  }

  let examples_dialog: Dialog
  let active_example_tab: number = 0

  // Examples follow the same rule as the starter: the real score key once the
  // eval has a valid name, the placeholder before then.
  $: examples = generate_examples(
    placeholder_score_key && expected_score_keys.length === 0
      ? placeholder_scores(output_scores)
      : output_scores,
  )

  function show_examples() {
    active_example_tab = 0
    examples_dialog.show()
  }

  function use_example(): boolean {
    const example = examples[active_example_tab]
    properties.code = example.code
    code_string = example.code
    code_editor?.setValue(example.code)
    // The snippet's tool calls fail against an allowlist that doesn't list them, so
    // taking an example grants what it needs. Unioned rather than replaced: it must
    // not drop tools the user already picked, and re-using an example is then a no-op.
    tool_allowlist = [
      ...new Set([...tool_allowlist, ...example.required_tool_ids]),
    ]
    user_has_edited = true
    return true
  }

  let code_editor: CodeEditor

  const examples_inline_action: InlineAction = {
    handler: show_examples,
    label: "Examples",
  }

  function on_code_change(e: CustomEvent<string>) {
    properties.code = e.detail
    code_string = e.detail
    if (e.detail !== last_generated_code) {
      user_has_edited = true
    }
  }
</script>

<div class="flex flex-col gap-4">
  <FormElement
    id="code_eval_score_function"
    label="Score Function"
    description="Define a Python score function to evaluate the model's work."
    info_description={SHOW_REFERENCE_DATA_UI
      ? "The Python function can use the model's output, trace, and eval's reference data to drive pragmatic scoring. Faster and cheaper than LLM as a judge."
      : "The Python function can use the model's output and trace to drive pragmatic scoring. Faster and cheaper than LLM as a judge."}
    inputType="header_only"
    inline_action={examples_inline_action}
    value=""
  />
  {#if score_key_note}
    <div data-testid="score-key-note">
      <Warning
        warning_message={score_key_note}
        warning_color="primary"
        warning_icon="info"
        inline={true}
        text_size="xs"
      />
    </div>
  {/if}
  <CodeEditor
    bind:this={code_editor}
    value={properties.code || initial_code()}
    min_height="300px"
    on:change={on_code_change}
  />

  <FormElement
    id="code_eval_timeout"
    label="Timeout (seconds)"
    description="Maximum time allowed for the score function to execute. Must be between 1 and 300 seconds."
    inputType="input_number"
    bind:value={timeout_seconds}
    placeholder="180"
    min={1}
    max={300}
  />

  <ToolsSelector
    {project_id}
    label="Tools"
    settings={{
      description: "The score function can only call tools listed here.",
      info_description:
        "Select the tools this score function is allowed to call. Use them from `score()` via the kiln.tools or kiln.async_tools module. LLM Judge runs an LLM-as-judge call using this eval's own score schema.",
      hide_create_kiln_task_tool_button: true,
      optional: true,
      empty_label: "None (no tool access)",
      sandbox_code_context: "code_eval",
    }}
    bind:tools={tool_allowlist}
  />
</div>

<Dialog
  bind:this={examples_dialog}
  title="Code Judge Examples"
  width="wide"
  action_buttons={[
    {
      label: "Use This Example",
      isPrimary: true,
      action: use_example,
    },
  ]}
>
  <div class="flex flex-col gap-4">
    <!-- .tabs is a grid in DaisyUI v4, so flex is needed for the row to wrap. -->
    <div class="tabs tabs-bordered flex flex-wrap">
      {#each examples as example, i}
        <button
          type="button"
          class="tab shrink-0 whitespace-nowrap {active_example_tab === i
            ? 'tab-active'
            : ''}"
          on:click={() => (active_example_tab = i)}
        >
          {example.label}
        </button>
      {/each}
    </div>
    <div
      class="bg-base-200 rounded-lg p-4 overflow-x-auto font-mono text-sm whitespace-pre"
    >
      {examples[active_example_tab].code}
    </div>
  </div>
</Dialog>
