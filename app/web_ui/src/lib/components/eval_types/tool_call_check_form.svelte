<script lang="ts">
  import type { components } from "$lib/api_schema"
  import FormElement from "$lib/utils/form_element.svelte"
  import FormList from "$lib/utils/form_list.svelte"
  import Collapse from "$lib/ui/collapse.svelte"
  import CloseIcon from "$lib/ui/icons/close_icon.svelte"
  import { available_tools, load_available_tools } from "$lib/stores"
  import type { OptionGroup, Option } from "$lib/ui/fancy_select_types"
  import type { ToolSetApiDescription } from "$lib/types"
  import { build_tool_option_groups } from "$lib/ui/run_config_component/tool_options"

  type ToolCallSpec = components["schemas"]["ToolCallSpec"]

  export let properties: components["schemas"]["ToolCallCheckProperties"] = {
    type: "tool_call_check",
    expected_tools: [],
    match_mode: "all",
    on_unexpected_tools: "ignore",
  }

  // Project context, used to load the available tools for the dropdown.
  export let project_id: string = ""

  $: load_available_tools(project_id)

  // Options for the tool-name dropdown, built by the same shared builder every
  // other tool picker uses, so a tool reads the same way here as it does on /run.
  // The value is the OpenAI-compatible function name a trace records, which the
  // builder surfaces beside the tool where it differs from its display name.
  $: tools_loading = $available_tools[project_id] === undefined
  $: tool_options = build_tool_options($available_tools[project_id])
  $: known_function_names = new Set(
    tool_options.flatMap((group) =>
      group.options.map((option) => String(option.value)),
    ),
  )

  function build_tool_options(
    tool_sets: ToolSetApiDescription[] | undefined,
  ): OptionGroup[] {
    // This form matches tool calls in an agent's trace, so it offers agent tools:
    // context "none". Whole sets the server marks as sandbox-only are dropped,
    // because nothing in one can appear in an agent's trace.
    const groups = build_tool_option_groups(tool_sets, {
      value_field: "function_name",
      sandbox_code_context: "none",
    })

    // Every skill is the same `skill` call in the trace, told apart by its name
    // argument, so the whole set collapses to one option instead of one per skill.
    const skill_set = (tool_sets ?? []).find(
      (tool_set) => tool_set.type === "skill" && tool_set.tools.length > 0,
    )
    if (skill_set) {
      groups.push({
        label: skill_set.set_name,
        options: [
          {
            value: "skill",
            label: "Skill",
            description:
              "Covers all skills; the specific skill is matched via the name argument.",
          },
        ],
      })
    }
    return groups
  }

  // Preserve tool names that aren't in the current tool list (legacy configs,
  // removed/renamed tools) so the dropdown still shows and keeps them.
  $: custom_values = [
    ...new Set(
      properties.expected_tools
        .map((t) => t.tool_name)
        .filter((name) => name && !known_function_names.has(name)),
    ),
  ]
  $: display_tool_options =
    custom_values.length > 0
      ? [
          ...tool_options,
          {
            label: "Other",
            options: custom_values.map(
              (name): Option => ({
                value: name,
                label: name,
                description: "Not found in this project's tools",
              }),
            ),
          },
        ]
      : tool_options

  type ArgMatch = components["schemas"]["ArgMatch"]
  type ArgRow = { name: string; value: string; match_mode: string }

  const empty_tool: ToolCallSpec = {
    tool_name: "",
    expected_args: null,
  }

  let arg_rows: ArgRow[][] = properties.expected_tools.map((tool) =>
    tool.expected_args
      ? Object.entries(tool.expected_args).map(([name, arg]) => ({
          name,
          value: JSON.stringify(arg.value),
          match_mode: arg.match_mode ?? "exact",
        }))
      : [],
  )

  // Keep arg_rows in sync when FormList adds/removes expected_tools entries.
  // We use a WeakMap so each tool object carries its own arg draft state,
  // surviving index shifts caused by removals.
  const tool_arg_map = new WeakMap<ToolCallSpec, ArgRow[]>()
  // Seed the map from the initial arg_rows
  for (let i = 0; i < properties.expected_tools.length; i++) {
    tool_arg_map.set(properties.expected_tools[i], arg_rows[i] ?? [])
  }
  // Data flows one-way: expected_tools → arg_rows (never the reverse).
  // Mutating arg_rows does NOT write back to expected_tools, so no reactive loop.
  $: {
    const synced: ArgRow[][] = []
    for (const tool of properties.expected_tools) {
      const existing = tool_arg_map.get(tool)
      if (existing) {
        synced.push(existing)
      } else {
        const fresh: ArgRow[] = []
        tool_arg_map.set(tool, fresh)
        synced.push(fresh)
      }
    }
    arg_rows = synced
  }

  function is_valid_json(raw: string): boolean {
    try {
      JSON.parse(raw)
      return true
    } catch {
      return false
    }
  }

  // Authoring-time check for one argument row. A value with no name, or a
  // non-empty value that isn't valid JSON, is an error the author must fix --
  // rather than being silently dropped or coerced to a raw string at save time.
  function validate_arg_row(row: ArgRow): {
    name: string | null
    value: string | null
  } {
    const has_name = row.name.trim().length > 0
    const has_value = row.value.trim().length > 0
    return {
      name: has_value && !has_name ? "Add a name, or clear the value." : null,
      value:
        has_value && !is_valid_json(row.value) ? "Must be valid JSON." : null,
    }
  }

  // Inline errors, one entry per row, mirroring the arg_rows shape.
  $: arg_errors = arg_rows.map((rows) => rows.map(validate_arg_row))

  function sync_args_to_properties() {
    for (let i = 0; i < properties.expected_tools.length; i++) {
      const rows = arg_rows[i]
      if (!rows || rows.length === 0) {
        properties.expected_tools[i].expected_args = null
        continue
      }
      const args: Record<string, ArgMatch> = {}
      for (const row of rows) {
        const name = row.name.trim()
        if (!name) continue
        // validate() gates save/test, so a non-empty value is guaranteed valid
        // JSON here; an empty value means "no value constraint" (stored as "").
        const raw = row.value.trim()
        args[name] = {
          value: (raw ? JSON.parse(raw) : "") as ArgMatch["value"],
          match_mode: row.match_mode as ArgMatch["match_mode"],
        }
      }
      properties.expected_tools[i].expected_args =
        Object.keys(args).length > 0 ? args : null
    }
  }

  export function getProperties(): components["schemas"]["ToolCallCheckProperties"] {
    sync_args_to_properties()
    return properties
  }

  export function validate(): string | null {
    if (!properties.expected_tools || properties.expected_tools.length === 0) {
      return "At least one expected tool must be defined."
    }
    for (let i = 0; i < properties.expected_tools.length; i++) {
      if (!properties.expected_tools[i].tool_name.trim()) {
        return `Expected Tool #${i + 1} is missing a name.`
      }
      const rows = arg_rows[i] ?? []
      for (let j = 0; j < rows.length; j++) {
        const err = validate_arg_row(rows[j])
        if (err.name) {
          return `Expected Tool #${i + 1}, argument #${j + 1}: ${err.name}`
        }
        if (err.value) {
          return `Expected Tool #${i + 1}, argument #${j + 1}: ${err.value}`
        }
      }
    }
    return null
  }

  function add_arg(tool_index: number) {
    while (arg_rows.length <= tool_index) {
      arg_rows.push([])
    }
    const new_row = { name: "", value: "", match_mode: "exact" }
    arg_rows[tool_index] = [...arg_rows[tool_index], new_row]
    const tool = properties.expected_tools[tool_index]
    if (tool) tool_arg_map.set(tool, arg_rows[tool_index])
    arg_rows = arg_rows
  }

  function remove_arg(tool_index: number, arg_index: number) {
    arg_rows[tool_index] = arg_rows[tool_index].filter(
      (_, i) => i !== arg_index,
    )
    const tool = properties.expected_tools[tool_index]
    if (tool) tool_arg_map.set(tool, arg_rows[tool_index])
    arg_rows = arg_rows
  }
</script>

<div class="flex flex-col gap-6">
  <div class="flex flex-col gap-3">
    <FormElement
      id="tool_call_expected_tools_header"
      inputType="header_only"
      label="Expected Tools"
      description={properties.match_mode === "never"
        ? "Define the tools the agent must NOT call."
        : "Define the tools the agent is expected to call."}
      value=""
    />
    <div class="ml-4 border-l border-base-300 pl-4">
      <FormList
        bind:content={properties.expected_tools}
        content_label="Expected Tool"
        empty_content={structuredClone(empty_tool)}
        let:item_index
      >
        <div class="ml-4 border-l border-base-300 pl-4">
          <div class="flex flex-col gap-2">
            <FormElement
              id="tool_name_{item_index}"
              label="Tool"
              description="The tool that should be called."
              info_description="Tools are matched by the function name a trace records, shown beside the tool when it differs from its display name in Kiln."
              inputType="fancy_select"
              fancy_select_options={display_tool_options}
              empty_label="Select a tool"
              empty_state_message={tools_loading
                ? "Loading tools..."
                : "No tools available"}
              empty_state_subtitle="Add Tools"
              empty_state_link={`/tools/${project_id}/add_tools`}
              bind:value={properties.expected_tools[item_index].tool_name}
            />

            <Collapse
              title="Expected Arguments"
              description="Optionally check specific argument values passed to this tool."
              open={(arg_rows[item_index] ?? []).length > 0}
            >
              {#each arg_rows[item_index] ?? [] as arg_row, arg_index}
                <div class="flex gap-2 items-center">
                  <div class="flex-1">
                    <FormElement
                      id="arg_name_{item_index}_{arg_index}"
                      label={arg_index === 0 ? "Arg Name" : ""}
                      inputType="input"
                      placeholder="e.g. query"
                      bind:value={arg_row.name}
                      error_message={arg_errors[item_index]?.[arg_index]?.name}
                    />
                  </div>
                  <div class="flex-1">
                    <FormElement
                      id="arg_value_{item_index}_{arg_index}"
                      label={arg_index === 0 ? "Expected Value (JSON)" : ""}
                      info_description={arg_index === 0
                        ? 'Values must be valid JSON. Strings must be quoted (e.g. "hello"), numbers and booleans are bare (e.g. 42, true).'
                        : ""}
                      inputType="input"
                      placeholder={'"hello", 42, true'}
                      bind:value={arg_row.value}
                      error_message={arg_errors[item_index]?.[arg_index]?.value}
                    />
                  </div>
                  <div class="w-32">
                    <FormElement
                      id="arg_match_{item_index}_{arg_index}"
                      label={arg_index === 0 ? "Comparison" : ""}
                      inputType="select"
                      bind:value={arg_row.match_mode}
                      select_options={[
                        ["exact", "Exact"],
                        ["contains", "Contains"],
                        ["regex", "Regex"],
                      ]}
                    />
                  </div>
                  <button
                    class="btn btn-ghost btn-sm btn-circle text-gray-500 hover:text-gray-700"
                    on:click={() => remove_arg(item_index, arg_index)}
                    aria-label="Remove argument"
                  >
                    <span class="h-4 w-4 block"><CloseIcon /></span>
                  </button>
                </div>
              {/each}
              <button
                class="btn btn-ghost btn-sm self-start"
                on:click={() => add_arg(item_index)}
              >
                + Add Expected Argument
              </button>
            </Collapse>
          </div>
        </div>
      </FormList>
    </div>
  </div>

  <FormElement
    id="tool_call_check_match_mode"
    label="Match Mode"
    description="How to match tools against the trace."
    inputType="radio"
    radio_options={[
      {
        value: "any",
        label: "Any",
        description: "Pass if at least one of the expected tools was called.",
      },
      {
        value: "all",
        label: "All (any order)",
        description: "Pass if every expected tool was called, in any order.",
      },
      {
        value: "ordered",
        label: "Ordered (in list order)",
        description:
          "Pass if all expected tools were called in the order listed.",
      },
      {
        value: "never",
        label: "Never",
        description: "Pass if none of the listed tools were called.",
      },
    ]}
    bind:value={properties.match_mode}
  />

  {#if properties.match_mode !== "never"}
    <FormElement
      id="tool_call_check_on_unexpected"
      label="Unlisted Tool Calls"
      description="What happens when the model calls tools not in your list."
      inputType="radio"
      radio_options={[
        {
          value: "ignore",
          label: "Allow",
          description: "Extra tool calls beyond the expected list are allowed.",
        },
        {
          value: "fail",
          label: "Fail",
          description:
            "Fail if any tool is called that is not in the expected list.",
        },
      ]}
      bind:value={properties.on_unexpected_tools}
    />
  {/if}
</div>
