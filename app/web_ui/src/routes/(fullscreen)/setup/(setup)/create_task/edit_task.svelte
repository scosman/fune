<script lang="ts">
  import type { Task, TurnMode } from "$lib/types"
  import Output from "$lib/ui/output.svelte"
  import FormElement from "$lib/utils/form_element.svelte"
  import FormList from "$lib/utils/form_list.svelte"
  import FormContainer from "$lib/utils/form_container.svelte"
  import {
    filename_string_validator_default,
    normalize_filename_string,
  } from "$lib/utils/input_validators"
  import SchemaSection from "./schema_section.svelte"
  import { current_project, load_current_task } from "$lib/stores"
  import { load_rating_options } from "$lib/stores/rating_options_store"
  import { goto } from "$app/navigation"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { ui_state, projects } from "$lib/stores"
  import { get } from "svelte/store"
  import { client } from "$lib/api_client"
  import { tick } from "svelte"
  import posthog from "posthog-js"
  import Collapse from "$lib/ui/collapse.svelte"

  // Prevents flash of complete UI if we're going to redirect
  export let redirect_on_created: string | null = "/"
  export let hide_example_task: boolean = false

  // Simplify the create view for onboarding
  export let onboarding: boolean = false

  // turn_mode is immutable after creation — backend rejects PATCHes.
  export let read_only_turn_mode: boolean = false

  // @ts-expect-error This is a partial task, which is fine.
  export let task: Task = {
    name: "",
    description: "",
    instruction: "",
    requirements: [],
  }

  $: creating = !task.id
  $: editing = !creating
  $: show_requirements = !onboarding && task.requirements.length > 0

  let turn_mode: TurnMode = task.turn_mode ?? "single_turn"
  $: is_multiturn = turn_mode === "multiturn"

  // These have their own custom VM, which is translated back to the model on save
  let outputSchemaSection: SchemaSection
  let inputSchemaSection: SchemaSection

  let error: KilnError | null = null
  let submitting = false
  export let saved: boolean = false

  // Track initial values to detect actual changes from the loaded state.
  // This prevents false "unsaved changes" warnings when opening an existing task.
  let initial_name: string
  let initial_description: string | null | undefined
  let initial_instruction: string
  let initial_thinking_instruction: string | null | undefined
  let initial_requirements: Array<{
    name: string | undefined
    instruction: string | undefined
    type: string | undefined
    priority: number | undefined
  }>

  function reset_initial_values() {
    initial_name = task.name
    initial_description = task.description
    initial_instruction = task.instruction
    initial_thinking_instruction = task.thinking_instruction
    initial_requirements = task.requirements.map((r) => ({
      name: r.name,
      instruction: r.instruction,
      type: r.type,
      priority: r.priority,
    }))
  }
  reset_initial_values()

  function requirements_changed(
    reqs: Task["requirements"],
    initial: typeof initial_requirements,
  ): boolean {
    if (reqs.length !== initial.length) return true
    return reqs.some(
      (r, i) =>
        (r.name || "") !== (initial[i].name || "") ||
        (r.instruction || "") !== (initial[i].instruction || "") ||
        (r.type || "") !== (initial[i].type || "") ||
        r.priority !== initial[i].priority,
    )
  }

  // Warn before unload only if there are actual changes from the initial state
  $: warn_before_unload =
    task.name !== initial_name ||
    (task.description || "") !== (initial_description || "") ||
    task.instruction !== initial_instruction ||
    (task.thinking_instruction || "") !==
      (initial_thinking_instruction || "") ||
    requirements_changed(task.requirements, initial_requirements)

  // Allow explicitly setting project ID, or infer current project ID
  export let explicit_project_id: string | undefined = undefined
  $: target_project_id = explicit_project_id || $current_project?.id || null

  export let project_target_name: string | null = null
  $: {
    if (!target_project_id) {
      project_target_name = null
    } else {
      project_target_name =
        $projects?.projects.find((p) => p.id === target_project_id)?.name ||
        "Project ID: " + target_project_id
    }
  }

  async function create_task() {
    try {
      saved = false
      if (!target_project_id) {
        error = new KilnError(
          "You must create a project before creating a task",
          null,
        )
        return
      }
      task.name = normalize_filename_string(task.name)
      let body: Record<string, unknown> = {
        name: task.name,
        description: task.description,
        instruction: task.instruction,
        requirements: task.requirements,
        thinking_instruction: task.thinking_instruction,
      }
      if (creating) {
        body.turn_mode = turn_mode
        if (!is_multiturn) {
          body.input_json_schema =
            inputSchemaSection?.get_schema_string("input_schema") ?? null
          body.output_json_schema =
            outputSchemaSection?.get_schema_string("output_schema") ?? null
        }
      }
      const project_id = target_project_id
      if (!project_id) {
        throw new KilnError("Current project not found", null)
      }
      let data: Task | undefined
      let network_error: unknown | null = null
      if (creating) {
        const { data: post_data, error: post_error } = await client.POST(
          "/api/projects/{project_id}/tasks",
          {
            params: {
              path: {
                project_id,
              },
            },
            body: body,
          },
        )
        data = post_data
        network_error = post_error
        if (!network_error) {
          posthog.capture("create_task", {})
        }
      } else {
        const { data: patch_data, error: patch_error } = await client.PATCH(
          "/api/projects/{project_id}/tasks/{task_id}",
          {
            params: {
              path: {
                project_id,
                task_id: task.id || "",
              },
            },
            body: body,
          },
        )
        data = patch_data
        network_error = patch_error
        if (!network_error) {
          posthog.capture("update_task", {})
        }
      }
      if (network_error || !data) {
        throw network_error
      }

      error = null
      // Make this the current task
      ui_state.set({
        ...get(ui_state),
        current_task_id: data.id ?? null,
        current_project_id: target_project_id ?? null,
      })
      saved = true
      reset_initial_values()

      // reload the current task to make sure changes propagate throughout the UI
      // e.g. the rating options
      await load_current_task(get(current_project)?.id)
      if (target_project_id && data.id) {
        try {
          await load_rating_options(target_project_id, data.id, true)
        } catch (refreshError) {
          console.warn(
            "Task was saved, but refreshing rating options failed",
            refreshError,
          )
        }
      }

      // Wait for the saved change to propagate to the warn_before_unload
      await tick()
      if (redirect_on_created) {
        goto(redirect_on_created)
      }
    } catch (e) {
      error = createKilnError(e)
    } finally {
      submitting = false
    }
  }

  export function has_edits(): boolean {
    let has_edited_requirements = task.requirements.some(
      (req) => !!req.name || !!req.instruction,
    )
    return (
      !!task.name ||
      !!task.description ||
      !!task.instruction ||
      !!task.thinking_instruction ||
      has_edited_requirements ||
      !!inputSchemaSection?.get_schema_string("input_schema") ||
      !!outputSchemaSection?.get_schema_string("output_schema")
    )
  }

  function example_task() {
    if (has_edits()) {
      if (
        !confirm("This will replace your current task edits. Are you sure?")
      ) {
        return
      }
    }

    // Multi-turn tasks are plain-text conversations (no input/output schema),
    // so the example is a generic chat assistant rather than the structured
    // joke generator used for single-turn.
    if (is_multiturn) {
      // @ts-expect-error This is a partial task, which is fine.
      task = {
        name: "Chat Assistant",
        description: "An example multi-turn task from the KilnAI team.",
        instruction:
          "You are a helpful assistant. Have a natural back-and-forth conversation with the user: answer their questions clearly and concisely, ask for clarification when you need it, and use the context from earlier in the conversation.",
        requirements: [],
      }
      turn_mode = "multiturn"
      return
    }

    // @ts-expect-error This is a partial task, which is fine.
    task = {
      name: "Joke Generator",
      description: "An example task from the KilnAI team.",
      instruction:
        "Generate a joke, given a theme. The theme will be provided as a word or phrase as the input to the model. The assistant should output a joke that is funny and relevant to the theme. If a style is provided, the joke should be in that style. The output should include a setup and punchline.",
      requirements: [],
      input_json_schema: JSON.stringify({
        type: "object",
        properties: {
          joke_topic: {
            title: "Joke Topic",
            type: "string",
            description: "The topic of the joke.",
          },
          joke_style: {
            title: "Joke Style",
            type: "string",
            description:
              "The style of the joke, such as 'dad joke' or 'kids joke'.",
          },
        },
        required: ["joke_topic"],
        additionalProperties: false,
      }),
      output_json_schema: JSON.stringify({
        type: "object",
        properties: {
          setup: {
            title: "setup",
            type: "string",
            description: "The setup to the joke",
          },
          punchline: {
            title: "punchline",
            type: "string",
            description: "The punchline to the joke",
          },
        },
        required: ["setup", "punchline"],
        additionalProperties: false,
      }),
    }
    turn_mode = "single_turn"
  }

  function prompt_description() {
    if (!editing) {
      return "The prompt for the model to follow."
    }
    if (task.requirements.length > 0) {
      return "The base prompt used by prompt generators (Basic, Few-shot, etc.). The task requirements below are appended to this. You can create additional prompts in the 'Prompts' tab to compare prompt performance."
    }
    return "The base prompt used by prompt generators (Basic, Few-shot, etc.). You can create additional prompts in the 'Prompts' tab to compare prompt performance."
  }
</script>

<div class="flex flex-col gap-2 w-full">
  <FormContainer
    submit_label={editing ? "Save Task" : "Create Task"}
    on:submit={create_task}
    bind:warn_before_unload
    bind:error
    bind:submitting
    bind:saved
  >
    <div>
      <div class="text-xl font-bold">Part 1: Overview</div>
      {#if creating && !hide_example_task}
        <h3 class="text-sm mt-1">
          Just exploring?
          <button class="link text-primary" on:click={example_task}
            >Try an example.</button
          >
        </h3>
      {/if}
    </div>
    <FormElement
      label="Task Name"
      id="task_name"
      description="A description for you and your team, not used by the model."
      bind:value={task.name}
      max_length={120}
      validator={filename_string_validator_default}
    />

    <FormElement
      label="Prompt / Task Instructions"
      inputType="textarea"
      id="task_instructions"
      height="medium"
      description={prompt_description()}
      bind:value={task.instruction}
    />

    <!-- Don't show these if onboarding, keep onboarding view as simple as possible -->
    {#if !onboarding}
      <!-- Don't show if Creating. Only on editing. -->
      {#if !creating}
        <FormElement
          label="Task Description"
          inputType="textarea"
          id="task_description"
          description="A description for you and your team, not used by the model."
          optional={true}
          bind:value={task.description}
        />
      {/if}

      <FormElement
        label="'Thinking' Instructions"
        inputType="textarea"
        id="thinking_instructions"
        optional={true}
        description="Instructions for how the model should 'think' about the task prior to answering. Used for chain of thought style prompting."
        info_description="Used when running a 'Chain of Thought' prompt. If left blank, a default 'think step by step' prompt will be used. Optionally customize this with your own instructions to better fit this task."
        bind:value={task.thinking_instruction}
      />
    {/if}

    <div class="text-sm font-medium text-left pt-6 flex flex-col gap-1">
      <div class="text-xl font-bold">Part 2: Task Type</div>
      <div class="text-xs text-gray-500">
        Will the task be a single exchange, or a back-and-forth conversation?
      </div>
    </div>

    <div data-testid="turn-mode-section">
      {#if read_only_turn_mode}
        <div class="flex flex-col gap-1" data-testid="turn-mode-readonly">
          <div class="text-sm">
            <span class="font-medium">Task type:</span>
            <span>{is_multiturn ? "Multi-turn" : "Single-turn"}</span>
          </div>
          <div class="text-xs text-gray-500">
            This setting can't be changed after the task is created.
          </div>
        </div>
      {:else}
        <div
          class="flex flex-col gap-2"
          data-testid="turn-mode-editable"
          role="radiogroup"
          aria-label="Task type"
        >
          <div class="form-control">
            <label class="label cursor-pointer flex flex-row gap-3 py-1">
              <input
                type="radio"
                name="radio-turn-mode"
                class="radio"
                data-testid="turn-mode-single-turn"
                value="single_turn"
                bind:group={turn_mode}
              />
              <div class="flex flex-col grow text-left">
                <span class="label-text">Single-turn</span>
                <span class="text-xs text-gray-500">
                  A single user message and one assistant response.
                </span>
              </div>
            </label>
          </div>
          <div class="form-control">
            <label class="label cursor-pointer flex flex-row gap-3 py-1">
              <input
                type="radio"
                name="radio-turn-mode"
                class="radio"
                data-testid="turn-mode-multiturn"
                value="multiturn"
                bind:group={turn_mode}
              />
              <div class="flex flex-col grow text-left">
                <span class="label-text">Multi-turn</span>
                <span class="text-xs text-gray-500">
                  A back-and-forth conversation with multiple turns.
                </span>
              </div>
            </label>
          </div>
        </div>
      {/if}
    </div>

    <!-- Multi-turn tasks are plain-text conversations with no input/output
         schema, so these sections are hidden entirely for them. -->
    {#if !is_multiturn}
      <div class="text-sm font-medium text-left pt-6 flex flex-col gap-1">
        <div class="text-xl font-bold">Part 3: Input Schema</div>
        <div class="text-xs text-gray-500">
          What kind of input will the model receive?
        </div>
      </div>

      <div>
        {#if editing}
          <div>
            <div class="text-sm mb-2 flex flex-col gap-1">
              <p>
                You can't edit an existing task's input format, as existing
                dataset items would not conform to the new schema.
              </p>
              <p>
                You can
                <a
                  class="link"
                  href="/settings/clone_task/{target_project_id}/{task.id}"
                  >clone this task</a
                >
                instead.
              </p>
            </div>
            <Output
              raw_output={task.input_json_schema || "Input Format: Plain text"}
            />
          </div>
        {:else}
          <SchemaSection
            bind:this={inputSchemaSection}
            bind:schema_string={task.input_json_schema}
          />
        {/if}
      </div>

      <div class="text-sm font-medium text-left pt-6 flex flex-col gap-1">
        <div class="text-xl font-bold">Part 4: Output Schema</div>
        <div class="text-xs text-gray-500">
          What kind of output will the model produce?
        </div>
      </div>

      <div>
        {#if editing}
          <div>
            <div class="text-sm mb-2 flex flex-col gap-1">
              <p>
                You can't edit an existing task's output format, as existing
                dataset items would not conform to the new schema.
              </p>
              <p>
                You can
                <a
                  class="link"
                  href="/settings/clone_task/{target_project_id}/{task.id}"
                  >clone this task</a
                >
                instead.
              </p>
            </div>
            <Output
              raw_output={task.output_json_schema ||
                "Output Format: Plain text"}
            />
          </div>
        {:else}
          <SchemaSection
            bind:this={outputSchemaSection}
            bind:schema_string={task.output_json_schema}
            warn_about_required={true}
          />
        {/if}
      </div>
    {/if}

    {#if show_requirements}
      <div class="pt-6">
        <Collapse title="Advanced Options" small={false}>
          <div class="text-sm font-medium text-left flex flex-col gap-1">
            <div class="flex flex-row gap-2 items-center">
              <div class="text-xl font-bold" id="requirements_part">
                Part 5: Requirements
              </div>
              <div class="badge badge-sm badge-outline">Deprecated</div>
            </div>
            <div class="text-xs text-gray-500">
              Requirements have been replaced by <a
                href="https://docs.kiln.tech/docs/evals-and-specs"
                target="_blank"
                class="link">Evals</a
              >
              and
              <a
                href="https://docs.kiln.tech/docs/prompts"
                target="_blank"
                class="link">Prompts</a
              >, which offer more powerful ways to evaluate and improve your
              task's output. Requirements will be removed from an upcoming
              release. We recommend migrating.
            </div>
          </div>

          <!-- Requirements Section -->
          <FormList
            content={task.requirements}
            content_label="Requirement"
            start_with_one={false}
            hide_add_button={true}
            empty_content={{
              name: "",
              description: "",
              instruction: "",
              priority: 1,
            }}
            let:item_index
          >
            <div class="flex flex-col gap-3">
              <div class="flex flex-row gap-1">
                <div class="grow flex flex-col gap-1">
                  <FormElement
                    label="Requirement Name"
                    info_description="A short name to uniquely identify the requirement. This will appear in the rating UI. It's not used by the model."
                    id="requirement_name_{item_index}"
                    light_label={true}
                    bind:value={task.requirements[item_index].name}
                    max_length={32}
                  />
                </div>
                <div class="flex flex-col gap-1">
                  <FormElement
                    label="Rating Type"
                    inputType="select"
                    id="requirement_type_{item_index}"
                    light_label={true}
                    select_options={[
                      ["five_star", "5 Star"],
                      ["pass_fail", "Pass / Fail"],
                      ["pass_fail_critical", "Pass / Fail / Critical"],
                    ]}
                    bind:value={task.requirements[item_index].type}
                  />
                </div>
                <div class="flex flex-col gap-1">
                  <FormElement
                    label="Priority"
                    inputType="select"
                    id="requirement_priority_{item_index}"
                    light_label={true}
                    select_options={[
                      [0, "P0 - Critical"],
                      [1, "P1 - High"],
                      [2, "P2 - Medium"],
                      [3, "P3 - Low"],
                    ]}
                    bind:value={task.requirements[item_index].priority}
                  />
                </div>
              </div>
              <div class="grow flex flex-col gap-1">
                <FormElement
                  label="Instructions: a few sentences describing this requirement for the model"
                  info_description="These instructions will be appended to the prompt and using during evals. It should be written with the model in mind. Example requirement: Name='Be Succinct' and Instruction='Use short sentences and simple language. Don't repeat yourself.'"
                  inputType="textarea"
                  id="requirement_instructions_{item_index}"
                  light_label={true}
                  bind:value={task.requirements[item_index].instruction}
                />
              </div>
            </div>
          </FormList>
        </Collapse>
      </div>
    {/if}
  </FormContainer>
</div>
