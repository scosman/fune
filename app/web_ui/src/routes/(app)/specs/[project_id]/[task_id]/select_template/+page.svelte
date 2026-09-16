<script lang="ts">
  import { goto } from "$app/navigation"
  import { page } from "$app/stores"
  import { onMount, type ComponentType } from "svelte"
  import AppPage from "../../../../app_page.svelte"
  import OptionList from "$lib/ui/option_list.svelte"
  import type { OptionListItem } from "$lib/ui/option_list_types"
  import SettingsHeader from "$lib/ui/settings_header.svelte"
  import { formatSpecTypeName } from "$lib/utils/formatters"
  import { spec_categories } from "./spec_templates"
  import type { SpecTemplateData } from "./spec_templates"
  import {
    next_page_after_template,
    judge_only_builder_url,
  } from "../spec_utils"
  import { agentInfo } from "$lib/agent"
  import Intro from "$lib/ui/intro.svelte"
  import EvalIcon from "$lib/ui/icons/eval_icon.svelte"
  import FormElement from "$lib/utils/form_element.svelte"
  import { kilnCopilotConnected } from "$lib/stores/copilot_connection_store"
  import { getV2EvalTypeMetadata } from "$lib/utils/eval_types/registry"
  import type { V2EvalType } from "$lib/utils/eval_types/registry"
  import { getEvalTypeIconComponent } from "$lib/components/eval_types/eval_type_icon.svelte"
  import type { SpecType, Task, TaskRunConfig } from "$lib/types"
  // The eval is authored against one run config: its prompt, tools and skills
  // shape the questions, the generated data and the judge. Asked on the way
  // into the builder so the choice is explicit instead of the task default
  // applying silently once the wizard is already running.
  import SavedRunConfigsDropdown from "$lib/ui/run_config_component/saved_run_configs_dropdown.svelte"
  import RunConfigComponent from "$lib/ui/run_config_component/run_config_component.svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import FormContainer from "$lib/utils/form_container.svelte"
  import { load_task } from "$lib/stores"
  import { createKilnError, type KilnError } from "$lib/utils/error_handlers"

  import DesiredBehaviourIcon from "$lib/ui/icons/spec_types/desired_behaviour_icon.svelte"
  import IssueIcon from "$lib/ui/icons/spec_types/issue_icon.svelte"
  import ToxicityIcon from "$lib/ui/icons/spec_types/toxicity_icon.svelte"
  import QnaIcon from "$lib/ui/icons/qna_icon.svelte"
  import BookIcon from "$lib/ui/icons/book_icon.svelte"
  import ScalesIcon from "$lib/ui/icons/scales_icon.svelte"
  import ShieldIcon from "$lib/ui/icons/shield_icon.svelte"
  import KeyIcon from "$lib/ui/icons/key_icon.svelte"

  // ### Spec Template Select ###

  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  $: agentInfo.set({
    name: "Select Eval Type",
    description: `Select an eval type as part of the eval creation process for project ID ${project_id}, task ID ${task_id}. Choose from available eval types.`,
  })

  function select_template(template_data: SpecTemplateData) {
    goto(next_page_after_template(project_id, task_id, template_data.spec_type))
  }

  // The programmatic checks section: judges chosen directly, without a
  // template. All of them create spec-less, template-less evals via the spec
  // builder's judge-only mode — the judge form collects any config it needs
  // (e.g. the tool list for Tool Call Check). The legacy "tool_call" template
  // is reserved for pre-spec LLM tool evals and is never recorded on new ones.
  const programmatic_judge_types: V2EvalType[] = [
    "code_eval",
    "tool_call_check",
    "exact_match",
    "pattern_match",
    "contains",
    "set_check",
    "step_count_check",
  ]

  const template_icons: Partial<Record<SpecType, ComponentType>> = {
    desired_behaviour: DesiredBehaviourIcon,
    issue: IssueIcon,
    reference_answer_accuracy: QnaIcon,
    factual_correctness: BookIcon,
    toxicity: ToxicityIcon,
    bias: ScalesIcon,
    maliciousness: ShieldIcon,
    jailbreak: KeyIcon,
  }

  const all_templates = spec_categories.flatMap(
    (category) => category.templates,
  )
  // Tool use is offered as the Tool Call Check programmatic judge, not an LLM
  // template.
  const llm_templates = all_templates.filter(
    (t) => t.spec_type !== "appropriate_tool_use",
  )

  const llm_options: OptionListItem[] = llm_templates.map((template_data) => ({
    id: template_data.spec_type,
    name: formatSpecTypeName(template_data.spec_type),
    description: template_data.description,
    icon: template_icons[template_data.spec_type],
  }))

  const programmatic_options: OptionListItem[] = programmatic_judge_types.map(
    (judge_type) => {
      const metadata = getV2EvalTypeMetadata(judge_type)
      return {
        id: judge_type,
        name: metadata.label,
        description: metadata.description,
        icon: getEvalTypeIconComponent(judge_type),
        tags: metadata.tags,
      }
    },
  )

  function select_llm_option(id: string) {
    const template_data = llm_templates.find((t) => t.spec_type === id)
    if (template_data) {
      select_template(template_data)
    }
  }

  function select_programmatic_option(id: string) {
    goto(judge_only_builder_url(project_id, task_id, id as V2EvalType))
  }

  // PREVIEW (09-03): the entry restructure. Both user types land here instead
  // of Pro users being sent straight to the builder, so the eval type and the
  // free-text description are one decision rather than two screens.
  //
  // The textbox is the main UI for anyone with Copilot: it is the builder's
  // first step, hoisted onto this page so the templates and the programmatic
  // checks are visible beside it rather than behind a link that reads like
  // opting out. Without Copilot there is nothing to describe to, so the
  // template list stays the primary choice and this section is absent.
  let description = ""

  // The run config the eval is written against, asked in a dialog on the way
  // into the builder rather than as another field on this form: it is a
  // confirmation of what the eval is about, not a second thing to fill in.
  // The picker is the run page's, dropdown over component, so the same control
  // chooses a config in both places and its model, tools and skills are
  // visible while choosing. The dropdown opens on the last config used for
  // this task, or the task default, exactly as it does on the run page.
  let task: Task | null = null
  let target_run_config_id: string | null = null
  let run_config_dialog: Dialog | null = null
  let run_config_component: RunConfigComponent | null = null
  let run_config_submitting = false
  let run_config_error: KilnError | null = null
  let save_config_error: KilnError | null = null
  let set_default_error: KilnError | null = null

  onMount(async () => {
    try {
      task = await load_task(project_id, task_id)
    } catch {
      // No task means no picker. Continue then hands over without a config
      // and the builder resolves the task default, which is what it did
      // before this page asked at all.
      task = null
    }
  })

  // The dropdown's own save path, so its "Save current options" action and
  // Continue below save the same way.
  async function handle_save_new_run_config(): Promise<TaskRunConfig> {
    if (!run_config_component) {
      throw new Error("Run configuration component is not loaded")
    }
    return await run_config_component.save_new_run_config()
  }

  // Without Copilot the page opens on the offer rather than the templates.
  // The Pro-vs-manual question is asked once, up front, instead of partway
  // through after a template is already chosen. Choosing manual reveals the
  // same list a Copilot user gets as the page's third section.
  let chose_manual = false
  // The offer is showing, as opposed to the picker behind it. Named once so
  // the page title and the body can never describe different screens.
  $: show_offer = $kilnCopilotConnected !== true && !chose_manual

  // "or use templates" moves down the page rather than navigating: the
  // description the user may have typed stays put.
  function scroll_to_templates() {
    document
      .getElementById("llm_judge_templates")
      ?.scrollIntoView({ behavior: "smooth", block: "start" })
  }

  function connect_kiln_pro() {
    goto("/specs/pro_auth")
  }

  // The handover into the wizard. The chosen config travels beside the
  // description; null means nothing was chosen and the builder resolves the
  // task default.
  function goto_builder(run_config_id: string | null) {
    const text = description.trim()
    const config_param = run_config_id
      ? `&run_config_id=${encodeURIComponent(run_config_id)}`
      : ""
    goto(
      `/specs/${project_id}/${task_id}/builder` +
        `?description=${encodeURIComponent(text)}` +
        config_param,
    )
  }

  // Continue asks which run config the eval is about before it hands the
  // description over. The description stays on the page, so closing the dialog
  // loses nothing.
  function open_run_config_dialog() {
    if (!description.trim()) return
    // A failure on a previous attempt is about that attempt, not this one.
    run_config_error = null
    // Without a task there is nothing to pick from, so opening would show an
    // empty dialog. Hand over as this page did before it asked.
    if (!task) {
      goto_builder(null)
      return
    }
    run_config_dialog?.show()
  }

  // Continue needs a SAVED config: the wizard, the drive and the saved eval all
  // refer to it by id, and "Custom" is only in the picker's local state. An
  // edited selection is therefore saved through the dropdown's own save path
  // first, which names it the way saving from the run page does.
  async function continue_with_description() {
    const text = description.trim()
    if (!text) return
    run_config_error = null
    run_config_submitting = true
    try {
      let run_config_id = target_run_config_id
      if (!run_config_id || run_config_id === "custom") {
        const saved = await handle_save_new_run_config()
        run_config_id = saved.id ?? null
        target_run_config_id = run_config_id
      }
      if (!run_config_id) {
        throw new Error("The saved run config has no id.")
      }
      goto_builder(run_config_id)
    } catch (e) {
      run_config_error = createKilnError(e)
    } finally {
      run_config_submitting = false
    }
  }
</script>

<div class="max-w-[1400px]">
  <AppPage
    title="Create Eval"
    subtitle={show_offer
      ? "Kiln Pro drafts the eval for you, or set one up yourself."
      : $kilnCopilotConnected === true
        ? "Three ways to create an eval"
        : "Two ways to create an eval"}
    breadcrumbs={[
      {
        label: "Evals",
        href: `/specs/${project_id}/${task_id}`,
      },
    ]}
  >
    {#if show_offer}
      <!-- The offer, before the templates. This is also the only place eval
           creation pitches Kiln Pro: the screen that used to carry that pitch
           sat partway through the old flow and is gone. -->
      <div class="flex justify-center mt-[10vh]">
        <Intro
          title="Let Kiln Pro write the eval"
          description_paragraphs={[
            "Describe what to check in plain language. Kiln Pro writes the eval, generates test data to run it on, and checks its judge against your own review.",
            "Or set one up yourself from a template.",
          ]}
          action_buttons={[
            {
              label: "Use Kiln Pro",
              onClick: connect_kiln_pro,
              is_primary: true,
            },
            {
              label: "Set Up Manually",
              onClick: () => (chose_manual = true),
              is_primary: false,
            },
          ]}
        >
          <div slot="icon" class="h-12 w-12">
            <EvalIcon />
          </div>
        </Intro>
      </div>
    {:else}
      <div class="pt-6 max-w-5xl flex flex-col gap-10">
        {#if $kilnCopilotConnected === true}
          <div class="flex flex-col gap-4">
            <SettingsHeader
              title="LLM Judge Assistant"
              subtitle="Describe what to evaluate in plain language. Kiln Pro writes the eval and generates the dataset."
            />
            <FormElement
              label="What should this eval check?"
              placeholder="e.g. Off-topic requests should be politely declined."
              id="eval_description"
              inputType="textarea"
              height="medium"
              bind:value={description}
            />
            <!-- The primary and its quiet alternative read as one action
                 group: someone who already knows the template they want
                 jumps to the list below instead of describing anything. -->
            <div class="flex flex-col items-end gap-1">
              <button
                class="btn btn-primary min-w-48"
                disabled={!description.trim()}
                on:click={open_run_config_dialog}
              >
                Write My Eval
              </button>
              <button
                type="button"
                class="link underline text-sm text-gray-500"
                on:click={scroll_to_templates}
              >
                or use templates
              </button>
            </div>
          </div>
        {:else}
          <div class="flex flex-col gap-4">
            <SettingsHeader title="LLM Judge Templates" />
            <OptionList
              options={llm_options}
              select_option={select_llm_option}
              two_columns={true}
              two_line_descriptions={true}
            />
          </div>
        {/if}
        <div class="flex flex-col gap-4">
          <SettingsHeader title="Programmatic Checks" />
          <OptionList
            options={programmatic_options}
            select_option={select_programmatic_option}
            two_columns={true}
            two_line_descriptions={true}
          />
        </div>
        {#if $kilnCopilotConnected === true}
          <!-- The same list the Pro-off arm opens on, third here so the
               assistant leads. Its id is what "or use templates" scrolls to. -->
          <div id="llm_judge_templates" class="flex flex-col gap-4">
            <SettingsHeader title="LLM Judge Templates" />
            <OptionList
              options={llm_options}
              select_option={select_llm_option}
              two_columns={true}
              two_line_descriptions={true}
            />
          </div>
        {/if}
      </div>
    {/if}
  </AppPage>

  <!-- The last stop before the wizard: the run page's own picker, trimmed to
       the rows the builder's own generation lane shows. Everything else folds
       into Advanced Options. -->
  <Dialog
    bind:this={run_config_dialog}
    title="Choose Run Config"
    subtitle="Choose how your task will be run for generating examples in the eval builder."
  >
    <FormContainer
      submit_label="Continue"
      bind:submitting={run_config_submitting}
      error={run_config_error}
      on:submit={continue_with_description}
      keyboard_submit={false}
    >
      {#if task}
        <SavedRunConfigsDropdown
          {project_id}
          current_task={task}
          bind:selected_run_config_id={target_run_config_id}
          bind:save_config_error
          bind:set_default_error
          info_description="The run config your task runs with in the eval builder. Kiln uses its tools and skills to write the questions and the judge, then runs it to create the eval dataset."
          save_new_run_config={handle_save_new_run_config}
        />
        <RunConfigComponent
          bind:this={run_config_component}
          {project_id}
          current_task={task}
          bind:selected_run_config_id={target_run_config_id}
          bind:set_default_error
          requires_structured_output={!!task.output_json_schema}
          show_name_field={false}
          hide_prompt_selector={true}
          show_tools_selector_in_advanced={true}
        />
      {/if}
    </FormContainer>
  </Dialog>
</div>
