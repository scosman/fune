<script lang="ts">
  import type {
    AnswerOptionWithSelection,
    QuestionSet,
    QuestionWithAnswer,
    SpecType,
  } from "$lib/types"
  import FormContainer from "$lib/utils/form_container.svelte"
  import FormElement from "$lib/utils/form_element.svelte"
  import { KilnError } from "$lib/utils/error_handlers"
  import Dialog from "$lib/ui/dialog.svelte"
  import SpecPropertiesDisplay from "../spec_properties_display.svelte"

  export let name: string
  export let spec_type: SpecType
  export let property_values: Record<string, string | null>
  export let question_set: QuestionSet
  export let on_submit: (questions_and_answers: QuestionWithAnswer[]) => void
  export let error: KilnError | null
  export let submitting: boolean
  export let warn_before_unload: boolean
  // Overridable so consumers can match their flow's advance-button wording.
  export let submit_label = "Continue"

  // Track the selected option for each question (bound to parent to survive remounts)
  // "other" means the user selected the "Other" option
  // number means the index of the selected predefined option
  // null means no selection
  export let selections: (number | "other" | null)[] =
    question_set.questions.map(() => null)

  // Track the "Other" text for each question (bound to parent to survive remounts)
  export let other_texts: string[] = question_set.questions.map(() => "")

  function validate(): string | null {
    for (let i = 0; i < question_set.questions.length; i++) {
      const selection = selections[i]
      if (selection === null) {
        return `Please answer question ${i + 1}`
      }
      if (selection === "other" && !other_texts[i].trim()) {
        return `Please provide feedback for question ${i + 1}`
      }
    }
    return null
  }

  function build_question_answers(): QuestionWithAnswer[] {
    return question_set.questions.map((question, q_index) => {
      const selection = selections[q_index]
      const is_other = selection === "other"

      const answer_options: AnswerOptionWithSelection[] =
        question.answer_options.map((option, o_index) => ({
          answer_title: option.answer_title,
          answer_description: option.answer_description,
          selected: !is_other && selection === o_index,
        }))

      const result: QuestionWithAnswer = {
        question_title: question.question_title,
        question_body: question.question_body,
        answer_options,
      }

      if (is_other) {
        result.custom_answer = other_texts[q_index].trim()
      }

      return result
    })
  }

  async function handle_submit() {
    const validation_error = validate()
    if (validation_error) {
      error = new KilnError(validation_error, null)
      submitting = false
      return
    }
    // Clear any earlier validation error once validation passes — error is
    // bound to the parent to survive remounts, so a stale message would
    // otherwise reappear if the user navigates back to this form.
    error = null

    const questions_and_answers = build_question_answers()
    await on_submit(questions_and_answers)
  }

  function select_option(question_index: number, option_index: number) {
    selections[question_index] = option_index
    selections = selections
  }

  function select_other(question_index: number) {
    selections[question_index] = "other"
    selections = selections
  }

  let spec_details_dialog: Dialog | null = null
  function open_details_dialog() {
    spec_details_dialog?.show()
  }
</script>

<div class="max-w-4xl">
  <FormContainer
    {submit_label}
    compact_button={true}
    on:submit={handle_submit}
    bind:error
    bind:submitting
    focus_on_mount={false}
    {warn_before_unload}
  >
    <div class="flex flex-col">
      <div class="font-medium">Answer Clarifying Questions</div>
      <div class="font-light text-gray-500 text-sm">
        <!-- The eval may be unnamed at this point (the v2 builder names it a
             step later) — only render the details link when a name exists. -->
        Your answers to these questions will help Kiln refine your eval{#if name.trim()}:
          <button
            class="link text-sm text-left text-gray-500 hover:text-gray-700"
            on:click={open_details_dialog}>{name}</button
          >{/if}.
      </div>
    </div>
    <div class="border-t" />
    <div class="flex flex-col gap-14">
      {#each question_set.questions as question, q_index}
        <div class="flex flex-col">
          <!-- Header row -->
          <h3 class="text-base font-medium pb-2">
            Question {q_index + 1}: {question.question_title}
          </h3>

          <!-- Content row: body on left, options on right -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-8">
            <!-- Left column: Question body -->
            <p class="text-sm text-gray-500">{question.question_body}</p>

            <!-- Right column: Options -->
            <div class="flex flex-col gap-3">
              {#each question.answer_options as option, o_index}
                <label class="flex items-start gap-3 cursor-pointer group">
                  <input
                    type="radio"
                    name="question-{q_index}"
                    class="radio radio-sm mt-0.5"
                    checked={selections[q_index] === o_index}
                    on:change={() => select_option(q_index, o_index)}
                  />
                  <div class="flex flex-col">
                    <span class="text-sm font-medium"
                      >{option.answer_title}</span
                    >
                    <span class="text-xs text-gray-500"
                      >{option.answer_description}</span
                    >
                  </div>
                </label>
              {/each}

              <label class="flex items-start gap-3 cursor-pointer group">
                <input
                  type="radio"
                  name="question-{q_index}"
                  class="radio radio-sm mt-0.5"
                  checked={selections[q_index] === "other"}
                  on:change={() => select_other(q_index)}
                />
                <div class="flex flex-col grow">
                  <span class="text-sm font-medium">Other</span>
                  <span
                    class="text-xs text-gray-500 {selections[q_index] ===
                    'other'
                      ? 'hidden'
                      : ''}">Provide a custom answer to this question.</span
                  >
                </div>
              </label>

              {#if selections[q_index] === "other"}
                <div class="ml-9">
                  <FormElement
                    id="other-text-{q_index}"
                    inputType="textarea"
                    bind:value={other_texts[q_index]}
                    label="Custom Answer"
                    placeholder="Enter a custom answer to the question."
                    height="medium"
                    aria_label="Custom answer for question {q_index + 1}"
                  />
                </div>
              {/if}
            </div>
          </div>
        </div>
      {/each}
    </div>
  </FormContainer>
</div>

<Dialog
  bind:this={spec_details_dialog}
  title={`Eval: ${name}`}
  width="wide"
  action_buttons={[
    {
      label: "Close",
      isCancel: true,
    },
  ]}
>
  <SpecPropertiesDisplay {spec_type} properties={property_values} />
</Dialog>
