<script lang="ts">
  // The "Create a Data Guide" offer, shared by synthetic data generation and
  // the eval builder. Both generate task inputs, and both open by offering a
  // guide before generating without one, so the copy, the buttons and the
  // analytics event are one component; `surface` is what tells the two apart
  // in analytics. Each surface decides what its two buttons do.
  import Intro from "$lib/ui/intro.svelte"
  import NotebookIcon from "$lib/ui/icons/notebook_icon.svelte"
  import posthog from "posthog-js"

  export let surface: "synth" | "builder"
  export let on_set_up: () => void
  export let on_skip: () => void
</script>

<div class="flex flex-col items-center justify-center min-h-[50vh] mt-12">
  <Intro
    title="Create a Data Guide"
    description_paragraphs={[
      "A Data Guide tells us what realistic inputs to your task look like. Without one, the model might guess.",
      "Add examples, rate the data we generate, and we'll refine the guide from there.",
    ]}
    action_buttons={[
      {
        label: "Set Up Data Guide",
        is_primary: true,
        onClick: () => {
          posthog.capture("data_guide_intro_clicked", {
            choice: "set_up",
            surface,
          })
          on_set_up()
        },
      },
      {
        label: "Continue Without Data Guide",
        is_primary: false,
        onClick: () => {
          posthog.capture("data_guide_intro_clicked", {
            choice: "skip",
            surface,
          })
          on_skip()
        },
      },
    ]}
  >
    <div slot="icon" class="h-12 w-12">
      <NotebookIcon />
    </div>
  </Intro>
</div>
