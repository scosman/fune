<script lang="ts">
  // PREVIEW (09-02): Step 5's entry screen. Reviewers arrived straight into
  // a stack of grading with no statement of why they were doing it. This
  // says what the step is for once, and gets out of the way.
  //
  // Uses the shared Intro control (the app's empty/entry screen), so this
  // screen inherits the same width, type scale, icon size and button
  // treatment as every other one rather than restating them here.
  //
  // Wording is the reviewer's, unchanged. The one break he wrote as a newline
  // is two paragraphs because that is how this control expresses one. The
  // noun is the only substitution: he wrote "examples", which is wrong on the
  // multi-turn arm where every item is a conversation.
  import Intro from "$lib/ui/intro.svelte"
  import ScalesIcon from "$lib/ui/icons/scales_icon.svelte"

  export let on_start: () => void
  export let judged_noun: string = "example"
</script>

<!-- Centred with flex rather than mx-auto: the step body is a full-width
     block here, so mx-auto on the wrapper has nothing to centre against and
     Intro's own 300px column would sit against the left edge. -->
<div class="flex justify-center mt-[10vh]">
  <Intro
    title="Validating the Judge"
    description_paragraphs={[
      "Let's confirm your judge is aligned to your expectations.",
      `We'll show a set of ${judged_noun}s, and you tell us if you agree with its judgement.`,
    ]}
    action_buttons={[
      {
        label: "Start",
        onClick: on_start,
        is_primary: true,
      },
    ]}
  >
    <div slot="icon" class="h-12 w-12">
      <ScalesIcon />
    </div>
  </Intro>
</div>
