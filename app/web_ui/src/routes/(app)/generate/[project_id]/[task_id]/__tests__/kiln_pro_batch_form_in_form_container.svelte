<script lang="ts">
  // Mounts the batch form rows inside a REAL FormContainer so a test can click
  // the submit button and see what a caller's form actually does with the
  // rows: whether validation blocks the click or the submit event fires.
  // A wrapper is needed because the rows are slot content, which
  // testing-library can't pass to FormContainer directly.
  import FormContainer from "$lib/utils/form_container.svelte"
  import KilnProBatchForm from "../kiln_pro_batch_form.svelte"

  export let count: number = 50
  export let guidance: string = ""
  // Left undefined unless a test sets it, so the rows fall back to their own
  // default and a test can pin what an unconfigured caller gets.
  export let guidance_optional: boolean | undefined = undefined
</script>

<FormContainer submit_label="Submit" focus_on_mount={false} on:submit>
  <KilnProBatchForm bind:count bind:guidance {guidance_optional} />
</FormContainer>
