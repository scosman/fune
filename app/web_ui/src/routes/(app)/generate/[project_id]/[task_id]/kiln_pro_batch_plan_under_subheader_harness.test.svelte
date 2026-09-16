<script lang="ts">
  // Test-only harness: Svelte slots cannot be passed through testing-library's
  // props, so the per-consumer clause under the sub-line is rendered through a
  // component that fills it.
  import KilnProBatchPlan from "./kiln_pro_batch_plan.svelte"
  export let plan: { prompts: string[]; summary: string }
</script>

<KilnProBatchPlan
  {plan}
  on_generate_inputs={() => {}}
  on_regenerate={() => {}}
  on_delete_prompt={() => {}}
>
  <svelte:fragment slot="under_subheader">
    <!-- Mirrors how a consumer has to write this clause: the leading {" "} is
         the only thing that keeps it off the sub-line's last word, because
         Svelte drops whitespace at the start of slot content. -->
    <span data-under-subheader>{" "}Planned using your data guide.</span>
  </svelte:fragment>
</KilnProBatchPlan>
