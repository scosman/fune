<script lang="ts">
  // Claim or overview prose with its inline [n] citations. A marker that
  // resolves to a citation renders as a chip that opens the trace at the
  // cited span; a marker with no citation stays as the plain text the model
  // wrote (see tokenize_claim_text). Renders inline content only, so the
  // caller owns the paragraph and its typography.
  import { tokenize_claim_text, type Citation } from "./claim_evidence"

  export let text: string
  export let citations: Citation[]
  export let on_cite: (citation: Citation) => void

  $: tokens = tokenize_claim_text(text, citations)
</script>

{#each tokens as token}{#if token.kind === "text"}{#each token.value.split("\n") as line, i}{#if i > 0}<br
        />{/if}{line}{/each}{:else}<button
      type="button"
      class="align-super text-xs text-primary hover:underline font-medium mx-0.5"
      on:click={() => on_cite(token.citation)}
      title="View in trace">[{token.n}]</button
    >{/if}{/each}
