<script lang="ts">
  import EvalResultScores from "./eval_result_scores.svelte"
  import CheckCircleIcon from "$lib/ui/icons/check_circle_icon.svelte"
  import XCircleIcon from "$lib/ui/icons/x_circle_icon.svelte"
  import type { EvalConfig } from "$lib/types"
  import { extractV2Props } from "$lib/utils/eval_types/registry"

  export let scores: Record<string, number> = {}
  export let skipped_reason: string | null = null
  export let skipped_detail: string | null = null
  export let eval_config: EvalConfig | null = null

  $: props = extractV2Props(eval_config, "contains")

  // Deterministic types emit one binary value per declared output score, keyed
  // by the eval's spec-name json_keys (not a literal "match"). All values are
  // identical, so the badge passes when every present score is 1.0.
  $: score_values = Object.values(scores)
  $: has_score = score_values.length > 0
  $: passed = has_score && score_values.every((v) => v === 1.0)

  function format_mode(mode: string): string {
    return mode === "must_contain" ? "Must contain" : "Must not contain"
  }
</script>

<div class="flex flex-col gap-2">
  {#if has_score && !skipped_reason}
    <div>
      {#if passed}
        <span class="badge badge-success badge-sm gap-1">
          <div class="w-3 h-3"><CheckCircleIcon /></div>
          Pass
        </span>
      {:else}
        <span class="badge badge-error badge-sm gap-1">
          <div class="w-3 h-3"><XCircleIcon /></div>
          Fail
        </span>
      {/if}
    </div>
  {/if}

  <EvalResultScores {scores} {skipped_reason} {skipped_detail} />

  {#if props}
    <div class="text-xs text-gray-400 flex flex-col gap-0.5">
      <div>Mode: {format_mode(props.mode)}</div>
      {#if props.substring != null}
        <div>
          Substring: <span class="font-mono bg-base-200 px-1 rounded"
            >{props.substring}</span
          >
        </div>
      {:else if props.reference_key != null}
        <div>
          Reference key: <span class="font-mono bg-base-200 px-1 rounded"
            >{props.reference_key}</span
          >
        </div>
      {/if}
      {#if !props.case_sensitive}
        <div>Case insensitive</div>
      {/if}
      {#if props.value_expression}
        <div>
          Expression: <span class="font-mono bg-base-200 px-1 rounded"
            >{props.value_expression}</span
          >
        </div>
      {/if}
    </div>
  {/if}
</div>
