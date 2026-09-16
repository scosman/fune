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

  $: props = extractV2Props(eval_config, "tool_call_check")

  // Deterministic types emit one binary value per declared output score, keyed
  // by the eval's spec-name json_keys (not a literal "match"). All values are
  // identical, so the badge passes when every present score is 1.0.
  $: score_values = Object.values(scores)
  $: has_score = score_values.length > 0
  $: passed = has_score && score_values.every((v) => v === 1.0)

  const match_mode_labels: Record<string, string> = {
    any: "Any expected tool called",
    all: "All expected tools called",
    ordered: "All expected tools called in order",
    never: "None of the listed tools called",
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
      <div>
        Match mode: {match_mode_labels[props.match_mode] ?? props.match_mode}
      </div>
      {#if props.expected_tools.length > 0}
        <div>
          Tools: {props.expected_tools.map((t) => t.tool_name).join(", ")}
        </div>
      {/if}
      {#if props.on_unexpected_tools === "fail"}
        <div>Fails on unexpected tool calls</div>
      {/if}
    </div>
  {/if}
</div>
