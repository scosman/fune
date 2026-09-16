<script lang="ts">
  // The batch-progress readout every waiting screen on this step renders: the
  // bar, with its count on a small grey line of its own underneath.
  //
  // The count sits here rather than inside the animation's title or its
  // description because a number inside a sentence re-lays the sentence out on
  // every tick — the line changes length, and on a narrow window it re-wraps.
  // Under the bar it changes in place, and the sentence above holds still.
  // Failures ride the same line, for the same reason. One component, so the
  // waiting screens on this step cannot drift apart.
  export let value: number
  export let max: number
  // The words after the count: "3 of 20 judged", "12 of up to 60 turns".
  export let noun: string
  // What the bar fills to, where that differs from the count above it. The
  // drive counts a failed case as finished — it is done, just not judged — so
  // the bar can reach full while the count beside it stays short.
  export let bar_value: number | null = null
  // The denominator is a ceiling rather than a total: a conversation that ends
  // early spends fewer turns, so the bar can finish short of full.
  export let max_is_ceiling = false
  export let failed = 0
</script>

<div class="flex flex-col items-center mt-6">
  <progress
    class="progress w-56 progress-success"
    value={bar_value ?? value}
    {max}
  ></progress>
  <!-- No denominator yet (the total lands with the first frame): the bar
       carries the wait on its own rather than showing "0 of 0". -->
  {#if max > 0}
    <div data-progress-count class="text-xs text-gray-500 text-center mt-1">
      {value} of {max_is_ceiling ? "up to " : ""}{max}
      {noun}{failed > 0 ? ` — ${failed} failed` : ""}
    </div>
  {/if}
</div>
