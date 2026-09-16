import type { Trace, RunChainEntry } from "$lib/types"

export type ForkTarget = {
  turn_index: number
  parent_run_id: string | null
  trace_index: number
  prefill: string
}

// Compute, for each trace index, the run id used to fork at that point. The
// fork affordance lives on the assistant message that ends a turn (forking
// "after" the assistant continues the conversation down a new branch). For a
// forkable turn K (K >= 2), we map the run id of turn K onto the assistant
// message immediately preceding the turn's first message (i.e. turn K-1's
// final assistant message), using the server-computed trace_start_index —
// chat strategies vary in how many user messages a turn emits (chain-of-
// thought adds a second one), so message-role counting can't locate turns.
// The result has the same length as `trace`; every other index is null. Turn
// 1 is never forkable: it's either the true root (no parent) or the first
// entry of a broken chain (parent missing, boundary unknown).
export function compute_forkable_run_ids(
  trace: Trace,
  chain: RunChainEntry[],
): (string | null)[] {
  const result: (string | null)[] = trace.map(() => null)
  for (const entry of chain) {
    if (entry.turn_index === 1) continue // turn 1 is not forkable
    const start = entry.trace_start_index
    // Unknown boundary, or one that doesn't point into the trace: skip.
    if (start == null || start <= 0 || start >= trace.length) continue
    // Place the fork affordance on the assistant message immediately
    // preceding this turn's first message (the previous turn's final
    // assistant response), not on the turn's own messages.
    const assistant_idx = start - 1
    if (trace[assistant_idx]?.role === "assistant") {
      result[assistant_idx] = entry.run_id
    }
  }
  return result
}

// Look up the data needed to open a fork composer for a clicked assistant
// block. `run_id` is the run mapped onto that assistant message by
// compute_forkable_run_ids — the turn that the new branch will create
// (turn K). Forking continues the conversation after the clicked assistant
// message with a fresh (un-seeded) next message, so prefill is always empty.
// Returns null if the click target can't be resolved (defensive — the trace
// component only renders the fork button when a chain entry was mapped).
export function fork_target_from_assistant_block(
  run_id: string,
  trace_index: number,
  chain: RunChainEntry[],
): ForkTarget | null {
  const this_turn = chain.find((c) => c.run_id === run_id)
  if (!this_turn) return null
  // The mapped run is always turn 2+ (turn 1 is filtered out upstream);
  // guard defensively anyway.
  if (this_turn.turn_index <= 1) return null
  const parent = chain.find((c) => c.turn_index === this_turn.turn_index - 1)
  return {
    turn_index: this_turn.turn_index,
    parent_run_id: parent?.run_id ?? null,
    // Truncate the displayed transcript just after the clicked assistant
    // message, so it stays visible while everything that followed is hidden.
    trace_index: trace_index + 1,
    // No seeding: forking from an assistant message starts a brand-new turn.
    prefill: "",
  }
}
