import { describe, it, expect } from "vitest"
import {
  compute_forkable_run_ids,
  fork_target_from_assistant_block,
} from "./fork_helpers"
import type { Trace, TraceMessage, RunChainEntry } from "$lib/types"

function userMsg(content: string): TraceMessage {
  return { role: "user", content } as TraceMessage
}
function assistantMsg(content: string): TraceMessage {
  return { role: "assistant", content } as TraceMessage
}
function systemMsg(content: string): TraceMessage {
  return { role: "system", content } as TraceMessage
}

describe("compute_forkable_run_ids", () => {
  it("maps each non-turn-1 chain entry onto the assistant before its turn start for a clean 3-turn chain", () => {
    const trace: Trace = [
      systemMsg("s"),
      userMsg("u1"),
      assistantMsg("a1"),
      userMsg("u2"),
      assistantMsg("a2"),
      userMsg("u3"),
      assistantMsg("a3"),
    ]
    const chain: RunChainEntry[] = [
      { run_id: "run-1", turn_index: 1, trace_start_index: 0 },
      { run_id: "run-2", turn_index: 2, trace_start_index: 3 },
      { run_id: "run-3", turn_index: 3, trace_start_index: 5 },
    ]
    const result = compute_forkable_run_ids(trace, chain)
    // The fork affordance sits on the assistant message that precedes each
    // forkable turn's start: turn 2 (run-2) maps onto assistant a1 (index 2)
    // and turn 3 (run-3) maps onto assistant a2 (index 4). The leaf's trailing
    // assistant a3 (index 6) is not forkable.
    expect(result).toEqual([null, null, "run-2", null, "run-3", null, null])
  })

  it("uses the turn boundary, not user-message counting, for a chain-of-thought root", () => {
    // A CoT root turn contains TWO user messages (the real input and the
    // final-answer prompt); the server anchors turn 2 after the full 5-message
    // root trace.
    const trace: Trace = [
      systemMsg("s"),
      userMsg("u1 + thinking instructions"),
      assistantMsg("thinking"),
      userMsg("final answer prompt"),
      assistantMsg("a1"),
      userMsg("u2"),
      assistantMsg("a2"),
    ]
    const chain: RunChainEntry[] = [
      { run_id: "run-1", turn_index: 1, trace_start_index: 0 },
      { run_id: "run-2", turn_index: 2, trace_start_index: 5 },
    ]
    const result = compute_forkable_run_ids(trace, chain)
    // run-2 maps onto a1 (index 4) — the root's final answer, not the
    // mid-turn thinking assistant.
    expect(result).toEqual([null, null, null, null, "run-2", null, null])
  })

  it("does not map the first entry of a broken chain (parent missing, boundary unknown)", () => {
    const trace: Trace = [
      systemMsg("s"),
      userMsg("u1"),
      assistantMsg("a1"),
      userMsg("u2"),
      assistantMsg("a2"),
      userMsg("u3"),
      assistantMsg("a3"),
    ]
    // The chain is broken above the last two runs: the suffix root's boundary
    // is unknown (and its parent is gone, so it can't be forked); the leaf is
    // still anchored to its turn start.
    const chain: RunChainEntry[] = [
      { run_id: "run-2", turn_index: 1, trace_start_index: null },
      { run_id: "run-3", turn_index: 2, trace_start_index: 5 },
    ]
    const result = compute_forkable_run_ids(trace, chain)
    expect(result).toEqual([null, null, null, null, "run-3", null, null])
  })

  it("returns all nulls when chain is empty", () => {
    const trace: Trace = [systemMsg("s"), userMsg("u1"), assistantMsg("a1")]
    const result = compute_forkable_run_ids(trace, [])
    expect(result).toEqual([null, null, null])
  })

  it("ignores a turn start pointing outside the trace", () => {
    const trace: Trace = [systemMsg("s"), userMsg("u1"), assistantMsg("a1")]
    const chain: RunChainEntry[] = [
      { run_id: "run-1", turn_index: 1, trace_start_index: 0 },
      { run_id: "run-2", turn_index: 2, trace_start_index: 3 },
    ]
    const result = compute_forkable_run_ids(trace, chain)
    expect(result).toEqual([null, null, null])
  })

  it("skips turn 1 even when chain includes it (single-turn chain)", () => {
    const trace: Trace = [systemMsg("s"), userMsg("u1"), assistantMsg("a1")]
    const chain: RunChainEntry[] = [
      { run_id: "run-1", turn_index: 1, trace_start_index: 0 },
    ]
    const result = compute_forkable_run_ids(trace, chain)
    expect(result).toEqual([null, null, null])
  })
})

describe("fork_target_from_assistant_block", () => {
  const chain: RunChainEntry[] = [
    { run_id: "run-1", turn_index: 1, trace_start_index: 0 },
    { run_id: "run-2", turn_index: 2, trace_start_index: 3 },
    { run_id: "run-3", turn_index: 3, trace_start_index: 5 },
  ]

  it("returns the parent run id and an empty prefill for an interior assistant click", () => {
    // Forking on turn 1's assistant (index 2) creates a new turn 2.
    const target = fork_target_from_assistant_block("run-2", 2, chain)
    expect(target).not.toBeNull()
    expect(target?.turn_index).toBe(2)
    expect(target?.parent_run_id).toBe("run-1")
    // Truncates just after the clicked assistant message.
    expect(target?.trace_index).toBe(3)
    expect(target?.prefill).toBe("")
  })

  it("returns null when the mapped turn is turn 1 (no parent exists)", () => {
    const target = fork_target_from_assistant_block("run-1", 0, chain)
    expect(target).toBeNull()
  })

  it("returns the leaf's parent when the assistant before the leaf turn is clicked", () => {
    // Forking on turn 2's assistant (index 4) creates a new turn 3.
    const target = fork_target_from_assistant_block("run-3", 4, chain)
    expect(target?.parent_run_id).toBe("run-2")
    expect(target?.turn_index).toBe(3)
    expect(target?.trace_index).toBe(5)
    expect(target?.prefill).toBe("")
  })

  it("returns null when the run id is not found in chain", () => {
    const target = fork_target_from_assistant_block("unknown", 2, chain)
    expect(target).toBeNull()
  })
})
