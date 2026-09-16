// Compose a user-driven abort signal with a hard deadline, keeping the two
// causes distinguishable after the fact. A fired deadline means the call is
// over-budget and the caller surfaces a retryable error, while a user abort
// must still cancel the whole flow — so the catch block needs to know which
// one rejected the request.
export type DeadlineSignal = {
  signal: AbortSignal
  // True iff the deadline fired and the user signal did not. Checking the
  // user signal guarantees an explicit cancel is never reported as a mere
  // timeout. (The reverse race — a cancel in the instant AFTER the deadline
  // already rejected the call — reads as an ordinary failure, not a cancel;
  // vanishing window, benign for both callers.)
  timed_out: () => boolean
}

export function with_deadline(
  user_signal: AbortSignal,
  timeout_ms: number,
): DeadlineSignal {
  const deadline = AbortSignal.timeout(timeout_ms)
  return {
    signal: AbortSignal.any([user_signal, deadline]),
    timed_out: () => deadline.aborted && !user_signal.aborted,
  }
}
