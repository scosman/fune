import { describe, it, expect } from "vitest"
import { with_deadline } from "./deadline_signal"

// Real timers on purpose: fake timers cannot drive AbortSignal.timeout —
// its clock lives in the platform, not in setTimeout.
const tick = (ms: number) => new Promise((r) => setTimeout(r, ms))

describe("with_deadline", () => {
  it("fires the composed signal when the deadline elapses, and reports timed_out", async () => {
    const user = new AbortController()
    const { signal, timed_out } = with_deadline(user.signal, 10)

    expect(signal.aborted).toBe(false)
    expect(timed_out()).toBe(false)

    await tick(30)

    expect(signal.aborted).toBe(true)
    expect(timed_out()).toBe(true)
  })

  it("fires on user abort without reporting timed_out", async () => {
    const user = new AbortController()
    const { signal, timed_out } = with_deadline(user.signal, 10_000)

    user.abort()

    expect(signal.aborted).toBe(true)
    expect(timed_out()).toBe(false)
  })

  it("treats a user abort after the deadline as a user abort — an explicit cancel is never masked as a timeout", async () => {
    const user = new AbortController()
    const { signal, timed_out } = with_deadline(user.signal, 10)

    await tick(30)
    expect(timed_out()).toBe(true)

    user.abort()

    expect(signal.aborted).toBe(true)
    expect(timed_out()).toBe(false)
  })

  it("leaves the deadline signal independent per call", async () => {
    const user = new AbortController()
    const first = with_deadline(user.signal, 10)
    const second = with_deadline(user.signal, 10_000)

    await tick(30)

    expect(first.timed_out()).toBe(true)
    expect(second.timed_out()).toBe(false)
    expect(second.signal.aborted).toBe(false)
  })
})
