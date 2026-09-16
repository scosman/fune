import { describe, it, expect } from "vitest"
import { sse_data_payloads } from "./sse"

function reader_of(chunks: string[]): ReadableStreamDefaultReader<Uint8Array> {
  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
  return stream.getReader()
}

async function collect(chunks: string[]): Promise<string[]> {
  const out: string[] = []
  for await (const payload of sse_data_payloads(reader_of(chunks))) {
    out.push(payload)
  }
  return out
}

describe("sse_data_payloads", () => {
  it("yields the payload of each data line", async () => {
    expect(
      await collect(['data: {"a":1}\n\ndata: {"b":2}\n\ndata: complete\n\n']),
    ).toEqual(['{"a":1}', '{"b":2}', "complete"])
  })

  it("handles CRLF line endings", async () => {
    expect(
      await collect(['data: {"a":1}\r\n\r\ndata: complete\r\n\r\n']),
    ).toEqual(['{"a":1}', "complete"])
  })

  it("buffers payloads split across chunks", async () => {
    expect(
      await collect(['data: {"long', '":"payload"}\n\ndata: next\n\n']),
    ).toEqual(['{"long":"payload"}', "next"])
  })

  it("skips non-data lines and blank payloads", async () => {
    expect(
      await collect([": comment\nevent: something\ndata: \ndata: real\n\n"]),
    ).toEqual(["real"])
  })

  it("drops an unterminated trailing line", async () => {
    // No trailing newline — the last line never completed, so it is not a
    // full SSE frame and must not be emitted as one.
    expect(await collect(["data: done\n\ndata: partial"])).toEqual(["done"])
  })
})
