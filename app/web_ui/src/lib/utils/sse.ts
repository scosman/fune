// Shared SSE reader for POST-initiated event streams (EventSource is
// GET-only, so streaming endpoints are consumed via fetch + ReadableStream).
//
// One line-splitting implementation for every consumer: handles CRLF line
// endings (the SSE spec allows \r\n; a server or proxy that emits them must
// not break parsing), buffers partial lines across chunks, and yields the
// payload of each `data:` line as a string. Protocol concerns above the
// line level (JSON parsing, terminator sentinels like "complete" or
// "[DONE]") stay with the caller — they differ per endpoint.

/**
 * Yield the payload of each `data:` line from an SSE byte stream.
 *
 * Takes the reader (not the Response) so callers that swap streams
 * mid-conversation can start a fresh generator per reader.
 */
export async function* sse_data_payloads(
  reader: ReadableStreamDefaultReader<Uint8Array>,
): AsyncGenerator<string> {
  const decoder = new TextDecoder()
  let buffer = ""
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split("\n")
    buffer = lines.pop() ?? ""
    for (const raw_line of lines) {
      // CRLF: strip the trailing \r the \n-split leaves behind.
      const line = raw_line.endsWith("\r") ? raw_line.slice(0, -1) : raw_line
      if (!line.startsWith("data: ")) continue
      const payload = line.slice("data: ".length).trim()
      if (payload) yield payload
    }
  }
}
