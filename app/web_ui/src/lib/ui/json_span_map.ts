// Map a character span from one rendering of a JSON document onto another.
//
// A citation's offsets index the JSON exactly as the judge read it — the raw
// string carried in the trace. On screen that same JSON is re-printed:
// `JSON.stringify(JSON.parse(raw), null, 2)`, which adds whitespace AND
// normalises values (`135.0` becomes `135`). So the two strings differ in
// both layout and content, and a raw offset means nothing in the printed text.
//
// Both strings are the same document though, so their JSON tokens appear in
// the same order. Pairing tokens by position gives a translation that survives
// re-printing, without either side having to agree on formatting.

export type Span = { start: number; end: number }

type Token = { start: number; end: number }

// Scan JSON into token spans, skipping whitespace. Structural characters,
// strings, numbers and literals each count as one token. Returns null on input
// the scanner cannot account for, so a caller draws nothing rather than
// guessing at a mapping.
export function tokenize_json(text: string): Token[] | null {
  const tokens: Token[] = []
  let i = 0
  while (i < text.length) {
    const ch = text[i]
    if (ch === " " || ch === "\n" || ch === "\r" || ch === "\t") {
      i++
      continue
    }
    if (
      ch === "{" ||
      ch === "}" ||
      ch === "[" ||
      ch === "]" ||
      ch === "," ||
      ch === ":"
    ) {
      tokens.push({ start: i, end: i + 1 })
      i++
      continue
    }
    if (ch === '"') {
      const start = i
      i++
      let closed = false
      while (i < text.length) {
        if (text[i] === "\\") {
          i += 2
          continue
        }
        if (text[i] === '"') {
          i++
          closed = true
          break
        }
        i++
      }
      // An unterminated string means the text is not the document we think it
      // is; refuse rather than emit a token with a guessed end.
      if (!closed) return null
      tokens.push({ start, end: i })
      continue
    }
    // Numbers and the bare literals (true/false/null) both run until the next
    // structural character or whitespace, so one scan covers them.
    const start = i
    while (
      i < text.length &&
      !" \n\r\t{}[],:".includes(text[i]) &&
      text[i] !== '"'
    ) {
      i++
    }
    if (i === start) return null
    tokens.push({ start, end: i })
  }
  return tokens
}

// Translate [span.start, span.end) from `raw` into the equivalent span of
// `printed`. Returns null when the two texts do not tokenize alike, or when
// the span covers no token at all — both mean we cannot say where the quote
// landed, and no highlight beats a wrong one.
export function map_json_span(
  raw: string,
  printed: string,
  span: Span,
): Span | null {
  if (span.end < span.start) return null
  const raw_tokens = tokenize_json(raw)
  const printed_tokens = tokenize_json(printed)
  if (!raw_tokens || !printed_tokens) return null
  // Same document, so the token streams must line up. A mismatch means one of
  // the two is not what we assumed, and every mapping below would be fiction.
  if (raw_tokens.length !== printed_tokens.length) return null
  if (raw_tokens.length === 0) return null

  // The first and last tokens the span touches. A zero-width span still anchors
  // to the token containing it, so a caret-style citation has a home.
  let first = -1
  let last = -1
  for (let i = 0; i < raw_tokens.length; i++) {
    const t = raw_tokens[i]
    const overlaps =
      span.start < t.end && (span.end > t.start || span.end === span.start)
    if (overlaps && span.start < t.end && span.end > t.start) {
      if (first < 0) first = i
      last = i
    }
  }
  if (first < 0) return null

  // Clip inside a token only when its text survived re-printing unchanged —
  // true of strings and structure, false of numbers like `135.0` -> `135`.
  // Where it changed, the whole token is the most precise honest answer.
  const raw_first = raw.slice(raw_tokens[first].start, raw_tokens[first].end)
  const printed_first = printed.slice(
    printed_tokens[first].start,
    printed_tokens[first].end,
  )
  const start =
    raw_first === printed_first && span.start > raw_tokens[first].start
      ? printed_tokens[first].start + (span.start - raw_tokens[first].start)
      : printed_tokens[first].start

  const raw_last = raw.slice(raw_tokens[last].start, raw_tokens[last].end)
  const printed_last = printed.slice(
    printed_tokens[last].start,
    printed_tokens[last].end,
  )
  const end =
    raw_last === printed_last && span.end < raw_tokens[last].end
      ? printed_tokens[last].start + (span.end - raw_tokens[last].start)
      : printed_tokens[last].end

  if (end < start) return null
  return { start, end }
}

// Wrap [start, end) of the RENDERED TEXT of an HTML fragment in a <mark>,
// leaving the markup around it intact. Used on highlight.js output, where the
// JSON is already split into <span> elements for colouring.
//
// Two details the offsets depend on: an entity (`&quot;`) is one character of
// text, not six; and a mark never straddles a tag, because that would nest
// elements improperly. Crossing a tag closes the mark and reopens it after, so
// one citation can render as several adjacent <mark> elements.
export function mark_html_range(
  html: string,
  start: number,
  end: number,
  class_name: string,
): string {
  if (end <= start) return html
  const open_tag = `<mark data-highlight-target class="${class_name}">`
  const close_tag = "</mark>"
  let out = ""
  let text_index = 0
  let open = false
  let i = 0
  while (i < html.length) {
    if (html[i] === "<") {
      const gt = html.indexOf(">", i)
      if (gt < 0) {
        out += html.slice(i)
        break
      }
      if (open) {
        out += close_tag
        open = false
      }
      out += html.slice(i, gt + 1)
      i = gt + 1
      continue
    }
    let piece = html[i]
    if (piece === "&") {
      const semi = html.indexOf(";", i)
      // Entities are short; a distant `;` is ordinary text, not an entity.
      if (semi > i && semi - i <= 10) piece = html.slice(i, semi + 1)
    }
    const inside = text_index >= start && text_index < end
    if (inside && !open) {
      out += open_tag
      open = true
    } else if (!inside && open) {
      out += close_tag
      open = false
    }
    out += piece
    text_index++
    i += piece.length
  }
  if (open) out += close_tag
  return out
}
