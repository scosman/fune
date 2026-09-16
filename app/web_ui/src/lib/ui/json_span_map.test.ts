import { describe, it, expect } from "vitest"
import { tokenize_json, map_json_span, mark_html_range } from "./json_span_map"

// The printed form is exactly what Output renders: parse, then re-stringify
// with two-space indent. Deriving it here rather than hand-writing it keeps
// the tests honest about what is actually on screen.
function printed(raw: string): string {
  return JSON.stringify(JSON.parse(raw), null, 2)
}

// The span of `quote` inside `raw` — how a citation reaches this code: the
// model quotes text, and the client resolves it to offsets.
function quote_span(raw: string, quote: string) {
  const start = raw.indexOf(quote)
  if (start < 0) throw new Error(`quote not present in raw: ${quote}`)
  return { start, end: start + quote.length }
}

// What the user would see marked on screen.
function marked(raw: string, quote: string): string | null {
  const out = printed(raw)
  const span = map_json_span(raw, out, quote_span(raw, quote))
  return span ? out.slice(span.start, span.end) : null
}

describe("tokenize_json", () => {
  it("gives one token per structural character, string, number and literal", () => {
    const tokens = tokenize_json('{"a": 1, "b": [true, null]}')
    // { "a" : 1 , "b" : [ true , null ] }
    expect(tokens?.length).toBe(13)
  })

  it("keeps an escaped quote inside its string rather than ending it", () => {
    const text = '{"a": "say \\"hi\\""}'
    const tokens = tokenize_json(text)
    expect(tokens?.length).toBe(5)
    const value = tokens![4 - 1]
    expect(text.slice(value.start, value.end)).toBe('"say \\"hi\\""')
  })

  it("refuses an unterminated string instead of guessing where it ends", () => {
    expect(tokenize_json('{"a": "unclosed')).toBeNull()
  })
})

describe("map_json_span", () => {
  const RAW = '{"a": 135.0, "b": 0.15}'

  it("marks a string value through re-printing", () => {
    const raw = '{"category": "refund_status", "priority": "normal"}'
    expect(marked(raw, '"refund_status"')).toBe('"refund_status"')
  })

  it("marks a key through re-printing", () => {
    expect(marked(RAW, '"b"')).toBe('"b"')
  })

  it("marks the whole document when the citation quotes all of it", () => {
    // Layout differs completely between the two forms, so this is the case
    // that would break any offset-carrying approach.
    expect(marked(RAW, RAW)).toBe(printed(RAW))
  })

  it("marks a normalised number as the whole token, since its text changed", () => {
    // The judge read `135.0`; the screen shows `135`. Clipping inside would
    // land on the wrong characters, so the whole value is marked.
    expect(marked(RAW, "135.0")).toBe("135")
  })

  it("marks the whole number when the citation starts partway into it", () => {
    // `35.0` starts one character into `135.0`, and that token was reprinted
    // as `135`. Carrying the offset in would mark from the wrong character, so
    // the whole token is taken.
    expect(marked(RAW, "35.0")).toBe("135")
  })

  it("marks part of a string value when its text survived unchanged", () => {
    const raw = '{"summary": "Customer asking for a refund"}'
    expect(marked(raw, "Customer asking")).toBe("Customer asking")
  })

  it("spans several tokens when the citation covers a key and its value", () => {
    expect(marked(RAW, '"b": 0.15')).toBe('"b": 0.15')
  })

  it("draws nothing when the two texts are different documents", () => {
    const span = { start: 1, end: 4 }
    expect(map_json_span('{"a": 1}', '{"a": 1, "b": 2}', span)).toBeNull()
  })

  it("draws nothing for a span that lands in whitespace only", () => {
    const raw = '{"a": 1,   "b": 2}'
    // The run of spaces between the two pairs is not part of any token.
    const span = { start: raw.indexOf("   "), end: raw.indexOf("   ") + 3 }
    expect(map_json_span(raw, printed(raw), span)).toBeNull()
  })

  it("draws nothing when a text cannot be tokenized", () => {
    // Output only pretty-prints what parses as JSON, so unparseable text never
    // reaches here in the product — but refusing beats guessing if it does.
    expect(
      map_json_span('{"a": "unclosed', "{}", { start: 0, end: 3 }),
    ).toBeNull()
  })

  it("draws nothing for a span past the end of the document", () => {
    const raw = '{"a": 1}'
    expect(map_json_span(raw, printed(raw), { start: 99, end: 120 })).toBeNull()
  })

  it("draws nothing for an inverted span", () => {
    const raw = '{"a": 1}'
    expect(map_json_span(raw, printed(raw), { start: 6, end: 2 })).toBeNull()
  })

  it("survives nesting, where indentation shifts every later offset", () => {
    const raw = '{"outer": {"inner": ["x", "y"]}, "tail": 1}'
    expect(marked(raw, '"y"')).toBe('"y"')
    expect(marked(raw, '"tail"')).toBe('"tail"')
  })
})

describe("mark_html_range", () => {
  // The rendered text of an HTML fragment, with tags and entities resolved the
  // way a browser would show them.
  function rendered(html: string): string {
    return html
      .replace(/<[^>]*>/g, "")
      .replace(/&quot;/g, '"')
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
  }

  function marked_text(html: string, start: number, end: number): string {
    const out = mark_html_range(html, start, end, "hl")
    // Everything inside any <mark>, concatenated — several are expected when
    // the range crosses a tag.
    const inner = [...out.matchAll(/<mark[^>]*>([\s\S]*?)<\/mark>/g)]
      .map((m) => m[1])
      .join("")
    return rendered(inner)
  }

  it("marks a range inside one text run", () => {
    expect(marked_text("<span>hello world</span>", 6, 11)).toBe("world")
  })

  it("marks across a tag boundary without nesting elements improperly", () => {
    const html = '<span class="a">ab</span><span class="b">cd</span>'
    const out = mark_html_range(html, 1, 3, "hl")
    expect(marked_text(html, 1, 3)).toBe("bc")
    // Two marks, each wholly inside its own span — never one spanning both.
    expect(out.match(/<mark/g)?.length).toBe(2)
    expect(out).not.toMatch(/<mark[^>]*>[^<]*<span/)
  })

  it("counts an entity as one character, not as its escape", () => {
    // hljs escapes quotes, so a JSON string is full of &quot;.
    const html = "<span>&quot;ab&quot;</span>"
    expect(rendered(html)).toBe('"ab"')
    // Characters 1..3 of the rendered text are `ab`.
    expect(marked_text(html, 1, 3)).toBe("ab")
  })

  it("leaves the fragment untouched for an empty range", () => {
    const html = "<span>abc</span>"
    expect(mark_html_range(html, 2, 2, "hl")).toBe(html)
  })

  it("preserves the full rendered text", () => {
    const html = '<span class="a">{&quot;k&quot;: 1}</span>'
    expect(rendered(mark_html_range(html, 1, 4, "hl"))).toBe(rendered(html))
  })
})
