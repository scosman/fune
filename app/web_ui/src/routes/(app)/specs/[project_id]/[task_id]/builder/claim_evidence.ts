// Claim review — client mirror of the kiln_server buildClaimEvidence task
// contract. Hand-mirrored because the task lives outside this repo's
// generated API schema.
//
// The server task is PER-TRACE: one call distills one trace (raw_input +
// raw_output) plus the judge's decision into an Overview the reviewer reads
// first and a short list of claims, each one decision the judge made, written
// so the reviewer can agree or disagree with it from the card alone. The UI
// holds N of these (one per generated trace) and manages trace identity
// itself, since the server output carries no id.

import type { components } from "$lib/api_schema"
import type { TraceMessage } from "$lib/types"
import { TASK_RESPONSE_TOOL_NAME } from "$lib/utils/task_response_tool"

export type CitationSource = "input" | "output"

// The judge's binary verdict, and the reviewer's overall call in the same
// vocabulary.
export type JudgeScore = "pass" | "fail"

// A start+end anchor into raw_input/raw_output. The parser highlights the span
// from the first occurrence of `from` through the end of `to`. Snippets (not
// long quotes) keep the model fast — it doesn't recite verbatim text.
export type Citation = {
  marker: number // the [n] referenced inline in the text
  source: CitationSource
  from: string
  to: string
}

// One decision the judge made, written so the reviewer can vote on it from
// the card alone. `text` carries the claim, its evidence and its inline [n]
// markers in one string; every marker resolves via `citations`. Grades have
// one direction on every claim: agree means the judge got this decision
// right, disagree means it got it wrong.
//
// `is_verdict` marks the claim that states the overall pass/fail. The builder
// may omit it; when present it is the LAST claim. The studio sets the flag
// from the builder's own opener convention, so nothing here parses prose to
// find the verdict.
export type Claim = {
  text: string
  citations: Citation[]
  is_verdict: boolean
}

// The neutral summary the reviewer reads before the claims. Same prose shape
// as a claim: [n] markers restart at [1] here and in each claim.
export type Overview = {
  text: string
  citations: Citation[]
}

// ── Client-side per-trace bundle ─────────────────────────────────────────

// Claims build lazily on both arms (the pipeline streams stop at the
// judge): "unbuilt" until the trace is selected/opened, then a
// build_claims round trip moves it through "building" to "built" or "error".
export type ClaimsBuildState = "unbuilt" | "building" | "built" | "error"

// One generated trace + the claims built for it. raw_input/raw_output are kept
// client-side so the trace modal can render them and resolve citation spans.
// leaf_run_id is the durable TaskRun identity on both arms — the chain leaf
// from run_cases_batch, or the single-turn pipeline's persisted run — that
// the save path writes the golden rating onto and calibration rounds
// re-judge through; null/"" when the runner emitted no id for a case.
// overview/claims are null until claims_state is "built".
export type TraceClaims = {
  trace_id: string
  leaf_run_id: string | null
  raw_input: string
  raw_output: string
  judge_score: JudgeScore
  judge_reasoning: string
  overview: Overview | null
  claims: Claim[] | null
  claims_state: ClaimsBuildState
  claims_error: string | null
  // The run's structured trace. On multi-turn it is what the judge saw and
  // raw_output is its lossy flattening; the trace modal renders THIS in the
  // house chat UI and remaps output-source citation spans back onto it. On
  // single-turn it is a UI-only echo (the judge scores the I/O pair).
  // Absent/null when the run recorded none — the modal keeps the raw view.
  trace?: TraceMessage[] | null
}

// ── Human review (UI output) ─────────────────────────────────────────────

// Server claims carry no id, so verdicts are positional (index into
// TraceClaims.claims).
export type ClaimVerdict = {
  agrees: boolean | null // null = not yet reviewed
  why: string // required when the human disagrees — feeds the refine loop
}

export type TraceReview = {
  trace_id: string
  claim_verdicts: ClaimVerdict[]
}

// ── Claim text ───────────────────────────────────────────────────────────
//
// Claim and overview prose carries inline [n] markers. The card turns each
// marker that resolves to a citation into a chip that opens the trace at the
// cited span. A marker with NO citation stays as plain text: the model quotes
// bracketed digits out of traces (a numbered list, a log line), and a chip
// that opens nothing reads as evidence that is not there.

export type ClaimTextToken =
  | { kind: "text"; value: string }
  | { kind: "cite"; n: number; citation: Citation }

export function tokenize_claim_text(
  text: string,
  citations: Citation[],
): ClaimTextToken[] {
  const out: ClaimTextToken[] = []
  let pending = ""
  const flush = () => {
    if (pending) out.push({ kind: "text", value: pending })
    pending = ""
  }
  const re = /\[(\d+)\]/g
  let last = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    pending += text.slice(last, m.index)
    const n = Number(m[1])
    const citation = citations.find((c) => c.marker === n)
    if (citation) {
      flush()
      out.push({ kind: "cite", n, citation })
    } else {
      pending += m[0]
    }
    last = m.index + m[0].length
  }
  pending += text.slice(last)
  flush()
  return out
}

// The builder's "Note:" convention: a blank line, then a paragraph opening
// "Note:", trailing the claim. It renders apart from the claim and muted. The
// body is everything else — the "We suggest …" sentence included, since that
// is part of the ask, not an aside.
export function split_claim_note(text: string): {
  body: string
  note: string | null
} {
  const at = text.search(/\n[ \t]*\n[ \t]*(?=Note:)/)
  if (at < 0) return { body: text, note: null }
  return {
    body: text.slice(0, at).trimEnd(),
    note: text.slice(at).trimStart(),
  }
}

// ── Citation resolution ──────────────────────────────────────────────────

// A resolved citation: the whole [start, end) span, plus where the `from`
// anchor alone ends. from_end lets a caller that cannot place the whole span
// fall back to the anchor the model actually wrote, instead of nothing.
export type CitationSpan = {
  start: number
  end: number
  from_end: number
}

// Resolve a citation to a [start, end) span in its source text. Finds the
// first occurrence of `from`, then the first `to` at/after it (so `to` can sit
// later in the text; equal to `from` for a short span). Returns null if either
// anchor is absent — the UI then shows the citation without a highlight rather
// than highlighting the wrong place.
//
// Grep-safety note (open question): for repeated identical `from`
// snippets this anchors to the FIRST match, which can mis-locate. Mitigation is
// on the model side (pick a `from` long enough to be locally unique); a future
// optional occurrence index could disambiguate if it proves necessary.
export function resolve_citation_span(
  text: string,
  citation: Pick<Citation, "from" | "to">,
): CitationSpan | null {
  const haystack = fold_typography(text)
  const from = fold_typography(citation.from)
  const to = fold_typography(citation.to)
  const start = haystack.indexOf(from)
  if (start < 0) return null
  const to_at = haystack.indexOf(to, start)
  if (to_at < 0) return null
  // The fold is 1:1 here, so the anchor's folded length is its raw length.
  return { start, from_end: start + from.length, end: to_at + to.length }
}

// Curly punctuation folded to its straight form. Models retype anchors from
// text they read, so a citation often carries ' where the output has ’ and the
// anchor then misses. ONLY 1-code-unit-to-1-code-unit pairs belong here: the
// folded string must stay the same length as the original, since the spans
// resolved against it index the original text (they drive highlight slicing).
//
// Folding widens the match set rather than only rescuing misses: where the same
// snippet appears in the text in both quote styles, the first match can land on
// a different occurrence than before. Accepted — the variants are semantically
// the same sentence, so either occurrence is an honest highlight.
const TYPOGRAPHIC_FOLD: Record<string, string> = {
  "‘": "'",
  "’": "'",
  "“": '"',
  "”": '"',
}

// Derived from the map, never hand-written alongside it: a class listing a
// character the map lacks would substitute the string "undefined" and shift
// every offset after it, so the two cannot be allowed to drift.
const TYPOGRAPHIC_FOLD_PATTERN = new RegExp(
  `[${Object.keys(TYPOGRAPHIC_FOLD)
    .map((c) => c.replace(/[.*+?^${}()|[\]\\-]/g, "\\$&"))
    .join("")}]`,
  "g",
)

function fold_typography(text: string): string {
  // `?? c` keeps an unmapped match length-preserving even if the two ever part.
  return text.replace(TYPOGRAPHIC_FOLD_PATTERN, (c) => TYPOGRAPHIC_FOLD[c] ?? c)
}

// ── Fold with an offset map (anchoring a citation in a RE-RENDERED string) ──
//
// The fold above is strictly 1:1, which is what lets a span resolved against it
// index the raw text. A field that PRETTY-PRINTS its content has no such
// luxury: `{"a":1}` reaches the reviewer as `{\n  "a": 1\n}`, so anchors the
// model retyped from the raw string miss the rendered one on whitespace alone,
// and the field is stuck showing the raw blob to keep its mark.
//
// This generalizes the 1:1 fold: it still normalizes for matching, but it also
// records where each folded character came from, so a span found in folded
// space translates back into offsets that slice the ORIGINAL — here the
// pretty-printed string the reviewer is actually looking at.
//
// The rules are a transform table over RUNS of the original: each run is
// copied, swapped for a replacement, or dropped. Whitespace and typography are
// the two we need today; a markdown-strip fold (drop the `**`, keep the word)
// is the same shape and would be another rule rather than a second mechanism.

export type FoldRule = {
  // Matches ONE run to transform. Two things are required of it. It must not be
  // able to match the EMPTY string: a zero-width match consumes nothing, so its
  // replacement would be spurious characters in the folded string with no run
  // behind them. And it must match its own matches ANCHORED — `^(?:pattern)$`
  // against a run the pattern produced — because that is how the folder tells
  // which rule fired. A pattern anchored internally (`^…` or `…$` of its own,
  // or a lookaround reaching outside the run) breaks that and its runs are
  // copied through untouched instead.
  pattern: string
  // What the run becomes in the folded string. "" drops it entirely.
  replace: (run: string) => string
}

// A folded string plus where every character of it came from. map and map_end
// are separate because a folded character can stand for a WIDER run than
// itself: a span ending on a collapsed space has to stop past the whole run
// that space swallowed, and one starting on it has to start at that run's first
// character. Both arrays are folded.length long.
export type FoldedText = {
  folded: string
  // map[i] = index in the ORIGINAL string where folded[i]'s run begins.
  map: number[]
  // map_end[i] = index in the ORIGINAL just past that run. map[i] + 1 for a
  // character copied or swapped 1:1.
  map_end: number[]
}

export function fold_with_offsets(text: string, rules: FoldRule[]): FoldedText {
  const pattern = new RegExp(
    rules.map((r) => `(?:${r.pattern})`).join("|"),
    "g",
  )
  // Which rule produced a run is decided by re-testing the run against each
  // rule, not by counting capture groups in the alternation: a rule that
  // brought its own groups would shift that count and silently dispatch the
  // wrong rule. Same left-to-right precedence the alternation itself uses.
  const whole_run = rules.map((r) => new RegExp(`^(?:${r.pattern})$`))
  let folded = ""
  const map: number[] = []
  const map_end: number[] = []
  let cursor = 0
  // Everything between two matched runs is carried over one character per
  // character, so its offsets are its own.
  const copy_verbatim = (until: number) => {
    for (let i = cursor; i < until; i++) {
      folded += text[i]
      map.push(i)
      map_end.push(i + 1)
    }
    cursor = until
  }
  for (const match of text.matchAll(pattern)) {
    const at = match.index ?? 0
    copy_verbatim(at)
    const run = match[0]
    // A zero-width match transformed nothing, so it produces nothing (see the
    // FoldRule contract). Emitting a replacement here would inject characters
    // the original never had and desync every offset after them.
    if (!run) continue
    const rule_index = whole_run.findIndex((re) => re.test(run))
    // No rule owns up to its own run: the pattern breaks the anchored-match
    // half of the contract. Copy the run through as ordinary text rather than
    // indexing past the end of the table — a fold that skips one run still
    // yields exact offsets, where a crash yields no citation at all.
    if (rule_index < 0) {
      copy_verbatim(at + run.length)
      continue
    }
    const rule = rules[rule_index]
    // By code UNIT, not code point: the maps are indexed the way the folded
    // string is, and a replacement carrying a surrogate pair would otherwise
    // add two characters against one entry and shift everything after it.
    const into = rule.replace(run)
    for (let k = 0; k < into.length; k++) {
      folded += into[k]
      map.push(at)
      map_end.push(at + run.length)
    }
    cursor = at + run.length
  }
  copy_verbatim(text.length)
  return { folded, map, map_end }
}

// Whitespace RUNS to a single space: `"a": 1,\n  "b"` and `"a": 1, "b"` fold
// the same, which covers the difference pretty-printing makes between two
// tokens that were already separated.
export const WHITESPACE_TOLERANT_FOLD: FoldRule[] = [
  { pattern: TYPOGRAPHIC_FOLD_PATTERN.source, replace: fold_typography },
  { pattern: "\\s+", replace: () => " " },
]

// Whitespace runs dropped outright. Pretty-printing also SEPARATES tokens that
// were flush — `{"a":1` becomes `{\n  "a": 1` — and no collapse can match a
// space against nothing, so this is the wider second pass. It is not the first
// because dropping whitespace lets a needle match across a word boundary; the
// collapsing fold gets to answer while it can.
export const WHITESPACE_FREE_FOLD: FoldRule[] = [
  { pattern: TYPOGRAPHIC_FOLD_PATTERN.source, replace: fold_typography },
  { pattern: "\\s+", replace: () => "" },
]

// resolve_citation_span's whitespace-tolerant twin, for anchoring a citation in
// a re-rendered copy of the text it was written against. Same anchor semantics
// (first `from`, then the first `to` at/after it), but the returned span slices
// the string PASSED IN rather than the raw one. Null when either anchor is
// absent, and when either carries no non-whitespace text to locate. Callers use
// it to place the mark in a pretty-printed body; a miss is a real miss and they
// fall back to the raw text.
export function resolve_citation_span_whitespace_tolerant(
  text: string,
  citation: Pick<Citation, "from" | "to">,
): CitationSpan | null {
  return (
    resolve_span_in_fold(text, citation, WHITESPACE_TOLERANT_FOLD) ??
    resolve_span_in_fold(text, citation, WHITESPACE_FREE_FOLD)
  )
}

function resolve_span_in_fold(
  text: string,
  citation: Pick<Citation, "from" | "to">,
  rules: FoldRule[],
): CitationSpan | null {
  const haystack = fold_with_offsets(text, rules)
  // The needle folds through the SAME rules, or the two would be normalized
  // into different alphabets and never meet.
  const from = fold_with_offsets(citation.from, rules).folded
  const to = fold_with_offsets(citation.to, rules).folded
  // An anchor with nothing but whitespace left in it — empty to begin with, or
  // emptied by the fold that drops whitespace — matches at every position and
  // covers none of them. That is not a resolution: reporting one hands the
  // caller a span it reads as success, and it loses the raw mark it would
  // otherwise have fallen back to.
  if (!/\S/.test(from) || !/\S/.test(to)) return null
  const start = haystack.folded.indexOf(from)
  if (start < 0) return null
  const to_at = haystack.folded.indexOf(to, start)
  if (to_at < 0) return null
  const end = to_at + to.length
  return {
    start: span_start(haystack, start),
    // Mapped the same way the whole span's end is: a folded character can
    // stand for a wider run, so the anchor's end is not start + from.length.
    from_end: span_end(haystack, start + from.length - 1),
    end: span_end(haystack, end - 1),
  }
}

// Where a resolved span starts and ends in the original. A folded whitespace
// character stands for a whole collapsed run, so a span that begins or ends on
// one steps over it: carrying it along would put the line break and indent
// pretty-printing inserted inside the <mark>, which on screen reads as a
// rendering bug rather than a citation. The two are symmetric — an anchor with
// a leading space is as ordinary as one with a trailing space.
function span_start(haystack: FoldedText, first: number): number {
  return /\s/.test(haystack.folded[first])
    ? haystack.map_end[first]
    : haystack.map[first]
}

function span_end(haystack: FoldedText, last: number): number {
  return /\s/.test(haystack.folded[last])
    ? haystack.map[last]
    : haystack.map_end[last]
}

// ── Structured-trace span mapping (MULTI-TURN ONLY) ──────────────────────
//
// The trace modal renders a MULTI-TURN conversation (TraceMessage[]) in the
// house chat UI, but citations resolve against raw_output — the server's
// FLATTENED rendering of that trace. This maps a resolved [start, end) span in
// raw_output back onto the trace: which message, which block kind, and the
// offset within that block's text, so the chat UI can highlight the exact node.
//
// BOTH arms reach this now. A single-turn run is a conversation of one turn and
// renders in the same chat surface, so its citations map through here too —
// there is no second, arm-specific path any more.
//
// It works by re-computing the server flattener's block layout in TS. The port
// mirrors libs/core .../eval_trace_formatter.py EXACTLY — every block a message
// carries (reasoning, then content, then tool calls), the same role labels
// including the tool name on a result, the same tool-call formatting, the same
// tool-result .output unwrapping, and the same "\n\n" join over emitted blocks.
// Any drift would shift offsets, so the mapper verifies the recomputed block
// against raw_output before trusting it (see below).

type TraceHighlightKind = "content" | "reasoning" | "tool_calls" | "tool_result"

export type TraceHighlight = {
  trace_index: number
  kind: TraceHighlightKind
  start: number
  end: number
  // True when only the citation's `from` anchor could be marked because its
  // `to` anchor landed in a later turn. Callers surface it as a model-side
  // citation defect; absent means the whole span is highlighted.
  from_anchor_only?: boolean
}

// One emitted flattener block with its absolute span in the recomputed string.
type FlattenedBlock = {
  trace_index: number
  kind: TraceHighlightKind
  content_start: number
  content_end: number
  content: string
}

// Empty strings are falsy in Python's `if content:`, so a message whose block
// text is empty emits nothing — mirror that with JS truthiness throughout.
function trace_role(message: TraceMessage): string {
  return "role" in message && typeof message.role === "string"
    ? message.role
    : ""
}

// Mirror EvalTraceFormatter.content_from_message: string content only, and for
// tool messages unwrap the Kiln task-tool JSON to just its `output` field.
function flattener_content(message: TraceMessage): string | null {
  if (
    !("content" in message) ||
    typeof message.content !== "string" ||
    message.content.length === 0
  ) {
    return null
  }
  if (trace_role(message) === "tool") {
    try {
      const parsed = JSON.parse(message.content)
      if (parsed && typeof parsed === "object" && "output" in parsed) {
        // The server returns parsed["output"] verbatim; only a string output
        // reproduces its bytes here. A non-string output would need Python's
        // repr, which we can't match — leave it and let the byte-guard in the
        // mapper reject the block rather than highlight the wrong span.
        return typeof parsed.output === "string" ? parsed.output : null
      }
    } catch {
      // Not JSON — the formatter returns the content as-is.
    }
  }
  return message.content
}

function flattener_reasoning(message: TraceMessage): string | null {
  if (
    "reasoning_content" in message &&
    typeof message.reasoning_content === "string" &&
    message.reasoning_content.length > 0
  ) {
    return message.reasoning_content
  }
  return null
}

// Mirror EvalTraceFormatter.structured_output_from_message: the arguments of
// the last task_response call, which are the model's answer.
function flattener_structured_output(message: TraceMessage): string | null {
  if (!("tool_calls" in message) || !message.tool_calls) return null
  const calls = message.tool_calls
  if (!Array.isArray(calls)) return null
  let args: string | null = null
  for (const call of calls) {
    const fn = "function" in call ? call.function : undefined
    if (fn && fn.name === TASK_RESPONSE_TOOL_NAME) {
      args = typeof fn.arguments === "string" ? fn.arguments : null
    }
  }
  return args
}

// Mirror EvalTraceFormatter.formatted_tool_calls_from_message: one
// "- Tool Name: …\n- Arguments: …" per real call, joined by a blank line.
// The task_response wrapper is excluded — it is reported as the answer.
function flattener_tool_calls(message: TraceMessage): string | null {
  if (!("tool_calls" in message) || !message.tool_calls) return null
  const calls = message.tool_calls
  if (!Array.isArray(calls) || calls.length === 0) return null
  const parts = calls
    .filter((call) => {
      const fn = "function" in call ? call.function : undefined
      return !fn || fn.name !== TASK_RESPONSE_TOOL_NAME
    })
    .map((call) => {
      const fn = "function" in call ? call.function : undefined
      const name = fn && typeof fn.name === "string" ? fn.name : ""
      const args = fn && typeof fn.arguments === "string" ? fn.arguments : ""
      return `- Tool Name: ${name}\n- Arguments: ${args}`
    })
  const out = parts.join("\n\n")
  return out.length > 0 ? out : null
}

// The name of the tool call a tool message answers (matched by tool_call_id),
// searched across the whole trace — mirrors origin_tool_call_name_from_message.
// It names the result's role label; a result whose call cannot be found is
// still emitted, unnamed, exactly as the formatter does.
function origin_tool_call_name(
  message: TraceMessage,
  trace: TraceMessage[],
): string | null {
  const id =
    "tool_call_id" in message && typeof message.tool_call_id === "string"
      ? message.tool_call_id
      : null
  if (!id) return null
  for (const m of trace) {
    if (!("tool_calls" in m) || !Array.isArray(m.tool_calls)) continue
    for (const call of m.tool_calls) {
      if ("id" in call && call.id === id) {
        const fn = "function" in call ? call.function : undefined
        return fn && typeof fn.name === "string" ? fn.name : null
      }
    }
  }
  return null
}

type EmittedBlock = {
  role_label: string
  tag: string
  content: string
  kind: TraceHighlightKind
}

// The blocks one message flattens to, in emit order. Mirrors the formatter: a
// tool message emits its result (named when its originating call is found, and
// still emitted when it is not); every other message emits each of reasoning,
// content and tool calls that it carries. A message commonly carries more than
// one — a model narrates while it calls a tool — which is why this returns a
// list rather than a single block.
function emitted_blocks(
  message: TraceMessage,
  trace: TraceMessage[],
): EmittedBlock[] {
  const role = trace_role(message)
  const content = flattener_content(message)

  if (role === "tool" && content) {
    const name = origin_tool_call_name(message, trace)
    return [
      {
        role_label: name ? `tool result from ${name}` : "tool result",
        tag: `${role}_tool_message`,
        content,
        kind: "tool_result",
      },
    ]
  }

  const blocks: EmittedBlock[] = []
  const reasoning = flattener_reasoning(message)
  if (reasoning) {
    blocks.push({
      role_label: `${role} reasoning`,
      tag: `${role}_reasoning_message`,
      content: reasoning,
      kind: "reasoning",
    })
  }
  // Content before tool calls, matching the formatter: the text is the
  // narration introducing the call.
  if (content) {
    blocks.push({
      role_label: role,
      tag: `${role}_message`,
      content,
      kind: "content",
    })
  }
  // The structured answer renders as the answer, under the same tag a plain
  // message uses — mirroring the formatter.
  const structured_output = flattener_structured_output(message)
  if (structured_output) {
    blocks.push({
      role_label: role,
      tag: `${role}_message`,
      content: structured_output,
      kind: "content",
    })
  }
  const tool_calls = flattener_tool_calls(message)
  if (tool_calls) {
    blocks.push({
      role_label: `${role} requested tool calls`,
      tag: `${role}_requested_tool_calls`,
      content: tool_calls,
      kind: "tool_calls",
    })
  }
  return blocks
}

// Recompute the flattener's output string and record each block's absolute
// span. The "\n\n" separator is keyed on EMIT ORDER (matching the formatter's
// join over emitted blocks), not on position in the trace.
function flatten_output_blocks(trace: TraceMessage[]): FlattenedBlock[] {
  const blocks: FlattenedBlock[] = []
  let length = 0
  trace.forEach((message, index) => {
    for (const block of emitted_blocks(message, trace)) {
      if (blocks.length > 0) length += 2 // "\n\n"
      const header = `${block.role_label}:\n<${block.tag}>\n`
      const content_start = length + header.length
      const content_end = content_start + block.content.length
      blocks.push({
        trace_index: index,
        kind: block.kind,
        content_start,
        content_end,
        content: block.content,
      })
      // header + content + `\n</${tag}>`
      length = content_end + `\n</${block.tag}>`.length
    }
  })
  return blocks
}

// Map a resolved [start, end) span in raw_output to the trace node it came
// from. Returns null when the span doesn't start inside a block's text (e.g.
// it starts in the tag chrome), or when the recomputed layout doesn't match
// raw_output byte-for-byte at that offset — a mismatch means our port drifted
// from the server, so we surface NO highlight rather than a wrong one.
//
// A span whose `to` anchor landed in a LATER turn falls back to marking its
// `from` anchor alone (from_anchor_only). A chat node can only carry a span
// that lives inside it, and the model's own anchor is an honest partial
// highlight where the whole span has no node to live in.
//
// Two raw_output shapes reach this: the flattened transcript above, and the
// unflattened case where raw_output is one message's content verbatim. The
// flattened walk runs first and the verbatim match is the fallback, so a
// trace that fits neither still yields no highlight.
export function map_output_span_to_trace(
  trace: TraceMessage[],
  raw_output: string,
  span: { start: number; end: number; from_end?: number },
): TraceHighlight | null {
  return (
    map_span_in_flattened_layout(trace, raw_output, span) ??
    map_span_in_whole_message(trace, raw_output, span)
  )
}

function map_span_in_flattened_layout(
  trace: TraceMessage[],
  raw_output: string,
  span: { start: number; end: number; from_end?: number },
): TraceHighlight | null {
  for (const block of flatten_output_blocks(trace)) {
    // The block holding the `from` anchor. Blocks are disjoint, so at most one
    // matches and no other block could hold the whole span either.
    if (span.start < block.content_start || span.start >= block.content_end) {
      continue
    }
    // `to` resolved past this block, so scoping the search to the block leaves
    // the anchor alone as the span worth marking.
    const from_anchor_only = span.end > block.content_end
    const end = from_anchor_only ? span.from_end : span.end
    if (end === undefined || end < span.start || end > block.content_end) {
      return null
    }
    if (
      raw_output.slice(block.content_start, block.content_end) !== block.content
    ) {
      return null
    }
    return {
      trace_index: block.trace_index,
      kind: block.kind,
      start: span.start - block.content_start,
      end: end - block.content_start,
      from_anchor_only,
    }
  }
  return null
}

// Map a resolved [start, end) span in raw_input onto the conversation's
// opening user message. On multi-turn the input IS that message — the server
// derives raw_input from the first user message with non-empty string
// content, verbatim — so an input citation has an exact home in the chat:
// same string, same offsets. The pick mirrors the server's rule and the
// byte-identity guard rejects a raw_input that did not come from this
// trace's opening message (no highlight beats a wrong one).
export function map_input_span_to_trace(
  trace: TraceMessage[],
  raw_input: string,
  span: { start: number; end: number },
): TraceHighlight | null {
  const index = trace.findIndex(
    (message) =>
      trace_role(message) === "user" &&
      "content" in message &&
      typeof message.content === "string" &&
      message.content.length > 0,
  )
  if (index < 0) return null
  const message = trace[index]
  const content =
    "content" in message && typeof message.content === "string"
      ? message.content
      : ""
  if (content !== raw_input) return null
  if (span.start < 0 || span.end > raw_input.length) return null
  return {
    trace_index: index,
    kind: "content",
    start: span.start,
    end: span.end,
  }
}

// raw_output IS one message's content, with none of the flattener's role
// headers or tags, so the span's offsets carry onto that message
// unchanged. Requires exactly one byte-identical message — with two the
// cited one is ambiguous, and we'd rather show no highlight than the wrong one.
function map_span_in_whole_message(
  trace: TraceMessage[],
  raw_output: string,
  span: { start: number; end: number },
): TraceHighlight | null {
  const matches = trace
    .map((message, index) => ({ message, index }))
    .filter(
      ({ message }) => "content" in message && message.content === raw_output,
    )
  if (matches.length !== 1) return null
  if (span.start < 0 || span.end > raw_output.length) return null
  return {
    trace_index: matches[0].index,
    kind: "content",
    start: span.start,
    end: span.end,
  }
}

// ── Review-state helpers ─────────────────────────────────────────────────

export function build_trace_reviews(traces: TraceClaims[]): TraceReview[] {
  // Lazily-built traces start with no claim slots; the verdicts are sized
  // when their claims arrive (empty_claim_verdicts).
  return traces.map((t) => ({
    trace_id: t.trace_id,
    claim_verdicts: empty_claim_verdicts(t.claims ?? []),
  }))
}

// Fresh positional verdict slots for a trace whose claims just arrived.
export function empty_claim_verdicts(claims: Claim[]): ClaimVerdict[] {
  return claims.map(() => ({ agrees: null, why: "" }))
}

// Index of the claim carrying the overall verdict, or -1 when the builder
// omitted it. At most one claim is flagged (the studio flags only the last).
function verdict_claim_index(trace: Pick<TraceClaims, "claims">): number {
  return (trace.claims ?? []).findIndex((c) => c.is_verdict)
}

export function has_verdict_claim(trace: Pick<TraceClaims, "claims">): boolean {
  return verdict_claim_index(trace) >= 0
}

// A trace is reviewed once every claim on screen has a grade, every
// disagreement carries a reason, and the verdict claim is among them: the
// pass/fail call is that claim's grade and nothing else records it. A trace
// without one never reaches the reviewer (reviewable_subset), so this is the
// gate, not a fallback. Unbuilt, in-flight and failed traces are never
// reviewed: nothing was presented.
export function is_trace_reviewed(
  trace: TraceClaims,
  review: TraceReview | undefined,
): boolean {
  if (!review) return false
  if (trace.claims_state !== "built") return false
  const claims = trace.claims ?? []
  // Slots are sized when the claims arrive; until then nothing is gradable.
  if (review.claim_verdicts.length !== claims.length) return false
  const graded = review.claim_verdicts.every((v) => v.agrees !== null)
  const reasoned = review.claim_verdicts.every(
    (v) => v.agrees !== false || v.why.trim().length > 0,
  )
  if (!graded || !reasoned) return false
  return has_verdict_claim(trace)
}

export function reviewed_trace_count(
  traces: TraceClaims[],
  reviews: TraceReview[],
): number {
  return traces.filter((t, i) => is_trace_reviewed(t, reviews[i])).length
}

// The reviewer's call on a trace: the verdict claim's grade IS the call —
// agree keeps the judge's verdict, disagree flips it. Null while unanswered,
// and null for a trace with no verdict claim, which the reviewer is never
// shown. Nothing else records the call.
export function human_verdict(
  trace: TraceClaims,
  review: TraceReview,
): JudgeScore | null {
  const at = verdict_claim_index(trace)
  if (at < 0) return null
  const agrees = review.claim_verdicts[at]?.agrees ?? null
  if (agrees === null) return null
  if (agrees) return trace.judge_score
  return trace.judge_score === "pass" ? "fail" : "pass"
}

// The reviewer's overall verdict as the golden rating's boolean. Callers gate
// on is_trace_reviewed; an unanswered call is refused rather than guessed,
// because a guessed rating poisons the answer key silently.
export function user_says_meets_spec(
  trace: TraceClaims,
  review: TraceReview,
): boolean {
  const verdict = human_verdict(trace, review)
  if (verdict === null) {
    throw new Error(
      "Cannot read the reviewer's verdict before the trace is graded.",
    )
  }
  return verdict === "pass"
}

// ── Subset review (both arms) ────────────────────────────────────────────

// How many traces the reviewer must rate. The human-rated golden answer key is
// capped at 25% of the batch runs server-side, so N//4 is what would fill it
// exactly — but rating is human work and does not get cheaper as the batch
// grows, so it stops at REVIEW_TARGET_MAX. Past that the answer key is
// deliberately smaller than the server would allow: the server never pads
// golden with unrated items, so a short rated set simply yields a shorter key.
// Floor of 1 — a batch with no rated trace has no answer key at all.
const REVIEW_TARGET_MAX = 10

export function review_target(total: number): number {
  if (total <= 0) return 0
  return Math.min(REVIEW_TARGET_MAX, Math.max(1, Math.floor(total / 4)))
}

// The reviews the save gate demands during a calibration round: the standard
// target, capped by how many traces the round could actually surface. A
// re-judge shortfall can shrink the round's subset below floor(N/4), and
// demanding the full target then would deadlock the gate on traces the
// reviewer was never shown. First-round subsets are sized to the target, so
// this only ever bites mid-loop.
// The subset the reviewer actually walks: the selected traces minus the ones
// there is no claim review to do on. Two cases drop out, for the same reason.
//
// A FAILED build carries no overview and no claims, so there is nothing to
// grade. A BUILT trace whose claims carry no verdict claim has claims to grade
// but nothing that records the pass/fail call, and asking the reviewer to
// supply one the builder should have written is a different task from the one
// this step is built around. The builder's contract is that the last claim is
// the verdict, and it almost always is; a trace that breaks it is a prompt
// problem, not a reviewing problem.
//
// Nothing is built to replace either: the claims gate has already finished
// paying for the round's builds.
//
// An excluded trace is an unselected one in every sense: never shown, so never
// graded, so absent from the answer key and left to the train split unrated.
// Pair it with calibration_gate_target so the save gate, the step header's
// "reviewing N of M" and the review's own counter all read one number.
export function reviewable_subset(
  traces: Pick<TraceClaims, "claims_state" | "claims">[],
  selected: number[],
): number[] {
  return selected.filter((i) => {
    const trace = traces[i]
    if (!trace || trace.claims_state === "error") return false
    if (trace.claims_state === "built" && !has_verdict_claim(trace))
      return false
    return true
  })
}

export function calibration_gate_target(
  total: number,
  subset_size: number,
): number {
  return Math.min(review_target(total), subset_size)
}

// Stratified pick of k indices from a candidate pool: ~50/50
// judge-pass/judge-fail so the answer key calibrates both classes (random or
// take-first selection degenerates on an imbalanced batch), topped up from
// the other bucket on shortfall, spread evenly across plan order within each
// bucket. Shared by the first-round subset and the calibration top-up.
function stratified_pick(
  traces: Pick<TraceClaims, "judge_score">[],
  candidates: number[],
  k: number,
): number[] {
  if (k >= candidates.length) return [...candidates].sort((a, b) => a - b)
  const fails: number[] = []
  const passes: number[] = []
  for (const i of candidates) {
    ;(traces[i]?.judge_score === "fail" ? fails : passes).push(i)
  }
  // Evenly-spaced picks across a bucket's plan order (k <= bucket.length
  // gives k distinct positions).
  const spread = (bucket: number[], n: number): number[] =>
    Array.from(
      { length: n },
      (_, j) => bucket[Math.floor((j * bucket.length) / n)],
    )
  // Fails get the odd slot: failure examples are usually the scarcer and
  // more informative half of an answer key.
  const want_fail = Math.min(fails.length, Math.ceil(k / 2))
  const want_pass = Math.min(passes.length, k - want_fail)
  const picked = new Set([
    ...spread(fails, want_fail),
    ...spread(passes, want_pass),
  ])
  // Shortfall top-up (one bucket smaller than its quota): fill from the
  // unpicked candidates, keeping the even spread across plan order.
  if (picked.size < k) {
    const unpicked = candidates.filter((i) => !picked.has(i))
    for (const i of spread(unpicked, k - picked.size)) picked.add(i)
  }
  return [...picked].sort((a, b) => a - b)
}

// Deterministic pick of which traces to put in front of the reviewer:
// judge-stratified over the whole batch. Purely mechanical — no LLM in the
// selection loop. This is the exact set the reviewer grades: unselected
// traces are not surfaced in review (they fill the train split unrated).
// Returns ascending indices into `traces`.
export function select_review_subset(
  traces: Pick<TraceClaims, "judge_score">[],
): number[] {
  return stratified_pick(
    traces,
    traces.map((_, i) => i),
    review_target(traces.length),
  )
}

// Which traces a calibration round asks the reviewer to re-grade, sized by
// the same review_target math as the first round. Priority order: traces the
// reviewer previously disagreed on (did the refine fix them?), then traces
// whose verdict flipped under the refined judge (its behavior changed there),
// then a fresh stratified top-up of never-reviewed traces (a held-out check
// against overfitting the judge to one sample). Verdicts are binary, so
// within each stratum stable plan order (ascending index) decides; on
// overflow disagreed take precedence over flips over fresh. Only traces
// holding a fresh verdict this round are eligible — a case that failed to
// re-judge kept a stale result, and nothing stale may be re-presented for
// grading. Returns ascending indices into `traces`.
export function select_calibration_subset(
  traces: Pick<TraceClaims, "judge_score">[],
  round: {
    // Indices the reviewer disagreed with last round.
    disagreed: number[]
    // Indices whose verdict flipped under the refined judge.
    flipped: number[]
    // Indices graded in any earlier round — excluded from the fresh top-up.
    reviewed: number[]
    // Indices that re-judged successfully this round (fresh verdicts).
    judged: number[]
  },
): number[] {
  const target = review_target(traces.length)
  const judged = new Set(round.judged)
  const reviewed = new Set(round.reviewed)
  const picked = new Set<number>()
  const take = (indices: number[]) => {
    for (const i of [...indices].sort((a, b) => a - b)) {
      if (picked.size >= target) return
      if (judged.has(i)) picked.add(i)
    }
  }
  take(round.disagreed)
  take(round.flipped)
  if (picked.size < target) {
    const fresh = round.judged.filter((i) => !picked.has(i) && !reviewed.has(i))
    for (const i of stratified_pick(traces, fresh, target - picked.size)) {
      picked.add(i)
    }
  }
  return [...picked].sort((a, b) => a - b)
}

// ── Save payload (per-claim grades) ──────────────────────────────────────

// The studio save contract IS in the generated schema — alias it (don't
// hand-mirror) so a backend change to the payload shape fails to compile here.
type GradedClaim = components["schemas"]["GradedClaim"]
export type ClaimReviewPayload = components["schemas"]["ClaimReviewApi"]

function graded_claim(claim: Claim, verdict: ClaimVerdict): GradedClaim {
  // An ungraded claim has no honest encoding; refuse rather than write one.
  if (verdict.agrees === null) {
    throw new Error("Cannot record a grade the reviewer never gave.")
  }
  return {
    text: claim.text,
    human_grade: verdict.agrees ? "agree" : "disagree",
    human_feedback: verdict.why.trim() || null,
  }
}

// Build the persisted grades for one reviewed trace: the overview the
// reviewer read, every claim with its grade, and the call derived from the
// verdict claim. Every
// claim is graded by the time this runs (the gate demands it), so the record
// never has to encode "not reviewed". Throws unless the claims are built and
// the trace is fully graded: an invented grade would contradict the golden
// rating written beside it. Callers gate on is_trace_reviewed.
export function build_claim_review_payload(
  trace: TraceClaims,
  review: TraceReview,
): ClaimReviewPayload {
  if (trace.claims_state !== "built" || !trace.claims || !trace.overview) {
    throw new Error("Cannot build a claim review before claims are built.")
  }
  const verdict = human_verdict(trace, review)
  if (!is_trace_reviewed(trace, review) || verdict === null) {
    throw new Error("Cannot build a claim review before the trace is graded.")
  }
  return {
    judge_score: trace.judge_score,
    judge_reasoning: trace.judge_reasoning,
    overview: trace.overview.text,
    claims: trace.claims.map((claim, i) =>
      graded_claim(claim, review.claim_verdicts[i]),
    ),
    human_verdict: verdict,
  }
}

// Concatenated disagree-whys across the claims — the legacy free-text
// feedback field alongside the structured grades.
export function disagreement_feedback(review: TraceReview): string {
  return review.claim_verdicts
    .filter((v) => v.agrees === false && v.why.trim())
    .map((v) => v.why.trim())
    .join(" ")
}

// ── Refine judge loop ─────────────────────────────────────────────────────

// One reviewed trace's grades shaped to feed judge refinement: the persisted
// ClaimReview payload plus a trace_label the refine model cites in rationales.
export type GradedTracePayload = ClaimReviewPayload & { trace_label: string }

// The refine model's proposed edit + its one-line rationale.
type RefineJudgeChange = { change: string; rationale: string }

// The refine loop's response — a PROPOSAL, never auto-applied.
export type RefineJudgeProposal = {
  refined_judge_prompt: string
  changes: RefineJudgeChange[]
  not_incorporated_feedback: string | null
}

// Build the graded-traces payload for the refine call from the in-session
// review. Only fully graded traces with BUILT claims contribute: a
// half-graded trace is no signal, and a trace the reviewer never saw has no
// grades at all. trace_label is the
// durable run id when present, else the client trace id (opaque — the
// refine prompt tolerates that).
export function build_graded_traces(
  traces: TraceClaims[],
  reviews: TraceReview[],
): GradedTracePayload[] {
  return (
    traces
      .map((trace, i) => ({ trace, review: reviews[i] }))
      // is_trace_reviewed demands built claims, so it is the whole filter.
      .filter(({ trace, review }) => review && is_trace_reviewed(trace, review))
      .map(({ trace, review }) => ({
        trace_label: trace.leaf_run_id || trace.trace_id,
        ...build_claim_review_payload(trace, review),
      }))
  )
}

// How many graded traces carry a disagreement on any claim. This is the
// loop's entry predicate as a count: the forward action asks whether to
// improve the judge precisely when a click would otherwise start a
// calibration round.
export function grade_disagreement_count(
  graded: Pick<ClaimReviewPayload, "claims">[],
): number {
  return graded.filter((t) =>
    t.claims.some((c) => c.human_grade === "disagree"),
  ).length
}

// Whether the reviewer pushed back anywhere in the graded set — the signal
// that the judge needs refining before it ships.
export function has_grade_disagreement(
  graded: Pick<ClaimReviewPayload, "claims">[],
): boolean {
  return grade_disagreement_count(graded) > 0
}

// Indices of traces carrying any explicit disagreement on a claim — the
// highest-priority stratum of the next round's subset.
export function disagreed_trace_indices(reviews: TraceReview[]): number[] {
  return reviews
    .map((review, i) => ({ review, i }))
    .filter(({ review }) =>
      review.claim_verdicts.some((v) => v.agrees === false),
    )
    .map(({ i }) => i)
}

// ── Calibration loop ──────────────────────────────────────────────────────
//
// After a review with disagreements, the judge is refined from the grades and
// re-scores the eval data; the reviewer then re-grades against the refined
// judge's verdicts. Both arms run the same round: judge_traces re-judges the
// saved runs by durable id (verdicts only — claims rebuilt lazily for the
// next subset), select_calibration_subset picks what the reviewer re-grades,
// and cases that failed to re-judge keep stale verdicts and sit the round
// out. The helpers below are the loop's pure core — round control, verdict
// flips, state rebuild — so the wizard component only wires streams and
// screens around them.

// One case's fresh verdict from a re-judge round. raw_input/raw_output/
// trace are the stream's echoes of the reloaded run — the same content as
// drive time, re-echoed so citations and the trace modal stay anchored to
// exactly what this round's judge saw.
export type RejudgeCaseResult = {
  judge_score: JudgeScore
  judge_reasoning: string
  raw_input: string
  raw_output: string
  trace?: TraceMessage[] | null
}

// Indices whose pass/fail changed under the refined judge. Only re-judged
// cases can flip: a case with no fresh verdict is unknown, not unchanged.
export function flipped_indices(
  traces: Pick<TraceClaims, "judge_score">[],
  results: Map<number, Pick<RejudgeCaseResult, "judge_score">>,
): number[] {
  return [...results.entries()]
    .filter(([i, r]) => traces[i] && traces[i].judge_score !== r.judge_score)
    .map(([i]) => i)
    .sort((a, b) => a - b)
}

// Fold a re-judge round's verdicts into the trace list. Re-judged traces get
// the fresh verdict and their claims reset to unbuilt — claims argue a
// specific verdict, so they must be rebuilt against the new one. The new
// trace_id (unique per round) makes any still-in-flight claim build from the
// previous round miss the identity guard instead of corrupting the fresh
// state. Cases absent from `results` failed to re-judge and stay untouched.
export function apply_rejudge_results(
  traces: TraceClaims[],
  results: Map<number, RejudgeCaseResult>,
  round_tag: string,
): TraceClaims[] {
  return traces.map((t, i) => {
    const result = results.get(i)
    if (!result) return t
    return {
      ...t,
      trace_id: `${round_tag}_case_${i}`,
      judge_score: result.judge_score,
      judge_reasoning: result.judge_reasoning,
      raw_input: result.raw_input,
      raw_output: result.raw_output,
      trace: result.trace ?? t.trace,
      overview: null,
      claims: null,
      claims_state: "unbuilt",
      claims_error: null,
    }
  })
}

// What a save request should do next. A save with disagreement enters a
// calibration round on either arm — as many rounds as it takes, since the
// loop only exits on convergence or the explicit save-without-refining link.
// Arm-independent: unaddressed disagreement may never ship unseen.
export type SaveAction = { action: "save" } | { action: "calibrate" }

export function plan_save_action(args: {
  has_disagreement: boolean
}): SaveAction {
  return args.has_disagreement ? { action: "calibrate" } : { action: "save" }
}

// The honest shortfall notice when some eval data couldn't be re-checked: it
// kept stale verdicts, so it was left out of the round. `total` is every
// result carried over from the previous round, the ones with no id included —
// those cannot be re-checked either, so they count as failures here. Counting
// rather than naming keeps one sentence true on both arms.
export function rejudge_shortfall_notice(
  failed: number,
  total: number,
): string | null {
  if (failed <= 0) return null
  return `${failed} of ${total} couldn't be re-checked with the improved judge and kept their previous results. They were left out of this review round.`
}

// A judge prompt/rubric this long is almost certainly runaway model output,
// not a rubric — reject it rather than persist it into the judge config.
export const MAX_JUDGE_PROMPT_CHARS = 20000

// Mechanically validate a refined judge prompt before it is written into the
// judge config. The prompt is inserted into the judge harness verbatim (then
// raw-wrapped server-side), so it must be plain text: non-empty, no Jinja /
// template braces, no code fences, and a sane length. Returns an error message
// to surface to the user, or null when the prompt is safe to apply. The
// refined prompt is a PROPOSAL from an LLM — never trust it into a write blind.
export function validate_refined_judge_prompt(prompt: string): string | null {
  const text = (prompt ?? "").trim()
  if (!text) return "The refined judge prompt is empty."
  if (text.length > MAX_JUDGE_PROMPT_CHARS) {
    return `The refined judge prompt is too long (${text.length} characters; max ${MAX_JUDGE_PROMPT_CHARS}).`
  }
  const forbidden: [RegExp, string][] = [
    [/\{\{|\}\}/, "Jinja expression braces ({{ or }})"],
    [/\{%|%\}/, "Jinja statement braces ({% or %})"],
    [/\{#|#\}/, "Jinja comment braces ({# or #})"],
    [/```/, "code fences"],
  ]
  for (const [pattern, label] of forbidden) {
    if (pattern.test(text)) {
      return `The refined judge prompt contains ${label}; it must be plain text.`
    }
  }
  return null
}

// Strip a single code fence that WRAPS the whole prompt (opening fence with
// an optional language tag on its own first line, closing fence alone on the
// last), returning the inner text. A model that fence-wraps an otherwise-good
// prompt is a recoverable presentation slip, not bad content, so unwrapping it
// saves re-paying for the same answer. It recovers the wrapping ONLY: anything
// else — interior or one-sided fences, a second wrapping pair, Jinja braces,
// length — is left in place to fail validation as before.
export function strip_wrapping_code_fence(prompt: string): string {
  const text = (prompt ?? "").trim()
  const wrapped = /^```[^\n`]*\n([\s\S]*)\n```$/.exec(text)
  return wrapped ? wrapped[1] : prompt
}
