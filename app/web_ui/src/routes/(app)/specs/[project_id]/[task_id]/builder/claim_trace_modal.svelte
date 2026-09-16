<script lang="ts">
  // Trace modal for the claim review. Hidden by default; opened either to
  // view a whole trace or jumped to a specific [n] citation (from the overview
  // or a claim, the same citation shape either way), where it scrolls to and
  // highlights the cited span. Most reviewers never need the full trace; this
  // is the escape hatch for hard calls.
  //
  // Two arms, and the review component tells this one which it is on:
  //
  //   SINGLE-TURN renders the same two sections as the inline review surface
  //   (Input field, Output section) — one shape for one trace, whether it is
  //   read on the page or behind the button. A citation marks inside the
  //   section it cites.
  //
  //   MULTI-TURN renders the conversation alone in the house chat UI, with the
  //   citation mapped onto the exact chat node. No input panel: the opening
  //   user message IS the input, so a panel above would print it twice. The
  //   dialog title is the only framing, as on the run pages.
  //   A trace with no stored structure keeps the raw flattened panels.
  import { tick } from "svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import Output from "$lib/ui/output.svelte"
  import SettingsHeader from "$lib/ui/settings_header.svelte"
  import ChatTrace from "$lib/ui/trace/chat_trace.svelte"
  import {
    map_input_span_to_trace,
    map_output_span_to_trace,
    resolve_citation_span_whitespace_tolerant,
    type Citation,
    type CitationSource,
    type TraceClaims,
    type TraceHighlight,
  } from "./claim_evidence"

  let dialog: Dialog | null = null
  let trace: TraceClaims | null = null
  let content_el: HTMLElement | null = null
  // The active citation, if opened via one. active_span is its resolved span in
  // the raw source text (drives the <mark>); active_source says which side of
  // the trace it cites; null when just browsing.
  let active_source: CitationSource | null = null
  let active_span: { start: number; end: number } | null = null
  let active_citation: Citation | null = null

  // ── Single-turn ──────────────────────────────────────────────────────
  // The two sections, memoized on the trace content so reopening the dialog on
  // the same trace hands the rows the same array and keeps their expansion.
  // `error` is the trace having nothing to render at all; a missing output
  // alone is reported inside the Output section instead.

  // ── Multi-turn ───────────────────────────────────────────────────────
  // Render the chat UI when the trace carries the conversation; otherwise fall
  // back to the raw flattened input and output panels. One rendering for both
  // arms: a single-turn run is a conversation of one turn, and its tool calls
  // and results are the same nodes a multi-turn trace carries.
  $: use_chat = !!(trace && trace.trace && trace.trace.length > 0)

  function text_for(source: CitationSource): string {
    if (!trace) return ""
    return source === "input" ? trace.raw_input : trace.raw_output
  }

  // Map a citation onto the structured trace so ChatTrace can mark the exact
  // node: output citations through the flattener layout, input citations onto
  // the conversation's opening user message (on multi-turn the input IS that
  // message). Null for legacy traces or unmappable spans — ChatTrace then
  // renders without a highlight rather than a wrong one, and the miss is
  // logged so the silence is observable.
  //
  // The anchors resolve through the whitespace-tolerant resolver, which absorbs
  // the retyping drift a model introduces. That cannot mis-place a mark here:
  // the mapper's byte-identity guards still adjudicate whatever it returns.
  $: chat_highlight = compute_chat_highlight(
    trace,
    active_source,
    active_citation,
  )
  function compute_chat_highlight(
    t: TraceClaims | null,
    source: CitationSource | null,
    citation: Citation | null,
  ): TraceHighlight | null {
    // An empty trace renders the raw panels (which carry their own mark),
    // so it is not a mapping miss and must not warn.
    if (!t || !t.trace || t.trace.length === 0 || !source || !citation) {
      return null
    }
    const raw = source === "input" ? t.raw_input : t.raw_output
    const span = resolve_citation_span_whitespace_tolerant(raw, citation)
    if (!span) {
      warn_citation("anchor_not_found", citation, source)
      return null
    }
    const mapped =
      source === "input"
        ? map_input_span_to_trace(t.trace, t.raw_input, span)
        : map_output_span_to_trace(t.trace, t.raw_output, span)
    if (!mapped) {
      // A span that resolves in the raw text but sits in no recomputed block,
      // or one a byte-identity guard rejected: either way our flattener port
      // no longer matches what the server rendered.
      warn_citation("flattener_drift", citation, source)
      return null
    }
    if (!chat_renders_highlight(t.trace, mapped)) {
      // The flattener emits blocks for rows the chat drops (system /
      // developer), so a span can map onto a node ChatTrace will never
      // draw — passing it through would be a highlight with no target.
      warn_citation("row_not_rendered", citation, source)
      return null
    }
    if (mapped.from_anchor_only) {
      // The highlight is honest but partial. Worth logging anyway: a citation
      // whose two anchors sit in different turns is a model-side defect.
      warn_citation("spans_two_turns", citation, source)
    }
    return mapped
  }

  // One shape for every citation-mapping warning. The reason code is in the
  // message so a console filter finds it, and the anchors travel with it so the
  // miss can be reproduced without re-driving the trace.
  function warn_citation(
    reason: string,
    citation: Citation,
    source: CitationSource,
  ) {
    console.warn(`Citation highlight issue (${reason}).`, {
      marker: citation.marker,
      source,
      from: citation.from,
      to: citation.to,
    })
  }

  // Whether ChatTrace draws a row that can carry this highlight: tool
  // results render through their owning call's bubble, everything else
  // needs its own row, which system/developer/tool messages never get.
  function chat_renders_highlight(
    trace_messages: NonNullable<TraceClaims["trace"]>,
    h: TraceHighlight,
  ): boolean {
    if (h.kind === "tool_result") return true
    const message = trace_messages[h.trace_index]
    const role =
      message && "role" in message && typeof message.role === "string"
        ? message.role
        : ""
    return role !== "system" && role !== "developer" && role !== "tool"
  }

  // The scroll container survives across opens (trace stays set after close),
  // so it keeps whatever offset the last view left. Call position is
  // load-bearing: after show() (before it the dialog has no layout box, and
  // the write is a no-op) and before the awaited tick (the citation scrolls
  // all run after that tick, so the reset can never clobber them). On the
  // very first open it is not rendered yet and starts at the top anyway.
  function reset_scroll() {
    if (content_el) content_el.scrollTop = 0
  }

  export function open_trace(t: TraceClaims) {
    trace = t
    active_source = null
    active_span = null
    active_citation = null
    dialog?.show()
    reset_scroll()
  }

  export async function open_citation(t: TraceClaims, citation: Citation) {
    trace = t
    active_source = citation.source
    active_citation = citation
    active_span = resolve_citation_span_whitespace_tolerant(
      text_for(citation.source),
      citation,
    )
    dialog?.show()
    reset_scroll()
    // Wait for the <mark> to render, then bring it into view. Only the raw
    // panels need this: on multi-turn the chat panel scrolls itself (ChatTrace
    // reacts to its highlight prop).
    await tick()
    content_el
      ?.querySelector("[data-highlight-target]")
      ?.scrollIntoView({ block: "center", behavior: "smooth" })
  }

  // The raw panels only render when there is no conversation to show, so they
  // are also the only place a citation mark can land on that path. When the
  // chat renders, an unmappable citation shows no mark at all rather than one
  // on a duplicate copy of the text. The span is handed to Output, which owns
  // the mark and translates it onto the printed form when the text is JSON.
  $: input_mark =
    trace && !use_chat && active_source === "input" ? active_span : null
  $: output_mark =
    trace && !use_chat && active_source === "output" ? active_span : null
</script>

<Dialog bind:this={dialog} title="Trace" width="extra_wide">
  {#if trace}
    <!-- Sized to content, capped at 80% of the window. The second term is a
         fit guard for short windows: daisyUI caps modal-box at 100vh minus
         5rem and the box spends 6rem around this div (3rem padding, 2rem
         title row, 1rem top margin), so a pane over calc(100vh-11rem) would
         grow the box a second scrollbar; 12rem leaves 1rem of slack. The
         guard term wins below ~960px of window height, and is what keeps
         the box from double-scrolling below ~880px. -->
    <div
      class="space-y-4 text-sm max-h-[min(80vh,calc(100vh-12rem))] overflow-y-auto"
      bind:this={content_el}
    >
      {#if use_chat && trace.trace}
        <!-- The conversation on its own, as the run pages show it, with the
             citation mapped onto its exact node (or unhighlighted when just
             browsing / unmappable). -->
        <ChatTrace trace={trace.trace} highlight={chat_highlight} />
      {:else}
        <!-- No conversation recorded: the raw flattened input and output are
             the only rendering this trace has. -->
        <div>
          <SettingsHeader title="Input" />
          <div class="mt-2">
            <Output raw_output={trace.raw_input} mark={input_mark} />
          </div>
        </div>

        <div>
          <SettingsHeader title="Output" />
          <div class="mt-2">
            <Output raw_output={trace.raw_output} mark={output_mark} />
          </div>
        </div>
      {/if}
    </div>
  {/if}
</Dialog>
