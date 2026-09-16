<script context="module" lang="ts">
  // Whether a value should render as structured JSON rather than plain text.
  // A quoted string parses as JSON but is really prose, so it stays text.
  // Exported so other surfaces route content the same way this component does.
  export function is_non_string_json(text: string): boolean {
    try {
      return typeof JSON.parse(text) !== "string"
    } catch (_) {
      return false
    }
  }
</script>

<script lang="ts">
  import { onMount, onDestroy } from "svelte"
  import hljs from "highlight.js/lib/core"
  import json from "highlight.js/lib/languages/json"
  import { map_json_span, mark_html_range } from "$lib/ui/json_span_map"
  hljs.registerLanguage("json", json)

  export let raw_output: string
  export let max_height: string | null = null
  export let hide_toggle: boolean = false
  export let show_border: boolean = false
  // "transparent" lets a caller show JSON on its own tinted surface. Pair it
  // with max_height at your peril: the overflow fade and Show All button
  // paint base-200 and will float visibly over a non-default background.
  export let background_color: "default" | "white" | "transparent" = "default"

  export let no_padding: boolean = false
  // A citation's span, in the coordinates of `raw_output` — the text whoever
  // made the citation actually read. JSON is re-printed for display, so the
  // span is translated onto the printed form rather than used directly.
  // Null for every caller that isn't showing a citation.
  export let mark: { start: number; end: number } | null = null
  let formatted_json_html: string | null = null
  let is_expanded = false
  let content_element: HTMLElement
  let is_content_overflowing = false
  let resize_observer: ResizeObserver | null = null

  // Guarded by the predicate, so the parse here cannot throw.
  $: printed_json = is_non_string_json(raw_output)
    ? JSON.stringify(JSON.parse(raw_output), null, 2)
    : null
  $: formatted_json_html = printed_json
    ? hljs.highlight(printed_json, { language: "json" }).value
    : null

  // The mark rides on top of the normal rendering rather than replacing it:
  // the JSON looks the same cited or not, which is the point. A span that
  // cannot be translated yields no mark instead of a wrong one.
  $: display_json_html =
    formatted_json_html && printed_json && mark
      ? mark_printed_json(formatted_json_html, printed_json, mark)
      : formatted_json_html
  function mark_printed_json(
    html: string,
    printed: string,
    m: { start: number; end: number },
  ): string {
    const translated = map_json_span(raw_output, printed, m)
    if (!translated) return html
    return mark_html_range(
      html,
      translated.start,
      translated.end,
      JSON_MARK_CLASS,
    )
  }

  // Plain text needs no translation: the span already indexes what is shown.
  $: text_segments =
    mark && !formatted_json_html
      ? {
          before: raw_output.slice(0, Math.max(0, mark.start)),
          mark: raw_output.slice(Math.max(0, mark.start), mark.end),
          after: raw_output.slice(mark.end),
        }
      : null

  const MARK_CLASS = "bg-warning/40 rounded px-0.5"
  // Marking inside highlighted JSON emits one <mark> per coloured run, so the
  // pieces must butt together: padding and rounding would draw a seam between
  // every token of a single citation.
  const JSON_MARK_CLASS = "bg-warning/40"

  function compute_overflow(
    elem: HTMLElement | undefined,
    maxHeight: string | null,
    expanded: boolean,
  ) {
    if (!elem || !maxHeight || expanded) {
      is_content_overflowing = false
      return
    }

    const temp_element = elem.cloneNode(true) as HTMLElement
    temp_element.style.maxHeight = "none"
    temp_element.style.position = "absolute"
    temp_element.style.visibility = "hidden"
    temp_element.style.pointerEvents = "none"
    // Match current rendered width so wrapping matches the on-screen layout
    temp_element.style.width = `${elem.clientWidth}px`
    document.body.appendChild(temp_element)

    const natural_height = temp_element.scrollHeight
    document.body.removeChild(temp_element)

    const max_height_px = parseInt(maxHeight.replace("px", ""))
    is_content_overflowing = natural_height > max_height_px
  }

  // Recompute when inputs change
  $: compute_overflow(content_element, max_height, is_expanded)

  onMount(() => {
    if (!hide_toggle && max_height !== null) {
      // scenario where this matters: content initially fits when the container is at full width,
      // but on smaller viewport, the container is narrower and the content now overflows
      resize_observer = new ResizeObserver(() => {
        compute_overflow(content_element, max_height, is_expanded)
      })
      resize_observer.observe(content_element)
    }
  })

  onDestroy(() => {
    resize_observer?.disconnect()
  })

  function copy_to_clipboard() {
    navigator.clipboard.writeText(raw_output)
  }

  function toggle_expansion() {
    is_expanded = !is_expanded
  }
</script>

<head>
  <link rel="stylesheet" href="/styles/highlightjs.min.css" />
</head>

<div class="relative {show_border ? 'border rounded-lg' : ''}" translate="no">
  <div
    class="flex flex-row gap-2 {background_color === 'white'
      ? 'bg-white'
      : background_color === 'transparent'
        ? ''
        : 'bg-base-200'} p-1 rounded-lg {no_padding ? '' : 'p-1'} {max_height &&
    !is_expanded
      ? 'overflow-hidden'
      : ''}"
    style={max_height && !is_expanded ? `max-height: ${max_height}` : ""}
  >
    {#if $$slots.default}
      <!-- Slotted content takes the place of the printed text, inside this
           same surface: the caller owns what it renders and its typography,
           this control owns the panel, the fold and the copy button. The copy
           button still copies raw_output, so a caller that slots a rendering
           of its text passes that text as the prop. `mark` does nothing on
           this path: the span indexes raw_output, which is not what is on
           screen, so a caller that needs a highlight marks it itself. -->
      <!-- no_padding is honoured here and ignored on the printing path below,
           where p-3 is unconditional. That is a bug in the printing path, but
           eight shipped trace-viewer callers pass no_padding and render around
           the padding they actually get, so fixing it is the control owner's
           call, not this slot's. -->
      <div
        bind:this={content_element}
        class="grow min-w-0 {no_padding ? '' : 'p-3'}"
      >
        <slot />
      </div>
    {:else}
      <!-- eslint-disable svelte/no-at-html-tags -->
      <pre
        bind:this={content_element}
        class="grow p-3 whitespace-pre-wrap text-xs min-w-0 {no_padding
          ? ''
          : 'p-3'}"
        style="overflow-wrap: anywhere;">{#if display_json_html}{@html display_json_html}{:else if text_segments}{text_segments.before}<mark
            data-highlight-target
            class={MARK_CLASS}>{text_segments.mark}</mark
          >{text_segments.after}{:else}{raw_output}{/if}</pre>
      <!-- eslint-enable svelte/no-at-html-tags -->
    {/if}
    <div class="flex-none">
      <button
        on:click|stopPropagation={copy_to_clipboard}
        class="btn btn-sm btn-square h-8 w-8 shadow-none text-gray-400 hover:text-gray-900"
      >
        <!-- Uploaded to: SVG Repo, www.svgrepo.com, Generator: SVG Repo Mixer Tools -->
        <svg
          class="w-5 h-5 p-0"
          viewBox="0 0 64 64"
          xmlns="http://www.w3.org/2000/svg"
          stroke-width="3"
          stroke="currentColor"
          fill="none"
        >
          <rect x="11.13" y="17.72" width="33.92" height="36.85" rx="2.5" />
          <path
            d="M19.35,14.23V13.09a3.51,3.51,0,0,1,3.33-3.66H49.54a3.51,3.51,0,0,1,3.33,3.66V42.62a3.51,3.51,0,0,1-3.33,3.66H48.39"
          />
        </svg>
      </button>
    </div>
  </div>

  <!-- Toggle bar with gradient - only show when content overflows -->
  {#if max_height && is_content_overflowing && !is_expanded}
    <div
      class="absolute bottom-0 left-0 right-0 flex items-end justify-center pb-2 bg-gradient-to-t from-base-200 via-base-200/80 to-transparent h-12 pointer-events-none"
    >
      <button
        on:click={toggle_expansion}
        class="btn btn-xs btn-outline pointer-events-auto bg-base-200 {hide_toggle
          ? 'hidden'
          : ''}"
      >
        Show All
      </button>
    </div>
  {/if}

  <!-- Hide toggle when expanded -->
  {#if max_height && is_expanded && !hide_toggle}
    <div class="flex justify-center pt-2 pb-2">
      <button on:click={toggle_expansion} class="btn btn-xs btn-outline">
        Collapse
      </button>
    </div>
  {/if}
</div>
