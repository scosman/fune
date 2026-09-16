<script lang="ts">
  import { tick } from "svelte"
  import type { Trace, TraceMessage, ToolCallMessageParam } from "$lib/types"
  // Structured content renders through Output (pretty-print + syntax highlight)
  // rather than markdown, which collapses 2-space-indented JSON into a run-on
  // paragraph. Output owns the rule so every surface routes content the same.
  import Output, { is_non_string_json } from "$lib/ui/output.svelte"
  import ChatMarkdown from "$lib/ui/chat/chat_markdown.svelte"
  import { TASK_RESPONSE_TOOL_NAME } from "$lib/utils/task_response_tool"
  import ArrowRightUpIcon from "../icons/arrow_right_up_icon.svelte"
  import ChatMessageActions from "./chat_message_actions.svelte"
  import ToolCall from "./tool_call.svelte"
  import ToolMessagesDialog from "./tool_messages_dialog.svelte"
  import UsageInfoDialog from "./usage_info_dialog.svelte"

  export let trace: Trace
  export let project_id: string | undefined = undefined
  // Positional map from trace index to a TaskRun id for that user turn.
  export let forkable_run_ids: (string | null)[] | undefined = undefined
  // When set, messages at trace indices >= this value are hidden.
  export let truncate_at_trace_index: number | null = null
  // Invoked when the user clicks a fork affordance on a user block.
  export let on_fork:
    | ((run_id: string, trace_index: number) => void)
    | undefined = undefined
  // Show the per-message usage info button.
  export let show_per_message_usage: boolean = false
  // Optional citation highlight: a resolved span pointing at one block of one
  // message. When set, that node renders the span wrapped in <mark>, the
  // component scrolls to it, and any collapsed thinking/tool bubble it lives in
  // auto-expands. When null (the default — e.g. the dataset run page), the
  // render is unchanged. `start`/`end` index the block's RAW text, matching the
  // flattener the citation resolved against; content, reasoning, and
  // tool-result nodes render that raw text (markdown/pretty-print is dropped
  // for the marked node only) so the offsets line up, while tool-call blocks
  // just expand + scroll to the cited bubble.
  export let highlight: {
    trace_index: number
    kind: "content" | "reasoning" | "tool_calls" | "tool_result"
    start: number
    end: number
  } | null = null

  let root_el: HTMLElement | null = null

  // Split a block's raw text around the highlight span for <mark> rendering.
  function highlight_segments(
    text: string,
    h: { start: number; end: number },
  ): { before: string; mark: string; after: string } {
    const start = Math.max(0, Math.min(h.start, text.length))
    const end = Math.max(start, Math.min(h.end, text.length))
    return {
      before: text.slice(0, start),
      mark: text.slice(start, end),
      after: text.slice(end),
    }
  }

  // The raw text a tool-result citation's offsets index — the same unwrap
  // the citation mapper verified byte-for-byte against the flattened
  // transcript: a string `output` field of the Kiln tool JSON envelope, or
  // the message content as-is. Null when the displayed rendering diverges
  // from that raw text — the mark cannot be placed there and the bubble
  // stays the scroll target instead. Must track content_from_message's
  // unwrapping: any shape it displays differently from the flattener's raw
  // text has to return null here.
  function tool_result_citation_text(message: TraceMessage): string | null {
    if (
      !("content" in message) ||
      typeof message.content !== "string" ||
      !message.content
    ) {
      return null
    }
    try {
      const parsed = JSON.parse(message.content)
      if (parsed && typeof parsed === "object" && "output" in parsed) {
        // Non-string output can't be cited (the flattener emits no block
        // for it), so the null is defensive symmetry with the display's
        // JSON.stringify branch, not a reachable divergence.
        return typeof parsed.output === "string" ? parsed.output : null
      }
      if (
        parsed &&
        typeof parsed === "object" &&
        parsed.isError === true &&
        "error" in parsed
      ) {
        // The chat displays the unwrapped error while citation offsets
        // index the raw JSON — a mark in either would be misplaced. An
        // isError envelope WITHOUT an error field displays as the raw
        // JSON, which the offsets index, so it falls through and marks.
        return null
      }
    } catch (_) {
      // Not JSON — offsets index the content itself.
    }
    return message.content
  }

  // A tool-RESULT citation's trace_index is the tool message; find the
  // assistant turn + tool-call slot that renders it, so we can expand it.
  function owner_of_tool_result(
    tool_index: number,
  ): { index: number; tcIdx: number } | null {
    const m = trace[tool_index]
    const tid =
      m && "tool_call_id" in m && typeof m.tool_call_id === "string"
        ? m.tool_call_id
        : null
    if (!tid) return null
    for (let i = 0; i < trace.length; i++) {
      const tcs = tool_calls_from_message(trace[i])
      if (!tcs) continue
      for (let tcIdx = 0; tcIdx < tcs.length; tcIdx++) {
        if (tcs[tcIdx].id === tid) return { index: i, tcIdx }
      }
    }
    return null
  }

  // Expand whatever collapsible bubble the highlight lives in, so the cited
  // moment is visible before we scroll to it.
  $: apply_highlight_expansion(highlight)
  function apply_highlight_expansion(h: typeof highlight): void {
    if (!h) return
    if (h.kind === "reasoning") {
      thinkingExpanded[h.trace_index] = true
      thinkingExpanded = thinkingExpanded
    } else if (h.kind === "tool_calls") {
      const tcs = tool_calls_from_message(trace[h.trace_index]) ?? []
      tcs.forEach(
        (_, tcIdx) => (toolCallExpanded[`${h.trace_index}-${tcIdx}`] = true),
      )
      toolCallExpanded = toolCallExpanded
    } else if (h.kind === "tool_result") {
      const owner = owner_of_tool_result(h.trace_index)
      if (owner) toolCallExpanded[`${owner.index}-${owner.tcIdx}`] = true
      toolCallExpanded = toolCallExpanded
    }
  }

  // Bring the marked node (or the expanded target bubble) into view whenever
  // the highlight changes — the modal reuses one component across citations.
  $: scroll_to_highlight(highlight)
  async function scroll_to_highlight(h: typeof highlight): Promise<void> {
    if (!h || typeof document === "undefined") return
    await tick()
    const el = root_el?.querySelector("[data-highlight-target]")
    if (el && typeof el.scrollIntoView === "function") {
      el.scrollIntoView({ block: "center", behavior: "smooth" })
    }
  }

  // The assistant turn + tool-call slot a tool-result highlight targets.
  $: tool_result_owner =
    highlight && highlight.kind === "tool_result"
      ? owner_of_tool_result(highlight.trace_index)
      : null

  // Whether the highlighted node draws a <mark>, or only expands and scrolls to
  // the bubble: a tool-CALL block renders through a component rather than plain
  // text, and a tool result falls back to its bubble when the displayed text is
  // not the raw string the offsets index.
  $: highlight_draws_mark = draws_highlight_mark(
    highlight,
    tool_result_owner,
    tool_results_by_call_id,
  )
  function draws_highlight_mark(
    h: typeof highlight,
    owner: { index: number; tcIdx: number } | null,
    results: Map<string, { message: TraceMessage; trace_index: number }>,
  ): boolean {
    if (!h) return true
    if (h.kind === "tool_calls") {
      const calls = tool_calls_from_message(trace[h.trace_index]) ?? []
      return calls.some(
        (_, i) => tool_call_argument_mark(h, h.trace_index, i) !== null,
      )
    }
    if (h.kind !== "tool_result") return true
    if (!owner) return false
    const call = tool_calls_from_message(trace[owner.index])?.[owner.tcIdx]
    const result = call ? results.get(call.id) ?? null : null
    if (!result || result.trace_index !== h.trace_index) return false
    return tool_result_citation_text(result.message) !== null
  }

  // Log the scroll-only case. A citation that expands a bubble and draws
  // nothing looks identical to a broken one, so the silence has to end here.
  $: if (highlight && !highlight_draws_mark) {
    console.warn("Citation highlight issue (no_mark_drawn).", {
      trace_index: highlight.trace_index,
      kind: highlight.kind,
    })
  }

  let thinkingExpanded: Record<number, boolean> = {}
  // Keyed by `${trace_index}-${tool_call_index}` so each tool call within an
  // assistant message expands independently.
  let toolCallExpanded: Record<string, boolean> = {}
  // A structured-output task returns its answer as a call to the internal
  // `task_response` tool — the adapter strips that name from the real tool
  // flow for exactly this reason. The user never wrote such a tool, so showing
  // it as a tool call presents plumbing as something the agent chose to do.
  // It is the run's OUTPUT, and renders the way output renders everywhere
  // else in the app.
  // A tool-call citation's offsets index the FLATTENED block the judge read:
  // `- Tool Name: {name}\n- Arguments: {args}` per call, concatenated with no
  // separator (EvalTraceFormatter.formatted_tool_calls_from_message). Resolve
  // the span onto ONE call's arguments so the mark can be drawn there.
  function tool_call_argument_mark(
    h: typeof highlight,
    message_index: number,
    tcIdx: number,
  ): { start: number; end: number } | null {
    if (!h || h.kind !== "tool_calls" || h.trace_index !== message_index) {
      return null
    }
    const calls = tool_calls_from_message(trace[message_index])
    if (!calls) return null
    let offset = 0
    for (let i = 0; i < calls.length; i++) {
      const args = calls[i].function.arguments ?? ""
      const args_start =
        offset + `- Tool Name: ${calls[i].function.name}\n- Arguments: `.length
      const args_end = args_start + args.length
      if (i === tcIdx) {
        // Clipped to the arguments on purpose: a span reaching into the tool
        // NAME half has no home in the rendered card, and marking the wrong
        // bytes is worse than marking none.
        if (h.start < args_start || h.end > args_end || h.end < h.start) {
          return null
        }
        return { start: h.start - args_start, end: h.end - args_start }
      }
      offset = args_end
    }
    return null
  }

  function is_internal_answer(tc: ToolCallMessageParam): boolean {
    return tc.function?.name === TASK_RESPONSE_TOOL_NAME
  }
  let tool_messages_dialog: ToolMessagesDialog | null = null
  let usage_info_dialog: UsageInfoDialog | null = null

  // Build a map: tool_call_id -> tool message so we can nest tool results
  // under the assistant turn that requested them.
  $: tool_results_by_call_id = (() => {
    const m = new Map<string, { message: TraceMessage; trace_index: number }>()
    trace.forEach((message, idx) => {
      if (
        message.role === "tool" &&
        "tool_call_id" in message &&
        typeof message.tool_call_id === "string"
      ) {
        m.set(message.tool_call_id, { message, trace_index: idx })
      }
    })
    return m
  })()

  function content_from_message(message: TraceMessage): string | undefined {
    if (
      "content" in message &&
      message.content &&
      typeof message.content === "string"
    ) {
      if (message.role === "tool") {
        try {
          const parsed = JSON.parse(message.content)
          if (parsed && typeof parsed === "object" && "output" in parsed) {
            return typeof parsed.output === "string"
              ? parsed.output
              : JSON.stringify(parsed.output, null, 2)
          }
          if (
            parsed &&
            typeof parsed === "object" &&
            parsed.isError === true &&
            "error" in parsed
          ) {
            return typeof parsed.error === "string"
              ? parsed.error
              : JSON.stringify(parsed.error, null, 2)
          }
        } catch (_) {
          // Not JSON, return as-is.
        }
      }
      return message.content
    }
    return undefined
  }

  function tool_calls_from_message(
    message: TraceMessage,
  ): ToolCallMessageParam[] | undefined {
    if (
      "tool_calls" in message &&
      message.tool_calls &&
      message.tool_calls.length > 0
    ) {
      return message.tool_calls
    }
    return undefined
  }

  function reasoning_from_message(message: TraceMessage): string | undefined {
    if (
      "reasoning_content" in message &&
      message.reasoning_content &&
      typeof message.reasoning_content === "string"
    ) {
      return message.reasoning_content
    }
    return undefined
  }

  function is_tool_error(message: TraceMessage): boolean {
    if (message.role !== "tool") return false
    if ("is_error" in message && message.is_error) return true
    if ("content" in message && typeof message.content === "string") {
      try {
        const parsed = JSON.parse(message.content)
        if (parsed && typeof parsed === "object" && parsed.isError === true) {
          return true
        }
      } catch (_) {
        // Not JSON.
      }
    }
    return false
  }

  function kiln_task_tool_data_from_message(message: TraceMessage): {
    project_id: string
    tool_id: string
    task_id: string
    run_id: string
  } | null {
    if (
      "kiln_task_tool_data" in message &&
      message.kiln_task_tool_data &&
      typeof message.kiln_task_tool_data === "string"
    ) {
      const [p_id, tool_id, task_id, run_id] =
        message.kiln_task_tool_data.split(":::")
      if (p_id && tool_id && task_id && run_id) {
        return { project_id: p_id, tool_id, task_id, run_id }
      }
    }
    return null
  }

  function message_usage(message: TraceMessage) {
    if ("usage" in message && message.usage) return message.usage
    return null
  }

  function message_latency_ms(message: TraceMessage): number | null {
    if (
      "latency_ms" in message &&
      typeof message.latency_ms === "number" &&
      message.latency_ms > 0
    ) {
      return message.latency_ms
    }
    return null
  }

  function has_usage_info(message: TraceMessage): boolean {
    return (
      message_usage(message) !== null || message_latency_ms(message) !== null
    )
  }

  function open_usage_dialog(message: TraceMessage) {
    usage_info_dialog?.show({
      usage: message_usage(message),
      latency_ms: message_latency_ms(message),
    })
  }
</script>

<div class="flex flex-col gap-3 w-full" bind:this={root_el}>
  {#each trace as message, index}
    {#if (truncate_at_trace_index === null || index < truncate_at_trace_index) && message.role !== "tool" && message.role !== "system" && message.role !== "developer"}
      {@const fork_run_id = forkable_run_ids?.[index] ?? null}
      {@const show_fork = !!(message.role !== "user" && fork_run_id && on_fork)}
      {@const show_info = show_per_message_usage && has_usage_info(message)}
      {@const content = content_from_message(message)}
      {@const reasoning = reasoning_from_message(message)}
      {@const tool_calls = tool_calls_from_message(message)}
      {@const has_reasoning_bubble = !!reasoning}
      {@const has_content_bubble = !!content}
      {@const has_tc_bubble = !!(tool_calls && tool_calls.length > 0)}
      {@const empty_assistant =
        !has_reasoning_bubble &&
        !has_content_bubble &&
        !has_tc_bubble &&
        message.role !== "user"}

      {#if message.role === "user"}
        <!-- `group` so the actions row below reveals on hover/focus. -->
        <div class="group flex flex-col items-end" data-testid="chat-msg-user">
          <div class="rounded-xl bg-primary/10 px-4 py-3 max-w-[70%] text-sm">
            {#if content}
              {#if highlight && highlight.kind === "content" && highlight.trace_index === index}
                {@const seg = highlight_segments(content, highlight)}
                <!-- Marked node renders as plain text: the offsets index the
                     raw content, not rendered markdown. -->
                <div class="whitespace-pre-wrap">
                  {seg.before}<mark
                    data-highlight-target
                    class="bg-warning/40 rounded px-0.5">{seg.mark}</mark
                  >{seg.after}
                </div>
              {:else if is_non_string_json(content)}
                <!-- Transparent so the user turn keeps its tint. Markdown code
                     blocks already render on the tint here, so a JSON panel on
                     it matches the bubble's existing visual language. -->
                <Output
                  raw_output={content}
                  no_padding={true}
                  background_color="transparent"
                />
              {:else}
                <ChatMarkdown text={content} />
              {/if}
            {:else}
              <span class="text-gray-400 italic">(empty message)</span>
            {/if}
          </div>
          {#if show_info}
            <ChatMessageActions
              align="end"
              show_usage={show_info}
              show_fork={false}
              on_usage={() => open_usage_dialog(message)}
            />
          {/if}
        </div>
      {:else}
        <!-- One group per assistant turn: all of its bubbles share a single
             hover-revealed actions row rendered below them. -->
        <div
          class="group flex flex-col items-start"
          data-testid="chat-msg-assistant-turn"
        >
          <div class="flex w-full flex-col gap-3">
            {#if has_reasoning_bubble}
              <!-- Assistant reasoning bubble. Collapsed by default; toggling
                   reveals the model's thinking. -->
              <div
                class="flex flex-col items-start"
                data-testid="chat-msg-assistant"
              >
                <!-- Whole-bubble click expands when collapsed. The inner toggle
                     button uses |stopPropagation so it still owns collapsing
                     (and the container handler never accidentally re-expands). -->
                <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
                <div
                  class="rounded-xl bg-base-200 px-4 py-3 w-[70%] text-sm flex flex-col gap-2"
                  class:cursor-pointer={!thinkingExpanded[index]}
                  on:click={() => {
                    if (!thinkingExpanded[index]) thinkingExpanded[index] = true
                  }}
                >
                  <div data-testid="chat-msg-thinking">
                    <button
                      type="button"
                      class="flex w-full items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 cursor-pointer"
                      on:click|stopPropagation={() =>
                        (thinkingExpanded[index] = !thinkingExpanded[index])}
                      aria-expanded={!!thinkingExpanded[index]}
                    >
                      <span class="text-gray-400" aria-hidden="true">
                        {thinkingExpanded[index] ? "▼" : "▶"}
                      </span>
                      <span class="font-medium">Thinking</span>
                    </button>
                    {#if thinkingExpanded[index]}
                      <div class="mt-2">
                        {#if highlight && highlight.kind === "reasoning" && highlight.trace_index === index && reasoning}
                          {@const seg = highlight_segments(
                            reasoning,
                            highlight,
                          )}
                          <!-- Marked node renders as plain text: the offsets
                               index the raw reasoning, not rendered markdown. -->
                          <div class="whitespace-pre-wrap">
                            {seg.before}<mark
                              data-highlight-target
                              class="bg-warning/40 rounded px-0.5"
                              >{seg.mark}</mark
                            >{seg.after}
                          </div>
                        {:else if reasoning && is_non_string_json(reasoning)}
                          <!-- The legacy trace view renders reasoning through
                               Output, so structured reasoning read as a
                               collapsed paragraph only in the chat view. -->
                          <Output raw_output={reasoning} no_padding={true} />
                        {:else}
                          <ChatMarkdown text={reasoning} />
                        {/if}
                      </div>
                    {/if}
                  </div>
                </div>
              </div>
            {/if}

            {#if has_content_bubble}
              <!-- Assistant content bubble. -->
              <div
                class="flex flex-col items-start"
                data-testid="chat-msg-assistant"
              >
                <div
                  class="rounded-xl bg-base-200 px-4 py-3 w-[70%] text-sm flex flex-col gap-2"
                >
                  <div data-testid="chat-msg-content">
                    {#if highlight && highlight.kind === "content" && highlight.trace_index === index && content}
                      {@const seg = highlight_segments(content, highlight)}
                      <!-- Marked node renders as plain text: the offsets index
                           the raw content, not rendered markdown. -->
                      <div class="whitespace-pre-wrap">
                        {seg.before}<mark
                          data-highlight-target
                          class="bg-warning/40 rounded px-0.5">{seg.mark}</mark
                        >{seg.after}
                      </div>
                    {:else if content && is_non_string_json(content)}
                      <Output raw_output={content} no_padding={true} />
                    {:else}
                      <ChatMarkdown text={content} />
                    {/if}
                  </div>
                </div>
              </div>
            {/if}

            {#if has_tc_bubble && tool_calls}
              {#each tool_calls as tool_call, tcIdx}
                <!-- One bubble per tool call. -->
                {@const tc_key = `${index}-${tcIdx}`}
                {@const result =
                  tool_results_by_call_id.get(tool_call.id) ?? null}
                {@const result_content = result
                  ? content_from_message(result.message)
                  : undefined}
                {@const kiln_data = result
                  ? kiln_task_tool_data_from_message(result.message)
                  : null}
                {@const tool_error = result
                  ? is_tool_error(result.message)
                  : false}
                <!-- A tool-CALL citation lands on the whole bubble (the call
                     renders via a dedicated component, not plain text): mark
                     the bubble as the scroll target and let the reactive
                     expansion open it. A tool-RESULT citation marks the exact
                     span inside the result text below; the bubble is only its
                     fallback target when that text cannot carry the mark. -->
                {@const is_result_cited = !!(
                  highlight &&
                  highlight.kind === "tool_result" &&
                  tool_result_owner &&
                  tool_result_owner.index === index &&
                  tool_result_owner.tcIdx === tcIdx
                )}
                <!-- The rendered result is the LAST tool message for this
                     call id, while the citation names an exact trace index —
                     with duplicate ids (a retried call) they can disagree,
                     and slicing the other message would mark the wrong
                     bytes. The bubble fallback is the honest target then. -->
                {@const result_mark_text =
                  is_result_cited &&
                  result &&
                  highlight &&
                  result.trace_index === highlight.trace_index
                    ? tool_result_citation_text(result.message)
                    : null}
                {@const is_tc_target =
                  !!(
                    highlight &&
                    highlight.kind === "tool_calls" &&
                    highlight.trace_index === index &&
                    tcIdx === 0
                  ) ||
                  (is_result_cited && result_mark_text === null)}
                {#if is_internal_answer(tool_call)}
                  <!-- The task's structured answer, not a tool the agent
                       chose to call. Rendered through the shared Output
                       component so a reviewer is never shown a tool that
                       does not exist in their task. -->
                  <div
                    class="flex flex-col items-start"
                    data-testid="chat-msg-assistant"
                  >
                    <div
                      class="rounded-xl bg-base-200 px-4 py-3 w-[70%] text-sm flex flex-col gap-2"
                      data-highlight-target={is_tc_target ? "" : undefined}
                      data-testid="chat-msg-structured-output"
                    >
                      <div class="text-xs text-gray-500 font-medium">
                        Output
                      </div>
                      <Output raw_output={tool_call.function.arguments} />
                    </div>
                  </div>
                {:else}
                  <div
                    class="flex flex-col items-start"
                    data-testid="chat-msg-assistant"
                  >
                    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
                    <div
                      class="rounded-xl bg-base-200 px-4 py-3 w-[70%] text-sm flex flex-col gap-2"
                      data-highlight-target={is_tc_target ? "" : undefined}
                      class:cursor-pointer={!toolCallExpanded[tc_key]}
                      on:click={() => {
                        if (!toolCallExpanded[tc_key])
                          toolCallExpanded[tc_key] = true
                      }}
                    >
                      <div data-testid="chat-msg-toolcall">
                        <button
                          type="button"
                          class="flex w-full items-center gap-1.5 text-xs text-gray-600 hover:text-gray-900 cursor-pointer"
                          on:click|stopPropagation={() =>
                            (toolCallExpanded[tc_key] =
                              !toolCallExpanded[tc_key])}
                          aria-expanded={!!toolCallExpanded[tc_key]}
                        >
                          <span class="text-gray-400" aria-hidden="true">
                            {toolCallExpanded[tc_key] ? "▼" : "▶"}
                          </span>
                          <span class="font-medium">
                            Toolcall: <span class="font-mono"
                              >{tool_call.function.name}</span
                            >
                          </span>
                        </button>
                        {#if toolCallExpanded[tc_key]}
                          {@const arg_mark = tool_call_argument_mark(
                            highlight,
                            index,
                            tcIdx,
                          )}
                          <div
                            class="mt-3 flex flex-col gap-3"
                            data-testid="chat-tool-call"
                          >
                            <div>
                              <div class="text-xs text-gray-500 font-bold mb-1">
                                Invoked Tool Call
                              </div>
                              <ToolCall
                                {tool_call}
                                {project_id}
                                persistent_tool_id={kiln_data?.tool_id}
                                arguments_mark={arg_mark}
                              />
                            </div>
                            {#if result_content !== undefined}
                              <div>
                                <div
                                  class="text-xs font-bold mb-1 {tool_error
                                    ? 'text-error'
                                    : 'text-gray-500'}"
                                >
                                  {tool_error ? "Tool Error" : "Tool Result"}
                                </div>
                                <div
                                  class={tool_error
                                    ? "border border-error/20 rounded-lg p-2"
                                    : ""}
                                >
                                  <!-- The result keeps the rendering it has
                                       when nothing is cited; the mark rides on
                                       top. Passed only when the displayed text
                                       IS the string the offsets index, so a
                                       result whose display diverges gets the
                                       bubble as its target and no mark. -->
                                  <Output
                                    raw_output={result_content}
                                    no_padding={true}
                                    mark={highlight &&
                                    is_result_cited &&
                                    result_mark_text === result_content
                                      ? {
                                          start: highlight.start,
                                          end: highlight.end,
                                        }
                                      : null}
                                  />
                                </div>
                              </div>
                            {:else if result === null}
                              <div class="text-xs text-gray-400 italic">
                                No tool result recorded.
                              </div>
                            {/if}
                            {#if kiln_data}
                              <div>
                                <button
                                  class="link text-xs text-gray-500"
                                  on:click={() => {
                                    tool_messages_dialog?.show(kiln_data)
                                  }}
                                >
                                  <div class="flex flex-row items-center gap-1">
                                    <span>Subtask Message Trace</span>
                                    <div class="w-4 h-4">
                                      <ArrowRightUpIcon />
                                    </div>
                                  </div>
                                </button>
                              </div>
                            {/if}
                          </div>
                        {/if}
                      </div>
                    </div>
                  </div>
                {/if}
              {/each}
            {/if}

            {#if empty_assistant}
              <div
                class="flex flex-col items-start"
                data-testid="chat-msg-assistant"
              >
                <div
                  class="rounded-xl bg-base-200 px-4 py-3 w-[70%] text-sm text-gray-400 italic"
                >
                  (empty message)
                </div>
              </div>
            {/if}
          </div>
          {#if show_info || show_fork}
            <!-- Constrain to the bubble width so the right-aligned actions sit
                 at the bubble's right edge, not the far column edge. -->
            <div class="w-[70%]">
              <ChatMessageActions
                align="end"
                show_usage={show_info}
                show_fork={!!show_fork}
                on_usage={() => open_usage_dialog(message)}
                on_fork={() => {
                  if (fork_run_id) on_fork?.(fork_run_id, index)
                }}
              />
            </div>
          {/if}
        </div>
      {/if}
    {/if}
  {/each}
</div>

<ToolMessagesDialog bind:this={tool_messages_dialog} {project_id} />
<UsageInfoDialog bind:this={usage_info_dialog} />
