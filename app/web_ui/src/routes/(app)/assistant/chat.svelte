<script lang="ts">
  import { onMount, onDestroy, tick } from "svelte"
  import { get } from "svelte/store"
  import { fly } from "svelte/transition"
  import posthog from "posthog-js"
  import ChatCostDisclaimer from "./chat_cost_disclaimer.svelte"
  import type { ChatMessage, ChatMessagePart } from "$lib/chat/streaming_chat"
  import ChatMarkdown from "$lib/ui/chat/chat_markdown.svelte"
  import ArrowUpIcon from "$lib/ui/icons/arrow_up_icon.svelte"
  import StopIcon from "$lib/ui/icons/stop_icon.svelte"
  import CloseIcon from "$lib/ui/icons/close_icon.svelte"
  import Warning from "$lib/ui/warning.svelte"
  import {
    chatSessionStore,
    type ChatSessionStore,
  } from "$lib/chat/chat_session_store"
  import ChatWelcome from "./chat_welcome.svelte"
  import ChatHistory from "./chat_history.svelte"
  import ToolApprovalBox from "./tool_approval_box.svelte"
  import ChatLoading from "./chat_loading.svelte"
  import ChatStatusSteps from "./chat_status_steps.svelte"
  import ToolStatusLine from "./tool_status_line.svelte"
  import { env } from "$env/dynamic/public"

  export let store: ChatSessionStore = chatSessionStore

  let costDisclaimer: ChatCostDisclaimer
  $: store.onConsentNeeded = () => costDisclaimer.prompt()
  let chatHistory: { open: () => void }
  let input = ""
  let messagesContainer: HTMLDivElement | null = null
  let messagesEndRef: HTMLDivElement | null = null
  let scrollObserver: MutationObserver | null = null
  let textareaRef: HTMLTextAreaElement | null = null

  const showToolCallDetails = env.PUBLIC_SHOW_TOOL_CALL_DETAILS === "true"

  $: toolApprovalWaiter = $store.toolApprovalWaiter
  $: toolApprovalPicks = $store.toolApprovalPicks
  $: showActivityIndicator = $store.showActivityIndicator
  $: upgradeNudgeVersion = $store.upgradeNudgeVersion
  $: versionRequired = $store.versionRequired

  export let hasMessages = false
  $: messages = $store.messages
  $: hasMessages = messages.length > 0
  $: status = $store.status
  $: collapsedPartKeys = $store.collapsedPartKeys

  let expandedStepGroups: Record<string, boolean> = {}
  const MAX_VISIBLE_STEPS = 5

  function toggleStepGroupExpanded(key: string): void {
    suppressAutoScroll = true
    expandedStepGroups = {
      ...expandedStepGroups,
      [key]: !expandedStepGroups[key],
    }
    setTimeout(() => {
      suppressAutoScroll = false
    }, 50)
  }

  $: isLoading = status === "submitted" || status === "streaming"
  // Block input entirely when the client is too old: sending would just 426
  // again and the message would go nowhere.
  $: inputDisabled = isLoading || versionRequired

  let prevIsLoading = false
  $: {
    if (prevIsLoading && !isLoading) {
      tick().then(() => {
        textareaRef?.focus({ preventScroll: true })
      })
    }
    prevIsLoading = isLoading
  }

  $: lastMessage = messages[messages.length - 1]
  $: lastParts = lastMessage?.parts ?? []

  $: showStreamingCursor =
    isLoading && lastMessage?.role === "assistant" && lastParts.length === 0

  function isMessageVisible(message: ChatMessage): boolean {
    if (message.role !== "assistant") return true
    if (isLoading && message.id === lastMessage?.id) return true
    const parts = message.parts ?? []
    if (parts.length === 0 && !message.content) return false
    return true
  }

  function isReasoningStreaming(
    message: ChatMessage,
    partIndex: number,
    parts: ChatMessagePart[],
  ): boolean {
    const isLastMessage =
      messages.length > 0 && message.id === messages[messages.length - 1]?.id
    const isLastPart = partIndex === parts.length - 1
    return isLastMessage && status === "streaming" && isLastPart
  }

  function partKey(
    message: ChatMessage,
    part: ChatMessagePart,
    partIndex: number,
  ): string {
    if (part.type === "reasoning") return `${message.id}-reasoning-${partIndex}`
    if (
      typeof part.type === "string" &&
      part.type.startsWith("tool-") &&
      "toolCallId" in part
    ) {
      return `${message.id}-tool-${(part as { toolCallId: string }).toolCallId}`
    }
    return `${message.id}-part-${partIndex}`
  }

  function shouldAutoCollapse(
    message: ChatMessage,
    partIndex: number,
    parts: ChatMessagePart[],
  ): boolean {
    const isLastMessage =
      messages.length > 0 && message.id === messages[messages.length - 1]?.id
    const isLastPart = partIndex === parts.length - 1
    const isCurrentStreaming =
      isLastMessage && status === "streaming" && isLastPart
    return !isCurrentStreaming
  }

  function isPartCollapsed(
    state: Record<string, boolean>,
    message: ChatMessage,
    part: ChatMessagePart,
    partIndex: number,
    parts: ChatMessagePart[],
  ): boolean {
    if (
      typeof part.type === "string" &&
      part.type.startsWith("tool-") &&
      "toolCallId" in part &&
      toolApprovalWaiter &&
      toolApprovalWaiter.payload.items.some(
        (i) => i.toolCallId === (part as { toolCallId: string }).toolCallId,
      )
    ) {
      return false
    }
    const key = partKey(message, part, partIndex)
    if (key in state) return state[key]
    return shouldAutoCollapse(message, partIndex, parts)
  }

  function togglePartCollapsed(
    message: ChatMessage,
    part: ChatMessagePart,
    partIndex: number,
  ): void {
    const key = partKey(message, part, partIndex)
    const parts = message.parts ?? []
    const current = isPartCollapsed(
      collapsedPartKeys,
      message,
      part,
      partIndex,
      parts,
    )
    suppressAutoScroll = true
    store.togglePartCollapsed(key, current)
    setTimeout(() => {
      suppressAutoScroll = false
    }, 50)
  }

  type RenderSegment =
    | { kind: "text"; part: ChatMessagePart; partIndex: number }
    | {
        kind: "step-group"
        items: Array<{ part: ChatMessagePart; partIndex: number }>
      }

  function groupPartsForSimplifiedView(
    parts: ChatMessagePart[],
  ): RenderSegment[] {
    const segments: RenderSegment[] = []
    let currentGroup: Array<{ part: ChatMessagePart; partIndex: number }> = []

    for (let i = 0; i < parts.length; i++) {
      if (parts[i].type === "text") {
        if (currentGroup.length > 0) {
          segments.push({ kind: "step-group", items: currentGroup })
          currentGroup = []
        }
        segments.push({ kind: "text", part: parts[i], partIndex: i })
      } else {
        currentGroup.push({ part: parts[i], partIndex: i })
      }
    }
    if (currentGroup.length > 0) {
      segments.push({ kind: "step-group", items: currentGroup })
    }
    return segments
  }

  function isStepGroupLoading(
    message: ChatMessage,
    groupSegment: RenderSegment & { kind: "step-group" },
    allSegments: RenderSegment[],
  ): boolean {
    if (!(isLoading && message.id === lastMessage?.id)) return false

    const groupIdx = allSegments.indexOf(groupSegment)
    const hasTextAfter = allSegments
      .slice(groupIdx + 1)
      .some((s) => s.kind === "text")
    if (hasTextAfter) return false

    const toolItems = groupSegment.items.filter(
      (i) => typeof i.part.type === "string" && i.part.type.startsWith("tool-"),
    )
    const allToolsComplete =
      toolItems.length > 0 &&
      toolItems.every(
        (i) =>
          "output" in i.part &&
          (i.part as { output?: unknown }).output !== undefined,
      )
    if (allToolsComplete && !showActivityIndicator) return false

    return true
  }

  function formatToolName(type: string): string {
    const name = type.startsWith("tool-") ? type.slice(5) : type
    return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
  }

  function getToolCallId(part: ChatMessagePart): string {
    if ("toolCallId" in part && typeof part.toolCallId === "string") {
      return part.toolCallId
    }
    return ""
  }

  function hasToolInput(part: ChatMessagePart): boolean {
    if (!("input" in part) || part.input === undefined) return false
    if (typeof part.input !== "object" || part.input === null) return true
    return Object.keys(part.input).length > 0
  }

  function formatToolInput(input: unknown): string {
    return typeof input === "string" ? input : JSON.stringify(input, null, 2)
  }

  function formatToolOutput(output: unknown): string {
    if (typeof output === "string") {
      try {
        const parsed = JSON.parse(output)
        return JSON.stringify(parsed, null, 2)
      } catch {
        return output
      }
    }
    return JSON.stringify(output, null, 2)
  }

  function getToolInputString(input: unknown, key: string): string {
    if (typeof input === "object" && input !== null && key in input) {
      const val = (input as Record<string, unknown>)[key]
      return typeof val === "string" ? val : ""
    }
    return ""
  }

  type ToolPart = {
    type: `tool-${string}`
    toolCallId: string
    toolName?: string
    input?: unknown
    output?: unknown
  }

  function asToolPart(part: ChatMessagePart): ToolPart {
    return part as ToolPart
  }

  function getPartText(part: ChatMessagePart): string {
    return "text" in part && typeof part.text === "string" ? part.text : ""
  }

  function getPartReasoning(part: ChatMessagePart): string {
    return "reasoning" in part && typeof part.reasoning === "string"
      ? part.reasoning
      : ""
  }

  function getToolOutputError(part: ChatMessagePart): string {
    if (
      !("output" in part) ||
      typeof part.output !== "object" ||
      part.output === null
    )
      return "Error"
    return "error" in part.output &&
      typeof (part.output as { error?: string }).error === "string"
      ? (part.output as { error: string }).error
      : "Error"
  }

  let suppressAutoScroll = false
  let userNearBottom = true
  let isAutoScrolling = false
  const SCROLL_THRESHOLD = 0.5

  function handleScroll() {
    if (isAutoScrolling || !messagesContainer) return
    const { scrollTop, scrollHeight, clientHeight } = messagesContainer
    if (scrollTop + clientHeight >= scrollHeight - SCROLL_THRESHOLD) {
      userNearBottom = true
    }
  }

  function handleWheel(e: WheelEvent) {
    if (e.deltaY < 0) {
      userNearBottom = false
    }
  }

  function handleTouchStart(e: TouchEvent) {
    lastTouchY = e.touches[0]?.clientY ?? 0
  }

  function handleTouchMove(e: TouchEvent) {
    const currentY = e.touches[0]?.clientY ?? 0
    if (currentY > lastTouchY) {
      userNearBottom = false
    }
    lastTouchY = currentY
  }

  let lastTouchY = 0

  onMount(() => {
    // Surface the upgrade banners up front, before any message is sent.
    void store.checkVersionPolicy()

    const container = messagesContainer
    const end = messagesEndRef
    if (container && end) {
      container.addEventListener("scroll", handleScroll, { passive: true })
      container.addEventListener("wheel", handleWheel, { passive: true })
      container.addEventListener("touchstart", handleTouchStart, {
        passive: true,
      })
      container.addEventListener("touchmove", handleTouchMove, {
        passive: true,
      })
      if (messages.length > 0) {
        end.scrollIntoView({ block: "end", behavior: "auto" })
      }
      let rafPending = false
      scrollObserver = new MutationObserver(() => {
        if (!suppressAutoScroll && userNearBottom && !rafPending) {
          rafPending = true
          requestAnimationFrame(() => {
            rafPending = false
            isAutoScrolling = true
            end.scrollIntoView({ block: "end", behavior: "auto" })
            requestAnimationFrame(() => {
              isAutoScrolling = false
            })
          })
        }
      })
      scrollObserver.observe(container, {
        childList: true,
        subtree: true,
        attributes: true,
        characterData: true,
      })
    }
    tick().then(() => {
      textareaRef?.focus({ preventScroll: true })
    })
  })

  onDestroy(() => {
    messagesContainer?.removeEventListener("scroll", handleScroll)
    messagesContainer?.removeEventListener("wheel", handleWheel)
    messagesContainer?.removeEventListener("touchstart", handleTouchStart)
    messagesContainer?.removeEventListener("touchmove", handleTouchMove)
    scrollObserver?.disconnect()
    scrollObserver = null
  })

  function handleTextareaKeydown(e: KeyboardEvent): void {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      if (!isLoading && input.trim()) handleSubmit()
    }
  }

  function adjustTextareaHeight(e?: Event): void {
    const el = (e?.currentTarget as HTMLTextAreaElement) ?? textareaRef
    if (!el) return
    el.style.height = "auto"
    el.style.height = `${Math.min(el.scrollHeight + 2, window.innerHeight * 0.4)}px`
  }

  function applyToolApprovalRun(toolCallId: string): void {
    store.applyToolApprovalRun(toolCallId)
  }

  function applyToolApprovalSkip(toolCallId: string): void {
    store.applyToolApprovalSkip(toolCallId)
  }

  function retryLastRequest() {
    store.retryLastRequest()
  }

  function dismissUpgradeNudge() {
    store.dismissUpgradeNudge()
  }

  function stop() {
    store.stop()
  }

  function onChatHistoryApply(
    e: CustomEvent<{
      messages: ChatMessage[]
      continuationTraceId: string
    }>,
  ) {
    store.loadSession(e.detail.messages, e.detail.continuationTraceId)
    userNearBottom = true
    tick().then(() => {
      messagesEndRef?.scrollIntoView({ block: "end", behavior: "auto" })
      textareaRef?.focus({ preventScroll: true })
    })
  }

  export function newChat() {
    posthog.capture("chat_new_chat_clicked", {
      had_messages: get(store).messages.length > 0,
    })
    store.reset()
  }

  export function openHistory() {
    chatHistory.open()
  }

  async function handleSubmit(e?: Event) {
    if (e) e.preventDefault()
    const text = input.trim()
    if (!text || isLoading) return
    const sent = await store.sendMessage(text)
    if (!sent) return
    input = ""
    userNearBottom = true
    setTimeout(() => {
      adjustTextareaHeight()
      messagesEndRef?.scrollIntoView({ block: "end", behavior: "auto" })
    }, 0)
  }
</script>

<div class="flex flex-col flex-1 min-h-0">
  <div class="flex flex-col flex-1 min-h-0 overflow-hidden w-full">
    <ChatHistory bind:this={chatHistory} on:apply={onChatHistoryApply} />
    <div
      bind:this={messagesContainer}
      class="chat-messages-scroll flex-1 min-h-0 overflow-y-auto overflow-x-hidden"
      role="log"
      aria-live="polite"
    >
      <div
        class="flex flex-col gap-4 w-full min-h-full md:max-w-3xl mx-auto px-1"
      >
        {#if messages.length === 0 && !isLoading}
          <div class="flex-1 shrink-0"></div>
          <ChatWelcome
            on:select={async (e) => await store.sendMessage(e.detail)}
          />
          <div class="flex-[2] shrink-0"></div>
        {/if}
        {#each messages as message (message.id)}
          {#if isMessageVisible(message)}
            <div
              in:fly={{ y: 8, duration: 200 }}
              out:fly={{ y: -4, duration: 150 }}
              class={message.role === "user"
                ? "leading-tight rounded-xl bg-base-content/[0.06] px-3 py-2.5 max-w-2xl ml-auto text-sm"
                : message.role === "error"
                  ? "rounded-lg bg-error/10 border border-error/30 px-3 py-2.5 text-error text-sm"
                  : "flex flex-col gap-3"}
            >
              {#if message.role === "error"}
                <div class="flex items-center justify-between gap-3">
                  <span>{message.content}</span>
                  <button
                    type="button"
                    class="shrink-0 rounded-md bg-error/20 px-2 py-1 text-xs font-medium hover:bg-error/30 transition-colors"
                    on:click={retryLastRequest}
                    disabled={isLoading}
                  >
                    Retry
                  </button>
                </div>
              {:else}
                <div class="flex flex-col leading-tight">
                  {#if message.parts && message.parts.length > 0}
                    {#if !showToolCallDetails}
                      {@const segments = groupPartsForSimplifiedView(
                        message.parts ?? [],
                      )}
                      {#each segments as segment, segIdx}
                        {#if segment.kind === "text"}
                          {@const isFirstText =
                            segments.findIndex((s) => s.kind === "text") ===
                            segIdx}
                          {#if isFirstText}
                            {@const hasStepGroup = segments.some(
                              (s) => s.kind === "step-group",
                            )}
                            {#if !hasStepGroup}
                              <div
                                class="flex items-center gap-1.5 text-sm text-base-content/50 py-0.5"
                              >
                                <span class="inline-block w-3 text-center"
                                  >✓</span
                                >
                                <span>Thought</span>
                              </div>
                            {/if}
                          {/if}
                          <ChatMarkdown text={getPartText(segment.part)} />
                        {:else}
                          {@const groupLoading = isStepGroupLoading(
                            message,
                            segment,
                            segments,
                          )}
                          {@const hasReasoningInGroup = segment.items.some(
                            (i) => i.part.type === "reasoning",
                          )}
                          {@const hasToolsInGroup = segment.items.some(
                            (i) =>
                              typeof i.part.type === "string" &&
                              i.part.type.startsWith("tool-"),
                          )}
                          {@const stepGroupKey = `${message.id}-sg-${segIdx}`}
                          {@const isStepGroupExpanded =
                            expandedStepGroups[stepGroupKey] === true}
                          {@const totalSteps = segment.items.length}
                          {@const shouldCompress =
                            totalSteps > MAX_VISIBLE_STEPS &&
                            !isStepGroupExpanded}
                          {@const hiddenCount = totalSteps - MAX_VISIBLE_STEPS}
                          {@const visibleItems = shouldCompress
                            ? segment.items.slice(-MAX_VISIBLE_STEPS)
                            : segment.items}
                          <div class="flex items-start gap-3 min-w-0">
                            {#if groupLoading}
                              <img
                                src="/images/chat_icon_animated.svg"
                                alt=""
                                class="w-9 h-9 shrink-0 -mt-1.5"
                              />
                            {/if}
                            <div class="flex flex-col min-w-0 flex-1">
                              {#if totalSteps > MAX_VISIBLE_STEPS}
                                <button
                                  type="button"
                                  class="flex items-center gap-1.5 text-sm text-base-content/40 hover:text-base-content/60 transition-colors cursor-pointer py-0.5"
                                  on:click={() =>
                                    toggleStepGroupExpanded(stepGroupKey)}
                                >
                                  {#if isStepGroupExpanded}
                                    <span>{totalSteps} steps ▼</span>
                                  {:else}
                                    <span>… {hiddenCount} more steps ▶</span>
                                  {/if}
                                </button>
                              {/if}
                              {#if !shouldCompress && !hasReasoningInGroup && hasToolsInGroup}
                                <div
                                  class="flex items-center gap-1.5 text-sm text-base-content/50 py-0.5"
                                >
                                  <span class="inline-block w-3 text-center"
                                    >✓</span
                                  >
                                  <span>Thought</span>
                                </div>
                              {/if}
                              {#each visibleItems as item (partKey(message, item.part, item.partIndex))}
                                {#if item.part.type === "reasoning"}
                                  {@const collapsed = isPartCollapsed(
                                    collapsedPartKeys,
                                    message,
                                    item.part,
                                    item.partIndex,
                                    message.parts ?? [],
                                  )}
                                  {@const streaming = isReasoningStreaming(
                                    message,
                                    item.partIndex,
                                    message.parts ?? [],
                                  )}
                                  <div
                                    class="overflow-hidden text-sm text-base-content/60"
                                  >
                                    <button
                                      type="button"
                                      class="group/btn w-full flex items-center gap-1.5 py-1 text-left text-base-content/60 hover:text-base-content/80 transition-colors cursor-pointer"
                                      on:click={() =>
                                        togglePartCollapsed(
                                          message,
                                          item.part,
                                          item.partIndex,
                                        )}
                                    >
                                      <span
                                        class="flex items-center gap-1.5 min-w-0"
                                      >
                                        {#if streaming}
                                          <span
                                            class="inline-flex items-baseline gap-px"
                                          >
                                            Thinking
                                            <span
                                              class="thinking-dot"
                                              style="animation-delay: 0ms"
                                              >.</span
                                            ><span
                                              class="thinking-dot"
                                              style="animation-delay: 160ms"
                                              >.</span
                                            ><span
                                              class="thinking-dot"
                                              style="animation-delay: 320ms"
                                              >.</span
                                            >
                                          </span>
                                        {:else}
                                          <span class="font-semibold"
                                            >Thought</span
                                          >
                                        {/if}
                                        {#if collapsed}
                                          <span
                                            class="shrink-0 text-base-content/40 transition-opacity opacity-0 group-hover/btn:opacity-100"
                                            aria-hidden="true">▶</span
                                          >
                                        {:else}
                                          <span
                                            class="shrink-0 text-base-content/40"
                                            aria-hidden="true">▼</span
                                          >
                                        {/if}
                                      </span>
                                    </button>
                                    {#if !collapsed}
                                      <div class="pt-1">
                                        <ChatMarkdown
                                          text={getPartReasoning(item.part)}
                                        />
                                      </div>
                                    {/if}
                                  </div>
                                {:else if typeof item.part.type === "string" && item.part.type.startsWith("tool-")}
                                  {@const toolPart = asToolPart(item.part)}
                                  {@const tcId = getToolCallId(toolPart)}
                                  {@const approvalItem =
                                    toolApprovalWaiter?.payload.items.find(
                                      (i) => i.toolCallId === tcId,
                                    )}
                                  {@const pendingInlineApproval =
                                    toolApprovalWaiter !== null &&
                                    approvalItem !== undefined &&
                                    toolPart.output === undefined}
                                  {@const hasOutput =
                                    toolPart.output !== undefined}
                                  {@const method = getToolInputString(
                                    toolPart.input,
                                    "method",
                                  )}
                                  {@const urlPath = getToolInputString(
                                    toolPart.input,
                                    "url_path",
                                  )}
                                  {@const isGet = !method || method === "GET"}
                                  {@const detail =
                                    method && urlPath
                                      ? `(${method} ${urlPath})`
                                      : ""}
                                  {@const isActiveMessage =
                                    isLoading && message.id === lastMessage?.id}
                                  {@const effectivelyComplete =
                                    hasOutput || !isActiveMessage}
                                  {#if pendingInlineApproval && toolApprovalPicks[tcId] === undefined}
                                    <div class="mt-2 text-sm">
                                      <ToolApprovalBox
                                        description={approvalItem?.approvalDescription ??
                                          ""}
                                        method={getToolInputString(
                                          approvalItem?.input,
                                          "method",
                                        )}
                                        url={getToolInputString(
                                          approvalItem?.input,
                                          "url_path",
                                        )}
                                        onRun={() => applyToolApprovalRun(tcId)}
                                        onSkip={() =>
                                          applyToolApprovalSkip(tcId)}
                                      />
                                    </div>
                                  {:else}
                                    <ToolStatusLine
                                      variant={effectivelyComplete
                                        ? isGet
                                          ? "fetched"
                                          : "saved"
                                        : isGet
                                          ? "fetching"
                                          : "saving"}
                                      {detail}
                                    />
                                  {/if}
                                {/if}
                              {/each}
                              {#if segIdx === segments.length - 1 && message.role === "assistant" && message.id === lastMessage?.id}
                                {@const hasVisibleApproval =
                                  toolApprovalWaiter !== null &&
                                  toolApprovalWaiter.payload.items.some(
                                    (i) =>
                                      toolApprovalPicks[i.toolCallId] ===
                                      undefined,
                                  )}
                                {#if !hasVisibleApproval}
                                  <ChatStatusSteps
                                    parts={message.parts ?? []}
                                    isLoading={isLoading &&
                                      message.id === lastMessage?.id}
                                    isLastMessage={message.id ===
                                      lastMessage?.id}
                                    {showActivityIndicator}
                                  />
                                {/if}
                              {/if}
                            </div>
                          </div>
                        {/if}
                      {/each}
                      {#if message.role === "assistant"}
                        {@const segments = groupPartsForSimplifiedView(
                          message.parts ?? [],
                        )}
                        {@const lastSegIsText =
                          segments.length > 0 &&
                          segments[segments.length - 1].kind === "text"}
                        {#if lastSegIsText && message.id === lastMessage?.id}
                          {@const hasVisibleApproval =
                            toolApprovalWaiter !== null &&
                            toolApprovalWaiter.payload.items.some(
                              (i) =>
                                toolApprovalPicks[i.toolCallId] === undefined,
                            )}
                          {@const isActiveMessage =
                            isLoading && message.id === lastMessage?.id}
                          {#if !hasVisibleApproval}
                            {#if isActiveMessage && showActivityIndicator}
                              <div class="flex items-start gap-3">
                                <img
                                  src="/images/chat_icon_animated.svg"
                                  alt=""
                                  class="w-9 h-9 shrink-0 -mt-1.5"
                                />
                                <div class="flex flex-col">
                                  <ChatStatusSteps
                                    parts={message.parts ?? []}
                                    isLoading={true}
                                    isLastMessage={true}
                                    {showActivityIndicator}
                                  />
                                </div>
                              </div>
                            {:else}
                              <ChatStatusSteps
                                parts={message.parts ?? []}
                                isLoading={isActiveMessage}
                                isLastMessage={message.id === lastMessage?.id}
                                {showActivityIndicator}
                              />
                            {/if}
                          {/if}
                        {/if}
                      {/if}
                    {:else}
                      {#each message.parts as part, partIndex (partKey(message, part, partIndex))}
                        {#if part.type === "text"}
                          <ChatMarkdown text={part.text ?? ""} />
                        {:else if part.type === "reasoning"}
                          {@const collapsed = isPartCollapsed(
                            collapsedPartKeys,
                            message,
                            part,
                            partIndex,
                            message.parts ?? [],
                          )}
                          {@const streaming = isReasoningStreaming(
                            message,
                            partIndex,
                            message.parts ?? [],
                          )}
                          <div
                            class="mt-2 overflow-hidden text-sm text-base-content/60"
                          >
                            <button
                              type="button"
                              class="group/btn w-full flex items-center gap-1.5 py-1 text-left text-base-content/60 hover:text-base-content/80 transition-colors cursor-pointer"
                              on:click={() =>
                                togglePartCollapsed(message, part, partIndex)}
                            >
                              <span class="flex items-center gap-1.5 min-w-0">
                                {#if streaming}
                                  <span
                                    class="inline-flex items-baseline gap-px"
                                  >
                                    Thinking
                                    <span
                                      class="thinking-dot"
                                      style="animation-delay: 0ms">.</span
                                    ><span
                                      class="thinking-dot"
                                      style="animation-delay: 160ms">.</span
                                    ><span
                                      class="thinking-dot"
                                      style="animation-delay: 320ms">.</span
                                    >
                                  </span>
                                {:else}
                                  <span class="font-semibold">Thought</span>
                                {/if}
                                {#if collapsed}
                                  <span
                                    class="shrink-0 text-base-content/40 transition-opacity opacity-0 group-hover/btn:opacity-100"
                                    aria-hidden="true">▶</span
                                  >
                                {:else}
                                  <span
                                    class="shrink-0 text-base-content/40"
                                    aria-hidden="true">▼</span
                                  >
                                {/if}
                              </span>
                            </button>
                            {#if !collapsed}
                              <div class="pt-1">
                                <ChatMarkdown text={part.reasoning ?? ""} />
                              </div>
                            {/if}
                          </div>
                        {:else if typeof part.type === "string" && part.type.startsWith("tool-")}
                          {@const tcId = getToolCallId(part)}
                          {@const approvalItem =
                            toolApprovalWaiter?.payload.items.find(
                              (i) => i.toolCallId === tcId,
                            )}
                          {@const pendingInlineApproval =
                            toolApprovalWaiter !== null &&
                            approvalItem !== undefined &&
                            part.output === undefined}
                          {@const toolCollapsed = isPartCollapsed(
                            collapsedPartKeys,
                            message,
                            part,
                            partIndex,
                            message.parts ?? [],
                          )}
                          {@const hasOutput = part.output !== undefined}
                          {@const hasError =
                            hasOutput &&
                            typeof part.output === "object" &&
                            part.output !== null &&
                            "error" in part.output}
                          <div class="mt-2 overflow-hidden text-sm">
                            <button
                              type="button"
                              class="group/btn w-full flex items-center gap-1.5 py-1 text-left text-base-content/60 hover:text-base-content/80 transition-colors cursor-pointer"
                              on:click={() =>
                                togglePartCollapsed(message, part, partIndex)}
                            >
                              <span class="flex items-center gap-1.5">
                                {formatToolName(part.type)} was called
                                {#if toolCollapsed}
                                  <span
                                    class="shrink-0 text-base-content/40 transition-opacity opacity-0 group-hover/btn:opacity-100"
                                    aria-hidden="true">▶</span
                                  >
                                {:else}
                                  <span
                                    class="shrink-0 text-base-content/40"
                                    aria-hidden="true">▼</span
                                  >
                                {/if}
                              </span>
                            </button>
                            {#if !toolCollapsed}
                              <div
                                class="mt-2 overflow-hidden rounded-md {hasError
                                  ? 'bg-error/5 text-error'
                                  : 'bg-base-content/[0.04]'}"
                              >
                                <div class="px-3 py-2.5 flex flex-col gap-2.5">
                                  <div>
                                    <span
                                      class="text-base-content/50 text-xs font-medium"
                                      >Input</span
                                    >
                                    <div class="mt-0.5">
                                      {#if hasToolInput(part)}
                                        <pre
                                          class="text-xs overflow-x-auto rounded py-1.5 font-mono text-base-content/80">{formatToolInput(
                                            part.input,
                                          )}</pre>
                                      {:else}
                                        <span
                                          class="text-base-content/50 italic text-xs"
                                          >Calling…</span
                                        >
                                      {/if}
                                    </div>
                                  </div>
                                  <div>
                                    <span
                                      class="text-base-content/50 text-xs font-medium"
                                      >Output</span
                                    >
                                    <div class="mt-0.5">
                                      {#if hasError}
                                        <div class="text-xs">
                                          {getToolOutputError(part)}
                                        </div>
                                      {:else if hasOutput}
                                        <pre
                                          class="text-xs overflow-x-auto rounded py-1.5 font-mono text-base-content/80">{formatToolOutput(
                                            part.output,
                                          )}</pre>
                                      {:else if pendingInlineApproval}
                                        {#if toolApprovalPicks[tcId] === undefined}
                                          <ToolApprovalBox
                                            description={approvalItem?.approvalDescription ??
                                              ""}
                                            method={getToolInputString(
                                              approvalItem?.input,
                                              "method",
                                            )}
                                            url={getToolInputString(
                                              approvalItem?.input,
                                              "url_path",
                                            )}
                                            onRun={() =>
                                              applyToolApprovalRun(tcId)}
                                            onSkip={() =>
                                              applyToolApprovalSkip(tcId)}
                                          />
                                        {:else if toolApprovalPicks[tcId] === true}
                                          <p
                                            class="text-xs text-base-content/60 italic"
                                          >
                                            Approved.
                                          </p>
                                        {:else}
                                          <p
                                            class="text-xs text-base-content/60 italic"
                                          >
                                            Skipped.
                                          </p>
                                        {/if}
                                      {:else}
                                        <div
                                          class="flex items-center gap-2 text-base-content/50 italic text-xs"
                                        >
                                          <span
                                            class="inline-block w-3 h-3 rounded-full border border-base-content/30 border-t-base-content/60 animate-spin"
                                          />
                                          <span>…</span>
                                        </div>
                                      {/if}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            {/if}
                          </div>
                        {/if}
                      {/each}
                    {/if}
                  {:else if message.role === "assistant" && showStreamingCursor && message.id === lastMessage?.id}
                    {#if !showToolCallDetails}
                      <div class="flex items-start gap-3">
                        <img
                          src="/images/chat_icon_animated.svg"
                          alt=""
                          class="w-9 h-9 shrink-0 -mt-1.5"
                        />
                        <div class="flex flex-col">
                          <ChatStatusSteps
                            parts={[]}
                            isLoading={true}
                            isLastMessage={true}
                            {showActivityIndicator}
                          />
                        </div>
                      </div>
                    {:else}
                      <div class="flex items-center py-0.5" aria-hidden="true">
                        <ChatLoading />
                      </div>
                    {/if}
                  {:else if message.content}
                    <div class="whitespace-pre-wrap">{message.content}</div>
                  {/if}
                </div>
              {/if}
            </div>
          {/if}
        {/each}
        <div
          bind:this={messagesEndRef}
          class="shrink-0 min-w-[24px] min-h-[24px]"
          aria-hidden="true"
        />
      </div>
    </div>

    {#if versionRequired}
      <div class="flex-none w-full md:max-w-3xl md:mx-auto px-1 pt-2">
        <div class="rounded-lg border border-error/40 bg-error/5 px-3 py-2">
          <Warning
            warning_color="error"
            inline={true}
            markdown
            trusted
            warning_message={"A newer version of Kiln is required to continue using chat. [Check for updates](/settings/check_for_update)"}
          />
        </div>
      </div>
    {:else if upgradeNudgeVersion}
      <div class="flex-none w-full md:max-w-3xl md:mx-auto px-1 pt-2">
        <div
          class="flex items-center gap-2 rounded-lg border border-warning/40 bg-warning/5 px-3 py-2"
        >
          <div class="flex-1 min-w-0">
            <Warning
              warning_color="warning"
              inline={true}
              markdown
              trusted
              warning_message={`A newer version of Kiln (${upgradeNudgeVersion}) is available. [Check for updates](/settings/check_for_update)`}
            />
          </div>
          <button
            type="button"
            class="shrink-0 text-base-content/40 hover:text-base-content/70 transition-colors"
            on:click={dismissUpgradeNudge}
            aria-label="Dismiss upgrade notice"
          >
            <span class="size-4 block"><CloseIcon /></span>
          </button>
        </div>
      </div>
    {/if}

    <form
      class="flex-none relative w-full md:max-w-3xl md:mx-auto px-1 pt-2"
      on:submit|preventDefault={handleSubmit}
    >
      <textarea
        bind:this={textareaRef}
        class="input input-bordered w-full min-h-[80px] max-h-[40vh] resize-none overflow-y-auto py-3 pr-12 text-sm"
        aria-label="Chat message"
        placeholder="Type a message…"
        bind:value={input}
        disabled={inputDisabled}
        rows={3}
        on:input={() => adjustTextareaHeight()}
        on:keydown={handleTextareaKeydown}
      />
      {#if isLoading}
        <button
          type="button"
          class="absolute right-3 bottom-6 btn btn-sm btn-circle btn-neutral"
          on:click={stop}
          aria-label="Stop"
        >
          <span class="size-4 block"><StopIcon /></span>
        </button>
      {:else}
        <button
          type="submit"
          class="absolute right-3 bottom-6 btn btn-sm btn-circle btn-primary"
          disabled={!input.trim() || inputDisabled}
          aria-label="Send"
        >
          <span class="size-4 block"><ArrowUpIcon /></span>
        </button>
      {/if}
    </form>
  </div>
</div>

<ChatCostDisclaimer bind:this={costDisclaimer} />

<style>
  .chat-messages-scroll::-webkit-scrollbar {
    width: 6px;
  }

  .chat-messages-scroll::-webkit-scrollbar-track {
    background: transparent;
  }

  .chat-messages-scroll::-webkit-scrollbar-thumb {
    background-color: oklch(var(--bc) / 0.2);
    border-radius: 3px;
  }

  .chat-messages-scroll::-webkit-scrollbar-thumb:hover {
    background-color: oklch(var(--bc) / 0.35);
  }

  .chat-messages-scroll {
    scrollbar-width: thin;
    scrollbar-color: oklch(var(--bc) / 0.2) transparent;
  }
</style>
