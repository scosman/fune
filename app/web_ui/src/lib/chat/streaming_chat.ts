/**
 * Custom streaming chat: parses SSE from the backend (AI SDK protocol JSON events).
 * Does not use @ai-sdk/svelte because we use Svelte 4 and @ai-sdk/svelte uses Svelte 5.
 *
 * There is an ancient version of the lib that works with Svelte 4, but then that forces us
 * to use an old version of the protocol on the backend too, which is not a good idea.
 */

import { CHAT_CLIENT_VERSION_TOO_OLD } from "$lib/error_codes"
import { sse_data_payloads } from "$lib/utils/sse"

export type ChatMessagePart =
  | { type: "text"; text: string }
  | { type: "reasoning"; reasoning: string }
  | {
      type: `tool-${string}`
      toolCallId: string
      toolName?: string
      input?: unknown
      output?: unknown
    }

export interface ChatMessage {
  id: string
  role: "user" | "assistant" | "system" | "error"
  content?: string
  parts?: ChatMessagePart[]
  /** Server-issued id from ``kiln_chat_trace`` for this assistant turn */
  traceId?: string
  /** Machine-readable error code from upstream (e.g. ``CHAT_CLIENT_VERSION_TOO_OLD``) */
  errorCode?: string
}

/** Body for POST /api/chat: typically one new user message plus optional trace_id for continuation. */
export interface BackendChatRequest {
  messages: Array<{
    role: string
    content?: string
    parts?: Array<Record<string, unknown>>
  }>
  trace_id?: string
}

function toBackendMessage(m: ChatMessage): BackendChatRequest["messages"][0] {
  if (m.role === "user") {
    return { role: "user", content: m.content ?? "" }
  }
  if (m.role === "assistant" && m.parts?.length) {
    return {
      role: "assistant",
      parts: m.parts.map((p) => {
        if (p.type === "text") return { type: "text", text: p.text }
        if (p.type === "reasoning")
          return { type: "reasoning", reasoning: p.reasoning }
        return {
          type: p.type,
          toolCallId: p.toolCallId,
          toolName: p.toolName,
          input: p.input,
          output: p.output,
        }
      }),
    }
  }
  return { role: m.role, content: m.content ?? "" }
}

/** SSE event from backend (AI SDK stream event shape) */
interface StreamEvent {
  type: string
  delta?: string
  id?: string
  messageId?: string
  toolCallId?: string
  toolName?: string
  input?: unknown
  inputTextDelta?: string
  output?: unknown
  errorText?: string
  trace_id?: string
  message?: string
  code?: string
  messageMetadata?: { finishReason?: string; usage?: unknown }
  items?: ToolCallsPendingItem[]
  tool_count?: number
  /** Preferred client version from a ``kiln_client_upgrade_nudge`` event */
  preferred_version?: string
}

export interface ToolCallsPendingItem {
  toolCallId: string
  toolName: string
  input: unknown
  requiresApproval: boolean
  permission?: string
  approvalDescription?: string
}

export interface ToolCallsPendingPayload {
  items: ToolCallsPendingItem[]
}

export interface StreamChatOptions {
  apiUrl: string
  messages: ChatMessage[]
  /** Prior-turn id from kiln_chat_trace; send with the new user message only */
  traceId?: string
  onAssistantMessage: (update: (draft: ChatMessage) => void) => void
  /** Fired when upstream sends ``kiln_chat_trace`` (typically end of a turn) */
  onChatTrace?: (traceId: string) => void
  /** Fired when backend sends an inline error event */
  onInlineError?: (message: string, traceId?: string, code?: string) => void
  /**
   * Fired when the server nudges the client to upgrade (non-blocking): the
   * client is older than the server's preferred version but still above the
   * enforced minimum. ``preferredVersion`` is the version to suggest.
   */
  onVersionNudge?: (preferredVersion: string) => void
  /**
   * After ``tool-calls-pending`` the stream ends; return whether to run each
   * tool that requires approval (toolCallId → allowed).
   */
  onToolCallsPending?: (
    payload: ToolCallsPendingPayload,
  ) => Promise<Record<string, boolean>>
  onToolExecutionStart?: (toolCount: number) => void
  onToolExecutionEnd?: (toolCount: number) => void
  onShowActivityIndicator?: (show: boolean) => void
  onFinish: () => void
  onError: (error: Error) => void
  signal?: AbortSignal
}

/** ``POST /api/chat/execute-tools`` URL for a given ``POST /api/chat`` URL. */
export function chatExecuteToolsUrl(chatApiUrl: string): string {
  const trimmed = chatApiUrl.replace(/\/$/, "")
  return `${trimmed}/execute-tools`
}

function generateId(): string {
  return `msg-${crypto.randomUUID()}`
}

/** Latest stored assistant ``traceId`` for continuing the conversation. */
export function traceIdForNextChatRequest(
  msgs: ChatMessage[],
): string | undefined {
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i]
    if (m.role === "assistant" && m.traceId) {
      return m.traceId
    }
  }
  return undefined
}

type PartSlot =
  | { kind: "text"; id: string }
  | { kind: "reasoning"; id: string }
  | { kind: "tool"; id: string }

class StreamEventProcessor {
  private partOrder: PartSlot[] = []
  private textBlocks = new Map<string, string>()
  private reasoningBlocks = new Map<string, string>()
  private toolMap = new Map<
    string,
    {
      type: `tool-${string}`
      toolCallId: string
      toolName?: string
      input?: unknown
      output?: unknown
    }
  >()
  private toolInputBuffer = new Map<string, string>()
  private currentTextId: string | null = null
  private currentReasoningId: string | null = null
  private slotIdCounter = 0

  private onAssistantMessage: (update: (draft: ChatMessage) => void) => void
  private onChatTrace?: (traceId: string) => void
  private onInlineError?: (
    message: string,
    traceId?: string,
    code?: string,
  ) => void
  private onVersionNudge?: (preferredVersion: string) => void
  private onToolExecutionStart?: (toolCount: number) => void
  private onToolExecutionEnd?: (toolCount: number) => void
  private onShowActivityIndicator?: (show: boolean) => void

  private HANDLERS: Record<string, (event: StreamEvent) => void>

  constructor(opts: {
    onAssistantMessage: (update: (draft: ChatMessage) => void) => void
    onChatTrace?: (traceId: string) => void
    onInlineError?: (message: string, traceId?: string, code?: string) => void
    onVersionNudge?: (preferredVersion: string) => void
    onToolExecutionStart?: (toolCount: number) => void
    onToolExecutionEnd?: (toolCount: number) => void
    onShowActivityIndicator?: (show: boolean) => void
  }) {
    this.onAssistantMessage = opts.onAssistantMessage
    this.onChatTrace = opts.onChatTrace
    this.onInlineError = opts.onInlineError
    this.onVersionNudge = opts.onVersionNudge
    this.onToolExecutionStart = opts.onToolExecutionStart
    this.onToolExecutionEnd = opts.onToolExecutionEnd
    this.onShowActivityIndicator = opts.onShowActivityIndicator

    this.HANDLERS = {
      "text-start": (e) => {
        this.onShowActivityIndicator?.(false)
        this.handleTextStart(e)
      },
      "text-delta": (e) => this.handleTextDelta(e),
      "text-end": () => {
        this.onShowActivityIndicator?.(true)
        this.handleTextEnd()
      },
      "reasoning-start": () => {
        this.onShowActivityIndicator?.(true)
        this.handleReasoningStart()
      },
      "reasoning-delta": (e) => this.handleReasoningDelta(e),
      "reasoning-end": () => this.handleReasoningEnd(),
      "tool-input-start": (e) => {
        this.onShowActivityIndicator?.(true)
        this.handleToolInputStart(e)
      },
      "tool-input-delta": (e) => this.handleToolInputDelta(e),
      "tool-input-available": (e) => {
        this.onShowActivityIndicator?.(true)
        this.handleToolInputAvailable(e)
      },
      "tool-output-available": (e) => this.handleToolOutputAvailable(e),
      "tool-output-error": (e) => this.handleToolOutputError(e),
      kiln_chat_trace: (e) => this.handleChatTrace(e),
      kiln_client_upgrade_nudge: (e) => this.handleVersionNudge(e),
      "kiln-tool-execution-start": (e) => {
        this.onShowActivityIndicator?.(true)
        this.onToolExecutionStart?.(e.tool_count ?? 0)
      },
      "kiln-tool-execution-end": (e) =>
        this.onToolExecutionEnd?.(e.tool_count ?? 0),
      error: (e) => this.handleError(e),
    }
  }

  handleEvent(event: StreamEvent): void {
    const handler = this.HANDLERS[event.type]
    if (handler) {
      handler(event)
    }
  }

  private nextSlotId(): string {
    this.slotIdCounter += 1
    return `slot-${this.slotIdCounter}`
  }

  private flushAssistant() {
    this.onAssistantMessage((draft) => {
      const next: ChatMessagePart[] = []
      for (const slot of this.partOrder) {
        if (slot.kind === "text") {
          const text = this.textBlocks.get(slot.id)
          if (text) next.push({ type: "text", text })
        } else if (slot.kind === "reasoning") {
          const reasoning = this.reasoningBlocks.get(slot.id)
          if (reasoning) next.push({ type: "reasoning", reasoning })
        } else {
          const tool = this.toolMap.get(slot.id)
          if (tool) next.push(tool)
        }
      }
      draft.parts = next
    })
  }

  private ensureTextSlot(): void {
    if (this.currentTextId === null) {
      const id = this.nextSlotId()
      this.partOrder.push({ kind: "text", id })
      this.currentTextId = id
      this.textBlocks.set(id, "")
    }
  }

  private ensureReasoningSlot(): void {
    if (this.currentReasoningId === null) {
      const id = this.nextSlotId()
      this.partOrder.push({ kind: "reasoning", id })
      this.currentReasoningId = id
      this.reasoningBlocks.set(id, "")
    }
  }

  private ensureToolSlot(key: string, event: StreamEvent): void {
    if (!this.toolMap.has(key)) {
      this.partOrder.push({ kind: "tool", id: key })
      this.toolMap.set(key, {
        type: `tool-${event.toolName ?? "unknown"}`,
        toolCallId: key,
        toolName: event.toolName,
      })
    }
  }

  // -- Individual handlers --

  private handleTextStart(_event: StreamEvent): void {
    if (this.currentTextId !== null) {
      this.currentTextId = null
    }
    this.ensureTextSlot()
  }

  private handleTextDelta(event: StreamEvent): void {
    if (this.currentTextId === null) {
      this.ensureTextSlot()
    }
    if (event.delta != null && this.currentTextId !== null) {
      this.textBlocks.set(
        this.currentTextId,
        (this.textBlocks.get(this.currentTextId) ?? "") + event.delta,
      )
      this.flushAssistant()
    }
  }

  private handleTextEnd(): void {
    this.currentTextId = null
  }

  private handleReasoningStart(): void {
    if (this.currentReasoningId !== null) {
      this.currentReasoningId = null
    }
    this.ensureReasoningSlot()
  }

  private handleReasoningDelta(event: StreamEvent): void {
    if (this.currentReasoningId === null) {
      this.ensureReasoningSlot()
    }
    if (event.delta != null && this.currentReasoningId !== null) {
      this.reasoningBlocks.set(
        this.currentReasoningId,
        (this.reasoningBlocks.get(this.currentReasoningId) ?? "") + event.delta,
      )
      this.flushAssistant()
    }
  }

  private handleReasoningEnd(): void {
    this.currentReasoningId = null
  }

  private handleToolInputStart(event: StreamEvent): void {
    if (!event.toolCallId) return
    this.ensureToolSlot(event.toolCallId, event)
    this.flushAssistant()
  }

  private handleToolInputDelta(event: StreamEvent): void {
    if (!event.toolCallId || event.inputTextDelta == null) return
    const key = event.toolCallId
    const prev = this.toolInputBuffer.get(key) ?? ""
    this.toolInputBuffer.set(key, prev + event.inputTextDelta)
    this.ensureToolSlot(key, event)
    const entry = this.toolMap.get(key)!
    try {
      entry.input = JSON.parse(this.toolInputBuffer.get(key) ?? "{}") as unknown
    } catch {
      entry.input = this.toolInputBuffer.get(key)
    }
    this.flushAssistant()
  }

  private handleToolInputAvailable(event: StreamEvent): void {
    if (!event.toolCallId) return
    const key = event.toolCallId
    let entry = this.toolMap.get(key)
    if (!entry) {
      this.partOrder.push({ kind: "tool", id: key })
      entry = {
        type: `tool-${event.toolName ?? "unknown"}`,
        toolCallId: event.toolCallId,
        toolName: event.toolName,
        input: event.input,
      }
      this.toolMap.set(key, entry)
    } else {
      entry.input = event.input
    }
    this.toolInputBuffer.delete(key)
    this.flushAssistant()
  }

  private handleToolOutputAvailable(event: StreamEvent): void {
    if (!event.toolCallId) return
    const entry = this.toolMap.get(event.toolCallId)
    if (entry) {
      entry.output = event.output
      this.flushAssistant()
    }
  }

  private handleToolOutputError(event: StreamEvent): void {
    if (!event.toolCallId) return
    const entry = this.toolMap.get(event.toolCallId)
    if (entry) {
      entry.output = { error: event.errorText }
      this.flushAssistant()
    }
  }

  private handleChatTrace(event: StreamEvent): void {
    const tid = event.trace_id
    if (typeof tid === "string" && tid) {
      this.onChatTrace?.(tid)
    }
  }

  private handleVersionNudge(event: StreamEvent): void {
    const preferred = event.preferred_version
    if (typeof preferred === "string" && preferred) {
      this.onVersionNudge?.(preferred)
    }
  }

  private handleError(event: StreamEvent): void {
    this.onInlineError?.(
      event.message ?? "An error occurred.",
      event.trace_id,
      event.code,
    )
  }
}

async function drainReader(
  reader: ReadableStreamDefaultReader<Uint8Array>,
): Promise<void> {
  while (true) {
    const { done } = await reader.read()
    if (done) break
  }
}

function toolInputAsRecord(input: unknown): Record<string, unknown> {
  if (input !== null && typeof input === "object" && !Array.isArray(input)) {
    return input as Record<string, unknown>
  }
  return {}
}

// ---------------------------------------------------------------------------
// streamChat
// ---------------------------------------------------------------------------

/**
 * POST to apiUrl with the request messages (usually one user turn) and optional trace_id,
 * then parse SSE and call onAssistantMessage for assistant updates. Respects signal for abort.
 */
export async function streamChat(options: StreamChatOptions): Promise<void> {
  const {
    apiUrl,
    messages,
    traceId,
    onAssistantMessage,
    onChatTrace,
    onInlineError,
    onVersionNudge,
    onToolCallsPending,
    onToolExecutionStart,
    onToolExecutionEnd,
    onShowActivityIndicator,
    onFinish,
    onError,
    signal,
  } = options

  const body: BackendChatRequest = {
    messages: messages.map(toBackendMessage),
  }
  if (traceId) {
    body.trace_id = traceId
  }

  let response: Response
  try {
    response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    })
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      onFinish()
      return
    }
    onError(err instanceof Error ? err : new Error(String(err)))
    return
  }

  if (!response.ok) {
    const text = await response.text()
    let code: string | undefined
    try {
      const parsed = JSON.parse(text)
      code = parsed?.code ?? parsed?.message?.code
    } catch {
      /* not JSON */
    }
    if (code === CHAT_CLIENT_VERSION_TOO_OLD) {
      onInlineError?.(
        "Please update the Kiln desktop app to continue using chat.",
        undefined,
        code,
      )
    } else {
      onError(
        new Error(
          `Chat API error ${response.status}: ${text || response.statusText}`,
        ),
      )
    }
    return
  }

  const initialReader = response.body?.getReader()
  if (!initialReader) {
    onError(new Error("No response body"))
    return
  }

  const executeToolsUrl = chatExecuteToolsUrl(apiUrl)
  let currentTraceId: string | undefined = traceId
  let reader: ReadableStreamDefaultReader<Uint8Array> = initialReader

  const processor = new StreamEventProcessor({
    onAssistantMessage,
    onChatTrace: (tid) => {
      currentTraceId = tid
      onChatTrace?.(tid)
    },
    onInlineError,
    onVersionNudge,
    onToolExecutionStart,
    onToolExecutionEnd,
    onShowActivityIndicator,
  })

  try {
    outer: while (true) {
      // A fresh line generator per reader: the stream is swapped after
      // tool execution (`continue outer` below restarts on the new reader).
      for await (const payload of sse_data_payloads(reader)) {
        if (payload === "[DONE]") continue
        let event: StreamEvent
        try {
          event = JSON.parse(payload) as StreamEvent
        } catch {
          continue
        }
        if (event.type === "tool-calls-pending") {
          const items = event.items
          if (!Array.isArray(items) || items.length === 0) {
            onError(new Error("Invalid tool-calls-pending event from server"))
            return
          }
          if (!currentTraceId) {
            onError(
              new Error(
                "Missing trace id for tool execution; wait for chat trace before tools.",
              ),
            )
            return
          }
          let decisions: Record<string, boolean>
          if (onToolCallsPending) {
            decisions = await onToolCallsPending({ items })
          } else {
            decisions = {}
            for (const it of items) {
              if (it.requiresApproval && typeof it.toolCallId === "string") {
                decisions[it.toolCallId] = false
              }
            }
          }
          const decisionsPayload: Record<string, boolean> = {}
          for (const it of items) {
            if (it.requiresApproval && typeof it.toolCallId === "string") {
              decisionsPayload[it.toolCallId] =
                decisions[it.toolCallId] ?? false
            }
          }
          const toolCallsPayload = items.map((it) => ({
            toolCallId: it.toolCallId,
            toolName: it.toolName,
            input: toolInputAsRecord(it.input),
            requiresApproval: Boolean(it.requiresApproval),
          }))
          let postRes: Response
          try {
            postRes = await fetch(executeToolsUrl, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                trace_id: currentTraceId,
                tool_calls: toolCallsPayload,
                decisions: decisionsPayload,
              }),
              signal,
            })
          } catch (err) {
            if ((err as Error).name === "AbortError") {
              onFinish()
              return
            }
            onError(err instanceof Error ? err : new Error(String(err)))
            return
          }
          if (!postRes.ok) {
            const text = await postRes.text()
            onError(
              new Error(
                `Tool execute API error ${postRes.status}: ${text || postRes.statusText}`,
              ),
            )
            return
          }
          const nextReader = postRes.body?.getReader()
          if (!nextReader) {
            onError(new Error("No response body from execute-tools"))
            return
          }
          await drainReader(reader)
          reader = nextReader
          continue outer
        }
        processor.handleEvent(event)
      }
      // Stream ended without a reader swap — the conversation turn is done.
      break
    }
    onFinish()
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      onFinish()
      return
    }
    onError(err instanceof Error ? err : new Error(String(err)))
  }
}

export { generateId as chatGenerateId }
