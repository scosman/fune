// Shared capture registry for the run-page component test. The stubbed
// ChatTrace / MultiturnComposer write their latest props here so the test can
// assert on the truncation state and drive the composer's lifecycle callbacks.

export type ChatTraceCapture = {
  truncate_at_trace_index: number | null
  on_fork?: (run_id: string, trace_index: number) => void
}

export type ComposerCapture = {
  mode: string
  busy: boolean
  parent_task_run_id: string | null
  on_send_start?: (text: string) => void
  on_success?: (new_run_id: string) => void | Promise<void>
  on_send_settled?: (ok: boolean) => void
}

export const stubState: {
  chatTrace: ChatTraceCapture | null
  composers: Record<string, ComposerCapture>
} = { chatTrace: null, composers: {} }

export function resetStubState() {
  stubState.chatTrace = null
  stubState.composers = {}
}
