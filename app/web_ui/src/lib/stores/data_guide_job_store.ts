import { writable, get } from "svelte/store"
import { client } from "$lib/api_client"
import type { KilnAgentRunConfigProperties } from "$lib/types"
import type { DataGuideCaller } from "$lib/utils/data_guide_return"

// Per-task tracking for the Data Guide draft job. The draft runs as a
// kiln_server background job (see copilot_api.py); the user can leave the page
// and come back, so we persist enough per task to (a) resume the spinner page,
// (b) drive the task-wide progress indicator, and (c) re-seed the input page if
// the job fails.
//
// Each task's record lives under its own localStorage key so a quota failure on
// one task can't wipe another's tracking. An in-memory map mirrors them so
// components stay reactive, and a single polling loop in this module keeps
// running jobs' statuses fresh app-wide.

export type DataGuideJobStatus =
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"

// Statuses we treat as "still working". Anything the server returns that isn't
// a known terminal state is treated as running so we keep polling.
const TERMINAL_STATUSES = new Set<string>(["succeeded", "failed", "cancelled"])

export type DataGuideJobRecord = {
  job_id: string
  project_id: string
  task_id: string
  status: DataGuideJobStatus
  // Persisted so a failed job can drop the user back on the input page with
  // their examples intact. Dropped from storage if too large to persist (see
  // persist()).
  input_examples: string[]
  // Needed to generate preview inputs once the draft is ready, even after a
  // hard refresh that loses the in-memory run config.
  run_config_properties: KilnAgentRunConfigProperties
  // Which page opened the setup chain the job was started from, so the
  // progress widget's link returns there once the draft is reviewed.
  // Records written before this field existed read as no caller.
  caller?: DataGuideCaller | null
  // ISO timestamp string, set by the caller (Date.now is fine in the browser).
  created_at: string
  // The user dismissed the task-wide progress indicator for this job (closed
  // it, or clicked through once the draft completed). The record itself
  // survives — only the nudge is hidden.
  acknowledged?: boolean
}

const KEY_PREFIX = "data_guide_job_v1_"
const POLL_INTERVAL_MS = 3000
// Rough localStorage budget for a single record. input_examples can include
// full document text, so we drop them from storage (but keep them in memory)
// when the serialized record would blow past this.
const MAX_PERSISTED_RECORD_BYTES = 1_000_000

const isBrowser = typeof window !== "undefined" && !!window.localStorage

function map_key(project_id: string, task_id: string): string {
  return `${project_id}/${task_id}`
}

function storage_key(project_id: string, task_id: string): string {
  return `${KEY_PREFIX}${project_id}_${task_id}`
}

// In-memory mirror of all tracked jobs, keyed by `${project_id}/${task_id}`.
export const data_guide_jobs =
  writable<Record<string, DataGuideJobRecord>>(hydrate())

function hydrate(): Record<string, DataGuideJobRecord> {
  const jobs: Record<string, DataGuideJobRecord> = {}
  if (!isBrowser) return jobs
  // Removing an entry mid-loop shifts the remaining indices and would skip
  // items, so collect corrupt keys during the index walk and delete them after.
  const corrupt_keys: string[] = []
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i)
    if (!key || !key.startsWith(KEY_PREFIX)) continue
    try {
      const record = JSON.parse(
        localStorage.getItem(key) || "null",
      ) as DataGuideJobRecord | null
      if (record?.job_id && record.project_id && record.task_id) {
        jobs[map_key(record.project_id, record.task_id)] = record
      }
    } catch {
      // Corrupt entry — mark it for removal so it can't wedge the store.
      corrupt_keys.push(key)
    }
  }
  for (const key of corrupt_keys) {
    try {
      localStorage.removeItem(key)
    } catch {
      // ignore
    }
  }
  return jobs
}

function persist(record: DataGuideJobRecord) {
  if (!isBrowser) return
  const key = storage_key(record.project_id, record.task_id)
  const write = (value: DataGuideJobRecord) => {
    localStorage.setItem(key, JSON.stringify(value))
  }
  try {
    const serialized = JSON.stringify(record)
    if (serialized.length > MAX_PERSISTED_RECORD_BYTES) {
      // Too big to persist with examples — keep tracking, drop the examples.
      // The failure path will just ask the user to re-add inputs.
      write({ ...record, input_examples: [] })
    } else {
      write(record)
    }
  } catch {
    // Quota or serialization failure — retry without the bulky examples so we
    // at least keep the job tracking. If even that fails, give up silently.
    try {
      write({ ...record, input_examples: [] })
    } catch {
      // ignore
    }
  }
}

export function getDataGuideJob(
  project_id: string,
  task_id: string,
): DataGuideJobRecord | null {
  return get(data_guide_jobs)[map_key(project_id, task_id)] ?? null
}

export function setDataGuideJob(record: DataGuideJobRecord) {
  data_guide_jobs.update((jobs) => ({
    ...jobs,
    [map_key(record.project_id, record.task_id)]: record,
  }))
  persist(record)
  ensure_polling()
}

export function acknowledgeDataGuideJob(project_id: string, task_id: string) {
  const existing = getDataGuideJob(project_id, task_id)
  if (!existing || existing.acknowledged) return
  setDataGuideJob({ ...existing, acknowledged: true })
}

export function clearDataGuideJob(project_id: string, task_id: string) {
  const k = map_key(project_id, task_id)
  data_guide_jobs.update((jobs) => {
    const next = { ...jobs }
    delete next[k]
    return next
  })
  stop_polling(k)
  if (isBrowser) {
    try {
      localStorage.removeItem(storage_key(project_id, task_id))
    } catch {
      // ignore
    }
  }
}

function update_status(
  project_id: string,
  task_id: string,
  raw_status: string,
) {
  const status = (
    TERMINAL_STATUSES.has(raw_status) ? raw_status : "running"
  ) as DataGuideJobStatus
  const existing = getDataGuideJob(project_id, task_id)
  if (!existing || existing.status === status) return
  setDataGuideJob({ ...existing, status })
}

// --- Polling -------------------------------------------------------------
// One interval per running job, owned by this module. Both the spinner page
// and the progress widget read the shared store rather than polling
// themselves, so a job is polled exactly once regardless of how many views
// are mounted.

const pollers = new Map<string, ReturnType<typeof setInterval>>()

async function poll_once(record: DataGuideJobRecord) {
  try {
    const { data } = await client.GET(
      "/api/projects/{project_id}/tasks/{task_id}/copilot/data_guide_job/{job_id}/status",
      {
        params: {
          path: {
            project_id: record.project_id,
            task_id: record.task_id,
            job_id: record.job_id,
          },
        },
      },
    )
    if (data?.status) {
      update_status(record.project_id, record.task_id, data.status)
    }
  } catch {
    // Transient network errors are ignored — the next tick retries. A job that
    // genuinely failed reports a terminal status, which is handled above.
  }
}

function stop_polling(map_key_value: string) {
  const timer = pollers.get(map_key_value)
  if (timer) {
    clearInterval(timer)
    pollers.delete(map_key_value)
  }
}

export function ensure_polling() {
  if (!isBrowser) return
  const jobs = get(data_guide_jobs)
  // Start pollers for running jobs, stop them for finished/removed ones.
  for (const [k, record] of Object.entries(jobs)) {
    if (record.status === "running" && !pollers.has(k)) {
      const timer = setInterval(() => poll_once(record), POLL_INTERVAL_MS)
      pollers.set(k, timer)
      // Kick an immediate poll so a just-restored running job updates fast.
      poll_once(record)
    } else if (record.status !== "running") {
      stop_polling(k)
    }
  }
  for (const k of pollers.keys()) {
    if (!jobs[k]) stop_polling(k)
  }
}

// Begin polling any jobs restored from localStorage on first import.
ensure_polling()
