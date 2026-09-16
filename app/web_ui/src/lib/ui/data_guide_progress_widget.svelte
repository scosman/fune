<script lang="ts">
  // Task-wide nudge back to an in-flight (or just-finished) Data Guide draft
  // job. Mirrors ProgressWidget's look, but is backed by the data guide job
  // store and does its own status-driven copy. The shared store polls the job
  // in the background, so this stays live wherever the user navigates.
  //
  // Click behaviour matches the product spec: while the draft is in progress,
  // clicking jumps to the spinner page but the nudge persists (you'll want it
  // again); once complete, clicking acknowledges it so it goes away.
  import { goto } from "$app/navigation"
  import { page } from "$app/stores"
  import {
    data_guide_jobs,
    acknowledgeDataGuideJob,
    type DataGuideJobRecord,
  } from "$lib/stores/data_guide_job_store"
  import { with_data_guide_caller } from "$lib/utils/data_guide_return"

  // Show the most recently started job the user hasn't dismissed. There's
  // normally at most one in flight, but picking the newest keeps things sane if
  // several tasks have records.
  function pick_job(
    jobs: Record<string, DataGuideJobRecord>,
  ): DataGuideJobRecord | null {
    const candidates = Object.values(jobs).filter((j) => !j.acknowledged)
    if (candidates.length === 0) return null
    return candidates.sort((a, b) =>
      (b.created_at || "").localeCompare(a.created_at || ""),
    )[0]
  }

  $: job = pick_job($data_guide_jobs)

  function spinner_link(j: DataGuideJobRecord): string {
    return with_data_guide_caller(
      `/generate/${j.project_id}/${j.task_id}/data_guide_setup_copilot/${j.job_id}`,
      j.caller ?? null,
    )
  }

  // Hide while the user is already inside this job's setup flow (spinner, base
  // page, or review) — the nudge would just be pointing at the current page.
  $: on_job_page =
    !!job &&
    $page.url.pathname.startsWith(
      `/generate/${job.project_id}/${job.task_id}/data_guide_setup_copilot`,
    )

  $: visible = !!job && !on_job_page

  function open() {
    if (!job) return
    // Acknowledge once the draft is ready (or terminal) so the nudge doesn't
    // linger after the user has seen it. Leave it in place while still running.
    if (job.status !== "running") {
      acknowledgeDataGuideJob(job.project_id, job.task_id)
    }
    goto(spinner_link(job))
  }

  function close() {
    if (!job) return
    acknowledgeDataGuideJob(job.project_id, job.task_id)
  }

  // Mirror the prompt-optimization (GEPA) status badges: outline pill, primary
  // when complete, error when failed, with a spinner while in progress.
  $: is_failed = job?.status === "failed" || job?.status === "cancelled"
  $: is_running = !!job && job.status !== "succeeded" && !is_failed
  $: badge = is_failed
    ? { label: "Failed", cls: "badge-outline badge-error" }
    : job?.status === "succeeded"
      ? { label: "Complete", cls: "badge-outline badge-primary" }
      : { label: "In Progress", cls: "badge-outline badge-primary" }

  $: body = is_failed
    ? "We couldn't draft your data guide. Please try again."
    : job?.status === "succeeded"
      ? "Your data guide is ready to "
      : "Drafting your data guide."

  $: cta = is_failed ? "" : job?.status === "succeeded" ? "review." : ""
</script>

{#if visible && job}
  <!-- Structure mirrors progress_widget.svelte (the eval widget): a single
       card button so it gets the same DaisyUI sidebar-menu sizing. A wrapper
       div is treated differently by the menu and stretched the card on wide
       screens. The dismiss control is a nested button with stopPropagation —
       same pattern as the eval widget. -->
  <button
    class="bg-white border border-primary rounded-lg p-3 flex flex-col gap-1 items-start relative text-left text-xs 2xl:text-sm max-w-full mb-2"
    on:click={open}
  >
    <button
      class="hover:text-xl h-8 w-8 leading-none absolute top-0 right-0 flex items-center justify-center"
      aria-label="Dismiss data guide notification"
      on:click|stopPropagation={close}>&#x2715;</button
    >
    <div class="font-medium pr-6">Data Guide</div>
    <div class="font-light">
      {body}{#if cta}
        <span class="text-primary font-medium">{cta}</span>{/if}
    </div>
    <div class="badge px-3 py-1 gap-1 text-xs {badge.cls}">
      {#if is_running}
        <span class="loading loading-spinner h-[12px] w-[12px]"></span>
      {/if}
      {badge.label}
    </div>
  </button>
{/if}
