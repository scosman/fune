<script lang="ts">
  // Copilot path for creating an input data guide. New "create" step gathers
  // inputs (uploaded text docs + manual entries + selected task runs), then
  // calls the new copilot endpoint to seed an initial draft + preview. From
  // there we drop into the existing manual flow's preview/refine UI so the
  // post-seed experience is identical to the manual path.
  import AppPage from "../../../../app_page.svelte"
  import { client } from "$lib/api_client"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { onMount, onDestroy } from "svelte"
  import { page } from "$app/stores"
  import { goto, beforeNavigate } from "$app/navigation"
  import { get } from "svelte/store"
  import GuidePreview from "../data_guide_setup/guide_preview.svelte"
  import RunOptionsTiles from "../data_guide_setup/run_options_tiles.svelte"
  import GenerationSettingsTrigger from "../data_guide_setup/generation_settings_trigger.svelte"
  import FormContainer from "$lib/utils/form_container.svelte"
  import DataGenDescription from "../data_gen_description.svelte"
  import { SynthDataGuidanceDataModel } from "../synth_data_guidance_datamodel"
  import type { KilnAgentRunConfigProperties, Task } from "$lib/types"
  import { isKilnAgentRunConfig } from "$lib/types"
  import { agentInfo } from "$lib/agent"
  import { current_task } from "$lib/stores"
  import AnalyzingAnimation from "$lib/ui/animations/analyzing_animation.svelte"
  import RefiningAnimation from "$lib/ui/animations/refining_animation.svelte"
  import Completed from "$lib/ui/completed.svelte"
  import { dedupe_by_input } from "$lib/utils/dedupe_by_input"
  import InputExamplesUploader, {
    type InputExampleEntry,
    MAX_EXAMPLE_LENGTH,
  } from "./input_examples_uploader.svelte"
  import { pending_data_guide_example } from "../data_guide_setup/pending_example_store"
  import { pending_data_guide_draft } from "./pending_draft_store"
  import {
    getDataGuideJob,
    setDataGuideJob,
    clearDataGuideJob,
  } from "$lib/stores/data_guide_job_store"
  import posthog from "posthog-js"
  import { checkKilnCopilotAvailable } from "$lib/utils/copilot_utils"
  import {
    data_guide_return,
    read_data_guide_caller,
    stash_data_guide_caller,
    with_data_guide_caller,
  } from "$lib/utils/data_guide_return"
  import ExtractionDialog from "$lib/components/extraction_dialog.svelte"
  import type Dialog from "$lib/ui/dialog.svelte"

  type CopilotState =
    | "loading"
    | "create"
    | "analyzing"
    | "preview"
    | "refining"
    | "regenerating"
    | "load_error"

  let current_state: CopilotState = "loading"
  let error: KilnError | null = null
  // Example generation failed. Kept separate from `error` (which the review
  // form renders inline): this one takes over the review table on the preview
  // screen, so the user still lands on the page they were headed for, with the
  // drafted guide intact below.
  let preview_error: KilnError | null = null
  let submitting = false
  let saved = false

  let entries: InputExampleEntry[] = []

  let guide: string = ""
  type PreviewSample = { input: string }
  type ReviewedSample = { input: string; looks_good: boolean | undefined }
  let preview_samples: PreviewSample[] = []
  let preview_initial_guide: string = ""
  let reviewed_samples: ReviewedSample[] = []
  let general_feedback: string = ""

  let captured_input_run_config: KilnAgentRunConfigProperties | null = null
  let refine_iterations = 0

  let task: Task | null = null
  let run_options_tiles: RunOptionsTiles | null = null
  // Friendly names of the selected generation model + provider, for the trigger.
  let generation_model_name = ""
  let generation_provider = ""

  let guidance_data: SynthDataGuidanceDataModel =
    new SynthDataGuidanceDataModel()
  onDestroy(() => {
    guidance_data.destroy()
  })

  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  // Which page opened the setup chain (no key means synthetic data
  // generation): every link out of here forwards it, and the breadcrumb and
  // the finish action return to it.
  $: caller = read_data_guide_caller($page.url.searchParams)
  $: return_target = data_guide_return(caller, project_id, task_id)
  $: agentInfo.set({
    name: "Set Up Data Guide",
    description: `Use Kiln Pro to draft the input data guide for project ${project_id}, task ${task_id} from a list of example inputs.`,
  })

  $: has_entries = entries.length > 0
  // Document entries (uploads + library picks) to extract at Continue, scoped by
  // id so picks don't need a tag. The backend skips ones already extracted.
  $: extraction_document_ids = entries
    .filter((e) => e.source === "document" && !!e.document_id)
    .map((e) => e.document_id as string)

  // States that run foreground LLM work (generating/refining preview inputs)
  // tied to this page — unlike the draft, this work is NOT a resumable
  // background job, so navigating away discards it. Guard against accidental
  // loss while one of these is in flight.
  $: generating_in_progress =
    current_state === "analyzing" ||
    current_state === "refining" ||
    current_state === "regenerating"

  // Reviewing generated examples with work that isn't persisted yet. The draft
  // is not resumable from here: the ratings, feedback and guide edits only live
  // in this component, and coming back would re-run preview generation from the
  // job's draft. So leaving is a discard, not a pause — confirm it, then drop
  // the tracked job so a return starts setup cleanly (see discard_draft).
  $: reviewing_unsaved = current_state === "preview" && !saved

  // Abandon the draft job so the progress widget clears and the next visit to
  // this page lands on a fresh "create" state rather than regenerating examples
  // from the completed job.
  function discard_draft() {
    clearDataGuideJob(project_id, task_id)
    posthog.capture("data_guide_draft_discarded", { source: "copilot" })
  }

  // SvelteKit (in-app) navigation guard. Only prompt when actually leaving this
  // page — query-param / same-path transitions are fine.
  beforeNavigate((navigation) => {
    if (
      navigation.to?.url?.pathname &&
      navigation.to.url.pathname === navigation.from?.url?.pathname
    ) {
      return
    }
    if (generating_in_progress) {
      if (
        !confirm(
          "Examples are still being generated and will be lost if you leave.\n\n" +
            "Press Cancel to stay, OK to leave.",
        )
      ) {
        navigation.cancel()
      }
      return
    }
    if (reviewing_unsaved) {
      if (
        !confirm(
          "Leaving will discard this data guide draft and the examples you're reviewing.\n\n" +
            "Press Cancel to stay, OK to discard and leave.",
        )
      ) {
        navigation.cancel()
        return
      }
      discard_draft()
    }
  })

  // Browser reload / tab close guard.
  function handle_before_unload(event: BeforeUnloadEvent) {
    if (generating_in_progress || reviewing_unsaved) {
      event.preventDefault()
    }
  }
  // beforeunload can't tell us whether the user went through with it, so the
  // discard happens here: pagehide only fires once the page is actually being
  // torn down (reload, tab close, external nav). Keeps a reload consistent with
  // an in-app leave — both discard rather than silently regenerating examples.
  function handle_page_hide() {
    if (reviewing_unsaved) {
      clearDataGuideJob(project_id, task_id)
    }
  }
  onMount(() => {
    window.addEventListener("beforeunload", handle_before_unload)
    window.addEventListener("pagehide", handle_page_hide)
    return () => {
      window.removeEventListener("beforeunload", handle_before_unload)
      window.removeEventListener("pagehide", handle_page_hide)
    }
  })

  onMount(async () => {
    try {
      const { data, error: api_error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/data_gen_guide",
        { params: { path: { project_id, task_id } } },
      )
      if (api_error) {
        error = createKilnError(api_error)
        current_state = "load_error"
        return
      }
      if (data) {
        goto(
          with_data_guide_caller(
            `/generate/${project_id}/${task_id}/data_guide`,
            caller,
          ),
          { replaceState: true },
        )
        return
      }
    } catch (e) {
      error = createKilnError(e)
      current_state = "load_error"
      return
    }

    // Pro is required for the start endpoint. If the key isn't connected, send
    // the user to the static connect route, which returns here once connected.
    // Checked before any state is consumed below (pending example, job record)
    // so the round-trip through connect leaves the flow exactly as it found it.
    let pro_available = false
    try {
      pro_available = await checkKilnCopilotAvailable()
    } catch {
      pro_available = false
    }
    if (!pro_available) {
      // The connect route's URL is fixed for OAuth, so the caller rides a
      // stash across that hop instead of the URL.
      stash_data_guide_caller(caller)
      goto(`/generate/data_guide_pro_auth`, { replaceState: true })
      return
    }

    if ($current_task?.id === task_id) {
      task = $current_task
    } else {
      try {
        const { data: task_data } = await client.GET(
          "/api/projects/{project_id}/tasks/{task_id}",
          { params: { path: { project_id, task_id } } },
        )
        if (task_data) {
          task = task_data
        }
      } catch {
        // Non-critical
      }
    }

    // If the user added their first example via the synth-page intro and we
    // landed here through the chooser, seed it as a manual entry.
    const seeded = get(pending_data_guide_example)
    if (seeded?.input?.trim()) {
      entries = [
        {
          source: seeded.task_run_id ? "task_run" : "manual",
          label: seeded.task_run_id ? "Existing run" : "Manual 1",
          text: seeded.input,
        },
      ]
      pending_data_guide_example.set(null)
    }

    // --- Resume an in-flight / finished data guide draft job --------------
    // 1. Fresh draft just handed off from the spinner page → drop straight
    //    into preview generation.
    const draft = get(pending_data_guide_draft)
    if (draft) {
      // Always clear the one-shot stash. Only consume it when it matches this
      // task AND this task's currently-tracked job — a draft stashed by a job
      // that's since been superseded (user restarted) must not be picked up
      // here and mis-attributed to the new run.
      const tracked = getDataGuideJob(project_id, task_id)
      const matches =
        draft.project_id === project_id &&
        draft.task_id === task_id &&
        (!tracked || tracked.job_id === draft.job_id)
      pending_data_guide_draft.set(null)
      if (matches) {
        captured_input_run_config = draft.run_config_properties
        guide = draft.draft_guide
        await generate_preview_from_draft()
        return
      }
    }

    const failed_return = new URLSearchParams(window.location.search).has(
      "draft_failed",
    )
    const job = getDataGuideJob(project_id, task_id)

    // 2. A job is still running — the spinner page owns that view.
    if (!failed_return && job?.status === "running") {
      goto(
        with_data_guide_caller(
          `/generate/${project_id}/${task_id}/data_guide_setup_copilot/${job.job_id}`,
          caller,
        ),
        { replaceState: true },
      )
      return
    }

    // 3. A job already succeeded (e.g. hard refresh on this page) — re-fetch
    //    its draft and resume the review flow.
    if (!failed_return && job?.status === "succeeded") {
      captured_input_run_config = job.run_config_properties
      if (await load_draft_from_result(job.job_id)) {
        await generate_preview_from_draft()
        return
      }
      // Fall through to the create state if the result couldn't be fetched.
    }

    // 4. A job failed/cancelled — re-seed the inputs and surface the error so
    //    the user can retry.
    if (
      failed_return ||
      job?.status === "failed" ||
      job?.status === "cancelled"
    ) {
      if (job) {
        entries = job.input_examples.map((text, i) => ({
          source: "manual",
          label: `Example ${i + 1}`,
          text,
        }))
        clearDataGuideJob(project_id, task_id)
      }
      error = new KilnError(
        "The data guide draft didn't finish. Please review your inputs and try again.",
      )
    }

    current_state = "create"
  })

  // Fetch the draft markdown produced by a completed job into `guide`.
  // Returns false (and sets `error`) if the result couldn't be fetched.
  async function load_draft_from_result(job_id: string): Promise<boolean> {
    try {
      const { data, error: api_error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/copilot/data_guide_job/{job_id}/result",
        { params: { path: { project_id, task_id, job_id } } },
      )
      if (api_error) throw api_error
      if (!data?.draft_guide)
        throw new KilnError("No draft guide returned", null)
      guide = data.draft_guide
      return true
    } catch (e) {
      error = createKilnError(e)
      return false
    }
  }

  // Generate preview inputs from the current `guide` and enter the review
  // (preview) state. Used both for a fresh draft handoff and when resuming a
  // completed job. Mirrors the refine path's preview call.
  async function generate_preview_from_draft() {
    if (!captured_input_run_config) {
      error = new KilnError(
        "Pick a kiln_agent run config for input generation.",
      )
      current_state = "create"
      return
    }
    error = null
    preview_error = null
    submitting = true
    current_state = "analyzing"
    try {
      const { data, error: api_error } = await client.POST(
        "/api/projects/{project_id}/tasks/{task_id}/data_gen_guide_preview",
        {
          params: { path: { project_id, task_id } },
          body: {
            guide,
            run_config_properties: captured_input_run_config,
            num_samples: 5,
          },
        },
      )
      if (api_error) throw api_error
      if (!data) throw new KilnError("No preview inputs returned", null)
      preview_samples = dedupe_by_input(data as PreviewSample[])
      reviewed_samples = preview_samples.map((s) => ({
        input: s.input,
        looks_good: undefined,
      }))
      general_feedback = ""
      preview_initial_guide = guide
      current_state = "preview"
    } catch (e) {
      // The draft itself is fine — only the (foreground) input generation
      // failed, typically a provider error like an exhausted API key. Dropping
      // back to "create" here would strand the user: their examples aren't
      // re-seeded, and Continue would pay for a whole new Pro draft job. Land on
      // the review screen instead, with the error where the examples would have
      // been, and retry against the draft we already have.
      preview_error = createKilnError(e)
      preview_samples = []
      reviewed_samples = []
      general_feedback = ""
      preview_initial_guide = guide
      current_state = "preview"
    } finally {
      submitting = false
    }
  }

  // The review screen shows the generation settings, so retrying has to pick up a
  // model the user swapped there — a bad/unavailable model (or a provider whose
  // key is out of credits) is a common cause of the preview failing. Falls back to
  // the config captured earlier in the flow if the picker has nothing usable.
  function retry_preview() {
    const rc = run_options_tiles?.get_input_run_config()
    if (rc && isKilnAgentRunConfig(rc)) {
      captured_input_run_config = rc
    }
    posthog.capture("data_guide_preview_retried", { source: "copilot" })
    generate_preview_from_draft()
  }

  // Abandon the draft after a failed example generation and go back to gathering
  // inputs. Re-seed the examples from the job record first: on a resumed job they
  // were never loaded into `entries`, and making the user re-upload everything
  // after a transient provider error is the dead end Leonard hit.
  function start_over_from_preview_error() {
    const job = getDataGuideJob(project_id, task_id)
    if (entries.length === 0 && job?.input_examples?.length) {
      entries = job.input_examples.map((text, i) => ({
        source: "manual",
        label: `Example ${i + 1}`,
        text,
      }))
    }
    discard_draft()
    guide = ""
    preview_initial_guide = ""
    preview_samples = []
    reviewed_samples = []
    general_feedback = ""
    refine_iterations = 0
    error = null
    preview_error = null
    current_state = "create"
  }

  function handle_entries_change(
    event: CustomEvent<{ entries: InputExampleEntry[] }>,
  ) {
    entries = event.detail.entries
    // Editing the inputs is the user's response to a failure — a stale error
    // from the last attempt hanging over the form reads as if the new inputs
    // are broken too.
    error = null
  }

  let extraction_dialog: Dialog | null = null
  // True while the Continue-triggered extraction run is live. Bound from
  // ExtractionDialog so the Continue button can show progress and See All can
  // mark rows "Extracting…" after the user dismisses the run dialog.
  let extraction_running = false
  // Keep the Continue button spinning while a run is in flight, even after the
  // run dialog is dismissed.
  $: if (extraction_running) submitting = true

  async function handle_analyze() {
    error = null
    if (entries.length === 0) {
      error = new KilnError("Add at least one input example to continue.")
      return
    }
    const rc = run_options_tiles?.get_input_run_config()
    if (!rc || !isKilnAgentRunConfig(rc)) {
      error = new KilnError(
        "Pick a kiln_agent run config for input generation.",
      )
      return
    }
    captured_input_run_config = rc

    // Document entries are added without text — extraction is deferred to here.
    // If any are still pending, open the extractor picker; the run drives the
    // rest of the flow (hydrate text, then analyze) via handle_extraction_complete.
    const pending = entries.some(
      (e) => e.source === "document" && !e.text && !!e.document_id,
    )
    if (pending) {
      extraction_dialog?.show()
      return
    }

    try {
      await run_analyze()
    } catch (e) {
      error = createKilnError(e)
      current_state = "create"
    }
  }

  // Resolve text for document entries that don't have it yet. When the user
  // ran a specific extractor (extractor_config_id), prefer that extractor's
  // extraction — a doc can have several (one per extractor / RAG tool). Fall
  // back to the most recent extraction if there's no exact match.
  async function refresh_extractions(
    extractor_config_id?: string | null,
  ): Promise<InputExampleEntry[]> {
    const updated = await Promise.all(
      entries.map(async (e) => {
        if (e.source !== "document" || e.text || !e.document_id) return e
        try {
          const { data: extractions } = await client.GET(
            "/api/projects/{project_id}/documents/{document_id}/extractions",
            {
              params: {
                path: { project_id, document_id: e.document_id },
              },
            },
          )
          if (!extractions || extractions.length === 0) return e
          const matching = extractor_config_id
            ? extractions.filter((x) => x.extractor?.id === extractor_config_id)
            : []
          const pool = matching.length > 0 ? matching : extractions
          const chosen = [...pool].sort((a, b) =>
            (b.created_at || "").localeCompare(a.created_at || ""),
          )[0]
          if (!chosen) return e
          return {
            ...e,
            text: chosen.output_content,
            extraction_id: chosen.id,
          }
        } catch {
          return e
        }
      }),
    )
    entries = updated
    return updated
  }

  // The run finished (the dialog may already be closed). Hydrate the freshly
  // extracted entries; if any document still has no text it failed extraction —
  // surface a retryable error and stay in create rather than dropping it.
  // Otherwise continue straight into analyze.
  async function handle_extraction_complete(
    event: CustomEvent<{ extractor_config_id: string; error_count: number }>,
  ) {
    error = null
    const extractor_config_id = event.detail?.extractor_config_id
    try {
      const updated = await refresh_extractions(extractor_config_id)
      const still_pending = updated.filter(
        (e) => e.source === "document" && !e.text && !!e.document_id,
      )
      if (still_pending.length > 0) {
        submitting = false
        const n = still_pending.length
        error = new KilnError(
          `${n} document${n === 1 ? "" : "s"} couldn't be extracted. ` +
            `Click Continue to retry, or remove ${n === 1 ? "it" : "them"} from See All.`,
        )
        return
      }
      await run_analyze(updated.filter((e) => !!e.text))
    } catch (e) {
      error = createKilnError(e)
      current_state = "create"
      submitting = false
    }
  }

  // The run errored before completing (e.g. SSE stream failure). Mirror it onto
  // the Continue button so the user can retry — clicking Continue reopens the
  // picker and re-extracts only the still-pending docs.
  function handle_extraction_failed(
    event: CustomEvent<{ error: KilnError | null }>,
  ) {
    submitting = false
    error =
      event.detail?.error ??
      new KilnError("Document extraction failed. Click Continue to retry.")
  }

  // The run dialog was dismissed. If no run is live and we're still gathering
  // inputs, release the Continue button so the user can act again. (When a run
  // is live, or we've already advanced to analyze, leave state untouched.)
  function handle_extraction_dialog_closed() {
    if (!extraction_running && current_state === "create") {
      submitting = false
    }
  }

  async function run_analyze(override_entries?: InputExampleEntry[]) {
    const ready = (override_entries ?? entries).filter((e) => !!e.text)
    if (ready.length === 0) {
      throw new KilnError("No input examples with content to analyze.")
    }
    if (!captured_input_run_config) {
      throw new KilnError("Pick a kiln_agent run config for input generation.")
    }
    // Examples are capped to MAX_TOTAL_ENTRIES at add time (the uploader
    // truncates each add to the remaining room), so `ready` is already within
    // the limit here.
    const to_analyze = ready
    // Hard length cap: block (don't truncate) when any analyzed example exceeds
    // the per-example limit — name the offenders so the user can fix them.
    const over_limit = to_analyze.filter(
      (e) => e.text.length > MAX_EXAMPLE_LENGTH,
    )
    if (over_limit.length > 0) {
      const names = over_limit.map((e) => e.label).join(", ")
      throw new KilnError(
        `${over_limit.length} example${over_limit.length === 1 ? "" : "s"} ` +
          `exceed the ${MAX_EXAMPLE_LENGTH.toLocaleString()} character limit ` +
          `(${names}). Remove or shorten ${over_limit.length === 1 ? "it" : "them"} to continue.`,
      )
    }
    submitting = true
    const input_examples = to_analyze.map((e) => e.text)
    try {
      // Kick off the draft as a background job and hand the user off to the
      // spinner page, which polls it to completion. The job survives the user
      // leaving the page, and the task-wide progress widget links them back.
      const { data, error: api_error } = await client.POST(
        "/api/projects/{project_id}/tasks/{task_id}/copilot/data_guide_job/start",
        {
          params: { path: { project_id, task_id } },
          // Only the examples are sent. The server derives the prompt and input
          // schema from the task (identified by the route); the output schema
          // and description are intentionally never involved.
          body: { input_examples },
        },
      )
      if (api_error) throw api_error
      if (!data?.job_id) throw new KilnError("No job id returned", null)

      setDataGuideJob({
        job_id: data.job_id,
        project_id,
        task_id,
        status: "running",
        input_examples,
        run_config_properties: captured_input_run_config,
        created_at: new Date().toISOString(),
        caller,
      })
      posthog.capture("data_guide_copilot_job_started", {
        entry_count: to_analyze.length,
      })
      // Await the navigation so `submitting` stays true until it's underway.
      // The FormContainer's beforeNavigate guard keys off `!submitting`;
      // resetting it (in `finally`) before the guard runs would pop a spurious
      // "unsaved changes" prompt on this intentional Continue navigation.
      // Awaiting also surfaces a route-chunk load failure as a handled error
      // rather than an unhandled promise rejection.
      await goto(
        with_data_guide_caller(
          `/generate/${project_id}/${task_id}/data_guide_setup_copilot/${data.job_id}`,
          caller,
        ),
      )
    } catch (e) {
      error = createKilnError(e)
      current_state = "create"
    } finally {
      submitting = false
    }
  }

  async function handle_refine(
    event: CustomEvent<{
      feedback: string
      rated_samples: { input: string; looks_good: boolean }[]
    }>,
  ) {
    error = null
    submitting = true
    const has_negative_feedback =
      event.detail.rated_samples.some((s) => !s.looks_good) ||
      event.detail.feedback.trim().length > 0
    current_state = has_negative_feedback ? "refining" : "regenerating"

    try {
      if (!captured_input_run_config) {
        throw new KilnError("No model configuration available", null)
      }
      let refined_guide = guide
      if (has_negative_feedback) {
        const { data, error: api_error } = await client.POST(
          "/api/projects/{project_id}/tasks/{task_id}/data_gen_guide_refine",
          {
            params: { path: { project_id, task_id } },
            body: {
              current_guide: guide,
              source: "kiln_pro",
              feedback: event.detail.feedback,
              preview_samples: event.detail.rated_samples,
              run_config_properties: captured_input_run_config,
            },
          },
        )
        if (api_error) throw api_error
        if (!data) throw new KilnError("No refinement returned", null)
        refined_guide = data.refined_guide
      }
      const { data: preview_data, error: preview_error } = await client.POST(
        "/api/projects/{project_id}/tasks/{task_id}/data_gen_guide_preview",
        {
          params: { path: { project_id, task_id } },
          body: {
            guide: refined_guide,
            run_config_properties: captured_input_run_config,
            num_samples: 5,
          },
        },
      )
      if (preview_error) throw preview_error
      if (!preview_data) throw new KilnError("No preview inputs returned", null)
      guide = refined_guide
      preview_samples = dedupe_by_input(preview_data as PreviewSample[])
      reviewed_samples = preview_samples.map((s) => ({
        input: s.input,
        looks_good: undefined,
      }))
      general_feedback = ""
      preview_initial_guide = guide
      current_state = "preview"
      refine_iterations++
    } catch (e) {
      error = createKilnError(e)
      current_state = "preview"
    } finally {
      submitting = false
    }
  }

  async function handle_save() {
    error = null
    submitting = true
    try {
      const { error: api_error } = await client.PUT(
        "/api/projects/{project_id}/tasks/{task_id}/data_gen_guide",
        {
          params: { path: { project_id, task_id } },
          body: { guide, source: "kiln_pro" },
        },
      )
      if (api_error) throw api_error
      // `saved` both disables the unsaved-changes warn and flips the page to
      // the success screen (Completed), which returns the user to synth via a
      // button rather than auto-navigating.
      saved = true
      // The draft job's work is now persisted — drop the tracking record so
      // the progress widget clears and Create Data Guide starts fresh.
      clearDataGuideJob(project_id, task_id)
      posthog.capture("data_guide_saved", {
        method: "after_preview",
        source: "copilot",
        refine_iterations,
      })
    } catch (e) {
      error = createKilnError(e)
    } finally {
      submitting = false
    }
  }

  // Restart the whole setup process from the review screen: drop the tracked
  // job + draft and return to a clean input-gathering state. After a failed
  // example generation the user never got to review anything, so send them back
  // with their input examples still in hand rather than an empty create screen.
  function handle_restart() {
    if (preview_error) {
      start_over_from_preview_error()
      return
    }
    clearDataGuideJob(project_id, task_id)
    guide = ""
    preview_initial_guide = ""
    preview_samples = []
    reviewed_samples = []
    general_feedback = ""
    refine_iterations = 0
    entries = []
    error = null
    saved = false
    posthog.capture("data_guide_restarted", { source: "copilot" })
    current_state = "create"
  }
</script>

<div class="max-w-[1400px]">
  <AppPage
    title="Set Up Data Guide"
    subtitle="Your Data Guide will help us generate better synthetic inputs."
    sub_subtitle="Read the Docs"
    sub_subtitle_link="https://docs.kiln.tech/docs/synthetic-data-generation"
    breadcrumbs={[
      {
        label: return_target.label,
        href: return_target.href,
        replace_state: true,
      },
    ]}
  >
    <DataGenDescription bind:guidance_data />

    {#if saved}
      <Completed
        title="Data Guide Saved"
        subtitle={`Your Data Guide is saved. Click Continue to return to ${return_target.label}.`}
        link={return_target.href}
        button_text="Continue"
      />
    {:else if current_state === "loading"}
      <div class="flex flex-col items-center justify-center py-24 gap-4">
        <span class="loading loading-spinner loading-lg" />
      </div>
    {:else if current_state === "create"}
      <FormContainer
        submit_label="Continue"
        submitting_status={extraction_running ? "Extracting documents…" : ""}
        on:submit={handle_analyze}
        bind:error
        bind:submitting
        submit_disabled={!has_entries || extraction_running}
        compact_button={true}
        warn_before_unload={has_entries && !submitting}
        submit_visible={has_entries}
      >
        <InputExamplesUploader
          {project_id}
          {task_id}
          {task}
          {entries}
          extraction_in_progress={extraction_running}
          on:change={handle_entries_change}
        />

        {#if has_entries}
          <GenerationSettingsTrigger
            model_name={generation_model_name}
            provider={generation_provider}
            open={() => run_options_tiles?.open_combined_dialog()}
          />
        {/if}
      </FormContainer>
    {:else if current_state === "analyzing"}
      <AnalyzingAnimation
        title="Generating Examples"
        description="Generating example inputs from your data guide for you to review. Leaving this page will discard them."
      />
    {:else if current_state === "preview"}
      <GuidePreview
        initial_guide={preview_initial_guide}
        bind:guide
        bind:error
        bind:submitting
        bind:reviewed_samples
        bind:general_feedback
        {saved}
        samples_error={preview_error}
        {generation_model_name}
        {generation_provider}
        open_generation_settings={() =>
          run_options_tiles?.open_combined_dialog()}
        show_restart={true}
        warn_on_leave={false}
        on:refine={handle_refine}
        on:save={handle_save}
        on:restart={handle_restart}
        on:retry={retry_preview}
      />
    {:else if current_state === "refining"}
      <RefiningAnimation
        title="Refining Data Guide"
        description="Refining your data guide with the feedback you provided and generating fresh inputs to review."
      />
    {:else if current_state === "regenerating"}
      <AnalyzingAnimation
        title="Regenerating Inputs"
        description="Regenerating synthetic inputs to test your edited data guide. Leaving this page will discard them."
      />
    {:else if current_state === "load_error"}
      <div
        class="w-full min-h-[50vh] flex flex-col justify-center items-center gap-2"
      >
        <div class="text-error text-sm">
          {error?.getMessage() ?? "An unknown error occurred"}
        </div>
      </div>
    {/if}
  </AppPage>
</div>

<!-- Renders no inline UI of its own (just the settings dialog), so it lives
     outside the state branches: the review screen's generation-settings trigger
     needs to open it too, not only the create form's. -->
<!-- Seed the picker with the config this draft actually ran with (persisted on
     the job record). The page remounts between the setup step and the review
     step, so without this the user's settings would silently revert to the
     defaults on the review screen — and a retry there would use those defaults
     rather than what they picked. -->
<RunOptionsTiles
  bind:this={run_options_tiles}
  bind:selected_model_name_display={generation_model_name}
  bind:selected_provider_display={generation_provider}
  initial_run_config={captured_input_run_config}
  {project_id}
/>

<ExtractionDialog
  bind:dialog={extraction_dialog}
  bind:extracting={extraction_running}
  target_document_ids={extraction_document_ids}
  preselect_default_extractor={true}
  on:extraction_complete={handle_extraction_complete}
  on:extraction_failed={handle_extraction_failed}
  on:close={handle_extraction_dialog_closed}
/>
