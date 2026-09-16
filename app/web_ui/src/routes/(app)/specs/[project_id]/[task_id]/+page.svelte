<script lang="ts">
  import AppPage from "../../../app_page.svelte"
  import { page } from "$app/stores"
  import { createKilnError, KilnError } from "$lib/utils/error_handlers"
  import { client } from "$lib/api_client"
  import { onMount } from "svelte"
  import { get } from "svelte/store"
  import { indexedDBStore } from "$lib/stores/index_db_store"
  import {
    builder_draft_key,
    create_eval_button_label,
    create_eval_destination,
    draft_is_resumable,
    EMPTY_BUILDER_DRAFT,
  } from "./builder/builder_draft"
  import Intro from "$lib/ui/intro.svelte"
  import type { Spec, SpecStatus, Eval, Priority } from "$lib/types"
  import { goto, replaceState } from "$app/navigation"
  import Dialog from "$lib/ui/dialog.svelte"
  import FilterTagsDialog from "$lib/ui/filter_tags_dialog.svelte"
  import TableToolbar from "$lib/ui/table_toolbar.svelte"
  import AddTagsDialog from "$lib/ui/add_tags_dialog.svelte"
  import RemoveTagsDialog from "$lib/ui/remove_tags_dialog.svelte"
  import { capitalize, formatDate, formatPriority } from "$lib/utils/formatters"
  import { eval_type_display } from "$lib/utils/eval_types/eval_type_display"
  import type { OptionGroup } from "$lib/ui/fancy_select_types"
  import EditablePriorityField from "./editable_priority_field.svelte"
  import EditableStatusField from "./editable_status_field.svelte"
  import {
    updateEvalPriority as updateEvalPriorityUtil,
    updateEvalStatus as updateEvalStatusUtil,
  } from "./spec_utils"
  import {
    compute_table,
    resolved_status,
    type SortableColumn,
    type TableRow,
  } from "./spec_table"
  import { checkKilnCopilotAvailable } from "$lib/utils/copilot_utils"
  import EvalIcon from "$lib/ui/icons/eval_icon.svelte"
  import InfoTooltip from "$lib/ui/info_tooltip.svelte"
  import Banner from "$lib/ui/banner.svelte"
  import posthog from "posthog-js"
  import { agentInfo } from "$lib/agent"

  // ### Spec Table ###

  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  $: agentInfo.set({
    name: "Evals",
    description: `Evals list for project ID ${project_id}, task ID ${task_id}. Shows all evals, their requirements, and test cases.`,
  })

  let specs: Spec[] | null = null
  let specs_error: KilnError | null = null
  let specs_loading = true
  let evals: Eval[] | null = null
  let evals_error: KilnError | null = null
  let evals_loading = true
  let eval_load_error_count = 0

  let settings_loading = true
  let settings_error: KilnError | null = null
  let has_kiln_copilot = false

  // The draft peek is in the page's loading gate, not outside it: its answer
  // is what the create button is LABELLED, so a button rendered before it
  // lands says "Create Eval" and then rewrites itself a few ms later. Held
  // with the rest, the button and the body arrive together and the label is
  // right the first time it is painted.
  let draft_loading = true

  $: loading =
    specs_loading || evals_loading || settings_loading || draft_loading
  $: error = specs_error || evals_error || settings_error

  // Eval lookup for spec rows; priority/status resolution lives in spec_table.ts.
  $: evals_by_id = new Map((evals || []).map((e) => [e.id ?? "", e]))

  // Default judge type per eval (for the Type column). Loads after the table;
  // rows fall back to spec/template-derived display until it arrives.
  let judge_types: Map<string, string> = new Map()

  async function load_judge_types(req_project_id: string, req_task_id: string) {
    try {
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/eval_default_judge_types",
        {
          params: {
            path: { project_id: req_project_id, task_id: req_task_id },
          },
        },
      )
      if (req_project_id !== project_id || req_task_id !== task_id) return
      if (error) {
        throw error
      }
      judge_types = new Map(Object.entries(data || {}))
    } catch (error) {
      // Non-fatal: the Type column degrades to spec/template-derived values.
      console.warn("Failed to load eval judge types:", error)
    }
  }

  let sortColumn: "name" | "type" | "priority" | "status" | "created_at" =
    "created_at"
  let sortDirection: "asc" | "desc" = "desc"
  let filter_tags = ($page.url.searchParams.getAll("tags") || []) as string[]
  let filtered_specs: Spec[] | null = null
  let sorted_specs: TableRow[] | null = null
  let tags_dialog: Dialog | null = null
  let selected_spec_tags: string[] = []
  let filter_tags_dialog: FilterTagsDialog | null = null

  let select_mode: boolean = false
  let selected_specs: Set<string> = new Set()
  let select_summary: "all" | "none" | "some" = "none"
  $: {
    if (selected_specs.size >= (filtered_specs?.length || 0)) {
      select_summary = "all"
    } else if (selected_specs.size > 0) {
      select_summary = "some"
    } else {
      select_summary = "none"
    }
  }

  $: archive_action_state = (() => {
    if (selected_specs.size === 0) return null
    const selected_spec_objects = (filtered_specs || []).filter(
      (spec) => spec.id && selected_specs.has(spec.id),
    )
    if (selected_spec_objects.length === 0) return null

    const all_archived = selected_spec_objects.every(
      (spec) => resolved_status(spec, evals_by_id) === "archived",
    )
    const all_unarchived = selected_spec_objects.every(
      (spec) => resolved_status(spec, evals_by_id) !== "archived",
    )

    if (all_archived) return "unarchive"
    if (all_unarchived) return "archive"
    return "mixed"
  })()

  let add_tags: string[] = []
  let remove_tags: Set<string> = new Set()
  let add_tags_dialog: AddTagsDialog | null = null
  let remove_tags_dialog: RemoveTagsDialog | null = null
  let removeable_tags: Record<string, number> = {}
  let show_archived = false

  type TableColumn = {
    key: string
    label: string
    sortable: boolean
    sortKey?: SortableColumn
  }
  const tableColumns: TableColumn[] = [
    { key: "name", label: "Name", sortable: true, sortKey: "name" },
    { key: "type", label: "Type", sortable: true, sortKey: "type" },
    { key: "priority", label: "Priority", sortable: true, sortKey: "priority" },
    { key: "status", label: "Status", sortable: true, sortKey: "status" },
    { key: "tags", label: "Tags", sortable: false },
    {
      key: "created_at",
      label: "Created At",
      sortable: true,
      sortKey: "created_at",
    },
  ]

  $: {
    const url = new URL(window.location.href)
    filter_tags = url.searchParams.getAll("tags") as string[]
  }

  $: is_empty = (!specs || specs.length === 0) && (!evals || evals.length === 0)
  $: has_archived_specs =
    (specs || []).some(
      (spec) => resolved_status(spec, evals_by_id) === "archived",
    ) || (evals || []).some((e) => e.status === "archived")

  // When everything is archived, an empty-looking list would read as "no
  // evals" -- show the archived rows instead. One-way: never flips back off.
  $: if (
    !loading &&
    specs &&
    specs.length > 0 &&
    specs.every((spec) => resolved_status(spec, evals_by_id) === "archived")
  ) {
    show_archived = true
  }

  $: if (project_id && task_id) {
    load_specs(project_id, task_id)
    load_evals(project_id, task_id)
    load_judge_types(project_id, task_id)
    // Per-task, so it belongs with the loads keyed on the task rather than in
    // onMount: navigating between two tasks keeps this component mounted, and
    // a once-only check would advertise the previous task's draft.
    check_eval_draft(project_id, task_id)
  }

  // Bounded, for the same reason checkKilnCopilotAvailable is: the page's
  // spinner waits on this peek, so one that never settles must give up rather
  // than hold the whole page forever. IndexedDB really can hang — index_db_store
  // resolves its open() on success and error but not on `blocked`, which fires
  // instead of either while another tab holds the database at a different
  // version. The draft is a label on one button, so timing out and reading
  // "Create Eval" is a far better answer than a spinner that never stops.
  const DRAFT_CHECK_TIMEOUT_MS = 2000

  // Rejects once the deadline passes. The timer is cleared whichever side
  // wins, so a peek that lands first leaves nothing pending behind it.
  function within_draft_deadline(peek: Promise<unknown>): Promise<unknown> {
    let timer: ReturnType<typeof setTimeout> | undefined
    const deadline = new Promise((_, reject) => {
      timer = setTimeout(
        () => reject(new Error("draft check timed out")),
        DRAFT_CHECK_TIMEOUT_MS,
      )
    })
    return Promise.race([peek, deadline]).finally(() => clearTimeout(timer))
  }

  // Whether the v2 builder has a resumable draft for this task — the
  // create button advertises it ("Continue Eval Draft"). Read-only peek at
  // the draft store; the builder owns all writes. IndexedDB has no
  // synchronous read, so this is held behind the page's spinner rather than
  // allowed to land late and relabel a button already on screen.
  let has_eval_draft = false
  async function check_eval_draft(req_project_id: string, req_task_id: string) {
    try {
      draft_loading = true
      const { store, initialized } = indexedDBStore(
        builder_draft_key(req_project_id, req_task_id),
        EMPTY_BUILDER_DRAFT,
      )
      await within_draft_deadline(initialized)
      // A slower earlier task must not overwrite the task now on screen.
      if (req_project_id !== project_id || req_task_id !== task_id) return
      has_eval_draft = draft_is_resumable(get(store))
    } catch {
      // No draft signal is ever worth an error surface here — a failed or
      // timed-out peek means "no draft to continue", which is what the button
      // already says by default.
      if (req_project_id !== project_id || req_task_id !== task_id) return
      has_eval_draft = false
    } finally {
      if (req_project_id === project_id && req_task_id === task_id) {
        draft_loading = false
      }
    }
  }
  $: create_eval_label = create_eval_button_label(
    has_kiln_copilot,
    has_eval_draft,
  )

  // Not per-task, so it stays a once-only load.
  onMount(async () => {
    await load_has_kiln_copilot()
  })

  async function load_has_kiln_copilot() {
    try {
      settings_loading = true
      settings_error = null
      has_kiln_copilot = await checkKilnCopilotAvailable()
    } catch (e) {
      settings_error = createKilnError(e)
    } finally {
      settings_loading = false
    }
  }

  async function load_specs(req_project_id: string, req_task_id: string) {
    try {
      specs_loading = true
      specs_error = null
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/specs",
        {
          params: {
            path: { project_id: req_project_id, task_id: req_task_id },
          },
        },
      )
      if (req_project_id !== project_id || req_task_id !== task_id) return
      if (error) {
        throw error
      }
      specs = data
    } catch (error) {
      if (req_project_id !== project_id || req_task_id !== task_id) return
      specs_error = createKilnError(error)
    } finally {
      if (req_project_id === project_id && req_task_id === task_id) {
        specs_loading = false
      }
    }
  }

  async function load_evals(req_project_id: string, req_task_id: string) {
    try {
      evals_loading = true
      evals_error = null
      eval_load_error_count = 0
      const { data, error } = await client.GET(
        "/api/projects/{project_id}/tasks/{task_id}/evals",
        {
          params: {
            path: { project_id: req_project_id, task_id: req_task_id },
          },
        },
      )
      if (req_project_id !== project_id || req_task_id !== task_id) return
      if (error) {
        throw error
      }
      evals = data.evals
      eval_load_error_count = data.load_error_count
    } catch (error) {
      if (req_project_id !== project_id || req_task_id !== task_id) return
      evals_error = createKilnError(error)
    } finally {
      if (req_project_id === project_id && req_task_id === task_id) {
        evals_loading = false
      }
    }
  }

  // The table is a pure derivation of its inputs. This must not be an
  // imperative "recompute" call: reactive values like evals_by_id only flush
  // at the end of the task, so a synchronous `evals = ...; recompute()` would
  // read the previous map and partition rows by stale statuses.
  $: ({ filtered: filtered_specs, rows: sorted_specs } = compute_table(
    specs,
    evals,
    evals_by_id,
    judge_types,
    show_archived,
    filter_tags,
    sortColumn,
    sortDirection,
  ))

  function handleSort(column: SortableColumn) {
    let newDirection: "asc" | "desc" = "desc"
    if (sortColumn === column) {
      newDirection = sortDirection === "asc" ? "desc" : "asc"
    }
    sortColumn = column
    sortDirection = newDirection
  }

  function handleColumnClick(sortKey?: string) {
    if (sortKey) {
      handleSort(sortKey as SortableColumn)
    }
  }

  function remove_filter_tag(tag: string) {
    const newTags = filter_tags.filter((t) => t !== tag)
    updateURL({ tags: newTags })
  }

  function add_filter_tag(tag: string) {
    const newTags = [...new Set([...filter_tags, tag])]
    updateURL({ tags: newTags })
  }

  function updateURL(params: Record<string, string | string[]>) {
    const url = new URL(window.location.href)

    if (params.tags) {
      url.searchParams.delete("tags")
    }

    Object.entries(params).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        value.forEach((v) => url.searchParams.append(key, v))
      } else {
        url.searchParams.set(key, value.toString())
      }
    })

    if (params.tags) {
      filter_tags = params.tags as string[]
    }

    replaceState(url, {})
  }

  $: available_filter_tags = get_available_filter_tags(
    filtered_specs,
    filter_tags,
  )

  function get_available_filter_tags(
    filtered_specs: Spec[] | null,
    filter_tags: string[],
  ): Record<string, number> {
    if (!filtered_specs) return {}

    const remaining_tags: Record<string, number> = {}
    filtered_specs.forEach((spec) => {
      spec.tags?.forEach((tag) => {
        if (filter_tags.includes(tag)) return
        if (typeof tag === "string") {
          remaining_tags[tag] = (remaining_tags[tag] || 0) + 1
        }
      })
    })
    return remaining_tags
  }

  function formatTagsDisplay(tags: string[]): {
    firstTag: string
    othersCount: number
  } {
    if (tags.length === 0) {
      return { firstTag: "", othersCount: 0 }
    }
    const sortedTags = [...tags].sort()
    return {
      firstTag: sortedTags[0],
      othersCount: sortedTags.length - 1,
    }
  }

  function showTagsDialog(tags: string[], event: Event) {
    event.stopPropagation()
    selected_spec_tags = [...tags].sort()
    tags_dialog?.show()
  }

  function toggle_selection(spec_id: string): boolean {
    const was_selected = selected_specs.has(spec_id)
    if (was_selected) {
      selected_specs.delete(spec_id)
    } else {
      selected_specs.add(spec_id)
    }
    selected_specs = selected_specs
    return !was_selected
  }

  function select_all_clicked(event: Event) {
    event.preventDefault()
    if (select_summary === "all" || select_summary === "some") {
      selected_specs.clear()
    } else {
      filtered_specs?.forEach((spec) => {
        if (spec.id) {
          selected_specs.add(spec.id)
        }
      })
    }
    selected_specs = selected_specs
  }

  function show_add_tags_modal() {
    add_tags_dialog?.show()
  }

  function show_remove_tags_modal() {
    remove_tags = new Set()
    update_removeable_tags()
    remove_tags_dialog?.show()
  }

  function update_removeable_tags() {
    let selected_spec_contents: Spec[] = []
    for (const spec of filtered_specs || []) {
      if (spec.id && selected_specs.has(spec.id)) {
        selected_spec_contents.push(spec)
      }
    }
    removeable_tags = get_available_filter_tags(
      selected_spec_contents,
      Array.from(remove_tags),
    )
  }

  async function add_selected_tags(): Promise<boolean> {
    remove_tags = new Set()
    return await edit_tags()
  }

  async function remove_selected_tags(): Promise<boolean> {
    add_tags = []
    return await edit_tags()
  }

  async function edit_tags(): Promise<boolean> {
    let success = false
    try {
      const spec_ids = Array.from(selected_specs)
      const specs_to_update = (filtered_specs || []).filter(
        (spec) => spec.id && spec_ids.includes(spec.id),
      )

      for (const spec of specs_to_update) {
        if (!spec.id) continue

        const current_tags = spec.tags || []
        let updated_tags = [...current_tags]

        if (remove_tags.size > 0) {
          updated_tags = updated_tags.filter((tag) => !remove_tags.has(tag))
        }

        if (add_tags.length > 0) {
          updated_tags = [...new Set([...updated_tags, ...add_tags])]
        }

        const { error } = await client.PATCH(
          "/api/projects/{project_id}/tasks/{task_id}/specs/{spec_id}",
          {
            params: {
              path: { project_id, task_id, spec_id: spec.id },
            },
            body: {
              tags: updated_tags,
            },
          },
        )

        if (error) {
          throw error
        }
      }

      posthog.capture("update_spec_tags_batch", {
        num_specs: specs_to_update.length,
        num_tags_added: add_tags.length,
        num_tags_removed: remove_tags.size,
      })

      add_tags = []
      success = true
      return true
    } finally {
      // Only clear selection and reload if operation succeeded
      // If error occurred, Dialog will stay open and show error, keeping selection for retry
      if (success) {
        selected_specs = new Set()
        add_tags = []
        select_mode = false
        await load_specs(project_id, task_id)
      }
    }
  }

  function handle_tags_changed(tags: string[]) {
    add_tags = tags
  }

  function handle_remove_tag(tag: string) {
    remove_tags.delete(tag)
    remove_tags = remove_tags
    update_removeable_tags()
  }

  function handle_add_tag_to_remove(tag: string) {
    remove_tags.add(tag)
    remove_tags = remove_tags
    update_removeable_tags()
  }

  function show_archive_modal() {
    archive_dialog?.show()
  }

  async function archive_selected_specs(): Promise<boolean> {
    const spec_ids = Array.from(selected_specs)
    const specs_to_update = (filtered_specs || []).filter(
      (spec) => spec.id && spec_ids.includes(spec.id),
    )

    const should_archive = archive_action_state === "archive"
    const should_unarchive = archive_action_state === "unarchive"

    if (!should_archive && !should_unarchive) {
      return false
    }

    // Status lives on the eval; a spec without one can't be updated.
    const updatable: { spec: Spec; evaluator: Eval }[] = []
    const failed_names: string[] = []
    const failed_spec_ids: string[] = []
    for (const spec of specs_to_update) {
      const evaluator = spec.eval_id ? evals_by_id.get(spec.eval_id) : null
      if (evaluator) {
        updatable.push({ spec, evaluator })
      } else {
        failed_names.push(spec.name)
        if (spec.id) failed_spec_ids.push(spec.id)
      }
    }

    let succeeded = 0
    for (const { spec, evaluator } of updatable) {
      const new_status = should_archive ? "archived" : "active"
      const updated = await updateEvalStatus(
        evaluator,
        new_status as SpecStatus,
      )
      if (updated) {
        succeeded++
      } else {
        failed_names.push(spec.name)
        if (spec.id) failed_spec_ids.push(spec.id)
      }
    }

    if (succeeded > 0) {
      posthog.capture(should_archive ? "archive_specs" : "unarchive_specs", {
        num_specs: succeeded,
      })
      await load_specs(project_id, task_id)
    }

    if (failed_names.length > 0) {
      // Partial (or total) failure: report it and keep the failed specs
      // selected so the user can retry. The dialog stays open.
      evals_error = new KilnError(
        `Could not update ${failed_names.join(", ")}. The remaining selection can be retried.`,
      )
      selected_specs = new Set(failed_spec_ids)
      return false
    }

    selected_specs = new Set()
    select_mode = false
    return true
  }

  let archive_dialog: Dialog | null = null
  let updating_priorities: Set<string> = new Set()
  let updating_statuses: Set<string> = new Set()

  function getPriorityOptions(): OptionGroup[] {
    return [
      {
        options: [
          { label: "P0", value: 0 },
          { label: "P1", value: 1 },
          { label: "P2", value: 2 },
          { label: "P3", value: 3 },
        ],
      },
    ]
  }

  function getStatusOptions(): OptionGroup[] {
    return [
      {
        options: [
          { label: "Active", value: "active" },
          { label: "Future", value: "future" },
          { label: "Deprecated", value: "deprecated" },
          { label: "Archived", value: "archived" },
        ],
      },
    ]
  }

  async function updateEvalPriority(evaluator: Eval, newPriority: number) {
    if (
      !evaluator.id ||
      evaluator.priority === newPriority ||
      updating_priorities.has(evaluator.id)
    ) {
      return
    }

    updating_priorities.add(evaluator.id)
    try {
      const data = await updateEvalPriorityUtil(
        project_id,
        task_id,
        evaluator,
        newPriority,
      )

      if (data && evals) {
        const index = evals.findIndex((e) => e.id === evaluator.id)
        if (index !== -1) {
          evals[index] = data
          evals = evals
        }
      }
    } catch (error) {
      evals_error = createKilnError(error)
    } finally {
      updating_priorities.delete(evaluator.id)
    }
  }

  // Returns whether the update actually happened, so bulk callers can report
  // partial failures instead of assuming success.
  async function updateEvalStatus(
    evaluator: Eval,
    newStatus: SpecStatus,
  ): Promise<boolean> {
    if (!evaluator.id || updating_statuses.has(evaluator.id)) {
      return false
    }
    if (evaluator.status === newStatus) {
      return true
    }

    updating_statuses.add(evaluator.id)
    try {
      const data = await updateEvalStatusUtil(
        project_id,
        task_id,
        evaluator,
        newStatus,
      )

      if (data && evals) {
        const index = evals.findIndex((e) => e.id === evaluator.id)
        if (index !== -1) {
          evals[index] = data
          evals = evals
        }
      }
      return true
    } catch (error) {
      evals_error = createKilnError(error)
      return false
    } finally {
      updating_statuses.delete(evaluator.id)
    }
  }

  function handlePriorityUpdate(evaluator: Eval, value: Priority) {
    updateEvalPriority(evaluator, value)
  }

  function handleStatusUpdate(evaluator: Eval, value: SpecStatus) {
    updateEvalStatus(evaluator, value)
  }

  function create_eval() {
    posthog.capture("eval_v2_cta_clicked", {
      branch: has_kiln_copilot ? "v2" : "v1_manual",
      has_pro: has_kiln_copilot,
    })
    // PREVIEW (09-03): the entry restructure. Both user types start on the
    // eval-type page now. With Copilot it leads with the free-text box and
    // folds the templates behind it; without, the templates are the choice.
    // One page either way, so the eval type and the description are settled
    // together instead of across two screens. A draft in progress skips it:
    // the builder restores the draft on entry, which is what the button
    // promised.
    const destination = create_eval_destination(
      has_kiln_copilot,
      has_eval_draft,
    )
    goto(`/specs/${project_id}/${task_id}/${destination}`)
  }
</script>

<AppPage
  limit_max_width={true}
  title="Evals"
  subtitle="Define the behaviours to enforce or avoid for your task, and automatically measure quality."
  sub_subtitle={"Read the Docs"}
  sub_subtitle_link="https://docs.kiln.tech/docs/evals-and-specs"
  action_buttons={loading || is_empty
    ? []
    : [
        {
          label: create_eval_label,
          handler: async () => {
            create_eval()
          },
          primary: true,
        },
      ]}
>
  <div class="flex flex-col gap-4">
    {#if !loading && !error && eval_load_error_count > 0}
      <div class="text-error text-sm">
        {eval_load_error_count === 1
          ? "1 eval failed to load"
          : `${eval_load_error_count} evals failed to load`}
        <InfoTooltip
          tooltip_text="You may need to update Kiln. Some evals could not be opened by this version of Kiln."
          no_pad={true}
        />
      </div>
    {/if}
    {#if loading}
      <div class="flex justify-center items-center h-full">
        <div class="loading loading-spinner loading-lg"></div>
      </div>
    {:else if error}
      <div class="text-error text-sm">
        {error?.getMessage() || "An unknown error occurred"}
      </div>
    {:else if is_empty}
      <div class="mx-auto mt-[10vh]">
        <Intro
          title="Evals Ensure AI Quality"
          align_title_left={true}
          description_paragraphs={[
            "Specify how your AI task should behave, then use evaluations to verify performance.",
          ]}
          action_buttons={[
            {
              label: create_eval_label,
              onClick: async () => {
                create_eval()
              },
              is_primary: true,
            },
            {
              label: "Docs & Guide",
              href: "https://docs.kiln.tech/docs/evals-and-specs",
              is_primary: false,
              new_tab: true,
            },
          ]}
        >
          <div slot="icon" class="h-12 w-12">
            <EvalIcon />
          </div>
        </Intro>
      </div>
    {:else if sorted_specs}
      <Banner
        href={`/specs/${project_id}/${task_id}/compare`}
        title="Compare Models, Prompts, Tools and Fine-Tunes"
        description="Find the best way to run this task by comparing models, prompts, tools and fine-tunes using evals, cost and performance."
        button_label="Compare Run Configurations"
      >
        <div slot="icon" class="rounded-lg bg-blue-50 p-4">
          <svg
            class="h-12 aspect-760/621"
            viewBox="0 0 760 621"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <g clip-path="url(#clip0_1603_4)">
              <rect
                x="10"
                y="10"
                width="740"
                height="601"
                rx="25"
                fill="white"
                stroke="#628BD9"
                stroke-width="20"
              />
              <line
                x1="137"
                y1="90.9778"
                x2="137.999"
                y2="541.978"
                stroke="#628BD9"
                stroke-width="20"
              />
              <line
                x1="656"
                y1="490"
                x2="82"
                y2="490"
                stroke="#628BD9"
                stroke-width="20"
              />
              <circle cx="352" cy="241" r="28" fill="#628BD9" />
              <circle cx="473" cy="317" r="28" fill="#628BD9" />
              <circle cx="564" cy="153" r="28" fill="#628BD9" />
              <circle cx="232" cy="384" r="28" fill="#628BD9" />
            </g>
            <defs>
              <clipPath id="clip0_1603_4">
                <rect width="760" height="621" fill="white" />
              </clipPath>
            </defs>
          </svg>
        </div>
      </Banner>

      <div class="mb-4">
        <div class="-mb-4">
          <TableToolbar
            bind:select_mode
            selected_count={selected_specs.size}
            filter_tags_count={filter_tags.length}
            onToggleSelectMode={() => (select_mode = true)}
            onCancelSelection={() => {
              select_mode = false
              selected_specs = new Set()
            }}
            onShowFilterDialog={() => filter_tags_dialog?.show()}
            onShowArchived={has_archived_specs
              ? () => {
                  show_archived = !show_archived
                }
              : undefined}
            {show_archived}
            onShowAddTags={show_add_tags_modal}
            onShowRemoveTags={show_remove_tags_modal}
            onShowDelete={archive_action_state === "archive" ||
            archive_action_state === "unarchive"
              ? show_archive_modal
              : undefined}
            action_type="archive"
          />
        </div>
        <div class="overflow-x-auto rounded-lg border">
          <table class="table">
            <thead>
              <tr>
                {#if select_mode}
                  <th>
                    {#key select_summary}
                      <input
                        type="checkbox"
                        class="checkbox checkbox-sm mt-1"
                        checked={select_summary === "all"}
                        indeterminate={select_summary === "some"}
                        on:change={(e) => select_all_clicked(e)}
                      />
                    {/key}
                  </th>
                {/if}
                {#each tableColumns as column}
                  {#if column.sortable && column.sortKey}
                    <th
                      on:click={() => handleColumnClick(column.sortKey)}
                      class="hover:bg-base-200 cursor-pointer"
                    >
                      {column.label}
                      <span class="inline-block w-3 text-center">
                        {sortColumn === column.sortKey
                          ? sortDirection === "asc"
                            ? "▲"
                            : "▼"
                          : "\u200B"}
                      </span>
                    </th>
                  {:else}
                    <th>
                      {column.label}
                    </th>
                  {/if}
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each sorted_specs || [] as row}
                {#if row.type === "spec"}
                  {@const spec = row.data}
                  {@const spec_eval = spec.eval_id
                    ? evals_by_id.get(spec.eval_id)
                    : null}
                  <tr
                    class="{select_mode
                      ? ''
                      : 'hover'} cursor-pointer {select_mode &&
                    spec.id &&
                    selected_specs.has(spec.id)
                      ? 'bg-base-200'
                      : ''} {resolved_status(spec, evals_by_id) === 'archived'
                      ? 'text-base-content/60'
                      : ''}"
                    on:click={() => {
                      if (select_mode) {
                        toggle_selection(spec.id || "")
                      } else {
                        goto(`/specs/${project_id}/${task_id}/${spec.id}`)
                      }
                    }}
                  >
                    {#if select_mode}
                      <td>
                        <input
                          type="checkbox"
                          class="checkbox checkbox-sm"
                          checked={(spec.id && selected_specs.has(spec.id)) ||
                            false}
                        />
                      </td>
                    {/if}
                    <td class="font-medium">{spec.name}</td>
                    <td>
                      {eval_type_display(
                        spec,
                        spec_eval,
                        spec.eval_id ? judge_types.get(spec.eval_id) : null,
                      )}
                    </td>
                    <td>
                      {#if spec_eval}
                        <EditablePriorityField
                          evaluator={spec_eval}
                          options={getPriorityOptions()}
                          aria_label="Priority"
                          onUpdate={handlePriorityUpdate}
                        />
                      {:else}
                        <span class="px-2">{formatPriority(spec.priority)}</span
                        >
                      {/if}
                    </td>
                    <td>
                      {#if spec_eval}
                        <EditableStatusField
                          evaluator={spec_eval}
                          options={getStatusOptions()}
                          aria_label="Status"
                          onUpdate={handleStatusUpdate}
                        />
                      {:else}
                        <span class="px-2">{capitalize(spec.status)}</span>
                      {/if}
                    </td>
                    <td>
                      {#if spec.tags && spec.tags.length > 0}
                        {@const tagDisplay = formatTagsDisplay(spec.tags)}
                        <div
                          class="badge bg-gray-200 text-gray-500 py-3 px-3 max-w-full cursor-pointer hover:bg-gray-300"
                          on:click={(e) => showTagsDialog(spec.tags, e)}
                          role="button"
                          tabindex="0"
                          on:keydown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault()
                              showTagsDialog(spec.tags, e)
                            }
                          }}
                        >
                          <span class="truncate">{tagDisplay.firstTag}</span>
                          {#if tagDisplay.othersCount > 0}
                            <span class="ml-1 font-medium text-nowrap">
                              +{tagDisplay.othersCount} more
                            </span>
                          {/if}
                        </div>
                      {:else}
                        <span class="text-gray-500">None</span>
                      {/if}
                    </td>
                    <td class="text-sm text-gray-500">
                      {formatDate(spec.created_at)}
                    </td>
                  </tr>
                {:else if row.type === "legacy_eval"}
                  {@const eval_data = row.data}
                  <tr
                    class="{select_mode
                      ? ''
                      : 'hover'} cursor-pointer {eval_data.status === 'archived'
                      ? 'text-base-content/60'
                      : ''}"
                    on:click={() => {
                      if (!select_mode && eval_data.id) {
                        goto(
                          `/specs/${project_id}/${task_id}/legacy/${eval_data.id}`,
                        )
                      }
                    }}
                  >
                    {#if select_mode}
                      <td></td>
                    {/if}
                    <td class="font-medium">{eval_data.name}</td>
                    <td>
                      {eval_type_display(
                        null,
                        eval_data,
                        eval_data.id ? judge_types.get(eval_data.id) : null,
                      )}
                    </td>
                    <td>
                      <EditablePriorityField
                        evaluator={eval_data}
                        options={getPriorityOptions()}
                        aria_label="Priority"
                        onUpdate={handlePriorityUpdate}
                      />
                    </td>
                    <td>
                      <EditableStatusField
                        evaluator={eval_data}
                        options={getStatusOptions()}
                        aria_label="Status"
                        onUpdate={handleStatusUpdate}
                      />
                    </td>
                    <td class="text-gray-500">None</td>
                    <td class="text-sm text-gray-500">
                      {formatDate(eval_data.created_at)}
                    </td>
                  </tr>
                {/if}
              {/each}
            </tbody>
          </table>
        </div>
      </div>
    {/if}
  </div>
</AppPage>

<Dialog
  bind:this={tags_dialog}
  title="Tags"
  action_buttons={[
    {
      label: "Close",
      isCancel: true,
    },
  ]}
>
  <div class="flex flex-row flex-wrap gap-2">
    {#each selected_spec_tags as tag}
      <div class="badge bg-gray-200 text-gray-500 py-3 px-3 max-w-full">
        <span class="truncate">{tag}</span>
      </div>
    {/each}
  </div>
</Dialog>

<FilterTagsDialog
  bind:this={filter_tags_dialog}
  title="Filter Evals by Tags"
  {filter_tags}
  {available_filter_tags}
  onRemoveFilterTag={remove_filter_tag}
  onAddFilterTag={add_filter_tag}
/>

<AddTagsDialog
  bind:this={add_tags_dialog}
  title={selected_specs.size > 1
    ? "Add Tags to " + selected_specs.size + " Evals"
    : "Add Tags to Eval"}
  {project_id}
  {task_id}
  tag_type="task_run"
  bind:add_tags
  onTagsChanged={handle_tags_changed}
  onAddTags={add_selected_tags}
/>

<RemoveTagsDialog
  bind:this={remove_tags_dialog}
  title={selected_specs.size > 1
    ? "Remove Tags from " + selected_specs.size + " Evals"
    : "Remove Tags from Eval"}
  bind:remove_tags
  available_tags={removeable_tags}
  onRemoveTag={handle_remove_tag}
  onAddTagToRemove={handle_add_tag_to_remove}
  onRemoveTags={remove_selected_tags}
/>

<Dialog
  bind:this={archive_dialog}
  title={archive_action_state === "unarchive"
    ? selected_specs.size > 1
      ? `Unarchive ${selected_specs.size} Evals`
      : "Unarchive Eval"
    : selected_specs.size > 1
      ? `Archive ${selected_specs.size} Evals`
      : "Archive Eval"}
  action_buttons={[
    { label: "Cancel", isCancel: true },
    {
      label: archive_action_state === "unarchive" ? "Unarchive" : "Archive",
      asyncAction: archive_selected_specs,
      isError: true,
    },
  ]}
>
  <div class="mt-6">
    <p class="text-sm text-gray-500 mt-2">
      {archive_action_state === "unarchive"
        ? "Unarchived evals will be set back to an active state."
        : "Archived evals will be hidden from this list but can be restored later by unarchiving them."}
    </p>
  </div>
</Dialog>
