<script lang="ts">
  import AppPage from "../../../../app_page.svelte"
  import { page } from "$app/stores"
  import { get } from "svelte/store"
  import Dialog from "$lib/ui/dialog.svelte"
  import InfoTooltip from "$lib/ui/info_tooltip.svelte"
  import {
    get_splits_from_url_param,
    splits_equal,
  } from "$lib/utils/splits_util"

  import SelectDocumentsdialog from "./select_documents_dialog.svelte"
  import Extractiondialog from "$lib/components/extraction_dialog.svelte"
  import GenerateQnadialog from "./generate_qna_dialog.svelte"
  import EditSplitsDialog from "./edit_splits_dialog.svelte"
  import QnaDocumentNode from "./qna_document_node.svelte"
  import FileIcon from "$lib/ui/icons/file_icon.svelte"
  import QnaGenIntro from "./qna_gen_intro.svelte"
  import { DEFAULT_QNA_GUIDANCE } from "./guidance"
  import {
    createQnaStore,
    step_numbers,
    step_names,
    step_descriptions,
    type QnaStore,
    type QnADocPart,
  } from "./qna_ui_store"
  import Warning from "$lib/ui/warning.svelte"
  import CheckmarkIcon from "$lib/ui/icons/checkmark_icon.svelte"
  import FormContainer from "$lib/utils/form_container.svelte"
  import type { KilnDocument, RunConfigProperties } from "$lib/types"
  import posthog from "posthog-js"
  import { agentInfo } from "$lib/agent"

  let session_id = Math.floor(Math.random() * 1000000000000).toString()
  let ui_show_errors = false
  let ui_show_generation_errors = false

  $: project_id = $page.params.project_id!
  $: task_id = $page.params.task_id!
  $: agentInfo.set({
    name: "Q&A Data Generation",
    description: `Q&A data generation for project ID ${project_id}, task ID ${task_id}. Generate question-answer pairs from documents.`,
  })

  let qna: QnaStore
  $: qnaCurrentStep = qna?.currentStep
  $: qnaMaxStep = qna?.maxStep
  $: qnaAutoStep = qna?.autoStep
  $: qnaPendingSaveCount = qna?.pendingSaveCount
  $: qnaSaveAllStatus = qna?.saveAllStatus
  $: qnaExtractorId = qna?.extractorId
  $: qnaPairsPerPart = qna?.pairsPerPart
  $: qnaGuidance = qna?.guidance
  $: qnaTemplate = qna?.template

  let localGuidance: string = $qnaGuidance || ""
  let localTemplate: "custom" | "query_answer_generation" =
    $qnaTemplate || "query_answer_generation"
  let lastStoreGuidance: string = $qnaGuidance || ""
  let lastStoreTemplate: "custom" | "query_answer_generation" =
    $qnaTemplate || "query_answer_generation"

  $: {
    if ($qnaGuidance !== lastStoreGuidance) {
      localGuidance = $qnaGuidance || ""
      lastStoreGuidance = $qnaGuidance || ""
    }
    if ($qnaTemplate !== lastStoreTemplate) {
      localTemplate = $qnaTemplate || "query_answer_generation"
      lastStoreTemplate = $qnaTemplate || "query_answer_generation"
    }
  }

  $: {
    if (localGuidance !== lastStoreGuidance && qna) {
      qna.guidance.set(localGuidance)
      lastStoreGuidance = localGuidance
    }
    if (localTemplate !== lastStoreTemplate && qna) {
      qna.template.set(localTemplate)
      lastStoreTemplate = localTemplate
    }
  }
  $: qnaUseFullDocuments = qna?.useFullDocuments
  $: qnaChunkSizeTokens = qna?.chunkSizeTokens
  $: qnaChunkOverlapTokens = qna?.chunkOverlapTokens
  $: qnaTargetType = qna?.targetType
  $: qnaTargetDescription = qna?.targetDescription
  $: qnaGeneratedCount = qna?.generatedCount
  $: qnaTotalCount = qna?.totalCount
  $: qnaGenerationErrors = qna?.generationErrors
  $: qnaStatus = qna?.status
  $: qnaSelectedTags = qna?.selectedTags
  $: qnaExtractionErrorCount = qna?.extractionErrorCount

  let pendingUrlSplits: Record<string, number> = {}

  let last_initialized_key: string | null = null

  $: if (project_id && task_id) {
    const key = `${project_id}/${task_id}?${$page.url.searchParams.toString()}`
    if (last_initialized_key !== key) {
      last_initialized_key = key
      init_qna(project_id, task_id)
    }
  }

  async function init_qna(req_project_id: string, req_task_id: string) {
    const new_qna = createQnaStore(req_project_id, req_task_id)
    await new_qna.init(DEFAULT_QNA_GUIDANCE)
    if (req_project_id !== project_id || req_task_id !== task_id) return
    qna = new_qna

    // Check for splits in URL parameters (for non-reference answer evals)
    const splitsParam = $page.url.searchParams.get("splits")
    const urlSplits = splitsParam ? get_splits_from_url_param(splitsParam) : {}
    const savedSplits = get(qna).splits

    const hasDocuments = get(qna).documents.length > 0
    const splitsChanged = !splits_equal(urlSplits, savedSplits)
    if (hasDocuments && splitsChanged) {
      pendingUrlSplits = urlSplits
      clear_existing_state_dialog?.show()
    } else if (!hasDocuments && Object.keys(urlSplits).length > 0) {
      qna.setSplits(urlSplits)
    }
  }

  // Dialogs
  let clear_existing_state_dialog: Dialog | null = null
  let edit_splits_dialog: EditSplitsDialog | null = null
  let generating_dialog: Dialog | null = null
  let save_all_dialog: Dialog | null = null
  let rechunk_warning_dialog: Dialog | null = null

  function clear_state_and_reload() {
    qna.clearAll(DEFAULT_QNA_GUIDANCE)
    if (Object.keys(pendingUrlSplits).length > 0) {
      qna.setSplits(pendingUrlSplits)
      pendingUrlSplits = {}
    }
    return true
  }

  let current_dialog_type:
    | "select_documents"
    | "extraction"
    | "generate_qna"
    | "save_all"
    | null = null
  let show_select_documents_dialog: Dialog | null = null
  let show_extraction_dialog: Dialog | null = null
  let show_generate_qna_dialog: Dialog | null = null

  function open_select_documents_dialog() {
    current_dialog_type = "select_documents"
    show_select_documents_dialog?.show()
  }

  function open_extraction_dialog() {
    current_dialog_type = "extraction"
    show_extraction_dialog?.show()
  }

  function open_generate_qna_dialog() {
    current_dialog_type = "generate_qna"
    qna.setPendingTarget({ type: "all" })

    const has_existing_parts = $qna.documents.some((d) => d.parts.length > 0)
    if (has_existing_parts) {
      rechunk_warning_dialog?.show()
    } else {
      show_generate_qna_dialog?.show()
    }
  }

  function proceed_with_regeneration() {
    rechunk_warning_dialog?.close()
    show_generate_qna_dialog?.show()
  }

  function handle_documents_added(
    event: CustomEvent<{
      documents: KilnDocument[]
      tags: string[]
    }>,
  ) {
    const { documents, tags } = event.detail
    qna.addDocuments(documents, tags)

    posthog.capture("setup_qna_gen", {
      document_count: documents.length,
    })
  }

  function handle_extraction_complete(
    event: CustomEvent<{ extractor_config_id: string; error_count: number }>,
  ) {
    const { extractor_config_id, error_count } = event.detail
    qna.markExtractionComplete(extractor_config_id, error_count)
  }

  function handle_generate_for_document(
    e: CustomEvent<{ document_id: string }>,
  ) {
    qna.setPendingTarget({
      type: "document",
      document_id: e.detail.document_id,
    })
    current_dialog_type = "generate_qna"
    show_generate_qna_dialog?.show()
  }

  function handle_generate_for_part(
    e: CustomEvent<{ document_id: string; part_id: string }>,
  ) {
    qna.setPendingTarget({
      type: "part",
      document_id: e.detail.document_id,
      part_id: e.detail.part_id,
    })
    current_dialog_type = "generate_qna"
    show_generate_qna_dialog?.show()
  }

  async function handle_generate_requested(
    event: CustomEvent<{
      pairs_per_part: number
      guidance: string
      split_documents_into_chunks: boolean
      chunk_size_tokens: number | null
      chunk_overlap_tokens: number | null
      runConfigProperties: RunConfigProperties
    }>,
  ) {
    show_generate_qna_dialog?.close()
    current_dialog_type = null
    ui_show_generation_errors = false
    generating_dialog?.show()
    try {
      await qna.generate({
        pairsPerPart: event.detail.pairs_per_part,
        guidance: event.detail.guidance,
        useFullDocuments: !event.detail.split_documents_into_chunks,
        chunkSizeTokens: event.detail.chunk_size_tokens,
        chunkOverlapTokens: event.detail.chunk_overlap_tokens,
        runConfigProperties: event.detail.runConfigProperties,
      })
    } catch (e) {
      console.error("Q&A generation failed", e)
    } finally {
      if (
        !get(qna.generationErrors) ||
        get(qna.generationErrors).length === 0
      ) {
        generating_dialog?.close()
      }
    }
  }

  async function handle_extractor_config_selected(
    e: CustomEvent<{ extractor_config_id: string }>,
  ) {
    qna.setExtractor(e.detail.extractor_config_id)
  }

  function delete_document(event: CustomEvent) {
    const { document_id } = event.detail
    qna.deleteDocument(document_id)
  }

  function edit_splits() {
    edit_splits_dialog?.show()
  }

  $: total_qa_pairs = $qna
    ? $qna.documents.reduce(
        (total, doc) =>
          total +
          doc.parts.reduce(
            (partTotal: number, part: QnADocPart) =>
              partTotal + part.qa_pairs.length,
            0,
          ),
        0,
      )
    : 0

  $: has_documents = $qna ? $qna.documents.length > 0 : false
  $: is_empty = !has_documents

  function show_save_all_dialog() {
    current_dialog_type = "save_all"
    save_all_dialog?.show()
  }

  async function save_all_qna_pairs() {
    await qna.saveAll(session_id)

    const state = get(qna)
    const firstPair = state.documents
      .flatMap((d) => d.parts)
      .flatMap((p) => p.qa_pairs)
      .find((p) => p.model_name && p.model_provider)
    if (firstPair) {
      posthog.capture("save_qna_data", {
        model_name: firstPair.model_name,
        provider: firstPair.model_provider,
        template: state.template,
        total_pairs: total_qa_pairs,
      })
    }
  }
</script>

<div class="max-w-[1400px]">
  <AppPage
    title="Synthetic Data Generation"
    no_y_padding
    sub_subtitle="Read the Docs"
    sub_subtitle_link="https://docs.kiln.tech/docs/evaluations/evaluate-rag-accuracy-q-and-a-evals"
    action_buttons={[
      {
        label: "Reset",
        handler: () => {
          if (
            confirm(
              "Are you sure you want to clear all Q&A generation state? This cannot be undone.",
            )
          ) {
            clear_state_and_reload()
          }
        },
      },
      {
        label: "Docs & Guide",
        href: "https://docs.kiln.tech/docs/evaluations/evaluate-rag-accuracy-q-and-a-evals",
      },
    ]}
  >
    <!-- Header with Goal, Template, Tags -->
    <div class="card flex-row border px-6 py-3 mt-3 mb-2 shadow-sm text-sm">
      <div class="flex flex-row items-center gap-3 flex-grow">
        <div>
          <svg
            class="h-6 w-6 text-gray-500"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M5 22V14M5 14V4M5 14L7.47067 13.5059C9.1212 13.1758 10.8321 13.3328 12.3949 13.958C14.0885 14.6354 15.9524 14.7619 17.722 14.3195L17.9364 14.2659C18.5615 14.1096 19 13.548 19 12.9037V5.53669C19 4.75613 18.2665 4.18339 17.5092 4.3727C15.878 4.78051 14.1597 4.66389 12.5986 4.03943L12.3949 3.95797C10.8321 3.33284 9.1212 3.17576 7.47067 3.50587L5 4M5 4V2"
              stroke="currentColor"
              stroke-width="1.5"
              stroke-linecap="round"
            />
          </svg>
        </div>
        <div class="flex flex-col">
          <div class="text-xs text-gray-500 uppercase font-medium">Goal</div>
          <div class="whitespace-nowrap">
            Evaluation
            <InfoTooltip
              tooltip_text="The goal of the data generation task. This impacts the type of data that will be generated."
              no_pad={true}
            />
          </div>
        </div>
      </div>
      <div
        class="border-l border-gray-200 self-stretch mx-4 border-[0.5px]"
      ></div>
      <div class="flex flex-row items-center gap-3 flex-grow">
        <div class="h-6 w-6 text-gray-500">
          <FileIcon kind="document" />
        </div>
        <div class="flex flex-col">
          <div class="text-xs text-gray-500 uppercase font-medium">
            Template
          </div>
          <div class="whitespace-nowrap">
            {$qnaTemplate || "query_answer_generation"}
            <InfoTooltip
              tooltip_text="A prompt template used to generate data. You can edit the template when generating query-answer pairs."
              no_pad={true}
            />
          </div>
        </div>
      </div>
      <div
        class="border-l border-gray-200 self-stretch mx-4 border-[0.5px]"
      ></div>
      <div class="flex flex-row items-center gap-3 flex-grow">
        <div>
          <svg
            class="h-6 w-6 text-gray-500"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M4.72848 16.1369C3.18295 14.5914 2.41018 13.8186 2.12264 12.816C1.83509 11.8134 2.08083 10.7485 2.57231 8.61875L2.85574 7.39057C3.26922 5.59881 3.47597 4.70292 4.08944 4.08944C4.70292 3.47597 5.59881 3.26922 7.39057 2.85574L8.61875 2.57231C10.7485 2.08083 11.8134 1.83509 12.816 2.12264C13.8186 2.41018 14.5914 3.18295 16.1369 4.72848L17.9665 6.55812C20.6555 9.24711 22 10.5916 22 12.2623C22 13.933 20.6555 15.2775 17.9665 17.9665C15.2775 20.6555 13.933 22 12.2623 22C10.5916 22 9.24711 20.6555 6.55812 17.9665L4.72848 16.1369Z"
              stroke="currentColor"
              stroke-width="1.5"
            />
            <circle
              cx="8.60724"
              cy="8.87891"
              r="2"
              transform="rotate(-45 8.60724 8.87891)"
              stroke="currentColor"
              stroke-width="1.5"
            />
          </svg>
        </div>
        <div class="flex flex-col">
          <div class="text-xs text-gray-500 uppercase font-medium">Tags</div>
          <div>
            <button class="hover:underline text-left" on:click={edit_splits}>
              {#if $qna && Object.keys($qna.splits).length == 0}
                No tag assignments
              {:else if $qna}
                {@const split_descriptions = Object.entries($qna.splits).map(
                  ([split, percent]) => `${split} (${percent * 100}%)`,
                )}
                {split_descriptions.join(", ")}
              {/if}
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Stepper -->
    {#if $qnaMaxStep && $qnaMaxStep > 1}
      <div
        class="pb-1 2xl:pt-1 mt-12 mb-4 gap-2 sticky top-0 z-2 backdrop-blur bg-white/70 z-10"
      >
        <div class="flex flex-col">
          <div class="flex justify-center">
            <ul class="steps">
              {#each step_numbers as step}
                {@const isDisabled = $qnaAutoStep && step > $qnaAutoStep}
                <li
                  class="step {$qnaCurrentStep && $qnaCurrentStep >= step
                    ? 'step-primary'
                    : ''}"
                >
                  <button
                    class="px-4 text-sm md:min-w-[155px] {$qnaCurrentStep &&
                    $qnaCurrentStep == step
                      ? 'font-medium cursor-default'
                      : isDisabled
                        ? 'text-gray-400'
                        : 'text-gray-500 hover:underline hover:text-gray-700'}"
                    disabled={isDisabled}
                    on:click={() => qna.setCurrentStep(step)}
                    aria-label={`Go to step ${step} - ${step_names[step]}`}
                  >
                    {step_names[step]}
                  </button>
                </li>
              {/each}
            </ul>
          </div>
          <div class="max-w-3xl mx-auto mt-2 2xl:mt-4 text-center">
            <div class="font-light">
              <span class="font-medium">Step {$qnaCurrentStep}:</span>
              {#if $qnaCurrentStep}
                {step_descriptions[$qnaCurrentStep]}
              {/if}
            </div>
            <div class="mt-1 2xl:mt-2">
              {#if $qnaCurrentStep == 1}
                <div class="flex justify-center">
                  {#if $qnaSelectedTags?.length > 0}
                    <Warning
                      warning_message={`Document Tag${$qnaSelectedTags?.length > 1 ? "s" : ""}: ${[
                        ...($qnaSelectedTags || []),
                      ]
                        .sort()
                        .join(", ")}`}
                      warning_icon="check"
                      warning_color="success"
                      inline={true}
                    />
                  {:else}
                    <Warning
                      warning_message="All Documents in Library"
                      warning_icon="check"
                      warning_color="success"
                      inline={true}
                    />
                  {/if}
                </div>
              {:else if $qnaCurrentStep == 2}
                <button
                  class="btn btn-sm {$qna &&
                  $qna.extraction_complete &&
                  $qnaExtractionErrorCount === 0
                    ? 'btn-outline btn-primary'
                    : 'btn-primary'}"
                  on:click={open_extraction_dialog}
                  disabled={!has_documents}
                >
                  {$qna &&
                  $qna.extraction_complete &&
                  $qnaExtractionErrorCount > 0
                    ? "Retry Extraction"
                    : "Run Extraction"}
                </button>
                {#if $qna && $qna.extraction_complete && $qnaExtractionErrorCount === 0}
                  <button
                    class="btn btn-sm btn-primary ml-2"
                    on:click={() => qna.setCurrentStep(3)}
                  >
                    Next Step
                  </button>
                {/if}
                {#if $qnaExtractionErrorCount && $qnaExtractionErrorCount > 0}
                  <div class="mt-3 flex flex-col items-center">
                    <Warning
                      warning_message="{$qnaExtractionErrorCount} document{$qnaExtractionErrorCount >
                      1
                        ? 's'
                        : ''} failed to extract. Retry or delete documents."
                      warning_color="error"
                      warning_icon="exclaim"
                      inline={true}
                    />
                  </div>
                {/if}
              {:else if $qnaCurrentStep == 3}
                <button
                  class="btn btn-sm {total_qa_pairs > 0
                    ? 'btn-outline btn-primary'
                    : 'btn-primary'}"
                  on:click={open_generate_qna_dialog}
                  disabled={$qna && !$qna.extraction_complete}
                >
                  Generate Q&A Pairs
                </button>
                {#if total_qa_pairs > 0}
                  <button
                    class="btn btn-sm btn-primary ml-2"
                    on:click={() => qna.setCurrentStep(4)}
                  >
                    Next Step
                  </button>
                {/if}
                {#if $qnaGenerationErrors && $qnaGenerationErrors.length > 0}
                  <div class="mt-3 flex flex-col items-center">
                    <Warning
                      warning_message="{$qnaGenerationErrors.length} error{$qnaGenerationErrors.length >
                      1
                        ? 's'
                        : ''} occurred during generation"
                      warning_color="error"
                      warning_icon="exclaim"
                      inline={true}
                    />
                    <button
                      class="link text-sm mt-1"
                      on:click={() => {
                        ui_show_generation_errors = true
                        generating_dialog?.show()
                      }}
                    >
                      Show Errors
                    </button>
                  </div>
                {/if}
              {:else if $qnaCurrentStep == 4}
                {@const already_saved_count =
                  total_qa_pairs - ($qnaPendingSaveCount || 0)}
                {@const has_pending = ($qnaPendingSaveCount || 0) > 0}
                {@const all_saved = already_saved_count > 0 && !has_pending}

                {#if has_pending}
                  <button
                    class="btn btn-sm btn-primary"
                    on:click={show_save_all_dialog}
                  >
                    Save All ({$qnaPendingSaveCount || 0})
                  </button>
                {:else if all_saved}
                  <div class="flex flex-row justify-center">
                    <Warning
                      warning_message="All items saved into the dataset!"
                      warning_color="success"
                      warning_icon="check"
                      inline={true}
                    />
                  </div>
                {:else}
                  <button class="btn btn-sm btn-disabled">Save All</button>
                {/if}
                {#if $qnaGenerationErrors && $qnaGenerationErrors.length > 0}
                  <div class="mt-3 flex flex-col items-center">
                    <Warning
                      warning_message="{$qnaGenerationErrors.length} error{$qnaGenerationErrors.length >
                      1
                        ? 's'
                        : ''} occurred during generation"
                      warning_color="error"
                      warning_icon="exclaim"
                      inline={true}
                    />
                    <button
                      class="link text-sm mt-1"
                      on:click={() => {
                        ui_show_generation_errors = true
                        generating_dialog?.show()
                      }}
                    >
                      Show Errors
                    </button>
                  </div>
                {/if}
              {/if}
            </div>
          </div>
        </div>
      </div>
    {/if}

    <!-- Empty State or Table Display -->
    {#if is_empty}
      <QnaGenIntro on_select_documents={open_select_documents_dialog} />
    {:else}
      <div class="rounded-lg border">
        <table class="table table-fixed">
          <thead class={total_qa_pairs === 0 ? "hidden-header" : ""}>
            <tr>
              <th style="width: calc(50% - 70px)"
                >Query <InfoTooltip
                  tooltip_text="The query to ask about the document content."
                  position="bottom"
                /></th
              >
              <th style="width: calc(50% - 110px)"
                >Answer <InfoTooltip
                  tooltip_text="The answer to the query based on the document content."
                  position="bottom"
                /></th
              >
              <th style="width: 140px">Status</th>
              <th style="width: 40px"></th>
            </tr>
          </thead>
          <tbody>
            {#if $qna}
              {#each $qna.documents as document}
                <QnaDocumentNode
                  {project_id}
                  {task_id}
                  {document}
                  {qna}
                  depth={1}
                  on:delete_document={delete_document}
                  on:generate_for_document={handle_generate_for_document}
                  on:generate_for_part={handle_generate_for_part}
                />
              {/each}
            {/if}
          </tbody>
        </table>
      </div>
    {/if}
  </AppPage>
</div>

{#if qna}
  <SelectDocumentsdialog
    bind:dialog={show_select_documents_dialog}
    {project_id}
    on:documents_added={handle_documents_added}
    on:close={() => (current_dialog_type = null)}
    keyboard_submit={current_dialog_type === "select_documents"}
  />

  <Extractiondialog
    target_tags={$qnaSelectedTags || []}
    keyboard_submit={current_dialog_type === "extraction"}
    bind:dialog={show_extraction_dialog}
    selected_extractor_id={$qnaExtractorId}
    on:extraction_complete={handle_extraction_complete}
    on:extractor_config_selected={handle_extractor_config_selected}
    on:close={() => (current_dialog_type = null)}
  />

  <GenerateQnadialog
    bind:dialog={show_generate_qna_dialog}
    {project_id}
    pairs_per_part={$qnaPairsPerPart}
    bind:guidance={localGuidance}
    bind:selected_template={localTemplate}
    split_documents_into_chunks={!$qnaUseFullDocuments}
    chunk_size_tokens={$qnaChunkSizeTokens}
    chunk_overlap_tokens={$qnaChunkOverlapTokens}
    target_description={$qnaTargetDescription || "all documents"}
    generation_target_type={$qnaTargetType || "all"}
    on:generate_requested={handle_generate_requested}
    on:close={() => (current_dialog_type = null)}
    keyboard_submit={current_dialog_type === "generate_qna"}
  />
{/if}

<Dialog
  bind:this={generating_dialog}
  title="Generating Q&A"
  width="normal"
  action_buttons={$qnaStatus && $qnaStatus !== "running"
    ? [
        {
          label: "Close",
          action: () => {
            generating_dialog?.close()
            return true
          },
          isPrimary: true,
        },
      ]
    : []}
>
  <div class="min-h-[200px] flex flex-col justify-center items-center">
    {#if $qnaStatus === "running"}
      <div class="loading loading-spinner loading-lg mb-6 text-success"></div>
    {/if}
    {#if $qnaTotalCount && $qnaTotalCount > 0}
      <progress
        class="progress w-56 progress-success"
        value={$qnaGeneratedCount}
        max={$qnaTotalCount}
      ></progress>
      <div class="font-light text-xs text-center mt-1">
        {$qnaGeneratedCount} of {$qnaTotalCount} generated
      </div>
      {#if $qnaGenerationErrors && $qnaGenerationErrors.length > 0}
        <div class="text-error font-light text-sm mt-4">
          {$qnaGenerationErrors.length} document{$qnaGenerationErrors.length > 1
            ? "s"
            : ""} or chunk{$qnaGenerationErrors.length > 1 ? "s" : ""} failed to
          generate
          <button
            class="link"
            on:click={() =>
              (ui_show_generation_errors = !ui_show_generation_errors)}
          >
            {ui_show_generation_errors ? "Hide Errors" : "Show Errors"}
          </button>
        </div>
        <div
          class="flex flex-col gap-2 mt-4 text-xs text-error max-w-md max-h-48 overflow-y-auto {ui_show_generation_errors
            ? ''
            : 'hidden'}"
        >
          {#each $qnaGenerationErrors as error}
            <div class="text-left border-l-2 border-error pl-2 py-1">
              {error.message}
            </div>
          {/each}
        </div>
      {/if}
    {/if}
  </div>
</Dialog>

<Dialog
  title="Save Q&A Pairs to Dataset"
  sub_subtitle="All the unsaved Q&A pairs will be saved to the dataset."
  bind:this={save_all_dialog}
>
  {#if $qnaSaveAllStatus}
    {#if $qnaSaveAllStatus.running}
      {@const totalToSave =
        $qnaSaveAllStatus.savedCount + ($qnaPendingSaveCount || 0)}
      <div class="min-h-[200px] flex flex-col justify-center items-center">
        <div class="loading loading-spinner loading-lg mb-6 text-success"></div>
        <progress
          class="progress w-56 progress-success"
          value={$qnaSaveAllStatus.savedCount}
          max={totalToSave}
        ></progress>
        <div class="font-light text-xs text-center mt-1">
          {$qnaSaveAllStatus.savedCount} of {totalToSave}
          {#if $qnaSaveAllStatus.errors.length > 0}
            complete — {$qnaSaveAllStatus.errors.length} failed
          {/if}
        </div>
      </div>
    {:else if $qnaSaveAllStatus.completed}
      <div
        class="text-center flex flex-col items-center justify-center min-h-[150px] p-12"
      >
        {#if $qnaSaveAllStatus.savedCount > 0}
          <div class="size-10 text-success mb-2">
            <CheckmarkIcon />
          </div>
        {/if}
        <div class="font-medium">
          Saved {$qnaSaveAllStatus.savedCount} new items.
        </div>
        <div class="font-light text-sm">
          These are now available in the <a
            href={`/dataset/${project_id}/${task_id}`}
            class="link">dataset tab</a
          >.
        </div>
        <div class="font-light text-xs mt-4 text-gray-500">
          All items are tagged with &quot;qna_session_{session_id}&quot;
        </div>
        {#if $qnaSaveAllStatus.errors.length > 0}
          <div class="text-error font-light text-sm mt-4">
            {$qnaSaveAllStatus.errors.length} samples failed to save. Running again
            may resolve transient issues.
            <button
              class="link"
              on:click={() => (ui_show_errors = !ui_show_errors)}
            >
              {ui_show_errors ? "Hide Errors" : "Show Errors"}
            </button>
          </div>
          <div
            class="flex flex-col gap-2 mt-4 text-xs text-error {ui_show_errors
              ? ''
              : 'hidden'}"
          >
            {#each $qnaSaveAllStatus.errors as error}
              <div>{error.message}</div>
            {/each}
          </div>
        {/if}
      </div>
    {:else if ($qnaPendingSaveCount || 0) === 0}
      {@const already_saved_count =
        total_qa_pairs - ($qnaPendingSaveCount || 0)}
      <div
        class="flex flex-col items-center justify-center min-h-[150px] gap-2"
      >
        <div class="font-medium">No Items to Save</div>
        <div class="font-light">
          Generate Q&A pairs before attempting to save.
        </div>
        {#if already_saved_count > 0}
          <div class="font-light text-sm">
            {already_saved_count} existing items already saved.
          </div>
        {/if}
      </div>
    {:else}
      {@const already_saved_count =
        total_qa_pairs - ($qnaPendingSaveCount || 0)}
      <FormContainer submit_label="Save All" on:submit={save_all_qna_pairs}>
        <div>
          <div class="font-medium text-sm">Status</div>
          <div class="font-light">
            {$qnaPendingSaveCount || 0} item{$qnaPendingSaveCount > 1
              ? "s"
              : ""} pending
            {#if already_saved_count > 0}
              / {already_saved_count} already saved
            {/if}
          </div>
        </div>
        <Warning
          warning_color="warning"
          warning_message="Please review all Q&A pairs carefully before saving to ensure their accuracy."
        />
      </FormContainer>
    {/if}
  {/if}
</Dialog>

<Dialog
  title="Existing Session"
  bind:this={clear_existing_state_dialog}
  action_buttons={[
    {
      label: "New Session",
      action: clear_state_and_reload,
    },
    {
      label: "Continue Session",
      isPrimary: true,
      action: () => {
        pendingUrlSplits = {}
        clear_existing_state_dialog?.close()
        return true
      },
    },
  ]}
>
  <div class="flex flex-col gap-2">
    <div class="font-light flex flex-col gap-2">
      <p>A synthetic data generation session is already in progress.</p>
    </div>
    <div class="flex flex-row gap-2"></div>
  </div></Dialog
>

{#if qna}
  <EditSplitsDialog bind:this={edit_splits_dialog} {qna} />
{/if}

<Dialog
  title="Generate Q&A Pairs"
  bind:this={rechunk_warning_dialog}
  action_buttons={[
    {
      label: "Cancel",
      action: () => {
        rechunk_warning_dialog?.close()
        return true
      },
    },
    {
      label: "Continue",
      action: () => {
        proceed_with_regeneration()
        return true
      },
      isPrimary: true,
    },
  ]}
>
  <div class="flex flex-col gap-3">
    <!-- A flex item never collapses margins with its child, so this wrapper
         carries the full 16px the dialog title had before. -->
    <div class="mt-4">
      <Warning
        large_icon={true}
        warning_icon="exclaim"
        warning_color="warning"
        warning_message="If you proceed, all existing Q&A pairs will be lost and any document chunks will be replaced. Consider generating Q&A for
        specific documents or chunks instead."
      />
    </div>
  </div>
</Dialog>

<style>
  .hidden-header {
    height: 0;
    overflow: hidden;
    visibility: hidden;
  }
  .hidden-header th {
    height: 0;
    padding: 0;
    border: 0;
    font-size: 0;
    line-height: 0;
    visibility: hidden !important;
    height: 0 !important;
  }
  .hidden-header th *,
  .hidden-header th * * {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
  }
</style>
