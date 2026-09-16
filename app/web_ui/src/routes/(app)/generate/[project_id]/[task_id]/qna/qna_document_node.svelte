<script lang="ts">
  import TableActionMenu from "$lib/ui/table_action_menu.svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import Output from "$lib/ui/output.svelte"
  import { createEventDispatcher } from "svelte"
  import type { QnaStore } from "./qna_ui_store"
  import Warning from "$lib/ui/warning.svelte"

  type QnAPair = {
    id: string
    query: string
    answer: string
    generated: boolean
    model_name?: string
    model_provider?: string
    saved_id: string | null
  }

  type QnADocPart = {
    id: string
    text_preview: string
    qa_pairs: QnAPair[]
  }

  type QnADocumentNode = {
    id: string
    name: string
    tags: string[]
    extracted: boolean
    extraction_failed: boolean
    parts: QnADocPart[]
  }

  export let project_id: string
  export let task_id: string

  export let document: QnADocumentNode
  export let qna: QnaStore
  export let depth: number = 1

  let expandedQAPairs: Record<string, boolean> = {}
  const dispatch = createEventDispatcher()

  $: qnaMaxStep = qna?.maxStep

  // we don't enable the part dialog until after the extraction step and the chunking step
  $: output_dialog_possible = $qnaMaxStep && $qnaMaxStep > 3

  // Dialogs
  let part_output_dialog: Dialog | null = null
  let selected_part_text: string | null = null
  let document_parts_dialog: Dialog | null = null
  let generate_document_warning_dialog: Dialog | null = null

  function toggleQAPairExpand(qaId: string) {
    expandedQAPairs[qaId] = !expandedQAPairs[qaId]
    expandedQAPairs = expandedQAPairs
  }

  function delete_qa_pair(part_id: string, qa_id: string) {
    qna.removePair(document.id, part_id, qa_id)
  }

  function remove_part(part_id: string) {
    qna.removePart(document.id, part_id)
  }

  $: has_parts = document.parts.length > 0

  function open_part_dialog(part_text: string) {
    selected_part_text = part_text
    part_output_dialog?.show()
  }

  function open_document_parts_dialog() {
    document_parts_dialog?.show()
  }

  function openQAPairInDataset(qa: QnAPair) {
    window.open(
      `/dataset/${project_id}/${task_id}/${qa.saved_id}/run`,
      "_blank",
    )
  }

  function handle_generate_for_document() {
    if (document.parts.length > 1) {
      generate_document_warning_dialog?.show()
    } else {
      proceed_with_generate_for_document()
    }
  }

  function proceed_with_generate_for_document() {
    dispatch("generate_for_document", { document_id: document.id })
  }
</script>

<!-- Document Header Row -->
<tr class="bg-base-200 border-base-100">
  <td colspan="3" class="py-2" style="padding-left: {(depth - 1) * 25 + 20}px">
    <div class="font-medium flex flex-row pr-4 w-full">
      <div class="flex items-center">
        {#if depth > 1}
          <span class="text-xs relative" style="top: -3px">⮑</span>
        {/if}
        <button
          on:click|stopPropagation={output_dialog_possible
            ? open_document_parts_dialog
            : null}
          class:cursor-pointer={output_dialog_possible}
          class:hover:underline={output_dialog_possible}
        >
          {document.name}
        </button>
        {#if document.extraction_failed}
          <div class="badge badge-sm badge-error badge-outline ml-2">
            Extraction Failed
          </div>
        {:else if document.extracted}
          <div class="badge badge-sm badge-secondary badge-outline ml-2">
            Extracted
          </div>
        {:else if $qnaMaxStep && $qnaMaxStep > 2}
          <span class="badge badge-sm badge-warning badge-outline ml-2"
            >Not Extracted</span
          >
        {/if}
      </div>
    </div>
  </td>
  <td class="p-0">
    <TableActionMenu
      items={[
        {
          label: "Remove Document",
          onclick: () =>
            dispatch("delete_document", { document_id: document.id }),
        },
        {
          label: "Generate Q&A Pairs",
          onclick: handle_generate_for_document,
          hidden: !(
            $qnaMaxStep &&
            $qnaMaxStep > 2 &&
            !document.extraction_failed
          ),
        },
      ]}
    />
  </td>
</tr>

{#if $qnaMaxStep && $qnaMaxStep > 2 && !has_parts}
  <tr>
    <td colspan="4" style="padding-left: {depth * 25 + 20}px" class="py-2">
      <div class="text-sm text-gray-500 italic">No Q&A pairs generated yet</div>
    </td>
  </tr>
{:else}
  {#each document.parts as part, partIndex}
    {#if document.parts.length > 1}
      <!-- Part Header Row (only show if document has multiple parts) -->
      <tr
        class="bg-base-200 border-t border-base-100"
        on:click={() =>
          output_dialog_possible ? open_part_dialog(part.text_preview) : null}
        class:cursor-pointer={output_dialog_possible}
        class:hover:underline={output_dialog_possible}
      >
        <td colspan="3" class="py-2" style="padding-left: {depth * 25 + 20}px">
          <div class="font-medium flex flex-row pr-4 w-full">
            <div class="flex-1">
              <span class="text-xs relative" style="top: -3px">⮑</span>
              Chunk {partIndex + 1}
            </div>
          </div>
        </td>
        <td class="p-0">
          <TableActionMenu
            items={[
              { label: "Remove Chunk", onclick: () => remove_part(part.id) },
              {
                label: "Generate Q&A Pairs for Chunk",
                onclick: () =>
                  dispatch("generate_for_part", {
                    document_id: document.id,
                    part_id: part.id,
                  }),
              },
            ]}
          />
        </td>
      </tr>
    {/if}

    {#each part.qa_pairs as qa}
      <tr on:click={() => toggleQAPairExpand(qa.id)} class="cursor-pointer">
        <td
          style="padding-left: {(depth + (document.parts.length > 1 ? 1 : 0)) *
            25 +
            20}px"
          class="py-2"
        >
          {#if expandedQAPairs[qa.id]}
            <pre class="whitespace-pre-wrap">{qa.query}</pre>
          {:else}
            <div class="truncate w-0 min-w-full">{qa.query}</div>
          {/if}
        </td>
        <td class="py-2">
          {#if !qa.answer}
            Not Generated
          {:else if expandedQAPairs[qa.id]}
            <pre class="whitespace-pre-wrap">{qa.answer}</pre>
          {:else}
            <div class="truncate w-0 min-w-full">{qa.answer}</div>
          {/if}
        </td>
        <td class="py-2">
          {#if qa.saved_id}
            <a href={`#`} class="hover:underline">Saved</a>
          {:else if qa.answer}
            Unsaved
          {:else}
            No Answer
          {/if}
        </td>
        <td class="p-0">
          <TableActionMenu
            items={[
              {
                label: "Delete Q&A Pair",
                onclick: () => delete_qa_pair(part.id, qa.id),
              },
              {
                label: "View in Dataset",
                onclick: () => openQAPairInDataset(qa),
                hidden: !qa.saved_id,
              },
            ]}
          />
        </td>
      </tr>
    {/each}
  {/each}
{/if}

<!-- Single Part Output Dialog -->
<Dialog
  bind:this={part_output_dialog}
  title="Part Content"
  width="wide"
  action_buttons={[{ label: "Close", isCancel: true }]}
>
  {#if selected_part_text}
    <div class="mb-2 text-sm text-gray-500">
      The extractor produced the following output:
    </div>
    <Output raw_output={selected_part_text} />
  {:else}
    <div class="text-sm text-gray-500">No content.</div>
  {/if}
</Dialog>

<!-- All Parts for Document Dialog -->
<Dialog
  bind:this={document_parts_dialog}
  title="Document Parts"
  width="wide"
  action_buttons={[{ label: "Close", isCancel: true }]}
>
  {#if document.parts.length === 0}
    <div class="text-sm text-gray-500">No parts available.</div>
  {:else}
    <div class="space-y-6">
      {#each document.parts as part, idx}
        <div>
          <div class="mb-2 text-sm text-gray-500">Part {idx + 1}</div>
          <Output raw_output={part.text_preview} />
        </div>
      {/each}
    </div>
  {/if}
</Dialog>

<!-- The idea is that generating Q&A for a whole document might let user mess with the chunking so the whole structure would be different -->
<Dialog
  title="Generate Q&A Pairs"
  bind:this={generate_document_warning_dialog}
  action_buttons={[
    {
      label: "Cancel",
      action: () => {
        generate_document_warning_dialog?.close()
        return true
      },
    },
    {
      label: "Continue",
      action: () => {
        proceed_with_generate_for_document()
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
        warning_message="If you proceed, all existing Q&A pairs for this document will be lost and any document chunks will be replaced. Consider generating Q&A for specific chunks instead."
      />
    </div>
  </div>
</Dialog>
