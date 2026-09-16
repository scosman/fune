<script lang="ts">
  import AppPage from "../../../../app_page.svelte"
  import { base_url, client } from "$lib/api_client"
  import type { ExtractionSummary, KilnDocument } from "$lib/types"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { page } from "$app/stores"
  import { formatDate, formatSize } from "$lib/utils/formatters"
  import Dialog from "$lib/ui/dialog.svelte"
  import PropertyList from "$lib/ui/property_list.svelte"
  import DeleteDialog from "$lib/ui/delete_dialog.svelte"
  import { isMacOS } from "$lib/utils/platform"
  import { goto } from "$app/navigation"
  import Output from "$lib/ui/output.svelte"
  import { capitalize } from "$lib/utils/formatters"
  import TagPicker from "$lib/ui/tag_picker.svelte"
  import { ragProgressStore } from "$lib/stores/rag_progress_store"
  import { ui_state } from "$lib/stores"
  import InfoTooltip from "$lib/ui/info_tooltip.svelte"
  import TableActionMenu from "$lib/ui/table_action_menu.svelte"
  import EditDialog from "$lib/ui/edit_dialog.svelte"
  import Warning from "$lib/ui/warning.svelte"
  import { agentInfo } from "$lib/agent"

  let initial_document: KilnDocument | null = null
  let updated_document: KilnDocument | null = null
  $: document = updated_document || initial_document
  let error: KilnError | null = null
  let document_loading = true
  let extractions_loading = true
  let folder_loading = false
  $: loading = document_loading || extractions_loading || folder_loading
  let results: ExtractionSummary[] | null = null

  $: project_id = $page.params.project_id!
  $: document_id = $page.params.document_id!
  $: agentInfo.set({
    name: "Document Detail",
    description: `Document detail page for document ID ${document_id} in project ID ${project_id}. Document name: ${document?.friendly_name ?? "[loading]"}. Shows document properties, extractions, and tagging.`,
  })

  // dialog state
  let output_dialog: Dialog | null = null
  let dialog_extraction: ExtractionSummary | null = null

  let edit_dialog: EditDialog | null = null

  $: download_document_url = `${base_url}/api/projects/${project_id}/documents/${document_id}/download`

  $: if (project_id && document_id) {
    initial_document = null
    updated_document = null
    results = null
    error = null
    get_document(project_id, document_id)
    get_extractions(project_id, document_id)
  }

  async function get_document(req_project_id: string, req_document_id: string) {
    try {
      document_loading = true
      const { data: document_response, error: get_error } = await client.GET(
        "/api/projects/{project_id}/documents/{document_id}",
        {
          params: {
            path: {
              project_id: req_project_id,
              document_id: req_document_id,
            },
          },
        },
      )
      if (req_project_id !== project_id || req_document_id !== document_id)
        return
      if (get_error) {
        throw get_error
      }
      updated_document = document_response
    } catch (e) {
      if (req_project_id !== project_id || req_document_id !== document_id)
        return
      if (e instanceof Error && e.message.includes("Load failed")) {
        error = new KilnError(
          "Could not load dataset. It may belong to a project you don't have access to.",
          null,
        )
      } else {
        error = createKilnError(e)
      }
    } finally {
      if (req_project_id === project_id && req_document_id === document_id) {
        document_loading = false
      }
    }
  }

  async function get_extractions(
    req_project_id: string,
    req_document_id: string,
  ) {
    try {
      extractions_loading = true
      const { data: extractions_response, error: get_error } = await client.GET(
        "/api/projects/{project_id}/documents/{document_id}/extractions",
        {
          params: {
            path: {
              project_id: req_project_id,
              document_id: req_document_id,
            },
          },
        },
      )
      if (req_project_id !== project_id || req_document_id !== document_id)
        return
      if (get_error) {
        throw get_error
      }
      results = extractions_response
    } finally {
      if (req_project_id === project_id && req_document_id === document_id) {
        extractions_loading = false
      }
    }
  }

  async function open_enclosing_folder() {
    try {
      folder_loading = true
      const { error: open_error } = await client.POST(
        "/api/projects/{project_id}/documents/{document_id}/open_enclosing_folder",
        {
          params: {
            path: {
              project_id,
              document_id,
            },
          },
        },
      )
      if (open_error) {
        throw createKilnError(open_error)
      }
    } finally {
      folder_loading = false
    }
  }

  let delete_document_dialog: DeleteDialog | null = null
  $: delete_document_url = `/api/projects/${project_id}/documents/${document_id}`
  function after_document_delete() {
    ragProgressStore.run_all_rag_configs(project_id).catch((error) => {
      console.error("Error running all rag configs", error)
    })
    goto(`/docs/library/${project_id}`)
  }

  let delete_extraction_id: string | null = null
  let delete_extraction_dialog: DeleteDialog | null = null
  $: delete_extraction_url = `/api/projects/${project_id}/documents/${document_id}/extractions/${delete_extraction_id}`
  async function after_delete_extraction() {
    get_extractions(project_id, document_id)
  }

  let tags_error: KilnError | null = null

  async function patch_document(patch_body: {
    name?: string
    description?: string
    tags?: string[]
  }) {
    const { data: document_response, error: patch_error } = await client.PATCH(
      "/api/projects/{project_id}/documents/{document_id}",
      {
        params: {
          path: {
            project_id,
            document_id,
          },
        },
        body: patch_body,
      },
    )
    if (patch_error) {
      throw patch_error
    }
    return document_response
  }

  async function save_tags(tags: string[]) {
    try {
      let patch_body = {
        tags: tags,
      }
      updated_document = await patch_document(patch_body)
      tags_error = null
    } catch (err) {
      tags_error = createKilnError(err)
    }
  }

  function truncate_string(str: string, max_length: number) {
    if (!str) {
      return ""
    }
    if (str.length <= max_length) {
      return str
    }
    return str.slice(0, max_length) + "…"
  }
</script>

<AppPage
  title="Document"
  subtitle={document?.friendly_name
    ? truncate_string(document.friendly_name, 100)
    : undefined}
  sub_subtitle="Read the Docs"
  sub_subtitle_link="https://docs.kiln.tech/docs/documents-and-search-rag#building-a-search-tool"
  limit_max_width
  breadcrumbs={[
    {
      label: "Optimize",
      href: `/optimize/${project_id}/${$ui_state.current_task_id}`,
    },
    {
      label: "Docs & Search",
      href: `/docs/${project_id}`,
    },
    {
      label: "Document Library",
      href: `/docs/library/${project_id}`,
    },
  ]}
  action_buttons={[
    {
      label: "Edit",
      handler: () => {
        edit_dialog?.show()
      },
    },
    {
      icon: "/images/download.svg",
      href: download_document_url || "",
    },
    {
      icon: "/images/folder.svg",
      handler: () => {
        open_enclosing_folder()
      },
    },
    {
      icon: "/images/delete.svg",
      handler: () => delete_document_dialog?.show(),
      shortcut: isMacOS() ? "Backspace" : "Delete",
    },
  ]}
>
  {#if loading}
    <div class="w-full min-h-[50vh] flex justify-center items-center">
      <div class="loading loading-spinner loading-lg"></div>
    </div>
  {:else if document}
    <div class="flex flex-col xl:flex-row gap-8 xl:gap-16">
      {#if results}
        <div class="flex-grow">
          <div class="text-xl font-bold flex flex-row items-center gap-2">
            Document Extractions
            <InfoTooltip
              no_pad
              tooltip_text="Extractors generate a text representation of this document which can be indexed, searched and read by language models."
              position="bottom"
            />
          </div>
          {#if results.length == 0}
            <div class="mt-4 text-sm text-gray-500">
              No extractions found for this document.
            </div>
          {:else}
            <div class="text-sm overflow-x-auto rounded-lg border mt-4">
              <table class="table">
                <thead>
                  <tr>
                    <th>Extraction Details</th>
                    <th>Extraction Output</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {#each results as result}
                    <tr>
                      <td class="flex justify-start">
                        <div
                          class="grid grid-cols-[auto_1fr] gap-y-2 gap-x-4 text-sm items-start"
                        >
                          <div>ID</div>
                          <div class="text-gray-500">{result.id}</div>
                          <div>Source</div>
                          <div class="text-gray-500">
                            {#if result.source == "processed"}
                              Model Output
                            {:else if result.source == "passthrough"}
                              Unprocessed (Original)
                            {:else}
                              {result.source}
                            {/if}
                          </div>
                          <div>Extractor</div>
                          <div class="text-gray-500">
                            <a
                              href={`/docs/extractors/${project_id}/${result.extractor.id}/extractor`}
                              class="link"
                            >
                              {result.extractor.name}
                            </a>
                          </div>
                        </div>
                      </td>
                      <td>
                        <button
                          class="text-left"
                          on:click={() => {
                            dialog_extraction = result
                            output_dialog?.show()
                          }}
                        >
                          <Output
                            raw_output={result.output_content}
                            max_height="200px"
                            hide_toggle={true}
                          />
                        </button>
                      </td>
                      <td class="flex justify-start">
                        <TableActionMenu
                          items={[
                            {
                              label: "Delete Extraction",
                              onclick: () => {
                                delete_extraction_id = result.id || null
                                delete_extraction_dialog?.show()
                              },
                            },
                          ]}
                        />
                      </td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}
        </div>
      {/if}

      <div class="w-72 2xl:w-96 flex-none flex flex-col gap-4">
        <PropertyList
          properties={[
            { name: "ID", value: document.id || "Unknown" },
            {
              name: "Name",
              value: document.friendly_name,
            },
            {
              name: "File Size",
              value: formatSize(document.original_file.size),
            },
            {
              name: "Kind",
              value: capitalize(document.kind),
            },
            {
              name: "MIME Type",
              value: document.original_file.mime_type,
            },
            {
              name: "Created At",
              value: formatDate(document.created_at),
            },
            {
              name: "Created By",
              value: document.created_by || "Unknown",
            },
            {
              name: "Description",
              value: document.description || "None",
            },
          ]}
          title="Properties"
        />
        <div class="mt-8 mb-4">
          <div class="text-xl font-bold">Tags</div>
          <TagPicker
            tags={document.tags}
            tag_type="doc"
            {project_id}
            initial_expanded={false}
            on:tags_changed={(event) => {
              const { current } = event.detail
              document.tags = current
              save_tags(current)
            }}
          />
          {#if tags_error}
            <div class="text-error text-sm mt-2">
              {tags_error.getMessage() || "Error updating tags"}
            </div>
          {/if}
        </div>
      </div>
    </div>
  {:else if error}
    <div
      class="w-full min-h-[50vh] flex flex-col justify-center items-center gap-2"
    >
      <div class="font-medium">Error Loading Document</div>
      <div class="text-error text-sm">
        {error.getMessage() || "An unknown error occurred"}
      </div>
    </div>
  {/if}
</AppPage>

<Dialog
  bind:this={output_dialog}
  title="Extraction Output"
  width="wide"
  action_buttons={[
    {
      label: "Close",
      isCancel: true,
    },
    {
      label: "Download",
      isPrimary: true,
      action: () => {
        if (!dialog_extraction) {
          return false
        }
        window.open(
          `${base_url}/api/projects/${project_id}/documents/${document_id}/download_extraction/${dialog_extraction.id}`,
          "_blank",
        )
        return false
      },
    },
  ]}
>
  {#if dialog_extraction}
    {#if dialog_extraction.output_content_truncated}
      <div class="mb-4 flex flex-row gap-2">
        <Warning
          warning_message="The extraction output cannot be displayed in full because it is too long. To view the full output, please download the document."
          warning_color="warning"
          inline={true}
          large_icon={false}
          warning_icon="exclaim"
        />
      </div>
    {/if}
    <div class="mb-2 text-sm text-gray-500">
      The extractor produced the following output:
    </div>
    <Output
      raw_output={dialog_extraction.output_content +
        (dialog_extraction.output_content_truncated
          ? "\n\n[EXTRACTION IS TOO LONG TO DISPLAY - DOWNLOAD TO VIEW FULL OUTPUT]"
          : "")}
    />
  {:else}
    <div class="text-sm text-gray-500">No extraction output found.</div>
  {/if}
</Dialog>

<DeleteDialog
  name={document?.name || "Document"}
  bind:this={delete_document_dialog}
  delete_url={delete_document_url}
  after_delete={after_document_delete}
/>

<DeleteDialog
  name="Extraction"
  bind:this={delete_extraction_dialog}
  delete_url={delete_extraction_url}
  after_delete={after_delete_extraction}
/>

<EditDialog
  bind:this={edit_dialog}
  after_save={() => {
    get_document(project_id, document_id)
    get_extractions(project_id, document_id)
  }}
  name="Document"
  patch_url={`/api/projects/${project_id}/documents/${document_id}`}
  fields={[
    {
      label: "Name",
      description:
        "A name for your own reference. It does not need to contain the file extension or be unique.",
      api_name: "name_override",
      value: document?.friendly_name || "",
      input_type: "input",
      optional: true, // if empty, we revert to the original name
      info_description: "If empty, the original file name will be used.",
    },
    {
      label: "Description",
      description: "A description of the document for your own reference.",
      api_name: "description",
      value: document?.description || "",
      input_type: "textarea",
      optional: true,
    },
  ]}
/>
