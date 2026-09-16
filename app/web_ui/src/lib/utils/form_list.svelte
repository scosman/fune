<script lang="ts">
  export let content: unknown[] = []
  export let content_label: string = "Item"
  export let hide_item_header: boolean = false
  export let start_with_one: boolean = true
  export let empty_content: unknown = {}
  // Some uses of this component let svelte objects manage content. They just want a unique ID for the list.
  export let generate_uuid_content: boolean = false
  export let frozen: boolean = false
  export let hide_add_button: boolean = false
  export let empty_description: string | null = null

  // Unique ID for the list, for scrolling to top after removal
  let id = "form_list_" + Math.random().toString(36).substring(2, 15)

  // The add button, used as a stable in-tree anchor to announce row changes.
  let add_button_el: HTMLButtonElement | undefined

  // Adding/removing a row is a plain button click, which fires no native
  // input/change event. Emit a bubbling change like a native control would
  // (mirrors fancy_select) so ancestor on:change handlers — e.g. unsaved-changes
  // and tested-state tracking — react to structural row edits. The initial
  // start_with_one seed runs before mount, so add_button_el is still undefined
  // then and no spurious event fires.
  function announce_row_change() {
    add_button_el?.dispatchEvent(new Event("change", { bubbles: true }))
  }

  function remove_item(index: number) {
    let item = content[index]
    let isItemUnedited = false

    if (item === null || item === undefined) {
      return
    } else if (item instanceof Object && empty_content instanceof Object) {
      // Compare item to empty_content to check if it's unedited
      isItemUnedited = Object.keys(item).every((key) => {
        return (
          item[key as keyof typeof item] ===
          empty_content[key as keyof typeof empty_content]
        )
      })
    }

    if (
      generate_uuid_content ||
      isItemUnedited ||
      confirm(
        "Are you sure you want to remove " +
          content_label +
          " #" +
          (index + 1) +
          "?\n\nYou've set content and it hasn't been saved.",
      )
    ) {
      content.splice(index, 1)
      // trigger reactivity
      content = content
      announce_row_change()
      // Move the page to the top anchor
      const list = document.getElementById(id)
      if (list) {
        // Scroll to the top of the list because removing sets focus to first form element for some reason
        setTimeout(() => {
          list.scrollIntoView()
        }, 1)
      }
    }
  }

  async function add_item(focus: boolean = true) {
    if (generate_uuid_content) {
      content.push(Math.random().toString(36))
    } else {
      content.push(structuredClone(empty_content))
    }
    // Trigger reactivity
    content = content
    announce_row_change()
    if (focus) {
      // Scroll into view. Async to allow rendering first
      setTimeout(() => {
        const list = document.getElementById(
          "list_item_" + (content.length - 1) + "_" + id,
        )
        if (list) {
          list.scrollIntoView({ block: "center" })
        }
      }, 1)
    }
  }

  // Add one empty item to start so the list isn't empty
  if (start_with_one && content.length === 0) {
    add_item(false)
  }
</script>

{#if content.length > 0}
  <div class="flex flex-col gap-8" {id}>
    {#each content as item, item_index (item)}
      <div id={"list_item_" + item_index + "_" + id}>
        <div class="flex flex-row gap-3 font-medium text-sm pb-2">
          <div class="grow {hide_item_header ? 'opacity-0' : ''}">
            {content_label} #{item_index + 1}
          </div>
          <button
            class="link text-xs text-gray-500 {frozen ? 'hidden' : ''}"
            on:click={() => remove_item(item_index)}
          >
            remove
          </button>
        </div>
        <slot {item} {item_index} />
      </div>
    {/each}
  </div>
{:else if empty_description}
  <div class="text-xs font-medium text-left">
    {empty_description}
  </div>
{/if}

<div class="flex place-content-center">
  <button
    bind:this={add_button_el}
    class="btn btn-sm mt-4 {frozen || hide_add_button ? 'hidden' : ''}"
    on:click={() => add_item(true)}
    id={id + "_add_button"}
  >
    Add {content_label}
  </button>
</div>
