<script lang="ts">
  import { goto } from "$app/navigation"
  import type { OptionGroup } from "./fancy_select_types"
  import { filter_option_groups } from "./fancy_select_search"
  import { computePosition, autoUpdate, offset } from "@floating-ui/dom"
  import { onMount, onDestroy } from "svelte"

  export let aria_label: string | null = null
  export let options: OptionGroup[] = []
  export let selected: unknown
  export let empty_label: string = "Select an option"
  export let empty_state_message: string = "No options available"
  export let empty_state_subtitle: string | null = null
  export let empty_state_link: string | null = null
  export let multi_select: boolean = false
  export let disabled: boolean = false
  export let error_outline: boolean = false

  // Add this variable to track scrollability
  let isMenuScrollable = false
  let menuElement: HTMLElement
  let dropdownElement: HTMLElement
  let selectedElement: HTMLElement
  let scrolling = false
  let scrollInterval: number | null = null
  let focusedIndex = -1
  let listVisible = false
  let cleanupAutoUpdate: (() => void) | null = null
  let mounted = false
  let naturalDropdownHeight: number | null = null
  const id = Math.random().toString(36).substring(2, 15)

  // Search functionality variables
  let searchText = ""
  let isSearching = false
  let searchInputElement: HTMLInputElement

  // Computed filtered options based on search
  $: filteredOptions = filter_option_groups(options, searchText)

  // Reset search when dropdown closes
  $: if (!listVisible) {
    searchText = ""
    isSearching = false
    naturalDropdownHeight = null // Reset cached height when dropdown closes
  }

  // Used for multi select, where we want a typed array of values
  let selected_values: unknown[] = []

  onMount(() => {
    mounted = true
    if (multi_select && Array.isArray(selected)) {
      // Allow the caller to initialize the selected values
      selected_values = selected
    }
  })

  onDestroy(() => {
    stopScroll()
    if (cleanupAutoUpdate) {
      cleanupAutoUpdate()
    }
  })

  // Select a prompt
  function selectOption(option: unknown) {
    // Check if this option is disabled
    const flatOptions = options.flatMap((group) => group.options)
    const optionObj = flatOptions.find((item) => item.value === option)
    if (optionObj?.disabled) {
      return
    }

    if (multi_select) {
      // Deselect if already selected, select if not
      if (selected_values.includes(option)) {
        selected_values = selected_values.filter((value) => value !== option)
      } else {
        selected_values.push(option)
      }
      // Update selected, which is what we expose outside the component
      selected = selected_values

      // Don't close the dropdown for multi select
    } else {
      selected = option

      // Delay hiding the dropdown to ensure the click event is fully processed
      setTimeout(() => {
        listVisible = false
        setTimeout(() => {
          // Return focus to the main select element so keyboard nav still works
          if (selectedElement) {
            selectedElement.focus()
          }
        }, 0)
      }, 0)
    }

    // Fire a bubbling change like a native <select> would, so ancestor
    // on:change handlers (e.g. unsaved-changes tracking) react to picks made
    // through this custom dropdown, which produces no native change event.
    selectedElement?.dispatchEvent(new Event("change", { bubbles: true }))
  }

  // Make it reactive, when selected changes, update the selected_values
  $: if (multi_select && selected instanceof Array) {
    selected_values = selected
  }

  // Function to check if menu is scrollable
  function checkIfScrollable() {
    if (menuElement) {
      isMenuScrollable = menuElement.scrollHeight > menuElement.clientHeight
    }
  }

  // Create an action to bind to the menu element
  function scrollableCheck(node: HTMLElement) {
    menuElement = node
    checkIfScrollable()

    // Create a mutation observer to detect content changes
    const observer = new MutationObserver(checkIfScrollable)
    observer.observe(node, { childList: true, subtree: true })

    return {
      destroy() {
        observer.disconnect()
      },
    }
  }

  // Watch for changes to options and recheck scrollability
  $: void (options, setTimeout(checkIfScrollable, 0))

  // Set up floating UI positioning when dropdown becomes visible
  $: if (listVisible && selectedElement && dropdownElement && mounted) {
    setupFloatingPosition()
  } else if (!listVisible && cleanupAutoUpdate) {
    cleanupAutoUpdate()
    cleanupAutoUpdate = null
  }

  function getEffectiveViewportHeight(element: HTMLElement): number {
    // Check if we're inside a Kiln Dialog.svelte or DaisyUI dialog element
    const dialog = element.closest(".modal-box")
    if (dialog) {
      return dialog.clientHeight
    }

    // Fall back to window height if not in a dialog
    return window.innerHeight
  }

  function setupFloatingPosition() {
    if (!selectedElement || !dropdownElement) return

    const updatePosition = () => {
      computePosition(selectedElement, dropdownElement, {
        placement: "bottom-start",
        strategy: "fixed",
        middleware: [
          offset(2), // Small gap between trigger and dropdown
          // Custom positioning and sizing middleware to handle the 3 cases
          {
            name: "customPositioningAndSizing",
            fn: ({ x, y, rects, elements }) => {
              // Find the effective viewport height - could be constrained by a dialog/modal
              const viewportHeight = getEffectiveViewportHeight(selectedElement)
              const referenceRect = rects.reference
              const padding = 10
              const floatingEl = elements.floating

              // Calculate space below reference element
              const spaceBelow =
                viewportHeight - (referenceRect.y + referenceRect.height) - 2 // 2px gap

              // Get the natural height of the dropdown content (measure only once)
              if (naturalDropdownHeight === null) {
                // First time measuring - temporarily set max height to auto to get natural height
                floatingEl.style.maxHeight = "none"
                naturalDropdownHeight = floatingEl.scrollHeight
              }

              // Calculate width - minimum 300px or reference width, whichever is larger
              const minWidth = 320
              const referenceWidth = referenceRect.width
              const desiredWidth = Math.max(minWidth, referenceWidth)
              const maxWidth = Math.min(
                desiredWidth,
                window.innerWidth - 2 * padding,
              )

              let finalHeight
              let finalY = y

              // CASE 1: Content fits below reference element
              if (naturalDropdownHeight <= spaceBelow) {
                finalHeight = naturalDropdownHeight
                finalY = y // Normal positioning (2px below reference)
                // Add some height for the search input, allowing it to grow downwards. The max-height isn't really used in this case as it will be it's natural height.
                finalHeight = finalHeight + 100
              }
              // CASE 2: Doesn't fit below but shorter than viewport - 20px (padding)
              else if (naturalDropdownHeight <= viewportHeight - 20) {
                finalHeight = naturalDropdownHeight
                // Anchor to bottom of viewport with 10px padding
                finalY = viewportHeight - naturalDropdownHeight - padding
              }
              // CASE 3: Taller than viewport - 20px (padding)
              else {
                finalHeight = viewportHeight - 2 * padding
                // Position from top of viewport with 10px padding
                finalY = padding
              }

              // Set the width of the dropdown
              floatingEl.style.setProperty("--dropdown-width", `${maxWidth}px`)

              // Also set max-height directly on the dropdown element for immediate effect
              floatingEl.style.maxHeight = `${finalHeight}px`

              return { x, y: finalY }
            },
          },
        ],
      }).then(({ x, y }) => {
        Object.assign(dropdownElement.style, {
          left: `${x}px`,
          top: `${y}px`,
          position: "fixed",
        })
      })
    }

    // Initial positioning
    updatePosition()

    // Set up auto-update with proper options for scroll handling
    cleanupAutoUpdate = autoUpdate(
      selectedElement,
      dropdownElement,
      updatePosition,
      {
        // Enable all update triggers for maximum compatibility
        ancestorScroll: true,
        ancestorResize: true,
        elementResize: true,
        layoutShift: true,
        animationFrame: false, // Set to true if you have animations
      },
    )
  }

  // Add scroll functionality when hovering the indicator
  function startScroll() {
    if (!scrolling && isMenuScrollable) {
      scrolling = true
      scrollInterval = window.setInterval(() => {
        if (menuElement) {
          menuElement.scrollTop += 8

          // Stop scrolling if we've reached the bottom
          if (
            menuElement.scrollTop + menuElement.clientHeight >=
            menuElement.scrollHeight
          ) {
            stopScroll()
          }
        }
      }, 20)
    }
  }

  function stopScroll() {
    scrolling = false
    if (scrollInterval !== null) {
      window.clearInterval(scrollInterval)
      scrollInterval = null
    }
  }

  function scrollToFocusedIndex() {
    if (listVisible && menuElement) {
      const optionElement = document.getElementById(
        `option-${id}-${focusedIndex}`,
      )
      if (optionElement) {
        // Check if the element is fully in view
        const menuRect = menuElement.getBoundingClientRect()
        const optionRect = optionElement.getBoundingClientRect()

        const isInView =
          optionRect.top >= menuRect.top && optionRect.bottom <= menuRect.bottom

        // Only scroll if the element is not in view
        if (!isInView) {
          optionElement.scrollIntoView({ block: "nearest" })
        }
      }
    }
  }

  function getFlatOptions() {
    return filteredOptions.flatMap((group) => group.options)
  }

  function findNextEnabledOptionIndex(
    currentIndex: number,
    flatOptions: Array<{ disabled?: boolean }>,
  ): number {
    const originalIndex = currentIndex
    let nextIndex = currentIndex + 1
    while (nextIndex < flatOptions.length && flatOptions[nextIndex].disabled) {
      nextIndex++
    }
    // If we couldn't find a next enabled option, stay at the original index
    if (nextIndex >= flatOptions.length) {
      return originalIndex
    }
    return nextIndex
  }

  function findPreviousEnabledOptionIndex(
    currentIndex: number,
    flatOptions: Array<{ disabled?: boolean }>,
  ): number {
    const originalIndex = currentIndex
    let prevIndex = currentIndex - 1
    while (prevIndex >= 0 && flatOptions[prevIndex].disabled) {
      prevIndex--
    }
    // If we couldn't find a previous enabled option, stay at the original index
    if (prevIndex < 0) {
      return originalIndex
    }
    return prevIndex
  }

  // Handle clear search
  function clearSearch() {
    searchText = ""
    isSearching = false
    focusedIndex = 0
    if (searchInputElement) {
      searchInputElement.blur()
    }
    // Return focus to the main select element so escape key works properly
    if (selectedElement) {
      selectedElement.focus()
    }
  }

  // Handle key input when dropdown is open
  function handleKeyInput(event: KeyboardEvent) {
    // Don't interfere if we're already focused on search input
    if (isSearching && document.activeElement === searchInputElement) {
      return
    }

    // Don't interfere if the fancy select is not focused either
    if (document.activeElement !== selectedElement) {
      return
    }

    // Don't interfere with navigation keys
    if (
      event.key === "ArrowDown" ||
      event.key === "ArrowUp" ||
      event.key === "Enter" ||
      event.key === "Escape" ||
      event.key === "Tab"
    ) {
      return
    }

    // If it's a printable character, start search mode
    if (
      event.key.length === 1 &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey
    ) {
      event.preventDefault()
      if (!isSearching) {
        isSearching = true
        searchText = event.key
        focusedIndex = 0
        // Focus the search input after it's rendered
        setTimeout(() => {
          if (searchInputElement) {
            searchInputElement.focus()
          }
        }, 0)
      }
    }
  }

  // Handle click outside to close dropdown
  function handleCloseElementClick(event: Event) {
    if (
      listVisible &&
      selectedElement &&
      !selectedElement.contains(event.target as Node) &&
      dropdownElement &&
      !dropdownElement.contains(event.target as Node)
    ) {
      event.preventDefault()
      listVisible = false
    }
  }

  // The "click away to close" element. Document unless a modal is open.
  function getClickToCloseElement(): Document | Element {
    const topModal = [...document.querySelectorAll("dialog[open]")].pop()
    return topModal ? topModal : document
  }

  let target_close_element: Document | Element | null = null
  $: if (mounted) {
    if (listVisible) {
      // Save the element we're adding event listeners to so we can remove them later
      target_close_element = getClickToCloseElement()
      target_close_element.addEventListener("click", handleCloseElementClick)
      document.addEventListener("keydown", handleKeyInput)
    } else {
      target_close_element?.removeEventListener(
        "click",
        handleCloseElementClick,
      )
      document.removeEventListener("keydown", handleKeyInput)
    }
  }

  function selectedLabel(
    selected: unknown,
    selected_values: unknown[],
    options: OptionGroup[],
    empty_label: string,
  ) {
    if (multi_select && selected_values.length > 1) {
      return (
        "" +
        selected_values.length +
        " Selected: " +
        selected_values
          .map((value) => {
            const flatOptions = options.flatMap((group) => group.options)
            const selectedOption = flatOptions.find(
              (item) => item.value === value,
            )
            return selectedOption ? selectedOption.label : empty_label
          })
          .join(", ")
      )
    }

    let effective_selected = selected
    if (multi_select) {
      if (selected_values.length === 1) {
        // Use the labeling system for single select if only one is selected
        effective_selected = selected_values[0]
      } else {
        return empty_label
      }
    }

    const flatOptions = options.flatMap((group) => group.options)
    const selectedOption = flatOptions.find(
      (item) => item.value === effective_selected,
    )
    return selectedOption ? selectedOption.label : empty_label
  }

  function containerKeyDown(event: KeyboardEvent) {
    if (disabled) {
      return
    }
    const key_is_alpha_numeric = /^[a-zA-Z0-9]$/.test(event.key)
    if (
      !listVisible &&
      (event.key === "ArrowDown" ||
        event.key === "ArrowUp" ||
        event.key === "Enter" ||
        key_is_alpha_numeric)
    ) {
      event.preventDefault()
      listVisible = true
      focusedIndex = 0
      // Typing should start a search
      if (key_is_alpha_numeric) {
        searchText = event.key
        isSearching = true
        focusedIndex = 0
        // Focus the search input after it's rendered
        setTimeout(() => {
          if (searchInputElement) {
            searchInputElement.focus()
          }
        }, 0)
      }
      return
    }
    if (event.key === "Escape") {
      event.preventDefault()
      listVisible = false
      return
    }
    if (event.key === "ArrowDown") {
      event.preventDefault()
      const flatOptions = getFlatOptions()
      focusedIndex = findNextEnabledOptionIndex(focusedIndex, flatOptions)
      scrollToFocusedIndex()
    } else if (event.key === "ArrowUp") {
      event.preventDefault()
      const flatOptions = getFlatOptions()
      focusedIndex = findPreviousEnabledOptionIndex(focusedIndex, flatOptions)
      scrollToFocusedIndex()
    } else if (event.key === "Enter") {
      const flatOptions = getFlatOptions()
      if (flatOptions[focusedIndex]) {
        selectOption(flatOptions[focusedIndex].value)
      }
    }
  }

  // One source of truth for badge sizing/color, shared by the inline and
  // below-label placements so they can't drift apart.
  function badge_classes(
    badge: string,
    badge_color: string | undefined,
  ): string {
    const shape = badge.length <= 2 ? "rounded-full w-5 h-5" : "px-2"
    const color = badge_color === "primary" ? "badge-primary" : "badge-ghost"
    return `badge badge-sm text-xs ${shape} ${color}`
  }
</script>

<div class="dropdown w-full relative">
  <div
    aria-label={aria_label}
    tabindex={disabled ? -1 : 0}
    role="listbox"
    class="select select-bordered w-full flex items-center {!listVisible
      ? 'focus:ring-2 focus:ring-offset-2 focus:ring-base-300'
      : ''} {disabled ? 'opacity-50 cursor-not-allowed' : ''} {error_outline
      ? 'border-error'
      : ''}"
    bind:this={selectedElement}
    on:click={() => {
      if (!disabled) {
        listVisible = true
      }
    }}
    on:blur={(_) => {
      if (multi_select) {
        return
      }
      // Only close if focus is not moving to the dropdown
      setTimeout(() => {
        if (
          dropdownElement &&
          !dropdownElement.contains(document.activeElement)
        ) {
          listVisible = false
        }
      }, 0)
    }}
    on:keydown={containerKeyDown}
  >
    <span class="truncate">
      {selectedLabel(selected, selected_values, options, empty_label)}
    </span>
  </div>

  {#if listVisible && mounted}
    {@const first_group_has_label = filteredOptions[0]?.label}
    <div
      bind:this={dropdownElement}
      class="bg-base-100 rounded-box z-[1000] p-2 {first_group_has_label
        ? 'pt-0'
        : ''} shadow border flex flex-col fixed"
      style="width: var(--dropdown-width, {selectedElement?.offsetWidth ||
        0}px);"
    >
      <!-- Search input - only show when searching -->
      {#if isSearching}
        <div
          class="flex items-center gap-2 p-2 border-b border-base-200 bg-base-100 mt-2"
        >
          <input
            bind:this={searchInputElement}
            bind:value={searchText}
            type="text"
            placeholder="Search..."
            class="input input-sm flex-1 focus:outline-none focus:ring-2 focus:ring-primary/50"
            on:keydown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault()
                clearSearch()
              } else if (event.key === "ArrowDown") {
                event.preventDefault()
                const flatOptions = getFlatOptions()
                focusedIndex = findNextEnabledOptionIndex(
                  focusedIndex,
                  flatOptions,
                )
                scrollToFocusedIndex()
              } else if (event.key === "ArrowUp") {
                event.preventDefault()
                const flatOptions = getFlatOptions()
                focusedIndex = findPreviousEnabledOptionIndex(
                  focusedIndex,
                  flatOptions,
                )
                scrollToFocusedIndex()
              } else if (event.key === "Enter") {
                event.preventDefault()
                const flatOptions = getFlatOptions()
                if (flatOptions[focusedIndex]) {
                  selectOption(flatOptions[focusedIndex].value)
                }
              }
            }}
          />
          <button
            type="button"
            class="btn btn-ghost btn-sm btn-square"
            on:click={clearSearch}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>
      {/if}

      {#if options.length === 0}
        <!-- Empty state -->
        <button
          class="px-4 pt-4 pb-2 text-center text-base-content/60 {empty_state_link
            ? 'cursor-pointer'
            : 'cursor-default'}"
          on:mousedown={() => {
            if (empty_state_link) {
              goto(empty_state_link)
            }
          }}
        >
          <div>
            {empty_state_message}
          </div>
          {#if empty_state_subtitle}
            <div class="text-sm {empty_state_link ? 'link' : ''}">
              {empty_state_subtitle}
            </div>
          {/if}
        </button>
      {/if}

      <ul
        class="menu overflow-y-auto overflow-x-hidden flex-nowrap pt-0 mt-2 custom-scrollbar flex-1"
        use:scrollableCheck
      >
        {#each filteredOptions as option, sectionIndex}
          {#if option.label}
            <li
              class="menu-title pl-1 sticky top-0 bg-white z-10 flex flex-row items-center justify-between"
            >
              {option.label}
              {#if option.action_label}
                <button
                  type="button"
                  class="btn btn-xs btn-primary btn-outline rounded-full"
                  on:mousedown|preventDefault|stopPropagation
                  on:click|preventDefault|stopPropagation={() => {
                    option.action_handler?.()
                  }}
                >
                  {option.action_label}
                </button>
              {/if}
            </li>
          {/if}

          {#each option.options as item, index}
            {@const overallIndex =
              filteredOptions
                .slice(0, sectionIndex)
                .reduce((count, group) => count + group.options.length, 0) +
              index}
            <li id={`option-${id}-${overallIndex}`}>
              <button
                role="option"
                aria-selected={multi_select
                  ? selected_values.includes(item.value)
                  : selected === item.value}
                aria-disabled={item.disabled || false}
                class="pointer-events-auto flex {focusedIndex === overallIndex
                  ? ' active'
                  : 'hover:bg-transparent'} {item.disabled
                  ? 'opacity-50 cursor-not-allowed'
                  : ''}"
                disabled={item.disabled}
                on:mousedown={(event) => {
                  if (item.disabled) {
                    event.preventDefault()
                    return
                  }
                  event.stopPropagation()
                  if (!item.disabled) {
                    selectOption(item.value)
                  }
                }}
                on:mouseenter={() => {
                  if (!item.disabled) {
                    focusedIndex = overallIndex
                  }
                }}
              >
                <div class="flex flex-row gap-3 items-center flex-1">
                  {#if multi_select && !item.hide_check}
                    <input
                      type="checkbox"
                      class="checkbox checkbox-sm no-animation pointer-events-none"
                      checked={selected_values.includes(item.value)}
                      disabled={item.disabled}
                    />
                  {/if}
                  <div class="flex-grow flex flex-col text-left gap-[1px]">
                    <div class="w-full flex flex-row gap-2 items-center">
                      <div class="flex-grow">
                        {item.label}
                      </div>
                      {#if item.badge && item.badge_placement !== "below"}
                        <div
                          class={badge_classes(item.badge, item.badge_color)}
                        >
                          {item.badge}
                        </div>
                      {/if}
                    </div>
                    {#if item.badge && item.badge_placement === "below"}
                      <div class="w-full">
                        <div
                          class={badge_classes(item.badge, item.badge_color)}
                        >
                          {item.badge}
                        </div>
                      </div>
                    {/if}
                    {#if item.description}
                      <div
                        class="text-xs font-medium text-base-content/40 w-full line-clamp-3 whitespace-pre-line"
                      >
                        {item.description}
                      </div>
                    {/if}
                  </div>
                </div>
              </button>
            </li>
          {/each}
        {/each}

        <!-- Show "No results" message when filtering returns empty -->
        {#if isSearching && filteredOptions.length === 0}
          <li class="p-4 text-center text-base-content/60">
            No results found for "{searchText}"
          </li>
        {/if}
      </ul>

      <!-- Scroll indicator - only show if scrollable -->
      {#if isMenuScrollable}
        <div class="h-5">&nbsp;</div>
        <!--svelte-ignore a11y-no-static-element-interactions -->
        <div
          class="absolute bottom-0 left-0 right-0 pointer-events-auto rounded-b-md stroke-[2px] hover:stroke-[4px] border-t border-base-200"
          on:mouseenter={startScroll}
          on:mouseleave={stopScroll}
        >
          <div
            class="bg-gradient-to-b from-transparent to-white w-full flex justify-center items-center py-1 cursor-pointer rounded-b-xl"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-linecap="round"
              stroke-linejoin="round"
              class="opacity-60"
            >
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  /* Custom scrollbar styling */
  .custom-scrollbar::-webkit-scrollbar {
    width: 6px;
  }

  .custom-scrollbar::-webkit-scrollbar-track {
    background: transparent;
  }

  .custom-scrollbar::-webkit-scrollbar-thumb {
    background-color: rgba(115, 115, 115, 0.5);
    border-radius: 20px;
  }

  /* Firefox */
  .custom-scrollbar {
    scrollbar-width: thin;
    scrollbar-color: rgba(115, 115, 115, 0.5) transparent;
  }
</style>
