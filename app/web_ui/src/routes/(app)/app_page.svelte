<script lang="ts">
  import { goto } from "$app/navigation"
  import type { ActionButton } from "$lib/types"

  export let title: string = ""
  export let subtitle: string = ""
  export let sub_subtitle: string = ""
  export let sub_subtitle_link: string | undefined = undefined
  // An in-page action for the sub-subtitle, for a line that opens something on
  // this page (a dialog) rather than navigating away. Takes precedence over
  // sub_subtitle_link, which opens documentation in a new tab; a caller that
  // passes both wants the action, since only one line renders.
  export let sub_subtitle_action: (() => void) | undefined = undefined
  export let no_y_padding: boolean = false
  export let action_buttons: ActionButton[] = []
  export let limit_max_width: boolean = false

  type Breadcrumb = {
    label: string
    href: string
    // Use replaceState instead of pushState. Use this when the breadcrumb
    // target is effectively where the user came from (e.g. a sub-flow page
    // pointing at its parent), so clicking the breadcrumb doesn't stack a
    // new history entry that confuses the browser back button.
    replace_state?: boolean
  }
  export let breadcrumbs: Breadcrumb[] = []

  function handle_breadcrumb_click(event: MouseEvent, breadcrumb: Breadcrumb) {
    if (!breadcrumb.replace_state) return
    // Let the browser handle modifier-key clicks (open in new tab, etc.).
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
    event.preventDefault()
    goto(breadcrumb.href, { replaceState: true })
  }

  function run_action_button(action_button: ActionButton) {
    if (action_button.disabled || action_button.loading) {
      return
    }
    if (action_button.handler) {
      action_button.handler()
    } else if (action_button.href) {
      if (action_button.href.startsWith("http")) {
        // Open in new tab + goto doesn't work for external links
        window.open(action_button.href, "_blank")
      } else {
        goto(action_button.href)
      }
    }
  }

  function handle_key_down(event: KeyboardEvent) {
    // Skip if any input element is focused (unless it's an allow-keyboard-nav element)
    if (
      (document.activeElement instanceof HTMLInputElement ||
        document.activeElement instanceof HTMLTextAreaElement ||
        document.activeElement instanceof HTMLSelectElement) &&
      !document.activeElement.classList.contains("allow-keyboard-nav")
    ) {
      return
    }

    for (const action_button of action_buttons) {
      if (event.key === action_button.shortcut) {
        event.preventDefault()
        run_action_button(action_button)
        return
      }
    }
  }
</script>

<svelte:window on:keydown={handle_key_down} />

{#if breadcrumbs.length > 0}
  <div class="breadcrumbs text-sm pt-0">
    <ul>
      {#each breadcrumbs as breadcrumb}
        <li>
          <a
            href={breadcrumb.href}
            on:click={(e) => handle_breadcrumb_click(e, breadcrumb)}
            >{breadcrumb.label}</a
          >
        </li>
      {/each}
      <!-- Adds the last ">" separator -->
      <li></li>
    </ul>
  </div>
{/if}
<div class="flex flex-row {limit_max_width ? 'max-w-[1400px]' : ''}">
  <div class="flex flex-col grow">
    <h1 class="text-2xl font-bold">{title}</h1>
    {#if subtitle}
      <p class="text-base font-medium mt-1">{subtitle}</p>
    {/if}
    {#if sub_subtitle}
      <!-- One paragraph whatever the line does, so a caller cannot change the
           header's typography by choosing an action over a link. -->
      <p class="text-sm font-light mt-1">
        {#if sub_subtitle_action}
          <button class="link" on:click={sub_subtitle_action}>
            {sub_subtitle}
          </button>
        {:else if sub_subtitle_link}
          <a href={sub_subtitle_link} class="link" target="_blank">
            {sub_subtitle}
          </a>
        {:else}
          {sub_subtitle}
        {/if}
      </p>
    {/if}
  </div>
  <div class="flex flex-col md:flex-row gap-2 shrink-0">
    {#each action_buttons as action_button}
      <div>
        <button
          on:click={() => run_action_button(action_button)}
          class="btn btn-xs md:btn-md md:whitespace-nowrap {!action_button.icon
            ? 'md:px-6'
            : ''} {action_button.primary ? 'btn-primary' : ''}"
          disabled={(action_button.disabled ?? false) ||
            (action_button.loading ?? false)}
        >
          {#if action_button.loading}
            <span class="loading loading-spinner loading-sm"></span>
          {:else}
            {#if action_button.notice}
              <span class="bg-primary rounded-full w-3 h-3 mr-1" />
            {/if}
            {#if action_button.icon}
              <img
                alt={action_button.label || ""}
                src={action_button.icon}
                class="w-6 h-6"
              />
            {/if}
          {/if}
          {action_button.label || ""}
        </button>
      </div>
    {/each}
  </div>
</div>

<div
  class="{no_y_padding ? '' : 'mt-8 mb-12'} {limit_max_width
    ? 'max-w-[1400px]'
    : ''}"
>
  <slot />
</div>
