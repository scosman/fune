<script lang="ts">
  import { createEventDispatcher } from "svelte"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import CloseIcon from "$lib/ui/icons/close_icon.svelte"

  const dispatch = createEventDispatcher()

  export let title: string
  export let center_content: boolean = false
  export let subtitle: string | null = null
  export let sub_subtitle: string | null = null
  export let sub_subtitle_link: string | null = null
  export let blur_background: boolean = false
  // Dialog width. "extra_wide" is for content that reads as a full page in
  // miniature (a whole conversation), where 3xl forces constant wrapping.
  // Dialogs rendered INSIDE another dialog's content are display:none while
  // closed (app.css .modal-box rule) — closed nested overlays otherwise
  // inflate the outer box's scroll area via its permanent transform.
  export let width: "normal" | "wide" | "extra_wide" = "normal"
  const id: string = "dialog-" + Math.random().toString(36)
  type ActionButton = {
    label: string
    // both return if the dialog should be closed after the action is performed
    asyncAction?: () => Promise<boolean>
    action?: () => boolean
    isCancel?: boolean
    isPrimary?: boolean
    isError?: boolean
    isWarning?: boolean
    disabled?: boolean
    loading?: boolean
    hide?: boolean
    width?: "normal" | "wide"
  }
  export let action_buttons: ActionButton[] = []
  let action_running = false

  type HeaderButton = {
    image_path: string
    alt_text: string
    action: () => void
  }
  export let header_buttons: HeaderButton[] = []

  let error: KilnError | null = null

  export function show() {
    // Clear the error, so the dialog can be used again
    error = null
    dispatch("show")
    const dialogElement = document.getElementById(id)
    const htmlDialog = dialogElement as HTMLDialogElement | null
    // Check if dialog is already open before calling showModal()
    if (htmlDialog && !htmlDialog.open) {
      htmlDialog.showModal()
    }
    // Focus the dialog itself to prevent auto-focus on the first link
    htmlDialog?.focus()
  }

  export function close() {
    // @ts-expect-error close is not a method on HTMLElement
    document.getElementById(id)?.close()
  }

  async function perform_button_action(button: ActionButton) {
    let shouldClose = true
    try {
      action_running = true
      // New run, so clear prior errors if any
      error = null
      if (button.asyncAction) {
        shouldClose = await button.asyncAction()
      } else if (button.action) {
        shouldClose = button.action()
      }
    } catch (e) {
      error = createKilnError(e)
      shouldClose = false
    } finally {
      action_running = false
    }

    if (shouldClose) {
      close()
    }
  }
</script>

<dialog
  {id}
  class="modal"
  tabindex="-1"
  on:close={() => dispatch("close")}
  on:cancel={(e) => dispatch("cancel", e)}
>
  <div
    class="modal-box text-base-content {width === 'extra_wide'
      ? 'w-11/12 max-w-7xl'
      : width === 'wide'
        ? 'w-11/12 max-w-3xl'
        : ''}"
  >
    <!-- Hidden div to force the compiler to find these classes -->
    <div class="hidden w-11/12 max-w-3xl max-w-7xl"></div>
    <div class="flex flex-row gap-2 items-start">
      <div
        class="grow flex flex-col {center_content
          ? 'items-center'
          : 'items-start'}"
      >
        <h3 class="text-lg font-medium">
          {title}
        </h3>
        {#if $$slots.subtitle}
          <!-- Rich subtitle (e.g. with an inline link). Takes precedence over
               the plain `subtitle` prop. -->
          <div class="text-base"><slot name="subtitle" /></div>
        {:else if subtitle}
          <p class="text-base">{subtitle}</p>
        {/if}
        {#if sub_subtitle}
          {#if sub_subtitle_link}
            <p class="text-sm font-light">
              <a
                href={sub_subtitle_link}
                class="link"
                target="_blank"
                rel="noopener noreferrer"
              >
                {sub_subtitle}
              </a>
            </p>
          {:else}
            <p class="text-sm font-light">{sub_subtitle}</p>
          {/if}
        {/if}
      </div>
      <div class="flex flex-row gap-2 items-center">
        {#each header_buttons as button}
          <button
            class="btn btn-sm h-8 w-8 btn-circle btn-ghost focus:outline-none"
            on:click={button.action}
            aria-label={button.alt_text}
            title={button.alt_text}
          >
            <img
              class="h-6 w-6 mb-[1px]"
              src={button.image_path}
              alt={button.alt_text}
              aria-hidden="true"
            />
          </button>
        {/each}
        <form method="dialog">
          <button
            class="btn btn-sm h-8 w-8 btn-circle btn-ghost focus:outline-none"
            aria-label="Close dialog"
          >
            <span class="h-6 w-6 block"><CloseIcon /></span>
          </button>
        </form>
      </div>
    </div>

    <div class="mt-4">
      {#if action_running}
        <div class="flex flex-col items-center justify-center min-h-[100px]">
          <div class="loading loading-spinner loading-lg"></div>
        </div>
      {:else if error}
        <div class="text-error text-sm font-medium">
          {error.getMessage() || "An unknown error occurred"}
        </div>
      {:else}
        <slot />
      {/if}

      {#if error || (action_buttons.length > 0 && !action_running)}
        <div class="flex flex-row gap-2 justify-end mt-6">
          {#if error}
            <form method="dialog">
              <button class="btn btn-sm h-10 btn-outline min-w-24">Close</button
              >
            </form>
          {:else}
            {#each action_buttons as button}
              {#if button.hide}
                <!-- do nothing -->
              {:else if button.isCancel}
                <form method="dialog">
                  <button class="btn btn-sm h-10 btn-outline min-w-24"
                    >{button.label || "Cancel"}</button
                  >
                </form>
              {:else}
                <button
                  class="btn btn-sm h-10 min-w-24 {button.width === 'wide'
                    ? 'w-full'
                    : ''} {button.isPrimary ? 'btn-primary' : ''}
                  {button.isError ? 'btn-error' : ''}
                  {button.isWarning ? 'btn-warning' : ''}"
                  disabled={button.disabled || button.loading}
                  on:click={() => perform_button_action(button)}
                >
                  {#if button.loading}
                    <div class="loading loading-spinner loading-sm"></div>
                  {/if}
                  {button.label || "Confirm"}
                </button>
              {/if}
            {/each}
          {/if}
        </div>
      {/if}
    </div>
  </div>
  <form
    method="dialog"
    class="modal-backdrop {blur_background ? 'backdrop-blur-sm' : ''}"
  >
    <button>close</button>
  </form>
</dialog>
