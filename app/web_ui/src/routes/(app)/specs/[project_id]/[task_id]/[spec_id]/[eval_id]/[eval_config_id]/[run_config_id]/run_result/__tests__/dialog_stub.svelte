<script lang="ts">
  // Stub for $lib/ui/dialog.svelte: jsdom does not implement <dialog>.showModal(),
  // so the stub tracks open state itself. The real dialog's slot is inside a
  // <dialog> the user cannot see until show(), so the stub only renders its
  // content once opened — a test that forgets to open it sees nothing.
  export let title: string = ""
  export let blur_background: boolean = false
  export let action_buttons: Array<Record<string, unknown>> = []
  export let width: string = "normal"
  // Referenced so svelte-check does not flag them unused; the page passes them.
  void blur_background
  void action_buttons
  void width
  let is_open = false
  export function show() {
    is_open = true
  }
  export function close() {
    is_open = false
  }
</script>

<div data-testid="dialog-stub" data-title={title} data-open={is_open}>
  {#if is_open}
    <slot />
  {/if}
</div>
