<script lang="ts">
  // Test-only harness: Svelte slots cannot be passed through testing-library's
  // props, so the slotted case is rendered through a component that fills it.
  import Output from "./output.svelte"
  export let raw_output: string
  export let max_height: string | null = null
  export let no_padding: boolean = false
</script>

<Output {raw_output} {max_height} {no_padding}>
  <span data-slotted>rendered content</span>
</Output>
