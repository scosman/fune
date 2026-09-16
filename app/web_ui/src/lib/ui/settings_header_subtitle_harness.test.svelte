<script lang="ts">
  // Test-only harness: Svelte slots cannot be passed through testing-library's
  // props, so the subtitle slot is rendered through a component that fills it.
  import SettingsHeader from "./settings_header.svelte"
  export let title: string
  export let subtitle: string | null = null
</script>

<SettingsHeader {title} {subtitle}>
  <svelte:fragment slot="subtitle">
    Planned using your <button data-subtitle-action class="link"
      >data guide</button
    >.
  </svelte:fragment>
</SettingsHeader>
