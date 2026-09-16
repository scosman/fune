<script lang="ts">
  // Test-only harness: the subtitle slot on the branch that also carries
  // actions, which the control lays out with its own markup.
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
  <svelte:fragment slot="actions">
    <button data-action>Refine Plan</button>
  </svelte:fragment>
</SettingsHeader>
