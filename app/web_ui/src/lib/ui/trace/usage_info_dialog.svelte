<script lang="ts">
  import Dialog from "$lib/ui/dialog.svelte"
  import PropertyList from "$lib/ui/property_list.svelte"
  import type { UiProperty } from "$lib/ui/property_list"
  import type { components } from "$lib/api_schema"
  import { formatLatency } from "$lib/utils/formatters"

  type MessageUsage = components["schemas"]["MessageUsage"]

  let dialog: Dialog | null = null
  let usage: MessageUsage | null = null
  let latency_ms: number | null = null

  export function show(payload: {
    usage: MessageUsage | null
    latency_ms: number | null
  }) {
    usage = payload.usage
    latency_ms = payload.latency_ms
    dialog?.show()
  }

  function build_properties(
    u: MessageUsage | null,
    lat: number | null,
  ): UiProperty[] {
    const props: UiProperty[] = []
    if (u) {
      if (typeof u.cost === "number") {
        props.push({ name: "Cost", value: `$${u.cost.toFixed(6)}` })
      }
      if (typeof u.total_tokens === "number") {
        props.push({ name: "Total tokens", value: u.total_tokens })
      }
      if (typeof u.input_tokens === "number") {
        props.push({ name: "Input tokens", value: u.input_tokens })
      }
      if (typeof u.output_tokens === "number") {
        props.push({ name: "Output tokens", value: u.output_tokens })
      }
      if (typeof u.cached_tokens === "number") {
        props.push({ name: "Cached tokens", value: u.cached_tokens })
      }
    }
    if (lat !== null) {
      props.push({ name: "Latency", value: formatLatency(lat) })
    }
    return props
  }

  $: properties = build_properties(usage, latency_ms)
</script>

<Dialog bind:this={dialog} title="Usage">
  {#if properties.length > 0}
    <PropertyList {properties} />
  {:else}
    <div class="text-gray-500 text-sm py-2">No usage data recorded.</div>
  {/if}
</Dialog>
