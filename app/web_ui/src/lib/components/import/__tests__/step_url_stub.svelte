<script lang="ts" context="module">
  // Captures the callbacks import_project passes so tests can drive the url
  // step's success paths without any real network/OAuth.
  export const stepUrlProps: {
    on_success?: (url: string, auth_method: string) => void
    on_auth_required?: (url: string) => void
  } = {}
</script>

<script lang="ts">
  export let initial_url = ""
  export let pat_token: string | null = null
  export let on_success: (url: string, auth_method: string) => void
  export let on_auth_required: (url: string) => void
  stepUrlProps.on_success = on_success
  stepUrlProps.on_auth_required = on_auth_required
</script>

<div
  data-testid="step-url-stub"
  data-initial-url={initial_url}
  data-pat-token={pat_token}
></div>
