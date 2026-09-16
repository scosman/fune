<script lang="ts" context="module">
  // Captures the success callback so tests can simulate a verified token.
  export const stepCredentialsProps: {
    on_success?: (token: string, auth_method: string) => void
  } = {}
</script>

<script lang="ts">
  export let git_url = ""
  export let initial_token: string | null = null
  export let on_success: (token: string, auth_method: string) => void
  stepCredentialsProps.on_success = on_success
</script>

<div
  data-testid="step-credentials-stub"
  data-git-url={git_url}
  data-initial-token={initial_token}
></div>
