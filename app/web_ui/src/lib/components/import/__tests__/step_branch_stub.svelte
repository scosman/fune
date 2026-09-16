<script lang="ts">
  // Marker stub. Its presence in a test means the wizard advanced to the branch
  // step (i.e. the trust gate was skipped).
  export let git_url = ""
  export let pat_token: string | null = null
  export let oauth_token: string | null = null
  export let auth_mode = ""
  export let on_selected: (
    branch: string,
    path: string,
    needs_credentials: boolean,
  ) => void
</script>

<div
  data-testid="step-branch-stub"
  data-git-url={git_url}
  data-pat-token={pat_token}
  data-oauth-token={oauth_token}
  data-auth-mode={auth_mode}
  data-has-on-selected={!!on_selected}
></div>
