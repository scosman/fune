<script lang="ts">
  import { onMount, onDestroy } from "svelte"
  import Warning from "$lib/ui/warning.svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import {
    getConfig,
    updateConfig,
    deleteConfig,
    testAccess,
    isGitHubUrl,
    isGitLabUrl,
    gitHubClassicPatDeepLink,
    gitHubFineGrainedPatDeepLink,
    gitLabPatDeepLink,
    type GitSyncConfigResponse,
  } from "$lib/git_sync/api"
  import {
    createOAuthWithInstall,
    INITIAL_STATE,
    type OAuthWithInstallFlow,
  } from "$lib/git_sync/oauth_with_install"
  import OAuthInstallStep from "$lib/git_sync/oauth_install_step.svelte"
  import PopupBlockedFallback from "$lib/git_sync/popup_blocked_fallback.svelte"

  type AuthFormMode = "oauth" | "pat"

  export let project_id: string

  let config: GitSyncConfigResponse | null = null
  let loading = true
  let error: KilnError | null = null
  let show_auth_form = false
  let new_pat_token = ""
  let saving_token = false
  let token_error: KilnError | null = null
  let removing = false
  let disable_dialog: Dialog
  let mode: AuthFormMode = "pat"

  let oauth_flow: OAuthWithInstallFlow | null = null
  let oauth: typeof INITIAL_STATE = { ...INITIAL_STATE }
  let unsubscribe_oauth: (() => void) | null = null

  function init_oauth_flow() {
    if (unsubscribe_oauth) {
      unsubscribe_oauth()
      unsubscribe_oauth = null
    }
    if (oauth_flow) {
      oauth_flow.destroy()
    }
    if (!config?.git_url) return
    oauth_flow = createOAuthWithInstall({
      git_url: config.git_url,
      on_success: async (token) => {
        config = await updateConfig(project_id, {
          oauth_token: token,
          auth_mode: "github_oauth",
        })
        close_auth_form()
      },
    })
    unsubscribe_oauth = oauth_flow.state.subscribe((s) => {
      oauth = s
    })
  }

  onMount(async () => {
    await load_config()
  })

  onDestroy(() => {
    if (unsubscribe_oauth) {
      unsubscribe_oauth()
      unsubscribe_oauth = null
    }
    if (oauth_flow) {
      oauth_flow.destroy()
    }
  })

  async function load_config() {
    try {
      loading = true
      error = null
      config = await getConfig(project_id)
    } catch (e) {
      error = createKilnError(e)
    } finally {
      loading = false
    }
  }

  function open_auth_form() {
    show_auth_form = true
    token_error = null
    new_pat_token = ""
    if (config?.git_url && isGitHubUrl(config.git_url)) {
      mode = config?.auth_mode === "github_oauth" ? "oauth" : "pat"
    } else {
      mode = "pat"
    }
    init_oauth_flow()
  }

  function close_auth_form() {
    show_auth_form = false
    token_error = null
    if (oauth_flow) {
      oauth_flow.reset()
    }
  }

  async function save_token() {
    if (!config || !new_pat_token.trim()) return
    try {
      saving_token = true
      token_error = null

      if (config.git_url) {
        const result = await testAccess(
          config.git_url,
          new_pat_token,
          "pat_token",
        )
        if (!result.success) {
          token_error = new KilnError(result.message)
          return
        }
      }

      const pat_payload: { pat_token: string; auth_mode?: string } =
        config.auth_mode === "pat_token"
          ? { pat_token: new_pat_token }
          : { pat_token: new_pat_token, auth_mode: "pat_token" }
      config = await updateConfig(project_id, pat_payload)
      close_auth_form()
      new_pat_token = ""
    } catch (e) {
      token_error = createKilnError(e)
    } finally {
      saving_token = false
    }
  }

  async function disable_sync(): Promise<boolean> {
    try {
      removing = true
      error = null
      await deleteConfig(project_id)
      config = null
      return true
    } catch (e) {
      error = createKilnError(e)
      return false
    } finally {
      removing = false
    }
  }

  $: is_github = config?.git_url ? isGitHubUrl(config.git_url) : false
  $: is_gitlab = config?.git_url ? isGitLabUrl(config.git_url) : false
  $: is_system_keys = config?.auth_mode === "system_keys"
  $: is_github_oauth = config?.auth_mode === "github_oauth"
</script>

{#if loading}
  <div class="flex items-center gap-2 py-2">
    <span class="loading loading-spinner loading-sm"></span>
    <span class="text-sm text-gray-500">Loading sync status...</span>
  </div>
{:else if config}
  <div class="border rounded-lg p-4">
    <div class="flex items-center justify-between gap-4">
      <div class="min-w-0">
        <h3 class="text-sm font-medium">Git Auto Sync</h3>
        <p class="text-xs text-gray-500 mt-1">
          Syncing with <span class="font-medium">{config.branch}</span> branch
        </p>
        {#if config.git_url}
          <p class="text-xs text-gray-400 mt-0.5 truncate max-w-sm">
            {config.git_url}
          </p>
        {/if}
      </div>
      <div class="flex flex-col gap-1.5 flex-shrink-0">
        <button
          class="btn btn-sm btn-ghost"
          on:click={() => disable_dialog.show()}
        >
          Disable
        </button>
        <button
          class="btn btn-sm btn-ghost"
          on:click={() => {
            if (show_auth_form) {
              close_auth_form()
            } else {
              open_auth_form()
            }
          }}
        >
          {show_auth_form ? "Cancel" : "Update Auth"}
        </button>
      </div>
    </div>

    {#if show_auth_form}
      <div class="mt-3 border-t pt-3">
        {#if is_system_keys}
          <div class="mt-2">
            <Warning
              warning_message="This repo was connected via system SSH keys. Either fix your SSH key connection to your git provider, or remove this project and re-add it with another auth mechanism like tokens."
              warning_color="warning"
            />
          </div>
        {:else if is_github && mode === "oauth" && oauth_flow}
          {#if oauth.needs_install}
            <OAuthInstallStep
              state={oauth}
              open_install={oauth_flow.open_install}
              verify_access={oauth_flow.verify_access}
              reset={oauth_flow.reset}
              compact
            />
          {:else}
            {#if oauth.oauth_error}
              <div class="mt-2 mb-3">
                <Warning
                  warning_message={oauth.oauth_error}
                  warning_color="error"
                />
              </div>
            {/if}

            {#if (oauth.oauth_starting || oauth.checking_access) && !oauth.oauth_error}
              {#if oauth.popup_blocked && oauth.authorize_url && !oauth.checking_access}
                <div class="flex flex-col py-2 gap-2">
                  <PopupBlockedFallback
                    url={oauth.authorize_url}
                    message="Your browser blocked the GitHub popup. Copy the link below and open it manually in a new tab to continue."
                    compact
                  />
                  <div class="text-center">
                    <button
                      class="btn btn-xs btn-ghost"
                      on:click={oauth_flow.reset}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              {:else}
                <div class="flex flex-col items-center py-6 gap-3">
                  <span class="loading loading-spinner loading-md text-primary"
                  ></span>
                  <p class="text-sm text-gray-500">
                    {oauth.checking_access
                      ? "Verifying access..."
                      : "Waiting for GitHub authorization..."}
                  </p>
                  <button
                    class="btn btn-sm btn-ghost mt-1"
                    on:click={oauth_flow.reset}
                  >
                    Cancel
                  </button>
                </div>
              {/if}
            {:else}
              <button
                class="btn btn-primary btn-sm w-full"
                on:click={oauth_flow.start}
                disabled={oauth.oauth_starting}
              >
                {#if oauth.oauth_starting}
                  <span class="loading loading-spinner loading-xs"></span>
                {/if}
                {#if oauth.oauth_error}
                  Retry with GitHub
                {:else if is_github_oauth}
                  Reconnect with GitHub
                {:else}
                  Connect with GitHub
                {/if}
              </button>
              <div class="mt-3 text-center">
                <button
                  class="btn btn-link btn-xs text-gray-500 no-underline hover:text-gray-700 hover:underline focus-visible:underline"
                  on:click={() => {
                    mode = "pat"
                    if (oauth_flow) oauth_flow.reset()
                  }}
                >
                  or use a Personal Access Token
                </button>
              </div>
            {/if}
          {/if}
        {:else}
          <label class="label" for="update_pat">
            <span class="label-text text-sm">New Personal Access Token</span>
          </label>
          <div class="flex gap-2">
            <input
              id="update_pat"
              type="password"
              class="input input-bordered input-sm flex-1"
              bind:value={new_pat_token}
              placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
            />
            <button
              class="btn btn-sm btn-primary"
              on:click={save_token}
              disabled={saving_token || !new_pat_token.trim()}
            >
              {#if saving_token}
                <span class="loading loading-spinner loading-xs"></span>
              {:else}
                Save
              {/if}
            </button>
          </div>
          {#if is_github}
            <div class="text-xs text-gray-500 mt-1">
              <a
                href={gitHubClassicPatDeepLink(config?.git_url || "")}
                target="_blank"
                rel="noopener noreferrer"
                class="link text-primary"
              >
                Generate a GitHub token</a
              >. It must have read/write access to the selected repo.
              <span class="text-gray-400"
                >You can also use <a
                  href={gitHubFineGrainedPatDeepLink(config?.git_url || "")}
                  target="_blank"
                  rel="noopener noreferrer"
                  class="link">fine-grained access tokens</a
                >, however they are harder to set up and may require approval by
                an org administrator.</span
              >
            </div>
          {:else if is_gitlab && config?.git_url}
            <div class="text-xs text-gray-500 mt-1">
              <a
                href={gitLabPatDeepLink(config.git_url)}
                target="_blank"
                rel="noopener noreferrer"
                class="link text-primary"
              >
                Generate a GitLab token</a
              >. Set expiration to at least 1 year.
            </div>
          {/if}
          {#if token_error}
            <div class="mt-2">
              <Warning
                warning_message={token_error.getMessage()}
                warning_color="error"
              />
            </div>
          {/if}
          {#if is_github}
            <div class="mt-3 text-center">
              <button
                class="btn btn-link btn-xs text-gray-500 no-underline hover:text-gray-700 hover:underline focus-visible:underline"
                on:click={() => {
                  mode = "oauth"
                  token_error = null
                }}
              >
                or connect with GitHub
              </button>
            </div>
          {/if}
        {/if}
      </div>
    {/if}
  </div>
{/if}

{#if error}
  <div class="mt-2">
    <Warning warning_message={error.getMessage()} warning_color="error" />
  </div>
{/if}

<Dialog
  bind:this={disable_dialog}
  title="Remove Git Connection"
  action_buttons={[
    {
      label: "Delete",
      isError: true,
      asyncAction: disable_sync,
      loading: removing,
    },
    {
      label: "Cancel",
      isCancel: true,
    },
  ]}
>
  <p class="text-sm">
    Connection will be removed and sync will no longer work. You'll need to
    reconnect to re-establish sync.
  </p>
</Dialog>
