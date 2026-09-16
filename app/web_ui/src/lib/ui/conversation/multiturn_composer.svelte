<script lang="ts">
  import { tick } from "svelte"
  import { createKilnError, KilnError } from "$lib/utils/error_handlers"
  import FormContainer from "$lib/utils/form_container.svelte"
  import RunInputForm from "../../../routes/(app)/run/run_input_form.svelte"
  import Dialog from "$lib/ui/dialog.svelte"
  import ForkIcon from "$lib/ui/icons/fork_icon.svelte"
  import {
    send_multiturn,
    type RunConfigController,
  } from "$lib/services/multiturn_send"
  import type { TaskRun } from "$lib/types"

  export let mode: "append" | "fork" = "append"
  export let project_id: string
  export let task_id: string
  export let parent_task_run_id: string | null
  // The run config component the page renders elsewhere (typically in the
  // right-hand Options column). The composer dispatches its Send through
  // this controller via send_multiturn.
  export let run_config_component: RunConfigController | null = null
  // Fork-mode only.
  export let prefill_text: string = ""
  export let forked_turn_index: number | undefined = undefined
  // Fork-mode only. True when earlier turns couldn't be recovered, so
  // forked_turn_index counts from the recovered portion rather than the
  // conversation's true start. Keeps the turn label from reading as absolute.
  export let chain_broken: boolean = false
  export let on_success: (
    new_run_id: string,
    created_run?: TaskRun,
  ) => void | Promise<void>
  export let on_cancel: (() => void) | undefined = undefined
  // Append-mode optimistic hooks: on_send_start fires with the message text
  // the moment a send begins (so the parent can show it in the transcript
  // right away), and on_send_settled fires once the send resolves or fails
  // (ok=false means it errored). on success the parent keeps the composer
  // busy until the new run renders.
  export let on_send_start: ((text: string) => void) | undefined = undefined
  export let on_send_settled: ((ok: boolean) => void) | undefined = undefined
  // Externally-driven busy state: keeps the composer disabled after a
  // successful send while the parent navigates to / loads the new run, so the
  // user can't fire a second send against the stale parent run.
  export let busy: boolean = false
  // When true, sending with no parent_task_run_id starts a new root
  // conversation (the first turn). Used by the /run page.
  export let allow_root_turn: boolean = false

  // forked_turn_index is required when mode is "fork" — it's used for both
  // the context-strip heading and dirty-tracking baseline. Callers always
  // supply it (see fork_target_from_user_block). If they don't, surface the
  // bug rather than silently rendering "Forking turn undefined".
  $: if (mode === "fork" && forked_turn_index === undefined) {
    throw new Error(
      "MultiturnComposer: forked_turn_index is required when mode is 'fork'",
    )
  }

  let input_form: RunInputForm
  let submitting = false
  let run_error: KilnError | null = null
  let discard_dialog: Dialog | null = null
  // Used by the dirty-confirm swap flow (see request_swap). When set, the
  // confirmation dialog runs this callback on Discard instead of on_cancel.
  let pending_swap_callback: (() => void) | null = null

  // The baseline text used to detect "dirty" edits when the user clicks
  // Cancel. Tracks the most-recently seeded prefill, so swapping the fork
  // target re-bases dirty detection to the new turn's text.
  let baseline_text = prefill_text

  // Re-seed the textarea whenever the fork target changes (or the composer
  // first gets a reference to the child form). The seed key combines the
  // identity of the input_form reference with the (turn_index, prefill_text)
  // pair that defines the current target. Any change — a new turn, a
  // different prefill, or a freshly-mounted child form — produces a new
  // tuple and triggers a re-seed (and re-bases the dirty baseline).
  let seeded_key: [RunInputForm, number | undefined, string] | null = null
  $: current_seed_key = input_form
    ? ([input_form, forked_turn_index, prefill_text] as [
        RunInputForm,
        number | undefined,
        string,
      ])
    : null
  $: if (mode === "fork" && current_seed_key) {
    if (
      !seeded_key ||
      seeded_key[0] !== current_seed_key[0] ||
      seeded_key[1] !== current_seed_key[1] ||
      seeded_key[2] !== current_seed_key[2]
    ) {
      seed_prefill(current_seed_key)
    }
  }

  function seed_prefill(key: [RunInputForm, number | undefined, string]) {
    if (typeof input_form?.set_plaintext_input !== "function") return
    input_form.set_plaintext_input(prefill_text)
    baseline_text = prefill_text
    seeded_key = key
    // Schedule focus after Svelte flushes the bind:value to the DOM. The
    // child form exposes a focus helper so we don't reach into its markup.
    tick().then(() => {
      input_form?.focus_plaintext_input?.()
    })
  }

  function current_text(): string {
    return input_form?.get_plaintext_input_data() ?? ""
  }

  // True when the user's typed text differs from the most-recently seeded
  // prefill. Exposed via `bind:this` so the parent page can gate fork-target
  // swaps (and prompt for discard if dirty).
  export function is_dirty(): boolean {
    return current_text() !== baseline_text
  }

  // Called by the parent when the user clicks fork on a different user
  // block while this composer is already open. If clean, runs `on_proceed`
  // immediately; if dirty, opens the discard dialog and runs `on_proceed`
  // only if the user clicks Discard.
  export function request_swap(on_proceed: () => void): void {
    if (!is_dirty()) {
      on_proceed()
      return
    }
    pending_swap_callback = on_proceed
    discard_dialog?.show()
  }

  async function handle_submit() {
    run_error = null
    submitting = true
    // Capture the text, surface it optimistically, and clear the textarea
    // immediately so the composer reads as "sent". The text is passed to
    // send_multiturn explicitly so clearing doesn't drop the in-flight
    // message; on failure we put it back so the user can retry.
    const text = current_text()
    on_send_start?.(text)
    input_form?.clear_input?.()
    let ok = false
    try {
      const result = await send_multiturn({
        project_id,
        task_id,
        parent_task_run_id,
        run_config_component,
        input_form,
        on_success,
        allow_root_turn,
        plaintext: text,
      })
      if (result.ok) {
        ok = true
      } else {
        run_error = createKilnError(result.error)
      }
    } catch (e) {
      // on_success threw (e.g. goto/load_run failed).
      run_error = createKilnError(e)
    } finally {
      if (!ok) {
        // Restore the message so it isn't lost, and surface the failure.
        input_form?.set_plaintext_input?.(text)
        if (!run_error) {
          run_error = createKilnError(new Error("Failed to send message."))
        }
      }
      on_send_settled?.(ok)
      submitting = false
      await tick()
    }
  }

  function handle_cancel_click() {
    if (!on_cancel) return
    if (!is_dirty()) {
      on_cancel()
      return
    }
    pending_swap_callback = null
    discard_dialog?.show()
  }

  function confirm_discard(): boolean {
    // If a pending swap is queued, route the Discard click into the swap
    // callback; otherwise fall through to the Cancel handler.
    const swap = pending_swap_callback
    pending_swap_callback = null
    if (swap) {
      swap()
    } else {
      on_cancel?.()
    }
    return true
  }
</script>

<div data-testid="multiturn-composer" class="flex flex-col gap-4">
  {#if mode === "fork"}
    <!-- Status header for the composer, not a chat bubble. The fork
         affordance is metadata about what Send will do, so it's styled as a
         label + helper text aligned to the input below — the filled rounded
         bubble treatment is reserved for actual messages. -->
    <div
      data-testid="multiturn-fork-context-strip"
      class="flex flex-col gap-0.5"
    >
      <div class="flex flex-row items-center gap-2 text-sm font-medium">
        <span class="w-4 h-4 text-gray-500 flex-none"><ForkIcon /></span>
        <span>
          {#if chain_broken}
            Forking turn {forked_turn_index} of the recovered conversation
          {:else}
            Forking turn {forked_turn_index}
          {/if}
        </span>
      </div>
      <p class="text-xs text-gray-500">
        Your next message will start a new conversation branch from this point.
        The original conversation is preserved unchanged in your dataset.
      </p>
    </div>
  {/if}
  <!-- We render two near-identical FormContainer branches because Svelte 4
       requires <svelte:fragment slot=...> to be a direct child of the
       component (no surrounding {#if}). FormContainer also keys its layout
       off `$$slots.submit_left` which is truthy whenever the slot is
       provided at all — even if its content renders nothing — so we can't
       collapse via an empty-conditional slot without regressing append-mode
       layout (full-width button vs. fork-mode's right-aligned button next
       to Cancel). The cancel-button markup is the only difference. While
       submitting, FormContainer disables/spinners the Send button and we
       disable the textarea — the area stays put rather than being hidden. -->
  {#if mode === "fork" && on_cancel}
    <FormContainer
      submit_label="Send"
      on:submit={handle_submit}
      bind:error={run_error}
      bind:submitting
      primary={true}
      keyboard_submit={true}
      focus_on_mount={false}
      gap={4}
    >
      <div data-testid="multiturn-composer-input">
        <RunInputForm
          bind:this={input_form}
          input_schema={null}
          label="Message"
          placeholder="Write a message…"
          hide_label={true}
          disabled={submitting}
          height="compact"
        />
      </div>
      <svelte:fragment slot="submit_left">
        <button
          type="button"
          class="btn"
          data-testid="multiturn-composer-cancel"
          on:click={handle_cancel_click}
          disabled={submitting}
        >
          Cancel
        </button>
      </svelte:fragment>
    </FormContainer>
  {:else}
    <FormContainer
      submit_label="Send"
      on:submit={handle_submit}
      bind:error={run_error}
      bind:submitting
      submit_disabled={busy}
      primary={true}
      keyboard_submit={!busy}
      focus_on_mount={false}
      gap={4}
    >
      <div data-testid="multiturn-composer-input">
        <RunInputForm
          bind:this={input_form}
          input_schema={null}
          label="Message"
          placeholder="Write a message…"
          hide_label={true}
          disabled={submitting || busy}
          height="compact"
        />
      </div>
    </FormContainer>
  {/if}
</div>

<Dialog
  bind:this={discard_dialog}
  title="Discard your changes?"
  action_buttons={[
    { label: "Keep editing", isCancel: true },
    { label: "Discard", isError: true, action: confirm_discard },
  ]}
>
  <p class="text-sm">
    The text you entered will be lost. The original message on the parent run is
    preserved either way.
  </p>
</Dialog>
