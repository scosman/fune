<script lang="ts">
  // Generic stand-in for the run page's heavier side components. Declares the
  // bound props the page two-way-binds so those bindings resolve cleanly.
  export let selected_run_config_id: string | null = null
  export let set_default_error: unknown = null
  export let save_config_error: unknown = null
  export let selected_model_specific_run_config_id: string | null = null
  $: void [
    selected_run_config_id,
    set_default_error,
    save_config_error,
    selected_model_specific_run_config_id,
  ]
</script>

<div></div>
