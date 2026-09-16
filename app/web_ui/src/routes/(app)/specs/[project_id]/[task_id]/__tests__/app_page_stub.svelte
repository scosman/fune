<script lang="ts">
  export let title: string = ""
  export let subtitle: string = ""
  export let sub_subtitle: string = ""
  export let sub_subtitle_link: string | undefined = undefined
  export let limit_max_width: boolean = false
  export let action_buttons: Array<Record<string, unknown>> = []
</script>

<div
  data-testid="app-page-stub"
  data-title={title}
  data-subtitle={subtitle}
  data-sub-subtitle={sub_subtitle || undefined}
  data-sub-subtitle-link={sub_subtitle_link || undefined}
  data-limit-max-width={limit_max_width}
  data-action-button-count={action_buttons.length}
  data-action-button-labels={action_buttons
    .map((button) => button.label)
    .join(",")}
>
  <slot />
</div>
