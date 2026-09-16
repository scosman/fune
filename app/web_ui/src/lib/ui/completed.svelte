<script lang="ts">
  import CheckmarkIcon from "./icons/checkmark_icon.svelte"
  import Intro from "./intro.svelte"

  export let title: string
  export let subtitle: string
  export let link: string
  export let button_text: string
</script>

<div class="flex flex-col items-center justify-center min-h-[60vh]">
  <Intro
    {title}
    description_paragraphs={[subtitle]}
    action_buttons={[
      {
        label: button_text,
        href: link,
        is_primary: true,
      },
    ]}
  >
    <div slot="icon">
      <div class="h-12 w-12 text-success">
        <CheckmarkIcon />
      </div>
    </div>
  </Intro>
</div>
