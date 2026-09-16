<script lang="ts">
  export let warning_message: string | undefined | null = undefined
  export let markdown: boolean = false
  export let warning_color:
    | "error"
    | "warning"
    | "success"
    | "primary"
    | "gray"
    | undefined = undefined
  export let warning_icon: "exclaim" | "info" | "check" = "exclaim"
  export let large_icon: boolean = false
  // Warning owns no outer margin: the container it sits in owns the spacing
  // around it, so the same Warning reads the same in a form column, a dialog,
  // or a table cell. The one exception is `outline`, a bordered callout that
  // still carries its own bottom margin. The text indent is the one explicit variant: `inline`
  // narrows the icon-to-text gap to 4px for a warning inside a prose column
  // or a table cell; the default 16px matches a form's label-to-field rhythm.
  export let inline: boolean = false
  export let trusted: boolean = false
  export let outline: boolean = false
  export let text_size: "xs" | "sm" | "base" | "lg" = "sm"
  import MarkdownBlock from "./markdown_block.svelte"
  // Default to error if no color is provided
  $: color = warning_color || "error"
  $: text_size_class = {
    xs: "text-xs",
    sm: "text-sm",
    base: "text-base",
    lg: "text-lg",
  }[text_size]

  function html_warning_message() {
    let message = warning_message
    if (!message) {
      return ""
    }
    message = message.replace(
      /https?:\/\/\S+/g,
      '<a href="$&" class="link underline" target="_blank">$&</a>',
    )
    const paragraphs = message.split("\n")
    return paragraphs
      .map((paragraph: string) => {
        return `<p>${paragraph}</p>`
      })
      .join("")
  }

  function get_color(type: "text" | "border" = "text") {
    switch (color) {
      case "error":
        return type === "text" ? "text-error" : "border-error"
      case "warning":
        return type === "text" ? "text-warning" : "border-warning"
      case "success":
        return type === "text" ? "text-success" : "border-success"
      case "primary":
        return type === "text" ? "text-primary" : "border-primary"
      case "gray":
        return type === "text" ? "text-gray-500" : "border-gray-500"
      default:
        return ""
    }
  }

  // Held as reactive values rather than called from the markup. Svelte tracks
  // the variables a template expression names, so calling get_color() there
  // depends on the function, not on `color` — a caller whose colour resolves
  // after mount (a prop that waits on a fetch, say) would keep the colour from
  // its first render forever.
  $: text_color_class = color ? get_color("text") : ""
  $: border_color_class = color ? get_color("border") : ""
</script>

{#if warning_message}
  <div
    class="{text_size_class} text-gray-500 flex flex-row items-center {outline
      ? `border-2 ${border_color_class} rounded-lg px-4 py-2 mb-6`
      : ''}"
  >
    {#if warning_icon === "exclaim" || warning_icon === "info"}
      <svg
        class="{large_icon
          ? 'w-8 h-8'
          : 'w-5 h-5'} flex-none transition-[width,height,transform] duration-500 ease-in-out {text_color_class} {warning_icon ===
        'info'
          ? 'rotate-180'
          : 'rotate-0'}"
        style="transform-origin: center; will-change: transform, width, height, rotate;"
        fill="currentColor"
        viewBox="0 0 256 256"
        id="Flat"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M128,20.00012a108,108,0,1,0,108,108A108.12217,108.12217,0,0,0,128,20.00012Zm0,192a84,84,0,1,1,84-84A84.0953,84.0953,0,0,1,128,212.00012Zm-12-80v-52a12,12,0,1,1,24,0v52a12,12,0,1,1-24,0Zm28,40a16,16,0,1,1-16-16A16.018,16.018,0,0,1,144,172.00012Z"
        />
      </svg>
    {:else if warning_icon === "check"}
      <!-- Uploaded to: SVG Repo, www.svgrepo.com, Generator: SVG Repo Mixer Tools -->
      <svg
        class="{large_icon
          ? 'w-8 h-8'
          : 'w-5 h-5'} transition-[width,height,transform] duration-500 ease-in-out flex-none {text_color_class}"
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M16 9L10 15.5L7.5 13M12 21C16.9706 21 21 16.9706 21 12C21 7.02944 16.9706 3 12 3C7.02944 3 3 7.02944 3 12C3 16.9706 7.02944 21 12 21Z"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        />
      </svg>
    {/if}

    <div class="{inline ? 'pl-1' : 'pl-4'} flex flex-col gap-2">
      {#if markdown && trusted}
        <MarkdownBlock markdown_text={warning_message} />
      {:else if trusted}
        <!-- eslint-disable-next-line svelte/no-at-html-tags -->
        {@html html_warning_message()}
      {:else}
        {warning_message}
      {/if}
    </div>
  </div>
{/if}
