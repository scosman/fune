<script lang="ts">
  import { goto } from "$app/navigation"
  import InfoTooltip from "./info_tooltip.svelte"
  import type { UiProperty } from "./property_list"
  import Warning from "./warning.svelte"
  import Dialog from "./dialog.svelte"

  export let properties: UiProperty[]
  export let title: string | null = null
  // When true, links open in a new tab. Useful when the list is rendered inside
  // a form with an unsaved-changes warning, so following a link doesn't trigger
  // a same-tab navigation (and the warning popup).
  export let open_links_in_new_tab: boolean = false

  $: link_target = open_links_in_new_tab ? "_blank" : undefined
  $: link_rel = open_links_in_new_tab ? "noopener noreferrer" : undefined

  let badges_dialog: Dialog
  let modal_title = ""
  let modal_values: string[] = []
  let modal_links: (string | null)[] | undefined = undefined

  function open_badges_modal(property: UiProperty) {
    if (!Array.isArray(property.value)) {
      return
    }
    modal_title = property.name
    modal_values = property.value
    modal_links =
      property.links && property.links.length === property.value.length
        ? property.links
        : undefined
    badges_dialog?.show()
  }
</script>

<div>
  {#if title}
    <div class="text-xl font-bold mb-4">{title}</div>
  {/if}
  <div class="grid grid-cols-[auto,1fr] gap-y-2 gap-x-4 text-sm 2xl:text-base">
    {#each properties || [] as property}
      <div class="flex items-center">
        {property.name}
        {#if property.tooltip}
          <InfoTooltip tooltip_text={property.tooltip} />
        {/if}
      </div>
      <div
        class="flex items-center overflow-x-hidden flex-wrap {property.error
          ? 'text-error'
          : 'text-gray-500'}"
      >
        {#if property.warn_icon}
          <Warning
            warning_message=" "
            warning_icon="exclaim"
            warning_color="warning"
            inline={true}
          />
        {/if}
        {#if property.use_custom_slot && $$slots.custom_value}
          <slot name="custom_value" {property} />
        {:else if Array.isArray(property.value)}
          {#if property.badge}
            {@const collapse =
              property.collapse_badges && property.value.length > 1}
            <div class="flex flex-wrap gap-1 items-center">
              {#each collapse ? property.value.slice(0, 1) : property.value as value, i}
                {@const link = property.links
                  ? property.links.length == property.value.length
                    ? property.links[i]
                    : null
                  : null}
                {#if link}
                  <a
                    href={link}
                    target={link_target}
                    rel={link_rel}
                    class="badge badge-outline h-auto hover:bg-base-200"
                  >
                    {value}
                  </a>
                {:else}
                  <span class="badge badge-outline h-auto">{value}</span>
                {/if}
              {/each}
              {#if collapse}
                <button
                  class="badge badge-outline h-auto hover:bg-base-200"
                  on:click={() => open_badges_modal(property)}
                >
                  +{property.value.length - 1} more
                </button>
              {/if}
            </div>
          {:else}
            <div class="flex flex-wrap gap-1">
              {#each property.value as value, i}
                {@const link =
                  property.links &&
                  property.links.length === property.value.length
                    ? property.links[i]
                    : null}

                {#if link}
                  <a
                    href={link}
                    target={link_target}
                    rel={link_rel}
                    class="link">{value}</a
                  >
                {:else}
                  <span>{value}</span>
                {/if}
              {/each}
            </div>
          {/if}
        {:else if property.badge}
          <button
            class="badge badge-outline h-auto"
            on:click={() => {
              if (property.link) {
                goto(property.link)
              }
            }}
          >
            {property.value}
          </button>
        {:else if property.value_with_link}
          <span>
            {property.value_with_link.prefix}
            <a
              href={property.value_with_link.link}
              target={link_target}
              rel={link_rel}
              class="link"
            >
              {property.value_with_link.link_text}
            </a>
          </span>
        {:else if property.action}
          <button class="link text-left" on:click={property.action}
            >{property.value}</button
          >
        {:else if property.link}
          <a
            href={property.link}
            target={link_target}
            rel={link_rel}
            class="link">{property.value}</a
          >
        {:else}
          {property.value}
        {/if}
      </div>
    {/each}
  </div>
</div>

<Dialog
  bind:this={badges_dialog}
  title={modal_title}
  width="wide"
  action_buttons={[{ label: "Close", isCancel: true }]}
>
  <div class="flex flex-wrap gap-1 text-sm text-gray-500">
    {#each modal_values as value, i}
      {@const link =
        modal_links && modal_links.length === modal_values.length
          ? modal_links[i]
          : null}
      {#if link}
        <a
          href={link}
          target={link_target}
          rel={link_rel}
          class="badge badge-outline h-auto hover:bg-base-200"
        >
          {value}
        </a>
      {:else}
        <span class="badge badge-outline h-auto">{value}</span>
      {/if}
    {/each}
  </div>
</Dialog>
