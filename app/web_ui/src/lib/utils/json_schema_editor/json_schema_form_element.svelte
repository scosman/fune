<script lang="ts">
  import { empty_schema_model } from "./json_schema_templates"
  import type { SchemaModel } from "./json_schema_templates"
  import FormElement from "../form_element.svelte"

  let validation_errors: string[] = []
  let id = Math.random().toString(36)

  export let schema_model: SchemaModel = empty_schema_model()

  async function add_property() {
    schema_model.properties.push({
      title: "",
      description: "",
      type: "string",
      required: true,
    })
    // Trigger reactivity
    schema_model = schema_model
    // Scroll new item into view. Async to allow rendering first
    setTimeout(() => {
      const property = document.getElementById(
        "property_" + (schema_model.properties.length - 1) + "_" + id,
      )
      if (property) {
        property.scrollIntoView({ block: "center" })
      }
    }, 1)
  }

  function remove_property(index: number) {
    const property = schema_model.properties[index]
    const isPropertyEdited = property.title || property.description

    if (
      !isPropertyEdited ||
      confirm(
        "Are you sure you want to remove Property #" +
          (index + 1) +
          "?\n\nIt has content which hasn't been saved.",
      )
    ) {
      schema_model.properties.splice(index, 1)
      // trigger reactivity
      schema_model = schema_model
      // Move the page to the top anchor
      const list = document.getElementById(id)
      if (list) {
        // Scroll to the top of the list
        setTimeout(() => {
          list.scrollIntoView()
        }, 1)
      }
    }
  }
</script>

{#if validation_errors.length > 0}
  <div class="validation-errors">
    {#each validation_errors as error}
      <div class="text-error">{error}</div>
    {/each}
  </div>
{:else}
  <div class="flex flex-col gap-8 pt-6" {id}>
    {#each schema_model.properties as property, index}
      {#if property}
        <!-- ignore that we don't use this var-->
      {/if}

      <div class="flex flex-col gap-2">
        <div
          class="flex flex-row gap-3 font-medium text-sm pb-2"
          id={"property_" + index + "_" + id}
        >
          <div class="grow">
            Property #{index + 1}
          </div>
          <button
            class="link text-xs text-gray-500"
            on:click={() => remove_property(index)}
          >
            remove
          </button>
        </div>
        <div class="flex flex-row gap-3">
          <div class="grow">
            <FormElement
              id={"property_{name}_title"}
              label="Property Name"
              inputType="input"
              bind:value={schema_model.properties[index].title}
              light_label={true}
            />
          </div>
          <FormElement
            id={"property_{name}_type"}
            label="Type"
            inputType="select"
            bind:value={schema_model.properties[index].type}
            select_options={[
              ["string", "String"],
              ["number", "Number"],
              ["integer", "Integer"],
              ["boolean", "Boolean"],
            ]}
            light_label={true}
          />
          <FormElement
            id={"property_{name}_required"}
            label="Required"
            inputType="select"
            bind:value={schema_model.properties[index].required}
            select_options={[
              [true, "True"],
              [false, "False"],
            ]}
            light_label={true}
          />
        </div>
        <FormElement
          id={"property_{name}_description"}
          label="Description"
          inputType="input"
          bind:value={schema_model.properties[index].description}
          light_label={true}
        />
      </div>
    {/each}
    <div class="flex place-content-center">
      <button
        class="btn btn-sm"
        on:click={() => add_property()}
        id={"add_button_" + id}
      >
        Add Property
      </button>
    </div>
  </div>
{/if}
