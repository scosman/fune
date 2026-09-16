<script lang="ts">
  import type { SampleDataNode } from "./gen_model"
  import IncrementUi from "$lib/ui/increment_ui.svelte"
  import { KilnError } from "$lib/utils/error_handlers"
  import { client } from "$lib/api_client"
  import { createKilnError } from "$lib/utils/error_handlers"
  import SynthDataGuidance from "./synth_data_guidance.svelte"
  import SynthDataGuide from "./synth_data_guide.svelte"
  import type { SynthDataGuidanceDataModel } from "./synth_data_guidance_datamodel"
  import posthog from "posthog-js"
  import RunConfigComponent from "$lib/ui/run_config_component/run_config_component.svelte"
  import type { RunConfigProperties } from "$lib/types"
  import { isKilnAgentRunConfig } from "$lib/types"
  import { get } from "svelte/store"

  export let guidance_data: SynthDataGuidanceDataModel
  const data_guide_store = guidance_data.data_guide
  const use_data_guide_store = guidance_data.use_data_guide
  // Local instance for dynamic reactive updates
  const selected_template = guidance_data.selected_template

  $: project_id = guidance_data.project_id
  let run_config_component: RunConfigComponent | null = null

  type GenerateSamplesOutcome = {
    topic: TopicNodeWithPath
    error: KilnError | null
  }

  interface TopicNodeWithPath {
    path: string[]
    node: SampleDataNode
  }

  export let id: string
  export let data: SampleDataNode
  export let path: string[]
  export let num_samples_to_generate: number = 8
  export let custom_topics_string: string | null = null

  /**
   * If true, generate samples for each topic leaf descendant of the current topic.
   */
  export let cascade_mode: boolean = false

  let ui_show_errors = false
  let generate_samples_outcomes: Record<string, GenerateSamplesOutcome> = {}

  $: topics_failed_to_generate = Object.values(
    generate_samples_outcomes,
  ).filter((o) => o.error)

  $: topics_failed_to_generate_count = topics_failed_to_generate.length

  export let on_completed: () => void
  export let on_partial_complete: (() => void) | null = null

  function add_synthetic_samples(
    topic: SampleDataNode,
    samples: unknown[],
    model_name: string,
    model_provider: string,
  ) {
    for (const sample of samples) {
      if (!sample) {
        continue
      }
      let input: string | null = null
      if (typeof sample == "string") {
        input = sample
      } else if (typeof sample == "object" || Array.isArray(sample)) {
        input = JSON.stringify(sample)
      }
      if (input) {
        topic.samples.push({
          input: input,
          output: null,
          saved_id: null,
          model_name,
          model_provider,
        })
      }
    }
  }

  type GenerateSampleResponse = {
    error: KilnError | null
  }

  let sample_generating: boolean = false
  async function request_generate_samples(
    topic: TopicNodeWithPath,
    run_config_properties: RunConfigProperties | null,
  ): Promise<GenerateSampleResponse> {
    try {
      if (!run_config_properties) {
        throw new KilnError("No run config properties.", null)
      }
      if (!isKilnAgentRunConfig(run_config_properties)) {
        throw new KilnError(
          "Synthetic data generation requires a kiln_agent run config.",
          null,
        )
      }
      if (!guidance_data.gen_type) {
        throw new KilnError("No generation type selected.", null)
      }
      if (
        !run_config_properties.model_name ||
        !run_config_properties.model_provider_name
      ) {
        throw new KilnError("Invalid model selected.", null)
      }
      const input_guidance = guidance_data.guidance_for_type("inputs")
      const data_guide = get(guidance_data.use_data_guide)
        ? get(guidance_data.data_guide)
        : ""
      const { data: generate_response, error: generate_error } =
        await client.POST(
          "/api/projects/{project_id}/tasks/{task_id}/generate_inputs",
          {
            body: {
              topic: topic.path,
              num_samples: num_samples_to_generate,
              run_config_properties: run_config_properties,
              guidance: input_guidance ? input_guidance : null, // clear empty string
              data_guide,
              gen_type: guidance_data.gen_type,
            },
            params: {
              path: {
                project_id: guidance_data.project_id,
                task_id: guidance_data.task_id,
              },
            },
          },
        )
      if (generate_error) {
        throw generate_error
      }
      posthog.capture("generate_synthetic_inputs", {
        num_samples: num_samples_to_generate,
        model_name: run_config_properties.model_name,
        provider: run_config_properties.model_provider_name,
        tools: run_config_properties.tools_config?.tools ?? [],
        structured_output_mode: run_config_properties.structured_output_mode,
        gen_type: guidance_data.gen_type,
      })
      const response = JSON.parse(generate_response.output.output)
      if (
        !response ||
        !response.generated_samples ||
        !Array.isArray(response.generated_samples)
      ) {
        throw new KilnError("No options returned.", null)
      }
      // Add new samples
      add_synthetic_samples(
        topic.node,
        response.generated_samples,
        run_config_properties.model_name,
        run_config_properties.model_provider_name,
      )
    } catch (e) {
      if (e instanceof KilnError) {
        return { error: e }
      }
      return { error: createKilnError(e) }
    }

    return { error: null }
  }

  /**
   * Return all descendant topic nodes in the tree (excluding the node itself).
   */
  function get_descendant_topic_nodes(
    node: TopicNodeWithPath,
  ): TopicNodeWithPath[] {
    const descendants: TopicNodeWithPath[] = []

    const stack: TopicNodeWithPath[] = [node]
    while (stack.length > 0) {
      const curr = stack.pop()!
      descendants.push(curr)

      // add children and construct their paths
      curr.node.sub_topics.forEach((n) => {
        stack.push({
          path: curr.path.concat(n.topic),
          node: n,
        })
      })
    }

    // remove the starting node from the list
    return descendants.slice(1)
  }

  /**
   * Return all leaf topic nodes in the tree (including the node itself if it is a leaf).
   */
  function collect_leaf_topic_nodes(): TopicNodeWithPath[] {
    const node_with_path = { path, node: data }
    return [node_with_path]
      .concat(get_descendant_topic_nodes(node_with_path))
      .filter((t) => t.node.sub_topics.length == 0)
  }

  async function generate_samples() {
    // Capture run config properties before modal closes and component is destroyed
    const run_config_properties =
      run_config_component?.run_options_as_run_config_properties() ?? null

    sample_generating = true
    generate_samples_outcomes = {}

    const queue = cascade_mode
      ? collect_leaf_topic_nodes()
      : [{ path, node: data }]

    // Create and start 5 workers
    // 5 because browsers can only handle 6 concurrent requests. The 6th is for the rest of the UI to keep working.
    const workers = Array(5)
      .fill(null)
      .map(() => worker(queue, run_config_properties))

    // Wait for all workers to complete
    await Promise.all(workers)

    sample_generating = false

    // Always trigger save/update so successfully generated samples are visible
    // even if some topics failed
    if (on_partial_complete) {
      on_partial_complete()
    }

    // if every topic was generated successfully, close the modal
    // otherwise we stay here to show the user the errors
    if (topics_failed_to_generate_count === 0) {
      on_completed()
    }
  }

  async function worker(
    queue: TopicNodeWithPath[],
    run_config_properties: RunConfigProperties | null,
  ) {
    while (queue.length > 0) {
      const topic = queue.shift()!
      const result = await request_generate_samples(
        topic,
        run_config_properties,
      )

      const path_serialized = serialize_topic_path(topic.path)
      if (result.error) {
        generate_samples_outcomes[path_serialized] = {
          topic,
          error: result.error,
        }
      } else {
        generate_samples_outcomes[path_serialized] = {
          topic,
          error: null,
        }
      }

      // Trigger reactivity
      generate_samples_outcomes = generate_samples_outcomes
    }
  }

  function serialize_topic_path(path: string[]) {
    return path.join("/")
  }
</script>

<dialog id={`${id}-generate-samples`} class="modal">
  <div class="modal-box">
    <form method="dialog">
      <button
        class="btn btn-sm text-xl btn-circle btn-ghost absolute right-2 top-2 focus:outline-none"
        >✕</button
      >
    </form>
    <h3 class="text-lg font-bold">Generate Inputs</h3>
    <p class="text-sm font-light mb-8">
      Generate synthetic inputs: the data that will be passed into the task.
      {#if path.length > 0}
        {cascade_mode ? "For each subtopic of: " : "For the topic: "}
        <span class="font-mono text-xs bg-gray-100">{path.join(" → ")}</span>
      {/if}
    </p>
    {#if sample_generating}
      <div class="flex flex-row justify-center">
        <div class="loading loading-spinner loading-lg my-12"></div>
      </div>
    {:else}
      <div class="flex flex-col gap-6">
        <div class="flex flex-row items-center gap-4 mt-4 mb-2">
          <div class="flex-grow font-medium text-sm">Sample Count</div>
          <IncrementUi bind:value={num_samples_to_generate} />
        </div>
        <div>
          <SynthDataGuidance guidance_type="inputs" {guidance_data} />
        </div>
        <!-- No wrapper div: SynthDataGuide renders nothing when there's no
             data guide yet, so an empty wrapper would otherwise leave a stray
             gap-6 gap in this column. -->
        <SynthDataGuide
          project_id={guidance_data.project_id}
          task_id={guidance_data.task_id}
          data_guide={$data_guide_store}
          bind:use_data_guide={$use_data_guide_store}
        />
        <RunConfigComponent
          bind:this={run_config_component}
          {project_id}
          requires_structured_output={true}
          hide_prompt_selector={true}
          show_tools_selector_in_advanced={true}
          show_name_field={false}
          model_dropdown_settings={{
            requires_data_gen: true,
            requires_uncensored_data_gen:
              guidance_data.suggest_uncensored($selected_template),
            suggested_mode: guidance_data.suggest_uncensored($selected_template)
              ? "uncensored_data_gen"
              : "data_gen",
          }}
        />

        <!-- display errors after the generation has completed -->
        {#if topics_failed_to_generate_count > 0}
          {#if !cascade_mode}
            <div class="text-error font-light text-sm mt-4">
              Failed to generate samples for {topics_failed_to_generate[0].topic.path.join(
                " → ",
              )}. Running again may resolve transient issues.
              <div>
                {topics_failed_to_generate[0].error?.getMessage()}
              </div>
            </div>
          {:else}
            <div class="text-error font-light text-sm mt-4">
              {topics_failed_to_generate_count} topics failed. Running again may
              resolve transient issues.
              <button
                class="link"
                on:click={() => (ui_show_errors = !ui_show_errors)}
              >
                {ui_show_errors ? "Hide Errors" : "Show Errors"}
              </button>
            </div>
            <div
              class="flex flex-col gap-2 mt-4 text-xs text-error {ui_show_errors
                ? ''
                : 'hidden'}"
            >
              {#each topics_failed_to_generate as outcome}
                <div>
                  {outcome.topic.path.join(" → ")}: {outcome.error?.getMessage()}
                </div>
              {/each}
            </div>
          {/if}
        {/if}

        <button
          class="btn mt-6 {custom_topics_string ? '' : 'btn-primary'}"
          on:click={generate_samples}
        >
          Generate {num_samples_to_generate} Model Inputs
          {#if cascade_mode}
            For Each Topic
          {/if}
        </button>
      </div>
    {/if}
  </div>
  <form method="dialog" class="modal-backdrop">
    <button>close</button>
  </form>
</dialog>
