<script lang="ts">
  import type { Trace, TraceMessage, ToolCallMessageParam } from "$lib/types"
  import Output from "$lib/ui/output.svelte"
  import ArrowRightUpIcon from "../icons/arrow_right_up_icon.svelte"
  import ToolCall from "./tool_call.svelte"
  import ToolMessagesDialog from "./tool_messages_dialog.svelte"

  export let trace: Trace
  export let project_id: string | undefined = undefined
  // Indices into `trace` whose messages start expanded. Default empty, which
  // is every message collapsed — a long trace opens as a scannable list of
  // rows and the reader expands what they want. A caller passes indices when
  // one specific message is the thing being read and should cost no click.
  export let auto_expand_indices: number[] = []

  // Track collapsed state for each message (true = expanded, false = collapsed).
  // Built ONCE, from the trace this component mounted with: a reader's clicks
  // own the state from then on, and a parent that reassigns a structurally
  // identical trace (the run page does on save) must not close what they
  // opened. A caller that needs the expansion re-applied for a different
  // trace remounts this component — see the review surface's {#key}.
  let messageExpanded: boolean[] = trace.map((_, index) =>
    auto_expand_indices.includes(index),
  )

  function getRoleDisplayName(role: string): string {
    return (
      {
        system: "System",
        user: "User",
        assistant: "Assistant",
        tool: "Tool",
        function: "Function",
        developer: "Developer",
      }[role] || role
    )
  }

  function getMessagePreview(message: TraceMessage): string {
    let content = content_from_message(message)
    let truncated_content = content
    // Truncate to keep DOM reasonable for unexpanded messages. The CSS separately truncates and adds ellipsis
    if (content && content.length > 200) {
      truncated_content = content.slice(0, 200)
    }
    let tool_calls = tool_calls_from_message(message)
    let reasoning_content = reasoning_content_from_message(message)

    // Tool result
    if (message.role === "tool" && content) {
      if (is_tool_error(message)) {
        return "Tool error: " + truncated_content
      }
      return "Tool result: " + truncated_content
    }

    // Mixed content
    let content_types = []
    if (tool_calls) {
      content_types.push("tool calls")
    }
    if (content) {
      content_types.push("message")
    }
    if (reasoning_content) {
      content_types.push("reasoning")
    }
    if (content_types.length > 1) {
      return "Multi-content: " + content_types.join(", ")
    }

    // Typical content message - just show the content
    if (truncated_content) {
      return truncated_content
    }

    // Tool calls - show the number of tool calls
    if (tool_calls && tool_calls.length > 0) {
      if (tool_calls.length === 1) {
        return `Requested tool call: ${tool_calls[0].function.name}`
      }
      return `Requested ${tool_calls.length} tool calls`
    }

    // Fallback
    return "Unrenderable message"
  }

  function tool_calls_from_message(
    message: TraceMessage,
  ): ToolCallMessageParam[] | undefined {
    if (
      "tool_calls" in message &&
      message.tool_calls &&
      message.tool_calls.length > 0
    ) {
      return message.tool_calls
    }
    return undefined
  }

  function is_tool_error(message: TraceMessage): boolean {
    if (message.role === "tool" && "is_error" in message && message.is_error) {
      return true
    }
    if (
      message.role === "tool" &&
      "content" in message &&
      typeof message.content === "string"
    ) {
      try {
        const parsed = JSON.parse(message.content)
        if (parsed && typeof parsed === "object" && parsed.isError === true) {
          return true
        }
      } catch (_) {
        // Not JSON
      }
    }
    return false
  }

  function content_from_message(message: TraceMessage) {
    if (
      "content" in message &&
      message.content &&
      typeof message.content === "string"
    ) {
      // For Kiln task tools, extract just the output field from the JSON response
      if (message.role === "tool") {
        try {
          const parsed = JSON.parse(message.content)
          if (parsed && typeof parsed === "object" && "output" in parsed) {
            return parsed.output
          }
          if (
            parsed &&
            typeof parsed === "object" &&
            parsed.isError === true &&
            "error" in parsed
          ) {
            return parsed.error
          }
        } catch (_) {
          // Content is not JSON, return as-is
        }
      }
      return message.content
    }
    return undefined
  }

  function origin_tool_call_by_id(tool_call_id: string) {
    for (const message of trace) {
      const tool_calls = tool_calls_from_message(message)
      if (tool_calls) {
        for (const tool_call of tool_calls) {
          if (tool_call.id === tool_call_id) {
            return tool_call
          }
        }
      }
    }
    return undefined
  }

  function reasoning_content_from_message(
    message: TraceMessage,
  ): string | undefined {
    if ("reasoning_content" in message && message.reasoning_content) {
      return message.reasoning_content
    }
    return undefined
  }

  function kiln_task_tool_data_from_message(message: TraceMessage): {
    project_id: string
    tool_id: string
    task_id: string
    run_id: string
  } | null {
    if ("kiln_task_tool_data" in message && message.kiln_task_tool_data) {
      const [project_id, tool_id, task_id, run_id] =
        message.kiln_task_tool_data.split(":::")
      if (project_id && tool_id && task_id && run_id) {
        return {
          project_id,
          tool_id,
          task_id,
          run_id,
        }
      } else {
        console.warn(
          "Invalid kiln task tool data format. Expected format: <project_id>,<tool_id>,<task_id>,<run_id>",
          message.kiln_task_tool_data,
        )
        return null
      }
    } else {
      return null
    }
  }

  let tool_messages_dialog: ToolMessagesDialog | null = null
</script>

<div class="flex flex-col gap-3 w-full">
  {#each trace as message, index}
    <!-- Message Bubble -->
    <div class="">
      <div class="collapse collapse-arrow bg-base-200 rounded-lg">
        <input
          type="checkbox"
          class="peer"
          bind:checked={messageExpanded[index]}
        />
        <div
          class="collapse-title flex items-center justify-between cursor-pointer min-w-0"
          role="presentation"
        >
          <span class="font-medium text-xs min-w-[80px] text-gray-500 uppercase"
            >{getRoleDisplayName(message.role)}</span
          >
          <!-- Collapsed Preview -->
          <div
            class="px-2 text-sm text-gray-600 overflow-hidden text-ellipsis whitespace-nowrap grow {messageExpanded[
              index
            ]
              ? 'hidden'
              : ''}"
          >
            {getMessagePreview(message)}
          </div>
        </div>

        <div class="collapse-content">
          <!-- Don't render unless they expand. There's a lot of content here in a long list. -->
          {#if messageExpanded[index]}
            {@const tool_calls = tool_calls_from_message(message)}
            {@const content = content_from_message(message)}
            {@const reasoning_content = reasoning_content_from_message(message)}
            <!-- Expanded View -->
            <div class="flex flex-col gap-3">
              {#if tool_calls}
                <div>
                  <div class="text-xs text-gray-500 font-bold mb-1">
                    Requested Tool Calls
                  </div>
                  <div class="flex flex-col gap-2">
                    {#each tool_calls as tool_call, index}
                      <ToolCall
                        {tool_call}
                        {project_id}
                        nameTag={tool_calls.length > 1
                          ? `Tool Call #${index + 1}`
                          : "Tool Call"}
                      />
                    {/each}
                  </div>
                </div>
              {/if}
              {#if reasoning_content}
                <div>
                  <div class="text-xs text-gray-500 font-bold mb-1">
                    Reasoning
                  </div>
                  <Output raw_output={reasoning_content} no_padding={true} />
                </div>
              {/if}

              <!-- Message content cases -->
              {#if content && message.role === "tool"}
                {@const origin_tool_call = origin_tool_call_by_id(
                  message.tool_call_id,
                )}
                {@const kiln_task_tool_data =
                  kiln_task_tool_data_from_message(message)}
                {@const tool_error = is_tool_error(message)}
                {#if origin_tool_call}
                  <div>
                    <div class="text-xs text-gray-500 font-bold mb-1">
                      Invoked Tool Call
                    </div>
                    <ToolCall
                      tool_call={origin_tool_call}
                      {project_id}
                      persistent_tool_id={kiln_task_tool_data?.tool_id}
                    />
                  </div>
                {/if}
                <div>
                  <div
                    class="text-xs font-bold mb-1 {tool_error
                      ? 'text-error'
                      : 'text-gray-500'}"
                  >
                    {tool_error ? "Tool Error" : "Tool Result"}
                  </div>
                  <div
                    class={tool_error
                      ? "border border-error/20 rounded-lg p-2"
                      : ""}
                  >
                    <Output raw_output={content} no_padding={true} />
                  </div>
                </div>
                {#if kiln_task_tool_data}
                  <div>
                    <button
                      class="link text-xs text-gray-500"
                      on:click={() => {
                        tool_messages_dialog?.show(kiln_task_tool_data)
                      }}
                    >
                      <div class="flex flex-row items-center gap-1">
                        <span>Subtask Message Trace</span>
                        <div class="w-4 h-4">
                          <ArrowRightUpIcon />
                        </div>
                      </div>
                    </button>
                  </div>
                {/if}
              {:else if content}
                <div>
                  <!-- Header logic: skip if only a message, just for a cleaner ui -->
                  {#if tool_calls || reasoning_content}
                    <div class="text-xs text-gray-500 font-bold mb-1">
                      Content
                    </div>
                  {/if}
                  <Output raw_output={content} no_padding={true} />
                </div>
              {/if}
            </div>
          {/if}
        </div>
      </div>
    </div>
  {/each}
</div>

<ToolMessagesDialog bind:this={tool_messages_dialog} {project_id} />
