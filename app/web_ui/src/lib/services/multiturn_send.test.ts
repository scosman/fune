// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from "vitest"

const postMock = vi.fn()
vi.mock("$lib/api_client", () => ({
  client: {
    POST: (...args: unknown[]) => postMock(...args),
  },
  base_url: "http://test:8000",
}))

import { send_multiturn } from "./multiturn_send"
import type { RunConfigController, InputFormController } from "./multiturn_send"
import type { RunConfigProperties } from "$lib/types"

type RunConfigMock = {
  clear_run_options_errors: ReturnType<typeof vi.fn>
  clear_model_dropdown_error: ReturnType<typeof vi.fn>
  set_model_dropdown_error: ReturnType<typeof vi.fn>
  run_options_as_run_config_properties: ReturnType<typeof vi.fn>
  get_selected_model: ReturnType<typeof vi.fn>
}

type InputFormMock = {
  get_plaintext_input_data: ReturnType<typeof vi.fn>
  clear_input: ReturnType<typeof vi.fn>
}

function makeRunConfig(
  overrides: {
    properties?: RunConfigProperties
    selected_model?: string | null
  } = {},
): RunConfigMock {
  const properties: RunConfigProperties =
    overrides.properties ??
    ({
      type: "kiln_agent",
      model_provider_name: "openai",
      model_name: "gpt-4o",
      prompt_id: "simple_prompt_builder",
      temperature: 1,
      top_p: 1,
      structured_output_mode: "default",
      thinking_level: null,
      tools_config: { tools: [] },
    } as unknown as RunConfigProperties)
  const selected_model =
    overrides.selected_model === undefined
      ? "openai/gpt-4o"
      : overrides.selected_model
  return {
    clear_run_options_errors: vi.fn(),
    clear_model_dropdown_error: vi.fn(),
    set_model_dropdown_error: vi.fn(),
    run_options_as_run_config_properties: vi.fn().mockReturnValue(properties),
    get_selected_model: vi.fn().mockReturnValue(selected_model),
  }
}

function asRunConfigController(m: RunConfigMock): RunConfigController {
  return m as unknown as RunConfigController
}

function makeInputForm(text: string | null = "hello there"): InputFormMock {
  return {
    get_plaintext_input_data: vi.fn().mockReturnValue(text),
    clear_input: vi.fn(),
  }
}

function asInputFormController(m: InputFormMock): InputFormController {
  return m as unknown as InputFormController
}

beforeEach(() => {
  postMock.mockReset()
})

describe("send_multiturn", () => {
  it("posts parent_task_run_id matching the leaf run id, plaintext_input, and tags", async () => {
    postMock.mockResolvedValue({ data: { id: "new-run-99" }, error: null })
    const on_success = vi.fn()
    const run_config = makeRunConfig()
    const input_form = makeInputForm("hi there")

    const result = await send_multiturn({
      project_id: "proj-1",
      task_id: "task-1",
      parent_task_run_id: "leaf-42",
      run_config_component: asRunConfigController(run_config),
      input_form: asInputFormController(input_form),
      on_success,
    })

    expect(result).toEqual({ ok: true, new_run_id: "new-run-99" })
    expect(postMock).toHaveBeenCalledTimes(1)
    const [path, opts] = postMock.mock.calls[0]
    expect(path).toBe("/api/projects/{project_id}/tasks/{task_id}/run")
    const body = (opts as { body: Record<string, unknown> }).body
    expect(body.parent_task_run_id).toBe("leaf-42")
    expect(body.plaintext_input).toBe("hi there")
    expect(body.tags).toEqual(["manual_run"])
    expect(body.structured_input).toBeNull()
    expect(body.run_config_properties).toBeDefined()
    expect(on_success).toHaveBeenCalledWith("new-run-99", { id: "new-run-99" })
  })

  it("forwards custom tags when provided", async () => {
    postMock.mockResolvedValue({ data: { id: "r-2" }, error: null })
    await send_multiturn({
      project_id: "p",
      task_id: "t",
      parent_task_run_id: "leaf",
      run_config_component: asRunConfigController(makeRunConfig()),
      input_form: asInputFormController(makeInputForm("x")),
      on_success: vi.fn(),
      tags: ["custom_tag"],
    })
    const body = (
      postMock.mock.calls[0][1] as { body: Record<string, unknown> }
    ).body
    expect(body.tags).toEqual(["custom_tag"])
  })

  it("calls on_success with the new run id then clears the input on success", async () => {
    postMock.mockResolvedValue({ data: { id: "new-run-99" }, error: null })
    const calls: string[] = []
    const on_success = vi.fn(async () => {
      calls.push("on_success")
    })
    const input_form = makeInputForm()
    input_form.clear_input = vi.fn().mockImplementation(() => {
      calls.push("clear_input")
    })
    const run_config = makeRunConfig()

    await send_multiturn({
      project_id: "proj-1",
      task_id: "task-1",
      parent_task_run_id: "leaf-42",
      run_config_component: asRunConfigController(run_config),
      input_form: asInputFormController(input_form),
      on_success,
    })

    expect(calls).toEqual(["on_success", "clear_input"])
  })

  it("does not throw when on_success unmounts the form before clear_input runs", async () => {
    postMock.mockResolvedValue({ data: { id: "new-run-99" }, error: null })
    const input_form = makeInputForm()
    // Simulate the real case: on_success sets `run = null` and navigates, which
    // unmounts the bound RunInputForm. Svelte may leave the bind:this reference
    // pointing at a stale object whose methods are gone.
    const on_success = vi.fn(async () => {
      // @ts-expect-error simulate the unmount: clear_input is no longer a fn
      input_form.clear_input = undefined
    })

    const result = await send_multiturn({
      project_id: "p",
      task_id: "t",
      parent_task_run_id: "leaf",
      run_config_component: asRunConfigController(makeRunConfig()),
      input_form: asInputFormController(input_form),
      on_success,
    })

    expect(result.ok).toBe(true)
    expect(on_success).toHaveBeenCalledWith("new-run-99", { id: "new-run-99" })
  })

  it("returns ok:false and does NOT clear input when the API returns an error", async () => {
    postMock.mockResolvedValue({ data: null, error: { message: "boom" } })
    const input_form = makeInputForm("preserved text")
    const on_success = vi.fn()

    const result = await send_multiturn({
      project_id: "p",
      task_id: "t",
      parent_task_run_id: "leaf",
      run_config_component: asRunConfigController(makeRunConfig()),
      input_form: asInputFormController(input_form),
      on_success,
    })

    expect(result.ok).toBe(false)
    if (!result.ok) {
      expect(result.error).toEqual({ message: "boom" })
    }
    expect(input_form.clear_input).not.toHaveBeenCalled()
    expect(on_success).not.toHaveBeenCalled()
  })

  it("preserves the input text when on_success throws (e.g. goto/load_run fails)", async () => {
    postMock.mockResolvedValue({ data: { id: "new-run-99" }, error: null })
    const input_form = makeInputForm("still here")
    const on_success = vi.fn(async () => {
      throw new Error("goto failed")
    })

    await expect(
      send_multiturn({
        project_id: "p",
        task_id: "t",
        parent_task_run_id: "leaf",
        run_config_component: asRunConfigController(makeRunConfig()),
        input_form: asInputFormController(input_form),
        on_success,
      }),
    ).rejects.toThrow("goto failed")
    expect(input_form.clear_input).not.toHaveBeenCalled()
  })

  it("rejects when parent_task_run_id is missing — does not POST", async () => {
    const input_form = makeInputForm()
    const on_success = vi.fn()
    const run_config = makeRunConfig()

    const result = await send_multiturn({
      project_id: "p",
      task_id: "t",
      parent_task_run_id: null,
      run_config_component: asRunConfigController(run_config),
      input_form: asInputFormController(input_form),
      on_success,
    })

    expect(result.ok).toBe(false)
    expect(postMock).not.toHaveBeenCalled()
    expect(input_form.clear_input).not.toHaveBeenCalled()
    expect(on_success).not.toHaveBeenCalled()
  })

  it("rejects when run_config_component is missing", async () => {
    const result = await send_multiturn({
      project_id: "p",
      task_id: "t",
      parent_task_run_id: "leaf",
      run_config_component: null,
      input_form: asInputFormController(makeInputForm()),
      on_success: vi.fn(),
    })
    expect(result.ok).toBe(false)
    expect(postMock).not.toHaveBeenCalled()
  })

  it("flags the model dropdown error and rejects when no model is selected (kiln_agent)", async () => {
    const run_config = makeRunConfig({ selected_model: null })
    run_config.get_selected_model = vi.fn().mockReturnValue(null)
    const result = await send_multiturn({
      project_id: "p",
      task_id: "t",
      parent_task_run_id: "leaf",
      run_config_component: asRunConfigController(run_config),
      input_form: asInputFormController(makeInputForm()),
      on_success: vi.fn(),
    })
    expect(result.ok).toBe(false)
    expect(run_config.set_model_dropdown_error).toHaveBeenCalledWith("Required")
    expect(postMock).not.toHaveBeenCalled()
  })

  it("allows MCP run configs to send without a selected model", async () => {
    postMock.mockResolvedValue({ data: { id: "r-3" }, error: null })
    const mcpProps = {
      type: "mcp",
      tool_reference: { tool_id: "x" },
    } as unknown as RunConfigProperties
    const run_config = makeRunConfig({
      properties: mcpProps,
      selected_model: null,
    })
    run_config.get_selected_model = vi.fn().mockReturnValue(null)

    const result = await send_multiturn({
      project_id: "p",
      task_id: "t",
      parent_task_run_id: "leaf",
      run_config_component: asRunConfigController(run_config),
      input_form: asInputFormController(makeInputForm()),
      on_success: vi.fn(),
    })
    expect(result.ok).toBe(true)
    expect(postMock).toHaveBeenCalledTimes(1)
  })

  it("rejects when the server returns no id", async () => {
    postMock.mockResolvedValue({ data: { id: null }, error: null })
    const input_form = makeInputForm("kept")
    const result = await send_multiturn({
      project_id: "p",
      task_id: "t",
      parent_task_run_id: "leaf",
      run_config_component: asRunConfigController(makeRunConfig()),
      input_form: asInputFormController(input_form),
      on_success: vi.fn(),
    })
    expect(result.ok).toBe(false)
    expect(input_form.clear_input).not.toHaveBeenCalled()
  })

  it("seeds the POST body's run_config_properties from the component (multiturn defaults from previous run)", async () => {
    postMock.mockResolvedValue({ data: { id: "ok" }, error: null })
    const props: RunConfigProperties = {
      type: "kiln_agent",
      model_provider_name: "openai",
      model_name: "gpt-4o",
      prompt_id: "simple_prompt_builder",
      temperature: 0.5,
      top_p: 1,
      structured_output_mode: "default",
      thinking_level: null,
      tools_config: { tools: [] },
    } as unknown as RunConfigProperties
    const run_config = makeRunConfig({ properties: props })

    await send_multiturn({
      project_id: "p",
      task_id: "t",
      parent_task_run_id: "leaf",
      run_config_component: asRunConfigController(run_config),
      input_form: asInputFormController(makeInputForm("text")),
      on_success: vi.fn(),
    })

    const body = (
      postMock.mock.calls[0][1] as { body: Record<string, unknown> }
    ).body
    expect(body.run_config_properties).toEqual(props)
  })
})
