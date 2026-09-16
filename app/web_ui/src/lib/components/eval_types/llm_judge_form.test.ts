// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from "vitest"
import { render, cleanup, fireEvent } from "@testing-library/svelte"
import * as svelteMod from "svelte"
import { tick } from "svelte"
import { writable } from "svelte/store"
import type { AvailableModels } from "$lib/types"

// ---------------------------------------------------------------------------
// Mock available_models store
// ---------------------------------------------------------------------------
const mockAvailableModels = writable<AvailableModels[]>([])

vi.mock("$lib/stores", () => ({
  available_models: mockAvailableModels,
}))

vi.mock("$lib/ui/provider_image", () => ({
  get_provider_image: () => "/fake-image.png",
}))

vi.mock(
  "$lib/ui/run_config_component/available_models_dropdown.svelte",
  async () => {
    const StubModule = await import(
      "./__tests__/available_models_dropdown_stub.svelte"
    )
    return { default: StubModule.default }
  },
)

const { mockGetDefaultLlmJudgePrompt } = vi.hoisted(() => ({
  mockGetDefaultLlmJudgePrompt: vi.fn().mockResolvedValue({
    judge_prompt: "Default judge prompt {{ task_input }} {{ final_message }}",
    system_prompt: "You are an evaluator.",
  }),
}))

vi.mock("$lib/api/v2_eval_api", () => ({
  getDefaultLlmJudgePrompt: mockGetDefaultLlmJudgePrompt,
}))

const LlmJudgeForm = (await import("./llm_judge_form.svelte")).default
const LlmJudgeFormHarness = (
  await import("./__tests__/llm_judge_form_harness.svelte")
).default

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeModel(
  id: string,
  name: string,
  opts: {
    supports_logprobs?: boolean
    suggested_for_evals?: boolean
    supports_structured_output?: boolean
  } = {},
) {
  return {
    id,
    name,
    supports_structured_output: opts.supports_structured_output ?? true,
    supports_data_gen: false,
    suggested_for_data_gen: false,
    supports_logprobs: opts.supports_logprobs ?? false,
    suggested_for_evals: opts.suggested_for_evals ?? false,
    suggested_for_synthetic_user: false,
    supports_function_calling: false,
    uncensored: false,
    suggested_for_uncensored_data_gen: false,
    supports_vision: false,
    supports_doc_extraction: false,
    suggested_for_doc_extraction: false,
    multimodal_capable: false,
    multimodal_mime_types: null,
    structured_output_mode: "json_mode" as const,
    available_thinking_levels: null,
    default_thinking_level: null,
    untested_model: false,
    deprecated: false,
  }
}

function setModels(models: AvailableModels[]) {
  mockAvailableModels.set(models)
}

// Provider where the model does NOT support logprobs (single algorithm: llm_as_judge only)
const noLogprobsProvider: AvailableModels = {
  provider_name: "OpenAI",
  provider_id: "openai",
  models: [
    makeModel("gpt-4o", "GPT-4o", {
      suggested_for_evals: true,
      supports_logprobs: false,
    }),
  ],
}

// Provider where the model DOES support logprobs (two algorithms: llm_as_judge + g_eval)
const logprobsProvider: AvailableModels = {
  provider_name: "OpenAI",
  provider_id: "openai",
  models: [
    makeModel("gpt-4o", "GPT-4o", {
      suggested_for_evals: true,
      supports_logprobs: true,
    }),
  ],
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("LlmJudgeForm", () => {
  afterEach(() => {
    cleanup()
    setModels([])
  })

  describe("E16: model card min-height", () => {
    it("model cards have min-h-[120px] class", async () => {
      setModels([noLogprobsProvider])
      const { container } = render(LlmJudgeForm, {
        props: { task_id: "task1", project_id: "proj1", eval_id: "eval1" },
      })
      await tick()

      const modelCards = container.querySelectorAll(
        ".card.card-bordered.cursor-pointer",
      )
      expect(modelCards.length).toBeGreaterThan(0)
      for (const card of modelCards) {
        expect(card.classList.contains("min-h-[120px]")).toBe(true)
      }
    })
  })

  describe("E17: dropped string", () => {
    it("does not contain 'Supports graded scoring across the full score range.'", () => {
      setModels([noLogprobsProvider])
      const { container } = render(LlmJudgeForm, {
        props: { task_id: "task1", project_id: "proj1", eval_id: "eval1" },
      })
      expect(container.textContent).not.toContain(
        "Supports graded scoring across the full score range.",
      )
    })
  })

  describe("E18: hide algorithm section for single-algorithm model", () => {
    it("hides algorithm section when model only supports default algorithm", async () => {
      setModels([noLogprobsProvider])
      // Use provider_id / model_id as model_name / provider_name (matches dropdown behavior)
      const { container } = render(LlmJudgeForm, {
        props: {
          task_id: "task1",
          project_id: "proj1",
          eval_id: "eval1",
          model_name: "gpt-4o",
          provider_name: "openai",
          combined_model_name: "openai/gpt-4o",
        },
      })
      await tick()

      const algoSection = container.querySelector(
        '[data-testid="algorithm-section"]',
      )
      expect(algoSection).toBeNull()
      expect(container.textContent).not.toContain("Select Judge Algorithm")
    })

    it("shows algorithm section when model supports G-Eval (logprobs)", async () => {
      setModels([logprobsProvider])
      const { container } = render(LlmJudgeForm, {
        props: {
          task_id: "task1",
          project_id: "proj1",
          eval_id: "eval1",
          model_name: "gpt-4o",
          provider_name: "openai",
          combined_model_name: "openai/gpt-4o",
        },
      })
      await tick()

      const algoSection = container.querySelector(
        '[data-testid="algorithm-section"]',
      )
      expect(algoSection).not.toBeNull()
      expect(container.textContent).toContain("Select Judge Algorithm")
    })

    it("auto-selects the single algorithm so form is valid on model select (section hidden)", async () => {
      // When only one algorithm is available, the section is hidden and
      // update_unsupported_algos_and_default_algo auto-selects llm_as_judge.
      // This means can_submit_llm (!!selected_algo && !!combined_model_name) is true.
      setModels([noLogprobsProvider])
      const { container } = render(LlmJudgeForm, {
        props: {
          task_id: "task1",
          project_id: "proj1",
          eval_id: "eval1",
          model_name: "gpt-4o",
          provider_name: "openai",
          combined_model_name: "openai/gpt-4o",
        },
      })
      await tick()

      // Section hidden = algorithm was auto-selected (only 1 option)
      const algoSection = container.querySelector(
        '[data-testid="algorithm-section"]',
      )
      expect(algoSection).toBeNull()

      // No radio buttons visible - user doesn't need to pick
      const radios = container.querySelectorAll('input[type="radio"]')
      expect(radios.length).toBe(0)
    })

    it("algorithm section is shown if the section would wrongly appear for single-option model", async () => {
      // This test verifies the section does NOT show for a model without logprobs,
      // and DOES show for a model with logprobs. This is a guard test: if someone
      // reverts the hide logic, the single-algo test above fails.
      setModels([noLogprobsProvider])
      const { container: c1 } = render(LlmJudgeForm, {
        props: {
          task_id: "task1",
          project_id: "proj1",
          eval_id: "eval1",
          model_name: "gpt-4o",
          provider_name: "openai",
          combined_model_name: "openai/gpt-4o",
        },
      })
      await tick()
      expect(c1.querySelector('[data-testid="algorithm-section"]')).toBeNull()

      cleanup()

      setModels([logprobsProvider])
      const { container: c2 } = render(LlmJudgeForm, {
        props: {
          task_id: "task1",
          project_id: "proj1",
          eval_id: "eval1",
          model_name: "gpt-4o",
          provider_name: "openai",
          combined_model_name: "openai/gpt-4o",
        },
      })
      await tick()
      expect(
        c2.querySelector('[data-testid="algorithm-section"]'),
      ).not.toBeNull()
    })
  })

  describe("E19: algorithm card sizing (reverted to pre-Phase-6)", () => {
    it("algorithm cards use w-[260px] and aspect-[5/6]", async () => {
      setModels([logprobsProvider])
      const { container } = render(LlmJudgeForm, {
        props: {
          task_id: "task1",
          project_id: "proj1",
          eval_id: "eval1",
          model_name: "gpt-4o",
          provider_name: "openai",
          combined_model_name: "openai/gpt-4o",
        },
      })
      await tick()

      const algoCards = container.querySelectorAll(
        '[data-testid="algorithm-section"] .card.card-bordered',
      )
      expect(algoCards.length).toBeGreaterThan(0)
      for (const card of algoCards) {
        expect(card.classList.contains("w-[260px]")).toBe(true)
        expect(card.classList.contains("aspect-[5/6]")).toBe(true)
        expect(card.classList.contains("p-6")).toBe(true)
      }
    })

    it("algorithm cards do NOT use the shrunken Phase-6 sizes", async () => {
      setModels([logprobsProvider])
      const { container } = render(LlmJudgeForm, {
        props: {
          task_id: "task1",
          project_id: "proj1",
          eval_id: "eval1",
          model_name: "gpt-4o",
          provider_name: "openai",
          combined_model_name: "openai/gpt-4o",
        },
      })
      await tick()

      const algoCards = container.querySelectorAll(
        '[data-testid="algorithm-section"] .card.card-bordered',
      )
      expect(algoCards.length).toBeGreaterThan(0)
      for (const card of algoCards) {
        expect(card.classList.contains("w-[156px]")).toBe(false)
        expect(card.classList.contains("p-4")).toBe(false)
      }
    })

    it("radio buttons use my-8 spacing (pre-Phase-6)", async () => {
      setModels([logprobsProvider])
      const { container } = render(LlmJudgeForm, {
        props: {
          task_id: "task1",
          project_id: "proj1",
          eval_id: "eval1",
          model_name: "gpt-4o",
          provider_name: "openai",
          combined_model_name: "openai/gpt-4o",
        },
      })
      await tick()

      const radios = container.querySelectorAll(
        '[data-testid="algorithm-section"] input[type="radio"]',
      )
      expect(radios.length).toBeGreaterThan(0)
      for (const radio of radios) {
        expect(radio.classList.contains("my-8")).toBe(true)
        expect(radio.classList.contains("my-4")).toBe(false)
      }
    })
  })

  describe("Advanced: Judge Prompt section", () => {
    afterEach(() => {
      mockGetDefaultLlmJudgePrompt.mockResolvedValue({
        judge_prompt:
          "Default judge prompt {{ task_input }} {{ final_message }}",
        system_prompt: "You are an evaluator.",
      })
    })

    it("renders the collapse with judge prompt helper text", () => {
      setModels([noLogprobsProvider])
      const { container } = render(LlmJudgeForm, {
        props: { task_id: "task1", project_id: "proj1", eval_id: "eval1" },
      })
      expect(container.textContent).toContain("Advanced: Judge Prompt")
      expect(container.textContent).toContain(
        "Customizing the judge prompt can improve eval quality",
      )
    })

    // Svelte 4 async onMount callbacks do not execute in jsdom/vitest,
    // so we capture onMount callbacks via vi.spyOn and invoke them manually.

    it("pre-fills judge_prompt and system_prompt from API on mount", async () => {
      setModels([noLogprobsProvider])

      const onMountCallbacks: Array<() => unknown> = []
      const spy = vi
        .spyOn(svelteMod, "onMount")
        .mockImplementation((fn: () => unknown) => {
          onMountCallbacks.push(fn)
        })

      const { container } = render(LlmJudgeForm, {
        props: { task_id: "task1", project_id: "proj1", eval_id: "eval1" },
      })

      spy.mockRestore()

      for (const cb of onMountCallbacks) {
        await cb()
      }
      await tick()

      expect(mockGetDefaultLlmJudgePrompt).toHaveBeenCalledWith(
        "proj1",
        "task1",
        "eval1",
      )

      const judgeTextarea = container.querySelector(
        "#judge_prompt",
      ) as HTMLTextAreaElement
      expect(judgeTextarea).not.toBeNull()
      expect(judgeTextarea.value).toContain("Default judge prompt")

      const systemTextarea = container.querySelector(
        "#system_prompt",
      ) as HTMLTextAreaElement
      expect(systemTextarea).not.toBeNull()
      expect(systemTextarea.value).toBe("You are an evaluator.")
    })

    it("shows warning when fetch fails", async () => {
      mockGetDefaultLlmJudgePrompt.mockRejectedValueOnce(
        new Error("Network error"),
      )

      setModels([noLogprobsProvider])

      const onMountCallbacks: Array<() => unknown> = []
      const spy = vi
        .spyOn(svelteMod, "onMount")
        .mockImplementation((fn: () => unknown) => {
          onMountCallbacks.push(fn)
        })

      const { container } = render(LlmJudgeForm, {
        props: { task_id: "task1", project_id: "proj1", eval_id: "eval1" },
      })

      spy.mockRestore()

      for (const cb of onMountCallbacks) {
        await cb()
      }
      await tick()

      expect(container.textContent).toContain(
        "Could not load default judge prompt",
      )
    })

    it("renders both textarea fields for judge and system prompt", () => {
      setModels([noLogprobsProvider])
      const { container } = render(LlmJudgeForm, {
        props: { task_id: "task1", project_id: "proj1", eval_id: "eval1" },
      })

      const judgeTextarea = container.querySelector("#judge_prompt")
      const systemTextarea = container.querySelector("#system_prompt")
      expect(judgeTextarea).not.toBeNull()
      expect(systemTextarea).not.toBeNull()
    })
  })
  describe("Judge criteria field (removed)", () => {
    it("does not render the old spec-less criteria field", () => {
      setModels([noLogprobsProvider])
      const { container } = render(LlmJudgeForm, {
        props: { task_id: "task1", project_id: "proj1", eval_id: "eval1" },
      })
      expect(container.querySelector("#judge_criteria")).toBeNull()
    })
  })

  describe("Evaluation instructions steps UI", () => {
    // The UI is keyed off the server's default prompt: it appears exactly
    // when the prompt references {{ judge_instructions }} (evals with no
    // derivable steps), so client and server can never drift.
    async function renderWithDefaultPrompt(judge_prompt: string) {
      setModels([noLogprobsProvider])
      mockGetDefaultLlmJudgePrompt.mockResolvedValueOnce({
        judge_prompt,
        system_prompt: "You are an evaluator.",
      })
      const onMountCallbacks: Array<() => unknown> = []
      const spy = vi
        .spyOn(svelteMod, "onMount")
        .mockImplementation((fn: () => unknown) => {
          onMountCallbacks.push(fn)
        })
      const rendered = render(LlmJudgeForm, {
        props: { task_id: "task1", project_id: "proj1", eval_id: "eval1" },
      })
      spy.mockRestore()
      for (const cb of onMountCallbacks) {
        await cb()
      }
      await tick()
      return rendered
    }

    it("is hidden when the default prompt has derivable steps", async () => {
      const { container, queryByText } = await renderWithDefaultPrompt(
        "Judge {{ final_message }} using baked steps",
      )
      expect(container.querySelector("#judge_instruction_0")).toBeNull()
      expect(queryByText("Evaluation Instructions")).toBeNull()
      // Judge prompt stays inside the Advanced collapse
      expect(queryByText("Advanced: Judge Prompt")).not.toBeNull()
    })

    it("renders steps editor and top-level judge prompt when the default prompt binds instructions", async () => {
      const { container, queryByText } = await renderWithDefaultPrompt(
        "Judge {{ final_message }} with <steps>{{ judge_instructions }}</steps>",
      )
      expect(queryByText("Evaluation Instructions")).not.toBeNull()
      const step = container.querySelector(
        "#judge_instruction_0",
      ) as HTMLTextAreaElement
      expect(step).not.toBeNull()
      expect(step.placeholder).toContain("issue X")
      // Judge prompt is top-level, not collapsed
      expect(queryByText("Advanced: Judge Prompt")).toBeNull()
      expect(container.querySelector("#judge_prompt")).not.toBeNull()
      // The judge_instructions variable is documented
      expect(queryByText("{{ judge_instructions }}")).not.toBeNull()
    })

    it("Add Evaluation Step appends another step row", async () => {
      const { container, getByText } = await renderWithDefaultPrompt(
        "Judge {{ final_message }} with <steps>{{ judge_instructions }}</steps>",
      )
      expect(container.querySelector("#judge_instruction_1")).toBeNull()

      // Two empty rows must coexist (regression: value-keyed each blew up on
      // duplicate empty strings, making the button a no-op).
      await fireEvent.click(getByText("Add Evaluation Step"))
      await tick()
      expect(container.querySelector("#judge_instruction_1")).not.toBeNull()

      await fireEvent.click(getByText("Add Evaluation Step"))
      await tick()
      expect(container.querySelector("#judge_instruction_2")).not.toBeNull()
    })
  })
})

describe("state bound out to the builder", () => {
  async function renderHarness() {
    setModels([noLogprobsProvider])
    const onMountCallbacks: Array<() => unknown> = []
    const spy = vi
      .spyOn(svelteMod, "onMount")
      .mockImplementation((fn: () => unknown) => {
        onMountCallbacks.push(fn)
      })
    const rendered = render(LlmJudgeFormHarness)
    spy.mockRestore()
    for (const cb of onMountCallbacks) {
      await cb()
    }
    await tick()
    return rendered
  }

  it("clears default_prompt_unavailable once a fetch succeeds", async () => {
    // The harness starts it true, standing in for a builder that kept it from an
    // earlier failure. Switching eval type away from llm_judge and back recreates this
    // form, and `bind:` seeds it from the parent — so without an explicit clear one
    // transient network error would pin the reference-data input open for the session.
    mockGetDefaultLlmJudgePrompt.mockResolvedValueOnce({
      judge_prompt: "Rate {{ final_message }}",
      system_prompt: "You are an evaluator.",
      reference_keys: [],
    })

    const { container } = await renderHarness()

    expect(
      container.querySelector('[data-testid="harness-prompt-unavailable"]')
        ?.textContent,
    ).toBe("false")
  })

  it("keeps default_prompt_unavailable set when the fetch fails", async () => {
    mockGetDefaultLlmJudgePrompt.mockRejectedValueOnce(
      new Error("Network error"),
    )

    const { container } = await renderHarness()

    expect(
      container.querySelector('[data-testid="harness-prompt-unavailable"]')
        ?.textContent,
    ).toBe("true")
  })

  it("binds out the reference keys the server derived", async () => {
    mockGetDefaultLlmJudgePrompt.mockResolvedValueOnce({
      judge_prompt: "Grade against {{ reference_data.reference_answer }}",
      system_prompt: "You are an evaluator.",
      reference_keys: ["reference_answer"],
    })

    const { container } = await renderHarness()

    expect(
      container
        .querySelector('[data-testid="harness-reference-keys"]')
        ?.textContent?.trim(),
    ).toBe("reference_answer")
  })
})
