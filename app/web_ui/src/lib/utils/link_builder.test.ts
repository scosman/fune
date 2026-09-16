import { describe, it, expect } from "vitest"
import { prompt_link, tool_link, tool_set_type_label } from "./link_builder"

describe("prompt_link", () => {
  const mockProjectId = "test-project"
  const mockTaskId = "test-task"

  describe("parameter validation", () => {
    it("returns undefined when project_id is missing", () => {
      const result = prompt_link("", mockTaskId, "123456")
      expect(result).toBeUndefined()
    })

    it("returns undefined when task_id is missing", () => {
      const result = prompt_link(mockProjectId, "", "123456")
      expect(result).toBeUndefined()
    })

    it("returns undefined when prompt_id is missing", () => {
      const result = prompt_link(mockProjectId, mockTaskId, "")
      expect(result).toBeUndefined()
    })

    it("returns undefined when all parameters are missing", () => {
      const result = prompt_link("", "", "")
      expect(result).toBeUndefined()
    })
  })

  describe("fine-tuned prompts", () => {
    it("handles fine-tuned prompt with simple ID", () => {
      const promptId = "fine_tune_prompt::123456"
      const result = prompt_link(mockProjectId, mockTaskId, promptId)

      expect(result).toBe(
        `/fine_tune/${mockProjectId}/${mockTaskId}/fine_tune/123456`,
      )
    })

    it("handles fine-tuned prompt with complex ID containing double colons", () => {
      const promptId = "fine_tune_prompt::789123::456789::987654"
      const result = prompt_link(mockProjectId, mockTaskId, promptId)

      // Should extract the last part after the final "::"
      expect(result).toBe(
        `/fine_tune/${mockProjectId}/${mockTaskId}/fine_tune/987654`,
      )
    })

    it("handles fine-tuned prompt with no additional parts", () => {
      const promptId = "fine_tune_prompt::"
      const result = prompt_link(mockProjectId, mockTaskId, promptId)

      expect(result).toBe(
        `/fine_tune/${mockProjectId}/${mockTaskId}/fine_tune/`,
      )
    })

    it("handles fine-tuned prompt with multiple numeric IDs", () => {
      const promptId =
        "fine_tune_prompt::306673458440::629828465317::310597603461"
      const result = prompt_link(mockProjectId, mockTaskId, promptId)

      // Should extract the last numeric ID
      expect(result).toBe(
        `/fine_tune/${mockProjectId}/${mockTaskId}/fine_tune/310597603461`,
      )
    })

    it("URL encodes fine-tuned prompt IDs with special characters", () => {
      const promptId = "fine_tune_prompt::123456789"
      const result = prompt_link(mockProjectId, mockTaskId, promptId)

      expect(result).toBe(
        `/fine_tune/${mockProjectId}/${mockTaskId}/fine_tune/123456789`,
      )
    })
  })

  describe("saved prompts (ID style)", () => {
    it("links to saved prompts for prompts with double colons", () => {
      const promptId = "111222::333444::555666"
      const result = prompt_link(mockProjectId, mockTaskId, promptId)

      expect(result).toBe(
        `/prompts/${mockProjectId}/${mockTaskId}/saved/111222%3A%3A333444%3A%3A555666`,
      )
    })

    it("handles saved prompts with single double colon", () => {
      const promptId = "777888::999000"
      const result = prompt_link(mockProjectId, mockTaskId, promptId)

      expect(result).toBe(
        `/prompts/${mockProjectId}/${mockTaskId}/saved/777888%3A%3A999000`,
      )
    })

    it("URL encodes saved prompt IDs with special characters", () => {
      const promptId = "123456::789012"
      const result = prompt_link(mockProjectId, mockTaskId, promptId)

      expect(result).toBe(
        `/prompts/${mockProjectId}/${mockTaskId}/saved/123456%3A%3A789012`,
      )
    })
  })

  describe("edge cases", () => {
    it("handles project and task IDs that need URL encoding", () => {
      const projectWithSpaces = "project with spaces"
      const taskWithSymbols = "task&symbols"
      const promptId = "777888::999000"

      const result = prompt_link(projectWithSpaces, taskWithSymbols, promptId)

      expect(result).toBe(
        `/prompts/project with spaces/task&symbols/saved/777888%3A%3A999000`,
      )
    })
  })
})

describe("tool_link", () => {
  const project = "proj1"

  it("returns null for empty inputs", () => {
    expect(tool_link("", "some_id")).toBeNull()
    expect(tool_link(project, "")).toBeNull()
  })

  it("links code tools to their detail page", () => {
    expect(tool_link(project, "kiln_tool::code::abc123")).toBe(
      "/tools/proj1/code_tools/abc123",
    )
  })

  it("links mcp remote tools to their server page", () => {
    expect(tool_link(project, "mcp::remote::srv1")).toBe(
      "/tools/proj1/tool_servers/srv1",
    )
  })

  it("links mcp local tools to their server page", () => {
    expect(tool_link(project, "mcp::local::srv2")).toBe(
      "/tools/proj1/tool_servers/srv2",
    )
  })

  it("links kiln_task tools", () => {
    expect(tool_link(project, "kiln_task::task1")).toBe(
      "/tools/proj1/kiln_task/task1",
    )
  })

  it("links skill tools", () => {
    expect(tool_link(project, "kiln_tool::skill::s1")).toBe("/skills/proj1/s1")
  })

  it("links rag tools", () => {
    expect(tool_link(project, "kiln_tool::rag::r1")).toBe(
      "/docs/rag_configs/proj1/r1/rag_config",
    )
  })

  it("falls back to tools index for unknown kiln_tool prefixes", () => {
    expect(tool_link(project, "kiln_tool::other::x")).toBe("/tools/proj1")
  })

  it("returns null for unrecognized prefixes", () => {
    expect(tool_link(project, "unknown::thing")).toBeNull()
  })
})

describe("tool_set_type_label", () => {
  it("returns friendly labels for known types", () => {
    expect(tool_set_type_label("code")).toBe("Code Tool")
    expect(tool_set_type_label("mcp")).toBe("MCP")
    expect(tool_set_type_label("search")).toBe("Search")
    expect(tool_set_type_label("kiln_task")).toBe("Kiln Task")
    expect(tool_set_type_label("demo")).toBe("Demo")
    expect(tool_set_type_label("skill")).toBe("Skill")
    expect(tool_set_type_label("builtin")).toBe("Built-in")
    expect(tool_set_type_label("sandbox_code")).toBe("Sandbox Code")
  })

  it("throws for an invalid type via assertNever", () => {
    expect(() =>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      tool_set_type_label("invalid_type" as any),
    ).toThrow("Unexpected value")
  })
})
