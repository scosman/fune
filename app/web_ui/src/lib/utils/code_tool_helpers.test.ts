import { describe, it, expect } from "vitest"
import {
  extractParams,
  generateCodeToolPlaceholder,
  generateImportHelper,
  shouldInsertImport,
  isCodeUnmodified,
  generateExamples,
  formatParamPreview,
  plainTextParamsSchema,
  resolveStep2Code,
} from "./code_tool_helpers"

describe("extractParams", () => {
  it("extracts string params", () => {
    const schema = {
      type: "object",
      properties: { name: { type: "string" } },
      required: ["name"],
    }
    const params = extractParams(schema)
    expect(params).toEqual([
      { name: "name", pythonType: "str", required: true },
    ])
  })

  it("extracts integer params", () => {
    const schema = {
      type: "object",
      properties: { count: { type: "integer" } },
      required: ["count"],
    }
    const params = extractParams(schema)
    expect(params).toEqual([
      { name: "count", pythonType: "int", required: true },
    ])
  })

  it("extracts number params", () => {
    const schema = {
      type: "object",
      properties: { ratio: { type: "number" } },
      required: ["ratio"],
    }
    const params = extractParams(schema)
    expect(params).toEqual([
      { name: "ratio", pythonType: "float", required: true },
    ])
  })

  it("extracts boolean params", () => {
    const schema = {
      type: "object",
      properties: { active: { type: "boolean" } },
      required: ["active"],
    }
    const params = extractParams(schema)
    expect(params).toEqual([
      { name: "active", pythonType: "bool", required: true },
    ])
  })

  it("extracts array params with item type", () => {
    const schema = {
      type: "object",
      properties: {
        tags: { type: "array", items: { type: "string" } },
      },
      required: ["tags"],
    }
    const params = extractParams(schema)
    expect(params).toEqual([
      { name: "tags", pythonType: "list[str]", required: true },
    ])
  })

  it("extracts array params without item type", () => {
    const schema = {
      type: "object",
      properties: {
        data: { type: "array" },
      },
      required: ["data"],
    }
    const params = extractParams(schema)
    expect(params).toEqual([
      { name: "data", pythonType: "list", required: true },
    ])
  })

  it("extracts object params", () => {
    const schema = {
      type: "object",
      properties: { config: { type: "object" } },
      required: ["config"],
    }
    const params = extractParams(schema)
    expect(params).toEqual([
      { name: "config", pythonType: "dict", required: true },
    ])
  })

  it("marks optional params correctly", () => {
    const schema = {
      type: "object",
      properties: {
        name: { type: "string" },
        limit: { type: "integer" },
      },
      required: ["name"],
    }
    const params = extractParams(schema)
    expect(params).toEqual([
      { name: "name", pythonType: "str", required: true },
      { name: "limit", pythonType: "int", required: false },
    ])
  })

  it("returns empty for schema with no properties", () => {
    const schema = { type: "object" }
    const params = extractParams(schema)
    expect(params).toEqual([])
  })

  it("returns empty for empty properties object", () => {
    const schema = { type: "object", properties: {} }
    const params = extractParams(schema)
    expect(params).toEqual([])
  })

  it("handles missing required array", () => {
    const schema = {
      type: "object",
      properties: { x: { type: "string" } },
    }
    const params = extractParams(schema)
    expect(params).toEqual([{ name: "x", pythonType: "str", required: false }])
  })
})

describe("generateCodeToolPlaceholder", () => {
  it("generates zero-arg stub", () => {
    const schema = { type: "object", properties: {} }
    const result = generateCodeToolPlaceholder(schema, "Do something")
    expect(result).toContain("def run() -> str:")
    expect(result).toContain('"""Do something"""')
    expect(result).toContain("# Your code here")
    expect(result).toContain('return "result"')
  })

  it("generates typed params with required first", () => {
    const schema = {
      type: "object",
      properties: {
        query: { type: "string" },
        limit: { type: "integer" },
      },
      required: ["query"],
    }
    const result = generateCodeToolPlaceholder(schema, "Search things")
    expect(result).toContain(
      "def run(query: str, limit: int | None = None) -> str:",
    )
  })

  it("generates all required params", () => {
    const schema = {
      type: "object",
      properties: {
        a: { type: "string" },
        b: { type: "integer" },
      },
      required: ["a", "b"],
    }
    const result = generateCodeToolPlaceholder(schema, "test")
    expect(result).toContain("def run(a: str, b: int) -> str:")
  })

  it("generates all optional params", () => {
    const schema = {
      type: "object",
      properties: {
        x: { type: "boolean" },
        y: { type: "number" },
      },
      required: [],
    }
    const result = generateCodeToolPlaceholder(schema, "test")
    expect(result).toContain(
      "def run(x: bool | None = None, y: float | None = None) -> str:",
    )
  })

  it("handles nested array types", () => {
    const schema = {
      type: "object",
      properties: {
        ids: { type: "array", items: { type: "integer" } },
      },
      required: ["ids"],
    }
    const result = generateCodeToolPlaceholder(schema, "test")
    expect(result).toContain("def run(ids: list[int]) -> str:")
  })

  // The sandbox invokes run() with keyword arguments keyed by the schema's exact
  // property names, so the stub must never rename them. Reserved-word names are
  // rejected by CodeTool validation server-side, not rewritten here.
  it("uses schema property names verbatim, without renaming", () => {
    const schema = {
      type: "object",
      properties: {
        match: { type: "string" },
        type: { type: "integer" },
      },
      required: ["match"],
    }
    const result = generateCodeToolPlaceholder(schema, "test")
    expect(result).toContain(
      "def run(match: str, type: int | None = None) -> str:",
    )
  })

  // Each case asserts the exact stub so the docstring stays a valid Python string
  // for any description (verified against compile() when this table was built).
  const docstringCases: { name: string; desc: string; escaped: string }[] = [
    {
      name: "embedded triple quotes",
      desc: 'has """quotes"""',
      escaped: 'has \\"\\"\\"quotes\\"\\"\\"',
    },
    {
      name: "trailing double quote",
      desc: 'ends with "',
      escaped: 'ends with \\"',
    },
    {
      name: "trailing backslash",
      desc: "ends with \\",
      escaped: "ends with \\\\",
    },
    {
      name: "backslash followed by triple quote",
      desc: '\\"""',
      escaped: '\\\\\\"\\"\\"',
    },
    {
      name: "lone double quote",
      desc: '"',
      escaped: '\\"',
    },
    {
      name: "lone backslash",
      desc: "\\",
      escaped: "\\\\",
    },
    {
      name: "escaped-looking quote sequence",
      desc: 'mix \\" of \\\\ and """ end "',
      escaped: 'mix \\\\\\" of \\\\\\\\ and \\"\\"\\" end \\"',
    },
    {
      name: "multiline description",
      desc: "line one\nline two",
      escaped: "line one\nline two",
    },
    {
      name: "empty description",
      desc: "",
      escaped: "",
    },
  ]

  for (const { name, desc, escaped } of docstringCases) {
    it(`escapes docstring safely: ${name}`, () => {
      const schema = { type: "object", properties: {} }
      const result = generateCodeToolPlaceholder(schema, desc)
      expect(result).toBe(
        `def run() -> str:\n    """${escaped}"""\n    # Your code here\n    return "result"\n`,
      )
    })
  }

  it("handles deeply nested array/object types", () => {
    const schema = {
      type: "object",
      properties: {
        matrix: {
          type: "array",
          items: { type: "array", items: { type: "integer" } },
        },
      },
      required: ["matrix"],
    }
    const result = generateCodeToolPlaceholder(schema, "test")
    expect(result).toContain("matrix: list[list[int]]")
  })
})

describe("generateImportHelper", () => {
  it("includes the function name in the comment", () => {
    const result = generateImportHelper("get_user")
    expect(result).toContain("tools.get_user(...)")
    expect(result).toContain("async_tools.get_user(...)")
  })

  it("includes the import statement", () => {
    const result = generateImportHelper("search")
    expect(result).toContain("from kiln import tools, async_tools")
  })
})

describe("shouldInsertImport", () => {
  it("returns true when import is missing", () => {
    expect(shouldInsertImport("def run():\n    pass")).toBe(true)
  })

  it("returns false when import is present", () => {
    expect(
      shouldInsertImport("from kiln import tools\ndef run():\n    pass"),
    ).toBe(false)
  })

  it("returns false when import with async_tools is present", () => {
    expect(
      shouldInsertImport(
        "from kiln import tools, async_tools\ndef run():\n    pass",
      ),
    ).toBe(false)
  })

  it("returns true for empty code", () => {
    expect(shouldInsertImport("")).toBe(true)
  })
})

describe("isCodeUnmodified", () => {
  it("returns true for empty code", () => {
    expect(isCodeUnmodified("", "placeholder")).toBe(true)
  })

  it("returns true for whitespace-only code", () => {
    expect(isCodeUnmodified("   \n  ", "placeholder")).toBe(true)
  })

  it("returns true for exact match", () => {
    const placeholder = 'def run() -> str:\n    """test"""\n    pass\n'
    expect(isCodeUnmodified(placeholder, placeholder)).toBe(true)
  })

  it("returns false for modified code", () => {
    const placeholder = 'def run() -> str:\n    """test"""\n    pass\n'
    expect(isCodeUnmodified(placeholder + "# edited", placeholder)).toBe(false)
  })

  it("returns false when import helper has been prepended", () => {
    const placeholder = 'def run() -> str:\n    """test"""\n    pass\n'
    const with_import = generateImportHelper("get_user") + placeholder
    expect(isCodeUnmodified(with_import, placeholder)).toBe(false)
  })
})

describe("resolveStep2Code", () => {
  const placeholderA = 'def run() -> str:\n    """A"""\n    return "result"\n'
  const placeholderB =
    'def run(x: str) -> str:\n    """B"""\n    return "result"\n'

  it("first visit with no clone seeds the fresh placeholder", () => {
    const result = resolveStep2Code({
      code: "",
      newPlaceholder: placeholderA,
      generatedPlaceholder: "",
      schemaChangedHint: false,
      cloneCode: null,
    })
    expect(result).toEqual({
      code: placeholderA,
      generatedPlaceholder: placeholderA,
      schemaChangedHint: false,
      cloneConsumed: false,
    })
  })

  it("first visit with a clone seeds the clone code and marks it consumed", () => {
    const cloneCode = "def run():\n    return 'cloned'\n"
    const result = resolveStep2Code({
      code: "",
      newPlaceholder: placeholderA,
      generatedPlaceholder: "",
      schemaChangedHint: false,
      cloneCode,
    })
    expect(result).toEqual({
      code: cloneCode,
      generatedPlaceholder: placeholderA,
      schemaChangedHint: false,
      cloneConsumed: true,
    })
  })

  it("regenerates an untouched placeholder after a schema change", () => {
    const result = resolveStep2Code({
      code: placeholderA,
      newPlaceholder: placeholderB,
      generatedPlaceholder: placeholderA,
      schemaChangedHint: false,
      cloneCode: null,
    })
    expect(result).toEqual({
      code: placeholderB,
      generatedPlaceholder: placeholderB,
      schemaChangedHint: false,
      cloneConsumed: false,
    })
  })

  // The regression this guards: returning to the Code step (browser Back makes
  // the wizard read as a first visit) must not overwrite authored code.
  it("preserves user-authored code and flags the schema change", () => {
    const userCode = placeholderA + "\n# my real implementation\n"
    const result = resolveStep2Code({
      code: userCode,
      newPlaceholder: placeholderB,
      generatedPlaceholder: placeholderA,
      schemaChangedHint: false,
      cloneCode: null,
    })
    expect(result).toEqual({
      code: userCode,
      generatedPlaceholder: placeholderB,
      schemaChangedHint: true,
      cloneConsumed: false,
    })
  })

  it("leaves user-authored code untouched when the schema is unchanged", () => {
    const userCode = placeholderA + "\n# my real implementation\n"
    const result = resolveStep2Code({
      code: userCode,
      newPlaceholder: placeholderA,
      generatedPlaceholder: placeholderA,
      schemaChangedHint: true,
      cloneCode: null,
    })
    expect(result).toEqual({
      code: userCode,
      generatedPlaceholder: placeholderA,
      schemaChangedHint: true,
      cloneConsumed: false,
    })
  })

  it("does not re-seed from the clone once the user has code", () => {
    const userCode = "def run():\n    return 'edited'\n"
    const result = resolveStep2Code({
      code: userCode,
      newPlaceholder: placeholderB,
      generatedPlaceholder: placeholderA,
      schemaChangedHint: false,
      // A stale clone that was already consumed would still be passed as null
      // by the caller; even if present, existing code wins.
      cloneCode: "def run():\n    return 'cloned'\n",
    })
    expect(result.code).toBe(userCode)
    expect(result.cloneConsumed).toBe(false)
  })
})

describe("generateExamples", () => {
  it("returns three examples", () => {
    const examples = generateExamples()
    expect(examples.length).toBe(3)
  })

  it("each example has label and code", () => {
    const examples = generateExamples()
    for (const ex of examples) {
      expect(typeof ex.label).toBe("string")
      expect(ex.label.length).toBeGreaterThan(0)
      expect(typeof ex.code).toBe("string")
      expect(ex.code.length).toBeGreaterThan(0)
    }
  })

  it("examples contain def run or async def run", () => {
    const examples = generateExamples()
    for (const ex of examples) {
      expect(
        ex.code.includes("def run(") || ex.code.includes("async def run("),
      ).toBe(true)
    }
  })

  it("parallel example uses threads", () => {
    const examples = generateExamples()
    const parallel = examples.find((e) => e.label === "Parallel with Retries")
    expect(parallel).toBeDefined()
    expect(parallel!.code).toContain("ThreadPoolExecutor")
  })

  it("async example uses async_tools", () => {
    const examples = generateExamples()
    const asyncEx = examples.find((e) => e.label === "Async Fan-Out")
    expect(asyncEx).toBeDefined()
    expect(asyncEx!.code).toContain("async_tools")
  })

  it("filter example uses json.loads", () => {
    const examples = generateExamples()
    const filterEx = examples.find((e) => e.label === "Filter & Transform")
    expect(filterEx).toBeDefined()
    expect(filterEx!.code).toContain("json.loads")
  })
})

describe("plainTextParamsSchema", () => {
  it("returns a valid JSON Schema object with a single 'input' string property", () => {
    const schema = plainTextParamsSchema()
    expect(schema.type).toBe("object")
    expect(schema.additionalProperties).toBe(false)
    expect(schema.required).toEqual([])
    const props = schema.properties as Record<
      string,
      { type: string; title: string; description: string }
    >
    expect(Object.keys(props)).toEqual(["input"])
    expect(props.input.type).toBe("string")
    expect(props.input.title).toBe("input")
    expect(typeof props.input.description).toBe("string")
    expect(props.input.description.length).toBeGreaterThan(0)
  })

  it("returns a fresh object each call (no shared mutation)", () => {
    const a = plainTextParamsSchema()
    const b = plainTextParamsSchema()
    expect(a).toEqual(b)
    expect(a).not.toBe(b)
  })
})

describe("plainTextParamsSchema integration with generateCodeToolPlaceholder", () => {
  it("produces def run(input: str | None = None) -> str:", () => {
    const schema = plainTextParamsSchema()
    const result = generateCodeToolPlaceholder(schema, "A tool")
    expect(result).toContain("def run(input: str | None = None) -> str:")
  })

  it("plain-text schema differs from empty structured schema", () => {
    const plainText = plainTextParamsSchema()
    const emptyStructured = {
      type: "object",
      properties: {},
      required: [],
      additionalProperties: false,
    }

    const plainResult = generateCodeToolPlaceholder(plainText, "test")
    const structuredResult = generateCodeToolPlaceholder(
      emptyStructured,
      "test",
    )

    expect(plainResult).toContain("def run(input: str | None = None) -> str:")
    expect(structuredResult).toContain("def run() -> str:")
    expect(plainResult).not.toEqual(structuredResult)
  })
})

describe("formatParamPreview", () => {
  it("returns empty string for undefined", () => {
    expect(formatParamPreview(undefined)).toBe("")
  })

  it("returns empty string for null", () => {
    expect(formatParamPreview(null)).toBe("")
  })

  it("returns string unchanged", () => {
    expect(formatParamPreview("hello")).toBe("hello")
  })

  it("returns long string unchanged", () => {
    const long = "a".repeat(500)
    expect(formatParamPreview(long)).toBe(long)
  })

  it("renders number as string", () => {
    expect(formatParamPreview(42)).toBe("42")
  })

  it("renders float as string", () => {
    expect(formatParamPreview(3.14)).toBe("3.14")
  })

  it("renders boolean true", () => {
    expect(formatParamPreview(true)).toBe("true")
  })

  it("renders boolean false", () => {
    expect(formatParamPreview(false)).toBe("false")
  })

  it("serialises object to JSON", () => {
    expect(formatParamPreview({ a: 1 })).toBe('{"a":1}')
  })

  it("serialises array to JSON", () => {
    expect(formatParamPreview([1, 2, 3])).toBe("[1,2,3]")
  })
})
