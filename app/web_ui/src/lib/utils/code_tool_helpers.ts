/**
 * Helpers for Code Tool UI: typed placeholder codegen, import-helper, examples.
 */

/**
 * Return a static JSON Schema for plain-text parameter mode.
 * When the user selects "Plain Text" instead of "Structured Parameter List",
 * the tool still needs a single `input` string parameter so the model knows
 * to pass text and the placeholder codegen produces
 * `def run(input: str | None = None) -> str:`.
 */
export function plainTextParamsSchema(): { [key: string]: unknown } {
  return {
    type: "object",
    properties: {
      input: {
        type: "string",
        title: "input",
        description: "Plain text input passed to the tool.",
      },
    },
    required: [],
    additionalProperties: false,
  }
}

interface JsonSchemaProperty {
  type?: string
  items?: JsonSchemaProperty
  properties?: Record<string, JsonSchemaProperty>
  required?: string[]
  description?: string
}

/**
 * Map a JSON Schema type to a Python type hint string.
 */
function jsonTypeToPython(prop: JsonSchemaProperty): string {
  switch (prop.type) {
    case "string":
      return "str"
    case "integer":
      return "int"
    case "number":
      return "float"
    case "boolean":
      return "bool"
    case "array": {
      if (prop.items) {
        return `list[${jsonTypeToPython(prop.items)}]`
      }
      return "list"
    }
    case "object":
      return "dict"
    default:
      return "str"
  }
}

export interface CodeToolParam {
  name: string
  pythonType: string
  required: boolean
}

/**
 * Extract typed parameter info from a JSON Schema object.
 */
export function extractParams(schema: {
  [key: string]: unknown
}): CodeToolParam[] {
  const properties = schema.properties as
    | Record<string, JsonSchemaProperty>
    | undefined
  if (!properties) return []

  const required = Array.isArray(schema.required)
    ? (schema.required as string[])
    : []

  return Object.entries(properties).map(([name, prop]) => ({
    name,
    pythonType: jsonTypeToPython(prop),
    required: required.includes(name),
  }))
}

/**
 * Build the Python parameter list string for def run(...).
 * Required params come first, then optional params with `| None = None`.
 */
function buildParamList(params: CodeToolParam[]): string {
  if (params.length === 0) return ""

  const requiredParams = params.filter((p) => p.required)
  const optionalParams = params.filter((p) => !p.required)

  // Schema property names are used verbatim: the sandbox calls run() with
  // keyword arguments keyed by these exact names, so renaming any would break the call.
  const parts: string[] = []
  for (const p of requiredParams) {
    parts.push(`${p.name}: ${p.pythonType}`)
  }
  for (const p of optionalParams) {
    parts.push(`${p.name}: ${p.pythonType} | None = None`)
  }
  return parts.join(", ")
}

/**
 * Generate a typed placeholder `def run(...)` stub from the schema.
 */
export function generateCodeToolPlaceholder(
  schema: { [key: string]: unknown },
  toolDescription: string,
): string {
  const params = extractParams(schema)
  const paramList = buildParamList(params)
  // Escape backslashes then double-quotes so the description can neither close the
  // triple-quoted docstring nor leave a dangling escape (e.g. a trailing " or \).
  const safeDesc = toolDescription.replace(/\\/g, "\\\\").replace(/"/g, '\\"')

  return `def run(${paramList}) -> str:
    """${safeDesc}"""
    # Your code here
    return "result"
`
}

/**
 * Generate the import block to prepend when tools are selected.
 */
export function generateImportHelper(functionName: string): string {
  return `# Run tools with \`tools.${functionName}(...)\` or \`await async_tools.${functionName}(...)\`
from kiln import tools, async_tools

`
}

/**
 * Check whether the import line is already present in the code.
 */
export function shouldInsertImport(currentCode: string): boolean {
  return !currentCode.includes("from kiln import tools")
}

/**
 * Check if the code is still the original generated placeholder (or empty),
 * meaning it's safe to regenerate it after a schema change.
 */
export function isCodeUnmodified(
  currentCode: string,
  originalPlaceholder: string,
): boolean {
  return currentCode.trim() === "" || currentCode === originalPlaceholder
}

export interface Step2CodeResolution {
  code: string
  generatedPlaceholder: string
  schemaChangedHint: boolean
  // Whether the clone's seed code was used. When true the caller drops it so a
  // later return to the Code step doesn't overwrite the user's edits with it.
  cloneConsumed: boolean
}

/**
 * Decide the Code step's editor contents when the user advances from the
 * Define step.
 *
 * Keyed on whether code already exists — not on the wizard step — because the
 * browser Back button pops the shallow-routing step state, so the step alone
 * can't tell a first visit from a return. Defaulting to "first visit" would
 * overwrite the user's authored code with a fresh placeholder. The cases:
 *
 *  - No code yet: seed from the clone (if any), else the generated placeholder.
 *  - Untouched placeholder: regenerate it for the (possibly changed) schema.
 *  - User-authored code: preserve it, and flag when the schema changed so the
 *    user knows to reconcile run()'s parameters.
 */
export function resolveStep2Code(args: {
  code: string
  newPlaceholder: string
  generatedPlaceholder: string
  schemaChangedHint: boolean
  cloneCode: string | null
}): Step2CodeResolution {
  const {
    code,
    newPlaceholder,
    generatedPlaceholder,
    schemaChangedHint,
    cloneCode,
  } = args

  if (code === "") {
    const useClone = !!cloneCode
    return {
      code: useClone ? (cloneCode as string) : newPlaceholder,
      generatedPlaceholder: newPlaceholder,
      schemaChangedHint: false,
      cloneConsumed: useClone,
    }
  }

  if (isCodeUnmodified(code, generatedPlaceholder)) {
    return {
      code: newPlaceholder,
      generatedPlaceholder: newPlaceholder,
      schemaChangedHint: false,
      cloneConsumed: false,
    }
  }

  if (newPlaceholder !== generatedPlaceholder) {
    return {
      code,
      generatedPlaceholder: newPlaceholder,
      schemaChangedHint: true,
      cloneConsumed: false,
    }
  }

  // User-authored code and the schema is unchanged: leave everything as-is.
  return {
    code,
    generatedPlaceholder,
    schemaChangedHint,
    cloneConsumed: false,
  }
}

/**
 * Format a parameter value for inline preview display.
 *
 * - `null` / `undefined` → empty string (the component renders a "—" fallback)
 * - strings are returned as-is
 * - other values are JSON-serialised
 */
export function formatParamPreview(value: unknown): string {
  if (value === null || value === undefined) return ""
  return typeof value === "string" ? value : JSON.stringify(value)
}

/**
 * Generate example code snippets for the "More Examples" dialog.
 *
 * WARNING: These examples are validated by Python tests that execute the exact
 * code strings through the real code-tool engine. Do not modify the examples
 * without updating the corresponding tests in:
 *   libs/core/kiln_ai/sandbox/test_code_tool_execution.py
 *   (search for "TestUIExample")
 */
export function generateExamples(): { label: string; code: string }[] {
  return [
    {
      label: "Parallel with Retries",
      code: `import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from kiln import tools

def run(urls: list[str], max_retries: int = 3) -> str:
    """Fetch multiple URLs in parallel with retries."""
    results = {}

    def fetch_with_retry(url):
        for attempt in range(max_retries):
            try:
                result = tools.fetch_url(url=url)
                return url, json.loads(result)
            except Exception as e:
                if attempt == max_retries - 1:
                    return url, {"error": str(e)}
                time.sleep(0.5 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(fetch_with_retry, u) for u in urls]
        for future in as_completed(futures):
            url, data = future.result()
            results[url] = data

    return json.dumps(results)
`,
    },
    {
      label: "Async Fan-Out",
      code: `import json
import asyncio
from kiln import async_tools

async def run(user_ids: list[str]) -> str:
    """Fetch user details concurrently using async_tools."""
    async def fetch_user(uid):
        result = await async_tools.get_user(id=uid)
        return json.loads(result)

    users = await asyncio.gather(*(fetch_user(uid) for uid in user_ids))
    return json.dumps(users)
`,
    },
    {
      label: "Filter & Transform",
      code: `import json
from kiln import tools

def run(query: str, max_results: int = 10) -> str:
    """Search and filter results, returning only relevant fields."""
    raw = tools.search(query=query)
    results = json.loads(raw)

    filtered = [
        {"title": r["title"], "url": r["url"]}
        for r in results[:max_results]
        if "title" in r and "url" in r
    ]

    return json.dumps(filtered)
`,
    },
  ]
}
