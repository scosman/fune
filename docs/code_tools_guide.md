# Code Tools: Authoring Guide

Code tools let you write Python functions that run as tools inside Kiln's agent harness. A code tool can call other tools it has been granted access to, making it ideal for orchestration, filtering, and multi-step workflows.

This guide covers the authoring contract: how to write the Python, how to call other tools, and how results and errors work.

## Entry point

Every code tool must define a module-level function named `run`. Both sync and async forms are supported:

```python
def run(query: str, max_results: int = 10) -> str:
    """Search and return filtered results."""
    # your code here
    return "result"
```

```python
async def run(urls: list[str], concurrency: int = 50) -> list:
    """Fetch URLs concurrently."""
    import asyncio
    # your async code here
    results = await asyncio.gather(*(fetch(u) for u in urls))
    return results
```

**Parameters** are passed as keyword arguments matching the tool's `parameters_schema` (JSON Schema). Optional parameters not provided by the caller are simply not passed -- use Python defaults.

**Return values** become the tool output string sent to the model:
- `str` passes through as-is.
- `dict`, `list`, `int`, `float`, `bool`, `None` are JSON-serialized (`json.dumps`).
- Anything else that isn't JSON-serializable is an error.

**Async notes:** When using `async def run`, the runtime owns the event loop. Your code should `await` and use `asyncio.gather` freely. Do not call `asyncio.run()` inside an async `run` -- there is already a running loop. A sync `def run` may call `asyncio.run()` if it needs async internally.

## Environment

- **Interpreter**: The Kiln app's bundled Python. You can import anything from the standard library and anything shipped in the Kiln bundle.
- **No sandbox**: There are no import restrictions or resource limits beyond the wall-clock timeout. Code runs on your machine with full access.
- **Security**: Code tools only execute in trusted projects. The trust gate prevents execution in untrusted projects.

## Calling other tools

Code tools can call other tools they've been granted access to via the allowlist. Two modules are available inside the code tool environment:

```python
from kiln import tools          # sync -- blocks until the tool returns
from kiln import async_tools    # async -- awaitable, concurrent under gather
```

### Sync usage

```python
from kiln import tools

def run(user_id: str) -> str:
    result = tools.get_user(id=user_id)
    return result
```

### Async usage

```python
from kiln import async_tools
import asyncio

async def run(user_ids: list[str]) -> str:
    results = await asyncio.gather(
        *(async_tools.get_user(id=uid) for uid in user_ids)
    )
    return results
```

Use `async_tools` in async code for true concurrency. Using `tools` (sync) inside an `async def run` blocks the event loop -- it works but defeats the purpose of async.

### Tool call results are always `str`

Every tool call returns a string -- byte-for-byte what the model would see from that tool. When a tool returns JSON, parse it yourself:

```python
import json
from kiln import tools

def run(query: str) -> str:
    raw = tools.search(query=query)
    results = json.loads(raw)  # parse the JSON string
    filtered = [r for r in results if r.get("score", 0) > 0.5]
    return json.dumps(filtered)
```

This is deliberate: results are always strings for type stability and fidelity to what the model sees. `json.loads` is one line.

### Listing available tools

```python
from kiln import tools

def run() -> str:
    available = tools.list_tools()
    # Returns: [{"name": "...", "description": "...", "parameters_schema": {...}}, ...]
    return available
```

`list_tools()` is a reserved name that returns the tools in your allowlist. It takes priority over any allowlisted tool with the same name.

## Exceptions

Tool calls can raise typed exceptions. Import them from either module:

```python
from kiln.tools import ToolNotAllowed, ToolTimeout, ToolCallError
# or equivalently:
from kiln.async_tools import ToolNotAllowed, ToolTimeout, ToolCallError
```

| Exception | When |
|---|---|
| `ToolNotAllowed` | The tool name is not in the allowlist, or an allowlisted tool can no longer be resolved (e.g. its MCP server was deleted) and so is left out of your tool map. The `.message` lists available tool names. |
| `ToolTimeout` | A nested tool call timed out. |
| `ToolCallError` | Everything else: the tool returned an error, arguments failed schema validation, or positional arguments were used instead of keyword arguments. The `.message` includes the expected parameter schema. Has `.tool`, `.message`, and `.raw` (the raw output string when available). |

Use these for retry logic:

```python
from kiln import tools
from kiln.tools import ToolCallError

def run(item_id: str) -> str:
    for attempt in range(3):
        try:
            return tools.fetch_item(id=item_id)
        except ToolCallError:
            if attempt == 2:
                raise
```

## Timeout and allowlist

- **Timeout**: Wall-clock timeout for one invocation, including time spent in nested tool calls. Default is 60 seconds, minimum 1, no maximum. Set when creating the tool.
- **Allowlist**: Explicit list of tools this code tool may call. Only tools in the allowlist are callable. Tools are resolved by the canonical function name (the name returned by `list_tools()`).
- **Nesting**: Code tools can call other code tools (they're just tools). A maximum nesting depth of 10 prevents runaways.

## Concurrency

You can use threads and asyncio freely. Tool calls are safe from multiple threads or gathered coroutines simultaneously. The harness executes them concurrently.

There is no built-in batch/parallelism helper -- use standard library constructs (`ThreadPoolExecutor`, `asyncio.gather`, etc.) as shown in the examples below.

## Examples

### Parallel with retries (threads + `tools`)

Fetch multiple URLs in parallel with exponential backoff retries, using the sync `tools` API from worker threads:

```python
import json
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
```

### Async fan-out (`async_tools` + `gather`)

Fetch user details concurrently using the async API for true parallelism:

```python
import json
import asyncio
from kiln import async_tools

async def run(user_ids: list[str]) -> str:
    """Fetch user details concurrently using async_tools."""
    async def fetch_user(uid):
        result = await async_tools.get_user(id=uid)
        return json.loads(result)

    users = await asyncio.gather(*(fetch_user(uid) for uid in user_ids))
    return json.dumps(users)
```

### Filter and transform (`json.loads` result filtering)

Search and filter results, demonstrating how to parse and reshape tool output:

```python
import json
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
```

## Testing your tool

Kiln stores your tool's source as a real Python file named `tool.py`, right next to the tool's `code_tool.kiln`:

```
{project}/code_tools/{id} - {name}/
  ├── code_tool.kiln   # metadata (no code)
  └── tool.py          # your source -- byte-for-byte what runs
```

Because it's a plain, importable file, you can write standard `pytest` tests against it -- no Kiln-specific runner, no sandbox. Kiln does not store, display, or run these tests; testing happens in a normal Python environment with `kiln_ai` installed.

### Setup

Install `kiln_ai` into your environment (`pip install kiln-ai`, or your project's dev dependencies). That's all the setup required: `kiln_ai` ships a `pytest` plugin (the `kiln` test shim) that is auto-discovered, so the `from kiln import tools` at the top of your `tool.py` resolves under `pytest` and a `kiln_tools` fixture becomes available.

### Writing a test

Create `test_tool.py` **in the same folder** as `tool.py`, then `import tool` and drive `run(...)` directly:

```python
import json

import pytest

import tool  # the artifact's tool.py -- imports cleanly under pytest


def test_happy_path(kiln_tools):
    kiln_tools.set("get_user", '{"id": 1234, "name": "Alice"}')      # static reply
    kiln_tools.set("list_jobs", lambda **kw: json.dumps(["a", "b"]))  # or a callable

    out = tool.run(job_ids=["a", "b"])

    assert json.loads(out)["name"] == "Alice"
    assert kiln_tools.calls[0].name == "get_user"                     # call assertions


def test_unknown_tool_raises(kiln_tools):
    from kiln.tools import ToolNotAllowed

    # A name that was never registered behaves like a non-allowlisted tool.
    with pytest.raises(ToolNotAllowed):
        tool.run(job_ids=["a"])
```

Then run `pytest` from that folder.

### The `kiln_tools` fixture

The fixture stubs the tools your code calls and records the calls it makes. It is function-scoped and auto-resets between tests.

| Member | What it does |
|---|---|
| `kiln_tools.set(name, reply)` | Register a reply for a tool name. `reply` is a `str` (returned verbatim, matching the string-returns contract) or a callable `(**kwargs) -> str`. |
| `kiln_tools.set_error(name, exc)` | Make a tool name raise a given `ToolNotAllowed` / `ToolTimeout` / `ToolCallError`. |
| `kiln_tools.calls` | Ordered record of the calls made, each with `.name` and `.arguments`. |
| `list_tools()` | `tools.list_tools()` returns the declarations you registered. |

Fidelity notes -- the shim behaves like the real runtime:

- An **unregistered** tool name raises `ToolNotAllowed` (the same as an allowlist miss).
- Calling a tool with **positional** arguments raises `ToolCallError` (tool calls are keyword-only).
- The exception classes are the *same* classes the runtime raises, so `except kiln.tools.ToolCallError` catches real runtime behavior.
- `async_tools` is backed by the same registry, so `await asyncio.gather(async_tools.a(...), async_tools.b(...))` works in tests of async `run`.

### Caveat: one tool folder per `pytest` run

The file is always named `tool.py`, so two different tool folders both expose a top-level module named `tool`. A single `pytest` run that spans multiple tool folders will hit a module-name collision. Test **one tool folder at a time** (run `pytest` from inside that folder), or use `pytest --import-mode=importlib`. This is a `pytest` import-mode detail, not a Kiln limitation.

## Testing your judge

Code judges (code evals) follow the same pattern. Kiln stores the judge's source as `scorer.py` beside its `eval_config.kiln`:

```
{task}/.../eval_configs/{id} - {name}/
  ├── eval_config.kiln   # metadata (no code)
  └── scorer.py          # your score() source
```

Judges need **no shim**: `score()` receives its inputs as plain keyword arguments and depends only on the standard library plus `kiln_ai`, so you can import and call it directly.

### The `score()` contract

`score()` may declare any of these parameters; at runtime Kiln passes only the ones your function actually declares (it must accept at least `output` or `trace`):

- `output` -- the model's final output string.
- `trace` -- the conversation as a list of message dicts.
- `reference_data` -- a dict of ground-truth data for the case, or `None`. For an EvalInput-backed case it is that input's `reference` dict (`None` if none is set); for a TaskRun-backed dataset case it is `{"reference_answer": <the item's stored output>}`, since that output is the curated answer. It is also `None` when a TaskRun is scored as itself (judge calibration), so scorers must not assume `None` means "TaskRun dataset".
- `task_input` -- the original task input string.

It returns a `dict` keyed by each of your eval's output scores' **JSON key** -- the score's display name normalized to lowercase, snake_case (for example a score named `"Exact Match"` has the key `"exact_match"`, and `"Accuracy"` has the key `"accuracy"`). At runtime the code-eval adapter checks the returned keys against exactly this set and raises a "Score key mismatch" error otherwise, so return the JSON keys, not the raw display names.

### Writing a test

Create `test_scorer.py` beside `scorer.py`. Key your assertions by the JSON key, matching what the real eval expects:

```python
from scorer import score


# For an eval whose output score is named "Exact Match" (JSON key: "exact_match").
def test_scores_exact_match():
    result = score(output="42", reference_data={"answer": "42"})
    assert result["exact_match"] == 1.0  # the JSON key, not "Exact Match"


def test_scores_mismatch():
    result = score(output="41", reference_data={"answer": "42"})
    assert result["exact_match"] == 0.0
```

Then run `pytest`. The same one-folder-at-a-time caveat applies -- `scorer.py` is a fixed name, so run `pytest` from inside a single eval-config folder (or use `--import-mode=importlib`) to avoid module-name collisions across folders.
