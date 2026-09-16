# Cross-OS Spawn Sanity Checklist

Code tools and code evals share the same multiprocessing spawn infrastructure (`kiln_ai/sandbox/spawn.py`). This checklist documents what must be verified on each platform before a release that includes code tools or code evals.

## Shared infrastructure

- `_spawn_lock` (threading.Lock) is process-global, shared between code-evals (`run_scorer`) and code tools (`PythonCodeTool`). Both features MUST use the same lock to prevent PyInstaller bug #7410 exposure during concurrent spawns.
- `start_process_with_light_main(p)` swaps `sys.modules["__main__"]` to a lightweight stub during `p.start()`, preventing the child from re-importing the heavy parent main module (e.g. `desktop.py` which pulls in litellm, google-cloud, etc.).
- `call_entrypoint(fn, kwargs)` handles both sync and async entry points, using `asyncio.run()` for async functions.
- The child process uses `multiprocessing.get_context("spawn")` explicitly (never fork), with `daemon=True`.

## macOS

- [x] **Verified on this host (2026-07-05)**: multiprocessing spawn context works correctly. The full test suite (`test_code_tool_execution.py`, `test_sandbox_shared.py`, and the code-eval suite) passes, confirming:
  - Sync and async `run()` entry points
  - Nested tool calls via IPC queues
  - Timeout and crash handling
  - Concurrent spawn under `_spawn_lock`
  - stdout/stderr capture
  - Return-value serialization

## Windows

- [ ] **Must verify before release**: Run the full test suite on a Windows build machine.
- Key concerns:
  - `multiprocessing.freeze_support()` is called in `desktop.py` (line 186) and `dev_server.py` (line 25). This MUST execute before any spawn-context process creation. Code tools do not call `freeze_support()` themselves — they rely on the app entry point having called it.
  - `start_process_with_light_main` stays within `multiprocessing.spawn`'s bootstrap, so `freeze_support()` continues to work.
  - Verify that PyInstaller frozen builds can spawn child processes for code tools (same validation as code-evals).
  - Verify `_spawn_lock` prevents concurrent-spawn issues under PyInstaller on Windows.

## Linux

- [ ] **Must verify before release**: Run the full test suite on a Linux build machine.
- Key concerns:
  - Same `freeze_support()` and `_spawn_lock` requirements as Windows.
  - PyInstaller bug #7410 (concurrent spawn crash) was originally reported on Linux — this is the platform where the `_spawn_lock` is most critical.
  - Verify that the `__main__` stub-swap works correctly in Linux frozen builds.

## Frozen build (PyInstaller) verification

For each platform (macOS, Windows, Linux):

1. Build the frozen app with PyInstaller.
2. Launch the app and create a code tool with a simple `def run()` that returns a string.
3. Test the code tool from the UI test panel.
4. Create a code tool that calls another tool (e.g. a demo math tool) and test it.
5. Verify that code-evals still work (shared `_spawn_lock` regression check).
6. Run two tests concurrently if possible to verify the spawn lock serializes correctly.

## Notes

- The existing code-eval frozen-build verification covers most of the spawn infrastructure. Code tools add the nested-tool-call IPC layer on top, which is the primary new thing to verify.
- stdout/stderr are redirected to `StringIO` in the child process. This also serves as the PyInstaller `--windowed` None-stdout guard (on macOS `.app` and Windows GUI builds, `sys.stdout` can be `None`).
