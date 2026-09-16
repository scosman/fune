---
status: complete
---

# Input Transformer UI

This project adds a basic UI for the **input transformers** feature recently added to task run configs (see the `templates` project / diff from `main` for the backend details).

Input transformers let a run config transform the raw task input before it's sent to the model. V1 ships one type: a Jinja2 template (`JinjaInputTransform`, `{ type: "jinja", template: str }`). The field lives on `KilnAgentRunConfigProperties.input_transform` and is optional/nullable (`None` = no transform, current behavior).

## The Problem

Previously there was zero indication in the UI of whether a run config had an input transformer. Example run-config details URL:
`/optimize/189194447825/920784333097/run_config/finetune_run_config::189194447825::920784333097::251360413936`

## Goals

- **Run config details — properties list:** Add an "Input Transformer" row. Value is "Custom Template" (a link/button opening a modal) for a Jinja2 transform, or "None" when absent.
- **Modal:** Clicking "Custom Template" opens a simple modal:
  - Title: "Input Transformer"
  - Subtitle: identifying info (note: a transform has no ID of its own in the data model — see open questions)
  - Body: a copyable text area showing the transform body (monospace). Reuse the existing body control — don't reinvent it.
  - Anything else worth adding (TBD).
- **Strong typing:** Use proper enum/discriminated-union types on the TypeScript side so that if a new `input_transform` type is added on the backend, we get a compile-time error until the UI is updated to support it. Use the repo's existing exhaustiveness pattern (`const _exhaustive: never = ...`).
- **Reuse coverage:** If the details page builds its property list via a shared component/function, the new row should appear everywhere that shared code is used.
- **Other summaries:** Find every place that summarizes a run config (model + prompt) and add an input-transformer indicator where appropriate. Known candidates:
  - The compare page (`/specs/.../compare`) — run config summary at the top of each column.
  - The run config selector (on `/run` and elsewhere) — currently shows model + prompt only.
  - Any others discovered during research.

## Out of Scope

- Authoring / editing input transformers in the UI (this is read-only display only).
- Backend changes to the input transformer feature.
