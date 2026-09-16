---
status: complete
---

# Input Transform Create UI

This project added the ability to create input transformers from the Kiln UI. (See the `templates` project for backend details about input transformers, and the `input_transform_ui` project for the existing read-only display of transforms — both are already built. This project adds **authoring**, which was explicitly out of scope for `input_transform_ui`.)

## Project Details

- Add an **"Input Transform"** dropdown under advanced options in `/run`. It is the **last** item in the advanced options.
- Info tooltip: "Transform the provided input using a jinja template, before sending the input to the model. Allows you to add context, or filter data."
- A fancy select with options: **"None"**, **"Create Template"**, and (if set) **"Custom Template"**.
  - "None" selected if no jinja template is set.
  - If the user picks "Create Template", show a modal allowing them to type in a jinja template.
  - If set to "Custom Template", the run passes in `"input_transform": { "type": "jinja", "template": "[TEMPLATE]" }`.

## Integration

- If the user selects a run config from the top "Run Configuration" dropdown, it should properly populate the template field (null or set to the right string), which updates the "Input Transform" dropdown state.
- If the user customizes the transformer via "Create Template" and sets it to a new string, the "Run Configuration" dropdown should jump to "Custom" if it was previously set to a saved config (the run config is no longer the same as the saved one). The user should be able to save this run configuration, including its new input transform.

## Create Modal UI

- Title: "Input Transform"
- Subtitle: "Transform the provided input using a jinja template, before sending the input to the model. Allows you to add context, or filter data."
- A textbox and a "Create" button.
- Add a simple API for validating that a template is valid jinja (we have helpers already). When attempting to "Create", call this and show an error if invalid. Only save if it passes.

## Out of Scope

- Backend changes to the input transformer engine/feature itself (already built in the `templates` project).
- Read-only display of transforms on run-config detail / summary surfaces (already built in the `input_transform_ui` project).
- Additional transform types beyond `JinjaInputTransform` (V1 has one type).
