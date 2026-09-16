from typing import Literal

from pydantic import Field

from kiln_ai.datamodel.basemodel import KilnParentedModel

DataGuideSource = Literal["manual", "kiln_pro"]


class DataGuide(KilnParentedModel):
    """Persistent input data guide for synthetic data generation, stored as a
    child of a Task.

    The guide describes what realistic *inputs* to this task look like — input
    shape, format, distribution, and the kinds of values inputs contain. It is
    consumed at the topic and input generation stages of synthetic data
    generation, never at the output stage. Output behavior is the job of the
    task's system prompt, not this guide.

    The guide is a single markdown body. Two canonical shapes are supported
    depending on origin:

    - **Manual flow** (user-authored examples): leads with `# Reference Inputs`
      (user-owned `## Example N` blocks), followed by `# Semantics`, `# Style`,
      `# Presentation Defaults`.
    - **Kiln Pro / Copilot flow** (analyze pipeline): only `# Semantics`,
      `# Style`, `# Presentation Defaults` — the analyze prompt derives rules
      from input documents rather than quoting them.

    The metaprompter treats the whole body as one editable artifact and returns
    a refined version on each refine pass; refine auto-detects which shape it
    is working on by checking for a `# Reference Inputs` heading.
    """

    guide: str = Field(
        default="",
        description="Markdown body of the input data guide. Manual-flow guides start with `# Reference Inputs`; Kiln Pro / Copilot-flow guides have only `# Semantics`, `# Style`, `# Presentation Defaults`.",
    )

    source: DataGuideSource = Field(
        default="manual",
        description="Which flow created this guide. Refine + verify pipelines branch on this to choose the right metaprompter (manual = user-curated examples + edits all sections; kiln_pro = LLM-derived, refine is feedback-only surgical edits).",
    )
