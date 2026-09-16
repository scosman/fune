"""Internal models for the synthetic-user player.

`SyntheticUserInfo` is declared in the datamodel (it persists on multi-turn
eval inputs) and re-exported here for the runtime side: the parser produces
it from the tagged wire blob and `prompt.render_system_prompt` consumes it.
`SyntheticUserDriverConfig` carries the per-eval runtime config — model,
provider, role visibility.
"""

from typing import Literal

from pydantic import BaseModel, Field

from kiln_ai.datamodel.datamodel_enums import ModelProviderName
from kiln_ai.datamodel.eval import SyntheticUserInfo as SyntheticUserInfo

VisibleMessageRole = Literal["user", "assistant"]

# The exact message the SU sends, alone and with nothing else, to end a
# conversation before the turn ceiling is reached. It lives in this leaf module
# so the drive loop that acts on it and the prompt that teaches it can share
# one string without importing each other.
EARLY_STOP_SENTINEL = "<DONE>"

# The tag written on a conversation the SU ended with the sentinel: the
# persisted counterpart of the string above, marking a trace that is complete
# even though it holds fewer turns than the item asked for. It is on disk
# because the eval runner's completeness gate has to tell such a trace apart
# from a truncated one (a partial record, a drive that died mid-way) long after
# the drive that produced it is gone, and the trace alone cannot say which it
# is. It lives beside the sentinel — same concept, same lifetime — so a reader
# of a stored trace need not import the batch runner for one string.
TAG_SU_ENDED_CONVERSATION = "synthetic_user_ended_conversation"


class SyntheticUserDriverConfig(BaseModel):
    """Per-eval runtime config for the SU's LLM driver.

    No `temperature` field — runs at the chosen model's default. The driver
    intentionally does not own temperature: the persona-playing prompt and
    `behavior_guidance` carry style; temperature is a model-level concern
    surfaced elsewhere when it matters.
    """

    model_name: str
    model_provider_name: ModelProviderName
    visible_message_roles: list[VisibleMessageRole] = Field(
        default_factory=lambda: ["user", "assistant"]
    )
