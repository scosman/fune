"""SyntheticUserCase — input contract for the multi-turn SU runner.

Two-field shape: `seed_prompt` (the opening user message) and
`synthetic_user_info` (an opaque tagged blob; the runner parses it into a
typed SyntheticUserInfo before building the SyntheticUserDriver).

Callers holding a serialized case convert it to this model before
invoking the runner.
"""

from pydantic import BaseModel, Field


class SyntheticUserCase(BaseModel):
    """One case for the multi-turn SU drive loop.

    `seed_prompt` is the first user-side message sent into the target
    task. `synthetic_user_info` is the persona/goal/behavior_guidance
    blob, parsed by the caller into the typed SyntheticUserInfo the
    driver builds the SU's system prompt from.
    """

    seed_prompt: str = Field(..., min_length=1)
    synthetic_user_info: str = Field(..., min_length=1)
