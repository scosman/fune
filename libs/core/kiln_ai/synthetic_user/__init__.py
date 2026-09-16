"""Local synthetic-user player.

Plays the synthetic-user side of a conversation locally, calling the LLM
with the user's own provider keys.

Public surface:

- `SyntheticUserDriver` — construct once per case, call `respond()` per turn.
- `SyntheticUserInfo` / `SyntheticUserDriverConfig` — typed configs.
- `SyntheticUserCase` — input contract for the multi-turn drive loop.
- `parse_synthetic_user_info` / `build_synthetic_user_info` — tagged-blob codec.
- `SyntheticUserInfoParseError` — raised on malformed blob.
- `role_swap` — exposed for callers that drive the loop themselves.
- `drive_case_for_eval` — transient one-case drive for the eval runner.
"""

from kiln_ai.synthetic_user.case import SyntheticUserCase
from kiln_ai.synthetic_user.driver import SyntheticUserDriver
from kiln_ai.synthetic_user.eval_drive import drive_case_for_eval
from kiln_ai.synthetic_user.models import (
    SyntheticUserDriverConfig,
    SyntheticUserInfo,
    VisibleMessageRole,
)
from kiln_ai.synthetic_user.parser import (
    SyntheticUserInfoParseError,
    build_synthetic_user_info,
    parse_synthetic_user_info,
)
from kiln_ai.synthetic_user.role_swap import role_swap

__all__ = [
    "SyntheticUserCase",
    "SyntheticUserDriver",
    "SyntheticUserDriverConfig",
    "SyntheticUserInfo",
    "SyntheticUserInfoParseError",
    "VisibleMessageRole",
    "build_synthetic_user_info",
    "drive_case_for_eval",
    "parse_synthetic_user_info",
    "role_swap",
]
