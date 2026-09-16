"""Parse / build the tagged synthetic_user_info blob.

The blob is a serialization format only — persisted eval inputs store
the parsed SyntheticUserInfo, never the blob.

Tag rules:
- First match per known tag, non-greedy content; nested tags not supported.
- Whitespace trimmed from each tag's content.
- Unknown tags are ignored (forward-compat for future generator output).
- `<persona>` and `<goal>` are required; parse raises if missing or empty.
- `<behavior_guidance>` is optional; missing → that section is omitted
  from the rendered system prompt.
"""

import re

from kiln_ai.synthetic_user.models import SyntheticUserInfo


class SyntheticUserInfoParseError(ValueError):
    """The blob is missing a required tag, or all required tags are empty."""


def _extract(blob: str, tag: str) -> str | None:
    """First match, non-greedy content. Returns trimmed content, or None."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", blob, re.DOTALL)
    if m is None:
        return None
    return m.group(1).strip()


def parse_synthetic_user_info(blob: str) -> SyntheticUserInfo:
    """Parse the tagged blob. Strict on required tags; lenient on optional."""
    persona = _extract(blob, "persona")
    if not persona:
        raise SyntheticUserInfoParseError(
            "Missing or empty required tag <persona> in synthetic_user_info blob."
        )
    goal = _extract(blob, "goal")
    if not goal:
        raise SyntheticUserInfoParseError(
            "Missing or empty required tag <goal> in synthetic_user_info blob."
        )
    behavior_guidance = _extract(blob, "behavior_guidance") or None
    return SyntheticUserInfo(
        persona=persona,
        goal=goal,
        behavior_guidance=behavior_guidance,
    )


def build_synthetic_user_info(info: SyntheticUserInfo) -> str:
    """Inverse of parse — for tests and any callers that need to construct a blob."""
    parts = [
        f"<persona>{info.persona}</persona>",
        f"<goal>{info.goal}</goal>",
    ]
    if info.behavior_guidance:
        parts.append(f"<behavior_guidance>{info.behavior_guidance}</behavior_guidance>")
    return "".join(parts)
