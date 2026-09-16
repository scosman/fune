"""Unit tests for the tagged blob parser/builder."""

import pytest

from kiln_ai.synthetic_user.models import SyntheticUserInfo
from kiln_ai.synthetic_user.parser import (
    SyntheticUserInfoParseError,
    build_synthetic_user_info,
    parse_synthetic_user_info,
)

# ───────────────────────── parse ─────────────────────────


def test_parse_all_three_tags() -> None:
    blob = "<persona>P</persona><goal>G</goal><behavior_guidance>B</behavior_guidance>"
    info = parse_synthetic_user_info(blob)
    assert info.persona == "P"
    assert info.goal == "G"
    assert info.behavior_guidance == "B"


def test_parse_only_required_tags() -> None:
    blob = "<persona>P</persona><goal>G</goal>"
    info = parse_synthetic_user_info(blob)
    assert info.behavior_guidance is None


def test_parse_strips_whitespace() -> None:
    blob = (
        "<persona>   P with spaces   </persona>"
        "<goal>\n\tG with newlines\n</goal>"
        "<behavior_guidance>\n B\n</behavior_guidance>"
    )
    info = parse_synthetic_user_info(blob)
    assert info.persona == "P with spaces"
    assert info.goal == "G with newlines"
    assert info.behavior_guidance == "B"


def test_parse_dotall_allows_multiline_content() -> None:
    blob = "<persona>line 1\nline 2\nline 3</persona><goal>G</goal>"
    info = parse_synthetic_user_info(blob)
    assert info.persona == "line 1\nline 2\nline 3"


def test_parse_ignores_unknown_tags() -> None:
    # Forward-compat: a future generator may add new tags like <tone>.
    blob = (
        "<persona>P</persona>"
        "<goal>G</goal>"
        "<tone>cheerful</tone>"
        "<knowledge_level>expert</knowledge_level>"
    )
    info = parse_synthetic_user_info(blob)
    assert info.persona == "P"
    assert info.goal == "G"
    # Unknown tags don't end up anywhere.
    assert not hasattr(info, "tone")


def test_parse_ignores_preamble_and_trailing() -> None:
    blob = "preamble text <persona>P</persona><goal>G</goal> trailing text"
    info = parse_synthetic_user_info(blob)
    assert info.persona == "P"
    assert info.goal == "G"


def test_parse_first_match_wins_for_duplicate_tag() -> None:
    blob = "<persona>first</persona><persona>second</persona><goal>G</goal>"
    info = parse_synthetic_user_info(blob)
    assert info.persona == "first"


def test_parse_missing_persona_raises() -> None:
    with pytest.raises(SyntheticUserInfoParseError, match="<persona>"):
        parse_synthetic_user_info("<goal>G</goal>")


def test_parse_missing_goal_raises() -> None:
    with pytest.raises(SyntheticUserInfoParseError, match="<goal>"):
        parse_synthetic_user_info("<persona>P</persona>")


def test_parse_empty_persona_raises() -> None:
    with pytest.raises(SyntheticUserInfoParseError, match="<persona>"):
        parse_synthetic_user_info("<persona></persona><goal>G</goal>")


def test_parse_whitespace_only_persona_raises() -> None:
    # After trim, content is empty — should be treated as missing.
    with pytest.raises(SyntheticUserInfoParseError, match="<persona>"):
        parse_synthetic_user_info("<persona>   \n  </persona><goal>G</goal>")


def test_parse_completely_unstructured_blob_raises() -> None:
    with pytest.raises(SyntheticUserInfoParseError):
        parse_synthetic_user_info("just plain text with no tags at all")


# ───────────────────────── build ─────────────────────────


def test_build_all_three_tags() -> None:
    info = SyntheticUserInfo(persona="P", goal="G", behavior_guidance="B")
    assert (
        build_synthetic_user_info(info)
        == "<persona>P</persona><goal>G</goal><behavior_guidance>B</behavior_guidance>"
    )


def test_build_omits_behavior_guidance_when_none() -> None:
    info = SyntheticUserInfo(persona="P", goal="G")
    assert build_synthetic_user_info(info) == "<persona>P</persona><goal>G</goal>"


def test_build_omits_empty_behavior_guidance() -> None:
    # The model permits None for behavior_guidance; build's truthiness check
    # also skips empty strings, which matches the parser's "missing → None".
    info = SyntheticUserInfo(persona="P", goal="G", behavior_guidance="")
    assert build_synthetic_user_info(info) == "<persona>P</persona><goal>G</goal>"


# ───────────────────────── roundtrip ─────────────────────────


def test_roundtrip_all_three() -> None:
    info = SyntheticUserInfo(persona="P\nmultiline", goal="G", behavior_guidance="B")
    assert parse_synthetic_user_info(build_synthetic_user_info(info)) == info


def test_roundtrip_required_only() -> None:
    info = SyntheticUserInfo(persona="P", goal="G")
    assert parse_synthetic_user_info(build_synthetic_user_info(info)) == info


def test_roundtrip_preserves_internal_whitespace() -> None:
    # Whitespace WITHIN content (after the leading/trailing trim) is preserved.
    info = SyntheticUserInfo(
        persona="word1  word2   word3", goal="G", behavior_guidance=None
    )
    assert (
        parse_synthetic_user_info(build_synthetic_user_info(info)).persona
        == info.persona
    )
