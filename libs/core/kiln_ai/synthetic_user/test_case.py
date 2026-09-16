"""Unit tests for SyntheticUserCase."""

import pytest
from pydantic import ValidationError

from kiln_ai.synthetic_user.case import SyntheticUserCase


def test_case_accepts_valid_fields() -> None:
    case = SyntheticUserCase(
        seed_prompt="hi there",
        synthetic_user_info="<persona>p</persona><goal>g</goal>",
    )
    assert case.seed_prompt == "hi there"
    assert "persona>p</persona" in case.synthetic_user_info


def test_case_rejects_empty_seed_prompt() -> None:
    with pytest.raises(ValidationError):
        SyntheticUserCase(
            seed_prompt="", synthetic_user_info="<persona>p</persona><goal>g</goal>"
        )


def test_case_rejects_empty_synthetic_user_info() -> None:
    with pytest.raises(ValidationError):
        SyntheticUserCase(seed_prompt="hi", synthetic_user_info="")
