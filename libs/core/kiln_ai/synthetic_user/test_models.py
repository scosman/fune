"""Unit tests for SyntheticUserInfo / SyntheticUserDriverConfig."""

import pytest
from pydantic import ValidationError

from kiln_ai.datamodel.datamodel_enums import ModelProviderName
from kiln_ai.synthetic_user.models import (
    SyntheticUserDriverConfig,
    SyntheticUserInfo,
)


def test_synthetic_user_info_required_fields() -> None:
    info = SyntheticUserInfo(persona="p", goal="g")
    assert info.persona == "p"
    assert info.goal == "g"
    assert info.behavior_guidance is None


def test_synthetic_user_info_accepts_behavior_guidance() -> None:
    info = SyntheticUserInfo(persona="p", goal="g", behavior_guidance="b")
    assert info.behavior_guidance == "b"


def test_synthetic_user_info_rejects_missing_persona() -> None:
    with pytest.raises(ValidationError):
        SyntheticUserInfo(goal="g")  # type: ignore[call-arg]


def test_synthetic_user_info_rejects_missing_goal() -> None:
    with pytest.raises(ValidationError):
        SyntheticUserInfo(persona="p")  # type: ignore[call-arg]


def test_driver_config_default_visible_roles() -> None:
    cfg = SyntheticUserDriverConfig(
        model_name="x", model_provider_name=ModelProviderName.openrouter
    )
    assert cfg.visible_message_roles == ["user", "assistant"]


def test_driver_config_default_is_per_instance() -> None:
    # Pydantic v2 + default_factory must not share the list across instances.
    a = SyntheticUserDriverConfig(
        model_name="x", model_provider_name=ModelProviderName.openrouter
    )
    b = SyntheticUserDriverConfig(
        model_name="y", model_provider_name=ModelProviderName.openrouter
    )
    a.visible_message_roles.append("user")  # type: ignore[arg-type]
    assert b.visible_message_roles == ["user", "assistant"]


def test_driver_config_accepts_explicit_visible_roles() -> None:
    cfg = SyntheticUserDriverConfig(
        model_name="x",
        model_provider_name=ModelProviderName.openrouter,
        visible_message_roles=["assistant"],
    )
    assert cfg.visible_message_roles == ["assistant"]


def test_driver_config_rejects_invalid_role_literal() -> None:
    with pytest.raises(ValidationError):
        SyntheticUserDriverConfig(
            model_name="x",
            model_provider_name=ModelProviderName.openrouter,
            visible_message_roles=["system"],  # type: ignore[list-item]
        )
