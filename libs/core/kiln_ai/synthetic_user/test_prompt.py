"""Unit tests for the persona-playing system prompt template.

These are mostly structural assertions — we don't check exact wording (that's
allowed to drift as we tune the persona prompt) but we DO check the prompt
has every required section and teaches the early-stop sentinel.

The exceptions are the world-state rules and the ending block, which are
pinned verbatim. The two world-state lines exist because the synthetic user
was measured re-asserting records the assistant had already reported as
missing; a reworded version could quietly lose that fix. The ending block is
pinned beside them so a change to the world-state lines cannot edit the
ending rules themselves: a not-found reply changes the user's route, not
whether the user may end.
"""

from kiln_ai.synthetic_user.models import EARLY_STOP_SENTINEL, SyntheticUserInfo
from kiln_ai.synthetic_user.prompt import render_system_prompt


def _full_info() -> SyntheticUserInfo:
    return SyntheticUserInfo(
        persona="A 30-something professional, anxious about taxes.",
        goal="Find the 2016 RRSP contribution limit.",
        behavior_guidance="Press for specific numbers if the agent gives vague answers.",
    )


def test_includes_persona_text() -> None:
    info = _full_info()
    rendered = render_system_prompt(info)
    assert info.persona in rendered


def test_includes_goal_text() -> None:
    info = _full_info()
    rendered = render_system_prompt(info)
    assert info.goal in rendered


def test_includes_behavior_guidance_when_set() -> None:
    info = _full_info()
    rendered = render_system_prompt(info)
    assert info.behavior_guidance is not None
    assert info.behavior_guidance in rendered
    assert "How you behave" in rendered


def test_omits_behavior_guidance_section_when_none() -> None:
    info = SyntheticUserInfo(persona="P", goal="G")  # behavior_guidance = None
    rendered = render_system_prompt(info)
    assert "How you behave" not in rendered


def test_includes_conventions_block() -> None:
    rendered = render_system_prompt(_full_info())
    assert "Conversation style" in rendered


def test_includes_early_stop_sentinel() -> None:
    rendered = render_system_prompt(_full_info())
    assert EARLY_STOP_SENTINEL in rendered


def test_does_not_include_cancel_sentinel() -> None:
    # Ending is the only control signal the SU has; a second sentinel in the
    # prompt would be one the drive loop does not act on.
    rendered = render_system_prompt(_full_info())
    assert "<CANCEL>" not in rendered


def test_early_stop_sentinel_comes_after_behavior_guidance() -> None:
    # The ending rules come after the persona sections so they get the last
    # word, including the bullet that defers to behavior guidance.
    info = _full_info()
    rendered = render_system_prompt(info)
    assert info.behavior_guidance is not None
    assert rendered.index(EARLY_STOP_SENTINEL) > rendered.index("How you behave")


def test_section_order_opening_persona_goal_behavior_conventions() -> None:
    info = _full_info()
    rendered = render_system_prompt(info)
    assert info.behavior_guidance is not None

    # Use distinctive substrings from each section.
    opening_marker = "Stay in character"
    persona_section = "Your persona"
    goal_section = "Your goal in this conversation"
    behavior_section = "How you behave"
    conventions_marker = "Conversation style"

    indexes = [
        rendered.index(opening_marker),
        rendered.index(persona_section),
        rendered.index(goal_section),
        rendered.index(behavior_section),
        rendered.index(conventions_marker),
    ]
    assert indexes == sorted(indexes), f"Sections out of expected order: {indexes}"


def test_section_order_skips_behavior_when_none() -> None:
    info = SyntheticUserInfo(persona="P", goal="G")
    rendered = render_system_prompt(info)

    indexes = [
        rendered.index("Stay in character"),
        rendered.index("Your persona"),
        rendered.index("Your goal in this conversation"),
        rendered.index("Conversation style"),
    ]
    assert indexes == sorted(indexes)


# The synthetic user cannot see the assistant's systems, so a record it names
# without being told about it is invented. This line is what stops it.
_NO_INVENTED_STATE_LINE = (
    "- You are a person using this assistant. You have no way to see inside its "
    "systems, so anything you have not been told by the assistant is a guess: "
    "never invent identifiers, record numbers, names or dates. Beyond what your "
    "persona and goal already gave you, ask the assistant what exists, or ask it "
    "to create what you need, then work with what it reports."
)

# A correct not-found reply is not a mistake to push through; the user takes
# another route instead of repeating the reference.
_ACCEPT_NOT_FOUND_LINE = (
    "- When the assistant says something cannot be found or does not exist, that "
    "is the truth of its system. Do not repeat the reference or insist, even if "
    "your guidance tells you to press: instead give a description, ask it to list "
    "what is there, or ask it to create the record, and continue toward your goal "
    "from what it reports."
)

# The ending block, pinned verbatim. The world-state rules change how the
# user continues, not whether it may end; the one line here that mentions
# them only says guidance cannot override them.
_ENDING_BLOCK = f"""## Ending the conversation
- Before anything else, count the assistant's messages in the conversation so far. If there is only one, you cannot end yet: send a real message, even if the goal already seems met. Confirm a detail, ask about a next step, or check something you are unsure of.
- Once the assistant has replied at least twice, and your goal has been fully and specifically met, and you have nothing left to ask, end the conversation: your entire reply is exactly {EARLY_STOP_SENTINEL}. It replaces your message. Do not write a thank-you or a closing line first; {EARLY_STOP_SENTINEL} is the whole reply.
- Never end because the assistant refused, deflected, gave a partial or general answer, asked you a question, or made a mistake. In those cases keep going.
- If your persona or behavior guidance asks for more than the goal, such as pressing further or covering more ground, follow that instead of ending. Guidance never overrides the two rules above about what exists."""


def test_tells_user_it_cannot_know_what_exists() -> None:
    rendered = render_system_prompt(_full_info())
    assert _NO_INVENTED_STATE_LINE in rendered
    # It is a conversation-style rule, not an ending rule.
    assert rendered.index(_NO_INVENTED_STATE_LINE) < rendered.index(
        "Ending the conversation"
    )


def test_tells_user_to_accept_a_not_found_reply() -> None:
    rendered = render_system_prompt(_full_info())
    assert _ACCEPT_NOT_FOUND_LINE in rendered
    assert rendered.index(_ACCEPT_NOT_FOUND_LINE) < rendered.index(
        "Ending the conversation"
    )


def test_ending_block_is_unchanged_by_the_world_state_rules() -> None:
    rendered = render_system_prompt(_full_info())
    # The ending block is the tail of the prompt: nothing about not-found
    # replies was appended to or inserted into it.
    assert rendered.endswith(_ENDING_BLOCK)
