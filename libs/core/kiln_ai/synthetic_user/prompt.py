"""Persona-playing system prompt rendered per request for the synthetic user.

The prompt is built once per case (when the driver is constructed) and reused
across all turns. It teaches the SU `EARLY_STOP_SENTINEL` from `models` and
the rule for using it: send the sentinel alone only when the goal has genuinely
been met, never in reply to the assistant's first message, and never because
the assistant refused, deflected, answered partially or got something wrong.
Persona or behavior guidance that asks for more than the goal wins over
ending. The drive loop treats `turns` as a ceiling and stops on the sentinel,
so a case that finishes early costs the turns it actually needed.

The prompt also tells the SU that it cannot know what exists inside the
assistant's systems: it must not invent identifiers or records, and when the
assistant reports something as not found it accepts that and takes another
route to the goal instead of re-asserting the reference. Without this, the
instruction to vary its approach "when the assistant is vague, evasive, or
wrong" reads a correct not-found reply as a mistake to push through, and the
SU keeps a nonexistent record alive across the whole conversation. Behavior
guidance that tells the SU to press cannot override these two rules, because
the cases that name a record most often are the ones whose guidance scripts
the insistence. The ending rules themselves are unchanged: the rules change
how the user continues, not when it may end.

The first-message rule is phrased as counting the assistant's messages rather
than "don't end on your first reply" because the SU sees the conversation with
the roles swapped: its own opening message is already in the transcript, so
"your first reply" reads ambiguously — the model cannot tell whether the
opener counts. The assistant's message count has no such ambiguity. It still
reads correctly under the swap: the opening line names the counterpart as
the assistant, and at every point the SU is asked to reply the two sides have
sent the same number of messages, so either reading gives the same count.
"""

from kiln_ai.synthetic_user.models import EARLY_STOP_SENTINEL, SyntheticUserInfo

_OPENING = (
    "You are playing the role of a user interacting with an AI assistant. "
    "Stay in character — respond as the user would, not as the assistant. "
    "Your entire output is the user's next message, verbatim and nothing "
    "else: no narration, no meta-commentary, no quotes, no labels like "
    '"User:". Just the message text the user would type.'
)

_CONVENTIONS = f"""## Conversation style
- Reply naturally and conversationally, as a real user would.
- React to what the assistant actually said in their last message.
- Keep replies short (one or two sentences) unless the situation calls for more detail.
- Do not narrate your own behavior ("As a user, I will now…"). Speak as the user.
- Stay engaged and keep pursuing your goal with follow-ups, varying your approach when the assistant is vague, evasive, or wrong.
- You are a person using this assistant. You have no way to see inside its systems, so anything you have not been told by the assistant is a guess: never invent identifiers, record numbers, names or dates. Beyond what your persona and goal already gave you, ask the assistant what exists, or ask it to create what you need, then work with what it reports.
- When the assistant says something cannot be found or does not exist, that is the truth of its system. Do not repeat the reference or insist, even if your guidance tells you to press: instead give a description, ask it to list what is there, or ask it to create the record, and continue toward your goal from what it reports.

## Ending the conversation
- Before anything else, count the assistant's messages in the conversation so far. If there is only one, you cannot end yet: send a real message, even if the goal already seems met. Confirm a detail, ask about a next step, or check something you are unsure of.
- Once the assistant has replied at least twice, and your goal has been fully and specifically met, and you have nothing left to ask, end the conversation: your entire reply is exactly {EARLY_STOP_SENTINEL}. It replaces your message. Do not write a thank-you or a closing line first; {EARLY_STOP_SENTINEL} is the whole reply.
- Never end because the assistant refused, deflected, gave a partial or general answer, asked you a question, or made a mistake. In those cases keep going.
- If your persona or behavior guidance asks for more than the goal, such as pressing further or covering more ground, follow that instead of ending. Guidance never overrides the two rules above about what exists."""


def render_system_prompt(info: SyntheticUserInfo) -> str:
    """Render the persona-playing system prompt for one eval case."""
    sections = [
        _OPENING,
        f"## Your persona\n{info.persona}",
        f"## Your goal in this conversation\n{info.goal}",
    ]
    if info.behavior_guidance:
        sections.append(f"## How you behave\n{info.behavior_guidance}")
    sections.append(_CONVENTIONS)
    return "\n\n".join(sections)
