from enum import Enum
from typing import Annotated

from pydantic import AfterValidator

from kiln_ai.datamodel.basemodel import ID_TYPE

ToolId = Annotated[
    str,
    AfterValidator(lambda v: _check_tool_id(v)),
]
"""
A pydantic type that validates strings containing a valid tool ID.

Tool IDs can be one of:
- A kiln built-in tool name: kiln_tool::add_numbers
- A remote MCP tool: mcp::remote::<server_id>::<tool_name>
- A local MCP tool: mcp::local::<server_id>::<tool_name>
- A Kiln task tool: kiln_task::<server_id>
- An SDK / adapter-injected unmanaged tool: kiln_unmanaged::<id> (single slug, not from the registry)
- More coming soon like kiln_project_tool::rag::RAG_CONFIG_ID
"""


class KilnBuiltInToolId(str, Enum):
    """Built-in tool IDs for Kiln's demo tools."""

    ADD_NUMBERS = "kiln_tool::add_numbers"
    SUBTRACT_NUMBERS = "kiln_tool::subtract_numbers"
    MULTIPLY_NUMBERS = "kiln_tool::multiply_numbers"
    DIVIDE_NUMBERS = "kiln_tool::divide_numbers"
    CALL_KILN_API = "kiln_tool::call_kiln_api"
    LLM = "kiln_tool::llm"
    LLM_JUDGE = "kiln_tool::llm_judge"


MCP_REMOTE_TOOL_ID_PREFIX = "mcp::remote::"
RAG_TOOL_ID_PREFIX = "kiln_tool::rag::"
MCP_LOCAL_TOOL_ID_PREFIX = "mcp::local::"
KILN_TASK_TOOL_ID_PREFIX = "kiln_task::"
SKILL_TOOL_ID_PREFIX = "kiln_tool::skill::"
KILN_UNMANAGED_TOOL_ID_PREFIX = "kiln_unmanaged::"
CODE_TOOL_ID_PREFIX = "kiln_tool::code::"


def kiln_unmanaged_tool_slug_from_id(id: str) -> str:
    """
    Parse ``kiln_unmanaged::<slug>`` and return ``slug``.

    Use a unique slug per tool (e.g. ``model_info``, ``myapp_get_user``). Not used by ``tool_from_id``.
    """
    parts = id.split("::")
    if len(parts) != 2 or parts[0] != "kiln_unmanaged":
        raise ValueError(
            f"Invalid kiln_unmanaged tool ID: {id}. Expected format: 'kiln_unmanaged::<slug>'."
        )
    slug = parts[1]
    if not slug.strip():
        raise ValueError(
            f"Invalid kiln_unmanaged tool ID: {id}. Expected format: 'kiln_unmanaged::<slug>'."
        )
    return slug.strip()


def build_kiln_unmanaged_tool_id(unique_id: str) -> str:
    """Construct a tool ID for :class:`kiln_ai.tools.base_tool.UnmanagedKilnTool` and similar."""
    if not unique_id.strip():
        raise ValueError("unique_id must be non-empty")
    if "::" in unique_id:
        raise ValueError("unique_id must not contain '::'")
    return f"{KILN_UNMANAGED_TOOL_ID_PREFIX}{unique_id}"


def _check_tool_id(id: str) -> str:
    """
    Check that the tool ID is valid.
    """
    if not id or not isinstance(id, str):
        raise ValueError(f"Invalid tool ID: {id}")

    # Build in tools
    if id in KilnBuiltInToolId.__members__.values():
        return id

    # MCP remote tools must have format: mcp::remote::<server_id>::<tool_name>
    if id.startswith(MCP_REMOTE_TOOL_ID_PREFIX):
        server_id, tool_name = mcp_server_and_tool_name_from_id(id)
        if not server_id or not tool_name:
            raise ValueError(
                f"Invalid remote MCP tool ID: {id}. Expected format: 'mcp::remote::<server_id>::<tool_name>'."
            )
        return id

    # MCP local tools must have format: mcp::local::<server_id>::<tool_name>
    if id.startswith(MCP_LOCAL_TOOL_ID_PREFIX):
        server_id, tool_name = mcp_server_and_tool_name_from_id(id)
        if not server_id or not tool_name:
            raise ValueError(
                f"Invalid local MCP tool ID: {id}. Expected format: 'mcp::local::<server_id>::<tool_name>'."
            )
        return id

    # RAG tools must have format: kiln_tool::rag::<rag_config_id>
    if id.startswith(RAG_TOOL_ID_PREFIX):
        rag_config_id = rag_config_id_from_id(id)
        if not rag_config_id:
            raise ValueError(
                f"Invalid RAG tool ID: {id}. Expected format: 'kiln_tool::rag::<rag_config_id>'."
            )
        return id

    # Kiln task tools must have format: kiln_task::<server_id>
    if id.startswith(KILN_TASK_TOOL_ID_PREFIX):
        server_id = kiln_task_server_id_from_tool_id(id)
        if not server_id:
            raise ValueError(
                f"Invalid Kiln task tool ID: {id}. Expected format: 'kiln_task::<server_id>'."
            )
        return id

    # Skill tools must have format: kiln_tool::skill::<skill_id>
    if id.startswith(SKILL_TOOL_ID_PREFIX):
        skill_id = skill_id_from_tool_id(id)
        if not skill_id:
            raise ValueError(
                f"Invalid skill tool ID: {id}. Expected format: 'kiln_tool::skill::<skill_id>'."
            )
        return id

    # Code tools must have format: kiln_tool::code::<code_tool_id>
    if id.startswith(CODE_TOOL_ID_PREFIX):
        # Raises ValueError on malformed IDs; the extracted ID isn't needed here.
        code_tool_id_from_tool_id(id)
        return id

    # SDK / AdapterConfig.unmanaged_tools — not resolved by tool_from_id
    if id.startswith(KILN_UNMANAGED_TOOL_ID_PREFIX):
        kiln_unmanaged_tool_slug_from_id(id)
        return id

    raise ValueError(f"Invalid tool ID: {id}")


def mcp_server_and_tool_name_from_id(id: str) -> tuple[str, str]:
    """
    Get the tool server ID and tool name from the ID.
    """
    parts = id.split("::")
    if len(parts) != 4:
        # Determine if it's remote or local for the error message
        if id.startswith(MCP_REMOTE_TOOL_ID_PREFIX):
            raise ValueError(
                f"Invalid remote MCP tool ID: {id}. Expected format: 'mcp::remote::<server_id>::<tool_name>'."
            )
        elif id.startswith(MCP_LOCAL_TOOL_ID_PREFIX):
            raise ValueError(
                f"Invalid local MCP tool ID: {id}. Expected format: 'mcp::local::<server_id>::<tool_name>'."
            )
        else:
            raise ValueError(
                f"Invalid MCP tool ID: {id}. Expected format: 'mcp::(remote|local)::<server_id>::<tool_name>'."
            )
    return parts[2], parts[3]  # server_id, tool_name


def rag_config_id_from_id(id: str) -> str:
    """
    Get the RAG config ID from the ID.
    """
    parts = id.split("::")
    if not id.startswith(RAG_TOOL_ID_PREFIX) or len(parts) != 3:
        raise ValueError(
            f"Invalid RAG tool ID: {id}. Expected format: 'kiln_tool::rag::<rag_config_id>'."
        )
    return parts[2]


def build_rag_tool_id(rag_config_id: ID_TYPE) -> str:
    """Construct the tool ID for a RAG configuration."""

    return f"{RAG_TOOL_ID_PREFIX}{rag_config_id}"


def build_kiln_task_tool_id(server_id: ID_TYPE) -> str:
    """Construct the tool ID for a Kiln task server."""
    return f"{KILN_TASK_TOOL_ID_PREFIX}{server_id}"


def skill_id_from_tool_id(id: str) -> str:
    """Get the skill ID from a skill tool ID."""
    parts = id.split("::")
    if not id.startswith(SKILL_TOOL_ID_PREFIX) or len(parts) != 3:
        raise ValueError(
            f"Invalid skill tool ID: {id}. Expected format: 'kiln_tool::skill::<skill_id>'."
        )
    return parts[2]


def build_skill_tool_id(skill_id: str) -> str:
    """Construct the tool ID for a skill."""
    return f"{SKILL_TOOL_ID_PREFIX}{skill_id}"


def build_code_tool_id(code_tool_id: ID_TYPE) -> str:
    """Construct the tool ID for a code tool."""
    return f"{CODE_TOOL_ID_PREFIX}{code_tool_id}"


def code_tool_id_from_tool_id(tool_id: str) -> str:
    """Extract the code tool ID from a code tool tool ID."""
    parts = tool_id.split("::")
    if not tool_id.startswith(CODE_TOOL_ID_PREFIX) or len(parts) != 3 or not parts[2]:
        raise ValueError(
            f"Invalid code tool ID: {tool_id}. Expected format: 'kiln_tool::code::<code_tool_id>'."
        )
    return parts[2]


def kiln_task_server_id_from_tool_id(tool_id: str) -> str:
    """
    Get the server ID from the tool ID.
    """
    if not tool_id.startswith(KILN_TASK_TOOL_ID_PREFIX):
        raise ValueError(
            f"Invalid Kiln task tool ID format: {tool_id}. Expected format: 'kiln_task::<server_id>'."
        )

    # Remove prefix and split on ::
    remaining = tool_id[len(KILN_TASK_TOOL_ID_PREFIX) :]
    if not remaining:
        raise ValueError(
            f"Invalid Kiln task tool ID format: {tool_id}. Expected format: 'kiln_task::<server_id>'."
        )
    parts = remaining.split("::")

    if len(parts) != 1 or not parts[0].strip():
        raise ValueError(
            f"Invalid Kiln task tool ID format: {tool_id}. Expected format: 'kiln_task::<server_id>'."
        )

    return parts[0]  # server_id


def validate_tool_allowlist(
    tool_allowlist: list[ToolId],
    *,
    caller: str,
    self_tool_id: ToolId | None = None,
) -> None:
    """Validate a sandboxed-code tool allowlist, raising ValueError on the first problem.

    Shared by every model that carries one (code tools, code evals) so the rules
    cannot drift apart: the sandbox bridge resolves them all through the same path,
    so what one accepts the other must too.

    ``caller`` names the owner in error messages ("code tools", "code evals").
    ``self_tool_id`` is the owner's own tool id, when it has one, to reject
    self-reference.
    """
    seen: set[str] = set()
    for tool_id in tool_allowlist:
        if tool_id.startswith(SKILL_TOOL_ID_PREFIX):
            raise ValueError(
                f"Skill tool IDs cannot be used in tool_allowlist: {tool_id}. "
                f"Skills are adapter-resolved and not callable from {caller}."
            )
        if tool_id.startswith(KILN_UNMANAGED_TOOL_ID_PREFIX):
            raise ValueError(
                f"Unmanaged tool IDs cannot be used in tool_allowlist: {tool_id}. "
                "Unmanaged tools are SDK-injected and not resolvable by the registry."
            )
        if tool_id in seen:
            raise ValueError(f"Duplicate tool ID in tool_allowlist: {tool_id}")
        seen.add(tool_id)

    # Only code tools are themselves tools, so only they can self-reference; the
    # wording stays concrete rather than being generated from `caller`.
    if self_tool_id is not None and self_tool_id in seen:
        raise ValueError("A code tool cannot reference itself in tool_allowlist.")
