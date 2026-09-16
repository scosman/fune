from kiln_ai.datamodel.project import Project
from kiln_ai.datamodel.rag import RagConfig
from kiln_ai.datamodel.task import Task
from kiln_ai.datamodel.tool_id import (
    CODE_TOOL_ID_PREFIX,
    KILN_TASK_TOOL_ID_PREFIX,
    MCP_LOCAL_TOOL_ID_PREFIX,
    MCP_REMOTE_TOOL_ID_PREFIX,
    RAG_TOOL_ID_PREFIX,
    SKILL_TOOL_ID_PREFIX,
    KilnBuiltInToolId,
    code_tool_id_from_tool_id,
    kiln_task_server_id_from_tool_id,
    mcp_server_and_tool_name_from_id,
    rag_config_id_from_id,
)
from kiln_ai.tools.base_tool import KilnToolInterface, ToolCallDefinition
from kiln_ai.tools.built_in_tools.kiln_api_call_tool import KilnApiCallTool
from kiln_ai.tools.built_in_tools.math_tools import (
    AddTool,
    DivideTool,
    MultiplyTool,
    SubtractTool,
)
from kiln_ai.tools.kiln_task_tool import KilnTaskTool
from kiln_ai.tools.mcp_server_tool import MCPServerTool
from kiln_ai.utils.config import Config
from kiln_ai.utils.exhaustive_error import raise_exhaustive_enum_error


def tool_from_id_and_project(
    tool_id: str,
    project: Project | None = None,
    task: Task | None = None,
) -> KilnToolInterface:
    """Resolve a tool from its ID, given a project directly.

    This is the core resolution function. ``tool_from_id`` is a thin
    wrapper that derives the project from the task.
    """
    # Check built-in tools
    if tool_id in [member.value for member in KilnBuiltInToolId]:
        typed_tool_id = KilnBuiltInToolId(tool_id)
        match typed_tool_id:
            case KilnBuiltInToolId.ADD_NUMBERS:
                return AddTool()
            case KilnBuiltInToolId.SUBTRACT_NUMBERS:
                return SubtractTool()
            case KilnBuiltInToolId.MULTIPLY_NUMBERS:
                return MultiplyTool()
            case KilnBuiltInToolId.DIVIDE_NUMBERS:
                return DivideTool()
            case KilnBuiltInToolId.CALL_KILN_API:
                api_base_url = Config.shared().kiln_local_api_base_url()
                if not api_base_url:
                    raise ValueError(
                        "kiln_local_api_base_url is not configured. The server must set this before starting."
                    )
                return KilnApiCallTool(api_base_url=api_base_url)
            case KilnBuiltInToolId.LLM:
                # Lazy import to avoid the tools -> adapter -> tool_registry cycle.
                from kiln_ai.tools.built_in_tools.llm_tools import LlmTool

                return LlmTool()
            case KilnBuiltInToolId.LLM_JUDGE:
                # Lazy import to avoid the tools -> adapter -> tool_registry cycle.
                from kiln_ai.tools.built_in_tools.llm_tools import LlmJudgeTool

                return LlmJudgeTool()
            case _:
                raise_exhaustive_enum_error(typed_tool_id)

    # Check if this looks like an MCP or Kiln Task tool ID that requires a project
    is_mcp_tool = is_mcp_tool_id(tool_id)
    is_kiln_task_tool = tool_id.startswith(KILN_TASK_TOOL_ID_PREFIX)

    if is_mcp_tool or is_kiln_task_tool:
        if project is None or project.id is None:
            raise ValueError(
                f"Unable to resolve tool from id: {tool_id}. Requires a parent project/task."
            )

        # Check MCP Server Tools
        if is_mcp_tool:
            # Get the tool server ID and tool name from the ID
            tool_server_id, tool_name = mcp_server_and_tool_name_from_id(
                tool_id
            )  # Fixed function name

            server = next(
                (
                    server
                    for server in project.external_tool_servers()
                    if server.id == tool_server_id
                ),
                None,
            )
            if server is None:
                raise ValueError(
                    f"External tool server not found: {tool_server_id} in project ID {project.id}"
                )

            return MCPServerTool(server, tool_name)

        # Check Kiln Task Tools
        if is_kiln_task_tool:
            server_id = kiln_task_server_id_from_tool_id(tool_id)

            server = next(
                (
                    server
                    for server in project.external_tool_servers()
                    if server.id == server_id
                ),
                None,
            )
            if server is None:
                raise ValueError(
                    f"Kiln Task External tool server not found: {server_id} in project ID {project.id}"
                )

            return KilnTaskTool(project.id, tool_id, server)

    elif tool_id.startswith(RAG_TOOL_ID_PREFIX):
        if project is None:
            raise ValueError(
                f"Unable to resolve tool from id: {tool_id}. Requires a parent project/task."
            )

        rag_config_id = rag_config_id_from_id(tool_id)
        rag_config = RagConfig.from_id_and_parent_path(rag_config_id, project.path)
        if rag_config is None:
            raise ValueError(
                f"RAG config not found: {rag_config_id} in project {project.id} for tool {tool_id}"
            )

        # Lazy import to avoid circular dependency
        from kiln_ai.tools.rag_tools import RagTool

        return RagTool(tool_id, rag_config)

    elif tool_id.startswith(CODE_TOOL_ID_PREFIX):
        if project is None:
            raise ValueError(
                f"Unable to resolve tool from id: {tool_id}. Requires a parent project/task."
            )

        ct_id = code_tool_id_from_tool_id(tool_id)

        # Lazy import to avoid circular dependency
        from kiln_ai.datamodel.code_tool import CodeTool

        code_tool = CodeTool.from_id_and_parent_path(ct_id, project.path)
        if code_tool is None:
            raise ValueError(
                f"Code tool not found: {ct_id} in project {project.id} for tool {tool_id}"
            )

        # Lazy import — PythonCodeTool pulls in the multiprocessing execution
        # runtime, which most registry lookups never need.
        from kiln_ai.tools.code_tool import PythonCodeTool

        return PythonCodeTool(code_tool, project, task)

    elif tool_id.startswith(SKILL_TOOL_ID_PREFIX):
        raise ValueError(
            f"Skill tool IDs are resolved by the adapter, not tool_from_id: {tool_id}"
        )

    raise ValueError(f"Tool ID {tool_id} not found in tool registry")


def tool_from_id(tool_id: str, task: Task | None = None) -> KilnToolInterface:
    """Get a tool from its ID.

    Thin wrapper around ``tool_from_id_and_project`` that derives
    the project from *task*.
    """
    project = task.parent_project() if task is not None else None
    return tool_from_id_and_project(tool_id, project=project, task=task)


async def validate_unique_allowlist_tool_names(
    tool_allowlist: list[str],
    project: Project,
    task: Task | None = None,
) -> None:
    """Reject a tool allowlist whose tools would collide on their dispatch name.

    Tool names may repeat across a project, but sandboxed code calls
    allowlisted tools by name (see sandbox_bridge.name_map), so an allowlist
    carrying two tools with the same function name would make those calls
    ambiguous. Raises ValueError naming the colliding function names.

    A collision only among MCP tools is tolerated: their names come from the
    servers and cannot be renamed in Kiln, so rejecting would leave the user
    no fix short of dropping a server's tools entirely. The sandbox bridge's
    call-time ambiguity error remains the backstop — for those, and because
    tools can be renamed after the allowlist is saved.
    """
    carriers: dict[str, list[str]] = {}
    for tool_id in tool_allowlist:
        name = await tool_from_id_and_project(
            tool_id, project=project, task=task
        ).name()
        carriers.setdefault(name, []).append(tool_id)
    duplicates = sorted(
        name
        for name, tool_ids in carriers.items()
        if len(tool_ids) > 1 and not all(is_mcp_tool_id(tid) for tid in tool_ids)
    )
    if duplicates:
        raise ValueError(
            f"Multiple allowlisted tools share the same function name: "
            f"{', '.join(duplicates)}. Tools are called by name from sandboxed "
            "code, so each tool in the allowlist must have a unique name."
        )


async def tool_definitions_from_ids(
    tool_ids: list[str], task: Task | None = None
) -> list[ToolCallDefinition]:
    """
    Get OpenAI-compatible tool definitions from a list of tool IDs.
    """
    tool_definitions = []
    for tool_id in tool_ids:
        try:
            tool = tool_from_id(tool_id, task)
            tool_def = await tool.toolcall_definition()
            tool_definitions.append(tool_def)
        except Exception as e:
            raise ValueError(
                f"Failed to get tool definition for tool ID: {tool_id}. Original error: {e}"
            )
    return tool_definitions


def is_mcp_tool_id(tool_id: str) -> bool:
    return tool_id.startswith((MCP_REMOTE_TOOL_ID_PREFIX, MCP_LOCAL_TOOL_ID_PREFIX))
