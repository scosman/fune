"""CodeTool data model — a user-authored Python function stored as a project artifact."""

import ast
import keyword
import re
from typing import Any

from pydantic import (
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    field_validator,
    model_serializer,
    model_validator,
)
from typing_extensions import Self

from kiln_ai.datamodel.basemodel import FilenameString, KilnParentedModel
from kiln_ai.datamodel.code_file_storage import (
    read_code_from_sibling_file,
    write_code_to_sibling_file,
)
from kiln_ai.datamodel.json_schema import validate_schema_dict
from kiln_ai.datamodel.tool_id import (
    ToolId,
    build_code_tool_id,
    validate_tool_allowlist,
)

_FUNCTION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# Fixed name of the sibling file that holds a code tool's Python source, stored
# beside its code_tool.kiln. Fixed so authored tests can `from tool import run`.
TOOL_CODE_FILENAME = "tool.py"


class CodeTool(KilnParentedModel):
    """A user-authored Python function that runs as a tool inside the agent harness.

    Functional content (code, schema, allowlist, etc.) is immutable post-create
    — the API enforces this; changing code means cloning into a new tool.
    """

    # Editable metadata
    name: FilenameString = Field(description="User-facing display name.")
    description: str | None = Field(
        default=None,
        description="User-facing notes shown in the UI. Not shown to models.",
    )
    is_archived: bool = Field(
        default=False,
        description="Archived tools are hidden from pickers but still resolve if referenced.",
    )

    # Functional content — immutable post-create (enforced at the API layer)
    tool_function_name: str = Field(
        description="The function name exposed to the model. Snake_case identifier."
    )
    tool_description: str = Field(
        min_length=1,
        description="Shown to agents as the tool description.",
    )
    parameters_schema: dict[str, Any] = Field(
        description="JSON Schema for the tool's parameters. Root must be type: object.",
    )
    code: str = Field(
        description="Python source, stored in a sibling tool.py file (in memory as a string). Validated for syntax and entry-point presence.",
    )
    timeout_seconds: int = Field(
        default=60,
        ge=1,
        description="Wall-clock timeout for one invocation, including nested tool calls.",
    )
    tool_allowlist: list[ToolId] = Field(
        default_factory=list,
        description="Explicit per-tool allowlist of tools this code tool may call.",
    )

    @model_validator(mode="before")
    @classmethod
    def _read_code_file(cls, data: Any, info: ValidationInfo) -> Any:
        """When loading from disk, inject `code` from the sibling tool.py.

        The source is stored in tool.py beside code_tool.kiln, not inline in the
        JSON. On load the base model puts the artifact folder in the validation
        context (`source_dir`); the shared helper reads the file here, before
        field validation, so the existing validate_code trio runs against the
        loaded string unchanged.
        """
        return read_code_from_sibling_file(
            data,
            info.context or {},
            filename=TOOL_CODE_FILENAME,
            kiln_filename="code_tool.kiln",
            model_label="CodeTool",
        )

    @model_serializer(mode="wrap")
    def _serialize(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, Any]:
        """On disk-save, write `code` to tool.py and omit it from the .kiln JSON.

        Delegates to the shared sibling-file helper, which uses the same save
        context attachments use (`save_attachments` + `dest_path`). Without that
        context — normal model_dump / API responses — `code` is left in the
        output and no file is written, so the API contract is unchanged.

        Trade-off: a custom model_serializer collapses the *serialization-mode*
        JSON schema to an untyped object (`model_json_schema(mode="serialization")`
        loses per-field typing). This is acceptable and consistent with the
        existing KilnAttachmentModel precedent, which uses the same pattern:
        - Validation-mode schema is unaffected, so request bodies stay fully typed.
        - No endpoint uses `response_model=CodeTool`; every code-tool endpoint
          returns a dedicated response model, and the generated web schema never
          references CodeTool's serialization schema.
        If a typed serialization schema is ever needed off this model, add a
        `__get_pydantic_json_schema__` override rather than removing this
        serializer. (CodeEvalProperties keeps exactly such an override because it
        IS a FastAPI response_model member.)
        """
        return write_code_to_sibling_file(
            handler(self),
            info.context or {},
            filename=TOOL_CODE_FILENAME,
            code=self.code,
        )

    @field_validator("tool_function_name")
    @classmethod
    def validate_function_name(cls, v: str) -> str:
        if not _FUNCTION_NAME_RE.fullmatch(v):
            raise ValueError(
                f"tool_function_name must match ^[a-z][a-z0-9_]{{0,63}}$, got: '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_parameters_schema(self, info: ValidationInfo) -> Self:
        validate_schema_dict(self.parameters_schema, require_object=True)

        # Parameters are passed to run() as keyword arguments keyed by these names,
        # and a Python reserved word can never be declared as a run() parameter, so
        # reject it at authoring. Loaded files are exempt so old projects still open.
        # Soft keywords (match/case/type/_) are valid identifiers and stay allowed.
        if not self.loaded_from_file(info):
            for param_name in self.parameters_schema.get("properties", {}):
                if keyword.iskeyword(param_name):
                    raise ValueError(
                        f"Parameter name '{param_name}' is a reserved Python keyword "
                        "and cannot be used as a run() parameter. Choose a different name."
                    )
        return self

    @model_validator(mode="after")
    def validate_code(self) -> Self:
        code_bytes = self.code.encode("utf-8")
        if len(code_bytes) > 64 * 1024:
            raise ValueError(
                f"Code is too large ({len(code_bytes)} bytes). Maximum size is 64KB."
            )

        try:
            compile(self.code, "<code_tool>", "exec")
        except SyntaxError as e:
            raise ValueError(f"Code has a syntax error: {e}") from e

        tree = ast.parse(self.code)
        has_run_fn = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
            for node in ast.iter_child_nodes(tree)
        )
        if not has_run_fn:
            raise ValueError(
                "Code must define a module-level 'run' function (def run(...) or async def run(...))."
            )

        return self

    @model_validator(mode="after")
    def validate_allowlist(self) -> Self:
        validate_tool_allowlist(
            self.tool_allowlist,
            caller="code tools",
            self_tool_id=build_code_tool_id(self.id) if self.id is not None else None,
        )
        return self
