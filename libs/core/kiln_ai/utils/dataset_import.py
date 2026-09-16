import csv
import json
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Literal, Protocol

from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import BaseModel, Field, ValidationError

from kiln_ai.datamodel import DataSource, DataSourceType, Task, TaskOutput, TaskRun
from kiln_ai.datamodel.datamodel_enums import TurnMode
from kiln_ai.utils.open_ai_types import (
    ChatCompletionAssistantMessageParamWrapper,
    ChatCompletionMessageParam,
)

logger = logging.getLogger(__name__)

# Python's csv module defaults to 131,072 bytes per field, which legitimate
# imports can exceed when a row contains a long prompt, response, or chat
# transcript (surfacing as ``_csv.Error: field larger than field limit``).
# 100 MiB is far larger than any realistic prompt/response (even a 2M-token
# context window serializes to <10 MiB of text) while still bounding memory
# from a maliciously crafted file, and fits in a 32-bit signed C long so
# ``csv.field_size_limit`` accepts it on every supported platform
# (including 64-bit Windows).
_CSV_FIELD_SIZE_LIMIT_BYTES = 100 * 1024 * 1024

csv.field_size_limit(_CSV_FIELD_SIZE_LIMIT_BYTES)


class DatasetImportFormat(str, Enum):
    """
    The format of the dataset to import.
    """

    CSV = "csv"


@dataclass
class ImportConfig:
    """Configuration for importing a dataset"""

    dataset_type: DatasetImportFormat
    dataset_path: str
    dataset_name: str
    """
    A set of splits to assign to the import (as dataset tags).
    The keys are the names of the splits (tag name), and the values are the proportions of the dataset to include in each split (should sum to 1).
    """
    tag_splits: Dict[str, float] | None = None

    def validate_tag_splits(self) -> None:
        if self.tag_splits:
            EPSILON = 0.001  # Allow for small floating point errors
            if abs(sum(self.tag_splits.values()) - 1) > EPSILON:
                raise ValueError(
                    "Splits must sum to 1. The following splits do not: "
                    + ", ".join(f"{k}: {v}" for k, v in self.tag_splits.items())
                )


@dataclass
class ImportResult:
    """Outcome of a dataset import.

    `imported_run_count` counts every TaskRun saved. For multiturn imports, a single
    conversation produces N runs (one per assistant turn); `imported_conversation_count`
    captures the number of conversations and is `None` for single-turn imports.
    """

    imported_run_count: int
    imported_conversation_count: int | None


class Importer(Protocol):
    """Protocol for dataset importers"""

    def __call__(
        self,
        task: Task,
        config: ImportConfig,
    ) -> ImportResult: ...


class CSVRowSchema(BaseModel):
    """Schema for validating rows in a CSV file."""

    input: str = Field(description="The input to the model")
    output: str = Field(description="The output of the model")
    reasoning: str | None = Field(
        description="The reasoning of the model (optional)",
        default=None,
    )
    chain_of_thought: str | None = Field(
        description="The chain of thought of the model (optional)",
        default=None,
    )
    tags: list[str] = Field(
        default_factory=list,
        description="The tags of the run (optional)",
    )


class CSVMultiturnRowSchema(BaseModel):
    """Schema for validating rows of a multiturn CSV file."""

    trace: str = Field(description="JSON-encoded list of OpenAI chat messages")
    tags: list[str] = Field(
        default_factory=list,
        description="The tags applied to every run in the conversation (optional)",
    )


ALLOWED_MULTITURN_ROLES = {"user", "assistant"}


@dataclass
class ValidatedMessage:
    """A trace message that has passed structural validation."""

    role: Literal["user", "assistant"]
    content: str
    reasoning_content: str | None


def generate_import_tags(session_id: str) -> list[str]:
    return [
        "imported",
        f"imported_{session_id}",
    ]


class KilnInvalidImportFormat(Exception):
    """Raised when the import format is invalid"""

    def __init__(self, message: str, row_number: int | None = None):
        self.row_number = row_number
        if row_number is not None:
            message = f"Error in row {row_number}: {message}"
        super().__init__(message)


def format_validation_error(e: ValidationError) -> str:
    """Convert a Pydantic validation error into a human-readable message."""
    error_messages = []
    for error in e.errors():
        location = " -> ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        error_messages.append(f"- {location}: {message}")

    return "Validation failed:\n" + "\n".join(error_messages)


def deserialize_tags(tags_serialized: str | None) -> list[str]:
    """Deserialize tags from a comma-separated string to a list of strings."""
    if tags_serialized:
        return [tag.strip() for tag in tags_serialized.split(",") if tag.strip()]
    return []


def without_none_values(d: dict) -> dict:
    """Return a copy of the dictionary with all None values removed."""
    return {k: v for k, v in d.items() if v is not None}


def add_tag_splits(runs: list[TaskRun], tag_splits: Dict[str, float] | None) -> None:
    """Assign split tags to runs according to configured proportions.

    Args:
        runs: List of TaskRun objects to assign tags to
        tag_splits: Dictionary mapping tag names to their desired proportions

    The assignment is random but ensures the proportions match the configured splits
    as closely as possible given the number of runs.
    """
    if not tag_splits:
        return

    # Calculate exact number of runs for each split
    total_runs = len(runs)
    split_counts = {
        tag: int(proportion * total_runs) for tag, proportion in tag_splits.items()
    }

    # Handle rounding errors by adjusting the largest split
    remaining = total_runs - sum(split_counts.values())
    if remaining != 0:
        largest_split = max(split_counts.items(), key=lambda x: x[1])
        split_counts[largest_split[0]] += remaining

    # Create a list of tags with the correct counts
    tags_to_assign = []
    for tag, count in split_counts.items():
        tags_to_assign.extend([tag] * count)

    # Shuffle the tags to randomize assignment
    random.shuffle(tags_to_assign)

    # Assign tags to runs
    for run, tag in zip(runs, tags_to_assign):
        run.tags.append(tag)


def create_task_run_from_csv_row(
    task: Task,
    row: dict[str, str],
    dataset_name: str,
    session_id: str,
) -> TaskRun:
    """Validate and create a TaskRun from a CSV row, without saving to file"""

    # first we validate the row from the CSV file
    validated_row = CSVRowSchema.model_validate(
        {
            **row,
            "tags": deserialize_tags(row.get("tags")),
        }
    )

    tags = generate_import_tags(session_id)
    if validated_row.tags:
        tags.extend(validated_row.tags)

    # note that we don't persist the run yet, we just create and validate it
    # this instantiation may raise pydantic validation errors
    run = TaskRun(
        parent=task,
        input=validated_row.input,
        input_source=DataSource(
            type=DataSourceType.file_import,
            properties={
                "file_name": dataset_name,
            },
        ),
        output=TaskOutput(
            output=validated_row.output,
            source=DataSource(
                type=DataSourceType.file_import,
                properties={
                    "file_name": dataset_name,
                },
            ),
        ),
        intermediate_outputs=without_none_values(
            {
                "reasoning": validated_row.reasoning,
                "chain_of_thought": validated_row.chain_of_thought,
            }
        )
        or None,
        tags=tags,
    )

    return run


def import_csv(
    task: Task,
    config: ImportConfig,
) -> ImportResult:
    """Import a CSV dataset, dispatched on `task.turn_mode`.

    All rows are validated before any are persisted to files, and a save
    failure rolls back the runs already saved (best effort), so a failed
    import leaves no partial data behind."""

    if task.turn_mode == TurnMode.multiturn:
        return _import_csv_multiturn(task, config)
    return _import_csv_single_turn(task, config)


def _save_runs_with_rollback(runs: list[TaskRun]) -> None:
    """Save all runs; if any save fails, delete the runs already saved (best
    effort) and re-raise the original error.

    Keeps a failed import all-or-nothing. This matters most for multiturn
    chains, where a truncated chain's last-saved run would pose as a complete
    conversation's leaf.
    """
    saved: list[TaskRun] = []
    try:
        for run in runs:
            run.save_to_file()
            saved.append(run)
    except Exception:
        for prior in saved:
            try:
                prior.delete()
            except Exception:
                logger.warning(
                    f"Failed to clean up partially imported run {prior.id}",
                    exc_info=True,
                )
        raise


def _import_csv_single_turn(
    task: Task,
    config: ImportConfig,
) -> ImportResult:
    """Import a single-turn CSV: one row per TaskRun."""

    session_id = str(int(time.time()))
    dataset_path = config.dataset_path
    dataset_name = config.dataset_name
    tag_splits = config.tag_splits

    required_headers = {"input", "output"}  # minimum required headers
    optional_headers = {"reasoning", "tags", "chain_of_thought"}  # optional headers

    rows: list[TaskRun] = []
    with open(dataset_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        # Check if we have headers
        if not reader.fieldnames:
            raise KilnInvalidImportFormat(
                "CSV file appears to be empty or missing headers"
            )

        # Check for required headers
        actual_headers = set(reader.fieldnames)
        # Detect a multiturn-shaped CSV uploaded to a single-turn task and
        # tell the user how to fix it. Fires unconditionally on `trace` so
        # multiturn data isn't silently dropped when single-turn columns are
        # also present.
        if "trace" in actual_headers:
            raise KilnInvalidImportFormat(
                "Task is single-turn; expected columns: input, output "
                "(and optional reasoning, chain_of_thought, tags). Got: "
                f"{', '.join(sorted(actual_headers))}."
            )
        missing_headers = required_headers - actual_headers
        if missing_headers:
            raise KilnInvalidImportFormat(
                f"Missing required headers: {', '.join(missing_headers)}. "
                f"Required headers are: {', '.join(required_headers)}"
            )

        # Warn about unknown headers (not required or optional)
        unknown_headers = actual_headers - (required_headers | optional_headers)
        if unknown_headers:
            logger.warning(
                f"Unknown headers in CSV file will be ignored: {', '.join(unknown_headers)}"
            )

        # enumeration starts at 2 because row 1 is headers
        for row_number, row in enumerate(reader, start=2):
            try:
                run = create_task_run_from_csv_row(
                    task=task,
                    row=row,
                    dataset_name=dataset_name,
                    session_id=session_id,
                )
            except ValidationError as e:
                logger.warning(f"Invalid row {row_number}: {row}", exc_info=True)
                human_readable = format_validation_error(e)
                raise KilnInvalidImportFormat(
                    human_readable,
                    row_number=row_number,
                ) from e
            rows.append(run)

    add_tag_splits(rows, tag_splits)

    # now that we know all rows are valid, we can save them
    _save_runs_with_rollback(rows)

    return ImportResult(imported_run_count=len(rows), imported_conversation_count=None)


def _validate_csv_tags(tags: list[str], row_number: int) -> None:
    """Validate CSV-supplied tags with row-tagged, CSV-friendly error messages.

    Mirrors `TaskRun.validate_tags`, but raised here so the user sees a row-level
    error (`Error in row N: ...`) rather than a pydantic data-model path
    (`tags -> 0: ...`) when constructing TaskRuns downstream.
    """
    for tag in tags:
        if not tag:
            raise KilnInvalidImportFormat(
                "Tags cannot be empty strings.",
                row_number=row_number,
            )
        if " " in tag:
            raise KilnInvalidImportFormat(
                f"Tags cannot contain spaces. Try underscores. Got: '{tag}'.",
                row_number=row_number,
            )


def _validate_trace(trace_str: str, row_number: int) -> list[ValidatedMessage]:
    """Parse and validate a multiturn `trace` JSON string. Returns validated messages."""

    try:
        trace = json.loads(trace_str)
    except json.JSONDecodeError as e:
        raise KilnInvalidImportFormat(
            "trace is not valid JSON.",
            row_number=row_number,
        ) from e

    if not isinstance(trace, list):
        raise KilnInvalidImportFormat(
            "trace must be a JSON array of messages.",
            row_number=row_number,
        )
    if len(trace) < 2:
        raise KilnInvalidImportFormat(
            "trace must contain at least one user message followed by one assistant message.",
            row_number=row_number,
        )

    messages: list[ValidatedMessage] = []
    for k, msg in enumerate(trace, start=1):
        if not isinstance(msg, dict):
            raise KilnInvalidImportFormat(
                f"message {k}: must be a JSON object.",
                row_number=row_number,
            )

        role = msg.get("role")
        if role is None:
            raise KilnInvalidImportFormat(
                f"message {k}: 'role' is required.",
                row_number=row_number,
            )
        if role in ("system", "developer"):
            raise KilnInvalidImportFormat(
                f"message {k}: trace contains a {role} message. Multiturn tasks define "
                "their system prompt on the task itself, not per-conversation. Remove "
                "system/developer messages from your CSV, or update the task's system "
                "prompt to match.",
                row_number=row_number,
            )
        if role == "tool" or "tool_calls" in msg:
            raise KilnInvalidImportFormat(
                f"message {k}: tool calls and tool messages are not supported in CSV import.",
                row_number=row_number,
            )
        if role not in ALLOWED_MULTITURN_ROLES:
            raise KilnInvalidImportFormat(
                f"message {k}: unsupported role '{role}'. Allowed: user, assistant.",
                row_number=row_number,
            )

        content = msg.get("content")
        if not isinstance(content, str) or not content:
            raise KilnInvalidImportFormat(
                f"message {k}: 'content' must be a non-empty string.",
                row_number=row_number,
            )

        # Alternation: 1-indexed odd positions are user, even are assistant.
        expected = "user" if k % 2 == 1 else "assistant"
        if role != expected:
            raise KilnInvalidImportFormat(
                f"message {k}: expected role '{expected}', got '{role}'.",
                row_number=row_number,
            )

        reasoning: str | None = None
        if role != "assistant" and "reasoning_content" in msg:
            raise KilnInvalidImportFormat(
                f"message {k}: 'reasoning_content' is only allowed on assistant messages.",
                row_number=row_number,
            )
        if role == "assistant":
            rc = msg.get("reasoning_content")
            if rc is not None:
                if not isinstance(rc, str):
                    raise KilnInvalidImportFormat(
                        f"message {k}: 'reasoning_content' must be a string.",
                        row_number=row_number,
                    )
                if not rc:
                    raise KilnInvalidImportFormat(
                        f"message {k}: 'reasoning_content' must be a non-empty string.",
                        row_number=row_number,
                    )
                reasoning = rc

        # At this point, role is narrowed to Literal["user", "assistant"] by the
        # membership check above; assert to make the narrowing explicit to type
        # checkers that can't read through `set[str]`.
        assert role == "user" or role == "assistant"
        messages.append(
            ValidatedMessage(role=role, content=content, reasoning_content=reasoning)
        )

    if messages[-1].role != "assistant":
        raise KilnInvalidImportFormat(
            "trace must end with an assistant message.",
            row_number=row_number,
        )

    return messages


def _to_openai_message(message: ValidatedMessage) -> ChatCompletionMessageParam:
    """Render a ValidatedMessage as a typed OpenAI chat completion message."""

    if message.role == "user":
        user_msg: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": message.content,
        }
        return user_msg

    assistant_msg: ChatCompletionAssistantMessageParamWrapper = {
        "role": "assistant",
        "content": message.content,
    }
    if message.reasoning_content:
        assistant_msg["reasoning_content"] = message.reasoning_content
    return assistant_msg


def _build_chain(
    task: Task,
    messages: list[ValidatedMessage],
    file_name: str,
    session_id: str,
    csv_tags: list[str],
) -> list[TaskRun]:
    """Build a chain of TaskRuns from a validated trace. Order: root → leaf."""

    base_tags = generate_import_tags(session_id) + list(csv_tags)
    chain: list[TaskRun] = []

    for turn_index in range(0, len(messages), 2):
        user_msg = messages[turn_index]
        assistant_msg = messages[turn_index + 1]

        cumulative_trace = [_to_openai_message(m) for m in messages[: turn_index + 2]]

        intermediate: dict[str, str] = {}
        if assistant_msg.reasoning_content:
            intermediate["reasoning"] = assistant_msg.reasoning_content

        run = TaskRun(
            parent=task,
            input=user_msg.content,
            input_source=DataSource(
                type=DataSourceType.file_import,
                properties={"file_name": file_name},
            ),
            output=TaskOutput(
                output=assistant_msg.content,
                source=DataSource(
                    type=DataSourceType.file_import,
                    properties={"file_name": file_name},
                ),
            ),
            intermediate_outputs=intermediate or None,
            trace=cumulative_trace,
            parent_task_run_id=chain[-1].id if chain else None,
            tags=list(base_tags),
        )
        chain.append(run)

    return chain


def _import_csv_multiturn(
    task: Task,
    config: ImportConfig,
) -> ImportResult:
    """Import a multiturn CSV: one row per conversation, each materialized as a TaskRun chain."""

    session_id = str(int(time.time()))
    dataset_path = config.dataset_path
    dataset_name = config.dataset_name
    tag_splits = config.tag_splits

    required_headers = {"trace"}
    optional_headers = {"tags"}

    chains: list[list[TaskRun]] = []
    with open(dataset_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        if not reader.fieldnames:
            raise KilnInvalidImportFormat(
                "CSV file appears to be empty or missing headers"
            )

        actual_headers = set(reader.fieldnames)
        # Detect a single-turn-shaped CSV uploaded to a multiturn task and
        # tell the user how to fix it. Fires unconditionally on `input`/`output`
        # so single-turn data isn't silently dropped when `trace` is also present.
        if "input" in actual_headers or "output" in actual_headers:
            raise KilnInvalidImportFormat(
                "Task is multiturn; expected column: trace (and optional tags). "
                f"Got: {', '.join(sorted(actual_headers))}."
            )
        missing_headers = required_headers - actual_headers
        if missing_headers:
            raise KilnInvalidImportFormat(
                f"Missing required headers: {', '.join(missing_headers)}. "
                f"Required headers are: {', '.join(required_headers)}"
            )

        unknown_headers = actual_headers - (required_headers | optional_headers)
        if unknown_headers:
            logger.warning(
                f"Unknown headers in CSV file will be ignored: {', '.join(unknown_headers)}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                validated_row = CSVMultiturnRowSchema.model_validate(
                    {
                        **row,
                        "tags": deserialize_tags(row.get("tags")),
                    }
                )
            except ValidationError as e:
                logger.warning(f"Invalid row {row_number}: {row}", exc_info=True)
                raise KilnInvalidImportFormat(
                    format_validation_error(e),
                    row_number=row_number,
                ) from e

            _validate_csv_tags(validated_row.tags, row_number)
            messages = _validate_trace(validated_row.trace, row_number)
            chain = _build_chain(
                task=task,
                messages=messages,
                file_name=dataset_name,
                session_id=session_id,
                csv_tags=validated_row.tags,
            )
            chains.append(chain)

    # Splits apply only to leaves; intermediate runs are filtered out of
    # dataset views and downstream sets.
    leaves = [chain[-1] for chain in chains]
    add_tag_splits(leaves, tag_splits)

    all_runs = [run for chain in chains for run in chain]
    _save_runs_with_rollback(all_runs)

    return ImportResult(
        imported_run_count=len(all_runs),
        imported_conversation_count=len(chains),
    )


DATASET_IMPORTERS: Dict[DatasetImportFormat, Importer] = {
    DatasetImportFormat.CSV: import_csv,
}


class DatasetFileImporter:
    """Import a dataset from a file"""

    def __init__(self, task: Task, config: ImportConfig):
        self.task = task
        config.validate_tag_splits()
        self.config = config

    def create_runs_from_file(self) -> ImportResult:
        fn = DATASET_IMPORTERS[self.config.dataset_type]
        return fn(
            self.task,
            self.config,
        )
