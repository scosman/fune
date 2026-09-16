"""
Tools for splitting datasets into train/test/validation splits. Includes filters for selecting which task runs to include in each split.
"""

import math
import random
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from kiln_ai.datamodel.basemodel import FilenameString, KilnParentedModel
from kiln_ai.datamodel.dataset_filters import (
    DatasetFilter,
    DatasetFilterId,
    dataset_filter_from_id,
)
from kiln_ai.datamodel.run_config import KilnAgentRunConfigProperties
from kiln_ai.datamodel.task_run import TaskRun

if TYPE_CHECKING:
    from kiln_ai.datamodel.task import Task


class DatasetToolInfo(BaseModel):
    """
    Information about tools used across task runs in a dataset split.
    """

    has_tool_mismatch: bool = Field(
        description="Whether the tools from each run match across all runs in the dataset split."
    )
    tools: list[str] | None = Field(
        default_factory=list,
        description="Common tool IDs shared by every run. Empty means every run has no tools. None means the dataset contains mismatched tool sets.",
    )


class DatasetSplitDefinition(BaseModel):
    """
    A definition of a split in a dataset.

    Example: name="train", description="The training set", percentage=0.8 (80% of the dataset)
    """

    name: FilenameString = Field(
        description="The name of the dataset split definition."
    )
    description: str | None = Field(
        default=None,
        description="A description of the dataset for you and your team. Not used in training.",
    )
    percentage: float = Field(
        ge=0.0,
        le=1.0,
        description="The percentage of the dataset that this split represents (between 0 and 1).",
    )


AllSplitDefinition: list[DatasetSplitDefinition] = [
    DatasetSplitDefinition(name="all", percentage=1.0)
]
Train80Test20SplitDefinition: list[DatasetSplitDefinition] = [
    DatasetSplitDefinition(name="train", percentage=0.8),
    DatasetSplitDefinition(name="test", percentage=0.2),
]
Train80Val20SplitDefinition: list[DatasetSplitDefinition] = [
    DatasetSplitDefinition(name="train", percentage=0.8),
    DatasetSplitDefinition(name="val", percentage=0.2),
]
Train60Test20Val20SplitDefinition: list[DatasetSplitDefinition] = [
    DatasetSplitDefinition(name="train", percentage=0.6),
    DatasetSplitDefinition(name="test", percentage=0.2),
    DatasetSplitDefinition(name="val", percentage=0.2),
]
Train80Test10Val10SplitDefinition: list[DatasetSplitDefinition] = [
    DatasetSplitDefinition(name="train", percentage=0.8),
    DatasetSplitDefinition(name="test", percentage=0.1),
    DatasetSplitDefinition(name="val", percentage=0.1),
]


class DatasetSplit(KilnParentedModel):
    """
    A collection of task runs, with optional splits (train, test, validation).

    Used to freeze a dataset into train/test/validation splits for repeatable fine-tuning or other tasks.

    Maintains a list of IDs for each split, to avoid data duplication.
    """

    name: FilenameString = Field(description="The name of the dataset split.")
    description: str | None = Field(
        default=None,
        description="A description of the dataset for you and your team. Not used in training.",
    )
    splits: list[DatasetSplitDefinition] = Field(
        default_factory=list,
        description="The splits in the dataset.",
    )
    split_contents: dict[str, list[str]] = Field(
        description="The contents of each split in the dataset. The key is the split name, and the value is a list of task run IDs.",
    )
    filter: DatasetFilterId | None = Field(
        default=None,
        description="The filter used to build the dataset.",
    )

    @model_validator(mode="after")
    def validate_split_percentages(self) -> "DatasetSplit":
        total = sum(split.percentage for split in self.splits)
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValueError(f"The sum of split percentages must be 1.0 (got {total})")
        return self

    @classmethod
    def from_task(
        cls,
        name: str,
        task: "Task",
        splits: list[DatasetSplitDefinition],
        filter_id: DatasetFilterId = "all",
        description: str | None = None,
    ):
        """
        Build a dataset split from a task.
        """
        filter = dataset_filter_from_id(filter_id)
        split_contents = cls.build_split_contents(task, splits, filter)
        return cls(
            parent=task,
            name=name,
            description=description,
            splits=splits,
            split_contents=split_contents,
            filter=filter_id,
        )

    @classmethod
    def build_split_contents(
        cls,
        task: "Task",
        splits: list[DatasetSplitDefinition],
        filter: DatasetFilter,
    ) -> dict[str, list[str]]:
        # Datasets must not include intermediate multiturn turns — only leaves.
        runs = list(task.runs())
        valid_ids = []
        for task_run in runs:
            if filter(task_run):
                valid_ids.append(task_run.id)

        # Shuffle and split by split percentage
        random.shuffle(valid_ids)
        split_contents = {}
        start_idx = 0
        remaining_items = len(valid_ids)

        # Handle all splits except the last one
        for split in splits[:-1]:
            split_size = round(len(valid_ids) * split.percentage)
            split_contents[split.name] = valid_ids[start_idx : start_idx + split_size]
            start_idx += split_size
            remaining_items -= split_size

        # Last split gets all remaining items (for rounding)
        if splits:
            split_contents[splits[-1].name] = valid_ids[start_idx:]

        return split_contents

    def parent_task(self) -> "Task | None":
        # inline import to avoid circular import
        from kiln_ai.datamodel import Task

        if not isinstance(self.parent, Task):
            return None
        return self.parent

    def missing_count(self) -> int:
        """
        Returns:
            int: the number of task runs that have an ID persisted in this dataset split, but no longer exist in the dataset
        """
        parent = self.parent_task()
        if parent is None:
            raise ValueError("DatasetSplit has no parent task")

        # Include intermediate runs: a split snapshots run IDs at creation
        # time (leaves-only), but the user can later extend a multiturn
        # conversation - turning a run that was a leaf when the split was
        # built into an intermediate. The run is still on disk, so it
        # shouldn't be reported as "missing".
        runs = parent.runs(include_intermediate_runs=True, readonly=True)
        all_ids = set(run.id for run in runs)
        all_ids_in_splits = set()
        for ids in self.split_contents.values():
            all_ids_in_splits.update(ids)
        missing = all_ids_in_splits - all_ids
        return len(missing)

    def _get_runs(self) -> list[TaskRun]:
        """
        Get all task runs referenced in this dataset split.

        Returns:
            list[TaskRun]: list of task runs in this dataset split
        """
        parent = self.parent_task()
        if parent is None:
            return []

        runs = []
        all_run_ids = set()
        for run_ids in self.split_contents.values():
            all_run_ids.update(run_ids)

        # Include intermediate runs: a split snapshots run IDs at creation
        # time (leaves-only), but the user can later extend a multiturn
        # conversation - turning a run that was a leaf when the split was
        # built into an intermediate. We still want to resolve it.
        for task_run in parent.runs(include_intermediate_runs=True, readonly=True):
            if task_run.id in all_run_ids:
                runs.append(task_run)

        return runs

    @staticmethod
    def compute_tool_info(runs: list[TaskRun]) -> DatasetToolInfo:
        """
        Compute tool info from a list of task runs.

        Args:
            runs: list of task runs to analyze

        Returns:
            DatasetToolInfo: information about tools used across the task runs
        """

        has_tool_mismatch = False
        tools: set[str] | None = None

        for run in runs:
            # Extract tools from run config, treating missing source/run_config/tools_config as empty tools
            run_tools: set[str] = set()
            source = run.output.source if run.output else None
            if source is not None and isinstance(
                source.run_config, KilnAgentRunConfigProperties
            ):
                tools_config = source.run_config.tools_config
                if tools_config is not None:
                    run_tools = set(tools_config.tools)

            # First run establishes the expected tool set (including empty)
            if tools is None:
                tools = run_tools
            elif run_tools != tools:
                # Mismatch found
                has_tool_mismatch = True
                tools = None
                break

        # If no valid runs were processed, return empty tools
        if tools is None:
            if not has_tool_mismatch:
                tools = set()

        return DatasetToolInfo(
            has_tool_mismatch=has_tool_mismatch,
            tools=None if tools is None else sorted(tools),
        )

    def tool_info(self) -> DatasetToolInfo:
        """
        Helper method to compute tool info for the dataset split. Iterate through all runs in the dataset split and check the tools used in each run config.

        Returns:
            DatasetToolInfo: information about tools used across task runs in this dataset split
        """
        runs = self._get_runs()
        tool_info = self.compute_tool_info(runs)
        return tool_info
