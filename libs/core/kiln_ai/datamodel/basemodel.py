import json
import os
import re
import shutil
import tempfile
import unicodedata
import uuid
from abc import ABCMeta
from builtins import classmethod
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SerializationInfo,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_core import ErrorDetails
from typing_extensions import Annotated, Self

from kiln_ai.datamodel.model_cache import ModelCache
from kiln_ai.utils.config import Config
from kiln_ai.utils.formatting import snake_case
from kiln_ai.utils.mime_type import guess_extension


def generate_model_id() -> str:
    return str(uuid.uuid4().int)[:12]


# ID is a 12 digit random integer string.
# Should be unique per item, at least inside the context of a parent/child relationship.
# Use integers to make it easier to type for a search function.
# Allow none, even though we generate it, because we clear it in the REST API if the object is ephemeral (not persisted to disk)
ID_FIELD = Field(
    default_factory=generate_model_id,
    description="Unique identifier for this record.",
)
ID_TYPE = Optional[str]
T = TypeVar("T", bound="KilnBaseModel")
PT = TypeVar("PT", bound="KilnParentedModel")


class ReadOnlyMutationError(RuntimeError):
    """Raised when attempting to mutate a readonly KilnBaseModel instance."""

    pass


# Naming conventions:
# 1) Names are filename safe as they may be used as file names. They are informational and not to be used in prompts/training/validation.
# 2) Descriptions are for Kiln users to describe/understanding the purpose of this object. They must never be used in prompts/training/validation. Use "instruction/requirements" instead.

# Forbidden chars are not allowed in filenames on one or more platforms.
# ref: https://en.wikipedia.org/wiki/Filename#Problematic_characters
FORBIDDEN_CHARS_REGEX = r"[/\\?%*:|\"<>.,;=\n]"
FORBIDDEN_CHARS = "/ \\ ? % * : | < > . , ; = \\n"

MAX_FILENAME_LENGTH = 120


def name_validator(*, min_length: int, max_length: int) -> Callable[[Any], str]:
    def fn(name: Any) -> str:
        if name is None:
            raise ValueError("Name is required")
        if not isinstance(name, str):
            raise ValueError(f"Input should be a valid string, got {type(name)}")
        if len(name) < min_length:
            raise ValueError(
                f"Name is too short. Min length is {min_length} characters, got {len(name)}"
            )
        if len(name) > max_length:
            raise ValueError(
                f"Name is too long. Max length is {max_length} characters, got {len(name)}"
            )
        if string_to_valid_name(name) != name:
            raise ValueError(
                f"Name is invalid. The name cannot contain any of the following characters: {FORBIDDEN_CHARS}, consecutive whitespace/underscores, or leading/trailing whitespace/underscores"
            )
        return name

    return fn


def string_to_valid_name(name: str, truncate_to_max_length: bool = False) -> str:
    """Convert a string to a valid filename.
    Args:
        name (str): The string to convert.
        truncate_to_max_length (bool): If True, the name will be truncated to the maximum filename length.
    Returns:
        str: The valid filename.
    """
    # https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize
    valid_name = unicodedata.normalize("NFKD", name)
    # Replace any forbidden chars with an underscore
    valid_name = re.sub(FORBIDDEN_CHARS_REGEX, " ", valid_name)
    # Replace control characters with an underscore
    valid_name = re.sub(r"[\x00-\x1F]", " ", valid_name)
    # Replace consecutive whitespace with a single space
    valid_name = re.sub(r"\s+", " ", valid_name)
    # Replace consecutive underscores with a single underscore
    valid_name = re.sub(r"_+", "_", valid_name)
    # Remove leading and trailing underscores or whitespace
    valid_name = valid_name.strip("_").strip()
    if truncate_to_max_length and len(valid_name) > MAX_FILENAME_LENGTH:
        valid_name = valid_name[:MAX_FILENAME_LENGTH]
        # After truncating, we might have new trailing whitespace, dots, or underscores, so strip again.
        valid_name = valid_name.strip("_").strip().rstrip(".")
    return valid_name


class KilnAttachmentModel(BaseModel):
    path: Path | None = Field(
        default=None,
        description="The path to the attachment relative to the parent model's path",
    )

    input_path: Path | None = Field(
        default=None,
        description="The input absolute path to the attachment. The file will be copied to its permanent location when the model is saved.",
    )

    @model_validator(mode="after")
    def check_file_exists(self, info: ValidationInfo) -> Self:
        context = info.context or {}

        if self.path is None and self.input_path is None:
            raise ValueError("Path or input path is not set")
        if self.path is not None and self.input_path is not None:
            raise ValueError("Path and input path cannot both be set")

        # when loading from file, we only know the relative path so we cannot check if it exists
        # without knowing the parent path
        if context.get("loading_from_file", False):
            if isinstance(self.path, str):
                self.path = Path(self.path)
            self.input_path = None
            if self.path and (
                self.path.is_absolute()
                or PureWindowsPath(self.path).is_absolute()
                or ".." in PureWindowsPath(self.path).parts
            ):
                raise ValueError(
                    f"Attachment path must be a relative path within the parent: {self.path}"
                )
            return self

        # when creating a new attachment, the path is not set yet (it is set when the model is saved)
        # so we only expect the absolute path to be set
        if self.input_path is not None:
            if isinstance(self.input_path, str):
                self.input_path = Path(self.input_path)
            if not self.input_path.is_absolute():
                raise ValueError(f"Input path is not absolute: {self.input_path}")
            if not os.path.exists(self.input_path):
                raise ValueError(f"Input path does not exist: {self.input_path}")
            if not os.path.isfile(self.input_path):
                raise ValueError(f"Input path is not a file: {self.input_path}")

            # this normalizes the path and resolves symlinks
            self.input_path = self.input_path.resolve()

            return self

        if self.path is not None:
            if isinstance(self.path, str):
                self.path = Path(self.path)
            if self.path.is_absolute():
                raise ValueError(
                    f"Path is absolute but should be relative: {self.path}"
                )
            if not os.path.exists(self.path):
                raise ValueError(f"Path does not exist: {self.path}")
            if not os.path.isfile(self.path):
                raise ValueError(f"Path is not a file: {self.path}")

        return self

    @model_serializer
    def serialize(self, info: SerializationInfo) -> dict[str, Path] | None:
        # when the attachment is optional on the model, we get None here
        if self is None:
            return None

        context = info.context or {}

        # serialization may also be called by other parts of the system, the callers should
        # explicitly set save_attachments to True if they want to save attachments
        save_attachments: bool = context.get("save_attachments", False)
        if not save_attachments:
            path_val = self.path if self.path is not None else self.input_path
            if path_val is None:
                raise ValueError("Attachment has no path")
            return {"path": path_val}

        dest_path: Path | None = context.get("dest_path", None)
        if not dest_path or not isinstance(dest_path, Path):
            raise ValueError(
                f"dest_path must be a valid Path object when saving attachments, got: {dest_path}"
            )
        if not dest_path.is_dir():
            raise ValueError("dest_path must be a directory when saving attachments")

        # the attachment is already in the parent folder, so we don't need to copy it
        # if the path is already relative, we consider it has been copied already
        if self.path is not None:
            return {"path": self.path}

        # copy file and update the path to be relative to the dest_path
        new_path = self.copy_file_to(dest_path, context.get("filename_prefix", None))

        self.path = new_path.relative_to(dest_path)
        self.input_path = None

        return {"path": self.path}

    def copy_file_to(
        self, dest_folder: Path, filename_prefix: str | None = None
    ) -> Path:
        if self.input_path is None:
            raise ValueError("Attachment has no input path to copy")

        filename = f"{str(uuid.uuid4().int)[:12]}{self.input_path.suffix}"
        if filename_prefix:
            filename = f"{filename_prefix}_{filename}"
        target_path = dest_folder / filename
        shutil.copy(self.input_path, target_path)
        return target_path

    @classmethod
    def from_data(cls, data: str | bytes, mime_type: str) -> Self:
        """Create an attachment from str or byte data, in a temp file. The attachment is persisted to
        its permanent location when the model is saved.
        """
        extension = guess_extension(mime_type) or ".unknown"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
        if isinstance(data, str):
            temp_file.write(data.encode("utf-8"))
        else:
            temp_file.write(data)
        temp_file.close()
        return cls(input_path=Path(temp_file.name))

    @classmethod
    def from_file(cls, path: Path | str) -> Self:
        """Create an attachment from a file path. The attachment is persisted to
        its permanent location when the model is saved.
        """
        if isinstance(path, str):
            path = Path(path)
        return cls(input_path=path)

    def resolve_path(self, parent_path: Path | None = None) -> Path:
        """
        Resolve the path of the attachment relative to the parent path. The attachment does not know
        its parent, so we need to call this to get the full path.
        Args:
            parent_path (Path): The absolute path to the parent folder. Must be provided if the model is saved to disk.
        Returns:
            Path: The resolved path of the attachment
        """
        if self.input_path is not None:
            return self.input_path
        if self.path is None:
            raise ValueError("Attachment path is not set")
        if parent_path is None:
            raise ValueError("Parent path is not set")
        if not parent_path.is_absolute():
            raise ValueError(
                f"Failed to resolve attachment path for {self.path} because parent path is not absolute: {parent_path}"
            )
        parent_resolved = parent_path.resolve()
        resolved = (parent_resolved / self.path).resolve()
        try:
            resolved.relative_to(parent_resolved)
        except ValueError:
            raise ValueError(
                f"Attachment path {self.path!r} escapes its parent directory {parent_resolved}"
            )
        return resolved


# Usage:
# class MyModel(KilnBaseModel):
#     name: FilenameString = Field(description="The name of the model.")
#     name_short: FilenameStringShort = Field(description="The short name of the model.")
FilenameString = Annotated[
    str,
    BeforeValidator(name_validator(min_length=1, max_length=MAX_FILENAME_LENGTH)),
    StringConstraints(min_length=1, max_length=MAX_FILENAME_LENGTH),
]
FilenameStringShort = Annotated[
    str,
    BeforeValidator(name_validator(min_length=1, max_length=32)),
    StringConstraints(min_length=1, max_length=32),
]


class KilnBaseModel(BaseModel):
    """Base model for all Kiln data models with common functionality for persistence and versioning.

    Attributes:
        v (int): Schema version number for migration support
        id (str): Unique identifier for the model instance
        path (Path): File system path where the model is stored
        created_at (datetime): Timestamp when the model was created
        created_by (str): User ID of the creator
    """

    model_config = ConfigDict(validate_assignment=True)

    v: int = Field(default=1, description="Schema version for migration support.")
    id: ID_TYPE = ID_FIELD
    path: Optional[Path] = Field(
        default=None, description="File system path where the record is stored."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now().astimezone(),
        description="Timestamp when the model was created. Timezone-aware; stores the writer's local offset.",
    )

    @field_validator("created_at", mode="after")
    @classmethod
    def _normalize_created_at_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.astimezone()
        return v

    created_by: str = Field(
        default_factory=lambda: Config.shared().user_id,
        description="User ID of the creator.",
    )

    _loaded_from_file: bool = False
    _readonly: bool = False

    @computed_field()
    def model_type(self) -> str:
        return self.type_name()

    def mark_as_readonly(self) -> None:
        """Mark this model instance as readonly to prevent mutations."""
        self._readonly = True

    def _ensure_not_readonly(self, attr_name: str) -> None:
        """Check if model is readonly and raise exception if it is"""
        if self._readonly:
            attr_msg = f" '{attr_name}'" if attr_name else ""
            raise ReadOnlyMutationError(
                f"Cannot mutate readonly model{attr_msg} of type {self.__class__.__name__}. "
                f"Load with readonly=False to get a mutable copy."
            )

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to prevent mutations on readonly models."""
        # Check for error condition: attempting to mutate readonly model
        # Allow private attributes and certain safe operations even on readonly models
        readonly_safe_attrs = {"parent", "path"}
        if (
            not name.startswith("_")
            and self._readonly
            and name not in readonly_safe_attrs
        ):
            self._ensure_not_readonly(name)

        # proceed with attribute setting
        super().__setattr__(name, value)

    def mutable_copy(self) -> Self:
        """Create a mutable copy of the model, resetting readonly flag."""
        copy = super().model_copy(deep=True)
        # Reset readonly flag on copies so they can be mutated
        copy._readonly = False
        return copy  # type: ignore[return-value]

    # if changing the model name, should keep the original name here for parsing old files
    @classmethod
    def type_name(cls) -> str:
        return snake_case(cls.__name__)

    # used as /obj_folder/base_filename.kiln
    @classmethod
    def base_filename(cls) -> str:
        return cls.type_name() + ".kiln"

    @classmethod
    def load_from_folder(cls: Type[T], folderPath: Path) -> T:
        """Load a model instance from a folder using the default filename.

        Args:
            folderPath (Path): Directory path containing the model file

        Returns:
            T: Instance of the model
        """
        path = folderPath / cls.base_filename()
        return cls.load_from_file(path)

    @classmethod
    def load_from_file(cls: Type[T], path: Path | str, readonly: bool = False) -> T:
        """Load a model instance from a specific file path.

        Args:
            path (Path): Path to the model file
            readonly (bool): If True, the model will be returned in readonly mode (cached instance, not a copy, not safe to mutate)

        Returns:
            T: Instance of the model

        Raises:
            ValueError: If the loaded model is not of the expected type or version
            FileNotFoundError: If the file does not exist
        """
        if isinstance(path, str):
            path = Path(path)
        cached_model = ModelCache.shared().get_model(path, cls, readonly=readonly)
        if cached_model is not None:
            return cached_model
        with open(path, "r", encoding="utf-8") as file:
            # modified time of file for cache invalidation. From file descriptor so it's atomic w read.
            mtime_ns = os.fstat(file.fileno()).st_mtime_ns
            file_data = file.read()
            parsed_json = json.loads(file_data)
            m = cls.model_validate(
                parsed_json,
                context={"loading_from_file": True, "source_dir": path.parent},
            )
            if not isinstance(m, cls):
                raise ValueError(f"Loaded model is not of type {cls.__name__}")
            m._loaded_from_file = True
            file_data = None
        m.path = path
        if m.v > m.max_schema_version():
            raise ValueError(
                f"Cannot load from file because the schema version is higher than the current version. Upgrade kiln to the latest version. "
                f"Class: {m.__class__.__name__}, id: {getattr(m, 'id', None)}, path: {path}, "
                f"version: {m.v}, max version: {m.max_schema_version()}"
            )
        if parsed_json["model_type"] != cls.type_name():
            raise ValueError(
                f"Cannot load from file because the model type is incorrect. Expected {cls.type_name()}, got {parsed_json['model_type']}. "
                f"Class: {m.__class__.__name__}, id: {getattr(m, 'id', None)}, path: {path}, "
                f"version: {m.v}, max version: {m.max_schema_version()}"
            )

        if readonly:
            m.mark_as_readonly()
            # Cache, but only if readonly. Mutable models should not be cached.
            ModelCache.shared().set_model(path, m, mtime_ns)

        return m

    def loaded_from_file(self, info: ValidationInfo | None = None) -> bool:
        # Two methods of indicated it's loaded from file:
        # 1) info.context.get("loading_from_file") -> During actual loading, before we can set _loaded_from_file
        # 2) self._loaded_from_file -> After loading, set by the loader
        if self.loading_from_file(info):
            return True
        return self._loaded_from_file

    # indicates the model is currently being loaded from file (not mutating it after)
    def loading_from_file(self, info: ValidationInfo | None = None) -> bool:
        # info.context.get("loading_from_file") -> During actual loading, before we can set _loaded_from_file
        if (
            info is not None
            and info.context is not None
            and info.context.get("loading_from_file", False)
        ):
            return True
        return False

    def save_to_file(self) -> None:
        """Save the model instance to a file.

        Raises:
            ValueError: If the path is not set
        """
        path = self.build_path()
        if path is None:
            raise ValueError(
                f"Cannot save to file because 'path' is not set. Class: {self.__class__.__name__}, "
                f"id: {getattr(self, 'id', None)}, path: {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)

        json_data = self.model_dump_json(
            indent=2,
            exclude={"path"},
            # dest_path is used by the attachment serializer to save attachments to the correct location
            # and update the paths to be relative to path.parent
            context={
                "save_attachments": True,
                "dest_path": path.parent,
            },
        )
        with open(path, "w", encoding="utf-8") as file:
            file.write(json_data)
        # save the path so even if something like name changes, the file doesn't move
        self.path = path
        # We could save, but invalidating will trigger load on next use.
        # This ensures everything in cache is loaded from disk, and the cache perfectly reflects what's on disk
        ModelCache.shared().invalidate(path)

    def delete(self) -> None:
        if self.path is None:
            raise ValueError("Cannot delete model because path is not set")
        dir_path = self.path.parent if self.path.is_file() else self.path
        if dir_path is None:
            raise ValueError("Cannot delete model because path is not set")
        shutil.rmtree(dir_path)
        ModelCache.shared().invalidate(self.path)
        self.path = None

    def build_path(self) -> Path | None:
        if self.path is not None:
            return self.path
        return None

    # increment for breaking changes
    def max_schema_version(self) -> int:
        return 1


class ChildLoadError(BaseModel):
    """A child model file that could not be loaded, and why.

    Returned by ``KilnParentedModel.all_children_of_parent_path_with_errors`` so callers
    can surface partial results instead of failing the whole load.
    """

    model_config = ConfigDict(frozen=True)

    path: Path = Field(description="Path of the child file that failed to load.")
    message: str = Field(description="Why the child file failed to load.")


class KilnParentedModel(KilnBaseModel, metaclass=ABCMeta):
    """Base model for Kiln models that have a parent-child relationship. This base class is for child models.

    This class provides functionality for managing hierarchical relationships between models,
    including parent reference handling and file system organization.

    Attributes:
        parent (KilnBaseModel): Reference to the parent model instance. Not persisted, just in memory.
    """

    # Parent is an in memory only reference to parent. If it's set we use that. If not we'll try to load it from disk based on the path.
    # We don't persist the parent reference to disk. See the accessors below for how we make it a clean api (parent accessor will lazy load from disk)
    parent: Optional[KilnBaseModel] = Field(default=None, exclude=True)

    def __getattribute__(self, name: str) -> Any:
        if name == "parent":
            return self.load_parent()
        return super().__getattribute__(name)

    def cached_parent(self) -> Optional[KilnBaseModel]:
        return object.__getattribute__(self, "parent")

    def load_parent(self) -> Optional[KilnBaseModel]:
        """Get the parent model instance, loading it from disk if necessary.

        Returns:
            Optional[KilnBaseModel]: The parent model instance or None if not set
        """
        cached_parent = self.cached_parent()
        if cached_parent is not None:
            return cached_parent

        # lazy load parent from path
        if self.path is None:
            return None
        # Note: this only works with base_filename. If we every support custom names, we need to change this.
        parent_path = (
            self.path.parent.parent.parent
            / self.__class__.parent_type().base_filename()
        )
        if parent_path is None:
            return None
        if not parent_path.exists():
            return None
        loaded_parent = self.__class__.parent_type().load_from_file(parent_path)
        self.parent = loaded_parent
        return loaded_parent

    # Dynamically implemented by KilnParentModel method injection
    @classmethod
    def relationship_name(cls) -> str:
        raise NotImplementedError("Relationship name must be implemented")

    # Dynamically implemented by KilnParentModel method injection
    @classmethod
    def parent_type(cls) -> Type[KilnBaseModel]:
        raise NotImplementedError("Parent type must be implemented")

    @model_validator(mode="after")
    def check_parent_type(self) -> Self:
        cached_parent = self.cached_parent()
        if cached_parent is not None:
            expected_parent_type = self.__class__.parent_type()
            if not isinstance(cached_parent, expected_parent_type):
                raise ValueError(
                    f"Parent must be of type {expected_parent_type}, but was {type(cached_parent)}"
                )

        return self

    def build_child_dirname(self) -> Path:
        # Default implementation for readable folder names.
        # {id} - {name}/{type}.kiln
        if self.id is None:
            # consider generating an ID here. But if it's been cleared, we've already used this without one so raise for now.
            raise ValueError("ID is not set - can not save or build path")
        path = self.id
        name = getattr(self, "name", None)
        if name is not None:
            name_part = string_to_valid_name(name[:32], truncate_to_max_length=False)
            if not name_part:
                raise ValueError(
                    "Cannot build child folder dirname: `string_to_valid_name` on the "
                    f"first 32 characters of `name` produced an empty string (name length "
                    f"{len(name)}). Use leading characters that are not only whitespace, "
                    "underscores, or characters removed by filename sanitization."
                )
            path = f"{path} - {name_part}"
        return Path(path)

    def build_path(self) -> Path | None:
        # if specifically loaded from an existing path, keep that no matter what
        # this ensures the file structure is easy to use with git/version control
        # and that changes to things like name (which impacts default path) don't leave dangling files
        if self.path is not None:
            return self.path
        # Build a path under parent_folder/relationship/file.kiln
        if self.parent is None:
            return None
        parent_path = self.parent.build_path()
        if parent_path is None:
            return None
        parent_folder = parent_path.parent
        if parent_folder is None:
            return None
        return (
            parent_folder
            / self.__class__.relationship_name()
            / self.build_child_dirname()
            / self.__class__.base_filename()
        )

    @classmethod
    def iterate_children_paths_of_parent_path(cls: Type[PT], parent_path: Path | None):
        if parent_path is None:
            # children are disk based. If not saved, they don't exist
            return []

        # Determine the parent folder
        if parent_path.is_file():
            parent_folder = parent_path.parent
        else:
            parent_folder = parent_path

        parent = cls.parent_type().load_from_file(parent_path)
        if parent is None:
            raise ValueError("Parent must be set to load children")

        # Ignore type error: this is abstract base class, but children must implement relationship_name
        relationship_folder = parent_folder / Path(cls.relationship_name())  # type: ignore

        if not relationship_folder.exists() or not relationship_folder.is_dir():
            return []

        # Collect all /relationship/{id}/{base_filename.kiln} files in the relationship folder
        # manual code instead of glob for performance (5x speedup over glob)

        base_filename = cls.base_filename()
        # Iterate through immediate subdirectories using scandir for better performance
        # Benchmark: scandir is 10x faster than glob, so worth the extra code
        with os.scandir(relationship_folder) as entries:
            for entry in entries:
                if not entry.is_dir():
                    continue

                child_file = Path(entry.path) / base_filename
                if child_file.is_file():
                    yield child_file

    @classmethod
    def _load_children_of_parent_path(
        cls: Type[PT],
        parent_path: Path | None,
        readonly: bool,
        error_on_first: bool,
    ) -> tuple[list[PT], list[ChildLoadError]]:
        children: list[PT] = []
        errors: list[ChildLoadError] = []
        for child_path in cls.iterate_children_paths_of_parent_path(parent_path):
            try:
                children.append(cls.load_from_file(child_path, readonly=readonly))
            except Exception as e:
                if error_on_first:
                    raise
                errors.append(ChildLoadError(path=child_path, message=str(e)))
        return children, errors

    @classmethod
    def all_children_of_parent_path(
        cls: Type[PT], parent_path: Path | None, readonly: bool = False
    ) -> list[PT]:
        """Load every child of a parent, raising on the first child that fails to load.

        Use this when a child that can't be loaded should fail the whole operation.
        Use ``all_children_of_parent_path_with_errors`` when partial results are useful.
        """
        children, _ = cls._load_children_of_parent_path(
            parent_path, readonly=readonly, error_on_first=True
        )
        return children

    @classmethod
    def all_children_of_parent_path_with_errors(
        cls: Type[PT], parent_path: Path | None, readonly: bool = False
    ) -> tuple[list[PT], list[ChildLoadError]]:
        """Load every child of a parent, collecting per-child failures instead of raising.

        Project folders sync between clients running different Kiln versions, so a child
        file written by a newer build can be unreadable here while its siblings are fine.
        Returns the children that loaded, and a ``ChildLoadError`` for each one that didn't.

        Errors raised while listing the children (for example an unreadable parent) are
        not per-child failures, and still propagate.
        """
        return cls._load_children_of_parent_path(
            parent_path, readonly=readonly, error_on_first=False
        )

    @classmethod
    def from_id_and_parent_path(
        cls: Type[PT], id: str, parent_path: Path | None
    ) -> PT | None:
        """
        Fast search by ID using the cache. Avoids the model_copy overhead on all but the exact match.

        Uses cache so still slow on first load.
        """
        if parent_path is None:
            return None

        # Note: we're using the in-file ID. We could make this faster using the path-ID if this becomes perf bottleneck, but it's better to have 1 source of truth.
        for child_path in cls.iterate_children_paths_of_parent_path(parent_path):
            child_id = ModelCache.shared().get_model_id(child_path, cls)
            if child_id == id:
                return cls.load_from_file(child_path)
            if child_id is None:
                child = cls.load_from_file(child_path)
                if child.id == id:
                    return child
        return None

    @classmethod
    def from_ids_and_parent_path(
        cls: Type[PT], ids: Set[str], parent_path: Path | None, readonly: bool = False
    ) -> Dict[str, PT]:
        """
        Bulk equivalent of from_id_and_parent_path, much faster for large collections.

        It picks out the matching models from the directory only once. This avoids
        doing individual costly lookups that scan the whole directory in scenarios
        where we need to iterate over a large collection of models (e.g. bulk tagging).

        Pass readonly=True for read-only consumers: cache hits are returned directly,
        skipping a deep copy per matched model.
        """
        if parent_path is None:
            return {}

        children = {}

        # Note: we're using the in-file ID. We could make this faster using the path-ID if this becomes perf bottleneck, but it's better to have 1 source of truth.
        for child_path in cls.iterate_children_paths_of_parent_path(parent_path):
            child_id = ModelCache.shared().get_model_id(child_path, cls)
            if child_id in ids:
                children[child_id] = cls.load_from_file(child_path, readonly=readonly)
            if child_id is None:
                child = cls.load_from_file(child_path, readonly=readonly)
                if child.id in ids:
                    children[child.id] = child

        return children


class ParentOfRelationship(BaseModel):
    """Specifies a parent-child relationship for KilnParentModel.

    Use this form in ``parent_of`` when the Python attribute name should differ
    from the on-disk folder name - for example, when a subclass wants to
    override the auto-generated relationship method with custom filtering
    logic, the relationship can be registered under a private name like
    ``"_runs"`` while preserving the public ``runs/`` folder layout on disk.

    For the common case where both names match, pass the child class directly
    (``parent_of={"foo": FooModel}``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    model: Type[KilnParentedModel]
    filesystem_name: str


# Parent create methods for all child relationships
# You must pass in parent_of in the subclass definition, defining the child relationships
class KilnParentModel(KilnBaseModel, metaclass=ABCMeta):
    """Base model for Kiln models that can have child models.

    This class provides functionality for managing collections of child models and their persistence.
    Child relationships must be defined using the parent_of parameter in the class definition.

    Args:
        parent_of (Dict[str, Type[KilnParentedModel] | ParentOfRelationship]): Mapping of Python
            attribute names to either child model classes (where the key is also used as the
            on-disk folder name) or to ``ParentOfRelationship`` values that decouple the Python
            attribute name from the on-disk folder name.
    """

    @classmethod
    def _create_child_method(
        cls,
        python_name: str,
        child_class: Type[KilnParentedModel],
    ):
        def child_method(self, readonly: bool = False) -> list[child_class]:  # type: ignore[invalid-type-form]
            return child_class.all_children_of_parent_path(self.path, readonly=readonly)

        child_method.__name__ = python_name
        child_method.__annotations__ = {"return": List[child_class]}  # type: ignore[invalid-type-form]
        setattr(cls, python_name, child_method)

    @classmethod
    def _create_parent_methods(
        cls, targetCls: Type[KilnParentedModel], filesystem_name: str
    ):
        def parent_class_method() -> Type[KilnParentModel]:
            return cls

        parent_class_method.__name__ = "parent_type"
        parent_class_method.__annotations__ = {"return": Type[KilnParentModel]}
        setattr(targetCls, "parent_type", parent_class_method)

        def relationship_name_method() -> str:
            return filesystem_name

        relationship_name_method.__name__ = "relationship_name"
        relationship_name_method.__annotations__ = {"return": str}
        setattr(targetCls, "relationship_name", relationship_name_method)

    @classmethod
    def __init_subclass__(
        cls,
        parent_of: Dict[str, Type[KilnParentedModel] | ParentOfRelationship],
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        cls._parent_of = parent_of

        # Reject duplicate child-class registration. `_create_parent_methods`
        # writes `relationship_name` / `parent_type` onto the child class
        # itself, so two entries pointing at the same child would silently
        # overwrite each other - and every on-disk lookup that goes through
        # `cls.relationship_name()` would target whichever filesystem name was
        # registered last. That is a corruption-class bug; fail loudly here.
        seen_child_classes: Dict[Type[KilnParentedModel], str] = {}
        # Reject filesystem-name collisions. Two relationships writing to the
        # same on-disk folder would interleave their files and load each
        # other's children. Same corruption class.
        seen_filesystem_names: Dict[str, str] = {}
        filesystem_name_to_python_name: Dict[str, str] = {}
        for python_name, value in parent_of.items():
            child_class: Type[KilnParentedModel]
            filesystem_name: str
            if isinstance(value, ParentOfRelationship):
                child_class = value.model
                filesystem_name = value.filesystem_name
                filesystem_name_to_python_name[filesystem_name] = python_name
            else:
                child_class = value
                filesystem_name = python_name

            if child_class in seen_child_classes:
                raise ValueError(
                    f"parent_of for {cls.__name__} registers child class "
                    f"{child_class.__name__} twice (under '{seen_child_classes[child_class]}' "
                    f"and '{python_name}'). A child class can only appear once - it "
                    f"holds a single `relationship_name()` / `parent_type()` pair."
                )
            seen_child_classes[child_class] = python_name

            if filesystem_name in seen_filesystem_names:
                raise ValueError(
                    f"parent_of for {cls.__name__} registers filesystem folder "
                    f"'{filesystem_name}' twice (under '{seen_filesystem_names[filesystem_name]}' "
                    f"and '{python_name}'). Each on-disk folder must belong to "
                    f"exactly one relationship."
                )
            seen_filesystem_names[filesystem_name] = python_name

            cls._create_child_method(python_name, child_class)
            cls._create_parent_methods(child_class, filesystem_name)

        # Cache the reverse-lookup used by the misuse safeguard in
        # `_validate_nested`; this is a pure function of `_parent_of` and
        # never changes after class definition.
        cls._filesystem_name_to_python_name = filesystem_name_to_python_name

    @classmethod
    def validate_and_save_with_subrelations(
        cls,
        data: Dict[str, Any],
        path: Path | None = None,
        parent: KilnBaseModel | None = None,
    ):
        """Validate and save a model instance along with all its nested child relationships.

        Args:
            data (Dict[str, Any]): Model data including child relationships
            path (Path, optional): Path where the model should be saved
            parent (KilnBaseModel, optional): Parent model instance for parented models

        Returns:
            KilnParentModel: The validated and saved model instance

        Raises:
            ValidationError: If validation fails for the model or any of its children
        """
        # Validate first, then save. Don't want error half way through, and partly persisted
        # We should save to a tmp dir and move atomically, but need to merge directories later.
        cls._validate_nested(data, save=False, path=path, parent=parent)
        instance = cls._validate_nested(data, save=True, path=path, parent=parent)
        return instance

    @classmethod
    def _validate_nested(
        cls,
        data: Dict[str, Any],
        save: bool = False,
        parent: KilnBaseModel | None = None,
        path: Path | None = None,
    ):
        # Collect all validation errors so we can report them all at once
        validation_errors = []

        # Reject misuse of a relationship's on-disk folder name as a payload
        # key before any side-effects. ``_parent_of`` is set at class-definition
        # time, so this check depends only on ``data`` and ``cls`` - running it
        # before ``save_to_file`` guarantees no half-saved parent on misuse.
        for key in data:
            if key not in cls._parent_of and key in cls._filesystem_name_to_python_name:
                python_name = cls._filesystem_name_to_python_name[key]
                raise ValueError(
                    f"Nested payload key '{key}' matches the on-disk folder name "
                    f"of relationship '{python_name}'. Use the Python attribute "
                    f"name '{python_name}' instead."
                )

        try:
            instance = cls.model_validate(data)
            if path is not None:
                instance.path = path
            if parent is not None and isinstance(instance, KilnParentedModel):
                instance.parent = parent
            if save:
                instance.save_to_file()
        except ValidationError as e:
            instance = None
            for suberror in e.errors():
                validation_errors.append(suberror)

        for key, value_list in data.items():
            if key in cls._parent_of:
                parent_type_or_rel = cls._parent_of[key]
                if isinstance(parent_type_or_rel, ParentOfRelationship):
                    parent_type = parent_type_or_rel.model
                else:
                    parent_type = parent_type_or_rel
                if not isinstance(value_list, list):
                    raise ValueError(
                        f"Expected a list for {key}, but got {type(value_list)}"
                    )
                for value_index, value in enumerate(value_list):
                    try:
                        if issubclass(parent_type, KilnParentModel):
                            kwargs = {"data": value, "save": save}
                            if instance is not None:
                                kwargs["parent"] = instance
                            parent_type._validate_nested(**kwargs)  # type: ignore[invalid-argument-type]
                        elif issubclass(parent_type, KilnParentedModel):
                            # Root node
                            subinstance = parent_type.model_validate(value)
                            if instance is not None:
                                subinstance.parent = instance
                            if save:
                                subinstance.save_to_file()
                        else:
                            raise ValueError(
                                f"Invalid type {parent_type}. Should be KilnBaseModel based."
                            )
                    except ValidationError as e:
                        for suberror in e.errors():
                            cls._append_loc(suberror, key, value_index)
                            validation_errors.append(suberror)

        if len(validation_errors) > 0:
            raise ValidationError.from_exception_data(
                title=f"Validation failed for {cls.__name__}",
                line_errors=validation_errors,
                input_type="json",
            )

        return instance

    @classmethod
    def _append_loc(
        cls, error: ErrorDetails, current_loc: str, value_index: int | None = None
    ):
        orig_loc = error["loc"] if "loc" in error else None
        new_loc: list[str | int] = [current_loc]
        if value_index is not None:
            new_loc.append(value_index)
        if isinstance(orig_loc, tuple):
            new_loc.extend(list(orig_loc))
        elif isinstance(orig_loc, list):
            new_loc.extend(orig_loc)
        error["loc"] = tuple(new_loc)
