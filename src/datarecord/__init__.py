# SPDX-FileCopyrightText: datarecord contributors
#
# SPDX-License-Identifier: MIT

import importlib.metadata
import warnings

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError as e:  # pragma: no cover
    warnings.warn(f"Could not determine version of {__name__}\n{e!s}", stacklevel=2)
    __version__ = "unknown"

from datarecord.duck import connect, layer_dir
from datarecord.layered.revision import Record, Revision
from datarecord.layered.write import write_record
from datarecord.mutable import Directory, NewChild, WorkingRecord
from datarecord.record import (
    Flags,
    Frames,
    LazyFrames,
    RecordLike,
)
from datarecord.schema import (
    AttributeSpec,
    Dimension,
    Group,
    Schema,
    Trait,
    TypeAttribute,
    TypeSpec,
)

__all__ = [
    "AttributeSpec",
    "Dimension",
    "Directory",
    "Flags",
    "Frames",
    "Group",
    "LazyFrames",
    "NewChild",
    "Record",
    "RecordLike",
    "Revision",
    "Schema",
    "Trait",
    "TypeAttribute",
    "TypeSpec",
    "WorkingRecord",
    "connect",
    "layer_dir",
    "write_record",
]
