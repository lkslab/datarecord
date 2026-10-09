# SPDX-FileCopyrightText: datarecord contributors
#
# SPDX-License-Identifier: MIT

"""Where a layer's rows come from, behind a protocol the fold reads them through.

The fold names files, not locations: `inputs/p_nom.parquet` is what it wants
and a source hands over its rows. Three answer - a parquet directory under
`layers/<uuid>/`, a plain directory read as one layer, and a staging area whose
rows are tables rather than files (`mutable.StagedSource`) - so the fold is
written once and knows about none of them.

Notes
-----
- [the record format](https://energy-models.github.io/datarecord/design/format/)
- [the DuckDB read path](https://energy-models.github.io/datarecord/design/read-path/)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID, uuid5

from datarecord.duck import (
    layer_dir,
    parquet_names,
    try_read_parquet,
)
from datarecord.layered.fold import Fold
from datarecord.record import Kind

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection, DuckDBPyRelation

    from datarecord.schema import Schema


@runtime_checkable
class LayerSource(Protocol):
    """One layer's own rows, however they are stored.

    "The layer as it would be written", not "the rows as stored": a source
    hands over what `write_record` would persist, so however a staging area
    reaches one row per key happens behind it and the fold never learns about it.
    `None` means this layer wrote nothing of that kind.

    Rows only. Everything the fold does to them - padding to the long schema,
    expanding broadcasts against an axis, the ownership aggregate, `order_key` -
    stays in the fold, which is the point: an implementation computing any of it
    would be a second copy of the fold.

    Structural, so an implementation needs no import from here: `mutable.py`'s
    `StagedSource` satisfies it by shape alone, and a source outside this
    package would too.

    Attributes
    ----------
    layer_id : UUID
        What the fold stamps as `layer_uuid`, and what a read dispatches a
        winning row back through.
    frozen : bool
        Whether these rows can change under a reader. `False` for a staging
        area, which is what stops the fold materialising past it.

    Notes
    -----
    - [the record format](https://energy-models.github.io/datarecord/design/format/)
    - [a layer's data is write-once](https://energy-models.github.io/datarecord/design/layers/#a-layers-data-is-write-once)
    """

    # Properties rather than variables, so a frozen dataclass satisfies this:
    # mypy reads a plain annotation here as a *settable* attribute, which a
    # frozen field is not.
    @property
    def layer_id(self) -> UUID: ...

    @property
    def frozen(self) -> bool: ...

    @property
    def schema(self) -> Schema:
        """The record's one manifest, passed in at construction, never re-read.

        A layer holds only data - its schema lives beside the tree, not in the
        layer - so this is the reference the constructing reader already held,
        carried so `write_record` can read it off the source it is handed. Every
        source in one fold carries the same one.

        Notes
        -----
        - [one schema per record](https://energy-models.github.io/datarecord/design/schema/#one-schema-per-record)
        """
        ...

    def materialised(self, con: DuckDBPyConnection, schema: Schema) -> Fold | None:
        """This layer's resolved `Fold` if it is materialised, else `None`.

        A filesystem fact, distinct from `frozen`: `frozen` says the rows cannot
        change under a reader, `materialised` says a `resolved/` cache exists on
        disk. The deepest materialised source in a fold's list is its base - the
        prior fold it starts from rather than re-folds. Only a `ParquetLayer`
        ever answers non-`None`; a directory or a staging area has no cache.
        """
        ...

    def axes(self) -> set[str]:
        """Which dims this layer has an axis file for.

        One listing rather than a probe per declared dim: `resolve_coords` asks
        each source once and folds only the dims some source holds.
        """
        ...

    def axis(self, dim: str) -> DuckDBPyRelation | None:
        """`dims/<dim>.parquet` - one axis's full row, not the key alone.

        `entity` is a dim like any other, so the entity axis is `axis("entity")`;
        what differs is only that the fold takes it as the components map,
        with `order_key` and tombstones, where `resolve_coords` folds the rest.
        """
        ...

    def entity_types(self) -> set[str]:
        """Which types this layer has a member file for.

        For `write_record`'s benefit, not the fold's: a read learns which types
        exist from the resolved entity axis, never by listing a source's files.
        A directory listing, like `axes()` - empty wherever no member file
        exists, which for a schema declaring no entity-type axis is always.
        """
        ...

    def entity_type(self, name: str) -> DuckDBPyRelation | None:
        """`dims/entity_type/<name>.parquet` - one type's wide member rows.

        A different thing from `axis("entity")`: read after the map named a
        winner, rather than folded to find one.
        """
        ...

    def groups(self) -> set[str]:
        """Which groups this layer has a row for, by file - like `axes()`.

        For `write_record`'s benefit; not scoped to a schema, which a
        `LayerSource` does not carry - a file some other source's declaration
        does not recognise is the reader's concern, not this listing's.
        """
        ...

    def group(self, name: str) -> DuckDBPyRelation | None:
        """`groups/<name>.parquet` - one group's rows, tombstones included."""
        ...

    def attributes(self, kind: Kind = "inputs") -> set[str]:
        """Which `<kind>/*.parquet` files this layer has, by name.

        For `write_record`'s benefit: a read learns owned attributes from the
        owner map, never by listing a source's own files.
        """
        ...

    def stored_map(self, kind: str = "inputs") -> DuckDBPyRelation | None:
        """`owner_map/<kind>.parquet` - the map `write_record` stored beside the rows.

        Only a standalone record directory carries one: its rows are written
        whole and renamed into place, so the map folded at write time is the
        record's resolved map forever. It answers for a fold of this source
        alone, and carries no `layer_uuid` - a directory's layer id derives
        from its location, which a move changes, so the reader stamps its own.
        `None` for a tree layer or a staging area, which are resolved by
        folding.

        Notes
        -----
        - [the stored owner map](https://energy-models.github.io/datarecord/design/read-path/#owner-map)
        """
        ...

    def attribute(self, name: str, kind: Kind = "inputs") -> DuckDBPyRelation | None:
        """`<kind>/<name>.parquet` - one attribute's own columns, unpadded.

        Not the singular of `all_attributes`: this is the *owned* read, and it
        must keep exactly the columns the file has, a padded one being ambiguous
        against the broadcast rule.

        Notes
        -----
        - [the broadcast rule](https://energy-models.github.io/datarecord/design/record/#the-broadcast-rule)
        """
        ...

    def all_attributes(self, kind: Kind = "inputs") -> DuckDBPyRelation | None:
        """Every `<kind>/*.parquet` unioned by name, unprojected.

        Only the long kinds have one: they share `input_key`, so a single scan
        answers the ownership `GROUP BY` for every attribute at once. An axis or
        a group folds on a key of its own, so a union across them would have
        nothing to fold on.
        """
        ...


class _FileLayer:
    """The layer layout over a URI, shared by every file-backed source.

    Each member is one file, addressed by what keys it. A subclass supplies
    where the layer root is (`uri`), the connection to read it with, and the
    schema its reader held, and inherits every accessor unchanged - which is
    what keeps "a directory read as a layer" from being a second reading of the
    format.
    """

    schema: Schema
    con: DuckDBPyConnection | None
    frozen: bool = True

    def uri(self, path: str = "") -> str:
        """Where `path` is, relative to the layer root.

        Empty is the root itself, with its trailing slash.
        """
        raise NotImplementedError

    @property
    def _con(self) -> DuckDBPyConnection:
        """The connection to read through.

        Optional on the field so `uri` alone is usable without one - which the
        write path does, having nothing to read.
        """
        if self.con is None:
            msg = f"{type(self).__name__} was built without a connection to read with"
            raise ValueError(msg)
        return self.con

    def _read(self, path: str, **kwargs: object) -> DuckDBPyRelation | None:
        return try_read_parquet(self.uri(path), self._con, **kwargs)

    def axes(self) -> set[str]:
        return {
            name.removesuffix(".parquet")
            for name in parquet_names(self.uri("dims/"), self._con)
        }

    def axis(self, dim: str) -> DuckDBPyRelation | None:
        return self._read(f"dims/{dim}.parquet", union_by_name=True)

    def entity_types(self) -> set[str]:
        return {
            name.removesuffix(".parquet")
            for name in parquet_names(self.uri("dims/entity_type/"), self._con)
        }

    def entity_type(self, name: str) -> DuckDBPyRelation | None:
        return self._read(f"dims/entity_type/{name}.parquet")

    def groups(self) -> set[str]:
        return {
            name.removesuffix(".parquet")
            for name in parquet_names(self.uri("groups/"), self._con)
        }

    def group(self, name: str) -> DuckDBPyRelation | None:
        return self._read(f"groups/{name}.parquet", union_by_name=True)

    def attributes(self, kind: Kind = "inputs") -> set[str]:
        return {
            name.removesuffix(".parquet")
            for name in parquet_names(self.uri(f"{kind}/"), self._con)
        }

    def attribute(self, name: str, kind: Kind = "inputs") -> DuckDBPyRelation | None:
        return self._read(f"{kind}/{name}.parquet")

    def all_attributes(self, kind: Kind = "inputs") -> DuckDBPyRelation | None:
        return self._read(f"{kind}/*.parquet", union_by_name=True)

    def stored_map(self, kind: str = "inputs") -> DuckDBPyRelation | None:
        return self._read(f"owner_map/{kind}.parquet")

    def materialised(self, con: DuckDBPyConnection, schema: Schema) -> Fold | None:
        """No cache by default: only a `ParquetLayer` has a revision to key one by."""
        return None


@dataclass(frozen=True)
class ParquetLayer(_FileLayer):
    """A layer as a parquet directory, located from its revision UUID.

    The location is derived, never stored, so changing the layout is a change
    to `layer_dir` and nothing else.

    Notes
    -----
    - [the record format](https://energy-models.github.io/datarecord/design/format/)
    """

    revision_id: UUID
    schema: Schema
    con: DuckDBPyConnection | None = None
    base_uri: str | None = None
    frozen: bool = True

    @property
    def layer_id(self) -> UUID:
        return self.revision_id

    def uri(self, path: str = "") -> str:
        return layer_dir(self.revision_id, self.base_uri) + path

    def materialised(self, con: DuckDBPyConnection, schema: Schema) -> Fold | None:
        """This node's resolved `Fold` if its `resolved/` cache exists, else `None`."""
        return Fold.read(self.revision_id, con, schema, self.base_uri)


@dataclass(frozen=True)
class DirectorySource(_FileLayer):
    """A plain parquet directory read as one layer.

    What `Record.at(uri)` folds over, and the whole of what reading a plain
    directory takes: the layout is the same as a tree layer's, so every member
    is one of the directory's files and the fold needs no conditional path.

    `layer_id` is derived rather than stored, as `ParquetLayer`'s location is
    and for the mirrored reason: a directory has no revision to be stamped
    with, but it has a location, and that is what makes it one layer rather
    than another. So two readings of one directory agree on which layer they
    read without being told, and a caller has nothing to allocate or keep.

    It is no node in a layer tree either way - there is nothing for
    `NewChild()` to branch from, so this makes such a record *readable* through
    the fold, not committable to a tree.

    Notes
    -----
    - [the record format](https://energy-models.github.io/datarecord/design/format/)
    """

    base: str
    schema: Schema
    con: DuckDBPyConnection | None = None
    frozen: bool = True

    def __post_init__(self) -> None:
        # A caller's directory URI, with or without the trailing slash every
        # member path is appended to. Normalised here rather than in `uri` so
        # `layer_id` sees it too: `/x` and `/x/` are one directory, and hashing
        # them apart would read one location as two layers.
        if not self.base.endswith("/"):
            object.__setattr__(self, "base", self.base + "/")

    @property
    def layer_id(self) -> UUID:
        return uuid5(_DIRECTORY_NAMESPACE, self.base)

    def uri(self, path: str = "") -> str:
        return self.base + path


# A fixed namespace, so one directory's `layer_id` is the same in every process
# - a random one would make it per-reader, which is what deriving it avoids.
_DIRECTORY_NAMESPACE = UUID("6f3d9f4e-1c2b-4a5d-8e7f-0a1b2c3d4e5f")
