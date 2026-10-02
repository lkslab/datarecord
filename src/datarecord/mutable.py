# SPDX-FileCopyrightText: datarecord contributors
#
# SPDX-License-Identifier: MIT

"""Editing a record: staged edits, materialised on commit.

What `Record` (read-only) and `write_record` (a whole record at once) do not
cover. Accumulate-then-commit: an edit costs a row in a staging table rather
than a rewrite, and nothing touches the record until `commit()`.

Notes
-----
- [WorkingRecord](https://energy-models.github.io/datarecord/design/working-record/)
"""

from __future__ import annotations

import re
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast, overload
from uuid import UUID, uuid4

import duckdb
import narwhals as nw
from duckdb import ColumnExpression as col
from duckdb import ConstantExpression as lit
from duckdb import DuckDBPyRelation, Expression
from duckdb import SQLExpression as sql

from datarecord.duck import (
    DuckTypes,
    as_relation,
    null_safe,
    union_all_by_name,
)
from datarecord.layered.fold import Fold
from datarecord.layered.resolve import Resolver
from datarecord.layered.revision import Record, Revision
from datarecord.layered.write import write_record
from datarecord.record import (
    EMPTY,
    Frames,
    LazyFrames,
    RecordLike,
)
from datarecord.schema import LONG_TAIL, Schema

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


# -- commit targets (https://energy-models.github.io/datarecord/design/working-record/#committing) ---------------------------------------------------


@dataclass(frozen=True)
class NewChild:
    """Write the staged rows as a new child layer of `record`.

    Only the edits are written; the fold resolves the rest from the parent.

    `record` defaults to the node the `WorkingRecord` was built over, which is
    what a caller branching from a revision means every time. Passing one
    explicitly is for the rarer case of re-parenting the edits elsewhere; a
    `WorkingRecord` over a base that is not a layered node (a directory, a
    framework object) has nothing to default to and must supply it.

    Notes
    -----
    - [committing](https://energy-models.github.io/datarecord/design/working-record/#committing)
    """

    record: Revision | None = None


@dataclass(frozen=True)
class Directory:
    """Write a standalone record at `uri`: staged rows *plus* what the record
    already reads, there being no parent to resolve against.

    Notes
    -----
    - [committing](https://energy-models.github.io/datarecord/design/working-record/#committing)
    """

    uri: str


Target = NewChild | Directory


# -- value normalisation (https://energy-models.github.io/datarecord/design/working-record/#set) ----------------------------------------------


def _incoming(frame: Any, con: DuckDBPyConnection) -> nw.LazyFrame:
    """A caller's frame as a lazy frame on `con`, whatever backend it arrived on.

    Every edit converts at this one point, so the steps behind it join and union
    in narwhals without minding where the frame came from.

    Notes
    -----
    - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
    """
    return nw.from_native(as_relation(nw.from_native(frame).lazy(), con)).lazy()


def _is_frame(value: Any) -> bool:
    """Whether `value` supplies its own keys rather than being a value.

    Notes
    -----
    - [set](https://energy-models.github.io/datarecord/design/working-record/#set)
    """
    if isinstance(value, nw.DataFrame | nw.LazyFrame):
        return True
    try:
        nw.from_native(value)
    except TypeError:
        return False
    return True


def _series_index(value: Any) -> Sequence[Any] | None:
    """`value`'s labels if it is a one-dimensional labelled series, else None."""
    index = getattr(value, "index", None)
    if index is None or getattr(value, "ndim", None) != 1:
        return None
    return list(index)


def _series_index_name(value: Any) -> str | None:
    """A labelled series' index name, where it has one a caller could have meant.

    `pd.Series(...).index.name` is the caller already saying what the index
    holds, so an `indexed_by=` repeating it is noise. A `MultiIndex` has `names`
    rather than one `name` and is no one-dimensional index, so it answers None
    and the caller says it explicitly.
    """
    index = getattr(value, "index", None)
    name = getattr(index, "name", None)
    return name if isinstance(name, str) else None


def normalise_value(
    value: Any,
    names: Sequence[str] | None,
    *,
    indexed_by: str | None = None,
) -> tuple[list[str] | None, list[Any], dict[str, list[Any]]]:
    """One of `set`'s four `value` forms as per-name values.

    Parameters
    ----------
    value
        Scalar, sequence, mapping, or a one-dimensional labelled series.
    names
        The names to broadcast or align to.
    indexed_by
        Which axis a labelled series' index holds, where the caller's `entity=`
        or `**dims` said so. `None` means the index holds entity names, which is
        the only other thing it can be.

    Returns
    -------
    names
        The names each value belongs to, or None to keep the caller's.
    values
        One value per name.
    per_dim
        Dim -> label, where a labelled series indexed by an axis was given.

    Raises
    ------
    ValueError
        If a sequence's length does not match `names`.

    Notes
    -----
    - [set](https://energy-models.github.io/datarecord/design/working-record/#set)
    """
    labels = _series_index(value)
    if labels is not None:
        # Told, never inferred: an axis label may be a string just like an entity
        # name, so testing membership would make one call mean different things
        # in two records (https://energy-models.github.io/datarecord/design/working-record/#set).
        if indexed_by is not None:
            return None, list(value), {indexed_by: list(labels)}
        return list(labels), list(value), {}

    if isinstance(value, Mapping):
        return list(value), list(value.values()), {}

    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        # A scalar: broadcast to every name.
        if names is None:
            return None, [value], {}
        return list(names), [value] * len(names), {}

    if names is None:
        msg = "a sequence needs `names` to align to"
        raise ValueError(msg)
    if len(value) != len(names):
        msg = (
            f"{len(value)} values for {len(names)} names; a length mismatch is an "
            f"error at the call rather than a truncated edit"
        )
        raise ValueError(msg)
    return list(names), list(value), {}


@dataclass(frozen=True)
class StagedSource:
    """A staging area as one layer's rows - the last source the fold reads.

    "The layer as it would be written": each member reads its staging table,
    which an edit replaces by key rather than appending to (`_replace`), so the
    table is already one row per key and what `write_record` would persist -
    the entity axis apart, whose tombstone anti-join reaches another file.

    Unfrozen, which is the whole of what distinguishes it: a `set` changes these
    rows under a reader, so the fold must stay a relation past this point rather
    than materialise. Nothing here needs invalidating in exchange - a relation
    over a staging table reads whatever the table holds when it is collected.

    Structural rather than inheriting `LayerSource`, so `mutable.py` keeps its
    rule of importing from `layered/` only inside function bodies.

    Notes
    -----
    - [reading with pending edits](https://energy-models.github.io/datarecord/design/working-record/#reading-with-pending-edits)
    - [committing](https://energy-models.github.io/datarecord/design/working-record/#committing)
    """

    record: WorkingRecord
    layer_id: UUID
    frozen: bool = False

    @property
    def schema(self) -> Schema:
        """This record's schema - the `LayerData` `write_record` reads for one.

        A staging area is not a `LayerSource` a fold folds under a schema
        someone else supplies; it is also what `write_record` writes directly
        for a `NewChild` commit, where it needs its own.
        """
        return self.record.schema

    def materialised(self, con: DuckDBPyConnection, schema: Schema) -> Fold | None:
        """A staging area has no `resolved/` cache: it is never a fold's base."""
        return None

    def axes(self) -> set[str]:
        """The dims with staged rows, `entity` among them.

        `entity` is an axis file like any other to a reader, so an `add` or a
        `remove` contributes to it - which is what puts a staged component in
        the components map.
        """
        return set(self.record._staged_dims())

    def axis(self, dim: str) -> DuckDBPyRelation | None:
        """One axis as this layer would write it - `_axis_layer`, exactly.

        Which matters for the columns rather than the labels: the fold is
        last-writer-wins per *label*, over the whole row, so a source handing
        over only the column its `set` named would blank every sibling
        attribute on that label. A `set` on an axis carries the siblings into
        the staged row when it patches it (`_patch_axis`), so the table is
        already the row `commit` writes.

        `entity` included, staged as an axis like the rest - what differs is
        only the extent `_axis_layer` gives it.

        Notes
        -----
        - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
        - [partial](https://energy-models.github.io/datarecord/design/schema/#partial-the-granularity-of-an-override)
        """
        if self.record._rows(f"{_AXIS_PREFIX}{dim}") is None:
            return None
        return self.record._axis_layer(dim)

    def entity_types(self) -> set[str]:
        """Which types have a staged member table."""
        return set(self.record._staged_types())

    def entity_type(self, name: str) -> DuckDBPyRelation | None:
        """One type's staged member rows, from that type's own table.

        Notes
        -----
        - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
        """
        return self.record._collapsed_members(name)

    def groups(self) -> set[str]:
        """Which declared groups have staged rows."""
        return set(self.record._staged_groups())

    def group(self, name: str) -> DuckDBPyRelation | None:
        return self.record._collapsed_group(name)

    def attributes(self, kind: str = "inputs") -> set[str]:
        """Which attributes of `kind` have staged rows."""
        return set(self.record._staged_attributes_of(kind))

    def attribute(self, name: str, kind: str = "inputs") -> DuckDBPyRelation | None:
        if kind == "outputs":
            # Results do not overlay, so the table is already what is written -
            # one row per coordinate (`_replace`) and no fold to apply.
            return self.record._rows("outputs", name)
        return self.record._collapsed_inputs(name)

    def all_attributes(self, kind: str = "inputs") -> DuckDBPyRelation | None:
        """Every staged attribute of `kind`, unioned by name and unprojected.

        By name because the tables carry per-attribute column variation exactly
        as the files do - one attribute's coordinates and no others - which is
        what lets `fold_inputs` pad both the same way.

        Notes
        -----
        - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
        """
        arms = [
            rel
            for name in self.record._staged_attributes_of(kind)
            if (rel := self.attribute(name, kind)) is not None
        ]
        if not arms:
            return None
        return union_all_by_name(arms, self.record.con)


# -- the DuckDB-backed implementation (https://energy-models.github.io/datarecord/design/working-record/#staging) ---------------------------------


class WorkingRecord(Record):
    """A `Record` whose last source is a staging area, plus the edit surface.

    A `Record` in the type as well as in the fold: what it reads is the data
    *with its pending edits applied*, and it reads it by being one layer deeper
    than its base rather than by overlaying anything of its own. Every read
    member is inherited unchanged - `outputs` alone is overridden, results not
    overlaying - so an edit reads back through the same code path a committed
    layer would.

    Staged rows live in connection-scoped DuckDB tables, the *only* place a
    staged row exists: the reads fold them rather than holding a copy, so what
    is staged is asked of the reads themselves.

    The `Resolver` is fixed at construction and the source list never changes;
    a `set` changes what the last source's tables hold, not which sources there
    are. That is what lets the inherited members stay correct - they cache a key
    set only where the fold is stable, which a staged source makes it not.

    Notes
    -----
    - [WorkingRecord](https://energy-models.github.io/datarecord/design/working-record/)
    - [staging](https://energy-models.github.io/datarecord/design/working-record/#staging)
    - [reading with pending edits](https://energy-models.github.io/datarecord/design/working-record/#reading-with-pending-edits)
    """

    #: This record's one identity, which is the staged layer's: the fold stamps
    #: it as `layer_uuid` and dispatches a winning row back through the source
    #: carrying it. Synthetic, the staged layer having no revision until
    #: `commit` writes one, and it names the staging tables too - so two
    #: `WorkingRecord`s on one connection collide in neither.
    _layer_id: UUID
    #: What the base resolves from, which every use of the base here wants: the
    #: schema, one dim's axis, one attribute's rows and the revision to branch
    #: from are all members of one.
    _base: Resolver
    #: Keyed by `(kind, attribute)`, the attribute being None for the entity
    #: kinds. A long kind stages one table per attribute because that is the
    #: file it stands for: one `value` column at the attribute's own type, and
    #: its own coordinates and no others.
    _staged: dict[tuple[str, str | None], str]

    def __init__(self, base: RecordLike, con: DuckDBPyConnection) -> None:
        # `object.__setattr__` throughout: the base is a frozen dataclass, and
        # these are set before `super().__init__` because the `StagedSource`
        # below reads them off `self`.
        object.__setattr__(self, "_layer_id", uuid4())
        object.__setattr__(self, "_staged", {})
        base_cache = _base_resolver(base, con)
        object.__setattr__(self, "_base", base_cache)
        # This record *is* the fold one layer deeper, and that layer is the
        # staging area - so the field the base class holds is that fold, whose
        # last source is the staged layer `commit` writes.
        super().__init__(base_cache.with_source(StagedSource(self, self._layer_id)))

    # -- staging tables -----------------------------------------------------

    def _table(self, kind: str, attribute: str | None = None) -> str:
        """A staging table's name, unique per record and per attribute.

        The attribute is hashed rather than spelled: it is a caller's string,
        and a table name is the one thing here that cannot be an expression, so
        it would be an injection and a quoting problem at once.
        """
        if kind.startswith(_AXIS_PREFIX):
            # Hashed for the same reason the attribute is: a dim name is a
            # caller's string, and this becomes an identifier.
            digest = sha256(kind[len(_AXIS_PREFIX) :].encode()).hexdigest()[:16]
            return f"staged_axis_{digest}_{self._layer_id.hex}"
        if attribute is None:
            return f"staged_{kind}_{self._layer_id.hex}"
        digest = sha256(attribute.encode()).hexdigest()[:16]
        return f"staged_{kind}_{digest}_{self._layer_id.hex}"

    def _ensure(self, kind: str, attribute: str | None = None) -> str:
        """The staging table for `kind`, created on first use.

        `kind` is one of the fixed three, or a declared group's name - a group
        gets a table shaped by its own coordinates.

        A long kind takes an `attribute` and gets a table per attribute, shaped
        like the file it becomes: `long_columns_for` for the columns, and the
        declared dtype for `value`. `_MEMBERS` takes an entity *type* in the
        same slot and is shaped the same way, one table per per-type file.

        Notes
        -----
        - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
        """
        key = (kind, attribute)
        if key in self._staged:
            return self._staged[key]
        name = self._table(kind, attribute)
        self._shape(kind, attribute).create(name)
        self._staged[key] = name
        return name

    def _shape(self, kind: str, attribute: str | None) -> DuckDBPyRelation:
        """A row-less relation shaped like the file `kind` becomes.

        The one place a staging table's columns are decided, so a new kind is a
        branch here rather than a second dispatch beside it.
        """
        if attribute is not None and kind != _MEMBERS:
            return self._empty_long(attribute)
        if kind == _MEMBERS:
            assert attribute is not None, "a member table is per entity type"
            columns = _member_columns(self.schema, attribute)
        elif kind.startswith(_AXIS_PREFIX):
            columns = _axis_columns(self.schema, kind[len(_AXIS_PREFIX) :])
        else:
            columns = _group_columns(self.schema, kind)
        return DuckTypes(self.con).empty_relation(**columns)

    def _empty_long(self, attribute: str) -> DuckDBPyRelation:
        """A row-less relation shaped like one attribute's long file.

        What the staging table is created from, so the table's shape is a
        projection rather than assembled DDL - the same expressions the inserts
        then project, which is what keeps the two from drifting.

        `value` takes the attribute's declared type, results being declared
        beside inputs. A name neither vocabulary holds falls back to a string,
        which is the widest thing a value column can be.

        Notes
        -----
        - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
        - [the shape of an edit](https://energy-models.github.io/datarecord/design/working-record/#the-shape-of-an-edit)
        """
        duck_types = DuckTypes(self.con)
        value_type = self.schema.value_type(attribute) or nw.String()
        types = {
            "attribute": nw.String(),
            "breakpoint": nw.Float64(),
            "value": value_type,
        }
        shaped = {
            c: types.get(c) or self._column_type(c)
            for c in self.schema.long_columns_for(attribute)
        }
        return duck_types.empty_relation(**shaped)

    def _rows(self, kind: str, attribute: str | None = None) -> DuckDBPyRelation | None:
        name = self._staged.get((kind, attribute))
        return None if name is None else self.con.table(name)

    def _staged_attributes_of(self, kind: str) -> tuple[str, ...]:
        """Which attributes `kind` has staged rows for, in insertion order.

        The staging map is the answer, so this is not a query: a table exists
        exactly where rows were staged.
        """
        return tuple(a for (k, a), _ in self._staged.items() if k == kind and a)

    def _column_type(self, column: str) -> nw.dtypes.DType:
        return _column_type(self.schema, column)

    def _staged_coordinates(self, attribute: str) -> tuple[str, ...]:
        """The dim columns one attribute's staging table has, in table order.

        `long_columns_for` minus the fixed tail, rather than `coordinates_of`:
        the two disagree for an *undeclared* attribute, where the first widens
        to every declared dim and the second answers none. The table is built
        from the first, so an insert deriving its columns from the second would
        supply too few - which is the shape a result arrives in.

        Notes
        -----
        - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
        """
        return tuple(
            c for c in self.schema.long_columns_for(attribute) if c not in LONG_TAIL
        )

    # -- Record, one fold deeper (https://energy-models.github.io/datarecord/design/working-record/#reading-with-pending-edits) --------------------------------

    # `schema`, `dims`, `entity_types`, `groups`, `attributes` and `flags` are
    # inherited from `Record` unchanged, which is the property this design
    # exists to have: a staged edit is read by the same fold that reads a
    # committed layer, so there is no second overlay to keep in step. Only
    # `outputs` below differs, and only because results do not overlay.

    def _owned_whole(self, attribute: str) -> tuple[str, ...]:
        """`AttributeSpec.dims` minus the fold key - the value axes owned whole.

        The complement of `owned_per`: a dim the attribute varies over but that is
        not in `partial_dims` (a non-`partial` value axis like `timestep`) is
        owned whole, so a patch to one value restates the attribute's whole extent
        along it. Membership keys are in `partial_dims`, so never here - a layer
        patches one component's or connection's value, never restating the rest.

        Notes
        -----
        - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
        - [one fold for every axis](https://energy-models.github.io/datarecord/design/read-path/#one-fold-for-every-axis)
        """
        spec = self.schema.attributes.get(attribute)
        whole = (
            frozenset() if spec is None else spec.dims - set(self.schema.partial_dims)
        )
        return tuple(d for d in self.schema.dims if d in whole)

    @property
    def outputs(self) -> Frames:
        """Staged results, keyed by attribute - what a tool handed back.

        Results reach a record through `set(..., kind="outputs")`, so a tool can
        solve against this record's pending inputs and attach what it computed
        without committing first. The base's results are *not* included: they
        were computed from inputs these edits may have changed, and results do
        not overlay, so what is staged is the whole answer.

        Keeping them coherent with the inputs is the caller's business - editing
        an input after attaching results leaves results describing a record that
        no longer exists, and nothing here silently discards them.

        Notes
        -----
        - [outputs](https://energy-models.github.io/datarecord/design/read-path/#outputs)
        - [set](https://energy-models.github.io/datarecord/design/working-record/#set)
        """
        names = tuple(sorted(self._staged_attributes_of("outputs")))
        if not names:
            return EMPTY

        def frame(attr: str) -> nw.LazyFrame:
            rel = cast("DuckDBPyRelation", self._rows("outputs", attr))
            # The table is already the shape of the file: results do not overlay,
            # so there is nothing to collapse them against (https://energy-models.github.io/datarecord/design/read-path/#outputs).
            return nw.from_native(rel)

        return LazyFrames(names, frame)

    # -- edits (https://energy-models.github.io/datarecord/design/working-record/#set, https://energy-models.github.io/datarecord/design/working-record/#an-nwexpr-value-derived-from-the-current-one, https://energy-models.github.io/datarecord/design/working-record/#add-remove) ----------------------------------------

    def _series_axis(
        self, attribute: str, value: Any, indexed_by: str | None
    ) -> str | None:
        """Which axis a labelled series' index holds, or None for entity names.

        The caller says which - `indexed_by="snapshot"`, or the series' own
        `index.name` where it names a coordinate of this attribute. Never read
        off the labels: an axis label may be a string just like an entity name,
        so a membership test would make one call mean different things in two
        records.

        An `index.name` naming no coordinate is ignored rather than rejected; it
        may be `None`, or a pandas artefact like `"index"`.

        Notes
        -----
        - [set](https://energy-models.github.io/datarecord/design/working-record/#set)
        """
        if _series_index(value) is None:
            if indexed_by is not None:
                msg = (
                    f"`set({attribute!r}, ..., indexed_by={indexed_by!r})` says what "
                    f"a series index holds, but the value is not a labelled series"
                )
                raise ValueError(msg)
            return None
        named = indexed_by if indexed_by is not None else _series_index_name(value)
        if named is None:
            return None
        coordinates = [
            c for c in self.schema.coordinates_of(attribute) if c != "entity"
        ]
        if named not in coordinates:
            if indexed_by is None:
                return None
            msg = (
                f"`indexed_by={named!r}` is no coordinate of {attribute!r}, "
                f"which is addressed by {coordinates or ['entity']}"
            )
            raise ValueError(msg)
        return named

    def _axis_of(self, attribute: str) -> str | None:
        """The dim whose axis file carries `attribute`, or `None`.

        A declared attribute addressed by one dim alone is a column of that
        dim's axis file rather than a long row, so an edit to it stages an axis
        row. `attributes_on` is the rule: `entity` answers with its
        sole-coordinate attributes where no group declares the type axis - they
        are columns of `dims/entity.parquet` then - and answers None where one
        does, those being per-type member columns `add` stages instead. An
        undeclared attribute is never one either - only a result is undeclared,
        and a result is always long.

        Notes
        -----
        - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
        """
        spec = self.schema.attributes.get(attribute)
        if spec is None or spec.varying:
            return None
        (dim,) = spec.dims
        return dim if attribute in self.schema.attributes_on(dim) else None

    def _stage_axis(
        self,
        dim: str,
        attribute: str,
        value: Any,
        *,
        entity: Sequence[str] | None,
    ) -> None:
        """Stage one axis-file attribute, keyed by the axis's own labels.

        `value` is a mapping from label to value, or a scalar for every label the
        axis currently has. A mapping may name a label no layer has written yet,
        which becomes a row of this layer's axis file - the fold keys per label,
        so introducing one displaces nothing.

        One *complete* row per label: this edit's column over the label's
        current row - the one already staged where there is one, else the base's
        - so a second `set` on the same axis replaces the row rather than adding
        beside it and losing the first's column.

        Notes
        -----
        - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
        - [set](https://energy-models.github.io/datarecord/design/working-record/#set)
        """
        # `entity=` names labels of *this* axis where `dim` is `entity` - an
        # untyped record's component columns live here, so `set(attr, v,
        # entity=[...])` selects the rows to patch, the same names a member-file
        # edit would (https://energy-models.github.io/datarecord/design/format/#where-a-value-lives). For any other axis `entity` is not a
        # coordinate, so naming it is the caller error the message describes.
        if entity is not None and dim != "entity":
            msg = (
                f"`set({attribute!r}, ..., entity=...)` names components, but "
                f"{attribute!r} is addressed by {dim!r} alone - it is a column of "
                f"dims/{dim}.parquet and belongs to no component"
            )
            raise ValueError(msg)
        if entity is not None and not isinstance(value, Mapping):
            # A scalar broadcast to the named labels, or a sequence aligned to
            # them - the same two forms `set` takes for a long attribute, folded
            # to the axis's label->value mapping the rest of this method wants.
            names = list(entity)
            vals = (
                list(value)
                if isinstance(value, (list, tuple))
                else [value] * len(names)
            )
            if len(vals) != len(names):
                msg = (
                    f"`set({attribute!r}, <sequence>, entity=...)` has "
                    f"{len(vals)} values for {len(names)} names"
                )
                raise ValueError(msg)
            value = dict(zip(names, vals, strict=True))
        elif _is_frame(value) or isinstance(value, (list, tuple)):
            msg = (
                f"`set({attribute!r}, <sequence>)` has no labels to align to; "
                f"{attribute!r} is addressed by {dim!r} alone, so pass a mapping "
                f"from {dim!r} label to value, or a scalar for every label"
            )
            raise ValueError(msg)
        if len(self.schema.axis_key(dim)) > 1:
            msg = (
                f"{attribute!r} is addressed by {dim!r}, which is `within` "
                f"{sorted(self.schema.dimensions[dim].within)}; a nested axis's "
                f"labels identify a point only within its parents, which a "
                f"mapping from label alone cannot name"
            )
            raise ValueError(msg)

        table = self._ensure(f"{_AXIS_PREFIX}{dim}", None)
        if isinstance(value, Mapping):
            if not value:
                return
            edit = self._values_relation(
                {dim: list(value), attribute: list(value.values())},
                {
                    dim: self._column_type(dim),
                    attribute: self.schema.value_type(attribute),
                },
            )
        else:
            base_axis = self._base.dims.axes.get(dim)
            if base_axis is None:
                msg = (
                    f"`set({attribute!r}, <scalar>)` reaches every label the "
                    f"{dim!r} axis has, and it has none; name the labels as a "
                    f"mapping, or write the axis file first"
                )
                raise ValueError(msg)
            edit = base_axis.project(col(dim), lit(value).alias(attribute))

        self._patch_axis(dim, attribute, table, edit)

    def _patch_axis(
        self, dim: str, attribute: str, table: str, edit: DuckDBPyRelation
    ) -> None:
        """Set `attribute` on each label `edit` names, in place.

        An axis row's columns are independently editable, so this patches the one
        column rather than replacing the row - a sibling a `set` did not name is
        never read and so cannot be lost. A label falls to one of two statements
        by whether a prior `set` already staged it:

        - **INSERT** a label not yet staged, taking its siblings from the base row
          where the base has one and the edited column from `edit`. Its siblings
          have to travel with it: the fold is whole-row last-writer-wins per label
          (`fold_axis`), so a staged patch carrying only its own column would win
          the label and blank the rest - the resolved row is the layer's, not one
          the read rebuilds column by column. A label the base also lacks is new,
          and the edit is the whole of its row.
        - **UPDATE** a label already staged, patching the one column in place so
          the siblings an earlier edit carried stay put.

        `partial` decides only the label *extent* the layer carries, which is
        `_axis_layer`'s business, not this.

        Notes
        -----
        - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
        - [partial](https://energy-models.github.io/datarecord/design/schema/#partial-the-granularity-of-an-override)
        - [set](https://energy-models.github.io/datarecord/design/working-record/#set)
        """
        staged = self.con.table(table)
        base = self._base.dims.axes.get(dim)
        # Raw SQL because `UPDATE ... FROM` is a join-update, which the relational
        # `update` cannot express (https://duckdb.org/docs/stable/clients/python/relational_api).
        self.con.execute(
            f"UPDATE {table} AS t SET {col(attribute)} = {col('e', attribute)} "
            f"FROM edit e "
            f"WHERE {col('t', dim)} IS NOT DISTINCT FROM {col('e', dim)}"
        )
        fresh = edit.set_alias("e").join(
            staged.set_alias("s"), null_safe("e", "s", [dim]), how="anti"
        )
        if base is None:
            rows = fresh.set_alias("e").select(col(dim), col(attribute))
        else:
            # A left join, since a fresh label the base also lacks has no row.
            siblings = [c for c in base.columns if c not in (dim, attribute)]
            rows = (
                fresh.set_alias("e")
                .join(base.set_alias("b"), null_safe("e", "b", [dim]), how="left")
                .select(
                    col("e", dim), col("e", attribute), *(col("b", c) for c in siblings)
                )
            )
        self._insert(rows, table, {})

    def _resolved_names(self, ctype: str) -> list[str]:
        """Every name `ctype` currently resolves to, base plus staged.

        Off the components map rather than the member frame, for the reason
        `_name_types` gives: membership is what the map decided, and resolving
        the type's wide rows to read one column of it is the expensive way to
        ask. `str` because a backend's column yields its own scalar type (a
        `pyarrow.StringScalar`, say), which would compare unequal to the plain
        strings an edit names.

        Notes
        -----
        - [reading with pending edits](https://energy-models.github.io/datarecord/design/working-record/#reading-with-pending-edits)
        - [the owner map](https://energy-models.github.io/datarecord/design/read-path/#owner-map)
        """
        axis = self.resolver.entity_axis
        if axis is None:
            return []
        rows = (
            axis.filter(col("entity_type") == lit(ctype)).project("entity").fetchall()
        )
        return [str(n) for (n,) in rows]

    def _name_types(self) -> nw.LazyFrame | None:
        """`(name, entity_type)` over everything this record resolves.

        Straight off the resolved entity axis, which is keyed by entity and
        carries the type: what every validating caller here asks is "what type is
        this name", and that is the axis's own column. Assembling it from the
        member frames instead would resolve each type's wide rows - a union and a
        join per type - to read two columns the fold already decided.

        `None` where the schema declares no type axis: there is no `entity_type`
        column then, no name has a type, and every caller here reads that as "no
        cross-type question to ask" (https://energy-models.github.io/datarecord/design/format/#where-a-value-lives).

        Notes
        -----
        - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
        - [one fold for every axis](https://energy-models.github.io/datarecord/design/read-path/#one-fold-for-every-axis)
        """
        axis = self.resolver.entity_axis
        if axis is None or self.schema.entity_type_dim is None:
            return None
        rel = axis.project("entity", "entity_type")
        if rel.limit(1).fetchone() is None:
            return None
        return nw.from_native(rel).lazy()

    def _require_unique(self, ctype: str, lazy: nw.LazyFrame) -> None:
        """Reject an `add` whose names another type already holds.

        Re-adding a name of the *same* type is an edit to that member, which
        the fold resolves last-writer-wins - so only a cross-type clash
        raises.

        Notes
        -----
        - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
        - [validation](https://energy-models.github.io/datarecord/design/working-record/#validation)
        """
        known = self._name_types()
        if known is None:
            return
        clashing = (
            lazy.select("entity")
            .unique("entity")
            .join(
                known.filter(nw.col("entity_type") != ctype),
                on="entity",
                how="inner",
            )
            .unique(["entity", "entity_type"])
            .select("entity", "entity_type")  # the order `iter_rows` unpacks
            .collect()
        )
        if not clashing.is_empty():
            # Sorted here rather than in the query, as `collision_detail` is:
            # the message must be deterministic, and this is a handful of rows.
            detail = ", ".join(
                f"{n!r} is already a {t}" for n, t in sorted(clashing.iter_rows())
            )
            msg = (
                f"cannot add {ctype} components whose names are taken: {detail}; "
                f"names are unique across every component type (https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)"
            )
            raise ValueError(msg)

    def _resolve_types(self, names: Sequence[str]) -> dict[str, str]:
        """`names` mapped to their types, rejecting any the record does not resolve.

        A value keyed to a name with no member row would resolve to nothing, so
        it is caught here rather than dropped at read time.

        Notes
        -----
        - [validation](https://energy-models.github.io/datarecord/design/working-record/#validation)
        """
        wanted = list(dict.fromkeys(names))
        known = self._name_types()
        if known is None or not wanted:
            found: dict[str, str] = {}
        else:
            matched = (
                known.filter(nw.col("entity").is_in(wanted))
                .select("entity", "entity_type")
                .collect()
            )
            found = {str(n): str(t) for n, t in matched.iter_rows()}
        unknown = sorted({n for n in wanted if n not in found})
        if unknown:
            msg = (
                f"no member row for {unknown}; `add` them first - a value for a "
                f"name no layer declares would resolve to nothing"
            )
            raise KeyError(msg)
        return {n: found[n] for n in names}

    def _validate_dims(self, dims: Mapping[str, Any]) -> None:
        """The dim vocabulary, checked for either `kind`.

        Notes
        -----
        - [results through kind="outputs"](https://energy-models.github.io/datarecord/design/working-record/#results-through-kindoutputs)
        - [validation](https://energy-models.github.io/datarecord/design/working-record/#validation)
        """
        unknown = sorted(set(dims) - set(self.schema.dims))
        if unknown:
            msg = f"the schema declares no dims {unknown}"
            raise KeyError(msg)

    def _validate_result(self, attribute: str, dims: Mapping[str, Any]) -> None:
        """A result's name and dims, against the schema's `results`.

        The attribute check only - not membership, which stays relaxed for a
        result: a solve may produce rows for a component type it derived rather
        than read, and rejecting those would refuse a legitimate result.

        Notes
        -----
        - [results through kind="outputs"](https://energy-models.github.io/datarecord/design/working-record/#results-through-kindoutputs)
        - [validation](https://energy-models.github.io/datarecord/design/working-record/#validation)
        """
        self._validate_dims(dims)
        spec = self.schema.results.get(attribute)
        if spec is None:
            known = sorted(self.schema.results)
            msg = (
                f"the schema declares no result {attribute!r}; it declares "
                f"{known or 'none'}. A result is declared like an input, so a "
                f"tool states its vocabulary before attaching what it computed"
            )
            raise KeyError(msg)
        outside = sorted(set(dims) - spec.dims)
        if outside:
            msg = (
                f"result {attribute!r} does not vary over {outside}; "
                f"it varies over {sorted(spec.dims) or 'nothing'}"
            )
            raise ValueError(msg)

    def _validate_attribute(
        self,
        ctype: str,
        attribute: str,
        dims: Mapping[str, Any],
        *,
        name: str | None = None,
    ) -> None:
        """One name's attribute checks, against the spec of *its* type.

        Inputs only: a result is declared in `results` rather than per type, so
        `_validate_result` checks its name and dims and nothing checks its
        membership - what a solve computes need not be a component the record
        declares.

        `name` is reported where known: with the type derived rather than passed, the name is what the caller can act on.

        Notes
        -----
        - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
        - [outputs](https://energy-models.github.io/datarecord/design/read-path/#outputs)
        - [validation](https://energy-models.github.io/datarecord/design/working-record/#validation)
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
        """
        who = f" (for {name!r})" if name is not None else ""
        # The type's existence and its vocabulary are separate questions once
        # attributes are declared record-wide: `entity_types` answers the
        # first, `attributes_for` the second, and a type carrying nothing is
        # not the same as a type the schema never declared. An empty
        # `entity_types` is no vocabulary rather than an empty one - the axis
        # is undeclared or a plain string - so there is nothing to check
        # against and any label passes.
        known = self.schema.entity_types
        if known and ctype not in known:
            msg = f"the schema declares no entity type {ctype!r}{who}"
            raise KeyError(msg)
        declared = self.schema.attributes_for(ctype)
        if attribute not in declared:
            msg = f"{ctype} does not carry {attribute!r}{who}"
            raise KeyError(msg)
        spec = declared[attribute]
        outside = sorted(set(dims) - spec.dims)
        if outside:
            msg = (
                f"{ctype}.{attribute} does not vary over {outside}{who}; "
                f"it varies over {sorted(spec.dims) or 'nothing'}"
            )
            raise ValueError(msg)

    def set(
        self,
        attribute: str,
        value: Any,
        *,
        entity: Sequence[str] | None = None,
        kind: Literal["inputs", "outputs"] = "inputs",
        indexed_by: str | None = None,
        **dims: Any,
    ) -> None:
        """Stage an attribute value for a group of components.

        A labelled series must say what its index holds: `indexed_by="snapshot"`
        names the axis, and `entity=` implies it where the attribute has exactly
        one other coordinate. Without either the index is read as entity names.
        Never inferred from the labels themselves - an axis label may be a string
        just like a name, so that would make one call mean different things in
        two records.

        `value` takes five forms: a scalar broadcast to every name, a sequence
        aligned positionally to `names`, a mapping keyed by name, a long frame
        supplying its own keys, and a narwhals expression - which is a *function
        of the current value* rather than a value, so it reads before it stages
        and two such calls compose.

        No `entity_type` parameter: the type is looked up from the entity,
        so one call may span types and each is validated against its own
        type's spec. `entity=None` means every component whose type
        declares `attribute`.

        Every other coordinate goes through `**dims`, including a group's -
        `bus="north"` for a connection attribute, `from=`/`to=` for a corridor.
        None has a parameter of its own, since which coordinates exist is
        declared rather than fixed.

        `kind` names the destination in the format's own terms:
        `"outputs"` stages into `outputs/` instead of `inputs/`, which is how a
        tool hands results back. Results use the same long schema; what differs
        is that they do not overlay.

        Two checks are skipped for `"outputs"`, both because a result is not a
        value the schema governs: the attribute need not be declared, and a
        result's `name` need not resolve to a declared member. A solve may
        produce rows for a component type it derived rather than read - PyPSA's
        `SubNetwork` is one - and rejecting those would refuse a legitimate
        result. An *input* for an undeclared name stays an error.

        Notes
        -----
        - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
        - [outputs](https://energy-models.github.io/datarecord/design/read-path/#outputs)
        - [the shape of an edit](https://energy-models.github.io/datarecord/design/working-record/#the-shape-of-an-edit)
        - [set](https://energy-models.github.io/datarecord/design/working-record/#set)
        - [a derived value](https://energy-models.github.io/datarecord/design/working-record/#an-nwexpr-value-derived-from-the-current-one)
        - [validation](https://energy-models.github.io/datarecord/design/working-record/#validation)
        """
        is_long_frame = _is_frame(value) and _series_index(value) is None
        if is_long_frame:
            lazy = _incoming(value, self.con)
            if "entity_type" in lazy.collect_schema().names():
                msg = (
                    f"`set({attribute!r}, <frame>)` was given a `entity_type` "
                    f"column; names are unique across every type, so an attribute row "
                    f"carries no type and the column would be ignored (https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)"
                )
                raise ValueError(msg)
            if kind == "inputs":
                self._validate_frame(lazy, attribute, dims)
            else:
                self._validate_result(attribute, dims)
            self._stage_long(attribute, lazy, kind, dims)
            return

        if isinstance(value, nw.Expr):
            if kind == "inputs":
                self._validate_dims(dims)
            else:
                self._validate_result(attribute, dims)
            self._stage_derived(attribute, value, entity=entity, kind=kind, **dims)
            return

        axis = self._axis_of(attribute) if kind == "inputs" else None
        if axis is not None:
            self._stage_axis(axis, attribute, value, entity=entity)
            return

        target = (
            list(entity) if entity is not None else self._names_declaring(attribute)
        )
        keys, values, per_dim = normalise_value(
            value,
            target,
            indexed_by=self._series_axis(attribute, value, indexed_by),
        )
        if keys is None:
            keys = target
            if len(values) == 1 and len(keys) > 1:
                values = values * len(keys)
        if kind == "inputs":
            self._validate_dims(dims)
            # One lookup serves both: rejects a name with no member row, and
            # returns the type whose spec is checked (https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types).
            for name, ctype in self._resolve_types(keys).items():
                self._validate_attribute(ctype, attribute, dims, name=name)
        else:
            self._validate_result(attribute, dims)

        table = self._ensure(kind, attribute)
        self._stage_rows(attribute, table, keys, values, per_dim, dims)

    def _names_declaring(self, attribute: str) -> list[str]:
        """Every resolved name whose type declares `attribute` - `names=None`.

        The types come from the record rather than from `types_declaring` where
        the schema declares no entity-type labels: the axis is then a plain
        string and its labels are data, so what types exist is a question only
        the resolved components can answer.

        Notes
        -----
        - [set](https://energy-models.github.io/datarecord/design/working-record/#set)
        """
        declared = self.schema.types_declaring(attribute)
        if not self.schema.entity_types:
            declared = frozenset(
                c
                for c in self.entity_types
                if attribute in self.schema.attributes_for(c)
            )
        return [
            name
            for ctype in sorted(declared)
            if ctype in self.entity_types
            for name in self._resolved_names(ctype)
        ]

    def _validate_frame(
        self, lazy: nw.LazyFrame, attribute: str, dims: Mapping[str, Any]
    ) -> None:
        """A long input frame's dims, names and per-name specs.

        The frame supplies its own names, so each is resolved to its type and
        checked against that type's spec - one frame may legitimately span
        types, since names are unique.

        Notes
        -----
        - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
        - [validation](https://energy-models.github.io/datarecord/design/working-record/#validation)
        """
        self._validate_dims(dims)
        if "entity" not in lazy.collect_schema().names():
            return
        names = [
            str(n)
            for n in lazy.select("entity")
            .unique("entity")
            .collect()["entity"]
            .to_list()
        ]
        for name, ctype in self._resolve_types(names).items():
            self._validate_attribute(ctype, attribute, dims, name=name)

    def _stage_rows(
        self,
        attribute: str,
        table: str,
        keys: list[str],
        values: list[Any],
        per_dim: dict[str, list[Any]],
        dims: Mapping[str, Any],
    ) -> None:
        """Stage the scalar/sequence/mapping forms, as a relation rather than rows.

        The rows are `keys` x `per_dim`'s labels - every named entity at every
        coordinate the value covers - so a per-snapshot series over a year for a
        thousand components is millions of them. Both factors are small, and only
        their product is not, so each becomes a one-column relation and DuckDB
        joins them: the product never exists as Python objects.

        `keys` and `values` are positionally aligned where there is no `per_dim`;
        with one, `values` aligns to its labels and every key takes all of them.

        Notes
        -----
        - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
        - [set](https://energy-models.github.io/datarecord/design/working-record/#set)
        """
        entity = {"entity": self._column_type("entity")}
        # `value` is inferred: an undeclared result has no declared dtype, and
        # the staging table already took its type from `_scalar_dtype`.
        if per_dim:
            (dim, labels) = next(iter(per_dim.items()))
            # The value belongs to the label, so it rides on that side of the
            # join and every entity picks it up.
            rel = self._values_relation({"entity": keys}, entity).cross(
                self._values_relation(
                    {dim: labels, "value": values},
                    {dim: self._column_type(dim), "value": None},
                )
            )
            present = {"entity", dim}
        else:
            rel = self._values_relation(
                {"entity": keys, "value": values}, {**entity, "value": None}
            )
            present = {"entity"}

        self._insert_long(rel, table, attribute, present, dims)

    def _insert(
        self,
        rel: DuckDBPyRelation,
        table: str,
        supplied: Mapping[str, Expression],
        key: Sequence[str] | None = None,
    ) -> None:
        """Project `rel` into `table`'s column order and insert it.

        `insert_into` is positional, and a staging table's order is its own -
        `ALTER TABLE` appends each extra column as `add` first sees one - so the
        projection is built from the table rather than from the caller. A column
        `supplied` does not name is taken from `rel` where it carries one, and is
        otherwise a NULL typed from the table, which is what spares the insert a
        coercion.

        `key` given, the rows `rel` names by it are deleted first, so the table
        holds one row per key and *is* the file it becomes - no fold on the way
        out. The match is `null_safe` so a broadcast coordinate's NULL replaces
        the same NULL rather than sitting beside it (https://energy-models.github.io/datarecord/design/record/#the-broadcast-rule).
        `rel` is a whole-row edit built from the caller's frame, never from the
        staged table, so the delete cannot change what the insert then reads -
        the one path that patches columns in place (`_patch_axis`) does not pass
        a `key`.

        Notes
        -----
        - [committing](https://energy-models.github.io/datarecord/design/working-record/#committing)
        """
        staging = self.con.table(table)
        carried = {c.lower() for c in rel.columns}

        def column(name: str, dtype: str) -> Expression:
            if name in supplied:
                return supplied[name].alias(name)
            if name.lower() in carried:
                return col(name)
            return lit(None).cast(dtype).alias(name)

        projected = rel.project(
            *(
                column(c, str(t))
                for c, t in zip(staging.columns, staging.types, strict=True)
            )
        )
        if key is not None:
            # `null_safe`, not `IN`/`=`: those never match a broadcast row's NULL.
            on = null_safe(table, "p", key)
            self.con.execute(f"DELETE FROM {table} USING projected p WHERE {on}")
        try:
            projected.insert_into(table)
        except duckdb.ConversionException as exc:
            # An `Enum` dim reaches DuckDB as one, so this is what rejects a
            # label its vocabulary does not hold - as a conversion to `UINT8`,
            # the enum's storage type, naming neither the dim nor what it
            # declares. The source column is in the message, which is what makes
            # it restatable.
            match = re.search(r"casting from source column (\w+)", str(exc))
            if match is None:
                raise
            source = match.group(1)
            dim = self.schema.dimensions.get(source)
            if dim is None or not isinstance(dim.dtype, nw.Enum):
                raise
            msg = (
                f"{source!r} declares no such label; its dtype is an Enum over "
                f"{sorted(dim.dtype.categories)}, which pins the vocabulary"
            )
            raise ValueError(msg) from exc

    def _insert_long(
        self,
        rel: DuckDBPyRelation,
        table: str,
        attribute: str,
        present: Container[str],
        dims: Mapping[str, Any],
    ) -> None:
        """Insert `rel` as one attribute's long rows.

        `present` names the coordinates `rel` already carries; the rest come from
        `dims` where the caller scoped them and are otherwise NULL - "every value
        of it" by the broadcast rule - typed as the schema declares them either
        way, so the insert needs no coercion.

        Only this attribute's own coordinates: the staging table is the shape of
        the file it becomes, so a dim it is not addressed by has no column to
        fill, and the projection is in that table's order.

        Notes
        -----
        - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
        - [the broadcast rule](https://energy-models.github.io/datarecord/design/record/#the-broadcast-rule)
        """
        duck_types = DuckTypes(rel)
        shaped = rel.project(
            *(
                col(d)
                if d in present
                else duck_types.lit(dims.get(d), self._column_type(d)).alias(d)
                for d in self._staged_coordinates(attribute)
            ),
            lit(attribute).alias("attribute"),
            duck_types.null(nw.Float64()).alias("breakpoint"),
            col("value"),
        )
        self._insert(shaped, table, {}, key=self._long_key(attribute))
        self._complete_owned_whole(attribute, table)

    def _long_key(self, attribute: str) -> tuple[str, ...]:
        """The coordinate columns one long table is keyed on - what an edit replaces.

        The fold partitions on these too (`_collapsed_inputs` before this change),
        so replacing them per edit removes exactly what it would have discarded.

        Notes
        -----
        - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
        - [partial](https://energy-models.github.io/datarecord/design/schema/#partial-the-granularity-of-an-override)
        """
        columns = set(self.schema.long_columns_for(attribute))
        return tuple(
            c
            for c in (*self.schema.input_key, *self.schema.dims, "breakpoint")
            if c in columns
        )

    def _complete_owned_whole(self, attribute: str, table: str) -> None:
        """Carry the base extent a non-partial axis obliges the staged rows to hold.

        Done as the rows are staged rather than at commit, so the staging table
        *is* the layer: touching one snapshot of a series makes
        this layer the owner of that key's whole extent along the dim, and the
        untouched coordinates have to be carried or the commit would report a
        loss.

        Scoped by the keys already staged, not by the attribute: the semi-join
        below reaches only the keys some edit named, so a component this record
        never touched stays in the parent.

        Idempotent, which is what lets it run per insert instead of once: the
        anti-join drops any coordinate the table already holds, so a second edit
        to the same attribute carries only what the first did not. That
        anti-join is also the whole of what keeps a fill off an edited
        coordinate - it keys on `_long_key`, the same coordinate a later `set`
        deletes on (`_replace`), so the fill it excludes and the row that
        replaces one are one key, and no ordering column is needed to rank them.

        Notes
        -----
        - [partial](https://energy-models.github.io/datarecord/design/schema/#partial-the-granularity-of-an-override)
        - [committing](https://energy-models.github.io/datarecord/design/working-record/#committing)
        """
        whole = self._owned_whole(attribute)
        if not whole or attribute not in self._base.attributes():
            return

        # Intersected with the attribute's own columns: a key column its file
        # does not carry is absent from both sides rather than NULL in them,
        # and joining on it would fail to bind (`long_columns_for`).
        columns = set(self.schema.long_columns_for(attribute))
        scope = [c for c in self.schema.input_key if c not in whole and c in columns]
        present = [d for d in whole if d in columns]
        if not present:
            # No column for any whole-owned dim, so the file holds one row per
            # key and there is no extent to complete.
            return
        coordinate = self._long_key(attribute)
        staged = self.con.table(table)
        base = self._base.attribute(attribute)
        # A staged row leaving a whole-owned dim NULL already covers that dim's
        # whole extent by the broadcast rule, so its key has nothing left to
        # carry and a base row there would overlap it.
        broadcast = staged.filter(
            sql(" OR ".join(f"{col(d)} IS NULL" for d in present))
        )
        carried = (
            base.set_alias("b")
            # The keys some edit touched, minus those a broadcast already covers.
            .join(staged.set_alias("s"), null_safe("b", "s", scope), how="semi")
            .set_alias("b")
            .join(broadcast.set_alias("s"), null_safe("b", "s", scope), how="anti")
            # Then away the coordinates already staged, leaving the rest of the
            # extent the layer now owns whole and so must carry.
            .set_alias("b")
            .join(staged.set_alias("s"), null_safe("b", "s", coordinate), how="anti")
        )
        self._insert(carried, table, {})

    def _values_relation(
        self,
        columns: Mapping[str, Sequence[Any]],
        schema: Mapping[str, nw.dtypes.DType | None],
    ) -> DuckDBPyRelation:
        """A relation over `columns`, each an equal-length sequence of scalars.

        Column-wise, so nothing here is per-row: a caller's product of entities
        and labels never becomes Python objects.

        A `None` in `schema` is inferred, which an undeclared result's value
        column needs. An `Enum` is built as its `String` and cast by the insert -
        narwhals cannot construct an arrow enum - which is also what rejects a
        label the dtype does not declare.
        """
        buildable = {
            name: (nw.String() if isinstance(dtype, nw.Enum) else dtype)
            for name, dtype in schema.items()
        }
        return as_relation(
            nw.DataFrame.from_dict(
                dict(columns), schema=buildable, backend="pyarrow"
            ).lazy(),
            self.con,
        )

    def _stage_long(
        self, attribute: str, lazy: nw.LazyFrame, kind: str, dims: dict[str, Any]
    ) -> None:
        """Stage a long frame that supplies its own keys.

        Its keys are the entity and whatever coordinates it carries; `dims`
        supplies any the frame leaves out, so a caller may scope a whole frame
        to one connection without repeating it per row.

        Notes
        -----
        - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
        - [set](https://energy-models.github.io/datarecord/design/working-record/#set)
        """
        table = self._ensure(kind, attribute)
        rel = as_relation(lazy, self.con)
        self._insert_long(
            rel, table, attribute, set(lazy.collect_schema().names()), dims
        )

    def _stage_derived(
        self,
        attribute: str,
        expr: nw.Expr,
        *,
        entity: Sequence[str] | None = None,
        kind: str = "inputs",
        **dims: Any,
    ) -> None:
        """Stage a value derived from the current one - the `Expr` form.

        Reads before it stages, so what it derives from is the resolved value
        *including earlier pending edits*, and two such calls compose. What is
        staged is the result, never the expression, so a committed layer holds
        ordinary rows and nothing stores that a value was derived.

        On a layered base the read is a fold, so this is the one edit whose cost
        scales with the ancestry rather than with the rows written.

        Unscoped, this derives from every row of the attribute across the types
        declaring it - there being no type keyword to narrow it.

        Notes
        -----
        - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
        - [a derived value](https://energy-models.github.io/datarecord/design/working-record/#an-nwexpr-value-derived-from-the-current-one)
        """
        source = self.outputs if kind == "outputs" else self.attributes
        # Once: three uses follow, and something iterating only once would be
        # exhausted by the first.
        names = None if entity is None else list(entity)
        if attribute not in source:
            frame = None
        else:
            frame = source[attribute]
            if names is not None:
                if kind == "inputs":
                    for name, ctype in self._resolve_types(names).items():
                        self._validate_attribute(ctype, attribute, dims, name=name)
                frame = frame.filter(nw.col("entity").is_in(names))
            for dim, value in dims.items():
                frame = frame.filter(nw.col(dim) == value)

        # A named target that resolves to no row is a failed change, not a
        # no-op: the caller asked for these rows to take a new value and there
        # is nothing to derive one from. With `entity=None` and no scope the
        # instruction is "whatever resolves", so an empty result is an answer.
        if names is not None or dims:
            # `head(1)`: `is_empty` is a `DataFrame` method, so the question
            # costs a collect either way - this one collects a single row.
            if frame is None or frame.select("entity").head(1).collect().is_empty():
                scope = ", ".join(
                    filter(
                        None,
                        [
                            f"entity={names}" if names is not None else "",
                            *(f"{d}={v!r}" for d, v in dims.items()),
                        ],
                    )
                )
                msg = (
                    f"no {attribute!r} rows resolve for {scope}, so there is "
                    f"no current value to derive from; `set` a value directly to "
                    f"create one (https://energy-models.github.io/datarecord/design/working-record/#an-nwexpr-value-derived-from-the-current-one)"
                )
                raise KeyError(msg)
        if frame is None:
            return
        self._stage_resolved(frame.with_columns(expr.alias("value")), attribute, kind)

    def _stage_resolved(
        self, frame: nw.LazyFrame, attribute: str, kind: str = "inputs"
    ) -> None:
        """Stage an already-long frame carrying every key column.

        `value` needs no cast: the table is this attribute's own, so its column
        already has the attribute's type (`_empty_long`).
        """
        table = self._ensure(kind, attribute)
        # A coordinate the frame leaves out is one it broadcasts over, so it is
        # filled with a typed NULL rather than left to `INSERT ... BY NAME`:
        # projecting the table's full column list keeps the insert positional
        # and the types the table's own (https://energy-models.github.io/datarecord/design/record/#the-broadcast-rule).
        present = set(frame.collect_schema().names())

        rel = as_relation(frame, self.con)
        duck_types = DuckTypes(rel)

        def column(c: str) -> Expression:
            if c in present:
                return col(c)
            if c == "attribute":
                return lit(attribute).alias(c)
            dtype = nw.Float64() if c == "breakpoint" else self._column_type(c)
            return duck_types.null(dtype).alias(c)

        shaped = rel.project(
            *(column(c) for c in self.schema.long_columns_for(attribute))
        )
        # Keyed the same whether input or result: a repeat coordinate is the
        # caller restating one, and replacing it is the answer folding it gave.
        # No `_complete_owned_whole`: the derived frame resolved the current
        # value, so it already carries the extent an edit would have to.
        self._insert(shaped, table, {}, key=self._long_key(attribute))

    def add(self, ctype: str, frame: Any) -> None:
        """Stage new components from a wide frame.

        Splits it: attributes addressed by `entity` alone stay in `dims/entity_type/`,
        varying ones become `inputs/` rows. Which is which comes from the
        schema, so this needs no framework registry.

        Not a sequence of `set` calls: a component exists by virtue of its
        member row, so staging attribute values for a name no layer declares is
        what `_validate_attribute` rejects. Adding a bus with no attributes makes
        the point - nothing to `set`, yet the bus must exist.

        `ctype` stays a parameter where `set` loses it: this is the call that
        establishes a name's type, so there is nothing yet to look it up in. It is also where uniqueness is enforced.

        Notes
        -----
        - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
        - [add / remove](https://energy-models.github.io/datarecord/design/working-record/#add-remove)
        """
        lazy = _incoming(frame, self.con)
        columns = lazy.collect_schema().names()
        if "entity" not in columns:
            msg = "`add` needs an `entity` column"
            raise ValueError(msg)
        self._require_unique(ctype, lazy)

        declared = self.schema.attributes_for(ctype)
        varying = [c for c in columns if declared.get(c) and declared[c].varying]
        # An attribute addressed by a group belongs to that group's table, not
        # the member frame - putting it there would introduce a column the
        # ancestors' files lack, which then reads as NULL for their rows. It
        # says so by naming the group among its `dims`, which is what replaced
        # a field of its own (https://energy-models.github.io/datarecord/design/record/#connections).
        # Only a group `entity` itself keys: a wide frame adding one component
        # can describe that component's own row of such a group (`bus`, for
        # `connection`), but not a `corridor`, which relates two entities
        # neither is "the" one being added here - that goes through `add_group`.
        by_group: dict[str, list[str]] = {}
        for c in columns:
            if not declared.get(c) or c in varying:
                continue
            for group in self.schema.groups_of(c):
                if "entity" in self.schema.group_key(group):
                    by_group.setdefault(group, []).append(c)
        ports = [c for cols in by_group.values() for c in cols]
        member_cols = [c for c in columns if c not in varying and c not in ports]

        rel = as_relation(lazy, self.con)
        # Where a group declares the type axis a component's constant columns are
        # a per-type member file; where none does there is no type to classify
        # into, so they are columns of the entity axis itself and there is no
        # member table (https://energy-models.github.io/datarecord/design/format/#where-a-value-lives).
        typed = self.schema.entity_type_dim is not None
        axis_supplied = {"deleted": lit(False)}  # noqa: FBT003
        if typed:
            axis_supplied["entity_type"] = lit(ctype)
            members = self._ensure(_MEMBERS, ctype)
            self._reject_undeclared(f"add({ctype!r}, ...)", members, member_cols)
            self._release_from_other_types(ctype, rel)
        else:
            self._reject_undeclared(
                f"add({ctype!r}, ...)", self._ensure(_ENTITY_AXIS), member_cols
            )

        # One row per component on the entity axis, replacing any this record
        # already staged for the name: it says the component exists, of what type
        # where there are types, and - untyped - carries its constant columns.
        # `add` after `remove` of the same name is thus one row: the tombstone is
        # deleted, not left to be outranked. Where typed, a second row goes to
        # the member table for those constant columns.
        self._insert(rel, self._ensure(_ENTITY_AXIS), axis_supplied, key=("entity",))
        if typed:
            self._insert(
                rel,
                members,
                {"deleted": lit(False)},  # noqa: FBT003
                key=("entity",),
            )
        for attribute in varying:
            # Always `inputs`: `add` declares components, and a component's
            # attribute values are inputs whatever a later solve produces.
            self._stage_long(
                attribute,
                lazy.select("entity", nw.col(attribute).alias("value")),
                "inputs",
                {},
            )
        # A group's coordinates name the row itself rather than being an
        # attribute of one, so they become that group's row; an attribute
        # addressed by the group rides along
        # (https://energy-models.github.io/datarecord/design/record/#connections).
        for group, group_cols in by_group.items():
            coordinates = self.schema.group_coordinates(group)
            extra = [c for c in group_cols if c not in coordinates]
            self.add_group(
                group,
                lazy.select(
                    "entity",
                    *(nw.col(c) for c in coordinates if c != "entity"),
                    *(nw.col(c) for c in extra),
                ),
            )

    def _release_from_other_types(self, ctype: str, rel: DuckDBPyRelation) -> None:
        """Drop `rel`'s names from every *other* type's staged member table.

        A name belongs to one type, so claiming it for `ctype` is what ends any
        other type's claim - and the rows behind that claim are in a table
        `ctype`'s own collapse cannot see. Left there, a `remove` under one type
        and an `add` under another would write the name into two member files,
        which `write_record` rejects as a collision.

        Deleted rather than tombstoned, and here rather than filtered on every
        read: the staged rows are what the layer *will* write, so a row no
        longer true of the layer has no reason to be in them.

        Only the staged tables. A name held by an ancestor is released by this
        layer's own axis row naming the new type, which is the fold's job and
        needs nothing removed.

        Notes
        -----
        - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
        - [add / remove](https://energy-models.github.io/datarecord/design/working-record/#add-remove)
        """
        others = [t for t in self._staged_types() if t != ctype]
        if not others:
            return
        names = rel.project(col("entity")).distinct()
        for other in others:
            table = self._staged[(_MEMBERS, other)]
            # `IN` over the incoming names, which DuckDB reads off the relation
            # by replacement scan - the same crossing every insert here makes.
            _names = names  # noqa: F841 - bound by the scan below
            self.con.execute(
                f"DELETE FROM {table} WHERE entity IN (SELECT entity FROM _names)"
            )

    def _reject_undeclared(self, call: str, table: str, columns: Sequence[str]) -> None:
        """Refuse a column the staging table's file has no place for.

        A staging table is shaped from the schema, like the file it becomes, so
        a column outside that shape has no declared dtype and no reader that
        knows what it means - the same thing `_validate_frame` refuses of an
        axis file, at the edit rather than at the write. Widening the table to
        fit instead would have to guess the dtype from the caller's frame.

        A tool that grows a column changes its schema first, which
        `_reconcile_schema` accepts as a widening; there is no path here that
        needs a column the record cannot describe.

        Raises
        ------
        ValueError
            Naming the columns and what the file does hold.

        Notes
        -----
        - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
        - [versioning](https://energy-models.github.io/datarecord/design/schema/#versioning)
        """
        known = list(self.con.table(table).columns)
        allowed = {c.lower() for c in known}
        extra = sorted(c for c in columns if c.lower() not in allowed)
        if not extra:
            return
        msg = (
            f"`{call}` was given columns {extra} the schema does not declare "
            f"there; it holds {sorted(known)}. An attribute's `dims` are what "
            f"put it in a file, so declare it before writing it (https://energy-models.github.io/datarecord/design/schema/#versioning)"
        )
        raise ValueError(msg)

    def _stage_tombstones(
        self,
        kind: str,
        fixed: tuple[str, ...],
        keys: list[list[Any]],
        key: tuple[str, ...],
        attribute: str | None = None,
    ) -> None:
        """Replace one `deleted` row per key.

        Shared by `remove` and `remove_group`, which differ only in their
        columns - a group's key where `remove` carries the entity. One helper so
        the shape is derived from the column list rather than restated per
        caller, which is what let the two drift out of step.

        `fixed` are the columns the tombstone carries; `key` is the subset an
        edit replaces on, which is `entity` alone even where the entity axis
        also carries the type - so a tombstone deletes an `add` row whatever type
        it named.

        `keys` arrives row-oriented and is transposed to build the relation,
        columns being how every insert here crosses into DuckDB.

        Notes
        -----
        - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
        - [add / remove](https://energy-models.github.io/datarecord/design/working-record/#add-remove)
        """
        table = self._ensure(kind, attribute)
        if not keys:
            return
        by_column = dict(zip(fixed, zip(*keys, strict=True), strict=True))
        rel = self._values_relation(by_column, {c: self._column_type(c) for c in fixed})
        self._insert(
            rel,
            table,
            {"deleted": lit(True)},  # noqa: FBT003
            key=key,
        )

    def remove(self, ctype: str, names: Sequence[str]) -> None:
        """Stage a tombstone per entity.

        Need not enumerate what it deletes: one row per key, and the fold
        applies it to every attribute. Nor scope it - a component exists or it
        does not, so a deletion removes it whole.

        One row, on the entity axis, which is where membership lives and the only
        place the fold reads a tombstone from. A member file holds values, never
        a `deleted` (`_member_columns`), so no second write there keeps step
        with this one. The axis row carries the type only where a group declares
        the axis; the delete keys on `entity` alone regardless, so it lands
        whatever type the `add` named.

        Notes
        -----
        - [add / remove](https://energy-models.github.io/datarecord/design/working-record/#add-remove)
        - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
        """
        if self.schema.entity_type_dim is not None:
            self._stage_tombstones(
                _ENTITY_AXIS,
                ("entity_type", "entity"),
                [[ctype, name] for name in names],
                ("entity",),
            )
        else:
            self._stage_tombstones(
                _ENTITY_AXIS, ("entity",), [[name] for name in names], ("entity",)
            )

    def add_group(self, group: str, frame: Any) -> None:
        """Stage rows of one declared group from a frame carrying its coordinates.

        The one path every group is added through, a record's `connection`
        group included.

        No component type, which is no coordinate of a group
        (https://energy-models.github.io/datarecord/design/format/#where-a-value-lives).

        Notes
        -----
        - [groups](https://energy-models.github.io/datarecord/design/schema/#groups)
        """
        coordinates = self.schema.group_coordinates(group)
        lazy = _incoming(frame, self.con)
        columns = lazy.collect_schema().names()
        for required in coordinates:
            if required not in columns:
                msg = f"`add_group({group!r}, ...)` needs a {required!r} column"
                raise ValueError(msg)
        table = self._ensure(group)
        extra = [c for c in columns if c not in coordinates]
        self._reject_undeclared(f"add_group({group!r}, ...)", table, extra)
        self._insert(
            as_relation(lazy, self.con),
            table,
            {"deleted": lit(False)},  # noqa: FBT003
            key=self.schema.group_key(group),
        )

    def remove_group(self, group: str, keys: Sequence[tuple[Any, ...]]) -> None:
        """Stage a tombstone per key, over one declared group's `group_key`.

        An `into` label is no part of a key: the tuple is removed, whatever
        label it carried.

        Notes
        -----
        - [groups](https://energy-models.github.io/datarecord/design/schema/#groups)
        """
        group_key = self.schema.group_key(group)
        self._stage_tombstones(group, group_key, [list(key) for key in keys], group_key)

    # -- commit / rollback (https://energy-models.github.io/datarecord/design/working-record/#committing) -------------------------

    def rollback(self) -> None:
        """Clear every staged row without writing.

        Notes
        -----
        - [WorkingRecord](https://energy-models.github.io/datarecord/design/working-record/)
        """
        for name in self._staged.values():
            self.con.execute(f"DROP TABLE IF EXISTS {name}")
        self._staged.clear()

    def _collapsed_inputs(self, attribute: str) -> DuckDBPyRelation | None:
        """One attribute's staged rows, tombstones applied.

        The rows are already one per coordinate - an edit replaced its key rather
        than appending beside it (`_replace`) - so this is the table scan minus
        the coordinates a component tombstone reaches. That anti-join stays, being
        a cross-table fact rather than an ordering one: a `remove` on the entity
        axis has to clear this attribute's rows for the name too.

        None where nothing is staged for it, which is what says the base's rows
        stand alone.

        Notes
        -----
        - [committing](https://energy-models.github.io/datarecord/design/working-record/#committing)
        """
        rel = self._rows("inputs", attribute)
        if rel is None:
            return None
        dead = self._tombstoned()
        if dead is None:
            return rel
        # A deleted component has no attributes, so the tombstone drops a staged
        # value for its name (https://energy-models.github.io/datarecord/design/working-record/#committing).
        #
        # Matched on `name` alone: the tombstone carries a type and a staged
        # input row does not (https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types).
        on = null_safe("l", "d", ("entity",))
        return rel.set_alias("l").join(dead.set_alias("d"), on, how="anti")

    def _tombstoned(self) -> DuckDBPyRelation | None:
        """Component keys whose staged entity-axis row is a tombstone.

        One row per name in the table already (`_replace` on `entity`), so an
        `add` after a `remove` has replaced the tombstone rather than sitting
        above it - the scan is the answer.

        Notes
        -----
        - [committing](https://energy-models.github.io/datarecord/design/working-record/#committing)
        """
        rel = self._rows(_ENTITY_AXIS)
        if rel is None:
            return None
        return rel.filter(col("deleted")).project(col("entity"))

    def _collapsed_entities(self) -> DuckDBPyRelation | None:
        """The staged entity axis: which names exist, of what type, dead or live.

        A table scan: one row per name, keyed by `entity` alone with the type
        carried (`_replace`), so a `remove` under one type and an `add` under
        another are already one row - a retype replaced the row rather than
        merging into a member that is both a `Bus` and deleted.

        Notes
        -----
        - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
        - [committing](https://energy-models.github.io/datarecord/design/working-record/#committing)
        """
        return self._rows(_ENTITY_AXIS)

    def _collapsed_members(self, ctype: str) -> DuckDBPyRelation | None:
        """One type's staged member rows - a table scan, one per entity.

        A member file holds values, never a tombstone (`_member_columns`):
        membership is the entity axis's, and `remove` writes only there, so this
        table carries no `deleted` row to keep out.

        No cross-type filter. A name this layer gave to another type is not in
        this table at all, `add` having released it (`_release_from_other_types`)
        - so one table's rows are already only its own, and the type never has to
        be resolved on the way out.

        Notes
        -----
        - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
        - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
        - [committing](https://energy-models.github.io/datarecord/design/working-record/#committing)
        """
        return self._rows(_MEMBERS, ctype)

    def _collapsed_group(self, group: str) -> DuckDBPyRelation | None:
        """One group's staged rows - a table scan, one per `group_key`.

        Restating a tuple replaced its row (`_replace` on `group_key`), so the
        table already holds one per key; a different `into` label is that edit,
        not a second row.

        Notes
        -----
        - [committing](https://energy-models.github.io/datarecord/design/working-record/#committing)
        """
        return self._rows(group)

    # -- what commit writes (https://energy-models.github.io/datarecord/design/working-record/#committing) -----------------------------------------

    def _staged_dims(self) -> tuple[str, ...]:
        """Which axes have staged rows, in declaration order.

        Rows rather than tables: `_ensure` creates the table before the label
        checks run, so a rejected `set` leaves an empty one behind and a table
        that exists is not yet an edit.

        `entity` is one of them, staged as `_ENTITY_AXIS` like any other dim -
        so both the fold's `axes()` and what `commit` writes come from here,
        rather than one of them special-casing it. `Schema` declares `entity`
        like any dim, so `schema.dims` names it and no arm is needed for it.

        Notes
        -----
        - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
        - [the schema](https://energy-models.github.io/datarecord/design/schema/)
        """
        staged = {
            k[len(_AXIS_PREFIX) :]
            for (k, _), _ in self._staged.items()
            if k.startswith(_AXIS_PREFIX)
        }
        return tuple(
            d
            for d in self.schema.dims
            if d in staged
            and self.con.table(self._table(f"{_AXIS_PREFIX}{d}")).limit(1).fetchone()
        )

    def _axis_layer(self, dim: str) -> DuckDBPyRelation:
        """One axis as this layer writes it, which `partial` decides the extent of.

        The staged table already holds one complete row per label an edit touched
        (`_patch_axis`), so there is nothing to merge on the way out - what
        `partial` decides is only how many labels are in it:

        - **`partial`** - the touched labels alone, the fold resolving the rest
          from the parent. The staged table is exactly that.
        - **not `partial`** - a dim a layer owns whole once it touches it, so the
          untouched labels are carried from the base, unioned by name so a base
          row lacking a newly-added column reads NULL there.

        `entity` is a table scan either way: membership is one row per name
        already (`_collapsed_entities`), never completed from the base - a layer
        names the components it touched and the fold resolves the rest.

        Notes
        -----
        - [partial](https://energy-models.github.io/datarecord/design/schema/#partial-the-granularity-of-an-override)
        - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
        """
        if dim == "entity":
            entities = self._collapsed_entities()
            assert entities is not None, "only a staged dim reaches an axis layer"
            return entities
        staged = self._rows(f"{_AXIS_PREFIX}{dim}")
        assert staged is not None, "only a staged dim reaches an axis layer"
        base = self._base.dims.axes.get(dim)
        if base is None or dim in self.schema.partial_dims:
            return staged
        # The untouched labels the layer owns whole and so must carry, taken from
        # the base with their whole rows - unioned by name, the staged side
        # carrying any column the base lacks and the base side the reverse.
        untouched = base.set_alias("b").join(
            staged.set_alias("s"), null_safe("b", "s", [dim]), how="anti"
        )
        return union_all_by_name([staged, untouched], self.con)

    def _staged_types(self) -> tuple[str, ...]:
        """Which entity types have a staged member table, in insertion order."""
        return tuple(t for (k, t), _ in self._staged.items() if k == _MEMBERS and t)

    def _staged_groups(self) -> Frames:
        """The staged group rows, keyed by group - one frame each.

        A `Frames` for `StagedSource.groups()`'s key set, not a builder of the
        rows themselves - `StagedSource.group` reads those directly off
        `_collapsed_group`.
        """
        staged = {
            g: rel
            for g in self.schema.groups
            if (rel := self._collapsed_group(g)) is not None
        }
        return LazyFrames(tuple(staged), lambda group: nw.from_native(staged[group]))

    def _base_revision(self) -> Revision:
        """The `Revision` this record's base resolves, for `NewChild()`'s default.

        Only a base that is a node in the tree has one, which is asked of the
        `revisions` table rather than of the base's type: a directory's
        `revision_id` is derived from where it is, so it names no row, and that
        - not which class the base was - is what makes it unbranchable.

        A missing row and a missing table are the same answer here: `connect`
        creates the table, so a connection without one was made by hand and has
        no tree in it either.
        """
        try:
            return Revision.get(self._base.revision_id, self.con)
        except (KeyError, duckdb.Error):
            msg = (
                "`NewChild()` needs a revision to branch from, and this "
                "`WorkingRecord`'s base is no node in a layer tree; pass one as "
                "`NewChild(revision)`, or commit to a standalone record with "
                "`Directory(uri)` (https://energy-models.github.io/datarecord/design/working-record/#committing)"
            )
            raise ValueError(msg) from None

    @overload
    def commit(self, target: NewChild) -> Revision: ...

    @overload
    def commit(self, target: Directory) -> None: ...

    def commit(self, target: Target) -> Revision | None:
        """Write everything staged and clear it.

        Returns
        -------
        The new child for a `NewChild` target, so the caller can read what it
        just wrote without going back to the record table; `None` for a
        `Directory`, which belongs to no record. Overloaded on the target, so a
        caller committing to a child holds a `Revision` rather than an optional
        one - which target it passed is what decides, and it is always literal
        at the call site.

        The layer lands in the *child*, never in the node that was branched
        from - layers are write-once - so it is the returned node that
        reads back the edits.

        Notes
        -----
        - [a layer's data is write-once](https://energy-models.github.io/datarecord/design/layers/#a-layers-data-is-write-once)
        - [committing](https://energy-models.github.io/datarecord/design/working-record/#committing)
        """
        if isinstance(target, NewChild):
            parent = (
                target.record if target.record is not None else self._base_revision()
            )
            child = parent.child()
            # The staged layer's own rows - the fold's last source, a `LayerData`
            # like any other - so only the edits are written, the fold resolving
            # the rest from the parent (https://energy-models.github.io/datarecord/design/working-record/#committing).
            write_record(child.id, self.resolver.sources[-1], self.con)
            self.rollback()
            return child
        # The `Resolver`, not `self`: it is the same `LayerData` a fold answers
        # for everything folded in, which is exactly base-plus-staged flattened
        # (https://energy-models.github.io/datarecord/design/working-record/#reading-with-pending-edits).
        write_record(None, self.resolver, self.con, uri=target.uri)
        self.rollback()
        return None


def _base_resolver(base: RecordLike, con: DuckDBPyConnection) -> Resolver:
    """What `base` resolves from, as a `Resolver` a staged layer can extend.

    A `Record` is one already - whether it came from a revision or from
    `Record.at(uri)`, since a directory read as its one layer is a fold like
    any other.

    Raises
    ------
    TypeError
        For anything else. A framework object hands over narwhals frames and has
        no layer layout behind it: `axis("entity")` wants the entity axis and
        `RecordLike` exposes `entity_types` keyed by type, so synthesising one
        would be rebuilding the format from a protocol that deliberately lacks
        it.
    ValueError
        For a `Record` on another connection, which would read its rows through
        one connection and stage them on another.
    """
    if not isinstance(base, Record):
        msg = (
            f"a `WorkingRecord` reads by folding its staged rows over the base's, "
            f"and a {type(base).__name__} has no layer layout to fold - pass a "
            f"`Record`, which `Revision.record` and `Record.at(uri)` both give"
        )
        raise TypeError(msg)
    if base.con is not con:
        msg = (
            "a `WorkingRecord` stages its rows on the connection it reads the "
            "base through, and this base reads through another one"
        )
        raise ValueError(msg)
    return base.resolver


def _column_type(schema: Schema, column: str) -> nw.dtypes.DType:
    """A staged column's declared type, for a typed NULL or literal.

    Every column a staging table has is a declared dim or a structural one,
    both of which the schema types - so a missing type is a disagreement
    between the table's shape and the schema, not a column to guess at.

    Raises
    ------
    ValueError
        If the schema declares no type for `column`.
    """
    dtype = schema.column_type(column)
    if dtype is None:
        msg = (
            f"the schema declares no type for {column!r}, which a staged "
            f"row needs to fill (https://energy-models.github.io/datarecord/design/format/#the-long-schema)"
        )
        raise ValueError(msg)
    return dtype


def _member_columns(schema: Schema, ctype: str) -> dict[str, nw.dtypes.DType]:
    """One type's member columns: the key, and the attributes it alone carries.

    The shape of `dims/entity_type/{ctype}.parquet`: `entity` and one column
    per attribute the type carries, nothing else. No `entity_type`, which is the
    file's name and would be a second copy of it. No `deleted` either -
    membership is the entity axis's alone, and a tombstone reaches the fold from
    `dims/entity.parquet`, never from a member file.

    Non-varying and not addressed by a group: a varying attribute is a long row
    and a group's is that group's file, so neither is a column here. That is the
    same split `add` makes of an incoming frame, read off the schema instead of
    off the columns a caller passed.

    Notes
    -----
    - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
    - [types](https://energy-models.github.io/datarecord/design/schema/#types-what-a-type-carries)
    """
    own = {
        name: schema.value_type(name) or nw.String()
        for name, spec in schema.attributes_for(ctype).items()
        if not spec.varying and not schema.groups_of(name)
    }
    return {
        "entity": _column_type(schema, "entity"),
        **own,
    }


def _group_columns(schema: Schema, group: str) -> dict[str, nw.dtypes.DType]:
    """One group's staged columns: its coordinates, and the fold's own.

    No `entity_type`, which is no coordinate of a group
    (https://energy-models.github.io/datarecord/design/format/#where-a-value-lives).

    An attribute over the group is a column of the group's file, so it is
    declared here too. `role` on a connection reads as a framework's own label,
    but the framework declaring it is what puts it in the schema - and an
    undeclared one has no dtype to give the column.

    Notes
    -----
    - [groups](https://energy-models.github.io/datarecord/design/schema/#groups)
    - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
    """
    over = {
        name: schema.value_type(name) or nw.String()
        for name, spec in schema.attributes.items()
        if not spec.varying and group in schema.groups_of(name)
    }
    return {
        **{c: _column_type(schema, c) for c in schema.group_coordinates(group)},
        "deleted": nw.Boolean(),
        **over,
    }


def _axis_columns(schema: Schema, dim: str) -> dict[str, nw.dtypes.DType]:
    """One axis's staged columns: its key, its tombstone, the attributes it carries.

    The shape of `dims/{dim}.parquet` for the attributes addressed by `dim`
    alone (`attributes_on`), plus the structural columns an axis file may hold:
    `deleted`, which `_validate_frame` admits on any axis, and `entity_type` on
    the entity axis where a group declares the type axis.

    `entity_type` on the entity axis only where a group declares the axis. Where
    none does there is no type to classify a component into, so no member files
    and no column: `attributes_on("entity")` then returns the component's
    non-varying columns and they live here, on the entity axis itself. Where a
    group does declare it the column is `entity_type`, typed by the declaration,
    and `attributes_on("entity")` is empty - the columns are per-type member
    files instead.

    Notes
    -----
    - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
    - [entity types](https://energy-models.github.io/datarecord/design/schema/#entity_type-the-axis-of-kinds)
    """
    into = schema.entity_type_dim
    classifying = (
        # The column is `entity_type` whatever the dim is called, that being the
        # name `dims/entity.parquet` carries it under.
        {"entity_type": _column_type(schema, into)}
        if dim == "entity" and into is not None
        else {}
    )
    return {
        **{c: _column_type(schema, c) for c in schema.axis_key(dim)},
        **classifying,
        "deleted": nw.Boolean(),
        **{a: schema.value_type(a) or nw.String() for a in schema.attributes_on(dim)},
    }


# A staging kind, so it becomes part of a table name: no punctuation a SQL
# identifier would need quoting for, and no collision with a group's name,
# which `Schema` already rejects for colliding with a declared dim.
_AXIS_PREFIX = "axis_of_"

_ENTITY_AXIS = f"{_AXIS_PREFIX}entity"
"""The entity axis's staging kind - an axis like any other, named like one.

`dims/entity.parquet` is a file a layer holds, so staging it as an axis is what
makes the staged layer the same shape as every other. What differs is only how an
edit keys it: membership replaces on `entity` alone, so a `remove` then an `add`
under another type resolves to one row, where an ordinary axis patches a column
in place (`_patch_axis`) and keeps its siblings.
"""

_MEMBERS = "members"
"""The kind whose second slot is an entity type rather than an attribute.

One table per type, shaped like `dims/entity_type/<Type>.parquet`. Separate from
`entities`, which holds membership alone: sharing one table made every type's
columns every other type's, so a `Bus` frame carried a `Generator`'s.
"""
