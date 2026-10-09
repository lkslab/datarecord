# SPDX-FileCopyrightText: datarecord contributors
#
# SPDX-License-Identifier: MIT

"""Writing a whole record as a layer.

A `LayerData` hands over relations and this module turns them into parquet;
producing one from a framework's own object is a tool's job.

Notes
-----
- [writing a whole record](https://energy-models.github.io/datarecord/design/writing/)
- [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
- [module layout](https://energy-models.github.io/datarecord/design/module-layout/)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from duckdb import ColumnExpression as col
from duckdb import ConstantExpression as lit
from duckdb import StarExpression as star

from datarecord.duck import as_relation, base_uri_of, fn, layer_dir, union_all_by_name
from datarecord.layered.resolve import (
    cast_declared,
    read_schema,
    store_owner_maps,
    write_schema,
)
from datarecord.record import Frames, LayerData, RecordLike, collision_detail
from datarecord.schema import Schema

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection, DuckDBPyRelation


def write_record(
    revision_id: UUID | None,
    source: LayerData | RecordLike,
    con: DuckDBPyConnection,
    *,
    uri: str | None = None,
) -> None:
    """Write `source` as `revision_id`'s layer, which must not exist yet.

    An existing layer directory is an error rather than an overwrite or a merge,
    so a whole-record write can never half-replace what a record holds. Keys are
    looked up one at a time and each file written before the next is built, so a
    lazily-building source does one read per file rather than one per key up
    front.

    Parameters
    ----------
    revision_id
        The record whose layer this is; `layer_dir` derives the path.
        `None` only together with `uri`, for a standalone record that belongs
        to no record.
    uri
        Write here instead of at the revision's own layer - how a `Directory`
        commit target produces a record outside the layer tree.
    source
        The layer's contents: a `LayerData` - a `StagedSource` for a `NewChild`
        commit, a `Resolver` for a `Directory` one - or a framework's own
        `RecordLike`, wrapped in a thin adapter reading its `Frames` through the
        same enumerate-and-read pairs. Validated against its own schema before
        anything is written.
    con
        Connection to write through.

    Raises
    ------
    FileExistsError
        If the layer directory already exists.
    ValueError
        If a long frame is missing a long-schema column, or the schema declares a key
        dim no frame carries - either would make the fold misresolve the layer.

    Notes
    -----
    - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
    - [writing a whole record](https://energy-models.github.io/datarecord/design/writing/)
    - [committing](https://energy-models.github.io/datarecord/design/working-record/#committing)
    - [module layout](https://energy-models.github.io/datarecord/design/module-layout/)
    """
    if uri is None:
        if revision_id is None:
            msg = "write_record needs a revision_id or a uri"
            raise ValueError(msg)
        base = layer_dir(revision_id)
    else:
        base = uri if uri.endswith("/") else uri + "/"
    local = "://" not in base
    if local and Path(base).exists():
        msg = f"layer {base} already exists; write_record creates a new layer (https://energy-models.github.io/datarecord/design/writing/)"
        raise FileExistsError(msg)

    data = (
        source if isinstance(source, LayerData) else _RecordLikeAsLayerData(source, con)
    )
    schema = data.schema
    if uri is None:
        # One schema for the whole tree (https://energy-models.github.io/datarecord/design/schema/#one-schema-per-record). The first layer written
        # declares it; every later one is checked against it, so a layer
        # cannot quietly redefine what an attribute means.
        _reconcile_schema(schema, con)

    # Staged then renamed, so a frame that fails validation part-way through
    # leaves no layer rather than half of one (https://energy-models.github.io/datarecord/design/writing/). Validation happens as each
    # frame is built, since building it twice would defeat the laziness.
    staging = f"{base.rstrip('/')}.staging/" if local else base
    if local:
        Path(staging).mkdir(parents=True)
    try:
        # A layer holds only data: a layered record's one schema lives beside
        # `layers/`, not inside any of them (https://energy-models.github.io/datarecord/design/schema/#one-schema-per-record). A standalone directory *is*
        # one record, so there the schema belongs in the directory.
        if local and uri is not None:
            with open(staging + "manifest.json", "w") as fh:
                fh.write(schema.model_dump_json())
        kinds = [
            ("dims", data.axes(), data.axis, "dims"),
            ("entities", data.entity_types(), data.entity_type, "dims/entity_type"),
            ("groups", data.groups(), data.group, "groups"),
            ("attributes", data.attributes(), data.attribute, "inputs"),
        ]
        # `outputs/` only for a source carrying results, so a record with none
        # produces a layer without the directory rather than an empty one (https://energy-models.github.io/datarecord/design/writing/).
        output_names = data.attributes("outputs")
        if output_names:
            kinds.append(
                (
                    "outputs",
                    output_names,
                    lambda name: data.attribute(name, "outputs"),
                    "outputs",
                )
            )
        # Each type's names, to check record-wide uniqueness once every component
        # frame has been seen (https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types).
        tagged: list[DuckDBPyRelation] = []
        for kind, keys, read, subdir in kinds:
            for key in keys:
                rel = read(
                    key
                )  # looked up exactly once (https://energy-models.github.io/datarecord/design/writing/)
                if rel is None:
                    continue
                _validate_frame(rel, kind, key, schema)
                if kind == "entities":
                    tagged.append(rel.project("entity", lit(key).alias("entity_type")))
                _write_frame(
                    rel,
                    f"{staging}{subdir}/{key}.parquet",
                    schema,
                    # A per-type member file is indexed by `entity` and holds
                    # one column per attribute; the type is the file it is in,
                    # and `dims/entity.parquet` is what carries it for every
                    # later reader. A column repeating it here would be a third
                    # copy that can disagree.
                    drop=("entity_type",) if kind == "entities" else (),
                )
        _require_unique(tagged, con)
        # A standalone record is written whole and renamed into place, so the
        # map folded over the staged rows is its resolved map forever; a tree
        # layer's map depends on its ancestors, which `materialise` folds.
        if local and uri is not None:
            store_owner_maps(staging, schema, con)
    except BaseException:
        if local:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    if local:
        Path(staging).rename(base.rstrip("/"))


def _reconcile_schema(schema: Schema, con: DuckDBPyConnection) -> None:
    """Declare the record's schema, or check this layer agrees with it.

    A schema is not layered data, so there is nothing to fold: the first writer
    states it and the rest must be `compatible_with` it. Read and written beside
    `con`'s own layers, so one record never consults another's manifest.

    Raises
    ------
    ValueError
        If this layer's schema would make the record's existing layers
        unreadable.

    Notes
    -----
    - [one schema per record](https://energy-models.github.io/datarecord/design/schema/#one-schema-per-record)
    - [versioning](https://energy-models.github.io/datarecord/design/schema/#versioning)
    """
    base = base_uri_of(con)
    existing = read_schema(con)
    if not (existing.dimensions or existing.attributes):
        write_schema(schema, base)
        return
    if schema == existing:
        return
    problems = schema.compatible_with(existing)
    if problems:
        msg = (
            f"this layer's schema is incompatible with the record's: "
            f"{'; '.join(problems)} (https://energy-models.github.io/datarecord/design/schema/#versioning)"
        )
        raise ValueError(msg)
    # Compatible, so it supersedes: a widened schema still reads every layer
    # written under the narrower one.
    write_schema(schema, base)


DERIVED = ("order_key",)
"""Columns a resolved frame carries that no layer file may.

The fold's answer *about* a frame rather than data in it, so writing one would
both put a column in a file the format does not define and read as stored order
where the fold always re-derives it from file order.

Notes
-----
- [the owner map](https://energy-models.github.io/datarecord/design/read-path/#owner-map)
"""


def _write_frame(
    rel: DuckDBPyRelation,
    uri: str,
    schema: Schema,
    *,
    drop: tuple[str, ...] = (),
) -> None:
    """Persist one relation as parquet, unmaterialised.

    Columns are cast to their declared types on the way out, so a reader can
    trust them rather than re-casting an all-NULL column pandas typed as float.

    `DERIVED` is dropped from every file, and `drop` names what is redundant in
    *this* one - both here rather than in the callers, so a column a source
    happens to carry cannot reach a file by a path that forgot to strip it.

    Notes
    -----
    - [Frames](https://energy-models.github.io/datarecord/design/record/#frames)
    - [writing a whole record](https://energy-models.github.io/datarecord/design/writing/)
    """
    if "://" not in uri:
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
    unwritable = [c for c in (*DERIVED, *drop) if c in rel.columns]
    if unwritable:
        rel = rel.project(star(exclude=unwritable))
    cast_declared(schema, rel).to_parquet(uri)


def _require_unique(tagged: list[DuckDBPyRelation], con: DuckDBPyConnection) -> None:
    """Reject a record whose component types share a name.

    Unlike `_validate_frame`'s checks this reads the rows, uniqueness being a
    property of the data. A tombstone still occupies the name, so `deleted` is
    not filtered out.

    Parameters
    ----------
    tagged
        One relation per component type, each `(entity, entity_type)`.

    Raises
    ------
    ValueError
        Naming each clashing name and the types claiming it.

    Notes
    -----
    - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
    """
    if len(tagged) < 2:  # nothing to collide with
        return
    pairs = union_all_by_name(tagged, con).distinct()
    clashing = (
        pairs.set_alias("p")
        .join(
            pairs.aggregate(
                [col("entity"), fn.count(col("entity_type")).alias("_types")]
            )
            .filter(col("_types") > lit(1))
            .project("entity")
            .set_alias("c"),
            "p.entity = c.entity",
        )
        .project(col("p", "entity").alias("entity"), col("p", "entity_type"))
    )
    rows = clashing.fetchall()
    if rows:
        detail = collision_detail(rows)
        msg = (
            f"component types reuse names: {detail}; a name identifies one "
            f"component across every type (https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)"
        )
        raise ValueError(msg)


def _validate_frame(rel: DuckDBPyRelation, kind: str, key: str, schema: Schema) -> None:
    """Check one frame is shaped for the fold to resolve it.

    Structural only: a long frame carries its own attribute's coordinates, and a
    `dims/` frame carries every dim the schema declares it keyed by. Values
    are not checked - which component types and attribute names are valid
    belongs to whatever vocabulary the schema declares, and the record layer
    knows none.

    An attribute's coordinates are what its `dims` declare, so one file's column
    set is not another's and neither is every declared dim. A result the schema
    does not declare has no coordinates to derive, so it falls back to the fixed
    columns every long row has.

    Reads the schema rather than the rows, so validating an unmaterialised
    relation costs nothing.

    Notes
    -----
    - [the Record protocol](https://energy-models.github.io/datarecord/design/record/)
    - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
    - [the schema](https://energy-models.github.io/datarecord/design/schema/)
    - [partial](https://energy-models.github.io/datarecord/design/schema/#partial-the-granularity-of-an-override)
    """
    columns = set(rel.columns)

    # `outputs/` uses the same long schema as `inputs/`; it just does not
    # overlay (https://energy-models.github.io/datarecord/design/read-path/#outputs), which is a read-path property rather than a shape one.
    if kind in ("attributes", "outputs"):
        subdir = "inputs" if kind == "attributes" else "outputs"
        # An input's shape comes from its spec, so one the schema does not
        # declare has no shape to check it against - and writing it would put a
        # file in `inputs/` that no read path knows the columns of. A *result*
        # is never declared, a tool deriving those from its own registry.
        if kind == "attributes" and key not in schema.attributes:
            msg = (
                f"inputs/{key}.parquet is not a declared attribute; its `dims` "
                f"are what say which columns the file carries (https://energy-models.github.io/datarecord/design/schema/#attributespec)"
            )
            raise ValueError(msg)
        # A result's shape is not the schema's to fix, even where its name
        # matches a declared attribute: `outputs/control.parquet` may vary over
        # axes the *input* `control` does not. So only the fixed columns every
        # long row has are required of one (https://energy-models.github.io/datarecord/design/read-path/#outputs).
        required = (
            set(schema.long_columns_for(key))
            if kind == "attributes"
            else {"attribute", "breakpoint", "value"}
        )
        missing = sorted(required - columns)
        if missing:
            msg = (
                f"{subdir}/{key}.parquet is missing long-schema columns {missing}; "
                f"the resolved relation needs {sorted(required)} (https://energy-models.github.io/datarecord/design/format/#the-long-schema)"
            )
            raise ValueError(msg)
        # And an *input* carries nothing else: a coordinate the attribute is
        # not addressed by would be a column the read path never projects,
        # written as a fact about a value that does not have one. Reported
        # rather than dropped, since a source emitting one disagrees with the
        # schema about what the attribute is - the source's bug to fix.
        #
        # A result is exempt for the same reason it need not be declared: its
        # shape is a framework's business, and a name it shares with an input
        # says nothing about which coordinates the *result* varies over.
        extra = sorted(columns - required) if kind == "attributes" else []
        if extra:
            msg = (
                f"{subdir}/{key}.parquet carries columns {extra} the attribute is "
                f"not addressed by; its `dims` say {sorted(required)} (https://energy-models.github.io/datarecord/design/format/#the-long-schema)"
            )
            raise ValueError(msg)
        return

    if kind == "dims":
        # A nested axis is keyed by `(*parents, dim)` (https://energy-models.github.io/datarecord/design/schema/#within-an-axis-inside-an-axis), so its file needs
        # a column per parent - without one the fold would key by a column that
        # is not there, and two periods' identically-labelled timesteps would
        # resolve as one row. An undeclared dim has no nesting to check: the
        # schema's vocabulary is what `axis_key` reads, and a source may hand
        # over an axis the schema does not name.
        if key not in schema.dimensions:
            return
        missing = sorted(set(schema.axis_key(key)) - columns)
        if missing:
            msg = (
                f"dims/{key}.parquet is missing axis key columns {missing}; "
                f"{key!r} is `within` {sorted(schema.dimensions[key].within)} so "
                f"its labels identify a point only within them (https://energy-models.github.io/datarecord/design/schema/#within-an-axis-inside-an-axis)"
            )
            raise ValueError(msg)
        # An attribute addressed by this axis alone is a column here, and not
        # required: a record may declare one before any layer sets it, which
        # resolves to its `default` (https://energy-models.github.io/datarecord/design/format/#where-a-value-lives).
        #
        # A column no declaration accounts for is rejected, as a long frame's
        # extras are: one riding along uninvited would be read back as data
        # nothing knows the dtype or meaning of.
        known = (
            set(schema.axis_key(key))
            | set(schema.attributes_on(key))
            # The structural columns an axis file may carry: a tombstone, and an
            # explicit order key. Not every name in `STRUCTURAL_TYPES` - most of
            # those are a long row's, and `attribute` or `breakpoint` here would
            # be a long frame written to the wrong place.
            | {"deleted", "order_key"}
        )
        # The one classification column an axis file carries, every other group
        # being its own file. Admitted only where a group declares the type axis:
        # the label then says which `dims/entity_type/<Type>.parquet` a
        # component's non-varying attributes are in. Where no group declares it
        # there is no member file and no column - those attributes are
        # `attributes_on("entity")`, columns of this file, already in `known` -
        # and an `entity_type` column is rejected as any undeclared one is
        # (https://energy-models.github.io/datarecord/design/format/#where-a-value-lives).
        if key == "entity" and schema.entity_type_dim is not None:
            known.add("entity_type")
        extra = sorted(columns - known)
        if extra:
            msg = (
                f"dims/{key}.parquet carries columns {extra} the schema does not "
                f"declare for the {key!r} axis; an axis file holds its key and "
                f"the attributes addressed by it alone (https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)"
            )
            raise ValueError(msg)
        return

    if kind != "groups" or key not in schema.groups:
        return
    # A group's row is keyed by its coordinates, `into` among them, so a frame
    # lacking one would be keyed by a column that is not there.
    missing = sorted(set(schema.group_coordinates(key)) - columns)
    if missing:
        msg = (
            f"groups/{key}.parquet is missing the group's coordinates "
            f"{missing}; the fold would key by a column that is not there (https://energy-models.github.io/datarecord/design/schema/#groups)"
        )
        raise ValueError(msg)


class _RecordLikeAsLayerData:
    """A `RecordLike` read through `LayerData`'s enumerate-and-read pairs.

    The adapter that lets `write_record` stay one code path over raw
    relations: a framework's `to_datarecord()` hands over narwhals `Frames`,
    one lookup per key exactly as `write_record` already does, so this wraps
    each mapping rather than eagerly converting it. `con` is needed only to
    land a non-DuckDB frame as a relation (`as_relation`).

    Notes
    -----
    - [LayerData](https://energy-models.github.io/datarecord/design/record/#layerdata)
    """

    def __init__(self, source: RecordLike, con: DuckDBPyConnection) -> None:
        self._source = source
        self._con = con

    @property
    def schema(self) -> Schema:
        return self._source.schema

    @property
    def frozen(self) -> bool:
        # A framework object is read once to produce a layer, never folded
        # under a reader, so there is nothing for staleness to mean here.
        return True

    def _read(self, frames: Frames, key: str) -> DuckDBPyRelation | None:
        if key not in frames:
            return None
        return as_relation(frames[key], self._con)

    def axes(self) -> set[str]:
        return set(self._source.dims)

    def axis(self, dim: str) -> DuckDBPyRelation | None:
        return self._read(self._source.dims, dim)

    def entity_types(self) -> set[str]:
        return set(self._source.entity_types)

    def entity_type(self, name: str) -> DuckDBPyRelation | None:
        return self._read(self._source.entity_types, name)

    def groups(self) -> set[str]:
        return set(self._source.groups)

    def group(self, name: str) -> DuckDBPyRelation | None:
        return self._read(self._source.groups, name)

    def attributes(self, kind: str = "inputs") -> set[str]:
        frames = self._source.attributes if kind == "inputs" else self._source.outputs
        return set(frames)

    def attribute(self, name: str, kind: str = "inputs") -> DuckDBPyRelation | None:
        frames = self._source.attributes if kind == "inputs" else self._source.outputs
        return self._read(frames, name)
