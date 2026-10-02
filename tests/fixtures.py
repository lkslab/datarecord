# SPDX-FileCopyrightText: datarecord contributors
#
# SPDX-License-Identifier: MIT

"""Hand-built patch layers, since the v2 write path does not exist.

Notes
-----
- [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
"""

from pathlib import Path

import narwhals as nw
import pandas as pd

from datarecord.layered.resolve import write_schema as record_write_schema
from datarecord.schema import (
    AttributeSpec,
    Dimension,
    Group,
    Schema,
    TypeAttribute,
    TypeSpec,
)

# No `entity_type`: an attribute row is keyed by `name`, unique across every type
# (https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types). The entity tables below keep it.
LONG_COLUMNS = [
    "entity",
    "bus",
    "snapshot",
    "scenario",
    "period",
    "attribute",
    "breakpoint",
    "value",
]


def write_input(
    layer: str, attribute: str, rows: list[dict], *, snapshot_dtype="datetime64[ns]"
) -> None:
    """Write `inputs/<attribute>.parquet` in the long schema.

    Each row needs at least `name` and `value`; missing dimension columns
    default to NULL, i.e. "applies to the whole axis".
    `bus` set marks a per-connection attribute, `breakpoint`
    a piecewise-linear one; both NULL is the ordinary component-level
    scalar.

    Notes
    -----
    - [wide and long rows](https://energy-models.github.io/datarecord/design/record/#wide-and-long-rows)
    - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
    """
    df = pd.DataFrame(rows)
    df["attribute"] = attribute
    for col in LONG_COLUMNS:
        if col not in df:
            df[col] = None
    df["snapshot"] = pd.Series(df["snapshot"]).astype(snapshot_dtype)
    df["scenario"] = df["scenario"].astype("string")
    df["bus"] = df["bus"].astype("string")
    df["period"] = df["period"].astype("Int64")
    df["breakpoint"] = df["breakpoint"].astype("float64")
    df["value"] = df["value"].astype("float64")

    target = Path(layer, "inputs")
    target.mkdir(parents=True, exist_ok=True)
    df[LONG_COLUMNS].to_parquet(target / f"{attribute}.parquet", index=False)


def write_group(layer: str, group: str, rows: list[dict]) -> None:
    """Write `groups/<group>.parquet` from plain rows, whatever columns they carry.

    The generic form of `write_connections`: one file per group, keyed by its
    coordinates, with `deleted` supplied where a row does not carry it.

    Notes
    -----
    - [where the rows live](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
    """
    df = pd.DataFrame(rows)
    if "deleted" not in df:
        df["deleted"] = False
    df["deleted"] = df["deleted"].fillna(False).astype(bool)
    target = Path(layer, "groups")
    target.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target / f"{group}.parquet", index=False)


def write_connections(layer: str, rows: list[dict]) -> None:
    """Write `groups/connection.parquet`, including the `deleted` tombstone.

    Each row needs `entity` and `bus`; `role` describes the connection and keys
    nothing, so it is optional here.

    No component type - one file holds every type's rows. Appended rather than
    replaced, since a layer may write them a call at a time.

    Notes
    -----
    - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
    - [where the rows live](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
    """
    df = pd.DataFrame(rows)
    for col in ("scenario", "role"):
        if col not in df:
            df[col] = None
        df[col] = df[col].astype("string")
    if "deleted" not in df:
        df["deleted"] = False
    df["deleted"] = df["deleted"].fillna(False).astype(bool)

    lead = ["entity", "bus", "role", "scenario", "deleted"]
    ordered = lead + [c for c in df.columns if c not in lead]
    target = Path(layer, "groups")
    target.mkdir(parents=True, exist_ok=True)
    path = target / "connection.parquet"
    out = df[ordered]
    if path.exists():
        out = pd.concat([pd.read_parquet(path), out], ignore_index=True)
    out.to_parquet(path, index=False)


def tombstone_connection(layer: str, pairs: list[tuple[str, str]]) -> None:
    """Mark connections deleted in this layer, by `(entity, bus)`.

    Notes
    -----
    - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
    """
    write_connections(
        layer,
        [{"entity": name, "bus": bus, "deleted": True} for name, bus in pairs],
    )


def write_entity_type(layer: str, ctype: str, rows: list[dict]) -> None:
    """Write `dims/entity_type/<ctype>.parquet` *and* this type's entity rows.

    Membership and tombstones live on `dims/entity.parquet`, which the writer
    derives from the per-type frames - so a hand-built layer has to keep the
    two in step the way `write_record` does.

    Notes
    -----
    - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
    """
    df = pd.DataFrame(rows)
    df["entity_type"] = ctype
    if "deleted" not in df:
        df["deleted"] = False
    df["deleted"] = df["deleted"].fillna(False).astype(bool)

    lead = ["entity_type", "entity", "deleted"]
    ordered = lead + [c for c in df.columns if c not in lead]
    target = Path(layer, "dims", "entity_type")
    target.mkdir(parents=True, exist_ok=True)
    df[ordered].to_parquet(target / f"{ctype}.parquet", index=False)

    # Appended rather than replaced: several types land in one entity axis, and
    # a layer may write them one call at a time.
    axis = Path(layer, "dims", "entity.parquet")
    entities = df[["entity", "entity_type", "deleted"]]
    if axis.exists():
        entities = pd.concat([pd.read_parquet(axis), entities], ignore_index=True)
    entities.to_parquet(axis, index=False)


def tombstone(layer: str, ctype: str, names: list[str]) -> None:
    """Mark components deleted in this layer.

    Notes
    -----
    - [deletion](https://energy-models.github.io/datarecord/design/layers/#deletion)
    """
    write_entity_type(
        layer,
        ctype,
        [{"entity": n, "deleted": True} for n in names],
    )


def write_scenarios(layer: str, rows: list[dict]) -> None:
    """Write `dims/scenario.parquet`; each row needs `scenario` and `weight`."""
    df = pd.DataFrame(rows)
    target = Path(layer, "dims")
    target.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target / "scenario.parquet", index=False)


def write_periods(layer: str, rows: list[dict]) -> None:
    """Write `dims/period.parquet`; each row needs `period`."""
    df = pd.DataFrame(rows)
    target = Path(layer, "dims")
    target.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target / "period.parquet", index=False)


def write_snapshots(layer: str, rows: list[dict]) -> None:
    """Write `dims/snapshot.parquet`; each row needs `snapshot`.

    A `period` column makes it a nested axis, keyed by `(period,
    snapshot)` rather than by the timestamp alone.

    Notes
    -----
    - [within](https://energy-models.github.io/datarecord/design/schema/#within-an-axis-inside-an-axis)
    """
    df = pd.DataFrame(rows)
    df["snapshot"] = pd.Series(df["snapshot"]).astype("datetime64[ns]")
    if "period" in df:
        df["period"] = df["period"].astype("Int64")
    target = Path(layer, "dims")
    target.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target / "snapshot.parquet", index=False)


def write_axis(layer: str, dim: str, rows: list[dict]) -> None:
    """Write `dims/<dim>.parquet` from plain rows, whatever columns they carry.

    The generic form of `write_scenarios`/`write_periods`: an axis file is its
    key column plus whatever else it holds - a mapping's column, an attribute
    addressed by the axis alone.

    Notes
    -----
    - [the record format](https://energy-models.github.io/datarecord/design/format/)
    """
    target = Path(layer, "dims")
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target / f"{dim}.parquet", index=False)


def rename_components(n, ctype: str, suffix: str) -> None:
    """Suffix one type's member names, in `static` and every dynamic container.

    PyPSA's example networks scope names per component type - a `Load` named
    after its `Bus`, a `Generator` after its `Carrier` - which a record cannot
    represent, names being unique across types.
    `PyPSA.to_datarecord` rejects such a network rather than renaming it,
    so the suffix here is the test suite standing in for the caller that has to
    reconcile the two vocabularies.

    Both containers, because they are keyed by the same names: renaming only
    `static` would orphan every dynamic column, and so silently drop that
    attribute from the record. A stochastic network is keyed by
    `(scenario, name)`, so only the `name` level moves.

    The renamed level is cast back to the dtype it had: `rename` yields an
    `object` index where PyPSA's own is `str`, and `assert_networks_equal`
    compares index dtypes exactly - so without this the helper, not the code
    under test, would fail the round-trip.

    Notes
    -----
    - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    c = n.c[ctype]
    index = c.static.index
    nested = "name" in (index.names or []) and index.nlevels > 1
    if nested:
        level = index.get_level_values("name")
        renamed = {name: f"{name}{suffix}" for name in level}
        c.static.rename(index=renamed, level="name", inplace=True)
        c.static.index = c.static.index.set_levels(
            c.static.index.levels[index.names.index("name")].astype(level.dtype),
            level="name",
        )
        # Per *level*, which is what `rename` does on a MultiIndex - a
        # tuple-keyed mapping matches nothing and silently leaves the columns
        # pointing at names `static` no longer has.
        for frame in c.dynamic.values():
            frame.rename(columns=renamed, level="name", inplace=True)
        return
    renamed = {name: f"{name}{suffix}" for name in index}
    c.static.rename(index=renamed, inplace=True)
    c.static.index = c.static.index.astype(index.dtype)
    for frame in c.dynamic.values():
        frame.rename(columns=renamed, inplace=True)


def export_network(n, revision, con) -> None:
    """Write `n` as `revision`'s layer, through the record layer's own writer.

    Not `n.export_to_parquet`: that emits PyPSA's upstream manifest format,
    which is a different vocabulary from the schema a record declares.
    Going through `write_record` means a test record is written exactly as
    `blocks` writes one.

    Notes
    -----
    - [one schema per record](https://energy-models.github.io/datarecord/design/schema/#one-schema-per-record)
    """
    from datarecord.layered.write import write_record
    from datarecord.tools.pypsa import PyPSA

    write_record(revision.id, PyPSA.to_datarecord(n), con)


def write_schema(schema: Schema, base_uri: str | None = None) -> None:
    """Declare the record's one schema, beside the layers.

    Not per layer: a layer holds only data, so this writes the record-level
    `manifest.json` that every layer in the tree is read under.

    Notes
    -----
    - [one schema per record](https://energy-models.github.io/datarecord/design/schema/#one-schema-per-record)
    """
    record_write_schema(schema, base_uri)


def write_directory_schema(directory: str, schema: Schema) -> None:
    """Write `manifest.json` *inside* `directory`, for a standalone record.

    Notes
    -----
    - [one schema per record](https://energy-models.github.io/datarecord/design/schema/#one-schema-per-record)
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    Path(directory, "manifest.json").write_text(schema.model_dump_json())


def _default_attributes(
    dims: dict[str, nw.dtypes.DType], groups: dict[str, dict[str, str]]
):
    """The attributes tests write, declared over whichever dims are in play.

    Writing an attribute the schema does not declare is rejected, since its
    `dims` are what say which columns its file carries - so every attribute a
    test writes has to be declared, and these are the ones they write.

    Addressed over every declared dim rather than a narrower set, which is the
    widest shape and so the one that accepts any row a test writes.
    `efficiency` is the exception, being over the `connection` group where one
    is declared: that is what puts a `bus` column on its file.

    `weight` is the other, addressed by `scenario` alone - so it is a column of
    `dims/scenario.parquet` rather than a long row, and it is declared because
    `write_scenarios` writes that column and an axis file rejects one no
    declaration accounts for.
    """
    varying = {"entity", *dims}
    connection = "connection" if "connection" in groups else "entity"
    declared = {
        "p_nom": AttributeSpec(dtype=nw.Float64(), dims=varying),
        "e_nom": AttributeSpec(dtype=nw.Float64(), dims=varying),
        "p_max_pu": AttributeSpec(dtype=nw.Float64(), dims=varying),
        "p_min_pu": AttributeSpec(dtype=nw.Float64(), dims=varying),
        "marginal_cost": AttributeSpec(
            dtype=nw.Float64(), dims=varying, breakpoints=True
        ),
        "efficiency": AttributeSpec(dtype=nw.Float64(), dims={connection, *dims}),
    }
    if "scenario" in dims:
        declared["weight"] = AttributeSpec(
            dtype=nw.Float64(),
            dims={"scenario"},
            description="How much this scenario counts in the expectation.",
        )
    return declared


def schema(
    *,
    partial: set[str] = {"scenario"},
    attributes: dict[str, dict[str, AttributeSpec]] | None = None,
    dims: dict[str, nw.dtypes.DType] = {
        "snapshot": nw.Datetime(),
        "period": nw.Int64(),
        "scenario": nw.String(),
    },
    groups: dict[str, dict[str, str]] = {
        "connection": {"entity": "entity", "bus": "bus"}
    },
    within: dict[str, set[str]] | None = None,
) -> Schema:
    """A schema shaped like the PyPSA records most tests build on.

    Defaults match `PyPSA.to_datarecord`: the `entity` axis and a `connection`
    group over `(entity, bus)`, and three declared dims. Override `partial` to
    pin a different layering granularity, `dims` to declare another axis,
    `groups` to declare a different sparse relation, `within` to nest one axis
    inside another.

    `entity` and every group coordinate are declared dims and are `partial`:
    a layer patches one component's value, or one connection's, without
    restating the rest, which is what `partial` means. The schema requires it,
    so this supplies it rather than leaving each caller to.

    Notes
    -----
    - [the schema](https://energy-models.github.io/datarecord/design/schema/)
    - [groups](https://energy-models.github.io/datarecord/design/schema/#groups)
    - [within](https://energy-models.github.io/datarecord/design/schema/#within-an-axis-inside-an-axis)
    """
    nesting = within or {}
    # Callers declare per type, which is how a modelling framework thinks and
    # how the schema stores it: one record-wide spec per attribute, granted per
    # type in `types` with the facets on the grant. Splitting here is exactly
    # what a tool does on the way in.
    flat: dict[str, AttributeSpec] = {}
    per_type: dict[str, dict[str, TypeAttribute]] = {}
    for ctype, attrs in (attributes or {}).items():
        grants = per_type.setdefault(ctype, {})
        for attr, spec in attrs.items():
            flat.setdefault(
                attr,
                spec.model_copy(
                    update={"default": None, "unit": None, "description": None}
                ),
            )
            grants[attr] = TypeAttribute(
                default=spec.default, unit=spec.unit, description=spec.description
            )
    # Declared whether or not a caller named them: a test writing `p_max_pu`
    # needs it declared, and one passing `attributes=` is saying what a type
    # *carries* rather than shortening the record's vocabulary - so the default
    # attributes are granted to every declared type, as an untyped schema
    # already carries them everywhere.
    for attr, spec in _default_attributes(dims, groups).items():
        flat.setdefault(attr, spec)
        if per_type and spec.dims & ({"entity"} | set(groups)):
            for grants in per_type.values():
                grants.setdefault(attr, TypeAttribute())
    types = {c: TypeSpec(attributes=g) for c, g in per_type.items()}
    # A group's coordinates are dims like any other, so they are declared here
    # rather than assumed - which is what lets a caller pass a group over
    # coordinates that are not called `bus`.
    coordinates = {c for over in groups.values() for c in over}
    declared = {
        "entity": nw.String(),
        **{c: nw.String() for c in coordinates},
        **dims,
    }
    return Schema(
        groups={g: Group(over=over) for g, over in groups.items()}
        # `into` over `entity` alone is what makes `entity_type` the
        # entity-type axis (https://energy-models.github.io/datarecord/design/schema/#entity_type-the-axis-of-kinds).
        | {"entity_type": Group(over=["entity"], into="entity_type")},
        dimensions={
            d: Dimension(dtype=t, within=frozenset(nesting.get(d, set())))
            for d, t in declared.items()
        }
        # An enum where the caller declared per type - `types` requires one -
        # and a plain string otherwise, so untyped fixtures keep naming types
        # freely (https://energy-models.github.io/datarecord/design/schema/#entity_type-the-axis-of-kinds).
        | {
            "entity_type": Dimension(
                dtype=nw.Enum(sorted(types)) if types else nw.String()
            )
        },
        attributes=flat,
        types=types,
        # `partial` names value dims a layer patches per value; membership keys
        # (`entity`, a group's coordinates) are in the fold key by being
        # membership, not by being `partial` (https://energy-models.github.io/datarecord/design/read-path/#one-fold-for-every-axis).
        partial=frozenset(partial),
    )


def relation(revision, attribute: str):
    """The resolved long relation for one input attribute, as a DuckDB relation.

    A test helper rather than a `Revision` method: `Revision` presents its data
    through `.record` (a `Record`), and a DuckDB-shaped accessor beside it would
    duplicate `record.attributes[attr]` while inverting what `outputs` means -
    a relation on the revision against a `Frames` mapping on the record. Tests
    want relations because they assert on `.df()`, so the affordance lives here.
    """
    return revision.resolver.attribute(attribute)


def outputs(revision, attribute: str):
    """One result attribute as a DuckDB relation; outputs do not overlay.

    Notes
    -----
    - [outputs](https://energy-models.github.io/datarecord/design/read-path/#outputs)
    """
    return revision.resolver.attribute(attribute, "outputs")
