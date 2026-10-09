# SPDX-FileCopyrightText: datarecord contributors
#
# SPDX-License-Identifier: MIT

"""Writing a layer from long-format frames.

Notes
-----
- [writing a whole record](https://energy-models.github.io/datarecord/design/writing/)
"""

from pathlib import Path

import narwhals as nw
import pandas as pd
import pytest

from datarecord import Revision
from datarecord.duck import layer_dir
from datarecord.layered.resolve import read_schema
from datarecord.layered.revision import Record
from datarecord.layered.write import write_record
from datarecord.record import EMPTY, LazyFrames, RecordLike
from datarecord.schema import Schema
from datarecord.tools.pypsa import PyPSA
from tests.fixtures import export_network, relation, schema


class _Source:
    """A minimal `Record` over ready-made frames, counting each build."""

    def __init__(
        self,
        schema,
        attributes=None,
        entity_types=None,
        connections=None,
        outputs=None,
        dims=None,
    ):
        self._schema = schema
        self.built: list[str] = []
        self._attributes = attributes or {}
        self._entity_types = entity_types or {}
        self._connections = connections or {}
        self._outputs = outputs or {}
        self._dims = dims or {}

    @property
    def schema(self):
        return self._schema

    def _frames(self, mapping, tag):
        def build(key):
            self.built.append(f"{tag}:{key}")
            return nw.from_native(mapping[key]).lazy()

        return LazyFrames(tuple(mapping), build)

    @property
    def dims(self):
        return self._frames(self._dims, "dims") if self._dims else EMPTY

    @property
    def entity_types(self):
        return self._frames(self._entity_types, "entity_types")

    @property
    def groups(self):
        """Keyed by group, one frame each - the type is no coordinate of a group."""
        return self._frames(self._connections, "groups")

    @property
    def attributes(self):
        return self._frames(self._attributes, "attributes")

    @property
    def outputs(self):
        return self._frames(self._outputs, "outputs")

    def flags(self, ctype):
        return {}


_SCHEMA = schema()


def _long(**overrides) -> pd.DataFrame:
    """One long row carrying its attribute's own coordinates, and no others.

    Shaped from the spec rather than spelled: a source handing over a column
    the attribute is not addressed by is what `write_record` now rejects, so a
    helper that spelled every declared dim would be testing against a record no
    reader would accept.

    Notes
    -----
    - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
    """
    row = {
        "entity": "steel_dri",
        "bus": None,
        "snapshot": None,
        "scenario": None,
        "period": None,
        "attribute": "p_nom",
        "breakpoint": None,
        "value": 1.0,
    }
    row.update(overrides)
    columns = _SCHEMA.long_columns_for(str(row["attribute"]))
    return pd.DataFrame([{c: row[c] for c in columns}])


# -- the lazy mapping (https://energy-models.github.io/datarecord/design/format/) -------------------------------------------------


def test_source_is_explorable_without_building(con, base_uri):
    """Keys list, `in` answers, iteration repeats - none of it builds a frame."""
    source = _Source(_SCHEMA, attributes={"p_nom": _long(), "e_nom": _long()})

    assert list(source.attributes) == ["p_nom", "e_nom"]
    assert "p_nom" in source.attributes
    assert "nope" not in source.attributes
    assert len(source.attributes) == 2
    # Re-iterating works, unlike a generator, and still nothing is built.
    assert list(source.attributes) == ["p_nom", "e_nom"]
    assert source.built == []

    source.attributes["p_nom"]
    assert source.built == ["attributes:p_nom"]

    with pytest.raises(KeyError):
        source.attributes["nope"]


def test_write_record_builds_each_key_once(con, base_uri):
    """The writer looks up every key exactly once, and only what it writes."""
    revision = Revision.create(con)
    source = _Source(
        _SCHEMA,
        attributes={"p_nom": _long(), "e_nom": _long(attribute="e_nom")},
        entity_types={
            "Process": pd.DataFrame({"entity": ["steel_dri"], "scenario": [None]})
        },
    )
    write_record(revision.id, source, con)

    assert sorted(source.built) == [
        "attributes:e_nom",
        "attributes:p_nom",
        "entity_types:Process",
    ]


# -- creating a layer -------------------------------------------------------


def test_write_record_creates_a_new_layer(con, base_uri):
    """Files land where `layer_dir` says - data only, no schema.

    The record's one schema goes beside `layers/`, so a layer directory holds
    nothing but data. That is what keeps it a plain parquet directory a reader
    knowing nothing about layering can open.

    Notes
    -----
    - [one schema per record](https://energy-models.github.io/datarecord/design/schema/#one-schema-per-record)
    """
    revision = Revision.create(con)
    write_record(revision.id, _Source(_SCHEMA, attributes={"p_nom": _long()}), con)

    base = Path(layer_dir(revision.id))
    assert (base / "inputs" / "p_nom.parquet").exists()
    assert not (base / "manifest.json").exists()
    # Written once for the whole tree, and it is what the layer is read under.
    assert read_schema() == _SCHEMA


def test_no_layer_file_carries_order_key(con, base_uri, ac_dc, tmp_path):
    """`order_key` is the fold's answer about a frame, never a column of one.

    A source handing over *resolved* frames carries it - which is what
    committing a `WorkingRecord` to a `Directory` does, the record itself being
    what is written. Writing it would put a struct column in files the
    format promises a foreign reader can open, and would look like stored order
    where the fold always re-derives it from file order.

    Notes
    -----
    - [the owner map](https://energy-models.github.io/datarecord/design/read-path/#owner-map)
    - [writing a whole record](https://energy-models.github.io/datarecord/design/writing/)
    """
    from datarecord.mutable import Directory, WorkingRecord

    revision = Revision.create(con)
    write_record(revision.id, PyPSA.to_datarecord(ac_dc), con)

    staged = WorkingRecord(revision.record, con)
    staged.set("p_nom", 150.0, entity=["Manchester Wind"])
    out = str(tmp_path / "flat")
    staged.commit(Directory(out))

    written = sorted(Path(out).rglob("*.parquet"))
    assert written, "the commit must have written something to check"
    for path in written:
        columns = con.sql(f"SELECT * FROM read_parquet('{path}')").columns
        assert "order_key" not in columns, f"{path.relative_to(out)} carries order_key"


def test_a_per_type_member_file_does_not_repeat_its_type(con, base_uri, ac_dc):
    """`dims/entity_type/<T>.parquet` is indexed by `entity`, one column per attribute.

    The type is the file the rows are in, and it reaches a reader already:
    `dims/entity.parquet` carries `entity -> entity_type` so nobody has to
    glob. A column repeating it here would be a second copy, and the one that
    can disagree.

    The *axis* file keeps it, and that is a different file: there `entity_type`
    is the key, not a restatement of the path.

    Notes
    -----
    - [the record format](https://energy-models.github.io/datarecord/design/format/)
    - [the entity axis](https://energy-models.github.io/datarecord/design/format/#the-entity-axis)
    """
    revision = Revision.create(con)
    write_record(revision.id, PyPSA.to_datarecord(ac_dc), con)
    layer = Path(layer_dir(revision.id))

    members = layer / "dims" / "entity_type" / "Generator.parquet"
    columns = con.sql(f"SELECT * FROM read_parquet('{members}')").columns
    assert "entity" in columns, "still indexed by entity"
    assert "entity_type" not in columns, "the filename already says which type"

    # Derived from those files all the same, so the type still reaches a reader.
    axis = con.sql(f"SELECT * FROM read_parquet('{layer / 'dims' / 'entity.parquet'}')")
    assert "entity_type" in axis.columns
    types = {t for (t,) in axis.project("entity_type").distinct().fetchall()}
    assert "Generator" in types, "the type survives being off the member file"


def test_a_directory_target_carries_its_own_schema(con, base_uri, tmp_path):
    """A standalone record *is* one record, so its schema goes in the directory.

    Notes
    -----
    - [one schema per record](https://energy-models.github.io/datarecord/design/schema/#one-schema-per-record)
    """
    out = str(tmp_path / "standalone")
    write_record(None, _Source(_SCHEMA, attributes={"p_nom": _long()}), con, uri=out)

    assert (Path(out) / "manifest.json").exists()
    assert Record.at(out, con).schema == _SCHEMA

    # And it is that file answering, not the connection's root: a standalone
    # record is one whole record, so it must read the same through a connection
    # rooted somewhere with no manifest at all.
    from datarecord import duck

    elsewhere = duck.connect(base_uri=str(tmp_path / "unrelated"))
    try:
        assert read_schema(elsewhere) == Schema(), "the other root declares nothing"
        assert Record.at(out, elsewhere).schema == _SCHEMA
    finally:
        elsewhere.close()


def test_write_record_refuses_an_existing_layer(con, base_uri):
    """A whole-layer write never half-replaces what a record already holds.

    Notes
    -----
    - [the record format](https://energy-models.github.io/datarecord/design/format/)
    """
    revision = Revision.create(con)
    source = _Source(_SCHEMA, attributes={"p_nom": _long()})
    write_record(revision.id, source, con)

    with pytest.raises(FileExistsError, match="already exists"):
        write_record(revision.id, source, con)


def test_a_file_carries_only_its_own_attributes_coordinates(con, base_uri, ac_dc):
    """One attribute is one file, so one column set - not every declared dim.

    A component attribute has no `bus` column, a connection attribute does, and
    neither carries a dim it is not addressed by. The uniform prefix this
    replaces put an all-NULL `bus` on every file and a `period` column on
    attributes that never vary over one.

    Notes
    -----
    - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
    - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
    """
    revision = Revision.create(con)
    export_network(ac_dc, revision, con)
    inputs = Path(layer_dir(revision.id), "inputs")

    def columns(attribute: str) -> set[str]:
        return set(con.read_parquet(str(inputs / f"{attribute}.parquet")).columns)

    assert columns("p_max_pu") == {
        "entity",
        "snapshot",
        "attribute",
        "breakpoint",
        "value",
    }, "a component attribute carries `entity`, not the connection group's `bus`"

    efficiency = columns("efficiency")
    assert "bus" in efficiency, "a connection attribute carries the group's coordinates"
    assert "period" not in efficiency, "and no dim it is not addressed by"


# -- validation -------------------------------------------------------------


def test_write_record_rejects_an_undeclared_attribute(con, base_uri):
    """An attribute with no spec has no shape, so there is nothing to write it as.

    Its `dims` are what say which columns the file carries, so writing one the
    schema does not declare would put a file in `inputs/` whose column set no
    reader could derive. A *result* is exempt - a tool derives those from its
    own registry, never from the schema.

    Notes
    -----
    - [AttributeSpec](https://energy-models.github.io/datarecord/design/schema/#attributespec)
    """
    revision = Revision.create(con)
    source = _Source(_SCHEMA, attributes={"not_declared": _long(attribute="nope")})

    with pytest.raises(ValueError, match="not a declared attribute"):
        write_record(revision.id, source, con)


def test_write_record_rejects_a_coordinate_the_attribute_lacks(con, base_uri):
    """A column the attribute is not addressed by is a disagreement, not a spare.

    The read path projects an attribute's own coordinates, so a `bus` on a
    component attribute would be written and never read - and a source emitting
    one means something different by the attribute than the schema does.
    Reported rather than dropped, since silently narrowing would hide that.

    Notes
    -----
    - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
    """
    revision = Revision.create(con)
    wide = _long().assign(bus=None)
    source = _Source(_SCHEMA, attributes={"p_nom": wide})

    with pytest.raises(
        ValueError, match=r"carries columns \['bus'\].*not addressed by"
    ):
        write_record(revision.id, source, con)


def test_write_record_rejects_a_missing_long_column(con, base_uri):
    """A frame the fold could not resolve is refused before anything is written."""
    revision = Revision.create(con)
    short = _long().drop(columns=["breakpoint"])
    source = _Source(_SCHEMA, attributes={"p_nom": short})

    with pytest.raises(ValueError, match="missing long-schema columns.*breakpoint"):
        write_record(revision.id, source, con)
    assert not Path(layer_dir(revision.id)).exists()


def test_write_record_rejects_a_group_frame_missing_a_coordinate(con, base_uri):
    """A group's row is keyed by its coordinates, so one lacking them misresolves.

    Notes
    -----
    - [groups](https://energy-models.github.io/datarecord/design/schema/#groups)
    """
    revision = Revision.create(con)
    source = _Source(
        _SCHEMA,
        connections={"connection": pd.DataFrame({"entity": ["steel_dri"]})},  # no `bus`
    )

    with pytest.raises(ValueError, match="coordinates.*bus"):
        write_record(revision.id, source, con)


def test_write_record_rejects_a_name_two_types_share(con, base_uri):
    """Names are unique across every type, checked before anything is written.

    The attribute rows record no type, so two components sharing a name would
    silently share every attribute key - which is why this is enforced rather
    than assumed.

    Notes
    -----
    - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
    """
    revision = Revision.create(con)
    source = _Source(
        _SCHEMA,
        entity_types={
            "Process": pd.DataFrame({"entity": ["shared"], "scenario": [None]}),
            "Widget": pd.DataFrame({"entity": ["shared"], "scenario": [None]}),
        },
    )

    # The detail names the name and the types claiming it, in that order - the
    # message is what a caller acts on, so a transposed pair is a defect.
    with pytest.raises(ValueError, match=r"'shared' is a Process and a Widget"):
        write_record(revision.id, source, con)
    assert not Path(layer_dir(revision.id)).exists()


def test_write_record_accepts_one_name_per_type(con, base_uri):
    """The negative half: the same two types with distinct names write fine."""
    revision = Revision.create(con)
    source = _Source(
        _SCHEMA,
        entity_types={
            "Process": pd.DataFrame({"entity": ["a"], "scenario": [None]}),
            "Widget": pd.DataFrame({"entity": ["b"], "scenario": [None]}),
        },
    )

    write_record(revision.id, source, con)
    assert Path(layer_dir(revision.id)).exists()


def test_the_uniqueness_check_spans_backends(con, base_uri):
    """A `Record` may hand over one type as DuckDB and another as pandas.

    `WorkingRecord` does exactly this, mixing base frames with staged ones, so
    the check must not assume the component frames share a backend - `nw.concat`
    refuses a mixed list outright.

    Notes
    -----
    - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
    """
    revision = Revision.create(con)
    source = _Source(
        _SCHEMA,
        entity_types={
            # DuckDB-backed, and pandas-backed, colliding on `shared`.
            "Process": con.sql("SELECT 'shared' AS entity, NULL AS scenario"),
            "Widget": pd.DataFrame({"entity": ["shared"], "scenario": [None]}),
        },
    )

    with pytest.raises(ValueError, match="component types reuse names"):
        write_record(revision.id, source, con)


def test_write_record_rejects_a_nested_axis_without_its_parent(con, base_uri):
    """A `within` dim's file needs a column per parent, or the fold miskeys it.

    `snapshot within period` makes the axis key `(period, snapshot)`, so a
    `snapshot.parquet` carrying only timestamps would fold two periods'
    identically labelled hours into one row.

    Notes
    -----
    - [within](https://energy-models.github.io/datarecord/design/schema/#within-an-axis-inside-an-axis)
    """
    revision = Revision.create(con)
    nested = schema(within={"snapshot": {"period"}})
    source = _Source(
        nested,
        dims={"snapshot": pd.DataFrame({"snapshot": pd.to_datetime(["2020-01-01"])})},
    )

    with pytest.raises(ValueError, match="axis key columns.*period"):
        write_record(revision.id, source, con)
    assert not Path(layer_dir(revision.id)).exists()


def test_write_record_rejects_an_undeclared_axis_column(con, base_uri):
    """An axis file's payload is the schema's to state, like a long frame's.

    A column no declaration accounts for would be read back with no dtype and
    no meaning, so it is refused rather than carried along.

    Notes
    -----
    - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
    """
    revision = Revision.create(con)
    source = _Source(
        schema(),
        dims={"scenario": pd.DataFrame({"scenario": ["high"], "nonsense": [1.0]})},
    )

    with pytest.raises(ValueError, match="does not declare for the 'scenario' axis"):
        write_record(revision.id, source, con)
    assert not Path(layer_dir(revision.id)).exists()


def test_an_axis_carries_the_attributes_addressed_by_it_alone(con, base_uri):
    """`weight` over `scenario` alone is a column of that axis's file.

    Declared, so it round-trips with a dtype - which is what distinguishes it
    from the undeclared column above.

    Notes
    -----
    - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
    """
    revision = Revision.create(con)
    declared = schema()
    assert "weight" in declared.attributes_on("scenario")

    source = _Source(
        declared,
        dims={"scenario": pd.DataFrame({"scenario": ["high"], "weight": [0.4]})},
    )
    write_record(revision.id, source, con)

    axis = revision.resolver.dims.axes["scenario"].df()
    assert dict(zip(axis["scenario"], axis["weight"])) == {"high": 0.4}


# -- the PyPSA source (https://energy-models.github.io/datarecord/design/format/) ------------------------------------------------


def test_to_datarecord_lists_without_unpivoting(con, base_uri, ac_dc):
    """Key sets come off the network and its registry, so listing is cheap."""
    source = PyPSA.to_datarecord(ac_dc)

    assert isinstance(source, RecordLike)
    assert "Generator" in source.entity_types
    assert "connection" in source.groups
    assert "p_max_pu" in source.attributes
    # Non-varying attributes belong to `dims/entity_type/`, not `inputs/` (https://energy-models.github.io/datarecord/design/record/).
    assert "v_nom" not in source.attributes
    # A port attribute is one bus-keyed attribute, not one per port (https://energy-models.github.io/datarecord/design/record/#connections).
    assert "efficiency" in source.attributes
    assert "efficiency2" not in source.attributes


def test_write_then_build_round_trips(con, base_uri, ac_dc):
    """A network written by blocks and read back through `build` is unchanged.

    Distinct from `test_roundtrip.py`, which reads an `export_to_parquet`
    record: this exercises the writer and the connection collapse in one pass.

    Notes
    -----
    - [the record format](https://energy-models.github.io/datarecord/design/format/)
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    revision = Revision.create(con)
    write_record(revision.id, PyPSA.to_datarecord(ac_dc), con)

    assert not PyPSA.verify(revision.record)
    back = PyPSA.build(revision.record)

    for ctype in ("Bus", "Generator", "Link", "Line", "Load"):
        original, rebuilt = ac_dc.c[ctype].static, back.c[ctype].static
        assert list(rebuilt.index) == list(original.index), ctype
        # Every column survives, custom ones included (`Bus.country` has no
        # registry entry and must not be silently dropped).
        assert set(rebuilt.columns) == set(original.columns), ctype
        for column in original.columns:
            assert rebuilt[column].astype(str).equals(original[column].astype(str)), (
                ctype,
                column,
            )


def test_multi_port_links_round_trip_through_connections(con, base_uri, ac_dc):
    """`bus0`/`bus1` become connection rows and come back as columns.

    Notes
    -----
    - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    revision = Revision.create(con)
    write_record(revision.id, PyPSA.to_datarecord(ac_dc), con)

    # Stored bus-keyed, with a role from PyPSA's sign convention. One file for
    # every type, so the Links are reached by their entities.
    rows = con.read_parquet(layer_dir(revision.id) + "groups/connection.parquet").df()
    links = rows[rows["entity"].isin(ac_dc.c["Link"].static.index)]
    assert set(links["role"]) == {"input", "output"}
    assert set(links["bus"]) >= set(ac_dc.c["Link"].static["bus0"])

    back = PyPSA.build(revision.record)
    assert list(back.c["Link"].static["bus0"]) == list(ac_dc.c["Link"].static["bus0"])
    assert list(back.c["Link"].static["bus1"]) == list(ac_dc.c["Link"].static["bus1"])


def test_single_port_components_keep_their_unsuffixed_bus(con, base_uri, ac_dc):
    """A Generator's one `bus` is a connection too, and stays `bus`.

    Notes
    -----
    - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
    """
    revision = Revision.create(con)
    write_record(revision.id, PyPSA.to_datarecord(ac_dc), con)

    rows = con.read_parquet(layer_dir(revision.id) + "groups/connection.parquet").df()
    mine = rows[rows["entity"].isin(ac_dc.c["Generator"].static.index)]
    assert set(mine["role"]) == {"attached"}

    back = PyPSA.build(revision.record)
    assert list(back.c["Generator"].static["bus"]) == list(
        ac_dc.c["Generator"].static["bus"]
    )


def test_static_series_split_survives_the_writer(con, base_uri, ac_dc):
    """Only the components with a series get a `dynamic` column.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    revision = Revision.create(con)
    write_record(revision.id, PyPSA.to_datarecord(ac_dc), con)
    back = PyPSA.build(revision.record)

    assert sorted(back.c["Generator"].dynamic["p_max_pu"].columns) == sorted(
        ac_dc.c["Generator"].dynamic["p_max_pu"].columns
    )


def test_written_layer_overlays(con, base_uri, ac_dc):
    """A written layer is an ordinary layer: a child patches it as any other."""
    from tests.fixtures import write_input

    root = Revision.create(con)
    write_record(root.id, PyPSA.to_datarecord(ac_dc), con)
    root.materialise()

    child = root.child()
    write_input(
        layer_dir(child.id),
        "p_nom",
        [{"entity": "Manchester Wind", "value": 999.0}],
    )

    resolved = relation(child, "p_nom").filter("entity = 'Manchester Wind'").df()
    assert list(resolved["value"]) == [999.0]


def test_a_directory_target_stores_its_owner_map(con, base_uri, tmp_path):
    """A standalone record carries its own inputs map, with no `layer_uuid`.

    Written whole and renamed into place, the record's map cannot go stale, so
    folding it at write time is what spares every later open the aggregation
    over `inputs/`. The layer id is left out: it derives from the directory's
    location, which the rename out of staging - or any later move - changes,
    so the reader stamps its own.

    Notes
    -----
    - [the owner map](https://energy-models.github.io/datarecord/design/read-path/#owner-map)
    """
    out = tmp_path / "standalone"
    write_record(
        None, _Source(_SCHEMA, attributes={"p_nom": _long()}), con, uri=str(out)
    )

    stored = con.read_parquet(str(out / "owner_map" / "inputs.parquet"))
    assert "layer_uuid" not in stored.columns
    rows = stored.df()
    assert list(zip(rows["entity"], rows["attribute"])) == [("steel_dri", "p_nom")]


def test_a_tree_layer_stores_no_owner_map(con, base_uri):
    """A tree layer's map depends on its ancestors; `materialise` is its cache.

    Notes
    -----
    - [materialised node caches](https://energy-models.github.io/datarecord/design/layers/#materialised-node-caches)
    """
    revision = Revision.create(con)
    write_record(revision.id, _Source(_SCHEMA, attributes={"p_nom": _long()}), con)
    assert not Path(layer_dir(revision.id), "owner_map").exists()
