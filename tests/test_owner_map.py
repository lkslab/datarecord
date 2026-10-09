# SPDX-FileCopyrightText: datarecord contributors
#
# SPDX-License-Identifier: MIT

"""The fold, the cache and persistence.

Notes
-----
- [materialised node caches](https://energy-models.github.io/datarecord/design/layers/#materialised-node-caches)
- [the owner map](https://energy-models.github.io/datarecord/design/read-path/#owner-map)
- [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
"""

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from datarecord import NewChild, Revision, WorkingRecord
from datarecord.duck import layer_dir, resolved_dir, union_all_by_name
from datarecord.layered import resolve
from datarecord.layered.resolve import write_schema
from datarecord.layered.revision import Record
from datarecord.layered.sources import DirectorySource, LayerSource, ParquetLayer
from datarecord.layered.write import write_record
from datarecord.schema import Schema
from datarecord.tools.pypsa import PyPSA
from tests.fixtures import export_network, tombstone, write_input


@pytest.fixture
def parent(con, base_uri, ac_dc):
    revision = Revision.create(con)
    export_network(ac_dc, revision, con)
    revision.materialise()
    return revision


def test_union_all_by_name_folds_every_relation(con):
    """Three arms, not two: the fold's union binds `u`/`rel` by name.

    Two relations pass whatever the loop body does, since the first pairing is
    the only one. Three is what catches the failure mode the helper's variables
    invite - a scan that re-reads `rels[0]` instead of advancing would give
    `[1, 1]` here, with no error to point at it.

    Notes
    -----
    - [resolving a relation](https://energy-models.github.io/datarecord/design/read-path/#resolving-a-relation)
    """
    rels = [con.sql(f"SELECT {i} AS x") for i in (1, 2, 3)]
    got = union_all_by_name(rels, con).fetchall()
    assert sorted(v for (v,) in got) == [1, 2, 3]


def test_union_all_by_name_fills_a_missing_column_with_null(con):
    """By *name*, so an arm lacking a column reads NULL there.

    What lets a layer written before `bus`/`breakpoint` existed still resolve,
    and a persisted owner map survive a newly declared dim.

    Notes
    -----
    - [Flags](https://energy-models.github.io/datarecord/design/record/#flags)
    - [versioning](https://energy-models.github.io/datarecord/design/schema/#versioning)
    """
    rels = [con.sql("SELECT 1 AS x, 'a' AS y"), con.sql("SELECT 2 AS x")]
    got = union_all_by_name(rels, con).fetchall()
    assert sorted(got, key=lambda r: r[0]) == [(1, "a"), (2, None)]


def keys(revision, con):
    """The inputs map's keys: `(name, attribute)`, no type.

    The map is tombstone-pruned in the fold (`fold_inputs` anti-joins each
    membership's deletions), so a key whose component or group tuple was deleted
    is already gone from it - the read needs no further gating.

    Notes
    -----
    - [the owner map](https://energy-models.github.io/datarecord/design/read-path/#owner-map)
    - [deletion](https://energy-models.github.io/datarecord/design/layers/#deletion)
    """
    df = revision.resolver.inputs.df()
    return {(r["entity"], str(r.attribute)) for _, r in df.iterrows()}


def entity_names(revision):
    return set(revision.resolver.entity_axis.df()["entity"])


def test_root_map_is_its_own_layer(con, parent):
    """Every key of a root's inputs map points at the root itself.

    The entity axis has no `layer_uuid` - it folds like an axis, its winning row
    the whole row in one file (https://energy-models.github.io/datarecord/design/read-path/#one-fold-for-every-axis).
    """
    om = parent.resolver
    assert set(om.inputs.df()["layer_uuid"]) == {parent.id}
    assert "layer_uuid" not in om.entity_axis.columns
    assert ("Manchester Wind", "p_max_pu") in keys(parent, con)


def test_materialise_writes_the_map_under_resolved(con, parent):
    """`materialise` writes the maps under `resolved/`, not at the layer root.

    Notes
    -----
    - [materialised node caches](https://energy-models.github.io/datarecord/design/layers/#materialised-node-caches)
    """
    assert Path(resolved_dir(parent.id), "owner_map", "inputs.parquet").exists()
    # The entity axis folds like an axis now, materialised under `dims/`, not as
    # an owner map (https://energy-models.github.io/datarecord/design/read-path/#one-fold-for-every-axis).
    assert Path(resolved_dir(parent.id), "dims", "entity.parquet").exists()
    assert not Path(resolved_dir(parent.id), "owner_map", "entities.parquet").exists()
    # The cache shares the record's directory but stays out of the layer's own
    # namespace, so a reader that knows nothing about layering still sees a
    # plain parquet directory: every glob into a layer is single-level, so nothing
    # under `resolved/` is reachable by one (https://energy-models.github.io/datarecord/design/layers/#deletion).
    assert not Path(layer_dir(parent.id), "owner_map").exists()
    # The globs the fold and `Record.at` actually use must not reach a
    # cached file.
    layer = Path(layer_dir(parent.id))
    reachable = {
        p
        for pattern in (
            "*.parquet",
            "inputs/*.parquet",
            "outputs/*.parquet",
            "dims/*.parquet",
            "dims/*/*.parquet",
        )
        for p in layer.glob(pattern)
    }
    assert reachable
    assert not any("resolved" in p.parts for p in reachable)


def test_last_writer_wins_per_key(con, parent):
    """A child's key overrides the parent's, others stay with the parent."""
    child = parent.child()
    write_input(
        layer_dir(child.id),
        "p_max_pu",
        [{"entity": "Manchester Wind", "value": 0.42}],
    )
    df = child.resolver.inputs.df()

    def owner_of(name, attr):
        sel = df[(df["entity"] == name) & (df["attribute"].astype(str) == attr)]
        return sel["layer_uuid"].iloc[0]

    assert owner_of("Manchester Wind", "p_max_pu") == child.id
    assert owner_of("Manchester Wind", "marginal_cost") == parent.id
    assert owner_of("Norway Wind", "p_max_pu") == parent.id


def test_tombstone_removes_all_attributes(con, parent):
    """A tombstone removes every attribute and the component row of the component.

    Notes
    -----
    - [deletion](https://energy-models.github.io/datarecord/design/layers/#deletion)
    """
    before = {k for k in keys(parent, con) if k[0] == "Norway Gas"}
    assert len(before) > 1
    assert "Norway Gas" in entity_names(parent)

    child = parent.child()
    tombstone(layer_dir(child.id), "Generator", ["Norway Gas"])
    assert not {k for k in keys(child, con) if k[0] == "Norway Gas"}
    assert "Norway Gas" not in entity_names(child)


def test_live_fold_is_cached_per_connection(con, parent):
    """A node with no materialised cache folds once and caches the table.

    Nothing invalidates it: layers are write-once, so a fold's inputs
    cannot change under it.

    Notes
    -----
    - [materialised node caches](https://energy-models.github.io/datarecord/design/layers/#materialised-node-caches)
    - [open questions](https://energy-models.github.io/datarecord/design/open-questions/)
    """
    child = parent.child()
    om = child.resolver
    om.inputs.fetchall()
    # Only `inputs` keeps an owner-map table; the entity axis folds like an axis.
    assert con.execute(
        "SELECT 1 FROM duckdb_tables() WHERE table_name = ?",
        [f"owner_map_inputs_{child.id.hex}"],
    ).fetchone()


def test_materialising_does_not_change_the_map(con, parent):
    """`materialise` is purely additive: same answer, fewer layers read.

    Notes
    -----
    - [materialised node caches](https://energy-models.github.io/datarecord/design/layers/#materialised-node-caches)
    """
    child = parent.child()
    write_input(
        layer_dir(child.id),
        "p_max_pu",
        [{"entity": "Manchester Wind", "value": 0.42}],
    )
    live = keys(child, con)
    child.materialise()
    assert keys(child, con) == live


def test_a_removed_cache_falls_back_to_the_fold(con, parent):
    """A cache is an optimisation, so losing it costs work rather than answers.

    The old model had to raise here: a closed node's ancestry was truncated to
    itself, so re-folding would have silently dropped every ancestor. Gating on
    the cache's presence instead means the untruncated path is still available.

    Notes
    -----
    - [materialised node caches](https://energy-models.github.io/datarecord/design/layers/#materialised-node-caches)
    """
    child = parent.child()
    write_input(
        layer_dir(child.id),
        "p_min_pu",
        [{"entity": "Norway Gas", "value": 0.1}],
    )
    child.materialise()
    expected = keys(child, con)

    shutil.rmtree(Path(resolved_dir(child.id), "owner_map"))
    fresh = Revision.get(child.id, con)
    assert keys(fresh, con) == expected


def test_a_materialised_node_reads_the_same_as_an_unmaterialised_one(con, parent):
    """The invariant the base/source split protects: a value read through a
    materialised ancestor equals the same value re-folded from the own layers.

    `parent` is materialised, so the child's fold seeds from `resolved/`. Removing
    that cache forces the fold to walk the parent's own layer instead - a
    different code path to the same answer, which is the whole point of the seed
    standing for the fold above it.

    Notes
    -----
    - [materialised node caches](https://energy-models.github.io/datarecord/design/layers/#materialised-node-caches)
    - [resolving a relation](https://energy-models.github.io/datarecord/design/read-path/#resolving-a-relation)
    """
    child = parent.child()
    write_input(
        layer_dir(child.id),
        "p_max_pu",
        [{"entity": "Manchester Wind", "value": 0.42}],
    )

    def read() -> list[str]:
        rel = Revision.get(child.id, con).resolver.attribute("p_max_pu")
        return sorted(repr(r) for r in rel.df().itertuples(index=False))

    seeded = read()
    shutil.rmtree(Path(resolved_dir(parent.id)))
    refolded = read()
    assert refolded == seeded, "the seed carries the fold above it"


def test_any_node_may_be_a_parent(con, parent):
    """A layer is write-once, so no node needs preparing to branch from.

    Notes
    -----
    - [a layer's data is write-once](https://energy-models.github.io/datarecord/design/layers/#a-layers-data-is-write-once)
    """
    child = parent.child()
    write_input(
        layer_dir(child.id),
        "p_max_pu",
        [{"entity": "Manchester Wind", "value": 0.42}],
    )
    grandchild = child.child()
    # The child was never materialised, yet its layer resolves for the
    # grandchild all the same.
    df = grandchild.resolver.inputs.df()
    row = df[(df["entity"] == "Manchester Wind") & (df["attribute"] == "p_max_pu")]
    assert set(row["layer_uuid"]) == {child.id}


def test_ancestry_is_root_first(con, parent):
    """`ancestry` returns the root->node path in resolution order.

    Notes
    -----
    - [layered resolution](https://energy-models.github.io/datarecord/design/layers/)
    """
    child = parent.child()
    child.materialise()
    grandchild = child.child()
    assert grandchild.ancestry() == [parent.id, child.id, grandchild.id]


def test_a_record_with_no_manifest_folds(con, base_uri):
    """A record that declares no schema resolves to an empty map, not a crash.

    `Schema()` is "no manifest yet", which is what a record reads before
    anything has been written to it. The map's flag columns are structs with a
    field per declared dim and DuckDB has no empty struct, so this is
    the one path where there are none to declare.

    Notes
    -----
    - [one schema per record](https://energy-models.github.io/datarecord/design/schema/#one-schema-per-record)
    - [the owner map](https://energy-models.github.io/datarecord/design/read-path/#owner-map)
    """
    revision = Revision.create(con)
    assert revision.record.schema.dims == ()

    inputs = revision.resolver.inputs
    assert "varies" in inputs.columns
    assert inputs.fetchall() == []


def test_a_parquet_layer_locates_a_layers_files(base_uri):
    """The fold names files and the source says where they are.

    `layer_dir` is what a `ParquetLayer` derives from, so this pins the seam
    rather than the layout: a reader asks for `inputs/p_nom.parquet` and never
    builds the path itself.

    Notes
    -----
    - [the record format](https://energy-models.github.io/datarecord/design/format/)
    """
    revision_id = uuid4()
    source = ParquetLayer(revision_id, Schema())
    assert isinstance(source, LayerSource), (
        "structural, so no import is needed to be one"
    )

    assert source.uri() == layer_dir(revision_id), "empty is the layer root"
    assert source.uri("inputs/p_nom.parquet") == (
        layer_dir(revision_id) + "inputs/p_nom.parquet"
    )
    # A glob is a path like any other: the source neither parses nor validates.
    assert source.uri("inputs/*.parquet").endswith("inputs/*.parquet")


def test_a_parquet_layer_takes_the_base_it_was_given(tmp_path):
    """Two records on two roots locate their layers apart, as `layer_dir` does."""
    revision_id = uuid4()
    root = str(tmp_path / "elsewhere")
    assert ParquetLayer(revision_id, Schema(), base_uri=root).uri(
        "dims/entity.parquet"
    ) == (layer_dir(revision_id, root) + "dims/entity.parquet")


def test_a_directory_source_derives_its_layer_id_from_where_it_is():
    """A directory has no revision to be stamped with, so its location is its identity.

    Derived rather than allocated, which is what makes it the *same* layer in
    every process and every reader - the fold keys a source by UUID, and a
    per-reader one would make two readings of one directory disagree about
    which layer they are reading. `uuid5` is what pins that across processes; a
    stable-per-process id would pass an in-process comparison and still be
    wrong for a materialised map read back later.

    Notes
    -----
    - [one record over one fold](https://energy-models.github.io/datarecord/design/read-path/#one-record-over-one-fold)
    """
    a = DirectorySource("/records/one/", Schema())
    assert a.layer_id == DirectorySource("/records/one/", Schema()).layer_id, (
        "same place"
    )
    assert a.layer_id != DirectorySource("/records/two/", Schema()).layer_id, (
        "different place"
    )
    # The literal value, so the derivation cannot drift silently: a layer id
    # that changed between versions would orphan every materialised map naming
    # the old one.
    assert str(a.layer_id) == "dbc5401e-335a-506d-89db-395e6ea37662"
    assert isinstance(a, LayerSource), "structural, with `layer_id` a property"


def _largest_intermediate(con, rel):
    """The most rows any operator produced while DuckDB executed `rel`."""
    profile = json.loads(
        con.sql("EXPLAIN (ANALYZE, FORMAT JSON) FROM rel").fetchall()[0][1]
    )

    def largest(node):
        return max(
            [node.get("operator_cardinality") or 0]
            + [largest(child) for child in node.get("children", [])]
        )

    return largest(profile)


def test_a_read_joins_no_more_rows_than_the_layers_hold(con, base_uri):
    """The owner-map join is keyed on a partial dim, not filtered on it afterwards.

    The join matched a partial dim with `raw IS NULL OR raw IS NOT DISTINCT FROM
    owned`, which DuckDB cannot hash. It joined on `entity` and `layer_uuid`
    only and filtered on `snapshot` after the join, so one layer of `T`
    snapshots per entity produced `T * T` rows per entity before the filter,
    and a read slowed with the square of the series length.

    Notes
    -----
    - [resolving a relation](https://energy-models.github.io/datarecord/design/read-path/#resolving-a-relation)
    """
    write_schema(
        Schema(
            dimensions={
                "entity": {"dtype": "String"},
                "entity_type": {"dtype": "String"},
                "snapshot": {"dtype": "Int64"},
            },
            groups={"entity_type": {"over": ["entity"], "into": "entity_type"}},
            attributes={
                "p_max_pu": {"dtype": "Float64", "dims": ["entity", "snapshot"]}
            },
            partial=frozenset({"snapshot"}),
        )
    )
    names, snapshots = ["wind", "solar"], 200
    staged = WorkingRecord(Revision.create(con).record, con)
    staged.add("Generator", pd.DataFrame({"entity": names}))
    staged.set(
        "p_max_pu",
        pd.DataFrame(
            {
                "entity": [n for n in names for _ in range(snapshots)],
                "snapshot": list(range(snapshots)) * len(names),
                "value": 0.5,
            }
        ),
    )
    revision = staged.commit(NewChild())

    rel = revision.record.attributes["p_max_pu"].to_native()
    stored = len(names) * snapshots
    assert rel.count("*").fetchone() == (stored,), "one row per stored value"
    assert _largest_intermediate(con, rel) <= stored, (
        "no operator produces more rows than the layer holds"
    )


# -- the stored owner map (standalone records) --------------------------------


@pytest.fixture
def standalone(con, base_uri, ac_dc, tmp_path):
    """One network written whole as a standalone record directory."""
    out = tmp_path / "standalone"
    write_record(None, PyPSA.to_datarecord(ac_dc), con, uri=str(out))
    return out


def test_a_standalone_record_reads_its_stored_map(con, standalone, monkeypatch):
    """`Record.at` answers off the stored file; the live fold never runs.

    The point of storing the map: opening a standalone record costs a read of
    one small file rather than an aggregation over every `inputs/` file, per
    connection.

    Notes
    -----
    - [the owner map](https://energy-models.github.io/datarecord/design/read-path/#owner-map)
    """

    def boom(*args, **kwargs):
        msg = "the stored map should answer, not a live fold"
        raise AssertionError(msg)

    monkeypatch.setattr(resolve, "fold_inputs", boom)
    record = Record.at(str(standalone), con)
    assert "p_max_pu" in record.attributes
    assert "p_max_pu" in record.flags("Generator")


def test_a_moved_standalone_record_still_resolves_rows(con, standalone, tmp_path):
    """The reader stamps its own `layer_uuid`, so the map survives a move.

    A directory's layer id derives from its location (and the record is even
    *written* under a staging path), so a stored id would name a layer no
    reader holds and every row read would come back empty.

    Notes
    -----
    - [one record over one fold](https://energy-models.github.io/datarecord/design/read-path/#one-record-over-one-fold)
    """
    moved = tmp_path / "elsewhere"
    standalone.rename(moved)
    rows = Record.at(str(moved), con).resolver.attribute("p_max_pu").df()
    assert not rows.empty


def test_a_record_without_a_stored_map_folds_and_agrees(con, standalone):
    """An old record (no stored map) still opens, folding live to the same map.

    Notes
    -----
    - [the owner map](https://energy-models.github.io/datarecord/design/read-path/#owner-map)
    """

    def snapshot(record):
        inputs = record.resolver.inputs.df()
        return (
            {(r["entity"], str(r.attribute)) for _, r in inputs.iterrows()},
            record.flags("Generator"),
        )

    stored_keys, stored_flags = snapshot(Record.at(str(standalone), con))
    shutil.rmtree(standalone / "owner_map")
    folded_keys, folded_flags = snapshot(Record.at(str(standalone), con))
    assert folded_keys == stored_keys
    assert folded_flags == stored_flags
