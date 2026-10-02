# SPDX-FileCopyrightText: datarecord contributors
#
# SPDX-License-Identifier: MIT

"""Overlay semantics over a parent/child pair.

Notes
-----
- [layered resolution](https://energy-models.github.io/datarecord/design/layers/)
- [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
"""

import narwhals as nw
import pandas as pd
import pytest

from datarecord import Revision
from datarecord.duck import layer_dir
from datarecord.layered.resolve import (
    Resolver,
    read_schema,
    sources_to_read,
    write_schema,
)
from datarecord.layered.revision import ancestry
from datarecord.layered.sources import ParquetLayer
from datarecord.layered.write import write_record
from datarecord.record import EMPTY
from datarecord.schema import AttributeSpec, TypeAttribute
from datarecord.tools.pypsa import PyPSA
from tests.fixtures import export_network, outputs, relation, tombstone, write_input


@pytest.fixture
def parent(con, base_uri, ac_dc):
    revision = Revision.create(con)
    export_network(ac_dc, revision, con)
    revision.materialise()
    return revision


def test_child_overwrites_component(con, parent):
    """A child's rows replace all of that component's rows for the attribute.

    Notes
    -----
    - [partial](https://energy-models.github.io/datarecord/design/schema/#partial-the-granularity-of-an-override)
    """
    child = parent.child()
    write_input(
        layer_dir(child.id),
        "p_max_pu",
        [{"entity": "Manchester Wind", "value": 0.42}],
    )

    df = relation(child, "p_max_pu").df()
    manchester = df[df["entity"] == "Manchester Wind"]
    # The parent's 10 series rows are gone, replaced by the child's single row.
    assert len(manchester) == 1
    assert manchester["value"].iloc[0] == 0.42
    assert pd.isna(manchester["snapshot"].iloc[0])

    # Siblings on the same layer are untouched.
    assert len(df[df["entity"] == "Norway Wind"]) == 10


def test_child_overwrite_reaches_model(con, parent):
    """The overwrite turns a series component into a static one in the model."""
    child = parent.child()
    write_input(
        layer_dir(child.id),
        "p_max_pu",
        [{"entity": "Manchester Wind", "value": 0.42}],
    )

    n = PyPSA.build(child.record)
    assert n.c["Generator"].static.loc["Manchester Wind", "p_max_pu"] == 0.42
    assert "Manchester Wind" not in n.c["Generator"].dynamic["p_max_pu"].columns
    assert "Norway Wind" in n.c["Generator"].dynamic["p_max_pu"].columns


def test_tombstone_removes_component(con, parent):
    """A tombstone removes the component from every attribute and dimension.

    Notes
    -----
    - [deletion](https://energy-models.github.io/datarecord/design/layers/#deletion)
    """
    child = parent.child()
    tombstone(layer_dir(child.id), "Generator", ["Norway Gas"])

    om = child.resolver.entity_axis.df()
    assert "Norway Gas" not in set(om["entity"])
    assert "Norway Gas" in set(parent.resolver.entity_axis.df()["entity"])

    n = PyPSA.build(child.record)
    assert "Norway Gas" not in n.c["Generator"].static.index
    assert "Norway Wind" in n.c["Generator"].static.index


def test_child_adds_attribute(con, parent):
    """A child may write an attribute no ancestor had."""
    child = parent.child()
    write_input(
        layer_dir(child.id),
        "p_min_pu",
        [{"entity": "Norway Gas", "value": 0.1}],
    )

    n = PyPSA.build(child.record)
    assert n.c["Generator"].static.loc["Norway Gas", "p_min_pu"] == 0.1
    # Untouched generators keep the catalog default.
    assert n.c["Generator"].static.loc["Norway Wind", "p_min_pu"] == 0.0


def test_sibling_branch_unaffected(con, parent):
    """A tombstone only affects the branch that carries it.

    Notes
    -----
    - [deletion](https://energy-models.github.io/datarecord/design/layers/#deletion)
    """
    deleting = parent.child()
    tombstone(layer_dir(deleting.id), "Generator", ["Norway Gas"])
    sibling = parent.child()

    assert "Norway Gas" not in set(deleting.resolver.entity_axis.df()["entity"])
    assert "Norway Gas" in set(sibling.resolver.entity_axis.df()["entity"])


def test_grandchild_resolves_through_ancestry(con, parent):
    """Resolution walks the whole root->node path, nearest layer winning."""
    child = parent.child()
    write_input(
        layer_dir(child.id),
        "p_max_pu",
        [{"entity": "Manchester Wind", "value": 0.42}],
    )
    child.materialise()

    grandchild = child.child()
    write_input(
        layer_dir(grandchild.id),
        "p_max_pu",
        [{"entity": "Manchester Wind", "value": 0.99}],
    )

    df = relation(grandchild, "p_max_pu").df()
    manchester = df[df["entity"] == "Manchester Wind"]
    assert len(manchester) == 1
    assert manchester["value"].iloc[0] == 0.99
    assert len(df[df["entity"] == "Norway Wind"]) == 10


def test_closed_child_reads_own_resolver(con, parent):
    """Reading a closed non-root record uses its own persisted dims/manifest/map.

    Its `ancestry_since_closed` is just itself, so the raw layer (which has
    no `dims/` or `manifest.json` of its own here) cannot be the source.

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
    child.materialise()

    reloaded = Revision.get(child.id, con)
    n = PyPSA.build(reloaded.record)
    assert n.c["Generator"].static.loc["Manchester Wind", "p_max_pu"] == 0.42

    df = relation(reloaded, "p_max_pu").df()
    assert df[df["entity"] == "Manchester Wind"]["value"].tolist() == [0.42]


def test_outputs_do_not_overlay(con, parent):
    """Results come from the node's own layer only.

    Notes
    -----
    - [outputs](https://energy-models.github.io/datarecord/design/read-path/#outputs)
    """
    child = parent.child()
    assert outputs(child, "p").df().empty


def test_resolved_reads_same_as_unresolved(con, parent):
    """A materialised node reads identically to the same node folded from its layers.

    The invariant the base/source split protects: a materialised ancestor's
    resolved fold, read as the base, gives every reader the same answer as
    re-folding the whole ancestry from the root. Materialise a grandchild, then
    build one node cache the truncated way (base = the materialised parent) and
    one the long way (every layer as its own `ParquetLayer`, no truncation), and
    assert they agree on the owner map, the entity axis, every attribute
    relation, and the group frames.
    """
    child = parent.child()
    write_input(
        layer_dir(child.id),
        "p_max_pu",
        [{"entity": "Manchester Wind", "value": 0.42}],
    )
    child.materialise()

    grandchild = child.child()
    write_input(
        layer_dir(grandchild.id),
        "p_max_pu",
        [{"entity": "Manchester Wind", "value": 0.99}],
    )

    # `unresolved` folds the whole ancestry from the root: every layer read from
    # its own directory, no base short-circuit, even though the ancestors are
    # materialised. `_Unmaterialised` forces `materialised()` to `None` so the
    # fold cannot take a base and must re-derive what the base would carry.
    class _Unmaterialised(ParquetLayer):
        def materialised(self, con, schema):  # noqa: ARG002
            return None

    schema = read_schema(con)
    full = ancestry(con, grandchild.id)
    truncated = Resolver(grandchild.id, sources_to_read(full, con, schema), con, schema)
    unresolved = Resolver(
        grandchild.id, [_Unmaterialised(uid, schema, con) for uid in full], con, schema
    )

    def ownership(nc):
        return sorted(
            nc.inputs.project("attribute, entity, layer_uuid").fetchall(),
            key=str,
        )

    assert ownership(truncated) == ownership(unresolved), (
        "the owner map folded through the base matches folding from the root"
    )

    t_axis, u_axis = truncated.entity_axis, unresolved.entity_axis
    assert t_axis is not None and u_axis is not None
    assert set(t_axis.df()["entity"]) == set(u_axis.df()["entity"]), (
        "the resolved entity axis is the same either way"
    )

    for attr in truncated.attributes():
        a = sorted(truncated.attribute(attr).fetchall(), key=str)
        b = sorted(unresolved.attribute(attr).fetchall(), key=str)
        assert a == b, f"{attr} resolves the same through the base as from the root"

    manchester = truncated.attribute("p_max_pu").df()
    assert manchester[manchester["entity"] == "Manchester Wind"]["value"].tolist() == [
        0.99
    ], "grandchild's value wins over the materialised base"


def test_a_new_attribute_is_a_schema_amendment(con, parent):
    """Adding an attribute amends the record's one schema, not a layer's.

    A schema is not layered data: folding it would let a layer redefine what
    an attribute means, and make the schema unknowable without walking the
    ancestry. So the amendment lands beside the layers, and every layer in the
    tree - including ones already written - is read under it.

    Notes
    -----
    - [one schema per record](https://energy-models.github.io/datarecord/design/schema/#one-schema-per-record)
    """
    amended = read_schema()
    # Declared once, record-wide, then granted to the type that carries it -
    # the two halves an amendment has, with the default on the grant.
    amended.attributes["p_min_pu"] = AttributeSpec(
        dtype=nw.Float64(), dims={"entity", "snapshot"}
    )
    was = amended.types["Generator"]
    amended.types["Generator"] = was.model_copy(
        update={
            "attributes": {**was.attributes, "p_min_pu": TypeAttribute(default=0.25)}
        }
    )
    write_schema(amended)

    # Adding an attribute is compatible, so the layers written before the
    # amendment stay readable (https://energy-models.github.io/datarecord/design/schema/#versioning).
    assert amended.compatible_with(read_schema()) == []

    child = parent.child()
    write_input(
        layer_dir(child.id),
        "p_min_pu",
        [{"entity": "Norway Gas", "value": 0.1}],
    )
    n = PyPSA.build(child.record)
    assert n.c["Generator"].static.loc["Norway Gas", "p_min_pu"] == 0.1
    # And the amendment is visible from the record, not just from the layer
    # that happens to carry a row for it.
    assert "p_min_pu" in child.record.schema.attributes_for("Generator")


def test_a_schema_narrowing_is_refused(con, parent, ac_dc):
    """A layer cannot redefine what an attribute means.

    Notes
    -----
    - [one schema per record](https://energy-models.github.io/datarecord/design/schema/#one-schema-per-record)
    - [versioning](https://energy-models.github.io/datarecord/design/schema/#versioning)
    """
    narrowed = read_schema()
    narrowed.attributes["p_max_pu"] = AttributeSpec(
        dtype=nw.Float64(), dims=frozenset()
    )

    class _Narrowed:
        """The record's own source, with one attribute's dims taken away."""

        schema = narrowed
        dims = EMPTY
        entity_types = EMPTY
        groups: dict = {}
        attributes = EMPTY
        outputs = EMPTY

        def flags(self, ctype):
            return {}

    child = parent.child()
    with pytest.raises(ValueError, match="no longer varies over"):
        write_record(child.id, _Narrowed(), con)


def test_member_order_survives_closed_intermediate(con, parent, ac_dc):
    """Component order still follows the true owning layer through a closed
    intermediate node that changes nothing about membership.

    `members()` resolves straight from the owner map's `order_key`, not from
    a per-file depth lookup over `ancestry_since_closed` - which wouldn't
    even find `parent` here, since `middle` sits between it and `grandchild`.

    Notes
    -----
    - [materialised node caches](https://energy-models.github.io/datarecord/design/layers/#materialised-node-caches)
    - [the owner map](https://energy-models.github.io/datarecord/design/read-path/#owner-map)
    """
    middle = parent.child()
    middle.materialise()
    grandchild = middle.child()

    n = PyPSA.build(grandchild.record)
    pd.testing.assert_index_equal(
        n.c["Generator"].static.index,
        ac_dc.c["Generator"].static.index,
        check_names=False,
    )
