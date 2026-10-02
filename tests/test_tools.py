# SPDX-FileCopyrightText: datarecord contributors
#
# SPDX-License-Identifier: MIT

"""The tool layer: verify a record, build a model, read results back.

Notes
-----
- [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
"""

from pathlib import Path

import narwhals as nw
import pandas as pd
import pytest

from datarecord import Revision
from datarecord.duck import layer_dir
from datarecord.layered.resolve import read_schema, write_schema
from datarecord.schema import Dimension, TypeSpec
from datarecord.tools.base import Requirements, Schema, UnsupportedRecordError
from datarecord.tools.pypsa import ENTITY_TYPE, PyPSA, _colliding_names
from tests.fixtures import (
    export_network,
    relation,
    write_entity_type,
    write_input,
)


@pytest.fixture
def single_revision(con, base_uri, ac_dc):
    revision = Revision.create(con)
    export_network(ac_dc, revision, con)
    return revision


def test_requirements_falsy_when_empty():
    """`bool(Requirements())` distinguishes "nothing missing" from "nothing checked"."""
    assert not Requirements()
    assert Requirements(dims=frozenset({"snapshot"}))
    assert "dims ['snapshot']" in Requirements(dims=frozenset({"snapshot"})).describe()
    assert Requirements().describe() == "nothing"


def test_the_record_layer_imports_no_tool():
    """`datarecord` is framework-free: reading a record never imports PyPSA.

    The call runs from the tool inward (`PyPSA.build(revision.record)`), so there is no
    registry and no name dispatch to drag a framework in.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    import subprocess
    import sys

    # A fresh interpreter, so an already-imported PyPSA from another test
    # cannot mask a real import edge.
    code = (
        "import sys; import datarecord; import datarecord.tools; "
        "assert 'pypsa' not in sys.modules, sorted(sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_verify_accepts_a_complete_record(single_revision):
    """A record written by `export_to_parquet` supplies everything PyPSA needs."""
    assert not PyPSA.verify(single_revision.record)


def test_requires_reports_the_records_own_types(single_revision):
    """`requires` is record-dependent: PyPSA's axes plus this record's types."""
    req = PyPSA.requires(single_revision.record)
    assert {"snapshot", "period", "scenario"} <= req.dims
    assert "Generator" in req.entity_types
    # Required attributes come from PyPSA's registry, not a list we maintain.
    assert ("Generator", "bus") in req.attributes
    assert ("Line", "x") in req.attributes
    # `name` is the member identity, never an attribute row.
    assert not [a for a in req.attributes if a[1] == "entity"]


def test_verify_reports_a_missing_dim(con, base_uri, ac_dc):
    """A schema that declares no `scenario` dim cannot build a network.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    revision = Revision.create(con)
    export_network(ac_dc, revision, con)
    _with_schema(revision, dims={"snapshot": nw.Datetime()}, partial=set())

    missing = PyPSA.verify(revision.record)
    assert missing
    assert missing.dims == {"period", "scenario"}
    # A build refuses rather than failing deep inside PyPSA.
    with pytest.raises(UnsupportedRecordError, match="scenario"):
        PyPSA.build(revision.record)


def test_verify_reports_a_type_the_tool_does_not_know(con, base_uri, ac_dc):
    """A type outside PyPSA's registry is reported by `verify`, not raised in the fold.

    The type must be one the *schema* declares - the entity-type axis is an
    `Enum` and that vocabulary is upheld everywhere, so a record cannot hold a
    type its own schema excludes. What it can hold is a type this tool has no
    registry entry for, which reads back fine and is this tool's business rather
    than the record layer's. `Requirements.entity_types` is what carries it.

    Notes
    -----
    - [entity types](https://energy-models.github.io/datarecord/design/schema/#entity_type-the-axis-of-kinds)
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    revision = Revision.create(con)
    export_network(ac_dc, revision, con)
    # Declared alongside PyPSA's own types, since the axis pins the vocabulary.
    declared = read_schema(con)
    axis = declared.dimensions[ENTITY_TYPE]
    assert isinstance(axis.dtype, nw.Enum), "PyPSA declares its types as an enum"
    write_schema(
        declared.model_copy(
            update={
                "dimensions": {
                    **declared.dimensions,
                    ENTITY_TYPE: axis.model_copy(
                        update={"dtype": nw.Enum([*axis.dtype.categories, "Widget"])}
                    ),
                },
                # `types` keys must equal the labels, and a Widget carries
                # whatever it likes - the record layer upholds only its own
                # schema's vocabulary.
                "types": {**declared.types, "Widget": TypeSpec()},
            }
        )
    )
    write_entity_type(layer_dir(revision.id), "Widget", [{"entity": "w1"}])

    missing = PyPSA.verify(revision.record)
    assert missing.entity_types == {"Widget"}
    # The types PyPSA does know are not reported.
    assert "Generator" not in missing.entity_types
    with pytest.raises(UnsupportedRecordError, match="Widget"):
        PyPSA.build(revision.record)


_DIMS = {
    "snapshot": nw.Datetime(),
    "period": nw.Int64(),
    "scenario": nw.String(),
}


def _without_default(revision, ctype: str, attribute: str) -> None:
    """Drop one attribute's declared default on one type, leaving the rest.

    The default is a fact of the grant, so dropping it for `ctype` leaves every
    other type's reading alone.
    """
    was = read_schema()
    assert attribute in was.attributes_for(ctype)
    grant = was.types[ctype].attributes[attribute]
    was.types[ctype].attributes[attribute] = grant.model_copy(update={"default": None})
    write_schema(was)


def _with_schema(revision, dims=_DIMS, partial={"scenario"}) -> None:
    """Redeclare the record's schema with different axes or granularity.

    Written directly rather than through `write_record`, which would reject an
    incompatible redeclaration - here the point is to hand the tool a
    schema it must report on rather than one the writer accepted. Everything
    but the non-structural dims and `partial` is kept as declared.

    Notes
    -----
    - [versioning](https://energy-models.github.io/datarecord/design/schema/#versioning)
    """
    was = read_schema()
    kept = {
        d: spec
        for d, spec in was.dimensions.items()
        if d in ("entity", "bus", ENTITY_TYPE)
    }
    now = was.model_copy(
        update={
            "dimensions": {
                **kept,
                **{d: Dimension(dtype=t) for d, t in dims.items()},
            },
            "partial": frozenset(partial),
        }
    )
    write_schema(now)


def test_verify_reports_a_snapshot_key(con, base_uri, ac_dc):
    """`snapshot` as a key is reported, not crashed on.

    A limit of this tool's *representation*, not of the layer format: PyPSA's
    static/series split needs a component's whole series for an attribute to
    come from one layer, so a stored `snapshot = NULL` broadcast row and a
    descendant's per-snapshot row would coexist with no single container to
    put the result in. The record layer therefore permits the
    declaration - every file does carry the column - and the tool catches it.

    Notes
    -----
    - [partial](https://energy-models.github.io/datarecord/design/schema/#partial-the-granularity-of-an-override)
    """
    revision = Revision.create(con)
    write_schema(PyPSA.to_datarecord(ac_dc).schema)
    export_network(ac_dc, revision, con)
    _with_schema(revision, partial={"scenario", "snapshot"})

    missing = PyPSA.verify(revision.record)
    assert ("input_key", "snapshot") in missing.unsupported_keys
    assert "snapshot" in missing.describe()
    with pytest.raises(UnsupportedRecordError, match="snapshot"):
        PyPSA.build(revision.record)


def test_verify_reports_a_missing_required_attribute(con, base_uri, ac_dc):
    """A component type with no `bus` anywhere - not in the frame, not in the catalog."""
    revision = Revision.create(con)
    export_network(ac_dc, revision, con)
    # A Generator member carrying no `bus` column at all, no connection row
    # supplying one (https://energy-models.github.io/datarecord/design/record/#connections), and a schema with no default for it either.
    write_entity_type(layer_dir(revision.id), "Generator", [{"entity": "g1"}])
    # One file across every type, so the Generators' rows are dropped from it
    # rather than a per-type file being unlinked.
    path = Path(layer_dir(revision.id), "groups", "connection.parquet")
    rows = pd.read_parquet(path)
    generators = set(ac_dc.c["Generator"].static.index) | {"g1"}
    rows[~rows["entity"].isin(generators)].to_parquet(path, index=False)
    _without_default(revision, "Generator", "bus")

    missing = PyPSA.verify(revision.record)
    assert ("Generator", "bus") in missing.attributes


def test_verify_accepts_a_declared_default_for_a_required_attribute(
    con, base_uri, ac_dc
):
    """A declared default makes an attribute resolvable with no row anywhere.

    Notes
    -----
    - [AttributeSpec](https://energy-models.github.io/datarecord/design/schema/#attributespec)
    """
    revision = Revision.create(con)
    export_network(ac_dc, revision, con)
    write_entity_type(layer_dir(revision.id), "Generator", [{"entity": "g1"}])

    # PyPSA's own registry already declares `bus` with a `""` default, which
    # is exactly the case this pins - so the record is left as written.
    assert ("Generator", "bus") not in PyPSA.verify(revision.record).attributes


def test_verify_reports_a_piecewise_linear_attribute(con, base_uri, ac_dc):
    """A curve is stored correctly; it is the PyPSA translation that cannot express it.

    Notes
    -----
    - [wide and long rows](https://energy-models.github.io/datarecord/design/record/#wide-and-long-rows)
    """
    revision = Revision.create(con)
    export_network(ac_dc, revision, con)
    write_input(
        layer_dir(revision.id),
        "marginal_cost",
        [
            {
                "entity": "Manchester Wind",
                "breakpoint": x,
                "value": v,
            }
            for x, v in ((0.0, 20.0), (50.0, 35.0))
        ],
    )

    missing = PyPSA.verify(revision.record)
    assert ("Generator", "marginal_cost") in missing.unsupported_values
    assert "piecewise-linear" in missing.describe()
    with pytest.raises(UnsupportedRecordError):
        PyPSA.build(revision.record)


def test_verify_accepts_a_scalar_attribute(single_revision):
    """The same attribute without breakpoints is not reported (the negative half)."""
    write_input(
        layer_dir(single_revision.id),
        "marginal_cost",
        [{"entity": "Manchester Wind", "value": 20.0}],
    )
    assert not PyPSA.verify(single_revision.record).unsupported_values


def test_to_datarecord_rejects_a_cross_type_name_collision():
    """PyPSA scopes names per type; a record scopes them across every type.

    Reported rather than repaired: renaming to `Generator:north` would hand back
    a network whose components PyPSA can no longer find by their own names, so
    the mismatch is the caller's to reconcile.

    Notes
    -----
    - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    import pypsa

    n = pypsa.Network()
    n.add("Bus", "north")
    n.add("Generator", "north", bus="north")

    with pytest.raises(UnsupportedRecordError, match="more than one component type"):
        PyPSA.to_datarecord(n)


def test_to_datarecord_accepts_distinct_names():
    """The negative half: the same network with distinct names is buildable."""
    import pypsa

    n = pypsa.Network()
    n.add("Bus", "north")
    n.add("Generator", "north gen", bus="north")

    assert not _colliding_names(n)
    assert PyPSA.to_datarecord(n) is not None


def test_a_stochastic_network_is_not_a_false_collision():
    """A `(scenario, name)` index is one name per component, not one per scenario.

    Comparing the index tuples would report every scenario's copy as a distinct
    name while missing a real cross-type clash.
    """
    import pypsa

    n = pypsa.Network()
    n.set_scenarios(["a", "b"])
    n.add("Bus", "north")
    n.add("Generator", "north gen", bus="north")

    assert not _colliding_names(n)


def test_pypsa_schema_is_the_identity(single_revision):
    """PyPSA defines the record vocabulary today, so nothing is renamed.

    The seam still routes every attribute, so an entry added later takes
    effect with no change to `build`.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    assert PyPSA.schema.attrs == {}
    assert PyPSA.schema.sources("Generator", "p_max_pu") == ("p_max_pu",)
    identity = PyPSA.schema.resolve(single_revision.record, "Generator", "p_max_pu")
    assert identity.fetchall() == relation(single_revision, "p_max_pu").fetchall()


def test_schema_renames_and_computes():
    """A rename maps one source; a computed attribute maps several (`Attr.compute`)."""
    from datarecord.tools.base import Attr

    schema = Schema(
        {
            "Generator": (
                Attr(name="p_max_pu", source=("availability",)),
                Attr(
                    name="marginal_cost",
                    source=("fuel_cost", "efficiency"),
                    compute=lambda fuel, eff: fuel,
                ),
            )
        }
    )
    assert schema.sources("Generator", "p_max_pu") == ("availability",)
    assert schema.sources("Generator", "marginal_cost") == ("fuel_cost", "efficiency")
    # Anything unmapped, and any other type, stays the identity.
    assert schema.sources("Generator", "p_nom") == ("p_nom",)
    assert schema.sources("Link", "p_max_pu") == ("p_max_pu",)
    # A multi-source rename with no way to combine them is a declaration bug.
    with pytest.raises(ValueError, match="exactly one"):
        Attr(name="x", source=("a", "b"))


def test_results_extracts_long_form_outputs(single_revision):
    """A solved network's results come back keyed by attribute, long-form.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    n = PyPSA.build(single_revision.record)
    n.optimize(solver_name="highs")

    results = PyPSA.results(n)
    assert "p" in results
    assert "p_nom_opt" in results

    # Narwhals frames, so the seam names no one dataframe library, and lazy so a
    # tool may fetch on demand (https://energy-models.github.io/datarecord/design/tools/).
    assert isinstance(results["p"], nw.LazyFrame)
    p = results["p"].collect()
    # The long schema's columns (https://energy-models.github.io/datarecord/design/record/), so the write path can persist it as-is -
    # and no `entity_type`, an attribute row being keyed by `name` (https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types).
    assert {"entity", "snapshot", "scenario", "period", "value"} <= set(p.columns)
    assert "entity_type" not in p.columns
    assert set(p["attribute"].to_list()) == {"p"}
    # Series output: one row per (name, snapshot), for the Generator rows -
    # selected by name, since the frame no longer carries the type.
    gens = set(n.c["Generator"].static.index)
    gen = p.filter(nw.col("entity").is_in(list(gens)))
    assert len(gen) == len(n.snapshots) * len(gens)

    # A static output has no snapshot, and only the components whose value
    # differs from the default appear (some generators solve to p_nom_opt=0).
    nom = results["p_nom_opt"].collect()
    nom = nom.filter(nw.col("entity").is_in(list(gens)))
    assert nom["snapshot"].is_null().all()
    nonzero = n.c["Generator"].static["p_nom_opt"] != 0.0
    assert set(nom["entity"].to_list()) == set(n.c["Generator"].static.index[nonzero])


def test_results_concatenate_every_type_under_one_attribute(single_revision):
    """One `p` frame holds every type's rows, matching `outputs/p.parquet`.

    Notes
    -----
    - [Flags](https://energy-models.github.io/datarecord/design/record/#flags)
    """
    n = PyPSA.build(single_revision.record)
    n.optimize(solver_name="highs")

    p = PyPSA.results(n)["p"].collect()
    names = set(p["entity"].to_list())
    # `p` is a result of several types, so the concat is what is being tested;
    # keying by `(type, attribute)` would have split these into separate frames.
    # The names identify which type each row came from, no tag column needed -
    # that being what unique names buy the union (https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types).
    contributing = {
        c.name
        for c in n.components
        if not c.static.empty and set(c.static.index) & names
    }
    assert len(contributing) > 1
    assert not p["entity"].is_null().any()


def test_results_skips_outputs_still_at_their_default(single_revision):
    """An unsolved network yields no `p`/`p_nom_opt` rows (the outputs default rule).

    Notes
    -----
    - [outputs](https://energy-models.github.io/datarecord/design/read-path/#outputs)
    """
    n = PyPSA.build(single_revision.record)
    results = PyPSA.results(n)
    assert "p" not in results
    assert "p_nom_opt" not in results


def test_a_second_tool_needs_no_record_change(con, base_uri, ac_dc):
    """A tool is a plain object taking a record; adding one changes nothing here.

    No registration and no name dispatch: conformance to `Tool` is structural,
    so a second framework's module defines its own singleton and callers import
    it.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    from datarecord.tools.base import Tool

    class FakeTool:
        name = "fake"
        schema = Schema()

        def requires(self, record):
            return Requirements(dims=frozenset({"snapshot"}))

        def verify(self, record):
            return Requirements(entity_types=frozenset({"Nope"}))

        def build(self, record):
            return "fake-model"

        def results(self, model):
            return {("Thing", "x"): model}

        def to_datarecord(self, model):
            return f"layer-of-{model}"

    fake = FakeTool()
    assert isinstance(fake, Tool)

    revision = Revision.create(con)
    export_network(ac_dc, revision, con)

    assert fake.requires(revision.record).dims == {"snapshot"}
    assert fake.verify(revision.record).entity_types == {"Nope"}
    assert fake.build(revision.record) == "fake-model"
    assert fake.results("m") == {("Thing", "x"): "m"}
    assert fake.to_datarecord("m") == "layer-of-m"
    # The record itself knows nothing of either tool.
    assert not hasattr(revision, "to_model")
    assert not PyPSA.verify(revision.record)


def test_schema_dims_stay_generic(con, base_uri, ac_dc):
    """`Coords` carries a dim PyPSA knows nothing about; axis names live in the tool.

    Declared but not `partial`, so the record's files need no column for it.
    Keying it would be a different matter, reported by the tool against the
    real record - see `test_verify_reports_unsupported_keys`.
    """
    revision = Revision.create(con)
    export_network(ac_dc, revision, con)
    _with_schema(revision, dims={**_DIMS, "vintage": nw.String()})
    dims = revision.resolver.dims
    assert "vintage" in dims.schema.dims
    # No axis rows anywhere, so the dim is absent from the mapping rather than
    # present-and-empty (https://energy-models.github.io/datarecord/design/record/#frames).
    assert "vintage" not in dims.axes
    # PyPSA's own required dims are still satisfied, and the extra dim is
    # simply not something the tool looks at.
    assert not PyPSA.verify(revision.record)
    assert "vintage" in dims.schema.long_columns
