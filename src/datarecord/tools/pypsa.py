# SPDX-FileCopyrightText: datarecord contributors
#
# SPDX-License-Identifier: MIT

"""The PyPSA tool: record -> `pypsa.Network` -> results.

The only module that knows PyPSA's network shape - that its axes are
`snapshot`/`period`/`scenario`, that a stochastic network is indexed by
`(scenario, name)`, and how its static/series split maps onto the record's
`dims/entity_type` + `inputs/` split.

Notes
-----
- [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
- [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property, reduce
from typing import TYPE_CHECKING, Any

import narwhals as nw
import numpy as np
import pandas as pd
from duckdb import CoalesceOperator as coalesce
from duckdb import ColumnExpression as col
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from duckdb import StarExpression as star

from datarecord.duck import ex_all
from datarecord.record import Flags, Frames, LazyFrames, RecordLike
from datarecord.schema import AttributeSpec, Dimension, Group, TypeAttribute, TypeSpec
from datarecord.schema import Schema as RecordSchema
from datarecord.tools.base import (
    Requirements,
    Schema,
    Tool,
    UnsupportedRecordError,
    to_relation,
)

if TYPE_CHECKING:
    import pypsa

# The axes a PyPSA network is built from; all three must be declared in the
# schema's `dimensions`, since a network shape is built from them (https://energy-models.github.io/datarecord/design/tools/).
# A declared axis with no rows is fine - that is just a deterministic or
# single-period network.
SNAPSHOT, PERIOD, SCENARIO = "snapshot", "period", "scenario"
# PyPSA's own name for the column `scenario_weightings` carries, so a record
# round-trips it without renaming.
SCENARIO_WEIGHT = "weight"
ENTITY, BUS, CONNECTION = "entity", "bus", "connection"
ENTITY_TYPE = "entity_type"
# An attribute over the `connection` group, not a column the format fixes: the
# port vocabulary below is this tool's, so the record layer never names it.
ROLE = "role"
REQUIRED_DIMS = frozenset({SNAPSHOT, PERIOD, SCENARIO})

# PyPSA's `defaults["typ"]` (the attribute's Python type) mapped to the narwhals
# type the long schema stores, `String` for anything unlisted. Keyed on `typ` -
# what a value *is* - not `type`, whose `"static or series"`-style strings
# describe only where it lives and grow a new variant per PyPSA release.
_DTYPES = {
    bool: nw.Boolean(),
    int: nw.Int64(),
    float: nw.Float64(),
    str: nw.String(),
}


def _default(value: Any) -> Any:
    """One `defaults["default"]` cell as JSON-storable, NaN as absent.

    Notes
    -----
    - [AttributeSpec](https://energy-models.github.io/datarecord/design/schema/#attributespec)
    """
    if isinstance(value, float) and math.isnan(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _text(value: Any) -> str | None:
    """One `unit`/`description` cell as text, or None where PyPSA has none.

    Its registry leaves both NaN rather than empty, and `None` is the schema's
    "undeclared" - `""` would claim the attribute is documented as blank, or
    dimensionless, neither of which a missing cell says.

    Notes
    -----
    - [unit and description](https://energy-models.github.io/datarecord/design/schema/#unit-and-description)
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class NetworkShape:
    """PyPSA's reading of a record's axis frames.

    The axis names and the index convention live here rather than on the
    record, which stays schema-generic: a record may declare any dims, and it is
    this tool that decides `snapshot` is the time axis and that a non-empty
    `scenario` axis means a stochastic network.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """

    dims: Frames

    def _axis(self, dim: str) -> pd.DataFrame:
        """`dim`'s axis as a DataFrame, empty if the record declares none.

        A dim with no rows anywhere is absent from the mapping rather than
        present-and-empty, and both read the same way here: an empty
        frame, which is a deterministic or single-period network.

        Notes
        -----
        - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
        """
        frame = self.dims.get(dim)
        return (
            frame.to_native().df() if frame is not None else pd.DataFrame(columns=[dim])
        )

    @cached_property
    def snapshots(self) -> pd.DataFrame:
        return self._axis(SNAPSHOT)

    @cached_property
    def periods(self) -> pd.DataFrame:
        return self._axis(PERIOD)

    @cached_property
    def scenarios(self) -> pd.DataFrame:
        return self._axis(SCENARIO)

    @property
    def stochastic(self) -> bool:
        return not self.scenarios.empty

    @property
    def multiperiod(self) -> bool:
        return not self.periods.empty

    @property
    def index_names(self) -> list[str]:
        """The static frame's index: PyPSA carries `scenario` as a level.

        Notes
        -----
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
        """
        return [SCENARIO, "entity"] if self.stochastic else ["entity"]


def _connection(record: RecordLike) -> DuckDBPyConnection:
    """The DuckDB connection `record`'s frames belong to.

    Off the concrete backing, since the protocol stays backend-agnostic.
    Needed because a relation exposes no reachable reference to its connection,
    and this tool's `PIVOT` - which narwhals cannot express and
    `relation.query` refuses as a `MULTI` statement - needs one.

    Raises
    ------
    TypeError
        If `record` exposes no connection, i.e. is not DuckDB-backed.

    Notes
    -----
    - [the protocol names no engine](https://energy-models.github.io/datarecord/design/record/#the-protocol-names-no-engine)
    """
    con = getattr(record, "con", None)
    if not isinstance(con, DuckDBPyConnection):
        msg = (
            f"{type(record).__name__} exposes no DuckDB connection; "
            "the PyPSA tool builds with DuckDB SQL"
        )
        raise TypeError(msg)
    return con


# -- long -> wide -----------------------------------------------------------

# Separates pivoted index-level values in a combined PIVOT column name; chosen
# to never collide with a component/scenario name (the broadcast rule).
_KEY_SEP = "\x01"


@dataclass(frozen=True)
class Index:
    """A pandas Index paired with a matching DuckDB relation.

    `rel` carries `_pos`, the index's row position, so a join against it can
    be restored to `index`'s order (not necessarily sorted) by ordering on
    `_pos` rather than by value.
    """

    index: pd.Index
    rel: DuckDBPyRelation

    @classmethod
    def of(cls, index: pd.Index, con: DuckDBPyConnection) -> Index:
        df = index.to_frame(index=False)  # noqa: F841 - queried by name below
        return cls(index, con.sql("SELECT *, row_number() OVER () AS _pos FROM df"))

    def expand(self, rel: DuckDBPyRelation, column: str) -> DuckDBPyRelation:
        """Broadcast rows with a null `column` across this index, others as-is.

        `rel` must already carry the alias the caller wants to select its
        own columns by - `expand` doesn't impose one.
        """
        if len(self.index) == 0:
            return rel
        return rel.join(
            self.rel.set_alias("_ax"), col(rel.alias, column).isnull(), how="left"
        ).project(
            *(col(rel.alias, c) for c in rel.columns if c != column),
            coalesce(col(rel.alias, column), col("_ax", column)).alias(column),
        )

    def left_join(self, other: DuckDBPyRelation, on: Sequence[str]) -> DuckDBPyRelation:
        """LEFT JOIN `other` onto this index, restored to `index`'s row order."""
        return self.rel.join(
            other,
            ex_all(col(self.rel.alias, f) == col(other.alias, f) for f in on),
            how="left",
        ).order(f"{self.rel.alias}._pos")


def _assign_static(
    static: DuckDBPyRelation,
    attributes: dict[str, tuple[DuckDBPyRelation, Flags]],
    shape: NetworkShape,
    scenarios: Index,
) -> pd.DataFrame:
    """Set each varying attribute's static-valued entries, resolved to a wide frame."""
    keys = shape.index_names

    # `static` already carries a `_pos` in member order, assigned before the
    # connection collapse could scramble it (`build`); the wide frame is ordered
    # by it below (https://energy-models.github.io/datarecord/design/read-path/#one-fold-for-every-axis).

    # An entity exists once, whatever the scenario, so the record's member
    # frame carries one row per entity - where PyPSA's static container is
    # `(scenario, name)`-indexed. Repeated here, and any attribute that really
    # differs between scenarios arrived as `inputs/` rows and is assigned over
    # the repeats below.
    if shape.stochastic and SCENARIO not in static.columns:
        static = scenarios.expand(
            static.project(f"*, NULL AS {SCENARIO}").set_alias("m"), SCENARIO
        )

    joined = static
    attr_exprs = {}
    for attr, (long, flags) in attributes.items():
        # PyPSA's `static` container takes the rows that do not vary over
        # `snapshot`. Two ways an attribute has them, and the flags distinguish
        # the two because they are scoped to what it is addressed by (https://energy-models.github.io/datarecord/design/record/#flags):
        # a NULL-snapshot row on an attribute that *may* vary over it puts
        # `snapshot` in `broadcast`, while one not addressed by it at all has
        # `snapshot` in neither set and every row static.
        addressed = SNAPSHOT in (flags.varies | flags.broadcast)
        if addressed and SNAPSHOT not in flags.broadcast:
            continue
        # No column to filter on where the attribute is not addressed by it.
        rows = long.filter("snapshot IS NULL") if addressed else long
        if shape.stochastic:
            rows = scenarios.expand(rows, SCENARIO)
        # First-wins on a duplicate key, same as `values[~values.index.duplicated()]`;
        # any deterministic order is fine since a valid record never actually collides.
        rows = rows.project(
            f"*, row_number() OVER (PARTITION BY {', '.join(keys)}) AS _rn"
        ).filter("_rn = 1")
        # LEFT JOIN from `static`, not `rows`, so a value with no matching static
        # row is dropped rather than added (`common = values.index.intersection(...)`).
        on = ex_all(col(static.alias, k) == col(rows.alias, k) for k in keys)
        joined = joined.join(rows, on, how="left")
        value = col(rows.alias, "value")
        attr_exprs[attr] = (
            coalesce(value, col(static.alias, attr))
            if attr in static.columns
            else value
        ).alias(attr)

    wide = joined.project(
        *(col(static.alias, c) for c in static.columns if c not in attr_exprs),
        *attr_exprs.values(),
    )
    frame = wide.order("_pos").project(star(exclude=["_pos"])).df()
    # Scenario-major: PyPSA's `(scenario, name)` index groups a scenario's
    # components together, and `_pos` gives the order within one - the
    # entity axis carrying one row per entity rather than one per scenario, so
    # it cannot order the outer level itself. A stable sort keeps that order.
    if shape.stochastic:
        frame = frame.sort_values(
            SCENARIO,
            kind="stable",
            key=lambda s: s.map({v: i for i, v in enumerate(scenarios.index)}),
        )
    return frame.set_index(keys)


def _flat_index(index: pd.Index, sep: str = "_") -> pd.Index[str]:
    """Join a MultiIndex's levels into one string-valued Index; other indexes as-is."""
    if not isinstance(index, pd.MultiIndex):
        return index

    return reduce(
        lambda x, y: x + sep + y,
        (index.get_level_values(lvl) for lvl in range(index.nlevels)),
    )


def _series_frame(
    rows: DuckDBPyRelation,
    shape: NetworkShape,
    periods: Index,
    scenarios: Index,
    snapshots: Index,
    static_index: pd.Index,
    con: DuckDBPyConnection,
) -> pd.DataFrame:
    """Pivot long series rows to the `n.snapshots x static_index` frame.

    `snapshots` (the row axis) is known before the pivot, so missing rows are
    filled by joining against it directly rather than reindexing pandas-side
    after. Columns stay restricted to what the data actually pivoted, only
    ordered to match `static_index`: `_import_series_from_df` takes
    `df.columns` as-is without merging against `static`, so a column added
    just to complete `static_index` would wrongly force that component into
    `dynamic` instead of leaving it static.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    if shape.multiperiod:
        rows = periods.expand(rows, PERIOD)
    if shape.stochastic:
        rows = scenarios.expand(rows, SCENARIO)
    group_by = [PERIOD, SNAPSHOT] if shape.multiperiod else [SNAPSHOT]
    pivot_key = f" || '{_KEY_SEP}' || ".join(shape.index_names)
    piv = con.sql(
        f"PIVOT rows ON {pivot_key} USING first(value) GROUP BY {', '.join(group_by)}"
    )

    present = set(piv.columns) - set(group_by)
    pivoted_names = _flat_index(static_index, _KEY_SEP)
    keep = pivoted_names.isin(present)

    wide = (
        snapshots.left_join(piv.set_alias("piv"), group_by)
        .project(*(col("piv", name) for name in pivoted_names[keep]))
        .df()
    )
    wide.index = snapshots.index
    wide.columns = static_index[keep]
    return wide


# -- build (https://energy-models.github.io/datarecord/design/tools/) -------------------------------------------------------------


def _exported(c) -> bool:
    """Whether `c`'s members belong in a record at all.

    Empty types have nothing to write. Standard types (`LineType`,
    `TransformerType`) are excluded because PyPSA prepopulates them on every
    fresh `Network` and `_broadcast_standard_types` restores them on build -
    writing them would import each row a second time on the way back.
    """
    return not c.static.empty and c.name not in c.n.standard_type_components


def _colliding_names(n: pypsa.Network) -> frozenset[str]:
    """Names more than one exported component type claims.

    Only exported types count, since only those become member rows. Read off the
    `name` level rather than the index: a stochastic network is keyed
    `(scenario, name)`, and comparing tuples would read one component in two
    scenarios as two names while missing a real collision.

    Notes
    -----
    - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    seen: dict[str, str] = {}
    clashing: set[str] = set()
    for c in n.components:
        if not _exported(c):
            continue
        index = c.static.index
        names = (
            index.get_level_values("name") if "name" in (index.names or []) else index
        )
        for value in names:
            name = str(value)
            if seen.setdefault(name, c.name) != c.name:
                clashing.add(name)
    return frozenset(clashing)


# TODO(pypsa): every function below, up to `_new_network`, ports a method that
# only exists on PyPSA's unreleased data-records branch (https://energy-models.github.io/datarecord/design/tools/). Delete the
# ported copy and call the real method once a PyPSA release carries it.


def _apply_snapshots_import(n: pypsa.Network, df: pd.DataFrame) -> None:
    """Set snapshots and snapshot weightings from an imported axis table.

    Ported from `pypsa.Network._apply_snapshots_import` (unreleased) -
    built entirely from public surface (`set_snapshots`, `snapshot_weightings`).

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    snapshot_levels = {"period", "timestep", "snapshot"}.intersection(df.columns)
    if snapshot_levels:
        df = df.set_index(sorted(snapshot_levels))
    n.set_snapshots(df.index)

    cols = ["objective", "stores", "generators"]
    if not df.columns.intersection(cols).empty:
        existing_cols = [c for c in cols if c in df.columns]
        n.snapshot_weightings = df.reindex(index=n.snapshots, columns=existing_cols)
    elif "weightings" in df.columns:
        n.snapshot_weightings = df["weightings"].reindex(n.snapshots)


def _broadcast_standard_types(n: pypsa.Network) -> None:
    """Broadcast standard-type static tables across scenarios after import.

    Ported from `pypsa.Network._broadcast_standard_types` (unreleased).

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    for component in n.standard_type_components:
        comp = n.components[component]
        if n.has_scenarios and not isinstance(comp.static.index, pd.MultiIndex):
            comp.static = pd.concat(
                dict.fromkeys(n.scenarios, comp.static), names=["scenario"]
            )


def _collect_network_attributes(n: pypsa.Network) -> dict[str, Any]:
    """The serializable scalar network attributes (incl. `name`, `pypsa_version`).

    Ported from `pypsa.Network._collect_network_attributes` (unreleased),
    trimmed of nothing - the reflection over `dir(n)` is exactly what decides
    which attributes are safe to round-trip, so there is no smaller version of
    this that still answers the same question.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    numpy_types = tuple(np.sctypeDict.values())
    allowed_types = (float, int, bool, str)
    skip_attrs = {
        "component_attrs",
        "df",
        "pnl",
        "static",
        "dynamic",
        "iterate_components",
        "_name",
        "_pypsa_version",
    }

    attrs: dict[str, Any] = {}
    for attr in dir(n):
        if attr.startswith("__") or attr in skip_attrs:
            continue
        # Skip read-only properties (except pypsa_version) without invoking
        # their getters, which may emit warnings (e.g. model, objective).
        prop = getattr(type(n), attr, None)
        if isinstance(prop, property) and prop.fset is None and attr != "pypsa_version":
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*component_attrs is deprecated as of 1\.0.*",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*the API for how to access components data has changed.*",
                category=DeprecationWarning,
            )
            value = getattr(n, attr)
        if isinstance(value, numpy_types):
            attrs[attr] = value.item()
        elif isinstance(value, allowed_types):
            attrs[attr] = value
    return attrs


def _apply_network_attributes(n: pypsa.Network, attrs: dict[str, Any]) -> None:
    """Apply scalar network attributes read back from a record.

    Ported from `pypsa.Network._apply_network_attributes` (unreleased),
    stripped of the version-compat warnings and PyPI update check the real
    method also does: those are for a human importing a file written by an
    older PyPSA, and have nothing to check here since `to_datarecord` and this
    function always agree on what `attrs` holds.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    attrs = dict(attrs)
    name = attrs.pop("name", None)
    if name is not None:
        n.name = name
    attrs.pop("pypsa_version", None)
    for attr, value in attrs.items():
        if attr in ("model", "objective", "objective_constant"):
            setattr(n, f"_{attr}", value)
        else:
            setattr(n, attr, value)


def _new_network(schema: RecordSchema, shape: NetworkShape) -> pypsa.Network:
    """An empty network with the record's axes already established."""
    import pypsa

    n = pypsa.Network()
    _apply_network_attributes(n, schema.meta.get("attributes") or {})
    n.meta = schema.meta.get("meta") or {}

    _apply_snapshots_import(n, shape.snapshots)
    if shape.multiperiod:
        ip = shape.periods.set_index(PERIOD)
        n.periods = ip.index
        n._investment_periods_data = ip.reindex(n.investment_periods)
    if shape.stochastic:
        scen = shape.scenarios.set_index(SCENARIO)
        scen.index = scen.index.astype(str).rename(SCENARIO)
        # `set_scenarios` rather than assigning `_scenarios_data`: it rebuilds
        # each component's empty `static`/`dynamic` frames with the `scenario`
        # level, so later imports union against a correctly named 2-level index
        # instead of the single-level one a fresh component carries.
        n.set_scenarios(scen["weight"])
    return n


def _add_component_type(
    n: pypsa.Network,
    ctype: str,
    static: DuckDBPyRelation,
    attributes: dict[str, tuple[DuckDBPyRelation, Flags]],
    shape: NetworkShape,
    con: DuckDBPyConnection,
) -> None:
    """Assign one type's static frame, then its series."""
    scenarios = Index.of(n.scenarios, con)
    static_df = _assign_static(static, attributes, shape, scenarios)
    n._import_components_from_df(static_df, ctype)

    periods = Index.of(n.investment_periods, con)
    snapshots = Index.of(n.snapshots, con)
    static_index = n.c[ctype].static.index
    for attr, (long, flags) in attributes.items():
        # ... and `dynamic` the per-snapshot ones.
        if SNAPSHOT not in flags.varies:
            continue
        ts_rows = long.filter("snapshot IS NOT NULL")
        # Partial columns keep non-series components static (https://energy-models.github.io/datarecord/design/tools/).
        wide = _series_frame(
            ts_rows, shape, periods, scenarios, snapshots, static_index, con
        )
        n._import_series_from_df(wide, ctype, attr)


# -- results (https://energy-models.github.io/datarecord/design/read-path/#outputs) ---------------------------------------------------------


def _output_attributes(c: pypsa.Components) -> list[str]:
    """A component type's result attributes, from PyPSA's own registry.

    `defaults["status"]` marks them ("Output"), so a PyPSA upgrade that adds a
    result attribute is picked up rather than needing a list kept here.
    """
    defaults = c.defaults
    is_output = defaults["status"].astype(str).str.startswith("Output")
    return list(defaults.index[is_output])


# -- network -> layer (https://energy-models.github.io/datarecord/design/format/) ------------------------------------------------

# Which end of the component a port is. PyPSA encodes this only by sign
# convention - `p0` flows in at `bus0`, `p1` out at `bus1` - so the mapping to
# a record's `role` lives here, the one place that convention is written down
# (https://energy-models.github.io/datarecord/design/record/#connections). Ports beyond the first two are outputs: a multi-port Link's `bus2`
# onward are additional sinks.
_INPUT_PORT, _OUTPUT_PORT, _SINGLE_PORT = "input", "output", "attached"

# Attribute stems that exist once per port. A record stores each as one
# bus-keyed attribute (https://energy-models.github.io/datarecord/design/record/#connections), so these are the names whose port suffix is
# undone on write and reapplied on build.
_PORT_STEMS = ("bus", "efficiency", "p")


def _port_role(port: str) -> str:
    """The `role` for a port index.

    A single-port component (`c.ports == [""]`, e.g. a Generator's one `bus`)
    is neither an input nor an output end - it has only one attachment - so it
    gets its own role rather than being forced into the two-ended convention.

    Notes
    -----
    - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
    """
    if port == "":
        return _SINGLE_PORT
    return _INPUT_PORT if port == "0" else _OUTPUT_PORT


def _port_attribute(stem: str, port: str) -> str:
    """PyPSA's name for `stem` at `port`: `bus0`, `efficiency`, `efficiency2`, ...

    `bus` is suffixed from `0`, every other per-port attribute from `2` - the
    quirk that makes the port vocabulary unguessable from a registry and is why
    a record keys connections by bus instead.

    Notes
    -----
    - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
    """
    if stem == "bus":
        return f"bus{port}"
    return stem if port == "1" else f"{stem}{port}"


def _scenario_varying(c: pypsa.Components, columns: Sequence[str]) -> frozenset[str]:
    """Which of `columns` differ between scenarios for some component.

    PyPSA's static frame is `(scenario, name)`-indexed on a stochastic network,
    and its own `consistency_check` permits most attributes to differ across
    scenarios - only an `INVARIANT_ATTRS` set may not. So "static" there means
    "not per snapshot", not "the same everywhere".

    Detected per attribute rather than declared, because whether one differs is
    a property of the data: a network may hold a per-scenario `capital_cost`
    and a shared `carrier`, and only the first varies over `scenario`.

    Notes
    -----
    - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
    """
    index = c.static.index
    if not isinstance(index, pd.MultiIndex) or SCENARIO not in (index.names or []):
        return frozenset()
    by_entity = c.static.groupby(level="name")
    return frozenset(
        x for x in columns if (by_entity[x].nunique(dropna=False) > 1).any()
    )


def _connection_rows(c: pypsa.Components) -> pd.DataFrame:
    """One type's connections as `(name, bus, role)` rows, from its port columns.

    Undoes PyPSA's positional encoding: `c.ports` is the authoritative port
    list (`["0", "1"]` plus `additional_ports`), so the port count comes from
    the network rather than a guessed bound.

    Notes
    -----
    - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
    """
    static = c.static
    frames = []
    for port in c.ports:
        column = _port_attribute("bus", port)
        if column not in static.columns:
            continue
        buses = static[column]
        attached = buses[buses.astype(str) != ""]
        if attached.empty:
            continue
        rows = attached.rename("bus").reset_index()
        rows = rows.rename(columns={"name": "entity"})
        rows["role"] = _port_role(port)
        rows["deleted"] = False
        frames.append(rows)
    if not frames:
        return pd.DataFrame(columns=["entity", "bus", "role", "deleted"])
    return pd.concat(frames, ignore_index=True)


# TODO(pypsa): every function below, up to `_as_long`, ports a function that
# only exists on PyPSA's unreleased data-records branch (https://energy-models.github.io/datarecord/design/tools/). Delete the
# ported copy and call the real one once a PyPSA release carries it.


def _is_output(defaults: pd.DataFrame, attr: str) -> bool:
    """Whether `attr` is an output (custom attrs with no status are inputs).

    Ported from `pypsa.common._is_output` rather than depended on, since only
    a released PyPSA can be relied on here (no `_as_long` yet).

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    if attr not in defaults.index:
        return False
    return str(defaults.at[attr, "status"]).startswith("Output")


def _drop_default_rows(rows: pd.DataFrame, default: Any) -> pd.DataFrame:
    """Remove scalar rows that just repeat the attribute's default value.

    Ported from `pypsa.components.array._drop_default_rows`.
    """
    if pd.isnull(default):
        return rows[rows["value"].notna()]
    return rows[rows["value"] != default]


def _concat_long(series_rows: pd.DataFrame, scalar_rows: pd.DataFrame) -> pd.DataFrame:
    """Concatenate series rows and null-snapshot scalar rows of one attribute.

    Ported from `pypsa.components.array._concat_long`. The scalar rows carry
    plain NaN in the dimension columns, which pandas would flag as an all-NA
    entry with a conflicting dtype - take the dtypes from the series rows
    instead.
    """
    if scalar_rows.empty:
        return series_rows
    if series_rows.empty:
        return scalar_rows
    scalar_rows = scalar_rows.copy()
    for column in ("snapshot", "period"):
        scalar_rows[column] = series_rows[column].iloc[:0].reindex(scalar_rows.index)
    return pd.concat([series_rows, scalar_rows[series_rows.columns]], ignore_index=True)


def _stack_dynamic(c: pypsa.Components, wide: pd.DataFrame) -> pd.DataFrame:
    """Melt a wide `snapshots x components` frame into long rows.

    Ported from `pypsa.components.array.Components._stack_dynamic`.
    """
    stochastic = isinstance(wide.columns, pd.MultiIndex)
    wide = wide.rename_axis(columns=["scenario", "entity"] if stochastic else "entity")

    stacked = wide.stack(level=list(range(wide.columns.nlevels)), future_stack=True)
    long = stacked.rename("value").reset_index()
    if isinstance(c.snapshots, pd.MultiIndex):
        long = long.rename(columns={"timestep": "snapshot"})
    if "scenario" not in long.columns:
        long["scenario"] = np.nan
    if "period" not in long.columns:
        long["period"] = np.nan
    return long


def _scalar_rows(c: pypsa.Components, attr: str, exclude: pd.Index) -> pd.DataFrame:
    """Build null-snapshot long rows from the static scalars of `attr`.

    Ported from `pypsa.components.array.Components._scalar_rows`.
    """
    static = c.static[attr]
    if len(exclude):
        static = static[~static.index.isin(exclude)]
    out = static.rename("value").reset_index().rename(columns={"name": "entity"})
    if "scenario" not in out.columns:
        out["scenario"] = np.nan
    out["snapshot"] = np.nan
    out["period"] = np.nan
    return out


def _as_long(
    c: pypsa.Components, attr: str, *, drop_defaults: bool = False
) -> pd.DataFrame:
    """`attr` as a long/tidy DataFrame; `drop_defaults` omits default-valued scalars.

    Ported from `pypsa.components.array.Components._as_long` (with its
    helpers above) rather than called on `c`: only an unreleased PyPSA branch
    carries this method, and a tool may not require one. The reshaping
    is otherwise built entirely from public surface - `c.dynamic`, `c.static`,
    `c.defaults`, `c.has_scenarios`, `c.snapshots` - so porting it costs
    nothing PyPSA doesn't already expose.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    varying = attr in c.dynamic or (
        attr in c.defaults.index and bool(c.defaults.at[attr, "varying"])
    )
    # A scenario-varying static attribute has a long representation too: its
    # rows carry the scenario they hold for, which `_scalar_rows` preserves.
    if (
        not varying
        and not _is_output(c.defaults, attr)
        and attr not in c.static.columns
    ):
        msg = f"'{attr}' of '{c.name}' components has no long representation."
        raise AttributeError(msg)

    columns = ["entity", "snapshot", "scenario", "period", "value"]
    if not varying:
        out = _scalar_rows(c, attr, pd.Index([]))[columns]
        if drop_defaults:
            out = _drop_default_rows(out, c.defaults.at[attr, "default"])
        return out
    wide = c.dynamic[attr]
    out = _stack_dynamic(c, wide)[columns]
    # scalar rows import into `c.static[attr]`, so only attrs with a static
    # column get them
    static_backed = attr in c.defaults.index and bool(c.defaults.at[attr, "static"])
    if static_backed:
        scalars = _scalar_rows(c, attr, wide.columns)[columns]
        if drop_defaults:
            scalars = _drop_default_rows(scalars, c.defaults.at[attr, "default"])
        out = _concat_long(out, scalars)
    return out


def _long_rows(
    c: pypsa.Components, attribute: str, dims: tuple[str, ...]
) -> pd.DataFrame:
    """One attribute's long rows for one component type, in the long schema.

    `_as_long` already emits the dim columns and `value`; this adds the
    columns the record's schema fixes and PyPSA has no notion of - `attribute`,
    and the NULL `breakpoint` that marks the value a scalar. No
    `entity_type`, and no `bus`: that is the `connection` group's
    coordinate, so it belongs to the rows of an attribute addressed by that
    group and `_per_port_long_rows` is what fills it.

    The widest shape any caller needs, from which `_long_frame` selects the
    attribute's own columns - `dims` is every axis a network has, not one
    attribute's.

    Notes
    -----
    - [the Record protocol](https://energy-models.github.io/datarecord/design/record/)
    - [wide and long rows](https://energy-models.github.io/datarecord/design/record/#wide-and-long-rows)
    - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
    """
    long = _as_long(c, attribute, drop_defaults=False)
    long = long.assign(attribute=attribute, breakpoint=None)
    for dim in dims:
        if dim not in long.columns:
            long[dim] = None
    return long


def _per_port_long_rows(
    c: pypsa.Components, stem: str, dims: tuple[str, ...]
) -> pd.DataFrame:
    """A per-connection attribute's long rows, `bus` filled from the port.

    The inverse of the positional collapse: `efficiency`/`efficiency2` become
    one attribute whose rows carry the bus each port attaches to, so the
    overlay owns them per connection rather than per component.

    Notes
    -----
    - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
    """
    static = c.static
    frames = []
    for port in c.ports:
        column = _port_attribute(stem, port)
        bus_column = _port_attribute("bus", port)
        if column not in c.defaults.index or bus_column not in static.columns:
            continue
        long = _long_rows(c, column, dims)
        if long.empty:
            continue
        # `static` is `(scenario, name)`-indexed on a stochastic network, so map
        # the bus by name alone - a component's bus does not vary by scenario, so
        # the name level deduplicates to one bus per component.
        buses = static[bus_column]
        if isinstance(buses.index, pd.MultiIndex):
            buses = buses.droplevel(SCENARIO)
            buses = buses[~buses.index.duplicated()]
        long["bus"] = long["entity"].map(buses)
        long["attribute"] = stem
        frames.append(long[long["bus"].astype(str) != ""])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# -- the tool ---------------------------------------------------------------


class PyPSATool(Tool):
    """Build a `pypsa.Network` from a record and read its results back."""

    name = "pypsa"

    # PyPSA defines the record vocabulary today, so every attribute maps to
    # itself and there is nothing to declare. The seam is here for the first
    # tool whose names differ, or whose value is computed from several of the
    # record's (`Attr.compute`).
    schema = Schema()

    def component_types(self) -> frozenset[str]:
        """The component types PyPSA knows, from its type registry.

        Taken from the component *type registry*, not a network instance:
        `pypsa.Network().components` iterates only the components attached to
        that network, which for an empty one is just the standard types.
        """
        from pypsa.components.types import all_components

        return frozenset(c.name for c in all_components.values())

    def requires(self, record: RecordLike) -> Requirements:
        """PyPSA's axes, the record's own component types, and their required attributes.

        The attribute half is record-dependent: only types the record
        actually defines members for are required, and which of their
        attributes are mandatory comes from PyPSA's registry
        (`defaults["status"] == "Input (required)"`), not a list kept here.
        Reported by source attribute (`Schema.sources`), so a renamed or
        computed one names what the record must actually supply.
        """
        known = self.component_types()
        ctypes = {ct for ct in record.entity_types if ct in known}
        return Requirements(
            dims=REQUIRED_DIMS,
            entity_types=frozenset(ctypes),
            attributes=frozenset(
                (ct, src)
                for ct in ctypes
                for attr in _required_attributes(ct)
                for src in self.schema.sources(ct, attr)
            ),
        )

    def verify(self, record: RecordLike) -> Requirements:
        """What this record fails to supply for a PyPSA build; falsy if it is usable.

        Checks what a build needs: the schema declares PyPSA's axes, its key dims are ones this tool can honour, every component
        type it names is one PyPSA knows, and each type's required attributes
        are resolvable - either owned by some layer (the owner map), supplied
        by the `dims/entity_type` frame, or carrying a declared default. A renamed or computed attribute is checked against its sources
        (`Schema.sources`), which is also how it is reported.

        The key checks come first and return early: the owner-map fold keys
        its relations by those dims, so reading the map at all could raise a
        binder error before anything below could report it.

        Notes
        -----
        - [AttributeSpec](https://energy-models.github.io/datarecord/design/schema/#attributespec)
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
        """
        dims = REQUIRED_DIMS - set(record.schema.dims)
        unsupported_keys = self._unsupported_keys(record)
        if unsupported_keys:
            return Requirements(
                dims=frozenset(dims), unsupported_keys=frozenset(unsupported_keys)
            )

        entity_types: set[str] = set()
        attributes: set[tuple[str, str]] = set()
        unsupported_values: set[tuple[str, str]] = set()

        known = self.component_types()
        declared = record.schema
        for ctype in sorted(record.entity_types):
            # A type PyPSA has no registry entry for, though the record's own
            # schema declares it. Reported rather than raised: the record layer
            # upholds its schema's vocabulary and knows no framework's, so this
            # reads back fine and it is this tool's business that it cannot be
            # built.
            if ctype not in known:
                entity_types.add(ctype)
                continue
            resolved = record.flags(ctype)
            owned = set(resolved)
            # A curve, not a scalar (https://energy-models.github.io/datarecord/design/record/#wide-and-long-rows). PyPSA takes a scalar for every
            # attribute this build assigns, so the record is storing something
            # correct that this translation cannot express - reported here
            # rather than silently pivoting one arbitrary breakpoint.
            unsupported_values |= {
                (ctype, attr) for attr, flags in resolved.items() if flags.breakpoints
            }
            static_cols = _static_columns(record, ctype)
            # A port attribute the record supplies as connection rows rather
            # than as a column: `bus0`/`bus1` are satisfied by a connection
            # per port, so the collapse in `build` can name them (https://energy-models.github.io/datarecord/design/record/#connections).
            from_connections = _connection_attributes(record, ctype)
            specs = declared.attributes_for(ctype)
            for attr in _required_attributes(ctype):
                for src in self.schema.sources(ctype, attr):
                    if src in owned or src in static_cols or src in from_connections:
                        continue
                    # A declared default makes the attribute resolvable even
                    # with no row anywhere (the broadcast rule).
                    spec = specs.get(src)
                    if spec is not None and spec.default is not None:
                        continue
                    attributes.add((ctype, src))
        return Requirements(
            dims=frozenset(dims),
            entity_types=frozenset(entity_types),
            attributes=frozenset(attributes),
            unsupported_values=frozenset(unsupported_values),
        )

    def _unsupported_keys(self, record: RecordLike) -> set[tuple[str, str]]:
        """`(key, dim)` pairs this tool cannot honour: `snapshot`, as any key.

        PyPSA's static/series split needs a component's whole series to come
        from one layer, so a snapshot-keyed overlay could leave a broadcast row
        and a descendant's per-snapshot row with no single container to land in. A limit of the *representation*, not the format - which is why
        the record layer permits the declaration and this reports it.

        Notes
        -----
        - [partial](https://energy-models.github.io/datarecord/design/schema/#partial-the-granularity-of-an-override)
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
        """
        defs = record.schema
        return {("input_key", SNAPSHOT)} if SNAPSHOT in defs.partial_dims else set()

    def build(self, record: RecordLike) -> pypsa.Network:
        """The resolved network, one component type at a time.

        Raises
        ------
        UnsupportedRecordError
            If `verify` reports anything missing - a partial build would
            surface as a confusing PyPSA error several frames later.

        Notes
        -----
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
        """
        missing = self.verify(record)
        if missing:
            raise UnsupportedRecordError(self.name, missing)

        con = _connection(record)
        schema = record.schema
        shape = NetworkShape(record.dims)
        n = _new_network(schema, shape)

        for ctype in sorted(record.entity_types):
            static = to_relation(record.entity_types[ctype])
            if not shape.stochastic and SCENARIO in static.columns:
                static = static.project(star(exclude=[SCENARIO]))
            # `_pos` in member order before any join scrambles it: the member
            # frame arrives in member order (https://energy-models.github.io/datarecord/design/read-path/#one-fold-for-every-axis),
            # and `_assign_static` sorts the wide frame back by it at the end.
            static = static.project("*, row_number() OVER () AS _pos")
            # Connections back to the positional columns PyPSA expects (https://energy-models.github.io/datarecord/design/tools/).
            static = _collapse_connections(static, record, ctype, con)

            # Frames are built and released per type, so peak memory is one
            # type's wide frames rather than the whole network (https://energy-models.github.io/datarecord/design/tools/).
            attributes = {}
            carried = schema.attributes_for(ctype)
            for attr, flags in record.flags(ctype).items():
                if attr not in carried:
                    continue
                # Both sets empty no longer means "no rows": the flags are
                # scoped to what an attribute is addressed by (https://energy-models.github.io/datarecord/design/record/#flags), so an
                # attribute over `entity` alone has no broadcast dim to report
                # and still has values. An attribute with no rows at all is
                # absent from the map entirely, which is what this filtered.
                # Through the schema, so a renamed or computed attribute
                # reaches the pivot below as an ordinary long relation. Scoped
                # by a semi-join against `static`, this type's entity table (https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types).
                long = self.schema.resolve(record, ctype, attr)
                # An attribute addressed by `entity` scopes to this type's
                # members; one addressed by an axis alone has no entity column
                # to scope by, and belongs to the record rather than to a
                # component (https://energy-models.github.io/datarecord/design/format/#the-long-schema).
                if ENTITY in long.columns:
                    long = long.set_alias("a").join(
                        static.project("entity").distinct().set_alias("m"),
                        "a.entity = m.entity",
                        how="semi",
                    )
                attributes[attr] = (long, flags)

            _add_component_type(n, ctype, static, attributes, shape, con)

        _broadcast_standard_types(n)
        return n

    def results(self, model: pypsa.Network) -> Frames:
        """A solved network's result attributes in the record's long form.

        Keyed by attribute, matching `outputs/<attr>.parquet`: every component
        type's rows for one attribute are concatenated into one frame, exactly as
        `attributes` presents `inputs/`. The union needs no
        `entity_type` to tell the arms apart. Which attributes count as
        results comes from PyPSA's registry, so an upgrade that adds one is
        picked up.

        `_as_long` reshapes a solved `Network`'s in-memory containers eagerly;
        the frames are wrapped with `.lazy()` and concatenated as a plan, so the
        union costs nothing until collected.

        Rows still at the attribute's default are dropped: a static output like
        `Bus.control` carries its default whether or not the network was solved,
        and an absent output file means exactly "take the default".

        Notes
        -----
        - [the broadcast rule](https://energy-models.github.io/datarecord/design/record/#the-broadcast-rule)
        - [the long schema](https://energy-models.github.io/datarecord/design/format/#the-long-schema)
        - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
        - [outputs](https://energy-models.github.io/datarecord/design/read-path/#outputs)
        """
        per_attribute: dict[str, list[nw.LazyFrame]] = {}
        for c in model.components:
            if not _exported(c):
                continue
            for attr in _output_attributes(c):
                try:
                    long = _as_long(c, attr, drop_defaults=True)
                except (AttributeError, KeyError):
                    # Not every registry output attribute has a container on
                    # every network (e.g. duals absent unless assigned).
                    continue
                if long.empty:
                    continue
                frame = (
                    nw.from_native(long, eager_only=True)
                    .with_columns(attribute=nw.lit(attr))
                    .lazy()
                )
                per_attribute.setdefault(attr, []).append(frame)
        return {
            attr: frames[0] if len(frames) == 1 else nw.concat(frames)
            for attr, frames in per_attribute.items()
        }

    def to_datarecord(self, model: pypsa.Network) -> RecordLike:
        """Present a `Network` as the `Record` `write_record` persists.

        The inverse of `build`, and the only place PyPSA's shape is undone:
        `c.static`/`c.dynamic` become long rows, and `bus0`/`bus1`/`efficiency2`
        become connection rows carrying a `role`. Key sets are read off the
        network, so listing unpivots nothing and a lookup only what is asked for.

        Raises
        ------
        UnsupportedRecordError
            If two component types share a name: PyPSA scopes names per type, a
            record record-wide, and this reports rather than renames.

        Notes
        -----
        - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
        - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
        - [writing a whole record](https://energy-models.github.io/datarecord/design/writing/)
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
        """
        clashing = _colliding_names(model)
        if clashing:
            raise UnsupportedRecordError(self.name, Requirements(names=clashing))
        return _NetworkSource(model)


@dataclass(frozen=True)
class _NetworkSource:
    """A `pypsa.Network` presented as a `Record`.

    Every `LazyFrames` here is built from key sets read off the network and
    its registry, so nothing is unpivoted until a key is looked up.

    Notes
    -----
    - [the record format](https://energy-models.github.io/datarecord/design/format/)
    """

    n: pypsa.Network

    # The dims a PyPSA network has axes for; `Coords` stays generic, this tool
    # decides these three are what a network is shaped by (https://energy-models.github.io/datarecord/design/tools/).
    _DIMS = (SNAPSHOT, PERIOD, SCENARIO)

    @property
    def schema(self) -> RecordSchema:
        """The layer's schema, derived from PyPSA's own registry.

        `c.defaults` already declares what an `AttributeSpec` asks for, so this reads it
        rather than restating it. Every stored attribute, not only the varying
        ones - `dims=frozenset()` is what puts one in `dims/entity_type/`.

        Results go to `results` rather than `attributes`, read off the same
        registry (`status` starting "Output"), so a PyPSA upgrade adding one is
        still picked up rather than needing a list kept here. They get no
        grant: `types` is the input vocabulary a type is validated and split
        against, and a result is neither.

        Notes
        -----
        - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
        - [the schema](https://energy-models.github.io/datarecord/design/schema/)
        - [AttributeSpec](https://energy-models.github.io/datarecord/design/schema/#attributespec)
        - [outputs](https://energy-models.github.io/datarecord/design/read-path/#outputs)
        """
        # What PyPSA's `varying` flag means as coordinates: the time axis, plus
        # the scenario axis where the network has one. A stochastic network's
        # series differ per scenario, so an attribute not declaring it would be
        # one the fold owns once across every scenario.
        varying_dims = (SNAPSHOT, SCENARIO) if self.n.has_scenarios else (SNAPSHOT,)
        # Which end of the component an attachment is, which PyPSA's sign
        # convention reads off the port index (`_port_role`). Declared over the
        # `connection` group, so it is a column of that group's file rather than
        # a long row - and so the record layer needs no rule of its own for a
        # word that is this tool's vocabulary (https://energy-models.github.io/datarecord/design/format/#where-a-value-lives).
        attributes: dict[str, AttributeSpec] = {
            # Facet-free like every per-component spec below: in a typed schema
            # a description lives on the grants, so `role`'s is repeated per
            # granting type rather than stated here.
            ROLE: AttributeSpec(dtype=nw.String(), dims=frozenset({CONNECTION}))
        }
        role_grant = TypeAttribute(
            description="Which end of the component this attachment is."
        )
        if self.n.has_scenarios:
            # A probability per scenario, which `dims` addresses by the scenario
            # axis alone - so it is a column of `dims/scenario.parquet`, where
            # `scenario_weightings` already puts it, rather than a long row.
            # Declared because it is data a consumer reads back: `set_scenarios`
            # takes it on import, and an undeclared column would carry no dtype.
            attributes[SCENARIO_WEIGHT] = AttributeSpec(
                dtype=nw.Float64(),
                dims=frozenset({SCENARIO}),
                description="How much this scenario counts in the expectation.",
            )
        results: dict[str, AttributeSpec] = {}
        types: dict[str, TypeSpec] = {}
        for c in self.n.components:
            if not _exported(c):
                continue
            defaults = c.defaults
            # Per-port columns collapse to their stem, which a record keys by
            # bus rather than by position (https://energy-models.github.io/datarecord/design/record/#connections); `efficiency2` and `efficiency`
            # agree on every declaration, so either row answers.
            per_port = self._port_stems(c)
            outputs = set(_output_attributes(c))
            grants: dict[str, TypeAttribute] = {}
            for attr in defaults.index:
                if attr == "name":
                    continue
                stem = per_port.get(attr, attr)
                row = defaults.loc[attr]
                # `entity` is a dim like any other, so a component attribute
                # declares it: `dims` is the whole address, and an attribute
                # omitting it would be one no component owns (https://energy-models.github.io/datarecord/design/schema/#attributespec).
                # A per-port attribute names the `connection` group instead,
                # which expands to `(entity, bus)` (https://energy-models.github.io/datarecord/design/schema/#groups).
                dims = set(varying_dims) if row["varying"] else set()
                dims.add(CONNECTION if attr in per_port else ENTITY)
                dtype = _DTYPES.get(row["typ"], nw.String())
                # One attribute, one spec, record-wide: two types declaring the
                # same name must agree on its shape, since one
                # `<kind>/<attr>.parquet` with one `value` column serves both.
                # PyPSA's registry does agree everywhere today, so the first
                # type to declare one wins and a later disagreement is a schema
                # error rather than a silent per-type divergence the storage
                # could not have honoured. Default, unit and description are
                # free to differ per type, and land on the grant below.
                if attr in outputs:
                    # A result is declared but not granted, `types` being the
                    # input vocabulary a type is validated and split against -
                    # so its facets stay on the spec, first type wins.
                    results.setdefault(
                        stem,
                        AttributeSpec(
                            dtype=dtype,
                            dims=frozenset(dims),
                            default=_default(row["default"]),
                            unit=_text(row.get("unit")),
                            description=_text(row.get("description")),
                        ),
                    )
                    continue
                if stem in grants:
                    continue
                grants[stem] = TypeAttribute(
                    default=_default(row["default"]),
                    unit=_text(row.get("unit")),
                    description=_text(row.get("description")),
                )
                attributes.setdefault(
                    stem, AttributeSpec(dtype=dtype, dims=frozenset(dims))
                )
            # A stem whose record-wide spec no entity addresses is no grant: a
            # stochastic record's `weight` is the scenario axis's column, and a
            # same-named registry attribute loses to it (first declaration wins)
            # rather than granting a per-type reading the file cannot hold.
            grants = {
                a: g
                for a, g in grants.items()
                if {ENTITY, CONNECTION} & attributes[a].dims
            }
            if c.ports:
                grants[ROLE] = role_grant
            if grants:
                types[c.name] = TypeSpec(
                    attributes=grants,
                    description=_text(_component_description(c.name)),
                )
        # A name PyPSA registers as an output on one type and an input on
        # another would be both here; the input declaration wins, one file
        # holding one `value` column either way.
        results = {a: s for a, s in results.items() if a not in attributes}
        return RecordSchema(
            dimensions={
                SNAPSHOT: Dimension(
                    dtype=nw.Datetime(),
                    description="A point in the operational time series.",
                ),
                PERIOD: Dimension(
                    dtype=nw.Int64(),
                    unit="year",
                    description="An investment period, labelled by its year.",
                ),
                SCENARIO: Dimension(
                    dtype=nw.String(),
                    description="One realisation of a stochastic problem.",
                ),
                # The two axes the `connection` group is over. `entity` is the
                # component axis the format knows by name; `bus` is an
                # ordinary dim, one coordinate of one group.
                ENTITY: Dimension(dtype=nw.String(), description="A component."),
                BUS: Dimension(dtype=nw.String(), description="A node of the network."),
                # The kinds a component may be: the labels are PyPSA's registry,
                # so an enum rather than a bare string, and a type outside it is
                # rejected on write.
                ENTITY_TYPE: Dimension(
                    dtype=nw.Enum(sorted(types)),
                    description="What kind of component an entity is.",
                ),
            },
            groups={
                CONNECTION: Group(
                    over={"entity": ENTITY, "bus": BUS},
                    description="A component's attachment to one bus.",
                ),
                # `into` over `entity` alone is what makes `entity_type` the
                # entity-type axis (https://energy-models.github.io/datarecord/design/schema/#entity_type-the-axis-of-kinds).
                ENTITY_TYPE: Group(
                    over={ENTITY: ENTITY},
                    into=ENTITY_TYPE,
                    description="What kind of component each entity is.",
                ),
            },
            attributes=attributes,
            results=results,
            types=types,
            # `partial` names value dims a layer patches per value: a layer may
            # set one generator's `p_nom` per scenario without restating the
            # rest. Membership keys - `entity`, the `connection` group's `bus` -
            # are in the fold key by being membership, not by being `partial`
            # (https://energy-models.github.io/datarecord/design/read-path/#one-fold-for-every-axis).
            partial=frozenset({SCENARIO}),
            meta={
                "format": "pypsa-parquet",
                "attributes": _collect_network_attributes(self.n),
                "meta": dict(self.n.meta),
            },
        )

    @property
    def dims(self) -> LazyFrames:
        """Every axis this network holds, `entity` among them.

        `entity` is an axis like the others rather than something the writer
        works out: a record supplies its own membership, so nothing downstream
        has to reconstruct it from the per-type files.

        Notes
        -----
        - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
        """
        axes = {
            SNAPSHOT: lambda: self.n.snapshots.to_frame(index=False),
            PERIOD: lambda: self.n.investment_periods.to_frame(index=False),
            # With its weights: an axis frame carries the whole row, not just
            # the key column (https://energy-models.github.io/datarecord/design/record/#axis-order), and `scenarios` alone is a bare Index.
            SCENARIO: lambda: self.n.scenario_weightings.reset_index(),
        }
        present = tuple(d for d in self._DIMS if len(axes[d]()) > 0)
        axes[ENTITY] = self._entity_axis_frame
        return LazyFrames(
            (*present, ENTITY), lambda dim: nw.from_native(axes[dim]()).lazy()
        )

    def _entity_axis_frame(self) -> pd.DataFrame:
        """`(entity, entity_type, deleted)` over every exported type.

        The type is a column here and in no member file: one file per type is
        what says a row's type there, and this axis is what carries it for every
        later reader.

        Notes
        -----
        - [entity is unique across types](https://energy-models.github.io/datarecord/design/format/#entity-is-unique-across-types)
        """
        frames = [
            self._entity_type_frame(ctype)
            .select(ENTITY, "deleted")
            .with_columns(**{ENTITY_TYPE: nw.lit(ctype)})
            .collect(backend="pandas")
            .to_native()
            for ctype in self.entity_types
        ]
        if not frames:
            return pd.DataFrame(columns=[ENTITY, ENTITY_TYPE, "deleted"])
        return pd.concat(frames, ignore_index=True)

    @property
    def entity_types(self) -> LazyFrames:
        types = tuple(c.name for c in self.n.components if _exported(c))
        return LazyFrames(types, self._entity_type_frame)

    @property
    def groups(self) -> LazyFrames:
        """The `connection` group, one frame across every type.

        Every type with any port, single-attachment ones included: a
        Generator's one `bus` is as much a connection as a Link's `bus0`,
        and `c.ports == [""]` makes `_port_attribute` name it correctly.

        The entity-type group is not here, its rows being the entity axis's -
        `dims["entity"]`, which this record supplies like any other axis.

        Notes
        -----
        - [groups](https://energy-models.github.io/datarecord/design/schema/#groups)
        """
        return LazyFrames((CONNECTION,), lambda _: self._connection_frame())

    def _connection_frame(self) -> nw.LazyFrame:
        """Every type's connections in one frame, `(entity, bus, role)`.

        Notes
        -----
        - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
        """
        types = [c.name for c in self.n.components if _exported(c) and c.ports]
        frames = [_connection_rows(self.n.c[ctype]) for ctype in types]
        rows = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["entity", "bus", "role"])
        )
        # A stochastic network repeats its connections per scenario, which
        # collapse to one row: an attachment exists or it does not, so nothing
        # scopes it per value of an axis (`_tagged` says the same of members).
        if SCENARIO in rows.columns:
            rows = rows.drop(columns=[SCENARIO]).drop_duplicates(
                subset=["entity", "bus"]
            )
        return nw.from_native(rows).lazy()

    @property
    def attributes(self) -> LazyFrames:
        names: dict[str, list[str]] = {}
        for c in self.n.components:
            if not _exported(c):
                continue
            for attr in self._record_attributes(c):
                names.setdefault(attr, []).append(c.name)
        return LazyFrames(
            tuple(names), lambda attr: self._long_frame(attr, names[attr])
        )

    @property
    def outputs(self) -> LazyFrames:
        """Result attributes, keyed by name, from PyPSA's own registry.

        Keys are named per type by the registry and merged, matching the file
        layout - one `outputs/<attr>.parquet` across types, like `inputs/`.
        A network that was never solved still names them; the frame is then
        empty and `write_record` writes an empty file rather than none, which
        reads back as "take the default" either way.

        Notes
        -----
        - [the broadcast rule](https://energy-models.github.io/datarecord/design/record/#the-broadcast-rule)
        - [outputs](https://energy-models.github.io/datarecord/design/read-path/#outputs)
        """
        names: dict[str, list[str]] = {}
        for c in self.n.components:
            if not _exported(c):
                continue
            for attr in _output_attributes(c):
                names.setdefault(attr, []).append(c.name)
        return LazyFrames(
            tuple(names), lambda attr: self._output_frame(attr, names[attr])
        )

    def flags(self, ctype: str) -> dict[str, Flags]:
        """Never consulted: `write_record` persists frames, not flags.

        A network-backed source exists to be written, and the write path reads
        only the frame mappings. Answering properly is easy - `c.static` and
        `c.dynamic` are PyPSA's own split on the snapshot axis, so it needs no
        scan - but an implementation no caller reaches is one no test pins, so
        the honest answer is the empty one.

        Notes
        -----
        - [writing a whole record](https://energy-models.github.io/datarecord/design/writing/)
        """
        return {}

    # -- key sets -----------------------------------------------------------

    def _record_attributes(self, c: pypsa.Components) -> list[str]:
        """`c`'s input attributes that go to `inputs/`, in the record's vocabulary.

        Those varying over `snapshot`, which PyPSA's registry flags - and those
        varying over `scenario`, which it does not, a stochastic network being
        free to hold a different `capital_cost` per scenario. Either way the
        attribute varies over something, so `inputs/` is where it lives.

        Per-port attributes collapse to their stem (`efficiency2` ->
        `efficiency`), a record keying them by bus rather than position.
        Outputs belong to `outputs/`.

        Notes
        -----
        - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
        - [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
        - [outputs](https://energy-models.github.io/datarecord/design/read-path/#outputs)
        """
        defaults = c.defaults
        outputs = set(_output_attributes(c))
        per_port = self._port_stems(c)
        diverging = _scenario_varying(c, [x for x in c.static.columns])
        stems: list[str] = []
        for attr in defaults.index:
            if attr in outputs or attr == "name":
                continue
            if not defaults.loc[attr, "varying"] and attr not in diverging:
                continue
            stem = per_port.get(attr, attr)
            if stem not in stems:
                stems.append(stem)
        return stems

    def _port_stems(self, c: pypsa.Components) -> dict[str, str]:
        """Per-port attribute name -> its stem, for this type's actual ports."""
        mapping = {}
        for stem in _PORT_STEMS:
            for port in c.ports:
                name = _port_attribute(stem, port)
                if name in c.defaults.index:
                    mapping[name] = stem
        return mapping

    # -- frames -------------------------------------------------------------

    def _entity_type_frame(self, ctype: str) -> nw.LazyFrame:
        """Wide members of one type: the non-varying, non-port static columns.

        Notes
        -----
        - [the Record protocol](https://energy-models.github.io/datarecord/design/record/)
        """
        c = self.n.c[ctype]
        defaults = c.defaults
        per_port = set(self._port_stems(c))

        def keep(column: str) -> bool:
            if column in per_port:
                return False  # a connection supplies it (https://energy-models.github.io/datarecord/design/record/#connections)
            if column not in defaults.index:
                # A custom column the registry has no entry for (e.g.
                # `ac_dc_meshed`'s `Bus.country`). Non-varying by construction -
                # a varying attribute needs a registry entry to have a
                # container - so it belongs in the static frame, and dropping
                # it would silently lose data the protocol says to keep.
                return True
            return (
                not defaults.loc[column, "varying"]
                and defaults.loc[column, "status"] != "Output"
            )

        columns = [x for x in c.static.columns if keep(x)]
        # A stochastic network repeats its static frame per scenario, and
        # PyPSA permits the repeats to differ - `capital_cost` may be one value
        # in `high` and another in `low`. Such an attribute varies over
        # `scenario`, so it belongs in `inputs/` and is dropped here; what is
        # left is the same in every scenario and collapses to one entity row.
        columns = [x for x in columns if x not in _scenario_varying(c, columns)]
        frame = c.static[columns].reset_index().rename(columns={"name": "entity"})
        return nw.from_native(self._tagged(frame)).lazy()

    @staticmethod
    def _tagged(frame: pd.DataFrame) -> pd.DataFrame:
        """A `dims/` frame with the tombstone column the fold scopes by.

        `deleted` because the fold reads it from this same file. Not the type -
        a per-type member file is the file its rows are in, and the entity axis
        (`_entity_axis_frame`) is what states it.

        No `scenario`: an entity exists or it does not, so nothing scopes
        membership per value of an axis. A stochastic network repeats its
        static frame per scenario, which is a *value* being per-scenario
        rather than the component - so the repeats collapse to one row here
        rather than reaching the record as duplicate entities.

        Notes
        -----
        - [deletion](https://energy-models.github.io/datarecord/design/layers/#deletion)
        - [the record format](https://energy-models.github.io/datarecord/design/format/)
        """
        if SCENARIO in frame.columns:
            frame = frame.drop(columns=[SCENARIO]).drop_duplicates(subset=["entity"])
        if "deleted" not in frame.columns:
            frame["deleted"] = False
        return frame

    def _long_frame(self, attribute: str, ctypes: list[str]) -> nw.LazyFrame:
        """One attribute's long rows, across every type that has it.

        A per-connection attribute contributes rows carrying the bus each port
        attaches to; a component-level one has no `bus` column at all, that
        being the connection group's coordinate rather than a column the format
        fixes.

        Notes
        -----
        - [the Record protocol](https://energy-models.github.io/datarecord/design/record/)
        - [wide and long rows](https://energy-models.github.io/datarecord/design/record/#wide-and-long-rows)
        """
        dims = self._DIMS
        frames = []
        for ctype in ctypes:
            c = self.n.c[ctype]
            if attribute in self._port_stems(c).values():
                rows = _per_port_long_rows(c, attribute, dims)
            elif attribute in c.defaults.index:
                rows = _long_rows(c, attribute, dims)
            else:
                continue
            if not rows.empty:
                frames.append(rows)
        # This attribute's own columns, from its spec: one file is one
        # attribute, so a component attribute hands over no `bus` and none of
        # them a dim they are not addressed by (https://energy-models.github.io/datarecord/design/format/#the-long-schema).
        columns = list(self.schema.long_columns_for(attribute))
        if not frames:
            return nw.from_native(pd.DataFrame(columns=columns)).lazy()
        # No dtype fixing here: `write_record` casts every schema column to the
        # record layer's declared type on the way out (https://energy-models.github.io/datarecord/design/writing/), so an all-NULL
        # column pandas typed as float lands as the right type regardless.
        return nw.from_native(pd.concat(frames, ignore_index=True)[columns]).lazy()

    def _output_frame(self, attribute: str, ctypes: list[str]) -> nw.LazyFrame:
        """One result attribute's long rows, across every type that has it.

        `drop_defaults` matters here as it does in `results`: a static output
        carries its default whether or not the network was solved, and an
        absent value means exactly "take the default" to a reader.

        Notes
        -----
        - [outputs](https://energy-models.github.io/datarecord/design/read-path/#outputs)
        """
        dims = self._DIMS
        frames = []
        for ctype in ctypes:
            c = self.n.c[ctype]
            try:
                long = _as_long(c, attribute, drop_defaults=True)
            except (AttributeError, KeyError):
                # Not every registry output has a container on every network.
                continue
            if long.empty:
                continue
            long = long.assign(attribute=attribute, breakpoint=None)
            for dim in dims:
                if dim not in long.columns:
                    long[dim] = None
            frames.append(long)
        # No `bus`: a result comes from a per-entity container, never a per-port
        # one - PyPSA keeps `Link.p0` and `p1` as separate result attributes
        # where it collapses the *input* `efficiency`/`efficiency2` to one over
        # the connection group. So there is no port to key a result by.
        columns = ["entity", *dims, "attribute", "breakpoint", "value"]
        if not frames:
            return nw.from_native(pd.DataFrame(columns=columns)).lazy()
        return nw.from_native(pd.concat(frames, ignore_index=True)[columns]).lazy()


def _component_description(ctype: str) -> Any:
    """`ctype`'s prose description, from PyPSA's type registry."""
    from pypsa.components.types import get as get_component_type

    return getattr(get_component_type(ctype), "description", None)


def _required_attributes(ctype: str) -> frozenset[str]:
    """`ctype`'s mandatory input attributes, from PyPSA's registry.

    `name` is the component's identity, always supplied by the record's member
    list rather than an attribute row, so it never counts as missing.
    """
    from pypsa.components.types import get as get_component_type

    defaults = get_component_type(ctype).defaults
    required = defaults.index[defaults["status"].astype(str) == "Input (required)"]
    return frozenset(required) - {"name"}


def _ordered_connections(record: RecordLike, ctype: str) -> pd.DataFrame | None:
    """One type's connections with a port index assigned per component.

    The positional collapse: connections come in member order (an
    overlay's `order_key`: first-introduced) and are numbered within each
    component, so a port index follows the order connections were introduced
    and a patch that adds one appends rather than renumbering. Inputs are
    placed before outputs, so `bus0` is the input end PyPSA's sign convention
    expects.

    Scoped to `ctype` by that type's entities, one frame holding every type's
    connections.

    Notes
    -----
    - [the owner map](https://energy-models.github.io/datarecord/design/read-path/#owner-map)
    - [one record over one fold](https://energy-models.github.io/datarecord/design/read-path/#one-record-over-one-fold)
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    frame = record.groups.get(CONNECTION)
    members = record.entity_types.get(ctype)
    if frame is None or members is None:
        return None
    df = frame.collect(backend="pandas").to_native()
    mine = set(members.select("entity").collect(backend="pandas").to_native()["entity"])
    df = df[df["entity"].isin(mine)]
    if df.empty:
        return None
    # Per component: ports are numbered within the entity they belong to.
    within = ["entity"]
    if "role" in df.columns:
        df = df.assign(_role_rank=(df["role"] != _INPUT_PORT).astype(int)).sort_values(
            [*within, "_role_rank"], kind="stable"
        )
    # A single-attachment component keeps PyPSA's unsuffixed `bus`, so its port
    # is the empty index rather than `0` (`_port_attribute`).
    # `dropna=False`: a dim left NULL means "every value of it" (https://energy-models.github.io/datarecord/design/record/#the-broadcast-rule), which
    # is an ordinary group here, not a row to drop.
    grouped = df.groupby(within, dropna=False)
    single = grouped["bus"].transform("size").eq(1) & df["role"].eq(_SINGLE_PORT)
    df["port"] = grouped.cumcount().astype(str).where(~single, "")
    return df


def _connection_attributes(record: RecordLike, ctype: str) -> frozenset[str]:
    """Port attribute names this type's connection rows can supply.

    `bus0`/`bus1`/... for the ports that actually exist, so `verify` knows a
    required `bus0` is satisfied by a connection rather than by a column.

    Notes
    -----
    - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
    """
    df = _ordered_connections(record, ctype)
    if df is None:
        return frozenset()
    return frozenset(_port_attribute("bus", port) for port in df["port"].unique())


def _collapse_connections(
    static: DuckDBPyRelation, record: RecordLike, ctype: str, con: DuckDBPyConnection
) -> DuckDBPyRelation:
    """Add `bus0`/`bus1`/... columns to a static frame from its connection rows.

    PyPSA wants a column per port; the record stores a row per connection. This is the seam between them, and the only direction that needs
    positions at all.

    Notes
    -----
    - [connections](https://energy-models.github.io/datarecord/design/record/#connections)
    """
    df = _ordered_connections(record, ctype)
    if df is None:
        return static
    # Pivoted on the same key the ports were numbered within, so each
    # (component, scenario, ...) keeps its own port columns.
    index = ["entity"]
    # A dim left NULL means "every value of it" (https://energy-models.github.io/datarecord/design/record/#the-broadcast-rule), so it is a real key
    # here - but `pivot` drops NaN index levels, so such a dim is folded away
    # instead: every row shares its value, and the join below matches on the
    # dims that actually distinguish rows.
    varying = [d for d in index[1:] if df[d].notna().any()]
    wide = df.pivot(index=["entity", *varying], columns="port", values="bus")
    wide.columns = [_port_attribute("bus", port) for port in wide.columns]
    wide = wide.reset_index()  # noqa: F841 - queried by name below
    ports = con.sql("FROM wide").set_alias("ports")
    on = ["entity", *(d for d in varying if d in static.columns)]
    joined = static.set_alias("s").join(
        ports, ex_all(col("s", c) == col("ports", c) for c in on), how="left"
    )
    return joined.project(
        *(col("s", c) for c in static.columns),
        *(col("ports", c) for c in ports.columns if c not in ("entity", *varying)),
    )


def _static_columns(record: RecordLike, ctype: str) -> frozenset[str]:
    """Columns `dims/entity_type/<ctype>.parquet` supplies for this record.

    The non-varying half of the static frame: an attribute present here
    needs no `inputs/` row to be resolvable.

    Notes
    -----
    - [the Record protocol](https://energy-models.github.io/datarecord/design/record/)
    """
    frame = record.entity_types.get(ctype)
    return (
        frozenset(frame.collect_schema().names()) if frame is not None else frozenset()
    )


# The tool is a module-level singleton, imported rather than looked up by
# name: `from datarecord.tools.pypsa import PyPSA` then `PyPSA.build(revision.record)`.
PyPSA = PyPSATool()
