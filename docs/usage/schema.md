<!--
SPDX-FileCopyrightText: datarecord contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# The schema

One schema per record, written down as `manifest.json` ([design](../design/schema.md)).

```python
import narwhals as nw
from datarecord import (
    AttributeSpec,
    Dimension,
    Group,
    Schema,
    TypeAttribute,
    TypeSpec,
)

schema = Schema(
    version=1,
    dimensions={
        "entity": Dimension(dtype=nw.String()),
        "entity_type": Dimension(dtype=nw.Enum(["Bus", "Generator", "Link"])),
        "bus": Dimension(dtype=nw.String()),
        "scenario": Dimension(dtype=nw.String()),
        "timestep": Dimension(dtype=nw.Datetime()),
    },
    groups={
        "connection": Group(over={"entity": "entity", "bus": "bus"}),
        "entity_type": Group(over=["entity"], into="entity_type"),
    },
    attributes={
        "p_nom": AttributeSpec(dtype=nw.Float64(), dims={"entity"}),
        "carrier": AttributeSpec(dtype=nw.String(), dims={"entity"}),
        "p_max_pu": AttributeSpec(
            dtype=nw.Float64(), dims={"entity", "scenario", "timestep"}
        ),
        "efficiency": AttributeSpec(
            dtype=nw.Float64(), dims={"connection", "timestep"}
        ),
    },
    types={
        "Bus": TypeSpec(attributes={"carrier": TypeAttribute()}),
        "Generator": TypeSpec(
            attributes={
                "carrier": TypeAttribute(),
                "p_nom": TypeAttribute(default=0.0, unit="MW"),
                "p_max_pu": TypeAttribute(default=1.0),
                "efficiency": TypeAttribute(default=1.0),
            }
        ),
        "Link": TypeSpec(
            attributes={
                "carrier": TypeAttribute(),
                "p_nom": TypeAttribute(default=0.0, unit="MW"),
                "efficiency": TypeAttribute(default=1.0),
            }
        ),
    },
    partial={"entity", "bus", "scenario"},
)
```

- **`Dimension`** declares one axis: its `dtype` (a narwhals dtype, translated to its DuckDB name — a DuckDB type name works too, for a type narwhals does not spell) and `within` for an axis whose labels identify a point only inside another's — multi-period time being the case ([design](../design/schema.md#within-an-axis-inside-an-axis)).
- **`Group`** declares which tuples over several dims exist, mapping coordinate name → dim; a list is sugar where the two coincide. `connection` over `(entity, bus)` is the one every network has. Adding **`into`** declares the group _functional_ into that dim — each `over` tuple carrying exactly one of its labels — which is a constraint checkable on write ([design](../design/schema.md#groups)).
- **`entity_type` is an ordinary functional group** — `over=["entity"], into="entity_type"` — so its rows land on `dims/entity.parquet` like any classification's. An `Enum` pins the vocabulary; a plain `nw.String()` leaves the labels as data. Omit the group entirely and components have no types ([design](../design/schema.md#entity_type-the-axis-of-kinds)).
- **`attributes` is flat** — one attribute, one spec, record-wide for what the file fixes (`dtype`, `dims`, `breakpoints`) — and **`types` grants** each attribute to the entity types that carry it, with the per-type facets (`default`, `unit`, `description`) on the grant. `p_max_pu` above reaches only `Generator` because only its entry grants it. `Schema.attributes_for("Generator")` hands back each granted spec with that type's facets in place ([design](../design/schema.md#types-what-a-type-carries)). A `Trait` is a capability some components of a type opt into via a boolean switch attribute, and says nothing about presence ([design](../design/schema.md#traits)).
- **`AttributeSpec.dims`** is the only addressing mechanism, naming dims and groups alike — `efficiency` is a connection attribute because its `dims` name the group. A name is the dim of that name if one is declared, and otherwise the group expanded to its coordinates ([design](../design/schema.md#addressing-dims-x)). It is what makes a scenario-varying `p_nom` a violation rather than data, and it decides the file split: naming exactly one coordinate puts an attribute on that thing's own table, anything more in `inputs/` ([design](../design/format.md#where-a-value-lives)).
- **`partial`** is the layering granularity — which dims a layer may patch value by value. `scenario` is patchable; `timestep` is not, so a patch to one hour restates that component's whole series rather than leaving a curve resolved across two layers with a hole in it ([design](../design/schema.md#partial-the-granularity-of-an-override)). `entity` and every group _key_ coordinate must be in it, since [neither broadcasts](../design/record.md#the-broadcast-rule). Omit it entirely for a record with no layers.
- **`unit`** and **`description`** are stored and never interpreted — no conversion, no dimensional analysis. `None` is undeclared, `""` genuinely dimensionless ([design](../design/schema.md#unit-and-description)).

## Compatibility

`schema.compatible_with(other)` answers whether layers written under `other` still read under `self`, returning one reason per incompatibility and an empty list when the change is compatible ([design](../design/schema.md#versioning)).

```python
reasons = new_schema.compatible_with(old_schema)
if reasons:
    raise ValueError("\n".join(reasons))
```
