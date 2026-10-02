# SPDX-FileCopyrightText: datarecord contributors
#
# SPDX-License-Identifier: MIT

"""The typed schema: declarations, derived keys, validation, versioning.

Notes
-----
- [the schema](https://energy-models.github.io/datarecord/design/schema/)
"""

from typing import Any

import duckdb
import narwhals as nw
import pytest
from pydantic import ValidationError

from datarecord.duck import DuckTypes
from datarecord.schema import (
    AttributeSpec,
    Dimension,
    Group,
    Schema,
    Trait,
    TypeAttribute,
    TypeSpec,
    flag_type,
)


def _schema(**overrides) -> Schema:
    """A schema shaped like a stochastic multi-period record."""
    kwargs: dict[str, Any] = {
        "dimensions": {
            "entity": Dimension(dtype=nw.String()),
            "bus": Dimension(dtype=nw.String()),
            "period": Dimension(dtype=nw.Int64()),
            "timestep": Dimension(dtype=nw.Datetime(), within={"period"}),
            "scenario": Dimension(dtype=nw.String()),
            "entity_type": Dimension(dtype=nw.Enum(["Generator", "Link"])),
        },
        "groups": {
            "connection": Group(over={"entity": "entity", "bus": "bus"}),
            # The functional group over `entity` alone is what makes
            # `entity_type` the entity-type axis.
            "entity_type": Group(over=["entity"], into="entity_type"),
        },
        # Declared once, record-wide; `types` grants one to the types that
        # carry it, with the per-type facets on the grant.
        "attributes": {
            "p_nom": AttributeSpec(dtype=nw.Float64(), dims={"entity"}),
            "p_max_pu": AttributeSpec(
                dtype=nw.Float64(), dims={"entity", "scenario", "timestep"}
            ),
            "marginal_cost": AttributeSpec(
                dtype=nw.Float64(), dims={"entity", "scenario"}, breakpoints=True
            ),
            "carrier": AttributeSpec(dtype=nw.String(), dims={"entity"}),
            # A connection attribute says so by naming the group among its
            # dims, rather than by a field of its own.
            "efficiency": AttributeSpec(
                dtype=nw.Float64(), dims={"connection", "scenario", "timestep"}
            ),
        },
        "types": {
            "Generator": TypeSpec(
                attributes={
                    "p_nom": TypeAttribute(),
                    "p_max_pu": TypeAttribute(),
                    "marginal_cost": TypeAttribute(),
                    "carrier": TypeAttribute(),
                }
            ),
            "Link": TypeSpec(
                attributes={
                    "carrier": TypeAttribute(),
                    "efficiency": TypeAttribute(),
                }
            ),
        },
        "partial": frozenset({"scenario"}),
    }
    kwargs.update(overrides)
    return Schema(**kwargs)


# -- derived keys (https://energy-models.github.io/datarecord/design/schema/#partial-the-granularity-of-an-override) -----------------------------------------------------


def test_ownership_is_derived_not_declared():
    """`owned_per` is `dims` and `partial` together, never a third declaration."""
    s = _schema()
    # Varies over both time axes, but only `scenario` is partial among them -
    # so a patch to one timestep restates that scenario's whole series.
    # `entity` joins it because a layer patches one component's value.
    assert s.owned_per("p_max_pu") == frozenset({"entity", "scenario"})
    assert s.owned_per("marginal_cost") == frozenset({"entity", "scenario"})
    # A first-stage decision: one value per component, owned once across
    # every axis it does not vary over.
    assert s.owned_per("p_nom") == frozenset({"entity"})
    assert s.owned_per("carrier") == frozenset({"entity"})


def test_a_scenario_varying_capacity_is_a_schema_violation():
    """`dims` is what forbids it: a capacity is decided before the scenario is known."""
    s = _schema()
    assert "scenario" not in s.attributes["p_nom"].dims
    # Nothing owns it per scenario, so the fold writes NULL there and one value
    # applies to every scenario.
    assert s.owned_per("p_nom") == frozenset({"entity"}), (
        "owned per entity, not scenario"
    )


def test_partial_dims_is_the_union_over_attributes():
    """The fold's key is one fixed tuple, so an unowned dim is NULL rather than absent."""
    s = _schema()
    # `entity` and `bus` are always partial - they do not broadcast.
    assert s.partial_dims == ("entity", "bus", "scenario")

    # Make `timestep` partial too and it joins the key.
    wider = _schema(partial=frozenset({"scenario", "timestep"}))
    assert wider.partial_dims == ("entity", "bus", "timestep", "scenario")


def test_file_split_follows_dims():
    """Varying over nothing is what puts an attribute in `dims/entity_type/`.

    Notes
    -----
    - [AttributeSpec](https://energy-models.github.io/datarecord/design/schema/#attributespec)
    """
    s = _schema()
    assert not s.attributes["p_nom"].varying
    assert not s.attributes["carrier"].varying
    assert s.attributes["p_max_pu"].varying


# -- membership keys (https://energy-models.github.io/datarecord/design/schema/#keys-which-entity-tables-a-dim-keys) --------------------------------------------------


def test_a_membership_key_is_in_the_fold_key_without_partial():
    """A non-broadcast dim is a membership key, in the fold key by being one.

    `entity` does not broadcast, so it is patched per row by every layer - it
    lands in `partial_dims` (the fold key) without being declared `partial`,
    which is for value dims a layer patches per value.
    """
    s = Schema(
        dimensions={
            "entity": Dimension(dtype=nw.String()),
            "scenario": Dimension(dtype=nw.String()),
        },
        partial=frozenset({"scenario"}),
    )
    assert s.membership_keys == ("entity",)
    assert s.partial_dims == ("entity", "scenario")


def test_partial_may_not_name_a_membership_key():
    """`partial` is for value dims; a membership key named there is a category error."""
    with pytest.raises(ValidationError, match="membership keys"):
        Schema(
            dimensions={
                "entity": Dimension(dtype=nw.String()),
                "scenario": Dimension(dtype=nw.String()),
            },
            partial=frozenset({"entity", "scenario"}),
        )


# -- entity types and grants (https://energy-models.github.io/datarecord/design/schema/#types-what-a-type-carries) ------------------------------------


def test_presence_is_the_grant():
    """`attributes_for` reads `types[ctype]` and nothing else."""
    s = _schema()
    assert "p_max_pu" in s.attributes_for("Generator")
    assert "p_max_pu" not in s.attributes_for("Link"), "granted to Generator alone"
    assert "carrier" in s.attributes_for("Link"), "granted to both"


def test_an_ungranted_entity_attribute_is_rejected():
    """Declared but reachable by no type is a forgotten grant or a typo."""
    with pytest.raises(ValidationError, match="granted by no type"):
        _schema(
            attributes={
                **_schema().attributes,
                "sign": AttributeSpec(dtype=nw.Float64(), dims={"entity"}),
            }
        )


def test_a_grant_carries_the_per_type_facets():
    """One attribute, one spec - and each type reads it with its own facets.

    `Schema.attributes[a]` is untouched; the copy `attributes_for` hands back
    is the resolved per-type view.
    """
    base = _schema()
    s = _schema(
        attributes={
            **base.attributes,
            "sign": AttributeSpec(dtype=nw.Float64(), dims={"entity"}),
        },
        types={
            "Generator": base.types["Generator"].model_copy(
                update={
                    "attributes": {
                        **base.types["Generator"].attributes,
                        "sign": TypeAttribute(default=1.0, unit=""),
                    }
                }
            ),
            "Link": base.types["Link"].model_copy(
                update={
                    "attributes": {
                        **base.types["Link"].attributes,
                        "sign": TypeAttribute(default=-1.0, description="Flipped."),
                    }
                }
            ),
        },
    )
    assert s.attributes_for("Generator")["sign"].default == 1.0
    assert s.attributes_for("Generator")["sign"].unit == ""
    assert s.attributes_for("Link")["sign"].default == -1.0
    assert s.attributes_for("Link")["sign"].description == "Flipped."
    assert s.attributes["sign"].default is None, "the record-wide spec is untouched"
    assert s.types_declaring("sign") == frozenset({"Generator", "Link"})


def test_types_keys_must_equal_the_labels():
    """A missing key and a stray key are both typos worth an error."""
    base = _schema()
    with pytest.raises(ValidationError, match="missing \\['Link'\\]"):
        _schema(types={"Generator": base.types["Generator"]})
    with pytest.raises(ValidationError, match="unknown \\['Store'\\]"):
        _schema(types={**base.types, "Store": TypeSpec()})


def test_types_requires_an_enum_axis():
    """A `str`-typed axis keeps its labels as data, so there is nothing to grant to."""
    base = _schema()
    with pytest.raises(ValidationError, match="Enum dtype"):
        _schema(
            dimensions={
                **base.dimensions,
                "entity_type": Dimension(dtype=nw.String()),
            }
        )


def test_a_grant_must_name_a_declared_entity_attribute():
    base = _schema()

    def granting(attribute: str) -> dict:
        grants = {**base.types["Generator"].attributes, attribute: TypeAttribute()}
        return {**base.types, "Generator": TypeSpec(attributes=grants)}

    with pytest.raises(ValidationError, match="undeclared attribute 'nope'"):
        _schema(types=granting("nope"))
    with pytest.raises(ValidationError, match="no entity addresses"):
        _schema(
            attributes={
                **base.attributes,
                "weighting": AttributeSpec(dtype=nw.Float64(), dims={"timestep"}),
            },
            types=granting("weighting"),
        )
    with pytest.raises(ValidationError, match="which is a result"):
        _schema(
            results={"p": AttributeSpec(dtype=nw.Float64(), dims={"entity"})},
            types=granting("p"),
        )


def test_a_spec_level_facet_is_rejected_in_a_typed_schema():
    """Facets live where the attribute is addressed, so one fact has one slot."""
    with pytest.raises(ValidationError, match="live on the grants"):
        _schema(
            attributes={
                **_schema().attributes,
                "p_nom": AttributeSpec(dtype=nw.Float64(), dims={"entity"}, unit="MW"),
            }
        )


def test_an_attribute_addressing_no_entity_reaches_no_type():
    """A record-level attribute belongs to the record, and needs no grant.

    With no `entity` among its coordinates there is no component for it to
    reach, so it escapes the granted-by-no-type error and `names=None` targets
    nothing rather than every component in the record.
    """
    s = _schema(
        attributes={
            **_schema().attributes,
            "weighting": AttributeSpec(dtype=nw.Float64(), dims={"timestep"}),
        }
    )
    assert not s.addresses_entity("weighting")
    assert "weighting" not in s.attributes_for("Generator")
    assert s.types_declaring("weighting") == frozenset()


def test_a_group_addressed_attribute_is_granted_like_any_other():
    """`efficiency` is over `connection`, whose coordinates include `entity`."""
    s = _schema()
    assert s.addresses_entity("efficiency")
    assert "efficiency" in s.attributes_for("Link")
    assert "efficiency" not in s.attributes_for("Generator")


def _committable(*, grant_bundle_to: set[str]) -> Schema:
    """`_schema` plus a `committable` capability, its grants set per type."""
    base = _schema()
    extra = {"committable", "start_up_cost"}
    return _schema(
        attributes={
            **base.attributes,
            "start_up_cost": AttributeSpec(dtype=nw.Float64(), dims={"entity"}),
            "committable": AttributeSpec(dtype=nw.Boolean(), dims={"entity"}),
        },
        types={
            ctype: spec.model_copy(
                update={
                    "attributes": {
                        **spec.attributes,
                        **{
                            a: TypeAttribute()
                            for a in (extra if ctype in grant_bundle_to else ())
                        },
                    }
                }
            )
            for ctype, spec in base.types.items()
        },
        traits={
            "committable": Trait(attributes={"start_up_cost"}, switch="committable")
        },
    )


def test_a_trait_requires_a_switch():
    """A trait that gates nothing says nothing: presence is `types`' alone."""
    with pytest.raises(ValidationError, match="switch"):
        # The missing argument is the case under test.
        Trait(attributes={"start_up_cost"})  # type: ignore[call-arg]


def test_a_trait_switch_is_folded_into_attributes():
    """`switch` need not be named twice in `attributes`."""
    t = Trait(attributes={"start_up_cost"}, switch="committable")
    assert t.attributes == {"start_up_cost", "committable"}


def test_a_trait_switch_narrows_neither_attributes_for_nor_the_switch_itself():
    """`switch` is a validation and query mechanism, not a change to `attributes_for`.

    The schema-level answer stays the grants': `committable` and its bundle are
    carried by every type granted them, whatever any component's switch value.
    """
    s = _committable(grant_bundle_to={"Generator"})
    assert "start_up_cost" in s.attributes_for("Generator")
    assert "committable" in s.attributes_for("Generator")
    assert "start_up_cost" not in s.attributes_for("Link")


def test_a_traits_bundle_travels_with_its_switch():
    """A type granted the switch without the bundle has a gate on nothing."""
    base = _schema()
    grants = {
        **base.types["Generator"].attributes,
        "committable": TypeAttribute(),
    }
    with pytest.raises(ValidationError, match="bundle travels with its switch"):
        _schema(
            attributes={
                **base.attributes,
                "start_up_cost": AttributeSpec(dtype=nw.Float64(), dims={"entity"}),
                "committable": AttributeSpec(dtype=nw.Boolean(), dims={"entity"}),
            },
            types={
                **base.types,
                "Generator": TypeSpec(attributes=grants),
                "Link": base.types["Link"].model_copy(
                    update={
                        "attributes": {
                            **base.types["Link"].attributes,
                            "start_up_cost": TypeAttribute(),
                        }
                    }
                ),
            },
            traits={
                "committable": Trait(attributes={"start_up_cost"}, switch="committable")
            },
        )


def test_a_switched_trait_needs_no_entity_type_axis():
    """A switch alone narrows by component, with no type scope at all."""
    s = Schema(
        dimensions={"entity": Dimension(dtype=nw.String())},
        attributes={
            "committable": AttributeSpec(
                dtype=nw.Boolean(), dims={"entity"}, default=False
            ),
        },
        traits={"committable": Trait(switch="committable")},
        partial=frozenset(),
    )
    assert s.traits["committable"].attributes == {"committable"}


def test_a_trait_switch_must_be_addressed_by_entity_alone():
    """A switch narrower or wider than `entity` alone has no per-component reading."""
    with pytest.raises(ValidationError, match="addressed by `entity` alone"):
        Schema(
            dimensions={
                "entity": Dimension(dtype=nw.String()),
                "scenario": Dimension(dtype=nw.String()),
            },
            attributes={
                "committable": AttributeSpec(
                    dtype=nw.Boolean(), dims={"entity", "scenario"}, default=False
                ),
            },
            traits={"committable": Trait(switch="committable")},
            partial=frozenset({"scenario"}),
        )


def test_a_functional_group_may_not_key_an_attribute_with_what_it_maps_from():
    """`into` says the label follows from the key, so the row is keyed twice.

    Stated for the entity-type axis, which is the case the format names, but
    the rule is general - see the `country`-over-`bus` case below.
    """
    with pytest.raises(ValidationError, match="keys a row twice over"):
        Schema(
            dimensions={
                "entity": Dimension(dtype=nw.String()),
                "entity_type": Dimension(dtype=nw.Enum(["Bus"])),
            },
            groups={"entity_type": Group(over=["entity"], into="entity_type")},
            attributes={
                "p_nom": AttributeSpec(
                    dtype=nw.Float64(), dims={"entity", "entity_type"}
                )
            },
            partial=frozenset(),
        )


def test_the_redundant_addressing_rule_covers_every_functional_group():
    """Not an `entity_type` special case: `country` over `bus` is the same shape."""
    with pytest.raises(ValidationError, match="keys a row twice over"):
        Schema(
            dimensions={
                "bus": Dimension(dtype=nw.String()),
                "country": Dimension(dtype=nw.String()),
            },
            groups={"in_country": Group(over=["bus"], into="country")},
            attributes={
                "x": AttributeSpec(dtype=nw.Float64(), dims={"bus", "country"})
            },
            partial=frozenset(),
        )


def test_an_attribute_may_be_addressed_by_the_entity_type_alone():
    """A per-type icon is a value per type, keyed once - an axis-file column.

    The type axis is a dim like any other; what may not key a row alongside it
    is the `entity` the group maps into it.
    """
    s = Schema(
        dimensions={
            "entity": Dimension(dtype=nw.String()),
            "entity_type": Dimension(dtype=nw.Enum(["Bus", "Generator"])),
        },
        groups={"entity_type": Group(over=["entity"], into="entity_type")},
        attributes={
            "p_nom": AttributeSpec(dtype=nw.Float64(), dims={"entity"}),
            "icon": AttributeSpec(dtype=nw.String(), dims={"entity_type"}),
        },
        types={
            "Bus": TypeSpec(),
            "Generator": TypeSpec(attributes={"p_nom": TypeAttribute()}),
        },
        partial=frozenset(),
    )
    assert s.attributes_on("entity_type") == ("icon",), "a column of the type axis"
    assert not s.attributes["icon"].varying, "addressed by one dim, so not varying"
    assert "icon" not in s.attributes_for("Bus"), "it belongs to no component"


def test_only_one_group_may_classify_entity():
    """A component has one type, so two vocabularies have no resolved answer."""
    with pytest.raises(ValidationError, match="all classify `entity`"):
        Schema(
            dimensions={
                "entity": Dimension(dtype=nw.String()),
                "entity_type": Dimension(dtype=nw.Enum(["Bus"])),
                "kind": Dimension(dtype=nw.Enum(["thing"])),
            },
            groups={
                "entity_type": Group(over=["entity"], into="entity_type"),
                "kind": Group(over=["entity"], into="kind"),
            },
            attributes={"p_nom": AttributeSpec(dtype=nw.Float64(), dims={"entity"})},
            partial=frozenset(),
        )


def test_an_entity_type_axis_is_no_long_schema_coordinate():
    """Its column is on the entity axis, so a long row never carries it.

    Nor does an attribute addressed by the type alone put it here: that is a
    column of the type axis file, not a long row.
    """
    s = _schema()
    assert "entity_type" not in s.long_columns
    assert "entity_type" not in s.broadcast_dims
    assert "entity_type" not in s.partial_dims


def test_a_record_may_declare_no_entity_type_at_all():
    """Types are optional: with no such axis every entity carries everything.

    A tool needing types requires the axis in the schema it builds, which is
    where that requirement belongs.
    """
    s = Schema(
        dimensions={
            "entity": Dimension(dtype=nw.String()),
            "timestep": Dimension(dtype=nw.Datetime()),
        },
        attributes={
            "p_nom": AttributeSpec(dtype=nw.Float64(), dims={"entity"}),
            "weighting": AttributeSpec(dtype=nw.Float64(), dims={"timestep"}),
        },
        partial=frozenset(),
    )
    assert s.entity_types == frozenset(), "no axis, so no declared vocabulary"
    assert sorted(s.attributes_for("anything")) == ["p_nom"], (
        "entity-addressed only, whatever label is asked for"
    )


def test_a_string_entity_type_axis_leaves_the_labels_as_data():
    """An `Enum` pins the vocabulary; a plain string does not declare one."""
    s = Schema(
        dimensions={
            "entity": Dimension(dtype=nw.String()),
            "entity_type": Dimension(dtype=nw.String()),
        },
        groups={"entity_type": Group(over=["entity"], into="entity_type")},
        attributes={"p_nom": AttributeSpec(dtype=nw.Float64(), dims={"entity"})},
        partial=frozenset(),
    )
    assert s.entity_type_dim == "entity_type"
    assert s.entity_types == frozenset(), "labels are data, not declarations"
    assert "p_nom" in s.attributes_for("Whatever")


# -- nesting (https://energy-models.github.io/datarecord/design/schema/#within-an-axis-inside-an-axis) ----------------------------------------------------------


def test_axis_key_is_parents_then_dim():
    """A nested axis's labels identify only within its parents."""
    s = _schema()
    assert s.axis_key("timestep") == ("period", "timestep")
    assert s.axis_key("period") == ("period",)
    assert s.axis_key("scenario") == ("scenario",)


def test_nesting_is_transitive():
    """Naming a parent pulls in that parent's own parents.

    Notes
    -----
    - [within](https://energy-models.github.io/datarecord/design/schema/#within-an-axis-inside-an-axis)
    """
    s = Schema(
        dimensions={
            "horizon": Dimension(dtype=nw.Int64()),
            "period": Dimension(dtype=nw.Int64(), within={"horizon"}),
            "timestep": Dimension(dtype=nw.Datetime(), within={"period"}),
        }
    )
    assert s.axis_key("timestep") == ("horizon", "period", "timestep")


def test_several_direct_parents():
    """A set, since two axes may each qualify a label without containing each other."""
    s = Schema(
        dimensions={
            "period": Dimension(dtype=nw.Int64()),
            "stage": Dimension(dtype=nw.String()),
            "timestep": Dimension(dtype=nw.Datetime(), within={"period", "stage"}),
        }
    )
    assert s.axis_key("timestep") == ("period", "stage", "timestep")


def test_nesting_must_name_declared_dims():
    with pytest.raises(ValidationError, match="undeclared"):
        Schema(dimensions={"timestep": Dimension(dtype=nw.Datetime(), within={"nope"})})


def test_nesting_must_be_acyclic():
    with pytest.raises(ValidationError, match="cyclic"):
        Schema(
            dimensions={
                "a": Dimension(dtype=nw.Int64(), within={"b"}),
                "b": Dimension(dtype=nw.Int64(), within={"a"}),
            }
        )


def test_a_dim_cannot_be_within_itself():
    with pytest.raises(ValidationError, match="within` itself"):
        Schema(dimensions={"a": Dimension(dtype=nw.Int64(), within={"a"})})


def test_an_attribute_cannot_vary_over_an_undeclared_dim():
    with pytest.raises(ValidationError, match="undeclared"):
        Schema(
            dimensions={"scenario": Dimension(dtype=nw.String())},
            attributes={"p": AttributeSpec(dtype=nw.Float64(), dims={"nope"})},
        )


# -- defaults through the manifest (https://energy-models.github.io/datarecord/design/schema/#attributespec, https://energy-models.github.io/datarecord/design/schema/#one-schema-per-record) ------------------------------


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), 0.0, None, "AC"])
def test_a_default_survives_the_manifest_round_trip(value):
    """An `inf` default must read back as `inf`, not as "no default".

    JSON has no literal for a non-finite float, and pydantic's own serialiser
    emits `null` for one - which would silently turn an unbounded capacity into
    an absent bound. PyPSA declares `inf` defaults (`p_nom_max`), so this is the
    ordinary case rather than an edge one, and it is only the encoding in
    `AttributeSpec` that keeps it.

    Notes
    -----
    - [AttributeSpec](https://energy-models.github.io/datarecord/design/schema/#attributespec)
    """
    schema = Schema(
        dimensions={"scenario": Dimension(dtype=nw.String())},
        attributes={"p_nom_max": AttributeSpec(dtype=nw.Float64(), default=value)},
    )
    back = Schema.model_validate_json(schema.model_dump_json())
    assert repr(back.attributes["p_nom_max"].default) == repr(value)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), -1.0, None, "AC"])
def test_a_grant_default_survives_the_manifest_round_trip(value):
    """A grant's default takes the same `__inf__` encoding the spec's does."""
    schema = _schema()
    schema.types["Generator"].attributes["p_nom"] = TypeAttribute(default=value)
    back = Schema.model_validate_json(schema.model_dump_json())
    assert back == schema
    assert repr(back.types["Generator"].attributes["p_nom"].default) == repr(value)
    assert repr(back.attributes_for("Generator")["p_nom"].default) == repr(value)


# -- versioning (https://energy-models.github.io/datarecord/design/schema/#versioning) -------------------------------------------------------


def test_adding_an_attribute_is_compatible():
    old = _schema()
    new = _schema()
    new.attributes["p_min_pu"] = AttributeSpec(dtype=nw.Float64(), dims={"scenario"})
    assert new.compatible_with(old) == []


def test_widening_dims_is_compatible():
    """Rows that set fewer dims still decode: an unset dim is NULL, and NULL means all."""
    old = _schema()
    new = _schema()
    new.attributes["marginal_cost"] = AttributeSpec(
        dtype=nw.Float64(), dims={"entity", "scenario", "timestep"}, breakpoints=True
    )
    assert new.compatible_with(old) == []


def test_widening_partial_is_compatible():
    """Ownership becomes finer; an old row is owned at the coarser granularity."""
    old = _schema()
    new = _schema(partial=frozenset({"scenario", "timestep"}))
    assert new.compatible_with(old) == []


def test_narrowing_dims_is_incompatible():
    old = _schema()
    new = _schema()
    # `entity` kept, so `timestep` is the one axis under test.
    new.attributes["p_max_pu"] = AttributeSpec(
        dtype=nw.Float64(), dims={"entity", "scenario"}
    )
    (reason,) = new.compatible_with(old)
    assert "no longer varies over ['timestep']" in reason


def test_changing_a_dtype_is_incompatible():
    old = _schema()
    new = _schema()
    new.attributes["p_nom"] = AttributeSpec(dtype=nw.Int64(), dims={"entity"})
    (reason,) = new.compatible_with(old)
    assert "Float64 -> Int64" in reason


def test_removing_from_partial_is_incompatible():
    """A layer that patched one value is now a partial override of a whole axis."""
    old = _schema(partial=frozenset({"scenario", "timestep"}))
    new = _schema()
    reasons = new.compatible_with(old)
    assert any("no longer `partial`" in r for r in reasons)


def test_granting_a_type_a_further_attribute_is_compatible():
    """The type's old rows decode as before; new rows gain a reading."""
    old = _schema()
    new = _schema()
    new.types["Link"].attributes["p_nom"] = TypeAttribute()
    assert new.compatible_with(old) == []


def test_removing_a_grant_is_incompatible():
    """The type's rows are still in the file, now with no valid reading."""
    old = _schema()
    new = _schema()
    del new.types["Generator"].attributes["p_max_pu"]
    (reason,) = new.compatible_with(old)
    assert "'Generator' no longer carries ['p_max_pu']" in reason


def test_changing_nesting_is_incompatible():
    """Un-nesting `timestep` changes the axis key's shape, so old rows misread.

    `new` is `_schema()` with the one difference under test: `timestep` no
    longer `within` `period`, where every other dim is restated unchanged.
    """
    old = _schema()
    new = _schema(
        dimensions={
            **old.dimensions,
            "timestep": Dimension(dtype=nw.Datetime()),
        }
    )
    reasons = new.compatible_with(old)
    assert any("nesting changed" in r for r in reasons)


# -- unit and description (https://energy-models.github.io/datarecord/design/schema/#unit-and-description) ---------------------------------------------


def test_unit_and_description_are_declared_on_both():
    """An axis may carry them too, not just an attribute.

    Notes
    -----
    - [unit and description](https://energy-models.github.io/datarecord/design/schema/#unit-and-description)
    """
    s = Schema(
        dimensions={
            "vintage": Dimension(
                dtype=nw.Int64(), unit="year", description="Build year."
            ),
            "scenario": Dimension(dtype=nw.String(), description="One realisation."),
        },
        attributes={
            "p_nom": AttributeSpec(
                dtype=nw.Float64(), unit="MW", description="Nominal power."
            )
        },
    )
    assert s.dimensions["vintage"].unit == "year"
    assert s.attributes["p_nom"].unit == "MW"
    # An axis whose labels are not a quantity declares none.
    assert s.dimensions["scenario"].unit is None


def test_undeclared_is_none_not_empty():
    """`None` is "undeclared", `""` is "genuinely dimensionless".

    Notes
    -----
    - [unit and description](https://energy-models.github.io/datarecord/design/schema/#unit-and-description)
    """
    assert AttributeSpec(dtype=nw.Float64()).unit is None
    assert AttributeSpec(dtype=nw.Float64(), unit="").unit == ""


def test_changing_a_unit_is_compatible():
    """No facet decides how a row decodes, so editing one is compatible.

    Notes
    -----
    - [versioning](https://energy-models.github.io/datarecord/design/schema/#versioning)
    """
    old = _schema()
    new = _schema()
    new.types["Generator"].attributes["p_nom"] = TypeAttribute(
        default=0.0, unit="kW", description="Rated power."
    )
    assert new.compatible_with(old) == []


# -- serialisation (https://energy-models.github.io/datarecord/design/schema/#one-schema-per-record) ----------------------------------------------------


def test_round_trips_through_json():
    """`manifest.json` is how a schema is written down, so this must be lossless."""
    s = _schema()
    back = Schema.model_validate_json(s.model_dump_json())
    assert back == s


def test_column_types_cover_structural_dims_and_flags():
    s = _schema()
    # A declared dim wins over the structural default, so the type axis carries
    # the `Enum` that pins its vocabulary rather than a bare string - and one a
    # schema calls `kind` is typed the same way.
    assert s.column_type("entity_type") == nw.Enum(["Generator", "Link"])
    assert s.column_type("entity") == nw.String()
    assert s.column_type("timestep") == nw.Datetime()
    # One struct per flag column, a BOOLEAN field per declared dim (https://energy-models.github.io/datarecord/design/read-path/#owner-map), so
    # the map's column set does not widen when a dim is declared.
    for column in ("varies", "broadcast"):
        column_type = s.column_type(column)
        assert column_type == flag_type(s.broadcast_dims)
        assert column_type is not None
        duck_types = DuckTypes(duckdb.connect())
        assert "scenario BOOLEAN" in str(duck_types(column_type))
    assert s.column_type("value") is None
    assert s.value_type("p_nom") == nw.Float64()


def test_attributes_need_at_least_one_dim():
    """Attribute data varying over no axis is a table, not a record.

    Rejected at the schema rather than handled in the fold: the owner map's
    flag columns are structs with a field per dim, and DuckDB has no empty
    struct - so forbidding the case is what keeps a placeholder field out of
    every fold.

    Notes
    -----
    - [dimensions](https://energy-models.github.io/datarecord/design/schema/#dimensions)
    """
    with pytest.raises(ValidationError, match="at least one dim"):
        Schema(attributes={"p_nom": AttributeSpec(dtype=nw.Float64())})


def test_a_schema_declaring_nothing_stays_legal():
    """`Schema()` is "no manifest yet", not a claim that there are no axes.

    Notes
    -----
    - [one schema per record](https://energy-models.github.io/datarecord/design/schema/#one-schema-per-record)
    """
    assert Schema().dims == ()
    assert Schema().attributes == {}
