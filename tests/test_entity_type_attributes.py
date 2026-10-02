# SPDX-FileCopyrightText: Contributors to datarecord <https://github.com/energy-models/datarecord>
#
# SPDX-License-Identifier: MIT

"""An attribute addressed by the entity-type axis alone.

A per-type `icon` is a value per type, keyed once, and so a column of
`dims/entity_type.parquet` - the same treatment any axis a functional group is
`into` gets. What the type axis may *not* do is key a value alongside `entity`,
where the type is determined by the entity and the row would be keyed twice
over.

Notes
-----
- [entity types](https://energy-models.github.io/datarecord/design/schema/#entity_type-the-axis-of-kinds)
- [where a value lives](https://energy-models.github.io/datarecord/design/format/#where-a-value-lives)
"""

import narwhals as nw
import pandas as pd
import pytest
from pydantic import ValidationError

from datarecord import Revision
from datarecord.duck import layer_dir
from datarecord.layered.resolve import write_schema
from datarecord.mutable import NewChild, WorkingRecord
from datarecord.schema import (
    AttributeSpec,
    Dimension,
    Group,
    Schema,
    TypeAttribute,
    TypeSpec,
)
from tests.fixtures import write_axis

TYPES = ["Bus", "Generator"]


@pytest.fixture
def typed_schema():
    """Two component types, with an `icon` addressed by the type axis alone."""
    return Schema(
        dimensions={
            "entity": Dimension(dtype=nw.String()),
            "entity_type": Dimension(dtype=nw.Enum(TYPES)),
        },
        groups={"entity_type": Group(over=["entity"], into="entity_type")},
        attributes={
            "p_nom": AttributeSpec(dtype=nw.Float64(), dims={"entity"}),
            "icon": AttributeSpec(
                dtype=nw.String(), dims={"entity_type"}, default="dot"
            ),
        },
        # `icon` appears in no grant: addressed by the type axis alone, it
        # belongs to no component, and a grant naming it is rejected.
        types={
            "Bus": TypeSpec(),
            "Generator": TypeSpec(attributes={"p_nom": TypeAttribute()}),
        },
        # `entity_type` is deliberately *not* partial: `partial` widens the owner
        # map and the resolution it keys, so it is kept to what a layer really
        # patches value by value. A layer touching one type's icon therefore owns
        # the whole type axis and restates it.
        partial=frozenset(),
    )


@pytest.fixture
def root(con, base_uri, typed_schema):
    """A record with both types, and an icon for each."""
    write_schema(typed_schema, base_uri)
    revision = Revision.create(con)
    staged = WorkingRecord(revision.record, con)
    staged.add("Bus", pd.DataFrame([{"entity": "b1"}]))
    staged.add("Generator", pd.DataFrame([{"entity": "g1", "p_nom": 1.0}]))
    child = staged.commit(NewChild(revision))
    write_axis(
        layer_dir(child.id),
        "entity_type",
        [
            {"entity_type": "Bus", "icon": "node"},
            {"entity_type": "Generator", "icon": "turbine"},
        ],
    )
    return child


def _icons(record) -> dict[str, str | None]:
    frame = record.dims["entity_type"].collect().to_native()
    return {
        str(k): (None if v is None else str(v))
        for k, v in zip(frame["entity_type"], frame["icon"])
    }


def test_the_type_axis_carries_it_as_a_column(typed_schema):
    """One addressing coordinate, so a column on that thing's own table."""
    s = typed_schema
    assert s.attributes_on("entity_type") == ("icon",)
    assert not s.attributes["icon"].varying
    assert "icon" not in s.long_columns, "not a long row"
    assert s.column_type("icon") == nw.String(), (
        "an axis file's attribute column has a declared type, so writes cast it"
    )


def test_it_belongs_to_no_component(typed_schema):
    """A value per type is not a value per entity, so no type carries it."""
    s = typed_schema
    assert not s.addresses_entity("icon")
    for ctype in TYPES:
        assert "icon" not in s.attributes_for(ctype)


def test_it_reads_back_from_the_type_axis(root):
    assert _icons(root.record) == {"Bus": "node", "Generator": "turbine"}
    assert "icon" not in root.record.attributes, "an axis column is no long frame"


def test_set_states_one_types_value(root, con):
    """A mapping keyed by label, with every untouched label left alone."""
    staged = WorkingRecord(root.record, con)
    staged.set("icon", {"Generator": "windmill"})
    assert _icons(staged) == {"Bus": "node", "Generator": "windmill"}


def test_a_scalar_reaches_every_type(root, con):
    staged = WorkingRecord(root.record, con)
    staged.set("icon", "square")
    assert _icons(staged) == {"Bus": "square", "Generator": "square"}


def test_a_child_layer_restates_the_type_axis_it_owns_whole(root, con):
    """`entity_type` is not `partial`, so touching it carries every label.

    No exception for being the type axis: a dim outside `partial` is one a layer
    owns entirely once it touches it, and its axis file then says what the axis
    is at that layer rather than being readable only against its parent.
    """
    staged = WorkingRecord(root.record, con)
    staged.set("icon", {"Generator": "windmill"})

    axis = staged.resolver.sources[-1].axis("entity_type")
    assert axis is not None
    patch = axis.df()
    assert sorted(str(t) for t in patch["entity_type"]) == ["Bus", "Generator"], (
        "the whole axis, not just the edited label"
    )

    child = staged.commit(NewChild(root))
    assert _icons(child.record) == {"Bus": "node", "Generator": "windmill"}, (
        "the untouched type keeps its icon because this layer carried it"
    )


def test_an_enum_label_the_dtype_does_not_declare_is_refused(root, con):
    """The staging column is the axis's `Enum`, so the insert itself rejects it.

    Wrapped rather than surfaced: DuckDB says "could not convert to UINT8",
    naming the enum's storage type instead of the dim or its vocabulary.
    """
    staged = WorkingRecord(root.record, con)
    with pytest.raises(ValueError, match="pins the vocabulary"):
        staged.set("icon", {"Nope": "x"})
    assert "entity_type" not in staged.resolver.sources[-1].axes(), "nothing staged"


def test_entity_is_refused_for_a_type_addressed_attribute(root, con):
    """`entity=` names components, and an icon belongs to none."""
    staged = WorkingRecord(root.record, con)
    with pytest.raises(ValueError, match="belongs to no component"):
        staged.set("icon", "x", entity=["g1"])


def test_naming_the_type_alongside_entity_is_refused():
    """The entity determines the type, so the row would be keyed twice over."""
    with pytest.raises(ValidationError, match="keys a row twice over"):
        Schema(
            dimensions={
                "entity": Dimension(dtype=nw.String()),
                "entity_type": Dimension(dtype=nw.Enum(TYPES)),
            },
            groups={"entity_type": Group(over=["entity"], into="entity_type")},
            attributes={
                "p_nom": AttributeSpec(
                    dtype=nw.Float64(), dims={"entity", "entity_type"}
                )
            },
            partial=frozenset(),
        )
