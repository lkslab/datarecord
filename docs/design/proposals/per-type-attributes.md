<!--
SPDX-FileCopyrightText: datarecord contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Proposal: per-type attributes - what a type carries, declared

Status: **Implemented** · Drafted 2026-09-22 · Implemented 2026-09-22

**This page is the argument, not the design.** What shipped is described by [the schema](../schema.md) — [types](../schema.md#types-what-a-type-carries), [traits](../schema.md#traits) — which is authoritative where the two disagree.

Supersedes the narrowing half of [dims, groups and traits](dims-groups-traits.md) and the `on` half of [trait switches](trait-switches.md); the switch itself is untouched.

## What starts it

An attribute's default, unit and description vary per component type, and the schema has one record-wide slot for each.
Measured on PyPSA's registry: 7 attributes differ per type on `default`, 14 on `unit`, 54 on `description`.
`sign` is the smallest case - `1.0` on a `Generator`, `-1.0` on a `Load` - and one record-wide default is right for at most one of them.
Today the first type to declare wins and every other type's fact is silently dropped.

The first fix tried was `Trait.defaults`: a trait carries the default an attribute takes on the types it is `on`.
It was implemented and abandoned unmerged, because it bends the trait vocabulary out of shape twice over.
Giving `Load` its own `sign` default meant a trait `on={"Load"}` bundling `sign` - and since [a trait narrows](../schema.md#traits), that bundling silently stripped `sign` from every other type, so each of them needed a trait of its own restating it.
The end state is one trait per type, each listing that type's attributes: per-type membership, written in a vocabulary designed to say something else.
The PyPSA tool already synthesizes exactly that shape today, defaults aside, because a framework registry *is* per-type membership and "a trait narrows, it does not grant" gives it nowhere honest to land.

So the narrowing direction is the real subject.
It was adopted so an attribute belonging to no type has somewhere to live, and that half survives - an axis-addressed `weighting` still belongs to the record.
What does not survive is deriving per-type presence from grant-minus-narrowing across traits: presence is a fact frameworks state directly, per type, together with the facets that vary with it.

## The construct

A schema declares what each entity type carries, per type, with the per-type facets stated on the grant:

```python
class TypeAttribute(BaseModel):
    """One type's facets for one attribute it carries."""

    default: Any | None = None
    unit: str | None = None
    description: str | None = None


class TypeSpec(BaseModel):
    """One entity type: what it carries, and what it is."""

    attributes: dict[str, TypeAttribute] = {}  # grant: key present = carried
    description: str | None = None


class Schema(BaseModel):
    ...
    types: dict[str, TypeSpec] = {}  # keyed by entity-type label
```

```json
"types": {
  "Generator": {
    "description": "A device attached to a bus that can generate power.",
    "attributes": {
      "carrier": {},
      "sign": { "default": 1.0 },
      "p_nom_max": { "default": "__inf__", "unit": "MW" }
    }
  },
  "Load": {
    "attributes": {
      "carrier": {},
      "sign": { "default": -1.0 }
    }
  }
}
```

Five rules, each carrying one decision:

**Pure grant.** In a typed schema, `attributes_for(ctype)` reads `types[ctype]` and nothing else.
An entity-addressed attribute granted by no type is a schema error - it is a forgotten grant or a typo, and "declared but unreachable" has no use.
The hybrid alternative - unlisted-by-everyone reaches everyone - was rejected because it re-imports the failure mode this proposal removes: adding one type's entry would silently strip the attribute from every other type, at a distance.

**Keys are the Enum labels, exactly.** The entity-type axis stays the sole declaration of the *vocabulary* - the [`into` group over `entity`](../schema.md#entity_type-the-axis-of-kinds) and its dim's `Enum`.
`types` requires that axis with an `Enum` dtype, and its keys must equal the labels: a missing key and a stray key are both typos worth an error.
An entry may be empty, for a type carrying nothing entity-addressed.
A `str`-typed axis keeps its labels as data and admits no `types` table - there is nothing declared to grant to - and a schema with no type axis at all is untouched: everything entity-addressed reaches every component, as before.

**Facets live where the attribute is addressed.** `dtype`, `dims` and `breakpoints` stay record-wide on `AttributeSpec`: one attribute is one file with one `value` column, so [the flatness argument](../schema.md#attributespec) is about storage and per-type storage facts stay unrepresentable.
`default`, `unit` and `description` are never stored, so they are free to be facts of the `(type, attribute)` pair - and in a typed schema they are *only* that: a spec-level facet on an entity-addressed attribute is rejected, so the manifest never holds two slots that both look authoritative.
An attribute addressed by an axis alone has no type, so its facets stay on the spec; in an untyped schema everything's do.
There is no inheritance and no override: an absent facet is undeclared, which also dissolves the "does `null` mean override-to-None or inherit" question `Trait.defaults` had to answer with presence-is-the-override semantics.

**A type is an object.** `TypeSpec.description` says what a `Generator` *is* - prose a framework registry carries and an `Enum` label cannot.

**Traits keep only the capability.** With presence declared, a trait's narrowing role is gone, and a trait that neither narrows nor gates is a comment - so `switch` becomes required and `on` is dropped.
`on`'s information - which types carry the capability - is derivable: it is the types granted the switch attribute, one source of truth instead of two kept in agreement.
`investable` and `dispatchable` stop being traits; they were type facts all along, and the `types` table says them directly.
What remains is `committable`: a bundle, a switch, and one validation - every type granted the switch must be granted the whole bundle.
The switch's semantics are untouched: it narrows which components carry a *value*, never the vocabulary, exactly as [trait switches](trait-switches.md) argues.

### The cost of stating the shared fact per type

The repetition is real and deliberate.
`carrier`'s description appears on every type that carries it, a reader cannot distinguish "same by design" from "happens to agree", and editing "the default of `p_nom`" is one edit per granting type - drift between them is legal, since they are per-type facts.
For a generated manifest that is nothing; PyPSA's registry is the shape being written down.
For a hand-written schema it is lines, paid for a model with one lookup rule and no merge semantics.
The alternative - record-wide facets with per-type overrides - states the shared fact once at the price of a two-place read plus a merge rule on every lookup, and was rejected as a second mechanism riding on the enumeration pure grant already pays for.

## What it changes for `attributes_for`

Nothing at the call sites.
`Schema.attributes_for(ctype)` keeps its signature, stays answerable from the schema alone, and still returns `dict[str, AttributeSpec]` - each spec a copy with the grant's facets in place, so a consumer asking "what is `sign` on a Load and what is its default" reads one object.
`Schema.attributes[a]` is untouched.

What backs it inverts.
Today presence is computed - grant from `addresses_entity`, narrowing resolved across all traits - and the answer for a typed schema becomes a read of `types[ctype]`.
The untyped answer (everything entity-addressed, whatever `ctype`) and the axis-addressed answer (no type carries it, however addressed) are unchanged, and `test_untyped.py` and `test_entity_type_attributes.py` pin both.

`types_declaring(attribute)` stops being a loop over `attributes_for` and becomes a read of the grants.

## Versioning

The compatibility rules keep their shape; only the wording moves from traits to grants:

- **Compatible**: adding a type, granting a type a further attribute, changing any facet on a grant or a spec - a default, unit or description is never stored, so no written row decodes differently.
- **Incompatible**: removing a grant, since the type's rows are still in the file with no valid reading - the same reason "unsubscribing a trait" was incompatible.

`compatible_with` already diffs `set(attributes_for(ctype))` per type and needs no change to enforce this.

## Implementation sketch

- `schema.py`: `TypeAttribute` and `TypeSpec` models; `Schema.types`; `Trait` loses `on` (and the never-merged `defaults`), `switch` becomes required.
  `TypeAttribute.default` reuses the [`__inf__` encoding](../schema.md#attributespec) `AttributeSpec.default` has, extracted to shared helpers so a third site cannot forget it.
- Validation, replacing the trait-narrowing rules: `types` requires the Enum axis and keys equal to its labels; a grant names a declared, entity-addressed, non-result attribute; every entity-addressed attribute is granted somewhere; spec-level facets rejected on entity-addressed attributes in a typed schema; a trait's bundle granted on every type granted its switch.
- `attributes_for` reads the grants; `types_declaring` reads them directly.
- The PyPSA tool emits `types` from the registry - presence, per-type default/unit/description, and the component description - instead of synthesizing one trait per type, and emits no traits at all: the registry declares no capabilities.
- `tests/fixtures.py::schema()` already takes attributes keyed by ctype and flattens them into the trait-per-type shape; it becomes a direct mapping, and most test schemas follow it.

Out of scope: the switch's data-reading checks (write-time rejection, `entities_with`), which stay where [trait switches](trait-switches.md#what-it-costs) left them - declared first, wired after.

## What it costs

**A second spelling of the type vocabulary.** `types`' keys restate the Enum's labels, kept equal by validation.
Deriving the Enum from the keys instead was considered and rejected: the axis is a dim like any other, and its dtype deciding write-time rejection is a property of the axis, not of the grants.

**Verbose manifests.** Every type enumerates everything it carries, facets included.
Generated manifests do not care; hand-written ones pay per type.

**A migration with no compatibility path.** A manifest carrying `traits` with `on` does not parse under the new model.
Pre-1.0, and the error names the field.

## What it opens

- **Per-type facets on results.** Deliberately not taken: `results` is exempt from the per-type vocabulary by design - a result may name a component the record does not declare, and nothing validates results per type - so granting them presence would contradict the exemption, and facet-overrides-without-presence would be a second mechanism. Revisit with a concrete case.
- **A per-entity `attributes_for` counterpart**, unchanged from [trait switches](trait-switches.md#what-it-opens) and [open questions](../open-questions.md).
- **An `Enum` switch with `switch_value`**, unchanged from [trait switches](trait-switches.md#which-dtypes).
