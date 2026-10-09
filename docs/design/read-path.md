<!--
SPDX-FileCopyrightText: datarecord contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# The DuckDB read path

## Owner map

The owner map answers, for a node, which layer owns each key.
**One map: `inputs`.** It is the only relation whose winning _key_ and winning row's _values_ live in different files — an attribute's ownership spans every `inputs/<attr>.parquet`, so the map names the owning layer per key and a read then goes to that layer's file for the value.
Every axis — a [dim](schema.md#partial-the-granularity-of-an-override)'s coordinates, the entity axis, a [group](schema.md#groups) — is a single keyed file whose winning row _is_ the whole row, so none needs a map; each folds to a [resolved relation](#one-fold-for-every-axis) read inline.

The inputs map's columns, keys first:

```text
# inputs
<partial dims>          -- the fold key: membership keys + partial value dims
  entity     -- never NULL
  <group coordinates>
  <owned_per value dims>
attribute
layer_uuid              -- the owning layer
varies      STRUCT(<dim>: BOOLEAN, ...)
broadcast   STRUCT(<dim>: BOOLEAN, ...)
breakpoints BOOLEAN
```

It maps each key to the owning `layer_uuid`, with deletions already applied.
It carries no `value`, no varying dim's value, and no `breakpoint`, so it stays small regardless of the series data or the size of a curve.

The **inputs key is schema-derived**, not spelled: it is [`partial_dims`](schema.md#partial-the-granularity-of-an-override) plus `attribute`.
`partial_dims` is the _membership keys_ — `entity` and every group coordinate, which [address a row rather than broadcasting](record.md#the-broadcast-rule) — plus the broadcast value dims a layer may patch per value (`partial`).
A coordinate an attribute's own file does not carry reads as NULL, which is what keeps the key one fixed tuple across attributes whose columns differ.

`entity_type` is **not** on `inputs`: an attribute row is addressed by `entity` alone ([entity is unique across types](format.md#entity-is-unique-across-types)), and the resolved entity axis is what says which type an entity is.
That axis is the entity mapping every type-scoped question goes through — [`flags(ctype)`](record.md#flags) joins it, as does a consumer wanting one type's frame — and it carries `entity_type` as a column, functionally determined by `entity` rather than keyed alongside it.
Keying on the type would let an entity resolve to two rows, admitting at read time the collision [name uniqueness](format.md#entity-is-unique-across-types) rejects at write time; so the fold aggregates the type over the group-by instead of grouping on it, and the staging area collapses a `remove` under one type followed by an `add` under another to the later edit ([committing](working-record.md#committing)).

The map is built by folding along the root→node path: parent map minus deletions and overrides, union the layer's own keys.
A node whose caches are [materialised](layers.md#materialised-node-caches) persists it (and the resolved axes beside it), so a read needs only the ancestry **back to the nearest materialised node** — the key scalability property.

A **standalone record** has no node to materialise, but the same map: [`write_record`](writing.md) folds it over the rows it just wrote and stores it as `owner_map/inputs.parquet` in the record directory ([format](format.md)), and a single-source fold reads it back instead of re-aggregating every `inputs/` file per connection.
Written whole and renamed into place, the record cannot change under its map, so the stored map is its resolved map forever.
The stored file carries **no `layer_uuid`**: a directory's layer id derives from its location, which the rename out of staging — or any later move — changes, so the reader stamps its own id over every row.
A record written before the map existed simply misses the probe and folds live, as a tree layer always does.
Every membership's tombstones reach this map in the fold: `fold_inputs` anti-joins the parent against the deleted rows of the entity axis, of each group, and of each partial dim — read from the same file that membership folds from — so a key whose entity, connection tuple or dim coordinate was deleted is absent from the resolved map rather than filtered at read.
This is not a cascade: each membership honours only its _own_ tombstones, so deleting a component drops the component's own row but not its connection tuples — those stay until deleted in turn.
The coordinate a membership keys on [never broadcasts](record.md#the-broadcast-rule), so a row not addressed by one carries NULL there and its NULL-safe anti-join never takes it; only a row naming a dead coordinate is dropped.

## One fold for every axis

A dim's coordinates, the entity axis, and a group are one construct: a keyed relation a layer patches per key.
They fold by one path — last-writer-wins per key, [`deleted` honoured](layers.md#deletion), static columns carried on the winning row (`weight` on `dims/scenario.parquet`, `entity_type` on the entity axis, `into` on a group) — producing one **resolved relation** with no `deleted` column and no `layer_uuid`.

The resolved relation is returned **in first-introduced member order** — root first, then file order within a layer — and a node's caches persist it _in that order_, so a reader recovers member order from the resolved file's own row number.
There is no persisted `order_key` column: member order is the file's row order.
A consumer wanting positional ports numbers a component's connections by this order, so a patch layer adding a connection appends rather than renumbering — the [positional-keying failure](record.md#connections) that order exists to prevent.
Across a materialised parent it still holds: the resolved seed is read in its own row order and a descendant's new rows number after it.

A component's wide static columns are the one axis value that lives in another file — per type, in `dims/entity_type/<ctype>.parquet`, not on the entity axis — so a node materialises the resolved per-type frames beside the axis, and [`entity_type`](#resolving-a-relation) reads them and gates against the resolved entity axis.

The fold runs live over an unmaterialised tail, cached per connection; since [layers are write-once](layers.md#a-layers-data-is-write-once), such a cache never needs invalidating.

The [flags](record.md#flags) are folded in alongside the ownership group-by, so they cost nothing beyond it.
They are computed **per key**, so per component: whether _this_ component's `p_max_pu` sets `timestep` is a different question from whether any does.

The structs have a field per **broadcast** dim rather than per declared dim: an address coordinate [never broadcasts](record.md#the-broadcast-rule), so "did a row set it" is not a question about it — `entity` and a group's coordinate are always set, by construction.

Two **structs** rather than a `varies_<dim>` column per dim, because which dims exist is [declared](schema.md#dimensions) and a flat layout would make the map's _column set_ depend on the schema.
[Versioning](schema.md#versioning) calls adding a dim compatible; that has to hold for a map already persisted at a [materialised node](layers.md#materialised-node-caches), not only for the layers.
With a struct the difference is a missing _field_, which `UNION ALL BY NAME` fills with NULL exactly as it would a missing column, and the new dim reads as unset — which it is, since no row mentions it.
The map's columns are then fixed, and only the fields move.

`breakpoints` stays outside both structs, being no dim ([wide and long rows](record.md#wide-and-long-rows)).
That also means the dim namespace lives entirely inside `varies`/`broadcast`, so a dim named `breakpoints` would collide with nothing.

`Record.flags(ctype)` unions them over the entities of one type, which is [the granularity every consumer works at](record.md#flags) — where what the union means, and why it stops at the type boundary, is argued.

## Resolving a relation

A resolved relation joins the owning layers' files to the `inputs` map, keeping only owned rows:

```sql
SELECT u.entity, u.bus, u.timestep,
       COALESCE(u.scenario, o.__owned_scenario) AS scenario,   -- one per owned_per dim
       u.attribute, u.breakpoint, u.value
FROM ( -- one arm per distinct layer the map names for this attribute
  SELECT ?::UUID AS layer_uuid, * FROM read_parquet(<layer>/inputs/<attr>.parquet)
  UNION ALL BY NAME
  ...
) u
JOIN ( -- the map keyed as its owning layer stored each key
  SELECT * EXCLUDE (scenario), scenario AS __owned_scenario,
         UNNEST(list_concat(
           CASE WHEN COALESCE(varies.scenario, false)   THEN [scenario] END,
           CASE WHEN COALESCE(broadcast.scenario, true) THEN [NULL] END
         )) AS scenario
  FROM inputs
) o
  ON o.entity     IS NOT DISTINCT FROM u.entity    -- address coordinates: NULL-safe,
 AND o.bus        IS NOT DISTINCT FROM u.bus       -- never expanded against an axis
 AND o.attribute  = u.attribute
 AND o.layer_uuid = u.layer_uuid
 AND o.scenario   IS NOT DISTINCT FROM u.scenario  -- owned-per dims: as stored, NULL included
```

The projected coordinates are the **attribute's own**, not a fixed prefix: `entity | bus` for a connection attribute, `from | to` for one over a corridor, neither for a record-level weighting ([the long schema](format.md#the-long-schema)).
The join's address columns follow from the same place, so the query shape is derived from the schema rather than spelling `entity` and `bus` as literals — which is what lets a second group resolve through the identical code.

The map already names the winning layer per key, so resolution reads only the owning layers' files.
There is no per-read `MAX`/group-by and no tombstone filter — deletions are already absent from the map.

Each owned-per dim's arm is **NULL-aware**: a stored NULL means "all values", and the map may own it for only some of them, so the row joins every entry naming its layer and takes that value in the output.
The arm cannot be the obvious condition: `u.scenario IS NULL OR u.scenario IS NOT DISTINCT FROM o.scenario` is no hash key, so DuckDB joins on the other columns and filters the dim afterwards, which costs the square of the dim's length per key.
So the map is keyed the way its owning layer stored each key, which its [flags](#owner-map) record: a map row keeps its value where the layer's row set the dim (`varies`) and holds NULL where the row left it NULL (`broadcast`), with the owned value carried beside it.
A layer row then finds its owned keys by its stored values, NULLs included, on NULL-safe equality — one hash join, scanning the layers and the map once each; the keyed map holds one row per owned key, two for a key its layer wrote both ways.
A map row NULL in the dim itself, broadcast over an empty axis, has `varies` false, so only its NULL form exists and it matches once.
A flag field missing from a map persisted before the dim was declared counts as `broadcast` here, unlike in the flags a consumer reads: the layers' rows hold NULL there.

A **group coordinate** like `bus` is joined **NULL-safely** rather than NULL-aware against the map, being [an address rather than a broadcast dim](record.md#the-broadcast-rule).
There is no membership gate at read: an attribute row is keyed by every membership its coordinates name — the entity, each group tuple, each dim coordinate — and each of those is [tombstone-pruned in the fold](#one-fold-for-every-axis), so a row whose entity, connection tuple or dim coordinate was deleted is already gone from the map.
The fold anti-joins each membership against its own `deleted` rows, read from the same file that membership folds from; it is not a cascade — deleting a component does not delete its connections, only the component's own row.
Because the coordinate a membership keys on [never broadcasts](record.md#the-broadcast-rule), a row not addressed by a membership carries NULL there and its NULL-safe anti-join never takes it — only a row naming a dead coordinate is dropped.

`breakpoint` is projected but not joined on, being no part of the key: a curve is owned whole ([wide and long rows](record.md#wide-and-long-rows)), so every breakpoint of a key comes from the winning layer.

Non-key dims pass through unchanged, because within one key-dim combination the rows come from one layer.

An attribute no layer wrote is absent from the map; its relation is empty, and the consumer applies [the schema's `default`](schema.md#attributespec).

## One record over one fold

There is one `Record`, and it is the narwhals interface over a [`Resolver`](#owner-map). A plain parquet directory is not a second implementation of it: `Record.at(uri)` folds over a single [`DirectorySource`](format.md), and over one source the fold degenerates to a scan of it — there is one layer, so every key is owned by it and the anti-join has nothing to evict.

So a directory takes the same properties as a tree node, rather than its own column of exceptions:

- **`flags`** are computed in the ownership `GROUP BY`, one scan rather than the second one a separate aggregate would cost. They need a real aggregate either way: parquet's footer statistics are per row group, not per component type, so a file mixing one type's series rows with another's constant says nothing about either.
- **Member order** is `order_key`, which over one source is `(0, file order)` — file order, arrived at by the general rule.
- **`schema.partial`** is the granularity of a patch, and one layer patches nothing, so it is inert rather than absent.

Being a node in a layer tree is not what the fold requires; being a layer _layout_ is, and that is what a record directory is. A directory copied out of any tree, with no `revisions` row, reads identically.

## Outputs

`outputs/<attr>.parquet` does not overlay.
An output relation reads the node's own layer only: if that layer has no `outputs/`, the record has no results, and an ancestor's are not inherited.
