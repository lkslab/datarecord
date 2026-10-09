<!--
SPDX-FileCopyrightText: datarecord contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Writing a whole record

```python
def write_record(
    revision_id: UUID, source: LayerData, con: DuckDBPyConnection
) -> None: ...
```

Writes `source` as a new layer.
An existing layer directory is an error rather than an overwrite or a merge, so a whole-record write can never half-replace what a record holds.

`outputs/` is written only for a source whose [`outputs` mapping](record.md) is non-empty, so a record with no results produces a layer with no `outputs/` rather than an empty directory.

Keys are looked up one at a time and each file written before the next is built, so a lazily-building source does one read per file written rather than one per key up front.
Frames are staged into a sibling directory and renamed on success, so a frame the fold could not resolve leaves no layer rather than half of one.

A directory target additionally gets its [owner map](read-path.md#owner-map) folded over the staged rows and stored as `owner_map/inputs.parquet` ([format](format.md)): a standalone record is written whole, so the map cannot go stale, and storing it spares every later open the aggregation over `inputs/` that a tree node pays only until it is [materialised](layers.md#materialised-node-caches).

Every column the schema declares a type for is cast to it on the way out, so a record's files carry the schema's types and a reader can trust them.
Without that a source may hand over an all-NULL column its dataframe library typed as float, and every reader would re-cast defensively instead.

Validation is structural: a long frame carries [its attribute's own coordinates](format.md#the-long-schema), and an entity frame carries every dim it is keyed by.
Which component types are valid belongs to [the schema's vocabulary](schema.md), which the record layer does not interpret.

An **input attribute the schema does not declare is rejected**, unlike a component type: its [`dims`](schema.md#attributespec) are what say which columns its file carries, so an undeclared one has no shape to write it in and would leave a file no reader could derive the columns of.
A **result is exempt** — [`Tool.results`](tools.md) derives which attributes count as results from a framework's own registry, so an unknown name is an error for an input and simply unknowable for a result.

A frame carrying a column its attribute is **not** addressed by is rejected too, rather than narrowed on the way out.
The read path projects an attribute's own coordinates, so such a column would be written and never read — and a source emitting one means something different by the attribute than the schema does, which is worth reporting rather than absorbing.
A result is exempt from both checks: its shape is a framework's business, and a name it shares with an input says nothing about which coordinates the result varies over.

The input is a [`LayerData`](record.md#layerdata): "the rows of one thing, enumerated and read" — the same interface a [`LayerSource`](read-path.md#owner-map) answers for its own layer and a `Resolver` answers for a whole fold, so `write_record` cannot tell which it was handed and does not need to.
A staged layer's own rows and a resolved record are both a `LayerData`, which is what lets `commit` write either without a third shape adapting one to the other.

A framework object exposing narwhals frames — [`Record`](record.md) rather than `LayerData` — is not itself one: `write_record` wraps it in a thin adapter that reads its `Frames` mappings as the enumerate-and-read pairs `LayerData` declares, so a tool stays narwhals-facing and the layered write path stays raw-relation throughout.
