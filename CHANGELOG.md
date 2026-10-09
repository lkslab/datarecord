<!--
SPDX-FileCopyrightText: datarecord contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Changelog

All notable changes to datarecord are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `write_record` with a directory target stores the record's inputs owner map
  as `owner_map/inputs.parquet` beside the rows, and a single-source fold reads
  it back (`LayerSource.stored_map`), so opening a standalone record no longer
  re-aggregates every `inputs/` file per connection. The stored file carries no
  `layer_uuid` — the reader stamps its own, so a moved record still resolves —
  and a record without the file folds live as before.

### Changed

- Per-type attribute facts are first-class: `Schema.types` declares what each
  entity type carries, with per-type `default`, `unit` and `description` on the
  grant, and `attributes_for` reads it alone. Traits no longer decide presence:
  `Trait.on` is gone, `switch` is required, and a trait is a per-component
  capability. The PyPSA tool emits the types table from the registry instead of
  one trait per type. See `docs/design/proposals/per-type-attributes.md`.

### Fixed

- Reading an attribute over a `partial` dim no longer slows with the square
  of that dim's length.
