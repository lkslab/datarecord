<!--
SPDX-FileCopyrightText: datarecord contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Changelog

All notable changes to datarecord are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- Per-type attribute facts are first-class: `Schema.types` declares what each
  entity type carries, with per-type `default`, `unit` and `description` on the
  grant, and `attributes_for` reads it alone. Traits no longer decide presence:
  `Trait.on` is gone, `switch` is required, and a trait is a per-component
  capability. The PyPSA tool emits the types table from the registry instead of
  one trait per type. See `docs/design/proposals/per-type-attributes.md`.
