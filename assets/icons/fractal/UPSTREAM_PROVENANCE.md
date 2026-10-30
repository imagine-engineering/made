# Fractal semantic icon provenance

The SVG files in this directory are original Fractal geometry authored for
the native semantic registry. They are distributed under this repository's
Apache-2.0 license. All use `currentColor`, a 24×24 view box, and logical LTR
geometry; directional mirroring is applied by `Icon::rtl` exactly once.

The registry also maps semantic roles to existing files in `assets/icons/`.
Those files are Lucide icons distributed under the ISC license, with the
Feather-derived subset additionally covered by MIT. The authoritative license
copy audited for this import is `react/node_modules/lucide-react/LICENSE`
(Lucide Icons and Contributors, 2026). No font or icon with unverified
redistribution rights is introduced.

Checksums are validated by the `theme::semantic_icon` test against the packaged
bytes; changing geometry requires updating the expected manifest and this
provenance record.
