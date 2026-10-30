# Third-party notices and asset provenance

## Base icon set

SVG files directly under `assets/icons/` are derived from the Lucide icon set,
including its Feather lineage. They retain the Lucide ISC license and the
applicable Feather MIT license. The audited upstream license is
`react/node_modules/lucide-react/LICENSE` (Lucide Icons and Contributors,
2026). Redistribution is permitted by those licenses; this notice does not
relicense the upstream geometry.

## Fractal semantic additions

SVG files under `assets/icons/fractal/` are original Fractal geometry created
for this repository and distributed under Apache-2.0. Their detailed source,
view-box, color, RTL, and checksum policy is recorded in
`assets/icons/fractal/PROVENANCE.md`. Generated semantic registry metadata and
the packaged `assets/themes/fractal/fractal-themes.json` are derived from
Apache-2.0 Fractal source and distributed under Apache-2.0.

## Fonts and other media

No font files are bundled by `gpui-fractal-assets`. Applications are
responsible for licensing and registering any custom font bytes they add. No
bitmap, audio, video, remote, or executable asset is included in the parity
payload.

The release-policy gate classifies every packaged asset, rejects executable SVG
content, requires bounded view boxes and semantic `currentColor` for Fractal
additions, and fails when an unreviewed asset path appears.
