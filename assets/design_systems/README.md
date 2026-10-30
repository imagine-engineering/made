# Fractal Neutral in Made

Reference repository: https://github.com/simtropolis/fractal
Reference revision: `0613fab3abf72848cedbd982363c603317e092a7`.

`fractal-neutral.json` contains the two Neutral entries from
`gpui/crates/assets/assets/themes/fractal/fractal-themes.json` at that revision.
It is the checked-in source for `python3 script/fractal-theme`; `--check` verifies
that the generated Made theme is current. No network access is needed to regenerate.
The React visual foundations are in `react/packages/themes/neutral/src/neutralTheme.ts`.

## Integration

Made keeps its existing GPUI runtime and UI, editor, terminal and workspace entities.
Fractal's `gpui-fractal` package depends on upstream Git GPUI, its own global Theme
and Root, and a separate component lifecycle. Adding it directly would introduce
incompatible GPUI package identities and duplicate root/theme infrastructure.
The integration therefore adapts Made's existing controls to Fractal's semantic
contract and reuses its SVG assets. No external Rust dependency is introduced.

`crates/ui/src/styles/fractal.rs` holds the source contract's geometry and typography.
Buttons use 28/32/36px heights, 24px for the source's extra-small variant, 10px element
radii and 12px container radii. List interiors use the 6px inner radius. Spacing
continues through Made's existing 4px-based DynamicSpacing scale and density setting.
The 16px rem reference is retained: body labels are 14px with 20px line height;
supporting labels are 12px, large labels 17px, headings 12/14/17/20/24px. Explicit
UI font-size choices continue to scale rem-based controls and text proportionally.

The generated palette maps semantic body/surface/popover, foreground, border and
state roles onto Made's existing theme schema. Editor backgrounds and syntax come
from Fractal's syntax contract. Terminal and diagnostics retain their semantic
hues. ANSI foregrounds use readable syntax tonal stops; bright variants deliberately
retain those stops rather than manufacturing a second unrelated palette.
Filled controls derive inverse text from the selected Made theme background so
custom themes remain usable; this slightly differs from Fractal's dedicated
on-accent foreground. Dropdowns use neutral outlined surfaces. Elevations use
the source medium/high drop shadows; existing borders replace CSS inset shadows.
For readable editable code, faint punctuation and operators use the secondary
foreground. Empty/loading/error workflows keep their existing behaviour and use
shared text/icon/status tokens. Rust, debugger and agent-specific icons without a
matching Fractal asset retain their existing semantic geometry.

## Assets and licences

- GPUI semantic contract and adaptations: Fractal Apache-2.0 (`LICENSE-APACHE`).
- React Neutral design source: Meta Platforms, Inc. and affiliates, MIT (`LICENSE-MIT`).
- `assets/icons/fractal`: reused Lucide/Feather icons (ISC/MIT) and Fractal additions
  (Apache-2.0), with adjacent licence and provenance notices.
- `fractal-icons.json`: 85 Made semantic names mapped to 66 imported SVGs.
  `fractal-icon-sources.json` records original paths and SHA-256 checksums.
- Figtree variable upright/italic fonts: Google Fonts revision
  `a60a77e14f28abd4ef243a1b5dfc48df0cec5205`, `ofl/figtree`, from
  https://github.com/google/fonts. Original family and font names are retained.
  Copyright 2022 The Figtree Project Authors; SIL OFL 1.1 accompanies the fonts in
  `assets/fonts/figtree/OFL.txt`. Made loads these through its existing embedded
  TrueType font loader and retains platform fallback fonts.
- Lucide licence source: https://github.com/lucide-icons/lucide at
  `2bfb9bb1bae5d74f6a9f81640ddd8bccc2c71860`, `LICENSE`.

## Adding a product theme

Set the product's `appearance` table in `products/flavors.toml` to a bundled theme
file, its light/dark variant names, a font family and a bundled TrueType font asset.
Run `cargo xtask products` and `cargo xtask products --check`. The validator requires
an appearance for enabled products, validates safe bundled paths, verifies both
variant names and appearance modes, and checks the font asset. Planned flavours
remain disabled and need no appearance until enabled.

Generated constants expand the default settings template before parsing, preserving
comments in the default-settings editor. User/global/project preference precedence
is unchanged. The UI consumes semantic tokens without branching on product IDs.
