# Copper artwork

`source/copper-mascot-tile-original.png` is the unchanged user-supplied 1254×1254 PNG (SHA-256 `e314675e6364000e9ef11961a1d7909c4618273e7617d3afcb1051b96abf7ea5`). This copper rounded tile and white spiky mascot supersede the earlier gear and letter designs. Do not recreate, trace, recolor, add lettering, or remove the dark eyes and their white reflections.

Regenerate with Python 3 and Pillow **12.3.0** (build-time only):

```sh
python3 script/generate-copper-icons.py
python3 script/generate-copper-icons.py --check
python3 script/generate-copper-icons.py --check \
  --before artifacts/copper-mascot-branding-20261029/before/app-icon@2x.png \
  --contact-sheet artifacts/copper-mascot-branding-20261029/contact-sheet.png
```

Application exports (`app-icon.*`, `icons/`) retain the complete supplied tile. Only corner-connected dark background outside the tile is removed; enclosed mascot, eyes and reflections remain opaque. The square crop `(24,8,1230,1214)` puts the tile at about 92% of canvas width, with no stretching or second background. No artificial rounded-corner mask is added to this already rounded artwork.

Standalone exports (`logo.png`, `logo.svg`, `logos/`) retain the connected white mascot and fill its enclosed eye regions with the original source RGB, preserving both eyes and reflections. The square crop `(148,126,1106,1084)` puts the mascot at about 93% width. This is distinct from the primary tile icon. White artwork naturally has low contrast on white surroundings; no unrequested outline or recoloring is added.

Both masks operate at original resolution. Retained RGB is unchanged before scaling. Every requested size receives one alpha-aware Lanczos downsample, appropriate for the supplied smooth curves. No redrawing, sharpening, substituted eyes or additional blur is applied. At 16/24px, fine spikes and eye reflections lose subpixel detail. All larger representations and the unchanged original retain the detail. `generated-icons.json` records crop, mask, tool version and exact hashes. SVG files are self-contained raster wrappers for established paths, not vector masters. ICO frames and ICNS slots use independently derived size-specific exports.

`products/flavors.toml` owns this family. Onboarding, Welcome and About use embedded `app-icon.png` through `product_flavor::UI_LOGO_ASSET`. Linux windows use the 256px representation, hicolor packages install available sizes, Windows consumes the multi-resolution ICO, and macOS consumes ICNS before signing. Document.icns reuses the selected product ICNS. No separate Copper DMG background exists; Finder displays the packaged application icon. Standalone mascot exports have no current GPUI logo consumer. Archived Copper copies and historical gear evidence remain unchanged. The generator writes no other repository or product family.

The current specification is `.agents/specs/copper-mascot-branding/`; the prior `copper-gear-branding` pack documents the superseded implementation. The former active gear source is preserved in `artifacts/copper-mascot-branding-20261029/before/` and `before.tar.gz`, outside the active asset family.
