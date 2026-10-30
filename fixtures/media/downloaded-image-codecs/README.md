These encoded JPEG/WebP fixtures exercise the pinned ComfyUI downloaded-image
helper, which opens an image with Pillow, converts directly to RGBA, and divides
float32 channels by 255. EXIF orientation is retained without transposing pixels.
The fixtures cover baseline and progressive JPEG, chroma subsampling, grayscale,
CMYK, WebP lossless/lossy color and alpha, and orientation metadata.

`manifest.json` records the source and codec versions, encoded SHA-256, decoded
RGBA SHA-256, and tensor byte SHA-256. `.rgba` files are Pillow's decoded channel
bytes. `.f32le` files are the actual pinned source tensor's little-endian float32
bytes. They are generated independently of the native decoder.

Verify the immutable files with the recorded Python/Pillow/NumPy/PyTorch versions:

```sh
python source-oracle.py --source /path/to/ComfyUI/comfy_api_nodes/util/conversions.py
```

Generation requires the explicit `--generate` flag and must run separately from
Rust tests. Do not update expected pixels from native decoder output or replace
exact comparisons with tolerances.

The six WebP cases match the native decoder exactly. Downloaded JPEG uses the
canonical vendored decoder's explicit `PillowSource` mode, preserving the
original zune mode for all other callers. Both original codec tests are mandatory.
The initial JPEG failure logs remain diagnostic history; none of their source
golds were changed to implement the correction.

`jpeg-boundaries` adds 222 independent source cases: 37 small, odd, block-edge and
MCU-edge dimensions for 4:4:4, 4:2:2, 4:2:0, progressive 4:2:0, grayscale and
CMYK. Its source oracle records three malformed-header rejection cases too.
`legacy-zune-0.5.15.json` records independent pristine cached decoder RGB hashes,
verified identically in default SIMD and scalar builds. These legacy hashes
check preservation of upstream behavior; they are separate from Pillow source
tensor golds and must never substitute for source acceptance.

```sh
cargo test -p comfy_media --lib downloaded_images_match_pinned
cargo test -p comfy_media --lib downloaded_jpeg
cargo test -p comfy_media --lib jpeg_source_mode_preserves_pristine
```
