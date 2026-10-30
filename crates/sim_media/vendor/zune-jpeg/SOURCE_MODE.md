This software is based in part on the work of the Independent JPEG Group.
The accompanying README.ijg contains the complete copyright, no-warranty notice
and distribution terms. Existing zune sources retain their original MIT,
Apache-2.0 or Zlib licensing. The new source pixel equations also carry IJG terms.

This is the cached zune-jpeg 0.5.15 source, with its original licenses and Git
provenance retained. UPSTREAM_FILES.json records its original file hashes.
The marker parser, Huffman/entropy decoder, quantization and both MCU/progressive
owners are unchanged. There is no native library dependency or FFI.

Local changes:

- Cargo.toml isolates a nested workspace and declares the additional IJG license.
  Its core dependency is pinned to the existing workspace-locked 0.5.1.
- lib.rs exports JpegPixelMode and includes source_pixels.rs. It also removes the
  unused crate-level macro_use attribute rejected by strict CI compiler flags.
  The upstream file hashes remain the original provenance record.
- decoder.rs adds an explicit per-instance pixel constructor and guards IDCT,
  color-conversion and upsampling dispatch. Existing constructors select Native.
- components.rs retains optional source-only logical component dimensions.
- worker.rs extends actual image edges before source-only chroma interpolation.
- Ten inherited trailing-whitespace lines are normalized in decoder.rs,
  color_convert/avx.rs and upsampler/portable_simd.rs; token sequences are unchanged.
- source_pixels.rs implements nonmutating 13-bit integer IDCT, masked range
  limiting, 16-bit color equations and phase-specific chroma interpolation.

SOURCE_ALGORITHMS.json records the exact libjpeg-turbo 3.1.4.1 primary sources
and SHA-256 values. Equations are adapted from that version's jidctint.c,
jdmaster.c, jdcolor.c and jdsample.c. Widths of two or fewer chroma samples use
libjpeg's box filter. The default Native mode retains upstream dispatch and
pixel arithmetic. Source mode uses the same scalar equations on every platform.

The downloaded-image façade selects PillowSource explicitly. Other image
callers retain Native behavior. Immutable actual ComfyUI/Pillow tensor fixtures
and independent pristine zune fingerprints live under
fixtures/media/downloaded-image-codecs in the repository. Source parity tests
compare all F32 bytes exactly; legacy tests compare pristine decoder hashes.
This corpus covers baseline/progressive, 4:4:4/4:2:2/4:2:0, grayscale, CMYK,
small/odd/block/MCU boundary sizes and malformed header failures. It does not
establish parity for every optional JPEG encoding or recovery of corrupt data.
