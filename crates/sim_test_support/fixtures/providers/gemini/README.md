# Gemini pinned-source request and transport evidence

The oracle executes all eight pinned node classes and real request, media
planning, response filtering and token-price helpers. It includes ten success
requests, five preflight failures, fifteen pricing cases and five response
filter/error cases. The generated source fingerprints include the API models,
shared client, validation, upload, conversion and download helpers.

The four catalog rows preserve their identities. 0190/0191 are synthetic prefix
tombstones; 0192/0193 are executable dynamic POST templates. The provider alias
is `vertexai`. `GeminiInputFiles` is local and requires no authentication or
paid request. Legacy nodes are retained for saved-workflow compatibility.

```sh
python3 crates/sim_test_support/fixtures/providers/gemini/generate_source_contract.py --source-root projects/sim/ComfyUI --check
```

The native source probes are explicitly structural boundaries. Their marker
strings describe PNG/WebP/MP3/MP4 encoding calls; they are never claimed as
encoded media, decoded images/videos, native tensor allocations or submitted
through the transport. Actual executable media callbacks reject invalid base64
and preserve declared MIME, codec and upload order. Legacy image parts upload
ten PNG URLs then inline PNG overflow; legacy audio/video remain inline
MP3/libmp3lame and MP4/H.264. V2/Omni prioritize videos, audio, then images in
one ten-URL budget; image overflow is WebP, capped at 18 MiB of base64.

`downloaded-image.png` is the real RGB 1x1 PNG used by bounded download tests;
its exact bytes and receipt reach the materializer. Source conversion uses
RGBA, so canonical decoding must retain four channels. Its Base64 twin exercises
inline transfer preservation. Neither fixture is labeled WebP/video/audio.
`input.txt` contains real UTF-8 text; its source-encoded part is prepended to
existing file parts. STRING results use canonical SDK scalar ABI bytes and
atomic publication. Native media/custom-file model output is not fabricated.

Seed is forwarded only by NodeV2 and only when positive. Other nodes preserve
source seed controls without adding ignored seeds to requests. Source request
defaults include PNG image output options and image-storage upload requests.
Text combines every candidate text part, including thought text. Generated and
thought images have separate output ports; missing thought images request the
canonical owner's source zero image of shape [1, 1024, 1024, 4]. Video selects
only the first viable source part. Blocked candidates and absent media retain
source model explanations. Price-extractor failure is nonfatal as in the
shared source client and remains visible in `price_warning`.

Remaining full Task 441 acceptance gates are canonical native shape/frame and
file-order inspection, image/audio/video encoding and storage upload, input
asset permissions and custom GEMINI_INPUT_FILES allocation, native image/video
and zero-thought tensor allocation/publication, the production host/WIT bridge,
signed compiled guest packaging, real worker restart and native output visual
checks. This batch covers source/request/transport behavior and successful
canonical STRING publication; it does not mark Task 441 complete or perform
paid/live calls, use credentials/private signing keys, push or open a PR.

Task441 prompt-strip follow-up passes current-branch native validation (audit/2026-10-03-provider-prompt-status-whitespace.md).
Six actual source execute branches call unchanged validate_string with Python
strip and min_length1. The bounded predicate admits original nonempty prompt
bytes unchanged, rejects U+001C–U+001F-only input, and leaves U+200B/BOM nonempty.
Legacy GeminiNode retains its original unvalidated prompt admission; response
error message strip, media preparation, cost/auth/cancellation and all other
branches are unchanged semantically. Three native regressions pass in the current twenty-case suite and cover six-branch
pre-request rejection, byte-exact parts and legacy/checked-cost boundaries.
A separately labeled ignored pure-AST probe executes only the original helper
with six callsite keyword bindings:84 observations (42 admitted,42 exceptions).
That helper probe omits node bodies/API constructors. A separate84-case
original-execute capture covers seven unchanged method bodies and genuine
API/Pydantic constructors:36 validation errors and48 captured requests,
including12 legacy admissions. IO schema/media/output wrapping, live HTTP/cost,
guest, real clock/permission, restart and visual acceptance remain unproved. Original Source JSON/media/generator remain untouched.
