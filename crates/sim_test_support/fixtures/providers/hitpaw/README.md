# HitPaw pinned-source request and transport evidence

The oracle executes the two pinned HitPaw classes, actual dimension and duration
validation helpers, model mappings, image scale selection and task-error paths.
Its 56 successful cases cover both image models and scale choices, automatic
scale/downscale branches, all seven video models and their declared resolution
choices, square dimensions, portrait aspect arithmetic and source duration
boundaries. Five preflight failures and four independently executed creation
errors preserve the source's field and control-flow contracts.

All three catalog rows are executable POSTs: 0049 photo-enhancer, 0050 task-status
and 0051 video-enhancer. There are no aliases or synthetic route claims.

```sh
python3 crates/sim_test_support/fixtures/providers/hitpaw/generate_source_contract.py --source-root projects/sim/ComfyUI --check
```

The image adapter reproduces the source's 32,000,000 output-pixel budget and
4/2/1 scale selection. It delegates any required native downscale to the
canonical owner using Lanczos and the selected pixel limit; it does not replace
pixel processing with a metadata-only tensor. Upload preserves the source
`mime_type=None` and `total_pixels=None`: the converter encodes PNG with a UUID
`.png` filename, while storage creation omits `content_type` and signed PUT
suppresses automatic Content-Type. The shared source transport DTOs now preserve optional MIME, but the current
production WIT upload request still requires a MIME string. The canonical
optional-MIME host/WIT bridge remains an explicit acceptance gap. The canonical image uploader owns first-image
behavior, encoding and storage. The oracle's native probes
are explicitly structural; their dimensions describe request selection rather
than decoded or published native values.

Video target dimensions retain width/height order, source binary64 arithmetic
and truncation. Square 4K and 8K use 2048 and 4096; nonsquare targets use shorter
sides of 2160 and 4320. Smaller target resolutions fail before upload. Duration
uses the source 0.0001-second epsilon around 0.5 and 3600 seconds; unavailable
source duration skips that check while the host owner must surface inspection
failures. Video upload requires canonical MP4/H.264 preparation.

Image and video poll with POST `{"job_id":...}` and ten-second supervisor
intervals. Authoritative cost is the initial `consume_coins/1000`, independent
of UI estimates. Task creation code/message failures and missing coins remain
failures; video preserves the source's price-before-code ordering. Shared queued
statuses do not consume the 480 active-attempt limit, and a finite supervisor
total bound prevents indefinite worker retention. Retries delegate parsing and
delay to the caller, require fresh handles and preserve exact bodies. Credentials
never accompany absolute downloads.

`downloaded-image.png` is an authentic 69-byte RGB 1x1 PNG; exact bytes and
receipts reach the canonical materializer callback. Source output conversion
requires RGBA. The injected decoder rejects allocation and proves zero batch
publication. No placeholder IMAGE/VIDEO allocation or fabricated MP4 is used.
Clean re-invocation coverage does not claim a real worker process restart.

Full Task 443 remains partial pending canonical native dimension inspection,
Lanczos pixel validation, first-image MIME/encoding/storage preparation,
MP4/H.264 conversion/storage, native RGBA/VIDEO allocation, production host/WIT,
signed compiled guest packaging, actual restart and visual output acceptance.
No live paid calls, credentials, private signing keys, shared owner changes,
push or PR are included in this bounded batch.

A separately prepared status-normalization follow-up runs the unchanged original
`poll_op`/`poll_op_raw` bodies with genuine HitPaw API models and both original
node status and price extractors. Its 26 observations comprise 16 typed completed
outcomes, 8 original failed-state exceptions and 2 injected HTTP interruptions.
U+001C–U+001F are stripped; 481 queued replies do not exhaust the 480 active
attempts. U+200B/BOM remain active, and control-only strings become empty. HTTP,
clock, delay, UI and interruption boundaries are captured; original class/IO
execution, creation cost calculation and actual clocks are not certified.

The proposed native change only adds Python's four control whitespace characters
to the existing status trim; raw JSON, prices, errors, cancellation and budgets
are retained. Three native regression drafts exercise both nodes with the real
PNG and existing provenance-bound LTXV MP4 download bytes. Injected upload owners
return captured URLs; they do not prove native video preparation or decoding.
These new native tests and required lint/full suites remain UNRUN. Source failures
still use Python JSON spacing and the outer polling-abort wrapper; existing Rust
failure formatting is unchanged and exact exception-text parity is not claimed.

2026-10-03 current-branch follow-up: the proposed correction and its three new
native regressions pass focused and complete provider tests, normal nested
Clippy, both formatting checks and immutable source-contract checks. Earlier
UNRUN labels describe the retained preparation phase. See audit/2026-10-03-provider-prompts-status-and-error-retention.md.
The broader media/guest/clock/cost/restart/visual acceptance remains open.

2026-10-03 native first-image foundation: original IMAGE batch frame0 is encoded
through sim_tensor/sim_media without resizing. Compact transport preserves
the full tensor ABI identity and attaches a separate PNG content digest. Exact
signed HitPaw/COMFY-NODE-0226/HitPawGeneralImageEnhance/image binding admits that
single frame; unrelated namespaces/nodes/features/ports retain normal cardinality.
The historical Source image fixture remains unchanged. A new Linux x86_64
NumPy1.26.4/Torch2.8.0+cpu/Pillow12.3.0 profile captures real Source pixels and
invalid-cast boundaries with explicit wheel/binary/CPU provenance. Out-of-range
NumPy casts are platform dependent, so other profiles remain unverified.
See audit/2026-10-03-hitpaw-native-image-input.md for current validation and the
retained earlier Source/native failures. Genuine compiled Guest execution,
resizing/video/storage, optional HTTP MIME/cost ordering, output allocation,
actual clock/UI and restart criteria remain open; Task443 remains partial.


HitPaw image-only WIT admission probe: retained preparation record

The separately named unsigned input-admission component reads the image node's retained tensor and one prepared first-frame PNG through the real provider-v2 WIT imports. It declares only COMFY-NODE-0226, the image node's four Source inputs and IMAGE output schema. It declares no uploads, no cost requests and no result implementation. It starts the retained photo-enhancer request head and checks cancellation. The scripted actuator returns the canonical Cancelled error; if it does not, the probe fails InvalidContract before any body or output. This exercises input admission and cancellation, not a successful HitPaw generation.

The build requires exact canonical native-contract and SDK binding-set digests from the dedicated registry test receipt. No placeholder certificate is built. The locked source crate reuses the existing Anthropic component toolchain, WIT bindings and ComponentEncoder; the recipe records actual artifact/source/lock hashes only after a future authorized build. No artifact currently exists.

Three integration test drafts cover canonical binding receipt production, genuine native batch1/batch2 input through the compiled guest with cancellation and noncancelled typed failure, and missing/corrupt media rejection before a stream starts. Expected first-frame PNG pixels and digests use the existing canonical PNG owner and unchanged Source first-frame semantics. All native/compiler/WASM outcomes are unrun.

Current WIT UploadRequest.media_type is a required string, while original HitPaw upload passes None; it cannot certify original storage or signed PUT MIME omission. InvocationContext contains only invocation/generation and carries no caller cost ceiling. The probe therefore adds neither a price constant nor a paid upload grant. Native IMAGE result receipt/allocation is still absent from the encoded-value result bridge. Original resize/auto_downscale, video, exact full Source phase timing, guest completion, private-worker restart and actual-app visual acceptance remain unproved.

Static review revision2 preserves the original draft and strengthens exact typed rejection and request-context/header/cancellation-handle assertions. Missing media and correct-digest wrong MIME/signature cases must fail InvalidUpload without any request; corrupt source/content digests must fail the canonical input host's media-content-digest error. No catch-all error counts as success. WIT input cancellation and timeout variants are preserved instead of being flattened into upload/authority errors. These changes are only prepared, not compiled or executed. The runtime tests do not yet exercise cancellation or timeout between individual WIT input calls, and checking the PNG signature is not a full PNG parser. Canonical input preparation plus real digest checks remain required.

2026-10-04 current compiled-input validation supersedes the preparation-only
labels above. The unsigned 84,836-byte component has SHA-256
`3bea4a945a4d48f330401b7644e074ff09453add4734d8ea99b9a1e42af56556`.
Its canonical binding receipt, locked offline Wasm build, validating native
ComponentEncoder build, and real host invocation passed. Four batch/cancellation
cases and five missing/corrupt-media cases passed; the current full host
conformance suite passed all 12 tests. Normal host and nested guest Clippy,
workspace/guest formatting, and the diff check passed. The original cache
permission failure and redundant test clone failure remain recorded.
See audit/2026-10-04-hitpaw-compiled-input-probe.md. This fixture remains unsigned
and image-only; it cannot activate the required complete two-node namespace.
Guest error mapping preserves typed variants received from the input host;
secret-query host propagation and cancellation between individual input calls
still require separate validation. Upload/MIME/cost/body/output success, video,
actual timing, worker restart and application visual acceptance remain open.
