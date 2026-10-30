# Ideogram source/request and transport slice (Task 445)

The fixture proves two reviewed node claims and three executable POST routes,
with no aliases or synthetic routes. `generate_source_contract.py` executes the
pinned source node methods, validation function, and shared client multipart
branch independently of Rust. It records 170 successes, six preflight failures,
and six response failures, including all listed V3/V4 resolutions, every V3
aspect ratio, legacy BALANCED speed, optional-field omission and file ordering.
The source API classes execute using actual Pydantic; their request defaults,
enums and assignment behavior are not a Rust-derived model stand-in. A separate
response schema probe retains datetime/coercion acceptance evidence.
Native tensor operations are explicitly structural owner probes. Their marker
bytes are never image fixtures, decoded tensors or published native values.

V3 always sends multipart, including generation without files. Field order
follows the source API model; file order is image, mask, optional character image,
optional literal `character_mask_binary`. Shared client source uses
`application/octet-stream` for the PNG BytesIO parts. Generation with a character
reference adds style_type=AUTO. Edits ignore resolution/aspect. Character
preparation precedes incomplete edit-pair rejection. V4 sends JSON, validates the
trimmed prompt but forwards the original, uses the resolution prefix, and omits
seed exactly as source does. No source price extractor or polling path exists;
caller-approved maximum cost remains a host capability, and the UI price badge
is not treated as an authoritative receipt.

`downloaded-image.png` is an authentic Pillow RGB 1x1 PNG with (128,64,32).
`prepared-mask.png` is an authentic Pillow L 1x1 white PNG. These test bytes are
returned by injected owner callbacks; this does not prove native tensor mask
resizing. V3 mask policy delegates nearest-exact resize, crop=disabled, strict
threshold >0.5 followed by inversion, uint8 scaling, PNG encoding, and source
shape checks to the canonical owner. Character masks are encoded before their
images; edit images before their masks. Native owner failures send no body.
The checked common multipart façade owns serialization and limits. An actual
minimum-supported aiohttp 3.11.8 wire probe verifies text/plain; charset=utf-8
for form text and application/octet-stream for files. This is a minimum-version
probe, not a locked source dependency. aiohttp emits each part Content-Type
before Content-Disposition; the canonical encoder currently reverses those two
headers. Semantic multipart content matches; exact per-part header order remains
a coordinator-owned canonical encoder acceptance gap. The dynamic
Content-Type callback must replace the staged head before request bytes and
must be preserved by the host retry capability; its production WIT bridge is
still pending.

The adapter preserves exact JSON/media bytes and completion receipts, filters
empty image URLs, downloads the remaining URLs in order, and requests RGB
materialization and ordered batch concatenation. It never substitutes RGBA.
Typed decoder rejection publishes zero outputs. Tests cover source failures,
hidden auth precedence, absolute-download credential isolation, canonical
multipart with real PNG bytes, native/auth/cost gates, ordinary retry=1,
rate-limit retry=16, download retry=5, Retry-After delegation, bounded chunk/body/
header/receipt state, cancellation, worker-loss propagation and a fresh
supervised invocation. Fresh in-memory invocation is not a process-restart test.

Task 445 remains partial: actual native image/mask ownership and shape checks,
RGB tensor allocation/batch concatenation, canonical API coercion and full
created datetime validation (the adapter currently bounds/types metadata and
preserves raw responses), guest/native/WIT bridging, signed component execution,
worker-process restart and applicable visual checks are not claimed. No fake
IMAGE bytes, live paid requests, credentials, signatures or decoder substitute
are included. The Task 434 nested library/fixture auto-discovery is exercised;
full native guest acceptance remains coordinator-owned.

Regenerate/check independently with:

```sh
/home/asvegah/repos/sim/target/sim-oracle-venv/bin/python crates/sim_test_support/fixtures/providers/ideogram/generate_source_contract.py \
  --source-root /home/asvegah/repos/sim/projects/sim/ComfyUI --check
```

The bounded V4 prompt-whitespace follow-up uses Python strip semantics, including U+001C–U+001F, only for the original minimum-length check. Raw nonempty prompt bytes are forwarded unchanged; U+200B, U+FEFF, internal controls and prompts over 10000 characters remain accepted by this source method. V3 is unchanged. A separately labeled genuine Source capture records 40 observations: 12 unchanged validation-helper calls, 16 unchanged V4 execute captures, and 12 actual request-model constructions. Seven request boundaries preserve exact prompt bytes. The extraction class shell omits full Comfy IO inheritance/schema and stops at captured sync_op before response/download/output; the whole genuine API module, validation helper and ApiEndpoint run without stubs. Native speed/seed admission before prompt validation differs from the direct Source body and remains outside this whitespace repair. Three added native regressions pass in the current eighteen-case Ideogram suite; see audit/2026-10-03-provider-source-whitespace.md. Native codecs, guest/host binding, restart and visual acceptance remain partial. Immutable source JSON, media, generator and conformance claim bytes are unchanged.

The bounded response-seed follow-up admits signed decimal text within the
adapter's existing i64 metadata domain. The retained genuine Pydantic schema
probe accepts seed="42" and records integer42; previously the Rust response
parser rejected this response before download. The new field deserializer keeps
integer, null and omitted values, rejects malformed seed values, and leaves raw
response bytes and completion receipts untouched. Three added native regressions
consume the two existing valid-string/null-created Source probes, exercise
ordered credential-isolated downloads for V3/V4, and verify malformed seeds
start no download. The current 27-case Ideogram suite and complete 484-library plus five-conformance
suite passed in the coordinator-admitted native run; normal provider Clippy passed.
Wider Pydantic integer domains and other coercions, full created error equivalence,
native media, guest/host binding, process restart and app criteria remain partial.
Original Source golds, media, generator and claim bytes remain unchanged.

The retained `created-datetime-source-observations.json` is a byte-backed projection of a separate genuine original schema observation (Pydantic2.13.5/core2.46.5), not regenerated golden provider output. Its63 Python-input cases retain exact raw payload bytes, admission, typed calendar/offset/microsecond observations and Source error fields. JSON nonfinite extensions are represented by original raw bytes plus explicit direct-float test inputs; serde JSON refusal is not claimed to exercise the Pydantic float path. Huge-integer diagnostics and wider coercion domains remain qualified. The speedate0.17.0 vendor adapter passed the 27-case native Ideogram suite and the complete provider suite, and normal Clippy passed. Production WASI/guest/runtime acceptance remains UNRUN. See `.agents/specs/sim-parity/audit/2026-10-05-ideogram-response-schema.md` for actual receipts and remaining domains. Existing source-contract.json and conformance fixture bytes remain unchanged.

A separate bounded seed-text follow-up implements the six previously open
Source40 cases: zero-only decimal suffixes (`42.0`, `-42.000`), ASCII edge
whitespace, single underscores (`4_2`), trailing space and Unicode U+2003 edge
whitespace. Field-local normalization follows the exact pinned Pydantic-core
`clean_int_str`/leading-zero rules, then delegates to the standard signed i64
parser. It never converts text through floating point. Original raw schema
capture/projection bytes remain immutable, including their historical coverage
flags; new tests select these six cases explicitly. Three added regressions
cover typed captured values, both routes' unchanged raw responses/receipts and
ordered credential-isolated downloads, and precision/invalid-text rejection
before any download. The current 32-case Ideogram suite, complete 489 library
plus five conformance tests, and normal release/all-target/all-feature
warnings-denied Clippy passed; see the seed-coercion audit below.

Seven Source40 cases remain explicitly outside parity: four arbitrary integer
values beyond i64 and three nonfinite JSON extensions. Wider configured digit
limits/error fields and historical dependency-version equivalence remain
qualified. Native media, production WASI/guest/worker, restart and app criteria
remain partial. No Source gold, schema capture, generator or shared parser was
changed for this follow-up.

A separate source-derived F64 lower-endpoint correction rejects floating JSON
seed values at or below -2^63, following the pinned original core's strict lower
comparison before the existing checked integer conversion. The exact integer
JSON token i64::MIN remains admitted, as do the first interior representable
F64 and integer values above 2^53 without floating conversion. Integer tokens,
six-text normalization, booleans, null/omission and the created-field adapter
are unchanged. Two added mock regressions cover decimal/exponent F64 tokens,
integer distinction/precision and V3/V4 refusal before any download. The same
current native and Clippy gates passed. A genuine Python 3.12.14 / Pydantic
2.13.5 / core 2.46.5 observation through both original model methods confirms
that the floating endpoint rejects, its first interior F64 admits, and the
integer endpoint admits. This is API-schema evidence, not full Source-node
execution. See `.agents/specs/sim-parity/audit/2026-10-05-ideogram-seed-coercion.md`.
Wider integer/nonfinite JSON/error/digit-limit
domains and full production media/guest/worker/app criteria remain qualified.
