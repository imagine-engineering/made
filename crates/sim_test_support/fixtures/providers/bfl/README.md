BFL source contract and transport fixtures

Task 437 is partially implemented. This directory is auto-discovered by the
Task 434 nested provider component harness. It covers the exact ten declared
node claims and nine routes with no shared registry, manifest, or lock changes.

`generate_source_contract.py` executes the pinned BFL node request AST and
actual scalar/aspect/dimension validation helpers against explicit media boundary
probes, preserving Python request-model defaults, aliases, omitted optional
fields, reference ordering, integer seed range, and decimal rounding. Run it
with `--check` to verify the checked request oracle without writing fixtures.
The fixture records hashes for the node, API models, transport, authentication,
validation, download and image-conversion source files.

The media probes do not establish numerical image preprocessing equivalence.
Canonical RGB projection, mask resize/conversion, first-image/batch selection,
reference resizing and base64 encoding must be supplied by the host's existing
media owners. The source image decoder converts downloaded images to RGBA F32;
`downloaded-image.png` is an actual CRC-valid 1x1 RGB PNG containing [128,64,32]
and therefore expects RGBA [128/255,64/255,32/255,1] from that canonical decoder.

`bfl.rs` retains the response bytes, source URL, terminal receipt and optional
source price in cents converted to USD. Its narrow BflTransport callbacks
require the canonical caller to enforce delay, retries, Retry-After HTTP-date
clock interpretation and progress. Generation, polling and download have
separate pinned retry limits and backoff. Relative cloud endpoints receive
host capability placeholders with token-before-key selection; absolute polling
and download endpoints receive no credentials. Canonical endpoint policy remains
in the host. A caller-authorized maximum is required before request-body
transmission; host cost/idempotency authority must account for all retries.

BflImageMaterializer delegates decoding to the canonical host image owner.
The callback failure test proves no output is published. No successful native
image allocation, WASM guest deployment, signing payload, real worker restart,
production signature, private credentials or live paid request is claimed by
these fixtures. Task 437 remains incomplete until those executable acceptance
criteria pass through the existing shared owners and required final gates.

The bounded ratio-strip follow-up corrects Python's U+001C–U+001F whitespace
admission around aspect-ratio integer parts for Ultra and both Kontext nodes.
Validation strips those separators locally; original aspect-ratio and prompt
bytes remain unchanged in the paid request. Original validate_string is called
with strip_whitespace=False and no minimum length, so empty and whitespace-only
prompts remain accepted. U+200B, BOM and U+180E remain invalid numeric wrappers.

A separate tiny Python 3.12.14 probe executed the exact pinned AST bodies of
validate_aspect_ratio_string, _parse_aspect_ratio_string, _ratio_from_tuple,
_assert_ratio_bounds and validate_string: 109 ratio observations (92 accepted,
17 rejected), plus six preserved empty/nonempty prompt observations passed.
The ratio matrix covers all 29 Python whitespace codepoints, three non-whitespace
negatives, ratio bounds and three broader Python integer admission examples.
This is genuine original helper execution without Torch, not full node/model
execution; source JSON, existing AST-node cases and media golds are unchanged.
At the ratio-strip checkpoint, Python underscore numerals, Unicode decimal
numerals and integers beyond u64 were distinct native parsing gaps. The
subsequent bounded underscore draft below addresses only the checked u64
grammar; it does not fabricate an arbitrary-integer owner. Source response
constructors/status enums and current host cost guards are unchanged.

Two runtime regressions compare each affected request against its existing
independent source gold with only ratio/prompt fields changed, invoke twelve
scripted submit→poll→real PNG-download paths, and reject 33 invalid ratios before
any cost/transport callback. Source boundary ratios remain inclusive. All 13
focused BFL tests passed, followed by 320 library tests and five conformance
tests. Normal repository release/all-target/all-feature warnings-denied Clippy,
nested/workspace formatting, immutable original-source and diff checks passed.
The one-job, single-test-thread warm-cache run used nice10 and one-thread
numerical libraries while the coordinator's existing debug inference reader
continued. Its resource monitor saw only the owned component compile and
available disk stayed above the required 5 GiB floor. Frozen source/media,
product and dependency hashes stayed stable during the eight terminal gates.

The first monitor attempt stopped the intended compile because its allowlist
used the Rust identifier rather than Cargo's hyphenated package name; that
interrupted receipt remains preserved. The first focused run passed 12 and
failed one new scripted request on JSON key order. The test now independently
checks request values/path against the unchanged source gold, then scripts the
canonical Rust serialized bytes for exact transport replay; Python and Rust
JSON insertion order is not treated as identical wire encoding. No source
gold, numeric tolerance or production behavior was adjusted for that failure.

A separately recorded Python 3.12.14/Pydantic 2.13.5 capture executed unchanged
complete Ultra/KontextPro/Max class AST bodies, original validation helpers,
and original API models: 90 captured successful paths and 18 expected source
rejections across 108 class invocations, plus six accepted status-enum values
and one rejected unknown status. The original Ultra custom validate_inputs
hook was invoked and logged separately before execute. A None prompt fails
before media preparation when there is no image; with an image, the captured
media callback runs before the genuine Pydantic constructor rejects None.
IO/schema/base classes/NodeOutput, endpoint/network/cost, poll timing, base64
and download owners are explicit captures. Full original module imports,
actual media encoding/decoding and production cost authority were not run.
This evidence does not convert captured media into native tensor success.
Numerical image allocation, compiled guest, actual clock/restart and production
host authority gates remain partial.

The validated underscore follow-up accepts Python decimal underscores between ASCII
digits, an optional leading plus sign and the existing Python whitespace
wrappers within checked u64 capacity. Leading, trailing, repeated or
sign-adjacent underscores still fail; zero/nonpositive parts and ratio bounds
retain their existing typed rejection. Numeric accumulation uses checked
standard u64 operations, with no private bigint arithmetic or dependency
change. Original prompt and aspect-ratio bytes remain in the request. Two
new focused regressions cover 30 independently source-derived request values,
three full scripted submit/poll/real-PNG-byte downloads, 48 original-source
rejections before any cost/transport callback and checked overflow. The
current native tests and lint gates passed as recorded below.

A tiny genuine Python 3.12.14 stdlib probe executed the original helper bodies
for 44 observations: 22 accepted and 22 source-rejected. It distinguishes
Unicode decimal admission, arbitrary-size integers, the current configured
4300-digit decimal limit, float overflow and correctly rounded integer
division from the executable underscore subset. Unicode digit decoding,
arbitrary-size integer authority and configured digit-limit parity remain
canonical numerical prerequisites. The historical independently cast quotient
rejected Source-valid u64 boundaries at 4.0 and 0.25; the 2026-10-05 shared
locked rational owner fixes this bounded rounding gap. Broader Python integer
parsing and execution acceptance remain incomplete. Existing source JSON, generator and media golds
remain unchanged. New source receipts live under
`target/sim-validation/bfl-integer-grammar/`; no full node, native tensor
allocation, learned media or production request authority is claimed.

Historical intended-checkout validation passed all 15 BFL tests, the complete
336-test worker library suite, all 5 generated conformance tests, normal
release/all-target/all-feature repository script Clippy with warnings denied,
nested and workspace formatting, immutable original-source checking and diff
checking. The source/media golds and reviewed excluded dependencies stayed
hash-identical throughout all seven terminal gates. The resource-bounded run
used the existing warm target, one build job/test thread, low OS priority and
one-thread numerical libraries while the coordinator's separate pose runtime
remained active. Only the provider package rebuilt; no resource stop or
failed gate occurred. Exact commands, counts, timing and monitored resources
are retained in `target/sim-validation/bfl-integer-grammar/`. These passes
certify only the checked-u64 underscore subset and existing hermetic paths;
the broader numerical and native/guest acceptance gaps remain unresolved.

## Current scalar admission validation — 2026-10-05

BFL now converts checked positive u64 integer ratios through the locked num-rational0.4.2 Ratio<u64> owner, rounding the quotient once. The formerly rejected Source-valid 4.0 boundary and reciprocal 0.25 boundary are admitted without changing raw request ratio/prompt bytes or the existing lexical grammar. Shared numeric ownership and the two direct dependency edges are coordinator harness changes; no package versions or checksums changed. Anthropic prompt admission now includes Python strip whitespace U+001C through U+001F; controls-only prompts reject and admitted strings retain their original bytes.

Current coordinator checks passed18 BFL,12 Anthropic,2 shared numeric tests, the full475 library plus5 conformance suite, normal repository Clippy, both format checks and unchanged Source checking. The selected std-only rational artifact matches13 genuine original helper float bit patterns with zero observed allocations. See `.agents/specs/sim-parity/audit/2026-10-05-provider-scalar-admission.md` for exact logs and qualifications.

Tasks435 and437 remain partial. Unicode decimal parsing, arbitrary-size Python integers, configured4300-digit limits, broader feature profiles, universal allocation/cancellation authority, WASM/SDK, native media, full claim/route conformance, real billing/poll clocks, OS restart and application/visual acceptance remain open. Native Contract errors do not reproduce Python exception types. Anthropic build_request tests prove prompt rejection before request construction, not every execute/upload ordering or public-schema behavior.
