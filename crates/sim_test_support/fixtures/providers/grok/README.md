# Grok pinned-source request and transport evidence

The oracle executes the seven pinned Grok node classes and actual source
validation and price helpers. Nine successful request cases cover every node;
twelve independent preflight failures cover prompt, reference count, custom
aspect ratio, video model requirements, duration epsilon and the 50 MiB limit.
Four pricing cases preserve USD ticks divided by 10,000,000,000 and the source
1.43 multiplier for video 1.5, reference generation and extension.

All seven catalog row identities remain unchanged. 0213 is the synthetic GET
prefix tombstone; the other six routes are executable. Every video branch polls
0217 at `/proxy/xai/v1/videos/{initial_response.request_id}`. The only alias is
`xai`. Legacy Image Edit remains available for saved workflows.

```sh
python3 crates/sim_test_support/fixtures/providers/grok/generate_source_contract.py --source-root projects/sim/ComfyUI --check
```

Image edit and video first-frame inputs use PNG data URLs in exact canonical
owner order. The modern image editor preserves flattened dynamic input order.
Video references use ordered PNG storage uploads and the source helper's first
seven images. Video edit and extension request MP4/H.264 upload through the
canonical media owner. The source omits extension seed and model from the
request despite their UI controls. Prompt validation checks stripped text but
forwards the original string. Source video duration validation includes its
0.0001 second epsilon; unavailable duration skips this source validation while
the canonical owner remains responsible for surfacing inspection failures.

Native probes in the source oracle are explicit structural boundaries. Their
encoding marker strings are never submitted as image/video bytes. Executable
callbacks reject invalid inline Base64. `downloaded-image.png` is an actual
69-byte RGB 1x1 PNG, and its Base64 twin carries the same bytes. Download tests
preserve exact bytes and receipts without credentials on absolute URLs. The
source image converter defaults to RGBA; only the canonical decoder may allocate
or concatenate those native tensors. Video download tests end at a typed host
failure rather than inventing an MP4 or decoded VIDEO.

Retries preserve immutable bodies on fresh handles and delegate delay parsing,
backoff waiting and polling sleeps to the transport supervisor. The source
shared client uses three ordinary/16 rate-limit retries, and polls use ten
ordinary retries with 1.4 backoff. Downloads retain their separate five-retry
policy. Null/missing video status becomes `complete`; normalized source queued
statuses do not consume the 480 active-poll budget. A finite supervisor total
bound also protects against perpetual queueing. Typed cancellation, worker loss,
malformed responses, oversized chunks/bodies and materializer failures publish
no native outputs. Clean re-invocation tests do not claim an actual worker
process restart.

Full Task 442 remains partial pending canonical native shape, input ordering,
MP4/H.264 preparation/storage upload, RGBA batch and VIDEO allocation, production
host/WIT integration, signed compiled guest packaging, real worker restart and
native visual acceptance. No fake native output, live paid call, credentials,
private signing key, push or PR is included in this bounded batch.

The bounded Python-whitespace follow-up preserves the published Grok cost
operation fix and reuses one local strip predicate for prompt validation and
poll status normalization. Python strips U+001C through U+001F as whitespace;
Rust's default trim excludes them. Separator-only prompts must fail before
request handles, cost approval or encoding. Nonempty prompts retain their
original bytes, and U+200B zero-width space is not stripped. The conformance
fixture pins the actual original validation/client helper hashes and states
these source semantics without changing nodes, routes, aliases or source golds.

A standalone original-AST helper probe passed 36 prompt cases (31 rejected,
five accepted) and 147 status cases across all 29 Python whitespace codepoints,
including U+001C–U+001F and U+200B negative cases. Static original callsite
inspection bound the same validate_string(strip_whitespace=True,min_length=1)
to all seven Grok execute methods. The probe used the existing original oracle
Python 3.12.14 without importing Torch, API modules or network clients. A
separate Python 3.14.4 observation also passed; neither observation executes
the seven original full API nodes.

Four inline regressions passed with all 20 focused Grok tests. They exercise
all seven Rust invocation preflights, original nonempty prompt bytes, wrapped
queued/processing/completed/failed polling, and an unknown U+200B status.
Completion probes stop at a typed host download failure; image checks retain
the existing real PNG bytes. The isolated provider all-target suite passed
311 library tests and five conformance tests. Normal repository release,
all-target/all-feature, warnings-denied Clippy, nested/workspace formatting,
immutable original-source check and diff checks passed.

The initial locked Cargo attempt failed before compilation because the worker's
nested manifest and lock were stale. Exact published coordinator prerequisites
were imported with committed-byte and vendor-source verification; no dependency
resolution or version change was performed. The initial draft writer's Python
Unicode-escape SyntaxError occurred before source mutation or runtime. Both
failures remain in ignored evidence separately from source outcomes. The
existing source JSON and media golds remain unchanged. This follow-up does not
close native media/guest/worker/restart or full Task 442 acceptance.
