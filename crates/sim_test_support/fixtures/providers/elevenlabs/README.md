# ElevenLabs source and transport evidence

This fixture executes all eight pinned ElevenLabs node classes and the actual
shared client multipart plain-field loop. The ten request probes and five
preflight failures preserve source model fields, omission, ordering, booleans,
query formats, uploads, UUID and native audio preparation boundaries.
The catalog has nine rows: seven executable POST routes and two dynamic-path
prefix tombstones (0027 and 0031). VoiceSelector uses no authentication or paid
request. All 21 named source voices are checked.

Regenerate or compare only when no tests are reading fixtures:

```sh
python3 crates/sim_test_support/fixtures/providers/elevenlabs/generate_source_contract.py --source-root projects/sim/ComfyUI --check
```

`canonical-response.wav` is a real 60-byte PCM WAV: mono, 8000 Hz, eight signed
16-bit samples `[0, 1000, -1000, 2000, -2000, 500, -500, 0]`. It exercises raw
binary response preservation and the injected audio materializer boundary.
The source raw response decoder autodetects encoded audio; tests do not claim
this WAV is MP3, Opus or AAC. Canonical multipart tests also embed these actual
WAV bytes with `audio/wav`, preserving repeated files and input order. Those
are serializer tests, not fabricated source MP4/AAC input preparation.

The provider delegates retry requests, Retry-After dates, cancellation and
request staging to its transport. HTTP statuses 408/500/502/503/504 get three
ordinary retries; 429 gets sixteen independent rate-limit retries, exponential
backoff and the shared 150-second Retry-After cap. Retries replay the exact body
and reject stale/cross-invocation handles. Supervisor cost approval precedes
native preparation/upload/body submission. Source price badges remain per-unit
metadata; invocation maximum cost is explicitly supplied by the host.

The checked multipart façade reuses the existing common serializer and its
field/size validation. JSON and injected transcription requests are executable
through the hermetic transport. The typed transcription adapter preserves
source word defaults, omitted nulls, two-space pretty JSON and ASCII Unicode
escapes. Clone and selector results retain the custom `ELEVENLABS_VOICE` type
(Model family); they are not relabeled as STRING.

Remaining full Task 440 acceptance gates are canonical native AUDIO input
preparation (MP4/AAC), actual storage upload, host UUID/file-order resolution,
immutable multipart-head binding through the production host/compiled guest,
canonical native AUDIO/custom-voice allocation and publication, compiled guest
signing/distribution, real worker-loss/restart and native output visual checks.
Tests deliberately fail the missing native callbacks with zero publication;
no metadata-only audio or fabricated model descriptors count as decoded output.
This batch does not mark Task 440 complete, contact paid endpoints, use real
credentials or publish a guest/PR.

The bounded Python-whitespace follow-up uses one local strip predicate for
text validation and language-code omission. Original Python strips U+001C
through U+001F; Rust's default trim does not. Text-to-speech, sound effects and
each dialogue segment reject separator-only text. Speech-to-text, text-to-speech
and dialogue omit separator-only language codes. Nonempty original text and
language bytes are preserved; U+200B and BOM remain nonempty.

A genuine standalone original-AST probe used Python 3.12.14 and passed 36 text
cases (31 rejected, five accepted) plus 108 original language-conditional cases
(93 omitted, 15 original strings preserved), across all 29 Python whitespace
codepoints and explicit U+200B/BOM negatives. Static binding covers the three
actual validate_string calls and three language .strip() conditionals. This
executes the original helpers/expressions, not full API nodes or paid transport,
and imports no Torch. The source speech-to-text request evaluates its actual
audio upload before its language conditional. The existing provider performs
side-effect-free request validation with an explicit non-executable upload
resolution before approval, then invokes upload before final multipart/body
submission; this follow-up preserves that existing host safety boundary.

Three inline regressions passed with all 18 focused ElevenLabs tests. They
cover every dialogue position up to ten, the three text branches before
approval/preparation, original nonempty bytes, language omission in all three
branches, and speech-to-text's injected upload-before-submission and typed
upload failure. Raw audio responses retain the actual existing WAV bytes;
these checks do not certify MP4/AAC preparation or decoded native audio.
The isolated provider all-target suite passed 314 library tests and five
conformance tests. Normal repository release, all-target/all-feature,
warnings-denied Clippy, nested/workspace formatting, immutable original-source
check and diff checks passed. Protected source, product and dependency hashes
remained stable throughout the serialized run. Existing source JSON, media
golds, eight nodes, nine routes and aliases remain unchanged. Full Task 440
and its native/guest/restart gates remain partial.

The separate multipart-head follow-up predeclares the canonical encoder's
checked boundary in the initial immutable request head, before transport start
and host cost approval. Body preparation follows approval; the final encoded
Content-Type must equal that authorized header. A missing header callback,
invalid boundary or cancellation fails before any head/body/approval; a changed
encoded header fails before body submission. The vendor transport no longer
requires a mutable Content-Type callback absent from the streaming WIT.

Original client.py:663–693 constructs one aiohttp FormData, preserving plain
field order and repeated file order, and removes its fixed JSON Content-Type;
aiohttp serializes that same form boundary with its request at line 710.
Original isolation/clone/speech-to-speech MP4/AAC preparation and transcription
upload occur before the request call. The host's existing authorization before
native preparation remains an intentional phase difference, not a claim of
identical source actuation order. Four regressions cover header binding
at cost approval for all four multipart branches, default-unbound/invalid and
cancelled callbacks, JSON independence, mismatched encoded headers, and exact
canonical multipart retry replay. Existing actual-WAV serializer order and
body-limit tests remain. All 22 focused tests passed, including these four new
regressions. The isolated all-target suite passed 318 library tests and five
conformance tests; normal repository release/all-target/all-feature
warnings-denied Clippy, nested/workspace formatting, the unchanged 144 original
helper observations, immutable original-source and diff checks passed. No
compiler or runtime failures occurred in this follow-up. Frozen inputs stayed
unchanged throughout all eight gates. This validation covers the native
hermetic provider library and injected adapters. Actual native
MP4/AAC, storage signing, compiled guest/header adapter and native AUDIO/custom
voice materialization remain unproved; no new source JSON or media gold is
regenerated and Task 440 remains partial.
