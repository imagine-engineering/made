# Pinned Krea source conformance

Task447 owns two source nodes and six route claims. The `/proxy/krea/jobs/`
claim is a synthetic prefix tombstone; the job-ID GET is executable. Aliases
are empty. `source-contract.json` executes the pinned node `execute` AST and
actual Pydantic API classes, rather than deriving expected bodies from Rust.
It records 103 generation cases, seven preflight failures, two copied style
chains, real source Torch/Pillow PNG bytes, and actual aiohttp multipart bytes.

The source model uses title-case choices, forwards the seed, and validates
prompts without stripping or a maximum length. Moodboard IDs are stripped
with Python whitespace rules before UUID checking. Style generation omits a
null URL but requires the URL key; the append node preserves existing dict
keys and checks its ten-item limit before image preparation/upload. Uploads
use `/proxy/krea/assets`, PNG, and a 2048² total-pixel budget. Source resize is
Lanczos/disabled with even truncated dimensions; source tensor bytes truncate
without a clamp. Only the first completed result image URL is downloaded in
RGBA mode with source orientation behavior. Native preparation/decoding is
owned by canonical media/context adapters, not implemented inside this leaf.
The public `style_preparation_requirement` preflight must run before those
owners allocate/encode, preserving the source ten-chain check order and its
first-batch/PNG/total-pixel/Lanczos/disabled preparation contract.

## Reproduction

```sh
PYTHONPATH=/tmp/sim-aiohttp-source-probe \
 /home/asvegah/repos/sim/target/sim-oracle-venv/bin/python \
 crates/sim_test_support/fixtures/providers/krea/generate_source_contract.py --check
```

The cached aiohttp 3.11.8 probe is the source minimum-supported version; it is
not a locked source dependency. Torch/Pillow/aiohttp versions and every source
file SHA are retained in the oracle. The UUID generator is instrumented only
to freeze the randomized source filename. All encoding and multipart bytes
come from actual source functions and the actual encoder/HTTP library.

## Executable scope and remaining acceptance

The leaf handles one active HTTP request, distinct request receipts, fresh
handles retained under one invocation, rejection of any repeated handle,
source create/upload/poll/download retry budgets, rate-limit backoff and
Retry-After bounds, and caller-approved cost ceilings. Upload ordinary retries
are one, generation three, polling ten, downloading five; API rate retries
are sixteen. Required host deadline hooks preserve source 3600-second create
and 120-second poll deadlines, with no download total timeout. Krea queue
states are backlogged/queued/scheduled; `in_queue` consumes the source
480 processing-poll budget. A separate 4096 total-poll and 8192 handle bound
fails closed even if source queueing would otherwise run indefinitely.
Host stream guards must verify the exact retained prompt/attempt/execution
node/plan identity for every continuation; receipt bytes are selectors for
trusted verification, never transport grants. Host cancellation also owns
cleanup of failed outstanding requests.

Materialization accepts only caller-owned SDK ABI values after canonical
receipt/context verification and actual RGBA allocation or custom-chain
construction. Values validate as one atomic batch before publication.
`KREA_STYLE_REF` is the source runtime socket literal. The existing checked
registry binds the catalog identity `KreaIO.STYLE_REF` to
`sim:custom-krea-style-ref@1`, family Model/ModelV1. The leaf uses that
registration; its native retained custom model-handle construction remains
a shared owner gate. No invented alias or scalar substitution is used. NativePayload
and SDK PluginValue are separate representations and require the checked
shared bridge. Actual compiled guest/WIT continuation, upload preparation,
custom model-handle retention/native allocation, restart/app visual checks and
source elapsed-progress ticker remain shared gates. Nonfinite float and
non-JSON incoming values fail closed. Unsafe job route selectors and altered
response job identity are rejected as host authority checks.

The source price expression looks up lower-case model rate keys while schema
choices are title-case. The widget normalization needed to evaluate that
expression is unproven. This leaf does not fabricate a normalized source
price; the caller-authorized maximum remains independent.

The leaf originally exposed reversed multipart header order through an
explicitly run, failing source-wire diagnostic. Coordinator integration now
emits Content-Type before Content-Disposition through the common checked
encoder and accepts source empty-field values. The unchanged actual aiohttp
bytes pass the normal, non-ignored `source_exact_multipart_wire_diagnostic`.
Independent common FormData captures also cover repeated files, UTF-8 text,
empty text and empty file bytes. Source golds were unchanged. The initial red
wire result remains recorded; production guest/upload/native acceptance still
requires the owner bindings described above.

Task447 request-admission ordering follow-up passes current-branch native validation
(audit/2026-10-03-provider-reference-error-order.md).
Original execute checks model selection, moodboard and style admission before
request aspect_ratio/resolution/creativity lookups. Implementation defers only those
three existing string lookups, preserving valid raw request fields, numerical
and whitespace rules, upload/media, cost/auth/cancellation and route ownership.
Three native regressions pass in the thirteen-case suite and cover combined-fault precedence, zero transport/cost on request
failure, and valid raw optional bodies plus later missing-base-field failure.
A genuine unchanged execute/API capture has thirteen cases with explicit sync
capture before HTTP/cost/poll/media. No full IO schema, original decoder,
guest, real clocks/restart/visual or learned media acceptance is claimed.
All source JSON/generator/media/conformance fixture golds remain untouched.

Current-branch Krea follow-up validation:13 focused cases, complete452 library
plus5 conformance cases, normal nested Clippy, both formatting checks, immutable
Source checks and diff pass. See audit/2026-10-03-provider-reference-error-order.md.
The three proposed native regressions now pass; remaining media/guest/real
clock/permission/restart/visual gates remain unverified.
