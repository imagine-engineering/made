# Pinned Magnific source contract

Task450 owns five nodes, fifteen route claims, and the `freepik` alias. Five
prefix claims are synthetic tombstones; ten routes execute. `source-contract.json`
executes the pinned `execute` AST and actual Pydantic API classes. It records
227 successful requests, 24 preflight failures, 36 independent source price
cases, seven response alias cases, actual source Torch/Pillow PNG bytes, and
actual storage request DTOs. Source hashes and library versions are retained.

Reproduce without mutating the frozen fixture:

```sh
/home/asvegah/repos/sim/target/sim-oracle-venv/bin/python \
 crates/sim_test_support/fixtures/providers/magnific/generate_source_contract.py --check
```

All nodes require exactly one image, dimensions at least 160, and an inclusive
1:3 through 3:1 aspect ratio. Style transfer requires one reference; relight
allows an optional reference. Creative upscale limits output to 25.3 million
pixels; precise V2 limits the longest output axis to 10060. Auto downscale
selects source factors and source even, truncated resize dimensions. Shape-only
resize probes execute source decision logic and record Lanczos/disabled calls;
they do not prove native numerical resize parity. Creative retains its source
string scale, while precise sends an integer. Progress prices use the source
EUR tiers, 1.19 conversion, and rounding to cents, independently of the caller's
approved maximum cost.

Every upload defaults to PNG but deliberately omits MIME in the source storage
create body and signed PUT. `None` is preserved by a checked `UploadRequirement`.
The existing required-String storage DTO cannot represent this omission; an
empty String is not a substitute. `MagnificUploadOwner` defaults to typed
`HostFailure` until the canonical optional-content-type binding is available.
That owner must encode canonical source tensors, retain the exact authorized
invocation context, validate storage create and signed PUT completion receipts,
and return their fresh handles. The leaf checks distinct handles, nonempty
bounded receipts, roles, URLs, and common invocation identity before generation.
Receipt bytes are trusted-owner selectors, never transport grants. Unit mocks
exercise this boundary but do not prove actual storage HTTP/native upload.

Request fields preserve source omission, portrait and advanced controls, and
skin's unknown-mode fallback to flexible. Task responses unwrap a dict `data`,
prefer `status` over `task_status` even when the former is null, and require the
source strings. Skin queries use the generic skin-enhancer/task-id route.
Only the first generated URL downloads, without Comfy credentials. Source
queued/completed/failed statuses, ten-second poll intervals, create/poll/download
ordinary retry budgets of 3/10/5, API rate retries of 16, Retry-After bounds,
and source create/poll deadlines of 3600/120 seconds are retained. Downloads
have no total timeout. Additional bounded 4096 total polls and 8192 retained
handles fail closed; source's 480 nonqueued polls remain separate.

Poll status/queue/price notification is an intermediate host callback. It does
not finalize or acknowledge the whole invocation. The source elapsed-time
progress ticker and exact GUI widgets remain shared integration work. A missing
callback returns typed `HostFailure` instead of silently dropping progress.
Actual downloaded images must use the canonical RGBA decoder and caller-owned
compute/context/receipt binding. SDK PluginValue and NativePayload require their
checked bridge; this leaf never fabricates tensor metadata. Output validation
precedes one atomic publication, with cancellation checked before publication.
The negative native-owner regression proves zero publication when unbound.

Compiled guest/WIT continuation, canonical optional-MIME upload, source tensor
resize/encode allocation, native image materialization, receipt persistence and
restart, elapsed progress, and app visual checks remain shared acceptance gates.
These fixtures establish source request/preparation/transport contracts, not
full Task450 completion.
