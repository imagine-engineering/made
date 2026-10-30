# WIP transfer checkpoint — 2026-10-15

The user replaced implementation with immediate preservation and handoff. Stop
feature work and broad validation. This checkpoint does **not** complete the
collaborative-workspace remediation, any production journey, visual parity or
functional parity. Resume only on a new explicit instruction after delivery.

## Preserved revisions and scope

- Original checkout/base and fetched `origin/dev`:
  `6c4db6ee24461921f98483e44a52661a0a745f63`.
- Source WIP commit: `d4828cc93da67e6568a2c45908167d94132b57de`.
  It preserves all112 remaining source/test/tooling/deployment/documentation
  files, including previously separated domain, service and CLI work. It adds
  migration **definitions**; no deployed migration or deployment was performed.
- The initial fresh inventory was258 changed files:101 tracked,157 untracked.
  [File inventory](file-inventory.json) contains every path, size and SHA-256.
- Frozen **Source15247** is a narrower primary UI checkpoint over the base:
  manifest `4a7f0de4b5ebfa410eb4c58a4a2a527489eaf6189053dead53433689fa43e440`,
  patch `d61559fc5740ab02c95eb6289740db62bf0915a7f2886fd346db42fe8f6c93bb`.
  Its results do not qualify the combined source WIP commit.
- Find this document's preservation commit with
  `git log -1 --format=%H -- docs/collaboration/handoffs/20261015/README.md`.
  Final delivery SHAs and remote verification are reported in the sending chat;
  the commands below independently verify them.

## What works, and what remains unfinished

Source152's rendered picker tests pass in both product profiles. Its normal
Multiplayer application builds; affected16-owner repository lint passes in both
profiles. Native model Configure closes the picker before opening settings,
returns to one integrated conversation, and restores composer input. Empty
search, Escape, repeated Configure, global settings return, light/dark rendering
and the captured narrow state have bounded evidence.

The approved marker-only provider request completed in verified Source149:
`COPPER_V44_HISTORY`. Its inline terminal output moved into conversation history;
targeted Stop retained output and independently removed shell82621/parent82614.
Source152 then restored that completed conversation after normal application
replacement. Earlier CredentialsMissing failure remains retained historical
failure evidence, not the current provider state. New-machine sign-in is required.

**Combined production acceptance remains incomplete.** Standard normal current
application, unfiltered current owner suites, full profiles, complete matched
Delta/Copper journeys, all remaining Buzz/domain/service/CLI tasks and applicable
SQL/physical-standby cleanup validation are pending. Authenticated Buzz
collaboration remains deferred. Existing task checkboxes are not promoted.

A separate **uninstalled Source153 candidate** changes AgentPanel, MessageEditor
and ThreadView, with five unrun regressions. It addresses:

1. `CW-INLINE-TERMINAL-ONLY-DRAFT-RETARGET-20261015`: terminal-only drafts can be
   mistaken for empty when changing agents; preserve original content/agent and
   retain true-empty/whitespace retarget behavior.
2. `CW-INLINE-DELAYED-SUBMISSION-REVISION-20261015`: text-only edit revisions miss
   terminal custom-block additions during asynchronous submission.
3. `CW-INLINE-LATE-OPEN-FOCUS-20261015`: delayed terminal opening can steal focus
   after another conversation becomes active.

These map to21.2/55.3/56.3/59.6. Candidate patch SHA-256:
`64271c9e6e56eac36a1dbd33d14cff1f1f7cc635e6cd93afa5fdb9d0bf409885`.
Private formatting passed; compilation/runtime/native acceptance was **not run**.
Do not apply automatically. Review and validate it after explicit resumption.

## Exact validation and evidence

[Validation results](validation-results.json) records exact commands, receipt
hashes, source binding and disk minima. Closed Source152 results:

| Check | Result | Evidence |
| --- | --- | --- |
| Focused ACP Configure, language-model Configure and canonical Picker cleanup/reopen |3PASS0FAIL0IGNORE in each Standard/Multiplayer profile|Source152 focused receipts in new private capsule|
| Normal non-test Copper Multiplayer build |exit0|Source152 normal build receipt/compiler artifact|
| Affected16-owner `./script/clippy` |exit0 for Standard and Multiplayer|Both Source152 lint receipts|
| Fresh WIP `cargo fmt --all -- --check` |exit0|`preservation-checks.json`|
| WIP diff/staged whitespace checks |exit0|`preservation-checks.json`|
| Initial structured/static preservation scan |51JSON parsed;46 public archives inspected;0 bounded secret-pattern alerts|`preservation-integrity.json`|

Historical Source150 compilation exited101 before tests. Source151 compiled but
both exact rendered Configure tests failed with duplicate dismissal events;
Source152 passed unchanged assertions after the canonical repair. Source149's
full Multiplayer retry was interrupted by ROOT custody drift; no profile pass is
claimed. The complete combined WIP state was deliberately not rebuilt for this
transfer, as requested.

Existing archives are reused under
[the evidence archive tree](../../../../.agents/evidence-archives/collaborative-workspace).
The new
[Source150–153 WIP retention receipt](../../../../.agents/evidence-archives/collaborative-workspace/current150152-closed-validation-live149-and-uninstalled153-wip-20261015/README.md)
retains closed failures/passes, Source149 successful live-response evidence,
Source152 native01–12, Delta settings pairs, frozen source/validation inputs and
all19 files of the uninstalled candidate/controls. Its ignored private archive
has237 verified regular members and SHA-256
`e6dba0352147f16faa32fe04a364e7883d9590d81006f3dbd7dea66f6aac4dd1`.
No screenshots containing account sign-in details or credential/profile payloads
are added to Git. No original evidence or warm cache was deleted.

## What Git will not transfer

The compressed manifests beside this document are the authoritative exhaustive
file lists, with original absolute path, size/checksum or symlink target,
classification and transfer/review instructions. Their checksums and counts are
in `transfer-summary.json`. They distinguish required archival/support material,
reference snapshots, already-retained raw aliases, and preserved local-only
files requiring review. **Do not copy an entire user profile or `/private/tmp`.**

Required groups are ignored private evidence archives/manifests; current replay
support not already retained; the pinned WebRTC SDK if reusing this offline
macOS build; and exact ignored reference source snapshots where required for
comparison. Reproducible Rust caches and historical application binaries are
preserved locally but are not prerequisites for a new build. Original temporary
clipboard reference image has disappeared; tracked reference images remain
unchanged and their exact retained locations are recorded in the manifests.

From `docs/collaboration/handoffs/20261015`, use
`python3 transfer-files.py summary`, then `verify` on the source machine.
For the explicitly selected repository files, emit a relative list with
`python3 transfer-files.py paths > /private/tmp/collaborative-transfer-files.txt`.
Copy those files through your approved private channel, for example:

```sh
rsync -a --files-from=/private/tmp/collaborative-transfer-files.txt \
  /Users/ahmad.vegah/repos/imagineerings/zed/ /path/to/destination-checkout/
```

Use the tool's `verify --repository /path/to/destination-checkout` afterward.
External temporary replay helpers are retained in the support archive and must
be reconstructed according to its manifest, not copied as stale live locks.
Inspect any `review-before-transfer` rows separately; they are preserved locally
and must not be silently dropped or sent without checking for credentials. After
review, `--include-reviewed-references` adds the exact42,988 reference-source
rows to `paths` or `verify`; it never selects local-only credential/cache rows.

The default copy list contains **29,485 files /2,498,013,614 bytes**:242
private evidence/support files and29,243 SDK files. The optional reference-source
list is separate. All64 otherwise missing helpers are now in the support capsule,
so no live `/private/tmp` helper needs a separate ad hoc copy.

[The support retention receipt](../../../../.agents/evidence-archives/collaborative-workspace/current153-transfer-support-and-unmatched-evidence-wip-20261015/README.md)
records the13,752-file raw audit.11,159 paths already had exact retained payloads;
2,093 safe unmatched text paths were added to the private support archive
(35,797,701 bytes, SHA-256
`753a0926c706513a27b86a95a69c664bc8acc0c507d07e293c0ffd58db9459a9`).
Its manifest SHA-256 is
`a71988f9a456eb4072f9a3db23ec3fdcfe871d61601db6a8455e46ef370d4c35`.

Exactly **500 unmatched raw files /393,936,793 bytes** remain local, all listed
in `local-only-files.json.gz`:213 screenshots need privacy review,179 compressed
cache/recovery or unreviewed archives,93 historical source/fixture payloads
(current source/canonical packets retained),9 credential/OAuth-like flagged text
files and6 machine-state files. No flagged content is archived or committed.
These are explicit preservation exclusions, not deletion candidates. The93
historical payloads and213 screenshots remain available for optional provenance
review. Do not transfer the credential or machine-state rows.

**Credentials and machine state:** Copper isolated profiles, account databases,
macOS Keychain, SSH/AWS credentials, local environment files and queue-lock
ownership are excluded. Do not archive or transport them. Create a fresh
isolated Copper profile and complete provider sign-in on the other machine.
Authenticated collaboration is still deferred. Reference trees have no separate
Git metadata; their enclosing checkout SHA is not their upstream revision.

## Prerequisites and next commands after explicit resumption

Use Rust1.97.1 from `rust-toolchain.toml`, Cargo.lock, matching platform toolchains,
Xcode/Metal tools for native macOS rendering, Python3 and Docker Desktop. Current
native artifact was macOS AArch64. The stable WebRTC library is:

`/Users/ahmad.vegah/repos/imagineerings/zed/.build/collaborative-sdk/webrtc-0001d84-4/mac-arm64-release/lib/libwebrtc.a`

SHA-256 `281d5c33ff1ca988d0c1434c9c8133d027b15057efda3110ac4e199ce3509c6b`.
Preserve/reconstruct its full matching header/support tree. Retained validator
scripts contain old absolute paths, HEAD/index/source pins and receipts; **rebind
them deliberately to the new checkout**. They must not be run unchanged as if a
new commit or another machine were the old immutable checkpoint. Keep the
existing10GiB reserve and one shared heavyweight queue; prefer20–30GiB stable
free capacity. Preserve warm caches and SDK paths; no other builds were active
at transfer preparation. Six unrelated stopped Docker containers were retained;
future disposable runs must verify their own cleanup against a fresh baseline.

First inspect the ledger and uninstalled candidate, then run focused tests for
actual changes. The archived `commands.json` records exact22 stage/profile
commands; `validation-results.json` contains the executed canonical commands.
Required later gates include:

```sh
git status --short
cargo fmt --all -- --check
./script/check-multiplayer-tools --full --profile standard
./script/check-multiplayer-tools --full --profile multiplayer
# Use the exact affected16-owner ./script/clippy commands in validation-results.json.
# Build both normal product profiles with ZED_PRODUCT_ID=rust; no test/headless app.
ZED_PRODUCT_ID=rust cargo build --locked -j1 -p zed --bin zed \
  --no-default-features --features agentic-tools,rust-tools,gpui_platform/runtime_shaders,multiplayer-tools
ZED_PRODUCT_ID=rust cargo build --locked -j1 -p zed --bin zed \
  --no-default-features --features agentic-tools,rust-tools,gpui_platform/runtime_shaders
./script/validate-collaboration-storage --label transfer-resume-unique \
  --jobs 1 --physical-standby --evidence-dir /private/tmp/transfer-resume-storage-unique
python3 .agents/specs/collaborative-workspace/scripts/validate-community-sql.py \
  --evidence /private/tmp/transfer-resume-community-unique
```

Run heavyweight commands one at a time with reserve/source guards. The two SQL
commands create only disposable services, never point them at a live database.
Use unique evidence labels/directories. Recheck Docker responsiveness/cleanup.
Verify executable path, product catalog, feature profile, source and executable
hashes before matched real-Delta/Copper native acceptance. Delta is the UI and
interaction authority, Buzz the domain authority, Fractal the visual language.
Component/fixture/native/live-service evidence remain separate.

For remote verification after fetching:

```sh
git fetch origin dev
git merge-base --is-ancestor d4828cc93da67e6568a2c45908167d94132b57de origin/dev
git ls-remote --heads origin refs/heads/dev
```

The isolated Copper Source152 validation app was normally quit for handoff;
app55720 and the disposable shell/parent are independently absent. Both assisting
agents finished. No other app, service or warm cache was interrupted. All29,485
selected manual-transfer files were rehashed successfully with0 failures.

No implementation resumes automatically after this WIP transfer.
