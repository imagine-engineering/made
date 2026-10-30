# Validation scope

The required `run_tests` workflow validates the enabled Copper (`rust`) product rather than every test target in the repository. Its `copper_tests` job is generated from the product manifest features and an explicit package list in `tooling/xtask/src/tasks/workflows/run_tests.rs`.

### Graphics product boundary

The repository toolchain is Rust 1.98.0 for development and CI. Native graphics
are an optional dependency admitted by `zed/sim`; Copper and the guarded
Groovy preview do not link or register Graphite actions. Shared GPUI rendering
remains available in every product. The required `graphics_feature_validation`
job runs the graphics core/UI suites, native Sim lifecycle checks, and action,
menu, and keymap-reload checks in Sim and Groovy configurations. Copper runs
the disabled-graphics check in its ordinary product test suite. The existing
Rust feature-boundary script also checks all three dependency graphs.

### Cargo dashboard correctness and performance

`cargo_dashboard_reconciles_ten_thousand_rows` runs in the normal concurrent suite. It verifies exactly 10,000 unique rows in projection order, visible-range contents and boundaries, selection/expansion retention after reconciliation, and clearing a removed snapshot.

`cargo_dashboard_foreground_budget` runs in a separate, single-test-thread measurement step using the **same unoptimized test profile and Copper feature graph**. It warms up three times, collects all 21 measurements, reports min/median/p95/max, and gates the median against the unchanged 250 ms budget. Samples are measurements, not retries: there is no early success, ignored test, increased threshold, or retry-on-failure. Nextest also reserves all test slots for both dashboard budget tests when developers run an unsplit suite. Fixture projection/cloning and correctness assertions are outside the measured `new host → replace snapshot → expand all` boundary.

`cargo_dashboard_production_foreground_budget` retains the live-panel measurement from `dev`: metadata refresh, action notices, preset changes, stable row identities, and a painted bottom viewport with at least 10,000 rows. It independently gates each phase against 250 ms for foreground scheduling, admission, reconciliation, and paint. Both budget tests run sequentially in the isolated measurement step; the production measurement keeps its original per-phase assertions. The panel fixture supplies deterministic metadata and Rust test-discovery runners, so unrelated discovery does not launch host processes or wake GPUI from an external filesystem thread.

Reproduce the focused measurement locally (Rust 1.98.0, matching CI):

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_TEST_DEBUG=0 \
ZED_PRODUCT_ID=rust ZED_PRODUCT_UPDATE_BASE_URL=http://test.example \
CC=clang CXX=clang++ \
cargo nextest run --no-default-features \
  --features zed/agentic-tools,zed/rust-tools,remote_server/rust-tools \
  -p cargo_ui -p zed -p remote_server \
  -E 'package(cargo_ui)' --test-threads 1 --success-output immediate
```

The linked failure on `54ac8b76a08c` measured 250.820748 ms while running 3,663 tests on Ubuntu 22.04. That commit did not change the dashboard test or tree reconciliation. On macOS ARM64 with the same Rust profile/features, the unchanged test passed 12/12 isolated samples (117–136 ms); 12/24 samples failed with 12 concurrent test processes (up to 452 ms). This establishes sensitivity to scheduling load, not a hardware-normalized Linux performance baseline. The dedicated Linux CI measurement remains the acceptance check for that runner.

### Website refresh after publication

The `refresh_website` job runs **directly after successful `publish_product`** in `release_promotion.yml`, using the prepared release tag. It does not depend on a new release event or another `workflow_run`, so GitHub's `GITHUB_TOKEN` event restrictions do not suppress it. Publication and website refresh remain separate outcomes: a failed refresh never removes or republishes Copper.

Both automatic refresh and the manual `website_refresh.yml` workflow embed the exact same `script/refresh_website.py` through the Rust generator. Neither checks out application code nor builds, tags, uploads, or publishes Copper. Their GitHub token has only `contents: read` to verify real published release metadata. The jobs serialize against the same website refresh concurrency group; an active run is never cancelled. GitHub may replace an older pending run when several arrive, so recover any cancelled pending refresh explicitly if needed.

The script validates configuration before sending requests, then reads the existing GitHub release. It checks the release's article on `https://simtropolis.ai/releases` for the triggering **version, publication timestamp, and exact uploaded asset URL set**. A website commit marker, successful POST, unrelated version, or URLs embedded only in script text cannot satisfy verification. Already-current content avoids an unnecessary rebuild. Missing/draft/incomplete release metadata fails rather than producing an empty list.

- HTTP 429/503: at most three POST attempts, exponential delay unless Retry-After specifies a delay, and at most 120 seconds of retry sleep. A longer required delay fails without retrying early. Content is checked again before another POST. An explicit 503 can still follow service-side acceptance; at most three duplicate website builds are possible and are reported as a risk in diagnostics.
- Lost response, timeout, HTTP 408/500/502/504: acceptance is ambiguous. **Do not resend the non-idempotent POST automatically.** Observe the public page for up to 15 minutes; if fresh content appears, succeed, otherwise fail visibly and inspect the existing build queue before recovery.
- Invalid URL, redirects, authentication/permission errors, and other permanent errors: fail immediately without retry.
- Accepted POST: report rebuild acceptance separately, then poll for the triggering release content for up to 15 minutes. Lack of matching deployed content is a failed refresh, even when the POST succeeded. A matching page proves the required content is live; it does not identify which of concurrent/pre-existing builds delivered it.

Diagnostics contain only a stage, HTTP status, and controlled messages. They never print credential values, URL query strings, request/response bodies, headers, or exception details. `AMPLIFY_REBUILD_WEBHOOK` is the branch-scoped **Made → Amplify** rebuild credential. It is separate from `RELEASES_GITHUB_TOKEN`, the website's **Amplify → GitHub** metadata credential. Neither belongs in browser code or generated release data.

#### Account setup and target verification

As checked on 7 September 2026, neither repository-level nor available organization-level Actions secrets include `AMPLIFY_REBUILD_WEBHOOK`. Its target app/branch therefore cannot be confirmed. The URL itself is opaque and does not prove its destination. No credential value is needed in chat.

1. Sign into the AWS Amplify console and open the **existing application serving `simtropolis.ai`**. Keep its application ID, domain, and DNS unchanged.
2. In **Hosting → Branch settings**, confirm repository `simtropolis/website`, production branch `main`, and automatic builds enabled. In **Hosting → Custom domains**, confirm `simtropolis.ai` maps to `main`.
3. Open **Hosting → Build settings → Incoming webhooks → Create webhook**, name it `Copper release refresh`, and set **Branch to build: main**. Verify the displayed app and branch before copying the generated URL.
4. Open **GitHub → simtropolis/made → Settings → Secrets and variables → Actions → New repository secret**. Name it `AMPLIFY_REBUILD_WEBHOOK` and enter the complete generated URL there. Do not send it in chat, paste it into a workflow, or print it from a CLI.
5. If confirming an existing webhook with an authenticated AWS CLI, use `aws amplify get-webhook --webhook-id <ID> --query 'webhook.{appId:appId,branchName:branchName}'` and compare the app ID and `main`. This deliberately excludes `webhookUrl`. The repository secret listing proves presence only, not target or validity.
6. Run the refresh-only workflow below for an **existing** published tag. If content is already current, it will verify content and skip POST; that outcome does not test webhook delivery. Confirm real delivery after the next normal Copper publication. Do not publish a dummy release or republish binaries to test configuration.

#### Refresh-only manual recovery

After checking Amplify's queue for an accepted or running build, use **Actions → website_refresh → Run workflow → Branch: main → release_tag: the existing tag**, or:

```sh
gh workflow run website_refresh.yml --repo simtropolis/made --ref main \
  -f release_tag=rust-v1.16.6
```

Replace the tag with the triggering published release. This workflow has one refresh job; it never enters `release_promotion.yml`. Do not rerun the publication workflow to repair a website-only failure. It is safe to rerun refresh-only recovery after fixing configuration: matching live content skips duplicate POSTs. If a previous response was lost and content is still stale, inspect Amplify's queue first; incoming webhooks do not provide an idempotency key.

For read-only verification without the webhook credential:

```sh
COPPER_RELEASE_TAG=rust-v1.16.6 python3 script/refresh_website.py --verify-only
```

Local tests use fake HTTP responses, a fake clock, and no real webhook: `python3 -m unittest discover -s script -p 'test_refresh_website.py'`. They run in required release automation CI, alongside generator tests and workflow consistency checks. Update the Python source and Rust generator first, then regenerate YAML with `cargo xtask workflows` and run `cargo xtask check-workflows`.

References: [Amplify incoming webhooks](https://docs.aws.amazon.com/amplify/latest/userguide/create-incoming-webhook.html), [GetWebhook target fields](https://docs.aws.amazon.com/amplify/latest/APIReference/API_GetWebhook.html), [Nextest heavy-test isolation](https://nexte.st/docs/configuration/threads-required/).

## Copper releases

Copper is the only product currently enabled in `products/flavors.toml`. The
generated `release_promotion.yml` has three supported entry points:

- a successful completion of push-triggered `run_tests` on `release` from this repository;
- manual dispatch on `release` for a patch, minor, major, or explicit recovery version;
- a manually created `rust-v*` tag that points to a tested promotion commit on
  `release` after the bootstrap boundary.

The automatic path checks out `workflow_run.head_sha`, so it packages the exact
commit that passed CI. If there is no existing `rust-v*` tag, the first release
uses `crates/zed/Cargo.toml`; subsequent automatic releases increment the patch
component. Tags use `rust-vX.Y.Z`, and releases are titled `Copper X.Y.Z`.

The workflow resolves the version, builds Linux x86_64, macOS ARM64, and
Windows x86_64 artifacts, and waits for every matrix entry. The final job then
fetches tags again, repeats the version/history checks, validates exactly one
artifact per target against the product manifest, creates an annotated tag,
and creates or updates the GitHub Release. A failed or cancelled test run,
fork-originated run, platform build failure, version conflict, or stale release
decision cannot publish a tag or release. Workflow concurrency serializes the
entire release operation without cancelling an active release.

Request a manual release from the release branch with:

```sh
gh workflow run release_promotion.yml --repo simtropolis/made --ref release -f bump=minor
gh workflow run release_promotion.yml --repo simtropolis/made --ref release -f bump=major
```

For recovery, `version` may contain an exact stable `X.Y.Z` and `commit_sha`
may identify a full tested promotion commit SHA on `release`:

```sh
gh workflow run release_promotion.yml --repo simtropolis/made --ref release \
  -f bump=patch \
  -f version=<X.Y.Z> \
  -f commit_sha=<FULL_COMMIT_SHA>
```

Rerunning a completed or interrupted release is safe: an existing tag is reused
only when it already points to the selected commit, an existing release is
updated, and missing assets are uploaded without duplicating existing assets.
A tag conflict fails visibly and is never moved. The final publication job alone has
`contents: write`; preparation and build jobs have `contents: read`. New
releases remain drafts until the remote asset list exactly matches the three
validated local artifacts.

### Development and release promotion

`main` remains the default branch and the target of normal development pull
requests. Both `main` and `release` pushes run required CI. Only a successful
same-repository push run on `release` can automatically start release builds.
To release later, open a pull request with **base `release`, head `main`**, merge
it using a merge commit (not squash/rebase), and let release-branch CI finish.
Do not develop directly on `release` or merge it back into `main`.

Before preparation and again before publication, the workflow requires:

- repository variable `COPPER_RELEASE_ENABLED` equal to `true`;
- `COPPER_RELEASE_BOOTSTRAP_SHA` containing the permanent initial release tip;
- the selected SHA to strictly descend from that bootstrap and occur on the
  first-parent history of `origin/release`;
- the latest exact-commit `run_tests` push run on `release` to be completed and
  successful, with both repository identities matching this repository.

The automatic path also verifies the triggering CI run ID. Manual dispatch
must select `--ref release`; its optional commit must pass the same checks.
Tag pushes use the same gate. API errors and missing/invalid configuration
block publication. A failed or in-progress rerun cannot borrow an earlier
successful result. Preparation evaluates the gate before checking out and
executing candidate code; publication repeats it before resolving the version
and writing tags or releases. Only the two gate jobs need `actions: read`.

### Safe one-time cutover

Do not dispatch a release or perform a release promotion to test the migration.
Use explicit `--repo simtropolis/made` on every `gh` command: this checkout also
has an upstream remote, which GitHub CLI can otherwise select by default.

1. Record the current remote `rust-v*` refs and GitHub Releases (including drafts).
   Disable the old workflow with
   `gh workflow disable release.yml --repo simtropolis/made` and set
   `gh variable set COPPER_RELEASE_ENABLED --repo simtropolis/made --body false`.
   List **all** non-completed runs of the old workflow, cancel them, and verify
   they are terminal before proceeding. Disabling alone does not drain runs.
2. Land the generator, generated YAML, tests, and documentation on `main`.
   Retain the inert `release.yml` placeholder and verify its workflow identity
   is explicitly disabled after the merge; disable it again if necessary.
   Keep the new workflow disabled through the variable while cutover continues.
   Verify `main` remains the default branch and fetch the latest `origin/main`.
3. Create `release` at that exact migrated `origin/main` SHA and push it normally.
   Never overwrite an existing branch; inspect it if it already exists.
   Record this exact SHA in `COPPER_RELEASE_BOOTSTRAP_SHA` **before enabling**:
   `gh variable set COPPER_RELEASE_BOOTSTRAP_SHA --repo simtropolis/made --body <FULL_INITIAL_SHA>`.
   Confirm the remote release tip matches it and no new tags or releases exist.
4. Set `COPPER_RELEASE_ENABLED` to `true`. Ensure `release_promotion.yml` is active
   and the old `release.yml` remains disabled. Initial branch CI may finish before
   or after this step: the permanent bootstrap exclusion blocks it either way,
   including all later reruns. No additional merge is part of setup.
5. Verify the default branch, both workflow states, variables, release branch,
   and unchanged tag/release inventory. Retain the bootstrap SHA permanently;
   do not clear or advance it during ordinary releases.

The new workflow file gives the guarded pipeline a separate workflow identity.
Keep the inert `release.yml` file and its workflow identity disabled permanently.
Deleting the file changes its GitHub state to `deleted`, and GitHub rejects an
explicit disable request in that state. If recovering from such a cutover,
restore the inert definition, then disable the identity once GitHub registers it
as active. The placeholder has only manual dispatch and an always-skipped,
read-only job; it cannot build or publish. Keep its identity disabled to prevent
historical tags or dispatches from executing the old implementation. Never rerun
historical runs of that retired workflow. Rollback means setting
`COPPER_RELEASE_ENABLED=false`, disabling the new workflow and cancelling its
in-flight runs; do not re-enable the old main-based release workflow.

GitHub documents [workflow disabling](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/disable-and-enable-workflows)
and [workflow-run triggers](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run).
The latter consumes the workflow definition on the default branch, which is why
the migration must reach `main` before release promotion is enabled.

Signing retains the product bundler's `auto` policy. Signed macOS artifacts
require `MACOS_SIGNING_IDENTITY`, `MACOS_CERTIFICATE`,
`MACOS_CERTIFICATE_PASSWORD`, `APPLE_NOTARIZATION_KEY`,
`APPLE_NOTARIZATION_KEY_ID`, and `APPLE_NOTARIZATION_ISSUER_ID`. Signed Windows
artifacts require `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`,
`AZURE_CLIENT_SECRET`, `ACCOUNT_NAME`, `CERT_PROFILE_NAME`, `ENDPOINT`,
`FILE_DIGEST`, `TIMESTAMP_DIGEST`, and `TIMESTAMP_SERVER`. When a complete set
is unavailable, the current `auto` policy produces an ad-hoc-signed macOS
bundle or unsigned Windows bundle; it does not fail a stable release.

The required desktop collaboration baseline includes `client`, `call`, `channel`, and `collab_ui` without `multiplayer-tools`. These packages provide application connectivity, calls, channels, and collaboration UI used by the Rust product even when its multiplayer workspace is not compiled.

## Multiplayer desktop profile validation

The independent `multiplayer_tools_standard` and `multiplayer_tools_multiplayer`
jobs run `./script/check-multiplayer-tools --full --profile standard` and the
corresponding `multiplayer` command. Both are required by `shared_validation`;
profile failure cannot be hidden by Copper's Standard product test result.
The harness checks each application configuration, the closed capability boundary,
unfiltered Workspace/Git/Sidebar/Agent UI owner suites, warning-denied
`./script/clippy`, package dry runs and Standard dependency denial. Local and
remote executions must be reported separately; job presence is not a passing run.

This validation does not enable `multiplayer-tools` in the shipped Rust product,
change its product-manifest features, or certify authenticated collaboration.
The existing Copper baseline test and release boundaries remain intact.

## Deferred suites

Deferred suites are not counted as passing when they do not run.

### Hosted Collab server

The `collab` package is the hosted collaboration server. Its `collab_tests` integration target requires the `test-support` feature and contains both portable tests using their intended default backend and tests that explicitly select PostgreSQL. The generated `hosted_collab_tests` workflow provides PostgreSQL 15 through `COLLAB_TEST_DATABASE_URL` without globally overriding the backend selected by each test, and runs manually or whenever hosted collaboration paths change:

```sh
cargo nextest run --package collab --features test-support --test collab_tests \
  --no-fail-fast --no-tests=warn
```

This workflow becomes a required product gate if Copper enables `multiplayer-tools`, operates its own Collab service, or otherwise makes hosted collaboration part of the shipped product. It remains a separate PostgreSQL-backed gate rather than being mixed into lightweight desktop product tests.

### Sim runtime and evidence

The Linux, macOS, and Windows Sim backend warning-denied Clippy matrix remains required. Sim runtime and evidence tests are deferred until Sim becomes a shipped product feature and CI has the pinned source/evidence fixtures plus the filesystem and runtime resources those suites certify. Restore them as a dedicated job with:

```sh
cargo nextest run --no-fail-fast --no-tests=warn \
  -p sim_api -p sim_media -p sim_model -p sim_nodes \
  -p sim_plugin_host -p sim_plugin_sdk -p sim_plugin_worker_fixture \
  -p sim_runtime -p sim_sampler -p sim_tensor -p sim_test_support \
  -p sim_types -p sim_ui -p sim_worker
```

The dedicated job must provision the repository-pinned Sim source/evidence fixture trees, filesystem space, and any certified runtime resources before this command becomes required.
