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

### Foreground budget isolation

Copper's serial measurement step and local Nextest slot reservation use the same
filter. Test names containing `foreground_budget` are included automatically.
The filter also includes the flamegraph paint, deep-selection, denied-batch and
completed-import-disposal checks in `language_tools`, whose names do not contain
that marker. Platform-specific tests run only where compiled. This covers six
Linux timing tests and four additional macOS profile-import/lifecycle tests.

The unoptimized test profile, Copper features, 250 ms limits and functional
assertions stay unchanged. The 100,000-range coverage test must still publish
immediately and update all 10,000 active-editor gutter markers and the live
summary. The 100,000-node flamegraph test must paint every flat frame while
virtualizing offscreen depths. No timing test is ignored or retried. CI runs the
ordinary suite using the complement of the measurement filter, then requires
matching timing tests to execute with one test thread.

### Product releases and website refresh

The generated `release_prepare.yml` shares one signed packaging pipeline
across these entry points.

- A human publishes a Copper prerelease for an existing catalog tag. Actions
  verifies eligibility and attaches the Apple Silicon package to that release ID.
- A maintainer dispatches from main with `release_tag`. Actions builds the
  selected platform scope and prepares a draft for human review and publication.
- A maintainer dispatches without a tag for non-publishing qualification.
  Signing-related pushes to main and `codex/**` run credential-free contracts.

The previous `release.yml` and `release_promotion.yml` placeholders are removed.
The signing contract and qualification jobs from `release_signing_check.yml`
now use `release_prepare.yml`. The collaboration candidate requires a public
compatibility contract before it can package and upload from GitHub Actions.
Maintainers can exercise the existing contract locally with
`script/collaboration-release-contract --dry-run` and the compatibility policy
under `.agents/local/collaboration/`.
Website refresh is paused, including manual recovery. See
[Product release operation](../../docs/src/development/product-releases.md)
for eligibility, setup, retries, and updater selection.

## Multiplayer desktop profile validation

The independent `multiplayer_tools_standard` and `multiplayer_tools_multiplayer`
jobs run `./script/check-multiplayer-tools --full --profile standard` and the
corresponding `multiplayer` command. Both are required by `shared_validation`;
profile failure cannot be hidden by Copper's Standard product test result.
The harness checks each application configuration, the closed capability boundary,
unfiltered Workspace/Git/Sidebar/Agent UI owner suites, warning-denied
`./script/clippy`, package dry runs and Standard dependency denial. Local and
remote executions must be reported separately; job presence is not a passing run.

With the private collaboration spec available locally, run
`python3 script/validate-spec.py .agents/local/collaboration/collaborative-workspace`.
Run `script/check-collaborative-workspace-inventory --catalog all` in a local
checkout that also has the Buzz sources. CI runs the inventory checker's
synthetic tests using tracked inputs.

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
