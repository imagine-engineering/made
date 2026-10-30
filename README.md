# Made

[![CI](https://github.com/simtropolis/made/actions/workflows/run_tests.yml/badge.svg)](https://github.com/simtropolis/made/actions/workflows/run_tests.yml)

**Made** stands for **Multiplayer Agentic Development Environment**. It is a
Rust workspace for building focused development environments from a shared,
Zed-derived editor platform.

The currently enabled product is **Copper**, a Rust-focused desktop environment
with native agent workflows, Rust and `rust-analyzer` defaults, Cargo tooling,
and the editor, terminal, Git, debugger, extension, remote-development, and
collaboration foundations inherited from Zed.

Made is under active development. This repository is currently source-first;
after main changes are promoted to `release` and required CI succeeds,
automation publishes Copper artifacts for the exact commit that passed the test
workflow.

## What is implemented

| Area | Current state |
| --- | --- |
| Copper (`rust`) | Enabled for Linux x86_64, macOS ARM64, and Windows x86_64 |
| Editor platform | GPU-accelerated GPUI editor with LSP, Tree-sitter, terminal, Git, debugger, remote development, and WASM extensions |
| Agentic development | Enabled in Copper through the native agent, ACP, skills, prompt, and language-model crates |
| Rust development | Built-in Rust support, `rust-analyzer`, Rust toolchain onboarding, a Cargo panel, Cargo actions, test exploration, and a Rust-focused agent profile |
| Multiplayer workspace | Implemented behind the opt-in `multiplayer-tools` feature; it is not part of the default Copper release profile |
| Native visual workflows | Sim provides native runtime, UI, plugin, model, tensor, worker, and backend crates for ComfyUI-compatible workflows exist behind opt-in features; they are not part of the current Copper release profile |
| Orbit (`jvm`) and Forge (`game`) | Catalogued as planned products and deliberately rejected by the build tooling |

Internal crate names and project-local `.zed` configuration paths remain
unchanged intentionally. Copper has separate application, data, URL-scheme,
updater, and remote-server identities defined by the product catalog.

## Repository layout

| Path | Purpose |
| --- | --- |
| `products/flavors.toml` | Product catalog and enabled/planned product definitions |
| `crates/zed` | Desktop application entry point and feature composition |
| `crates/product_flavor` | Generated compile-time product identity |
| `crates/gpui`, `crates/editor`, `crates/project`, `crates/workspace` | Core UI and editor platform |
| `crates/agent*`, `crates/acp_*`, `crates/language_model*` | Agent runtime, ACP integration, skills, prompts, and model providers |
| `crates/cargo_ui`, `crates/tasks_ui` | Copper's Cargo and Rust test workflows |
| `crates/collaboration_*`, `crates/nostr_compat`, `services/` | Multiplayer workflow, compatibility, relay, and gateway components |
| `crates/sim_*` | Opt-in native visual workflow runtime and accelerator backends |
| `tooling/xtask`, `script/`, `.github/workflows/` | Generation, validation, packaging, and CI automation |
| `docs/` | mdBook user and development documentation |
| `.agents/specs/` | Design and implementation records; these are not proof that unfinished features ship |

## Prerequisites

- Git.
- [Rustup](https://rustup.rs/). The repository pins Rust `1.98.0` in
  [`rust-toolchain.toml`](./rust-toolchain.toml); rustup selects it
  automatically.
- Platform build dependencies:
  - [macOS](./docs/src/development/macos.md): Xcode, Xcode command-line tools,
    and CMake.
  - [Linux](./docs/src/development/linux.md): run `./script/linux` to install
    the distribution-specific native dependencies.
  - [Windows](./docs/src/development/windows.md): MSVC C++ build tools,
    Spectre-mitigated libraries, a Windows SDK, and CMake.

The supported Copper bundle targets are `x86_64-unknown-linux-gnu`,
`aarch64-apple-darwin`, and `x86_64-pc-windows-msvc`.

## Build and run Copper

Clone the repository:

```sh
git clone https://github.com/simtropolis/made.git
cd made
```

On Linux, install the native dependencies before building:

```sh
./script/linux
```

Run the exact enabled Copper feature set:

```sh
cargo run -p zed --no-default-features --features agentic-tools,rust-tools
```

The `rust` product is the development default, so `ZED_PRODUCT_ID` does not
need to be set for this command. Cargo's internal package and binary remain
named `zed`; the running application's product identity is Copper.

To compile a release build without packaging it:

```sh
cargo build --release -p zed --no-default-features --features agentic-tools,rust-tools
```

To inspect the product-aware Linux bundle plan without compiling:

```sh
cargo xtask bundle --product rust \
  --platform linux \
  --target x86_64-unknown-linux-gnu \
  --dry-run
```

To build a bundle for the current host, use the product-aware entry point:

```sh
cargo xtask bundle --product rust
```

See [Product Flavors](./docs/src/development/product-flavors.md) for target,
signing, update, and identity details.

### Opt-in multiplayer build

The full Made multiplayer workspace is not enabled in the default Copper
release manifest. To run it explicitly:

```sh
cargo run -p zed --no-default-features --features multiplayer-tools,rust-tools
```

Use the repository's profile checks when changing this boundary:

```sh
script/check-multiplayer-tools --quick
```

See [Multiplayer Build Profiles](./docs/src/development/multiplayer-tools.md)
for the feature boundary and release-equivalent checks.

## Validation and development

Run these checks from the repository root:

```sh
# Check formatting without modifying files
cargo fmt --all -- --check

# Run the repository lint entry point
./script/clippy

# Verify generated product metadata
cargo xtask products --check

# Run a focused crate test suite
cargo test -p product_flavor
```

The required CI suite uses `cargo-nextest` with an explicit package and feature
matrix rather than treating every deferred or infrastructure-dependent test as
a passing product gate. See the [workflow scope](./.github/workflows/README.md)
and [`run_tests.yml`](./.github/workflows/run_tests.yml) for the current commands.

When changing product or workflow definitions, regenerate and verify their
checked-in outputs:

```sh
cargo xtask products
cargo xtask products --check
cargo xtask workflows
cargo xtask check-workflows
```

## Releases

Main remains the default development branch. Merge normal development PRs into
`main`, then promote changes with a `main` → `release` pull request. Main CI
does not publish. The initial release-branch commit is permanently excluded
from publication, including manual recovery and reruns.

Copper is currently the only enabled release product. A successful
[`run_tests`](./.github/workflows/run_tests.yml) push run on `release` starts the
generated [`release_promotion`](./.github/workflows/release_promotion.yml) workflow for that exact
tested commit. The workflow builds all three supported bundles before it
creates an annotated `rust-vX.Y.Z` tag and a GitHub Release named
`Copper X.Y.Z`; a failed or incomplete build publishes neither.

With no existing `rust-v*` tag, the first automatic release uses the version in
`crates/zed/Cargo.toml`. Later automatic releases increment the latest release's
patch version. To request a manual minor or major release from `release`:

```sh
gh workflow run release_promotion.yml --repo simtropolis/made --ref release -f bump=minor
gh workflow run release_promotion.yml --repo simtropolis/made --ref release -f bump=major
```

Manual recovery can select an exact stable version and tested promotion commit on `release`:

```sh
gh workflow run release_promotion.yml --repo simtropolis/made --ref release \
  -f bump=patch \
  -f version=<X.Y.Z> \
  -f commit_sha=<FULL_COMMIT_SHA>
```

Release operations are serialized. A rerun reuses an existing tag when it
already points to the selected commit and updates or completes the associated
release; it never moves a conflicting tag. Manually pushed `rust-v*` tags
remain supported when they identify a tested promotion commit on `release`
after the bootstrap boundary.

Release signing uses the bundle command's `auto` policy. Complete macOS or
Windows credential sets produce signed artifacts; otherwise macOS uses ad-hoc
signing and Windows remains unsigned rather than failing the release. The
required secret names are listed in
[Product Flavors](./docs/src/development/product-flavors.md#signing), and
release-operation details are in the
[workflow documentation](./.github/workflows/README.md#copper-releases).

Use `./script/clippy`, not `cargo clippy` directly. Add focused tests for the
crate you change and preserve the intentional internal Zed terminology. Read
[`AGENTS.md`](./AGENTS.md) and [`CONTRIBUTING.md`](./CONTRIBUTING.md) before
submitting a change.

### Release website

The Copper downloads and changelog page is served at [simtropolis.ai/releases](https://simtropolis.ai/releases) by the existing AWS Amplify website application. The [website repository](https://github.com/simtropolis/website) generates it from current GitHub release metadata during each build. Binary downloads remain in GitHub Releases. After successful publication, the `refresh_website` job requests an Amplify rebuild; see the [workflow guide](.github/workflows/README.md#website-release-refresh) for setup and verification.

## Configuration

No environment variables are required for a local Copper debug build.

| Variable | Required | Purpose |
| --- | --- | --- |
| `ZED_PRODUCT_ID` | No | Compile-time product selection. Defaults to `rust`; `jvm` and `game` currently fail because they are planned. |
| `ZED_RELEASE_CHANNEL` | No | Selects `dev`, `nightly`, `preview`, or `stable`. Defaults to `crates/zed/RELEASE_CHANNEL`, currently `stable`. |
| `ZED_PRODUCT_UPDATE_BASE_URL` | No | Enables the product updater for non-development builds. Without it, Copper's updater remains disabled. |
| `RELEASE_VERSION` | Packaging only | Sets the version used in product artifact names; local dry runs default to `dev`. |

Signing credentials are needed only for signed release bundles. Keep them in
the environment or CI secrets; the complete macOS and Windows variable lists
are documented in [Product Flavors](./docs/src/development/product-flavors.md#signing).

## Optional Sim workflows

Sim is Made's first-party native visual workflow functionality. Enable it explicitly
with `cargo run -p zed --no-default-features --features agentic-tools,rust-tools,sim`.
To include Sim and its worker in a product bundle, use
`cargo xtask bundle --product rust --sim`.

Settings read the persisted `comfy_runtime` key as well as `sim_runtime`; use
one key per settings file. Existing keybindings using `comfy_shell`/`comfy_graph`
actions and `ComfyGraph`/`ComfyTextInput` contexts continue to resolve. These are
saved user formats. Existing workflow and database formats and private storage
paths retain their identities. Native Rust APIs, build features, command-line
entry points, and owned files use Sim directly. ComfyUI, source catalogs, and
third-party product names keep their original names.

## Current limitations

- Copper is the only enabled product. Orbit and Forge are future catalog
  entries, not runnable applications.
- The default Copper release includes `agentic-tools` and `rust-tools`, but not
  `multiplayer-tools` or `sim`.
- The Sim runtime and backend crates are being developed and validated, but
  the runtime/evidence suite is not yet a Copper product gate.
- The application updater has no production service unless a build supplies
  `ZED_PRODUCT_UPDATE_BASE_URL`.
- Copper's marketing name and `dev.ideflavors.*` bundle identity are explicitly
  provisional.
- Some detailed documentation and contribution material still uses upstream
  Zed terminology. Treat source, manifests, and generated workflows as the
  authority when they disagree.

## Licensing

The source is primarily licensed under
[`GPL-3.0-or-later`](./LICENSE-GPL), with
[`Apache-2.0`](./LICENSE-APACHE) components where marked. Third-party license
metadata is validated with `cargo-about`; see the repository license files and
[`CONTRIBUTING.md`](./CONTRIBUTING.md) for contribution requirements.

Made is developed by **Simtropolis, Inc.**
