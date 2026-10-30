# Product Flavors

Product flavors build independently installable applications from the shared
Zed codebase. Product identity and packaging data live in
`products/flavors.toml`. Internal crate and module names remain unchanged.

## Phase 1 {#phase-1}

Phase 1 enables only the stable product ID `rust`. Its current display name is
provisional and can change in the catalog without changing the product ID,
bundle identifier, URL scheme, data directory, or updater namespace. The `jvm`
and `game` entries reserve future identities and cannot be bundled or released.
Groovy has a guarded local debug build for validation while `jvm` remains
planned.

The Rust product enables the `agentic-tools` and `rust-tools` application features,
the `rust-tools` remote-server feature, built-in Rust and rust-analyzer support,
Cargo commands, Rust toolchain onboarding, and a Rust-focused agent profile.

## Validate the catalog {#validate-the-catalog}

Generate the compile-time Copper and Groovy selections after editing the catalog:

```sh
cargo xtask products
cargo xtask products --check
```

The bundle validator rejects planned products, duplicate operating-system
identities, unsafe paths and filename templates, unsupported targets, and
inconsistent Rust feature mappings before compilation starts.

## Validate planned Groovy locally {#validate-planned-groovy}

Build a development-only Groovy candidate with explicit identity and feature
selection:

```sh
ZED_PRODUCT_ID=jvm ZED_PRODUCT_DEV_PREVIEW=1 ZED_RELEASE_CHANNEL=dev \
  cargo build -p zed --no-default-features \
  --features agentic-tools,jvm-tools,runtime-shaders
```

The candidate uses Groovy's catalog name, defaults, and isolated development
namespace. The opt-in works only for Cargo's debug profile and the `dev`
channel. It does not change `jvm.status`, add CI or release rows, or allow a
bundle: `cargo xtask bundle --product jvm --dry-run` still fails. Normal,
stable-channel, and release-profile attempts to compile the planned product
also fail. This binary supports local workflow validation; it is not a release
package or proof that Java/Kotlin extensions work on every platform.

## Product appearance {#product-appearance}

The Rust product defaults to Fractal Neutral Light and Dark, following your
system appearance, with the bundled Figtree UI font. Your explicit theme, font,
icon-theme and density preferences take precedence. Editor and terminal fonts
remain monospaced.

Each enabled product selects an `appearance` table in `products/flavors.toml`:

```toml
appearance = { theme = "assets/themes/fractal-neutral/fractal-neutral.json", light = "Fractal Neutral Light", dark = "Fractal Neutral Dark", font_family = "Figtree", font_asset = "assets/fonts/figtree/Figtree.ttf" }
```

To select another theme for a future product, bundle its theme JSON under
`assets/themes/` and its TrueType font under `assets/fonts/`, then set these
fields to the corresponding asset paths, variant names and font family. Both
variant names must exist in the theme file with the matching appearance. Run
`cargo xtask products` followed by `cargo xtask products --check` to regenerate
and validate the defaults. Planned products remain disabled.

The shared controls retain Fractal geometry across product themes. Theme colours
use the existing semantic theme fields, so individual controls do not need
product-specific branches. The default-settings view resolves product defaults
from the generated catalog values.

The pinned Fractal source, licences, token mapping and reuse decisions are
recorded in `assets/design_systems/README.md`. Regenerate the Neutral palette
with `python3 script/fractal-theme`; verify it with
`python3 script/fractal-theme --check`.

Copper's current application artwork is the supplied copper rounded tile with a
white spiky mascot. The unchanged original and platform exports live under
`assets/product_flavors/rust/`; `script/generate-copper-icons.py` reproduces them
with Pillow 12.3.0. Run it with `--check` to verify exports. Application icons
retain the full tile with transparency only outside its rounded edge; `logo.*`
and `logos/` contain the separate transparent mascot, including its eyes and
reflections. The asset-family README documents exact hashes, padding and
small-size sampling. Earlier gear/letter artwork remains historical evidence,
not the active product identity.

## Build a product bundle {#build-a-product-bundle}

Use the product-aware entry point rather than invoking a platform script
directly:

```sh
cargo xtask bundle --product rust
```

You can inspect a platform plan without compiling:

```sh
cargo xtask bundle --product rust \
  --platform linux \
  --target x86_64-unknown-linux-gnu \
  --dry-run
```

Supported Phase 1 targets are Linux x86_64, macOS ARM64, and Windows x86_64.
Each plan uses `--no-default-features`, exact catalog feature lists, and a
product-specific Cargo target directory.

## Signing {#signing}

Signing defaults to `auto`. A bundle remains unsigned or ad-hoc signed when the
complete platform credential set is unavailable. Use `--signing required` to
fail before packaging execution when credentials are incomplete (dry runs only
resolve the plan), or `--signing off` to
force an unsigned development bundle.

Signed macOS releases use these environment variables:

- `MACOS_SIGNING_IDENTITY`
- `MACOS_CERTIFICATE`
- `MACOS_CERTIFICATE_PASSWORD`
- `APPLE_NOTARIZATION_KEY`
- `APPLE_NOTARIZATION_KEY_ID`
- `APPLE_NOTARIZATION_ISSUER_ID`

Signed Windows releases use these environment variables:

- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`
- `ACCOUNT_NAME`
- `CERT_PROFILE_NAME`
- `ENDPOINT`
- `WINDOWS_SIGNING_PUBLISHER`
- `FILE_DIGEST`
- `TIMESTAMP_DIGEST`
- `TIMESTAMP_SERVER`

Production Linux tar archives require `LINUX_SIGNING_KEY`,
`LINUX_SIGNING_KEY_PASSWORD`, and `LINUX_SIGNING_KEY_FINGERPRINT` for detached GPG
archive signing. This is separate from OS application signing and DEB/RPM signing.

The guarded release jobs explicitly require signing. The separate
`release_signing_check.yml` workflow reuses them without a publisher job. See
[production signing operations](./copper-signing.md) for exact jobs, secrets,
verification, provisioning, renewal, failure recovery and qualification limits.

Credential values remain in the environment and GitHub secrets. Do not add
them to the product catalog.

## Identity and migration limits {#identity-and-migration-limits}

The provisional `dev.ideflavors` bundle identifier root must be replaced with a
publisher-owned reverse-DNS root before a production identity is finalized.
The application updater stays disabled unless `ZED_PRODUCT_UPDATE_BASE_URL` is
set at build time because no production product update service is configured.
GitHub Releases still receive the platform artifacts generated by the release
workflow.

The Rust product uses a new data namespace and does not automatically import
from Zed directories. Phase 1 deliberately provides no in-application importer:
an automatic import cannot reliably distinguish upstream Zed data from data
written by an older fork. Copy settings manually only after selecting and
confirming the source, and leave the source installation intact.

## Copper release version

Copper's first release target is `0.1.0`. `products/flavors.toml` owns this as
`initial_version`; upstream Cargo package and dependency versions are independent.
`cargo xtask products` generates the runtime fallback used by About, CLI version
output and the updater. An explicit `RELEASE_VERSION` embeds the selected later
release in the app and platform metadata. `cargo xtask bundle --product rust`
defaults to the catalog version, including package filenames and installer metadata.
The release resolver uses it only before the first `rust-v*` tag; later releases
follow Copper's tag history. Only successful same-repository push CI on `release`,
or a guarded manual dispatch on `release`, can start publication. `main` and tag
pushes cannot publish. See [production signing](copper-signing.md).
