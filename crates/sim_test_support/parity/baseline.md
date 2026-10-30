# Comfy parity source baseline

## Purpose

This baseline fixes the source snapshots, evidence vocabulary, and discovery
limits used by the Comfy-to-Zed parity specification. The source directories
contain no nested Git metadata. Package versions and content-sensitive tree
fingerprints are used throughout; a commit SHA is asserted only when the user
supplied and approved that exact upstream identity independently of the clean
source-distribution tree.

## Source snapshots

| Product | Snapshot path | Declared version | Git metadata | Source files observed | Content fingerprint |
| --- | --- | --- | --- | ---: | --- |
| ComfyUI | `projects/comfy/ComfyUI` | `0.27.1` in `pyproject.toml` and `comfyui_version.py` | unavailable | 949 | `21de8fece20d8d5bfa94daaa52d6ccfe2db6726ca0803ca3b383ad164cbd1d5f` |
| ComfyUI Frontend | `projects/comfy/ComfyUI-Frontend` | `1.48.2` in `package.json` | unavailable | 4,697 | `aeb208b759effdacf2ea3b1929f0a3e583201f0b7b3cb006f36f1007364b8ca3` |
| Comfy Desktop | `projects/comfy/Comfy-Desktop` | `1.0.28` in `package.json` | unavailable | 735 | `2442854931f3a5a80e68aa55eab21a26dcefe868b4e875251a5b4d811668e448` |
| comfy-cli | `projects/comfy/comfy-cli` | `0.0.0` in `pyproject.toml`, explicitly a CI placeholder and not a release identity | unavailable | 312 | `09d0b5f262bce3105f83777a310f1e391c4624f95142da5e3230626b68a276e6` |
| Comfy documentation | `projects/comfy/docs` | no project version declared; locked tooling versions are recorded below | unavailable | 5,800 | `1f4c9c460b8f5b35e30eb4d2d64bc201a958f247ab21af6c68743cce28c33931` |
| Comfy embedded docs | `projects/comfy/embedded-docs` | `0.5.7` in `pyproject.toml`; ComfyUI pins `comfyui-embedded-docs==0.5.6` | unavailable | 10,298 | `5aebf925cf36fe7b8df3c89466ad96ffa42110542a392ec6156b88fc807ec956` |
| Spandrel | `projects/comfy/Spandrel` | `0.4.2`; user-approved tag `v0.4.2`, commit `724cca389f28c38e1050689d4862a452fd644484` | absent from the clean sdist tree; exact upstream identity supplied by the user | 180 | `e1870c42b314fddb290f4d5322a03743076d98d0c6d288fc73691e3013994bbb` |
| spandrel-extra-arches | `projects/comfy/spandrel-extra-arches` | `0.2.0`; user-approved tag `v0.4.0`, commit `a1db3f5debbeeacbe02fb4114c69feee56ba5e21` | absent from the clean sdist tree; exact upstream identity supplied by the user | 52 | `7c0915d2e0df7db2131117087744fa5e73954dcad72aa785386d6bf8c1efb3aa` |
| Zed target | repository root | `1.10.2` in `crates/zed/Cargo.toml` | unavailable | 3,310 fingerprint inputs, including 2,839 files under `crates/` | `99ceb40a1cc3359cde6e0865fe1b6138a06317d5fbd892f1595de10a96b07e9a` |

The source-file coverage catalogs contain the reconciled production, test,
generated, documentation, asset, and infrastructure counts. The counts above
are filesystem orientation counts and are not registry totals.

### Fingerprint algorithm

For each Comfy product, the command ran from that product's source root. Files
were sorted by relative path and each file's SHA-256 record was hashed again as
one stream:

```sh
find . -type f \
  -not -path './node_modules/*' \
  -not -path '*/__pycache__/*' \
  -not -name '*.pyc' \
  -not -name '.DS_Store' -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 shasum -a 256 \
  | shasum -a 256
```

For Desktop, `-not -path './out/*' -not -path './dist/*'` is also applied.
Those output directories were absent. Runtime-generated Python bytecode caches
and operating-system `.DS_Store` files are excluded because they are
interpreter/host artifacts rather than snapshot source. Probe-generated caches
and reports were removed after observation, and source fingerprints were
reverified. Every excluded path is named in the applicable reconciliation
report. No checked-in source is excluded: hidden, generated, translated, test,
documentation, fixture, and build-support files are part of the source
fingerprint and source-file ledger.

The Zed fingerprint is reproducible from the repository root with the exact
manifest and path spelling below. Specification files are outside the manifest;
`target/` build output is excluded:

```sh
find crates assets .rules Cargo.toml Cargo.lock -type f \
  -not -path '*/target/*' -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 shasum -a 256 \
  | shasum -a 256
```

The command has 3,310 inputs: 2,839 under `crates/` and 471 across `assets/`,
`.rules`, `Cargo.toml`, and `Cargo.lock`. Generating this pack therefore does
not change the target baseline.

## Baseline evidence

| Fact | Evidence |
| --- | --- |
| ComfyUI package and runtime version are 0.27.1 | `projects/comfy/ComfyUI/pyproject.toml:3`; `projects/comfy/ComfyUI/comfyui_version.py:3` |
| ComfyUI requires Python 3.10 or newer | `projects/comfy/ComfyUI/pyproject.toml:6` |
| Frontend package version is 1.48.2 | `projects/comfy/ComfyUI-Frontend/package.json:3` |
| Desktop package version is 1.0.28 and requires Node 22 or newer | `projects/comfy/Comfy-Desktop/package.json:3`; `projects/comfy/Comfy-Desktop/package.json:15` |
| comfy-cli declares Python 3.10+ and the package version is only a `0.0.0` CI placeholder | `projects/comfy/comfy-cli/pyproject.toml`; [evidence-comfy-cli.md](evidence-comfy-cli.md) |
| docs declares no project release version; locked tooling includes Mint 4.2.585, Sharp 0.33.5, and Playwright MCP 1.0.12 | `projects/comfy/docs/package.json`; `projects/comfy/docs/pnpm-lock.yaml`; [evidence-documentation.md](evidence-documentation.md) |
| embedded-docs declares 0.5.7 while this ComfyUI snapshot pins 0.5.6 | `projects/comfy/embedded-docs/pyproject.toml`; ComfyUI dependency declaration; `catalogs/docs-reconciliation.json` |
| Spandrel 0.4.2 came from the official `spandrel-0.4.2.tar.gz` source distribution whose published and observed SHA-256 is `fefa4ea966c6a5b7721dcf24f3e2062a5a96a395c8bedcb570fb55971fdcbccb` | explicit user authority; official PyPI release metadata; `catalogs/spandrel-image-model-contract.json` |
| spandrel-extra-arches 0.2.0 came from the official `spandrel_extra_arches-0.2.0.tar.gz` source distribution whose published and observed SHA-256 is `9216877ecabc9c97e001ad5d49c4f8d2b1f6c6f82d1e77c8e2b350c586b6e64a` | explicit user authority; official PyPI release metadata; `catalogs/spandrel-image-model-contract.json` |
| Spandrel is a development oracle only; production Rust never imports or executes it, no model weights are approved, and model-use rights are evaluated independently | explicit user disposition; `catalogs/spandrel-image-model-contract.json` |
| Extra architectures are reference-only by default; restrictive, copyleft-incompatible, non-commercial, ambiguous, or unverified rows fail closed | explicit user disposition; `catalogs/spandrel-image-model-contract.json` |
| Both sdists omit the per-architecture license artifacts that their README says are included under each `__arch/` directory, so all 52 observed architecture rows remain rejected rather than receiving agent-authored legal approval | deterministic source audit; `catalogs/spandrel-image-model-contract.json` |
| Zed package version is 1.10.2 | `crates/zed/Cargo.toml:5` |
| No Comfy-specific target implementation exists | repository-wide search and direct architecture inspection outside `projects/comfy/**` and this pack found no Comfy workflow, node, tensor, model, sampler, worker, API-host, plugin, or GPUI implementation |

## Platform and distribution baseline

| Product | Availability represented in the snapshot | Notes |
| --- | --- | --- |
| ComfyUI | Windows, macOS, Linux; local and remotely reachable server modes; NVIDIA CUDA, CPU, DirectML, Intel oneAPI, Apple MPS, and conditional accelerator backends | Hardware and dtype support is capability- and dependency-dependent. Exact modes are cataloged rather than generalized from platform names. |
| Frontend | Browser/local server, desktop distribution, remote server, cloud distribution, App Mode, and developer/test distributions | Cloud, paid, experimental, disabled, and developer surfaces remain inventoried even when Zed parity is deferred. |
| Desktop | Windows, macOS, and Linux packages with platform branches and platform-specific installation/update behavior | Native behavior is cataloged per IPC, menu, window event, setting, and platform branch. |
| comfy-cli | Python 3.10+ terminal client; local/project/registry/cloud/partner lifecycle, event, schema, configuration, and extension surfaces | Source Python lifecycle and custom-node/frontend override execution are architecture conflicts; observable automation maps to native `zed comfy` or an explicit migration/defer response. |
| docs | Documentation site, English source content, localized mirrors, Cloud OpenAPI, redirects, tooling, generated node pages, staging, and developer workflows | Documentation is not executable evidence without code/test corroboration. English is the source of truth under the nested repository instructions. |
| embedded-docs | Python package of 855 node records across 12 locales with bundled assets and source fingerprints | Documentation availability and version skew are inventoried separately from node execution support. |
| Zed | Windows, macOS, Linux, and FreeBSD branches are present in the target; GPUI accessibility is currently opt-in through `ZED_EXPERIMENTAL_A11Y=1` | A Comfy capability receives an `equivalent` status only when its observable contract is already present, not merely because a generic Zed primitive exists. |

## Repository instructions applied

- Root `AGENTS.md`/`.rules` applies to the target pack and requires, among
  other rules, `./script/clippy` instead of `cargo clippy`, explicit async error
  propagation, GPUI task lifetime discipline, no `mod.rs` for new files, and
  non-conflicting task writes.
- `projects/comfy/Comfy-Desktop/AGENTS.md` was read in full and applies to the
  Desktop source/test interpretation. No Desktop source was modified.
- The Frontend root and every discovered scoped instruction were read in full:
  `projects/comfy/ComfyUI-Frontend/AGENTS.md`, `.github/AGENTS.md`,
  `.storybook/AGENTS.md`,
  `apps/website/src/pages/cloud/supported-nodes/AGENTS.md`,
  `browser_tests/AGENTS.md`,
  `browser_tests/tests/propertiesPanel/AGENTS.md`, `src/AGENTS.md`,
  `src/components/AGENTS.md`, `src/components/ui/AGENTS.md`,
  `src/lib/litegraph/AGENTS.md`, and `src/types/AGENTS.md`. Their scoped
  security, serialization, branded-ID, browser-test, component, Storybook,
  Cloud-catalog, and extension-compatibility rules informed the evidence and
  validation ledgers; no Frontend source was modified.
- `projects/comfy/docs/AGENTS.md` was read in full. It makes English the source
  of truth and defines documentation/localization conventions. The docs source
  was not modified.
- `projects/comfy/embedded-docs/.cursorrules` was read as repository guidance;
  the embedded source was not modified.
- No nested `AGENTS.md` or other applicable repository instruction was found in
  `projects/comfy/comfy-cli`. No nested Git metadata exists in any of the three
  added sources.

The documentation instructions govern source interpretation, not evidence
promotion. A page remains `documented-only` unless executable code, a focused
test, or a recorded runtime observation corroborates its behavioral claim.

## Discovery method

The inventory reconciles the following evidence classes rather than treating a
README or visible screen as authoritative:

Checked-in extractors exist for Desktop, comfy-cli, docs/embedded-docs,
tensor/autograd/RNG, Frontend component supplements, Desktop renderer
supplements, Zed evidence, native planning, and the master/trace artifacts.
The base ComfyUI and Frontend catalogs do not have checked-in extractors in
this pack. They are therefore classified as checksum-locked source snapshot
inputs, not falsely described as generated outputs. Their source-only fields
are pinned by `catalogs/source-snapshot-manifest.json`; target-only columns are
synchronized by the master generator. `regenerate_all.py` defines the canonical
order and two-pass freshness check. A future source-baseline refresh must rerun
the recorded extraction method, reconcile registries/tests/source files, review
the diff, and explicitly refresh the snapshot manifest.

1. Entrypoints, initialization order, package manifests, and repository file
   manifests.
2. Runtime and generated registries, including node mappings, object-info
   schemas, routes, WebSocket event producers/consumers, Electron IPC and
   preload bridges, commands, menus, keybindings, settings, flags, and
   migrations.
3. Executable code paths for validation, state transitions, persistence,
   cancellation, retries, timeouts, recovery, permissions, and side effects.
4. Unit, component, integration, browser, visual, fixture, Storybook, snapshot,
   and migration tests.
5. Documentation and changelogs as supporting evidence, with
   `documented-only` used when no executable or test evidence was found.
6. Safe local runtime probes without accounts, credentials, paid services,
   model downloads, dependency installation, or external mutation.
7. A source-file ledger in which every file is mapped to feature IDs or given
   an explicit infrastructure, test-only, generated, deprecated/dead, asset,
   documentation, or out-of-scope classification and reason.
8. Forward and reverse traceability plus an independent orphan search after
   the primary inventory is assembled.

## Evidence levels

Evidence levels are ordered by directness, not by whether the behavior is
desirable:

| Level | Meaning |
| --- | --- |
| `observed` | Directly confirmed by a runtime probe recorded in this pack. |
| `test-backed` | An existing automated test explicitly demonstrates the behavior. The catalog identifies the test; it does not imply that the test ran successfully in this environment unless a run is separately recorded. |
| `code-inferred` | Executable production code supports the behavior, but it was not dynamically confirmed and no focused existing test was found. |
| `documented-only` | The capability appears only in documentation, examples, changelogs, or comments after orphan reconciliation. |
| `unverified` | Evidence is insufficient, contradictory, dynamically generated beyond available dependencies, or requires an unavailable account/service/platform. |

When more than one level applies, the highest direct evidence is recorded and
the other sources remain in the evidence fields.

## Availability classes

| Class | Meaning |
| --- | --- |
| `active` | Available in the normal supported product path. |
| `conditional` | Requires a setting, dependency, server capability, model, extension, entitlement, or runtime condition. |
| `platform-specific` | Implemented only on identified operating systems, hardware, or package variants. |
| `experimental` | Marked experimental, beta, prototype, or gated for evaluation. |
| `developer-only` | Intended for development, debugging, tests, CI, diagnostics, or internal operation. |
| `cloud/paid` | Requires a Comfy cloud deployment, account, billing, credits, or paid/external service. |
| `deprecated/dead` | Deprecated, legacy-only, unreachable, disabled without a supported path, or retained solely for migration. |
| `infrastructure-only` | Enables other features but is not itself a user workflow. |
| `uncertain` | Availability cannot be established from the snapshot. |

Availability values are not mutually exclusive in the machine-readable
catalogs; a platform-specific experimental feature may carry both labels.

## Confidence

| Value | Rule |
| --- | --- |
| `high` | Registry/route/schema is explicit, or production behavior is corroborated by a focused test or runtime observation. |
| `medium` | Production code is clear but depends on unavailable runtime state, platform, generated data, or external service. |
| `low` | Documentation conflicts with code, reachability is unclear, or a dynamic/external registry could not be materialized. |

## Runtime probes and constraints

| Probe | Result | Evidence consequence |
| --- | --- | --- |
| `python3 projects/comfy/ComfyUI/main.py --help` | Exit 0; the parser's CLI help printed | CLI argument presence and help text are `observed`. |
| `python3 projects/comfy/ComfyUI/main.py --list-feature-flags` | Exit 0; `show_signin_button=false` and `enable_telemetry=false` registry printed | Those server feature-flag defaults are `observed`. |
| Python environment check | Python 3.9.6; project requires 3.10+; `torch`, `aiohttp`, `safetensors`, `yaml`, `PIL`, `numpy`, `sqlalchemy`, and `alembic` absent | Server launch, model execution, route exercise, database migrations, and backend tests were not runnable without violating the no-dependency-change constraint. |
| Frontend dependency check | No `node_modules`; lockfile present | Unit/browser/Storybook execution was not runnable without installing dependencies. Existing focused tests still provide `test-backed` evidence. |
| Desktop dependency check | No `node_modules`; lockfile present | Electron, IPC, platform E2E, and desktop unit tests were not runnable without installing dependencies. |
| `PYTHONDONTWRITEBYTECODE=1 HOME=/tmp/comfy-cli-audit-home PYTHONPATH=projects/comfy/comfy-cli python3 -m comfy_cli --help-json` | Exit 1 before command construction: host Python 3.9.6 is below the declared 3.10 minimum and `questionary` is unavailable | No comfy-cli feature is `observed`; 2,295 focused test functions may support `test-backed` rows but zero tests ran. No dependency or network mutation occurred. |
| comfy-cli static/parser audit | All 228 Python files AST-parse; 30/31 JSON files strict-parse, with `pyrightconfig.json` retained as JSONC/trailing-comma input | Registrations, commands, flags, schemas, errors, events, configuration, formats, lifecycle, CQL, OpenAPI mappings, tests, and source coverage are code/test evidence, not runtime observation. |
| docs local link checker | 4,988 documentation files checked; validator passed | The validator result is `observed` documentation-tooling behavior only; linked product claims keep their own evidence level. |
| docs Bun tests | 8/8 passed | The exact tooling/tests are observed; prose is not promoted. |
| docs translation scan | 51 issues reported; generated reports removed and fingerprint restored | Translation/localization uncertainty remains explicit in `catalogs/docs-reconciliation.json`. |
| embedded-docs link checker | Passed | Embedded link integrity is observed; node execution remains separately corroborated or documented-only. |
| External services | No accounts, credentials, paid services, model downloads, or mutating requests used | Cloud/paid and authenticated behavior is code/test/documentation evidence or explicit uncertainty. |

No source application was launched in a mode that could download models,
update software, create an account, alter a remote server, or modify the source
snapshot. No source or target dependency was added.

ComfyUI runtime observation is a development-time evidence activity only. The
production target may not use those commands, source trees, Python packages,
or a Comfy endpoint. Native release validation runs with the network disabled,
no Python executable on `PATH`, and source directories absent.

## Zed status vocabulary

| Status | Rule |
| --- | --- |
| `equivalent` | The current Zed behavior matches the externally observable Comfy contract and has target evidence. |
| `partial` | Some observable behavior matches, but at least one required state, format, interaction, side effect, error, recovery, or compatibility contract is absent. |
| `missing` | No Comfy-specific observable capability exists in Zed. Generic GPUI/workspace infrastructure alone does not change this status. |
| `conflicting` | Current Zed behavior or format is incompatible with the Comfy contract and requires a compatibility boundary or migration. |
| `deferred` | The parity decision deliberately schedules the capability after prerequisite work or a product/legal decision; the source feature remains traced. |
| `uncertain` | Target support or source behavior cannot be established with the available evidence. |

## Known baseline uncertainties

- Dynamic custom-node and frontend-extension registries can add capabilities at
  runtime. The pack specifies compatibility contracts and uses fixture
  extensions, but cannot enumerate code not present in the snapshots.
- Cloud APIs, billing, entitlements, surveys, remote tasks, and some feature
  flags depend on server-side behavior not present locally. Their client-side
  contracts are inventoried; server behavior remains explicit uncertainty.
- Hardware/model execution could not be observed on supported accelerators in
  this environment. Code and focused tests define the planned contract suite.
- Platform-native desktop behavior could not be executed on Windows or Linux
  from the macOS host. Platform branches and tests are evidence, not runtime
  confirmation.
- Generated registries whose generation requires absent dependencies are
  reconciled from source registrations and checked-in fixtures. Any unresolved
  delta is retained as `unverified`, not inferred away.
- comfy-cli has a shadowed hidden `models` function, an orphan `comfy version`
  schema mapping, prose-only `comfy query`, four event-union discrepancies, a
  lockfile spelling conflict, and one documentation-only cloud/paid Keyframe
  Relay claim. These remain explicit rather than normalized away.
- Docs has path-case/navigation/localization deltas, three Cloud OpenAPI
  operations without executable route-shape corroboration, and 38
  provider-unverified embedded node claims. Embedded docs declares 0.5.7 while
  ComfyUI pins 0.5.6.
- The native compute backend ecosystem, vendor SDK/driver FFI, codec libraries,
  distribution size/licensing, and device lab availability remain architecture
  and implementation uncertainties. They do not relax the native-only
  production boundary or permit a Python fallback.
