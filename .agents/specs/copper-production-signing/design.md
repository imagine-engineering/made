# Design: Copper production signing

## Overview

Use the generated release workflow and existing product packaging scripts as the owners. Compile phases remain credential-free; package phases explicitly require production signing. A separate manual workflow reuses the same jobs and has no publication job.

## Existing context

`tooling/xtask/src/tasks/workflows/release.rs` generates `.github/workflows/release_promotion.yml`. `script/bundle-mac` uses Developer ID and Apple's notary service; `script/bundle-windows.ps1` uses Azure Trusted Signing via `crates/zed/resources/windows/sign.ps1`. Linux distributes a tar archive, with no existing DEB/RPM signing owner. Managed llama.cpp artifacts are pinned in `crates/llama_cpp/managed_catalog.json`.

## Design decisions

### Packaging and verification

Require the complete existing credential set before work. Use private temporary macOS stores and explicit keychain selection. Sign nested code inside out, preserving helper-specific entitlements. Notarize the application archive before stapling it, then construct, sign, notarize and staple the final DMG. Verify Developer ID, Team, entitlements, nested code, timestamps and Gatekeeper. Windows signs all staged PE payloads and installer/uninstaller through the existing service, then verifies Authenticode and timestamps. Its existing updater moves and rolls back the packaged runtime directory with the application. Actual installed/mounted artifacts repeat platform trust checks before launch. Linux signs the tar archive with a publisher GPG key in a private temporary home. The exact release asset set expands from three payloads to those same payloads plus the detached Linux signature and public key; all publication eligibility gates remain intact.

### Managed runtime

Package reviewed runtime archives from the existing catalog, preserving original digests in receipts before platform signing. The native manager resolves bundled runtimes first and rejects incomplete bundled installations instead of falling back to downloaded unsigned production code. Keep manually configured server behavior unchanged.

### CI and safety

Generate `release_signing_check.yml` with manual dispatch only, an explicit exact version (default 1.0.0), read-only contents permissions and the same build/package owners. It defaults to contract tests and bundle plans without secrets or signing-service calls. Only the explicit boolean `sign_artifacts=true` enables its production preflight and dependent package jobs; the authorized release workflow always requires signing. Package and qualification preparation jobs pin Python 3.12 for safe archive extraction. No tag/release/deployment operation belongs in that graph. Preserve both promotion guards. A normal initial source commit is explicitly authorized on a non-release branch. Hosted signing still requires approved production identities and credentials; source import never enables release gates or dispatches signing.

## Requirements traceability

| Requirement | Design element | Verification |
| --- | --- | --- |
| 1.1 | macOS owner | Negative preflight, nested signature and stapling checks; live signing when credentials exist |
| 1.2 | Windows owner | Policy tests; actual Windows CI required |
| 1.3 | Linux archive signing | Workflow asset contract and GPG verification |
| 1.4 | Runtime packaging/manager | Pinned extraction and bundled resolution tests |
| 2.1 | Required policy | Missing/partial/disabled credential cases and development checks |
| 2.2 | Temporary stores | Cleanup failure tests and trace audit |
| 2.3 | Generator | Existing promotion/version tests and generated workflow comparison |
| 2.4 | Validation workflow | Graph tests and manual hosted run when available |
| 2.5 | Operator documentation | Secret inventory and evidence report |

## Testing strategy

Run focused Rust and script tests, generated workflow checks, formatting and `./script/clippy` for changed Rust owners. Validate existing ad-hoc packages are rejected by production checks. Preserve explicit blockers for missing credentials, hosted CI, non-host platforms, and signed native-agent execution.

<!-- impl: tooling/xtask/src/tasks/workflows/release.rs#product_builds -->
<!-- impl: script/bundle-managed-runtime#stage -->
<!-- impl: crates/llama_cpp/src/managed.rs#bundled_runtime -->
