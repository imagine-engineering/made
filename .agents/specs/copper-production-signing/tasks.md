# Implementation Plan: Copper production signing

## Approach

Harden packaging owners, include reviewed runtime payloads, then reuse those owners in guarded release and non-publishing workflows. Live trust qualification remains separate from local contract checks.

## Tasks

All leaves are implemented locally. Partial state records the pending actual production identities, hosted CI and signed-platform qualification. Evidence: `artifacts/copper-production-signing-20261030/REPORT.md`. The user explicitly authorized a normal initial source commit and push to the empty repository on a non-release branch. No release, tag, promotion or live signing dispatch is authorized by that import.

### Milestone 1: Reviewable signing and qualification

- [~] 1. Production packaging
  - [~] 1.1. Require and verify macOS production signing
    - _Requirements: 1.1, 2.1, 2.2_
    - _Depends on: none_
    - _Reads: script/bundle-mac, script/lib/macos-signing.sh_
    - _Writes: script/bundle-mac, script/lib/macos-signing.sh, script/verify-macos-signature_
    - _Validation: bash syntax and signing lifecycle tests_
- [~] 2. Windows trust
  - [~] 2.1. Require and verify Windows production signing
    - _Requirements: 1.2, 2.1_
    - _Depends on: none_
    - _Reads: script/bundle-windows.ps1, crates/zed/resources/windows/sign.ps1_
    - _Writes: script/bundle-windows.ps1, crates/zed/resources/windows/sign.ps1, crates/zed/resources/windows/zed.iss, crates/auto_update_helper/src/updater.rs_
    - _Validation: shared signing-policy tests pass locally; Windows Authenticode, installer and updater execution require hosted CI_
- [~] 3. Managed inference
  - [~] 3.1. Bundle the pinned native runtime and resolve it from the package
    - _Requirements: 1.4_
    - _Depends on: 1.1, 2.1_
    - _Reads: script/bundle-managed-runtime, crates/llama_cpp/src/managed.rs_
    - _Writes: script/bundle-managed-runtime, crates/llama_cpp/src/managed.rs, crates/llama_cpp/runtime-LICENSE_
    - _Validation: real pinned macOS archive staging, 23 native-manager tests, debug and installed development-package bundled-runtime native-agent workflows; production packaged trust remains blocked_
- [~] 4. CI qualification
  - [~] 4.2. Sign Linux archives through the existing package and release asset owners
    - _Requirements: 1.3_
    - _Depends on: none_
    - _Reads: script/bundle-linux, tooling/xtask/src/tasks/workflows/release.rs_
    - _Writes: script/sign-linux-archive, script/bundle-linux_
    - _Validation: real ephemeral GPG key signs/verifies/rejects tampering in a network-disabled Linux container; production key and hosted Linux execution remain blocked_
  - [~] 4.1. Enforce production jobs and generate a non-publishing workflow
    - _Requirements: 2.3, 2.4_
    - _Depends on: 3.1_
    - _Reads: tooling/xtask/src/tasks/workflows/release.rs, tooling/xtask/src/tasks/workflows.rs, .github/workflows/release_promotion.yml, .github/workflows/release_signing_check.yml_
    - _Writes: tooling/xtask/src/tasks/workflows/release.rs, tooling/xtask/src/tasks/workflows.rs, .github/workflows/release_promotion.yml, .github/workflows/release_signing_check.yml, script/smoke-product-bundle, script/check-signing-environment, script/test-production-signing, tooling/xtask/src/tasks/bundle.rs, tooling/xtask/tests/release_promotion.rs_
    - _Validation: xtask workflow and release tests; hosted run_
- [~] 5. Operator handoff
  - [~] 5.1. Document secrets, recovery and evidence
    - _Requirements: 2.5_
    - _Depends on: 4.1_
    - _Reads: docs/src/development/copper-signing.md, docs/src/development/product-flavors.md_
    - _Writes: docs/src/development/copper-signing.md, docs/src/development/product-flavors.md_
    - _Validation: documentation and source identity audit_
