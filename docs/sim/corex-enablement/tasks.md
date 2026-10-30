# Tasks: Native CoreX enablement

These tasks were transferred from `.agents/specs/comfy-parity/`. Every task remains pending. None is implementation-complete without the stated proprietary or hardware input, and no task may weaken the currently compiled zero-symbol typed-Unbound boundary.

- [ ] 1. Bind and package the native device ABI: Iluvatar CoreX/IXUCA
  - _id: comfy-parity-device-foundation-iluvatar-corex-ixuca-comfy-model-0020
  - _Wave: 1
  - _Reads: crates/comfy_backend_corex, crates/comfy_types/src/comfy_types.rs, crates/comfy_runtime/src/trust.rs, .agents/specs/comfy-parity/catalogs/native-backend-dependencies.json
  - _Writes: crates/comfy_backend_corex/build.rs, crates/comfy_backend_corex/src/abi.rs, crates/comfy_backend_corex/src/loader.rs, crates/comfy_backend_corex/abi/symbols-v1.json, crates/comfy_backend_corex/abi/reviewed-bindings-v1.txt, crates/comfy_backend_corex/abi/verify-completion-evidence.sh, crates/comfy_backend_corex/LICENSES, nix/comfy-backends/corex, script/package-comfy-backend-corex
  - _Validation: script/package-comfy-backend-corex --verify-sdk "$IXRT_HOME"; bash crates/comfy_backend_corex/abi/verify-completion-evidence.sh "$IXRT_HOME" target/evidence/comfy-corex-ixrt-0.8-proof.json; cargo test --locked -p comfy_backend_corex --all-targets; ./script/clippy -p comfy_backend_corex
  - _Requirements: 1.1, 1.2, 2.1, 5.1, 5.2
  - Outcome: Replace the zero-symbol manifest only after lawful IXRT/IXBLAS inputs establish exact reviewed declarations, layouts, symbols, targets, licenses, package policy, and typed unavailable behavior.
  - Design: D1, D3
  - Done when: Complete independently reviewed ABI evidence and native C/Rust measurements agree; missing or unapproved inputs still leave the existing zero-symbol adapter unchanged and Unbound.

- [ ] 2. Implement native device adapter: Iluvatar CoreX/IXUCA
  - _id: comfy-parity-native-device-iluvatar-corex-ixuca-comfy-model-0020
  - _Wave: 2
  - _Blocked_by: comfy-parity-device-foundation-iluvatar-corex-ixuca-comfy-model-0020
  - _Reads: crates/comfy_backend_corex, crates/comfy_tensor/src/comfy_tensor.rs, crates/comfy_tensor/src/operation.rs, crates/comfy_worker/src/memory_modes.rs
  - _Writes: crates/comfy_tensor/src/backends/iluvatar_corex_ixuca_comfy_model_0020.rs, crates/comfy_tensor/src/ops/backend_iluvatar_corex_ixuca_comfy_model_0020.rs, crates/comfy_tensor/tests/backends/iluvatar_corex_ixuca_comfy_model_0020.rs, crates/comfy_tensor/tests/iluvatar_corex_ixuca_comfy_model_0020.rs
  - _Validation: cargo check --locked -p comfy_tensor --features corex; cargo test --locked -p comfy_tensor --features corex --all-targets; ./script/clippy -p comfy_tensor --features corex
  - _Requirements: 2.1, 2.2, 5.1, 5.2
  - Outcome: Implement only the reviewed CoreX semantic rows through the canonical TensorBackend capability, storage, workspace, accounting, resource, event, cancellation, and error owners.
  - Design: D1, D3
  - Done when: Every advertised row executes through the reviewed ABI harness, every unadvertised row fails typed before native effects, and no adapter-owned generic service or CPU fallback exists.

- [ ] 3. Provision signed native FFI contracts: Iluvatar CoreX/IXUCA
  - _id: comfy-parity-provision-native-ffi-contracts-iluvatar-corex-ixuca-comfy-model-0020
  - _Wave: 3
  - _Blocked_by: comfy-parity-native-device-iluvatar-corex-ixuca-comfy-model-0020
  - _Reads: crates/comfy_backend_corex/src/abi.rs, crates/comfy_backend_corex/src/loader.rs, crates/comfy_runtime/src/trust.rs, crates/comfy_runtime/src/settings.rs, nix/comfy-backends/corex
  - _Writes: crates/comfy_runtime/src/native_ffi_corex.rs, crates/comfy_runtime/src/comfy_runtime.rs, crates/comfy_runtime/src/trust.rs, crates/comfy_runtime/src/settings.rs, crates/settings_content/src/settings_content.rs, nix/comfy-backends/corex/package-policy.json, nix/comfy-backends/corex/ffi-contracts-v1.schema.json, assets/settings/default.json
  - _Validation: cargo test --locked -p comfy_runtime --features corex --all-targets; cargo test --locked -p comfy_test_support --test native_release_boundary; cargo test --locked -p comfy_test_support --test ownership_consolidation val_ownership_001; ./script/clippy -p comfy_runtime -p comfy_test_support
  - _Requirements: 3.1, 5.1, 5.2
  - Outcome: Add the separately reviewed CoreX package signature domain and strict contract mapping into the sole native FFI registry without allowing local observations to self-authorize.
  - Design: D1, D2, D3
  - Done when: Exact signed fixtures admit only covered retained images; malformed, unsigned, tampered, stale, incomplete, or wrong-target packages fail before registry construction or loader entry.

- [ ] 4. Integrate certified native device into production selection: Iluvatar CoreX/IXUCA
  - _id: comfy-parity-integrate-device-iluvatar-corex-ixuca-comfy-model-0020
  - _Wave: 4
  - _Blocked_by: comfy-parity-provision-native-ffi-contracts-iluvatar-corex-ixuca-comfy-model-0020
  - _Reads: crates/comfy_runtime/src/native_ffi_corex.rs, crates/comfy_tensor/src/backends/iluvatar_corex_ixuca_comfy_model_0020.rs, crates/comfy_worker/src/comfy_worker.rs, crates/comfy_worker/src/supervisor.rs, crates/zed/src/zed.rs, crates/comfy_ui/src/execution_surfaces.rs
  - _Writes: crates/comfy_runtime/src/native_execution_controller.rs, crates/comfy_runtime/src/runtime_supervisor.rs, crates/comfy_types/src/worker_protocol.rs, crates/comfy_worker/src/comfy_worker.rs, crates/comfy_worker/src/supervisor.rs, crates/comfy_worker/src/comfy_worker_main.rs, crates/zed/src/zed.rs, crates/zed/src/comfy_cli.rs, crates/comfy_ui/src/execution_surfaces.rs, crates/comfy_test_support/tests/native_controller_e2e.rs, crates/comfy_test_support/tests/native_release_boundary.rs
  - _Validation: cargo test --locked -p comfy_worker --features corex --all-targets; cargo test --locked -p zed --features corex --all-targets; cargo test --locked -p comfy_test_support --test native_controller_e2e; ./script/clippy -p comfy_worker -p zed -p comfy_test_support
  - _Requirements: 2.2, 3.2, 5.1, 5.2
  - Outcome: Add CoreX to the one native profile, worker session, readiness, protocol, recovery, CLI, and GPUI selection chain with no alternate owner or fallback.
  - Design: D2, D3
  - Done when: Ready is reachable only after exact package verification, registry certification, retained session construction, device probing, matrix negotiation, and a real readiness transaction; restart recertifies and every failure remains typed.

- [ ] 5. Certify native device adapter on hardware: Iluvatar CoreX/IXUCA
  - _id: comfy-parity-certify-device-iluvatar-corex-ixuca-comfy-model-0020
  - _Wave: 5
  - _Blocked_by: comfy-parity-integrate-device-iluvatar-corex-ixuca-comfy-model-0020
  - _Reads: crates/comfy_tensor/src/backends/iluvatar_corex_ixuca_comfy_model_0020.rs, crates/comfy_test_support/src/device_certification.rs, .agents/specs/comfy-parity/catalogs/native-tensor-operation-contracts.csv, .agents/specs/comfy-parity/catalogs/native-backend-abi/corex.json
  - _Writes: crates/comfy_test_support/tests/device_iluvatar_corex_ixuca_comfy_model_0020.rs, .agents/specs/comfy-corex-enablement/artifacts/native-device-certification/iluvatar-corex-ixuca-comfy-model-0020.json
  - _Validation: cargo test --locked -p comfy_test_support --features corex,hardware-certification --test device_iluvatar_corex_ixuca_comfy_model_0020; ./script/clippy -p comfy_test_support --features corex,hardware-certification
  - _Requirements: 4.1, 4.2, 5.1, 5.2
  - Outcome: Run the complete matrix on approved physical CoreX hardware and create the signed exact-environment artifact using separately authorized lab signing material.
  - Design: D3, D4
  - Done when: The exact live lab replay verifies every matrix and provenance row with zero unexplained skips; without hardware or signing material this task remains pending and the parity baseline remains typed Unbound.
