# Validation: Native CoreX enablement

This is a future hardware/vendor-input specification. Missing proprietary inputs or hardware never pass; they leave the applicable task pending while the existing Comfy parity CoreX adapter remains compiled, zero-symbol, loader-free, and canonically typed `Unbound`.

## Scenarios

### VAL-COREX-ABI-001: Reviewed IXRT/IXBLAS ABI closure

- Verify lawful input provenance, exact header/version digests, complete declarations, signatures, layouts, targets, licenses, native C measurements, generated Rust measurements, and package policy.
- Command: `script/package-comfy-backend-corex --verify-sdk "$IXRT_HOME"` and `bash crates/comfy_backend_corex/abi/verify-completion-evidence.sh "$IXRT_HOME" target/evidence/comfy-corex-ixrt-0.8-proof.json`.

### VAL-COREX-UNBOUND-001: Fail-closed baseline preservation

- On every host before full closure, compile and test the CoreX feature and prove the manifest has zero callable symbols, no runtime loader path executes, every certificate projection is rejected, and the canonical state is `Unbound`.
- Command: `cargo test --locked -p comfy_backend_corex --all-targets` and `cargo test --locked -p comfy_tensor --features corex --all-targets`.

### VAL-COREX-OWNERSHIP-001: Authoritative ownership

- Search all Zed and comfy crates for competing CoreX ABI, trust, binding, capability, resource, memory, workspace, event, cancellation, queue, persistence, recovery, or transaction owners.
- Command: `cargo test --locked -p comfy_test_support --test ownership_consolidation val_ownership_001`.

### VAL-COREX-ADAPTER-001: Native semantic adapter

- Execute every reviewed row through the exact ABI harness; reject unadvertised or mismatched target/library/symbol/version/device rows before native effects and prove no CPU fallback.
- Command: `cargo test --locked -p comfy_tensor --features corex --all-targets`.

### VAL-COREX-TRUST-001: Signed package admission

- Verify strict canonical coverage, distinct signature domain, exact retained images, registry-only certificate issuance, tamper rejection, settings safety, and no self-authorization by discovery, package scripts, or feature compilation.
- Command: `cargo test --locked -p comfy_runtime --features corex --all-targets` and `cargo test --locked -p comfy_test_support --test native_release_boundary`.

### VAL-COREX-INTEGRATION-001: Production worker integration

- Verify profile selection, private protocol, retained session/workspace ownership, readiness operation, matrix negotiation, cancellation, device loss, teardown, restart recertification, CLI/GPUI typed unavailable behavior, and no external execution path.
- Command: `cargo test --locked -p comfy_worker --features corex --all-targets`, `cargo test --locked -p zed --features corex --all-targets`, and `cargo test --locked -p comfy_test_support --test native_controller_e2e`.

### VAL-COREX-HARDWARE-001: Physical CoreX certification

- On the approved lab only, verify the signed schema-v2 artifact binds the exact ABI/implementation/package/device/environment/memory/provenance/matrix payload and exact live replay.
- Missing hardware or approved signing material leaves this validation pending.

### VAL-COREX-NATIVE-BOUNDARY-001: Native-only release boundary

- Inspect dependencies, package contents, binary strings, process/network traces, and failure paths for zero Python, JavaScript extension, ComfyUI, external server, or CPU fallback behavior.
- Command: `cargo test --locked -p comfy_test_support --test native_release_boundary` and `./script/clippy`.

## Pack gate

Before this future specification can be declared complete, every task and validation above must pass and:

```sh
python3 .agents/skills/coding/scripts/validate_spec.py .agents/specs/comfy-corex-enablement --require-complete
```
