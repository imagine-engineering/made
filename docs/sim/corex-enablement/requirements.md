# Requirements: Native CoreX enablement

> Current first-party naming: **Sim** (`sim_*` crates and the opt-in `sim` feature). ComfyUI source names, COMFY parity/task IDs, historical evidence and completed task states remain unchanged. Legacy Comfy identifiers are compatibility names; see [the rename specification](../comfy-to-sim/design.md). This naming update does not implement pending parity or extraction tasks.


## Problem

The Comfy parity baseline contains a compiled, zero-symbol Iluvatar CoreX adapter that must remain fail-closed because the repository does not contain lawfully supplied and independently reviewed IXRT/IXBLAS headers, runtime libraries, signing material, or CoreX hardware evidence. This future specification owns the work that may enable CoreX without weakening that baseline.

## Scope

- In scope: reviewed IXRT/IXBLAS ABI evidence, focused native adapter implementation, signed package admission, production worker integration, and actual CoreX hardware certification.
- Out of scope: changing CPU or Apple Metal baseline certification, Python or external-server fallback, redistributing unapproved vendor payloads, or treating compilation and discovery as availability.

## Requirements

### Requirement 1: Admit proprietary ABI evidence lawfully

#### Acceptance criteria

1. WHEN IXRT/IXBLAS headers are supplied under an approved license THEN the implementation SHALL record exact versions, complete declaration ranges, normalized digests, symbol signatures, struct layouts, targets, and redistribution constraints without copying unapproved proprietary source into release artifacts.
2. IF complete reviewed ABI evidence is absent, inconsistent, unlicensed, or unverifiable THEN CoreX SHALL remain the zero-symbol canonical typed `Unbound` adapter and no loader SHALL execute.

### Requirement 2: Implement a focused native CoreX adapter

#### Acceptance criteria

1. WHEN the reviewed ABI contract is complete THEN `sim_backend_corex` SHALL be the sole CoreX ABI, loader, unsafe-call, opaque-resource, and package adapter beneath the canonical `NativeFfiRegistry`, `NativeBackendBindingStatus`, `BackendCapabilityMatrix`, tensor resource, memory, workspace, event, and cancellation owners.
2. WHEN any target, library, symbol, version, device, capability row, package contract, or certificate is missing or mismatched THEN selection SHALL fail with a typed unavailable or `Unbound` result before graph dispatch and SHALL NOT retry on CPU.

### Requirement 3: Provision signed trust and production integration

#### Acceptance criteria

1. WHEN a CoreX package is admitted THEN a separately reviewed strict contract catalog and backend-specific signature domain SHALL cover every permitted library, digest, ABI, symbol, unsafe owner, target, license, and package-policy fact before the sole native FFI registry issues certificates.
2. WHEN CoreX is selected in production THEN the private Rust worker SHALL retain the certified session and canonical backend/workspace pair, complete a real readiness transaction, publish only the instance-derived matrix, and repeat the entire certification chain after restart.

### Requirement 4: Certify only on actual CoreX hardware

#### Acceptance criteria

1. WHEN an approved CoreX lab runs the certification matrix THEN the signed artifact SHALL bind the exact ABI and implementation digests, package evidence, driver, SDK, OS, device identity, named non-zero memory observations, provenance, and every pass, unsupported, or failure row.
2. IF CoreX hardware, approved signing material, or the exact certified environment is unavailable THEN the hardware gate SHALL remain pending and SHALL NOT be represented as a pass, simulated certification, or implementation-completion claim.

### Requirement 5: Preserve the baseline fail-closed boundary

#### Acceptance criteria

1. UNTIL every preceding requirement is implemented and validated, the Comfy parity pack SHALL continue compiling CoreX only as its zero-symbol structural adapter with no runtime loader, no certificate projection, no executable kernel, and canonical typed `Unbound` state.
2. THE CoreX implementation SHALL remain entirely native Rust and SHALL NOT launch, embed, manage, require, or connect to Python, JavaScript extensions, ComfyUI, or an external execution server.
