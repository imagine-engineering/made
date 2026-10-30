# Design: Native CoreX enablement

> Current first-party naming: **Sim** (`sim_*` crates and the opt-in `sim` feature). ComfyUI source names, COMFY parity/task IDs, historical evidence and completed task states remain unchanged. Legacy Comfy identifiers are compatibility names; see [the rename specification](../comfy-to-sim/design.md). This naming update does not implement pending parity or extraction tasks.


## Overview

This future pack extends the already compiled fail-closed CoreX structural boundary only after proprietary inputs are lawfully supplied. It does not replace any canonical Zed service. The adapter remains a focused vendor boundary under existing trust, binding, tensor, memory, worker, persistence, queue, cancellation, recovery, and GPUI owners.

## Decisions

### D1: Keep one focused CoreX vendor owner

- Choice: `sim_backend_corex` alone owns reviewed IXRT/IXBLAS declarations, loading, unsafe calls, opaque resources, and structural package metadata.
- Rationale: NativeFfiRegistry, NativeBackendBindingStatus, BackendCapabilityMatrix, and the existing tensor/worker services already own trust, binding, semantic capability, resources, and lifecycle.
- Consequence: Vendor DTOs map explicitly into canonical domain types and cannot self-certify availability.

### D2: Require a complete signed-package-to-session chain

- Choice: A CoreX-specific strict catalog and signature domain map exact retained library images into NativeFfiRegistry certificates before any unsafe loader or device operation runs.
- Rationale: Headers, installed SDKs, package receipts, successful discovery, and compiled features are observations, not authority.
- Consequence: Missing or mismatched evidence returns typed unavailable before graph dispatch, with no CPU fallback.

### D3: Preserve Unbound until atomic enablement closure

- Choice: The existing zero-symbol structural adapter remains the production behavior until ABI, semantic, trust, integration, and hardware tasks all pass.
- Rationale: Partial enablement would create an unverifiable execution and security boundary.
- Consequence: No intermediate task may advertise a kernel, issue a certificate, or change canonical `Unbound` state by itself.

### D4: Separate implementation from hardware observation

- Choice: Implementation harnesses validate exact adapters and fail-closed paths, while only an approved physical CoreX lab may create the signed hardware artifact.
- Rationale: Fake SDKs and non-target hosts cannot establish real driver, memory, device-loss, or performance facts.
- Consequence: Unavailable hardware remains a pending gate, never a completed task.

## Failure and recovery

Every malformed, incomplete, unlicensed, unsigned, stale, wrong-target, wrong-device, or tampered input fails before loader entry. Worker restart destroys the old session and repeats package verification, registry certification, retained-image loading, device probing, readiness, and negotiation. Cancellation and device loss use existing canonical owners and commit no partial output.

## Traceability

| Criterion | Design coverage | Verification type | Planned check / expected signal |
| --- | --- | --- | --- |
| 1.1 | D1 | Static/ABI | `VAL-COREX-ABI-001` records reviewed declarations and measured layouts |
| 1.2 | D3 | Failure/static | `VAL-COREX-UNBOUND-001` proves zero-symbol typed Unbound without loader calls |
| 2.1 | D1 | Ownership/integration | `VAL-COREX-OWNERSHIP-001` finds one vendor owner and canonical service mappings |
| 2.2 | D2, D3 | Failure integration | `VAL-COREX-ADAPTER-001` rejects every missing or mismatched prerequisite before dispatch |
| 3.1 | D2 | Security/integration | `VAL-COREX-TRUST-001` verifies signed exact-image registry admission |
| 3.2 | D2 | Protocol/E2E | `VAL-COREX-INTEGRATION-001` proves readiness, execution, teardown, and restart recertification |
| 4.1 | D4 | Hardware | `VAL-COREX-HARDWARE-001` verifies a signed exact-lab artifact |
| 4.2 | D3, D4 | Failure/manual | Missing lab inputs leave the task pending and canonical Unbound unchanged |
| 5.1 | D3 | Static/integration | `VAL-COREX-UNBOUND-001` passes before the complete enablement chain |
| 5.2 | D1, D2 | Boundary | `VAL-COREX-NATIVE-BOUNDARY-001` finds no Python, JavaScript, ComfyUI, or external-server path |
