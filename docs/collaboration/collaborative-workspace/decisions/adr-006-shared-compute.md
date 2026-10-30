# ADR-006: Relay Mesh and Shared-Compute Policy

- **Status:** Accepted
- **Decision date:** 2026-08-14
- **Approval:** The product owner approved shared compute as default-off, initially self-hosted, fail-closed and governed by explicit trust, resource and fairness policy with no silent execution fallback.
- **Requirements:** 2.1, 16.3, 19.2
- **Capabilities:** CAP-035

## Context

Buzz has two related mesh capabilities. The relay mesh connects replicas using authenticated Iroh/QUIC peers and treats membership gossip only as a routing hint while Redis-fenced session generations remain authoritative. Buzz shared compute lets an explicitly opted-in community member advertise a local model and serve encrypted inference traffic directly to other members. Current behavior includes signed discovery, endpoint validation, relay allowlists and an honest privacy tradeoff: prompts leave the requesting machine and run on hardware controlled by another trusted community member.

Community membership is necessary but not sufficient consent to execute another person's code or prompts, consume their hardware, forward secrets or trust self-reported capacity. A scheduler must not turn temporary mesh unavailability into an invisible cloud/local-provider substitution. Multi-tenant deployments must also prevent a co-tenant community from discovering, selecting or consuming another community's compute.

## Decision

### Deployment and product defaults

Shared compute is disabled by default at the deployment, community, user and device levels. Enabling the feature requires all applicable operator/community policy plus an explicit device-owner **Share compute** action. Consuming shared compute requires the user or authorized agent configuration to select the shared-compute provider; ordinary agent/model defaults do not change merely because capacity appears.

The initial supported deployment policy is self-hosted only:

- compute nodes are controlled by the same Zed deployment/operator or by active members of the requesting community who explicitly share their own device;
- scheduling is restricted to the same host-derived community and approved deployment federation boundary;
- no public marketplace, anonymous provider, cross-community pool, paid third-party broker or provider discovered from an arbitrary client URL is eligible; and
- external Iroh relays may transport end-to-end encrypted traffic only when locally allowlisted; they do not become compute providers or trust authorities.

Third-party community compute requires a new approval record covering legal/data-processing terms, provider identity/attestation, billing, regional routing, abuse response, incident ownership, prompt retention, confidentiality and user-visible trust labels. It is prohibited until then.

Relay mesh is a separate operator feature. A single-instance deployment does not construct it. A multi-replica deployment may enable it only for replicas enrolled under the same deployment trust root and typed tenant boundary. Relay-mesh enablement does not enable shared compute.

### Canonical ownership

Zed's native agent/ACP job state remains the canonical execution owner. The remote/shared-compute scheduler selects an execution lease for that job; it does not create a second job or transcript. Agent permissions, cancellation, activity, result delivery and audit use the same canonical owners as local execution.

The collaboration domain owns provider-neutral job identity, requested model/capabilities, execution-location policy and authorization inputs. `remote` owns mesh protocol, advertisements, eligibility, scheduling and expiring execution leases. The shared-compute runtime owns bounded inference execution on an admitted serving device. Mesh gossip, local caches and presence are derived hints, never job or membership authority.

For relay-to-relay sessions, the canonical session directory/fenced generation is the ownership arbiter. A peer advertisement or gossip record may suggest a route but cannot grant a session, job, huddle or compute lease.

### Trust and consent policy

Eligibility requires every gate below at selection time and again before execution admission:

1. shared compute is enabled by deployment and community policy;
2. the requester is an active authorized community member or owner-attested agent whose delegation permits remote inference;
3. the serving owner is an active member of the same trusted community and has explicitly enabled sharing on that device;
4. the compute node identity, owner identity and endpoint tokens have valid signatures/attestations and supported protocol versions;
5. advertisements are fresh, bounded and match locally allowed Iroh relay/direct-transport policy;
6. requested model, capability, privacy class, context size and execution type are allowed by requester, owner and community policy;
7. the node has an available resource/concurrency lease within truthful locally enforced capacity;
8. neither owner, node, model, version nor endpoint is revoked, draining, quarantined or stale; and
9. the job's current canonical version is still pending and has no other executor lease.

Membership loss, owner opt-out, policy change, attestation failure or revocation immediately blocks new leases and cancels or drains active work according to the recorded policy. A passive gossip connection or leaked invite token cannot satisfy execution admission.

The UI must state that prompts, selected context and outputs are processed on another member's hardware, name the trust class and execution location, and distinguish community hardware from deployment-managed hardware. Enabling sharing states what models/resources are exposed, who may submit work, what metadata/usage is visible, and how to stop/drain it.

Secrets, credential handles, unrestricted filesystem paths, environment variables and private agent memory are never included merely because a job uses shared compute. Each transferable context class is explicitly selected by the canonical agent permission/context policy. Unsupported or unclassified sensitive context makes the node ineligible.

### Resource and isolation policy

Every serving device advertises policy-bounded capacity, but local enforcement is authoritative. A resource lease fixes:

- model identifier, verified artifact/digest and allowed capability;
- maximum prompt/context and output tokens/bytes;
- maximum wall-clock, idle and cancellation grace durations;
- CPU, GPU/device, memory, model-cache/disk and network budgets;
- maximum concurrent requests and queued requests;
- requester/community rate and usage ceilings; and
- expiry, nonce, canonical job ID and fenced executor generation.

Serving runs under the approved model runtime and sandbox/process owner with no implicit host filesystem, keyring, network, shell or tool access. Shared inference is a model provider, not a remote agent process or general code-execution grant. A future distributed/sharded model must enforce the same aggregate limits and identify every participating node before prompt release.

Advertisements that claim capacity outside local/deployment policy are rejected or clamped before display and cannot widen the server's actual enforcement. Exceeding a runtime bound cancels the lease, emits a bounded/redacted failure and cleans up model/session resources. Partial output is marked incomplete and cannot be reported as a successful agent turn.

### Fair scheduling

Scheduling occurs only among eligible nodes after trust and resource filtering. The default policy is deterministic weighted fair queuing by community member/requester, with equal default weight, per-requester concurrency and queue caps, per-device owner-reserved capacity and bounded aging so a sustained high-volume requester cannot starve others.

Fairness operates inside one community; capacity, queue length and usage from other tenants are neither considered nor exposed. Administrative weights and reservations require role-gated, versioned policy with audit attribution and bounded ranges. Payment, bidding, hidden priority and model-owner preference are not scheduling inputs in the initial self-hosted policy.

Admission either acquires one fenced resource/executor lease or returns a visible queued/no-capacity/policy-denied result. The scheduler cannot overcommit based only on stale advertisements. Lease expiry, cancellation, owner drain and node loss release capacity idempotently. Metrics include wait distributions, admission/rejection reason, utilization, cancellations and per-policy fairness ratios using bounded non-content labels.

### No-silent-fallback rule

If the user or job selects shared compute, Zed never silently executes that request through a local model, commercial API, different trust class, unapproved community, or generic remote-agent provider. No eligible capacity, policy denial, stale mesh, partition, model loss and execution failure are distinct visible outcomes.

Automatic retry may use the same node only while the canonical job's idempotency and delivery state prove that duplicate inference is safe. Moving a prompt to a different owner/node after bytes may have been delivered requires a policy that explicitly permits multi-node disclosure and an observable retry decision; the default is to ask the user. An unknown execution outcome is not converted into success or replayed invisibly.

The user may explicitly choose an approved alternative provider and retry as a new, linked execution attempt. UI and activity records show the old failure, new execution location and changed trust/cost boundary. Provider defaults are never mutated as a recovery side effect.

### Mesh protocol and fencing

Relay and compute mesh peers use versioned, size-bounded, authenticated frames over locally approved Iroh/direct transports. Runtime/node identities are boot/session scoped and attested by the appropriate deployment or member signing identity. Replay nonces, expiries and monotonic/fenced generations reject stale membership, advertisements, streams and datagrams.

For relay mesh, Redis/session-directory state remains the linearizable owner/fence. Gossip membership, phi suspicion and load are hints. Both sending and receiving seams validate tenant, profile, owner runtime and generation. Unknown versions, future/stale generations, missing leases, owner mismatch, draining peers and ambiguous tenant mappings fail closed with typed metrics and sanitized client errors.

For compute, the canonical executor/resource lease supplies the equivalent job fence. Results from a stale, wrong or duplicate executor cannot complete the job. Partition recovery re-reads membership, advertisements and canonical job/lease state before admitting or accepting further work.

### Data, privacy and observability

The relay coordinates signed discovery and canonical job state but does not receive inference prompt/output bytes when the direct encrypted path is healthy. Transport relays observe bounded network metadata. Serving devices necessarily receive the selected prompt/context and produce output; the consent UI states this plainly.

Prompt/output content is not placed in gossip, discovery, metrics, status events or operational logs. Retention on serving nodes is disabled by default beyond active execution buffers; model caches contain artifacts, not prompts. Any diagnostic capture or content retention requires a separate permission and policy.

Operators receive health/readiness, peer/version counts, fence/replay rejects, stale advertisement counts, lease state, capacity, queue/backpressure, fairness, latency, cancellation, partition and cleanup metrics with tenant access controls and bounded labels. Users see execution location, trust class, selected model, queued/running state, no capacity, policy denial, disconnect, cancellation and failure/recovery actions.

### Deployment and rollback policy

Shared-compute binaries, model runtimes and mesh listeners ship disabled with explicit kill switches, allowlists, resource ceilings and readiness checks. Configuration validation rejects wildcard community/provider trust, unbounded resources, unknown relay URLs, mutable/unverified model identities and missing identity material. Enabling is a reversible configuration action; this ADR authorizes specification and implementation, not production activation.

Rollback stops new admissions, drains or cancels active leases under policy, records terminal/unknown job state, disables advertisements/listeners and returns affected jobs to visible user action. It never reroutes them to another provider. Redis/gossip/advertisement state can expire; canonical job and audit state is retained. Source Buzz mesh remains available for compatibility only until Zed mesh conformance and client cutover pass, and never runs the same canonical job concurrently.

## Future third-party approval gate

Allowing compute outside the initial self-hosted boundary requires explicit approval of:

- provider identity, hardware/runtime attestation and key rotation;
- contractual confidentiality, prompt/output retention and deletion proof;
- data region/residency and cross-border disclosure;
- billing, quotas, fraud, dispute and fairness policy;
- provider isolation, sandbox, egress and incident-response evidence;
- user/organization allowlists and per-job trust labels/consent;
- independent security, privacy, abuse, load and revocation tests; and
- migration, disablement, provider failure and no-silent-fallback behavior.

Until that gate is accepted, third-party advertisements are ineligible even when protocol-valid.

## Alternatives rejected

1. **Enable sharing for every community member by default:** rejected because membership does not consent to hardware use or prompt disclosure.
2. **Treat signed advertisements as sufficient capacity/trust:** rejected because advertisements are stale, self-authored hints and local enforcement is authoritative.
3. **Use any community on a shared deployment as one pool:** rejected because tenant isolation and membership are community-scoped.
4. **Fall back to local/cloud inference on mesh failure:** rejected because it silently changes privacy, cost, model and execution location.
5. **Allow arbitrary third-party providers initially:** rejected because provider trust, legal, retention, billing and attestation policy is unapproved.
6. **Make relay gossip the session/job owner:** rejected because fenced canonical leases, not reachability hints, arbitrate ownership.

## Implementation and validation trace

- Task 4.8 defines independent mesh/shared-compute threat boundaries and negative tests.
- Tasks 33.1–33.6 preserve canonical remote-job provider, lease, secret, cancellation and cleanup ownership.
- Tasks 41.1–41.5 implement fenced mesh protocol, advertisements, eligibility/fair scheduling, native state UI and partition/security/load evidence.
- Tasks 44.3–44.5 own disabled-by-default configuration, kill switches, readiness and bounded observability.
- Tasks 46.1–46.6 own compatibility shadowing, divergence, rollback and removal of duplicate execution paths.
- Tasks 43.1–43.11 own companion-client compatibility, Tasks 44.6–44.7 own release/rollback procedures, Tasks 47.1–47.6 gate Buzz mesh retirement, and Tasks 48.1–48.4 retain final parity evidence.

Approval review requires fail-closed eligibility tables, same-community/self-hosted boundaries, explicit owner and consumer consent, resource/sandbox limits, fairness simulations, fencing/replay/partition tests, visible location/failure states, no-silent-fallback scenarios and disabled-by-default deployment evidence.
