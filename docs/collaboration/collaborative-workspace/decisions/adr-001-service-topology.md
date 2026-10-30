# ADR-001: Collaboration Service and Database Topology

- **Status:** Accepted
- **Decision date:** 2026-08-14
- **Approval:** The product owner approved the recommended ADR-001 decision: the existing Zed `collab` deployment is the final collaboration-service owner, and a Buzz-derived ingress sidecar is temporary and bounded.
- **Requirements:** 2.1, 2.2, 2.3
- **Capabilities:** CAP-003, CAP-005, CAP-043

## Context

Buzz currently treats `buzz-relay` and its signed Nostr event log as the collaboration authority. It uses Axum 0.8 and SQLx 0.9 around Postgres, Redis, object storage and protocol-specific services. Zed already operates the `collab` binary and owns its deployment, authentication integration and relational collaboration data through Axum 0.6, SeaORM 1.1.10 and SQLx 0.8. Directly combining those dependency graphs before compatibility work would create an unsafe migration and release boundary.

The final product cannot operate two collaboration control planes or two authoritative message, identity, presence, workflow, project, Git, transcript or agent-session stores. Existing Buzz signed events and wire protocols must nevertheless remain verifiable and interoperable throughout migration.

## Decision

### Final process topology

The existing Zed `collab` deployment is the final service and operational owner for server-side collaboration. In the final topology it hosts, or directly supervises as non-authoritative protocol listeners, all Zed RPC, Nostr WebSocket/HTTP, authentication, search, media, Git, workflow, notification, administration and compatibility ingress required by this specification.

Protocol listeners translate admitted requests into a shared, versioned collaboration-domain command contract. They do not author independent domain state. Cross-subsystem coordination occurs through canonical commands, authoritative records and an ordered transactional outbox rather than private listener databases.

Zed build, packaging, release, configuration, secrets, logging, telemetry, health, readiness and deployment conventions are canonical. `projects/buzz` remains a buildable reference and migration source until its retirement gates pass; it is not shipped as a nested product or permanent service control plane.

### Temporary Nostr ingress sidecar

A Buzz-derived Nostr ingress sidecar is allowed only during migration Phases 2 through 7 while Axum, SQLx and SeaORM integration is aligned. The sidecar:

1. implements versioned Nostr WebSocket/HTTP compatibility and exact response behavior;
2. receives a trusted, typed `TenantContext` from the same tenant catalog and admission policy used by `collab`;
3. submits admitted operations through the same versioned domain-command and outbox contract as in-process listeners;
4. returns protocol responses using authoritative result IDs and versions;
5. has no independent migration runner, tenant catalog, projection writer, administrative authority or source-of-truth store; and
6. emits operation IDs, event IDs, domain versions and compatibility metrics needed for reconciliation.

The sidecar may read canonical compatibility projections and the signed event log through narrow service interfaces. It may not directly mutate projection tables, accept writes when the canonical command service is unavailable, or silently fall back to a legacy Buzz database.

### Dependency-version ownership

The Zed workspace and `crates/collab/Cargo.toml` own final server dependency versions. Axum, SQLx, SeaORM, Tokio, TLS, serialization and observability dependencies must be aligned and validated in the Zed workspace before Nostr routes move in-process. During the bounded sidecar period, legacy versions may remain isolated in the compatibility binary, but the binary communicates only through the versioned adapter contract and cannot leak its framework types into the domain or persistence layers.

Dependency alignment is complete only when the combined service passes protocol fixtures, database migration tests, tenant-isolation tests, load/backpressure tests and rollback drills under the Zed release toolchain.

### Database and migration authority

There is one Postgres deployment for server collaboration data and one Zed-owned migration authority. The canonical migration runner owns ordering, checksums, forward/backward compatibility metadata, locks and release gates for both preserved Buzz tables and new/consolidated Zed tables. The sidecar and compatibility clients never execute schema migrations.

Authority is assigned by aggregate:

| Aggregate or state | Canonical authoring owner | Compatibility or derived representation |
| --- | --- | --- |
| Nostr-authored messages, social records and externally authored workflow records | Verified, immutable signed event log in the collaboration database | Tenant-fenced relational projections, search documents, Nostr query responses and Zed RPC payloads |
| Service-issued membership, authorization summaries and bounds | Authorized Zed collaboration domain service | Relay-signed protocol events and client projections |
| Local projects, worktrees and files | Existing Zed `project` and `worktree` owners | Community/project mappings and signed protocol references |
| Local Git working tree, index and diffs | Existing Zed `git`, `project::git_store` and `git_ui` owners | Hosted refs, patches, reviews and status events |
| Native ACP sessions, transcripts and actions | Existing Zed `agent`, `acp_thread`, `agent_ui` and permission stores | Agent/job/activity events and compatibility frames |
| Projection rows | Their declared authoritative record and version | Rebuildable rows carrying community, source kind, source ID, source version and projection timestamp |
| Redis presence, typing, fan-out and caches | No durable authority; current canonical service decision plus TTL | Derived, tenant-scoped, expiring runtime state rebuilt from live input or authoritative events |
| Object and hosted-Git blobs | Zed-owned storage metadata referencing content-addressed objects | Blossom, Git smart-HTTP and legacy object coordinates |

Existing Zed local project, Git and ACP persistence remains separate because it owns different local aggregates, not duplicate collaboration authority. Overlapping Buzz and Zed channel/member/room tables are consolidated only after provenance-aware backfill and differential-read evidence.

### Write, projection and reconciliation rules

Every accepted write has one stable operation or signed-event ID. The canonical service commits the authoritative record and one ordered outbox entry atomically. Search, notification, audit, compatibility and other projections consume that outbox idempotently. Clients and listeners never dual-write two authorities.

When temporary dual reads or derived legacy writes are required, reconciliation compares tenant, source ID, source version, ordering/cursor state and visibility—not merely row counts. Drift is observable by aggregate, tenant, adapter version and deployment. A projection can be discarded and rebuilt from authoritative records; Redis state is allowed to expire and repopulate.

## Sidecar entry, removal and rollback gates

### Entry gates

The sidecar may receive traffic only after all of the following are true:

- ADR-001 and ADR-002 are accepted;
- baseline Nostr protocol, authorization and tenant-isolation fixtures pass;
- the typed tenant catalog and versioned command/outbox contract are deployed;
- the sidecar has no schema-migration credentials and no direct projection-write path; and
- dashboards expose command rejection, event/projection/outbox lag, divergence, compatibility version and tenant-boundary failures.

### Removal gates

The sidecar must be removed when all of the following are true:

- final Zed-owned Axum/SQLx/SeaORM dependency alignment is complete;
- Nostr WebSocket and HTTP routes run in the final `collab` service deployment;
- Buzz differential protocol, authentication, authorization, pagination, backpressure and reconnect suites pass against the in-process routes;
- supported desktop, CLI, web and mobile compatibility clients pass version negotiation and end-to-end fixtures;
- the observation window records zero unexplained authorization, tenant, ordering, projection or response divergence;
- no production route, release manifest, migration job or operator runbook depends on the sidecar; and
- rollback to the last compatible `collab` release has been rehearsed without restoring a second database authority.

Removal deletes sidecar routing and deployment artifacts only after retained compatibility evidence and source-retirement approval. It does not delete Buzz source data or immutable signed events.

### Rollback

Before canonical write cutover, rollback stops the new ingress and resumes the prior relay write path against preserved source data. During shadow reads, derived projections may be discarded and rebuilt. During canonical write cutover, operators freeze admission, drain and verify the outbox, confirm zero unexplained divergence, then switch compatible routing to the prior binary against the same authoritative signed event log. After a documented point-of-no-return schema write, rollback requires the recorded snapshot, coordinated write freeze and approved recovery procedure; it never enables concurrent authorities.

## Consequences

- Zed has one final collaboration service owner, one server migration authority and one authoring owner per aggregate.
- Buzz protocol code can be preserved and tested without making Buzz a second product.
- Dependency alignment and in-process route movement become explicit implementation and removal work rather than an assumed manifest merge.
- Postgres projection convergence is aggregate-specific and evidence-gated; Redis remains disposable derived state.
- The temporary sidecar adds deployment and observability cost during Phases 2–7, but its permissions, data access and lifetime are explicitly bounded.

## Alternatives rejected

1. **Permanent Buzz relay beside `collab`:** rejected because it preserves a second control plane, migration authority and failure surface.
2. **Immediate manifest and schema merge:** rejected because current Axum/SQLx/SeaORM generations and overlapping schemas require compatibility and rollback evidence first.
3. **Independent databases with application dual writes:** rejected because partial failure creates competing truths and unverifiable ordering.
4. **Replacing the signed event log with Zed projections:** rejected because it breaks signed-event provenance, Nostr replacement semantics and established client interoperability.
5. **Making the signed event log authoritative for local projects, Git or ACP transcripts:** rejected because those aggregates already have complete canonical Zed owners.

## Implementation and validation trace

- **Service/adapters:** Tasks 14.1, 14.6, 44.1 and 44.2 implement the approved listener and service boundary.
- **Persistence/outbox:** Tasks 15.1–15.7 implement the single migration authority, signed event repository, projection provenance and transactional outbox.
- **Migration/reconciliation:** Tasks 17.1–17.9 and 46.1–46.6 implement inventory, import, shadowing, bounded mirroring, divergence reporting and rollback.
- **Operations/removal:** Tasks 44.3–44.7, 47.1–47.6 and 48.1–48.4 implement deployment ownership, compatibility gates and source retirement.

Architecture review acceptance requires evidence that exactly one migration runner can mutate the collaboration schema, every listener reaches the same command/outbox authority, sidecar credentials cannot write projections or migrations, and every removal gate above has an owned validation task and observable signal.
