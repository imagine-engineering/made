# Collaborative Workspace architecture

This document is the canonical maintainer and operator map for Collaborative
Workspace after the Buzz consolidation. It describes the implemented product
boundary, the sole owner of each state family, supported compatibility
adapters, durable data flows, migration history, rollback ceiling and retained
artifacts. It does not authorize production activation, a live authority
cutover, an irreversible migration, source deletion or data deletion.

The implementation evidence is complete for all 45 capabilities and 93
acceptance criteria in the [parity report](../../.agents/specs/collaborative-workspace/parity-report.md).
The repository-only product and operational approval is recorded in the
[final sign-off](../../.agents/specs/collaborative-workspace/final-signoff.md);
production actions remain separately gated.

## Product and process boundary

Zed remains one product with two build and presentation profiles:

- Standard Zed does not compile or package `multiplayer-tools`. A persisted
  Collaborative presentation resolves to Editor for that run without changing
  the saved preference or collaborative data.
- Multiplayer Zed explicitly enables `zed/multiplayer-tools` and adds the
  Collaborative Workspace presentation, collaboration clients and approved
  service/package surfaces. It does not fork Editor, project, worktree, Git,
  credentials, settings, ACP or transcript state.
- `collab` 0.44.0 is the sole collaboration command and durable-write service.
  Separate listeners, transports and delivery processes are adapters around
  its admitted commands or already-authorized work.
- The React/Tauri Buzz desktop is not an architecture template or runtime
  dependency. The supplied Buzz tree remains a read-only reference and
  rollback source while its retirement gate is on hold.

Logical topology:

```text
Zed GPUI / supported legacy clients / CLI / ACP peers
                         |
       RPC, HTTP, Nostr, CLI, ACP, Git and media adapters
                         |
       version negotiation -> tenant admission -> authorization
                         |
               canonical domain command
                         |
           collab transaction and durable receipt
              /          |             \
 signed event/state   transactional   canonical aggregate
     repositories        outbox       repositories/resources
              \          |             /
       projections, search, notifications and bounded workers
                         |
      GPUI stores, query adapters and redacted observability
```

An adapter may duplicate a representation, but never authority. Every write is
negotiated before tenant lookup, admitted to one tenant and principal,
authorized against the canonical state, and assigned an idempotent operation
or signed-event identity before a transaction. Projection, delivery and
execution workers claim fenced outbox, lease or generation records; retries do
not create a second writer.

## Canonical ownership

The following table is exhaustive at the durable aggregate-family level.
“Adapter or derived boundary” identifies allowed consumers and translators,
not an alternate transition owner.

| Aggregate family | Canonical transition owner | Durable writer or resource owner | Adapter or derived boundary |
| --- | --- | --- | --- |
| Account binding, principal and collaboration identity | `collaboration_domain` identity/binding rules plus native credential policy | Collab identity-binding repository; keys remain in Zed credential storage | Pairing, Nostr identity and Buzz imports must verify current binding and custody |
| Community, membership, role, invitation and join policy | `collaboration_domain` community, membership and authorization aggregates | Collab tenant/domain repositories | RPC, Nostr, web, mobile and CLI call the same admitted commands |
| Channel, message, reaction, thread, DM, marker, read state, reminder and scheduled message | `collaboration_domain` communication aggregates | Collab channel/message/event repositories and transactional outbox | Redis fan-out, client stores and legacy routes are derived or adapter-only |
| Signed event and replaceable head | `nostr_compat` verification plus Collab ingest policy | Collab `EventRepository` and event/outbox transaction | Nostr codecs and frozen Buzz fixtures own no store |
| Projection, search and subscription cursor | Collab rebuild, index, query and subscription policy | Canonical projection/search/outbox tables and cursor writes | Search indexes and caches rebuild from authority and cannot repair it |
| Local project, worktree, repository, index and diff | Existing Zed `project`, `worktree`, `git` and `git_ui` owners | Native filesystem and Git resources | Collaborative bindings and NIP-34 records reference, never reconstruct, local authority |
| Hosted project, repository, branch, review and CI status | Collaboration-domain project/Git/review rules under the selected hosted authority | Collab project/Git/review repositories; repository bytes remain in the configured Git resource | Git smart HTTP, NIP-34 and CLI are admitted adapters |
| Media object and metadata | Collaboration media admission, validation and retention policy | Collab media metadata plus the configured object store | Blossom and clients cannot bypass canonical storage admission |
| Push notification and wake | Canonical notification policy and Collab push admission | Collab `PushOutboxRepository` | Push gateway owns only fenced provider-delivery attempts and cannot synthesize notifications |
| Native agent session, transcript and action activity | Zed `agent`, `acp_thread` and `action_log` | Native agent/session stores | ACP, NIP-AO and collaboration activity are adapters/projections |
| Agent configuration, personas, teams and imported private state | Native Agent/settings/credential owners plus collaboration-domain configuration rules | Existing Zed stores and verified import receipts | Buzz snapshots and projections are one-time, versioned import inputs |
| Collaboration job, delegation and usage | `collaboration_domain::job` plus Agent `JobExecutionCoordinator` | Collab `JobRepository` and canonical audit/usage chain | Remote providers execute under one generation-fenced lease |
| Workflow definition, trigger, run, step, retry, approval, queue admission and audit | Collab workflow trigger/evaluator/action/approval modules and `WorkflowScheduler` | `WorkflowRepository`, workflow lease/admission tables and audit/outbox in one PostgreSQL authority | Webhook, Nostr and CLI submit only; limits are 1,000 queued/community, 10,000 queued/deployment, 16 running/community and 4 running/definition |
| Audit, moderation, feedback, archive, retention and community deletion | Collaboration-domain policy plus Collab administrative executors | Canonical audit/moderation/archive/checkpoint repositories | Import and rollback reads grant no mutation authority; irreversible transitions remain approval-gated |
| Presence, typing, huddle, pairing and relay-mesh state | Canonical ephemeral admission and generation owners at each documented boundary | Presence, typing and relay frames have no durable authority; durable huddle/account effects use canonical repositories | LiveKit, pair relay and mesh peers are bounded transports, not domain owners |
| Desktop settings, drafts, layout and platform resources | Existing Zed settings, workspace, project, worktree, terminal and platform owners | Native Zed stores, filesystem and process owners | Buzz desktop data remains read-only migration input until receipt and deletion gates pass |
| Migration checkpoint, cutover authority and rollback evidence | Canonical migration and cutover state machines | Collab checkpoint/cutover repositories; DDL belongs only to the privileged migration runner | Buzz schema and snapshots are read-only source/rollback evidence |

The [no-duplicate audit](../../.agents/specs/collaborative-workspace/retirement/no-duplicate-audit.md)
is the detailed source and dependency proof behind this matrix. There is no
dual-write row.

## Adapter and compatibility boundaries

All supported ranges are closed. The machine-readable
[compatibility matrix](compatibility.md) is normative when a number below and
that matrix ever disagree.

| Boundary | Supported contract | Authority constraint |
| --- | --- | --- |
| Collaboration HTTP | Version 1 | Negotiates before tenant/resource lookup; only `supported` admits a write |
| Zed RPC | Version 68 | Carries canonical IDs and versions into the same domain command path |
| Nostr ingress | Version 1 | Verifies exact event/auth semantics, then submits one canonical command |
| Domain command | Version 1 | Sole mutation-facing service contract; durable receipt makes retries idempotent |
| Buzz CLI forwarding | Version 1; shim 0.1.0 | Syntax and output adapter to canonical APIs; no database or server authority |
| NIP-AB / NIP-44 / NIP-PL | Versions 1 / 2 / 1 | Pairing, encrypted payload and push-lease adapters retain canonical custody/admission |
| Buzz audio gateway | Versions 1–2, non-write-bearing | Media compatibility over the canonical huddle lifecycle and LiveKit transport |
| Buzz Postgres | Schema 30 exactly, read-only | Import and retained rollback source only; no canonical service may serve or write it |

Supported client releases are Zed desktop 1.16.2, Buzz desktop 0.5.11,
Buzz mobile 0.0.0+1, Buzz web 0.1.0, Buzz CLI shim 0.1.0 and Buzz admin web
0.1.0. Each frozen Buzz client reaches Collab through its declared adapter.
Unknown clients, features, protocols or schema combinations reject before
tenant lookup and do not fall back to a Buzz writer.

The separately deployed push gateway claims canonical outbox work and owns
provider delivery only. `pair-relay` forwards opaque encrypted frames and owns
no account or workspace state. The temporary Nostr sidecar, legacy-read
comparison, one-way projection bridge, ACP/provider shim, CLI shim, audio
gateway and deep-link aliases follow the bridge register in the
[operations runbook](operations.md#temporary-bridge-register). None may use
bidirectional last-writer-wins reconciliation.

## Authoritative data flows

### Signed collaboration write

1. The client or adapter negotiates its exact closed version and requested
   write features without consulting tenant state.
2. Authentication resolves one current principal; tenant admission and common
   authorization run before content observation or resource allocation.
3. Protocol decoding and cryptographic verification produce one versioned
   canonical command with a stable operation or event ID.
4. Collab applies the domain transition, durable receipt, authoritative record
   and outbox record in one transaction under tenant policy.
5. Fenced workers build projections, search, notification, audit and client
   results. A projection mismatch rebuilds from authority; it never writes back.

### Project, Git and agent activity

Native Zed project/worktree/Git and agent/ACP state remains authoritative for
local resources. Collaboration records bind tenant-scoped IDs and positive
versions to those owners. Hosted Git commands pass the selected authority and
repository policy before bytes change. Job or provider execution obtains one
generation-fenced claim from `JobExecutionCoordinator`; its semantic activity
is projected to Collaborative Workspace without creating a second transcript.

### Durable workflows

Trigger ingestion first records a durable candidate. The workflow owner
resolves one immutable definition revision, evaluates policy and atomically
admits or rejects the candidate through the scheduler counters. Admitted work
uses fenced run and step leases, durable retries, idempotent actions and
explicit approval state. Crash/restart recovery reconstructs the ready queue
from PostgreSQL; no in-memory or adapter queue owns admission. OL-EXE-04 limits
are enforced in the same authority: 1,000 pending workflows per community,
10,000 per deployment, 16 concurrent workflows per community and 4 concurrent
runs per workflow definition.

### Reads, imports and retirement

Reads use authorized canonical projections or a declared compatibility
projection. Buzz schema 30, desktop files, snapshots and object/Git baselines
are immutable migration inputs. Import receipts and provenance bind each source
record to one canonical result. Derived Redis presence, typing and cache state
is allowed to expire and repopulate; it is never migrated as authority.

## Schema and migration history

The canonical Collab migration manifest contains 22 ordered, checksummed
migrations. They add, in order:

| Versions | Durable scope |
| --- | --- |
| `20260815000100` | Identity bindings |
| `20260820000100`–`20260820000900` | Signed events/heads, projections, outbox, search, migration checkpoints, channels and messages |
| `20260822000100`–`20260822000400` | Channel search, push, projects and hosted Git |
| `20260823000100`–`20260823000200` | Git review and collaboration jobs |
| `20260824000100`–`20260824000500` | Workflow definitions/runs, run leases, approvals, audit and moderation |
| `20260825000100` | Durable workflow scheduler admission and OL-EXE-04 counters |
| `20260826000100` | Zed community/channel/principal bindings and message read state |
| `20260930000100` | Room participant authorization generation |
| `20261009000100`–`20261010000600` | Canonical identity profile, verified registration/observation, optional moderation subject targets, accepted NIP-05 names, and signed restriction reason bounds |

`deploy/collaboration/migrations/manifest.json` is the checksum and ordering
authority. The release and runtime schema ceiling is exactly
`20261010000600`. Buzz's 30 migrations remain preserved read-only source
evidence and are not part of the canonical DDL chain.

Migration phases advance authority one aggregate at a time: baseline and ADRs;
native GPUI composition; domain/identity/service foundations; communication
shadow reads; communication write cutover; project/Git/agent integration;
workflow and infrastructure cutover; client/deployment migration; then
retirement. The detailed entry, exit and reconciliation gates are in the
[migration plan](../../.agents/specs/collaborative-workspace/migration-plan.md).

## Rollback ceiling and recovery

The current package targets Collab 0.44.0 with canonical schema
`20261010000600`; live qualification is tracked separately in the validation
report. A prior binary may be used only when its published
maximum admits the deployed schema. Disabling `multiplayer-tools` is a binary
and presentation rollback: it does not down-migrate or delete data.

Before service activation and before the migration floor is sealed, an
authorized migration owner may reverse only to a checksummed target that is not
below the stored rollback floor. After activation, sealing or a new-only write,
there is no safe in-place down migration. Recovery requires a global write
freeze and coordinated restoration of database, object, Git and configuration
checkpoints into an isolated target, followed by compatibility and
reconciliation proof. Community deletion becomes forward-repair-only after its
recorded irreversible checkpoint.

Operators must choose exactly one of `RB-01` through `RB-11` in the
[rollout and rollback runbook](operations.md#rollback-paths-and-last-reversible-checkpoints).
The runbook also owns release roles, immutable change records, preflight,
canary stages, automatic stops, incident response and the required tabletop
record. No architecture document is an approval to execute those procedures.

## Deployment and observability

The supported deployment paths are the checked Compose and Helm profiles under
`deploy/collaboration`. They use immutable image digests, separate runtime and
DDL credentials, one explicit route, persistent Git storage, bounded resources
and private monitoring. Release tooling verifies eleven archives, twelve signed
subjects, the compatibility matrix, notices and migration inputs without
publishing or deploying them.

Readiness and automatic stops cover schema/migration status, tenant admission,
authorization failures, event/projection/outbox lag, drift, queue/backpressure,
replica freshness, direct legacy-write attempts, workflow/job/provider lease
state, huddle/push/mesh lifecycle and rollback readiness. A cross-tenant
mismatch, signature/authorization disagreement, data-loss count, unbounded
queue or legacy-only write after cutover is a stop-ship. Logs and metrics obey
the runbook's redaction policy and never carry tenant, user, content, path, URL,
prompt, output, token, key or credential material.

## Retained artifacts and retirement state

The following remain required even after a runtime retirement:

| Artifact | Retention rule |
| --- | --- |
| Frozen protocol, migration and client fixtures | Permanent byte/hash evidence with independent checkers |
| Custom NIP specifications and adapters | Retain while signed history or a supported client uses the contract |
| Buzz migration adapters and 30 SQL migrations | Retain through source/data rollback and legal-retention windows |
| Independent conformance package and formal models | Preserve as non-production oracles |
| `buzz` CLI compatibility shim | Retain while listed in the supported-client matrix or until an approved usage gate removes it |
| Visual baselines | Retain as native composition evidence |
| Compatibility, parity, security, scale, cutover and retirement reports | Retain with release and specification history |
| License and provenance records | Retain permanently; releases carrying derived compatibility artifacts include `LICENSES/buzz.md` and applicable root notices |
| Database, object, Git, configuration and source-import checkpoints | Retain through the named rollback window and restore-drill policy |
| Complete Buzz source history or content-addressed archive | Required before deleting the supplied `projects/buzz` reference tree |

The [preserved-artifact ledger](../../.agents/specs/collaborative-workspace/retirement/preserved-artifacts.md)
contains exact counts and digests. Source retirement is currently **HOLD**:
the supplied Buzz directory has no independently proven durable Git history,
and no approved immutable repository, tag, bundle or content-addressed archive
has replaced its local-path record. Live process, traffic, rollback-window,
data and human-approval gates also remain environment-specific. Do not remove a
bridge, source tree, prior release, snapshot or compatibility range merely
because implementation parity passed.

## Maintainer and operator index

- Normative compatibility ranges and negotiation:
  [compatibility.md](compatibility.md)
- Preflight, rollout, rollback, incident and bridge procedures:
  [operations.md](operations.md)
- Detailed architecture decisions and implementation anchors:
  [design.md](../../.agents/specs/collaborative-workspace/design.md)
- Migration phases, precedence and removal gates:
  [migration-plan.md](../../.agents/specs/collaborative-workspace/migration-plan.md)
- Final capability and acceptance-criterion evidence:
  [parity-report.md](../../.agents/specs/collaborative-workspace/parity-report.md)
- Repository-only product and operational decision:
  [final-signoff.md](../../.agents/specs/collaborative-workspace/final-signoff.md)
- Canonical owner/dependency proof:
  [no-duplicate-audit.md](../../.agents/specs/collaborative-workspace/retirement/no-duplicate-audit.md)
- Retained artifact checksums and source-history gate:
  [preserved-artifacts.md](../../.agents/specs/collaborative-workspace/retirement/preserved-artifacts.md)

Canonical message authors retain their stable principal identifier and original signed event. A verified virtual-agent author also retains its canonical profile, owning principal, exact original owner attestation and admission membership versions. Current message mutations and duplicate retries recheck the verified profile, owner and explicit channel membership under the existing moderation and identity locks. Message transactions request Read Committed so reads after a lock wait observe committed revocation; unrelated outbox mutations retain their configured isolation. Historical attribution verifies the admitted proof at its original source time, independently of later renewal, expiry or owner deactivation.

Schema 30 gives channel memberships, messages, auxiliary actors and read frontiers mutually exclusive ordinary-member and canonical-agent references. Ordinary foreign keys and tenant fences retain their prior behavior. Agent references require an existing verified agent profile; they do not create a physical community membership or inherit an owner's administrative role. Rollback refuses retained virtual references before changing any data. The message DTO continues to carry the same stable principal identifier; author display metadata is read through the canonical authored-profile owner and does not grant identity or membership authority.
