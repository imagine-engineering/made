# Collaboration rollout and rollback runbook

This runbook is the operator handoff for the Zed-owned collaboration service,
schema migrations, companion-client compatibility adapters, and temporary Buzz
bridges. It defines preparation, canary, stop, rollback, reconciliation, and
incident procedures. It does not authorize a production deployment, migration,
cutover, destructive restore, or source retirement.

Production promotion remains blocked until the applicable compatibility,
security, fault-injection, load, and rollback gates in Tasks 45–48 pass. In
particular, `/healthz` currently proves process and database reachability, while
the deployment observability bundle defines consumers for the complete readiness
and kill-switch contract. Operators must not treat either as proof that all
runtime emitters or admission controls exist until their owning verification
tasks pass.

## Safety invariants

1. There is one named release, deployment, migration, incident, and rollback
   owner for a change. One person may hold more than one role, but an ownership
   conflict stops the operation.
2. One aggregate has one writable authority. A rollback never enables a prior
   writer until new admissions are stopped, in-flight work is drained or
   cancelled, and reconciliation proves that no target-only mutation would be
   lost.
3. Release and migration inputs are immutable and digest-pinned. Production
   values contain references to existing Secrets, never secret values.
4. Postgres, object, Git, source-import, and configuration checkpoints are
   retained until the recorded rollback window closes. Redis presence, typing,
   caches, and advertisements are derived and are rebuilt rather than restored.
5. A schema-compatible prior binary is preferred over a down migration. A down
   migration is allowed only before the migration rollback floor is sealed and
   only with explicit migration authorization.
6. Snapshot restore, a production migration or cutover, destructive cleanup,
   and crossing any recorded irreversible checkpoint require separate explicit
   approval. This runbook is not that approval.
7. Operational output follows
   [`logging-policy.json`](../../deploy/collaboration/observability/logging-policy.json):
   no tenant, user, event, repository path, URL, message, media, prompt, output,
   token, key, or credential is copied into a ticket, command line, metric label,
   or log attachment.

## Roles and authorization boundaries

| Role               | May                                                                                                                                                         | Must not                                                                                                    |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Release owner      | Select one source commit, verify the candidate manifest and provenance, and record immutable artifact and image digests                                     | Publish, deploy, read runtime secrets, or declare schema compatibility from semantic-version similarity     |
| Deployment owner   | Render approved Compose/Helm configuration, operate routing and admission controls, deploy or restore a schema-compatible runtime, and verify rollout state | Use DDL credentials, change compatibility ranges, restore data, or enable two writers                       |
| Migration owner    | Use the DDL-only Secret to validate, apply, verify, seal, or—before the floor—reverse canonical migrations                                                  | Route traffic, use the runtime Secret for DDL, clear a migration halt, or cross the rollback floor          |
| Data owner         | Create and verify database, object, Git, source-import, and configuration checkpoints; authorize a restore plan                                             | Delete a retained source or snapshot, expose private data in evidence, or restore while writes are admitted |
| Incident commander | Declare the incident, choose containment, name the sole rollback owner, and coordinate evidence and communication                                           | Perform an unapproved cutover or override an irreversible checkpoint                                        |
| Rollback owner     | Execute exactly one approved rollback path and maintain the operation record                                                                                | Run competing rollback paths, down-migrate after activation, or resume writes before reconciliation         |
| Tenant operator    | Validate explicitly scoped client behavior and report tenant-safe outcomes                                                                                  | Access deployment/DDL credentials, infer another tenant's state, or authorize platform rollback             |

The runtime Secret and migration Secret remain distinct. Release candidates and
rendered manifests contain no credentials. Access to private metrics or redacted
logs does not grant tenant, migration, or deployment authority.

## Change record

Create one immutable change record before an environment is mutated. Record:

- change ID, environment, source commit, release-manifest digest, and the
  provenance-verification result for the manifest and every selected artifact;
- current and target runtime image digests, migration-image digest, chart
  version, effective configuration checksum, and compatibility policy version;
- current schema, target schema, migration status, rollback floor, and the
  previous binary's maximum admitted schema;
- database checkpoint, object-version/checkpoint, Git-volume checkpoint,
  source-import snapshot, and encrypted configuration backup identifiers;
- canary communities, routing owner, observation window, success thresholds,
  automatic stop signals, and the tested admission-disable mechanism;
- every active temporary bridge, its serving/writing authority, precedence rule,
  reconciliation cursor, removal date or gate, and rollback route;
- the named owners above, approval references, start time, and the last
  reversible checkpoint.

Never record secret values, raw tenant identifiers, content samples, or bearer
credentials. Use environment-owned opaque references and pseudonymous
correlation hashes.

## Preflight

### 1. Verify release inputs

Download the release candidate into an empty directory. Verify every subject's
GitHub OIDC provenance against the expected repository and source commit using
the organization-approved verifier, then run the checked-in contract:

```sh
script/collaboration-release-contract verify \
  --manifest target/collaboration-release/release-manifest.json
```

The contract must report a valid manifest, all eleven archives, the exact
compatibility matrix, versions, checksums, notices, and signature subject list.
The candidate workflow does not publish production images. The release owner
must separately record the approved promotion evidence connecting each selected
container digest to its verified candidate artifact. Absence or ambiguity stops
the rollout.

### 2. Validate configuration without secrets

For Compose, preserve the current `.env`, fill a new protected environment file,
and render it before start:

```sh
COLLABORATION_ENV_FILE=.env.next deploy/collaboration/compose/run.sh config
python3 deploy/collaboration/compose/check.py
```

For Kubernetes, keep environment-owned values outside the repository and render
exactly what the deployment controller will apply:

```sh
helm lint deploy/collaboration/charts/collaboration \
  -f deploy/collaboration/charts/collaboration/values-production.yaml \
  -f "$COLLABORATION_ENVIRONMENT_VALUES"
helm template "$COLLABORATION_RELEASE_NAME" \
  deploy/collaboration/charts/collaboration \
  -f deploy/collaboration/charts/collaboration/values-production.yaml \
  -f "$COLLABORATION_ENVIRONMENT_VALUES" >"$COLLABORATION_RENDERED_MANIFEST"
```

Review the render for immutable runtime and migration digests, HTTPS endpoints,
one explicit Ingress or Gateway attachment, separate runtime/DDL Secrets,
private monitoring, bounded resources, required network ranges, persistent Git
storage, and a schema requirement equal to `20261010000500`. Do not attach the
route or apply the manifest during preflight.

### 3. Verify migration and observability contracts

These commands are read-only unless `DATABASE_URL` is supplied to a database
subcommand:

```sh
deploy/collaboration/migrations/migrate.py validate
deploy/collaboration/migrations/migrate.py plan
python3 deploy/collaboration/migrations/check.py
python3 deploy/collaboration/observability/check.py
```

With migration authorization and the DDL-only credential, initialize/read the
operator control state and capture validated database state before applying
schema migrations. These database subcommands create the protected migration
control tables when absent:

```sh
DATABASE_URL="$COLLABORATION_DDL_DATABASE_URL" \
  deploy/collaboration/migrations/migrate.py status
DATABASE_URL="$COLLABORATION_DDL_DATABASE_URL" \
  deploy/collaboration/migrations/migrate.py verify
```

The expected preflight state is `ready`, with no halt reason, a history matching
the packaged checksums, and a current version and rollback floor copied exactly
into the change record. `checksum_drift`, `history_drift`, `execution_failure`,
an unknown version, or a schema outside the published compatibility matrix stops
the rollout. Operators never edit the control/history tables to clear a halt.

### 4. Prove recovery readiness

The data owner must verify that all checkpoints are restorable, not merely that
backup jobs reported success. Production recovery objectives are a recovery
point of at most 15 minutes, a recovery time of at most four hours, and one
database/object/key/checkpoint restore drill at least every 90 days. A backup
older than 30 minutes or a restore drill older than 90 days blocks rollout.

The recovery proof includes database consistency, object hashes and tenant
prefixes, Git object/ref integrity, migration/import checkpoints, credential
reference availability without exporting keys, and configuration recovery. A
restore rehearsal uses an isolated target and cannot overwrite the serving
environment.

## Rollout procedure

### Stage 0: Offline gate

1. Complete the change record and obtain separate production migration/cutover
   approval when applicable.
2. Pass release, configuration, migration, compatibility, observability, and
   recovery preflight.
3. Confirm the prior runtime digest admits the target schema. If it does not,
   the runtime rollback path is closed and the approved snapshot-restore path
   must be ready before any mutation.
4. Exercise the admission-disable mechanism and require acknowledgement within
   60 seconds with zero accepted admissions afterward. Until Task 45.3 proves
   that exact path, production promotion is blocked.
5. Confirm all canary and serving routes are detached from the candidate.

### Stage 1: Migrate with no candidate traffic

Only the migration owner may start the DDL job. The Helm pre-install/pre-upgrade
hook runs the separately pinned migration image with the DDL-only Secret. For a
manual approved recovery or rehearsal, the equivalent command is:

```sh
DATABASE_URL="$COLLABORATION_DDL_DATABASE_URL" \
COLLABORATION_REQUIRED_SCHEMA_VERSION=20261010000500 \
  deploy/collaboration/migrations/migrate.py up
DATABASE_URL="$COLLABORATION_DDL_DATABASE_URL" \
  deploy/collaboration/migrations/migrate.py verify
```

Do not seal the rollback floor. A failed or interrupted run stays halted or
resumes from its committed prefix; it is never restarted against changed source
bytes. Compare schema state, source/import checkpoints, outbox responsibility,
object/Git integrity, and every compatibility range before continuing.

### Stage 2: Unrouted runtime canary

The deployment owner applies the previously reviewed render through the
environment's approved deployment controller without attaching public traffic.
Require all candidate replicas to start, remain database-reachable, expose only
the private metrics contract, and report the expected binary/schema/policy
versions. Full readiness requires the authority, schema, migration, compatibility
and kill-switch signals specified by OL-OPS-01; `/healthz` alone is insufficient.

Any missing emitter, unsupported peer, public metrics listener, unknown label,
secret/content canary, stale replica, migration halt, or mismatched configuration
checksum stops the rollout and selects a rollback path below.

### Stage 3: Read-only community canary

Use environment-owned tenant-sticky routing to admit only the approved canary
communities for reads. Legacy remains serving authority during shadow reads;
canonical results are diagnostic. Compare authorization, ordering, unread state,
search, notifications, cursors, overlays, object/Git hashes, and client
compatibility. Do not advance with unexplained divergence or a route that can
mix one community across authorities.

### Stage 4: Write canary

This is an explicit authority cutover and requires its own approval. Freeze the
affected communities, drain the authoritative outbox, record its cursor, prove
zero unexplained divergence, and atomically select one canonical writer. Every
legacy write path must reject before the candidate accepts writes. Admit one
approved community cohort, then verify signed-event/operation IDs, one mutation,
one outbox responsibility, projections, client negotiation, and audit
attribution. Never dual-write for comparison.

### Stage 5: Expand or hold

Advance only after the approved observation window completes with every success
gate green and no stop signal firing. Expand by whole community cohorts so tenant
authority never splits. At every cohort boundary, record routing, schema,
outbox/reconciliation cursors, compatibility peers, projection drift, replica
freshness, queue age, and the current last reversible checkpoint.

When all approved cohorts are stable, the migration owner may seal the schema
rollback floor at the activated version using the same verified migration
artifact. The source form below is allowed only when its manifest and migration
checksums match that recorded artifact:

```sh
DATABASE_URL="$COLLABORATION_DDL_DATABASE_URL" \
  deploy/collaboration/migrations/migrate.py seal \
  --expected-version 20261010000500
```

Sealing is the schema point of no return. It requires explicit approval and
closes every down-migration path at or below that version. It does not authorize
removing snapshots, source data, bridges, or the prior release.

## Automatic stop and alert response

Every signal in
[`stop-signals.json`](../../deploy/collaboration/observability/stop-signals.json)
is a page-level gate. The first action is closed; the incident commander may
narrow impact only after containment, never weaken it from the alert text.

| Signal                         | Immediate action                                              | Required evidence before resume                                                            |
| ------------------------------ | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `STOP-CROSS-TENANT`            | Halt the tenant and rollback; escalate as a security incident | Tenant boundary trace is reconciled with no foreign result or count                        |
| `STOP-REPLAY-OR-STALE`         | Halt deployment and rollback                                  | Replay/fence source is fixed and the accepted-write ledger is reconciled                   |
| `STOP-DUPLICATE-EXECUTION`     | Disable execution and preserve state                          | Exactly one executor remains; every job/session has a terminal or explicit unknown outcome |
| `STOP-SHADOW-DIVERGENCE`       | Halt cutover and preserve state                               | Divergence has a classified source and zero unexplained records remain                     |
| `STOP-SECRET-CANARY`           | Disable admissions and rollback                               | Exposure path is contained; redacted retained evidence proves no secret remains in output  |
| `STOP-CONTENT-CANARY`          | Disable admissions and rollback                               | Content path is contained and all operational surfaces are content-free                    |
| `STOP-READY-AUTHORITY`         | Remove instances from readiness and rollback                  | Every required authority is current and checked by readiness                               |
| `STOP-READY-SCHEMA`            | Remove instances from readiness and halt migration            | Migration state/history and service schema window are valid                                |
| `STOP-READY-SECURITY`          | Disable admissions and rollback                               | Required security control is restored and independently exercised                          |
| `STOP-QUEUE-BOUNDARY`          | Halt scoped admissions and reconcile                          | Oldest outbox age is at most 30 seconds normally and no responsibility is lost             |
| `STOP-CLAIM-BOUNDARY`          | Halt scoped admissions and reconcile                          | No claim exceeds twice its lease and every claim has one owner                             |
| `STOP-LEASE-BOUNDARY`          | Disable execution and reconcile                               | No mesh lease exceeds 60 seconds and stale results are rejected                            |
| `STOP-CLIENT-TELEMETRY`        | Halt release                                                  | Disabled client telemetry produces exactly zero requests                                   |
| `STOP-ADMISSION-AFTER-DISABLE` | Halt deployment and preserve state                            | The kill switch is observed and no post-disable admission remains unexplained              |
| `STOP-KILL-SWITCH-ACK`         | Halt deployment and preserve state                            | All owners acknowledge disablement within 60 seconds                                       |
| `STOP-ROLLBACK-OWNER-CONFLICT` | Halt rollback and preserve state                              | One rollback owner and one path are recorded                                               |

No alert is cleared by editing metrics, labels, migration history, or audit data.
Resume requires a new approved change record or an explicit continuation of the
existing record with the incident commander, rollback owner, evidence, and new
last reversible checkpoint recorded.

## Rollback paths and last reversible checkpoints

Use exactly one row. If its precondition is not true, stop and escalate; do not
compose two rollback paths ad hoc.

| ID      | Scope and trigger                                                                            | Last reversible checkpoint                                                                                                              | Authorized rollback                                                                                                                                                                                                                         | Completion evidence                                                                                                                                                                                               |
| ------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RB-01` | Compose runtime/configuration failure with a schema-compatible prior image                   | Immediately before `./run.sh start`, with `.env`, database/object/Git checkpoint, current schema, and prior digest recorded             | Deployment owner sets the immutable `COLLABORATION_PREVIOUS_IMAGE` and runs `deploy/collaboration/compose/run.sh rollback`; no volumes or dependencies are removed                                                                          | Only Collab is recreated, `/healthz` passes, schema is unchanged, and compatibility plus reconciliation gates pass                                                                                                |
| `RB-02` | Kubernetes runtime/configuration failure with a schema-compatible prior image                | Immediately before the approved deployment controller applies the candidate render                                                      | Deployment owner renders `values-production.yaml`, environment values, then `values-rollback.yaml` with the prior digest and its maximum schema; migration must render absent                                                               | Render selects only the prior digest, preserves Git/external authorities, contains no migration Job, rollout becomes fully ready, and reconciliation passes                                                       |
| `RB-03` | Canonical migration failure before service activation and before the rollback floor          | The verified database/object/Git checkpoint immediately before `migrate.py up`                                                          | Migration owner keeps admissions disabled, runs `migrate.py verify`, and may run `migrate.py down --target-version "$COLLABORATION_PREVIOUS_SCHEMA"` only when history is clean and the target is not below the stored floor                | `verify` reports ready at the target, checksums match, prior runtime admits it, and retained data/integrity checks pass                                                                                           |
| `RB-04` | Schema or authoritative-data failure after activation/sealing or after a new-only write      | The last verified coordinated checkpoint before the first new-only write; there is no safe in-place down migration                      | Incident commander keeps a global write freeze; data owner restores database/object/Git/configuration together into an isolated target, verifies it, then the deployment owner atomically switches to the matching prior runtime and routes | Restored checkpoint identities match, the recovery point and any uncovered work are recorded, no post-freeze write is accepted, compatibility/readiness/reconciliation pass, and recovery objectives are recorded |
| `RB-05` | Multiplayer desktop presentation or binary regression without authority/schema change        | The prior signed schema-compatible desktop/service release before cohort routing                                                        | Release/deployment owner routes the prior compatible service; users may run Standard Zed, which resolves Collaborative presentation to Editor without changing saved state                                                                  | Saved presentation/credentials/data remain intact, incompatible commands reject before tenant lookup, and re-enable round trip passes                                                                             |
| `RB-06` | Communication read shadow or pre-point-of-no-return write cutover divergence                 | The last clean tenant/aggregate reconciliation cursor before canonical write activation                                                 | Deployment owner freezes affected writes, drains/verifies outbox, stops new ingress, and selects the prior adapter/binary against the same authoritative signed event log; derived projections may be rebuilt                               | One writer, zero legacy-only or unexplained writes, event IDs/heads and client responses reconcile, and no source data is deleted                                                                                 |
| `RB-07` | Temporary bridge or hosted-authority transfer failure                                        | The bridge's recorded last clean source version and mapping before atomic activation                                                    | Aggregate owner freezes writes and restores prior routing/mapping only when source/target heads prove no target-only mutation; otherwise use approved forward repair                                                                        | Precedence rule holds, one authority is writable, both candidate records are retained for conflicts, and divergence alerts clear from reconciled evidence                                                         |
| `RB-08` | Workflow, agent/provider, huddle, push, pairing, or mesh cutover failure                     | The last clean admission/lease/run/session/outbox checkpoint before enabling the aggregate                                              | Aggregate owner stops new admissions, drains or cancels under the documented ceiling, preserves terminal/unknown state, disables listeners/advertisements, and restores prior routing without reroute                                       | No duplicate executor/delivery, every owned resource is released or explicitly uncertain, canonical audit state remains, and derived Redis/gossip state expires safely                                            |
| `RB-09` | Community deletion or other lifecycle operation reaches its recorded irreversible checkpoint | The durable checkpoint immediately before `rollback_irreversible` acquires a timestamp                                                  | Before that timestamp, lifecycle owner may use the canonical recovery transition; after it, rollback is prohibited and incident response may only complete/repair forward                                                                   | Authority, stage, checkpoint, retained identifiers, and permitted recovery action agree; tenant reuse remains blocked until terminal completion                                                                   |
| `RB-10` | Buzz pre-1321 source normalization fails                                                     | The mandatory source snapshot taken before `1321_backfill_default_community.sql`                                                        | Data owner restores that source snapshot under a coordinated write freeze; canonical migration down files do not apply                                                                                                                      | Buzz source schema/count/hash evidence returns to the recorded snapshot and canonical import has accepted no divergent writes                                                                                     |
| `RB-11` | Phase-8 source retirement or compatibility removal fails after removal approval              | The final retained version-control release, source/data snapshot, compatibility evidence, and rollback window checkpoint before removal | Release owner restores an explicitly approved release from version control; an old binary may run only if it admits the retained schema and cannot create a second authority                                                                | Build/deployment no longer depends on an incompatible source, required fixtures/notices/history remain, and every active compatibility owner is explicit                                                          |

For `RB-02`, render the rollback candidate before submitting it to the approved
deployment controller:

```sh
helm template "$COLLABORATION_RELEASE_NAME" \
  deploy/collaboration/charts/collaboration \
  -f deploy/collaboration/charts/collaboration/values-production.yaml \
  -f "$COLLABORATION_ENVIRONMENT_VALUES" \
  -f deploy/collaboration/charts/collaboration/values-rollback.yaml \
  --set "rollback.targetImageDigest=$COLLABORATION_PREVIOUS_IMAGE_DIGEST" \
  --set "rollback.maximumSchemaVersion=$COLLABORATION_PREVIOUS_MAXIMUM_SCHEMA" \
  >"$COLLABORATION_ROLLBACK_MANIFEST"
```

The chart rejects a prior binary whose maximum schema is below the deployed
schema and suppresses the migration hook in rollback mode. Never bypass that
check or use a chart revision that changes persistent ownership.

## Incident procedure

1. **Declare and contain.** Open the redacted incident record, name the incident
   commander and sole rollback owner, identify the firing stop signal, detach
   candidate traffic or stop scoped admissions, and preserve state. Security and
   tenant-boundary signals receive immediate security escalation.
2. **Fence authority.** Record current writer, routing generation, outbox and
   projection cursors, migration status/floor, runtime/configuration digests,
   active executions, and irreversible lifecycle checkpoints. Do not resume a
   prior writer yet.
3. **Choose one path.** Select `RB-01` through `RB-11` only when its precondition
   and last reversible checkpoint are proven. If none applies, keep admissions
   disabled and prepare an explicitly approved forward repair.
4. **Execute under authorization.** The role named by the row performs the
   operation. A migration, restore, cutover, deletion recovery, or source
   restoration requires its separate approval gate.
5. **Reconcile.** Verify one writer; signed event and operation IDs; source and
   target versions; outbox/claim/lease ownership; projection and client results;
   object/Git hashes; workflow/job/session terminal or unknown states; and audit
   continuity. Never reconcile with last-writer-wins.
6. **Recover service.** Require full authority/schema/migration/compatibility/
   security readiness, private observability, zero stop signals, compatible
   clients, and the approved observation window before reopening admissions.
7. **Close safely.** Record actual recovery point/time, data or work not covered
   by the restored checkpoint, follow-up owners, retained evidence, and the new
   rollback window. Do not remove a snapshot, bridge, prior release, or source as
   part of incident cleanup.

## Temporary bridge register

Every active bridge must appear in the change record. The canonical precedence
and rollback rules are:

| Bridge                                | Serving/writing rule                                                           | Reconciliation and rollback                                                                                     | Removal gate                                              |
| ------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Nostr ingress sidecar                 | Protocol adapter only; all writes enter the canonical command/outbox authority | Compare operation/event ID, domain version, and response; stop sidecar and restore prior ingress before cutover | Final Collab routes and full conformance pass             |
| Legacy versus canonical reads         | Legacy serves; canonical is diagnostic                                         | Compare tenant/query/cursor/overlay hashes; discard/rebuild derived projection on rollback                      | Approved zero-unexplained-divergence window               |
| Canonical outbox to legacy projection | One-way derived write only                                                     | Reconcile outbox sequence and source ID/version; halt on legacy-only write                                      | Every supported client reads canonical projection/adapter |
| Buzz ACP/provider shim                | Canonical Zed job/session owner executes once                                  | Reconcile stable job/session ID and executor lease; quiesce and restore routing, never duplicate                | Providers use Zed-owned runtime directly                  |
| `buzz` CLI shim                       | Syntax adapter to canonical APIs                                               | Golden output/error/exit evidence; select a compatible service, never a Buzz database writer                    | Approved usage threshold or explicit long-term owner      |
| Buzz Opus huddle adapter              | Media/lifecycle compatibility only                                             | Reconcile session/participant/event identity; quiesce room before route change                                  | ADR-004 criterion and supported-client floor              |
| Old URLs/deep links                   | Alias to canonical navigation                                                  | Reconcile normalized entity IDs; restore alias without changing authority                                       | Compatibility policy explicitly permits removal           |

No bridge uses bidirectional last-writer-wins. A bridge without a recorded owner,
precedence rule, reconciliation cursor, alert, rollback route, and dated removal
gate is not eligible for canary traffic.

## Tabletop review record

Before production use, reviewers walk `RB-01` through `RB-11` in order and fill
one row per scenario. A scenario passes only when the team can identify the
actual immutable input, authorized actor, command or controller action, last
reversible checkpoint, observable stop, expected recovery evidence, and the
condition that forbids rollback. “Use the latest backup” or “roll back Helm” is
not evidence.

| Scenario | Trigger injected | Last reversible checkpoint resolved | Authorization resolved | Stop/rollback path followed | Recovery evidence observed | Result |
| -------- | ---------------- | ----------------------------------- | ---------------------- | --------------------------- | -------------------------- | ------ |
| `RB-01`  |                  |                                     |                        |                             |                            |        |
| `RB-02`  |                  |                                     |                        |                             |                            |        |
| `RB-03`  |                  |                                     |                        |                             |                            |        |
| `RB-04`  |                  |                                     |                        |                             |                            |        |
| `RB-05`  |                  |                                     |                        |                             |                            |        |
| `RB-06`  |                  |                                     |                        |                             |                            |        |
| `RB-07`  |                  |                                     |                        |                             |                            |        |
| `RB-08`  |                  |                                     |                        |                             |                            |        |
| `RB-09`  |                  |                                     |                        |                             |                            |        |
| `RB-10`  |                  |                                     |                        |                             |                            |        |
| `RB-11`  |                  |                                     |                        |                             |                            |        |

Attach only redacted output and immutable references. An incomplete or failed
row blocks production promotion and becomes an owned follow-up; it is not waived
by the rest of the table passing.

## Post-rollout and retirement

Keep the prior signed runtime, verified snapshots, migration/source data,
compatibility adapters, and reconciliation evidence for the approved rollback
window. Continue 30-second private metric scrapes, 14-day redacted operational
log retention, compatibility observations for at least seven days and through
one rollback window, and the 90-day restore-drill cadence.

Source retirement remains a separate approval gate. Do not remove `projects/buzz`,
turn off a bridge, narrow a compatibility window, delete a source record, or
remove a prior artifact until its Task 47–48 usage, parity, traffic, security,
rollback-window, notice/history, and ownership exit criteria pass.

## Optional profile HTTP admission configuration

The existing ordinary signed-profile `/events` and `/query` service and public
metadata/directory routes are disabled unless an explicit community/public-URL
mapping and shared API quota are supplied. Compose passes through
`COLLABORATION_NOSTR_ROUTES` (a JSON array of `community_id`/`public_url` objects)
and `COLLABORATION_NOSTR_API_CALLS_PER_MINUTE` when set. Helm binds the same
settings from `nostrHttp.enabled`, `nostrHttp.routes` and
`nostrHttp.callsPerMinute`, requiring shared Redis and adding only exact
`/events` and `/query` ingress paths. Configuration participates in the existing
pod checksum; defaults remain disabled. Optional
`COLLABORATION_NOSTR_CLOSED_WEBSOCKET` / `nostrHttp.closedWebsocket` opts into the
existing closed-current-member NIP42 socket on the same exact root, with explicit
routes/quota and shared Redis still required. Unset omits the flag, explicit
`false`/`true` is preserved; no additional public exposure or membership authority
is created. See the [chart contract](../../deploy/collaboration/charts/collaboration/README.md)
and [Compose contract](../../deploy/collaboration/compose/README.md) for examples.

The configured communities must already exist with matching current serving
hosts and Active lifecycle state. Routing does not establish membership,
permissions, accepted NIP05 names, agent ownership or unimplemented signed
subscription ingress. The optional closed socket supports the canonical signed
9040–43 restriction path after genuine NIP42/current-member admission; rendering
the flag is not runtime or live-service proof. PostgreSQL, tenant/RLS, genuine NIP98/event
signatures, current authorization and the canonical shared Redis quota remain
mandatory; zero quota intentionally denies all genuine API attempts. Missing
or malformed configuration refuses startup rather than falling back to an
unprotected route. Rendering these settings does not authorize deployment or
replace current-source genuine service and cleanup evidence.

Signed Nostr restriction reasons use the existing 512 KiB complete EVENT frame
limit, with exact retained source and reason bytes. General moderation sources
keep their 1024-byte public-reason limit. Security audit records retain only the
canonical moderation identifiers and outcome, under their unchanged 16 KiB
encoding and 32-field limits. The signed-reason migration refuses rollback while
longer reasons remain; it never truncates certification or moderation evidence.
