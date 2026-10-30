# Collaborative Workspace operational limits

## Purpose and policy

This registry makes acceptance criteria 8.4, 19.3 and 19.5 measurable for CAP-004, CAP-006, CAP-028 and CAP-043. It consolidates the limits required by the Collaborative Workspace threat models and accepted ADRs without changing their ownership or migration strategy.

Every row below is normative for the first production-capable Zed deployment unless its value is explicitly described as a compatibility ceiling. A lower deployment limit is allowed when advertised through the appropriate protocol/configuration and validated at startup. Raising a hard security or compatibility ceiling requires security review, conformance evidence and a versioned configuration change; an environment variable alone cannot bypass it.

Metric names are target Zed names for Tasks 44.5 and 45.4–45.5. All metric labels are closed enums plus bounded deployment/region/service identifiers. Community IDs, user/agent keys, channel/job/event/request IDs, repository paths, URLs, model prompts/output, message/media content, endpoint tokens and secrets are prohibited as labels. Per-community diagnosis uses access-controlled logs/traces with pseudonymous correlation handles and retention, never public Prometheus labels.

Alerts use the following conventions:

- **Page** means user traffic is unsafe, silently divergent or broadly unavailable and requires immediate operator action.
- **Warn** means a bounded failure budget is being consumed and requires investigation during the support window.
- A ratio is evaluated only with at least 100 relevant operations in the stated window unless the row says any occurrence.
- Hard-limit rejection is an expected safe outcome; the alert detects abuse, bad clients, misconfiguration or insufficient capacity rather than asking operators to weaken the limit.

## Limit registry

### Connections, protocol and synchronization

| Limit ID | Enforced limit and failure behavior | Canonical owner | Metric and alert threshold | Focused verification |
| --- | --- | --- | --- | --- |
| OL-CON-01 | At most 10,000 live collaboration WebSockets per replica; additional upgrades return typed capacity unavailable before allocating connection state. | `collab::transport` connection semaphore | `collab_connections_active`; warn at ≥8,000 for 10m, page at ≥9,500 for 5m or capacity rejects >1% for 5m | Tasks 22.1, 44.3, 45.4 |
| OL-CON-02 | At most 1,024 concurrent inbound handlers per replica; work waits only in the bounded dispatcher and otherwise receives overload. | `collab::transport` dispatcher | `collab_handlers_active`, `collab_handler_rejections_total`; warn at ≥80% for 5m, page at ≥95% for 5m | Tasks 13.2, 22.1, 45.4 |
| OL-CON-03 | Per-connection outbound queue holds at most 1,000 frames; slow consumers are closed with a typed slow-client reason, never backed by an unbounded queue. | `collab::transport` connection writer | `collab_outbound_queue_depth`, `collab_slow_client_closes_total`; warn p99 depth ≥800 for 5m, page when closes exceed 1% for 5m | Tasks 22.1, 22.3, 45.4 |
| OL-CON-04 | A WebSocket message/frame is at most 512 KiB and event content at most 256 KiB before decode/domain mutation. NIP-11 advertises the effective value. | `nostr_compat` framing plus `collab` ingress | `collab_frame_rejections_total{reason}`; warn above 1 rejection/s for 5m, page if an oversized frame is accepted in conformance | Tasks 11.6, 13.2, 45.1, 45.4 |
| OL-CON-05 | REQ/COUNT subscription IDs are at most 256 bytes, each request contains at most 10 filters and a connection owns at most 1,024 subscriptions. | `nostr_compat` codec and `collab::subscription` | `collab_protocol_rejections_total{reason}`, `collab_subscriptions_active`; warn p99 active ≥900 for 10m or rejection ratio >5% for 10m | Tasks 11.8, 22.2, 45.4 |
| OL-CON-06 | Unauthenticated connections complete NIP-42 within 5s; expiration or shared replay/current-policy unavailability closes the connection. | `collab::auth` | `collab_auth_duration_seconds`, `collab_auth_failures_total{reason}`; warn p95 >2s for 10m, page timeout ratio >1% for 5m | Tasks 14.2, 22.1, 45.2 |
| OL-CON-07 | Realtime presence/typing is lossy and bounded: state expires after 60s without refresh and per-principal publication is accepted at no more than 2 updates/s with burst 10. | `collab::realtime` derived-state store | `collab_ephemeral_entries`, `collab_ephemeral_rate_limited_total`; warn expiry-lag p99 >15s for 10m, page any entry age >120s | Tasks 21.1–21.3, 22.3, 45.4 |
| OL-CON-08 | Client reconnect backoff is full-jitter 250ms–30s, resets only after 60s stable connectivity and never exceeds one concurrent connect attempt per account/community endpoint. | native collaboration transport/client lifecycle | `collab_reconnect_attempts_total{outcome}`, `collab_reconnect_delay_seconds`; warn attempt failure >20% for 10m, page >50% for 5m | Tasks 13.4, 21.6, 43.8, 45.4 |
| OL-CON-09 | Relay graceful drain is 30s inside a 60s pod grace period; reconnect jitter is configurable from 0–20s and any undrained connection is closed visibly at the deadline. | `collab` service lifecycle and deployment | `collab_drain_active`, `collab_drain_forced_closes_total`; warn any forced close, page drain exceeds 30s or pod grace is <60s | Tasks 22.1, 44.3, 45.4 |

### Durable queues, projections, search and read state

| Limit ID | Enforced limit and failure behavior | Canonical owner | Metric and alert threshold | Focused verification |
| --- | --- | --- | --- | --- |
| OL-DAT-01 | Accepted commands create durable projection/outbox responsibility atomically. Oldest ready-item age is ≤30s normally and ≤5m during declared degraded operation; no item remains claimed past twice its lease. | `collab` command/outbox repositories | `collab_outbox_oldest_seconds`, `collab_outbox_stuck_claims`; warn >30s for 5m, page >300s or any stuck claim for 5m | Tasks 15.4, 16.3, 22.13, 44.5 |
| OL-DAT-02 | Projection drift is zero by source ID/version after a worker pass; lag is ≤10s normally and ≤60s during recovery. Drift blocks compatibility removal and unsafe cutover. | canonical projection workers | `collab_projection_lag_seconds`, `collab_projection_drift_total`; warn p99 >10s for 10m, page >60s or drift >0 for 5m | Tasks 16.3, 35.5, 44.5, 46.3 |
| OL-DAT-03 | Replica freshness heartbeat is every 5s and expires after 15s. A stale replica is removed from readiness/routing and cannot claim authoritative work. | `collab::freshness` plus deployment | `collab_replica_freshness_seconds`; warn >10s, page >15s for any ready replica | Tasks 16.3, 22.3, 44.5 |
| OL-DAT-04 | Per-command dedup/idempotency evidence outlives the maximum client retry/replay window and is retained at least 24h; cleanup never removes a current tombstone/generation floor. | canonical command repositories | `collab_dedup_oldest_seconds`, `collab_duplicate_commands_total`; page any duplicate mutation; warn cleanup age >48h | Tasks 15.4, 22.13, 37.8, 45.3 |
| OL-DAT-05 | Search text is at most 4,096 Unicode scalar values; page size is 1–500 and page index 1–1,000. Larger requests fail before query construction. | `collab::search` query parser/index | `collab_search_rejections_total{reason}`; warn rejection ratio >5% for 10m, page any query exceeding the SQL/engine bound | Tasks 23.2, 23.3, 45.4 |
| OL-DAT-06 | Search p95 latency is ≤500ms and p99 ≤2s at the approved corpus/load; queue wait is ≤250ms p95. Timeout returns partial/unavailable, never cross-tenant or unbounded fallback results. | `collab::search` service | `collab_search_duration_seconds`, `collab_search_queue_seconds`; warn p95 >500ms for 10m, page p99 >2s for 5m | Tasks 23.7, 44.5, 45.4 |
| OL-DAT-07 | Feed/window page defaults to 50 and is capped at 200; thread/search compatibility surfaces may request at most 500 when their contract requires it. Cursor loops, repeated IDs and no-progress pages terminate. | `collaboration_domain` paging plus adapters | `collab_page_items`, `collab_paging_failures_total{reason}`; warn rejection/error ratio >2% for 10m, page any no-progress loop | Tasks 20.3, 20.4, 22.11, 45.4 |
| OL-DAT-08 | Unread/read-state reconciliation converges within 10s online and one successful page after reconnect; override/frontier blobs retain the wire ceilings of 32 KiB and 10,000 entries. | `collaboration_domain::read_state` and Nostr adapter | `collab_read_state_lag_seconds`, `collab_read_state_rejections_total`; warn p99 >10s for 10m, page cross-tenant divergence or accepted oversized blob | Tasks 21.1, 21.6, 45.1, 45.4 |

### Workflow, agent and remote execution

| Limit ID | Enforced limit and failure behavior | Canonical owner | Metric and alert threshold | Focused verification |
| --- | --- | --- | --- | --- |
| OL-EXE-01 | Workflow condition input is ≤4,096 bytes and evaluation wall time ≤100ms; the executor also caps concurrent blocking evaluations at 2×available CPU, minimum 2 and maximum 32. | `collab::workflow` evaluator | `collab_workflow_condition_seconds`, `collab_workflow_condition_rejections_total`; warn p95 >50ms, page any >100ms completion or saturated pool >5m | Tasks 34.3, 34.8, 45.5 |
| OL-EXE-02 | A workflow delay is ≤270s, default step timeout 300s and declared timeout ≤600s. Timeout cancels/records the attempt; it never reports success. | `collab::workflow` executor | `collab_workflow_step_seconds`, `collab_workflow_timeouts_total`; warn timeout ratio >2% for 10m, page any timed-out success | Tasks 34.3–34.5, 34.8, 45.5 |
| OL-EXE-03 | Inbound and outbound webhook bodies are ≤1 MiB; outbound connect+response timeout is 10s, redirects/proxies are disabled and response capture truncates at 1 MiB. | workflow ingress/action adapters | `collab_webhook_duration_seconds`, `collab_webhook_rejections_total{reason}`; warn timeout >2% for 10m, page any private-address/redirect escape | Tasks 34.2, 34.4, 34.8, 45.2 |
| OL-EXE-04 | Workflow ready queues are bounded to 1,000 pending runs per community and 10,000 per deployment; per-community execution concurrency defaults to 16 and per-definition concurrency to 4. Admission is typed queued/capacity unavailable. | `collab::workflow` scheduler | `collab_workflow_queue_depth{scope}`, `collab_workflow_queue_oldest_seconds`; warn ≥80% or oldest >30s for 10m, page at cap or oldest >5m | Tasks 34.5, 34.8, 34.10, 45.5 |
| OL-EXE-05 | Provider configuration contains ≤20 flat scalar fields and ≤64 KiB; provider stdout ≤1 MiB, stderr ≤64 KiB, inspect/info ≤10s and deploy ≤600s. | `agent` remote-provider adapter | `agent_provider_rejections_total{reason}`, `agent_provider_duration_seconds{operation}`; warn duration p95 >80% ceiling, page accepted oversize/secret or nonzero-success | Tasks 29.3, 33.2, 33.6, 45.2 |
| OL-EXE-06 | Each canonical agent turn has one executor, one active permission request and one terminal outcome. Cancellation reaches the process/provider within 2s and all children/resources finish within 10s locally or the provider's declared ≤30s remote grace. | ACP/agent session and remote execution owner | `agent_cancellation_seconds`, `agent_duplicate_executors_total`, `agent_orphan_resources`; warn cancel p95 >2s, page duplicate executor or orphan >30s | Tasks 28.2, 28.3, 33.5, 33.6, 45.5 |
| OL-EXE-07 | Delegation depth is ≤8, active children per job ≤16 and total active jobs per community ≤256 by default. Exceeding a bound requires explicit queueing, never recursive unbounded spawn. | canonical jobs/delegation repository | `agent_jobs_active`, `agent_delegation_rejections_total{reason}`; warn ≥80% for 10m, page at cap for 5m or accepted depth >8 | Tasks 31.2, 31.3, 31.6, 45.5 |

### Push delivery

| Limit ID | Enforced limit and failure behavior | Canonical owner | Metric and alert threshold | Focused verification |
| --- | --- | --- | --- | --- |
| OL-PUS-01 | Active push lease TTL ≤30 days, ciphertext ≤65,536 bytes, plaintext ≤32,768 bytes, active leases/installation ≤16 and subscriptions/lease ≤16. | `collab::push` lease admission | `collab_push_lease_rejections_total{reason}`; warn rejection ratio >5% for 10m, page accepted oversize/overcount | Tasks 22.6, 22.11, 45.2 |
| OL-PUS-02 | Accepted-event matcher batches at most 64 events and wake claims at most 16 jobs per community. Idle polling backs off from 250ms to 2s. | `collab::push` matcher/outbox | `collab_push_match_batch`, `collab_push_wake_queue_oldest_seconds`; warn oldest >30s for 5m, page >5m | Tasks 22.7, 22.8, 22.12, 45.4 |
| OL-PUS-03 | A delivery claim lasts 30s, event usefulness ≤1h and delivery has at most 8 attempts. Expired/exhausted jobs become terminal and cleanup is idempotent. | push outbox executor | `collab_push_claim_age_seconds`, `collab_push_attempts`, `collab_push_exhausted_total`; warn claim >30s or attempts ≥6, page claim >60s for 5m | Tasks 22.8, 22.12, 22.13, 45.4 |
| OL-PUS-04 | Public gateway JSON is ≤8,192 bytes with concurrency ≤256 and request timeout 20s. Overload/timeout uses the closed temporary-unavailable response. | Zed push gateway HTTP admission | `push_gateway_inflight`, `push_gateway_request_seconds`; warn inflight ≥205 or p95 >10s for 5m, page ≥243 or p99 >20s | Tasks 22.9, 22.10, 22.12 |
| OL-PUS-05 | App Attest input ≤16 KiB, assertion ≤1 KiB, opaque grant ≤4 KiB, endpoint hex ≤512 bytes and challenge lifetime 300s. | push gateway authority | `push_gateway_admission_total{result}`; warn invalid/replay >5% for 10m, page any accepted oversize or non-monotonic counter | Tasks 22.9, 43.7, 45.2 |
| OL-PUS-06 | APNs timeout is 15s, expired provider token refresh occurs once, and `Retry-After` is clamped to 1–3,600s. Retries are bounded by OL-PUS-03. | APNs adapter | `push_gateway_provider_seconds`, `push_gateway_delivery_total{outcome}`; warn retry/configuration outcomes >2% for 10m, page configuration fault for 10m | Tasks 22.8–22.10, 45.4 |
| OL-PUS-07 | Default endpoint quota is 10 deliveries/10s, configurable only within window 1–86,400s and count 1–10,000. Admission reserves quota atomically across replicas. | push gateway authority store | `push_gateway_quota_rejections_total`; warn >1% for 10m, page any admitted request beyond configured quota | Tasks 22.9, 22.12, 45.4 |

### Media and huddles

| Limit ID | Enforced limit and failure behavior | Canonical owner | Metric and alert threshold | Focused verification |
| --- | --- | --- | --- | --- |
| OL-MED-01 | Upload byte limit is configured by media class and reserved before body read; first release defaults are 25 MiB image/audio and 500 MiB video, never above deployment storage/quota policy. | `collab::media` quota/admission | `collab_media_upload_bytes`, `collab_media_quota_rejections_total`; warn reserved capacity ≥80% for 10m, page at 100% or unreserved object write | Tasks 38.1, 38.2, 45.4 |
| OL-MED-02 | Images decode to ≤25,000,000 pixels; videos are ≤600s and ≤3840×2160; MP4 parsing allows ≤1,024 top-level atoms, 100,000 boxes and depth 32. | media validation adapter | `collab_media_validation_total{result}`; warn rejection ratio >5% for 10m, page any accepted structural oversize | Tasks 38.2, 38.7, 45.2 |
| OL-MED-03 | Media validation/thumbnail concurrency is at most min(8, 2×CPU), each operation ≤30s and decoded working memory ≤256 MiB. Timeout/resource exceed cancels and removes temporary/derived files. | media worker pool | `collab_media_workers_active`, `collab_media_validation_seconds`, `collab_media_temp_orphans`; warn ≥80% for 10m or p95 >15s, page timeout/orphan >5m | Tasks 38.2, 38.5, 38.7, 45.4 |
| OL-HUD-01 | Legacy huddle admission is ≤25 peers/room, ≤8,192-byte WebSocket messages and ≤4,096-byte Opus frames; native room policy may be stricter but not wider during compatibility. | huddle lifecycle and Buzz gateway adapter | `collab_huddle_peers`, `collab_huddle_frame_rejections_total`; warn room ≥20 peers, page accepted oversize or >25 legacy peers | Tasks 39.2, 39.3, 39.8, 45.2 |
| OL-HUD-02 | Per-peer media queue holds 8 frames (~160ms) and reliable control queue 32 entries. Media drops on pressure; control overflow disconnects/resynchronizes instead of dropping state. | media bridge | `collab_huddle_media_drops_total`, `collab_huddle_control_overflow_total`; warn media drops >1% for 5m, page any unrecovered control overflow | Tasks 39.3, 39.8, 45.4 |
| OL-HUD-03 | Heartbeat interval is 30s and disconnect occurs after 3 missed pongs; NIP-42 admission remains 5s. Stale room generations cannot refresh liveness. | huddle gateway/session owner | `collab_huddle_heartbeat_age_seconds`, `collab_huddle_disconnects_total{reason}`; warn age >60s, page ready connection age >90s | Tasks 39.2, 39.8, 45.4 |
| OL-HUD-04 | STT queue holds ≤50 ~100ms batches, a speech segment ≤30s at 16kHz, and TTS chunks contain ≤50 tokens. Overload drops/terminates with a visible incomplete transcript marker. | native voice workers | `collab_voice_queue_depth`, `collab_voice_segment_seconds`, `collab_voice_drops_total`; warn queue ≥40 for 5m, page cap for 1m or unlabeled incomplete output | Tasks 39.5, 39.6, 39.8, 45.4 |
| OL-HUD-05 | Imported voice audio is ≤25 MiB, 2–30s and 8–96kHz; acquisition and model archives obey hash, byte and traversal bounds before atomic installation. | native voice/model store | `collab_voice_import_total{result}`; warn failure >5% for 10m, page accepted invalid bounds/hash/path | Tasks 39.5, 39.6, 45.2 |

### Relay mesh and shared compute

| Limit ID | Enforced limit and failure behavior | Canonical owner | Metric and alert threshold | Focused verification |
| --- | --- | --- | --- | --- |
| OL-MSH-01 | Mesh streams use a 16 MiB hard frame ceiling; compute-control frames use a stricter 1 MiB ceiling, invite tokens ≤64 KiB, bootstrap addresses ≤8 and endpoint transports ≤16. | `remote::mesh::protocol` | `remote_mesh_frame_rejections_total{reason}`; warn >1/s for 5m, page any accepted oversize | Tasks 41.1, 41.5, 45.2 |
| OL-MSH-02 | Relay ready record refresh is 15s and expiry 45s; gossip heartbeat 2s, reconcile 5s and compute advertisements expire no later than 60s. Expired state only removes candidates. | `remote::mesh` membership/advertisement | `remote_mesh_record_age_seconds`, `remote_mesh_stale_records`; warn age >30s, page a selected record >45s relay or >60s compute | Tasks 41.1, 41.2, 41.5 |
| OL-MSH-03 | Shared compute is disabled unless every policy/consent gate is true; no queue or listener exists merely because relay mesh is enabled. Any third-party provider count must remain zero. | deployment/community/user/device policy | `remote_mesh_eligible_nodes`, `remote_mesh_policy_rejections_total{reason}`; page any execution with a false gate or third-party provider | Tasks 41.2, 41.3, 44.3, 45.2 |
| OL-MSH-04 | Default per-requester mesh concurrency is 2, queue cap 16 and per-community queue cap 256; weighted-fair wait p95 ≤30s with max/min normalized service ratio ≤1.25 under equal demand. | `remote::mesh::scheduler` | `remote_mesh_queue_depth`, `remote_mesh_wait_seconds`, `remote_mesh_fairness_ratio`; warn p95 >30s or ratio >1.25 for 10m, page cap or ratio >1.5 | Tasks 41.3, 41.5, 45.5 |
| OL-MSH-05 | Executor/resource leases expire ≤30s without renewal; cancellation reaches the serving runtime ≤2s and cleanup ≤30s. Stale/wrong-generation results are always rejected. | canonical jobs plus mesh scheduler/runtime | `remote_mesh_lease_age_seconds`, `remote_mesh_cancel_seconds`, `remote_mesh_stale_results_total`; warn cancel p95 >2s, page lease >60s, cleanup >30s or accepted stale result | Tasks 33.5, 41.3, 41.5, 45.5 |
| OL-MSH-06 | Each lease fixes prompt/output token/byte, wall/idle, CPU/GPU/memory/disk/network and model-cache ceilings; admission requires all finite nonzero values. No production default is inferred from an advertisement. | serving runtime and deployment policy | `remote_mesh_resource_usage_ratio{resource}`, `remote_mesh_resource_cancellations_total{resource}`; warn ≥80% for 5m, page >100% or missing enforced ceiling | Tasks 41.2, 41.3, 44.3, 45.5 |

### Migration, compatibility, health, logs and telemetry

| Limit ID | Enforced limit and failure behavior | Canonical owner | Metric and alert threshold | Focused verification |
| --- | --- | --- | --- | --- |
| OL-OPS-01 | Liveness checks process progress only; readiness checks every required authority, schema/compatibility floor, migration state and kill switch. Probe timeout is 3s, readiness period 5s with 3 failures, liveness period 10s with 3 failures and startup budget 120s. | service health owner and deployment | `collab_readiness{cause}`, `collab_readiness_failures_total{cause}`; page readiness false >5m or any ready instance with failed authority/schema | Tasks 22.10, 44.3, 44.5 |
| OL-OPS-02 | Migration batches commit at most 1,000 records or 5s of work; checkpoints persist after every batch. A batch retries at most 5 times with 1–30s jittered backoff before halting visibly. | Zed migration runner | `collab_migration_checkpoint_age_seconds`, `collab_migration_retries_total`; warn no progress >5m, page >15m or retry exhaustion | Tasks 17.3–17.6, 44.4, 45.3 |
| OL-OPS-03 | Shadow comparison lag is ≤60s and unexplained divergence is exactly zero before cutover/removal. Compatibility observations are retained at least 7 days and through one rollback window. | compatibility/migration controller | `collab_shadow_lag_seconds`, `collab_shadow_divergence_total{class}`; warn lag >60s for 10m, page any unexplained divergence | Tasks 46.1–46.4, 48.2 |
| OL-OPS-04 | Migration/cutover halts before mutation when source or target compatibility version is unknown; mixed-version support is limited to the explicitly published compatibility window. | migration and protocol version owners | `collab_compatibility_peers{result}`, `collab_cutover_halts_total{reason}`; page any incompatible write or unsupported ready peer | Tasks 43.2, 44.4, 45.1, 47.1 |
| OL-OPS-05 | Rollback/kill-switch acknowledgement is ≤60s; new admissions cease immediately after configuration observation and in-flight work follows its documented drain/cancel ceiling. | deployment controller and each service owner | `collab_kill_switch_age_seconds`, `collab_admissions_after_disable_total`; page any post-disable admission or acknowledgement >60s | Tasks 44.3, 44.7, 45.3 |
| OL-OPS-06 | Structured operational logs retain 14 days by default, audit logs follow Requirement 15 policy, and sensitive values are redacted before emission. One log record is ≤64 KiB and repeated identical errors are sampled after 100/min/service. | service logging layer and operator policy | `collab_log_redactions_total{class}`, `collab_log_dropped_total{reason}`; page any seeded secret/content canary; warn drop/sampling >1% for 10m | Tasks 35.5, 44.5, 45.2 |
| OL-OPS-07 | Metrics endpoints bind only to private health/metrics listeners and scrape every 30s with 15s timeout. Cardinality is ≤10,000 active series/service and every label value comes from a reviewed closed set. | deployment observability | `collab_metrics_series`, `collab_metrics_scrape_failures_total`; warn ≥8,000 series or scrape failure >5% for 10m, page public exposure or ≥10,000 | Tasks 44.3, 44.5, 45.2 |
| OL-OPS-08 | Client metrics and diagnostics remain disabled by default. While disabled, queued client events/identity are cleared and `/telemetry/events` request count is exactly zero; local logs and server metrics remain active. | existing `TelemetrySettings` and `client::telemetry::Telemetry` | deterministic client fake-HTTP counter; page release gate on any request while disabled. No server metric may claim client consent. | Tasks 44.5, 45.2, 48.2 |
| OL-OPS-09 | Operational status and errors expose no content or secret, use a closed reason alphabet and truncate user-visible diagnostic detail to 4 KiB after redaction. Raw protocol/activity detail remains permission-gated. | domain error/redaction layer and GPUI presentation | `collab_redaction_failures_total{surface}`; page any seeded private fixture match, warn unknown reason code >0 | Tasks 18.6, 44.5, 45.2 |
| OL-OPS-10 | Backup/recovery drills prove database/object/key/checkpoint recovery at least every 90 days; recovery-point objective ≤15m and recovery-time objective ≤4h for production. Failure keeps services unready and blocks destructive cleanup. | deployment operator under Zed runbook | `collab_backup_age_seconds`, `collab_restore_drill_age_seconds`; warn backup age >15m or drill >75d, page backup >30m or drill >90d | Tasks 44.7, 45.3, 48.4 |

## Required dashboards and stop signals

Task 44.5 must render four access-controlled dashboards from the registry rather than recreate product state:

1. **Admission and realtime:** OL-CON-01–09, current ready replicas, protocol/version rejects, active connections/subscriptions, queue depth, reconnects and slow-client closures.
2. **Durable collaboration:** OL-DAT-01–08 and OL-PUS-01–07, projection/outbox/read/search/push age, retry, expiry, suppression and poison state.
3. **Execution and media:** OL-EXE-01–07, OL-MED-01–03, OL-HUD-01–05 and OL-MSH-01–06, with resource, cancellation, fairness and cleanup signals but no content or stable user/job labels.
4. **Migration and release:** OL-OPS-01–10, schema/checkpoint/shadow/version/rollback/backup state and every active kill switch.

The following are automatic stop/rollback signals rather than informational alerts:

- any cross-community authorization, routing, search, media, push, workflow, huddle or mesh result;
- accepted replay/stale generation, duplicate canonical execution or unexplained shadow divergence;
- secret/content canary in logs, metrics, push payloads, discovery or protocol errors;
- ready service with unavailable canonical authority, incompatible schema/version or disabled security control;
- queue/claim/lease state beyond its page boundary without successful reconciliation;
- client telemetry HTTP while the authoritative setting is disabled; or
- inability to stop new admissions and preserve one canonical owner during rollback.

## Verification and change control

- Tasks 22.12 and 45.4 run connection, subscription, fan-out, paging, search and push gates against OL-CON, OL-DAT and OL-PUS.
- Tasks 34.8, 41.5 and 45.5 run workflow, agent and mesh resource/cancellation/fairness gates against OL-EXE and OL-MSH.
- Tasks 38.7, 39.8 and 45.2 run media, huddle, tenant, redaction and protocol negative gates against OL-MED, OL-HUD and the relevant hard ceilings.
- Tasks 44.3–44.5 prove configuration ranges, private metrics networking, readiness dependencies, alerts and dashboards. A chart default is not approved merely because Buzz used it.
- Tasks 45.3, 46.1–46.6 and 48.4 prove migration, rollback, shadow and recovery limits before source retirement.
- Task 48.4 may declare final operational sign-off only when every row has enforcement evidence and its metric/alert appears in the operational artifact or is explicitly verified as a compile-time/client-only invariant.

Changing a limit requires the canonical owner, threat-model impact, compatibility effect, metric and alert to change in the same review. Relaxing a hard bound or removing a stop signal requires explicit security approval; measured tuning inside an already approved configurable range does not.
