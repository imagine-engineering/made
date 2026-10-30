# Source Inventory: Collaborative Workspace

## Purpose and audit basis

This inventory is the coverage ledger for migrating `projects/buzz` into Zed. A row is complete only when its capability IDs have a disposition in `reuse-audit.md`, requirements coverage, and one or more leaf tasks in `tasks.md`. The audit was made from the source tree on 2026-08-13, including Buzz manifests, Rust modules, SQL migrations, protocol documents, desktop feature directories and Tauri modules, mobile and web source trees, deployment charts, workflows, scripts, tests, examples, and benchmarks. Generated assets, lockfiles, vendored package metadata, and duplicated test fixtures are covered by their owning component rather than listed file-by-file.

Coverage states:

- **Inventoried**: source and observable responsibility identified.
- **Mapped**: canonical Zed disposition exists in `reuse-audit.md`.
- **Approval gate**: implementation depends on an explicit architecture or product decision.

## Visual reference artifacts

The checked-in screenshots are the canonical visual evidence for CAP-036 and the viewport checks in Requirement 4:

| Artifact | Native viewport | SHA-256 | Acceptance use |
| --- | --- | --- | --- |
| `screenshots/screenshot-1.png` | 1930×1262 | `31f179f12708d571c5e3f12ae1c873b96b7733b433bae78aaf691ca247768e71` | Expanded three-pane composition with left navigation, timeline, review pane and status surfaces |
| `screenshots/screenshot-2.png` | 1928×1298 | `1854c0e39d0e0fbe940dcf103e0caf5ede3aecf7ddd57758fdd14f5419572f60` | Collapsed-review, full-width timeline composition with left navigation and status surfaces |

The hashes establish provenance for the specification baseline. An intentional replacement requires updating this table, the visual fixtures and the Requirement 4 approval evidence together.

## Capability catalog

| ID | Capability | Observable Buzz behavior | Principal Buzz evidence | Coverage |
| --- | --- | --- | --- | --- |
| CAP-001 | Signed event domain | Canonical Nostr serialization, event IDs, Schnorr verification, filters, replaceable/addressable heads, tag grammar, kind registry | `crates/buzz-core/src/{event,filter,kind,verification}.rs`, `crates/buzz-sdk` | Mapped |
| CAP-002 | Nostr and HTTP interoperability | NIP-01 WebSocket, NIP-42, NIP-98, NIP-11/05, `/events`, `/query`, `/count`, standard and custom event kinds | `NOSTR.md`, `crates/buzz-relay/src/{protocol,router,nip11}.rs`, `api/bridge.rs` | Mapped |
| CAP-003 | Communities and multitenancy | Host-derived community binding precedes authentication; row, cache, pub/sub, search, media, Git, workflow, audit, and admin paths fail closed | `docs/multi-tenant-{relay,conformance}.md`, `crates/buzz-core/src/tenant.rs`, `crates/buzz-relay/src/tenant.rs` | Mapped; ADR-001 |
| CAP-004 | Relay lifecycle and subscriptions | Authenticated WebSocket connections, REQ/COUNT, EOSE/CLOSED/OK, bounded frames, subscription registry, local and Redis fan-out | `crates/buzz-relay/src/{connection,subscription,handlers}.rs` | Mapped |
| CAP-005 | Persistence and projections | Partitioned Postgres event log plus channels, membership, threads, DMs, workflows, push, moderation, Git, usage, and derived projections | `crates/buzz-db`, `schema/`, `migrations/` | Mapped; ADR-001 |
| CAP-006 | Realtime pub/sub | Redis cross-replica fan-out, presence TTL, typing windows, rate-limit/replay stores, cache invalidation, connection control | `crates/buzz-pubsub` | Mapped |
| CAP-007 | Identity and profiles | Human and agent npubs, profiles, status, follows, mute/pin/bookmark/emoji lists, owner attestations, archived identities | `buzz-core`, NIP-OA, NIP-IA, desktop `identity-archive`, `profile`, `user-status` | Mapped; ADR-002 |
| CAP-008 | Authentication and authorization | NIP-42/NIP-98, API tokens/scopes, replay protection, NIP-AA virtual membership, channel/role gates, invite admission | `crates/buzz-auth`, relay admission/ingest, NIP-AA | Mapped |
| CAP-009 | Secret and signing-key custody | OS keyring, verified plaintext migration, owner-only fallback file, environment injection for agents, key backup/pairing | desktop Tauri `secret_store.rs`, `identity_storage.rs`, `key_backup.rs`, `app_state_keyring.rs` | Mapped |
| CAP-010 | Channels and membership | NIP-29 groups, open/private/DM/ephemeral/forum/huddle types, roles, invites, join/leave, channel templates/canvas/topic | `buzz-db/src/channel.rs`, relay handlers, desktop `channels`, `channel-templates` | Mapped |
| CAP-011 | Messaging and threads | Timeline messages, edits, deletes, reactions, pins, bookmarks, schedules, replies, thread trees/summaries, pagination and aux closure | desktop `messages`, NIP-CW, relay bridge/thread handlers | Mapped |
| CAP-012 | Direct messages and privacy | Gift-wrapped DMs, DM group lifecycle, per-viewer hide state, owner/participant gates | `buzz-db/src/dm.rs`, NIP-DV, desktop/mobile DM surfaces | Mapped |
| CAP-013 | Read, unread, reminders, drafts | Cross-device encrypted read state, manual-unread overrides, local drafts, reminders/bookmarks, unread projections | NIP-RS, NIP-ER, desktop `home`, `reminders`, channel read-state modules | Mapped |
| CAP-014 | Presence and typing | Ephemeral signed presence/typing events, snapshots, TTL expiry, per-community aggregation | `buzz-core/src/presence.rs`, `buzz-pubsub/src/presence.rs`, desktop/mobile presence | Mapped |
| CAP-015 | Search and discovery | Community-scoped Postgres FTS with privacy-sensitive exclusions, channel/member/project/global search and recent searches | `crates/buzz-search`, migration 0008, desktop/mobile `search` | Mapped |
| CAP-016 | Notifications and push | Desktop notifications, membership notifications, blind capability-gated push leases, APNs/App Attest, durable wake outbox | desktop `notifications`, `crates/buzz-push-gateway`, NIP-PL | Mapped; ADR-005 |
| CAP-017 | Home, inbox, pulse, forum and culture | Activity/inbox projection, notes, forum posts/votes/comments, reminders, custom emoji, feedback and status | desktop `home`, `pulse`, `forum`, `custom-emoji`; mobile equivalents | Mapped |
| CAP-018 | Project grouping | Signed multi-repository projects, project-channel binding, visibility, cross-owner grouping without permission transfer | NIP-MP, `VISION_PROJECTS.md`, desktop `projects` | Mapped |
| CAP-019 | Git forge protocol | NIP-34 repositories/ref state/patches/PRs/issues/status, smart HTTP storage, permissions, NIP-98 credentials, Nostr commit signing | relay Git paths, `git-credential-nostr`, `git-sign-nostr`, NIP-GS | Mapped; ADR-003 |
| CAP-020 | Branch collaboration and review | Branch-as-channel activity, patch/review/approval/CI events, inline diffs and links between conversation and Git state | `VISION_PROJECTS.md`, desktop message diff and project views | Mapped |
| CAP-021 | Agent ACP bridge | Channel mentions become ACP sessions; bounded queue/pool, cancellation, heartbeat, response gates, bring-your-own ACP harness | `crates/buzz-acp` | Mapped |
| CAP-022 | Agent runtime and MCP tools | Minimal ACP agent; MCP shell/read/edit/search/tree/image/todo tools; provider/model auth, compaction and handoff | `crates/buzz-agent`, `crates/buzz-dev-mcp`, `VISION_AGENT.md` | Mapped |
| CAP-023 | Managed agents, personas and teams | Persona packs, public/private projections, teams, catalogs, runtime/model/environment configuration, local lifecycle | `crates/buzz-persona`, desktop `agents`, NIP-AP/PMA | Mapped |
| CAP-024 | Agent memory, snapshot and metrics | Encrypted engrams, core/memory records, managed-agent/team/persona snapshots, encrypted per-turn metrics and local archive | NIP-AE/AM/PMA, desktop `agent-memory`, `local-archive`, Tauri archive | Mapped |
| CAP-025 | Agent observability and semantic activity | Ephemeral encrypted observer frames; semantic message/thought/plan/tool/edit/shell/permission/error render classes; raw rail | NIP-AO, `VISION_ACTIVITY.md`, desktop agent activity render classes | Mapped |
| CAP-026 | Jobs and delegation | Signed request/accept/progress/result/cancel/error events, delegation trees, agent teams and orchestration | kind 43001-43006, CLI/desktop agent surfaces, benchmark orchestra | Mapped |
| CAP-027 | Workflows and approvals | YAML definitions, cron/webhook/event triggers, conditions, step lifecycle, approval grants/denials, durable runs | `crates/buzz-workflow`, relay workflow sink, desktop `workflows` | Mapped |
| CAP-028 | Audit and usage | Per-community hash-chain audit, usage records and agent turn accounting, admin inspection | `crates/buzz-audit`, `buzz-db/src/usage.rs`, NIP-AM | Mapped |
| CAP-029 | Moderation | Reports, bans, timeouts, resolution, local mutes, queue, role enforcement, product feedback | migration 0006, relay moderation handlers, desktop `moderation`, `VISION_MODERATION.md` | Mapped |
| CAP-030 | Retention, deletion and recovery | Event TTL, archive, identity archive, durable whole-community deletion, recovery before irreversible work | `crates/buzz-deletion`, migrations 0007-0011, 0016, 0022-0024, 0029-0030 | Mapped |
| CAP-031 | Media and attachments | Blossom/S3 uploads, authenticated access, MIME/magic-byte validation, thumbnails, bucket index and attachment rendering | `crates/buzz-media`, relay media API, desktop/mobile media code | Mapped |
| CAP-032 | Voice, huddles and transcription | Huddle lifecycle, Opus audio rooms, push-to-talk, TTS, local voice models and transcript channel projection | relay `audio`, `crates/buzz-voice`, desktop Tauri `huddle`, desktop/mobile `huddle` | Mapped; ADR-004 |
| CAP-033 | Pairing and identity transfer | Ephemeral pairing relay, NIP-AB QR/crypto/session flow, CLI and mobile scanner | `buzz-pair-relay`, `buzz-pairing-cli`, `buzz-core/src/pairing`, mobile `pairing` | Mapped |
| CAP-034 | Remote-agent providers | Discovery-only `buzz-backend-*` ABI, hostile-output handling, remote lifecycle, presence-as-status, Kubernetes binding | `docs/remote-agents.md`, `buzz-backend-kubernetes` | Mapped |
| CAP-035 | Relay mesh and shared compute | Iroh/QUIC mesh membership, gossip, fenced wire contract, compute advertisements, remote execution/mesh LLM | `buzz-relay-mesh`, desktop `mesh-compute`, Tauri `mesh_llm`, `VISION_MESH.md` | Mapped; ADR-006 |
| CAP-036 | Native collaborative desktop | Communities/projects/tasks rail; human/agent timeline; review pane; composer; status; settings; local archive and terminal | `desktop/src/app`, 29 desktop feature areas, `screenshots/screenshot-{1,2}.png` | Mapped |
| CAP-037 | Onboarding and workspace selection | Identity/community setup, backup, welcome guide agents, configurable post-onboarding navigation | desktop `onboarding`; target Zed `crates/onboarding` | Mapped |
| CAP-038 | Agent-first CLI | Signed commands for channels, DMs, messages, projects, repos, patches, PRs, agents, workflows, memory, moderation, social and uploads | `crates/buzz-cli` | Mapped |
| CAP-039 | Web client | Invite redemption and NIP-98-authenticated repository browsing/download | `web/src` | Mapped |
| CAP-040 | Mobile client | Communities, auth/pairing, activity, messaging/threads, forum/pulse, profile/presence, search, media and push-oriented lifecycle | `mobile/lib` | Mapped |
| CAP-041 | Administration | Operator CLI, admin web resources, provisioning, membership, archival, deletion and metrics | `buzz-admin`, `admin-web`, relay operator API | Mapped |
| CAP-042 | Entity/deep links and compatibility | `buzz://` and HTTPS entity links, channel/message/project/repo navigation, protocol-aware cards | `docs/buzz-entity-links.md`, desktop/mobile/web deep-link code | Mapped |
| CAP-043 | Build, release and deployment | Compose, Helm, K8s, release candidates, signed canaries, multi-platform packages, schema jobs, health/readiness/metrics | `deploy/`, `.github/workflows`, release scripts | Mapped |
| CAP-044 | Test, conformance and formal evidence | Unit/integration/E2E, two-relay harnesses, protocol fixtures, independent trace checker, TLA+/Tamarin/Spthy models, performance benchmarks | `buzz-test-client`, `buzz-conformance`, `docs/spec`, `TESTING.md`, `benchmarks`, `perf` | Mapped |
| CAP-045 | Migration and local archive | Legacy identity/team/config/archive migrations, event sync, local metrics/transcript archive and cutover scripts | desktop Tauri `migration`, `archive`, `event_sync`; `scripts/cutover` | Mapped |

## Rust workspace components

Every Buzz workspace package is assigned below. A package may serve several capabilities; coverage is by capability rather than by copying the package boundary.

| Buzz package | Responsibility | Capability IDs | Expected final disposition |
| --- | --- | --- | --- |
| `buzz-core` | Pure event, tenant, network, pairing, identity and protocol rules | CAP-001-004, CAP-007, CAP-010-014, CAP-018-020, CAP-023-030, CAP-033 | Split into Zed-owned domain/protocol modules; retire package name after parity |
| `buzz-sdk` | Typed signed-event builders and mentions | CAP-001-002, CAP-007-020, CAP-023-030 | Retain as Nostr compatibility adapter, then rename under Zed ownership |
| `buzz-relay` | WebSocket/HTTP ingest, read, fan-out, Git/media/audio/workflow orchestration | CAP-002-006, CAP-008-020, CAP-027-032, CAP-041, CAP-043 | Temporary service boundary; consolidate into Zed collaboration deployment after ADR-001 |
| `buzz-db` | Postgres event store and projections | CAP-003, CAP-005, CAP-010-020, CAP-024, CAP-027-030 | Port migrations/projections; one owner per aggregate |
| `buzz-auth` | NIP auth, scopes, access and replay | CAP-007-008 | Merge with Zed service auth behind explicit Nostr adapter |
| `buzz-pubsub` | Redis realtime/presence/typing | CAP-006, CAP-014 | Port missing semantics into collaboration service; retire duplicate connection registry |
| `buzz-search` | Community-scoped FTS | CAP-015 | Port server search projections; native client uses common search result model |
| `buzz-audit` | Hash-chain audit | CAP-028 | Port as lower-level service module |
| `buzz-deletion` | Community deletion state machine | CAP-030, CAP-041 | Port unchanged semantics, adapt storage interfaces |
| `buzz-workflow` | Workflow engine | CAP-027 | Port as non-GPUI domain/service component; complete known stubs before parity claim |
| `buzz-media` | S3/Blossom media | CAP-031 | Retain wire adapter; integrate media domain with Zed HTTP/credentials |
| `buzz-push-gateway` | NIP-PL push executor | CAP-016, CAP-043 | Long-term compatibility service with Zed-owned deployment and domain contracts |
| `buzz-relay-mesh` | Iroh mesh | CAP-035 | Port unique capability; keep transport isolated behind mesh domain interface |
| `buzz-conformance` | Independent relay trace checker | CAP-003, CAP-044 | Retain independent from production dependencies |
| `buzz-datastore-tracing` | Privacy-preserving database tracing macros | CAP-028, CAP-043-044 | Merge into Zed observability conventions |
| `buzz-ws-client` | Shared NIP-42 WebSocket client | CAP-002, CAP-004 | Merge into client transport adapter |
| `buzz-test-client` | Relay integration/E2E harness | CAP-044 | Retain as black-box compatibility suite until final parity |
| `buzz-admin` | Operator CLI | CAP-030, CAP-041 | Merge commands into canonical Zed operator surface; retain command shim during transition |
| `buzz-cli` | Agent-first user/agent CLI | CAP-038, CAP-042 | Merge into Zed CLI namespace; retain `buzz` shim for scripts |
| `buzz-acp` | Relay-to-ACP harness | CAP-021, CAP-025-026, CAP-034 | Adapt to Zed ACP runtime; retain remote launcher compatibility until migrated |
| `buzz-agent` | Minimal ACP agent | CAP-022 | Reuse Zed agent where parity holds; retain external ACP conformance profile |
| `buzz-dev-mcp` | Developer MCP tools | CAP-022 | Map to Zed tools; retain only protocol-compatibility shims |
| `buzz-persona` | Persona pack parser/merge/validation | CAP-023-024 | Port parser into Zed agent configuration owner |
| `sprig` | Bundled ACP/agent/MCP runtime | CAP-021-022, CAP-034 | Retain remote compatibility image, then rebuild from Zed-owned binaries |
| `buzz-backend-kubernetes` | Remote Kubernetes provider | CAP-034, CAP-043 | Port provider ABI and implementation under Zed remote-agent ownership |
| `buzz-pair-relay` | Ephemeral pairing relay | CAP-033 | Retain compatibility service with Zed release ownership |
| `buzz-pairing-cli` | Pairing interop CLI | CAP-033, CAP-038 | Retain as conformance utility or Zed CLI subcommand |
| `git-credential-nostr` | NIP-98 Git credential helper | CAP-019 | Retain compatibility binary; use Zed credential store |
| `git-sign-nostr` | Git signing helper | CAP-019 | Retain compatibility binary; integrate with Zed Git settings |
| `buzz-voice` | Local voice primitives/models | CAP-032 | Merge with Zed audio/voice owners after ADR-004 |
| `countdown-bot` | Example event-driven bot | CAP-010-011, CAP-044 | Port as compatibility example/test fixture |

## Protocol extensions and kind families

| Protocol | Semantics | Capability IDs | Migration obligation |
| --- | --- | --- | --- |
| NIP-AA | Agent authentication via owner attestation and virtual membership | CAP-007-008, CAP-021 | Preserve verification and revocation semantics |
| NIP-AE | Encrypted owner-readable agent engrams | CAP-024 | Preserve coordinates, HMAC slugs, relay-set behavior and encryption |
| NIP-AM | Encrypted durable agent turn metrics | CAP-024, CAP-028 | Preserve privacy gates and exclude from FTS |
| NIP-AO | Ephemeral encrypted agent observability/control | CAP-025 | Preserve non-persistence and owner result gates |
| NIP-AP | Personas, teams, managed agents and shareable catalogs | CAP-023 | Preserve public/private projection and sharing rules |
| NIP-CW | Stable top-level channel windows, aux closure and signed overlays | CAP-011 | Preserve keyset cursor, overlay trust and degradation |
| NIP-DV | Private per-viewer DM visibility snapshot | CAP-012 | Preserve result gating and relay-derived projection |
| NIP-ER | Encrypted author-only reminders | CAP-013 | Preserve due-time behavior and recovery query semantics |
| NIP-GS | Git object signing with Nostr keys | CAP-019 | Preserve Git signing/verification ABI |
| NIP-IA | Relay-scoped identity archival | CAP-007, CAP-030 | Preserve history without conflating archive and access revocation |
| NIP-MP | Cross-owner multi-repository projects | CAP-018 | Preserve metadata-only authority boundary |
| NIP-OA | Owner-to-agent authorization evidence | CAP-007-008, CAP-021-026 | Preserve agent authorship and capability conditions |
| NIP-PL | Encrypted expiring push leases | CAP-016 | Preserve blind wake-only payload and loss-tolerant reconnect |
| NIP-PMA | Private CAS-managed agent aggregate | CAP-023-024 | Keep gated until documented privacy/CAS/backup criteria pass |
| NIP-RS | Encrypted cross-device read state | CAP-013 | Preserve manual-unread durability and mixed-version guards |
| NIP-WP | Workspace/community icon through NIP-11 | CAP-003, CAP-036 | Preserve standard read path and role-gated write |

The authoritative registry currently includes 137 scalar `u32` constants: 133 registered event kinds and four range-boundary constants. They span standard profile/social/list kinds; authentication; agent profiles/engrams/personas/teams/metrics/observer frames; NIP-29/43 membership; moderation and identity archive; channel windows; messages/edits/pins/bookmarks/reminders/diffs/canvas/system summaries; DMs; jobs; forum; workflows/approvals/audit; huddles/media; NIP-34 Git; and projects. Task 1.2 records every constant and protocol document in the checked catalog so future additions cannot escape this ledger.

## Database migrations

| Migration | Behavior introduced | Capability IDs |
| --- | --- | --- |
| 0001 | Initial multitenant event, channel, membership, token, workflow, audit and partition schema | CAP-003, CAP-005, CAP-008, CAP-010-011, CAP-027-028 |
| 0002 | NIP-34 Git repository-name registry | CAP-019 |
| 0003 | Per-community NIP-11 icon | CAP-003, CAP-036 |
| 0004 | GIN event tag index | CAP-011, CAP-015 |
| 0005 | Agent metrics FTS exclusion | CAP-024, CAP-015 |
| 0006 | Moderation reports, bans/timeouts and actions | CAP-029 |
| 0007 | Bounded NIP-RS storage and replacement ordering | CAP-013, CAP-030 |
| 0008 | Positive FTS allowlist for fresh installations | CAP-015 |
| 0009 | Mixed-version NIP-RS database retention guards | CAP-013, CAP-030 |
| 0010 | Exact replay watermark guard replacing 0009 behavior | CAP-013, CAP-030 |
| 0011 | Exact NIP-RS tag-cardinality guards | CAP-013, CAP-030 |
| 0012 | Durable push lease state and wake outbox | CAP-016 |
| 0013 | Generation-scoped push endpoint invalidation | CAP-016 |
| 0014 | Push-lease ciphertext exclusion from FTS | CAP-015, CAP-016 |
| 0015 | Deployment-global push gateway authority | CAP-016 |
| 0016 | Community archival | CAP-003, CAP-030 |
| 0017 | Signed product feedback | CAP-017, CAP-029 |
| 0018 | Durable event-to-push match queue | CAP-016 |
| 0019 | Mesh status retention | CAP-035 |
| 0020 | Versioned join-policy acceptance evidence | CAP-003, CAP-008 |
| 0021 | Commit-time created-at replica fence | CAP-003-005 |
| 0022 | Transactional ephemeral-channel TTL refresh | CAP-010, CAP-030 |
| 0023 | Community-level push match enqueue gate | CAP-016 |
| 0024 | Shared-lock repair for concurrent TTL refresh | CAP-010, CAP-030 |
| 0025 | Use-limited relay invites | CAP-008, CAP-010 |
| 0026 | Replica heartbeat/read freshness | CAP-004-006, CAP-043 |
| 0027 | Channel-to-community lookup covering index | CAP-003, CAP-010 |
| 0028 | Long custom-emoji reactions | CAP-011, CAP-017 |
| 0029 | Durable whole-community deletion control plane | CAP-030, CAP-041 |
| 0030 | Terminal recovery before irreversible deletion work | CAP-030, CAP-041 |

## Desktop and native-host inventory

The React desktop feature areas are: `agent-memory`, `agents`, `channel-templates`, `channels`, `chat`, `communities`, `community-members`, `custom-emoji`, `forum`, `home`, `huddle`, `identity-archive`, `local-archive`, `mesh-compute`, `messages`, `moderation`, `notifications`, `onboarding`, `presence`, `profile`, `projects`, `pulse`, `reminders`, `search`, `settings`, `sidebar`, `terminal`, `user-status`, and `workflows`. Each maps to CAP-007 and CAP-010 through CAP-040 as applicable; none is a GPUI component boundary.

The Tauri host owns app/keyring state, native WebSocket, relay admission, event conversion/sync, managed-agent lifecycle/config/snapshots/personas/teams, local archive and migrations, media proxy, egress guard, huddle/TTS, mesh LLM, terminal runtime/transport, deep links, notifications, tray/menu/shortcuts, sleep inhibition, rendering workarounds and platform assets. The separate `buzz-terminal` crate is covered by CAP-022, CAP-025, CAP-034, CAP-036 and CAP-045. Task 1.4 must maintain a file-to-capability generated inventory for this surface until Tauri retirement.

Routes cover home, channels and forum posts, new messages, agents, projects, pulse, reminders, settings and workflows. Visual behavior from `screenshots/screenshot-1.png` and `screenshots/screenshot-2.png` is governed by CAP-036 and CAP-037.

## Companion clients and operator surfaces

| Surface | Scope | Capability IDs | Final expectation |
| --- | --- | --- | --- |
| Mobile Flutter | Auth/community storage, pairing, activity/inbox/reminders, channels/threads/reactions/media, forum/pulse, profiles/status/presence, search, read state and theme | CAP-003, CAP-007-017, CAP-031, CAP-033, CAP-040, CAP-042 | Continue interoperating during server migration; move branding/build ownership only after protocol parity |
| Web React | Invite redemption, repository listing/detail/blob browsing/download, NIP-98 signer and relay client | CAP-002, CAP-008, CAP-019, CAP-039, CAP-042 | Preserve URLs and wire behavior; may remain a web client, not embedded desktop UI |
| Admin web | Relay/community/member/invite/deletion/metrics resources | CAP-003, CAP-029-030, CAP-041 | Repoint to canonical operator APIs; preserve least privilege |
| Agent-first CLI | 54+ subcommands across agents, channels, DMs, messages, social, projects/repos/patches/PRs/issues, workflows, memory, moderation and media | CAP-038 | Preserve scripts through compatibility aliases and exit codes |

## Operations, release, formal methods and examples

- `deploy/charts/buzz` covers relay, pairing relay, Postgres/object storage integration, ingress/HTTPRoute, autoscaling, disruption budgets, persistence, service accounts, monitors and Git storage.
- `deploy/charts/buzz-push-gateway` covers gateway deployment, migrations, network policy, PDB, monitoring and production alerts.
- `deploy/compose` and `deploy/local` cover self-hosted and local/HA startup.
- Eighteen GitHub workflows cover CI, Docker/Helm, desktop/mobile candidates, signed canaries, Sprig/provider images, push gateway, mesh lifecycle, promotion and release tagging.
- Scripts cover schema cutover/backfill, release contracts, build bundling, isolated relay/test setup, Git permissions, screenshots, seed/reset, maintenance, mobile/desktop promotion and live smoke tests.
- `docs/spec` contains Git-on-object-storage and multitenant TLA+/Spthy models; `buzz-conformance` is deliberately independent of production code.
- `benchmarks/harbor-buzz-orchestra` exercises multi-agent orchestration and container substrates; `perf/RELAY_BUS_SCALING.*` models fan-out scaling.
- `examples/countdown-bot` and `examples/meadow-core` cover bot, persona/team, plugin and skill interoperability.

All are mapped by CAP-034-035 and CAP-038-045. Release/deployment tasks are handoff operations and are not executed by this specification.

## Known source gaps that parity must not conceal

The source itself records incomplete behavior: production rate limiting is not implemented; huddle recording/per-track publishing is absent; approval gates are not wired end-to-end; workflow `send_dm` and `set_channel_topic` actions are stubs; SQL queries lack an offline cache. These are parity requirements to complete or explicitly accept, not evidence for excluding the capability.

## Inventory completion rule

Before `projects/buzz` can be declared reference-only, CI must regenerate the crate, kind, NIP, migration, desktop-feature, client-route and deployment inventories and fail when a new source component lacks a CAP ID, ownership row, requirement reference and leaf task or approved not-applicable record.
