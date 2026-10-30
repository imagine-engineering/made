# Requirements: Collaborative Workspace and Buzz Consolidation

## Problem

Zed has mature native editor, project, Git, agent, ACP, remote-development and collaboration primitives. Buzz has a broader signed-event collaboration product and operational stack, but it is nested as a separate Rust/React/Tauri/mobile/web system. Shipping both unchanged would split identity, projects, conversations, agents, diffs and operational ownership. The product needs a native GPUI collaborative presentation and complete Buzz capability parity while converging each responsibility on one canonical Zed owner.

## Scope

### In scope

- Every capability identified as CAP-001 through CAP-045 in `source-inventory.md`.
- A first-shippable native vertical slice using existing Zed projects, ACP threads and diffs, with `screenshots/screenshot-1.png` and `screenshots/screenshot-2.png` as its canonical visual references.
- Complete protocol, service, data, client, operational and source-retirement migration.
- Compatibility with existing signed events, identities, deployments, CLI scripts, mobile/web clients and stored data unless an approval gate explicitly authorizes a break.
- Completing documented Buzz stubs or defects before declaring parity.
- Collaborative ACP presentation reuses the canonical `AgentThread`/`ThreadView` entry renderer; channel presentation remains the authorized `collab_ui::MessageTimeline`. Visual evidence distinguishes deterministic GPUI content raster from host-owned macOS window chrome and never synthesizes traffic-light controls.

### Out of scope

- Product capabilities not present in Buzz or required to consolidate it with Zed.
- Replacing Zed's editor, language tooling, local project/worktree model or Git implementation with Buzz equivalents.
- Embedding the Buzz React/Tauri desktop in Zed.
- Executing production deployment, deletion or irreversible cutover operations while creating or implementing this specification.

Restored execution state must satisfy existing failure, persistence and truthful
status criteria: legacy or unrecorded outcomes remain Unknown; saved in-flight
work is Interrupted without claiming a live process; completed, cancelled and
failed outcomes stay distinct. Fresh canonical execution overrides its restored
snapshot, and older turn completions cannot change the newer turn's outcome or
entries. Component tests and database evidence do not replace the same-journey
native Delta/Made restart comparison.

## Glossary

- **Canonical owner**: the only component permitted to author authoritative state for an aggregate after cutover.
- **Compatibility adapter**: a boundary that preserves an external wire, file or process contract while translating to/from canonical state.
- **Community**: the tenant/workspace boundary selected from trusted request context before authentication or data access.
- **Collaborative Workspace**: Zed's native GPUI collaboration-oriented workspace presentation; it does not create a second project or data model.
- **Parity**: all inventoried behavior passes reuse evidence or implementation/conformance tests, including security and failure semantics.
- **Signed event log**: the immutable interoperable record for Nostr-authored collaboration events; relational projections derived from it are not independent authorities.

## Delta, Buzz and Fractal authority (2026-10-03)

The real Delta application is the source of truth for UI structure, views,
navigation, interactions and user journeys. Buzz defines collaboration capabilities
and domain behavior, including identity, community, channel, membership and
authorization. Fractal defines typography, icons, colors, spacing and control
styling. This supersedes the earlier assignment of interaction authority to
screenshots alone. Actual native observations, assumptions and unavailable flows
are separated in [the Delta parity matrix](delta-parity-matrix.md).

The latest user-supplied conversation reference governs the review-collapsed state;
the earlier Image 2 governs the corresponding review-expanded composition. Preserve
the checked-in references. Conflicts require recording the actual Delta state and
explicit specification reconciliation. In particular, a separate composer band is
an implementation discrepancy to investigate, not an approved Delta adaptation.
Delta interaction patterns must use existing Made owners and Buzz authorization;
they do not authorize duplicate identity or membership authorities or fake flows.

### Journey acceptance gate (2026-10-04)

Before implementing a surface or journey, inspect its corresponding flow in the
verified real Delta application and record placement, entry point, states,
interactions, keyboard/focus behavior and persistence. Map the observation to Buzz
capabilities and the existing canonical Made owner, with concrete acceptance and
evidence requirements in the tracked parity checklist. If reference access is
unavailable, mark only that comparison unverified and continue independent work.

Acceptance requires comparing the same journey and state in Delta and a verified
repository-built Made executable. Component functionality, fixture screenshots,
visual placement, native interaction and authenticated service evidence are
separate claims. Reusing an existing component does not approve its inherited
placement or workflow. Reopen prior UI acceptance supported only by component
proof; retain that proof for its actual scope.

The conversation terminal must use the canonical terminal/PTY inside the inline
conversation flow. A native terminal dock or static command-output card cannot
satisfy this requirement. The separately approved selected-dock behavior remains
valid, but is not inline-terminal parity. Detailed observations and acceptance
checks are in [the Delta parity matrix](delta-parity-matrix.md#inline-terminal-journey).

## Historical Fractal reconciliation (2026-09-07)

The user confirmed on 2026-09-10 that the pinned `fractal-ui` contract remains
authoritative for typography, icons and density, while current Image 2 guides
layout and behavior. The original screenshots remain composition references for hierarchy,
pane relationships and collaborative workflows. Typography, semantic icons,
spacing, density, colours, control geometry and interaction states follow the
shared Fractal adaptation. Explicit user appearance preferences still win. The
original images are preserved; palette-dependent exact pixel equality is diagnostic,
not a substitute for populated native rendering, geometry, readable content and
functional controls. Acceptance criteria 4.1–4.5 retain these functional and
accessibility obligations under the updated design authority.

## Requirements

### Requirement 1: Exhaustive coverage governance

**User story:** As a migration owner, I want every Buzz component tracked, so that parity cannot be claimed by overlooking difficult infrastructure or client behavior.

#### Acceptance criteria

1. **1.1** THE migration SHALL maintain stable capability IDs for every meaningful Buzz crate, service, migration, protocol extension, client feature area, deployment component, operational tool, example, benchmark and conformance artifact.
2. **1.2** WHEN the Buzz source inventory changes, THEN CI SHALL fail until each new component maps to a canonical owner, disposition, requirement and leaf task or an approved evidence-backed not-applicable decision.
3. **1.3** THE migration SHALL record known incomplete Buzz behavior as required completion work and SHALL NOT use an existing defect to omit the capability.
4. **1.4** WHEN parity is reported, THEN THE report SHALL enumerate all capability IDs with passing reuse evidence, compatibility evidence or implementation validation and no unexplained state.

### Requirement 2: Single canonical ownership

**User story:** As a maintainer, I want one owner for each aggregate, so that users and services never reconcile permanent competing truths.

#### Acceptance criteria

1. **2.1** THE target architecture SHALL assign exactly one canonical authoring owner to projects, Git working state, messages, agent transcripts, identities, presence, workflows, media metadata and every other persisted aggregate.
2. **2.2** IF a compatibility representation or derived projection exists, THEN THE system SHALL identify its canonical source, version, rebuild/reconciliation rule and drift signal.
3. **2.3** WHERE temporary dual reads or writes are required, THE migration SHALL bound them by phase, reconcile deterministically, expose divergence, support rollback and include removal criteria.
4. **2.4** THE final system SHALL run no hidden Buzz desktop, parallel agent runtime or second authoritative project, Git, message, transcript, identity, presence, workflow or session store.
5. **2.5** WHEN Collaborative Workspace composes native capability, THEN its retained adapter SHALL carry canonical entity handles, subscriptions, actions or view-local state only; source validation SHALL reject new Collaborative `Store`, `Repository`, `Database` or persistence owners unless an approved requirement identifies genuinely new canonical functionality.

### Requirement 3: Reversible workspace presentation

**User story:** As a Zed user, I want to choose the editor or multiplayer presentation without forking my data, so that I can work in the mode appropriate to my task.

#### Acceptance criteria

1. **3.1** WHILE `multiplayer-tools` is enabled and onboarding is shown, Zed SHALL present exactly “Editor Workspace” and “Multiplayer Workspace” as clear choices and explain that both use the same underlying projects and data; WHILE `multiplayer-tools` is disabled, onboarding SHALL omit the complete Workspace selector section and use Editor Workspace implicitly.
2. **3.2** WHEN a choice is made, THEN Zed SHALL persist the presentation preference and open the selected presentation on subsequent launches.
3. **3.3** WHEN the user switches presentation later, THEN Zed SHALL preserve project, worktree, Git, identity, credentials, agent session and collaboration state without copying or forking it.
4. **3.4** IF an existing user has not selected Multiplayer Workspace, THEN current Editor Workspace behavior SHALL remain unchanged.

### Requirement 4: Native collaborative composition and accessibility

**User story:** As a collaborator, I want a dense native workspace matching the reference images, so that conversation, activity and review form one coherent work surface.

#### Acceptance criteria

1. **4.1** WHILE Collaborative Workspace is active, THE GPUI layout SHALL provide Delta's observed navigation, participant/task top bar, integrated conversation and inline input composition, optional resizable review pane and bottom/status surface, retaining Fractal styling and Buzz domain semantics; the conversation reference governs collapsed review and Image 2 governs expanded review, with conflicts reconciled against recorded actual Delta behavior.
2. **4.2** WHEN the review pane is collapsed or the window narrows, THEN the timeline SHALL expand into the reference full-width composition without losing review state or navigation context.
3. **4.3** THE workspace SHALL persist pane visibility/widths, active community/project/task, navigation history and relevant filters across restart.
4. **4.4** THE workspace SHALL support keyboard navigation, visible focus, screen-reader labels, logical focus order, zoom, theme tokens, reduced motion and usable narrow-window behavior without hardcoded light-theme colors.
5. **4.5** WHEN compared with the checked-in baselines at their native 1930×1262 and 1928×1298 viewports, THEN approved visual tests SHALL verify major geometry, typography, density, feed spacing, participant/state indicators and expanded/collapsed review compositions; replacement of either baseline SHALL require explicit visual-review approval.
6. **4.6** WHILE the review region presents a native file opened from review, IF the user closes every opened item, THEN the workspace SHALL release the native pane and restore the registered review provider and the conversation, without closing, duplicating or discarding the retained provider; an empty native pane SHALL NOT occupy the review region.

### Requirement 5: Protocol and signed-event interoperability

**User story:** As an existing Buzz client or operator, I want established wire contracts preserved, so that migration does not strand clients or signed history.

#### Acceptance criteria

1. **5.1** THE system SHALL preserve verification and encoding semantics for all registered standard and Buzz event kinds, including replaceable/addressable head selection, exact tag grammars, privacy gates and ephemeral non-persistence.
2. **5.2** WHEN a Nostr client uses supported NIP-01, NIP-11, NIP-29, NIP-42, NIP-45, NIP-50 or NIP-98 behavior, THEN the compatibility adapter SHALL produce protocol-equivalent results and failure frames.
3. **5.3** WHEN a client uses NIP-AA, AE, AM, AO, AP, CW, DV, ER, GS, IA, MP, OA, PL, PMA, RS or WP, THEN the system SHALL preserve the documented security, ordering, encryption, visibility, version and degradation contracts.
4. **5.4** IF a protocol version or event is unsupported or malformed, THEN the system SHALL reject or degrade exactly as its contract specifies, without fabricating canonical state.

### Requirement 6: Communities, authentication and tenant isolation

**User story:** As a community member or operator, I want all access scoped before it reaches storage, so that one community cannot observe or mutate another.

#### Acceptance criteria

1. **6.1** WHEN any WebSocket, HTTP, RPC, media, Git, search, workflow, push, pub/sub, mesh or administrative request arrives, THEN the system SHALL derive a typed community boundary from trusted connection/request context before authentication, authorization or data access.
2. **6.2** WHEN a human, agent, token or service authenticates, THEN one authorization policy SHALL enforce community membership, roles, scopes, owner attestations, channel membership and resource permissions consistently across transports.
3. **6.3** IF tenant context is missing, unknown, conflicting or derived from an untrusted payload, THEN the request SHALL fail closed without revealing existence, counts, timing-sensitive result sets or private metadata.
4. **6.4** THE system SHALL preserve invite redemption, join-policy evidence, membership revocation, virtual agent membership, archival and administrative role semantics across mixed-version clients.

### Requirement 7: Identity, profiles, credentials and signing

**User story:** As a human or agent, I want a durable verifiable identity with safe key custody, so that my contributions remain attributable without exposing secrets.

#### Acceptance criteria

1. **7.1** THE system SHALL support human and agent Nostr identities, profiles, status, owner attestations, social lists and relay-scoped archival while retaining explicit bindings to Zed service accounts where required.
2. **7.2** WHEN a signing key is imported, generated, paired, rotated, backed up or restored, THEN Zed's canonical credentials provider SHALL use protected storage, verify round trips, redact outputs and avoid deleting the prior source before successful verification.
3. **7.3** IF protected key storage is unavailable or corrupt, THEN the system SHALL fail safely or use an explicitly documented owner-only fallback without silently generating a replacement identity.
4. **7.4** WHEN an identity is archived, revoked or rotated, THEN historical authorship SHALL remain intact while active access, autocomplete, agent authorization and future signatures reflect the new state.

### Requirement 8: Relay, realtime synchronization and recovery

**User story:** As a collaborator, I want reliable realtime updates with visible recovery, so that temporary network or replica failures do not corrupt the workspace.

#### Acceptance criteria

1. **8.1** THE service SHALL support bounded authenticated connections, subscriptions, historical queries, live fan-out, counts, backpressure and cross-replica delivery without duplicate local echoes.
2. **8.2** WHEN a client disconnects and reconnects, THEN it SHALL reauthenticate, refetch the authoritative head/window, rearm live subscriptions and reconcile optimistic/local items deterministically.
3. **8.3** IF delivery, persistence, projection, pub/sub or replica freshness is partial or unavailable, THEN the UI and operator surfaces SHALL expose the affected scope, retry/recovery action and last trustworthy state.
4. **8.4** THE system SHALL bound frames, queries, subscriptions, queues, retries and retained realtime state, and SHALL clean up connection, process and subscription resources on cancellation or shutdown.

### Requirement 9: Communication, awareness, search and notification parity

**User story:** As a team member, I want complete collaborative communication, so that humans and agents can coordinate without leaving the workspace.

#### Acceptance criteria

1. **9.1** THE system SHALL support Buzz channel types, membership/roles/invites, DMs, messages, replies, edits, deletions, reactions, pins, bookmarks, schedules, canvases, forum posts, custom emoji and entity links with equivalent visibility rules.
2. **9.2** WHEN timelines or threads are paged, THEN rows, replies, aux events, summaries and bounds SHALL retain stable order and exact continuation under same-second events, deletions and concurrent live updates.
3. **9.3** THE system SHALL synchronize read/unread/manual-unread state, drafts, reminders, presence and typing with documented privacy, expiry, cross-device and offline behavior.
4. **9.4** WHEN search or discovery runs, THEN authorization and privacy exclusions SHALL be applied before limit/ranking, and results MAY compose with existing Zed file/project search without exposing private event content.
5. **9.5** WHEN a native or push notification is emitted, THEN it SHALL be deduplicated, permission-aware and privacy-preserving; push payloads SHALL be wake-only and authoritative data SHALL be fetched after reconnect.

The Rust-product channel subset currently uses the versioned collaborative-message RPC, canonical signed-event/message projections, PostgreSQL outbox replay and Redis notification transport. Channel history, create/edit/delete/reaction/read operations, dense keyset pagination, idempotent retry and reconnect replay have live PostgreSQL coverage; the remaining requirement qualification still depends on production GPUI server-backed composition, channel-thread selection, a full local Compose/two-client demonstration and the broader communication capabilities named by 9.1–9.5. <!-- impl: crates/collab/src/messages/channel_service.rs#CanonicalMessageService --> <!-- impl: crates/collab/src/messages/channel_runtime.rs#CanonicalMessageRuntime --> <!-- impl: crates/collab_ui/src/channel_messaging.rs#ChannelMessagingTransport -->

### Requirement 10: Projects, Git forge and review parity

**User story:** As a developer, I want project conversation and code review connected to canonical Git state, so that work and its rationale remain traceable.

#### Acceptance criteria

1. **10.1** THE system SHALL support cross-owner multi-repository projects and channel binding without granting project signers authority over member repositories.
2. **10.2** THE system SHALL preserve NIP-34 repository, ref, patch, pull-request, issue and status interoperability plus NIP-98 Git authentication and Nostr commit/tag signing.
3. **10.3** WHEN a branch, patch, review comment, CI result, approval or merge event changes, THEN the canonical collaboration timeline SHALL link that event to the relevant repository, commit, branch and native diff state.
4. **10.4** WHEN reviewing agent or human changes, THEN the review surface SHALL reuse native unified/split diffs, file navigation, additions/deletions, keep/reject, stage and review actions where semantically valid and expose conflicts/staleness.

### Requirement 11: Agent platform parity

**User story:** As a human supervising agents, I want identities, runtimes and delegated work to compose with Zed's native ACP platform, so that agents are first-class collaborators without duplicate execution engines.

#### Acceptance criteria

1. **11.1** THE system SHALL accept supported ACP agents and MCP servers while using Zed's agent, ACP thread, tool-permission and process-lifecycle owners for native execution.
2. **11.2** THE system SHALL support agent identities, owner attestations, managed agents, personas, teams, catalogs, runtime/model/provider/environment configuration and share/private projection rules.
3. **11.3** THE system SHALL preserve encrypted engrams, private managed-agent state, snapshots, local archive and per-turn usage metrics with explicit ownership, retention, import/export and privacy semantics.
4. **11.4** WHEN jobs or delegated tasks are requested, accepted, progressed, completed, cancelled or failed, THEN one idempotent state machine SHALL authorize the transition and expose it to humans and participating agents.
5. **11.5** WHEN agents run locally, remotely or through a provider, THEN identity, permissions, cancellation, bounded liveness, presence, cleanup and result delivery SHALL remain equivalent apart from explicitly surfaced substrate capabilities.

### Requirement 12: Semantic activity and supervision

**User story:** As an agent supervisor, I want a legible semantic activity feed, so that I can understand progress, confidence and required intervention at a glance.

#### Acceptance criteria

1. **12.1** WHEN activity is received from ACP, NIP-AO, Git, workflow, CI, moderation or system sources, THEN every event SHALL map exactly once to a semantic presentation or truthful generic fallback.
2. **12.2** THE feed SHALL lead with verb, object and outcome for messages, thoughts, plans, reads, searches, edits, shell commands, tests, permissions, errors and lifecycle changes, with detail and raw data behind progressive disclosure.
3. **12.3** WHEN a running action changes state, THEN its existing feed item SHALL update in place from pending through terminal outcome without duplicate status rows.
4. **12.4** IF an agent is idle, waiting, silent, timed out, cancelled, disconnected or failed, THEN the feed SHALL show that state and any required intervention rather than going blank.

### Requirement 13: Workflows, approvals, audit and usage

**User story:** As a team or auditor, I want automation and decisions to be durable and attributable, so that work can be reproduced and reviewed.

#### Acceptance criteria

1. **13.1** THE system SHALL support versioned workflow definitions, schedules, webhooks, event triggers, conditions, step actions and durable run state scoped to a community/project.
2. **13.2** WHEN a workflow requires approval, THEN it SHALL suspend durably, expose the request, accept exactly one authorized grant/deny outcome and resume or terminate deterministically.
3. **13.3** IF workflow evaluation, webhook delivery or an action fails, THEN the run SHALL record a bounded/redacted error, retry only under explicit policy and never bypass a permission or approval gate.
4. **13.4** THE system SHALL maintain a per-community tamper-evident audit chain and privacy-preserving usage records for security-relevant, administrative, workflow and agent operations.

### Requirement 14: Media, voice and huddles

**User story:** As a collaborator, I want rich media and synchronous conversation integrated with project context, so that all relevant work remains in one workspace.

#### Acceptance criteria

1. **14.1** WHEN media is uploaded or downloaded, THEN the system SHALL authenticate, tenant-scope, validate type/size/content, store and render it without exposing object-store credentials or cross-community paths.
2. **14.2** THE system SHALL preserve Blossom compatibility, attachment metadata, thumbnails, images, video and link-preview behavior through native Zed media renderers where possible.
3. **14.3** WHEN a huddle starts, joins, leaves or ends, THEN lifecycle, participant, reaction, audio-control and transcript events SHALL remain consistent across the native transport and supported Buzz compatibility transport.
4. **14.4** IF microphone, speaker, network, local voice model, transcription or TTS fails, THEN the UI SHALL expose the failed function, retain the huddle/conversation state and offer a safe retry or fallback.

### Requirement 15: Moderation, retention, deletion and administration

**User story:** As a community operator, I want enforceable lifecycle and safety controls, so that I can administer a community without violating privacy or losing recovery options.

#### Acceptance criteria

1. **15.1** THE system SHALL support reports, personal mutes, bans, timeouts, resolution, identity archive, community archive and role-gated administration with complete audit attribution.
2. **15.2** WHEN retention or ephemeral expiry applies, THEN storage, search, caches, push queues and projections SHALL converge on the same deletion/visibility result under mixed versions and retries.
3. **15.3** WHEN whole-community deletion is requested, THEN a durable state machine SHALL verify authority, expose progress, support recovery before irreversible work and prevent partial tenant reuse.
4. **15.4** IF an administrative operation is unauthorized, stale, ambiguous or partially failed, THEN it SHALL fail closed and expose a redacted operator-safe diagnostic without weakening tenant isolation.

### Requirement 16: Pairing, remote agents, shared compute and client surfaces

**User story:** As a user across devices and compute locations, I want secure interoperable access, so that the same community and agents can work locally, remotely and from companion clients.

#### Acceptance criteria

1. **16.1** THE system SHALL preserve NIP-AB device pairing, QR/session expiry, replay protection and verified import into Zed's canonical credential store.
2. **16.2** THE remote-agent provider boundary SHALL preserve discovery, hostile-output validation, secret/config separation, identity fail-closed, presence-as-status, bounded cleanup and at-most-one-instance semantics.
3. **16.3** WHERE shared compute or relay mesh is enabled, THE system SHALL authenticate community membership, enforce approved resource/trust policy, fence stale peers and expose availability/failure without silently falling back to an unapproved provider.
4. **16.4** THE agent-first CLI, web repository/invite client, mobile client and administration surfaces SHALL remain interoperable throughout migration and retain documented commands, URLs, deep links and exit/error contracts or an approved versioned replacement.

### Requirement 17: Data migration, import and rollback

**User story:** As an existing user or operator, I want all Buzz and Zed state migrated safely, so that upgrading does not lose work or identities.

#### Acceptance criteria

1. **17.1** BEFORE changing canonical ownership, THE migration SHALL inventory and version existing Buzz Postgres, Redis-derived, object-store, keyring/fallback, desktop archive/config, agent snapshot, event-sync and Zed persistence data.
2. **17.2** WHEN an importer or schema migration runs, THEN it SHALL be resumable, idempotent, integrity-checked, tenant-scoped and observable, and SHALL preserve the original until verification passes.
3. **17.3** IF migration verification or compatibility health fails before the point of no return, THEN operators SHALL be able to restore the prior binary/configuration and authoritative data without accepting divergent writes.
4. **17.4** WHEN a temporary bridge, dual read or dual write is introduced, THEN the migration SHALL define precedence, reconciliation, divergence alerts, rollback and a dated removal gate.

### Requirement 18: Compatibility, versioning and source retirement

**User story:** As a client or automation owner, I want a predictable compatibility period, so that I can upgrade without coordinated downtime.

#### Acceptance criteria

1. **18.1** THE migration SHALL publish a compatibility matrix for Zed desktop versions, Buzz desktop/mobile/web/CLI versions, relay/service versions, protocol features and stored-schema versions.
2. **18.2** WHEN a compatibility boundary changes, THEN clients SHALL negotiate or receive a clear minimum-version error before performing an incompatible write.
3. **18.3** BEFORE retiring any Buzz component, THE system SHALL meet its parity, migration, traffic/usage, rollback-window, documentation and ownership exit criteria.
4. **18.4** WHEN retirement completes, THEN `projects/buzz` SHALL be reference-only or removed according to approved licensing/history policy, and builds/releases SHALL no longer depend on its retired desktop or duplicate runtime implementations.

### Requirement 19: Security, operations and release readiness

**User story:** As an operator, I want the consolidated platform observable, bounded and deployable, so that expanded capability does not reduce security or reliability.

#### Acceptance criteria

1. **19.1** THE design SHALL threat-model signing keys, identity binding, untrusted events/content, provider binaries, MCP tools, webhooks, media, search, push, mesh, database access and cross-tenant timing/metadata leaks.
2. **19.2** THE implementation SHALL preserve or strengthen frame/body/output limits, SSRF and redirect policy, secret redaction, sandbox/permission gates, TLS configuration, process cleanup, retention and least privilege.
3. **19.3** THE consolidated services SHALL provide health/readiness, metrics, structured redacted logs, migration status, projection drift, queue/backpressure, replica freshness and compatibility-version observability.
4. **19.4** WHEN deployed through local, Compose, Helm or release pipelines, THEN configuration/schema validation, signed artifacts, migration jobs, rollback inputs and platform packages SHALL follow Zed's canonical release conventions.
5. **19.5** WHILE telemetry is disabled by Zed settings, THE collaborative workspace SHALL not re-enable client telemetry through Buzz-derived code; local operational logging and server observability SHALL remain available under their documented policies.

### Requirement 20: Verification and final parity

**User story:** As a reviewer, I want independent evidence for equivalence and native quality, so that the migration can be accepted without trusting implementation similarity.

#### Acceptance criteria

1. **20.1** THE verification program SHALL include focused unit, GPUI, integration, end-to-end, migration, compatibility, security, fault-injection, load and visual tests appropriate to each capability.
2. **20.2** THE multitenant conformance checker and protocol fixtures SHALL remain independent of production reducers and SHALL run against both compatibility and consolidated service paths during migration.
3. **20.3** WHEN behavior is reused unchanged from Zed, THEN parity SHALL require semantic evidence covering security, persistence, failure and user-visible behavior, not a matching component name.
4. **20.4** THE migration SHALL NOT declare complete until every CAP ID and acceptance criterion has passing evidence, all approved migration/removal gates are satisfied, documented known Buzz gaps are completed or explicitly accepted, and no prohibited duplicate owner remains.
5. **20.5** WHERE a surface is painted through a cached view, THE visual and geometric evidence method SHALL assert presentation through canonical state rather than element geometry, because debug geometry is not re-recorded for an unchanged cached remount; absence of a geometry record alone SHALL NOT be reported as absence or unmount of the surface, and semantic presentation state alone SHALL NOT be reported as proof of painted pixels.

### Requirement 21: Compile-time multiplayer feature isolation

**User story:** As a release owner, I want one compile-time capability switch for multiplayer functionality, so that Standard Zed remains unchanged while Multiplayer Zed includes the complete Collaborative Workspace and Buzz compatibility platform.

#### Acceptance criteria

1. **21.1** THE canonical `zed` application package SHALL define one public Cargo feature named `multiplayer-tools`, SHALL keep it outside the default feature set, and SHALL forward it only to narrowly scoped internal crate features and optional dependencies.
2. **21.2** WHILE `multiplayer-tools` is disabled, THE Zed application SHALL build, test, package and start without multiplayer-only crates, dependencies, services, transports, migrations, assets, actions, settings surfaces or background jobs, and Editor Workspace behavior SHALL remain unchanged.
3. **21.3** WHILE `multiplayer-tools` is enabled, THE Zed application SHALL offer the approved Collaborative Workspace, adapters and services without forking canonical Editor, project, worktree, Git, identity, credential, transcript or agent-session state.
4. **21.4** IF an unflagged build reads a persisted Collaborative Workspace preference, THEN it SHALL use Editor Workspace for that run without deleting collaborative data, overwriting the saved preference or requiring a multiplayer-only crate; WHEN a compatible flagged build returns, THEN it SHALL restore that preference.
5. **21.5** WHILE `multiplayer-tools` is disabled, THE application SHALL omit multiplayer onboarding choices, workspace-switch actions, menus, settings pages and service registrations; IF a retained compatibility entry point recognizes a multiplayer-only operation, THEN it SHALL return a deterministic “not included in this build” result without disclosing tenant or resource existence.
6. **21.6** WHEN a desktop, service or companion client negotiates capabilities, THEN it SHALL advertise multiplayer availability explicitly and SHALL reject unsupported multiplayer-only writes before tenant or resource lookup.
7. **21.7** THE feature boundary SHALL leave shared Editor, project, worktree, Git, ACP, credentials, settings and existing collaboration functionality always compiled, and SHALL preserve one canonical domain/state representation across both configurations.
8. **21.8** WHEN Standard Zed is packaged or deployed, THEN exclusive Buzz services, migrations and assets SHALL be absent; WHEN Multiplayer Zed is packaged or deployed, THEN its release command SHALL enable `multiplayer-tools` explicitly and record that capability in artifact metadata.
9. **21.9** CI SHALL build, test, warning-denied lint and smoke both configurations, inspect the default dependency tree for forbidden multiplayer-only packages, and fail when feature unification or packaging causes an unflagged artifact to include multiplayer code or dependencies.
10. **21.10** THE dual-profile check SHALL execute the unfiltered library test suites for the verified Collaborative composition-owner scope (`workspace`, `git_ui`, `sidebar`, `agent_ui`) for both configurations; it SHALL NOT certify a configuration using only name-filtered subsets, because a test asserting behavior gated behind a feature the configuration does not enable still compiles and its failure would otherwise go unreported.

### Requirement 22: Task row operations and additional reference acceptance

**User story:** As a collaborator, I want to manage a specific task without changing my current conversation or losing its project relationships.

#### Acceptance criteria

1. **22.1** WHEN a task context menu opens by mouse, row overflow or keyboard, THEN it SHALL target that row’s canonical ThreadId, preserve the active conversation, anchor to the initiating row/control, support focus movement, Escape/outside dismissal and focus restoration, and resolve availability, permissions and shortcuts from current canonical state and platform bindings.
2. **22.2** WHEN Pin or Unpin is chosen, THEN the existing persisted navigation owner SHALL update that task’s pinned state and all affected rows without changing project associations or activating the task.
3. **22.3** WHEN Share is chosen, THEN the canonical supported transcript sharing path SHALL share the targeted task with visible progress, errors and recovery; unsupported providers or missing authorization SHALL be explicitly unavailable without simulated success.
4. **22.4** WHEN Rename or Regenerate title is chosen, THEN the canonical thread metadata and summary provider SHALL update the targeted title, preserve explicit user overrides according to the native policy, and expose pending, unavailable and failed states.
5. **22.5** WHEN Mark as unread is chosen, THEN the canonical persisted task read-state owner SHALL update that task’s unread indication, independently of active conversation selection, and later reading SHALL clear it through the same owner.
6. **22.6** WHEN View details is chosen, THEN the interface SHALL display the targeted task’s actual identity, provider, timestamps, execution context and project associations from canonical owners.
7. **22.7** WHEN Copy content or Copy JSON is chosen, THEN the canonical transcript owner SHALL export the targeted content or structured transcript, including actual attribution and reply relationships, to the clipboard; metadata-only or fabricated content SHALL NOT count as a transcript export.
8. **22.8** WHEN project associations are viewed, added or removed, THEN the canonical association owner SHALL persist and update project/task navigation relationships without rewriting execution worktree paths or forking Project, Git, transcript or session ownership.
9. **22.9** WHEN Archive or Delete is chosen, THEN the interface SHALL perform distinct canonical operations: archive retains resumable history and restoration; delete follows provider-supported transcript deletion and native confirmation/error policy; neither operation SHALL be represented by merely removing a project association.
10. **22.10** THE additional references in screenshots/README.md SHALL be preserved unchanged and evaluated using populated production render paths with the menu closed, open and opened on a non-active task, with effective theme, fonts, unstable.ui_density, logical/physical dimensions and zoom recorded; missing host-window or authenticated-service evidence SHALL remain explicitly unverified.

## Constraints

- The primary desktop implementation is native Rust/GPUI.
- Lower-level domain, persistence and protocol modules must not depend on GPUI.
- Use `./script/clippy`, repository Rust conventions and GPUI executor timers in GPUI tests.
- Preserve Apache-2.0 notices and attribution for imported Buzz code within the GPL-3.0-or-later Zed repository.
- Security and tenant boundaries fail closed; compatibility never weakens authorization.
- Production mutations and source retirement require separate authorization.
- `multiplayer-tools` is the only public Cargo feature for this product boundary and is non-default; internal forwarding features must not be selected independently by release users.

## Resolved architecture decisions

ADR-001 through ADR-006 were accepted on 2026-08-14 without changing the acceptance criteria above. Their normative records are `decisions/adr-001-service-topology.md` through `decisions/adr-006-shared-compute.md`. Production activation, irreversible migration, compatibility breaks and source retirement remain separately approval-gated.
