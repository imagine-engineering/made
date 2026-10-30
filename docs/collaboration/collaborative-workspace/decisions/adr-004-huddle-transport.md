# ADR-004: Native Huddle Transport

- **Status:** Accepted
- **Decision date:** 2026-08-14
- **Approval:** The product owner approved LiveKit as the native huddle transport with a versioned Buzz Opus/WebSocket compatibility adapter.
- **Requirements:** 2.1, 14.3, 14.4
- **Capabilities:** CAP-032

## Context

Zed already has `livekit_api`, `livekit_client`, native audio capture/playback, participant, room, token and device abstractions. Buzz has a separate bounded Opus-over-WebSocket huddle protocol with NIP-42 authentication, community/channel admission, room-version pinning, participant indexes, roster control, connection limits, heartbeats, dropped-on-backpressure media, cross-pod fencing and lifecycle events. Buzz also supplies local voice/TTS and transcription behavior.

Shipping both transports as peers would create two room owners, participant rosters and failure models. Removing Buzz audio immediately would break existing desktop/mobile clients. The huddle lifecycle therefore needs one transport-neutral domain owner, one native media authority and a compatibility route whose behavior and lifetime are explicit.

## Decision

### Canonical domain and native transport

The Zed collaboration domain owns transport-neutral huddle identity and lifecycle: community, channel, session/generation, start, join, leave, end, participant identity, role, reaction, moderation state and transcript references. Domain events are authoritative for collaboration history and policy; they contain media/session references rather than audio frames.

LiveKit is the sole native realtime media transport and room authority. A canonical huddle generation maps to exactly one LiveKit room. LiveKit owns current media participants, track publication/subscription, reconnect state and media quality. Zed collaboration authorization issues short-lived, room-scoped participant tokens only after trusted tenant, membership, role and huddle-generation checks.

The existing `livekit_api` and `livekit_client` crates remain the native API/client owners. `audio` owns device capture/playback and transport integration. `collaboration_domain` must not depend on LiveKit, GPUI, devices or wire codecs.

### Transport-neutral lifecycle mapping

| Canonical lifecycle behavior | LiveKit/native mapping | Buzz compatibility mapping |
| --- | --- | --- |
| Start | Authorize and create/resolve one room for a new huddle generation | Old start/lifecycle event resolves the same generation; it never creates a second room |
| Join | Issue scoped token and admit participant identity to the room | NIP-42 plus community/channel checks, then gateway admission to the same room |
| Participant present | Verified canonical participant ID linked to LiveKit identity and track state | Verified npub/agent ID linked to bounded compatibility peer index and gateway track |
| Mute/PTT/device state | Local capture publication state plus visible canonical participant state where shareable | Supported control/absence-of-frames translated without trusting client VU telemetry |
| Reaction/moderation | Authorized huddle domain command and event | Versioned control/event translation to the same command |
| Leave/disconnect | Remove participant mapping and publish idempotent leave; retain reconnect grace explicitly | WebSocket cleanup removes peer/gateway mapping and publishes the same idempotent leave |
| End/empty room | Authorized end or final-participant policy closes the generation and revokes tokens | Adapter receives terminal state and closes compatible peers; it cannot keep the room alive independently |
| Transcript | Provenance-aware bounded segments reference the canonical huddle generation | Old transcript/lifecycle representation projects to the same records |

Duplicate join, leave and end inputs are idempotent by huddle generation and participant identity. LiveKit callbacks and compatibility control frames pass through the same reducer, so a mixed-client huddle has one roster and one terminal outcome.

### Buzz Opus compatibility adapter

The compatibility adapter preserves supported Buzz huddle protocol versions 1 and 2 unless a later compatibility decision explicitly changes the floor. It retains exact legacy admission and safety behavior where externally observable:

- NIP-42 authentication and host-derived community/channel membership checks;
- bounded text and binary frames, authentication timeout, heartbeat and connection capacity;
- v1 default negotiation, per-session compatible-version rules and generic version/capacity failures;
- v2 sequence/timestamp/header parsing with untrusted audio-level clamping;
- bounded peer indexes and ordered roster control isolated from media backpressure;
- malformed-frame rejection, cancellation and deterministic resource cleanup; and
- compatibility lifecycle/event shapes required by supported clients.

The adapter is a gateway into the canonical LiveKit room, not a second huddle transport authority. It maps an authenticated legacy participant to a scoped gateway participant/track, converts supported Opus frames through a bounded media bridge, and converts subscribed native audio back to the negotiated Buzz frame format. It maintains only expiring connection, jitter/sequence and peer-index state. It cannot create durable participant, transcript or room state outside canonical commands.

If the media bridge, LiveKit room, token service or canonical command path is unavailable, legacy admission fails visibly. The adapter never falls back to an independent Buzz room or silently isolates legacy participants from native participants. Unsupported codec parameters, versions or mixed-generation frames fail before media forwarding.

### Participant identity and privacy

LiveKit identity values are opaque, scoped references derived from canonical participant and huddle-generation IDs; they do not expose secrets or grant collaboration membership. Token grants name one room, participant and bounded lifetime. Reconnect or device switching cannot change the authenticated participant.

Audio frames are ephemeral and are not written to the signed event log, application logs or metrics. Audio level, sequence, timestamps and device information are untrusted telemetry and cannot authorize moderation or membership. Recording and transcription are separate permissioned actions with visible state, retention policy and participant notice. Transcript records store bounded text/provenance, not reusable media credentials.

Community isolation is enforced before token issuance and legacy WebSocket upgrade. Room names, gateway maps, metrics and cache keys include typed community and huddle-generation identity. Unknown or conflicting tenants, stale generations and cross-community channel IDs fail closed without exposing room existence.

### Platform support

Native Zed desktop huddles use the existing LiveKit Rust/client integration on every supported packaged desktop target for which its media dependencies pass build and device tests. A platform without a validated LiveKit client is reported as unsupported for native huddles; it does not silently use the legacy relay.

Companion web, iOS and Android clients may use their platform LiveKit SDKs behind the same token and lifecycle contracts. During migration, released Buzz companion clients continue through the Opus adapter. Platform-specific implementations do not own huddle domain state.

The release gate includes microphone permission, input/output selection, mute, push-to-talk, device loss/switch, reconnect, background/lifecycle behavior and accessible failure UI on every advertised platform. Screen sharing and video may reuse LiveKit but are not substitutes for the required audio lifecycle.

### TTS and transcription

Local TTS is a cancellable media producer under Zed's audio/model/permission owners. It joins or publishes into the canonical huddle only with visible agent/human attribution and the same participant policy. A missing model, synthesis error or cancellation affects only that action and cannot end or fork the huddle.

Transcription consumes an explicitly authorized canonical audio source and emits partial/final segments with huddle, participant, time-range, model/provider, consent and retention provenance. Partial segments can be replaced deterministically; final segments project into authorized channel records. Failure, retry and redaction remain visible and do not fabricate a completed transcript.

## Compatibility support and retirement

The Buzz adapter remains supported through migration Phase 8. It can be retired only after all of the following are satisfied:

1. all supported native desktop, web, iOS and Android client versions negotiate LiveKit and pass lifecycle/media compatibility tests;
2. at least two consecutive supported stable client release generations no longer require Buzz audio for new sessions;
3. an observation window shows no unexplained lifecycle, participant, authorization, media or transcript divergence and identifies the remaining legacy-client population;
4. administrators can detect and upgrade or explicitly block every client below the LiveKit compatibility floor;
5. mixed native/legacy huddle, reconnect, device-loss, adapter-restart and LiveKit-failure drills pass;
6. deployment manifests, routing, dashboards and runbooks have a reviewed adapter-removal change and rollback plan; and
7. the compatibility break and final support floor receive explicit release approval.

If a supported client population still requires Buzz audio at Phase 8, the adapter becomes a named long-term compatibility boundary with an owner, security updates, capacity targets and conformance tests. It does not regain room or domain authority.

Removal first disables creation of new legacy sessions, observes drain and upgrades, then removes routing after no live legacy generation remains. Rollback can re-enable the same versioned gateway against canonical LiveKit rooms; it never restores the Buzz audio relay as an independent authority.

## Failure and recovery behavior

- Token/auth/membership/version failure rejects before joining and exposes a scoped actionable reason without room-existence leakage.
- Media disconnection preserves the canonical participant/session identity during a bounded reconnect window and shows reconnecting state.
- Device failure preserves room membership where safe while stopping the affected track and offering retry/device selection.
- Adapter restart removes or reconstructs only expiring gateway state; canonical roster reconciliation and LiveKit participant state repair duplicate/missing callbacks.
- LiveKit outage prevents new media admission and visibly marks affected huddles; messages and the surrounding workspace continue operating.
- Stale generation callbacks, duplicate roster frames and late audio are ignored or reconciled idempotently and cannot reopen an ended huddle.
- Cancellation tears down capture, playback, subscriptions, gateway tracks, timers and credentials with leak-focused tests.

## Alternatives rejected

1. **Retain Buzz audio as a peer native transport:** rejected because it creates two room, roster, reconnect and deployment authorities.
2. **Remove Buzz audio at the first LiveKit release:** rejected because it breaks established companion clients without a versioned transition.
3. **Let legacy-only rooms bypass LiveKit:** rejected because compatibility demand cannot create a second canonical media topology.
4. **Copy Buzz room/mesh state into the huddle domain:** rejected because connection leases, peer indexes and frames are expiring adapter state, not durable collaboration records.
5. **Use client-authored VU metadata for trust decisions:** rejected because it is untrusted telemetry.

## Implementation and validation trace

- Tasks 39.1–39.2 implement the transport-neutral reducer and native LiveKit adapter.
- Task 39.3 implements the bounded Buzz Opus/WebSocket gateway and version support.
- Tasks 39.4–39.7 implement device controls, TTS, transcription and native workspace UI.
- Task 39.8 proves native/legacy/mixed-client lifecycle and media equivalence.
- Tasks 44.3–44.5 own deployment, observability and readiness for LiveKit and the adapter.
- Tasks 43.1–43.8 own companion-client negotiation and compatibility; Tasks 45.1–45.6 own independent validation gates, Tasks 47.1–47.6 gate adapter retirement, and Tasks 48.1–48.4 retain final parity evidence.

Review acceptance requires start/join/leave/end/duplicate lifecycle parity, mixed native and legacy participation, platform device/permission coverage, bounded media behavior, tenant and generation isolation, visible failure/reconnect/cancellation, transcript provenance and every adapter-retirement criterion above.
