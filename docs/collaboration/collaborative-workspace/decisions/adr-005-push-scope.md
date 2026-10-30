# ADR-005: Push Platform Scope

- **Status:** Accepted
- **Decision date:** 2026-08-14
- **Approval:** The product owner approved APNs parity as the first mobile-cutover support floor, with platform-neutral lease/executor boundaries that can add providers later.
- **Requirements:** 2.1, 9.5, 19.2
- **Capabilities:** CAP-016

## Context

Buzz implements NIP-PL blind push and a stateful APNs gateway. The relay owns authorization-aware lease matching and durable wake jobs; the gateway owns installation authority, encrypted APNs-token custody, App Attest verification, relay-scoped delivery grants and the fixed APNs reconnect payload. Current NIP-PL documentation explicitly leaves FCM and UnifiedPush profiles undefined because their fixed-payload and hostile-endpoint security contracts have not been registered.

Zed has native desktop notification UI but no equivalent mobile push executor. The first companion-client cutover must preserve iOS wake behavior without placing event content, relay identity, channel, sender, unread count, ciphertext or deep links in provider payloads. Extensibility must not weaken that privacy boundary or delay the first cutover on platforms Buzz does not currently support.

## Decision

### First mobile-cutover floor

The first mobile cutover supports these push targets:

| Target | First-cutover status | Required profile |
| --- | --- | --- |
| iOS production and TestFlight | Required | APNs production profile with the configured application identifier, topic, production credentials and Apple App Attest |
| iOS development validation | Required for release validation | APNs sandbox profile with its matching application identifier, topic, environment and App Attest configuration |
| Android FCM | Not required for the first cutover | No conforming NIP-PL profile is approved by this ADR |
| UnifiedPush | Not required for the first cutover | No conforming public-gateway profile is approved by this ADR |
| Desktop | Existing native Zed notifications; not a mobile push transport | Authoritative data is already available through the connected client |

TestFlight distribution uses the production APNs environment and production App Attest. Development/sandbox evidence does not establish TestFlight delivery. Verify the signed distribution entitlements and matching gateway profile before release validation. See Apple’s [APNs environment guidance](https://developer.apple.com/library/archive/technotes/tn2265/_index.html) and [App Attest environment requirements](https://developer.apple.com/documentation/devicecheck/preparing-to-use-the-app-attest-service).

APNs parity includes installation enrollment, App Attest verification, assertion counters, endpoint rotation, relay delegation, encrypted endpoint custody, lease generation/expiry/revocation, durable wake delivery, permanent-endpoint invalidation, bounded transient retry, production/sandbox separation and privacy conformance.

An installation or deployment that cannot complete the required App Attest and APNs configuration does not receive public-gateway push. It falls back visibly to normal foreground synchronization, reconnect/manual refresh and local notifications after authenticated fetch. There is no unsigned enrollment, shared device token, generic webhook, rich relay-authored payload or attestation bypass.

### Canonical ownership

The collaboration domain owns provider-neutral push leases, generation state, expiry, revocation and notification eligibility. The canonical collaboration persistence/outbox owns accepted lease state and durable wake jobs, partitioned by trusted community/origin and device authority.

The push gateway owns last-hop installation authority, encrypted platform endpoint custody, replay-resistant enrollment/delegation challenges, provider credentials and provider response classification. It receives an opaque delivery capability and bounded routing controls, not event content. APNs is implemented behind a common provider contract; that contract cannot expose arbitrary application-payload bytes.

The mobile client owns local permission prompts, platform token acquisition, installation assertion creation, lease publication and authoritative fetch after waking. APNs acceptance means only that the provider accepted the attempt, not that the application read an event.

### Wake-only invariant

Every provider attempt sends a provider-profile-owned constant reconnect body. Relay requests and durable wake jobs contain no title, subtitle, message text, URL, deep link, event/lease/channel/sender identifier, unread count, ciphertext, preview or generic extension map.

For the approved APNs profile, the application body remains the exact registered NIP-PL constant:

```json
{"aps":{"alert":{"body":"Reconnect to your relay now"},"mutable-content":1}}
```

Only provider-owned routing controls may vary: destination, authenticated topic/environment, expiration, canonical request ID, push type and priority. The gateway rejects unknown request members rather than ignoring them. On receipt, the app synchronizes every locally configured authorized origin as needed and computes badge/preview state only from freshly fetched canonical data.

Timing and frequency remain metadata leakage risks. Eligibility, coalescing, quotas, rate limits and retries are bounded; the executor rechecks active lease generation, expiry, endpoint authority and current event-read authorization immediately before delivery.

### APNs installation and attestation requirements

Enrollment uses a single-use, short-lived gateway challenge and Apple App Attest proof bound to every authority-bearing enrollment field. The gateway verifies the Apple chain/root, application identifier, key identifier, challenge transcript and bounded CBOR before accepting an installation. Subsequent delegation, rotation and revocation require assertions from that installed key with a strictly increasing counter and exact canonical transcript.

The gateway stores the App Attest public key and counter, an encrypted APNs token, a non-reversible token fingerprint, endpoint epoch, profile and expiry. It never stores a plaintext endpoint in relay lease state or returns the endpoint to a relay. Production and sandbox profiles have distinct validated configuration and cannot share topics, credentials or endpoint authority by accidental fallback.

Attestation, assertion, endpoint and capability inputs have strict size/encoding limits, closed JSON schemas and generic rejection responses that do not reveal installation existence. Challenge replay, counter rollback, profile mismatch, stale endpoint epoch, stale lease generation and wrong relay delegation fail closed.

### Lease, endpoint and delivery behavior

- A lease is a signed wake request, never a read grant. Matching and delivery both recheck tenant/origin and current authorization.
- Lease replacements and revocations require strictly increasing generations. Rejected updates leave the prior effective lease and watermark unchanged.
- Revocation tombstones survive long enough to prevent replay resurrection and cancel undelivered work where practical.
- Endpoint rotation advances an independent endpoint epoch and invalidates only the prior endpoint generation.
- Permanent APNs invalid-endpoint responses disable only that endpoint generation; provider/configuration faults do not revoke users or sibling installations.
- Transient failures use bounded, jittered retries and sanitized provider hints. Retry exhaustion is visible and leaves no unbounded queue.
- Wake coalescing and deduplication use stable job/source IDs without including content in the last-hop request.
- Sign-out revokes only the selected installation/leases and preserves sibling devices.

### Provider extension boundary

The common provider interface accepts only a validated profile ID, opaque endpoint handle/capability, canonical request ID and bounded expiration, and returns sanitized outcomes such as accepted, retryable, invalid endpoint or configuration fault. It does not accept serialized payloads or provider-specific arbitrary maps.

FCM or UnifiedPush may be added after the first cutover only through a separately approved profile that supplies:

1. one registered, byte-pinned reconnect body with the same noninterference proof;
2. endpoint enrollment and custody authority appropriate to the platform;
3. attestation or an explicitly reviewed alternative anti-abuse/trust boundary;
4. closed schemas, bounded endpoints and no client-selected callback amplification;
5. endpoint uniqueness, rotation, revocation, replay and quota semantics equivalent to NIP-PL;
6. provider response sanitization and permanent/transient classification;
7. privacy, tenant/origin isolation, load, abuse and hostile-endpoint tests; and
8. mobile lifecycle, permission, background-delivery and version-negotiation E2E evidence.

Until such a profile is approved, descriptors must not advertise it and clients must not silently downgrade an unsupported transport to APNs, a webhook or a rich notification. UnifiedPush in particular cannot dereference arbitrary distributor endpoints without an approved SSRF/amplification and fixed-body design.

### Compatibility floor and cutover gates

The first mobile cutover can proceed when:

- current supported Buzz iOS production and sandbox fixtures enroll, rotate, delegate, revoke and wake through the Zed-owned gateway;
- NIP-PL lease encryption, generation, origin isolation, matching and replacement fixtures pass without resigning valid existing leases;
- every APNs attempt carries the exact constant body and no relay/event data;
- App Attest production validation, assertion replay/counter tests and generic-error tests pass;
- endpoint/provider failure, retry exhaustion, outbox crash recovery and revocation races are exercised;
- push permission denial and unavailable push are visible while foreground/manual synchronization remains correct;
- deployment configuration fails readiness on missing/mismatched App Attest, APNs, sealing or database secrets; and
- load tests meet approved queue, retry, coalescing and gateway resource bounds.

Existing accepted leases and gateway authority data are migrated with encrypted values, generations, endpoint epochs and provenance intact. Source state is retained until new delivery and revocation reconciliation passes. Rollback stops the new executor, drains/records outbox state and restores compatible routing without replaying stale generations or exposing raw endpoints.

FCM/UnifiedPush absence is not a parity failure for the first cutover because Buzz has no conforming current profile for them. It remains a deliberate approval gate, not a perpetual deferred bucket: a future product request must either approve and task the profile above or explicitly retain the APNs-only product floor.

## Alternatives rejected

1. **Require FCM and UnifiedPush before the first cutover:** rejected because their NIP-PL security profiles are undefined and Buzz does not currently provide parity behavior to port.
2. **Use rich APNs payloads for better previews:** rejected because it exposes private collaboration metadata and changes NIP-PL's transport noninterference contract.
3. **Allow self-hosted deployments to bypass App Attest on the public profile:** rejected because it creates an unverified installation and abuse path under the same advertised security contract.
4. **Store APNs tokens in relay leases:** rejected because relays need only opaque delivery capabilities and should not own platform endpoints.
5. **Treat push as reliable event delivery:** rejected because push is best-effort wake-up; authenticated synchronization remains authoritative.

## Implementation and validation trace

- Tasks 22.4–22.7 define eligibility, provider-neutral leases and canonical encrypted/outbox persistence.
- Tasks 22.8–22.10 implement the gateway executor, APNs/App Attest adapter and deployment artifacts.
- Tasks 22.11–22.13 implement privacy, load/failure and persistence conformance.
- Task 17.8 imports encrypted Buzz leases and wake state without contacting a provider.
- Tasks 43.1–43.2 own compatibility negotiation, and Tasks 43.6–43.8 implement mobile API, lifecycle, push/pairing and E2E behavior.
- Tasks 44.3–44.5 own configuration, observability, health and readiness.
- Tasks 46.1–46.6 own cutover and rollback; Tasks 43.5, 43.8 and 43.11 own companion-client E2E evidence, and Tasks 47.1–47.6 gate source retirement.

Approval review requires the target table, App Attest and APNs production/sandbox controls, wake-only noninterference, unavailable-push fallback, encrypted endpoint custody, lease/endpoint lifecycle, provider extension contract and first-cutover compatibility gates above.
