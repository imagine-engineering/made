# ADR-002: Zed Account and Nostr Identity Binding

- **Status:** Accepted
- **Decision date:** 2026-08-14
- **Approval:** The product owner approved the recommended model: a Zed account may use multiple community-local Nostr identities, with exactly one active signing identity for each community/profile.
- **Requirements:** 2.1, 7.1, 7.4
- **Capabilities:** CAP-007, CAP-008, CAP-009

## Context

Zed service accounts and organizations currently identify users for hosted services, while Buzz identities are Nostr keypairs whose public keys author immutable signed events. Buzz also supports independent agent keys and NIP-OA owner attestations. Conflating these concepts would either make a service-account mutation rewrite cryptographic authorship or make possession of a Nostr key sufficient to assume unrelated service-account privileges.

The migration must preserve existing npubs, signatures, profiles, social lists, agent provenance and archive history. It must also use Zed credential providers for secret-key custody, work across multiple communities, prevent ambiguous active signers and fail closed when verification or protected storage is unavailable.

## Decision

### Separate identity concepts

The canonical model keeps four related concepts distinct:

1. A **Zed service account** is the canonical human account, subscription and organization principal owned by `client::UserStore` and Zed service authentication.
2. A **Nostr signing identity** is a public key and protected signing-key reference. Its public key is the immutable author of events it signs.
3. A **community profile** is community-scoped display, status, social and archival state associated with a signing identity.
4. An **agent identity** is its own Nostr signing identity and profile. A verified NIP-OA attestation records authorization provenance from an owner identity but never changes the event author.

Bindings connect these concepts explicitly; they do not merge their identifiers or histories.

### Cardinality and active signer

A Zed service account may bind zero or more Nostr identities. Bindings are community-scoped and profile-scoped so the same account may intentionally use different npubs in different communities or distinct profiles in one community.

The uniqueness rules are:

- exactly zero or one binding may be `active` for a `(community_id, service_account_id, profile_id)` tuple;
- one binding record names exactly one Nostr public key and one service-account/profile tuple;
- the same public key may be recognized for the same service account in multiple communities without sharing membership, roles, rate limits or active-state decisions;
- within a community, a public key cannot be actively bound to two service accounts or two conflicting human owners;
- historical, rotated, revoked and archived binding versions remain addressable but cannot become active through an ordinary profile update; and
- agent keys remain separate authors and are related to owners only by a verified, bounded attestation or explicit managed-agent record.

Selecting a different presentation or switching communities changes the resolved active binding; it never copies a key or rewrites an event.

### Binding record and authority

The Zed collaboration identity-binding repository is the canonical owner of binding state. Each version records:

- community, service-account, profile and public-key identifiers;
- binding status (`pending`, `verified`, `active`, `rotated`, `revoked` or `archived`);
- verification method and evidence reference;
- predecessor/successor binding versions where applicable;
- creation, verification, activation and terminal timestamps;
- organization-policy version and actor principal; and
- an optimistic-concurrency version and audit reference.

No binding table stores a private key, backup phrase, raw attestation secret or reusable challenge. Zed's canonical credentials provider owns protected secret references. The immutable signed event log remains the authorship authority; a binding is authorization and presentation metadata, not permission to rewrite an author.

### Create and link verification

A newly generated key is created in the canonical credentials provider, round-trip tested with a challenge signature and exposed to binding activation only after protected storage confirms the exact public key. An imported, paired or restored key remains in its source until the canonical provider verifies storage and signs a fresh domain-separated challenge.

Linking an existing public key requires both authenticated control of the Zed service account and proof of current key possession. The service issues a single-use, short-lived, community- and account-bound challenge. The Nostr key signs the exact challenge; replay, host/community mismatch, expired evidence, unsupported signature formats and public-key mismatch fail before a binding write. Organization administrators cannot forge possession evidence.

A pending binding grants no signing, membership, role or recovery authority. Activation is an optimistic transaction that verifies policy and uniqueness, makes the selected binding active and, when rotating, transitions the predecessor in the same transaction.

### Rotation and revocation

Rotation creates and verifies a successor key before changing active state. The atomic activation transaction marks the prior binding `rotated`, links predecessor and successor versions, updates future signing resolution and invalidates cached/autocomplete active-identity projections. If successor storage, challenge verification, policy evaluation or persistence fails, the prior active identity remains unchanged and usable.

Revocation is a fail-closed authorization action. It prevents future signatures through Zed, active access derived from the binding, autocomplete as an active identity and issuance of new owner attestations. It does not invalidate historical event signatures or erase public profiles required to attribute retained events. Compromise revocation also cancels outstanding pairing, recovery and delegation evidence for that key and requires reauthentication under organization policy.

NIP-OA attestations remain independently verifiable historical evidence. Revoking an owner or agent binding prevents new Zed-authorized use but cannot cryptographically retract already signed attestations; policy evaluates their bounds, event time and current authorization where required.

### Archive and relay-scoped state

Archiving hides the identity from active selection, mentions and discovery according to community policy and removes active authorization, while retaining public-key, profile tombstone, binding versions and historical authorship. Relay-scoped archive events remain compatibility representations of the canonical community profile transition. Unarchive requires a still-valid verified key, current membership and policy authorization; a revoked or rotated binding cannot be silently resurrected by an archive toggle.

Service-account deletion and community departure detach active service authorization but follow retention policy for binding evidence and signed-event attribution. They do not delete cryptographic history merely because the service account is gone.

### Recovery and backup

Recovery proves control of a recoverable secret or approved organization recovery factor and then imports the key into Zed's canonical credentials provider. NIP-49 `ncryptsec`, legacy nsec/hex and approved pairing formats remain compatibility inputs. Every recovery:

1. applies bounded parsing and KDF/resource limits;
2. redacts private material from logs, telemetry and errors;
3. verifies the recovered public key against the intended binding;
4. writes protected canonical storage and signs a fresh round-trip challenge;
5. leaves the old source intact until the user confirms successful activation; and
6. records recovery evidence without recording the secret.

Protected-storage corruption or unavailability never generates a replacement key. The operation fails safely and presents recovery or owner-only documented fallback actions. A recovered revoked identity remains revoked; recovery proves possession, not current authorization. Organization recovery cannot replace Nostr authorship and must either recover the same key or create a separately verified successor through rotation.

### Organization policy

Organization policy may require managed storage, prohibit raw secret export, restrict key generation/import/pairing, require reauthentication or multiple approvers for recovery/rotation/revocation, set attestation lifetimes, and require escrow-compatible recovery. Policy can narrow behavior but cannot:

- expose or centrally derive an unmanaged user's private key;
- bind a key without possession proof;
- silently replace an identity when storage fails;
- assign one community key to conflicting account owners;
- reinterpret an agent's NIP-OA-authorized event as authored by its owner; or
- delete historical authorship needed by retention and audit policy.

Policy is evaluated from the host-derived community and authenticated organization context. Client-supplied community, profile or organization identifiers cannot widen it. A policy-version change re-evaluates future actions and access; it does not mutate past signatures.

## State transitions

| Operation | Preconditions | Atomic result | Failure result |
| --- | --- | --- | --- |
| Create | Authenticated account, allowed policy, protected storage available | Verified key reference and pending/active binding | No binding; no synthetic identity |
| Link | Account auth plus fresh key-possession proof | Verified binding eligible for activation | Challenge consumed or expires; no authority granted |
| Activate | Verified binding, unique tuple, current policy | One active binding; prior active rotates if supplied | Existing active binding is unchanged |
| Rotate | Verified successor plus current active authority | Successor active, predecessor rotated and linked | Predecessor stays active |
| Revoke | Authorized, reauthenticated actor and current version | Binding revoked; future authorization and cached active state invalidated | Prior state retained; failure audited |
| Archive | Authorized community/profile transition | Profile and binding unavailable for active selection; history retained | Active state retained |
| Restore backup | Valid bounded format, intended pubkey and protected storage | Verified canonical key reference; binding state unchanged until activation | Source preserved; no key or binding replacement |
| Recover account access | Approved factor/policy and proof for same key or verified successor | Same binding re-enabled only if not terminal, or explicit rotation begins | Revoked/rotated key is not resurrected |

## Security and isolation consequences

- Authentication yields a typed principal containing independently verified service-account and/or Nostr-key claims; authorization decides when an explicit binding is required.
- Membership, roles, challenges, binding uniqueness, caches and rate limits are scoped by trusted `CommunityId` even when the same npub appears in another community.
- Private-key material exists only behind canonical credentials-provider handles and is never persisted in collaboration tables.
- Historical event authorship is stable across account rename, organization transfer, rotation, revocation, archive or recovery.
- Owner attestations communicate agent provenance, never impersonation or author substitution.

## Alternatives rejected

1. **One global npub per Zed account:** rejected because it prevents community-local identity and privacy policy while forcing cross-community correlation.
2. **Unlimited simultaneously active keys per profile:** rejected because signing, mentions, permissions and recovery would have ambiguous authority.
3. **Treat any verified npub as a Zed account:** rejected because key possession does not grant unrelated hosted account, organization or billing privileges.
4. **Let administrators relink keys without possession proof:** rejected because it enables authorship takeover.
5. **Rewrite or delete historical authorship after rotation/revocation:** rejected because signed-event authorship is immutable and required for compatibility and audit.
6. **Treat NIP-OA owner attestation as owner authorship:** rejected by the protocol and by the independent human/agent identity model.

## Implementation and validation trace

- Tasks 12.1–12.3 implement binding records, profile separation and tenant-fenced persistence.
- Tasks 12.4–12.6 implement protected import, lifecycle, backup and recovery behavior.
- Task 12.7 implements optimistic repository transitions and isolation.
- Tasks 13.1–13.6 apply trusted tenant and common authorization boundaries.
- Tasks 17.3 and 17.6 import existing identities, credentials and profile state without deleting sources before verification.
- Tasks 40.1–40.6 preserve pairing compatibility and replay/expiry controls.

Identity review acceptance requires scenarios for create, link, rotate, revoke, archive, restore and recover; conflict and replay failures; protected-storage unavailability; cross-community reuse without authority leakage; agent-owner provenance; historical authorship; and organization policies that narrow rather than silently replace identity authority.
