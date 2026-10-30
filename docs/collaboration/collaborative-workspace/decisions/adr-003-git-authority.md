# ADR-003: Hosted Git Authority

- **Status:** Accepted
- **Decision date:** 2026-08-14
- **Approval:** The product owner approved a per-repository hosted authority. Zed remains the local project and Git owner; each hosted repository selects either Zed NIP-34 hosting or one external provider as its sole hosted authority.
- **Requirements:** 2.1, 10.1, 10.2
- **Capabilities:** CAP-018, CAP-019, CAP-020

## Context

Zed already owns local repositories, working trees, indexes, branches, diffs and native review UI. It also recognizes many Git hosting providers from repository remotes. Buzz adds NIP-MP project grouping, NIP-34 repositories, patches, pull requests, issues and statuses, NIP-98 Git authentication, Nostr commit/tag signing and a hosted Git implementation backed by content-addressed packs plus an object-store manifest pointer.

Neither a community project nor a collaboration channel may acquire authority over the member repositories it groups. Supporting Zed hosting and external forges also cannot mean writing hosted refs, pull requests or approvals to two providers and reconciling whichever responds last.

## Decision

### Local authority is unchanged

For every open repository, Zed's existing `project`, `worktree`, `git`, `project::git_store` and `git_ui` components remain the canonical owners of local filesystem content, the working tree, index, local refs, configured remotes, local diff state and native keep/reject/stage actions.

Collaborative Workspace is another presentation over those entities. NIP-MP groups, channel bindings, timeline events and hosted-provider records reference stable repository identities; they never reconstruct or fork local Git state.

### One hosted authority per repository

Each collaboration repository identity has one versioned `HostedAuthority` selection:

- **SimHostedNip34:** the Zed collaboration platform hosts Git and NIP-34 forge records; or
- **ExternalProvider:** one provider instance and repository coordinate supported by `git_hosting_providers` or a versioned provider adapter.

An unhosted local repository may have no hosted authority. Multiple Git remotes may still exist locally, but only the selected authority can be used by canonical hosted-ref, pull-request, issue, status, approval and merge commands. Mirrors are explicitly read-only or derived and cannot accept authoritative mutations.

Authority selection is tenant-fenced, repository-specific and optimistic-versioned. Project membership, channel membership, NIP-MP signing and project ownership do not imply write access to a member repository. Every operation rechecks repository authority and provider-specific permission.

### Authority table

| State or operation | Zed-hosted repository | External-provider repository | Always canonical locally |
| --- | --- | --- | --- |
| Hosted refs and default branch | Object-store manifest pointer updated by linearizable CAS | Selected provider's advertised refs/API | Local refs remain owned by native Git and move only through explicit fetch/push/checkout |
| Pack/object durability | Zed content-addressed, create-only object store and published manifest | Selected provider | Local object database |
| Push acceptance | Zed Git policy plus successful manifest CAS | Selected provider's receive/API result | Local commit/index creation |
| Repository announcement | Signed NIP-34 authoritative record linked to the manifest repository | Provenance-aware NIP-34 compatibility projection of provider identity | Stable Zed repository mapping |
| Portable signed patch | Immutable signed NIP-34 event in the collaboration event log | Same portable signed event; applying or accepting it still targets the selected provider | Native diff materialization for review |
| Pull request and issue lifecycle | Authorized signed NIP-34/domain records | Selected provider records; NIP-34/Zed representations are versioned projections | Local working changes are unaffected until explicit Git operations |
| Review comments and approvals attached to hosted review | Authorized collaboration review records and signed compatibility events | Selected provider records when the action is provider review; ordinary collaboration discussion remains a linked message, not a provider approval | Native review UI projects either authority without owning the hosted decision |
| CI/status/check result | Authorized service/provider record in canonical collaboration storage | Selected provider/check producer record, imported with source ID/version | Local command/test activity remains agent/action-log state |
| Merge decision and hosted ref mutation | Zed policy followed by manifest CAS | Selected provider merge result | Local branch/worktree updates only after explicit fetch/update |
| Ref-change event | Derived notification after the authoritative hosted ref commits | Derived provider event/webhook after provider confirmation | Timeline projection only |

For Zed-hosted Git, the mutable manifest pointer is the sole hosted-ref commit point. Pack and manifest objects are written first; client-visible push success and ref-change events occur only after successful CAS. Relay events are signals to reread the pointer and cannot establish ref state.

For an external provider, an adapter submits one provider operation with a stable idempotency/source key where supported, waits for the provider's authoritative response, then records a provenance-aware projection. A timeout or ambiguous provider result is surfaced as unknown and reconciled by rereading the provider; it is never converted into a successful local-only review or ref mutation.

### Projects and cross-owner grouping

NIP-MP project groups contain stable repository references and optional channel bindings. Each repository retains its own hosted authority, owner and authorization policy. A project containing repositories owned by different people or organizations does not create a common signing key, push grant, review grant or provider token. Project-level actions fan out into separately authorized repository commands and report partial outcomes without weakening a failed repository's policy.

### Repository identity and provider mapping

A stable collaboration repository ID maps local repository identity, normalized remote coordinates, NIP-34 coordinates and the selected hosted authority. Provider URLs are parsed by the existing registry, but URL recognition is not proof of authority or permission. Mappings record provider instance, repository coordinate, source version, verification evidence and last reconciled head.

Mappings fail closed on ambiguous remotes, conflicting NIP-34 announcements, tenant mismatch, provider-instance mismatch or stale authority versions. Credentials remain owned by Zed credential providers and are scoped to the selected provider and repository.

### Changing hosted authority

Authority transfer is an explicit migration, not a settings toggle. It requires:

1. authorization from the current and target repository policies;
2. a write freeze for hosted mutations;
3. source inventory of refs, default branch, protection rules, open patches/PRs/issues, reviews, approvals and statuses;
4. content and ref transfer with object/hash verification;
5. differential reads and a recorded last source version;
6. one atomic mapping-version activation; and
7. disabling writes through the former authority before target writes begin.

If activation fails, the old authority remains selected and writable only after reconciliation confirms no target mutation escaped. After activation, the old provider is a read-only compatibility source for the bounded support window. Rollback freezes writes, verifies both heads and review state, restores the prior mapping only when it can do so without losing target-only mutations; otherwise it requires an approved forward repair.

Permanent bidirectional ref, pull-request, issue, approval or status mirroring is prohibited. A long-term compatibility adapter may translate reads and operations, but every response identifies the selected authority and source version.

### Patches, reviews and native diffs

Signed patches remain portable collaboration artifacts and do not mutate a hosted repository by themselves. Applying a patch materializes changes through native Zed project/Git state. Publishing a branch, opening a pull request, approving or merging routes to the repository's selected hosted authority.

The native review surface renders unified/split diffs from canonical local Git state or verified hosted commit/blob coordinates. Timeline records link repository, commit, branch, patch, review and CI identities with provenance. Keep, reject and stage remain local actions; hosted review comments, approval and merge actions are enabled only when the selected authority supports them and the latest authority version/head is known. Stale or conflicting state is visible and fails closed.

### Authentication and signing

Zed-hosted Git preserves NIP-98 authentication and repository permission checks. External-provider operations use provider-scoped credentials and existing provider authentication. Nostr commit and tag signing is an explicit Git signing method backed by Zed's canonical credentials provider; it does not replace provider authentication or grant hosted permission. Verification surfaces distinguish commit/tag signatures, signed NIP-34 events and provider-authenticated actions.

## Compatibility and migration consequences

- Existing Buzz-hosted repositories import into `SimHostedNip34`; their verified manifest pointer becomes the hosted-ref authority without changing object IDs.
- Existing Zed repositories with recognized external remotes default to that verified provider only after mapping confirmation; source detection alone does not activate write authority.
- Existing local-only repositories remain unhosted until the user or administrator selects and verifies a host.
- NIP-34, NIP-98 and Git smart-HTTP wire behavior remains available through adapters for Zed-hosted repositories.
- External repositories can expose NIP-34 compatibility projections without claiming that the relay owns provider refs or review decisions.

## Alternatives rejected

1. **Make Zed hosting authoritative for every repository:** rejected because it would silently fork established external-provider refs and review state.
2. **Treat Zed NIP-34 hosting as merely another simultaneous remote authority:** rejected because dual authoritative refs, approvals and merges cannot be reconciled safely.
3. **Let NIP-MP project ownership grant member-repository access:** rejected because cross-owner grouping is organizational metadata, not a Git capability.
4. **Use ref-change events as the Zed-hosted commit point:** rejected because events may lag, duplicate or replay; the manifest CAS is the durability fence.
5. **Make native diff UI the review-state authority:** rejected because presentation and local staging actions must not overwrite hosted provider or signed review decisions.

## Implementation and validation trace

- Tasks 24.1–24.5 implement project grouping, stable repository identity and non-escalating channel bindings.
- Tasks 25.1–25.10 implement NIP-34 codecs, hosted authority schema, object storage, Git smart HTTP, authentication, signing and conformance.
- Tasks 26.1–26.5 link branch activity to channels without making events authoritative for refs.
- Tasks 27.1–27.7 implement provenance-aware review, approval, CI and native diff projections.
- Task 25.10 owns repository permission checks; Tasks 36.3–36.4 enforce common authorization and operator APIs, and Tasks 43.9–43.11 expose canonical administration through the admin client without project-level privilege escalation.
- Tasks 46.1–46.6 own authority migration, reconciliation and rollback; Tasks 47.1–47.6 gate legacy write removal, and Tasks 48.1–48.4 retain final ownership and parity evidence.

Decision-table validation requires one named authority for local working state, hosted refs, pack durability, patches, pull requests, issues, review comments, approvals, statuses and merges in Zed-hosted, external-provider and unhosted cases. Conformance must cover cross-owner projects, ambiguous remotes, stale mappings, provider timeouts, CAS races, partial migration and rollback without concurrent writers.
