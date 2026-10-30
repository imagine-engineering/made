# Requirements: Copper production signing

## Problem

The release packaging jobs can currently fall back to development signatures and do not prove production trust. Copper needs fail-closed signing through its existing packaging owners.

## Scope

Production macOS ARM64 and Windows x86_64 signing, Linux x86_64 detached archive signatures, credential lifecycle, actual-artifact verification, and a non-publishing CI workflow. Publication, promotion, account provisioning, archive edits, and unrelated Sim/branding changes are excluded.

## Requirements

### Requirement 1: Trusted production packages

**User story:** As a Copper user, I want authenticated packages so that I can identify their publisher and provenance.

#### Acceptance criteria

1. **1.1** WHEN a production macOS package is built THEN the packaging script SHALL require Developer ID signing, hardened runtime, nested-code verification, accepted notarization, and valid stapled tickets on the application and DMG.
2. **1.2** WHEN a production Windows package is built THEN the packaging script SHALL require the existing Azure signing service, SHA-256 signatures, trusted timestamps, and executable/library/installer verification.
3. **1.3** WHEN a Linux release archive is produced THEN the workflow SHALL require a verified detached GPG signature and public key without misrepresenting archive signing as DEB/RPM or application signing.
4. **1.4** WHERE managed local inference is available in a production package, THE package SHALL include the reviewed pinned runtime and sign its nested code on macOS and Windows.

### Requirement 2: Safe operation and qualification

**User story:** As a maintainer, I want safe signing and qualification so that missing credentials cannot cause unsigned publication.

#### Acceptance criteria

1. **2.1** IF required production credentials or verification are unavailable THEN packaging SHALL fail before uploading a production artifact; development builds SHALL remain usable without credentials.
2. **2.2** THE signing process SHALL avoid logging credentials and SHALL clean temporary stores on success and failure without changing the user's default keychain.
3. **2.3** THE existing release eligibility, exact version selection, and product feature boundaries SHALL remain intact.
4. **2.4** THE repository SHALL provide a manual non-publishing CI workflow reusing the production packaging jobs and record actual versus blocked cryptographic verification.
5. **2.5** THE current documentation SHALL identify exact secret names, provisioning/renewal, recovery, costs/prerequisites, and remaining administrator actions.

## Constraints

No credentials in source or logs; no ad-hoc substitute for production signatures. Preserve prior release qualification and all archived content. Do not claim production verification without actual signed artifacts.
