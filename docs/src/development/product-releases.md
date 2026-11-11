# Product releases

Copper uses **push tag → automatically prepare draft → review → publish**.
Pushing a valid Copper tag prepares **Apple Silicon macOS only**. It never
publishes. Stable and prerelease versions use the same process. Main pushes run
CI and release-policy contracts; publication events do not build packages.
Website refresh remains paused, including manual website dispatches.

## Eligibility and version selection

`products/flavors.toml` owns product IDs, release prefixes, supported targets,
required CI, package identities and updater namespaces. Copper (`rust`) uses
`copper-v<semver>`: `copper-v0.1.0` selects stable; `copper-v0.1.0-rc.1` selects
preview and marks the draft as a prerelease. Unknown, ambiguous and malformed
tags fail. Numeric version components must fit platform fields (0–65535).
Groovy and Forge remain planned and retain their independent release controls.
Copper has no repository-variable enablement gate or replacement switch.

Before signing credentials are supplied, preparation verifies the existing tag,
its exact commit, main ancestry, repository identity and the latest successful
same-repository **main-push CI for that exact SHA**, for every catalog-required
workflow. Missing, pending, failed, cancelled, skipped, PR and fork CI do not
qualify. A newer failed attempt invalidates an earlier success. If CI is still
running, wait for success and use manual recovery; never move the tag.

The tag must contain the current release contract and have passed its CI.
The tagged catalog supplies package configuration; current main also checks
that product/targets remain eligible and requires the union of current and
tagged CI requirements. A tag pointing to an older incompatible catalog must
be replaced by a **new version tag** at reviewed current source, never retargeted.
Automatic preparation also binds the pushed tag object (including annotated
tags), not just its peeled commit, and rejects deleted or force-updated events.

The full semantic version reaches application, installer, remote-server and
updater metadata through `RELEASE_VERSION`. Stable and preview keep distinct
application IDs, data namespaces and installer IDs. Numeric macOS bundle fields
and Windows resources encode X.Y.Z; `ProductReleaseVersion`, display versions
and artifact names retain the complete semantic version. Dependency versions
are independent.

## One-time setup

Allow Actions to read workflow results and grant release contents-write only
to `upload_release`. Signing jobs use repository secrets, or organization
secrets explicitly allowed for this repository. Environment-only secrets need
a separately reviewed environment binding. Never put secret values in source,
chat, command arguments or logs.

| Platform | Required GitHub secrets |
| --- | --- |
| macOS | `MACOS_SIGNING_IDENTITY`, `MACOS_CERTIFICATE`, `MACOS_CERTIFICATE_PASSWORD`, `APPLE_NOTARIZATION_KEY`, `APPLE_NOTARIZATION_KEY_ID`, `APPLE_NOTARIZATION_ISSUER_ID` |
| Windows | `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `ACCOUNT_NAME`, `CERT_PROFILE_NAME`, `ENDPOINT`, `WINDOWS_SIGNING_PUBLISHER`, `FILE_DIGEST`, `TIMESTAMP_DIGEST`, `TIMESTAMP_SERVER` |
| Linux | `LINUX_SIGNING_KEY`, `LINUX_SIGNING_KEY_PASSWORD`, `LINUX_SIGNING_KEY_FINGERPRINT` |

macOS needs an Apple Developer account, Developer ID Application certificate
with its private key exported as the existing base64 PKCS#12 secret, and an
App Store Connect notarization API key. Windows needs the existing Azure
Trusted Signing account/profile, service principal access and timestamp service.
Linux uses a passphrase-protected GPG signing key; this is a detached archive
signature, not DEB/RPM or application signing. Account and signing-service fees
are administrator prerequisites, not services provisioned by the workflow.
Renew expiring certificates/keys in their existing secrets and rerun explicit
qualification. Keep temporary signing-store cleanup in the platform scripts.

Qualify actual signed packages, including installed native-agent behavior,
before claiming platform readiness. Policy tests do not prove production trust.
Missing Windows/Linux credentials do not block the automatic macOS scope, but
they **do** fail full production preparation. Adding credentials alone triggers
no preparation and never changes a published release.

## Prepare and publish

These are operator commands, not implementation validation. Choose a reviewed
main commit with successful required CI and a new version:

```sh
git fetch origin main --tags
git tag -a copper-v0.1.0-rc.1 <reviewed-main-SHA> -m 'Copper 0.1.0-rc.1'
git push origin refs/tags/copper-v0.1.0-rc.1
```

Follow `release_prepare` in Actions. Its generated tag filter comes from the
catalog; strict tag validation happens before signing. GitHub path filters
apply to branch pushes, not tag pushes. The automatic matrix contains only
`aarch64-apple-darwin`. Stable tags use the identical flow without `-rc.1`.

After all selected packages pass verification, the upload job creates or resumes
an owned **draft**, labelled **macOS only**, with the DMG, verification receipt,
product/version verification manifest, SHA256SUMS, notes and CI/run evidence.
Review the packages, install and exercise them, and curate the notes while
retaining the ownership and verification markers. Wait for a successful run;
never publish a draft labelled “preparing” or “Preparation is incomplete”.
Then explicitly click **Publish release** in GitHub. No publication event builds
or adds packages.

## Manual qualification, recovery and platform selection

One generated workflow owns all modes; there is no separate signing or
publication workflow to dispatch. The former release, release-promotion and
signing-check entry points remain retired. Archives remain unchanged.

For non-publishing production qualification, leave `release_tag` empty, select
an exact `version`, `preparation_scope=macos-only` or `full`, and set
`sign_artifacts=true`:

```sh
gh workflow run release_prepare.yml --ref <reviewed-ref> \
  -f version=0.1.0-rc.1 -f preparation_scope=macos-only -f sign_artifacts=true
```

This submits to signing/notarization services and uploads Actions artifacts;
it cannot create a tag or GitHub Release. It requires separate operator approval.
With `sign_artifacts=false`, manual qualification produces explicitly labelled
`UNSIGNED-DEVELOPMENT-*` Actions artifacts without signing credentials, including
Windows/Linux when `full` is selected. These packages are development evidence,
not release assets. The upload guard rejects them; no updater consumes CI artifacts.

For recovery after tag-push CI/runner failure, or explicit full-platform
production preparation, dispatch **from main** with the existing tag:

```sh
gh workflow run release_prepare.yml --ref main \
  -f release_tag=copper-v0.1.0-rc.1 -f preparation_scope=macos-only
```

Select `preparation_scope=full` for all Copper catalog targets: Linux x86_64,
macOS ARM64 and Windows x86_64. Full is the manual default. Every selected
platform requires its production credentials and final verification. The tag
supplies the version; qualification's version/signing inputs cannot downgrade
release preparation. No platform is silently skipped and there is no unsigned
fallback. The pipeline calls:

```sh
RELEASE_VERSION=<tag-version> cargo xtask bundle --product rust \
  --platform <platform> --target <triple> --channel <stable-or-preview> \
  --phase package --signing required
```

Existing packaging owners sign nested executables, libraries and the managed
llama.cpp runtime. macOS requires Developer ID Application signatures, secure
timestamps, notarization and stapled tickets for app and DMG. Windows requires
trusted signatures and timestamps for executables, native libraries and installer.
Linux requires verified detached signatures. Platform smoke verification runs
against the final package and installed contents, then emits hash-bound receipts.

To add Windows/Linux later, explicitly prepare a **full** draft with final signed
packages, receipts and regenerated checksums. An existing macOS-only draft has
an immutable scope binding: full preparation stops rather than merging artifact
sets or replacing reviewed notes. Before changing that scope, an operator must
review and retain its notes/evidence and explicitly retire the unpublished draft,
then run full preparation for the same unmoved tag. This is not automatic cleanup.
Published releases cannot be expanded or overwritten; use a new version/tag.

## Safe retries and updater boundaries

The draft marker, plans and receipts bind product, tag, commit, version, channel
and platform scope. Uploads download only selected production artifacts from the
same run. Before creation and every upload, preparation rechecks the remote tag,
main ancestry, required CI and draft ownership. Uploads are bound to the same
release ID. Existing assets are reused only for identical GitHub SHA-256 digests.
Conflicting, unexpected, duplicate or incomplete assets stop the run; there is
no clobber or automatic deletion. Reviewed notes survive retries.

Rerun failed upload jobs with the original successful package artifacts. Fresh
signing can change timestamps and hashes, so a rebuilt package can conflict with
an earlier upload. Review such conflicts rather than replacing assets silently.
Immutability settings that prevent uploads fail with an actionable error. Keep
the release a mutable draft until preparation succeeds. GitHub has no atomic
conditional lock against concurrent human publication; do not publish mid-run.

Drafts never enter updater selection. Even after publication, Copper requires
the final verification marker and complete, digest-bearing package/evidence set.
Unsigned CI artifacts and interrupted drafts are ineligible. macOS-only releases
are eligible only for `--platform macos`; generic, Windows and Linux feeds still
require full-platform releases. Stable and preview remain separate:

```sh
python3 script/product_release.py latest --product rust --channel preview --platform macos
```

Selection uses the catalog prefix and highest semantic version, never GitHub's
globally latest release. Groovy cannot become Copper's latest version. Drafts
use `make_latest=false`. The app retains `rust/<channel>` updater namespaces and
product-ID checks. A reviewed service at `ZED_PRODUCT_UPDATE_BASE_URL` remains
necessary; this change neither deploys one nor resumes website refresh.

## Non-publishing validation

```sh
python3 -m unittest discover -s script -p test_product_release.py
python3 -m unittest discover -s script -p test_refresh_website.py
python3 script/test-production-signing
cargo test -p xtask --locked
cargo xtask products --check
cargo xtask workflows
cargo xtask check-workflows
cargo fmt --all -- --check
./script/clippy --no-all-features -p xtask
```

Git fixtures use disposable local repositories. Policy tests use fake GitHub
responses and signature receipts; these are not hosted builds, actual signatures,
notarization or installed-application evidence. No live tag, workflow dispatch
or release operation is required for these checks.
