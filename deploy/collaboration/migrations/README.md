# Collaboration schema migrations

This directory packages the canonical Collab SQL migration pairs as one
release artifact. `manifest.json` fixes their order, forward and backward
SHA-256 values, current schema ceiling, and the service-activation rollback
boundary. The manifest must cover every `*.up.sql` and `*.down.sql` file under
`crates/collab/migrations`; `check.py` rejects omissions or changed bytes.

The current ceiling is `20261026000100`, including the channel/principal bindings,
read-state tables and room authorization generation required by the current
production owners, canonical identity profiles, verified registration/observation,
signed moderation subject targets, and the same-tenant accepted NIP-05 name
projection with retained signed provenance, and signed restriction reasons within
the existing EVENT bound. All original 30 pairs retain their SQL checksums. The additive 31st pair repairs
version retirement in the presence of stored generated columns while retaining
immutability of every ordinary column and the prohibition on deletion. The shared
Zed schema must already exist before these collaboration migrations run. The
disposable smoke test initializes it from the canonical Collab test schema dump;
the migration image does not take ownership of the shared Cloud schema.

The runner uses `DATABASE_URL` without placing it in process arguments. It
accepts PostgreSQL URLs with explicitly allowlisted libpq connection options,
creates an operator-owned history and control record, revokes their public
privileges, takes a session advisory lock, and applies each SQL file and its
history row in one transaction. A terminated process can resume from the
committed prefix. A SQL error or source/history checksum mismatch records a
closed halt reason and blocks further migration until operators restore or
repair the database under an approved recovery procedure.

Forward application defaults to the manifest ceiling:

```sh
DATABASE_URL=postgres://... ./deploy/collaboration/migrations/migrate.py up
```

`seal --expected-version VERSION` advances the monotonic rollback floor after
service activation. Before that boundary, `down --target-version VERSION`
applies checked backward files in reverse order. It can never cross the stored
floor. Normal service-image rollback does not run this command; the Helm chart
selects a schema-compatible prior binary while preserving data.

The Kubernetes hook uses the separately built migration image and a DDL-only
Secret. The chart's required schema version must equal the packaged manifest
ceiling before any database connection. Build it from the repository root;
release automation must pin both its base image and published result by digest:

```sh
docker build -f deploy/collaboration/migrations/Dockerfile .
```

`tests/smoke.sh` builds the artifact and uses a disposable PostgreSQL instance
to prove staged apply/resume, idempotence, backward execution before the
boundary, refusal after sealing, and durable checksum-drift halt behavior.

Buzz's `1321_backfill_default_community.sql` is deliberately excluded. It is a
one-off source-normalization cutover with no down path; its required pre-run
snapshot remains its rollback authority. Canonical Collab schema application
starts only after the separately owned Buzz import path has verified that
source state.

The accepted-name layer may be reversed only before any retained name decision
exists. Its DOWN guard requires all-tenant visibility and refuses to discard
processed or accepted signed provenance. Disabling the feature does not remove
that data or lower the service schema requirement.

The signed-reason pair retains the general 1024-byte public-reason budget and
extends only genuine Nostr restriction action provenance to the existing
512 KiB event bound. The ingress still bounds the complete signed EVENT frame;
no reason is truncated or copied into the bounded security audit payload.
DOWN refuses, under every-tenant visibility and a table lock, if any longer
reason remains. Retained reasons must be preserved; rollback never clears them.
