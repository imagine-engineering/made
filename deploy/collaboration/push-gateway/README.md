# Collaboration push gateway deployment contract

This Helm chart is disabled by default. It records the configuration, migration,
health and resource contract for the collaboration push gateway without making
the service deployable from repository defaults.

An enabled release must provide an HTTPS delivery URL, the pinned App Attest
application/root reference, at least one of the two approved APNs profiles and
separate runtime and DDL-capable database secrets. The runtime secret contains
`DATABASE_URL`, `ZED_PUSH_GRANT_KEYS` and `ZED_PUSH_TOKEN_KEYS`; the latter two
must be different entries. Each enabled APNs profile uses separate credential
and configuration Secrets. Secret values are never accepted through chart
values or rendered into manifests.

The public Service exposes only port 8080. Liveness, dependency-aware readiness
and metrics share the pod-private port 8081; metrics ingress is available only
when a PodMonitor and non-empty monitoring namespace/pod selectors are enabled
together. Readiness is expected to reject missing or mismatched database,
schema, App Attest, keyring, profile, topic or credential authority.

Forward upgrades use the pre-upgrade migration Job with a DDL-only secret. A
rollback selects a previous immutable image through `rollback.targetImageDigest`,
proves that binary accepts the current schema through
`rollback.maximumSchemaVersion`, and renders no migration Job. Rollback never
attempts a down migration.

Validate all normal, missing-secret and rollback cases from the repository root:

```sh
deploy/collaboration/push-gateway/tests/render.sh
```
