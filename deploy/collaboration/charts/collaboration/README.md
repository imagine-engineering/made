# Collaboration Helm chart

This chart owns the Kubernetes shape of the canonical Zed collaboration
service. It is disabled by default. A production release must supply an
immutable image digest, existing runtime and migration Secrets, HTTPS service
endpoints, an Ingress or Gateway API attachment, and explicit egress ranges.

The runtime Secret must contain the keys named under `runtimeSecret` in
`values.yaml`. Database, object-store, internal API, Git-hook, push and mesh
credentials are never accepted as chart values. The migration Secret is
separate because it may have DDL privileges that the runtime identity must not
have.

`values-production.yaml` is intentionally unrenderable on its own. The release
system supplies environment-owned values, including a separately published
`collaboration-migrations` digest, then runs:

```sh
helm lint deploy/collaboration/charts/collaboration \
  -f deploy/collaboration/charts/collaboration/values-production.yaml
helm template collaboration deploy/collaboration/charts/collaboration \
  -f deploy/collaboration/charts/collaboration/values-production.yaml
```

The pre-install/pre-upgrade hook runs the Task 44.4 migration artifact with a
DDL-only Secret. Its immutable manifest owns ordering, checksums, compatibility
ceiling and halt/resume behavior independently from the runtime image.

Rollback overlays `values-rollback.yaml`, selects only a previous immutable
digest, requires that binary to admit the deployed schema floor, and suppresses
the migration hook. It leaves the Git claim and every external authority
unchanged.

Run `tests/render.sh` from the repository root to lint and render the default,
production, Ingress, optional pairing/push/mesh and rollback contracts, plus
their failure cases.

## Optional signed profile HTTP

Set `nostrHttp.enabled: true`, explicit `nostrHttp.routes` and
`nostrHttp.callsPerMinute` to enable the existing profile service. For example:

```yaml
nostrHttp:
  enabled: true
  routes:
    - communityId: 11111111-1111-4111-8111-111111111111
      publicUrl: https://collaboration.example.com
  callsPerMinute: 60
```

Each mapping must name the existing canonical community and its current serving
host. Attach every configured host through the existing TLS Ingress or Gateway
hostnames. The chart emits the canonical `COLLABORATION_NOSTR_ROUTES` JSON and
`COLLABORATION_NOSTR_API_CALLS_PER_MINUTE` fields and exact `/events` and `/query`
paths. The existing root, `/info` and NIP05 routes remain composed by the service.
Redis and its runtime Secret key are mandatory. Zero is an explicit deny-all
quota, not a missing setting. Quotas are shared by the real Redis owner.

Set optional `nostrHttp.closedWebsocket: true` to opt into the existing
closed-current-member NIP42 WebSocket on the same exact root route. This requires
`nostrHttp.enabled: true`, explicit routes/quota and shared Redis. The optional
value defaults to `null`, omitting `COLLABORATION_NOSTR_CLOSED_WEBSOCKET`; explicit
Boolean `false` and `true` are emitted unchanged and participate in the existing
rollout checksum. It adds no public path or exposure and does not enable general
subscriptions or grant membership/roles. Rendering is not service qualification.

Disabled defaults omit optional environment fields and additional public paths.
Changing this configuration participates in the pod configuration checksum.
It does not create communities, membership, accepted names or signing authority;
startup and each canonical transaction retain their actual host/lifecycle,
signature, tenant, authorization and SQL/RLS admission checks. Configuration
rendering is not production service or deployment certification.
