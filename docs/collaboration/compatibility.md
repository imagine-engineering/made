# Collaborative Workspace compatibility policy

This document is the release authority for client, service, protocol and stored-schema compatibility during the Buzz-to-Zed migration. All ranges are inclusive and closed. A version or combination absent from the machine-readable matrix is unsupported; operators and clients must not infer compatibility from a nearby version.

The policy applies to Multiplayer builds. Standard Zed does not include the collaboration capability and returns `not_included_in_build` before tenant or resource lookup.

## Client support window

| Client | Supported versions | Negotiation | Reads | Writes |
| --- | --- | --- | --- | --- |
| Zed desktop | 1.16.2 | Direct HTTP/RPC preflight | Direct | Direct |
| Buzz desktop | 0.5.11 | Version asserted by the legacy adapter | Adapter | Adapter |
| Buzz mobile | 0.0.0+1 | Version asserted by the legacy adapter | Adapter | Adapter |
| Buzz web | 0.1.0 | Version asserted by the legacy adapter | Adapter | Adapter |
| Buzz CLI shim | 0.1.0 | Local protocol/minimum-version preflight | Adapter | Adapter |
| Buzz admin web | 0.1.0 | Version asserted by the legacy adapter | Adapter | Adapter |

“Adapter” means that the retained client surface reaches the same canonical command/query owners through a bounded compatibility adapter. It never means that the legacy client may write a Buzz database, dual-write authorities or bypass common authorization. These exact frozen versions remain supported until the retirement gates in the migration plan approve a narrower window.

## Service, protocol and schema windows

| Service | Supported versions | Stored schema |
| --- | --- | --- |
| Collab | 0.44.0 | Canonical collaboration Postgres 20261010000600 |
| Buzz relay adapter | 0.1.0 | Canonical collaboration Postgres 20261010000600 |
| Push gateway | 0.1.0 | Canonical push Postgres 20260822000200–20261010000600 |
| Pair relay | 0.1.0 | None |

| Protocol | Supported versions | Write-bearing |
| --- | --- | --- |
| Collaboration HTTP negotiation | 1 | Yes |
| Zed RPC | 68 | Yes |
| Nostr ingress | 1 | Yes |
| Canonical domain command | 1 | Yes |
| Buzz CLI forwarding | 1 | Yes |
| NIP-AB pairing | 1 | Yes |
| NIP-44 payload | 2 | Yes |
| NIP-PL push lease | 1 | Yes |
| Buzz audio gateway | 1–2 | No; media transport only |

Buzz Postgres schema 30 is supported only as a read-only import source. It is not a serving schema for any listed canonical service.

## Machine-readable matrix

The JSON block is normative. Release tooling must parse the block whose `matrix_schema` is `1`, reject duplicate or missing IDs and apply every range exactly as written.

```json compatibility-matrix
{
  "matrix_schema": 1,
  "policy_version": 1,
  "published_at": "2026-08-25",
  "clients": [
    {
      "id": "zed-desktop",
      "minimum_version": "1.16.2",
      "maximum_version": "1.16.2",
      "negotiation": "direct",
      "read_mode": "direct",
      "write_mode": "direct",
      "service": "collab",
      "protocols": ["collaboration-http@1", "zed-rpc@68"],
      "schema": "canonical-collaboration-postgres",
      "incompatible_write": "upgrade_required_before_tenant_lookup"
    },
    {
      "id": "buzz-desktop",
      "minimum_version": "0.5.11",
      "maximum_version": "0.5.11",
      "negotiation": "adapter_asserted",
      "read_mode": "adapter",
      "write_mode": "adapter",
      "service": "collab",
      "protocols": ["collaboration-http@1", "nostr-ingress@1"],
      "schema": "canonical-collaboration-postgres",
      "incompatible_write": "upgrade_required_before_tenant_lookup"
    },
    {
      "id": "buzz-mobile",
      "minimum_version": "0.0.0+1",
      "maximum_version": "0.0.0+1",
      "negotiation": "adapter_asserted",
      "read_mode": "adapter",
      "write_mode": "adapter",
      "service": "collab",
      "protocols": ["collaboration-http@1", "nostr-ingress@1", "nip-ab@1", "nip44-payload@2"],
      "schema": "canonical-collaboration-postgres",
      "incompatible_write": "upgrade_required_before_tenant_lookup"
    },
    {
      "id": "buzz-web",
      "minimum_version": "0.1.0",
      "maximum_version": "0.1.0",
      "negotiation": "adapter_asserted",
      "read_mode": "adapter",
      "write_mode": "adapter",
      "service": "collab",
      "protocols": ["collaboration-http@1", "nostr-ingress@1"],
      "schema": "canonical-collaboration-postgres",
      "incompatible_write": "upgrade_required_before_tenant_lookup"
    },
    {
      "id": "buzz-cli",
      "minimum_version": "0.1.0",
      "maximum_version": "0.1.0",
      "negotiation": "local_preflight",
      "read_mode": "adapter",
      "write_mode": "adapter",
      "service": "collab",
      "protocols": ["buzz-cli-forward@1", "collaboration-http@1"],
      "schema": "canonical-collaboration-postgres",
      "incompatible_write": "upgrade_required_before_tenant_lookup"
    },
    {
      "id": "buzz-admin-web",
      "minimum_version": "0.1.0",
      "maximum_version": "0.1.0",
      "negotiation": "adapter_asserted",
      "read_mode": "adapter",
      "write_mode": "adapter",
      "service": "collab",
      "protocols": ["collaboration-http@1"],
      "schema": "canonical-collaboration-postgres",
      "incompatible_write": "upgrade_required_before_tenant_lookup"
    }
  ],
  "services": [
    {
      "id": "collab",
      "minimum_version": "0.44.0",
      "maximum_version": "0.44.0",
      "protocols": ["collaboration-http@1", "zed-rpc@68", "nostr-ingress@1", "domain-command@1", "buzz-audio@1", "buzz-audio@2"],
      "schema": "canonical-collaboration-postgres"
    },
    {
      "id": "buzz-relay-adapter",
      "minimum_version": "0.1.0",
      "maximum_version": "0.1.0",
      "protocols": ["collaboration-http@1", "nostr-ingress@1", "domain-command@1"],
      "schema": "canonical-collaboration-postgres"
    },
    {
      "id": "push-gateway",
      "minimum_version": "0.1.0",
      "maximum_version": "0.1.0",
      "protocols": ["nip-pl@1", "nip44-payload@2"],
      "schema": "canonical-push-postgres"
    },
    {
      "id": "pair-relay",
      "minimum_version": "0.1.0",
      "maximum_version": "0.1.0",
      "protocols": ["nip-ab@1", "nip44-payload@2"],
      "schema": "none"
    }
  ],
  "protocols": [
    {"id": "collaboration-http", "minimum_version": 1, "maximum_version": 1, "writes": true},
    {"id": "zed-rpc", "minimum_version": 68, "maximum_version": 68, "writes": true},
    {"id": "nostr-ingress", "minimum_version": 1, "maximum_version": 1, "writes": true},
    {"id": "domain-command", "minimum_version": 1, "maximum_version": 1, "writes": true},
    {"id": "buzz-cli-forward", "minimum_version": 1, "maximum_version": 1, "writes": true},
    {"id": "nip-ab", "minimum_version": 1, "maximum_version": 1, "writes": true},
    {"id": "nip44-payload", "minimum_version": 2, "maximum_version": 2, "writes": true},
    {"id": "nip-pl", "minimum_version": 1, "maximum_version": 1, "writes": true},
    {"id": "buzz-audio", "minimum_version": 1, "maximum_version": 2, "writes": false}
  ],
  "schemas": [
    {
      "id": "canonical-collaboration-postgres",
      "minimum_version": "20261010000600",
      "maximum_version": "20261010000600",
      "writers": ["collab@0.44.0"],
      "below_minimum": "service_not_ready",
      "above_maximum": "binary_schema_incompatible"
    },
    {
      "id": "canonical-push-postgres",
      "minimum_version": "20260822000200",
      "maximum_version": "20261010000600",
      "writers": ["push-gateway@0.1.0", "collab@0.44.0"],
      "below_minimum": "service_not_ready",
      "above_maximum": "binary_schema_incompatible"
    },
    {
      "id": "buzz-postgres",
      "minimum_version": "30",
      "maximum_version": "30",
      "writers": [],
      "below_minimum": "import_rejected",
      "above_maximum": "import_rejected"
    }
  ]
}
```

`buzz-postgres@30` is an import and retained-reference schema, not a serving target after canonical write cutover. `pair-relay` owns no stored schema. The canonical schema maximum is the newest migration validated for the listed binary, not permission to run an older binary against a newer database. Rollback tooling must use the maximum declared by the rollback binary and must never down-migrate authoritative data.

## Negotiation contract

Before any collaboration write, the client or its versioned adapter sends its exact client ID, client version, requested feature IDs and protocol versions. The service responds without consulting tenant, membership, resource or content state:

- `supported`: every requested write feature and protocol is present, the client is inside its closed version range and the ready service schema is inside every selected service range;
- `read_only`: the identified client/version has an explicitly supported read projection, but at least one requested write feature is unavailable; and
- `upgrade_required`: the client/version, protocol or requested feature is unknown or outside the matrix.

Only `supported` may proceed to write admission. `read_only` and `upgrade_required` reject the write before tenant lookup, authorization, persistence, outbox work or legacy mutation. A negotiation response is advisory for reads and never grants tenant or resource access; every subsequent read still uses canonical admission and authorization.

The response carries `policy_version`, `outcome`, the exact client minimum and maximum, service minimum and maximum, accepted protocol versions, selected features and current schema ID/version. It contains no tenant, membership, resource-existence or credential information. Clients must renegotiate after reconnect, service-version change, policy-version change or an `upgrade_required` response.

HTTP uses status `426 Upgrade Required` for incompatible writes and a structured body with `error: "upgrade_required"`, the closed minimum/maximum versions and `retryable: false`. Nostr ingress returns a terminal `CLOSED`/`OK` reason beginning `upgrade-required:`. The Buzz CLI shim preserves its structured stderr envelope and exit code `1`. These errors may name only client, protocol, feature and schema policy; they must not reveal whether a tenant or resource exists.

## Compatibility decisions

The following combinations are deliberately unsupported:

- any client, service, protocol or schema version outside an exact matrix range;
- any write when negotiation is missing, ambiguous, stale or returns `read_only`;
- a direct write from a frozen Buzz client that bypasses its declared adapter;
- any Buzz Postgres write after canonical cutover;
- a service starting below its schema minimum or above its schema maximum; and
- a rollback binary whose declared maximum is lower than the deployed schema.

Changing a boundary requires one new `policy_version`, updated closed ranges, endpoint/client tests for supported, read-only and upgrade-required outcomes, a schema-startup/rollback test and an updated migration rollback statement. Expanding a range based only on semantic-version similarity is prohibited.

## Version authorities

- Zed desktop and Collab service: `crates/zed/Cargo.toml` and `crates/collab/Cargo.toml`.
- Zed RPC: `crates/rpc/src/rpc.rs`.
- Frozen web, mobile, CLI and admin clients: `.agents/specs/collaborative-workspace/fixtures/clients/manifest.json`.
- Buzz desktop: `projects/buzz/desktop/package.json` and `projects/buzz/desktop/src-tauri/Cargo.toml`.
- Buzz relay and source schema: `projects/buzz/Cargo.toml` and its 30 migration baseline; import code admits schema `30` exactly.
- Canonical schemas: `crates/collab/migrations`; the push floor is also pinned by `deploy/collaboration/push-gateway/values.yaml`.
- CLI, pairing, ciphertext, push and audio protocols: `tools/buzz_compat`, `crates/nostr_compat` and `crates/collab/src/huddle/buzz_audio.rs`.
