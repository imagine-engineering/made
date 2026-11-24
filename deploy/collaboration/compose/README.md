# Collaboration Compose environment

This directory is the Zed-owned single-node development and self-hosting
topology. It starts Collab only after Postgres, Redis and the S3-compatible
object store are healthy and after the private object bucket exists. Named
volumes retain database, Redis, object and Git state independently of the
Collab image.

Collab and MinIO share one container network namespace so the canonical object
endpoint remains the explicitly permitted loopback URL
`http://127.0.0.1:9000`; object traffic from Collab never crosses the Compose
bridge in plaintext. The object-store service publishes Collab's port 8080 from
that shared namespace, and the private readiness probe reaches the same listener
as `http://object-store:8080/healthz`.

The Compose file supplies both the canonical `COLLABORATION_*` configuration
from Task 44.1 and the current Collab runtime names for the same dependencies.
The duplicate names are a temporary binding, not two configuration sources:
their equality is checked by `check.py`. The application image must contain
`/app/collab`, as produced by `Dockerfile-collab`.

## Local development

```sh
cd deploy/collaboration/compose
cp .env.example .env
# Replace every CHANGE_ME value, then use the repository Dockerfile.
COLLABORATION_BUILD_LOCAL=true ./run.sh config
COLLABORATION_BUILD_LOCAL=true ./run.sh start
```

`start` waits for container health and then runs a private, one-shot probe of
Collab's dependency-aware `/healthz` endpoint. The local override also publishes
Postgres, Redis and object-store ports for development tools. Do not enable it
on an Internet-facing host.

The checked-in Collab binary runs its embedded SQLx migrations before opening
the application database. A fresh local Postgres volume therefore acquires the
canonical collaboration message projections, operation receipts and outbox as
part of startup.

After the stack is healthy, launch two independently authenticated Rust-product
clients with the existing local seed users:

```sh
./run.sh clients /absolute/path/to/a/project
```

The launcher builds `zed` with `--no-default-features --features
multiplayer-tools,rust-tools`, sets `ZED_PRODUCT_ID=rust`, and reuses
`script/zed-local -2 --stateful`. It reads the local admin token from the chosen
Compose environment file without printing it. Both clients connect to the local
Collab RPC listener and retain separate application state directories.

## Self-hosting and rollback

Set `COLLABORATION_IMAGE` to a release image pinned by sha256 digest. Back up
`.env` and take one consistent Postgres/object/Git checkpoint before changing
the image. Do not run `docker compose down --volumes`; ordinary `./run.sh stop`
retains every canonical volume.

Keep the known-good digest in `COLLABORATION_PREVIOUS_IMAGE`. Before the schema
crosses that binary's compatibility ceiling, roll back with:

```sh
./run.sh rollback
```

Rollback rejects tags and accepts only `<repository>@sha256:<digest>`. It pulls
and recreates only the Collab container, preserves dependencies and all named
volumes, performs no down migration, and requires `/healthz` to pass. If the
probe fails, leave the new container stopped and follow the migration incident
procedure rather than deleting or rewriting persistent state.

## Validation

```sh
python3 deploy/collaboration/compose/check.py
bash -n deploy/collaboration/compose/run.sh
deploy/collaboration/compose/run.sh smoke
```

`check.py` renders both current- and prior-image configurations and verifies the
canonical settings, health dependency graph, volumes and rollback isolation.
`smoke` uses `compose.smoke.yaml` and `.env.smoke` to exercise actual Compose
health ordering in a separate project with disposable BusyBox processes. The
overlay is validation-only: it proves orchestration mechanics without claiming
to validate a release binary, production dependencies or schema migrations.

## Optional signed profile HTTP

The base file passes through `COLLABORATION_NOSTR_ROUTES`,
`COLLABORATION_NOSTR_API_CALLS_PER_MINUTE` and optional
`COLLABORATION_NOSTR_CLOSED_WEBSOCKET` only when explicitly present in the
selected environment. Leave both unset to retain disabled behavior; do not
supply empty strings. To enable the existing service, add both to `.env`:

```dotenv
COLLABORATION_NOSTR_ROUTES=[{"community_id":"11111111-1111-4111-8111-111111111111","public_url":"https://collaboration.example.com"}]
COLLABORATION_NOSTR_API_CALLS_PER_MINUTE=60
```

To opt into the existing closed-current-member NIP42 socket on the same root,
set `COLLABORATION_NOSTR_CLOSED_WEBSOCKET=true` alongside explicit routes/quota.
Unset remains disabled; explicit `false` remains false. The flag creates no
member/role authority, additional published port or general subscriptions.
Missing routes/quota/Redis still refuse canonical startup.

Use the real current community and serving host, with an external TLS listener
that preserves Host and routes the published Collab port. Explicit loopback
HTTP remains governed by the canonical service's local endpoint rules. Existing
Postgres and authenticated shared Redis bindings are retained. There is no
implicit quota: zero denies every genuine admitted API request; supplying only
one field or malformed routes causes the canonical service to refuse startup.
The existing `/events`, `/query`, `/info`, NIP05 and Accept-aware root flow is
opt-in; enabling it does not create membership or identity authority. `check.py`
validates enabled and disabled bindings without starting containers. Compose's
JSON configuration retains unresolved pass-through keys as `null`; these keys
are removed from the service container environment according to the
[Compose environment specification](https://docs.docker.com/reference/compose-file/services/#environment).
An explicit empty string is a present value and is never treated as disabled;
the service's existing parser rejects it. The fixture distinguishes null/missing,
empty, configured and incomplete bindings and preserves explicit quota `0`.
Actual container environment inspection and genuine service/SQL/Redis/deployment
verification remain separate evidence.
