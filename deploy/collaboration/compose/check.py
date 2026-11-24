#!/usr/bin/env python3

import json
import os
import pathlib
import subprocess


DIRECTORY = pathlib.Path(__file__).resolve().parent
CURRENT_IMAGE = "ghcr.io/zed-industries/collaboration@sha256:" + "1" * 64
PREVIOUS_IMAGE = "ghcr.io/zed-industries/collaboration@sha256:" + "2" * 64


def render(
    image: str,
    files: tuple[str, ...] = ("compose.yaml",),
    nostr_environment: dict[str, str] | None = None,
) -> dict:
    environment = os.environ.copy()
    environment.update(
        {
            "COLLABORATION_IMAGE": image,
            "COLLABORATION_PUBLIC_URL": "https://collaboration.example.test",
            "POSTGRES_PASSWORD": "postgres-canary",
            "REDIS_PASSWORD": "redis-canary",
            "COLLABORATION_OBJECT_ACCESS_KEY": "object-access-canary",
            "COLLABORATION_OBJECT_SECRET_KEY": "object-secret-canary",
            "ZED_CLOUD_INTERNAL_API_KEY": "internal-api-canary",
        }
    )
    for key in ("COLLABORATION_NOSTR_ROUTES", "COLLABORATION_NOSTR_API_CALLS_PER_MINUTE", "COLLABORATION_NOSTR_CLOSED_WEBSOCKET"):
        environment.pop(key, None)
    environment.update(nostr_environment or {})
    command = ["docker", "compose", "--env-file", os.devnull]
    for file in files:
        command.extend(("-f", file))
    command.extend(("--profile", "validation", "config", "--format", "json"))
    result = subprocess.run(
        command,
        cwd=DIRECTORY,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    document = render(CURRENT_IMAGE)
    services = document["services"]
    require(
        set(services)
        == {
            "collaboration",
            "collaboration-readiness",
            "postgres",
            "redis",
            "object-store",
            "object-store-init",
        },
        "unexpected service inventory",
    )

    collaboration = services["collaboration"]
    require(collaboration["image"] == CURRENT_IMAGE, "current image selection drifted")
    require(
        collaboration["depends_on"]
        == {
            "object-store": {"condition": "service_healthy", "required": True},
            "object-store-init": {
                "condition": "service_completed_successfully",
                "required": True,
            },
            "postgres": {"condition": "service_healthy", "required": True},
            "redis": {"condition": "service_healthy", "required": True},
        },
        "Collab dependency health ordering drifted",
    )
    environment = collaboration["environment"]
    # Compose config JSON retains unresolved pass-through keys as null; Compose removes these from the container environment, while explicit empty values remain strings.
    require(
        environment.get("COLLABORATION_NOSTR_ROUTES") is None
        and environment.get("COLLABORATION_NOSTR_API_CALLS_PER_MINUTE") is None
        and environment.get("COLLABORATION_NOSTR_CLOSED_WEBSOCKET") is None,
        "absent optional Nostr settings must stay absent, not empty strings",
    )
    routes = [{
        "community_id": "11111111-1111-4111-8111-111111111111",
        "public_url": "https://collaboration.example.test",
    }]
    for quota in ("60", "0"):
        settings = {
            "COLLABORATION_NOSTR_ROUTES": json.dumps(routes),
            "COLLABORATION_NOSTR_API_CALLS_PER_MINUTE": quota,
        }
        configured = render(CURRENT_IMAGE, nostr_environment=settings)
        configured_environment = configured["services"]["collaboration"]["environment"]
        require(json.loads(configured_environment["COLLABORATION_NOSTR_ROUTES"]) == routes,
                "explicit Nostr community routes drifted")
        require(configured_environment["COLLABORATION_NOSTR_API_CALLS_PER_MINUTE"] == quota,
                "explicit shared quota (including zero deny-all) drifted")
        require(configured_environment["COLLABORATION_REDIS_URL"] == environment["COLLABORATION_REDIS_URL"],
                "enabling Nostr must retain the canonical shared Redis endpoint")
        require(configured["services"]["collaboration"]["depends_on"] == collaboration["depends_on"],
                "Nostr must retain actual dependency ordering")
    routes_only = render(CURRENT_IMAGE, nostr_environment={"COLLABORATION_NOSTR_ROUTES": json.dumps(routes)})
    require(routes_only["services"]["collaboration"]["environment"].get("COLLABORATION_NOSTR_API_CALLS_PER_MINUTE") is None,
            "missing quota must remain absent for canonical startup rejection, not acquire a default")
    quota_only = render(CURRENT_IMAGE, nostr_environment={"COLLABORATION_NOSTR_API_CALLS_PER_MINUTE": "60"})
    require(quota_only["services"]["collaboration"]["environment"].get("COLLABORATION_NOSTR_ROUTES") is None,
            "missing routes must remain absent for canonical startup rejection")
    for closed_websocket in ("false", "true"):
        settings = {
            "COLLABORATION_NOSTR_ROUTES": json.dumps(routes),
            "COLLABORATION_NOSTR_API_CALLS_PER_MINUTE": "60",
            "COLLABORATION_NOSTR_CLOSED_WEBSOCKET": closed_websocket,
        }
        configured = render(CURRENT_IMAGE, nostr_environment=settings)
        configured_service = configured["services"]["collaboration"]
        configured_environment = configured_service["environment"]
        require(configured_environment.get("COLLABORATION_NOSTR_CLOSED_WEBSOCKET") == closed_websocket,
                "explicit closed WebSocket opt-in value lost")
        require(json.loads(configured_environment["COLLABORATION_NOSTR_ROUTES"]) == routes,
                "closed WebSocket changed trusted route bindings")
        require(configured_environment["COLLABORATION_NOSTR_API_CALLS_PER_MINUTE"] == "60",
                "closed WebSocket changed quota")
        require(configured_environment["COLLABORATION_REDIS_URL"] == environment["COLLABORATION_REDIS_URL"],
                "closed WebSocket changed canonical Redis")
        for field in ("network_mode", "depends_on", "volumes", "healthcheck"):
            require(configured_service[field] == collaboration[field],
                    f"closed WebSocket changed existing {field}")
        require(configured["services"]["object-store"]["ports"] == services["object-store"]["ports"],
                "closed WebSocket added public exposure")
    closed_only = render(CURRENT_IMAGE, nostr_environment={"COLLABORATION_NOSTR_CLOSED_WEBSOCKET": "true"})
    closed_environment = closed_only["services"]["collaboration"]["environment"]
    require(closed_environment.get("COLLABORATION_NOSTR_ROUTES") is None
            and closed_environment.get("COLLABORATION_NOSTR_API_CALLS_PER_MINUTE") is None,
            "closed WebSocket must not fabricate missing routes or quota")
    explicitly_empty = render(CURRENT_IMAGE, nostr_environment={
        "COLLABORATION_NOSTR_ROUTES": "",
        "COLLABORATION_NOSTR_API_CALLS_PER_MINUTE": "",
        "COLLABORATION_NOSTR_CLOSED_WEBSOCKET": "",
    })["services"]["collaboration"]["environment"]
    for optional in ("COLLABORATION_NOSTR_ROUTES", "COLLABORATION_NOSTR_API_CALLS_PER_MINUTE", "COLLABORATION_NOSTR_CLOSED_WEBSOCKET"):
        require(explicitly_empty.get(optional) == "" and explicitly_empty.get(optional) is not None,
                "explicit empty Nostr settings must not masquerade as unset values")
    for canonical in (
        "COLLABORATION_PUBLIC_URL",
        "COLLABORATION_DATABASE_URL",
        "COLLABORATION_REDIS_URL",
        "COLLABORATION_OBJECT_ENDPOINT",
        "COLLABORATION_OBJECT_REGION",
        "COLLABORATION_OBJECT_BUCKET",
        "COLLABORATION_OBJECT_ACCESS_KEY",
        "COLLABORATION_OBJECT_SECRET_KEY",
        "COLLABORATION_OBJECT_ADDRESSING_STYLE",
        "COLLABORATION_GIT_REPOSITORY_PATH",
        "COLLABORATION_REPLICA_COUNT",
        "COLLABORATION_PUSH_ENABLED",
        "COLLABORATION_PAIRING_ENABLED",
        "COLLABORATION_RELAY_MESH_ENABLED",
    ):
        require(canonical in environment, f"missing canonical setting {canonical}")
    require(
        environment["DATABASE_URL"] == environment["COLLABORATION_DATABASE_URL"],
        "runtime and canonical database bindings differ",
    )
    require(
        environment["BLOB_STORE_URL"] == environment["COLLABORATION_OBJECT_ENDPOINT"],
        "runtime and canonical object endpoints differ",
    )
    require(
        environment["COLLABORATION_OBJECT_ENDPOINT"] == "http://127.0.0.1:9000",
        "the local object endpoint left its admitted loopback boundary",
    )
    require(
        collaboration["network_mode"] == "service:object-store",
        "Collab no longer shares the object-store loopback namespace",
    )

    for service in ("collaboration", "postgres", "redis", "object-store"):
        require("healthcheck" in services[service], f"{service} has no healthcheck")
    require(
        services["collaboration-readiness"]["profiles"] == ["validation"],
        "readiness probe must remain private to explicit validation",
    )
    require(
        services["collaboration-readiness"]["depends_on"]["collaboration"]["condition"]
        == "service_healthy",
        "readiness probe must follow Collab liveness",
    )

    expected_volumes = {
        "collaboration-postgres",
        "collaboration-redis",
        "collaboration-objects",
        "collaboration-git",
    }
    require(set(document["volumes"]) == expected_volumes, "canonical volume inventory drifted")

    rollback = render(PREVIOUS_IMAGE)
    require(
        rollback["services"]["collaboration"]["image"] == PREVIOUS_IMAGE,
        "prior immutable image did not replace the current image",
    )
    for service in expected_volumes:
        require(
            rollback["volumes"][service] == document["volumes"][service],
            f"rollback changed volume {service}",
        )

    smoke = render(CURRENT_IMAGE, ("compose.yaml", "compose.smoke.yaml"))
    require(
        smoke["services"]["collaboration"]["image"] == "busybox:1.37.0-musl",
        "smoke overlay can invoke a release image",
    )
    require(
        smoke["services"]["object-store"].get("ports") is None,
        "isolated smoke publishes a host port",
    )
    require(
        smoke["services"]["collaboration-readiness"]["command"][-1]
        == "http://object-store:8080/healthz",
        "smoke readiness does not use the shared namespace",
    )

    script = (DIRECTORY / "run.sh").read_text()
    require("@sha256:[0-9a-f]{64}" in script, "rollback digest gate is missing")
    require("--no-deps collaboration" in script, "rollback could recreate dependencies")
    require("verify_readiness" in script, "rollback does not verify readiness")
    print("collaboration Compose contract checks passed")


if __name__ == "__main__":
    main()
