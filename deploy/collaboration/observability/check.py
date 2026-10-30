#!/usr/bin/env python3

import json
import pathlib
import re
import subprocess
import sys


DIRECTORY = pathlib.Path(__file__).resolve().parent
REPOSITORY = DIRECTORY.parents[2]
DASHBOARDS = DIRECTORY / "dashboards"
LIMITS = REPOSITORY / ".agents/specs/collaborative-workspace/security/operational-limits.md"
PRIVATE_FIXTURE = REPOSITORY / ".agents/specs/collaborative-workspace/fixtures/protocol/events.json"
REQUIRED_SIGNALS = {
    "STOP-CROSS-TENANT",
    "STOP-REPLAY-OR-STALE",
    "STOP-DUPLICATE-EXECUTION",
    "STOP-SHADOW-DIVERGENCE",
    "STOP-SECRET-CANARY",
    "STOP-CONTENT-CANARY",
    "STOP-READY-AUTHORITY",
    "STOP-READY-SCHEMA",
    "STOP-READY-SECURITY",
    "STOP-QUEUE-BOUNDARY",
    "STOP-CLAIM-BOUNDARY",
    "STOP-LEASE-BOUNDARY",
    "STOP-CLIENT-TELEMETRY",
    "STOP-ADMISSION-AFTER-DISABLE",
    "STOP-KILL-SWITCH-ACK",
    "STOP-ROLLBACK-OWNER-CONFLICT",
}
REQUIRED_METRICS = {
    "collab_readiness",
    "collab_outbox_oldest_seconds",
    "collab_projection_drift_total",
    "collab_replica_freshness_seconds",
    "collab_compatibility_peers",
    "collab_migration_checkpoint_age_seconds",
    "collab_log_redactions_total",
}
ALLOWED_ACTION_WORDS = {
    "disable",
    "execution",
    "halt",
    "migration",
    "release",
    "reconcile",
    "remove",
    "rollback",
    "scoped",
    "state",
    "preserve",
    "tenant",
    "instances",
    "deployment",
    "and",
    "ready",
    "cutover",
    "admissions",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def private_contents() -> set[str]:
    events = json.loads(PRIVATE_FIXTURE.read_text())["events"]
    return {
        event["content"]
        for event in events.values()
        if isinstance(event.get("content"), str) and len(event["content"]) >= 8
    }


def verify_artifacts(dashboards: str, artifacts: str, signals: list[dict], rules: str) -> None:
    ids = {signal["id"] for signal in signals}
    require(ids == REQUIRED_SIGNALS, "automatic stop-signal inventory drifted")
    require(len(ids) == len(signals), "duplicate stop-signal ID")
    for signal in signals:
        require(signal["metric"] in dashboards, f"{signal['id']} missing from dashboards")
        require(signal["id"] in rules, f"{signal['id']} missing from Prometheus rules")
        require(json.dumps(signal["expression"]) in rules, f"{signal['id']} expression drifted")
        action_words = set(signal["action"].split("_"))
        require(action_words <= ALLOWED_ACTION_WORDS, f"{signal['id']} has an open action")
    for canary in private_contents():
        require(canary not in artifacts, "private protocol fixture content reached observability")


def main() -> None:
    subprocess.run([sys.executable, str(DIRECTORY / "render.py"), "--check"], check=True)
    dashboard_paths = sorted(DASHBOARDS.glob("*.json"))
    require(len(dashboard_paths) == 4, "exactly four dashboards are required")
    dashboards = [json.loads(path.read_text()) for path in dashboard_paths]
    require(
        all(
            dashboard["editable"] is False
            and "access-controlled" in dashboard["tags"]
            and "content-free" in dashboard["tags"]
            and dashboard["refresh"] == "30s"
            for dashboard in dashboards
        ),
        "dashboard access, privacy, or scrape cadence drifted",
    )
    dashboard_text = "\n".join(path.read_text() for path in dashboard_paths)
    artifact_text = "\n".join(
        path.read_text()
        for path in sorted(DIRECTORY.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    )
    require(
        REQUIRED_METRICS <= set(re.findall(r"[a-z][a-z0-9_]+", dashboard_text)),
        "required metric missing",
    )
    limit_ids = set(re.findall(r"\| (OL-[A-Z]+-[0-9]+) \|", LIMITS.read_text()))
    require(
        limit_ids <= set(re.findall(r"OL-[A-Z]+-[0-9]+", dashboard_text)),
        "limit missing from dashboards",
    )

    signals = json.loads((DIRECTORY / "stop-signals.json").read_text())["signals"]
    rules = (DIRECTORY / "prometheus-rules.yaml").read_text()
    verify_artifacts(dashboard_text, artifact_text, signals, rules)

    logging = json.loads((DIRECTORY / "logging-policy.json").read_text())
    require(logging["metrics_listener"]["scope"] == "private", "metrics listener became public")
    require(logging["metrics_listener"]["scrape_interval_seconds"] == 30, "scrape interval drifted")
    require(logging["metrics_listener"]["scrape_timeout_seconds"] == 15, "scrape timeout drifted")
    require(logging["logs"]["maximum_record_bytes"] == 65_536, "log byte ceiling drifted")
    require(logging["logs"]["unknown_fields"] == "drop", "unknown log fields are retained")
    require(
        not set(logging["logs"]["allowed_fields"]) & set(logging["logs"]["forbidden_fields"]),
        "a forbidden log field is allowed",
    )
    require(logging["client_telemetry"]["default"] == "disabled", "client telemetry default drifted")
    require(logging["client_telemetry"]["disabled_request_count"] == 0, "disabled telemetry sends")

    try:
        verify_artifacts(dashboard_text, artifact_text, signals[:-1], rules)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing-signal checker bite was accepted")
    fixture_canary = sorted(private_contents(), key=len, reverse=True)[0]
    try:
        verify_artifacts(dashboard_text, artifact_text + fixture_canary, signals, rules)
    except AssertionError:
        pass
    else:
        raise AssertionError("private-content checker bite was accepted")
    print(
        f"observability checks passed: dashboards={len(dashboards)} "
        f"limits={len(limit_ids)} stop_signals={len(signals)}"
    )


if __name__ == "__main__":
    main()
