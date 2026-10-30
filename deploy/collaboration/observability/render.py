#!/usr/bin/env python3

import argparse
import json
import pathlib
import re
import sys


DIRECTORY = pathlib.Path(__file__).resolve().parent
REPOSITORY = DIRECTORY.parents[2]
LIMITS = REPOSITORY / ".agents/specs/collaborative-workspace/security/operational-limits.md"
DASHBOARD_DIRECTORY = DIRECTORY / "dashboards"
RULES = DIRECTORY / "prometheus-rules.yaml"
DASHBOARD_BY_PREFIX = {
    "CON": "admission-realtime",
    "DAT": "durable-collaboration",
    "PUS": "durable-collaboration",
    "EXE": "execution-media",
    "MED": "execution-media",
    "HUD": "execution-media",
    "MSH": "execution-media",
    "OPS": "migration-release",
}
TITLES = {
    "admission-realtime": "Collaboration admission and realtime",
    "durable-collaboration": "Durable collaboration",
    "execution-media": "Execution and media",
    "migration-release": "Migration and release",
}
METRIC_PATTERN = re.compile(r"\b(?:agent|client|collab|push_gateway|remote)_[a-z0-9_]+\b")
FALLBACK_METRICS = {"OL-OPS-08": "client_telemetry_requests_total"}


def load_limits() -> dict[str, list[tuple[str, str]]]:
    dashboards = {name: [] for name in TITLES}
    for line in LIMITS.read_text().splitlines():
        if not line.startswith("| OL-"):
            continue
        columns = line.split("|")
        limit_id = columns[1].strip()
        prefix = limit_id.split("-")[1]
        dashboard = DASHBOARD_BY_PREFIX[prefix]
        metrics = sorted(set(METRIC_PATTERN.findall(columns[4])))
        if not metrics and limit_id in FALLBACK_METRICS:
            metrics = [FALLBACK_METRICS[limit_id]]
        for metric in metrics:
            dashboards[dashboard].append((limit_id, metric))
    return dashboards


def load_signals() -> list[dict]:
    return json.loads((DIRECTORY / "stop-signals.json").read_text())["signals"]


def dashboard_document(name: str, metrics: list[tuple[str, str]], signals: list[dict]) -> dict:
    panels = []
    seen = set()
    for limit_id, metric in metrics:
        if metric in seen:
            continue
        seen.add(metric)
        panels.append((f"{limit_id} · {metric}", metric))
    for signal in signals:
        metric = signal["metric"]
        if signal["dashboard"] == name and metric not in seen:
            seen.add(metric)
            panels.append((f"{signal['id']} · {metric}", metric))

    rendered_panels = []
    for index, (title, expression) in enumerate(panels, start=1):
        rendered_panels.append(
            {
                "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                "fieldConfig": {"defaults": {}, "overrides": []},
                "gridPos": {
                    "h": 8,
                    "w": 12,
                    "x": 0 if index % 2 else 12,
                    "y": ((index - 1) // 2) * 8,
                },
                "id": index,
                "options": {"legend": {"displayMode": "table", "placement": "bottom"}},
                "targets": [{"expr": expression, "refId": "A"}],
                "title": title,
                "type": "timeseries",
            }
        )
    return {
        "annotations": {"list": []},
        "editable": False,
        "graphTooltip": 1,
        "panels": rendered_panels,
        "refresh": "30s",
        "schemaVersion": 39,
        "tags": ["collaboration", "access-controlled", "content-free"],
        "templating": {"list": []},
        "time": {"from": "now-6h", "to": "now"},
        "title": TITLES[name],
        "uid": f"zed-collaboration-{name}",
        "version": 1,
    }


def rules_document(signals: list[dict]) -> str:
    lines = [
        "groups:",
        "  - name: zed-collaboration-stop-signals",
        "    interval: 30s",
        "    rules:",
    ]
    for signal in signals:
        alert = "Collaboration" + "".join(word.title() for word in signal["id"].split("-")[1:])
        lines.extend(
            [
                f"      - alert: {alert}",
                f"        expr: {json.dumps(signal['expression'])}",
                f"        for: {signal['duration']}",
                "        labels:",
                "          severity: page",
                f"          stop_signal: {signal['id']}",
                "        annotations:",
                f"          action: {signal['action']}",
                f"          summary: {json.dumps('Automatic collaboration stop signal ' + signal['id'])}",
            ]
        )
    return "\n".join(lines) + "\n"


def outputs() -> dict[pathlib.Path, str]:
    signals = load_signals()
    documents = {
        DASHBOARD_DIRECTORY / f"{name}.json": json.dumps(
            dashboard_document(name, metrics, signals), indent=2, sort_keys=True
        )
        + "\n"
        for name, metrics in load_limits().items()
    }
    documents[RULES] = rules_document(signals)
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    failures = []
    for path, content in outputs().items():
        if arguments.check:
            if not path.exists() or path.read_text() != content:
                failures.append(str(path.relative_to(REPOSITORY)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if failures:
        print("observability render drift: " + ", ".join(failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
