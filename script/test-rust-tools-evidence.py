"""Validate operator-reviewed physical records; never turn CLI smoke into certification."""

import argparse
import hashlib
import subprocess
import datetime
import json
from pathlib import Path
import re
import sys


CELLS = {"local-macos", "local-linux", "local-windows", "ssh-linux", "wsl",
         "dev-container", "multiplayer"}
REQUIRED = {"local-macos", "local-linux", "local-windows", "ssh-linux", "multiplayer"}
CHECKS = {"discovery", "run", "cancel", "reconnect_stale_rejection", "debug",
          "terminal_reveal", "source_navigation", "host_only_execution",
          "no_local_fallback", "capability_mismatch"}
MAX_BYTES = 256 * 1024
MAX_AGE_DAYS = 30


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded_text(value, name):
    require(isinstance(value, str) and 0 < len(value.strip()) <= 2048,
            f"{name} must be nonempty bounded text")
    require(not any(character in value for character in "\n\r\t\x00"),
            f"{name} must not contain control separators")
    return value


def validate(document, revision, worktree_diff_sha256, today=None):
    today = today or datetime.date.today()
    require(isinstance(document, dict), "evidence must be an object")
    require(type(document.get("schema_version")) is int and document["schema_version"] == 1, "unsupported evidence schema")
    require(document.get("policy") == "rust-tools-physical-v1", "unknown required-cell policy")
    require(document.get("revision") == revision and re.fullmatch(r"[0-9a-f]{40}", revision),
            "evidence must match the tested checkout revision")
    require(document.get("worktree_diff_sha256") == worktree_diff_sha256,
            "evidence does not cover the current uncommitted implementation changes")
    records = document.get("records")
    require(isinstance(records, list) and len(records) <= len(CELLS), "invalid cell records")
    results = {}
    for record in records:
        require(isinstance(record, dict), "cell record must be an object")
        cell = record.get("cell")
        require(isinstance(cell, str) and cell in CELLS and cell not in results,
                "unknown or duplicate cell")
        for key in ("tester", "reviewer", "client_os", "host_os", "architecture",
                    "rust_version", "dap_version", "zed_build", "observation"):
            bounded_text(record.get(key), key)
        require(record.get("review_approved") is True, "record has no explicit review approval")
        try:
            recorded = datetime.date.fromisoformat(record.get("date", ""))
        except (TypeError, ValueError) as error:
            raise ValueError("invalid record date") from error
        require(0 <= (today - recorded).days <= MAX_AGE_DAYS, "record is stale or future-dated")
        if cell == "ssh-linux":
            require(record["host_os"].lower().startswith("linux"), "SSH certification requires Linux")
        if cell.startswith("local-"):
            require(all(record[key].lower().startswith(cell.removeprefix("local-"))
                        for key in ("client_os", "host_os")),
                    "local cell OS does not match")
        capability = record.get("capability")
        require(capability in ("supported", "unsupported"), "missing capability result")
        if capability == "unsupported":
            bounded_text(record.get("capability_reason"), "capability reason")
            results[cell] = ("unsupported", "reviewed unsupported capability; see supplied record")
            continue
        outcomes = record.get("checks")
        require(isinstance(outcomes, dict) and set(outcomes) == CHECKS,
                "record must contain every physical checklist item")
        for name, outcome in outcomes.items():
            require(isinstance(outcome, dict), "check outcome must be an object")
            status = outcome.get("status")
            require(status in ("pass", "unsupported"), "physical check failed or is incomplete")
            bounded_text(outcome.get("observation"), f"{name} observation")
            require(status != "unsupported" or name == "debug",
                    "required transport checks cannot be waived as unsupported")
            if status == "unsupported":
                bounded_text(outcome.get("capability_reason"), "debug capability reason")
        if cell == "multiplayer":
            for name in ("read_only_denied", "peer_scoped_cancel"):
                require(record.get(name) is True, f"multiplayer {name} was not verified")
        results[cell] = ("certified", f"reviewed physical record dated {recorded.isoformat()}")
    certified = {cell for cell, (status, _) in results.items() if status == "certified"}
    require(REQUIRED <= certified, "required physical cells remain uncertified")
    require(bool({"wsl", "dev-container"} & certified),
            "an available WSL or development-container cell remains uncertified")
    return results


def checkout_identity():
    root = Path(__file__).resolve().parent.parent
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    # Spec/evidence edits must not invalidate the implementation they describe.
    paths = [".", ":!.agents", ":!docs"]
    changes = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", *paths], cwd=root)
    digest = hashlib.sha256(changes)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", *paths], cwd=root)
    for name in sorted(filter(None, untracked.split(b"\0"))):
        digest.update(name + b"\0")
        with (root / name.decode()).open("rb") as source:
            for chunk in iter(lambda: source.read(65536), b""):
                digest.update(chunk)
    return revision, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, nargs="?")
    parser.add_argument("--revision")
    parser.add_argument("--describe-checkout", action="store_true")
    arguments = parser.parse_args()
    try:
        revision, worktree_digest = checkout_identity()
        if arguments.describe_checkout:
            print(json.dumps({"schema_version": 1, "policy": "rust-tools-physical-v1",
                              "revision": revision, "worktree_diff_sha256": worktree_digest}))
            return 0
        require(arguments.evidence is not None and arguments.revision == revision,
                "provide evidence and the current checkout revision")
        with arguments.evidence.open("rb") as source:
            data = source.read(MAX_BYTES + 1)
        require(len(data) <= MAX_BYTES, "evidence exceeds the byte limit")
        records = validate(json.loads(data), revision, worktree_digest)
    except (OSError, UnicodeError, ValueError, RecursionError, subprocess.CalledProcessError) as error:
        print(f"physical evidence rejected: {error}", file=sys.stderr)
        return 1
    for cell, (status, detail) in sorted(records.items()):
        print(f"physical-{cell}\t{status}\t{detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
