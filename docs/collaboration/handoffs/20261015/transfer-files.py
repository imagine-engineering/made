#!/usr/bin/env python3
"""List or verify explicitly inventoried handoff files without copying payloads."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=["summary", "paths", "verify"])
parser.add_argument("--repository", type=Path)
parser.add_argument("--include-reviewed-references", action="store_true")
arguments = parser.parse_args()
base = Path(__file__).resolve().parent
summary = json.loads((base / "transfer-summary.json").read_text())
source_root = Path(summary["root_snapshot"])
destination_root = arguments.repository or source_root
rows = []
for manifest in summary["manifests"]:
    compressed = (base / manifest["path"]).read_bytes()
    if hashlib.sha256(compressed).hexdigest() != manifest["sha256"]:
        raise SystemExit("Compressed manifest checksum mismatch: " + manifest["path"])
    uncompressed = gzip.decompress(compressed)
    if hashlib.sha256(uncompressed).hexdigest() != manifest["uncompressed_sha256"]:
        raise SystemExit("Manifest content checksum mismatch: " + manifest["path"])
    document = json.loads(uncompressed)
    for row in document.get("files", []):
        selected = row.get("direct_copy_required", False)
        if arguments.include_reviewed_references and row.get("transport") == "review-before-transfer-reference-source":
            selected = True
        if selected:
            rows.append(row)
if arguments.action == "summary":
    print(json.dumps({"manifests": summary["manifests"], "selected_copy_files": len(rows), "selected_bytes": sum(row.get("bytes", 0) for row in rows), "reference_sources_require_review": not arguments.include_reviewed_references}, indent=2))
    raise SystemExit(0)
failures = []
seen = set()
for row in rows:
    original = Path(row["path"])
    try:
        relative = original.relative_to(source_root)
    except ValueError:
        failures.append({"path": str(original), "reason": "External original must be restored from its retained archive"})
        continue
    if relative in seen:
        continue
    seen.add(relative)
    if arguments.action == "paths":
        print(relative.as_posix())
        continue
    path = destination_root / relative
    if row.get("type") == "symlink":
        if not path.is_symlink():
            failures.append({"path": str(path), "reason": "Missing symlink"})
            continue
        payload = str(path.readlink()).encode()
        digest = hashlib.sha256(payload).hexdigest()
        size = len(payload)
    elif path.is_file():
        digest_object = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest_object.update(chunk)
        digest = digest_object.hexdigest()
        size = path.stat().st_size
    else:
        failures.append({"path": str(path), "reason": "Missing regular file"})
        continue
    if size != row["bytes"] or digest != row["sha256"]:
        failures.append({"path": str(path), "reason": "Size or checksum differs"})
if arguments.action == "verify":
    print(json.dumps({"verified_file_count": len(seen) - len(failures), "failures": failures, "repository": str(destination_root)}, indent=2))
if failures:
    print("Transfer verification requires attention; originals were not changed.", file=sys.stderr)
    raise SystemExit(1)
