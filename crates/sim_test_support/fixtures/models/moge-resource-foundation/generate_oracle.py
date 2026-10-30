#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

from source_graph import build_oracle

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[4]
SOURCES = {
    "projects/sim/ComfyUI/sim_extras/nodes_moge.py": "160f48e4b6bb1e34617f9de78380758ef5d04caa7c8ea7768ce31b98fccee265",
    "projects/sim/ComfyUI/sim/ldm/moge/model.py": "68ee3db2ff7eb96c8a90234b182559129c5c094374128e5ba99baed7caf0cb3c",
    "projects/sim/ComfyUI/sim/ldm/moge/modules.py": "3655abdce2de058624bd4ea2f02757ab42bd811d9afab0fef824fe480afdb2a6",
    "projects/sim/ComfyUI/sim/ldm/moge/geometry.py": "db8e2da75f13028a98067c517d6495fd9818b2878b44599beaa281ff7fde397c",
    "projects/sim/ComfyUI/sim/image_encoders/dino2.py": "1dec8c1d6104c268e593cea20302d925f637266edce2a6e4dfa142af8a00d579",
    "projects/sim/ComfyUI/sim/text_encoders/bert.py": "3f1f32353da95790285a10f452959a871aa949aab15a89b646a95abc6165955c",
    "projects/sim/ComfyUI/sim/ldm/modules/attention.py": "436e1d91f8d5d84c5667e051cdf3ab2f91d8db25b66d88a084c89a202de0579e",
    "projects/sim/ComfyUI/sim/ops.py": "9d8a4ec8357a9bfcd98dddbf06fcc2a0244643a392aacbe0970d945462c86a42",
    "projects/sim/ComfyUI/sim/model_management.py": "c2ca243c80a5262ecafe19feb15cec22d4003c16e523b5376f543f0f75acabaa",
    "projects/sim/ComfyUI/sim/model_patcher.py": "96d21eeaf16d4723355374a5e2b93b35d512ad8f6dc8a1a3a4253cdd71dfd5b0",
    "projects/sim/ComfyUI/sim/utils.py": "8b8805ca837e20c922a846854156d10e214654f69df96be90969522f9def2bdb",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def documents():
    for source, expected in SOURCES.items():
        actual = sha256(ROOT / source)
        if actual != expected:
            raise RuntimeError(f"pinned source changed: {source}: {actual}")
    oracle = build_oracle()
    source_graph_sha256 = sha256(DIRECTORY / "source_graph.py")
    generator_sha256 = sha256(DIRECTORY / "generate_oracle.py")
    dino_oracle_sha256 = sha256(
        ROOT / "crates/sim_test_support/fixtures/models/dinov2-backbone-owner-foundation/oracle.json"
    )
    oracle.update(
        pinned_sources=SOURCES,
        source_graph_sha256=source_graph_sha256,
        generator_sha256=generator_sha256,
        dino_oracle_sha256=dino_oracle_sha256,
        generator_command="PYTHONDONTWRITEBYTECODE=1 python3 crates/sim_test_support/fixtures/models/moge-resource-foundation/generate_oracle.py --check",
    )
    oracle_bytes = encoded(oracle)
    oracle_sha256 = hashlib.sha256(oracle_bytes).hexdigest()
    manifest = {
        "format": "moge-resource-foundation-manifest-v1",
        "oracle_domain": oracle["format"],
        "oracle_sha256": oracle_sha256,
        "generator_sha256": generator_sha256,
        "source_graph_sha256": source_graph_sha256,
        "dino_oracle_sha256": dino_oracle_sha256,
        "profiles": ["v1", "v2"],
        "mutations": sorted(oracle["mutations"]),
    }
    provenance = {
        **manifest,
        "format": "moge-resource-foundation-provenance-v1",
        "command": oracle["generator_command"],
        "arithmetic_contract": "IEEE-754 binary32 boundaries with source-derived transcendental ULP limits",
        "runtime_boundary": "Model architecture and downstream geometry equations remain an independently derived binary32 foundation, with unchanged recorded ULP bounds and no learned-model certification. Camera focal/shift golds execute the three unchanged tagged geometry functions with CPU Torch 2.8.0, NumPy 1.26.4 and SciPy 1.16.2, including original interpolation, mask handling, library-default MINPACK and final scalar promotion. Solver status/nfev receipts are genuine SciPy observations, not invented accepted/rejected iteration counts. Native numerical results do not supply gold values. No network, credential or native-model subprocess is used.",
        "pinned_sources": SOURCES,
    }
    return {
        "oracle.json": oracle_bytes,
        "manifest.json": encoded(manifest),
        "provenance.json": encoded(provenance),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    failures = []
    for name, content in documents().items():
        path = DIRECTORY / name
        if arguments.check:
            if not path.exists() or path.read_bytes() != content:
                failures.append(name)
        else:
            path.write_bytes(content)
    if failures:
        print("stale MoGe fixture: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
