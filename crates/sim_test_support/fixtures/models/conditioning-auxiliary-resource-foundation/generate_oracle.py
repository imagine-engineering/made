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
    "projects/sim/ComfyUI/nodes.py": "b8dfdde1de8975be762b085048143cc2dda8fc9202695e460ecc2c8dfe44bc4b",
    "projects/sim/ComfyUI/sim/sd.py": "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
    "projects/sim/ComfyUI/sim/ops.py": "9d8a4ec8357a9bfcd98dddbf06fcc2a0244643a392aacbe0970d945462c86a42",
    "projects/sim/ComfyUI/sim/t2i_adapter/adapter.py": "efc52cc85f941e11b509c0339e8950a6680d031b09bec36736d8834b9ccfa1af",
    "projects/sim/ComfyUI/sim/ldm/flux/redux.py": "3b7abf43e15fc7b9613a64e701635f943351fa47f5036d87427ea672f93c7952",
    "projects/sim/ComfyUI/sim_extras/nodes_photomaker.py": "c7e86d4684d2884eda20250bd7122e31a8aaf7bfa003245ad0e0431a9ef1957e",
    "projects/sim/ComfyUI/sim/clip_model.py": "08be993d86c3b494b58305fb868638b4b525bbe40abead89e9c94da021716845",
    "projects/sim/ComfyUI/sim/clip_vision.py": "8e3cc5d5d257b52d120885ba7427ff3fdf56129a485fdadac3b6215ae2c67b20",
    "projects/sim/ComfyUI/sim/gligen.py": "87c4297809a1a0a7727e3623cef0930463080dfedc068eb97c5e19bc6e155d0c",
    "projects/sim/ComfyUI/sim/ldm/modules/attention.py": "436e1d91f8d5d84c5667e051cdf3ab2f91d8db25b66d88a084c89a202de0579e",
    "projects/sim/ComfyUI/sim/ldm/modules/diffusionmodules/openaimodel.py": "9d27fb036cab8a262ef3d866a643f7fdc40994022616f1b8be14b7d919f57f96",
    "projects/sim/ComfyUI/sim/samplers.py": "d882256ae9baa1d23f1367ab2ec3b021fdc15fe39ce4cb49ea2c1ee10026a649",
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
    generator_sha256 = sha256(DIRECTORY / "generate_oracle.py")
    source_graph_sha256 = sha256(DIRECTORY / "source_graph.py")
    oracle.update(
        pinned_sources=SOURCES,
        generator_sha256=generator_sha256,
        source_graph_sha256=source_graph_sha256,
        generator_command=(
            "PYTHONDONTWRITEBYTECODE=1 python3 "
            "crates/sim_test_support/fixtures/models/conditioning-auxiliary-resource-foundation/"
            "generate_oracle.py --check"
        ),
    )
    oracle_bytes = encoded(oracle)
    oracle_sha256 = hashlib.sha256(oracle_bytes).hexdigest()
    manifest = {
        "format": "conditioning-auxiliary-resource-foundation-manifest-v1",
        "oracle_domain": oracle["format"],
        "oracle_sha256": oracle_sha256,
        "generator_sha256": generator_sha256,
        "source_graph_sha256": source_graph_sha256,
        "profiles": ["style", "redux", "photomaker", "gligen"],
        "storage_dtypes": ["float32", "float16", "bfloat16"],
        "style_state_count": 42,
        "redux_state_count": 4,
        "photomaker_state_count": 407,
        "gligen_state_count": 42,
        "mutations": sorted([*oracle["mutations"], *oracle["photomaker_mutations"]]),
        "reduced_profiles_are_source_exact": False,
    }
    provenance = {
        **manifest,
        "format": "conditioning-auxiliary-resource-foundation-provenance-v1",
        "command": oracle["generator_command"],
        "arithmetic_contract": (
            "Pure Python standard-library source equations with explicit IEEE-754 binary32 "
            "rounding, exact-rational fused multiply-add with ties-to-even binary32 rounding, "
            "source-order f32 attention accumulation, f64 layer-normalization statistics, and "
            "f64 final matmul accumulation."
        ),
        "runtime_boundary": (
            "The fixture executes reduced dimensions only and never labels them source-exact "
            "production profiles. It does not import torch, numpy, a model runtime, a codec, or "
            "a native library and uses no subprocess, network, host credential, or generated "
            "product output. Q/K baseline projections are exactly zero, making the source "
            "softmax an exact uniform four-token disposition; V, output projection, MLP, "
            "QuickGELU, final projection, and both Redux linears remain live. A separate batch-two "
            "case activates asymmetric Q and K projections in both heads of every layer to "
            "discriminate fused splitting, scale, softmax, head grouping, and batch/sequence order. "
            "The PhotoMaker profile retains the exact 407-key source order while reducing tensor "
            "dimensions, consumes the canonical CLIP pooled post-layer-normalization state, and "
            "executes both projection branches, noncontiguous mask pairing, both residual MLPs, "
            "final normalization, and row-major scatter. Source torch GELU is evaluated with the "
            "standard-library erf equation; the fixture separately evaluates the canonical Rust "
            "f32 erf approximation over every admitted reduced value and records the exact maximum "
            "downstream output ULP distance plus a max-plus-one rejection discriminator. The "
            "GLIGEN case independently projects the exact PositionNet Fourier/null/MLP equations, "
            "sparse structural-to-compact fuser order, canonical self-attention, GEGLU, gates, and "
            "residuals for every admitted storage dtype; a second schema case locks the 768-key "
            "eight-head rule without importing product code."
        ),
        "pinned_sources": SOURCES,
    }
    return {
        "manifest.json": encoded(manifest),
        "oracle.json": oracle_bytes,
        "provenance.json": encoded(provenance),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    failures = []
    for name, content in sorted(documents().items()):
        path = DIRECTORY / name
        if arguments.check:
            if not path.exists() or path.read_bytes() != content:
                failures.append(name)
        else:
            path.write_bytes(content)
    if failures:
        print("stale conditioning auxiliary fixture: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
