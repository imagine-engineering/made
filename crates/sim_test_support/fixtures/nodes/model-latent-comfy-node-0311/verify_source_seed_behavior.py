#!/usr/bin/env python3
"""Recheck the pinned LatentBatchSeedBehavior method without importing ComfyUI."""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "projects/sim/ComfyUI/sim_extras/nodes_latent.py"
FIXTURE = Path(__file__).with_name("seed_behavior.json")
PINNED_SHA256 = "cda1a84ae9e0f33db7395b05e813dc3ec592fbe61ab1af44dda72af1486ff961"


def expected_cases():
    source = SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != PINNED_SHA256:
        raise RuntimeError("Pinned nodes_latent.py source changed")
    tree = ast.parse(source, filename=str(SOURCE))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LatentBatchSeedBehavior")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"io": SimpleNamespace(NodeOutput=lambda value: value)}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    cases = []
    for mode, indices in [("random", [9, 10, 11]), ("fixed", [9, 10, 11]), ("fixed", None), ("random", None)]:
        samples = {"samples": SimpleNamespace(shape=(3, 4, 8, 8)), "extra": "retained"}
        if indices is not None:
            samples["batch_index"] = list(indices)
        result = namespace["execute"](None, samples, mode)
        cases.append({
            "seed_behavior": mode,
            "input_batch_index": indices,
            "expected_batch_index": result.get("batch_index"),
            "retained_extra": result["extra"],
            "input_unchanged": samples.get("batch_index") == indices,
        })
    return {"source_sha256": PINNED_SHA256, "class_type": cls.name, "shape": [3, 4, 8, 8], "cases": cases}


if __name__ == "__main__":
    expected = expected_cases()
    if sys.argv[1:] == ["--write"]:
        FIXTURE.write_text(json.dumps(expected, indent=2) + "\n")
    elif json.loads(FIXTURE.read_text()) != expected:
        raise SystemExit("Seed-behavior source fixture is stale")
    print(f"Verified {len(expected['cases'])} pinned source cases")
