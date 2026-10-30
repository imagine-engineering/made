#!/usr/bin/env python3
"""Recheck the pinned LatentCut shape and boundary equations without importing ComfyUI."""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "projects/sim/ComfyUI/sim_extras/nodes_latent.py"
FIXTURE = Path(__file__).with_name("cut.json")
PINNED_SHA256 = "cda1a84ae9e0f33db7395b05e813dc3ec592fbe61ab1af44dda72af1486ff961"


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(shape)


def narrow(tensor, dimension, start, amount):
    if start < 0:
        start += tensor.shape[dimension]
    if start < 0 or amount < 0 or start + amount > tensor.shape[dimension]:
        raise ValueError("invalid source narrow")
    shape = list(tensor.shape)
    shape[dimension] = amount
    return FakeTensor(shape)


def expected_cases():
    source = SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != PINNED_SHA256:
        raise RuntimeError("Pinned nodes_latent.py source changed")
    tree = ast.parse(source, filename=str(SOURCE))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LatentCut")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    method.decorator_list = []
    namespace = {
        "io": SimpleNamespace(NodeOutput=lambda value: value),
        "torch": SimpleNamespace(narrow=narrow),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(SOURCE), "exec"), namespace)
    cases = []
    for dim, index, amount in [("x", 99, 4), ("y", -2, 4), ("t", -99, 2), ("x", 1, 3)]:
        samples = {"samples": FakeTensor([2, 4, 3, 5, 7]), "extra": "retained"}
        result = namespace["execute"](None, samples, dim, index, amount)
        cases.append({
            "dim": dim, "index": index, "amount": amount,
            "expected_shape": list(result["samples"].shape),
            "retained_extra": result["extra"],
            "input_shape": list(samples["samples"].shape),
        })
    return {"source_sha256": PINNED_SHA256, "class_type": cls.name, "cases": cases}


if __name__ == "__main__":
    expected = expected_cases()
    if sys.argv[1:] == ["--write"]:
        FIXTURE.write_text(json.dumps(expected, indent=2) + "\n")
    elif json.loads(FIXTURE.read_text()) != expected:
        raise SystemExit("LatentCut source fixture is stale")
    print(f"Verified {len(expected['cases'])} pinned source cases")
