#!/usr/bin/env python3
"""Recheck pinned LatentCutToBatch shape and index passthrough without ComfyUI imports."""
import ast
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "projects/sim/ComfyUI/sim_extras/nodes_latent.py"
FIXTURE = Path(__file__).with_name("cut_to_batch.json")
PINNED_SHA256 = "cda1a84ae9e0f33db7395b05e813dc3ec592fbe61ab1af44dda72af1486ff961"


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(shape)

    def movedim(self, source, destination):
        shape = list(self.shape)
        value = shape.pop(source)
        shape.insert(destination, value)
        return FakeTensor(shape)

    def __getitem__(self, indices):
        shape = []
        for size, index in zip(self.shape, indices):
            start, stop, step = index.indices(size)
            shape.append(len(range(start, stop, step)))
        shape.extend(self.shape[len(indices):])
        return FakeTensor(shape)

    def reshape(self, new_shape):
        shape = list(new_shape)
        if -1 in shape:
            known = math.prod(value for value in shape if value != -1)
            shape[shape.index(-1)] = math.prod(self.shape) // known
        if math.prod(shape) != math.prod(self.shape):
            raise ValueError("invalid source reshape")
        return FakeTensor(shape)


def expected_cases():
    source = SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != PINNED_SHA256:
        raise RuntimeError("Pinned nodes_latent.py source changed")
    tree = ast.parse(source, filename=str(SOURCE))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LatentCutToBatch")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    method.decorator_list = []
    namespace = {"io": SimpleNamespace(NodeOutput=lambda value: value), "math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(SOURCE), "exec"), namespace)
    cases = []
    for shape, dim, slice_size, indices in [
        ([2, 4, 5, 3, 3], "t", 2, [9, 10]),
        ([2, 4, 5, 3, 3], "x", 2, None),
        ([2, 4, 5, 3, 3], "y", 9, None),
        ([2, 4, 3, 3], "t", 2, [9, 10]),
    ]:
        samples = {"samples": FakeTensor(shape), "extra": "retained"}
        if indices is not None:
            samples["batch_index"] = list(indices)
        result = namespace["execute"](None, samples, dim, slice_size)
        cases.append({
            "shape": shape, "dim": dim, "slice_size": slice_size,
            "batch_index": indices, "expected_shape": list(result["samples"].shape),
            "expected_batch_index": result.get("batch_index"),
            "returns_original": result is samples, "retained_extra": result["extra"],
        })
    return {"source_sha256": PINNED_SHA256, "class_type": cls.name, "cases": cases}


if __name__ == "__main__":
    expected = expected_cases()
    if sys.argv[1:] == ["--write"]:
        FIXTURE.write_text(json.dumps(expected, indent=2) + "\n")
    elif json.loads(FIXTURE.read_text()) != expected:
        raise SystemExit("LatentCutToBatch source fixture is stale")
    print(f"Verified {len(expected['cases'])} pinned source cases")
