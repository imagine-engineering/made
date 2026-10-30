#!/usr/bin/env python3
"""Recheck pinned LatentBatch and LatentConcat equations with shape-only tensors."""
import ast
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "projects/sim/ComfyUI/sim_extras/nodes_latent.py"
UTILS = ROOT / "projects/sim/ComfyUI/sim/utils.py"
FIXTURE = Path(__file__).with_name("batch_concat.json")
PINS = {
    SOURCE: "cda1a84ae9e0f33db7395b05e813dc3ec592fbe61ab1af44dda72af1486ff961",
    UTILS: "8b8805ca837e20c922a846854156d10e214654f69df96be90969522f9def2bdb",
}


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def repeat(self, repeats):
        return FakeTensor([value * factor for value, factor in zip(self.shape, repeats)])

    def narrow(self, dimension, start, amount):
        shape = list(self.shape)
        shape[dimension] = amount
        return FakeTensor(shape)


def cat(tensors, dim=0):
    dimension = dim
    shape = list(tensors[0].shape)
    shape[dimension] = sum(tensor.shape[dimension] for tensor in tensors)
    if any(any(left != right for index, (left, right) in enumerate(zip(shape, tensor.shape)) if index != dimension % len(shape)) for tensor in tensors):
        raise ValueError("incompatible source cat")
    return FakeTensor(shape)


def common_upscale(tensor, width, height, method, crop):
    assert method == "bilinear" and crop == "center"
    return FakeTensor([*tensor.shape[:-2], height, width])


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def expected_cases():
    for path, pin in PINS.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != pin:
            raise RuntimeError(f"Pinned source changed: {path}")
    source_tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    utils_tree = ast.parse(UTILS.read_text(), filename=str(UTILS))
    methods = []
    for name in ("LatentBatch", "LatentConcat"):
        cls = next(node for node in source_tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
        method.decorator_list = []
        method.name = f"execute_{name}"
        methods.append(method)
    namespace = {
        "io": SimpleNamespace(NodeOutput=lambda value: value),
        "torch": SimpleNamespace(cat=cat),
        "math": math,
    }
    sim_utils = SimpleNamespace(common_upscale=common_upscale)
    sim_utils.repeat_to_batch_size = None
    namespace["sim"] = SimpleNamespace(utils=sim_utils)
    module = ast.fix_missing_locations(ast.Module(body=[function(utils_tree, "repeat_to_batch_size"), function(source_tree, "reshape_latent_to"), *methods], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    sim_utils.repeat_to_batch_size = namespace["repeat_to_batch_size"]
    cases = []
    for class_type, first_shape, second_shape, first_indices, second_indices, dim in [
        ("LatentBatch", [2, 4, 8, 8], [1, 4, 8, 8], None, [9], None),
        ("LatentBatch", [2, 4, 8, 8], [3, 4, 6, 6], [5, 6], None, None),
        ("LatentBatch", [1, 4, 8, 8], [1, 4, 8, 8], [7, 8], None, None),
        ("LatentConcat", [2, 4, 8, 8], [1, 4, 8, 2], [5, 6], None, "x"),
        ("LatentConcat", [2, 4, 8, 8], [1, 4, 3, 8], None, None, "-y"),
    ]:
        first = {"samples": FakeTensor(first_shape), "extra": "first"}
        second = {"samples": FakeTensor(second_shape), "extra": "second"}
        if first_indices is not None:
            first["batch_index"] = list(first_indices)
        if second_indices is not None:
            second["batch_index"] = list(second_indices)
        args = (first, second) if dim is None else (first, second, dim)
        result = namespace[f"execute_{class_type}"](None, *args)
        cases.append({
            "class_type": class_type, "first_shape": first_shape, "second_shape": second_shape,
            "first_batch_index": first_indices, "second_batch_index": second_indices,
            "dim": dim, "expected_shape": list(result["samples"].shape),
            "expected_batch_index": result.get("batch_index"), "retained_extra": result["extra"],
        })
    return {"source_sha256": PINS[SOURCE], "utils_sha256": PINS[UTILS], "cases": cases}


if __name__ == "__main__":
    expected = expected_cases()
    if sys.argv[1:] == ["--write"]:
        FIXTURE.write_text(json.dumps(expected, indent=2) + "\n")
    elif json.loads(FIXTURE.read_text()) != expected:
        raise SystemExit("Batch/Concat source fixture is stale")
    print(f"Verified {len(expected['cases'])} pinned source cases")
