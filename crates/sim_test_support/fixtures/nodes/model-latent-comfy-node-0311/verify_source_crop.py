#!/usr/bin/env python3
"""Recheck the pinned V1 LatentCrop slice geometry without importing ComfyUI."""
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "projects/sim/ComfyUI/nodes.py"
FIXTURE = Path(__file__).with_name("crop.json")
PINNED_SHA256 = "b8dfdde1de8975be762b085048143cc2dda8fc9202695e460ecc2c8dfe44bc4b"


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def __getitem__(self, indices):
        shape = []
        for size, index in zip(self.shape, indices):
            if not isinstance(index, slice):
                raise ValueError("expected source slice")
            start, stop, step = index.indices(size)
            shape.append(len(range(start, stop, step)))
        return FakeTensor(shape)


def expected_cases():
    source = SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != PINNED_SHA256:
        raise RuntimeError("Pinned nodes.py source changed")
    tree = ast.parse(source, filename=str(SOURCE))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LatentCrop")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "crop")
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(SOURCE), "exec"), namespace)
    cases = []
    for width, height, x, y in [(64, 64, 0, 0), (512, 512, 0, 0), (64, 64, 200, 200), (80, 72, 40, 64)]:
        samples = {"samples": FakeTensor([1, 4, 16, 20]), "extra": "retained"}
        result = namespace["crop"](None, samples, width, height, x, y)[0]
        cases.append({
            "width": width, "height": height, "x": x, "y": y,
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
        raise SystemExit("LatentCrop source fixture is stale")
    print(f"Verified {len(expected['cases'])} pinned source cases")
