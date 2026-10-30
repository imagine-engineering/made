import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class Tensor(np.ndarray):
    def new_zeros(self, shape):
        return np.zeros(shape, dtype=self.dtype).view(Tensor)


fixture = json.loads(Path(__file__).with_name("fixture.json").read_text())
root = Path(__file__).resolve().parents[5]
for row in fixture["nodes"]:
    source = root / "projects/sim/ComfyUI" / row["source"]["path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row["source"]["sha256"]

reference = fixture["vp_reference"]
source = root / reference["source"]
assert hashlib.sha256(source.read_bytes()).hexdigest() == reference["source_sha256"]
torch = SimpleNamespace(
    linspace=lambda start, end, steps, device="cpu": np.linspace(
        start, end, steps, dtype=np.float32
    ).view(Tensor),
    cat=lambda values: np.concatenate(values).view(Tensor),
    sqrt=lambda values: np.sqrt(values).view(Tensor),
    special=SimpleNamespace(expm1=lambda values: np.expm1(values).view(Tensor)),
)
functions = [
    node
    for node in ast.parse(source.read_bytes()).body
    if isinstance(node, ast.FunctionDef)
    and node.name in {"append_zero", "get_sigmas_vp"}
]
assert len(functions) == 2
namespace = {"torch": torch}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
for case in reference["cases"]:
    with np.errstate(over="ignore", invalid="raise"):
        sigmas = namespace["get_sigmas_vp"](
            case["steps"], case["beta_d"], case["beta_min"], case["eps_s"]
        )
    actual = ["+infinity" if np.isposinf(value) else float(value) for value in sigmas]
    assert actual == case["sigmas"], case
print("Verified", len(reference["cases"]), "pinned VP schedule cases")
