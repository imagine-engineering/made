import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch


def source_results():
    root = Path(__file__).resolve().parents[5] / "projects/sim/ComfyUI"
    namespace = {"torch": torch, "math": math, "io": SimpleNamespace(NodeOutput=lambda *values: values)}
    hashes = {}
    for relative, names in [
        ("sim/nested_tensor.py", {"NestedTensor"}),
        ("sim/utils.py", {"common_upscale", "repeat_to_batch_size"}),
        ("sim_extras/nodes_latent.py", {"reshape_latent_to"}),
    ]:
        path = root / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        selected = [node for node in ast.parse(path.read_text()).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        exec(compile(ast.Module(body=ast.parse("from __future__ import annotations").body + selected, type_ignores=[]),
                     str(path), "exec"), namespace)
    path = root / "sim_extras/nodes_latent.py"
    original = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "LatentAdd")
    method = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    selected = ast.ClassDef(name=original.name, bases=[], keywords=[], body=[method], decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=ast.parse("from __future__ import annotations").body + [selected], type_ignores=[])), str(path), "exec"), namespace)
    namespace["sim"] = SimpleNamespace(utils=SimpleNamespace(common_upscale=namespace["common_upscale"], repeat_to_batch_size=namespace["repeat_to_batch_size"]))
    configurations = [
        (True, True, [1, 2, 1, 1, 4], [1, 3, 4]),
        (True, True, [2, 2, 1, 1, 4], [2, 3, 4]),
        (True, True, [1, 2, 1, 1, 2], [1, 3, 4]),
        (True, True, [1, 2, 1, 1, 4], [1, 1, 4]),
        (False, True, [1, 2, 1, 1, 4], [1, 3, 4]),
        (True, False, [1, 1, 1, 1, 4], None),
        (True, False, [1, 1, 1, 4], None),
    ]
    cases = []
    for first_nested, second_nested, second_shape, second_audio_shape in configurations:
        first_shape, first_audio_shape = [1, 2, 1, 1, 4], [1, 3, 4]
        first = torch.full(first_shape, 2.0)
        if first_nested:
            first = namespace["NestedTensor"]([first, torch.full(first_audio_shape, 3.0)])
        second = torch.full(second_shape, 5.0)
        if second_nested:
            second = namespace["NestedTensor"]([second, torch.full(second_audio_shape, 7.0)])
        case = dict(first_nested=first_nested, second_nested=second_nested, first_shape=first_shape,
                    first_audio_shape=first_audio_shape, second_shape=second_shape, second_audio_shape=second_audio_shape)
        try:
            output = namespace["LatentAdd"].execute(dict(samples=first), dict(samples=second))[0]["samples"]
            tensors = output.tensors if getattr(output, "is_nested", False) else [output]
            case["outputs"] = [dict(shape=list(tensor.shape), values=tensor.flatten().tolist()) for tensor in tensors]
        except (TypeError, AttributeError, RuntimeError, ValueError) as error:
            case.update(error=type(error).__name__, message=str(error))
        cases.append(case)
    return dict(provenance="Actual pinned NestedTensor, LatentAdd and reshape/repeat helpers with PyTorch CPU tensors, including nested broadcasting and source failures.", source_sha256=hashes, pytorch_version=torch.__version__, cases=cases)


if __name__ == "__main__":
    actual = source_results()
    expected = json.loads(Path(__file__).with_name("source_nested_add.json").read_text())
    assert actual["source_sha256"] == expected["source_sha256"]
    assert actual["cases"] == expected["cases"]
    print(len(actual["cases"]), "actual-source nested-add cases")
