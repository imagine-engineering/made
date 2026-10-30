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
    source_hashes = {}
    for relative, names in [
        ("sim/utils.py", {"common_upscale", "repeat_to_batch_size"}),
        ("sim_extras/nodes_latent.py", {"reshape_latent_to"}),
    ]:
        path = root / relative
        source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        selected = [node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=ast.parse("from __future__ import annotations").body + selected,
                                type_ignores=[]), str(path), "exec"), namespace)
    path = root / "sim_extras/nodes_latent.py"
    original = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.ClassDef) and node.name == "LatentAdd")
    method = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    selected = ast.ClassDef(name=original.name, bases=[], keywords=[], body=[method], decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=ast.parse("from __future__ import annotations").body + [selected], type_ignores=[],
    )), str(path), "exec"), namespace)
    namespace["sim"] = SimpleNamespace(utils=SimpleNamespace(
        common_upscale=namespace["common_upscale"], repeat_to_batch_size=namespace["repeat_to_batch_size"],
    ))
    shapes = [[1, 4, 2, 3], [2, 4, 2, 3], [1, 4, 3, 2], [1, 4, 5, 3],
              [1, 8, 2, 3], [1, 4, 2, 2, 3], [2, 4, 2, 3, 2], [1, 64, 5],
              [1, 1, 2, 3], [3, 4, 2, 3]]
    cases = []
    for first_shape in shapes:
        for second_shape in shapes:
            first_samples = torch.arange(math.prod(first_shape), dtype=torch.float32).reshape(first_shape) / 32
            second_samples = torch.arange(math.prod(second_shape), dtype=torch.float32).reshape(second_shape) / 16
            mask_shape = [1, 2, 3] if len(first_shape) != 3 else [1, 1, 3]
            mask = torch.ones(mask_shape)
            indices = [11 + index for index in range(first_shape[0])]
            first = dict(samples=first_samples, noise_mask=mask, batch_index=indices, downscale_ratio_spacial=8)
            second = dict(samples=second_samples, downscale_ratio_spacial=16)
            case = dict(first_shape=first_shape, second_shape=second_shape, mask_shape=mask_shape)
            try:
                output = namespace["LatentAdd"].execute(first, second)[0]
                assert output["noise_mask"] is mask and output["batch_index"] is indices
                assert output["downscale_ratio_spacial"] == 8
                case.update(output_shape=list(output["samples"].shape),
                            output_values=output["samples"].flatten().tolist(),
                            batch_indices=indices, original_mask_retained=True)
            except Exception as error:
                case.update(error=type(error).__name__, message=str(error))
            cases.append(case)
    return dict(provenance="Actual pinned LatentAdd.execute, reshape_latent_to, repeat_to_batch_size and common_upscale with CPU float32 PyTorch tensors, broadcast channels and batch repeat/truncation.",
                pytorch_version=torch.__version__, source_sha256=source_hashes, cases=cases)


if __name__ == "__main__":
    actual = source_results()
    expected = json.loads(Path(__file__).with_name("source_latent_add.json").read_text())
    assert actual["source_sha256"] == expected["source_sha256"]
    assert actual["cases"] == expected["cases"]
    print(len(actual["cases"]), "actual source latent-add cases;",
          sum("error" not in case for case in actual["cases"]), "successes")
