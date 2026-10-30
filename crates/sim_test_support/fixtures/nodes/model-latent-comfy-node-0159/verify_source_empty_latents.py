import ast
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def source_results():
    root = Path(__file__).resolve().parents[5]
    source = root / "projects/sim/ComfyUI"
    identifiers = {f"COMFY-NODE-{number:04}" for number in [159, 161, 162, 163, 164, 165, 166, 167]}
    namespace = {
        "torch": torch,
        "io": SimpleNamespace(NodeOutput=lambda *values: values),
        "IO": SimpleNamespace(NodeOutput=lambda *values: values),
        "sim": SimpleNamespace(model_management=SimpleNamespace(
            intermediate_device=lambda: "cpu", intermediate_dtype=lambda: torch.float32,
        )),
    }
    cases = []
    sources = {}
    with (root / "crates/sim_test_support/parity/catalogs/backend-nodes.csv").open() as catalog:
        for row in csv.DictReader(catalog):
            if row["feature_id"] not in identifiers:
                continue
            path = source / row["source_file"]
            sources[row["source_file"]] = hashlib.sha256(path.read_bytes()).hexdigest()
            original = next(node for node in ast.parse(path.read_text()).body
                            if isinstance(node, ast.ClassDef) and node.name == row["class_name"])
            method = next(node for node in original.body
                          if isinstance(node, ast.FunctionDef) and node.name in {"execute", "generate"})
            selected = ast.ClassDef(name=original.name, bases=[], keywords=[], body=[method], decorator_list=[])
            module = ast.fix_missing_locations(ast.Module(
                body=ast.parse("from __future__ import annotations").body + [selected], type_ignores=[],
            ))
            exec(compile(module, str(path), "exec"), namespace)
            instance = namespace[original.name]()
            names = {argument.arg for argument in method.args.args + method.args.kwonlyargs} - {"cls", "self"}
            if "seconds" in names:
                inputs = [dict(seconds=seconds, batch_size=batch)
                          for seconds in [1.0, 1.02, 1.06, 2.5, 47.6] for batch in [1, 2]]
            elif "resolution" in names:
                inputs = [dict(resolution=resolution, batch_size=batch)
                          for resolution in [1, 2, 33, 3072] for batch in [1, 2]]
            else:
                inputs = []
                for width, height in [(64, 64), (65, 67), (96, 128)]:
                    for batch in [1, 2]:
                        if "length" in names:
                            lengths = [7, 8, 13, 14] if original.name == "EmptyMochiLatentVideo" else [1, 4, 5, 9]
                            inputs.extend(dict(width=width, height=height, batch_size=batch, length=length) for length in lengths)
                        elif "layers" in names:
                            inputs.extend(dict(width=width, height=height, batch_size=batch, layers=layers) for layers in [0, 1, 3])
                        else:
                            inputs.append(dict(width=width, height=height, batch_size=batch))
            for values in inputs:
                output = getattr(instance, method.name)(**values)[0]
                samples = output["samples"]
                assert samples.dtype == torch.float32 and torch.count_nonzero(samples).item() == 0
                cases.append(dict(feature_id=row["feature_id"], class_type=row["node_identifier"],
                                  inputs=values, shape=list(samples.shape),
                                  metadata={key: value for key, value in output.items() if key != "samples"},
                                  dtype=str(samples.dtype), all_zero=True))
    return dict(provenance="Pinned source execute/generate methods with actual CPU float32 PyTorch allocations; adapters supply only NodeOutput containers and the CPU intermediate device/dtype.",
                pytorch_version=torch.__version__, source_sha256=sources, cases=cases)


if __name__ == "__main__":
    actual = source_results()
    expected = json.loads(Path(__file__).with_name("source_empty_latents.json").read_text())
    assert actual["source_sha256"] == expected["source_sha256"]
    assert actual["cases"] == expected["cases"]
    print(len(actual["cases"]), "actual-source empty latent cases")
