import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch

root = Path(__file__).resolve().parents[5]
source_root = root / "projects/sim/ComfyUI/sim"
namespace = {"torch": torch, "math": math}
sources = [source_root / name for name in ["latent_formats.py", "sample.py", "utils.py"]]
classes = {"LatentFormat", "SD15", "SD3", "Flux2", "Wan21", "LTXV"}
functions = {"fix_empty_latent_channels", "repeat_to_batch_size", "common_upscale"}
for source in sources:
    selected = [node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name in classes
                or isinstance(node, ast.FunctionDef) and node.name in functions]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
namespace["sim"] = SimpleNamespace(utils=SimpleNamespace(
    repeat_to_batch_size=namespace["repeat_to_batch_size"],
    common_upscale=namespace["common_upscale"]))
cases = []
for name in ["SD15", "SD3", "Flux2", "Wan21", "LTXV"]:
    definition = namespace[name]()
    model = SimpleNamespace(get_model_object=lambda key: definition)
    for shape in [[1, 1, 3, 5], [2, 4, 4, 6], [1, 2, 3, 3, 5]]:
        for nonempty in [False, True]:
            for spatial in [None, 8, 16, 32]:
                for temporal in [None, 1, 8]:
                    samples = torch.zeros(shape, dtype=torch.float32)
                    if nonempty:
                        samples.flatten()[-1] = 1.25
                    case = dict(format=name, shape=shape, nonempty=nonempty,
                                spatial=spatial, temporal=temporal)
                    try:
                        output = namespace["fix_empty_latent_channels"](
                            model, samples, spatial, temporal)
                        case["output_shape"] = list(output.shape)
                        case["output_sum"] = output.double().sum().item()
                        case["shares_storage"] = output.untyped_storage().data_ptr() == samples.untyped_storage().data_ptr()
                    except Exception as error:
                        case["error"] = type(error).__name__
                    cases.append(case)
result = {"provenance": "Actual pinned latent-format classes and fix_empty_latent_channels, repeat_to_batch_size, common_upscale functions executed with real PyTorch tensors; model wrapper supplies only its actual latent-format object.",
          "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
          "cases": cases}
path = Path(__file__).with_name("source_empty_latents.json")
assert json.loads(json.dumps(result)) == json.loads(path.read_text()), "pinned empty-latent source changed"
print(len(cases), "source empty-latent cases verified")
