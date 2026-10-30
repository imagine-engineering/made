import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def source_results():
    root = Path(__file__).resolve().parents[5] / "projects/sim/ComfyUI"
    namespace = {"torch": torch, "nn": torch.nn, "ops": torch.nn,
                 "io": SimpleNamespace(NodeOutput=lambda *values: values)}
    source_hashes = {}
    for relative, names in [
        ("sim/utils.py", {"common_upscale", "bislerp"}),
        ("sim/ldm/modules/diffusionmodules/model.py", {"VideoConv3d"}),
        ("sim/ldm/hunyuan_video/upsampler.py", {"SRResidualCausalBlock3D", "SRModel3DV2"}),
    ]:
        path = root / relative
        source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        selected = [node for node in ast.parse(path.read_text()).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        exec(compile(ast.Module(body=ast.parse("from __future__ import annotations").body + selected,
                                type_ignores=[]), str(path), "exec"), namespace)
    for relative, class_name, method_name in [
        ("sim/ldm/hunyuan_video/upsampler.py", "HunyuanVideo15SRModel", "resample_latent"),
        ("sim_extras/nodes_hunyuan.py", "HunyuanVideo15LatentUpscaleWithModel", "execute"),
    ]:
        path = root / relative
        source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        original = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.ClassDef) and node.name == class_name)
        method = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
        selected = ast.ClassDef(name=original.name, bases=[], keywords=[], body=[method], decorator_list=[])
        exec(compile(ast.fix_missing_locations(ast.Module(
            body=ast.parse("from __future__ import annotations").body + [selected], type_ignores=[],
        )), str(path), "exec"), namespace)
    namespace["sim"] = SimpleNamespace(
        utils=SimpleNamespace(common_upscale=namespace["common_upscale"]),
        model_management=SimpleNamespace(load_model_gpu=lambda patcher: None),
    )
    model = namespace["SRModel3DV2"](in_channels=1, out_channels=1, hidden_channels=1,
                                    num_blocks=1, global_residual=False).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.zero_()
            if name.endswith("weight"):
                parameter.flatten()[22] = 1.0
    state = {name: dict(shape=list(value.shape), values=value.flatten().tolist())
             for name, value in model.state_dict().items()}
    wrapper = namespace["HunyuanVideo15SRModel"]()
    wrapper.model = model
    wrapper.load_device = "cpu"
    wrapper.patcher = model
    samples = torch.arange(12, dtype=torch.float32).reshape(1, 1, 2, 2, 3) / 8 - 0.75
    mask = torch.ones(1, 2, 3)
    bundle = dict(samples=samples, noise_mask=mask, batch_index=[11], downscale_ratio_spacial=16)
    cases = []
    with torch.no_grad():
        for method in ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]:
            for crop in ["disabled", "center"]:
                for width, height in [(0, 0), (0, 80), (80, 0), (1, 1), (64, 80), (81, 65)]:
                    output = namespace["HunyuanVideo15LatentUpscaleWithModel"].execute(
                        wrapper, bundle, method, width, height, crop,
                    )[0]
                    cases.append(dict(inputs=dict(upscale_method=method, crop=crop, width=width, height=height),
                                      output_shape=list(output["samples"].shape),
                                      output_values=output["samples"].flatten().tolist(),
                                      aliases_input=output is bundle, output_keys=sorted(output)))
    return dict(provenance="Actual pinned node, resample_latent, SRModel3DV2, residual blocks, VideoConv3d, common_upscale and bislerp executed with PyTorch CPU tensors. The reduced one-channel, one-block model uses complete identity-convolution state; only model residency admission is a no-op on CPU.",
                pytorch_version=torch.__version__, source_sha256=source_hashes,
                state=state, input_shape=list(samples.shape), input_values=samples.flatten().tolist(), cases=cases)


if __name__ == "__main__":
    actual = source_results()
    expected = json.loads(Path(__file__).with_name("source_hunyuan_upscale.json").read_text())
    for key in ["source_sha256", "state", "input_shape", "input_values", "cases"]:
        assert actual[key] == expected[key], key
    print(len(actual["cases"]), "actual-source Hunyuan upscale cases")
