#!/usr/bin/env python3
"""Run the pinned AuraFlow MMDiT classes at the declared reduced fixture geometry."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import types

import torch
from torch import einsum, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[5]
SOURCE_HASHES = {
    "projects/sim/ComfyUI/sim/ldm/common_dit.py": "5e2459be4a47073a17e6b806f2b93962b4df1cdcaa2ac1890c9573bf942edf82",
    "projects/sim/ComfyUI/sim/ldm/aura/mmdit.py": "0104396eda01a9f78e8aa5b9d15470fc551aa8b0e05137d264f3515fd1739db1",
    "projects/sim/ComfyUI/sim/ldm/modules/attention.py": "436e1d91f8d5d84c5667e051cdf3ab2f91d8db25b66d88a084c89a202de0579e",
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def extract(path, names, namespace):
    source = (ROOT / path).read_bytes()
    if path in SOURCE_HASHES and digest(source) != SOURCE_HASHES[path]:
        raise ValueError(f"pinned source mismatch: {path}")
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if len(selected) != len(names):
        raise ValueError(f"missing selected source symbols: {path}")
    symbol_hashes = {node.name: digest(ast.dump(node, include_attributes=False).encode()) for node in selected}
    for node in selected:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), path, "exec"), namespace)
    return {"sha256": digest(source), "symbols": symbol_hashes}


def pinned_model():
    if sys.version_info[:3] != (3, 12, 14) or torch.__version__ != "2.8.0+cpu":
        raise ValueError("oracle requires Python 3.12.14 and PyTorch 2.8.0+cpu")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    namespace = dict(torch=torch, nn=nn, F=F, math=math, einsum=einsum,
                     args=types.SimpleNamespace(dont_upcast_attention=False), FORCE_UPCAST_ATTENTION_DTYPE=None)
    sources = {}
    attention_path = "projects/sim/ComfyUI/sim/ldm/modules/attention.py"
    sources[attention_path] = extract(attention_path, ["get_attn_precision", "exists", "attention_basic"], namespace)
    padding_path = "projects/sim/ComfyUI/sim/ldm/common_dit.py"
    sources[padding_path] = extract(padding_path, ["pad_to_patch_size"], namespace)
    namespace["optimized_attention"] = namespace["attention_basic"]
    namespace["sim"] = types.SimpleNamespace(
        ops=types.SimpleNamespace(cast_to_input=lambda value, target: value.to(device=target.device, dtype=target.dtype)),
        ldm=types.SimpleNamespace(common_dit=types.SimpleNamespace(pad_to_patch_size=namespace["pad_to_patch_size"])))
    model_path = "projects/sim/ComfyUI/sim/ldm/aura/mmdit.py"
    sources[model_path] = extract(model_path, ["modulate", "find_multiple", "MLP", "MultiHeadLayerNorm", "SingleAttention", "DoubleAttention", "MMDiTBlock", "DiTBlock", "TimestepEmbedder", "MMDiT"], namespace)
    model = namespace["MMDiT"](dim=2, n_layers=2, n_double_layers=1, n_heads=1,
                                global_conddim=2, cond_seq_dim=2048, max_seq=16,
                                dtype=torch.float32, operations=nn)
    return model, sources


def generate():
    model, sources = pinned_model()
    weights = {}
    for name, parameter in model.named_parameters():
        key = "native." + name
        indexes = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        if key.endswith(".bias"):
            values = (indexes.remainder(11) - 5) * 0.002
        elif ".mlpC." in key or ".mlpX." in key or "single_layers.0.mlp." in key:
            values = (indexes.remainder(17) - 8) * 0.5
        elif "native.double_layers.0.attn." in key:
            values = (indexes.remainder(17) - 8) * 0.1
        else:
            values = (indexes.remainder(17) - 8) * 0.00075
        with torch.no_grad():
            parameter.copy_(values)
        weights[key] = {"shape": list(parameter.shape), "values": values.flatten().tolist()}
    legacy_latent = ((torch.arange(36, dtype=torch.float32) - 18) * 0.025).reshape(1, 4, 3, 3)
    legacy_conditioning = ((torch.arange(4096, dtype=torch.float32).remainder(17) - 8) * 0.00075).reshape(1, 2, 2048)
    legacy_output = model._forward(legacy_latent, torch.tensor([0.31415927]), legacy_conditioning).detach().flatten().tolist()
    for name, parameter in model.named_parameters():
        indexes = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        pattern = (indexes * 5 + sum(name.encode())).remainder(17) - 8
        if name.endswith(".bias"):
            values = (indexes.remainder(11) - 5) * (0.0009765625 if name == "init_x_linear.bias" else 0.0625)
        elif name == "final_linear.weight":
            values = pattern * 0.5
        elif name == "cond_seq_linear.weight":
            values = pattern * 0.00075
        elif name in {"init_x_linear.weight", "positional_encoding", "register_tokens"} or ".mlpC." in name or ".mlpX." in name or "single_layers.0.mlp." in name:
            values = pattern * 0.0009765625
        elif name in {"single_layers.0.attn.w1q.weight", "single_layers.0.attn.w1k.weight"}:
            values = pattern * 0.00390625
        else:
            values = pattern * 0.0625
        with torch.no_grad():
            parameter.copy_(values)
        weights["native." + name] = {"shape": list(parameter.shape), "values": values.flatten().tolist()}
    cases = []
    for height, width in [(3, 3), (4, 4)]:
        elements = 4 * height * width
        latent = ((torch.arange(elements, dtype=torch.float32) - 18) * 0.025).reshape(1, 4, height, width)
        conditioning = ((torch.arange(4096, dtype=torch.float32).remainder(17) - 8) * 0.00075).reshape(1, 2, 2048)
        model_time = torch.tensor([0.31415927], dtype=torch.float32)
        target = ((torch.arange(elements, dtype=torch.float32) - 18) * 0.03125).reshape(latent.shape)
        model.zero_grad(set_to_none=True)
        output = model._forward(latent, model_time, conditioning)
        loss = (output - target).square().mean()
        loss.backward()
        gradients = {"native." + name: parameter.grad.flatten().tolist() for name, parameter in model.named_parameters()}
        cases.append({"shape": list(latent.shape), "latent": latent.flatten().tolist(),
                      "conditioning_shape": list(conditioning.shape), "conditioning": conditioning.flatten().tolist(),
                      "model_time": model_time.tolist(), "target": target.flatten().tolist(),
                      "output": output.detach().flatten().tolist(), "loss": loss.item(), "gradients": gradients})
    return {"schema_version": 1, "generator_sha256": digest(Path(__file__).read_bytes()),
            "sources": sources, "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "threads": 1, "mkldnn": False},
            "scope": "Actual pinned MMDiT._forward, source attention_basic and circular padding; width 2, one double and one single block, one head. All outputs and all 9466 parameter VJPs; no full-size architecture, optimizer, RNG or training-service state claim.",
            "adapters": "torch.nn Linear/LayerNorm, no wrapper or patch hooks; cast_to_input performs only source device/dtype cast; attention wrapper removed with no configured wrappers",
            "output_absolute_tolerance": 1e-6, "gradient_absolute_tolerance": 2e-6, "gradient_relative_tolerance": 2e-5,
            "legacy_output": legacy_output, "weights": weights, "cases": cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("source-oracle.json"))
    arguments = parser.parse_args()
    encoded = (json.dumps(generate(), indent=2, sort_keys=True) + "\n").encode()
    if arguments.check:
        if arguments.output.read_bytes() != encoded:
            raise ValueError("checked source oracle differs from pinned source execution")
    else:
        arguments.output.write_bytes(encoded)


if __name__ == "__main__":
    main()
