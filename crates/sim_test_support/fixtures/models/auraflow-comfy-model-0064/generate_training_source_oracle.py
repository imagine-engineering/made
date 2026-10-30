#!/usr/bin/env python3
"""Evaluate pinned MMDiT and LoraDiff from explicit native fixture inputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch

from generate_source_oracle import ROOT, pinned_model


def generate(input_path):
    model, sources = pinned_model()
    source_path = ROOT / "projects/sim/ComfyUI/sim/weight_adapter/lora.py"
    source = source_path.read_bytes()
    expected = "e506062b4eb189be4c36f88270e0fd4dcce038c79a98f7163604ed6b44efe4b5"
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError("pinned LoraDiff source changed")
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LoraDiff"]
    if len(selected) != 1:
        raise ValueError("pinned LoraDiff class is absent")
    namespace = {"torch": torch, "F": torch.nn.functional, "WeightAdapterTrainBase": torch.nn.Module}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    sources["projects/sim/ComfyUI/sim/weight_adapter/lora.py"] = {
        "sha256": expected,
        "symbols": {"LoraDiff": hashlib.sha256(ast.dump(selected[0], include_attributes=False).encode()).hexdigest()},
    }
    inputs = json.loads(input_path.read_bytes())
    tensor = lambda value: torch.tensor(value["values"], dtype=torch.float32).reshape(value["shape"])
    base = {name: tensor(value) for name, value in inputs["base"].items()}
    trainable = set(inputs["trainable_names"])
    parameters = {name: torch.nn.Parameter(tensor(value), requires_grad=name in trainable)
                  for name, value in inputs["parameters"].items()}
    adapters = {}
    for name in base:
        up_name = name + ".lora_up.weight"
        if up_name in parameters:
            down_name = name + ".lora_down.weight"
            alpha_name = name + ".alpha"
            adapter = namespace["LoraDiff"]((parameters[up_name], parameters[down_name],
                                              parameters[alpha_name].item(), None, None, None))
            adapter.lora_up.weight = parameters[up_name]
            adapter.lora_down.weight = parameters[down_name]
            adapter.alpha = parameters[alpha_name]
            adapters[name] = adapter
    model.forward = model._forward
    model_names = set(dict(model.named_parameters()))
    optimizer = torch.optim.AdamW([parameters[name] for name in inputs["trainable_names"]],
                                 lr=0.005, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                                 foreach=False, fused=False)
    target = torch.tensor(inputs["target"]).reshape(1, 4, 3, 3)
    conditioning = torch.tensor(inputs["conditioning"]).reshape(1, 2, 2048)
    losses = []
    for index, microstep in enumerate(inputs["microsteps"]):
        effective = {}
        for name, value in base.items():
            source_name = name.removeprefix("native.")
            if source_name not in model_names:
                continue
            if name in adapters:
                value = adapters[name](value)
            elif name + ".diff" in parameters:
                value = value + parameters[name + ".diff"]
            effective[source_name] = value
        if set(effective) != model_names:
            raise ValueError("source model weight mapping is incomplete")
        latent = torch.tensor(microstep["latent"]).reshape(target.shape)
        model_time = torch.tensor(microstep["model_time"])
        output = torch.func.functional_call(model, effective, (latent, model_time, conditioning), strict=True)
        loss = (output - target).square().mean()
        losses.append(loss.item())
        (loss / 2).backward()
        if (index + 1) % 2 == 0:
            # The native optimizer supplies explicit zero gradients for unused detector-only fixture weights.
            for name in trainable:
                if parameters[name].grad is None:
                    parameters[name].grad = torch.zeros_like(parameters[name])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    if len(losses) != 4:
        raise ValueError("source training oracle requires four microsteps")
    return {
        "schema_version": 1,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source_builder_sha256": hashlib.sha256(Path(__file__).with_name("generate_source_oracle.py").read_bytes()).hexdigest(),
        "environment": {"torch": torch.__version__, "threads": 1, "mkldnn": False},
        "sources": sources,
        "scope": "Pinned MMDiT forward and LoraDiff composition, Torch autograd and AdamW, complete final fixture adapter state and four losses. Native initial weights/noisy samples are explicit inputs; no source RNG or complete training-policy equivalence claim.",
        "adapters": "WeightAdapterTrainBase uses nn.Module initialization only; source LoraDiff constructor/call unchanged. Functional call supplies all source parameters; no wrapper/patch hooks. Explicit zero gradients for unused detector-only fixture keys.",
        "absolute_tolerance": 2e-6,
        "relative_tolerance": 2e-5,
        "losses": losses,
        "state": {name: {"shape": list(value.shape), "values": value.detach().flatten().tolist()}
                  for name, value in parameters.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("training-source-inputs.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("training-source-oracle.json"))
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    encoded = (json.dumps(generate(arguments.input), indent=2, sort_keys=True) + "\n").encode()
    if arguments.check:
        if arguments.output.read_bytes() != encoded:
            raise ValueError("source training oracle differs from checked source execution")
    else:
        arguments.output.write_bytes(encoded)


if __name__ == "__main__":
    main()
