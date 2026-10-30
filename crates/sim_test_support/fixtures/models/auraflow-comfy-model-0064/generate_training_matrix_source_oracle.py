#!/usr/bin/env python3
"""Execute pinned F32 LoHa/OFT adapters and MMDiT from explicit fixture inputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Optional

import torch

from generate_source_oracle import ROOT, pinned_model

SOURCE_HASHES = {
    "loha": "579ca1e33e0d244e0d7eedd30fb727913341f8e7bfbd74b51221f567612286d5",
    "oft": "88be3c32f610478bc6900a10009eaadf6fe2af973ed4861731e4f21e1afacf89",
}


def generate(algorithm):
    model, sources = pinned_model()
    path = ROOT / f"projects/sim/ComfyUI/sim/weight_adapter/{algorithm}.py"
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASHES[algorithm]:
        raise ValueError("pinned adapter source changed")
    symbols = {"HadaWeight", "HadaWeightTucker", "LohaDiff"} if algorithm == "loha" else {"OFTDiff"}
    selected = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name in symbols]
    if len(selected) != len(symbols):
        raise ValueError("source adapter classes are incomplete")
    namespace = {"torch": torch, "F": torch.nn.functional, "Optional": Optional,
                 "WeightAdapterTrainBase": torch.nn.Module}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    sources[path.relative_to(ROOT).as_posix()] = {"sha256": SOURCE_HASHES[algorithm], "symbols": {
        node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() for node in selected}}
    input_path = Path(__file__).with_name(f"{algorithm}-training-inputs.json")
    inputs = json.loads(input_path.read_bytes())
    if inputs["compute_dtype"] not in ("F32", "Preserve"):
        raise ValueError("this oracle does not establish BF16 compute equivalence")
    if inputs["batch_size"] != 1 or len(inputs["samples"]) != inputs["batch_size"]:
        raise ValueError("matrix input samples must match the checked one-sample request")
    tensor = lambda item: torch.tensor(item["values"], dtype=torch.float32).reshape(item["shape"])
    base = {name: tensor(item) for name, item in inputs["base"].items()}
    parameters = {name: torch.nn.Parameter(tensor(item), requires_grad=name in inputs["trainable_names"])
                  for name, item in inputs["parameters"].items()}
    adapters = {}
    for name in base:
        alpha = parameters.get(name + ".alpha")
        if name + ".hada_w1_a" in parameters:
            suffixes = (".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b")
            values = [parameters[name + suffix] for suffix in suffixes]
            adapter = namespace["LohaDiff"]((values[0], values[1], alpha.item(), values[2], values[3], None, None, None))
            for suffix, value in zip(suffixes, values):
                setattr(adapter, suffix.removeprefix('.'), value)
            adapter.alpha = alpha
            adapters[name] = adapter
        elif name + ".oft_blocks" in parameters:
            blocks = parameters[name + ".oft_blocks"]
            adapter = namespace["OFTDiff"]((blocks, None, alpha.item(), None))
            adapter.oft_blocks = blocks
            adapter.alpha = alpha
            adapters[name] = adapter
    projection_cases = []
    if algorithm == "oft":
        projection_base = torch.tensor([[0.2, -0.3, 0.7], [0.8, 0.1, -0.2],
                                        [-0.4, 0.6, 0.9], [0.3, -0.8, 0.5]])
        projection_blocks = torch.tensor([[[0.1, 0.3], [-0.2, 0.4]],
                                          [[-0.3, 0.7], [0.1, 0.2]]])
        for constraint in (0.0, 0.5, 2.0):
            projection = namespace["OFTDiff"]((projection_blocks.clone(), None, constraint, None))
            output = projection(projection_base)
            projection_cases.append({"constraint": constraint,
                                     "base": {"shape": list(projection_base.shape), "values": projection_base.flatten().tolist()},
                                     "blocks": {"shape": list(projection_blocks.shape), "values": projection_blocks.flatten().tolist()},
                                     "output": output.detach().flatten().tolist()})
    model.forward = model._forward
    model_names = set(dict(model.named_parameters()))
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
        raise ValueError("source model mapping is incomplete")
    conditioning = torch.tensor(inputs["conditioning"]).reshape(1, 2, 2048)
    loss_sum = torch.tensor(0.0)
    count = 0
    for sample in inputs["samples"]:
        shape = sample["shape"]
        target = torch.tensor(sample["target"]).reshape(shape)
        latent = torch.tensor(sample["latent"]).reshape(shape)
        time = torch.tensor(sample["model_time"])
        output = torch.func.functional_call(model, effective, (latent, time, conditioning), strict=True)
        if inputs["loss"] == "Huber":
            loss_sum = loss_sum + torch.nn.functional.huber_loss(output, target, reduction="sum", delta=1.0)
        elif inputs["loss"] == "L1":
            loss_sum = loss_sum + torch.nn.functional.l1_loss(output, target, reduction="sum")
        else:
            raise ValueError("unreviewed matrix loss")
        count += target.numel()
    loss = loss_sum / count
    optimizer_type = torch.optim.Adam if inputs["optimizer"] == "Adam" else torch.optim.SGD
    optimizer = optimizer_type([parameters[name] for name in inputs["trainable_names"]], lr=0.005, foreach=False)
    loss.backward()
    for name in inputs["trainable_names"]:
        if parameters[name].grad is None:
            parameters[name].grad = torch.zeros_like(parameters[name])
    optimizer.step()
    return {"schema_version": 1, "sources": sources,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_builder_sha256": hashlib.sha256(Path(__file__).with_name("generate_source_oracle.py").read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "scope": "Actual pinned F32 MMDiT and LohaDiff/OFTDiff, Torch autograd/optimizer/loss, complete final adapter state. Explicit native initialization/noise/dataset inputs; no source RNG or BF16-policy claim.",
            "environment": {"torch": torch.__version__, "threads": 1, "mkldnn": False},
            "absolute_tolerance": 2e-6, "relative_tolerance": 2e-5,
            "oft_projection_cases": projection_cases,
            "losses": [loss.item()], "state": {name: {"shape": list(value.shape), "values": value.detach().flatten().tolist()} for name, value in parameters.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    for algorithm in ("loha", "oft"):
        path = Path(__file__).with_name(f"{algorithm}-training-source-oracle.json")
        encoded = (json.dumps(generate(algorithm), indent=2, sort_keys=True) + "\n").encode()
        if arguments.check:
            if path.read_bytes() != encoded:
                raise ValueError(f"{algorithm} oracle differs from pinned source execution")
        else:
            path.write_bytes(encoded)


if __name__ == "__main__":
    main()
