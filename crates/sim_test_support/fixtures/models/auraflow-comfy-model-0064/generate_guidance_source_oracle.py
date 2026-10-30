#!/usr/bin/env python3
"""Verify source MMDiT guidance branches with the declared legacy input policy."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MODEL_GENERATOR = HERE / "generate_source_oracle.py"


def generate():
    specification = importlib.util.spec_from_file_location("pinned_aura_source", MODEL_GENERATOR)
    source = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(source)
    model, sources = source.pinned_model()
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
    latent = ((torch.arange(36, dtype=torch.float32) - 18) * 0.025).reshape(1, 4, 3, 3)
    indexes = torch.arange(4096, dtype=torch.float32)
    positive = ((indexes.remainder(17) - 8) * 0.00075).reshape(1, 2, 2048)
    negative = ((indexes.remainder(11) - 5) * 0.002).reshape(1, 2, 2048)
    with torch.no_grad():
        results = {name: model._forward(latent, torch.tensor([0.5]), conditioning).flatten().tolist()
                   for name, conditioning in [("positive", positive), ("negative", negative)]}
    return {"schema_version": 1, "scope": "Actual pinned MMDiT forward at sigma/model_time0.5, legacy reduced model weights and explicit positive/negative conditioning; source RNG and broader CFG/training policy are not claimed.",
            "sources": sources, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_generator_sha256": hashlib.sha256(MODEL_GENERATOR.read_bytes()).hexdigest(),
            "absolute_tolerance": 1e-6, "branches": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    arguments = parser.parse_args()
    output = HERE / "guidance-source-oracle.json"
    encoded = (json.dumps(generate(), indent=2, sort_keys=True) + "\n").encode()
    if arguments.generate:
        output.write_bytes(encoded)
    elif output.read_bytes() != encoded:
        raise RuntimeError("Pinned AuraFlow guidance source oracle is stale")


if __name__ == "__main__":
    main()
