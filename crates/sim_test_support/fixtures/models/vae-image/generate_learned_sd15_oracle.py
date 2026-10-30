"""Independent CPU oracle for the test-only, width-four SD15 VAE boundary."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform

import torch
import torch.nn.functional as functional


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[4]
OUTPUT = DIRECTORY / "learned-sd15-reduced.json"
SOURCE = "projects/sim/ComfyUI/sim/ldm/modules/diffusionmodules/model.py"
SOURCE_SHA256 = "6b4422fd14b0c8fc0f709aebbc94b51a712f2667fb607dd245c5643e6e316be4"
LATENT_SOURCE = "projects/sim/ComfyUI/sim/latent_formats.py"
LATENT_SOURCE_SHA256 = "ab6278dbbd3a999d051a21b59c138fe2c42b303792d47d744e7f63da18544aac"


def generate():
    source = (ROOT / SOURCE).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Pinned Comfy VAE source changed")
    if hashlib.sha256((ROOT / LATENT_SOURCE).read_bytes()).hexdigest() != LATENT_SOURCE_SHA256:
        raise RuntimeError("Pinned Comfy latent scaling source changed")
    if torch.__version__ != "2.8.0+cpu":
        raise RuntimeError("Fixture regeneration requires the recorded torch 2.8.0+cpu oracle")
    # Execute the pinned nonlinearity without importing Comfy's application services.
    definition = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "nonlinearity"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), SOURCE, "exec"), namespace)
    nonlinearity = namespace["nonlinearity"]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False

    def weight(shape):
        count = 1
        for dimension in shape:
            count *= dimension
        values = ((torch.arange(count) + 21) % 17).to(torch.float32) - 8
        return (values * 0.0075).reshape(shape)

    bias = torch.tensor(0.001, dtype=torch.float32) * 3

    def convolution(value, output_channels, kernel=3, stride=1, padding=1):
        weights = weight([output_channels, value.shape[1], kernel, kernel])
        return functional.conv2d(value, weights, bias.expand(output_channels), stride, padding)

    def normalize(value):
        return functional.group_norm(value, 1, torch.ones(4), bias.expand(4), 1e-6)

    def block(value):
        hidden = convolution(nonlinearity(normalize(value)), 4)
        hidden = convolution(nonlinearity(normalize(hidden)), 4)
        return value + hidden

    pixels = ((torch.arange(768) * 37) % 251).to(torch.float32).reshape(1, 3, 16, 16) / 250
    hidden = block(convolution(pixels * 2 - 1, 4))
    for _ in range(3):
        hidden = block(convolution(hidden, 4, stride=2))
    hidden = convolution(nonlinearity(normalize(hidden)), 8)
    encoded = convolution(hidden, 8, kernel=1, padding=0)[:, :4].contiguous() * 0.18215

    hidden = convolution(encoded / 0.18215, 4, kernel=1, padding=0)
    hidden = block(convolution(hidden, 4))
    for _ in range(3):
        hidden = functional.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = block(convolution(hidden, 4))
    hidden = convolution(nonlinearity(normalize(hidden)), 3)
    decoded = ((hidden + 1) * 0.5).clamp(0, 1)

    return {
        "schema_version": 1,
        "fixture_id": "sd15-learned-reduced-boundary-v1",
        "oracle_kind": "independent-pytorch-cpu-source-equations",
        "production_dependency": False,
        "source": {"path": SOURCE, "sha256": SOURCE_SHA256},
        "latent_source": {"path": LATENT_SOURCE, "sha256": LATENT_SOURCE_SHA256},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform(), "threads": 1, "mkldnn": False},
        "reduction": {"width": 4, "groups": 1, "epsilon": 1e-6, "seed": 3,
                      "shared_residual_weights": True, "downsample_padding": 1,
                      "latent_scale": 0.18215, "latent_mode": "first-four-moment-channels",
                      "pixels": "float32((index * 37) % 251) / 250",
                      "weights": "float32(((index + seed * 7) % 17) - 8) * 0.0075",
                      "bias": "float32(0.001) * seed", "norm_weight": 1.0},
        "tolerance": {"absolute": 2e-6, "relative": 2e-5},
        "encoded": {"shape": list(encoded.shape), "values": encoded.flatten().tolist()},
        "decoded": {"shape": list(decoded.shape), "values": decoded.flatten().tolist()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    encoded = (json.dumps(generate(), indent=2) + "\n").encode()
    if arguments.check:
        if OUTPUT.read_bytes() != encoded:
            raise RuntimeError("Independent learned SD15 VAE oracle is stale")
        print("Independent learned SD15 VAE source oracle is current")
    else:
        OUTPUT.write_bytes(encoded)


if __name__ == "__main__":
    main()
