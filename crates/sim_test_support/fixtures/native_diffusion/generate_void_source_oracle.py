import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import generate_source_equations as source

DIRECTORY = Path(__file__).resolve().parent
OUTPUT = DIRECTORY / "void-source-equations.json"
VOID_SOURCE = source.ROOT / "projects/sim/ComfyUI/sim_extras/nodes_void.py"
VOID_SOURCE_SHA256 = "b80cd0b5831335087fd29f959be2659820b7927c8b7622cb4b3e9c9fde2c6d4f"


def generate():
    if source.digest(VOID_SOURCE) != VOID_SOURCE_SHA256:
        raise RuntimeError("Pinned VOID source changed")
    selected = [node for node in ast.parse(VOID_SOURCE.read_bytes()).body
                if isinstance(node, ast.ClassDef) and node.name == "VOID_DDIM"]
    if len(selected) != 1:
        raise RuntimeError("Pinned VOID sampler class missing")
    namespace = {"torch": torch, "trange": lambda count, **options: range(count),
                 "sim": SimpleNamespace(samplers=SimpleNamespace(Sampler=object))}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(VOID_SOURCE), "exec"), namespace)

    def sample(guided, initial, sigmas, callback):
        def model_wrap(latent, sigma, **options):
            return guided(latent, sigma)

        def observe(step, denoised, latent, total):
            callback({"x": latent, "denoised": denoised})

        return namespace["VOID_DDIM"]().sample(
            model_wrap, sigmas, {"seed": 0x0123456789ABCDEF}, observe, initial,
            disable_pbar=True)

    oracle = source.generate(sampler_override=sample, raw_initial_noise=True)
    oracle["fixture_id"] = "sd15-tiny-void-source-equations-v1"
    oracle["sources"][str(VOID_SOURCE.relative_to(source.ROOT))] = VOID_SOURCE_SHA256
    oracle["base_generator_sha256"] = oracle["generator_sha256"]
    oracle["generator_sha256"] = source.digest(Path(__file__))
    oracle["scope"] = ("Actual pinned VOID_DDIM class with raw noise initialization and alpha-space updates; "
                       "pinned CLIP and EPS/model-time source functions and reduced UNet/VAE source equations. "
                       "Noise and sigmas are exact input fixtures, not source RNG or schedule generation.")
    return oracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    encoded = (json.dumps(generate(), indent=2, sort_keys=True) + "\n").encode()
    if arguments.check:
        if OUTPUT.read_bytes() != encoded:
            raise RuntimeError("Independent VOID source oracle is stale")
        print("Independent VOID source oracle is current")
    else:
        OUTPUT.write_bytes(encoded)


if __name__ == "__main__":
    main()
