import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
from types import SimpleNamespace

import torch
import torch.nn.functional as functional


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[3]
FIXTURE = DIRECTORY.parent / "models/sd15-tiny-v1"
OUTPUT = DIRECTORY / "source-equations.json"
COMFY = ROOT / "projects/sim/ComfyUI/sim"
ADDITIONAL_SOURCES = {
    "clip_model.py": "08be993d86c3b494b58305fb868638b4b525bbe40abead89e9c94da021716845",
    "model_sampling.py": "8559ac2f700b788babf30cc16f410c93890e1f71485ac8f6a299ea36e2cb8717",
    "ldm/modules/diffusionmodules/util.py": "fb58652a35521fc23bdcb75d91adace8e4cc79e2d5b13af1617a38d0c0f7142e",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensors(path):
    data = path.read_bytes()
    header_length = struct.unpack_from("<Q", data)[0]
    header = json.loads(data[8:8 + header_length])
    start = 8 + header_length
    result = {}
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if descriptor["dtype"] != "F32":
            raise RuntimeError(f"Unsupported source input dtype: {name}")
        lower, upper = descriptor["data_offsets"]
        raw = bytearray(data[start + lower:start + upper])
        result[name] = torch.frombuffer(raw, dtype=torch.float32).clone().reshape(descriptor["shape"])
    return result


def extract(relative, names, namespace):
    nodes = ast.parse((COMFY / relative).read_bytes()).body
    selected = [node for node in nodes if
                isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
                or isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    found = {node.name if hasattr(node, "name") else node.targets[0].id for node in selected}
    if found != names:
        raise RuntimeError(f"Incomplete source projection: {relative}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), relative, "exec"), namespace)


def attention(query, key, value, heads, mask=None):
    batch, queries, channels = query.shape
    keys = key.shape[1]
    query = query.reshape(batch, queries, heads, channels // heads).transpose(1, 2)
    key = key.reshape(batch, keys, heads, channels // heads).transpose(1, 2)
    value = value.reshape(batch, keys, heads, channels // heads).transpose(1, 2)
    scores = query @ key.transpose(-2, -1) / math.sqrt(channels // heads)
    if mask is not None:
        scores = scores + mask
    result = scores.softmax(dim=-1) @ value
    return result.transpose(1, 2).reshape(batch, queries, channels)


class SourceEmbedding(torch.nn.Embedding):
    def forward(self, input_tokens, out_dtype=None):
        value = super().forward(input_tokens)
        return value.to(out_dtype) if out_dtype is not None else value


def generate(sampler_override=None, raw_initial_noise=False):
    if torch.__version__ != "2.8.0+cpu":
        raise RuntimeError("Regeneration requires the recorded torch 2.8.0+cpu oracle")
    provenance = json.loads((FIXTURE / "oracle-provenance.json").read_text())
    sources = dict(provenance["sources"])
    sources.update({str((COMFY / path).relative_to(ROOT)): sha for path, sha in ADDITIONAL_SOURCES.items()})
    for path, expected in sources.items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Pinned source changed: {path}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    weights = tensors(FIXTURE / "model.safetensors")
    operations = SimpleNamespace(Linear=torch.nn.Linear, LayerNorm=torch.nn.LayerNorm,
                                 Embedding=SourceEmbedding)
    namespace = {"torch": torch, "math": math,
                 "optimized_attention_for_device": lambda *args, **kwargs: attention,
                 "sim": SimpleNamespace(ops=SimpleNamespace(cast_to=lambda value, **options: value.to(**options)))}
    extract("clip_model.py", {"CLIPAttention", "ACTIVATIONS", "CLIPMLP", "CLIPLayer",
                              "CLIPEncoder", "CLIPEmbeddings", "CLIPTextModel_"}, namespace)
    configuration = {"num_hidden_layers": 2, "hidden_size": 32, "num_attention_heads": 4,
                     "intermediate_size": 128, "hidden_act": "quick_gelu",
                     "max_position_embeddings": 77, "eos_token_id": 49407}
    clip = namespace["CLIPTextModel_"](configuration, torch.float32, "cpu", operations)
    prefix = "cond_stage_model.transformer.text_model."
    clip.load_state_dict({name[len(prefix):]: value for name, value in weights.items() if name.startswith(prefix)}, strict=True)
    tokens = json.loads((FIXTURE / "tokens.json").read_text())
    with torch.no_grad():
        conditioning = {name: clip(torch.tensor([tokens[name]]))[0] for name in ["positive", "negative"]}

    extract("ldm/modules/diffusionmodules/util.py", {"make_beta_schedule"}, namespace)
    extract("model_sampling.py", {"reshape_sigma", "EPS", "ModelSamplingDiscrete"}, namespace)
    sampling = namespace["ModelSamplingDiscrete"]()
    epsilon = namespace["EPS"]()
    epsilon.sigma_data = 1.0
    extract("k_diffusion/utils.py", {"append_dims"}, namespace)
    namespace["utils"] = SimpleNamespace(append_dims=namespace["append_dims"])
    namespace["trange"] = lambda count, **kwargs: range(count)
    extract("k_diffusion/sampling.py", {"to_d", "sample_euler"}, namespace)

    def linear(value, prefix):
        return functional.linear(value, weights[prefix + ".weight"], weights[prefix + ".bias"])

    def convolution(value, prefix, stride=1, padding=1):
        return functional.conv2d(value, weights[prefix + ".weight"], weights[prefix + ".bias"], stride, padding)

    def normalize(value, prefix):
        # A one-lane reduced group is valid algebraically but PyTorch's convenience
        # entry point rejects it; evaluate its biased-variance equation directly.
        batch, channels, height, width = value.shape
        grouped = value.reshape(batch, min(32, channels), -1)
        mean = grouped.mean(dim=-1, keepdim=True)
        variance = ((grouped - mean) ** 2).mean(dim=-1, keepdim=True)
        normalized = ((grouped - mean) / (variance + 1e-6).sqrt()).reshape(value.shape)
        return normalized * weights[prefix + ".weight"].reshape(1, channels, 1, 1) + weights[prefix + ".bias"].reshape(1, channels, 1, 1)

    def residual(value, time, prefix):
        hidden = convolution(functional.silu(normalize(value, prefix + ".norm1")), prefix + ".conv1")
        if time is not None:
            hidden = hidden + linear(time, prefix + ".time_emb_proj").reshape(1, -1, 1, 1)
        hidden = convolution(functional.silu(normalize(hidden, prefix + ".norm2")), prefix + ".conv2")
        if value.shape[1] != hidden.shape[1]:
            value = convolution(value, prefix + ".nin_shortcut", padding=0)
        return value + hidden

    def denoiser(value, model_time, context):
        prefix = "model.diffusion_model"
        frequency = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(16) / 16)
        argument = model_time.float() * frequency
        time = torch.cat([argument.cos(), argument.sin()]).reshape(1, 32)
        time = linear(functional.silu(linear(time, prefix + ".time_embed.0")), prefix + ".time_embed.2")
        context = linear(context, prefix + ".context_projection")
        hidden = convolution(value, prefix + ".input_blocks.0.0")
        skips = []
        for level in range(4):
            hidden = residual(hidden, time, f"{prefix}.down.{level}.res")
            skips.append(hidden)
            if level < 3:
                hidden = convolution(hidden, f"{prefix}.down.{level}.downsample", stride=2)
        hidden = residual(hidden, time, prefix + ".mid.block_1")
        base = prefix + ".mid.attn_1"
        batch, channels, height, width = hidden.shape
        queries = normalize(hidden, base + ".norm").flatten(2).transpose(1, 2)
        attended = attention(linear(queries, base + ".q"), linear(context, base + ".k"), linear(context, base + ".v"), 4)
        hidden = hidden + linear(attended, base + ".proj_out").transpose(1, 2).reshape(batch, channels, height, width)
        hidden = residual(hidden, time, prefix + ".mid.block_2")
        for level in range(4):
            skip = skips.pop()
            hidden = functional.interpolate(hidden, size=skip.shape[2:], mode="nearest")
            hidden = residual(torch.cat([hidden, skip], dim=1), time, f"{prefix}.up.{level}.res")
            if level < 3:
                if hidden.shape[2] < skips[-1].shape[2]:
                    hidden = functional.interpolate(hidden, scale_factor=2, mode="nearest")
                hidden = convolution(hidden, f"{prefix}.up.{level}.upsample")
        return convolution(functional.silu(normalize(hidden, prefix + ".out.0")), prefix + ".out.2")

    noise = tensors(FIXTURE / "initial-noise.safetensors")["noise"]
    sigmas = torch.tensor(struct.unpack("<5d", (FIXTURE / "normal-sigmas.f64le").read_bytes()), dtype=torch.float32)
    initial = noise if raw_initial_noise else epsilon.noise_scaling(sigmas[0], noise, torch.zeros_like(noise), max_denoise=True)
    latents = []
    evaluations = []

    def guided(value, sigma):
        scaled = epsilon.calculate_input(sigma, value)
        time = sampling.timestep(sigma)
        positive = denoiser(scaled, time, conditioning["positive"])
        negative = denoiser(scaled, time, conditioning["negative"])
        return epsilon.calculate_denoised(sigma, negative + (positive - negative) * 7, value)

    def callback(value):
        latents.append(value["x"].clone())
        evaluations.append(value["denoised"].clone())

    with torch.no_grad():
        sampler = sampler_override or namespace["sample_euler"]
        final = sampler(guided, initial, sigmas, callback=callback)
    latents.append(final)

    with torch.no_grad():
        prefix = "first_stage_model"
        hidden = convolution(final / 0.18215, prefix + ".post_quant_conv", padding=0)
        hidden = convolution(hidden, prefix + ".decoder.conv_in")
        hidden = residual(hidden, None, prefix + ".decoder.mid.block_1")
        base = prefix + ".decoder.mid.attn_1"
        batch, channels, height, width = hidden.shape
        queries = normalize(hidden, base + ".norm").flatten(2).transpose(1, 2)
        context = hidden.flatten(2).transpose(1, 2)
        attended = attention(linear(queries, base + ".q"), linear(context, base + ".k"), linear(context, base + ".v"), 4)
        hidden = hidden + linear(attended, base + ".proj_out").transpose(1, 2).reshape(batch, channels, height, width)
        hidden = residual(hidden, None, prefix + ".decoder.mid.block_2")
        for level in range(4):
            hidden = residual(hidden, None, f"{prefix}.decoder.up.{level}.block")
            if level < 3:
                hidden = functional.interpolate(hidden, scale_factor=2, mode="nearest")
                hidden = convolution(hidden, f"{prefix}.decoder.up.{level}.upsample")
        decoded = ((convolution(functional.silu(normalize(hidden, prefix + ".decoder.norm_out")), prefix + ".decoder.conv_out").tanh() + 1) * 0.5).clamp(0, 1)

    def record(value):
        return {"shape": list(value.shape), "values": value.flatten().tolist()}

    return {
        "schema_version": 1, "fixture_id": "sd15-tiny-source-equations-v1",
        "oracle_kind": "independent-pytorch-cpu-source-projection", "production_dependency": False,
        "sources": sources, "generator_sha256": digest(Path(__file__)),
        "inputs": {name: digest(FIXTURE / name) for name in ["model.safetensors", "tokens.json", "initial-noise.safetensors", "normal-sigmas.f64le"]},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "threads": 1, "mkldnn": False},
        "scope": "Pinned CLIP source classes and EPS/model-time/Euler source functions; topology-reduced UNet and VAE source-equation projections with the fixture's declared weighted graph, four-head attention and tanh decoder output. Noise and sigmas are exact input fixtures, not independently generated by this oracle. PNG encoding is checked separately.",
        "tolerance": {"absolute": 2e-6, "relative": 2e-5},
        "conditioning": {name: record(value) for name, value in conditioning.items()},
        "noise": record(noise),
        "denoiser_evaluations": [record(value) for value in evaluations],
        "latents": [record(value) for value in latents],
        "decoded": record(decoded),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    encoded = (json.dumps(generate(), indent=2, sort_keys=True) + "\n").encode()
    if arguments.check:
        if OUTPUT.read_bytes() != encoded:
            raise RuntimeError("Independent tiny diffusion source oracle is stale")
        print("Independent tiny diffusion source oracle is current")
    else:
        OUTPUT.write_bytes(encoded)


if __name__ == "__main__":
    main()
