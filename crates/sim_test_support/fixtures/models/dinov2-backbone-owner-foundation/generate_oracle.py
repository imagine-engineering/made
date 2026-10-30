#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import platform
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
DA3_FIXTURE = HERE.parent / "depth-anything-3-resource-foundation"
SOURCES = {
    "projects/sim/ComfyUI/sim/image_encoders/dino2.py": "1dec8c1d6104c268e593cea20302d925f637266edce2a6e4dfa142af8a00d579",
    "projects/sim/ComfyUI/sim/text_encoders/bert.py": "3f1f32353da95790285a10f452959a871aa949aab15a89b646a95abc6165955c",
    "projects/sim/ComfyUI/sim/ldm/modules/attention.py": "436e1d91f8d5d84c5667e051cdf3ab2f91d8db25b66d88a084c89a202de0579e",
    "projects/sim/ComfyUI/sim/ldm/depth_anything_3/reference_view_selector.py": "24e9428a820b5287d622bc865d4fd6520486294c4337a28de71fca6ec62e0c29",
}
MASK_TOKEN_KEY = "native.backbone.embeddings.mask_token"
MASK_TOKEN_VALUES = [-0.035, -0.0025, 0.03, -0.01]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits(values):
    return [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]


def ordinary_state(graph, source_dtype, mutation=None):
    state_mutation = mutation if mutation is not None and mutation[0] != MASK_TOKEN_KEY else None
    state = graph.make_state("dpt", mutation=state_mutation, source_dtype=source_dtype)
    values = [graph.project_storage(graph.f32(value), source_dtype) for value in MASK_TOKEN_VALUES]
    if mutation is not None and mutation[0] == MASK_TOKEN_KEY:
        values[mutation[1]] = graph.project_storage(
            graph.fadd(values[mutation[1]], mutation[2]), source_dtype
        )
    state[MASK_TOKEN_KEY] = graph.Tensor((1, 4), values)
    return state


def storage_bytes(values, source_dtype):
    if source_dtype == "f16":
        return b"".join(struct.pack("<e", value) for value in values)
    if source_dtype == "bf16":
        return b"".join(
            ((struct.unpack("<I", struct.pack("<f", value))[0] >> 16) & 0xFFFF).to_bytes(2, "little")
            for value in values
        )
    return b"".join(struct.pack("<f", value) for value in values)


def mask_token_identity(source_dtype, values):
    source = storage_bytes(values, source_dtype)
    projected = b"".join(struct.pack("<f", value) for value in values)
    digest = hashlib.sha256()
    digest.update(b"dinov2-forward-unused-mask-token-v1")
    encoded_key = MASK_TOKEN_KEY.encode()
    digest.update(len(encoded_key).to_bytes(8, "little"))
    digest.update(encoded_key)
    encoded_dtype = source_dtype.encode()
    digest.update(len(encoded_dtype).to_bytes(8, "little"))
    digest.update(encoded_dtype)
    digest.update((2).to_bytes(8, "little"))
    digest.update((1).to_bytes(8, "little"))
    digest.update((4).to_bytes(8, "little"))
    digest.update(len(source).to_bytes(8, "little"))
    digest.update(source)
    digest.update(len(projected).to_bytes(8, "little"))
    digest.update(projected)
    return {
        "identity_sha256": digest.hexdigest(),
        "projected_bits": bits(values),
        "source_bytes": list(source),
        "storage_bytes": len(source),
    }


def load_source_graph():
    path = DA3_FIXTURE / "source_graph.py"
    specification = importlib.util.spec_from_file_location("da3_source_graph", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module, path


def ordinary_block(graph, state, values, layer):
    prefix = f"native.backbone.encoder.layer.{layer}"
    normalized = graph.norm_state(state, values, prefix + ".norm1", graph.f32(1.0e-6))
    query = graph.linear_state(state, normalized, prefix + ".attention.attention.query")
    key = graph.linear_state(state, normalized, prefix + ".attention.attention.key")
    value = graph.linear_state(state, normalized, prefix + ".attention.attention.value")
    attended = graph.attention(query, key, value, values.shape[0], values.shape[1], 1, 4)
    attended = graph.linear_state(state, attended, prefix + ".attention.output.dense")
    residual = graph.add(values, graph.multiply_channels(attended, state[prefix + ".layer_scale1.lambda1"]))
    normalized = graph.norm_state(state, residual, prefix + ".norm2", graph.f32(1.0e-6))
    hidden = graph.gelu(graph.linear_state(state, normalized, prefix + ".mlp.fc1"))
    hidden = graph.linear_state(state, hidden, prefix + ".mlp.fc2")
    return graph.add(residual, graph.multiply_channels(hidden, state[prefix + ".layer_scale2.lambda1"]))


def ordinary_backbone(graph, state, image):
    patch = graph.convolution(
        state,
        image,
        "native.backbone.embeddings.patch_embeddings.projection",
        2,
        0,
    )
    batch, hidden, patch_height, patch_width = patch.shape
    patches = patch_height * patch_width
    positions = graph.interpolate_positions(
        state["native.backbone.embeddings.position_embeddings"],
        patch_height,
        patch_width,
    )
    cls = state["native.backbone.embeddings.cls_token"]
    values = graph.Tensor((batch, patches + 1, hidden), [0.0] * (batch * (patches + 1) * hidden))
    for batch_index in range(batch):
        for channel in range(hidden):
            values.set(
                graph.fadd(cls.get(0, 0, channel), positions.get(0, 0, channel)),
                batch_index,
                0,
                channel,
            )
        for y in range(patch_height):
            for x in range(patch_width):
                token = y * patch_width + x + 1
                for channel in range(hidden):
                    values.set(
                        graph.fadd(
                            patch.get(batch_index, channel, y, x),
                            positions.get(0, token, channel),
                        ),
                        batch_index,
                        token,
                        channel,
                    )
    outputs = []
    for layer in range(4):
        values = ordinary_block(graph, state, values, layer)
        normalized = graph.final_norm(state, values, hidden)
        cls_values = []
        patch_values = []
        for batch_index in range(batch):
            cls_values.extend(normalized.values[(batch_index * (patches + 1)) * hidden:(batch_index * (patches + 1) + 1) * hidden])
            patch_values.extend(normalized.values[(batch_index * (patches + 1) + 1) * hidden:(batch_index + 1) * (patches + 1) * hidden])
        outputs.append(
            {
                "layer": layer,
                "patch_shape": [batch, patches, hidden],
                "patch_bits": bits(patch_values),
                "class_shape": [batch, hidden],
                "class_bits": bits(cls_values),
            }
        )
    return outputs


def document(generator_sha256):
    for relative, expected in SOURCES.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"source hash drift for {relative}: {actual}")
    graph, graph_path = load_source_graph()
    da3_oracle_path = DA3_FIXTURE / "oracle.json"
    da3_oracle = json.loads(da3_oracle_path.read_text())
    input_values = [struct.unpack("<f", struct.pack("<I", value))[0] for value in da3_oracle["input_bits"]]
    image = graph.preprocess(input_values)
    routes = {}
    mask_token = {}
    for dtype in ["f32", "f16", "bf16"]:
        state = ordinary_state(graph, dtype)
        routes[dtype] = ordinary_backbone(graph, state, image)
        mask_token[dtype] = mask_token_identity(dtype, state[MASK_TOKEN_KEY].values)
    mutations = {}
    for name, mutation in {
        "patch_embedding": ("native.backbone.embeddings.patch_embeddings.projection.bias", 0, graph.f32(0.125)),
        "position_interpolation": ("native.backbone.embeddings.position_embeddings", 7, graph.f32(0.125)),
        "attention_qkv": ("native.backbone.encoder.layer.1.attention.attention.query.bias", 0, graph.f32(0.125)),
        "attention_projection": ("native.backbone.encoder.layer.1.attention.output.dense.bias", 0, graph.f32(0.125)),
        "layer_scale": ("native.backbone.encoder.layer.1.layer_scale1.lambda1", 0, graph.f32(0.125)),
        "mlp": ("native.backbone.encoder.layer.1.mlp.fc1.bias", 0, graph.f32(0.125)),
        "normalization": ("native.backbone.layernorm.bias", 0, graph.f32(0.125)),
        "forward_unused_mask_token": (MASK_TOKEN_KEY, 1, graph.f32(0.125)),
    }.items():
        state = ordinary_state(graph, "f32", mutation=mutation)
        outputs = ordinary_backbone(graph, state, image)
        mutations[name] = {
            "state_key": mutation[0],
            "lane": mutation[1],
            "delta_bits": bits([mutation[2]])[0],
            "changes_output": name != "forward_unused_mask_token",
            "outputs": outputs,
        }
        if name == "forward_unused_mask_token":
            mutations[name]["mask_token"] = mask_token_identity("f32", state[MASK_TOKEN_KEY].values)
    return {
        "format": "dinov2-backbone-owner-foundation-v1",
        "generator_command": "PYTHONDONTWRITEBYTECODE=1 python3 crates/sim_test_support/fixtures/models/dinov2-backbone-owner-foundation/generate_oracle.py --check",
        "generator_sha256": generator_sha256,
        "pinned_sources": SOURCES,
        "source_graph_sha256": sha256(graph_path),
        "da3_oracle_sha256": sha256(da3_oracle_path),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "input_shape": da3_oracle["input_shape"],
        "input_bits": da3_oracle["input_bits"],
        "ordinary_routes": routes,
        "ordinary_mutations": mutations,
        "forward_unused_mask_token": mask_token,
        "da3_reference_fixture": da3_oracle["reference_fixture"],
        "storage_projection": da3_oracle["storage_projection"],
        "owner_contract": {
            "ordinary_route": "forward/get_intermediate_layers",
            "da3_route": "get_intermediate_layers_da3",
            "forward_unused_state": "ordinary/MoGe retains embeddings.mask_token for strict loading while DA3 rejects it; execution never reads it",
            "swiglu": "weights_in -> split -> SiLU(first) * second -> weights_out",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    generator_sha256 = sha256(Path(__file__))
    oracle = json.dumps(document(generator_sha256), indent=2, sort_keys=True) + "\n"
    oracle_sha256 = hashlib.sha256(oracle.encode()).hexdigest()
    manifest = json.dumps(
        {
            "da3_oracle_sha256": sha256(DA3_FIXTURE / "oracle.json"),
            "format": "dinov2-backbone-owner-foundation-v1",
            "future_consumers": ["sim_model::moge"],
            "generator_sha256": generator_sha256,
            "ordinary_route": "forward/get_intermediate_layers",
            "da3_route": "get_intermediate_layers_da3",
            "oracle_sha256": oracle_sha256,
            "owner": "sim_model::dino2::NativeDino2Backbone",
            "production_python": False,
            "source_graph_sha256": sha256(DA3_FIXTURE / "source_graph.py"),
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
    manifest_sha256 = hashlib.sha256(manifest.encode()).hexdigest()
    provenance = json.dumps(
        {
            "cross_check": "The unchanged DA3 oracle, reference fixture, storage projection, alias-residency tests, and lifecycle tests remain authoritative for the DA3 adapter.",
            "da3_oracle_sha256": sha256(DA3_FIXTURE / "oracle.json"),
            "generator_sha256": generator_sha256,
            "independence": "The generator executes scalar source equations and never invokes production Rust.",
            "manifest_sha256": manifest_sha256,
            "oracle_kind": "pure-standard-library source-equation translation",
            "oracle_sha256": oracle_sha256,
            "platform": platform.platform(),
            "production_python": False,
            "python": platform.python_version(),
            "source_graph_sha256": sha256(DA3_FIXTURE / "source_graph.py"),
            "source_profile": "pinned ComfyUI DINOv2 ordinary and DA3 routes",
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
    outputs = {
        HERE / "oracle.json": oracle,
        HERE / "manifest.json": manifest,
        HERE / "provenance.json": provenance,
    }
    if arguments.check:
        stale = [str(path.name) for path, expected in outputs.items() if not path.exists() or path.read_text() != expected]
        if stale:
            raise SystemExit(f"DINOv2 fixture is stale ({', '.join(stale)}); run generate_oracle.py")
        return
    for path, encoded in outputs.items():
        path.write_text(encoded)


if __name__ == "__main__":
    main()
