#!/usr/bin/env python3
import argparse
import ast
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import types

PINNED = {
    "nodes.py": "b8dfdde1de8975be762b085048143cc2dda8fc9202695e460ecc2c8dfe44bc4b",
    "sim/clip_vision.py": "8e3cc5d5d257b52d120885ba7427ff3fdf56129a485fdadac3b6215ae2c67b20",
    "folder_paths.py": "380961714742085363301f980a6c9458f36aca05b1e36ed9b8afa3fb64ccdb50",
    "sim/utils.py": "8b8805ca837e20c922a846854156d10e214654f69df96be90969522f9def2bdb",
}


def definitions(root, relative, names, namespace):
    selected = [node for node in ast.parse((root / relative).read_text()).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise RuntimeError(f"Original definitions missing in {relative}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(root / relative), "exec"), namespace)


class SourceConstructionGate(Exception):
    def __init__(self, configuration):
        super().__init__("original full ClipVisionModel constructor is unbound; no completed source model")
        self.configuration = configuration


def generate():
    import torch
    import safetensors
    torch.set_num_threads(1)
    root = Path(os.environ.get("COMFY_SOURCE_ROOT", "/home/asvegah/repos/sim/projects/sim/ComfyUI"))
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in PINNED}
    if hashes != PINNED:
        raise RuntimeError("Pinned original CLIP vision source changed")
    events = []
    paths = {"os": os, "logging": logging, "folder_names_and_paths": {}}
    definitions(root, "folder_paths.py", ["map_legacy", "get_full_path", "get_full_path_or_raise"], paths)
    utility = {"torch": torch, "safetensors": safetensors, "MMAP_TORCH_FILES": False, "DISABLE_MMAP": False,
               "sim": types.SimpleNamespace(memory_management=types.SimpleNamespace(aimdo_enabled=False))}
    definitions(root, "sim/utils.py", ["load_torch_file", "state_dict_prefix_replace", "transformers_convert"], utility)

    def get_path(category, name):
        events.append({"phase": "source_path_request", "category": category, "name": name})
        result = paths["get_full_path_or_raise"](category, name)
        events.append({"phase": "source_path_resolved", "name": Path(result).name})
        return result

    def load_file(name, safe_load=False):
        events.append({"phase": "source_torch_load", "safe_load_argument": safe_load, "actual_weights_only": True})
        state = utility["load_torch_file"](name, safe_load=safe_load)
        events.append({"phase": "source_state_loaded", "keys": list(state)})
        return state

    def constructor(config):
        if isinstance(config, dict):
            document = config
            label = "source_inline_configuration"
        else:
            file = Path(config)
            label = str(file.relative_to(root))
            hashes[label] = hashlib.sha256(file.read_bytes()).hexdigest()
            document = json.loads(file.read_text())
        events.append({"phase": "original_full_constructor_boundary", "configuration_source": label})
        raise SourceConstructionGate({"source": label, "config": document})

    dino_tree = ast.parse((root / "sim/image_encoders/dino3.py").read_text())
    assignment = next(node for node in dino_tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "DINOV3_VITH_CONFIG" for target in node.targets))
    dino_config = ast.literal_eval(assignment.value)
    hashes["sim/image_encoders/dino3.py"] = hashlib.sha256((root / "sim/image_encoders/dino3.py").read_bytes()).hexdigest()
    vision = {"__file__": str(root / "sim/clip_vision.py"), "os": os, "logging": logging,
              "load_torch_file": load_file, "transformers_convert": utility["transformers_convert"],
              "state_dict_prefix_replace": utility["state_dict_prefix_replace"], "ClipVisionModel": constructor,
              "sim": types.SimpleNamespace(image_encoders=types.SimpleNamespace(dino3=types.SimpleNamespace(DINOV3_VITH_CONFIG=dino_config)))}
    definitions(root, "sim/clip_vision.py", ["convert_to_transformers", "load_clipvision_from_sd", "load"], vision)
    node_globals = {"folder_paths": types.SimpleNamespace(get_full_path_or_raise=get_path, get_filename_list=lambda category: ["l.pt", "nested/h.pt"]),
                    "sim": types.SimpleNamespace(clip_vision=types.SimpleNamespace(load=vision["load"]))}
    definitions(root, "nodes.py", ["CLIPVisionLoader"], node_globals)
    node = node_globals["CLIPVisionLoader"]
    def layer(index):
        return {f"vision_model.encoder.layers.{index}.layer_norm1.weight": torch.zeros(2)}
    def selected_l(position=5, width=2, patch_rank=4):
        return {**layer(22), "vision_model.encoder.layers.0.layer_norm1.weight": torch.zeros(width),
                "vision_model.embeddings.position_embedding.weight": torch.zeros(position, 2),
                "vision_model.embeddings.patch_embedding.weight": torch.zeros(*(([2, 3, 1, 1]) if patch_rank == 4 else [2, 3]))}
    cases_spec = [
        ("CLIP L source selection", "l.pt", selected_l(), None),
        ("CLIP H source selection", "nested/h.pt", layer(30), None),
        ("CLIP G source selection", "g.pt", layer(47), None),
        ("CLIP336 source selection", "336.pt", selected_l(577), None),
        ("CLIP336 LLaVA source selection", "llava.pt", {**selected_l(577), "multi_modal_projector.linear_1.bias": torch.zeros(2)}, None),
        ("SigLIP384 source selection", "siglip384.pt", selected_l(729, 1152), None),
        ("SigLIP512 source selection", "siglip512.pt", selected_l(1024, 1152), None),
        ("SigLIP2 source selection", "siglip2.pt", selected_l(4, 1152, 2), None),
        ("DINO2 giant source selection", "dino2g.pt", {"encoder.layer.39.layer_scale2.lambda1": torch.zeros(2)}, None),
        ("DINO2 large source selection", "dino2l.pt", {"encoder.layer.23.layer_scale2.lambda1": torch.zeros(2)}, None),
        ("DINO3 source selection", "dino3.pt", {"layer.0.mlp.gate_proj.weight": torch.zeros(2), "layer.31.norm1.weight": torch.zeros(2)}, None),
        ("OpenAI key conversion", "openai.pt", {"visual.transformer.resblocks.0.attn.in_proj_weight": torch.arange(12).reshape(6, 2),
          "visual.transformer.resblocks.22.ln_1.weight": torch.zeros(2), "visual.transformer.resblocks.0.ln_1.weight": torch.zeros(2),
          "visual.positional_embedding": torch.zeros(5, 2)}, None),
        ("unrecognized original state", "unknown.pt", {"unrecognized.weight": torch.zeros(2)}, None),
        ("empty original state", "empty.pt", {}, None),
        ("missing original file", "absent.pt", None, None),
        ("corrupt original file", "corrupt.pt", None, b"not a Torch file"),
        ("missing position marker", "missing-position.pt", layer(22), None),
        ("missing first layer marker", "missing-layer.pt", {**layer(22), "vision_model.embeddings.position_embedding.weight": torch.zeros(5, 2)}, None),
        ("unsupported SigLIP positions", "unsupported-position.pt", selected_l(5, 1152), None),
        ("normalized original filename", "./l.pt", None, None),
        ("original absolute prefix normalization", "/l.pt", None, None),
        ("original traversal normalization", "../l.pt", None, None),
        ("F16 configuration preparation", "half.pt", {key: value.half() for key, value in selected_l().items()}, None),
    ]
    cases = []
    with tempfile.TemporaryDirectory(prefix="sim-source-clip-vision-") as temporary:
        directory = Path(temporary)
        paths["folder_names_and_paths"]["clip_vision"] = ([str(directory)], {".pt"})
        for _, name, state, raw in cases_spec:
            if state is not None or raw is not None:
                file = directory / name
                file.parent.mkdir(parents=True, exist_ok=True)
                if raw is not None:
                    file.write_bytes(raw)
                else:
                    torch.save({"state_dict": state}, file)
        for label, name, _, _ in cases_spec:
            events.clear()
            case = {"label": label, "source_name": name}
            try:
                node().load_clip(name)
                raise RuntimeError("Lightweight probe unexpectedly materialized a full model")
            except SourceConstructionGate as gate:
                case["source_preparation_gate"] = {"configuration": gate.configuration, "reason": str(gate)}
            except Exception as error:
                case["source_error"] = {"type": type(error).__name__, "message": str(error).replace(str(directory), "<source-temporary>")}
            case["events"] = list(events)
            cases.append(case)
    return {"format": "sim.source.clip-vision-loader.v1", "claim": "COMFY-NODE-0076", "source_sha256": hashes,
            "schema": {"inputs": node.INPUT_TYPES(), "outputs": node.RETURN_TYPES, "function": node.FUNCTION, "category": node.CATEGORY},
            "torch": torch.__version__, "cases": cases, "execution_scope": "original node/path/Torch load/conversion/config selection; actual full constructor unbound",
            "acceptance_gaps": ["original full constructor and learned artifacts", "permissive missing/unexpected state load and mutation after constructor", "native source config/dtype/device profile parity", "source filename normalization vs canonical strict admission", "UI and OS process recovery"]}


FULL_ORIGINAL_PINNED = {
    "nodes.py": "b8dfdde1de8975be762b085048143cc2dda8fc9202695e460ecc2c8dfe44bc4b",
    "sim/clip_vision.py": "8e3cc5d5d257b52d120885ba7427ff3fdf56129a485fdadac3b6215ae2c67b20",
    "sim/clip_model.py": "08be993d86c3b494b58305fb868638b4b525bbe40abead89e9c94da021716845",
    "sim/clip_vision_config_vitl.json": "faae0c4010113c22118115dcd510f4daf56ada0683262d64d3c646ea67c58e0e",
    "sim/model_patcher.py": "96d21eeaf16d4723355374a5e2b93b35d512ad8f6dc8a1a3a4253cdd71dfd5b0",
    "sim/model_management.py": "c2ca243c80a5262ecafe19feb15cec22d4003c16e523b5376f543f0f75acabaa",
    "sim/memory_management.py": "f624823cbee835506377134cd0e576fdf5c22fa474afab3de6c5ae165f835df5",
    "sim/ops.py": "9d8a4ec8357a9bfcd98dddbf06fcc2a0244643a392aacbe0970d945462c86a42",
    "sim/utils.py": "8b8805ca837e20c922a846854156d10e214654f69df96be90969522f9def2bdb",
    "folder_paths.py": "380961714742085363301f980a6c9458f36aca05b1e36ed9b8afa3fb64ccdb50"
}

def generate_full_original_cpu_f32():
    import gc
    import importlib.metadata
    import shutil
    import sys

    root = Path(os.environ.get("COMFY_SOURCE_ROOT", "/home/asvegah/repos/sim/projects/sim/ComfyUI")).resolve()
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FULL_ORIGINAL_PINNED}
    if hashes != FULL_ORIGINAL_PINNED:
        raise RuntimeError("Pinned full original CLIPVision prerequisites changed")
    expected_tensor_bytes = 1215864832
    if shutil.disk_usage(tempfile.gettempdir()).free < expected_tensor_bytes + 256 * 1024 * 1024:
        raise RuntimeError("Dedicated disk TMPDIR needs space for one full CLIP ViT-L checkpoint and its header")
    sys.path.insert(0, str(root))
    sys.argv = ["original-clip-vision-loader", "--cpu", "--fp32-text-enc", "--disable-dynamic-vram", "--disable-xformers"]
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import sim.options
    sim.options.enable_args_parsing()
    import torch
    import safetensors.torch
    import sim.model_patcher
    import sim.clip_vision as vision
    import nodes
    import folder_paths
    from sim import model_management

    torch.set_num_threads(1)
    for module in (vision, nodes, folder_paths, sim.model_patcher, model_management):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f"Original module is not from pinned source: {module.__name__}")
    if sim.model_patcher.CoreModelPatcher is not sim.model_patcher.ModelPatcher:
        raise RuntimeError("Original CoreModelPatcher binding differs")
    if str(model_management.text_encoder_device()) != "cpu" or model_management.text_encoder_dtype(torch.device("cpu")) != torch.float32:
        raise RuntimeError("The full original probe requires genuine CPU/F32 source policy")
    config_path = root / "sim/clip_vision_config_vitl.json"
    config = json.loads(config_path.read_text())
    template = vision.ClipVisionModel(str(config_path))
    state = {name: torch.zeros_like(tensor, dtype=torch.float32, device="cpu") for name, tensor in template.get_sd().items()}
    manifest = [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "bytes": tensor.numel() * tensor.element_size()} for name, tensor in sorted(state.items())]
    if len(state) != 392 or sum(item["bytes"] for item in manifest) != expected_tensor_bytes:
        raise RuntimeError("The actual complete CLIP ViT-L/14 state differs from source geometry")
    if tuple(state["vision_model.embeddings.position_embedding.weight"].shape) != (257, 1024) or tuple(state["vision_model.encoder.layers.22.layer_norm1.weight"].shape) != (1024,):
        raise RuntimeError("Original complete-state configuration-selection markers differ")
    del template
    gc.collect()
    existing_paths, existing_extensions = folder_paths.folder_names_and_paths["clip_vision"]
    old_paths = (list(existing_paths), set(existing_extensions))
    old_cache = folder_paths.filename_list_cache.get("clip_vision")
    try:
        with tempfile.TemporaryDirectory(prefix="sim-full-original-clip-vision-") as temporary:
            directory = Path(temporary)
            filename = directory / "full-vitl224-zero.safetensors"
            safetensors.torch.save_file(state, str(filename))
            digest = hashlib.sha256()
            with filename.open("rb") as checkpoint:
                for block in iter(lambda: checkpoint.read(1024 * 1024), b""):
                    digest.update(block)
            checkpoint_bytes = filename.stat().st_size
            del state
            gc.collect()
            folder_paths.add_model_folder_path("clip_vision", str(directory), is_default=True)
            source_choices = nodes.CLIPVisionLoader.INPUT_TYPES()["required"]["clip_name"][0]
            if filename.name not in source_choices:
                raise RuntimeError("Actual original COMBO did not discover the full safetensors file")
            if Path(folder_paths.get_full_path_or_raise("clip_vision", filename.name)) != filename:
                raise RuntimeError("Actual original folder transport resolved another file")
            output = nodes.CLIPVisionLoader().load_clip(filename.name)
            if not isinstance(output, tuple) or len(output) != 1:
                raise RuntimeError("Original V1 CLIPVisionLoader did not return its one-slot tuple")
            model = output[0]
            if not isinstance(model, vision.ClipVisionModel) or not isinstance(model.patcher, sim.model_patcher.CoreModelPatcher) or model.config != config:
                raise RuntimeError("Original marker-selected full configuration/wrapper/Patcher differs")
            loaded = model.get_sd()
            loaded_manifest = [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "bytes": tensor.numel() * tensor.element_size()} for name, tensor in sorted(loaded.items())]
            if loaded_manifest != manifest or any(torch.count_nonzero(tensor).item() != 0 for tensor in loaded.values()):
                raise RuntimeError("Actual original node/file path did not retain the complete F32 zero state")
            missing, unexpected = model.load_sd(loaded)
            if missing or unexpected or model.model.training or model.patcher.is_dynamic():
                raise RuntimeError("Actual full state admission/eval/nondynamic source policy differs")
            if model.dtype != torch.float32 or str(model.load_device) != "cpu" or str(model.patcher.offload_device) != "cpu":
                raise RuntimeError("Actual full source device/dtype policy differs")
            result = {
                "format": "sim.source.clip-vision-loader.full-original-cpu-f32.v1",
                "scope": "Genuine full CLIP ViT-L/14 224px CPU/F32 zero-state constructor, actual nodes.CLIPVisionLoader, original folder/file loading and marker-selected built-in configuration; no learned forward",
                "source_sha256": hashes,
                "selected_profile": "CLIP ViT-L/14 224px; no other profile is certified",
                "configuration_source": "sim/clip_vision_config_vitl.json", "configuration": config,
                "packages": {name: importlib.metadata.version(name) for name in ["torch", "torchaudio", "torchvision", "safetensors", "sim-aimdo", "transformers", "einops", "psutil", "tqdm", "av", "torchsde"]},
                "tensor_count": len(manifest), "aggregate_tensor_bytes": sum(item["bytes"] for item in manifest), "manifest": manifest,
                "checkpoint": {"name": filename.name, "format": "safetensors", "bytes": checkpoint_bytes, "sha256": digest.hexdigest(), "sidecar_supplied": False},
                "node_class": "nodes.CLIPVisionLoader", "return_types": list(nodes.CLIPVisionLoader.RETURN_TYPES), "output_type": type(output).__name__, "output_slots": len(output),
                "missing": list(missing), "unexpected": list(unexpected), "dtype": str(model.dtype),
                "load_device": str(model.load_device), "offload_device": str(model.patcher.offload_device),
                "training": model.model.training, "dynamic": model.patcher.is_dynamic(), "model_type": model.model_type,
                "image_size": model.image_size, "image_mean": model.image_mean, "image_std": model.image_std, "return_all_hidden_states": model.return_all_hidden_states,
                "state_all_zero": True, "learned_forward_executed": False,
                "original_file_memory_policy": {"aimdo_enabled": sim.memory_management.aimdo_enabled, "MMAP_TORCH_FILES": sim.utils.MMAP_TORCH_FILES, "DISABLE_MMAP": sim.utils.DISABLE_MMAP},
                "acceptance_gaps": ["learned image numerical execution", "other original CLIP/SigLIP/DINO profiles", "other dtype/device policies", "native automatic built-in config and raw-file parser parity", "source permissive state/filename and mutation observation equivalence", "UI and OS process recovery"],
            }
            del loaded, model, output
            gc.collect()
            return result
    finally:
        folder_paths.folder_names_and_paths["clip_vision"] = old_paths
        if old_cache is None:
            folder_paths.filename_list_cache.pop("clip_vision", None)
        else:
            folder_paths.filename_list_cache["clip_vision"] = old_cache


def run_full_original_subprocess():
    import subprocess
    import sys

    dependencies = Path(os.environ.get("SIM_AUDIO_ORIGINAL_DEPS", "/home/asvegah/repos/sim/target/sim-audio-encoder-original-deps"))
    stage_c_dependencies = Path(os.environ.get("SIM_STAGE_C_ORACLE_DEPS", "/home/asvegah/.codex/worktrees/sim-bfl-provider/sim/target/sim-stage-c-oracle-deps"))
    if not dependencies.is_dir() or not stage_c_dependencies.is_dir():
        raise RuntimeError("Genuine full original dependency overlays must be provisioned")
    environment = dict(os.environ)
    environment.update(PYTHONPATH=os.pathsep.join([str(dependencies), str(stage_c_dependencies)]), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with tempfile.TemporaryDirectory(prefix="sim-original-clip-vision-child-") as temporary:
        result_file = Path(temporary) / "original-result.json"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--full-original-child", str(result_file)], env=environment, check=True)
        return json.loads(result_file.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--full-original", action="store_true", help="Execute genuine full CLIP ViT-L/14 CPU/F32 node/file path; requires provisioned dependencies and a heavy slot")
    parser.add_argument("--full-original-child", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.full_original_child is not None:
        arguments.full_original_child.write_text(json.dumps(generate_full_original_cpu_f32(), indent=2, ensure_ascii=False) + "\n")
        return
    output = Path(__file__).with_name("source-contract.json")
    contract = json.loads(json.dumps(generate(), ensure_ascii=False))
    existing = json.loads(output.read_text())
    original = {key: value for key, value in existing.items() if key != "full_original_cpu_f32"}
    if contract != original:
        raise RuntimeError("Frozen original23 extraction objects/schema/hash/gates changed")
    if arguments.full_original:
        contract["full_original_cpu_f32"] = run_full_original_subprocess()
    elif "full_original_cpu_f32" in existing:
        contract["full_original_cpu_f32"] = existing["full_original_cpu_f32"]
    document = json.dumps(contract, indent=2, ensure_ascii=False) + "\n"
    if arguments.check:
        if output.read_text() != document:
            raise RuntimeError("Immutable source contract differs; do not calibrate from Rust")
    else:
        output.write_text(document)
    cases = contract["cases"]
    if "full_original_cpu_f32" in contract:
        print("full-original CLIP ViT-L CPU/F32 result", "re-executed" if arguments.full_original else "preserved, NOT re-executed by lightweight --check")
    if arguments.full_original:
        full = contract["full_original_cpu_f32"]
        print("genuine full-original node", full["node_class"], full["tensor_count"], "complete F32 zero tensors;", full["aggregate_tensor_bytes"], "tensor bytes; one completed original model; no learned forward")
    print("lightweight original extraction", len(cases), "cases;", sum("source_preparation_gate" in case for case in cases), "full-constructor gates;",
          sum("source_error" in case for case in cases), "original errors; zero completed source models")


if __name__ == "__main__":
    main()
