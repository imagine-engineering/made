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
    "sim_extras/nodes_audio_encoder.py": "fbc6f4d8ca0e099dc2f35f9420dce9aced3763325b9804b031a1641db2bb4a8a",
    "sim/audio_encoders/audio_encoders.py": "c8d3260799ea0222b6bf9e1bde8d16f105a46aaf1944213bf66fdaa05433dec8",
    "sim/audio_encoders/wav2vec2.py": "32494297021e54e42845276255a026ce8ab62be5d54f6d40cecfb042c8b238e7",
    "sim/audio_encoders/whisper.py": "f0e214e79fdfa9926fbc863038fe3e0455caa61f486a1b37a3039ca7253dea22",
}


def source_root():
    candidate = Path(os.environ.get("COMFY_SOURCE_ROOT", "/home/asvegah/repos/sim/projects/sim/ComfyUI"))
    if not candidate.is_dir():
        raise RuntimeError("COMFY_SOURCE_ROOT must identify the pinned original source checkout")
    return candidate


def execute_definitions(root, relative, names, namespace):
    tree = ast.parse((root / relative).read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise RuntimeError(f"Original definitions missing in {relative}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(root / relative), "exec"), namespace)


class SourceMaterializationGate(Exception):
    def __init__(self, configuration):
        super().__init__("actual full original AudioEncoderModel construction is unbound in the lightweight oracle")
        self.configuration = configuration


class SchemaRecorder:
    def __init__(self, **fields):
        self.fields = fields


class InputRecorder:
    @staticmethod
    def Input(name, **fields):
        return {"name": name, **fields}


class OutputRecorder:
    @staticmethod
    def Output():
        return {"type": "AUDIO_ENCODER"}


def generate():
    import torch
    import safetensors
    torch.set_num_threads(1)
    root = source_root()
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in [*PINNED, "folder_paths.py", "sim/utils.py"]}
    for name, expected in PINNED.items():
        if hashes[name] != expected:
            raise RuntimeError(f"Pinned source changed: {name}")
    events = []
    io = types.SimpleNamespace(ComfyNode=object, NodeOutput=object, Schema=SchemaRecorder, Combo=InputRecorder, AudioEncoder=OutputRecorder)
    paths = {"os": os, "logging": logging, "folder_names_and_paths": {}}
    execute_definitions(root, "folder_paths.py", ["map_legacy", "get_full_path", "get_full_path_or_raise"], paths)
    utility = {"torch": torch, "safetensors": safetensors, "MMAP_TORCH_FILES": False, "DISABLE_MMAP": False,
               "sim": types.SimpleNamespace(memory_management=types.SimpleNamespace(aimdo_enabled=False))}
    execute_definitions(root, "sim/utils.py", ["load_torch_file", "state_dict_prefix_replace"], utility)

    def model_factory(configuration):
        events.append({"phase": "original_model_constructor_boundary", "configuration": configuration.copy()})
        raise SourceMaterializationGate(configuration.copy())

    encoder_globals = {"sim": types.SimpleNamespace(utils=types.SimpleNamespace(state_dict_prefix_replace=utility["state_dict_prefix_replace"])),
                       "AudioEncoderModel": model_factory, "logging": logging}
    execute_definitions(root, "sim/audio_encoders/audio_encoders.py", ["load_audio_encoder_from_sd"], encoder_globals)

    def get_path(category, name):
        events.append({"phase": "source_path_request", "category": category, "name": name})
        result = paths["get_full_path_or_raise"](category, name)
        events.append({"phase": "source_path_resolved", "name": Path(result).name})
        return result

    def load_file(name, safe_load=False):
        events.append({"phase": "source_torch_load", "safe_load": safe_load})
        result = utility["load_torch_file"](name, safe_load=safe_load)
        events.append({"phase": "source_state_loaded", "keys": list(result)})
        return result

    def load_encoder(state):
        events.append({"phase": "original_architecture_selection", "keys": list(state)})
        return encoder_globals["load_audio_encoder_from_sd"](state)

    namespace = {"io": io, "folder_paths": types.SimpleNamespace(get_full_path_or_raise=get_path, get_filename_list=lambda category: ["base.pt", "nested/large.pt"]),
                 "sim": types.SimpleNamespace(utils=types.SimpleNamespace(load_torch_file=load_file), audio_encoders=types.SimpleNamespace(audio_encoders=types.SimpleNamespace(load_audio_encoder_from_sd=load_encoder)))}
    execute_definitions(root, "sim_extras/nodes_audio_encoder.py", ["AudioEncoderLoader"], namespace)
    node = namespace["AudioEncoderLoader"]
    schema = node.define_schema().fields
    definitions = [
        ("base configuration", "base.pt", {"encoder.layer_norm.bias": torch.zeros(768)}, None),
        ("large prefixed configuration", "nested/large.pt", {"wav2vec2.encoder.layer_norm.bias": torch.zeros(1024)}, None),
        ("whisper configuration", "whisper.pt", {"model.encoder.embed_positions.weight": torch.zeros(2, 4)}, None),
        ("unsupported embedding width", "bad-width.pt", {"encoder.layer_norm.bias": torch.zeros(512)}, None),
        ("wrong marker rank", "bad-rank.pt", {"encoder.layer_norm.bias": torch.zeros(1, 768)}, None),
        ("unrecognized state", "unknown.pt", {"unrecognized.weight": torch.ones(2)}, None),
        ("empty state", "empty.pt", {}, None),
        ("missing file", "absent.pt", None, None),
        ("corrupt safe torch file", "corrupt.pt", None, b"not a Torch checkpoint"),
        ("normalized source filename", "./base.pt", None, None),
        ("original absolute prefix normalization", "/base.pt", None, None),
        ("original traversal normalization", "../base.pt", None, None),
        ("missing nested file", "nested/absent.pt", None, None),
        ("provided F16 marker configuration", "half.pt", {"encoder.layer_norm.bias": torch.zeros(768, dtype=torch.float16)}, None),
        ("prefix overwrite source constructor", "collision.pt", {"wav2vec2.encoder.layer_norm.bias": torch.zeros(768), "encoder.layer_norm.bias": torch.zeros(512)}, None),
    ]
    cases = []
    with tempfile.TemporaryDirectory(prefix="sim-source-audio-loader-") as temporary:
        output = Path(temporary)
        paths["folder_names_and_paths"]["audio_encoders"] = ([str(output)], {".pt"})
        for label, name, state, raw in definitions:
            if state is not None or raw is not None:
                filename = output / name
                filename.parent.mkdir(parents=True, exist_ok=True)
                if raw is not None:
                    filename.write_bytes(raw)
                else:
                    torch.save({"state_dict": state}, filename)
        for label, name, state, raw in definitions:
            events.clear()
            case = {"label": label, "source_name": name}
            try:
                node.execute(name)
                raise RuntimeError("Lightweight source probe unexpectedly materialized a full model")
            except SourceMaterializationGate as gate:
                case["source_preparation_gate"] = {"configuration": gate.configuration, "reason": str(gate)}
            except Exception as error:
                case["source_error"] = {"type": type(error).__name__, "message": str(error).replace(str(output), "<source-temporary>")}
            case["events"] = list(events)
            cases.append(case)
    return {"format": "sim.source.audio-encoder-loader.v1", "claim": "COMFY-NODE-0012", "source_sha256": hashes,
            "schema": schema, "torch": torch.__version__, "execution_scope": "original node/path/safe load/architecture selection; original full constructor unbound; no completed source model",
            "source_memory_policy": {"aimdo_enabled": False, "MMAP_TORCH_FILES": False, "DISABLE_MMAP": False},
            "cases": cases, "acceptance_gaps": ["actual full source model construction and learned artifact numerical proof", "source permissive missing/extra state vs strict canonical admission", "source filename normalization vs canonical strict request policy", "production dtype/device policy", "UI/OS process recovery"]}

FULL_ORIGINAL_PINNED = {
    "sim_extras/nodes_audio_encoder.py": "fbc6f4d8ca0e099dc2f35f9420dce9aced3763325b9804b031a1641db2bb4a8a",
    "sim/audio_encoders/audio_encoders.py": "c8d3260799ea0222b6bf9e1bde8d16f105a46aaf1944213bf66fdaa05433dec8",
    "sim/audio_encoders/wav2vec2.py": "32494297021e54e42845276255a026ce8ab62be5d54f6d40cecfb042c8b238e7",
    "sim/audio_encoders/whisper.py": "f0e214e79fdfa9926fbc863038fe3e0455caa61f486a1b37a3039ca7253dea22",
    "folder_paths.py": "380961714742085363301f980a6c9458f36aca05b1e36ed9b8afa3fb64ccdb50",
    "sim/utils.py": "8b8805ca837e20c922a846854156d10e214654f69df96be90969522f9def2bdb",
    "sim/model_patcher.py": "96d21eeaf16d4723355374a5e2b93b35d512ad8f6dc8a1a3a4253cdd71dfd5b0",
    "sim/model_management.py": "c2ca243c80a5262ecafe19feb15cec22d4003c16e523b5376f543f0f75acabaa",
    "sim/memory_management.py": "f624823cbee835506377134cd0e576fdf5c22fa474afab3de6c5ae165f835df5",
    "sim/ops.py": "9d8a4ec8357a9bfcd98dddbf06fcc2a0244643a392aacbe0970d945462c86a42",
    "sim_api/latest/__init__.py": "5d2c1f3e37ecac36d9a4075a644d43ab211d0bbeab4c86698b6c987523ca072d",
    "sim_api/latest/_io.py": "ac986092cace7600461b8547e21f770126958d8c15eef59d79d6709cca4b7717",
    "sim_api/latest/_io_public.py": "c805d754d61aba1ea72ed2d9d5c371de35be9affa5305d35426fb298fcb7c046"
}


def generate_full_original_cpu_f32():
    import gc
    import importlib.metadata
    import sys

    root = source_root().resolve()
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FULL_ORIGINAL_PINNED}
    if hashes != FULL_ORIGINAL_PINNED:
        raise RuntimeError("Actual original full-loader prerequisites differ from the pinned source")
    sys.path.insert(0, str(root))
    sys.argv = ["original-audio-loader", "--cpu", "--fp32-text-enc", "--disable-dynamic-vram", "--disable-xformers"]
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import sim.options
    sim.options.enable_args_parsing()
    import torch
    import safetensors.torch
    import sim.model_patcher
    import sim.audio_encoders.audio_encoders as audio
    import sim_extras.nodes_audio_encoder as nodes
    import folder_paths
    from sim_api.latest import io
    from sim import model_management

    torch.set_num_threads(1)
    for module in (audio, nodes, folder_paths, sim.model_patcher, model_management, io):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f"Original module is not from the pinned source: {module.__name__}")
    if sim.model_patcher.CoreModelPatcher is not sim.model_patcher.ModelPatcher:
        raise RuntimeError("Original CoreModelPatcher binding differs")
    if str(model_management.text_encoder_device()) != "cpu" or model_management.text_encoder_dtype(torch.device("cpu")) != torch.float32:
        raise RuntimeError("The full original loader probe requires genuine CPU/F32 source policy")

    tree = ast.parse((root / "sim/audio_encoders/audio_encoders.py").read_text())
    configs = []
    for assignment in ast.walk(tree):
        if isinstance(assignment, ast.Assign) and isinstance(assignment.value, ast.Dict):
            try:
                configuration = ast.literal_eval(assignment.value)
            except (ValueError, TypeError):
                continue
            if configuration.get("model_type") == "wav2vec2" and configuration.get("embed_dim") == 768:
                configs.append(configuration)
    if len(configs) != 1:
        raise RuntimeError("Expected exactly the original Wav2Vec2Base configuration")
    template = audio.AudioEncoderModel(dict(configs[0]))
    state = {name: torch.zeros_like(tensor, dtype=torch.float32, device="cpu") for name, tensor in template.get_sd().items()}
    manifest = [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "bytes": tensor.numel() * tensor.element_size()} for name, tensor in sorted(state.items())]
    if len(state) != 211 or sum(entry["bytes"] for entry in manifest) != 377486848 or tuple(state["masked_spec_embed"].shape) != (768,):
        raise RuntimeError("The genuine complete original Wav2Vec2Base state changed")
    del template
    gc.collect()

    existing_paths, existing_extensions = folder_paths.folder_names_and_paths["audio_encoders"]
    old_paths = (list(existing_paths), set(existing_extensions))
    old_cache = folder_paths.filename_list_cache.get("audio_encoders")
    try:
        with tempfile.TemporaryDirectory(prefix="sim-full-original-audio-loader-") as temporary:
            directory = Path(temporary)
            filename = directory / "full-base-zero.safetensors"
            safetensors.torch.save_file(state, str(filename))
            digest = hashlib.sha256()
            with filename.open("rb") as checkpoint:
                for block in iter(lambda: checkpoint.read(1024 * 1024), b""):
                    digest.update(block)
            checkpoint_bytes = filename.stat().st_size
            del state
            gc.collect()
            folder_paths.add_model_folder_path("audio_encoders", str(directory), is_default=True)
            schema = nodes.AudioEncoderLoader.define_schema()
            source_input = next(item for item in schema.inputs if item.id == "audio_encoder_name")
            if filename.name not in source_input.options:
                raise RuntimeError("Actual source COMBO did not discover the real safetensors file")
            if Path(folder_paths.get_full_path_or_raise("audio_encoders", filename.name)) != filename:
                raise RuntimeError("Actual original folder transport resolved a different file")
            output = nodes.AudioEncoderLoader.execute(filename.name)
            if not isinstance(output, io.NodeOutput) or output.result is None or len(output.result) != 1:
                raise RuntimeError("Actual AudioEncoderLoader did not return its real one-slot NodeOutput")
            model = output.result[0]
            if not isinstance(model, audio.AudioEncoderModel) or not isinstance(model.patcher, sim.model_patcher.CoreModelPatcher):
                raise RuntimeError("Actual original wrapper/patcher output differs")
            loaded = model.get_sd()
            loaded_manifest = [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "bytes": tensor.numel() * tensor.element_size()} for name, tensor in sorted(loaded.items())]
            if loaded_manifest != manifest or any(torch.count_nonzero(tensor).item() != 0 for tensor in loaded.values()):
                raise RuntimeError("Original file/node path did not retain the complete F32 zero state")
            missing, unexpected = model.load_sd(loaded)
            if missing or unexpected or model.model.training or model.patcher.is_dynamic():
                raise RuntimeError("Actual original complete state admission/eval/nondynamic policy differs")
            if model.dtype != torch.float32 or str(model.load_device) != "cpu" or str(model.patcher.offload_device) != "cpu" or model.model_sample_rate != 16000:
                raise RuntimeError("Actual original model device/dtype/sample-rate differs")
            result = {
                "format": "sim.source.audio-encoder-loader.full-original-cpu-f32.v1",
                "scope": "Genuine pinned module imports, complete original zero-state constructor, real safetensors file/folder transport, actual AudioEncoderLoader and io.NodeOutput; no learned forward",
                "source_sha256": hashes,
                "source_configuration": configs[0],
                "packages": {name: importlib.metadata.version(name) for name in ["torch", "torchaudio", "torchvision", "safetensors", "sim-aimdo", "transformers", "einops", "psutil", "tqdm", "av"]},
                "tensor_count": len(manifest), "aggregate_tensor_bytes": sum(item["bytes"] for item in manifest),
                "manifest": manifest,
                "checkpoint": {"name": filename.name, "format": "safetensors", "bytes": checkpoint_bytes, "sha256": digest.hexdigest()},
                "node_id": schema.node_id, "node_output_type": type(output).__name__, "node_output_slots": len(output.result),
                "missing": list(missing), "unexpected": list(unexpected),
                "dtype": str(model.dtype), "load_device": str(model.load_device), "offload_device": str(model.patcher.offload_device),
                "training": model.model.training, "dynamic": model.patcher.is_dynamic(), "model_sample_rate": model.model_sample_rate,
                "state_all_zero": True, "learned_forward_executed": False,
                "original_file_memory_policy": {"aimdo_enabled": sim.memory_management.aimdo_enabled, "MMAP_TORCH_FILES": sim.utils.MMAP_TORCH_FILES, "DISABLE_MMAP": sim.utils.DISABLE_MMAP},
                "acceptance_gaps": ["learned waveform numerical execution", "other architectures and device/dtype policies", "native raw-file parser equivalence", "source permissive state and filename validation equivalence", "UI and OS process recovery"],
            }
            del loaded, model, output
            gc.collect()
            return result
    finally:
        folder_paths.folder_names_and_paths["audio_encoders"] = old_paths
        if old_cache is None:
            folder_paths.filename_list_cache.pop("audio_encoders", None)
        else:
            folder_paths.filename_list_cache["audio_encoders"] = old_cache


def run_full_original_subprocess():
    import subprocess
    import sys

    dependencies = Path(os.environ.get("SIM_AUDIO_ORIGINAL_DEPS", "/home/asvegah/repos/sim/target/sim-audio-encoder-original-deps"))
    stage_c_dependencies = Path(os.environ.get("SIM_STAGE_C_ORACLE_DEPS", "/home/asvegah/.codex/worktrees/sim-bfl-provider/sim/target/sim-stage-c-oracle-deps"))
    if not dependencies.is_dir() or not stage_c_dependencies.is_dir():
        raise RuntimeError("Genuine original audio and torchvision dependency overlays must be provisioned")
    environment = dict(os.environ)
    environment.update(PYTHONPATH=os.pathsep.join([str(dependencies), str(stage_c_dependencies)]), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with tempfile.TemporaryDirectory(prefix="sim-original-audio-child-") as temporary:
        result_file = Path(temporary) / "original-result.json"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--full-original-child", str(result_file)], env=environment, check=True)
        return json.loads(result_file.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--full-original", action="store_true", help="Run the separately labeled real full CPU/F32 module/node/file path; requires genuine source dependencies and a heavy slot")
    parser.add_argument("--full-original-child", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.full_original_child is not None:
        arguments.full_original_child.write_text(json.dumps(generate_full_original_cpu_f32(), indent=2, ensure_ascii=False) + "\n")
        return
    output = Path(__file__).with_name("source-contract.json")
    contract = generate()
    existing = json.loads(output.read_text())
    original = {key: value for key, value in existing.items() if key != "full_original_cpu_f32"}
    if contract != original:
        raise RuntimeError("Frozen original 15-case extraction evidence changed")
    if arguments.full_original:
        contract["full_original_cpu_f32"] = run_full_original_subprocess()
    elif "full_original_cpu_f32" in existing:
        contract["full_original_cpu_f32"] = existing["full_original_cpu_f32"]
    document = json.dumps(contract, indent=2, ensure_ascii=False) + "\n"
    if arguments.check:
        if output.read_text() != document:
            raise RuntimeError("Immutable source contract differs; do not calibrate from Rust outputs")
    else:
        output.write_text(document)
    contract = json.loads(document)
    if "full_original_cpu_f32" in contract:
        print("full-original CPU/F32 result", "re-executed" if arguments.full_original else "preserved, NOT re-executed by lightweight --check")
    if arguments.full_original:
        full = contract["full_original_cpu_f32"]
        print("genuine full-original node", full["node_id"], full["tensor_count"], "complete F32 zero tensors;", full["aggregate_tensor_bytes"], "tensor bytes; one completed original model; no learned forward")
    print("lightweight original extraction", len(contract["cases"]), "cases;", sum("source_preparation_gate" in case for case in contract["cases"]), "explicit full-constructor gates;", sum("source_error" in case for case in contract["cases"]), "source errors; zero completed original models")


if __name__ == "__main__":
    main()
