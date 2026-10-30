#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description="Build the unsigned HitPaw input-admission probe only after a canonical binding receipt exists")
parser.add_argument("--repository", type=Path, required=True)
parser.add_argument("--binding-recipe", type=Path, required=True)
parser.add_argument("--target-dir", type=Path, required=True)
parser.add_argument("--execute", action="store_true")
arguments = parser.parse_args()
repository = arguments.repository.resolve()
source = repository / "crates/sim_test_support/fixtures/providers/hitpaw/component_source"
recipe_bytes = arguments.binding_recipe.read_bytes()
recipe_sha256 = hashlib.sha256(recipe_bytes).hexdigest()
recipe = json.loads(recipe_bytes)
if recipe.get("node_id") != "HitPawGeneralImageEnhance" or recipe.get("feature_id") != "COMFY-NODE-0226" or recipe.get("namespace") != "zed.sim.provider.hitpaw":
    raise SystemExit("wrong canonical image-probe binding receipt")
for field in ("native_contract_sha256", "bindings_sha256"):
    digest = recipe.get(field)
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SystemExit("missing canonical lowercase SHA256 field: " + field)
target = arguments.target_dir.resolve()
if not target.is_relative_to(repository / "target"):
    raise SystemExit("build target must be an isolated ignored repository target")
artifact = repository / "crates/sim_test_support/fixtures/providers/hitpaw/provider_v2_image_input_probe_component"
if artifact.exists():
    raise SystemExit("refuse to overwrite an existing probe artifact")
manifest = source / "Cargo.toml"
source_paths = {name: source / name for name in ("Cargo.toml", "Cargo.lock", "guest.rs", "rebuild_fixture.rs")}
source_paths["build_unsigned_probe.py"] = Path(__file__).resolve()
source_paths["unsigned-fixture-recipe.json"] = Path(__file__).resolve().with_name("unsigned-fixture-recipe.json")
wit_root = repository / "crates/sim_plugin_sdk/wit/provider-v2"
source_paths.update({str(path.relative_to(repository)): path for path in sorted(wit_root.rglob("*.wit"))})
source_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths.items()}

def verify_frozen_inputs():
    if arguments.binding_recipe.read_bytes() != recipe_bytes:
        raise SystemExit("canonical binding recipe changed during the build")
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[name]
           for name, path in source_paths.items()):
        raise SystemExit("component source or WIT input changed during the build")

module = target / "wasm32-unknown-unknown/release/sim_provider_hitpaw_image_input_probe_component.wasm"
commands = [
    ["cargo", "build", "--locked", "--offline", "--manifest-path", str(manifest), "--target", "wasm32-unknown-unknown", "--release", "--lib"],
    ["cargo", "run", "--locked", "--offline", "--manifest-path", str(manifest), "--bin", "rebuild_fixture", "--", str(module), str(target / "probe-build-evidence/unsigned-component.wasm")],
]
if not arguments.execute:
    print(json.dumps({"state": "PREPARED_ONLY", "commands": commands, "artifact": str(artifact)}, indent=2))
    raise SystemExit(0)
environment = os.environ.copy()
environment.update({"CARGO_BUILD_JOBS": "1", "CARGO_INCREMENTAL": "0", "CARGO_TARGET_DIR": str(target),
                    "SIM_HITPAW_NATIVE_CONTRACT_SHA256": recipe["native_contract_sha256"],
                    "SIM_HITPAW_BINDINGS_SHA256": recipe["bindings_sha256"]})
logs = target / "probe-build-evidence"
logs.mkdir(parents=True, exist_ok=False)
results = []
for index, command in enumerate(commands):
    verify_frozen_inputs()
    if index == 1 and not module.read_bytes().startswith(b"\x00asm\x01\x00\x00\x00"):
        raise SystemExit("guest build did not produce a core Wasm v1 module")
    started = time.monotonic()
    with (logs / (str(index) + ".log")).open("wb") as output:
        try:
            result = subprocess.run(command, cwd=repository, env=environment, stdout=output,
                                    stderr=subprocess.STDOUT, timeout=3600)
        except subprocess.TimeoutExpired:
            results.append({"command": command, "state": "OUTER_CARGO_SUPERVISION_CAP",
                            "elapsed_seconds": time.monotonic() - started})
            (logs / "commands.json").write_text(json.dumps(results, indent=2) + "\n")
            raise SystemExit("Cargo gate exceeded its 3600-second supervision cap; external owned process-group supervisor must confirm child closure")
    results.append({"command": command, "exit": result.returncode, "elapsed_seconds": time.monotonic() - started})
    (logs / "commands.json").write_text(json.dumps(results, indent=2) + "\n")
    if result.returncode != 0:
        raise SystemExit(result.returncode)
verify_frozen_inputs()
staged_artifact = logs / "unsigned-component.wasm"
component = staged_artifact.read_bytes()
if not component.startswith(b"\x00asm\x0d\x00\x01\x00"):
    raise SystemExit("validated encoder output is not a component-model v13 artifact")
if artifact.exists():
    raise SystemExit("probe artifact appeared during the build; refusing replacement")
os.link(staged_artifact, artifact)
staged_artifact.unlink()
receipt = {"state": "BUILT_UNSIGNED_NOT_NATIVE_VALIDATED", "component": str(artifact), "component_bytes": len(component),
           "component_sha256": hashlib.sha256(component).hexdigest(),
           "canonical_binding_recipe_sha256": recipe_sha256,
           "canonical_binding_recipe_unchanged_after_build": True,
           "source_files": source_hashes,
           "source_and_WIT_unchanged_after_build": True,
           "artifact_component_header": component[:8].hex(),
           "canonical_validation": "wit-component 0.227.1 ComponentEncoder.validate(true) -> wasmparser Validator.validate_all",
           "host_manifest_preflight": "UNRUN",
           "commands": results, "no_upload_cost_output_acceptance": True}
(logs / "unsigned-component-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
