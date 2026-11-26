#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SPECIFICATION_ROOT = Path(__file__).resolve().parent
LEDGER_PATH = SPECIFICATION_ROOT / "catalogs" / "native-backend-dependencies.json"
TASK_ID = "comfy-parity-vendor-dependency-lock"
CERTIFICATION_TASK_ID = "comfy-parity-certify-device-apple-metal-mps-comfy-model-0015"
CERTIFICATION_WRITES = {
    "Cargo.lock",
    "crates/sim_runtime/Cargo.toml",
    "crates/sim_test_support/Cargo.toml",
}
AUTHORIZED_INTEGRATION_WRITES = {
    "comfy-parity-provider-worker-stream-bridge": {
        "Cargo.lock",
        "Cargo.toml",
        "crates/sim_media/Cargo.toml",
        "crates/sim_plugin_host/Cargo.toml",
        "crates/sim_plugin_host/provider_components/Cargo.toml",
        "crates/sim_test_support/Cargo.toml",
    },
    "comfy-parity-provider-deployment-lifecycle": {
        "Cargo.lock",
        "crates/sim_api/Cargo.toml",
        "crates/extension_host/Cargo.toml",
        "crates/zed/Cargo.toml",
    },
    "comfy-parity-native-node-runtime-foundation": {
        "Cargo.lock",
        "crates/sim_nodes/Cargo.toml",
    },
    "comfy-parity-native-node-compute-value-foundation": {
        "Cargo.lock",
        "crates/sim_nodes/Cargo.toml",
        "crates/sim_media/Cargo.toml",
        "crates/sim_plugin_host/Cargo.toml",
        "crates/sim_plugin_sdk/Cargo.toml",
    },
    "comfy-parity-native-node-asset-effect-foundation": {
        "crates/sim_media/Cargo.toml",
    },
    "comfy-parity-native-node-provider-invocation-foundation": {
        "Cargo.lock",
        "crates/sim_plugin_host/Cargo.toml",
        "crates/sim_worker/Cargo.toml",
    },
    "comfy-parity-opt-in-product-build-boundary": {
        "Cargo.lock",
        "crates/extension_host/Cargo.toml",
        "crates/zed/Cargo.toml",
    },
    "comfy-parity-native-text-value-regex-foundation": {
        "Cargo.lock",
        "crates/sim_nodes/Cargo.toml",
    },
    "comfy-parity-provider-hermetic-component-harness": {
        "Cargo.lock",
        "crates/sim_plugin_host/Cargo.toml",
        "crates/sim_plugin_host/provider_components/Cargo.toml",
    },
    "comfy-parity-native-media-text-rendering-foundation": {
        "Cargo.toml",
        "Cargo.lock",
        "crates/sim_media/Cargo.toml",
    },
    "comfy-parity-native-shader-execution-foundation": {
        "Cargo.toml",
        "Cargo.lock",
        "crates/sim_tensor/Cargo.toml",
    },
    "comfy-parity-native-latent-bundle-foundation": {
        "Cargo.toml",
    },
    "comfy-parity-native-qwen2-tokenizer-foundation": {
        "Cargo.toml",
        "Cargo.lock",
        "crates/sim_model/Cargo.toml",
    },
    "comfy-parity-native-sdpose-heatmap-projection-foundation": {
        "Cargo.lock",
        "crates/sim_model/Cargo.toml",
    },
    "comfy-parity-native-frame-interpolate-node-foundation": {
        "crates/sim_nodes/Cargo.toml",
    },
    "comfy-parity-provider-worker-stream-protocol": {
        "Cargo.lock",
        "crates/sim_types/Cargo.toml",
    },
    "comfy-parity-zed-all-target-baseline-assertion-correction": {
        "crates/collab_ui/Cargo.toml",
    },
    "comfy-parity-native-visual-asset-decode-foundation": {
        "Cargo.toml",
        "Cargo.lock",
        "crates/sim_media/Cargo.toml",
    },
    "comfy-parity-native-image-output-codec-effect-foundation": {
        "Cargo.toml",
        "Cargo.lock",
        "crates/sim_media/Cargo.toml",
    },
}

ADAPTERS = (
    "corex",
    "cuda",
    "directml",
    "metal",
    "mlu",
    "npu",
    "rocm",
    "xpu",
)
COMMON_DEPENDENCIES = (
    "anyhow",
    "sim_types",
    "serde",
    "serde_json",
    "thiserror",
)
LINUX_ADAPTERS = frozenset({"corex", "cuda", "mlu", "npu", "rocm", "xpu"})
DIRECTML_FEATURES = (
    "Win32_Foundation",
    "Win32_Graphics_Direct3D12",
    "Win32_Graphics_Dxgi",
    "Win32_Graphics_Dxgi_Common",
    "Win32_Security_WinTrust",
    "Win32_System_Com",
    "Win32_System_LibraryLoader",
)
REASONS = {
    "anyhow": "Adapter-local context for fallible ABI discovery and diagnostics.",
    "sim_types": "Canonical device and native-binding domain types; adapters own no competing device state.",
    "libc": "Linux-only C ABI loader primitives and vendor declaration types.",
    "metal": "macOS-only safe Metal framework binding pinned by the workspace.",
    "objc": "macOS-only checked Objective-C framework shims required beside Metal.",
    "serde": "Versioned adapter ABI, package, and capability metadata serialization.",
    "serde_json": "Machine-readable symbol, layout, package, and diagnostic manifests.",
    "thiserror": "Typed adapter failures without a second device-error domain.",
    "windows": "Windows-only D3D12, DXGI, COM, library-loading, and signature-verification APIs.",
}
FORBIDDEN_SOURCE_PATTERNS = (
    ("bindgen", re.compile(r"\bbindgen\b", re.IGNORECASE)),
    (
        "user-build SDK download command",
        re.compile(
            r"(?:Command::new|\.command)\s*\(\s*(?:r[#]*)?[\"']"
            r"(?:curl|wget|git|powershell|pwsh|certutil)[\"']",
            re.IGNORECASE,
        ),
    ),
    (
        "user-build SDK download API",
        re.compile(
            r"\b(?:download_file|download_to|url_download_to_file|invoke-webrequest)\b",
            re.IGNORECASE,
        ),
    ),
)
PLATFORM_LOADER_NON_OWNER_PATHS = (
    Path("crates/gpui_windows/src"),
    Path("crates/zed/src/main.rs"),
)
SIM_COMPUTE_OWNER_SYMBOLS = (
    "sim_backend_",
    "BackendCapabilityMatrix",
    "NativeBackendBindingStatus",
    "TensorBackend",
)


class ValidationError(Exception):
    pass


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_manifest_path(adapter: str) -> Path:
    return REPOSITORY_ROOT / "crates" / f"sim_backend_{adapter}" / "Cargo.toml"


def expected_target_dependencies(adapter: str) -> dict[str, tuple[str, ...]]:
    if adapter in LINUX_ADAPTERS:
        return {'cfg(target_os = "linux")': ("libc",)}
    if adapter == "metal":
        return {'cfg(target_os = "macos")': ("metal", "objc")}
    if adapter == "directml":
        return {'cfg(target_os = "windows")': ("windows",)}
    raise ValidationError(f"unknown adapter {adapter}")


def validate_workspace_dependencies(workspace: dict) -> None:
    dependencies = workspace["workspace"]["dependencies"]
    expected_requirements = {
        "anyhow": "1.0.86",
        "libc": "0.2",
        "metal": "0.33",
        "objc": "0.2",
        "serde": "1.0.221",
        "serde_json": "1.0.144",
        "thiserror": "2.0.12",
        "windows": "0.61",
    }
    for name, expected in expected_requirements.items():
        declaration = dependencies.get(name)
        actual = declaration.get("version") if isinstance(declaration, dict) else declaration
        if actual != expected:
            raise ValidationError(
                f"workspace dependency {name} must remain {expected}, found {actual!r}"
            )

    sim_types = dependencies.get("sim_types")
    if sim_types != {"path": "crates/sim_types"}:
        raise ValidationError("sim_types must remain the canonical workspace-path domain dependency")

    windows_features = set(dependencies["windows"].get("features", ()))
    missing = sorted(set(DIRECTML_FEATURES) - windows_features)
    if missing:
        raise ValidationError(f"workspace windows dependency is missing DirectML features: {missing}")


def validate_workspace_reference(name: str, declaration: object, context: str) -> None:
    if declaration != {"workspace": True}:
        raise ValidationError(f"{context} dependency {name} must be an unmodified workspace reference")


def validate_adapter_source_policy(adapter: str) -> None:
    adapter_root = adapter_manifest_path(adapter).parent
    source_paths = sorted(
        path
        for path in adapter_root.rglob("*")
        if path.is_file() and path.suffix in {".rs", ".toml"}
    )
    if not source_paths:
        raise ValidationError(f"{adapter_root} contains no auditable adapter sources")
    for path in source_paths:
        source = path.read_text(encoding="utf-8")
        for description, pattern in FORBIDDEN_SOURCE_PATTERNS:
            if pattern.search(source):
                relative = path.relative_to(REPOSITORY_ROOT)
                raise ValidationError(f"{relative} contains forbidden {description}")


def validate_platform_loader_non_ownership() -> None:
    inspected = 0
    for relative_root in PLATFORM_LOADER_NON_OWNER_PATHS:
        root = REPOSITORY_ROOT / relative_root
        paths = [root] if root.is_file() else sorted(root.rglob("*.rs"))
        for path in paths:
            inspected += 1
            source = path.read_text(encoding="utf-8")
            conflicts = [symbol for symbol in SIM_COMPUTE_OWNER_SYMBOLS if symbol in source]
            if conflicts:
                relative = path.relative_to(REPOSITORY_ROOT)
                raise ValidationError(
                    f"platform loader {relative} claims Sim compute ownership: {conflicts}"
                )
    if inspected == 0:
        raise ValidationError("no Zed or GPUI platform loader sources were inspected")


def validate_adapter_manifests() -> dict[str, dict]:
    manifests = {}
    for adapter in ADAPTERS:
        path = adapter_manifest_path(adapter)
        manifest = load_toml(path)
        manifests[adapter] = manifest
        expected_name = f"sim_backend_{adapter}"
        if manifest.get("package", {}).get("name") != expected_name:
            raise ValidationError(f"{path} must declare package {expected_name}")

        dependencies = manifest.get("dependencies", {})
        if set(dependencies) != set(COMMON_DEPENDENCIES):
            raise ValidationError(
                f"{path} common dependencies differ: {sorted(dependencies)}"
            )
        for name in COMMON_DEPENDENCIES:
            validate_workspace_reference(name, dependencies[name], str(path))

        targets = manifest.get("target", {})
        expected_targets = expected_target_dependencies(adapter)
        if set(targets) != set(expected_targets):
            raise ValidationError(f"{path} target dependency sections differ: {sorted(targets)}")
        for target, expected_names in expected_targets.items():
            target_dependencies = targets[target].get("dependencies", {})
            if set(target_dependencies) != set(expected_names):
                raise ValidationError(
                    f"{path} {target} dependencies differ: {sorted(target_dependencies)}"
                )
            for name in expected_names:
                declaration = target_dependencies[name]
                if name == "windows":
                    if declaration.get("workspace") is not True:
                        raise ValidationError(f"{path} windows must inherit its workspace version")
                    if tuple(declaration.get("features", ())) != DIRECTML_FEATURES:
                        raise ValidationError(
                            f"{path} DirectML features must be exactly {list(DIRECTML_FEATURES)}"
                        )
                    if set(declaration) != {"workspace", "features"}:
                        raise ValidationError(f"{path} windows declaration contains an unowned setting")
                else:
                    validate_workspace_reference(name, declaration, f"{path} {target}")

        forbidden_sections = {"build-dependencies", "dev-dependencies"} & set(manifest)
        if forbidden_sections:
            raise ValidationError(f"{path} contains unapproved sections: {sorted(forbidden_sections)}")
        validate_adapter_source_policy(adapter)
    return manifests


def validate_runtime_manifest() -> None:
    path = REPOSITORY_ROOT / "crates" / "sim_runtime" / "Cargo.toml"
    manifest = load_toml(path)
    features = manifest.get("features", {})
    dependencies = manifest.get("dependencies", {})
    for adapter in ADAPTERS:
        package = f"sim_backend_{adapter}"
        if features.get(adapter) != [f"dep:{package}"]:
            raise ValidationError(
                f"{path} feature {adapter} must select only optional dependency {package}"
            )
        if dependencies.get(package) != {"workspace": True, "optional": True}:
            raise ValidationError(
                f"{path} dependency {package} must be an optional workspace reference"
            )
    if dependencies.get("libc") != {"workspace": True}:
        raise ValidationError(
            f"{path} must inherit workspace libc for retained native-library descriptors"
        )
    windows_target = 'cfg(target_os = "windows")'
    targets = manifest.get("target", {})
    windows_dependencies = targets.get(windows_target, {}).get("dependencies", {})
    capture_dependencies = {"cap-fs-ext": "3.4.6", "cap-std": {"workspace": True}}
    for name, declaration in capture_dependencies.items():
        if (
            windows_dependencies.get(name) != declaration
            or name in dependencies
            or any(
                name in target.get("dependencies", {})
                for condition, target in targets.items()
                if condition != windows_target
            )
        ):
            raise ValidationError(
                f"{path} must declare {name} as {declaration!r} only for Windows file identity"
            )


def validate_production_feature_manifests() -> None:
    manifests = {
        "crates/sim_worker/Cargo.toml": lambda adapter: [
            f"sim_runtime/{adapter}",
            f"sim_tensor/{adapter}",
        ],
        "crates/sim_test_support/Cargo.toml": lambda adapter: (
            ["sim_runtime/metal", "sim_tensor/metal"]
            if adapter == "metal"
            else [f"sim_tensor/{adapter}"]
        ),
        "crates/zed/Cargo.toml": lambda adapter: ["sim", f"sim_runtime/{adapter}"],
    }
    for relative_path, expected_feature in manifests.items():
        path = REPOSITORY_ROOT / relative_path
        features = load_toml(path).get("features", {})
        for adapter in ADAPTERS:
            if features.get(adapter) != expected_feature(adapter):
                raise ValidationError(
                    f"{path} feature {adapter} must be exactly {expected_feature(adapter)}"
                )


def lock_package(lock: dict, name: str, version: str | None = None) -> dict:
    matches = [
        package
        for package in lock["package"]
        if package["name"] == name and (version is None or package["version"] == version)
    ]
    if len(matches) != 1:
        raise ValidationError(f"Cargo.lock has {len(matches)} matches for {name} {version or ''}".strip())
    return matches[0]


def dependency_version(lock: dict, adapter: str, name: str) -> str:
    adapter_package = lock_package(lock, f"sim_backend_{adapter}")
    dependency_records = adapter_package.get("dependencies", ())
    matches = [record for record in dependency_records if record == name or record.startswith(f"{name} ")]
    if len(matches) != 1:
        raise ValidationError(
            f"Cargo.lock adapter {adapter} has {len(matches)} dependency records for {name}"
        )
    fields = matches[0].split()
    if len(fields) > 1:
        return fields[1]
    versions = {package["version"] for package in lock["package"] if package["name"] == name}
    if len(versions) != 1:
        raise ValidationError(f"Cargo.lock dependency {name} needs an explicit version qualifier")
    return versions.pop()


def package_record(workspace: dict, lock: dict, name: str, adapter: str) -> dict:
    declaration = workspace["workspace"]["dependencies"][name]
    if isinstance(declaration, str):
        requirement = declaration
        workspace_features = []
        source = "registry+https://github.com/rust-lang/crates.io-index"
    elif "path" in declaration:
        requirement = f"path:{declaration['path']}"
        workspace_features = list(declaration.get("features", ()))
        source = "workspace-path"
    else:
        requirement = declaration["version"]
        workspace_features = list(declaration.get("features", ()))
        source = "registry+https://github.com/rust-lang/crates.io-index"
    resolved_version = dependency_version(lock, adapter, name)
    package = lock_package(lock, name, resolved_version)
    if package.get("source", "workspace-path") != source:
        raise ValidationError(f"Cargo.lock source for {name} differs from the workspace owner")
    return {
        "declared_requirement": requirement,
        "resolved_version": resolved_version,
        "source": source,
        "workspace_features": workspace_features,
    }


def dependency_record(package: str, target: str, features: tuple[str, ...] = ()) -> dict:
    return {
        "package": package,
        "target": target,
        "features": list(features),
        "reason": REASONS[package],
    }


def expected_ledger(workspace: dict, lock: dict) -> dict:
    package_adapters = {
        "anyhow": "cuda",
        "sim_types": "cuda",
        "libc": "cuda",
        "metal": "metal",
        "objc": "metal",
        "serde": "cuda",
        "serde_json": "cuda",
        "thiserror": "cuda",
        "windows": "directml",
    }
    packages = {
        name: package_record(workspace, lock, name, adapter)
        for name, adapter in package_adapters.items()
    }
    adapters = []
    for adapter in ADAPTERS:
        dependencies = [dependency_record(name, "all") for name in COMMON_DEPENDENCIES]
        if adapter in LINUX_ADAPTERS:
            dependencies.append(dependency_record("libc", 'cfg(target_os = "linux")'))
        elif adapter == "metal":
            dependencies.extend(
                dependency_record(name, 'cfg(target_os = "macos")')
                for name in ("metal", "objc")
            )
        elif adapter == "directml":
            dependencies.append(
                dependency_record(
                    "windows", 'cfg(target_os = "windows")', DIRECTML_FEATURES
                )
            )
        adapters.append(
            {
                "package": f"sim_backend_{adapter}",
                "manifest": f"crates/sim_backend_{adapter}/Cargo.toml",
                "dependencies": dependencies,
            }
        )
    runtime_adapters = [
        {
            "feature": adapter,
            "package": f"sim_backend_{adapter}",
            "manifest": "crates/sim_runtime/Cargo.toml",
            "optional": True,
            "reason": (
                "Compiled CoreX structural adapter is forwarded only to preserve the identifier "
                "and canonical typed Unbound state; it has no callable contract, certificate "
                "projection, retained image, loader, or availability path in this pack."
                if adapter == "corex"
                else "Feature-gated runtime certification adapter consumes the focused vendor ABI "
                "crate while sim_runtime::NativeFfiRegistry remains the sole certificate issuer."
            ),
        }
        for adapter in ADAPTERS
    ]
    production_feature_adapters = [
        {
            "feature": adapter,
            "worker": [f"sim_runtime/{adapter}", f"sim_tensor/{adapter}"],
            "test_support": (
                ["sim_runtime/metal", "sim_tensor/metal"]
                if adapter == "metal"
                else [f"sim_tensor/{adapter}"]
            ),
            "zed": ["sim", f"sim_runtime/{adapter}"],
            "reason": (
                "CoreX forwarding compiles the zero-symbol structural adapter through each layer "
                "only so all production surfaces report the same canonical typed Unbound state; "
                "no layer may construct a runtime session, loader, capability row, certificate, "
                "availability claim, or CPU fallback."
                if adapter == "corex"
                else "The private worker alone combines runtime certification and tensor execution "
                "in production; test support selects tensor conformance, with Metal additionally "
                "selecting the existing runtime verifier solely for development hardware "
                "certification, and host Zed selects runtime/profile presentation only."
            ),
        }
        for adapter in ADAPTERS
    ]
    return {
        "schema_version": 1,
        "owner_task": TASK_ID,
        "ownership": {
            "third_party_version_and_workspace_feature_owner": "Cargo.toml [workspace.dependencies]",
            "canonical_device_domain_owner": "sim_types",
            "semantic_backend_and_capability_owner": "sim_tensor::BackendCapabilityMatrix",
            "vendor_abi_owner_boundary": "each focused sim_backend_* adapter",
            "native_ffi_certificate_owner": "sim_runtime::NativeFfiRegistry",
            "runtime_adapter_boundary": "enabled vendor adapters inspect exact immutable bytes and consume certificates before unsafe loading; CoreX is the sole zero-symbol typed-Unbound forwarding exception with no loader or certificate projection",
            "platform_loader_non_owner_boundary": "Zed and GPUI platform/UI loaders own no Sim compute semantics",
        },
        "lockfile": {
            "path": "Cargo.lock",
            "sha256": sha256(REPOSITORY_ROOT / "Cargo.lock"),
        },
        "packages": packages,
        "adapters": adapters,
        "runtime_adapters": runtime_adapters,
        "production_feature_adapters": production_feature_adapters,
        "prohibitions": [
            "no bindgen",
            "no user-build SDK downloads",
            "no unscoped later Cargo manifest writer",
            "no unscoped later Cargo.lock writer",
        ],
    }


def validate_later_task_writes() -> None:
    # Recorded task IDs and write lists remain evidence of their original owners.
    workspace_dependencies = load_toml(REPOSITORY_ROOT / "Cargo.toml")["workspace"]["dependencies"]
    renamed_roots = {"crates/comfy_" + name.removeprefix("sim_") + "/": "crates/" + name + "/"
                     for name in workspace_dependencies if name.startswith("sim_")}
    def current_owner_path(path: str) -> str:
        for old_root, new_root in renamed_roots.items():
            if path.startswith(old_root):
                return new_root + path.removeprefix(old_root)
        return path

    tasks_path = SPECIFICATION_ROOT / "tasks.md"
    text = tasks_path.read_text(encoding="utf-8")
    blocks = re.split(r"(?=^- \[[ xX~-]\] \d+\.)", text, flags=re.MULTILINE)
    owner_index = next(
        (index for index, block in enumerate(blocks) if f"_id: {TASK_ID}" in block),
        None,
    )
    if owner_index is None:
        raise ValidationError(f"{TASK_ID} is absent from tasks.md")
    violations = []
    for block in blocks[owner_index + 1 :]:
        heading = block.splitlines()[0] if block.splitlines() else "unknown task"
        writes_line = next(
            (line for line in block.splitlines() if line.startswith("  - Writes: ")),
            None,
        )
        if writes_line is None:
            continue
        writes = [
            current_owner_path(path.strip())
            for path in writes_line.removeprefix("  - Writes: ").split(",")
            if path.strip()
        ]
        forbidden = [
            path
            for path in writes
            if path == "Cargo.toml" or path == "Cargo.lock" or path.endswith("/Cargo.toml")
        ]
        if f"_id: {CERTIFICATION_TASK_ID}" in block:
            if set(forbidden) == CERTIFICATION_WRITES:
                continue
            violations.append(
                f"{heading}: expected exact certification dependency writes "
                f"{sorted(CERTIFICATION_WRITES)}, found {sorted(forbidden)}"
            )
            continue
        task_id = next(
            (
                line.removeprefix("  - _id: ")
                for line in block.splitlines()
                if line.startswith("  - _id: ")
            ),
            None,
        )
        if task_id in AUTHORIZED_INTEGRATION_WRITES:
            expected_writes = AUTHORIZED_INTEGRATION_WRITES[task_id]
            if set(forbidden) == expected_writes:
                continue
            violations.append(
                f"{heading}: expected exact integration dependency writes "
                f"{sorted(expected_writes)}, found {sorted(forbidden)}"
            )
            continue
        if forbidden:
            violations.append(f"{heading}: {', '.join(forbidden)}")
    if violations:
        raise ValidationError("later tasks own Cargo dependency files:\n" + "\n".join(violations))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the native backend dependency ownership ledger")
    parser.add_argument(
        "--render",
        action="store_true",
        help="print the exact expected ledger without reading the checked-in ledger",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace the checked-in ledger with the exact validated workspace projection",
    )
    arguments = parser.parse_args()
    if arguments.render and arguments.write:
        parser.error("--render and --write are mutually exclusive")

    try:
        workspace = load_toml(REPOSITORY_ROOT / "Cargo.toml")
        lock = load_toml(REPOSITORY_ROOT / "Cargo.lock")
        validate_workspace_dependencies(workspace)
        validate_adapter_manifests()
        validate_runtime_manifest()
        validate_production_feature_manifests()
        validate_platform_loader_non_ownership()
        validate_later_task_writes()
        expected = expected_ledger(workspace, lock)
        if arguments.render:
            print(json.dumps(expected, indent=2) + "\n", end="")
            return 0
        if arguments.write:
            LEDGER_PATH.write_text(
                json.dumps(expected, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"wrote exact native backend dependency ledger to {LEDGER_PATH}")
            return 0
        actual = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValidationError(
                f"{LEDGER_PATH} differs from the exact workspace, manifest, or lockfile state"
            )
    except (KeyError, OSError, tomllib.TOMLDecodeError, json.JSONDecodeError, ValidationError) as error:
        print(f"native backend dependency validation failed: {error}", file=sys.stderr)
        return 1

    print(
        "native backend dependency validation passed: "
        f"{len(ADAPTERS)} adapters, {len(expected['packages'])} packages, "
        f"lock {expected['lockfile']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
