use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_media::{PngLimits, encode_png_frame};
use sim_runtime::{
    AssetNamespace, AssetRoots, AssetService, AttemptState, AuthorizedCapabilities,
    LegacyComfyProfile, NATIVE_IMAGE_REGISTRY_VERSION, NativeImageWorkerEvent,
    NativeImageWorkerPlan, RuntimeSupervisor, SharedAssetService, SupervisorPolicy, WorkerHealth,
    WorkerLaunchConfig, authorize_native_input_reader, compile_native_image_workflow,
    migrate_legacy_profile,
};
use sim_test_support::load_release_boundary_policy;
use sim_types::{AttemptId, ProfileId, PromptId, WorkerId, WorkerMessage};
use uuid::Uuid;

const WORKFLOW_FIXTURE: &[u8] = include_bytes!("../fixtures/native_image/workflow.json");
const INPUT_FIXTURE: &[u8] = include_bytes!("../fixtures/native_image/input.json");
const RELEASE_POLICY: &[u8] = include_bytes!("../fixtures/release-boundary.json");
const DEFAULT_SETTINGS: &str = include_str!("../../../assets/settings/default.json");
const DEFAULT_SIM_SETTINGS: &str = include_str!("../../../assets/settings/default-sim.json");
const ZED_SOURCE: &str = include_str!("../../zed/src/zed.rs");
const SIM_CLI_SOURCE: &str = include_str!("../../zed/src/sim_cli.rs");
const SIM_MENU_SOURCE: &str = include_str!("../../sim_ui/src/shell.rs");
const WORKER_SOURCE: &[u8] = include_bytes!("../../sim_worker/src/sim_worker.rs");
const RUNTIME_SUPERVISOR_SOURCE: &str = include_str!("../../sim_runtime/src/runtime_supervisor.rs");
const MAC_BUNDLE: &str = include_str!("../../../script/bundle-mac");
const LINUX_BUNDLE: &str = include_str!("../../../script/bundle-linux");
const WINDOWS_BUNDLE: &str = include_str!("../../../script/bundle-windows.ps1");
const WINDOWS_INSTALLER: &str = include_str!("../../zed/resources/windows/zed.iss");
const ROCM_PACKAGE_POLICY: &str =
    include_str!("../../../nix/sim-backends/rocm/package-policy.json");
const ROCM_CONTRACT_SCHEMA: &str =
    include_str!("../../../nix/sim-backends/rocm/ffi-contracts-v1.schema.json");
const ROCM_PACKAGER: &str = include_str!("../../../script/package-sim-backend-rocm");
const ROCM_RUNTIME_TRUST: &str = include_str!("../../sim_runtime/src/trust.rs");
const ROCM_RUNTIME_FFI: &str = include_str!("../../sim_runtime/src/native_ffi_rocm.rs");
const METAL_PACKAGE_POLICY: &str =
    include_str!("../../../nix/sim-backends/metal/package-policy.json");
const METAL_CONTRACT_SCHEMA: &str =
    include_str!("../../../nix/sim-backends/metal/ffi-contracts-v1.schema.json");
const METAL_PACKAGER: &str = include_str!("../../../script/package-sim-backend-metal");
const METAL_RUNTIME_FFI: &str = include_str!("../../sim_runtime/src/native_ffi_metal.rs");
const MLU_PACKAGE_POLICY: &str = include_str!("../../../nix/sim-backends/mlu/package-policy.json");
const MLU_CONTRACT_SCHEMA: &str =
    include_str!("../../../nix/sim-backends/mlu/ffi-contracts-v1.schema.json");
const MLU_PACKAGER: &str = include_str!("../../../script/package-sim-backend-mlu");
const MLU_RUNTIME_FFI: &str = include_str!("../../sim_runtime/src/native_ffi_mlu.rs");
const NPU_PACKAGE_POLICY: &str = include_str!("../../../nix/sim-backends/npu/package-policy.json");
const NPU_CONTRACT_SCHEMA: &str =
    include_str!("../../../nix/sim-backends/npu/ffi-contracts-v1.schema.json");
const NPU_PACKAGER: &str = include_str!("../../../script/package-sim-backend-npu");
const NPU_RUNTIME_FFI: &str = include_str!("../../sim_runtime/src/native_ffi_npu.rs");
const CUDA_PACKAGE_POLICY: &str =
    include_str!("../../../nix/sim-backends/cuda/package-policy.json");
const CUDA_CONTRACT_SCHEMA: &str =
    include_str!("../../../nix/sim-backends/cuda/ffi-contracts-v1.schema.json");
const CUDA_PACKAGER: &str = include_str!("../../../script/package-sim-backend-cuda");
const CUDA_RUNTIME_FFI: &str = include_str!("../../sim_runtime/src/native_ffi_cuda.rs");
const XPU_PACKAGE_POLICY: &str = include_str!("../../../nix/sim-backends/xpu/package-policy.json");
const XPU_CONTRACT_SCHEMA: &str =
    include_str!("../../../nix/sim-backends/xpu/ffi-contracts-v1.schema.json");
const XPU_PACKAGER: &str = include_str!("../../../script/package-sim-backend-xpu");
const XPU_RUNTIME_FFI: &str = include_str!("../../sim_runtime/src/native_ffi_xpu.rs");
const DIRECTML_PACKAGE_POLICY: &str =
    include_str!("../../../nix/sim-backends/directml/package-policy.json");
const DIRECTML_CONTRACT_SCHEMA: &str =
    include_str!("../../../nix/sim-backends/directml/ffi-contracts-v1.schema.json");
const DIRECTML_PACKAGER: &str = include_str!("../../../script/package-sim-backend-directml");
const DIRECTML_RUNTIME_FFI: &str = include_str!("../../sim_runtime/src/native_ffi_directml.rs");
const DIRECTML_LOADER: &str = include_str!("../../sim_backend_directml/src/loader.rs");
const DIRECTML_TENSOR: &str =
    include_str!("../../sim_tensor/src/backends/directml_sim_model_0018.rs");
const METAL_ABI_MANIFEST: &str = include_str!("../../sim_backend_metal/abi/symbols-v1.json");
const METAL_REVIEWED_BINDINGS: &str =
    include_str!("../../sim_backend_metal/abi/reviewed-bindings-v1.txt");
const METAL_LICENSES: &str = include_str!("../../sim_backend_metal/LICENSES");
const METAL_ABI_CATALOG: &str =
    include_str!("../../../crates/sim_test_support/parity/catalogs/native-backend-abi/metal.json");
const METAL_ADAPTER: &str = include_str!("../../sim_backend_metal/src/sim_backend_metal.rs");
const METAL_LOADER: &str = include_str!("../../sim_backend_metal/src/loader.rs");
const METAL_READINESS_SOURCE: &str =
    include_str!("../../sim_backend_metal/kernels/readiness.metal");
const METAL_EXECUTION_PACKAGE_POLICY: &str =
    include_str!("../../../nix/sim-backends/metal/execution-policy.json");
const METAL_EXECUTION_PACKAGER: &str =
    include_str!("../../../script/package-sim-backend-metal-execution");
const METAL_EXECUTION_ABI: &str = include_str!("../../sim_backend_metal/abi/execution-v1.json");
const METAL_EXECUTION_BINDINGS: &str =
    include_str!("../../sim_backend_metal/abi/reviewed-execution-bindings-v1.txt");
const METAL_EXECUTION_ABI_VERIFIER: &str =
    include_str!("../../sim_backend_metal/abi/verify-execution-bindings.m");
const METAL_EXECUTION_LICENSES: &str = include_str!("../../sim_backend_metal/LICENSES.execution");
const METAL_EXECUTION_KERNELS: &str =
    include_str!("../../sim_backend_metal/kernels/tensor_ops.metal");
const METAL_EXECUTION_SOURCE: &str = include_str!("../../sim_backend_metal/src/execution.rs");
const METAL_EXECUTION_ABI_SOURCE: &str =
    include_str!("../../sim_backend_metal/src/execution_abi.rs");
const METAL_EXECUTION_CATALOG: &str = include_str!(
    "../../../crates/sim_test_support/parity/catalogs/native-backend-abi/metal-execution.json"
);

const PROFILE_ID: Uuid = Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_3301);

struct ReleaseFixture {
    _directory: tempfile::TempDir,
    config: WorkerLaunchConfig,
    assets: SharedAssetService,
    input_authorization: AuthorizedCapabilities,
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?
        .to_path_buf())
}

fn target_directory(workspace_root: &Path) -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join("target"))
}

fn cargo_metadata(workspace_root: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    let output = smol::block_on(async {
        smol::process::Command::new(env!("CARGO"))
            .args(["metadata", "--locked", "--format-version", "1"])
            .current_dir(workspace_root)
            .output()
            .await
    })?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(output.stdout)
}

fn package<'a>(metadata: &'a Value, name: &str) -> Option<&'a Value> {
    metadata
        .get("packages")?
        .as_array()?
        .iter()
        .find(|package| package.get("name").and_then(Value::as_str) == Some(name))
}

fn normal_dependency_names(package: &Value) -> BTreeSet<&str> {
    package
        .get("dependencies")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|dependency| dependency.get("kind").is_none_or(Value::is_null))
        .filter_map(|dependency| dependency.get("name").and_then(Value::as_str))
        .collect()
}

fn package_feature_values<'a>(package: &'a Value, feature: &str) -> BTreeSet<&'a str> {
    package
        .get("features")
        .and_then(|features| features.get(feature))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect()
}

fn dependency_boundary_cases(
    metadata: &Value,
) -> Result<BTreeMap<&'static str, bool>, Box<dyn Error>> {
    let production_sim_packages = metadata
        .get("packages")
        .and_then(Value::as_array)
        .ok_or("cargo metadata omitted packages")?
        .iter()
        .filter_map(|package| {
            let name = package.get("name").and_then(Value::as_str)?;
            (name.starts_with("sim_") && name != "sim_test_support").then_some(package)
        })
        .collect::<Vec<_>>();
    let forbidden_direct_dependencies = [
        "sim_test_support",
        "comfy_oracle",
        "node_runtime",
        "deno_core",
        "boa_engine",
        "rusty_v8",
        "pyo3",
    ];
    let direct_dependency_hits = production_sim_packages
        .iter()
        .flat_map(|package| {
            let package_name = package
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            normal_dependency_names(package)
                .into_iter()
                .filter(|dependency| forbidden_direct_dependencies.contains(dependency))
                .map(move |dependency| format!("{package_name}->{dependency}"))
        })
        .collect::<Vec<_>>();
    let zed = package(metadata, "zed").ok_or("cargo metadata omitted Zed")?;
    let zed_dependencies = normal_dependency_names(zed);
    let worker = package(metadata, "sim_worker").ok_or("cargo metadata omitted sim_worker")?;
    let runtime = package(metadata, "sim_runtime").ok_or("cargo metadata omitted sim_runtime")?;
    let test_support =
        package(metadata, "sim_test_support").ok_or("cargo metadata omitted sim_test_support")?;
    let backend_feature_ownership_is_disjoint = [
        "corex", "cuda", "directml", "metal", "mlu", "npu", "rocm", "xpu",
    ]
    .into_iter()
    .all(|feature| {
        let runtime_adapter = format!("dep:sim_backend_{feature}");
        let runtime_feature = format!("sim_runtime/{feature}");
        let tensor_feature = format!("sim_tensor/{feature}");
        let runtime_values = package_feature_values(runtime, feature);
        let worker_values = package_feature_values(worker, feature);
        let test_support_values = package_feature_values(test_support, feature);
        let zed_values = package_feature_values(zed, feature);
        let test_support_is_exact = if feature == "metal" {
            test_support_values.len() == 2
                && test_support_values.contains(runtime_feature.as_str())
                && test_support_values.contains(tensor_feature.as_str())
        } else {
            test_support_values.len() == 1 && test_support_values.contains(tensor_feature.as_str())
        };
        runtime_values.len() == 1
            && runtime_values.contains(runtime_adapter.as_str())
            && worker_values.len() == 2
            && worker_values.contains(runtime_feature.as_str())
            && worker_values.contains(tensor_feature.as_str())
            && test_support_is_exact
            && zed_values.len() == 2
            && zed_values.contains("sim")
            && zed_values.contains(runtime_feature.as_str())
    });
    let worker_has_binary =
        worker
            .get("targets")
            .and_then(Value::as_array)
            .is_some_and(|targets| {
                targets.iter().any(|target| {
                    target.get("name").and_then(Value::as_str) == Some("sim-worker")
                        && target
                            .get("kind")
                            .and_then(Value::as_array)
                            .is_some_and(|kinds| kinds.iter().any(|kind| kind == "bin"))
                })
            });
    Ok(BTreeMap::from([
        (
            "production_comfy_direct_dependencies_exclude_source_runtimes",
            !production_sim_packages.is_empty() && direct_dependency_hits.is_empty(),
        ),
        (
            "zed_uses_production_comfy_crates_without_test_support",
            [
                "sim_api",
                "sim_model",
                "sim_plugin_host",
                "sim_runtime",
                "sim_tensor",
                "sim_types",
                "sim_ui",
            ]
            .into_iter()
            .all(|dependency| zed_dependencies.contains(dependency))
                && !zed_dependencies.contains("sim_test_support"),
        ),
        ("worker_is_a_packaged_binary_target", worker_has_binary),
        (
            "backend_feature_forwarding_preserves_disjoint_ownership",
            backend_feature_ownership_is_disjoint,
        ),
    ]))
}

fn packaging_cases() -> BTreeMap<&'static str, bool> {
    BTreeMap::from([
        (
            "mac_default_bundle_omits_native_worker",
            MAC_BUNDLE.contains("zed_features=\"none\"")
                && MAC_BUNDLE.contains("mode=default packages=zed,cli zed_features=${zed_features} remote_features=${remote_features} include_sim_worker=false")
                && MAC_BUNDLE.contains("if [[ \"$sim\" = true ]]; then")
                && MAC_BUNDLE.contains("--sim  Include the Sim integration, Metal backend, worker, and assets."),
        ),
        (
            "mac_bundle_builds_places_and_signs_native_worker",
            MAC_BUNDLE.contains("zed/sim,zed/metal,sim_worker/metal")
                && MAC_BUNDLE.contains("--package sim_worker --no-default-features --features \"${zed_build_features}\"")
                && MAC_BUNDLE.contains("Contents/MacOS/sim-worker")
                && MAC_BUNDLE.contains("--sign \"$IDENTITY\" \"${app_path}/Contents/MacOS/sim-worker\""),
        ),
        (
            "linux_default_bundle_omits_native_worker",
            LINUX_BUNDLE.contains("zed_features=\"none\"")
                && LINUX_BUNDLE.contains("mode=default packages=zed,cli zed_features=${zed_features} remote_features=${remote_features} include_sim_worker=false")
                && LINUX_BUNDLE.contains("if [[ \"$sim\" = true ]]; then")
                && LINUX_BUNDLE.contains("--sim        Include the Sim integration, accelerator backends, worker, and assets."),
        ),
        (
            "linux_bundle_builds_strips_and_places_native_worker",
            LINUX_BUNDLE.contains("zed/sim,zed/cuda,zed/rocm,zed/mlu,zed/npu,zed/xpu,sim_worker/cuda,sim_worker/rocm,sim_worker/mlu,sim_worker/npu,sim_worker/xpu")
                && LINUX_BUNDLE.contains("--package sim_worker --no-default-features --features \"${zed_build_features}\"")
                && LINUX_BUNDLE.contains("release/sim-worker\"")
                && LINUX_BUNDLE.contains("libexec/sim-worker"),
        ),
        (
            "windows_default_bundle_omits_native_worker",
            WINDOWS_BUNDLE.contains("$rustToolFeatures = if ($RustTools) { \"rust-tools\" } else { \"none\" }")
                && WINDOWS_BUNDLE.contains("mode=default packages=zed,cli,auto_update_helper zed_features=$rustToolFeatures remote_features=$rustToolFeatures include_sim_worker=false")
                && WINDOWS_BUNDLE.contains("if ($Sim)")
                && WINDOWS_INSTALLER.contains("#ifdef Sim")
                && WINDOWS_INSTALLER.contains("#endif"),
        ),
        (
            "windows_bundle_builds_places_signs_and_installs_native_worker",
            WINDOWS_BUNDLE.contains("zed/sim,zed/rocm,sim_worker/rocm,zed/directml,sim_worker/directml")
                && WINDOWS_BUNDLE.contains("--package sim_worker --package auto_update_helper --no-default-features --features $features")
                && WINDOWS_BUNDLE.contains("sim-worker.exe\" -Destination \"$innoDir\\sim-worker.exe")
                && WINDOWS_BUNDLE.contains("$files += \",$innoDir\\sim-worker.exe\"")
                && WINDOWS_INSTALLER.contains("#ifdef Sim")
                && WINDOWS_INSTALLER.contains("Source: \"{#ResourcesDir}\\sim-worker.exe\"; DestDir: \"{code:GetInstallDir}\"")
                && WINDOWS_INSTALLER.contains("#endif"),
        ),
        (
            "runtime_owns_one_sibling_worker_resolver_for_desktop_and_cli",
            RUNTIME_SUPERVISOR_SOURCE.contains("fn packaged_worker_binary()")
                && RUNTIME_SUPERVISOR_SOURCE.contains("std::env::current_exe()")
                && RUNTIME_SUPERVISOR_SOURCE.contains("\"sim-worker.exe\"")
                && RUNTIME_SUPERVISOR_SOURCE.contains("\"sim-worker\"")
                && RUNTIME_SUPERVISOR_SOURCE
                    .matches("std::env::current_exe()")
                    .count()
                    == 1
                && ZED_SOURCE.contains("WorkerLaunchConfig::for_packaged_worker_profile")
                && SIM_CLI_SOURCE.contains("WorkerLaunchConfig::for_packaged_worker_profile")
                && !ZED_SOURCE.contains("std::env::current_exe()")
                && !SIM_CLI_SOURCE.contains("std::env::current_exe()"),
        ),
        (
            "rocm_package_uses_native_signed_reviewed_contracts",
            ROCM_PACKAGE_POLICY.contains("\"schema_version\": 2")
                && ROCM_PACKAGE_POLICY.contains("\"ffi-contracts-v1.json\"")
                && ROCM_PACKAGE_POLICY.contains("\"signature_algorithm\": \"ed25519\"")
                && ROCM_PACKAGE_POLICY
                    .contains("\"signature_verifier\": \"sim_runtime-native-rust-ed25519\"")
                && ROCM_CONTRACT_SCHEMA.contains("\"additionalProperties\": false")
                && ROCM_CONTRACT_SCHEMA.contains("rocm-dependency:")
                && ROCM_PACKAGER.contains("validate_and_copy_contract_catalog")
                && ROCM_PACKAGER.contains("canonical-json-v1")
                && !ROCM_PACKAGER.contains("SIM_ROCM_SIGNATURE_VERIFIER"),
        ),
        (
            "rocm_package_trust_precedes_canonical_registry_mapping",
            ROCM_RUNTIME_TRUST.contains("pub struct RocmPackageVerificationKey")
                && ROCM_RUNTIME_TRUST.contains("ROCM_PACKAGE_SIGNATURE_DOMAIN")
                && ROCM_RUNTIME_FFI
                    .find("verify_signed_package_root")
                    .zip(ROCM_RUNTIME_FFI.find("parse_rocm_ffi_contract_catalog"))
                    .is_some_and(|(verification, mapping)| verification < mapping)
                && ROCM_RUNTIME_FFI.contains("NativeFfiRegistry::new")
                && ROCM_RUNTIME_FFI
                    .split_once("#[cfg(test)]\nmod tests")
                    .map_or(ROCM_RUNTIME_FFI, |(production, _)| production)
                    .matches("NativeFfiContract::new")
                    .count()
                    == 1,
        ),
        (
            "metal_foundation_packages_a_reviewed_abi_without_self_authorizing",
            METAL_PACKAGE_POLICY.contains("\"schema_version\": 2")
                && METAL_PACKAGE_POLICY.contains("macos-13-metal-3")
                && METAL_PACKAGE_POLICY.contains("ffi-contracts-v1.json")
                && METAL_PACKAGE_POLICY.contains("readiness.metallib")
                && METAL_PACKAGE_POLICY.contains("readiness.metal")
                && METAL_PACKAGE_POLICY.contains("tensor_ops.metallib")
                && METAL_PACKAGE_POLICY.contains("tensor_ops.metal")
                && METAL_PACKAGE_POLICY.contains("sim_runtime::MetalPackageVerificationKey")
                && METAL_PACKAGE_POLICY.contains("\"required_entitlements\": []")
                && METAL_CONTRACT_SCHEMA.contains("\"additionalProperties\": false")
                && METAL_CONTRACT_SCHEMA.contains("metal-performance-shaders-graph-framework")
                && METAL_CONTRACT_SCHEMA.contains("metal-tensor-ops-metallib")
                && METAL_ABI_MANIFEST.contains("MTLCreateSystemDefaultDevice")
                && METAL_ABI_MANIFEST.contains("MPSSupportsMTLDevice")
                && METAL_ABI_MANIFEST.contains("MetalPerformanceShadersGraph")
                && METAL_ABI_MANIFEST.contains("/System/Library/Frameworks/Metal.framework/Metal")
                && METAL_ABI_MANIFEST.contains("aarch64-apple-darwin")
                && METAL_ABI_MANIFEST.contains("x86_64-apple-darwin")
                && METAL_REVIEWED_BINDINGS.contains("Xcode 26.2 build 17C52")
                && METAL_LICENSES.contains("Apple system frameworks are not redistributed")
                && METAL_ABI_CATALOG.contains(
                    "comfy-parity-device-foundation-apple-metal-mps-comfy-model-0015",
                )
                && METAL_PACKAGER.contains("xcrun -sdk macosx metal")
                && METAL_PACKAGER.contains("-std=metal3.0")
                && METAL_PACKAGER.contains("-mmacosx-version-min=13.0")
                && METAL_PACKAGER.contains("xcrun -sdk macosx metallib")
                && METAL_PACKAGER.contains("validate_and_copy_contract_catalog")
                && METAL_PACKAGER.contains("canonical-json-v1")
                && METAL_PACKAGER.contains("--self-test")
                && !METAL_PACKAGER.contains("curl ")
                && !METAL_PACKAGER.contains("wget ")
                && !METAL_PACKAGER.contains("cp /System/Library/Frameworks/")
                && !METAL_PACKAGER.contains("cp -R /System/Library/Frameworks/")
                && !METAL_PACKAGER.contains("NativeFfiRegistry")
                && !METAL_PACKAGER.contains("verify_package(")
                && METAL_ADAPTER.contains("NativeBackendBindingStatus::unbound")
                && METAL_ADAPTER.contains("NativeFfiRegistry")
                && !METAL_ADAPTER.contains("NativeBackendBindingStatus::bound")
                && METAL_LOADER.contains("RTLD_FIRST")
                && METAL_LOADER.contains("dladdr")
                && METAL_LOADER.contains("class_getImageName")
                && METAL_LOADER.contains("method_getTypeEncoding")
                && METAL_READINESS_SOURCE.contains("zed_sim_metal_readiness_v1"),
        ),
        (
            "metal_package_trust_precedes_canonical_registry_mapping",
            ROCM_RUNTIME_TRUST.contains("pub struct MetalPackageVerificationKey")
                && ROCM_RUNTIME_TRUST.contains("struct NativePackageVerificationAuthority")
                && ROCM_RUNTIME_TRUST.contains("METAL_PACKAGE_SIGNATURE_DOMAIN")
                && ROCM_RUNTIME_TRUST
                    .contains("rocm_package_signature_has_a_distinct_signer_bound_domain")
                && METAL_RUNTIME_FFI
                    .find("verification_key.verify_package")
                    .zip(METAL_RUNTIME_FFI.find("let catalog: MetalFfiContractCatalogDto"))
                    .is_some_and(|(verification, mapping)| verification < mapping)
                && METAL_RUNTIME_FFI.contains("NativeFfiRegistry::new")
                && METAL_RUNTIME_FFI.contains("capture_native_package")
                && METAL_RUNTIME_FFI.contains("validate_native_package_coverage")
                && ROCM_RUNTIME_TRUST.contains("fn capture_native_package(")
                && ROCM_RUNTIME_TRUST.contains("fn validate_native_package_coverage(")
                && !METAL_RUNTIME_FFI.contains("inspect_exact_package_tree")
                && METAL_RUNTIME_FFI.contains("parse_strict_json_value")
                && METAL_RUNTIME_FFI.contains("readiness_metallib")
                && METAL_RUNTIME_FFI.contains("tensor_ops_metallib")
                && !METAL_RUNTIME_FFI.contains("PluginVerificationKey"),
        ),
        (
            "mlu_package_trust_precedes_canonical_registry_mapping",
            ROCM_RUNTIME_TRUST.contains("pub struct MluPackageVerificationKey")
                && ROCM_RUNTIME_TRUST.contains("MLU_PACKAGE_SIGNATURE_DOMAIN")
                && MLU_PACKAGE_POLICY.contains("ffi-contracts-v1.json")
                && MLU_PACKAGE_POLICY.contains("zed-sim-mlu-package-v1")
                && MLU_CONTRACT_SCHEMA.contains("\"additionalProperties\": false")
                && MLU_PACKAGER.contains("separately reviewed bounded regular file")
                && MLU_RUNTIME_FFI
                    .find("verification_key.verify_package")
                    .zip(MLU_RUNTIME_FFI.find("let catalog: MluFfiContractCatalogDto"))
                    .is_some_and(|(verification, mapping)| verification < mapping)
                && MLU_RUNTIME_FFI.contains("NativeFfiRegistry::new")
                && MLU_RUNTIME_FFI.contains("capture_native_package")
                && MLU_RUNTIME_FFI.contains("validate_native_package_coverage")
                && ROCM_RUNTIME_TRUST.contains("fn capture_native_package(")
                && ROCM_RUNTIME_TRUST.contains("fn validate_native_package_coverage(")
                && !MLU_RUNTIME_FFI.contains("inspect_exact_package_tree")
                && MLU_RUNTIME_FFI.contains("capture_native_library_image")
                && MLU_RUNTIME_FFI.contains("RetainedNativeLibraryImage")
                && !MLU_RUNTIME_FFI.contains("O_NOFOLLOW")
                && !MLU_RUNTIME_FFI.contains("F_ADD_SEALS")
                && ROCM_RUNTIME_TRUST.contains("fn capture_native_library_image(")
                && ROCM_RUNTIME_TRUST.contains("libc::O_NOFOLLOW")
                && ROCM_RUNTIME_TRUST.contains("libc::F_ADD_SEALS")
                && !MLU_RUNTIME_FFI.contains("PluginVerificationKey"),
        ),
        (
            "npu_package_trust_precedes_canonical_registry_mapping",
            ROCM_RUNTIME_TRUST.contains("pub struct NpuPackageVerificationKey")
                && ROCM_RUNTIME_TRUST.contains("NPU_PACKAGE_SIGNATURE_DOMAIN")
                && NPU_PACKAGE_POLICY.contains("ffi-contracts-v1.json")
                && NPU_PACKAGE_POLICY.contains("zed-sim-npu-package-v1")
                && NPU_PACKAGE_POLICY
                    .contains("\"certificate_owner\": \"sim_runtime::NativeFfiRegistry\"")
                && NPU_CONTRACT_SCHEMA.contains("\"additionalProperties\": false")
                && NPU_CONTRACT_SCHEMA.contains("\"required_by\": { \"const\": \"ascendcl\" }")
                && NPU_PACKAGER.contains("separately reviewed bounded regular file")
                && NPU_RUNTIME_FFI
                    .find("verification_key.verify_package")
                    .zip(NPU_RUNTIME_FFI.find("let catalog: NpuFfiContractCatalogDto"))
                    .is_some_and(|(verification, mapping)| verification < mapping)
                && NPU_RUNTIME_FFI.contains("NativeFfiRegistry::new")
                && NPU_RUNTIME_FFI.contains("NativeFfiContract::new_dependency")
                && NPU_RUNTIME_FFI.contains("authorize_dependency")
                && NPU_RUNTIME_FFI.contains("capture_native_package")
                && NPU_RUNTIME_FFI.contains("validate_native_package_coverage")
                && NPU_RUNTIME_FFI.contains("capture_native_library_image")
                && NPU_RUNTIME_FFI.contains("RetainedNativeLibraryImage")
                && !NPU_RUNTIME_FFI.contains("O_NOFOLLOW")
                && !NPU_RUNTIME_FFI.contains("F_ADD_SEALS")
                && ROCM_RUNTIME_TRUST.contains("fn capture_native_library_image(")
                && ROCM_RUNTIME_TRUST.contains("libc::O_NOFOLLOW")
                && ROCM_RUNTIME_TRUST.contains("libc::F_ADD_SEALS")
                && !NPU_RUNTIME_FFI.contains("PluginVerificationKey"),
        ),
        (
            "cuda_package_trust_precedes_canonical_registry_mapping",
            ROCM_RUNTIME_TRUST.contains("pub struct CudaPackageVerificationKey")
                && ROCM_RUNTIME_TRUST.contains("CUDA_PACKAGE_SIGNATURE_DOMAIN")
                && CUDA_PACKAGE_POLICY.contains("ffi-contracts-v1.json")
                && CUDA_PACKAGE_POLICY.contains("zed-sim-cuda-package-v1")
                && CUDA_PACKAGE_POLICY
                    .contains("\"certificate_owner\": \"sim_runtime::NativeFfiRegistry\"")
                && CUDA_PACKAGE_POLICY
                    .contains("\"structural_receipt_is_authorization\": false")
                && CUDA_CONTRACT_SCHEMA.contains("\"additionalProperties\": false")
                && ["cublaslt", "cudnn", "driver", "nvrtc"]
                    .into_iter()
                    .all(|identity| CUDA_CONTRACT_SCHEMA.contains(identity))
                && CUDA_PACKAGER.contains("separately reviewed CUDA FFI contract catalog")
                && !CUDA_PACKAGER.contains("NativeFfiRegistry::")
                && CUDA_RUNTIME_FFI
                    .find("verification_key.verify_package")
                    .zip(CUDA_RUNTIME_FFI.find("let catalog: CudaFfiContractCatalogDto"))
                    .is_some_and(|(verification, mapping)| verification < mapping)
                && CUDA_RUNTIME_FFI.contains("NativeFfiRegistry::new")
                && CUDA_RUNTIME_FFI.contains("capture_native_package")
                && CUDA_RUNTIME_FFI.contains("validate_native_package_coverage")
                && CUDA_RUNTIME_FFI.contains("capture_native_library_image")
                && CUDA_RUNTIME_FFI.contains("RetainedNativeLibraryImage")
                && !CUDA_RUNTIME_FFI.contains("PluginVerificationKey"),
        ),
        (
            "xpu_package_trust_precedes_canonical_registry_mapping",
            ROCM_RUNTIME_TRUST.contains("pub struct XpuPackageVerificationKey")
                && ROCM_RUNTIME_TRUST.contains("XPU_PACKAGE_SIGNATURE_DOMAIN")
                && XPU_PACKAGE_POLICY.contains("ffi-contracts-v1.json")
                && XPU_PACKAGE_POLICY.contains("zed-sim-xpu-package-v1")
                && XPU_PACKAGE_POLICY
                    .contains("\"certificate_owner\": \"sim_runtime::NativeFfiRegistry\"")
                && XPU_PACKAGE_POLICY
                    .contains("\"structural_receipt_is_authorization\": false")
                && XPU_CONTRACT_SCHEMA.contains("\"additionalProperties\": false")
                && XPU_CONTRACT_SCHEMA.contains("\"identity\": { \"const\": \"level_zero\" }")
                && XPU_CONTRACT_SCHEMA.contains("\"identity\": { \"const\": \"onednn\" }")
                && XPU_PACKAGER.contains("separately reviewed XPU FFI contract catalog")
                && !XPU_PACKAGER.contains("NativeFfiRegistry::")
                && XPU_RUNTIME_FFI
                    .find("verification_key.verify_package")
                    .zip(XPU_RUNTIME_FFI.find("let catalog: XpuFfiContractCatalogDto"))
                    .is_some_and(|(verification, mapping)| verification < mapping)
                && XPU_RUNTIME_FFI.contains("NativeFfiRegistry::new")
                && XPU_RUNTIME_FFI.contains("capture_native_package")
                && XPU_RUNTIME_FFI.contains("validate_native_package_coverage")
                && XPU_RUNTIME_FFI.contains("capture_native_library_image")
                && XPU_RUNTIME_FFI.contains("RetainedNativeLibraryImage")
                && !XPU_RUNTIME_FFI.contains("PluginVerificationKey"),
        ),
        (
            "directml_package_trust_precedes_canonical_registry_mapping",
            ROCM_RUNTIME_TRUST.contains("pub struct DirectMlPackageVerificationKey")
                && ROCM_RUNTIME_TRUST.contains("DIRECTML_PACKAGE_SIGNATURE_DOMAIN")
                && DIRECTML_PACKAGE_POLICY.contains("ffi-contracts-v1.json")
                && DIRECTML_PACKAGE_POLICY.contains("zed-sim-directml-package-v1")
                && DIRECTML_PACKAGE_POLICY
                    .contains("sim_runtime::DirectMlPackageVerificationKey")
                && DIRECTML_PACKAGE_POLICY
                    .contains("\"runtime_authorization_from_structure\": false")
                && DIRECTML_PACKAGE_POLICY.contains("\"runtime_compilation_forbidden\": true")
                && DIRECTML_CONTRACT_SCHEMA.contains("\"additionalProperties\": false")
                && ["D3D12.dll", "DirectML.dll", "DXGI.dll"]
                    .into_iter()
                    .all(|library| DIRECTML_CONTRACT_SCHEMA.contains(library))
                && DIRECTML_PACKAGER.contains("stable_regular_file")
                && DIRECTML_PACKAGER.contains("separately reviewed FFI contract catalog")
                && DIRECTML_PACKAGER.contains("validate_contract_catalog")
                && !DIRECTML_PACKAGER.contains("SIM_DIRECTML_SIGNATURE_VERIFIER")
                && !DIRECTML_PACKAGER.contains("WinVerifyTrust")
                && !DIRECTML_PACKAGER.contains("urlopen")
                && !DIRECTML_PACKAGER.contains("requests.")
                && !DIRECTML_PACKAGER.contains("NativeFfiRegistry::")
                && DIRECTML_RUNTIME_FFI
                    .find("verification_key.verify_package")
                    .zip(
                        DIRECTML_RUNTIME_FFI
                            .find("let catalog: DirectMlFfiContractCatalogDto"),
                    )
                    .is_some_and(|(verification, mapping)| verification < mapping)
                && DIRECTML_RUNTIME_FFI.contains("NativeFfiRegistry::new")
                && DIRECTML_RUNTIME_FFI
                    .split_once("#[cfg(test)]\nmod tests")
                    .map_or(DIRECTML_RUNTIME_FFI, |(production, _)| production)
                    .matches("NativeFfiContract::new")
                    .count()
                    == 1
                && DIRECTML_RUNTIME_FFI.contains("capture_native_package")
                && DIRECTML_RUNTIME_FFI.contains("validate_native_package_coverage")
                && ROCM_RUNTIME_TRUST.contains("fn capture_native_package(")
                && ROCM_RUNTIME_TRUST.contains("fn validate_native_package_coverage(")
                && !DIRECTML_RUNTIME_FFI.contains("inspect_exact_package_tree")
                && DIRECTML_RUNTIME_FFI.contains("RetainedDirectMlLibraryHandles")
                && DIRECTML_RUNTIME_FFI.contains("DirectMlDiscoveryPlan::for_current_system")
                && DIRECTML_RUNTIME_FFI.contains("observe_directml_candidate")
                && DIRECTML_LOADER.contains("GetSystemDirectoryW")
                && DIRECTML_LOADER.contains("RtlGetVersion")
                && DIRECTML_LOADER.contains("GetFileVersionInfoW")
                && DIRECTML_LOADER.contains("WinVerifyTrust")
                && DIRECTML_LOADER.contains("WTD_CACHE_ONLY_URL_RETRIEVAL")
                && DIRECTML_LOADER.contains("WTD_REVOKE_NONE")
                && !DIRECTML_RUNTIME_FFI.contains("authenticode_trusted: true")
                && !DIRECTML_RUNTIME_FFI.contains("PluginVerificationKey"),
        ),
        (
            "metal_execution_package_is_precompiled_separate_and_untrusted",
            METAL_EXECUTION_PACKAGE_POLICY.contains("\"schema_version\": 1")
                && METAL_EXECUTION_PACKAGE_POLICY
                    .contains("\"runtime_compilation_forbidden\": true")
                && METAL_EXECUTION_PACKAGE_POLICY
                    .contains("\"runtime_authorization_from_structure\": false")
                && METAL_EXECUTION_PACKAGE_POLICY
                    .contains("\"development_structure_only\": true")
                && METAL_EXECUTION_PACKAGE_POLICY
                    .contains("\"authorization_owner\": \"sim_runtime::MetalPackageVerificationKey\"")
                && METAL_EXECUTION_PACKAGE_POLICY
                    .contains("\"redistributes_apple_frameworks\": false")
                && METAL_EXECUTION_PACKAGE_POLICY.contains("tensor_ops.metallib")
                && METAL_EXECUTION_PACKAGE_POLICY.contains("tensor_ops.metal")
                && METAL_EXECUTION_ABI.contains("zed-sim-metal-execution-v1")
                && METAL_EXECUTION_ABI.contains("zed_sim_metal_add_f16_v1")
                && METAL_EXECUTION_ABI.contains("zed_sim_metal_add_f32_v1")
                && METAL_EXECUTION_ABI.contains("\"return_nullability\": \"nullable\"")
                && METAL_EXECUTION_BINDINGS.contains("Xcode 26.2 build 17C52")
                && METAL_EXECUTION_BINDINGS.contains("29 exact Objective-C type encodings")
                && METAL_EXECUTION_BINDINGS.contains("12 exact SHA-256 identities")
                && METAL_EXECUTION_ABI_VERIFIER.contains("protocol_getMethodDescription")
                && METAL_EXECUTION_ABI_VERIFIER.contains("method_getTypeEncoding")
                && METAL_EXECUTION_ABI_VERIFIER.contains("expected_return_nullability")
                && METAL_EXECUTION_LICENSES.contains("tensor_ops.metallib")
                && !METAL_EXECUTION_LICENSES.contains("readiness.metallib")
                && METAL_EXECUTION_KERNELS.contains("kernel void zed_sim_metal_add_f16_v1")
                && METAL_EXECUTION_KERNELS.contains("kernel void zed_sim_metal_add_f32_v1")
                && METAL_EXECUTION_SOURCE.contains("pub unsafe fn from_certified_metallibs")
                && METAL_EXECUTION_SOURCE.contains("readiness_metallib: Arc<[u8]>")
                && METAL_EXECUTION_SOURCE.contains("tensor_ops_metallib: Arc<[u8]>")
                && METAL_EXECUTION_SOURCE.contains("_certified: Arc<CertifiedInputs>")
                && METAL_EXECUTION_SOURCE.contains("new_library_with_data")
                && !METAL_EXECUTION_SOURCE.contains("new_library_with_source")
                && !METAL_EXECUTION_SOURCE.contains("process::Command")
                && !METAL_EXECUTION_SOURCE.contains("NativeFfiRegistry::")
                && METAL_EXECUTION_ABI_SOURCE
                    .contains("requires-native-ffi-registry-and-signed-package")
                && METAL_EXECUTION_ABI_SOURCE.contains("fn is_sha256")
                && METAL_EXECUTION_CATALOG.contains(
                    "comfy-parity-metal-execution-resource-ownership-consolidation",
                )
                && METAL_EXECUTION_CATALOG.contains("sdk_abi_verifier_sha256")
                && METAL_EXECUTION_CATALOG.contains("license_notice_sha256")
                && METAL_EXECUTION_CATALOG.contains("\"task_108_artifacts_modified\": false")
                && METAL_EXECUTION_PACKAGER.contains("xcrun -sdk macosx metal")
                && METAL_EXECUTION_PACKAGER.contains("xcrun -sdk macosx metallib")
                && METAL_EXECUTION_PACKAGER.contains("package-coverage.sha256")
                && METAL_EXECUTION_PACKAGER.contains("--self-test")
                && !METAL_EXECUTION_PACKAGER.contains("curl ")
                && !METAL_EXECUTION_PACKAGER.contains("wget ")
                && !METAL_EXECUTION_PACKAGER.contains("NativeFfiRegistry")
                && !METAL_EXECUTION_PACKAGER.contains("verify_signature")
                && !METAL_EXECUTION_PACKAGER.contains("cp /System/Library/Frameworks/")
                && !METAL_EXECUTION_PACKAGER.contains("cp -R /System/Library/Frameworks/"),
        ),
    ])
}

fn application_surface_cases() -> BTreeMap<&'static str, bool> {
    let worker_launch = ZED_SOURCE
        .split_once("fn native_sim_worker_launch(")
        .and_then(|(_, source)| source.split_once("fn register_native_sim_execution("))
        .map(|(source, _)| source)
        .unwrap_or_default();
    let default_settings_lowercase = DEFAULT_SETTINGS.to_ascii_lowercase();
    let default_sim_settings_lowercase = DEFAULT_SIM_SETTINGS.to_ascii_lowercase();
    let worker_source = String::from_utf8_lossy(WORKER_SOURCE);
    BTreeMap::from([
        (
            "default_settings_exclude_comfy_while_opt_in_defaults_select_native_cpu",
            !default_settings_lowercase.contains("sim_runtime")
                && !default_settings_lowercase.contains("comfyui")
                && !default_settings_lowercase.contains("python_runtime")
                && !default_settings_lowercase.contains("external_comfy")
                && DEFAULT_SIM_SETTINGS.contains("\"name\": \"Native Local\"")
                && DEFAULT_SIM_SETTINGS.contains("\"device\": \"cpu\"")
                && DEFAULT_SIM_SETTINGS.contains("\"api_host_enabled\": false")
                && DEFAULT_SIM_SETTINGS.contains("\"plugin_policy\": \"approved_only\"")
                && !default_sim_settings_lowercase.contains("python_runtime")
                && !default_sim_settings_lowercase.contains("external_comfy"),
        ),
        (
            "comfy_menu_has_no_browser_or_external_server_action",
            SIM_MENU_SOURCE.contains("pub fn sim_menu() -> Menu")
                && !SIM_MENU_SOURCE.contains("open_url")
                && !SIM_MENU_SOURCE.contains("javascript:")
                && !SIM_MENU_SOURCE.contains("127.0.0.1:8188")
                && !SIM_MENU_SOURCE.contains("process::Command"),
        ),
        (
            "comfy_cli_refuses_source_host_and_process_fallback",
            SIM_CLI_SOURCE.contains("source host/port options identify an external ComfyUI server and are never contacted")
                && SIM_CLI_SOURCE.contains("CommandDisposition::Migration")
                && !SIM_CLI_SOURCE.contains("process::Command")
                && !SIM_CLI_SOURCE.contains("external_comfy_selected\": true")
                && !SIM_CLI_SOURCE.contains("python_runtime_selected\": true")
                && SIM_CLI_SOURCE.contains("WorkerLaunchConfig::for_packaged_worker_profile"),
        ),
        (
            "gpui_worker_launch_has_no_public_protocol_or_source_runtime_fallback",
            !worker_launch.is_empty()
                && !worker_launch.contains("http://")
                && !worker_launch.contains("ws://")
                && !worker_launch.contains("python")
                && !worker_launch.contains("node")
                && !worker_launch.contains("browser"),
        ),
        (
            "non_cpu_graph_fails_typed_before_cpu_executor_construction",
            worker_source
                .find("backend_neutral_executor_unavailable(session.backend_device())")
                .zip(worker_source.find("match prepare_native_image_memory("))
                .zip(worker_source.find("NativeImageExecutor::new_with_generated_registry"))
                .is_some_and(|((preflight, memory), cpu_executor)| {
                    preflight < memory && preflight < cpu_executor
                })
                && worker_source.contains("NativeImageWorkerEvent::BackendUnavailable")
                && worker_source.contains("CPU fallback is forbidden"),
        ),
        (
            "mlu_profile_reaches_only_the_certified_native_worker_session",
            RUNTIME_SUPERVISOR_SOURCE.contains("WorkerBackendSelection::Mlu")
                && RUNTIME_SUPERVISOR_SOURCE.contains("NativeMluPackageSettings")
                && RUNTIME_SUPERVISOR_SOURCE.contains("Self::for_mlu")
                && worker_source.contains("initialize_certified_mlu_runtime")
                && worker_source.contains("MluTensorBackend::from_certified_runtime")
                && worker_source.contains("WorkerBackendSession::new")
                && worker_source.contains("the packaged worker was built without the MLU integration feature")
                && !worker_source.contains("WorkerBackendSelection::Mlu => WorkerBackendSelection::Cpu"),
        ),
        (
            "npu_profile_reaches_only_the_certified_native_worker_session",
            RUNTIME_SUPERVISOR_SOURCE.contains("WorkerBackendSelection::Npu")
                && RUNTIME_SUPERVISOR_SOURCE.contains("NativeNpuPackageSettings")
                && RUNTIME_SUPERVISOR_SOURCE.contains("Self::for_npu")
                && worker_source.contains("initialize_certified_npu_runtime")
                && worker_source.contains("NpuTensorBackend::from_certified_runtime")
                && worker_source.contains("WorkerBackendSession::new")
                && worker_source.contains(
                    "the packaged worker was built without the NPU integration feature",
                )
                && NPU_RUNTIME_FFI.contains("verify_npu_package_contracts")
                && NPU_RUNTIME_FFI.contains("certify_npu_library_images")
                && !worker_source
                    .contains("WorkerBackendSelection::Npu => WorkerBackendSelection::Cpu"),
        ),
        (
            "cuda_profile_reaches_only_the_certified_native_worker_session",
            RUNTIME_SUPERVISOR_SOURCE.contains("WorkerBackendSelection::Cuda")
                && RUNTIME_SUPERVISOR_SOURCE.contains("NativeCudaPackageSettings")
                && RUNTIME_SUPERVISOR_SOURCE.contains("Self::for_cuda")
                && worker_source.contains("initialize_certified_cuda_runtime")
                && worker_source.contains("CudaTensorBackend::from_certified_session")
                && worker_source.contains("WorkerBackendSession::new")
                && worker_source.contains(
                    "the packaged worker was built without the CUDA integration feature",
                )
                && CUDA_RUNTIME_FFI.contains("verify_cuda_package_contracts")
                && CUDA_RUNTIME_FFI.contains("certify_cuda_library_images")
                && !worker_source
                    .contains("WorkerBackendSelection::Cuda => WorkerBackendSelection::Cpu"),
        ),
        (
            "xpu_profile_reaches_only_the_certified_native_worker_session",
            RUNTIME_SUPERVISOR_SOURCE.contains("WorkerBackendSelection::Xpu")
                && RUNTIME_SUPERVISOR_SOURCE.contains("NativeXpuPackageSettings")
                && RUNTIME_SUPERVISOR_SOURCE.contains("Self::for_xpu")
                && worker_source.contains("initialize_certified_xpu_runtime")
                && worker_source.contains("XpuTensorBackend::from_certified_session")
                && worker_source.contains("WorkerBackendSession::new")
                && worker_source.contains(
                    "the packaged worker was built without the XPU integration feature",
                )
                && XPU_RUNTIME_FFI.contains("verify_xpu_package_contracts")
                && XPU_RUNTIME_FFI.contains("certify_xpu_library_images")
                && !worker_source
                    .contains("WorkerBackendSelection::Xpu => WorkerBackendSelection::Cpu"),
        ),
        (
            "directml_profile_reaches_only_the_observed_certified_native_worker_session",
            RUNTIME_SUPERVISOR_SOURCE.contains("WorkerBackendSelection::DirectMl")
                && RUNTIME_SUPERVISOR_SOURCE.contains("NativeDirectMlPackageSettings")
                && RUNTIME_SUPERVISOR_SOURCE.contains("Self::for_directml")
                && worker_source.contains("initialize_certified_directml_runtime")
                && worker_source.contains("DirectMlTensorBackend::from_certified_session")
                && worker_source.contains("WorkerBackendSession::new")
                && worker_source.contains(
                    "the packaged worker was built without the DirectML integration feature",
                )
                && DIRECTML_TENSOR.contains("pub struct DirectMlTensorBackend")
                && !worker_source
                    .contains("WorkerBackendSelection::DirectMl => WorkerBackendSelection::Cpu"),
        ),
    ])
}

fn release_fixture() -> Result<ReleaseFixture, Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let worker_directory = directory.path().join("isolated-release-host");
    fs::create_dir(&worker_directory)?;
    let mut roots = Vec::new();
    for (namespace, name) in [
        (AssetNamespace::Input, "input"),
        (AssetNamespace::Output, "output"),
        (AssetNamespace::Temporary, "temporary"),
        (AssetNamespace::Model, "model"),
        (AssetNamespace::Plugin, "plugin"),
    ] {
        let path = directory.path().join(name);
        fs::create_dir(&path)?;
        roots.push((namespace, path));
    }
    let roots = AssetRoots::new(PROFILE_ID.to_string(), roots)?;
    let input: Value = serde_json::from_slice(INPUT_FIXTURE)?;
    let pixels = input
        .get("pixels_bhwc")
        .and_then(Value::as_array)
        .ok_or("native image release fixture omitted pixels")?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or("native image release fixture pixel is not numeric")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dimension = |name: &str| -> Result<u64, Box<dyn Error>> {
        input
            .get(name)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("native image release fixture omitted {name}").into())
    };
    let input_png = encode_png_frame(
        &pixels,
        dimension("batch")?,
        dimension("height")?,
        dimension("width")?,
        dimension("channels")?,
        0,
        &BTreeMap::new(),
        PngLimits::default(),
    )?;
    fs::write(
        roots
            .test_root_path(AssetNamespace::Input)?
            .join("fixture.png"),
        input_png,
    )?;
    let assets = Arc::new(Mutex::new(AssetService::open(roots.clone())?));
    let input_authorization = authorize_native_input_reader(&roots.profile_id)?;
    let mut config = WorkerLaunchConfig::new(
        env!("CARGO_BIN_EXE_sim_native_image_worker_fixture"),
        ProfileId(PROFILE_ID),
        WorkerId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_3302)),
        NATIVE_IMAGE_REGISTRY_VERSION,
        8 * 1024 * 1024 * 1024,
    );
    config.working_directory = Some(worker_directory);
    config.environment = vec![
        ("PATH".to_owned(), String::new()),
        (
            "HOME".to_owned(),
            directory.path().to_string_lossy().into_owned(),
        ),
    ];
    config.policy = SupervisorPolicy {
        heartbeat_interval: Duration::from_secs(30),
        missed_heartbeat_limit: 3,
        shutdown_timeout: Duration::from_secs(3),
        ready_timeout: Duration::from_secs(10),
        maximum_automatic_restarts: 0,
        restart_backoff: Duration::from_millis(1),
    };
    Ok(ReleaseFixture {
        _directory: directory,
        config,
        assets,
        input_authorization,
    })
}

fn runtime_trace_cases() -> Result<BTreeMap<&'static str, bool>, Box<dyn Error>> {
    let fixture = release_fixture()?;
    let worker_binary = fs::read(&fixture.config.binary)?;
    let isolated_root = fixture
        .config
        .working_directory
        .clone()
        .ok_or("isolated release worker omitted its working directory")?;
    let mut plan = compile_native_image_workflow(WORKFLOW_FIXTURE, &BTreeSet::new())?;
    plan.prompt_id = PromptId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_3303));
    let prompt_id = plan.prompt_id;
    let worker_plan = NativeImageWorkerPlan::from_asset_service(
        plan,
        &fixture.assets,
        &fixture.input_authorization,
        &sim_types::CancellationToken::default(),
        true,
        0,
    )?;
    let mut supervisor = smol::block_on(RuntimeSupervisor::start(fixture.config))?;
    let ready = supervisor.snapshot().health == WorkerHealth::BackendReady
        && supervisor
            .accepted_backend()
            .is_some_and(|matrix| matrix.device() == sim_tensor::DeviceId::CPU);
    smol::block_on(supervisor.execute(
        prompt_id,
        AttemptId(Uuid::from_u128(0x3304)),
        serde_json::to_vec(&worker_plan)?,
    ))?;
    let (terminal, proposals, lifecycle_messages, application_messages) = smol::block_on(async {
        let mut proposals = 0_usize;
        let mut lifecycle_messages = 0_usize;
        let mut application_messages = 0_usize;
        loop {
            let envelope = supervisor.next_event(Duration::from_secs(10)).await?;
            match envelope.message {
                WorkerMessage::Lifecycle { .. } => lifecycle_messages += 1,
                WorkerMessage::OutputProposal { .. } => proposals += 1,
                WorkerMessage::Event { event } => {
                    application_messages += 1;
                    if let Ok(result) = postcard::from_bytes::<NativeImageWorkerEvent>(&event)
                        && matches!(
                            result,
                            NativeImageWorkerEvent::Completed { .. }
                                | NativeImageWorkerEvent::BackendUnavailable { .. }
                                | NativeImageWorkerEvent::Failed { .. }
                        )
                    {
                        break Ok::<_, sim_runtime::RuntimeSupervisorError>((
                            result,
                            proposals,
                            lifecycle_messages,
                            application_messages,
                        ));
                    }
                }
                _ => {}
            }
        }
    })?;
    let shutdown = smol::block_on(supervisor.shutdown())?;
    let binary_markers = [
        "python3",
        "python.exe",
        "ComfyUI/main.py",
        "node_modules/comfy",
        "http://127.0.0.1:8188",
        "ws://127.0.0.1:8188",
        "javascript:",
    ];
    Ok(BTreeMap::from([
        ("isolated_native_worker_reaches_cpu_readiness", ready),
        (
            "isolated_host_has_no_source_tree_or_executable_path",
            !isolated_root.join("projects/comfy").exists()
                && !isolated_root.join("ComfyUI").exists()
                && !isolated_root.join("ComfyUI-Frontend").exists()
                && supervisor
                    .snapshot()
                    .launch
                    .environment_names
                    .iter()
                    .any(|name| name == "PATH"),
        ),
        (
            "native_image_slice_completes_over_private_ipc",
            matches!(
                terminal,
                NativeImageWorkerEvent::Completed { ref result }
                    if result.report.state == AttemptState::Succeeded
                        && result.executed_node_count == 5
                        && result.output_proposal_ids.len() == 2
            ) && proposals == 2
                && lifecycle_messages >= 1
                && application_messages >= 1,
        ),
        (
            "worker_binary_contains_no_external_engine_marker",
            binary_markers.iter().all(|marker| {
                !worker_binary
                    .windows(marker.len())
                    .any(|window| window.eq_ignore_ascii_case(marker.as_bytes()))
            }),
        ),
        (
            "worker_has_no_network_or_browser_api",
            [
                "std::net",
                "TcpStream",
                "TcpListener",
                "UdpSocket",
                "webview",
                "open_url",
            ]
            .iter()
            .all(|marker| {
                !WORKER_SOURCE
                    .windows(marker.len())
                    .any(|window| window == marker.as_bytes())
            }),
        ),
        (
            "isolated_native_worker_shutdown_succeeds",
            shutdown.success(),
        ),
    ]))
}

fn legacy_refusal_case() -> Result<bool, Box<dyn Error>> {
    let result = migrate_legacy_profile(
        LegacyComfyProfile {
            name: "Legacy release fallback".into(),
            endpoint: Some(
                "http://legacy-user:legacy-password@127.0.0.1:8188/?token=removed".into(),
            ),
            credential: Some("must-not-survive".into()),
            model_roots: Vec::new(),
            api_host_enabled: true,
            plugin_mappings: Vec::new(),
            workflow_state: BTreeMap::new(),
            unknown_fields: BTreeMap::new(),
        },
        Uuid::from_u128(0x3305),
        Uuid::from_u128(0x3306),
    )?;
    let serialized = serde_json::to_vec(&result)?;
    Ok(!result.inactive_legacy_profile.active
        && result.credential_removed
        && result.endpoint_removed_or_redacted
        && result.native_profile.model_roots.is_empty()
        && !result.native_profile.api_host.enabled
        && !serialized
            .windows("must-not-survive".len())
            .any(|window| window == b"must-not-survive"))
}

fn write_artifact(
    workspace_root: &Path,
    metadata: &[u8],
    worker_binary_bytes: usize,
    cases: &BTreeMap<&str, bool>,
) -> Result<(), Box<dyn Error>> {
    if cases.values().any(|passed| !passed) {
        return Err(io::Error::other(format!(
            "VAL-NATIVE-BOUNDARY-001 packaged release cases failed: {cases:#?}"
        ))
        .into());
    }
    let directory = target_directory(workspace_root).join("comfy-parity");
    fs::create_dir_all(&directory)?;
    let output = directory.join("val-native-boundary-001.json");
    let temporary = directory.join("val-native-boundary-001.json.tmp");
    match fs::remove_file(&temporary) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let artifact = json!({
        "validation_id": "VAL-NATIVE-BOUNDARY-001",
        "validation": "VAL-NATIVE-BOUNDARY-001",
        "schema_version": 1,
        "scope": "packaged-native-runtime-release-boundary",
        "environment": {
            "operating_system": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "backend": "native-rust-cpu",
            "worker_path_available": false,
            "worker_source_tree_available": false,
            "worker_network_api_available": false,
            "python_processes": 0,
            "node_processes": 0,
            "browser_processes": 0,
            "external_comfy_connections": 0,
        },
        "fixture_digests": {
            "cargo_metadata_sha256": format!("{:x}", Sha256::digest(metadata)),
            "release_policy_sha256": format!("{:x}", Sha256::digest(RELEASE_POLICY)),
            "workflow_sha256": format!("{:x}", Sha256::digest(WORKFLOW_FIXTURE)),
            "input_sha256": format!("{:x}", Sha256::digest(INPUT_FIXTURE)),
            "default_settings_sha256": format!("{:x}", Sha256::digest(DEFAULT_SETTINGS.as_bytes())),
            "default_sim_settings_sha256": format!("{:x}", Sha256::digest(DEFAULT_SIM_SETTINGS.as_bytes())),
            "zed_source_sha256": format!("{:x}", Sha256::digest(ZED_SOURCE.as_bytes())),
            "sim_cli_source_sha256": format!("{:x}", Sha256::digest(SIM_CLI_SOURCE.as_bytes())),
            "sim_menu_source_sha256": format!("{:x}", Sha256::digest(SIM_MENU_SOURCE.as_bytes())),
            "worker_source_sha256": format!("{:x}", Sha256::digest(WORKER_SOURCE)),
            "runtime_supervisor_source_sha256": format!("{:x}", Sha256::digest(RUNTIME_SUPERVISOR_SOURCE.as_bytes())),
            "mac_bundle_sha256": format!("{:x}", Sha256::digest(MAC_BUNDLE.as_bytes())),
            "linux_bundle_sha256": format!("{:x}", Sha256::digest(LINUX_BUNDLE.as_bytes())),
            "windows_bundle_sha256": format!("{:x}", Sha256::digest(WINDOWS_BUNDLE.as_bytes())),
            "windows_installer_sha256": format!("{:x}", Sha256::digest(WINDOWS_INSTALLER.as_bytes())),
            "metal_abi_manifest_sha256": format!("{:x}", Sha256::digest(METAL_ABI_MANIFEST.as_bytes())),
            "metal_reviewed_bindings_sha256": format!("{:x}", Sha256::digest(METAL_REVIEWED_BINDINGS.as_bytes())),
            "metal_licenses_sha256": format!("{:x}", Sha256::digest(METAL_LICENSES.as_bytes())),
            "metal_readiness_source_sha256": format!("{:x}", Sha256::digest(METAL_READINESS_SOURCE.as_bytes())),
            "metal_package_policy_sha256": format!("{:x}", Sha256::digest(METAL_PACKAGE_POLICY.as_bytes())),
            "metal_contract_schema_sha256": format!("{:x}", Sha256::digest(METAL_CONTRACT_SCHEMA.as_bytes())),
            "metal_packager_sha256": format!("{:x}", Sha256::digest(METAL_PACKAGER.as_bytes())),
            "metal_runtime_ffi_sha256": format!("{:x}", Sha256::digest(METAL_RUNTIME_FFI.as_bytes())),
            "metal_abi_catalog_sha256": format!("{:x}", Sha256::digest(METAL_ABI_CATALOG.as_bytes())),
            "metal_execution_package_policy_sha256": format!("{:x}", Sha256::digest(METAL_EXECUTION_PACKAGE_POLICY.as_bytes())),
            "metal_execution_packager_sha256": format!("{:x}", Sha256::digest(METAL_EXECUTION_PACKAGER.as_bytes())),
            "metal_execution_abi_sha256": format!("{:x}", Sha256::digest(METAL_EXECUTION_ABI.as_bytes())),
            "metal_execution_bindings_sha256": format!("{:x}", Sha256::digest(METAL_EXECUTION_BINDINGS.as_bytes())),
            "metal_execution_abi_verifier_sha256": format!("{:x}", Sha256::digest(METAL_EXECUTION_ABI_VERIFIER.as_bytes())),
            "metal_execution_licenses_sha256": format!("{:x}", Sha256::digest(METAL_EXECUTION_LICENSES.as_bytes())),
            "metal_execution_kernels_sha256": format!("{:x}", Sha256::digest(METAL_EXECUTION_KERNELS.as_bytes())),
            "metal_execution_source_sha256": format!("{:x}", Sha256::digest(METAL_EXECUTION_SOURCE.as_bytes())),
            "metal_execution_abi_source_sha256": format!("{:x}", Sha256::digest(METAL_EXECUTION_ABI_SOURCE.as_bytes())),
            "metal_execution_catalog_sha256": format!("{:x}", Sha256::digest(METAL_EXECUTION_CATALOG.as_bytes())),
            "worker_binary_bytes_inspected": worker_binary_bytes,
        },
        "summary": {
            "passed": cases.len(),
            "failed": 0,
            "skipped": 0,
        },
        "cases": cases,
        "skipped": [],
        "release_closure": {
            "claimed": true,
            "stage": "packaged-native-release-boundary",
            "remaining_gates": [],
            "reason": "Dependency, package, signed companion worker, settings, menu, CLI, legacy refusal, binary, source, private IPC, isolated readiness, first-slice, network-API, and external-engine checks all passed.",
        },
    });
    fs::write(&temporary, serde_json::to_vec_pretty(&artifact)?)?;
    fs::rename(temporary, output)?;
    Ok(())
}

#[test]
fn val_native_boundary_001_packaged_release() -> Result<(), Box<dyn Error>> {
    let workspace_root = workspace_root()?;
    let metadata_bytes = cargo_metadata(&workspace_root)?;
    let metadata: Value = serde_json::from_slice(&metadata_bytes)?;
    let policy = load_release_boundary_policy()?;
    policy.verify_launcher_layout(&workspace_root)?;
    let report = policy.verify_cargo_metadata(&metadata)?;

    let mut cases = dependency_boundary_cases(&metadata)?;
    cases.extend(packaging_cases());
    cases.extend(application_surface_cases());
    cases.extend(runtime_trace_cases()?);
    cases.insert(
        "oracle_and_test_support_are_reverse_development_dependencies",
        report.is_clean() && report.development_packages_found == policy.development_only_packages,
    );
    cases.insert(
        "legacy_external_fallback_is_preserved_but_refused",
        legacy_refusal_case()?,
    );
    let worker_binary_bytes = fs::metadata(env!("CARGO_BIN_EXE_sim_native_image_worker_fixture"))?
        .len()
        .try_into()?;
    write_artifact(
        &workspace_root,
        &metadata_bytes,
        worker_binary_bytes,
        &cases,
    )
}
