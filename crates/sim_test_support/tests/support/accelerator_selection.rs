use std::collections::BTreeMap;

const WORKER_SOURCE: &str = include_str!("../../../sim_worker/src/sim_worker.rs");
const SUPERVISOR_SOURCE: &str = include_str!("../../../sim_runtime/src/runtime_supervisor.rs");
const DIRECTML_RUNTIME_SOURCE: &str =
    include_str!("../../../sim_runtime/src/native_ffi_directml.rs");
const DIRECTML_LOADER_SOURCE: &str = include_str!("../../../sim_backend_directml/src/loader.rs");
const NPU_RUNTIME_SOURCE: &str = include_str!("../../../sim_runtime/src/native_ffi_npu.rs");
const CUDA_RUNTIME_SOURCE: &str = include_str!("../../../sim_runtime/src/native_ffi_cuda.rs");
const XPU_RUNTIME_SOURCE: &str = include_str!("../../../sim_runtime/src/native_ffi_xpu.rs");

pub(crate) fn accelerator_selection_contract_cases() -> BTreeMap<&'static str, bool> {
    let mut cases = BTreeMap::new();
    let directml_branch = source_between(
        WORKER_SOURCE,
        "WorkerBackendSelection::DirectMl { package } =>",
        "WorkerBackendSelection::Rocm",
    );
    let mlu_branch = source_between(
        WORKER_SOURCE,
        "WorkerBackendSelection::Mlu {",
        "WorkerBackendSelection::Npu {",
    );
    let npu_branch = source_between(
        WORKER_SOURCE,
        "WorkerBackendSelection::Npu {",
        "WorkerBackendSelection::Cuda {",
    );
    let cuda_branch = source_between(
        WORKER_SOURCE,
        "WorkerBackendSelection::Cuda {",
        "WorkerBackendSelection::Xpu {",
    );
    let xpu_branch = source_between(WORKER_SOURCE, "WorkerBackendSelection::Xpu {", "\n    }\n}");

    cases.insert(
        "directml_selection_constructs_only_the_certified_semantic_session",
        directml_branch.is_some_and(|branch| {
            branch.contains("initialize_certified_directml_runtime")
                && branch.contains("DirectMlTensorBackend::from_certified_session")
                && branch.contains("WorkerBackendSession::new")
                && branch.contains("(initialized, None)")
                && !branch.contains("WorkerBackendSession::cpu")
                && !branch.contains("WorkerBackendSelection::Cpu")
        }),
    );
    cases.insert(
        "mlu_selection_constructs_only_the_certified_semantic_session",
        mlu_branch.is_some_and(|branch| {
            branch.contains("initialize_certified_mlu_runtime")
                && branch.contains("MluTensorBackend::from_certified_runtime")
                && branch.contains("WorkerBackendSession::new")
                && branch.contains("(initialized, None)")
                && !branch.contains("WorkerBackendSession::cpu")
                && !branch.contains("WorkerBackendSelection::Cpu")
        }),
    );
    cases.insert(
        "npu_selection_constructs_only_the_certified_semantic_session",
        npu_branch.is_some_and(|branch| {
            branch.contains("initialize_certified_npu_runtime")
                && branch.contains("device_ordinal")
                && branch.contains("NpuTensorBackend::from_certified_runtime")
                && branch.contains("WorkerBackendSession::new")
                && branch.contains("(initialized, None)")
                && !branch.contains("WorkerBackendSession::cpu")
                && !branch.contains("WorkerBackendSelection::Cpu")
        }),
    );
    cases.insert(
        "npu_initialization_verifies_trust_before_discovery_certification_and_loader_entry",
        source_after(
            NPU_RUNTIME_SOURCE,
            "pub fn initialize_certified_npu_runtime_with_discovery",
        )
        .is_some_and(|initializer| {
            ordered(
                initializer,
                &[
                    "verify_npu_package_contracts",
                    "certify_npu_library_images",
                    "load_execution_runtime",
                ],
            )
        }),
    );
    cases.insert(
        "cuda_selection_constructs_only_the_certified_semantic_session",
        cuda_branch.is_some_and(|branch| {
            branch.contains("initialize_certified_cuda_runtime")
                && branch.contains("device_ordinal")
                && branch.contains("CudaTensorBackend::from_certified_session")
                && branch.contains("WorkerBackendSession::new")
                && branch.contains("(initialized, None)")
                && !branch.contains("WorkerBackendSession::cpu")
                && !branch.contains("WorkerBackendSelection::Cpu")
        }),
    );
    cases.insert(
        "cuda_initialization_verifies_trust_before_discovery_certification_and_loader_entry",
        source_after(
            CUDA_RUNTIME_SOURCE,
            "pub fn initialize_certified_cuda_runtime_with_candidates",
        )
        .is_some_and(|initializer| {
            ordered(
                initializer,
                &[
                    "verify_cuda_package_contracts",
                    "certify_cuda_library_images",
                    "load_execution_runtime",
                ],
            )
        }),
    );
    cases.insert(
        "xpu_selection_constructs_only_the_certified_semantic_session",
        xpu_branch.is_some_and(|branch| {
            branch.contains("initialize_certified_xpu_runtime")
                && branch.contains("device_ordinal")
                && branch.contains("XpuTensorBackend::from_certified_session")
                && branch.contains("WorkerBackendSession::new")
                && branch.contains("(initialized, None)")
                && !branch.contains("WorkerBackendSession::cpu")
                && !branch.contains("WorkerBackendSelection::Cpu")
        }),
    );
    cases.insert(
        "xpu_initialization_verifies_trust_before_discovery_certification_and_loader_entry",
        source_after(
            XPU_RUNTIME_SOURCE,
            "pub fn initialize_certified_xpu_runtime_with_discovery",
        )
        .is_some_and(|initializer| {
            ordered(
                initializer,
                &[
                    "verify_xpu_package_contracts",
                    "certify_xpu_library_images",
                    "load_execution_runtime",
                ],
            )
        }),
    );
    cases.insert(
        "directml_initialization_observes_the_real_host_after_package_trust",
        source_after(
            DIRECTML_RUNTIME_SOURCE,
            "pub fn initialize_certified_directml_runtime",
        )
        .is_some_and(|initializer| {
            ordered(
                initializer,
                &[
                    "verify_directml_package_contracts",
                    "DirectMlDiscoveryPlan::for_current_system",
                    "observe_directml_candidate",
                    "certify_directml_library_images",
                    "load_execution_session",
                ],
            )
        }) && DIRECTML_LOADER_SOURCE.contains("GetSystemDirectoryW")
            && DIRECTML_LOADER_SOURCE.contains("RtlGetVersion")
            && DIRECTML_LOADER_SOURCE.contains("GetFileVersionInfoW")
            && DIRECTML_LOADER_SOURCE.contains("WinVerifyTrust")
            && !DIRECTML_RUNTIME_SOURCE.contains("authenticode_trusted: true"),
    );
    cases.insert(
        "replacement_worker_reuses_the_selected_launch_contract_and_recertifies",
        SUPERVISOR_SOURCE.contains("Self::start(self.launch_config.clone()).await?"),
    );
    cases
}

fn source_after<'a>(source: &'a str, marker: &str) -> Option<&'a str> {
    source.split_once(marker).map(|(_, remainder)| remainder)
}

fn source_between<'a>(source: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let (_, remainder) = source.split_once(start)?;
    let (branch, _) = remainder.split_once(end)?;
    Some(branch)
}

fn ordered(source: &str, needles: &[&str]) -> bool {
    let mut remainder = source;
    for needle in needles {
        let Some((_, suffix)) = remainder.split_once(needle) else {
            return false;
        };
        remainder = suffix;
    }
    true
}
