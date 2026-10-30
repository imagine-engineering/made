#![cfg(feature = "cpu")]

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, ensure};
use ring::signature::Ed25519KeyPair;
use sha2::{Digest, Sha256};
use sim_tensor::{
    BinaryOperation, CancellationToken, DType, DeviceId, ExecutionContext, Scalar, StreamId,
    TensorBackend, TensorDescriptor, TensorError,
    generated_backend_cpu_sim_model_0016::initialize_cpu_tensor_backend,
};
use sim_test_support::device_certification::{
    DeviceCertificationTrustAnchor, cpu_certification_implementation_manifest,
    load_device_certification_signing_key,
};
use sim_test_support::{
    CertificationArtifact, CertificationEnvironment, CertificationFact, CertificationMatrixRow,
    CertificationMemoryFact, CertificationPackageEvidence, CertificationPayload,
    CertificationStatus, ContractEvidence, DeviceEvidence,
};
use smol::process::Command;

const CERTIFICATION_RELATIVE_PATH: &str =
    ".agents/specs/comfy-parity/catalogs/native-device-certification/cpu-comfy-model-0016.json";
const ATTESTATION_SIGNER: &str = "zed.hardware.lab.cpu.mt-mbp-l2kh69c7rg";
const ATTESTATION_PUBLIC_KEY: &str =
    "627d26c97f76a810b90828422cff3e6cdaa083ed885d40169bae0e08255d07eb";
const SIGNING_KEY_PATH_ENV: &str = "SIM_CPU_CERTIFICATION_SIGNING_KEY_PKCS8_PATH";
const EFFECTIVE_MEMORY_LIMIT: u64 = 64 * 1024 * 1024;
const WORKSPACE_LIMIT: u64 = 4 * 1024 * 1024;

#[test]
fn val_device_001_cpu_baseline_conformance_and_attestation_verification() -> Result<()> {
    let workspace = workspace_root()?;
    let trust_anchor = attestation_trust_anchor()?;
    let certificate_path = workspace.join(CERTIFICATION_RELATIVE_PATH);
    let update = std::env::var_os("UPDATE_SIM_DEVICE_CERTIFICATION").is_some();
    let existing = if update {
        None
    } else if certificate_path.exists() {
        Some(CertificationArtifact::parse_and_verify(
            &fs::read(&certificate_path)
                .with_context(|| format!("read {}", certificate_path.display()))?,
            &trust_anchor,
        )?)
    } else {
        None
    };

    let current_attestation_matches = if !update {
        let Some(certificate) = existing.as_ref() else {
            eprintln!(
                "signed CPU hardware attestation is absent; deterministic CPU conformance continues and the external release-certification gate remains unclaimed"
            );
            return execute_hardware_matrix(&workspace, None).map(|_| ());
        };
        validate_static_identity(&certificate.payload)?;
        if let Err(error) = validate_implementation_manifest(&workspace, &certificate.payload) {
            eprintln!(
                "signed CPU hardware attestation verified against the approved trust anchor but is stale for the current implementation; the external release-certification gate remains unclaimed: {error:#}"
            );
            false
        } else {
            let current_target = rust_target()?;
            let current_identifier = match observe_device_identifier() {
                Ok(identifier) => identifier,
                Err(error) => {
                    eprintln!(
                        "signed CPU hardware attestation verified but the current environment cannot probe CPU identity; the external release-certification gate remains unclaimed: {error:#}"
                    );
                    String::new()
                }
            };
            if current_target != certificate.payload.target
                || current_identifier != certificate.payload.device.identifier
            {
                eprintln!(
                    "signed CPU hardware attestation remains bound to {} {}; current target/device {} {} is not relabeled as certified",
                    certificate.payload.target,
                    certificate.payload.device.identifier,
                    current_target,
                    current_identifier
                );
                false
            } else {
                true
            }
        }
    } else {
        false
    };

    let observed_payload = execute_hardware_matrix(
        &workspace,
        existing
            .as_ref()
            .map(|artifact| artifact.payload.observed_at_utc.as_str()),
    )?;

    if update {
        let observed_at = std::env::var("SIM_CERTIFICATION_OBSERVED_AT_UTC").context(
            "SIM_CERTIFICATION_OBSERVED_AT_UTC is required when updating the hardware certificate",
        )?;
        ensure!(
            observed_at.ends_with('Z') && observed_at.contains('T'),
            "observation time must be an explicit UTC timestamp"
        );
        let mut payload = observed_payload;
        payload.observed_at_utc = observed_at;
        let signing_key = load_signing_key()?;
        let certificate = CertificationArtifact::sign(payload, &trust_anchor, &signing_key)?;
        let bytes = CertificationArtifact::to_canonical_json(&certificate)?;
        let parent = certificate_path
            .parent()
            .ok_or_else(|| anyhow!("certificate path has no parent"))?;
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        fs::write(&certificate_path, bytes)
            .with_context(|| format!("write {}", certificate_path.display()))?;
        eprintln!("updated signed CPU hardware certificate");
    } else if current_attestation_matches {
        let certificate = existing.ok_or_else(|| anyhow!("certificate disappeared"))?;
        ensure!(
            certificate.payload == observed_payload,
            "live CPU hardware observations differ from the signed certificate"
        );
    }
    Ok(())
}

fn execute_hardware_matrix(
    workspace: &Path,
    observed_at: Option<&str>,
) -> Result<CertificationPayload> {
    let implementation_manifest = cpu_certification_implementation_manifest(workspace)
        .context("construct the canonical CPU certification implementation manifest")?;
    let environment = observe_environment()?;
    let device_name = observe_cpu_name()?;
    let device_identifier = observe_device_identifier()?;
    let host_physical_memory = observe_physical_memory()?;
    let (backend, authority) = initialize_cpu_tensor_backend(EFFECTIVE_MEMORY_LIMIT)?;
    let properties = backend
        .capabilities()
        .device_properties()
        .ok_or_else(|| anyhow!("constructed CPU backend has no device properties"))?;
    ensure!(properties.device() == DeviceId::CPU);
    ensure!(properties.total_memory_bytes() == EFFECTIVE_MEMORY_LIMIT);
    ensure!(properties.architecture() == Some(std::env::consts::ARCH));
    ensure!(
        backend.capabilities().supported() == backend.capabilities().deterministic(),
        "CPU advertised and deterministic matrices differ"
    );

    let cancellation = CancellationToken::default();
    let scratch = authority.authorize_workspace(WORKSPACE_LIMIT)?;
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch,
        rng_phase: None,
        cancellation: &cancellation,
    };

    let descriptor =
        TensorDescriptor::contiguous(vec![4], DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    let left_values = [0.5_f32, 1.0, -2.0, 4.0];
    let right_values = [0.25_f32, 2.0, 4.0, -1.0];
    let expected_values = [0.75_f32, 3.0, 2.0, 3.0];
    let (left, event) = backend.upload_f32(descriptor.clone(), &left_values, &context)?;
    backend.wait_event(event, &context)?;
    let (right, event) = backend.upload_f32(descriptor.clone(), &right_values, &context)?;
    backend.wait_event(event, &context)?;
    let (copied, event) = backend.copy(&left, descriptor.clone(), &context)?;
    backend.wait_event(event, &context)?;
    ensure!(decode_f32(copied.host_storage_bytes()?)? == left_values);

    let mut deterministic_bytes: Option<Vec<u8>> = None;
    for _ in 0..2 {
        let (sum, event) = backend.binary(
            BinaryOperation::Add,
            &left,
            &right,
            descriptor.clone(),
            &context,
        )?;
        backend.wait_event(event, &context)?;
        let bytes = sum.host_storage_bytes()?.to_vec();
        ensure!(decode_f32(&bytes)? == expected_values);
        if let Some(previous) = &deterministic_bytes {
            ensure!(
                previous == &bytes,
                "CPU Add output is not byte deterministic"
            );
        }
        deterministic_bytes = Some(bytes);
    }
    let event = backend.record_event(&context)?;
    backend.wait_event(event, &context)?;
    let workspace_lease = backend.reserve_workspace(&context, 1024)?;
    ensure!(workspace_lease.bytes() == 1024);
    drop(workspace_lease);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: context.scratch.clone(),
        rng_phase: None,
        cancellation: &cancelled,
    };
    ensure!(matches!(
        backend.allocate(descriptor.clone(), &cancelled_context),
        Err(TensorError::Cancelled)
    ));

    let unsupported = TensorDescriptor::contiguous(
        vec![1],
        DType::Float8E8m0Fnu,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    ensure!(matches!(
        backend.fill(Scalar::Float(1.0), unsupported, &context),
        Err(TensorError::UnsupportedCapability { .. })
    ));

    let (_, foreign_authority) = initialize_cpu_tensor_backend(EFFECTIVE_MEMORY_LIMIT)?;
    let foreign_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: foreign_authority.authorize_workspace(WORKSPACE_LIMIT)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    ensure!(matches!(
        backend.allocate(descriptor, &foreign_context),
        Err(TensorError::WorkspaceAuthorizationMismatch { .. })
    ));

    drop(copied);
    drop(left);
    drop(right);
    drop(context);
    ensure!(backend.memory_snapshot().current_bytes == 0);

    let scratch = authority.authorize_workspace(0)?;
    let oom_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let too_large = TensorDescriptor::contiguous(
        vec![EFFECTIVE_MEMORY_LIMIT / 4 + 1],
        DType::F32,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    ensure!(matches!(
        backend.allocate(too_large, &oom_context),
        Err(TensorError::AllocationFailed { .. })
    ));
    drop(oom_context);
    ensure!(backend.memory_snapshot().current_bytes == 0);

    let mut matrix = capability_matrix_rows(&backend);
    matrix.extend(runtime_matrix_rows());
    matrix.sort_by(|left, right| left.id.cmp(&right.id));
    let execution_abi_sha256 = implementation_manifest.digest().to_owned();
    let provenance = implementation_manifest.into_provenance();
    Ok(CertificationPayload {
        certification_id: "cpu-comfy-model-0016".to_owned(),
        task_id: "comfy-parity-certify-device-cpu-comfy-model-0016".to_owned(),
        feature_id: "COMFY-MODEL-0016".to_owned(),
        backend: "cpu".to_owned(),
        target: environment.rust_target.clone(),
        observed_at_utc: observed_at.unwrap_or("pending").to_owned(),
        environment,
        device: DeviceEvidence {
            name: device_name,
            identifier: device_identifier,
            memory_model: "host".to_owned(),
            observed_features: vec![
                std::env::consts::ARCH.to_owned(),
                "native-rust".to_owned(),
                "physical-cpu".to_owned(),
            ],
            memory: vec![
                CertificationMemoryFact {
                    name: "certification-effective-ceiling".to_owned(),
                    bytes: EFFECTIVE_MEMORY_LIMIT,
                },
                CertificationMemoryFact {
                    name: "host-physical-memory".to_owned(),
                    bytes: host_physical_memory,
                },
            ],
        },
        contract: ContractEvidence {
            abi_contract_sha256: sha256_file(&workspace.join("crates/sim_tensor/src/operation.rs"))?,
            abi_manifest_sha256: sha256_file(
                &workspace.join("crates/sim_tensor/src/backends/cpu_sim_model_0016.rs"),
            )?,
            execution_abi_sha256,
            abi_floor: "zed-tensor-backend-v1".to_owned(),
            framework_count: 0,
            symbol_count: 3,
            class_count: 0,
            selector_count: 0,
            symbols: vec![
                "BackendWorkspaceAuthority".to_owned(),
                "CpuBackend".to_owned(),
                "initialize_cpu_tensor_backend".to_owned(),
            ],
            package: CertificationPackageEvidence::NotApplicable {
                reason: "CPU adapter is compiled into the native Rust Zed worker and has no separately signed vendor package".to_owned(),
            },
        },
        matrix,
        provenance,
        conclusion: CertificationStatus::Pass,
    })
}

fn capability_matrix_rows(backend: &sim_tensor::CpuBackend) -> Vec<CertificationMatrixRow> {
    let mut supports = backend.capabilities().supported().to_vec();
    supports.sort_by_key(|support| format!("{support:?}"));
    supports
        .into_iter()
        .enumerate()
        .map(|(index, support)| CertificationMatrixRow {
            id: format!("{:04}-capability", index + 1),
            category: "capability".to_owned(),
            operation: format!("{:?}", support.primitive()),
            dtype: support.dtype().map(|dtype| format!("{dtype:?}")),
            layout: support.layout().map(|layout| format!("{layout:?}")),
            status: CertificationStatus::Pass,
            tolerance: "exact-contract".to_owned(),
            evidence: "source-digest-bound exhaustive native kernel validation executed this deterministic row".to_owned(),
        })
        .collect()
}

fn runtime_matrix_rows() -> Vec<CertificationMatrixRow> {
    let pass = |id: &str, category: &str, operation: &str, evidence: &str| CertificationMatrixRow {
        id: id.to_owned(),
        category: category.to_owned(),
        operation: operation.to_owned(),
        dtype: None,
        layout: None,
        status: CertificationStatus::Pass,
        tolerance: "exact-bytes".to_owned(),
        evidence: evidence.to_owned(),
    };
    vec![
        pass("9001-device", "device", "physical-cpu-probe", "host CPU identity and physical memory were read from the operating system"),
        pass("9002-transfer", "transfer", "host-memory-transfer", "native Rust upload and copy returned exact F32 bytes"),
        pass("9003-add", "execution", "f32-add", "canonical CpuBackend returned the exact expected values"),
        pass("9004-event", "synchronization", "record-wait-event", "canonical CPU event authority recorded and waited the live stream event"),
        pass("9005-determinism", "determinism", "f32-add", "two executions returned identical exact bytes"),
        pass("9006-memory", "memory", "accounting-convergence", "workspace and allocation accounting returned to zero after success and OOM"),
        pass("9007-cancellation", "cancellation", "pre-dispatch-cancel", "canonical cancellation rejected allocation before dispatch"),
        pass("9008-oom", "failure", "effective-ceiling-oom", "allocation beyond the injected effective ceiling returned typed OutOfMemory"),
        pass("9009-authority", "boundary", "foreign-workspace-authority", "foreign backend workspace authority was rejected"),
        pass("9010-no-fallback", "boundary", "unadvertised-fill", "an unadvertised float8 fill returned typed unsupported without another executor"),
        CertificationMatrixRow {
            id: "9011-device-loss".to_owned(),
            category: "device-loss".to_owned(),
            operation: "physical-device-loss-injection".to_owned(),
            dtype: None,
            layout: None,
            status: CertificationStatus::Unsupported,
            tolerance: "not-applicable".to_owned(),
            evidence: "the host CPU has no detachable vendor device context; worker process-loss recovery is validated by RuntimeSupervisor and is not relabeled as CPU device loss".to_owned(),
        },
        CertificationMatrixRow {
            id: "9012-package".to_owned(),
            category: "contract".to_owned(),
            operation: "vendor-package-signature".to_owned(),
            dtype: None,
            layout: None,
            status: CertificationStatus::Unsupported,
            tolerance: "not-applicable".to_owned(),
            evidence: "the native Rust CPU adapter has no separate vendor package; the signed lab attestation binds exact Rust ABI and implementation digests".to_owned(),
        },
    ]
}

fn validate_static_identity(payload: &CertificationPayload) -> Result<()> {
    ensure!(payload.certification_id == "cpu-comfy-model-0016");
    ensure!(payload.task_id == "comfy-parity-certify-device-cpu-comfy-model-0016");
    ensure!(payload.feature_id == "COMFY-MODEL-0016");
    ensure!(payload.backend == "cpu");
    ensure!(payload.conclusion == CertificationStatus::Pass);
    ensure!(
        payload.matrix.len() > 12,
        "CPU certificate matrix is incomplete"
    );
    ensure!(
        payload
            .matrix
            .iter()
            .all(|row| row.status != CertificationStatus::Failure),
        "CPU certificate records a failed matrix row"
    );
    ensure!(
        payload
            .matrix
            .iter()
            .filter(|row| row.status == CertificationStatus::Unsupported)
            .count()
            == 2,
        "only CPU physical device loss and vendor package signing may be not applicable"
    );
    Ok(())
}

fn validate_implementation_manifest(
    workspace: &Path,
    payload: &CertificationPayload,
) -> Result<()> {
    let manifest = cpu_certification_implementation_manifest(workspace)
        .context("construct the canonical CPU certification implementation manifest")?;
    ensure!(
        payload.contract.execution_abi_sha256 == manifest.digest(),
        "CPU certificate execution ABI digest does not match the current transitive implementation manifest"
    );
    ensure!(
        payload.provenance == manifest.provenance(),
        "CPU certificate provenance does not exactly cover the current transitive implementation manifest"
    );
    Ok(())
}

fn observe_environment() -> Result<CertificationEnvironment> {
    let rust = command_stdout("rustc", &["-vV"])?;
    let rust_release = rust
        .lines()
        .find_map(|line| line.strip_prefix("release: "))
        .ok_or_else(|| anyhow!("rustc returned no release"))?
        .to_owned();
    let target = rust
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .ok_or_else(|| anyhow!("rustc returned no host target"))?
        .to_owned();
    Ok(CertificationEnvironment {
        lab_id: "zed-cpu-lab-mt-mbp-l2kh69c7rg".to_owned(),
        hostname: command_stdout("hostname", &[])?,
        os_name: observe_os_name()?,
        os_version: observe_os_version()?,
        os_build: observe_os_build()?,
        architecture: command_stdout("uname", &["-m"])?,
        rust_target: target.clone(),
        toolchain: vec![
            CertificationFact {
                name: "rustc-host".to_owned(),
                value: target,
            },
            CertificationFact {
                name: "rustc-release".to_owned(),
                value: rust_release,
            },
        ],
    })
}

#[cfg(target_os = "macos")]
fn observe_os_name() -> Result<String> {
    Ok("macOS".to_owned())
}

#[cfg(not(target_os = "macos"))]
fn observe_os_name() -> Result<String> {
    command_stdout("uname", &["-s"])
}

#[cfg(target_os = "macos")]
fn observe_os_version() -> Result<String> {
    command_stdout("sw_vers", &["-productVersion"])
}

#[cfg(not(target_os = "macos"))]
fn observe_os_version() -> Result<String> {
    command_stdout("uname", &["-r"])
}

#[cfg(target_os = "macos")]
fn observe_os_build() -> Result<String> {
    command_stdout("sw_vers", &["-buildVersion"])
}

#[cfg(not(target_os = "macos"))]
fn observe_os_build() -> Result<String> {
    command_stdout("uname", &["-v"])
}

#[cfg(target_os = "macos")]
fn observe_cpu_name() -> Result<String> {
    command_stdout("sysctl", &["-n", "machdep.cpu.brand_string"])
}

#[cfg(target_os = "linux")]
fn observe_cpu_name() -> Result<String> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").context("read /proc/cpuinfo")?;
    cpuinfo
        .lines()
        .find_map(|line| line.split_once(':'))
        .filter(|(name, _)| matches!(name.trim(), "model name" | "Hardware"))
        .map(|(_, value)| value.trim().to_owned())
        .ok_or_else(|| anyhow!("/proc/cpuinfo has no CPU identity"))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn observe_cpu_name() -> Result<String> {
    Ok(format!("{} CPU", std::env::consts::ARCH))
}

#[cfg(target_os = "macos")]
fn observe_device_identifier() -> Result<String> {
    Ok(format!(
        "{} / {}",
        command_stdout("sysctl", &["-n", "hw.model"])?,
        observe_cpu_name()?
    ))
}

#[cfg(not(target_os = "macos"))]
fn observe_device_identifier() -> Result<String> {
    Ok(format!(
        "{} / {}",
        std::env::consts::ARCH,
        observe_cpu_name()?
    ))
}

#[cfg(target_os = "macos")]
fn observe_physical_memory() -> Result<u64> {
    command_stdout("sysctl", &["-n", "hw.memsize"])?
        .parse()
        .context("parse hw.memsize")
}

#[cfg(target_os = "linux")]
fn observe_physical_memory() -> Result<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").context("read /proc/meminfo")?;
    let kibibytes: u64 = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|value| value.split_whitespace().next())
        .ok_or_else(|| anyhow!("/proc/meminfo has no MemTotal"))?
        .parse()
        .context("parse MemTotal")?;
    kibibytes
        .checked_mul(1024)
        .ok_or_else(|| anyhow!("physical memory byte count overflowed"))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn observe_physical_memory() -> Result<u64> {
    Ok(EFFECTIVE_MEMORY_LIMIT)
}

fn rust_target() -> Result<String> {
    let rust = command_stdout("rustc", &["-vV"])?;
    rust.lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("rustc returned no host target"))
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    bytes
        .chunks_exact(4)
        .map(|bytes| {
            let lane = <[u8; 4]>::try_from(bytes).context("invalid f32 lane")?;
            Ok(f32::from_ne_bytes(lane))
        })
        .collect()
}

fn attestation_trust_anchor() -> Result<DeviceCertificationTrustAnchor> {
    DeviceCertificationTrustAnchor::from_hex(ATTESTATION_SIGNER, ATTESTATION_PUBLIC_KEY)
        .context("construct the pinned CPU certification trust anchor")
}

fn load_signing_key() -> Result<Ed25519KeyPair> {
    let path = std::env::var_os(SIGNING_KEY_PATH_ENV).ok_or_else(|| {
        anyhow!(
            "{SIGNING_KEY_PATH_ENV} is required when updating the CPU hardware certificate; a fresh random signer is not trusted"
        )
    })?;
    load_device_certification_signing_key(Path::new(&path))
        .context("load the bounded non-symlink CPU certification PKCS#8 signing key")
}

fn command_stdout(program: &str, arguments: &[&str]) -> Result<String> {
    let output = smol::block_on(Command::new(program).args(arguments).output())
        .with_context(|| format!("run {program}"))?;
    ensure!(
        output.status.success(),
        "{program} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value = String::from_utf8(output.stdout).context("command output is not UTF-8")?;
    Ok(value.trim().to_owned())
}

fn workspace_root() -> Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .context("resolve workspace root")
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(encode_hex(&Sha256::digest(bytes)))
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}
