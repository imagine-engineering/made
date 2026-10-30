#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use half::f16;
use ring::signature::{Ed25519KeyPair, KeyPair};
use sha2::{Digest, Sha256};
use sim_backend_metal::{MetalExecutionError, probe_abi, probe_device};
use sim_runtime::{
    NativeMetalPackageSettings, initialize_certified_metal_runtime, metal_package_signing_payload,
};
use sim_tensor::{
    BinaryOperation, CancellationToken, DType, DeviceId, ExecutionContext, Layout,
    MetalTensorBackend, OperationSupport, StreamId, TensorBackend, TensorDescriptor, TensorError,
};
use sim_test_support::device_certification::{
    DeviceCertificationTrustAnchor, load_device_certification_signing_key,
};
use sim_test_support::{
    CertificationArtifact, CertificationEnvironment, CertificationFact, CertificationMatrixRow,
    CertificationMemoryFact, CertificationPackageEvidence, CertificationPayload,
    CertificationProvenance, CertificationStatus, ContractEvidence, DeviceEvidence,
    PackageEvidence,
};
use sim_types::DeviceKind;
use smol::process::Command;
use tempfile::TempDir;

const CERTIFICATION_RELATIVE_PATH: &str = "crates/sim_test_support/parity/catalogs/native-device-certification/apple-metal-mps-comfy-model-0015.json";
const PACKAGE_SIGNER: &str = "metal.lab.mt-mbp-l2kh69c7rg";
const ATTESTATION_SIGNER: &str = "zed.hardware.lab.mt-mbp-l2kh69c7rg";
const ATTESTATION_PUBLIC_KEY: &str =
    "0f17d6edad8968968b48bb4a00a332ba1759342aeb750cebaca3f3413f7951cf";
const ATTESTATION_SIGNING_KEY_PATH_ENV: &str = "SIM_METAL_CERTIFICATION_SIGNING_KEY_PKCS8_PATH";
const PACKAGE_SIGNING_KEY_PATH_ENV: &str = "SIM_METAL_PACKAGE_SIGNING_KEY_PKCS8_PATH";
const TARGET: &str = "aarch64-apple-darwin";

struct PreparedPackage {
    _directory: TempDir,
    root: PathBuf,
    evidence: PackageEvidence,
}

#[test]
fn val_device_001_signed_apple_metal_mps_hardware_certification() -> Result<()> {
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

    if !update {
        let certificate = existing.as_ref().ok_or_else(|| {
            anyhow!(
                "signed Metal hardware certificate is missing; run this test with UPDATE_SIM_DEVICE_CERTIFICATION=1 on the approved lab"
            )
        })?;
        validate_static_identity(&certificate.payload)?;
        let device = probe_device().context("probe the current Metal/MPS device")?;
        if device.name != certificate.payload.device.name
            || format!("{:#018x}", device.registry_id) != certificate.payload.device.identifier
        {
            eprintln!(
                "signed Metal certification remains bound to {} {}; current device {} {:#018x} is not relabeled as certified",
                certificate.payload.device.name,
                certificate.payload.device.identifier,
                device.name,
                device.registry_id
            );
            return Ok(());
        }
    }

    let package = prepare_package(&workspace, existing.as_ref(), update)?;
    let (observed_payload, package_public_key) = execute_hardware_matrix(
        &workspace,
        &package,
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
        let key_pair = load_signing_key(
            ATTESTATION_SIGNING_KEY_PATH_ENV,
            "Metal hardware certification attestation",
        )?;
        let certificate = CertificationArtifact::sign(payload, &trust_anchor, &key_pair)?;
        let bytes = CertificationArtifact::to_canonical_json(&certificate)?;
        let parent = certificate_path
            .parent()
            .ok_or_else(|| anyhow!("certificate path has no parent"))?;
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        fs::write(&certificate_path, bytes)
            .with_context(|| format!("write {}", certificate_path.display()))?;
        eprintln!(
            "updated signed Metal hardware certificate with package key {}",
            package_public_key
        );
    } else {
        let certificate = existing.ok_or_else(|| anyhow!("certificate disappeared"))?;
        ensure!(
            certificate.payload == observed_payload,
            "live Metal hardware observations differ from the signed certificate"
        );
    }
    Ok(())
}

fn prepare_package(
    workspace: &Path,
    existing: Option<&sim_test_support::SignedDeviceCertification>,
    update: bool,
) -> Result<PreparedPackage> {
    let directory = tempfile::tempdir().context("create Metal certification package directory")?;
    let catalog = directory.path().join("ffi-contracts-v1.json");
    let receipt = directory.path().join("adapter-manifest.sig");
    let root = directory.path().join("package");
    run_checked(
        Command::new(workspace.join("script/package-sim-backend-metal"))
            .arg("--write-contract-catalog")
            .arg(&catalog)
            .arg(TARGET),
        "generate the pinned Metal FFI contract catalog",
    )?;

    let configured_key = if update {
        Some(load_signing_key(
            PACKAGE_SIGNING_KEY_PATH_ENV,
            "Metal certification package",
        )?)
    } else {
        None
    };
    let (signer, public_key, initial_signature) = match (configured_key.as_ref(), existing) {
        (Some(key_pair), _) => (
            PACKAGE_SIGNER.to_owned(),
            encode_hex(key_pair.public_key().as_ref()),
            "0".repeat(128),
        ),
        (None, Some(certificate)) => {
            let package = certificate
                .payload
                .contract
                .package
                .signed()
                .ok_or_else(|| anyhow!("Metal certificate is missing signed package evidence"))?;
            (
                package.signer.clone(),
                package.signer_public_key.clone(),
                package.signature.clone(),
            )
        }
        (None, None) => bail!("package signing evidence is unavailable"),
    };
    write_receipt(&receipt, &initial_signature)?;
    run_checked(
        Command::new(workspace.join("script/package-sim-backend-metal"))
            .arg(&root)
            .arg(TARGET)
            .arg(&receipt)
            .arg(&signer)
            .arg(&catalog),
        "compile and package the pinned Metal kernels",
    )?;

    let coverage = fs::read(root.join("package-coverage.sha256"))?;
    let coverage_sha256 = sha256_hex(&coverage);
    let signature = if let Some(key_pair) = configured_key {
        let payload = metal_package_signing_payload(&signer, &coverage)
            .context("construct canonical Metal package signing payload")?;
        let signature = encode_hex(key_pair.sign(&payload).as_ref());
        write_receipt(&root.join("adapter-manifest.sig"), &signature)?;
        signature
    } else {
        initial_signature
    };
    let verification_key = sim_runtime::MetalPackageVerificationKey::new(
        signer.clone(),
        decode_hex::<32>(&public_key).ok_or_else(|| anyhow!("invalid package public key"))?,
    )?;
    let receipt_bytes = fs::read(root.join("adapter-manifest.sig"))?;
    verification_key
        .verify_package(&signer, &coverage, &receipt_bytes)
        .context("verify the exact compiled Metal package signature")?;

    if let Some(certificate) = existing {
        let package = certificate
            .payload
            .contract
            .package
            .signed()
            .ok_or_else(|| anyhow!("Metal certificate is missing signed package evidence"))?;
        ensure!(
            coverage_sha256 == package.coverage_sha256,
            "fresh pinned package coverage differs from the signed lab artifact"
        );
    }
    let evidence = PackageEvidence {
        format: "zed-sim-metal-package-v1".to_owned(),
        signer,
        signer_public_key: public_key,
        signature,
        coverage_sha256,
        manifest_sha256: sha256_file(&root.join("adapter-manifest.json"))?,
        contract_catalog_sha256: sha256_file(&root.join("ffi-contracts-v1.json"))?,
        payloads: vec![
            CertificationProvenance {
                path: "kernels/readiness.metallib".to_owned(),
                sha256: sha256_file(&root.join("kernels/readiness.metallib"))?,
            },
            CertificationProvenance {
                path: "kernels/tensor_ops.metallib".to_owned(),
                sha256: sha256_file(&root.join("kernels/tensor_ops.metallib"))?,
            },
        ],
    };
    Ok(PreparedPackage {
        _directory: directory,
        root,
        evidence,
    })
}

fn execute_hardware_matrix(
    workspace: &Path,
    package: &PreparedPackage,
    observed_at: Option<&str>,
) -> Result<(CertificationPayload, String)> {
    let environment = observe_environment()?;
    let abi = probe_abi().context("probe the exact reviewed Metal ABI")?;
    let device = probe_device().context("probe the actual Metal/MPS hardware")?;
    ensure!(
        abi.target == TARGET,
        "Metal ABI target differs from the certification target"
    );
    ensure!(
        device.metal_3 && device.mps_supported,
        "Metal 3/MPS support is incomplete"
    );

    let settings = NativeMetalPackageSettings::from_public_authority(
        &package.root,
        package.evidence.signer.clone(),
        &package.evidence.signer_public_key,
    )
    .map_err(anyhow::Error::msg)?;
    let cancellation = CancellationToken::default();
    let certified = initialize_certified_metal_runtime(&settings, &cancellation)
        .context("initialize the production-certified Metal runtime")?;
    ensure!(
        certified.certificates().len() == 5,
        "expected five retained certificates"
    );
    let host_physical_memory_bytes = certified.host_physical_memory_bytes();
    ensure!(
        matches!(
            certified.runtime().inject_test_command_failure(11),
            Err(MetalExecutionError::InvalidCertifiedInputs { .. })
        ),
        "real Metal runtime unexpectedly exposed fake device-loss injection"
    );
    let runtime = certified.into_runtime();
    let requested_limit = runtime
        .properties()
        .recommended_working_set_bytes()
        .min(64 * 1024 * 1024);
    let (backend, authority) = MetalTensorBackend::from_certified_runtime(
        runtime,
        host_physical_memory_bytes,
        requested_limit,
        &cancellation,
    )?;
    let scratch = authority.authorize_workspace(requested_limit)?;
    let context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch,
        rng_phase: None,
        cancellation: &cancellation,
    };
    ensure!(
        backend.capabilities().supported().len() == 12,
        "unexpected capability row count"
    );

    let mut matrix = base_matrix_rows();
    {
        for dtype in [DType::F16, DType::F32] {
            exercise_dtype(&backend, &context, dtype)?;
        }
        let event = backend.record_event(&context)?;
        backend.wait_event(event, &context)?;
        let lease = backend.reserve_workspace(&context, 1024)?;
        ensure!(lease.bytes() == 1024, "workspace lease differs");
        drop(lease);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: context.scratch.clone(),
            rng_phase: None,
            cancellation: &cancelled,
        };
        ensure!(
            matches!(
                backend.allocate(descriptor(vec![1], DType::F32)?, &cancelled_context),
                Err(TensorError::Cancelled)
            ),
            "cancelled allocation reached native dispatch"
        );
        let unsupported_layout = TensorDescriptor::new_strided(
            vec![2],
            vec![2],
            0,
            DType::F32,
            Layout::Strided,
            metal_device(),
            StreamId::DEFAULT,
        )?;
        ensure!(
            matches!(
                backend.allocate(unsupported_layout, &context),
                Err(TensorError::UnsupportedCapability { .. })
            ),
            "unsupported Metal layout was dispatched"
        );
        let (left, event) = backend.upload_bytes(
            descriptor(vec![2], DType::F32)?,
            &encode(DType::F32, &[1.0, 2.0])?,
            &context,
        )?;
        backend.wait_event(event, &context)?;
        ensure!(
            matches!(
                backend.binary(
                    BinaryOperation::Multiply,
                    &left,
                    &left,
                    descriptor(vec![2], DType::F32)?,
                    &context,
                ),
                Err(TensorError::UnsupportedCapability { .. })
            ),
            "unsupported operation fell through to another executor"
        );
    }
    ensure!(
        context.scratch.in_use_bytes() == 0,
        "workspace accounting did not converge"
    );
    drop(context);
    ensure!(
        backend.memory_snapshot().current_bytes == 0,
        "logical memory did not converge"
    );
    ensure!(
        backend.physical_memory_snapshot().current_bytes == 0,
        "physical Metal allocations did not converge"
    );
    matrix.sort_by(|left, right| left.id.cmp(&right.id));

    let symbols = vec![
        "MPSSupportsMTLDevice".to_owned(),
        "MTLCreateSystemDefaultDevice".to_owned(),
        "zed_sim_metal_add_f16_v1".to_owned(),
        "zed_sim_metal_add_f32_v1".to_owned(),
        "zed_sim_metal_readiness_v1".to_owned(),
    ];
    let provenance = collect_provenance(workspace)?;
    let payload = CertificationPayload {
        certification_id: "apple-metal-mps-comfy-model-0015".to_owned(),
        task_id: "comfy-parity-certify-device-apple-metal-mps-comfy-model-0015".to_owned(),
        feature_id: "COMFY-MODEL-0015".to_owned(),
        backend: "metal".to_owned(),
        target: TARGET.to_owned(),
        observed_at_utc: observed_at.unwrap_or("pending").to_owned(),
        environment,
        device: DeviceEvidence {
            name: device.name,
            identifier: format!("{:#018x}", device.registry_id),
            memory_model: if device.unified_memory {
                "unified".to_owned()
            } else {
                "managed".to_owned()
            },
            observed_features: vec!["metal-3".to_owned(), "metal-performance-shaders".to_owned()],
            memory: vec![
                CertificationMemoryFact {
                    name: "certification-effective-ceiling".to_owned(),
                    bytes: requested_limit,
                },
                CertificationMemoryFact {
                    name: "host-physical-memory".to_owned(),
                    bytes: host_physical_memory_bytes,
                },
                CertificationMemoryFact {
                    name: "recommended-working-set".to_owned(),
                    bytes: device.recommended_working_set_bytes,
                },
            ],
        },
        contract: ContractEvidence {
            abi_contract_sha256: sha256_file(
                &workspace
                    .join("crates/sim_test_support/parity/catalogs/native-backend-abi/metal.json"),
            )?,
            abi_manifest_sha256: sha256_file(
                &workspace.join("crates/sim_backend_metal/abi/symbols-v1.json"),
            )?,
            execution_abi_sha256: sha256_file(
                &workspace.join("crates/sim_backend_metal/abi/execution-v1.json"),
            )?,
            abi_floor: "macos-13-metal-3".to_owned(),
            framework_count: abi.framework_count,
            symbol_count: abi.symbol_count,
            class_count: abi.class_count,
            selector_count: abi.selector_count,
            symbols,
            package: CertificationPackageEvidence::Signed(package.evidence.clone()),
        },
        matrix,
        provenance,
        conclusion: CertificationStatus::Pass,
    };
    Ok((payload, package.evidence.signer_public_key.clone()))
}

fn exercise_dtype(
    backend: &MetalTensorBackend,
    context: &ExecutionContext<'_>,
    dtype: DType,
) -> Result<()> {
    for row in [
        OperationSupport::allocation(dtype, Layout::Contiguous),
        OperationSupport::copy_input(dtype, Layout::Contiguous),
        OperationSupport::copy_output(dtype, Layout::Contiguous),
        OperationSupport::binary_input(BinaryOperation::Add, dtype, Layout::Contiguous),
        OperationSupport::binary_output(BinaryOperation::Add, dtype, Layout::Contiguous),
    ] {
        ensure!(
            backend.capabilities().supports(row),
            "missing advertised Metal row"
        );
        ensure!(
            backend.capabilities().is_deterministic(row),
            "Metal row is not deterministic"
        );
    }
    let left_values = [0.5_f32, 1.0, -2.0];
    let right_values = [0.25_f32, 2.0, 4.0];
    let expected = [0.75_f32, 3.0, 2.0];
    let tensor_descriptor = descriptor(vec![3], dtype)?;
    let (left, event) = backend.upload_bytes(
        tensor_descriptor.clone(),
        &encode(dtype, &left_values)?,
        context,
    )?;
    backend.wait_event(event, context)?;
    let (right, event) = backend.upload_bytes(
        tensor_descriptor.clone(),
        &encode(dtype, &right_values)?,
        context,
    )?;
    backend.wait_event(event, context)?;
    let (copied, event) = backend.copy(&left, tensor_descriptor.clone(), context)?;
    backend.wait_event(event, context)?;
    ensure!(decode(dtype, &backend.download_bytes(&copied, context)?)? == left_values);

    let mut deterministic_bytes: Option<Vec<u8>> = None;
    for _ in 0..2 {
        let (sum, event) = backend.binary(
            BinaryOperation::Add,
            &left,
            &right,
            tensor_descriptor.clone(),
            context,
        )?;
        backend.wait_event(event, context)?;
        let bytes = backend.download_bytes(&sum, context)?;
        ensure!(decode(dtype, &bytes)? == expected);
        if let Some(prior) = &deterministic_bytes {
            ensure!(
                &bytes == prior,
                "Metal Add output is not byte deterministic"
            );
        }
        deterministic_bytes = Some(bytes);
    }
    Ok(())
}

fn base_matrix_rows() -> Vec<CertificationMatrixRow> {
    let pass = |id: &str, category: &str, operation: &str, dtype: Option<&str>, evidence: &str| {
        CertificationMatrixRow {
            id: id.to_owned(),
            category: category.to_owned(),
            operation: operation.to_owned(),
            dtype: dtype.map(str::to_owned),
            layout: dtype.map(|_| "contiguous".to_owned()),
            status: CertificationStatus::Pass,
            tolerance: "exact-bytes".to_owned(),
            evidence: evidence.to_owned(),
        }
    };
    let mut rows = vec![
        pass(
            "001-abi",
            "contract",
            "abi-probe",
            None,
            "three fixed frameworks, two C symbols, three classes, twelve selectors",
        ),
        pass(
            "002-package",
            "contract",
            "signed-package",
            None,
            "canonical runtime verifier accepted deterministic pinned package coverage",
        ),
        pass(
            "003-device",
            "device",
            "device-probe",
            None,
            "actual default Apple GPU passed Metal 3 and MPS probes",
        ),
    ];
    for (base, dtype) in [(4_u16, "f16"), (9_u16, "f32")] {
        for (offset, operation) in [
            (0, "allocation"),
            (1, "copy-input"),
            (2, "copy-output"),
            (3, "add-input"),
            (4, "add-output"),
        ] {
            rows.push(pass(
                &format!("{:03}-{dtype}-{operation}", base + offset),
                "capability",
                operation,
                Some(dtype),
                "advertised row executed through the retained native Metal runtime",
            ));
        }
    }
    rows.extend([
        pass("014-upload", "transfer", "host-to-device", None, "actual shared-buffer upload completed and was event-fenced"),
        pass("015-download", "transfer", "device-to-host", None, "actual shared-buffer download returned exact bytes"),
        pass("016-record-event", "synchronization", "record-event", None, "native command-buffer event was recorded"),
        pass("017-wait-event", "synchronization", "wait-event", None, "native event completed and retired"),
        pass("018-determinism-f16", "determinism", "add", Some("f16"), "two native executions returned identical exact bytes"),
        pass("019-determinism-f32", "determinism", "add", Some("f32"), "two native executions returned identical exact bytes"),
        pass("020-memory", "memory", "accounting-convergence", None, "logical workspace and physical allocation counters returned to zero"),
        pass("021-cancellation", "cancellation", "pre-dispatch-cancel", None, "canonical cancellation rejected allocation before native dispatch"),
        pass("022-layout", "boundary", "strided-layout", None, "unsupported layout was rejected before native dispatch"),
        pass("023-no-fallback", "boundary", "multiply", None, "unsupported operation returned typed unsupported without CPU fallback"),
        CertificationMatrixRow {
            id: "024-device-loss".to_owned(),
            category: "device-loss".to_owned(),
            operation: "physical-device-loss-injection".to_owned(),
            dtype: None,
            layout: None,
            status: CertificationStatus::Unsupported,
            tolerance: "not-applicable".to_owned(),
            evidence: "safe failure injection is deliberately unavailable on a physical Metal runtime; deterministic fake-runtime tests cover typed device-loss mapping".to_owned(),
        },
    ]);
    rows
}

fn validate_static_identity(payload: &CertificationPayload) -> Result<()> {
    ensure!(payload.certification_id == "apple-metal-mps-comfy-model-0015");
    ensure!(payload.task_id == "comfy-parity-certify-device-apple-metal-mps-comfy-model-0015");
    ensure!(payload.feature_id == "COMFY-MODEL-0015");
    ensure!(payload.backend == "metal" && payload.target == TARGET);
    ensure!(payload.conclusion == CertificationStatus::Pass);
    ensure!(
        payload.matrix.len() == 24,
        "certificate matrix is incomplete"
    );
    ensure!(
        payload
            .matrix
            .iter()
            .all(|row| row.status != CertificationStatus::Failure),
        "certificate records a failed matrix row"
    );
    ensure!(
        payload
            .matrix
            .iter()
            .filter(|row| row.status == CertificationStatus::Unsupported)
            .count()
            == 1,
        "certificate must record exactly the physical device-loss injection row as unsupported"
    );
    Ok(())
}

fn observe_environment() -> Result<CertificationEnvironment> {
    let xcode = command_stdout("xcodebuild", &["-version"])?;
    let xcode_version = xcode
        .lines()
        .next()
        .ok_or_else(|| anyhow!("xcodebuild returned no version"))?
        .trim_start_matches("Xcode ")
        .to_owned();
    let xcode_build = xcode
        .lines()
        .find_map(|line| line.strip_prefix("Build version "))
        .ok_or_else(|| anyhow!("xcodebuild returned no build identity"))?
        .to_owned();
    Ok(CertificationEnvironment {
        lab_id: "zed-metal-lab-mt-mbp-l2kh69c7rg".to_owned(),
        hostname: command_stdout("hostname", &[])?,
        os_name: "macOS".to_owned(),
        os_version: command_stdout("sw_vers", &["-productVersion"])?,
        os_build: command_stdout("sw_vers", &["-buildVersion"])?,
        architecture: command_stdout("uname", &["-m"])?,
        rust_target: TARGET.to_owned(),
        toolchain: vec![
            CertificationFact {
                name: "sdk".to_owned(),
                value: format!(
                    "macosx{}",
                    command_stdout("xcrun", &["-sdk", "macosx", "--show-sdk-version"])?
                ),
            },
            CertificationFact {
                name: "xcode-build".to_owned(),
                value: xcode_build,
            },
            CertificationFact {
                name: "xcode-version".to_owned(),
                value: xcode_version,
            },
        ],
    })
}

fn collect_provenance(workspace: &Path) -> Result<Vec<CertificationProvenance>> {
    let paths = [
        "crates/sim_test_support/parity/catalogs/native-backend-abi/metal.json",
        "crates/sim_test_support/parity/catalogs/native-tensor-operation-contracts.csv",
        "crates/sim_backend_metal/abi/execution-v1.json",
        "crates/sim_backend_metal/abi/reviewed-execution-bindings-v1.txt",
        "crates/sim_backend_metal/abi/symbols-v1.json",
        "crates/sim_backend_metal/kernels/readiness.metal",
        "crates/sim_backend_metal/kernels/tensor_ops.metal",
        "nix/sim-backends/metal/package-policy.json",
    ];
    paths
        .into_iter()
        .map(|path| {
            Ok(CertificationProvenance {
                path: path.to_owned(),
                sha256: sha256_file(&workspace.join(path))?,
            })
        })
        .collect()
}

fn descriptor(shape: Vec<u64>, dtype: DType) -> Result<TensorDescriptor, TensorError> {
    TensorDescriptor::contiguous(shape, dtype, metal_device(), StreamId::DEFAULT)
}

fn metal_device() -> DeviceId {
    DeviceId::new(DeviceKind::Metal, 0)
}

fn encode(dtype: DType, values: &[f32]) -> Result<Vec<u8>> {
    match dtype {
        DType::F16 => Ok(values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect()),
        DType::F32 => Ok(values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()),
        _ => bail!("Metal hardware fixture accepts f16/f32 only"),
    }
}

fn decode(dtype: DType, bytes: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|bytes| {
                let lane = <[u8; 2]>::try_from(bytes).context("invalid f16 lane")?;
                Ok(f16::from_bits(u16::from_le_bytes(lane)).to_f32())
            })
            .collect(),
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|bytes| {
                let lane = <[u8; 4]>::try_from(bytes).context("invalid f32 lane")?;
                Ok(f32::from_le_bytes(lane))
            })
            .collect(),
        _ => bail!("Metal hardware fixture accepts f16/f32 only"),
    }
}

fn attestation_trust_anchor() -> Result<DeviceCertificationTrustAnchor> {
    DeviceCertificationTrustAnchor::from_hex(ATTESTATION_SIGNER, ATTESTATION_PUBLIC_KEY)
        .context("construct the pinned Metal certification trust anchor")
}

fn load_signing_key(environment_variable: &str, purpose: &str) -> Result<Ed25519KeyPair> {
    let path = std::env::var_os(environment_variable).ok_or_else(|| {
        anyhow!(
            "{environment_variable} is required when updating the {purpose}; a fresh random signer is not trusted"
        )
    })?;
    load_device_certification_signing_key(Path::new(&path))
        .with_context(|| format!("load the bounded non-symlink {purpose} PKCS#8 signing key"))
}

fn write_receipt(path: &Path, signature: &str) -> Result<()> {
    let bytes = format!(
        "{{\"schema_version\":1,\"algorithm\":\"ed25519\",\"signature\":\"{signature}\"}}\n"
    );
    fs::write(path, bytes).with_context(|| format!("write package receipt {}", path.display()))
}

fn run_checked(command: &mut Command, purpose: &str) -> Result<()> {
    let output =
        smol::block_on(command.output()).with_context(|| format!("run command to {purpose}"))?;
    if !output.status.success() {
        bail!(
            "failed to {purpose}: status={} stdout={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
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
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    encode_hex(&Sha256::digest(bytes))
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

fn decode_hex<const LENGTH: usize>(value: &str) -> Option<[u8; LENGTH]> {
    if value.len() != LENGTH.checked_mul(2)? {
        return None;
    }
    let mut output = [0_u8; LENGTH];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = decode_nibble(*pair.first()?)?;
        let low = decode_nibble(*pair.get(1)?)?;
        *output.get_mut(index)? = (high << 4) | low;
    }
    Some(output)
}

fn decode_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}
