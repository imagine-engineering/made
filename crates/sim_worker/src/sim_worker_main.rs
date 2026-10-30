use std::{env, path::PathBuf};

use sim_runtime::{
    NativeCudaPackageSettings, NativeDirectMlPackageSettings,
    NativeGeneralVideoCodecPackageSettings, NativeMetalPackageSettings, NativeMluPackageSettings,
    NativeNpuPackageSettings, NativeRocmPackageSettings, NativeXpuPackageSettings,
    PluginAuthorizationVerifier, WorkerBackendSelection,
};

const DEFAULT_WORKER_MEMORY_LIMIT_BYTES: u64 = 8 * 1024 * 1024 * 1024;
fn main() -> anyhow::Result<()> {
    let configuration = parse_configuration()?;
    smol::block_on(
        sim_worker::run_worker_process_with_backend_selection_and_video_codec_package(
            configuration.memory_limit_bytes,
            configuration.backend_selection,
            configuration.general_video_codec_package,
            configuration.plugin_authorization_verifier,
        ),
    )
}

struct WorkerConfiguration {
    memory_limit_bytes: u64,
    backend_selection: WorkerBackendSelection,
    general_video_codec_package: Option<NativeGeneralVideoCodecPackageSettings>,
    plugin_authorization_verifier: Option<PluginAuthorizationVerifier>,
}

fn parse_configuration() -> anyhow::Result<WorkerConfiguration> {
    parse_configuration_arguments(env::args_os().skip(1))
}

fn parse_configuration_arguments(
    mut arguments: impl Iterator<Item = std::ffi::OsString>,
) -> anyhow::Result<WorkerConfiguration> {
    let mut memory_limit_bytes = DEFAULT_WORKER_MEMORY_LIMIT_BYTES;
    let mut plugin_authorization_verifier = None;
    let mut backend = None;
    let mut backend_device_ordinal = None;
    let mut rocm_package_root = None;
    let mut rocm_package_signer = None;
    let mut rocm_package_public_key = None;
    let mut metal_package_root = None;
    let mut metal_package_signer = None;
    let mut metal_package_public_key = None;
    let mut mlu_package_root = None;
    let mut mlu_package_signer = None;
    let mut mlu_package_public_key = None;
    let mut npu_package_root = None;
    let mut npu_package_signer = None;
    let mut npu_package_public_key = None;
    let mut cuda_package_root = None;
    let mut cuda_package_signer = None;
    let mut cuda_package_public_key = None;
    let mut xpu_package_root = None;
    let mut xpu_package_signer = None;
    let mut xpu_package_public_key = None;
    let mut directml_package_root = None;
    let mut directml_package_signer = None;
    let mut directml_package_public_key = None;
    let mut video_codec_package_root = None;
    let mut video_codec_package_signer = None;
    let mut video_codec_package_public_key = None;
    while let Some(argument) = arguments.next() {
        if argument == "--memory-limit-bytes" {
            let value = arguments
                .next()
                .ok_or_else(|| anyhow::anyhow!("--memory-limit-bytes requires a value"))?;
            let value = value
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("memory limit must be UTF-8 decimal bytes"))?;
            memory_limit_bytes = value
                .parse::<u64>()
                .map_err(|error| anyhow::anyhow!("invalid worker memory limit: {error}"))?;
        } else if argument == "--backend" {
            let value = required_utf8_argument(&mut arguments, "--backend")?;
            if backend.replace(value).is_some() {
                return Err(anyhow::anyhow!("--backend was specified more than once"));
            }
        } else if argument == "--backend-device-ordinal" {
            let value = required_utf8_argument(&mut arguments, "--backend-device-ordinal")?;
            let value = value
                .parse::<u32>()
                .map_err(|error| anyhow::anyhow!("invalid backend device ordinal: {error}"))?;
            if backend_device_ordinal.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--backend-device-ordinal was specified more than once"
                ));
            }
        } else if argument == "--rocm-package-root" {
            let value = required_utf8_argument(&mut arguments, "--rocm-package-root")?;
            if rocm_package_root.replace(PathBuf::from(value)).is_some() {
                return Err(anyhow::anyhow!(
                    "--rocm-package-root was specified more than once"
                ));
            }
        } else if argument == "--rocm-package-signer" {
            let value = required_utf8_argument(&mut arguments, "--rocm-package-signer")?;
            if rocm_package_signer.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--rocm-package-signer was specified more than once"
                ));
            }
        } else if argument == "--rocm-package-public-key" {
            let value = required_utf8_argument(&mut arguments, "--rocm-package-public-key")?;
            if rocm_package_public_key.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--rocm-package-public-key was specified more than once"
                ));
            }
        } else if argument == "--metal-package-root" {
            let value = required_utf8_argument(&mut arguments, "--metal-package-root")?;
            if metal_package_root.replace(PathBuf::from(value)).is_some() {
                return Err(anyhow::anyhow!(
                    "--metal-package-root was specified more than once"
                ));
            }
        } else if argument == "--metal-package-signer" {
            let value = required_utf8_argument(&mut arguments, "--metal-package-signer")?;
            if metal_package_signer.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--metal-package-signer was specified more than once"
                ));
            }
        } else if argument == "--metal-package-public-key" {
            let value = required_utf8_argument(&mut arguments, "--metal-package-public-key")?;
            if metal_package_public_key.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--metal-package-public-key was specified more than once"
                ));
            }
        } else if argument == "--mlu-package-root" {
            let value = required_utf8_argument(&mut arguments, "--mlu-package-root")?;
            if mlu_package_root.replace(PathBuf::from(value)).is_some() {
                return Err(anyhow::anyhow!(
                    "--mlu-package-root was specified more than once"
                ));
            }
        } else if argument == "--mlu-package-signer" {
            let value = required_utf8_argument(&mut arguments, "--mlu-package-signer")?;
            if mlu_package_signer.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--mlu-package-signer was specified more than once"
                ));
            }
        } else if argument == "--mlu-package-public-key" {
            let value = required_utf8_argument(&mut arguments, "--mlu-package-public-key")?;
            if mlu_package_public_key.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--mlu-package-public-key was specified more than once"
                ));
            }
        } else if argument == "--npu-package-root" {
            let value = required_utf8_argument(&mut arguments, "--npu-package-root")?;
            if npu_package_root.replace(PathBuf::from(value)).is_some() {
                return Err(anyhow::anyhow!(
                    "--npu-package-root was specified more than once"
                ));
            }
        } else if argument == "--npu-package-signer" {
            let value = required_utf8_argument(&mut arguments, "--npu-package-signer")?;
            if npu_package_signer.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--npu-package-signer was specified more than once"
                ));
            }
        } else if argument == "--npu-package-public-key" {
            let value = required_utf8_argument(&mut arguments, "--npu-package-public-key")?;
            if npu_package_public_key.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--npu-package-public-key was specified more than once"
                ));
            }
        } else if argument == "--cuda-package-root" {
            let value = required_utf8_argument(&mut arguments, "--cuda-package-root")?;
            if cuda_package_root.replace(PathBuf::from(value)).is_some() {
                return Err(anyhow::anyhow!(
                    "--cuda-package-root was specified more than once"
                ));
            }
        } else if argument == "--cuda-package-signer" {
            let value = required_utf8_argument(&mut arguments, "--cuda-package-signer")?;
            if cuda_package_signer.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--cuda-package-signer was specified more than once"
                ));
            }
        } else if argument == "--cuda-package-public-key" {
            let value = required_utf8_argument(&mut arguments, "--cuda-package-public-key")?;
            if cuda_package_public_key.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--cuda-package-public-key was specified more than once"
                ));
            }
        } else if argument == "--xpu-package-root" {
            let value = required_utf8_argument(&mut arguments, "--xpu-package-root")?;
            if xpu_package_root.replace(PathBuf::from(value)).is_some() {
                return Err(anyhow::anyhow!(
                    "--xpu-package-root was specified more than once"
                ));
            }
        } else if argument == "--xpu-package-signer" {
            let value = required_utf8_argument(&mut arguments, "--xpu-package-signer")?;
            if xpu_package_signer.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--xpu-package-signer was specified more than once"
                ));
            }
        } else if argument == "--xpu-package-public-key" {
            let value = required_utf8_argument(&mut arguments, "--xpu-package-public-key")?;
            if xpu_package_public_key.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--xpu-package-public-key was specified more than once"
                ));
            }
        } else if argument == "--directml-package-root" {
            let value = required_utf8_argument(&mut arguments, "--directml-package-root")?;
            if directml_package_root
                .replace(PathBuf::from(value))
                .is_some()
            {
                return Err(anyhow::anyhow!(
                    "--directml-package-root was specified more than once"
                ));
            }
        } else if argument == "--directml-package-signer" {
            let value = required_utf8_argument(&mut arguments, "--directml-package-signer")?;
            if directml_package_signer.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--directml-package-signer was specified more than once"
                ));
            }
        } else if argument == "--directml-package-public-key" {
            let value = required_utf8_argument(&mut arguments, "--directml-package-public-key")?;
            if directml_package_public_key.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--directml-package-public-key was specified more than once"
                ));
            }
        } else if argument == "--video-codec-package-root" {
            let value = required_utf8_argument(&mut arguments, "--video-codec-package-root")?;
            if video_codec_package_root
                .replace(PathBuf::from(value))
                .is_some()
            {
                return Err(anyhow::anyhow!(
                    "--video-codec-package-root was specified more than once"
                ));
            }
        } else if argument == "--video-codec-package-signer" {
            let value = required_utf8_argument(&mut arguments, "--video-codec-package-signer")?;
            if video_codec_package_signer.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--video-codec-package-signer was specified more than once"
                ));
            }
        } else if argument == "--video-codec-package-public-key" {
            let value = required_utf8_argument(&mut arguments, "--video-codec-package-public-key")?;
            if video_codec_package_public_key.replace(value).is_some() {
                return Err(anyhow::anyhow!(
                    "--video-codec-package-public-key was specified more than once"
                ));
            }
        } else if argument == "--plugin-authorization-verification-key" {
            let value = arguments.next().ok_or_else(|| {
                anyhow::anyhow!("--plugin-authorization-verification-key requires a value")
            })?;
            let value = value.to_str().ok_or_else(|| {
                anyhow::anyhow!("plugin authorization verification key must be UTF-8 hex")
            })?;
            if plugin_authorization_verifier
                .replace(PluginAuthorizationVerifier::from_token(value)?)
                .is_some()
            {
                return Err(anyhow::anyhow!(
                    "plugin authorization verification key was specified more than once"
                ));
            }
        } else {
            return Err(anyhow::anyhow!("unknown sim-worker argument"));
        }
    }
    let xpu_authority_present = xpu_package_root.is_some()
        || xpu_package_signer.is_some()
        || xpu_package_public_key.is_some();
    let cuda_authority_present = cuda_package_root.is_some()
        || cuda_package_signer.is_some()
        || cuda_package_public_key.is_some();
    let backend = backend.as_deref().unwrap_or("cpu");
    if backend != "cuda" && cuda_authority_present {
        return Err(anyhow::anyhow!(
            "CUDA package authority requires CUDA backend selection"
        ));
    }
    let backend_selection = match backend {
        "cpu" => {
            if backend_device_ordinal.is_some()
                || rocm_package_root.is_some()
                || rocm_package_signer.is_some()
                || rocm_package_public_key.is_some()
                || metal_package_root.is_some()
                || metal_package_signer.is_some()
                || metal_package_public_key.is_some()
                || mlu_package_root.is_some()
                || mlu_package_signer.is_some()
                || mlu_package_public_key.is_some()
                || npu_package_root.is_some()
                || npu_package_signer.is_some()
                || npu_package_public_key.is_some()
                || directml_package_root.is_some()
                || directml_package_signer.is_some()
                || directml_package_public_key.is_some()
                || xpu_authority_present
            {
                return Err(anyhow::anyhow!(
                    "CPU selection cannot include accelerator package or device arguments"
                ));
            }
            WorkerBackendSelection::Cpu
        }
        "rocm" => {
            if metal_package_root.is_some()
                || metal_package_signer.is_some()
                || metal_package_public_key.is_some()
                || mlu_package_root.is_some()
                || mlu_package_signer.is_some()
                || mlu_package_public_key.is_some()
                || npu_package_root.is_some()
                || npu_package_signer.is_some()
                || npu_package_public_key.is_some()
                || directml_package_root.is_some()
                || directml_package_signer.is_some()
                || directml_package_public_key.is_some()
                || xpu_authority_present
            {
                return Err(anyhow::anyhow!(
                    "ROCm selection cannot include Metal, MLU, NPU, or DirectML package authority"
                ));
            }
            let package_root = rocm_package_root
                .ok_or_else(|| anyhow::anyhow!("ROCm selection requires --rocm-package-root"))?;
            let signer = rocm_package_signer
                .ok_or_else(|| anyhow::anyhow!("ROCm selection requires --rocm-package-signer"))?;
            let public_key = rocm_package_public_key.ok_or_else(|| {
                anyhow::anyhow!("ROCm selection requires --rocm-package-public-key")
            })?;
            let package =
                NativeRocmPackageSettings::from_public_authority(package_root, signer, &public_key)
                    .map_err(|error| anyhow::anyhow!(error))?;
            WorkerBackendSelection::Rocm {
                package,
                device_ordinal: backend_device_ordinal.unwrap_or(0),
            }
        }
        "metal" => {
            if backend_device_ordinal.is_some()
                || rocm_package_root.is_some()
                || rocm_package_signer.is_some()
                || rocm_package_public_key.is_some()
                || mlu_package_root.is_some()
                || mlu_package_signer.is_some()
                || mlu_package_public_key.is_some()
                || npu_package_root.is_some()
                || npu_package_signer.is_some()
                || npu_package_public_key.is_some()
                || directml_package_root.is_some()
                || directml_package_signer.is_some()
                || directml_package_public_key.is_some()
                || xpu_authority_present
            {
                return Err(anyhow::anyhow!(
                    "Metal selection cannot include an ordinal, ROCm, MLU, NPU, or DirectML package authority"
                ));
            }
            let package_root = metal_package_root
                .ok_or_else(|| anyhow::anyhow!("Metal selection requires --metal-package-root"))?;
            let signer = metal_package_signer.ok_or_else(|| {
                anyhow::anyhow!("Metal selection requires --metal-package-signer")
            })?;
            let public_key = metal_package_public_key.ok_or_else(|| {
                anyhow::anyhow!("Metal selection requires --metal-package-public-key")
            })?;
            let package = NativeMetalPackageSettings::from_public_authority(
                package_root,
                signer,
                &public_key,
            )
            .map_err(|error| anyhow::anyhow!(error))?;
            WorkerBackendSelection::Metal { package }
        }
        "mlu" => {
            if rocm_package_root.is_some()
                || rocm_package_signer.is_some()
                || rocm_package_public_key.is_some()
                || metal_package_root.is_some()
                || metal_package_signer.is_some()
                || metal_package_public_key.is_some()
                || npu_package_root.is_some()
                || npu_package_signer.is_some()
                || npu_package_public_key.is_some()
                || directml_package_root.is_some()
                || directml_package_signer.is_some()
                || directml_package_public_key.is_some()
                || xpu_authority_present
            {
                return Err(anyhow::anyhow!(
                    "MLU selection cannot include ROCm, Metal, NPU, or DirectML package authority"
                ));
            }
            let package_root = mlu_package_root
                .ok_or_else(|| anyhow::anyhow!("MLU selection requires --mlu-package-root"))?;
            let signer = mlu_package_signer
                .ok_or_else(|| anyhow::anyhow!("MLU selection requires --mlu-package-signer"))?;
            let public_key = mlu_package_public_key.ok_or_else(|| {
                anyhow::anyhow!("MLU selection requires --mlu-package-public-key")
            })?;
            let package =
                NativeMluPackageSettings::from_public_authority(package_root, signer, &public_key)
                    .map_err(|error| anyhow::anyhow!(error))?;
            WorkerBackendSelection::Mlu {
                package,
                device_ordinal: backend_device_ordinal.unwrap_or(0),
            }
        }
        "npu" => {
            if rocm_package_root.is_some()
                || rocm_package_signer.is_some()
                || rocm_package_public_key.is_some()
                || metal_package_root.is_some()
                || metal_package_signer.is_some()
                || metal_package_public_key.is_some()
                || mlu_package_root.is_some()
                || mlu_package_signer.is_some()
                || mlu_package_public_key.is_some()
                || directml_package_root.is_some()
                || directml_package_signer.is_some()
                || directml_package_public_key.is_some()
                || xpu_authority_present
            {
                return Err(anyhow::anyhow!(
                    "NPU selection cannot include ROCm, Metal, MLU, or DirectML package authority"
                ));
            }
            let package_root = npu_package_root
                .ok_or_else(|| anyhow::anyhow!("NPU selection requires --npu-package-root"))?;
            let signer = npu_package_signer
                .ok_or_else(|| anyhow::anyhow!("NPU selection requires --npu-package-signer"))?;
            let public_key = npu_package_public_key.ok_or_else(|| {
                anyhow::anyhow!("NPU selection requires --npu-package-public-key")
            })?;
            let package =
                NativeNpuPackageSettings::from_public_authority(package_root, signer, &public_key)
                    .map_err(|error| anyhow::anyhow!(error))?;
            WorkerBackendSelection::Npu {
                package,
                device_ordinal: backend_device_ordinal.unwrap_or(0),
            }
        }
        "cuda" => {
            if rocm_package_root.is_some()
                || rocm_package_signer.is_some()
                || rocm_package_public_key.is_some()
                || metal_package_root.is_some()
                || metal_package_signer.is_some()
                || metal_package_public_key.is_some()
                || mlu_package_root.is_some()
                || mlu_package_signer.is_some()
                || mlu_package_public_key.is_some()
                || npu_package_root.is_some()
                || npu_package_signer.is_some()
                || npu_package_public_key.is_some()
                || directml_package_root.is_some()
                || directml_package_signer.is_some()
                || directml_package_public_key.is_some()
                || xpu_authority_present
            {
                return Err(anyhow::anyhow!(
                    "CUDA selection cannot include ROCm, Metal, MLU, NPU, XPU, or DirectML package authority"
                ));
            }
            let package_root = cuda_package_root
                .ok_or_else(|| anyhow::anyhow!("CUDA selection requires --cuda-package-root"))?;
            let signer = cuda_package_signer
                .ok_or_else(|| anyhow::anyhow!("CUDA selection requires --cuda-package-signer"))?;
            let public_key = cuda_package_public_key.ok_or_else(|| {
                anyhow::anyhow!("CUDA selection requires --cuda-package-public-key")
            })?;
            let package =
                NativeCudaPackageSettings::from_public_authority(package_root, signer, &public_key)
                    .map_err(|error| anyhow::anyhow!(error))?;
            WorkerBackendSelection::Cuda {
                package,
                device_ordinal: backend_device_ordinal.unwrap_or(0),
            }
        }
        "xpu" => {
            if rocm_package_root.is_some()
                || rocm_package_signer.is_some()
                || rocm_package_public_key.is_some()
                || metal_package_root.is_some()
                || metal_package_signer.is_some()
                || metal_package_public_key.is_some()
                || mlu_package_root.is_some()
                || mlu_package_signer.is_some()
                || mlu_package_public_key.is_some()
                || npu_package_root.is_some()
                || npu_package_signer.is_some()
                || npu_package_public_key.is_some()
                || directml_package_root.is_some()
                || directml_package_signer.is_some()
                || directml_package_public_key.is_some()
            {
                return Err(anyhow::anyhow!(
                    "XPU selection cannot include ROCm, Metal, MLU, NPU, or DirectML package authority"
                ));
            }
            let package_root = xpu_package_root
                .ok_or_else(|| anyhow::anyhow!("XPU selection requires --xpu-package-root"))?;
            let signer = xpu_package_signer
                .ok_or_else(|| anyhow::anyhow!("XPU selection requires --xpu-package-signer"))?;
            let public_key = xpu_package_public_key.ok_or_else(|| {
                anyhow::anyhow!("XPU selection requires --xpu-package-public-key")
            })?;
            let package =
                NativeXpuPackageSettings::from_public_authority(package_root, signer, &public_key)
                    .map_err(|error| anyhow::anyhow!(error))?;
            WorkerBackendSelection::Xpu {
                package,
                device_ordinal: backend_device_ordinal.unwrap_or(0),
            }
        }
        "directml" => {
            if backend_device_ordinal.is_some()
                || rocm_package_root.is_some()
                || rocm_package_signer.is_some()
                || rocm_package_public_key.is_some()
                || metal_package_root.is_some()
                || metal_package_signer.is_some()
                || metal_package_public_key.is_some()
                || mlu_package_root.is_some()
                || mlu_package_signer.is_some()
                || mlu_package_public_key.is_some()
                || npu_package_root.is_some()
                || npu_package_signer.is_some()
                || npu_package_public_key.is_some()
                || xpu_authority_present
            {
                return Err(anyhow::anyhow!(
                    "DirectML selection cannot include an ordinal, ROCm, Metal, MLU, or NPU package authority"
                ));
            }
            let package_root = directml_package_root.ok_or_else(|| {
                anyhow::anyhow!("DirectML selection requires --directml-package-root")
            })?;
            let signer = directml_package_signer.ok_or_else(|| {
                anyhow::anyhow!("DirectML selection requires --directml-package-signer")
            })?;
            let public_key = directml_package_public_key.ok_or_else(|| {
                anyhow::anyhow!("DirectML selection requires --directml-package-public-key")
            })?;
            let package = NativeDirectMlPackageSettings::from_public_authority(
                package_root,
                signer,
                &public_key,
            )
            .map_err(|error| anyhow::anyhow!(error))?;
            WorkerBackendSelection::DirectMl { package }
        }
        value => return Err(anyhow::anyhow!("unsupported worker backend {value}")),
    };
    let general_video_codec_package = match (
        video_codec_package_root,
        video_codec_package_signer,
        video_codec_package_public_key,
    ) {
        (None, None, None) => None,
        (Some(package_root), Some(signer), Some(public_key)) => Some(
            NativeGeneralVideoCodecPackageSettings::from_public_authority(
                package_root,
                signer,
                &public_key,
            )
            .map_err(|error| anyhow::anyhow!(error))?,
        ),
        _ => {
            return Err(anyhow::anyhow!(
                "general video codec package root, signer, and public key must be specified together"
            ));
        }
    };
    Ok(WorkerConfiguration {
        memory_limit_bytes,
        backend_selection,
        general_video_codec_package,
        plugin_authorization_verifier,
    })
}

fn required_utf8_argument(
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
    name: &str,
) -> anyhow::Result<String> {
    arguments
        .next()
        .ok_or_else(|| anyhow::anyhow!("{name} requires a value"))?
        .into_string()
        .map_err(|_| anyhow::anyhow!("{name} requires a UTF-8 value"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codec_arguments() -> Vec<std::ffi::OsString> {
        vec![
            "--backend".into(),
            "cpu".into(),
            "--video-codec-package-root".into(),
            "/reviewed/general-video".into(),
            "--video-codec-package-signer".into(),
            "codec.release".into(),
            "--video-codec-package-public-key".into(),
            "11".repeat(32).into(),
        ]
    }

    #[test]
    fn video_codec_package_bootstrap_cli_is_backend_independent_and_complete() {
        let configuration = parse_configuration_arguments(codec_arguments().into_iter())
            .expect("complete general-video authority is accepted");
        assert!(matches!(
            configuration.backend_selection,
            WorkerBackendSelection::Cpu
        ));
        let package = configuration
            .general_video_codec_package
            .expect("codec authority is retained");
        assert_eq!(
            package.package_root(),
            std::path::Path::new("/reviewed/general-video")
        );
        assert_eq!(package.verification_key().signer(), "codec.release");

        for mask in 1_u8..7 {
            let mut arguments = vec!["--backend".into(), "cpu".into()];
            if mask & 1 != 0 {
                arguments.extend([
                    "--video-codec-package-root".into(),
                    "/reviewed/general-video".into(),
                ]);
            }
            if mask & 2 != 0 {
                arguments.extend([
                    "--video-codec-package-signer".into(),
                    "codec.release".into(),
                ]);
            }
            if mask & 4 != 0 {
                arguments.extend([
                    "--video-codec-package-public-key".into(),
                    "11".repeat(32).into(),
                ]);
            }
            assert!(parse_configuration_arguments(arguments.into_iter()).is_err());
        }
    }

    #[test]
    fn video_codec_package_bootstrap_cli_rejects_duplicates_and_fixture_authority() {
        let mut duplicate = codec_arguments();
        duplicate.extend([
            "--video-codec-package-signer".into(),
            "other.release".into(),
        ]);
        assert!(parse_configuration_arguments(duplicate.into_iter()).is_err());

        let mut fixture_signer = codec_arguments();
        let signer = fixture_signer
            .iter_mut()
            .position(|value| value == "codec.release")
            .expect("signer value is present");
        fixture_signer[signer] = "sim.fixture.general-video".into();
        assert!(parse_configuration_arguments(fixture_signer.into_iter()).is_err());

        let mut fixture_key = codec_arguments();
        let key = fixture_key
            .iter_mut()
            .position(|value| value == &std::ffi::OsString::from("11".repeat(32)))
            .expect("key value is present");
        fixture_key[key] =
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a".into();
        assert!(parse_configuration_arguments(fixture_key.into_iter()).is_err());
    }

    #[test]
    fn authorization_verifier_cli_projection_requires_exact_lowercase_hex()
    -> Result<(), sim_runtime::TrustError> {
        assert_eq!(
            PluginAuthorizationVerifier::from_token(&format!("1:{}", "01".repeat(32)))?
                .public_key_bytes(),
            &[1; 32]
        );
        for invalid in ["01".repeat(31), "01".repeat(33), "GG".repeat(32)] {
            assert!(PluginAuthorizationVerifier::from_token(&format!("1:{invalid}")).is_err());
        }
        assert!(
            PluginAuthorizationVerifier::from_token(&format!("0:{}", "01".repeat(32))).is_err()
        );
        Ok(())
    }

    #[test]
    fn rocm_cli_projection_requires_complete_public_authority() {
        let public_key_hex = "11".repeat(32);
        let arguments = [
            "--backend",
            "rocm",
            "--backend-device-ordinal",
            "3",
            "--rocm-package-root",
            "/reviewed/rocm",
            "--rocm-package-signer",
            "rocm.release",
            "--rocm-package-public-key",
            &public_key_hex,
        ]
        .into_iter()
        .map(std::ffi::OsString::from);
        let configuration =
            parse_configuration_arguments(arguments).expect("checked ROCm CLI projection");
        let WorkerBackendSelection::Rocm {
            package,
            device_ordinal,
        } = configuration.backend_selection
        else {
            panic!("ROCm CLI selected a different backend");
        };
        assert_eq!(device_ordinal, 3);
        assert_eq!(
            package.package_root(),
            std::path::Path::new("/reviewed/rocm")
        );
        assert_eq!(package.verification_key().signer(), "rocm.release");
    }

    #[test]
    fn cpu_cli_cannot_smuggle_rocm_authority() {
        let arguments = ["--backend", "cpu", "--rocm-package-root", "/reviewed/rocm"]
            .into_iter()
            .map(std::ffi::OsString::from);
        assert!(parse_configuration_arguments(arguments).is_err());
    }

    #[test]
    fn metal_cli_projection_requires_complete_public_authority() {
        let public_key_hex = "22".repeat(32);
        let arguments = [
            "--backend",
            "metal",
            "--metal-package-root",
            "/reviewed/metal",
            "--metal-package-signer",
            "metal.release",
            "--metal-package-public-key",
            &public_key_hex,
        ]
        .into_iter()
        .map(std::ffi::OsString::from);
        let configuration =
            parse_configuration_arguments(arguments).expect("checked Metal CLI projection");
        let WorkerBackendSelection::Metal { package } = configuration.backend_selection else {
            panic!("Metal CLI selected a different backend");
        };
        assert_eq!(
            package.package_root(),
            std::path::Path::new("/reviewed/metal")
        );
        assert_eq!(package.verification_key().signer(), "metal.release");
    }

    #[test]
    fn mlu_cli_projection_requires_complete_public_authority_and_ordinal() {
        let public_key_hex = "33".repeat(32);
        let arguments = [
            "--backend",
            "mlu",
            "--backend-device-ordinal",
            "4",
            "--mlu-package-root",
            "/reviewed/mlu",
            "--mlu-package-signer",
            "mlu.release",
            "--mlu-package-public-key",
            &public_key_hex,
        ]
        .into_iter()
        .map(std::ffi::OsString::from);
        let configuration =
            parse_configuration_arguments(arguments).expect("checked MLU CLI projection");
        let WorkerBackendSelection::Mlu {
            package,
            device_ordinal,
        } = configuration.backend_selection
        else {
            panic!("MLU CLI selected a different backend");
        };
        assert_eq!(device_ordinal, 4);
        assert_eq!(
            package.package_root(),
            std::path::Path::new("/reviewed/mlu")
        );
        assert_eq!(package.verification_key().signer(), "mlu.release");
    }

    #[test]
    fn npu_cli_projection_requires_complete_public_authority_and_ordinal() {
        let public_key_hex = "35".repeat(32);
        let arguments = [
            "--backend",
            "npu",
            "--backend-device-ordinal",
            "2",
            "--npu-package-root",
            "/reviewed/npu",
            "--npu-package-signer",
            "npu.release",
            "--npu-package-public-key",
            &public_key_hex,
        ]
        .into_iter()
        .map(std::ffi::OsString::from);
        let configuration =
            parse_configuration_arguments(arguments).expect("checked NPU CLI projection");
        let WorkerBackendSelection::Npu {
            package,
            device_ordinal,
        } = configuration.backend_selection
        else {
            panic!("NPU CLI selected a different backend");
        };
        assert_eq!(device_ordinal, 2);
        assert_eq!(
            package.package_root(),
            std::path::Path::new("/reviewed/npu")
        );
        assert_eq!(package.verification_key().signer(), "npu.release");

        for omitted_flag in [
            "--npu-package-root",
            "--npu-package-signer",
            "--npu-package-public-key",
        ] {
            let arguments = [
                ("--backend", "npu"),
                ("--npu-package-root", "/reviewed/npu"),
                ("--npu-package-signer", "npu.release"),
                ("--npu-package-public-key", public_key_hex.as_str()),
            ]
            .into_iter()
            .filter(|(flag, _)| *flag != omitted_flag)
            .flat_map(|(flag, value)| [flag, value])
            .map(std::ffi::OsString::from);
            assert!(parse_configuration_arguments(arguments).is_err());
        }
    }

    #[test]
    fn cuda_cli_projection_requires_complete_public_authority_and_ordinal() {
        let public_key_hex = "56".repeat(32);
        let arguments = [
            "--backend",
            "cuda",
            "--backend-device-ordinal",
            "3",
            "--cuda-package-root",
            "/reviewed/cuda",
            "--cuda-package-signer",
            "cuda.release",
            "--cuda-package-public-key",
            &public_key_hex,
        ]
        .into_iter()
        .map(std::ffi::OsString::from);
        let configuration =
            parse_configuration_arguments(arguments).expect("checked CUDA CLI projection");
        let WorkerBackendSelection::Cuda {
            package,
            device_ordinal,
        } = configuration.backend_selection
        else {
            panic!("CUDA CLI selected a different backend");
        };
        assert_eq!(device_ordinal, 3);
        assert_eq!(
            package.package_root(),
            std::path::Path::new("/reviewed/cuda")
        );
        assert_eq!(package.verification_key().signer(), "cuda.release");

        for omitted_flag in [
            "--cuda-package-root",
            "--cuda-package-signer",
            "--cuda-package-public-key",
        ] {
            let arguments = [
                ("--backend", "cuda"),
                ("--cuda-package-root", "/reviewed/cuda"),
                ("--cuda-package-signer", "cuda.release"),
                ("--cuda-package-public-key", public_key_hex.as_str()),
            ]
            .into_iter()
            .filter(|(flag, _)| *flag != omitted_flag)
            .flat_map(|(flag, value)| [flag, value])
            .map(std::ffi::OsString::from);
            assert!(parse_configuration_arguments(arguments).is_err());
        }
    }

    #[test]
    fn xpu_cli_projection_requires_complete_public_authority_and_ordinal() {
        let public_key_hex = "46".repeat(32);
        let arguments = [
            "--backend",
            "xpu",
            "--backend-device-ordinal",
            "5",
            "--xpu-package-root",
            "/reviewed/xpu",
            "--xpu-package-signer",
            "xpu.release",
            "--xpu-package-public-key",
            &public_key_hex,
        ]
        .into_iter()
        .map(std::ffi::OsString::from);
        let configuration =
            parse_configuration_arguments(arguments).expect("checked XPU CLI projection");
        let WorkerBackendSelection::Xpu {
            package,
            device_ordinal,
        } = configuration.backend_selection
        else {
            panic!("XPU CLI selected a different backend");
        };
        assert_eq!(device_ordinal, 5);
        assert_eq!(
            package.package_root(),
            std::path::Path::new("/reviewed/xpu")
        );
        assert_eq!(package.verification_key().signer(), "xpu.release");

        for omitted_flag in [
            "--xpu-package-root",
            "--xpu-package-signer",
            "--xpu-package-public-key",
        ] {
            let arguments = [
                ("--backend", "xpu"),
                ("--xpu-package-root", "/reviewed/xpu"),
                ("--xpu-package-signer", "xpu.release"),
                ("--xpu-package-public-key", public_key_hex.as_str()),
            ]
            .into_iter()
            .filter(|(flag, _)| *flag != omitted_flag)
            .flat_map(|(flag, value)| [flag, value])
            .map(std::ffi::OsString::from);
            assert!(parse_configuration_arguments(arguments).is_err());
        }
    }

    #[test]
    fn directml_cli_projection_requires_exact_public_authority_and_fixed_ordinal() {
        let public_key_hex = "44".repeat(32);
        let arguments = [
            "--backend",
            "directml",
            "--directml-package-root",
            "/reviewed/directml",
            "--directml-package-signer",
            "directml.release",
            "--directml-package-public-key",
            &public_key_hex,
        ]
        .into_iter()
        .map(std::ffi::OsString::from);
        let configuration =
            parse_configuration_arguments(arguments).expect("checked DirectML CLI projection");
        let WorkerBackendSelection::DirectMl { package } = configuration.backend_selection else {
            panic!("DirectML CLI selected a different backend");
        };
        assert_eq!(
            package.package_root(),
            std::path::Path::new("/reviewed/directml")
        );
        assert_eq!(package.verification_key().signer(), "directml.release");

        for omitted_flag in [
            "--directml-package-root",
            "--directml-package-signer",
            "--directml-package-public-key",
        ] {
            let arguments = [
                ("--backend", "directml"),
                ("--directml-package-root", "/reviewed/directml"),
                ("--directml-package-signer", "directml.release"),
                ("--directml-package-public-key", public_key_hex.as_str()),
            ]
            .into_iter()
            .filter(|(flag, _)| *flag != omitted_flag)
            .flat_map(|(flag, value)| [flag, value])
            .map(std::ffi::OsString::from);
            assert!(parse_configuration_arguments(arguments).is_err());
        }
    }

    #[test]
    fn directml_cli_rejects_device_ordinal() {
        let public_key_hex = "44".repeat(32);
        let arguments = [
            "--backend",
            "directml",
            "--backend-device-ordinal",
            "0",
            "--directml-package-root",
            "/reviewed/directml",
            "--directml-package-signer",
            "directml.release",
            "--directml-package-public-key",
            &public_key_hex,
        ]
        .into_iter()
        .map(std::ffi::OsString::from);
        assert!(parse_configuration_arguments(arguments).is_err());
    }

    #[test]
    fn backend_authorities_are_mutually_exclusive() {
        let public_key_hex = "22".repeat(32);
        let arguments = [
            "--backend",
            "metal",
            "--rocm-package-root",
            "/reviewed/rocm",
            "--metal-package-root",
            "/reviewed/metal",
            "--metal-package-signer",
            "metal.release",
            "--metal-package-public-key",
            &public_key_hex,
        ]
        .into_iter()
        .map(std::ffi::OsString::from);
        assert!(parse_configuration_arguments(arguments).is_err());

        let arguments = [
            "--backend",
            "mlu",
            "--metal-package-root",
            "/reviewed/metal",
            "--mlu-package-root",
            "/reviewed/mlu",
            "--mlu-package-signer",
            "mlu.release",
            "--mlu-package-public-key",
            &public_key_hex,
        ]
        .into_iter()
        .map(std::ffi::OsString::from);
        assert!(parse_configuration_arguments(arguments).is_err());

        let arguments = [
            "--backend",
            "directml",
            "--mlu-package-root",
            "/reviewed/mlu",
            "--directml-package-root",
            "/reviewed/directml",
            "--directml-package-signer",
            "directml.release",
            "--directml-package-public-key",
            &public_key_hex,
        ]
        .into_iter()
        .map(std::ffi::OsString::from);
        assert!(parse_configuration_arguments(arguments).is_err());
    }
}
