use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::{File, OpenOptions},
    io::Read,
    net::IpAddr,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use ring::{
    rand::{SecureRandom, SystemRandom},
    signature::{ED25519, Ed25519KeyPair, KeyPair, UnparsedPublicKey},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use url::{Host, Url};
use zeroize::Zeroizing;

#[cfg(any(
    test,
    feature = "directml",
    feature = "metal",
    feature = "mlu",
    feature = "npu",
    feature = "cuda",
    feature = "xpu"
))]
use sim_model::ArtifactRoot;
use sim_model::artifact_index::CapturedPrivateArtifact;
use sim_plugin_sdk::{
    ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES, PLUGIN_SIGNATURE_ALGORITHM,
    PluginContractError, PluginManifest, ProviderBindingSet, ProviderHttpMethodV2,
    ProviderPluginManifestV2, ProviderStreamingContractV2, TypeRegistry,
};
use sim_tensor::CancellationToken;
use sim_types::CancellationError;

#[cfg(any(test, target_os = "linux", target_os = "windows"))]
use std::io::{Seek, SeekFrom, Write};

use crate::{
    AuthorizedCapabilities, CapabilitySet, PermissionError, PermissionPolicy,
    PermissionPolicyGeneration,
    native_ffi_elf::{NativeElfInspectionError, inspect_elf64_dynamic_contract},
    native_video_codec_abi::{
        general_video_codec_library_contracts, general_video_codec_library_extensions,
        general_video_codec_symbol_version_namespace, video_codec_library_contracts,
        video_codec_symbol_version_namespace,
    },
};

pub const SEALED_PLUGIN_AUTHORIZATION_VERSION: u16 = 2;
pub const MAX_SEALED_PLUGIN_AUTHORIZATION_BYTES: usize = 2 * 1024 * 1024;
pub const CUDART_LIBRARY_ID: &str = "nvidia-cudart";
const PLUGIN_AUTHORIZATION_SIGNATURE_DOMAIN: &[u8] = b"zed-sim-plugin-authorization-v2\0";
const PROVIDER_COST_ACCEPTANCE_SIGNATURE_DOMAIN: &[u8] = b"zed-sim-provider-cost-acceptance-v2\0";
const PROVIDER_RESULT_RECEIPT_SIGNATURE_DOMAIN: &[u8] = b"zed-sim-provider-result-receipt-v1\0";
const PROVIDER_RUNTIME_RECEIPT_SIGNATURE_DOMAIN_V2: &[u8] =
    b"zed-sim-provider-runtime-receipt-v2\0";
pub const PROVIDER_RUNTIME_RECEIPT_VERSION_V2: u16 = 2;
const MAX_PROVIDER_COST_ACCEPTANCE_LIFETIME: Duration = Duration::from_secs(24 * 60 * 60);
pub const MAX_PROVIDER_RESULT_RECEIPT_LIFETIME: Duration = Duration::from_secs(24 * 60 * 60);
pub const MAX_PROVIDER_RESULT_RECEIPT_BYTES: usize = 32 * 1024;
const ROCM_PACKAGE_SIGNATURE_DOMAIN: &[u8] = b"zed-sim-rocm-package-v1\0";
const METAL_PACKAGE_SIGNATURE_DOMAIN: &[u8] = b"zed-sim-metal-package-v1\0";
const MLU_PACKAGE_SIGNATURE_DOMAIN: &[u8] = b"zed-sim-mlu-package-v1\0";
const NPU_PACKAGE_SIGNATURE_DOMAIN: &[u8] = b"zed-sim-npu-package-v1\0";
const CUDA_PACKAGE_SIGNATURE_DOMAIN: &[u8] = b"zed-sim-cuda-package-v1\0";
const XPU_PACKAGE_SIGNATURE_DOMAIN: &[u8] = b"zed-sim-xpu-package-v1\0";
const DIRECTML_PACKAGE_SIGNATURE_DOMAIN: &[u8] = b"zed-sim-directml-package-v1\0";
const VIDEO_CODEC_PACKAGE_SIGNATURE_DOMAIN: &[u8] = b"zed-sim-video-codec-package-v1\0";
pub(crate) const GENERAL_VIDEO_CODEC_PACKAGE_SIGNATURE_DOMAIN: &[u8] =
    b"zed-sim-general-video-codec-package-v1\0";
pub(crate) const GENERAL_VIDEO_CODEC_DEPENDENCY_CONTRACT_SIGNATURE_DOMAIN: &[u8] =
    b"zed-sim-general-video-codec-dependency-contract-v1\0";
const VIDEO_CODEC_DEPENDENCY_CONTRACT_SIGNATURE_DOMAIN: &[u8] =
    b"zed-sim-video-codec-dependency-contract-v1\0";
const NATIVE_PACKAGE_SIGNATURE_ALGORITHM: &str = "ed25519";
const MAX_NATIVE_PACKAGE_SIGNATURE_RECEIPT_BYTES: usize = 1_024;
const MAX_NATIVE_LIBRARY_IMAGE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const NATIVE_LIBRARY_IMAGE_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevel {
    Restricted,
    Blocked,
}

#[derive(Clone, Eq, PartialEq)]
pub struct PluginVerificationKey {
    key_id: String,
    public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
}

impl PluginVerificationKey {
    pub fn new(key_id: impl Into<String>, key: impl AsRef<[u8]>) -> Result<Self, TrustError> {
        let key_id = key_id.into();
        let public_key = key
            .as_ref()
            .try_into()
            .map_err(|_| TrustError::InvalidVerificationKey)?;
        if !valid_ascii_identifier(&key_id, 1_024) {
            return Err(TrustError::InvalidVerificationKey);
        }
        Ok(Self { key_id, public_key })
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub fn public_key_bytes(&self) -> &[u8; ED25519_PUBLIC_KEY_BYTES] {
        &self.public_key
    }
}

impl fmt::Debug for PluginVerificationKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PluginVerificationKey")
            .field("key_id", &self.key_id)
            .field("public_key", &encode_hex(&self.public_key))
            .finish()
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct RocmPackageVerificationKey {
    authority: NativePackageVerificationAuthority,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct NativePackageSignatureReceipt {
    schema_version: u16,
    algorithm: String,
    signature: String,
}

#[derive(Clone, Eq, PartialEq)]
pub struct MetalPackageVerificationKey {
    authority: NativePackageVerificationAuthority,
}

#[derive(Clone, Eq, PartialEq)]
pub struct MluPackageVerificationKey {
    authority: NativePackageVerificationAuthority,
}

#[derive(Clone, Eq, PartialEq)]
pub struct NpuPackageVerificationKey {
    authority: NativePackageVerificationAuthority,
}

#[derive(Clone, Eq, PartialEq)]
pub struct CudaPackageVerificationKey {
    authority: NativePackageVerificationAuthority,
}

#[derive(Clone, Eq, PartialEq)]
pub struct XpuPackageVerificationKey {
    authority: NativePackageVerificationAuthority,
}

#[derive(Clone, Eq, PartialEq)]
pub struct DirectMlPackageVerificationKey {
    authority: NativePackageVerificationAuthority,
}

#[derive(Clone, Eq, PartialEq)]
pub struct VideoCodecPackageVerificationKey {
    authority: NativePackageVerificationAuthority,
}

#[derive(Clone, Eq, PartialEq)]
pub struct GeneralVideoCodecPackageVerificationKey {
    authority: NativePackageVerificationAuthority,
}

pub(crate) const GENERAL_VIDEO_CODEC_FIXTURE_SIGNER: &str = "sim.fixture.general-video";
pub(crate) const GENERAL_VIDEO_CODEC_FIXTURE_PUBLIC_KEY_HEX: &str =
    "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";

#[derive(Clone, Eq, PartialEq)]
struct NativePackageVerificationAuthority {
    signer: String,
    public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
}

impl NativePackageVerificationAuthority {
    fn new(
        signer: impl Into<String>,
        key: impl AsRef<[u8]>,
        invalid_key: TrustError,
    ) -> Result<Self, TrustError> {
        let signer = signer.into();
        let public_key = key.as_ref().try_into().map_err(|_| invalid_key.clone())?;
        if !valid_ascii_identifier(&signer, 256) {
            return Err(invalid_key);
        }
        Ok(Self { signer, public_key })
    }

    fn verify(
        &self,
        domain: &[u8],
        signer: &str,
        coverage: &[u8],
        receipt_bytes: &[u8],
        unknown_signer: TrustError,
        invalid_signature: TrustError,
    ) -> Result<(), TrustError> {
        if signer != self.signer {
            return Err(unknown_signer);
        }
        if receipt_bytes.is_empty()
            || receipt_bytes.len() > MAX_NATIVE_PACKAGE_SIGNATURE_RECEIPT_BYTES
        {
            return Err(invalid_signature);
        }
        let receipt_value =
            parse_strict_json_value(receipt_bytes).map_err(|_| invalid_signature.clone())?;
        let receipt: NativePackageSignatureReceipt =
            serde_json::from_value(receipt_value).map_err(|_| invalid_signature.clone())?;
        if receipt.schema_version != 1 || receipt.algorithm != NATIVE_PACKAGE_SIGNATURE_ALGORITHM {
            return Err(invalid_signature);
        }
        let signature = decode_hex_exact::<ED25519_SIGNATURE_BYTES>(&receipt.signature)
            .ok_or_else(|| invalid_signature.clone())?;
        let mut canonical_receipt =
            serde_json::to_vec(&receipt).map_err(|_| invalid_signature.clone())?;
        canonical_receipt.push(b'\n');
        if receipt_bytes != canonical_receipt {
            return Err(invalid_signature);
        }
        let signing_payload =
            package_signing_payload(domain, signer, coverage, invalid_signature.clone())?;
        UnparsedPublicKey::new(&ED25519, self.public_key)
            .verify(&signing_payload, &signature)
            .map_err(|_| invalid_signature)
    }
}

impl RocmPackageVerificationKey {
    pub fn new(signer: impl Into<String>, key: impl AsRef<[u8]>) -> Result<Self, TrustError> {
        Ok(Self {
            authority: NativePackageVerificationAuthority::new(
                signer,
                key,
                TrustError::InvalidRocmPackageVerificationKey,
            )?,
        })
    }

    pub fn signer(&self) -> &str {
        &self.authority.signer
    }

    pub fn public_key_bytes(&self) -> &[u8; ED25519_PUBLIC_KEY_BYTES] {
        &self.authority.public_key
    }

    pub fn verify_package(
        &self,
        signer: &str,
        coverage: &[u8],
        receipt_bytes: &[u8],
    ) -> Result<(), TrustError> {
        self.authority.verify(
            ROCM_PACKAGE_SIGNATURE_DOMAIN,
            signer,
            coverage,
            receipt_bytes,
            TrustError::UnknownRocmPackageSigner,
            TrustError::InvalidRocmPackageSignature,
        )
    }
}

impl MetalPackageVerificationKey {
    pub fn new(signer: impl Into<String>, key: impl AsRef<[u8]>) -> Result<Self, TrustError> {
        Ok(Self {
            authority: NativePackageVerificationAuthority::new(
                signer,
                key,
                TrustError::InvalidMetalPackageVerificationKey,
            )?,
        })
    }

    pub fn signer(&self) -> &str {
        &self.authority.signer
    }

    pub fn public_key_bytes(&self) -> &[u8; ED25519_PUBLIC_KEY_BYTES] {
        &self.authority.public_key
    }

    pub fn verify_package(
        &self,
        signer: &str,
        coverage: &[u8],
        receipt_bytes: &[u8],
    ) -> Result<(), TrustError> {
        self.authority.verify(
            METAL_PACKAGE_SIGNATURE_DOMAIN,
            signer,
            coverage,
            receipt_bytes,
            TrustError::UnknownMetalPackageSigner,
            TrustError::InvalidMetalPackageSignature,
        )
    }
}

impl MluPackageVerificationKey {
    pub fn new(signer: impl Into<String>, key: impl AsRef<[u8]>) -> Result<Self, TrustError> {
        Ok(Self {
            authority: NativePackageVerificationAuthority::new(
                signer,
                key,
                TrustError::InvalidMluPackageVerificationKey,
            )?,
        })
    }

    pub fn signer(&self) -> &str {
        &self.authority.signer
    }

    pub fn public_key_bytes(&self) -> &[u8; ED25519_PUBLIC_KEY_BYTES] {
        &self.authority.public_key
    }

    pub fn verify_package(
        &self,
        signer: &str,
        coverage: &[u8],
        receipt_bytes: &[u8],
    ) -> Result<(), TrustError> {
        self.authority.verify(
            MLU_PACKAGE_SIGNATURE_DOMAIN,
            signer,
            coverage,
            receipt_bytes,
            TrustError::UnknownMluPackageSigner,
            TrustError::InvalidMluPackageSignature,
        )
    }
}

impl NpuPackageVerificationKey {
    pub fn new(signer: impl Into<String>, key: impl AsRef<[u8]>) -> Result<Self, TrustError> {
        Ok(Self {
            authority: NativePackageVerificationAuthority::new(
                signer,
                key,
                TrustError::InvalidNpuPackageVerificationKey,
            )?,
        })
    }

    pub fn signer(&self) -> &str {
        &self.authority.signer
    }

    pub fn public_key_bytes(&self) -> &[u8; ED25519_PUBLIC_KEY_BYTES] {
        &self.authority.public_key
    }

    pub fn verify_package(
        &self,
        signer: &str,
        coverage: &[u8],
        receipt_bytes: &[u8],
    ) -> Result<(), TrustError> {
        self.authority.verify(
            NPU_PACKAGE_SIGNATURE_DOMAIN,
            signer,
            coverage,
            receipt_bytes,
            TrustError::UnknownNpuPackageSigner,
            TrustError::InvalidNpuPackageSignature,
        )
    }
}

impl XpuPackageVerificationKey {
    pub fn new(signer: impl Into<String>, key: impl AsRef<[u8]>) -> Result<Self, TrustError> {
        Ok(Self {
            authority: NativePackageVerificationAuthority::new(
                signer,
                key,
                TrustError::InvalidXpuPackageVerificationKey,
            )?,
        })
    }

    pub fn signer(&self) -> &str {
        &self.authority.signer
    }

    pub fn public_key_bytes(&self) -> &[u8; ED25519_PUBLIC_KEY_BYTES] {
        &self.authority.public_key
    }

    pub fn verify_package(
        &self,
        signer: &str,
        coverage: &[u8],
        receipt_bytes: &[u8],
    ) -> Result<(), TrustError> {
        self.authority.verify(
            XPU_PACKAGE_SIGNATURE_DOMAIN,
            signer,
            coverage,
            receipt_bytes,
            TrustError::UnknownXpuPackageSigner,
            TrustError::InvalidXpuPackageSignature,
        )
    }
}

impl CudaPackageVerificationKey {
    pub fn new(signer: impl Into<String>, key: impl AsRef<[u8]>) -> Result<Self, TrustError> {
        Ok(Self {
            authority: NativePackageVerificationAuthority::new(
                signer,
                key,
                TrustError::InvalidCudaPackageVerificationKey,
            )?,
        })
    }

    pub fn signer(&self) -> &str {
        &self.authority.signer
    }

    pub fn public_key_bytes(&self) -> &[u8; ED25519_PUBLIC_KEY_BYTES] {
        &self.authority.public_key
    }

    pub fn verify_package(
        &self,
        signer: &str,
        coverage: &[u8],
        receipt_bytes: &[u8],
    ) -> Result<(), TrustError> {
        self.authority.verify(
            CUDA_PACKAGE_SIGNATURE_DOMAIN,
            signer,
            coverage,
            receipt_bytes,
            TrustError::UnknownCudaPackageSigner,
            TrustError::InvalidCudaPackageSignature,
        )
    }
}

impl DirectMlPackageVerificationKey {
    pub fn new(signer: impl Into<String>, key: impl AsRef<[u8]>) -> Result<Self, TrustError> {
        Ok(Self {
            authority: NativePackageVerificationAuthority::new(
                signer,
                key,
                TrustError::InvalidDirectMlPackageVerificationKey,
            )?,
        })
    }

    pub fn signer(&self) -> &str {
        &self.authority.signer
    }

    pub fn public_key_bytes(&self) -> &[u8; ED25519_PUBLIC_KEY_BYTES] {
        &self.authority.public_key
    }

    pub fn verify_package(
        &self,
        signer: &str,
        coverage: &[u8],
        receipt_bytes: &[u8],
    ) -> Result<(), TrustError> {
        self.authority.verify(
            DIRECTML_PACKAGE_SIGNATURE_DOMAIN,
            signer,
            coverage,
            receipt_bytes,
            TrustError::UnknownDirectMlPackageSigner,
            TrustError::InvalidDirectMlPackageSignature,
        )
    }
}

impl VideoCodecPackageVerificationKey {
    pub fn new(signer: impl Into<String>, key: impl AsRef<[u8]>) -> Result<Self, TrustError> {
        Ok(Self {
            authority: NativePackageVerificationAuthority::new(
                signer,
                key,
                TrustError::InvalidVideoCodecPackageVerificationKey,
            )?,
        })
    }

    pub fn signer(&self) -> &str {
        &self.authority.signer
    }

    pub fn public_key_bytes(&self) -> &[u8; ED25519_PUBLIC_KEY_BYTES] {
        &self.authority.public_key
    }

    pub fn verify_package(
        &self,
        signer: &str,
        coverage: &[u8],
        receipt_bytes: &[u8],
    ) -> Result<(), TrustError> {
        self.authority.verify(
            VIDEO_CODEC_PACKAGE_SIGNATURE_DOMAIN,
            signer,
            coverage,
            receipt_bytes,
            TrustError::UnknownVideoCodecPackageSigner,
            TrustError::InvalidVideoCodecPackageSignature,
        )
    }
}

impl GeneralVideoCodecPackageVerificationKey {
    pub fn new(signer: impl Into<String>, key: impl AsRef<[u8]>) -> Result<Self, TrustError> {
        Ok(Self {
            authority: NativePackageVerificationAuthority::new(
                signer,
                key,
                TrustError::InvalidGeneralVideoCodecPackageVerificationKey,
            )?,
        })
    }

    pub fn signer(&self) -> &str {
        &self.authority.signer
    }

    pub fn public_key_bytes(&self) -> &[u8; ED25519_PUBLIC_KEY_BYTES] {
        &self.authority.public_key
    }

    pub fn verify_package(
        &self,
        signer: &str,
        coverage: &[u8],
        receipt_bytes: &[u8],
    ) -> Result<(), TrustError> {
        self.authority.verify(
            GENERAL_VIDEO_CODEC_PACKAGE_SIGNATURE_DOMAIN,
            signer,
            coverage,
            receipt_bytes,
            TrustError::UnknownGeneralVideoCodecPackageSigner,
            TrustError::InvalidGeneralVideoCodecPackageSignature,
        )
    }

    pub fn verify_dependency_contract(
        &self,
        signer: &str,
        contract: &[u8],
        receipt_bytes: &[u8],
    ) -> Result<(), TrustError> {
        self.authority.verify(
            GENERAL_VIDEO_CODEC_DEPENDENCY_CONTRACT_SIGNATURE_DOMAIN,
            signer,
            contract,
            receipt_bytes,
            TrustError::UnknownGeneralVideoCodecPackageSigner,
            TrustError::InvalidGeneralVideoCodecPackageSignature,
        )
    }
}

impl fmt::Debug for RocmPackageVerificationKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RocmPackageVerificationKey")
            .field("signer", &self.authority.signer)
            .field("public_key", &encode_hex(&self.authority.public_key))
            .finish()
    }
}

impl fmt::Debug for MetalPackageVerificationKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetalPackageVerificationKey")
            .field("signer", &self.authority.signer)
            .field("public_key", &encode_hex(&self.authority.public_key))
            .finish()
    }
}

impl fmt::Debug for MluPackageVerificationKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MluPackageVerificationKey")
            .field("signer", &self.authority.signer)
            .field("public_key", &encode_hex(&self.authority.public_key))
            .finish()
    }
}

impl fmt::Debug for NpuPackageVerificationKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NpuPackageVerificationKey")
            .field("signer", &self.authority.signer)
            .field("public_key", &encode_hex(&self.authority.public_key))
            .finish()
    }
}

impl fmt::Debug for XpuPackageVerificationKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XpuPackageVerificationKey")
            .field("signer", &self.authority.signer)
            .field("public_key", &encode_hex(&self.authority.public_key))
            .finish()
    }
}

impl fmt::Debug for CudaPackageVerificationKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaPackageVerificationKey")
            .field("signer", &self.authority.signer)
            .field("public_key", &encode_hex(&self.authority.public_key))
            .finish()
    }
}

impl fmt::Debug for DirectMlPackageVerificationKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DirectMlPackageVerificationKey")
            .field("signer", &self.authority.signer)
            .field("public_key", &encode_hex(&self.authority.public_key))
            .finish()
    }
}

impl fmt::Debug for GeneralVideoCodecPackageVerificationKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeneralVideoCodecPackageVerificationKey")
            .field("signer", &self.authority.signer)
            .field("public_key", &encode_hex(&self.authority.public_key))
            .finish()
    }
}

#[cfg(test)]
pub(crate) fn rocm_package_signing_payload(
    signer: &str,
    coverage: &[u8],
) -> Result<Vec<u8>, TrustError> {
    package_signing_payload(
        ROCM_PACKAGE_SIGNATURE_DOMAIN,
        signer,
        coverage,
        TrustError::InvalidRocmPackageSignature,
    )
}

#[cfg(any(test, feature = "signing-tooling"))]
pub fn metal_package_signing_payload(signer: &str, coverage: &[u8]) -> Result<Vec<u8>, TrustError> {
    package_signing_payload(
        METAL_PACKAGE_SIGNATURE_DOMAIN,
        signer,
        coverage,
        TrustError::InvalidMetalPackageSignature,
    )
}

#[cfg(any(test, feature = "signing-tooling"))]
pub fn mlu_package_signing_payload(signer: &str, coverage: &[u8]) -> Result<Vec<u8>, TrustError> {
    package_signing_payload(
        MLU_PACKAGE_SIGNATURE_DOMAIN,
        signer,
        coverage,
        TrustError::InvalidMluPackageSignature,
    )
}

#[cfg(any(test, feature = "signing-tooling"))]
pub fn npu_package_signing_payload(signer: &str, coverage: &[u8]) -> Result<Vec<u8>, TrustError> {
    package_signing_payload(
        NPU_PACKAGE_SIGNATURE_DOMAIN,
        signer,
        coverage,
        TrustError::InvalidNpuPackageSignature,
    )
}

#[cfg(any(test, feature = "signing-tooling"))]
pub fn xpu_package_signing_payload(signer: &str, coverage: &[u8]) -> Result<Vec<u8>, TrustError> {
    package_signing_payload(
        XPU_PACKAGE_SIGNATURE_DOMAIN,
        signer,
        coverage,
        TrustError::InvalidXpuPackageSignature,
    )
}

#[cfg(any(test, feature = "signing-tooling"))]
pub fn cuda_package_signing_payload(signer: &str, coverage: &[u8]) -> Result<Vec<u8>, TrustError> {
    package_signing_payload(
        CUDA_PACKAGE_SIGNATURE_DOMAIN,
        signer,
        coverage,
        TrustError::InvalidCudaPackageSignature,
    )
}

#[cfg(any(test, feature = "signing-tooling"))]
pub fn directml_package_signing_payload(
    signer: &str,
    coverage: &[u8],
) -> Result<Vec<u8>, TrustError> {
    package_signing_payload(
        DIRECTML_PACKAGE_SIGNATURE_DOMAIN,
        signer,
        coverage,
        TrustError::InvalidDirectMlPackageSignature,
    )
}

#[cfg(any(test, feature = "signing-tooling"))]
pub fn video_codec_package_signing_payload(
    signer: &str,
    coverage: &[u8],
) -> Result<Vec<u8>, TrustError> {
    package_signing_payload(
        VIDEO_CODEC_PACKAGE_SIGNATURE_DOMAIN,
        signer,
        coverage,
        TrustError::InvalidVideoCodecPackageSignature,
    )
}

#[cfg(any(test, feature = "signing-tooling"))]
pub fn general_video_codec_package_signing_payload(
    signer: &str,
    coverage: &[u8],
) -> Result<Vec<u8>, TrustError> {
    package_signing_payload(
        GENERAL_VIDEO_CODEC_PACKAGE_SIGNATURE_DOMAIN,
        signer,
        coverage,
        TrustError::InvalidGeneralVideoCodecPackageSignature,
    )
}

#[cfg(any(test, feature = "signing-tooling"))]
pub fn general_video_codec_dependency_contract_signing_payload(
    signer: &str,
    contract: &[u8],
) -> Result<Vec<u8>, TrustError> {
    package_signing_payload(
        GENERAL_VIDEO_CODEC_DEPENDENCY_CONTRACT_SIGNATURE_DOMAIN,
        signer,
        contract,
        TrustError::InvalidGeneralVideoCodecPackageSignature,
    )
}

#[cfg(any(test, feature = "signing-tooling"))]
pub fn video_codec_dependency_contract_signing_payload(
    signer: &str,
    coverage: &[u8],
) -> Result<Vec<u8>, TrustError> {
    package_signing_payload(
        VIDEO_CODEC_DEPENDENCY_CONTRACT_SIGNATURE_DOMAIN,
        signer,
        coverage,
        TrustError::InvalidVideoCodecPackageSignature,
    )
}

fn package_signing_payload(
    domain: &[u8],
    signer: &str,
    coverage: &[u8],
    invalid_signature: TrustError,
) -> Result<Vec<u8>, TrustError> {
    let signer_length = u64::try_from(signer.len()).map_err(|_| invalid_signature.clone())?;
    let coverage_length = u64::try_from(coverage.len()).map_err(|_| invalid_signature.clone())?;
    let capacity = domain
        .len()
        .checked_add(16)
        .and_then(|length| length.checked_add(signer.len()))
        .and_then(|length| length.checked_add(coverage.len()))
        .ok_or_else(|| invalid_signature.clone())?;
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(capacity)
        .map_err(|_| invalid_signature)?;
    payload.extend_from_slice(domain);
    payload.extend_from_slice(&signer_length.to_be_bytes());
    payload.extend_from_slice(signer.as_bytes());
    payload.extend_from_slice(&coverage_length.to_be_bytes());
    payload.extend_from_slice(coverage);
    Ok(payload)
}

#[cfg(any(
    test,
    feature = "directml",
    feature = "metal",
    feature = "mlu",
    feature = "npu",
    feature = "cuda",
    feature = "xpu"
))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NativePackagePayloadLimit {
    path: &'static str,
    maximum_bytes: usize,
}

#[derive(Debug, Error)]
pub(crate) enum NativeLibraryImageError {
    #[error("native-library image capture was cancelled")]
    Cancelled,
    #[error("native-library image capture or sealing is unsupported on this platform")]
    #[cfg_attr(any(unix, target_os = "windows"), allow(dead_code))]
    UnsupportedPlatform,
    #[error("native-library image is invalid: {0}")]
    Invalid(String),
}

impl From<CancellationError> for NativeLibraryImageError {
    fn from(_: CancellationError) -> Self {
        Self::Cancelled
    }
}

pub(crate) struct CapturedNativeLibraryImage {
    bytes: Vec<u8>,
    digest_sha256: String,
}

pub(crate) fn capture_native_library_bytes(
    captured: CapturedPrivateArtifact,
    expected_digest_sha256: &str,
) -> Result<CapturedNativeLibraryImage, NativeLibraryImageError> {
    let digest_sha256 = captured.digest_sha256().to_owned();
    let bytes = captured.into_bytes();
    if bytes.is_empty()
        || u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_NATIVE_LIBRARY_IMAGE_BYTES
    {
        return Err(NativeLibraryImageError::Invalid(
            "image must be nonempty and within the native-library byte bound".to_owned(),
        ));
    }
    if digest_sha256 != expected_digest_sha256 {
        return Err(NativeLibraryImageError::Invalid(
            "captured native-library digest differs from signed coverage".to_owned(),
        ));
    }
    Ok(CapturedNativeLibraryImage {
        bytes,
        digest_sha256,
    })
}

impl CapturedNativeLibraryImage {
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn digest_sha256(&self) -> &str {
        &self.digest_sha256
    }

    pub(crate) fn seal(
        self,
        snapshot_name: &str,
        cancellation: &CancellationToken,
    ) -> Result<RetainedNativeLibraryImage, NativeLibraryImageError> {
        self.seal_with_check(snapshot_name, || cancellation.check())
    }

    pub(crate) fn seal_with_check(
        self,
        snapshot_name: &str,
        mut check_cancellation: impl FnMut() -> Result<(), CancellationError>,
    ) -> Result<RetainedNativeLibraryImage, NativeLibraryImageError> {
        seal_native_library_image(&self.bytes, snapshot_name, &mut check_cancellation)
    }
}

#[cfg_attr(
    not(all(
        target_os = "linux",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )),
    allow(dead_code)
)]
pub(crate) struct RetainedNativeLibraryImage {
    _file: File,
    #[allow(dead_code)]
    loader_path: PathBuf,
    #[cfg(target_os = "windows")]
    _temporary_directory: tempfile::TempDir,
    #[cfg(all(test, not(any(target_os = "linux", target_os = "windows"))))]
    _temporary_path: tempfile::TempPath,
}

#[cfg_attr(
    not(all(
        target_os = "linux",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )),
    allow(dead_code)
)]
impl RetainedNativeLibraryImage {
    #[allow(dead_code)]
    pub(crate) fn loader_path(&self) -> &Path {
        &self.loader_path
    }

    #[cfg(test)]
    pub(crate) fn file(&self) -> &File {
        &self._file
    }
}

pub(crate) fn capture_native_library_image(
    path: &Path,
    cancellation: &CancellationToken,
) -> Result<CapturedNativeLibraryImage, NativeLibraryImageError> {
    capture_native_library_image_with_check(path, || cancellation.check())
}

pub(crate) fn capture_native_library_image_with_check(
    path: &Path,
    mut check_cancellation: impl FnMut() -> Result<(), CancellationError>,
) -> Result<CapturedNativeLibraryImage, NativeLibraryImageError> {
    capture_native_library_image_with_limit(
        path,
        MAX_NATIVE_LIBRARY_IMAGE_BYTES,
        &mut check_cancellation,
    )
}

#[cfg(unix)]
fn capture_native_library_image_with_limit(
    path: &Path,
    maximum_bytes: u64,
    check_cancellation: &mut impl FnMut() -> Result<(), CancellationError>,
) -> Result<CapturedNativeLibraryImage, NativeLibraryImageError> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

    check_cancellation()?;
    if maximum_bytes == 0 {
        return Err(NativeLibraryImageError::Invalid(
            "native-library byte bound must be nonzero".to_owned(),
        ));
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    let before = file
        .metadata()
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    if !before.is_file() || before.len() == 0 || before.len() > maximum_bytes {
        return Err(NativeLibraryImageError::Invalid(format!(
            "image must be a nonempty regular file no larger than {maximum_bytes} bytes"
        )));
    }
    let expected_length = usize::try_from(before.len()).map_err(|_| {
        NativeLibraryImageError::Invalid(
            "image length exceeds the process address space".to_owned(),
        )
    })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(expected_length.min(NATIVE_LIBRARY_IMAGE_CHUNK_BYTES))
        .map_err(|error| {
            NativeLibraryImageError::Invalid(format!("image buffer allocation failed: {error}"))
        })?;
    let mut hasher = Sha256::new();
    let mut chunk = [0_u8; NATIVE_LIBRARY_IMAGE_CHUNK_BYTES];
    loop {
        check_cancellation()?;
        let remaining = maximum_bytes
            .checked_add(1)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(usize::MAX)
            .saturating_sub(bytes.len());
        if remaining == 0 {
            break;
        }
        let chunk_limit = remaining.min(chunk.len());
        let read = file
            .read(&mut chunk[..chunk_limit])
            .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
        if read == 0 {
            break;
        }
        bytes.try_reserve_exact(read).map_err(|error| {
            NativeLibraryImageError::Invalid(format!("image buffer allocation failed: {error}"))
        })?;
        bytes.extend_from_slice(&chunk[..read]);
        hasher.update(&chunk[..read]);
    }
    check_cancellation()?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != before.len()
        || u64::try_from(bytes.len()).unwrap_or(u64::MAX) > maximum_bytes
    {
        return Err(NativeLibraryImageError::Invalid(
            "image length changed while it was captured".to_owned(),
        ));
    }
    let after = file
        .metadata()
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    if before.dev() != after.dev()
        || before.ino() != after.ino()
        || before.len() != after.len()
        || before.mtime() != after.mtime()
        || before.mtime_nsec() != after.mtime_nsec()
        || before.ctime() != after.ctime()
        || before.ctime_nsec() != after.ctime_nsec()
    {
        return Err(NativeLibraryImageError::Invalid(
            "image metadata changed while it was captured".to_owned(),
        ));
    }
    check_cancellation()?;
    Ok(CapturedNativeLibraryImage {
        bytes,
        digest_sha256: format!("{:x}", hasher.finalize()),
    })
}

#[cfg(target_os = "windows")]
fn capture_native_library_image_with_limit(
    path: &Path,
    maximum_bytes: u64,
    check_cancellation: &mut impl FnMut() -> Result<(), CancellationError>,
) -> Result<CapturedNativeLibraryImage, NativeLibraryImageError> {
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};

    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;

    check_cancellation()?;
    if maximum_bytes == 0 {
        return Err(NativeLibraryImageError::Invalid(
            "native-library byte bound must be nonzero".to_owned(),
        ));
    }
    let mut file = OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN)
        .open(path)
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    let before = file
        .metadata()
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    if !before.is_file()
        || before.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
        || before.len() == 0
        || before.len() > maximum_bytes
    {
        return Err(NativeLibraryImageError::Invalid(format!(
            "image must be a nonempty non-reparse regular file no larger than {maximum_bytes} bytes"
        )));
    }
    let first = read_bounded_native_library_image(
        &mut file,
        before.len(),
        maximum_bytes,
        check_cancellation,
    )?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    let second = read_bounded_native_library_image(
        &mut file,
        before.len(),
        maximum_bytes,
        check_cancellation,
    )?;
    if first != second {
        return Err(NativeLibraryImageError::Invalid(
            "image changed while it was captured".to_owned(),
        ));
    }
    let after = file
        .metadata()
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    if before.volume_serial_number() != after.volume_serial_number()
        || before.file_index() != after.file_index()
        || before.len() != after.len()
        || before.last_write_time() != after.last_write_time()
        || before.creation_time() != after.creation_time()
    {
        return Err(NativeLibraryImageError::Invalid(
            "image metadata changed while it was captured".to_owned(),
        ));
    }
    check_cancellation()?;
    Ok(CapturedNativeLibraryImage {
        digest_sha256: format!("{:x}", Sha256::digest(&first)),
        bytes: first,
    })
}

#[cfg(target_os = "windows")]
fn read_bounded_native_library_image(
    file: &mut File,
    expected_bytes: u64,
    maximum_bytes: u64,
    check_cancellation: &mut impl FnMut() -> Result<(), CancellationError>,
) -> Result<Vec<u8>, NativeLibraryImageError> {
    let capacity = usize::try_from(expected_bytes).map_err(|_| {
        NativeLibraryImageError::Invalid("image length exceeds addressable memory".to_owned())
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|error| {
        NativeLibraryImageError::Invalid(format!("image buffer allocation failed: {error}"))
    })?;
    let mut chunk = [0_u8; NATIVE_LIBRARY_IMAGE_CHUNK_BYTES];
    loop {
        check_cancellation()?;
        let count = file
            .read(&mut chunk)
            .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
        if count == 0 {
            break;
        }
        bytes.try_reserve_exact(count).map_err(|error| {
            NativeLibraryImageError::Invalid(format!("image buffer allocation failed: {error}"))
        })?;
        bytes.extend_from_slice(&chunk[..count]);
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > maximum_bytes {
            return Err(NativeLibraryImageError::Invalid(
                "image exceeds the native-library byte bound".to_owned(),
            ));
        }
    }
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != expected_bytes {
        return Err(NativeLibraryImageError::Invalid(
            "image length changed while it was captured".to_owned(),
        ));
    }
    Ok(bytes)
}

#[cfg(all(not(unix), not(target_os = "windows")))]
fn capture_native_library_image_with_limit(
    _path: &Path,
    _maximum_bytes: u64,
    _check_cancellation: &mut impl FnMut() -> Result<(), CancellationError>,
) -> Result<CapturedNativeLibraryImage, NativeLibraryImageError> {
    Err(NativeLibraryImageError::UnsupportedPlatform)
}

#[cfg(target_os = "linux")]
fn seal_native_library_image(
    bytes: &[u8],
    snapshot_name: &str,
    check_cancellation: &mut impl FnMut() -> Result<(), CancellationError>,
) -> Result<RetainedNativeLibraryImage, NativeLibraryImageError> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};

    check_cancellation()?;
    let name = CString::new(format!("zed-native-{snapshot_name}")).map_err(|error| {
        NativeLibraryImageError::Invalid(format!("snapshot name is invalid: {error}"))
    })?;
    let descriptor =
        unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
    if descriptor < 0 {
        return Err(NativeLibraryImageError::Invalid(
            std::io::Error::last_os_error().to_string(),
        ));
    }
    let mut file = unsafe { File::from_raw_fd(descriptor) };
    for chunk in bytes.chunks(NATIVE_LIBRARY_IMAGE_CHUNK_BYTES) {
        check_cancellation()?;
        file.write_all(chunk)
            .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    }
    check_cancellation()?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    let required_seals =
        libc::F_SEAL_SEAL | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE;
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, required_seals) } != 0 {
        return Err(NativeLibraryImageError::Invalid(
            std::io::Error::last_os_error().to_string(),
        ));
    }
    let actual_seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
    if actual_seals < 0 || actual_seals & required_seals != required_seals {
        return Err(NativeLibraryImageError::Invalid(
            "snapshot does not carry every required immutable seal".to_owned(),
        ));
    }
    let loader_path = PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()));
    Ok(RetainedNativeLibraryImage {
        _file: file,
        loader_path,
    })
}

#[cfg(target_os = "windows")]
fn seal_native_library_image(
    bytes: &[u8],
    snapshot_name: &str,
    check_cancellation: &mut impl FnMut() -> Result<(), CancellationError>,
) -> Result<RetainedNativeLibraryImage, NativeLibraryImageError> {
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_ATTRIBUTE_TEMPORARY: u32 = 0x0000_0100;
    const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;

    if snapshot_name.is_empty()
        || snapshot_name.len() > 128
        || !snapshot_name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(NativeLibraryImageError::Invalid(
            "snapshot name is invalid".to_owned(),
        ));
    }
    check_cancellation()?;
    let temporary_directory = tempfile::Builder::new()
        .prefix("zed-native-")
        .tempdir()
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    let loader_path = temporary_directory
        .path()
        .join(format!("{snapshot_name}.dll"));
    let mut writer = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .share_mode(FILE_SHARE_READ)
        .custom_flags(FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_SEQUENTIAL_SCAN)
        .open(&loader_path)
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    for chunk in bytes.chunks(NATIVE_LIBRARY_IMAGE_CHUNK_BYTES) {
        check_cancellation()?;
        writer
            .write_all(chunk)
            .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    }
    writer
        .sync_all()
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    drop(writer);
    let mut permissions = std::fs::metadata(&loader_path)
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?
        .permissions();
    permissions.set_readonly(true);
    std::fs::set_permissions(&loader_path, permissions)
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    let file = OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ)
        .custom_flags(FILE_FLAG_SEQUENTIAL_SCAN)
        .open(&loader_path)
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    check_cancellation()?;
    Ok(RetainedNativeLibraryImage {
        _file: file,
        loader_path,
        _temporary_directory: temporary_directory,
    })
}

#[cfg(all(test, not(any(target_os = "linux", target_os = "windows"))))]
fn seal_native_library_image(
    bytes: &[u8],
    _snapshot_name: &str,
    check_cancellation: &mut impl FnMut() -> Result<(), CancellationError>,
) -> Result<RetainedNativeLibraryImage, NativeLibraryImageError> {
    let mut named_file = tempfile::NamedTempFile::new()
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    for chunk in bytes.chunks(NATIVE_LIBRARY_IMAGE_CHUNK_BYTES) {
        check_cancellation()?;
        named_file
            .as_file_mut()
            .write_all(chunk)
            .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    }
    check_cancellation()?;
    named_file
        .as_file_mut()
        .seek(SeekFrom::Start(0))
        .map_err(|error| NativeLibraryImageError::Invalid(error.to_string()))?;
    let (file, temporary_path) = named_file.into_parts();
    let loader_path = temporary_path.to_path_buf();
    Ok(RetainedNativeLibraryImage {
        _file: file,
        loader_path,
        _temporary_path: temporary_path,
    })
}

#[cfg(all(not(target_os = "linux"), not(target_os = "windows"), not(test)))]
fn seal_native_library_image(
    _bytes: &[u8],
    _snapshot_name: &str,
    _check_cancellation: &mut impl FnMut() -> Result<(), CancellationError>,
) -> Result<RetainedNativeLibraryImage, NativeLibraryImageError> {
    Err(NativeLibraryImageError::UnsupportedPlatform)
}

#[cfg(any(
    test,
    feature = "directml",
    feature = "metal",
    feature = "mlu",
    feature = "npu",
    feature = "cuda",
    feature = "xpu"
))]
impl NativePackagePayloadLimit {
    pub(crate) const fn new(path: &'static str, maximum_bytes: usize) -> Self {
        Self {
            path,
            maximum_bytes,
        }
    }

    #[cfg(all(
        test,
        any(
            feature = "directml",
            feature = "metal",
            feature = "mlu",
            feature = "npu",
            feature = "cuda",
            feature = "xpu"
        )
    ))]
    pub(crate) const fn path(&self) -> &'static str {
        self.path
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub(crate) enum NativePackageAdmissionError {
    #[cfg_attr(
        not(any(
            test,
            feature = "directml",
            feature = "metal",
            feature = "mlu",
            feature = "npu",
            feature = "cuda",
            feature = "xpu"
        )),
        expect(
            dead_code,
            reason = "constructed by feature-gated canonical native-package capture"
        )
    )]
    #[error("native package admission was cancelled")]
    Cancelled,
    #[cfg_attr(
        not(any(
            test,
            feature = "directml",
            feature = "metal",
            feature = "mlu",
            feature = "npu",
            feature = "cuda",
            feature = "xpu"
        )),
        expect(
            dead_code,
            reason = "constructed by feature-gated canonical native-package capture"
        )
    )]
    #[error("native package tree is unsafe or incomplete: {0}")]
    UnsafePackage(String),
    #[error("native package signature coverage is invalid: {0}")]
    InvalidCoverage(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NativePackageCoverageEntry {
    pub(crate) digest_sha256: String,
    pub(crate) size: u64,
}

pub(crate) fn parse_native_package_coverage(
    coverage: &[u8],
    excludes: &[&str],
    maximum_bytes: usize,
) -> Result<BTreeMap<String, NativePackageCoverageEntry>, NativePackageAdmissionError> {
    if maximum_bytes == 0
        || coverage.is_empty()
        || coverage.len() > maximum_bytes
        || coverage.last() != Some(&b'\n')
        || !coverage.is_ascii()
    {
        return Err(NativePackageAdmissionError::InvalidCoverage(
            "coverage must be bounded nonempty newline-terminated ASCII".to_owned(),
        ));
    }
    if excludes.windows(2).any(|pair| pair[0] >= pair[1])
        || excludes.iter().any(|path| !valid_native_package_path(path))
    {
        return Err(NativePackageAdmissionError::InvalidCoverage(
            "coverage exclusions must be strict, sorted package paths".to_owned(),
        ));
    }
    let text = std::str::from_utf8(coverage).map_err(|_| {
        NativePackageAdmissionError::InvalidCoverage("coverage must be UTF-8".to_owned())
    })?;
    let mut entries = BTreeMap::new();
    let mut previous_path: Option<&str> = None;
    for line in text.lines() {
        let (digest_and_size, path) = line.split_once("  ").ok_or_else(|| {
            NativePackageAdmissionError::InvalidCoverage(
                "coverage row must contain digest, size, and path".to_owned(),
            )
        })?;
        let (digest, size_text) = digest_and_size.split_once(' ').ok_or_else(|| {
            NativePackageAdmissionError::InvalidCoverage(
                "coverage row has an invalid digest/size separator".to_owned(),
            )
        })?;
        let size = size_text.parse::<u64>().map_err(|_| {
            NativePackageAdmissionError::InvalidCoverage(
                "coverage size is not canonical decimal".to_owned(),
            )
        })?;
        if !valid_lower_hex_sha256(digest)
            || size.to_string() != size_text
            || !valid_native_package_path(path)
            || excludes.binary_search(&path).is_ok()
            || previous_path.is_some_and(|previous| previous >= path)
            || entries
                .insert(
                    path.to_owned(),
                    NativePackageCoverageEntry {
                        digest_sha256: digest.to_owned(),
                        size,
                    },
                )
                .is_some()
        {
            return Err(NativePackageAdmissionError::InvalidCoverage(
                "coverage rows must be canonical, strictly sorted, unique, and non-excluded"
                    .to_owned(),
            ));
        }
        previous_path = Some(path);
    }
    if entries.is_empty() {
        return Err(NativePackageAdmissionError::InvalidCoverage(
            "coverage must contain at least one payload".to_owned(),
        ));
    }
    Ok(entries)
}

#[cfg(any(
    test,
    feature = "directml",
    feature = "metal",
    feature = "mlu",
    feature = "npu",
    feature = "cuda",
    feature = "xpu"
))]
pub(crate) fn capture_native_package(
    root: &ArtifactRoot,
    limits: &[NativePackagePayloadLimit],
    maximum_entries: usize,
    maximum_total_bytes: usize,
    cancellation: &CancellationToken,
) -> Result<BTreeMap<String, Vec<u8>>, NativePackageAdmissionError> {
    capture_native_package_with_hook(
        root,
        limits,
        maximum_entries,
        maximum_total_bytes,
        cancellation,
        || Ok(()),
    )
}

#[cfg(any(
    test,
    feature = "directml",
    feature = "metal",
    feature = "mlu",
    feature = "npu",
    feature = "cuda",
    feature = "xpu"
))]
fn capture_native_package_with_hook(
    root: &ArtifactRoot,
    limits: &[NativePackagePayloadLimit],
    maximum_entries: usize,
    maximum_total_bytes: usize,
    cancellation: &CancellationToken,
    after_capture: impl FnOnce() -> Result<(), NativePackageAdmissionError>,
) -> Result<BTreeMap<String, Vec<u8>>, NativePackageAdmissionError> {
    check_package_admission_cancellation(cancellation)?;
    if limits.is_empty() || maximum_entries == 0 || maximum_total_bytes == 0 {
        return Err(NativePackageAdmissionError::UnsafePackage(
            "package admission bounds and payload set must be nonzero".to_owned(),
        ));
    }
    let expected_paths = limits
        .iter()
        .map(|limit| limit.path.to_owned())
        .collect::<Vec<_>>();
    if limits
        .iter()
        .any(|limit| limit.maximum_bytes == 0 || !valid_native_package_path(limit.path))
        || expected_paths
            .windows(2)
            .any(|pair| matches!(pair, [left, right] if left >= right))
    {
        return Err(NativePackageAdmissionError::UnsafePackage(
            "package payload limits are not a strict sorted canonical set".to_owned(),
        ));
    }

    let observed_paths = list_native_package_paths(root, maximum_entries, cancellation)?;
    if observed_paths != expected_paths {
        return Err(NativePackageAdmissionError::UnsafePackage(
            "payload membership differs from the exact signed package contract".to_owned(),
        ));
    }

    let mut payloads = BTreeMap::new();
    let mut total_bytes = 0_usize;
    for limit in limits {
        check_package_admission_cancellation(cancellation)?;
        let bytes = root
            .read_private_file(limit.path, limit.maximum_bytes)
            .map_err(|error| NativePackageAdmissionError::UnsafePackage(error.to_string()))?
            .ok_or_else(|| {
                NativePackageAdmissionError::UnsafePackage(format!(
                    "required payload is missing: {}",
                    limit.path
                ))
            })?;
        check_package_admission_cancellation(cancellation)?;
        if bytes.is_empty() {
            return Err(NativePackageAdmissionError::UnsafePackage(format!(
                "required payload is empty: {}",
                limit.path
            )));
        }
        total_bytes = total_bytes.checked_add(bytes.len()).ok_or_else(|| {
            NativePackageAdmissionError::UnsafePackage(
                "package byte accounting overflowed".to_owned(),
            )
        })?;
        if total_bytes > maximum_total_bytes {
            return Err(NativePackageAdmissionError::UnsafePackage(format!(
                "package exceeds the {maximum_total_bytes}-byte aggregate bound"
            )));
        }
        payloads.insert(limit.path.to_owned(), bytes);
    }

    after_capture()?;
    check_package_admission_cancellation(cancellation)?;
    if list_native_package_paths(root, maximum_entries, cancellation)? != observed_paths {
        return Err(NativePackageAdmissionError::UnsafePackage(
            "package tree changed while it was captured".to_owned(),
        ));
    }
    for limit in limits {
        check_package_admission_cancellation(cancellation)?;
        let current = root
            .read_private_file(limit.path, limit.maximum_bytes)
            .map_err(|error| NativePackageAdmissionError::UnsafePackage(error.to_string()))?
            .ok_or_else(|| {
                NativePackageAdmissionError::UnsafePackage(format!(
                    "required payload disappeared after capture: {}",
                    limit.path
                ))
            })?;
        if payloads.get(limit.path) != Some(&current) {
            return Err(NativePackageAdmissionError::UnsafePackage(format!(
                "payload changed after capture: {}",
                limit.path
            )));
        }
    }
    check_package_admission_cancellation(cancellation)?;
    Ok(payloads)
}

#[cfg(any(
    test,
    feature = "directml",
    feature = "metal",
    feature = "mlu",
    feature = "npu",
    feature = "cuda",
    feature = "xpu"
))]
pub(crate) fn validate_native_package_coverage(
    coverage: &[u8],
    payloads: &BTreeMap<String, Vec<u8>>,
    excludes: &[&str],
    maximum_bytes: usize,
) -> Result<(), NativePackageAdmissionError> {
    if excludes.iter().any(|path| !payloads.contains_key(*path)) {
        return Err(NativePackageAdmissionError::InvalidCoverage(
            "coverage exclusions are not a strict sorted payload subset".to_owned(),
        ));
    }
    let entries = parse_native_package_coverage(coverage, excludes, maximum_bytes)?;
    let expected_paths = payloads
        .keys()
        .filter(|path| excludes.binary_search(&path.as_str()).is_err())
        .collect::<BTreeSet<_>>();
    if entries.keys().collect::<BTreeSet<_>>() != expected_paths {
        return Err(NativePackageAdmissionError::InvalidCoverage(
            "coverage paths must exactly equal the non-excluded payload set".to_owned(),
        ));
    }
    for (path, entry) in entries {
        let payload = payloads.get(&path).ok_or_else(|| {
            NativePackageAdmissionError::InvalidCoverage(
                "coverage references an unknown payload".to_owned(),
            )
        })?;
        if entry.size != u64::try_from(payload.len()).unwrap_or(u64::MAX)
            || entry.digest_sha256 != format!("{:x}", Sha256::digest(payload))
        {
            return Err(NativePackageAdmissionError::InvalidCoverage(format!(
                "coverage does not match payload {path}"
            )));
        }
    }
    Ok(())
}

#[cfg(any(
    test,
    feature = "directml",
    feature = "metal",
    feature = "mlu",
    feature = "npu",
    feature = "cuda",
    feature = "xpu"
))]
fn list_native_package_paths(
    root: &ArtifactRoot,
    maximum_entries: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<String>, NativePackageAdmissionError> {
    root.list_contained_regular_files_recursive(maximum_entries, cancellation)
        .map_err(|error| {
            if matches!(error, sim_model::ArtifactIndexError::Cancelled) {
                NativePackageAdmissionError::Cancelled
            } else {
                NativePackageAdmissionError::UnsafePackage(error.to_string())
            }
        })?
        .into_iter()
        .map(|path| {
            path.components()
                .map(|component| {
                    component.as_os_str().to_str().ok_or_else(|| {
                        NativePackageAdmissionError::UnsafePackage(
                            "package path is not UTF-8".to_owned(),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|components| components.join("/"))
        })
        .collect()
}

#[cfg(any(
    test,
    feature = "directml",
    feature = "metal",
    feature = "mlu",
    feature = "npu",
    feature = "cuda",
    feature = "xpu"
))]
fn check_package_admission_cancellation(
    cancellation: &CancellationToken,
) -> Result<(), NativePackageAdmissionError> {
    cancellation
        .check()
        .map_err(|_| NativePackageAdmissionError::Cancelled)
}

fn valid_native_package_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.starts_with('\\')
        && !path.contains('\\')
        && path.split('/').all(|component| {
            !component.is_empty()
                && !matches!(component, "." | "..")
                && !component.chars().any(char::is_control)
        })
}

fn valid_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PluginTrustPolicy {
    keys: BTreeMap<String, PluginVerificationKey>,
}

impl PluginTrustPolicy {
    pub fn new(keys: impl IntoIterator<Item = PluginVerificationKey>) -> Result<Self, TrustError> {
        let mut checked_keys = BTreeMap::new();
        for key in keys {
            let key_id = key.key_id.clone();
            if checked_keys.insert(key_id.clone(), key).is_some() {
                return Err(TrustError::DuplicateVerificationKey(key_id));
            }
        }
        Ok(Self { keys: checked_keys })
    }

    pub fn authorize_manifest(
        &self,
        manifest: &PluginManifest,
        permissions: &PermissionPolicy,
    ) -> Result<PluginAuthorization, TrustError> {
        validate_plugin_manifest(manifest)?;
        let verification_key = self
            .keys
            .get(&manifest.signature.key_id)
            .ok_or_else(|| TrustError::UnknownVerificationKey(manifest.signature.key_id.clone()))?;
        let signature = decode_hex_exact::<ED25519_SIGNATURE_BYTES>(&manifest.signature.value)
            .ok_or(TrustError::InvalidPluginSignature)?;
        UnparsedPublicKey::new(&ED25519, verification_key.public_key)
            .verify(&manifest.signing_payload(), &signature)
            .map_err(|_| TrustError::InvalidPluginSignature)?;
        let requested_capabilities = manifest
            .capabilities
            .iter()
            .map(crate::Capability::from_plugin_request)
            .collect::<Result<Vec<_>, _>>()?;
        let capabilities = permissions.authorize(
            &manifest.identifier,
            &CapabilitySet::new(requested_capabilities),
        )?;
        Ok(PluginAuthorization {
            plugin_id: manifest.identifier.clone(),
            digest_sha256: manifest.digest_sha256.clone(),
            signing_payload_sha256: Sha256::digest(manifest.signing_payload()).into(),
            policy_generation: permissions.generation(),
            level: TrustLevel::Restricted,
            capabilities,
        })
    }

    pub fn authorize_provider_manifest_v2(
        &self,
        manifest: &ProviderPluginManifestV2,
        permissions: &PermissionPolicy,
    ) -> Result<ProviderManifestAuthorizationV2, TrustError> {
        let registry = TypeRegistry::built_in().map_err(PluginContractError::from)?;
        manifest.validate(&registry)?;
        let signing_payload = manifest.signing_payload()?;
        let verification_key = self
            .keys
            .get(&manifest.signature.key_id)
            .ok_or_else(|| TrustError::UnknownVerificationKey(manifest.signature.key_id.clone()))?;
        let signature = decode_hex_exact::<ED25519_SIGNATURE_BYTES>(&manifest.signature.value)
            .ok_or(TrustError::InvalidPluginSignature)?;
        UnparsedPublicKey::new(&ED25519, verification_key.public_key)
            .verify(&signing_payload, &signature)
            .map_err(|_| TrustError::InvalidPluginSignature)?;
        let authorization = self.authorize_manifest(&manifest.manifest, permissions)?;
        let provider_binding = manifest
            .manifest
            .provider_binding
            .clone()
            .ok_or(PluginContractError::InvalidProviderBindingSet)?;
        Ok(ProviderManifestAuthorizationV2 {
            authorization,
            outer_signing_payload_sha256: Sha256::digest(signing_payload).into(),
            provider_binding,
            streaming: manifest.streaming.clone(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderManifestAuthorizationV2 {
    authorization: PluginAuthorization,
    outer_signing_payload_sha256: [u8; 32],
    provider_binding: ProviderBindingSet,
    streaming: ProviderStreamingContractV2,
}

impl ProviderManifestAuthorizationV2 {
    #[cfg(test)]
    pub(crate) fn fixture(
        authorization: PluginAuthorization,
        outer_signing_payload_sha256: [u8; 32],
        provider_binding: ProviderBindingSet,
        streaming: ProviderStreamingContractV2,
    ) -> Self {
        Self {
            authorization,
            outer_signing_payload_sha256,
            provider_binding,
            streaming,
        }
    }

    pub fn authorization(&self) -> &PluginAuthorization {
        &self.authorization
    }

    pub fn outer_signing_payload_sha256(&self) -> &[u8; 32] {
        &self.outer_signing_payload_sha256
    }

    pub fn provider_binding(&self) -> &ProviderBindingSet {
        &self.provider_binding
    }

    pub fn streaming_contract(&self) -> &ProviderStreamingContractV2 {
        &self.streaming
    }
}

fn validate_plugin_manifest(manifest: &PluginManifest) -> Result<(), TrustError> {
    let registry = TypeRegistry::built_in().map_err(PluginContractError::from)?;
    manifest.validate(&registry)?;
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PluginAuthorization {
    plugin_id: String,
    digest_sha256: String,
    signing_payload_sha256: [u8; 32],
    policy_generation: PermissionPolicyGeneration,
    level: TrustLevel,
    capabilities: AuthorizedCapabilities,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SealedPluginAuthorization {
    payload: SealedPluginAuthorizationPayload,
    signature_algorithm: String,
    signature: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SealedPluginAuthorizationPayload {
    version: u16,
    plugin_id: String,
    digest_sha256: String,
    signing_payload_sha256: [u8; 32],
    policy_generation: PermissionPolicyGeneration,
    level: TrustLevel,
    capabilities: AuthorizedCapabilities,
}

pub struct PluginAuthorizationSealer {
    seed: Zeroizing<[u8; 32]>,
    policy_generation: PermissionPolicyGeneration,
}

impl PluginAuthorizationSealer {
    pub fn generate(policy_generation: PermissionPolicyGeneration) -> Result<Self, TrustError> {
        let mut seed = [0_u8; 32];
        SystemRandom::new()
            .fill(&mut seed)
            .map_err(|_| TrustError::AuthorizationSealingUnavailable)?;
        Self::from_seed(seed, policy_generation)
    }

    pub fn from_seed(
        seed: [u8; 32],
        policy_generation: PermissionPolicyGeneration,
    ) -> Result<Self, TrustError> {
        Ed25519KeyPair::from_seed_unchecked(&seed)
            .map_err(|_| TrustError::AuthorizationSealingUnavailable)?;
        Ok(Self {
            seed: Zeroizing::new(seed),
            policy_generation,
        })
    }

    pub fn verifier(&self) -> Result<PluginAuthorizationVerifier, TrustError> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(self.seed.as_ref())
            .map_err(|_| TrustError::AuthorizationSealingUnavailable)?;
        let public_key = key_pair
            .public_key()
            .as_ref()
            .try_into()
            .map_err(|_| TrustError::AuthorizationSealingUnavailable)?;
        Ok(PluginAuthorizationVerifier {
            public_key,
            policy_generation: self.policy_generation,
        })
    }

    fn sign(&self, payload: &[u8]) -> Result<[u8; ED25519_SIGNATURE_BYTES], TrustError> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(self.seed.as_ref())
            .map_err(|_| TrustError::AuthorizationSealingUnavailable)?;
        key_pair
            .sign(payload)
            .as_ref()
            .try_into()
            .map_err(|_| TrustError::AuthorizationSealingUnavailable)
    }
}

impl fmt::Debug for PluginAuthorizationSealer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PluginAuthorizationSealer([REDACTED])")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PluginAuthorizationVerifier {
    public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
    policy_generation: PermissionPolicyGeneration,
}

impl PluginAuthorizationVerifier {
    pub fn new(
        public_key: impl AsRef<[u8]>,
        policy_generation: PermissionPolicyGeneration,
    ) -> Result<Self, TrustError> {
        let public_key = public_key
            .as_ref()
            .try_into()
            .map_err(|_| TrustError::InvalidAuthorizationVerificationKey)?;
        Ok(Self {
            public_key,
            policy_generation,
        })
    }

    pub fn public_key_bytes(&self) -> &[u8; ED25519_PUBLIC_KEY_BYTES] {
        &self.public_key
    }

    pub fn from_hex(
        value: &str,
        policy_generation: PermissionPolicyGeneration,
    ) -> Result<Self, TrustError> {
        let public_key = decode_hex_exact::<ED25519_PUBLIC_KEY_BYTES>(value)
            .ok_or(TrustError::InvalidAuthorizationVerificationKey)?;
        Self::new(public_key, policy_generation)
    }

    pub fn to_hex(&self) -> String {
        encode_hex(&self.public_key)
    }

    pub const fn policy_generation(&self) -> PermissionPolicyGeneration {
        self.policy_generation
    }

    pub fn to_token(&self) -> String {
        format!("{}:{}", self.policy_generation.get(), self.to_hex())
    }

    pub fn from_token(value: &str) -> Result<Self, TrustError> {
        let (generation, public_key) = value
            .split_once(':')
            .ok_or(TrustError::InvalidAuthorizationVerificationKey)?;
        if generation.is_empty() || generation.starts_with('0') {
            return Err(TrustError::InvalidAuthorizationVerificationKey);
        }
        let generation = generation
            .parse::<u64>()
            .map_err(|_| TrustError::InvalidAuthorizationVerificationKey)?;
        let generation = PermissionPolicyGeneration::new(generation)
            .map_err(|_| TrustError::InvalidAuthorizationVerificationKey)?;
        Self::from_hex(public_key, generation)
    }
}

impl PluginAuthorization {
    #[cfg(test)]
    pub(crate) fn fixture_with_digest_sha256(&self, digest_sha256: impl Into<String>) -> Self {
        let mut fixture = self.clone();
        fixture.digest_sha256 = digest_sha256.into();
        fixture
    }

    pub fn plugin_id(&self) -> &str {
        &self.plugin_id
    }

    pub fn digest_sha256(&self) -> &str {
        &self.digest_sha256
    }

    pub fn level(&self) -> TrustLevel {
        self.level
    }

    pub fn capabilities(&self) -> &AuthorizedCapabilities {
        &self.capabilities
    }

    pub const fn policy_generation(&self) -> PermissionPolicyGeneration {
        self.policy_generation
    }

    pub fn sealed_bytes(&self, sealer: &PluginAuthorizationSealer) -> Result<Vec<u8>, TrustError> {
        if self.policy_generation != sealer.policy_generation
            || self.capabilities.policy_generation() != self.policy_generation
        {
            return Err(TrustError::InvalidSealedPluginAuthorization);
        }
        let payload = SealedPluginAuthorizationPayload {
            version: SEALED_PLUGIN_AUTHORIZATION_VERSION,
            plugin_id: self.plugin_id.clone(),
            digest_sha256: self.digest_sha256.clone(),
            signing_payload_sha256: self.signing_payload_sha256,
            policy_generation: self.policy_generation,
            level: self.level,
            capabilities: self.capabilities.clone(),
        };
        let payload_bytes = serde_json::to_vec(&payload)
            .map_err(|_| TrustError::InvalidSealedPluginAuthorization)?;
        let sealed = SealedPluginAuthorization {
            payload,
            signature_algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
            signature: encode_hex(&sealer.sign(&authorization_signing_payload(&payload_bytes))?),
        };
        let bytes = serde_json::to_vec(&sealed)
            .map_err(|_| TrustError::InvalidSealedPluginAuthorization)?;
        if bytes.len() > MAX_SEALED_PLUGIN_AUTHORIZATION_BYTES {
            return Err(TrustError::SealedPluginAuthorizationTooLarge);
        }
        Ok(bytes)
    }

    pub fn from_sealed_bytes(
        bytes: &[u8],
        manifest: &PluginManifest,
        verifier: &PluginAuthorizationVerifier,
        expected_policy_generation: PermissionPolicyGeneration,
        expected_profile_id: &str,
    ) -> Result<Self, TrustError> {
        if bytes.is_empty() || bytes.len() > MAX_SEALED_PLUGIN_AUTHORIZATION_BYTES {
            return Err(TrustError::SealedPluginAuthorizationTooLarge);
        }
        let sealed: SealedPluginAuthorization = serde_json::from_slice(bytes)
            .map_err(|_| TrustError::InvalidSealedPluginAuthorization)?;
        if sealed.signature_algorithm != PLUGIN_SIGNATURE_ALGORITHM {
            return Err(TrustError::InvalidSealedPluginAuthorization);
        }
        let signature = decode_hex_exact::<ED25519_SIGNATURE_BYTES>(&sealed.signature)
            .ok_or(TrustError::InvalidSealedPluginAuthorization)?;
        let payload_bytes = serde_json::to_vec(&sealed.payload)
            .map_err(|_| TrustError::InvalidSealedPluginAuthorization)?;
        UnparsedPublicKey::new(&ED25519, verifier.public_key)
            .verify(&authorization_signing_payload(&payload_bytes), &signature)
            .map_err(|_| TrustError::InvalidSealedPluginAuthorization)?;
        let sealed = sealed.payload;
        if sealed.version != SEALED_PLUGIN_AUTHORIZATION_VERSION
            || sealed.level != TrustLevel::Restricted
            || sealed.capabilities.subject_id() != sealed.plugin_id
            || sealed.capabilities.profile_id() != expected_profile_id
            || sealed.policy_generation != expected_policy_generation
            || sealed.policy_generation != verifier.policy_generation
            || sealed.capabilities.policy_generation() != sealed.policy_generation
        {
            return Err(TrustError::InvalidSealedPluginAuthorization);
        }
        sealed.capabilities.validate_sealed()?;
        let authorization = Self {
            plugin_id: sealed.plugin_id,
            digest_sha256: sealed.digest_sha256,
            signing_payload_sha256: sealed.signing_payload_sha256,
            policy_generation: sealed.policy_generation,
            level: sealed.level,
            capabilities: sealed.capabilities,
        };
        authorization.require_manifest(manifest)?;
        Ok(authorization)
    }

    pub fn require_manifest(&self, manifest: &PluginManifest) -> Result<(), TrustError> {
        validate_plugin_manifest(manifest)?;
        let signing_payload_sha256: [u8; 32] = Sha256::digest(manifest.signing_payload()).into();
        if manifest.identifier != self.plugin_id
            || manifest.digest_sha256 != self.digest_sha256
            || !constant_time_equal(&signing_payload_sha256, &self.signing_payload_sha256)
        {
            return Err(TrustError::AuthorizationManifestMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct SecretId(String);

impl SecretId {
    pub fn new(value: impl Into<String>) -> Result<Self, TrustError> {
        let value = value.into();
        if !valid_ascii_identifier(&value, 1_024) {
            return Err(TrustError::InvalidSecretId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProviderId(String);

impl ProviderId {
    pub fn new(value: impl Into<String>) -> Result<Self, TrustError> {
        let value = value.into();
        if !valid_ascii_identifier(&value, 256) {
            return Err(TrustError::InvalidProviderId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for SecretId {
    fn deserialize<Deserializer>(deserializer: Deserializer) -> Result<Self, Deserializer::Error>
    where
        Deserializer: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

pub struct SecretValue(Zeroizing<Vec<u8>>);

impl SecretValue {
    pub fn new(value: Vec<u8>) -> Self {
        Self(Zeroizing::new(value))
    }

    pub fn expose_to<R>(&self, operation: impl FnOnce(&[u8]) -> R) -> R {
        operation(self.0.as_slice())
    }
}

impl fmt::Debug for SecretValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SecretValue([REDACTED])")
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ProviderEndpoint {
    provider: ProviderId,
    endpoint: String,
}

impl ProviderEndpoint {
    pub fn new(
        provider: impl Into<String>,
        endpoint: impl Into<String>,
    ) -> Result<Self, TrustError> {
        let provider = ProviderId::new(provider)?;
        let endpoint = endpoint.into();
        if endpoint.len() > 2_048
            || endpoint != endpoint.trim()
            || endpoint.chars().any(char::is_control)
        {
            return Err(TrustError::InvalidProviderEndpoint);
        }
        let parsed = Url::parse(&endpoint).map_err(|_| TrustError::InvalidProviderEndpoint)?;
        let host = parsed.host().ok_or(TrustError::InvalidProviderEndpoint)?;
        let loopback_http = parsed.scheme() == "http"
            && match host {
                Host::Domain(domain) => domain.eq_ignore_ascii_case("localhost"),
                Host::Ipv4(address) => address.is_loopback(),
                Host::Ipv6(address) => address.is_loopback(),
            };
        if (parsed.scheme() != "https" && !loopback_http)
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.query().is_some()
            || parsed.fragment().is_some()
            || parsed.to_string() != endpoint
        {
            return Err(TrustError::InvalidProviderEndpoint);
        }
        Ok(Self { provider, endpoint })
    }

    pub fn provider(&self) -> &ProviderId {
        &self.provider
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CredentialScope {
    profile_id: String,
    subject_id: String,
    provider: ProviderId,
    secret_id: SecretId,
}

impl CredentialScope {
    pub fn new(
        profile_id: impl Into<String>,
        subject_id: impl Into<String>,
        provider: impl Into<String>,
        secret_id: SecretId,
    ) -> Result<Self, TrustError> {
        let profile_id = profile_id.into();
        let subject_id = subject_id.into();
        if !valid_ascii_identifier(&profile_id, 1_024)
            || !valid_ascii_identifier(&subject_id, 1_024)
        {
            return Err(TrustError::InvalidCredentialScope);
        }
        Ok(Self {
            profile_id,
            subject_id,
            provider: ProviderId::new(provider)?,
            secret_id,
        })
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }

    pub fn provider(&self) -> &ProviderId {
        &self.provider
    }

    pub fn secret_id(&self) -> &SecretId {
        &self.secret_id
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProviderMode {
    #[default]
    Disabled,
    Offline,
    Enabled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderPolicy {
    profile_id: Option<String>,
    mode: ProviderMode,
    allowed_endpoints: BTreeSet<ProviderEndpoint>,
    credential_scopes: BTreeSet<CredentialScope>,
}

impl Default for ProviderPolicy {
    fn default() -> Self {
        Self {
            profile_id: None,
            mode: ProviderMode::Disabled,
            allowed_endpoints: BTreeSet::new(),
            credential_scopes: BTreeSet::new(),
        }
    }
}

impl ProviderPolicy {
    pub fn new(
        profile_id: impl Into<String>,
        mode: ProviderMode,
        allowed_endpoints: impl IntoIterator<Item = ProviderEndpoint>,
        credential_scopes: impl IntoIterator<Item = CredentialScope>,
    ) -> Result<Self, TrustError> {
        let profile_id = profile_id.into();
        if !valid_ascii_identifier(&profile_id, 1_024) {
            return Err(TrustError::InvalidProviderProfile);
        }
        let mut allowed_endpoint_set = BTreeSet::new();
        for endpoint in allowed_endpoints {
            if !allowed_endpoint_set.insert(endpoint) {
                return Err(TrustError::InvalidProviderEndpoint);
            }
        }
        let mut credential_scope_set = BTreeSet::new();
        for scope in credential_scopes {
            if !credential_scope_set.insert(scope) {
                return Err(TrustError::InvalidCredentialScope);
            }
        }
        if credential_scope_set
            .iter()
            .any(|scope| scope.profile_id() != profile_id)
            || credential_scope_set.iter().any(|scope| {
                !allowed_endpoint_set
                    .iter()
                    .any(|endpoint| endpoint.provider() == scope.provider())
            })
        {
            return Err(TrustError::InvalidCredentialScope);
        }
        Ok(Self {
            profile_id: Some(profile_id),
            mode,
            allowed_endpoints: allowed_endpoint_set,
            credential_scopes: credential_scope_set,
        })
    }

    pub fn authorize(
        &self,
        profile_id: &str,
        subject_id: &str,
        provider: &str,
        endpoint: &str,
        secret_id: Option<&SecretId>,
    ) -> Result<AuthorizedProviderRequest, TrustError> {
        match self.mode {
            ProviderMode::Disabled => return Err(TrustError::ProviderDisabled),
            ProviderMode::Offline => return Err(TrustError::Offline),
            ProviderMode::Enabled => {}
        }
        if self.profile_id.as_deref() != Some(profile_id) {
            return Err(TrustError::ProviderProfileMismatch);
        }
        let requested_endpoint = ProviderEndpoint::new(provider, endpoint)?;
        if !self.allowed_endpoints.contains(&requested_endpoint) {
            return Err(TrustError::ProviderEndpointDenied);
        }
        let provider_id = ProviderId::new(provider)?;
        if secret_id.is_some_and(|secret_id| {
            CredentialScope::new(
                profile_id,
                subject_id,
                provider_id.as_str(),
                secret_id.clone(),
            )
            .map_or(true, |scope| !self.credential_scopes.contains(&scope))
        }) {
            return Err(TrustError::MissingSecretGrant);
        }
        Ok(AuthorizedProviderRequest {
            profile_id: profile_id.to_owned(),
            subject_id: subject_id.to_owned(),
            endpoint: requested_endpoint,
            secret_id: secret_id.cloned(),
            idempotency_key_sha256: None,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthorizedProviderRequest {
    profile_id: String,
    subject_id: String,
    endpoint: ProviderEndpoint,
    secret_id: Option<SecretId>,
    idempotency_key_sha256: Option<String>,
}

impl AuthorizedProviderRequest {
    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn provider(&self) -> &str {
        self.endpoint.provider.as_str()
    }

    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint.endpoint
    }

    pub fn secret_id(&self) -> Option<&SecretId> {
        self.secret_id.as_ref()
    }

    pub fn idempotency_key_sha256(&self) -> Option<&str> {
        self.idempotency_key_sha256.as_deref()
    }

    pub(crate) fn with_idempotency_key_sha256(
        mut self,
        idempotency_key_sha256: impl Into<String>,
    ) -> Result<Self, TrustError> {
        let idempotency_key_sha256 = idempotency_key_sha256.into();
        validate_sha256(&idempotency_key_sha256)
            .map_err(|()| TrustError::InvalidProviderInvocationIdentity)?;
        self.idempotency_key_sha256 = Some(idempotency_key_sha256);
        Ok(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderPriceBound {
    currency_code: String,
    maximum_microunits: u64,
}

impl ProviderPriceBound {
    pub fn new(
        currency_code: impl Into<String>,
        maximum_microunits: u64,
    ) -> Result<Self, TrustError> {
        let currency_code = currency_code.into();
        if currency_code.len() != 3
            || !currency_code.bytes().all(|byte| byte.is_ascii_uppercase())
            || maximum_microunits == 0
        {
            return Err(TrustError::InvalidProviderPriceBound);
        }
        Ok(Self {
            currency_code,
            maximum_microunits,
        })
    }

    pub fn currency_code(&self) -> &str {
        &self.currency_code
    }

    pub fn maximum_microunits(&self) -> u64 {
        self.maximum_microunits
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProviderCostNonce([u8; 32]);

impl ProviderCostNonce {
    pub fn new(bytes: [u8; 32]) -> Result<Self, TrustError> {
        if bytes.iter().all(|byte| *byte == 0) {
            return Err(TrustError::InvalidProviderCostAcceptance);
        }
        Ok(Self(bytes))
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderInvocationIdentity {
    principal_id: String,
    profile_id: String,
    prompt_id: String,
    prompt_sha256: String,
    attempt_id: String,
    node_id: String,
    request_ordinal: u32,
    request_sha256: String,
    plugin_id: String,
    plugin_digest_sha256: String,
    provider_binding_sha256: String,
    endpoint: ProviderEndpoint,
}

impl ProviderInvocationIdentity {
    pub fn idempotency_key_sha256(&self) -> String {
        let mut digest = Sha256::new();
        for field in [
            b"zed.sim.provider-idempotency.v1".as_slice(),
            self.profile_id.as_bytes(),
            self.prompt_sha256.as_bytes(),
            self.attempt_id.as_bytes(),
            self.node_id.as_bytes(),
            self.provider_binding_sha256.as_bytes(),
            self.request_sha256.as_bytes(),
        ] {
            digest.update((field.len() as u64).to_le_bytes());
            digest.update(field);
        }
        digest.update(self.request_ordinal.to_le_bytes());
        format!("{:x}", digest.finalize())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        principal_id: impl Into<String>,
        profile_id: impl Into<String>,
        prompt_id: impl Into<String>,
        prompt_sha256: impl Into<String>,
        attempt_id: impl Into<String>,
        node_id: impl Into<String>,
        request_ordinal: u32,
        request_sha256: impl Into<String>,
        plugin_id: impl Into<String>,
        plugin_digest_sha256: impl Into<String>,
        provider_binding_sha256: impl Into<String>,
        provider: impl Into<String>,
        endpoint: impl Into<String>,
    ) -> Result<Self, TrustError> {
        let principal_id = principal_id.into();
        let profile_id = profile_id.into();
        let prompt_id = prompt_id.into();
        let prompt_sha256 = prompt_sha256.into();
        let attempt_id = attempt_id.into();
        let node_id = node_id.into();
        let request_sha256 = request_sha256.into();
        let plugin_id = plugin_id.into();
        let plugin_digest_sha256 = plugin_digest_sha256.into();
        let provider_binding_sha256 = provider_binding_sha256.into();
        if principal_id.is_empty()
            || principal_id.len() > 1_024
            || principal_id != principal_id.trim()
            || principal_id.chars().any(char::is_control)
            || !valid_ascii_identifier(&profile_id, 1_024)
            || !valid_ascii_identifier(&prompt_id, 1_024)
            || validate_sha256(&prompt_sha256).is_err()
            || !valid_ascii_identifier(&attempt_id, 1_024)
            || !valid_ascii_node_identifier(&node_id, 1_024)
            || validate_sha256(&request_sha256).is_err()
            || !valid_ascii_identifier(&plugin_id, 1_024)
            || validate_sha256(&plugin_digest_sha256).is_err()
            || validate_sha256(&provider_binding_sha256).is_err()
        {
            return Err(TrustError::InvalidProviderInvocationIdentity);
        }
        Ok(Self {
            principal_id,
            profile_id,
            prompt_id,
            prompt_sha256,
            attempt_id,
            node_id,
            request_ordinal,
            request_sha256,
            plugin_id,
            plugin_digest_sha256,
            provider_binding_sha256,
            endpoint: ProviderEndpoint::new(provider, endpoint)?,
        })
    }

    pub fn principal_id(&self) -> &str {
        &self.principal_id
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn prompt_id(&self) -> &str {
        &self.prompt_id
    }

    pub fn prompt_sha256(&self) -> &str {
        &self.prompt_sha256
    }

    pub fn attempt_id(&self) -> &str {
        &self.attempt_id
    }

    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    pub fn request_ordinal(&self) -> u32 {
        self.request_ordinal
    }

    pub fn request_sha256(&self) -> &str {
        &self.request_sha256
    }

    pub fn plugin_id(&self) -> &str {
        &self.plugin_id
    }

    pub fn plugin_digest_sha256(&self) -> &str {
        &self.plugin_digest_sha256
    }

    pub fn provider_binding_sha256(&self) -> &str {
        &self.provider_binding_sha256
    }

    pub fn provider(&self) -> &str {
        self.endpoint.provider().as_str()
    }

    pub fn endpoint(&self) -> &str {
        self.endpoint.endpoint()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderCostAcceptanceScope {
    identity: ProviderInvocationIdentity,
    price_bound: ProviderPriceBound,
}

impl ProviderCostAcceptanceScope {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        principal_id: impl Into<String>,
        profile_id: impl Into<String>,
        prompt_id: impl Into<String>,
        prompt_sha256: impl Into<String>,
        attempt_id: impl Into<String>,
        node_id: impl Into<String>,
        request_ordinal: u32,
        request_sha256: impl Into<String>,
        plugin_id: impl Into<String>,
        plugin_digest_sha256: impl Into<String>,
        provider_binding_sha256: impl Into<String>,
        provider: impl Into<String>,
        endpoint: impl Into<String>,
        price_bound: ProviderPriceBound,
    ) -> Result<Self, TrustError> {
        let identity = ProviderInvocationIdentity::new(
            principal_id,
            profile_id,
            prompt_id,
            prompt_sha256,
            attempt_id,
            node_id,
            request_ordinal,
            request_sha256,
            plugin_id,
            plugin_digest_sha256,
            provider_binding_sha256,
            provider,
            endpoint,
        )
        .map_err(|error| match error {
            TrustError::InvalidProviderInvocationIdentity => {
                TrustError::InvalidProviderCostAcceptance
            }
            other => other,
        })?;
        Ok(Self {
            identity,
            price_bound,
        })
    }

    pub fn identity(&self) -> &ProviderInvocationIdentity {
        &self.identity
    }

    pub fn principal_id(&self) -> &str {
        self.identity.principal_id()
    }

    pub fn profile_id(&self) -> &str {
        self.identity.profile_id()
    }

    pub fn prompt_id(&self) -> &str {
        self.identity.prompt_id()
    }

    pub fn prompt_sha256(&self) -> &str {
        self.identity.prompt_sha256()
    }

    pub fn attempt_id(&self) -> &str {
        self.identity.attempt_id()
    }

    pub fn node_id(&self) -> &str {
        self.identity.node_id()
    }

    pub fn request_ordinal(&self) -> u32 {
        self.identity.request_ordinal()
    }

    pub fn request_sha256(&self) -> &str {
        self.identity.request_sha256()
    }

    pub fn plugin_id(&self) -> &str {
        self.identity.plugin_id()
    }

    pub fn plugin_digest_sha256(&self) -> &str {
        self.identity.plugin_digest_sha256()
    }

    pub fn provider_binding_sha256(&self) -> &str {
        self.identity.provider_binding_sha256()
    }

    pub fn provider(&self) -> &str {
        self.identity.provider()
    }

    pub fn endpoint(&self) -> &str {
        self.identity.endpoint()
    }

    pub fn price_bound(&self) -> &ProviderPriceBound {
        &self.price_bound
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProviderCostAcceptanceClaims {
    scope: ProviderCostAcceptanceScope,
    issued_at_milliseconds: u64,
    expires_at_milliseconds: u64,
    nonce: ProviderCostNonce,
}

pub struct ProviderCostAcceptance {
    claims: ProviderCostAcceptanceClaims,
    signature: [u8; ED25519_SIGNATURE_BYTES],
}

impl ProviderCostAcceptance {
    pub fn nonce(&self) -> ProviderCostNonce {
        self.claims.nonce
    }

    pub fn receipt_bytes(&self) -> Result<Vec<u8>, TrustError> {
        let identity = self.claims.scope.identity();
        let bytes = serde_json::to_vec(&ProviderCostAcceptanceReceiptWire {
            version: 1,
            principal_id: identity.principal_id().to_owned(),
            profile_id: identity.profile_id().to_owned(),
            prompt_id: identity.prompt_id().to_owned(),
            prompt_sha256: identity.prompt_sha256().to_owned(),
            attempt_id: identity.attempt_id().to_owned(),
            node_id: identity.node_id().to_owned(),
            request_ordinal: identity.request_ordinal(),
            request_sha256: identity.request_sha256().to_owned(),
            plugin_id: identity.plugin_id().to_owned(),
            plugin_digest_sha256: identity.plugin_digest_sha256().to_owned(),
            provider_binding_sha256: identity.provider_binding_sha256().to_owned(),
            provider: identity.provider().to_owned(),
            endpoint: identity.endpoint().to_owned(),
            currency_code: self.claims.scope.price_bound().currency_code().to_owned(),
            maximum_microunits: self.claims.scope.price_bound().maximum_microunits(),
            issued_at_milliseconds: self.claims.issued_at_milliseconds,
            expires_at_milliseconds: self.claims.expires_at_milliseconds,
            nonce: encode_hex(self.claims.nonce.as_bytes()),
            signature: encode_hex(&self.signature),
        })
        .map_err(|_| TrustError::InvalidProviderCostAcceptance)?;
        if bytes.is_empty() || bytes.len() > MAX_PROVIDER_RESULT_RECEIPT_BYTES {
            return Err(TrustError::InvalidProviderCostAcceptance);
        }
        Ok(bytes)
    }
}

#[derive(Serialize)]
struct ProviderCostAcceptanceReceiptWire {
    version: u16,
    principal_id: String,
    profile_id: String,
    prompt_id: String,
    prompt_sha256: String,
    attempt_id: String,
    node_id: String,
    request_ordinal: u32,
    request_sha256: String,
    plugin_id: String,
    plugin_digest_sha256: String,
    provider_binding_sha256: String,
    provider: String,
    endpoint: String,
    currency_code: String,
    maximum_microunits: u64,
    issued_at_milliseconds: u64,
    expires_at_milliseconds: u64,
    nonce: String,
    signature: String,
}

impl fmt::Debug for ProviderCostAcceptance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ProviderCostAcceptance([SEALED])")
    }
}

pub struct ProviderCostAcceptanceIssuer {
    seed: Zeroizing<[u8; 32]>,
    clock_origin: Instant,
}

impl ProviderCostAcceptanceIssuer {
    pub fn generate(clock_origin: Instant) -> Result<Self, TrustError> {
        let mut seed = [0_u8; 32];
        SystemRandom::new()
            .fill(&mut seed)
            .map_err(|_| TrustError::ProviderCostAcceptanceSealingUnavailable)?;
        Self::from_seed(seed, clock_origin)
    }

    pub fn from_seed(seed: [u8; 32], clock_origin: Instant) -> Result<Self, TrustError> {
        Ed25519KeyPair::from_seed_unchecked(&seed)
            .map_err(|_| TrustError::ProviderCostAcceptanceSealingUnavailable)?;
        Ok(Self {
            seed: Zeroizing::new(seed),
            clock_origin,
        })
    }

    pub fn verifier(&self) -> Result<ProviderCostAcceptanceVerifier, TrustError> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(self.seed.as_ref())
            .map_err(|_| TrustError::ProviderCostAcceptanceSealingUnavailable)?;
        let public_key = key_pair
            .public_key()
            .as_ref()
            .try_into()
            .map_err(|_| TrustError::ProviderCostAcceptanceSealingUnavailable)?;
        Ok(ProviderCostAcceptanceVerifier {
            public_key,
            clock_origin: self.clock_origin,
        })
    }

    pub fn issue(
        &self,
        scope: ProviderCostAcceptanceScope,
        issued_at: Instant,
        expires_at: Instant,
        nonce: ProviderCostNonce,
    ) -> Result<ProviderCostAcceptance, TrustError> {
        let issued_at_milliseconds = provider_cost_milliseconds(self.clock_origin, issued_at)?;
        let expires_at_milliseconds = provider_cost_milliseconds(self.clock_origin, expires_at)?;
        if expires_at <= issued_at
            || expires_at
                .checked_duration_since(issued_at)
                .is_none_or(|duration| duration > MAX_PROVIDER_COST_ACCEPTANCE_LIFETIME)
        {
            return Err(TrustError::InvalidProviderCostAcceptance);
        }
        let claims = ProviderCostAcceptanceClaims {
            scope,
            issued_at_milliseconds,
            expires_at_milliseconds,
            nonce,
        };
        let key_pair = Ed25519KeyPair::from_seed_unchecked(self.seed.as_ref())
            .map_err(|_| TrustError::ProviderCostAcceptanceSealingUnavailable)?;
        let signature = key_pair
            .sign(&provider_cost_acceptance_signing_payload(&claims))
            .as_ref()
            .try_into()
            .map_err(|_| TrustError::ProviderCostAcceptanceSealingUnavailable)?;
        Ok(ProviderCostAcceptance { claims, signature })
    }
}

impl fmt::Debug for ProviderCostAcceptanceIssuer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ProviderCostAcceptanceIssuer([REDACTED])")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifiedProviderCostAcceptance {
    nonce: ProviderCostNonce,
    expires_at: Instant,
}

impl VerifiedProviderCostAcceptance {
    pub fn nonce(self) -> ProviderCostNonce {
        self.nonce
    }

    pub fn expires_at(self) -> Instant {
        self.expires_at
    }
}

#[derive(Clone, Debug)]
pub struct ProviderCostAcceptanceVerifier {
    public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
    clock_origin: Instant,
}

impl ProviderCostAcceptanceVerifier {
    pub fn verify(
        &self,
        acceptance: &ProviderCostAcceptance,
        expected_scope: &ProviderCostAcceptanceScope,
        now: Instant,
    ) -> Result<VerifiedProviderCostAcceptance, TrustError> {
        UnparsedPublicKey::new(&ED25519, self.public_key)
            .verify(
                &provider_cost_acceptance_signing_payload(&acceptance.claims),
                &acceptance.signature,
            )
            .map_err(|_| TrustError::InvalidProviderCostAcceptance)?;
        if acceptance.claims.scope != *expected_scope {
            return Err(TrustError::InvalidProviderCostAcceptance);
        }
        let now_milliseconds = provider_cost_milliseconds(self.clock_origin, now)?;
        if now_milliseconds < acceptance.claims.issued_at_milliseconds
            || now_milliseconds >= acceptance.claims.expires_at_milliseconds
        {
            return Err(TrustError::ExpiredProviderCostAcceptance);
        }
        let expires_at = self
            .clock_origin
            .checked_add(Duration::from_millis(
                acceptance.claims.expires_at_milliseconds,
            ))
            .ok_or(TrustError::InvalidProviderCostAcceptance)?;
        Ok(VerifiedProviderCostAcceptance {
            nonce: acceptance.claims.nonce,
            expires_at,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProviderResultNonce([u8; 32]);

impl ProviderResultNonce {
    pub fn generate() -> Result<Self, TrustError> {
        let mut bytes = [0_u8; 32];
        SystemRandom::new()
            .fill(&mut bytes)
            .map_err(|_| TrustError::ProviderResultReceiptSealingUnavailable)?;
        Self::new(bytes)
    }

    pub fn new(bytes: [u8; 32]) -> Result<Self, TrustError> {
        if bytes.iter().all(|byte| *byte == 0) {
            return Err(TrustError::InvalidProviderResultReceipt);
        }
        Ok(Self(bytes))
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProviderResultReceiptClaims {
    identity: ProviderInvocationIdentity,
    result_sha256: String,
    issued_at_milliseconds: u64,
    expires_at_milliseconds: u64,
    nonce: ProviderResultNonce,
}

pub struct ProviderResultReceipt {
    claims: ProviderResultReceiptClaims,
    signature: [u8; ED25519_SIGNATURE_BYTES],
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProviderResultReceiptWire {
    principal_id: String,
    profile_id: String,
    prompt_id: String,
    prompt_sha256: String,
    attempt_id: String,
    node_id: String,
    request_ordinal: u32,
    request_sha256: String,
    plugin_id: String,
    plugin_digest_sha256: String,
    provider_binding_sha256: String,
    provider: String,
    endpoint: String,
    result_sha256: String,
    issued_at_milliseconds: u64,
    expires_at_milliseconds: u64,
    nonce: String,
    signature: String,
}

impl ProviderResultReceipt {
    pub fn identity(&self) -> &ProviderInvocationIdentity {
        &self.claims.identity
    }

    pub fn result_sha256(&self) -> &str {
        &self.claims.result_sha256
    }

    pub fn nonce(&self) -> ProviderResultNonce {
        self.claims.nonce
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, TrustError> {
        let identity = &self.claims.identity;
        let wire = ProviderResultReceiptWire {
            principal_id: identity.principal_id().to_owned(),
            profile_id: identity.profile_id().to_owned(),
            prompt_id: identity.prompt_id().to_owned(),
            prompt_sha256: identity.prompt_sha256().to_owned(),
            attempt_id: identity.attempt_id().to_owned(),
            node_id: identity.node_id().to_owned(),
            request_ordinal: identity.request_ordinal(),
            request_sha256: identity.request_sha256().to_owned(),
            plugin_id: identity.plugin_id().to_owned(),
            plugin_digest_sha256: identity.plugin_digest_sha256().to_owned(),
            provider_binding_sha256: identity.provider_binding_sha256().to_owned(),
            provider: identity.provider().to_owned(),
            endpoint: identity.endpoint().to_owned(),
            result_sha256: self.claims.result_sha256.clone(),
            issued_at_milliseconds: self.claims.issued_at_milliseconds,
            expires_at_milliseconds: self.claims.expires_at_milliseconds,
            nonce: encode_hex(self.claims.nonce.as_bytes()),
            signature: encode_hex(&self.signature),
        };
        let bytes =
            serde_json::to_vec(&wire).map_err(|_| TrustError::InvalidProviderResultReceipt)?;
        if bytes.len() > MAX_PROVIDER_RESULT_RECEIPT_BYTES {
            return Err(TrustError::ProviderResultReceiptTooLarge);
        }
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrustError> {
        if bytes.is_empty() || bytes.len() > MAX_PROVIDER_RESULT_RECEIPT_BYTES {
            return Err(TrustError::ProviderResultReceiptTooLarge);
        }
        let wire: ProviderResultReceiptWire =
            serde_json::from_slice(bytes).map_err(|_| TrustError::InvalidProviderResultReceipt)?;
        let identity = ProviderInvocationIdentity::new(
            wire.principal_id,
            wire.profile_id,
            wire.prompt_id,
            wire.prompt_sha256,
            wire.attempt_id,
            wire.node_id,
            wire.request_ordinal,
            wire.request_sha256,
            wire.plugin_id,
            wire.plugin_digest_sha256,
            wire.provider_binding_sha256,
            wire.provider,
            wire.endpoint,
        )
        .map_err(|_| TrustError::InvalidProviderResultReceipt)?;
        validate_sha256(&wire.result_sha256)
            .map_err(|_| TrustError::InvalidProviderResultReceipt)?;
        let nonce = decode_hex_exact::<32>(&wire.nonce)
            .and_then(|bytes| ProviderResultNonce::new(bytes).ok())
            .ok_or(TrustError::InvalidProviderResultReceipt)?;
        let signature = decode_hex_exact::<ED25519_SIGNATURE_BYTES>(&wire.signature)
            .ok_or(TrustError::InvalidProviderResultReceipt)?;
        Ok(Self {
            claims: ProviderResultReceiptClaims {
                identity,
                result_sha256: wire.result_sha256,
                issued_at_milliseconds: wire.issued_at_milliseconds,
                expires_at_milliseconds: wire.expires_at_milliseconds,
                nonce,
            },
            signature,
        })
    }
}

impl fmt::Debug for ProviderResultReceipt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ProviderResultReceipt([SEALED])")
    }
}

pub struct ProviderResultReceiptIssuer {
    seed: Zeroizing<[u8; 32]>,
    clock_origin: Instant,
}

impl ProviderResultReceiptIssuer {
    pub fn generate(clock_origin: Instant) -> Result<Self, TrustError> {
        let mut seed = [0_u8; 32];
        SystemRandom::new()
            .fill(&mut seed)
            .map_err(|_| TrustError::ProviderResultReceiptSealingUnavailable)?;
        Self::from_seed(seed, clock_origin)
    }

    pub fn from_seed(seed: [u8; 32], clock_origin: Instant) -> Result<Self, TrustError> {
        Ed25519KeyPair::from_seed_unchecked(&seed)
            .map_err(|_| TrustError::ProviderResultReceiptSealingUnavailable)?;
        Ok(Self {
            seed: Zeroizing::new(seed),
            clock_origin,
        })
    }

    pub fn verifier(&self) -> Result<ProviderResultReceiptVerifier, TrustError> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(self.seed.as_ref())
            .map_err(|_| TrustError::ProviderResultReceiptSealingUnavailable)?;
        let public_key = key_pair
            .public_key()
            .as_ref()
            .try_into()
            .map_err(|_| TrustError::ProviderResultReceiptSealingUnavailable)?;
        Ok(ProviderResultReceiptVerifier {
            public_key,
            clock_origin: self.clock_origin,
        })
    }

    pub fn issue(
        &self,
        identity: ProviderInvocationIdentity,
        result_sha256: impl Into<String>,
        issued_at: Instant,
        expires_at: Instant,
        nonce: ProviderResultNonce,
    ) -> Result<ProviderResultReceipt, TrustError> {
        let result_sha256 = result_sha256.into();
        validate_sha256(&result_sha256).map_err(|_| TrustError::InvalidProviderResultReceipt)?;
        let issued_at_milliseconds = provider_result_milliseconds(self.clock_origin, issued_at)?;
        let expires_at_milliseconds = provider_result_milliseconds(self.clock_origin, expires_at)?;
        if expires_at <= issued_at
            || expires_at
                .checked_duration_since(issued_at)
                .is_none_or(|duration| duration > MAX_PROVIDER_RESULT_RECEIPT_LIFETIME)
        {
            return Err(TrustError::InvalidProviderResultReceipt);
        }
        let claims = ProviderResultReceiptClaims {
            identity,
            result_sha256,
            issued_at_milliseconds,
            expires_at_milliseconds,
            nonce,
        };
        let key_pair = Ed25519KeyPair::from_seed_unchecked(self.seed.as_ref())
            .map_err(|_| TrustError::ProviderResultReceiptSealingUnavailable)?;
        let signature = key_pair
            .sign(&provider_result_receipt_signing_payload(&claims))
            .as_ref()
            .try_into()
            .map_err(|_| TrustError::ProviderResultReceiptSealingUnavailable)?;
        Ok(ProviderResultReceipt { claims, signature })
    }
}

impl fmt::Debug for ProviderResultReceiptIssuer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ProviderResultReceiptIssuer([REDACTED])")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedProviderResultReceipt {
    result_sha256: String,
    nonce: ProviderResultNonce,
    expires_at: Instant,
}

impl VerifiedProviderResultReceipt {
    pub fn result_sha256(&self) -> &str {
        &self.result_sha256
    }

    pub fn nonce(&self) -> ProviderResultNonce {
        self.nonce
    }

    pub fn expires_at(&self) -> Instant {
        self.expires_at
    }
}

#[derive(Clone, Debug)]
pub struct ProviderResultReceiptVerifier {
    public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
    clock_origin: Instant,
}

impl ProviderResultReceiptVerifier {
    pub fn verify(
        &self,
        receipt: &ProviderResultReceipt,
        expected_identity: &ProviderInvocationIdentity,
        expected_result_sha256: &str,
        now: Instant,
    ) -> Result<VerifiedProviderResultReceipt, TrustError> {
        validate_sha256(expected_result_sha256)
            .map_err(|_| TrustError::InvalidProviderResultReceipt)?;
        UnparsedPublicKey::new(&ED25519, self.public_key)
            .verify(
                &provider_result_receipt_signing_payload(&receipt.claims),
                &receipt.signature,
            )
            .map_err(|_| TrustError::InvalidProviderResultReceipt)?;
        if receipt.claims.identity != *expected_identity
            || receipt.claims.result_sha256 != expected_result_sha256
        {
            return Err(TrustError::InvalidProviderResultReceipt);
        }
        let now_milliseconds = provider_result_milliseconds(self.clock_origin, now)?;
        if now_milliseconds < receipt.claims.issued_at_milliseconds
            || now_milliseconds >= receipt.claims.expires_at_milliseconds
        {
            return Err(TrustError::ExpiredProviderResultReceipt);
        }
        let expires_at = self
            .clock_origin
            .checked_add(Duration::from_millis(
                receipt.claims.expires_at_milliseconds,
            ))
            .ok_or(TrustError::InvalidProviderResultReceipt)?;
        Ok(VerifiedProviderResultReceipt {
            result_sha256: receipt.claims.result_sha256.clone(),
            nonce: receipt.claims.nonce,
            expires_at,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderRuntimeReceiptIdentityV2 {
    pub provider: String,
    pub method: ProviderHttpMethodV2,
    pub endpoint: String,
    pub ordered_headers_sha256: String,
    pub secret_id: Option<String>,
    pub request_head_sha256: String,
    pub request_body_sha256: String,
    pub provider_manifest_sha256: String,
    pub component_generation: u64,
    pub component_digest_sha256: String,
    pub binding_generation: u64,
    pub binding_set_sha256: String,
    pub accepted_cost_microunits: u64,
    pub request_ordinal: u32,
    pub response_status: u16,
    pub response_headers_sha256: String,
    pub ordered_uploads_sha256: String,
    pub ordered_chunks_sha256: String,
    pub terminal_receipt_sha256: String,
    pub idempotency_identity_sha256: String,
}

impl ProviderRuntimeReceiptIdentityV2 {
    pub fn validate(&self) -> Result<(), TrustError> {
        if !valid_ascii_identifier(&self.provider, 256)
            || ProviderEndpoint::new(&self.provider, &self.endpoint).is_err()
            || self
                .secret_id
                .as_deref()
                .is_some_and(|value| SecretId::new(value).is_err())
            || self.component_generation == 0
            || self.binding_generation == 0
            || !(200..=599).contains(&self.response_status)
            || [
                self.ordered_headers_sha256.as_str(),
                self.request_head_sha256.as_str(),
                self.request_body_sha256.as_str(),
                self.provider_manifest_sha256.as_str(),
                self.component_digest_sha256.as_str(),
                self.binding_set_sha256.as_str(),
                self.response_headers_sha256.as_str(),
                self.ordered_uploads_sha256.as_str(),
                self.ordered_chunks_sha256.as_str(),
                self.terminal_receipt_sha256.as_str(),
                self.idempotency_identity_sha256.as_str(),
            ]
            .into_iter()
            .any(|value| validate_sha256(value).is_err())
        {
            return Err(TrustError::InvalidProviderRuntimeReceipt);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProviderRuntimeReceiptClaimsV2 {
    version: u16,
    identity: ProviderRuntimeReceiptIdentityV2,
    issued_at_milliseconds: u64,
    expires_at_milliseconds: u64,
    nonce: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProviderRuntimeReceiptWireV2 {
    claims: ProviderRuntimeReceiptClaimsV2,
    signature: String,
}

pub struct ProviderRuntimeReceiptV2 {
    claims: ProviderRuntimeReceiptClaimsV2,
    signature: [u8; ED25519_SIGNATURE_BYTES],
}

impl ProviderRuntimeReceiptV2 {
    pub fn identity(&self) -> &ProviderRuntimeReceiptIdentityV2 {
        &self.claims.identity
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, TrustError> {
        let bytes = serde_json::to_vec(&ProviderRuntimeReceiptWireV2 {
            claims: self.claims.clone(),
            signature: encode_hex(&self.signature),
        })
        .map_err(|_| TrustError::InvalidProviderRuntimeReceipt)?;
        if bytes.is_empty() || bytes.len() > MAX_PROVIDER_RESULT_RECEIPT_BYTES {
            return Err(TrustError::ProviderResultReceiptTooLarge);
        }
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrustError> {
        if bytes.is_empty() || bytes.len() > MAX_PROVIDER_RESULT_RECEIPT_BYTES {
            return Err(TrustError::ProviderResultReceiptTooLarge);
        }
        let wire: ProviderRuntimeReceiptWireV2 =
            serde_json::from_slice(bytes).map_err(|_| TrustError::InvalidProviderRuntimeReceipt)?;
        if wire.claims.version != PROVIDER_RUNTIME_RECEIPT_VERSION_V2
            || wire.claims.nonce.len() != 64
        {
            return Err(TrustError::InvalidProviderRuntimeReceipt);
        }
        wire.claims.identity.validate()?;
        decode_hex_exact::<32>(&wire.claims.nonce)
            .filter(|nonce| nonce.iter().any(|byte| *byte != 0))
            .ok_or(TrustError::InvalidProviderRuntimeReceipt)?;
        let signature = decode_hex_exact::<ED25519_SIGNATURE_BYTES>(&wire.signature)
            .ok_or(TrustError::InvalidProviderRuntimeReceipt)?;
        Ok(Self {
            claims: wire.claims,
            signature,
        })
    }
}

impl fmt::Debug for ProviderRuntimeReceiptV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ProviderRuntimeReceiptV2([SEALED])")
    }
}

pub struct ProviderRuntimeReceiptIssuerV2 {
    seed: Zeroizing<[u8; 32]>,
    clock_origin: Instant,
}

impl ProviderRuntimeReceiptIssuerV2 {
    pub fn from_seed(seed: [u8; 32], clock_origin: Instant) -> Result<Self, TrustError> {
        Ed25519KeyPair::from_seed_unchecked(&seed)
            .map_err(|_| TrustError::ProviderResultReceiptSealingUnavailable)?;
        Ok(Self {
            seed: Zeroizing::new(seed),
            clock_origin,
        })
    }

    pub fn verifier(&self) -> Result<ProviderRuntimeReceiptVerifierV2, TrustError> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(self.seed.as_ref())
            .map_err(|_| TrustError::ProviderResultReceiptSealingUnavailable)?;
        Ok(ProviderRuntimeReceiptVerifierV2 {
            public_key: key_pair
                .public_key()
                .as_ref()
                .try_into()
                .map_err(|_| TrustError::ProviderResultReceiptSealingUnavailable)?,
            clock_origin: self.clock_origin,
        })
    }

    pub fn issue(
        &self,
        identity: ProviderRuntimeReceiptIdentityV2,
        issued_at: Instant,
        expires_at: Instant,
        nonce: [u8; 32],
    ) -> Result<ProviderRuntimeReceiptV2, TrustError> {
        identity.validate()?;
        if nonce.iter().all(|byte| *byte == 0)
            || expires_at <= issued_at
            || expires_at
                .checked_duration_since(issued_at)
                .is_none_or(|duration| duration > MAX_PROVIDER_RESULT_RECEIPT_LIFETIME)
        {
            return Err(TrustError::InvalidProviderRuntimeReceipt);
        }
        let claims = ProviderRuntimeReceiptClaimsV2 {
            version: PROVIDER_RUNTIME_RECEIPT_VERSION_V2,
            identity,
            issued_at_milliseconds: provider_result_milliseconds(self.clock_origin, issued_at)?,
            expires_at_milliseconds: provider_result_milliseconds(self.clock_origin, expires_at)?,
            nonce: encode_hex(&nonce),
        };
        let key_pair = Ed25519KeyPair::from_seed_unchecked(self.seed.as_ref())
            .map_err(|_| TrustError::ProviderResultReceiptSealingUnavailable)?;
        let signature = key_pair
            .sign(&provider_runtime_receipt_signing_payload_v2(&claims)?)
            .as_ref()
            .try_into()
            .map_err(|_| TrustError::ProviderResultReceiptSealingUnavailable)?;
        Ok(ProviderRuntimeReceiptV2 { claims, signature })
    }
}

pub struct ProviderRuntimeReceiptVerifierV2 {
    public_key: [u8; ED25519_PUBLIC_KEY_BYTES],
    clock_origin: Instant,
}

pub struct VerifiedProviderRuntimeReceiptV2 {
    identity: ProviderRuntimeReceiptIdentityV2,
    bytes: Vec<u8>,
}

impl VerifiedProviderRuntimeReceiptV2 {
    pub fn identity(&self) -> &ProviderRuntimeReceiptIdentityV2 {
        &self.identity
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl fmt::Debug for VerifiedProviderRuntimeReceiptV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("VerifiedProviderRuntimeReceiptV2([SEALED])")
    }
}

impl ProviderRuntimeReceiptVerifierV2 {
    pub fn verify(
        &self,
        receipt: &ProviderRuntimeReceiptV2,
        expected: &ProviderRuntimeReceiptIdentityV2,
        now: Instant,
    ) -> Result<VerifiedProviderRuntimeReceiptV2, TrustError> {
        expected.validate()?;
        UnparsedPublicKey::new(&ED25519, self.public_key)
            .verify(
                &provider_runtime_receipt_signing_payload_v2(&receipt.claims)?,
                &receipt.signature,
            )
            .map_err(|_| TrustError::InvalidProviderRuntimeReceipt)?;
        let now = provider_result_milliseconds(self.clock_origin, now)?;
        if receipt.claims.identity != *expected
            || now < receipt.claims.issued_at_milliseconds
            || now >= receipt.claims.expires_at_milliseconds
        {
            return Err(TrustError::InvalidProviderRuntimeReceipt);
        }
        Ok(VerifiedProviderRuntimeReceiptV2 {
            identity: receipt.claims.identity.clone(),
            bytes: receipt.to_bytes()?,
        })
    }
}

fn provider_runtime_receipt_signing_payload_v2(
    claims: &ProviderRuntimeReceiptClaimsV2,
) -> Result<[u8; 32], TrustError> {
    let claims =
        serde_json::to_vec(claims).map_err(|_| TrustError::InvalidProviderRuntimeReceipt)?;
    let mut digest = Sha256::new();
    digest.update(PROVIDER_RUNTIME_RECEIPT_SIGNATURE_DOMAIN_V2);
    digest.update((claims.len() as u64).to_le_bytes());
    digest.update(claims);
    Ok(digest.finalize().into())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RemoteExposureApproval {
    LoopbackOnly,
    Approved,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeApiExposure {
    bind_address: IpAddr,
    tls_enabled: bool,
    allowed_origins: BTreeSet<String>,
    reverse_proxy_trusted: bool,
    approval: RemoteExposureApproval,
}

impl NativeApiExposure {
    pub fn new(
        bind_address: IpAddr,
        tls_enabled: bool,
        allowed_origins: impl IntoIterator<Item = String>,
        reverse_proxy_trusted: bool,
        approval: RemoteExposureApproval,
    ) -> Result<Self, TrustError> {
        let allowed_origins = allowed_origins
            .into_iter()
            .map(|origin| canonical_origin(&origin))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let exposure = Self {
            bind_address,
            tls_enabled,
            allowed_origins,
            reverse_proxy_trusted,
            approval,
        };
        exposure.validate()?;
        Ok(exposure)
    }

    pub fn bind_address(&self) -> IpAddr {
        self.bind_address
    }

    pub fn tls_enabled(&self) -> bool {
        self.tls_enabled
    }

    pub fn allowed_origins(&self) -> &BTreeSet<String> {
        &self.allowed_origins
    }

    pub fn reverse_proxy_trusted(&self) -> bool {
        self.reverse_proxy_trusted
    }

    fn validate(&self) -> Result<(), TrustError> {
        if self.bind_address.is_loopback() {
            return Ok(());
        }
        if self.approval != RemoteExposureApproval::Approved
            || !self.tls_enabled
            || self.allowed_origins.is_empty()
        {
            return Err(TrustError::UnsafeApiExposure);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalNavigationPolicy {
    allowed_schemes: BTreeSet<String>,
    require_user_gesture: bool,
}

impl ExternalNavigationPolicy {
    pub fn https_user_gesture() -> Self {
        Self {
            allowed_schemes: BTreeSet::from(["https".to_owned()]),
            require_user_gesture: true,
        }
    }

    pub fn new(
        allowed_schemes: impl IntoIterator<Item = String>,
        require_user_gesture: bool,
    ) -> Result<Self, TrustError> {
        let allowed_schemes = allowed_schemes
            .into_iter()
            .map(|scheme| scheme.to_ascii_lowercase())
            .collect::<BTreeSet<_>>();
        if allowed_schemes.is_empty()
            || allowed_schemes
                .iter()
                .any(|scheme| !valid_navigation_scheme(scheme))
        {
            return Err(TrustError::InvalidNavigationPolicy);
        }
        Ok(Self {
            allowed_schemes,
            require_user_gesture,
        })
    }

    pub fn authorize(&self, url: &str, user_gesture: bool) -> Result<(), TrustError> {
        if url.len() > 16 * 1024 || url.bytes().any(|byte| byte.is_ascii_control()) {
            return Err(TrustError::NavigationDenied);
        }
        let parsed = Url::parse(url).map_err(|_| TrustError::NavigationDenied)?;
        if !self.allowed_schemes.contains(parsed.scheme())
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || matches!(parsed.scheme(), "http" | "https") && !parsed.has_host()
            || (self.require_user_gesture && !user_gesture)
        {
            return Err(TrustError::NavigationDenied);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeFfiContract {
    library_id: String,
    digest_sha256: String,
    abi_version: String,
    required_symbols: BTreeSet<String>,
    required_by: Option<String>,
    unsafe_owner: String,
}

impl NativeFfiContract {
    pub fn new(
        library_id: impl Into<String>,
        digest_sha256: impl Into<String>,
        abi_version: impl Into<String>,
        required_symbols: impl IntoIterator<Item = String>,
        unsafe_owner: impl Into<String>,
    ) -> Result<Self, TrustError> {
        let contract = Self {
            library_id: library_id.into(),
            digest_sha256: digest_sha256.into(),
            abi_version: abi_version.into(),
            required_symbols: required_symbols.into_iter().collect(),
            required_by: None,
            unsafe_owner: unsafe_owner.into(),
        };
        contract.validate()?;
        Ok(contract)
    }

    pub fn new_dependency(
        library_id: impl Into<String>,
        digest_sha256: impl Into<String>,
        abi_version: impl Into<String>,
        required_by: impl Into<String>,
        unsafe_owner: impl Into<String>,
    ) -> Result<Self, TrustError> {
        let contract = Self {
            library_id: library_id.into(),
            digest_sha256: digest_sha256.into(),
            abi_version: abi_version.into(),
            required_symbols: BTreeSet::new(),
            required_by: Some(required_by.into()),
            unsafe_owner: unsafe_owner.into(),
        };
        contract.validate()?;
        Ok(contract)
    }

    fn validate(&self) -> Result<(), TrustError> {
        if self.library_id.trim().is_empty()
            || validate_sha256(&self.digest_sha256).is_err()
            || self.abi_version.trim().is_empty()
            || (self.required_symbols.is_empty() == self.required_by.is_none())
            || self.required_by.as_ref().is_some_and(|required_by| {
                required_by.trim().is_empty()
                    || required_by == &self.library_id
                    || !required_by
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.'))
            })
            || self.required_symbols.iter().any(|symbol| {
                symbol.is_empty()
                    || !symbol
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.'))
            })
            || self.unsafe_owner.trim().is_empty()
        {
            return Err(TrustError::InvalidFfiContract);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NativeFfiRegistry {
    contracts: BTreeMap<String, NativeFfiContract>,
}

impl NativeFfiRegistry {
    pub fn new(contracts: impl IntoIterator<Item = NativeFfiContract>) -> Result<Self, TrustError> {
        let mut checked_contracts = BTreeMap::new();
        for contract in contracts {
            contract.validate()?;
            let library_id = contract.library_id.clone();
            if checked_contracts
                .insert(library_id.clone(), contract)
                .is_some()
            {
                return Err(TrustError::DuplicateFfiContract(library_id));
            }
        }
        for contract in checked_contracts.values() {
            let mut current = contract;
            let mut visited = BTreeSet::new();
            while let Some(required_by) = current.required_by.as_deref() {
                if !visited.insert(current.library_id.as_str()) {
                    return Err(TrustError::InvalidFfiContract);
                }
                current = checked_contracts
                    .get(required_by)
                    .ok_or(TrustError::InvalidFfiContract)?;
            }
        }
        Ok(Self {
            contracts: checked_contracts,
        })
    }

    pub fn authorize(
        &self,
        library_id: &str,
        digest_sha256: &str,
        abi_version: &str,
        available_symbols: &BTreeSet<String>,
    ) -> Result<CertifiedNativeFfi, TrustError> {
        let contract = self
            .contracts
            .get(library_id)
            .ok_or(TrustError::UncertifiedFfi)?;
        if contract.required_by.is_some()
            || !constant_time_equal(contract.digest_sha256.as_bytes(), digest_sha256.as_bytes())
            || contract.abi_version != abi_version
            || !contract.required_symbols.is_subset(available_symbols)
        {
            return Err(TrustError::UncertifiedFfi);
        }
        Ok(CertifiedNativeFfi {
            library_id: contract.library_id.clone(),
            digest_sha256: contract.digest_sha256.clone(),
            abi_version: contract.abi_version.clone(),
            required_symbols: contract.required_symbols.clone(),
            unsafe_owner: contract.unsafe_owner.clone(),
        })
    }

    pub fn authorize_dependency(
        &self,
        library_id: &str,
        digest_sha256: &str,
        abi_version: &str,
        required_by: &str,
    ) -> Result<CertifiedNativeFfi, TrustError> {
        let contract = self
            .contracts
            .get(library_id)
            .ok_or(TrustError::UncertifiedFfi)?;
        if !contract.required_symbols.is_empty()
            || contract.required_by.as_deref() != Some(required_by)
            || !constant_time_equal(contract.digest_sha256.as_bytes(), digest_sha256.as_bytes())
            || contract.abi_version != abi_version
        {
            return Err(TrustError::UncertifiedFfi);
        }
        Ok(CertifiedNativeFfi {
            library_id: contract.library_id.clone(),
            digest_sha256: contract.digest_sha256.clone(),
            abi_version: contract.abi_version.clone(),
            required_symbols: BTreeSet::new(),
            unsafe_owner: contract.unsafe_owner.clone(),
        })
    }

    pub fn required_symbols_for(
        &self,
        library_id: &str,
        abi_version: &str,
        unsafe_owner: &str,
    ) -> Result<BTreeSet<String>, TrustError> {
        let contract = self
            .contracts
            .get(library_id)
            .ok_or(TrustError::UncertifiedFfi)?;
        if contract.abi_version != abi_version
            || contract.unsafe_owner != unsafe_owner
            || contract.required_symbols.is_empty()
        {
            return Err(TrustError::UncertifiedFfi);
        }
        Ok(contract.required_symbols.clone())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertifiedNativeFfi {
    library_id: String,
    digest_sha256: String,
    abi_version: String,
    required_symbols: BTreeSet<String>,
    unsafe_owner: String,
}

impl CertifiedNativeFfi {
    pub fn library_id(&self) -> &str {
        &self.library_id
    }

    pub fn digest_sha256(&self) -> &str {
        &self.digest_sha256
    }

    pub fn abi_version(&self) -> &str {
        &self.abi_version
    }

    pub fn required_symbols(&self) -> &BTreeSet<String> {
        &self.required_symbols
    }

    pub fn unsafe_owner(&self) -> &str {
        &self.unsafe_owner
    }
}

pub const VIDEO_CODEC_FFI_PROFILE: &str = "ffmpeg-7.1";
pub const VIDEO_CODEC_FFI_UNSAFE_OWNER: &str = "sim_runtime::native_video_codec_ffi";
pub const GENERAL_VIDEO_CODEC_ABI_PROFILE: &str = "ffmpeg-7.1-x86_64-gnu-general-video-v1";
pub const GENERAL_VIDEO_CODEC_ABI_MANIFEST_SHA256: &str =
    "772b01d7db041e0da57dd7d5a09e1b5ae267804b934b2b533ce1fc62e521089d";
const GENERAL_VIDEO_CODEC_ABI_MANIFEST: &[u8] =
    include_bytes!("../abi/video-codec/ffmpeg-7.1-x86_64-gnu-general-video-v1.json");

const VIDEO_CODEC_FFI_TARGETS: [&str; 6] = [
    "aarch64-apple-darwin",
    "aarch64-unknown-linux-gnu",
    "aarch64-pc-windows-msvc",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
];
const VIDEO_CODEC_EXTERNAL_ENCODERS: [&str; 4] = ["aac", "libsvtav1", "libvpx-vp9", "libx264"];

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct VideoCodecFfiCatalogDto {
    schema_version: u16,
    profile: String,
    target: String,
    signer: String,
    signature_algorithm: String,
    signature_domain: String,
    certificate_owner: String,
    unsafe_owner: String,
    runtime_compilation_forbidden: bool,
    redistributes_codec_libraries: bool,
    license_notice_sha256: String,
    external_encoders: Vec<String>,
    libraries: Vec<VideoCodecFfiLibraryDto>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct VideoCodecFfiLibraryDto {
    identity: String,
    filename: String,
    sha256: String,
    abi_major: u16,
    required_symbols: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoCodecAbiManifestDto {
    schema_version: u16,
    profile: String,
    target: String,
    source: GeneralVideoCodecSourceDto,
    compiler_oracle: GeneralVideoCodecCompilerOracleDto,
    historical_manifest_sha256: BTreeMap<String, String>,
    contract: GeneralVideoCodecContractDto,
    projections: BTreeMap<String, BTreeMap<String, u64>>,
    libraries: BTreeMap<String, GeneralVideoCodecLibraryDto>,
    claims: GeneralVideoCodecClaimsDto,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoCodecSourceDto {
    release: String,
    archive_sha256: String,
    signature_sha256: String,
    signing_key_fingerprint: String,
    #[serde(rename = "signature_verified")]
    reviewed_release_signature: bool,
    headers: BTreeMap<String, GeneralVideoCodecHeaderDto>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoCodecHeaderDto {
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoCodecCompilerOracleDto {
    compiler: String,
    compiler_target: String,
    oci_index_sha256: String,
    oci_linux_amd64_manifest_sha256: String,
    command: String,
    object_format: String,
    object_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoCodecContractDto {
    historical_library_count: usize,
    historical_symbol_count: usize,
    required_library_count: usize,
    supplemental_symbol_count: usize,
    general_symbol_count: usize,
    runtime_compilation: bool,
    runtime_loading: bool,
    runtime_symbol_resolution: bool,
    runtime_function_invocation: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoCodecLibraryDto {
    abi_major: u16,
    symbol_version_namespace: String,
    symbols: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoCodecClaimsDto {
    #[serde(rename = "official_release_signature_verified")]
    reviewed_official_release_signature: bool,
    reviewed_typed_declarations: bool,
    installed_ffmpeg_package: bool,
    loadable_native_library: bool,
    certified_native_ffi: bool,
    native_library_loaded: bool,
    runtime_symbol_address_resolved: bool,
    runtime_version_checked: bool,
    codec_availability_probed: bool,
    codec_available: bool,
    executable: bool,
    codec_execution: bool,
    media_allocation: bool,
    publication: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReviewedGeneralVideoCodecLibraryRequirement {
    abi_major: u16,
    symbol_version_namespace: String,
    symbols: BTreeSet<String>,
}

impl ReviewedGeneralVideoCodecLibraryRequirement {
    pub fn abi_major(&self) -> u16 {
        self.abi_major
    }

    pub fn symbol_version_namespace(&self) -> &str {
        &self.symbol_version_namespace
    }

    pub fn symbols(&self) -> &BTreeSet<String> {
        &self.symbols
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReviewedGeneralVideoCodecDeclarations {
    profile: String,
    target: String,
    manifest_sha256: String,
    libraries: BTreeMap<String, ReviewedGeneralVideoCodecLibraryRequirement>,
}

impl ReviewedGeneralVideoCodecDeclarations {
    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn libraries(&self) -> &BTreeMap<String, ReviewedGeneralVideoCodecLibraryRequirement> {
        &self.libraries
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum GeneralVideoCodecDeclarationReviewError {
    #[error("general video codec ABI verification was cancelled")]
    Cancelled,
    #[error("general video codec ABI manifest is malformed")]
    Malformed,
    #[error("general video codec ABI manifest differs from the reviewed declaration oracle")]
    ContractMismatch,
}

pub fn review_general_video_codec_declarations(
    manifest_bytes: &[u8],
    cancellation: &CancellationToken,
) -> Result<ReviewedGeneralVideoCodecDeclarations, GeneralVideoCodecDeclarationReviewError> {
    cancellation
        .check()
        .map_err(|_| GeneralVideoCodecDeclarationReviewError::Cancelled)?;
    if manifest_bytes.is_empty() || manifest_bytes.len() > 256 * 1024 {
        return Err(GeneralVideoCodecDeclarationReviewError::Malformed);
    }
    let manifest_sha256 = format!("{:x}", Sha256::digest(manifest_bytes));
    if manifest_sha256 != GENERAL_VIDEO_CODEC_ABI_MANIFEST_SHA256
        || manifest_bytes != GENERAL_VIDEO_CODEC_ABI_MANIFEST
    {
        return Err(GeneralVideoCodecDeclarationReviewError::ContractMismatch);
    }
    let value = parse_strict_json_value(manifest_bytes)
        .map_err(|_| GeneralVideoCodecDeclarationReviewError::Malformed)?;
    let manifest: GeneralVideoCodecAbiManifestDto = serde_json::from_value(value)
        .map_err(|_| GeneralVideoCodecDeclarationReviewError::Malformed)?;
    let libraries = validate_general_video_codec_abi(&manifest, cancellation)?;
    cancellation
        .check()
        .map_err(|_| GeneralVideoCodecDeclarationReviewError::Cancelled)?;
    Ok(ReviewedGeneralVideoCodecDeclarations {
        profile: manifest.profile,
        target: manifest.target,
        manifest_sha256,
        libraries,
    })
}

fn validate_general_video_codec_abi(
    manifest: &GeneralVideoCodecAbiManifestDto,
    cancellation: &CancellationToken,
) -> Result<
    BTreeMap<String, ReviewedGeneralVideoCodecLibraryRequirement>,
    GeneralVideoCodecDeclarationReviewError,
> {
    let source = &manifest.source;
    let compiler = &manifest.compiler_oracle;
    let contract = &manifest.contract;
    let claims = &manifest.claims;
    if manifest.schema_version != 1
        || manifest.profile != GENERAL_VIDEO_CODEC_ABI_PROFILE
        || manifest.target != "x86_64-unknown-linux-gnu"
        || source.release != "FFmpeg 7.1"
        || source.archive_sha256
            != "40973d44970dbc83ef302b0609f2e74982be2d85916dd2ee7472d30678a7abe6"
        || source.signature_sha256
            != "9bd1689dce76b109034dcc4765a406e84e8799a2fd857b000c0a4d9744b70617"
        || source.signing_key_fingerprint != "FCF986EA15E6E293A5644F10B4322F04D67658D8"
        || !source.reviewed_release_signature
        || compiler.compiler != "gcc 15.2.0"
        || compiler.compiler_target != "x86_64-linux-gnu"
        || compiler.oci_index_sha256
            != "3ae15afe768b06d0c0fe088d822ba5f8045c26630bdacc8d8e7713cf5d8e7289"
        || compiler.oci_linux_amd64_manifest_sha256
            != "c101370f78e4a30be178c11dd18aeee64c65d617908a98157db2392ca73ab04f"
        || compiler.command
            != "gcc -std=c11 -Wall -Wextra -Werror -I<verified-ffmpeg-7.1-source> -c verify-general-video-bindings.c"
        || compiler.object_format != "ELF 64-bit LSB relocatable, x86-64"
        || compiler.object_sha256
            != "c9eb3db4d182c44e71439513cc1108e239f63cc3cda8ee7decc52bbd5c029ae2"
        || contract.historical_library_count != 5
        || contract.historical_symbol_count != 54
        || contract.required_library_count != 6
        || contract.supplemental_symbol_count != 24
        || contract.general_symbol_count != 78
        || contract.runtime_compilation
        || contract.runtime_loading
        || contract.runtime_symbol_resolution
        || contract.runtime_function_invocation
        || !claims.reviewed_official_release_signature
        || !claims.reviewed_typed_declarations
        || claims.installed_ffmpeg_package
        || claims.loadable_native_library
        || claims.certified_native_ffi
        || claims.native_library_loaded
        || claims.runtime_symbol_address_resolved
        || claims.runtime_version_checked
        || claims.codec_availability_probed
        || claims.codec_available
        || claims.executable
        || claims.codec_execution
        || claims.media_allocation
        || claims.publication
    {
        return Err(GeneralVideoCodecDeclarationReviewError::ContractMismatch);
    }

    validate_general_video_codec_headers(&source.headers)?;
    validate_general_video_codec_historical_manifests(&manifest.historical_manifest_sha256)?;
    validate_general_video_codec_projections(&manifest.projections)?;

    let mut libraries = BTreeMap::new();
    let mut supplemental_count = 0_usize;
    let mut general_count = 0_usize;
    let supplemental = general_video_codec_library_extensions()
        .into_iter()
        .map(|(identity, _, symbols)| (identity, symbols.len()))
        .collect::<BTreeMap<_, _>>();
    for (identity, abi_major, expected_symbols) in general_video_codec_library_contracts() {
        cancellation
            .check()
            .map_err(|_| GeneralVideoCodecDeclarationReviewError::Cancelled)?;
        let row = manifest
            .libraries
            .get(identity)
            .ok_or(GeneralVideoCodecDeclarationReviewError::ContractMismatch)?;
        supplemental_count = supplemental_count
            .checked_add(
                *supplemental
                    .get(identity)
                    .ok_or(GeneralVideoCodecDeclarationReviewError::ContractMismatch)?,
            )
            .ok_or(GeneralVideoCodecDeclarationReviewError::ContractMismatch)?;
        general_count = general_count
            .checked_add(expected_symbols.len())
            .ok_or(GeneralVideoCodecDeclarationReviewError::ContractMismatch)?;
        let expected_symbols = expected_symbols.into_iter().collect::<BTreeSet<_>>();
        let manifest_symbols = row
            .symbols
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if row.abi_major != abi_major
            || row.symbol_version_namespace
                != general_video_codec_symbol_version_namespace(identity)
                    .ok_or(GeneralVideoCodecDeclarationReviewError::ContractMismatch)?
            || manifest_symbols.len() != row.symbols.len()
            || manifest_symbols != expected_symbols
            || row.symbols.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(GeneralVideoCodecDeclarationReviewError::ContractMismatch);
        }
        libraries.insert(
            identity.to_owned(),
            ReviewedGeneralVideoCodecLibraryRequirement {
                abi_major,
                symbol_version_namespace: row.symbol_version_namespace.clone(),
                symbols: row.symbols.iter().cloned().collect(),
            },
        );
    }
    if libraries.len() != manifest.libraries.len()
        || libraries.len() != contract.required_library_count
        || supplemental_count != contract.supplemental_symbol_count
        || general_count != contract.general_symbol_count
    {
        return Err(GeneralVideoCodecDeclarationReviewError::ContractMismatch);
    }
    Ok(libraries)
}

fn validate_general_video_codec_headers(
    headers: &BTreeMap<String, GeneralVideoCodecHeaderDto>,
) -> Result<(), GeneralVideoCodecDeclarationReviewError> {
    let expected = [
        (
            "libavcodec/avcodec.h",
            114_986,
            "d6dbc9694974237888592f71020092c4594511e762d609b0076a90e9696ad1b1",
        ),
        (
            "libavcodec/codec_par.h",
            7_727,
            "73b787e33b009eee85e62cc91ae6b58ba23565dc1909da98912ffe9104ef1e51",
        ),
        (
            "libavcodec/packet.h",
            30_025,
            "219679e1ffe55fd22bc81b151fce8315372c87028fea87780be686f2b38f305f",
        ),
        (
            "libavformat/avformat.h",
            119_096,
            "6171ac10e35a67fe04aa8ddfbf84263def2d7ae9159019dd71e5597be5dd3944",
        ),
        (
            "libavutil/channel_layout.h",
            33_387,
            "93191d16f2667e0d1aaced08b6de97ad7d5553bd06e6e1a0dc08fa97cb5b80d3",
        ),
        (
            "libavutil/dict.h",
            9_374,
            "c95cba1829e7886e31469f4e058bb9af3c8f215619c7ffddbc2045c9039f0554",
        ),
        (
            "libavutil/display.h",
            3_472,
            "b9c78c80aa9331b945802b6bcd1db4ecc9ec4f9fad41993cc82b880c0dec2576",
        ),
        (
            "libavutil/frame.h",
            41_581,
            "8218f0295206a6543e7d3974a6fcb1f22100a273b09234d0cb84c60cd1638e75",
        ),
        (
            "libavutil/mathematics.h",
            9_563,
            "64fac2eb3a42fd3788f5585ac8e65c7d5cd82711730d1f030042ba0a62fe1a62",
        ),
        (
            "libavutil/pixdesc.h",
            16_165,
            "5e5b269eae685c54a3f9aa2ff6ab3749e6237a5dfee5bfad1d2c4aef6171a66a",
        ),
        (
            "libswresample/swresample.h",
            22_360,
            "ed295cf600b31f445103b0ed08852f46d24c2dc1f8b3b6cd70a91c4b1eca8937",
        ),
        (
            "libswscale/swscale.h",
            16_928,
            "42ab58ed743efc74ba2152a049b521d41a0fae31595a9b9c6858742391570ca4",
        ),
        (
            "libavfilter/avfilter.h",
            53_419,
            "925d5c51412d5f7d4f404a6ff2f0f40f9d6b004b30e5275b4cf9dd849433c703",
        ),
        (
            "libavfilter/buffersrc.h",
            6_747,
            "b4e8e39bcd4179934064572a40fba7d0830f615dbee3c53859addb28bf2d487a",
        ),
        (
            "libavfilter/buffersink.h",
            6_923,
            "ab5fe19d11f039ba40d97bc40855d3ffc3df4c8a2d91be6efb141304510c8f34",
        ),
        (
            "libavfilter/version.h",
            1_653,
            "3469d7376608039a622aad725daaf5d624bd8e9e25d6956eea621ab2a237a66b",
        ),
        (
            "libavfilter/version_major.h",
            1_288,
            "7a8bf18dba3ba32195c213a29a5779e21949c43b41a7c2a7c28f2822e69be9e5",
        ),
    ];
    if headers.len() != expected.len()
        || expected.into_iter().any(|(name, bytes, sha256)| {
            headers
                .get(name)
                .is_none_or(|header| header.bytes != bytes || header.sha256 != sha256)
        })
    {
        return Err(GeneralVideoCodecDeclarationReviewError::ContractMismatch);
    }
    Ok(())
}

fn validate_general_video_codec_historical_manifests(
    historical: &BTreeMap<String, String>,
) -> Result<(), GeneralVideoCodecDeclarationReviewError> {
    let expected = [
        (
            "ffmpeg-7.1-x86_64-gnu-v1.json",
            "cefc4429df3abe09508a4f6c91ba2c749580109fab52f3f79e9db7a998b4bb76",
        ),
        (
            "ffmpeg-7.1-x86_64-gnu-data-plane-v1.json",
            "c5d4780cac865e9dd327f42f80278e6cfd4d85a94cbdd48cafbc981872b440ec",
        ),
        (
            "ffmpeg-7.1-x86_64-gnu-container-metadata-v1.json",
            "cb989b103743491f19d41a7f451bac06c055496a9ff2cf3d37b12d92b133192d",
        ),
        (
            "ffmpeg-7.1-x86_64-gnu-vp9-alpha-v1.json",
            "766d286761651abd80698aa986cb79e2c5022d432573f13e949c851ad10b05c1",
        ),
        (
            "ffmpeg-7.1-x86_64-gnu-av1-pixel-format-v1.json",
            "c467a7efa46dd2291bbe9a5fc1a0ddaa1a8685fa74de377f1b6b366ba464e6ca",
        ),
        (
            "ffmpeg-7.1-x86_64-gnu-h264-mp4-10bit-v1.json",
            "b81706ef04a75af4fb3547de17f6f950b1df826b2aac4f9b2cb78df0ab0a4d6a",
        ),
    ];
    if historical.len() != expected.len()
        || expected
            .into_iter()
            .any(|(name, sha256)| historical.get(name).is_none_or(|actual| actual != sha256))
    {
        return Err(GeneralVideoCodecDeclarationReviewError::ContractMismatch);
    }
    Ok(())
}

fn validate_general_video_codec_projections(
    projections: &BTreeMap<String, BTreeMap<String, u64>>,
) -> Result<(), GeneralVideoCodecDeclarationReviewError> {
    let expected = [
        (
            "AVCodecParameters",
            [
                ("size", 176),
                ("alignment", 8),
                ("codec_type", 0),
                ("codec_id", 4),
                ("format", 44),
                ("bits_per_raw_sample", 60),
                ("width", 72),
                ("height", 76),
                ("ch_layout", 128),
                ("sample_rate", 152),
            ]
            .as_slice(),
        ),
        (
            "AVStream",
            [
                ("size", 232),
                ("alignment", 8),
                ("codecpar", 16),
                ("time_base", 32),
                ("start_time", 40),
                ("duration", 48),
                ("nb_frames", 56),
                ("metadata", 80),
                ("avg_frame_rate", 88),
            ]
            .as_slice(),
        ),
        (
            "AVFormatContext",
            [
                ("size", 472),
                ("alignment", 8),
                ("nb_streams", 44),
                ("streams", 48),
                ("start_time", 96),
                ("duration", 104),
                ("metadata", 192),
            ]
            .as_slice(),
        ),
        (
            "AVFrame",
            [
                ("size", 440),
                ("alignment", 8),
                ("pts", 136),
                ("sample_rate", 192),
                ("best_effort_timestamp", 320),
                ("metadata", 336),
                ("ch_layout", 408),
                ("duration", 432),
            ]
            .as_slice(),
        ),
    ];
    if projections.len() != expected.len()
        || expected.into_iter().any(|(name, fields)| {
            projections.get(name).is_none_or(|actual| {
                actual.len() != fields.len()
                    || fields
                        .iter()
                        .any(|(field, value)| actual.get(*field) != Some(value))
            })
        })
    {
        return Err(GeneralVideoCodecDeclarationReviewError::ContractMismatch);
    }
    Ok(())
}

const VIDEO_CODEC_DEPENDENCY_CONTRACT_TARGET: &str = "x86_64-unknown-linux-gnu";
const MAX_VIDEO_CODEC_DEPENDENCY_IMAGES: usize = 64;
const MAX_VIDEO_CODEC_DEPENDENCY_EDGES: usize = 512;
const MAX_VIDEO_CODEC_DEPENDENCY_RETAINED_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const VIDEO_CODEC_SYSTEM_ELF_DEPENDENCIES: [&str; 8] = [
    "ld-linux-x86-64.so.2",
    "libc.so.6",
    "libdl.so.2",
    "libgcc_s.so.1",
    "libm.so.6",
    "libpthread.so.0",
    "librt.so.1",
    "libstdc++.so.6",
];

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct VideoCodecDependencyContractDto {
    schema_version: u16,
    profile: String,
    target: String,
    signer: String,
    signature_algorithm: String,
    signature_domain: String,
    certificate_owner: String,
    unsafe_owner: String,
    runtime_compilation_forbidden: bool,
    redistributes_codec_libraries: bool,
    primary_catalog_sha256: String,
    source_archive_sha256: String,
    build_recipe_sha256: String,
    license_bundle_sha256: String,
    system_libraries: Vec<String>,
    encoder_providers: Vec<VideoCodecEncoderProviderDto>,
    dependencies: Vec<VideoCodecDependencyDto>,
    edges: Vec<VideoCodecDependencyEdgeDto>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct VideoCodecEncoderProviderDto {
    encoder: String,
    provider: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct VideoCodecDependencyDto {
    identity: String,
    filename: String,
    sha256: String,
    abi_version: String,
    certificate_sponsor: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct VideoCodecDependencyEdgeDto {
    consumer: String,
    dependency: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct VideoCodecDependencyEdge {
    consumer: String,
    dependency: String,
}

impl VideoCodecDependencyEdge {
    pub fn consumer(&self) -> &str {
        &self.consumer
    }

    pub fn dependency(&self) -> &str {
        &self.dependency
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VideoCodecDependencyIdentity {
    filename: String,
    digest_sha256: String,
    abi_version: String,
    certificate_sponsor: String,
}

impl VideoCodecDependencyIdentity {
    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn digest_sha256(&self) -> &str {
        &self.digest_sha256
    }

    pub fn abi_version(&self) -> &str {
        &self.abi_version
    }

    pub fn certificate_sponsor(&self) -> &str {
        &self.certificate_sponsor
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VideoCodecFfiLibraryIdentity {
    filename: String,
    digest_sha256: String,
    abi_major: u16,
}

impl VideoCodecFfiLibraryIdentity {
    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn digest_sha256(&self) -> &str {
        &self.digest_sha256
    }

    pub fn abi_major(&self) -> u16 {
        self.abi_major
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedVideoCodecFfiCatalog {
    target: String,
    catalog_sha256: String,
    registry: NativeFfiRegistry,
    libraries: BTreeMap<String, VideoCodecFfiLibraryIdentity>,
}

impl VerifiedVideoCodecFfiCatalog {
    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn libraries(&self) -> &BTreeMap<String, VideoCodecFfiLibraryIdentity> {
        &self.libraries
    }

    pub fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedVideoCodecDependencyContract {
    target: String,
    primary_catalog_sha256: String,
    source_archive_sha256: String,
    build_recipe_sha256: String,
    license_bundle_sha256: String,
    system_libraries: BTreeSet<String>,
    encoder_providers: BTreeMap<String, String>,
    dependencies: BTreeMap<String, VideoCodecDependencyIdentity>,
    edges: BTreeSet<VideoCodecDependencyEdge>,
    _registry: NativeFfiRegistry,
}

impl VerifiedVideoCodecDependencyContract {
    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn primary_catalog_sha256(&self) -> &str {
        &self.primary_catalog_sha256
    }

    pub fn source_archive_sha256(&self) -> &str {
        &self.source_archive_sha256
    }

    pub fn build_recipe_sha256(&self) -> &str {
        &self.build_recipe_sha256
    }

    pub fn license_bundle_sha256(&self) -> &str {
        &self.license_bundle_sha256
    }

    pub fn system_libraries(&self) -> &BTreeSet<String> {
        &self.system_libraries
    }

    pub fn encoder_providers(&self) -> &BTreeMap<String, String> {
        &self.encoder_providers
    }

    pub fn dependencies(&self) -> &BTreeMap<String, VideoCodecDependencyIdentity> {
        &self.dependencies
    }

    pub fn edges(&self) -> &BTreeSet<VideoCodecDependencyEdge> {
        &self.edges
    }
}

pub struct CapturedVideoCodecPackage {
    target: String,
    catalog_sha256: String,
    libraries: BTreeMap<String, VideoCodecFfiLibraryIdentity>,
    _sealed_images: BTreeMap<String, RetainedNativeLibraryImage>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VideoCodecElfLibraryIdentity {
    soname: String,
    exported_symbols: BTreeSet<String>,
    callable_symbols: BTreeMap<String, VideoCodecCallableElfSymbolIdentity>,
    needed_libraries: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VideoCodecCallableElfSymbolIdentity {
    value: u64,
    size: u64,
    version_namespace: String,
}

impl VideoCodecCallableElfSymbolIdentity {
    pub fn value(&self) -> u64 {
        self.value
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn version_namespace(&self) -> &str {
        &self.version_namespace
    }
}

impl VideoCodecElfLibraryIdentity {
    pub fn soname(&self) -> &str {
        &self.soname
    }

    pub fn exported_symbols(&self) -> &BTreeSet<String> {
        &self.exported_symbols
    }

    pub fn callable_symbols(&self) -> &BTreeMap<String, VideoCodecCallableElfSymbolIdentity> {
        &self.callable_symbols
    }

    pub fn needed_libraries(&self) -> &BTreeSet<String> {
        &self.needed_libraries
    }
}

pub struct InspectedVideoCodecPackage {
    captured: CapturedVideoCodecPackage,
    elf_libraries: BTreeMap<String, VideoCodecElfLibraryIdentity>,
}

impl InspectedVideoCodecPackage {
    pub fn target(&self) -> &str {
        self.captured.target()
    }

    pub fn libraries(&self) -> &BTreeMap<String, VideoCodecFfiLibraryIdentity> {
        self.captured.libraries()
    }

    pub fn catalog_sha256(&self) -> &str {
        self.captured.catalog_sha256()
    }

    pub fn elf_libraries(&self) -> &BTreeMap<String, VideoCodecElfLibraryIdentity> {
        &self.elf_libraries
    }
}

pub struct CertifiedInspectedVideoCodecPackage {
    inspected: InspectedVideoCodecPackage,
    certificates: BTreeMap<String, CertifiedNativeFfi>,
}

impl CertifiedInspectedVideoCodecPackage {
    pub fn target(&self) -> &str {
        self.inspected.target()
    }

    pub fn libraries(&self) -> &BTreeMap<String, VideoCodecFfiLibraryIdentity> {
        self.inspected.libraries()
    }

    pub fn catalog_sha256(&self) -> &str {
        self.inspected.catalog_sha256()
    }

    pub fn elf_libraries(&self) -> &BTreeMap<String, VideoCodecElfLibraryIdentity> {
        self.inspected.elf_libraries()
    }

    pub fn certificates(&self) -> &BTreeMap<String, CertifiedNativeFfi> {
        &self.certificates
    }
}

impl CapturedVideoCodecPackage {
    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn libraries(&self) -> &BTreeMap<String, VideoCodecFfiLibraryIdentity> {
        &self.libraries
    }

    pub fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }
}

pub struct CertifiedVideoCodecDependencyClosure {
    primary: CertifiedInspectedVideoCodecPackage,
    contract: VerifiedVideoCodecDependencyContract,
    dependency_elf_libraries: BTreeMap<String, VideoCodecElfLibraryIdentity>,
    dependency_certificates: BTreeMap<String, CertifiedNativeFfi>,
    dependency_first_order: Vec<String>,
    retained_dependency_bytes: u64,
    _sealed_dependency_images: BTreeMap<String, RetainedNativeLibraryImage>,
}

impl CertifiedVideoCodecDependencyClosure {
    pub fn target(&self) -> &str {
        self.contract.target()
    }

    pub fn primary_catalog_sha256(&self) -> &str {
        self.primary.catalog_sha256()
    }

    pub fn primary_libraries(&self) -> &BTreeMap<String, VideoCodecFfiLibraryIdentity> {
        self.primary.libraries()
    }

    pub fn dependencies(&self) -> &BTreeMap<String, VideoCodecDependencyIdentity> {
        self.contract.dependencies()
    }

    pub fn edges(&self) -> &BTreeSet<VideoCodecDependencyEdge> {
        self.contract.edges()
    }

    pub fn encoder_providers(&self) -> &BTreeMap<String, String> {
        self.contract.encoder_providers()
    }

    pub fn primary_elf_libraries(&self) -> &BTreeMap<String, VideoCodecElfLibraryIdentity> {
        self.primary.elf_libraries()
    }

    pub fn dependency_elf_libraries(&self) -> &BTreeMap<String, VideoCodecElfLibraryIdentity> {
        &self.dependency_elf_libraries
    }

    pub fn primary_certificates(&self) -> &BTreeMap<String, CertifiedNativeFfi> {
        self.primary.certificates()
    }

    pub fn dependency_certificates(&self) -> &BTreeMap<String, CertifiedNativeFfi> {
        &self.dependency_certificates
    }

    pub fn dependency_first_order(&self) -> &[String] {
        &self.dependency_first_order
    }

    pub fn retained_dependency_bytes(&self) -> u64 {
        self.retained_dependency_bytes
    }

    pub(crate) fn reviewed_system_libraries(&self) -> &BTreeSet<String> {
        self.contract.system_libraries()
    }

    pub(crate) fn source_archive_sha256(&self) -> &str {
        self.contract.source_archive_sha256()
    }

    pub(crate) fn retained_loader_paths(&self) -> Option<BTreeMap<String, PathBuf>> {
        let mut paths = BTreeMap::new();
        for identity in &self.dependency_first_order {
            let retained = self
                .primary
                .inspected
                .captured
                ._sealed_images
                .get(identity)
                .or_else(|| self._sealed_dependency_images.get(identity))?;
            paths.insert(identity.clone(), retained.loader_path().to_path_buf());
        }
        Some(paths)
    }
}

#[derive(Debug, Error)]
pub enum VideoCodecDependencyClosureError {
    #[error("video codec dependency closure certification was cancelled")]
    Cancelled,
    #[error("video codec dependency closures are unsupported for this target")]
    UnsupportedTarget,
    #[error("video codec dependency paths are incomplete or contain unexpected identities")]
    IncompletePathSet,
    #[error("video codec dependency closure differs from the signed contract")]
    ContractMismatch,
    #[error("video codec dependency image {identity} could not be captured: {reason}")]
    InvalidImage { identity: String, reason: String },
    #[error("video codec dependency image {identity} is not an admitted ELF object: {reason}")]
    InvalidElf { identity: String, reason: String },
    #[error("video codec library {consumer} requires unaccounted dependency {soname}")]
    UnaccountedDependency { consumer: String, soname: String },
    #[error("video codec dependency graph is invalid")]
    InvalidGraph,
    #[error("video codec dependency closure exceeds the reviewed resource limits")]
    ResourceLimitExceeded,
    #[error(transparent)]
    Trust(#[from] TrustError),
}

#[derive(Debug, Error)]
pub enum VideoCodecPackageCaptureError {
    #[error("video codec package capture was cancelled")]
    Cancelled,
    #[error("video codec package capture is unsupported on this platform")]
    UnsupportedPlatform,
    #[error("video codec package paths are incomplete or contain unexpected libraries")]
    Incomplete,
    #[error("video codec package image {identity} differs from the signed catalog")]
    ContractMismatch { identity: String },
    #[error("video codec package image {identity} could not be captured: {reason}")]
    InvalidImage { identity: String, reason: String },
    #[error("video codec package image {identity} is not an admitted ELF object: {reason}")]
    InvalidElf { identity: String, reason: String },
    #[error("video codec package image {identity} does not export required symbol {symbol}")]
    MissingSymbol { identity: String, symbol: String },
    #[error("video codec package image {identity} has a non-callable required symbol {symbol}")]
    InvalidCallableSymbol { identity: String, symbol: String },
}

pub fn capture_video_codec_package(
    catalog: &VerifiedVideoCodecFfiCatalog,
    paths: BTreeMap<String, PathBuf>,
    cancellation: &CancellationToken,
) -> Result<CapturedVideoCodecPackage, VideoCodecPackageCaptureError> {
    capture_video_codec_package_internal(catalog, paths, None, cancellation)
        .map(|(captured, _)| captured)
}

pub fn capture_and_inspect_video_codec_package(
    catalog: &VerifiedVideoCodecFfiCatalog,
    paths: BTreeMap<String, PathBuf>,
    cancellation: &CancellationToken,
) -> Result<InspectedVideoCodecPackage, VideoCodecPackageCaptureError> {
    let expected_machine = match catalog.target() {
        "x86_64-unknown-linux-gnu" => 62,
        "aarch64-unknown-linux-gnu" => 183,
        _ => return Err(VideoCodecPackageCaptureError::UnsupportedPlatform),
    };
    let (captured, elf_libraries) =
        capture_video_codec_package_internal(catalog, paths, Some(expected_machine), cancellation)?;
    Ok(InspectedVideoCodecPackage {
        captured,
        elf_libraries,
    })
}

fn capture_video_codec_package_internal(
    catalog: &VerifiedVideoCodecFfiCatalog,
    paths: BTreeMap<String, PathBuf>,
    expected_machine: Option<u16>,
    cancellation: &CancellationToken,
) -> Result<
    (
        CapturedVideoCodecPackage,
        BTreeMap<String, VideoCodecElfLibraryIdentity>,
    ),
    VideoCodecPackageCaptureError,
> {
    cancellation
        .check()
        .map_err(|_| VideoCodecPackageCaptureError::Cancelled)?;
    if paths.len() != catalog.libraries.len()
        || !catalog
            .libraries
            .keys()
            .all(|identity| paths.contains_key(identity))
    {
        return Err(VideoCodecPackageCaptureError::Incomplete);
    }

    let mut sealed_images = BTreeMap::new();
    let mut elf_libraries = BTreeMap::new();
    for (identity, expected) in &catalog.libraries {
        cancellation
            .check()
            .map_err(|_| VideoCodecPackageCaptureError::Cancelled)?;
        let path = paths
            .get(identity)
            .ok_or(VideoCodecPackageCaptureError::Incomplete)?;
        if path.file_name().and_then(|name| name.to_str()) != Some(expected.filename()) {
            return Err(VideoCodecPackageCaptureError::ContractMismatch {
                identity: identity.clone(),
            });
        }
        let captured = capture_native_library_image(path, cancellation)
            .map_err(|error| map_video_codec_image_error(identity, error))?;
        if captured.digest_sha256() != expected.digest_sha256() {
            return Err(VideoCodecPackageCaptureError::ContractMismatch {
                identity: identity.clone(),
            });
        }
        if let Some(expected_machine) = expected_machine {
            let dynamic =
                inspect_elf64_dynamic_contract(captured.bytes(), expected_machine, cancellation)
                    .map_err(|error| match error {
                        NativeElfInspectionError::Cancelled(_) => {
                            VideoCodecPackageCaptureError::Cancelled
                        }
                        NativeElfInspectionError::Invalid(reason) => {
                            VideoCodecPackageCaptureError::InvalidElf {
                                identity: identity.clone(),
                                reason,
                            }
                        }
                    })?;
            if dynamic.soname() != Some(expected.filename()) {
                return Err(VideoCodecPackageCaptureError::ContractMismatch {
                    identity: identity.clone(),
                });
            }
            let required_symbols = catalog
                .registry
                .required_symbols_for(
                    identity,
                    &video_codec_abi_version(expected.abi_major()),
                    VIDEO_CODEC_FFI_UNSAFE_OWNER,
                )
                .map_err(|_| VideoCodecPackageCaptureError::ContractMismatch {
                    identity: identity.clone(),
                })?;
            if let Some(symbol) = required_symbols
                .iter()
                .find(|symbol| !dynamic.symbols().contains(*symbol))
            {
                return Err(VideoCodecPackageCaptureError::MissingSymbol {
                    identity: identity.clone(),
                    symbol: symbol.clone(),
                });
            }
            let callable_symbols =
                checked_video_codec_callable_symbols(identity, &required_symbols, &dynamic)?;
            elf_libraries.insert(
                identity.clone(),
                VideoCodecElfLibraryIdentity {
                    soname: expected.filename().to_owned(),
                    exported_symbols: dynamic.symbols().clone(),
                    callable_symbols,
                    needed_libraries: dynamic.needed().clone(),
                },
            );
        }
        let retained = captured
            .seal(&format!("video-codec-{identity}"), cancellation)
            .map_err(|error| map_video_codec_image_error(identity, error))?;
        sealed_images.insert(identity.clone(), retained);
    }
    cancellation
        .check()
        .map_err(|_| VideoCodecPackageCaptureError::Cancelled)?;
    Ok((
        CapturedVideoCodecPackage {
            target: catalog.target.clone(),
            catalog_sha256: catalog.catalog_sha256.clone(),
            libraries: catalog.libraries.clone(),
            _sealed_images: sealed_images,
        },
        elf_libraries,
    ))
}

fn checked_video_codec_callable_symbols(
    identity: &str,
    required_symbols: &BTreeSet<String>,
    dynamic: &crate::native_ffi_elf::NativeElfDynamicContract,
) -> Result<BTreeMap<String, VideoCodecCallableElfSymbolIdentity>, VideoCodecPackageCaptureError> {
    let expected_version_namespace =
        video_codec_symbol_version_namespace(identity).ok_or_else(|| {
            VideoCodecPackageCaptureError::ContractMismatch {
                identity: identity.to_owned(),
            }
        })?;
    let mut callable_symbols = BTreeMap::new();
    for symbol in required_symbols {
        let identities = dynamic.symbol_identities().get(symbol).ok_or_else(|| {
            VideoCodecPackageCaptureError::MissingSymbol {
                identity: identity.to_owned(),
                symbol: symbol.clone(),
            }
        })?;
        let [admitted] = identities.as_slice() else {
            return Err(VideoCodecPackageCaptureError::InvalidCallableSymbol {
                identity: identity.to_owned(),
                symbol: symbol.clone(),
            });
        };
        let Some(version) = admitted.version.as_ref() else {
            return Err(VideoCodecPackageCaptureError::InvalidCallableSymbol {
                identity: identity.to_owned(),
                symbol: symbol.clone(),
            });
        };
        if admitted.binding != 1
            || admitted.kind != 2
            || admitted.visibility != 0
            || admitted.section_index == 0
            || admitted.value == 0
            || !admitted.executable
            || !version.is_default
            || version.name != expected_version_namespace
        {
            return Err(VideoCodecPackageCaptureError::InvalidCallableSymbol {
                identity: identity.to_owned(),
                symbol: symbol.clone(),
            });
        }
        callable_symbols.insert(
            symbol.clone(),
            VideoCodecCallableElfSymbolIdentity {
                value: admitted.value,
                size: admitted.size,
                version_namespace: version.name.clone(),
            },
        );
    }
    Ok(callable_symbols)
}

fn map_video_codec_image_error(
    identity: &str,
    error: NativeLibraryImageError,
) -> VideoCodecPackageCaptureError {
    match error {
        NativeLibraryImageError::Cancelled => VideoCodecPackageCaptureError::Cancelled,
        NativeLibraryImageError::UnsupportedPlatform => {
            VideoCodecPackageCaptureError::UnsupportedPlatform
        }
        NativeLibraryImageError::Invalid(reason) => VideoCodecPackageCaptureError::InvalidImage {
            identity: identity.to_owned(),
            reason,
        },
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeVideoCodecLibraryObservation {
    identity: String,
    filename: String,
    digest_sha256: String,
    abi_major: u16,
    available_symbols: BTreeSet<String>,
}

impl NativeVideoCodecLibraryObservation {
    pub fn checked(
        identity: impl Into<String>,
        filename: impl Into<String>,
        digest_sha256: impl Into<String>,
        abi_major: u16,
        available_symbols: impl IntoIterator<Item = String>,
    ) -> Result<Self, VideoCodecFfiCertificationError> {
        let observation = Self {
            identity: identity.into(),
            filename: filename.into(),
            digest_sha256: digest_sha256.into(),
            abi_major,
            available_symbols: available_symbols.into_iter().collect(),
        };
        if !video_codec_library_identity_valid(&observation.identity)
            || !video_codec_filename_valid(&observation.filename)
            || validate_sha256(&observation.digest_sha256).is_err()
            || observation.abi_major == 0
            || observation.available_symbols.is_empty()
            || observation
                .available_symbols
                .iter()
                .any(|symbol| !video_codec_symbol_valid(symbol))
        {
            return Err(VideoCodecFfiCertificationError::InvalidObservation);
        }
        Ok(observation)
    }
}

#[derive(Clone, Debug)]
pub struct CertifiedVideoCodecFfi {
    target: String,
    certificates: BTreeMap<String, CertifiedNativeFfi>,
}

impl CertifiedVideoCodecFfi {
    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn certificates(&self) -> &BTreeMap<String, CertifiedNativeFfi> {
        &self.certificates
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum VideoCodecFfiCatalogError {
    #[error("video codec catalog verification was cancelled")]
    Cancelled,
    #[error(transparent)]
    Trust(#[from] TrustError),
    #[error("signed video codec FFI catalog is malformed")]
    Malformed,
    #[error("signed video codec FFI catalog differs from the reviewed ABI")]
    ContractMismatch,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum VideoCodecDependencyContractError {
    #[error("video codec dependency contract verification was cancelled")]
    Cancelled,
    #[error(transparent)]
    Trust(#[from] TrustError),
    #[error("signed video codec dependency contract is malformed")]
    Malformed,
    #[error("signed video codec dependency contract differs from the reviewed closure policy")]
    ContractMismatch,
    #[error("video codec dependency contracts are unsupported for this target")]
    UnsupportedTarget,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum VideoCodecFfiCertificationError {
    #[error("video codec FFI certification was cancelled")]
    Cancelled,
    #[error(transparent)]
    Trust(#[from] TrustError),
    #[error("video codec library observation is malformed")]
    InvalidObservation,
    #[error("video codec library observations are incomplete or duplicated")]
    IncompleteObservationSet,
    #[error("video codec library observation differs from the signed catalog")]
    ObservationMismatch,
    #[error("inspected video codec package differs from the signed catalog")]
    InspectedPackageMismatch,
}

pub fn verify_video_codec_ffi_catalog(
    catalog_bytes: &[u8],
    signature_receipt: &[u8],
    verification_key: &VideoCodecPackageVerificationKey,
    cancellation: &CancellationToken,
) -> Result<VerifiedVideoCodecFfiCatalog, VideoCodecFfiCatalogError> {
    cancellation
        .check()
        .map_err(|_| VideoCodecFfiCatalogError::Cancelled)?;
    if catalog_bytes.is_empty() || catalog_bytes.len() > 1024 * 1024 {
        return Err(VideoCodecFfiCatalogError::Malformed);
    }
    let value =
        parse_strict_json_value(catalog_bytes).map_err(|_| VideoCodecFfiCatalogError::Malformed)?;
    let catalog: VideoCodecFfiCatalogDto =
        serde_json::from_value(value).map_err(|_| VideoCodecFfiCatalogError::Malformed)?;
    let mut canonical =
        serde_json::to_vec(&catalog).map_err(|_| VideoCodecFfiCatalogError::Malformed)?;
    canonical.push(b'\n');
    if canonical != catalog_bytes {
        return Err(VideoCodecFfiCatalogError::Malformed);
    }
    validate_video_codec_catalog_envelope(&catalog)?;
    verification_key.verify_package(&catalog.signer, catalog_bytes, signature_receipt)?;
    cancellation
        .check()
        .map_err(|_| VideoCodecFfiCatalogError::Cancelled)?;

    let mut contracts = Vec::with_capacity(catalog.libraries.len());
    let mut identities = BTreeMap::new();
    for library in &catalog.libraries {
        cancellation
            .check()
            .map_err(|_| VideoCodecFfiCatalogError::Cancelled)?;
        let abi_version = video_codec_abi_version(library.abi_major);
        contracts.push(NativeFfiContract::new(
            library.identity.clone(),
            library.sha256.clone(),
            abi_version,
            library.required_symbols.clone(),
            VIDEO_CODEC_FFI_UNSAFE_OWNER,
        )?);
        identities.insert(
            library.identity.clone(),
            VideoCodecFfiLibraryIdentity {
                filename: library.filename.clone(),
                digest_sha256: library.sha256.clone(),
                abi_major: library.abi_major,
            },
        );
    }
    Ok(VerifiedVideoCodecFfiCatalog {
        target: catalog.target,
        catalog_sha256: format!("{:x}", Sha256::digest(catalog_bytes)),
        registry: NativeFfiRegistry::new(contracts)?,
        libraries: identities,
    })
}

pub fn verify_video_codec_dependency_contract(
    primary: &VerifiedVideoCodecFfiCatalog,
    contract_bytes: &[u8],
    signature_receipt: &[u8],
    verification_key: &VideoCodecPackageVerificationKey,
    cancellation: &CancellationToken,
) -> Result<VerifiedVideoCodecDependencyContract, VideoCodecDependencyContractError> {
    cancellation
        .check()
        .map_err(|_| VideoCodecDependencyContractError::Cancelled)?;
    if primary.target() != VIDEO_CODEC_DEPENDENCY_CONTRACT_TARGET {
        return Err(VideoCodecDependencyContractError::UnsupportedTarget);
    }
    if contract_bytes.is_empty() || contract_bytes.len() > 1024 * 1024 {
        return Err(VideoCodecDependencyContractError::Malformed);
    }
    let value = parse_strict_json_value(contract_bytes)
        .map_err(|_| VideoCodecDependencyContractError::Malformed)?;
    let contract: VideoCodecDependencyContractDto =
        serde_json::from_value(value).map_err(|_| VideoCodecDependencyContractError::Malformed)?;
    let mut canonical =
        serde_json::to_vec(&contract).map_err(|_| VideoCodecDependencyContractError::Malformed)?;
    canonical.push(b'\n');
    if canonical != contract_bytes {
        return Err(VideoCodecDependencyContractError::Malformed);
    }
    validate_video_codec_dependency_contract_envelope(primary, &contract, cancellation)?;
    verification_key.authority.verify(
        VIDEO_CODEC_DEPENDENCY_CONTRACT_SIGNATURE_DOMAIN,
        &contract.signer,
        contract_bytes,
        signature_receipt,
        TrustError::UnknownVideoCodecPackageSigner,
        TrustError::InvalidVideoCodecPackageSignature,
    )?;
    cancellation
        .check()
        .map_err(|_| VideoCodecDependencyContractError::Cancelled)?;

    let mut registry_contracts = primary
        .registry
        .contracts
        .values()
        .cloned()
        .collect::<Vec<_>>();
    let mut dependencies = BTreeMap::new();
    for dependency in contract.dependencies {
        cancellation
            .check()
            .map_err(|_| VideoCodecDependencyContractError::Cancelled)?;
        registry_contracts.push(NativeFfiContract::new_dependency(
            dependency.identity.clone(),
            dependency.sha256.clone(),
            dependency.abi_version.clone(),
            dependency.certificate_sponsor.clone(),
            VIDEO_CODEC_FFI_UNSAFE_OWNER,
        )?);
        dependencies.insert(
            dependency.identity,
            VideoCodecDependencyIdentity {
                filename: dependency.filename,
                digest_sha256: dependency.sha256,
                abi_version: dependency.abi_version,
                certificate_sponsor: dependency.certificate_sponsor,
            },
        );
    }
    let registry = NativeFfiRegistry::new(registry_contracts)?;
    let edges = contract
        .edges
        .into_iter()
        .map(|edge| VideoCodecDependencyEdge {
            consumer: edge.consumer,
            dependency: edge.dependency,
        })
        .collect();
    let encoder_providers = contract
        .encoder_providers
        .into_iter()
        .map(|provider| (provider.encoder, provider.provider))
        .collect();
    cancellation
        .check()
        .map_err(|_| VideoCodecDependencyContractError::Cancelled)?;
    Ok(VerifiedVideoCodecDependencyContract {
        target: contract.target,
        primary_catalog_sha256: contract.primary_catalog_sha256,
        source_archive_sha256: contract.source_archive_sha256,
        build_recipe_sha256: contract.build_recipe_sha256,
        license_bundle_sha256: contract.license_bundle_sha256,
        system_libraries: contract.system_libraries.into_iter().collect(),
        encoder_providers,
        dependencies,
        edges,
        _registry: registry,
    })
}

pub fn certify_video_codec_ffi(
    verified: VerifiedVideoCodecFfiCatalog,
    observations: impl IntoIterator<Item = NativeVideoCodecLibraryObservation>,
    cancellation: &CancellationToken,
) -> Result<CertifiedVideoCodecFfi, VideoCodecFfiCertificationError> {
    cancellation
        .check()
        .map_err(|_| VideoCodecFfiCertificationError::Cancelled)?;
    let mut observations_by_identity = BTreeMap::new();
    for observation in observations {
        if observations_by_identity
            .insert(observation.identity.clone(), observation)
            .is_some()
        {
            return Err(VideoCodecFfiCertificationError::IncompleteObservationSet);
        }
    }
    if observations_by_identity.len() != verified.libraries.len() {
        return Err(VideoCodecFfiCertificationError::IncompleteObservationSet);
    }

    let mut certificates = BTreeMap::new();
    for (identity, expected) in &verified.libraries {
        cancellation
            .check()
            .map_err(|_| VideoCodecFfiCertificationError::Cancelled)?;
        let observation = observations_by_identity
            .get(identity)
            .ok_or(VideoCodecFfiCertificationError::IncompleteObservationSet)?;
        if observation.filename != expected.filename
            || observation.digest_sha256 != expected.digest_sha256
            || observation.abi_major != expected.abi_major
        {
            return Err(VideoCodecFfiCertificationError::ObservationMismatch);
        }
        let certificate = verified.registry.authorize(
            identity,
            &observation.digest_sha256,
            &video_codec_abi_version(observation.abi_major),
            &observation.available_symbols,
        )?;
        certificates.insert(identity.clone(), certificate);
    }
    cancellation
        .check()
        .map_err(|_| VideoCodecFfiCertificationError::Cancelled)?;
    Ok(CertifiedVideoCodecFfi {
        target: verified.target,
        certificates,
    })
}

pub fn certify_inspected_video_codec_package(
    verified: &VerifiedVideoCodecFfiCatalog,
    inspected: InspectedVideoCodecPackage,
    cancellation: &CancellationToken,
) -> Result<CertifiedInspectedVideoCodecPackage, VideoCodecFfiCertificationError> {
    cancellation
        .check()
        .map_err(|_| VideoCodecFfiCertificationError::Cancelled)?;
    if inspected.target() != verified.target()
        || inspected.catalog_sha256() != verified.catalog_sha256()
        || inspected.libraries() != verified.libraries()
        || inspected.elf_libraries().len() != verified.libraries().len()
    {
        return Err(VideoCodecFfiCertificationError::InspectedPackageMismatch);
    }

    let mut certificates = BTreeMap::new();
    for (identity, expected) in verified.libraries() {
        cancellation
            .check()
            .map_err(|_| VideoCodecFfiCertificationError::Cancelled)?;
        let elf = inspected
            .elf_libraries()
            .get(identity)
            .ok_or(VideoCodecFfiCertificationError::InspectedPackageMismatch)?;
        if elf.soname() != expected.filename() {
            return Err(VideoCodecFfiCertificationError::InspectedPackageMismatch);
        }
        let certificate = verified.registry.authorize(
            identity,
            expected.digest_sha256(),
            &video_codec_abi_version(expected.abi_major()),
            &elf.callable_symbols().keys().cloned().collect(),
        )?;
        certificates.insert(identity.clone(), certificate);
    }
    cancellation
        .check()
        .map_err(|_| VideoCodecFfiCertificationError::Cancelled)?;
    Ok(CertifiedInspectedVideoCodecPackage {
        inspected,
        certificates,
    })
}

pub fn certify_video_codec_dependency_closure(
    primary: CertifiedInspectedVideoCodecPackage,
    contract: VerifiedVideoCodecDependencyContract,
    dependency_paths: BTreeMap<String, PathBuf>,
    cancellation: &CancellationToken,
) -> Result<CertifiedVideoCodecDependencyClosure, VideoCodecDependencyClosureError> {
    certify_video_codec_dependency_closure_with_limits(
        primary,
        contract,
        dependency_paths,
        MAX_VIDEO_CODEC_DEPENDENCY_IMAGES,
        MAX_VIDEO_CODEC_DEPENDENCY_RETAINED_BYTES,
        cancellation,
    )
}

fn certify_video_codec_dependency_closure_with_limits(
    primary: CertifiedInspectedVideoCodecPackage,
    contract: VerifiedVideoCodecDependencyContract,
    dependency_paths: BTreeMap<String, PathBuf>,
    maximum_dependency_images: usize,
    maximum_retained_bytes: u64,
    cancellation: &CancellationToken,
) -> Result<CertifiedVideoCodecDependencyClosure, VideoCodecDependencyClosureError> {
    cancellation
        .check()
        .map_err(|_| VideoCodecDependencyClosureError::Cancelled)?;
    if contract.target() != VIDEO_CODEC_DEPENDENCY_CONTRACT_TARGET
        || !cfg!(all(
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        ))
    {
        return Err(VideoCodecDependencyClosureError::UnsupportedTarget);
    }
    if primary.target() != contract.target()
        || primary.catalog_sha256() != contract.primary_catalog_sha256()
        || primary.libraries().len() != primary.elf_libraries().len()
        || primary.libraries().len() != primary.certificates().len()
    {
        return Err(VideoCodecDependencyClosureError::ContractMismatch);
    }
    if contract.dependencies().len() > maximum_dependency_images {
        return Err(VideoCodecDependencyClosureError::ResourceLimitExceeded);
    }
    if dependency_paths.len() != contract.dependencies().len()
        || !contract
            .dependencies()
            .keys()
            .all(|identity| dependency_paths.contains_key(identity))
    {
        return Err(VideoCodecDependencyClosureError::IncompletePathSet);
    }

    for (identity, library) in primary.libraries() {
        cancellation
            .check()
            .map_err(|_| VideoCodecDependencyClosureError::Cancelled)?;
        let elf = primary
            .elf_libraries()
            .get(identity)
            .ok_or(VideoCodecDependencyClosureError::ContractMismatch)?;
        let expected_certificate = contract._registry.authorize(
            identity,
            library.digest_sha256(),
            &video_codec_abi_version(library.abi_major()),
            &elf.callable_symbols().keys().cloned().collect(),
        )?;
        if primary.certificates().get(identity) != Some(&expected_certificate) {
            return Err(VideoCodecDependencyClosureError::ContractMismatch);
        }
    }

    let mut soname_identities = BTreeMap::new();
    for (identity, library) in primary.libraries() {
        if soname_identities
            .insert(library.filename().to_owned(), identity.clone())
            .is_some()
        {
            return Err(VideoCodecDependencyClosureError::ContractMismatch);
        }
    }
    for (identity, dependency) in contract.dependencies() {
        if contract.system_libraries().contains(dependency.filename())
            || soname_identities
                .insert(dependency.filename().to_owned(), identity.clone())
                .is_some()
        {
            return Err(VideoCodecDependencyClosureError::ContractMismatch);
        }
    }

    let mut dependency_elf_libraries = BTreeMap::new();
    let mut sealed_dependency_images = BTreeMap::new();
    let mut retained_dependency_bytes = 0_u64;
    for (identity, expected) in contract.dependencies() {
        cancellation
            .check()
            .map_err(|_| VideoCodecDependencyClosureError::Cancelled)?;
        let path = dependency_paths
            .get(identity)
            .ok_or(VideoCodecDependencyClosureError::IncompletePathSet)?;
        if path.file_name().and_then(|name| name.to_str()) != Some(expected.filename()) {
            return Err(VideoCodecDependencyClosureError::ContractMismatch);
        }
        let captured = capture_native_library_image(path, cancellation)
            .map_err(|error| map_video_codec_dependency_image_error(identity, error))?;
        if captured.digest_sha256() != expected.digest_sha256() {
            return Err(VideoCodecDependencyClosureError::ContractMismatch);
        }
        retained_dependency_bytes = retained_dependency_bytes
            .checked_add(
                u64::try_from(captured.bytes().len())
                    .map_err(|_| VideoCodecDependencyClosureError::ResourceLimitExceeded)?,
            )
            .ok_or(VideoCodecDependencyClosureError::ResourceLimitExceeded)?;
        if retained_dependency_bytes > maximum_retained_bytes {
            return Err(VideoCodecDependencyClosureError::ResourceLimitExceeded);
        }
        let dynamic = inspect_elf64_dynamic_contract(captured.bytes(), 62, cancellation).map_err(
            |error| match error {
                NativeElfInspectionError::Cancelled(_) => {
                    VideoCodecDependencyClosureError::Cancelled
                }
                NativeElfInspectionError::Invalid(reason) => {
                    VideoCodecDependencyClosureError::InvalidElf {
                        identity: identity.clone(),
                        reason,
                    }
                }
            },
        )?;
        if dynamic.soname() != Some(expected.filename()) {
            return Err(VideoCodecDependencyClosureError::ContractMismatch);
        }
        dependency_elf_libraries.insert(
            identity.clone(),
            VideoCodecElfLibraryIdentity {
                soname: expected.filename().to_owned(),
                exported_symbols: dynamic.symbols().clone(),
                callable_symbols: BTreeMap::new(),
                needed_libraries: dynamic.needed().clone(),
            },
        );
        let retained = captured
            .seal(&format!("video-codec-dependency-{identity}"), cancellation)
            .map_err(|error| map_video_codec_dependency_image_error(identity, error))?;
        sealed_dependency_images.insert(identity.clone(), retained);
    }

    let mut actual_edges = BTreeSet::new();
    for (consumer, elf) in primary
        .elf_libraries()
        .iter()
        .chain(dependency_elf_libraries.iter())
    {
        cancellation
            .check()
            .map_err(|_| VideoCodecDependencyClosureError::Cancelled)?;
        for needed in elf.needed_libraries() {
            let dependency = if let Some(identity) = soname_identities.get(needed) {
                identity.clone()
            } else if contract.system_libraries().contains(needed) {
                needed.clone()
            } else {
                return Err(VideoCodecDependencyClosureError::UnaccountedDependency {
                    consumer: consumer.clone(),
                    soname: needed.clone(),
                });
            };
            actual_edges.insert(VideoCodecDependencyEdge {
                consumer: consumer.clone(),
                dependency,
            });
        }
    }
    if &actual_edges != contract.edges() {
        return Err(VideoCodecDependencyClosureError::ContractMismatch);
    }

    let dependency_first_order = video_codec_dependency_first_order(
        primary.libraries().keys(),
        contract.dependencies().keys(),
        contract.edges(),
        cancellation,
    )?;
    let mut dependency_certificates = BTreeMap::new();
    for identity in &dependency_first_order {
        let Some(expected) = contract.dependencies().get(identity) else {
            continue;
        };
        cancellation
            .check()
            .map_err(|_| VideoCodecDependencyClosureError::Cancelled)?;
        let certificate = contract._registry.authorize_dependency(
            identity,
            expected.digest_sha256(),
            expected.abi_version(),
            expected.certificate_sponsor(),
        )?;
        if certificate.library_id() != identity
            || certificate.digest_sha256() != expected.digest_sha256()
            || certificate.abi_version() != expected.abi_version()
            || !certificate.required_symbols().is_empty()
            || certificate.unsafe_owner() != VIDEO_CODEC_FFI_UNSAFE_OWNER
        {
            return Err(VideoCodecDependencyClosureError::ContractMismatch);
        }
        dependency_certificates.insert(identity.clone(), certificate);
    }
    if dependency_certificates.len() != contract.dependencies().len() {
        return Err(VideoCodecDependencyClosureError::ContractMismatch);
    }
    cancellation
        .check()
        .map_err(|_| VideoCodecDependencyClosureError::Cancelled)?;
    Ok(CertifiedVideoCodecDependencyClosure {
        primary,
        contract,
        dependency_elf_libraries,
        dependency_certificates,
        dependency_first_order,
        retained_dependency_bytes,
        _sealed_dependency_images: sealed_dependency_images,
    })
}

fn video_codec_dependency_first_order<'a>(
    primary_identities: impl Iterator<Item = &'a String>,
    dependency_identities: impl Iterator<Item = &'a String>,
    edges: &BTreeSet<VideoCodecDependencyEdge>,
    cancellation: &CancellationToken,
) -> Result<Vec<String>, VideoCodecDependencyClosureError> {
    let package_identities = primary_identities
        .chain(dependency_identities)
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut remaining = package_identities
        .iter()
        .map(|identity| {
            let dependencies = edges
                .iter()
                .filter(|edge| {
                    edge.consumer() == identity && package_identities.contains(edge.dependency())
                })
                .map(|edge| edge.dependency().to_owned())
                .collect::<BTreeSet<_>>();
            (identity.clone(), dependencies)
        })
        .collect::<BTreeMap<_, _>>();
    let mut order = Vec::with_capacity(package_identities.len());
    while !remaining.is_empty() {
        cancellation
            .check()
            .map_err(|_| VideoCodecDependencyClosureError::Cancelled)?;
        let ready = remaining
            .iter()
            .find(|(_, dependencies)| {
                dependencies
                    .iter()
                    .all(|dependency| !remaining.contains_key(dependency))
            })
            .map(|(identity, _)| identity.clone())
            .ok_or(VideoCodecDependencyClosureError::InvalidGraph)?;
        remaining.remove(&ready);
        order.push(ready);
    }
    Ok(order)
}

fn map_video_codec_dependency_image_error(
    identity: &str,
    error: NativeLibraryImageError,
) -> VideoCodecDependencyClosureError {
    match error {
        NativeLibraryImageError::Cancelled => VideoCodecDependencyClosureError::Cancelled,
        NativeLibraryImageError::UnsupportedPlatform => {
            VideoCodecDependencyClosureError::UnsupportedTarget
        }
        NativeLibraryImageError::Invalid(reason) => {
            VideoCodecDependencyClosureError::InvalidImage {
                identity: identity.to_owned(),
                reason,
            }
        }
    }
}

fn validate_video_codec_catalog_envelope(
    catalog: &VideoCodecFfiCatalogDto,
) -> Result<(), VideoCodecFfiCatalogError> {
    if catalog.schema_version != 1
        || catalog.profile != VIDEO_CODEC_FFI_PROFILE
        || !VIDEO_CODEC_FFI_TARGETS.contains(&catalog.target.as_str())
        || !valid_ascii_identifier(&catalog.signer, 256)
        || catalog.signature_algorithm != NATIVE_PACKAGE_SIGNATURE_ALGORITHM
        || catalog.signature_domain != "zed-sim-video-codec-package-v1"
        || catalog.certificate_owner != "sim_runtime::NativeFfiRegistry"
        || catalog.unsafe_owner != VIDEO_CODEC_FFI_UNSAFE_OWNER
        || !catalog.runtime_compilation_forbidden
        || catalog.redistributes_codec_libraries
        || validate_sha256(&catalog.license_notice_sha256).is_err()
        || catalog.external_encoders != VIDEO_CODEC_EXTERNAL_ENCODERS.map(str::to_owned)
        || catalog.libraries.len() != video_codec_library_contracts().len()
    {
        return Err(VideoCodecFfiCatalogError::ContractMismatch);
    }
    for (row, (identity, abi_major, symbols)) in catalog
        .libraries
        .iter()
        .zip(video_codec_library_contracts())
    {
        if row.identity != identity
            || row.abi_major != abi_major
            || row.filename != video_codec_expected_filename(identity, abi_major, &catalog.target)
            || validate_sha256(&row.sha256).is_err()
            || row.required_symbols
                != symbols
                    .iter()
                    .map(|symbol| (*symbol).to_owned())
                    .collect::<Vec<_>>()
        {
            return Err(VideoCodecFfiCatalogError::ContractMismatch);
        }
    }
    Ok(())
}

fn validate_video_codec_dependency_contract_envelope(
    primary: &VerifiedVideoCodecFfiCatalog,
    contract: &VideoCodecDependencyContractDto,
    cancellation: &CancellationToken,
) -> Result<(), VideoCodecDependencyContractError> {
    let reviewed_system_libraries = VIDEO_CODEC_SYSTEM_ELF_DEPENDENCIES
        .map(str::to_owned)
        .to_vec();
    if contract.schema_version != 1
        || contract.profile != VIDEO_CODEC_FFI_PROFILE
        || contract.target != VIDEO_CODEC_DEPENDENCY_CONTRACT_TARGET
        || contract.target != primary.target()
        || !valid_ascii_identifier(&contract.signer, 256)
        || contract.signature_algorithm != NATIVE_PACKAGE_SIGNATURE_ALGORITHM
        || contract.signature_domain != "zed-sim-video-codec-dependency-contract-v1"
        || contract.certificate_owner != "sim_runtime::NativeFfiRegistry"
        || contract.unsafe_owner != VIDEO_CODEC_FFI_UNSAFE_OWNER
        || !contract.runtime_compilation_forbidden
        || contract.redistributes_codec_libraries
        || contract.primary_catalog_sha256 != primary.catalog_sha256()
        || validate_sha256(&contract.source_archive_sha256).is_err()
        || validate_sha256(&contract.build_recipe_sha256).is_err()
        || validate_sha256(&contract.license_bundle_sha256).is_err()
        || contract.system_libraries != reviewed_system_libraries
        || contract.dependencies.len() > MAX_VIDEO_CODEC_DEPENDENCY_IMAGES
        || contract.edges.len() > MAX_VIDEO_CODEC_DEPENDENCY_EDGES
    {
        return Err(VideoCodecDependencyContractError::ContractMismatch);
    }

    let dependency_keys = contract
        .dependencies
        .iter()
        .map(|dependency| dependency.identity.as_str())
        .collect::<Vec<_>>();
    if dependency_keys.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(VideoCodecDependencyContractError::ContractMismatch);
    }
    let edge_keys = contract
        .edges
        .iter()
        .map(|edge| (edge.consumer.as_str(), edge.dependency.as_str()))
        .collect::<Vec<_>>();
    if edge_keys.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(VideoCodecDependencyContractError::ContractMismatch);
    }
    let provider_keys = contract
        .encoder_providers
        .iter()
        .map(|provider| provider.encoder.as_str())
        .collect::<Vec<_>>();
    if provider_keys != VIDEO_CODEC_EXTERNAL_ENCODERS {
        return Err(VideoCodecDependencyContractError::ContractMismatch);
    }

    let primary_identities = primary.libraries().keys().cloned().collect::<BTreeSet<_>>();
    let dependency_identities = contract
        .dependencies
        .iter()
        .map(|dependency| dependency.identity.clone())
        .collect::<BTreeSet<_>>();
    let all_identities = primary_identities
        .union(&dependency_identities)
        .cloned()
        .collect::<BTreeSet<_>>();
    let primary_filenames = primary
        .libraries()
        .values()
        .map(|library| library.filename().to_owned())
        .collect::<BTreeSet<_>>();
    let mut dependency_filenames = BTreeSet::new();
    for dependency in &contract.dependencies {
        cancellation
            .check()
            .map_err(|_| VideoCodecDependencyContractError::Cancelled)?;
        if !valid_ascii_identifier(&dependency.identity, 255)
            || primary_identities.contains(&dependency.identity)
            || !video_codec_filename_valid(&dependency.filename)
            || primary_filenames.contains(&dependency.filename)
            || VIDEO_CODEC_SYSTEM_ELF_DEPENDENCIES.contains(&dependency.filename.as_str())
            || !dependency_filenames.insert(dependency.filename.clone())
            || validate_sha256(&dependency.sha256).is_err()
            || !video_codec_dependency_abi_valid(&dependency.abi_version)
            || !all_identities.contains(&dependency.certificate_sponsor)
            || dependency.certificate_sponsor == dependency.identity
        {
            return Err(VideoCodecDependencyContractError::ContractMismatch);
        }
    }

    let reviewed_system_set = contract
        .system_libraries
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let edges = contract
        .edges
        .iter()
        .map(|edge| (edge.consumer.clone(), edge.dependency.clone()))
        .collect::<BTreeSet<_>>();
    let mut package_dependencies = all_identities
        .iter()
        .map(|identity| (identity.clone(), BTreeSet::new()))
        .collect::<BTreeMap<_, _>>();
    for edge in &contract.edges {
        cancellation
            .check()
            .map_err(|_| VideoCodecDependencyContractError::Cancelled)?;
        if !all_identities.contains(&edge.consumer)
            || edge.consumer == edge.dependency
            || !(all_identities.contains(&edge.dependency)
                || reviewed_system_set.contains(&edge.dependency))
        {
            return Err(VideoCodecDependencyContractError::ContractMismatch);
        }
        if all_identities.contains(&edge.dependency) {
            package_dependencies
                .get_mut(&edge.consumer)
                .ok_or(VideoCodecDependencyContractError::ContractMismatch)?
                .insert(edge.dependency.clone());
        }
    }
    for dependency in &contract.dependencies {
        if !edges.contains(&(
            dependency.certificate_sponsor.clone(),
            dependency.identity.clone(),
        )) || !contract
            .edges
            .iter()
            .any(|edge| edge.dependency == dependency.identity)
        {
            return Err(VideoCodecDependencyContractError::ContractMismatch);
        }
    }
    for provider in &contract.encoder_providers {
        let provider_is_valid = match provider.encoder.as_str() {
            "aac" => provider.provider == "avcodec",
            _ => dependency_identities.contains(&provider.provider),
        };
        if !provider_is_valid {
            return Err(VideoCodecDependencyContractError::ContractMismatch);
        }
    }

    let mut reachable = primary_identities.clone();
    let mut pending = primary_identities.iter().cloned().collect::<Vec<_>>();
    while let Some(consumer) = pending.pop() {
        cancellation
            .check()
            .map_err(|_| VideoCodecDependencyContractError::Cancelled)?;
        if let Some(dependencies) = package_dependencies.get(&consumer) {
            for dependency in dependencies {
                if reachable.insert(dependency.clone()) {
                    pending.push(dependency.clone());
                }
            }
        }
    }
    if !dependency_identities.is_subset(&reachable) {
        return Err(VideoCodecDependencyContractError::ContractMismatch);
    }

    let mut remaining = package_dependencies;
    while !remaining.is_empty() {
        cancellation
            .check()
            .map_err(|_| VideoCodecDependencyContractError::Cancelled)?;
        let ready = remaining
            .iter()
            .find(|(_, dependencies)| {
                dependencies
                    .iter()
                    .all(|dependency| !remaining.contains_key(dependency))
            })
            .map(|(identity, _)| identity.clone())
            .ok_or(VideoCodecDependencyContractError::ContractMismatch)?;
        remaining.remove(&ready);
    }
    Ok(())
}

fn video_codec_expected_filename(identity: &str, major: u16, target: &str) -> String {
    if target.ends_with("windows-msvc") {
        format!("{identity}-{major}.dll")
    } else if target.ends_with("apple-darwin") {
        format!("lib{identity}.{major}.dylib")
    } else {
        format!("lib{identity}.so.{major}")
    }
}

fn video_codec_abi_version(major: u16) -> String {
    format!("{VIDEO_CODEC_FFI_PROFILE}:{major}")
}

fn video_codec_library_identity_valid(value: &str) -> bool {
    video_codec_library_contracts()
        .into_iter()
        .any(|(identity, _, _)| identity == value)
}

fn video_codec_filename_valid(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn video_codec_dependency_abi_valid(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':'))
}

fn video_codec_symbol_valid(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

pub fn cudart_exact_native<'a>(
    certification: &'a CertifiedNativeFfi,
    cancellation: &CancellationToken,
) -> Result<&'a CertifiedNativeFfi, TrustError> {
    cancellation.check().map_err(|_| TrustError::Cancelled)?;
    if certification.library_id() != CUDART_LIBRARY_ID
        || !certification
            .required_symbols()
            .contains("cudaHostRegister")
        || !certification
            .required_symbols()
            .contains("cudaHostUnregister")
    {
        return Err(TrustError::UncertifiedFfi);
    }
    cancellation.check().map_err(|_| TrustError::Cancelled)?;
    Ok(certification)
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TrustError {
    #[error("trust-policy operation was cancelled")]
    Cancelled,
    #[error("plugin signature is missing or invalid")]
    InvalidPluginSignature,
    #[error("plugin verification key is invalid")]
    InvalidVerificationKey,
    #[error("plugin verification key is duplicated: {0}")]
    DuplicateVerificationKey(String),
    #[error("plugin verification key is not trusted: {0}")]
    UnknownVerificationKey(String),
    #[error("ROCm package verification key is invalid")]
    InvalidRocmPackageVerificationKey,
    #[error("ROCm package signer is not the explicitly configured authority")]
    UnknownRocmPackageSigner,
    #[error("ROCm package signature is missing, malformed, or invalid")]
    InvalidRocmPackageSignature,
    #[error("Metal package verification key is invalid")]
    InvalidMetalPackageVerificationKey,
    #[error("Metal package signer is not the explicitly configured authority")]
    UnknownMetalPackageSigner,
    #[error("Metal package signature is missing, malformed, or invalid")]
    InvalidMetalPackageSignature,
    #[error("MLU package verification key is invalid")]
    InvalidMluPackageVerificationKey,
    #[error("MLU package signer is not the explicitly configured authority")]
    UnknownMluPackageSigner,
    #[error("MLU package signature is missing, malformed, or invalid")]
    InvalidMluPackageSignature,
    #[error("NPU package verification key is invalid")]
    InvalidNpuPackageVerificationKey,
    #[error("NPU package signer is not the explicitly configured authority")]
    UnknownNpuPackageSigner,
    #[error("NPU package signature is missing, malformed, or invalid")]
    InvalidNpuPackageSignature,
    #[error("XPU package verification key is invalid")]
    InvalidXpuPackageVerificationKey,
    #[error("XPU package signer is not the explicitly configured authority")]
    UnknownXpuPackageSigner,
    #[error("XPU package signature is missing, malformed, or invalid")]
    InvalidXpuPackageSignature,
    #[error("CUDA package verification key is invalid")]
    InvalidCudaPackageVerificationKey,
    #[error("CUDA package signer is not the explicitly configured authority")]
    UnknownCudaPackageSigner,
    #[error("CUDA package signature is missing, malformed, or invalid")]
    InvalidCudaPackageSignature,
    #[error("DirectML package verification key is invalid")]
    InvalidDirectMlPackageVerificationKey,
    #[error("DirectML package signer is not the explicitly configured authority")]
    UnknownDirectMlPackageSigner,
    #[error("DirectML package signature is missing, malformed, or invalid")]
    InvalidDirectMlPackageSignature,
    #[error("video codec package verification key is invalid")]
    InvalidVideoCodecPackageVerificationKey,
    #[error("video codec package signer is not the explicitly configured authority")]
    UnknownVideoCodecPackageSigner,
    #[error("video codec package signature is missing, malformed, or invalid")]
    InvalidVideoCodecPackageSignature,
    #[error("general video codec package verification key is invalid")]
    InvalidGeneralVideoCodecPackageVerificationKey,
    #[error("general video codec package signer is not the explicitly configured authority")]
    UnknownGeneralVideoCodecPackageSigner,
    #[error("general video codec package signature is missing, malformed, or invalid")]
    InvalidGeneralVideoCodecPackageSignature,
    #[error("plugin authorization does not belong to this manifest")]
    AuthorizationManifestMismatch,
    #[error("sealed plugin authorization is malformed or internally inconsistent")]
    InvalidSealedPluginAuthorization,
    #[error("sealed plugin authorization exceeds its bounded wire size")]
    SealedPluginAuthorizationTooLarge,
    #[error("plugin authorization sealing key generation failed")]
    AuthorizationSealingUnavailable,
    #[error("plugin authorization verification key is invalid")]
    InvalidAuthorizationVerificationKey,
    #[error(transparent)]
    PluginContract(#[from] PluginContractError),
    #[error(transparent)]
    Permission(#[from] PermissionError),
    #[error("secret identifier is invalid")]
    InvalidSecretId,
    #[error("provider identifier is invalid")]
    InvalidProviderId,
    #[error("provider credential scope is invalid")]
    InvalidCredentialScope,
    #[error("provider profile is invalid")]
    InvalidProviderProfile,
    #[error("provider endpoint is invalid")]
    InvalidProviderEndpoint,
    #[error("provider is disabled")]
    ProviderDisabled,
    #[error("provider request is forbidden while offline")]
    Offline,
    #[error("provider request belongs to a different profile")]
    ProviderProfileMismatch,
    #[error("provider endpoint is not explicitly allowed")]
    ProviderEndpointDenied,
    #[error("provider secret grant is missing")]
    MissingSecretGrant,
    #[error("provider price bound is invalid")]
    InvalidProviderPriceBound,
    #[error("provider cost acceptance is invalid or does not match the request")]
    InvalidProviderCostAcceptance,
    #[error("provider cost acceptance has expired")]
    ExpiredProviderCostAcceptance,
    #[error("provider cost acceptance sealing key generation failed")]
    ProviderCostAcceptanceSealingUnavailable,
    #[error("provider invocation identity is invalid")]
    InvalidProviderInvocationIdentity,
    #[error("provider result receipt is invalid or does not match the invocation")]
    InvalidProviderResultReceipt,
    #[error("provider runtime receipt is invalid, expired, or bound to another stream")]
    InvalidProviderRuntimeReceipt,
    #[error("provider result receipt has expired")]
    ExpiredProviderResultReceipt,
    #[error("provider result receipt exceeds its bounded wire size")]
    ProviderResultReceiptTooLarge,
    #[error("provider result receipt sealing key generation failed")]
    ProviderResultReceiptSealingUnavailable,
    #[error("native API remote exposure is unsafe")]
    UnsafeApiExposure,
    #[error("external navigation policy is invalid")]
    InvalidNavigationPolicy,
    #[error("external navigation is not permitted")]
    NavigationDenied,
    #[error("native FFI contract is invalid")]
    InvalidFfiContract,
    #[error("native FFI contract is duplicated: {0}")]
    DuplicateFfiContract(String),
    #[error("native FFI library is not certified")]
    UncertifiedFfi,
}

fn validate_sha256(value: &str) -> Result<(), ()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(())
    }
}

fn valid_ascii_identifier(value: &str, maximum_bytes: usize) -> bool {
    if value.is_empty() || value.len() > maximum_bytes || !value.is_ascii() {
        return false;
    }
    let bytes = value.as_bytes();
    if !bytes.first().is_some_and(u8::is_ascii_alphanumeric)
        || !bytes.last().is_some_and(u8::is_ascii_alphanumeric)
    {
        return false;
    }
    let mut previous_separator = false;
    for byte in bytes {
        if byte.is_ascii_lowercase() || byte.is_ascii_digit() {
            previous_separator = false;
        } else if matches!(byte, b'.' | b'-' | b'_') && !previous_separator {
            previous_separator = true;
        } else {
            return false;
        }
    }
    true
}

fn valid_ascii_node_identifier(value: &str, maximum_bytes: usize) -> bool {
    if value.is_empty() || value.len() > maximum_bytes || !value.is_ascii() {
        return false;
    }
    let bytes = value.as_bytes();
    if !bytes.first().is_some_and(u8::is_ascii_alphanumeric)
        || !bytes.last().is_some_and(u8::is_ascii_alphanumeric)
    {
        return false;
    }
    let mut previous_separator = false;
    for byte in bytes {
        if byte.is_ascii_alphanumeric() {
            previous_separator = false;
        } else if matches!(byte, b'.' | b'-' | b'_') && !previous_separator {
            previous_separator = true;
        } else {
            return false;
        }
    }
    true
}

fn canonical_origin(origin: &str) -> Result<String, TrustError> {
    if origin.bytes().any(|byte| byte.is_ascii_control()) || origin.contains('*') {
        return Err(TrustError::UnsafeApiExposure);
    }
    let parsed = Url::parse(origin).map_err(|_| TrustError::UnsafeApiExposure)?;
    let host = parsed.host().ok_or(TrustError::UnsafeApiExposure)?;
    let insecure_loopback = parsed.scheme() == "http"
        && match host {
            Host::Domain(domain) => domain.eq_ignore_ascii_case("localhost"),
            Host::Ipv4(address) => address.is_loopback(),
            Host::Ipv6(address) => address.is_loopback(),
        };
    if (parsed.scheme() != "https" && !insecure_loopback)
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.path() != "/"
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(TrustError::UnsafeApiExposure);
    }
    Ok(parsed.origin().ascii_serialization())
}

fn valid_navigation_scheme(scheme: &str) -> bool {
    let mut bytes = scheme.bytes();
    bytes.next().is_some_and(|byte| byte.is_ascii_alphabetic())
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.'))
        && !matches!(scheme, "data" | "file" | "javascript" | "vbscript")
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

fn decode_hex_exact<const LENGTH: usize>(value: &str) -> Option<[u8; LENGTH]> {
    if value.len() != LENGTH * 2 {
        return None;
    }
    let mut decoded = [0_u8; LENGTH];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = decode_hex_nibble(pair[0])?;
        let low = decode_hex_nibble(pair[1])?;
        decoded[index] = (high << 4) | low;
    }
    Some(decoded)
}

fn decode_hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

pub(crate) fn parse_strict_json_value(bytes: &[u8]) -> Result<serde_json::Value, String> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let strict =
        StrictJsonValue::deserialize(&mut deserializer).map_err(|error| error.to_string())?;
    deserializer.end().map_err(|error| error.to_string())?;
    Ok(strict.0)
}

struct StrictJsonValue(serde_json::Value);

impl<'de> Deserialize<'de> for StrictJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct Visitor;

        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = StrictJsonValue;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON value without duplicate object keys")
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(serde_json::Value::Bool(value)))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(serde_json::Value::Number(value.into())))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(serde_json::Value::Number(value.into())))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                serde_json::Number::from_f64(value)
                    .map(serde_json::Value::Number)
                    .map(StrictJsonValue)
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(serde_json::Value::String(value.to_owned())))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(serde_json::Value::String(value)))
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(serde_json::Value::Null))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(serde_json::Value::Null))
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element::<StrictJsonValue>()? {
                    values.push(value.0);
                }
                Ok(StrictJsonValue(serde_json::Value::Array(values)))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut values = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate object key: {key}"
                        )));
                    }
                    let value = map.next_value::<StrictJsonValue>()?;
                    values.insert(key, value.0);
                }
                Ok(StrictJsonValue(serde_json::Value::Object(values)))
            }
        }

        deserializer.deserialize_any(Visitor)
    }
}

fn authorization_signing_payload(payload: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(PLUGIN_AUTHORIZATION_SIGNATURE_DOMAIN);
    digest.update(
        u64::try_from(payload.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    digest.update(payload);
    digest.finalize().into()
}

fn provider_cost_milliseconds(origin: Instant, value: Instant) -> Result<u64, TrustError> {
    let duration = value
        .checked_duration_since(origin)
        .ok_or(TrustError::InvalidProviderCostAcceptance)?;
    u64::try_from(duration.as_millis()).map_err(|_| TrustError::InvalidProviderCostAcceptance)
}

fn provider_cost_acceptance_signing_payload(claims: &ProviderCostAcceptanceClaims) -> [u8; 32] {
    fn update_field(digest: &mut Sha256, value: &[u8]) {
        digest.update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_le_bytes());
        digest.update(value);
    }

    let mut digest = Sha256::new();
    digest.update(PROVIDER_COST_ACCEPTANCE_SIGNATURE_DOMAIN);
    for field in [
        claims.scope.principal_id().as_bytes(),
        claims.scope.profile_id().as_bytes(),
        claims.scope.prompt_id().as_bytes(),
        claims.scope.prompt_sha256().as_bytes(),
        claims.scope.attempt_id().as_bytes(),
        claims.scope.node_id().as_bytes(),
        claims.scope.request_sha256().as_bytes(),
        claims.scope.plugin_id().as_bytes(),
        claims.scope.plugin_digest_sha256().as_bytes(),
        claims.scope.provider_binding_sha256().as_bytes(),
        claims.scope.provider().as_bytes(),
        claims.scope.endpoint().as_bytes(),
        claims.scope.price_bound().currency_code().as_bytes(),
    ] {
        update_field(&mut digest, field);
    }
    digest.update(claims.scope.request_ordinal().to_le_bytes());
    digest.update(
        claims
            .scope
            .price_bound()
            .maximum_microunits()
            .to_le_bytes(),
    );
    digest.update(claims.issued_at_milliseconds.to_le_bytes());
    digest.update(claims.expires_at_milliseconds.to_le_bytes());
    digest.update(claims.nonce.as_bytes());
    digest.finalize().into()
}

fn provider_result_milliseconds(origin: Instant, value: Instant) -> Result<u64, TrustError> {
    let duration = value
        .checked_duration_since(origin)
        .ok_or(TrustError::InvalidProviderResultReceipt)?;
    u64::try_from(duration.as_millis()).map_err(|_| TrustError::InvalidProviderResultReceipt)
}

fn provider_result_receipt_signing_payload(claims: &ProviderResultReceiptClaims) -> [u8; 32] {
    fn update_field(digest: &mut Sha256, value: &[u8]) {
        digest.update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_le_bytes());
        digest.update(value);
    }

    let identity = &claims.identity;
    let mut digest = Sha256::new();
    digest.update(PROVIDER_RESULT_RECEIPT_SIGNATURE_DOMAIN);
    for field in [
        identity.principal_id().as_bytes(),
        identity.profile_id().as_bytes(),
        identity.prompt_id().as_bytes(),
        identity.prompt_sha256().as_bytes(),
        identity.attempt_id().as_bytes(),
        identity.node_id().as_bytes(),
        identity.request_sha256().as_bytes(),
        identity.plugin_id().as_bytes(),
        identity.plugin_digest_sha256().as_bytes(),
        identity.provider_binding_sha256().as_bytes(),
        identity.provider().as_bytes(),
        identity.endpoint().as_bytes(),
        claims.result_sha256.as_bytes(),
    ] {
        update_field(&mut digest, field);
    }
    digest.update(identity.request_ordinal().to_le_bytes());
    digest.update(claims.issued_at_milliseconds.to_le_bytes());
    digest.update(claims.expires_at_milliseconds.to_le_bytes());
    digest.update(claims.nonce.as_bytes());
    digest.finalize().into()
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    result
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        fs, io,
        net::Ipv4Addr,
        path::{Path, PathBuf},
        str::FromStr,
    };

    use serde_json::json;
    use sim_model::{ParserLimits, RestrictedPickleError, parse_restricted_pickle};
    use sim_plugin_sdk::{
        ApiRequirement, ApiVersion, CachePolicy, CapabilityKind, CapabilityQuota,
        CapabilityRequest, DeterminismPolicy, EffectPolicy, ManifestProvenance, ManifestSignature,
        PROVIDER_BINDING_API_FEATURE, PROVIDER_BINDING_SCHEMA_VERSION, PROVIDER_COMPONENT_WORLD_V2,
        PROVIDER_MANIFEST_SCHEMA_VERSION_V2, PROVIDER_STREAMING_API_FEATURE_V2, PluginManifest,
        PluginNode, PluginSigningKey, ProviderBindingClaim, ProviderBindingSet,
        ProviderHttpMethodV2, ProviderPluginManifestV2, ProviderStreamingContractV2,
    };

    use crate::{Capability, PermissionGrant};

    use super::*;

    const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const SIGNING_KEY: &[u8; 32] = b"0123456789abcdef0123456789abcdef";

    const PACKAGE_LIMITS: [NativePackagePayloadLimit; 4] = [
        NativePackagePayloadLimit::new("a.txt", 16),
        NativePackagePayloadLimit::new("nested/b.bin", 16),
        NativePackagePayloadLimit::new("package-coverage.sha256", 1024),
        NativePackagePayloadLimit::new("signature.sig", 1024),
    ];

    #[test]
    fn native_library_image_capture_owns_bounds_identity_digest_and_cancellation()
    -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            NativeLibraryImageError::UnsupportedPlatform.to_string(),
            "native-library image capture or sealing is unsupported on this platform"
        );
        let directory = tempfile::tempdir()?;
        let source = directory.path().join("libfixture.so");
        fs::write(&source, b"data")?;
        let cancellation = CancellationToken::default();
        let captured =
            capture_native_library_image_with_limit(&source, 4, &mut || cancellation.check())?;
        assert_eq!(captured.bytes(), b"data");
        assert_eq!(
            captured.digest_sha256(),
            format!("{:x}", Sha256::digest(b"data"))
        );

        fs::write(&source, b"")?;
        assert!(matches!(
            capture_native_library_image_with_limit(&source, 4, &mut || Ok(())),
            Err(NativeLibraryImageError::Invalid(reason)) if reason.contains("nonempty regular file")
        ));
        fs::write(&source, b"overs")?;
        assert!(matches!(
            capture_native_library_image_with_limit(&source, 4, &mut || Ok(())),
            Err(NativeLibraryImageError::Invalid(reason)) if reason.contains("no larger than 4")
        ));
        assert!(matches!(
            capture_native_library_image_with_limit(directory.path(), 4, &mut || Ok(())),
            Err(NativeLibraryImageError::Invalid(reason)) if reason.contains("regular file")
        ));

        fs::write(&source, vec![7_u8; NATIVE_LIBRARY_IMAGE_CHUNK_BYTES * 2])?;
        let cancellation = CancellationToken::default();
        let checks = std::cell::Cell::new(0_usize);
        assert!(matches!(
            capture_native_library_image_with_limit(
                &source,
                u64::try_from(NATIVE_LIBRARY_IMAGE_CHUNK_BYTES * 2)?,
                &mut || {
                    let current = checks.get();
                    checks.set(current + 1);
                    if current == 1 {
                        cancellation.cancel();
                    }
                    cancellation.check()
                },
            ),
            Err(NativeLibraryImageError::Cancelled)
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn native_library_image_capture_rejects_symlinks_and_concurrent_mutation()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir()?;
        let source = directory.path().join("libfixture.so");
        let link = directory.path().join("libfixture-link.so");
        fs::write(&source, vec![1_u8; NATIVE_LIBRARY_IMAGE_CHUNK_BYTES * 2])?;
        symlink(&source, &link)?;
        assert!(matches!(
            capture_native_library_image(&link, &CancellationToken::default()),
            Err(NativeLibraryImageError::Invalid(_))
        ));

        let checks = std::cell::Cell::new(0_usize);
        assert!(matches!(
            capture_native_library_image_with_limit(
                &source,
                u64::try_from(NATIVE_LIBRARY_IMAGE_CHUNK_BYTES * 3)?,
                &mut || {
                    let current = checks.get();
                    checks.set(current + 1);
                    if current == 2 {
                        fs::write(&source, b"changed length").map_err(|_| CancellationError)?;
                    }
                    Ok(())
                },
            ),
            Err(NativeLibraryImageError::Invalid(reason))
                if reason.contains("changed while it was captured")
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn native_library_image_capture_rejects_growth_shrink_and_in_place_mutation()
    -> Result<(), Box<dyn std::error::Error>> {
        fn capture_while_mutating(
            path: &Path,
            mutate: impl FnOnce(&Path) -> io::Result<()>,
        ) -> Result<CapturedNativeLibraryImage, NativeLibraryImageError> {
            let checks = std::cell::Cell::new(0_usize);
            let mut mutate = Some(mutate);
            capture_native_library_image_with_limit(
                path,
                u64::try_from(NATIVE_LIBRARY_IMAGE_CHUNK_BYTES * 3).unwrap_or(u64::MAX),
                &mut || {
                    let current = checks.get();
                    checks.set(current + 1);
                    if current == 2
                        && let Some(mutate) = mutate.take()
                    {
                        mutate(path).map_err(|_| CancellationError)?;
                    }
                    Ok(())
                },
            )
        }

        let directory = tempfile::tempdir()?;
        let source = directory.path().join("libfixture.so");
        let original = vec![11_u8; NATIVE_LIBRARY_IMAGE_CHUNK_BYTES * 2];

        fs::write(&source, &original)?;
        assert!(matches!(
            capture_while_mutating(&source, |path| {
                OpenOptions::new().append(true).open(path)?.write_all(b"growth")
            }),
            Err(NativeLibraryImageError::Invalid(reason))
                if reason.contains("changed while it was captured")
        ));

        fs::write(&source, &original)?;
        assert!(matches!(
            capture_while_mutating(&source, |path| {
                OpenOptions::new().write(true).open(path)?.set_len(1)
            }),
            Err(NativeLibraryImageError::Invalid(reason))
                if reason.contains("changed while it was captured")
        ));

        fs::write(&source, &original)?;
        assert!(matches!(
            capture_while_mutating(&source, |path| {
                let mut file = OpenOptions::new().write(true).open(path)?;
                file.seek(SeekFrom::Start(0))?;
                file.write_all(b"mutated")
            }),
            Err(NativeLibraryImageError::Invalid(reason))
                if reason.contains("changed while it was captured")
        ));
        Ok(())
    }

    #[test]
    fn retained_native_library_image_owns_loader_path_lifetime()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let source = directory.path().join("libfixture.so");
        fs::write(&source, b"immutable")?;
        let retained = capture_native_library_image(&source, &CancellationToken::default())?
            .seal("fixture", &CancellationToken::default())?;
        let loader_path = retained.loader_path().to_path_buf();
        assert_eq!(fs::read(&loader_path)?, b"immutable");
        assert_eq!(retained.file().metadata()?.len(), 9);

        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            use std::os::unix::fs::MetadataExt;

            let required =
                libc::F_SEAL_SEAL | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE;
            let actual = unsafe { libc::fcntl(retained.file().as_raw_fd(), libc::F_GET_SEALS) };
            assert_eq!(actual & required, required);
            assert_eq!(
                retained.file().metadata()?.ino(),
                fs::metadata(&loader_path)?.ino()
            );
            let mut writable = fs::OpenOptions::new().write(true).open(&loader_path)?;
            // F_SEAL_WRITE rejects writes, while reopening the descriptor can succeed.
            let error = writable
                .write_all(b"modified")
                .expect_err("sealed image accepted a write");
            assert_eq!(error.raw_os_error(), Some(libc::EPERM));
            assert_eq!(fs::read(&loader_path)?, b"immutable");
            assert!(retained.file().set_len(0).is_err());
        }

        drop(retained);
        assert!(!loader_path.exists());
        Ok(())
    }

    fn package_fixture_root()
    -> Result<(tempfile::TempDir, ArtifactRoot), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let root = ArtifactRoot::canonical(
            "native-package-admission-fixture",
            "native-ffi-package",
            directory.path(),
            std::iter::empty::<String>(),
        )?;
        root.write_private_file("a.txt", b"alpha")?;
        root.write_private_file("nested/b.bin", b"bravo")?;
        root.write_private_file("package-coverage.sha256", b"pending\n")?;
        root.write_private_file("signature.sig", b"pending\n")?;
        Ok((directory, root))
    }

    #[test]
    fn native_package_capture_owns_exact_bounded_stable_membership()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, root) = package_fixture_root()?;
        let cancellation = CancellationToken::default();
        let payloads = capture_native_package(&root, &PACKAGE_LIMITS, 8, 4096, &cancellation)?;
        assert_eq!(payloads.len(), PACKAGE_LIMITS.len());
        assert_eq!(
            payloads.get("a.txt").map(Vec::as_slice),
            Some(b"alpha".as_slice())
        );

        root.write_private_file("extra.bin", b"extra")?;
        assert!(matches!(
            capture_native_package(&root, &PACKAGE_LIMITS, 8, 4096, &cancellation),
            Err(NativePackageAdmissionError::UnsafePackage(message))
                if message.contains("membership")
        ));
        root.quarantine_private_file("extra.bin")?;
        root.quarantine_private_file("a.txt")?;
        assert!(matches!(
            capture_native_package(&root, &PACKAGE_LIMITS, 8, 4096, &cancellation),
            Err(NativePackageAdmissionError::UnsafePackage(message))
                if message.contains("membership")
        ));

        let (_directory, root) = package_fixture_root()?;
        let quarantined = root
            .quarantine_private_file("a.txt")?
            .ok_or("fixture payload was not quarantined")?;
        assert!(root.remove_contained_file(&quarantined)?);
        fs::create_dir(root.canonical_path().join("a.txt"))?;
        assert!(matches!(
            capture_native_package(&root, &PACKAGE_LIMITS, 8, 4096, &cancellation),
            Err(NativePackageAdmissionError::UnsafePackage(message))
                if message.contains("membership")
        ));
        Ok(())
    }

    #[test]
    fn native_package_capture_rejects_empty_oversized_replacement_and_cancellation()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, root) = package_fixture_root()?;
        let cancellation = CancellationToken::default();
        root.write_private_file("a.txt", b"")?;
        assert!(matches!(
            capture_native_package(&root, &PACKAGE_LIMITS, 8, 4096, &cancellation),
            Err(NativePackageAdmissionError::UnsafePackage(message))
                if message.contains("empty")
        ));

        root.write_private_file("a.txt", b"more-than-sixteen-bytes")?;
        assert!(matches!(
            capture_native_package(&root, &PACKAGE_LIMITS, 8, 4096, &cancellation),
            Err(NativePackageAdmissionError::UnsafePackage(_))
        ));

        root.write_private_file("a.txt", b"alpha")?;
        assert!(matches!(
            capture_native_package(&root, &PACKAGE_LIMITS, 8, 20, &cancellation),
            Err(NativePackageAdmissionError::UnsafePackage(message))
                if message.contains("aggregate bound")
        ));
        assert!(matches!(
            capture_native_package_with_hook(
                &root,
                &PACKAGE_LIMITS,
                8,
                4096,
                &cancellation,
                || {
                    root.write_private_file("a.txt", b"omega")
                        .map_err(|error| {
                            NativePackageAdmissionError::UnsafePackage(error.to_string())
                        })
                },
            ),
            Err(NativePackageAdmissionError::UnsafePackage(message))
                if message.contains("changed after capture")
        ));

        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        assert_eq!(
            capture_native_package(&root, &PACKAGE_LIMITS, 8, 4096, &cancelled),
            Err(NativePackageAdmissionError::Cancelled)
        );
        Ok(())
    }

    #[test]
    fn native_package_coverage_is_exact_sorted_unique_and_content_bound() {
        let payloads = BTreeMap::from([
            ("a.txt".to_owned(), b"alpha".to_vec()),
            ("nested/b.bin".to_owned(), b"bravo".to_vec()),
            ("package-coverage.sha256".to_owned(), b"pending\n".to_vec()),
            ("signature.sig".to_owned(), b"pending\n".to_vec()),
        ]);
        let row = |path: &str, bytes: &[u8]| {
            format!("{:x} {}  {path}\n", Sha256::digest(bytes), bytes.len())
        };
        let first = row("a.txt", b"alpha");
        let second = row("nested/b.bin", b"bravo");
        let valid = format!("{first}{second}");
        let excludes = ["package-coverage.sha256", "signature.sig"];
        assert!(
            validate_native_package_coverage(valid.as_bytes(), &payloads, &excludes, 1024).is_ok()
        );

        let cases = [
            format!("{second}{first}"),
            format!("{first}{first}{second}"),
            first,
            valid.replace("nested/b.bin", "unknown.bin"),
            valid.replacen(
                &format!("{:x}", Sha256::digest(b"alpha")),
                &"0".repeat(64),
                1,
            ),
            valid.replacen("5  a.txt", "6  a.txt", 1),
        ];
        for invalid in cases {
            assert!(matches!(
                validate_native_package_coverage(invalid.as_bytes(), &payloads, &excludes, 1024),
                Err(NativePackageAdmissionError::InvalidCoverage(_))
            ));
        }
    }

    #[test]
    fn rocm_package_signature_has_a_distinct_signer_bound_domain()
    -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| std::io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let verifier =
            RocmPackageVerificationKey::new("rocm.release", key_pair.public_key().as_ref())?;
        let coverage = b"fixture coverage\n";
        let payload = rocm_package_signing_payload("rocm.release", coverage)?;
        let signature = encode_hex(key_pair.sign(&payload).as_ref());
        let signed_receipt = |signature: String| -> Result<Vec<u8>, serde_json::Error> {
            let mut bytes = serde_json::to_vec(&NativePackageSignatureReceipt {
                schema_version: 1,
                algorithm: "ed25519".to_owned(),
                signature,
            })?;
            bytes.push(b'\n');
            Ok(bytes)
        };
        let receipt = signed_receipt(signature.clone())?;
        verifier.verify_package("rocm.release", coverage, &receipt)?;
        let metal_verifier =
            MetalPackageVerificationKey::new("rocm.release", key_pair.public_key().as_ref())?;
        assert_eq!(
            metal_verifier.verify_package("rocm.release", coverage, &receipt),
            Err(TrustError::InvalidMetalPackageSignature)
        );
        let metal_payload = metal_package_signing_payload("rocm.release", coverage)?;
        let metal_receipt = signed_receipt(encode_hex(key_pair.sign(&metal_payload).as_ref()))?;
        metal_verifier.verify_package("rocm.release", coverage, &metal_receipt)?;
        assert_eq!(
            verifier.verify_package("rocm.release", coverage, &metal_receipt),
            Err(TrustError::InvalidRocmPackageSignature)
        );
        let unknown_key_pair =
            Ed25519KeyPair::from_seed_unchecked(b"abcdef0123456789abcdef0123456789")
                .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let unknown_metal_key = MetalPackageVerificationKey::new(
            "rocm.release",
            unknown_key_pair.public_key().as_ref(),
        )?;
        assert_eq!(
            unknown_metal_key.verify_package("rocm.release", coverage, &metal_receipt),
            Err(TrustError::InvalidMetalPackageSignature)
        );
        let duplicate_receipt = String::from_utf8(receipt.clone())?
            .replacen(
                "{\"schema_version\":1,",
                "{\"schema_version\":1,\"schema_version\":1,",
                1,
            )
            .into_bytes();
        assert_eq!(
            verifier.verify_package("rocm.release", coverage, &duplicate_receipt),
            Err(TrustError::InvalidRocmPackageSignature)
        );
        assert_eq!(
            metal_verifier.verify_package("rocm.release", coverage, &duplicate_receipt),
            Err(TrustError::InvalidMetalPackageSignature)
        );
        let mut noncanonical_receipt = receipt.clone();
        if noncanonical_receipt.last() == Some(&b'\n') {
            noncanonical_receipt.pop();
        }
        assert_eq!(
            verifier.verify_package("rocm.release", coverage, &noncanonical_receipt),
            Err(TrustError::InvalidRocmPackageSignature)
        );
        assert_eq!(
            verifier.verify_package("plugin.release", coverage, &receipt),
            Err(TrustError::UnknownRocmPackageSigner)
        );
        assert_eq!(
            verifier.verify_package("rocm.release", b"changed coverage\n", &receipt),
            Err(TrustError::InvalidRocmPackageSignature)
        );
        let plugin_domain_signature = encode_hex(
            key_pair
                .sign(&authorization_signing_payload(coverage))
                .as_ref(),
        );
        let plugin_domain_receipt = signed_receipt(plugin_domain_signature)?;
        assert_eq!(
            verifier.verify_package("rocm.release", coverage, &plugin_domain_receipt),
            Err(TrustError::InvalidRocmPackageSignature)
        );
        let receipt_with_unknown_field = serde_json::to_vec(&json!({
            "schema_version": 1,
            "algorithm": "ed25519",
            "signature": signature,
            "plugin_key_id": "must-not-authorize-backends",
        }))?;
        assert_eq!(
            verifier.verify_package("rocm.release", coverage, &receipt_with_unknown_field),
            Err(TrustError::InvalidRocmPackageSignature)
        );
        assert_eq!(
            RocmPackageVerificationKey::new("plugin release", key_pair.public_key().as_ref()),
            Err(TrustError::InvalidRocmPackageVerificationKey)
        );
        Ok(())
    }

    #[test]
    fn directml_package_signature_has_a_distinct_signer_bound_domain()
    -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let verifier = DirectMlPackageVerificationKey::new(
            "directml.release",
            key_pair.public_key().as_ref(),
        )?;
        let coverage = b"fixture coverage\n";
        let payload = directml_package_signing_payload("directml.release", coverage)?;
        let signature = encode_hex(key_pair.sign(&payload).as_ref());
        let mut receipt = serde_json::to_vec(&NativePackageSignatureReceipt {
            schema_version: 1,
            algorithm: "ed25519".to_owned(),
            signature,
        })?;
        receipt.push(b'\n');

        verifier.verify_package("directml.release", coverage, &receipt)?;
        assert_eq!(
            verifier.verify_package("plugin.release", coverage, &receipt),
            Err(TrustError::UnknownDirectMlPackageSigner)
        );
        assert_eq!(
            verifier.verify_package("directml.release", b"changed coverage\n", &receipt),
            Err(TrustError::InvalidDirectMlPackageSignature)
        );

        let mlu_verifier =
            MluPackageVerificationKey::new("directml.release", key_pair.public_key().as_ref())?;
        assert_eq!(
            mlu_verifier.verify_package("directml.release", coverage, &receipt),
            Err(TrustError::InvalidMluPackageSignature)
        );

        let plugin_signature = encode_hex(
            key_pair
                .sign(&authorization_signing_payload(coverage))
                .as_ref(),
        );
        let mut plugin_receipt = serde_json::to_vec(&NativePackageSignatureReceipt {
            schema_version: 1,
            algorithm: "ed25519".to_owned(),
            signature: plugin_signature,
        })?;
        plugin_receipt.push(b'\n');
        assert_eq!(
            verifier.verify_package("directml.release", coverage, &plugin_receipt),
            Err(TrustError::InvalidDirectMlPackageSignature)
        );

        let duplicate_receipt = String::from_utf8(receipt.clone())?
            .replacen(
                "{\"schema_version\":1,",
                "{\"schema_version\":1,\"schema_version\":1,",
                1,
            )
            .into_bytes();
        assert_eq!(
            verifier.verify_package("directml.release", coverage, &duplicate_receipt),
            Err(TrustError::InvalidDirectMlPackageSignature)
        );
        assert_eq!(
            DirectMlPackageVerificationKey::new("plugin release", key_pair.public_key().as_ref()),
            Err(TrustError::InvalidDirectMlPackageVerificationKey)
        );
        Ok(())
    }

    #[test]
    fn task_59_cudart_requires_registry_issued_exact_symbol_certification()
    -> Result<(), Box<dyn std::error::Error>> {
        let contract = NativeFfiContract::new(
            CUDART_LIBRARY_ID,
            DIGEST,
            "cuda-runtime-v1",
            [
                "cudaHostRegister".to_owned(),
                "cudaHostUnregister".to_owned(),
            ],
            "sim_runtime::NativeFfiRegistry",
        )?;
        let registry = NativeFfiRegistry::new([contract])?;
        let certification = registry.authorize(
            CUDART_LIBRARY_ID,
            DIGEST,
            "cuda-runtime-v1",
            &BTreeSet::from([
                "cudaHostRegister".to_owned(),
                "cudaHostUnregister".to_owned(),
            ]),
        )?;
        let cancellation = CancellationToken::default();
        assert_eq!(
            cudart_exact_native(&certification, &cancellation)?,
            &certification
        );

        let incomplete = NativeFfiContract::new(
            CUDART_LIBRARY_ID,
            DIGEST,
            "cuda-runtime-v1",
            ["cudaHostRegister".to_owned()],
            "sim_runtime::NativeFfiRegistry",
        )?;
        let incomplete = NativeFfiRegistry::new([incomplete])?.authorize(
            CUDART_LIBRARY_ID,
            DIGEST,
            "cuda-runtime-v1",
            &BTreeSet::from(["cudaHostRegister".to_owned()]),
        )?;
        assert_eq!(
            cudart_exact_native(&incomplete, &cancellation),
            Err(TrustError::UncertifiedFfi)
        );
        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        assert_eq!(
            cudart_exact_native(&certification, &cancelled),
            Err(TrustError::Cancelled)
        );
        Ok(())
    }

    #[test]
    fn native_ffi_requirements_are_available_only_for_the_exact_registered_owner_and_abi()
    -> Result<(), Box<dyn std::error::Error>> {
        let registry = NativeFfiRegistry::new([NativeFfiContract::new(
            "rocm-dependency:libfixture.so",
            DIGEST,
            "6.1.0",
            ["fixture_symbol".to_owned()],
            "sim_backend_rocm::loader",
        )?])?;

        assert_eq!(
            registry.required_symbols_for(
                "rocm-dependency:libfixture.so",
                "6.1.0",
                "sim_backend_rocm::loader",
            )?,
            BTreeSet::from(["fixture_symbol".to_owned()])
        );
        for (library_id, abi_version, unsafe_owner) in [
            (
                "rocm-dependency:missing.so",
                "6.1.0",
                "sim_backend_rocm::loader",
            ),
            (
                "rocm-dependency:libfixture.so",
                "6.0.0",
                "sim_backend_rocm::loader",
            ),
            ("rocm-dependency:libfixture.so", "6.1.0", "sim_runtime"),
        ] {
            assert_eq!(
                registry.required_symbols_for(library_id, abi_version, unsafe_owner),
                Err(TrustError::UncertifiedFfi)
            );
        }
        Ok(())
    }

    #[test]
    fn native_ffi_dependency_contracts_are_exact_acyclic_and_never_callable()
    -> Result<(), Box<dyn std::error::Error>> {
        let registry = NativeFfiRegistry::new([
            NativeFfiContract::new(
                "ascendcl",
                DIGEST,
                "CANN-8.0.RC3-AscendCL",
                ["aclInit".to_owned()],
                "sim_backend_npu::loader",
            )?,
            NativeFfiContract::new_dependency(
                "runtime",
                DIGEST,
                "CANN-8.0.RC3-AscendCL",
                "ascendcl",
                "sim_backend_npu::loader",
            )?,
        ])?;
        assert_eq!(
            registry.required_symbols_for(
                "runtime",
                "CANN-8.0.RC3-AscendCL",
                "sim_backend_npu::loader",
            ),
            Err(TrustError::UncertifiedFfi)
        );
        assert_eq!(
            registry.authorize("runtime", DIGEST, "CANN-8.0.RC3-AscendCL", &BTreeSet::new(),),
            Err(TrustError::UncertifiedFfi)
        );
        registry.authorize_dependency("runtime", DIGEST, "CANN-8.0.RC3-AscendCL", "ascendcl")?;
        assert_eq!(
            registry.authorize_dependency("runtime", DIGEST, "CANN-8.0.RC3-AscendCL", "other",),
            Err(TrustError::UncertifiedFfi)
        );

        assert_eq!(
            NativeFfiRegistry::new([NativeFfiContract::new_dependency(
                "runtime",
                DIGEST,
                "CANN-8.0.RC3-AscendCL",
                "missing",
                "sim_backend_npu::loader",
            )?]),
            Err(TrustError::InvalidFfiContract)
        );
        assert_eq!(
            NativeFfiRegistry::new([
                NativeFfiContract::new_dependency(
                    "first",
                    DIGEST,
                    "CANN-8.0.RC3-AscendCL",
                    "second",
                    "sim_backend_npu::loader",
                )?,
                NativeFfiContract::new_dependency(
                    "second",
                    DIGEST,
                    "CANN-8.0.RC3-AscendCL",
                    "first",
                    "sim_backend_npu::loader",
                )?,
            ]),
            Err(TrustError::InvalidFfiContract)
        );
        Ok(())
    }

    fn signed_video_codec_catalog(
        key_pair: &Ed25519KeyPair,
    ) -> Result<(Vec<u8>, Vec<u8>, VideoCodecPackageVerificationKey), Box<dyn std::error::Error>>
    {
        signed_video_codec_catalog_with_digests(key_pair, &BTreeMap::new())
    }

    fn signed_video_codec_catalog_with_digests(
        key_pair: &Ed25519KeyPair,
        digests: &BTreeMap<String, String>,
    ) -> Result<(Vec<u8>, Vec<u8>, VideoCodecPackageVerificationKey), Box<dyn std::error::Error>>
    {
        let target = "x86_64-unknown-linux-gnu";
        let mut libraries = Vec::new();
        for (identity, abi_major, symbols) in video_codec_library_contracts() {
            libraries.push(VideoCodecFfiLibraryDto {
                identity: identity.to_owned(),
                filename: video_codec_expected_filename(identity, abi_major, target),
                sha256: digests
                    .get(identity)
                    .cloned()
                    .unwrap_or_else(|| DIGEST.to_owned()),
                abi_major,
                required_symbols: symbols.iter().map(|symbol| (*symbol).to_owned()).collect(),
            });
        }
        let catalog = VideoCodecFfiCatalogDto {
            schema_version: 1,
            profile: VIDEO_CODEC_FFI_PROFILE.to_owned(),
            target: target.to_owned(),
            signer: "video-codec.release".to_owned(),
            signature_algorithm: "ed25519".to_owned(),
            signature_domain: "zed-sim-video-codec-package-v1".to_owned(),
            certificate_owner: "sim_runtime::NativeFfiRegistry".to_owned(),
            unsafe_owner: VIDEO_CODEC_FFI_UNSAFE_OWNER.to_owned(),
            runtime_compilation_forbidden: true,
            redistributes_codec_libraries: false,
            license_notice_sha256: DIGEST.to_owned(),
            external_encoders: VIDEO_CODEC_EXTERNAL_ENCODERS.map(str::to_owned).to_vec(),
            libraries,
        };
        let mut catalog_bytes = serde_json::to_vec(&catalog)?;
        catalog_bytes.push(b'\n');
        let signature_receipt = sign_video_codec_catalog(&catalog_bytes, key_pair)?;
        let verification_key = VideoCodecPackageVerificationKey::new(
            "video-codec.release",
            key_pair.public_key().as_ref(),
        )?;
        Ok((catalog_bytes, signature_receipt, verification_key))
    }

    fn sign_video_codec_catalog(
        catalog_bytes: &[u8],
        key_pair: &Ed25519KeyPair,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let signing_payload =
            video_codec_package_signing_payload("video-codec.release", catalog_bytes)?;
        let mut signature_receipt = serde_json::to_vec(&NativePackageSignatureReceipt {
            schema_version: 1,
            algorithm: "ed25519".to_owned(),
            signature: encode_hex(key_pair.sign(&signing_payload).as_ref()),
        })?;
        signature_receipt.push(b'\n');
        Ok(signature_receipt)
    }

    fn video_codec_dependency_contract_fixture(
        primary_catalog_sha256: &str,
    ) -> VideoCodecDependencyContractDto {
        VideoCodecDependencyContractDto {
            schema_version: 1,
            profile: VIDEO_CODEC_FFI_PROFILE.to_owned(),
            target: VIDEO_CODEC_DEPENDENCY_CONTRACT_TARGET.to_owned(),
            signer: "video-codec.release".to_owned(),
            signature_algorithm: "ed25519".to_owned(),
            signature_domain: "zed-sim-video-codec-dependency-contract-v1".to_owned(),
            certificate_owner: "sim_runtime::NativeFfiRegistry".to_owned(),
            unsafe_owner: VIDEO_CODEC_FFI_UNSAFE_OWNER.to_owned(),
            runtime_compilation_forbidden: true,
            redistributes_codec_libraries: false,
            primary_catalog_sha256: primary_catalog_sha256.to_owned(),
            source_archive_sha256: "1".repeat(64),
            build_recipe_sha256: "2".repeat(64),
            license_bundle_sha256: "3".repeat(64),
            system_libraries: VIDEO_CODEC_SYSTEM_ELF_DEPENDENCIES
                .map(str::to_owned)
                .to_vec(),
            encoder_providers: [
                ("aac", "avcodec"),
                ("libsvtav1", "svtav1"),
                ("libvpx-vp9", "vpx"),
                ("libx264", "x264"),
            ]
            .into_iter()
            .map(|(encoder, provider)| VideoCodecEncoderProviderDto {
                encoder: encoder.to_owned(),
                provider: provider.to_owned(),
            })
            .collect(),
            dependencies: [
                ("svtav1", "libSvtAv1Enc.so.2", "4", "svt-av1:2"),
                ("vpx", "libvpx.so.9", "5", "libvpx:9"),
                ("x264", "libx264.so.164", "6", "libx264:164"),
            ]
            .into_iter()
            .map(
                |(identity, filename, digest_digit, abi_version)| VideoCodecDependencyDto {
                    identity: identity.to_owned(),
                    filename: filename.to_owned(),
                    sha256: digest_digit.repeat(64),
                    abi_version: abi_version.to_owned(),
                    certificate_sponsor: "avcodec".to_owned(),
                },
            )
            .collect(),
            edges: [
                ("avcodec", "avutil"),
                ("avcodec", "libc.so.6"),
                ("avcodec", "svtav1"),
                ("avcodec", "vpx"),
                ("avcodec", "x264"),
                ("avformat", "avcodec"),
                ("avformat", "avutil"),
                ("avutil", "libc.so.6"),
                ("svtav1", "libc.so.6"),
                ("swresample", "avutil"),
                ("swscale", "avutil"),
                ("vpx", "libc.so.6"),
                ("x264", "libc.so.6"),
            ]
            .into_iter()
            .map(|(consumer, dependency)| VideoCodecDependencyEdgeDto {
                consumer: consumer.to_owned(),
                dependency: dependency.to_owned(),
            })
            .collect(),
        }
    }

    fn sign_video_codec_dependency_contract(
        contract: &VideoCodecDependencyContractDto,
        key_pair: &Ed25519KeyPair,
    ) -> Result<(Vec<u8>, Vec<u8>), Box<dyn std::error::Error>> {
        let mut contract_bytes = serde_json::to_vec(contract)?;
        contract_bytes.push(b'\n');
        let payload = video_codec_dependency_contract_signing_payload(
            "video-codec.release",
            &contract_bytes,
        )?;
        let mut signature_receipt = serde_json::to_vec(&NativePackageSignatureReceipt {
            schema_version: 1,
            algorithm: "ed25519".to_owned(),
            signature: encode_hex(key_pair.sign(&payload).as_ref()),
        })?;
        signature_receipt.push(b'\n');
        Ok((contract_bytes, signature_receipt))
    }

    #[cfg(target_os = "linux")]
    fn video_codec_dependency_closure_fixture() -> Result<
        (
            tempfile::TempDir,
            CertifiedInspectedVideoCodecPackage,
            VerifiedVideoCodecDependencyContract,
            BTreeMap<String, PathBuf>,
        ),
        Box<dyn std::error::Error>,
    > {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let directory = tempfile::tempdir()?;
        let cancellation = CancellationToken::default();
        let dependency_rows = [
            ("svtav1", "libSvtAv1Enc.so.2"),
            ("vpx", "libvpx.so.9"),
            ("x264", "libx264.so.164"),
        ];
        let mut dependency_paths = BTreeMap::new();
        let mut dependency_digests = BTreeMap::new();
        for (identity, filename) in dependency_rows {
            let path = directory.path().join(filename);
            let bytes = crate::native_ffi_elf::tests::fixture(
                62,
                &BTreeSet::from([format!("{identity}_fixture_symbol")]),
                &["libc.so.6"],
                None,
                filename,
            );
            fs::write(&path, &bytes)?;
            dependency_digests.insert(identity.to_owned(), format!("{:x}", Sha256::digest(&bytes)));
            dependency_paths.insert(identity.to_owned(), path);
        }

        let filenames = BTreeMap::from([
            ("avcodec", "libavcodec.so.61"),
            ("avformat", "libavformat.so.61"),
            ("avutil", "libavutil.so.59"),
            ("svtav1", "libSvtAv1Enc.so.2"),
            ("swresample", "libswresample.so.5"),
            ("swscale", "libswscale.so.8"),
            ("vpx", "libvpx.so.9"),
            ("x264", "libx264.so.164"),
        ]);
        let primary_needed = BTreeMap::from([
            (
                "avcodec",
                vec!["avutil", "libc.so.6", "svtav1", "vpx", "x264"],
            ),
            ("avformat", vec!["avcodec", "avutil"]),
            ("avutil", vec!["libc.so.6"]),
            ("swresample", vec!["avutil"]),
            ("swscale", vec!["avutil"]),
        ]);
        let mut primary_paths = BTreeMap::new();
        let mut primary_digests = BTreeMap::new();
        for (identity, abi_major, symbols) in video_codec_library_contracts() {
            let filename = video_codec_expected_filename(
                identity,
                abi_major,
                VIDEO_CODEC_DEPENDENCY_CONTRACT_TARGET,
            );
            let needed = primary_needed
                .get(identity)
                .ok_or_else(|| io::Error::other("fixture primary dependency row is missing"))?
                .iter()
                .map(|dependency| {
                    filenames
                        .get(dependency)
                        .copied()
                        .unwrap_or(dependency)
                        .to_owned()
                })
                .collect::<Vec<_>>();
            let needed = needed.iter().map(String::as_str).collect::<Vec<_>>();
            let path = directory.path().join(&filename);
            let bytes = crate::native_ffi_elf::tests::fixture(
                62,
                &symbols.iter().map(|symbol| (*symbol).to_owned()).collect(),
                &needed,
                None,
                &filename,
            );
            fs::write(&path, &bytes)?;
            primary_digests.insert(identity.to_owned(), format!("{:x}", Sha256::digest(&bytes)));
            primary_paths.insert(identity.to_owned(), path);
        }
        let (catalog_bytes, catalog_receipt, verification_key) =
            signed_video_codec_catalog_with_digests(&key_pair, &primary_digests)?;
        let verified_primary = verify_video_codec_ffi_catalog(
            &catalog_bytes,
            &catalog_receipt,
            &verification_key,
            &cancellation,
        )?;
        let inspected = capture_and_inspect_video_codec_package(
            &verified_primary,
            primary_paths,
            &cancellation,
        )?;
        let certified_primary =
            certify_inspected_video_codec_package(&verified_primary, inspected, &cancellation)?;

        let mut contract =
            video_codec_dependency_contract_fixture(verified_primary.catalog_sha256());
        for dependency in &mut contract.dependencies {
            dependency.sha256 = dependency_digests
                .get(&dependency.identity)
                .ok_or_else(|| io::Error::other("fixture dependency digest is missing"))?
                .clone();
        }
        let (contract_bytes, contract_receipt) =
            sign_video_codec_dependency_contract(&contract, &key_pair)?;
        let verified_contract = verify_video_codec_dependency_contract(
            &verified_primary,
            &contract_bytes,
            &contract_receipt,
            &verification_key,
            &cancellation,
        )?;
        Ok((
            directory,
            certified_primary,
            verified_contract,
            dependency_paths,
        ))
    }

    #[test]
    fn video_codec_catalog_is_signed_complete_and_registry_certified()
    -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let (catalog_bytes, signature_receipt, verification_key) =
            signed_video_codec_catalog(&key_pair)?;
        let cancellation = CancellationToken::default();
        let verified = verify_video_codec_ffi_catalog(
            &catalog_bytes,
            &signature_receipt,
            &verification_key,
            &cancellation,
        )?;
        assert_eq!(verified.target(), "x86_64-unknown-linux-gnu");
        assert_eq!(verified.libraries().len(), 5);

        let observations = video_codec_library_contracts()
            .into_iter()
            .map(|(identity, abi_major, symbols)| {
                NativeVideoCodecLibraryObservation::checked(
                    identity,
                    video_codec_expected_filename(identity, abi_major, verified.target()),
                    DIGEST,
                    abi_major,
                    symbols.iter().map(|symbol| (*symbol).to_owned()),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let certified = certify_video_codec_ffi(verified, observations, &cancellation)?;
        assert_eq!(certified.target(), "x86_64-unknown-linux-gnu");
        assert_eq!(certified.certificates().len(), 5);
        for (identity, certificate) in certified.certificates() {
            assert_eq!(certificate.library_id(), identity);
            assert_eq!(certificate.digest_sha256(), DIGEST);
            assert_eq!(certificate.unsafe_owner(), VIDEO_CODEC_FFI_UNSAFE_OWNER);
            assert!(!certificate.required_symbols().is_empty());
        }
        Ok(())
    }

    #[test]
    fn video_codec_dependency_contract_is_signed_complete_and_non_callable()
    -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let (catalog_bytes, signature_receipt, verification_key) =
            signed_video_codec_catalog(&key_pair)?;
        let cancellation = CancellationToken::default();
        let primary = verify_video_codec_ffi_catalog(
            &catalog_bytes,
            &signature_receipt,
            &verification_key,
            &cancellation,
        )?;
        let contract = video_codec_dependency_contract_fixture(primary.catalog_sha256());
        let (contract_bytes, contract_receipt) =
            sign_video_codec_dependency_contract(&contract, &key_pair)?;
        let verified = verify_video_codec_dependency_contract(
            &primary,
            &contract_bytes,
            &contract_receipt,
            &verification_key,
            &cancellation,
        )?;

        assert_eq!(verified.target(), VIDEO_CODEC_DEPENDENCY_CONTRACT_TARGET);
        assert_eq!(verified.primary_catalog_sha256(), primary.catalog_sha256());
        assert_eq!(verified.source_archive_sha256(), "1".repeat(64));
        assert_eq!(verified.build_recipe_sha256(), "2".repeat(64));
        assert_eq!(verified.license_bundle_sha256(), "3".repeat(64));
        assert_eq!(verified.dependencies().len(), 3);
        assert_eq!(verified.edges().len(), 13);
        assert_eq!(verified.encoder_providers().len(), 4);
        assert_eq!(verified.system_libraries().len(), 8);
        let x264 = verified
            .dependencies()
            .get("x264")
            .ok_or_else(|| io::Error::other("verified x264 dependency is missing"))?;
        assert_eq!(x264.filename(), "libx264.so.164");
        assert_eq!(x264.abi_version(), "libx264:164");
        assert_eq!(x264.certificate_sponsor(), "avcodec");
        assert_eq!(x264.digest_sha256(), "6".repeat(64));
        assert_eq!(
            verified._registry.required_symbols_for(
                "x264",
                "libx264:164",
                VIDEO_CODEC_FFI_UNSAFE_OWNER,
            ),
            Err(TrustError::UncertifiedFfi)
        );
        assert_eq!(
            verified
                ._registry
                .authorize("x264", &"6".repeat(64), "libx264:164", &BTreeSet::new(),),
            Err(TrustError::UncertifiedFfi)
        );

        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        assert!(matches!(
            verify_video_codec_dependency_contract(
                &primary,
                &contract_bytes,
                &contract_receipt,
                &verification_key,
                &cancelled,
            ),
            Err(VideoCodecDependencyContractError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn video_codec_dependency_contract_rejects_graph_policy_and_signature_drift()
    -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let (catalog_bytes, signature_receipt, verification_key) =
            signed_video_codec_catalog(&key_pair)?;
        let cancellation = CancellationToken::default();
        let primary = verify_video_codec_ffi_catalog(
            &catalog_bytes,
            &signature_receipt,
            &verification_key,
            &cancellation,
        )?;
        let valid = video_codec_dependency_contract_fixture(primary.catalog_sha256());

        let mut variants = Vec::new();
        let mut wrong_catalog = valid.clone();
        wrong_catalog.primary_catalog_sha256 = "f".repeat(64);
        variants.push(wrong_catalog);
        let mut wrong_target = valid.clone();
        wrong_target.target = "aarch64-unknown-linux-gnu".to_owned();
        variants.push(wrong_target);
        let mut wrong_provenance = valid.clone();
        wrong_provenance.source_archive_sha256 = "not-a-digest".to_owned();
        variants.push(wrong_provenance);
        let mut broadened_system_policy = valid.clone();
        broadened_system_policy
            .system_libraries
            .push("libambient.so.1".to_owned());
        variants.push(broadened_system_policy);
        let mut missing_provider = valid.clone();
        missing_provider.encoder_providers.pop();
        variants.push(missing_provider);
        let mut wrong_provider = valid.clone();
        let provider = wrong_provider
            .encoder_providers
            .iter_mut()
            .find(|provider| provider.encoder == "libx264")
            .ok_or_else(|| io::Error::other("fixture libx264 provider is missing"))?;
        provider.provider = "avcodec".to_owned();
        variants.push(wrong_provider);
        let mut wrong_sponsor = valid.clone();
        let dependency = wrong_sponsor
            .dependencies
            .iter_mut()
            .find(|dependency| dependency.identity == "x264")
            .ok_or_else(|| io::Error::other("fixture x264 dependency is missing"))?;
        dependency.certificate_sponsor = "swscale".to_owned();
        variants.push(wrong_sponsor);
        let mut unknown_edge = valid.clone();
        unknown_edge.edges.push(VideoCodecDependencyEdgeDto {
            consumer: "x264".to_owned(),
            dependency: "libambient.so.1".to_owned(),
        });
        variants.push(unknown_edge);
        let mut missing_edge = valid.clone();
        missing_edge
            .edges
            .retain(|edge| !(edge.consumer == "avcodec" && edge.dependency == "x264"));
        variants.push(missing_edge);
        let mut duplicate_edge = valid.clone();
        duplicate_edge.edges.push(VideoCodecDependencyEdgeDto {
            consumer: "avcodec".to_owned(),
            dependency: "x264".to_owned(),
        });
        duplicate_edge.edges.sort_by(|left, right| {
            (&left.consumer, &left.dependency).cmp(&(&right.consumer, &right.dependency))
        });
        variants.push(duplicate_edge);
        let mut cycle = valid.clone();
        cycle.edges.push(VideoCodecDependencyEdgeDto {
            consumer: "x264".to_owned(),
            dependency: "avcodec".to_owned(),
        });
        cycle.edges.sort_by(|left, right| {
            (&left.consumer, &left.dependency).cmp(&(&right.consumer, &right.dependency))
        });
        variants.push(cycle);
        let mut unreachable = valid.clone();
        unreachable.dependencies.push(VideoCodecDependencyDto {
            identity: "orphan".to_owned(),
            filename: "liborphan.so.1".to_owned(),
            sha256: "7".repeat(64),
            abi_version: "orphan:1".to_owned(),
            certificate_sponsor: "avcodec".to_owned(),
        });
        unreachable
            .dependencies
            .sort_by(|left, right| left.identity.cmp(&right.identity));
        variants.push(unreachable);
        let mut malformed_filename = valid.clone();
        malformed_filename
            .dependencies
            .first_mut()
            .ok_or_else(|| io::Error::other("fixture dependency set is empty"))?
            .filename = "../libsvtav1.so".to_owned();
        variants.push(malformed_filename);
        let mut system_filename_collision = valid.clone();
        system_filename_collision
            .dependencies
            .first_mut()
            .ok_or_else(|| io::Error::other("fixture dependency set is empty"))?
            .filename = "libc.so.6".to_owned();
        variants.push(system_filename_collision);
        let mut malformed_digest = valid.clone();
        malformed_digest
            .dependencies
            .first_mut()
            .ok_or_else(|| io::Error::other("fixture dependency set is empty"))?
            .sha256 = "not-a-digest".to_owned();
        variants.push(malformed_digest);
        let mut malformed_abi = valid.clone();
        malformed_abi
            .dependencies
            .first_mut()
            .ok_or_else(|| io::Error::other("fixture dependency set is empty"))?
            .abi_version = "libsvtav1/2".to_owned();
        variants.push(malformed_abi);
        let mut duplicate_dependency = valid.clone();
        let first_dependency = duplicate_dependency
            .dependencies
            .first()
            .cloned()
            .ok_or_else(|| io::Error::other("fixture dependency set is empty"))?;
        duplicate_dependency.dependencies.push(first_dependency);
        duplicate_dependency
            .dependencies
            .sort_by(|left, right| left.identity.cmp(&right.identity));
        variants.push(duplicate_dependency);
        let mut missing_dependency = valid.clone();
        missing_dependency
            .dependencies
            .retain(|dependency| dependency.identity != "x264");
        variants.push(missing_dependency);

        for variant in variants {
            let (bytes, receipt) = sign_video_codec_dependency_contract(&variant, &key_pair)?;
            assert!(matches!(
                verify_video_codec_dependency_contract(
                    &primary,
                    &bytes,
                    &receipt,
                    &verification_key,
                    &cancellation,
                ),
                Err(VideoCodecDependencyContractError::ContractMismatch)
            ));
        }

        let (bytes, mut receipt) = sign_video_codec_dependency_contract(&valid, &key_pair)?;
        let last = receipt
            .last_mut()
            .ok_or_else(|| io::Error::other("fixture receipt is empty"))?;
        *last = b' ';
        assert!(matches!(
            verify_video_codec_dependency_contract(
                &primary,
                &bytes,
                &receipt,
                &verification_key,
                &cancellation,
            ),
            Err(VideoCodecDependencyContractError::Trust(
                TrustError::InvalidVideoCodecPackageSignature
            ))
        ));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn video_codec_callable_exports_require_exact_global_function_versions()
    -> Result<(), Box<dyn std::error::Error>> {
        let required = BTreeSet::from(["avcodec_version".to_owned()]);
        let bytes = crate::native_ffi_elf::tests::fixture(
            62,
            &required,
            &["libc.so.6"],
            None,
            "libavcodec.so.61",
        );
        let valid = inspect_elf64_dynamic_contract(&bytes, 62, &CancellationToken::default())?;
        let admitted = checked_video_codec_callable_symbols("avcodec", &required, &valid)?;
        assert_eq!(admitted.len(), 1);
        assert_eq!(
            admitted
                .get("avcodec_version")
                .map(VideoCodecCallableElfSymbolIdentity::version_namespace),
            Some("LIBAVCODEC_61")
        );

        let mutations: [fn(&mut crate::native_ffi_elf::NativeElfSymbolIdentity); 13] = [
            |identity| identity.binding = 0,
            |identity| identity.binding = 2,
            |identity| identity.kind = 0,
            |identity| identity.kind = 1,
            |identity| identity.kind = 6,
            |identity| identity.kind = 10,
            |identity| identity.visibility = 1,
            |identity| identity.visibility = 2,
            |identity| identity.visibility = 3,
            |identity| identity.section_index = 0,
            |identity| identity.value = 0,
            |identity| identity.executable = false,
            |identity| identity.version = None,
        ];
        for mutate in mutations {
            let mut changed = valid.clone();
            let identity = changed
                .symbol_identities
                .get_mut("avcodec_version")
                .and_then(|identities| identities.first_mut())
                .ok_or("fixture callable identity is missing")?;
            mutate(identity);
            assert!(matches!(
                checked_video_codec_callable_symbols("avcodec", &required, &changed),
                Err(VideoCodecPackageCaptureError::InvalidCallableSymbol { .. })
            ));
        }

        for (version_name, is_default) in [("LIBAVCODEC_60", true), ("LIBAVCODEC_61", false)] {
            let mut changed = valid.clone();
            let version = changed
                .symbol_identities
                .get_mut("avcodec_version")
                .and_then(|identities| identities.first_mut())
                .and_then(|identity| identity.version.as_mut())
                .ok_or("fixture symbol version is missing")?;
            version.name = version_name.to_owned();
            version.is_default = is_default;
            assert!(matches!(
                checked_video_codec_callable_symbols("avcodec", &required, &changed),
                Err(VideoCodecPackageCaptureError::InvalidCallableSymbol { .. })
            ));
        }

        let mut duplicate = valid;
        let identity = duplicate
            .symbol_identities
            .get("avcodec_version")
            .and_then(|identities| identities.first())
            .cloned()
            .ok_or("fixture callable identity is missing")?;
        duplicate
            .symbol_identities
            .get_mut("avcodec_version")
            .ok_or("fixture callable identity vector is missing")?
            .push(identity);
        assert!(matches!(
            checked_video_codec_callable_symbols("avcodec", &required, &duplicate),
            Err(VideoCodecPackageCaptureError::InvalidCallableSymbol { .. })
        ));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn video_codec_package_capture_seals_exact_catalog_images()
    -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let directory = tempfile::tempdir()?;
        let mut paths = BTreeMap::new();
        let mut digests = BTreeMap::new();
        for (identity, abi_major, symbols) in video_codec_library_contracts() {
            let filename =
                video_codec_expected_filename(identity, abi_major, "x86_64-unknown-linux-gnu");
            let path = directory.path().join(&filename);
            let symbols = symbols
                .iter()
                .map(|symbol| (*symbol).to_owned())
                .collect::<BTreeSet<_>>();
            let bytes = crate::native_ffi_elf::tests::fixture(
                62,
                &symbols,
                &["libc.so.6"],
                None,
                &filename,
            );
            std::fs::write(&path, &bytes)?;
            digests.insert(identity.to_owned(), format!("{:x}", Sha256::digest(&bytes)));
            paths.insert(identity.to_owned(), path);
        }
        let (catalog_bytes, signature_receipt, verification_key) =
            signed_video_codec_catalog_with_digests(&key_pair, &digests)?;
        let cancellation = CancellationToken::default();
        let verified = verify_video_codec_ffi_catalog(
            &catalog_bytes,
            &signature_receipt,
            &verification_key,
            &cancellation,
        )?;
        let captured = capture_video_codec_package(&verified, paths.clone(), &cancellation)?;
        assert_eq!(captured.target(), "x86_64-unknown-linux-gnu");
        assert_eq!(captured.libraries(), verified.libraries());
        let inspected =
            capture_and_inspect_video_codec_package(&verified, paths.clone(), &cancellation)?;
        assert_eq!(inspected.target(), "x86_64-unknown-linux-gnu");
        assert_eq!(inspected.elf_libraries().len(), 5);
        for (identity, library) in inspected.elf_libraries() {
            let expected = verified
                .libraries()
                .get(identity)
                .ok_or_else(|| io::Error::other("inspected library is not in the catalog"))?;
            let expected_symbols = video_codec_library_contracts()
                .into_iter()
                .find(|(expected_identity, _, _)| expected_identity == identity)
                .map(|(_, _, symbols)| {
                    symbols
                        .iter()
                        .map(|symbol| (*symbol).to_owned())
                        .collect::<BTreeSet<_>>()
                })
                .ok_or_else(|| io::Error::other("reviewed symbol contract is missing"))?;
            assert_eq!(library.soname(), expected.filename());
            assert_eq!(
                library
                    .callable_symbols()
                    .keys()
                    .cloned()
                    .collect::<BTreeSet<_>>(),
                expected_symbols
            );
            assert!(library.callable_symbols().values().all(|symbol| {
                symbol.value() != 0
                    && symbol.version_namespace()
                        == video_codec_symbol_version_namespace(identity).unwrap_or_default()
            }));
            assert_eq!(
                library.needed_libraries(),
                &BTreeSet::from(["libc.so.6".to_owned()])
            );
        }
        let certified = certify_inspected_video_codec_package(&verified, inspected, &cancellation)?;
        assert_eq!(certified.target(), "x86_64-unknown-linux-gnu");
        assert_eq!(certified.libraries(), verified.libraries());
        assert_eq!(certified.elf_libraries().len(), 5);
        assert_eq!(certified.certificates().len(), 5);
        for (identity, certificate) in certified.certificates() {
            let expected = verified
                .libraries()
                .get(identity)
                .ok_or_else(|| io::Error::other("certificate is not in the signed catalog"))?;
            assert_eq!(certificate.library_id(), identity);
            assert_eq!(certificate.digest_sha256(), expected.digest_sha256());
            assert_eq!(certificate.unsafe_owner(), VIDEO_CODEC_FFI_UNSAFE_OWNER);
        }

        let inspected =
            capture_and_inspect_video_codec_package(&verified, paths.clone(), &cancellation)?;
        let cancelled_certification = CancellationToken::default();
        cancelled_certification.cancel();
        assert!(matches!(
            certify_inspected_video_codec_package(&verified, inspected, &cancelled_certification,),
            Err(VideoCodecFfiCertificationError::Cancelled)
        ));

        let inspected =
            capture_and_inspect_video_codec_package(&verified, paths.clone(), &cancellation)?;
        let mut mismatched_digests = digests.clone();
        mismatched_digests.insert("avcodec".to_owned(), "f".repeat(64));
        let (catalog_bytes, signature_receipt, verification_key) =
            signed_video_codec_catalog_with_digests(&key_pair, &mismatched_digests)?;
        let mismatched = verify_video_codec_ffi_catalog(
            &catalog_bytes,
            &signature_receipt,
            &verification_key,
            &cancellation,
        )?;
        assert!(matches!(
            certify_inspected_video_codec_package(&mismatched, inspected, &cancellation),
            Err(VideoCodecFfiCertificationError::InspectedPackageMismatch)
        ));

        let mut missing = paths.clone();
        missing.remove("avcodec");
        assert!(matches!(
            capture_video_codec_package(&verified, missing, &cancellation),
            Err(VideoCodecPackageCaptureError::Incomplete)
        ));

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            capture_video_codec_package(&verified, paths.clone(), &cancelled),
            Err(VideoCodecPackageCaptureError::Cancelled)
        ));

        let avutil = paths
            .get("avutil")
            .ok_or_else(|| io::Error::other("fixture avutil path is missing"))?;
        std::fs::write(avutil, b"changed")?;
        assert!(matches!(
            capture_video_codec_package(&verified, paths.clone(), &cancellation),
            Err(VideoCodecPackageCaptureError::ContractMismatch { identity })
                if identity == "avutil"
        ));

        let (identity, abi_major, required) = video_codec_library_contracts()
            .into_iter()
            .find(|(identity, _, _)| *identity == "avcodec")
            .ok_or_else(|| io::Error::other("fixture avcodec contract is missing"))?;
        let mut incomplete_symbols = required
            .iter()
            .map(|symbol| (*symbol).to_owned())
            .collect::<BTreeSet<_>>();
        let missing_symbol = incomplete_symbols
            .iter()
            .next()
            .cloned()
            .ok_or_else(|| io::Error::other("fixture avcodec symbol set is empty"))?;
        incomplete_symbols.remove(&missing_symbol);
        let filename =
            video_codec_expected_filename(identity, abi_major, "x86_64-unknown-linux-gnu");
        let avcodec = paths
            .get(identity)
            .ok_or_else(|| io::Error::other("fixture avcodec path is missing"))?;
        let incomplete_elf = crate::native_ffi_elf::tests::fixture(
            62,
            &incomplete_symbols,
            &["libc.so.6"],
            None,
            &filename,
        );
        std::fs::write(avcodec, &incomplete_elf)?;
        let mut incomplete_digests = digests;
        incomplete_digests.insert(
            identity.to_owned(),
            format!("{:x}", Sha256::digest(&incomplete_elf)),
        );
        let (catalog_bytes, signature_receipt, verification_key) =
            signed_video_codec_catalog_with_digests(&key_pair, &incomplete_digests)?;
        let verified_incomplete = verify_video_codec_ffi_catalog(
            &catalog_bytes,
            &signature_receipt,
            &verification_key,
            &cancellation,
        )?;
        assert!(matches!(
            capture_and_inspect_video_codec_package(
                &verified_incomplete,
                paths,
                &cancellation,
            ),
            Err(VideoCodecPackageCaptureError::MissingSymbol {
                identity: rejected_identity,
                symbol: rejected_symbol,
            }) if rejected_identity == identity && rejected_symbol == missing_symbol
        ));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn video_codec_dependency_closure_certifies_retains_and_orders_exact_graph()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, primary, contract, dependency_paths) =
            video_codec_dependency_closure_fixture()?;
        let expected_bytes = dependency_paths.values().try_fold(0_u64, |total, path| {
            total
                .checked_add(fs::metadata(path)?.len())
                .ok_or_else(|| io::Error::other("fixture dependency byte total overflowed"))
        })?;
        let source_path = dependency_paths
            .get("x264")
            .ok_or_else(|| io::Error::other("fixture x264 path is missing"))?
            .clone();
        let closure = certify_video_codec_dependency_closure(
            primary,
            contract,
            dependency_paths,
            &CancellationToken::default(),
        )?;
        assert_eq!(closure.target(), VIDEO_CODEC_DEPENDENCY_CONTRACT_TARGET);
        assert_eq!(closure.primary_libraries().len(), 5);
        assert_eq!(closure.dependencies().len(), 3);
        assert_eq!(closure.primary_certificates().len(), 5);
        assert_eq!(closure.dependency_certificates().len(), 3);
        assert_eq!(closure.dependency_elf_libraries().len(), 3);
        assert_eq!(closure.edges().len(), 13);
        assert_eq!(closure.retained_dependency_bytes(), expected_bytes);
        assert_eq!(
            closure
                .encoder_providers()
                .get("libx264")
                .map(String::as_str),
            Some("x264")
        );
        for (identity, certificate) in closure.dependency_certificates() {
            assert_eq!(certificate.library_id(), identity);
            assert!(certificate.required_symbols().is_empty());
            assert_eq!(certificate.unsafe_owner(), VIDEO_CODEC_FFI_UNSAFE_OWNER);
        }
        let positions = closure
            .dependency_first_order()
            .iter()
            .enumerate()
            .map(|(index, identity)| (identity.as_str(), index))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(positions.len(), 8);
        for edge in closure.edges() {
            let Some(dependency_position) = positions.get(edge.dependency()) else {
                continue;
            };
            let consumer_position = positions
                .get(edge.consumer())
                .ok_or_else(|| io::Error::other("fixture consumer is absent from load order"))?;
            assert!(dependency_position < consumer_position);
        }

        fs::write(&source_path, b"changed after certification")?;
        let retained = closure
            ._sealed_dependency_images
            .get("x264")
            .ok_or_else(|| io::Error::other("retained x264 image is missing"))?;
        let mut retained_file = retained.file().try_clone()?;
        retained_file.seek(SeekFrom::Start(0))?;
        let mut retained_bytes = Vec::new();
        retained_file.read_to_end(&mut retained_bytes)?;
        let expected_digest = closure
            .dependencies()
            .get("x264")
            .ok_or_else(|| io::Error::other("x264 contract is missing"))?
            .digest_sha256();
        assert_eq!(
            format!("{:x}", Sha256::digest(&retained_bytes)),
            expected_digest
        );
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn video_codec_dependency_closure_rejects_contract_path_and_resource_drift()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, primary, contract, mut paths) = video_codec_dependency_closure_fixture()?;
        paths.remove("x264");
        assert!(matches!(
            certify_video_codec_dependency_closure(
                primary,
                contract,
                paths,
                &CancellationToken::default(),
            ),
            Err(VideoCodecDependencyClosureError::IncompletePathSet)
        ));

        let (_directory, primary, mut contract, paths) = video_codec_dependency_closure_fixture()?;
        contract.primary_catalog_sha256 = "f".repeat(64);
        assert!(matches!(
            certify_video_codec_dependency_closure(
                primary,
                contract,
                paths,
                &CancellationToken::default(),
            ),
            Err(VideoCodecDependencyClosureError::ContractMismatch)
        ));

        let (_directory, primary, mut contract, paths) = video_codec_dependency_closure_fixture()?;
        contract.edges.remove(&VideoCodecDependencyEdge {
            consumer: "x264".to_owned(),
            dependency: "libc.so.6".to_owned(),
        });
        assert!(matches!(
            certify_video_codec_dependency_closure(
                primary,
                contract,
                paths,
                &CancellationToken::default(),
            ),
            Err(VideoCodecDependencyClosureError::ContractMismatch)
        ));

        let (_directory, primary, contract, paths) = video_codec_dependency_closure_fixture()?;
        assert!(matches!(
            certify_video_codec_dependency_closure_with_limits(
                primary,
                contract,
                paths,
                2,
                u64::MAX,
                &CancellationToken::default(),
            ),
            Err(VideoCodecDependencyClosureError::ResourceLimitExceeded)
        ));
        let (_directory, primary, contract, paths) = video_codec_dependency_closure_fixture()?;
        assert!(matches!(
            certify_video_codec_dependency_closure_with_limits(
                primary,
                contract,
                paths,
                MAX_VIDEO_CODEC_DEPENDENCY_IMAGES,
                1,
                &CancellationToken::default(),
            ),
            Err(VideoCodecDependencyClosureError::ResourceLimitExceeded)
        ));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn video_codec_dependency_closure_cancellation_is_atomic_and_retryable()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, primary, contract, paths) = video_codec_dependency_closure_fixture()?;
        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        assert!(matches!(
            certify_video_codec_dependency_closure(primary, contract, paths, &cancelled),
            Err(VideoCodecDependencyClosureError::Cancelled)
        ));

        let (_directory, primary, contract, paths) = video_codec_dependency_closure_fixture()?;
        let closure = certify_video_codec_dependency_closure(
            primary,
            contract,
            paths,
            &CancellationToken::default(),
        )?;
        assert_eq!(closure.dependency_certificates().len(), 3);
        Ok(())
    }

    #[test]
    fn video_codec_catalog_rejects_tampering_incomplete_symbols_and_cancellation()
    -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let (catalog_bytes, signature_receipt, verification_key) =
            signed_video_codec_catalog(&key_pair)?;
        let cancellation = CancellationToken::default();

        let mut tampered = catalog_bytes.clone();
        let first_digit = tampered
            .iter()
            .position(|byte| *byte == b'7')
            .ok_or_else(|| io::Error::other("fixture profile digit is missing"))?;
        let digit = tampered
            .get_mut(first_digit)
            .ok_or_else(|| io::Error::other("fixture profile digit is out of bounds"))?;
        *digit = b'8';
        assert!(
            verify_video_codec_ffi_catalog(
                &tampered,
                &signature_receipt,
                &verification_key,
                &cancellation,
            )
            .is_err()
        );

        let verified = verify_video_codec_ffi_catalog(
            &catalog_bytes,
            &signature_receipt,
            &verification_key,
            &cancellation,
        )?;
        let observations = video_codec_library_contracts()
            .into_iter()
            .map(|(identity, abi_major, symbols)| {
                let available = if identity == "avcodec" {
                    symbols
                        .iter()
                        .skip(1)
                        .map(|symbol| (*symbol).to_owned())
                        .collect::<Vec<_>>()
                } else {
                    symbols.iter().map(|symbol| (*symbol).to_owned()).collect()
                };
                NativeVideoCodecLibraryObservation::checked(
                    identity,
                    video_codec_expected_filename(identity, abi_major, verified.target()),
                    DIGEST,
                    abi_major,
                    available,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert!(matches!(
            certify_video_codec_ffi(verified, observations, &cancellation),
            Err(VideoCodecFfiCertificationError::Trust(
                TrustError::UncertifiedFfi
            ))
        ));

        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        assert!(matches!(
            verify_video_codec_ffi_catalog(
                &catalog_bytes,
                &signature_receipt,
                &verification_key,
                &cancelled,
            ),
            Err(VideoCodecFfiCatalogError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn video_codec_catalog_rejects_noncanonical_policy_and_observation_mismatches()
    -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let (catalog_bytes, signature_receipt, verification_key) =
            signed_video_codec_catalog(&key_pair)?;
        let cancellation = CancellationToken::default();

        let catalog_value: serde_json::Value = serde_json::from_slice(&catalog_bytes)?;
        let mut noncanonical = serde_json::to_vec_pretty(&catalog_value)?;
        noncanonical.push(b'\n');
        let noncanonical_signature = sign_video_codec_catalog(&noncanonical, &key_pair)?;
        assert!(matches!(
            verify_video_codec_ffi_catalog(
                &noncanonical,
                &noncanonical_signature,
                &verification_key,
                &cancellation,
            ),
            Err(VideoCodecFfiCatalogError::Malformed)
        ));

        for (field, replacement) in [
            ("target", json!("riscv64-unknown-linux-gnu")),
            ("license_notice_sha256", json!("invalid")),
        ] {
            let mut changed = catalog_value.clone();
            let object = changed
                .as_object_mut()
                .ok_or_else(|| io::Error::other("catalog fixture is not an object"))?;
            object.insert(field.to_owned(), replacement);
            let mut changed_bytes = serde_json::to_vec(&changed)?;
            changed_bytes.push(b'\n');
            let changed_signature = sign_video_codec_catalog(&changed_bytes, &key_pair)?;
            assert!(matches!(
                verify_video_codec_ffi_catalog(
                    &changed_bytes,
                    &changed_signature,
                    &verification_key,
                    &cancellation,
                ),
                Err(VideoCodecFfiCatalogError::ContractMismatch)
            ));
        }

        let mut changed_abi = catalog_value;
        let libraries = changed_abi
            .get_mut("libraries")
            .and_then(serde_json::Value::as_array_mut)
            .ok_or_else(|| io::Error::other("catalog libraries are missing"))?;
        let avcodec = libraries
            .first_mut()
            .and_then(serde_json::Value::as_object_mut)
            .ok_or_else(|| io::Error::other("avcodec contract is missing"))?;
        avcodec.insert("abi_major".to_owned(), json!(62));
        let mut changed_abi_bytes = serde_json::to_vec(&changed_abi)?;
        changed_abi_bytes.push(b'\n');
        let changed_abi_signature = sign_video_codec_catalog(&changed_abi_bytes, &key_pair)?;
        assert!(matches!(
            verify_video_codec_ffi_catalog(
                &changed_abi_bytes,
                &changed_abi_signature,
                &verification_key,
                &cancellation,
            ),
            Err(VideoCodecFfiCatalogError::ContractMismatch)
        ));

        let observation_set = |filename_override: Option<&str>, digest: &str, abi_delta: u16| {
            video_codec_library_contracts()
                .into_iter()
                .map(|(identity, abi_major, symbols)| {
                    NativeVideoCodecLibraryObservation::checked(
                        identity,
                        filename_override
                            .filter(|_| identity == "avcodec")
                            .map(str::to_owned)
                            .unwrap_or_else(|| {
                                video_codec_expected_filename(
                                    identity,
                                    abi_major,
                                    "x86_64-unknown-linux-gnu",
                                )
                            }),
                        if identity == "avcodec" {
                            digest
                        } else {
                            DIGEST
                        },
                        if identity == "avcodec" {
                            abi_major.saturating_add(abi_delta)
                        } else {
                            abi_major
                        },
                        symbols.iter().map(|symbol| (*symbol).to_owned()),
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        };

        let missing = observation_set(None, DIGEST, 0)?
            .into_iter()
            .skip(1)
            .collect::<Vec<_>>();
        let verified = verify_video_codec_ffi_catalog(
            &catalog_bytes,
            &signature_receipt,
            &verification_key,
            &cancellation,
        )?;
        assert!(matches!(
            certify_video_codec_ffi(verified, missing, &cancellation),
            Err(VideoCodecFfiCertificationError::IncompleteObservationSet)
        ));

        let mut duplicate = observation_set(None, DIGEST, 0)?;
        let repeated = duplicate
            .first()
            .cloned()
            .ok_or_else(|| io::Error::other("observation fixture is empty"))?;
        duplicate.push(repeated);
        let verified = verify_video_codec_ffi_catalog(
            &catalog_bytes,
            &signature_receipt,
            &verification_key,
            &cancellation,
        )?;
        assert!(matches!(
            certify_video_codec_ffi(verified, duplicate, &cancellation),
            Err(VideoCodecFfiCertificationError::IncompleteObservationSet)
        ));

        for observations in [
            observation_set(Some("libavcodec.so.62"), DIGEST, 0)?,
            observation_set(None, &"f".repeat(64), 0)?,
            observation_set(None, DIGEST, 1)?,
        ] {
            let verified = verify_video_codec_ffi_catalog(
                &catalog_bytes,
                &signature_receipt,
                &verification_key,
                &cancellation,
            )?;
            assert!(matches!(
                certify_video_codec_ffi(verified, observations, &cancellation),
                Err(VideoCodecFfiCertificationError::ObservationMismatch)
            ));
        }
        Ok(())
    }

    fn permission_policy(capabilities: CapabilitySet) -> Result<PermissionPolicy, PermissionError> {
        PermissionPolicy::new(
            "profile-a",
            [PermissionGrant::new(
                "profile-a",
                "plugin.fixture",
                capabilities,
                "approved-settings",
            )?],
        )
    }

    fn capability(kind: CapabilityKind, scope: &str) -> CapabilityRequest {
        CapabilityRequest {
            kind,
            scope: scope.to_owned(),
            quota: CapabilityQuota {
                maximum_operations: 4,
                maximum_request_bytes: 4_096,
                maximum_response_bytes: 4_096,
                maximum_total_bytes: 8_192,
                maximum_handles: 4,
                timeout_milliseconds: 1_000,
            },
        }
    }

    fn signed_plugin(
        requested_capabilities: Vec<CapabilityRequest>,
    ) -> Result<PluginManifest, TrustError> {
        let mut manifest = PluginManifest {
            schema_version: 1,
            identifier: "plugin.fixture".to_owned(),
            plugin_version: ApiVersion::new(1, 0, 0),
            api: ApiRequirement {
                major: 1,
                minimum_minor: 0,
                maximum_minor: 0,
                required_features: Vec::new(),
            },
            digest_sha256: DIGEST.to_owned(),
            signature: ManifestSignature {
                algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
                key_id: "fixture.key".to_owned(),
                value: "0".repeat(ED25519_SIGNATURE_BYTES * 2),
            },
            provenance: ManifestProvenance {
                source: "fixture://plugin.fixture".to_owned(),
                publisher: "fixture publisher".to_owned(),
                registry: Some("fixture://registry".to_owned()),
            },
            provider_binding: None,
            nodes: vec![PluginNode {
                id: "node.fixture".to_owned(),
                version: ApiVersion::new(1, 0, 0),
                display_name: "Fixture".to_owned(),
                category: "tests".to_owned(),
                ports: Vec::new(),
                determinism: DeterminismPolicy::Deterministic,
                cache: CachePolicy::InputIdentity,
                effects: EffectPolicy::Pure,
            }],
            capabilities: requested_capabilities,
            ui: Vec::new(),
            routes: Vec::new(),
            legacy_mappings: Vec::new(),
        };
        manifest.signature.value = signing_key()?.sign_manifest(&manifest)?;
        Ok(manifest)
    }

    fn signed_provider_plugin_v2() -> Result<ProviderPluginManifestV2, Box<dyn std::error::Error>> {
        let endpoint = "https://provider.invalid/v2";
        let mut manifest = signed_plugin(vec![capability(
            CapabilityKind::NetworkProvider,
            &format!("plugin.fixture|{endpoint}"),
        )])?;
        manifest.api.required_features.extend([
            PROVIDER_BINDING_API_FEATURE.to_owned(),
            PROVIDER_STREAMING_API_FEATURE_V2.to_owned(),
        ]);
        manifest.nodes[0].determinism = DeterminismPolicy::External;
        manifest.nodes[0].cache = CachePolicy::Never;
        manifest.nodes[0].effects = EffectPolicy::Provider;
        let mut binding = ProviderBindingSet {
            schema_version: PROVIDER_BINDING_SCHEMA_VERSION,
            implementation_namespace: manifest.identifier.clone(),
            bindings_sha256: "0".repeat(64),
            bindings: vec![ProviderBindingClaim {
                feature_id: "COMFY-NODE-0001".to_owned(),
                node_id: manifest.nodes[0].id.clone(),
                contract_sha256: "3".repeat(64),
                transport_schema: "zed:sim-provider-transport@1".parse()?,
                materializer_schema: "zed:sim-provider-materializer@1".parse()?,
            }],
        };
        binding.bindings_sha256 = binding.canonical_bindings_sha256()?;
        manifest.provider_binding = Some(binding);
        manifest.signature.value = signing_key()?.sign_manifest(&manifest)?;
        let mut provider = ProviderPluginManifestV2 {
            schema_version: PROVIDER_MANIFEST_SCHEMA_VERSION_V2,
            component_world: PROVIDER_COMPONENT_WORLD_V2.to_owned(),
            manifest,
            streaming: ProviderStreamingContractV2 {
                methods: vec![ProviderHttpMethodV2::Post],
                maximum_headers: 8,
                maximum_header_bytes: 1_024,
                maximum_request_body_bytes: 1_024,
                maximum_response_body_bytes: 1_024,
                maximum_chunk_bytes: 256,
                maximum_ndjson_line_bytes: 256,
                maximum_wait_milliseconds: 1_000,
                maximum_uploads: 0,
                maximum_upload_body_bytes: 0,
                maximum_cost_requests: 0,
                maximum_progress_total: 100,
                uploads: false,
                cost_requests: false,
            },
            signature: ManifestSignature {
                algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
                key_id: signing_key()?.key_id().to_owned(),
                value: "0".repeat(ED25519_SIGNATURE_BYTES * 2),
            },
        };
        provider.signature.value = signing_key()?.sign_provider_manifest_v2(&provider)?;
        Ok(provider)
    }

    fn trust_policy() -> Result<PluginTrustPolicy, TrustError> {
        let signing_key = signing_key()?;
        PluginTrustPolicy::new([PluginVerificationKey::new(
            signing_key.key_id(),
            signing_key.verification_key_bytes()?,
        )?])
    }

    fn signing_key() -> Result<PluginSigningKey, PluginContractError> {
        PluginSigningKey::new("fixture.key", SIGNING_KEY)
    }

    fn reseal_authorization_value(
        value: serde_json::Value,
        sealer: &PluginAuthorizationSealer,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut sealed: SealedPluginAuthorization = serde_json::from_value(value)?;
        let payload = serde_json::to_vec(&sealed.payload)?;
        sealed.signature = encode_hex(&sealer.sign(&authorization_signing_payload(&payload))?);
        Ok(serde_json::to_vec(&sealed)?)
    }

    #[test]
    fn invalid_signatures_and_overprivileged_plugins_fail_closed()
    -> Result<(), Box<dyn std::error::Error>> {
        let requested = CapabilitySet::new([Capability::ModelHandle {
            model_id: "model.fixture".to_owned(),
        }]);
        let permissions = permission_policy(requested)?;
        let mut invalid = signed_plugin(vec![capability(CapabilityKind::Model, "model.fixture")])?;
        let replacement = if invalid.signature.value.starts_with('0') {
            "1"
        } else {
            "0"
        };
        invalid.signature.value.replace_range(..1, replacement);
        assert_eq!(
            trust_policy()?.authorize_manifest(&invalid, &permissions),
            Err(TrustError::InvalidPluginSignature)
        );
        let overprivileged = signed_plugin(vec![
            capability(CapabilityKind::Model, "model.fixture"),
            capability(CapabilityKind::Filesystem, "input-root"),
        ])?;
        assert_eq!(
            trust_policy()?.authorize_manifest(&overprivileged, &permissions),
            Err(TrustError::Permission(PermissionError::Denied(vec![
                Capability::Asset {
                    namespace: "input".to_owned(),
                    action: crate::AssetOperation::Read,
                }
            ])))
        );
        Ok(())
    }

    #[test]
    fn provider_v2_trust_requires_valid_inner_and_outer_signatures()
    -> Result<(), Box<dyn std::error::Error>> {
        let endpoint = "https://provider.invalid/v2";
        let permissions = permission_policy(CapabilitySet::new([Capability::ProviderNetwork {
            provider: "plugin.fixture".to_owned(),
            endpoint: endpoint.to_owned(),
        }]))?;
        let trust = trust_policy()?;
        let manifest = signed_provider_plugin_v2()?;
        let authorized = trust.authorize_provider_manifest_v2(&manifest, &permissions)?;

        let mut changed_streaming = manifest.clone();
        changed_streaming.streaming.maximum_response_body_bytes += 1;
        assert_eq!(
            trust.authorize_provider_manifest_v2(&changed_streaming, &permissions),
            Err(TrustError::InvalidPluginSignature)
        );

        let mut invalid_inner = manifest.clone();
        let replacement = if invalid_inner.manifest.signature.value.starts_with('f') {
            "e"
        } else {
            "f"
        };
        invalid_inner
            .manifest
            .signature
            .value
            .replace_range(0..1, replacement);
        invalid_inner.signature.value = signing_key()?.sign_provider_manifest_v2(&invalid_inner)?;
        assert_eq!(
            trust.authorize_provider_manifest_v2(&invalid_inner, &permissions),
            Err(TrustError::InvalidPluginSignature)
        );

        let mut unknown_outer_key = manifest.clone();
        unknown_outer_key.signature.key_id = "unknown.key".to_owned();
        assert!(matches!(
            trust.authorize_provider_manifest_v2(&unknown_outer_key, &permissions),
            Err(TrustError::UnknownVerificationKey(key)) if key == "unknown.key"
        ));

        let mut resigned_streaming = manifest;
        resigned_streaming.streaming.maximum_response_body_bytes += 1;
        resigned_streaming.signature.value =
            signing_key()?.sign_provider_manifest_v2(&resigned_streaming)?;
        let resigned = trust.authorize_provider_manifest_v2(&resigned_streaming, &permissions)?;
        assert_ne!(
            authorized.outer_signing_payload_sha256(),
            resigned.outer_signing_payload_sha256()
        );
        assert_eq!(authorized.provider_binding(), resigned.provider_binding());
        Ok(())
    }

    #[test]
    fn plugin_authorization_contains_only_requested_capabilities()
    -> Result<(), Box<dyn std::error::Error>> {
        let read_input = Capability::Asset {
            namespace: "input-root".to_owned(),
            action: crate::AssetOperation::Read,
        };
        let model = Capability::ModelHandle {
            model_id: "model.fixture".to_owned(),
        };
        let granted = CapabilitySet::new([read_input.clone(), model]);
        let manifest = signed_plugin(vec![capability(CapabilityKind::Model, "model.fixture")])?;
        let authorization =
            trust_policy()?.authorize_manifest(&manifest, &permission_policy(granted)?)?;
        assert_eq!(authorization.plugin_id(), "plugin.fixture");
        assert_eq!(authorization.digest_sha256(), DIGEST);
        assert_eq!(authorization.capabilities().capabilities().len(), 1);
        assert!(authorization.capabilities().require(&read_input).is_err());
        authorization.require_manifest(&manifest)?;
        let mut changed_manifest = manifest;
        changed_manifest.nodes[0].display_name = "Changed after authorization".to_owned();
        assert_eq!(
            authorization.require_manifest(&changed_manifest),
            Err(TrustError::AuthorizationManifestMismatch)
        );
        assert_eq!(
            trust_policy()?.authorize_manifest(
                &changed_manifest,
                &permission_policy(CapabilitySet::new([Capability::ModelHandle {
                    model_id: "model.fixture".to_owned(),
                }]))?
            ),
            Err(TrustError::InvalidPluginSignature)
        );
        Ok(())
    }

    #[test]
    fn sealed_plugin_authorization_rejects_tampering_and_untrusted_signers()
    -> Result<(), Box<dyn std::error::Error>> {
        let manifest = signed_plugin(vec![capability(CapabilityKind::Model, "model.fixture")])?;
        let authorization = trust_policy()?.authorize_manifest(
            &manifest,
            &permission_policy(CapabilitySet::new([Capability::ModelHandle {
                model_id: "model.fixture".to_owned(),
            }]))?,
        )?;
        let generation = PermissionPolicyGeneration::new(1)?;
        let sealer = PluginAuthorizationSealer::from_seed([7; 32], generation)?;
        let verifier = sealer.verifier()?;
        let bytes = authorization.sealed_bytes(&sealer)?;
        assert_eq!(
            PluginAuthorization::from_sealed_bytes(
                &bytes,
                &manifest,
                &verifier,
                generation,
                "profile-a",
            )?,
            authorization
        );

        let mut tampered: serde_json::Value = serde_json::from_slice(&bytes)?;
        tampered["payload"]["digest_sha256"] = serde_json::Value::String("f".repeat(64));
        let tampered = serde_json::to_vec(&tampered)?;
        assert_eq!(
            PluginAuthorization::from_sealed_bytes(
                &tampered,
                &manifest,
                &verifier,
                generation,
                "profile-a",
            ),
            Err(TrustError::InvalidSealedPluginAuthorization)
        );

        let mut expanded_grants: serde_json::Value = serde_json::from_slice(&bytes)?;
        expanded_grants["payload"]["capabilities"]["capabilities"]
            .as_array_mut()
            .ok_or("sealed capability set is not an array")?
            .push(serde_json::to_value(Capability::Asset {
                namespace: "input".to_owned(),
                action: crate::AssetOperation::Read,
            })?);
        assert_eq!(
            PluginAuthorization::from_sealed_bytes(
                &serde_json::to_vec(&expanded_grants)?,
                &manifest,
                &verifier,
                generation,
                "profile-a",
            ),
            Err(TrustError::InvalidSealedPluginAuthorization)
        );

        let mut unknown_envelope_field: serde_json::Value = serde_json::from_slice(&bytes)?;
        unknown_envelope_field["future_authority"] = serde_json::json!(true);
        assert_eq!(
            PluginAuthorization::from_sealed_bytes(
                &serde_json::to_vec(&unknown_envelope_field)?,
                &manifest,
                &verifier,
                generation,
                "profile-a",
            ),
            Err(TrustError::InvalidSealedPluginAuthorization)
        );
        let mut unknown_payload_field: serde_json::Value = serde_json::from_slice(&bytes)?;
        unknown_payload_field["payload"]["future_authority"] = serde_json::json!(true);
        assert_eq!(
            PluginAuthorization::from_sealed_bytes(
                &serde_json::to_vec(&unknown_payload_field)?,
                &manifest,
                &verifier,
                generation,
                "profile-a",
            ),
            Err(TrustError::InvalidSealedPluginAuthorization)
        );
        assert_eq!(
            PluginAuthorization::from_sealed_bytes(
                bytes
                    .get(..bytes.len().saturating_sub(1))
                    .ok_or("sealed bytes")?,
                &manifest,
                &verifier,
                generation,
                "profile-a",
            ),
            Err(TrustError::InvalidSealedPluginAuthorization)
        );

        let mut unsupported_version: serde_json::Value = serde_json::from_slice(&bytes)?;
        unsupported_version["payload"]["version"] =
            serde_json::json!(SEALED_PLUGIN_AUTHORIZATION_VERSION + 1);
        let unsupported_version = reseal_authorization_value(unsupported_version, &sealer)?;
        assert_eq!(
            PluginAuthorization::from_sealed_bytes(
                &unsupported_version,
                &manifest,
                &verifier,
                generation,
                "profile-a",
            ),
            Err(TrustError::InvalidSealedPluginAuthorization)
        );

        let mut cross_subject_payload: serde_json::Value = serde_json::from_slice(&bytes)?;
        cross_subject_payload["payload"]["plugin_id"] = serde_json::json!("plugin.other");
        let cross_subject_payload = reseal_authorization_value(cross_subject_payload, &sealer)?;
        assert_eq!(
            PluginAuthorization::from_sealed_bytes(
                &cross_subject_payload,
                &manifest,
                &verifier,
                generation,
                "profile-a",
            ),
            Err(TrustError::InvalidSealedPluginAuthorization)
        );

        let mut other_manifest = manifest.clone();
        other_manifest.identifier = "plugin.other".to_owned();
        other_manifest.signature.value = signing_key()?.sign_manifest(&other_manifest)?;
        assert_eq!(
            PluginAuthorization::from_sealed_bytes(
                &bytes,
                &other_manifest,
                &verifier,
                generation,
                "profile-a",
            ),
            Err(TrustError::AuthorizationManifestMismatch)
        );

        let other_verifier =
            PluginAuthorizationSealer::from_seed([8; 32], generation)?.verifier()?;
        assert_eq!(
            PluginAuthorization::from_sealed_bytes(
                &bytes,
                &manifest,
                &other_verifier,
                generation,
                "profile-a",
            ),
            Err(TrustError::InvalidSealedPluginAuthorization)
        );
        let next_generation = PermissionPolicyGeneration::new(2)?;
        assert_eq!(
            PluginAuthorization::from_sealed_bytes(
                &bytes,
                &manifest,
                &verifier,
                next_generation,
                "profile-a",
            ),
            Err(TrustError::InvalidSealedPluginAuthorization)
        );
        assert_eq!(
            PluginAuthorization::from_sealed_bytes(
                &bytes,
                &manifest,
                &verifier,
                generation,
                "profile-b",
            ),
            Err(TrustError::InvalidSealedPluginAuthorization)
        );
        let next_sealer = PluginAuthorizationSealer::from_seed([7; 32], next_generation)?;
        assert_eq!(
            authorization.sealed_bytes(&next_sealer),
            Err(TrustError::InvalidSealedPluginAuthorization)
        );
        assert_eq!(
            PluginAuthorizationVerifier::from_token(&verifier.to_token())?,
            verifier
        );
        for malformed in [
            "",
            "1",
            "0:0000000000000000000000000000000000000000000000000000000000000000",
            "01:0000000000000000000000000000000000000000000000000000000000000000",
            "1:00",
            "1:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        ] {
            assert_eq!(
                PluginAuthorizationVerifier::from_token(malformed),
                Err(TrustError::InvalidAuthorizationVerificationKey)
            );
        }
        Ok(())
    }

    #[test]
    fn secrets_are_checked_opaque_and_debug_redacted() {
        for invalid in ["", " secret", "secret ", "secret\nidentifier"] {
            assert_eq!(SecretId::new(invalid), Err(TrustError::InvalidSecretId));
        }
        let secret = SecretValue::new(b"do-not-log".to_vec());
        assert_eq!(format!("{secret:?}"), "SecretValue([REDACTED])");
        assert!(!format!("{secret:?}").contains("do-not-log"));
        assert_eq!(secret.expose_to(<[u8]>::len), 10);
        for invalid_json in [
            serde_json::json!(" secret"),
            serde_json::json!("secret\nidentifier"),
            serde_json::json!("x".repeat(1_025)),
        ] {
            assert!(serde_json::from_value::<SecretId>(invalid_json).is_err());
        }
    }

    #[test]
    fn providers_default_closed_and_are_profile_and_grant_scoped()
    -> Result<(), Box<dyn std::error::Error>> {
        let secret_id = SecretId::new("provider-key")?;
        assert_eq!(
            ProviderPolicy::default().authorize(
                "profile-a",
                "plugin.fixture",
                "fixture",
                "https://fixture.invalid/v1/generate",
                Some(&secret_id)
            ),
            Err(TrustError::ProviderDisabled)
        );
        let offline = ProviderPolicy::new(
            "profile-a",
            ProviderMode::Offline,
            [ProviderEndpoint::new(
                "fixture",
                "https://fixture.invalid/v1/generate",
            )?],
            [CredentialScope::new(
                "profile-a",
                "plugin.fixture",
                "fixture",
                secret_id.clone(),
            )?],
        )?;
        assert_eq!(
            offline.authorize(
                "profile-a",
                "plugin.fixture",
                "fixture",
                "https://fixture.invalid/v1/generate",
                Some(&secret_id),
            ),
            Err(TrustError::Offline)
        );
        let enabled = ProviderPolicy::new(
            "profile-a",
            ProviderMode::Enabled,
            [ProviderEndpoint::new(
                "fixture",
                "https://fixture.invalid/v1/generate",
            )?],
            [CredentialScope::new(
                "profile-a",
                "plugin.fixture",
                "fixture",
                secret_id.clone(),
            )?],
        )?;
        assert_eq!(
            enabled.authorize(
                "profile-b",
                "plugin.fixture",
                "fixture",
                "https://fixture.invalid/v1/generate",
                Some(&secret_id),
            ),
            Err(TrustError::ProviderProfileMismatch)
        );
        assert_eq!(
            enabled.authorize(
                "profile-a",
                "plugin.fixture",
                "fixture",
                "https://fixture.invalid/v1/other",
                None,
            ),
            Err(TrustError::ProviderEndpointDenied)
        );
        assert_eq!(
            enabled
                .authorize(
                    "profile-a",
                    "plugin.fixture",
                    "fixture",
                    "https://fixture.invalid/v1/generate",
                    Some(&secret_id),
                )?
                .secret_id(),
            Some(&secret_id)
        );
        assert!(ProviderEndpoint::new("fixture", "http://127.0.0.1:8188/v1/generate").is_ok());
        for invalid_endpoint in [
            "/v1/generate",
            "http://provider.invalid/v1/generate",
            "https://user@provider.invalid/v1/generate",
            "https://provider.invalid/v1/generate?token=value",
            "https://provider.invalid/v1/generate#fragment",
            "https://PROVIDER.invalid/v1/generate",
            "https://provider.invalid:443/v1/generate",
        ] {
            assert_eq!(
                ProviderEndpoint::new("fixture", invalid_endpoint),
                Err(TrustError::InvalidProviderEndpoint)
            );
        }
        assert_eq!(
            enabled.authorize(
                "profile-a",
                "plugin.other",
                "fixture",
                "https://fixture.invalid/v1/generate",
                Some(&secret_id),
            ),
            Err(TrustError::MissingSecretGrant)
        );
        for invalid_identifier in ["Fixture", "fixture..provider", "-fixture", "fixture-"] {
            assert_eq!(
                ProviderId::new(invalid_identifier),
                Err(TrustError::InvalidProviderId)
            );
            assert_eq!(
                SecretId::new(invalid_identifier),
                Err(TrustError::InvalidSecretId)
            );
        }
        for invalid in [" fixture", "fixture ", "fixture\n", &"x".repeat(1_025)] {
            assert_eq!(
                ProviderEndpoint::new(invalid, "https://fixture.invalid/v1/generate"),
                Err(TrustError::InvalidProviderId)
            );
            assert_eq!(
                ProviderEndpoint::new("fixture", invalid),
                Err(TrustError::InvalidProviderEndpoint)
            );
            assert_eq!(
                ProviderPolicy::new(
                    invalid,
                    ProviderMode::Enabled,
                    std::iter::empty(),
                    std::iter::empty(),
                ),
                Err(TrustError::InvalidProviderProfile)
            );
        }
        Ok(())
    }

    #[test]
    fn provider_cost_acceptance_is_sealed_exact_bounded_and_expiring()
    -> Result<(), Box<dyn std::error::Error>> {
        let origin = Instant::now();
        let issuer = ProviderCostAcceptanceIssuer::from_seed([7; 32], origin)?;
        let verifier = issuer.verifier()?;
        let price = ProviderPriceBound::new("USD", 25_000)?;
        let scope_for = |prompt_sha256: &str,
                         attempt_id: &str,
                         node_id: &str,
                         request_ordinal: u32,
                         request_sha256: &str,
                         provider_binding_sha256: &str| {
            ProviderCostAcceptanceScope::new(
                "principal-a",
                "00000000-0000-0000-0000-000000000001",
                "00000000-0000-0000-0000-000000000002",
                prompt_sha256,
                attempt_id,
                node_id,
                request_ordinal,
                request_sha256,
                "plugin.fixture",
                DIGEST,
                provider_binding_sha256,
                "fixture",
                "https://fixture.invalid/v1/generate",
                price.clone(),
            )
        };
        let prompt_sha256 = "c".repeat(64);
        let request_sha256 = "d".repeat(64);
        let binding_sha256 = "a".repeat(64);
        let scope = scope_for(
            &prompt_sha256,
            "00000000-0000-0000-0000-000000000003",
            "node.fixture",
            7,
            &request_sha256,
            &binding_sha256,
        )?;
        let nonce = ProviderCostNonce::new([9; 32])?;
        let acceptance = issuer.issue(
            scope.clone(),
            origin + Duration::from_secs(1),
            origin + Duration::from_secs(31),
            nonce,
        )?;

        verifier.verify(&acceptance, &scope, origin + Duration::from_secs(1))?;
        assert_eq!(acceptance.nonce(), nonce);
        assert_eq!(
            format!("{acceptance:?}"),
            "ProviderCostAcceptance([SEALED])"
        );
        assert_eq!(
            verifier.verify(&acceptance, &scope, origin + Duration::from_secs(31)),
            Err(TrustError::ExpiredProviderCostAcceptance)
        );
        for wrong_scope in [
            scope_for(
                &"e".repeat(64),
                scope.attempt_id(),
                scope.node_id(),
                scope.request_ordinal(),
                scope.request_sha256(),
                scope.provider_binding_sha256(),
            )?,
            scope_for(
                scope.prompt_sha256(),
                "00000000-0000-0000-0000-000000000004",
                scope.node_id(),
                scope.request_ordinal(),
                scope.request_sha256(),
                scope.provider_binding_sha256(),
            )?,
            scope_for(
                scope.prompt_sha256(),
                scope.attempt_id(),
                "node.other",
                scope.request_ordinal(),
                scope.request_sha256(),
                scope.provider_binding_sha256(),
            )?,
            scope_for(
                scope.prompt_sha256(),
                scope.attempt_id(),
                scope.node_id(),
                8,
                scope.request_sha256(),
                scope.provider_binding_sha256(),
            )?,
            scope_for(
                scope.prompt_sha256(),
                scope.attempt_id(),
                scope.node_id(),
                scope.request_ordinal(),
                &"f".repeat(64),
                scope.provider_binding_sha256(),
            )?,
            scope_for(
                scope.prompt_sha256(),
                scope.attempt_id(),
                scope.node_id(),
                scope.request_ordinal(),
                scope.request_sha256(),
                &"b".repeat(64),
            )?,
        ] {
            assert_eq!(
                verifier.verify(&acceptance, &wrong_scope, origin + Duration::from_secs(2)),
                Err(TrustError::InvalidProviderCostAcceptance)
            );
        }

        let mut forged = acceptance;
        forged.signature[0] ^= 1;
        assert_eq!(
            verifier.verify(&forged, &scope, origin + Duration::from_secs(2)),
            Err(TrustError::InvalidProviderCostAcceptance)
        );
        assert_eq!(
            ProviderCostNonce::new([0; 32]),
            Err(TrustError::InvalidProviderCostAcceptance)
        );
        assert_eq!(
            ProviderPriceBound::new("usd", 25_000),
            Err(TrustError::InvalidProviderPriceBound)
        );
        assert!(matches!(
            issuer.issue(
                scope,
                origin,
                origin + MAX_PROVIDER_COST_ACCEPTANCE_LIFETIME + Duration::from_millis(1),
                nonce,
            ),
            Err(TrustError::InvalidProviderCostAcceptance)
        ));
        Ok(())
    }

    #[test]
    fn provider_result_receipt_is_sealed_scoped_bounded_and_domain_separated()
    -> Result<(), Box<dyn std::error::Error>> {
        let origin = Instant::now();
        let issuer = ProviderResultReceiptIssuer::from_seed([17; 32], origin)?;
        let verifier = issuer.verifier()?;
        let identity_for = |node_id: &str, request_ordinal: u32, request_sha256: &str| {
            ProviderInvocationIdentity::new(
                "principal-a",
                "00000000-0000-0000-0000-000000000001",
                "00000000-0000-0000-0000-000000000002",
                "c".repeat(64),
                "00000000-0000-0000-0000-000000000003",
                node_id,
                request_ordinal,
                request_sha256,
                "plugin.fixture",
                DIGEST,
                "a".repeat(64),
                "fixture",
                "https://fixture.invalid/v1/generate",
            )
        };
        let identity = identity_for("node.fixture", 7, &"d".repeat(64))?;
        let result_sha256 = "e".repeat(64);
        let nonce = ProviderResultNonce::new([19; 32])?;
        let receipt = issuer.issue(
            identity.clone(),
            result_sha256.clone(),
            origin + Duration::from_secs(1),
            origin + Duration::from_secs(31),
            nonce,
        )?;
        let encoded = receipt.to_bytes()?;
        assert!(encoded.len() <= MAX_PROVIDER_RESULT_RECEIPT_BYTES);
        let decoded = ProviderResultReceipt::from_bytes(&encoded)?;
        let verified = verifier.verify(
            &decoded,
            &identity,
            &result_sha256,
            origin + Duration::from_secs(2),
        )?;
        assert_eq!(verified.result_sha256(), result_sha256);
        assert_eq!(verified.nonce(), nonce);
        assert_eq!(decoded.identity(), &identity);
        assert_eq!(decoded.result_sha256(), result_sha256);
        assert_eq!(format!("{decoded:?}"), "ProviderResultReceipt([SEALED])");
        assert_eq!(
            verifier.verify(
                &decoded,
                &identity,
                &result_sha256,
                origin + Duration::from_secs(31),
            ),
            Err(TrustError::ExpiredProviderResultReceipt)
        );

        let wrong_identity = identity_for("node.other", 7, &"d".repeat(64))?;
        assert_eq!(
            verifier.verify(
                &decoded,
                &wrong_identity,
                &result_sha256,
                origin + Duration::from_secs(2),
            ),
            Err(TrustError::InvalidProviderResultReceipt)
        );
        let wrong_ordinal = identity_for("node.fixture", 8, &"d".repeat(64))?;
        assert_eq!(
            verifier.verify(
                &decoded,
                &wrong_ordinal,
                &result_sha256,
                origin + Duration::from_secs(2),
            ),
            Err(TrustError::InvalidProviderResultReceipt)
        );
        let wrong_request = identity_for("node.fixture", 7, &"f".repeat(64))?;
        assert_eq!(
            verifier.verify(
                &decoded,
                &wrong_request,
                &result_sha256,
                origin + Duration::from_secs(2),
            ),
            Err(TrustError::InvalidProviderResultReceipt)
        );
        assert_eq!(
            verifier.verify(
                &decoded,
                &identity,
                &"b".repeat(64),
                origin + Duration::from_secs(2),
            ),
            Err(TrustError::InvalidProviderResultReceipt)
        );
        let foreign_verifier =
            ProviderResultReceiptIssuer::from_seed([18; 32], origin)?.verifier()?;
        assert_eq!(
            foreign_verifier.verify(
                &decoded,
                &identity,
                &result_sha256,
                origin + Duration::from_secs(2),
            ),
            Err(TrustError::InvalidProviderResultReceipt)
        );

        let cost_issuer = ProviderCostAcceptanceIssuer::from_seed([17; 32], origin)?;
        let cost_scope = ProviderCostAcceptanceScope {
            identity: identity.clone(),
            price_bound: ProviderPriceBound::new("USD", 25_000)?,
        };
        let cost_acceptance = cost_issuer.issue(
            cost_scope,
            origin + Duration::from_secs(1),
            origin + Duration::from_secs(31),
            ProviderCostNonce::new([19; 32])?,
        )?;
        let mut wrong_domain = ProviderResultReceipt::from_bytes(&encoded)?;
        wrong_domain.signature = cost_acceptance.signature;
        assert_eq!(
            verifier.verify(
                &wrong_domain,
                &identity,
                &result_sha256,
                origin + Duration::from_secs(2),
            ),
            Err(TrustError::InvalidProviderResultReceipt)
        );
        assert_eq!(
            ProviderResultNonce::new([0; 32]),
            Err(TrustError::InvalidProviderResultReceipt)
        );
        assert!(matches!(
            ProviderResultReceipt::from_bytes(&vec![b'x'; MAX_PROVIDER_RESULT_RECEIPT_BYTES + 1]),
            Err(TrustError::ProviderResultReceiptTooLarge)
        ));
        assert!(matches!(
            issuer.issue(
                identity,
                result_sha256,
                origin,
                origin + MAX_PROVIDER_RESULT_RECEIPT_LIFETIME + Duration::from_millis(1),
                nonce,
            ),
            Err(TrustError::InvalidProviderResultReceipt)
        ));
        Ok(())
    }

    #[test]
    fn unsafe_api_navigation_ffi_and_pickle_fail_before_effects()
    -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            NativeApiExposure::new(
                IpAddr::V4(Ipv4Addr::UNSPECIFIED),
                false,
                std::iter::empty(),
                false,
                RemoteExposureApproval::LoopbackOnly,
            ),
            Err(TrustError::UnsafeApiExposure)
        );
        for unsafe_origin in [
            "https://",
            "http://localhost.evil.com",
            "https://example.com/path",
        ] {
            assert_eq!(
                NativeApiExposure::new(
                    IpAddr::V4(Ipv4Addr::UNSPECIFIED),
                    true,
                    [unsafe_origin.to_owned()],
                    false,
                    RemoteExposureApproval::Approved,
                ),
                Err(TrustError::UnsafeApiExposure)
            );
        }
        let exposure = NativeApiExposure::new(
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            true,
            [
                "https://example.com".to_owned(),
                "http://localhost:3000".to_owned(),
            ],
            false,
            RemoteExposureApproval::Approved,
        )?;
        assert_eq!(
            exposure.allowed_origins(),
            &BTreeSet::from([
                "http://localhost:3000".to_owned(),
                "https://example.com".to_owned(),
            ])
        );
        let navigation = ExternalNavigationPolicy::new(["https".to_owned()], true)?;
        assert_eq!(
            navigation.authorize("javascript:alert(1)", true),
            Err(TrustError::NavigationDenied)
        );
        assert_eq!(
            navigation.authorize("https://example.com", false),
            Err(TrustError::NavigationDenied)
        );
        for malformed in ["https:", "https://", "https://user@example.com"] {
            assert_eq!(
                navigation.authorize(malformed, true),
                Err(TrustError::NavigationDenied)
            );
        }
        assert_eq!(
            ExternalNavigationPolicy::new(["javascript".to_owned()], true),
            Err(TrustError::InvalidNavigationPolicy)
        );

        let contract =
            NativeFfiContract::new("codec", DIGEST, "1", ["decode".to_owned()], "sim_media")?;
        let registry = NativeFfiRegistry::new([contract])?;
        assert_eq!(
            registry.authorize("codec", DIGEST, "2", &BTreeSet::from(["decode".to_owned()])),
            Err(TrustError::UncertifiedFfi)
        );
        assert!(
            NativeFfiRegistry::default()
                .authorize("codec", DIGEST, "1", &BTreeSet::from(["decode".to_owned()]))
                .is_err()
        );

        let executable = b"cos\nsystem\n.";
        assert!(matches!(
            parse_restricted_pickle(executable, &ParserLimits::default()),
            Err(RestrictedPickleError::ForbiddenTarget { .. })
        ));
        assert!(IpAddr::from_str("127.0.0.1").map(|address| address.is_loopback())?);
        Ok(())
    }

    #[test]
    fn every_cataloged_provider_endpoint_requires_an_explicit_grant()
    -> Result<(), Box<dyn std::error::Error>> {
        let catalog = include_str!(
            "../../../crates/sim_test_support/parity/catalogs/backend-external-services.csv"
        );
        let policy = ProviderPolicy::new(
            "profile-a",
            ProviderMode::Enabled,
            std::iter::empty(),
            std::iter::empty(),
        )?;
        let mut count = 0;
        for line in catalog.lines().skip(1) {
            let mut columns = line.splitn(4, ',');
            let _method = columns.next().ok_or("method column")?;
            let endpoint = columns.next().ok_or("path column")?;
            let provider = columns.next().ok_or("provider column")?;
            let endpoint = format!("https://catalog.invalid{endpoint}");
            assert!(matches!(
                policy.authorize("profile-a", "plugin.fixture", provider, &endpoint, None,),
                Err(TrustError::ProviderEndpointDenied | TrustError::InvalidProviderEndpoint)
            ));
            count += 1;
        }
        assert_eq!(count, 226);
        Ok(())
    }

    fn workspace_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
        Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?
            .to_path_buf())
    }

    fn source_digest(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
        Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
    }

    fn write_trust_validation_artifact(
        root: &Path,
        cases: &BTreeMap<&str, bool>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let artifact_directory = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("target"))
            .join("sim-parity");
        let artifact_path = artifact_directory.join("val-runtime-trust-001.json");
        match fs::remove_file(&artifact_path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let fixture_paths = [
            "crates/sim_runtime/src/permissions.rs",
            "crates/sim_runtime/src/trust.rs",
            "crates/sim_runtime/src/native_ffi_rocm.rs",
            "crates/sim_runtime/Cargo.toml",
            "crates/sim_runtime/src/settings.rs",
            "crates/sim_plugin_sdk/Cargo.toml",
            "crates/sim_plugin_sdk/src/sim_plugin_sdk.rs",
            "crates/sim_plugin_host/Cargo.toml",
            "crates/sim_worker/src/plugin_runtime.rs",
            "crates/settings_content/src/settings_content.rs",
            "crates/sim_model/src/artifact_index.rs",
            "crates/sim_model/src/formats.rs",
            "crates/sim_model/src/restricted_pickle.rs",
            "crates/sim_backend_rocm/src/loader.rs",
            "crates/sim_backend_rocm/abi/symbols-v1.json",
            "crates/sim_backend_rocm/abi/reviewed-bindings-v1.txt",
            "crates/sim_backend_rocm/abi/verify-completion-evidence.sh",
            "crates/sim_backend_rocm/build.rs",
            "script/package-sim-backend-rocm",
            "nix/sim-backends/rocm/package-policy.json",
            "crates/credentials_provider/src/credentials_provider.rs",
            "crates/sim_test_support/parity/catalogs/backend-external-services.csv",
            "crates/sim_test_support/parity/catalogs/authoritative-ownership.csv",
            "crates/sim_test_support/parity/generate_ownership_catalog.py",
            "crates/sim_test_support/parity/ownership-policy.json",
        ];
        let mut fixture_digests = BTreeMap::new();
        for relative_path in fixture_paths {
            fixture_digests.insert(relative_path, source_digest(&root.join(relative_path))?);
        }
        let failures = cases.values().filter(|passed| !**passed).count();
        let artifact = json!({
            "validation": "VAL-RUNTIME-TRUST-001",
            "task": "comfy-parity-trust-foundation",
            "environment": {
                "backend": "native-rust",
                "platform": std::env::consts::OS,
                "external_processes": 0,
                "network_requests": 0,
            },
            "authoritative_owners": {
                "authorization": "sim_runtime::PermissionPolicy",
                "artifact_paths": "sim_model::ArtifactRoot",
                "archive_model_parsing": "sim_model::formats",
                "credentials": "credentials_provider::CredentialsProvider",
                "model_content_trust": "sim_model::restricted_pickle",
                "plugin_verification": "sim_runtime::PluginTrustPolicy",
            },
            "fixture_digests": fixture_digests,
            "cases": cases,
            "passes": cases.len().saturating_sub(failures),
            "failures": failures,
            "skips": 0,
        });
        fs::create_dir_all(&artifact_directory)?;
        let mut bytes = serde_json::to_vec_pretty(&artifact)?;
        bytes.push(b'\n');
        fs::write(artifact_path, bytes)?;
        Ok(())
    }

    #[test]
    fn val_runtime_trust_001() -> Result<(), Box<dyn std::error::Error>> {
        let root = workspace_root()?;
        let read_input = Capability::Asset {
            namespace: "input-root".to_owned(),
            action: crate::AssetOperation::Read,
        };
        let model = Capability::ModelHandle {
            model_id: "model.fixture".to_owned(),
        };
        let permissions = permission_policy(CapabilitySet::new([read_input.clone(), model]))?;
        let manifest = signed_plugin(vec![capability(CapabilityKind::Model, "model.fixture")])?;
        let authorization = trust_policy()?.authorize_manifest(&manifest, &permissions)?;

        let mut invalid_signature = manifest.clone();
        let replacement = if invalid_signature.signature.value.starts_with('0') {
            "1"
        } else {
            "0"
        };
        invalid_signature
            .signature
            .value
            .replace_range(..1, replacement);

        let policy_generation = permissions.generation();
        let authorization_sealer =
            PluginAuthorizationSealer::from_seed([9; 32], policy_generation)?;
        let authorization_verifier = authorization_sealer.verifier()?;
        let sealed_authorization = authorization.sealed_bytes(&authorization_sealer)?;

        let mut unknown_envelope: serde_json::Value =
            serde_json::from_slice(&sealed_authorization)?;
        unknown_envelope["future_authority"] = serde_json::json!(true);
        let unknown_envelope = serde_json::to_vec(&unknown_envelope)?;
        let mut unsupported_version: serde_json::Value =
            serde_json::from_slice(&sealed_authorization)?;
        unsupported_version["payload"]["version"] =
            serde_json::json!(SEALED_PLUGIN_AUTHORIZATION_VERSION + 1);
        let unsupported_version =
            reseal_authorization_value(unsupported_version, &authorization_sealer)?;
        let mut expanded_grants: serde_json::Value = serde_json::from_slice(&sealed_authorization)?;
        expanded_grants["payload"]["capabilities"]["capabilities"]
            .as_array_mut()
            .ok_or("sealed capability set is not an array")?
            .push(serde_json::to_value(Capability::Asset {
                namespace: "input".to_owned(),
                action: crate::AssetOperation::Read,
            })?);
        let expanded_grants = serde_json::to_vec(&expanded_grants)?;
        let mut cross_subject_manifest = manifest.clone();
        cross_subject_manifest.identifier = "plugin.other".to_owned();
        cross_subject_manifest.signature.value =
            signing_key()?.sign_manifest(&cross_subject_manifest)?;
        let next_generation = PermissionPolicyGeneration::new(policy_generation.get() + 1)?;

        let provider_secret = SecretId::new("provider-key")?;
        let enabled_provider = ProviderPolicy::new(
            "profile-a",
            ProviderMode::Enabled,
            [ProviderEndpoint::new(
                "fixture",
                "https://fixture.invalid/v1/generate",
            )?],
            [CredentialScope::new(
                "profile-a",
                "plugin.fixture",
                "fixture",
                provider_secret.clone(),
            )?],
        )?;

        let navigation = ExternalNavigationPolicy::new(["https".to_owned()], true)?;
        let ffi_contract =
            NativeFfiContract::new("codec", DIGEST, "1", ["decode".to_owned()], "sim_media")?;
        let ffi_registry = NativeFfiRegistry::new([ffi_contract])?;

        let artifact_directory = tempfile::tempdir()?;
        let artifact_root = sim_model::ArtifactRoot::canonical(
            "models",
            "checkpoints",
            artifact_directory.path(),
            ["safetensors"],
        )?;

        let runtime_trust_source =
            fs::read_to_string(root.join("crates/sim_runtime/src/trust.rs"))?;
        let runtime_trust_production = runtime_trust_source
            .split_once("#[cfg(test)]")
            .map(|(production, _)| production)
            .ok_or("trust test boundary is unavailable")?;
        let runtime_rocm_source =
            fs::read_to_string(root.join("crates/sim_runtime/src/native_ffi_rocm.rs"))?;
        let backend_rocm_loader =
            fs::read_to_string(root.join("crates/sim_backend_rocm/src/loader.rs"))?;
        let backend_rocm_packager =
            fs::read_to_string(root.join("script/package-sim-backend-rocm"))?;
        let backend_rocm_package_policy =
            fs::read_to_string(root.join("nix/sim-backends/rocm/package-policy.json"))?;
        let signature_assertion_marker = ["signature", "verified"].join("_");
        let runtime_permissions_source =
            fs::read_to_string(root.join("crates/sim_runtime/src/permissions.rs"))?;
        let runtime_settings_source =
            fs::read_to_string(root.join("crates/sim_runtime/src/settings.rs"))?;
        let settings_content_source =
            fs::read_to_string(root.join("crates/settings_content/src/settings_content.rs"))?;
        let sdk_manifest_source =
            fs::read_to_string(root.join("crates/sim_plugin_sdk/Cargo.toml"))?;
        let sdk_source =
            fs::read_to_string(root.join("crates/sim_plugin_sdk/src/sim_plugin_sdk.rs"))?;
        let runtime_manifest_source =
            fs::read_to_string(root.join("crates/sim_runtime/Cargo.toml"))?;
        let host_manifest_source =
            fs::read_to_string(root.join("crates/sim_plugin_host/Cargo.toml"))?;
        let runtime_root_source =
            fs::read_to_string(root.join("crates/sim_runtime/src/sim_runtime.rs"))?;
        let artifact_source =
            fs::read_to_string(root.join("crates/sim_model/src/artifact_index.rs"))?;
        let model_formats_source =
            fs::read_to_string(root.join("crates/sim_model/src/formats.rs"))?;
        let credentials_source = fs::read_to_string(
            root.join("crates/credentials_provider/src/credentials_provider.rs"),
        )?;
        let ownership_catalog = fs::read_to_string(
            root.join("crates/sim_test_support/parity/catalogs/authoritative-ownership.csv"),
        )?;

        let cases = BTreeMap::from([
            (
                "authorization_envelope_rejects_unknown_version_and_truncation",
                PluginAuthorization::from_sealed_bytes(
                    &unknown_envelope,
                    &manifest,
                    &authorization_verifier,
                    policy_generation,
                    "profile-a",
                ) == Err(TrustError::InvalidSealedPluginAuthorization)
                    && PluginAuthorization::from_sealed_bytes(
                        &unsupported_version,
                        &manifest,
                        &authorization_verifier,
                        policy_generation,
                        "profile-a",
                    ) == Err(TrustError::InvalidSealedPluginAuthorization)
                    && PluginAuthorization::from_sealed_bytes(
                        sealed_authorization
                            .get(..sealed_authorization.len().saturating_sub(1))
                            .ok_or("sealed authorization bytes")?,
                        &manifest,
                        &authorization_verifier,
                        policy_generation,
                        "profile-a",
                    ) == Err(TrustError::InvalidSealedPluginAuthorization),
            ),
            (
                "authorization_is_bound_to_subject_profile_generation_and_exact_grants",
                PluginAuthorization::from_sealed_bytes(
                    &sealed_authorization,
                    &cross_subject_manifest,
                    &authorization_verifier,
                    policy_generation,
                    "profile-a",
                ) == Err(TrustError::AuthorizationManifestMismatch)
                    && PluginAuthorization::from_sealed_bytes(
                        &sealed_authorization,
                        &manifest,
                        &authorization_verifier,
                        policy_generation,
                        "profile-b",
                    ) == Err(TrustError::InvalidSealedPluginAuthorization)
                    && PluginAuthorization::from_sealed_bytes(
                        &sealed_authorization,
                        &manifest,
                        &authorization_verifier,
                        next_generation,
                        "profile-a",
                    ) == Err(TrustError::InvalidSealedPluginAuthorization)
                    && PluginAuthorization::from_sealed_bytes(
                        &expanded_grants,
                        &manifest,
                        &authorization_verifier,
                        policy_generation,
                        "profile-a",
                    ) == Err(TrustError::InvalidSealedPluginAuthorization),
            ),
            (
                "authorization_verifier_token_is_canonical_and_generation_bound",
                PluginAuthorizationVerifier::from_token(&authorization_verifier.to_token())
                    .is_ok_and(|parsed| parsed == authorization_verifier)
                    && PluginAuthorizationVerifier::from_token(
                        "01:0000000000000000000000000000000000000000000000000000000000000000",
                    ) == Err(TrustError::InvalidAuthorizationVerificationKey)
                    && PluginAuthorizationVerifier::from_token(
                        "0:0000000000000000000000000000000000000000000000000000000000000000",
                    ) == Err(TrustError::InvalidAuthorizationVerificationKey),
            ),
            (
                "artifact_root_is_sole_generic_path_validator",
                artifact_root
                    .resolve_for_create("nested/model.safetensors")
                    .is_ok()
                    && artifact_root.resolve_for_create("../escape").is_err()
                    && artifact_source.contains("pub fn resolve_existing")
                    && artifact_source.contains("pub fn resolve_for_create")
                    && !runtime_root_source.contains("secure_paths")
                    && ownership_catalog.contains(
                        "secure_root_path_validation,sim_model::ArtifactRoot",
                    ),
            ),
            (
                "archive_validation_has_a_focused_model_owner",
                model_formats_source.contains("fn canonical_archive_path")
                    && model_formats_source.contains("ArchiveEntry")
                    && ownership_catalog
                        .contains("model_archive_content_validation,sim_model::formats"),
            ),
            (
                "canonical_model_pickle_policy_rejects_execution",
                matches!(
                    parse_restricted_pickle(b"cos\nsystem\n.", &ParserLimits::default()),
                    Err(RestrictedPickleError::ForbiddenTarget { .. })
                ) && !runtime_trust_production.contains("ALLOWED_PICKLE_GLOBALS"),
            ),
            (
                "credentials_remain_identifier_only_at_policy_boundary",
                credentials_source.contains("pub trait CredentialsProvider")
                    && runtime_trust_source.contains("pub struct SecretId")
                    && !runtime_trust_source.contains("Serialize)]\npub struct SecretValue")
                    && ownership_catalog.contains(
                        "credential_secret_storage,credentials_provider::CredentialsProvider",
                    ),
            ),
            (
                "ffi_certification_is_registry_issued",
                ffi_registry
                    .authorize("codec", DIGEST, "1", &BTreeSet::from(["decode".to_owned()]))
                    .is_ok()
                    && NativeFfiRegistry::default()
                        .authorize("codec", DIGEST, "1", &BTreeSet::from(["decode".to_owned()]))
                        .is_err()
                    && !runtime_trust_production.contains("certified: bool")
                    && ownership_catalog
                        .contains("native_ffi_certification,sim_runtime::NativeFfiRegistry"),
            ),
            (
                "rocm_ffi_adapter_consumes_registry_authority_without_self_certification",
                runtime_rocm_source.contains("cancellation: &CancellationToken")
                    && runtime_rocm_source
                        .contains("inspect_elf64_dynamic_contract(image.bytes(), 62, cancellation)")
                    && runtime_rocm_source.contains("exactly one PT_DYNAMIC")
                    && runtime_rocm_source.contains("TRUSTED_SYSTEM_ELF_DEPENDENCIES")
                    && runtime_rocm_source.contains("CancellableChunks")
                    && runtime_rocm_source
                        .contains("exact_soname_and_system_only_dependencies_prevent_ambient_binding")
                    && runtime_rocm_source
                        .contains("cancellation_is_injected_during_read_hash_parse_and_snapshot")
                    && runtime_rocm_source.contains("let certificate = registry")
                    && runtime_rocm_source.contains(".authorize(")
                    && runtime_rocm_source.contains(".seal(")
                    && runtime_rocm_source
                        .split_once("#[cfg(test)]\nmod tests")
                        .map_or(runtime_rocm_source.as_str(), |(production, _)| production)
                        .matches("NativeFfiContract::new")
                        .count()
                        == 1
                    && backend_rocm_loader.contains("validate_sealed_memfd")
                    && backend_rocm_loader.contains("F_GET_SEALS")
                    && backend_rocm_loader.contains("validate_signed_package")
                    && backend_rocm_loader.contains("validate_tree_membership")
                    && backend_rocm_loader.contains("validate_coverage")
                    && backend_rocm_loader.contains("compiled reviewed policy")
                    && !backend_rocm_loader.contains("NativeFfiRegistry::")
                    && backend_rocm_package_policy
                        .contains("sim_runtime-native-rust-ed25519")
                    && !backend_rocm_packager.contains("SIM_ROCM_SIGNATURE_VERIFIER")
                    && backend_rocm_packager.contains("package-coverage.sha256"),
            ),
            (
                "rocm_package_trust_is_native_domain_bound_and_plugin_separate",
                runtime_trust_source.contains("pub struct RocmPackageVerificationKey")
                    && runtime_trust_source.contains("ROCM_PACKAGE_SIGNATURE_DOMAIN")
                    && runtime_trust_source.contains("canonical_receipt.push(b'\\n')")
                    && runtime_rocm_source.contains("NativeRocmPackageVerifier")
                    && runtime_rocm_source
                        .find("verify_signed_package_root")
                        .zip(runtime_rocm_source.find("parse_rocm_ffi_contract_catalog"))
                        .is_some_and(|(verification, catalog)| verification < catalog)
                    && runtime_settings_source.contains("project_rocm_package_settings")
                    && runtime_settings_source.contains("RocmPackageVerificationKey::new")
                    && runtime_settings_source.contains("is_private_signing_setting")
                    && ownership_catalog.contains(
                        "rocm_package_trust_and_contract_mapping,sim_runtime::RocmPackageVerificationKey",
                    ),
            ),
            (
                "navigation_requires_scheme_and_user_gesture",
                navigation.authorize("https://example.com", true).is_ok()
                    && navigation.authorize("https://example.com", false).is_err()
                    && navigation.authorize("javascript:alert(1)", true).is_err()
                    && navigation.authorize("https:", true).is_err()
                    && navigation.authorize("https://", true).is_err()
                    && ownership_catalog.contains(
                        "external_navigation_authorization,sim_runtime::ExternalNavigationPolicy",
                    ),
            ),
            (
                "permission_authorization_is_exact_and_sealed",
                authorization.capabilities().capabilities().len() == 1
                    && authorization
                        .capabilities()
                        .require(&read_input)
                        .is_err()
                    && runtime_permissions_source.contains("pub struct PermissionPolicy")
                    && runtime_permissions_source.contains("pub struct AuthorizedCapabilities")
                    && ownership_catalog.contains(
                        "permission_capability_domain,sim_runtime::PermissionPolicy and Capability",
                    ),
            ),
            (
                "plugin_signing_tooling_is_feature_gated_and_private_seeds_are_zeroized",
                sdk_manifest_source.contains("default = []")
                    && sdk_manifest_source
                        .contains("signing-tooling = [\"dep:ring\", \"dep:zeroize\"]")
                    && sdk_manifest_source
                        .contains("ring = { workspace = true, optional = true }")
                    && sdk_source.contains(
                        "#[cfg(any(feature = \"signing-tooling\", test))]\npub struct PluginSigningKey",
                    )
                    && sdk_source.contains("seed: Zeroizing<[")
                    && runtime_trust_source.contains("seed: Zeroizing<[")
                    && runtime_trust_source.contains("pub struct SecretValue(Zeroizing<Vec<u8>>)")
                    && runtime_manifest_source
                        .split_once("[dev-dependencies]")
                        .is_some_and(|(_, dependencies)| {
                            dependencies.contains(
                                "sim_plugin_sdk = { workspace = true, features = [\"signing-tooling\"] }",
                            )
                        })
                    && host_manifest_source
                        .split_once("[dev-dependencies]")
                        .is_some_and(|(_, dependencies)| {
                            dependencies.contains(
                                "sim_plugin_sdk = { workspace = true, features = [\"signing-tooling\"] }",
                            )
                        }),
            ),
            (
                "plugin_signature_is_verified_not_asserted",
                trust_policy()?.authorize_manifest(&invalid_signature, &permissions)
                    == Err(TrustError::InvalidPluginSignature)
                    && !runtime_trust_production.contains(&signature_assertion_marker)
                    && ownership_catalog.contains(
                        "plugin_signature_verification,sim_runtime::PluginTrustPolicy",
                    ),
            ),
            (
                "provider_policy_defaults_closed_and_is_profile_scoped",
                ProviderPolicy::default().authorize(
                    "profile-a",
                    "plugin.fixture",
                    "fixture",
                    "https://fixture.invalid/v1/generate",
                    None,
                )
                    == Err(TrustError::ProviderDisabled)
                    && enabled_provider.authorize(
                        "profile-b",
                        "plugin.fixture",
                        "fixture",
                        "https://fixture.invalid/v1/generate",
                        Some(&provider_secret),
                    ) == Err(TrustError::ProviderProfileMismatch)
                    && enabled_provider.authorize(
                        "profile-a",
                        "plugin.other",
                        "fixture",
                        "https://fixture.invalid/v1/generate",
                        Some(&provider_secret),
                    ) == Err(TrustError::MissingSecretGrant)
                    && ProviderEndpoint::new(
                        "fixture",
                        "http://provider.invalid/v1/generate",
                    ) == Err(TrustError::InvalidProviderEndpoint)
                    && ProviderEndpoint::new(
                        "fixture",
                        "https://PROVIDER.invalid/v1/generate",
                    ) == Err(TrustError::InvalidProviderEndpoint)
                    && ownership_catalog.contains(
                        "provider_request_authorization,sim_runtime::ProviderPolicy",
                    ),
            ),
            (
                "remote_api_exposure_fails_closed",
                NativeApiExposure::new(
                    IpAddr::V4(Ipv4Addr::UNSPECIFIED),
                    false,
                    std::iter::empty(),
                    false,
                    RemoteExposureApproval::LoopbackOnly,
                ) == Err(TrustError::UnsafeApiExposure)
                    && NativeApiExposure::new(
                        IpAddr::V4(Ipv4Addr::UNSPECIFIED),
                        true,
                        ["http://localhost.evil.com".to_owned()],
                        false,
                        RemoteExposureApproval::Approved,
                    ) == Err(TrustError::UnsafeApiExposure)
                    && NativeApiExposure::new(
                        IpAddr::V4(Ipv4Addr::UNSPECIFIED),
                        true,
                        ["https://example.com".to_owned()],
                        false,
                        RemoteExposureApproval::Approved,
                    )
                    .is_ok(),
            ),
            (
                "settings_store_public_verifiers_and_reject_private_or_global_secret_authority",
                settings_content_source.contains("pub public_key_hex: Option<String>")
                    && !settings_content_source.contains("pub private_key")
                    && !settings_content_source.contains("pub signing_key")
                    && !settings_content_source.contains("pub seed: Option")
                    && runtime_settings_source.contains(
                        "plugin security settings must not contain private signing material",
                    )
                    && runtime_settings_source.contains(
                        "legacy provider_secret_ids cannot grant credential authority",
                    ),
            ),
            (
                "secret_values_are_nonserializable_and_redacted",
                format!("{:?}", SecretValue::new(b"do-not-log".to_vec()))
                    == "SecretValue([REDACTED])"
                    && SecretId::new("secret\nidentifier").is_err()
                    && !runtime_trust_source
                        .contains("#[derive(Serialize, Deserialize)]\npub struct SecretValue"),
            ),
        ]);
        assert!(cases.values().all(|passed| *passed), "{cases:#?}");
        write_trust_validation_artifact(&root, &cases)?;
        Ok(())
    }

    #[test]
    fn general_video_codec_abi_is_exact_and_declaration_only()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let verified = review_general_video_codec_declarations(
            GENERAL_VIDEO_CODEC_ABI_MANIFEST,
            &cancellation,
        )?;
        assert_eq!(verified.profile(), GENERAL_VIDEO_CODEC_ABI_PROFILE);
        assert_eq!(verified.target(), "x86_64-unknown-linux-gnu");
        assert_eq!(
            verified.manifest_sha256(),
            GENERAL_VIDEO_CODEC_ABI_MANIFEST_SHA256
        );
        assert_eq!(verified.libraries().len(), 6);
        assert_eq!(
            verified
                .libraries()
                .values()
                .map(|library| library.symbols().len())
                .sum::<usize>(),
            78
        );
        assert_eq!(
            verified
                .libraries()
                .get("avfilter")
                .map(ReviewedGeneralVideoCodecLibraryRequirement::symbol_version_namespace),
            Some("LIBAVFILTER_10")
        );
        let certification_registry = NativeFfiRegistry::default();
        for (identity, declarations) in verified.libraries() {
            assert_eq!(
                certification_registry.authorize(
                    identity,
                    verified.manifest_sha256(),
                    &format!("ffmpeg-7.1:{}", declarations.abi_major()),
                    declarations.symbols(),
                ),
                Err(TrustError::UncertifiedFfi),
                "reviewed declarations unexpectedly certified {identity}"
            );
        }

        let original: serde_json::Value = serde_json::from_slice(GENERAL_VIDEO_CODEC_ABI_MANIFEST)?;
        let mut mutation_paths = vec![
            "/schema_version".to_owned(),
            "/profile".to_owned(),
            "/target".to_owned(),
            "/source/archive_sha256".to_owned(),
            "/source/signature_sha256".to_owned(),
            "/source/signing_key_fingerprint".to_owned(),
            "/source/signature_verified".to_owned(),
            "/compiler_oracle/compiler".to_owned(),
            "/compiler_oracle/compiler_target".to_owned(),
            "/compiler_oracle/oci_index_sha256".to_owned(),
            "/compiler_oracle/oci_linux_amd64_manifest_sha256".to_owned(),
            "/compiler_oracle/command".to_owned(),
            "/compiler_oracle/object_format".to_owned(),
            "/compiler_oracle/object_sha256".to_owned(),
            "/contract/historical_library_count".to_owned(),
            "/contract/historical_symbol_count".to_owned(),
            "/contract/required_library_count".to_owned(),
            "/contract/supplemental_symbol_count".to_owned(),
            "/contract/general_symbol_count".to_owned(),
        ];
        for name in original["claims"]
            .as_object()
            .ok_or("claims must be an object")?
            .keys()
        {
            mutation_paths.push(format!("/claims/{name}"));
        }
        for name in original["source"]["headers"]
            .as_object()
            .ok_or("headers must be an object")?
            .keys()
        {
            mutation_paths.push(format!(
                "/source/headers/{}/sha256",
                name.replace('~', "~0").replace('/', "~1")
            ));
        }
        for name in original["historical_manifest_sha256"]
            .as_object()
            .ok_or("historical manifests must be an object")?
            .keys()
        {
            mutation_paths.push(format!(
                "/historical_manifest_sha256/{}",
                name.replace('~', "~0").replace('/', "~1")
            ));
        }
        for (name, fields) in original["projections"]
            .as_object()
            .ok_or("projections must be an object")?
        {
            for field in fields
                .as_object()
                .ok_or("projection fields must be an object")?
                .keys()
            {
                mutation_paths.push(format!("/projections/{name}/{field}"));
            }
        }
        for name in original["libraries"]
            .as_object()
            .ok_or("libraries must be an object")?
            .keys()
        {
            mutation_paths.push(format!("/libraries/{name}/abi_major"));
            mutation_paths.push(format!("/libraries/{name}/symbol_version_namespace"));
            mutation_paths.push(format!("/libraries/{name}/symbols/0"));
        }

        for path in mutation_paths {
            let mut mutated = original.clone();
            let value = mutated
                .pointer_mut(&path)
                .ok_or_else(|| format!("missing mutation path {path}"))?;
            match value {
                serde_json::Value::Bool(value) => *value = !*value,
                serde_json::Value::Number(_) => *value = serde_json::json!(999_999),
                serde_json::Value::String(value) => value.push_str("-mutated"),
                _ => return Err(format!("unsupported mutation path {path}").into()),
            }
            let bytes = serde_json::to_vec(&mutated)?;
            assert_eq!(
                review_general_video_codec_declarations(&bytes, &cancellation),
                Err(GeneralVideoCodecDeclarationReviewError::ContractMismatch),
                "mutation was admitted: {path}"
            );
        }

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert_eq!(
            review_general_video_codec_declarations(GENERAL_VIDEO_CODEC_ABI_MANIFEST, &cancelled,),
            Err(GeneralVideoCodecDeclarationReviewError::Cancelled)
        );
        Ok(())
    }
}
