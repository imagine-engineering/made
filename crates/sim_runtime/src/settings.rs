use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    net::SocketAddr,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use settings_content::{
    SimPluginSecurityPolicyContent, SimRuntimeProfileContent, SimRuntimeSettingsContent,
};
use sim_types::DeviceKind;
use thiserror::Error;
use uuid::Uuid;

use crate::{
    Capability, CapabilitySet, CredentialScope, CudaPackageVerificationKey,
    DirectMlPackageVerificationKey, GENERAL_VIDEO_CODEC_FIXTURE_PUBLIC_KEY_HEX,
    GENERAL_VIDEO_CODEC_FIXTURE_SIGNER, GeneralVideoCodecPackageVerificationKey,
    MetalPackageVerificationKey, MluPackageVerificationKey, NpuPackageVerificationKey,
    PermissionGrant, PermissionPolicy, PluginTrustPolicy, PluginVerificationKey, ProviderEndpoint,
    ProviderMode, ProviderPolicy, RocmPackageVerificationKey, SecretId, XpuPackageVerificationKey,
};

pub fn native_profile_asset_root(data_directory: &Path, profile_id: Uuid) -> PathBuf {
    // Existing desktop profiles store inputs, outputs and private state here.
    // The CLI must resolve the same files after the first-party feature rename.
    data_directory
        .join("comfy")
        .join("native")
        .join(profile_id.to_string())
}

pub const CURRENT_NATIVE_PROFILE_VERSION: u16 = 1;
pub const DEFAULT_NATIVE_PROFILE_ID: Uuid =
    Uuid::from_u128(0x0000_0000_0000_4000_8000_0000_0000_0001);
pub const DEFAULT_COMPONENT_REGISTRY_GENERATION: u64 = 1;

const MAX_PLUGIN_VERIFICATION_KEYS: usize = 256;
const MAX_PLUGIN_PERMISSION_GRANTS: usize = 4_096;
const MAX_PLUGIN_CAPABILITIES_PER_GRANT: usize = 256;
const MAX_PLUGIN_PROVIDER_ENDPOINTS: usize = 1_024;
const MAX_PLUGIN_CREDENTIAL_SCOPES: usize = 1_024;
const MAX_PLUGIN_SETTING_IDENTIFIER_BYTES: usize = 1_024;
const MAX_PLUGIN_PROVIDER_ENDPOINT_BYTES: usize = 4_096;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPolicy {
    Conservative,
    #[default]
    Balanced,
    Performance,
}

impl MemoryPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Conservative => "conservative",
            Self::Balanced => "balanced",
            Self::Performance => "performance",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginPolicy {
    Disabled,
    #[default]
    ApprovedOnly,
    SignedRegistry,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeApiHostPolicy {
    pub enabled: bool,
    pub bind: String,
    pub allow_remote: bool,
}

impl Default for NativeApiHostPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            bind: "127.0.0.1:8188".into(),
            allow_remote: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeProfile {
    pub id: Uuid,
    pub name: String,
    pub model_roots: Vec<PathBuf>,
    pub device: DeviceKind,
    pub memory_policy: MemoryPolicy,
    pub api_host: NativeApiHostPolicy,
    pub plugin_policy: PluginPolicy,
    pub rocm_package: Option<NativeRocmPackageSettings>,
    pub metal_package: Option<NativeMetalPackageSettings>,
    #[serde(default)]
    pub mlu_package: Option<NativeMluPackageSettings>,
    #[serde(default)]
    pub npu_package: Option<NativeNpuPackageSettings>,
    #[serde(default)]
    pub cuda_package: Option<NativeCudaPackageSettings>,
    #[serde(default)]
    pub xpu_package: Option<NativeXpuPackageSettings>,
    #[serde(default)]
    pub directml_package: Option<NativeDirectMlPackageSettings>,
    #[serde(default)]
    pub general_video_codec_package: Option<NativeGeneralVideoCodecPackageSettings>,
    pub provider_scope: String,
    pub compatibility_version: u16,
    pub unknown_fields: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "NativeGeneralVideoCodecPackageSettingsWire",
    into = "NativeGeneralVideoCodecPackageSettingsWire"
)]
pub struct NativeGeneralVideoCodecPackageSettings {
    package_root: PathBuf,
    verification_key: GeneralVideoCodecPackageVerificationKey,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeGeneralVideoCodecPackageSettingsWire {
    package_root: PathBuf,
    signer: String,
    public_key_hex: String,
}

impl TryFrom<NativeGeneralVideoCodecPackageSettingsWire>
    for NativeGeneralVideoCodecPackageSettings
{
    type Error = String;

    fn try_from(value: NativeGeneralVideoCodecPackageSettingsWire) -> Result<Self, Self::Error> {
        Self::from_public_authority(value.package_root, value.signer, &value.public_key_hex)
    }
}

impl From<NativeGeneralVideoCodecPackageSettings> for NativeGeneralVideoCodecPackageSettingsWire {
    fn from(value: NativeGeneralVideoCodecPackageSettings) -> Self {
        Self {
            package_root: value.package_root,
            signer: value.verification_key.signer().to_owned(),
            public_key_hex: encode_public_key_hex(value.verification_key.public_key_bytes()),
        }
    }
}

impl NativeGeneralVideoCodecPackageSettings {
    pub fn from_public_authority(
        package_root: impl Into<PathBuf>,
        signer: impl Into<String>,
        public_key_hex: &str,
    ) -> Result<Self, String> {
        Self::from_authority(package_root, signer, public_key_hex, false)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn from_fixture_authority(
        package_root: impl Into<PathBuf>,
        signer: impl Into<String>,
        public_key_hex: &str,
    ) -> Result<Self, String> {
        Self::from_authority(package_root, signer, public_key_hex, true)
    }

    fn from_authority(
        package_root: impl Into<PathBuf>,
        signer: impl Into<String>,
        public_key_hex: &str,
        permit_fixture_authority: bool,
    ) -> Result<Self, String> {
        let package_root = package_root.into();
        let package_root_text = package_root
            .to_str()
            .ok_or_else(|| "general video codec package root must be UTF-8".to_owned())?;
        if package_root_text.is_empty()
            || package_root_text != package_root_text.trim()
            || package_root_text.len() > 4_096
            || package_root_text.chars().any(char::is_control)
        {
            return Err("general video codec package root is invalid".to_owned());
        }
        let signer = signer.into();
        if !permit_fixture_authority && signer == GENERAL_VIDEO_CODEC_FIXTURE_SIGNER {
            return Err(
                "the reserved general video codec fixture signer is not production authority"
                    .to_owned(),
            );
        }
        if !permit_fixture_authority && public_key_hex == GENERAL_VIDEO_CODEC_FIXTURE_PUBLIC_KEY_HEX
        {
            return Err(
                "the reserved general video codec fixture key is not production authority"
                    .to_owned(),
            );
        }
        let public_key = decode_public_key_hex(public_key_hex).ok_or_else(|| {
            "general video codec package public verification key must be 32 bytes of lowercase hexadecimal"
                .to_owned()
        })?;
        let verification_key = GeneralVideoCodecPackageVerificationKey::new(signer, public_key)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            package_root,
            verification_key,
        })
    }

    pub fn package_root(&self) -> &std::path::Path {
        &self.package_root
    }

    pub fn verification_key(&self) -> &GeneralVideoCodecPackageVerificationKey {
        &self.verification_key
    }

    pub fn public_key_hex(&self) -> String {
        encode_public_key_hex(self.verification_key.public_key_bytes())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "NativeRocmPackageSettingsWire",
    into = "NativeRocmPackageSettingsWire"
)]
pub struct NativeRocmPackageSettings {
    package_root: PathBuf,
    verification_key: RocmPackageVerificationKey,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeRocmPackageSettingsWire {
    package_root: PathBuf,
    signer: String,
    public_key_hex: String,
}

impl TryFrom<NativeRocmPackageSettingsWire> for NativeRocmPackageSettings {
    type Error = String;

    fn try_from(value: NativeRocmPackageSettingsWire) -> Result<Self, Self::Error> {
        Self::from_public_authority(value.package_root, value.signer, &value.public_key_hex)
    }
}

impl From<NativeRocmPackageSettings> for NativeRocmPackageSettingsWire {
    fn from(value: NativeRocmPackageSettings) -> Self {
        Self {
            package_root: value.package_root,
            signer: value.verification_key.signer().to_owned(),
            public_key_hex: encode_public_key_hex(value.verification_key.public_key_bytes()),
        }
    }
}

impl NativeRocmPackageSettings {
    pub fn from_public_authority(
        package_root: impl Into<PathBuf>,
        signer: impl Into<String>,
        public_key_hex: &str,
    ) -> Result<Self, String> {
        let package_root = package_root.into();
        let package_root_text = package_root
            .to_str()
            .ok_or_else(|| "ROCm package root must be UTF-8".to_owned())?;
        if package_root_text.is_empty()
            || package_root_text != package_root_text.trim()
            || package_root_text.len() > 4_096
            || package_root_text.chars().any(char::is_control)
        {
            return Err("ROCm package root is invalid".to_owned());
        }
        let public_key = decode_public_key_hex(public_key_hex).ok_or_else(|| {
            "ROCm package public verification key must be 32 bytes of lowercase hexadecimal"
                .to_owned()
        })?;
        let verification_key = RocmPackageVerificationKey::new(signer, public_key)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            package_root,
            verification_key,
        })
    }

    pub fn package_root(&self) -> &std::path::Path {
        &self.package_root
    }

    pub fn verification_key(&self) -> &RocmPackageVerificationKey {
        &self.verification_key
    }

    pub fn public_key_hex(&self) -> String {
        encode_public_key_hex(self.verification_key.public_key_bytes())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "NativeMetalPackageSettingsWire",
    into = "NativeMetalPackageSettingsWire"
)]
pub struct NativeMetalPackageSettings {
    package_root: PathBuf,
    verification_key: MetalPackageVerificationKey,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeMetalPackageSettingsWire {
    package_root: PathBuf,
    signer: String,
    public_key_hex: String,
}

impl TryFrom<NativeMetalPackageSettingsWire> for NativeMetalPackageSettings {
    type Error = String;

    fn try_from(value: NativeMetalPackageSettingsWire) -> Result<Self, Self::Error> {
        Self::from_public_authority(value.package_root, value.signer, &value.public_key_hex)
    }
}

impl From<NativeMetalPackageSettings> for NativeMetalPackageSettingsWire {
    fn from(value: NativeMetalPackageSettings) -> Self {
        Self {
            package_root: value.package_root,
            signer: value.verification_key.signer().to_owned(),
            public_key_hex: encode_public_key_hex(value.verification_key.public_key_bytes()),
        }
    }
}

impl NativeMetalPackageSettings {
    pub fn from_public_authority(
        package_root: impl Into<PathBuf>,
        signer: impl Into<String>,
        public_key_hex: &str,
    ) -> Result<Self, String> {
        let package_root = package_root.into();
        let package_root_text = package_root
            .to_str()
            .ok_or_else(|| "Metal package root must be UTF-8".to_owned())?;
        if package_root_text.is_empty()
            || package_root_text != package_root_text.trim()
            || package_root_text.len() > 4_096
            || package_root_text.chars().any(char::is_control)
        {
            return Err("Metal package root is invalid".to_owned());
        }
        let public_key = decode_public_key_hex(public_key_hex).ok_or_else(|| {
            "Metal package public verification key must be 32 bytes of lowercase hexadecimal"
                .to_owned()
        })?;
        let verification_key = MetalPackageVerificationKey::new(signer, public_key)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            package_root,
            verification_key,
        })
    }

    pub fn package_root(&self) -> &std::path::Path {
        &self.package_root
    }

    pub fn verification_key(&self) -> &MetalPackageVerificationKey {
        &self.verification_key
    }

    pub fn public_key_hex(&self) -> String {
        encode_public_key_hex(self.verification_key.public_key_bytes())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "NativeMluPackageSettingsWire",
    into = "NativeMluPackageSettingsWire"
)]
pub struct NativeMluPackageSettings {
    package_root: PathBuf,
    verification_key: MluPackageVerificationKey,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeMluPackageSettingsWire {
    package_root: PathBuf,
    signer: String,
    public_key_hex: String,
}

impl TryFrom<NativeMluPackageSettingsWire> for NativeMluPackageSettings {
    type Error = String;

    fn try_from(value: NativeMluPackageSettingsWire) -> Result<Self, Self::Error> {
        Self::from_public_authority(value.package_root, value.signer, &value.public_key_hex)
    }
}

impl From<NativeMluPackageSettings> for NativeMluPackageSettingsWire {
    fn from(value: NativeMluPackageSettings) -> Self {
        Self {
            package_root: value.package_root,
            signer: value.verification_key.signer().to_owned(),
            public_key_hex: encode_public_key_hex(value.verification_key.public_key_bytes()),
        }
    }
}

impl NativeMluPackageSettings {
    pub fn from_public_authority(
        package_root: impl Into<PathBuf>,
        signer: impl Into<String>,
        public_key_hex: &str,
    ) -> Result<Self, String> {
        let package_root = package_root.into();
        let package_root_text = package_root
            .to_str()
            .ok_or_else(|| "MLU package root must be UTF-8".to_owned())?;
        if package_root_text.is_empty()
            || package_root_text != package_root_text.trim()
            || package_root_text.len() > 4_096
            || package_root_text.chars().any(char::is_control)
        {
            return Err("MLU package root is invalid".to_owned());
        }
        let public_key = decode_public_key_hex(public_key_hex).ok_or_else(|| {
            "MLU package public verification key must be 32 bytes of lowercase hexadecimal"
                .to_owned()
        })?;
        let verification_key = MluPackageVerificationKey::new(signer, public_key)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            package_root,
            verification_key,
        })
    }

    pub fn package_root(&self) -> &std::path::Path {
        &self.package_root
    }

    pub fn verification_key(&self) -> &MluPackageVerificationKey {
        &self.verification_key
    }

    pub fn public_key_hex(&self) -> String {
        encode_public_key_hex(self.verification_key.public_key_bytes())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "NativeNpuPackageSettingsWire",
    into = "NativeNpuPackageSettingsWire"
)]
pub struct NativeNpuPackageSettings {
    package_root: PathBuf,
    verification_key: NpuPackageVerificationKey,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeNpuPackageSettingsWire {
    package_root: PathBuf,
    signer: String,
    public_key_hex: String,
}

impl TryFrom<NativeNpuPackageSettingsWire> for NativeNpuPackageSettings {
    type Error = String;

    fn try_from(value: NativeNpuPackageSettingsWire) -> Result<Self, Self::Error> {
        Self::from_public_authority(value.package_root, value.signer, &value.public_key_hex)
    }
}

impl From<NativeNpuPackageSettings> for NativeNpuPackageSettingsWire {
    fn from(value: NativeNpuPackageSettings) -> Self {
        Self {
            package_root: value.package_root,
            signer: value.verification_key.signer().to_owned(),
            public_key_hex: encode_public_key_hex(value.verification_key.public_key_bytes()),
        }
    }
}

impl NativeNpuPackageSettings {
    pub fn from_public_authority(
        package_root: impl Into<PathBuf>,
        signer: impl Into<String>,
        public_key_hex: &str,
    ) -> Result<Self, String> {
        let package_root = package_root.into();
        let package_root_text = package_root
            .to_str()
            .ok_or_else(|| "NPU package root must be UTF-8".to_owned())?;
        if package_root_text.is_empty()
            || package_root_text != package_root_text.trim()
            || package_root_text.len() > 4_096
            || package_root_text.chars().any(char::is_control)
        {
            return Err("NPU package root is invalid".to_owned());
        }
        let public_key = decode_public_key_hex(public_key_hex).ok_or_else(|| {
            "NPU package public verification key must be 32 bytes of lowercase hexadecimal"
                .to_owned()
        })?;
        let verification_key = NpuPackageVerificationKey::new(signer, public_key)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            package_root,
            verification_key,
        })
    }

    pub fn package_root(&self) -> &std::path::Path {
        &self.package_root
    }

    pub fn verification_key(&self) -> &NpuPackageVerificationKey {
        &self.verification_key
    }

    pub fn public_key_hex(&self) -> String {
        encode_public_key_hex(self.verification_key.public_key_bytes())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "NativeXpuPackageSettingsWire",
    into = "NativeXpuPackageSettingsWire"
)]
pub struct NativeXpuPackageSettings {
    package_root: PathBuf,
    verification_key: XpuPackageVerificationKey,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeXpuPackageSettingsWire {
    package_root: PathBuf,
    signer: String,
    public_key_hex: String,
}

impl TryFrom<NativeXpuPackageSettingsWire> for NativeXpuPackageSettings {
    type Error = String;

    fn try_from(value: NativeXpuPackageSettingsWire) -> Result<Self, Self::Error> {
        Self::from_public_authority(value.package_root, value.signer, &value.public_key_hex)
    }
}

impl From<NativeXpuPackageSettings> for NativeXpuPackageSettingsWire {
    fn from(value: NativeXpuPackageSettings) -> Self {
        Self {
            package_root: value.package_root,
            signer: value.verification_key.signer().to_owned(),
            public_key_hex: encode_public_key_hex(value.verification_key.public_key_bytes()),
        }
    }
}

impl NativeXpuPackageSettings {
    pub fn from_public_authority(
        package_root: impl Into<PathBuf>,
        signer: impl Into<String>,
        public_key_hex: &str,
    ) -> Result<Self, String> {
        let package_root = package_root.into();
        let package_root_text = package_root
            .to_str()
            .ok_or_else(|| "XPU package root must be UTF-8".to_owned())?;
        if package_root_text.is_empty()
            || package_root_text != package_root_text.trim()
            || package_root_text.len() > 4_096
            || package_root_text.chars().any(char::is_control)
        {
            return Err("XPU package root is invalid".to_owned());
        }
        let public_key = decode_public_key_hex(public_key_hex).ok_or_else(|| {
            "XPU package public verification key must be 32 bytes of lowercase hexadecimal"
                .to_owned()
        })?;
        let verification_key = XpuPackageVerificationKey::new(signer, public_key)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            package_root,
            verification_key,
        })
    }

    pub fn package_root(&self) -> &std::path::Path {
        &self.package_root
    }

    pub fn verification_key(&self) -> &XpuPackageVerificationKey {
        &self.verification_key
    }

    pub fn public_key_hex(&self) -> String {
        encode_public_key_hex(self.verification_key.public_key_bytes())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "NativeCudaPackageSettingsWire",
    into = "NativeCudaPackageSettingsWire"
)]
pub struct NativeCudaPackageSettings {
    package_root: PathBuf,
    verification_key: CudaPackageVerificationKey,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeCudaPackageSettingsWire {
    package_root: PathBuf,
    signer: String,
    public_key_hex: String,
}

impl TryFrom<NativeCudaPackageSettingsWire> for NativeCudaPackageSettings {
    type Error = String;

    fn try_from(value: NativeCudaPackageSettingsWire) -> Result<Self, Self::Error> {
        Self::from_public_authority(value.package_root, value.signer, &value.public_key_hex)
    }
}

impl From<NativeCudaPackageSettings> for NativeCudaPackageSettingsWire {
    fn from(value: NativeCudaPackageSettings) -> Self {
        Self {
            package_root: value.package_root,
            signer: value.verification_key.signer().to_owned(),
            public_key_hex: encode_public_key_hex(value.verification_key.public_key_bytes()),
        }
    }
}

impl NativeCudaPackageSettings {
    pub fn from_public_authority(
        package_root: impl Into<PathBuf>,
        signer: impl Into<String>,
        public_key_hex: &str,
    ) -> Result<Self, String> {
        let package_root = package_root.into();
        let package_root_text = package_root
            .to_str()
            .ok_or_else(|| "CUDA package root must be UTF-8".to_owned())?;
        if package_root_text.is_empty()
            || package_root_text != package_root_text.trim()
            || package_root_text.len() > 4_096
            || package_root_text.chars().any(char::is_control)
        {
            return Err("CUDA package root is invalid".to_owned());
        }
        let public_key = decode_public_key_hex(public_key_hex).ok_or_else(|| {
            "CUDA package public verification key must be 32 bytes of lowercase hexadecimal"
                .to_owned()
        })?;
        let verification_key = CudaPackageVerificationKey::new(signer, public_key)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            package_root,
            verification_key,
        })
    }

    pub fn package_root(&self) -> &std::path::Path {
        &self.package_root
    }

    pub fn verification_key(&self) -> &CudaPackageVerificationKey {
        &self.verification_key
    }

    pub fn public_key_hex(&self) -> String {
        encode_public_key_hex(self.verification_key.public_key_bytes())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "NativeDirectMlPackageSettingsWire",
    into = "NativeDirectMlPackageSettingsWire"
)]
pub struct NativeDirectMlPackageSettings {
    package_root: PathBuf,
    verification_key: DirectMlPackageVerificationKey,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeDirectMlPackageSettingsWire {
    package_root: PathBuf,
    signer: String,
    public_key_hex: String,
}

impl TryFrom<NativeDirectMlPackageSettingsWire> for NativeDirectMlPackageSettings {
    type Error = String;

    fn try_from(value: NativeDirectMlPackageSettingsWire) -> Result<Self, Self::Error> {
        Self::from_public_authority(value.package_root, value.signer, &value.public_key_hex)
    }
}

impl From<NativeDirectMlPackageSettings> for NativeDirectMlPackageSettingsWire {
    fn from(value: NativeDirectMlPackageSettings) -> Self {
        Self {
            package_root: value.package_root,
            signer: value.verification_key.signer().to_owned(),
            public_key_hex: encode_public_key_hex(value.verification_key.public_key_bytes()),
        }
    }
}

impl NativeDirectMlPackageSettings {
    pub fn from_public_authority(
        package_root: impl Into<PathBuf>,
        signer: impl Into<String>,
        public_key_hex: &str,
    ) -> Result<Self, String> {
        let package_root = package_root.into();
        let package_root_text = package_root
            .to_str()
            .ok_or_else(|| "DirectML package root must be UTF-8".to_owned())?;
        if package_root_text.is_empty()
            || package_root_text != package_root_text.trim()
            || package_root_text.len() > 4_096
            || package_root_text.chars().any(char::is_control)
        {
            return Err("DirectML package root is invalid".to_owned());
        }
        let public_key = decode_public_key_hex(public_key_hex).ok_or_else(|| {
            "DirectML package public verification key must be 32 bytes of lowercase hexadecimal"
                .to_owned()
        })?;
        let verification_key = DirectMlPackageVerificationKey::new(signer, public_key)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            package_root,
            verification_key,
        })
    }

    pub fn package_root(&self) -> &std::path::Path {
        &self.package_root
    }

    pub fn verification_key(&self) -> &DirectMlPackageVerificationKey {
        &self.verification_key
    }

    pub fn public_key_hex(&self) -> String {
        encode_public_key_hex(self.verification_key.public_key_bytes())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InactiveRuntimeProfile {
    pub content: SimRuntimeProfileContent,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeRuntimeSettings {
    pub active_profile_id: Uuid,
    pub profiles: Vec<NativeRuntimeProfile>,
    pub inactive_profiles: Vec<InactiveRuntimeProfile>,
    pub unknown_fields: BTreeMap<String, serde_json::Value>,
    plugin_security_policies: BTreeMap<Uuid, NativePluginSecurityPolicy>,
}

impl NativeRuntimeSettings {
    pub fn active_profile(&self) -> Option<&NativeRuntimeProfile> {
        self.profiles
            .iter()
            .find(|profile| profile.id == self.active_profile_id)
    }

    pub fn plugin_security_policy(&self, profile_id: Uuid) -> Option<&NativePluginSecurityPolicy> {
        self.plugin_security_policies.get(&profile_id)
    }

    pub fn active_plugin_security_policy(&self) -> Option<&NativePluginSecurityPolicy> {
        self.plugin_security_policy(self.active_profile_id)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativePluginSecurityPolicy {
    enabled: bool,
    plugin_policy: PluginPolicy,
    trust_policy: PluginTrustPolicy,
    permission_policy: PermissionPolicy,
    provider_policy: ProviderPolicy,
    component_registry_generation: u64,
}

impl NativePluginSecurityPolicy {
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn plugin_policy(&self) -> PluginPolicy {
        self.plugin_policy
    }

    pub fn trust_policy(&self) -> &PluginTrustPolicy {
        &self.trust_policy
    }

    pub fn permission_policy(&self) -> &PermissionPolicy {
        &self.permission_policy
    }

    pub fn provider_policy(&self) -> &ProviderPolicy {
        &self.provider_policy
    }

    pub fn component_registry_generation(&self) -> u64 {
        self.component_registry_generation
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RuntimeSettingsError {
    #[error("native runtime profile has no id")]
    MissingId,
    #[error("native runtime profile id cannot be nil")]
    NilId,
    #[error("native runtime profile id is invalid: {0}")]
    InvalidId(String),
    #[error("unsupported native device: {0}")]
    InvalidDevice(String),
    #[error("unsupported native memory policy: {0}")]
    InvalidMemoryPolicy(String),
    #[error("unsupported native plugin policy: {0}")]
    InvalidPluginPolicy(String),
    #[error("unsupported native runtime profile version: {0}")]
    UnsupportedVersion(u16),
    #[error("native runtime provider scope cannot be empty")]
    EmptyProviderScope,
    #[error("API bind must be loopback unless remote access is explicitly allowed")]
    UnsafeApiBind,
    #[error("API bind is invalid: {0}")]
    InvalidApiBind(String),
    #[error("native runtime settings have no active profile")]
    MissingActiveProfile,
    #[error("native runtime active profile id is invalid: {0}")]
    InvalidActiveProfileId(String),
    #[error("native runtime profile id is duplicated: {0}")]
    DuplicateProfileId(Uuid),
    #[error("native plugin security policy is invalid: {0}")]
    InvalidPluginSecurityPolicy(String),
    #[error("native ROCm package security policy is invalid: {0}")]
    InvalidRocmPackageSecurity(String),
    #[error("native Metal package security policy is invalid: {0}")]
    InvalidMetalPackageSecurity(String),
    #[error("native MLU package security policy is invalid: {0}")]
    InvalidMluPackageSecurity(String),
    #[error("native NPU package security policy is invalid: {0}")]
    InvalidNpuPackageSecurity(String),
    #[error("native CUDA package security policy is invalid: {0}")]
    InvalidCudaPackageSecurity(String),
    #[error("native XPU package security policy is invalid: {0}")]
    InvalidXpuPackageSecurity(String),
    #[error("native DirectML package security policy is invalid: {0}")]
    InvalidDirectMlPackageSecurity(String),
    #[error("native general video codec package security policy is invalid: {0}")]
    InvalidGeneralVideoCodecPackageSecurity(String),
    #[error("native runtime active profile is not present: {0}")]
    ActiveProfileNotFound(Uuid),
    #[error("native runtime active profile {profile_id} is inactive: {reason}")]
    ActiveProfileInvalid { profile_id: Uuid, reason: String },
}

impl TryFrom<&SimRuntimeProfileContent> for NativeRuntimeProfile {
    type Error = RuntimeSettingsError;

    fn try_from(value: &SimRuntimeProfileContent) -> Result<Self, Self::Error> {
        let id = value.id.as_deref().ok_or(RuntimeSettingsError::MissingId)?;
        let id = Uuid::parse_str(id).map_err(|_| RuntimeSettingsError::InvalidId(id.into()))?;
        if id.is_nil() {
            return Err(RuntimeSettingsError::NilId);
        }
        let device_name = value.device.as_deref().unwrap_or("cpu");
        let device = match device_name {
            "cpu" => DeviceKind::Cpu,
            "cuda" => DeviceKind::Cuda,
            "rocm" => DeviceKind::Rocm,
            "metal" => DeviceKind::Metal,
            "directml" => DeviceKind::DirectMl,
            "xpu" => DeviceKind::Xpu,
            "npu" => DeviceKind::Npu,
            "mlu" => DeviceKind::Mlu,
            "corex" => DeviceKind::CoreX,
            other => return Err(RuntimeSettingsError::InvalidDevice(other.into())),
        };
        let memory_policy = match value.memory_policy.as_deref().unwrap_or("balanced") {
            "conservative" => MemoryPolicy::Conservative,
            "balanced" => MemoryPolicy::Balanced,
            "performance" => MemoryPolicy::Performance,
            other => return Err(RuntimeSettingsError::InvalidMemoryPolicy(other.into())),
        };
        let plugin_policy = match value.plugin_policy.as_deref().unwrap_or("approved_only") {
            "disabled" => PluginPolicy::Disabled,
            "approved_only" => PluginPolicy::ApprovedOnly,
            "signed_registry" => PluginPolicy::SignedRegistry,
            other => return Err(RuntimeSettingsError::InvalidPluginPolicy(other.into())),
        };
        let bind = value
            .api_bind
            .clone()
            .unwrap_or_else(|| "127.0.0.1:8188".into());
        let socket = bind
            .parse::<SocketAddr>()
            .map_err(|_| RuntimeSettingsError::InvalidApiBind(bind.clone()))?;
        if !socket.ip().is_loopback() {
            return Err(RuntimeSettingsError::UnsafeApiBind);
        }
        let provider_scope = value
            .provider_scope
            .clone()
            .unwrap_or_else(|| "local".into());
        if provider_scope.trim().is_empty() {
            return Err(RuntimeSettingsError::EmptyProviderScope);
        }
        let compatibility_version = value
            .compatibility_version
            .unwrap_or(CURRENT_NATIVE_PROFILE_VERSION);
        if compatibility_version != CURRENT_NATIVE_PROFILE_VERSION {
            return Err(RuntimeSettingsError::UnsupportedVersion(
                compatibility_version,
            ));
        }
        let general_video_codec_configuration_present = value.video_codec_package_root.is_some()
            || value.video_codec_package_signer.is_some()
            || value.video_codec_package_public_key_hex.is_some();
        if contains_general_video_codec_private_signing_material(&value.unknown_fields) {
            return Err(invalid_general_video_codec_package_security(
                "runtime profile settings must not contain private signing material",
            ));
        }
        if contains_private_signing_material(&value.unknown_fields) {
            let message = "runtime profile settings must not contain private signing material";
            if general_video_codec_configuration_present {
                return Err(invalid_general_video_codec_package_security(message));
            }
            if device == DeviceKind::Mlu
                || value.mlu_package_root.is_some()
                || value.mlu_package_signer.is_some()
                || value.mlu_package_public_key_hex.is_some()
            {
                return Err(invalid_mlu_package_security(message));
            }
            if device == DeviceKind::Npu
                || value.npu_package_root.is_some()
                || value.npu_package_signer.is_some()
                || value.npu_package_public_key_hex.is_some()
            {
                return Err(invalid_npu_package_security(message));
            }
            if device == DeviceKind::Xpu
                || value.xpu_package_root.is_some()
                || value.xpu_package_signer.is_some()
                || value.xpu_package_public_key_hex.is_some()
            {
                return Err(invalid_xpu_package_security(message));
            }
            if device == DeviceKind::Cuda
                || value.cuda_package_root.is_some()
                || value.cuda_package_signer.is_some()
                || value.cuda_package_public_key_hex.is_some()
            {
                return Err(invalid_cuda_package_security(message));
            }
            if device == DeviceKind::DirectMl
                || value.directml_package_root.is_some()
                || value.directml_package_signer.is_some()
                || value.directml_package_public_key_hex.is_some()
            {
                return Err(invalid_directml_package_security(message));
            }
            if device == DeviceKind::Metal
                || value.metal_package_root.is_some()
                || value.metal_package_signer.is_some()
                || value.metal_package_public_key_hex.is_some()
            {
                return Err(invalid_metal_package_security(message));
            }
            return Err(invalid_rocm_package_security(message));
        }
        let general_video_codec_package = project_general_video_codec_package_settings(value)?;
        let rocm_package = project_rocm_package_settings(value)?;
        let metal_package = project_metal_package_settings(value)?;
        let mlu_package = project_mlu_package_settings(value)?;
        let npu_package = project_npu_package_settings(value)?;
        let cuda_package = project_cuda_package_settings(value)?;
        let xpu_package = project_xpu_package_settings(value)?;
        let directml_package = project_directml_package_settings(value)?;
        Ok(Self {
            id,
            name: value
                .name
                .clone()
                .unwrap_or_else(|| "Native Runtime".into()),
            model_roots: value
                .model_roots
                .clone()
                .unwrap_or_default()
                .into_iter()
                .map(PathBuf::from)
                .collect(),
            device,
            memory_policy,
            api_host: NativeApiHostPolicy {
                enabled: value.api_host_enabled.unwrap_or(false),
                bind,
                allow_remote: false,
            },
            plugin_policy,
            rocm_package,
            metal_package,
            mlu_package,
            npu_package,
            cuda_package,
            xpu_package,
            directml_package,
            general_video_codec_package,
            provider_scope,
            compatibility_version,
            unknown_fields: value.unknown_fields.clone(),
        })
    }
}

impl NativeRuntimeProfile {
    pub fn disabled_migration_replacement(
        id: Uuid,
        source_name: &str,
    ) -> Result<Self, RuntimeSettingsError> {
        if id.is_nil() {
            return Err(RuntimeSettingsError::NilId);
        }
        let source_name = source_name.trim();
        let source_name = if source_name.is_empty() {
            "Imported legacy runtime"
        } else {
            source_name
        };
        Ok(Self {
            id,
            name: format!("{source_name} (Native)"),
            model_roots: Vec::new(),
            device: DeviceKind::Cpu,
            memory_policy: MemoryPolicy::Balanced,
            api_host: NativeApiHostPolicy::default(),
            plugin_policy: PluginPolicy::Disabled,
            rocm_package: None,
            metal_package: None,
            mlu_package: None,
            npu_package: None,
            cuda_package: None,
            xpu_package: None,
            directml_package: None,
            general_video_codec_package: None,
            provider_scope: "local".into(),
            compatibility_version: CURRENT_NATIVE_PROFILE_VERSION,
            unknown_fields: BTreeMap::new(),
        })
    }

    pub fn project_plugin_security_policy(
        &self,
        content: Option<&SimPluginSecurityPolicyContent>,
    ) -> Result<NativePluginSecurityPolicy, RuntimeSettingsError> {
        let profile_id = self.id.to_string();
        let Some(content) = content else {
            return disabled_native_plugin_security_policy(
                profile_id,
                self.plugin_policy,
                DEFAULT_COMPONENT_REGISTRY_GENERATION,
            );
        };
        let enabled = content.enabled.unwrap_or(false);
        let generation = content
            .component_registry_generation
            .unwrap_or(DEFAULT_COMPONENT_REGISTRY_GENERATION);
        if generation == 0 {
            return Err(invalid_plugin_security(
                "component registry generation must be nonzero",
            ));
        }
        if contains_private_signing_material(&content.unknown_fields)
            || content.verification_keys.as_deref().is_some_and(|keys| {
                keys.iter()
                    .any(|key| contains_private_signing_material(&key.unknown_fields))
            })
            || content.permission_grants.as_deref().is_some_and(|grants| {
                grants
                    .iter()
                    .any(|grant| contains_private_signing_material(&grant.unknown_fields))
            })
            || content
                .provider_endpoints
                .as_deref()
                .is_some_and(|endpoints| {
                    endpoints
                        .iter()
                        .any(|endpoint| contains_private_signing_material(&endpoint.unknown_fields))
                })
            || content.credential_scopes.as_deref().is_some_and(|scopes| {
                scopes
                    .iter()
                    .any(|scope| contains_private_signing_material(&scope.unknown_fields))
            })
        {
            return Err(invalid_plugin_security(
                "plugin security settings must not contain private signing material",
            ));
        }
        if !enabled {
            let has_authority = content
                .verification_keys
                .as_ref()
                .is_some_and(|values| !values.is_empty())
                || content
                    .permission_grants
                    .as_ref()
                    .is_some_and(|values| !values.is_empty())
                || content
                    .provider_endpoints
                    .as_ref()
                    .is_some_and(|values| !values.is_empty())
                || content
                    .credential_scopes
                    .as_ref()
                    .is_some_and(|values| !values.is_empty())
                || content
                    .provider_secret_ids
                    .as_ref()
                    .is_some_and(|values| !values.is_empty())
                || content
                    .provider_mode
                    .as_deref()
                    .is_some_and(|mode| mode != "disabled");
            if has_authority {
                return Err(invalid_plugin_security(
                    "disabled plugin security policy cannot declare authority",
                ));
            }
            return disabled_native_plugin_security_policy(
                profile_id,
                self.plugin_policy,
                generation,
            );
        }
        if self.plugin_policy == PluginPolicy::Disabled {
            return Err(invalid_plugin_security(
                "disabled plugin policy cannot enable plugin authority",
            ));
        }
        let verification_key_content = content.verification_keys.as_deref().unwrap_or_default();
        require_count(
            "verification keys",
            verification_key_content.len(),
            MAX_PLUGIN_VERIFICATION_KEYS,
        )?;
        if verification_key_content.is_empty() {
            return Err(invalid_plugin_security(
                "enabled plugin security policy requires a verification key",
            ));
        }
        let verification_keys = verification_key_content
            .iter()
            .map(|key| {
                let key_id = required_setting_text(
                    key.key_id.as_deref(),
                    "verification key id",
                    MAX_PLUGIN_SETTING_IDENTIFIER_BYTES,
                )?;
                let public_key_hex = key
                    .public_key_hex
                    .as_deref()
                    .ok_or_else(|| invalid_plugin_security("verification key bytes are missing"))?;
                PluginVerificationKey::new(key_id, decode_verification_key_hex(public_key_hex)?)
                    .map_err(invalid_plugin_security)
            })
            .collect::<Result<Vec<_>, RuntimeSettingsError>>()?;
        let trust_policy =
            PluginTrustPolicy::new(verification_keys).map_err(invalid_plugin_security)?;

        let permission_generation =
            crate::PermissionPolicyGeneration::new(generation).map_err(invalid_plugin_security)?;
        let permission_policy = PermissionPolicy::native_runtime_services(profile_id.clone())
            .map_err(invalid_plugin_security)?
            .with_generation(permission_generation);
        let grant_content = content.permission_grants.as_deref().unwrap_or_default();
        require_count(
            "permission grants",
            grant_content.len(),
            MAX_PLUGIN_PERMISSION_GRANTS,
        )?;
        let grants = grant_content
            .iter()
            .map(|grant| {
                let subject_id = required_setting_text(
                    grant.subject_id.as_deref(),
                    "permission subject",
                    MAX_PLUGIN_SETTING_IDENTIFIER_BYTES,
                )?;
                let provenance = grant
                    .provenance
                    .as_deref()
                    .unwrap_or("native-runtime-settings");
                let provenance = required_setting_text(
                    Some(provenance),
                    "permission provenance",
                    MAX_PLUGIN_SETTING_IDENTIFIER_BYTES,
                )?;
                let capability_content = grant.capabilities.as_deref().unwrap_or_default();
                require_count(
                    "capabilities per permission grant",
                    capability_content.len(),
                    MAX_PLUGIN_CAPABILITIES_PER_GRANT,
                )?;
                let mut capabilities = BTreeSet::new();
                for capability in capability_content {
                    let capability = Capability::parse_wire_identifier(capability)
                        .map_err(invalid_plugin_security)?;
                    if !capabilities.insert(capability) {
                        return Err(invalid_plugin_security(
                            "permission grant contains a duplicate capability",
                        ));
                    }
                }
                PermissionGrant::new(
                    profile_id.clone(),
                    subject_id,
                    CapabilitySet::new(capabilities),
                    provenance,
                )
                .map_err(invalid_plugin_security)
            })
            .collect::<Result<Vec<_>, RuntimeSettingsError>>()?;
        let permission_policy = permission_policy
            .with_additional_grants(grants)
            .map_err(invalid_plugin_security)?;

        let provider_mode = match content.provider_mode.as_deref().unwrap_or("disabled") {
            "disabled" => ProviderMode::Disabled,
            "offline" => ProviderMode::Offline,
            "enabled" => ProviderMode::Enabled,
            mode => {
                return Err(invalid_plugin_security(format!(
                    "unsupported provider mode `{mode}`"
                )));
            }
        };
        let endpoint_content = content.provider_endpoints.as_deref().unwrap_or_default();
        require_count(
            "provider endpoints",
            endpoint_content.len(),
            MAX_PLUGIN_PROVIDER_ENDPOINTS,
        )?;
        let mut provider_endpoints = BTreeSet::new();
        for endpoint in endpoint_content {
            let provider = required_setting_text(
                endpoint.provider.as_deref(),
                "provider id",
                MAX_PLUGIN_SETTING_IDENTIFIER_BYTES,
            )?;
            let endpoint = required_setting_text(
                endpoint.endpoint.as_deref(),
                "provider endpoint",
                MAX_PLUGIN_PROVIDER_ENDPOINT_BYTES,
            )?;
            let endpoint =
                ProviderEndpoint::new(provider, endpoint).map_err(invalid_plugin_security)?;
            if !provider_endpoints.insert(endpoint) {
                return Err(invalid_plugin_security(
                    "plugin provider endpoint is duplicated",
                ));
            }
        }
        if content
            .provider_secret_ids
            .as_ref()
            .is_some_and(|values| !values.is_empty())
        {
            return Err(invalid_plugin_security(
                "legacy provider_secret_ids cannot grant credential authority; use credential_scopes",
            ));
        }
        let credential_scope_content = content.credential_scopes.as_deref().unwrap_or_default();
        require_count(
            "provider credential scopes",
            credential_scope_content.len(),
            MAX_PLUGIN_CREDENTIAL_SCOPES,
        )?;
        let mut credential_scopes = BTreeSet::new();
        for scope in credential_scope_content {
            let scope_profile_id = required_setting_text(
                scope.profile_id.as_deref(),
                "credential scope profile",
                MAX_PLUGIN_SETTING_IDENTIFIER_BYTES,
            )?;
            if scope_profile_id != profile_id {
                return Err(invalid_plugin_security(
                    "credential scope belongs to another runtime profile",
                ));
            }
            let subject_id = required_setting_text(
                scope.subject_id.as_deref(),
                "credential scope subject",
                MAX_PLUGIN_SETTING_IDENTIFIER_BYTES,
            )?;
            let provider = required_setting_text(
                scope.provider.as_deref(),
                "credential scope provider",
                MAX_PLUGIN_SETTING_IDENTIFIER_BYTES,
            )?;
            let secret_id = required_setting_text(
                scope.secret_id.as_deref(),
                "credential scope secret",
                MAX_PLUGIN_SETTING_IDENTIFIER_BYTES,
            )?;
            let scope = CredentialScope::new(
                scope_profile_id,
                subject_id,
                provider,
                SecretId::new(secret_id).map_err(invalid_plugin_security)?,
            )
            .map_err(invalid_plugin_security)?;
            if !credential_scopes.insert(scope) {
                return Err(invalid_plugin_security(
                    "plugin provider credential scope is duplicated",
                ));
            }
        }
        let provider_policy = ProviderPolicy::new(
            profile_id,
            provider_mode,
            provider_endpoints,
            credential_scopes,
        )
        .map_err(invalid_plugin_security)?;

        Ok(NativePluginSecurityPolicy {
            enabled,
            plugin_policy: self.plugin_policy,
            trust_policy,
            permission_policy,
            provider_policy,
            component_registry_generation: generation,
        })
    }
}

fn disabled_native_plugin_security_policy(
    profile_id: String,
    plugin_policy: PluginPolicy,
    component_registry_generation: u64,
) -> Result<NativePluginSecurityPolicy, RuntimeSettingsError> {
    Ok(NativePluginSecurityPolicy {
        enabled: false,
        plugin_policy,
        trust_policy: PluginTrustPolicy::new(std::iter::empty())
            .map_err(invalid_plugin_security)?,
        permission_policy: PermissionPolicy::native_runtime_services(profile_id.clone())
            .map_err(invalid_plugin_security)?
            .with_generation(
                crate::PermissionPolicyGeneration::new(component_registry_generation)
                    .map_err(invalid_plugin_security)?,
            ),
        provider_policy: ProviderPolicy::new(
            profile_id,
            ProviderMode::Disabled,
            std::iter::empty(),
            std::iter::empty(),
        )
        .map_err(invalid_plugin_security)?,
        component_registry_generation,
    })
}

fn invalid_plugin_security(error: impl ToString) -> RuntimeSettingsError {
    RuntimeSettingsError::InvalidPluginSecurityPolicy(error.to_string())
}

fn invalid_rocm_package_security(error: impl ToString) -> RuntimeSettingsError {
    RuntimeSettingsError::InvalidRocmPackageSecurity(error.to_string())
}

fn invalid_metal_package_security(error: impl ToString) -> RuntimeSettingsError {
    RuntimeSettingsError::InvalidMetalPackageSecurity(error.to_string())
}

fn invalid_mlu_package_security(error: impl ToString) -> RuntimeSettingsError {
    RuntimeSettingsError::InvalidMluPackageSecurity(error.to_string())
}

fn invalid_npu_package_security(error: impl ToString) -> RuntimeSettingsError {
    RuntimeSettingsError::InvalidNpuPackageSecurity(error.to_string())
}

fn invalid_xpu_package_security(error: impl ToString) -> RuntimeSettingsError {
    RuntimeSettingsError::InvalidXpuPackageSecurity(error.to_string())
}

fn invalid_cuda_package_security(error: impl ToString) -> RuntimeSettingsError {
    RuntimeSettingsError::InvalidCudaPackageSecurity(error.to_string())
}

fn invalid_directml_package_security(error: impl ToString) -> RuntimeSettingsError {
    RuntimeSettingsError::InvalidDirectMlPackageSecurity(error.to_string())
}

fn invalid_general_video_codec_package_security(error: impl ToString) -> RuntimeSettingsError {
    RuntimeSettingsError::InvalidGeneralVideoCodecPackageSecurity(error.to_string())
}

fn project_general_video_codec_package_settings(
    profile: &SimRuntimeProfileContent,
) -> Result<Option<NativeGeneralVideoCodecPackageSettings>, RuntimeSettingsError> {
    let fields = (
        profile.video_codec_package_root.as_deref(),
        profile.video_codec_package_signer.as_deref(),
        profile.video_codec_package_public_key_hex.as_deref(),
    );
    let (Some(package_root), Some(signer), Some(public_key_hex)) = fields else {
        if fields.0.is_none() && fields.1.is_none() && fields.2.is_none() {
            return Ok(None);
        }
        return Err(invalid_general_video_codec_package_security(
            "package root, signer, and public verification key must be configured together",
        ));
    };
    NativeGeneralVideoCodecPackageSettings::from_public_authority(
        package_root,
        signer,
        public_key_hex,
    )
    .map(Some)
    .map_err(invalid_general_video_codec_package_security)
}

fn project_rocm_package_settings(
    profile: &SimRuntimeProfileContent,
) -> Result<Option<NativeRocmPackageSettings>, RuntimeSettingsError> {
    let fields = (
        profile.rocm_package_root.as_deref(),
        profile.rocm_package_signer.as_deref(),
        profile.rocm_package_public_key_hex.as_deref(),
    );
    let (Some(package_root), Some(signer), Some(public_key_hex)) = fields else {
        if fields.0.is_none() && fields.1.is_none() && fields.2.is_none() {
            return Ok(None);
        }
        return Err(invalid_rocm_package_security(
            "package root, signer, and public verification key must be configured together",
        ));
    };
    NativeRocmPackageSettings::from_public_authority(package_root, signer, public_key_hex)
        .map(Some)
        .map_err(invalid_rocm_package_security)
}

fn project_metal_package_settings(
    profile: &SimRuntimeProfileContent,
) -> Result<Option<NativeMetalPackageSettings>, RuntimeSettingsError> {
    let fields = (
        profile.metal_package_root.as_deref(),
        profile.metal_package_signer.as_deref(),
        profile.metal_package_public_key_hex.as_deref(),
    );
    let (Some(package_root), Some(signer), Some(public_key_hex)) = fields else {
        if fields.0.is_none() && fields.1.is_none() && fields.2.is_none() {
            return Ok(None);
        }
        return Err(invalid_metal_package_security(
            "package root, signer, and public verification key must be configured together",
        ));
    };
    NativeMetalPackageSettings::from_public_authority(package_root, signer, public_key_hex)
        .map(Some)
        .map_err(invalid_metal_package_security)
}

fn project_mlu_package_settings(
    profile: &SimRuntimeProfileContent,
) -> Result<Option<NativeMluPackageSettings>, RuntimeSettingsError> {
    let fields = (
        profile.mlu_package_root.as_deref(),
        profile.mlu_package_signer.as_deref(),
        profile.mlu_package_public_key_hex.as_deref(),
    );
    let (Some(package_root), Some(signer), Some(public_key_hex)) = fields else {
        if fields.0.is_none() && fields.1.is_none() && fields.2.is_none() {
            return Ok(None);
        }
        return Err(invalid_mlu_package_security(
            "package root, signer, and public verification key must be configured together",
        ));
    };
    NativeMluPackageSettings::from_public_authority(package_root, signer, public_key_hex)
        .map(Some)
        .map_err(invalid_mlu_package_security)
}

fn project_npu_package_settings(
    profile: &SimRuntimeProfileContent,
) -> Result<Option<NativeNpuPackageSettings>, RuntimeSettingsError> {
    let fields = (
        profile.npu_package_root.as_deref(),
        profile.npu_package_signer.as_deref(),
        profile.npu_package_public_key_hex.as_deref(),
    );
    let (Some(package_root), Some(signer), Some(public_key_hex)) = fields else {
        if fields.0.is_none() && fields.1.is_none() && fields.2.is_none() {
            return Ok(None);
        }
        return Err(invalid_npu_package_security(
            "package root, signer, and public verification key must be configured together",
        ));
    };
    NativeNpuPackageSettings::from_public_authority(package_root, signer, public_key_hex)
        .map(Some)
        .map_err(invalid_npu_package_security)
}

fn project_xpu_package_settings(
    profile: &SimRuntimeProfileContent,
) -> Result<Option<NativeXpuPackageSettings>, RuntimeSettingsError> {
    let fields = (
        profile.xpu_package_root.as_deref(),
        profile.xpu_package_signer.as_deref(),
        profile.xpu_package_public_key_hex.as_deref(),
    );
    let (Some(package_root), Some(signer), Some(public_key_hex)) = fields else {
        if fields.0.is_none() && fields.1.is_none() && fields.2.is_none() {
            return Ok(None);
        }
        return Err(invalid_xpu_package_security(
            "package root, signer, and public verification key must be configured together",
        ));
    };
    NativeXpuPackageSettings::from_public_authority(package_root, signer, public_key_hex)
        .map(Some)
        .map_err(invalid_xpu_package_security)
}

fn project_cuda_package_settings(
    profile: &SimRuntimeProfileContent,
) -> Result<Option<NativeCudaPackageSettings>, RuntimeSettingsError> {
    let fields = (
        profile.cuda_package_root.as_deref(),
        profile.cuda_package_signer.as_deref(),
        profile.cuda_package_public_key_hex.as_deref(),
    );
    let (Some(package_root), Some(signer), Some(public_key_hex)) = fields else {
        if fields.0.is_none() && fields.1.is_none() && fields.2.is_none() {
            return Ok(None);
        }
        return Err(invalid_cuda_package_security(
            "package root, signer, and public verification key must be configured together",
        ));
    };
    NativeCudaPackageSettings::from_public_authority(package_root, signer, public_key_hex)
        .map(Some)
        .map_err(invalid_cuda_package_security)
}

fn project_directml_package_settings(
    profile: &SimRuntimeProfileContent,
) -> Result<Option<NativeDirectMlPackageSettings>, RuntimeSettingsError> {
    let fields = (
        profile.directml_package_root.as_deref(),
        profile.directml_package_signer.as_deref(),
        profile.directml_package_public_key_hex.as_deref(),
    );
    let (Some(package_root), Some(signer), Some(public_key_hex)) = fields else {
        if fields.0.is_none() && fields.1.is_none() && fields.2.is_none() {
            return Ok(None);
        }
        return Err(invalid_directml_package_security(
            "package root, signer, and public verification key must be configured together",
        ));
    };
    NativeDirectMlPackageSettings::from_public_authority(package_root, signer, public_key_hex)
        .map(Some)
        .map_err(invalid_directml_package_security)
}

fn is_private_signing_setting(field: &str) -> bool {
    matches!(
        field.to_ascii_lowercase().as_str(),
        "ed25519_seed"
            | "key_hex"
            | "private_key"
            | "private_key_hex"
            | "private_keys"
            | "seed"
            | "signing_key"
            | "signing_keys"
            | "signing_seed"
    )
}

fn contains_private_signing_material(fields: &BTreeMap<String, serde_json::Value>) -> bool {
    fields.iter().any(|(field, value)| {
        is_private_signing_setting(field)
            || match value {
                serde_json::Value::Array(values) => {
                    values.iter().any(value_contains_private_signing_material)
                }
                serde_json::Value::Object(fields) => fields.iter().any(|(field, value)| {
                    is_private_signing_setting(field)
                        || value_contains_private_signing_material(value)
                }),
                _ => false,
            }
    })
}

fn contains_general_video_codec_private_signing_material(
    fields: &BTreeMap<String, serde_json::Value>,
) -> bool {
    fields.keys().any(|field| {
        let field = field.to_ascii_lowercase();
        (field.contains("video_codec") || field.contains("video-codec"))
            && (field.contains("private")
                || field.contains("signing_key")
                || field.contains("signing-key")
                || field.contains("signing_seed")
                || field.contains("signing-seed")
                || field.ends_with("seed"))
    })
}

fn value_contains_private_signing_material(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Array(values) => {
            values.iter().any(value_contains_private_signing_material)
        }
        serde_json::Value::Object(fields) => fields.iter().any(|(field, value)| {
            is_private_signing_setting(field) || value_contains_private_signing_material(value)
        }),
        _ => false,
    }
}

fn require_count(field: &str, actual: usize, maximum: usize) -> Result<(), RuntimeSettingsError> {
    if actual > maximum {
        Err(invalid_plugin_security(format!(
            "{field} exceeds the {maximum} entry limit"
        )))
    } else {
        Ok(())
    }
}

fn required_setting_text(
    value: Option<&str>,
    field: &str,
    maximum_bytes: usize,
) -> Result<String, RuntimeSettingsError> {
    let value = value.ok_or_else(|| invalid_plugin_security(format!("{field} is missing")))?;
    if value.is_empty()
        || value != value.trim()
        || value.len() > maximum_bytes
        || value.chars().any(char::is_control)
    {
        return Err(invalid_plugin_security(format!("{field} is invalid")));
    }
    Ok(value.to_owned())
}

fn decode_verification_key_hex(value: &str) -> Result<Vec<u8>, RuntimeSettingsError> {
    decode_public_key_hex(value)
        .map(|key| key.to_vec())
        .ok_or_else(|| {
            invalid_plugin_security(
                "verification key must be exactly 32 bytes of lowercase hexadecimal",
            )
        })
}

fn encode_public_key_hex(value: &[u8; 32]) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in value {
        encoded.push(encode_hex_nibble(byte >> 4));
        encoded.push(encode_hex_nibble(byte & 0x0f));
    }
    encoded
}

fn encode_hex_nibble(value: u8) -> char {
    match value {
        0..=9 => char::from(b'0' + value),
        _ => char::from(b'a' + (value - 10)),
    }
}

fn decode_public_key_hex(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 {
        return None;
    }
    let mut decoded = [0_u8; 32];
    for (output, pair) in decoded.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        let high = decode_hex_nibble(*pair.first()?)?;
        let low = decode_hex_nibble(*pair.get(1)?)?;
        *output = (high << 4) | low;
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

pub fn parse_runtime_settings(
    content: &SimRuntimeSettingsContent,
) -> Result<NativeRuntimeSettings, RuntimeSettingsError> {
    let active_profile = content
        .active_profile
        .as_deref()
        .ok_or(RuntimeSettingsError::MissingActiveProfile)?;
    let active_profile_id = Uuid::parse_str(active_profile)
        .map_err(|_| RuntimeSettingsError::InvalidActiveProfileId(active_profile.into()))?;
    if active_profile_id.is_nil() {
        return Err(RuntimeSettingsError::InvalidActiveProfileId(
            active_profile.into(),
        ));
    }

    let mut identifiers = HashSet::new();
    let mut profiles = Vec::new();
    let mut inactive_profiles = Vec::new();
    let mut plugin_security_policies = BTreeMap::new();
    for profile_content in content.profiles.as_deref().unwrap_or_default() {
        let parsed_id = profile_content
            .id
            .as_deref()
            .and_then(|id| Uuid::parse_str(id).ok());
        if let Some(profile_id) = parsed_id
            && !identifiers.insert(profile_id)
        {
            return Err(RuntimeSettingsError::DuplicateProfileId(profile_id));
        }
        match NativeRuntimeProfile::try_from(profile_content) {
            Ok(profile) => match profile
                .project_plugin_security_policy(profile_content.plugin_security.as_ref())
            {
                Ok(policy) => {
                    plugin_security_policies.insert(profile.id, policy);
                    profiles.push(profile);
                }
                Err(error) if parsed_id == Some(active_profile_id) => {
                    return Err(RuntimeSettingsError::ActiveProfileInvalid {
                        profile_id: active_profile_id,
                        reason: error.to_string(),
                    });
                }
                Err(error) => inactive_profiles.push(InactiveRuntimeProfile {
                    content: profile_content.clone(),
                    reason: error.to_string(),
                }),
            },
            Err(error) if parsed_id == Some(active_profile_id) => {
                return Err(RuntimeSettingsError::ActiveProfileInvalid {
                    profile_id: active_profile_id,
                    reason: error.to_string(),
                });
            }
            Err(error) => inactive_profiles.push(InactiveRuntimeProfile {
                content: profile_content.clone(),
                reason: error.to_string(),
            }),
        }
    }
    if !profiles
        .iter()
        .any(|profile| profile.id == active_profile_id)
    {
        return Err(RuntimeSettingsError::ActiveProfileNotFound(
            active_profile_id,
        ));
    }
    Ok(NativeRuntimeSettings {
        active_profile_id,
        profiles,
        inactive_profiles,
        unknown_fields: content.unknown_fields.clone(),
        plugin_security_policies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use settings_content::{
        ParseStatus, RootUserSettings, SettingsContent, SimPluginPermissionGrantContent,
        SimPluginSecurityPolicyContent, SimPluginVerificationKeyContent,
        merge_from::MergeFrom as _,
    };
    use sha2::{Digest as _, Sha256};
    use std::{
        collections::BTreeMap,
        error::Error,
        fs, io,
        path::{Path, PathBuf},
    };

    fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
        Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?
            .to_path_buf())
    }

    fn write_settings_validation_artifact(
        workspace_root: &Path,
        cases: &BTreeMap<&str, bool>,
    ) -> Result<(), Box<dyn Error>> {
        let artifact_directory = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| workspace_root.join("target"))
            .join("sim-parity");
        let artifact_path = artifact_directory.join("val-runtime-settings-001.json");
        match fs::remove_file(&artifact_path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let mut fixture_digests = BTreeMap::new();
        for relative_path in [
            "assets/settings/default.json",
            "assets/settings/default-sim.json",
            "crates/settings_content/src/settings_content.rs",
            "crates/sim_runtime/src/settings.rs",
            "crates/zed/src/zed.rs",
        ] {
            fixture_digests.insert(
                relative_path,
                format!(
                    "{:x}",
                    Sha256::digest(fs::read(workspace_root.join(relative_path))?)
                ),
            );
        }
        fs::create_dir_all(&artifact_directory)?;
        let artifact = json!({
            "validation_id": "VAL-RUNTIME-SETTINGS-001",
            "scope": "native-runtime-profile-settings-foundation",
            "environment": {
                "operating_system": std::env::consts::OS,
                "architecture": std::env::consts::ARCH,
                "development_oracle_executed": false,
            },
            "fixture_digests": fixture_digests,
            "summary": {
                "passed": cases.len(),
                "failed": 0,
                "skipped": 0,
            },
            "cases": cases,
            "skipped": [],
            "validation_closure": {
                "claimed": true,
                "scope": "Task 4 native runtime settings parsing, precedence, inactive compatibility, and fail-closed production mapping",
            },
            "release_closure_required": false,
        });
        let temporary_path = artifact_directory.join("val-runtime-settings-001.json.tmp");
        match fs::remove_file(&temporary_path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        fs::write(&temporary_path, serde_json::to_vec_pretty(&artifact)?)?;
        fs::rename(temporary_path, artifact_path)?;
        Ok(())
    }

    #[test]
    fn remote_api_bind_fails_closed() {
        let profile = SimRuntimeProfileContent {
            id: Some(DEFAULT_NATIVE_PROFILE_ID.to_string()),
            api_bind: Some("0.0.0.0:8188".into()),
            ..Default::default()
        };
        assert_eq!(
            NativeRuntimeProfile::try_from(&profile),
            Err(RuntimeSettingsError::UnsafeApiBind)
        );
    }

    #[test]
    fn native_profile_asset_root_preserves_the_existing_desktop_storage() {
        assert_eq!(
            native_profile_asset_root(Path::new("/profile-data"), DEFAULT_NATIVE_PROFILE_ID),
            Path::new("/profile-data/comfy/native").join(DEFAULT_NATIVE_PROFILE_ID.to_string()),
        );
    }

    #[test]
    fn default_settings_do_not_register_a_native_sim_profile() {
        let defaults = include_str!("../../../assets/settings/default.json");
        let settings = <SettingsContent as RootUserSettings>::parse_json_with_comments(defaults)
            .expect("default settings parse");
        assert!(settings.sim_runtime.is_none());
    }

    #[test]
    fn opt_in_sim_settings_register_a_valid_native_profile() {
        let defaults = include_str!("../../../assets/settings/default-sim.json");
        let settings = <SettingsContent as RootUserSettings>::parse_json_with_comments(defaults)
            .expect("opt-in Sim settings parse");
        let sim = settings.sim_runtime.expect("opt-in native Sim defaults");
        let runtime = parse_runtime_settings(&sim).expect("valid native settings");
        assert_eq!(runtime.active_profile_id, DEFAULT_NATIVE_PROFILE_ID);
        assert_eq!(runtime.profiles.len(), 1);
        let active = runtime.active_profile().expect("active profile");
        assert_eq!(active.device, DeviceKind::Cpu);
        assert!(!active.api_host.enabled);
        let security = runtime
            .active_plugin_security_policy()
            .expect("default plugin security projection");
        assert!(!security.enabled());
        assert_eq!(security.trust_policy(), &PluginTrustPolicy::default());
        assert_eq!(
            security.component_registry_generation(),
            DEFAULT_COMPONENT_REGISTRY_GENERATION
        );
        assert!(
            security
                .permission_policy()
                .authorize("plugin.demo", &CapabilitySet::default())
                .is_err()
        );
        assert!(
            security
                .provider_policy()
                .authorize(
                    &active.id.to_string(),
                    "plugin.demo",
                    "provider.demo",
                    "https://provider.invalid/v1/run",
                    None,
                )
                .is_err()
        );
    }

    #[test]
    fn rocm_package_settings_preserve_only_explicit_public_authority() -> Result<(), Box<dyn Error>>
    {
        let package_root = "/opt/zed/rocm-package-reviewed";
        let content: SimRuntimeSettingsContent = serde_json::from_value(json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "profiles": [{
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "rocm",
                "rocm_package_root": package_root,
                "rocm_package_signer": "rocm.release",
                "rocm_package_public_key_hex": "11".repeat(32),
            }]
        }))?;
        let settings = parse_runtime_settings(&content)?;
        let package = settings
            .active_profile()
            .and_then(|profile| profile.rocm_package.as_ref())
            .ok_or("ROCm package settings were not projected")?;
        assert_eq!(package.package_root(), Path::new(package_root));
        assert_eq!(package.verification_key().signer(), "rocm.release");
        assert_eq!(package.verification_key().public_key_bytes(), &[0x11; 32]);
        let serialized = serde_json::to_value(settings.active_profile().ok_or("active profile")?)?;
        let decoded: NativeRuntimeProfile = serde_json::from_value(serialized)?;
        assert_eq!(decoded, *settings.active_profile().ok_or("active profile")?);

        for invalid_profile in [
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "rocm",
                "rocm_package_root": package_root,
                "rocm_package_signer": "rocm.release",
            }),
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "rocm",
                "rocm_package_root": package_root,
                "rocm_package_signer": "rocm.release",
                "rocm_package_public_key_hex": "AA".repeat(32),
            }),
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "rocm",
                "rocm_package_root": package_root,
                "rocm_package_signer": "rocm.release",
                "rocm_package_public_key_hex": "11".repeat(32),
                "private_key_hex": "22".repeat(32),
            }),
        ] {
            let profile: SimRuntimeProfileContent = serde_json::from_value(invalid_profile)?;
            assert!(matches!(
                NativeRuntimeProfile::try_from(&profile),
                Err(RuntimeSettingsError::InvalidRocmPackageSecurity(_))
            ));
        }
        let plugin_only: SimRuntimeProfileContent = serde_json::from_value(json!({
            "id": DEFAULT_NATIVE_PROFILE_ID,
            "device": "rocm",
            "plugin_policy": "signed_registry",
            "plugin_security": {
                "enabled": true,
                "verification_keys": [{
                    "key_id": "rocm.release",
                    "public_key_hex": "11".repeat(32),
                }],
            },
        }))?;
        assert!(
            NativeRuntimeProfile::try_from(&plugin_only)?
                .rocm_package
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn metal_package_settings_preserve_only_explicit_public_authority() -> Result<(), Box<dyn Error>>
    {
        let package_root = "/opt/zed/metal-package-reviewed";
        let content: SimRuntimeSettingsContent = serde_json::from_value(json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "profiles": [{
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "metal",
                "metal_package_root": package_root,
                "metal_package_signer": "metal.release",
                "metal_package_public_key_hex": "22".repeat(32),
            }]
        }))?;
        let settings = parse_runtime_settings(&content)?;
        let package = settings
            .active_profile()
            .and_then(|profile| profile.metal_package.as_ref())
            .ok_or("Metal package settings were not projected")?;
        assert_eq!(package.package_root(), Path::new(package_root));
        assert_eq!(package.verification_key().signer(), "metal.release");
        assert_eq!(package.verification_key().public_key_bytes(), &[0x22; 32]);
        assert_eq!(package.public_key_hex(), "22".repeat(32));
        let serialized = serde_json::to_value(settings.active_profile().ok_or("active profile")?)?;
        assert!(serialized.get("private_key").is_none());
        assert!(serialized.get("signing_key").is_none());
        let decoded: NativeRuntimeProfile = serde_json::from_value(serialized)?;
        assert_eq!(decoded, *settings.active_profile().ok_or("active profile")?);

        for invalid_profile in [
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "metal",
                "metal_package_root": package_root,
                "metal_package_signer": "metal.release",
            }),
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "metal",
                "metal_package_root": package_root,
                "metal_package_signer": "metal.release",
                "metal_package_public_key_hex": "AA".repeat(32),
            }),
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "metal",
                "metal_package_root": package_root,
                "metal_package_signer": "metal.release",
                "metal_package_public_key_hex": "22".repeat(32),
                "private_key_hex": "33".repeat(32),
            }),
        ] {
            let profile: SimRuntimeProfileContent = serde_json::from_value(invalid_profile)?;
            assert!(matches!(
                NativeRuntimeProfile::try_from(&profile),
                Err(RuntimeSettingsError::InvalidMetalPackageSecurity(_))
            ));
        }

        let plugin_only: SimRuntimeProfileContent = serde_json::from_value(json!({
            "id": DEFAULT_NATIVE_PROFILE_ID,
            "device": "metal",
            "plugin_policy": "signed_registry",
            "plugin_security": {
                "enabled": true,
                "verification_keys": [{
                    "key_id": "metal.release",
                    "public_key_hex": "22".repeat(32),
                }],
            },
        }))?;
        assert!(
            NativeRuntimeProfile::try_from(&plugin_only)?
                .metal_package
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn mlu_package_settings_preserve_only_explicit_public_authority() -> Result<(), Box<dyn Error>>
    {
        let package_root = "/opt/zed/mlu-package-reviewed";
        let content: SimRuntimeSettingsContent = serde_json::from_value(json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "profiles": [{
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "mlu",
                "mlu_package_root": package_root,
                "mlu_package_signer": "mlu.release",
                "mlu_package_public_key_hex": "33".repeat(32),
            }]
        }))?;
        let settings = parse_runtime_settings(&content)?;
        let package = settings
            .active_profile()
            .and_then(|profile| profile.mlu_package.as_ref())
            .ok_or("MLU package settings were not projected")?;
        assert_eq!(package.package_root(), Path::new(package_root));
        assert_eq!(package.verification_key().signer(), "mlu.release");
        assert_eq!(package.verification_key().public_key_bytes(), &[0x33; 32]);
        assert_eq!(package.public_key_hex(), "33".repeat(32));

        for invalid_profile in [
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "mlu",
                "mlu_package_root": package_root,
                "mlu_package_signer": "mlu.release",
            }),
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "mlu",
                "mlu_package_root": package_root,
                "mlu_package_signer": "mlu.release",
                "mlu_package_public_key_hex": "AA".repeat(32),
            }),
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "mlu",
                "mlu_package_root": package_root,
                "mlu_package_signer": "mlu.release",
                "mlu_package_public_key_hex": "33".repeat(32),
                "private_key_hex": "44".repeat(32),
            }),
        ] {
            let profile: SimRuntimeProfileContent = serde_json::from_value(invalid_profile)?;
            assert!(matches!(
                NativeRuntimeProfile::try_from(&profile),
                Err(RuntimeSettingsError::InvalidMluPackageSecurity(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn npu_package_settings_preserve_only_explicit_public_authority() -> Result<(), Box<dyn Error>>
    {
        let package_root = "/opt/zed/npu-package-reviewed";
        let content: SimRuntimeSettingsContent = serde_json::from_value(json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "profiles": [{
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "npu",
                "npu_package_root": package_root,
                "npu_package_signer": "npu.release",
                "npu_package_public_key_hex": "55".repeat(32),
            }]
        }))?;
        let settings = parse_runtime_settings(&content)?;
        let package = settings
            .active_profile()
            .and_then(|profile| profile.npu_package.as_ref())
            .ok_or("NPU package settings were not projected")?;
        assert_eq!(package.package_root(), Path::new(package_root));
        assert_eq!(package.verification_key().signer(), "npu.release");
        assert_eq!(package.verification_key().public_key_bytes(), &[0x55; 32]);
        assert_eq!(package.public_key_hex(), "55".repeat(32));

        for invalid_profile in [
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "npu",
                "npu_package_root": package_root,
                "npu_package_signer": "npu.release",
            }),
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "npu",
                "npu_package_root": package_root,
                "npu_package_signer": "npu.release",
                "npu_package_public_key_hex": "AA".repeat(32),
            }),
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "npu",
                "npu_package_root": package_root,
                "npu_package_signer": "npu.release",
                "npu_package_public_key_hex": "55".repeat(32),
                "private_key_hex": "66".repeat(32),
            }),
        ] {
            let profile: SimRuntimeProfileContent = serde_json::from_value(invalid_profile)?;
            assert!(matches!(
                NativeRuntimeProfile::try_from(&profile),
                Err(RuntimeSettingsError::InvalidNpuPackageSecurity(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn cuda_package_settings_preserve_only_explicit_public_authority() -> Result<(), Box<dyn Error>>
    {
        let package_root = "/opt/zed/cuda-package-reviewed";
        let content: SimRuntimeSettingsContent = serde_json::from_value(json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "profiles": [{
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "cuda",
                "cuda_package_root": package_root,
                "cuda_package_signer": "cuda.release",
                "cuda_package_public_key_hex": "56".repeat(32),
            }]
        }))?;
        let settings = parse_runtime_settings(&content)?;
        let profile = settings.active_profile().ok_or("active profile")?;
        let package = profile
            .cuda_package
            .as_ref()
            .ok_or("CUDA package settings were not projected")?;
        assert_eq!(package.package_root(), Path::new(package_root));
        assert_eq!(package.verification_key().signer(), "cuda.release");
        assert_eq!(package.verification_key().public_key_bytes(), &[0x56; 32]);
        assert_eq!(package.public_key_hex(), "56".repeat(32));
        let serialized = serde_json::to_value(profile)?;
        assert!(serialized.get("private_key").is_none());
        assert!(serialized.get("signing_key").is_none());
        let decoded: NativeRuntimeProfile = serde_json::from_value(serialized)?;
        assert_eq!(decoded, *profile);

        for invalid_profile in [
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "cuda",
                "cuda_package_root": package_root,
                "cuda_package_signer": "cuda.release",
            }),
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "cuda",
                "cuda_package_root": package_root,
                "cuda_package_signer": "cuda.release",
                "cuda_package_public_key_hex": "AA".repeat(32),
            }),
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "cuda",
                "cuda_package_root": package_root,
                "cuda_package_signer": "cuda.release",
                "cuda_package_public_key_hex": "56".repeat(32),
                "private_key_hex": "57".repeat(32),
            }),
        ] {
            let profile: SimRuntimeProfileContent = serde_json::from_value(invalid_profile)?;
            assert!(matches!(
                NativeRuntimeProfile::try_from(&profile),
                Err(RuntimeSettingsError::InvalidCudaPackageSecurity(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn xpu_package_settings_preserve_only_explicit_public_authority() -> Result<(), Box<dyn Error>>
    {
        let package_root = "/opt/zed/xpu-package-reviewed";
        let content: SimRuntimeSettingsContent = serde_json::from_value(json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "profiles": [{
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "xpu",
                "xpu_package_root": package_root,
                "xpu_package_signer": "xpu.release",
                "xpu_package_public_key_hex": "45".repeat(32),
            }]
        }))?;
        let settings = parse_runtime_settings(&content)?;
        let profile = settings.active_profile().ok_or("active profile")?;
        let package = profile
            .xpu_package
            .as_ref()
            .ok_or("XPU package settings were not projected")?;
        assert_eq!(package.package_root(), Path::new(package_root));
        assert_eq!(package.verification_key().signer(), "xpu.release");
        assert_eq!(package.verification_key().public_key_bytes(), &[0x45; 32]);
        assert_eq!(package.public_key_hex(), "45".repeat(32));
        let serialized = serde_json::to_value(profile)?;
        assert!(serialized.get("private_key").is_none());
        assert!(serialized.get("signing_key").is_none());
        let decoded: NativeRuntimeProfile = serde_json::from_value(serialized)?;
        assert_eq!(decoded, *profile);

        for invalid_profile in [
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "xpu",
                "xpu_package_root": package_root,
                "xpu_package_signer": "xpu.release",
            }),
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "xpu",
                "xpu_package_root": package_root,
                "xpu_package_signer": "xpu.release",
                "xpu_package_public_key_hex": "AA".repeat(32),
            }),
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "xpu",
                "xpu_package_root": package_root,
                "xpu_package_signer": "xpu.release",
                "xpu_package_public_key_hex": "45".repeat(32),
                "private_key_hex": "46".repeat(32),
            }),
        ] {
            let profile: SimRuntimeProfileContent = serde_json::from_value(invalid_profile)?;
            assert!(matches!(
                NativeRuntimeProfile::try_from(&profile),
                Err(RuntimeSettingsError::InvalidXpuPackageSecurity(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn directml_package_settings_preserve_only_explicit_public_authority()
    -> Result<(), Box<dyn Error>> {
        let package_root = "/opt/zed/directml-package-reviewed";
        let content: SimRuntimeSettingsContent = serde_json::from_value(json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "profiles": [{
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "directml",
                "directml_package_root": package_root,
                "directml_package_signer": "directml.release",
                "directml_package_public_key_hex": "44".repeat(32),
            }]
        }))?;
        let settings = parse_runtime_settings(&content)?;
        let package = settings
            .active_profile()
            .and_then(|profile| profile.directml_package.as_ref())
            .ok_or("DirectML package settings were not projected")?;
        assert_eq!(package.package_root(), Path::new(package_root));
        assert_eq!(package.verification_key().signer(), "directml.release");
        assert_eq!(package.verification_key().public_key_bytes(), &[0x44; 32]);
        assert_eq!(package.public_key_hex(), "44".repeat(32));

        let serialized = serde_json::to_value(settings.active_profile().ok_or("active profile")?)?;
        assert!(serialized.get("private_key").is_none());
        assert!(serialized.get("signing_key").is_none());
        let decoded: NativeRuntimeProfile = serde_json::from_value(serialized)?;
        assert_eq!(decoded, *settings.active_profile().ok_or("active profile")?);

        for invalid_profile in [
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "directml",
                "directml_package_root": package_root,
                "directml_package_signer": "directml.release",
            }),
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "directml",
                "directml_package_root": package_root,
                "directml_package_signer": "directml.release",
                "directml_package_public_key_hex": "AA".repeat(32),
            }),
            json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "directml",
                "directml_package_root": package_root,
                "directml_package_signer": "directml.release",
                "directml_package_public_key_hex": "44".repeat(32),
                "private_key_hex": "55".repeat(32),
            }),
        ] {
            let profile: SimRuntimeProfileContent = serde_json::from_value(invalid_profile)?;
            assert!(matches!(
                NativeRuntimeProfile::try_from(&profile),
                Err(RuntimeSettingsError::InvalidDirectMlPackageSecurity(_))
            ));
        }

        let plugin_only: SimRuntimeProfileContent = serde_json::from_value(json!({
            "id": DEFAULT_NATIVE_PROFILE_ID,
            "device": "directml",
            "plugin_policy": "signed_registry",
            "plugin_security": {
                "enabled": true,
                "verification_keys": [{
                    "key_id": "directml.release",
                    "public_key_hex": "44".repeat(32),
                }],
            },
        }))?;
        assert!(
            NativeRuntimeProfile::try_from(&plugin_only)?
                .directml_package
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn migration_replacement_is_disabled_and_contains_no_imported_roots() {
        let profile = NativeRuntimeProfile::disabled_migration_replacement(
            Uuid::from_u128(0x2903),
            "Legacy Runtime",
        )
        .expect("valid migration replacement");
        assert_eq!(profile.name, "Legacy Runtime (Native)");
        assert!(profile.model_roots.is_empty());
        assert_eq!(profile.device, DeviceKind::Cpu);
        assert!(!profile.api_host.enabled);
        assert!(!profile.api_host.allow_remote);
        assert_eq!(profile.plugin_policy, PluginPolicy::Disabled);
        assert_eq!(profile.provider_scope, "local");
        assert_eq!(
            NativeRuntimeProfile::disabled_migration_replacement(Uuid::nil(), "Legacy"),
            Err(RuntimeSettingsError::NilId)
        );
    }

    #[test]
    fn configured_plugin_security_projects_canonical_owners() -> Result<(), Box<dyn Error>> {
        let value = json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "profiles": [{
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "plugin_policy": "signed_registry",
                "plugin_security": {
                    "enabled": true,
                    "verification_keys": [{
                        "key_id": "registry.demo",
                        "public_key_hex": "11".repeat(32)
                    }],
                    "permission_grants": [{
                        "subject_id": "plugin.demo",
                        "capabilities": [
                            "asset:read:input",
                            "provider_network:provider.demo|https://provider.invalid/v1/run",
                            "secret:secret.demo"
                        ],
                        "provenance": "settings-test"
                    }],
                    "provider_mode": "enabled",
                    "provider_endpoints": [{
                        "provider": "provider.demo",
                        "endpoint": "https://provider.invalid/v1/run"
                    }],
                    "credential_scopes": [{
                        "profile_id": DEFAULT_NATIVE_PROFILE_ID,
                        "subject_id": "plugin.demo",
                        "provider": "provider.demo",
                        "secret_id": "secret.demo"
                    }],
                    "component_registry_generation": 7
                }
            }]
        });
        let content: SimRuntimeSettingsContent = serde_json::from_value(value)?;
        let settings = parse_runtime_settings(&content)?;
        let security = settings
            .active_plugin_security_policy()
            .ok_or("active plugin security policy is absent")?;
        assert!(security.enabled());
        assert_eq!(security.plugin_policy(), PluginPolicy::SignedRegistry);
        assert_eq!(security.component_registry_generation(), 7);
        assert_eq!(security.permission_policy().generation().get(), 7);
        assert_eq!(
            security.trust_policy(),
            &PluginTrustPolicy::new([PluginVerificationKey::new(
                "registry.demo",
                vec![0x11; 32],
            )?])?
        );
        let granted = CapabilitySet::new([
            Capability::parse_wire_identifier("asset:read:input")?,
            Capability::parse_wire_identifier(
                "provider_network:provider.demo|https://provider.invalid/v1/run",
            )?,
            Capability::parse_wire_identifier("secret:secret.demo")?,
        ]);
        assert!(
            security
                .permission_policy()
                .authorize("plugin.demo", &granted)
                .is_ok()
        );
        let secret_id = SecretId::new("secret.demo")?;
        assert!(
            security
                .provider_policy()
                .authorize(
                    &DEFAULT_NATIVE_PROFILE_ID.to_string(),
                    "plugin.demo",
                    "provider.demo",
                    "https://provider.invalid/v1/run",
                    Some(&secret_id)
                )
                .is_ok()
        );
        assert!(
            security
                .provider_policy()
                .authorize(
                    &DEFAULT_NATIVE_PROFILE_ID.to_string(),
                    "plugin.demo",
                    "provider.demo",
                    "https://provider.invalid/v1/other",
                    Some(&secret_id)
                )
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn plugin_security_rejects_zero_generation_disabled_authority_and_limits() {
        let profile = NativeRuntimeProfile::try_from(&SimRuntimeProfileContent {
            id: Some(DEFAULT_NATIVE_PROFILE_ID.to_string()),
            ..Default::default()
        })
        .expect("test profile");
        let zero_generation = SimPluginSecurityPolicyContent {
            component_registry_generation: Some(0),
            ..Default::default()
        };
        assert!(matches!(
            profile.project_plugin_security_policy(Some(&zero_generation)),
            Err(RuntimeSettingsError::InvalidPluginSecurityPolicy(message))
                if message.contains("nonzero")
        ));

        let disabled_authority = SimPluginSecurityPolicyContent {
            verification_keys: Some(vec![SimPluginVerificationKeyContent {
                key_id: Some("registry.demo".to_owned()),
                public_key_hex: Some("11".repeat(32)),
                ..Default::default()
            }]),
            ..Default::default()
        };
        assert!(matches!(
            profile.project_plugin_security_policy(Some(&disabled_authority)),
            Err(RuntimeSettingsError::InvalidPluginSecurityPolicy(message))
                if message.contains("cannot declare authority")
        ));

        let oversized_grants = SimPluginSecurityPolicyContent {
            enabled: Some(true),
            verification_keys: Some(vec![SimPluginVerificationKeyContent {
                key_id: Some("registry.demo".to_owned()),
                public_key_hex: Some("11".repeat(32)),
                ..Default::default()
            }]),
            permission_grants: Some(
                (0..=MAX_PLUGIN_PERMISSION_GRANTS)
                    .map(|index| SimPluginPermissionGrantContent {
                        subject_id: Some(format!("plugin.{index}")),
                        ..Default::default()
                    })
                    .collect(),
            ),
            ..Default::default()
        };
        assert!(matches!(
            profile.project_plugin_security_policy(Some(&oversized_grants)),
            Err(RuntimeSettingsError::InvalidPluginSecurityPolicy(message))
                if message.contains("permission grants exceeds")
        ));

        let invalid_verification_key = SimPluginSecurityPolicyContent {
            enabled: Some(true),
            verification_keys: Some(vec![SimPluginVerificationKeyContent {
                key_id: Some("registry.demo".to_owned()),
                public_key_hex: Some("11".repeat(33)),
                ..Default::default()
            }]),
            ..Default::default()
        };
        assert!(matches!(
            profile.project_plugin_security_policy(Some(&invalid_verification_key)),
            Err(RuntimeSettingsError::InvalidPluginSecurityPolicy(message))
                if message.contains("exactly 32 bytes")
        ));

        let legacy_shared_secret: SimPluginSecurityPolicyContent = serde_json::from_value(json!({
            "enabled": true,
            "signing_keys": [{
                "key_id": "registry.demo",
                "key_hex": "11".repeat(32)
            }]
        }))
        .expect("legacy security settings remain parseable as retained unknown fields");
        assert!(legacy_shared_secret.verification_keys.is_none());
        assert!(
            legacy_shared_secret
                .unknown_fields
                .contains_key("signing_keys")
        );
        assert!(matches!(
            profile.project_plugin_security_policy(Some(&legacy_shared_secret)),
            Err(RuntimeSettingsError::InvalidPluginSecurityPolicy(message))
                if message.contains("private signing material")
        ));
        let mut disabled_legacy_shared_secret = legacy_shared_secret;
        disabled_legacy_shared_secret.enabled = Some(false);
        assert!(matches!(
            profile.project_plugin_security_policy(Some(&disabled_legacy_shared_secret)),
            Err(RuntimeSettingsError::InvalidPluginSecurityPolicy(message))
                if message.contains("private signing material")
        ));
        let nested_private_key: SimPluginSecurityPolicyContent = serde_json::from_value(json!({
            "enabled": true,
            "verification_keys": [{
                "key_id": "registry.demo",
                "public_key_hex": "11".repeat(32),
                "metadata": {
                    "private_key_hex": "22".repeat(32)
                }
            }]
        }))
        .expect("nested private material remains parseable for explicit rejection");
        assert!(matches!(
            profile.project_plugin_security_policy(Some(&nested_private_key)),
            Err(RuntimeSettingsError::InvalidPluginSecurityPolicy(message))
                if message.contains("private signing material")
        ));

        let legacy_global_secret_scope: SimPluginSecurityPolicyContent =
            serde_json::from_value(json!({
                "enabled": true,
                "verification_keys": [{
                    "key_id": "registry.demo",
                    "public_key_hex": "11".repeat(32)
                }],
                "provider_secret_ids": ["secret.demo"]
            }))
            .expect("legacy provider secret settings remain parseable for explicit rejection");
        assert!(matches!(
            profile.project_plugin_security_policy(Some(&legacy_global_secret_scope)),
            Err(RuntimeSettingsError::InvalidPluginSecurityPolicy(message))
                if message.contains("cannot grant credential authority")
        ));
    }

    #[test]
    fn plugin_security_unknown_fields_round_trip_losslessly() -> Result<(), Box<dyn Error>> {
        let value = json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "profiles": [{
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "plugin_security": {
                    "enabled": false,
                    "component_registry_generation": 9,
                    "future_policy": {"retained": true},
                    "verification_keys": [],
                    "permission_grants": [],
                    "provider_endpoints": [],
                    "provider_secret_ids": []
                }
            }]
        });
        let content: SimRuntimeSettingsContent = serde_json::from_value(value.clone())?;
        let settings = parse_runtime_settings(&content)?;
        assert_eq!(
            settings
                .active_plugin_security_policy()
                .ok_or("active plugin security policy is absent")?
                .component_registry_generation(),
            9
        );
        assert_eq!(serde_json::to_value(content)?, value);
        Ok(())
    }

    #[test]
    fn invalid_plugin_security_obeys_active_and_inactive_profile_rules()
    -> Result<(), Box<dyn Error>> {
        let inactive_id = Uuid::from_u128(2);
        let value = json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "profiles": [
                {"id": DEFAULT_NATIVE_PROFILE_ID},
                {
                    "id": inactive_id,
                    "plugin_security": {"component_registry_generation": 0}
                }
            ]
        });
        let content: SimRuntimeSettingsContent = serde_json::from_value(value)?;
        let settings = parse_runtime_settings(&content)?;
        assert_eq!(settings.profiles.len(), 1);
        assert_eq!(settings.inactive_profiles.len(), 1);
        assert!(settings.inactive_profiles[0].reason.contains("nonzero"));
        assert!(settings.plugin_security_policy(inactive_id).is_none());

        let active_invalid = json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "profiles": [{
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "plugin_security": {"component_registry_generation": 0}
            }]
        });
        let active_invalid: SimRuntimeSettingsContent = serde_json::from_value(active_invalid)?;
        assert!(matches!(
            parse_runtime_settings(&active_invalid),
            Err(RuntimeSettingsError::ActiveProfileInvalid { reason, .. })
                if reason.contains("nonzero")
        ));
        Ok(())
    }

    #[test]
    fn invalid_nonactive_profiles_remain_lossless_and_inactive() {
        let unknown = json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "future_root": {"enabled": true},
            "profiles": [
                {
                    "id": DEFAULT_NATIVE_PROFILE_ID,
                    "device": "cpu"
                },
                {
                    "id": "00000000-0000-4000-8000-000000000002",
                    "memory_policy": "future_policy",
                    "future_profile": [1, 2, 3]
                }
            ]
        });
        let content: SimRuntimeSettingsContent =
            serde_json::from_value(unknown.clone()).expect("parse future settings");
        let runtime = parse_runtime_settings(&content).expect("active profile remains valid");
        assert_eq!(runtime.profiles.len(), 1);
        assert_eq!(runtime.inactive_profiles.len(), 1);
        assert!(
            runtime.inactive_profiles[0]
                .reason
                .contains("future_policy")
        );
        assert_eq!(
            serde_json::to_value(content).expect("serialize future settings"),
            unknown
        );
    }

    #[test]
    fn val_runtime_settings_001() -> Result<(), Box<dyn Error>> {
        let workspace_root = workspace_root()?;
        let mut cases = BTreeMap::new();

        let ordinary_defaults = include_str!("../../../assets/settings/default.json");
        let ordinary_content =
            <SettingsContent as RootUserSettings>::parse_json_with_comments(ordinary_defaults)?;
        cases.insert(
            "ordinary_default_settings_are_sim_free",
            ordinary_content.sim_runtime.is_none(),
        );

        let defaults = include_str!("../../../assets/settings/default-sim.json");
        let default_content =
            <SettingsContent as RootUserSettings>::parse_json_with_comments(defaults)?;
        let default_runtime = parse_runtime_settings(
            default_content
                .sim_runtime
                .as_ref()
                .ok_or("opt-in native Sim defaults are absent")?,
        )?;
        let default_profile = default_runtime
            .active_profile()
            .ok_or("native Sim default profile is absent")?;
        cases.insert(
            "registered_default_profile_identity",
            default_runtime.active_profile_id == DEFAULT_NATIVE_PROFILE_ID
                && default_profile.id == DEFAULT_NATIVE_PROFILE_ID,
        );
        cases.insert(
            "registered_default_policy",
            default_profile.device == DeviceKind::Cpu
                && default_profile.memory_policy == MemoryPolicy::Balanced
                && default_profile.plugin_policy == PluginPolicy::ApprovedOnly
                && default_profile.provider_scope == "local"
                && !default_profile.api_host.enabled
                && default_profile.api_host.bind == "127.0.0.1:8188"
                && default_profile.compatibility_version == CURRENT_NATIVE_PROFILE_VERSION,
        );
        let default_plugin_security = default_runtime
            .active_plugin_security_policy()
            .ok_or("default plugin security projection is absent")?;
        cases.insert(
            "absent_plugin_security_is_explicitly_disabled",
            !default_plugin_security.enabled()
                && default_plugin_security.trust_policy() == &PluginTrustPolicy::default()
                && default_plugin_security.component_registry_generation()
                    == DEFAULT_COMPONENT_REGISTRY_GENERATION
                && default_plugin_security
                    .permission_policy()
                    .authorize("plugin.demo", &CapabilitySet::default())
                    .is_err()
                && default_plugin_security
                    .provider_policy()
                    .authorize(
                        &DEFAULT_NATIVE_PROFILE_ID.to_string(),
                        "plugin.demo",
                        "provider.demo",
                        "https://provider.invalid/v1/run",
                        None,
                    )
                    .is_err(),
        );

        let configured_security_value = json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "profiles": [{
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "plugin_policy": "signed_registry",
                "plugin_security": {
                    "enabled": true,
                    "verification_keys": [{
                        "key_id": "registry.demo",
                        "public_key_hex": "22".repeat(32)
                    }],
                    "permission_grants": [{
                        "subject_id": "plugin.demo",
                        "capabilities": ["asset:read:input"]
                    }],
                    "provider_mode": "enabled",
                    "provider_endpoints": [{
                        "provider": "provider.demo",
                        "endpoint": "https://provider.invalid/v1/run"
                    }],
                    "credential_scopes": [{
                        "profile_id": DEFAULT_NATIVE_PROFILE_ID,
                        "subject_id": "plugin.demo",
                        "provider": "provider.demo",
                        "secret_id": "secret.demo"
                    }],
                    "component_registry_generation": 11,
                    "future_policy": {"retained": true}
                }
            }]
        });
        let configured_security_content: SimRuntimeSettingsContent =
            serde_json::from_value(configured_security_value.clone())?;
        let configured_security_settings = parse_runtime_settings(&configured_security_content)?;
        let configured_plugin_security = configured_security_settings
            .active_plugin_security_policy()
            .ok_or("configured plugin security projection is absent")?;
        cases.insert(
            "plugin_security_projects_canonical_owners",
            configured_plugin_security.enabled()
                && configured_plugin_security.plugin_policy() == PluginPolicy::SignedRegistry
                && configured_plugin_security.component_registry_generation() == 11
                && configured_plugin_security
                    .permission_policy()
                    .authorize(
                        "plugin.demo",
                        &CapabilitySet::new([Capability::parse_wire_identifier(
                            "asset:read:input",
                        )?]),
                    )
                    .is_ok()
                && configured_plugin_security
                    .provider_policy()
                    .authorize(
                        &DEFAULT_NATIVE_PROFILE_ID.to_string(),
                        "plugin.demo",
                        "provider.demo",
                        "https://provider.invalid/v1/run",
                        Some(&SecretId::new("secret.demo")?),
                    )
                    .is_ok(),
        );
        cases.insert(
            "plugin_security_unknown_fields_round_trip_exactly",
            serde_json::to_value(configured_security_content)? == configured_security_value,
        );

        let rocm_package_root = "/opt/zed/rocm-package-reviewed";
        let rocm_authority_value = json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "profiles": [{
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "rocm",
                "rocm_package_root": rocm_package_root,
                "rocm_package_signer": "rocm.release",
                "rocm_package_public_key_hex": "11".repeat(32),
            }]
        });
        let rocm_authority_content: SimRuntimeSettingsContent =
            serde_json::from_value(rocm_authority_value.clone())?;
        let rocm_authority_settings = parse_runtime_settings(&rocm_authority_content)?;
        let rocm_profile = rocm_authority_settings
            .active_profile()
            .ok_or("ROCm authority profile is absent")?;
        let rocm_package = rocm_profile
            .rocm_package
            .as_ref()
            .ok_or("ROCm package authority is absent")?;
        let serialized_rocm_profile = serde_json::to_value(rocm_profile)?;
        let decoded_rocm_profile: NativeRuntimeProfile =
            serde_json::from_value(serialized_rocm_profile.clone())?;
        cases.insert(
            "rocm_package_public_authority_round_trips_losslessly",
            rocm_package.package_root() == Path::new(rocm_package_root)
                && rocm_package.verification_key().signer() == "rocm.release"
                && rocm_package.verification_key().public_key_bytes() == &[0x11; 32]
                && decoded_rocm_profile == *rocm_profile
                && serde_json::to_value(&rocm_authority_content)? == rocm_authority_value,
        );
        let serialized_rocm_profile = serde_json::to_string(&serialized_rocm_profile)?;
        cases.insert(
            "rocm_package_profile_contains_no_private_signing_material",
            !serialized_rocm_profile.contains("private")
                && !serialized_rocm_profile.contains("signing_key")
                && !serialized_rocm_profile.contains("seed"),
        );

        let plugin_only_profile: SimRuntimeProfileContent = serde_json::from_value(json!({
            "id": DEFAULT_NATIVE_PROFILE_ID,
            "device": "rocm",
            "plugin_policy": "signed_registry",
            "plugin_security": {
                "enabled": true,
                "verification_keys": [{
                    "key_id": "rocm.release",
                    "public_key_hex": "11".repeat(32),
                }],
            },
        }))?;
        cases.insert(
            "plugin_verification_key_cannot_authorize_rocm_package",
            NativeRuntimeProfile::try_from(&plugin_only_profile)?
                .rocm_package
                .is_none(),
        );
        let private_key_profile: SimRuntimeProfileContent = serde_json::from_value(json!({
            "id": DEFAULT_NATIVE_PROFILE_ID,
            "device": "rocm",
            "rocm_package_root": rocm_package_root,
            "rocm_package_signer": "rocm.release",
            "rocm_package_public_key_hex": "11".repeat(32),
            "private_key_hex": "22".repeat(32),
        }))?;
        cases.insert(
            "private_rocm_signing_material_is_rejected",
            matches!(
                NativeRuntimeProfile::try_from(&private_key_profile),
                Err(RuntimeSettingsError::InvalidRocmPackageSecurity(_))
            ),
        );

        let directml_package_root = "/opt/zed/directml-package-reviewed";
        let directml_authority_value = json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "profiles": [{
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "directml",
                "directml_package_root": directml_package_root,
                "directml_package_signer": "directml.release",
                "directml_package_public_key_hex": "44".repeat(32),
            }]
        });
        let directml_authority_content: SimRuntimeSettingsContent =
            serde_json::from_value(directml_authority_value.clone())?;
        let directml_authority_settings = parse_runtime_settings(&directml_authority_content)?;
        let directml_profile = directml_authority_settings
            .active_profile()
            .ok_or("DirectML authority profile is absent")?;
        let directml_package = directml_profile
            .directml_package
            .as_ref()
            .ok_or("DirectML package authority is absent")?;
        let serialized_directml_profile = serde_json::to_value(directml_profile)?;
        let decoded_directml_profile: NativeRuntimeProfile =
            serde_json::from_value(serialized_directml_profile.clone())?;
        cases.insert(
            "directml_package_public_authority_round_trips_losslessly",
            directml_package.package_root() == Path::new(directml_package_root)
                && directml_package.verification_key().signer() == "directml.release"
                && directml_package.verification_key().public_key_bytes() == &[0x44; 32]
                && decoded_directml_profile == *directml_profile
                && serde_json::to_value(&directml_authority_content)? == directml_authority_value,
        );
        let serialized_directml_profile = serde_json::to_string(&serialized_directml_profile)?;
        cases.insert(
            "directml_package_profile_contains_no_private_signing_material",
            !serialized_directml_profile.contains("private")
                && !serialized_directml_profile.contains("signing_key")
                && !serialized_directml_profile.contains("seed"),
        );

        let plugin_only_directml_profile: SimRuntimeProfileContent =
            serde_json::from_value(json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "directml",
                "plugin_policy": "signed_registry",
                "plugin_security": {
                    "enabled": true,
                    "verification_keys": [{
                        "key_id": "directml.release",
                        "public_key_hex": "44".repeat(32),
                    }],
                },
            }))?;
        cases.insert(
            "plugin_verification_key_cannot_authorize_directml_package",
            NativeRuntimeProfile::try_from(&plugin_only_directml_profile)?
                .directml_package
                .is_none(),
        );
        let private_directml_key_profile: SimRuntimeProfileContent =
            serde_json::from_value(json!({
                "id": DEFAULT_NATIVE_PROFILE_ID,
                "device": "directml",
                "directml_package_root": directml_package_root,
                "directml_package_signer": "directml.release",
                "directml_package_public_key_hex": "44".repeat(32),
                "private_key_hex": "55".repeat(32),
            }))?;
        cases.insert(
            "private_directml_signing_material_is_rejected",
            matches!(
                NativeRuntimeProfile::try_from(&private_directml_key_profile),
                Err(RuntimeSettingsError::InvalidDirectMlPackageSecurity(_))
            ),
        );

        let override_json = format!(
            r#"{{
                "sim_runtime": {{
                    "active_profile": "{DEFAULT_NATIVE_PROFILE_ID}",
                    "future_root": {{"retained": true}},
                    "profiles": [{{
                        "id": "{DEFAULT_NATIVE_PROFILE_ID}",
                        "name": "Layer override",
                        "memory_policy": "conservative",
                        "future_profile": [1, 2, 3]
                    }}]
                }}
            }}"#
        );
        let override_content =
            <SettingsContent as RootUserSettings>::parse_json_with_comments(&override_json)?;
        let mut merged_content = default_content;
        merged_content.merge_from(&override_content);
        let merged_runtime = parse_runtime_settings(
            merged_content
                .sim_runtime
                .as_ref()
                .ok_or("merged native Sim settings are absent")?,
        )?;
        let merged_profile = merged_runtime
            .active_profile()
            .ok_or("merged active profile is absent")?;
        cases.insert(
            "zed_settings_precedence_is_reused",
            merged_profile.name == "Layer override"
                && merged_profile.memory_policy == MemoryPolicy::Conservative
                && merged_profile.device == DeviceKind::Cpu,
        );
        cases.insert(
            "unknown_merged_values_remain_lossless",
            merged_runtime.unknown_fields.get("future_root") == Some(&json!({"retained": true}))
                && merged_profile.unknown_fields.get("future_profile") == Some(&json!([1, 2, 3])),
        );

        let future_settings_value = json!({
            "active_profile": DEFAULT_NATIVE_PROFILE_ID,
            "future_root": {"enabled": true},
            "profiles": [
                {
                    "id": DEFAULT_NATIVE_PROFILE_ID,
                    "device": "cpu"
                },
                {
                    "id": "00000000-0000-4000-8000-000000000002",
                    "memory_policy": "future_policy",
                    "future_profile": [1, 2, 3]
                }
            ]
        });
        let future_content: SimRuntimeSettingsContent =
            serde_json::from_value(future_settings_value.clone())?;
        let future_settings = parse_runtime_settings(&future_content)?;
        cases.insert(
            "future_nonactive_profile_is_inactive",
            future_settings.profiles.len() == 1
                && future_settings.inactive_profiles.len() == 1
                && future_settings.inactive_profiles[0]
                    .reason
                    .contains("future_policy"),
        );
        cases.insert(
            "future_settings_round_trip_exactly",
            serde_json::to_value(&future_content)? == future_settings_value,
        );

        let wrong_type_json = format!(
            r#"{{
                "sim_runtime": {{
                    "active_profile": "{DEFAULT_NATIVE_PROFILE_ID}",
                    "profiles": [{{
                        "id": "{DEFAULT_NATIVE_PROFILE_ID}",
                        "device": "cpu",
                        "api_host_enabled": "not-a-boolean"
                    }}]
                }}
            }}"#
        );
        let (partially_valid, parse_status) =
            <SettingsContent as RootUserSettings>::parse_json(&wrong_type_json);
        let partially_valid = partially_valid.ok_or("valid settings fields were discarded")?;
        let partially_valid_runtime = parse_runtime_settings(
            partially_valid
                .sim_runtime
                .as_ref()
                .ok_or("partially valid native settings are absent")?,
        )?;
        cases.insert(
            "invalid_field_reports_without_discarding_valid_state",
            matches!(parse_status, ParseStatus::Failed { .. })
                && partially_valid_runtime
                    .active_profile()
                    .is_some_and(|profile| !profile.api_host.enabled),
        );

        let active_invalid = SimRuntimeSettingsContent {
            active_profile: Some(DEFAULT_NATIVE_PROFILE_ID.to_string()),
            profiles: Some(vec![SimRuntimeProfileContent {
                id: Some(DEFAULT_NATIVE_PROFILE_ID.to_string()),
                device: Some("future-device".into()),
                ..Default::default()
            }]),
            ..Default::default()
        };
        cases.insert(
            "invalid_active_profile_fails_closed",
            matches!(
                parse_runtime_settings(&active_invalid),
                Err(RuntimeSettingsError::ActiveProfileInvalid { .. })
            ),
        );

        let duplicate_profile = SimRuntimeProfileContent {
            id: Some(DEFAULT_NATIVE_PROFILE_ID.to_string()),
            device: Some("cpu".into()),
            ..Default::default()
        };
        let duplicate = SimRuntimeSettingsContent {
            active_profile: Some(DEFAULT_NATIVE_PROFILE_ID.to_string()),
            profiles: Some(vec![duplicate_profile.clone(), duplicate_profile]),
            ..Default::default()
        };
        cases.insert(
            "duplicate_profile_identity_is_rejected",
            matches!(
                parse_runtime_settings(&duplicate),
                Err(RuntimeSettingsError::DuplicateProfileId(id)) if id == DEFAULT_NATIVE_PROFILE_ID
            ),
        );

        let unsafe_bind = SimRuntimeProfileContent {
            id: Some(DEFAULT_NATIVE_PROFILE_ID.to_string()),
            api_bind: Some("0.0.0.0:8188".into()),
            ..Default::default()
        };
        cases.insert(
            "unsafe_api_bind_is_rejected",
            matches!(
                NativeRuntimeProfile::try_from(&unsafe_bind),
                Err(RuntimeSettingsError::UnsafeApiBind)
            ),
        );
        let unsupported_version = SimRuntimeProfileContent {
            id: Some(DEFAULT_NATIVE_PROFILE_ID.to_string()),
            compatibility_version: Some(CURRENT_NATIVE_PROFILE_VERSION + 1),
            ..Default::default()
        };
        cases.insert(
            "unsupported_profile_version_is_inactive",
            matches!(
                NativeRuntimeProfile::try_from(&unsupported_version),
                Err(RuntimeSettingsError::UnsupportedVersion(_))
            ),
        );

        if cases.values().any(|passed| !passed) {
            return Err(io::Error::other(format!(
                "VAL-RUNTIME-SETTINGS-001 cases failed: {cases:?}"
            ))
            .into());
        }
        write_settings_validation_artifact(&workspace_root, &cases)
    }

    fn general_video_profile() -> SimRuntimeProfileContent {
        SimRuntimeProfileContent {
            id: Some(Uuid::new_v4().to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn general_video_codec_package_settings_require_exact_public_authority() {
        let absent = general_video_profile();
        assert!(
            NativeRuntimeProfile::try_from(&absent)
                .expect("absent codec authority is valid")
                .general_video_codec_package
                .is_none()
        );

        for mask in 1_u8..7 {
            let mut profile = general_video_profile();
            if mask & 1 != 0 {
                profile.video_codec_package_root = Some("/reviewed/general-video".to_owned());
            }
            if mask & 2 != 0 {
                profile.video_codec_package_signer = Some("codec.release".to_owned());
            }
            if mask & 4 != 0 {
                profile.video_codec_package_public_key_hex = Some("11".repeat(32));
            }
            assert!(matches!(
                NativeRuntimeProfile::try_from(&profile),
                Err(RuntimeSettingsError::InvalidGeneralVideoCodecPackageSecurity(_))
            ));
        }

        let mut profile = general_video_profile();
        profile.video_codec_package_root = Some("/reviewed/general-video".to_owned());
        profile.video_codec_package_signer = Some("codec.release".to_owned());
        profile.video_codec_package_public_key_hex = Some("11".repeat(32));
        let projected = NativeRuntimeProfile::try_from(&profile)
            .expect("complete codec authority is valid")
            .general_video_codec_package
            .expect("codec authority is projected");
        assert_eq!(
            projected.package_root(),
            std::path::Path::new("/reviewed/general-video")
        );
        assert_eq!(projected.verification_key().signer(), "codec.release");
        assert_eq!(projected.public_key_hex(), "11".repeat(32));
    }

    #[test]
    fn production_general_video_codec_settings_reject_fixture_and_private_authority() {
        let mut fixture_signer = general_video_profile();
        fixture_signer.video_codec_package_root = Some("/reviewed/general-video".to_owned());
        fixture_signer.video_codec_package_signer =
            Some(GENERAL_VIDEO_CODEC_FIXTURE_SIGNER.to_owned());
        fixture_signer.video_codec_package_public_key_hex = Some("11".repeat(32));
        assert!(matches!(
            NativeRuntimeProfile::try_from(&fixture_signer),
            Err(RuntimeSettingsError::InvalidGeneralVideoCodecPackageSecurity(_))
        ));

        let mut fixture_key = general_video_profile();
        fixture_key.video_codec_package_root = Some("/reviewed/general-video".to_owned());
        fixture_key.video_codec_package_signer = Some("codec.release".to_owned());
        fixture_key.video_codec_package_public_key_hex =
            Some(GENERAL_VIDEO_CODEC_FIXTURE_PUBLIC_KEY_HEX.to_owned());
        assert!(matches!(
            NativeRuntimeProfile::try_from(&fixture_key),
            Err(RuntimeSettingsError::InvalidGeneralVideoCodecPackageSecurity(_))
        ));

        let mut private = general_video_profile();
        private.unknown_fields.insert(
            "video_codec_package_private_key".to_owned(),
            serde_json::Value::String("not permitted".to_owned()),
        );
        assert!(matches!(
            NativeRuntimeProfile::try_from(&private),
            Err(RuntimeSettingsError::InvalidGeneralVideoCodecPackageSecurity(_))
        ));
    }
}
