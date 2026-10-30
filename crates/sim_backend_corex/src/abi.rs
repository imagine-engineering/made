use serde::Deserialize;
use thiserror::Error;

pub const ABI_MANIFEST_JSON: &str = include_str!("../abi/symbols-v1.json");
pub const ABI_FLOOR: &str = "CoreX-IXRT-0.8-ABI-profile";
pub const UNSAFE_OWNER: &str = "sim_backend_corex::loader";
pub const CERTIFICATE_OWNER: &str = "sim_runtime::NativeFfiRegistry";

const TARGETS: [&str; 1] = ["x86_64-unknown-linux-gnu"];
const DISCOVERY_ORDER: [&str; 3] = ["SIM_COREX_ROOT", "IXRT_HOME", "signed_package_roots"];
const REQUIRED_LIBRARIES: [(&str, &str); 2] = [("ixblas", "libixblas.so"), ("ixrt", "libixrt.so")];
const OFFICIAL_REPOSITORY: &str = "https://github.com/Deep-Spark/iluvatar-corex-ixrt";
const OFFICIAL_COMMIT: &str = "0528f3ae5da5dd2255f21966b82bedcb2de65582";
const OFFICIAL_TREE: &str = "3b503f464cfbbe4adab6f54ec6bb46558bc4b386";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ReviewState {
    BlockedMissingVendorHeaders,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AbiManifest {
    pub schema_version: u32,
    pub backend: String,
    pub abi_floor: String,
    pub review_state: ReviewState,
    pub targets: Vec<String>,
    pub discovery_order: Vec<String>,
    pub libraries: Vec<LibraryContract>,
    pub layouts: Vec<LayoutContract>,
    pub evidence: EvidenceContract,
    pub missing_evidence: Vec<MissingEvidenceContract>,
    pub unsafe_owner: String,
    pub certificate_owner: String,
    pub package: PackageContract,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LibraryContract {
    pub id: String,
    pub filename: String,
    pub symbols: Vec<SymbolContract>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SymbolContract {
    pub name: String,
    pub signature: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LayoutContract {
    pub name: String,
    pub size: usize,
    pub align: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceContract {
    pub official_repository: String,
    pub pinned_tag: String,
    pub pinned_commit: String,
    pub pinned_tree: String,
    pub observed_build_file: String,
    pub observed_include_placeholder: String,
    pub observation: String,
    pub proves_ixrt_0_8_abi: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MissingEvidenceContract {
    pub id: String,
    pub required_for: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PackageContract {
    pub redistributes_vendor_runtime: bool,
    pub license_approval_required_for_redistribution: bool,
    pub signature_algorithm: String,
    pub signature_domain: String,
    pub final_application_signing_required: bool,
    pub runtime_loading_enabled: bool,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum AbiManifestError {
    #[error("CoreX ABI manifest is not strict JSON: {0}")]
    Json(String),
    #[error("CoreX ABI manifest violates the reviewed provenance ceiling: {0}")]
    Contract(String),
}

impl AbiManifest {
    pub fn embedded() -> Result<Self, AbiManifestError> {
        Self::parse(ABI_MANIFEST_JSON)
    }

    pub fn parse(json: &str) -> Result<Self, AbiManifestError> {
        let manifest: Self = serde_json::from_str(json)
            .map_err(|error| AbiManifestError::Json(error.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn symbol_count(&self) -> usize {
        self.libraries
            .iter()
            .map(|library| library.symbols.len())
            .sum()
    }

    pub fn validate(&self) -> Result<(), AbiManifestError> {
        if self.schema_version != 1
            || self.backend != "corex"
            || self.abi_floor != ABI_FLOOR
            || self.review_state != ReviewState::BlockedMissingVendorHeaders
            || self.unsafe_owner != UNSAFE_OWNER
            || self.certificate_owner != CERTIFICATE_OWNER
        {
            return Err(contract("identity, floor, review state, or owner differs"));
        }
        require_exact(&self.targets, &TARGETS, "targets")?;
        require_exact(&self.discovery_order, &DISCOVERY_ORDER, "discovery order")?;

        if self.libraries.len() != REQUIRED_LIBRARIES.len()
            || self
                .libraries
                .iter()
                .zip(REQUIRED_LIBRARIES)
                .any(|(actual, expected)| {
                    actual.id != expected.0
                        || actual.filename != expected.1
                        || !actual.symbols.is_empty()
                })
        {
            return Err(contract(
                "libraries must name ixblas and ixrt but contain no unreviewed symbols",
            ));
        }
        if !self.layouts.is_empty() {
            return Err(contract("unreviewed IXRT/IXBLAS layouts must remain empty"));
        }
        if self.evidence.official_repository != OFFICIAL_REPOSITORY
            || self.evidence.pinned_tag != "v4.3.0"
            || self.evidence.pinned_commit != OFFICIAL_COMMIT
            || self.evidence.pinned_tree != OFFICIAL_TREE
            || self.evidence.observed_build_file != "cmake/FindIxrt.cmake"
            || self.evidence.observed_include_placeholder != "ixrt/include"
            || self.evidence.proves_ixrt_0_8_abi
        {
            return Err(contract("pinned primary-source evidence differs"));
        }
        let missing_ids = self
            .missing_evidence
            .iter()
            .map(|missing| missing.id.as_str())
            .collect::<Vec<_>>();
        if missing_ids
            != [
                "ixblas-symbol-signatures",
                "ixrt-0.8-symbol-signatures",
                "ixrt-0.8-type-layouts",
                "normalized-header-digests",
            ]
        {
            return Err(contract("missing-evidence ledger differs"));
        }
        if self
            .missing_evidence
            .iter()
            .any(|missing| missing.required_for.trim().is_empty())
        {
            return Err(contract("missing-evidence reason is empty"));
        }
        if self.package.redistributes_vendor_runtime
            || !self.package.license_approval_required_for_redistribution
            || self.package.signature_algorithm != "ed25519"
            || self.package.signature_domain != "zed-sim-corex-package-v1"
            || !self.package.final_application_signing_required
            || self.package.runtime_loading_enabled
        {
            return Err(contract("package policy differs"));
        }
        Ok(())
    }
}

fn require_exact(
    actual: &[String],
    expected: &[&str],
    field: &'static str,
) -> Result<(), AbiManifestError> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .map(String::as_str)
            .zip(expected.iter().copied())
            .any(|(actual, expected)| actual != expected)
    {
        return Err(contract(format!("{field} differ")));
    }
    Ok(())
}

fn contract(reason: impl Into<String>) -> AbiManifestError {
    AbiManifestError::Contract(reason.into())
}
