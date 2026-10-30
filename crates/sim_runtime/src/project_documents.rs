use crate::assets::{
    AssetAction, AssetError, AssetIdentity, AssetNamespace, AssetService, check_cancelled,
    require_asset_authorization, sha256,
};
use crate::{
    AuthorizedCapabilities, OutputCommitError, OutputCommitReceipt, OutputCommitter,
    OutputOperationState, OutputProposal,
};
use serde::{Deserialize, Serialize};
use sim_tensor::CancellationToken;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;
use uuid::Uuid;

pub const SIM_DOCUMENT_SCHEMA_VERSION: u32 = 1;
pub const MAX_SIM_DOCUMENT_BYTES: usize = 16 * 1024 * 1024;
const MAX_PROJECT_VERSION_CATALOG_BYTES: usize = 16 * 1024 * 1024;
const MAX_PROJECT_VERSIONS: usize = 4096;
const MAX_PROJECT_PNG_BYTES: u64 = 64 * 1024 * 1024;
const VERSION_CATALOG_PATH: &str = ".sim/project-asset-versions.json";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ProjectAssetVersionReference {
    pub asset_id: Uuid,
    pub version_id: Uuid,
    pub content_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimEngineDocument {
    pub document_id: Uuid,
    pub engine: String,
    pub source_revision: String,
    pub serialization_format: String,
    pub payload: String,
    pub payload_sha256: String,
    pub dependencies: Vec<ProjectAssetVersionReference>,
}

impl SimEngineDocument {
    pub fn new(
        document_id: Uuid,
        engine: String,
        source_revision: String,
        serialization_format: String,
        payload: String,
        dependencies: Vec<ProjectAssetVersionReference>,
    ) -> Self {
        let payload_sha256 = sha256(payload.as_bytes());
        Self {
            document_id,
            engine,
            source_revision,
            serialization_format,
            payload,
            payload_sha256,
            dependencies,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimDocument {
    pub format: String,
    pub schema_version: u32,
    pub project_id: Uuid,
    pub documents: Vec<SimEngineDocument>,
    pub build_outputs: Vec<ProjectAssetVersionReference>,
    pub provenance: serde_json::Value,
}

impl SimDocument {
    pub fn validate(&self) -> Result<(), ProjectDocumentError> {
        if self.format != "sim" || self.schema_version != SIM_DOCUMENT_SCHEMA_VERSION {
            return Err(ProjectDocumentError::UnsupportedSchema);
        }
        if self.project_id.is_nil() || self.documents.is_empty() || self.documents.len() > 64 {
            return Err(ProjectDocumentError::InvalidDocument);
        }
        let mut document_ids = BTreeSet::new();
        for document in &self.documents {
            if document.document_id.is_nil()
                || !document_ids.insert(document.document_id)
                || document.engine.trim().is_empty()
                || document.source_revision.trim().is_empty()
                || document.serialization_format.trim().is_empty()
                || sha256(document.payload.as_bytes()) != document.payload_sha256
                || document.dependencies.len() > MAX_PROJECT_VERSIONS
            {
                return Err(ProjectDocumentError::InvalidDocument);
            }
            let mut references = BTreeSet::new();
            for reference in &document.dependencies {
                validate_reference(reference)?;
                if !references.insert(reference) {
                    return Err(ProjectDocumentError::InvalidDocument);
                }
            }
        }
        if self.build_outputs.len() > MAX_PROJECT_VERSIONS {
            return Err(ProjectDocumentError::InvalidDocument);
        }
        let mut outputs = BTreeSet::new();
        for reference in &self.build_outputs {
            validate_reference(reference)?;
            if !outputs.insert(reference) {
                return Err(ProjectDocumentError::InvalidDocument);
            }
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>, ProjectDocumentError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)?;
        if bytes.len() > MAX_SIM_DOCUMENT_BYTES {
            return Err(ProjectDocumentError::TooLarge);
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, ProjectDocumentError> {
        if bytes.len() > MAX_SIM_DOCUMENT_BYTES {
            return Err(ProjectDocumentError::TooLarge);
        }
        let document: Self = serde_json::from_slice(bytes)?;
        document.validate()?;
        Ok(document)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProjectEditableSourceReference {
    pub identity: AssetIdentity,
    pub content_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProjectAssetVersion {
    pub reference: ProjectAssetVersionReference,
    pub original_identity: AssetIdentity,
    pub stored_identity: AssetIdentity,
    pub byte_size: u64,
    pub content_type: String,
    pub dependencies: Vec<ProjectAssetVersionReference>,
    pub editable_source: Option<ProjectEditableSourceReference>,
    pub provenance: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct ResolvedProjectAssetVersion {
    version: ProjectAssetVersion,
    bytes: Vec<u8>,
}

impl ResolvedProjectAssetVersion {
    pub fn version(&self) -> &ProjectAssetVersion {
        &self.version
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Serialize, Deserialize)]
struct ProjectVersionCatalog {
    schema_version: u32,
    profile_id: String,
    versions: Vec<ProjectAssetVersion>,
}

#[derive(Debug, Error)]
pub enum ProjectDocumentError {
    #[error("This .sim container schema is unsupported.")]
    UnsupportedSchema,
    #[error("The editable .sim document is invalid or its payload integrity check failed.")]
    InvalidDocument,
    #[error("The .sim document or project version catalog exceeds its bounded storage limit.")]
    TooLarge,
    #[error("The project asset version is invalid.")]
    InvalidVersion,
    #[error("The immutable project asset version is missing: {0:?}")]
    MissingVersion(ProjectAssetVersionReference),
    #[error("The immutable project asset bytes do not match the saved version.")]
    VersionIntegrity,
    #[error("The PNG publication receipt does not match the project asset version.")]
    InvalidReceipt,
    #[error(transparent)]
    Asset(#[from] AssetError),
    #[error(transparent)]
    Publication(#[from] OutputCommitError),
    #[error("Cannot read or encode the .sim document: {0}")]
    Json(#[from] serde_json::Error),
}

fn validate_reference(
    reference: &ProjectAssetVersionReference,
) -> Result<(), ProjectDocumentError> {
    if reference.asset_id.is_nil()
        || reference.version_id.is_nil()
        || reference.content_sha256.len() != 64
        || !reference
            .content_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ProjectDocumentError::InvalidVersion);
    }
    Ok(())
}

impl AssetService {
    pub fn project_png_versions(
        &self,
        authorization: &AuthorizedCapabilities,
    ) -> Result<Vec<ProjectAssetVersion>, ProjectDocumentError> {
        Ok(self.project_version_catalog(authorization)?.versions)
    }

    fn project_version_catalog(
        &self,
        authorization: &AuthorizedCapabilities,
    ) -> Result<ProjectVersionCatalog, ProjectDocumentError> {
        require_asset_authorization(
            authorization,
            &self.roots().profile_id,
            AssetNamespace::Output,
            AssetAction::Read,
        )?;
        let identity = self
            .roots()
            .identity(AssetNamespace::Output, VERSION_CATALOG_PATH)?;
        let Some(bytes) = self
            .roots()
            .read_private(&identity, MAX_PROJECT_VERSION_CATALOG_BYTES)?
        else {
            return Ok(ProjectVersionCatalog {
                schema_version: 1,
                profile_id: self.roots().profile_id.clone(),
                versions: Vec::new(),
            });
        };
        let catalog: ProjectVersionCatalog = serde_json::from_slice(&bytes)?;
        if catalog.schema_version != 1
            || catalog.profile_id != self.roots().profile_id
            || catalog.versions.len() > MAX_PROJECT_VERSIONS
        {
            return Err(ProjectDocumentError::InvalidVersion);
        }
        let mut identities = BTreeSet::new();
        for version in &catalog.versions {
            validate_reference(&version.reference)?;
            if !identities.insert(version.reference.clone())
                || version.original_identity.profile_id != catalog.profile_id
                || version.stored_identity.profile_id != catalog.profile_id
                || version.stored_identity.namespace != AssetNamespace::Output
                || version.byte_size == 0
                || version.content_type != "image/png"
                || version.dependencies.len() > MAX_PROJECT_VERSIONS
            {
                return Err(ProjectDocumentError::InvalidVersion);
            }
            self.roots().artifact_key(&version.stored_identity)?;
            self.roots().artifact_key(&version.original_identity)?;
            let mut dependencies = BTreeSet::new();
            for reference in &version.dependencies {
                validate_reference(reference)?;
                if !dependencies.insert(reference) {
                    return Err(ProjectDocumentError::InvalidVersion);
                }
            }
            if let Some(source) = &version.editable_source {
                if source.identity.profile_id != catalog.profile_id
                    || source.identity.namespace != AssetNamespace::Output
                {
                    return Err(ProjectDocumentError::InvalidVersion);
                }
                self.roots().artifact_key(&source.identity)?;
                validate_reference(&ProjectAssetVersionReference {
                    asset_id: version.reference.asset_id,
                    version_id: version.reference.version_id,
                    content_sha256: source.content_sha256.clone(),
                })?;
            }
        }
        Ok(catalog)
    }

    pub fn resolve_project_asset_version(
        &self,
        reference: &ProjectAssetVersionReference,
        authorization: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
        maximum_bytes: u64,
    ) -> Result<ResolvedProjectAssetVersion, ProjectDocumentError> {
        validate_reference(reference)?;
        check_cancelled(cancellation)?;
        let catalog = self.project_version_catalog(authorization)?;
        let versions: BTreeMap<_, _> = catalog
            .versions
            .iter()
            .map(|version| (&version.reference, version))
            .collect();
        let mut pending = vec![(reference, 0)];
        let mut visiting = BTreeSet::new();
        let mut verified = BTreeSet::new();
        let mut resolved = None;
        while let Some((current, dependency_index)) = pending.pop() {
            check_cancelled(cancellation)?;
            if dependency_index == 0 && verified.contains(current) {
                continue;
            }
            let version = versions
                .get(current)
                .ok_or_else(|| ProjectDocumentError::MissingVersion(current.clone()))?;
            if dependency_index == 0 {
                if !visiting.insert(current) {
                    return Err(ProjectDocumentError::InvalidVersion);
                }
                let bytes = self.read_verified(
                    &version.stored_identity,
                    authorization,
                    cancellation,
                    if current == reference {
                        maximum_bytes
                    } else {
                        MAX_PROJECT_PNG_BYTES
                    },
                )?;
                if bytes.len() as u64 != version.byte_size
                    || sha256(&bytes) != current.content_sha256
                {
                    return Err(ProjectDocumentError::VersionIntegrity);
                }
                if let Some(source) = &version.editable_source {
                    self.verify_project_editable_source(
                        source,
                        &version.dependencies,
                        authorization,
                        cancellation,
                    )?;
                }
                if current == reference {
                    resolved = Some(ResolvedProjectAssetVersion {
                        version: (*version).clone(),
                        bytes,
                    });
                }
            }
            if let Some(dependency) = version.dependencies.get(dependency_index) {
                pending.push((current, dependency_index + 1));
                pending.push((dependency, 0));
            } else {
                visiting.remove(current);
                verified.insert(current);
            }
        }
        resolved.ok_or_else(|| ProjectDocumentError::MissingVersion(reference.clone()))
    }

    fn verify_project_editable_source(
        &self,
        source: &ProjectEditableSourceReference,
        dependencies: &[ProjectAssetVersionReference],
        authorization: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<(), ProjectDocumentError> {
        let bytes = self.read_verified(
            &source.identity,
            authorization,
            cancellation,
            MAX_SIM_DOCUMENT_BYTES as u64,
        )?;
        if sha256(&bytes) != source.content_sha256 {
            return Err(ProjectDocumentError::VersionIntegrity);
        }
        let document = SimDocument::decode(&bytes)?;
        if document.project_id.to_string() != self.roots().profile_id
            || dependencies.iter().any(|dependency| {
                !document
                    .documents
                    .iter()
                    .any(|engine| engine.dependencies.contains(dependency))
            })
        {
            return Err(ProjectDocumentError::InvalidVersion);
        }
        Ok(())
    }

    pub fn register_project_png_version(
        &mut self,
        receipt: &OutputCommitReceipt,
        original_identity: AssetIdentity,
        asset_id: Uuid,
        dependencies: Vec<ProjectAssetVersionReference>,
        editable_source: Option<ProjectEditableSourceReference>,
        provenance: serde_json::Value,
        reader: &AuthorizedCapabilities,
        publisher: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
    ) -> Result<ProjectAssetVersion, ProjectDocumentError> {
        require_asset_authorization(
            publisher,
            &self.roots().profile_id,
            AssetNamespace::Output,
            AssetAction::Write,
        )?;
        check_cancelled(cancellation)?;
        let operation = receipt.operation();
        if operation.state != OutputOperationState::Committed
            || operation.identity.namespace != AssetNamespace::Output
            || operation.identity.profile_id != self.roots().profile_id
            || original_identity.profile_id != self.roots().profile_id
            || operation
                .identity
                .relative_path
                .extension()
                .and_then(|extension| extension.to_str())
                != Some("png")
        {
            return Err(ProjectDocumentError::InvalidReceipt);
        }
        let reference = ProjectAssetVersionReference {
            asset_id,
            version_id: operation.operation_id,
            content_sha256: operation.sha256.clone(),
        };
        validate_reference(&reference)?;
        let mut catalog = self.project_version_catalog(reader)?;
        if dependencies.len() > MAX_PROJECT_VERSIONS {
            return Err(ProjectDocumentError::TooLarge);
        }
        let mut unique_dependencies = BTreeSet::new();
        for dependency in &dependencies {
            validate_reference(dependency)?;
            if !unique_dependencies.insert(dependency) || *dependency == reference {
                return Err(ProjectDocumentError::InvalidVersion);
            }
            if !catalog
                .versions
                .iter()
                .any(|version| version.reference == *dependency)
            {
                return Err(ProjectDocumentError::MissingVersion(dependency.clone()));
            }
            self.resolve_project_asset_version(dependency, reader, cancellation, 64 * 1024 * 1024)?;
        }
        self.roots().artifact_key(&original_identity)?;
        if let Some(source) = &editable_source {
            if source.identity.profile_id != self.roots().profile_id
                || source.identity.namespace != AssetNamespace::Output
            {
                return Err(ProjectDocumentError::InvalidVersion);
            }
            self.roots().artifact_key(&source.identity)?;
            validate_reference(&ProjectAssetVersionReference {
                asset_id,
                version_id: operation.operation_id,
                content_sha256: source.content_sha256.clone(),
            })?;
            self.verify_project_editable_source(source, &dependencies, reader, cancellation)?;
        }
        if let Some(existing) = catalog
            .versions
            .iter()
            .find(|version| version.reference == reference)
        {
            if existing.original_identity != original_identity
                || existing.dependencies != dependencies
                || existing.editable_source != editable_source
                || existing.provenance != provenance
            {
                return Err(ProjectDocumentError::InvalidVersion);
            }
            self.resolve_project_asset_version(
                &reference,
                reader,
                cancellation,
                operation.byte_size,
            )?;
            return Ok(existing.clone());
        }
        if catalog.versions.len() >= MAX_PROJECT_VERSIONS {
            return Err(ProjectDocumentError::TooLarge);
        }
        let bytes = self.read_verified(
            &operation.identity,
            reader,
            cancellation,
            operation.byte_size,
        )?;
        if bytes.len() as u64 != operation.byte_size || sha256(&bytes) != operation.sha256 {
            return Err(ProjectDocumentError::VersionIntegrity);
        }
        validate_project_png(&bytes)?;
        let version = ProjectAssetVersion {
            reference,
            original_identity,
            stored_identity: operation.identity.clone(),
            byte_size: operation.byte_size,
            content_type: "image/png".into(),
            dependencies,
            editable_source,
            provenance,
        };
        catalog.versions.push(version.clone());
        let bytes = serde_json::to_vec(&catalog)?;
        if bytes.len() > MAX_PROJECT_VERSION_CATALOG_BYTES {
            return Err(ProjectDocumentError::TooLarge);
        }
        check_cancelled(cancellation)?;
        let identity = self
            .roots()
            .identity(AssetNamespace::Output, VERSION_CATALOG_PATH)?;
        self.roots().write_private(&identity, &bytes)?;
        Ok(version)
    }

    pub fn pin_project_png_version(
        &mut self,
        committer: &mut OutputCommitter,
        original_identity: &AssetIdentity,
        expected_sha256: &str,
        input_reader: &AuthorizedCapabilities,
        output_reader: &AuthorizedCapabilities,
        publisher: &AuthorizedCapabilities,
        cancellation: &CancellationToken,
        maximum_bytes: u64,
    ) -> Result<ProjectAssetVersion, ProjectDocumentError> {
        if !matches!(
            original_identity.namespace,
            AssetNamespace::Input | AssetNamespace::Output
        ) {
            return Err(ProjectDocumentError::InvalidVersion);
        }
        let source =
            self.read_verified(original_identity, input_reader, cancellation, maximum_bytes)?;
        if sha256(&source) != expected_sha256 {
            return Err(ProjectDocumentError::VersionIntegrity);
        }
        let catalog = self.project_version_catalog(output_reader)?;
        let previous = catalog.versions.iter().find(|version| {
            version.original_identity == *original_identity
                && version.dependencies.is_empty()
                && version.editable_source.is_none()
        });
        let asset_id = previous.map_or_else(Uuid::new_v4, |version| version.reference.asset_id);
        if let Some(version) = catalog.versions.iter().find(|version| {
            version.reference.asset_id == asset_id
                && version.reference.content_sha256 == expected_sha256
                && version.original_identity == *original_identity
                && version.dependencies.is_empty()
                && version.editable_source.is_none()
        }) {
            self.resolve_project_asset_version(
                &version.reference,
                output_reader,
                cancellation,
                maximum_bytes,
            )?;
            return Ok(version.clone());
        }
        let (width, height) = validate_project_png(&source)?;
        let proposal = OutputProposal::new(Uuid::new_v4(), AssetNamespace::Output, format!(".sim/inputs/{asset_id}/{expected_sha256}"), "png", 0, width, height, source)?
            .with_projection_metadata(serde_json::to_vec(&serde_json::json!({"schema":"sim.project-input-version.v1", "asset_id":asset_id, "content_sha256":expected_sha256, "original_identity":original_identity}))?)?;
        let receipts = committer.commit_proposal_batch_and_register(
            &[proposal],
            chrono::Local::now().fixed_offset(),
            self,
            publisher,
            cancellation,
        )?;
        let receipt = receipts
            .first()
            .ok_or(ProjectDocumentError::InvalidReceipt)?;
        self.register_project_png_version(
            receipt,
            original_identity.clone(),
            asset_id,
            Vec::new(),
            None,
            serde_json::json!({"kind":"retained-input"}),
            output_reader,
            publisher,
            cancellation,
        )
    }
}

fn validate_project_png(bytes: &[u8]) -> Result<(u32, u32), ProjectDocumentError> {
    let decoded = sim_media::png::decode_png(
        bytes,
        sim_media::png::PngLimits {
            max_input_bytes: 64 * 1024 * 1024,
            max_dimension: 32768,
            max_pixels: 16 * 1024 * 1024,
            max_decoder_allocation_bytes: 128 * 1024 * 1024,
        },
    )
    .map_err(|_| ProjectDocumentError::InvalidVersion)?;
    Ok((decoded.width, decoded.height))
}

impl AssetService {
    pub(crate) fn project_asset_is_immutable(
        &self,
        identity: &AssetIdentity,
    ) -> Result<bool, AssetError> {
        project_asset_is_immutable(self.roots(), identity)
    }
}

pub(crate) fn project_asset_is_immutable(
    roots: &crate::assets::AssetRoots,
    identity: &AssetIdentity,
) -> Result<bool, AssetError> {
    if identity.namespace != AssetNamespace::Output {
        return Ok(false);
    }
    if identity.relative_path.starts_with(".sim") {
        return Ok(true);
    }
    let catalog_identity = roots.identity(AssetNamespace::Output, VERSION_CATALOG_PATH)?;
    let Some(bytes) = roots.read_private(&catalog_identity, MAX_PROJECT_VERSION_CATALOG_BYTES)?
    else {
        return Ok(false);
    };
    let catalog: ProjectVersionCatalog =
        serde_json::from_slice(&bytes).map_err(|_| AssetError::ReservedAsset(identity.clone()))?;
    if catalog.schema_version != 1
        || catalog.profile_id != roots.profile_id
        || catalog.versions.len() > MAX_PROJECT_VERSIONS
    {
        return Err(AssetError::ReservedAsset(identity.clone()));
    }
    Ok(catalog.versions.iter().any(|version| {
        version.stored_identity == *identity
            || version
                .editable_source
                .as_ref()
                .is_some_and(|source| source.identity == *identity)
    }))
}

#[cfg(test)]
#[path = "project_documents_tests.rs"]
mod tests;
