use anyhow::anyhow;
use sim_media::{PngLimits, decode_png};
use sim_runtime::{
    AssetError, AssetIdentity, AssetNamespace, AuthorizedCapabilities, ExecutionFailure,
    ExecutionFailureOrigin, SharedAssetService, UploadRequest, authorize_native_image_importer,
    authorize_native_image_output_reader,
};
use sim_types::CancellationToken;
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};
use uuid::Uuid;

const MAXIMUM_IMAGE_INPUT_BYTES: u64 = 16 * 1024 * 1024;
const MAXIMUM_ACTIVE_IMPORTS: usize = 16;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeImageImportBinding {
    pub(crate) profile_id: String,
    pub(crate) generation: Uuid,
}

impl NativeImageImportBinding {
    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn generation(&self) -> Uuid {
        self.generation
    }
}

pub struct PreparedNativeImageImport {
    filename: String,
    bytes: Vec<u8>,
}

impl PreparedNativeImageImport {
    pub fn new(filename: String, bytes: Vec<u8>) -> Self {
        Self { filename, bytes }
    }
}

pub struct ImportedNativeImage {
    pub(crate) identity: AssetIdentity,
    pub(crate) source_name: String,
}

impl ImportedNativeImage {
    pub fn identity(&self) -> &AssetIdentity {
        &self.identity
    }

    pub fn source_name(&self) -> &str {
        &self.source_name
    }
}

pub trait NativeImageImporter: Send + Sync {
    fn binding(&self) -> NativeImageImportBinding;
    fn maximum_input_bytes(&self) -> u64;
    fn import(
        &self,
        prepared: PreparedNativeImageImport,
        cancellation: &CancellationToken,
    ) -> Result<ImportedNativeImage, ExecutionFailure>;
}

pub trait NativeImageOutputReader: Send + Sync {
    fn binding(&self) -> NativeImageImportBinding;
    fn maximum_input_bytes(&self) -> u64;
    fn read(
        &self,
        reference: &str,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u8>, ExecutionFailure>;
}

pub(crate) struct NativeImageImportService {
    binding: NativeImageImportBinding,
    assets: SharedAssetService,
    authorization: AuthorizedCapabilities,
    output_authorization: AuthorizedCapabilities,
    lifetime: CancellationToken,
    active: std::sync::Mutex<BTreeMap<Uuid, CancellationToken>>,
}

impl NativeImageImportService {
    pub(crate) fn new(assets: SharedAssetService) -> anyhow::Result<Arc<Self>> {
        let profile_id = assets
            .lock()
            .map_err(|error| anyhow!("native image import assets are unavailable: {error}"))?
            .roots()
            .profile_id
            .clone();
        let authorization = authorize_native_image_importer(profile_id.clone())?;
        let output_authorization = authorize_native_image_output_reader(profile_id.clone())?;
        Ok(Arc::new(Self {
            binding: NativeImageImportBinding {
                profile_id,
                generation: Uuid::new_v4(),
            },
            assets,
            authorization,
            output_authorization,
            lifetime: CancellationToken::default(),
            active: std::sync::Mutex::new(BTreeMap::new()),
        }))
    }

    pub(crate) fn revoke(&self) {
        let mut active = match self.active.lock() {
            Ok(active) => active,
            Err(error) => {
                log::error!("native image import cancellation state is poisoned: {error}");
                error.into_inner()
            }
        };
        self.lifetime.cancel();
        for cancellation in active.values() {
            cancellation.cancel();
        }
        active.clear();
    }

    fn prepare_import(
        &self,
        prepared: PreparedNativeImageImport,
        cancellation: &CancellationToken,
    ) -> Result<ImportedNativeImage, ExecutionFailure> {
        cancellation.check().map_err(import_cancelled)?;
        let limits = PngLimits {
            max_input_bytes: MAXIMUM_IMAGE_INPUT_BYTES as usize,
            max_dimension: 8_192,
            max_pixels: 4_194_304,
            max_decoder_allocation_bytes: 64 * 1024 * 1024,
        };
        decode_png(&prepared.bytes, limits).map_err(|error| {
            ExecutionFailure::new("native_image_import_decode_failed", error.to_string())
                .with_origin(ExecutionFailureOrigin::Validation)
        })?;
        cancellation.check().map_err(import_cancelled)?;
        let mut assets = self.assets.lock().map_err(import_filesystem_failure)?;
        cancellation.check().map_err(import_cancelled)?;
        let uploaded = assets
            .upload(
                UploadRequest {
                    namespace: AssetNamespace::Input,
                    filename: prepared.filename,
                    subfolder: PathBuf::new(),
                    bytes: prepared.bytes,
                    overwrite: false,
                    tags: Default::default(),
                },
                &self.authorization,
                cancellation,
            )
            .map_err(import_asset_failure)?;
        let identity = uploaded.record.identity;
        let source_name = identity
            .relative_path
            .to_str()
            .ok_or_else(|| {
                ExecutionFailure::new(
                    "native_image_import_name_invalid",
                    "The imported image name cannot be represented in the graph.",
                )
                .with_origin(ExecutionFailureOrigin::Validation)
            })?
            .to_owned();
        Ok(ImportedNativeImage {
            identity,
            source_name,
        })
    }

    fn with_request<T>(
        &self,
        cancellation: &CancellationToken,
        operation: impl FnOnce() -> Result<T, ExecutionFailure>,
    ) -> Result<T, ExecutionFailure> {
        let identifier = Uuid::new_v4();
        {
            let mut active = self.active.lock().map_err(import_filesystem_failure)?;
            self.lifetime.check().map_err(import_cancelled)?;
            cancellation.check().map_err(import_cancelled)?;
            if active.len() >= MAXIMUM_ACTIVE_IMPORTS {
                return Err(ExecutionFailure::new(
                    "native_image_asset_busy",
                    "Too many image asset requests are already in progress.",
                )
                .with_origin(ExecutionFailureOrigin::Validation));
            }
            active.insert(identifier, cancellation.clone());
        }
        let result = operation();
        self.active
            .lock()
            .map_err(import_filesystem_failure)?
            .remove(&identifier);
        result
    }
}

impl NativeImageImporter for NativeImageImportService {
    fn binding(&self) -> NativeImageImportBinding {
        self.binding.clone()
    }

    fn maximum_input_bytes(&self) -> u64 {
        MAXIMUM_IMAGE_INPUT_BYTES
    }

    fn import(
        &self,
        prepared: PreparedNativeImageImport,
        cancellation: &CancellationToken,
    ) -> Result<ImportedNativeImage, ExecutionFailure> {
        self.with_request(cancellation, || self.prepare_import(prepared, cancellation))
    }
}

impl NativeImageOutputReader for NativeImageImportService {
    fn binding(&self) -> NativeImageImportBinding {
        self.binding.clone()
    }

    fn maximum_input_bytes(&self) -> u64 {
        MAXIMUM_IMAGE_INPUT_BYTES
    }

    fn read(
        &self,
        reference: &str,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u8>, ExecutionFailure> {
        self.with_request(cancellation, || {
            cancellation.check().map_err(import_cancelled)?;
            let assets = self.assets.lock().map_err(import_filesystem_failure)?;
            let identity = assets
                .roots()
                .identity_from_reference(reference)
                .map_err(import_asset_failure)?;
            assets
                .read_verified(
                    &identity,
                    &self.output_authorization,
                    cancellation,
                    MAXIMUM_IMAGE_INPUT_BYTES,
                )
                .map_err(import_asset_failure)
        })
    }
}

impl Drop for NativeImageImportService {
    fn drop(&mut self) {
        self.revoke();
    }
}

fn import_cancelled(error: impl ToString) -> ExecutionFailure {
    ExecutionFailure::new("native_image_import_cancelled", error.to_string())
        .with_origin(ExecutionFailureOrigin::Validation)
}

fn import_filesystem_failure(error: impl ToString) -> ExecutionFailure {
    ExecutionFailure::new("native_image_import_failed", error.to_string())
        .with_origin(ExecutionFailureOrigin::Filesystem)
        .retryable(true)
}

fn import_asset_failure(error: AssetError) -> ExecutionFailure {
    match error {
        AssetError::Cancelled => import_cancelled(error),
        AssetError::PermissionDenied { .. } | AssetError::ProfileMismatch { .. } => {
            ExecutionFailure::new("native_image_asset_denied", error.to_string())
                .with_origin(ExecutionFailureOrigin::Permission)
        }
        AssetError::Io { .. }
        | AssetError::ChangedDuringRead(_)
        | AssetError::Rollback { .. }
        | AssetError::AllocationFailed => import_filesystem_failure(error),
        _ => ExecutionFailure::new("native_image_asset_invalid", error.to_string())
            .with_origin(ExecutionFailureOrigin::Validation),
    }
}

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use super::*;
    use sim_media::encode_png_frame;
    use sim_runtime::{
        AssetRoots, AssetService, authorize_native_input_reader, authorize_native_output_committer,
    };
    use std::sync::Mutex;

    fn service() -> anyhow::Result<(tempfile::TempDir, Arc<NativeImageImportService>)> {
        let directory = tempfile::tempdir()?;
        let mut roots = Vec::new();
        for (namespace, name) in [
            (AssetNamespace::Input, "input"),
            (AssetNamespace::Output, "output"),
            (AssetNamespace::Temporary, "temporary"),
            (AssetNamespace::Model, "model"),
            (AssetNamespace::Plugin, "plugin"),
        ] {
            let path = directory.path().join(name);
            std::fs::create_dir(&path)?;
            roots.push((namespace, path));
        }
        let roots = AssetRoots::new(Uuid::new_v4().to_string(), roots)?;
        let assets = Arc::new(Mutex::new(AssetService::open(roots)?));
        Ok((directory, NativeImageImportService::new(assets)?))
    }

    fn png(red: f32) -> anyhow::Result<Vec<u8>> {
        Ok(encode_png_frame(
            &[red, 0.0, 0.0],
            1,
            1,
            1,
            3,
            0,
            &Default::default(),
            PngLimits::default(),
        )?)
    }

    #[test]
    fn image_import_uses_verified_assets_and_canonical_collisions() -> anyhow::Result<()> {
        let (_directory, service) = service()?;
        let cancellation = CancellationToken::default();
        let first_bytes = png(1.0)?;
        let first = service
            .import(
                PreparedNativeImageImport::new("image.png".into(), first_bytes.clone()),
                &cancellation,
            )
            .map_err(|failure| anyhow!("{}: {}", failure.code, failure.message))?;
        let duplicate = service
            .import(
                PreparedNativeImageImport::new("image.png".into(), first_bytes.clone()),
                &cancellation,
            )
            .map_err(|failure| anyhow!("{}: {}", failure.code, failure.message))?;
        let second_bytes = png(0.0)?;
        let second = service
            .import(
                PreparedNativeImageImport::new("image.png".into(), second_bytes.clone()),
                &cancellation,
            )
            .map_err(|failure| anyhow!("{}: {}", failure.code, failure.message))?;
        assert_eq!(first.identity(), duplicate.identity());
        assert_ne!(first.identity(), second.identity());
        assert_eq!(first.source_name(), "image.png");
        assert_eq!(first.identity().namespace, AssetNamespace::Input);
        let authorization = authorize_native_input_reader(service.binding.profile_id.clone())?;
        let assets = service.assets.lock().map_err(|error| anyhow!("{error}"))?;
        assert_eq!(
            assets.read_verified(
                first.identity(),
                &authorization,
                &cancellation,
                MAXIMUM_IMAGE_INPUT_BYTES,
            )?,
            first_bytes
        );
        assert_eq!(
            assets.read_verified(
                second.identity(),
                &authorization,
                &cancellation,
                MAXIMUM_IMAGE_INPUT_BYTES,
            )?,
            second_bytes
        );
        Ok(())
    }

    #[test]
    fn image_import_rejects_bad_png_paths_cancellation_and_revocation_before_upload()
    -> anyhow::Result<()> {
        let (directory, service) = service()?;
        let cancellation = CancellationToken::default();
        let malformed = service
            .import(
                PreparedNativeImageImport::new("bad.png".into(), vec![1, 2, 3]),
                &cancellation,
            )
            .err()
            .ok_or_else(|| anyhow!("malformed image was imported"))?;
        assert_eq!(malformed.code, "native_image_import_decode_failed");
        assert!(
            service
                .import(
                    PreparedNativeImageImport::new("../escape.png".into(), png(1.0)?),
                    &cancellation,
                )
                .is_err()
        );
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(
            service
                .import(
                    PreparedNativeImageImport::new("cancelled.png".into(), png(1.0)?),
                    &cancelled,
                )
                .is_err()
        );
        service.revoke();
        assert!(
            service
                .import(
                    PreparedNativeImageImport::new("revoked.png".into(), png(1.0)?),
                    &cancellation,
                )
                .is_err()
        );
        assert_eq!(
            std::fs::read_dir(directory.path().join("input"))?.count(),
            0
        );
        assert_eq!(
            service
                .active
                .lock()
                .map_err(|error| anyhow!("{error}"))?
                .len(),
            0
        );
        Ok(())
    }

    #[test]
    fn saved_image_reader_verifies_bytes_and_rejects_input_tamper_and_revocation()
    -> anyhow::Result<()> {
        let (directory, service) = service()?;
        let cancellation = CancellationToken::default();
        let bytes = png(1.0)?;
        let output_authorization =
            authorize_native_output_committer(service.binding.profile_id.clone())?;
        let uploaded = service
            .assets
            .lock()
            .map_err(|error| anyhow!("{error}"))?
            .upload(
                UploadRequest {
                    namespace: AssetNamespace::Output,
                    filename: "saved.png".to_owned(),
                    subfolder: PathBuf::new(),
                    bytes: bytes.clone(),
                    overwrite: false,
                    tags: Default::default(),
                },
                &output_authorization,
                &cancellation,
            )?;
        let reference = uploaded.record.identity.to_reference()?;
        assert_eq!(
            service
                .read(&reference, &cancellation)
                .map_err(|failure| anyhow!("{}: {}", failure.code, failure.message))?,
            bytes
        );
        let imported = service
            .import(
                PreparedNativeImageImport::new("input.png".into(), bytes),
                &cancellation,
            )
            .map_err(|failure| anyhow!("{}: {}", failure.code, failure.message))?;
        assert!(
            service
                .read(&imported.identity().to_reference()?, &cancellation)
                .is_err()
        );
        let mut other_profile = uploaded.record.identity;
        other_profile.profile_id = Uuid::new_v4().to_string();
        assert!(
            service
                .assets
                .lock()
                .map_err(|error| anyhow!("{error}"))?
                .read_verified(
                    &other_profile,
                    &service.output_authorization,
                    &cancellation,
                    MAXIMUM_IMAGE_INPUT_BYTES,
                )
                .is_err()
        );
        std::fs::write(directory.path().join("output/saved.png"), png(0.0)?)?;
        assert!(service.read(&reference, &cancellation).is_err());
        service.revoke();
        assert!(service.read(&reference, &cancellation).is_err());
        Ok(())
    }

    #[test]
    fn image_service_rebind_cancels_registered_operation_before_publication() -> anyhow::Result<()>
    {
        let (directory, service) = service()?;
        let cancellation = CancellationToken::default();
        let prepared = PreparedNativeImageImport::new("pending.png".into(), png(1.0)?);
        let result = service.with_request(&cancellation, || {
            service.revoke();
            service.prepare_import(prepared, &cancellation)
        });
        assert!(result.is_err());
        assert!(cancellation.is_cancelled());
        assert!(!directory.path().join("input/pending.png").exists());
        assert_eq!(
            service
                .active
                .lock()
                .map_err(|error| anyhow!("{error}"))?
                .len(),
            0
        );
        Ok(())
    }
}
