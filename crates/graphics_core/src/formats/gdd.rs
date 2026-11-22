use std::{
    collections::BTreeSet,
    io::{self, Cursor, Read, Seek, SeekFrom, Write},
    ops::Bound,
    path::PathBuf,
    sync::Arc,
};

pub use graphite_document_container::archive::ArchiveFormat;
use graphite_document_container::{
    AnyContainer, ByteHolder, Container, ContainerError,
    archive::{Archive, ArchiveWriter, Xz, Zip},
    backends::memory::MemoryBackend,
    validate_path,
};
use graphite_document_format::{
    ExportFormat, ExportOptions, GddV1, GddV1Layout, Layout, MANIFEST_CODEC, Manifest,
    io::path_for,
    manifest::{FORMAT_MAGIC, SUPPORTED_FORMAT_VERSION},
};
use graphite_graph_craft::application_io::resource::{HashMapResourceStorage, ResourceHash};
use zip::ZipArchive;

use super::OriginalJson;

mod xz;

pub struct GddArchiveBudget {
    pub max_archive_bytes: usize,
    pub max_entry_bytes: u64,
    pub max_total_entry_bytes: u64,
    pub max_entries: usize,
}

#[derive(Clone, Copy)]
pub struct GddExportContent {
    pub include_registry: bool,
    pub include_history: bool,
}

impl Default for GddExportContent {
    fn default() -> Self {
        let options = ExportOptions::default();
        Self {
            include_registry: options.include_registry,
            include_history: options.include_history,
        }
    }
}

impl GddExportContent {
    pub fn validate(self) -> Result<(), GddArchiveError> {
        self.options()
            .validate()
            .map_err(GddArchiveError::InvalidExportOptions)?;
        Ok(())
    }

    fn options(self) -> ExportOptions {
        ExportOptions {
            include_registry: self.include_registry,
            include_history: self.include_history,
            ..ExportOptions::default()
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GddArchiveError {
    #[error("This Graphite archive has an unsupported envelope.")]
    UnsupportedEnvelope,
    #[error("The Graphite archive exceeds its input or extracted-data budget.")]
    BudgetExceeded,
    #[error("The Graphite archive has inconsistent entry records.")]
    InconsistentEntries,
    #[error("Unsupported Graphite container format {format:?} version {version}.")]
    UnsupportedManifest { format: String, version: u32 },
    #[error("The Graphite archive is missing its recorded registry and history.")]
    MissingDocument,
    #[error("The Graphite document has no history to export.")]
    NoExportableHistory,
    #[error("{0}")]
    InvalidExportOptions(&'static str),
    #[error("An embedded Graphite resource has no recorded content hash.")]
    UnresolvedResource,
    #[error("The embedded Graphite resource {0} is missing or does not match its content hash.")]
    InvalidResource(ResourceHash),
    #[error("An archive entry could not be read: {0}")]
    Archive(#[from] zip::result::ZipError),
    #[error("An archive entry could not be decoded: {0}")]
    Io(#[from] std::io::Error),
    #[error("The Graphite manifest could not be read: {0}")]
    Manifest(#[from] serde_json::Error),
    #[error("The Graphite container could not be staged: {0}")]
    Container(#[from] ContainerError),
    #[error("The pinned Graphite document storage operation failed: {0}")]
    Storage(#[from] graphite_document_format::Error),
    #[error("The pinned Graphite writer could not export the archive: {0}")]
    Export(#[source] ContainerError),
}

#[derive(serde::Deserialize)]
struct ManifestHeader {
    format: String,
    format_version: u32,
}

pub struct PreservedGddV1 {
    original_bytes: Arc<[u8]>,
    entries: StagedArchive,
    document: GddV1,
}

impl PreservedGddV1 {
    pub fn original_bytes(&self) -> &[u8] {
        &self.original_bytes
    }

    pub fn manifest(&self) -> &Manifest {
        self.document.manifest()
    }

    pub fn read_entry(&self, path: &str) -> Result<ByteHolder, ContainerError> {
        self.entries.working.read(path)
    }

    pub fn history_len(&self) -> usize {
        self.document.session().history().count()
    }

    pub fn hot_log_len(&self) -> usize {
        self.document.session().hot_log().len()
    }

    pub fn folder_files(&self) -> Result<Vec<(PathBuf, Vec<u8>)>, ContainerError> {
        self.entries.folder_files()
    }

    pub async fn export_folder_files_with_content(
        &self,
        content: GddExportContent,
        budget: GddArchiveBudget,
    ) -> Result<Vec<(PathBuf, Vec<u8>)>, GddArchiveError> {
        if content.include_registry && content.include_history {
            return Ok(self.folder_files()?);
        }
        let bytes = self
            .export_filtered_archive(ArchiveFormat::Zip, content, &budget)
            .await?;
        Ok(stage_zip(&bytes, &budget)?.folder_files()?)
    }

    pub fn export_archive(
        &self,
        format: ArchiveFormat,
        max_archive_bytes: usize,
    ) -> Result<Vec<u8>, GddArchiveError> {
        match format {
            ArchiveFormat::Zip => self.write_archive::<Zip>(max_archive_bytes),
            ArchiveFormat::Xz => self.write_archive::<Xz>(max_archive_bytes),
        }
        .map_err(GddArchiveError::Export)
    }

    pub async fn export_archive_with_content(
        &self,
        format: ArchiveFormat,
        content: GddExportContent,
        budget: GddArchiveBudget,
    ) -> Result<Vec<u8>, GddArchiveError> {
        if content.include_registry && content.include_history {
            return self.export_archive(format, budget.max_archive_bytes);
        }
        self.export_filtered_archive(format, content, &budget).await
    }

    async fn export_filtered_archive(
        &self,
        format: ArchiveFormat,
        content: GddExportContent,
        budget: &GddArchiveBudget,
    ) -> Result<Vec<u8>, GddArchiveError> {
        content.validate()?;
        if content.include_history && self.history_len() == 0 {
            return Err(GddArchiveError::NoExportableHistory);
        }
        let format = match format {
            ArchiveFormat::Zip => ExportFormat::Zip,
            ArchiveFormat::Xz => ExportFormat::Xz,
        };
        let bytes = self
            .document
            .export_to_bytes(
                format,
                content.options(),
                &HashMapResourceStorage::new(),
                None,
            )
            .await?;
        if bytes.len() > budget.max_archive_bytes {
            return Err(GddArchiveError::BudgetExceeded);
        }
        Ok(bytes)
    }

    fn write_archive<Format: Archive>(
        &self,
        max_archive_bytes: usize,
    ) -> Result<Vec<u8>, ContainerError>
    where
        Format::Writer<BoundedArchive>: ArchiveWriter<Sink = BoundedArchive>,
    {
        let mut archive = Format::writer(BoundedArchive {
            bytes: Cursor::new(Vec::new()),
            limit: max_archive_bytes as u64,
        })?;
        for path in &self.entries.names {
            archive.write_entry(path, self.read_entry(path)?.as_slice())?;
        }
        Ok(archive.finish_into()?.bytes.into_inner())
    }
}

struct BoundedArchive {
    bytes: Cursor<Vec<u8>>,
    limit: u64,
}

impl Write for BoundedArchive {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.bytes.position().saturating_add(bytes.len() as u64) > self.limit {
            return Err(io::Error::new(
                io::ErrorKind::FileTooLarge,
                "Graphite archive export exceeds its byte budget",
            ));
        }
        self.bytes.write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.bytes.flush()
    }
}

impl Seek for BoundedArchive {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.bytes.seek(position)
    }
}

pub async fn open_gdd_zip(
    bytes: &Arc<[u8]>,
    budget: GddArchiveBudget,
) -> Result<PreservedGddV1, GddArchiveError> {
    let working = stage_zip(bytes, &budget)?;
    open_staged_gdd(bytes, working).await
}

pub async fn open_gdd_archive(
    bytes: &Arc<[u8]>,
    budget: GddArchiveBudget,
) -> Result<PreservedGddV1, GddArchiveError> {
    match ArchiveFormat::detect(bytes) {
        Some(ArchiveFormat::Zip) => open_gdd_zip(bytes, budget).await,
        Some(ArchiveFormat::Xz) => {
            let working = xz::stage_xz(bytes, &budget)?;
            open_staged_gdd(bytes, working).await
        }
        None => Err(GddArchiveError::UnsupportedEnvelope),
    }
}

async fn open_staged_gdd(
    bytes: &Arc<[u8]>,
    entries: StagedArchive,
) -> Result<PreservedGddV1, GddArchiveError> {
    let working = &entries.working;
    let layout = GddV1Layout;
    let path = path_for(layout.manifest_basename(), MANIFEST_CODEC);
    let OriginalJson(value) = serde_json::from_slice(working.read(&path)?.as_slice())?;
    let header = serde::Deserialize::deserialize(&value)?;
    let ManifestHeader {
        format,
        format_version,
    } = header;
    if format != FORMAT_MAGIC || format_version != SUPPORTED_FORMAT_VERSION {
        return Err(GddArchiveError::UnsupportedManifest {
            format,
            version: format_version,
        });
    }
    let manifest: Manifest = serde_json::from_value(value)?;
    let registry_exists = working.exists(&path_for(
        layout.registry_basename(),
        manifest.codecs.registry,
    ));
    let history_exists = working.exists(&path_for(
        layout.history_basename(),
        manifest.codecs.history,
    ));
    if !registry_exists && !history_exists {
        return Err(GddArchiveError::MissingDocument);
    }
    let document = GddV1::open_in(AnyContainer::Memory(working.clone()), layout).await?;
    for resource in document.registry().resources.values() {
        if resource.has_embedded_source() {
            let hash = resource.hash.ok_or(GddArchiveError::UnresolvedResource)?;
            if !document.has_resource(&hash).await {
                return Err(GddArchiveError::InvalidResource(hash));
            }
        }
    }
    Ok(PreservedGddV1 {
        original_bytes: bytes.clone(),
        entries,
        document,
    })
}

fn stage_zip(bytes: &[u8], budget: &GddArchiveBudget) -> Result<StagedArchive, GddArchiveError> {
    if bytes.len() > budget.max_archive_bytes {
        return Err(GddArchiveError::BudgetExceeded);
    }
    let (entry_count, directory_start) = zip_directory(bytes, budget.max_entries)?;
    let mut archive = ZipArchive::new(Cursor::new(bytes))?;
    // The ZIP reader folds identical names into its index. Require every
    // original directory entry to survive before installing any payloads.
    if archive.len() != entry_count || archive.central_directory_start() != directory_start {
        return Err(GddArchiveError::InconsistentEntries);
    }

    let mut entries = StagedEntries::new(budget);
    let mut next_header = 0_u64;
    for index in 0..entry_count {
        let entry = archive.by_index(index)?;
        if !entry.is_file() || entry.header_start() != next_header {
            return Err(GddArchiveError::InconsistentEntries);
        }
        next_header = entry
            .data_start()
            .and_then(|start| start.checked_add(entry.compressed_size()))
            .filter(|end| *end <= directory_start)
            .ok_or(GddArchiveError::InconsistentEntries)?;
        let name = entry.name().to_owned();
        let size = entry.size();
        entries.stage(name, size, entry)?;
    }
    if next_header != directory_start {
        return Err(GddArchiveError::InconsistentEntries);
    }
    entries.finish()
}

struct StagedEntries<'a> {
    budget: &'a GddArchiveBudget,
    working: MemoryBackend,
    names: BTreeSet<String>,
    total_bytes: u64,
}

struct StagedArchive {
    working: MemoryBackend,
    names: BTreeSet<String>,
}

impl StagedArchive {
    fn folder_files(&self) -> Result<Vec<(PathBuf, Vec<u8>)>, ContainerError> {
        self.names
            .iter()
            .map(|path| {
                let bytes = match self.working.read(path)? {
                    ByteHolder::Owned(bytes) => bytes,
                    #[cfg(not(target_family = "wasm"))]
                    other => other.as_slice().to_vec(),
                };
                Ok((PathBuf::from(path), bytes))
            })
            .collect()
    }
}

impl<'a> StagedEntries<'a> {
    fn new(budget: &'a GddArchiveBudget) -> Self {
        Self {
            budget,
            working: MemoryBackend::new(),
            names: BTreeSet::new(),
            total_bytes: 0,
        }
    }

    fn stage(&mut self, name: String, size: u64, input: impl Read) -> Result<(), GddArchiveError> {
        self.total_bytes = self.total_bytes.saturating_add(size);
        if self.names.len() >= self.budget.max_entries
            || size > self.budget.max_entry_bytes
            || self.total_bytes > self.budget.max_total_entry_bytes
        {
            return Err(GddArchiveError::BudgetExceeded);
        }
        validate_path(&name)?;
        if name.contains('\0') || !self.names.insert(name.clone()) {
            return Err(GddArchiveError::InconsistentEntries);
        }
        let mut materialized = Vec::new();
        input
            .take(size.saturating_add(1))
            .read_to_end(&mut materialized)?;
        if u64::try_from(materialized.len()) != Ok(size) {
            return Err(GddArchiveError::InconsistentEntries);
        }
        if let Some((directory, filename)) = name.split_once('/')
            && directory == GddV1Layout.resources_dir()
            && let Ok(hash) = filename.parse::<ResourceHash>()
            && ResourceHash::from(materialized.as_slice()) != hash
        {
            return Err(GddArchiveError::InvalidResource(hash));
        }
        self.working.write(&name, &materialized)?;
        Ok(())
    }

    fn finish(self) -> Result<StagedArchive, GddArchiveError> {
        if self.names.iter().any(|name| {
            let prefix = format!("{name}/");
            self.names
                .range::<str, _>((Bound::Included(prefix.as_str()), Bound::Unbounded))
                .next()
                .is_some_and(|descendant| descendant.starts_with(&prefix))
        }) {
            return Err(GddArchiveError::InconsistentEntries);
        }
        Ok(StagedArchive {
            working: self.working,
            names: self.names,
        })
    }
}

fn zip_directory(bytes: &[u8], max_entries: usize) -> Result<(usize, u64), GddArchiveError> {
    if !bytes.starts_with(b"PK\x03\x04") {
        return Err(GddArchiveError::UnsupportedEnvelope);
    }
    // The pinned exporter writes a single-volume ZIP with no archive comment.
    let footer = bytes
        .last_chunk::<22>()
        .ok_or(GddArchiveError::UnsupportedEnvelope)?;
    if !footer.starts_with(b"PK\x05\x06\0\0\0\0") || footer[20..] != [0, 0] {
        return Err(GddArchiveError::UnsupportedEnvelope);
    }
    let disk_entry_count = u16::from_le_bytes([footer[8], footer[9]]);
    let total_entry_count = u16::from_le_bytes([footer[10], footer[11]]);
    let entry_count = usize::from(total_entry_count);
    let directory_size = u64::from(u32::from_le_bytes([
        footer[12], footer[13], footer[14], footer[15],
    ]));
    let directory_start = u64::from(u32::from_le_bytes([
        footer[16], footer[17], footer[18], footer[19],
    ]));
    if entry_count == 0
        || disk_entry_count != total_entry_count
        || directory_start + directory_size != (bytes.len() - footer.len()) as u64
        || directory_size < 46 * entry_count as u64
    {
        return Err(GddArchiveError::InconsistentEntries);
    }
    if entry_count > max_entries {
        return Err(GddArchiveError::BudgetExceeded);
    }
    Ok((entry_count, directory_start))
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphite_document_format::Codec;
    use graphite_editor::messages::portfolio::document::DocumentMessageHandler;
    use graphite_graph_craft::application_io::resource::ResourceId;
    use std::path::Path;

    fn budget() -> GddArchiveBudget {
        GddArchiveBudget {
            max_archive_bytes: 1024 * 1024,
            max_entry_bytes: 256 * 1024,
            max_total_entry_bytes: 1024 * 1024,
            max_entries: 32,
        }
    }

    fn document() -> GddV1 {
        GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            0x459,
            "pinned Graphite editor".into(),
            "pinned Graphite library".into(),
        )
        .expect("actual Graphite working copy")
    }

    async fn export(document: &GddV1, legacy: Option<&[u8]>) -> Arc<[u8]> {
        document
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &HashMapResourceStorage::new(),
                legacy,
            )
            .await
            .expect("actual Graphite ZIP export")
            .into()
    }

    #[test]
    fn at_gra_gdd_009_010_011_exports_registry_history_and_complete_outputs() {
        futures::executor::block_on(async {
            for codec in [Codec::Postcard, Codec::Json] {
                let mut owner = document();
                if codec == Codec::Json {
                    owner
                        .update_manifest(|manifest| {
                            manifest.codecs.registry = Codec::Json;
                            manifest.codecs.history = Codec::JsonLines;
                        })
                        .expect("original codec selection");
                }
                let resource_id = ResourceId::from(32_u64);
                let resource = b"resource in source history";
                let hash = ResourceHash::from(resource.as_slice());
                owner
                    .add_resource(resource_id, resource)
                    .expect("authentic history before export");
                let original = export(&owner, Some(b"source legacy contents")).await;
                let source = open_gdd_archive(&original, budget())
                    .await
                    .expect("actual source GDD");
                let codecs = source.manifest().codecs;
                let registry = path_for(GddV1Layout.registry_basename(), codecs.registry);
                let history = path_for(GddV1Layout.history_basename(), codecs.history);
                for (include_registry, include_history) in
                    [(true, false), (false, true), (true, true)]
                {
                    let content = GddExportContent {
                        include_registry,
                        include_history,
                    };
                    for format in [ArchiveFormat::Zip, ArchiveFormat::Xz] {
                        let bytes = Arc::<[u8]>::from(
                            source
                                .export_archive_with_content(format, content, budget())
                                .await
                                .expect("source-qualified content export"),
                        );
                        let exported = open_gdd_archive(&bytes, budget())
                            .await
                            .expect("reopen selected archive contents");
                        assert_eq!(exported.manifest().codecs.registry, codec);
                        assert_eq!(exported.read_entry(&registry).is_ok(), include_registry);
                        assert_eq!(exported.read_entry(&history).is_ok(), include_history);
                        assert_eq!(
                            exported.read_entry(GddV1Layout.legacy_path()).is_ok(),
                            include_registry && include_history
                        );
                        assert_eq!(
                            exported
                                .document
                                .registry()
                                .resources
                                .get(&resource_id)
                                .and_then(|entry| entry.hash),
                            Some(hash)
                        );
                        assert_eq!(
                            exported
                                .read_entry(&GddV1Layout.resource_path(&hash))
                                .expect("retained embedded resource")
                                .as_slice(),
                            resource
                        );
                    }
                    let folder = source
                        .export_folder_files_with_content(content, budget())
                        .await
                        .expect("export folder contents");
                    assert_eq!(
                        folder.iter().any(|(path, _)| path == Path::new(&registry)),
                        include_registry
                    );
                    assert_eq!(
                        folder.iter().any(|(path, _)| path == Path::new(&history)),
                        include_history
                    );
                }
                assert_eq!(source.original_bytes(), original.as_ref());
                assert!(
                    source.read_entry(&registry).is_ok() && source.read_entry(&history).is_ok()
                );
            }
        });
    }

    #[test]
    fn at_gra_gdd_012_rejects_no_contents_and_an_empty_history_export() {
        futures::executor::block_on(async {
            let original = export(&document(), None).await;
            let source = open_gdd_archive(&original, budget())
                .await
                .expect("valid new document");
            let none = GddExportContent {
                include_registry: false,
                include_history: false,
            };
            let empty_history = GddExportContent {
                include_registry: false,
                include_history: true,
            };
            assert!(matches!(
                source
                    .export_archive_with_content(ArchiveFormat::Zip, none, budget())
                    .await,
                Err(GddArchiveError::InvalidExportOptions(_))
            ));
            assert!(matches!(
                source
                    .export_folder_files_with_content(none, budget())
                    .await,
                Err(GddArchiveError::InvalidExportOptions(_))
            ));
            assert!(matches!(
                source
                    .export_archive_with_content(ArchiveFormat::Zip, empty_history, budget())
                    .await,
                Err(GddArchiveError::NoExportableHistory)
            ));
            assert_eq!(source.original_bytes(), original.as_ref());
        });
    }

    #[test]
    fn at_gra_gdd_007_and_008_exports_every_entry_and_reopens_with_both_original_codecs() {
        futures::executor::block_on(async {
            for codec in [Codec::Postcard, Codec::Json] {
                let mut original = document();
                original
                    .add_resource(ResourceId::from(79_u64), b"original image bytes")
                    .expect("actual resource and its history");
                if codec == Codec::Json {
                    original
                        .update_manifest(|manifest| {
                            manifest.codecs.registry = Codec::Json;
                            manifest.codecs.history = Codec::JsonLines;
                        })
                        .expect("original codecs");
                }
                let legacy = DocumentMessageHandler::default().serialize_document();
                let original_bytes = export(&original, Some(legacy.as_bytes())).await;
                let source = open_gdd_zip(&original_bytes, budget())
                    .await
                    .expect("pinned ZIP");
                let mut fixture = Zip::writer(Cursor::new(Vec::new())).expect("pinned writer");
                for path in &source.entries.names {
                    fixture
                        .write_entry(
                            path,
                            source.read_entry(path).expect("source entry").as_slice(),
                        )
                        .expect("write raw source entry");
                }
                fixture
                    .write_entry("opaque/extension/state.bin", &[0, 213, 2])
                    .expect("unknown entry");
                let source_bytes = Arc::<[u8]>::from(
                    fixture
                        .finish_into()
                        .expect("valid original envelope")
                        .into_inner(),
                );
                let source = open_gdd_zip(&source_bytes, budget())
                    .await
                    .expect("admit full source");
                let entries = &source.entries.names;
                for format in [ArchiveFormat::Zip, ArchiveFormat::Xz] {
                    assert!(source.export_archive(format, 5).is_err());
                    let bytes = Arc::<[u8]>::from(
                        source
                            .export_archive(format, budget().max_archive_bytes)
                            .expect("export complete source copy"),
                    );
                    assert_eq!(ArchiveFormat::detect(&bytes), Some(format));
                    let copy = open_gdd_archive(&bytes, budget())
                        .await
                        .expect("reopen native export");
                    let pinned = GddV1::open_from_archive(
                        &bytes,
                        AnyContainer::Memory(MemoryBackend::new()),
                        GddV1Layout,
                    )
                    .await
                    .expect("original Graphite reader opens the export");
                    assert_eq!(copy.manifest().codecs.registry, codec);
                    assert_eq!(pinned.manifest().codecs.registry, codec);
                    assert_eq!(copy.history_len(), source.history_len());
                    assert_eq!(pinned.session().history().count(), source.history_len());
                    assert_eq!(&copy.entries.names, entries);
                    for path in entries {
                        assert_eq!(
                            copy.read_entry(path).expect("copy entry").as_slice(),
                            source.read_entry(path).expect("source entry").as_slice(),
                            "{path}"
                        );
                    }
                    assert_eq!(source.original_bytes(), source_bytes.as_ref());
                }
            }
        });
    }

    #[test]
    fn at_r06_2_rejects_a_file_that_is_also_an_archive_parent() {
        futures::executor::block_on(async {
            let mut writer = Zip::writer(Cursor::new(Vec::new())).expect("pinned ZIP writer");
            for path in ["opaque", "opaque-annotation", "opaque/child.bin"] {
                writer.write_entry(path, b"fixture").expect("source entry");
            }
            let bytes = Arc::<[u8]>::from(writer.finish_into().expect("ZIP envelope").into_inner());
            assert!(matches!(
                open_gdd_archive(&bytes, budget()).await,
                Err(GddArchiveError::InconsistentEntries)
            ));
        });
    }

    #[test]
    fn at_gra_gdd_005_1_opens_the_pinned_xz_archive_with_resources_and_history() {
        futures::executor::block_on(async {
            let mut source = document();
            let resource_id = ResourceId::from(19_u64);
            let resource = b"resource in the original compressed form";
            let hash = ResourceHash::from(resource.as_slice());
            source
                .add_resource(resource_id, resource)
                .expect("actual Graphite resource history");
            let legacy = DocumentMessageHandler::default().serialize_document();
            let bytes = Arc::<[u8]>::from(
                source
                    .export_to_bytes(
                        ExportFormat::Xz,
                        ExportOptions::default(),
                        &HashMapResourceStorage::new(),
                        Some(legacy.as_bytes()),
                    )
                    .await
                    .expect("original Graphite XZ export"),
            );
            let opened = open_gdd_archive(&bytes, budget())
                .await
                .expect("open source-authored XZ");
            assert_eq!(opened.original_bytes(), bytes.as_ref());
            assert_eq!(opened.manifest().document_id, 0x459);
            assert_eq!(opened.history_len(), source.session().history().count());
            assert_eq!(
                opened
                    .read_entry(&GddV1Layout.resource_path(&hash))
                    .expect("literal resource contents")
                    .as_slice(),
                resource
            );
            assert_eq!(
                opened
                    .read_entry(GddV1Layout.legacy_path())
                    .expect("original legacy contents")
                    .as_slice(),
                legacy.as_bytes()
            );
        });
    }

    #[test]
    fn at_gra_gdd_005_1_preserves_xz_failures_without_installing_entries() {
        futures::executor::block_on(async {
            let mut writer = Xz::writer(Cursor::new(Vec::new())).expect("pinned XZ writer");
            writer
                .write_entry("notes.txt", b"first")
                .expect("first record");
            writer
                .write_entry("notes.txt", b"second")
                .expect("duplicate record");
            let raw = writer
                .finish_into()
                .expect("complete XZ stream")
                .into_inner();
            let bytes = Arc::<[u8]>::from(raw.as_slice());
            assert!(matches!(
                open_gdd_archive(&bytes, budget()).await,
                Err(GddArchiveError::InconsistentEntries)
            ));
            assert_eq!(bytes.as_ref(), raw);

            let shortened = Arc::<[u8]>::from(&raw[..raw.len() - 1]);
            assert!(matches!(
                open_gdd_archive(&shortened, budget()).await,
                Err(GddArchiveError::InconsistentEntries)
            ));

            let mut tight = budget();
            tight.max_archive_bytes = 1;
            assert!(matches!(
                open_gdd_archive(&bytes, tight).await,
                Err(GddArchiveError::BudgetExceeded)
            ));
            let mut tight = budget();
            tight.max_entry_bytes = 4;
            assert!(matches!(
                open_gdd_archive(&bytes, tight).await,
                Err(GddArchiveError::BudgetExceeded)
            ));
        });
    }

    #[test]
    fn at_r06_2_opens_pinned_zip_with_recorded_codecs_history_and_resources() {
        futures::executor::block_on(async {
            for codec in [Codec::Postcard, Codec::Json] {
                let mut original = document();
                let resource_id = ResourceId::from(17_u64);
                let resource_bytes = b"persistent Graphite resource";
                let hash = ResourceHash::from(resource_bytes.as_slice());
                original
                    .add_resource(resource_id, resource_bytes)
                    .expect("commit the actual Graphite resource to history");
                if codec == Codec::Json {
                    original
                        .update_manifest(|manifest| {
                            manifest.codecs.registry = Codec::Json;
                            manifest.codecs.history = Codec::JsonLines;
                        })
                        .expect("configure the source-defined JSON layout");
                }
                let legacy = DocumentMessageHandler::default().serialize_document();
                let bytes = export(&original, Some(legacy.as_bytes())).await;
                let opened = open_gdd_zip(&bytes, budget())
                    .await
                    .expect("read the native source container");
                assert_eq!(opened.original_bytes(), bytes.as_ref());
                assert_eq!(opened.manifest().document_id, 0x459);
                assert_eq!(opened.manifest().codecs.registry, codec);
                assert!(opened.history_len() > 0);
                assert_eq!(opened.history_len(), original.session().history().count());
                assert_eq!(opened.hot_log_len(), original.session().hot_log().len());
                assert_eq!(
                    opened
                        .document
                        .registry()
                        .resources
                        .get(&resource_id)
                        .and_then(|entry| entry.hash),
                    Some(hash)
                );
                assert_eq!(
                    opened
                        .read_entry(&GddV1Layout.resource_path(&hash))
                        .expect("embedded bytes")
                        .as_slice(),
                    resource_bytes
                );
                assert_eq!(
                    opened
                        .read_entry(GddV1Layout.legacy_path())
                        .expect("embedded legacy snapshot")
                        .as_slice(),
                    legacy.as_bytes()
                );
            }
        });
    }

    #[test]
    fn at_r06_2_rejects_future_manifest_and_unreadable_archives_without_changing_original() {
        futures::executor::block_on(async {
            let mut original = document();
            original
                .update_manifest(|manifest| manifest.format_version = SUPPORTED_FORMAT_VERSION + 1)
                .expect("write a future manifest through the original owner");
            let bytes = export(&original, None).await;
            assert!(matches!(
                open_gdd_zip(&bytes, budget()).await,
                Err(GddArchiveError::UnsupportedManifest { format, version })
                    if format == FORMAT_MAGIC && version == SUPPORTED_FORMAT_VERSION + 1
            ));
            let truncated: Arc<[u8]> = bytes[..bytes.len() - 1].into();
            assert!(matches!(
                open_gdd_zip(&truncated, budget()).await,
                Err(GddArchiveError::UnsupportedEnvelope)
            ));
            let mut tight = budget();
            tight.max_entry_bytes = 1;
            assert!(matches!(
                open_gdd_zip(&bytes, tight).await,
                Err(GddArchiveError::BudgetExceeded)
            ));
            let restored: Arc<[u8]> = export(&original, None).await;
            assert_eq!(bytes, restored);
        });
    }

    #[test]
    fn at_r06_2_preserves_unknown_entries_and_rejects_duplicate_manifest_fields() {
        futures::executor::block_on(async {
            let preserved = |manifest: &[u8]| {
                let mut writer =
                    Zip::writer(Cursor::new(Vec::new())).expect("pinned archive writer");
                writer
                    .write_entry("manifest.json", manifest)
                    .expect("manifest");
                let document = document();
                let registry = Codec::Postcard
                    .write_single(document.registry())
                    .expect("pinned registry codec");
                writer
                    .write_entry("registry.bin", &registry)
                    .expect("registry");
                writer
                    .write_entry("future/opaque.bin", &[45, 0, 192])
                    .expect("opaque optional entry");
                Arc::<[u8]>::from(writer.finish_into().expect("complete archive").into_inner())
            };
            let source = document();
            let manifest = MANIFEST_CODEC
                .write_single(source.manifest())
                .expect("actual Graphite manifest");
            let bytes = preserved(&manifest);
            let opened = open_gdd_zip(&bytes, budget())
                .await
                .expect("open valid V1 storage");
            assert_eq!(opened.original_bytes(), bytes.as_ref());
            assert_eq!(
                opened
                    .read_entry("future/opaque.bin")
                    .expect("unknown bytes are retained")
                    .as_slice(),
                &[45, 0, 192]
            );
            let mut duplicate = br#"{"format_version":999,"#.to_vec();
            duplicate.extend_from_slice(&manifest[1..]);
            assert!(matches!(
                open_gdd_zip(&preserved(&duplicate), budget()).await,
                Err(GddArchiveError::Manifest(error)) if error.to_string().contains("duplicate object key")
            ));
        });
    }

    #[test]
    fn at_r06_2_rejects_missing_and_mismatched_embedded_resources() {
        futures::executor::block_on(async {
            let mut source = document();
            let resource = b"actual resource";
            let hash = ResourceHash::from(resource.as_slice());
            source
                .add_resource(ResourceId::from(18_u64), resource)
                .expect("actual Graphite registry entry");
            let path = GddV1Layout.resource_path(&hash);
            let manifest = MANIFEST_CODEC
                .write_single(source.manifest())
                .expect("actual manifest");
            let registry = Codec::Postcard
                .write_single(source.registry())
                .expect("actual registry");
            for replacement in [None, Some(b"wrong contents".as_slice())] {
                let mut writer =
                    Zip::writer(Cursor::new(Vec::new())).expect("pinned archive writer");
                writer
                    .write_entry("manifest.json", &manifest)
                    .expect("manifest");
                writer
                    .write_entry("registry.bin", &registry)
                    .expect("registry");
                if let Some(replacement) = replacement {
                    writer
                        .write_entry(&path, replacement)
                        .expect("mismatched resource file");
                }
                let bytes =
                    Arc::<[u8]>::from(writer.finish_into().expect("complete archive").into_inner());
                assert!(matches!(
                    open_gdd_zip(&bytes, budget()).await,
                    Err(GddArchiveError::InvalidResource(actual)) if actual == hash
                ));
            }
        });
    }
}
