pub mod agent_plugins;
mod capability_granter;
pub mod extension_settings;
pub mod headless_host;
pub mod wasm_host;

#[cfg(test)]
mod extension_store_test;

use anyhow::{Context as _, Result, anyhow, bail};
use async_compression::futures::bufread::GzipDecoder;
use async_tar::Archive;
use client::{Client, proto, telemetry::Telemetry};
use cloud_api_types::{ExtensionMetadata, ExtensionProvides, GetExtensionsResponse};
use collections::{BTreeMap, BTreeSet, FxHashSet, HashMap, HashSet, btree_map};
pub use extension::ExtensionManifest;
use extension::extension_builder::{CompileExtensionOptions, ExtensionBuilder};
use extension::{
    ExtensionContextServerProxy, ExtensionDebugAdapterProviderProxy, ExtensionEvents,
    ExtensionGrammarProxy, ExtensionHostProxy, ExtensionLanguageProxy,
    ExtensionLanguageServerProxy, ExtensionSnippetProxy, ExtensionThemeProxy,
};
use fs::{Fs, Metadata, RemoveOptions, RenameOptions};
use futures::future::{BoxFuture, join_all};
use futures::{
    AsyncReadExt as _, Future, FutureExt as _, StreamExt as _,
    channel::{
        mpsc::{UnboundedSender, unbounded},
        oneshot,
    },
    io::BufReader,
    select_biased,
};
use gpui::{
    App, AppContext as _, AsyncApp, Context, Entity, EventEmitter, Global, Task, TaskExt,
    UpdateGlobal as _, WeakEntity, actions,
};
use http_client::{AsyncBody, HttpClient, HttpClientWithUrl};
use language::{
    LanguageConfig, LanguageMatcher, LanguageName, LanguageQueries, LoadedLanguage,
    QUERY_FILENAME_PREFIXES, Rope,
};
use node_runtime::NodeRuntime;
use project::ContextProviderWithTasks;
use release_channel::ReleaseChannel;
use remote::RemoteClient;
use semver::Version;
use serde::{Deserialize, Serialize};
use settings::{SemanticTokenRules, Settings, SettingsStore};
use sha2::{Digest as _, Sha256};
use std::ops::RangeInclusive;
use std::str::FromStr;
use std::sync::LazyLock;
use std::{
    cmp::Ordering,
    path::{self, Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use task::TaskTemplates;
use url::Url;
use util::{PathExt, ResultExt, paths::RemotePathBuf};
pub use wasm_host::{
    ComponentRuntime, WasmExtension, WasmHost,
    wit::{is_supported_wasm_api_version, wasm_api_version_range},
};

pub use extension::{
    ExtensionLibraryKind, GrammarManifestEntry, OldExtensionManifest, SchemaVersion,
};
pub use extension_settings::ExtensionSettings;

pub const RELOAD_DEBOUNCE_DURATION: Duration = Duration::from_millis(200);
const FS_WATCH_LATENCY: Duration = Duration::from_millis(100);

/// The current extension [`SchemaVersion`] supported by Zed.
const CURRENT_SCHEMA_VERSION: SchemaVersion = SchemaVersion(1);

pub const SIM_COMPONENT_MANIFEST_FILE: &str = "sim-plugin.json";
pub const SIM_COMPONENT_BINARY_FILE: &str = "sim-plugin.wasm";
const MAXIMUM_SIM_COMPONENT_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
const MAXIMUM_SIM_COMPONENT_BINARY_BYTES: u64 = 64 * 1024 * 1024;
const MAXIMUM_EXTENSION_INDEX_BYTES: u64 = 16 * 1024 * 1024;
const MAXIMUM_SIM_COMPONENT_COUNT: usize = 1_024;
const MAXIMUM_SIM_COMPONENT_INVENTORY_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstalledComponent {
    extension_id: Arc<str>,
    extension_version: Arc<str>,
    manifest_bytes: Arc<[u8]>,
    component_bytes: Arc<[u8]>,
}

impl InstalledComponent {
    pub fn checked(
        extension_id: Arc<str>,
        extension_version: Arc<str>,
        manifest_bytes: Arc<[u8]>,
        component_bytes: Arc<[u8]>,
    ) -> Result<Self> {
        validate_component_extension_id(&extension_id)?;
        let manifest_length = u64::try_from(manifest_bytes.len())
            .map_err(|_| anyhow!("component manifest for `{extension_id}` is oversized"))?;
        validate_component_payload_length(
            manifest_length,
            MAXIMUM_SIM_COMPONENT_MANIFEST_BYTES,
            "component manifest",
            &extension_id,
        )?;
        let component_length = u64::try_from(component_bytes.len())
            .map_err(|_| anyhow!("component binary for `{extension_id}` is oversized"))?;
        validate_component_payload_length(
            component_length,
            MAXIMUM_SIM_COMPONENT_BINARY_BYTES,
            "component binary",
            &extension_id,
        )?;
        Ok(Self {
            extension_id,
            extension_version,
            manifest_bytes,
            component_bytes,
        })
    }

    pub fn extension_id(&self) -> &str {
        &self.extension_id
    }

    pub fn extension_version(&self) -> &str {
        &self.extension_version
    }

    pub fn manifest_bytes(&self) -> &[u8] {
        &self.manifest_bytes
    }

    pub fn component_bytes(&self) -> &[u8] {
        &self.component_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentInventoryCandidateDescriptorIdentity {
    extension_id: Arc<str>,
    extension_version: Arc<str>,
    manifest_sha256: Arc<str>,
    component_sha256: Arc<str>,
}

impl ComponentInventoryCandidateDescriptorIdentity {
    pub fn extension_id(&self) -> &str {
        &self.extension_id
    }

    pub fn extension_version(&self) -> &str {
        &self.extension_version
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn component_sha256(&self) -> &str {
        &self.component_sha256
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentInventoryCandidateIdentity {
    sha256: Arc<str>,
    descriptors: Arc<[ComponentInventoryCandidateDescriptorIdentity]>,
}

impl ComponentInventoryCandidateIdentity {
    pub fn as_sha256(&self) -> &str {
        &self.sha256
    }

    pub fn descriptors(&self) -> &[ComponentInventoryCandidateDescriptorIdentity] {
        &self.descriptors
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentInventoryCandidate {
    components: Vec<InstalledComponent>,
    payload_bytes: u64,
    identity: ComponentInventoryCandidateIdentity,
}

impl ComponentInventoryCandidate {
    pub fn components(&self) -> &[InstalledComponent] {
        &self.components
    }

    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    pub fn identity_sha256(&self) -> &str {
        self.identity.as_sha256()
    }

    pub fn identity(&self) -> &ComponentInventoryCandidateIdentity {
        &self.identity
    }

    pub fn into_components(self) -> Vec<InstalledComponent> {
        self.components
    }

    pub fn into_parts(self) -> (Vec<InstalledComponent>, ComponentInventoryCandidateIdentity) {
        (self.components, self.identity)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentInventoryRejection {
    diagnostic: Arc<str>,
}

impl ComponentInventoryRejection {
    fn new(error: impl std::fmt::Display) -> Self {
        Self {
            diagnostic: Arc::from(error.to_string()),
        }
    }

    pub fn diagnostic(&self) -> &str {
        &self.diagnostic
    }
}

impl std::fmt::Display for ComponentInventoryRejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.diagnostic)
    }
}

impl std::error::Error for ComponentInventoryRejection {}

pub trait ComponentLifecycleAdapter: Send + Sync {
    fn adapter_id(&self) -> &'static str;

    fn synchronize_candidate(
        &self,
        candidate: ComponentInventoryCandidate,
    ) -> BoxFuture<'static, std::result::Result<(), String>> {
        self.synchronize(candidate.into_components())
    }

    fn synchronize(
        &self,
        components: Vec<InstalledComponent>,
    ) -> BoxFuture<'static, std::result::Result<(), String>>;
}

struct RegisteredComponentAdapters(Vec<Arc<dyn ComponentLifecycleAdapter>>);

impl Global for RegisteredComponentAdapters {}

pub fn register_component_lifecycle_adapter(
    adapter: Arc<dyn ComponentLifecycleAdapter>,
    cx: &mut App,
) -> Result<()> {
    if ExtensionStore::try_global(cx).is_some() {
        bail!(
            "component lifecycle adapters must be registered before ExtensionStore initialization"
        );
    }
    let adapter_id = adapter.adapter_id();
    if adapter_id.is_empty() || adapter_id.len() > 256 || adapter_id.chars().any(char::is_control) {
        bail!("component lifecycle adapter identifier is invalid");
    }
    if cx.try_global::<RegisteredComponentAdapters>().is_none() {
        cx.set_global(RegisteredComponentAdapters(Vec::new()));
    }
    let adapters = &mut cx.global_mut::<RegisteredComponentAdapters>().0;
    if adapters
        .iter()
        .any(|registered| registered.adapter_id() == adapter_id)
    {
        bail!("component lifecycle adapter `{adapter_id}` is already registered");
    }
    adapters.push(adapter);
    adapters.sort_by_key(|adapter| adapter.adapter_id());
    Ok(())
}

fn validate_component_extension_id(extension_id: &str) -> Result<()> {
    const MAXIMUM_EXTENSION_ID_BYTES: usize = 64 * 1024;

    let portable_name = !extension_id.is_empty()
        && extension_id.len() <= MAXIMUM_EXTENSION_ID_BYTES
        && !extension_id.contains(['/', '\\', ':'])
        && !extension_id.ends_with(['.', ' '])
        && !extension_id.chars().any(char::is_control);
    let device_name = extension_id
        .split_once('.')
        .map_or(extension_id, |(name, _)| name)
        .to_ascii_uppercase();
    let reserved_device_name = matches!(device_name.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || device_name
            .strip_prefix("COM")
            .or_else(|| device_name.strip_prefix("LPT"))
            .is_some_and(|suffix| {
                suffix.len() == 1
                    && suffix
                        .as_bytes()
                        .first()
                        .is_some_and(|byte| matches!(byte, b'1'..=b'9'))
            });
    let mut components = Path::new(extension_id).components();
    if !portable_name
        || reserved_device_name
        || !matches!(components.next(), Some(path::Component::Normal(_)))
        || components.next().is_some()
    {
        bail!("component extension identifier `{extension_id}` is not one normal path component");
    }
    Ok(())
}

fn checked_extension_dir(installed_dir: &Path, extension_id: &str) -> Result<PathBuf> {
    validate_component_extension_id(extension_id)?;
    Ok(installed_dir.join(extension_id))
}

fn validate_component_payload_length(
    length: u64,
    maximum_bytes: u64,
    subject: &str,
    extension_id: &str,
) -> Result<()> {
    if length > maximum_bytes {
        bail!("{subject} for `{extension_id}` is oversized");
    }
    Ok(())
}

fn validate_component_file_metadata(
    initial: &Metadata,
    current: &Metadata,
    loaded_bytes: u64,
    maximum_bytes: u64,
    subject: &str,
    extension_id: &str,
) -> Result<()> {
    if initial.is_dir
        || initial.is_symlink
        || initial.is_fifo
        || current.is_dir
        || current.is_symlink
        || current.is_fifo
    {
        bail!("{subject} for `{extension_id}` is invalid");
    }
    validate_component_payload_length(initial.len, maximum_bytes, subject, extension_id)?;
    validate_component_payload_length(current.len, maximum_bytes, subject, extension_id)?;
    validate_component_payload_length(loaded_bytes, maximum_bytes, subject, extension_id)?;
    if initial.inode != current.inode
        || initial.mtime != current.mtime
        || initial.len != current.len
        || loaded_bytes != current.len
    {
        bail!("{subject} for `{extension_id}` changed while it was being loaded");
    }
    Ok(())
}

async fn load_bounded_component_file(
    fs: Arc<dyn Fs>,
    path: &Path,
    initial_metadata: Metadata,
    maximum_bytes: u64,
    subject: &str,
    extension_id: &str,
) -> Result<Vec<u8>> {
    let bytes = fs
        .load_bytes(path)
        .await
        .with_context(|| format!("loading {subject} for `{extension_id}`"))?;
    let current_metadata = fs
        .metadata(path)
        .await
        .with_context(|| format!("revalidating {subject} metadata for `{extension_id}`"))?
        .ok_or_else(|| anyhow!("{subject} for `{extension_id}` disappeared while loading"))?;
    let loaded_bytes = u64::try_from(bytes.len())
        .map_err(|_| anyhow!("{subject} for `{extension_id}` is oversized"))?;
    validate_component_file_metadata(
        &initial_metadata,
        &current_metadata,
        loaded_bytes,
        maximum_bytes,
        subject,
        extension_id,
    )?;
    Ok(bytes)
}

fn format_component_adapter_errors(errors: &BTreeMap<String, String>) -> String {
    errors
        .iter()
        .map(|(adapter, error)| format!("component lifecycle adapter {adapter} failed: {error}"))
        .collect::<Vec<_>>()
        .join("; ")
}

fn component_inventory_adapter_errors(
    diagnostic: &str,
    adapters: &[Arc<dyn ComponentLifecycleAdapter>],
) -> BTreeMap<String, String> {
    log::error!("Failed to load installed components: {diagnostic}");
    adapters
        .iter()
        .map(|adapter| (adapter.adapter_id().to_owned(), diagnostic.to_owned()))
        .collect()
}

/// Extensions that should no longer be loaded or downloaded.
///
/// These snippets should no longer be downloaded or loaded, because their
/// functionality has been integrated into the core editor.
static SUPPRESSED_EXTENSIONS: LazyLock<FxHashSet<&str>> = LazyLock::new(|| {
    FxHashSet::from_iter([
        "snippets",
        "ruff",
        "ty",
        "basedpyright",
        "basher",
        // ACP
        "opencode",
        "mistral-vibe",
        "auggie",
        "stakpak",
        "codebuddy",
        "autohand-acp",
        "corust-agent",
        "factory-droid",
        "qqcode",
    ])
});

/// Returns the [`SchemaVersion`] range that is compatible with this version of Zed.
pub fn schema_version_range() -> RangeInclusive<SchemaVersion> {
    SchemaVersion::ZERO..=CURRENT_SCHEMA_VERSION
}

/// Returns whether the given extension version is compatible with this version of Zed.
pub fn is_version_compatible(
    release_channel: ReleaseChannel,
    extension_version: &ExtensionMetadata,
) -> bool {
    let schema_version = extension_version.manifest.schema_version.unwrap_or(0);
    if CURRENT_SCHEMA_VERSION.0 < schema_version {
        return false;
    }

    if let Some(wasm_api_version) = extension_version
        .manifest
        .wasm_api_version
        .as_ref()
        .and_then(|wasm_api_version| Version::from_str(wasm_api_version).ok())
        && !is_supported_wasm_api_version(release_channel, wasm_api_version)
    {
        return false;
    }

    true
}

pub struct ExtensionStore {
    pub agent_plugins: std::collections::BTreeMap<String, agent_plugins::InstalledPackage>,
    pub agent_plugins_directory: PathBuf,
    pub agent_plugins_error: Option<String>,
    pub agent_plugins_busy: bool,
    pub agent_plugins_diagnostics: std::collections::BTreeMap<String, Vec<String>>,
    #[cfg(feature = "agentic")]
    agent_plugins_subscription: Option<gpui::Subscription>,
    #[cfg(feature = "agentic")]
    agent_plugins_refresh: Option<Task<()>>,
    #[cfg(feature = "agentic")]
    agent_plugins_registered: std::collections::BTreeSet<Arc<str>>,
    pub proxy: Arc<ExtensionHostProxy>,
    pub builder: Arc<ExtensionBuilder>,
    pub extension_index: ExtensionIndex,
    pub fs: Arc<dyn Fs>,
    pub http_client: Arc<HttpClientWithUrl>,
    pub telemetry: Option<Arc<Telemetry>>,
    pub reload_tx: UnboundedSender<Option<Arc<str>>>,
    reload_complete_senders: Vec<oneshot::Sender<std::result::Result<(), Arc<str>>>>,
    pub installed_dir: PathBuf,
    pub staging_dir: PathBuf,
    pub outstanding_operations: BTreeMap<Arc<str>, ExtensionOperation>,
    pending_product_default_installs: HashSet<Arc<str>>,
    product_default_retry_generations: HashMap<Arc<str>, u64>,
    pub index_path: PathBuf,
    pub modified_extensions: HashSet<Arc<str>>,
    pub wasm_host: Arc<WasmHost>,
    pub wasm_extensions: Vec<(Arc<ExtensionManifest>, WasmExtension)>,
    pub tasks: Vec<Task<()>>,
    pub remote_clients: Vec<WeakEntity<RemoteClient>>,
    pub ssh_registered_tx: UnboundedSender<()>,
    component_lifecycle_adapters: Vec<Arc<dyn ComponentLifecycleAdapter>>,
    component_adapter_errors: BTreeMap<String, String>,
    accepted_component_candidate: Option<ComponentInventoryCandidate>,
}

#[derive(Clone, Copy)]
pub enum ExtensionOperation {
    Upgrade,
    Install,
    Remove,
}

#[derive(Clone)]
pub enum Event {
    ExtensionsUpdated,
    StartedReloading,
    ExtensionInstalled(Arc<str>),
    ExtensionUninstalled(Arc<str>),
    ExtensionFailedToLoad(Arc<str>),
}

impl EventEmitter<Event> for ExtensionStore {}

struct GlobalExtensionStore(Entity<ExtensionStore>);

impl Global for GlobalExtensionStore {}

#[derive(Debug, Deserialize, Serialize, Default, PartialEq, Eq)]
pub struct ExtensionIndex {
    pub extensions: BTreeMap<Arc<str>, ExtensionIndexEntry>,
    pub themes: BTreeMap<Arc<str>, ExtensionIndexThemeEntry>,
    #[serde(default)]
    pub icon_themes: BTreeMap<Arc<str>, ExtensionIndexIconThemeEntry>,
    pub languages: BTreeMap<LanguageName, ExtensionIndexLanguageEntry>,
}

impl ExtensionIndex {
    fn extensions_to_sync_to_remote(&self) -> RemoteSyncExtensions {
        let mut extensions = RemoteSyncExtensions::default();

        for (id, entry) in &self.extensions {
            if entry.manifest.remote_load().is_some() {
                extensions.insert_extension_and_language_dependencies(self, id);
            }
        }

        extensions
    }
}

#[derive(Default)]
struct RemoteSyncExtensions(HashMap<Arc<str>, ExtensionIndexEntry>);

impl RemoteSyncExtensions {
    fn insert_extension_and_language_dependencies(
        &mut self,
        index: &ExtensionIndex,
        id: &Arc<str>,
    ) {
        if self.0.contains_key(id) {
            return;
        }

        let Some(entry) = index.extensions.get(id) else {
            return;
        };

        self.0.insert(id.clone(), entry.clone());

        let Some(remote_load) = entry.manifest.remote_load() else {
            return;
        };

        for language in remote_load.language_dependencies() {
            if let Some(language_entry) = index.languages.get(&language) {
                self.insert_extension_and_language_dependencies(index, &language_entry.extension);
            }
        }
    }

    fn into_entries(self) -> impl Iterator<Item = (Arc<str>, ExtensionIndexEntry)> {
        self.0.into_iter()
    }
}

async fn copy_java_remote_runnables(
    fs: &dyn Fs,
    manifest: &ExtensionManifest,
    source_extension_dir: &Path,
    destination_extension_dir: &Path,
    source_language_dir: &Path,
    destination_language_dir: &Path,
    include_java_runnables: bool,
) -> Result<()> {
    if !include_java_runnables || manifest.id.as_ref() != "java" {
        return Ok(());
    }

    let config_path = source_language_dir.join(LanguageConfig::FILE_NAME);
    let config: LanguageConfig = toml::from_str(&fs.load(&config_path).await?)?;
    if config.name != LanguageName::new("Java") {
        return Ok(());
    }

    let grammar = config
        .grammar
        .context("Java remote runnable support requires a grammar")?;
    if !manifest.grammars.contains_key(&grammar) {
        bail!("Java grammar {grammar} is absent from the extension manifest");
    }
    let grammar_file = format!("{grammar}.wasm");
    let source_grammar = source_extension_dir.join("grammars").join(&grammar_file);
    let destination_grammar_dir = destination_extension_dir.join("grammars");
    fs.create_dir(&destination_grammar_dir).await?;
    fs.copy_file(
        &source_grammar,
        &destination_grammar_dir.join(grammar_file),
        fs::CopyOptions::default(),
    )
    .await
    .with_context(|| format!("Preparing Java grammar {source_grammar:?} for the remote host"))?;

    let source_query = source_language_dir.join("runnables.scm");
    fs.copy_file(
        &source_query,
        &destination_language_dir.join("runnables.scm"),
        fs::CopyOptions::default(),
    )
    .await
    .with_context(|| format!("Preparing Java runnable query {source_query:?} for the remote host"))
}

#[derive(Clone, PartialEq, Eq, Debug, Deserialize, Serialize)]
pub struct ExtensionIndexEntry {
    pub manifest: Arc<ExtensionManifest>,
    pub dev: bool,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Deserialize, Serialize)]
pub struct ExtensionIndexThemeEntry {
    pub extension: Arc<str>,
    pub path: PathBuf,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Deserialize, Serialize)]
pub struct ExtensionIndexIconThemeEntry {
    pub extension: Arc<str>,
    pub path: PathBuf,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Deserialize, Serialize)]
pub struct ExtensionIndexLanguageEntry {
    pub extension: Arc<str>,
    pub path: PathBuf,
    pub matcher: Arc<LanguageMatcher>,
    pub hidden: bool,
    pub grammar: Option<Arc<str>>,
}

actions!(
    zed,
    [
        /// Reloads all installed extensions.
        ReloadExtensions
    ]
);

pub fn init(
    extension_host_proxy: Arc<ExtensionHostProxy>,
    fs: Arc<dyn Fs>,
    client: Arc<Client>,
    node_runtime: NodeRuntime,
    cx: &mut App,
) {
    let component_lifecycle_adapters = cx
        .try_global::<RegisteredComponentAdapters>()
        .map(|adapters| adapters.0.clone())
        .unwrap_or_default();
    let store = cx.new(move |cx| {
        ExtensionStore::new(
            paths::extensions_dir().clone(),
            None,
            extension_host_proxy,
            fs,
            client.http_client(),
            client.http_client(),
            Some(client.telemetry().clone()),
            node_runtime,
            component_lifecycle_adapters,
            cx,
        )
    });

    cx.on_action(|_: &ReloadExtensions, cx| {
        let store = cx.global::<GlobalExtensionStore>().0.clone();
        let reload = store.update(cx, |store, cx| store.reload(None, cx));
        cx.spawn(async move |_| reload.await).detach_and_log_err(cx);
    });

    cx.set_global(GlobalExtensionStore(store));
}

impl ExtensionStore {
    pub fn try_global(cx: &App) -> Option<Entity<Self>> {
        cx.try_global::<GlobalExtensionStore>()
            .map(|store| store.0.clone())
    }

    pub fn global(cx: &App) -> Entity<Self> {
        cx.global::<GlobalExtensionStore>().0.clone()
    }

    pub fn component_adapter_errors(&self) -> &BTreeMap<String, String> {
        &self.component_adapter_errors
    }

    pub fn new(
        extensions_dir: PathBuf,
        build_dir: Option<PathBuf>,
        extension_host_proxy: Arc<ExtensionHostProxy>,
        fs: Arc<dyn Fs>,
        http_client: Arc<HttpClientWithUrl>,
        builder_client: Arc<dyn HttpClient>,
        telemetry: Option<Arc<Telemetry>>,
        node_runtime: NodeRuntime,
        component_lifecycle_adapters: Vec<Arc<dyn ComponentLifecycleAdapter>>,
        cx: &mut Context<Self>,
    ) -> Self {
        let agent_plugins_directory = extensions_dir.join("agent-plugins");
        let (agent_plugins, agent_plugins_error) =
            match cx
                .foreground_executor()
                .block_on(agent_plugins::load_inventory(
                    fs.as_ref(),
                    &agent_plugins_directory,
                )) {
                Ok(records) => (records, None),
                Err(error) => (
                    std::collections::BTreeMap::new(),
                    Some(format!("{error:#}")),
                ),
            };
        let work_dir = extensions_dir.join("work");
        let build_dir = build_dir.unwrap_or_else(|| extensions_dir.join("build"));
        let installed_dir = extensions_dir.join("installed");
        let staging_dir = extensions_dir.join("staging");
        let index_path = extensions_dir.join("index.json");

        let (reload_tx, mut reload_rx) = unbounded();
        let (connection_registered_tx, mut connection_registered_rx) = unbounded();
        let mut this = Self {
            agent_plugins,
            agent_plugins_directory,
            agent_plugins_error,
            agent_plugins_busy: false,
            agent_plugins_diagnostics: Default::default(),
            #[cfg(feature = "agentic")]
            agent_plugins_subscription: None,
            #[cfg(feature = "agentic")]
            agent_plugins_refresh: None,
            #[cfg(feature = "agentic")]
            agent_plugins_registered: Default::default(),
            proxy: extension_host_proxy.clone(),
            extension_index: Default::default(),
            installed_dir,
            staging_dir,
            index_path,
            builder: Arc::new(ExtensionBuilder::new(builder_client, build_dir)),
            outstanding_operations: Default::default(),
            pending_product_default_installs: Default::default(),
            product_default_retry_generations: Default::default(),
            modified_extensions: Default::default(),
            reload_complete_senders: Vec::new(),
            wasm_host: WasmHost::new(
                fs.clone(),
                http_client.clone(),
                node_runtime,
                extension_host_proxy,
                work_dir,
                cx,
            ),
            wasm_extensions: Vec::new(),
            fs,
            http_client,
            telemetry,
            reload_tx,
            tasks: Vec::new(),

            remote_clients: Default::default(),
            ssh_registered_tx: connection_registered_tx,
            component_lifecycle_adapters,
            component_adapter_errors: BTreeMap::new(),
            accepted_component_candidate: None,
        };

        // The extensions store maintains an index file, which contains a complete
        // list of the installed extensions and the resources that they provide.
        // This index is loaded synchronously on startup.
        let (index_content, index_metadata, extensions_metadata) =
            cx.foreground_executor().block_on(async {
                futures::join!(
                    this.fs.load(&this.index_path),
                    this.fs.metadata(&this.index_path),
                    this.fs.metadata(&this.installed_dir),
                )
            });

        // Normally, there is no need to rebuild the index. But if the index file
        // is invalid or is out-of-date according to the filesystem mtimes, then
        // it must be asynchronously rebuilt.
        let mut extension_index = ExtensionIndex::default();
        let mut extension_index_needs_rebuild = true;
        if let Ok(index_content) = index_content
            && let Some(index) = serde_json::from_str(&index_content).log_err()
        {
            extension_index = index;
            if let (Ok(Some(index_metadata)), Ok(Some(extensions_metadata))) =
                (index_metadata, extensions_metadata)
                && index_metadata
                    .mtime
                    .bad_is_greater_than(extensions_metadata.mtime)
            {
                extension_index_needs_rebuild = false;
            }
        }

        // Immediately load all of the extensions in the initial manifest. If the
        // index needs to be rebuild, then enqueue
        let load_initial_extensions =
            this.extensions_updated(extension_index, !extension_index_needs_rebuild, cx);
        let mut reload_future = None;
        if extension_index_needs_rebuild {
            reload_future = Some(this.reload(None, cx));
        }

        cx.spawn(async move |this, cx| {
            if let Some(future) = reload_future {
                if let Err(error) = future.await {
                    log::error!("Failed to reload extensions during initialization: {error:#}");
                }
            }
            this.update(cx, |this, cx| this.auto_install_extensions(cx))
                .ok();
            this.update(cx, |this, cx| this.check_for_updates(cx)).ok();
        })
        .detach();

        // Perform all extension loading in a single task to ensure that we
        // never attempt to simultaneously load/unload extensions from multiple
        // parallel tasks.
        this.tasks.push(cx.spawn(async move |this, cx| {
            async move {
                load_initial_extensions.await;

                let mut index_changed = false;
                let mut debounce_timer = cx.background_spawn(futures::future::pending()).fuse();

                loop {
                    select_biased! {
                        _ = debounce_timer => {
                            if index_changed {
                                let index = this
                                    .update(cx, |this, cx| this.rebuild_extension_index(cx))?
                                    .await;
                                this.update(cx, |this, cx| this.extensions_updated(index, true, cx))?
                                    .await;
                                index_changed = false;
                            }

                            Self::update_remote_clients(&this, cx).await?;
                        }
                        _ = connection_registered_rx.next() => {
                            debounce_timer = cx.background_executor().timer(RELOAD_DEBOUNCE_DURATION).fuse()
                        }
                        extension_id = reload_rx.next() => {
                            let Some(extension_id) = extension_id else { break; };
                            this.update(cx, |this, _cx| {
                                this.modified_extensions.extend(extension_id);
                            })?;
                            index_changed = true;
                            debounce_timer = cx.background_executor().timer(RELOAD_DEBOUNCE_DURATION).fuse()
                        }
                    }
                }

                anyhow::Ok(())
            }
            .map(drop)
            .await;
        }));

        // Watch the installed extensions directory for changes. Whenever changes are
        // detected, rebuild the extension index, and load/unload any extensions that
        // have been added, removed, or modified.
        this.tasks.push(cx.background_spawn({
            let fs = this.fs.clone();
            let reload_tx = this.reload_tx.clone();
            let installed_dir = this.installed_dir.clone();
            async move {
                let (mut paths, _) = fs.watch(&installed_dir, FS_WATCH_LATENCY).await;
                while let Some(events) = paths.next().await {
                    for event in events {
                        let Ok(event_path) = event.path.strip_prefix(&installed_dir) else {
                            continue;
                        };

                        if let Some(path::Component::Normal(extension_dir_name)) =
                            event_path.components().next()
                            && let Some(extension_id) = extension_dir_name.to_str()
                        {
                            reload_tx.unbounded_send(Some(extension_id.into())).ok();
                        }
                    }
                }
            }
        }));

        this
    }

    pub fn reload(
        &mut self,
        modified_extension: Option<Arc<str>>,
        cx: &mut Context<Self>,
    ) -> impl Future<Output = Result<()>> + use<> {
        let (tx, rx) = oneshot::channel();
        self.reload_complete_senders.push(tx);
        if self.reload_tx.unbounded_send(modified_extension).is_err() {
            self.complete_reloads(Err("extension reload task exited".to_owned()));
        } else {
            cx.emit(Event::StartedReloading);
        }

        async move {
            match rx.await {
                Ok(Ok(())) => Ok(()),
                Ok(Err(error)) => Err(anyhow!(error.to_string())),
                Err(_) => Err(anyhow!(
                    "extension reload exited before reporting its result"
                )),
            }
        }
    }

    fn complete_reloads(&mut self, result: std::result::Result<(), String>) {
        let result = result.map_err(Arc::<str>::from);
        for sender in self.reload_complete_senders.drain(..) {
            if sender.send(result.clone()).is_err() {
                log::debug!("extension reload completion receiver was dropped");
            }
        }
    }

    fn extensions_dir(&self) -> PathBuf {
        self.installed_dir.clone()
    }

    pub fn outstanding_operations(&self) -> &BTreeMap<Arc<str>, ExtensionOperation> {
        &self.outstanding_operations
    }

    pub fn installed_extensions(&self) -> &BTreeMap<Arc<str>, ExtensionIndexEntry> {
        &self.extension_index.extensions
    }

    pub fn dev_extensions(&self) -> impl Iterator<Item = &Arc<ExtensionManifest>> {
        self.extension_index
            .extensions
            .values()
            .filter_map(|extension| extension.dev.then_some(&extension.manifest))
    }

    pub fn extension_manifest_for_id(&self, extension_id: &str) -> Option<&Arc<ExtensionManifest>> {
        self.extension_index
            .extensions
            .get(extension_id)
            .map(|extension| &extension.manifest)
    }

    /// Returns the names of themes provided by extensions.
    pub fn extension_themes<'a>(
        &'a self,
        extension_id: &'a str,
    ) -> impl Iterator<Item = &'a Arc<str>> {
        self.extension_index
            .themes
            .iter()
            .filter_map(|(name, theme)| theme.extension.as_ref().eq(extension_id).then_some(name))
    }

    /// Returns the path to the theme file within an extension, if there is an
    /// extension that provides the theme.
    pub fn path_to_extension_theme(&self, theme_name: &str) -> Option<PathBuf> {
        let entry = self.extension_index.themes.get(theme_name)?;

        Some(
            self.extensions_dir()
                .join(entry.extension.as_ref())
                .join(&entry.path),
        )
    }

    /// Returns the names of icon themes provided by extensions.
    pub fn extension_icon_themes<'a>(
        &'a self,
        extension_id: &'a str,
    ) -> impl Iterator<Item = &'a Arc<str>> {
        self.extension_index
            .icon_themes
            .iter()
            .filter_map(|(name, icon_theme)| {
                icon_theme
                    .extension
                    .as_ref()
                    .eq(extension_id)
                    .then_some(name)
            })
    }

    /// Returns the path to the icon theme file within an extension, if there is
    /// an extension that provides the icon theme.
    pub fn path_to_extension_icon_theme(
        &self,
        icon_theme_name: &str,
    ) -> Option<(PathBuf, PathBuf)> {
        let entry = self.extension_index.icon_themes.get(icon_theme_name)?;

        let icon_theme_path = self
            .extensions_dir()
            .join(entry.extension.as_ref())
            .join(&entry.path);
        let icons_root_path = self.extensions_dir().join(entry.extension.as_ref());

        Some((icon_theme_path, icons_root_path))
    }

    pub fn fetch_extensions(
        &self,
        search: Option<&str>,
        provides_filter: Option<&BTreeSet<ExtensionProvides>>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Vec<ExtensionMetadata>>> {
        let version = CURRENT_SCHEMA_VERSION.to_string();
        let mut query = vec![("max_schema_version", version.as_str())];
        if let Some(search) = search {
            query.push(("filter", search));
        }

        let provides_filter = provides_filter.map(|provides_filter| {
            provides_filter
                .iter()
                .map(|provides| provides.to_string())
                .collect::<Vec<_>>()
                .join(",")
        });
        if let Some(provides_filter) = provides_filter.as_deref() {
            query.push(("provides", provides_filter));
        }

        self.fetch_extensions_from_api("/extensions", &query, cx)
    }

    pub fn fetch_extensions_with_update_available(
        &mut self,
        cx: &mut Context<Self>,
    ) -> Task<Result<Vec<ExtensionMetadata>>> {
        let schema_versions = schema_version_range();
        let wasm_api_versions = wasm_api_version_range(ReleaseChannel::global(cx));
        let extension_settings = ExtensionSettings::get_global(cx);
        let extension_ids = self
            .extension_index
            .extensions
            .iter()
            .filter(|(id, entry)| !entry.dev && extension_settings.should_auto_update(id))
            .map(|(id, _)| id.as_ref())
            .collect::<Vec<_>>()
            .join(",");
        let task = self.fetch_extensions_from_api(
            "/extensions/updates",
            &[
                ("min_schema_version", &schema_versions.start().to_string()),
                ("max_schema_version", &schema_versions.end().to_string()),
                (
                    "min_wasm_api_version",
                    &wasm_api_versions.start().to_string(),
                ),
                ("max_wasm_api_version", &wasm_api_versions.end().to_string()),
                ("ids", &extension_ids),
            ],
            cx,
        );
        cx.spawn(async move |this, cx| {
            let extensions = task.await?;
            this.update(cx, |this, _cx| {
                extensions
                    .into_iter()
                    .filter(|extension| {
                        this.extension_index
                            .extensions
                            .get(&extension.id)
                            .is_none_or(|installed_extension| {
                                installed_extension.manifest.version != extension.manifest.version
                            })
                    })
                    .collect()
            })
        })
    }

    pub fn fetch_extension_versions(
        &self,
        extension_id: &str,
        cx: &mut Context<Self>,
    ) -> Task<Result<Vec<ExtensionMetadata>>> {
        self.fetch_extensions_from_api(&format!("/extensions/{extension_id}"), &[], cx)
    }

    /// Installs any extensions that should be included with Zed by default.
    ///
    /// This can be used to make certain functionality provided by extensions
    /// available out-of-the-box.
    pub fn auto_install_extensions(&mut self, cx: &mut Context<Self>) {
        if cfg!(test) {
            return;
        }

        let extension_settings = ExtensionSettings::get_global(cx);

        let extensions_to_install = extension_settings
            .auto_install_extensions
            .keys()
            .filter(|extension_id| self.should_start_auto_install(extension_id, extension_settings))
            .cloned()
            .collect::<Vec<_>>();

        cx.spawn(async move |this, cx| {
            for extension_id in extensions_to_install {
                let install = this.update(cx, |this, cx| {
                    if !this
                        .should_start_auto_install(&extension_id, ExtensionSettings::get_global(cx))
                    {
                        None
                    } else {
                        let retry_generation = this.product_default_retry_generation(&extension_id);
                        Some((
                            this.install_latest_extension(extension_id.clone(), cx),
                            retry_generation,
                        ))
                    }
                })?;
                let Some((install, retry_generation)) = install else {
                    continue;
                };
                if let Err(error) = install.await {
                    if !this.read_with(cx, |this, cx| {
                        this.auto_install_result_is_current(&extension_id, retry_generation)
                            && ExtensionSettings::get_global(cx).should_auto_install(&extension_id)
                    })? {
                        continue;
                    }
                    let fs = this.read_with(cx, |this, _| this.fs.clone())?;
                    let reason = format!("{error:#}").chars().take(512).collect::<String>();
                    let completion = cx.update(|cx| {
                        settings::update_settings_file_with_completion(
                            fs,
                            cx,
                            move |settings, cx| {
                                if ExtensionStore::global(cx)
                                    .read(cx)
                                    .auto_install_result_is_current(&extension_id, retry_generation)
                                {
                                    settings
                                        .extension
                                        .record_auto_install_result(extension_id, Some(reason));
                                }
                            },
                        )
                    });
                    completion.await??;
                    this.update(cx, |_, cx| cx.notify())?;
                } else if !cx.update(|cx| {
                    ExtensionSettings::get_global(cx)
                        .auto_install_extensions
                        .get(&extension_id)
                        .copied()
                        .unwrap_or(true)
                }) {
                    this.update(cx, |this, cx| this.uninstall_extension(extension_id, cx))?
                        .await?;
                }
            }
            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
    }

    fn should_start_auto_install(&self, extension_id: &str, settings: &ExtensionSettings) -> bool {
        settings.should_auto_install(extension_id)
            && !self.pending_product_default_installs.contains(extension_id)
            && !self.outstanding_operations.contains_key(extension_id)
            && !self.extension_index.extensions.contains_key(extension_id)
            && !SUPPRESSED_EXTENSIONS.contains(extension_id)
    }

    fn product_default_retry_generation(&self, extension_id: &str) -> u64 {
        self.product_default_retry_generations
            .get(extension_id)
            .copied()
            .unwrap_or_default()
    }

    fn auto_install_result_is_current(&self, extension_id: &str, retry_generation: u64) -> bool {
        self.product_default_retry_generation(extension_id) == retry_generation
    }

    pub fn check_for_updates(&mut self, cx: &mut Context<Self>) {
        let task = self.fetch_extensions_with_update_available(cx);
        cx.spawn(async move |this, cx| Self::upgrade_extensions(this, task.await?, cx).await)
            .detach();
    }

    async fn upgrade_extensions(
        this: WeakEntity<Self>,
        extensions: Vec<ExtensionMetadata>,
        cx: &mut AsyncApp,
    ) -> Result<()> {
        for extension in extensions {
            let task = this.update(cx, |this, cx| {
                if let Some(installed_extension) =
                    this.extension_index.extensions.get(&extension.id)
                {
                    let installed_version =
                        Version::from_str(&installed_extension.manifest.version).ok()?;
                    let latest_version = Version::from_str(&extension.manifest.version).ok()?;

                    if installed_version >= latest_version {
                        return None;
                    }
                }

                Some(this.upgrade_extension(extension.id, extension.manifest.version, cx))
            })?;

            if let Some(task) = task {
                task.await.log_err();
            }
        }
        anyhow::Ok(())
    }

    fn fetch_extensions_from_api(
        &self,
        path: &str,
        query: &[(&str, &str)],
        cx: &mut Context<ExtensionStore>,
    ) -> Task<Result<Vec<ExtensionMetadata>>> {
        let url = self.http_client.build_zed_api_url(path, query);
        let http_client = self.http_client.clone();
        cx.spawn(async move |_, _| {
            let mut response = http_client
                .get(url?.as_ref(), AsyncBody::empty(), true)
                .await?;

            let mut body = Vec::new();
            response
                .body_mut()
                .read_to_end(&mut body)
                .await
                .context("error reading extensions")?;

            if response.status().is_client_error() {
                let text = String::from_utf8_lossy(body.as_slice());
                bail!(
                    "status error {}, response: {text:?}",
                    response.status().as_u16()
                );
            }

            let mut response: GetExtensionsResponse = serde_json::from_slice(&body)?;

            response
                .data
                .retain(|extension| !SUPPRESSED_EXTENSIONS.contains(extension.id.as_ref()));

            Ok(response.data)
        })
    }

    pub fn install_agent_plugin(
        &mut self,
        snapshot: agent_plugins::PackageSnapshot,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        if self.agent_plugins_busy || self.agent_plugins_error.is_some() {
            return Task::ready(Err(anyhow!("Package storage is busy or needs attention")));
        }
        if self
            .agent_plugins
            .values()
            .any(|record| record.manifest.name == snapshot.manifest.name)
        {
            return Task::ready(Err(anyhow!(
                "This package is already installed; use Update"
            )));
        }
        self.agent_plugins_busy = true;
        cx.notify();
        let fs = self.fs.clone();
        let directory = self.agent_plugins_directory.clone();
        let mut records = self.agent_plugins.clone();
        let (completion_tx, completion_rx) = oneshot::channel();
        // Once confirmed, finish publication even if the management view closes.
        cx.spawn(async move |this, cx| {
            let result = async {
                let record = agent_plugins::InstalledPackage::from_snapshot(&snapshot);
                agent_plugins::write_snapshot(fs.as_ref(), &directory, &record, &snapshot).await?;
                records.insert(record.id.clone(), record);
                agent_plugins::save_inventory(fs.as_ref(), &directory, &records).await?;
                anyhow::Ok(records)
            }
            .await;
            let result = this
                .update(cx, |this, cx| {
                    this.agent_plugins_busy = false;
                    cx.notify();
                    match result {
                        Ok(records) => {
                            this.agent_plugins = records;
                            cx.emit(Event::ExtensionsUpdated);
                            Ok(())
                        }
                        Err(error) => Err(error),
                    }
                })
                .and_then(|result| result);
            if let Err(Err(error)) = completion_tx.send(result) {
                log::error!("Agent Plugin installation failed: {error:#}");
            }
        })
        .detach();
        cx.background_spawn(async move {
            completion_rx
                .await
                .context("Package installation was interrupted")?
        })
    }

    pub fn change_agent_plugin(
        &mut self,
        id: &str,
        change: agent_plugins::PackageChange,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        if self.agent_plugins_busy || self.agent_plugins_error.is_some() {
            return Task::ready(Err(anyhow!("Package storage is busy or needs attention")));
        }
        let Some(mut record) = self.agent_plugins.get(id).cloned() else {
            return Task::ready(Err(anyhow!("Package is no longer installed")));
        };
        if let Err(error) = record.apply_change(&change) {
            return Task::ready(Err(error));
        }
        if change.retires_authority() {
            self.agent_plugins.insert(id.into(), record.clone());
            cx.emit(Event::ExtensionsUpdated);
        }
        self.agent_plugins_busy = true;
        cx.notify();
        let fs = self.fs.clone();
        let directory = self.agent_plugins_directory.clone();
        let mut records = self.agent_plugins.clone();
        let (completion_tx, completion_rx) = oneshot::channel();
        cx.spawn(async move |this, cx| {
            let result = async {
                match &change {
                    agent_plugins::PackageChange::Update(snapshot) => {
                        agent_plugins::write_snapshot(fs.as_ref(), &directory, &record, snapshot).await?;
                    }
                    agent_plugins::PackageChange::ApproveServer { .. } | agent_plugins::PackageChange::ApproveServerConfiguration { .. } => {
                        let snapshot = agent_plugins::capture_package(fs.as_ref(), &record.snapshot_path(&directory), agent_plugins::SnapshotLimits::default()).await?;
                        anyhow::ensure!(snapshot.revision == record.revision && snapshot.servers == record.servers, "Installed package changed; update and review it before activation");
                    }
                    _ => {}
                }
                records.insert(record.id.clone(), record.clone());
                agent_plugins::save_inventory(fs.as_ref(), &directory, &records).await.context("Could not persist package changes")?;
                if let agent_plugins::PackageChange::Uninstall { delete_data } = change {
                    fs.remove_dir(&directory.join("snapshots").join(&record.id), RemoveOptions { recursive: true, ignore_if_not_exists: true }).await.context("Package is disabled but its files could not be removed; retry uninstall")?;
                    if delete_data && fs.is_dir(&directory.join("data").join(&record.id)).await {
                        fs.remove_dir(&directory.join("data").join(&record.id), RemoveOptions { recursive: true, ignore_if_not_exists: true }).await.context("Package is disabled but its data could not be removed; retry uninstall")?;
                    }
                    records.remove(&record.id);
                    agent_plugins::save_inventory(fs.as_ref(), &directory, &records).await?;
                }
                anyhow::Ok(records)
            }.await;
            let result = this.update(cx, |this, cx| {
                this.agent_plugins_busy = false;
                cx.notify();
                match result {
                    Ok(records) => { this.agent_plugins = records; cx.emit(Event::ExtensionsUpdated); Ok(()) }
                    Err(error) => Err(error),
                }
            }).and_then(|result| result);
            if let Err(Err(error)) = completion_tx.send(result) {
                log::error!("Agent Plugin change failed: {error:#}");
            }
        }).detach();
        cx.background_spawn(async move {
            completion_rx
                .await
                .context("Package change was interrupted")?
        })
    }

    pub fn install_extension(
        &mut self,
        extension_id: Arc<str>,
        version: Arc<str>,
        cx: &mut Context<Self>,
    ) {
        self.install_or_upgrade_extension(extension_id, version, ExtensionOperation::Install, cx)
            .detach_and_log_err(cx);
    }

    fn install_or_upgrade_extension_at_endpoint(
        &mut self,
        extension_id: Arc<str>,
        url: Url,
        operation: ExtensionOperation,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let extension_dir = match checked_extension_dir(&self.installed_dir, &extension_id) {
            Ok(extension_dir) => extension_dir,
            Err(error) => return Task::ready(Err(error)),
        };
        let staging_dir = self.staging_dir.clone();
        let http_client = self.http_client.clone();
        let fs = self.fs.clone();

        match self.outstanding_operations.entry(extension_id.clone()) {
            btree_map::Entry::Occupied(_) => {
                return Task::ready(Err(anyhow!(
                    "another extension operation is in progress; retry after it finishes"
                )));
            }
            btree_map::Entry::Vacant(e) => e.insert(operation),
        };
        cx.notify();

        cx.spawn(async move |this, cx| {
            let _finish = cx.on_drop(&this, {
                let extension_id = extension_id.clone();
                move |this, cx| {
                    this.outstanding_operations.remove(extension_id.as_ref());
                    cx.notify();
                }
            });

            cx.background_spawn(async move {
                let mut response = http_client
                    .get(url.as_ref(), Default::default(), true)
                    .await
                    .context("downloading extension")?;

                let content_length = response
                    .headers()
                    .get(http_client::http::header::CONTENT_LENGTH)
                    .and_then(|value| value.to_str().ok()?.parse::<usize>().ok());

                let mut body = BufReader::new(response.body_mut());
                let mut tar_gz_bytes = Vec::new();
                body.read_to_end(&mut tar_gz_bytes).await?;

                if let Some(content_length) = content_length {
                    let actual_len = tar_gz_bytes.len();
                    if content_length != actual_len {
                        bail!(
                            "downloaded extension size {actual_len} \
                        does not match content length {content_length}"
                        );
                    }
                }

                let decompressed_bytes = GzipDecoder::new(BufReader::new(tar_gz_bytes.as_slice()));
                let archive = Archive::new(decompressed_bytes);

                let remove_dir = || {
                    fs.remove_dir(
                        &extension_dir,
                        RemoveOptions {
                            recursive: true,
                            ignore_if_not_exists: true,
                        },
                    )
                };

                let temp_dir = fs
                    .create_dir(&staging_dir)
                    .await
                    .and_then(|()| tempfile::tempdir_in(&staging_dir).map_err(Into::into));

                match temp_dir {
                    Ok(temp_dir) => {
                        archive.unpack(temp_dir.path()).await?;
                        remove_dir().await?;
                        fs.rename(
                            temp_dir.path(),
                            &extension_dir,
                            RenameOptions {
                                overwrite: true,
                                ignore_if_exists: true,
                                create_parents: true,
                            },
                        )
                        .await
                    }
                    Err(_) => {
                        remove_dir().await?;
                        archive.unpack(extension_dir).await.map_err(Into::into)
                    }
                }
            })
            .await?;

            this.update(cx, |this, cx| this.reload(Some(extension_id.clone()), cx))?
                .await?;

            if let ExtensionOperation::Install = operation {
                this.update(cx, |this, cx| {
                    cx.emit(Event::ExtensionInstalled(extension_id.clone()));
                    if let Some(events) = ExtensionEvents::try_global(cx)
                        && let Some(manifest) = this.extension_manifest_for_id(&extension_id)
                    {
                        events.update(cx, |this, cx| {
                            this.emit(extension::Event::ExtensionInstalled(manifest.clone()), cx)
                        });
                    }
                })
                .ok();
            }

            anyhow::Ok(())
        })
    }

    pub fn install_latest_extension(
        &mut self,
        extension_id: Arc<str>,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        log::info!("installing extension {extension_id} latest version");

        let schema_versions = schema_version_range();
        let wasm_api_versions = wasm_api_version_range(ReleaseChannel::global(cx));

        let url = match self.http_client.build_zed_api_url(
            &format!("/extensions/{extension_id}/download"),
            &[
                ("min_schema_version", &schema_versions.start().to_string()),
                ("max_schema_version", &schema_versions.end().to_string()),
                (
                    "min_wasm_api_version",
                    &wasm_api_versions.start().to_string(),
                ),
                ("max_wasm_api_version", &wasm_api_versions.end().to_string()),
            ],
        ) {
            Ok(url) => url,
            Err(error) => return Task::ready(Err(error)),
        };

        self.install_or_upgrade_extension_at_endpoint(
            extension_id,
            url,
            ExtensionOperation::Install,
            cx,
        )
    }

    pub fn product_default_extension_download_origin(&self) -> Result<Url> {
        self.http_client.build_zed_api_url("/", &[])
    }

    pub fn can_retry_product_default_extension(&self, id: &str, cx: &App) -> bool {
        self.can_retry_product_default_extension_from_defaults(
            id,
            product_flavor::DEFAULT_EXTENSIONS,
            cx,
        )
    }

    fn can_retry_product_default_extension_from_defaults(
        &self,
        id: &str,
        defaults: &[&str],
        cx: &App,
    ) -> bool {
        defaults.contains(&id)
            && ExtensionSettings::get_global(cx)
                .auto_install_extensions
                .get(id)
                != Some(&false)
            && !self.installed_extensions().contains_key(id)
            && !self.product_default_install_in_progress(id)
    }

    pub fn retry_product_default_extension_preserving_preferences(
        &mut self,
        id: Arc<str>,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        self.retry_product_default_extension_from_defaults(
            id,
            product_flavor::DEFAULT_EXTENSIONS,
            cx,
        )
    }

    fn retry_product_default_extension_from_defaults(
        &mut self,
        id: Arc<str>,
        defaults: &[&str],
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        if !self.can_retry_product_default_extension_from_defaults(&id, defaults, cx) {
            return Task::ready(Err(anyhow!(
                "Product extension is already installed, unavailable, declined or busy"
            )));
        }
        // A download retry must preserve explicit declines and their authorization evidence.
        self.install_latest_extension(id, cx)
    }

    pub fn install_product_default_extension(
        &mut self,
        extension_id: Arc<str>,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        if let Err(error) = self.reserve_product_default_install(extension_id.clone()) {
            return Task::ready(Err(error));
        }
        cx.notify();
        let fs = self.fs.clone();
        cx.spawn(async move |this, cx| {
            let _finish = cx.on_drop(&this, {
                let extension_id = extension_id.clone();
                move |this, cx| {
                    this.pending_product_default_installs.remove(&extension_id);
                    cx.notify();
                }
            });
            let retry = cx.update(|cx| {
                settings::update_settings_file_with_completion(fs.clone(), cx, {
                    let extension_id = extension_id.clone();
                    move |settings, _| settings.extension.retry_auto_install(extension_id)
                })
            });
            retry.await??;
            let install = this.update(cx, |this, cx| {
                this.start_product_default_install(extension_id.clone(), cx)
            })?;
            let failure = install.await.err();
            let message = failure
                .as_ref()
                .map(|error| format!("{error:#}").chars().take(512).collect::<String>());
            let completed_extension_id = extension_id.clone();
            let completion = cx.update(|cx| {
                settings::update_settings_file_with_completion(fs, cx, move |settings, _| {
                    settings
                        .extension
                        .record_auto_install_result(extension_id, message)
                })
            });
            completion.await??;

            if failure.is_none()
                && !cx.update(|cx| {
                    ExtensionSettings::get_global(cx)
                        .auto_install_extensions
                        .get(&completed_extension_id)
                        .copied()
                        .unwrap_or(true)
                })
            {
                this.update(cx, |this, cx| {
                    this.uninstall_extension(completed_extension_id, cx)
                })?
                .await?;
            }

            if let Some(error) = failure {
                return Err(error);
            }
            Ok(())
        })
    }

    fn reserve_product_default_install(&mut self, extension_id: Arc<str>) -> Result<()> {
        if self.outstanding_operations.contains_key(&extension_id)
            || !self
                .pending_product_default_installs
                .insert(extension_id.clone())
        {
            bail!("another extension operation is in progress; retry after it finishes");
        }
        let retry_generation = self
            .product_default_retry_generations
            .entry(extension_id)
            .or_default();
        *retry_generation = retry_generation.wrapping_add(1);
        Ok(())
    }

    fn start_product_default_install(
        &mut self,
        extension_id: Arc<str>,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        if !ExtensionSettings::get_global(cx).should_auto_install(&extension_id) {
            return Task::ready(Err(anyhow!(
                "extension is unavailable for automatic installation; enable it before retrying"
            )));
        }
        if self.outstanding_operations.contains_key(&extension_id) {
            return Task::ready(Err(anyhow!(
                "another extension operation is in progress; retry after it finishes"
            )));
        }
        self.install_latest_extension(extension_id, cx)
    }

    pub fn product_default_install_in_progress(&self, extension_id: &str) -> bool {
        self.pending_product_default_installs.contains(extension_id)
            || self.outstanding_operations.contains_key(extension_id)
    }

    pub fn upgrade_extension(
        &mut self,
        extension_id: Arc<str>,
        version: Arc<str>,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        self.install_or_upgrade_extension(extension_id, version, ExtensionOperation::Upgrade, cx)
    }

    fn install_or_upgrade_extension(
        &mut self,
        extension_id: Arc<str>,
        version: Arc<str>,
        operation: ExtensionOperation,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        log::info!("installing extension {extension_id} {version}");
        let url = match self.http_client.build_zed_api_url(
            &format!("/extensions/{extension_id}/{version}/download"),
            &[],
        ) {
            Ok(url) => url,
            Err(error) => return Task::ready(Err(error)),
        };

        self.install_or_upgrade_extension_at_endpoint(extension_id, url, operation, cx)
    }

    pub fn uninstall_extension(
        &mut self,
        extension_id: Arc<str>,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let extension_dir = match checked_extension_dir(&self.installed_dir, &extension_id) {
            Ok(extension_dir) => extension_dir,
            Err(error) => return Task::ready(Err(error)),
        };
        let work_dir = self.wasm_host.work_dir.join(extension_id.as_ref());
        let fs = self.fs.clone();

        let extension_manifest = self.extension_manifest_for_id(&extension_id).cloned();

        match self.outstanding_operations.entry(extension_id.clone()) {
            btree_map::Entry::Occupied(_) => {
                return Task::ready(Err(anyhow!(
                    "another extension operation is in progress; retry after it finishes"
                )));
            }
            btree_map::Entry::Vacant(e) => e.insert(ExtensionOperation::Remove),
        };

        cx.spawn(async move |extension_store, cx| {
            let _finish = cx.on_drop(&extension_store, {
                let extension_id = extension_id.clone();
                move |this, cx| {
                    this.outstanding_operations.remove(extension_id.as_ref());
                    cx.notify();
                }
            });

            fs.remove_dir(
                &extension_dir,
                RemoveOptions {
                    recursive: true,
                    ignore_if_not_exists: true,
                },
            )
            .await
            .with_context(|| format!("Removing extension dir {extension_dir:?}"))?;

            extension_store
                .update(cx, |extension_store, cx| extension_store.reload(None, cx))?
                .await?;

            // There's a race between wasm extension fully stopping and the directory removal.
            // On Windows, it's impossible to remove a directory that has a process running in it.
            for i in 0..3 {
                cx.background_executor()
                    .timer(Duration::from_millis(i * 100))
                    .await;
                let removal_result = fs
                    .remove_dir(
                        &work_dir,
                        RemoveOptions {
                            recursive: true,
                            ignore_if_not_exists: true,
                        },
                    )
                    .await;
                match removal_result {
                    Ok(()) => break,
                    Err(e) => {
                        if i == 2 {
                            log::error!("Failed to remove extension work dir {work_dir:?} : {e}");
                        }
                    }
                }
            }

            extension_store.update(cx, |_, cx| {
                cx.emit(Event::ExtensionUninstalled(extension_id.clone()));
                if let Some(events) = ExtensionEvents::try_global(cx)
                    && let Some(manifest) = extension_manifest
                {
                    events.update(cx, |this, cx| {
                        this.emit(extension::Event::ExtensionUninstalled(manifest.clone()), cx)
                    });
                }
            })?;

            anyhow::Ok(())
        })
    }

    pub fn install_dev_extension(
        &mut self,
        extension_source_path: PathBuf,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let extensions_dir = self.extensions_dir();
        let fs = self.fs.clone();
        let builder = self.builder.clone();

        cx.spawn(async move |this, cx| {
            let mut extension_manifest =
                ExtensionManifest::load(fs.clone(), &extension_source_path).await?;
            let extension_id = extension_manifest.id.clone();

            if let Some(uninstall_task) = this
                .update(cx, |this, cx| {
                    this.extension_index
                        .extensions
                        .get(extension_id.as_ref())
                        .is_some_and(|index_entry| !index_entry.dev)
                        .then(|| this.uninstall_extension(extension_id.clone(), cx))
                })
                .ok()
                .flatten()
            {
                uninstall_task.await.log_err();
            }

            if !this.update(cx, |this, cx| {
                match this.outstanding_operations.entry(extension_id.clone()) {
                    btree_map::Entry::Occupied(_) => return false,
                    btree_map::Entry::Vacant(e) => e.insert(ExtensionOperation::Install),
                };
                cx.notify();
                true
            })? {
                return Ok(());
            }

            let _finish = cx.on_drop(&this, {
                let extension_id = extension_id.clone();
                move |this, cx| {
                    this.outstanding_operations.remove(extension_id.as_ref());
                    cx.notify();
                }
            });

            cx.background_spawn({
                let extension_source_path = extension_source_path.clone();
                let fs = fs.clone();
                async move {
                    builder
                        .compile_extension(
                            &extension_source_path,
                            &mut extension_manifest,
                            CompileExtensionOptions::dev(),
                            fs,
                        )
                        .await
                }
            })
            .await
            .inspect_err(|error| {
                util::log_err(error);
            })?;

            let output_path = &extensions_dir.join(extension_id.as_ref());
            if let Some(metadata) = fs.metadata(output_path).await? {
                if metadata.is_symlink {
                    fs.remove_file(
                        output_path,
                        RemoveOptions {
                            recursive: false,
                            ignore_if_not_exists: true,
                        },
                    )
                    .await?;
                } else {
                    bail!("extension {extension_id} is still installed");
                }
            }

            fs.create_symlink(output_path, extension_source_path)
                .await?;

            this.update(cx, |this, cx| this.reload(None, cx))?.await?;
            this.update(cx, |this, cx| {
                cx.emit(Event::ExtensionInstalled(extension_id.clone()));
                if let Some(events) = ExtensionEvents::try_global(cx)
                    && let Some(manifest) = this.extension_manifest_for_id(&extension_id)
                {
                    events.update(cx, |this, cx| {
                        this.emit(extension::Event::ExtensionInstalled(manifest.clone()), cx)
                    });
                }
            })?;

            Ok(())
        })
    }

    pub fn rebuild_dev_extension(&mut self, extension_id: Arc<str>, cx: &mut Context<Self>) {
        let path = match checked_extension_dir(&self.installed_dir, &extension_id) {
            Ok(path) => path,
            Err(error) => {
                log::error!("cannot rebuild extension `{extension_id}`: {error}");
                return;
            }
        };
        let builder = self.builder.clone();
        let fs = self.fs.clone();

        match self.outstanding_operations.entry(extension_id.clone()) {
            btree_map::Entry::Occupied(_) => return,
            btree_map::Entry::Vacant(e) => e.insert(ExtensionOperation::Upgrade),
        };

        cx.notify();
        let compile = cx.background_spawn(async move {
            let mut manifest = ExtensionManifest::load(fs.clone(), &path).await?;
            builder
                .compile_extension(&path, &mut manifest, CompileExtensionOptions::dev(), fs)
                .await
        });

        cx.spawn(async move |this, cx| {
            let result = compile.await;

            this.update(cx, |this, cx| {
                this.outstanding_operations.remove(&extension_id);
                cx.notify();
            })?;

            if result.is_ok() {
                this.update(cx, |this, cx| this.reload(Some(extension_id), cx))?
                    .await?;
            }

            result
        })
        .detach_and_log_err(cx)
    }

    /// Updates the set of installed extensions.
    ///
    /// First, this unloads any themes, languages, or grammars that are
    /// no longer in the manifest, or whose files have changed on disk.
    /// Then it loads any themes, languages, or grammars that are newly
    /// added to the manifest, or whose files have changed on disk.
    #[ztracing::instrument(skip_all)]
    fn extensions_updated(
        &mut self,
        mut new_index: ExtensionIndex,
        component_inventory_is_canonical: bool,
        cx: &mut Context<Self>,
    ) -> Task<()> {
        let old_index = &self.extension_index;

        let suppressed_extensions_to_remove = new_index
            .extensions
            .extract_if(.., |extension_id, _| {
                SUPPRESSED_EXTENSIONS.contains(extension_id.as_ref())
            })
            .collect::<Vec<_>>();

        // Determine which extensions need to be loaded and unloaded, based
        // on the changes to the manifest and the extensions that we know have been
        // modified.
        let mut extensions_to_unload = Vec::default();
        let mut extensions_to_load = Vec::default();
        {
            let mut old_keys = old_index.extensions.iter().peekable();
            let mut new_keys = new_index.extensions.iter().peekable();
            loop {
                match (old_keys.peek(), new_keys.peek()) {
                    (None, None) => break,
                    (None, Some(_)) => {
                        extensions_to_load.push(new_keys.next().unwrap().0.clone());
                    }
                    (Some(_), None) => {
                        extensions_to_unload.push(old_keys.next().unwrap().0.clone());
                    }
                    (Some((old_key, _)), Some((new_key, _))) => match old_key.cmp(new_key) {
                        Ordering::Equal => {
                            let (old_key, old_value) = old_keys.next().unwrap();
                            let (new_key, new_value) = new_keys.next().unwrap();
                            if old_value != new_value || self.modified_extensions.contains(old_key)
                            {
                                extensions_to_unload.push(old_key.clone());
                                extensions_to_load.push(new_key.clone());
                            }
                        }
                        Ordering::Less => {
                            extensions_to_unload.push(old_keys.next().unwrap().0.clone());
                        }
                        Ordering::Greater => {
                            extensions_to_load.push(new_keys.next().unwrap().0.clone());
                        }
                    },
                }
            }
            self.modified_extensions.clear();
        }

        let trigger_suppressed_extension_removal =
            move |this: &mut ExtensionStore, cx: &mut Context<ExtensionStore>| {
                for (id, _) in suppressed_extensions_to_remove {
                    this.uninstall_extension(id, cx).detach_and_log_err(cx);
                }
            };

        if extensions_to_load.is_empty()
            && extensions_to_unload.is_empty()
            && self.component_lifecycle_adapters.is_empty()
        {
            self.complete_reloads(Ok(()));
            trigger_suppressed_extension_removal(self, cx);
            return Task::ready(());
        }

        let reload_count = extensions_to_unload
            .iter()
            .filter(|id| extensions_to_load.contains(id))
            .count();

        log::info!(
            "extensions updated. loading {}, reloading {}, unloading {}",
            extensions_to_load.len() - reload_count,
            reload_count,
            extensions_to_unload.len() - reload_count
        );

        let extension_ids = extensions_to_load
            .iter()
            .filter_map(|id| {
                Some((
                    id.clone(),
                    new_index.extensions.get(id)?.manifest.version.clone(),
                ))
            })
            .collect::<Vec<_>>();

        telemetry::event!("Extensions Loaded", id_and_versions = extension_ids);

        let themes_to_remove = old_index
            .themes
            .iter()
            .filter_map(|(name, entry)| {
                if extensions_to_unload.contains(&entry.extension) {
                    Some(name.clone().into())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let icon_themes_to_remove = old_index
            .icon_themes
            .iter()
            .filter_map(|(name, entry)| {
                if extensions_to_unload.contains(&entry.extension) {
                    Some(name.clone().into())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let languages_to_remove = old_index
            .languages
            .iter()
            .filter_map(|(name, entry)| {
                if extensions_to_unload.contains(&entry.extension) {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let mut grammars_to_remove = Vec::new();
        let mut server_removal_tasks = Vec::with_capacity(extensions_to_unload.len());
        for extension_id in &extensions_to_unload {
            let Some(extension) = old_index.extensions.get(extension_id) else {
                continue;
            };
            grammars_to_remove.extend(extension.manifest.grammars.keys().cloned());
            for (language_server_name, config) in &extension.manifest.language_servers {
                for language in config.languages() {
                    server_removal_tasks.push(self.proxy.remove_language_server(
                        &language,
                        language_server_name,
                        cx,
                    ));
                }
            }

            for server_id in extension.manifest.context_servers.keys() {
                self.proxy.unregister_context_server(server_id.clone(), cx);
            }
            for adapter in extension.manifest.debug_adapters.keys() {
                self.proxy.unregister_debug_adapter(adapter.clone());
            }
            for locator in extension.manifest.debug_locators.keys() {
                self.proxy.unregister_debug_locator(locator.clone());
            }
        }

        self.wasm_extensions
            .retain(|(extension, _)| !extensions_to_unload.contains(&extension.id));
        self.proxy.remove_user_themes(themes_to_remove);
        self.proxy.remove_icon_themes(icon_themes_to_remove);
        self.proxy
            .remove_languages(&languages_to_remove, &grammars_to_remove);

        // Remove semantic token rules for languages being unloaded.
        if !languages_to_remove.is_empty() {
            SettingsStore::update_global(cx, |store, cx| {
                for language in &languages_to_remove {
                    store.remove_language_semantic_token_rules(language.as_ref(), cx);
                }
            });
        }

        let mut grammars_to_add = Vec::new();
        let mut themes_to_add = Vec::new();
        let mut icon_themes_to_add = Vec::new();
        let mut snippets_to_add = Vec::new();
        for extension_id in &extensions_to_load {
            let Some(extension) = new_index.extensions.get(extension_id) else {
                continue;
            };

            grammars_to_add.extend(extension.manifest.grammars.keys().map(|grammar_name| {
                let mut grammar_path = self.installed_dir.clone();
                grammar_path.extend([extension_id.as_ref(), "grammars"]);
                grammar_path.push(grammar_name.as_ref());
                grammar_path.set_extension("wasm");
                (grammar_name.clone(), grammar_path)
            }));
            themes_to_add.extend(extension.manifest.themes.iter().map(|theme_path| {
                let mut path = self.installed_dir.clone();
                path.extend([Path::new(extension_id.as_ref()), theme_path.as_std_path()]);
                path
            }));
            icon_themes_to_add.extend(extension.manifest.icon_themes.iter().map(
                |icon_theme_path| {
                    let mut path = self.installed_dir.clone();
                    path.extend([
                        Path::new(extension_id.as_ref()),
                        icon_theme_path.as_std_path(),
                    ]);

                    let mut icons_root_path = self.installed_dir.clone();
                    icons_root_path.extend([Path::new(extension_id.as_ref())]);

                    (path, icons_root_path)
                },
            ));
            snippets_to_add.extend(extension.manifest.snippets.iter().flat_map(|snippets| {
                snippets.paths().map(|snippets_path| {
                    let mut path = self.installed_dir.clone();
                    path.extend([Path::new(extension_id.as_ref()), snippets_path.as_path()]);
                    path
                })
            }));
        }

        self.proxy.register_grammars(grammars_to_add);
        let languages_to_add = new_index
            .languages
            .iter()
            .filter(|(_, entry)| extensions_to_load.contains(&entry.extension))
            .collect::<Vec<_>>();
        let mut semantic_token_rules_to_add: Vec<(LanguageName, SemanticTokenRules)> = Vec::new();
        for (language_name, language) in languages_to_add {
            let mut language_path = self.installed_dir.clone();
            language_path.extend([
                Path::new(language.extension.as_ref()),
                language.path.as_path(),
            ]);

            // Load semantic token rules if present in the language directory.
            let rules_path = language_path.join(SemanticTokenRules::FILE_NAME);
            if std::fs::exists(&rules_path).is_ok_and(|exists| exists)
                && let Some(rules) = SemanticTokenRules::load(&rules_path).log_err()
            {
                semantic_token_rules_to_add.push((language_name.clone(), rules));
            }

            self.proxy.register_language(
                language_name.clone(),
                language.grammar.clone(),
                language.matcher.clone(),
                language.hidden,
                Arc::new(move || {
                    let config =
                        LanguageConfig::load(language_path.join(LanguageConfig::FILE_NAME))?;
                    let queries = load_plugin_queries(&language_path);
                    let context_provider =
                        std::fs::read_to_string(language_path.join(TaskTemplates::FILE_NAME))
                            .ok()
                            .and_then(|contents| {
                                let definitions =
                                    serde_json_lenient::from_str(&contents).log_err()?;
                                Some(Arc::new(ContextProviderWithTasks::new(definitions)) as Arc<_>)
                            });

                    Ok(LoadedLanguage {
                        config,
                        queries,
                        context_provider,
                        toolchain_provider: None,
                        manifest_name: None,
                    })
                }),
            );
        }

        // Register semantic token rules for newly loaded extension languages.
        if !semantic_token_rules_to_add.is_empty() {
            SettingsStore::update_global(cx, |store, cx| {
                for (language_name, rules) in semantic_token_rules_to_add {
                    store.set_language_semantic_token_rules(language_name.0.clone(), rules, cx);
                }
            });
        }

        let fs = self.fs.clone();
        let wasm_host = self.wasm_host.clone();
        let root_dir = self.installed_dir.clone();
        let proxy = self.proxy.clone();
        let extension_entries = extensions_to_load
            .iter()
            .filter_map(|name| new_index.extensions.get(name).cloned())
            .collect::<Vec<_>>();
        let component_index_json = if component_inventory_is_canonical {
            serde_json::to_string_pretty(&new_index).map_err(ComponentInventoryRejection::new)
        } else {
            Err(ComponentInventoryRejection::new(
                "extension index is unavailable or stale and must be rebuilt",
            ))
        };
        let component_extensions_dir = self
            .installed_dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.installed_dir.clone());
        let component_index_path = self.index_path.clone();
        let component_lifecycle_adapters = self.component_lifecycle_adapters.clone();
        let accepted_component_candidate = self.accepted_component_candidate.clone();
        self.extension_index = new_index;
        cx.notify();
        cx.emit(Event::ExtensionsUpdated);

        cx.spawn(async move |this, cx| {
            cx.background_spawn({
                let fs = fs.clone();
                async move {
                    let _ = join_all(server_removal_tasks).await;
                    for theme_path in themes_to_add {
                        proxy
                            .load_user_theme(theme_path, fs.clone())
                            .await
                            .log_err();
                    }

                    for (icon_theme_path, icons_root_path) in icon_themes_to_add {
                        proxy
                            .load_icon_theme(icon_theme_path, icons_root_path, fs.clone())
                            .await
                            .log_err();
                    }

                    for snippets_path in &snippets_to_add {
                        match fs
                            .load(snippets_path)
                            .await
                            .with_context(|| format!("Loading snippets from {snippets_path:?}"))
                        {
                            Ok(snippets_contents) => {
                                proxy
                                    .register_snippet(snippets_path, &snippets_contents)
                                    .log_err();
                            }
                            Err(e) => log::error!("Cannot load snippets: {e:#}"),
                        }
                    }
                }
            })
            .await;

            let (component_adapter_errors, accepted_component_candidate) = match async {
                let component_index_json = component_index_json?;
                fs.save(
                    &component_index_path,
                    &component_index_json.as_str().into(),
                    Default::default(),
                )
                .await
                .map_err(ComponentInventoryRejection::new)?;
                Self::canonical_component_inventory_candidate(fs.clone(), &component_extensions_dir)
                    .await
            }
            .await
            {
                Ok(candidate) if accepted_component_candidate.as_ref() == Some(&candidate) => {
                    (BTreeMap::new(), None)
                }
                Ok(candidate) => {
                    let errors = Self::synchronize_component_candidate(
                        candidate.clone(),
                        &component_lifecycle_adapters,
                    )
                    .await;
                    let accepted = errors.is_empty().then_some(candidate);
                    (errors, accepted)
                }
                Err(error) => (
                    component_inventory_adapter_errors(
                        error.diagnostic(),
                        &component_lifecycle_adapters,
                    ),
                    None,
                ),
            };

            let mut wasm_extensions = Vec::new();
            for extension in extension_entries {
                if extension.manifest.lib.kind.is_none() {
                    continue;
                };

                let extension_path = root_dir.join(extension.manifest.id.as_ref());
                let wasm_extension = WasmExtension::load(
                    &extension_path,
                    &extension.manifest,
                    wasm_host.clone(),
                    cx,
                )
                .await
                .with_context(|| format!("Loading extension from {extension_path:?}"));

                match wasm_extension {
                    Ok(wasm_extension) => {
                        wasm_extensions.push((extension.manifest.clone(), wasm_extension))
                    }
                    Err(e) => {
                        log::error!(
                            "Failed to load extension: {}, {:#}",
                            extension.manifest.id,
                            e
                        );
                        this.update(cx, |_, cx| {
                            cx.emit(Event::ExtensionFailedToLoad(extension.manifest.id.clone()))
                        })
                        .ok();
                    }
                }
            }

            this.update(cx, |this, cx| {
                this.component_adapter_errors = component_adapter_errors;
                if let Some(candidate) = accepted_component_candidate {
                    this.accepted_component_candidate = Some(candidate);
                }
                let reload_result = if this.component_adapter_errors.is_empty() {
                    Ok(())
                } else {
                    Err(format_component_adapter_errors(
                        &this.component_adapter_errors,
                    ))
                };
                this.complete_reloads(reload_result);

                for (manifest, wasm_extension) in &wasm_extensions {
                    let extension = Arc::new(wasm_extension.clone());

                    for (language_server_id, language_server_config) in &manifest.language_servers {
                        for language in language_server_config.languages() {
                            this.proxy.register_language_server(
                                extension.clone(),
                                language_server_id.clone(),
                                language.clone(),
                            );
                        }
                    }

                    for id in manifest.context_servers.keys() {
                        this.proxy
                            .register_context_server(extension.clone(), id.clone(), cx);
                    }

                    for (debug_adapter, meta) in &manifest.debug_adapters {
                        let mut path = root_dir.clone();
                        path.push(Path::new(manifest.id.as_ref()));
                        if let Some(schema_path) = &meta.schema_path {
                            path.push(schema_path);
                        } else {
                            path.push("debug_adapter_schemas");
                            path.push(Path::new(debug_adapter.as_ref()).with_extension("json"));
                        }

                        this.proxy.register_debug_adapter(
                            extension.clone(),
                            debug_adapter.clone(),
                            &path,
                        );
                    }

                    for debug_adapter in manifest.debug_locators.keys() {
                        this.proxy
                            .register_debug_locator(extension.clone(), debug_adapter.clone());
                    }
                }

                this.wasm_extensions.extend(wasm_extensions);
                this.proxy.set_extensions_loaded();
                this.proxy.reload_current_theme(cx);
                this.proxy.reload_current_icon_theme(cx);
                trigger_suppressed_extension_removal(this, cx);

                if let Some(events) = ExtensionEvents::try_global(cx) {
                    events.update(cx, |this, cx| {
                        this.emit(extension::Event::ExtensionsInstalledChanged, cx)
                    });
                }
            })
            .ok();
        })
    }

    pub async fn synchronize_component_adapters(
        fs: Arc<dyn Fs>,
        installed_dir: &Path,
        entries: &[ExtensionIndexEntry],
        adapters: &[Arc<dyn ComponentLifecycleAdapter>],
    ) -> BTreeMap<String, String> {
        if adapters.is_empty() {
            return BTreeMap::new();
        }

        let candidate = match Self::component_inventory_candidate_from_entries(
            fs,
            installed_dir,
            entries,
        )
        .await
        {
            Ok(candidate) => candidate,
            Err(error) => {
                return component_inventory_adapter_errors(error.diagnostic(), adapters);
            }
        };

        Self::synchronize_component_candidate(candidate, adapters).await
    }

    async fn synchronize_component_candidate(
        candidate: ComponentInventoryCandidate,
        adapters: &[Arc<dyn ComponentLifecycleAdapter>],
    ) -> BTreeMap<String, String> {
        let mut errors = BTreeMap::new();
        for adapter in adapters {
            if let Err(error) = adapter.synchronize_candidate(candidate.clone()).await {
                log::error!(
                    "Component lifecycle adapter {} failed: {error}",
                    adapter.adapter_id()
                );
                errors.insert(adapter.adapter_id().to_owned(), error);
            }
        }
        errors
    }

    pub async fn canonical_component_inventory_candidate(
        fs: Arc<dyn Fs>,
        extensions_dir: &Path,
    ) -> Result<ComponentInventoryCandidate, ComponentInventoryRejection> {
        let index_path = extensions_dir.join("index.json");
        let installed_dir = extensions_dir.join("installed");
        let installed_metadata = fs
            .metadata(&installed_dir)
            .await
            .map_err(ComponentInventoryRejection::new)?
            .ok_or_else(|| {
                ComponentInventoryRejection::new("installed extension directory is missing")
            })?;
        if installed_metadata.is_symlink || !installed_metadata.is_dir {
            return Err(ComponentInventoryRejection::new(
                "installed extension directory is not a direct real directory",
            ));
        }
        let metadata = fs
            .metadata(&index_path)
            .await
            .map_err(ComponentInventoryRejection::new)?
            .ok_or_else(|| ComponentInventoryRejection::new("extension index is missing"))?;
        if !metadata.mtime.bad_is_greater_than(installed_metadata.mtime) {
            return Err(ComponentInventoryRejection::new(
                "extension index is stale relative to the installed directory",
            ));
        }
        validate_component_file_metadata(
            &metadata,
            &metadata,
            metadata.len,
            MAXIMUM_EXTENSION_INDEX_BYTES,
            "extension index",
            "installed",
        )
        .map_err(ComponentInventoryRejection::new)?;
        let index_json = fs
            .load(&index_path)
            .await
            .map_err(ComponentInventoryRejection::new)?;
        let loaded_bytes = u64::try_from(index_json.len())
            .map_err(|_| ComponentInventoryRejection::new("extension index length overflowed"))?;
        let loaded_metadata = fs
            .metadata(&index_path)
            .await
            .map_err(ComponentInventoryRejection::new)?
            .ok_or_else(|| {
                ComponentInventoryRejection::new("extension index disappeared while loading")
            })?;
        validate_component_file_metadata(
            &metadata,
            &loaded_metadata,
            loaded_bytes,
            MAXIMUM_EXTENSION_INDEX_BYTES,
            "extension index",
            "installed",
        )
        .map_err(ComponentInventoryRejection::new)?;
        if loaded_bytes == 0 {
            return Err(ComponentInventoryRejection::new(
                "extension index cannot be empty",
            ));
        }
        let index: ExtensionIndex =
            serde_json::from_str(&index_json).map_err(ComponentInventoryRejection::new)?;
        Self::component_inventory_candidate_for_index(fs, extensions_dir, &index).await
    }

    async fn component_inventory_candidate_for_index(
        fs: Arc<dyn Fs>,
        extensions_dir: &Path,
        index: &ExtensionIndex,
    ) -> Result<ComponentInventoryCandidate, ComponentInventoryRejection> {
        let entries = index.extensions.values().cloned().collect::<Vec<_>>();
        Self::component_inventory_candidate_from_entries(
            fs,
            &extensions_dir.join("installed"),
            &entries,
        )
        .await
    }

    async fn component_inventory_candidate_from_entries(
        fs: Arc<dyn Fs>,
        installed_dir: &Path,
        entries: &[ExtensionIndexEntry],
    ) -> Result<ComponentInventoryCandidate, ComponentInventoryRejection> {
        let components = Self::load_installed_components(fs, installed_dir, entries)
            .await
            .map_err(ComponentInventoryRejection::new)?;
        let payload_bytes = components.iter().try_fold(0_u64, |total, component| {
            let manifest_bytes = u64::try_from(component.manifest_bytes().len()).ok()?;
            let component_bytes = u64::try_from(component.component_bytes().len()).ok()?;
            total
                .checked_add(manifest_bytes)?
                .checked_add(component_bytes)
        });
        let payload_bytes = payload_bytes.ok_or_else(|| {
            ComponentInventoryRejection::new("component inventory payload length overflowed")
        })?;
        let mut digest = Sha256::new();
        digest.update(b"zed.extension-host.component-inventory-candidate.v1\0");
        digest.update(
            u64::try_from(components.len())
                .map_err(ComponentInventoryRejection::new)?
                .to_le_bytes(),
        );
        for component in &components {
            for field in [
                component.extension_id().as_bytes(),
                component.extension_version().as_bytes(),
                component.manifest_bytes(),
                component.component_bytes(),
            ] {
                digest.update(
                    u64::try_from(field.len())
                        .map_err(ComponentInventoryRejection::new)?
                        .to_le_bytes(),
                );
                digest.update(field);
            }
        }
        let descriptors = components
            .iter()
            .map(|component| ComponentInventoryCandidateDescriptorIdentity {
                extension_id: component.extension_id.clone(),
                extension_version: component.extension_version.clone(),
                manifest_sha256: Arc::from(format!(
                    "{:x}",
                    Sha256::digest(component.manifest_bytes())
                )),
                component_sha256: Arc::from(format!(
                    "{:x}",
                    Sha256::digest(component.component_bytes())
                )),
            })
            .collect::<Vec<_>>();
        Ok(ComponentInventoryCandidate {
            components,
            payload_bytes,
            identity: ComponentInventoryCandidateIdentity {
                sha256: Arc::from(format!("{:x}", digest.finalize())),
                descriptors: descriptors.into(),
            },
        })
    }

    async fn load_installed_components(
        fs: Arc<dyn Fs>,
        installed_dir: &Path,
        entries: &[ExtensionIndexEntry],
    ) -> Result<Vec<InstalledComponent>> {
        if entries.len() > MAXIMUM_SIM_COMPONENT_COUNT {
            bail!(
                "component inventory contains {} entries; maximum is {MAXIMUM_SIM_COMPONENT_COUNT}",
                entries.len()
            );
        }
        let mut components = Vec::new();
        components
            .try_reserve(entries.len())
            .map_err(|_| anyhow!("component inventory allocation failed"))?;
        let mut payload_bytes = 0_u64;
        for entry in entries {
            let extension_id = entry.manifest.id.clone();
            let extension_dir = checked_extension_dir(installed_dir, &extension_id)?;
            let manifest_path = extension_dir.join(SIM_COMPONENT_MANIFEST_FILE);
            let component_path = extension_dir.join(SIM_COMPONENT_BINARY_FILE);
            let manifest_metadata = fs.metadata(&manifest_path).await.with_context(|| {
                format!("reading component manifest metadata for `{extension_id}`")
            })?;
            let component_metadata = fs.metadata(&component_path).await.with_context(|| {
                format!("reading component binary metadata for `{extension_id}`")
            })?;
            if manifest_metadata.is_some() || component_metadata.is_some() {
                let extension_metadata = fs
                    .metadata(&extension_dir)
                    .await
                    .with_context(|| {
                        format!("reading extension directory metadata for `{extension_id}`")
                    })?
                    .ok_or_else(|| {
                        anyhow!("extension directory for `{extension_id}` is missing")
                    })?;
                if extension_metadata.is_symlink || !extension_metadata.is_dir {
                    bail!(
                        "extension directory for `{extension_id}` is not a direct real directory"
                    );
                }
                let canonical_installed_dir = fs
                    .canonicalize(installed_dir)
                    .await
                    .context("resolving the installed extension root")?;
                let canonical_extension_dir =
                    fs.canonicalize(&extension_dir).await.with_context(|| {
                        format!("resolving extension directory for `{extension_id}`")
                    })?;
                if canonical_extension_dir.parent() != Some(canonical_installed_dir.as_path()) {
                    bail!("extension directory for `{extension_id}` escapes the installed root");
                }
            }
            let (manifest_metadata, component_metadata) = match (
                manifest_metadata,
                component_metadata,
            ) {
                (None, None) => {
                    for legacy_name in ["comfy-plugin.json", "comfy-plugin.wasm"] {
                        if fs
                            .metadata(&extension_dir.join(legacy_name))
                            .await
                            .with_context(|| {
                                format!("checking legacy component layout for `{extension_id}`")
                            })?
                            .is_some()
                        {
                            bail!(
                                "extension `{extension_id}` contains legacy {legacy_name}; rebuild its component from the owning source for Sim and install {SIM_COMPONENT_MANIFEST_FILE} with {SIM_COMPONENT_BINARY_FILE}, then approve the new component digest. Existing files have been preserved"
                            );
                        }
                    }
                    continue;
                }
                (Some(manifest), Some(component)) => {
                    validate_component_file_metadata(
                        &manifest,
                        &manifest,
                        manifest.len,
                        MAXIMUM_SIM_COMPONENT_MANIFEST_BYTES,
                        "component manifest",
                        &extension_id,
                    )?;
                    validate_component_file_metadata(
                        &component,
                        &component,
                        component.len,
                        MAXIMUM_SIM_COMPONENT_BINARY_BYTES,
                        "component binary",
                        &extension_id,
                    )?;
                    (manifest, component)
                }
                _ => {
                    bail!(
                        "extension `{extension_id}` must provide both {SIM_COMPONENT_MANIFEST_FILE} and {SIM_COMPONENT_BINARY_FILE}"
                    );
                }
            };
            let manifest_bytes = load_bounded_component_file(
                fs.clone(),
                &manifest_path,
                manifest_metadata,
                MAXIMUM_SIM_COMPONENT_MANIFEST_BYTES,
                "component manifest",
                &extension_id,
            )
            .await?;
            let component_bytes = load_bounded_component_file(
                fs.clone(),
                &component_path,
                component_metadata,
                MAXIMUM_SIM_COMPONENT_BINARY_BYTES,
                "component binary",
                &extension_id,
            )
            .await?;
            payload_bytes =
                payload_bytes
                    .checked_add(u64::try_from(manifest_bytes.len()).map_err(|_| {
                        anyhow!("component manifest for `{extension_id}` is oversized")
                    })?)
                    .and_then(|total| {
                        u64::try_from(component_bytes.len())
                            .ok()
                            .and_then(|length| total.checked_add(length))
                    })
                    .ok_or_else(|| anyhow!("component inventory payload length overflowed"))?;
            if payload_bytes > MAXIMUM_SIM_COMPONENT_INVENTORY_BYTES {
                bail!(
                    "component inventory contains {payload_bytes} payload bytes; maximum is {MAXIMUM_SIM_COMPONENT_INVENTORY_BYTES}"
                );
            }
            components.push(InstalledComponent::checked(
                extension_id,
                entry.manifest.version.clone(),
                manifest_bytes.into(),
                component_bytes.into(),
            )?);
        }
        components.sort_by(|left, right| left.extension_id().cmp(right.extension_id()));
        Ok(components)
    }

    fn rebuild_extension_index(&self, cx: &mut Context<Self>) -> Task<ExtensionIndex> {
        let fs = self.fs.clone();
        let work_dir = self.wasm_host.work_dir.clone();
        let extensions_dir = self.installed_dir.clone();
        let index_path = self.index_path.clone();
        let proxy = self.proxy.clone();
        cx.background_spawn(async move {
            let start_time = Instant::now();
            let mut index = ExtensionIndex::default();

            fs.create_dir(&work_dir).await.log_err();
            fs.create_dir(&extensions_dir).await.log_err();

            let extension_paths = fs.read_dir(&extensions_dir).await;
            if let Ok(mut extension_paths) = extension_paths {
                while let Some(extension_dir) = extension_paths.next().await {
                    let Ok(extension_dir) = extension_dir else {
                        continue;
                    };

                    if extension_dir
                        .file_name()
                        .is_some_and(|file_name| file_name == ".DS_Store")
                    {
                        continue;
                    }

                    Self::add_extension_to_index(
                        fs.clone(),
                        extension_dir,
                        &mut index,
                        proxy.clone(),
                    )
                    .await
                    .log_err();
                }
            }

            if let Ok(index_json) = serde_json::to_string_pretty(&index) {
                fs.save(&index_path, &index_json.as_str().into(), Default::default())
                    .await
                    .context("failed to save extension index")
                    .log_err();
            }

            log::info!("rebuilt extension index in {:?}", start_time.elapsed());
            index
        })
    }

    async fn add_extension_to_index(
        fs: Arc<dyn Fs>,
        extension_dir: PathBuf,
        index: &mut ExtensionIndex,
        proxy: Arc<ExtensionHostProxy>,
    ) -> Result<()> {
        let mut extension_manifest = ExtensionManifest::load(fs.clone(), &extension_dir).await?;
        let extension_id = extension_manifest.id.clone();

        // TODO: distinguish dev extensions more explicitly, by the absence
        // of a checksum file that we'll create when downloading normal extensions.
        let is_dev = fs
            .metadata(&extension_dir)
            .await?
            .with_context(|| format!("missing extension directory {extension_dir:?}"))?
            .is_symlink;

        let language_dir = extension_dir.join("languages");
        if let Ok(mut language_paths) = fs.read_dir(&language_dir).await {
            while let Some(language_path) = language_paths.next().await {
                let language_path = language_path
                    .with_context(|| format!("reading entries in language dir {language_dir:?}"))?;
                let Ok(relative_path) = language_path.strip_prefix(&extension_dir) else {
                    continue;
                };
                let Ok(Some(fs_metadata)) = fs.metadata(&language_path).await else {
                    continue;
                };
                if !fs_metadata.is_dir {
                    continue;
                }
                let language_config_path = language_path.join(LanguageConfig::FILE_NAME);
                let config = fs.load(&language_config_path).await.with_context(|| {
                    format!("loading language config from {language_config_path:?}")
                })?;
                let config = ::toml::from_str::<LanguageConfig>(&config)?;

                let relative_path = relative_path.to_rel_path_buf()?;
                if !extension_manifest.languages.contains(&relative_path) {
                    extension_manifest.languages.push(relative_path.clone());
                }

                index.languages.insert(
                    config.name.clone(),
                    ExtensionIndexLanguageEntry {
                        extension: extension_id.clone(),
                        path: relative_path.as_std_path().to_path_buf(),
                        matcher: config.matcher,
                        hidden: config.hidden,
                        grammar: config.grammar,
                    },
                );
            }
        }

        if let Ok(mut theme_paths) = fs.read_dir(&extension_dir.join("themes")).await {
            while let Some(theme_path) = theme_paths.next().await {
                let theme_path = theme_path?;
                let Ok(relative_path) = theme_path.strip_prefix(&extension_dir) else {
                    continue;
                };

                let Some(theme_families) = proxy
                    .list_theme_names(theme_path.clone(), fs.clone())
                    .await
                    .log_err()
                else {
                    continue;
                };

                let relative_path = relative_path.to_rel_path_buf()?;
                if !extension_manifest.themes.contains(&relative_path) {
                    extension_manifest.themes.push(relative_path.clone());
                }

                for theme_name in theme_families {
                    index.themes.insert(
                        theme_name.into(),
                        ExtensionIndexThemeEntry {
                            extension: extension_id.clone(),
                            path: relative_path.as_std_path().to_path_buf(),
                        },
                    );
                }
            }
        }

        if let Ok(mut icon_theme_paths) = fs.read_dir(&extension_dir.join("icon_themes")).await {
            while let Some(icon_theme_path) = icon_theme_paths.next().await {
                let icon_theme_path = icon_theme_path?;
                let Ok(relative_path) = icon_theme_path.strip_prefix(&extension_dir) else {
                    continue;
                };

                let Some(icon_theme_families) = proxy
                    .list_icon_theme_names(icon_theme_path.clone(), fs.clone())
                    .await
                    .log_err()
                else {
                    continue;
                };

                let relative_path = relative_path.to_rel_path_buf()?;
                if !extension_manifest.icon_themes.contains(&relative_path) {
                    extension_manifest.icon_themes.push(relative_path.clone());
                }

                for icon_theme_name in icon_theme_families {
                    index.icon_themes.insert(
                        icon_theme_name.into(),
                        ExtensionIndexIconThemeEntry {
                            extension: extension_id.clone(),
                            path: relative_path.as_std_path().to_path_buf(),
                        },
                    );
                }
            }
        }

        let extension_wasm_path = extension_dir.join("extension.wasm");
        if fs.is_file(&extension_wasm_path).await {
            extension_manifest
                .lib
                .kind
                .get_or_insert(ExtensionLibraryKind::Rust);
        }

        index.extensions.insert(
            extension_id.clone(),
            ExtensionIndexEntry {
                dev: is_dev,
                manifest: Arc::new(extension_manifest),
            },
        );

        Ok(())
    }

    fn prepare_remote_extension(
        &mut self,
        extension_id: Arc<str>,
        is_dev: bool,
        tmp_dir: PathBuf,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let src_dir = self.extensions_dir().join(extension_id.as_ref());
        let Some(loaded_extension) = self.extension_index.extensions.get(&extension_id).cloned()
        else {
            return Task::ready(Err(anyhow!("extension no longer installed")));
        };
        let fs = self.fs.clone();
        cx.background_spawn(async move {
            const EXTENSION_TOML: &str = "extension.toml";
            const EXTENSION_WASM: &str = "extension.wasm";
            const CONFIG_TOML: &str = LanguageConfig::FILE_NAME;

            if is_dev {
                let manifest_toml = toml::to_string(&loaded_extension.manifest)?;
                fs.save(
                    &tmp_dir.join(EXTENSION_TOML),
                    &Rope::from(manifest_toml),
                    language::LineEnding::Unix,
                )
                .await?;
            } else {
                fs.copy_file(
                    &src_dir.join(EXTENSION_TOML),
                    &tmp_dir.join(EXTENSION_TOML),
                    fs::CopyOptions::default(),
                )
                .await?
            }

            if fs.is_file(&src_dir.join(EXTENSION_WASM)).await {
                fs.copy_file(
                    &src_dir.join(EXTENSION_WASM),
                    &tmp_dir.join(EXTENSION_WASM),
                    fs::CopyOptions::default(),
                )
                .await?
            }

            for language_path in loaded_extension.manifest.languages.iter() {
                if fs
                    .is_file(&src_dir.join(language_path).join(CONFIG_TOML))
                    .await
                {
                    fs.create_dir(&tmp_dir.join(language_path)).await?;
                    fs.copy_file(
                        &src_dir.join(language_path).join(CONFIG_TOML),
                        &tmp_dir.join(language_path).join(CONFIG_TOML),
                        fs::CopyOptions::default(),
                    )
                    .await?;
                    copy_java_remote_runnables(
                        fs.as_ref(),
                        &loaded_extension.manifest,
                        &src_dir,
                        &tmp_dir,
                        &src_dir.join(language_path),
                        &tmp_dir.join(language_path),
                        cfg!(feature = "jvm-test-runnables"),
                    )
                    .await?;
                }
            }

            for (adapter_name, meta) in loaded_extension.manifest.debug_adapters.iter() {
                let schema_path = extension::build_debug_adapter_schema_path(adapter_name, meta)?;

                if fs.is_file(&src_dir.join(&schema_path)).await {
                    if let Some(parent) = schema_path.parent() {
                        fs.create_dir(&tmp_dir.join(parent)).await?
                    }
                    fs.copy_file(
                        &src_dir.join(&schema_path),
                        &tmp_dir.join(&schema_path),
                        fs::CopyOptions::default(),
                    )
                    .await?
                }
            }

            Ok(())
        })
    }

    async fn sync_extensions_to_remotes(
        this: &WeakEntity<Self>,
        client: WeakEntity<RemoteClient>,
        cx: &mut AsyncApp,
    ) -> Result<()> {
        let extensions = this.update(cx, |this, _cx| {
            this.extension_index
                .extensions_to_sync_to_remote()
                .into_entries()
                .map(|(id, entry)| proto::Extension {
                    id: id.to_string(),
                    version: entry.manifest.version.to_string(),
                    dev: entry.dev,
                })
                .collect()
        })?;

        let response = client
            .update(cx, |client, _cx| {
                client
                    .proto_client()
                    .request(proto::SyncExtensions { extensions })
            })?
            .await?;
        let path_style = client.read_with(cx, |client, _| client.path_style())?;

        for missing_extension in response.missing_extensions.into_iter() {
            let tmp_dir = tempfile::tempdir()?;
            this.update(cx, |this, cx| {
                this.prepare_remote_extension(
                    missing_extension.id.clone().into(),
                    missing_extension.dev,
                    tmp_dir.path().to_owned(),
                    cx,
                )
            })?
            .await?;
            let dest_dir = RemotePathBuf::new(
                path_style
                    .join(&response.tmp_dir, &missing_extension.id)
                    .with_context(|| {
                        format!(
                            "failed to construct destination path: {:?}, {:?}",
                            response.tmp_dir, missing_extension.id,
                        )
                    })?,
                path_style,
            );
            log::info!(
                "Uploading extension {} to {:?}",
                missing_extension.clone().id,
                dest_dir
            );

            client
                .update(cx, |client, cx| {
                    client.upload_directory(tmp_dir.path().to_owned(), dest_dir.clone(), cx)
                })?
                .await?;

            log::info!(
                "Finished uploading extension {}",
                missing_extension.clone().id
            );

            let result = client
                .update(cx, |client, _cx| {
                    client.proto_client().request(proto::InstallExtension {
                        tmp_dir: dest_dir.to_proto(),
                        extension: Some(missing_extension.clone()),
                    })
                })?
                .await;

            if let Err(e) = result {
                log::error!(
                    "Failed to install extension {}: {}",
                    missing_extension.id,
                    e
                );
            }
        }

        anyhow::Ok(())
    }

    pub async fn update_remote_clients(this: &WeakEntity<Self>, cx: &mut AsyncApp) -> Result<()> {
        let clients = this.update(cx, |this, cx| {
            this.remote_clients.retain(|v| v.upgrade().is_some());
            connected_remote_clients(&this.remote_clients, cx)
        })?;

        for client in clients {
            Self::sync_extensions_to_remotes(this, client, cx)
                .await
                .log_err();
        }

        anyhow::Ok(())
    }

    pub fn register_remote_client(
        &mut self,
        client: Entity<RemoteClient>,
        _cx: &mut Context<Self>,
    ) {
        self.remote_clients.push(client.downgrade());
        self.ssh_registered_tx.unbounded_send(()).ok();
    }
}

fn connected_remote_clients(
    clients: &[WeakEntity<RemoteClient>],
    cx: &App,
) -> Vec<WeakEntity<RemoteClient>> {
    clients
        .iter()
        .filter(|client| {
            client
                .upgrade()
                .is_some_and(|client| client.read(cx).connection().is_some())
        })
        .cloned()
        .collect()
}

fn load_plugin_queries(root_path: &Path) -> LanguageQueries {
    let mut result = LanguageQueries::default();
    if let Some(entries) = std::fs::read_dir(root_path).log_err() {
        for entry in entries {
            let Some(entry) = entry.log_err() else {
                continue;
            };
            let path = entry.path();
            if let Some(remainder) = path.strip_prefix(root_path).ok().and_then(|p| p.to_str()) {
                if !remainder.ends_with(".scm") {
                    continue;
                }
                for (name, query) in QUERY_FILENAME_PREFIXES {
                    if remainder.starts_with(name) {
                        if let Some(contents) = std::fs::read_to_string(&path).log_err() {
                            match query(&mut result) {
                                None => *query(&mut result) = Some(contents.into()),
                                Some(r) => r.to_mut().push_str(contents.as_ref()),
                            }
                        }
                        break;
                    }
                }
            }
        }
    }
    result
}
