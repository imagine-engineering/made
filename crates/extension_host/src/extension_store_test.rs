use crate::{
    ComponentLifecycleAdapter, Event, ExtensionIndex, ExtensionIndexEntry,
    ExtensionIndexLanguageEntry, ExtensionIndexThemeEntry, ExtensionManifest, ExtensionStore,
    GrammarManifestEntry, InstalledComponent, RELOAD_DEBOUNCE_DURATION,
    RegisteredComponentAdapters, SIM_COMPONENT_BINARY_FILE, SIM_COMPONENT_MANIFEST_FILE,
    SchemaVersion, copy_java_remote_runnables, register_component_lifecycle_adapter,
    validate_component_file_metadata,
};
use async_compression::futures::bufread::GzipEncoder;
use collections::{BTreeMap, HashSet};
use extension::ExtensionHostProxy;
use fs::{FakeFs, Fs, RealFs};
use futures::{AsyncReadExt, FutureExt, StreamExt, future::BoxFuture, io::BufReader};
use gpui::{AppContext as _, BackgroundExecutor, TaskExt, TestAppContext};
use http_client::{FakeHttpClient, Response};
use language::{BinaryStatus, LanguageMatcher, LanguageName, LanguageRegistry};
use language_extension::LspAccess;
use lsp::LanguageServerName;
use node_runtime::NodeRuntime;
use parking_lot::Mutex;
use project::{DEFAULT_COMPLETION_CONTEXT, Project};
use release_channel::AppVersion;
use remote::RemoteClient;
use reqwest_client::ReqwestClient;
use serde_json::json;
use settings::{Settings, SettingsStore};
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    sync::Arc,
};
use theme::ThemeRegistry;
use util::{rel_path::rel_path_buf, test::TempTree};

#[cfg(test)]
#[ctor::ctor(unsafe)]
fn init_logger() {
    zlog::init_test();
}

fn remote_sync_entry(id: &str, manifest_body: &str) -> ExtensionIndexEntry {
    remote_sync_entry_with_version(id, "1.0.0", manifest_body)
}

fn remote_sync_entry_with_version(
    id: &str,
    version: &str,
    manifest_body: &str,
) -> ExtensionIndexEntry {
    let id = toml::Value::String(id.to_owned()).to_string();
    let version = toml::Value::String(version.to_owned()).to_string();
    let manifest = format!(
        r#"
        id = {id}
        name = {id}
        version = {version}
        schema_version = 0

        {manifest_body}
        "#
    );

    ExtensionIndexEntry {
        manifest: Arc::new(toml::from_str(&manifest).unwrap()),
        dev: false,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RecordedComponent {
    extension_id: String,
    extension_version: String,
    manifest_bytes: Vec<u8>,
    component_bytes: Vec<u8>,
}

#[derive(Clone)]
struct RecordingComponentAdapter {
    snapshots: Arc<Mutex<Vec<Vec<RecordedComponent>>>>,
    rejected_manifest: Option<Vec<u8>>,
}

impl RecordingComponentAdapter {
    fn new() -> Self {
        Self {
            snapshots: Arc::new(Mutex::new(Vec::new())),
            rejected_manifest: None,
        }
    }

    fn rejecting(rejected_manifest: Vec<u8>) -> Self {
        Self {
            snapshots: Arc::new(Mutex::new(Vec::new())),
            rejected_manifest: Some(rejected_manifest),
        }
    }

    fn snapshots(&self) -> Vec<Vec<RecordedComponent>> {
        self.snapshots.lock().clone()
    }
}

impl ComponentLifecycleAdapter for RecordingComponentAdapter {
    fn adapter_id(&self) -> &'static str {
        "test.component-lifecycle"
    }

    fn synchronize(
        &self,
        components: Vec<InstalledComponent>,
    ) -> BoxFuture<'static, Result<(), String>> {
        let snapshots = self.snapshots.clone();
        let rejected_manifest = self.rejected_manifest.clone();
        Box::pin(async move {
            if let Some(rejected_manifest) = rejected_manifest
                && components
                    .iter()
                    .any(|component| component.manifest_bytes() == rejected_manifest)
            {
                return Err("component verification failed".to_owned());
            }
            snapshots.lock().push(
                components
                    .into_iter()
                    .map(|component| RecordedComponent {
                        extension_id: component.extension_id().to_owned(),
                        extension_version: component.extension_version().to_owned(),
                        manifest_bytes: component.manifest_bytes().to_vec(),
                        component_bytes: component.component_bytes().to_vec(),
                    })
                    .collect(),
            );
            Ok(())
        })
    }
}

async fn assert_component_candidate_rejected_without_adapter_call(
    fs: Arc<dyn Fs>,
    installed_dir: &Path,
    entries: &[ExtensionIndexEntry],
    adapter: Arc<RecordingComponentAdapter>,
    diagnostic: &str,
) {
    let accepted_snapshots = adapter.snapshots();
    let adapters: Vec<Arc<dyn ComponentLifecycleAdapter>> = vec![adapter.clone()];
    let errors =
        ExtensionStore::synchronize_component_adapters(fs, installed_dir, entries, &adapters).await;
    assert!(
        errors
            .get(adapter.adapter_id())
            .is_some_and(|error| error.contains(diagnostic)),
        "missing canonical {diagnostic:?} rejection in {errors:?}"
    );
    assert_eq!(
        adapter.snapshots(),
        accepted_snapshots,
        "rejected candidate {diagnostic:?} must not invoke the adapter"
    );
}

fn remote_sync_language_entry(extension: &str, path: &str) -> ExtensionIndexLanguageEntry {
    ExtensionIndexLanguageEntry {
        extension: extension.into(),
        path: path.into(),
        matcher: LanguageMatcher::default().into(),
        hidden: false,
        grammar: None,
    }
}

fn remote_sync_extension_ids(index: &ExtensionIndex) -> Vec<String> {
    let mut extensions = index
        .extensions_to_sync_to_remote()
        .into_entries()
        .map(|(id, _)| id.to_string())
        .collect::<Vec<_>>();

    extensions.sort();

    extensions
}

#[gpui::test]
async fn legacy_component_layout_requires_a_source_rebuild_and_preserves_files(
    cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        "/component-extensions/installed/legacy",
        json!({"comfy-plugin.json": "prior-manifest", "comfy-plugin.wasm": "prior-component"}),
    )
    .await;
    let error = ExtensionStore::load_installed_components(
        fs.clone(),
        Path::new("/component-extensions/installed"),
        &[remote_sync_entry("legacy", "")],
    )
    .await
    .err()
    .expect("old component layout must not execute under the new ABI");
    assert!(
        error
            .to_string()
            .contains("rebuild its component from the owning source for Sim")
    );
    assert!(
        error
            .to_string()
            .contains("approve the new component digest")
    );
    assert_eq!(
        fs.load(Path::new(
            "/component-extensions/installed/legacy/comfy-plugin.json"
        ))
        .await
        .expect("preserved manifest"),
        "prior-manifest"
    );
    assert_eq!(
        fs.load(Path::new(
            "/component-extensions/installed/legacy/comfy-plugin.wasm"
        ))
        .await
        .expect("preserved component"),
        "prior-component"
    );
}

#[gpui::test]
async fn component_inventory_is_fixed_bounded_and_symlink_free(cx: &mut TestAppContext) {
    assert_eq!(SIM_COMPONENT_MANIFEST_FILE, "sim-plugin.json");
    assert_eq!(SIM_COMPONENT_BINARY_FILE, "sim-plugin.wasm");
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        "/component-extensions/installed/alpha",
        json!({
            "sim-plugin.json": r#"{"identifier":"alpha"}"#,
            "sim-plugin.wasm": "component-bytes",
        }),
    )
    .await;
    let entries = [remote_sync_entry("alpha", "")];
    let components = ExtensionStore::load_installed_components(
        fs.clone(),
        Path::new("/component-extensions/installed"),
        &entries,
    )
    .await
    .expect("load fixed component pair");
    assert_eq!(components.len(), 1);
    assert_eq!(components[0].extension_id(), "alpha");
    assert_eq!(components[0].extension_version(), "1.0.0");
    assert_eq!(components[0].manifest_bytes(), br#"{"identifier":"alpha"}"#);
    assert_eq!(components[0].component_bytes(), b"component-bytes");

    fs.insert_tree(
        "/component-extensions/installed/missing-pair",
        json!({
            "sim-plugin.json": r#"{"identifier":"missing-pair"}"#,
        }),
    )
    .await;
    let missing_pair = [remote_sync_entry("missing-pair", "")];
    let error = ExtensionStore::load_installed_components(
        fs.clone(),
        Path::new("/component-extensions/installed"),
        &missing_pair,
    )
    .await
    .err()
    .expect("component inventory must reject an incomplete fixed pair");
    assert!(error.to_string().contains("must provide both"));

    fs.insert_tree(
        "/component-extensions/installed/symlinked",
        json!({
            "sim-plugin.json": r#"{"identifier":"symlinked"}"#,
        }),
    )
    .await;
    fs.insert_tree(
        "/component-extensions/source",
        json!({"component.wasm": "outside-component"}),
    )
    .await;
    fs.insert_symlink(
        "/component-extensions/installed/symlinked/sim-plugin.wasm",
        PathBuf::from("/component-extensions/source/component.wasm"),
    )
    .await;
    let symlinked = [remote_sync_entry("symlinked", "")];
    let error = ExtensionStore::load_installed_components(
        fs.clone(),
        Path::new("/component-extensions/installed"),
        &symlinked,
    )
    .await
    .err()
    .expect("component inventory must reject symlinked component bytes");
    assert!(error.to_string().contains("is invalid"));

    fs.insert_tree(
        "/component-extensions/outside/symlink-parent",
        json!({
            "sim-plugin.json": r#"{"identifier":"symlink-parent"}"#,
            "sim-plugin.wasm": "outside-component",
        }),
    )
    .await;
    fs.insert_symlink(
        "/component-extensions/installed/symlink-parent",
        PathBuf::from("/component-extensions/outside/symlink-parent"),
    )
    .await;
    let symlink_parent = [remote_sync_entry("symlink-parent", "")];
    let error = ExtensionStore::load_installed_components(
        fs.clone(),
        Path::new("/component-extensions/installed"),
        &symlink_parent,
    )
    .await
    .err()
    .expect("component inventory must reject a symlinked extension directory");
    assert!(error.to_string().contains("not a direct real directory"));

    for extension_id in ["../escape", "nested/escape", r"nested\escape"] {
        let traversal = [remote_sync_entry(extension_id, "")];
        let error = ExtensionStore::load_installed_components(
            fs.clone(),
            Path::new("/component-extensions/installed"),
            &traversal,
        )
        .await
        .err()
        .expect("component inventory must reject a non-component extension identifier");
        assert!(
            error.to_string().contains("one normal path component"),
            "unexpected traversal error for {extension_id}: {error}"
        );
    }

    let oversized_manifest_dir = Path::new("/component-extensions/installed/oversized-manifest");
    fs.create_dir(oversized_manifest_dir)
        .await
        .expect("create oversized manifest extension directory");
    fs.insert_file(
        oversized_manifest_dir.join("sim-plugin.json"),
        vec![
            b'x';
            usize::try_from(crate::MAXIMUM_SIM_COMPONENT_MANIFEST_BYTES + 1)
                .expect("manifest limit fits usize")
        ],
    )
    .await;
    fs.insert_file(
        oversized_manifest_dir.join("sim-plugin.wasm"),
        b"component".to_vec(),
    )
    .await;
    let oversized_manifest = [remote_sync_entry("oversized-manifest", "")];
    let error = ExtensionStore::load_installed_components(
        fs.clone(),
        Path::new("/component-extensions/installed"),
        &oversized_manifest,
    )
    .await
    .err()
    .expect("component inventory must reject an oversized manifest");
    assert!(error.to_string().contains("is oversized"));

    let metadata = fs
        .metadata(Path::new(
            "/component-extensions/installed/alpha/sim-plugin.wasm",
        ))
        .await
        .expect("read component metadata")
        .expect("component binary exists");
    let mut changed_metadata = metadata;
    changed_metadata.len += 1;
    let error = validate_component_file_metadata(
        &metadata,
        &changed_metadata,
        changed_metadata.len,
        crate::MAXIMUM_SIM_COMPONENT_BINARY_BYTES,
        "component binary",
        "alpha",
    )
    .expect_err("component inventory must reject post-read identity changes");
    assert!(
        error
            .to_string()
            .contains("changed while it was being loaded")
    );

    let error = validate_component_file_metadata(
        &metadata,
        &metadata,
        crate::MAXIMUM_SIM_COMPONENT_BINARY_BYTES + 1,
        crate::MAXIMUM_SIM_COMPONENT_BINARY_BYTES,
        "component binary",
        "alpha",
    )
    .expect_err("component inventory must enforce bounds after reading bytes");
    assert!(error.to_string().contains("is oversized"));
}

#[test]
fn installed_component_constructor_owns_identity_and_payload_bounds() {
    let empty_bytes: Arc<[u8]> = Vec::new().into();
    let error = InstalledComponent::checked(
        "../escape".into(),
        "1.0.0".into(),
        empty_bytes.clone(),
        empty_bytes.clone(),
    )
    .err()
    .expect("installed component must reject a non-component identifier");
    assert!(error.to_string().contains("one normal path component"));

    for invalid_identifier in ["", ".", "..", "a/b", "a\\b", "a:b", "NUL", "trailing."] {
        let error = InstalledComponent::checked(
            invalid_identifier.into(),
            "1.0.0".into(),
            empty_bytes.clone(),
            empty_bytes.clone(),
        )
        .err()
        .expect("non-portable component identities must be rejected");
        assert!(error.to_string().contains("one normal path component"));
    }

    InstalledComponent::checked(
        "插件".into(),
        "1.0.0".into(),
        empty_bytes.clone(),
        empty_bytes,
    )
    .expect("one normal non-ASCII path component is a valid canonical identity");

    crate::validate_component_payload_length(
        crate::MAXIMUM_SIM_COMPONENT_MANIFEST_BYTES,
        crate::MAXIMUM_SIM_COMPONENT_MANIFEST_BYTES,
        "component manifest",
        "alpha",
    )
    .expect("manifest payload at the canonical bound is valid");
    let error = crate::validate_component_payload_length(
        crate::MAXIMUM_SIM_COMPONENT_MANIFEST_BYTES + 1,
        crate::MAXIMUM_SIM_COMPONENT_MANIFEST_BYTES,
        "component manifest",
        "alpha",
    )
    .expect_err("manifest payload over the canonical bound must fail");
    assert!(error.to_string().contains("is oversized"));

    let error = crate::validate_component_payload_length(
        crate::MAXIMUM_SIM_COMPONENT_BINARY_BYTES + 1,
        crate::MAXIMUM_SIM_COMPONENT_BINARY_BYTES,
        "component binary",
        "alpha",
    )
    .expect_err("component payload over the canonical bound must fail");
    assert!(error.to_string().contains("is oversized"));
}

#[test]
fn extension_lifecycle_mutations_share_one_checked_destination_mapper() {
    let installed = Path::new("/extensions/installed");
    assert_eq!(
        crate::checked_extension_dir(installed, "theme-pack")
            .expect("normal extension identifier must map below the installed root"),
        installed.join("theme-pack")
    );
    for extension_id in [
        "",
        ".",
        "..",
        "../escape",
        "nested/escape",
        r"nested\escape",
        "CON",
        "aux.txt",
        "trailing.",
    ] {
        let error = crate::checked_extension_dir(installed, extension_id)
            .expect_err("unsafe extension identifier must fail before a lifecycle mutation");
        assert!(
            error.to_string().contains("one normal path component"),
            "unexpected destination error for {extension_id:?}: {error}"
        );
    }
}

#[gpui::test]
async fn registered_component_adapter_tracks_lifecycle_and_restart(cx: &mut TestAppContext) {
    let fs = FakeFs::new(cx.executor());
    let installed_dir = Path::new("/component-lifecycle/installed");
    fs.insert_tree(
        installed_dir.join("alpha"),
        json!({
            "sim-plugin.json": "manifest-v1",
            "sim-plugin.wasm": "component-v1",
        }),
    )
    .await;

    let adapter = Arc::new(RecordingComponentAdapter::new());
    cx.update(|cx| {
        register_component_lifecycle_adapter(adapter.clone(), cx)
            .expect("register component lifecycle adapter")
    });
    let adapters = cx.update(|cx| cx.global::<RegisteredComponentAdapters>().0.clone());

    let mut entries = vec![remote_sync_entry_with_version("alpha", "1.0.0", "")];
    let errors = ExtensionStore::synchronize_component_adapters(
        fs.clone(),
        installed_dir,
        &entries,
        &adapters,
    )
    .await;
    assert!(errors.is_empty());
    assert_eq!(
        adapter.snapshots(),
        vec![vec![RecordedComponent {
            extension_id: "alpha".to_owned(),
            extension_version: "1.0.0".to_owned(),
            manifest_bytes: b"manifest-v1".to_vec(),
            component_bytes: b"component-v1".to_vec(),
        }]]
    );

    fs.insert_file(
        installed_dir.join("alpha/sim-plugin.json"),
        b"manifest-v2".to_vec(),
    )
    .await;
    fs.insert_file(
        installed_dir.join("alpha/sim-plugin.wasm"),
        b"component-v2".to_vec(),
    )
    .await;
    entries = vec![remote_sync_entry_with_version("alpha", "2.0.0", "")];
    let errors = ExtensionStore::synchronize_component_adapters(
        fs.clone(),
        installed_dir,
        &entries,
        &adapters,
    )
    .await;
    assert!(errors.is_empty());
    assert_eq!(
        adapter.snapshots().last(),
        Some(&vec![RecordedComponent {
            extension_id: "alpha".to_owned(),
            extension_version: "2.0.0".to_owned(),
            manifest_bytes: b"manifest-v2".to_vec(),
            component_bytes: b"component-v2".to_vec(),
        }])
    );

    let restarted_adapter = Arc::new(RecordingComponentAdapter::new());
    let restarted_adapters: Vec<Arc<dyn ComponentLifecycleAdapter>> =
        vec![restarted_adapter.clone()];
    let errors = ExtensionStore::synchronize_component_adapters(
        fs.clone(),
        installed_dir,
        &entries,
        &restarted_adapters,
    )
    .await;
    assert!(errors.is_empty());
    assert_eq!(
        restarted_adapter.snapshots(),
        vec![vec![RecordedComponent {
            extension_id: "alpha".to_owned(),
            extension_version: "2.0.0".to_owned(),
            manifest_bytes: b"manifest-v2".to_vec(),
            component_bytes: b"component-v2".to_vec(),
        }]]
    );

    let errors =
        ExtensionStore::synchronize_component_adapters(fs, installed_dir, &[], &adapters).await;
    assert!(errors.is_empty());
    assert_eq!(adapter.snapshots().last(), Some(&Vec::new()));
}

#[gpui::test]
async fn component_adapter_failures_converge_without_partial_verification(cx: &mut TestAppContext) {
    let fs = FakeFs::new(cx.executor());
    let installed_dir = Path::new("/component-failures/installed");
    fs.insert_tree(
        installed_dir.join("alpha"),
        json!({
            "sim-plugin.json": "accepted-manifest",
            "sim-plugin.wasm": "component",
        }),
    )
    .await;
    let entries = vec![remote_sync_entry("alpha", "")];
    let adapter = Arc::new(RecordingComponentAdapter::new());
    let adapters: Vec<Arc<dyn ComponentLifecycleAdapter>> = vec![adapter.clone()];

    let errors = ExtensionStore::synchronize_component_adapters(
        fs.clone(),
        installed_dir,
        &entries,
        &adapters,
    )
    .await;
    assert!(errors.is_empty());

    let missing_pair = vec![remote_sync_entry("missing-pair", "")];
    fs.insert_tree(
        installed_dir.join("missing-pair"),
        json!({"sim-plugin.json": "manifest"}),
    )
    .await;
    let errors = ExtensionStore::synchronize_component_adapters(
        fs.clone(),
        installed_dir,
        &missing_pair,
        &adapters,
    )
    .await;
    assert!(
        errors
            .get(adapter.adapter_id())
            .is_some_and(|error| error.contains("must provide both"))
    );
    assert_eq!(
        adapter.snapshots().last().map(Vec::len),
        Some(1),
        "rejected inventory must preserve the last accepted adapter state"
    );

    let errors = ExtensionStore::synchronize_component_adapters(
        fs.clone(),
        installed_dir,
        &entries,
        &adapters,
    )
    .await;
    assert!(errors.is_empty());
    assert_eq!(adapter.snapshots().last().map(Vec::len), Some(1));

    let rejecting_adapter = Arc::new(RecordingComponentAdapter::rejecting(
        b"rejected-manifest".to_vec(),
    ));
    let rejecting_adapters: Vec<Arc<dyn ComponentLifecycleAdapter>> =
        vec![rejecting_adapter.clone()];
    let errors = ExtensionStore::synchronize_component_adapters(
        fs.clone(),
        installed_dir,
        &entries,
        &rejecting_adapters,
    )
    .await;
    assert!(errors.is_empty());
    let accepted_snapshots = rejecting_adapter.snapshots();

    fs.insert_file(
        installed_dir.join("alpha/sim-plugin.json"),
        b"rejected-manifest".to_vec(),
    )
    .await;
    let errors = ExtensionStore::synchronize_component_adapters(
        fs.clone(),
        installed_dir,
        &entries,
        &rejecting_adapters,
    )
    .await;
    assert_eq!(
        errors.get(rejecting_adapter.adapter_id()),
        Some(&"component verification failed".to_owned())
    );
    assert_eq!(
        rejecting_adapter.snapshots(),
        accepted_snapshots,
        "verification failures must preserve the adapter's prior atomic state"
    );

    let errors =
        ExtensionStore::synchronize_component_adapters(fs, installed_dir, &[], &rejecting_adapters)
            .await;
    assert!(errors.is_empty());
    assert_eq!(rejecting_adapter.snapshots().last(), Some(&Vec::new()));
}

#[gpui::test]
async fn component_inventory_candidate_rejection_never_invokes_adapters(cx: &mut TestAppContext) {
    let fs = FakeFs::new(cx.executor());
    let installed_dir = Path::new("/component-candidate/installed");
    fs.insert_tree(
        installed_dir.join("alpha"),
        json!({
            "sim-plugin.json": "manifest-v1",
            "sim-plugin.wasm": "component-v1",
        }),
    )
    .await;
    let adapter = Arc::new(RecordingComponentAdapter::new());
    let adapters: Vec<Arc<dyn ComponentLifecycleAdapter>> = vec![adapter.clone()];
    let accepted = [remote_sync_entry_with_version("alpha", "1.0.0", "")];

    let errors = ExtensionStore::synchronize_component_adapters(
        fs.clone(),
        installed_dir,
        &accepted,
        &adapters,
    )
    .await;
    assert!(errors.is_empty());
    let accepted_snapshots = adapter.snapshots();
    assert_eq!(accepted_snapshots.len(), 1);

    fs.remove_file(
        installed_dir.join("alpha/sim-plugin.wasm").as_path(),
        Default::default(),
    )
    .await
    .expect("remove component half");
    assert_component_candidate_rejected_without_adapter_call(
        fs.clone(),
        installed_dir,
        &accepted,
        adapter.clone(),
        "must provide both",
    )
    .await;

    let traversal = [remote_sync_entry("../escape", "")];
    assert_component_candidate_rejected_without_adapter_call(
        fs.clone(),
        installed_dir,
        &traversal,
        adapter.clone(),
        "one normal path component",
    )
    .await;

    fs.insert_tree(
        "/component-candidate/outside/symlinked",
        json!({
            "sim-plugin.json": "manifest",
            "sim-plugin.wasm": "component",
        }),
    )
    .await;
    fs.insert_symlink(
        "/component-candidate/installed/symlinked",
        PathBuf::from("/component-candidate/outside/symlinked"),
    )
    .await;
    let symlinked = [remote_sync_entry("symlinked", "")];
    assert_component_candidate_rejected_without_adapter_call(
        fs.clone(),
        installed_dir,
        &symlinked,
        adapter.clone(),
        "not a direct real directory",
    )
    .await;

    let excessive_entries = (0..=crate::MAXIMUM_SIM_COMPONENT_COUNT)
        .map(|_| remote_sync_entry("alpha", ""))
        .collect::<Vec<_>>();
    assert_component_candidate_rejected_without_adapter_call(
        fs.clone(),
        installed_dir,
        &excessive_entries,
        adapter.clone(),
        "maximum",
    )
    .await;
    assert_eq!(adapter.snapshots(), accepted_snapshots);

    let valid_empty =
        ExtensionStore::component_inventory_candidate_from_entries(fs.clone(), installed_dir, &[])
            .await
            .expect("valid empty candidate");
    assert!(valid_empty.components().is_empty());
    assert_eq!(valid_empty.payload_bytes(), 0);
    let errors =
        ExtensionStore::synchronize_component_adapters(fs, installed_dir, &[], &adapters).await;
    assert!(errors.is_empty());
    assert_eq!(adapter.snapshots().last(), Some(&Vec::new()));
}

#[gpui::test]
async fn component_inventory_candidate_is_canonical_fresh_and_digest_stable(
    cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(cx.executor());
    let extensions_dir = Path::new("/canonical-component-candidate");
    fs.insert_tree(
        extensions_dir.join("installed/beta"),
        json!({
            "sim-plugin.json": "beta-manifest",
            "sim-plugin.wasm": "beta-component",
        }),
    )
    .await;
    fs.insert_tree(
        extensions_dir.join("installed/alpha"),
        json!({
            "sim-plugin.json": "alpha-manifest",
            "sim-plugin.wasm": "alpha-component",
        }),
    )
    .await;
    let mut index = ExtensionIndex::default();
    index.extensions.insert(
        Arc::from("beta"),
        remote_sync_entry_with_version("beta", "2.0.0", ""),
    );
    index.extensions.insert(
        Arc::from("alpha"),
        remote_sync_entry_with_version("alpha", "1.0.0", ""),
    );
    fs.insert_file(
        extensions_dir.join("index.json"),
        serde_json::to_vec(&index).expect("encode extension index"),
    )
    .await;

    let desktop =
        ExtensionStore::canonical_component_inventory_candidate(fs.clone(), extensions_dir)
            .await
            .expect("desktop canonical candidate");
    let headless =
        ExtensionStore::canonical_component_inventory_candidate(fs.clone(), extensions_dir)
            .await
            .expect("headless canonical candidate");
    assert_eq!(desktop, headless);
    assert_eq!(
        desktop
            .components()
            .iter()
            .map(InstalledComponent::extension_id)
            .collect::<Vec<_>>(),
        ["alpha", "beta"]
    );
    assert_eq!(desktop.identity_sha256().len(), 64);
    assert_eq!(
        desktop.identity_sha256(),
        "30e61cda7db99496dd3f5e68099583b3008629259de236c768df6b60ab83de38"
    );
    assert!(
        desktop
            .identity_sha256()
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    );

    fs.insert_file(
        extensions_dir.join("installed/alpha/extension-update"),
        b"newer-than-index".to_vec(),
    )
    .await;
    fs.touch_path(extensions_dir.join("installed")).await;
    let stale = ExtensionStore::canonical_component_inventory_candidate(fs, extensions_dir)
        .await
        .expect_err("stale index must not mint a candidate");
    assert!(stale.diagnostic().contains("stale"));
}

#[gpui::test]
async fn component_inventory_candidate_exact_replay_is_adapter_idempotent(cx: &mut TestAppContext) {
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    let extensions_dir = PathBuf::from("/component-replay");
    fs.insert_tree(
        extensions_dir.join("installed/alpha"),
        json!({
            "extension.json": r#"{
                "id": "alpha",
                "name": "Alpha",
                "version": "1.0.0"
            }"#,
            "sim-plugin.json": "manifest",
            "sim-plugin.wasm": "component",
        }),
    )
    .await;

    let proxy = Arc::new(ExtensionHostProxy::new());
    let theme_registry = Arc::new(ThemeRegistry::new(Box::new(())));
    theme_extension::init(proxy.clone(), theme_registry, cx.executor());
    let language_registry = Arc::new(LanguageRegistry::test(cx.executor()));
    language_extension::init(LspAccess::Noop, proxy.clone(), language_registry);
    let http_client = FakeHttpClient::with_200_response();
    let adapter = Arc::new(RecordingComponentAdapter::new());
    let store = cx.new(|cx| {
        ExtensionStore::new(
            extensions_dir,
            None,
            proxy,
            fs,
            http_client.clone(),
            http_client,
            None,
            NodeRuntime::unavailable(),
            vec![adapter.clone()],
            cx,
        )
    });

    cx.executor().advance_clock(RELOAD_DEBOUNCE_DURATION);
    cx.executor().run_until_parked();
    let accepted_nonempty = adapter.snapshots();
    assert_eq!(accepted_nonempty.len(), 1);
    for expected_calls in [1, 1] {
        let reload = store.update(cx, |store, cx| store.reload(None, cx));
        cx.executor().advance_clock(RELOAD_DEBOUNCE_DURATION);
        reload.await.expect("reload canonical component inventory");
        assert_eq!(
            adapter.snapshots().len(),
            expected_calls,
            "an exact accepted replay must not advance the adapter generation"
        );
    }
    let rejected_empty = store.update(cx, |store, cx| {
        store.extensions_updated(ExtensionIndex::default(), false, cx)
    });
    rejected_empty.await;
    assert_eq!(
        adapter.snapshots().len(),
        1,
        "failure-synthesized empty inventory must not invoke the adapter"
    );
    store.read_with(cx, |store, _| {
        assert!(
            store
                .component_adapter_errors()
                .get(adapter.adapter_id())
                .is_some_and(|error| error.contains("unavailable or stale"))
        );
    });

    let accepted_empty = store.update(cx, |store, cx| {
        store.extensions_updated(ExtensionIndex::default(), true, cx)
    });
    accepted_empty.await;
    let mut expected_snapshots = accepted_nonempty;
    expected_snapshots.push(Vec::new());
    assert_eq!(adapter.snapshots(), expected_snapshots);
    store.read_with(cx, |store, _| {
        assert!(store.accepted_component_candidate.is_some());
        assert!(store.component_adapter_errors().is_empty());
    });
}

#[gpui::test]
async fn component_adapter_failure_is_returned_by_reload(cx: &mut TestAppContext) {
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    let extensions_dir = PathBuf::from("/component-reload-result");
    fs.insert_tree(
        extensions_dir.join("installed/alpha"),
        json!({
            "extension.json": r#"{
                "id": "alpha",
                "name": "Alpha",
                "version": "1.0.0"
            }"#,
            "sim-plugin.json": "rejected-manifest",
            "sim-plugin.wasm": "component",
        }),
    )
    .await;

    let proxy = Arc::new(ExtensionHostProxy::new());
    let theme_registry = Arc::new(ThemeRegistry::new(Box::new(())));
    theme_extension::init(proxy.clone(), theme_registry, cx.executor());
    let language_registry = Arc::new(LanguageRegistry::test(cx.executor()));
    language_extension::init(LspAccess::Noop, proxy.clone(), language_registry);
    let http_client = FakeHttpClient::with_200_response();
    let adapter = Arc::new(RecordingComponentAdapter::rejecting(
        b"rejected-manifest".to_vec(),
    ));
    let store = cx.new(|cx| {
        ExtensionStore::new(
            extensions_dir,
            None,
            proxy,
            fs,
            http_client.clone(),
            http_client,
            None,
            NodeRuntime::unavailable(),
            vec![adapter],
            cx,
        )
    });

    cx.executor().advance_clock(RELOAD_DEBOUNCE_DURATION);
    cx.executor().run_until_parked();
    let reload = store.update(cx, |store, cx| store.reload(None, cx));
    cx.executor().advance_clock(RELOAD_DEBOUNCE_DURATION);
    let error = reload
        .await
        .expect_err("adapter verification failure must fail extension reload");
    assert!(error.to_string().contains("component verification failed"));
    store.read_with(cx, |store, _| {
        assert_eq!(
            store
                .component_adapter_errors()
                .get("test.component-lifecycle"),
            Some(&"component verification failed".to_owned())
        );
    });
}

#[test]
fn remote_sync_includes_language_dependencies() {
    let index = ExtensionIndex {
        extensions: [
            (
                "bar-language".into(),
                remote_sync_entry("bar-language", r#"languages = ["languages/bar"]"#),
            ),
            (
                "foo-lsp".into(),
                remote_sync_entry(
                    "foo-lsp",
                    r#"
                    [language_servers.foo]
                    language = "Foo"
                    "#,
                ),
            ),
            (
                "foo-language".into(),
                remote_sync_entry("foo-language", r#"languages = ["languages/foo"]"#),
            ),
        ]
        .into_iter()
        .collect(),
        languages: [
            (
                "Bar".into(),
                remote_sync_language_entry("bar-language", "languages/bar"),
            ),
            (
                "Foo".into(),
                remote_sync_language_entry("foo-language", "languages/foo"),
            ),
        ]
        .into_iter()
        .collect(),
        themes: BTreeMap::default(),
        icon_themes: BTreeMap::default(),
    };

    assert_eq!(
        remote_sync_extension_ids(&index),
        ["foo-language", "foo-lsp"]
    );
}

#[test]
fn remote_sync_keeps_shared_language_dependency_once() {
    let index = ExtensionIndex {
        extensions: [
            (
                "aaa-lsp".into(),
                remote_sync_entry(
                    "aaa-lsp",
                    r#"
                    [language_servers.aaa]
                    language = "Foo"
                    "#,
                ),
            ),
            (
                "bbb-lsp".into(),
                remote_sync_entry(
                    "bbb-lsp",
                    r#"
                    [language_servers.bbb]
                    language = "Foo"
                    "#,
                ),
            ),
            (
                "zzz-language".into(),
                remote_sync_entry("zzz-language", r#"languages = ["languages/foo"]"#),
            ),
        ]
        .into_iter()
        .collect(),
        languages: [(
            "Foo".into(),
            remote_sync_language_entry("zzz-language", "languages/foo"),
        )]
        .into_iter()
        .collect(),
        themes: BTreeMap::default(),
        icon_themes: BTreeMap::default(),
    };

    assert_eq!(
        remote_sync_extension_ids(&index),
        ["aaa-lsp", "bbb-lsp", "zzz-language"]
    );
}

#[test]
fn remote_sync_keeps_remote_loadable_extensions_without_language_dependency() {
    let index = ExtensionIndex {
        extensions: [(
            "foo".into(),
            remote_sync_entry(
                "foo",
                r#"
                [language_servers.foo]
                language = "Foo"
                "#,
            ),
        )]
        .into_iter()
        .collect(),
        languages: BTreeMap::default(),
        themes: BTreeMap::default(),
        icon_themes: BTreeMap::default(),
    };

    assert_eq!(remote_sync_extension_ids(&index), ["foo"]);
}

#[test]
fn remote_sync_keeps_debug_adapters() {
    let index = ExtensionIndex {
        extensions: [(
            "foo".into(),
            remote_sync_entry(
                "foo",
                r#"
                [debug_adapters.foo]
                "#,
            ),
        )]
        .into_iter()
        .collect(),
        languages: BTreeMap::default(),
        themes: BTreeMap::default(),
        icon_themes: BTreeMap::default(),
    };

    assert_eq!(remote_sync_extension_ids(&index), ["foo"]);
}

#[gpui::test]
async fn remote_java_runnables_upload_uses_existing_extension_files(cx: &mut TestAppContext) {
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        "/installed/java",
        json!({
            "grammars": { "java.wasm": "java-grammar" },
            "languages": { "java": {
                "config.toml": "name = 'Java'\ngrammar = 'java'\n",
                "runnables.scm": "(method_declaration) @run"
            } }
        }),
    )
    .await;
    fs.insert_tree("/remote", json!({ "languages": { "java": {} } }))
        .await;
    fs.insert_tree("/copper", json!({ "languages": { "java": {} } }))
        .await;
    let manifest = remote_sync_entry(
        "java",
        r#"
        languages = ["languages/java"]
        [grammars.java]
        repository = "https://example.com/java"
        rev = "test"
        "#,
    )
    .manifest;

    copy_java_remote_runnables(
        fs.as_ref(),
        &manifest,
        Path::new("/installed/java"),
        Path::new("/remote"),
        Path::new("/installed/java/languages/java"),
        Path::new("/remote/languages/java"),
        true,
    )
    .await
    .expect("JVM client uploads the grammar and runnable query");
    assert!(fs.is_file(Path::new("/remote/grammars/java.wasm")).await);
    assert!(
        fs.is_file(Path::new("/remote/languages/java/runnables.scm"))
            .await
    );

    copy_java_remote_runnables(
        fs.as_ref(),
        &manifest,
        Path::new("/installed/java"),
        Path::new("/copper"),
        Path::new("/installed/java/languages/java"),
        Path::new("/copper/languages/java"),
        false,
    )
    .await
    .expect("Copper leaves Java runnable data out of remote sync");
    assert!(!fs.is_file(Path::new("/copper/grammars/java.wasm")).await);
    assert!(
        !fs.is_file(Path::new("/copper/languages/java/runnables.scm"))
            .await
    );
}

#[gpui::test]
async fn test_extension_store(cx: &mut TestAppContext) {
    init_test(cx);

    let fs = FakeFs::new(cx.executor());
    let http_client = FakeHttpClient::with_200_response();

    fs.insert_tree(
        "/the-extension-dir",
        json!({
            "installed": {
                "zed-monokai": {
                    "extension.json": r#"{
                        "id": "zed-monokai",
                        "name": "Zed Monokai",
                        "version": "2.0.0",
                        "themes": {
                            "Monokai Dark": "themes/monokai.json",
                            "Monokai Light": "themes/monokai.json",
                            "Monokai Pro Dark": "themes/monokai-pro.json",
                            "Monokai Pro Light": "themes/monokai-pro.json"
                        }
                    }"#,
                    "themes": {
                        "monokai.json": r#"{
                            "name": "Monokai",
                            "author": "Someone",
                            "themes": [
                                {
                                    "name": "Monokai Dark",
                                    "appearance": "dark",
                                    "style": {}
                                },
                                {
                                    "name": "Monokai Light",
                                    "appearance": "light",
                                    "style": {}
                                }
                            ]
                        }"#,
                        "monokai-pro.json": r#"{
                            "name": "Monokai Pro",
                            "author": "Someone",
                            "themes": [
                                {
                                    "name": "Monokai Pro Dark",
                                    "appearance": "dark",
                                    "style": {}
                                },
                                {
                                    "name": "Monokai Pro Light",
                                    "appearance": "light",
                                    "style": {}
                                }
                            ]
                        }"#,
                    }
                },
                "zed-ruby": {
                    "extension.json": r#"{
                        "id": "zed-ruby",
                        "name": "Zed Ruby",
                        "version": "1.0.0",
                        "grammars": {
                            "ruby": "grammars/ruby.wasm",
                            "embedded_template": "grammars/embedded_template.wasm"
                        },
                        "languages": {
                            "ruby": "languages/ruby",
                            "erb": "languages/erb"
                        }
                    }"#,
                    "grammars": {
                        "ruby.wasm": "",
                        "embedded_template.wasm": "",
                    },
                    "languages": {
                        "ruby": {
                            "config.toml": r#"
                                name = "Ruby"
                                grammar = "ruby"
                                path_suffixes = ["rb"]
                            "#,
                            "highlights.scm": "",
                        },
                        "erb": {
                            "config.toml": r#"
                                name = "ERB"
                                grammar = "embedded_template"
                                path_suffixes = ["erb"]
                            "#,
                            "highlights.scm": "",
                        }
                    },
                }
            }
        }),
    )
    .await;

    let mut expected_index = ExtensionIndex {
        extensions: [
            (
                "zed-ruby".into(),
                ExtensionIndexEntry {
                    manifest: Arc::new(ExtensionManifest {
                        id: "zed-ruby".into(),
                        name: "Zed Ruby".into(),
                        version: "1.0.0".into(),
                        schema_version: SchemaVersion::ZERO,
                        description: None,
                        authors: Vec::new(),
                        repository: None,
                        themes: Default::default(),
                        icon_themes: Vec::new(),
                        lib: Default::default(),
                        languages: vec![
                            rel_path_buf("languages/erb"),
                            rel_path_buf("languages/ruby"),
                        ],
                        grammars: [
                            ("embedded_template".into(), GrammarManifestEntry::default()),
                            ("ruby".into(), GrammarManifestEntry::default()),
                        ]
                        .into_iter()
                        .collect(),
                        language_servers: BTreeMap::default(),
                        context_servers: BTreeMap::default(),
                        slash_commands: BTreeMap::default(),
                        snippets: None,
                        capabilities: Vec::new(),
                        debug_adapters: Default::default(),
                        debug_locators: Default::default(),
                        language_model_providers: BTreeMap::default(),
                    }),
                    dev: false,
                },
            ),
            (
                "zed-monokai".into(),
                ExtensionIndexEntry {
                    manifest: Arc::new(ExtensionManifest {
                        id: "zed-monokai".into(),
                        name: "Zed Monokai".into(),
                        version: "2.0.0".into(),
                        schema_version: SchemaVersion::ZERO,
                        description: None,
                        authors: vec![],
                        repository: None,
                        themes: vec![
                            rel_path_buf("themes/monokai-pro.json"),
                            rel_path_buf("themes/monokai.json"),
                        ],
                        icon_themes: Vec::new(),
                        lib: Default::default(),
                        languages: Default::default(),
                        grammars: BTreeMap::default(),
                        language_servers: BTreeMap::default(),
                        context_servers: BTreeMap::default(),
                        slash_commands: BTreeMap::default(),
                        snippets: None,
                        capabilities: Vec::new(),
                        debug_adapters: Default::default(),
                        debug_locators: Default::default(),
                        language_model_providers: BTreeMap::default(),
                    }),
                    dev: false,
                },
            ),
        ]
        .into_iter()
        .collect(),
        languages: [
            (
                "ERB".into(),
                ExtensionIndexLanguageEntry {
                    extension: "zed-ruby".into(),
                    path: "languages/erb".into(),
                    grammar: Some("embedded_template".into()),
                    hidden: false,
                    matcher: (LanguageMatcher {
                        path_suffixes: vec!["erb".into()],
                        first_line_pattern: None,
                        ..LanguageMatcher::default()
                    })
                    .into(),
                },
            ),
            (
                "Ruby".into(),
                ExtensionIndexLanguageEntry {
                    extension: "zed-ruby".into(),
                    path: "languages/ruby".into(),
                    grammar: Some("ruby".into()),
                    hidden: false,
                    matcher: (LanguageMatcher {
                        path_suffixes: vec!["rb".into()],
                        first_line_pattern: None,
                        ..LanguageMatcher::default()
                    })
                    .into(),
                },
            ),
        ]
        .into_iter()
        .collect(),
        themes: [
            (
                "Monokai Dark".into(),
                ExtensionIndexThemeEntry {
                    extension: "zed-monokai".into(),
                    path: "themes/monokai.json".into(),
                },
            ),
            (
                "Monokai Light".into(),
                ExtensionIndexThemeEntry {
                    extension: "zed-monokai".into(),
                    path: "themes/monokai.json".into(),
                },
            ),
            (
                "Monokai Pro Dark".into(),
                ExtensionIndexThemeEntry {
                    extension: "zed-monokai".into(),
                    path: "themes/monokai-pro.json".into(),
                },
            ),
            (
                "Monokai Pro Light".into(),
                ExtensionIndexThemeEntry {
                    extension: "zed-monokai".into(),
                    path: "themes/monokai-pro.json".into(),
                },
            ),
        ]
        .into_iter()
        .collect(),
        icon_themes: BTreeMap::default(),
    };

    let proxy = Arc::new(ExtensionHostProxy::new());
    let theme_registry = Arc::new(ThemeRegistry::new(Box::new(())));
    theme_extension::init(proxy.clone(), theme_registry.clone(), cx.executor());
    let language_registry = Arc::new(LanguageRegistry::test(cx.executor()));
    language_extension::init(LspAccess::Noop, proxy.clone(), language_registry.clone());
    let node_runtime = NodeRuntime::unavailable();

    let store = cx.new(|cx| {
        ExtensionStore::new(
            PathBuf::from("/the-extension-dir"),
            None,
            proxy.clone(),
            fs.clone(),
            http_client.clone(),
            http_client.clone(),
            None,
            node_runtime.clone(),
            Vec::new(),
            cx,
        )
    });

    cx.executor().advance_clock(RELOAD_DEBOUNCE_DURATION);
    store.read_with(cx, |store, _| {
        let index = &store.extension_index;
        assert_eq!(index.extensions, expected_index.extensions);

        for ((actual_key, actual_language), (expected_key, expected_language)) in
            index.languages.iter().zip(expected_index.languages.iter())
        {
            assert_eq!(actual_key, expected_key);
            assert_eq!(actual_language.grammar, expected_language.grammar);
            assert_eq!(actual_language.matcher, expected_language.matcher);
            assert_eq!(actual_language.hidden, expected_language.hidden);
        }
        assert_eq!(index.themes, expected_index.themes);

        assert_eq!(
            language_registry.language_names(),
            [
                LanguageName::new_static("ERB"),
                LanguageName::new_static("Plain Text"),
                LanguageName::new_static("Ruby"),
            ]
        );
        assert_eq!(
            theme_registry.list_names(),
            [
                "Monokai Dark",
                "Monokai Light",
                "Monokai Pro Dark",
                "Monokai Pro Light",
                "One Dark",
            ]
        );
    });

    fs.insert_tree(
        "/the-extension-dir/installed/zed-gruvbox",
        json!({
            "extension.json": r#"{
                "id": "zed-gruvbox",
                "name": "Zed Gruvbox",
                "version": "1.0.0",
                "themes": {
                    "Gruvbox": "themes/gruvbox.json"
                }
            }"#,
            "themes": {
                "gruvbox.json": r#"{
                    "name": "Gruvbox",
                    "author": "Someone Else",
                    "themes": [
                        {
                            "name": "Gruvbox",
                            "appearance": "dark",
                            "style": {}
                        }
                    ]
                }"#,
            }
        }),
    )
    .await;

    expected_index.extensions.insert(
        "zed-gruvbox".into(),
        ExtensionIndexEntry {
            manifest: Arc::new(ExtensionManifest {
                id: "zed-gruvbox".into(),
                name: "Zed Gruvbox".into(),
                version: "1.0.0".into(),
                schema_version: SchemaVersion::ZERO,
                description: None,
                authors: vec![],
                repository: None,
                themes: vec![rel_path_buf("themes/gruvbox.json")],
                icon_themes: Vec::new(),
                lib: Default::default(),
                languages: Default::default(),
                grammars: BTreeMap::default(),
                language_servers: BTreeMap::default(),
                context_servers: BTreeMap::default(),
                slash_commands: BTreeMap::default(),
                snippets: None,
                capabilities: Vec::new(),
                debug_adapters: Default::default(),
                debug_locators: Default::default(),
                language_model_providers: BTreeMap::default(),
            }),
            dev: false,
        },
    );
    expected_index.themes.insert(
        "Gruvbox".into(),
        ExtensionIndexThemeEntry {
            extension: "zed-gruvbox".into(),
            path: "themes/gruvbox.json".into(),
        },
    );

    let reload = store.update(cx, |store, cx| store.reload(None, cx));

    cx.executor().advance_clock(RELOAD_DEBOUNCE_DURATION);
    reload.await.expect("reload extension inventory");
    store.read_with(cx, |store, _| {
        let index = &store.extension_index;

        for ((actual_key, actual_language), (expected_key, expected_language)) in
            index.languages.iter().zip(expected_index.languages.iter())
        {
            assert_eq!(actual_key, expected_key);
            assert_eq!(actual_language.grammar, expected_language.grammar);
            assert_eq!(actual_language.matcher, expected_language.matcher);
            assert_eq!(actual_language.hidden, expected_language.hidden);
        }

        assert_eq!(index.extensions, expected_index.extensions);
        assert_eq!(index.themes, expected_index.themes);

        assert_eq!(
            theme_registry.list_names(),
            [
                "Gruvbox",
                "Monokai Dark",
                "Monokai Light",
                "Monokai Pro Dark",
                "Monokai Pro Light",
                "One Dark",
            ]
        );
    });

    let prev_fs_metadata_call_count = fs.metadata_call_count();
    let prev_fs_read_dir_call_count = fs.read_dir_call_count();

    // Create new extension store, as if Zed were restarting.
    drop(store);
    let store = cx.new(|cx| {
        ExtensionStore::new(
            PathBuf::from("/the-extension-dir"),
            None,
            proxy,
            fs.clone(),
            http_client.clone(),
            http_client.clone(),
            None,
            node_runtime.clone(),
            Vec::new(),
            cx,
        )
    });

    cx.executor().run_until_parked();
    store.read_with(cx, |store, _| {
        assert_eq!(store.extension_index.extensions, expected_index.extensions);
        assert_eq!(store.extension_index.themes, expected_index.themes);
        assert_eq!(
            store.extension_index.icon_themes,
            expected_index.icon_themes
        );

        for ((actual_key, actual_language), (expected_key, expected_language)) in store
            .extension_index
            .languages
            .iter()
            .zip(expected_index.languages.iter())
        {
            assert_eq!(actual_key, expected_key);
            assert_eq!(actual_language.grammar, expected_language.grammar);
            assert_eq!(actual_language.matcher, expected_language.matcher);
            assert_eq!(actual_language.hidden, expected_language.hidden);
        }

        assert_eq!(
            language_registry.language_names(),
            [
                LanguageName::new_static("ERB"),
                LanguageName::new_static("Plain Text"),
                LanguageName::new_static("Ruby"),
            ]
        );
        assert_eq!(
            language_registry.grammar_names(),
            ["embedded_template".into(), "ruby".into()]
        );
        assert_eq!(
            theme_registry.list_names(),
            [
                "Gruvbox",
                "Monokai Dark",
                "Monokai Light",
                "Monokai Pro Dark",
                "Monokai Pro Light",
                "One Dark",
            ]
        );

        // The on-disk manifest limits the number of FS calls that need to be made
        // on startup.
        assert_eq!(fs.read_dir_call_count(), prev_fs_read_dir_call_count);
        // The portable package inventory adds one metadata probe without scanning package files.
        assert_eq!(fs.metadata_call_count(), prev_fs_metadata_call_count + 12);
    });

    store.update(cx, |store, cx| {
        store
            .uninstall_extension("zed-ruby".into(), cx)
            .detach_and_log_err(cx);
    });

    cx.executor().advance_clock(RELOAD_DEBOUNCE_DURATION);
    expected_index.extensions.remove("zed-ruby");
    expected_index.languages.remove("Ruby");
    expected_index.languages.remove("ERB");

    store.read_with(cx, |store, _| {
        assert_eq!(store.extension_index.extensions, expected_index.extensions);
        assert_eq!(store.extension_index.themes, expected_index.themes);
        assert_eq!(
            store.extension_index.icon_themes,
            expected_index.icon_themes
        );

        for ((actual_key, actual_language), (expected_key, expected_language)) in store
            .extension_index
            .languages
            .iter()
            .zip(expected_index.languages.iter())
        {
            assert_eq!(actual_key, expected_key);
            assert_eq!(actual_language.grammar, expected_language.grammar);
            assert_eq!(actual_language.matcher, expected_language.matcher);
            assert_eq!(actual_language.hidden, expected_language.hidden);
        }

        assert_eq!(
            language_registry.language_names(),
            [LanguageName::new_static("Plain Text")]
        );
        assert_eq!(language_registry.grammar_names(), []);
    });
}

#[gpui::test]
async fn test_extension_store_with_test_extension(cx: &mut TestAppContext) {
    init_test(cx);
    cx.executor().allow_parking();

    let executor = cx.executor();
    async fn await_or_timeout<T>(
        executor: &BackgroundExecutor,
        what: &'static str,
        seconds: u64,
        future: impl std::future::Future<Output = T>,
    ) -> T {
        let timeout = executor.timer(std::time::Duration::from_secs(seconds));

        futures::select! {
            output = future.fuse() => output,
            _ = futures::FutureExt::fuse(timeout) => panic!(
            "[test_extension_store_with_test_extension] timed out after {seconds}s while {what}"
        )
        }
    }

    let root_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let cache_dir = root_dir.join("target");
    let test_extension_id = "test-extension";
    let test_extension_dir = root_dir.join("extensions").join(test_extension_id);

    let fs = Arc::new(RealFs::new(None, cx.executor()));
    let extensions_tree = TempTree::new(json!({
        "installed": {},
        "work": {}
    }));
    let project_dir = TempTree::new(json!({
        "test.gleam": ""
    }));

    let extensions_dir = extensions_tree.path().canonicalize().unwrap();
    let project_dir = project_dir.path().canonicalize().unwrap();

    let project = await_or_timeout(
        &executor,
        "awaiting Project::test",
        5,
        Project::test(fs.clone(), [project_dir.as_path()], cx),
    )
    .await;

    let proxy = Arc::new(ExtensionHostProxy::new());
    let theme_registry = Arc::new(ThemeRegistry::new(Box::new(())));
    theme_extension::init(proxy.clone(), theme_registry.clone(), cx.executor());
    let language_registry = project.read_with(cx, |project, _cx| project.languages().clone());
    language_extension::init(
        LspAccess::ViaLspStore(
            project
                .update(cx, |project, _| project.lsp_store())
                .downgrade(),
        ),
        proxy.clone(),
        language_registry.clone(),
    );
    let node_runtime = NodeRuntime::unavailable();

    let mut status_updates = language_registry.language_server_binary_statuses();

    struct FakeLanguageServerVersion {
        version: String,
        binary_contents: String,
        http_request_count: usize,
    }

    let language_server_version = Arc::new(Mutex::new(FakeLanguageServerVersion {
        version: "v1.2.3".into(),
        binary_contents: "the-binary-contents".into(),
        http_request_count: 0,
    }));

    let extension_client = FakeHttpClient::create({
        let language_server_version = language_server_version.clone();
        move |request| {
            let language_server_version = language_server_version.clone();
            async move {
                let version = language_server_version.lock().version.clone();
                let binary_contents = language_server_version.lock().binary_contents.clone();

                let github_releases_uri = "https://api.github.com/repos/gleam-lang/gleam/releases";
                let asset_download_uri =
                    format!("https://fake-download.example.com/gleam-{version}");

                let uri = request.uri().to_string();
                if uri == github_releases_uri {
                    language_server_version.lock().http_request_count += 1;
                    Ok(Response::new(
                        json!([
                            {
                                "tag_name": version,
                                "prerelease": false,
                                "tarball_url": "",
                                "zipball_url": "",
                                "assets": [
                                    {
                                        "name": format!("gleam-{version}-aarch64-apple-darwin.tar.gz"),
                                        "browser_download_url": asset_download_uri
                                    },
                                    {
                                        "name": format!("gleam-{version}-x86_64-unknown-linux-musl.tar.gz"),
                                        "browser_download_url": asset_download_uri
                                    },
                                    {
                                        "name": format!("gleam-{version}-aarch64-unknown-linux-musl.tar.gz"),
                                        "browser_download_url": asset_download_uri
                                    },
                                    {
                                        "name": format!("gleam-{version}-x86_64-pc-windows-msvc.tar.gz"),
                                        "browser_download_url": asset_download_uri
                                    }
                                ]
                            }
                        ])
                        .to_string()
                        .into(),
                    ))
                } else if uri == asset_download_uri {
                    language_server_version.lock().http_request_count += 1;
                    let mut bytes = Vec::<u8>::new();
                    let mut archive = async_tar::Builder::new(&mut bytes);
                    let mut header = async_tar::Header::new_gnu();
                    header.set_size(binary_contents.len() as u64);
                    archive
                        .append_data(&mut header, "gleam", binary_contents.as_bytes())
                        .await
                        .unwrap();
                    archive.into_inner().await.unwrap();
                    let mut gzipped_bytes = Vec::new();
                    let mut encoder = GzipEncoder::new(BufReader::new(bytes.as_slice()));
                    encoder.read_to_end(&mut gzipped_bytes).await.unwrap();
                    Ok(Response::new(gzipped_bytes.into()))
                } else {
                    Ok(Response::builder().status(404).body("not found".into())?)
                }
            }
        }
    });
    let user_agent = cx.update(|cx| {
        format!(
            "Zed/{} ({}; {})",
            AppVersion::global(cx),
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    });
    let builder_client =
        Arc::new(ReqwestClient::user_agent(&user_agent).expect("Could not create HTTP client"));

    let extension_store = cx.new(|cx| {
        ExtensionStore::new(
            extensions_dir.clone(),
            Some(cache_dir),
            proxy,
            fs.clone(),
            extension_client.clone(),
            builder_client,
            None,
            node_runtime,
            Vec::new(),
            cx,
        )
    });

    // Ensure that debounces fire.
    let mut events = cx.events(&extension_store);
    let executor = cx.executor();
    let _task = cx.executor().spawn(async move {
        while let Some(event) = events.next().await {
            if let Event::StartedReloading = event {
                executor.advance_clock(RELOAD_DEBOUNCE_DURATION);
            }
        }
    });

    extension_store.update(cx, |_, cx| {
        cx.subscribe(&extension_store, |_, _, event, _| {
            if matches!(event, Event::ExtensionFailedToLoad(_)) {
                panic!("extension failed to load");
            }
        })
        .detach();
    });

    let mut extension_events = cx.events(&cx.update(|cx| {
        extension::ExtensionEvents::try_global(cx)
            .expect("ExtensionEvents should be initialized in tests")
    }));

    let executor = cx.executor();
    await_or_timeout(
        &executor,
        "awaiting install_dev_extension",
        180,
        extension_store.update(cx, |store, cx| {
            store.install_dev_extension(test_extension_dir.clone(), cx)
        }),
    )
    .await
    .unwrap();

    await_or_timeout(
        &executor,
        "awaiting ExtensionsInstalledChanged",
        10,
        async {
            while let Some(event) = extension_events.next().await {
                if matches!(event, extension::Event::ExtensionsInstalledChanged) {
                    return;
                }
            }

            panic!(
                "[test_extension_store_with_test_extension] extension event stream ended before ExtensionsInstalledChanged"
            );
        },
    )
    .await;

    let mut fake_servers = language_registry.register_fake_lsp_server(
        LanguageServerName("gleam".into()),
        lsp::ServerCapabilities {
            completion_provider: Some(Default::default()),
            ..Default::default()
        },
        None,
    );
    cx.executor().run_until_parked();

    let mut project_events = cx.events(&project);
    let buffer_path = project_dir.join("test.gleam");
    let (buffer, _handle) = await_or_timeout(
        &executor,
        "awaiting open_local_buffer_with_lsp",
        5,
        project.update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(buffer_path.clone(), cx)
        }),
    )
    .await
    .unwrap();
    cx.executor().run_until_parked();

    let buffer_remote_id = buffer.read_with(cx, |buffer, _cx| buffer.remote_id());

    let fake_server = await_or_timeout(
        &executor,
        "awaiting first fake server spawn",
        10,
        fake_servers.next(),
    )
    .await
    .unwrap();

    let work_dir = extensions_dir.join(format!("work/{test_extension_id}"));
    let expected_server_path = work_dir.join("gleam-v1.2.3/gleam");
    let expected_binary_contents = language_server_version.lock().binary_contents.clone();

    // check that IO operations in extension work correctly
    assert!(work_dir.join("dir-created-with-rel-path").exists());
    assert!(work_dir.join("dir-created-with-abs-path").exists());
    assert!(work_dir.join("file-created-with-abs-path").exists());
    assert!(work_dir.join("file-created-with-rel-path").exists());

    assert_eq!(fake_server.binary.path, expected_server_path);
    assert_eq!(fake_server.binary.arguments, [OsString::from("lsp")]);
    assert_eq!(
        await_or_timeout(
            &executor,
            "awaiting fs.load(expected_server_path)",
            5,
            fs.load(&expected_server_path)
        )
        .await
        .unwrap(),
        expected_binary_contents
    );
    assert_eq!(language_server_version.lock().http_request_count, 2);
    assert_eq!(
        [
            await_or_timeout(
                &executor,
                "awaiting status_updates #1",
                5,
                status_updates.next()
            )
            .await
            .unwrap(),
            await_or_timeout(
                &executor,
                "awaiting status_updates #2",
                5,
                status_updates.next()
            )
            .await
            .unwrap(),
            await_or_timeout(
                &executor,
                "awaiting status_updates #3",
                5,
                status_updates.next()
            )
            .await
            .unwrap(),
            await_or_timeout(
                &executor,
                "awaiting status_updates #4",
                5,
                status_updates.next()
            )
            .await
            .unwrap(),
        ],
        [
            (
                LanguageServerName::new_static("gleam"),
                BinaryStatus::Starting
            ),
            (
                LanguageServerName::new_static("gleam"),
                BinaryStatus::CheckingForUpdate
            ),
            (
                LanguageServerName::new_static("gleam"),
                BinaryStatus::Downloading
            ),
            (LanguageServerName::new_static("gleam"), BinaryStatus::None)
        ]
    );

    // The extension creates custom labels for completion items.
    fake_server.set_request_handler::<lsp::request::Completion, _, _>(|_, _| async move {
        Ok(Some(lsp::CompletionResponse::Array(vec![
            lsp::CompletionItem {
                label: "foo".into(),
                kind: Some(lsp::CompletionItemKind::FUNCTION),
                detail: Some("fn() -> Result(Nil, Error)".into()),
                ..Default::default()
            },
            lsp::CompletionItem {
                label: "bar.baz".into(),
                kind: Some(lsp::CompletionItemKind::FUNCTION),
                detail: Some("fn(List(a)) -> a".into()),
                ..Default::default()
            },
            lsp::CompletionItem {
                label: "Quux".into(),
                kind: Some(lsp::CompletionItemKind::CONSTRUCTOR),
                detail: Some("fn(String) -> T".into()),
                ..Default::default()
            },
            lsp::CompletionItem {
                label: "my_string".into(),
                kind: Some(lsp::CompletionItemKind::CONSTANT),
                detail: Some("String".into()),
                ..Default::default()
            },
        ])))
    });

    // `register_fake_lsp_server` can yield a server instance before the client has fully registered
    // the buffer with the project LSP plumbing. Wait for the project to observe that registration
    // before issuing requests like completion.
    await_or_timeout(
        &executor,
        "awaiting LanguageServerBufferRegistered",
        5,
        async {
            while let Some(event) = project_events.next().await {
                if let project::Event::LanguageServerBufferRegistered { buffer_id, .. } = event {
                    if buffer_id == buffer_remote_id {
                        return;
                    }
                }
            }

            panic!(
                "[test_extension_store_with_test_extension] project event stream ended before buffer registration for {}",
                buffer_path.display()
            );
        },
    )
    .await;

    let completion_labels = await_or_timeout(
        &executor,
        "awaiting completions",
        5,
        project.update(cx, |project, cx| {
            project.completions(&buffer, 0, DEFAULT_COMPLETION_CONTEXT, cx)
        }),
    )
    .await
    .unwrap()
    .into_iter()
    .flat_map(|response| response.completions)
    .map(|c| c.label.text)
    .collect::<Vec<_>>();
    assert_eq!(
        completion_labels,
        [
            "foo: fn() -> Result(Nil, Error)".to_string(),
            "bar.baz: fn(List(a)) -> a".to_string(),
            "Quux: fn(String) -> T".to_string(),
            "my_string: String".to_string(),
        ]
    );

    // Simulate a new version of the language server being released
    language_server_version.lock().version = "v2.0.0".into();
    language_server_version.lock().binary_contents = "the-new-binary-contents".into();
    language_server_version.lock().http_request_count = 0;

    // Start a new instance of the language server.
    project.update(cx, |project, cx| {
        project.restart_language_servers_for_buffers(
            vec![buffer.clone()],
            HashSet::default(),
            true,
            cx,
        )
    });
    cx.executor().run_until_parked();

    // The extension has cached the binary path, and does not attempt
    // to reinstall it.
    let fake_server = await_or_timeout(
        &executor,
        "awaiting second fake server spawn",
        5,
        fake_servers.next(),
    )
    .await
    .unwrap();
    assert_eq!(fake_server.binary.path, expected_server_path);
    assert_eq!(
        await_or_timeout(
            &executor,
            "awaiting fs.load(expected_server_path) after restart",
            5,
            fs.load(&expected_server_path)
        )
        .await
        .unwrap(),
        expected_binary_contents
    );
    assert_eq!(language_server_version.lock().http_request_count, 0);

    // Reload the extension, clearing its cache.
    // Start a new instance of the language server.
    await_or_timeout(
        &executor,
        "awaiting extension_store.reload(test-extension)",
        5,
        extension_store.update(cx, |store, cx| {
            store.reload(Some("test-extension".into()), cx)
        }),
    )
    .await
    .expect("reload test extension");
    cx.executor().run_until_parked();
    project.update(cx, |project, cx| {
        project.restart_language_servers_for_buffers(
            vec![buffer.clone()],
            HashSet::default(),
            true,
            cx,
        )
    });

    // The extension re-fetches the latest version of the language server.
    let fake_server = await_or_timeout(
        &executor,
        "awaiting third fake server spawn",
        5,
        fake_servers.next(),
    )
    .await
    .unwrap();
    let new_expected_server_path =
        extensions_dir.join(format!("work/{test_extension_id}/gleam-v2.0.0/gleam"));
    let expected_binary_contents = language_server_version.lock().binary_contents.clone();
    assert_eq!(fake_server.binary.path, new_expected_server_path);
    assert_eq!(fake_server.binary.arguments, [OsString::from("lsp")]);
    assert_eq!(
        await_or_timeout(
            &executor,
            "awaiting fs.load(new_expected_server_path)",
            5,
            fs.load(&new_expected_server_path)
        )
        .await
        .unwrap(),
        expected_binary_contents
    );

    // The old language server directory has been cleaned up.
    assert!(
        await_or_timeout(
            &executor,
            "awaiting fs.metadata(expected_server_path)",
            5,
            fs.metadata(&expected_server_path)
        )
        .await
        .unwrap()
        .is_none()
    );
}

fn init_test(cx: &mut TestAppContext) {
    cx.update(|cx| {
        let store = SettingsStore::test(cx);
        cx.set_global(store);
        release_channel::init(semver::Version::new(0, 0, 0), cx);
        extension::init(cx);
        theme_settings::init(theme::LoadThemes::JustBase, cx);
        gpui_tokio::init(cx);
    });
}

#[gpui::test]
#[cfg(target_os = "macos")]
#[ignore = "requires pinned GROOVY_INSTALLED_EXTENSIONS, GROOVY_JAVA_PROXY, GROOVY_JAVA_CACHE and GROOVY_JAVA_HOME"]
async fn installed_jvm_extensions_respect_denied_host_capabilities(cx: &mut TestAppContext) {
    use crate::wasm_host::{WasmExtension, WasmHost};
    use extension::{Extension as _, WorktreeDelegate};
    use gpui::BorrowAppContext as _;
    use util::rel_path::RelPath;

    async fn cached_jdtls_command(
        extension: &WasmExtension,
        cx: &TestAppContext,
    ) -> extension::Command {
        let command = extension.language_server_command(
            LanguageServerName::from("jdtls"),
            LanguageName::new("Java"),
            Arc::new(DeniedWorktree),
        );
        match futures::future::select(
            Box::pin(command),
            cx.executor().timer(std::time::Duration::from_secs(30)),
        )
        .await
        {
            futures::future::Either::Left((result, _)) => result.expect("cached JDTLS command"),
            futures::future::Either::Right(_) => panic!("cached JDTLS workspace command timed out"),
        }
    }

    struct DeniedWorktree;

    #[async_trait::async_trait]
    impl WorktreeDelegate for DeniedWorktree {
        fn id(&self) -> u64 {
            1
        }

        fn root_path(&self) -> String {
            "/capability-fixture".into()
        }

        async fn read_text_file(&self, _: &RelPath) -> anyhow::Result<String> {
            anyhow::bail!("The capability fixture has no project files")
        }

        async fn which(&self, binary_name: String) -> Option<String> {
            (binary_name == "java").then(|| "/usr/bin/java".into())
        }

        async fn shell_env(&self) -> Vec<(String, String)> {
            vec![
                ("JAVA_HOME".into(), "/usr".into()),
                ("HOME".into(), "/capability-fixture".into()),
            ]
        }
    }

    init_test(cx);
    cx.executor().allow_parking();
    let installed = PathBuf::from(
        std::env::var_os("GROOVY_INSTALLED_EXTENSIONS")
            .expect("set GROOVY_INSTALLED_EXTENSIONS to the installed extension directory"),
    );
    let fs = Arc::new(RealFs::new(None, cx.executor()));
    let project_tree = TempTree::new(json!({ "pom.xml": "<project/>" }));
    let project = Project::test(fs.clone(), [project_tree.path()], cx).await;
    cx.update(|cx| {
        cx.update_global::<SettingsStore, _>(|store, cx| {
            store
                .set_user_settings(
                    r#"{"granted_extension_capabilities":[],"auto_install_extensions":{"java":true,"kotlin":true}}"#,
                    cx,
                )
                .expect("deny host capabilities while retaining product defaults");
        });
    });

    let requests = Arc::new(Mutex::new(Vec::new()));
    let http_client = FakeHttpClient::create({
        let requests = requests.clone();
        move |request| {
            let requests = requests.clone();
            async move {
                let uri = request.uri().to_string();
                requests.lock().push(uri.clone());
                let body = match uri.as_str() {
                    "https://raw.githubusercontent.com/Kotlin/kotlin-lsp/refs/heads/main/RELEASES.md" => {
                        "### v263.4702.0\n"
                    }
                    "https://api.github.com/repos/projectlombok/lombok/tags" => {
                        r#"[{"name":"v1.18.42"}]"#
                    }
                    "https://api.github.com/repos/fwcd/kotlin-language-server/releases" => {
                        r#"[{"tag_name":"1.3.13","prerelease":false,"assets":[{"name":"server.zip","browser_download_url":"https://github.com/fwcd/kotlin-language-server/releases/download/1.3.13/server.zip"}],"tarball_url":"","zipball_url":""}]"#
                    }
                    _ => anyhow::bail!(
                        "Unexpected transport access in denied-capability fixture: {uri}"
                    ),
                };
                Ok(Response::new(http_client::AsyncBody::from(body)))
            }
        }
    });
    let work_tree = TempTree::new(json!({}));
    let host = cx.update(|cx| {
        WasmHost::new(
            fs.clone(),
            http_client.clone(),
            NodeRuntime::unavailable(),
            Arc::new(ExtensionHostProxy::default()),
            work_tree.path().to_path_buf(),
            cx,
        )
    });
    assert!(host.granted_capabilities.is_empty());

    for (extension_id, version, server_id, language, denied_operation) in [
        ("java", "6.8.27", "jdtls", "Java", "download_file"),
        (
            "java",
            "6.8.27",
            "gradle-language-server",
            "Gradle",
            "download_file",
        ),
        ("kotlin", "0.3.2", "kotlin-lsp", "Kotlin", "download_file"),
        (
            "kotlin",
            "0.3.2",
            "kotlin-language-server",
            "Kotlin",
            "download_file",
        ),
    ] {
        let directory = installed.join(extension_id);
        let manifest = Arc::new(
            ExtensionManifest::load(fs.clone(), &directory)
                .await
                .expect("load installed JVM manifest"),
        );
        assert_eq!(manifest.id.as_ref(), extension_id);
        assert_eq!(manifest.version.as_ref(), version);
        let extension = WasmExtension::load(&directory, &manifest, host.clone(), &cx.to_async())
            .await
            .expect("load installed JVM WASM through the shared extension host");
        let command = extension.language_server_command(
            LanguageServerName::from(server_id),
            LanguageName::new(language),
            Arc::new(DeniedWorktree),
        );
        let error = match futures::future::select(
            Box::pin(command),
            cx.executor().timer(std::time::Duration::from_secs(30)),
        )
        .await
        {
            futures::future::Either::Left((result, _)) => {
                result.expect_err("a denied provider must not return a launch command")
            }
            futures::future::Either::Right(_) => panic!("denied provider acquisition timed out"),
        };
        let reason = format!("{error:#}");
        assert!(
            reason.contains(denied_operation),
            "{extension_id}: {reason}"
        );
        assert!(
            reason.contains("not granted by the extension host"),
            "{reason}"
        );
        if server_id == "gradle-language-server" {
            assert!(
                reason.contains("Failed to download Gradle LS VSIX"),
                "{reason}"
            );
            assert!(reason.contains("/vscode-gradle/3.18.0/"), "{reason}");
            let cache = work_tree.path().join("java/gradle-ls/3.18.0");
            assert!(
                std::fs::read_dir(cache)
                    .expect("inspect denied Gradle server cache")
                    .next()
                    .is_none(),
                "denied Gradle setup must not populate its runtime library cache"
            );
        }
        println!("{extension_id} {version} {server_id}: denied {denied_operation} before launch");
    }
    assert_eq!(
        requests.lock().as_slice(),
        [
            "https://raw.githubusercontent.com/Kotlin/kotlin-lsp/refs/heads/main/RELEASES.md",
            "https://api.github.com/repos/fwcd/kotlin-language-server/releases",
        ]
    );
    let kotlin_cache = std::fs::read_dir(work_tree.path().join("kotlin"))
        .expect("inspect disposable Kotlin work directory")
        .collect::<std::io::Result<Vec<_>>>()
        .expect("read disposable Kotlin work entries");
    assert!(
        kotlin_cache.is_empty(),
        "denied acquisition must not create a server archive or cache"
    );

    let proxy_binary = PathBuf::from(
        std::env::var_os("GROOVY_JAVA_PROXY")
            .expect("set GROOVY_JAVA_PROXY to the installed Java 6.8.27 native proxy"),
    );
    let proxy_cache = work_tree.path().join("java/bin/v6.8.27");
    std::fs::create_dir_all(&proxy_cache).expect("create disposable Java proxy cache");
    std::fs::copy(proxy_binary, proxy_cache.join("java-lsp-proxy"))
        .expect("copy cached native proxy without executing it");
    let directory = installed.join("java");
    let manifest = Arc::new(
        ExtensionManifest::load(fs.clone(), &directory)
            .await
            .expect("reload pinned Java manifest"),
    );
    let extension = WasmExtension::load(&directory, &manifest, host.clone(), &cx.to_async())
        .await
        .expect("load Java with its cached proxy and denied download capability");
    let command = extension.language_server_command(
        LanguageServerName::from("jdtls"),
        LanguageName::new("Java"),
        Arc::new(DeniedWorktree),
    );
    let error = match futures::future::select(
        Box::pin(command),
        cx.executor().timer(std::time::Duration::from_secs(30)),
    )
    .await
    {
        futures::future::Either::Left((result, _)) => {
            result.expect_err("denied Java acquisition must not return a launch command")
        }
        futures::future::Either::Right(_) => panic!("denied Java acquisition timed out"),
    };
    let reason = format!("{error:#}");
    assert!(reason.contains("download_file"), "{reason}");
    assert!(reason.contains("lombok"), "{reason}");
    assert!(
        reason.contains("not granted by the extension host"),
        "{reason}"
    );
    assert_eq!(
        requests.lock().as_slice(),
        [
            "https://raw.githubusercontent.com/Kotlin/kotlin-lsp/refs/heads/main/RELEASES.md",
            "https://api.github.com/repos/fwcd/kotlin-language-server/releases",
            "https://api.github.com/repos/projectlombok/lombok/tags",
        ]
    );
    println!("java 6.8.27 jdtls: denied Lombok download with a cached native proxy");

    let java_cache = PathBuf::from(
        std::env::var_os("GROOVY_JAVA_CACHE")
            .expect("set GROOVY_JAVA_CACHE to the installed JDTLS 1.61.0/Lombok 1.18.48 cache"),
    );
    let cached_lombok = java_cache.join("lombok/lombok-1.18.48.jar");
    let lombok_cache = work_tree.path().join("java/lombok");
    std::fs::create_dir_all(&lombok_cache).expect("create disposable Lombok cache");
    std::fs::copy(cached_lombok, lombok_cache.join("lombok-1.18.48.jar"))
        .expect("copy installed Lombok without executing it");
    fs::copy_recursive(
        fs.as_ref(),
        &java_cache.join("jdtls/1.61.0"),
        &work_tree.path().join("java/jdtls/1.61.0"),
        fs::CopyOptions::default(),
    )
    .await
    .expect("copy complete installed JDTLS cache without executing it");
    cx.update(|cx| {
        cx.update_global::<SettingsStore, _>(|store, cx| {
            store
                .set_user_settings(
                    r#"{"granted_extension_capabilities":[],"auto_install_extensions":{"java":true,"kotlin":true},"lsp":{"jdtls":{"settings":{"check_updates":"never"}}}}"#,
                    cx,
                )
                .expect("use cached provider files while retaining denied host capabilities");
        });
    });
    let extension = WasmExtension::load(&directory, &manifest, host.clone(), &cx.to_async())
        .await
        .expect("load Java with complete cached setup and denied process capability");
    let command = extension.language_server_command(
        LanguageServerName::from("jdtls"),
        LanguageName::new("Java"),
        Arc::new(DeniedWorktree),
    );
    let error = match futures::future::select(
        Box::pin(command),
        cx.executor().timer(std::time::Duration::from_secs(30)),
    )
    .await
    {
        futures::future::Either::Left((result, _)) => {
            result.expect_err("denied Java process probe must not return a launch command")
        }
        futures::future::Either::Right(_) => panic!("denied Java process probe timed out"),
    };
    let reason = format!("{error:#}");
    assert!(
        reason.contains("Failed to determine Java version"),
        "{reason}"
    );
    assert!(
        reason.contains(r#"capability for process:exec /usr/bin/java ["-version"]"#),
        "{reason}"
    );
    assert!(
        reason.contains("not granted by the extension host"),
        "{reason}"
    );
    assert_eq!(
        requests.lock().as_slice(),
        [
            "https://raw.githubusercontent.com/Kotlin/kotlin-lsp/refs/heads/main/RELEASES.md",
            "https://api.github.com/repos/fwcd/kotlin-language-server/releases",
            "https://api.github.com/repos/projectlombok/lombok/tags",
        ]
    );
    println!(
        "java 6.8.27 jdtls: denied Java version probe with cached JDTLS 1.61.0/Lombok 1.18.48 and check_updates=never"
    );
    drop(extension);
    let java_home = PathBuf::from(
        std::env::var_os("GROOVY_JAVA_HOME")
            .expect("set GROOVY_JAVA_HOME to a compatible explicit JDK 21+ installation"),
    );
    let java_executable = java_home.join("bin/java");
    assert!(java_executable.is_file(), "explicit fixture JDK must exist");
    fs::copy_recursive(
        fs.as_ref(),
        &java_cache.join("debugger"),
        &work_tree.path().join("java/debugger"),
        fs::CopyOptions::default(),
    )
    .await
    .expect("copy installed Debug cache without executing it");
    let metadata_requests = requests.lock().clone();
    for allow_probe in [true, false] {
        let capabilities = if allow_probe {
            vec![json!({
                "kind": "process:exec",
                "command": java_executable,
                "args": ["-version"],
            })]
        } else {
            Vec::new()
        };
        let settings = json!({
            "granted_extension_capabilities": capabilities,
            "auto_install_extensions": { "java": true, "kotlin": true },
            "lsp": { "jdtls": { "settings": {
                "check_updates": "never",
                "java_home": java_home,
            } } },
        });
        cx.update(|cx| {
            cx.update_global::<SettingsStore, _>(|store, cx| {
                store
                    .set_user_settings(&settings.to_string(), cx)
                    .expect("select an explicit JDK and only its exact version-probe capability");
            });
        });
        if allow_probe {
            let extension =
                WasmExtension::load(&directory, &manifest, host.clone(), &cx.to_async())
                    .await
                    .expect("reload Java on the retained host after changing settings");
            let command = extension.language_server_command(
                LanguageServerName::from("jdtls"),
                LanguageName::new("Java"),
                Arc::new(DeniedWorktree),
            );
            let error = match futures::future::select(
                Box::pin(command),
                cx.executor().timer(std::time::Duration::from_secs(30)),
            )
            .await
            {
                futures::future::Either::Left((result, _)) => {
                    result.expect_err("extension reload must retain the host's startup grants")
                }
                futures::future::Either::Right(_) => panic!("retained Java host probe timed out"),
            };
            let reason = format!("{error:#}");
            assert!(
                reason.contains("Failed to determine Java version"),
                "{reason}"
            );
            assert!(
                reason.contains("not granted by the extension host"),
                "{reason}"
            );
            assert!(
                reason.contains(
                    java_executable
                        .to_str()
                        .expect("fixture JDK path must be UTF-8")
                ),
                "{reason}"
            );
            assert!(host.granted_capabilities.is_empty());
            assert_eq!(*requests.lock(), metadata_requests);
            println!(
                "java 6.8.27 jdtls: reloaded extension on retained host still denies changed probe grant until host restart"
            );
        }
        let host = cx.update(|cx| {
            WasmHost::new(
                fs.clone(),
                http_client.clone(),
                NodeRuntime::unavailable(),
                Arc::new(ExtensionHostProxy::default()),
                work_tree.path().to_path_buf(),
                cx,
            )
        });
        assert_eq!(host.granted_capabilities.len(), usize::from(allow_probe));
        let extension = WasmExtension::load(&directory, &manifest, host, &cx.to_async())
            .await
            .expect("reload Java through the existing host with current capabilities");
        let command = extension.language_server_command(
            LanguageServerName::from("jdtls"),
            LanguageName::new("Java"),
            Arc::new(DeniedWorktree),
        );
        let result = match futures::future::select(
            Box::pin(command),
            cx.executor().timer(std::time::Duration::from_secs(30)),
        )
        .await
        {
            futures::future::Either::Left((result, _)) => result,
            futures::future::Either::Right(_) => panic!("Java capability transition timed out"),
        };
        if allow_probe {
            let command =
                result.expect("exact version-probe grant must resolve the cached command");
            assert_eq!(command.command, Path::new("bin/v6.8.27/java-lsp-proxy"));
            assert!(extension.work_dir.join(&command.command).is_file());
            assert!(
                command
                    .args
                    .iter()
                    .any(|argument| Path::new(argument) == java_executable),
                "configured host JDK must override the worktree's /usr candidate"
            );
            assert!(
                command.env.contains(&(
                    "JAVA_HOME".into(),
                    java_home
                        .to_str()
                        .expect("fixture JDK path must be UTF-8")
                        .into()
                ))
            );
            println!(
                "java 6.8.27 jdtls: exact JDK version-probe grant resolves cached launch command without launching it"
            );
            let workspace_name = "jdtls-81beaadd29cf3cb2b3af807a80f872ab2347c443";
            let workspace_argument = |command: &extension::Command| {
                let mut values = command.args.windows(2).filter_map(|arguments| {
                    (arguments.first().map(String::as_str) == Some("-data"))
                        .then(|| arguments.get(1))
                        .flatten()
                });
                let path = values.next().expect("JDTLS workspace argument").clone();
                assert!(values.next().is_none(), "one JDTLS workspace argument");
                PathBuf::from(path)
            };
            let expected_parent = if product_flavor::DEFAULT_LANGUAGE_SERVERS.contains(&"jdtls") {
                extension.work_dir.join("jdtls-workspaces")
            } else {
                PathBuf::from("/capability-fixture/Library/Caches")
            };
            assert_eq!(
                workspace_argument(&command),
                expected_parent.join(workspace_name)
            );

            let user_parent = work_tree.path().join("explicit-user-workspaces");
            let mut user_settings = settings.clone();
            user_settings
                .pointer_mut("/lsp/jdtls/settings")
                .and_then(serde_json::Value::as_object_mut)
                .expect("fixture JDTLS settings object")
                .insert("data_directory".into(), json!(user_parent));
            cx.update(|cx| {
                cx.update_global::<SettingsStore, _>(|store, cx| {
                    store
                        .set_user_settings(&user_settings.to_string(), cx)
                        .expect("explicit JDTLS user workspace override");
                });
            });
            let user_command = cached_jdtls_command(&extension, cx).await;
            assert_eq!(
                workspace_argument(&user_command),
                user_parent.join(workspace_name)
            );

            let project_parent = work_tree.path().join("explicit-project-workspaces");
            let project_settings =
                json!({"lsp":{"jdtls":{"settings":{"data_directory":project_parent}}}}).to_string();
            cx.update(|cx| {
                cx.update_global::<SettingsStore, _>(|store, cx| {
                    store
                        .set_local_settings(
                            settings::WorktreeId::from_proto(1),
                            settings::LocalSettingsPath::InWorktree(
                                util::rel_path::rel_path("").into(),
                            ),
                            settings::LocalSettingsKind::Settings,
                            Some(&project_settings),
                            cx,
                        )
                        .expect("explicit project workspace override");
                });
            });
            let project_command = cached_jdtls_command(&extension, cx).await;
            assert_eq!(
                workspace_argument(&project_command),
                project_parent.join(workspace_name)
            );
            cx.update(|cx| {
                cx.update_global::<SettingsStore, _>(|store, cx| {
                    store
                        .set_local_settings(
                            settings::WorktreeId::from_proto(1),
                            settings::LocalSettingsPath::InWorktree(
                                util::rel_path::rel_path("").into(),
                            ),
                            settings::LocalSettingsKind::Settings,
                            None,
                            cx,
                        )
                        .expect("remove only the fixture project override");
                    store
                        .set_user_settings(&settings.to_string(), cx)
                        .expect("restore fixture user settings");
                });
            });
            let restored_command = cached_jdtls_command(&extension, cx).await;
            assert_eq!(
                workspace_argument(&restored_command),
                expected_parent.join(workspace_name)
            );
            assert_eq!(*requests.lock(), metadata_requests);
            println!(
                "java 6.8.27 jdtls: product={} default-parent={} user/project override precedence and default restoration verified without server launch",
                product_flavor::ID,
                expected_parent.display(),
            );
        } else {
            let reason = format!(
                "{:#}",
                result.expect_err("fresh host must deny the revoked probe")
            );
            assert!(
                reason.contains("Failed to determine Java version"),
                "{reason}"
            );
            assert!(
                reason.contains("not granted by the extension host"),
                "{reason}"
            );
            assert!(
                reason.contains(
                    java_executable
                        .to_str()
                        .expect("fixture JDK path must be UTF-8")
                ),
                "{reason}"
            );
            println!(
                "java 6.8.27 jdtls: fresh host denies the explicit JDK probe after grant removal"
            );
        }
        assert_eq!(*requests.lock(), metadata_requests);
    }
    cx.read(|cx| {
        let settings = crate::ExtensionSettings::get_global(cx);
        for extension_id in ["java", "kotlin"] {
            assert_eq!(
                settings.auto_install_extensions.get(extension_id),
                Some(&true)
            );
            assert!(
                !settings
                    .auto_install_extension_failures
                    .contains_key(extension_id)
            );
        }
    });
    drop(project);
}

#[gpui::test]
async fn disconnected_remote_does_not_block_extension_sync_for_new_remote(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    init_test(cx);
    let (old_options, old_server_session, old_guard) = RemoteClient::fake_server(cx, server_cx);
    let ping_handler = server_cx.new(|_| ());
    old_server_session.add_request_handler::<rpc::proto::Ping, _, _, _>(
        ping_handler.downgrade(),
        |_entity, _envelope, _cx| async { Ok(rpc::proto::Ack {}) },
    );
    drop(old_guard);
    let old_client = RemoteClient::connect_mock(old_options, cx).await;

    let (new_options, new_server_session, new_guard) = RemoteClient::fake_server(cx, server_cx);
    new_server_session.add_request_handler::<rpc::proto::Ping, _, _, _>(
        ping_handler.downgrade(),
        |_entity, _envelope, _cx| async { Ok(rpc::proto::Ack {}) },
    );
    drop(new_guard);
    let new_client = RemoteClient::connect_mock(new_options, cx).await;

    old_client.update(cx, |client, cx| client.force_server_not_running(cx));

    let connected = cx.update(|cx| {
        assert!(old_client.read(cx).connection().is_none());
        assert!(new_client.read(cx).connection().is_some());
        crate::connected_remote_clients(&[old_client.downgrade(), new_client.downgrade()], cx)
    });
    assert_eq!(connected.len(), 1);
    assert_eq!(
        connected
            .first()
            .and_then(|client| client.upgrade())
            .map(|client| client.entity_id()),
        Some(new_client.entity_id())
    );
}

#[gpui::test]
async fn product_default_auto_install_skips_in_flight_operations_and_stale_failures(
    cx: &mut TestAppContext,
) {
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree("/default-extensions", json!({"installed": {}}))
        .await;
    let proxy = Arc::new(ExtensionHostProxy::new());
    theme_extension::init(
        proxy.clone(),
        Arc::new(ThemeRegistry::new(Box::new(()))),
        cx.executor(),
    );
    language_extension::init(
        LspAccess::Noop,
        proxy.clone(),
        Arc::new(LanguageRegistry::test(cx.executor())),
    );
    let http_client = FakeHttpClient::with_200_response();
    let store = cx.new(|cx| {
        ExtensionStore::new(
            PathBuf::from("/default-extensions"),
            None,
            proxy,
            fs,
            http_client.clone(),
            http_client,
            None,
            NodeRuntime::unavailable(),
            Vec::new(),
            cx,
        )
    });
    cx.executor().advance_clock(RELOAD_DEBOUNCE_DURATION);
    cx.executor().run_until_parked();
    let mut settings = crate::extension_settings::ExtensionSettings::default();
    settings.auto_install_extensions.insert("java".into(), true);
    settings.auto_install_extensions.insert("html".into(), true);
    cx.update(|cx| {
        assert!(store.read(cx).should_start_auto_install("java", &settings));
    });

    store.update(cx, |store, _| {
        store
            .outstanding_operations
            .insert("java".into(), crate::ExtensionOperation::Remove);
        assert!(!store.should_start_auto_install("java", &settings));
        assert!(store.should_start_auto_install("html", &settings));
        store.outstanding_operations.remove("java");
        assert!(store.should_start_auto_install("java", &settings));
        store.pending_product_default_installs.insert("java".into());
        assert!(!store.should_start_auto_install("java", &settings));
        store.pending_product_default_installs.remove("java");
        settings
            .auto_install_extensions
            .insert("java".into(), false);
        assert!(!store.should_start_auto_install("java", &settings));
    });

    let (immediate_retry, overlapping_retry) = store.update(cx, |store, cx| {
        store
            .outstanding_operations
            .insert("java".into(), crate::ExtensionOperation::Remove);
        let immediate_retry = store.install_product_default_extension("java".into(), cx);
        let retry = store.start_product_default_install("java".into(), cx);
        assert!(!store.pending_product_default_installs.contains("java"));
        store.outstanding_operations.remove("java");
        (immediate_retry, retry)
    });
    let immediate_error = immediate_retry
        .await
        .expect_err("a busy extension must not accept a new default install");
    assert!(
        immediate_error
            .to_string()
            .contains("another extension operation")
    );
    let error = overlapping_retry
        .await
        .expect_err("removal must not be treated as a successful install");
    assert!(error.to_string().contains("another extension operation"));

    let duplicate_retry = store.update(cx, |store, cx| {
        store.pending_product_default_installs.insert("java".into());
        let retry = store.install_product_default_extension("java".into(), cx);
        assert!(store.product_default_install_in_progress("java"));
        store.pending_product_default_installs.remove("java");
        retry
    });
    let error = duplicate_retry
        .await
        .expect_err("a reserved product-default retry must not report success");
    assert!(error.to_string().contains("another extension operation"));

    for extension_id in ["java", "html"] {
        let (busy_install, busy_removal) = store.update(cx, |store, cx| {
            store
                .outstanding_operations
                .insert(extension_id.into(), crate::ExtensionOperation::Install);
            let install = store.install_latest_extension(extension_id.into(), cx);
            let removal = store.uninstall_extension(extension_id.into(), cx);
            store.outstanding_operations.remove(extension_id);
            (install, removal)
        });
        let install_error = busy_install
            .await
            .expect_err("a second install must not report success while one is running");
        let removal_error = busy_removal
            .await
            .expect_err("removal must not report success while installation is running");
        assert!(
            install_error
                .to_string()
                .contains("another extension operation")
        );
        assert!(
            removal_error
                .to_string()
                .contains("another extension operation")
        );
    }

    store.update(cx, |store, _| {
        let failed_java_install = store.product_default_retry_generation("java");
        let copper_html_install = store.product_default_retry_generation("html");
        store
            .reserve_product_default_install("java".into())
            .expect("a finished Java install can be retried");
        assert!(!store.auto_install_result_is_current("java", failed_java_install));
        assert!(store.auto_install_result_is_current("html", copper_html_install));
        assert!(
            store
                .reserve_product_default_install("java".into())
                .is_err()
        );
        assert_eq!(
            store.product_default_retry_generation("java"),
            failed_java_install + 1
        );

        store.pending_product_default_installs.remove("java");
        let failed_retry = store.product_default_retry_generation("java");
        store
            .reserve_product_default_install("java".into())
            .expect("a later Java retry can be reserved");
        assert!(!store.auto_install_result_is_current("java", failed_retry));
        assert!(store.auto_install_result_is_current("html", copper_html_install));
        store.pending_product_default_installs.remove("java");
    });

    let mut disabled = crate::extension_settings::ExtensionSettings::default();
    disabled
        .auto_install_extensions
        .insert("java".into(), false);
    cx.update(|cx| crate::extension_settings::ExtensionSettings::override_global(disabled, cx));
    let cancelled_retry = store.update(cx, |store, cx| {
        store.start_product_default_install("java".into(), cx)
    });
    let error = cancelled_retry
        .await
        .expect_err("a disabled default must not report a successful retry");
    assert!(
        error
            .to_string()
            .contains("unavailable for automatic installation")
    );
    store.read_with(cx, |store, _| {
        assert!(!store.outstanding_operations.contains_key("java"));
    });
}

#[gpui::test]
async fn failed_product_default_install_stays_retryable_after_store_recreation(
    cx: &mut TestAppContext,
) {
    init_test(cx);
    cx.executor().allow_parking();
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree("/failed-default-extensions", json!({"installed": {}}))
        .await;
    let proxy = Arc::new(ExtensionHostProxy::new());
    theme_extension::init(
        proxy.clone(),
        Arc::new(ThemeRegistry::new(Box::new(()))),
        cx.executor(),
    );
    language_extension::init(
        LspAccess::Noop,
        proxy.clone(),
        Arc::new(LanguageRegistry::test(cx.executor())),
    );
    let requests = Arc::new(Mutex::new(Vec::new()));
    let http_client = FakeHttpClient::create({
        let requests = requests.clone();
        move |request| {
            let requests = requests.clone();
            async move {
                requests.lock().push(request.uri().path().to_string());
                Err::<Response<http_client::AsyncBody>, _>(anyhow::anyhow!("offline fixture"))
            }
        }
    });
    let store = cx.new(|cx| {
        ExtensionStore::new(
            PathBuf::from("/failed-default-extensions"),
            None,
            proxy.clone(),
            fs.clone(),
            http_client.clone(),
            http_client.clone(),
            None,
            NodeRuntime::unavailable(),
            Vec::new(),
            cx,
        )
    });
    cx.executor().advance_clock(RELOAD_DEBOUNCE_DURATION);
    cx.executor().run_until_parked();

    let install = store.update(cx, |store, cx| {
        store.install_product_default_extension("java".into(), cx)
    });
    let failure = install.await.expect_err("offline download must fail");
    assert!(format!("{failure:#}").contains("offline fixture"));
    let java_download_count = || {
        requests
            .lock()
            .iter()
            .filter(|path| path.as_str() == "/extensions/java/download")
            .count()
    };
    assert_eq!(java_download_count(), 1);
    let first_reason = cx.update(|cx| {
        let settings = crate::extension_settings::ExtensionSettings::get_global(cx);
        assert_eq!(settings.auto_install_extensions.get("java"), Some(&true));
        settings
            .auto_install_extension_failures
            .get("java")
            .cloned()
            .expect("the failed default retains a retry reason")
    });
    assert!(first_reason.contains("offline fixture"));

    let persisted_settings = fs
        .load(paths::settings_file())
        .await
        .expect("failed default saved to the product settings file");
    drop(store);
    cx.update(|cx| {
        let mut settings_store = SettingsStore::test(cx);
        settings_store
            .set_user_settings(&persisted_settings, cx)
            .expect("restore saved default-install failure");
        cx.set_global(settings_store);
        let settings = crate::extension_settings::ExtensionSettings::get_global(cx);
        assert_eq!(
            settings.auto_install_extension_failures.get("java"),
            Some(&first_reason)
        );
    });
    let relaunched = cx.new(|cx| {
        ExtensionStore::new(
            PathBuf::from("/failed-default-extensions"),
            None,
            proxy,
            fs.clone(),
            http_client.clone(),
            http_client,
            None,
            NodeRuntime::unavailable(),
            Vec::new(),
            cx,
        )
    });
    cx.executor().advance_clock(RELOAD_DEBOUNCE_DURATION);
    cx.executor().run_until_parked();
    relaunched.read_with(cx, |store, cx| {
        assert!(!store.should_start_auto_install(
            "java",
            crate::extension_settings::ExtensionSettings::get_global(cx)
        ));
    });
    assert_eq!(java_download_count(), 1);
    let retry = relaunched.update(cx, |store, cx| {
        store.install_product_default_extension("java".into(), cx)
    });
    retry
        .await
        .expect_err("retry still fails while the network is offline");
    assert_eq!(java_download_count(), 2);
    cx.update(|cx| {
        let settings = crate::extension_settings::ExtensionSettings::get_global(cx);
        assert_eq!(settings.auto_install_extensions.get("java"), Some(&true));
        assert!(
            settings
                .auto_install_extension_failures
                .contains_key("java")
        );
    });

    let decline = cx.update(|cx| {
        settings::update_settings_file_with_completion(fs.clone(), cx, |settings, _| {
            settings.extension.decline_auto_install("java".into());
        })
    });
    decline
        .await
        .expect("decline settings writer")
        .expect("persist declined Java default");
    let persisted_decline = fs
        .load(paths::settings_file())
        .await
        .expect("declined default saved to product settings");
    cx.update(|cx| {
        let mut settings_store = SettingsStore::test(cx);
        settings_store
            .set_user_settings(&persisted_decline, cx)
            .expect("restore declined Java default");
        cx.set_global(settings_store);
        let settings = crate::extension_settings::ExtensionSettings::get_global(cx);
        assert_eq!(settings.auto_install_extensions.get("java"), Some(&false));
        assert!(
            !settings
                .auto_install_extension_failures
                .contains_key("java")
        );
    });
    relaunched.read_with(cx, |store, cx| {
        assert!(!store.should_start_auto_install(
            "java",
            crate::extension_settings::ExtensionSettings::get_global(cx)
        ));
    });
    assert_eq!(java_download_count(), 2);
}

#[gpui::test]
async fn product_default_download_rejects_removal_and_preserves_a_later_decline(
    cx: &mut TestAppContext,
) {
    init_test(cx);
    cx.executor().allow_parking();
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree("/held-default-extensions", json!({"installed": {}}))
        .await;
    let proxy = Arc::new(ExtensionHostProxy::new());
    theme_extension::init(
        proxy.clone(),
        Arc::new(ThemeRegistry::new(Box::new(()))),
        cx.executor(),
    );
    language_extension::init(
        LspAccess::Noop,
        proxy.clone(),
        Arc::new(LanguageRegistry::test(cx.executor())),
    );
    let (download_started, download_started_receiver) = futures::channel::oneshot::channel();
    let (release_download, download_release_receiver) = futures::channel::oneshot::channel();
    let download_gate = Arc::new(Mutex::new(Some((
        download_started,
        download_release_receiver,
    ))));
    let requested_paths = Arc::new(Mutex::new(Vec::new()));
    let http_client = FakeHttpClient::create({
        let download_gate = download_gate.clone();
        let requested_paths = requested_paths.clone();
        move |request| {
            let path = request.uri().path().to_string();
            requested_paths.lock().push(path.clone());
            let gate = (path == "/extensions/java/download")
                .then(|| download_gate.lock().take())
                .flatten();
            async move {
                anyhow::ensure!(
                    path == "/extensions/java/download",
                    "unexpected fixture endpoint"
                );
                let (started, release) =
                    gate.ok_or_else(|| anyhow::anyhow!("unexpected second Java download"))?;
                started
                    .send(())
                    .map_err(|_| anyhow::anyhow!("download start receiver closed"))?;
                release.await?;
                Err::<Response<http_client::AsyncBody>, _>(anyhow::anyhow!(
                    "offline after download release"
                ))
            }
        }
    });
    let store = cx.new(|cx| {
        ExtensionStore::new(
            PathBuf::from("/held-default-extensions"),
            None,
            proxy,
            fs.clone(),
            http_client.clone(),
            http_client,
            None,
            NodeRuntime::unavailable(),
            Vec::new(),
            cx,
        )
    });
    cx.executor().advance_clock(RELOAD_DEBOUNCE_DURATION);
    cx.executor().run_until_parked();

    let install = store.update(cx, |store, cx| {
        store.install_product_default_extension("java".into(), cx)
    });
    let deadline = cx.executor().timer(std::time::Duration::from_secs(5));
    assert!(matches!(
        futures::future::select(download_started_receiver, deadline).await,
        futures::future::Either::Left((Ok(()), _))
    ));
    let removal = store.update(cx, |store, cx| store.uninstall_extension("java".into(), cx));
    let removal_error = removal
        .await
        .expect_err("removal must wait for the running default download");
    assert!(
        removal_error
            .to_string()
            .contains("another extension operation")
    );

    let decline = cx.update(|cx| {
        settings::update_settings_file_with_completion(fs.clone(), cx, |settings, _| {
            settings.extension.decline_auto_install("java".into());
        })
    });
    decline
        .await
        .expect("decline settings writer")
        .expect("persist declined Java default");
    release_download
        .send(())
        .expect("release held Java download");
    let install_error = install.await.expect_err("held download must fail offline");
    assert!(format!("{install_error:#}").contains("offline after download release"));
    assert_eq!(
        requested_paths
            .lock()
            .iter()
            .filter(|path| path.as_str() == "/extensions/java/download")
            .count(),
        1
    );
    cx.update(|cx| {
        let settings = crate::extension_settings::ExtensionSettings::get_global(cx);
        assert_eq!(settings.auto_install_extensions.get("java"), Some(&false));
        assert!(
            !settings
                .auto_install_extension_failures
                .contains_key("java")
        );
        assert!(!store.read(cx).product_default_install_in_progress("java"));
    });
    let persisted_settings = fs
        .load(paths::settings_file())
        .await
        .expect("decline saved after the late download failure");
    cx.update(|cx| {
        let mut settings_store = SettingsStore::test(cx);
        settings_store
            .set_user_settings(&persisted_settings, cx)
            .expect("restore declined Java default after download failure");
        cx.set_global(settings_store);
        let settings = crate::extension_settings::ExtensionSettings::get_global(cx);
        assert_eq!(settings.auto_install_extensions.get("java"), Some(&false));
        assert!(
            !settings
                .auto_install_extension_failures
                .contains_key("java")
        );
    });
}

#[gpui::test]
async fn versioned_extension_install_and_upgrade_report_invalid_endpoint(cx: &mut TestAppContext) {
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree("/versioned-extensions", json!({"installed": {}}))
        .await;
    let proxy = Arc::new(ExtensionHostProxy::new());
    theme_extension::init(
        proxy.clone(),
        Arc::new(ThemeRegistry::new(Box::new(()))),
        cx.executor(),
    );
    language_extension::init(
        LspAccess::Noop,
        proxy.clone(),
        Arc::new(LanguageRegistry::test(cx.executor())),
    );
    let http_client = FakeHttpClient::with_200_response();
    http_client.set_base_url("invalid endpoint");
    let store = cx.new(|cx| {
        ExtensionStore::new(
            PathBuf::from("/versioned-extensions"),
            None,
            proxy,
            fs,
            http_client.clone(),
            http_client,
            None,
            NodeRuntime::unavailable(),
            Vec::new(),
            cx,
        )
    });
    cx.executor().advance_clock(RELOAD_DEBOUNCE_DURATION);
    cx.executor().run_until_parked();

    for operation in [
        crate::ExtensionOperation::Install,
        crate::ExtensionOperation::Upgrade,
    ] {
        let result = store
            .update(cx, |store, cx| {
                store.install_or_upgrade_extension("java".into(), "6.8.27".into(), operation, cx)
            })
            .await;
        let error = result.expect_err("invalid endpoint must fail the operation");
        assert!(error.to_string().contains("relative URL"));
    }
    store.read_with(cx, |store, _| {
        assert!(store.outstanding_operations.is_empty());
        assert!(!store.extension_index.extensions.contains_key("java"));
    });
}

#[gpui::test]
async fn agent_plugins_install_is_owned_and_rejects_duplicates(cx: &mut TestAppContext) {
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree("/plugin-extensions", json!({"installed":{}}))
        .await;
    fs.insert_tree("/plugin-source", json!({"plugin.json":json!({"$schema":crate::agent_plugins::MANIFEST_SCHEMA,"name":"owned"}).to_string(), "mcp.json":json!({"$schema":crate::agent_plugins::MCP_SCHEMA,"mcpServers":{"local":{"type":"stdio","command":"node"}}}).to_string()})).await;
    let snapshot = crate::agent_plugins::capture_package(
        fs.as_ref(),
        Path::new("/plugin-source"),
        crate::agent_plugins::SnapshotLimits::default(),
    )
    .await
    .expect("inspect");
    let proxy = Arc::new(ExtensionHostProxy::new());
    theme_extension::init(
        proxy.clone(),
        Arc::new(ThemeRegistry::new(Box::new(()))),
        cx.executor(),
    );
    language_extension::init(
        LspAccess::Noop,
        proxy.clone(),
        Arc::new(LanguageRegistry::test(cx.executor())),
    );
    let http_client = FakeHttpClient::with_200_response();
    let store = cx.new(|cx| {
        ExtensionStore::new(
            PathBuf::from("/plugin-extensions"),
            None,
            proxy,
            fs.clone(),
            http_client.clone(),
            http_client,
            None,
            NodeRuntime::unavailable(),
            Vec::new(),
            cx,
        )
    });
    cx.executor().advance_clock(RELOAD_DEBOUNCE_DURATION);
    cx.executor().run_until_parked();
    let installation = store.update(cx, |store, cx| {
        store.install_agent_plugin(snapshot.clone(), cx)
    });
    let duplicate = store.update(cx, |store, cx| {
        store.install_agent_plugin(snapshot.clone(), cx)
    });
    assert!(duplicate.await.is_err());
    drop(installation);
    cx.executor().run_until_parked();
    cx.update(|cx| {
        assert_eq!(store.read(cx).agent_plugins.len(), 1);
        assert!(!store.read(cx).agent_plugins_busy);
        assert!(
            store
                .read(cx)
                .agent_plugins
                .values()
                .all(|record| record.approved_servers.is_empty())
        );
    });
    let duplicate = store.update(cx, |store, cx| store.install_agent_plugin(snapshot, cx));
    assert!(
        duplicate
            .await
            .expect_err("duplicate install")
            .to_string()
            .contains("already installed")
    );
    let persisted = crate::agent_plugins::load_inventory(
        fs.as_ref(),
        Path::new("/plugin-extensions/agent-plugins"),
    )
    .await
    .expect("persisted inventory");
    assert_eq!(persisted.len(), 1);
    use crate::agent_plugins::{PackageChange, SnapshotLimits, capture_package};
    let record = persisted.values().next().expect("record").clone();
    use project::context_server_store::registry::ContextServerDescriptor as _;
    let settings = crate::agent_plugins::prepare_server_settings(
        fs.as_ref(),
        Path::new("/plugin-extensions/agent-plugins"),
        &record,
        "local",
    )
    .await
    .expect("prepare without executing");
    let descriptor = crate::agent_plugins::PackageServerDescriptor {
        owner: store.downgrade(),
        package_id: record.id.clone(),
        revision: record.revision.clone(),
        server_name: "local".into(),
        approval_token: None,
        settings: settings.clone(),
    };
    cx.update(|cx| assert!(descriptor.authorize_settings(&settings, cx).is_err()));
    let registry =
        cx.new(|_| project::context_server_store::registry::ContextServerDescriptorRegistry::new());
    let descriptor = Arc::new(descriptor);
    registry.update(cx, |registry, cx| {
        registry.register_context_server_descriptor(
            "agent-plugin/test".into(),
            descriptor.clone(),
            cx,
        )
    });
    let worktrees = cx.new(|cx| {
        project::worktree_store::WorktreeStore::local(
            true,
            fs.clone(),
            project::worktree_store::WorktreeIdCounter::get(cx),
        )
    });
    let resolve = |settings: project::project_settings::ContextServerSettings,
                   cx: &mut TestAppContext| {
        let registry = registry.clone();
        let worktrees = worktrees.clone();
        cx.update(|cx| {
            cx.spawn(async move |cx| {
                project::context_server_store::ContextServerConfiguration::from_settings(
                    settings,
                    context_server::ContextServerId("agent-plugin/test".into()),
                    registry,
                    worktrees,
                    cx,
                )
                .await
            })
        })
    };
    assert!(resolve(settings.clone(), cx).await.is_none());
    if let project::project_settings::ContextServerSettings::Stdio { command, .. } = &settings {
        assert_eq!(
            command.cwd.as_deref(),
            Some(
                record
                    .snapshot_path(Path::new("/plugin-extensions/agent-plugins"))
                    .as_path()
            )
        );
        assert!(
            command
                .env
                .as_ref()
                .expect("reserved environment")
                .contains_key("PLUGIN_DATA")
        );
    } else {
        panic!("stdio settings");
    }
    store
        .update(cx, |store, cx| {
            store.change_agent_plugin(
                &record.id,
                PackageChange::ApproveServer {
                    name: "local".into(),
                    revision: record.revision.clone(),
                },
                cx,
            )
        })
        .await
        .expect("explicit approval");
    let descriptor = Arc::new(crate::agent_plugins::PackageServerDescriptor {
        approval_token: store.read_with(cx, |store, _| {
            store
                .agent_plugins
                .get(&record.id)
                .expect("installed")
                .approval_tokens
                .get("local")
                .cloned()
        }),
        ..descriptor.as_ref().clone()
    });
    registry.update(cx, |registry, cx| {
        registry.register_context_server_descriptor(
            "agent-plugin/test".into(),
            descriptor.clone(),
            cx,
        )
    });
    cx.update(|cx| assert!(descriptor.authorize_settings(&settings, cx).is_ok()));
    assert!(resolve(settings.clone(), cx).await.is_some());
    let mut changed_settings = settings.clone();
    if let project::project_settings::ContextServerSettings::Stdio { command, .. } =
        &mut changed_settings
    {
        command.args.push("changed".into());
    }
    cx.update(|cx| {
        assert!(
            descriptor
                .authorize_settings(&changed_settings, cx)
                .is_err()
        )
    });
    store
        .update(cx, |store, cx| {
            store.change_agent_plugin(
                &record.id,
                PackageChange::ApproveServerConfiguration {
                    name: "local".into(),
                    revision: record.revision.clone(),
                    settings: changed_settings.clone(),
                },
                cx,
            )
        })
        .await
        .expect("explicit changed configuration approval");
    cx.update(|cx| {
        assert!(
            descriptor
                .authorize_settings(&changed_settings, cx)
                .is_err()
        )
    });
    let descriptor = Arc::new(crate::agent_plugins::PackageServerDescriptor {
        approval_token: store.read_with(cx, |store, _| {
            store
                .agent_plugins
                .get(&record.id)
                .expect("installed")
                .approval_tokens
                .get("local")
                .cloned()
        }),
        ..descriptor.as_ref().clone()
    });
    registry.update(cx, |registry, cx| {
        registry.register_context_server_descriptor(
            "agent-plugin/test".into(),
            descriptor.clone(),
            cx,
        )
    });
    cx.update(|cx| {
        assert!(descriptor.authorize_settings(&changed_settings, cx).is_ok());
        assert!(descriptor.authorize_settings(&settings, cx).is_err());
    });
    cx.update(|cx| {
        assert!(
            store
                .read(cx)
                .agent_plugins
                .get(&record.id)
                .expect("installed")
                .approved_servers
                .contains("local")
        )
    });
    fs.insert_tree(
        format!("/plugin-extensions/agent-plugins/data/{}", record.id),
        json!({"keep":"user data"}),
    )
    .await;
    fs.insert_tree("/shared-credentials", json!({"account":"untouched"}))
        .await;
    fs.write(Path::new("/plugin-source/new-asset"), b"updated")
        .await
        .expect("source update");
    let updated = capture_package(
        fs.as_ref(),
        Path::new("/plugin-source"),
        SnapshotLimits::default(),
    )
    .await
    .expect("inspect update");
    store
        .update(cx, |store, cx| {
            store.change_agent_plugin(&record.id, PackageChange::Update(updated.clone()), cx)
        })
        .await
        .expect("explicit update");
    cx.update(|cx| {
        assert!(
            store
                .read(cx)
                .agent_plugins
                .get(&record.id)
                .expect("updated")
                .approved_servers
                .is_empty()
        )
    });
    assert!(
        store
            .update(cx, |store, cx| store.change_agent_plugin(
                &record.id,
                PackageChange::ApproveServer {
                    name: "local".into(),
                    revision: record.revision.clone()
                },
                cx
            ))
            .await
            .is_err()
    );
    store
        .update(cx, |store, cx| {
            store.change_agent_plugin(
                &record.id,
                PackageChange::Uninstall { delete_data: false },
                cx,
            )
        })
        .await
        .expect("uninstall retains data");
    let data_path = PathBuf::from(format!(
        "/plugin-extensions/agent-plugins/data/{}/keep",
        record.id
    ));
    assert_eq!(
        fs.load(&data_path).await.expect("retained data"),
        "user data"
    );
    assert!(
        crate::agent_plugins::load_inventory(
            fs.as_ref(),
            Path::new("/plugin-extensions/agent-plugins")
        )
        .await
        .expect("inventory")
        .is_empty()
    );
    store
        .update(cx, |store, cx| store.install_agent_plugin(updated, cx))
        .await
        .expect("reinstall");
    store
        .update(cx, |store, cx| {
            store.change_agent_plugin(
                &record.id,
                PackageChange::Uninstall { delete_data: true },
                cx,
            )
        })
        .await
        .expect("explicit data deletion");
    assert!(!fs.is_file(&data_path).await);
    assert_eq!(
        fs.load(Path::new("/shared-credentials/account"))
            .await
            .expect("shared account"),
        "untouched"
    );
}

#[cfg(feature = "agentic")]
#[gpui::test]
async fn agent_plugins_publish_retire_and_recover_contributions(cx: &mut TestAppContext) {
    use crate::agent_plugins::{
        MANIFEST_SCHEMA, MCP_SCHEMA, PackageChange, capture_package, load_inventory,
    };
    use agent_skills::SkillIndex;
    use project::context_server_store::registry::ContextServerDescriptorRegistry;
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree("/published", json!({"installed":{}})).await;
    fs.insert_tree("/portable", json!({
        "plugin.json": json!({"$schema":MANIFEST_SCHEMA,"name":"published"}).to_string(),
        "mcp.json": json!({"$schema":MCP_SCHEMA,"mcpServers":{"remote":{"type":"streamable-http","url":"https://example.com/mcp"}}}).to_string(),
        "skills": {
            "review":{"SKILL.md":"---\nname: review\ndescription: Review changes\n---\nReview this project."},
            "broken":{"SKILL.md":"invalid"}
        }
    })).await;
    let store = cx.new(|cx| {
        ExtensionStore::new(
            "/published".into(),
            None,
            Arc::new(ExtensionHostProxy::new()),
            fs.clone(),
            FakeHttpClient::with_200_response(),
            FakeHttpClient::with_200_response(),
            None,
            NodeRuntime::unavailable(),
            Vec::new(),
            cx,
        )
    });
    store.update(cx, |store, cx| store.init_agent_plugins(cx));
    cx.executor()
        .advance_clock(std::time::Duration::from_millis(200));
    cx.run_until_parked();
    let snapshot = capture_package(fs.as_ref(), Path::new("/portable"), Default::default())
        .await
        .expect("inspect");
    store
        .update(cx, |store, cx| store.install_agent_plugin(snapshot, cx))
        .await
        .expect("install");
    cx.run_until_parked();
    let registry = cx.update(ContextServerDescriptorRegistry::default_global);
    let record = store.read_with(cx, |store, _| {
        store
            .agent_plugins
            .values()
            .next()
            .expect("installed")
            .clone()
    });
    cx.update(|cx| {
        assert_eq!(cx.global::<SkillIndex>().package_skills.len(), 1);
        assert_eq!(
            cx.global::<SkillIndex>()
                .package_skills
                .first()
                .expect("published skill")
                .lookup_name(),
            "plugin/published:review"
        );
        assert!(registry.read(cx).context_server_descriptors().is_empty());
        assert!(
            store
                .read(cx)
                .agent_plugins_diagnostics
                .contains_key(&record.id)
        );
    });
    let approval = || PackageChange::ApproveServer {
        name: "remote".into(),
        revision: record.revision.clone(),
    };
    store
        .update(cx, |store, cx| {
            store.change_agent_plugin(&record.id, approval(), cx)
        })
        .await
        .expect("approve");
    cx.run_until_parked();
    let first = registry.read_with(cx, |registry, _| {
        registry
            .context_server_descriptors()
            .first()
            .expect("published server")
            .0
            .clone()
    });
    store
        .update(cx, |store, cx| {
            store.change_agent_plugin(&record.id, PackageChange::Enabled(false), cx)
        })
        .await
        .expect("disable");
    cx.run_until_parked();
    cx.update(|cx| {
        assert!(cx.global::<SkillIndex>().package_skills.is_empty());
        assert!(registry.read(cx).context_server_descriptors().is_empty());
    });
    store
        .update(cx, |store, cx| {
            store.change_agent_plugin(&record.id, PackageChange::Enabled(true), cx)
        })
        .await
        .expect("enable skills");
    store
        .update(cx, |store, cx| {
            store.change_agent_plugin(&record.id, approval(), cx)
        })
        .await
        .expect("reapprove");
    cx.run_until_parked();
    let current = registry.read_with(cx, |registry, _| {
        registry
            .context_server_descriptors()
            .first()
            .expect("republished server")
            .0
            .clone()
    });
    assert_ne!(first, current);
    let records = load_inventory(fs.as_ref(), Path::new("/published/agent-plugins"))
        .await
        .expect("restart inventory");
    assert_eq!(
        records
            .get(&record.id)
            .expect("persisted")
            .server_id("remote"),
        current.as_ref()
    );
    fs.write(
        &record
            .snapshot_path(Path::new("/published/agent-plugins"))
            .join("skills/review/SKILL.md"),
        b"changed",
    )
    .await
    .expect("simulate changed managed content");
    store
        .update(cx, |store, cx| {
            store.change_agent_plugin(&record.id, PackageChange::Enabled(true), cx)
        })
        .await
        .expect("refresh");
    cx.run_until_parked();
    cx.update(|cx| {
        assert!(cx.global::<SkillIndex>().package_skills.is_empty());
        assert!(registry.read(cx).context_server_descriptors().is_empty());
        assert!(
            store
                .read(cx)
                .agent_plugins_diagnostics
                .get(&record.id)
                .expect("diagnostics")
                .iter()
                .any(|error| error.contains("content changed"))
        );
    });
}

#[gpui::test]
async fn doctor_owner_repair_extension_retry_preserves_preferences(cx: &mut TestAppContext) {
    init_test(cx);
    cx.executor().allow_parking();
    let extensions = TempTree::new(json!({"installed": {}, "work": {}}));
    let extensions_directory = extensions
        .path()
        .canonicalize()
        .expect("Isolated extension directory");
    let filesystem = Arc::new(RealFs::new(None, cx.executor()));
    let proxy = Arc::new(ExtensionHostProxy::new());
    theme_extension::init(
        proxy.clone(),
        Arc::new(ThemeRegistry::new(Box::new(()))),
        cx.executor(),
    );
    language_extension::init(
        LspAccess::Noop,
        proxy.clone(),
        Arc::new(LanguageRegistry::test(cx.executor())),
    );
    let requests = Arc::new(Mutex::new(Vec::new()));
    let http = FakeHttpClient::create({
        let requests = requests.clone();
        move |request| {
            requests.lock().push(request.uri().path().to_owned());
            async move {
                if request.uri().path() == "/extensions" {
                    return anyhow::Ok(Response::new("[]".into()));
                }
                anyhow::ensure!(
                    request.uri().path() == "/extensions/java/download",
                    "Unexpected download endpoint"
                );
                let manifest = b"id = 'java'\nname = 'Doctor fixture'\nversion = '1.2.3'\nschema_version = 0\n";
                let mut bytes = Vec::new();
                let mut archive = async_tar::Builder::new(&mut bytes);
                let mut header = async_tar::Header::new_gnu();
                header.set_size(manifest.len() as u64);
                archive
                    .append_data(&mut header, "extension.toml", &manifest[..])
                    .await?;
                archive.into_inner().await?;
                let mut compressed = Vec::new();
                GzipEncoder::new(BufReader::new(bytes.as_slice()))
                    .read_to_end(&mut compressed)
                    .await?;
                anyhow::Ok(Response::new(compressed.into()))
            }
        }
    });
    let store = cx.new(|cx| {
        ExtensionStore::new(
            extensions_directory.clone(),
            None,
            proxy,
            filesystem.clone(),
            http.clone(),
            http,
            None,
            NodeRuntime::unavailable(),
            Vec::new(),
            cx,
        )
    });
    cx.executor().advance_clock(RELOAD_DEBOUNCE_DURATION);
    cx.run_until_parked();
    let mut preferences = crate::ExtensionSettings::default();
    preferences
        .auto_install_extensions
        .insert("java".into(), false);
    preferences
        .auto_install_extension_failures
        .insert("java".into(), "Previous failure".into());
    cx.update(|cx| crate::ExtensionSettings::override_global(preferences.clone(), cx));
    assert!(
        store
            .update(cx, |store, cx| store
                .retry_product_default_extension_from_defaults(
                    "java".into(),
                    &["java"],
                    cx
                ))
            .await
            .is_err()
    );
    assert!(
        !requests
            .lock()
            .iter()
            .any(|path| path == "/extensions/java/download")
    );
    preferences
        .auto_install_extensions
        .insert("java".into(), true);
    cx.update(|cx| crate::ExtensionSettings::override_global(preferences.clone(), cx));
    let revision = cx.update(|cx| cx.global::<SettingsStore>().observed_revision());
    let install = store.update(cx, |store, cx| {
        store.retry_product_default_extension_from_defaults("java".into(), &["java"], cx)
    });
    assert!(
        store
            .update(cx, |store, cx| store
                .retry_product_default_extension_from_defaults(
                    "java".into(),
                    &["java"],
                    cx
                ))
            .await
            .is_err()
    );
    install
        .await
        .expect("Existing extension owner installs fixture");
    cx.update(|cx| {
        assert_eq!(
            crate::ExtensionSettings::get_global(cx)
                .auto_install_extensions
                .get("java"),
            Some(&true)
        );
        assert_eq!(
            crate::ExtensionSettings::get_global(cx)
                .auto_install_extension_failures
                .get("java"),
            Some(&"Previous failure".to_owned())
        );
        assert_eq!(cx.global::<SettingsStore>().observed_revision(), revision);
    });
    store.read_with(cx, |store, cx| {
        assert_eq!(
            store
                .extension_manifest_for_id("java")
                .expect("Installed manifest")
                .version
                .as_ref(),
            "1.2.3"
        );
        assert!(!store.can_retry_product_default_extension_from_defaults("java", &["java"], cx));
        assert!(!store.can_retry_product_default_extension_from_defaults(
            "arbitrary",
            &["java"],
            cx
        ));
    });
    assert_eq!(
        requests
            .lock()
            .iter()
            .filter(|path| path.as_str() == "/extensions/java/download")
            .count(),
        1
    );
}

#[gpui::test]
async fn doctor_owner_repair_extension_cancel_preserves_later_decline(cx: &mut TestAppContext) {
    init_test(cx);
    let filesystem = FakeFs::new(cx.executor());
    filesystem
        .insert_tree("/doctor-held-extensions", json!({"installed": {}}))
        .await;
    let proxy = Arc::new(ExtensionHostProxy::new());
    theme_extension::init(
        proxy.clone(),
        Arc::new(ThemeRegistry::new(Box::new(()))),
        cx.executor(),
    );
    language_extension::init(
        LspAccess::Noop,
        proxy.clone(),
        Arc::new(LanguageRegistry::test(cx.executor())),
    );
    let (started, observed) = futures::channel::oneshot::channel();
    let started = Arc::new(Mutex::new(Some(started)));
    let http = FakeHttpClient::create(move |request| {
        let is_download = request.uri().path() == "/extensions/java/download";
        let started = is_download.then(|| started.lock().take()).flatten();
        async move {
            if request.uri().path() == "/extensions" {
                return anyhow::Ok(Response::new("[]".into()));
            }
            anyhow::ensure!(
                request.uri().path() == "/extensions/java/download",
                "Unexpected endpoint"
            );
            started
                .ok_or_else(|| anyhow::anyhow!("Duplicate request"))?
                .send(())
                .map_err(|_| anyhow::anyhow!("Start receiver dropped"))?;
            futures::future::pending::<anyhow::Result<Response<http_client::AsyncBody>>>().await
        }
    });
    let store = cx.new(|cx| {
        ExtensionStore::new(
            PathBuf::from("/doctor-held-extensions"),
            None,
            proxy,
            filesystem.clone(),
            http.clone(),
            http,
            None,
            NodeRuntime::unavailable(),
            Vec::new(),
            cx,
        )
    });
    cx.executor().advance_clock(RELOAD_DEBOUNCE_DURATION);
    cx.run_until_parked();
    let install = store.update(cx, |store, cx| {
        store.retry_product_default_extension_from_defaults("java".into(), &["java"], cx)
    });
    let deadline = cx.executor().timer(std::time::Duration::from_secs(5));
    assert!(matches!(
        futures::future::select(observed, deadline).await,
        futures::future::Either::Left((Ok(()), _))
    ));
    cx.update(|cx| {
        let mut preferences = crate::ExtensionSettings::get_global(cx).clone();
        preferences
            .auto_install_extensions
            .insert("java".into(), false);
        crate::ExtensionSettings::override_global(preferences, cx);
    });
    drop(install);
    cx.run_until_parked();
    store.read_with(cx, |store, cx| {
        assert!(!store.outstanding_operations.contains_key("java"));
        assert!(store.extension_manifest_for_id("java").is_none());
        assert_eq!(
            crate::ExtensionSettings::get_global(cx)
                .auto_install_extensions
                .get("java"),
            Some(&false)
        );
    });
    assert!(
        filesystem
            .metadata(Path::new("/doctor-held-extensions/installed/java"))
            .await
            .expect("Metadata")
            .is_none()
    );
}
