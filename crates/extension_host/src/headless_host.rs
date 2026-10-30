use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context as _, Result};
use client::{TypedEnvelope, proto};
use collections::{HashMap, HashSet};
use extension::{
    Extension, ExtensionDebugAdapterProviderProxy, ExtensionGrammarProxy, ExtensionHostProxy,
    ExtensionLanguageProxy, ExtensionLanguageServerProxy, ExtensionManifest,
};
use fs::{Fs, RemoveOptions, RenameOptions};
use futures::future::join_all;
use gpui::{App, AppContext as _, AsyncApp, Context, Entity, Task, WeakEntity};
use http_client::HttpClient;
use language::{LanguageConfig, LanguageName, LanguageQueries, LoadedLanguage};
use lsp::LanguageServerName;
use node_runtime::NodeRuntime;

use crate::wasm_host::{WasmExtension, WasmHost};

#[derive(Clone, Debug)]
pub struct ExtensionVersion {
    pub id: String,
    pub version: String,
    pub dev: bool,
}

pub struct HeadlessExtensionStore {
    pub fs: Arc<dyn Fs>,
    pub extension_dir: PathBuf,
    pub proxy: Arc<ExtensionHostProxy>,
    pub wasm_host: Arc<WasmHost>,
    pub loaded_extensions: HashMap<Arc<str>, Arc<str>>,
    pub loaded_languages: HashMap<Arc<str>, Vec<LanguageName>>,
    pub loaded_grammars: HashMap<Arc<str>, Vec<Arc<str>>>,
    pub loaded_language_servers: HashMap<Arc<str>, Vec<(LanguageServerName, LanguageName)>>,
    pub load_java_runnables: bool,
}

impl HeadlessExtensionStore {
    pub fn new(
        fs: Arc<dyn Fs>,
        http_client: Arc<dyn HttpClient>,
        extension_dir: PathBuf,
        extension_host_proxy: Arc<ExtensionHostProxy>,
        node_runtime: NodeRuntime,
        load_java_runnables: bool,
        cx: &mut App,
    ) -> Entity<Self> {
        cx.new(|cx| Self {
            fs: fs.clone(),
            wasm_host: WasmHost::new(
                fs.clone(),
                http_client.clone(),
                node_runtime,
                extension_host_proxy.clone(),
                extension_dir.join("work"),
                cx,
            ),
            extension_dir,
            proxy: extension_host_proxy,
            loaded_extensions: Default::default(),
            loaded_languages: Default::default(),
            loaded_grammars: Default::default(),
            loaded_language_servers: Default::default(),
            load_java_runnables,
        })
    }

    pub fn sync_extensions(
        &mut self,
        extensions: Vec<ExtensionVersion>,
        cx: &Context<Self>,
    ) -> Task<Result<Vec<ExtensionVersion>>> {
        let on_client = HashSet::from_iter(extensions.iter().map(|e| e.id.as_str()));
        let to_remove: Vec<Arc<str>> = self
            .loaded_extensions
            .keys()
            .filter(|id| !on_client.contains(id.as_ref()))
            .cloned()
            .collect();
        let to_load: Vec<ExtensionVersion> = extensions
            .into_iter()
            .filter(|e| {
                if e.dev {
                    return true;
                }
                self.loaded_extensions
                    .get(e.id.as_str())
                    .is_none_or(|loaded| loaded.as_ref() != e.version.as_str())
            })
            .collect();

        cx.spawn(async move |this, cx| {
            let mut missing = Vec::new();

            for extension_id in to_remove {
                log::info!("removing extension: {}", extension_id);
                this.update(cx, |this, cx| this.uninstall_extension(&extension_id, cx))?
                    .await?;
            }

            for extension in to_load {
                if let Err(e) = Self::load_extension(this.clone(), extension.clone(), cx).await {
                    log::info!("failed to load extension: {}, {:#}", extension.id, e);
                    missing.push(extension)
                } else if extension.dev {
                    missing.push(extension)
                }
            }

            Ok(missing)
        })
    }

    pub async fn load_extension(
        this: WeakEntity<Self>,
        extension: ExtensionVersion,
        cx: &mut AsyncApp,
    ) -> Result<()> {
        let (fs, wasm_host, extension_dir, java_runnables_enabled) =
            this.update(cx, |this, _cx| {
                this.loaded_extensions.insert(
                    extension.id.clone().into(),
                    extension.version.clone().into(),
                );
                (
                    this.fs.clone(),
                    this.wasm_host.clone(),
                    this.extension_dir.join(&extension.id),
                    this.load_java_runnables,
                )
            })?;

        let manifest = Arc::new(ExtensionManifest::load(fs.clone(), &extension_dir).await?);

        debug_assert!(!manifest.languages.is_empty() || manifest.allow_remote_load());

        if manifest.version.as_ref() != extension.version.as_str() {
            anyhow::bail!(
                "mismatched versions: ({}) != ({})",
                manifest.version,
                extension.version
            )
        }

        for language_path in &manifest.languages {
            let language_path = extension_dir.join(language_path);
            let config = fs
                .load(&language_path.join(LanguageConfig::FILE_NAME))
                .await?;
            let mut config = ::toml::from_str::<LanguageConfig>(&config)?;
            let java_runnables = if java_runnables_enabled
                && extension.id == "java"
                && config.name == LanguageName::new("Java")
            {
                let grammar = config
                    .grammar
                    .as_ref()
                    .context("The remote Java extension has no grammar")?;
                anyhow::ensure!(
                    manifest.grammars.contains_key(grammar),
                    "The remote Java extension does not declare its {grammar} grammar"
                );
                Some(
                    load_java_runnables(fs.as_ref(), &extension_dir, &language_path, grammar)
                        .await?,
                )
            } else {
                None
            };

            this.update(cx, |this, _cx| {
                this.loaded_languages
                    .entry(manifest.id.clone())
                    .or_default()
                    .push(config.name.clone());

                let runnable_query = if let Some((grammar, grammar_path, query)) = java_runnables {
                    this.proxy
                        .register_grammars(vec![(grammar.clone(), grammar_path)]);
                    this.loaded_grammars
                        .entry(manifest.id.clone())
                        .or_default()
                        .push(grammar);
                    Some(query)
                } else {
                    config.grammar = None;
                    None
                };

                this.proxy.register_language(
                    config.name.clone(),
                    config.grammar.clone(),
                    config.matcher.clone(),
                    config.hidden,
                    Arc::new(move || {
                        Ok(LoadedLanguage {
                            config: config.clone(),
                            queries: LanguageQueries {
                                runnables: runnable_query.clone().map(Cow::Owned),
                                ..Default::default()
                            },
                            context_provider: None,
                            toolchain_provider: None,
                            manifest_name: None,
                        })
                    }),
                );
            })?;
        }

        if !manifest.allow_remote_load() {
            return Ok(());
        }

        let wasm_extension: Arc<dyn Extension> =
            Arc::new(WasmExtension::load(&extension_dir, &manifest, wasm_host.clone(), cx).await?);

        for (language_server_id, language_server_config) in &manifest.language_servers {
            for language in language_server_config.languages() {
                this.update(cx, |this, _cx| {
                    this.loaded_language_servers
                        .entry(manifest.id.clone())
                        .or_default()
                        .push((language_server_id.clone(), language.clone()));
                    this.proxy.register_language_server(
                        wasm_extension.clone(),
                        language_server_id.clone(),
                        language.clone(),
                    );
                })?;
            }
            log::info!("Loaded language server: {}", language_server_id);
        }

        for (debug_adapter, meta) in &manifest.debug_adapters {
            let schema_path = extension::build_debug_adapter_schema_path(debug_adapter, meta)?;

            this.update(cx, |this, _cx| {
                this.proxy.register_debug_adapter(
                    wasm_extension.clone(),
                    debug_adapter.clone(),
                    &extension_dir.join(schema_path),
                );
            })?;
            log::info!("Loaded debug adapter: {}", debug_adapter);
        }

        for debug_locator in manifest.debug_locators.keys() {
            this.update(cx, |this, _cx| {
                this.proxy
                    .register_debug_locator(wasm_extension.clone(), debug_locator.clone());
            })?;
            log::info!("Loaded debug locator: {}", debug_locator);
        }

        Ok(())
    }

    fn uninstall_extension(
        &mut self,
        extension_id: &Arc<str>,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        self.loaded_extensions.remove(extension_id);

        let languages_to_remove = self
            .loaded_languages
            .remove(extension_id)
            .unwrap_or_default();
        let grammars_to_remove = self
            .loaded_grammars
            .remove(extension_id)
            .unwrap_or_default();
        self.proxy
            .remove_languages(&languages_to_remove, &grammars_to_remove);

        let servers_to_remove = self
            .loaded_language_servers
            .remove(extension_id)
            .unwrap_or_default();
        let proxy = self.proxy.clone();
        let path = self.extension_dir.join(&extension_id.to_string());
        let fs = self.fs.clone();
        cx.spawn(async move |_, cx| {
            let mut removal_tasks = Vec::with_capacity(servers_to_remove.len());
            cx.update(|cx| {
                for (language_server_name, language) in servers_to_remove {
                    removal_tasks.push(proxy.remove_language_server(
                        &language,
                        &language_server_name,
                        cx,
                    ));
                }
            });
            let _ = join_all(removal_tasks).await;

            fs.remove_dir(
                &path,
                RemoveOptions {
                    recursive: true,
                    ignore_if_not_exists: true,
                },
            )
            .await
            .with_context(|| format!("Removing directory {path:?}"))
        })
    }

    pub fn install_extension(
        &mut self,
        extension: ExtensionVersion,
        tmp_path: PathBuf,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let path = self.extension_dir.join(&extension.id);
        let fs = self.fs.clone();

        cx.spawn(async move |this, cx| {
            if fs.is_dir(&path).await {
                this.update(cx, |this, cx| {
                    this.uninstall_extension(&extension.id.clone().into(), cx)
                })?
                .await?;
            }

            fs.rename(&tmp_path, &path, RenameOptions::default())
                .await
                .with_context(|| format!("Failed to rename {tmp_path:?} to {path:?}"))?;

            Self::load_extension(this, extension, cx).await
        })
    }

    pub async fn handle_sync_extensions(
        extension_store: Entity<HeadlessExtensionStore>,
        envelope: TypedEnvelope<proto::SyncExtensions>,
        mut cx: AsyncApp,
    ) -> Result<proto::SyncExtensionsResponse> {
        let requested_extensions =
            envelope
                .payload
                .extensions
                .into_iter()
                .map(|p| ExtensionVersion {
                    id: p.id,
                    version: p.version,
                    dev: p.dev,
                });
        let missing_extensions = extension_store
            .update(&mut cx, |extension_store, cx| {
                extension_store.sync_extensions(requested_extensions.collect(), cx)
            })
            .await?;

        Ok(proto::SyncExtensionsResponse {
            missing_extensions: missing_extensions
                .into_iter()
                .map(|e| proto::Extension {
                    id: e.id,
                    version: e.version,
                    dev: e.dev,
                })
                .collect(),
            tmp_dir: paths::remote_extensions_uploads_dir()
                .to_string_lossy()
                .to_string(),
        })
    }

    pub async fn handle_install_extension(
        extensions: Entity<HeadlessExtensionStore>,
        envelope: TypedEnvelope<proto::InstallExtension>,
        mut cx: AsyncApp,
    ) -> Result<proto::Ack> {
        let extension = envelope
            .payload
            .extension
            .context("Invalid InstallExtension request")?;

        extensions
            .update(&mut cx, |extensions, cx| {
                extensions.install_extension(
                    ExtensionVersion {
                        id: extension.id,
                        version: extension.version,
                        dev: extension.dev,
                    },
                    PathBuf::from(envelope.payload.tmp_dir),
                    cx,
                )
            })
            .await?;

        Ok(proto::Ack {})
    }
}

async fn load_java_runnables(
    fs: &dyn Fs,
    extension_dir: &Path,
    language_path: &Path,
    grammar: &Arc<str>,
) -> Result<(Arc<str>, PathBuf, String)> {
    let grammar_path = extension_dir
        .join("grammars")
        .join(format!("{grammar}.wasm"));
    anyhow::ensure!(
        fs.is_file(&grammar_path).await,
        "Java grammar is missing from {grammar_path:?}"
    );
    let query_path = language_path.join("runnables.scm");
    let query = fs
        .load(&query_path)
        .await
        .with_context(|| format!("Could not load Java runnable query from {query_path:?}"))?;
    Ok((grammar.clone(), grammar_path, query))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs::FakeFs;
    use gpui::TestAppContext;
    use serde_json::json;

    #[gpui::test]
    async fn headless_java_runnable_query_uses_installed_extension_files(cx: &mut TestAppContext) {
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/extensions/java",
            json!({
                "grammars": { "java.wasm": "grammar" },
                "languages": { "java": {
                    "runnables.scm": "(method_declaration name: (identifier) @run @java_method_name)"
                } }
            }),
        )
        .await;
        let extension_dir = Path::new("/extensions/java");
        let language_path = Path::new("/extensions/java/languages/java");
        let grammar = Arc::<str>::from("java");
        let (loaded_grammar, grammar_path, query) =
            load_java_runnables(fs.as_ref(), extension_dir, language_path, &grammar)
                .await
                .expect("the host must load Java runnable evidence from the installed extension");
        assert_eq!(loaded_grammar, grammar);
        assert_eq!(grammar_path, extension_dir.join("grammars/java.wasm"));
        assert!(query.contains("@java_method_name"));
        let missing_query = load_java_runnables(
            fs.as_ref(),
            extension_dir,
            Path::new("/extensions/java/languages/missing"),
            &grammar,
        )
        .await
        .expect_err("a missing query cannot authorize a selected test");
        assert!(missing_query.to_string().contains("runnables.scm"));

        fs.remove_file(
            &extension_dir.join("grammars/java.wasm"),
            RemoveOptions::default(),
        )
        .await
        .expect("remove the cached grammar");
        let missing_grammar =
            load_java_runnables(fs.as_ref(), extension_dir, language_path, &grammar)
                .await
                .expect_err("a partial remote extension must be requested again");
        assert!(missing_grammar.to_string().contains("java.wasm"));
    }
}
