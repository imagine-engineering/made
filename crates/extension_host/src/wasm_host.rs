pub mod wit;

use crate::capability_granter::CapabilityGranter;
use crate::{ExtensionManifest, ExtensionSettings};
use anyhow::{Context as _, Result, anyhow, bail};
use async_trait::async_trait;
use dap::{DebugRequest, StartDebuggingRequestArgumentsRequest};
use extension::{
    CodeLabel, Command, Completion, ContextServerConfiguration, DebugAdapterBinary,
    DebugTaskDefinition, ExtensionCapability, ExtensionHostProxy, KeyValueStoreDelegate,
    ProjectDelegate, SlashCommand, SlashCommandArgumentCompletion, SlashCommandOutput, Symbol,
    WorktreeDelegate,
};
use fs::Fs;
use futures::future::LocalBoxFuture;
use futures::{
    Future, FutureExt, StreamExt as _,
    channel::{
        mpsc::{self, UnboundedSender},
        oneshot,
    },
    future::BoxFuture,
};
use gpui::{App, AsyncApp, BackgroundExecutor, Task};
use http_client::HttpClient;
use language::LanguageName;
use lsp::LanguageServerName;
use moka::sync::Cache;
use node_runtime::NodeRuntime;
use release_channel::ReleaseChannel;
use semver::Version;
use settings::Settings;
use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, OnceLock},
    time::Duration,
};
use task::{DebugScenario, SpawnInTerminal, TaskTemplate, ZedDebugConfig};
use util::paths::SanitizedPath;
use wasmtime::{
    CacheStore, Engine, Store,
    component::{Component, ResourceTable},
};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};
use wit::Extension;

fn apply_jdtls_workspace_default(
    configuration: &mut Option<serde_json::Value>,
    extension_id: &str,
    server_id: Option<&str>,
    default_language_servers: &[&str],
    work_dir: &Path,
) -> Result<()> {
    if extension_id != "java"
        || server_id != Some("jdtls")
        || !default_language_servers.contains(&"jdtls")
    {
        return Ok(());
    }
    // Resolve this on the extension host, after settings precedence, so a remote
    // provider never receives a default derived from the client's filesystem.
    let configuration = configuration.get_or_insert_with(|| serde_json::json!({}));
    let configuration = configuration.as_object_mut().context(
        "JDTLS settings must be an object; review lsp.jdtls.settings on the project host",
    )?;
    if !configuration.contains_key("data_directory") {
        anyhow::ensure!(
            work_dir.is_absolute(),
            "JDTLS extension work directory must be absolute"
        );
        let path = work_dir.join("jdtls-workspaces");
        let path = path
            .to_str()
            .context("JDTLS workspace directory must be valid Unicode")?;
        configuration.insert("data_directory".into(), path.into());
    }
    Ok(())
}

pub struct WasmHost {
    engine: Engine,
    release_channel: ReleaseChannel,
    http_client: Arc<dyn HttpClient>,
    node_runtime: NodeRuntime,
    pub(crate) proxy: Arc<ExtensionHostProxy>,
    fs: Arc<dyn Fs>,
    pub work_dir: PathBuf,
    /// The capabilities granted to extensions running on the host.
    pub(crate) granted_capabilities: Vec<ExtensionCapability>,
    _main_thread_message_task: Task<()>,
    main_thread_message_tx: mpsc::UnboundedSender<MainThreadCall>,
}

#[derive(Clone)]
pub struct ComponentRuntime {
    engine: Engine,
}

impl ComponentRuntime {
    pub fn no_wasi() -> Result<Self> {
        static NO_WASI_ENGINE: OnceLock<Result<Engine, String>> = OnceLock::new();
        let engine = NO_WASI_ENGINE.get_or_init(|| {
            let mut config = wasmtime::Config::new();
            config.wasm_component_model(true);
            config.consume_fuel(true);
            config.epoch_interruption(true);
            config.max_wasm_stack(2 * 1024 * 1024);
            config
                .enable_incremental_compilation(cache_store())
                .map_err(|error| error.to_string())?;
            let engine = Engine::new(&config).map_err(|error| error.to_string())?;
            let epoch_engine = engine.clone();
            let epoch_driver = std::thread::Builder::new()
                .name("zed-component-epoch".to_owned())
                .spawn(move || {
                    loop {
                        std::thread::sleep(COMPONENT_EPOCH_INTERVAL);
                        epoch_engine.increment_epoch();
                    }
                })
                .map_err(|error| error.to_string())?;
            drop(epoch_driver);
            Ok(engine)
        });
        match engine {
            Ok(engine) => Ok(Self {
                engine: engine.clone(),
            }),
            Err(error) => Err(anyhow!(
                "failed to initialize no-WASI component runtime: {error}"
            )),
        }
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn compile_component(&self, bytes: &[u8]) -> Result<Component> {
        Component::new(&self.engine, bytes)
    }

    pub fn increment_epoch(&self) {
        self.engine.increment_epoch();
    }
}

#[derive(Clone, Debug)]
pub struct WasmExtension {
    tx: UnboundedSender<ExtensionCall>,
    pub manifest: Arc<ExtensionManifest>,
    pub work_dir: Arc<Path>,
    #[allow(unused)]
    pub zed_api_version: Version,
    _task: Arc<Task<Result<(), gpui_tokio::JoinError>>>,
}

impl Drop for WasmExtension {
    fn drop(&mut self) {
        self.tx.close_channel();
    }
}

#[async_trait]
impl extension::Extension for WasmExtension {
    fn manifest(&self) -> Arc<ExtensionManifest> {
        self.manifest.clone()
    }

    fn work_dir(&self) -> Arc<Path> {
        self.work_dir.clone()
    }

    async fn language_server_command(
        &self,
        language_server_id: LanguageServerName,
        language_name: LanguageName,
        worktree: Arc<dyn WorktreeDelegate>,
    ) -> Result<Command> {
        self.call(|extension, store| {
            async move {
                let resource = store.data_mut().table.push(worktree)?;
                let command = extension
                    .call_language_server_command(
                        store,
                        &language_server_id,
                        &language_name,
                        resource,
                    )
                    .await?
                    .map_err(|err| store.data().extension_error(err))?;

                Ok(command.into())
            }
            .boxed()
        })
        .await?
    }

    async fn language_server_initialization_options(
        &self,
        language_server_id: LanguageServerName,
        language_name: LanguageName,
        worktree: Arc<dyn WorktreeDelegate>,
    ) -> Result<Option<String>> {
        self.call(|extension, store| {
            async move {
                let resource = store.data_mut().table.push(worktree)?;
                let options = extension
                    .call_language_server_initialization_options(
                        store,
                        &language_server_id,
                        &language_name,
                        resource,
                    )
                    .await?
                    .map_err(|err| store.data().extension_error(err))?;
                anyhow::Ok(options)
            }
            .boxed()
        })
        .await?
    }

    async fn language_server_workspace_configuration(
        &self,
        language_server_id: LanguageServerName,
        worktree: Arc<dyn WorktreeDelegate>,
    ) -> Result<Option<String>> {
        self.call(|extension, store| {
            async move {
                let resource = store.data_mut().table.push(worktree)?;
                let options = extension
                    .call_language_server_workspace_configuration(
                        store,
                        &language_server_id,
                        resource,
                    )
                    .await?
                    .map_err(|err| store.data().extension_error(err))?;
                anyhow::Ok(options)
            }
            .boxed()
        })
        .await?
    }

    async fn language_server_initialization_options_schema(
        &self,
        language_server_id: LanguageServerName,
        worktree: Arc<dyn WorktreeDelegate>,
    ) -> Result<Option<String>> {
        self.call(|extension, store| {
            async move {
                let resource = store.data_mut().table.push(worktree)?;
                extension
                    .call_language_server_initialization_options_schema(
                        store,
                        &language_server_id,
                        resource,
                    )
                    .await
            }
            .boxed()
        })
        .await?
    }

    async fn language_server_workspace_configuration_schema(
        &self,
        language_server_id: LanguageServerName,
        worktree: Arc<dyn WorktreeDelegate>,
    ) -> Result<Option<String>> {
        self.call(|extension, store| {
            async move {
                let resource = store.data_mut().table.push(worktree)?;
                extension
                    .call_language_server_workspace_configuration_schema(
                        store,
                        &language_server_id,
                        resource,
                    )
                    .await
            }
            .boxed()
        })
        .await?
    }

    async fn language_server_additional_initialization_options(
        &self,
        language_server_id: LanguageServerName,
        target_language_server_id: LanguageServerName,
        worktree: Arc<dyn WorktreeDelegate>,
    ) -> Result<Option<String>> {
        self.call(|extension, store| {
            async move {
                let resource = store.data_mut().table.push(worktree)?;
                let options = extension
                    .call_language_server_additional_initialization_options(
                        store,
                        &language_server_id,
                        &target_language_server_id,
                        resource,
                    )
                    .await?
                    .map_err(|err| store.data().extension_error(err))?;
                anyhow::Ok(options)
            }
            .boxed()
        })
        .await?
    }

    async fn language_server_additional_workspace_configuration(
        &self,
        language_server_id: LanguageServerName,
        target_language_server_id: LanguageServerName,
        worktree: Arc<dyn WorktreeDelegate>,
    ) -> Result<Option<String>> {
        self.call(|extension, store| {
            async move {
                let resource = store.data_mut().table.push(worktree)?;
                let options = extension
                    .call_language_server_additional_workspace_configuration(
                        store,
                        &language_server_id,
                        &target_language_server_id,
                        resource,
                    )
                    .await?
                    .map_err(|err| store.data().extension_error(err))?;
                anyhow::Ok(options)
            }
            .boxed()
        })
        .await?
    }

    async fn labels_for_completions(
        &self,
        language_server_id: LanguageServerName,
        completions: Vec<Completion>,
    ) -> Result<Vec<Option<CodeLabel>>> {
        self.call(|extension, store| {
            async move {
                let labels = extension
                    .call_labels_for_completions(
                        store,
                        &language_server_id,
                        completions.into_iter().map(Into::into).collect(),
                    )
                    .await?
                    .map_err(|err| store.data().extension_error(err))?;

                Ok(labels
                    .into_iter()
                    .map(|label| label.map(Into::into))
                    .collect())
            }
            .boxed()
        })
        .await?
    }

    async fn labels_for_symbols(
        &self,
        language_server_id: LanguageServerName,
        symbols: Vec<Symbol>,
    ) -> Result<Vec<Option<CodeLabel>>> {
        self.call(|extension, store| {
            async move {
                let labels = extension
                    .call_labels_for_symbols(
                        store,
                        &language_server_id,
                        symbols.into_iter().map(Into::into).collect(),
                    )
                    .await?
                    .map_err(|err| store.data().extension_error(err))?;

                Ok(labels
                    .into_iter()
                    .map(|label| label.map(Into::into))
                    .collect())
            }
            .boxed()
        })
        .await?
    }

    async fn complete_slash_command_argument(
        &self,
        command: SlashCommand,
        arguments: Vec<String>,
    ) -> Result<Vec<SlashCommandArgumentCompletion>> {
        self.call(|extension, store| {
            async move {
                let completions = extension
                    .call_complete_slash_command_argument(store, &command.into(), &arguments)
                    .await?
                    .map_err(|err| store.data().extension_error(err))?;

                Ok(completions.into_iter().map(Into::into).collect())
            }
            .boxed()
        })
        .await?
    }

    async fn run_slash_command(
        &self,
        command: SlashCommand,
        arguments: Vec<String>,
        delegate: Option<Arc<dyn WorktreeDelegate>>,
    ) -> Result<SlashCommandOutput> {
        self.call(|extension, store| {
            async move {
                let resource = if let Some(delegate) = delegate {
                    Some(store.data_mut().table.push(delegate)?)
                } else {
                    None
                };

                let output = extension
                    .call_run_slash_command(store, &command.into(), &arguments, resource)
                    .await?
                    .map_err(|err| store.data().extension_error(err))?;

                Ok(output.into())
            }
            .boxed()
        })
        .await?
    }

    async fn context_server_command(
        &self,
        context_server_id: Arc<str>,
        project: Arc<dyn ProjectDelegate>,
    ) -> Result<Command> {
        self.call(|extension, store| {
            async move {
                let project_resource = store.data_mut().table.push(project)?;
                let command = extension
                    .call_context_server_command(store, context_server_id.clone(), project_resource)
                    .await?
                    .map_err(|err| store.data().extension_error(err))?;
                anyhow::Ok(command.into())
            }
            .boxed()
        })
        .await?
    }

    async fn context_server_configuration(
        &self,
        context_server_id: Arc<str>,
        project: Arc<dyn ProjectDelegate>,
    ) -> Result<Option<ContextServerConfiguration>> {
        self.call(|extension, store| {
            async move {
                let project_resource = store.data_mut().table.push(project)?;
                let Some(configuration) = extension
                    .call_context_server_configuration(
                        store,
                        context_server_id.clone(),
                        project_resource,
                    )
                    .await?
                    .map_err(|err| store.data().extension_error(err))?
                else {
                    return Ok(None);
                };

                Ok(Some(configuration.try_into()?))
            }
            .boxed()
        })
        .await?
    }

    async fn suggest_docs_packages(&self, provider: Arc<str>) -> Result<Vec<String>> {
        self.call(|extension, store| {
            async move {
                let packages = extension
                    .call_suggest_docs_packages(store, provider.as_ref())
                    .await?
                    .map_err(|err| store.data().extension_error(err))?;

                Ok(packages)
            }
            .boxed()
        })
        .await?
    }

    async fn index_docs(
        &self,
        provider: Arc<str>,
        package_name: Arc<str>,
        kv_store: Arc<dyn KeyValueStoreDelegate>,
    ) -> Result<()> {
        self.call(|extension, store| {
            async move {
                let kv_store_resource = store.data_mut().table.push(kv_store)?;
                extension
                    .call_index_docs(
                        store,
                        provider.as_ref(),
                        package_name.as_ref(),
                        kv_store_resource,
                    )
                    .await?
                    .map_err(|err| store.data().extension_error(err))?;

                anyhow::Ok(())
            }
            .boxed()
        })
        .await?
    }

    async fn get_dap_binary(
        &self,
        dap_name: Arc<str>,
        config: DebugTaskDefinition,
        user_installed_path: Option<PathBuf>,
        worktree: Arc<dyn WorktreeDelegate>,
    ) -> Result<DebugAdapterBinary> {
        self.call(|extension, store| {
            async move {
                let resource = store.data_mut().table.push(worktree)?;
                let dap_binary = extension
                    .call_get_dap_binary(store, dap_name, config, user_installed_path, resource)
                    .await?
                    .map_err(|err| store.data().extension_error(err))?;
                let dap_binary = dap_binary.try_into()?;
                Ok(dap_binary)
            }
            .boxed()
        })
        .await?
    }
    async fn dap_request_kind(
        &self,
        dap_name: Arc<str>,
        config: serde_json::Value,
    ) -> Result<StartDebuggingRequestArgumentsRequest> {
        self.call(|extension, store| {
            async move {
                let kind = extension
                    .call_dap_request_kind(store, dap_name, config)
                    .await?
                    .map_err(|err| store.data().extension_error(err))?;
                Ok(kind.into())
            }
            .boxed()
        })
        .await?
    }

    async fn dap_config_to_scenario(&self, config: ZedDebugConfig) -> Result<DebugScenario> {
        self.call(|extension, store| {
            async move {
                let kind = extension
                    .call_dap_config_to_scenario(store, config)
                    .await?
                    .map_err(|err| store.data().extension_error(err))?;
                Ok(kind)
            }
            .boxed()
        })
        .await?
    }

    async fn dap_locator_create_scenario(
        &self,
        locator_name: String,
        build_config_template: TaskTemplate,
        resolved_label: String,
        debug_adapter_name: String,
    ) -> Result<Option<DebugScenario>> {
        self.call(|extension, store| {
            async move {
                extension
                    .call_dap_locator_create_scenario(
                        store,
                        locator_name,
                        build_config_template,
                        resolved_label,
                        debug_adapter_name,
                    )
                    .await
            }
            .boxed()
        })
        .await?
    }
    async fn run_dap_locator(
        &self,
        locator_name: String,
        config: SpawnInTerminal,
    ) -> Result<DebugRequest> {
        self.call(|extension, store| {
            async move {
                extension
                    .call_run_dap_locator(store, locator_name, config)
                    .await?
                    .map_err(|err| store.data().extension_error(err))
            }
            .boxed()
        })
        .await?
    }
}

pub struct WasmState {
    manifest: Arc<ExtensionManifest>,
    pub table: ResourceTable,
    ctx: WasiCtx,
    pub host: Arc<WasmHost>,
    pub(crate) capability_granter: CapabilityGranter,
}

type MainThreadCall = Box<dyn Send + for<'a> FnOnce(&'a mut AsyncApp) -> LocalBoxFuture<'a, ()>>;

type ExtensionCall = Box<
    dyn Send + for<'a> FnOnce(&'a mut Extension, &'a mut Store<WasmState>) -> BoxFuture<'a, ()>,
>;

fn wasm_engine(executor: &BackgroundExecutor) -> wasmtime::Engine {
    static WASM_ENGINE: OnceLock<wasmtime::Engine> = OnceLock::new();
    WASM_ENGINE
        .get_or_init(|| {
            let mut config = wasmtime::Config::new();
            config.wasm_component_model(true);
            config.async_support(true);
            config
                .enable_incremental_compilation(cache_store())
                .unwrap();
            // Async support introduces the issue that extension execution happens during `Future::poll`,
            // which could block an async thread.
            // https://docs.rs/wasmtime/latest/wasmtime/struct.Config.html#execution-in-poll
            //
            // Epoch interruption is a lightweight mechanism to allow the extensions to yield control
            // back to the executor at regular intervals.
            config.epoch_interruption(true);

            let engine = wasmtime::Engine::new(&config).unwrap();

            // It might be safer to do this on a non-async thread to make sure it makes progress
            // regardless of if extensions are blocking.
            // However, due to our current setup, this isn't a likely occurrence and we'd rather
            // not have a dedicated thread just for this. If it becomes an issue, we can consider
            // creating a separate thread for epoch interruption.
            let engine_ref = engine.weak();
            let executor2 = executor.clone();
            executor
                .spawn(async move {
                    // Somewhat arbitrary interval, as it isn't a guaranteed interval.
                    // But this is a rough upper bound for how long the extension execution can block on
                    // `Future::poll`.
                    loop {
                        executor2.timer(COMPONENT_EPOCH_INTERVAL).await;
                        // Exit the loop and thread once the engine is dropped.
                        let Some(engine) = engine_ref.upgrade() else {
                            break;
                        };
                        engine.increment_epoch();
                    }
                })
                .detach();

            engine
        })
        .clone()
}

const COMPONENT_EPOCH_INTERVAL: Duration = Duration::from_millis(100);

fn cache_store() -> Arc<IncrementalCompilationCache> {
    static CACHE_STORE: LazyLock<Arc<IncrementalCompilationCache>> =
        LazyLock::new(|| Arc::new(IncrementalCompilationCache::new()));
    CACHE_STORE.clone()
}

impl WasmHost {
    pub fn new(
        fs: Arc<dyn Fs>,
        http_client: Arc<dyn HttpClient>,
        node_runtime: NodeRuntime,
        proxy: Arc<ExtensionHostProxy>,
        work_dir: PathBuf,
        cx: &mut App,
    ) -> Arc<Self> {
        let (tx, mut rx) = mpsc::unbounded::<MainThreadCall>();
        let task = cx.spawn(async move |cx| {
            while let Some(message) = rx.next().await {
                message(cx).await;
            }
        });

        let extension_settings = ExtensionSettings::get_global(cx);

        Arc::new(Self {
            engine: wasm_engine(cx.background_executor()),
            fs,
            work_dir,
            http_client,
            node_runtime,
            proxy,
            release_channel: ReleaseChannel::global(cx),
            granted_capabilities: extension_settings.granted_capabilities.clone(),
            _main_thread_message_task: task,
            main_thread_message_tx: tx,
        })
    }

    pub fn no_wasi_component_runtime(&self) -> Result<ComponentRuntime> {
        ComponentRuntime::no_wasi()
    }

    pub fn load_extension(
        self: &Arc<Self>,
        wasm_bytes: Vec<u8>,
        manifest: &Arc<ExtensionManifest>,
        cx: &AsyncApp,
    ) -> Task<Result<WasmExtension>> {
        let this = self.clone();
        let manifest = manifest.clone();
        let executor = cx.background_executor().clone();

        // Parse version and compile component on gpui's background executor.
        // These are cpu-bound operations that don't require a tokio runtime.
        let compile_task = {
            let manifest_id = manifest.id.clone();
            let engine = this.engine.clone();

            executor.spawn(async move {
                let zed_api_version = parse_wasm_extension_version(&manifest_id, &wasm_bytes)?;
                let component = Component::from_binary(&engine, &wasm_bytes)
                    .context("failed to compile wasm component")?;

                anyhow::Ok((zed_api_version, component))
            })
        };

        let load_extension = |zed_api_version: Version, component| async move {
            let wasi_ctx = this.build_wasi_ctx(&manifest).await?;
            let mut store = wasmtime::Store::new(
                &this.engine,
                WasmState {
                    ctx: wasi_ctx,
                    manifest: manifest.clone(),
                    table: ResourceTable::new(),
                    host: this.clone(),
                    capability_granter: CapabilityGranter::new(
                        this.granted_capabilities.clone(),
                        manifest.clone(),
                    ),
                },
            );
            // Store will yield after 1 tick, and get a new deadline of 1 tick after each yield.
            store.set_epoch_deadline(1);
            store.epoch_deadline_async_yield_and_update(1);

            let mut extension = Extension::instantiate_async(
                &executor,
                &mut store,
                this.release_channel,
                zed_api_version.clone(),
                &component,
            )
            .await?;

            extension
                .call_init_extension(&mut store)
                .await
                .context("failed to initialize wasm extension")?;

            let (tx, mut rx) = mpsc::unbounded::<ExtensionCall>();
            let extension_task = async move {
                while let Some(call) = rx.next().await {
                    (call)(&mut extension, &mut store).await;
                }
            };

            anyhow::Ok((
                extension_task,
                manifest.clone(),
                this.work_dir.join(manifest.id.as_ref()).into(),
                tx,
                zed_api_version,
            ))
        };

        cx.spawn(async move |cx| {
            let (zed_api_version, component) = compile_task.await?;

            // Run wasi-dependent operations on tokio.
            // wasmtime_wasi internally uses tokio for I/O operations.
            let (extension_task, manifest, work_dir, tx, zed_api_version) =
                gpui_tokio::Tokio::spawn(cx, load_extension(zed_api_version, component)).await??;

            // Run the extension message loop on tokio since extension
            // calls may invoke wasi functions that require a tokio runtime.
            let task = Arc::new(gpui_tokio::Tokio::spawn(cx, extension_task));

            Ok(WasmExtension {
                manifest,
                work_dir,
                tx,
                zed_api_version,
                _task: task,
            })
        })
    }

    async fn build_wasi_ctx(&self, manifest: &Arc<ExtensionManifest>) -> Result<WasiCtx> {
        let extension_work_dir = self.work_dir.join(manifest.id.as_ref());
        self.fs
            .create_dir(&extension_work_dir)
            .await
            .context("failed to create extension work dir")?;

        let file_perms = wasmtime_wasi::FilePerms::all();
        let dir_perms = wasmtime_wasi::DirPerms::all();
        let path = SanitizedPath::new(&extension_work_dir).to_string();
        #[cfg(target_os = "windows")]
        let path = path.replace('\\', "/");

        let mut ctx = WasiCtxBuilder::new();
        ctx.inherit_stdio()
            .env("PWD", &path)
            .env("RUST_BACKTRACE", "full");

        ctx.preopened_dir(&path, ".", dir_perms, file_perms)?;
        ctx.preopened_dir(&path, &path, dir_perms, file_perms)?;

        Ok(ctx.build())
    }

    pub async fn writeable_path_from_extension(
        &self,
        id: &Arc<str>,
        path: &Path,
    ) -> Result<PathBuf> {
        let canonical_work_dir = self
            .fs
            .canonicalize(&self.work_dir)
            .await
            .with_context(|| format!("canonicalizing work dir {:?}", self.work_dir))?;
        let extension_work_dir = canonical_work_dir.join(id.as_ref());

        let absolute = if path.is_relative() {
            extension_work_dir.join(path)
        } else {
            path.to_path_buf()
        };

        let normalized = util::paths::normalize_lexically(&absolute)
            .map_err(|_| anyhow!("path {path:?} escapes its parent"))?;

        // Canonicalize the nearest existing ancestor to resolve any symlinks
        // in the on-disk portion of the path. Components beyond that ancestor
        // are re-appended, which lets this work for destinations that don't
        // exist yet (e.g. nested directories created by tar extraction).
        let mut existing = normalized.as_path();
        let mut tail_components = Vec::new();
        let canonical_prefix = loop {
            match self.fs.canonicalize(existing).await {
                Ok(canonical) => break canonical,
                Err(_) => {
                    if let Some(file_name) = existing.file_name() {
                        tail_components.push(file_name.to_owned());
                    }
                    existing = existing
                        .parent()
                        .context(format!("cannot resolve path {path:?}"))?;
                }
            }
        };

        let mut resolved = canonical_prefix;
        for component in tail_components.into_iter().rev() {
            resolved.push(component);
        }

        anyhow::ensure!(
            resolved.starts_with(&extension_work_dir),
            "cannot write to path {resolved:?}",
        );
        Ok(resolved)
    }
}

pub fn parse_wasm_extension_version(extension_id: &str, wasm_bytes: &[u8]) -> Result<Version> {
    let mut version = None;

    for part in wasmparser::Parser::new(0).parse_all(wasm_bytes) {
        if let wasmparser::Payload::CustomSection(s) =
            part.context("error parsing wasm extension")?
            && s.name() == "zed:api-version"
        {
            version = parse_wasm_extension_version_custom_section(s.data());
            if version.is_none() {
                bail!(
                    "extension {} has invalid zed:api-version section: {:?}",
                    extension_id,
                    s.data()
                );
            }
        }
    }

    // The reason we wait until we're done parsing all of the Wasm bytes to return the version
    // is to work around a panic that can happen inside of Wasmtime when the bytes are invalid.
    //
    // By parsing the entirety of the Wasm bytes before we return, we're able to detect this problem
    // earlier as an `Err` rather than as a panic.
    version.with_context(|| format!("extension {extension_id} has no zed:api-version section"))
}

fn parse_wasm_extension_version_custom_section(data: &[u8]) -> Option<Version> {
    if data.len() == 6 {
        Some(Version::new(
            u16::from_be_bytes([data[0], data[1]]) as _,
            u16::from_be_bytes([data[2], data[3]]) as _,
            u16::from_be_bytes([data[4], data[5]]) as _,
        ))
    } else {
        None
    }
}

impl WasmExtension {
    pub async fn load(
        extension_dir: &Path,
        manifest: &Arc<ExtensionManifest>,
        wasm_host: Arc<WasmHost>,
        cx: &AsyncApp,
    ) -> Result<Self> {
        let path = extension_dir.join("extension.wasm");

        let mut wasm_file = wasm_host
            .fs
            .open_sync(&path)
            .await
            .context(format!("opening wasm file, path: {path:?}"))?;

        let mut wasm_bytes = Vec::new();
        wasm_file
            .read_to_end(&mut wasm_bytes)
            .context(format!("reading wasm file, path: {path:?}"))?;

        wasm_host
            .load_extension(wasm_bytes, manifest, cx)
            .await
            .with_context(|| format!("loading wasm extension: {}", manifest.id))
    }

    pub async fn call<T, Fn>(&self, f: Fn) -> Result<T>
    where
        T: 'static + Send,
        Fn: 'static
            + Send
            + for<'a> FnOnce(&'a mut Extension, &'a mut Store<WasmState>) -> BoxFuture<'a, T>,
    {
        let (return_tx, return_rx) = oneshot::channel();
        self.tx
            .unbounded_send(Box::new(move |extension, store| {
                async {
                    let result = f(extension, store).await;
                    return_tx.send(result).ok();
                }
                .boxed()
            }))
            .map_err(|_| {
                anyhow!(
                    "wasm extension channel should not be closed yet, extension {} (id {})",
                    self.manifest.name,
                    self.manifest.id,
                )
            })?;
        return_rx.await.with_context(|| {
            format!(
                "wasm extension channel, extension {} (id {})",
                self.manifest.name, self.manifest.id,
            )
        })
    }
}

impl WasmState {
    fn on_main_thread<T, Fn>(&self, f: Fn) -> impl 'static + Future<Output = T>
    where
        T: 'static + Send,
        Fn: 'static + Send + for<'a> FnOnce(&'a mut AsyncApp) -> LocalBoxFuture<'a, T>,
    {
        let (return_tx, return_rx) = oneshot::channel();
        self.host
            .main_thread_message_tx
            .clone()
            .unbounded_send(Box::new(move |cx| {
                async {
                    let result = f(cx).await;
                    return_tx.send(result).ok();
                }
                .boxed_local()
            }))
            .unwrap_or_else(|_| {
                panic!(
                    "main thread message channel should not be closed yet, extension {} (id {})",
                    self.manifest.name, self.manifest.id,
                )
            });
        let name = self.manifest.name.clone();
        let id = self.manifest.id.clone();
        async move {
            return_rx.await.unwrap_or_else(|_| {
                panic!("main thread message channel, extension {name} (id {id})")
            })
        }
    }

    fn work_dir(&self) -> PathBuf {
        self.host.work_dir.join(self.manifest.id.as_ref())
    }

    fn extension_error(&self, message: String) -> anyhow::Error {
        anyhow!(
            "from extension \"{}\" version {}: {}",
            self.manifest.name,
            self.manifest.version,
            message
        )
    }
}

impl wasmtime::component::HasData for WasmState {
    type Data<'a> = &'a mut WasmState;
}

impl WasiView for WasmState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.ctx,
            table: &mut self.table,
        }
    }
}

/// Wrapper around a mini-moka bounded cache for storing incremental compilation artifacts.
/// Since wasm modules have many similar elements, this can save us a lot of work at the
/// cost of a small memory footprint. However, we don't want this to be unbounded, so we use
/// a LFU/LRU cache to evict less used cache entries.
#[derive(Debug)]
struct IncrementalCompilationCache {
    cache: Cache<Vec<u8>, Vec<u8>>,
}

impl IncrementalCompilationCache {
    fn new() -> Self {
        let cache = Cache::builder()
            // Cap this at 32 MB for now. Our extensions turn into roughly 512kb in the cache,
            // which means we could store 64 completely novel extensions in the cache, but in
            // practice we will more than that, which is more than enough for our use case.
            .max_capacity(32 * 1024 * 1024)
            .weigher(|k: &Vec<u8>, v: &Vec<u8>| (k.len() + v.len()).try_into().unwrap_or(u32::MAX))
            .build();
        Self { cache }
    }
}

impl CacheStore for IncrementalCompilationCache {
    fn get(&self, key: &[u8]) -> Option<Cow<'_, [u8]>> {
        self.cache.get(key).map(|v| v.into())
    }

    fn insert(&self, key: &[u8], value: Vec<u8>) -> bool {
        self.cache.insert(key.to_vec(), value);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use extension::ExtensionHostProxy;
    use fs::FakeFs;
    use gpui::TestAppContext;
    use http_client::FakeHttpClient;
    use node_runtime::NodeRuntime;
    use serde_json::json;
    use settings::SettingsStore;

    #[test]
    fn jdtls_workspace_defaults_use_the_provider_host_and_preserve_overrides() {
        let host_directory = std::env::temp_dir().join("groovy-dev/work/java");
        for directory in [
            &host_directory,
            &std::env::temp_dir().join("remote/groovy/work/java"),
        ] {
            let mut configuration = Some(json!({"java_home": "/explicit/jdk"}));
            apply_jdtls_workspace_default(
                &mut configuration,
                "java",
                Some("jdtls"),
                &["jdtls"],
                directory,
            )
            .expect("host default");
            assert_eq!(
                configuration,
                Some(
                    json!({"java_home": "/explicit/jdk", "data_directory": directory.join("jdtls-workspaces")})
                )
            );
        }
        for value in [json!("/user/chosen"), json!(null), json!(""), json!(42)] {
            let original = Some(json!({"data_directory": value}));
            let mut configuration = original.clone();
            apply_jdtls_workspace_default(
                &mut configuration,
                "java",
                Some("jdtls"),
                &["jdtls"],
                &host_directory,
            )
            .expect("explicit value remains provider-owned");
            assert_eq!(configuration, original);
        }
        for (extension, server, defaults) in [
            ("java", Some("jdtls"), &[][..]),
            ("kotlin", Some("jdtls"), &["jdtls"][..]),
            ("java", Some("other"), &["jdtls"][..]),
            ("java", None, &["jdtls"][..]),
        ] {
            let mut configuration = None;
            apply_jdtls_workspace_default(
                &mut configuration,
                extension,
                server,
                defaults,
                &host_directory,
            )
            .expect("other products/providers unchanged");
            assert_eq!(configuration, None);
        }
        for value in [json!(null), json!("invalid settings"), json!(false)] {
            let mut configuration = Some(value.clone());
            assert!(
                apply_jdtls_workspace_default(
                    &mut configuration,
                    "java",
                    Some("jdtls"),
                    &["jdtls"],
                    &host_directory,
                )
                .is_err()
            );
            assert_eq!(configuration, Some(value));
        }
        let mut configuration = None;
        assert!(
            apply_jdtls_workspace_default(
                &mut configuration,
                "java",
                Some("jdtls"),
                &["jdtls"],
                Path::new("relative/work")
            )
            .is_err()
        );
    }

    fn init_test(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let store = SettingsStore::test(cx);
            cx.set_global(store);
            release_channel::init(semver::Version::new(0, 0, 0), cx);
            extension::init(cx);
            gpui_tokio::init(cx);
        });
    }

    #[gpui::test]
    async fn test_wasi_guest_cannot_escape_production_preopens(cx: &mut TestAppContext) {
        use sha2::{Digest as _, Sha256};

        init_test(cx);
        cx.background_executor.allow_parking();
        let metadata: serde_json::Value =
            serde_json::from_str(include_str!("../test_data/wasi_confinement.json"))
                .expect("guest provenance");
        for (name, bytes) in [
            (
                "wasi_confinement.rs",
                include_bytes!("../test_data/wasi_confinement.rs").as_slice(),
            ),
            (
                "wasi_confinement.wasm",
                include_bytes!("../test_data/wasi_confinement.wasm").as_slice(),
            ),
        ] {
            let expected = metadata
                .get("sha256")
                .and_then(|hashes| hashes.get(name))
                .and_then(serde_json::Value::as_str)
                .expect("recorded source/component hash");
            assert_eq!(format!("{:x}", Sha256::digest(bytes)), expected);
        }
        let directory = tempfile::tempdir().expect("disposable confinement fixture");
        let work_dir =
            std::fs::canonicalize(directory.path()).expect("canonical fixture directory");
        let extension_dir = work_dir.join("confinement-test");
        let outside = work_dir.join("outside");
        let inside_directory = extension_dir.join("inside-directory");
        std::fs::create_dir_all(&inside_directory).expect("inside directory");
        std::fs::create_dir_all(outside.join("empty")).expect("outside directory");
        std::fs::write(extension_dir.join("inside.txt"), b"sandbox-positive").expect("inside file");
        std::fs::write(
            extension_dir.join("inside-directory/nested.txt"),
            b"nested-positive",
        )
        .expect("inside nested file");
        std::fs::write(outside.join("sensitive.txt"), b"outside-unchanged").expect("outside file");
        let trailing_target = PathBuf::from(format!("{}/", outside.display()));
        for (target, name) in [
            (outside.as_path(), "escape-direct"),
            (trailing_target.as_path(), "escape-target-slash"),
            (Path::new("inside-directory"), "inside-link"),
        ] {
            #[cfg(unix)]
            std::os::unix::fs::symlink(target, extension_dir.join(name))
                .expect("directory symlink");
            #[cfg(windows)]
            std::os::windows::fs::symlink_dir(target, extension_dir.join(name))
                .expect("native Windows symlink privileges are required");
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            outside.join("sensitive.txt"),
            extension_dir.join("escape-file"),
        )
        .expect("file symlink");
        #[cfg(windows)]
        std::os::windows::fs::symlink_file(
            outside.join("sensitive.txt"),
            extension_dir.join("escape-file"),
        )
        .expect("native Windows symlink privileges are required");

        let host = cx.update(|cx| {
            WasmHost::new(
                Arc::new(fs::RealFs::new(None, cx.background_executor().clone())),
                FakeHttpClient::with_200_response(),
                NodeRuntime::unavailable(),
                Arc::new(ExtensionHostProxy::default()),
                work_dir,
                cx,
            )
        });
        let manifest: Arc<ExtensionManifest> = Arc::new(
            serde_json::from_value(json!({
                "id": "confinement-test", "name": "Confinement fixture",
                "version": "0.0.0", "schema_version": 0
            }))
            .expect("minimal test manifest"),
        );
        let context = host
            .build_wasi_ctx(&manifest)
            .await
            .expect("actual production WASI preopens");
        gpui_tokio::Tokio::spawn_result(cx, async move {
            let engine = host.engine.clone();
            let component = Component::new(
                &engine,
                include_bytes!("../test_data/wasi_confinement.wasm"),
            )?;
            let mut linker = wasmtime::component::Linker::<WasmState>::new(&engine);
            wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;
            let mut store = Store::new(
                &engine,
                WasmState {
                    ctx: context,
                    table: ResourceTable::new(),
                    manifest: manifest.clone(),
                    capability_granter: CapabilityGranter::new(
                        host.granted_capabilities.clone(),
                        manifest,
                    ),
                    host,
                },
            );
            store.set_epoch_deadline(1);
            store.epoch_deadline_async_yield_and_update(1);
            let command = wasmtime_wasi::p2::bindings::Command::instantiate_async(
                &mut store, &component, &linker,
            )
            .await?;
            let result = command.wasi_cli_run().call_run(&mut store).await?;
            anyhow::ensure!(
                result.is_ok(),
                "guest confinement or positive-control assertion failed"
            );
            anyhow::Ok(())
        })
        .await
        .expect("execute the actual guest filesystem operations");

        assert_eq!(
            std::fs::read(outside.join("sensitive.txt")).expect("outside file remains"),
            b"outside-unchanged"
        );
        assert!(outside.join("empty").is_dir());
        assert!(!outside.join("new.txt").exists());
        assert!(!outside.join("new-directory").exists());
        assert!(!outside.join("moved.txt").exists());
        assert!(!extension_dir.join("moved-from-outside.txt").exists());
        assert_eq!(
            std::fs::read_dir(&outside)
                .expect("outside still accessible by host")
                .collect::<std::io::Result<Vec<_>>>()
                .expect("read every outside entry")
                .len(),
            2
        );
    }

    #[gpui::test]
    async fn test_writeable_path_rejects_escape_attempts(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/work",
            json!({
                "test-extension": {
                    "legit.txt": "legitimate content"
                }
            }),
        )
        .await;
        fs.insert_tree("/outside", json!({ "secret.txt": "sensitive data" }))
            .await;
        fs.insert_symlink("/work/test-extension/escape", PathBuf::from("/outside"))
            .await;

        let host = cx.update(|cx| {
            WasmHost::new(
                fs.clone(),
                FakeHttpClient::with_200_response(),
                NodeRuntime::unavailable(),
                Arc::new(ExtensionHostProxy::default()),
                PathBuf::from("/work"),
                cx,
            )
        });

        let extension_id: Arc<str> = "test-extension".into();

        // A path traversing through a symlink that points outside the work dir
        // must be rejected. Canonicalization resolves the symlink before the
        // prefix check, so this is caught.
        let result = host
            .writeable_path_from_extension(
                &extension_id,
                Path::new("/work/test-extension/escape/secret.txt"),
            )
            .await;
        assert!(
            result.is_err(),
            "symlink escape should be rejected, but got: {result:?}",
        );

        // A path using `..` to escape the extension work dir must be rejected.
        let result = host
            .writeable_path_from_extension(
                &extension_id,
                Path::new("/work/test-extension/../../outside/secret.txt"),
            )
            .await;
        assert!(
            result.is_err(),
            "parent traversal escape should be rejected, but got: {result:?}",
        );

        // A legitimate path within the extension work dir should succeed.
        let result = host
            .writeable_path_from_extension(
                &extension_id,
                Path::new("/work/test-extension/legit.txt"),
            )
            .await;
        assert!(
            result.is_ok(),
            "legitimate path should be accepted, but got: {result:?}",
        );

        // A relative path with non-existent intermediate directories should
        // succeed, mirroring the integration test pattern where an extension
        // downloads a tar to e.g. "gleam-v1.2.3" (creating the directory)
        // and then references "gleam-v1.2.3/gleam" inside it.
        let result = host
            .writeable_path_from_extension(&extension_id, Path::new("new-dir/nested/binary"))
            .await;
        assert!(
            result.is_ok(),
            "relative path with non-existent parents should be accepted, but got: {result:?}",
        );

        // A symlink deeper than the immediate parent must still be caught.
        // Here "escape" is a symlink to /outside, so "escape/deep/file.txt"
        // has multiple non-existent components beyond the symlink.
        let result = host
            .writeable_path_from_extension(&extension_id, Path::new("escape/deep/nested/file.txt"))
            .await;
        assert!(
            result.is_err(),
            "symlink escape through deep non-existent path should be rejected, but got: {result:?}",
        );
    }
}
