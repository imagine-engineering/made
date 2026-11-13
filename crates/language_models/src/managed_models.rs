use crate::AllLanguageModelSettings;
use fs::Fs;
use futures::FutureExt as _;
use gpui::{App, AppContext as _, Context, Entity, Global, Render, Task};
use http_client::HttpClient;
use llama_cpp::managed::{
    Catalog, DecodingMode, Hardware, LoadOptions, ManagedRuntime, ManagedStore, ModelArtifact,
    ModelState, catalog, gib,
};
use settings::{LanguageModelSelection, Settings, update_settings_file};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};
use ui::prelude::*;
use util::ResultExt as _;

pub(crate) struct ManagedModelsGlobal(Entity<ManagedModels>);
impl Global for ManagedModelsGlobal {}

pub struct ManagedModels {
    catalog: Option<Catalog>,
    store: Option<ManagedStore>,
    hardware: Option<Hardware>,
    states: BTreeMap<String, ModelState>,
    downloaded: BTreeSet<String>,
    http: Arc<dyn HttpClient>,
    local_http: Option<Arc<dyn HttpClient>>,
    runtime: Option<ManagedRuntime>,
    endpoint: Option<String>,
    error: Option<String>,
    progress: Arc<AtomicU64>,
    cancelled: Arc<AtomicBool>,
    operation: Option<Task<()>>,
    polling: Option<Task<()>>,
    busy_model: Option<String>,
    cpu_only: bool,
    decoding: ModelDecoding,
    selected_model: Option<String>,
    expanded_models: BTreeSet<String>,
    hardware_details: bool,
}

pub fn init(http: Arc<dyn HttpClient>, cx: &mut App) {
    let (local_http, initialization_error) = match reqwest_client::ReqwestClient::loopback() {
        Ok(client) => (Some(Arc::new(client) as Arc<dyn HttpClient>), None),
        Err(error) => (
            None,
            Some(format!(
                "Cannot initialize direct local inference transport: {error:#}"
            )),
        ),
    };
    let entity = cx.new(|cx| {
        cx.on_app_quit(|this: &mut ManagedModels, _| {
            this.cancelled.store(true, Ordering::Relaxed);
            this.operation = None;
            this.polling = None;
            this.runtime = None;
            async {}
        })
        .detach();
        ManagedModels {
            catalog: None,
            store: None,
            hardware: None,
            states: BTreeMap::new(),
            downloaded: BTreeSet::new(),
            http,
            local_http,
            runtime: None,
            endpoint: None,
            error: initialization_error,
            progress: Arc::new(AtomicU64::new(0)),
            cancelled: Arc::new(AtomicBool::new(false)),
            operation: None,
            polling: None,
            busy_model: None,
            cpu_only: !cfg!(all(target_os = "macos", target_arch = "aarch64")),
            decoding: ModelDecoding::default(),
            selected_model: None,
            expanded_models: BTreeSet::new(),
            hardware_details: false,
        }
    });
    cx.set_global(ManagedModelsGlobal(entity.clone()));
    entity.update(cx, |this, cx| this.initialize(cx));
}

#[derive(Default)]
struct ModelDecoding {
    mtp_enabled: BTreeSet<String>,
    failed_mtp_loads: BTreeMap<String, bool>,
}

impl ModelDecoding {
    fn next(&self, model: &ModelArtifact) -> DecodingMode {
        if model.experimental_mtp_memory_mib.is_some() && self.mtp_enabled.contains(&model.id) {
            DecodingMode::ExperimentalMtp
        } else {
            DecodingMode::Standard
        }
    }

    fn toggle(&mut self, model: &ModelArtifact) {
        match self.next(model) {
            DecodingMode::ExperimentalMtp => {
                self.mtp_enabled.remove(&model.id);
            }
            DecodingMode::Standard if model.experimental_mtp_memory_mib.is_some() => {
                self.mtp_enabled.insert(model.id.clone());
            }
            DecodingMode::Standard => {}
        }
        self.failed_mtp_loads.remove(&model.id);
    }

    fn retry_standard(&mut self, model_id: &str) -> Option<bool> {
        let select_default = self.failed_mtp_loads.remove(model_id)?;
        self.mtp_enabled.remove(model_id);
        Some(select_default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context as _;

    #[test]
    fn saved_model_starts_in_standard_and_retry_clears_only_its_mtp_opt_in() -> anyhow::Result<()> {
        let catalog = catalog()?;
        let model = |id| {
            catalog
                .models
                .iter()
                .find(|model| model.id == id)
                .context("Missing test model")
        };
        let first = model("unsloth-qwen3-8-27b-ud-q4-k-m")?;
        let second = model("unsloth-qwen3-8-27b-ud-iq3-s")?;
        let other = model("qwen3-4b-q4-k-m")?;
        let saved_model_id = first.id.clone();

        let mut decoding = ModelDecoding::default();
        assert_eq!(
            decoding.next(model(&saved_model_id)?),
            DecodingMode::Standard
        );
        decoding.toggle(first);
        assert_eq!(decoding.next(second), DecodingMode::Standard);
        decoding.toggle(second);
        decoding.toggle(other);
        assert_eq!(decoding.next(other), DecodingMode::Standard);

        decoding.failed_mtp_loads.insert(first.id.clone(), true);
        assert_eq!(decoding.retry_standard(&first.id), Some(true));
        assert_eq!(decoding.next(first), DecodingMode::Standard);
        assert_eq!(decoding.next(second), DecodingMode::ExperimentalMtp);
        assert_eq!(decoding.retry_standard(&first.id), None);
        assert_eq!(ModelDecoding::default().next(first), DecodingMode::Standard);
        Ok(())
    }
}

pub fn global(cx: &App) -> Entity<ManagedModels> {
    cx.global::<ManagedModelsGlobal>().0.clone()
}

pub fn endpoint(cx: &App) -> Option<String> {
    let selected = AllLanguageModelSettings::get_global(cx)
        .llama_cpp
        .managed_model
        .as_deref()?;
    let models = cx.try_global::<ManagedModelsGlobal>()?.0.read(cx);
    if models.runtime.as_ref()?.model_id != selected {
        return None;
    }
    models.endpoint.clone()
}

pub fn api_key(cx: &App) -> Option<Arc<str>> {
    cx.try_global::<ManagedModelsGlobal>().and_then(|global| {
        global
            .0
            .read(cx)
            .runtime
            .as_ref()
            .map(|runtime| runtime.api_key.clone())
    })
}

pub fn local_http(cx: &App) -> anyhow::Result<Arc<dyn HttpClient>> {
    cx.try_global::<ManagedModelsGlobal>()
        .and_then(|global| global.0.read(cx).local_http.clone())
        .ok_or_else(|| anyhow::anyhow!("Direct local inference transport is unavailable; no cloud or proxy fallback was used"))
}

impl ManagedModels {
    fn selected_default(cx: &App) -> Option<String> {
        language_model::LanguageModelRegistry::read_global(cx)
            .default_model()
            .filter(|configured| configured.provider.id().0.as_ref() == "llama.cpp")
            .map(|configured| configured.model.id().0.to_string())
    }

    fn persist_default(id: String, cx: &mut App) {
        update_settings_file(<dyn Fs>::global(cx), cx, move |settings, _| {
            settings
                .language_models
                .get_or_insert_default()
                .llama_cpp
                .get_or_insert_default()
                .managed_model = Some(id.clone());
            settings
                .agent
                .get_or_insert_default()
                .set_model(LanguageModelSelection {
                    provider: "llama.cpp".into(),
                    model: id,
                    enable_thinking: false,
                    effort: None,
                    speed: None,
                });
        });
    }
    pub(crate) fn stop(&mut self, cx: &mut Context<Self>) {
        self.cancel(cx);
        if let Some(runtime) = self.runtime.take() {
            self.states.insert(runtime.model_id, ModelState::Downloaded);
        }
        self.endpoint = None;
        cx.notify();
    }
    fn initialize(&mut self, cx: &mut Context<Self>) {
        let root = paths::data_dir().join("managed-local-models");
        let task = cx.background_spawn(async move {
            let catalog = catalog()?;
            let store = ManagedStore::open(root)?;
            let hardware = Hardware::detect(store.root())?;
            let states = store.states(&catalog.models);
            Ok::<_, anyhow::Error>((catalog, store, hardware, states))
        });
        self.operation = Some(cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |this, cx| {
                this.operation = None;
                match result {
                    Ok((catalog, store, hardware, states)) => {
                        this.catalog = Some(catalog);
                        this.store = Some(store);
                        this.hardware = Some(hardware);
                        this.downloaded = states
                            .iter()
                            .filter_map(|(id, state)| {
                                (*state == ModelState::Downloaded).then(|| id.clone())
                            })
                            .collect();
                        this.states = states;
                        let selected = AllLanguageModelSettings::get_global(cx)
                            .llama_cpp
                            .managed_model
                            .clone();
                        if let Some(model) = selected.and_then(|id| this.model(&id)) {
                            this.load(model, false, cx);
                        }
                    }
                    Err(error) => {
                        this.error = Some(format!(
                            "Local model storage could not initialize: {error:#}"
                        ))
                    }
                }
                cx.notify();
            })
            .log_err();
        }));
        self.polling=Some(cx.spawn(async move |this,cx| {
            loop {
                cx.background_executor().timer(Duration::from_millis(500)).await;
                if this.update(cx, |this,cx| {
                    if this.busy_model.is_some() { cx.notify(); }
                    if let Some(runtime)=this.runtime.as_mut() {
                        match runtime.is_running() {
                            Ok(true) => {},
                            result => {
                                let model_id=runtime.model_id.clone();
                                let message=match result {Err(error)=>format!("Could not check local runtime: {error:#}"),_=>"Local runtime stopped. Try loading the model again, choose CPU fallback, or select a smaller model. No cloud fallback was used.".into()};
                                this.states.insert(model_id,ModelState::Failed(message));
                                this.endpoint=None;this.runtime=None;cx.notify();
                            }
                        }
                    }
                    if let Some(runtime) = &this.runtime {
                        let active = Self::selected_default(cx).as_deref() == Some(runtime.model_id.as_str())
                            && AllLanguageModelSettings::get_global(cx).llama_cpp.managed_model.as_deref() == Some(runtime.model_id.as_str());
                        let state = if active { ModelState::Active } else { ModelState::Ready };
                        if this.states.get(&runtime.model_id) != Some(&state) {
                            this.states.insert(runtime.model_id.clone(), state);
                            cx.notify();
                        }
                    }
                }).is_err() { break; }
            }
        }));
    }

    fn model(&self, id: &str) -> Option<ModelArtifact> {
        self.catalog
            .as_ref()?
            .models
            .iter()
            .find(|model| model.id == id)
            .cloned()
    }

    fn cancel(&mut self, cx: &mut Context<Self>) {
        self.cancelled.store(true, Ordering::Relaxed);
        self.operation = None;
        if let Some(id) = self.busy_model.take() {
            self.states.insert(
                id.clone(),
                ModelState::Failed(if self.downloaded.contains(&id) {
                    "Loading cancelled. The downloaded model was preserved; choose Retry load."
                        .into()
                } else {
                    "Cancelled. Downloaded bytes are preserved; choose Resume / retry.".into()
                }),
            );
        }
        cx.notify();
    }

    fn toggle_backend(&mut self, cx: &mut Context<Self>) {
        self.cpu_only = !self.cpu_only;
        if let Some(model) = self
            .runtime
            .as_ref()
            .and_then(|runtime| self.model(&runtime.model_id))
        {
            self.load(model, false, cx);
        }
        cx.notify();
    }

    fn download(&mut self, model: ModelArtifact, cx: &mut Context<Self>) {
        if self.operation.is_some() {
            return;
        }
        let Some(store) = self.store.clone() else {
            return;
        };
        self.progress.store(0, Ordering::Relaxed);
        self.cancelled = Arc::new(AtomicBool::new(false));
        let progress = self.progress.clone();
        let cancelled = self.cancelled.clone();
        let http = self.http.clone();
        self.states
            .insert(model.id.clone(), ModelState::Downloading);
        self.busy_model = Some(model.id.clone());
        self.error = None;
        let task = cx.background_spawn(async move {
            store
                .download(&model, http.as_ref(), progress, cancelled)
                .await
        });
        self.operation = Some(cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |this, cx| {
                if let Some(id) = this.busy_model.take() {
                    let state = match result {
                        Ok(()) => {
                            this.downloaded.insert(id.clone());
                            ModelState::Downloaded
                        }
                        Err(error) => ModelState::Failed(format!("{error:#}")),
                    };
                    this.states.insert(id, state);
                }
                this.operation = None;
                this.refresh_hardware(cx);
                cx.notify();
            })
            .log_err();
        }));
        cx.notify();
    }

    fn refresh_hardware(&mut self, cx: &mut Context<Self>) {
        let Some(store) = self.store.clone() else {
            return;
        };
        let task = cx.background_spawn(async move { Hardware::detect(store.root()) });
        cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |this, cx| {
                match result {
                    Ok(mut hardware) => {
                        if let Some(current) = &this.hardware {
                            if this.runtime.is_some() {
                                hardware.backend = current.backend.clone();
                                hardware.gpu_memory = current.gpu_memory;
                            }
                        }
                        this.hardware = Some(hardware);
                    }
                    Err(error) => this.error = Some(format!("Capacity check failed: {error:#}")),
                };
                cx.notify();
            })
            .log_err();
        })
        .detach();
    }

    fn load(&mut self, model: ModelArtifact, select_default: bool, cx: &mut Context<Self>) {
        if self.operation.is_some() {
            return;
        }
        self.selected_model = Some(model.id.clone());
        let decoding = self.decoding.next(&model);
        if select_default
            && self
                .runtime
                .as_ref()
                .is_some_and(|runtime| runtime.model_id == model.id && runtime.decoding == decoding)
            && self.endpoint.is_some()
        {
            Self::persist_default(model.id, cx);
            cx.notify();
            return;
        }
        let (Some(store), Some(catalog)) = (self.store.clone(), self.catalog.clone()) else {
            return;
        };
        let http = self.http.clone();
        let options = LoadOptions {
            cpu_only: self.cpu_only,
            decoding,
        };
        let Some(local_http) = self.local_http.clone() else {
            self.error = Some("Direct local inference transport is unavailable. Restart Copper and retry; no proxy fallback was used.".into());
            cx.notify();
            return;
        };
        if let Some(runtime) = self.runtime.take() {
            self.states.insert(runtime.model_id, ModelState::Downloaded);
        }
        self.endpoint = None;
        self.error = None;
        self.cancelled = Arc::new(AtomicBool::new(false));
        let cancelled = self.cancelled.clone();
        self.busy_model = Some(model.id.clone());
        self.decoding.failed_mtp_loads.remove(&model.id);
        self.states.insert(model.id.clone(), ModelState::Loading);
        let id = model.id.clone();
        let executor = cx.background_executor().clone();
        let task = cx.background_spawn(async move {
            store
                .load_model(
                    &catalog,
                    &model,
                    http.as_ref(),
                    local_http.as_ref(),
                    options,
                    cancelled,
                    move |duration| executor.timer(duration).boxed(),
                )
                .await
        });
        self.operation = Some(cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, move |this, cx| {
                this.busy_model = None;
                this.operation = None;
                match result {
                    Ok((runtime, hardware, warning)) => {
                        this.hardware = Some(hardware);
                        this.error = warning;
                        this.endpoint = Some(runtime.endpoint.clone());
                        this.runtime = Some(runtime);
                        let selected = AllLanguageModelSettings::get_global(cx)
                            .llama_cpp
                            .managed_model
                            .as_deref()
                            == Some(id.as_str());
                        this.states.insert(
                            id.clone(),
                            if selected
                                && Self::selected_default(cx).as_deref() == Some(id.as_str())
                            {
                                ModelState::Active
                            } else {
                                ModelState::Ready
                            },
                        );
                        if select_default {
                            Self::persist_default(id, cx);
                        }
                    }
                    Err(error) => {
                        if decoding == DecodingMode::ExperimentalMtp {
                            this.decoding
                                .failed_mtp_loads
                                .insert(id.clone(), select_default);
                        }
                        this.states
                            .insert(id, ModelState::Failed(format!("{error:#}")));
                        this.refresh_hardware(cx);
                    }
                }
                cx.notify();
            })
            .log_err();
        }));
        cx.notify();
    }

    fn retry_standard(&mut self, model: ModelArtifact, cx: &mut Context<Self>) {
        if self.operation.is_none()
            && let Some(select_default) = self.decoding.retry_standard(&model.id)
        {
            self.load(model, select_default, cx);
        }
    }

    fn remove(&mut self, model: ModelArtifact, cx: &mut Context<Self>) {
        if self.operation.is_some() {
            return;
        }
        if self
            .runtime
            .as_ref()
            .is_some_and(|runtime| runtime.model_id == model.id)
        {
            self.runtime = None;
            self.endpoint = None;
        }
        let Some(store) = self.store.clone() else {
            return;
        };
        let id = model.id.clone();
        let task = cx.background_spawn(async move { store.remove(&model) });
        self.busy_model = Some(id.clone());
        self.operation = Some(cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |this, cx| {
                this.operation = None;
                this.busy_model = None;
                match result {
                    Ok(()) => {
                        this.downloaded.remove(&id);
                        this.decoding.failed_mtp_loads.remove(&id);
                        this.states.insert(id, ModelState::NotDownloaded);
                    }
                    Err(error) => {
                        this.states
                            .insert(id, ModelState::Failed(format!("{error:#}")));
                    }
                }
                this.refresh_hardware(cx);
                cx.notify();
            })
            .log_err();
        }));
        cx.notify();
    }
}

impl Render for ManagedModels {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        use llama_cpp::managed::RecommendationLevel;
        let busy = self.operation.is_some();
        let native_default = Self::selected_default(cx);
        let native_default_display = language_model::LanguageModelRegistry::read_global(cx)
            .default_model()
            .map(|configured| {
                format!(
                    "{} / {}",
                    configured.provider.name().0,
                    configured.model.name().0
                )
            })
            .unwrap_or_else(|| "None".to_owned());
        let saved_default = AllLanguageModelSettings::get_global(cx)
            .llama_cpp
            .managed_model
            .clone();
        let loaded = self
            .runtime
            .as_ref()
            .map(|runtime| runtime.model_id.as_str());
        let running_decoding = self.runtime.as_ref().map(|runtime| runtime.decoding);
        let name_for = |id: Option<&str>| {
            self.catalog
                .as_ref()
                .and_then(|catalog| {
                    catalog
                        .models
                        .iter()
                        .find(|model| Some(model.id.as_str()) == id)
                })
                .map(|model| model.name.clone())
                .unwrap_or_else(|| "None".to_owned())
        };
        let suggested = self.catalog.as_ref().and_then(|catalog| {
            self.hardware.as_ref().and_then(|hardware| {
                catalog
                    .models
                    .iter()
                    .filter(|model| {
                        let decoding = self.decoding.next(model);
                        let is_loaded = loaded == Some(model.id.as_str());
                        if is_loaded && running_decoding != Some(decoding) {
                            return false;
                        }
                        hardware
                            .recommendation(
                                model,
                                !self.downloaded.contains(&model.id),
                                is_loaded,
                                self.cpu_only,
                                decoding,
                            )
                            .level
                            == RecommendationLevel::Recommended
                    })
                    .min_by_key(|model| model.estimated_memory_for(self.decoding.next(model)))
                    .map(|model| model.name.clone())
            })
        });
        let backend = if self.cpu_only {
            "CPU fallback"
        } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            "Automatic (Metal, with CPU fallback)"
        } else {
            "Automatic (Vulkan, with CPU fallback)"
        };
        let mut rows = Vec::new();
        if let Some(catalog) = &self.catalog {
            for (index, model) in catalog.models.clone().into_iter().enumerate() {
                let state = self
                    .states
                    .get(&model.id)
                    .cloned()
                    .unwrap_or(ModelState::NotDownloaded);
                let is_loaded = loaded == Some(model.id.as_str());
                let decoding = self.decoding.next(&model);
                let reload_for_decoding = is_loaded && running_decoding != Some(decoding);
                let is_default = saved_default.as_deref() == Some(model.id.as_str());
                let is_selected = self.selected_model.as_deref() == Some(model.id.as_str());
                let downloaded = self.downloaded.contains(&model.id);
                let state_label = if state == ModelState::Downloading {
                    format!(
                        "Downloading {:.2} / {:.2} GiB",
                        gib(self.progress.load(Ordering::Relaxed)),
                        gib(model.bytes)
                    )
                } else if matches!(state, ModelState::Ready | ModelState::Active) {
                    if is_default
                        && native_default.as_deref() == Some(model.id.as_str())
                        && is_loaded
                    {
                        "Active · loaded default".to_owned()
                    } else {
                        "Ready · loaded".to_owned()
                    }
                } else {
                    state.label().to_owned()
                };
                let mut actions = h_flex().gap_1().flex_wrap();
                if !downloaded && matches!(state, ModelState::NotDownloaded | ModelState::Failed(_))
                {
                    let download = model.clone();
                    actions = actions.child(
                        Button::new(
                            ("download-local", index),
                            if matches!(state, ModelState::Failed(_)) {
                                "Resume / retry"
                            } else {
                                "Download"
                            },
                        )
                        .style(ButtonStyle::Filled)
                        .disabled(busy)
                        .on_click(
                            cx.listener(move |this, _, _, cx| this.download(download.clone(), cx)),
                        ),
                    );
                }
                if downloaded {
                    let load = model.clone();
                    actions = actions.child(
                        Button::new(
                            ("load-local", index),
                            if reload_for_decoding {
                                "Reload with selected decoding"
                            } else if is_loaded {
                                "Loaded"
                            } else if matches!(state, ModelState::Failed(_)) {
                                "Retry load"
                            } else {
                                "Load"
                            },
                        )
                        .style(ButtonStyle::Outlined)
                        .disabled(busy || (is_loaded && !reload_for_decoding))
                        .on_click(
                            cx.listener(move |this, _, _, cx| this.load(load.clone(), false, cx)),
                        ),
                    );
                    let select = model.clone();
                    actions = actions.child(
                        Button::new(
                            ("select-local", index),
                            if is_default && native_default.as_deref() == Some(model.id.as_str()) {
                                "Saved default"
                            } else {
                                "Set as default"
                            },
                        )
                        .style(ButtonStyle::Filled)
                        .disabled(
                            busy || (is_default
                                && native_default.as_deref() == Some(model.id.as_str())),
                        )
                        .on_click(
                            cx.listener(move |this, _, _, cx| this.load(select.clone(), true, cx)),
                        ),
                    );
                }
                if downloaded
                    && matches!(state, ModelState::Failed(_))
                    && self.decoding.failed_mtp_loads.contains_key(&model.id)
                {
                    let retry = model.clone();
                    actions = actions.child(
                        Button::new(("retry-standard", index), "Retry with standard decoding")
                            .style(ButtonStyle::Outlined)
                            .disabled(busy)
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.retry_standard(retry.clone(), cx);
                            })),
                    );
                }
                if matches!(state, ModelState::Ready | ModelState::Active) {
                    actions = actions.child(
                        Button::new(("unload-local", index), "Unload")
                            .style(ButtonStyle::Outlined)
                            .disabled(busy)
                            .on_click(cx.listener(|this, _, _, cx| this.stop(cx))),
                    );
                }
                if matches!(
                    state,
                    ModelState::Downloaded
                        | ModelState::Ready
                        | ModelState::Active
                        | ModelState::Failed(_)
                ) {
                    let remove = model.clone();
                    actions = actions.child(
                        Button::new(("remove-local", index), "Remove")
                            .style(ButtonStyle::OutlinedGhost)
                            .disabled(busy)
                            .on_click(
                                cx.listener(move |this, _, _, cx| this.remove(remove.clone(), cx)),
                            ),
                    );
                }
                if self.busy_model.as_deref() == Some(model.id.as_str()) {
                    actions = actions.child(
                        Button::new(("cancel-local", index), "Cancel")
                            .style(ButtonStyle::Outlined)
                            .on_click(cx.listener(|this, _, _, cx| this.cancel(cx))),
                    );
                }
                let recommendation =
                    self.hardware
                        .as_ref()
                        .filter(|_| !reload_for_decoding)
                        .map(|hardware| {
                            hardware.recommendation(
                                &model,
                                !downloaded,
                                is_loaded,
                                self.cpu_only,
                                decoding,
                            )
                        });
                let expanded = self.expanded_models.contains(&model.id);
                let detail_id = model.id.clone();
                let selected_id = model.id.clone();
                let mut row = v_flex()
                    .gap_2()
                    .p_3()
                    .border_1()
                    .rounded_md()
                    .border_color(if is_selected {
                        cx.theme().colors().border_focused
                    } else {
                        cx.theme().colors().border_variant
                    })
                    .child(
                        h_flex()
                            .gap_3()
                            .flex_wrap()
                            .items_start()
                            .child(
                                v_flex()
                                    .flex_1()
                                    .min_w_48()
                                    .child(
                                        Button::new(
                                            ("local-model-name", index),
                                            model.name.clone(),
                                        )
                                        .toggle_state(is_selected)
                                        .selected_style(ButtonStyle::Tinted(ui::TintColor::Accent))
                                        .on_click(
                                            cx.listener(move |this, _, _, cx| {
                                                this.selected_model = Some(selected_id.clone());
                                                cx.notify();
                                            }),
                                        ),
                                    )
                                    .child(
                                        Label::new(format!(
                                            "{} · {:.2} GiB download · ~{:.1} GiB runtime",
                                            model.quantization,
                                            gib(model.bytes),
                                            gib(model.estimated_memory_for(decoding))
                                        ))
                                        .size(LabelSize::Small)
                                        .color(Color::Muted),
                                    )
                                    .when(is_default, |view| {
                                        view.child(
                                            Label::new("Persisted local default")
                                                .size(LabelSize::Small)
                                                .color(Color::Accent),
                                        )
                                    }),
                            )
                            .child(div().w_48().child(actions))
                            .child(
                                v_flex()
                                    .w_32()
                                    .gap_1()
                                    .child(Label::new(state_label))
                                    .child(
                                        Label::new(if model.qualified {
                                            "Native-agent qualification recorded"
                                        } else {
                                            "Native-agent qualification pending"
                                        })
                                        .size(LabelSize::Small)
                                        .color(
                                            if model.qualified {
                                                Color::Muted
                                            } else {
                                                Color::Warning
                                            },
                                        ),
                                    ),
                            ),
                    )
                    .when(model.experimental_mtp_memory_mib.is_some(), |row| {
                        let model = model.clone();
                        row.child(
                            h_flex()
                                .gap_2()
                                .flex_wrap()
                                .child(
                                    Button::new(
                                        ("experimental-mtp", index),
                                        "Experimental MTP · Live inference unverified",
                                    )
                                    .style(ButtonStyle::Outlined)
                                    .toggle_state(decoding == DecodingMode::ExperimentalMtp)
                                    .selected_style(ButtonStyle::Tinted(ui::TintColor::Accent))
                                    .disabled(busy)
                                    .on_click(cx.listener(
                                        move |this, _, _, cx| {
                                            this.decoding.toggle(&model);
                                            cx.notify();
                                        },
                                    )),
                                )
                                .child(
                                    Label::new(format!(
                                        "Next load (this session): {}",
                                        decoding.label()
                                    ))
                                    .size(LabelSize::Small),
                                ),
                        )
                    })
                    .when(reload_for_decoding, |row| {
                        row.child(
                            Label::new(
                                "Copper checks available memory after unloading the current model.",
                            )
                            .size(LabelSize::Small),
                        )
                    })
                    .when_some(recommendation, |row, recommendation| {
                        row.child(
                            Label::new(format!(
                                "{}: {}",
                                recommendation.level.label(),
                                recommendation.reason
                            ))
                            .size(LabelSize::Small)
                            .color(match recommendation.level {
                                RecommendationLevel::Recommended => Color::Muted,
                                RecommendationLevel::LimitedHeadroom => Color::Warning,
                                RecommendationLevel::InsufficientCapacity => Color::Error,
                            }),
                        )
                    })
                    .child(
                        Button::new(
                            ("local-model-details", index),
                            if expanded {
                                "Hide model details"
                            } else {
                                "Model details"
                            },
                        )
                        .toggle_state(expanded)
                        .style(ButtonStyle::OutlinedGhost)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            if !this.expanded_models.remove(&detail_id) {
                                this.expanded_models.insert(detail_id.clone());
                            }
                            cx.notify();
                        })),
                    )
                    .when_some(
                        if let ModelState::Failed(message) = state {
                            Some(message)
                        } else {
                            None
                        },
                        |row, message| row.child(Label::new(message).color(Color::Error)),
                    );
                if expanded {
                    row = row.child(
                        v_flex().gap_1().p_2().bg(cx.theme().colors().element_background)
                            .child(Label::new(format!("{} · {} bytes", model.filename, model.bytes)).size(LabelSize::Small))
                            .child(Label::new(format!(
                                "Context: {} tokens (model limit {}). Standard runtime estimate: {:.2} GiB weights + {:.2} GiB FP16 KV cache + {:.2} GiB runtime overhead. The 1 GiB system reserve is additional.",
                                model.context_tokens, model.training_context, gib(model.bytes), gib(model.kv_cache_bytes()), gib(model.runtime_overhead_bytes())
                            )).size(LabelSize::Small))
                            .when(model.experimental_mtp_memory_mib.is_some(), |details| {
                                details.child(Label::new(format!("Experimental MTP adds an estimated {:.2} GiB of memory and uses the embedded draft head.", gib(model.mtp_memory_bytes()))).size(LabelSize::Small))
                            })
                            .child(Label::new(format!("Publisher: {} · revision {} · license {}", model.repository, model.revision, model.license)).size(LabelSize::Small))
                            .when_some(model.attribution.clone(), |details, attribution| {
                                details.child(Label::new(attribution).size(LabelSize::Small))
                            })
                            .child(Label::new(format!("SHA-256: {}", model.sha256)).size(LabelSize::Small))
                            .child(Label::new(format!("Template: {}. Tool support: {}. A healthy loaded runtime does not qualify native-agent tool reliability.", model.chat_template, model.tool_support)).size(LabelSize::Small))
                            .child(Button::new(("license-local", index), "Publisher and license")
                                .style(ButtonStyle::Outlined)
                                .on_click({ let url = model.license_url.clone(); move |_, _, cx| cx.open_url(&url) })),
                    );
                }
                rows.push(row.into_any_element());
            }
        }
        let runtime_backend = self
            .runtime
            .as_ref()
            .map(|runtime| {
                if runtime.cpu_only {
                    "CPU"
                } else {
                    "Automatic acceleration (runtime chooses devices)"
                }
            })
            .unwrap_or("Not running");
        let summary = v_flex().gap_1().p_3().border_1().rounded_md()
            .border_color(cx.theme().colors().border_variant)
            .child(Label::new(format!("Selected backend preference (this session): {backend}")))
            .child(Label::new(format!("Loaded: {} · Runtime backend: {runtime_backend} · Running decoding: {} · Persisted local default: {}", name_for(loaded), running_decoding.map_or("None", DecodingMode::label), name_for(saved_default.as_deref()))).size(LabelSize::Small))
            .child(Label::new(format!("Native-agent default: {native_default_display} · Suggested starting model: {}", suggested.unwrap_or_else(|| "No model has comfortable estimated capacity; review warnings below".to_owned()))).size(LabelSize::Small))
            .when_some(self.hardware.as_ref(), |view, hardware| view.child(Label::new(hardware.summary()).size(LabelSize::Small)))
            .child(h_flex().gap_2().flex_wrap()
                .child(Button::new("refresh-local-capacity", "Refresh capacity")
                    .style(ButtonStyle::Outlined).disabled(busy)
                    .on_click(cx.listener(|this, _, _, cx| this.refresh_hardware(cx))))
                .child(Button::new("local-cpu-fallback", match (self.cpu_only, loaded.is_some()) {
                    (true, true) => "Reload with acceleration",
                    (false, true) => "Reload on CPU",
                    (true, false) => "Use acceleration",
                    (false, false) => "Use CPU fallback",
                })
                    .style(ButtonStyle::Outlined).disabled(busy)
                    .on_click(cx.listener(|this, _, _, cx| this.toggle_backend(cx))))
                .child(Button::new("local-hardware-details", if self.hardware_details { "Hide hardware details" } else { "Hardware details" })
                    .style(ButtonStyle::OutlinedGhost).toggle_state(self.hardware_details)
                    .on_click(cx.listener(|this, _, _, cx| { this.hardware_details = !this.hardware_details; cx.notify(); }))))
            .when(self.hardware_details, |view| view.child(v_flex().gap_1()
                .children(self.hardware.as_ref().map(|hardware| Label::new(if hardware.unified_memory {
                    "CPU and GPU share one memory pool; their capacities are not added together."
                } else {
                    "RAM and dedicated GPU memory are separate limits; unknown GPU capacity requires a load test."
                }).size(LabelSize::Small)))
                .children(self.hardware.iter().flat_map(|hardware| hardware.device_details()).map(|line| Label::new(line).size(LabelSize::Small)))
                .child(Label::new("Capacity is an estimate using full weights, context/KV cache and runtime overhead. Loading measures runtime compatibility. Active means a loaded native-agent default; conversations may keep their own model selection. The suggested starting model is the smallest with comfortable estimated capacity.").size(LabelSize::Small))));
        v_flex().gap_3().w_full().min_w_0()
            .child(Headline::new("Local models").size(HeadlineSize::Small))
            .child(Label::new("Choose Download to obtain weights, then Load to try a model or Set as default to save it. Inference stays on this computer; there is no cloud fallback.").size(LabelSize::Small).color(Color::Muted))
            .child(summary)
            .child(h_flex().gap_3().flex_wrap()
                .child(div().flex_1().min_w_48().child(Label::new("Name")))
                .child(div().w_48().child(Label::new("Actions")))
                .child(div().w_32().child(Label::new("State"))))
            .children(rows)
            .when_some(self.error.clone(), |view, error| view.child(Label::new(error).color(Color::Error)))
    }
}
