use crate::AllLanguageModelSettings;
use fs::Fs;
use futures::FutureExt as _;
use gpui::{App, AppContext as _, Context, Entity, Global, Render, Task};
use http_client::HttpClient;
use llama_cpp::managed::{
    Catalog, Hardware, ManagedRuntime, ManagedStore, ModelArtifact, ModelState, catalog, gib,
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
        }
    });
    cx.set_global(ManagedModelsGlobal(entity.clone()));
    entity.update(cx, |this, cx| this.initialize(cx));
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
                        let active = Self::selected_default(cx).as_deref() == Some(runtime.model_id.as_str());
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
        if select_default
            && self
                .runtime
                .as_ref()
                .is_some_and(|runtime| runtime.model_id == model.id)
        {
            Self::persist_default(model.id, cx);
            cx.notify();
            return;
        }
        let (Some(store), Some(catalog)) = (self.store.clone(), self.catalog.clone()) else {
            return;
        };
        let http = self.http.clone();
        let cpu_only = self.cpu_only;
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
                    cpu_only,
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
                            if select_default || selected {
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
                        this.states
                            .insert(id, ModelState::Failed(format!("{error:#}")));
                    }
                }
                cx.notify();
            })
            .log_err();
        }));
        cx.notify();
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
        let busy = self.operation.is_some();
        let selected = Self::selected_default(cx);
        let mut rows = Vec::new();
        if let Some(catalog) = &self.catalog {
            for (index, model) in catalog.models.clone().into_iter().enumerate() {
                let state = self
                    .states
                    .get(&model.id)
                    .cloned()
                    .unwrap_or(ModelState::NotDownloaded);
                let state_label = if state == ModelState::Downloading {
                    format!(
                        "Downloading {:.2} / {:.2} GiB",
                        gib(self.progress.load(Ordering::Relaxed)),
                        gib(model.bytes)
                    )
                } else {
                    state.label().to_string()
                };
                let downloaded = self.downloaded.contains(&model.id);
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
                        .disabled(busy)
                        .on_click(
                            cx.listener(move |this, _, _, cx| this.download(download.clone(), cx)),
                        ),
                    );
                }
                if downloaded {
                    if matches!(state, ModelState::Downloaded | ModelState::Failed(_)) {
                        let load = model.clone();
                        actions =
                            actions.child(
                                Button::new(
                                    ("load-local", index),
                                    if matches!(state, ModelState::Failed(_)) {
                                        "Retry load"
                                    } else {
                                        "Load"
                                    },
                                )
                                .disabled(busy)
                                .on_click(cx.listener(
                                    move |this, _, _, cx| this.load(load.clone(), false, cx),
                                )),
                            );
                    }
                    let load = model.clone();
                    actions = actions.child(
                        Button::new(
                            ("default-local", index),
                            if selected.as_deref() == Some(model.id.as_str()) {
                                "Load default"
                            } else {
                                "Set as default"
                            },
                        )
                        .disabled(busy)
                        .on_click(
                            cx.listener(move |this, _, _, cx| this.load(load.clone(), true, cx)),
                        ),
                    );
                }
                if matches!(state, ModelState::Ready | ModelState::Active) {
                    actions = actions.child(
                        Button::new(("unload-local", index), "Unload")
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
                            .disabled(busy)
                            .on_click(
                                cx.listener(move |this, _, _, cx| this.remove(remove.clone(), cx)),
                            ),
                    );
                }
                if self.busy_model.as_deref() == Some(model.id.as_str()) {
                    actions = actions.child(
                        Button::new(("cancel-local", index), "Cancel")
                            .on_click(cx.listener(|this, _, _, cx| this.cancel(cx))),
                    );
                }
                let recommendation = self
                    .hardware
                    .as_ref()
                    .map(|hardware| hardware.recommendation(&model))
                    .unwrap_or_else(|| "Checking hardware and storage…".into());
                rows.push(v_flex().gap_1().py_3().border_b_1().border_color(cx.theme().colors().border_variant)
                    .child(h_flex().gap_3().items_start()
                        .child(v_flex().flex_1().min_w_0().child(Label::new(format!("{}{}",model.name,if selected.as_deref()==Some(model.id.as_str()){" · Default"}else{""})))
                            .child(Label::new(format!("{} · {:.2} GiB download · estimated {:.1} GiB memory",model.quantization,gib(model.bytes),gib(model.estimated_memory_bytes()))).size(LabelSize::Small).color(Color::Muted)))
                        .child(div().w_56().child(actions)).child(div().w_32().child(Label::new(state_label))))
                    .child(Label::new(recommendation).size(LabelSize::Small).color(Color::Muted))
                    .child(Label::new(format!("{} · {} bytes",model.filename,model.bytes)).size(LabelSize::Small).color(Color::Muted))
                    .child(Label::new(format!("{} tokens · {} · native-agent qualification pending",model.context_tokens,model.license)).size(LabelSize::Small).color(Color::Muted))
                    .child(Button::new(("license-local",index),"Publisher, license and artifact details").on_click({let url=model.license_url.clone();move |_,_,cx|cx.open_url(&url)}))
                    .when_some(if let ModelState::Failed(message)=state {Some(message)}else {None},|row,message|row.child(Label::new(message).color(Color::Error)))
                    .into_any_element());
            }
        }
        v_flex().gap_2().w_full()
            .child(Headline::new("Local models").size(HeadlineSize::Small))
            .child(Label::new("Download a model, then choose Set as default. Downloads start only when you choose Download. Inference stays on this computer; failures never switch to cloud.").size(LabelSize::Small).color(Color::Muted))
            .child(Label::new("Memory figures are estimates for the single context listed for each model, including full weights, conservative FP16 KV cache and runtime overhead. They are not measured compatibility results.").size(LabelSize::Small).color(Color::Muted))
            .when_some(self.hardware.as_ref(),|view,hardware|view.child(Label::new(format!("{} · {:.1} GiB estimated available RAM / {:.1} GiB total · {:.1} GiB free disk · {}",hardware.architecture,gib(hardware.available_memory),gib(hardware.total_memory),gib(hardware.free_disk),hardware.backend)).size(LabelSize::Small)))
            .child(Button::new("refresh-local-capacity", "Refresh capacity").disabled(busy).on_click(cx.listener(|this, _, _, cx| this.refresh_hardware(cx))))
            .child(Button::new("local-cpu-fallback",if self.cpu_only{"Use acceleration"}else{"Use CPU fallback"}).disabled(busy).on_click(cx.listener(|this,_,_,cx|this.toggle_backend(cx))))
            .child(h_flex().gap_3().child(div().flex_1().child(Label::new("Name"))).child(div().w_56().child(Label::new("Actions"))).child(div().w_32().child(Label::new("State")))).children(rows)
            .when_some(self.error.clone(),|view,error|view.child(Label::new(error).color(Color::Error)))
    }
}
